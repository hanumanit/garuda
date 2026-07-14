//! OpenAI-compatible HTTP API.
//!
//! Compatible in the ways that make clients work: `created` is a real timestamp,
//! streaming ends with the `data: [DONE]` sentinel the SDKs wait for, `usage` is
//! reported, `finish_reason` says what actually happened, and errors come back in
//! OpenAI's error envelope with the status code clients retry on (429 for rate
//! limits, 503 when the queue is full).
//!
//! The one place it deliberately differs: there is no authentication. Do not put
//! this on a public interface.

use crate::core::GarudaError;
use crate::runtime::{InferenceRuntime, SamplingParams};
use crate::scheduler::{Priority, Scheduler};
use crate::session::{self, Piece};
use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub const MODEL_ID: &str = "garuda-moe-v1";

pub struct ApiState {
    pub runtime: Arc<InferenceRuntime>,
    pub scheduler: Arc<Scheduler>,
    /// Applied when the request does not override them.
    pub defaults: SamplingParams,
    pub request_timeout: Duration,
    pub started: std::time::Instant,
}

pub type SharedState = Arc<ApiState>;

pub fn create_router(state: SharedState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/stats", get(stats))
        .with_state(state)
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct ErrorBody {
    message: String,
    r#type: String,
    code: String,
}

#[derive(Debug, Serialize)]
struct ErrorEnvelope {
    error: ErrorBody,
}

/// Map a domain error onto the status code a client should act on.
fn error_response(e: &GarudaError) -> Response {
    let (status, kind, code) = match e {
        GarudaError::RateLimit => (
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_error",
            "rate_limit_exceeded",
        ),
        GarudaError::Busy => (
            StatusCode::SERVICE_UNAVAILABLE,
            "server_error",
            "queue_full",
        ),
        GarudaError::Timeout => (StatusCode::GATEWAY_TIMEOUT, "server_error", "timeout"),
        GarudaError::Cancelled => (
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "cancelled",
        ),
        GarudaError::Config(_) | GarudaError::InvalidToken(_) | GarudaError::Inference(_) => (
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "invalid_request",
        ),
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            "internal_error",
        ),
    };

    (
        status,
        Json(ErrorEnvelope {
            error: ErrorBody {
                message: e.to_string(),
                r#type: kind.to_owned(),
                code: code.to_owned(),
            },
        }),
    )
        .into_response()
}

fn bad_request(msg: impl Into<String>) -> Response {
    error_response(&GarudaError::Inference(msg.into()))
}

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// The sampling knobs every OpenAI-shaped request may carry.
#[derive(Debug, Default, Deserialize)]
struct SamplingOverrides {
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    max_tokens: Option<usize>,
    seed: Option<u64>,
    /// Garuda extension: `low`, `normal`, `high`.
    priority: Option<String>,
}

impl SamplingOverrides {
    fn apply(&self, base: SamplingParams) -> Result<SamplingParams, GarudaError> {
        let p = SamplingParams {
            temperature: self.temperature.unwrap_or(base.temperature),
            top_p: self.top_p.unwrap_or(base.top_p),
            top_k: self.top_k.unwrap_or(base.top_k),
            max_tokens: self.max_tokens.unwrap_or(base.max_tokens),
            seed: self.seed.or(base.seed),
        };
        p.validate()?;
        Ok(p)
    }

    fn priority(&self) -> Result<Priority, GarudaError> {
        match &self.priority {
            Some(s) => s.parse(),
            None => Ok(Priority::Normal),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(flatten)]
    sampling: SamplingOverrides,
}

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default)]
    pub stream: bool,
    #[serde(flatten)]
    sampling: SamplingOverrides,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize)]
struct ChatChoice {
    index: usize,
    message: ChatMessage,
    finish_reason: String,
}

#[derive(Debug, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

#[derive(Debug, Default, Serialize)]
struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Debug, Serialize)]
struct ChunkChoice {
    index: usize,
    delta: Delta,
    finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct ChatChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChunkChoice>,
}

#[derive(Debug, Serialize)]
struct CompletionChoice {
    text: String,
    index: usize,
    finish_reason: String,
}

#[derive(Debug, Serialize)]
struct CompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<CompletionChoice>,
    usage: Usage,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    #[serde(default)]
    pub model: Option<String>,
    /// A string or an array of strings, as OpenAI allows.
    pub input: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct EmbeddingData {
    object: &'static str,
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct EmbeddingResponse {
    object: &'static str,
    data: Vec<EmbeddingData>,
    model: String,
    usage: Usage,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

async fn models() -> impl IntoResponse {
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
            "created": now(),
            "owned_by": "garuda",
        }],
    }))
}

async fn stats(State(state): State<SharedState>) -> impl IntoResponse {
    let s = state.scheduler.stats();
    let prompt = state.runtime.prompt_cache_stats();

    Json(serde_json::json!({
        "uptime_secs": state.started.elapsed().as_secs(),
        "scheduler": {
            "submitted": s.submitted,
            "completed": s.completed,
            "cancelled": s.cancelled,
            "timed_out": s.timed_out,
            "failed": s.failed,
            "rejected_busy": s.rejected_busy,
            "rejected_rate_limit": s.rejected_rate_limit,
        },
        "prompt_cache": {
            "hits": prompt.hits,
            "misses": prompt.misses,
            "entries": prompt.entries,
            "hit_ratio": prompt.hit_ratio(),
        },
        "context_window": state.runtime.max_context(),
    }))
}

/// Caller identity. There is no auth, so everyone shares one bucket unless they
/// name themselves — which is a scaffold's honest answer, not a security control.
fn user_id(headers: &axum::http::HeaderMap) -> String {
    headers
        .get("x-garuda-user")
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty() && s.len() <= 128)
        .unwrap_or("anonymous")
        .to_owned()
}

async fn chat_completions(
    State(state): State<SharedState>,
    headers: axum::http::HeaderMap,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    if req.messages.is_empty() {
        return bad_request("messages must not be empty");
    }

    let params = match req.sampling.apply(state.defaults) {
        Ok(p) => p,
        Err(e) => return error_response(&e),
    };
    let priority = match req.sampling.priority() {
        Ok(p) => p,
        Err(e) => return error_response(&e),
    };

    let rendered = session::render_chat(
        req.messages
            .iter()
            .map(|m| (m.role.as_str(), m.content.as_str())),
    );
    let prompt = state.runtime.tokenizer.encode(&rendered);
    let prompt_tokens = prompt.len();
    let model = req.model.unwrap_or_else(|| MODEL_ID.to_owned());

    let handle = match session::submit(&state, &user_id(&headers), prompt, params, priority) {
        Ok(h) => h,
        Err(e) => return error_response(&e),
    };
    let id = format!("chatcmpl-{}", handle.id);

    if req.stream {
        stream_chat(state, handle, id, model).into_response()
    } else {
        collect_chat(state, handle, id, model, prompt_tokens).await
    }
}

async fn collect_chat(
    state: SharedState,
    handle: crate::scheduler::Handle,
    id: String,
    model: String,
    prompt_tokens: usize,
) -> Response {
    let reply = match session::collect(&state, handle).await {
        Ok(r) => r,
        Err(e) => return error_response(&e),
    };

    Json(ChatCompletionResponse {
        id,
        object: "chat.completion",
        created: now(),
        model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_owned(),
                content: reply.text,
            },
            finish_reason: reply.reason.as_openai().to_owned(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens: reply.tokens,
            total_tokens: prompt_tokens + reply.tokens,
        },
    })
    .into_response()
}

fn stream_chat(
    state: SharedState,
    handle: crate::scheduler::Handle,
    id: String,
    model: String,
) -> Sse<impl futures_util::Stream<Item = Result<Event, std::convert::Infallible>>> {
    let stream = async_stream::stream! {
        let mut first = true;

        let chunk = |delta: Delta, finish: Option<String>| {
            Event::default().json_data(ChatChunk {
                id: id.clone(),
                object: "chat.completion.chunk",
                created: now(),
                model: model.clone(),
                choices: vec![ChunkChoice { index: 0, delta, finish_reason: finish }],
            })
        };

        // `handle` moves into `session::pieces`; when the client disconnects and axum
        // drops this stream, the handle drops with it — which cancels the request.
        let pieces = session::pieces(state, handle);
        futures_util::pin_mut!(pieces);
        while let Some(p) = pieces.next().await {
            match p {
                Piece::Text(text) => {
                    let delta = Delta {
                        role: first.then(|| "assistant".to_owned()),
                        content: Some(text),
                    };
                    first = false;
                    if let Ok(e) = chunk(delta, None) {
                        yield Ok(e);
                    }
                }
                Piece::Done(reason) => {
                    if let Ok(e) = chunk(Delta::default(), Some(reason.as_openai().to_owned())) {
                        yield Ok(e);
                    }
                }
                Piece::Error(e) => {
                    // The stream has already begun; the status line is long gone, so
                    // the error has to travel as an SSE event.
                    if let Ok(ev) = Event::default().json_data(ErrorEnvelope {
                        error: ErrorBody {
                            message: e.to_string(),
                            r#type: "server_error".to_owned(),
                            code: "stream_error".to_owned(),
                        },
                    }) {
                        yield Ok(ev);
                    }
                    return;
                }
            }
        }

        // The sentinel every OpenAI client waits for.
        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}

async fn completions(
    State(state): State<SharedState>,
    headers: axum::http::HeaderMap,
    Json(req): Json<CompletionRequest>,
) -> Response {
    if req.prompt.is_empty() {
        return bad_request("prompt must not be empty");
    }

    let params = match req.sampling.apply(state.defaults) {
        Ok(p) => p,
        Err(e) => return error_response(&e),
    };
    let priority = match req.sampling.priority() {
        Ok(p) => p,
        Err(e) => return error_response(&e),
    };

    let prompt = state.runtime.tokenizer.encode(&req.prompt);
    let prompt_tokens = prompt.len();
    let model = req.model.unwrap_or_else(|| MODEL_ID.to_owned());

    let handle = match session::submit(&state, &user_id(&headers), prompt, params, priority) {
        Ok(h) => h,
        Err(e) => return error_response(&e),
    };

    let id = format!("cmpl-{}", handle.id);
    if req.stream {
        return stream_chat(state, handle, id, model).into_response();
    }

    let reply = match session::collect(&state, handle).await {
        Ok(r) => r,
        Err(e) => return error_response(&e),
    };

    Json(CompletionResponse {
        id,
        object: "text_completion",
        created: now(),
        model,
        choices: vec![CompletionChoice {
            text: reply.text,
            index: 0,
            finish_reason: reply.reason.as_openai().to_owned(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens: reply.tokens,
            total_tokens: prompt_tokens + reply.tokens,
        },
    })
    .into_response()
}

/// Embeddings from the model's real pooled hidden state.
///
/// These are genuine forward passes, not the constant vector the previous version
/// returned. They are still not *useful*: the weights are untrained, so the vectors
/// carry no semantic structure. The endpoint is here because the shape and the cost
/// are real; treat the values as noise until a trained checkpoint is loaded.
async fn embeddings(
    State(state): State<SharedState>,
    Json(req): Json<EmbeddingRequest>,
) -> Response {
    let inputs: Vec<String> = match &req.input {
        serde_json::Value::String(s) => vec![s.clone()],
        serde_json::Value::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for i in items {
                match i.as_str() {
                    Some(s) => out.push(s.to_owned()),
                    None => return bad_request("input array must contain only strings"),
                }
            }
            out
        }
        _ => return bad_request("input must be a string or an array of strings"),
    };

    if inputs.is_empty() || inputs.iter().all(|s| s.is_empty()) {
        return bad_request("input must not be empty");
    }

    let runtime = state.runtime.clone();
    let model = req.model.unwrap_or_else(|| MODEL_ID.to_owned());

    // Forward passes are CPU-bound; keep them off the async executor.
    let computed = tokio::task::spawn_blocking(move || {
        let mut out = Vec::with_capacity(inputs.len());
        let mut total = 0usize;
        for text in &inputs {
            let tokens = runtime.tokenizer.encode(text);
            total += tokens.len();
            out.push(runtime.embed(&tokens)?);
        }
        Ok::<_, GarudaError>((out, total))
    })
    .await;

    let (vectors, prompt_tokens) = match computed {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => return error_response(&e),
        Err(e) => {
            return error_response(&GarudaError::Inference(format!(
                "embedding task failed: {e}"
            )))
        }
    };

    Json(EmbeddingResponse {
        object: "list",
        data: vectors
            .into_iter()
            .enumerate()
            .map(|(index, embedding)| EmbeddingData {
                object: "embedding",
                index,
                embedding,
            })
            .collect(),
        model,
        usage: Usage {
            prompt_tokens,
            completion_tokens: 0,
            total_tokens: prompt_tokens,
        },
    })
    .into_response()
}

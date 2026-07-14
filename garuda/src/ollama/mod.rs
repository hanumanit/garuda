//! Ollama-compatible API.
//!
//! A thin translation layer over the same scheduler the OpenAI API uses: it speaks
//! Ollama's `/api/generate` and `/api/chat` (newline-delimited JSON streaming, not
//! SSE), plus `/api/tags` and `/api/version`, so tools that target Ollama — Open
//! WebUI and friends — can point at Garuda unchanged. The engine doesn't know or care
//! which protocol asked; only the request/response shapes differ.

use crate::api::{SharedState, MODEL_ID};
use crate::core::GarudaError;
use crate::runtime::SamplingParams;
use crate::scheduler::Priority;
use crate::session::{self, Piece};
use axum::{
    body::Body,
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::json;
use std::convert::Infallible;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub fn create_ollama_router(state: SharedState) -> Router {
    Router::new()
        .route("/api/generate", post(generate))
        .route("/api/chat", post(chat))
        .route("/api/tags", get(tags))
        .route("/api/version", get(version))
        .with_state(state)
}

fn stream_default() -> bool {
    true // Ollama streams by default
}

#[derive(Debug, Default, Deserialize)]
struct Options {
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    /// Max tokens to generate. `-1`/`0` means "use the server default".
    num_predict: Option<i64>,
    seed: Option<u64>,
}

impl Options {
    fn apply(&self, base: SamplingParams) -> Result<SamplingParams, GarudaError> {
        let p = SamplingParams {
            temperature: self.temperature.unwrap_or(base.temperature),
            top_p: self.top_p.unwrap_or(base.top_p),
            top_k: self.top_k.unwrap_or(base.top_k),
            max_tokens: match self.num_predict {
                Some(n) if n > 0 => n as usize,
                _ => base.max_tokens,
            },
            seed: self.seed.or(base.seed),
        };
        p.validate()?;
        Ok(p)
    }
}

#[derive(Debug, Deserialize)]
struct GenerateRequest {
    #[serde(default)]
    model: Option<String>,
    prompt: String,
    #[serde(default)]
    system: Option<String>,
    #[serde(default = "stream_default")]
    stream: bool,
    #[serde(default)]
    options: Options,
}

#[derive(Debug, Deserialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<Message>,
    #[serde(default = "stream_default")]
    stream: bool,
    #[serde(default)]
    options: Options,
}

/// Which JSON shape a streamed/collected reply uses.
#[derive(Clone, Copy)]
enum Kind {
    Generate,
    Chat,
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// RFC 3339 UTC timestamp (Ollama's `created_at`), computed without a date crate.
fn rfc3339(secs: u64) -> String {
    let (days, rem) = (secs / 86_400, secs % 86_400);
    let (h, mi, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    // Days since 1970-01-01 → civil date (Howard Hinnant's algorithm).
    let z = days as i64 + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

fn error(status: StatusCode, msg: impl Into<String>) -> Response {
    (status, Json(json!({ "error": msg.into() }))).into_response()
}

fn map_error(e: &GarudaError) -> Response {
    let status = match e {
        GarudaError::RateLimit => StatusCode::TOO_MANY_REQUESTS,
        GarudaError::Busy => StatusCode::SERVICE_UNAVAILABLE,
        GarudaError::Config(_) | GarudaError::Inference(_) | GarudaError::InvalidToken(_) => {
            StatusCode::BAD_REQUEST
        }
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    error(status, e.to_string())
}

/// One streamed line, before the terminal `done`.
fn chunk(kind: Kind, model: &str, text: &str) -> String {
    let mut v = json!({ "model": model, "created_at": rfc3339(now_secs()), "done": false });
    match kind {
        Kind::Generate => v["response"] = json!(text),
        Kind::Chat => v["message"] = json!({ "role": "assistant", "content": text }),
    }
    format!("{v}\n")
}

/// The terminal line, `done: true`, with timing and token counts.
fn done_line(
    kind: Kind,
    model: &str,
    count: usize,
    dur: std::time::Duration,
    reason: &str,
) -> String {
    let ns = dur.as_nanos() as u64;
    let mut v = json!({
        "model": model,
        "created_at": rfc3339(now_secs()),
        "done": true,
        "done_reason": reason,
        "total_duration": ns,
        "eval_count": count,
        "eval_duration": ns,
    });
    match kind {
        Kind::Generate => v["response"] = json!(""),
        Kind::Chat => v["message"] = json!({ "role": "assistant", "content": "" }),
    }
    format!("{v}\n")
}

fn stop_reason(r: crate::runtime::StopReason) -> &'static str {
    match r {
        crate::runtime::StopReason::Eos => "stop",
        _ => "length",
    }
}

/// Submit a prompt and either stream NDJSON or collect one JSON object.
async fn run(
    state: SharedState,
    kind: Kind,
    model: String,
    tokens: Vec<crate::core::Token>,
    params: SamplingParams,
    stream: bool,
) -> Response {
    let handle = match session::submit(&state, "ollama", tokens, params, Priority::Normal) {
        Ok(h) => h,
        Err(e) => return map_error(&e),
    };

    if stream {
        let started = Instant::now();
        let body = async_stream::stream! {
            let pieces = session::pieces(state, handle);
            futures_util::pin_mut!(pieces);
            let mut count = 0usize;
            while let Some(p) = pieces.next().await {
                match p {
                    Piece::Text(text) => {
                        count += 1;
                        yield Ok::<_, Infallible>(chunk(kind, &model, &text));
                    }
                    Piece::Done(reason) => {
                        yield Ok(done_line(kind, &model, count, started.elapsed(), stop_reason(reason)));
                    }
                    Piece::Error(e) => {
                        yield Ok(format!("{}\n", json!({ "error": e.to_string() })));
                    }
                }
            }
        };
        return (
            [(header::CONTENT_TYPE, "application/x-ndjson")],
            Body::from_stream(body),
        )
            .into_response();
    }

    // Non-streaming: collect everything into a single object.
    let reply = match session::collect(&state, handle).await {
        Ok(r) => r,
        Err(e) => return map_error(&e),
    };
    let mut v = json!({
        "model": model,
        "created_at": rfc3339(now_secs()),
        "done": true,
        "done_reason": stop_reason(reply.reason),
        "eval_count": reply.tokens,
    });
    match kind {
        Kind::Generate => v["response"] = json!(reply.text),
        Kind::Chat => v["message"] = json!({ "role": "assistant", "content": reply.text }),
    }
    Json(v).into_response()
}

async fn generate(State(state): State<SharedState>, Json(req): Json<GenerateRequest>) -> Response {
    if req.prompt.is_empty() {
        return error(StatusCode::BAD_REQUEST, "prompt must not be empty");
    }
    let params = match req.options.apply(state.defaults) {
        Ok(p) => p,
        Err(e) => return map_error(&e),
    };
    let mut text = String::new();
    if let Some(sys) = &req.system {
        text.push_str(sys);
        text.push('\n');
    }
    text.push_str(&req.prompt);
    let tokens = state.runtime.tokenizer.encode(&text);
    let model = req.model.unwrap_or_else(|| MODEL_ID.to_owned());
    run(state, Kind::Generate, model, tokens, params, req.stream).await
}

async fn chat(State(state): State<SharedState>, Json(req): Json<ChatRequest>) -> Response {
    if req.messages.is_empty() {
        return error(StatusCode::BAD_REQUEST, "messages must not be empty");
    }
    let params = match req.options.apply(state.defaults) {
        Ok(p) => p,
        Err(e) => return map_error(&e),
    };
    let prompt = session::render_chat(
        req.messages
            .iter()
            .map(|m| (m.role.as_str(), m.content.as_str())),
    );
    let tokens = state.runtime.tokenizer.encode(&prompt);
    let model = req.model.unwrap_or_else(|| MODEL_ID.to_owned());
    run(state, Kind::Chat, model, tokens, params, req.stream).await
}

async fn tags() -> Response {
    let name = format!("{MODEL_ID}:latest");
    Json(json!({
        "models": [{
            "name": name,
            "model": name,
            "modified_at": rfc3339(now_secs()),
            "size": 0,
            "digest": "",
            "details": {
                "format": "gguf",
                "family": "llama",
                "families": ["llama"],
                "parameter_size": "",
                "quantization_level": ""
            }
        }]
    }))
    .into_response()
}

async fn version() -> Response {
    Json(json!({ "version": env!("CARGO_PKG_VERSION") })).into_response()
}

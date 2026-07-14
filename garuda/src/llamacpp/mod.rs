//! llama.cpp server-compatible `/completion` endpoint.
//!
//! Speaks the shape `llama-server` uses: an `n_predict`-style request, a single JSON
//! object with a `content` field for non-streaming, and SSE frames of
//! `{"content": ..., "stop": false}` for streaming, ending with a `"stop": true`
//! frame carrying the token counts. Clients written against llama.cpp's HTTP server —
//! and the many wrappers that target it — can point at Garuda unchanged. Like the
//! other front ends this is pure translation over the shared [`session`] core.

use crate::api::{SharedState, MODEL_ID};
use crate::core::GarudaError;
use crate::runtime::{SamplingParams, StopReason};
use crate::scheduler::Priority;
use crate::session::{self, Piece};
use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::post,
    Json, Router,
};
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::json;

pub fn create_llamacpp_router(state: SharedState) -> Router {
    Router::new()
        .route("/completion", post(completion))
        .with_state(state)
}

#[derive(Debug, Deserialize)]
struct CompletionRequest {
    #[serde(default)]
    prompt: String,
    /// Max tokens to generate. `-1`/`0` means "use the server default".
    #[serde(default)]
    n_predict: Option<i64>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stream: bool,
}

impl CompletionRequest {
    fn params(&self, base: SamplingParams) -> Result<SamplingParams, GarudaError> {
        let p = SamplingParams {
            temperature: self.temperature.unwrap_or(base.temperature),
            top_p: self.top_p.unwrap_or(base.top_p),
            top_k: self.top_k.unwrap_or(base.top_k),
            max_tokens: match self.n_predict {
                Some(n) if n > 0 => n as usize,
                _ => base.max_tokens,
            },
            seed: self.seed.or(base.seed),
        };
        p.validate()?;
        Ok(p)
    }
}

fn error(status: StatusCode, msg: impl Into<String>) -> Response {
    (
        status,
        Json(json!({
            "error": { "code": status.as_u16(), "message": msg.into(), "type": "server_error" }
        })),
    )
        .into_response()
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

async fn completion(
    State(state): State<SharedState>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    if req.prompt.is_empty() {
        return error(StatusCode::BAD_REQUEST, "prompt must not be empty");
    }
    let params = match req.params(state.defaults) {
        Ok(p) => p,
        Err(e) => return map_error(&e),
    };

    let tokens = state.runtime.tokenizer.encode(&req.prompt);
    let prompt_tokens = tokens.len();

    let handle = match session::submit(&state, "llamacpp", tokens, params, Priority::Normal) {
        Ok(h) => h,
        Err(e) => return map_error(&e),
    };

    if req.stream {
        let stream = async_stream::stream! {
            let mut predicted = 0usize;
            let pieces = session::pieces(state, handle);
            futures_util::pin_mut!(pieces);
            while let Some(p) = pieces.next().await {
                match p {
                    Piece::Text(text) => {
                        predicted += 1;
                        yield Ok::<_, std::convert::Infallible>(
                            Event::default().data(json!({ "content": text, "stop": false }).to_string())
                        );
                    }
                    Piece::Done(reason) => {
                        let eos = matches!(reason, StopReason::Eos);
                        yield Ok(Event::default().data(json!({
                            "content": "",
                            "stop": true,
                            "model": MODEL_ID,
                            "tokens_predicted": predicted,
                            "tokens_evaluated": prompt_tokens,
                            "stopped_eos": eos,
                            "stopped_limit": !eos,
                            "truncated": false,
                        }).to_string()));
                    }
                    Piece::Error(e) => {
                        yield Ok(Event::default().data(json!({
                            "error": { "code": 500, "message": e.to_string(), "type": "server_error" }
                        }).to_string()));
                        return;
                    }
                }
            }
        };
        return Sse::new(stream).keep_alive(KeepAlive::default()).into_response();
    }

    let reply = match session::collect(&state, handle).await {
        Ok(r) => r,
        Err(e) => return map_error(&e),
    };
    let eos = matches!(reply.reason, StopReason::Eos);
    Json(json!({
        "content": reply.text,
        "stop": true,
        "model": MODEL_ID,
        "tokens_predicted": reply.tokens,
        "tokens_evaluated": prompt_tokens,
        "stopped_eos": eos,
        "stopped_limit": !eos,
        "truncated": false,
    }))
    .into_response()
}

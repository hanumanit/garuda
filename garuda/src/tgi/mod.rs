//! Text Generation Inference (TGI) compatible endpoints.
//!
//! Hugging Face's TGI server exposes `/generate` (one JSON object with a
//! `generated_text` field) and `/generate_stream` (SSE `token` events, the last one
//! also carrying `generated_text` and `details`). Speaking that shape lets the many
//! clients and libraries built around TGI — including LangChain's and the HF
//! `InferenceClient` — target Garuda unchanged. As with the other front ends, this is
//! a translation layer over the shared [`session`] core; the engine is untouched.

use crate::api::SharedState;
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

pub fn create_tgi_router(state: SharedState) -> Router {
    Router::new()
        .route("/generate", post(generate))
        .route("/generate_stream", post(generate_stream))
        .with_state(state)
}

#[derive(Debug, Default, Deserialize)]
struct Parameters {
    max_new_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    seed: Option<u64>,
    #[serde(default)]
    details: bool,
}

impl Parameters {
    fn apply(&self, base: SamplingParams) -> Result<SamplingParams, GarudaError> {
        let p = SamplingParams {
            temperature: self.temperature.unwrap_or(base.temperature),
            top_p: self.top_p.unwrap_or(base.top_p),
            top_k: self.top_k.unwrap_or(base.top_k),
            max_tokens: self.max_new_tokens.unwrap_or(base.max_tokens),
            seed: self.seed.or(base.seed),
        };
        p.validate()?;
        Ok(p)
    }
}

#[derive(Debug, Deserialize)]
struct GenerateRequest {
    #[serde(default)]
    inputs: String,
    #[serde(default)]
    parameters: Parameters,
}

/// TGI's `finish_reason` vocabulary.
fn finish_reason(r: StopReason) -> &'static str {
    match r {
        StopReason::Eos => "eos_token",
        _ => "length",
    }
}

fn error(status: StatusCode, msg: impl Into<String>) -> Response {
    // TGI's error envelope.
    (
        status,
        Json(json!({ "error": msg.into(), "error_type": "generation" })),
    )
        .into_response()
}

fn map_error(e: &GarudaError) -> Response {
    let status = match e {
        GarudaError::RateLimit => StatusCode::TOO_MANY_REQUESTS,
        GarudaError::Busy => StatusCode::SERVICE_UNAVAILABLE,
        GarudaError::Config(_) | GarudaError::Inference(_) | GarudaError::InvalidToken(_) => {
            StatusCode::UNPROCESSABLE_ENTITY
        }
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    error(status, e.to_string())
}

async fn generate(State(state): State<SharedState>, Json(req): Json<GenerateRequest>) -> Response {
    if req.inputs.is_empty() {
        return error(StatusCode::UNPROCESSABLE_ENTITY, "inputs must not be empty");
    }
    let params = match req.parameters.apply(state.defaults) {
        Ok(p) => p,
        Err(e) => return map_error(&e),
    };
    let want_details = req.parameters.details;

    let tokens = state.runtime.tokenizer.encode(&req.inputs);
    let handle = match session::submit(&state, "tgi", tokens, params, Priority::Normal) {
        Ok(h) => h,
        Err(e) => return map_error(&e),
    };

    let reply = match session::collect(&state, handle).await {
        Ok(r) => r,
        Err(e) => return map_error(&e),
    };

    let mut body = json!({ "generated_text": reply.text });
    if want_details {
        body["details"] = json!({
            "finish_reason": finish_reason(reply.reason),
            "generated_tokens": reply.tokens,
            "seed": null,
        });
    }
    Json(body).into_response()
}

async fn generate_stream(
    State(state): State<SharedState>,
    Json(req): Json<GenerateRequest>,
) -> Response {
    if req.inputs.is_empty() {
        return error(StatusCode::UNPROCESSABLE_ENTITY, "inputs must not be empty");
    }
    let params = match req.parameters.apply(state.defaults) {
        Ok(p) => p,
        Err(e) => return map_error(&e),
    };

    let tokens = state.runtime.tokenizer.encode(&req.inputs);
    let handle = match session::submit(&state, "tgi", tokens, params, Priority::Normal) {
        Ok(h) => h,
        Err(e) => return map_error(&e),
    };

    let stream = async_stream::stream! {
        // TGI reports the full text only on the terminal event, so accumulate it.
        let mut full = String::new();
        let mut generated = 0usize;
        let pieces = session::pieces(state, handle);
        futures_util::pin_mut!(pieces);
        while let Some(p) = pieces.next().await {
            match p {
                Piece::Text(text) => {
                    full.push_str(&text);
                    generated += 1;
                    yield Ok::<_, std::convert::Infallible>(Event::default().data(json!({
                        "token": { "id": 0, "text": text, "logprob": 0.0, "special": false },
                        "generated_text": null,
                        "details": null,
                    }).to_string()));
                }
                Piece::Done(reason) => {
                    yield Ok(Event::default().data(json!({
                        "token": { "id": 0, "text": "", "logprob": 0.0, "special": true },
                        "generated_text": full,
                        "details": {
                            "finish_reason": finish_reason(reason),
                            "generated_tokens": generated,
                            "seed": null,
                        },
                    }).to_string()));
                }
                Piece::Error(e) => {
                    yield Ok(Event::default().data(json!({
                        "error": e.to_string(), "error_type": "generation"
                    }).to_string()));
                    return;
                }
            }
        }
    };
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

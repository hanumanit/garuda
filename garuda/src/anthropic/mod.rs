//! Anthropic Messages API (`/v1/messages`).
//!
//! Another translation layer over the shared scheduler, speaking Anthropic's wire
//! format: content blocks, and a typed SSE stream (`message_start`,
//! `content_block_delta`, …, `message_stop`) rather than OpenAI's chunk objects. So
//! the Anthropic SDK can talk to Garuda unchanged. As with the other front ends, the
//! engine is untouched — only the shapes differ.

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
use uuid::Uuid;

pub fn create_anthropic_router(state: SharedState) -> Router {
    Router::new()
        .route("/v1/messages", post(messages))
        .with_state(state)
}

#[derive(Debug, Deserialize)]
struct InMessage {
    role: String,
    /// A plain string, or an array of content blocks (`{"type":"text","text":...}`).
    content: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct MessagesRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<InMessage>,
    #[serde(default)]
    max_tokens: Option<usize>,
    /// A system prompt: a string or an array of text blocks.
    #[serde(default)]
    system: Option<serde_json::Value>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
}

/// Flatten a string-or-blocks content value into plain text.
fn text_of(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(blocks) => blocks
            .iter()
            .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"))
            .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

fn error(status: StatusCode, kind: &str, msg: impl Into<String>) -> Response {
    (
        status,
        Json(json!({
            "type": "error",
            "error": { "type": kind, "message": msg.into() }
        })),
    )
        .into_response()
}

fn map_error(e: &GarudaError) -> Response {
    let (status, kind) = match e {
        GarudaError::RateLimit => (StatusCode::TOO_MANY_REQUESTS, "rate_limit_error"),
        GarudaError::Busy => (StatusCode::SERVICE_UNAVAILABLE, "overloaded_error"),
        GarudaError::Config(_) | GarudaError::Inference(_) | GarudaError::InvalidToken(_) => {
            (StatusCode::BAD_REQUEST, "invalid_request_error")
        }
        _ => (StatusCode::INTERNAL_SERVER_ERROR, "api_error"),
    };
    error(status, kind, e.to_string())
}

fn stop_reason(r: StopReason) -> &'static str {
    match r {
        StopReason::Eos => "end_turn",
        _ => "max_tokens",
    }
}

async fn messages(State(state): State<SharedState>, Json(req): Json<MessagesRequest>) -> Response {
    if req.messages.is_empty() {
        return error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "messages must not be empty",
        );
    }

    let params = SamplingParams {
        temperature: req.temperature.unwrap_or(state.defaults.temperature),
        top_p: req.top_p.unwrap_or(state.defaults.top_p),
        top_k: req.top_k.unwrap_or(state.defaults.top_k),
        max_tokens: req.max_tokens.unwrap_or(state.defaults.max_tokens),
        seed: state.defaults.seed,
    };
    if let Err(e) = params.validate() {
        return map_error(&e);
    }

    // Render system + turns into a flat prompt.
    let mut prompt = String::new();
    if let Some(sys) = &req.system {
        let s = text_of(sys);
        if !s.is_empty() {
            prompt.push_str(&s);
            prompt.push_str("\n\n");
        }
    }
    for m in &req.messages {
        prompt.push_str(&m.role);
        prompt.push_str(": ");
        prompt.push_str(&text_of(&m.content));
        prompt.push('\n');
    }
    prompt.push_str("assistant: ");

    let tokens = state.runtime.tokenizer.encode(&prompt);
    let input_tokens = tokens.len();
    let model = req.model.unwrap_or_else(|| MODEL_ID.to_owned());
    let id = format!("msg_{}", Uuid::new_v4().simple());

    let handle = match session::submit(&state, "anthropic", tokens, params, Priority::Normal) {
        Ok(h) => h,
        Err(e) => return map_error(&e),
    };

    if req.stream {
        stream_messages(state, handle, id, model, input_tokens).into_response()
    } else {
        collect_message(state, handle, id, model, input_tokens).await
    }
}

async fn collect_message(
    state: SharedState,
    handle: crate::scheduler::Handle,
    id: String,
    model: String,
    input_tokens: usize,
) -> Response {
    let reply = match session::collect(&state, handle).await {
        Ok(r) => r,
        Err(e) => return map_error(&e),
    };

    Json(json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{ "type": "text", "text": reply.text }],
        "stop_reason": stop_reason(reply.reason),
        "stop_sequence": null,
        "usage": { "input_tokens": input_tokens, "output_tokens": reply.tokens },
    }))
    .into_response()
}

fn stream_messages(
    state: SharedState,
    handle: crate::scheduler::Handle,
    id: String,
    model: String,
    input_tokens: usize,
) -> Sse<impl futures_util::Stream<Item = Result<Event, std::convert::Infallible>>> {
    let stream = async_stream::stream! {
        let mut output_tokens = 0usize;

        // message_start
        yield Ok(Event::default().event("message_start").data(json!({
            "type": "message_start",
            "message": {
                "id": id, "type": "message", "role": "assistant", "model": model,
                "content": [], "stop_reason": null, "stop_sequence": null,
                "usage": { "input_tokens": input_tokens, "output_tokens": 0 }
            }
        }).to_string()));

        // one text content block
        yield Ok(Event::default().event("content_block_start").data(json!({
            "type": "content_block_start", "index": 0,
            "content_block": { "type": "text", "text": "" }
        }).to_string()));

        let mut reason = StopReason::Length;
        let pieces = session::pieces(state, handle);
        futures_util::pin_mut!(pieces);
        while let Some(p) = pieces.next().await {
            match p {
                Piece::Text(text) => {
                    output_tokens += 1;
                    yield Ok(Event::default().event("content_block_delta").data(json!({
                        "type": "content_block_delta", "index": 0,
                        "delta": { "type": "text_delta", "text": text }
                    }).to_string()));
                }
                Piece::Done(r) => { reason = r; }
                Piece::Error(e) => {
                    yield Ok(Event::default().event("error").data(json!({
                        "type": "error",
                        "error": { "type": "api_error", "message": e.to_string() }
                    }).to_string()));
                    return;
                }
            }
        }

        yield Ok(Event::default().event("content_block_stop").data(json!({
            "type": "content_block_stop", "index": 0
        }).to_string()));
        yield Ok(Event::default().event("message_delta").data(json!({
            "type": "message_delta",
            "delta": { "stop_reason": stop_reason(reason), "stop_sequence": null },
            "usage": { "output_tokens": output_tokens }
        }).to_string()));
        yield Ok(Event::default().event("message_stop").data(json!({
            "type": "message_stop"
        }).to_string()));
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}

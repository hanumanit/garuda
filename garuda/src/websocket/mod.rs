//! WebSocket streaming at `/v1/ws`.
//!
//! One request at a time per socket. Cancellation is real: a `{"cancel": true}`
//! message, or the socket closing, drops the scheduler handle, and generation stops
//! at the next token boundary.

use crate::api::SharedState;
use crate::scheduler::{Priority, RequestSpec, StreamEvent};
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
    routing::get,
    Router,
};
use serde::{Deserialize, Serialize};

pub fn create_ws_router(state: SharedState) -> Router {
    Router::new()
        .route("/v1/ws", get(ws_handler))
        .with_state(state)
}

#[derive(Debug, Deserialize)]
struct WsRequest {
    prompt: String,
    #[serde(default)]
    priority: Option<String>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
    /// Cancel whatever is currently generating on this socket.
    #[serde(default)]
    cancel: bool,
}

#[derive(Debug, Serialize)]
struct WsResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
    done: bool,
}

impl WsResponse {
    fn token(text: String) -> Self {
        Self {
            token: Some(text),
            error: None,
            finish_reason: None,
            done: false,
        }
    }
    fn done(reason: &str) -> Self {
        Self {
            token: None,
            error: None,
            finish_reason: Some(reason.to_owned()),
            done: true,
        }
    }
    fn error(msg: String) -> Self {
        Self {
            token: None,
            error: Some(msg),
            finish_reason: None,
            done: true,
        }
    }
}

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<SharedState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

/// Send `resp`; `false` means the socket is gone.
async fn send(socket: &mut WebSocket, resp: &WsResponse) -> bool {
    let Ok(text) = serde_json::to_string(resp) else {
        return false;
    };
    socket.send(Message::Text(text)).await.is_ok()
}

async fn handle_socket(mut socket: WebSocket, state: SharedState) {
    while let Some(Ok(msg)) = socket.recv().await {
        let text = match msg {
            Message::Text(t) => t,
            Message::Close(_) => break,
            // Ping/Pong are handled by axum; binary frames are not a protocol we speak.
            _ => continue,
        };

        let req: WsRequest = match serde_json::from_str(&text) {
            Ok(r) => r,
            Err(e) => {
                if !send(
                    &mut socket,
                    &WsResponse::error(format!("invalid JSON: {e}")),
                )
                .await
                {
                    return;
                }
                continue;
            }
        };

        // A cancel with nothing running is a no-op, not an error.
        if req.cancel {
            continue;
        }

        if !run_one(&mut socket, &state, req).await {
            return;
        }
    }
}

/// Run a single request to completion. Returns `false` if the socket died.
async fn run_one(socket: &mut WebSocket, state: &SharedState, req: WsRequest) -> bool {
    let mut params = state.defaults;
    if let Some(v) = req.temperature {
        params.temperature = v;
    }
    if let Some(v) = req.top_p {
        params.top_p = v;
    }
    if let Some(v) = req.top_k {
        params.top_k = v;
    }
    if let Some(v) = req.max_tokens {
        params.max_tokens = v;
    }
    if req.seed.is_some() {
        params.seed = req.seed;
    }

    if let Err(e) = params.validate() {
        return send(socket, &WsResponse::error(e.to_string())).await;
    }

    let priority: Priority = match req.priority.as_deref().unwrap_or("normal").parse() {
        Ok(p) => p,
        Err(e) => return send(socket, &WsResponse::error(e.to_string())).await,
    };

    if req.prompt.is_empty() {
        return send(
            socket,
            &WsResponse::error("prompt must not be empty".into()),
        )
        .await;
    }

    let handle = state.scheduler.submit(RequestSpec {
        user_id: "ws".to_owned(),
        prompt: state.runtime.tokenizer.encode(&req.prompt),
        params,
        priority,
        timeout: state.request_timeout,
    });

    let mut handle = match handle {
        Ok(h) => h,
        Err(e) => return send(socket, &WsResponse::error(e.to_string())).await,
    };

    let mut decoder = state.runtime.tokenizer.stream_decoder();

    loop {
        tokio::select! {
            // A client that hangs up, closes, or sends `{"cancel": true}` stops the
            // work: `handle` drops at the end of this function, which cancels it.
            incoming = socket.recv() => {
                match incoming {
                    Some(Ok(Message::Text(t))) => {
                        let wants_cancel = serde_json::from_str::<WsRequest>(&t)
                            .map(|r| r.cancel)
                            .unwrap_or(false);
                        if wants_cancel {
                            handle.cancel();
                            let _ = send(socket, &WsResponse::done("cancelled")).await;
                            return true;
                        }
                    }
                    Some(Ok(Message::Close(_))) | None | Some(Err(_)) => return false,
                    _ => {}
                }
            }

            event = handle.events.recv() => {
                match event {
                    Some(StreamEvent::Token(t)) => {
                        let text = decoder.push(t);
                        if text.is_empty() {
                            continue; // an incomplete UTF-8 character; wait for the rest
                        }
                        if !send(socket, &WsResponse::token(text)).await {
                            return false;
                        }
                    }
                    Some(StreamEvent::Done(reason)) => {
                        let tail = decoder.finish();
                        if !tail.is_empty() && !send(socket, &WsResponse::token(tail)).await {
                            return false;
                        }
                        return send(socket, &WsResponse::done(reason.as_openai())).await;
                    }
                    Some(StreamEvent::Error(e)) => {
                        return send(socket, &WsResponse::error(e.to_string())).await;
                    }
                    None => return send(socket, &WsResponse::done("stop")).await,
                }
            }
        }
    }
}

use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    response::IntoResponse,
    routing::get,
    Router, Extension,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;
use crate::api::ApiState;
use crate::scheduler::{Request, Priority};

#[derive(Debug, Deserialize)]
struct WsRequest {
    prompt: String,
    #[serde(default)]
    priority: Option<String>,
}

#[derive(Debug, Serialize)]
struct WsResponse {
    token: Option<String>,
    error: Option<String>,
    done: bool,
}

pub fn create_ws_router() -> Router {
    Router::new().route("/v1/ws", get(ws_handler))
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    Extension(state): Extension<Arc<ApiState>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: Arc<ApiState>) {
    while let Some(Ok(msg)) = socket.recv().await {
        if let Message::Text(text) = msg {
            let req_data: WsRequest = match serde_json::from_str(&text) {
                Ok(d) => d,
                Err(e) => {
                    let err_resp = WsResponse {
                        token: None,
                        error: Some(format!("Invalid JSON: {}", e)),
                        done: true,
                    };
                    let _ = socket.send(Message::Text(serde_json::to_string(&err_resp).unwrap())).await;
                    continue;
                }
            };

            let tokens = match state.runtime.tokenizer.encode(&req_data.prompt) {
                Ok(t) => t,
                Err(e) => {
                    let err_resp = WsResponse {
                        token: None,
                        error: Some(e.to_string()),
                        done: true,
                    };
                    let _ = socket.send(Message::Text(serde_json::to_string(&err_resp).unwrap())).await;
                    continue;
                }
            };

            let priority = match req_data.priority.as_deref() {
                Some("high") => Priority::High,
                Some("low") => Priority::Low,
                _ => Priority::Normal,
            };

            let req_id = Uuid::new_v4();
            let (response_tx, mut response_rx) = mpsc::unbounded_channel();
            let (_cancel_tx, cancel_rx) = oneshot::channel();

            let scheduler_req = Request {
                id: req_id,
                user_id: "ws_user".to_string(),
                tokens,
                priority,
                timeout: std::time::Duration::from_secs(30),
                response_tx,
                cancel_rx,
            };

            if let Err(e) = state.scheduler.submit_request(scheduler_req) {
                let err_resp = WsResponse {
                    token: None,
                    error: Some(e.to_string()),
                    done: true,
                };
                let _ = socket.send(Message::Text(serde_json::to_string(&err_resp).unwrap())).await;
                continue;
            }

            while let Some(res) = response_rx.recv().await {
                match res {
                    Ok(tok) => {
                        let decoded = state.runtime.tokenizer.decode(&[tok]).unwrap_or_default();
                        let resp = WsResponse {
                            token: Some(decoded),
                            error: None,
                            done: false,
                        };
                        if socket.send(Message::Text(serde_json::to_string(&resp).unwrap())).await.is_err() {
                            let _ = _cancel_tx.send(());
                            break;
                        }
                    }
                    Err(e) => {
                        let resp = WsResponse {
                            token: None,
                            error: Some(e.to_string()),
                            done: true,
                        };
                        let _ = socket.send(Message::Text(serde_json::to_string(&resp).unwrap())).await;
                        break;
                    }
                }
            }

            state.scheduler.release_rate_limit("ws_user");

            let final_resp = WsResponse {
                token: None,
                error: None,
                done: true,
            };
            let _ = socket.send(Message::Text(serde_json::to_string(&final_resp).unwrap())).await;
        }
    }
}

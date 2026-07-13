use axum::{
    routing::{get, post},
    Router, Json, Extension,
    response::{sse::{Event, Sse}, IntoResponse},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;
use crate::runtime::InferenceRuntime;
use crate::scheduler::{Scheduler, Request, Priority};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChatChunkChoice {
    pub index: usize,
    pub delta: ChatMessageDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Default)]
pub struct ChatMessageDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
}

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: usize,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: serde_json::Value, // String or Vec<String>
}

#[derive(Debug, Serialize)]
pub struct EmbeddingData {
    pub object: String,
    pub index: usize,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

pub struct ApiState {
    pub runtime: Arc<InferenceRuntime>,
    pub scheduler: Arc<Scheduler>,
}

pub fn create_router(state: Arc<ApiState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/models", get(models))
        .layer(Extension(state))
}

async fn chat_completions(
    Extension(state): Extension<Arc<ApiState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let mut combined_prompt = String::new();
    for msg in &req.messages {
        combined_prompt.push_str(&format!("{}: {}\n", msg.role, msg.content));
    }
    
    let tokens = match state.runtime.tokenizer.encode(&combined_prompt) {
        Ok(t) => t,
        Err(e) => return (axum::http::StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    };

    let req_id = Uuid::new_v4();
    let (response_tx, mut response_rx) = mpsc::unbounded_channel();
    let (_cancel_tx, cancel_rx) = oneshot::channel();

    let scheduler_req = Request {
        id: req_id,
        user_id: "default_user".to_string(),
        tokens,
        priority: Priority::Normal,
        timeout: std::time::Duration::from_secs(30),
        response_tx,
        cancel_rx,
    };

    if let Err(e) = state.scheduler.submit_request(scheduler_req) {
        return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
    }

    let model_name = req.model.clone();

    if req.stream {
        let stream = async_stream::stream! {
            let mut index = 0;
            while let Some(res) = response_rx.recv().await {
                match res {
                    Ok(tok) => {
                        let decoded = state.runtime.tokenizer.decode(&[tok]).unwrap_or_default();
                        let chunk = ChatCompletionChunk {
                            id: req_id.to_string(),
                            object: "chat.completion.chunk".to_string(),
                            created: 1234567890,
                            model: model_name.clone(),
                            choices: vec![ChatChunkChoice {
                                index,
                                delta: ChatMessageDelta {
                                    role: if index == 0 { Some("assistant".to_string()) } else { None },
                                    content: Some(decoded),
                                },
                                finish_reason: None,
                            }],
                        };
                        index += 1;
                        yield Ok::<_, axum::BoxError>(Event::default().json_data(&chunk).unwrap());
                    }
                    Err(e) => {
                        yield Err::<_, axum::BoxError>(Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())));
                    }
                }
            }
            state.scheduler.release_rate_limit("default_user");
            
            // Send final stop chunk
            let final_chunk = ChatCompletionChunk {
                id: req_id.to_string(),
                object: "chat.completion.chunk".to_string(),
                created: 1234567890,
                model: model_name.clone(),
                choices: vec![ChatChunkChoice {
                    index,
                    delta: ChatMessageDelta::default(),
                    finish_reason: Some("stop".to_string()),
                }],
            };
            yield Ok::<_, axum::BoxError>(Event::default().json_data(&final_chunk).unwrap());
        };

        Sse::new(stream).into_response()
    } else {
        // Collect all tokens
        let mut out_tokens = Vec::new();
        while let Some(res) = response_rx.recv().await {
            match res {
                Ok(tok) => out_tokens.push(tok),
                Err(e) => {
                    state.scheduler.release_rate_limit("default_user");
                    return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
                }
            }
        }
        state.scheduler.release_rate_limit("default_user");

        let content = state.runtime.tokenizer.decode(&out_tokens).unwrap_or_default();
        let response = ChatCompletionResponse {
            id: req_id.to_string(),
            object: "chat.completion".to_string(),
            created: 1234567890,
            model: req.model,
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content,
                },
                finish_reason: Some("stop".to_string()),
            }],
        };
        Json(response).into_response()
    }
}

async fn completions(
    Extension(state): Extension<Arc<ApiState>>,
    Json(req): Json<CompletionRequest>,
) -> impl IntoResponse {
    let tokens = match state.runtime.tokenizer.encode(&req.prompt) {
        Ok(t) => t,
        Err(e) => return (axum::http::StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    };

    let req_id = Uuid::new_v4();
    let (response_tx, mut response_rx) = mpsc::unbounded_channel();
    let (_cancel_tx, cancel_rx) = oneshot::channel();

    let scheduler_req = Request {
        id: req_id,
        user_id: "default_user".to_string(),
        tokens,
        priority: Priority::Normal,
        timeout: std::time::Duration::from_secs(30),
        response_tx,
        cancel_rx,
    };

    if let Err(e) = state.scheduler.submit_request(scheduler_req) {
        return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
    }

    let mut out_tokens = Vec::new();
    while let Some(res) = response_rx.recv().await {
        match res {
            Ok(tok) => out_tokens.push(tok),
            Err(e) => {
                state.scheduler.release_rate_limit("default_user");
                return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
            }
        }
    }
    state.scheduler.release_rate_limit("default_user");

    let text = state.runtime.tokenizer.decode(&out_tokens).unwrap_or_default();
    let response = CompletionResponse {
        id: req_id.to_string(),
        object: "text_completion".to_string(),
        created: 1234567890,
        model: req.model,
        choices: vec![CompletionChoice {
            text,
            index: 0,
            finish_reason: Some("stop".to_string()),
        }],
    };
    Json(response).into_response()
}

async fn embeddings(
    Extension(_state): Extension<Arc<ApiState>>,
    Json(req): Json<EmbeddingRequest>,
) -> impl IntoResponse {
    // Return dummy 128-dimensional embeddings for demonstration
    let response = EmbeddingResponse {
        object: "list".to_string(),
        data: vec![EmbeddingData {
            object: "embedding".to_string(),
            index: 0,
            embedding: vec![0.1; 128],
        }],
        model: req.model,
    };
    Json(response)
}

async fn models() -> impl IntoResponse {
    let response = ModelsResponse {
        object: "list".to_string(),
        data: vec![
            ModelInfo {
                id: "garuda-moe-v1".to_string(),
                object: "model".to_string(),
                created: 1718000000,
                owned_by: "garuda".to_string(),
            },
            ModelInfo {
                id: "garuda-dense-v1".to_string(),
                object: "model".to_string(),
                created: 1718000000,
                owned_by: "garuda".to_string(),
            },
        ],
    };
    Json(response)
}

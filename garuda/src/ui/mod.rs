//! A built-in chat page at `/`.
//!
//! It is a static, dependency-free page that talks to `/v1/chat/completions` —
//! the same endpoint any OpenAI-compatible client uses — so there is no separate
//! frontend to build, deploy, or keep in sync with the API. Same-origin, so it
//! works regardless of the `cors` setting.

use axum::{Router, response::Html, routing::get};

const CHAT_HTML: &str = include_str!("chat.html");

pub fn create_ui_router() -> Router {
    Router::new().route("/", get(chat_ui))
}

async fn chat_ui() -> Html<&'static str> {
    Html(CHAT_HTML)
}

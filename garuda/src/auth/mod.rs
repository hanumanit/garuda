//! `Authorization: Bearer` / `x-api-key` authentication.
//!
//! Off by default (`server.api_keys` empty) — do not expose the port to a network
//! you do not control unless keys are set. `GET /health` and `GET /` (the built-in
//! chat page's static HTML) are always reachable without a key, so a load balancer
//! can probe liveness and a browser can at least load the page before it has one to
//! offer; every other route requires one.
//!
//! Both header styles are accepted so every wire protocol garuda speaks works with
//! its ecosystem's own convention: OpenAI/llama.cpp/Ollama clients send
//! `Authorization: Bearer <key>`, Anthropic clients send `x-api-key: <key>`.

use crate::api::{ErrorBody, ErrorEnvelope};
use axum::extract::{Request, State};
use axum::http::{StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use std::sync::Arc;

/// The configured keys, cheap to clone (shares one `Vec` via `Arc`) so it can be
/// handed to axum as middleware state.
#[derive(Clone)]
pub struct ApiKeys(Arc<Vec<String>>);

impl ApiKeys {
    pub fn new(keys: Vec<String>) -> Self {
        Self(Arc::new(keys))
    }

    /// False when no keys are configured — authentication is off entirely.
    pub fn is_enabled(&self) -> bool {
        !self.0.is_empty()
    }

    /// Constant-time against every configured key, so response timing cannot leak
    /// how many characters of a guess were right.
    fn accepts(&self, presented: &str) -> bool {
        self.0
            .iter()
            .any(|k| constant_time_eq::constant_time_eq(k.as_bytes(), presented.as_bytes()))
    }
}

const EXEMPT_PATHS: [&str; 2] = ["/health", "/"];

fn presented_key(request: &Request) -> Option<&str> {
    let headers = request.headers();
    headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .or_else(|| headers.get("x-api-key").and_then(|v| v.to_str().ok()))
        .filter(|k| !k.is_empty())
}

pub async fn require_key(State(keys): State<ApiKeys>, request: Request, next: Next) -> Response {
    if !keys.is_enabled() || EXEMPT_PATHS.contains(&request.uri().path()) {
        return next.run(request).await;
    }

    match presented_key(&request) {
        Some(key) if keys.accepts(key) => next.run(request).await,
        _ => unauthorized(),
    }
}

fn unauthorized() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        Json(ErrorEnvelope {
            error: ErrorBody {
                message: "invalid or missing API key".into(),
                r#type: "authentication_error".into(),
                code: "unauthorized".into(),
            },
        }),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request as HttpRequest;
    use axum::routing::get;
    use axum::Router;
    use tower::ServiceExt;

    fn app(keys: Vec<&str>) -> Router {
        let keys = ApiKeys::new(keys.into_iter().map(str::to_owned).collect());
        Router::new()
            .route("/health", get(|| async { "ok" }))
            .route("/", get(|| async { "chat page" }))
            .route("/v1/models", get(|| async { "models" }))
            .layer(axum::middleware::from_fn_with_state(
                keys,
                require_key,
            ))
    }

    async fn status(app: &Router, path: &str, header: Option<(&str, &str)>) -> StatusCode {
        let mut req = HttpRequest::builder().uri(path);
        if let Some((name, value)) = header {
            req = req.header(name, value);
        }
        app.clone()
            .oneshot(req.body(Body::empty()).unwrap())
            .await
            .unwrap()
            .status()
    }

    #[tokio::test]
    async fn disabled_when_no_keys_are_configured() {
        let app = app(vec![]);
        assert_eq!(status(&app, "/v1/models", None).await, StatusCode::OK);
    }

    #[tokio::test]
    async fn health_and_root_are_always_exempt() {
        let app = app(vec!["secret"]);
        assert_eq!(status(&app, "/health", None).await, StatusCode::OK);
        assert_eq!(status(&app, "/", None).await, StatusCode::OK);
    }

    #[tokio::test]
    async fn a_protected_route_rejects_a_missing_or_wrong_key() {
        let app = app(vec!["secret"]);
        assert_eq!(
            status(&app, "/v1/models", None).await,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            status(&app, "/v1/models", Some(("authorization", "Bearer wrong"))).await,
            StatusCode::UNAUTHORIZED
        );
    }

    #[tokio::test]
    async fn bearer_and_x_api_key_both_work() {
        let app = app(vec!["secret"]);
        assert_eq!(
            status(&app, "/v1/models", Some(("authorization", "Bearer secret"))).await,
            StatusCode::OK
        );
        assert_eq!(
            status(&app, "/v1/models", Some(("x-api-key", "secret"))).await,
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn any_configured_key_is_accepted() {
        let app = app(vec!["one", "two"]);
        assert_eq!(
            status(&app, "/v1/models", Some(("x-api-key", "two"))).await,
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn an_empty_bearer_token_is_rejected_not_matched_against_an_empty_key() {
        // Guards the invariant `ApiKeys` construction itself can't: even if a caller
        // built one with an empty-string key, a present-but-empty header must not
        // authenticate.
        let app = app(vec![""]);
        assert_eq!(
            status(&app, "/v1/models", Some(("authorization", "Bearer "))).await,
            StatusCode::UNAUTHORIZED
        );
    }
}

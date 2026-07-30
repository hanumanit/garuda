//! End-to-end tests through the real HTTP router.
//!
//! Several of these exist specifically to pin bugs the previous version shipped:
//! the rate-limit slot that leaked on client disconnect (a trivial remote DoS), the
//! missing `[DONE]` sentinel, the hardcoded `created` timestamp, and generation
//! that echoed the prompt back instead of decoding.

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use garuda::anthropic::create_anthropic_router;
use garuda::api::{ApiState, create_router};
use garuda::config::AppConfig;
use garuda::llamacpp::create_llamacpp_router;
use garuda::ollama::create_ollama_router;
use garuda::scheduler::Scheduler;
use garuda::server::Engine;
use garuda::tgi::create_tgi_router;
use http_body_util::BodyExt;
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

struct Harness {
    app: Router,
    state: Arc<ApiState>,
    _dir: TempDir,
}

/// Removes its directory on drop, so a failing test does not leave state behind.
struct TempDir(std::path::PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn harness(tag: &str, tune: impl FnOnce(&mut AppConfig)) -> Harness {
    let dir = std::env::temp_dir().join(format!("garuda_it_{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let mut config = AppConfig::default();
    config.model.path = dir.clone();
    config.model.context = 512;
    config.sampling.max_tokens = 16;
    config.memory.expert_cache = "64MB".into();
    tune(&mut config);
    config.validate().unwrap();

    let engine = Engine::build(&config).unwrap();
    let scheduler = Scheduler::new(engine.runtime.clone(), config.scheduler());

    let state = Arc::new(ApiState {
        runtime: engine.runtime.clone(),
        scheduler,
        model_kind: "synthetic",
        embedding_slots: Arc::new(tokio::sync::Semaphore::new(config.server.max_concurrent)),
        defaults: config.sampling().unwrap(),
        request_timeout: config.request_timeout(),
        started: std::time::Instant::now(),
        chat: engine.chat,
    });

    // Every protocol front end, not just the OpenAI-shaped one: the scheduler
    // policies they share are only meaningfully tested through all of them.
    let app = create_router(state.clone())
        .merge(create_ollama_router(state.clone()))
        .merge(create_anthropic_router(state.clone()))
        .merge(create_llamacpp_router(state.clone()))
        .merge(create_tgi_router(state.clone()));

    Harness {
        app,
        state,
        _dir: TempDir(dir),
    }
}

impl Harness {
    async fn post(&self, path: &str, body: serde_json::Value) -> (StatusCode, String) {
        self.post_as(path, body, None).await
    }

    async fn post_as(
        &self,
        path: &str,
        body: serde_json::Value,
        user: Option<&str>,
    ) -> (StatusCode, String) {
        let mut req = Request::builder()
            .method("POST")
            .uri(path)
            .header("content-type", "application/json");
        if let Some(u) = user {
            req = req.header("x-garuda-user", u);
        }
        let req = req.body(Body::from(body.to_string())).unwrap();

        let res = self.app.clone().oneshot(req).await.unwrap();
        let status = res.status();
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        (status, String::from_utf8_lossy(&bytes).into_owned())
    }

    async fn get(&self, path: &str) -> (StatusCode, String) {
        let req = Request::builder()
            .method("GET")
            .uri(path)
            .body(Body::empty())
            .unwrap();
        let res = self.app.clone().oneshot(req).await.unwrap();
        let status = res.status();
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        (status, String::from_utf8_lossy(&bytes).into_owned())
    }

    fn json(body: &str) -> serde_json::Value {
        serde_json::from_str(body).unwrap_or_else(|e| panic!("not JSON: {e}\nbody: {body}"))
    }
}

fn chat(content: &str) -> serde_json::Value {
    serde_json::json!({
        "model": "garuda-moe-v1",
        "messages": [{ "role": "user", "content": content }],
        "temperature": 0.0,
        "max_tokens": 8,
    })
}

// ---------------------------------------------------------------------------

#[tokio::test]
async fn health_and_models_answer() {
    let h = harness("health", |_| {});

    let (status, body) = h.get("/health").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(Harness::json(&body)["status"], "ok");

    let (status, body) = h.get("/v1/models").await;
    assert_eq!(status, StatusCode::OK);
    let v = Harness::json(&body);
    assert_eq!(v["data"][0]["id"], "garuda-moe-v1");
    assert!(
        v["data"][0]["created"].as_u64().unwrap() > 1_700_000_000,
        "created must be a real timestamp"
    );
}

#[tokio::test]
async fn chat_completion_returns_a_well_formed_openai_response() {
    let h = harness("chat", |_| {});

    let (status, body) = h
        .post("/v1/chat/completions", chat("Explain Mixture of Experts."))
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let v = Harness::json(&body);
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    assert!(v["choices"][0]["message"]["content"].is_string());
    assert!(
        !v["choices"][0]["finish_reason"]
            .as_str()
            .unwrap()
            .is_empty(),
        "finish_reason must say what happened"
    );

    // The old code hardcoded `created: 1234567890` and had no usage field at all.
    assert!(
        v["created"].as_u64().unwrap() > 1_700_000_000,
        "created was {}",
        v["created"]
    );
    let usage = &v["usage"];
    assert!(usage["prompt_tokens"].as_u64().unwrap() > 0);
    assert_eq!(
        usage["total_tokens"].as_u64().unwrap(),
        usage["prompt_tokens"].as_u64().unwrap() + usage["completion_tokens"].as_u64().unwrap()
    );
}

#[tokio::test]
async fn completion_output_is_not_the_prompt_echoed_back() {
    // The previous scheduler emitted `(token + 1)` for each prompt token, so the
    // reply was always the prompt, shifted, and always exactly as long as it.
    let h = harness("noecho", |_| {});

    let prompt = "Explain Mixture of Experts.";
    let (status, body) = h
        .post(
            "/v1/completions",
            serde_json::json!({
                "prompt": prompt,
                "temperature": 0.0,
                "max_tokens": 8,
            }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let v = Harness::json(&body);
    let text = v["choices"][0]["text"].as_str().unwrap();
    let completion_tokens = v["usage"]["completion_tokens"].as_u64().unwrap();

    assert!(completion_tokens <= 8, "max_tokens was ignored");
    assert_ne!(text, prompt, "the model echoed the prompt");
    assert!(
        completion_tokens != v["usage"]["prompt_tokens"].as_u64().unwrap()
            || completion_tokens == 8,
        "output length tracked the prompt length"
    );
}

#[tokio::test]
async fn streaming_sends_chunks_and_the_done_sentinel() {
    let h = harness("stream", |_| {});

    let mut req = chat("hello");
    req["stream"] = serde_json::Value::Bool(true);
    let (status, body) = h.post("/v1/chat/completions", req).await;
    assert_eq!(status, StatusCode::OK);

    let lines: Vec<&str> = body.lines().filter(|l| l.starts_with("data: ")).collect();
    assert!(lines.len() >= 2, "expected several chunks, got: {body}");

    // Every OpenAI client waits for this. The old implementation never sent it and
    // well-behaved SDKs hung until their own timeout fired.
    assert_eq!(
        lines.last().unwrap().trim(),
        "data: [DONE]",
        "stream did not terminate with the [DONE] sentinel"
    );

    let first = Harness::json(lines[0].trim_start_matches("data: "));
    assert_eq!(first["object"], "chat.completion.chunk");
    assert_eq!(first["choices"][0]["delta"]["role"], "assistant");

    // The final chunk before [DONE] carries the finish reason.
    let last_chunk = Harness::json(lines[lines.len() - 2].trim_start_matches("data: "));
    assert!(
        last_chunk["choices"][0]["finish_reason"].is_string(),
        "no finish_reason before [DONE]: {last_chunk}"
    );
}

#[tokio::test]
async fn disconnecting_clients_do_not_permanently_lock_out_a_user() {
    // THE regression test. Previously: user_id was hardcoded for every HTTP caller,
    // and the concurrency slot was only released on a success path a disconnected
    // client never reached. Ten disconnects bricked the whole API with
    // `500 Rate limit exceeded`, permanently, for everyone.
    let h = harness("dos", |c| {
        c.server.max_concurrent_per_user = 2;
        c.sampling.max_tokens = 512; // long enough that we abandon it mid-flight
    });

    for _ in 0..20 {
        let mut req = chat("a prompt that will generate for a while");
        req["stream"] = serde_json::Value::Bool(true);
        req["max_tokens"] = serde_json::json!(512);

        let http = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header("x-garuda-user", "victim")
            .body(Body::from(req.to_string()))
            .unwrap();

        // Take the response and drop it without reading the body: a client hanging up.
        let res = h.app.clone().oneshot(http).await.unwrap();
        drop(res);
        tokio::task::yield_now().await;
    }

    // The user must recover.
    for attempt in 0..100 {
        let (status, body) = h
            .post_as("/v1/chat/completions", chat("hi"), Some("victim"))
            .await;
        if status == StatusCode::OK {
            return;
        }
        assert_eq!(
            status,
            StatusCode::TOO_MANY_REQUESTS,
            "unexpected status on attempt {attempt}: {body}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("user is still locked out after disconnecting repeatedly");
}

#[tokio::test]
async fn one_users_load_does_not_rate_limit_another() {
    // The old code hardcoded `user_id = "default_user"` for every HTTP caller, so
    // one user's traffic consumed everyone's budget.
    let h = harness("isolation", |c| {
        c.server.max_concurrent_per_user = 1;
        c.server.max_concurrent = 4;
    });

    let (a, _) = h
        .post_as("/v1/chat/completions", chat("hello"), Some("alice"))
        .await;
    let (b, _) = h
        .post_as("/v1/chat/completions", chat("hello"), Some("bob"))
        .await;

    assert_eq!(a, StatusCode::OK);
    assert_eq!(b, StatusCode::OK, "bob was rate-limited by alice's traffic");
}

#[tokio::test]
async fn every_protocol_honours_the_caller_identity_header() {
    // Each adapter used to hardcode its own protocol name as the user id, so every
    // Ollama caller shared one bucket, every Anthropic caller another, and
    // `X-Garuda-User` did nothing outside the OpenAI routes. With a per-user limit
    // of 1, two *different* users must both be served on every protocol.
    let h = harness("protocol_users", |c| {
        c.server.max_concurrent_per_user = 1;
        c.server.max_concurrent = 4;
    });

    let cases: [(&str, serde_json::Value); 4] = [
        (
            "/api/generate",
            serde_json::json!({ "prompt": "hello", "stream": false, "options": { "num_predict": 4 } }),
        ),
        (
            "/v1/messages",
            serde_json::json!({ "messages": [{ "role": "user", "content": "hello" }], "max_tokens": 4 }),
        ),
        (
            "/completion",
            serde_json::json!({ "prompt": "hello", "n_predict": 4 }),
        ),
        (
            "/generate",
            serde_json::json!({ "inputs": "hello", "parameters": { "max_new_tokens": 4 } }),
        ),
    ];

    for (path, body) in cases {
        let (a, ba) = h.post_as(path, body.clone(), Some("alice")).await;
        let (b, bb) = h.post_as(path, body, Some("bob")).await;
        assert_eq!(a, StatusCode::OK, "{path} rejected alice: {ba}");
        assert_eq!(
            b,
            StatusCode::OK,
            "{path} rate-limited bob behind alice: {bb}"
        );
    }
}

#[tokio::test]
async fn a_streamed_reply_reports_the_token_count_the_engine_measured() {
    // Adapters used to count SSE/NDJSON frames. The streaming decoder holds back
    // partial UTF-8, so frames and tokens do not correspond and the count came out
    // low. Compare against the non-streaming path, which reports the real number.
    // Sampled rather than greedy, so the byte-level output actually contains
    // multi-byte lead bytes and special ids — the tokens that produce no text frame
    // of their own, and which frame-counting therefore lost.
    let h = harness("stream_counts", |_| {});
    let body = serde_json::json!({
        "prompt": "count the tokens honestly",
        "n_predict": 32,
        "temperature": 1.2,
        "top_k": 0,
        "seed": 0,
    });

    let (status, plain) = h.post("/completion", body.clone()).await;
    assert_eq!(status, StatusCode::OK, "{plain}");
    let expected = Harness::json(&plain)["tokens_predicted"].as_u64().unwrap();
    assert!(expected > 0, "nothing was generated");

    let mut streamed = body;
    streamed["stream"] = serde_json::json!(true);
    let (status, sse) = h.post("/completion", streamed).await;
    assert_eq!(status, StatusCode::OK);

    let frames: Vec<serde_json::Value> = sse
        .lines()
        .filter(|l| l.starts_with("data: "))
        .map(|l| Harness::json(l.trim_start_matches("data: ")))
        .collect();
    let last = frames
        .iter()
        .find(|v| v["stop"] == serde_json::json!(true))
        .expect("no terminal frame");
    let text_frames = frames
        .iter()
        .filter(|v| v["stop"] == serde_json::json!(false))
        .count() as u64;

    assert_eq!(
        last["tokens_predicted"].as_u64().unwrap(),
        expected,
        "the streamed count disagrees with the engine's own"
    );
    // Guards the test itself: if frames and tokens happened to agree here, the
    // assertion above would hold even with the old frame-counting code, and this
    // would be pinning nothing.
    assert_ne!(
        text_frames, expected,
        "fixture no longer exercises the frame/token mismatch; pick another prompt"
    );
}

#[tokio::test]
async fn an_oversized_embeddings_batch_is_refused_rather_than_run() {
    // One request is one blocking task running its inputs serially, and nothing
    // downstream bounds how many there are. Without a cap, the body-size limit is
    // the only thing standing between a caller and thousands of forward passes.
    let h = harness("embed_cap", |_| {});

    let many: Vec<String> = (0..1024).map(|i| format!("input {i}")).collect();
    let (status, body) = h
        .post("/v1/embeddings", serde_json::json!({ "input": many }))
        .await;

    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let message = Harness::json(&body)["error"]["message"]
        .as_str()
        .unwrap()
        .to_owned();
    assert!(
        message.contains("1024") && message.contains("per request"),
        "unhelpful error: {message}"
    );

    // A batch inside the cap still works.
    let few: Vec<String> = (0..4).map(|i| format!("input {i}")).collect();
    let (status, body) = h
        .post("/v1/embeddings", serde_json::json!({ "input": few }))
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(Harness::json(&body)["data"].as_array().unwrap().len(), 4);
}

#[tokio::test]
async fn invalid_parameters_are_rejected_with_400_and_an_error_envelope() {
    let h = harness("badparams", |_| {});

    for bad in [
        serde_json::json!({ "messages": [{"role":"user","content":"x"}], "temperature": -1.0 }),
        serde_json::json!({ "messages": [{"role":"user","content":"x"}], "top_p": 5.0 }),
        serde_json::json!({ "messages": [{"role":"user","content":"x"}], "max_tokens": 0 }),
        serde_json::json!({ "messages": [] }),
        serde_json::json!({ "messages": [{"role":"user","content":"x"}], "priority": "urgent" }),
    ] {
        let (status, body) = h.post("/v1/chat/completions", bad.clone()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "accepted {bad}: {body}");

        let v = Harness::json(&body);
        assert!(
            v["error"]["message"].is_string(),
            "not an OpenAI error envelope: {body}"
        );
    }
}

#[tokio::test]
async fn embeddings_are_real_vectors_that_depend_on_the_input() {
    // The old endpoint returned `vec![0.1; 128]` for every input, always.
    let h = harness("embed", |_| {});

    let (status, body) = h
        .post(
            "/v1/embeddings",
            serde_json::json!({ "input": ["the first sentence", "an entirely different one"] }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let v = Harness::json(&body);
    let data = v["data"].as_array().unwrap();
    assert_eq!(data.len(), 2);

    let vec_of = |i: usize| -> Vec<f64> {
        data[i]["embedding"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect()
    };
    let (a, b) = (vec_of(0), vec_of(1));

    assert!(!a.is_empty());
    assert_ne!(a, b, "every input produced the same vector");
    assert!(
        a.iter().any(|x| (x - 0.1).abs() > 1e-6),
        "still returning the constant placeholder vector"
    );

    // The runtime L2-normalises what it returns.
    let norm: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!((norm - 1.0).abs() < 1e-3, "not normalised: {norm}");

    assert!(v["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn embeddings_reject_a_malformed_input_field() {
    let h = harness("embed_bad", |_| {});
    for bad in [
        serde_json::json!({ "input": 42 }),
        serde_json::json!({ "input": [1, 2, 3] }),
        serde_json::json!({ "input": "" }),
    ] {
        let (status, _) = h.post("/v1/embeddings", bad.clone()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "accepted {bad}");
    }
}

#[tokio::test]
async fn saturated_embedding_capacity_answers_503_without_queuing_cpu_work() {
    let h = harness("embed_busy", |c| c.server.max_concurrent = 1);
    let permit = h.state.embedding_slots.clone().try_acquire_owned().unwrap();

    let (status, body) = h
        .post(
            "/v1/embeddings",
            serde_json::json!({ "input": "would otherwise start a blocking task" }),
        )
        .await;

    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    drop(permit);
}

#[tokio::test]
async fn a_saturated_server_answers_503_rather_than_queueing_without_limit() {
    let h = harness("busy", |c| {
        c.server.max_concurrent = 1;
        c.server.queue_capacity = 1;
        c.server.max_concurrent_per_user = 1000;
    });

    // The requests have to be genuinely in flight together: one at a time would
    // never fill the queue no matter how many we sent.
    let mut inflight = Vec::new();
    for i in 0..64 {
        let mut req = chat("a long prompt that keeps the engine busy for a while");
        req["max_tokens"] = serde_json::json!(256);

        let http = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header("x-garuda-user", format!("user{i}"))
            .body(Body::from(req.to_string()))
            .unwrap();

        let app = h.app.clone();
        inflight.push(tokio::spawn(async move {
            app.oneshot(http).await.unwrap().status()
        }));
    }

    let mut statuses = Vec::new();
    for t in inflight {
        statuses.push(t.await.unwrap());
    }

    let shed = statuses
        .iter()
        .filter(|s| **s == StatusCode::SERVICE_UNAVAILABLE)
        .count();
    let served = statuses.iter().filter(|s| **s == StatusCode::OK).count();

    assert!(
        shed > 0,
        "the server accepted unbounded work instead of shedding load: {served} served, {shed} shed"
    );
    assert!(
        served > 0,
        "the server shed everything, including work it could do"
    );
}

#[tokio::test]
async fn identical_seeds_reproduce_identical_output() {
    let h = harness("seeded", |_| {});

    let req = serde_json::json!({
        "messages": [{ "role": "user", "content": "reproducibility" }],
        "temperature": 0.9,
        "seed": 42,
        "max_tokens": 12,
    });

    let (_, a) = h.post("/v1/chat/completions", req.clone()).await;
    let (_, b) = h.post("/v1/chat/completions", req).await;

    let content = |body: &str| -> String {
        Harness::json(body)["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .to_owned()
    };
    assert_eq!(content(&a), content(&b), "a pinned seed was not honoured");
}

#[tokio::test]
async fn stats_reports_measured_counters() {
    let h = harness("stats", |_| {});

    h.post("/v1/chat/completions", chat("one")).await;
    h.post("/v1/chat/completions", chat("two")).await;

    let (status, body) = h.get("/v1/stats").await;
    assert_eq!(status, StatusCode::OK);

    let v = Harness::json(&body);
    assert_eq!(v["scheduler"]["submitted"], 2);
    assert_eq!(v["scheduler"]["completed"], 2);
    assert_eq!(v["scheduler"]["rejected_rate_limit"], 0);
    assert_eq!(v["context_window"], 512);
    assert_eq!(v["model"]["kind"], "synthetic");
}

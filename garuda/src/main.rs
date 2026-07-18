use anyhow::Context;
use clap::Parser;
use garuda::anthropic::create_anthropic_router;
use garuda::api::{ApiState, create_router};
use garuda::auth::{self, ApiKeys};
use garuda::cli::{Cli, Commands};
use garuda::config::AppConfig;
use garuda::llamacpp::create_llamacpp_router;
use garuda::ollama::create_ollama_router;
use garuda::scheduler::Scheduler;
use garuda::server::{Backend, Engine, configure_thread_pool};
use garuda::tgi::create_tgi_router;
use garuda::ui::create_ui_router;
use garuda::websocket::create_ws_router;
use std::net::SocketAddr;
use std::sync::Arc;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_env("GARUDA_LOG").unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let args = Cli::parse();

    let mut config = if args.config.exists() {
        info!(path = %args.config.display(), "loading configuration");
        AppConfig::load(&args.config)?
    } else {
        info!(path = %args.config.display(), "no configuration file; using defaults");
        AppConfig::default()
    };

    if let Some(host) = args.host {
        config.server.host = host;
    }
    if let Some(port) = args.port {
        config.server.port = port;
    }
    config.validate()?;
    configure_thread_pool(config.runtime.threads);

    match args.command {
        Some(Commands::Benchmark { iterations, tokens }) => {
            garuda::benchmark::run(&config, iterations, tokens)
        }
        Some(Commands::Inspect { file }) => inspect(&file),
        Some(Commands::Serve) | None => serve(config).await,
    }
}

fn inspect(path: &std::path::Path) -> anyhow::Result<()> {
    // mmap rather than read: a checkpoint can be far larger than RAM, and printing
    // its metadata should not require loading the whole file first.
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .with_context(|| format!("mmapping {}", path.display()))?;
    let gguf = garuda::gguf::Gguf::parse(&mmap)?;

    println!("file          {}", path.display());
    println!("gguf version  {}", gguf.version);
    println!(
        "architecture  {}",
        gguf.architecture().unwrap_or("(unknown)")
    );
    if let Some(n) = gguf.expert_count() {
        println!(
            "experts       {n} (top-{})",
            gguf.expert_used_count().unwrap_or(0)
        );
    }
    println!("tensors       {}", gguf.tensors.len());
    println!("metadata keys {}", gguf.metadata.len());
    println!("data offset   {}", gguf.data_offset);

    // Report loadability honestly: F32/F16/Q4_0/Q8_0/k-quants decode, only the
    // llama architecture is wired up, and MoE blocks need either the merged
    // `..._exps` tensors or the older per-expert tensors — matching exactly what
    // `LlamaBackend::from_gguf` requires, so this never promises a load that fails.
    let arch = gguf.architecture().unwrap_or("(unknown)");
    let bad_tensor = gguf
        .tensors
        .iter()
        .find(|t| !garuda::quant::is_supported(t.ggml_type));
    let missing_experts = match (gguf.expert_count(), gguf.arch_u64("block_count")) {
        (Some(ne), Some(nl)) if ne > 0 => (0..nl).find(|&l| {
            let stacked = gguf
                .tensor(&format!("blk.{l}.ffn_gate_exps.weight"))
                .is_some();
            let split = gguf.tensor(&format!("blk.{l}.ffn_gate.0.weight")).is_some();
            !stacked && !split
        }),
        _ => None,
    };
    println!();
    if arch != "llama" {
        println!("loadable      no — only the llama architecture is supported (this is '{arch}').");
    } else if let Some(t) = bad_tensor {
        println!(
            "loadable      no — tensor '{}' is {} ({}), which needs a super-block decoder \
             that does not exist yet.",
            t.name,
            garuda::quant::type_name(t.ggml_type),
            t.ggml_type
        );
    } else if let Some(l) = missing_experts {
        println!(
            "loadable      no — block {l} declares experts but has neither `ffn_gate_exps` \
             (merged) nor `ffn_gate.0` (per-expert) tensors; this checkpoint's MoE tensor \
             layout is not recognised."
        );
    } else {
        println!(
            "loadable      yes. Run it with `model.gguf = \"{}\"`.",
            path.display()
        );
    }
    Ok(())
}

async fn serve(config: AppConfig) -> anyhow::Result<()> {
    let engine = Engine::build(&config)?;
    let auth = ApiKeys::new(config.server.api_keys.clone());
    match &engine.backend {
        Backend::SyntheticMoe => {
            info!(
                experts = config.model.experts,
                top_k = config.model.top_k,
                context = config.model.context,
                router = %config.model.router,
                prefetch = engine.prefetch.is_some(),
                auth = auth.is_enabled(),
                "synthetic MoE engine ready"
            );
            warn!("this runtime has no trained weights: generated text is meaningless.");
        }
        Backend::Gguf { path, layers } => {
            info!(
                model = %path,
                layers,
                d_model = engine.dims.d_model,
                vocab = engine.dims.vocab_size,
                context = engine.runtime.max_context(),
                mmap = config.model.mmap,
                prefetch = engine.prefetch.is_some(),
                auth = auth.is_enabled(),
                "loaded GGUF model"
            );
        }
    }
    if auth.is_enabled() {
        info!(
            keys = config.server.api_keys.len(),
            "API key authentication enabled"
        );
    } else {
        warn!("this server has no authentication; do not expose it to an untrusted network");
    }

    let scheduler = Scheduler::new(engine.runtime.clone(), config.scheduler());

    let state = Arc::new(ApiState {
        runtime: engine.runtime.clone(),
        scheduler,
        embedding_slots: Arc::new(tokio::sync::Semaphore::new(config.server.max_concurrent)),
        defaults: config.sampling()?,
        request_timeout: config.request_timeout(),
        started: std::time::Instant::now(),
    });

    let mut app = create_router(state.clone())
        .merge(create_ui_router())
        .merge(create_ws_router(state.clone()))
        .merge(create_ollama_router(state.clone()))
        .merge(create_anthropic_router(state.clone()))
        .merge(create_llamacpp_router(state.clone()))
        .merge(create_tgi_router(state))
        .layer(axum::middleware::from_fn_with_state(
            auth.clone(),
            auth::require_key,
        ));

    if config.server.cors {
        if auth.is_enabled() {
            warn!("permissive CORS is enabled; requests still need a valid API key");
        } else {
            warn!("permissive CORS is enabled and this server has no auth");
        }
        app = app.layer(tower_http::cors::CorsLayer::permissive());
    }
    app = app.layer(tower_http::trace::TraceLayer::new_for_http());

    let addr: SocketAddr = format!("{}:{}", config.server.host, config.server.port)
        .parse()
        .with_context(|| {
            format!(
                "invalid bind address {}:{}",
                config.server.host, config.server.port
            )
        })?;

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("binding {addr}"))?;
    info!(%addr, "listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("server error")?;

    info!("shut down");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install the ctrl-c handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install the SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
    info!("shutdown signal received");
}

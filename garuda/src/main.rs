use anyhow::Context;
use clap::Parser;
use garuda::api::{create_router, ApiState};
use garuda::cli::{Cli, Commands};
use garuda::config::AppConfig;
use garuda::scheduler::Scheduler;
use garuda::server::{configure_thread_pool, Backend, Engine};
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
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let gguf = garuda::gguf::Gguf::parse(&bytes)?;

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
    println!();
    println!("Garuda can read this file's metadata but cannot load its weights:");
    println!("nothing dequantises GGUF tensors into experts yet.");
    Ok(())
}

async fn serve(config: AppConfig) -> anyhow::Result<()> {
    let engine = Engine::build(&config)?;
    match &engine.backend {
        Backend::SyntheticMoe => {
            info!(
                experts = config.model.experts,
                top_k = config.model.top_k,
                context = config.model.context,
                router = %config.model.router,
                prefetch = engine.prefetch.is_some(),
                "synthetic MoE engine ready"
            );
            warn!(
                "this runtime has no trained weights: generated text is meaningless. \
                 it also has no authentication."
            );
        }
        Backend::Gguf { path, layers } => {
            info!(
                model = %path,
                layers,
                d_model = engine.dims.d_model,
                vocab = engine.dims.vocab_size,
                context = engine.runtime.max_context(),
                "loaded GGUF model"
            );
            warn!("this server has no authentication; do not expose it to an untrusted network");
        }
    }

    let scheduler = Scheduler::new(engine.runtime.clone(), config.scheduler());

    let state = Arc::new(ApiState {
        runtime: engine.runtime.clone(),
        scheduler,
        defaults: config.sampling()?,
        request_timeout: config.request_timeout(),
        started: std::time::Instant::now(),
    });

    let mut app = create_router(state.clone()).merge(create_ws_router(state));

    if config.server.cors {
        warn!("permissive CORS is enabled and this server has no auth");
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

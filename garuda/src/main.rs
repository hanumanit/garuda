use std::sync::Arc;
use std::net::SocketAddr;
use clap::Parser;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;
use garuda::config::AppConfig;
use garuda::cli::{Cli, Commands};
use garuda::memory::MemoryManager;
use garuda::tokenizer::Tokenizer;
use garuda::predictor::ExpertPredictor;
use garuda::prefetch::PrefetchEngine;
use garuda::moe::MoeEngine;
use garuda::runtime::InferenceRuntime;
use garuda::scheduler::Scheduler;
use garuda::api::{create_router, ApiState};
use garuda::websocket::create_ws_router;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let args = Cli::parse();
    info!("Garuda LLM Runtime starting up...");

    let config = if args.config.exists() {
        info!("Loading configuration from {:?}", args.config);
        AppConfig::load_from_toml(&args.config)?
    } else {
        info!("Configuration file {:?} not found, using defaults", args.config);
        AppConfig::default()
    };

    if let Some(Commands::Benchmark { iterations }) = args.command {
        garuda::benchmark::run_benchmarks(iterations).await;
        return Ok(());
    }

    let l1_capacity = 32;
    let ssd_path = std::path::PathBuf::from(&config.model.path).join("ssd_cache");
    let hdd_path = std::path::PathBuf::from(&config.model.path).join("hdd_archive");
    let _ = std::fs::create_dir_all(&ssd_path);
    let _ = std::fs::create_dir_all(&hdd_path);

    let memory_manager = Arc::new(MemoryManager::new(l1_capacity, ssd_path, hdd_path));
    let tokenizer = Tokenizer::new();
    let predictor = ExpertPredictor::new(8);
    let prefetch_engine = PrefetchEngine::new(memory_manager.clone(), predictor);
    let moe_engine = Arc::new(MoeEngine::new(garuda::router::RouterType::Mixtral, 8, 2, memory_manager.clone()));
    
    let runtime = Arc::new(InferenceRuntime::new(tokenizer, moe_engine, prefetch_engine));
    let scheduler = Arc::new(Scheduler::new(runtime.clone()));

    let state = Arc::new(ApiState {
        runtime,
        scheduler,
    });

    let app = create_router(state.clone())
        .merge(create_ws_router())
        .layer(axum::Extension(state));

    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    info!("Garuda API Server listening on {}", addr);
    
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

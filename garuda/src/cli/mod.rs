use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "garuda")]
#[command(about = "High Performance Rust LLM Runtime with Expert Streaming", long_about = None)]
pub struct Cli {
    #[arg(short, long, value_name = "FILE", default_value = "config.toml")]
    pub config: PathBuf,

    #[arg(short, long, value_name = "PORT", default_value_t = 8080)]
    pub port: u16,

    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand, Debug, Clone)]
pub enum Commands {
    Serve,
    Benchmark {
        #[arg(short, long, default_value_t = 100)]
        iterations: usize,
    },
}

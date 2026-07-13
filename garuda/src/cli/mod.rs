use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "garuda",
    version,
    about = "Rust MoE inference runtime with tiered expert storage"
)]
pub struct Cli {
    /// Configuration file. Defaults are used if it does not exist.
    #[arg(short, long, value_name = "FILE", default_value = "config.toml")]
    pub config: PathBuf,

    /// Override the configured port.
    #[arg(short, long, value_name = "PORT")]
    pub port: Option<u16>,

    /// Override the configured bind address.
    #[arg(long, value_name = "HOST")]
    pub host: Option<String>,

    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand, Debug, Clone)]
pub enum Commands {
    /// Run the API server (the default).
    Serve,
    /// Measure startup, expert loading, cache behaviour and decode throughput.
    Benchmark {
        #[arg(short, long, default_value_t = 64)]
        iterations: usize,
        /// Tokens to generate per iteration.
        #[arg(short, long, default_value_t = 32)]
        tokens: usize,
    },
    /// Print the metadata of a GGUF file.
    Inspect {
        /// Path to a .gguf model file.
        file: PathBuf,
    },
}

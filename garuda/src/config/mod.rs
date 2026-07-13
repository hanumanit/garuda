use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub path: String,
    pub context: usize,
    pub gpu: bool,
    pub threads: usize,
    pub expert_cache: String,
    pub prefetch: bool,
    pub predictor: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub model: ModelConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig {
                path: "/models".to_string(),
                context: 32768,
                gpu: false,
                threads: 8,
                expert_cache: "256GB".to_string(),
                prefetch: true,
                predictor: true,
            },
        }
    }
}

impl AppConfig {
    pub fn load_from_toml<P: AsRef<Path>>(path: P) -> Result<Self, anyhow::Error> {
        let content = std::fs::read_to_string(path)?;
        let config: AppConfig = toml::from_str(&content)?;
        Ok(config)
    }
}

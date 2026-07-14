//! Configuration.
//!
//! Every field here reaches something. A knob that is parsed and then ignored is
//! worse than no knob, so anything that cannot be honoured — `gpu = true`, for
//! instance, when no GPU backend exists — is a startup error rather than a
//! silent no-op.

use crate::core::{GarudaError, ModelDims};
use crate::router::RouterType;
use crate::runtime::SamplingParams;
use crate::scheduler::SchedulerConfig;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct ModelConfig {
    /// Root for the L2 expert cache (and the L3 archive, if enabled).
    pub path: PathBuf,
    /// Path to a GGUF checkpoint. When set, Garuda loads that model instead of the
    /// synthetic MoE, and the `router`/`experts`/`top_k` knobs below are ignored.
    pub gguf: String,
    /// Keep the checkpoint's weights packed in a memory-mapped file and dequantise
    /// them per row at inference time, instead of expanding everything to `f32` in RAM.
    /// Far less memory (the model uses roughly its on-disk size), but slower per token.
    pub mmap: bool,
    /// `mixtral`, `deepseek` or `qwen`. Ignored when `gguf` is set.
    pub router: String,
    /// Context window, in tokens. A loaded model caps this at its own trained length.
    pub context: usize,
    pub experts: usize,
    pub top_k: usize,
    /// Attention window; `0` attends to the whole context.
    pub sliding_window: usize,
    /// There is no GPU backend. `true` is rejected at startup rather than ignored.
    pub gpu: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from("./models"),
            gguf: String::new(),
            mmap: false,
            router: "mixtral".into(),
            context: 4096,
            experts: 8,
            top_k: 2,
            sliding_window: 0,
            gpu: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct MemoryConfig {
    /// L1 budget for expert weights, e.g. `"512MB"`, `"8GB"`.
    pub expert_cache: String,
    /// KV blocks held in RAM per sequence before spilling to disk.
    pub kv_resident_blocks: usize,
    /// Prompt prefixes to remember.
    pub prompt_cache_entries: usize,
    /// Cold archive tier. Empty disables L3.
    pub archive_path: String,
    /// Spill KV blocks to disk when a sequence exceeds `kv_resident_blocks`.
    pub kv_spill: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            expert_cache: "512MB".into(),
            kv_resident_blocks: 512,
            prompt_cache_entries: 64,
            archive_path: String::new(),
            kv_spill: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct RuntimeConfig {
    /// Compute threads. `0` uses every core.
    pub threads: usize,
    pub prefetch: bool,
    pub predictor: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            threads: 0,
            prefetch: true,
            predictor: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    /// Sequences decoding at once.
    pub max_concurrent: usize,
    /// Requests allowed to wait for a slot before submissions are refused.
    pub queue_capacity: usize,
    /// Concurrent requests one user may have in flight.
    pub max_concurrent_per_user: usize,
    pub request_timeout_secs: u64,
    /// Permissive CORS. Off by default: this server has no authentication, so
    /// letting any origin call it is a decision the operator has to make.
    pub cors: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 8080,
            max_concurrent: 4,
            queue_capacity: 256,
            max_concurrent_per_user: 8,
            request_timeout_secs: 120,
            cors: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub max_tokens: usize,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_p: 0.95,
            top_k: 40,
            max_tokens: 256,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct AppConfig {
    pub model: ModelConfig,
    pub memory: MemoryConfig,
    pub runtime: RuntimeConfig,
    pub server: ServerConfig,
    pub sampling: SamplingConfig,
}

/// Parse `"512MB"`, `"8 GiB"`, `"1024"` into bytes.
///
/// `KB`/`MB`/`GB` are powers of 1024, as operators mean them when sizing a cache.
pub fn parse_size(s: &str) -> Result<usize, GarudaError> {
    let t = s.trim().to_ascii_uppercase().replace(' ', "");
    if t.is_empty() {
        return Err(GarudaError::Config("size is empty".into()));
    }

    let (digits, unit) = t.split_at(
        t.find(|c: char| !c.is_ascii_digit() && c != '.')
            .unwrap_or(t.len()),
    );
    let value: f64 = digits
        .parse()
        .map_err(|_| GarudaError::Config(format!("'{s}' is not a valid size")))?;
    if !value.is_finite() || value < 0.0 {
        return Err(GarudaError::Config(format!("'{s}' is not a valid size")));
    }

    let mult: f64 = match unit {
        "" | "B" => 1.0,
        "K" | "KB" | "KIB" => 1024.0,
        "M" | "MB" | "MIB" => 1024.0 * 1024.0,
        "G" | "GB" | "GIB" => 1024.0 * 1024.0 * 1024.0,
        "T" | "TB" | "TIB" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
        other => {
            return Err(GarudaError::Config(format!(
                "unknown size unit '{other}' in '{s}'"
            )));
        }
    };

    let bytes = value * mult;
    if bytes > usize::MAX as f64 {
        return Err(GarudaError::Config(format!(
            "size '{s}' does not fit in memory"
        )));
    }
    Ok(bytes as usize)
}

impl AppConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, GarudaError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| GarudaError::Config(format!("reading {}: {e}", path.display())))?;
        let cfg: AppConfig = toml::from_str(&text)
            .map_err(|e| GarudaError::Config(format!("parsing {}: {e}", path.display())))?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Reject anything the runtime cannot actually honour.
    pub fn validate(&self) -> Result<(), GarudaError> {
        if self.model.gpu {
            return Err(GarudaError::Config(
                "gpu = true, but Garuda has no GPU backend. Implement core::InferenceBackend \
                 and wire it up, or set gpu = false."
                    .into(),
            ));
        }
        // The router and MoE dimensions only matter for the synthetic engine; a
        // loaded checkpoint brings its own architecture.
        if self.gguf_path().is_none() {
            self.router()?;
            self.dims()?.validate()?;
        }
        self.sampling()?.validate()?;
        parse_size(&self.memory.expert_cache)?;

        if self.model.context == 0 {
            return Err(GarudaError::Config(
                "model.context must be at least 1".into(),
            ));
        }
        if self.model.sliding_window > self.model.context {
            return Err(GarudaError::Config(format!(
                "model.sliding_window ({}) exceeds model.context ({})",
                self.model.sliding_window, self.model.context
            )));
        }
        if self.server.max_concurrent == 0 {
            return Err(GarudaError::Config(
                "server.max_concurrent must be at least 1".into(),
            ));
        }
        if self.server.queue_capacity == 0 {
            return Err(GarudaError::Config(
                "server.queue_capacity must be at least 1".into(),
            ));
        }
        if self.server.max_concurrent_per_user == 0 {
            return Err(GarudaError::Config(
                "server.max_concurrent_per_user must be at least 1".into(),
            ));
        }
        if self.server.request_timeout_secs == 0 {
            return Err(GarudaError::Config(
                "server.request_timeout_secs must be at least 1".into(),
            ));
        }
        Ok(())
    }

    /// The GGUF checkpoint to load, if one is configured.
    pub fn gguf_path(&self) -> Option<PathBuf> {
        let p = self.model.gguf.trim();
        (!p.is_empty()).then(|| PathBuf::from(p))
    }

    pub fn router(&self) -> Result<RouterType, GarudaError> {
        self.model.router.parse()
    }

    pub fn dims(&self) -> Result<ModelDims, GarudaError> {
        let d = ModelDims {
            n_experts: self.model.experts,
            top_k: self.model.top_k,
            ..Default::default()
        };
        d.validate()?;
        Ok(d)
    }

    pub fn expert_cache_bytes(&self) -> Result<usize, GarudaError> {
        parse_size(&self.memory.expert_cache)
    }

    pub fn sliding_window(&self) -> Option<usize> {
        (self.model.sliding_window > 0).then_some(self.model.sliding_window)
    }

    pub fn archive_path(&self) -> Option<PathBuf> {
        let p = self.memory.archive_path.trim();
        (!p.is_empty()).then(|| PathBuf::from(p))
    }

    pub fn sampling(&self) -> Result<SamplingParams, GarudaError> {
        let p = SamplingParams {
            temperature: self.sampling.temperature,
            top_p: self.sampling.top_p,
            top_k: self.sampling.top_k,
            max_tokens: self.sampling.max_tokens,
            seed: None,
        };
        p.validate()?;
        Ok(p)
    }

    pub fn scheduler(&self) -> SchedulerConfig {
        SchedulerConfig {
            max_concurrent: self.server.max_concurrent,
            queue_capacity: self.server.queue_capacity,
            max_concurrent_per_user: self.server.max_concurrent_per_user,
        }
    }

    pub fn request_timeout(&self) -> Duration {
        Duration::from_secs(self.server.request_timeout_secs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sizes_with_and_without_units() {
        assert_eq!(parse_size("1024").unwrap(), 1024);
        assert_eq!(parse_size("1KB").unwrap(), 1024);
        assert_eq!(parse_size("2 MB").unwrap(), 2 * 1024 * 1024);
        assert_eq!(parse_size("1GiB").unwrap(), 1024 * 1024 * 1024);
        assert_eq!(parse_size("0.5GB").unwrap(), 512 * 1024 * 1024);
        assert_eq!(parse_size("8gb").unwrap(), 8 * 1024 * 1024 * 1024);
    }

    #[test]
    fn rejects_nonsense_sizes() {
        for bad in ["", "abc", "12XB", "-5MB", "MB"] {
            assert!(parse_size(bad).is_err(), "accepted '{bad}'");
        }
    }

    #[test]
    fn default_config_is_valid() {
        AppConfig::default().validate().unwrap();
    }

    #[test]
    fn gpu_true_is_a_startup_error_not_a_silent_no_op() {
        let mut c = AppConfig::default();
        c.model.gpu = true;
        let err = c.validate().unwrap_err();
        assert!(matches!(err, GarudaError::Config(_)));
        assert!(err.to_string().contains("no GPU backend"));
    }

    #[test]
    fn rejects_a_top_k_larger_than_the_expert_count() {
        let mut c = AppConfig::default();
        c.model.experts = 4;
        c.model.top_k = 8;
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_a_sliding_window_wider_than_the_context() {
        let mut c = AppConfig::default();
        c.model.context = 128;
        c.model.sliding_window = 256;
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_zero_valued_server_limits() {
        for mutate in [
            (|c: &mut AppConfig| c.server.max_concurrent = 0) as fn(&mut AppConfig),
            |c: &mut AppConfig| c.server.queue_capacity = 0,
            |c: &mut AppConfig| c.server.max_concurrent_per_user = 0,
            |c: &mut AppConfig| c.server.request_timeout_secs = 0,
        ] {
            let mut c = AppConfig::default();
            mutate(&mut c);
            assert!(c.validate().is_err());
        }
    }

    #[test]
    fn an_unknown_key_is_an_error_rather_than_being_ignored() {
        let toml = r#"
            [model]
            path = "/models"
            contxt = 4096
        "#;
        let err = toml::from_str::<AppConfig>(toml).unwrap_err();
        assert!(err.to_string().contains("contxt"), "got: {err}");
    }

    #[test]
    fn a_partial_file_fills_the_rest_from_defaults() {
        let toml = r#"
            [model]
            context = 2048

            [server]
            port = 9999
        "#;
        let c: AppConfig = toml::from_str(toml).unwrap();
        c.validate().unwrap();

        assert_eq!(c.model.context, 2048);
        assert_eq!(c.server.port, 9999);
        assert_eq!(c.model.router, "mixtral", "should fall back to the default");
        assert_eq!(c.sampling.top_k, 40);
    }

    #[test]
    fn every_router_name_resolves() {
        for name in ["mixtral", "deepseek", "qwen"] {
            let mut c = AppConfig::default();
            c.model.router = name.into();
            assert!(c.validate().is_ok(), "{name} was rejected");
        }
        let mut c = AppConfig::default();
        c.model.router = "gpt".into();
        assert!(c.validate().is_err());
    }
}

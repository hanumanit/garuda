//! Assembling the engine from configuration.
//!
//! `serve` and `benchmark` both go through here, so they cannot drift apart. Two
//! engines can be built: the synthetic MoE (no checkpoint) and a real model loaded
//! from GGUF. Both expose the same runtime, so nothing downstream knows which it is.

use crate::cache::KvConfig;
use crate::config::AppConfig;
use crate::core::{ExpertLoader, InferenceBackend, ModelDims, StorageBackend};
use crate::gguf::Gguf;
use crate::llama::LlamaBackend;
use crate::memory::MemoryManager;
use crate::moe::MoeEngine;
use crate::predictor::ExpertPredictor;
use crate::prefetch::PrefetchEngine;
use crate::router::Router;
use crate::runtime::InferenceRuntime;
use crate::storage::LocalStorageBackend;
use crate::tokenizer::{Tokenize, Tokenizer, spm::SpmTokenizer};
use crate::weights::ModelWeights;
use anyhow::Context;
use std::sync::Arc;

/// Which backend the engine is running.
#[derive(Debug, Clone)]
pub enum Backend {
    /// The synthetic MoE with pseudo-random weights.
    SyntheticMoe,
    /// A real checkpoint loaded from GGUF.
    Gguf { path: String, layers: usize },
}

pub struct Engine {
    pub dims: ModelDims,
    pub backend: Backend,
    /// The tiered expert store — only the synthetic MoE uses one.
    pub memory: Option<Arc<MemoryManager>>,
    pub runtime: Arc<InferenceRuntime>,
    pub prefetch: Option<Arc<PrefetchEngine>>,
}

impl Engine {
    pub fn build(config: &AppConfig) -> anyhow::Result<Self> {
        config.validate()?;
        match config.gguf_path() {
            Some(path) => Self::build_gguf(config, &path),
            None => Self::build_synthetic(config),
        }
    }

    /// Somewhere for the KV cache to spill. Without it a sequence is still bounded by
    /// the context window; it just holds all of it in RAM.
    fn kv_storage(config: &AppConfig) -> anyhow::Result<Option<Arc<dyn StorageBackend>>> {
        if !config.memory.kv_spill {
            return Ok(None);
        }
        let dir = config.model.path.join("kv_spill");
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating KV spill directory at {}", dir.display()))?;
        Ok(Some(Arc::new(LocalStorageBackend::new(dir))))
    }

    fn build_gguf(config: &AppConfig, path: &std::path::Path) -> anyhow::Result<Self> {
        // Either memory-map the file (weights stay packed, low RAM) or read it into a
        // buffer and expand every weight to f32 (more RAM, faster).
        let mmap: Option<Arc<memmap2::Mmap>> = if config.model.mmap {
            let file = std::fs::File::open(path)
                .with_context(|| format!("opening checkpoint {}", path.display()))?;
            // Safety: the file is opened read-only and the mapping is held for the
            // process lifetime inside the backend; we never mutate it.
            let map = unsafe { memmap2::Mmap::map(&file) }
                .with_context(|| format!("mmapping checkpoint {}", path.display()))?;
            Some(Arc::new(map))
        } else {
            None
        };

        let owned;
        let bytes: &[u8] = match &mmap {
            Some(m) => &m[..],
            None => {
                owned = std::fs::read(path)
                    .with_context(|| format!("reading checkpoint {}", path.display()))?;
                &owned
            }
        };

        let gguf = Gguf::parse(bytes)?;
        let tokenizer: Arc<dyn Tokenize> = Arc::new(SpmTokenizer::from_gguf(&gguf)?);
        let backend = LlamaBackend::from_gguf(&gguf, bytes, mmap.clone())?;
        let lc = backend.config();
        let dims = backend.dims();

        // Never promise a longer context than the model was trained for.
        let max_positions = config.model.context.min(lc.context).max(1);

        let kv = KvConfig {
            dims,
            kv_dim: lc.kv_dim(),
            n_layers: lc.n_layers,
            max_positions,
            max_resident_blocks: config.memory.kv_resident_blocks,
            sliding_window: config.sliding_window(),
            storage: Self::kv_storage(config)?,
        };

        let runtime = Arc::new(InferenceRuntime::new(
            tokenizer,
            Arc::new(backend),
            kv,
            config.memory.prompt_cache_entries,
        ));

        Ok(Self {
            dims,
            backend: Backend::Gguf {
                path: path.display().to_string(),
                layers: lc.n_layers,
            },
            memory: None,
            runtime,
            prefetch: None,
        })
    }

    fn build_synthetic(config: &AppConfig) -> anyhow::Result<Self> {
        let dims = config.dims()?;
        let router = Router::new(config.router()?, dims)?;

        let l2_dir = config.model.path.join("l2_cache");
        std::fs::create_dir_all(&l2_dir)
            .with_context(|| format!("creating L2 cache at {}", l2_dir.display()))?;
        let l2: Arc<dyn StorageBackend> = Arc::new(LocalStorageBackend::new(&l2_dir));

        let l3: Option<Arc<dyn StorageBackend>> = match config.archive_path() {
            Some(p) => {
                std::fs::create_dir_all(&p)
                    .with_context(|| format!("creating L3 archive at {}", p.display()))?;
                Some(Arc::new(LocalStorageBackend::new(p)))
            }
            None => None,
        };

        let memory = Arc::new(MemoryManager::new(
            dims,
            config.expert_cache_bytes()?,
            l2,
            l3,
        )?);

        let prefetch = if config.runtime.prefetch && config.runtime.predictor {
            let predictor = Arc::new(ExpertPredictor::new(dims.n_experts));
            let loader: Arc<dyn ExpertLoader> = memory.clone();
            Some(Arc::new(PrefetchEngine::new(
                loader, predictor, true, dims.top_k,
            )))
        } else {
            None
        };

        let weights = Arc::new(ModelWeights::synthesize(dims)?);
        let backend = Arc::new(MoeEngine::new(
            dims,
            weights,
            router,
            memory.clone(),
            prefetch.clone(),
        )?);

        let kv = KvConfig::mha(
            dims,
            config.model.context,
            config.memory.kv_resident_blocks,
            config.sliding_window(),
            Self::kv_storage(config)?,
        );

        let runtime = Arc::new(InferenceRuntime::new(
            Arc::new(Tokenizer::new()),
            backend,
            kv,
            config.memory.prompt_cache_entries,
        ));

        Ok(Self {
            dims,
            backend: Backend::SyntheticMoe,
            memory: Some(memory),
            runtime,
            prefetch,
        })
    }
}

/// Size the rayon pool from `runtime.threads`. `0` leaves rayon's default (all cores).
///
/// Called once; a second call is a no-op, because rayon's global pool can only be
/// built once per process.
pub fn configure_thread_pool(threads: usize) {
    if threads == 0 {
        return;
    }
    if let Err(e) = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
    {
        tracing::warn!(error = %e, "could not size the rayon pool; using the default");
    }
}

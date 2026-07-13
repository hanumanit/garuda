//! Assembling the engine from configuration.
//!
//! `serve` and `benchmark` both go through here, so they cannot drift apart.

use crate::cache::KvConfig;
use crate::config::AppConfig;
use crate::core::{ExpertLoader, ModelDims, StorageBackend};
use crate::memory::MemoryManager;
use crate::moe::MoeEngine;
use crate::predictor::ExpertPredictor;
use crate::prefetch::PrefetchEngine;
use crate::router::Router;
use crate::runtime::InferenceRuntime;
use crate::storage::LocalStorageBackend;
use crate::tokenizer::Tokenizer;
use crate::weights::ModelWeights;
use anyhow::Context;
use std::sync::Arc;

pub struct Engine {
    pub dims: ModelDims,
    pub memory: Arc<MemoryManager>,
    pub runtime: Arc<InferenceRuntime>,
    pub prefetch: Option<Arc<PrefetchEngine>>,
}

impl Engine {
    pub fn build(config: &AppConfig) -> anyhow::Result<Self> {
        config.validate()?;

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

        // Spilling needs somewhere to spill to. Without it, a sequence is still
        // bounded by the context window; it just holds all of it in RAM.
        let kv_storage: Option<Arc<dyn StorageBackend>> = if config.memory.kv_spill {
            let dir = config.model.path.join("kv_spill");
            std::fs::create_dir_all(&dir)
                .with_context(|| format!("creating KV spill directory at {}", dir.display()))?;
            Some(Arc::new(LocalStorageBackend::new(dir)))
        } else {
            None
        };

        let kv = KvConfig {
            dims,
            max_positions: config.model.context,
            max_resident_blocks: config.memory.kv_resident_blocks,
            sliding_window: config.sliding_window(),
            storage: kv_storage,
        };

        let runtime = Arc::new(InferenceRuntime::new(
            Tokenizer::new(),
            backend,
            kv,
            config.memory.prompt_cache_entries,
        ));

        Ok(Self {
            dims,
            memory,
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

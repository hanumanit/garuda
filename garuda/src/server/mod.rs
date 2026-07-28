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
use crate::prefetch::{GgufPagePrefetcher, PrefetchEngine};
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

    /// Physical RAM in bytes, or `None` when it cannot be determined.
    ///
    /// Used only to guess whether a checkpoint can live in the page cache. An unknown
    /// answer is treated as "assume it fits", which selects the order that is faster
    /// in that case and never worse than what the code did before.
    fn physical_memory() -> Option<u64> {
        #[cfg(target_os = "macos")]
        {
            let mut bytes: u64 = 0;
            let mut len = std::mem::size_of::<u64>();
            let name = c"hw.memsize";
            // Safety: `name` is a valid C string, and the out-pointer and its length
            // describe the `u64` above.
            let rc = unsafe {
                libc::sysctlbyname(
                    name.as_ptr(),
                    (&raw mut bytes).cast(),
                    &raw mut len,
                    std::ptr::null_mut(),
                    0,
                )
            };
            return (rc == 0 && bytes > 0).then_some(bytes);
        }
        #[cfg(target_os = "linux")]
        {
            let text = std::fs::read_to_string("/proc/meminfo").ok()?;
            let kb: u64 = text
                .lines()
                .find_map(|l| l.strip_prefix("MemTotal:"))?
                .split_whitespace()
                .next()?
                .parse()
                .ok()?;
            return Some(kb * 1024);
        }
        #[allow(unreachable_code)]
        None
    }

    /// How many prompt tokens should share one pass over a layer's weights.
    ///
    /// Layer-major prefill trades CPU-cache locality for page-cache locality, so it
    /// is only worth it when the checkpoint cannot sit in the page cache — see
    /// `LlamaBackend::prefill_chunk` for the measurements on both sides. The rule of
    /// thumb: if the file is over half of physical RAM, the page cache is not going
    /// to hold it alongside everything else, and re-reading every layer once per
    /// prompt token becomes the dominant cost.
    ///
    /// Only applies to a memory-mapped checkpoint. Without `mmap` the weights are
    /// already expanded in RAM and there is no paging to avoid.
    fn prefill_chunk(config: &AppConfig, checkpoint_bytes: u64) -> usize {
        match config.model.prefill_batch {
            0 => {}               // auto: decide below
            n => return n.max(1), // operator override, including 1 = off
        }
        if !config.model.mmap {
            return 1;
        }
        match Self::physical_memory() {
            Some(ram) if checkpoint_bytes * 2 > ram => crate::llama::DEFAULT_PREFILL_CHUNK,
            _ => 1,
        }
    }

    /// Warn when the KV cache is configured to spill under full attention.
    ///
    /// Attention over the whole context has to read every earlier position, so each
    /// decode step calls `ensure_resident(0, len)` and pulls the entire spilled
    /// prefix back into RAM — which the next `append` then spills out again. The
    /// spill is undone and redone once per token, turning disk I/O quadratic in
    /// sequence length. A sliding window bounds what has to be resident and makes
    /// spilling behave; without one, `kv_resident_blocks` has to cover the context.
    fn warn_if_kv_spill_thrashes(kv: &KvConfig) {
        let blocks = kv.max_positions.div_ceil(kv.dims.block_size.max(1));
        if kv.storage.is_some() && kv.sliding_window.is_none() && kv.max_resident_blocks < blocks {
            tracing::warn!(
                kv_resident_blocks = kv.max_resident_blocks,
                blocks_for_full_context = blocks,
                "a full-context sequence will spill and immediately reload its KV cache every \
                 token; set model.sliding_window, or raise memory.kv_resident_blocks to at \
                 least {blocks}, or set memory.kv_spill = false"
            );
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

        let chunk = Self::prefill_chunk(config, bytes.len() as u64);
        let why = match (config.model.prefill_batch, chunk) {
            (0, 1) => "token-major (the checkpoint should fit in the page cache)",
            (0, _) => "layer-major (the checkpoint is large relative to RAM)",
            (_, 1) => "token-major (set by model.prefill_batch)",
            (_, _) => "layer-major (set by model.prefill_batch)",
        };
        tracing::info!(
            prefill_batch = chunk,
            checkpoint_mb = bytes.len() / 1_048_576,
            ram_mb = Self::physical_memory().map(|r| r / 1_048_576),
            "prefill order: {why}"
        );
        let backend = backend.with_prefill_chunk(chunk);

        // Prefetching only helps a mmapped MoE checkpoint: it hides an expert's page
        // faults by touching its pages on a background thread while the current step
        // still computes. A dense model, or one expanded to f32 in RAM, has nothing
        // to warm — mmap is what makes touching a page ever cost a disk read.
        let prefetch = if config.runtime.prefetch && config.runtime.predictor && lc.n_experts > 0 {
            mmap.clone().map(|m| {
                let ranges = backend.expert_page_ranges();
                let predictor = Arc::new(ExpertPredictor::new(lc.n_layers * lc.n_experts));
                let loader: Arc<dyn ExpertLoader> = Arc::new(GgufPagePrefetcher::new(m, ranges));
                Arc::new(PrefetchEngine::new(
                    loader,
                    predictor,
                    true,
                    lc.n_experts_used.max(1),
                ))
            })
        } else {
            None
        };
        let backend = match &prefetch {
            Some(pf) => backend.with_prefetch(pf.clone()),
            None => backend,
        };

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
        Self::warn_if_kv_spill_thrashes(&kv);

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
            prefetch,
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
        Self::warn_if_kv_spill_thrashes(&kv);

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

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(mmap: bool, prefill_batch: usize) -> AppConfig {
        let mut c = AppConfig::default();
        c.model.mmap = mmap;
        c.model.prefill_batch = prefill_batch;
        c
    }

    #[test]
    fn physical_memory_is_readable_on_this_platform() {
        // The prefill heuristic degrades to token-major without it, so a silent
        // `None` would quietly disable the optimisation on a machine that needs it.
        let ram = Engine::physical_memory();
        assert!(ram.is_some(), "could not read physical memory");
        assert!(
            ram.unwrap() > 256 << 20,
            "implausible RAM reading: {ram:?} bytes"
        );
    }

    #[test]
    fn prefill_batching_turns_on_only_for_a_checkpoint_too_big_to_cache() {
        let ram = Engine::physical_memory().expect("physical memory");

        // Comfortably cacheable: token-major, which measures faster in that case.
        assert_eq!(Engine::prefill_chunk(&cfg(true, 0), ram / 8), 1);
        // Over half of RAM: the page cache will not hold it beside everything else.
        assert_eq!(
            Engine::prefill_chunk(&cfg(true, 0), ram),
            crate::llama::DEFAULT_PREFILL_CHUNK
        );
        // Without mmap the weights are already expanded in RAM; there is no paging
        // to avoid, however big the file was on disk.
        assert_eq!(Engine::prefill_chunk(&cfg(false, 0), ram * 4), 1);
    }

    #[test]
    fn an_explicit_prefill_batch_overrides_the_heuristic_both_ways() {
        let ram = Engine::physical_memory().expect("physical memory");
        // Forced on for a small checkpoint the heuristic would leave token-major.
        assert_eq!(Engine::prefill_chunk(&cfg(true, 64), ram / 8), 64);
        // Forced off for a huge one it would have batched.
        assert_eq!(Engine::prefill_chunk(&cfg(true, 1), ram * 4), 1);
    }
}

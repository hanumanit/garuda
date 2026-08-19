//! Assembling the engine from configuration.
//!
//! `serve` and `benchmark` both go through here, so they cannot drift apart. Two
//! engines can be built: the synthetic MoE (no checkpoint) and a real model loaded
//! from GGUF. Both expose the same runtime, so nothing downstream knows which it is.

use crate::cache::KvConfig;
use crate::chat::ChatFormat;
use crate::config::AppConfig;
use crate::core::{ExpertLoader, InferenceBackend, ModelDims, StorageBackend};
use crate::gguf::Gguf;
use crate::llama::LlamaBackend;
use crate::memory::MemoryManager;
use crate::moe::MoeEngine;
use crate::predictor::ExpertPredictor;
use crate::prefetch::{GgufPagePrefetcher, PrefetchEngine};
use crate::qwen35::Qwen35Backend;
use crate::router::Router;
use crate::runtime::InferenceRuntime;
use crate::storage::LocalStorageBackend;
use crate::tokenizer::{Tokenize, Tokenizer, bpe::BpeTokenizer, spm::SpmTokenizer};
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
    /// The turn markup this checkpoint was fine-tuned on, read from its own
    /// `tokenizer.chat_template`. The API adapters render with it.
    pub chat: ChatFormat,
}

impl Engine {
    pub fn build(config: &AppConfig) -> anyhow::Result<Self> {
        config.validate()?;
        match config.gguf_path() {
            Some(path) => Self::build_gguf(config, &path),
            None => Self::build_synthetic(config),
        }
    }

    /// Load the draft checkpoint and the cache shape it needs.
    ///
    /// Refuses a vocabulary that does not match the main model's. A draft that
    /// tokenises differently would hand back ids meaning different words, and every
    /// layer below here would accept them without complaint — the guesses would
    /// simply always be wrong, or worse, occasionally right by coincidence.
    fn load_draft(
        config: &AppConfig,
        path: &std::path::Path,
        target_vocab: usize,
    ) -> anyhow::Result<(Arc<dyn InferenceBackend>, KvConfig)> {
        let mmap: Option<Arc<memmap2::Mmap>> = if config.model.draft_mmap {
            let file = std::fs::File::open(path)
                .with_context(|| format!("opening draft checkpoint {}", path.display()))?;
            // Safety: opened read-only, held for the process lifetime, never mutated.
            Some(Arc::new(
                unsafe { memmap2::Mmap::map(&file) }
                    .with_context(|| format!("mmapping draft checkpoint {}", path.display()))?,
            ))
        } else {
            None
        };
        let owned;
        let bytes: &[u8] = match &mmap {
            Some(m) => &m[..],
            None => {
                owned = std::fs::read(path)
                    .with_context(|| format!("reading draft checkpoint {}", path.display()))?;
                &owned
            }
        };

        let gguf = Gguf::parse(bytes)?;
        let draft = LlamaBackend::from_gguf(&gguf, bytes, mmap.clone())?;
        let dc = draft.config();
        if dc.vocab != target_vocab {
            anyhow::bail!(
                "draft checkpoint {} has a {}-token vocabulary but the model has {} — \
                 they must be the same, or a token id means different words to each",
                path.display(),
                dc.vocab,
                target_vocab
            );
        }

        let kv = KvConfig {
            dims: draft.dims(),
            kv_dim: dc.kv_dim(),
            n_layers: dc.n_layers,
            kv_dims: None,
            max_positions: config.model.context.min(dc.context).max(1),
            max_resident_blocks: config.memory.kv_resident_blocks,
            sliding_window: config.sliding_window(),
            // Small and short-lived; spilling it would cost more than it saves.
            storage: None,
        };
        tracing::info!(
            draft = %path.display(),
            layers = dc.n_layers,
            vocab = dc.vocab,
            checkpoint_mb = bytes.len() / 1_048_576,
            "draft model loaded"
        );
        Ok((Arc::new(draft), kv))
    }

    /// How many prompt tokens share one pass over a layer's weights.
    ///
    /// Batching is a straight win — see [`crate::llama::LlamaBackend::with_prefill_chunk`]
    /// for the measurements — so this is only about honouring an override.
    fn prefill_chunk(config: &AppConfig) -> usize {
        match config.model.prefill_batch {
            0 => crate::llama::DEFAULT_PREFILL_CHUNK,
            n => n.max(1),
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

    /// Load a Qwen3.5-family checkpoint — the hybrid architecture in [`crate::qwen35`].
    ///
    /// Three quarters of its blocks keep a fixed-size recurrent state instead of a
    /// growing KV cache, which changes what this function has to arrange: per-layer
    /// cache widths (zero for the recurrent blocks), a prompt-cache budget that can
    /// actually hold one of these sequences, and no draft model, because a state that
    /// summarises every token it has read cannot be rewound when a guess is rejected.
    fn build_qwen35(
        config: &AppConfig,
        path: &std::path::Path,
        gguf: &Gguf,
        bytes: &[u8],
        mmap: Option<Arc<memmap2::Mmap>>,
    ) -> anyhow::Result<Self> {
        let tokenizer: Arc<dyn Tokenize> = Arc::new(BpeTokenizer::from_gguf(gguf)?);
        let mmap_for_prefetch = mmap.clone();
        let backend = Qwen35Backend::from_gguf(gguf, bytes, mmap)?;
        let qc = backend.config();
        let dims = backend.dims();
        if dims.vocab_size != tokenizer.vocab_size() {
            anyhow::bail!(
                "the checkpoint's output head covers {} tokens but its vocabulary lists {} — \
                 a sampled id would decode to the wrong word",
                dims.vocab_size,
                tokenizer.vocab_size()
            );
        }

        if let Some(draft) = config.draft_path() {
            anyhow::bail!(
                "model.draft_gguf is set ({}), but this architecture cannot verify \
                 speculated tokens: its recurrent layers summarise every token they \
                 read and cannot be rewound when a guess is rejected. Clear the key to \
                 run this checkpoint.",
                draft.display()
            );
        }

        let chunk = Self::prefill_chunk(config);
        let backend = backend.with_prefill_chunk(chunk);

        // A dense checkpoint has no experts to predict, but it does have a next block,
        // and on one larger than RAM that block has to come off disk while the current
        // one computes or the CPU spends the wait idle. Only worth anything when the
        // weights are mapped: an in-RAM checkpoint has nothing to warm.
        let prefetch = if config.runtime.prefetch && backend.is_mmapped() {
            mmap_for_prefetch.map(|m| {
                Arc::new(crate::prefetch::LayerPrefetcher::new(
                    m,
                    backend.layer_spans().to_vec(),
                ))
            })
        } else {
            None
        };
        let backend = match &prefetch {
            Some(pf) => backend.with_prefetch(pf.clone()),
            None => backend,
        };
        let recurrent = qc.recurrent.iter().filter(|&&r| r).count();
        tracing::info!(
            prefill_batch = chunk,
            checkpoint_mb = bytes.len() / 1_048_576,
            blocks = qc.n_layers,
            recurrent_blocks = recurrent,
            attention_blocks = qc.n_layers - recurrent,
            heads = format!("{}x{}", qc.n_heads, qc.head_dim),
            kv_heads = qc.n_kv_heads,
            recurrent_state_mb = qc.linear_state_bytes() / 1_048_576,
            prefetch = prefetch.is_some(),
            "qwen3.5: hybrid attention, {} of {} blocks recurrent",
            recurrent,
            qc.n_layers
        );

        // Never promise a longer context than the model was trained for.
        let max_positions = config.model.context.min(qc.context).max(1);
        let kv = KvConfig {
            dims,
            kv_dim: qc.kv_dim(),
            n_layers: qc.n_layers,
            // Zero for the recurrent blocks: they count positions and store nothing.
            kv_dims: Some(qc.kv_dims()),
            max_positions,
            max_resident_blocks: config.memory.kv_resident_blocks,
            sliding_window: config.sliding_window(),
            storage: Self::kv_storage(config)?,
        };
        Self::warn_if_kv_spill_thrashes(&kv);

        // A cached prompt carries the recurrent state, whose size does not depend on
        // the prompt. If one does not fit the budget, nothing ever will — the cache
        // would take every insertion and decline it.
        let state_bytes = qc.linear_state_bytes();
        let budget = config.prompt_cache_bytes()?;
        if state_bytes > budget {
            tracing::warn!(
                recurrent_state_mb = state_bytes / 1_048_576,
                prompt_cache_mb = budget / 1_048_576,
                "one sequence's recurrent state is larger than the whole prompt cache, so \
                 no prompt will ever be cached; raise memory.prompt_cache past {} MB or \
                 accept that repeated prompts re-run prefill",
                state_bytes / 1_048_576
            );
        }

        let chat = ChatFormat::detect(
            gguf.get("tokenizer.chat_template")
                .and_then(crate::gguf::Value::as_str),
        )
        .with_thinking(config.model.thinking);
        let turn_end = chat.turn_end().and_then(|m| tokenizer.token_id(m));
        tracing::info!(
            format = chat.as_str(),
            turn_end = ?turn_end,
            "chat template"
        );

        let runtime = Arc::new(
            InferenceRuntime::new(
                tokenizer,
                Arc::new(backend),
                kv,
                config.memory.prompt_cache_entries,
                budget,
            )
            .with_turn_end(turn_end),
        );

        Ok(Self {
            dims,
            backend: Backend::Gguf {
                path: path.display().to_string(),
                layers: qc.n_layers,
            },
            memory: None,
            runtime,
            // `Engine::prefetch` is the MoE expert engine, which a dense model has no
            // use for; the block prefetcher lives inside the backend.
            prefetch: None,
            chat,
        })
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

        // One file format, more than one architecture inside it. Each loader refuses
        // what it does not implement, so a checkpoint never half-loads.
        if gguf.architecture() == Some("qwen35") {
            return Self::build_qwen35(config, path, &gguf, bytes, mmap.clone());
        }

        let tokenizer: Arc<dyn Tokenize> = Arc::new(SpmTokenizer::from_gguf(&gguf)?);
        let backend = LlamaBackend::from_gguf(&gguf, bytes, mmap.clone())?;
        let lc = backend.config();
        let dims = backend.dims();

        let chunk = Self::prefill_chunk(config);
        tracing::info!(
            prefill_batch = chunk,
            checkpoint_mb = bytes.len() / 1_048_576,
            "prefill: {}",
            if chunk > 1 {
                "layer-major, tokens grouped by expert"
            } else {
                "token-major (batching disabled by model.prefill_batch)"
            }
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
            kv_dims: None,
            max_positions,
            max_resident_blocks: config.memory.kv_resident_blocks,
            sliding_window: config.sliding_window(),
            storage: Self::kv_storage(config)?,
        };
        Self::warn_if_kv_spill_thrashes(&kv);

        // A checkpoint states its own prompt format. Ignoring it does not fail
        // loudly: an instruction-tuned model handed a `user: ...` transcript reverts
        // to completing the document, answering and then writing the user's next turn.
        let chat = ChatFormat::detect(
            gguf.get("tokenizer.chat_template")
                .and_then(crate::gguf::Value::as_str),
        );
        let turn_end = chat.turn_end().and_then(|m| tokenizer.token_id(m));
        tracing::info!(
            format = chat.as_str(),
            turn_end = ?turn_end,
            "chat template"
        );
        if chat == ChatFormat::Plain {
            tracing::warn!(
                "this checkpoint names no chat template; falling back to a plain \
                 role transcript, which an instruction-tuned model may continue \
                 past its own turn"
            );
        }

        let mut runtime = InferenceRuntime::new(
            tokenizer,
            Arc::new(backend),
            kv,
            config.memory.prompt_cache_entries,
            config.prompt_cache_bytes()?,
        )
        .with_turn_end(turn_end);
        if let Some(path) = config.draft_path() {
            let (drafter, draft_kv) = Self::load_draft(config, &path, dims.vocab_size)?;
            runtime = runtime.with_drafter(drafter, draft_kv);
        }
        let runtime = Arc::new(runtime);

        Ok(Self {
            dims,
            backend: Backend::Gguf {
                path: path.display().to_string(),
                layers: lc.n_layers,
            },
            memory: None,
            runtime,
            prefetch,
            chat,
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
            config.prompt_cache_bytes()?,
        ));

        Ok(Self {
            dims,
            backend: Backend::SyntheticMoe,
            memory: Some(memory),
            runtime,
            prefetch,
            // Synthetic weights are random, so no markup is more correct than another.
            chat: ChatFormat::Plain,
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

    #[test]
    fn prefill_batching_is_on_by_default_and_overridable() {
        let mut c = AppConfig::default();
        assert_eq!(c.model.prefill_batch, 0, "0 means 'use the default'");
        assert_eq!(
            Engine::prefill_chunk(&c),
            crate::llama::DEFAULT_PREFILL_CHUNK
        );

        // Grouping tokens by expert makes batching a win on both the packed and the
        // f32 paths, so there is no configuration it is withheld for — only an
        // explicit opt-out.
        c.model.prefill_batch = 1;
        assert_eq!(Engine::prefill_chunk(&c), 1);
        c.model.prefill_batch = 64;
        assert_eq!(Engine::prefill_chunk(&c), 64);
    }
}

//! A Llama-family transformer backend, loaded from a GGUF checkpoint.
//!
//! Real weights, real math: `token_embd`, per-block RMSNorm + grouped-query attention
//! with RoPE, and a feed-forward network that is either a dense SwiGLU or a mixture of
//! experts (a router picks the top-k experts to run per token). A final norm and an
//! output head produce the logits. It implements the same [`InferenceBackend`] as the
//! synthetic MoE engine, so it drops into the existing runtime, scheduler and API.
//!
//! Weights load in any format [`crate::quant`] decodes (F32/F16/Q4_0/Q8_0/Q2_K–Q6_K).
//! With `mmap`, each projection — including per-expert matrices — stays packed in the
//! mapped file and is dequantised a row at a time; for MoE that means a token only
//! pages in the top-k experts it routes to.
//!
//! MoE experts load from either GGUF layout in the wild: a single stacked
//! `..._exps` tensor (newer llama.cpp conversions), or one tensor per expert like
//! `blk.0.ffn_gate.3.weight` (older conversions, e.g. the original TheBloke Mixtral
//! quantisations). See `ExpertWeight`.

use crate::cache::{KVCacheState, SeqState};
use crate::core::{ExpertId, GarudaError, InferenceBackend, ModelDims, Tensor, Token};
use crate::gguf::Gguf;
use crate::{quant, simd};
use memmap2::Mmap;
use std::sync::Arc;

/// Default prompt tokens driven through one layer before moving to the next, when
/// the checkpoint is large enough for it to pay. See
/// [`LlamaBackend::with_prefill_chunk`] for when that is, and why.
///
/// The value bounds the activation buffer (`chunk * d_model * 4` bytes — 4 MB at
/// Mixtral's width), not the benefit, which saturates long before this.
pub const DEFAULT_PREFILL_CHUNK: usize = 256;

/// One weight matrix, either expanded to `f32` in RAM or kept packed (quantised) in a
/// memory-mapped file and dequantised a row at a time during matmul.
///
/// `Full` is fast (dequantised once at load) but holds the whole `f32` matrix; `Packed`
/// trades speed for memory — the model occupies its on-disk (quantised) size, so a
/// checkpoint far larger than RAM can run via demand paging.
///
/// Shared with [`crate::qwen35`], which loads a different architecture out of the same
/// file format and wants the same choice between the two.
pub(crate) enum Weight {
    Full {
        data: Vec<f32>,
        cols: usize,
    },
    Packed {
        qtype: u32,
        cols: usize,
        src: Bytes,
        start: usize,
    },
}

/// Where a packed weight's bytes live: in the mapped file, or in a buffer this process
/// read them into and holds.
///
/// The distinction is the difference between "the kernel may evict this at any moment"
/// and "this is ours until we drop it". On a checkpoint larger than RAM the kernel's
/// LRU is not a good judge — every block is read exactly once per token, so there is no
/// recency to exploit, and what it evicts is whatever it happens to have. Pinning a
/// chosen subset takes that decision away from it: those blocks never fault again, and
/// the ones left mapped are streamed by [`crate::prefetch::LayerPrefetcher`].
#[derive(Clone)]
pub(crate) enum Bytes {
    Mapped(Arc<Mmap>),
    Owned(Arc<Vec<u8>>),
}

impl Bytes {
    pub(crate) fn as_slice(&self) -> &[u8] {
        match self {
            Bytes::Mapped(m) => &m[..],
            Bytes::Owned(v) => &v[..],
        }
    }

    fn len(&self) -> usize {
        match self {
            Bytes::Mapped(m) => m.len(),
            Bytes::Owned(v) => v.len(),
        }
    }
}

impl Weight {
    /// `out[r] = dot(row r, x)` over the whole matrix.
    pub(crate) fn matvec(&self, x: &[f32], out: &mut [f32]) -> Result<(), GarudaError> {
        self.matvec_rows(0, x, out)
    }

    /// `out[i] = dot(row (row_start + i), x)`, i.e. a matvec over the `out.len()` rows
    /// starting at `row_start`. Used to view one expert's slice of a stacked 3D expert
    /// tensor without copying it out.
    fn matvec_rows(&self, row_start: usize, x: &[f32], out: &mut [f32]) -> Result<(), GarudaError> {
        let n = out.len();
        match self {
            Weight::Full { data, cols } => {
                let off = row_start * cols;
                simd::matvec(&data[off..off + n * cols], n, *cols, x, out);
                Ok(())
            }
            Weight::Packed {
                qtype,
                cols,
                src,
                start,
            } => {
                let row_bytes = quant::byte_size(*qtype, *cols)?;
                let off = start + row_start * row_bytes;
                quant::matvec(
                    *qtype,
                    &src.as_slice()[off..off + n * row_bytes],
                    n,
                    *cols,
                    x,
                    out,
                )
            }
        }
    }

    /// `out[b*n + i] = dot(row (row_start + i), xs[b])` over `n` vectors at once,
    /// where `n = xs.len() / cols`. The batched twin of [`Self::matvec_rows`].
    pub(crate) fn matmul_rows(
        &self,
        row_start: usize,
        xs: &[f32],
        n: usize,
        out: &mut [f32],
    ) -> Result<(), GarudaError> {
        let rows = out.len() / n.max(1);
        match self {
            Weight::Full { data, cols } => {
                let off = row_start * cols;
                simd::matmul(&data[off..off + rows * cols], rows, *cols, xs, n, out);
                Ok(())
            }
            Weight::Packed {
                qtype,
                cols,
                src,
                start,
            } => {
                let row_bytes = quant::byte_size(*qtype, *cols)?;
                let off = start + row_start * row_bytes;
                quant::matmul(
                    *qtype,
                    &src.as_slice()[off..off + rows * row_bytes],
                    rows,
                    *cols,
                    xs,
                    n,
                    out,
                )
            }
        }
    }

    /// Dequantise a single row (e.g. one embedding).
    pub(crate) fn row(&self, r: usize) -> Result<Vec<f32>, GarudaError> {
        match self {
            Weight::Full { data, cols } => Ok(data[r * cols..(r + 1) * cols].to_vec()),
            Weight::Packed {
                qtype,
                cols,
                src,
                start,
            } => {
                let row_bytes = quant::byte_size(*qtype, *cols)?;
                let off = start + r * row_bytes;
                quant::dequantize(*qtype, &src.as_slice()[off..off + row_bytes], *cols)
            }
        }
    }
}

/// One expert's slot within a gate/up/down projection: either a row-slice of a
/// single stacked tensor, or its own separate tensor.
///
/// GGUF checkpoints disagree on layout. Newer llama.cpp conversions merge all
/// experts into one `..._exps` tensor (`Stacked`, expert `e` is rows
/// `[e·block, (e+1)·block)`). Older conversions — including the original
/// TheBloke Mixtral quantisations — give each expert its own tensor, e.g.
/// `blk.0.ffn_gate.3.weight` (`Split`).
enum ExpertWeight {
    Stacked(Weight),
    Split(Vec<Weight>),
}

impl ExpertWeight {
    /// Expert `e` applied to `n` activation vectors at once — the batched twin of
    /// [`Self::matvec_expert`]. This is where an MoE layer stops paying for its
    /// weights once per token: the expert's rows are decoded once for every token
    /// that routed to it.
    fn matmul_expert(
        &self,
        e: usize,
        block: usize,
        xs: &[f32],
        n: usize,
        out: &mut [f32],
    ) -> Result<(), GarudaError> {
        match self {
            ExpertWeight::Stacked(w) => w.matmul_rows(e * block, xs, n, out),
            ExpertWeight::Split(ws) => ws[e].matmul_rows(0, xs, n, out),
        }
    }

    /// Byte range of expert `e`'s packed weight in the backing mmap, for
    /// prefetching. `block` is one expert's row count, as in [`Self::matvec_expert`].
    /// `None` if this weight is not packed (mmap is off) or `e` is out of range.
    fn byte_range(&self, e: usize, block: usize) -> Option<(usize, usize)> {
        match self {
            ExpertWeight::Stacked(Weight::Packed {
                qtype, cols, start, ..
            }) => {
                let row_bytes = quant::byte_size(*qtype, *cols).ok()?;
                Some((start + e * block * row_bytes, block * row_bytes))
            }
            ExpertWeight::Split(ws) => match ws.get(e) {
                Some(Weight::Packed {
                    qtype, cols, start, ..
                }) => {
                    let len = quant::byte_size(*qtype, block * *cols).ok()?;
                    Some((*start, len))
                }
                _ => None,
            },
            ExpertWeight::Stacked(Weight::Full { .. }) => None,
        }
    }
}

/// Where each token in a batch appends its keys and values.
///
/// Prefill is a batch of prompt tokens sharing one sequence's cache, appended in
/// order; a batched decode step is one token from each of several sequences, each
/// with its own. The layer code is identical either way, so it takes this instead of
/// a cache.
enum KvTargets<'a, 'b> {
    Shared(&'a mut KVCacheState),
    // Two lifetimes: `&mut` is invariant, so tying the slice and its contents to one
    // would force every caller's borrows to have identical extent.
    PerToken(&'a mut [&'b mut KVCacheState]),
}

impl KvTargets<'_, '_> {
    fn get(&mut self, token: usize) -> &mut KVCacheState {
        match self {
            KvTargets::Shared(kv) => kv,
            KvTargets::PerToken(kvs) => kvs[token],
        }
    }
}

/// A small f32 tensor — a norm weight or a bias — expanded whatever the file holds.
///
/// These are tiny and read on every token, so there is nothing to gain from keeping
/// them packed and a per-row dequantisation to lose.
pub(crate) fn load_norm(
    g: &Gguf,
    bytes: &[u8],
    name: &str,
    n: usize,
) -> Result<Vec<f32>, GarudaError> {
    let data = g.tensor_f32(bytes, name)?;
    if data.len() != n {
        return Err(GarudaError::Model(format!(
            "tensor '{name}' has {} values, expected {n}",
            data.len()
        )));
    }
    Ok(data)
}

/// A packed `rows x cols` weight matrix inside `src`, which holds the file's bytes
/// from offset `base` onwards.
///
/// `base` is zero for a whole-file map and the block's own offset for a buffer holding
/// one block — see [`Bytes`] for why a caller would read a block into one.
pub(crate) fn pinned_weight(
    g: &Gguf,
    src: &Bytes,
    base: usize,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Weight, GarudaError> {
    let t = g
        .tensor(name)
        .ok_or_else(|| GarudaError::Model(format!("tensor '{name}' not found")))?;
    if t.n_elements() as usize != rows * cols {
        return Err(GarudaError::Model(format!(
            "tensor '{name}' has {} elements, expected {}",
            t.n_elements(),
            rows * cols
        )));
    }
    let len = quant::byte_size(t.ggml_type, rows * cols)?;
    let start = (g.data_offset + t.offset as usize)
        .checked_sub(base)
        .ok_or_else(|| GarudaError::Model(format!("tensor '{name}' starts before its buffer")))?;
    if start + len > src.len() {
        return Err(GarudaError::Model(format!(
            "tensor '{name}' runs past the end of its buffer"
        )));
    }
    Ok(Weight::Packed {
        qtype: t.ggml_type,
        cols,
        src: src.clone(),
        start,
    })
}

/// A `rows x cols` weight matrix from a GGUF file: packed in the memory map when
/// `mmap` is `Some`, expanded to `f32` in RAM when it is `None`.
///
/// Shared by every architecture this crate loads — see [`Weight`].
pub(crate) fn load_weight(
    g: &Gguf,
    bytes: &[u8],
    mmap: &Option<Arc<Mmap>>,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Weight, GarudaError> {
    let t = g
        .tensor(name)
        .ok_or_else(|| GarudaError::Model(format!("tensor '{name}' not found")))?;
    if t.n_elements() as usize != rows * cols {
        return Err(GarudaError::Model(format!(
            "tensor '{name}' has {} elements, expected {}",
            t.n_elements(),
            rows * cols
        )));
    }
    match mmap {
        Some(src) => pinned_weight(g, &Bytes::Mapped(src.clone()), 0, name, rows, cols),
        None => Ok(Weight::Full {
            data: g.tensor_f32(bytes, name)?,
            cols,
        }),
    }
}

/// The three matrices a SwiGLU expert is made of. They are always used together, and
/// always for the same expert index, so they travel as one.
#[derive(Clone, Copy)]
struct Swiglu<'a> {
    gate: &'a ExpertWeight,
    up: &'a ExpertWeight,
    down: &'a ExpertWeight,
}

/// The architecture parameters read from GGUF metadata.
#[derive(Debug, Clone, Copy)]
pub struct LlamaConfig {
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub d_ff: usize,
    pub vocab: usize,
    pub context: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    /// Number of experts in a mixture-of-experts FFN. `0` means a dense FFN.
    pub n_experts: usize,
    /// Experts activated per token (top-k). Unused when `n_experts == 0`.
    pub n_experts_used: usize,
}

impl LlamaConfig {
    fn from_gguf(g: &Gguf) -> Result<Self, GarudaError> {
        if g.architecture() != Some("llama") {
            return Err(GarudaError::Model(format!(
                "architecture '{}' is not supported (this runtime loads llama and qwen35)",
                g.architecture().unwrap_or("unknown")
            )));
        }
        let need = |suffix: &str| {
            g.arch_u64(suffix)
                .ok_or_else(|| GarudaError::Model(format!("gguf is missing llama.{suffix}")))
        };

        let d_model = need("embedding_length")? as usize;
        let n_heads = need("attention.head_count")? as usize;
        let n_kv_heads = g
            .arch_u64("attention.head_count_kv")
            .unwrap_or(n_heads as u64) as usize;
        let n_layers = need("block_count")? as usize;
        let d_ff = need("feed_forward_length")? as usize;

        if n_heads == 0 || n_kv_heads == 0 || d_model % n_heads != 0 {
            return Err(GarudaError::Model(format!(
                "inconsistent head configuration: d_model={d_model}, heads={n_heads}, kv_heads={n_kv_heads}"
            )));
        }
        if n_heads % n_kv_heads != 0 {
            return Err(GarudaError::Model(format!(
                "head_count {n_heads} is not a multiple of head_count_kv {n_kv_heads}"
            )));
        }

        Ok(Self {
            d_model,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim: d_model / n_heads,
            d_ff,
            vocab: g
                .get("tokenizer.ggml.tokens")
                .and_then(crate::gguf::Value::as_array)
                .map(|a| a.len())
                .ok_or_else(|| GarudaError::Model("gguf has no token list".into()))?,
            context: g.arch_u64("context_length").unwrap_or(2048) as usize,
            rms_eps: g
                .arch_f32("attention.layer_norm_rms_epsilon")
                .unwrap_or(1e-5),
            rope_theta: g.arch_f32("rope.freq_base").unwrap_or(10_000.0),
            n_experts: g.arch_u64("expert_count").unwrap_or(0) as usize,
            n_experts_used: g.arch_u64("expert_used_count").unwrap_or(0) as usize,
        })
    }

    /// Width of one stored key/value vector under grouped-query attention.
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    /// The runtime-facing shape. `n_experts`/`top_k` are unused by a dense model but
    /// must satisfy [`ModelDims::validate`], so they are set to the trivial 1/1.
    pub fn model_dims(&self) -> ModelDims {
        ModelDims {
            d_model: self.d_model,
            n_heads: self.n_heads,
            head_dim: self.head_dim,
            d_ff: self.d_ff,
            n_experts: 1,
            top_k: 1,
            vocab_size: self.vocab,
            block_size: 32,
            rope_theta: self.rope_theta,
        }
    }
}

/// A block's feed-forward network: either a single dense SwiGLU, or a mixture of
/// experts where a router picks the top-k experts to run for each token.
///
/// For MoE, `gate`/`up`/`down` hold each expert's matrix, stacked or split (see
/// [`ExpertWeight`]). Under `mmap`, only the selected experts' rows are ever paged
/// in — the streaming win.
enum Ffn {
    Dense {
        gate: ExpertWeight,
        up: ExpertWeight,
        down: ExpertWeight,
    },
    Moe {
        router: Weight,
        gate: ExpertWeight,
        up: ExpertWeight,
        down: ExpertWeight,
    },
}

/// One transformer block's weights. Norms are small and always `f32`; the projection
/// matrices may be packed.
struct Layer {
    attn_norm: Vec<f32>,
    wq: Weight,
    wk: Weight,
    wv: Weight,
    wo: Weight,
    ffn_norm: Vec<f32>,
    ffn: Ffn,
}

pub struct LlamaBackend {
    cfg: LlamaConfig,
    /// `Arc` because a checkpoint with tied embeddings uses this same matrix as its
    /// output head. Without the sharing, the non-mmap path dequantised
    /// `token_embd.weight` twice into two independent `f32` buffers — over a
    /// gigabyte of duplication on a tied model with a large vocabulary.
    token_embd: Arc<Weight>,
    output_norm: Vec<f32>,
    output: Arc<Weight>,
    layers: Vec<Layer>,
    prefetch: Option<Arc<crate::prefetch::PrefetchEngine>>,
    /// Prompt tokens pushed through one layer before moving to the next; `1` is
    /// token-major. Set via [`Self::with_prefill_chunk`], which documents the
    /// tradeoff.
    prefill_chunk: usize,
}

impl LlamaBackend {
    /// Load a checkpoint from a GGUF file's bytes, expanding weights to `f32` in RAM.
    pub fn load(bytes: &[u8]) -> Result<Self, GarudaError> {
        let g = Gguf::parse(bytes)?;
        Self::from_gguf(&g, bytes, None)
    }

    /// Load from an already-parsed GGUF header plus the file bytes.
    ///
    /// When `mmap` is `Some`, the projection matrices are kept packed in the mapped
    /// file and dequantised per row at inference time (low RAM, slower). When `None`,
    /// every weight is expanded to `f32` in RAM (more RAM, faster). `bytes` must be the
    /// same data the mmap covers.
    pub fn from_gguf(g: &Gguf, bytes: &[u8], mmap: Option<Arc<Mmap>>) -> Result<Self, GarudaError> {
        let cfg = LlamaConfig::from_gguf(g)?;
        let (d, f, v, hk) = (cfg.d_model, cfg.d_ff, cfg.vocab, cfg.kv_dim());

        // A small f32 tensor (norm), always expanded.
        let norm = |name: &str, n: usize| load_norm(g, bytes, name, n);

        // A weight matrix: packed if mmapping, otherwise expanded to f32.
        let weight =
            |name: &str, rows: usize, cols: usize| load_weight(g, bytes, &mmap, name, rows, cols);

        let token_embd = Arc::new(weight("token_embd.weight", v, d)?);
        let output_norm = norm("output_norm.weight", d)?;
        // Some checkpoints tie the output head to the embeddings and omit `output`.
        // Share the one already loaded rather than decoding the same tensor again:
        // under mmap both would merely point into the same map, but on the f32 path a
        // second `Weight::Full` is a second full copy of a `vocab x d_model` matrix.
        let output = if g.tensor("output.weight").is_some() {
            Arc::new(weight("output.weight", v, d)?)
        } else {
            token_embd.clone()
        };

        let ne = cfg.n_experts;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            let p = |name: &str| format!("blk.{l}.{name}.weight");
            // Per-expert tensor name for the split (un-merged) layout, e.g.
            // `blk.0.ffn_gate.3.weight`.
            let pe = |name: &str, e: usize| format!("blk.{l}.{name}.{e}.weight");
            let split =
                |name: &str, rows: usize, cols: usize| -> Result<ExpertWeight, GarudaError> {
                    let mut ws = Vec::with_capacity(ne);
                    for e in 0..ne {
                        ws.push(weight(&pe(name, e), rows, cols)?);
                    }
                    Ok(ExpertWeight::Split(ws))
                };

            // A layer is MoE if the model declares experts and the block has either
            // the merged stacked tensors or the older per-expert tensors; otherwise
            // it is a plain dense FFN.
            let ffn = if ne > 0 && g.tensor(&p("ffn_gate_exps")).is_some() {
                Ffn::Moe {
                    router: weight(&p("ffn_gate_inp"), ne, d)?,
                    gate: ExpertWeight::Stacked(weight(&p("ffn_gate_exps"), ne * f, d)?),
                    up: ExpertWeight::Stacked(weight(&p("ffn_up_exps"), ne * f, d)?),
                    down: ExpertWeight::Stacked(weight(&p("ffn_down_exps"), ne * d, f)?),
                }
            } else if ne > 0 && g.tensor(&pe("ffn_gate", 0)).is_some() {
                Ffn::Moe {
                    router: weight(&p("ffn_gate_inp"), ne, d)?,
                    gate: split("ffn_gate", f, d)?,
                    up: split("ffn_up", f, d)?,
                    down: split("ffn_down", d, f)?,
                }
            } else {
                Ffn::Dense {
                    gate: ExpertWeight::Stacked(weight(&p("ffn_gate"), f, d)?),
                    up: ExpertWeight::Stacked(weight(&p("ffn_up"), f, d)?),
                    down: ExpertWeight::Stacked(weight(&p("ffn_down"), d, f)?),
                }
            };

            layers.push(Layer {
                attn_norm: norm(&p("attn_norm"), d)?,
                wq: weight(&p("attn_q"), d, d)?,
                wk: weight(&p("attn_k"), hk, d)?,
                wv: weight(&p("attn_v"), hk, d)?,
                wo: weight(&p("attn_output"), d, d)?,
                ffn_norm: norm(&p("ffn_norm"), d)?,
                ffn,
            });
        }

        Ok(Self {
            cfg,
            token_embd,
            output_norm,
            output,
            layers,
            prefetch: None,
            // Token-major unless something tells us the checkpoint is too big to
            // cache; `Engine::build` makes that call.
            prefill_chunk: 1,
        })
    }

    /// True when weights are kept packed in a memory-mapped file.
    pub fn is_mmapped(&self) -> bool {
        matches!(*self.token_embd, Weight::Packed { .. })
    }

    /// The backing memory map, if this checkpoint was loaded with `mmap`.
    pub fn mmap(&self) -> Option<Arc<Mmap>> {
        match &*self.token_embd {
            Weight::Packed {
                src: Bytes::Mapped(m),
                ..
            } => Some(m.clone()),
            _ => None,
        }
    }

    /// True when the output head is the embedding matrix rather than its own tensor.
    pub fn has_tied_embeddings(&self) -> bool {
        Arc::ptr_eq(&self.token_embd, &self.output)
    }

    /// Byte ranges in the backing mmap for every `(layer, expert)` pair's gate/up/
    /// down weights, indexed `[layer * n_experts + expert]` — what a
    /// [`crate::prefetch::PrefetchEngine`] should warm ahead of a token routing
    /// there. Empty for a dense model or a backend that is not mmapped.
    pub fn expert_page_ranges(&self) -> Vec<Vec<(usize, usize)>> {
        let ne = self.cfg.n_experts;
        if ne == 0 {
            return Vec::new();
        }
        let (d, f) = (self.cfg.d_model, self.cfg.d_ff);
        let mut out = vec![Vec::new(); self.cfg.n_layers * ne];
        for (l, layer) in self.layers.iter().enumerate() {
            let Ffn::Moe { gate, up, down, .. } = &layer.ffn else {
                continue;
            };
            for (e, slot) in out[l * ne..(l + 1) * ne].iter_mut().enumerate() {
                slot.extend(gate.byte_range(e, f));
                slot.extend(up.byte_range(e, f));
                slot.extend(down.byte_range(e, d));
            }
        }
        out
    }

    /// Set how many prompt tokens share one pass over a layer's weights. `0` and
    /// `1` both mean token-major: a token traverses every layer before the next one
    /// starts, which is what this did before batching existed.
    ///
    /// Batching prefill saves work at two levels at once.
    ///
    /// **Page cache.** Token-major re-reads every layer once per prompt token, so
    /// the working set is the whole model *per token* — 7.1 GB for Mixtral-8x7B
    /// Q4_K_M, against 816 MB for a single layer with all eight experts. Above what
    /// the page cache holds, that is the difference between streaming the model off
    /// disk once per token and once per chunk.
    ///
    /// **Row decode.** Within a layer the tokens are grouped by the expert they
    /// routed to, so an expert's packed rows are decoded once for every token that
    /// chose it rather than once per token. Under `mmap` that decode is the dominant
    /// cost of the forward pass, and it is why batching wins even when the whole
    /// model already fits in RAM.
    ///
    /// Measured on a 620 MB MoE checkpoint resident in RAM, 128-token prefill,
    /// warm cache, best of three — the case that has *nothing* to gain from the page
    /// cache, so this is the decode saving alone:
    ///
    /// | | token-major | batched (256) |
    /// |---|---|---|
    /// | mmapped (packed) | 2.68 s | 1.34 s (2.0×) |
    /// | expanded to `f32` | 2.12 s | 1.28 s (1.7×) |
    ///
    /// Either order computes the same thing, and the tests assert it: each token's
    /// hidden state flows through the layers independently, and the only coupling —
    /// layer `l`'s attention reading layer `l`'s KV — is appended in token order
    /// either way. Attention itself stays strictly sequential for that reason; only
    /// the feed-forward is batched.
    pub fn with_prefill_chunk(mut self, chunk: usize) -> Self {
        self.prefill_chunk = chunk.max(1);
        self
    }

    /// Attach a prefetch engine, so each MoE layer's routing decision warms the
    /// likely next-step experts' pages while the rest of this step still computes.
    pub fn with_prefetch(mut self, prefetch: Arc<crate::prefetch::PrefetchEngine>) -> Self {
        self.prefetch = Some(prefetch);
        self
    }

    pub fn config(&self) -> LlamaConfig {
        self.cfg
    }

    /// RMSNorm followed by an elementwise scale, as Llama applies it.
    fn norm(&self, x: &[f32], weight: &[f32]) -> Vec<f32> {
        let mut h = x.to_vec();
        simd::rmsnorm(&mut h, self.cfg.rms_eps);
        simd::mul_assign(&mut h, weight);
        h
    }

    /// Q, K and V for a whole batch — three batched matmuls instead of `3n` matvecs,
    /// so each projection's rows are read and decoded once for every token.
    #[allow(clippy::type_complexity)]
    fn project_qkv(
        &self,
        layer: &Layer,
        hs: &[f32],
        n: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), GarudaError> {
        let (d, kv_dim) = (self.cfg.d_model, self.cfg.kv_dim());
        let mut q = vec![0.0; n * d];
        let mut k = vec![0.0; n * kv_dim];
        let mut v = vec![0.0; n * kv_dim];
        layer.wq.matmul_rows(0, hs, n, &mut q)?;
        layer.wk.matmul_rows(0, hs, n, &mut k)?;
        layer.wv.matmul_rows(0, hs, n, &mut v)?;
        Ok((q, k, v))
    }

    /// One token's grouped-query causal attention, given its already-projected
    /// `q`/`k`/`v`: rotate by position, append to `kv`, and read out the context.
    ///
    /// This is the part that cannot be batched away. Attention is causal, so a
    /// token's keys and values must be in the cache before anything attends to them,
    /// and `pos` is read from the cache itself — which is what keeps a batch of
    /// prompt tokens sharing one cache in the right order.
    fn attend_one(
        &self,
        q: &mut [f32],
        k: &mut [f32],
        v: &[f32],
        kv: &mut KVCacheState,
        context: &mut [f32],
    ) -> Result<(), GarudaError> {
        let LlamaConfig {
            n_heads,
            n_kv_heads,
            head_dim: hd,
            rope_theta,
            ..
        } = self.cfg;
        let group = n_heads / n_kv_heads;

        let pos = kv.len();
        for hh in 0..n_heads {
            simd::rope(&mut q[hh * hd..(hh + 1) * hd], pos, rope_theta);
        }
        for hh in 0..n_kv_heads {
            simd::rope(&mut k[hh * hd..(hh + 1) * hd], pos, rope_theta);
        }

        kv.append(k, v)?;

        let start = kv.attention_start();
        let end = pos + 1;
        kv.ensure_resident(start, end)?;

        let scale = 1.0 / (hd as f32).sqrt();
        for hh in 0..n_heads {
            let q_h = &q[hh * hd..(hh + 1) * hd];
            let kv_head = hh / group; // GQA: several query heads share a kv head
            let kr = kv_head * hd..(kv_head + 1) * hd;

            let mut scores = Vec::with_capacity(end - start);
            for j in start..end {
                let key = kv
                    .key_at(j)
                    .ok_or_else(|| GarudaError::Cache(format!("missing key at {j}")))?;
                scores.push(simd::dot(q_h, &key[kr.clone()]) * scale);
            }
            simd::softmax(&mut scores);

            let out_h = &mut context[hh * hd..(hh + 1) * hd];
            for (j, &p) in (start..end).zip(scores.iter()) {
                let val = kv
                    .value_at(j)
                    .ok_or_else(|| GarudaError::Cache(format!("missing value at {j}")))?;
                simd::add_scaled(out_h, &val[kr.clone()], p);
            }
        }
        Ok(())
    }

    /// One layer over a whole batch of tokens.
    ///
    /// Attention stays strictly token-by-token: it is causal, and each token's keys
    /// and values have to be in the cache before the next one attends to them. The
    /// feed-forward has no such ordering — a token's FFN depends only on its own
    /// hidden state — so the whole batch goes through it at once, which is what lets
    /// [`Self::feed_forward_batch`] group tokens by expert.
    fn layer_batch(
        &self,
        l: usize,
        xs: &mut [Vec<f32>],
        kv: &mut KVCacheState,
    ) -> Result<(), GarudaError> {
        self.layer_batch_over(l, xs, KvTargets::Shared(kv))
    }

    /// The same layer over a batch whose tokens each belong to a *different*
    /// sequence — one decode step across concurrent requests. Everything except the
    /// attention read is shared, so N sequences cost one pass over the weights.
    fn layer_batch_multi(
        &self,
        l: usize,
        xs: &mut [Vec<f32>],
        kvs: &mut [&mut KVCacheState],
    ) -> Result<(), GarudaError> {
        self.layer_batch_over(l, xs, KvTargets::PerToken(kvs))
    }

    fn layer_batch_over(
        &self,
        l: usize,
        xs: &mut [Vec<f32>],
        mut kvs: KvTargets<'_, '_>,
    ) -> Result<(), GarudaError> {
        let layer = &self.layers[l];
        let (d, kv_dim, n) = (self.cfg.d_model, self.cfg.kv_dim(), xs.len());

        // Every token's attention input depends only on its own layer input, so the
        // whole batch can be normed and projected before any of it attends.
        let mut hs = Vec::with_capacity(n * d);
        for x in xs.iter() {
            hs.extend_from_slice(&self.norm(x, &layer.attn_norm));
        }
        let (mut q, mut k, v) = self.project_qkv(layer, &hs, n)?;

        let mut context = vec![0.0; n * d];
        for i in 0..n {
            self.attend_one(
                &mut q[i * d..(i + 1) * d],
                &mut k[i * kv_dim..(i + 1) * kv_dim],
                &v[i * kv_dim..(i + 1) * kv_dim],
                kvs.get(i),
                &mut context[i * d..(i + 1) * d],
            )?;
        }

        let mut attn_out = vec![0.0; n * d];
        layer.wo.matmul_rows(0, &context, n, &mut attn_out)?;

        let mut hs_ffn = Vec::with_capacity(n * d);
        for (i, x) in xs.iter_mut().enumerate() {
            simd::add_assign(x, &attn_out[i * d..(i + 1) * d]);
            hs_ffn.extend_from_slice(&self.norm(x, &layer.ffn_norm));
        }

        let ffn = self.feed_forward_batch(l, layer, &hs_ffn, n, &mut kvs)?;
        for (i, x) in xs.iter_mut().enumerate() {
            simd::add_assign(x, &ffn[i * d..(i + 1) * d]);
        }
        Ok(())
    }

    /// The block's feed-forward over `n` tokens, `hs` being their `n * d_model`
    /// hidden states.
    ///
    /// For MoE this is the whole point of batching. Run token by token, an expert's
    /// gate/up/down matrices are read — and, when packed, decoded — once per token
    /// that routes to it. Gathered into one call per expert, they are read once for
    /// all of them: with 256 tokens choosing 2 of 8 experts, each expert serves ~64
    /// tokens per pass instead of one.
    fn feed_forward_batch(
        &self,
        l: usize,
        layer: &Layer,
        hs: &[f32],
        n: usize,
        kvs: &mut KvTargets<'_, '_>,
    ) -> Result<Vec<f32>, GarudaError> {
        let d = self.cfg.d_model;

        let (router, w) = match &layer.ffn {
            Ffn::Dense { gate, up, down } => {
                // A dense FFN is one expert spanning the whole tensor: every token
                // goes through it, so the batch needs no grouping at all.
                let mut out = vec![0.0; n * d];
                let w = Swiglu { gate, up, down };
                self.expert_batch(w, 0, hs, n, &mut out)?;
                return Ok(out);
            }
            Ffn::Moe {
                router,
                gate,
                up,
                down,
            } => (router, Swiglu { gate, up, down }),
        };

        let (ne, k) = (self.cfg.n_experts, self.cfg.n_experts_used.max(1));

        // Route every token, in order. The order matters only for the predictor: its
        // model is a token-to-token transition, so it has to see the same sequence of
        // steps a token-at-a-time run would have produced.
        let mut picks: Vec<Vec<(usize, f32)>> = Vec::with_capacity(n);
        for t in 0..n {
            let h = &hs[t * d..(t + 1) * d];
            let mut scores = vec![0.0; ne];
            router.matvec(h, &mut scores)?;
            simd::softmax(&mut scores);

            let mut ranked: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
            ranked.truncate(k);
            let sum: f32 = ranked.iter().map(|(_, w)| w).sum();
            let norm = if sum > 0.0 { 1.0 / sum } else { 1.0 / k as f32 };
            for (_, w) in ranked.iter_mut() {
                *w *= norm;
            }

            if let Some(pf) = &self.prefetch {
                // Routing history is per sequence *and* per layer: the predictor's
                // model is a token-to-token transition, so a batch drawn from
                // different sequences must not be chained together.
                let kv = kvs.get(t);
                let base = (l * ne) as ExpertId;
                let used: Vec<ExpertId> =
                    ranked.iter().map(|&(e, _)| base + e as ExpertId).collect();
                let predicted = pf.observe_step(&kv.last_experts, &used, &kv.last_predicted);
                kv.last_predicted = predicted;
                kv.last_experts = used;
            }
            picks.push(ranked);
        }

        // Invert the routing: which tokens each expert owes an answer to.
        let mut members: Vec<Vec<(usize, f32)>> = vec![Vec::new(); ne];
        for (t, ranked) in picks.iter().enumerate() {
            for &(e, w) in ranked {
                members[e].push((t, w));
            }
        }

        let mut out = vec![0.0; n * d];
        let mut gathered = Vec::new();
        let mut expert_out = Vec::new();
        for (e, tokens) in members.iter().enumerate() {
            if tokens.is_empty() {
                continue;
            }
            gathered.clear();
            for &(t, _) in tokens {
                gathered.extend_from_slice(&hs[t * d..(t + 1) * d]);
            }
            expert_out.clear();
            expert_out.resize(tokens.len() * d, 0.0);
            self.expert_batch(w, e, &gathered, tokens.len(), &mut expert_out)?;

            for (i, &(t, w)) in tokens.iter().enumerate() {
                simd::add_scaled(
                    &mut out[t * d..(t + 1) * d],
                    &expert_out[i * d..(i + 1) * d],
                    w,
                );
            }
        }
        Ok(out)
    }

    /// `down_e(silu(gate_e·h) ⊙ (up_e·h))` for `n` hidden states at once, the batched
    /// twin of [`Self::expert`]. `out` is `n * d_model`, vector-major.
    fn expert_batch(
        &self,
        w: Swiglu<'_>,
        e: usize,
        hs: &[f32],
        n: usize,
        out: &mut [f32],
    ) -> Result<(), GarudaError> {
        let (d, f) = (self.cfg.d_model, self.cfg.d_ff);
        let mut g = vec![0.0; n * f];
        let mut u = vec![0.0; n * f];
        w.gate.matmul_expert(e, f, hs, n, &mut g)?;
        w.up.matmul_expert(e, f, hs, n, &mut u)?;
        // SwiGLU is elementwise, so the whole batch is one pass.
        simd::silu(&mut g);
        simd::mul_assign(&mut g, &u);
        debug_assert_eq!(out.len(), n * d);
        w.down.matmul_expert(e, d, &g, n, out)
    }

    /// The last `n_last` positions' final hidden states, from one pass.
    ///
    /// Every new token's state is computed anyway; `hidden` simply discards all but
    /// the last. Keeping the tail is what lets a run of speculated tokens be checked
    /// without paying for a second pass over the weights.
    fn forward_tail(
        &self,
        context: &[Token],
        seq: &mut SeqState,
        n_last: usize,
    ) -> Result<Vec<Vec<f32>>, GarudaError> {
        if context.is_empty() {
            return Err(GarudaError::Inference("empty context".into()));
        }
        if seq.n_layers() != self.cfg.n_layers {
            return Err(GarudaError::Inference(format!(
                "sequence has {} kv layers but the model has {}",
                seq.n_layers(),
                self.cfg.n_layers
            )));
        }
        let already = seq.len();
        if already > context.len() {
            return Err(GarudaError::Inference(
                "sequence state is ahead of the context".into(),
            ));
        }

        let d = self.cfg.d_model;
        let new = &context[already..];
        if new.is_empty() {
            return Err(GarudaError::Inference(
                "no new tokens to process for this context".into(),
            ));
        }

        // Refuse an over-long prefill before touching anything. Discovering it
        // partway through would leave the layers at different lengths, and
        // layer-major makes that gap a whole chunk wide rather than one position.
        let capacity = seq.max_positions();
        if already + new.len() > capacity {
            return Err(GarudaError::Cache(format!(
                "{} tokens do not fit the {capacity}-position context window ({already} used)",
                new.len()
            )));
        }

        // A rolling window of the most recent hidden states, so `logits_multi` can
        // answer for more than the last position without a second pass. It rolls
        // across chunks rather than resetting per chunk: the prefill chunk size is
        // about weight locality and has nothing to do with how many positions the
        // caller asked about.
        let mut tail: Vec<Vec<f32>> = Vec::new();
        for chunk in new.chunks(self.prefill_chunk.max(1)) {
            // Embed the chunk first, so an out-of-vocabulary token is rejected before
            // any layer has run rather than after the tokens ahead of it.
            let mut xs: Vec<Vec<f32>> = Vec::with_capacity(chunk.len());
            for &token in chunk {
                let idx = token as usize;
                if idx >= self.cfg.vocab {
                    return Err(GarudaError::InvalidToken(token));
                }
                xs.push(self.token_embd.row(idx)?);
            }

            // Layer-major: one layer, every token in the chunk, then the next layer.
            // Each token's hidden state flows through the layers independently, and
            // the only coupling — layer `l`'s attention reading layer `l`'s KV — is
            // built in token order either way, so this is the same arithmetic in the
            // same order. What changes is which weights are hot: see PREFILL_CHUNK.
            for l in 0..self.cfg.n_layers {
                let kv = seq.layer(l);
                self.layer_batch(l, &mut xs, kv)?;
            }
            tail.append(&mut xs);
            if tail.len() > n_last {
                tail.drain(..tail.len() - n_last);
            }
        }

        if n_last > tail.len() {
            return Err(GarudaError::Inference(format!(
                "asked for the last {n_last} positions but only {} were computed in \
                 this pass",
                tail.len()
            )));
        }
        tail.drain(..tail.len() - n_last);
        for x in tail.iter_mut() {
            simd::rmsnorm(x, self.cfg.rms_eps);
            simd::mul_assign(x, &self.output_norm);
        }
        let _ = d;
        Ok(tail)
    }
}

impl InferenceBackend for LlamaBackend {
    fn dims(&self) -> ModelDims {
        self.cfg.model_dims()
    }

    /// One decode step across several sequences, sharing one pass over the weights.
    ///
    /// Only the attention read is per sequence; the projections, the router and the
    /// experts all see the batch at once. For a checkpoint larger than RAM that is
    /// the difference between streaming the model once per token and once per batch
    /// of tokens — a single sequence decoding alone has no way to amortise it.
    ///
    /// Falls back to one call each unless every sequence contributes exactly one new
    /// token. That is the decode case, and it is the only one where the sequences do
    /// equal work; a mixed batch would leave most of them idle behind the longest.
    fn logits_batch(
        &self,
        contexts: &[&[Token]],
        seqs: &mut [&mut SeqState],
    ) -> Result<Vec<Tensor>, GarudaError> {
        if contexts.len() != seqs.len() {
            return Err(GarudaError::Inference(format!(
                "logits_batch got {} contexts for {} sequences",
                contexts.len(),
                seqs.len()
            )));
        }
        let n = contexts.len();
        let one_step_each = n > 1
            && contexts
                .iter()
                .zip(seqs.iter())
                .all(|(c, s)| c.len() == s.len() + 1 && s.n_layers() == self.cfg.n_layers);
        if !one_step_each {
            return contexts
                .iter()
                .zip(seqs.iter_mut())
                .map(|(c, s)| self.logits(c, s))
                .collect();
        }

        // Reject before touching any cache, so a batch cannot be left half-advanced.
        for (c, s) in contexts.iter().zip(seqs.iter()) {
            let token = *c.last().expect("checked non-empty by the length test");
            if token as usize >= self.cfg.vocab {
                return Err(GarudaError::InvalidToken(token));
            }
            if s.len() + 1 > s.max_positions() {
                return Err(GarudaError::Cache(format!(
                    "context window of {} positions is exhausted",
                    s.max_positions()
                )));
            }
        }

        let (d, vocab) = (self.cfg.d_model, self.cfg.vocab);
        let mut xs: Vec<Vec<f32>> = Vec::with_capacity(n);
        for c in contexts {
            xs.push(
                self.token_embd
                    .row(*c.last().expect("non-empty") as usize)?,
            );
        }

        for l in 0..self.cfg.n_layers {
            // Each sequence's own cache for this layer; distinct elements, so the
            // borrows do not overlap.
            let mut kvs: Vec<&mut KVCacheState> = seqs.iter_mut().map(|s| s.layer(l)).collect();
            self.layer_batch_multi(l, &mut xs, &mut kvs)?;
        }

        let mut hidden = Vec::with_capacity(n * d);
        for x in xs.iter_mut() {
            simd::rmsnorm(x, self.cfg.rms_eps);
            simd::mul_assign(x, &self.output_norm);
            hidden.extend_from_slice(x);
        }

        let mut all = vec![0.0; n * vocab];
        self.output.matmul_rows(0, &hidden, n, &mut all)?;
        (0..n)
            .map(|i| Tensor::new(vec![vocab], all[i * vocab..(i + 1) * vocab].to_vec()))
            .collect()
    }

    fn hidden(&self, context: &[Token], seq: &mut SeqState) -> Result<Tensor, GarudaError> {
        let mut tail = self.forward_tail(context, seq, 1)?;
        Tensor::new(
            vec![self.cfg.d_model],
            tail.pop().expect("forward_tail returns n_last states"),
        )
    }

    fn logits(&self, context: &[Token], seq: &mut SeqState) -> Result<Tensor, GarudaError> {
        Ok(self
            .logits_multi(context, seq, 1)?
            .pop()
            .expect("logits_multi returns n tensors"))
    }

    fn speculation_supported(&self) -> bool {
        true
    }

    fn logits_multi(
        &self,
        context: &[Token],
        seq: &mut SeqState,
        n: usize,
    ) -> Result<Vec<Tensor>, GarudaError> {
        if n == 0 {
            return Err(GarudaError::Inference("logits_multi needs n >= 1".into()));
        }
        let tail = self.forward_tail(context, seq, n)?;
        let d = self.cfg.d_model;
        let vocab = self.cfg.vocab;

        // One batched matmul over the output head rather than `n` matvecs: the head
        // is `vocab x d_model`, the largest single matrix in the model.
        let flat: Vec<f32> = tail.into_iter().flatten().collect();
        let mut all = vec![0.0; n * vocab];
        self.output.matmul_rows(0, &flat, n, &mut all)?;

        let _ = d;
        (0..n)
            .map(|i| Tensor::new(vec![vocab], all[i * vocab..(i + 1) * vocab].to_vec()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{KvConfig, SeqState};
    use crate::core::ExpertLoader;
    use crate::predictor::ExpertPredictor;
    use crate::prefetch::{GgufPagePrefetcher, PrefetchEngine};
    use std::io::Write;

    // ---- a minimal GGUF writer, to build a synthetic MoE model with real data ----

    fn put_str(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }
    fn kv_u32(out: &mut Vec<u8>, key: &str, v: u32) {
        put_str(out, key);
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&v.to_le_bytes());
    }
    fn kv_f32(out: &mut Vec<u8>, key: &str, v: f32) {
        put_str(out, key);
        out.extend_from_slice(&6u32.to_le_bytes());
        out.extend_from_slice(&v.to_le_bytes());
    }
    fn kv_str(out: &mut Vec<u8>, key: &str, v: &str) {
        put_str(out, key);
        out.extend_from_slice(&8u32.to_le_bytes());
        put_str(out, v);
    }
    fn kv_str_array(out: &mut Vec<u8>, key: &str, vals: &[String]) {
        put_str(out, key);
        out.extend_from_slice(&9u32.to_le_bytes()); // ARRAY
        out.extend_from_slice(&8u32.to_le_bytes()); // of STRING
        out.extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for v in vals {
            put_str(out, v);
        }
    }

    /// Deterministic small weights, distinct per tensor `seed`.
    fn r#gen(seed: usize, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let h = seed
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(i.wrapping_mul(40_503));
                (h % 2000) as f32 / 2000.0 - 0.5
            })
            .collect()
    }

    /// Which GGUF tensor layout to emit for MoE experts.
    #[derive(Clone, Copy)]
    enum ExpertLayout {
        /// One stacked `..._exps` tensor, expert `e` at rows `[e·block, (e+1)·block)`.
        Merged,
        /// One tensor per expert, e.g. `blk.0.ffn_gate.3.weight`.
        Split,
    }

    /// Whether the checkpoint ships its own `output.weight` or ties the output head
    /// to `token_embd.weight` (which many real checkpoints do, and which is the case
    /// that used to load the embedding matrix twice).
    #[derive(Clone, Copy, PartialEq)]
    enum Head {
        Separate,
        Tied,
    }

    fn build_moe_gguf(layout: ExpertLayout) -> Vec<u8> {
        build_gguf(layout, Head::Separate)
    }

    /// Build a tiny MoE llama GGUF (F32 weights) entirely in memory, in either
    /// expert tensor layout. Both layouts get identical numbers (each gate/up/down
    /// is generated once as a flat array and either kept whole or sliced per
    /// expert), so outputs from the two layouts can be compared directly.
    fn build_gguf(layout: ExpertLayout, head: Head) -> Vec<u8> {
        let (d, kv_dim, ff, nl, vocab, ne) = (32usize, 16usize, 32usize, 2usize, 64usize, 4usize);

        // (name, ne-order dims, data)
        let mut tensors: Vec<(String, Vec<u64>, Vec<f32>)> = Vec::new();
        let add = |name: String,
                   dims: Vec<u64>,
                   seed: usize,
                   tv: &mut Vec<(String, Vec<u64>, Vec<f32>)>| {
            let n: usize = dims.iter().product::<u64>() as usize;
            let data = if name.contains("norm") {
                vec![1.0; n] // norms near 1 so rmsnorm output is sane
            } else {
                r#gen(seed, n)
            };
            tv.push((name, dims, data));
        };
        let mut s = 1;
        add(
            "token_embd.weight".into(),
            vec![d as u64, vocab as u64],
            s,
            &mut tensors,
        );
        s += 1;
        add("output_norm.weight".into(), vec![d as u64], s, &mut tensors);
        s += 1;
        if head == Head::Separate {
            add(
                "output.weight".into(),
                vec![d as u64, vocab as u64],
                s,
                &mut tensors,
            );
        }
        s += 1;
        for l in 0..nl {
            let p = |n: &str| format!("blk.{l}.{n}.weight");
            for (name, dims) in [
                (p("attn_norm"), vec![d as u64]),
                (p("attn_q"), vec![d as u64, d as u64]),
                (p("attn_k"), vec![d as u64, kv_dim as u64]),
                (p("attn_v"), vec![d as u64, kv_dim as u64]),
                (p("attn_output"), vec![d as u64, d as u64]),
                (p("ffn_norm"), vec![d as u64]),
                (p("ffn_gate_inp"), vec![d as u64, ne as u64]),
            ] {
                add(name, dims, s, &mut tensors);
                s += 1;
            }

            for (base, out_rows, cols) in
                [("ffn_gate", ff, d), ("ffn_up", ff, d), ("ffn_down", d, ff)]
            {
                let flat = r#gen(s, ne * out_rows * cols);
                s += 1;
                match layout {
                    ExpertLayout::Merged => tensors.push((
                        format!("blk.{l}.{base}_exps.weight"),
                        vec![cols as u64, out_rows as u64, ne as u64],
                        flat,
                    )),
                    ExpertLayout::Split => {
                        let block = out_rows * cols;
                        for e in 0..ne {
                            tensors.push((
                                format!("blk.{l}.{base}.{e}.weight"),
                                vec![cols as u64, out_rows as u64],
                                flat[e * block..(e + 1) * block].to_vec(),
                            ));
                        }
                    }
                }
            }
        }

        // metadata
        let mut meta = Vec::new();
        let mut kv_count = 0u64;
        macro_rules! m {
            ($f:expr_2021) => {{
                $f;
                kv_count += 1;
            }};
        }
        m!(kv_str(&mut meta, "general.architecture", "llama"));
        m!(kv_u32(&mut meta, "llama.embedding_length", d as u32));
        m!(kv_u32(&mut meta, "llama.block_count", nl as u32));
        m!(kv_u32(&mut meta, "llama.attention.head_count", 4));
        m!(kv_u32(&mut meta, "llama.attention.head_count_kv", 2));
        m!(kv_u32(&mut meta, "llama.feed_forward_length", ff as u32));
        m!(kv_u32(&mut meta, "llama.context_length", 64));
        m!(kv_f32(
            &mut meta,
            "llama.attention.layer_norm_rms_epsilon",
            1e-5
        ));
        m!(kv_u32(&mut meta, "llama.expert_count", ne as u32));
        m!(kv_u32(&mut meta, "llama.expert_used_count", 2));
        let toks: Vec<String> = (0..vocab).map(|i| format!("t{i}")).collect();
        m!(kv_str_array(&mut meta, "tokenizer.ggml.tokens", &toks));

        // tensor data offsets (each aligned to 32, relative to the data section)
        let align = |x: usize| x.next_multiple_of(32);
        let mut offsets = Vec::new();
        let mut cursor = 0usize;
        for (_, _, data) in &tensors {
            let off = align(cursor);
            offsets.push(off as u64);
            cursor = off + data.len() * 4;
        }
        let data_size = align(cursor);

        // tensor infos
        let mut infos = Vec::new();
        for ((name, dims, _), &off) in tensors.iter().zip(&offsets) {
            put_str(&mut infos, name);
            infos.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for &dim in dims {
                infos.extend_from_slice(&dim.to_le_bytes());
            }
            infos.extend_from_slice(&0u32.to_le_bytes()); // type F32
            infos.extend_from_slice(&off.to_le_bytes());
        }

        // assemble
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&kv_count.to_le_bytes());
        out.extend_from_slice(&meta);
        out.extend_from_slice(&infos);
        let data_start = out.len().next_multiple_of(32);
        out.resize(data_start, 0);
        let base = out.len();
        out.resize(base + data_size, 0);
        for ((_, _, data), &off) in tensors.iter().zip(&offsets) {
            let at = base + off as usize;
            for (i, &v) in data.iter().enumerate() {
                out[at + i * 4..at + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    fn seq_for(b: &LlamaBackend) -> SeqState {
        let lc = b.config();
        SeqState::new(
            KvConfig {
                dims: b.dims(),
                kv_dim: lc.kv_dim(),
                n_layers: lc.n_layers,
                kv_dims: None,
                max_positions: 64,
                max_resident_blocks: 64,
                sliding_window: None,
                storage: None,
            },
            1,
        )
    }

    #[test]
    fn matvec_rows_agrees_between_full_and_packed() {
        let (rows, cols) = (6usize, 4usize);
        let mat: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.1 - 1.0).collect();
        let x = vec![0.5, -1.0, 2.0, 0.25];

        let full = Weight::Full {
            data: mat.clone(),
            cols,
        };

        // Packed(F32) over a memory-mapped copy of the same bytes.
        let dir = std::env::temp_dir().join("garuda_matvec_rows");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("m.bin");
        let mut bytes = Vec::new();
        for &v in &mat {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&path, &bytes).unwrap();
        let mmap = Arc::new(unsafe { Mmap::map(&std::fs::File::open(&path).unwrap()).unwrap() });
        let packed = Weight::Packed {
            qtype: crate::quant::F32,
            cols,
            src: Bytes::Mapped(mmap),
            start: 0,
        };

        // A sub-range of rows [2, 5).
        let mut of = vec![0.0; 3];
        let mut op = vec![0.0; 3];
        full.matvec_rows(2, &x, &mut of).unwrap();
        packed.matvec_rows(2, &x, &mut op).unwrap();

        for r in 0..3 {
            let naive: f32 = (0..cols).map(|c| mat[(2 + r) * cols + c] * x[c]).sum();
            assert!((of[r] - naive).abs() < 1e-5, "full row {r}");
            assert!((op[r] - naive).abs() < 1e-5, "packed row {r}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn seq_with_capacity(b: &LlamaBackend, max_positions: usize) -> SeqState {
        let lc = b.config();
        SeqState::new(
            KvConfig {
                dims: b.dims(),
                kv_dim: lc.kv_dim(),
                n_layers: lc.n_layers,
                kv_dims: None,
                max_positions,
                max_resident_blocks: 1024,
                sliding_window: None,
                storage: None,
            },
            1,
        )
    }

    /// Prefill drives a chunk of tokens through one layer before moving to the next,
    /// so a layer's weights are read once for many tokens instead of once each. That
    /// reorders *nothing* the arithmetic depends on — each token's hidden state flows
    /// through the layers independently, and layer `l`'s attention reads layer `l`'s
    /// KV, which is appended in token order either way — so a batched prefill must
    /// agree exactly with feeding the same tokens one at a time.
    #[test]
    fn a_batched_prefill_matches_feeding_tokens_one_at_a_time() {
        let bytes = build_moe_gguf(ExpertLayout::Merged);
        let batched_backend = LlamaBackend::load(&bytes).unwrap().with_prefill_chunk(16);
        let stepwise_backend = LlamaBackend::load(&bytes).unwrap().with_prefill_chunk(1);
        let tokens: Vec<Token> = vec![3, 7, 1, 9, 2, 5, 4, 8, 6, 0, 11, 13];

        // One call, layer-major over a single chunk of 12.
        let mut batched_seq = seq_for(&batched_backend);
        let batched = batched_backend.logits(&tokens, &mut batched_seq).unwrap();

        // Token-major, one token per call — the order this replaced.
        let mut stepwise_seq = seq_for(&stepwise_backend);
        let mut stepwise = None;
        for n in 1..=tokens.len() {
            stepwise = Some(
                stepwise_backend
                    .logits(&tokens[..n], &mut stepwise_seq)
                    .unwrap(),
            );
        }

        assert_eq!(
            batched.data(),
            stepwise.unwrap().data(),
            "batching the prefill changed the output"
        );
        assert_eq!(batched_seq.len(), stepwise_seq.len());
    }

    /// The same equivalence across a chunk boundary, where the layer loop restarts
    /// and each layer's KV is already part-filled by the previous chunk.
    #[test]
    fn a_prefill_spanning_several_chunks_matches_a_stepwise_one() {
        const CHUNK: usize = 32;
        let backend = LlamaBackend::load(&build_moe_gguf(ExpertLayout::Merged))
            .unwrap()
            .with_prefill_chunk(CHUNK);
        let n = CHUNK + CHUNK / 2; // 1.5 chunks: one full pass, one partial
        let vocab = backend.config().vocab as Token;
        let tokens: Vec<Token> = (0..n).map(|i| (i as Token * 7 + 1) % vocab).collect();

        let mut batched_seq = seq_with_capacity(&backend, n + 8);
        let batched = backend.logits(&tokens, &mut batched_seq).unwrap();

        // Two calls that split at a point unrelated to the chunk size, so the second
        // call's chunking starts from a non-zero `already`.
        let mut split_seq = seq_with_capacity(&backend, n + 8);
        backend.logits(&tokens[..13], &mut split_seq).unwrap();
        let split = backend.logits(&tokens, &mut split_seq).unwrap();

        assert_eq!(
            batched.data(),
            split.data(),
            "chunk boundary changed output"
        );
        assert_eq!(batched_seq.len(), n);
    }

    /// An over-long prefill is refused before any layer runs, rather than discovered
    /// partway through — which under layer-major would leave the layers a whole chunk
    /// apart instead of one position.
    #[test]
    fn an_oversized_prefill_is_refused_before_the_layers_run() {
        let backend = LlamaBackend::load(&build_moe_gguf(ExpertLayout::Merged)).unwrap();
        let vocab = backend.config().vocab as Token;
        let tokens: Vec<Token> = (0..40).map(|i| (i as Token) % vocab).collect();

        let mut seq = seq_with_capacity(&backend, 16);
        let err = backend.logits(&tokens, &mut seq).unwrap_err();
        assert!(matches!(err, GarudaError::Cache(_)), "got {err:?}");
        assert_eq!(
            seq.len(),
            0,
            "a refused prefill must not have touched the kv"
        );
        for l in 0..backend.config().n_layers {
            assert_eq!(seq.layer(l).len(), 0, "layer {l} was advanced anyway");
        }
    }

    /// A batched decode step must produce exactly what decoding each sequence on its
    /// own produces. The sequences share nothing but the weights, so batching them is
    /// only ever an optimisation — this is the test that says so.
    #[test]
    fn a_batched_decode_step_matches_decoding_each_sequence_alone() {
        let backend = LlamaBackend::load(&build_moe_gguf(ExpertLayout::Merged)).unwrap();
        let vocab = backend.config().vocab as Token;

        // Deliberately different lengths and contents, so the batch is not secretly
        // uniform and a mixed-up index would show.
        let convos: Vec<Vec<Token>> = vec![
            vec![3, 7, 1, 9],
            vec![11, 2],
            vec![5, 5, 5, 5, 5, 5],
            vec![vocab - 1, 0, 42],
        ];

        // Bring each sequence up to date one at a time, then take one more step —
        // once alone, once as part of a batch.
        let mut alone_seqs: Vec<SeqState> = convos.iter().map(|_| seq_for(&backend)).collect();
        let mut batch_seqs: Vec<SeqState> = convos.iter().map(|_| seq_for(&backend)).collect();
        for (i, c) in convos.iter().enumerate() {
            backend.logits(c, &mut alone_seqs[i]).unwrap();
            backend.logits(c, &mut batch_seqs[i]).unwrap();
        }

        let stepped: Vec<Vec<Token>> = convos
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let mut c = c.clone();
                c.push((i as Token * 13 + 4) % vocab);
                c
            })
            .collect();

        let alone: Vec<Tensor> = stepped
            .iter()
            .enumerate()
            .map(|(i, c)| backend.logits(c, &mut alone_seqs[i]).unwrap())
            .collect();

        let refs: Vec<&[Token]> = stepped.iter().map(|c| c.as_slice()).collect();
        let mut batch_refs: Vec<&mut SeqState> = batch_seqs.iter_mut().collect();
        let batched = backend.logits_batch(&refs, &mut batch_refs).unwrap();
        drop(batch_refs);

        assert_eq!(batched.len(), alone.len());
        for (i, (b, a)) in batched.iter().zip(&alone).enumerate() {
            assert_eq!(b.data(), a.data(), "sequence {i} differs when batched");
            assert_eq!(batch_seqs[i].len(), alone_seqs[i].len(), "kv length {i}");
        }
    }

    /// The batched path is for equal work per sequence. A batch where the sequences
    /// have different amounts to catch up on must still be answered correctly, by
    /// falling back rather than by producing something wrong.
    #[test]
    fn a_ragged_batch_falls_back_and_still_answers_correctly() {
        let backend = LlamaBackend::load(&build_moe_gguf(ExpertLayout::Merged)).unwrap();
        let a: Vec<Token> = vec![3, 7, 1];
        let b: Vec<Token> = vec![9, 2, 5, 4, 8];

        let mut alone = [seq_for(&backend), seq_for(&backend)];
        let want = [
            backend.logits(&a, &mut alone[0]).unwrap(),
            backend.logits(&b, &mut alone[1]).unwrap(),
        ];

        // Both sequences start empty, so they have 3 and 5 tokens to consume: ragged.
        let mut owned = [seq_for(&backend), seq_for(&backend)];
        let mut seqs: Vec<&mut SeqState> = owned.iter_mut().collect();
        let got = backend
            .logits_batch(&[a.as_slice(), b.as_slice()], &mut seqs)
            .unwrap();

        for i in 0..2 {
            assert_eq!(got[i].data(), want[i].data(), "sequence {i}");
        }
    }

    #[test]
    fn logits_batch_rejects_mismatched_lengths_and_bad_tokens() {
        let backend = LlamaBackend::load(&build_moe_gguf(ExpertLayout::Merged)).unwrap();
        let c: Vec<Token> = vec![1, 2];
        let mut owned = [seq_for(&backend), seq_for(&backend)];
        let mut seqs: Vec<&mut SeqState> = owned.iter_mut().collect();

        assert!(
            backend.logits_batch(&[c.as_slice()], &mut seqs).is_err(),
            "one context for two sequences must not be accepted"
        );

        // Bring both to the same length, then step with an out-of-vocabulary token.
        backend.logits(&c, seqs[0]).unwrap();
        backend.logits(&c, seqs[1]).unwrap();
        let bad: Vec<Token> = vec![1, 2, backend.config().vocab as Token + 3];
        let err = backend
            .logits_batch(&[bad.as_slice(), bad.as_slice()], &mut seqs)
            .unwrap_err();
        assert!(matches!(err, GarudaError::InvalidToken(_)), "got {err:?}");
        assert_eq!(
            seqs[0].len(),
            2,
            "a refused batch must not advance any cache"
        );
        assert_eq!(seqs[1].len(), 2);
    }

    /// Verifying speculated tokens rests entirely on this: the logits `logits_multi`
    /// reports for an earlier position must be exactly the logits the model would
    /// have produced had it stopped there. If they differ, speculation accepts tokens
    /// the model would not have chosen.
    #[test]
    fn logits_multi_matches_stopping_at_each_position() {
        let backend = LlamaBackend::load(&build_moe_gguf(ExpertLayout::Merged)).unwrap();
        let tokens: Vec<Token> = vec![3, 7, 1, 9, 2, 5];

        // One pass over all six, asking for the last four positions.
        let n = 4;
        let mut seq = seq_for(&backend);
        let many = backend.logits_multi(&tokens, &mut seq, n).unwrap();
        assert_eq!(many.len(), n);
        assert_eq!(seq.len(), tokens.len(), "the pass must consume every token");

        // The same positions reached one prefix at a time.
        for (k, got) in many.iter().enumerate() {
            let upto = tokens.len() - n + k + 1;
            let mut alone = seq_for(&backend);
            let want = backend.logits(&tokens[..upto], &mut alone).unwrap();
            assert_eq!(
                got.data(),
                want.data(),
                "position {upto} differs from stopping there"
            );
        }
    }

    #[test]
    fn logits_multi_refuses_more_positions_than_it_computed() {
        let backend = LlamaBackend::load(&build_moe_gguf(ExpertLayout::Merged)).unwrap();
        let tokens: Vec<Token> = vec![3, 7, 1];
        let mut seq = seq_for(&backend);
        backend.logits(&tokens, &mut seq).unwrap();

        // Only one token is new, so there is no earlier position to report on.
        let more: Vec<Token> = vec![3, 7, 1, 9];
        let err = backend.logits_multi(&more, &mut seq, 3).unwrap_err();
        assert!(matches!(err, GarudaError::Inference(_)), "got {err:?}");
        assert!(backend.speculation_supported());
    }

    /// A draft model must not change what the target produces. Greedy is the strict
    /// case: a guess is kept only where the target's own argmax agrees, so the tokens
    /// have to match plain decoding exactly — and both caches have to end up
    /// describing those same tokens, or the next round's guesses are built on fiction.
    #[test]
    fn a_draft_model_does_not_change_greedy_output() {
        use crate::cache::KvConfig;
        use crate::runtime::{InferenceRuntime, SamplingParams};

        let bytes = build_moe_gguf(ExpertLayout::Merged);
        let target = Arc::new(LlamaBackend::load(&bytes).unwrap());
        // A second, independently-loaded copy stands in for a draft model: it shares
        // the vocabulary, which is the only thing the mechanism requires. Its guesses
        // will often be right, which exercises the accept path properly.
        let draft = Arc::new(LlamaBackend::load(&bytes).unwrap());
        let lc = target.config();
        let dims = target.dims();
        let kv = move || KvConfig {
            dims,
            kv_dim: lc.kv_dim(),
            n_layers: lc.n_layers,
            kv_dims: None,
            max_positions: 64,
            max_resident_blocks: 64,
            sliding_window: None,
            storage: None,
        };

        let plain = InferenceRuntime::new(
            Arc::new(crate::tokenizer::Tokenizer::new()),
            target.clone(),
            kv(),
            4,
            1 << 20,
        );
        let spec = InferenceRuntime::new(
            Arc::new(crate::tokenizer::Tokenizer::new()),
            target,
            kv(),
            4,
            1 << 20,
        )
        .with_drafter(draft, kv());
        assert!(spec.has_drafter());

        let p = SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            max_tokens: 20,
            seed: Some(1),
        };
        let prompt: Vec<Token> = vec![3, 7, 1, 9, 2];

        let mut a = plain.start(&prompt, &p).unwrap();
        let mut want = Vec::new();
        while let Ok(t) = plain.next_token(&mut a, &p) {
            want.push(t);
        }

        let mut b = spec.start(&prompt, &p).unwrap();
        let mut got = Vec::new();
        let mut multi = 0;
        loop {
            let mut batch = Vec::new();
            let done = spec
                .next_tokens_speculative(&mut b, &p, 4, &mut batch)
                .is_err();
            if batch.len() > 1 {
                multi += 1;
            }
            got.extend(batch);
            if done {
                break;
            }
        }

        assert_eq!(got, want, "the draft model changed the output");
        assert!(
            multi > 0,
            "no round ever won more than one token, so the accept path went untested"
        );
        assert_eq!(a.generated(), b.generated());
    }

    /// The same row-range plumbing as above, but over a k-quant. F32 rows are 4 bytes
    /// an element; a Q6_K row is 210 bytes per 256 elements, so `row_start * row_bytes`
    /// is doing genuinely different arithmetic — and a stacked MoE expert tensor is
    /// addressed entirely through it. An off-by-one here silently reads a neighbouring
    /// expert's weights.
    #[test]
    fn packed_k_quant_rows_are_addressed_correctly() {
        use crate::quant;

        let (rows, cols) = (6usize, 512usize); // 2 Q6_K super-blocks per row
        let blocks_per_row = cols / 256;
        let row_bytes = 210 * blocks_per_row;

        // Valid-but-arbitrary Q6_K bytes, deterministic.
        let mut state = 4242u64;
        let mut rnd = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let mut packed = Vec::with_capacity(rows * row_bytes);
        for _ in 0..rows * blocks_per_row {
            let mut b = [0u8; 210];
            for byte in b[0..192].iter_mut() {
                *byte = (rnd() % 256) as u8;
            }
            for s in b[192..208].iter_mut() {
                *s = ((rnd() % 127) as i32 - 63) as i8 as u8;
            }
            b[208..210].copy_from_slice(&0x1400u16.to_le_bytes()); // f16 d, ~0.0039
            packed.extend_from_slice(&b);
        }

        let dir = std::env::temp_dir().join("garuda_packed_q6k");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("q6k.bin");
        std::fs::write(&path, &packed).unwrap();
        let mmap = Arc::new(unsafe { Mmap::map(&std::fs::File::open(&path).unwrap()).unwrap() });

        let full = Weight::Full {
            data: quant::dequantize(quant::Q6_K, &packed, rows * cols).unwrap(),
            cols,
        };
        let packed_w = Weight::Packed {
            qtype: quant::Q6_K,
            cols,
            src: Bytes::Mapped(mmap),
            start: 0,
        };

        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.017).sin()).collect();

        // A sub-range of rows [2, 5), the way one expert's slice of a stacked tensor
        // is read.
        let mut of = vec![0.0; 3];
        let mut op = vec![0.0; 3];
        full.matvec_rows(2, &x, &mut of).unwrap();
        packed_w.matvec_rows(2, &x, &mut op).unwrap();

        for r in 0..3 {
            // The packed path quantises the activation to int8, so this is close
            // rather than exact — but it must be the *same row*, which a wrong offset
            // would not be.
            let tol = of[r].abs() * 0.05 + 1e-3;
            assert!(
                (of[r] - op[r]).abs() < tol,
                "row {r}: full {} vs packed {} (tol {tol})",
                of[r],
                op[r]
            );
        }

        // And a wrong offset must be detectable: neighbouring rows differ.
        let mut other = vec![0.0; 3];
        full.matvec_rows(3, &x, &mut other).unwrap();
        assert!(
            (of[0] - other[0]).abs() > 1e-6,
            "fixture is degenerate: adjacent rows produce the same dot product"
        );

        // Single-row reads go through a different branch; check that one too.
        let row2 = packed_w.row(2).unwrap();
        assert_eq!(row2.len(), cols);
        let naive: f32 = row2.iter().zip(&x).map(|(a, b)| a * b).sum();
        assert!((naive - of[0]).abs() < of[0].abs() * 0.01 + 1e-4);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn synthetic_moe_model_loads_routes_and_runs() {
        let bytes = build_moe_gguf(ExpertLayout::Merged);

        // f32-expand path
        let f32b = LlamaBackend::load(&bytes).unwrap();
        assert_eq!(f32b.config().n_experts, 4);
        assert_eq!(f32b.config().n_experts_used, 2);
        assert!(!f32b.is_mmapped());

        let mut s1 = seq_for(&f32b);
        let a = f32b.logits(&[3, 7, 1], &mut s1).unwrap();
        assert_eq!(a.shape(), &[64]);
        assert!(
            a.data().iter().all(|v| v.is_finite()),
            "MoE produced non-finite logits"
        );

        // Different context must give different logits (the model is actually routing
        // and computing, not degenerate).
        let mut s2 = seq_for(&f32b);
        let b = f32b.logits(&[9, 2, 5], &mut s2).unwrap();
        assert_ne!(a.data(), b.data());

        // mmap path: same model, weights kept packed and dequantised per row.
        let dir = std::env::temp_dir().join("garuda_moe_gguf");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("moe.gguf");
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&bytes)
            .unwrap();
        let mmap = Arc::new(unsafe { Mmap::map(&std::fs::File::open(&path).unwrap()).unwrap() });
        let g = Gguf::parse(&mmap).unwrap();
        let mmb = LlamaBackend::from_gguf(&g, &mmap, Some(mmap.clone())).unwrap();
        assert!(mmb.is_mmapped());

        let mut s3 = seq_for(&mmb);
        let c = mmb.logits(&[3, 7, 1], &mut s3).unwrap();

        // The packed path must match f32 exactly-ish — proving the per-expert slice
        // offsets (e·d_ff, e·d_model) into the stacked expert tensors are correct.
        for (x, y) in a.data().iter().zip(c.data()) {
            assert!((x - y).abs() < 1e-3, "f32 {x} vs mmap {y}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Older llama.cpp conversions (e.g. the original TheBloke Mixtral GGUFs) store
    /// each expert as its own tensor (`blk.0.ffn_gate.3.weight`) instead of one
    /// stacked `..._exps` tensor. Both must load and produce the same logits, since
    /// [`build_moe_gguf`] gives the two layouts identical numbers.
    #[test]
    fn split_expert_tensors_match_merged_layout() {
        let merged = LlamaBackend::load(&build_moe_gguf(ExpertLayout::Merged)).unwrap();
        let split = LlamaBackend::load(&build_moe_gguf(ExpertLayout::Split)).unwrap();
        assert_eq!(split.config().n_experts, 4);

        let mut s1 = seq_for(&merged);
        let mut s2 = seq_for(&split);
        let a = merged.logits(&[3, 7, 1], &mut s1).unwrap();
        let b = split.logits(&[3, 7, 1], &mut s2).unwrap();
        assert_eq!(a.data(), b.data());
    }

    /// A checkpoint that omits `output.weight` uses the embedding matrix as its
    /// output head. That must hold it exactly once: the loader used to call the
    /// tensor reader a second time, producing an independent `f32` copy of a
    /// `vocab x d_model` matrix — on a real tied model with a 128k vocabulary, a
    /// gigabyte of duplicate weights.
    #[test]
    fn tied_embeddings_share_one_allocation_and_still_produce_the_same_logits() {
        let tied = LlamaBackend::load(&build_gguf(ExpertLayout::Merged, Head::Tied)).unwrap();
        assert!(tied.has_tied_embeddings(), "head should be tied");
        assert!(
            Arc::ptr_eq(&tied.token_embd, &tied.output),
            "the tied head must be the same allocation, not a copy"
        );

        let separate =
            LlamaBackend::load(&build_gguf(ExpertLayout::Merged, Head::Separate)).unwrap();
        assert!(!separate.has_tied_embeddings());
        assert!(!Arc::ptr_eq(&separate.token_embd, &separate.output));

        // Sharing must not change the arithmetic: a tied head is still a real matvec
        // against the embedding matrix.
        let mut seq = seq_for(&tied);
        let logits = tied.logits(&[3, 7, 1], &mut seq).unwrap();
        assert_eq!(logits.shape(), &[tied.config().vocab]);
        assert!(logits.data().iter().all(|v| v.is_finite()));

        let mut naive = vec![0.0f32; tied.config().vocab];
        let mut hidden_seq = seq_for(&tied);
        let hidden = tied.hidden(&[3, 7, 1], &mut hidden_seq).unwrap();
        tied.token_embd.matvec(hidden.data(), &mut naive).unwrap();
        assert_eq!(logits.data(), &naive[..]);
    }

    /// A fresh, uniquely-named temp directory per call — tests run in parallel, and
    /// a shared fixed path lets one test's `remove_dir_all` cleanup race another
    /// test (or another loop iteration) still using it.
    fn mmap_of(bytes: &[u8], name: &str) -> (std::path::PathBuf, Arc<Mmap>) {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("garuda_prefetch_test_{n}"));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join(name);
        std::fs::File::create(&path)
            .unwrap()
            .write_all(bytes)
            .unwrap();
        let mmap = Arc::new(unsafe { Mmap::map(&std::fs::File::open(&path).unwrap()).unwrap() });
        (dir, mmap)
    }

    /// A mmapped model's expert page ranges must be non-empty, in-bounds, and
    /// distinct per (layer, expert) — the table [`GgufPagePrefetcher`] warms from.
    #[test]
    fn expert_page_ranges_are_in_bounds_and_distinct() {
        for layout in [ExpertLayout::Merged, ExpertLayout::Split] {
            let bytes = build_moe_gguf(layout);
            let (dir, mmap) = mmap_of(&bytes, "ranges.gguf");
            let g = Gguf::parse(&mmap).unwrap();
            let backend = LlamaBackend::from_gguf(&g, &mmap, Some(mmap.clone())).unwrap();

            let cfg = backend.config();
            let ranges = backend.expert_page_ranges();
            assert_eq!(ranges.len(), cfg.n_layers * cfg.n_experts);

            let mut seen = std::collections::HashSet::new();
            for id_ranges in &ranges {
                // 3 weights (gate, up, down) per expert, each present under mmap.
                assert_eq!(id_ranges.len(), 3);
                for &(start, len) in id_ranges {
                    assert!(len > 0);
                    assert!(start + len <= mmap.len(), "range runs past the file");
                    assert!(
                        seen.insert((start, len)),
                        "duplicate range {start}..{}",
                        start + len
                    );
                }
            }
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// Attaching a `PrefetchEngine` must not change what the model produces — it is
    /// advisory only — but across a real multi-step decode it must actually predict
    /// and warm pages, proving the wiring is live rather than plumbing that merely
    /// compiles.
    #[test]
    fn prefetch_engine_warms_experts_without_changing_logits() {
        let bytes = build_moe_gguf(ExpertLayout::Merged);
        let (dir, mmap) = mmap_of(&bytes, "moe.gguf");
        let g = Gguf::parse(&mmap).unwrap();
        let tokens: Vec<Token> = vec![3, 7, 1, 9, 2, 5, 4, 8, 6, 0];

        // Baseline: no prefetch.
        let plain = LlamaBackend::from_gguf(&g, &mmap, Some(mmap.clone())).unwrap();
        let mut s_plain = seq_for(&plain);
        let baseline: Vec<_> = (1..=tokens.len())
            .map(|n| plain.logits(&tokens[..n], &mut s_plain).unwrap())
            .collect();

        // Same model, with a real prefetch engine attached.
        let with_pf = LlamaBackend::from_gguf(&g, &mmap, Some(mmap.clone())).unwrap();
        let cfg = with_pf.config();
        let ranges = with_pf.expert_page_ranges();
        let predictor = Arc::new(ExpertPredictor::new(cfg.n_layers * cfg.n_experts));
        let loader: Arc<dyn ExpertLoader> = Arc::new(GgufPagePrefetcher::new(mmap.clone(), ranges));
        let pf = Arc::new(PrefetchEngine::new(
            loader,
            predictor,
            true,
            cfg.n_experts_used.max(1),
        ));
        let with_pf = with_pf.with_prefetch(pf.clone());

        let mut s_pf = seq_for(&with_pf);
        for (n, want) in (1..=tokens.len()).zip(&baseline) {
            let got = with_pf.logits(&tokens[..n], &mut s_pf).unwrap();
            assert_eq!(
                got.data(),
                want.data(),
                "prefetch changed output at step {n}"
            );
        }

        // Background warms are spawned on rayon; poll rather than trust a fixed
        // sleep to outrace them when the full suite is loading every core.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while pf.launched() + pf.skipped() == 0 && std::time::Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert!(
            pf.launched() + pf.skipped() > 0,
            "prefetcher predicted nothing across {} decode steps",
            tokens.len()
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}

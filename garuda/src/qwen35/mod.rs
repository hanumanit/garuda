//! The Qwen3.5 family: a hybrid transformer whose layers are mostly *not* attention.
//!
//! Three out of every four blocks are a **gated delta net** — a linear-attention layer
//! that keeps a fixed-size recurrent state instead of a growing cache of keys and
//! values. The fourth is ordinary grouped-query attention. `Qwen3.8-27B` stacks 64
//! blocks that way (48 recurrent, 16 attention); the same architecture covers
//! `Qwen3.5-0.8B` through `Qwen3.6-27B`, which differ only in their dimensions.
//!
//! What that changes for this runtime:
//!
//! * **The KV cache is a quarter the size.** Only the attention blocks store keys and
//!   values; the recurrent blocks hold [`LinearState`], which is the same size at
//!   100 000 tokens as at ten. Those blocks still count positions, so every layer of a
//!   sequence advances together as the [`InferenceBackend`] contract requires — they
//!   just store nothing (`KvConfig::kv_dims`).
//! * **A sequence cannot be rewound.** A recurrent state summarises every token it has
//!   read, and no arithmetic takes the last few back out. So speculative decoding is
//!   off for this architecture ([`Self::speculation_supported`] is `false`) rather than
//!   quietly wrong, and [`crate::cache::SeqState::truncate`] refuses.
//! * **The attention blocks are not Llama's.** Heads are 256 wide against a 5120-wide
//!   residual stream, the query projection emits a gate alongside the query, queries
//!   and keys are RMS-normalised per head before rotation, and only the first quarter
//!   of each head's dimensions is rotated at all.
//!
//! # What is implemented, and what is not
//!
//! Text in, text out, on a `qwen35` GGUF checkpoint: every arithmetic step of the
//! decoder, F32/F16 and every quant [`crate::quant`] decodes, in RAM or memory-mapped.
//!
//! Not implemented: the vision tower (a Qwen3.5 GGUF ships it as a separate `mmproj`
//! file, and this runtime has no image input), the multi-token-prediction head (also a
//! separate file), and the mixture-of-experts variant `qwen35moe`. Loading refuses
//! those rather than half-running them.
//!
//! # mRoPE, and why text needs none of it
//!
//! Qwen3.5 positions are three-axis (time, height, width) so that an image's patches
//! can be placed in two dimensions. For text every axis holds the same value, and
//! rotation by three equal positions is rotation by one — so the sections in
//! `qwen35.rope.dimension_sections` do not change a text-only forward pass, and this
//! implementation rotates by position as usual. An image would need them; there are no
//! images here.
//!
//! # The delta rule
//!
//! Each recurrent head carries a `key_head_dim x value_head_dim` matrix `S`. Per
//! token, with decay `α`, write strength `β`, and L2-normalised `q`/`k`:
//!
//! ```text
//! S ← αS + β·k ⊗ (v − Sᵀk)      out = Sᵀ(q/√d)
//! ```
//!
//! `Sᵀk` is what the state currently predicts for this key; the update writes the
//! error, scaled by `β`, along `k`. `α = exp(a·softplus(alpha·x + dt_bias))` with `a`
//! negative in the file, so `α` lands in `(0, 1)` and old writes fade.
//!
//! Prefill runs that recurrence token by token. The chunked matrix form llama.cpp uses
//! is an optimisation for parallel hardware, not a different answer, and this runtime
//! is scalar CPU code where the sequential form is both simpler and no slower per
//! token.

use crate::cache::{LinearState, SeqState};
use crate::core::{GarudaError, InferenceBackend, ModelDims, Tensor, Token};
use crate::gguf::{Gguf, Value};
use crate::llama::{Weight, load_norm, load_weight};
use crate::{quant, simd};
use memmap2::Mmap;
use std::sync::Arc;

/// Which key head a delta net's value head reads, when the value heads outnumber
/// them: value head `hv` reads key head `hv % n_k_heads`.
///
/// This is a **file** convention, not a mathematical one, and it is worth stating
/// plainly because the reference implementations disagree on the surface. The
/// `transformers` model repeats each key head for a contiguous run of value heads
/// (`repeat_interleave`, so `j / group`); llama.cpp's graph repeats the whole key-head
/// block instead (`ggml_repeat_4d`, so `j % n_k_heads`). Both are right for their own
/// weight layout: the GGUF converter writes the heads in the order llama.cpp reads
/// them, and a GGUF file is what this runtime loads.
///
/// Getting it backwards is not a crash and not obviously wrong output. Every head
/// still reads a real memory — just one another head wrote — so the activations keep
/// their usual magnitudes and the model degenerates into copying its prompt. On
/// `Qwen3.8-27B` (48 value heads over 16 key heads) "The capital of France is"
/// continued as " capital of France is capital of France is" under the wrong mapping
/// and " Paris" under this one. Checkpoints with as many value heads as key heads —
/// `Qwen3.5-0.8B`, for instance — cannot tell the two apart, so the small model that
/// is convenient to test with will not catch this.
fn key_head_of(hv: usize, n_k_heads: usize) -> usize {
    hv % n_k_heads
}

/// Prompt tokens driven through one layer before moving to the next. Same tradeoff as
/// [`crate::llama::DEFAULT_PREFILL_CHUNK`]: it bounds the activation buffer, not the
/// benefit.
pub const DEFAULT_PREFILL_CHUNK: usize = 256;

/// The architecture parameters, read from GGUF metadata.
#[derive(Debug, Clone)]
pub struct Qwen35Config {
    pub d_model: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub vocab: usize,
    pub context: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    /// Dimensions of each attention head that rotate. A quarter of `head_dim` for
    /// every published Qwen3.5 checkpoint (`partial_rotary_factor = 0.25`).
    pub n_rot: usize,

    // Attention blocks.
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,

    // Gated delta net blocks.
    pub conv_kernel: usize,
    pub n_k_heads: usize,
    pub n_v_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    /// One in every `full_attention_interval` blocks is an attention block, the last
    /// of each group.
    pub full_attention_interval: usize,
    /// `true` where the block is a gated delta net, indexed by block.
    pub recurrent: Vec<bool>,
    /// Multi-token-prediction blocks the file carries beyond the decoder stack.
    ///
    /// `qwen35.block_count` counts these, so the stack this backend runs is the first
    /// `block_count - n_nextn` blocks. Their weights are a draft head — llama.cpp runs
    /// them to propose tokens the main stack then checks — and this backend does not
    /// use them: see the module docs on why speculation is off here. Some publishers
    /// bundle them (the 0.8B does), others ship them as a separate `mtp-*.gguf` (the
    /// 27B does), and both have to load.
    pub n_nextn: usize,
}

impl Qwen35Config {
    fn from_gguf(g: &Gguf) -> Result<Self, GarudaError> {
        match g.architecture() {
            Some("qwen35") => {}
            Some("qwen35moe") => {
                return Err(GarudaError::Model(
                    "architecture 'qwen35moe' is not supported: this backend runs the dense \
                     Qwen3.5 series, whose blocks have a single feed-forward network"
                        .into(),
                ));
            }
            other => {
                return Err(GarudaError::Model(format!(
                    "architecture '{}' is not Qwen3.5",
                    other.unwrap_or("unknown")
                )));
            }
        }
        let need = |suffix: &str| {
            g.arch_u64(suffix)
                .ok_or_else(|| GarudaError::Model(format!("gguf is missing qwen35.{suffix}")))
        };

        let d_model = need("embedding_length")? as usize;
        let n_blocks = need("block_count")? as usize;
        let n_nextn = g.arch_u64("nextn_predict_layers").unwrap_or(0) as usize;
        if n_nextn >= n_blocks {
            return Err(GarudaError::Model(format!(
                "qwen35.nextn_predict_layers ({n_nextn}) leaves no decoder blocks out of                  {n_blocks}"
            )));
        }
        // The prediction blocks sit after the stack and are not part of it.
        let n_layers = n_blocks - n_nextn;
        let n_heads = need("attention.head_count")? as usize;
        let n_kv_heads = need("attention.head_count_kv")? as usize;
        let head_dim = need("attention.key_length")? as usize;
        let value_length = g
            .arch_u64("attention.value_length")
            .unwrap_or(head_dim as u64) as usize;
        let d_ff = need("feed_forward_length")? as usize;

        // The delta net dimensions travel under the state-space names llama.cpp gives
        // every recurrent architecture: `state_size` is one key/value head's width,
        // `group_count` the number of key heads, `time_step_rank` the number of value
        // heads, and `inner_size` the full value width.
        let conv_kernel = need("ssm.conv_kernel")? as usize;
        let key_head_dim = need("ssm.state_size")? as usize;
        let n_k_heads = need("ssm.group_count")? as usize;
        let n_v_heads = need("ssm.time_step_rank")? as usize;
        let inner = need("ssm.inner_size")? as usize;

        if n_heads == 0 || n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
            return Err(GarudaError::Model(format!(
                "head_count {n_heads} must be a non-zero multiple of head_count_kv {n_kv_heads}"
            )));
        }
        if head_dim != value_length {
            return Err(GarudaError::Model(format!(
                "attention key_length {head_dim} and value_length {value_length} differ, \
                 which this backend does not implement"
            )));
        }
        if n_k_heads == 0 || n_v_heads == 0 || n_v_heads % n_k_heads != 0 {
            return Err(GarudaError::Model(format!(
                "ssm.time_step_rank {n_v_heads} must be a non-zero multiple of \
                 ssm.group_count {n_k_heads}"
            )));
        }
        if conv_kernel < 2 {
            return Err(GarudaError::Model(format!(
                "ssm.conv_kernel {conv_kernel} must be at least 2"
            )));
        }
        if inner % n_v_heads != 0 {
            return Err(GarudaError::Model(format!(
                "ssm.inner_size {inner} is not divisible by ssm.time_step_rank {n_v_heads}"
            )));
        }
        let value_head_dim = inner / n_v_heads;
        if value_head_dim != key_head_dim {
            return Err(GarudaError::Model(format!(
                "delta net key heads are {key_head_dim} wide and value heads \
                 {value_head_dim}; this backend implements the square case the \
                 published checkpoints use"
            )));
        }

        // Which blocks are recurrent. A checkpoint may list them outright; otherwise
        // the interval says it — the last block of each group attends, the rest do not.
        let full_attention_interval = g.arch_u64("full_attention_interval").unwrap_or(4) as usize;
        if full_attention_interval == 0 {
            return Err(GarudaError::Model(
                "qwen35.full_attention_interval must be non-zero".into(),
            ));
        }
        let recurrent = match g
            .get("qwen35.attention.recurrent_layers")
            .and_then(Value::as_array)
        {
            Some(list) if list.len() >= n_layers => (0..n_layers)
                .map(|l| list[l].as_bool().unwrap_or(false))
                .collect(),
            _ => (0..n_layers)
                .map(|l| (l + 1) % full_attention_interval != 0)
                .collect(),
        };

        Ok(Self {
            d_model,
            n_layers,
            d_ff,
            vocab: g
                .get("tokenizer.ggml.tokens")
                .and_then(Value::as_array)
                .map(<[Value]>::len)
                .ok_or_else(|| GarudaError::Model("gguf has no token list".into()))?,
            context: g.arch_u64("context_length").unwrap_or(4096) as usize,
            rms_eps: g
                .arch_f32("attention.layer_norm_rms_epsilon")
                .unwrap_or(1e-6),
            rope_theta: g.arch_f32("rope.freq_base").unwrap_or(10_000_000.0),
            n_rot: g
                .arch_u64("rope.dimension_count")
                .unwrap_or(head_dim as u64) as usize,
            n_heads,
            n_kv_heads,
            head_dim,
            conv_kernel,
            n_k_heads,
            n_v_heads,
            key_head_dim,
            value_head_dim,
            full_attention_interval,
            recurrent,
            n_nextn,
        })
    }

    /// Width of one stored key/value vector, in the blocks that store any.
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    /// Concatenated width of all attention heads. Wider than `d_model` for every
    /// published checkpoint — 6144 against 5120 for the 27B — which the output
    /// projection narrows back down.
    pub fn attn_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }

    /// Query, key and value width of a delta net block's joint projection.
    pub fn conv_dim(&self) -> usize {
        2 * self.n_k_heads * self.key_head_dim + self.n_v_heads * self.value_head_dim
    }

    pub fn key_dim(&self) -> usize {
        self.n_k_heads * self.key_head_dim
    }

    pub fn value_dim(&self) -> usize {
        self.n_v_heads * self.value_head_dim
    }

    /// Per-layer KV width: the real one for an attention block, zero for a recurrent
    /// block, which stores no keys or values at all.
    pub fn kv_dims(&self) -> Vec<usize> {
        self.recurrent
            .iter()
            .map(|&r| if r { 0 } else { self.kv_dim() })
            .collect()
    }

    /// Sizes of one recurrent block's state: the convolution history, then the
    /// matrix state.
    pub fn linear_state_shape(&self) -> (usize, usize) {
        (
            self.conv_dim() * (self.conv_kernel - 1),
            self.n_v_heads * self.key_head_dim * self.value_head_dim,
        )
    }

    /// Bytes of recurrent state one sequence carries, across every block. Constant in
    /// sequence length — 144 MB for the 27B — and the reason the prompt cache counts
    /// it (see [`SeqState::resident_bytes`]).
    pub fn linear_state_bytes(&self) -> usize {
        let (conv, state) = self.linear_state_shape();
        let blocks = self.recurrent.iter().filter(|&&r| r).count();
        blocks * (conv + state) * std::mem::size_of::<f32>()
    }

    /// The runtime-facing shape. The attention heads are wider than the residual
    /// stream, which [`ModelDims::validate`] permits; `n_experts`/`top_k` are unused
    /// by a dense model and set to the trivial 1/1.
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

/// One block's token mixer: grouped-query attention, or a gated delta net.
enum Mixer {
    Attn {
        /// Query *and* output gate, one after the other per head: `2 * head_dim` rows
        /// per head.
        wqg: Weight,
        wk: Weight,
        wv: Weight,
        wo: Weight,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
    },
    Linear {
        /// Joint query/key/value projection, `conv_dim` rows.
        wqkv: Weight,
        /// The output gate `z`, `value_dim` rows.
        wz: Weight,
        /// Depthwise causal convolution, `conv_dim` rows of `conv_kernel` taps.
        conv: Vec<f32>,
        /// Write strength, one row per value head.
        wbeta: Weight,
        /// Decay, one row per value head.
        walpha: Weight,
        /// Bias added before the softplus.
        dt_bias: Vec<f32>,
        /// `-exp(A_log)`, as the converter stores it: negative, so the decay it scales
        /// lands in `(0, 1)`.
        a: Vec<f32>,
        /// Gated RMSNorm weight, one value head wide.
        norm: Vec<f32>,
        wout: Weight,
    },
}

/// Where a layer's per-token state lives: one sequence's, or one sequence each.
///
/// A prefill drives many tokens of a single sequence (`Shared`); a decode step drives
/// one token of each of several (`PerToken`). Everything except the state lookup is
/// identical, which is what lets N sequences cost one pass over the weights instead of
/// N — the whole point of batching on a checkpoint larger than RAM.
enum Targets<'a, 'b> {
    Shared(&'a mut SeqState),
    PerToken(&'a mut [&'b mut SeqState]),
}

impl Targets<'_, '_> {
    fn get(&mut self, token: usize) -> &mut SeqState {
        match self {
            Targets::Shared(s) => s,
            Targets::PerToken(ss) => ss[token],
        }
    }
}

struct Layer {
    attn_norm: Vec<f32>,
    post_attn_norm: Vec<f32>,
    mixer: Mixer,
    ffn_gate: Weight,
    ffn_up: Weight,
    ffn_down: Weight,
}

pub struct Qwen35Backend {
    cfg: Qwen35Config,
    /// `Arc` because a checkpoint that ties its output head to the embeddings uses
    /// this same matrix for both.
    token_embd: Arc<Weight>,
    output_norm: Vec<f32>,
    output: Arc<Weight>,
    layers: Vec<Layer>,
    prefill_chunk: usize,
    /// Byte range of each block's weights in the mapped file, for prefetching.
    layer_spans: Vec<(usize, usize)>,
    /// Warms the next block while this one computes. Only useful with `mmap`.
    prefetch: Option<Arc<crate::prefetch::LayerPrefetcher>>,
}

impl Qwen35Backend {
    /// Load a checkpoint from a GGUF file's bytes, expanding weights to `f32` in RAM.
    pub fn load(bytes: &[u8]) -> Result<Self, GarudaError> {
        let g = Gguf::parse(bytes)?;
        Self::from_gguf(&g, bytes, None)
    }

    /// Load from an already-parsed GGUF header plus the file bytes.
    ///
    /// With `mmap`, projections stay packed in the mapped file and are dequantised a
    /// row at a time (low RAM, slower); without it every weight is expanded to `f32`.
    pub fn from_gguf(g: &Gguf, bytes: &[u8], mmap: Option<Arc<Mmap>>) -> Result<Self, GarudaError> {
        let cfg = Qwen35Config::from_gguf(g)?;
        let (d, f, v) = (cfg.d_model, cfg.d_ff, cfg.vocab);

        let norm = |name: &str, n: usize| load_norm(g, bytes, name, n);
        let weight =
            |name: &str, rows: usize, cols: usize| load_weight(g, bytes, &mmap, name, rows, cols);

        let token_embd = Arc::new(weight("token_embd.weight", v, d)?);
        let output_norm = norm("output_norm.weight", d)?;
        let output = if g.tensor("output.weight").is_some() {
            Arc::new(weight("output.weight", v, d)?)
        } else {
            token_embd.clone()
        };

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            let p = |name: &str| format!("blk.{l}.{name}.weight");
            let mixer = if cfg.recurrent[l] {
                Mixer::Linear {
                    wqkv: weight(&p("attn_qkv"), cfg.conv_dim(), d)?,
                    wz: weight(&p("attn_gate"), cfg.value_dim(), d)?,
                    // Channel-major: `conv_kernel` taps per row, one row per channel.
                    conv: norm(&p("ssm_conv1d"), cfg.conv_dim() * cfg.conv_kernel)?,
                    wbeta: weight(&p("ssm_beta"), cfg.n_v_heads, d)?,
                    walpha: weight(&p("ssm_alpha"), cfg.n_v_heads, d)?,
                    dt_bias: norm(&format!("blk.{l}.ssm_dt.bias"), cfg.n_v_heads)?,
                    a: norm(&format!("blk.{l}.ssm_a"), cfg.n_v_heads)?,
                    norm: norm(&p("ssm_norm"), cfg.value_head_dim)?,
                    wout: weight(&p("ssm_out"), d, cfg.value_dim())?,
                }
            } else {
                Mixer::Attn {
                    wqg: weight(&p("attn_q"), 2 * cfg.attn_dim(), d)?,
                    wk: weight(&p("attn_k"), cfg.kv_dim(), d)?,
                    wv: weight(&p("attn_v"), cfg.kv_dim(), d)?,
                    wo: weight(&p("attn_output"), d, cfg.attn_dim())?,
                    q_norm: norm(&p("attn_q_norm"), cfg.head_dim)?,
                    k_norm: norm(&p("attn_k_norm"), cfg.head_dim)?,
                }
            };
            layers.push(Layer {
                attn_norm: norm(&p("attn_norm"), d)?,
                post_attn_norm: norm(&p("post_attention_norm"), d)?,
                mixer,
                ffn_gate: weight(&p("ffn_gate"), f, d)?,
                ffn_up: weight(&p("ffn_up"), f, d)?,
                ffn_down: weight(&p("ffn_down"), d, f)?,
            });
        }

        // Where each block's weights live in the file. The converter writes a block's
        // tensors together, so this is one span per block rather than a scatter — which
        // is what makes warming it a large sequential read.
        let mut layer_spans = vec![(usize::MAX, 0usize); cfg.n_layers];
        for t in &g.tensors {
            let Some(rest) = t.name.strip_prefix("blk.") else {
                continue;
            };
            let Some((idx, _)) = rest.split_once('.') else {
                continue;
            };
            let Ok(l) = idx.parse::<usize>() else {
                continue;
            };
            if l >= cfg.n_layers {
                continue; // a prediction block, which this backend does not run
            }
            let start = g.data_offset + t.offset as usize;
            let len = quant::byte_size(t.ggml_type, t.n_elements() as usize).unwrap_or(0);
            let span = &mut layer_spans[l];
            span.0 = span.0.min(start);
            span.1 = span.1.max(start + len);
        }
        let layer_spans = layer_spans
            .into_iter()
            .map(|(start, end)| {
                if start == usize::MAX {
                    (0, 0)
                } else {
                    (start, end - start)
                }
            })
            .collect();

        Ok(Self {
            cfg,
            token_embd,
            output_norm,
            output,
            layers,
            prefill_chunk: 1,
            layer_spans,
            prefetch: None,
        })
    }

    /// The byte range of each block's weights in the mapped file.
    pub fn layer_spans(&self) -> &[(usize, usize)] {
        &self.layer_spans
    }

    /// Warm each block's weights on a background thread while the previous one
    /// computes. See [`crate::prefetch::LayerPrefetcher`] for why a dense model wants
    /// this and a resident one does not.
    pub fn with_prefetch(mut self, prefetch: Arc<crate::prefetch::LayerPrefetcher>) -> Self {
        self.prefetch = Some(prefetch);
        self
    }

    /// True when weights are kept packed in a memory-mapped file.
    pub fn is_mmapped(&self) -> bool {
        matches!(*self.token_embd, Weight::Packed { .. })
    }

    pub fn has_tied_embeddings(&self) -> bool {
        Arc::ptr_eq(&self.token_embd, &self.output)
    }

    /// Prompt tokens that share one pass over a layer's weights. See
    /// [`crate::llama::LlamaBackend::with_prefill_chunk`] for the tradeoff; it is the
    /// same one here, minus the expert grouping a dense model has no use for.
    pub fn with_prefill_chunk(mut self, chunk: usize) -> Self {
        self.prefill_chunk = chunk.max(1);
        self
    }

    pub fn config(&self) -> Qwen35Config {
        self.cfg.clone()
    }

    /// RMSNorm followed by an elementwise scale.
    fn norm(&self, x: &[f32], weight: &[f32]) -> Vec<f32> {
        let mut h = x.to_vec();
        simd::rmsnorm(&mut h, self.cfg.rms_eps);
        simd::mul_assign(&mut h, weight);
        h
    }

    /// One block over a batch of tokens that share a sequence.
    ///
    /// Everything except the mixer's own recurrence or attention read sees the whole
    /// batch at once, so a layer's weights are read and dequantised once for all of
    /// them — the same reason [`crate::llama`] does it this way.
    fn layer_batch(
        &self,
        l: usize,
        xs: &mut [Vec<f32>],
        mut targets: Targets<'_, '_>,
    ) -> Result<(), GarudaError> {
        let layer = &self.layers[l];
        let (d, n) = (self.cfg.d_model, xs.len());

        let mut hs = Vec::with_capacity(n * d);
        for x in xs.iter() {
            hs.extend_from_slice(&self.norm(x, &layer.attn_norm));
        }

        let mixed = match &layer.mixer {
            Mixer::Attn { .. } => self.attn_batch(l, layer, &hs, n, &mut targets)?,
            Mixer::Linear { .. } => self.delta_net_batch(l, layer, &hs, n, &mut targets)?,
        };

        let mut hs_ffn = Vec::with_capacity(n * d);
        for (i, x) in xs.iter_mut().enumerate() {
            simd::add_assign(x, &mixed[i * d..(i + 1) * d]);
            hs_ffn.extend_from_slice(&self.norm(x, &layer.post_attn_norm));
        }

        let ffn = self.feed_forward_batch(layer, &hs_ffn, n)?;
        for (i, x) in xs.iter_mut().enumerate() {
            simd::add_assign(x, &ffn[i * d..(i + 1) * d]);
        }
        Ok(())
    }

    /// Dense SwiGLU: `down(silu(gate(x)) * up(x))`, batched.
    fn feed_forward_batch(
        &self,
        layer: &Layer,
        hs: &[f32],
        n: usize,
    ) -> Result<Vec<f32>, GarudaError> {
        let (d, f) = (self.cfg.d_model, self.cfg.d_ff);
        let mut gate = vec![0.0; n * f];
        let mut up = vec![0.0; n * f];
        layer.ffn_gate.matmul_rows(0, hs, n, &mut gate)?;
        layer.ffn_up.matmul_rows(0, hs, n, &mut up)?;
        simd::silu(&mut gate);
        simd::mul_assign(&mut gate, &up);
        let mut out = vec![0.0; n * d];
        layer.ffn_down.matmul_rows(0, &gate, n, &mut out)?;
        Ok(out)
    }

    /// Grouped-query attention with a per-head output gate, per-head query/key norms
    /// and partial rotation.
    fn attn_batch(
        &self,
        l: usize,
        layer: &Layer,
        hs: &[f32],
        n: usize,
        targets: &mut Targets<'_, '_>,
    ) -> Result<Vec<f32>, GarudaError> {
        let Mixer::Attn {
            wqg,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
        } = &layer.mixer
        else {
            return Err(GarudaError::Inference("not an attention block".into()));
        };
        let cfg = &self.cfg;
        let (d, hd, a_dim, kv_dim) = (cfg.d_model, cfg.head_dim, cfg.attn_dim(), cfg.kv_dim());

        let mut qg = vec![0.0; n * 2 * a_dim];
        let mut k = vec![0.0; n * kv_dim];
        let mut v = vec![0.0; n * kv_dim];
        wqg.matmul_rows(0, hs, n, &mut qg)?;
        wk.matmul_rows(0, hs, n, &mut k)?;
        wv.matmul_rows(0, hs, n, &mut v)?;

        let group = cfg.n_heads / cfg.n_kv_heads;
        let scale = 1.0 / (hd as f32).sqrt();
        // The gated attention output of every token in the batch, so the output
        // projection is read once for all of them rather than once each.
        let mut contexts = vec![0.0; n * a_dim];

        for i in 0..n {
            // The query projection emits `[query | gate]` for each head in turn.
            let row = &qg[i * 2 * a_dim..(i + 1) * 2 * a_dim];
            let mut q = vec![0.0; a_dim];
            let mut gate = vec![0.0; a_dim];
            for h in 0..cfg.n_heads {
                let src = &row[h * 2 * hd..(h + 1) * 2 * hd];
                q[h * hd..(h + 1) * hd].copy_from_slice(&src[..hd]);
                gate[h * hd..(h + 1) * hd].copy_from_slice(&src[hd..]);
            }

            let k_i = &mut k[i * kv_dim..(i + 1) * kv_dim];
            let v_i = &v[i * kv_dim..(i + 1) * kv_dim];

            // Attention is causal, so a token's keys and values have to be in its own
            // sequence's cache before anything attends to them. `pos` comes from that
            // cache, which is what keeps a batch of prompt tokens in the right order.
            let kv = targets.get(i).layer(l);
            let pos = kv.len();
            for h in 0..cfg.n_heads {
                let head = &mut q[h * hd..(h + 1) * hd];
                simd::rmsnorm(head, cfg.rms_eps);
                simd::mul_assign(head, q_norm);
                rope_partial(head, pos, cfg.rope_theta, cfg.n_rot);
            }
            for h in 0..cfg.n_kv_heads {
                let head = &mut k_i[h * hd..(h + 1) * hd];
                simd::rmsnorm(head, cfg.rms_eps);
                simd::mul_assign(head, k_norm);
                rope_partial(head, pos, cfg.rope_theta, cfg.n_rot);
            }

            kv.append(k_i, v_i)?;
            let start = kv.attention_start();
            let end = pos + 1;
            kv.ensure_resident(start, end)?;

            let context = &mut contexts[i * a_dim..(i + 1) * a_dim];
            for h in 0..cfg.n_heads {
                let q_h = &q[h * hd..(h + 1) * hd];
                let kr = (h / group) * hd..(h / group + 1) * hd;

                let mut scores = Vec::with_capacity(end - start);
                for j in start..end {
                    let key = kv
                        .key_at(j)
                        .ok_or_else(|| GarudaError::Cache(format!("missing key at {j}")))?;
                    scores.push(simd::dot(q_h, &key[kr.clone()]) * scale);
                }
                simd::softmax(&mut scores);

                let out_h = &mut context[h * hd..(h + 1) * hd];
                for (j, &p) in (start..end).zip(scores.iter()) {
                    let val = kv
                        .value_at(j)
                        .ok_or_else(|| GarudaError::Cache(format!("missing value at {j}")))?;
                    simd::add_scaled(out_h, &val[kr.clone()], p);
                }
            }

            // The gate decides, per dimension, how much of the attention output
            // survives — the "output gate" the config names.
            for (c, g) in context.iter_mut().zip(gate.iter()) {
                *c *= sigmoid(*g);
            }
        }

        let mut out = vec![0.0; n * d];
        wo.matmul_rows(0, &contexts, n, &mut out)?;
        Ok(out)
    }

    /// A gated delta net block over a batch of tokens.
    ///
    /// The projections and the output are batched; the convolution and the delta rule
    /// walk the batch in order, because both carry state from one token to the next.
    fn delta_net_batch(
        &self,
        l: usize,
        layer: &Layer,
        hs: &[f32],
        n: usize,
        targets: &mut Targets<'_, '_>,
    ) -> Result<Vec<f32>, GarudaError> {
        let Mixer::Linear {
            wqkv,
            wz,
            conv,
            wbeta,
            walpha,
            dt_bias,
            a,
            norm,
            wout,
        } = &layer.mixer
        else {
            return Err(GarudaError::Inference("not a delta net block".into()));
        };
        let cfg = &self.cfg;
        let (d, k_dim, v_dim, c_dim) =
            (cfg.d_model, cfg.key_dim(), cfg.value_dim(), cfg.conv_dim());
        let (kh, vh) = (cfg.key_head_dim, cfg.value_head_dim);
        let kernel = cfg.conv_kernel;

        let mut qkv = vec![0.0; n * c_dim];
        let mut z = vec![0.0; n * v_dim];
        let mut beta = vec![0.0; n * cfg.n_v_heads];
        let mut alpha = vec![0.0; n * cfg.n_v_heads];
        wqkv.matmul_rows(0, hs, n, &mut qkv)?;
        wz.matmul_rows(0, hs, n, &mut z)?;
        wbeta.matmul_rows(0, hs, n, &mut beta)?;
        walpha.matmul_rows(0, hs, n, &mut alpha)?;

        let (conv_len, state_len) = cfg.linear_state_shape();
        let mut mixed = vec![0.0; n * v_dim];

        for i in 0..n {
            let raw = &qkv[i * c_dim..(i + 1) * c_dim];
            // This token's own sequence: its convolution history and its state.
            let seq = targets.get(i);
            let ls: &mut LinearState = seq.linear(l, conv_len, state_len)?;

            // Depthwise causal convolution over the last `kernel` inputs of each
            // channel, then SiLU. `ls.conv` holds the `kernel - 1` previous ones,
            // oldest first.
            let mut y = vec![0.0; c_dim];
            for (c, (yc, &now)) in y.iter_mut().zip(raw.iter()).enumerate() {
                let taps = &conv[c * kernel..(c + 1) * kernel];
                let hist = &ls.conv[c * (kernel - 1)..(c + 1) * (kernel - 1)];
                let mut acc = taps[kernel - 1] * now;
                for (tap, &h) in taps.iter().zip(hist.iter()) {
                    acc += tap * h;
                }
                *yc = acc;
            }
            simd::silu(&mut y);
            for (c, &now) in raw.iter().enumerate() {
                let hist = &mut ls.conv[c * (kernel - 1)..(c + 1) * (kernel - 1)];
                hist.rotate_left(1);
                hist[kernel - 2] = now;
            }

            let (q, rest) = y.split_at(k_dim);
            let (k, v) = rest.split_at(k_dim);

            // L2-normalise every key and query head, as the kernel this was trained
            // with does before the recurrence.
            let mut qn = q.to_vec();
            let mut kn = k.to_vec();
            for h in 0..cfg.n_k_heads {
                l2_normalize(&mut qn[h * kh..(h + 1) * kh]);
                l2_normalize(&mut kn[h * kh..(h + 1) * kh]);
            }

            let q_scale = 1.0 / (kh as f32).sqrt();
            let mut head_out = vec![0.0; v_dim];
            for hv in 0..cfg.n_v_heads {
                let hk = key_head_of(hv, cfg.n_k_heads);
                let q_h = &qn[hk * kh..(hk + 1) * kh];
                let k_h = &kn[hk * kh..(hk + 1) * kh];
                let v_h = &v[hv * vh..(hv + 1) * vh];

                let g = a[hv] * softplus(alpha[i * cfg.n_v_heads + hv] + dt_bias[hv]);
                let decay = g.exp();
                let b = sigmoid(beta[i * cfg.n_v_heads + hv]);

                let s = &mut ls.state[hv * kh * vh..(hv + 1) * kh * vh];
                let out_h = &mut head_out[hv * vh..(hv + 1) * vh];
                let gates = DeltaGates {
                    decay,
                    beta: b,
                    q_scale,
                };
                delta_step(s, q_h, k_h, v_h, gates, out_h);
            }

            // Gated RMSNorm, per head: normalise, scale, then gate by silu(z).
            let z_i = &z[i * v_dim..(i + 1) * v_dim];
            for hv in 0..cfg.n_v_heads {
                let out_h = &mut head_out[hv * vh..(hv + 1) * vh];
                simd::rmsnorm(out_h, cfg.rms_eps);
                simd::mul_assign(out_h, norm);
                let mut gate = z_i[hv * vh..(hv + 1) * vh].to_vec();
                simd::silu(&mut gate);
                simd::mul_assign(out_h, &gate);
            }
            mixed[i * v_dim..(i + 1) * v_dim].copy_from_slice(&head_out);

            // A recurrent block stores nothing per position, but it still counts them:
            // the runtime reads `seq.len()` from layer 0 and every layer has to agree.
            targets.get(i).layer(l).append(&[], &[])?;
        }

        let mut out = vec![0.0; n * d];
        wout.matmul_rows(0, &mixed, n, &mut out)?;
        Ok(out)
    }

    /// The hidden states of the last `n_last` positions this call consumes, normed by
    /// the final output norm and ready for the head.
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
                "sequence has {} layers but the model has {}",
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
        let new = &context[already..];
        if new.is_empty() {
            return Err(GarudaError::Inference(
                "no new tokens to process for this context".into(),
            ));
        }
        let capacity = seq.max_positions();
        if already + new.len() > capacity {
            return Err(GarudaError::Cache(format!(
                "{} tokens do not fit the {capacity}-position context window ({already} used)",
                new.len()
            )));
        }

        let mut tail: Vec<Vec<f32>> = Vec::new();
        for chunk in new.chunks(self.prefill_chunk.max(1)) {
            let mut xs: Vec<Vec<f32>> = Vec::with_capacity(chunk.len());
            for &token in chunk {
                let idx = token as usize;
                if idx >= self.cfg.vocab {
                    return Err(GarudaError::InvalidToken(token));
                }
                xs.push(self.token_embd.row(idx)?);
            }
            for l in 0..self.cfg.n_layers {
                // Ask for the blocks ahead before running this one, so the reads and
                // the arithmetic overlap instead of taking turns. More than one, so
                // the drive has more than one request to work on.
                if let Some(pf) = &self.prefetch {
                    pf.hint_ahead(l);
                }
                self.layer_batch(l, &mut xs, Targets::Shared(seq))?;
            }
            tail.append(&mut xs);
            if tail.len() > n_last {
                tail.drain(..tail.len() - n_last);
            }
        }

        if n_last > tail.len() {
            return Err(GarudaError::Inference(format!(
                "asked for the last {n_last} positions but only {} were computed in this pass",
                tail.len()
            )));
        }
        tail.drain(..tail.len() - n_last);
        for x in tail.iter_mut() {
            simd::rmsnorm(x, self.cfg.rms_eps);
            simd::mul_assign(x, &self.output_norm);
        }
        Ok(tail)
    }
}

impl InferenceBackend for Qwen35Backend {
    fn dims(&self) -> ModelDims {
        self.cfg.model_dims()
    }

    fn hidden(&self, context: &[Token], seq: &mut SeqState) -> Result<Tensor, GarudaError> {
        let mut tail = self.forward_tail(context, seq, 1)?;
        let last = tail.pop().expect("forward_tail returns one state");
        Tensor::new(vec![self.cfg.d_model], last)
    }

    fn logits(&self, context: &[Token], seq: &mut SeqState) -> Result<Tensor, GarudaError> {
        let h = self.hidden(context, seq)?;
        let mut logits = vec![0.0; self.cfg.vocab];
        self.output.matvec(h.data(), &mut logits)?;
        Tensor::new(vec![self.cfg.vocab], logits)
    }

    /// One decode step across several sequences, sharing one pass over the weights.
    ///
    /// Only the mixer's state lookup is per sequence; the norms, the projections and
    /// the feed-forward see the whole batch. On a checkpoint larger than RAM that is
    /// the difference between streaming 19 GB once per token and once per token *per
    /// caller* — a single sequence decoding alone has no way to amortise it.
    ///
    /// Falls back to one call each unless every sequence contributes exactly one new
    /// token, which is the shape a decode step has; a prefill does not qualify and does
    /// not need to.
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

        // Reject before touching any state, so a batch cannot be left half-advanced —
        // and for a recurrent layer that matters more than for a cache, because there
        // is no truncation that would put it back.
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
            if let Some(pf) = &self.prefetch {
                pf.hint_ahead(l);
            }
            self.layer_batch(l, &mut xs, Targets::PerToken(seqs))?;
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

    /// Off for this architecture, deliberately.
    ///
    /// Verifying a run of guesses means being able to throw the rejected ones away.
    /// Three quarters of these blocks keep a recurrent state that summarises every
    /// token it has read, so there is nothing to throw away — the state cannot be
    /// rewound. Answering `false` here keeps the runtime on the plain decode path
    /// instead of producing a state that describes tokens the caller discarded.
    fn speculation_supported(&self) -> bool {
        false
    }
}

/// One token's scalars for one delta net head: how much of the state survives, how
/// hard this token writes, and the query scale.
#[derive(Debug, Clone, Copy)]
struct DeltaGates {
    decay: f32,
    beta: f32,
    q_scale: f32,
}

/// One head's gated delta rule for one token, in place.
///
/// `s` is the head's `key_head_dim x value_head_dim` state, row-major with the key
/// dimension outermost, so `s[r * vh + j]` couples key component `r` to value
/// component `j`. With decay `α`, write strength `β` and the query scale `q_scale`:
///
/// ```text
/// S ← αS + β·k ⊗ (v − Sᵀk)      out = Sᵀ(q · q_scale)
/// ```
///
/// The zero checks are not micro-optimisation for its own sake: `k` and `q` are L2
/// normalised, so a head that has seen nothing contributes exact zeros, and skipping
/// those rows keeps a 27B model's 48 heads per block from doing 128 pointless passes
/// over a 128-wide row each.
fn delta_step(s: &mut [f32], q: &[f32], k: &[f32], v: &[f32], g: DeltaGates, out: &mut [f32]) {
    let DeltaGates {
        decay,
        beta,
        q_scale,
    } = g;
    let vh = v.len();
    for x in s.iter_mut() {
        *x *= decay;
    }
    // What the state currently predicts for this key, Sᵀk.
    let mut mem = vec![0.0; vh];
    for (r, &kr) in k.iter().enumerate() {
        if kr != 0.0 {
            simd::add_scaled(&mut mem, &s[r * vh..(r + 1) * vh], kr);
        }
    }
    // The write: β(v − Sᵀk), along k.
    let mut delta = vec![0.0; vh];
    for (j, d) in delta.iter_mut().enumerate() {
        *d = (v[j] - mem[j]) * beta;
    }
    for (r, &kr) in k.iter().enumerate() {
        if kr != 0.0 {
            simd::add_scaled(&mut s[r * vh..(r + 1) * vh], &delta, kr);
        }
    }
    // The read: Sᵀ(q · q_scale).
    for (r, &qr) in q.iter().enumerate() {
        let qr = qr * q_scale;
        if qr != 0.0 {
            simd::add_scaled(out, &s[r * vh..(r + 1) * vh], qr);
        }
    }
}

/// Rotary embedding over the first `n_rot` dimensions of one head, rotate-half
/// (GPT-NeoX) style, leaving the rest untouched.
///
/// Qwen3.5 rotates a quarter of each 256-wide head. The pairs are `(i, i + n_rot/2)`,
/// not `(2i, 2i+1)` — a different convention from [`simd::rope`], and using the wrong
/// one produces fluent text that ignores word order.
fn rope_partial(head: &mut [f32], pos: usize, theta: f32, n_rot: usize) {
    let rot = n_rot.min(head.len()) / 2 * 2;
    if rot == 0 {
        return;
    }
    let half = rot / 2;
    for i in 0..half {
        let freq = 1.0 / theta.powf(2.0 * i as f32 / rot as f32);
        let (sin, cos) = (pos as f32 * freq).sin_cos();
        let (a, b) = (head[i], head[i + half]);
        head[i] = a * cos - b * sin;
        head[i + half] = b * cos + a * sin;
    }
}

/// `x / sqrt(sum(x²) + eps)`, the L2 norm the delta net kernel applies to queries and
/// keys. The epsilon matches the reference implementation's, which is fixed at 1e-6
/// rather than taken from the checkpoint.
fn l2_normalize(x: &mut [f32]) {
    let sum: f32 = x.iter().map(|v| v * v).sum();
    let inv = (sum + 1e-6).sqrt().recip();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `ln(1 + eˣ)`, evaluated so that a large `x` does not overflow the exponential.
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::KvConfig;
    use std::io::Write;

    // ---- a minimal GGUF writer, to build a synthetic qwen35 model with real data ----
    //
    // The dimensions are tiny but the *shapes* are the real ones, including the two
    // that a Llama-shaped loader gets wrong: attention heads wider than the residual
    // stream (4 x 16 = 64 against d_model 32), and a query projection that emits a
    // gate next to every head's query.

    const D: usize = 32;
    const N_LAYERS: usize = 4;
    const D_FF: usize = 32;
    const VOCAB: usize = 64;
    const N_HEADS: usize = 4;
    const N_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 16;
    const N_ROT: usize = 4;
    const CONV_KERNEL: usize = 4;
    const KEY_HEAD_DIM: usize = 8;
    const N_K_HEADS: usize = 2;
    const N_V_HEADS: usize = 4;
    const INNER: usize = N_V_HEADS * KEY_HEAD_DIM;
    const CONV_DIM: usize = 2 * N_K_HEADS * KEY_HEAD_DIM + INNER;
    const ATTN_DIM: usize = N_HEADS * HEAD_DIM;
    const KV_DIM: usize = N_KV_HEADS * HEAD_DIM;

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

    #[derive(Clone, Copy, PartialEq)]
    enum Head {
        Separate,
        Tied,
    }

    fn build_qwen35_gguf(head: Head, arch: &str) -> Vec<u8> {
        let mut tensors: Vec<(String, Vec<u64>, Vec<f32>)> = Vec::new();
        let mut seed = 1usize;
        let add = |name: String, dims: Vec<u64>, tensors: &mut Vec<_>, seed: &mut usize| {
            let n: usize = dims.iter().product::<u64>() as usize;
            let data = if name.contains("norm") {
                vec![1.0; n]
            } else if name.ends_with("ssm_a") {
                // The converter stores -exp(A_log), so these are negative and the
                // decay they scale lands in (0, 1). A positive value here would make
                // the recurrence diverge, which is exactly the bug worth pinning.
                r#gen(*seed, n).iter().map(|v| -(v.abs()) - 0.01).collect()
            } else if name.ends_with("ssm_dt.bias") {
                vec![0.5; n]
            } else {
                r#gen(*seed, n)
            };
            *seed += 1;
            tensors.push((name, dims, data));
        };

        add(
            "token_embd.weight".into(),
            vec![D as u64, VOCAB as u64],
            &mut tensors,
            &mut seed,
        );
        add(
            "output_norm.weight".into(),
            vec![D as u64],
            &mut tensors,
            &mut seed,
        );
        if head == Head::Separate {
            add(
                "output.weight".into(),
                vec![D as u64, VOCAB as u64],
                &mut tensors,
                &mut seed,
            );
        }

        for l in 0..N_LAYERS {
            let p = |n: &str| format!("blk.{l}.{n}.weight");
            add(p("attn_norm"), vec![D as u64], &mut tensors, &mut seed);
            add(
                p("post_attention_norm"),
                vec![D as u64],
                &mut tensors,
                &mut seed,
            );

            // Three blocks in four are recurrent; the last of each group attends.
            let recurrent = (l + 1) % 4 != 0;
            if recurrent {
                for (name, dims) in [
                    (p("attn_qkv"), vec![D as u64, CONV_DIM as u64]),
                    (p("attn_gate"), vec![D as u64, INNER as u64]),
                    (p("ssm_conv1d"), vec![CONV_KERNEL as u64, CONV_DIM as u64]),
                    (p("ssm_beta"), vec![D as u64, N_V_HEADS as u64]),
                    (p("ssm_alpha"), vec![D as u64, N_V_HEADS as u64]),
                    (format!("blk.{l}.ssm_dt.bias"), vec![N_V_HEADS as u64]),
                    (format!("blk.{l}.ssm_a"), vec![N_V_HEADS as u64]),
                    (p("ssm_norm"), vec![KEY_HEAD_DIM as u64]),
                    (p("ssm_out"), vec![INNER as u64, D as u64]),
                ] {
                    add(name, dims, &mut tensors, &mut seed);
                }
            } else {
                for (name, dims) in [
                    (p("attn_q"), vec![D as u64, 2 * ATTN_DIM as u64]),
                    (p("attn_k"), vec![D as u64, KV_DIM as u64]),
                    (p("attn_v"), vec![D as u64, KV_DIM as u64]),
                    (p("attn_output"), vec![ATTN_DIM as u64, D as u64]),
                    (p("attn_q_norm"), vec![HEAD_DIM as u64]),
                    (p("attn_k_norm"), vec![HEAD_DIM as u64]),
                ] {
                    add(name, dims, &mut tensors, &mut seed);
                }
            }

            for (name, dims) in [
                (p("ffn_gate"), vec![D as u64, D_FF as u64]),
                (p("ffn_up"), vec![D as u64, D_FF as u64]),
                (p("ffn_down"), vec![D_FF as u64, D as u64]),
            ] {
                add(name, dims, &mut tensors, &mut seed);
            }
        }

        let mut meta = Vec::new();
        let mut kv_count = 0u64;
        macro_rules! m {
            ($f:expr_2021) => {{
                $f;
                kv_count += 1;
            }};
        }
        m!(kv_str(&mut meta, "general.architecture", arch));
        m!(kv_u32(&mut meta, "qwen35.embedding_length", D as u32));
        m!(kv_u32(&mut meta, "qwen35.block_count", N_LAYERS as u32));
        m!(kv_u32(
            &mut meta,
            "qwen35.attention.head_count",
            N_HEADS as u32
        ));
        m!(kv_u32(
            &mut meta,
            "qwen35.attention.head_count_kv",
            N_KV_HEADS as u32
        ));
        m!(kv_u32(
            &mut meta,
            "qwen35.attention.key_length",
            HEAD_DIM as u32
        ));
        m!(kv_u32(
            &mut meta,
            "qwen35.attention.value_length",
            HEAD_DIM as u32
        ));
        m!(kv_u32(&mut meta, "qwen35.feed_forward_length", D_FF as u32));
        m!(kv_u32(&mut meta, "qwen35.context_length", 64));
        m!(kv_f32(
            &mut meta,
            "qwen35.attention.layer_norm_rms_epsilon",
            1e-6
        ));
        m!(kv_f32(&mut meta, "qwen35.rope.freq_base", 10_000_000.0));
        m!(kv_u32(
            &mut meta,
            "qwen35.rope.dimension_count",
            N_ROT as u32
        ));
        m!(kv_u32(
            &mut meta,
            "qwen35.ssm.conv_kernel",
            CONV_KERNEL as u32
        ));
        m!(kv_u32(
            &mut meta,
            "qwen35.ssm.state_size",
            KEY_HEAD_DIM as u32
        ));
        m!(kv_u32(
            &mut meta,
            "qwen35.ssm.group_count",
            N_K_HEADS as u32
        ));
        m!(kv_u32(
            &mut meta,
            "qwen35.ssm.time_step_rank",
            N_V_HEADS as u32
        ));
        m!(kv_u32(&mut meta, "qwen35.ssm.inner_size", INNER as u32));
        m!(kv_u32(&mut meta, "qwen35.full_attention_interval", 4));
        let toks: Vec<String> = (0..VOCAB).map(|i| format!("t{i}")).collect();
        m!(kv_str_array(&mut meta, "tokenizer.ggml.tokens", &toks));

        let align = |x: usize| x.next_multiple_of(32);
        let mut offsets = Vec::new();
        let mut cursor = 0usize;
        for (_, _, data) in &tensors {
            let off = align(cursor);
            offsets.push(off as u64);
            cursor = off + data.len() * 4;
        }
        let data_size = align(cursor);

        let mut infos = Vec::new();
        for ((name, dims, _), &off) in tensors.iter().zip(&offsets) {
            put_str(&mut infos, name);
            infos.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for &dim in dims {
                infos.extend_from_slice(&dim.to_le_bytes());
            }
            infos.extend_from_slice(&0u32.to_le_bytes()); // F32
            infos.extend_from_slice(&off.to_le_bytes());
        }

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

    fn seq_for(b: &Qwen35Backend) -> SeqState {
        let cfg = b.config();
        SeqState::new(
            KvConfig {
                dims: b.dims(),
                kv_dim: cfg.kv_dim(),
                n_layers: cfg.n_layers,
                kv_dims: Some(cfg.kv_dims()),
                max_positions: 64,
                max_resident_blocks: 64,
                sliding_window: None,
                storage: None,
            },
            1,
        )
    }

    fn backend(head: Head) -> Qwen35Backend {
        Qwen35Backend::load(&build_qwen35_gguf(head, "qwen35")).unwrap()
    }

    #[test]
    fn a_hybrid_checkpoint_loads_and_answers() {
        let b = backend(Head::Separate);
        let cfg = b.config();

        assert_eq!(cfg.n_layers, N_LAYERS);
        assert_eq!(cfg.recurrent, vec![true, true, true, false]);
        assert_eq!(cfg.attn_dim(), ATTN_DIM);
        assert!(cfg.attn_dim() > cfg.d_model, "heads are wider than d_model");
        assert_eq!(cfg.conv_dim(), CONV_DIM);
        assert_eq!(cfg.value_head_dim, KEY_HEAD_DIM);
        assert_eq!(cfg.kv_dims(), vec![0, 0, 0, KV_DIM]);
        b.dims().validate().unwrap();

        let mut seq = seq_for(&b);
        let logits = b.logits(&[1, 2, 3], &mut seq).unwrap();
        assert_eq!(logits.len(), VOCAB);
        assert!(
            logits.data().iter().all(|v| v.is_finite()),
            "a diverging recurrence shows up here first"
        );

        // Same input, same answer: the prompt cache relies on it.
        let mut again = seq_for(&b);
        let repeat = b.logits(&[1, 2, 3], &mut again).unwrap();
        assert_eq!(logits.data(), repeat.data());
    }

    /// Invariant 2 of the backend contract, which a hybrid model is the first to be
    /// able to break: the recurrent blocks store no keys or values, but they still
    /// have to count positions, or `seq.len()` (which reads layer 0) stops speaking
    /// for the rest.
    #[test]
    fn every_layer_advances_one_position_per_token() {
        let b = backend(Head::Separate);
        let mut seq = seq_for(&b);

        for n in 1..=5 {
            let ctx: Vec<Token> = (0..n as Token).collect();
            b.logits(&ctx, &mut seq).unwrap();
            assert_eq!(seq.len(), n);
            for l in 0..N_LAYERS {
                assert_eq!(seq.layer(l).len(), n, "layer {l} fell behind");
            }
        }

        // The recurrent blocks hold a fixed-size state and no per-position bytes; the
        // attention block holds keys and values.
        for l in 0..3 {
            assert_eq!(seq.layer(l).resident_bytes(), 0, "layer {l} stored vectors");
        }
        assert!(seq.layer(3).resident_bytes() > 0);
        assert!(seq.has_linear_state());
    }

    #[test]
    fn recurrent_state_is_sized_by_the_config_and_reused() {
        let b = backend(Head::Separate);
        let cfg = b.config();
        let (conv, state) = cfg.linear_state_shape();
        assert_eq!(conv, CONV_DIM * (CONV_KERNEL - 1));
        assert_eq!(state, N_V_HEADS * KEY_HEAD_DIM * KEY_HEAD_DIM);

        let mut seq = seq_for(&b);
        b.logits(&[1, 2], &mut seq).unwrap();
        let before = seq.resident_bytes();
        let s0 = seq.linear(0, conv, state).unwrap().state.clone();
        assert!(s0.iter().any(|v| *v != 0.0), "the state never got written");

        // Asking for a different shape is a bug, not a silent reallocation that would
        // throw away everything the sequence has read.
        assert!(seq.linear(0, conv + 1, state).is_err());
        assert_eq!(seq.linear(0, conv, state).unwrap().state, s0);

        // Recurrent state does not grow with the sequence: only the attention block's
        // keys and values do.
        b.logits(&[1, 2, 3, 4, 5, 6], &mut seq).unwrap();
        let growth = seq.resident_bytes() - before;
        assert_eq!(growth, 4 * KV_DIM * 2 * std::mem::size_of::<f32>());
    }

    /// A batch of prompt tokens must produce exactly what feeding them one at a time
    /// does. For the delta net that is a real risk: its state carries from token to
    /// token, so a batched projection that lost the order would still return numbers.
    #[test]
    fn a_batched_prefill_matches_feeding_tokens_one_at_a_time() {
        let bytes = build_qwen35_gguf(Head::Separate, "qwen35");
        let stepwise = Qwen35Backend::load(&bytes).unwrap().with_prefill_chunk(1);
        let batched = Qwen35Backend::load(&bytes).unwrap().with_prefill_chunk(8);

        let ctx: Vec<Token> = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
        let mut a = seq_for(&stepwise);
        let mut b = seq_for(&batched);
        let one = stepwise.logits(&ctx, &mut a).unwrap();
        let many = batched.logits(&ctx, &mut b).unwrap();

        for (i, (x, y)) in one.data().iter().zip(many.data()).enumerate() {
            assert!(
                (x - y).abs() < 1e-4,
                "logit {i} differs: stepwise {x} vs batched {y}"
            );
        }
    }

    /// The delta rule against the recurrence written out longhand, straight from the
    /// reference implementation's loop.
    #[test]
    fn the_delta_rule_matches_a_naive_reference() {
        let (kh, vh) = (8usize, 8usize);
        let q = r#gen(11, kh);
        let k = r#gen(12, kh);
        let v = r#gen(13, vh);
        let gates = DeltaGates {
            decay: 0.7,
            beta: 0.3,
            q_scale: 1.0 / (kh as f32).sqrt(),
        };

        let mut s = r#gen(14, kh * vh);
        let mut got = vec![0.0; vh];
        delta_step(&mut s, &q, &k, &v, gates, &mut got);

        // Reference: S ← αS; mem = Sᵀk; S += k ⊗ β(v − mem); out = Sᵀ(q·scale).
        let mut want_s = r#gen(14, kh * vh);
        for x in want_s.iter_mut() {
            *x *= gates.decay;
        }
        let mut mem = vec![0.0; vh];
        for j in 0..vh {
            for r in 0..kh {
                mem[j] += want_s[r * vh + j] * k[r];
            }
        }
        for j in 0..vh {
            let d = (v[j] - mem[j]) * gates.beta;
            for r in 0..kh {
                want_s[r * vh + j] += k[r] * d;
            }
        }
        let mut want = vec![0.0; vh];
        for j in 0..vh {
            for r in 0..kh {
                want[j] += want_s[r * vh + j] * q[r] * gates.q_scale;
            }
        }

        for (i, (x, y)) in got.iter().zip(&want).enumerate() {
            assert!((x - y).abs() < 1e-5, "output {i}: {x} vs {y}");
        }
        for (i, (x, y)) in s.iter().zip(&want_s).enumerate() {
            assert!((x - y).abs() < 1e-5, "state {i}: {x} vs {y}");
        }
    }

    /// A decay of one and a write strength of one make the state a plain sum of outer
    /// products, which is easy to reason about — and shows the rule is writing the
    /// *error*, not the value.
    #[test]
    fn a_second_write_of_the_same_key_corrects_rather_than_accumulates() {
        let (kh, vh) = (4usize, 4usize);
        let k = vec![1.0, 0.0, 0.0, 0.0];
        let v = vec![2.0, 0.0, 0.0, 0.0];
        let gates = DeltaGates {
            decay: 1.0,
            beta: 1.0,
            q_scale: 1.0,
        };

        let mut s = vec![0.0; kh * vh];
        let mut out = vec![0.0; vh];
        delta_step(&mut s, &k, &k, &v, gates, &mut out);
        assert!(
            (out[0] - 2.0).abs() < 1e-6,
            "first read returns what was written"
        );

        // Writing the same key and value again changes nothing: the error is zero.
        let before = s.clone();
        let mut out2 = vec![0.0; vh];
        delta_step(&mut s, &k, &k, &v, gates, &mut out2);
        assert_eq!(s, before);
        assert!((out2[0] - 2.0).abs() < 1e-6);
    }

    /// The value-to-key head pairing, pinned because it is a file convention that
    /// looks like a free choice: the two reference implementations write it two
    /// different ways, only one matches a GGUF's head order, and a checkpoint with as
    /// many value heads as key heads cannot tell them apart. See `key_head_of`.
    #[test]
    fn value_heads_read_key_heads_in_the_order_the_file_stores_them() {
        // 48 value heads over 16 key heads, as Qwen3.8-27B has: the key-head block
        // repeats, so consecutive value heads read *different* key heads.
        let got: Vec<usize> = (0..6).map(|hv| key_head_of(hv, 16)).collect();
        assert_eq!(got, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(key_head_of(16, 16), 0, "the block repeats after n_k_heads");
        assert_eq!(key_head_of(47, 16), 15);

        // The alternative — each key head serving a contiguous run — would give
        // 0, 0, 0, 1, 1, 1. It is what `transformers` does in its own layout, and the
        // wrong answer here.
        assert_ne!(got, vec![0, 0, 0, 1, 1, 1]);

        // With as many value heads as key heads, which is what the small checkpoints
        // have, both conventions are the identity — hence the real-weights test.
        assert_eq!(
            (0..16).map(|h| key_head_of(h, 16)).collect::<Vec<_>>(),
            (0..16).collect::<Vec<_>>()
        );
    }

    #[test]
    fn partial_rotation_touches_only_the_first_quarter_of_a_head() {
        let head_dim = 16;
        let n_rot = 4;
        let base: Vec<f32> = (0..head_dim).map(|i| 1.0 + i as f32).collect();

        // Position zero is the identity, whatever the rotation width.
        let mut at_zero = base.clone();
        rope_partial(&mut at_zero, 0, 10_000_000.0, n_rot);
        assert_eq!(at_zero, base);

        let mut rotated = base.clone();
        rope_partial(&mut rotated, 7, 10_000_000.0, n_rot);
        assert_eq!(
            rotated[n_rot..],
            base[n_rot..],
            "dimensions past n_rot must pass through untouched"
        );

        // Rotate-half: pairs are (i, i + n_rot/2), not (2i, 2i+1).
        let half = n_rot / 2;
        for i in 0..half {
            let freq = 1.0 / 10_000_000f32.powf(2.0 * i as f32 / n_rot as f32);
            let (sin, cos) = (7.0 * freq).sin_cos();
            let (a, b) = (base[i], base[i + half]);
            assert!((rotated[i] - (a * cos - b * sin)).abs() < 1e-5);
            assert!((rotated[i + half] - (b * cos + a * sin)).abs() < 1e-5);
        }
    }

    /// Speculation is off, and the cache says why: a recurrent state cannot be rewound.
    #[test]
    fn a_sequence_carrying_recurrent_state_refuses_to_be_rewound() {
        let b = backend(Head::Separate);
        assert!(!b.speculation_supported());
        assert!(b.logits_multi(&[1, 2, 3], &mut seq_for(&b), 2).is_err());

        let mut seq = seq_for(&b);
        b.logits(&[1, 2, 3], &mut seq).unwrap();
        assert!(seq.truncate(3).is_ok(), "a no-op truncation is fine");
        let err = seq.truncate(1).unwrap_err().to_string();
        assert!(err.contains("recurrent state"), "unexpected error: {err}");
        assert_eq!(seq.len(), 3, "a refused truncation changes nothing");
    }

    #[test]
    fn packed_and_expanded_weights_agree() {
        let bytes = build_qwen35_gguf(Head::Separate, "qwen35");
        let expanded = Qwen35Backend::load(&bytes).unwrap();

        let dir = std::env::temp_dir().join("garuda_qwen35_mmap_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&bytes)
            .unwrap();
        let map = Arc::new(unsafe { Mmap::map(&std::fs::File::open(&path).unwrap()).unwrap() });
        let g = Gguf::parse(&map).unwrap();
        let packed = Qwen35Backend::from_gguf(&g, &map, Some(map.clone())).unwrap();
        assert!(packed.is_mmapped() && !expanded.is_mmapped());

        let ctx: Vec<Token> = vec![7, 8, 9, 10];
        let a = expanded.logits(&ctx, &mut seq_for(&expanded)).unwrap();
        let b = packed.logits(&ctx, &mut seq_for(&packed)).unwrap();
        for (i, (x, y)) in a.data().iter().zip(b.data()).enumerate() {
            assert!(
                (x - y).abs() < 1e-4,
                "logit {i}: expanded {x} vs packed {y}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_checkpoint_that_ties_its_head_to_the_embeddings_still_answers() {
        let b = backend(Head::Tied);
        assert!(b.has_tied_embeddings());
        let logits = b.logits(&[5, 6], &mut seq_for(&b)).unwrap();
        assert_eq!(logits.len(), VOCAB);
        assert!(logits.data().iter().all(|v| v.is_finite()));
    }

    /// The mixture-of-experts sibling is a different architecture, and half-running it
    /// would be worse than refusing it.
    #[test]
    fn the_mixture_of_experts_variant_is_refused_by_name() {
        let load_err =
            |arch: &str| match Qwen35Backend::load(&build_qwen35_gguf(Head::Separate, arch)) {
                Err(e) => e.to_string(),
                Ok(_) => panic!("architecture '{arch}' should not have loaded"),
            };

        let err = load_err("qwen35moe");
        assert!(err.contains("qwen35moe"), "unexpected error: {err}");
        let err = load_err("llama");
        assert!(err.contains("not Qwen3.5"), "unexpected error: {err}");
    }

    #[test]
    fn an_oversized_prompt_is_refused_before_any_layer_runs() {
        let b = backend(Head::Separate);
        let cfg = b.config();
        let mut seq = SeqState::new(
            KvConfig {
                dims: b.dims(),
                kv_dim: cfg.kv_dim(),
                n_layers: cfg.n_layers,
                kv_dims: Some(cfg.kv_dims()),
                max_positions: 4,
                max_resident_blocks: 8,
                sliding_window: None,
                storage: None,
            },
            2,
        );
        let ctx: Vec<Token> = (0..5).collect();
        assert!(b.logits(&ctx, &mut seq).is_err());
        assert_eq!(seq.len(), 0, "nothing was consumed");
    }

    /// Batching is an optimisation, never a change of answer: a batched decode step
    /// has to return exactly what decoding each sequence alone returns. For a hybrid
    /// model this is where a mixer that reached for the wrong sequence's recurrent
    /// state would show up — and it would still return plausible numbers.
    #[test]
    fn a_batched_decode_step_matches_decoding_each_sequence_alone() {
        let b = backend(Head::Separate);
        let ctxs: Vec<Vec<Token>> = vec![vec![1, 2, 3, 4], vec![9, 8], vec![5, 5, 6]];

        // Alone: prefill, then one more token each.
        let mut alone = Vec::new();
        for c in &ctxs {
            let mut seq = seq_for(&b);
            b.logits(&c[..c.len() - 1], &mut seq).unwrap();
            alone.push(b.logits(c, &mut seq).unwrap());
        }

        // Together: the same prefills, then one batched step.
        let mut seqs: Vec<SeqState> = ctxs.iter().map(|_| seq_for(&b)).collect();
        for (c, s) in ctxs.iter().zip(seqs.iter_mut()) {
            b.logits(&c[..c.len() - 1], s).unwrap();
        }
        let mut refs: Vec<&mut SeqState> = seqs.iter_mut().collect();
        let views: Vec<&[Token]> = ctxs.iter().map(|c| c.as_slice()).collect();
        let batched = b.logits_batch(&views, &mut refs).unwrap();

        assert_eq!(batched.len(), alone.len());
        for (i, (x, y)) in batched.iter().zip(&alone).enumerate() {
            for (j, (a, c)) in x.data().iter().zip(y.data()).enumerate() {
                assert!(
                    (a - c).abs() < 1e-4,
                    "sequence {i}, logit {j}: batched {a} vs alone {c}"
                );
            }
        }
        for (i, s) in seqs.iter().enumerate() {
            assert_eq!(s.len(), ctxs[i].len(), "sequence {i} did not advance once");
        }
    }

    /// A ragged batch — sequences at different distances from their contexts — is not
    /// one decode step, so it falls back to running them separately and still answers.
    #[test]
    fn a_ragged_batch_falls_back_and_still_answers_correctly() {
        let b = backend(Head::Separate);
        let ctxs: Vec<Vec<Token>> = vec![vec![1, 2, 3], vec![4, 5, 6, 7]];

        let mut seqs: Vec<SeqState> = ctxs.iter().map(|_| seq_for(&b)).collect();
        // Only the first sequence is one token behind its context.
        b.logits(&[1, 2], &mut seqs[0]).unwrap();
        let mut refs: Vec<&mut SeqState> = seqs.iter_mut().collect();
        let views: Vec<&[Token]> = ctxs.iter().map(|c| c.as_slice()).collect();
        let out = b.logits_batch(&views, &mut refs).unwrap();

        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|t| t.len() == VOCAB));
        assert_eq!(seqs[0].len(), 3);
        assert_eq!(seqs[1].len(), 4);

        // Mismatched lengths are an error, not a panic.
        let mut one: Vec<&mut SeqState> = seqs.iter_mut().take(1).collect();
        assert!(b.logits_batch(&views, &mut one).is_err());
    }

    #[test]
    fn a_batch_is_rejected_before_any_sequence_advances() {
        let b = backend(Head::Separate);
        let mut seqs: Vec<SeqState> = (0..2).map(|_| seq_for(&b)).collect();
        for s in seqs.iter_mut() {
            b.logits(&[1, 2], s).unwrap();
        }
        // The second batch member names a token outside the vocabulary.
        let ctxs: Vec<Vec<Token>> = vec![vec![1, 2, 3], vec![1, 2, VOCAB as Token]];
        let views: Vec<&[Token]> = ctxs.iter().map(|c| c.as_slice()).collect();
        let mut refs: Vec<&mut SeqState> = seqs.iter_mut().collect();
        assert!(matches!(
            b.logits_batch(&views, &mut refs),
            Err(GarudaError::InvalidToken(_))
        ));
        for s in &seqs {
            assert_eq!(s.len(), 2, "a refused batch leaves every sequence alone");
        }
    }

    #[test]
    fn an_out_of_vocabulary_token_is_rejected() {
        let b = backend(Head::Separate);
        let err = b.logits(&[VOCAB as Token], &mut seq_for(&b)).unwrap_err();
        assert!(matches!(err, GarudaError::InvalidToken(_)));
    }
}

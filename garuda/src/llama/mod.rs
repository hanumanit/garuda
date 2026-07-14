//! A Llama-family dense transformer backend, loaded from a GGUF checkpoint.
//!
//! This is the real thing running on real trained weights: `token_embd`, per-block
//! RMSNorm + grouped-query attention with RoPE + SwiGLU feed-forward, a final norm,
//! and an output projection. It implements the same [`InferenceBackend`] as the
//! synthetic MoE engine, so it drops into the existing runtime, scheduler and API
//! with nothing else changed — which is the whole point of the trait.
//!
//! Only F32/F16 checkpoints load; quantised tensors are rejected by the GGUF reader
//! because no dequantiser exists yet. In practice that means small models (the
//! TinyStories checkpoints are F32).

use crate::cache::{KVCacheState, SeqState};
use crate::core::{GarudaError, InferenceBackend, ModelDims, Tensor, Token};
use crate::gguf::Gguf;
use crate::{quant, simd};
use memmap2::Mmap;
use std::sync::Arc;

/// One weight matrix, either expanded to `f32` in RAM or kept packed (quantised) in a
/// memory-mapped file and dequantised a row at a time during matmul.
///
/// `Full` is fast (dequantised once at load) but holds the whole `f32` matrix; `Packed`
/// trades speed for memory — the model occupies its on-disk (quantised) size, so a
/// checkpoint far larger than RAM can run via demand paging.
enum Weight {
    Full {
        data: Vec<f32>,
        rows: usize,
        cols: usize,
    },
    Packed {
        qtype: u32,
        rows: usize,
        cols: usize,
        src: Arc<Mmap>,
        start: usize,
        len: usize,
    },
}

impl Weight {
    /// `out[r] = dot(row r, x)`.
    fn matvec(&self, x: &[f32], out: &mut [f32]) -> Result<(), GarudaError> {
        match self {
            Weight::Full { data, rows, cols } => {
                simd::matvec(data, *rows, *cols, x, out);
                Ok(())
            }
            Weight::Packed {
                qtype,
                rows,
                cols,
                src,
                start,
                len,
            } => quant::matvec(*qtype, &src[*start..*start + *len], *rows, *cols, x, out),
        }
    }

    /// Dequantise a single row (e.g. one embedding).
    fn row(&self, r: usize) -> Result<Vec<f32>, GarudaError> {
        match self {
            Weight::Full { data, cols, .. } => Ok(data[r * cols..(r + 1) * cols].to_vec()),
            Weight::Packed {
                qtype,
                cols,
                src,
                start,
                ..
            } => {
                let row_bytes = quant::byte_size(*qtype, *cols)?;
                let off = start + r * row_bytes;
                quant::dequantize(*qtype, &src[off..off + row_bytes], *cols)
            }
        }
    }
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
}

impl LlamaConfig {
    fn from_gguf(g: &Gguf) -> Result<Self, GarudaError> {
        if g.architecture() != Some("llama") {
            return Err(GarudaError::Model(format!(
                "architecture '{}' is not supported (only llama)",
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

/// One transformer block's weights. Norms are small and always `f32`; the four
/// projection matrices may be packed.
struct Layer {
    attn_norm: Vec<f32>,
    wq: Weight,
    wk: Weight,
    wv: Weight,
    wo: Weight,
    ffn_norm: Vec<f32>,
    gate: Weight,
    up: Weight,
    down: Weight,
}

pub struct LlamaBackend {
    cfg: LlamaConfig,
    token_embd: Weight,
    output_norm: Vec<f32>,
    output: Weight,
    layers: Vec<Layer>,
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
        let norm = |name: &str, n: usize| -> Result<Vec<f32>, GarudaError> {
            let data = g.tensor_f32(bytes, name)?;
            if data.len() != n {
                return Err(GarudaError::Model(format!(
                    "tensor '{name}' has {} values, expected {n}",
                    data.len()
                )));
            }
            Ok(data)
        };

        // A weight matrix: packed if mmapping, otherwise expanded to f32.
        let weight = |name: &str, rows: usize, cols: usize| -> Result<Weight, GarudaError> {
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
            match &mmap {
                Some(src) => {
                    let len = quant::byte_size(t.ggml_type, rows * cols)?;
                    let start = g.data_offset + t.offset as usize;
                    if start + len > src.len() {
                        return Err(GarudaError::Model(format!(
                            "tensor '{name}' runs past the end of the file"
                        )));
                    }
                    Ok(Weight::Packed {
                        qtype: t.ggml_type,
                        rows,
                        cols,
                        src: src.clone(),
                        start,
                        len,
                    })
                }
                None => Ok(Weight::Full {
                    data: g.tensor_f32(bytes, name)?,
                    rows,
                    cols,
                }),
            }
        };

        let token_embd = weight("token_embd.weight", v, d)?;
        let output_norm = norm("output_norm.weight", d)?;
        // Some checkpoints tie the output head to the embeddings and omit `output`.
        let output = if g.tensor("output.weight").is_some() {
            weight("output.weight", v, d)?
        } else {
            weight("token_embd.weight", v, d)?
        };

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            let p = |name: &str| format!("blk.{l}.{name}.weight");
            layers.push(Layer {
                attn_norm: norm(&p("attn_norm"), d)?,
                wq: weight(&p("attn_q"), d, d)?,
                wk: weight(&p("attn_k"), hk, d)?,
                wv: weight(&p("attn_v"), hk, d)?,
                wo: weight(&p("attn_output"), d, d)?,
                ffn_norm: norm(&p("ffn_norm"), d)?,
                gate: weight(&p("ffn_gate"), f, d)?,
                up: weight(&p("ffn_up"), f, d)?,
                down: weight(&p("ffn_down"), d, f)?,
            });
        }

        Ok(Self {
            cfg,
            token_embd,
            output_norm,
            output,
            layers,
        })
    }

    /// True when weights are kept packed in a memory-mapped file.
    pub fn is_mmapped(&self) -> bool {
        matches!(self.token_embd, Weight::Packed { .. })
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

    /// One block: `x + attn(norm(x))`, then `x + ffn(norm(x))`.
    fn block(&self, l: usize, x: &mut [f32], kv: &mut KVCacheState) -> Result<(), GarudaError> {
        let layer = &self.layers[l];

        let h = self.norm(x, &layer.attn_norm);
        let attn = self.attention(layer, &h, kv)?;
        simd::add_assign(x, &attn);

        let h = self.norm(x, &layer.ffn_norm);
        let ffn = self.feed_forward(layer, &h)?;
        simd::add_assign(x, &ffn);
        Ok(())
    }

    /// Grouped-query causal attention with rotary embeddings for one token.
    fn attention(
        &self,
        layer: &Layer,
        h: &[f32],
        kv: &mut KVCacheState,
    ) -> Result<Vec<f32>, GarudaError> {
        let LlamaConfig {
            d_model: d,
            n_heads,
            n_kv_heads,
            head_dim: hd,
            rope_theta,
            ..
        } = self.cfg;
        let kv_dim = self.cfg.kv_dim();
        let group = n_heads / n_kv_heads;

        let mut q = vec![0.0; d];
        let mut k = vec![0.0; kv_dim];
        let mut v = vec![0.0; kv_dim];
        layer.wq.matvec(h, &mut q)?;
        layer.wk.matvec(h, &mut k)?;
        layer.wv.matvec(h, &mut v)?;

        let pos = kv.len();
        for hh in 0..n_heads {
            simd::rope(&mut q[hh * hd..(hh + 1) * hd], pos, rope_theta);
        }
        for hh in 0..n_kv_heads {
            simd::rope(&mut k[hh * hd..(hh + 1) * hd], pos, rope_theta);
        }

        kv.append(&k, &v)?;

        let start = kv.attention_start();
        let end = pos + 1;
        kv.ensure_resident(start, end)?;

        let scale = 1.0 / (hd as f32).sqrt();
        let mut context = vec![0.0; d];

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

        let mut out = vec![0.0; d];
        layer.wo.matvec(&context, &mut out)?;
        Ok(out)
    }

    /// SwiGLU: `down(silu(gate·h) ⊙ (up·h))`.
    fn feed_forward(&self, layer: &Layer, h: &[f32]) -> Result<Vec<f32>, GarudaError> {
        let (d, f) = (self.cfg.d_model, self.cfg.d_ff);
        let mut gate = vec![0.0; f];
        let mut up = vec![0.0; f];
        layer.gate.matvec(h, &mut gate)?;
        layer.up.matvec(h, &mut up)?;
        simd::silu(&mut gate);
        simd::mul_assign(&mut gate, &up);
        let mut out = vec![0.0; d];
        layer.down.matvec(&gate, &mut out)?;
        Ok(out)
    }
}

impl InferenceBackend for LlamaBackend {
    fn dims(&self) -> ModelDims {
        self.cfg.model_dims()
    }

    fn hidden(&self, context: &[Token], seq: &mut SeqState) -> Result<Tensor, GarudaError> {
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
        let mut last = None;
        for &token in &context[already..] {
            let idx = token as usize;
            if idx >= self.cfg.vocab {
                return Err(GarudaError::InvalidToken(token));
            }
            let mut x = self.token_embd.row(idx)?;
            for l in 0..self.cfg.n_layers {
                // Split the layer borrow from the per-layer cache borrow.
                let kv = seq.layer(l);
                self.block(l, &mut x, kv)?;
            }
            last = Some(x);
        }

        let mut x = last.ok_or_else(|| {
            GarudaError::Inference("no new tokens to process for this context".into())
        })?;
        simd::rmsnorm(&mut x, self.cfg.rms_eps);
        simd::mul_assign(&mut x, &self.output_norm);
        Tensor::new(vec![d], x)
    }

    fn logits(&self, context: &[Token], seq: &mut SeqState) -> Result<Tensor, GarudaError> {
        let x = self.hidden(context, seq)?;
        let mut logits = vec![0.0; self.cfg.vocab];
        self.output.matvec(x.data(), &mut logits)?;
        Tensor::new(vec![self.cfg.vocab], logits)
    }
}

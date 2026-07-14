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
//! quantisations). See [`ExpertWeight`].

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
        cols: usize,
    },
    Packed {
        qtype: u32,
        cols: usize,
        src: Arc<Mmap>,
        start: usize,
    },
}

impl Weight {
    /// `out[r] = dot(row r, x)` over the whole matrix.
    fn matvec(&self, x: &[f32], out: &mut [f32]) -> Result<(), GarudaError> {
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
                quant::matvec(*qtype, &src[off..off + n * row_bytes], n, *cols, x, out)
            }
        }
    }

    /// Dequantise a single row (e.g. one embedding).
    fn row(&self, r: usize) -> Result<Vec<f32>, GarudaError> {
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
                quant::dequantize(*qtype, &src[off..off + row_bytes], *cols)
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
    /// Expert `e`'s matvec. `block` is one expert's row count in the stacked
    /// layout; it is ignored for `Split`, where each tensor already holds
    /// exactly one expert.
    fn matvec_expert(
        &self,
        e: usize,
        block: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<(), GarudaError> {
        match self {
            ExpertWeight::Stacked(w) => w.matvec_rows(e * block, x, out),
            ExpertWeight::Split(ws) => ws[e].matvec(x, out),
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
    /// Number of experts in a mixture-of-experts FFN. `0` means a dense FFN.
    pub n_experts: usize,
    /// Experts activated per token (top-k). Unused when `n_experts == 0`.
    pub n_experts_used: usize,
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
                        cols,
                        src: src.clone(),
                        start,
                    })
                }
                None => Ok(Weight::Full {
                    data: g.tensor_f32(bytes, name)?,
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

        let ne = cfg.n_experts;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            let p = |name: &str| format!("blk.{l}.{name}.weight");
            // Per-expert tensor name for the split (un-merged) layout, e.g.
            // `blk.0.ffn_gate.3.weight`.
            let pe = |name: &str, e: usize| format!("blk.{l}.{name}.{e}.weight");
            let split = |name: &str, rows: usize, cols: usize| -> Result<ExpertWeight, GarudaError> {
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

    /// One SwiGLU expert `e`: `down_e(silu(gate_e·h) ⊙ (up_e·h))`. `e = 0` for a dense
    /// FFN (a single expert spanning the whole tensor).
    fn expert(
        &self,
        gate: &ExpertWeight,
        up: &ExpertWeight,
        down: &ExpertWeight,
        e: usize,
        h: &[f32],
    ) -> Result<Vec<f32>, GarudaError> {
        let (d, f) = (self.cfg.d_model, self.cfg.d_ff);
        let mut g = vec![0.0; f];
        let mut u = vec![0.0; f];
        gate.matvec_expert(e, f, h, &mut g)?;
        up.matvec_expert(e, f, h, &mut u)?;
        simd::silu(&mut g);
        simd::mul_assign(&mut g, &u);
        let mut out = vec![0.0; d];
        down.matvec_expert(e, d, &g, &mut out)?;
        Ok(out)
    }

    /// The block's feed-forward, dense or mixture-of-experts.
    fn feed_forward(&self, layer: &Layer, h: &[f32]) -> Result<Vec<f32>, GarudaError> {
        match &layer.ffn {
            Ffn::Dense { gate, up, down } => self.expert(gate, up, down, 0, h),
            Ffn::Moe {
                router,
                gate,
                up,
                down,
            } => {
                let d = self.cfg.d_model;
                let (ne, k) = (self.cfg.n_experts, self.cfg.n_experts_used.max(1));

                // Route: score every expert, softmax, take the top-k, renormalise their
                // weights — the standard Mixtral gating.
                let mut scores = vec![0.0; ne];
                router.matvec(h, &mut scores)?;
                simd::softmax(&mut scores);

                let mut ranked: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
                ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
                ranked.truncate(k);
                let sum: f32 = ranked.iter().map(|(_, w)| w).sum();
                let norm = if sum > 0.0 { 1.0 / sum } else { 1.0 / k as f32 };

                // Run only the chosen experts and blend by weight. Under mmap, this is
                // the only place a token touches expert weights — top-k of ne per layer.
                let mut out = vec![0.0; d];
                for (e, w) in ranked {
                    let y = self.expert(gate, up, down, e, h)?;
                    simd::add_scaled(&mut out, &y, w * norm);
                }
                Ok(out)
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{KvConfig, SeqState};
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

    /// Build a tiny MoE llama GGUF (F32 weights) entirely in memory, in either
    /// expert tensor layout. Both layouts get identical numbers (each gate/up/down
    /// is generated once as a flat array and either kept whole or sliced per
    /// expert), so outputs from the two layouts can be compared directly.
    fn build_moe_gguf(layout: ExpertLayout) -> Vec<u8> {
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
        add(
            "output.weight".into(),
            vec![d as u64, vocab as u64],
            s,
            &mut tensors,
        );
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

            for (base, out_rows, cols) in [("ffn_gate", ff, d), ("ffn_up", ff, d), ("ffn_down", d, ff)] {
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
            src: mmap,
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
}

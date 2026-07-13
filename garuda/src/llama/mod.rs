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
use crate::simd;

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

/// One transformer block's weights.
struct Layer {
    attn_norm: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    ffn_norm: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
}

pub struct LlamaBackend {
    cfg: LlamaConfig,
    token_embd: Vec<f32>,
    output_norm: Vec<f32>,
    output: Vec<f32>,
    layers: Vec<Layer>,
}

impl LlamaBackend {
    /// Load a checkpoint from a GGUF file's bytes.
    pub fn load(bytes: &[u8]) -> Result<Self, GarudaError> {
        let g = Gguf::parse(bytes)?;
        Self::from_gguf(&g, bytes)
    }

    /// Load from an already-parsed GGUF header plus the file bytes, so the caller can
    /// also build the tokenizer from the same parse without parsing twice.
    pub fn from_gguf(g: &Gguf, bytes: &[u8]) -> Result<Self, GarudaError> {
        let cfg = LlamaConfig::from_gguf(g)?;

        // Every tensor is validated for length and finiteness as it is read.
        let get = |name: &str| g.tensor_f32(bytes, name);
        let expect = |data: Vec<f32>, name: &str, n: usize| -> Result<Vec<f32>, GarudaError> {
            if data.len() != n {
                return Err(GarudaError::Model(format!(
                    "tensor '{name}' has {} values, expected {n}",
                    data.len()
                )));
            }
            Ok(data)
        };

        let (d, f, v, hk) = (cfg.d_model, cfg.d_ff, cfg.vocab, cfg.kv_dim());

        let token_embd = expect(get("token_embd.weight")?, "token_embd.weight", v * d)?;
        let output_norm = expect(get("output_norm.weight")?, "output_norm.weight", d)?;
        // Some checkpoints tie the output head to the embeddings and omit `output`.
        let output = match g.tensor("output.weight") {
            Some(_) => expect(get("output.weight")?, "output.weight", v * d)?,
            None => token_embd.clone(),
        };

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            let p = |name: &str| format!("blk.{l}.{name}.weight");
            layers.push(Layer {
                attn_norm: expect(get(&p("attn_norm"))?, "attn_norm", d)?,
                wq: expect(get(&p("attn_q"))?, "attn_q", d * d)?,
                wk: expect(get(&p("attn_k"))?, "attn_k", hk * d)?,
                wv: expect(get(&p("attn_v"))?, "attn_v", hk * d)?,
                wo: expect(get(&p("attn_output"))?, "attn_output", d * d)?,
                ffn_norm: expect(get(&p("ffn_norm"))?, "ffn_norm", d)?,
                gate: expect(get(&p("ffn_gate"))?, "ffn_gate", f * d)?,
                up: expect(get(&p("ffn_up"))?, "ffn_up", f * d)?,
                down: expect(get(&p("ffn_down"))?, "ffn_down", d * f)?,
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
        let ffn = self.feed_forward(layer, &h);
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
        simd::matvec(&layer.wq, d, d, h, &mut q);
        simd::matvec(&layer.wk, kv_dim, d, h, &mut k);
        simd::matvec(&layer.wv, kv_dim, d, h, &mut v);

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
        simd::matvec(&layer.wo, d, d, &context, &mut out);
        Ok(out)
    }

    /// SwiGLU: `down(silu(gate·h) ⊙ (up·h))`.
    fn feed_forward(&self, layer: &Layer, h: &[f32]) -> Vec<f32> {
        let (d, f) = (self.cfg.d_model, self.cfg.d_ff);
        let mut gate = vec![0.0; f];
        let mut up = vec![0.0; f];
        simd::matvec(&layer.gate, f, d, h, &mut gate);
        simd::matvec(&layer.up, f, d, h, &mut up);
        simd::silu(&mut gate);
        simd::mul_assign(&mut gate, &up);
        let mut out = vec![0.0; d];
        simd::matvec(&layer.down, d, f, &gate, &mut out);
        out
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
            let mut x = self.token_embd[idx * d..(idx + 1) * d].to_vec();
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
        let (d, v) = (self.cfg.d_model, self.cfg.vocab);
        let mut logits = vec![0.0; v];
        simd::matvec(&self.output, v, d, x.data(), &mut logits);
        Tensor::new(vec![v], logits)
    }
}

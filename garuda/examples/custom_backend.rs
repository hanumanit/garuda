//! A complete, minimal `InferenceBackend` plugin — the worked example for
//! PLUGIN.md. Run it with:
//!
//!   cargo run --example custom_backend
//!
//! Like the built-in synthetic MoE, this does real arithmetic over made-up weights,
//! so its output is meaningless. The point is the plumbing: a self-contained backend
//! that satisfies the `InferenceBackend` contract and drops straight into the real
//! runtime, scheduler and sampler.

use garuda::cache::{KvConfig, SeqState};
use garuda::core::{GarudaError, InferenceBackend, ModelDims, Tensor, Token};
use garuda::runtime::{InferenceRuntime, SamplingParams};
use garuda::tokenizer::{Tokenizer, VOCAB_SIZE};
use std::sync::Arc;

/// A toy backend: embed each token, then project the last hidden state to logits.
/// Deterministic, so it honours the contract's reproducibility rule.
struct ToyBackend {
    dims: ModelDims,
}

impl ToyBackend {
    fn new() -> Self {
        // The only hard requirements: n_heads * head_dim == d_model, top_k in
        // 1..=n_experts, and vocab_size matching the tokenizer paired with it.
        let dims = ModelDims {
            d_model: 16,
            n_heads: 2,
            head_dim: 8,
            d_ff: 32,
            n_experts: 1,
            top_k: 1,
            vocab_size: VOCAB_SIZE,
            block_size: 8,
            rope_theta: 10_000.0,
        };
        Self { dims }
    }

    /// A fixed pseudo-embedding for a token — no training, just something reproducible.
    fn embed(&self, t: Token) -> Vec<f32> {
        (0..self.dims.d_model)
            .map(|i| (((t as usize) * 131 + i * 17) % 97) as f32 / 97.0 - 0.5)
            .collect()
    }
}

impl InferenceBackend for ToyBackend {
    fn dims(&self) -> ModelDims {
        self.dims
    }

    fn hidden(&self, ctx: &[Token], seq: &mut SeqState) -> Result<Tensor, GarudaError> {
        if ctx.is_empty() {
            return Err(GarudaError::Inference("empty context".into())); // invariant 4
        }
        let d = self.dims.d_model;
        let mut last = None;

        // Invariant 1: process only the positions the sequence has not seen yet.
        for &tok in &ctx[seq.len()..] {
            if (tok as usize) >= self.dims.vocab_size {
                return Err(GarudaError::InvalidToken(tok)); // invariant 4
            }
            // Invariant 2: append exactly one KV position per token. A real model
            // would store this token's attention key/value; this toy stores zeros
            // just to keep the cache length in step. `append` also surfaces the
            // context-window-exhausted error for us.
            let zero = vec![0.0; d];
            seq.kv().append(&zero, &zero)?;

            last = Some(self.embed(tok));
        }

        let x = last.ok_or_else(|| GarudaError::Inference("no new tokens".into()))?;
        Tensor::new(vec![d], x)
    }

    fn logits(&self, ctx: &[Token], seq: &mut SeqState) -> Result<Tensor, GarudaError> {
        let h = self.hidden(ctx, seq)?;
        let v = self.dims.vocab_size;

        // Invariant 3: the logits tensor is exactly `vocab_size` long. Deterministic
        // projection — no randomness here; that belongs to the sampler.
        let mut logits = vec![0.0; v];
        for (t, out) in logits.iter_mut().enumerate() {
            *out = h
                .data()
                .iter()
                .enumerate()
                .map(|(i, &hi)| hi * (((t * 7 + i * 13) % 11) as f32 - 5.0))
                .sum();
        }
        Tensor::new(vec![v], logits)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build the backend and pair it with a tokenizer whose vocab_size it matches.
    let backend = Arc::new(ToyBackend::new());
    let dims = backend.dims();

    // One attention layer, key/value width == d_model (full multi-head attention).
    let kv = KvConfig::mha(dims, 256, 64, None, None);
    let runtime = InferenceRuntime::new(Arc::new(Tokenizer::new()), backend, kv, 8);

    let params = SamplingParams {
        temperature: 0.8, // sample; the seed keeps the run reproducible
        top_p: 0.95,
        top_k: 40,
        max_tokens: 16,
        seed: Some(1),
    };

    let prompt = runtime.tokenizer.encode("hello plugin");
    let mut session = runtime.start(&prompt, &params)?;

    let mut out = Vec::new();
    while let Ok(token) = runtime.next_token(&mut session, &params) {
        out.push(token);
    }

    println!("prompt:    {:?}", "hello plugin");
    println!("generated: {} tokens", out.len());
    println!("decoded:   {:?}", runtime.tokenizer.decode(&out)?);
    println!("(gibberish, as expected — untrained weights)");
    Ok(())
}

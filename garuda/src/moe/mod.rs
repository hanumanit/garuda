//! The MoE transformer block, and the forward pass that turns a context into logits.
//!
//! One block: `x + attention(rmsnorm(x))`, then `x + moe_ffn(rmsnorm(x))`, then a
//! final norm and a tied-embedding projection to vocabulary logits. Each expert is
//! a SwiGLU feed-forward network — `down(silu(gate·x) * (up·x))`.
//!
//! "Expert streaming" means exactly this: only the `top_k` experts a token routes
//! to are ever pulled into RAM for that token, and they are pulled through the
//! tiered [`crate::memory::MemoryManager`] rather than all being held resident.
//!
//! The arithmetic is the real thing. The weights are untrained, so the *output* is
//! meaningless — see [`crate::weights`].

use crate::cache::SeqState;
use crate::core::{
    Expert, ExpertId, ExpertLoader, GarudaError, InferenceBackend, ModelDims, Tensor, Token,
};
use crate::prefetch::PrefetchEngine;
use crate::router::Router;
use crate::simd;
use crate::weights::ModelWeights;
use std::sync::Arc;

const NORM_EPS: f32 = 1e-5;

pub struct MoeEngine {
    dims: ModelDims,
    weights: Arc<ModelWeights>,
    router: Router,
    attention: crate::attention::Attention,
    loader: Arc<dyn ExpertLoader>,
    prefetch: Option<Arc<PrefetchEngine>>,
}

impl MoeEngine {
    pub fn new(
        dims: ModelDims,
        weights: Arc<ModelWeights>,
        router: Router,
        loader: Arc<dyn ExpertLoader>,
        prefetch: Option<Arc<PrefetchEngine>>,
    ) -> Result<Self, GarudaError> {
        dims.validate()?;
        Ok(Self {
            attention: crate::attention::Attention::new(dims)?,
            dims,
            weights,
            router,
            loader,
            prefetch,
        })
    }

    pub fn weights(&self) -> &Arc<ModelWeights> {
        &self.weights
    }

    pub fn router(&self) -> &Router {
        &self.router
    }

    /// One expert's SwiGLU feed-forward pass.
    ///
    /// Validates the expert's shape against the model's dims first. A checkpoint
    /// whose tensors disagree with the configuration is a load-time error here,
    /// not a panic or a silently wrong result deeper in.
    fn expert_ffn(&self, expert: &Expert, h: &[f32]) -> Result<Vec<f32>, GarudaError> {
        let (d, f) = (self.dims.d_model, self.dims.d_ff);

        if expert.gate.len() != f * d || expert.up.len() != f * d || expert.down.len() != d * f {
            return Err(GarudaError::Model(format!(
                "expert {} has tensors of size ({}, {}, {}) but dims require ({}, {}, {})",
                expert.id,
                expert.gate.len(),
                expert.up.len(),
                expert.down.len(),
                f * d,
                f * d,
                d * f
            )));
        }

        let mut gate = vec![0.0; f];
        let mut up = vec![0.0; f];
        simd::matvec(&expert.gate, f, d, h, &mut gate);
        simd::matvec(&expert.up, f, d, h, &mut up);

        simd::silu(&mut gate);
        simd::mul_assign(&mut gate, &up);

        let mut out = vec![0.0; d];
        simd::matvec(&expert.down, d, f, &gate, &mut out);
        Ok(out)
    }

    /// Route `h`, run the selected experts, and blend them by gate weight.
    /// Returns the block output and the experts that actually fired.
    fn moe_ffn(&self, h: &[f32]) -> Result<(Vec<f32>, Vec<ExpertId>), GarudaError> {
        let routed = self.router.route(h, &self.weights)?;

        let mut out = vec![0.0; self.dims.d_model];
        let mut used = Vec::with_capacity(routed.len());

        for (id, weight) in routed {
            let expert = self.loader.load(id)?;
            let y = self.expert_ffn(&expert, h)?;
            simd::add_scaled(&mut out, &y, weight);
            used.push(id);
        }

        Ok((out, used))
    }

    /// Run one token through the block, updating `seq`.
    fn step(&self, token: Token, seq: &mut SeqState) -> Result<Vec<f32>, GarudaError> {
        let mut x = self.weights.embed(token)?.to_vec();

        let mut h = x.clone();
        simd::rmsnorm(&mut h, NORM_EPS);
        let attn = self.attention.forward(&h, &self.weights, seq.kv())?;
        simd::add_assign(&mut x, &attn);

        let mut h = x.clone();
        simd::rmsnorm(&mut h, NORM_EPS);
        let (ffn, used) = self.moe_ffn(&h)?;
        simd::add_assign(&mut x, &ffn);

        // Learn from this step and warm what the next one probably needs. This is
        // advisory: if it predicts nothing, or predicts wrong, the next step still
        // loads whatever it routes to.
        let kv = seq.kv();
        if let Some(pf) = &self.prefetch {
            let predicted = pf.observe_step(&kv.last_experts, &used, &kv.last_predicted);
            kv.last_predicted = predicted;
        }
        kv.last_experts = used;

        Ok(x)
    }
}

impl InferenceBackend for MoeEngine {
    fn dims(&self) -> ModelDims {
        self.dims
    }

    fn hidden(&self, context: &[Token], seq: &mut SeqState) -> Result<Tensor, GarudaError> {
        if context.is_empty() {
            return Err(GarudaError::Inference("empty context".into()));
        }

        let already = seq.len();
        if already > context.len() {
            return Err(GarudaError::Inference(format!(
                "sequence state holds {already} positions but the context is only {} long",
                context.len()
            )));
        }

        // Only tokens the cache has not seen are computed. On a fresh sequence that
        // is the whole prompt (prefill); on each decode step, exactly one token.
        let mut last = None;
        for &token in &context[already..] {
            last = Some(self.step(token, seq)?);
        }

        // Nothing new to consume: the caller asked about a context the cache has
        // already fully absorbed, which means they lost track of their own state.
        let mut x = last.ok_or_else(|| {
            GarudaError::Inference("no new tokens to process for this context".into())
        })?;

        simd::rmsnorm(&mut x, NORM_EPS);
        Tensor::new(vec![self.dims.d_model], x)
    }

    fn logits(&self, context: &[Token], seq: &mut SeqState) -> Result<Tensor, GarudaError> {
        let x = self.hidden(context, seq)?;

        // Tied embeddings: the output head is the embedding matrix.
        let v = self.dims.vocab_size;
        let d = self.dims.d_model;
        let mut logits = vec![0.0; v];
        simd::matvec(&self.weights.embedding, v, d, x.data(), &mut logits);

        Tensor::new(vec![v], logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{KvConfig, SeqState};
    use crate::core::StorageBackend;
    use crate::memory::MemoryManager;
    use crate::router::RouterType;
    use crate::storage::LocalStorageBackend;
    use std::path::PathBuf;

    struct Fixture {
        engine: MoeEngine,
        dims: ModelDims,
        dir: PathBuf,
    }

    fn fixture(tag: &str) -> Fixture {
        let dims = ModelDims::default();
        let dir = std::env::temp_dir().join(format!("garuda_moe_{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let l2: Arc<dyn StorageBackend> = Arc::new(LocalStorageBackend::new(&dir));
        let budget = Expert::n_params(&dims) * 4 * dims.n_experts;
        let mm = Arc::new(MemoryManager::new(dims, budget, l2, None).unwrap());
        let weights = Arc::new(ModelWeights::synthesize(dims).unwrap());
        let router = Router::new(RouterType::Mixtral, dims).unwrap();

        let engine = MoeEngine::new(dims, weights, router, mm, None).unwrap();
        Fixture { engine, dims, dir }
    }

    fn seq(dims: ModelDims) -> SeqState {
        SeqState::new(KvConfig::mha(dims, 512, 64, None, None), 1)
    }

    #[test]
    fn logits_have_vocab_shape_and_are_finite() {
        let f = fixture("shape");
        let mut s = seq(f.dims);

        let out = f.engine.logits(&[10, 20, 30], &mut s).unwrap();
        assert_eq!(out.shape(), &[f.dims.vocab_size]);
        assert!(out.data().iter().all(|v| v.is_finite()));
        assert_eq!(s.len(), 3, "three tokens should have entered the kv cache");

        let _ = std::fs::remove_dir_all(&f.dir);
    }

    #[test]
    fn empty_context_is_an_error() {
        let f = fixture("empty");
        let mut s = seq(f.dims);
        assert!(matches!(
            f.engine.logits(&[], &mut s).unwrap_err(),
            GarudaError::Inference(_)
        ));
        let _ = std::fs::remove_dir_all(&f.dir);
    }

    #[test]
    fn out_of_vocab_token_is_an_error_not_a_panic() {
        let f = fixture("oov");
        let mut s = seq(f.dims);
        let bad = f.dims.vocab_size as Token + 5;
        assert!(matches!(
            f.engine.logits(&[bad], &mut s).unwrap_err(),
            GarudaError::InvalidToken(_)
        ));
        let _ = std::fs::remove_dir_all(&f.dir);
    }

    #[test]
    fn incremental_decode_matches_a_full_recompute() {
        // The whole point of the kv cache: feeding tokens one at a time must give
        // the same logits as processing the entire context in one call.
        let f = fixture("incremental");
        let ctx = [7u32, 42, 99, 13];

        let mut whole = seq(f.dims);
        let full = f.engine.logits(&ctx, &mut whole).unwrap();

        let mut piecewise = seq(f.dims);
        let mut last = None;
        for i in 1..=ctx.len() {
            last = Some(f.engine.logits(&ctx[..i], &mut piecewise).unwrap());
        }
        let incremental = last.unwrap();

        for (a, b) in full.data().iter().zip(incremental.data().iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
        let _ = std::fs::remove_dir_all(&f.dir);
    }

    #[test]
    fn only_unseen_tokens_are_consumed() {
        let f = fixture("resume");
        let mut s = seq(f.dims);

        f.engine.logits(&[1, 2, 3], &mut s).unwrap();
        assert_eq!(s.len(), 3);

        f.engine.logits(&[1, 2, 3, 4], &mut s).unwrap();
        assert_eq!(s.len(), 4, "only the new token should have been processed");

        // Re-asking for a context the cache has fully absorbed is an error, not a
        // silent recompute against a stale cache.
        assert!(f.engine.logits(&[1, 2, 3, 4], &mut s).is_err());

        let _ = std::fs::remove_dir_all(&f.dir);
    }

    #[test]
    fn different_contexts_produce_different_logits() {
        let f = fixture("distinct");
        let mut a = seq(f.dims);
        let mut b = seq(f.dims);

        let la = f.engine.logits(&[5, 6, 7], &mut a).unwrap();
        let lb = f.engine.logits(&[90, 91, 92], &mut b).unwrap();
        assert_ne!(la.data(), lb.data(), "the model ignored its input");

        let _ = std::fs::remove_dir_all(&f.dir);
    }

    #[test]
    fn forward_is_deterministic() {
        let f = fixture("determinism");
        let mut a = seq(f.dims);
        let mut b = seq(f.dims);
        assert_eq!(
            f.engine.logits(&[3, 1, 4], &mut a).unwrap().data(),
            f.engine.logits(&[3, 1, 4], &mut b).unwrap().data()
        );
        let _ = std::fs::remove_dir_all(&f.dir);
    }

    #[test]
    fn only_top_k_experts_are_loaded_per_token() {
        // "Expert streaming": a token must not pull the whole layer into RAM.
        let dims = ModelDims::default();
        let dir = std::env::temp_dir().join("garuda_moe_topk");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let l2: Arc<dyn StorageBackend> = Arc::new(LocalStorageBackend::new(&dir));
        let budget = Expert::n_params(&dims) * 4 * dims.n_experts;
        let mm = Arc::new(MemoryManager::new(dims, budget, l2, None).unwrap());
        let weights = Arc::new(ModelWeights::synthesize(dims).unwrap());
        let router = Router::new(RouterType::Mixtral, dims).unwrap();
        let engine = MoeEngine::new(dims, weights, router, mm.clone(), None).unwrap();

        let mut s = seq(dims);
        engine.logits(&[11], &mut s).unwrap();

        let c = mm.tier_counts();
        let loaded = c.l1 + c.l2 + c.l3 + c.synthesised;
        assert_eq!(
            loaded as usize, dims.top_k,
            "one token loaded {loaded} experts; top_k is {}",
            dims.top_k
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn context_window_exhaustion_is_an_error_not_a_panic() {
        let f = fixture("ctxfull");
        let mut s = SeqState::new(KvConfig::mha(f.dims, 2, 4, None, None), 1);
        let err = f.engine.logits(&[1, 2, 3], &mut s).unwrap_err();
        assert!(matches!(err, GarudaError::Cache(_)), "got {err:?}");
        let _ = std::fs::remove_dir_all(&f.dir);
    }
}

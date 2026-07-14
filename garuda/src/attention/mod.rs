//! Causal multi-head attention with rotary embeddings, over the paged KV cache.
//!
//! This is the real operation: `softmax(QKᵀ/√d) · V`, restricted to positions at
//! or before the current one and, when configured, inside the sliding window.
//! The weights are untrained (see [`crate::weights`]); the arithmetic is not.

use crate::cache::KVCacheState;
use crate::core::{GarudaError, ModelDims};
use crate::simd;
use crate::weights::ModelWeights;

pub struct Attention {
    dims: ModelDims,
}

impl Attention {
    pub fn new(dims: ModelDims) -> Result<Self, GarudaError> {
        dims.validate()?;
        Ok(Self { dims })
    }

    pub fn dims(&self) -> ModelDims {
        self.dims
    }

    /// Attend from the token whose hidden state is `x`, appending its key/value to
    /// `kv` and returning the `d_model`-dim attention output.
    pub fn forward(
        &self,
        x: &[f32],
        w: &ModelWeights,
        kv: &mut KVCacheState,
    ) -> Result<Vec<f32>, GarudaError> {
        let ModelDims {
            d_model: d,
            n_heads,
            head_dim,
            rope_theta,
            ..
        } = self.dims;

        if x.len() != d {
            return Err(GarudaError::Inference(format!(
                "attention expects a {d}-dim input, got {}",
                x.len()
            )));
        }

        let mut q = vec![0.0; d];
        let mut k = vec![0.0; d];
        let mut v = vec![0.0; d];
        simd::matvec(&w.wq, d, d, x, &mut q);
        simd::matvec(&w.wk, d, d, x, &mut k);
        simd::matvec(&w.wv, d, d, x, &mut v);

        // Position of the token we are about to add.
        let pos = kv.len();
        for h in 0..n_heads {
            let r = h * head_dim..(h + 1) * head_dim;
            simd::rope(&mut q[r.clone()], pos, rope_theta);
            simd::rope(&mut k[r], pos, rope_theta);
        }

        kv.append(&k, &v)?;

        // Attend over [start, pos]. `attention_start` applies the sliding window;
        // blocks in that range may be on disk, so page them back in first.
        let start = kv.attention_start();
        let end = pos + 1;
        kv.ensure_resident(start, end)?;

        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut context = vec![0.0; d];

        for h in 0..n_heads {
            let hr = h * head_dim..(h + 1) * head_dim;
            let q_h = &q[hr.clone()];

            let mut scores = Vec::with_capacity(end - start);
            for j in start..end {
                let key = kv.key_at(j).ok_or_else(|| {
                    GarudaError::Cache(format!("key at position {j} is missing from the kv cache"))
                })?;
                scores.push(simd::dot(q_h, &key[hr.clone()]) * scale);
            }
            simd::softmax(&mut scores);

            let out_h = &mut context[hr.clone()];
            for (j, &p) in (start..end).zip(scores.iter()) {
                let value = kv.value_at(j).ok_or_else(|| {
                    GarudaError::Cache(format!(
                        "value at position {j} is missing from the kv cache"
                    ))
                })?;
                simd::add_scaled(out_h, &value[hr.clone()], p);
            }
        }

        let mut out = vec![0.0; d];
        simd::matvec(&w.wo, d, d, &context, &mut out);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{KVCacheState, KvConfig};

    fn cfg(dims: ModelDims, sliding_window: Option<usize>) -> KvConfig {
        KvConfig::mha(dims, 256, 64, sliding_window, None)
    }

    fn setup() -> (Attention, ModelWeights, KVCacheState) {
        let dims = ModelDims::default();
        (
            Attention::new(dims).unwrap(),
            ModelWeights::synthesize(dims).unwrap(),
            KVCacheState::new(cfg(dims, None), 1),
        )
    }

    #[test]
    fn rejects_wrong_width_instead_of_panicking() {
        let (attn, w, mut kv) = setup();
        // The previous implementation read shape[0] before validating, and panicked.
        assert!(matches!(
            attn.forward(&[], &w, &mut kv).unwrap_err(),
            GarudaError::Inference(_)
        ));
        assert!(matches!(
            attn.forward(&[1.0, 2.0], &w, &mut kv).unwrap_err(),
            GarudaError::Inference(_)
        ));
        assert_eq!(kv.len(), 0, "a rejected call must not touch the cache");
    }

    #[test]
    fn appends_exactly_one_position_per_call_and_stays_finite() {
        let (attn, w, mut kv) = setup();
        let d = attn.dims().d_model;
        for step in 0..5 {
            let x: Vec<f32> = (0..d).map(|i| ((step + i) as f32).sin()).collect();
            let out = attn.forward(&x, &w, &mut kv).unwrap();
            assert_eq!(out.len(), d);
            assert!(out.iter().all(|v| v.is_finite()));
            assert_eq!(kv.len(), step + 1);
        }
    }

    #[test]
    fn first_token_attends_only_to_itself() {
        // With a single position, softmax over one score is exactly 1.0, so the
        // context is that token's own value vector: out == Wo · (Wv · x).
        let (attn, w, mut kv) = setup();
        let d = attn.dims().d_model;
        let x: Vec<f32> = (0..d).map(|i| (i as f32 * 0.05).cos()).collect();

        let got = attn.forward(&x, &w, &mut kv).unwrap();

        let mut v = vec![0.0; d];
        simd::matvec(&w.wv, d, d, &x, &mut v);
        let mut expect = vec![0.0; d];
        simd::matvec(&w.wo, d, d, &v, &mut expect);

        for (a, b) in got.iter().zip(expect.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn is_causal_earlier_output_is_unaffected_by_later_tokens() {
        let dims = ModelDims::default();
        let d = dims.d_model;
        let attn = Attention::new(dims).unwrap();
        let w = ModelWeights::synthesize(dims).unwrap();
        let inputs: Vec<Vec<f32>> = (0..4)
            .map(|s| (0..d).map(|i| ((s * 7 + i) as f32 * 0.03).sin()).collect())
            .collect();

        let mut kv_a = KVCacheState::new(cfg(dims, None), 1);
        attn.forward(&inputs[0], &w, &mut kv_a).unwrap();
        let short = attn.forward(&inputs[1], &w, &mut kv_a).unwrap();

        // Run all four. Token 1's output must be identical: a causal model cannot
        // see positions 2 and 3 from position 1.
        let mut kv_b = KVCacheState::new(cfg(dims, None), 2);
        attn.forward(&inputs[0], &w, &mut kv_b).unwrap();
        let long = attn.forward(&inputs[1], &w, &mut kv_b).unwrap();
        attn.forward(&inputs[2], &w, &mut kv_b).unwrap();
        attn.forward(&inputs[3], &w, &mut kv_b).unwrap();

        for (a, b) in short.iter().zip(long.iter()) {
            assert!((a - b).abs() < 1e-6, "causality violated: {a} vs {b}");
        }
    }

    #[test]
    fn sliding_window_bounds_the_attended_range() {
        let dims = ModelDims::default();
        let d = dims.d_model;
        let attn = Attention::new(dims).unwrap();
        let w = ModelWeights::synthesize(dims).unwrap();

        let x: Vec<f32> = (0..d).map(|i| (i as f32 * 0.02).sin()).collect();
        let mut windowed = KVCacheState::new(cfg(dims, Some(2)), 1);
        let mut full = KVCacheState::new(cfg(dims, None), 2);

        for _ in 0..6 {
            assert!(
                attn.forward(&x, &w, &mut windowed)
                    .unwrap()
                    .iter()
                    .all(|v| v.is_finite())
            );
            assert!(
                attn.forward(&x, &w, &mut full)
                    .unwrap()
                    .iter()
                    .all(|v| v.is_finite())
            );
        }
        assert_eq!(
            windowed.attention_start(),
            4,
            "window of 2 over 6 positions"
        );
        assert_eq!(full.attention_start(), 0);
    }
}

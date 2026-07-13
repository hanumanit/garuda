//! MoE gating: score every expert against the hidden state, keep the top-k.
//!
//! The router types differ in *where* the softmax sits relative to top-k
//! selection, which is the real distinction between these architectures:
//!
//! - `Mixtral` — take the top-k logits, then softmax over just those k.
//! - `DeepSeek` — softmax over all experts, take the top-k, use those affinities as
//!   they are. They sum to less than 1; the remainder is the mass the model assigned
//!   to experts it did not select.
//! - `QwenMoe` — softmax over all experts, take the top-k, then renormalise.

use crate::core::{ExpertId, GarudaError, ModelDims};
use crate::simd;
use crate::weights::ModelWeights;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterType {
    Mixtral,
    DeepSeek,
    QwenMoe,
}

impl FromStr for RouterType {
    type Err = GarudaError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().replace(['-', '_'], "").as_str() {
            "mixtral" => Ok(Self::Mixtral),
            "deepseek" => Ok(Self::DeepSeek),
            "qwen" | "qwenmoe" => Ok(Self::QwenMoe),
            other => Err(GarudaError::Config(format!(
                "unknown router '{other}' (expected mixtral, deepseek or qwen)"
            ))),
        }
    }
}

pub struct Router {
    router_type: RouterType,
    dims: ModelDims,
}

impl Router {
    pub fn new(router_type: RouterType, dims: ModelDims) -> Result<Self, GarudaError> {
        dims.validate()?;
        Ok(Self { router_type, dims })
    }

    pub fn router_type(&self) -> RouterType {
        self.router_type
    }

    /// The `top_k` experts for this hidden state, with their gate weights.
    ///
    /// Weights are non-negative. They sum to 1 for `Mixtral` and `QwenMoe`; for
    /// `DeepSeek` they sum to at most 1, by design.
    pub fn route(
        &self,
        hidden: &[f32],
        w: &ModelWeights,
    ) -> Result<Vec<(ExpertId, f32)>, GarudaError> {
        let (d, n, k) = (self.dims.d_model, self.dims.n_experts, self.dims.top_k);
        if hidden.len() != d {
            return Err(GarudaError::Inference(format!(
                "router expects a {d}-dim input, got {}",
                hidden.len()
            )));
        }

        let mut scores = vec![0.0f32; n];
        simd::matvec(&w.router, n, d, hidden, &mut scores);

        if matches!(self.router_type, RouterType::DeepSeek | RouterType::QwenMoe) {
            simd::softmax(&mut scores);
        }

        let mut ranked: Vec<(ExpertId, f32)> = scores
            .iter()
            .enumerate()
            .map(|(i, &s)| (i as ExpertId, s))
            .collect();
        // Ties break on expert id, so routing is deterministic for a given input.
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        ranked.truncate(k);

        match self.router_type {
            RouterType::Mixtral => {
                let mut top: Vec<f32> = ranked.iter().map(|(_, s)| *s).collect();
                simd::softmax(&mut top);
                for ((_, s), p) in ranked.iter_mut().zip(top) {
                    *s = p;
                }
            }
            RouterType::DeepSeek => {
                // Already softmaxed across all experts; the affinities stand as they are.
            }
            RouterType::QwenMoe => {
                let sum: f32 = ranked.iter().map(|(_, s)| *s).sum();
                if sum > 0.0 {
                    for (_, s) in ranked.iter_mut() {
                        *s /= sum;
                    }
                } else {
                    let uniform = 1.0 / k as f32;
                    for (_, s) in ranked.iter_mut() {
                        *s = uniform;
                    }
                }
            }
        }

        Ok(ranked)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup(rt: RouterType) -> (Router, ModelWeights, Vec<f32>) {
        let dims = ModelDims::default();
        let w = ModelWeights::synthesize(dims).unwrap();
        let h: Vec<f32> = (0..dims.d_model).map(|i| (i as f32 * 0.1).sin()).collect();
        (Router::new(rt, dims).unwrap(), w, h)
    }

    #[test]
    fn selects_top_k_distinct_experts_in_range() {
        let dims = ModelDims::default();
        let (r, w, h) = setup(RouterType::Mixtral);
        let routed = r.route(&h, &w).unwrap();

        assert_eq!(routed.len(), dims.top_k);
        let ids: Vec<_> = routed.iter().map(|(id, _)| *id).collect();
        let mut uniq = ids.clone();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(uniq.len(), ids.len(), "an expert was selected twice");
        assert!(ids.iter().all(|&id| (id as usize) < dims.n_experts));
    }

    #[test]
    fn mixtral_and_qwen_sum_to_one_deepseek_does_not_exceed_it() {
        for rt in [RouterType::Mixtral, RouterType::QwenMoe] {
            let (r, w, h) = setup(rt);
            let sum: f32 = r.route(&h, &w).unwrap().iter().map(|(_, s)| s).sum();
            assert!((sum - 1.0).abs() < 1e-5, "{rt:?} weights summed to {sum}");
        }

        let (r, w, h) = setup(RouterType::DeepSeek);
        let routed = r.route(&h, &w).unwrap();
        let sum: f32 = routed.iter().map(|(_, s)| s).sum();
        assert!(sum > 0.0 && sum <= 1.0 + 1e-5, "deepseek summed to {sum}");
        assert!(routed.iter().all(|(_, s)| *s >= 0.0));
    }

    #[test]
    fn ranking_is_descending_and_deterministic() {
        let (r, w, h) = setup(RouterType::Mixtral);
        assert_eq!(r.route(&h, &w).unwrap(), r.route(&h, &w).unwrap());
        for pair in r.route(&h, &w).unwrap().windows(2) {
            assert!(pair[0].1 >= pair[1].1, "weights must be descending");
        }
    }

    #[test]
    fn different_inputs_can_select_different_experts() {
        let dims = ModelDims::default();
        let r = Router::new(RouterType::Mixtral, dims).unwrap();
        let w = ModelWeights::synthesize(dims).unwrap();

        let mut seen = std::collections::HashSet::new();
        for s in 0..32 {
            let h: Vec<f32> = (0..dims.d_model)
                .map(|i| ((s * 13 + i) as f32 * 0.21).sin())
                .collect();
            for (id, _) in r.route(&h, &w).unwrap() {
                seen.insert(id);
            }
        }
        assert!(
            seen.len() > 1,
            "routing collapsed onto one expert for every input"
        );
    }

    #[test]
    fn rejects_wrong_width_input() {
        let (r, w, _) = setup(RouterType::Mixtral);
        assert!(matches!(
            r.route(&[1.0, 2.0], &w).unwrap_err(),
            GarudaError::Inference(_)
        ));
    }

    #[test]
    fn parses_from_config_strings() {
        assert_eq!(
            "mixtral".parse::<RouterType>().unwrap(),
            RouterType::Mixtral
        );
        assert_eq!(
            "DeepSeek".parse::<RouterType>().unwrap(),
            RouterType::DeepSeek
        );
        assert_eq!(
            "qwen-moe".parse::<RouterType>().unwrap(),
            RouterType::QwenMoe
        );
        assert!("llama".parse::<RouterType>().is_err());
    }
}

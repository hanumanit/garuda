//! Expert-activation predictor: a first-order Markov model over expert usage.
//!
//! Every decode step tells the predictor which experts actually fired. It keeps a
//! transition count `C[prev][next]`, and predicts the next step's experts as the
//! highest-count successors of the experts that fired last. Cold, it predicts
//! nothing and the prefetcher stays quiet — no made-up guesses.
//!
//! The point is to hide L2/L3 load latency: the experts for step *n+1* start
//! loading while step *n* is still computing. Being wrong costs a wasted load,
//! not a wrong answer, so `predict` is free to be approximate.

use crate::core::ExpertId;
use parking_lot::RwLock;

pub struct ExpertPredictor {
    n_experts: usize,
    /// Row-major `[n_experts, n_experts]` transition counts.
    counts: RwLock<Vec<u32>>,
    hits: RwLock<PredictStats>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PredictStats {
    /// Experts predicted that were then actually used.
    pub correct: u64,
    /// Experts predicted that were not used.
    pub wasted: u64,
    /// Experts used that were not predicted.
    pub missed: u64,
}

impl PredictStats {
    /// Share of predicted experts that turned out to be needed.
    pub fn precision(&self) -> f64 {
        let total = self.correct + self.wasted;
        if total == 0 {
            0.0
        } else {
            self.correct as f64 / total as f64
        }
    }

    /// Share of needed experts that had been predicted.
    pub fn recall(&self) -> f64 {
        let total = self.correct + self.missed;
        if total == 0 {
            0.0
        } else {
            self.correct as f64 / total as f64
        }
    }
}

impl ExpertPredictor {
    pub fn new(n_experts: usize) -> Self {
        let n = n_experts.max(1);
        Self {
            n_experts: n,
            counts: RwLock::new(vec![0; n * n]),
            hits: RwLock::new(PredictStats::default()),
        }
    }

    pub fn n_experts(&self) -> usize {
        self.n_experts
    }

    /// Learn from an observed step: `prev` fired, then `next` fired.
    pub fn observe(&self, prev: &[ExpertId], next: &[ExpertId]) {
        let n = self.n_experts;
        let mut counts = self.counts.write();
        for &p in prev {
            for &q in next {
                let (p, q) = (p as usize, q as usize);
                if p < n && q < n {
                    counts[p * n + q] = counts[p * n + q].saturating_add(1);
                }
            }
        }
    }

    /// The `k` most likely experts to fire after `current`.
    ///
    /// Empty when the model has seen nothing relevant — an untrained predictor
    /// should stay silent rather than send the prefetcher after arbitrary experts.
    pub fn predict(&self, current: &[ExpertId], k: usize) -> Vec<ExpertId> {
        if current.is_empty() || k == 0 {
            return Vec::new();
        }
        let n = self.n_experts;
        let counts = self.counts.read();

        let mut score = vec![0u64; n];
        for &p in current {
            let p = p as usize;
            if p >= n {
                continue;
            }
            for (q, s) in score.iter_mut().enumerate() {
                *s += counts[p * n + q] as u64;
            }
        }

        let mut ranked: Vec<(ExpertId, u64)> = score
            .iter()
            .enumerate()
            .filter(|&(_, &s)| s > 0)
            .map(|(i, &s)| (i as ExpertId, s))
            .collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        ranked.truncate(k);
        ranked.into_iter().map(|(id, _)| id).collect()
    }

    /// Score a prediction against what actually happened.
    pub fn score(&self, predicted: &[ExpertId], actual: &[ExpertId]) {
        let mut s = self.hits.write();
        for p in predicted {
            if actual.contains(p) {
                s.correct += 1;
            } else {
                s.wasted += 1;
            }
        }
        for a in actual {
            if !predicted.contains(a) {
                s.missed += 1;
            }
        }
    }

    pub fn stats(&self) -> PredictStats {
        *self.hits.read()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicts_nothing_before_it_has_learned_anything() {
        let p = ExpertPredictor::new(8);
        assert!(p.predict(&[0, 1], 2).is_empty());
        assert!(p.predict(&[], 2).is_empty());
    }

    #[test]
    fn learns_a_repeated_transition() {
        let p = ExpertPredictor::new(8);
        // Experts {0,1} are consistently followed by {3,4}.
        for _ in 0..10 {
            p.observe(&[0, 1], &[3, 4]);
        }
        let got = p.predict(&[0, 1], 2);
        assert_eq!(got.len(), 2);
        assert!(got.contains(&3) && got.contains(&4), "got {got:?}");
    }

    #[test]
    fn prefers_the_more_frequent_successor() {
        let p = ExpertPredictor::new(4);
        for _ in 0..9 {
            p.observe(&[0], &[1]);
        }
        p.observe(&[0], &[2]);
        assert_eq!(p.predict(&[0], 1), vec![1]);
        assert_eq!(p.predict(&[0], 2), vec![1, 2]);
    }

    #[test]
    fn ignores_out_of_range_expert_ids() {
        let p = ExpertPredictor::new(4);
        p.observe(&[99], &[1]);
        p.observe(&[0], &[99]);
        assert!(p.predict(&[99], 2).is_empty());
        assert!(
            p.predict(&[0], 2).is_empty(),
            "the 99 transition must not be recorded"
        );
    }

    #[test]
    fn saturating_counts_do_not_overflow() {
        let p = ExpertPredictor::new(2);
        {
            let mut c = p.counts.write();
            c[0] = u32::MAX;
        }
        p.observe(&[0], &[0]); // would overflow a wrapping add
        assert_eq!(p.predict(&[0], 1), vec![0]);
    }

    #[test]
    fn scoring_tracks_precision_and_recall() {
        let p = ExpertPredictor::new(8);
        p.score(&[1, 2], &[2, 3]); // 1 correct (2), 1 wasted (1), 1 missed (3)
        let s = p.stats();
        assert_eq!(s.correct, 1);
        assert_eq!(s.wasted, 1);
        assert_eq!(s.missed, 1);
        assert!((s.precision() - 0.5).abs() < 1e-9);
        assert!((s.recall() - 0.5).abs() < 1e-9);
    }
}

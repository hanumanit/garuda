//! Expert prefetching.
//!
//! After each decode step the engine asks the predictor which experts the *next*
//! step will probably need, and warms them on a rayon worker while the current
//! step is still finishing. A wrong guess costs one wasted load; it can never
//! change the answer, because the forward pass loads what it actually needs
//! regardless of what was prefetched.
//!
//! Prefetches are deduplicated: an expert that is already resident, or already in
//! flight, is not fetched again.

use crate::core::{ExpertId, ExpertLoader};
use crate::predictor::{ExpertPredictor, PredictStats};
use parking_lot::Mutex;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct PrefetchEngine {
    loader: Arc<dyn ExpertLoader>,
    predictor: Arc<ExpertPredictor>,
    enabled: bool,
    depth: usize,
    /// Shared with the spawned workers, which outlive any borrow of `self`.
    inflight: Arc<Mutex<HashSet<ExpertId>>>,
    launched: AtomicU64,
    skipped: AtomicU64,
}

impl PrefetchEngine {
    /// `depth` is how many experts to warm per step — usually the model's `top_k`.
    pub fn new(
        loader: Arc<dyn ExpertLoader>,
        predictor: Arc<ExpertPredictor>,
        enabled: bool,
        depth: usize,
    ) -> Self {
        Self {
            loader,
            predictor,
            enabled,
            depth,
            inflight: Arc::new(Mutex::new(HashSet::new())),
            launched: AtomicU64::new(0),
            skipped: AtomicU64::new(0),
        }
    }

    pub fn predictor(&self) -> &Arc<ExpertPredictor> {
        &self.predictor
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Experts whose background load has been started (cumulative).
    pub fn launched(&self) -> u64 {
        self.launched.load(Ordering::Relaxed)
    }

    /// Predictions dropped because the expert was already resident or in flight.
    pub fn skipped(&self) -> u64 {
        self.skipped.load(Ordering::Relaxed)
    }

    pub fn predictor_stats(&self) -> PredictStats {
        self.predictor.stats()
    }

    /// Record one decode step and warm what is likely to come next.
    ///
    /// `previous` and `used` are the experts that fired on the last step and this
    /// one. `predicted_last_step` is what we guessed before seeing `used`; it is
    /// scored against reality. Returns this step's prediction, to be handed back
    /// on the next call.
    pub fn observe_step(
        &self,
        previous: &[ExpertId],
        used: &[ExpertId],
        predicted_last_step: &[ExpertId],
    ) -> Vec<ExpertId> {
        if !self.enabled {
            return Vec::new();
        }

        if !predicted_last_step.is_empty() {
            self.predictor.score(predicted_last_step, used);
        }
        if !previous.is_empty() {
            self.predictor.observe(previous, used);
        }

        let predicted = self.predictor.predict(used, self.depth);
        for &id in &predicted {
            self.warm(id);
        }
        predicted
    }

    /// Start a background load for `id`, unless it is resident or already in flight.
    fn warm(&self, id: ExpertId) {
        if self.loader.is_resident(id) {
            self.skipped.fetch_add(1, Ordering::Relaxed);
            return;
        }
        // Claim the slot before spawning, so two steps cannot both launch the same
        // load. The worker releases it.
        if !self.inflight.lock().insert(id) {
            self.skipped.fetch_add(1, Ordering::Relaxed);
            return;
        }

        self.launched.fetch_add(1, Ordering::Relaxed);
        let loader = self.loader.clone();
        let inflight = self.inflight.clone();
        rayon::spawn(move || {
            if let Err(e) = loader.prefetch(id) {
                tracing::debug!(
                    expert = id,
                    error = %e,
                    "prefetch failed; the forward pass will load it"
                );
            }
            inflight.lock().remove(&id);
        });
    }

    /// Block until every in-flight prefetch has finished. Tests only.
    #[cfg(test)]
    fn drain(&self) {
        while !self.inflight.lock().is_empty() {
            std::thread::yield_now();
        }
    }
}

/// An [`ExpertLoader`] for a real, mmapped GGUF checkpoint: "loading" an expert
/// means touching the mmap pages its packed weights live on, so the page fault
/// happens now, on a background rayon worker, instead of synchronously the next
/// time the forward pass actually dots against them.
///
/// It never materialises an [`crate::core::Expert`] — `LlamaBackend` reads straight
/// out of the same mmap via `Weight::Packed`, so there is nothing to hand back.
/// `load`/`unload` exist only to satisfy the trait; [`PrefetchEngine`] never calls
/// them (only `prefetch`/`is_resident`), so they error/no-op rather than pretend.
pub struct GgufPagePrefetcher {
    mmap: Arc<memmap2::Mmap>,
    /// `ranges[id]` = the byte ranges (start, len) to warm for expert `id`, where
    /// `id` is `layer * n_experts + expert`. Empty for a dense layer or an
    /// out-of-range id.
    ranges: Vec<Vec<(usize, usize)>>,
}

impl GgufPagePrefetcher {
    pub fn new(mmap: Arc<memmap2::Mmap>, ranges: Vec<Vec<(usize, usize)>>) -> Self {
        Self { mmap, ranges }
    }
}

impl ExpertLoader for GgufPagePrefetcher {
    fn load(&self, id: ExpertId) -> Result<Arc<crate::core::Expert>, crate::core::GarudaError> {
        Err(crate::core::GarudaError::Model(format!(
            "GgufPagePrefetcher only warms mmap pages for expert {id}; it never \
             materialises an Expert, and nothing should call load() on it"
        )))
    }

    fn unload(&self, _id: ExpertId) {}

    fn prefetch(&self, id: ExpertId) -> Result<(), crate::core::GarudaError> {
        // One touch per (likely) page is enough to fault it in; page size varies by
        // platform, so stride conservatively rather than query it.
        const STRIDE: usize = 4096;
        let Some(ranges) = self.ranges.get(id as usize) else {
            return Ok(());
        };
        let mut sink = 0u8;
        for &(start, len) in ranges {
            let end = (start + len).min(self.mmap.len());
            let mut i = start;
            while i < end {
                sink ^= self.mmap.get(i).copied().unwrap_or(0);
                i += STRIDE;
            }
        }
        // Nothing reads `sink`; without this the touching loop above is dead code an
        // optimiser is free to remove entirely.
        std::hint::black_box(sink);
        Ok(())
    }

    fn is_resident(&self, _id: ExpertId) -> bool {
        // Unknown from user space, and a wrong "yes" would skip a genuinely cold
        // expert. Touching an already-hot page costs a cheap page-table lookup, so
        // always attempting is the safe default; `PrefetchEngine`'s own in-flight
        // set still dedupes concurrent requests for the same id.
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Expert, GarudaError, ModelDims};
    use crate::weights::synthesize_expert;
    use std::sync::atomic::AtomicUsize;

    /// Counts what the engine asks for, without touching a disk.
    struct SpyLoader {
        resident: Mutex<HashSet<ExpertId>>,
        prefetch_calls: AtomicUsize,
    }

    impl SpyLoader {
        fn new() -> Self {
            Self {
                resident: Mutex::new(HashSet::new()),
                prefetch_calls: AtomicUsize::new(0),
            }
        }
        fn calls(&self) -> usize {
            self.prefetch_calls.load(Ordering::SeqCst)
        }
    }

    impl ExpertLoader for SpyLoader {
        fn load(&self, id: ExpertId) -> Result<Arc<Expert>, GarudaError> {
            self.resident.lock().insert(id);
            Ok(Arc::new(synthesize_expert(id, &ModelDims::default())))
        }
        fn unload(&self, id: ExpertId) {
            self.resident.lock().remove(&id);
        }
        fn prefetch(&self, id: ExpertId) -> Result<(), GarudaError> {
            self.prefetch_calls.fetch_add(1, Ordering::SeqCst);
            self.load(id).map(|_| ())
        }
        fn is_resident(&self, id: ExpertId) -> bool {
            self.resident.lock().contains(&id)
        }
    }

    fn engine(enabled: bool) -> (Arc<SpyLoader>, PrefetchEngine) {
        let spy = Arc::new(SpyLoader::new());
        let predictor = Arc::new(ExpertPredictor::new(8));
        let e = PrefetchEngine::new(spy.clone(), predictor, enabled, 2);
        (spy, e)
    }

    #[test]
    fn disabled_engine_does_nothing() {
        let (spy, e) = engine(false);
        for _ in 0..5 {
            e.observe_step(&[0], &[1], &[]);
        }
        assert_eq!(e.launched(), 0);
        assert_eq!(spy.calls(), 0);
    }

    #[test]
    fn cold_engine_makes_no_prediction() {
        let (spy, e) = engine(true);
        let predicted = e.observe_step(&[], &[0, 1], &[]);
        assert!(
            predicted.is_empty(),
            "an untrained predictor must stay quiet"
        );
        e.drain();
        assert_eq!(spy.calls(), 0);
    }

    #[test]
    fn warms_the_experts_a_learned_pattern_implies() {
        let (spy, e) = engine(true);

        // Teach it {0,1} -> {4,5}, alternating, without ever letting the loader
        // keep anything resident (so every prediction is a real fetch).
        let mut predicted = Vec::new();
        let mut prev: Vec<ExpertId> = Vec::new();
        for step in 0..12 {
            let used: Vec<ExpertId> = if step % 2 == 0 {
                vec![0, 1]
            } else {
                vec![4, 5]
            };
            predicted = e.observe_step(&prev, &used, &predicted);
            prev = used;
            spy.unload(4);
            spy.unload(5);
            spy.unload(0);
            spy.unload(1);
        }
        e.drain();

        assert!(e.launched() > 0, "nothing was ever prefetched");
        assert!(
            spy.calls() > 0,
            "the loader was never asked to warm anything"
        );

        let stats = e.predictor_stats();
        assert!(
            stats.correct > 0,
            "a perfectly regular pattern was never predicted"
        );
        assert!(
            stats.precision() > 0.5,
            "precision {:.2} on a deterministic alternating pattern",
            stats.precision()
        );
    }

    #[test]
    fn does_not_refetch_a_resident_expert() {
        let (spy, e) = engine(true);

        // Teach a two-cycle so both experts have a recorded successor and both end
        // up fetched (the spy keeps whatever it loads).
        for _ in 0..10 {
            e.observe_step(&[1], &[0], &[]);
            e.observe_step(&[0], &[1], &[]);
        }
        e.drain();
        assert!(
            spy.is_resident(0) && spy.is_resident(1),
            "the pattern should have warmed both experts"
        );

        let calls_before = spy.calls();
        let skips_before = e.skipped();

        // Predicting an expert that is already in L1 must not fetch it again.
        e.observe_step(&[0], &[1], &[]);
        e.drain();

        assert_eq!(
            spy.calls(),
            calls_before,
            "refetched an expert already in L1"
        );
        assert!(e.skipped() > skips_before, "the skip was not recorded");
    }
}

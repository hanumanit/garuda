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
        let Some(ranges) = self.ranges.get(id as usize) else {
            return Ok(());
        };

        // `madvise(MADV_WILLNEED)` rather than reading a byte per page.
        //
        // Touching pages by hand faulted them in one at a time: one expert of a
        // Mixtral-sized model is ~99 MB, which at a 16 KB page size is over six
        // thousand separate faults, and it dragged every page through the CPU,
        // evicting cache lines the forward pass was still using. Handing the kernel
        // the whole range instead lets it issue large sequential reads. Measured cold
        // over 400 MB, twice with the order reversed: 217 ms / 232 ms for the advice
        // against 1.12 s / 1.43 s for the byte loop — 5-6x.
        //
        // Note this does not return before the I/O does, at least on macOS; the
        // advice is faster, not asynchronous. That is fine here — the point of the
        // prefetcher is to spend a *background* thread on that wait while the
        // foreground step computes — but it is why this is not a fire-and-forget call.
        //
        // Advisory in both directions: a kernel that ignores the hint, or a range it
        // declines, costs nothing. The forward pass still faults whatever it needs.
        for &(start, len) in ranges {
            let len = len.min(self.mmap.len().saturating_sub(start));
            if len == 0 {
                continue;
            }
            if let Err(e) = self
                .mmap
                .advise_range(memmap2::Advice::WillNeed, start, len)
            {
                tracing::debug!(expert = id, error = %e, "madvise(WILLNEED) declined");
            }
        }
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

/// Warms the *next* block's weights while the current one computes.
///
/// A dense model has nothing to predict: every token reads every byte of every
/// block, in order, so the next block is known before the current one starts. What
/// there is to gain is overlap. Left to demand paging, a checkpoint larger than RAM
/// faults its weights in a page at a time with the CPU idle in between — measured on
/// Qwen3.8-27B (19 GB, 16 GB machine): 140% CPU out of 800% available, and 0.7 GB/s
/// out of a disk that reads 3.9 GB/s sequentially. The bytes were the bottleneck and
/// nobody was fetching them ahead.
///
/// So one thread does nothing but hand the kernel whole blocks to read while the
/// others compute. `madvise(WILLNEED)` does not return until the read does (see
/// [`GgufPagePrefetcher`]), which is exactly why it needs a thread of its own.
///
/// Advisory in both directions: a hint that arrives too late, or a kernel that
/// declines it, costs nothing — the forward pass faults whatever it still needs.
pub struct LayerPrefetcher {
    tx: std::sync::mpsc::SyncSender<usize>,
    /// How many blocks ahead to ask for, which is also how many reads can be in
    /// flight at once.
    workers: usize,
    launched: AtomicU64,
    skipped: AtomicU64,
}

impl LayerPrefetcher {
    /// `spans[l]` is the `(start, len)` byte range of block `l`'s weights in `mmap`.
    ///
    /// Spawns [`Self::WORKERS`] threads, which live until this is dropped.
    pub fn new(mmap: Arc<memmap2::Mmap>, spans: Vec<(usize, usize)>) -> Self {
        Self::with_workers(mmap, spans, Self::WORKERS)
    }

    /// How many blocks may be read at once, and therefore how deep the queue to the
    /// drive gets.
    ///
    /// One thread is not enough: `madvise(WILLNEED)` blocks until its read finishes,
    /// so a single worker issues one request at a time and an SSD that wants a deep
    /// queue to reach its rated bandwidth never gets one. Measured on Qwen3.8-27B
    /// (19 GB, 16 GB machine), alternating runs: 27-31 s per forward pass with no
    /// prefetch, 15-16 s with one worker one block ahead, and see
    /// [`Self::with_workers`] for what more of them buys.
    pub const WORKERS: usize = 3;

    pub fn with_workers(
        mmap: Arc<memmap2::Mmap>,
        spans: Vec<(usize, usize)>,
        workers: usize,
    ) -> Self {
        // Bounded by the worker count: a hint that cannot be picked up promptly is a
        // *staler* block by the time anyone gets to it, and the forward pass has
        // already moved on. Dropping it is better than queueing it.
        let (tx, rx) = std::sync::mpsc::sync_channel::<usize>(workers.max(1));
        let rx = Arc::new(Mutex::new(rx));
        let spans = Arc::new(spans);
        for w in 0..workers.max(1) {
            let (rx, spans, mmap) = (rx.clone(), spans.clone(), mmap.clone());
            std::thread::Builder::new()
                .name(format!("garuda-prefetch-{w}"))
                .spawn(move || {
                    loop {
                        // Held only across `recv`, never across the read itself, so
                        // the workers genuinely overlap.
                        let layer = match rx.lock().recv() {
                            Ok(l) => l,
                            Err(_) => return, // the prefetcher was dropped
                        };
                        let Some(&(start, len)) = spans.get(layer) else {
                            continue;
                        };
                        let len = len.min(mmap.len().saturating_sub(start));
                        if len == 0 {
                            continue;
                        }
                        if let Err(e) = mmap.advise_range(memmap2::Advice::WillNeed, start, len) {
                            tracing::debug!(layer, error = %e, "madvise(WILLNEED) declined");
                        }
                    }
                })
                .expect("spawning a prefetch thread");
        }
        Self {
            tx,
            workers: workers.max(1),
            launched: AtomicU64::new(0),
            skipped: AtomicU64::new(0),
        }
    }

    /// Ask for every block this prefetcher can keep in flight after `layer`.
    pub fn hint_ahead(&self, layer: usize) {
        for ahead in 1..=self.workers {
            self.hint(layer + ahead);
        }
    }

    /// Ask for block `layer` to be warmed. Never blocks: if the worker is still on the
    /// previous block, the hint is dropped rather than queued behind it.
    pub fn hint(&self, layer: usize) {
        match self.tx.try_send(layer) {
            Ok(()) => self.launched.fetch_add(1, Ordering::Relaxed),
            Err(_) => self.skipped.fetch_add(1, Ordering::Relaxed),
        };
    }

    /// Blocks handed to the kernel, and hints dropped because the worker was busy.
    /// A high skip count means the reads are slower than the compute they overlap —
    /// which is the case this exists for, so it is not by itself a fault.
    pub fn stats(&self) -> (u64, u64) {
        (
            self.launched.load(Ordering::Relaxed),
            self.skipped.load(Ordering::Relaxed),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Expert, GarudaError, ModelDims};
    use crate::weights::synthesize_expert;
    use std::sync::atomic::AtomicUsize;

    /// The block prefetcher is advisory, so what a test can hold it to is its
    /// bookkeeping: every hint is either handed to the worker or deliberately
    /// dropped, an unknown block is a no-op rather than a panic, and the worker
    /// shuts down with the prefetcher.
    #[test]
    fn the_layer_prefetcher_accounts_for_every_hint() {
        use std::io::Write;

        let dir = std::env::temp_dir().join("garuda_layer_prefetch_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("weights.bin");
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&vec![7u8; 256 * 1024])
            .unwrap();
        let map =
            Arc::new(unsafe { memmap2::Mmap::map(&std::fs::File::open(&path).unwrap()).unwrap() });

        let spans = vec![
            (0, 64 * 1024),
            (64 * 1024, 64 * 1024),
            (128 * 1024, 64 * 1024),
        ];
        let pf = LayerPrefetcher::new(map, spans);

        for l in 0..3 {
            pf.hint(l);
        }
        // Past the end of the block list, and past the end of the file: both are
        // ignored, because a hint is never load-bearing.
        pf.hint(99);

        let (launched, skipped) = pf.stats();
        assert_eq!(launched + skipped, 4, "every hint is accounted for");

        drop(pf); // closes the channel, which ends the worker
        let _ = std::fs::remove_dir_all(&dir);
    }

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

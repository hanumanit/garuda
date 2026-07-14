//! Tiered expert storage: L1 RAM → L2 SSD → L3 archive.
//!
//! A miss in L1 falls through to L2, then L3. Anything found deeper is promoted
//! on the way back up (L3 hits are copied into L2, everything lands in L1). If an
//! expert exists in no tier it is synthesised deterministically and written to L2,
//! so the second request for it takes the real disk path.

use crate::cache::{CacheStats, ExpertCache};
use crate::core::{Expert, ExpertId, ExpertLoader, GarudaError, ModelDims, StorageBackend};
use crate::weights;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTier {
    L1Ram,
    L2Ssd,
    L3Archive,
    /// Not present anywhere; weights were synthesised.
    Synthesised,
}

#[derive(Debug, Default)]
pub struct TierStats {
    pub l1: AtomicU64,
    pub l2: AtomicU64,
    pub l3: AtomicU64,
    pub synthesised: AtomicU64,
    pub prefetched: AtomicU64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TierCounts {
    pub l1: u64,
    pub l2: u64,
    pub l3: u64,
    pub synthesised: u64,
    pub prefetched: u64,
}

pub struct MemoryManager {
    dims: ModelDims,
    l1: ExpertCache,
    l2: Arc<dyn StorageBackend>,
    l3: Option<Arc<dyn StorageBackend>>,
    /// Serialises concurrent loads of the same expert so N requests that miss
    /// together do one disk read rather than N.
    loading: Mutex<HashMap<ExpertId, Arc<Mutex<()>>>>,
    stats: TierStats,
}

impl MemoryManager {
    pub fn new(
        dims: ModelDims,
        l1_budget_bytes: usize,
        l2: Arc<dyn StorageBackend>,
        l3: Option<Arc<dyn StorageBackend>>,
    ) -> Result<Self, GarudaError> {
        dims.validate()?;
        Ok(Self {
            dims,
            l1: ExpertCache::new(l1_budget_bytes),
            l2,
            l3,
            loading: Mutex::new(HashMap::new()),
            stats: TierStats::default(),
        })
    }

    pub fn dims(&self) -> ModelDims {
        self.dims
    }

    pub fn l1_stats(&self) -> CacheStats {
        self.l1.stats()
    }

    pub fn tier_counts(&self) -> TierCounts {
        TierCounts {
            l1: self.stats.l1.load(Ordering::Relaxed),
            l2: self.stats.l2.load(Ordering::Relaxed),
            l3: self.stats.l3.load(Ordering::Relaxed),
            synthesised: self.stats.synthesised.load(Ordering::Relaxed),
            prefetched: self.stats.prefetched.load(Ordering::Relaxed),
        }
    }

    pub fn is_resident(&self, id: ExpertId) -> bool {
        self.l1.contains(id)
    }

    fn expert_path(id: ExpertId) -> PathBuf {
        PathBuf::from(format!("expert_{id}.bin"))
    }

    /// Where `id` currently lives, without loading it.
    pub fn locate(&self, id: ExpertId) -> MemoryTier {
        let path = Self::expert_path(id);
        if self.l1.contains(id) {
            MemoryTier::L1Ram
        } else if self.l2.exists(&path) {
            MemoryTier::L2Ssd
        } else if self.l3.as_ref().is_some_and(|s| s.exists(&path)) {
            MemoryTier::L3Archive
        } else {
            MemoryTier::Synthesised
        }
    }

    /// Read `id` from the deepest tier that has it, promoting as it comes up.
    fn load_uncached(&self, id: ExpertId) -> Result<Arc<Expert>, GarudaError> {
        let path = Self::expert_path(id);

        if self.l2.exists(&path) {
            let bytes = self.l2.read(&path)?;
            let expert = weights::expert_from_bytes(id, &self.dims, &bytes)?;
            self.stats.l2.fetch_add(1, Ordering::Relaxed);
            return Ok(Arc::new(expert));
        }

        if let Some(l3) = &self.l3 {
            if l3.exists(&path) {
                let bytes = l3.read(&path)?;
                let expert = weights::expert_from_bytes(id, &self.dims, &bytes)?;
                // Promote into L2 so the next miss is one tier shallower. A failed
                // promotion is not fatal: we already hold the weights.
                if let Err(e) = self.l2.write(&path, &bytes) {
                    tracing::warn!(expert = id, error = %e, "failed to promote expert into L2");
                }
                self.stats.l3.fetch_add(1, Ordering::Relaxed);
                return Ok(Arc::new(expert));
            }
        }

        // Nowhere on disk. Synthesise deterministically and materialise into L2 so
        // subsequent loads exercise the real read path.
        let expert = weights::synthesize_expert(id, &self.dims);
        if let Err(e) = self.l2.write(&path, &weights::expert_to_bytes(&expert)) {
            tracing::warn!(expert = id, error = %e, "failed to materialise expert into L2");
        }
        self.stats.synthesised.fetch_add(1, Ordering::Relaxed);
        Ok(Arc::new(expert))
    }

    /// Per-expert lock, so a thundering herd on one expert does one load.
    fn load_lock(&self, id: ExpertId) -> Arc<Mutex<()>> {
        self.loading
            .lock()
            .entry(id)
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }
}

impl ExpertLoader for MemoryManager {
    fn load(&self, id: ExpertId) -> Result<Arc<Expert>, GarudaError> {
        if (id as usize) >= self.dims.n_experts {
            return Err(GarudaError::Model(format!(
                "expert {id} does not exist (model has {})",
                self.dims.n_experts
            )));
        }

        if let Some(e) = self.l1.get(id) {
            self.stats.l1.fetch_add(1, Ordering::Relaxed);
            return Ok(e);
        }

        let lock = self.load_lock(id);
        let _guard = lock.lock();

        // Another thread may have loaded it while we waited for the lock.
        if let Some(e) = self.l1.get(id) {
            self.stats.l1.fetch_add(1, Ordering::Relaxed);
            return Ok(e);
        }

        let expert = self.load_uncached(id)?;
        self.l1.insert(id, expert.clone());
        Ok(expert)
    }

    fn unload(&self, id: ExpertId) {
        self.l1.remove(id);
    }

    fn prefetch(&self, id: ExpertId) -> Result<(), GarudaError> {
        if self.l1.contains(id) {
            return Ok(());
        }
        self.load(id)?;
        self.stats.prefetched.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn is_resident(&self, id: ExpertId) -> bool {
        self.l1.contains(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::LocalStorageBackend;
    use std::path::Path;

    struct Tiers {
        mm: MemoryManager,
        l2_dir: PathBuf,
        l3_dir: PathBuf,
    }

    fn tiers(tag: &str, l1_budget: usize) -> Tiers {
        let base = std::env::temp_dir().join(format!("garuda_mem_{tag}"));
        let _ = std::fs::remove_dir_all(&base);
        let l2_dir = base.join("l2");
        let l3_dir = base.join("l3");
        std::fs::create_dir_all(&l2_dir).unwrap();
        std::fs::create_dir_all(&l3_dir).unwrap();

        let mm = MemoryManager::new(
            ModelDims::default(),
            l1_budget,
            Arc::new(LocalStorageBackend::new(&l2_dir)),
            Some(Arc::new(LocalStorageBackend::new(&l3_dir))),
        )
        .unwrap();
        Tiers { mm, l2_dir, l3_dir }
    }

    fn big_budget() -> usize {
        Expert::n_params(&ModelDims::default()) * 4 * 16
    }

    #[test]
    fn first_load_synthesises_then_second_load_hits_l1() {
        let t = tiers("synth", big_budget());
        assert_eq!(t.mm.locate(0), MemoryTier::Synthesised);

        let a = t.mm.load(0).unwrap();
        assert_eq!(t.mm.tier_counts().synthesised, 1);
        assert_eq!(t.mm.locate(0), MemoryTier::L1Ram);

        let b = t.mm.load(0).unwrap();
        assert_eq!(t.mm.tier_counts().l1, 1);
        assert!(
            Arc::ptr_eq(&a, &b),
            "L1 should hand back the same allocation"
        );

        // Synthesis materialised a real file in L2.
        assert!(t.l2_dir.join("expert_0.bin").exists());

        let _ = std::fs::remove_dir_all(t.l2_dir.parent().unwrap());
    }

    #[test]
    fn evicted_expert_is_reloaded_from_l2_not_resynthesised() {
        // Budget for a single expert, so loading a second evicts the first.
        let one = Expert::n_params(&ModelDims::default()) * 4;
        let t = tiers("evict", one);

        let first = t.mm.load(0).unwrap();
        t.mm.load(1).unwrap();
        assert!(!t.mm.is_resident(0), "expert 0 should have been evicted");
        assert_eq!(t.mm.locate(0), MemoryTier::L2Ssd);

        let again = t.mm.load(0).unwrap();
        assert_eq!(t.mm.tier_counts().l2, 1, "should have come back from L2");
        assert_eq!(
            first.gate, again.gate,
            "reload must reproduce the weights exactly"
        );

        let _ = std::fs::remove_dir_all(t.l2_dir.parent().unwrap());
    }

    #[test]
    fn l3_hit_is_promoted_into_l2() {
        let t = tiers("l3", big_budget());
        let dims = ModelDims::default();

        // Seed L3 only.
        let expert = weights::synthesize_expert(5, &dims);
        let bytes = weights::expert_to_bytes(&expert);
        std::fs::write(t.l3_dir.join("expert_5.bin"), &bytes).unwrap();
        assert_eq!(t.mm.locate(5), MemoryTier::L3Archive);

        let loaded = t.mm.load(5).unwrap();
        assert_eq!(t.mm.tier_counts().l3, 1);
        assert_eq!(loaded.down, expert.down);
        assert!(
            t.l2_dir.join("expert_5.bin").exists(),
            "an L3 hit must be promoted into L2"
        );

        let _ = std::fs::remove_dir_all(t.l2_dir.parent().unwrap());
    }

    #[test]
    fn a_corrupt_expert_file_is_an_error_not_a_panic() {
        let t = tiers("corrupt", big_budget());
        // The old loader read the first 100 floats of whatever it found, then divided
        // by the resulting length — a 2-byte file panicked with a zero divisor.
        std::fs::write(t.l2_dir.join("expert_3.bin"), [0u8, 1]).unwrap();

        let err = t.mm.load(3).unwrap_err();
        assert!(matches!(err, GarudaError::Model(_)), "got {err:?}");

        let _ = std::fs::remove_dir_all(t.l2_dir.parent().unwrap());
    }

    #[test]
    fn out_of_range_expert_is_rejected() {
        let t = tiers("range", big_budget());
        let n = ModelDims::default().n_experts as ExpertId;
        assert!(matches!(t.mm.load(n).unwrap_err(), GarudaError::Model(_)));
        let _ = std::fs::remove_dir_all(t.l2_dir.parent().unwrap());
    }

    #[test]
    fn concurrent_misses_on_one_expert_do_a_single_load() {
        let t = Arc::new(tiers("herd", big_budget()));
        let mm = &t.mm;

        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    mm.load(2).unwrap();
                });
            }
        });

        let c = mm.tier_counts();
        assert_eq!(c.synthesised, 1, "expert was built more than once");
        assert_eq!(
            c.l1 + c.synthesised,
            8,
            "every thread should be accounted for"
        );

        let _ = std::fs::remove_dir_all(t.l2_dir.parent().unwrap());
    }

    #[test]
    fn prefetch_warms_l1_and_is_a_no_op_when_already_resident() {
        let t = tiers("prefetch", big_budget());
        assert!(!t.mm.is_resident(4));

        t.mm.prefetch(4).unwrap();
        assert!(t.mm.is_resident(4));
        assert_eq!(t.mm.tier_counts().prefetched, 1);

        t.mm.prefetch(4).unwrap();
        assert_eq!(
            t.mm.tier_counts().prefetched,
            1,
            "second prefetch should do nothing"
        );

        let _ = std::fs::remove_dir_all(t.l2_dir.parent().unwrap());
    }

    #[test]
    fn works_without_an_l3_tier() {
        let base = std::env::temp_dir().join("garuda_mem_nol3");
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();

        let mm = MemoryManager::new(
            ModelDims::default(),
            big_budget(),
            Arc::new(LocalStorageBackend::new(&base)),
            None,
        )
        .unwrap();

        assert!(mm.load(0).is_ok());
        assert!(mm.l2.exists(Path::new("expert_0.bin")));
        let _ = std::fs::remove_dir_all(&base);
    }
}

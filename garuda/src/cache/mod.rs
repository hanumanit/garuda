//! Caches: expert LRU, paged KV cache with real disk spill, and a prompt prefix cache.

use crate::core::{Expert, ExpertId, GarudaError, ModelDims, StorageBackend, Token};
use parking_lot::Mutex;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Expert LRU
// ---------------------------------------------------------------------------

/// Byte-budgeted LRU over loaded experts.
///
/// The budget is in bytes rather than entries because bytes are what the operator
/// actually has (`expert_cache = "8GB"`), and because experts stop being uniformly
/// sized as soon as real per-layer checkpoints are loaded.
pub struct ExpertCache {
    inner: Mutex<ExpertCacheInner>,
    budget_bytes: usize,
}

struct ExpertCacheInner {
    loaded: HashMap<ExpertId, Arc<Expert>>,
    /// Least-recently-used at the front.
    lru: VecDeque<ExpertId>,
    bytes: usize,
    hits: u64,
    misses: u64,
    evictions: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub entries: usize,
    pub bytes: usize,
}

impl CacheStats {
    pub fn hit_ratio(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

impl ExpertCache {
    /// `budget_bytes` is clamped up so at least one byte is allowed; a cache that
    /// cannot hold anything would thrash forever.
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            inner: Mutex::new(ExpertCacheInner {
                loaded: HashMap::new(),
                lru: VecDeque::new(),
                bytes: 0,
                hits: 0,
                misses: 0,
                evictions: 0,
            }),
            budget_bytes: budget_bytes.max(1),
        }
    }

    pub fn get(&self, id: ExpertId) -> Option<Arc<Expert>> {
        let mut inner = self.inner.lock();
        match inner.loaded.get(&id).cloned() {
            Some(e) => {
                inner.hits += 1;
                inner.lru.retain(|&x| x != id);
                inner.lru.push_back(id);
                Some(e)
            }
            None => {
                inner.misses += 1;
                None
            }
        }
    }

    /// Insert `expert`, evicting least-recently-used entries until the budget holds.
    /// Returns the evicted ids.
    pub fn insert(&self, id: ExpertId, expert: Arc<Expert>) -> Vec<ExpertId> {
        let size = expert.size_bytes();
        let mut inner = self.inner.lock();

        if let Some(old) = inner.loaded.remove(&id) {
            inner.bytes -= old.size_bytes();
            inner.lru.retain(|&x| x != id);
        }

        let mut evicted = Vec::new();
        while inner.bytes + size > self.budget_bytes {
            let Some(victim) = inner.lru.pop_front() else {
                break; // Nothing left to evict: this entry alone exceeds the budget.
            };
            if let Some(e) = inner.loaded.remove(&victim) {
                inner.bytes -= e.size_bytes();
                inner.evictions += 1;
                evicted.push(victim);
            }
        }

        inner.bytes += size;
        inner.loaded.insert(id, expert);
        inner.lru.push_back(id);
        evicted
    }

    pub fn remove(&self, id: ExpertId) {
        let mut inner = self.inner.lock();
        if let Some(e) = inner.loaded.remove(&id) {
            inner.bytes -= e.size_bytes();
        }
        inner.lru.retain(|&x| x != id);
    }

    pub fn contains(&self, id: ExpertId) -> bool {
        self.inner.lock().loaded.contains_key(&id)
    }

    pub fn stats(&self) -> CacheStats {
        let inner = self.inner.lock();
        CacheStats {
            hits: inner.hits,
            misses: inner.misses,
            evictions: inner.evictions,
            entries: inner.loaded.len(),
            bytes: inner.bytes,
        }
    }
}

// ---------------------------------------------------------------------------
// Paged KV cache
// ---------------------------------------------------------------------------

/// One page of the KV cache: up to `block_size` contiguous positions.
#[derive(Debug, Clone)]
struct KvBlock {
    /// `filled * d_model` values, row-major by position.
    keys: Vec<f32>,
    values: Vec<f32>,
    filled: usize,
}

/// How a [`KVCacheState`] is sized and where it may spill.
#[derive(Clone)]
pub struct KvConfig {
    pub dims: ModelDims,
    /// Hard cap on sequence length (the context window).
    pub max_positions: usize,
    /// Blocks kept in RAM before spilling begins.
    pub max_resident_blocks: usize,
    pub sliding_window: Option<usize>,
    pub storage: Option<Arc<dyn StorageBackend>>,
}

/// Attention state for one sequence.
///
/// Positions are grouped into fixed-size blocks. Once more than
/// `max_resident_blocks` are in RAM and a storage backend is configured, the
/// oldest complete block is written to disk and dropped; reading it back is a
/// real file read, not a bookkeeping entry.
///
/// Length is hard-capped at `max_positions` (the context window), so the cache
/// cannot grow without bound no matter what a client sends.
pub struct KVCacheState {
    dims: ModelDims,
    resident: BTreeMap<usize, KvBlock>,
    spilled: BTreeMap<usize, PathBuf>,
    len: usize,
    max_positions: usize,
    max_resident_blocks: usize,
    sliding_window: Option<usize>,
    storage: Option<Arc<dyn StorageBackend>>,
    /// Namespaces spill files so two sequences cannot collide.
    seq_id: u64,
    spills: u64,
    reloads: u64,
}

impl Clone for KVCacheState {
    /// Duplicates resident state; spilled blocks are shared by path, which is safe
    /// because a spill file is written once and never mutated afterwards.
    fn clone(&self) -> Self {
        Self {
            dims: self.dims,
            resident: self.resident.clone(),
            spilled: self.spilled.clone(),
            len: self.len,
            max_positions: self.max_positions,
            max_resident_blocks: self.max_resident_blocks,
            sliding_window: self.sliding_window,
            storage: self.storage.clone(),
            seq_id: self.seq_id,
            spills: self.spills,
            reloads: self.reloads,
        }
    }
}

impl std::fmt::Debug for KVCacheState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KVCacheState")
            .field("len", &self.len)
            .field("resident_blocks", &self.resident.len())
            .field("spilled_blocks", &self.spilled.len())
            .field("spills", &self.spills)
            .field("reloads", &self.reloads)
            .finish()
    }
}

impl KVCacheState {
    pub fn new(cfg: KvConfig, seq_id: u64) -> Self {
        Self {
            dims: cfg.dims,
            resident: BTreeMap::new(),
            spilled: BTreeMap::new(),
            len: 0,
            max_positions: cfg.max_positions.max(1),
            max_resident_blocks: cfg.max_resident_blocks.max(1),
            sliding_window: cfg.sliding_window,
            storage: cfg.storage,
            seq_id,
            spills: 0,
            reloads: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn max_positions(&self) -> usize {
        self.max_positions
    }

    pub fn spill_count(&self) -> u64 {
        self.spills
    }

    pub fn reload_count(&self) -> u64 {
        self.reloads
    }

    /// The oldest position attention may attend to, given the sliding window.
    pub fn attention_start(&self) -> usize {
        match self.sliding_window {
            Some(w) => self.len.saturating_sub(w),
            None => 0,
        }
    }

    fn spill_path(&self, block: usize) -> PathBuf {
        PathBuf::from(format!("kv/seq_{}/block_{}.bin", self.seq_id, block))
    }

    /// Append the key/value vectors for the next position.
    pub fn append(&mut self, key: &[f32], value: &[f32]) -> Result<(), GarudaError> {
        let d = self.dims.d_model;
        if key.len() != d || value.len() != d {
            return Err(GarudaError::Cache(format!(
                "kv append expects a {d}-dim key and value, got {} and {}",
                key.len(),
                value.len()
            )));
        }
        if self.len >= self.max_positions {
            return Err(GarudaError::Cache(format!(
                "context window of {} positions is exhausted",
                self.max_positions
            )));
        }

        let idx = self.len / self.dims.block_size;
        let (bs, dm) = (self.dims.block_size, d);
        let block = self.resident.entry(idx).or_insert_with(|| KvBlock {
            keys: Vec::with_capacity(bs * dm),
            values: Vec::with_capacity(bs * dm),
            filled: 0,
        });
        block.keys.extend_from_slice(key);
        block.values.extend_from_slice(value);
        block.filled += 1;
        self.len += 1;

        self.enforce_residency(idx)?;
        Ok(())
    }

    /// Spill oldest complete blocks until the residency budget is met.
    ///
    /// `current` is never spilled: it is the block being written to.
    fn enforce_residency(&mut self, current: usize) -> Result<(), GarudaError> {
        let Some(storage) = self.storage.clone() else {
            // No spill target. Growth is still bounded by `max_positions`.
            return Ok(());
        };

        while self.resident.len() > self.max_resident_blocks {
            let Some(victim) = self.resident.keys().copied().find(|&k| k != current) else {
                break;
            };
            let block = self
                .resident
                .remove(&victim)
                .expect("key came from the map");

            let mut bytes = Vec::with_capacity((block.keys.len() + block.values.len()) * 4 + 8);
            bytes.extend_from_slice(&(block.filled as u64).to_le_bytes());
            for v in block.keys.iter().chain(block.values.iter()) {
                bytes.extend_from_slice(&v.to_le_bytes());
            }

            let path = self.spill_path(victim);
            if let Err(e) = storage.write(&path, &bytes) {
                // Spilling failed. Keep the block in RAM rather than lose attention
                // state and silently produce wrong output.
                self.resident.insert(victim, block);
                return Err(e);
            }
            self.spilled.insert(victim, path);
            self.spills += 1;
        }
        Ok(())
    }

    /// Read `block` back from disk into RAM.
    fn reload(&mut self, block: usize) -> Result<(), GarudaError> {
        let Some(path) = self.spilled.get(&block).cloned() else {
            return Err(GarudaError::Cache(format!(
                "block {block} was never spilled"
            )));
        };
        let storage = self
            .storage
            .clone()
            .ok_or_else(|| GarudaError::Cache("no storage backend to reload from".into()))?;

        let bytes = storage.read(&path)?;
        if bytes.len() < 8 {
            return Err(GarudaError::Cache(format!(
                "spill file for block {block} is truncated"
            )));
        }
        let filled = u64::from_le_bytes(bytes[..8].try_into().expect("checked length")) as usize;
        let d = self.dims.d_model;
        let expected = 8 + filled * d * 2 * 4;
        if bytes.len() != expected || filled > self.dims.block_size {
            return Err(GarudaError::Cache(format!(
                "spill file for block {block} is malformed: {} bytes, filled={filled}",
                bytes.len()
            )));
        }

        let vals: Vec<f32> = bytes[8..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let (keys, values) = vals.split_at(filled * d);

        self.resident.insert(
            block,
            KvBlock {
                keys: keys.to_vec(),
                values: values.to_vec(),
                filled,
            },
        );
        self.spilled.remove(&block);
        self.reloads += 1;
        Ok(())
    }

    /// Make every block covering `start..end` resident. Call before reading.
    pub fn ensure_resident(&mut self, start: usize, end: usize) -> Result<(), GarudaError> {
        if start >= end {
            return Ok(());
        }
        let bs = self.dims.block_size;
        let needed: Vec<usize> = (start / bs..=(end - 1) / bs)
            .filter(|b| self.spilled.contains_key(b))
            .collect();
        for b in needed {
            self.reload(b)?;
        }
        Ok(())
    }

    /// Key vector at `pos`, if resident. `None` means spilled or out of range —
    /// call [`KVCacheState::ensure_resident`] first.
    pub fn key_at(&self, pos: usize) -> Option<&[f32]> {
        self.slice_at(pos, true)
    }

    pub fn value_at(&self, pos: usize) -> Option<&[f32]> {
        self.slice_at(pos, false)
    }

    fn slice_at(&self, pos: usize, key: bool) -> Option<&[f32]> {
        if pos >= self.len {
            return None;
        }
        let bs = self.dims.block_size;
        let d = self.dims.d_model;
        let block = self.resident.get(&(pos / bs))?;
        let off = (pos % bs) * d;
        let src = if key { &block.keys } else { &block.values };
        src.get(off..off + d)
    }

    /// True when some of this sequence's attention state currently lives on disk.
    pub fn has_spill(&self) -> bool {
        !self.spilled.is_empty()
    }

    /// Give this state a new sequence identity, so its future spill files cannot
    /// collide with the state it was cloned from.
    ///
    /// Rejected if anything is already spilled: those files are named for the old
    /// id, and renaming them here would race with the original owner.
    pub fn rekey(&mut self, seq_id: u64) -> Result<(), GarudaError> {
        if self.has_spill() {
            return Err(GarudaError::Cache(
                "cannot rekey a sequence with spilled blocks".into(),
            ));
        }
        self.seq_id = seq_id;
        Ok(())
    }

    /// Delete this sequence's spill files.
    pub fn purge_spill_files(&mut self) {
        if let Some(storage) = &self.storage {
            for path in self.spilled.values() {
                let _ = storage.remove(path);
            }
        }
        self.spilled.clear();
    }
}

// ---------------------------------------------------------------------------
// Sequence state
// ---------------------------------------------------------------------------

/// Everything one in-flight sequence carries between decode steps: its attention
/// cache, plus the routing history the prefetcher learns from.
#[derive(Debug, Clone)]
pub struct SeqState {
    pub kv: KVCacheState,
    /// Experts that fired on the previous step.
    pub last_experts: Vec<ExpertId>,
    /// What the predictor guessed the current step would need.
    pub last_predicted: Vec<ExpertId>,
}

impl SeqState {
    pub fn new(cfg: KvConfig, seq_id: u64) -> Self {
        Self {
            kv: KVCacheState::new(cfg, seq_id),
            last_experts: Vec::new(),
            last_predicted: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.kv.len()
    }

    pub fn is_empty(&self) -> bool {
        self.kv.is_empty()
    }
}

impl Drop for SeqState {
    fn drop(&mut self) {
        // A sequence that ends — completed, cancelled, timed out, or dropped because
        // the client hung up — takes its spill files with it.
        self.kv.purge_spill_files();
    }
}

// ---------------------------------------------------------------------------
// Prompt prefix cache
// ---------------------------------------------------------------------------

/// Maps an exact prompt to the sequence state produced by prefilling it, so that
/// re-sending the same prompt skips prefill entirely.
///
/// Bounded by entry count, LRU eviction. (The previous implementation was keyed by
/// the full token vector, never evicted, and threw the cached value away on read —
/// it could only grow.)
///
/// Only states with nothing spilled are cached. A cached entry is handed out by
/// clone, and a clone must be able to spill under a fresh identity; sharing an id
/// with the entry it came from would let two sequences write the same files.
pub struct PromptCache {
    inner: Mutex<PromptCacheInner>,
    capacity: usize,
}

struct PromptCacheInner {
    entries: HashMap<[u8; 32], SeqState>,
    lru: VecDeque<[u8; 32]>,
    hits: u64,
    misses: u64,
}

fn prompt_key(tokens: &[Token]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    for t in tokens {
        hasher.update(&t.to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

impl PromptCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(PromptCacheInner {
                entries: HashMap::new(),
                lru: VecDeque::new(),
                hits: 0,
                misses: 0,
            }),
            capacity: capacity.max(1),
        }
    }

    /// A ready-to-continue state for `tokens`, if this exact prefix has been seen.
    ///
    /// `fresh_seq_id` must be unique across live sequences: the returned state is a
    /// clone, and it needs its own identity before it can spill.
    pub fn get(&self, tokens: &[Token], fresh_seq_id: u64) -> Option<SeqState> {
        let key = prompt_key(tokens);
        let mut inner = self.inner.lock();

        let Some(entry) = inner.entries.get(&key) else {
            inner.misses += 1;
            return None;
        };

        let mut state = entry.clone();
        inner.hits += 1;
        inner.lru.retain(|k| k != &key);
        inner.lru.push_back(key);
        drop(inner);

        state
            .kv
            .rekey(fresh_seq_id)
            .expect("cached states never hold spilled blocks");
        Some(state)
    }

    /// Cache the state for `tokens`.
    ///
    /// States holding spilled blocks are refused: a cache entry gets handed out by
    /// clone, and a clone that shared spill-file paths with a live sequence would
    /// delete that sequence's attention state when it was evicted.
    pub fn insert(&self, tokens: &[Token], state: SeqState) {
        if state.kv.has_spill() {
            return;
        }
        let key = prompt_key(tokens);
        let mut inner = self.inner.lock();

        inner.entries.insert(key, state);
        inner.lru.retain(|k| k != &key);
        inner.lru.push_back(key);

        while inner.lru.len() > self.capacity {
            let Some(victim) = inner.lru.pop_front() else {
                break;
            };
            inner.entries.remove(&victim); // Drop purges anything it owns.
        }
    }

    pub fn stats(&self) -> CacheStats {
        let inner = self.inner.lock();
        CacheStats {
            hits: inner.hits,
            misses: inner.misses,
            evictions: 0,
            entries: inner.entries.len(),
            bytes: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::LocalStorageBackend;
    use crate::weights::synthesize_expert;

    fn dims() -> ModelDims {
        ModelDims {
            block_size: 4,
            ..Default::default()
        }
    }

    fn kv_cfg(storage: Option<Arc<dyn StorageBackend>>, max_resident: usize) -> KvConfig {
        KvConfig {
            dims: dims(),
            max_positions: 64,
            max_resident_blocks: max_resident,
            sliding_window: None,
            storage,
        }
    }

    fn walk(dir: &std::path::Path) -> Vec<PathBuf> {
        let mut out = Vec::new();
        if let Ok(rd) = std::fs::read_dir(dir) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    out.extend(walk(&p));
                } else {
                    out.push(p);
                }
            }
        }
        out
    }

    #[test]
    fn expert_cache_evicts_by_byte_budget() {
        let d = ModelDims::default();
        let one = synthesize_expert(0, &d);
        let size = one.size_bytes();

        // Budget for exactly two experts.
        let cache = ExpertCache::new(size * 2);
        cache.insert(0, Arc::new(one));
        cache.insert(1, Arc::new(synthesize_expert(1, &d)));
        assert!(cache.contains(0) && cache.contains(1));

        // Touch 0 so that 1 becomes the least-recently-used, then overflow.
        assert!(cache.get(0).is_some());
        let evicted = cache.insert(2, Arc::new(synthesize_expert(2, &d)));

        assert_eq!(evicted, vec![1], "LRU victim should be expert 1");
        assert!(cache.contains(0) && cache.contains(2));
        assert!(!cache.contains(1));
        assert!(cache.stats().bytes <= size * 2);
    }

    #[test]
    fn kv_append_is_capped_by_the_context_window() {
        let d = dims();
        let mut kv = KVCacheState::new(
            KvConfig {
                max_positions: 3,
                ..kv_cfg(None, 8)
            },
            1,
        );
        let v = vec![0.5; d.d_model];
        for _ in 0..3 {
            kv.append(&v, &v).unwrap();
        }
        let err = kv.append(&v, &v).unwrap_err();
        assert!(matches!(err, GarudaError::Cache(_)), "got {err:?}");
        assert_eq!(kv.len(), 3);
    }

    #[test]
    fn kv_append_rejects_wrong_dimension() {
        let mut kv = KVCacheState::new(kv_cfg(None, 8), 1);
        assert!(kv.append(&[1.0, 2.0], &[1.0, 2.0]).is_err());
        assert_eq!(kv.len(), 0);
    }

    #[test]
    fn kv_spills_to_disk_and_reads_back_identical_values() {
        let dir = std::env::temp_dir().join("garuda_kv_spill_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorageBackend::new(&dir));

        let d = dims();
        // block_size is 4 and only one block may stay resident, so spilling starts early.
        let mut kv = KVCacheState::new(kv_cfg(Some(storage), 1), 42);

        let mut expected = Vec::new();
        for p in 0..12 {
            let k: Vec<f32> = (0..d.d_model).map(|i| p as f32 + i as f32 * 0.01).collect();
            let v: Vec<f32> = k.iter().map(|x| -x).collect();
            kv.append(&k, &v).unwrap();
            expected.push((k, v));
        }

        assert!(kv.spill_count() > 0, "nothing was spilled");
        assert!(
            kv.key_at(0).is_none(),
            "position 0 should be on disk, not in RAM"
        );
        assert!(!walk(&dir).is_empty(), "spill wrote no bytes to disk");

        kv.ensure_resident(0, 12).unwrap();
        assert!(kv.reload_count() > 0);
        for (p, (k, v)) in expected.iter().enumerate() {
            assert_eq!(kv.key_at(p).unwrap(), &k[..], "key at {p}");
            assert_eq!(kv.value_at(p).unwrap(), &v[..], "value at {p}");
        }

        kv.purge_spill_files();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn prompt_cache_hits_and_stays_bounded() {
        let cache = PromptCache::new(2);
        let state = |id| SeqState::new(kv_cfg(None, 4), id);

        cache.insert(&[1, 2, 3], state(1));
        assert!(
            cache.get(&[1, 2, 3], 10).is_some(),
            "exact prompt should hit"
        );
        assert!(cache.get(&[1, 2, 4], 11).is_none());

        cache.insert(&[4], state(2));
        cache.insert(&[5], state(3)); // evicts the least-recently-used entry

        assert_eq!(cache.stats().entries, 2, "capacity must hold");
        assert!(
            cache.get(&[1, 2, 3], 12).is_none(),
            "oldest should have been evicted"
        );
        assert!(cache.get(&[5], 13).is_some());
    }

    #[test]
    fn prompt_cache_hands_out_distinct_sequence_ids() {
        // Two requests hitting the same cached prompt must not share spill paths.
        let dir = std::env::temp_dir().join("garuda_prompt_rekey");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorageBackend::new(&dir));

        let cache = PromptCache::new(4);
        cache.insert(&[7, 8], SeqState::new(kv_cfg(Some(storage.clone()), 1), 1));

        let mut a = cache.get(&[7, 8], 100).unwrap();
        let mut b = cache.get(&[7, 8], 200).unwrap();

        let d = dims();
        let v = vec![0.25; d.d_model];
        // Force both to spill, then confirm they wrote to different files.
        for _ in 0..12 {
            a.kv.append(&v, &v).unwrap();
            b.kv.append(&v, &v).unwrap();
        }
        assert!(a.kv.has_spill() && b.kv.has_spill());

        // Dropping `a` purges only `a`'s files; `b` must still read back.
        drop(a);
        b.kv.ensure_resident(0, b.kv.len()).unwrap();
        assert_eq!(b.kv.key_at(0).unwrap(), &v[..]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn prompt_cache_refuses_to_store_spilled_state() {
        let dir = std::env::temp_dir().join("garuda_prompt_nospill");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorageBackend::new(&dir));

        let d = dims();
        let mut state = SeqState::new(
            KvConfig {
                max_resident_blocks: 1,
                ..kv_cfg(Some(storage), 8)
            },
            9,
        );
        let v = vec![0.5; d.d_model];
        for _ in 0..12 {
            state.kv.append(&v, &v).unwrap();
        }
        assert!(state.kv.has_spill());

        let cache = PromptCache::new(4);
        cache.insert(&[1], state);
        assert_eq!(cache.stats().entries, 0, "spilled state must not be cached");

        let _ = std::fs::remove_dir_all(&dir);
    }
}

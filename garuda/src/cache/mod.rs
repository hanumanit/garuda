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

/// How a sequence's KV state is sized and where it may spill.
#[derive(Clone)]
pub struct KvConfig {
    pub dims: ModelDims,
    /// Width of one stored key/value vector. Equals `d_model` for full multi-head
    /// attention; for grouped-query attention it is `n_kv_heads * head_dim`, which
    /// is narrower.
    pub kv_dim: usize,
    /// Attention layers, each of which gets its own cache. `1` for the single-block
    /// MoE engine; a real model has one per transformer block.
    pub n_layers: usize,
    /// Per-layer key/value width, for a model whose layers are not all the same.
    /// `None` gives every layer [`Self::kv_dim`].
    ///
    /// A hybrid model — Qwen3.5 alternates three gated-delta-net layers with one
    /// attention layer — stores no keys or values at all in its recurrent layers,
    /// which carry a fixed-size state instead (see [`LinearState`]). Those layers get
    /// a width of `0`: they still count positions, so every layer advances together
    /// as the backend contract requires, but they hold nothing and cost nothing.
    pub kv_dims: Option<Vec<usize>>,
    /// Hard cap on sequence length (the context window).
    pub max_positions: usize,
    /// Blocks kept in RAM before spilling begins.
    pub max_resident_blocks: usize,
    pub sliding_window: Option<usize>,
    pub storage: Option<Arc<dyn StorageBackend>>,
}

impl KvConfig {
    /// Full multi-head attention: one layer, key/value width equal to `d_model`.
    pub fn mha(
        dims: ModelDims,
        max_positions: usize,
        max_resident_blocks: usize,
        sliding_window: Option<usize>,
        storage: Option<Arc<dyn StorageBackend>>,
    ) -> Self {
        Self {
            kv_dim: dims.d_model,
            n_layers: 1,
            kv_dims: None,
            dims,
            max_positions,
            max_resident_blocks,
            sliding_window,
            storage,
        }
    }
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
///
/// # Spilling pairs with a sliding window
///
/// Spilling only pays off when something bounds how far back attention reads.
/// Full attention reads every earlier position, so each step calls
/// [`Self::ensure_resident`] over the whole prefix, reloading everything that was
/// spilled — which the next [`Self::append`] spills straight back out. The result
/// is correct but quadratic in disk I/O. With `sliding_window` set, only the window
/// has to be resident and the spilled tail stays on disk where it belongs.
/// [`crate::server::Engine`] warns at startup when a configuration would thrash.
pub struct KVCacheState {
    kv_dim: usize,
    block_size: usize,
    resident: BTreeMap<usize, KvBlock>,
    spilled: BTreeMap<usize, PathBuf>,
    len: usize,
    max_positions: usize,
    max_resident_blocks: usize,
    sliding_window: Option<usize>,
    storage: Option<Arc<dyn StorageBackend>>,
    /// Namespaces spill files so two sequences (and two layers) cannot collide.
    seq_id: u64,
    layer: usize,
    spills: u64,
    reloads: u64,
    /// MoE prefetch bookkeeping for this layer: experts that fired here last step,
    /// and what the predictor guessed this step would need. Each transformer layer
    /// routes independently, so this lives per-layer rather than on `SeqState`.
    pub last_experts: Vec<ExpertId>,
    pub last_predicted: Vec<ExpertId>,
}

impl Clone for KVCacheState {
    /// Duplicates resident state; spilled blocks are shared by path, which is safe
    /// because a spill file is written once and never mutated afterwards.
    fn clone(&self) -> Self {
        Self {
            kv_dim: self.kv_dim,
            block_size: self.block_size,
            resident: self.resident.clone(),
            spilled: self.spilled.clone(),
            len: self.len,
            max_positions: self.max_positions,
            max_resident_blocks: self.max_resident_blocks,
            sliding_window: self.sliding_window,
            storage: self.storage.clone(),
            seq_id: self.seq_id,
            layer: self.layer,
            spills: self.spills,
            reloads: self.reloads,
            last_experts: self.last_experts.clone(),
            last_predicted: self.last_predicted.clone(),
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
        Self::for_layer(&cfg, seq_id, 0)
    }

    fn for_layer(cfg: &KvConfig, seq_id: u64, layer: usize) -> Self {
        // A per-layer width of 0 is deliberate and means "count positions, store
        // nothing": see `KvConfig::kv_dims`. The uniform width keeps its old floor of
        // 1, so a caller that leaves `kv_dim` at zero by accident still gets a cache
        // that holds something.
        let kv_dim = match &cfg.kv_dims {
            Some(dims) => dims.get(layer).copied().unwrap_or(cfg.kv_dim),
            None => cfg.kv_dim.max(1),
        };
        Self {
            kv_dim,
            block_size: cfg.dims.block_size.max(1),
            resident: BTreeMap::new(),
            spilled: BTreeMap::new(),
            len: 0,
            max_positions: cfg.max_positions.max(1),
            max_resident_blocks: cfg.max_resident_blocks.max(1),
            sliding_window: cfg.sliding_window,
            storage: cfg.storage.clone(),
            seq_id,
            layer,
            spills: 0,
            reloads: 0,
            last_experts: Vec::new(),
            last_predicted: Vec::new(),
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
        PathBuf::from(format!(
            "kv/seq_{}/layer_{}/block_{}.bin",
            self.seq_id, self.layer, block
        ))
    }

    /// Append the key/value vectors for the next position.
    pub fn append(&mut self, key: &[f32], value: &[f32]) -> Result<(), GarudaError> {
        let d = self.kv_dim;
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

        let idx = self.len / self.block_size;
        let (bs, dm) = (self.block_size, d);
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
        if self.kv_dim == 0 {
            // A position-only layer (see `KvConfig::kv_dims`) holds no vectors, so its
            // blocks occupy nothing and spilling them would write empty files and read
            // them back for no reason.
            return Ok(());
        }
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
        let d = self.kv_dim;
        let expected = 8 + filled * d * 2 * 4;
        if bytes.len() != expected || filled > self.block_size {
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
        let bs = self.block_size;
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
        let bs = self.block_size;
        let d = self.kv_dim;
        let block = self.resident.get(&(pos / bs))?;
        let off = (pos % bs) * d;
        let src = if key { &block.keys } else { &block.values };
        src.get(off..off + d)
    }

    /// Drop every position at or after `len`.
    ///
    /// Speculative decoding needs this: it appends several guessed tokens, checks
    /// which the model would actually have produced, and has to undo the rest — the
    /// cache must end up exactly as if the rejected tokens had never been seen.
    ///
    /// Refuses to cut into a spilled block rather than silently reloading one: a
    /// truncation always lands on the newest positions, which are resident by
    /// construction, so needing to is a sign the caller is doing something else.
    pub fn truncate(&mut self, len: usize) -> Result<(), GarudaError> {
        if len >= self.len {
            return Ok(());
        }
        let bs = self.block_size;
        let (cut_block, keep) = (len / bs, len % bs);

        if keep > 0 && self.spilled.contains_key(&cut_block) {
            return Err(GarudaError::Cache(format!(
                "cannot truncate to {len}: block {cut_block} is spilled to disk"
            )));
        }

        // Whole blocks past the cut go, along with any files they left behind.
        self.resident
            .retain(|&b, _| b < cut_block || (b == cut_block && keep > 0));
        let stale: Vec<usize> = self
            .spilled
            .keys()
            .copied()
            .filter(|&b| b > cut_block || (b == cut_block && keep == 0))
            .collect();
        for b in stale {
            if let Some(path) = self.spilled.remove(&b) {
                if let Some(storage) = &self.storage {
                    let _ = storage.remove(&path);
                }
            }
        }

        // The block the cut lands inside keeps only what precedes it.
        if keep > 0 {
            if let Some(block) = self.resident.get_mut(&cut_block) {
                block.keys.truncate(keep * self.kv_dim);
                block.values.truncate(keep * self.kv_dim);
                block.filled = keep;
            }
        }

        self.len = len;
        Ok(())
    }

    /// Bytes this layer's attention state occupies in RAM. Spilled blocks are on
    /// disk and cost nothing here.
    pub fn resident_bytes(&self) -> usize {
        self.resident
            .values()
            .map(|b| (b.keys.len() + b.values.len()) * std::mem::size_of::<f32>())
            .sum()
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

/// Everything one in-flight sequence carries between decode steps: one attention
/// cache per transformer layer. Each layer's [`KVCacheState`] also carries that
/// layer's own MoE routing history, since a real model's layers route independently.
///
/// A single-block engine (the MoE) has one layer; a real model has one per block.
/// All layers advance together — each token appends exactly one position to every
/// layer — so `len()` reads the first layer and speaks for all of them.
#[derive(Debug, Clone)]
pub struct SeqState {
    kvs: Vec<KVCacheState>,
    /// Recurrent state for the layers that have it, indexed like `kvs`. `None` for an
    /// ordinary attention layer, which keeps its history in `kvs` instead.
    linear: Vec<Option<LinearState>>,
    /// Set while a round of speculated tokens is being verified: the backend then
    /// copies each recurrent layer's state before folding a token into it.
    recording: bool,
}

/// What a linear-attention layer carries between tokens, instead of a growing cache
/// of keys and values.
///
/// A gated delta net summarises everything it has read into two fixed-size buffers,
/// so this costs the same whether the sequence is ten tokens long or a hundred
/// thousand — which is the point of the architecture. It is also why a linear layer
/// cannot be rewound by arithmetic: the state is a summary, not a log. The way back is
/// a copy taken on the way in — see [`SeqState::begin_recording`] and
/// [`SeqState::truncate`].
#[derive(Debug, Clone, PartialEq)]
pub struct LinearState {
    /// The `kernel - 1` most recent inputs to the depthwise causal convolution,
    /// oldest first, `conv_dim` values each.
    pub conv: Vec<f32>,
    /// The recurrent matrix state: one `key_head_dim * value_head_dim` matrix per
    /// value head.
    pub state: Vec<f32>,
    /// Tokens folded into `state` so far.
    folded: usize,
    /// Copies of `(folded, conv, state)` taken before folding a token, oldest first.
    ///
    /// Only recorded while a sequence is verifying speculated tokens — see
    /// [`SeqState::begin_recording`]. The state cannot be rewound by arithmetic, so
    /// the only way back is a copy taken on the way in, and a copy is the size of the
    /// state: 149 MB per position on Qwen3.8-27B. That is why they are kept for
    /// exactly as long as a round of guesses and dropped the moment it is settled.
    history: Vec<(usize, Vec<f32>, Vec<f32>)>,
}

impl LinearState {
    fn zeros(conv_len: usize, state_len: usize) -> Self {
        Self {
            conv: vec![0.0; conv_len],
            state: vec![0.0; state_len],
            folded: 0,
            history: Vec::new(),
        }
    }

    fn bytes(&self) -> usize {
        let live = self.conv.len() + self.state.len();
        let kept: usize = self.history.iter().map(|(_, c, s)| c.len() + s.len()).sum();
        (live + kept) * std::mem::size_of::<f32>()
    }

    /// Copy the state as it stands, before the next token is folded into it.
    ///
    /// Called by the backend once per token while a speculative round is being
    /// verified. Cheap relative to what it protects — a copy is a memcpy against a
    /// forward pass that reads the whole model.
    pub fn snapshot(&mut self) {
        self.history
            .push((self.folded, self.conv.clone(), self.state.clone()));
    }

    /// Account for a token having been folded in.
    pub fn advance(&mut self) {
        self.folded += 1;
    }

    /// Put the state back to what it was after `len` tokens, if a copy from then is
    /// still held.
    fn rewind_to(&mut self, len: usize) -> bool {
        if len == self.folded {
            self.history.clear();
            return true;
        }
        let Some(at) = self.history.iter().position(|(n, _, _)| *n == len) else {
            return false;
        };
        let (n, conv, state) = self.history.swap_remove(at);
        self.conv = conv;
        self.state = state;
        self.folded = n;
        self.history.clear();
        true
    }
}

impl SeqState {
    pub fn new(cfg: KvConfig, seq_id: u64) -> Self {
        let n = cfg.n_layers.max(1);
        Self {
            kvs: (0..n)
                .map(|l| KVCacheState::for_layer(&cfg, seq_id, l))
                .collect(),
            linear: vec![None; n],
            recording: false,
        }
    }

    /// This layer's recurrent state, allocated zeroed on first use.
    ///
    /// The backend owns the shapes, so it passes them in rather than the cache
    /// guessing: `conv_len` is `conv_dim * (kernel - 1)` and `state_len` is
    /// `n_heads * key_head_dim * value_head_dim`. Asking for a different shape than
    /// the state already has is a backend bug and an error, not a silent reallocation
    /// that would throw away everything the sequence has read.
    pub fn linear(
        &mut self,
        l: usize,
        conv_len: usize,
        state_len: usize,
    ) -> Result<&mut LinearState, GarudaError> {
        if l >= self.linear.len() {
            return Err(GarudaError::Cache(format!(
                "layer {l} is out of range for a {}-layer sequence",
                self.linear.len()
            )));
        }
        match &self.linear[l] {
            Some(s) if s.conv.len() != conv_len || s.state.len() != state_len => {
                return Err(GarudaError::Cache(format!(
                    "layer {l} holds a {}/{} recurrent state but {conv_len}/{state_len} was asked for",
                    s.conv.len(),
                    s.state.len()
                )));
            }
            Some(_) => {}
            None => self.linear[l] = Some(LinearState::zeros(conv_len, state_len)),
        }
        Ok(self.linear[l].as_mut().expect("just populated"))
    }

    /// True when any layer of this sequence carries recurrent state.
    pub fn has_linear_state(&self) -> bool {
        self.linear.iter().any(Option::is_some)
    }

    /// Start keeping a copy of each recurrent layer's state per token consumed, so
    /// that this sequence can be put back to any position in the round about to run.
    ///
    /// This is what makes speculative decoding possible on a recurrent architecture:
    /// guesses are appended, checked in one pass, and the rejected ones have to leave
    /// no trace. The copies are the price — one state per position, and a state does
    /// not shrink with the sequence — so they are recorded only for the round and
    /// dropped by [`Self::truncate`] as soon as it is settled.
    pub fn begin_recording(&mut self) {
        for s in self.linear.iter_mut().flatten() {
            s.history.clear();
        }
        self.recording = true;
    }

    pub fn end_recording(&mut self) {
        self.recording = false;
    }

    /// Whether the backend should snapshot each recurrent layer as it consumes tokens.
    pub fn recording(&self) -> bool {
        self.recording
    }

    pub fn n_layers(&self) -> usize {
        self.kvs.len()
    }

    /// The cache for layer `l`.
    ///
    /// # Panics
    /// If `l` is out of range. The backend owns both the layer count and the cache,
    /// so an out-of-range layer is a backend bug, not runtime input.
    pub fn layer(&mut self, l: usize) -> &mut KVCacheState {
        &mut self.kvs[l]
    }

    /// The first (and, for the MoE engine, only) layer's cache.
    pub fn kv(&mut self) -> &mut KVCacheState {
        &mut self.kvs[0]
    }

    pub fn len(&self) -> usize {
        self.kvs[0].len()
    }

    /// The context window every layer of this sequence is capped at. Immutable, so a
    /// caller can check whether work will fit without taking a mutable borrow of the
    /// cache it is about to check on behalf of.
    pub fn max_positions(&self) -> usize {
        self.kvs[0].max_positions()
    }

    pub fn is_empty(&self) -> bool {
        self.kvs[0].is_empty()
    }

    pub fn has_spill(&self) -> bool {
        self.kvs.iter().any(KVCacheState::has_spill)
    }

    /// Bytes this sequence's attention state occupies in RAM, across every layer.
    ///
    /// Recurrent state counts: it is the larger half for a hybrid model — a Qwen3.5
    /// 27B sequence carries ~144 MB of it regardless of length — and the prompt cache
    /// budgets by this number.
    pub fn resident_bytes(&self) -> usize {
        let kv: usize = self.kvs.iter().map(KVCacheState::resident_bytes).sum();
        let linear: usize = self
            .linear
            .iter()
            .filter_map(|s| s.as_ref().map(LinearState::bytes))
            .sum();
        kv + linear
    }

    /// Give every layer a fresh sequence identity. Fails if anything is spilled.
    pub fn rekey(&mut self, seq_id: u64) -> Result<(), GarudaError> {
        for kv in &mut self.kvs {
            kv.rekey(seq_id)?;
        }
        Ok(())
    }

    pub fn purge_spill_files(&mut self) {
        for kv in &mut self.kvs {
            kv.purge_spill_files();
        }
    }

    /// Drop every position at or after `len`, in every layer. See
    /// [`KVCacheState::truncate`].
    /// Drop every position at or after `len`, across every layer.
    ///
    /// Refused outright for a sequence carrying recurrent state. A gated delta net
    /// folds each token into a fixed-size summary, and there is no arithmetic that
    /// takes the last few tokens back out of it — the only honest rewind is to replay
    /// the sequence from a saved state. Returning an error here is what keeps
    /// speculative decoding (the one caller that rewinds) from silently continuing
    /// with a state that describes tokens the caller has thrown away; a hybrid
    /// backend says so up front by answering `false` to
    /// [`InferenceBackend::speculation_supported`](crate::core::InferenceBackend::speculation_supported).
    pub fn truncate(&mut self, len: usize) -> Result<(), GarudaError> {
        // A recurrent layer is put back from a copy taken on the way in, or not at
        // all. Checked across every layer before anything is changed, so a refusal
        // leaves the sequence exactly as it was rather than half rewound.
        if len < self.len() && self.has_linear_state() {
            let recoverable = self
                .linear
                .iter()
                .flatten()
                .all(|s| s.folded == len || s.history.iter().any(|(n, _, _)| *n == len));
            if !recoverable {
                return Err(GarudaError::Cache(format!(
                    "cannot truncate to {len}: this sequence carries recurrent state, which \
                     summarises every token it has read, and no copy of it from that \
                     position is held"
                )));
            }
        }
        for s in self.linear.iter_mut().flatten() {
            s.rewind_to(len);
        }
        for kv in &mut self.kvs {
            kv.truncate(len)?;
        }
        Ok(())
    }
}

impl Drop for SeqState {
    fn drop(&mut self) {
        // A sequence that ends — completed, cancelled, timed out, or dropped because
        // the client hung up — takes its spill files with it.
        self.purge_spill_files();
    }
}

// ---------------------------------------------------------------------------
// Prompt prefix cache
// ---------------------------------------------------------------------------

/// Maps an exact prompt to the sequence state produced by prefilling it, so that
/// re-sending the same prompt skips prefill entirely.
///
/// Bounded by **bytes as well as entries**, LRU eviction. Entries alone is not a
/// bound at all here: one entry holds a whole sequence's attention state, and what
/// that costs depends on the model and the prompt. A cached 2048-token prefix is
/// 0.1 MB on the synthetic engine and 512 MB on Mixtral-8x7B, so the sixty-four
/// entries that are nothing in the first case are 32 GB in the second — on a machine
/// whose whole point is running a checkpoint bigger than its RAM. The expert cache
/// has been byte-budgeted since it was written; this is the same lesson, arrived at
/// later.
///
/// Only states with nothing spilled are cached. A cached entry is handed out by
/// clone, and a clone must be able to spill under a fresh identity; sharing an id
/// with the entry it came from would let two sequences write the same files.
pub struct PromptCache {
    inner: Mutex<PromptCacheInner>,
    capacity: usize,
    budget_bytes: usize,
}

struct PromptCacheInner {
    entries: HashMap<[u8; 32], SeqState>,
    lru: VecDeque<[u8; 32]>,
    bytes: usize,
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
    /// `budget_bytes` caps the RAM the cached states may hold; `capacity` caps how
    /// many there may be. Whichever binds first does the evicting.
    pub fn new(capacity: usize, budget_bytes: usize) -> Self {
        Self {
            inner: Mutex::new(PromptCacheInner {
                entries: HashMap::new(),
                lru: VecDeque::new(),
                bytes: 0,
                hits: 0,
                misses: 0,
            }),
            capacity: capacity.max(1),
            budget_bytes,
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
        if state.has_spill() {
            return;
        }
        let size = state.resident_bytes();
        // One entry larger than the whole budget would evict everything and then sit
        // there alone, so decline it instead: the prefill it saves is not worth
        // emptying the cache for every other caller.
        if size > self.budget_bytes {
            return;
        }

        let key = prompt_key(tokens);
        let mut inner = self.inner.lock();

        if let Some(old) = inner.entries.remove(&key) {
            inner.bytes -= old.resident_bytes();
            inner.lru.retain(|k| k != &key);
        }
        inner.entries.insert(key, state);
        inner.bytes += size;
        inner.lru.push_back(key);

        while inner.lru.len() > self.capacity || inner.bytes > self.budget_bytes {
            let Some(victim) = inner.lru.pop_front() else {
                break;
            };
            if let Some(evicted) = inner.entries.remove(&victim) {
                // Drop purges anything it owns.
                inner.bytes -= evicted.resident_bytes();
            }
        }
    }

    pub fn stats(&self) -> CacheStats {
        let inner = self.inner.lock();
        CacheStats {
            hits: inner.hits,
            misses: inner.misses,
            evictions: 0,
            entries: inner.entries.len(),
            bytes: inner.bytes,
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
        KvConfig::mha(dims(), 64, max_resident, None, storage)
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

    /// Truncating must leave the cache byte-for-byte as if the dropped positions had
    /// never been appended — that is what lets a rejected speculation be undone.
    #[test]
    fn truncate_rewinds_to_exactly_the_state_before_the_extra_appends() {
        let d = dims();
        let make = || KVCacheState::new(kv_cfg(None, 64), 1);
        let vec_at =
            |p: usize| -> Vec<f32> { (0..d.d_model).map(|i| p as f32 + i as f32 * 0.01).collect() };

        // Ten positions, then five more that we will take back.
        let mut rewound = make();
        for p in 0..15 {
            let v = vec_at(p);
            rewound.append(&v, &v).unwrap();
        }
        rewound.truncate(10).unwrap();

        // The same cache built with only the first ten.
        let mut reference = make();
        for p in 0..10 {
            let v = vec_at(p);
            reference.append(&v, &v).unwrap();
        }

        assert_eq!(rewound.len(), reference.len());
        for p in 0..10 {
            assert_eq!(rewound.key_at(p), reference.key_at(p), "key at {p}");
            assert_eq!(rewound.value_at(p), reference.value_at(p), "value at {p}");
        }
        assert!(rewound.key_at(10).is_none(), "position 10 survived the cut");

        // And appending after a rewind continues from the right place.
        let v = vec_at(99);
        rewound.append(&v, &v).unwrap();
        assert_eq!(rewound.len(), 11);
        assert_eq!(rewound.key_at(10).unwrap(), &v[..]);
    }

    #[test]
    fn truncate_handles_block_boundaries_and_no_ops() {
        let d = dims(); // block_size 4
        let mut kv = KVCacheState::new(kv_cfg(None, 64), 1);
        let v = vec![0.5; d.d_model];
        for _ in 0..12 {
            kv.append(&v, &v).unwrap();
        }

        // Exactly on a block boundary: the whole block goes.
        kv.truncate(8).unwrap();
        assert_eq!(kv.len(), 8);
        assert!(kv.key_at(8).is_none());

        // Asking to truncate to something longer is a no-op, not an error.
        kv.truncate(100).unwrap();
        assert_eq!(kv.len(), 8);

        // All the way back to empty.
        kv.truncate(0).unwrap();
        assert_eq!(kv.len(), 0);
        assert!(kv.is_empty());
        kv.append(&v, &v).unwrap();
        assert_eq!(kv.len(), 1);
    }

    #[test]
    fn truncate_refuses_to_cut_into_a_spilled_block() {
        let dir = std::env::temp_dir().join("garuda_truncate_spill");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorageBackend::new(&dir));

        let d = dims();
        let mut kv = KVCacheState::new(kv_cfg(Some(storage), 1), 7);
        let v = vec![0.25; d.d_model];
        for _ in 0..12 {
            kv.append(&v, &v).unwrap();
        }
        assert!(kv.has_spill(), "fixture did not spill");

        // Position 1 sits inside block 0, which is on disk. Refuse rather than
        // quietly reload it — a speculation rewind never reaches back this far.
        let err = kv.truncate(1).unwrap_err();
        assert!(matches!(err, GarudaError::Cache(_)), "got {err:?}");
        assert_eq!(kv.len(), 12, "a refused truncate must change nothing");

        kv.purge_spill_files();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn prompt_cache_hits_and_stays_bounded() {
        let cache = PromptCache::new(2, 64 << 20);
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

        let cache = PromptCache::new(4, 64 << 20);
        cache.insert(&[7, 8], SeqState::new(kv_cfg(Some(storage.clone()), 1), 1));

        let mut a = cache.get(&[7, 8], 100).unwrap();
        let mut b = cache.get(&[7, 8], 200).unwrap();

        let d = dims();
        let v = vec![0.25; d.d_model];
        // Force both to spill, then confirm they wrote to different files.
        for _ in 0..12 {
            a.kv().append(&v, &v).unwrap();
            b.kv().append(&v, &v).unwrap();
        }
        assert!(a.has_spill() && b.has_spill());

        // Dropping `a` purges only `a`'s files; `b` must still read back.
        drop(a);
        let n = b.len();
        b.kv().ensure_resident(0, n).unwrap();
        assert_eq!(b.kv().key_at(0).unwrap(), &v[..]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Entries alone do not bound this cache: one entry is a whole sequence's
    /// attention state. A budget in bytes has to evict even when the entry cap is
    /// nowhere near reached, or a large model fills memory with cached prefixes.
    #[test]
    fn prompt_cache_evicts_on_bytes_before_the_entry_cap_is_reached() {
        let d = dims();
        // A state holding four positions, so its size is easy to reason about.
        let state = |id: u64| {
            let mut s = SeqState::new(kv_cfg(None, 64), id);
            let v = vec![0.5; d.d_model];
            for _ in 0..4 {
                s.kv().append(&v, &v).unwrap();
            }
            s
        };
        let one = state(1).resident_bytes();
        assert!(one > 0, "a populated state should occupy something");

        // Room for two by bytes, but a hundred by entry count.
        let cache = PromptCache::new(100, one * 2);
        cache.insert(&[1], state(1));
        cache.insert(&[2], state(2));
        assert_eq!(cache.stats().entries, 2);
        assert_eq!(cache.stats().bytes, one * 2);

        cache.insert(&[3], state(3));
        assert_eq!(
            cache.stats().entries,
            2,
            "the byte budget did not evict, though the entry cap was far away"
        );
        assert!(
            cache.stats().bytes <= one * 2,
            "over budget after inserting"
        );
        assert!(cache.get(&[1], 10).is_none(), "the oldest should have gone");
        assert!(cache.get(&[3], 11).is_some(), "the newest should be here");
    }

    #[test]
    fn prompt_cache_declines_an_entry_larger_than_the_whole_budget() {
        let d = dims();
        let mut big = SeqState::new(kv_cfg(None, 64), 1);
        let v = vec![0.5; d.d_model];
        for _ in 0..8 {
            big.kv().append(&v, &v).unwrap();
        }
        // Budget smaller than this single entry: taking it would evict everything
        // else and then sit there alone.
        let cache = PromptCache::new(100, big.resident_bytes() / 2);
        cache.insert(&[9], big);
        assert_eq!(cache.stats().entries, 0);
        assert_eq!(cache.stats().bytes, 0);
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
            state.kv().append(&v, &v).unwrap();
        }
        assert!(state.has_spill());

        let cache = PromptCache::new(4, 64 << 20);
        cache.insert(&[1], state);
        assert_eq!(cache.stats().entries, 0, "spilled state must not be cached");

        let _ = std::fs::remove_dir_all(&dir);
    }
}

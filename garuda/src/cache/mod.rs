use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use crate::core::{ExpertId, Expert, Token, GarudaError, Tensor};
use parking_lot::RwLock;

pub struct KVCacheState {
    pub paged_blocks: HashMap<u32, Tensor>, // block_id -> tensor
    pub active_blocks: Vec<u32>,
    pub sliding_window: Option<usize>,
    pub max_blocks: usize,
    pub spilled_blocks: HashMap<u32, String>, // block_id -> spilled file path
}

impl KVCacheState {
    pub fn new(max_blocks: usize, sliding_window: Option<usize>) -> Self {
        Self {
            paged_blocks: HashMap::new(),
            active_blocks: Vec::new(),
            sliding_window,
            max_blocks,
            spilled_blocks: HashMap::new(),
        }
    }

    pub fn insert_block(&mut self, block_id: u32, block: Tensor) -> Result<(), GarudaError> {
        if self.paged_blocks.len() >= self.max_blocks {
            self.spill_block()?;
        }
        self.paged_blocks.insert(block_id, block);
        self.active_blocks.push(block_id);
        Ok(())
    }

    fn spill_block(&mut self) -> Result<(), GarudaError> {
        if let Some(evict_id) = self.active_blocks.first().copied() {
            if let Some(_tensor) = self.paged_blocks.remove(&evict_id) {
                // Spill to "disk" (simulated spill path)
                let spill_path = format!("kv_spill_block_{}.bin", evict_id);
                self.spilled_blocks.insert(evict_id, spill_path);
                self.active_blocks.retain(|&x| x != evict_id);
            }
        }
        Ok(())
    }

    pub fn load_spilled_block(&mut self, block_id: u32) -> Result<Tensor, GarudaError> {
        if let Some(_path) = self.spilled_blocks.remove(&block_id) {
            let tensor = Tensor::zeros(vec![1, 128]);
            self.paged_blocks.insert(block_id, tensor.clone());
            self.active_blocks.push(block_id);
            Ok(tensor)
        } else {
            Err(GarudaError::Cache("Block not found in spill".to_string()))
        }
    }
}

pub struct PromptCache {
    cache: RwLock<HashMap<Vec<Token>, KVCacheState>>,
}

impl PromptCache {
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
        }
    }

    pub fn get(&self, tokens: &[Token]) -> Option<KVCacheState> {
        let cache = self.cache.read();
        cache.get(tokens).map(|_| KVCacheState::new(100, None))
    }

    pub fn insert(&self, tokens: Vec<Token>, state: KVCacheState) {
        let mut cache = self.cache.write();
        cache.insert(tokens, state);
    }
}

pub struct ExpertCache {
    capacity: usize,
    loaded: RwLock<HashMap<ExpertId, Arc<Expert>>>,
    lru: RwLock<VecDeque<ExpertId>>,
}

impl ExpertCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            loaded: RwLock::new(HashMap::new()),
            lru: RwLock::new(VecDeque::new()),
        }
    }

    pub fn get(&self, id: ExpertId) -> Option<Arc<Expert>> {
        let loaded = self.loaded.read();
        if let Some(expert) = loaded.get(&id) {
            let mut lru = self.lru.write();
            lru.retain(|&x| x != id);
            lru.push_back(id);
            Some(expert.clone())
        } else {
            None
        }
    }

    pub fn insert(&self, id: ExpertId, expert: Arc<Expert>) -> Option<ExpertId> {
        let mut loaded = self.loaded.write();
        let mut lru = self.lru.write();

        let mut evicted = None;
        if loaded.len() >= self.capacity {
            if let Some(old_id) = lru.pop_front() {
                loaded.remove(&old_id);
                evicted = Some(old_id);
            }
        }

        loaded.insert(id, expert);
        lru.retain(|&x| x != id);
        lru.push_back(id);
        evicted
    }

    pub fn remove(&self, id: ExpertId) {
        let mut loaded = self.loaded.write();
        let mut lru = self.lru.write();
        loaded.remove(&id);
        lru.retain(|&x| x != id);
    }
}

pub struct EmbeddingCache {
    cache: RwLock<HashMap<Token, Tensor>>,
}

impl EmbeddingCache {
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
        }
    }

    pub fn get(&self, token: Token) -> Option<Tensor> {
        self.cache.read().get(&token).cloned()
    }

    pub fn insert(&self, token: Token, embedding: Tensor) {
        self.cache.write().insert(token, embedding);
    }
}

pub struct TokenizerCache {
    cache: RwLock<HashMap<String, Vec<Token>>>,
}

impl TokenizerCache {
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
        }
    }

    pub fn get(&self, text: &str) -> Option<Vec<Token>> {
        self.cache.read().get(text).cloned()
    }

    pub fn insert(&self, text: String, tokens: Vec<Token>) {
        self.cache.write().insert(text, tokens);
    }
}

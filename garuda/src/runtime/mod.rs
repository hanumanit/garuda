use crate::core::{Token, Tensor, GarudaError};
use crate::tokenizer::Tokenizer;
use crate::cache::{PromptCache, KVCacheState};
use crate::moe::MoeEngine;
use crate::prefetch::PrefetchEngine;
use std::sync::Arc;

pub struct InferenceRuntime {
    pub tokenizer: Tokenizer,
    pub prompt_cache: PromptCache,
    pub moe_engine: Arc<MoeEngine>,
    pub prefetch_engine: PrefetchEngine,
}

impl InferenceRuntime {
    pub fn new(tokenizer: Tokenizer, moe_engine: Arc<MoeEngine>, prefetch_engine: PrefetchEngine) -> Self {
        Self {
            tokenizer,
            prompt_cache: PromptCache::new(),
            moe_engine,
            prefetch_engine,
        }
    }

    pub fn forward(&self, tokens: &[Token]) -> Result<Tensor, GarudaError> {
        let mut _kv_state = if let Some(cached_kv) = self.prompt_cache.get(tokens) {
            cached_kv
        } else {
            KVCacheState::new(100, None)
        };

        let output = self.moe_engine.forward(tokens)?;

        self.prompt_cache.insert(tokens.to_vec(), _kv_state);

        Ok(output)
    }

    pub fn sample(&self, logits: &Tensor) -> Token {
        if logits.data.is_empty() {
            return 0;
        }
        let mut max_idx = 0;
        let mut max_val = logits.data[0];
        for (i, &val) in logits.data.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        (max_idx as Token) % 1000
    }
}

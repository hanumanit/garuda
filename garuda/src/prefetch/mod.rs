use crate::core::GarudaError;
use crate::memory::MemoryManager;
use crate::predictor::ExpertPredictor;
use std::sync::Arc;

pub struct PrefetchEngine {
    pub memory_manager: Arc<MemoryManager>,
    pub predictor: ExpertPredictor,
}

impl PrefetchEngine {
    pub fn new(memory_manager: Arc<MemoryManager>, predictor: ExpertPredictor) -> Self {
        Self {
            memory_manager,
            predictor,
        }
    }

    pub async fn prefetch_for_tokens(&self, tokens: &[crate::core::Token]) -> Result<(), GarudaError> {
        let predicted = self.predictor.predict_next_experts(tokens)?;
        let mem = self.memory_manager.clone();
        
        for expert_id in predicted {
            let mem_clone = mem.clone();
            tokio::spawn(async move {
                let _ = mem_clone.get_expert(expert_id);
            });
        }
        
        Ok(())
    }
}

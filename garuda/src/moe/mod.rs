use crate::core::{Token, Tensor, GarudaError};
use crate::router::{Router, RouterType};
use crate::memory::MemoryManager;
use std::sync::Arc;

pub struct MoeEngine {
    pub router: Router,
    pub memory_manager: Arc<MemoryManager>,
}

impl MoeEngine {
    pub fn new(router_type: RouterType, num_experts: usize, top_k: usize, memory_manager: Arc<MemoryManager>) -> Self {
        Self {
            router: Router::new(router_type, num_experts, top_k),
            memory_manager,
        }
    }

    pub fn forward(&self, tokens: &[Token]) -> Result<Tensor, GarudaError> {
        let mut final_output_data = vec![0.0; tokens.len() * 128];
        
        for (token_idx, &token) in tokens.iter().enumerate() {
            let routed = self.router.route(token)?;
            
            for (expert_id, weight) in routed {
                let expert = self.memory_manager.get_expert(expert_id)?;
                
                let expert_data = &expert.weights.data;
                for i in 0..128 {
                    let w_val = expert_data[i % expert_data.len()];
                    final_output_data[token_idx * 128 + i] += w_val * (token as f32) * weight;
                }
            }
        }
        
        Ok(Tensor::new(vec![tokens.len(), 128], final_output_data))
    }
}

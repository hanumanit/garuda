use crate::core::{Token, ExpertId, GarudaError};

#[derive(Debug, Clone, Copy)]
pub enum RouterType {
    Mixtral,
    DeepSeek,
    QwenMoe,
}

pub struct Router {
    pub router_type: RouterType,
    pub num_experts: usize,
    pub top_k: usize,
}

impl Router {
    pub fn new(router_type: RouterType, num_experts: usize, top_k: usize) -> Self {
        Self {
            router_type,
            num_experts,
            top_k,
        }
    }

    pub fn route(&self, token: Token) -> Result<Vec<(ExpertId, f32)>, GarudaError> {
        let mut expert_scores = Vec::new();
        for i in 0..self.num_experts {
            let score = (((token * (i as u32 + 1)) % 100) as f32) / 100.0;
            expert_scores.push((i as ExpertId, score));
        }

        expert_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_k_experts: Vec<(ExpertId, f32)> = expert_scores.into_iter().take(self.top_k).collect();

        let total_score: f32 = top_k_experts.iter().map(|(_, s)| s).sum();
        let normalized = top_k_experts
            .into_iter()
            .map(|(id, s)| (id, if total_score > 0.0 { s / total_score } else { 1.0 / self.top_k as f32 }))
            .collect();

        Ok(normalized)
    }
}

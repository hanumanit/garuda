use crate::core::{Token, ExpertId, GarudaError};

pub struct ExpertPredictor {
    pub num_experts: usize,
}

impl ExpertPredictor {
    pub fn new(num_experts: usize) -> Self {
        Self { num_experts }
    }

    pub fn predict_next_experts(&self, tokens: &[Token]) -> Result<Vec<ExpertId>, GarudaError> {
        if tokens.is_empty() {
            return Ok(vec![0]);
        }
        let last_token = tokens[tokens.len() - 1];
        let predicted_1 = (last_token % self.num_experts as u32) as ExpertId;
        let predicted_2 = ((last_token + 1) % self.num_experts as u32) as ExpertId;
        
        Ok(vec![predicted_1, predicted_2])
    }
}

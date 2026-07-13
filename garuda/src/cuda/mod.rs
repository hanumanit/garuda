use crate::core::{Token, Tensor, GarudaError, InferenceBackend};

pub struct CudaBackend {
    pub device_id: u32,
}

impl CudaBackend {
    pub fn new(device_id: u32) -> Self {
        Self { device_id }
    }
}

impl InferenceBackend for CudaBackend {
    fn forward(&self, tokens: &[Token], _kv_cache: &mut crate::cache::KVCacheState) -> Result<Tensor, GarudaError> {
        let mut data = vec![0.0; tokens.len() * 128];
        for i in 0..data.len() {
            data[i] = (tokens[i % tokens.len()] as f32) * 1.5;
        }
        Ok(Tensor::new(vec![tokens.len(), 128], data))
    }
}

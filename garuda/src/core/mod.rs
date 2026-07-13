use std::sync::Arc;
use thiserror::Error;

pub type Token = u32;
pub type ExpertId = u32;

#[derive(Error, Debug, Clone)]
pub enum GarudaError {
    #[error("I/O error: {0}")]
    Io(String),
    
    #[error("Storage error: {0}")]
    Storage(String),
    
    #[error("Inference error: {0}")]
    Inference(String),
    
    #[error("Scheduler error: {0}")]
    Scheduler(String),
    
    #[error("Cache error: {0}")]
    Cache(String),
    
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
    
    #[error("Model error: {0}")]
    Model(String),

    #[error("Timeout error")]
    Timeout,

    #[error("Rate limit exceeded")]
    RateLimit,
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        Self { shape, data }
    }
    
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self {
            shape,
            data: vec![0.0; size],
        }
    }
}

#[derive(Debug)]
pub struct Expert {
    pub id: ExpertId,
    pub weights: Tensor,
    pub hits: usize,
    pub loaded_at: std::time::Instant,
}

// Traits for plugin-based architecture

pub trait StorageBackend: Send + Sync {
    fn read_file(&self, path: &str) -> Result<Vec<u8>, GarudaError>;
    fn read_mmap(&self, path: &str) -> Result<memmap2::Mmap, GarudaError>;
    fn write_file(&self, path: &str, data: &[u8]) -> Result<(), GarudaError>;
}

pub trait ExpertLoader: Send + Sync {
    fn load(&self, id: ExpertId) -> Result<Arc<Expert>, GarudaError>;
    fn unload(&self, id: ExpertId) -> Result<(), GarudaError>;
    fn prefetch(&self, id: ExpertId) -> Result<(), GarudaError>;
}

pub trait InferenceBackend: Send + Sync {
    fn forward(&self, tokens: &[Token], kv_cache: &mut crate::cache::KVCacheState) -> Result<Tensor, GarudaError>;
}

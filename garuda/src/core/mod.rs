//! Core types, errors and the plugin traits every backend implements.

use std::path::Path;
use std::sync::Arc;
use thiserror::Error;

pub type Token = u32;
pub type ExpertId = u32;

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum GarudaError {
    #[error("i/o error: {0}")]
    Io(String),

    #[error("storage error: {0}")]
    Storage(String),

    #[error("inference error: {0}")]
    Inference(String),

    #[error("scheduler error: {0}")]
    Scheduler(String),

    #[error("cache error: {0}")]
    Cache(String),

    #[error("model error: {0}")]
    Model(String),

    #[error("token id {0} is outside the vocabulary")]
    InvalidToken(Token),

    #[error("configuration error: {0}")]
    Config(String),

    #[error("request timed out")]
    Timeout,

    #[error("request cancelled")]
    Cancelled,

    #[error("server is at capacity, retry later")]
    Busy,

    #[error("too many concurrent requests for this user")]
    RateLimit,
}

/// Shape of the (untrained) model this runtime executes.
///
/// The dimensions are deliberately small so the whole thing runs in a few MB of
/// RAM. They are the knobs a real GGUF loader would fill in from file metadata.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelDims {
    pub d_model: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub d_ff: usize,
    pub n_experts: usize,
    pub top_k: usize,
    pub vocab_size: usize,
    /// Positions per KV cache block.
    pub block_size: usize,
    /// Base frequency for rotary embeddings.
    pub rope_theta: f32,
}

impl Default for ModelDims {
    fn default() -> Self {
        Self {
            d_model: 128,
            n_heads: 4,
            head_dim: 32,
            d_ff: 256,
            n_experts: 8,
            top_k: 2,
            vocab_size: crate::tokenizer::VOCAB_SIZE,
            block_size: 16,
            rope_theta: 10_000.0,
        }
    }
}

impl ModelDims {
    pub fn validate(&self) -> Result<(), GarudaError> {
        if self.n_heads * self.head_dim != self.d_model {
            return Err(GarudaError::Config(format!(
                "n_heads * head_dim ({} * {}) must equal d_model ({})",
                self.n_heads, self.head_dim, self.d_model
            )));
        }
        if self.top_k == 0 || self.top_k > self.n_experts {
            return Err(GarudaError::Config(format!(
                "top_k ({}) must be in 1..={}",
                self.top_k, self.n_experts
            )));
        }
        if self.block_size == 0 {
            return Err(GarudaError::Config("block_size must be non-zero".into()));
        }
        Ok(())
    }
}

/// A dense row-major f32 tensor.
///
/// `new` validates that `shape` and `data` agree, so no downstream code has to
/// defend against a tensor whose dimensions lie about its contents.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self, GarudaError> {
        let expected: usize = shape.iter().product();
        if expected != data.len() {
            return Err(GarudaError::Inference(format!(
                "shape {shape:?} implies {expected} elements but {} were supplied",
                data.len()
            )));
        }
        Ok(Self { shape, data })
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self {
            shape,
            data: vec![0.0; size],
        }
    }

    /// 1-D tensor from a vector.
    pub fn vector(data: Vec<f32>) -> Self {
        Self {
            shape: vec![data.len()],
            data,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn into_data(self) -> Vec<f32> {
        self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// One MoE expert: a SwiGLU feed-forward network.
///
/// `gate`/`up` are `[d_ff, d_model]` row-major, `down` is `[d_model, d_ff]`.
#[derive(Debug)]
pub struct Expert {
    pub id: ExpertId,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
    pub dims: ModelDims,
    pub loaded_at: std::time::Instant,
}

impl Expert {
    /// Number of f32 values one expert occupies, given `dims`.
    pub fn n_params(dims: &ModelDims) -> usize {
        2 * dims.d_ff * dims.d_model + dims.d_model * dims.d_ff
    }

    pub fn size_bytes(&self) -> usize {
        (self.gate.len() + self.up.len() + self.down.len()) * std::mem::size_of::<f32>()
    }
}

/// Byte-level access to a tier of storage. Implemented by [`crate::storage::LocalStorageBackend`];
/// an S3/NFS backend would slot in here without touching the cache or scheduler.
pub trait StorageBackend: Send + Sync {
    fn read(&self, path: &Path) -> Result<Vec<u8>, GarudaError>;
    fn write(&self, path: &Path, data: &[u8]) -> Result<(), GarudaError>;
    fn remove(&self, path: &Path) -> Result<(), GarudaError>;
    fn exists(&self, path: &Path) -> bool;
}

/// Resolves an [`ExpertId`] to loaded weights, wherever they currently live.
pub trait ExpertLoader: Send + Sync {
    fn load(&self, id: ExpertId) -> Result<Arc<Expert>, GarudaError>;
    fn unload(&self, id: ExpertId);
    /// Best-effort: warm `id` into the fastest tier. Errors are advisory —
    /// the forward pass loads what it needs regardless.
    fn prefetch(&self, id: ExpertId) -> Result<(), GarudaError>;
    /// True when `id` is already in the fastest tier, so prefetching it is pointless.
    fn is_resident(&self, id: ExpertId) -> bool;
}

/// Produces next-token logits for a context. This is the intended extension point
/// for other compute backends; [`crate::moe::MoeEngine`] is the only implementation
/// that exists — there is no GPU backend, and this trait is where one would go.
/// # Plugin contract
///
/// This trait is the extension point for compute backends. Implementations get to
/// pick their own architecture and weights; in return they must uphold the
/// invariants below, which the runtime relies on and does not re-check:
///
/// 1. **Consume only unseen positions.** Process exactly `context[seq.len()..]` and
///    no more. The runtime calls with a `context` that grows by one token per decode
///    step; re-processing the whole prefix each time would make decoding O(n²) and
///    corrupt the KV cache by appending duplicate positions.
/// 2. **Advance every KV layer by one position per new token.** After consuming `k`
///    new tokens, every layer of `seq` must be exactly `k` positions longer, so
///    `seq.len()` (which reads layer 0) stays in lockstep with all layers. Append to
///    `seq.layer(l)` — do not keep KV state anywhere else.
/// 3. **`dims()` must agree with the tokenizer.** `dims().vocab_size` must equal the
///    paired `Tokenize::vocab_size`, and `logits` must return a tensor of that
///    length. `dims()` must satisfy [`ModelDims::validate`].
/// 4. **Errors, never panics, on bad input.** An out-of-vocabulary token is
///    [`GarudaError::InvalidToken`]; an exhausted context window is the
///    [`GarudaError::Cache`] the KV cache returns — propagate it. An empty `context`
///    is an error.
/// 5. **Determinism.** For the same `context` and internal weights, `logits` must be
///    reproducible. All randomness lives in the sampler, not here — the prompt cache
///    assumes a cache hit reproduces the same continuation.
///
/// See [`crate::moe::MoeEngine`] (synthetic) and [`crate::llama::LlamaBackend`]
/// (real GGUF model) for two implementations.
pub trait InferenceBackend: Send + Sync {
    /// The model's shape. Must satisfy [`ModelDims::validate`] and agree with the
    /// tokenizer's vocabulary size.
    fn dims(&self) -> ModelDims;

    /// The model's `d_model`-dim final hidden state after consuming `context`.
    ///
    /// Used for embeddings. Subject to every invariant in the trait contract.
    fn hidden(
        &self,
        context: &[Token],
        seq: &mut crate::cache::SeqState,
    ) -> Result<Tensor, GarudaError>;

    /// Logits over the vocabulary for the token that follows `context`. Length must
    /// equal `dims().vocab_size`.
    fn logits(
        &self,
        context: &[Token],
        seq: &mut crate::cache::SeqState,
    ) -> Result<Tensor, GarudaError>;
}

//! Garuda — a Rust MoE inference runtime with tiered expert storage.
//!
//! ## What this is
//!
//! A working inference *engine* with no trained model behind it. The transformer
//! arithmetic is real — causal multi-head attention with RoPE, top-k MoE routing,
//! SwiGLU experts, a paged KV cache that spills to disk, temperature/top-k/top-p
//! sampling — and it runs over weights that are pseudo-random but deterministic
//! (see [`weights`]). Generated text is therefore meaningless. Everything *around*
//! the weights — scheduling, tiering, caching, streaming, cancellation — is real
//! and is what this codebase is actually for.
//!
//! ## What this is not
//!
//! - It cannot load a trained checkpoint. [`gguf`] parses GGUF metadata and tensor
//!   descriptors correctly, but nothing dequantises those tensors into [`weights`].
//!   That is the gap between this and a usable runtime.
//! - There is no GPU backend. [`core::InferenceBackend`] is where one would go;
//!   `gpu = true` in the config is a startup error, not a silent fallback.
//! - There is no authentication. Do not expose it to a network you do not control.

pub mod anthropic;
pub mod api;
pub mod attention;
pub mod benchmark;
pub mod cache;
pub mod cli;
pub mod config;
pub mod core;
pub mod gguf;
pub mod llama;
pub mod llamacpp;
pub mod memory;
pub mod moe;
pub mod ollama;
pub mod predictor;
pub mod prefetch;
pub mod quant;
pub mod router;
pub mod runtime;
pub mod scheduler;
pub mod server;
pub mod session;
pub mod simd;
pub mod storage;
pub mod tgi;
pub mod tokenizer;
pub mod websocket;
pub mod weights;

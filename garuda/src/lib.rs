//! Garuda — a Rust MoE inference runtime with tiered expert storage.
//!
//! ## What this is
//!
//! A working inference *engine* with two interchangeable backends behind the same
//! [`core::InferenceBackend`] trait, scheduler and API:
//!
//! - [`llama`] loads a real GGUF checkpoint — F32/F16/Q4_0/Q8_0/Q2_K–Q6_K, dense or
//!   MoE (merged `..._exps` tensors or the older per-expert tensor layout) — and
//!   runs real causal multi-head attention with RoPE, top-k MoE routing, and SwiGLU
//!   experts over the actual trained weights. With `mmap`, a checkpoint far larger
//!   than RAM runs by paging in only the rows a token touches.
//! - [`moe`] is the same transformer arithmetic over weights that are pseudo-random
//!   but deterministic (see [`weights`]) when no checkpoint is configured. Generated
//!   text is therefore meaningless, but it exercises everything *around* the
//!   weights — scheduling, tiering, caching, streaming, cancellation — without
//!   needing a checkpoint on disk.
//!
//! ## What this is not
//!
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
pub mod ui;
pub mod websocket;
pub mod weights;

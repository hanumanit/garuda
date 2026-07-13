# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-07-13

### Added
- **Core Abstractions (`core`)**: Defined main types (`Token`, `ExpertId`, `Expert`, `Tensor`, `GarudaError`) and plugin traits (`StorageBackend`, `ExpertLoader`, `InferenceBackend`).
- **Configuration Module (`config`)**: Loaded TOML-based settings representing the execution profiles.
- **tiered Storage Memory Manager (`memory`)**: Configured L1 RAM, L2 SSD Cache, and L3 HDD tier routing using `mmap`-backed loaders.
- **Storage Subsystem (`storage`)**: Provided local disk access and mmap support.
- **Tokenizer Module (`tokenizer`)**: Built an encoder/decoder vocabulary parser.
- **GGUF Reader (`gguf`)**: Designed a header and metadata reader for GGUF model files.
- **Attention Layer (`attention`)**: Modeled dot-product attention calculation.
- **Router Module (`router`)**: Crafted routers for Mixtral, DeepSeek, and Qwen MoE architectures.
- **Mixture of Experts Engine (`moe`)**: Created forwarding loops and Expert Streaming orchestration.
- **Cache Module (`cache`)**: Configured Prompt Cache, Expert Cache (LRU), Embedding Cache, Tokenizer Cache, and Paged KV Cache with Disk Spilling.
- **Predictor Module (`predictor`)**: Developed Expert Predictor to anticipate next-step active experts.
- **Prefetch Engine (`prefetch`)**: Wired asynchronous expert prefetching based on predictor forecasts.
- **SIMD Accelerations (`simd`)**: Added compiler auto-vectorizable math helpers.
- **GPU Inference Backend (`cuda`)**: Created trait implementations for GPU forwarding.
- **Scheduler (`scheduler`)**: Structured priority queuing, batch merging, streaming, cancellation, timeouts, and rate limits.
- **Axum REST API Server (`api`)**: Set up OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`) supporting SSE streaming.
- **Axum WebSocket Server (`websocket`)**: Enabled bi-directional token streaming and client-side cancellation.
- **CLI Subcommands (`cli`)**: Programmed Clap-based command configurations (`Serve`, `Benchmark`).
- **Microbenchmarks (`benchmark`)**: Verifies targets (Startup, Expert Load, Hit Rate, Token Latency, Throughput).
- **Integration Tests**: Added verification suites for the tokenizer, MoE, and scheduler pipelines.

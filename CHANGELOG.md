# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0] - 2026-07-13

Garuda can now load and run a real model. Point it at a GGUF checkpoint and it
generates real text — the TinyStories 260K model produces coherent children's
stories through the same runtime, scheduler and API as everything else.

### Added

- **`llama::LlamaBackend`** — a Llama-family dense transformer loaded from GGUF:
  per-block RMSNorm, grouped-query attention with RoPE, SwiGLU feed-forward, a
  final norm and an output projection. It implements the existing
  `core::InferenceBackend`, so it drops into the runtime, scheduler and API with
  nothing else changed. This is the plugin architecture paying off: the real model
  is a new backend behind a trait the rest of the system already depended on.
- **`tokenizer::spm::SpmTokenizer`** — the real SentencePiece tokenizer, loaded from
  the checkpoint's vocabulary and scores, using llama.cpp's bigram-merge
  resegmentation with byte fallback. Matching the model's own tokenization is what
  makes the output coherent instead of noise.
- **`Tokenize` and `StreamDecode` traits** — the runtime now holds its tokenizer
  behind a trait, so the byte-level and SentencePiece tokenizers are swappable the
  same way the backends are.
- **GGUF weight loading** — `Gguf::tensor_f32` reads F32/F16 tensors (with F16
  dequantised), bounds-checked, rejecting non-finite values. Quantised formats
  (`Q4_K`, `Q8_0`, …) are a clear error: their decoders are not written yet.
- `model.gguf` config key to select a checkpoint; `garuda inspect` now reports a
  file's architecture, experts and tokenizer.

### Changed

- **The KV cache is now multi-layer and GQA-aware.** `KvConfig` gained `kv_dim`
  (key/value width, narrower than `d_model` under grouped-query attention) and
  `n_layers`; `SeqState` holds one cache per transformer block. The synthetic MoE
  uses a single layer with `kv_dim == d_model` via `KvConfig::mha`, so its
  behaviour is unchanged.
- `server::Engine::build` chooses between the synthetic MoE and a loaded checkpoint;
  it is the only place that knows which backend is running.

## [0.2.0] - 2026-07-13

An audit of 0.1.0 found that the runtime did not perform inference. Every compute
path was simulated, and several of the simulations were remotely exploitable. This
release makes the engine real and the documentation honest.

### The headline

0.1.0 did not generate text. The scheduler emitted `(prompt_token + 1)` for each
token of the prompt, so a reply was always the prompt, shifted, and always exactly
as long as it. Expert weights were `Tensor::zeros(1024)`, so the MoE output was
zero regardless of input. The `attention` module computed `q[i] * scale + v[i]`,
which is not attention, and nothing called it anyway.

Garuda now runs a real transformer forward pass. It still has no trained weights —
see the README — but the arithmetic is genuine and tested.

### Security

- **Fixed a remote denial of service.** Every HTTP caller was hardcoded to the user id
  `default_user`, and the concurrency slot was released only on a success path that a
  disconnected SSE client never reached. Ten aborted streams — one `curl` loop — locked
  the entire API out permanently with `500 Rate limit exceeded`. Slots are now RAII
  permits held inside the request, returned on every path: completion, failure, timeout,
  cancellation, or the client hanging up. Pinned by
  `disconnecting_clients_do_not_permanently_lock_out_a_user`.
- **Fixed unbounded memory growth from untrusted input.** `Tokenizer::encode` inserted
  every unseen word into a shared vocabulary under a write lock, so a stream of random
  words grew the process without limit and serialised every request behind one lock. The
  tokenizer is now byte-level: a fixed 260-entry vocabulary, no growth, no lock.
- **Fixed two reachable panics.** `attention` read `q.shape[0]` before validating the
  shape (index out of bounds on an empty tensor). `moe` computed `i % expert_data.len()`,
  which divided by zero whenever an expert file was smaller than four bytes. Both are now
  errors, and the loader rejects any expert file whose length disagrees with the
  configured dimensions instead of silently truncating it to the first 100 floats.
- Added path-traversal rejection to the storage backend, and bounds-checked every length
  field in the GGUF parser.
- Added backpressure: the request queue is bounded and sheds load with `503` rather than
  absorbing unlimited work into `tokio::spawn`.

### Added

- **A real forward pass** — causal multi-head attention with rotary embeddings over a
  paged KV cache, top-k MoE routing, SwiGLU experts, RMSNorm, and a tied output head.
- **Real sampling** — greedy, temperature, top-k and nucleus (top-p), with a seeded PRNG,
  so a pinned seed reproduces a run exactly.
- **Deterministic weight synthesis** (`weights`) — pseudo-random but reproducible tensors,
  so the engine can run end to end without a checkpoint while remaining honest that it has
  none. This is the single place a GGUF loader would replace.
- **A real GGUF parser** — header, metadata key/values (including nested arrays) and tensor
  descriptors, with every length checked against the buffer. A truncated or hostile file is
  an error, never a panic. Exposed via `garuda inspect <file>`.
- **A working predictor and prefetcher** — a first-order Markov model over which experts
  actually fire, warming its predictions on a background thread. It stays silent until it
  has learned something, and its precision and recall are measured rather than asserted.
- Graceful shutdown, `/health`, and `/v1/stats` with measured counters.

### Changed

- **The scheduler was rewritten.** It sorted a batch by priority and then immediately
  spawned every entry, which orders nothing. Requests now wait in a priority heap and are
  pulled from it as decode slots free up, so priority is meaningful under contention.
  Cancellation is checked between tokens instead of once, before generation started.
- **Cancellation now works at all.** Both HTTP handlers created a `oneshot` and dropped the
  sender immediately, so the channel was closed before generation began and no cancel signal
  could ever arrive. A dropped response stream now cancels the request.
- **OpenAI compatibility.** `created` was hardcoded to `1234567890`; it is now a real
  timestamp. Streams never sent the `data: [DONE]` sentinel, so well-behaved SDKs hung until
  their own timeout; they do now. Added `usage`, honest `finish_reason` values, and OpenAI's
  error envelope with meaningful status codes.
- **`/v1/embeddings` returned `vec![0.1; 128]`** for every input. It now returns the model's
  real pooled hidden state, L2-normalised — genuinely computed, and genuinely meaningless
  until a trained checkpoint is loaded. The README says so.
- **The prompt cache did nothing but grow.** It was keyed by the full token vector, never
  evicted, and `get` discarded the cached value and returned a fresh empty state. It is now
  a bounded LRU prefix cache that actually skips prefill on a repeat prompt.
- **KV cache spilling wrote no bytes.** `spill_block` recorded a filename in a `HashMap` and
  dropped the tensor. Blocks are now serialised to disk through the storage backend and read
  back byte-identical, and a sequence's spill files are removed when it ends.
- **The benchmark printed `Cache Hit >95% 100.0% PASS` as a string literal.** Every figure it
  reports is now measured; figures that cannot be measured are not printed.
- **Configuration is now honoured.** `context`, `threads`, `expert_cache`, `prefetch` and
  `predictor` were parsed into a struct and never read — only `model.path` was used. Every
  key now reaches something, unknown keys are a startup error, and `gpu = true` fails at
  startup instead of silently running on the CPU.
- The `RouterType` variants were decorative; Mixtral, DeepSeek and Qwen now differ in where
  the softmax sits relative to top-k selection, which is the actual distinction.
- Added a `[profile.release]` with LTO and a single codegen unit. 0.1.0 claimed "compilation
  optimization" while only disabling debug symbols in dev and test builds.

### Removed

- `cuda` — an `InferenceBackend` that returned `token * 1.5` and was never wired up. There is
  no GPU backend; `core::InferenceBackend` is where one would go.
- `grpc` — an empty struct whose `run()` returned `Ok(())`.
- Ten unused dependencies: `dashmap`, `crossbeam`, `lru`, `zstd`, `bytes`, `headers`,
  `memmap2`, `tokio-stream`, and others. `memmap2` went with the claim that experts were
  memory-mapped; they were read into `Vec<f32>` and truncated to 100 elements.

### Tests

104 unit tests and 12 end-to-end HTTP tests, up from 2. Several exist specifically to pin the
bugs above: the disconnect DoS, the `[DONE]` sentinel, the hardcoded timestamp, the constant
embedding vector, and generation that echoed the prompt.

## [0.1.0] - 2026-07-13

Initial scaffold. See 0.2.0 for what it actually did.

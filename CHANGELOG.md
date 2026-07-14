# Changelog

All notable changes to this project will be documented in this file.

## [0.7.0] - 2026-07-14

Two more API front ends — Garuda now speaks OpenAI, Ollama and Anthropic, so most
existing clients work against it unchanged.

### Added

- **Ollama-compatible API** (`ollama` module): `POST /api/generate`, `POST /api/chat`
  (newline-delimited-JSON streaming, params under `options`), plus `GET /api/tags` and
  `GET /api/version`. Includes an RFC 3339 `created_at` computed without a date crate.
- **Anthropic Messages API** (`anthropic` module): `POST /v1/messages`, with content
  blocks, a system prompt, and the full typed SSE stream (`message_start`,
  `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`,
  `message_stop`).

Both are thin translation layers over the existing scheduler — the engine is untouched,
exactly like the OpenAI and WebSocket front ends. Verified end to end: streaming and
non-streaming replies, the correct wire shapes and event sequences, and content-block /
`options` parsing.

## [0.6.1] - 2026-07-14

An integer matmul kernel for Q8_0 — 2.6× faster on Apple Silicon.

### Added

- **`simd::dot_i8`** — an `i8` dot product that uses baseline NEON on aarch64 (widening
  `i8×i8→i16` multiply + pairwise accumulate into `i32`) and a scalar fallback elsewhere.
  Tested to equal the exact integer result on this machine.
- **A Q8_0 integer matmul.** For packed Q8_0 weights, `quant::matvec` now quantises the
  activation to int8 once (per 32-block, ggml-style) and dots it against the already-int8
  weight rows with `dot_i8`, never expanding weights to f32.

Measured on the Q8_0 build of TinyStories 15M under `mmap`: 116 → **306 tok/s** (2.6×),
with identical generated text — the small activation-quantisation error, the same tradeoff
llama.cpp makes, doesn't change the output.

### Not done

- The k-quants keep the dequantise-to-f32 path: they are bottlenecked on unpacking their
  sub-byte quants, not on the dot, so an integer kernel helps far less and is much fiddlier.

## [0.6.0] - 2026-07-14

A mixture-of-experts feed-forward path — the streaming payoff of the packed-weight
work. A token now runs only the experts it routes to.

### Added

- **MoE in the Llama backend.** When a checkpoint declares experts (`llama.expert_count`)
  and a block has the stacked expert tensors, its feed-forward becomes a mixture of
  experts: a router (`ffn_gate_inp`) scores the experts, softmax + top-k + renormalise
  (standard Mixtral gating), and only the selected experts run — each read as a row-slice
  of the stacked `ffn_{gate,up,down}_exps` tensors via the new `Weight::matvec_rows`.
  Under `mmap`, a token therefore pages in only its top-k experts, not the whole layer —
  the expert-streaming property.
- A minimal in-memory GGUF writer in the tests, used to build a synthetic 4-expert
  (top-2) model and verify the MoE path end to end.

Verified: the synthetic MoE model loads, routes and produces finite logits; different
contexts give different outputs; and the packed (`mmap`) run matches the f32-expanded
run — which proves the per-expert byte offsets into the stacked tensors are right. The
dense path (TinyLlama Q4_K_M) is unchanged, still "Paris" in both modes.

### Not done

- No real large MoE (e.g. Mixtral) was run: the smallest Mixtral quant is ~16 GB and
  this environment had ~5 GB of disk and a slow link. The MoE path is verified
  structurally and against a synthetic model, not against a famous checkpoint's output.

## [0.5.1] - 2026-07-14

Made the packed (`mmap`) path faster.

### Changed

- The quantised decoders now write into a caller-supplied buffer (`quant::dequant_into`)
  instead of returning a fresh `Vec`. `quant::matvec` gives each rayon worker one reusable
  buffer, so a packed matmul no longer allocates per row, and it skips the per-row
  finiteness check that the batch `dequantize` still does. Same math, same output — every
  Q2_K…Q6_K model still answers "Paris" in both modes.
- Measured on TinyLlama-1.1B Q4_K_M: the `mmap` path's slowdown versus f32-expand went from
  ~1.8× to ~1.34×, at the same ~6× memory saving.

## [0.5.0] - 2026-07-14

Memory-mapped, packed weights — the second phase of the disk-streaming rebuild. A
quantised model can now run without expanding to `f32` in RAM.

### Added

- **`mmap = true`** (config): the Llama backend keeps each projection matrix packed in
  the memory-mapped GGUF file and dequantises it a row at a time during matmul, via the
  new `quant::matvec`. Weights never expand to `f32`, so the process uses roughly the
  file's on-disk size.
- A `Weight` abstraction in the backend with two forms — `Full` (expanded `f32`, the
  fast default) and `Packed` (mmap + per-row dequant) — behind one `matvec`/`row` API,
  so the forward pass doesn't know which it's using.

Measured on TinyLlama-1.1B Q4_K_M: resident memory dropped from **3953 MB to 622 MB**
(~6.4×, near the 638 MB file), with identical output ("Paris") and about 1.8× slower
generation — the packed-weights tradeoff.

### Changed

- `memmap2` is a dependency again, and now actually used.
- The remaining limit is reframed honestly: this is the packed-weight foundation, but
  the backend is a *dense* Llama, so a model larger than RAM would page all its weights
  every token. Efficient streaming needs a real MoE backend (load only the routed
  experts) — the next phase — and an integer matmul kernel would cut the per-row
  dequant cost.

## [0.4.2] - 2026-07-14

The rest of the k-quants — Garuda now decodes every `Q2_K … Q6_K` format, so nearly
any GGUF download loads.

### Added

- **`Q2_K`, `Q3_K` and `Q5_K` dequantisation**, completing the k-quant set:
  - Q5_K: Q4_K plus a 5th bit per quant selected from `qh` by a per-group mask.
  - Q2_K: 2-bit quants with 4-bit packed scale/min pairs.
  - Q3_K: 3-bit quants with an inverted high-bit mask, and the 16 signed 6-bit scales
    unpacked from 12 bytes via ggml's 32-bit word juggling — the fiddliest of the set.

Verified end to end: TinyLlama-1.1B in **Q2_K, Q3_K_M and Q5_K_M** all load and answer
"the capital of France is" with "Paris" (Q3_K_M's reply, "Paris, and the official
language is French", exercises Q3_K, Q4_K, Q5_K and Q6_K in one forward pass).

### Changed

- Load support is now F32, F16, Q4_0, Q8_0 and the whole k-quant family Q2_K–Q6_K.
  The one real limit left: weights expand to `f32` at load, so a model must fit in RAM
  at full precision — the memory-mapped, integer-kernel phase is still ahead. (The
  `*_1` linear quants and IQ imatrix quants also remain undecoded.)

## [0.4.1] - 2026-07-14

The k-quants — so Garuda now loads the `*_K_M` checkpoints that make up most GGUF
downloads.

### Added

- **`Q4_K` and `Q6_K` dequantisation** in the `quant` module: the super-block scale
  and min unpacking (ggml's `get_scale_min_k4`) and the 6-bit `ql`/`qh` assembly,
  byte-for-byte with the reference. Together they cover a `*_K_M` file whole.

Verified end to end: **TinyLlama-1.1B Q4_K_M** (real Q4_K + Q6_K weights) loads and
answers "the capital of France is" with "Paris" — a wrong decoder would produce noise.

### Changed

- The load limit went from "F32/F16/Q4_0/Q8_0" to add `Q4_K`/`Q6_K`. Still missing,
  and named as the next phases: the remaining k-quants (`Q2_K`/`Q3_K`/`Q5_K`), and
  keeping weights packed with an integer matmul kernel so a model larger than RAM can
  run — today everything is expanded to `f32` at load.

## [0.4.0] - 2026-07-14

First step toward running the quantised checkpoints people actually download, and
toward the disk-streaming architecture that lets a model larger than RAM run.

### Added

- **`quant` module** — GGUF weight dequantisation for `Q4_0` and `Q8_0` (alongside
  `F32`/`F16`), the two simplest linear quants. `Gguf::tensor_f32` now delegates all
  block formats to it, so quantised `q4_0`/`q8_0` model files load whole. Verified
  end to end: the Q8_0 and Q4_0 builds of TinyStories 15M both load and generate
  coherent stories.
- `garuda inspect` reports which tensor blocks a file's decoder is missing, rather
  than lumping everything quantised together.

### Changed

- The "F32/F16 only" limit is now "F32/F16/Q4_0/Q8_0". The k-quant super-block
  formats (`Q4_K`, `Q6_K`, …) that dominate modern downloads still need a decoder
  that is not written yet — and weights are still fully expanded to `f32` at load, so
  this does not yet enable models larger than RAM. Both are named as the next phases.

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

# Garuda — a Rust MoE inference runtime with tiered expert storage

![Garuda Cover](./garuda_cover.jpg)

Garuda is an inference **engine** for Mixture-of-Experts models: a scheduler, a
tiered expert cache, a paged KV cache, and an OpenAI-compatible API, written in
Rust.

## Read this first

**Garuda has no trained model.** It cannot load one yet. The transformer
arithmetic is real — causal multi-head attention with RoPE, top-k MoE routing,
SwiGLU experts, a KV cache that spills to disk, temperature/top-k/top-p sampling
— but it runs over weights that are pseudo-random and deterministic. **The text it
generates is meaningless.** That is by construction, not a bug.

What is real is everything *around* the weights: the scheduling, the memory
tiering, the caching, the streaming, the cancellation, the load shedding. That is
what this codebase is for. It is an honest skeleton with a real skeleton's
mechanics — and a clearly marked hole where the model goes.

| | Status |
|---|---|
| Transformer forward pass (attention, RoPE, MoE, SwiGLU, sampling) | Real, tested |
| Tiered expert storage (L1 RAM → L2 disk → L3 archive) | Real, tested |
| Paged KV cache with disk spill | Real, tested |
| Scheduler (priority, concurrency limits, cancellation, timeouts, backpressure) | Real, tested |
| OpenAI-compatible API + SSE + WebSocket | Real, tested |
| GGUF metadata + tensor descriptors | Parsed correctly |
| **Loading GGUF weights (dequantisation)** | **Not implemented** |
| **Trained model / meaningful output** | **Not implemented** |
| **GPU backend** | **Not implemented** (`gpu = true` is a startup error) |
| **Authentication** | **Not implemented** — do not expose this to a network |

The gap between this and a usable runtime is one thing: turning GGUF's quantised
tensor blocks into `weights::Expert`. Everything else is waiting for it.

---

## Architecture

```mermaid
graph TD
    Client([Client]) -->|REST / SSE / WS| API[axum API]
    API -->|submit| Sched[Scheduler: priority heap,<br/>bounded concurrency]
    Sched -->|one token at a time| RT[Runtime: decode loop + sampler]

    subgraph Forward pass
        RT --> Embed[Embedding]
        Embed --> Attn[Causal MHA + RoPE]
        Attn --> Router[Router: mixtral / deepseek / qwen]
        Router --> Experts[Top-k SwiGLU experts]
        Experts --> Logits[Tied output head]
    end

    Attn <-->|read / append| KV[Paged KV cache]
    KV -.->|spill / reload| Disk[(Disk)]

    Experts -->|load| MM[Memory manager]
    MM --> L1[L1 RAM: byte-budgeted LRU]
    L1 -.->|miss| L2[L2 disk cache]
    L2 -.->|miss| L3[L3 archive]

    Experts -->|experts used| Pred[Markov predictor]
    Pred -->|likely next experts| Pre[Prefetcher]
    Pre -.->|warm in background| L1
```

**Expert streaming** means what it says: a token pulls in only the `top_k` experts
it routes to, through the tiered cache — not the whole layer. The predictor learns
a first-order Markov model over which experts actually fire, and the prefetcher
warms its guesses on a background thread while the current token is still
computing. A wrong guess costs one wasted load and can never change the output.

---

## Getting started

```bash
cd garuda

# Run the API server (config.toml is read if present)
cargo run --release -- serve

# Measure startup, expert-load latency, cache behaviour and decode throughput
cargo run --release -- benchmark --iterations 40 --tokens 32

# Read a GGUF file's metadata (weights cannot be loaded yet)
cargo run --release -- inspect model.gguf

cargo test
```

Configuration lives in [`garuda/config.toml`](garuda/config.toml). Every key
reaches something; an unknown key is a startup error rather than being silently
ignored.

---

## API

OpenAI-compatible where it counts: `created` is a real timestamp, streams end with
the `data: [DONE]` sentinel SDKs wait for, `usage` is reported, `finish_reason`
says what actually happened, and errors arrive in OpenAI's error envelope with the
status code clients act on — `429` for rate limits, `503` when the queue is full.

| Endpoint | Notes |
|---|---|
| `POST /v1/chat/completions` | `stream: true` for SSE |
| `POST /v1/completions` | |
| `POST /v1/embeddings` | Real pooled hidden states. Untrained, so they carry no meaning — see below |
| `GET /v1/models` | |
| `GET /v1/stats` | Measured scheduler and cache counters |
| `GET /health` | |
| `WS /v1/ws` | Bidirectional streaming with `{"cancel": true}` |

```bash
curl -s localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":16,"stream":true}'
```

Two extensions beyond the OpenAI shape:

- `X-Garuda-User` identifies the caller for per-user concurrency limits. Absent, everyone
  shares the `anonymous` bucket. **This is not authentication** — anyone can claim any
  name. It is a fairness knob, not a security control.
- `"priority": "low" | "normal" | "high"` on any request.

**About `/v1/embeddings`:** the vectors are genuine — a real forward pass, mean
of the final hidden state, L2-normalised. They are also *useless*, because the
weights are untrained: two similar sentences have no reason to land near each
other. The endpoint returns the right shape, computed the right way, from a model
that knows nothing. It exists so the plumbing is exercised, not so you can search
with it.

---

## Where the model would go

1. **Dequantise GGUF tensors.** [`gguf`](garuda/src/gguf/mod.rs) already reads the
   header, metadata and tensor descriptors correctly, with every length bounds-checked.
   What is missing is the block-format decoders (`Q4_K`, `Q6_K`, …) that turn tensor
   data into `f32`.
2. **Replace weight synthesis.** [`weights::ModelWeights::synthesize`](garuda/src/weights/mod.rs)
   and `synthesize_expert` are the only two functions that invent values. Point them
   at the dequantised tensors and the rest of the pipeline does not move.
3. **Load the real tokenizer.** [`tokenizer`](garuda/src/tokenizer/mod.rs) is byte-level:
   lossless and bounded, but not the BPE the checkpoint was trained with. The merge table
   is in the GGUF metadata.
4. **Support more than one block.** [`moe`](garuda/src/moe/mod.rs) runs a single
   transformer block. Real models stack dozens; the loop is the easy part, the weight
   layout is the work.

---

## Licence

MIT OR Apache-2.0

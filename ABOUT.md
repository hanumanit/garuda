# About Garuda

In Hindu, Buddhist, and Thai myth the garuda (ครุฑ) is the golden divine eagle —
the mount that carries the gods, an emblem of speed and sovereignty. This project
borrows the name for the same reason: it is a runtime built to *carry models*,
efficiently, on hardware that would otherwise strain under them.

## What it is

Garuda is a Mixture-of-Experts inference runtime written in Rust. It runs in two
modes:

- **With a real checkpoint.** Point it at a GGUF file and it loads the weights and
  generates real text. A Llama-architecture forward pass — grouped-query attention
  with rotary embeddings, SwiGLU feed-forward, a SentencePiece tokenizer read from
  the file — flows through the same runtime, scheduler and API as everything else.
- **Without one.** It runs a synthetic MoE over deterministic pseudo-random weights.
  The arithmetic is real; the weights are not, so the output is meaningless. This
  mode exists to exercise the parts that are the actual point: the scheduling, the
  tiered memory, the caching, the streaming, the cancellation, the load shedding.

## What's real

The engine, not the marketing. Causal multi-head and grouped-query attention with
RoPE, top-k MoE routing, SwiGLU experts, a paged KV cache that spills to disk, and
temperature / top-k / top-p sampling. Around them: a tiered expert store that pages
weights across RAM → SSD → archive, a priority scheduler with per-user concurrency
limits and real cancellation, and an OpenAI-compatible API with SSE and WebSocket
streaming.

## What it is not

Garuda is honest about its edges. It decodes **F32, F16, Q4_0, Q8_0 and every k-quant
from `Q2_K` to `Q6_K`** — the formats nearly every GGUF download uses — and with
`mmap = true` keeps the weights packed so the model uses roughly its on-disk size.
The backend runs both dense and mixture-of-experts FFNs, executing only the top-k
experts a token routes to (and, under `mmap`, paging in only those); both GGUF expert
tensor layouts load (the merged `..._exps` tensors newer conversions use, and the
older per-expert tensors some, including the original TheBloke Mixtral quantisations,
use instead). That MoE path is verified against a real large MoE now — Mixtral-8x7B,
Q4_K_M, 26 GB — loading and generating on a 16 GB machine via `mmap`. `Q8_0` and
`Q4_K` (the dominant tensor type in that file) dot directly against an int8-quantised
activation rather than expanding to `f32` first. There is **no GPU backend**
(`gpu = true` is a startup error, not a silent fallback), and authentication is off
by default — set `server.api_keys` to require one, or do not expose it to a network
you do not control.

The plugin architecture is what makes the real model a first-class citizen rather
than a special case: a checkpoint is just another `InferenceBackend` behind a trait
the runtime already depended on. See [PLUGIN.md](PLUGIN.md).

## Facts

- **Language:** Rust (edition 2024, 1.85+)
- **Tests:** 147 (135 unit + 12 end-to-end HTTP)
- **Verified:** loads and runs both the TinyStories 260K checkpoint and a real
  Mixtral-8x7B (Q4_K_M, 26 GB) end to end, the latter on a 16 GB machine via `mmap`
- **API:** OpenAI-compatible REST + SSE + WebSocket, plus a built-in chat page at `/`
- **Licence:** MIT OR Apache-2.0

---

Copyright © 2026 HANUMANIT Co., Ltd. Dual-licensed under [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE).

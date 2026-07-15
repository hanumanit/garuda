# Writing a Garuda plugin

A plugin is a Rust type that implements one of Garuda's extension traits. There is
no plugin manifest, dynamic loading, or separate ABI: you implement a trait, and
`server::Engine::build` constructs your type instead of a built-in one. The runtime,
scheduler and API depend on the traits, not the implementations, so they never
change.

This guide walks through a complete, runnable example. The code lives at
[`garuda/examples/custom_backend.rs`](garuda/examples/custom_backend.rs) and runs
with:

```bash
cd garuda
cargo run --example custom_backend
```

For the authoritative contract — the invariants your implementation must uphold —
read the doc comments on the traits themselves (`cargo doc --open`), summarised in
the [README](README.md#adding-a-plugin). This guide shows how to satisfy them.

## The extension points

| Trait | Job | Built-in implementations |
|---|---|---|
| [`core::InferenceBackend`](garuda/src/core/mod.rs) | context → logits | `moe::MoeEngine`, `llama::LlamaBackend` |
| [`tokenizer::Tokenize`](garuda/src/tokenizer/mod.rs) | text ↔ tokens | `Tokenizer` (byte), `spm::SpmTokenizer` |
| [`core::StorageBackend`](garuda/src/core/mod.rs) | bytes on some medium | `storage::LocalStorageBackend` |
| [`core::ExpertLoader`](garuda/src/core/mod.rs) | id → expert weights | `memory::MemoryManager`, `prefetch::GgufPagePrefetcher` |

Most plugins are backends or tokenizers, so this guide centres on those.

---

## Walkthrough: a custom `InferenceBackend`

A backend turns a token context into next-token logits. The trait is three methods,
but the work is in honouring five invariants the runtime relies on and does not
re-check. We build a toy backend that satisfies all of them. Like the built-in
synthetic MoE, it runs real arithmetic over made-up weights, so its output is
meaningless — the point is the mechanics.

### Step 1 — declare the model's shape

```rust
use garuda::core::{InferenceBackend, ModelDims, Tensor, Token, GarudaError};
use garuda::cache::SeqState;
use garuda::tokenizer::VOCAB_SIZE;

struct ToyBackend { dims: ModelDims }

impl ToyBackend {
    fn new() -> Self {
        let dims = ModelDims {
            d_model: 16, n_heads: 2, head_dim: 8, d_ff: 32,
            n_experts: 1, top_k: 1,          // unused by a dense model; keep them valid
            vocab_size: VOCAB_SIZE,          // MUST match the tokenizer you pair with
            block_size: 8, rope_theta: 10_000.0,
        };
        Self { dims }
    }
}
```

`dims()` must pass `ModelDims::validate`: `n_heads * head_dim == d_model` and
`top_k` in `1..=n_experts`. **`vocab_size` must equal the paired tokenizer's
`vocab_size()`** — the sampler draws indices in `0..vocab_size`, and the tokenizer
has to be able to decode every one of them.

### Step 2 — `hidden`: consume unseen tokens, one KV position each

```rust
fn hidden(&self, ctx: &[Token], seq: &mut SeqState) -> Result<Tensor, GarudaError> {
    if ctx.is_empty() {
        return Err(GarudaError::Inference("empty context".into())); // invariant 4
    }
    let d = self.dims.d_model;
    let mut last = None;

    // Invariant 1: process ONLY the positions the sequence has not seen.
    for &tok in &ctx[seq.len()..] {
        if (tok as usize) >= self.dims.vocab_size {
            return Err(GarudaError::InvalidToken(tok));            // invariant 4
        }
        // Invariant 2: append exactly one KV position per token. A real model stores
        // this token's attention key/value; the toy stores zeros just to keep the
        // cache length in step. `append` also surfaces the context-full error.
        let zero = vec![0.0; d];
        seq.kv().append(&zero, &zero)?;

        last = Some(self.embed(tok)); // your forward pass for this token
    }

    let x = last.ok_or_else(|| GarudaError::Inference("no new tokens".into()))?;
    Tensor::new(vec![d], x) // returns the d_model-dim hidden state
}
```

The two invariants that trip people up:

- **Invariant 1 — only unseen positions.** The runtime grows `ctx` by one token per
  decode step and calls again. If you reprocess the whole prefix each time, decoding
  is O(n²) *and* you append duplicate positions to the KV cache, corrupting it. Slice
  from `seq.len()`.
- **Invariant 2 — one KV position per token, per layer.** `seq.len()` reads layer 0
  and must speak for every layer, so each new token appends exactly once to each. A
  multi-layer model loops over `seq.layer(l)`; this single-layer toy uses the
  `seq.kv()` shorthand. This is also what keeps `seq.len()` advancing so invariant 1
  works next call.

### Step 3 — `logits`: project to the vocabulary, deterministically

```rust
fn logits(&self, ctx: &[Token], seq: &mut SeqState) -> Result<Tensor, GarudaError> {
    let h = self.hidden(ctx, seq)?;
    let v = self.dims.vocab_size;

    // Invariant 3: exactly vocab_size long. Invariant 5: no randomness here — the
    // sampler owns that, and the prompt cache assumes a repeat produces the same result.
    let mut logits = vec![0.0; v];
    for (t, out) in logits.iter_mut().enumerate() {
        *out = h.data().iter().enumerate()
            .map(|(i, &hi)| hi * (((t * 7 + i * 13) % 11) as f32 - 5.0))
            .sum();
    }
    Tensor::new(vec![v], logits)
}
```

### Step 4 — run it through the real runtime

Nothing about the runtime, scheduler or sampler is special-cased for your backend.
Wrap it in an `InferenceRuntime` and generate:

```rust
let backend = std::sync::Arc::new(ToyBackend::new());
let dims = backend.dims();
let kv = KvConfig::mha(dims, 256, 64, None, None); // 1 layer, kv width == d_model
let runtime = InferenceRuntime::new(Arc::new(Tokenizer::new()), backend, kv, 8);

let params = SamplingParams { temperature: 0.8, top_p: 0.95, top_k: 40,
                              max_tokens: 16, seed: Some(1) };
let prompt = runtime.tokenizer.encode("hello plugin");
let mut session = runtime.start(&prompt, &params)?;
while let Ok(token) = runtime.next_token(&mut session, &params) {
    // stream `token` to the client
}
```

Running the full example prints reproducible gibberish — proof that a bare-minimum
backend already flows through the real sampler, KV cache and tokenizer:

```
$ cargo run --example custom_backend
prompt:    "hello plugin"
generated: 9 tokens
decoded:   "…"      (gibberish — untrained weights)
```

Run it twice: the `seed` makes the output identical, which is invariant 5 in action.

---

## Making it a first-class backend

The example wires the runtime by hand. To make `garuda serve` use your backend, add
a branch to [`server::Engine::build`](garuda/src/server/mod.rs) — the single place
that chooses a backend — and a config key to select it. This is exactly how
`llama::LlamaBackend` is wired:

```rust
// in Engine::build
match config.gguf_path() {
    Some(path) => Self::build_gguf(config, &path),   // llama::LlamaBackend
    None       => Self::build_synthetic(config),     // moe::MoeEngine
    // add your own arm here, selected by a new config key
}
```

Your build arm constructs the backend, pairs it with a tokenizer, and builds a
`KvConfig` matching the model's layer count and key/value width:

- **Full multi-head attention:** `KvConfig::mha(dims, max_positions, resident_blocks,
  sliding_window, storage)` — one layer, key/value width `d_model`.
- **Grouped-query / multi-layer:** set `KvConfig { kv_dim, n_layers, .. }` explicitly,
  as `build_gguf` does. `kv_dim` is `n_kv_heads * head_dim`; `n_layers` is the number
  of transformer blocks (`SeqState` allocates one cache per layer).

---

## A tokenizer plugin

Implement [`Tokenize`](garuda/src/tokenizer/mod.rs) to bring your own vocabulary.
The contract: `encode` is pure and thread-safe (it is shared across requests behind
an `Arc`), `decode(encode(s))` round-trips, `decode` skips special tokens, and a
streaming decoder must agree with a batch `decode`. `vocab_size()` must match the
backend's `dims().vocab_size`.

[`spm::SpmTokenizer`](garuda/src/tokenizer/spm.rs) is a full example — it loads a
SentencePiece vocabulary from a GGUF file and implements the bigram-merge encoding
with byte fallback.

---

## Testing your plugin

The backend invariants are testable directly, without the HTTP layer. The built-in
backends' test modules show the pattern; the ones worth copying:

- **Incremental decode equals a full recompute.** Feeding tokens one at a time must
  produce the same logits as processing the whole context at once — this catches
  invariant-1 and invariant-2 bugs. See
  `moe::tests::incremental_decode_matches_a_full_recompute`.
- **Determinism.** The same context yields the same logits.
- **Bad input is an error, not a panic.** An out-of-vocab token and an exhausted
  context window both return `Err`.
- **Round-trip and streaming** for a tokenizer — see `tokenizer::spm::tests`.

## Checklist

- [ ] `dims()` passes `ModelDims::validate` and its `vocab_size` matches the tokenizer.
- [ ] `hidden`/`logits` process only `ctx[seq.len()..]`.
- [ ] Exactly one KV position appended per token, per layer.
- [ ] `logits` returns a `vocab_size`-length tensor.
- [ ] Out-of-vocab token, empty context, and context-full all return `Err`, never panic.
- [ ] Output is deterministic for a fixed context and weights.
- [ ] Registered in `server::Engine::build`, selected by a config key.

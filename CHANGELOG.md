# Changelog

All notable changes to this project will be documented in this file.

## [0.27.1] - 2026-08-19

The chat page asks for a length the server can actually reach.

### Fixed

- **The built-in chat page opened at 512 tokens whatever was loaded**, which on a
  model that takes seconds per token is a request that cannot finish inside the
  server's own request timeout. Serving Qwen3.8-27B, every reply from the page came
  back as `Error: request timed out` — after the work had been done and thrown away,
  since the deadline is measured from submission and 512 tokens at ~24 s each is three
  and a half hours against a 900-second limit.

  `/v1/stats` now reports the sampling defaults this server would apply to a request
  that asks for nothing, and the page seeds its Max tokens and Temperature controls
  from them. A slow checkpoint's shipped `max_tokens` is what the page asks for; a
  fast one is unchanged in practice.

- **`qwen3.8.toml` could not finish its own default reply.** `max_tokens = 48` at
  ~24 s per token is ~19 minutes against a `request_timeout_secs = 900` limit. Now 32
  tokens against 1800 seconds, with the arithmetic written next to both keys, because
  the failure it produces looks like a server fault rather than a configuration that
  asked for more than it allowed.

## [0.27.0] - 2026-08-19

Qwen3.8-27B, whose blocks are mostly not attention.

### Added

- **The `qwen35` architecture** — `qwen35::Qwen35Backend`, a second
  `InferenceBackend` alongside `LlamaBackend`. It runs the dense Qwen3.5 family:
  **Qwen3.8-27B**, Qwen3.6-27B, and Qwen3.5-0.8B through 27B, which are all the same
  arithmetic at different widths.

  Three of every four blocks are a **gated delta net** rather than attention: linear
  attention that folds the whole sequence into a fixed-size recurrent state. Per token
  and per head, with decay `α`, write strength `β` and L2-normalised `q`/`k`:

  ```text
  S ← αS + β·k ⊗ (v − Sᵀk)      out = Sᵀ(q/√d)
  ```

  `Sᵀk` is what the state already predicts for this key, so the update writes the
  *error* along `k` — a second write of the same key changes nothing, which is a unit
  test rather than a claim. Ahead of it sits a depthwise causal convolution over the
  joint query/key/value projection, and after it a gated RMSNorm. `α` comes out of
  `exp(a · softplus(alpha·x + dt_bias))` with `a` stored negative in the file (checked
  against the real checkpoint's own `ssm_a` tensor, all 48 values in
  −0.34…−0.004), so the decay lands in `(0, 1)` and old writes fade.

  The remaining blocks are grouped-query attention, but not Llama's: heads are 256
  wide against a 5120-wide residual stream, the query projection emits a **per-head
  output gate** next to each query, queries and keys are RMS-normalised per head
  before rotation, and only the **first quarter** of each head's dimensions is rotated
  (`rope.dimension_count = 64` of 256, rotate-half pairs `(i, i+32)` — a different
  convention from `simd::rope`, and the wrong one produces fluent text that ignores
  word order).

  Verified against real weights, not only against itself. On `Qwen3.8-27B` Q4_K_M
  (19 GB, memory-mapped on a 16 GB machine) and on `Qwen3.5-0.8B` Q8_0 and Q4_K_M (24
  blocks of the identical architecture): greedy continuation of "The capital of France
  is" → `" Paris"`; a chat turn rendered with the checkpoint's own template →
  `" Paris"`; the same question in Thai, through the served OpenAI API, → `"กรุงเทพฯ"`
  from the 27B and `"กรุงเทพมหานคร"` from the 0.8B, streaming included. `tests/qwen35_real.rs` is that check, gated on
  `GARUDA_QWEN35_GGUF` so the suite still runs with no checkpoint on the machine, and
  `examples/qwen35_probe.rs` prints the most likely next tokens from a single forward
  pass, which is how a 24-second question gets asked of a model that takes minutes to
  decode a sentence.

  **The value heads read the key heads in the order the *file* stores them**, which is
  `hv % n_k_heads` and not the contiguous run `transformers` writes
  (`repeat_interleave`, `hv / group`). Both are right for their own weight layout — the
  GGUF converter writes heads in the order llama.cpp's graph reads them — and picking
  the wrong one is not a crash or obvious noise: every head still reads a real memory,
  just one another head wrote, so magnitudes stay normal and the model degenerates into
  copying its prompt. On the 27B, "The capital of France is" continued as
  `" capital of France is capital of France is"`. Nothing smaller catches it: a
  checkpoint with as many value heads as key heads — the 0.8B, and every other size
  below the 27B — makes the two conventions identical, which is why the small model
  answered correctly throughout.

- **Byte-level BPE tokenizer** (`tokenizer::bpe::BpeTokenizer`) for `gpt2`
  vocabularies, which is what Qwen ships — 248 320 entries and 247 587 merges in the
  27B's file. The three stages are pre-tokenization, the GPT-2 byte alphabet, and the
  merge list applied lowest rank first.

  The pre-tokenizer is the part that decides everything downstream, and GGUF stores its
  *name* rather than its pattern. Qwen's pattern is written out by hand here —
  alternatives tried in order, each greedy within itself, as a backtracking engine
  would — because the crate has no regex dependency and because two of the
  alternatives (`\s+(?!\S)`, and `\s*[\r\n]+` after a backtrack) need hand-holding
  either way; llama.cpp hand-codes the same splits for the same reason. That is what
  makes `"1234"` four tokens, gives a word its leading space, and keeps Thai vowel
  signs attached to their consonants.

- **Recurrent state in the sequence cache** (`cache::LinearState`). A hybrid model's
  recurrent blocks store nothing per position, so `KvConfig::kv_dims` gives them a
  width of zero: they still count positions — every layer of a sequence has to advance
  together, or `seq.len()` stops speaking for all of them — and hold nothing. What they
  do carry is a convolution history and one matrix per head, the same size at ten
  tokens as at a hundred thousand: **~144 MB for Qwen3.8-27B**, 19 MB for the 0.8B.
  `SeqState::resident_bytes` counts it, so the prompt cache's byte budget is still
  honest, and startup warns when one sequence's state exceeds the whole budget —
  otherwise the cache would accept every insertion and decline it.

- **`model.thinking`** — the Qwen3.5 chat template opens a `<think>` block for the
  assistant. `false`, the default, closes it immediately (the template's own
  `enable_thinking = false`), so a reply is the answer and nothing else; `true` leaves
  it open, which is what the checkpoint does by default, and the reasoning then arrives
  as ordinary content ahead of the answer. This server has nowhere else to put
  reasoning, and on the default 256-token budget an answer might otherwise never
  arrive — hence the default, and hence the switch.

- **[`garuda/qwen3.8.toml`](garuda/qwen3.8.toml)** — a worked config for Qwen3.8-27B
  Q4_K_M (~19 GB) on a 16 GB machine, and `garuda inspect` now reports a hybrid
  checkpoint's split (`64 (48 recurrent, 16 attention)`) along with any
  multi-token-prediction blocks it carries.

### Changed

- **Turn markers are placed as vocabulary entries when the vocabulary holds them.**
  `encode_chat` put `<|im_start|>` into the prompt as *text*, so a Qwen checkpoint read
  `<`, `|`, `im`, `_start`, `|`, `>` where it expects one marker — the same class of
  bug 0.26.0 fixed for the end-of-turn token, one layer up. Each marker is now resolved
  against the vocabulary and falls back to text only when the entry is genuinely
  absent, which is what keeps the SentencePiece checkpoints rendering exactly as
  before. On the real 0.8B this is the difference between a reply of
  `"<|im_start|>\nThe capital of France is Paris.\n</"` and `" Paris"`.

- **`ModelDims::validate` requires the heads to cover `d_model`, not to equal it.**
  Qwen3.5 projects 24 heads of 256 dimensions out of a 5120-wide residual stream and
  narrows the 6144-wide concatenation back down in the output projection. The synthetic
  `attention::Attention`, which slices `d_model` into its own heads, now checks for
  equality itself rather than relying on the shared invariant.

- **Speculative decoding is off on `qwen35`, and says why.** A recurrent state
  summarises every token it has read; there is no arithmetic that takes the last few
  back out, so a rejected guess cannot be discarded. `speculation_supported()` answers
  `false` (which keeps the runtime on the plain decode path),
  `SeqState::truncate` refuses to rewind a sequence carrying such state rather than
  silently continuing, and `model.draft_gguf` is a startup error with this architecture
  loaded instead of a key that quietly does nothing.

- **`llama::Weight` and the two GGUF tensor loaders are shared** between the backends
  rather than copied. Both architectures want the same choice between an expanded `f32`
  matrix and a packed one dequantised per row out of the memory map.

### Tests

- The delta rule against the recurrence written out longhand, including that the state
  is unchanged by rewriting the same key with the same value; partial rotation against
  the rotate-half formula, and that everything past `n_rot` passes through untouched;
  a batched prefill matching tokens fed one at a time, which is where a recurrence that
  lost the token order would still return plausible numbers; every layer advancing one
  position per token on a hybrid stack; mmapped weights agreeing with expanded ones;
  the mixture-of-experts sibling `qwen35moe` refused by name rather than half-run.

- For the tokenizer: the pre-tokenizer's split, case by case, including Thai and
  combining marks, plus a property — the split covers its input exactly once, in order,
  with no empty pieces. And the contract's own invariants: control tokens reachable by
  name but never produced by `encode`, streaming decode agreeing with batch decode
  across a character split between tokens.

## [0.26.0] - 2026-07-30

The prompt format a checkpoint was actually trained on.

### Fixed

- **Chat turns are rendered with the checkpoint's own template.** A GGUF carries
  `tokenizer.chat_template`; nothing read it. Every chat adapter built a generic
  `user: …\nassistant: ` transcript instead, which is not what any instruction-tuned
  model was fine-tuned on.

  This failed silently, which is why it survived. Handed a transcript, a chat model
  reverts to the document completer it started as: it answers, then writes the user's
  next turn and keeps going. Observed on TinyLlama — "🇫🇷 Paris" followed by
  `user: Okay, so the capital of France is Paris. Can you tell me...` — with
  `finish_reason: length` and no error anywhere. After the fix the same four prompts
  all end at `finish_reason: stop`, and the system prompt takes effect
  ("answer in one short sentence" → `2 + 2 = 4`).

  Recognised by the markers a template mentions, not by running Jinja: `<|user|>` for
  TinyLlama and Zephyr, `[INST]` for Mistral and Mixtral, `<|im_start|>` for ChatML,
  `<|start_header_id|>` for Llama 3. Verified against the two checkpoints on hand by
  decoding the assembled ids: Mixtral renders exactly
  `<s>[INST] … [/INST]`, TinyLlama exactly `<|system|>\n…</s>\n<|user|>\n…\n<|assistant|>\n`.
  ChatML and Llama 3 follow their published formats and are unit-tested, not measured.

- **A reply stops on the chat format's end-of-turn token, not only on end-of-sequence.**
  ChatML ends a turn with `<|im_end|>` and Llama 3 with `<|eot_id|>`, keeping `</s>` for
  the end of the document. A decoder watching only `eos()` never stops on those
  checkpoints. The three decode paths — single-token, speculative, batched — now share
  one `ends_turn`, because a stop condition that holds in two of them is a reply that
  runs on depending which path served it.

- **SentencePiece applies its dummy space prefix once per prompt, not once per
  fragment.** Assembling a prompt from pieces put a stray `▁` at every turn boundary:
  TinyLlama's two-turn prompt tokenized to 39 ids where the canonical string is 37,
  with a space token wedged between `</s>` and the newline. Invisible in the decoded
  text, and a position the model was never trained on.

### Added

- **`chat` module** — `ChatFormat::detect`, and `encode_chat`, which assembles a prompt
  as token ids rather than one string. Two reasons it cannot be a string: markers like
  `<|eot_id|>` are single vocabulary entries that `encode` deliberately does not
  recognise, and teaching `encode` to recognise them would let a user's own message
  close their turn and open another. Confirmed on both real vocabularies —
  `encode("</s>")` yields `</`, `s`, `>` as ordinary text, never the control id.

- **`Tokenize::encode_fragment` and `Tokenize::token_id`** — encoding a continuation
  without a leading begin-of-sequence, and exact-entry lookup for the renderer that
  legitimately needs a control id.

### Changed

- The Anthropic adapter no longer hand-rolls its own transcript. It was the third copy
  of the same format, and all three were wrong for a checkpoint that names a template.

## [0.25.0] - 2026-07-29

Speculation that pays at the temperature people actually use.

### Added

- **`model.draft_gguf`** — a small checkpoint that proposes tokens for the main model
  to check. Prompt lookup, added in 0.22.0, can only guess text already present in
  the context; 0.24.0 made it safe for sampled requests but measured no gain from it
  at `temperature = 0.8`, because a deterministic guess is kept with probability
  `p(guess)` and that is small across forty candidates. A model proposes a whole
  distribution, and the acceptance rule corrects against it, so guesses land far more
  often.

  Measured on Mixtral-8x7B Q4_K_M (25 GB, 16 GB RAM) at the shipped default
  `temperature = 0.8`, grounded prompt, with a TinyLlama-1.1B Q4_K_M draft (638 MB):

  | | s/token |
  |---|---|
  | no speculation | 9.01 (and 8.75 / 12.40 earlier) |
  | prompt lookup | 8.51 (and 10.31 / 10.03 earlier) |
  | **draft model** | **5.33 / 4.58** |

  So **1.7–2.0×** where prompt lookup won nothing. The draft competes with the target
  for page cache — a real worry on a machine already running a 25 GB model in 16 GB —
  and it still comes out ahead, which the arithmetic suggested but did not settle.

- **The vocabularies are checked at startup and a mismatch is refused.** This is the
  one failure that could not be caught later: a token id is a token id, so a draft
  that tokenises differently would hand back ids meaning different words and every
  layer below would accept them without complaint. The guesses would simply be wrong,
  or worse, right by coincidence.

### Changed

- The two rejection rules became one. `verify_drafted` was a wrapper for the
  deterministic case; there is now a single `verify_against` taking an optional draft
  distribution, so a lookup is the point-mass special case rather than a second
  implementation that has to agree with the first.

### Tests

- A draft model must not change greedy output: the tokens have to match plain
  decoding exactly, *and* the target and draft caches must end up describing the same
  tokens — a draft left holding positions for rejected guesses would poison every
  round after it. The test also asserts some round won more than one token, so it
  cannot pass by never speculating.

## [0.24.0] - 2026-07-29

### Changed

- **Sampled requests speculate too, without distorting what the caller asked for.**
  Until now guessing was greedy-only: keeping a guess because it matched the argmax
  would have handed someone who asked for `temperature = 0.8` the greedy answer.
  A guess is now kept with the probability the caller's own distribution assigns it,
  and otherwise replaced by a draw from that distribution with the guess removed —
  the standard speculative-sampling rule, which collapses to this because the
  prompt-lookup drafter proposes a single token with certainty. Over many steps that
  emits exactly the caller's distribution, which a 60,000-draw test asserts to within
  one percentage point per token, including when the guess was cut away by top-k or
  top-p and can never be kept at all.

  **This made sampled speculation correct, not fast.** Measured on Mixtral-8x7B
  (25 GB, 16 GB RAM) at the shipped default `temperature = 0.8`, paired runs with the
  order reversed:

  | | speculating | not |
  |---|---|---|
  | run 1 | 10.31 s/token | 8.75 s/token |
  | run 2 | 10.03 s/token | 12.40 s/token |

  The sign flips between rounds, so there is no gain to claim. The reason is in the
  rule itself: acceptance probability *is* `p(guess)`, and at 0.8 across forty
  candidates that is small. The 2.6× reported in 0.22.0 remains a greedy result.
  Where this does extend the win is at low-but-nonzero temperatures; the crossover
  was not measured.

  Sequences that find their guesses missing still stop drafting, so the honest
  summary is that sampled requests now *may* benefit and no longer pay when they do
  not.

- `sample` is unchanged, deliberately. The candidate-building it shares with the new
  verification was factored out, but its own arithmetic was left byte-for-byte as it
  was, so no seeded output moves.

## [0.23.0] - 2026-07-29

### Fixed

- **The prompt cache is bounded in bytes, not just entries** (`memory.prompt_cache`,
  default `512MB`). It evicted by entry count alone, which is not a bound: one entry
  holds a whole sequence's attention state, and what that costs depends entirely on
  the model. Per cached position —

  | | per position | 64 entries × 2048-token prefixes |
  |---|---|---|
  | synthetic MoE | 1 KB | 0.12 GB |
  | TinyLlama 1.1B | 44 KB | 5.5 GB |
  | Mixtral-8x7B | 256 KB | **32 GB** |

  — so the shipped default was capable of asking for 32 GB on the machine this
  runtime exists for, one already running a 25 GB checkpoint in 16 GB of RAM. The
  expert cache has been byte-budgeted since it was written (`expert_cache`); this is
  the same lesson, arrived at later and in the more dangerous place.

  An entry larger than the whole budget is declined rather than admitted, since
  taking it would evict everything else and then sit there alone. Confirmed on the
  620 MB MoE checkpoint: with `512MB` five ~76 MB prefixes are retained, with `64MB`
  none are.

- `/v1/stats` reports the prompt cache's byte usage, which it previously hardcoded
  to zero.

### Verified: a 25-minute soak

Nothing here had been run for longer than about ten minutes at a stretch, which left
leaks and drift as the largest unexamined risk. Twenty-five minutes, ten concurrent
workers over six caller identities, ~3 requests/second, 4438 requests, against the
620 MB MoE checkpoint under `mmap`: a mix of streaming and non-streaming, greedy and
sampled, four wire protocols, embeddings, varied prompt lengths to churn the prompt
cache, and **1079 deliberate mid-stream disconnects** — the path that once leaked a
concurrency permit per hang-up and locked users out permanently.

| | first third | last third |
|---|---|---|
| RSS | 467 MB mean (373–553) | 486 MB mean (431–559) |
| threads | 29 | 30 |
| file descriptors | 25 | 25 |

The counters balance exactly: **3847 submitted = 2768 completed + 1079 cancelled +
0 failed + 0 timed out, nothing unaccounted for.** Descriptors never moved. RSS
drifted 4% between thirds with the ranges overlapping heavily, which is the
memory-mapped model's pages coming and going rather than a leak. The prompt cache
held at 8 entries and a 99.7% hit ratio, so the byte budget above does bound it under
sustained churn. No panics; the one `ERROR` line is `tower_http` reporting the single
`503` that backpressure produced. Latency p50 2.8 s, p95 7.5 s.

The first attempt at this measured almost nothing: the generator sent prompts longer
than the configured context window, so half the load was rejected at the door with a
`400` and never reached the decode path. Worth recording, because a soak that soaks
nothing looks exactly like a soak that passes.

## [0.22.0] - 2026-07-29

Speculative decoding, by prompt lookup. This is the answer to the finding in 0.21.0
that decode on a larger-than-RAM checkpoint is bandwidth-bound: the fix is to read
the model fewer times, not to read it more cleverly.

### Added

- **`model.speculative_lookahead`.** A lone request guesses the next few tokens by
  finding where its recent context occurred earlier and copying what followed, then
  checks the whole run in one pass. Guesses are kept only where the model would have
  chosen them anyway, so greedy output is unchanged — the tests assert exactly that
  against plain decoding, including that the caches end up identical.

  No draft model and no extra memory, which matters on the machine this is for: one
  already running a 25 GB checkpoint in 16 GB of RAM.

  Measured on Mixtral-8x7B Q4_K_M, greedy, paired runs:

  | | no speculation | speculating |
  |---|---|---|
  | grounded prompt (answer echoes the input) | 12.47 s/token | **4.84 s/token** |
  | open-ended prompt | 13.15 s/token | 11.55 s/token |

  So ~2.6× where the guessing suits the workload, and no cost where it does not.

- `SeqState::truncate`, `InferenceBackend::logits_multi` and
  `speculation_supported`. Verifying a run of guesses needs logits at several
  positions from one pass and the ability to give back the positions belonging to
  guesses that were wrong. `logits_multi` defaults to handling one position by
  delegating and refusing more, so a backend that cannot do it says so.

### The bit that needed measuring twice

A first cut used the configured lookahead every round. On the grounded prompt that
was faster still — 3.6 s/token — but on the open-ended one it was **1.6× slower than
not guessing at all**: 16.0 s/token against 9.9. A guess is not free, because the
verification pass computes every position drafted, so six guesses that win one token
do six positions of expert arithmetic for it.

So each sequence now keeps a running average of what its guesses have been winning
and asks for a little more than that, or stops asking entirely when it falls behind
(retrying occasionally, since text turns repetitive halfway through often enough).
That trades a little of the best case — 4.84 s/token rather than 3.58 — for removing
the bad one. An operator who knows their workload echoes can raise the ceiling.

Sampled requests (`temperature > 0`) take the ordinary path throughout. Keeping a
guess because it matches the argmax would quietly replace the caller's distribution
with a greedy one; doing it properly needs rejection sampling, which is not here yet.

## [0.21.0] - 2026-07-29

Expert prefetch, measured on the case it was built for. It turns out to help
somewhere other than where it was described as helping.

The feature has been in since 0.9.0 and marked "Real, tested", but the tests only
ever asserted that it *predicts* something and does not change the output. Nothing
established that it made anything faster — and the workload it exists for, a
checkpoint larger than RAM, was not measurable here until now.

**Mixtral-8x7B Q4_K_M, 25 GB, on 16 GB of RAM**, `mmap`, batched prefill, three
paired runs with the order varied between them:

| | run 1 | run 2 | run 3 |
|---|---|---|---|
| time to first token, prefetch on | 32.7 s | 40.2 s | 51.1 s |
| time to first token, prefetch off | 61.0 s | 62.4 s | 83.0 s |
| decode, prefetch on | 11.01 s/token | 12.12 s/token | 16.64 s/token |
| decode, prefetch off | 14.45 s/token | 12.18 s/token | 10.85 s/token |

So it is worth **~1.6× on time to first token**, consistently and in every ordering.
On decode it made **no measurable difference**: the two sets overlap completely and
the sign flips between runs. The first pair alone looked like a 1.3× decode win,
which is exactly why the runs are paired and the order varied — one pair would have
supported the wrong conclusion.

The asymmetry has a plausible cause, though this measurement does not prove it. The
predictor is per layer: it warms what layer `l` will want *next token*. During
prefill many tokens pass through layer `l` back to back, so that guess is needed
almost immediately. During decode the next visit to layer `l` is a whole pass
through the other thirty-one layers away — by which time, on a model this much
larger than RAM, the pages it warmed can already have been evicted.

### Documentation

- The README row says what was measured rather than describing the mechanism. The
  "5–6×" it used to quote was `madvise` against hand-faulting *inside* the
  prefetcher, which is a fact about the implementation and reads like an end-to-end
  speedup.

### Tried and rejected: predicting the next *layer* rather than the next token

The obvious response to the result above is that the predictor is aimed the wrong
way for decode. It learns "what will layer `l` want next token", which during decode
is a whole pass away; the useful question would be "what will layer `l+1` want, now",
which is milliseconds away. There was a mechanism to hope for, too: decode on this
checkpoint runs at ~7.1 GB per token in about 11 s, i.e. ~620 MB/s, which looks like
demand paging one fault at a time — and `madvise` over a whole expert range moves
bytes about five times faster than that. Turning thousands of small faults into a few
large reads should have been worth something.

It was implemented — a second transition matrix over the `l -> l+1` axis, threaded
down the layers of each forward pass — and measured the same way, three paired runs
with the order varied:

| | run 1 | run 2 | run 3 |
|---|---|---|---|
| decode, within-layer only | 15.06 s/token | 10.91 s/token | 10.86 s/token |
| decode, plus cross-layer | 12.25 s/token | 12.02 s/token | 11.61 s/token |
| ttft, within-layer only | 42.3 s | 33.3 s | 36.4 s |
| ttft, plus cross-layer | 58.6 s | 43.2 s | 33.8 s |

No decode improvement — the ranges overlap and the baseline is faster in two rounds
of three — and time-to-first-token got worse in two of three. So it was reverted
rather than kept on the strength of the argument.

The likely reason is that decode here is bandwidth-bound, not latency-bound.
Prefetching reorders I/O; it does not create bandwidth. Every wrong guess spends
bandwidth the real reads then have to wait for, and a router's choice at layer `l+1`
depends on layer `l`'s output — the very thing that just changed — so the guesses are
probably not good enough to pay for themselves. Recorded here so the next person with
this idea can skip building it.

## [0.20.0] - 2026-07-29

The larger-than-RAM claim is measured. It had been the one number in this project
quoted from arithmetic rather than a stopwatch, since demonstrating it needs a
checkpoint bigger than the host's memory — which turned out to be sitting on this
machine all along.

**Mixtral-8x7B Q4_K_M, 25 GB, on 16 GB of RAM**, `mmap`, prefetch off so this
isolates prefill order, alternating runs so neither side gets a systematically
warmer page cache. Time to first token:

| prompt | `prefill_batch = 1` (token-major) | `prefill_batch = 256` (batched) | |
|---|---|---|---|
| 20 tokens | 195.8 s | 48.6 s | 4.0× |
| 38 tokens | 386.4 s | 48.3 s / 51.8 s | 7.5–8.0× |

The ratio is the least interesting part. **Batched prefill is flat in prompt
length** — 48.6 s at 20 tokens, 48.3 s at 38 — because it reads the model once
either way. **Token-major doubles when the prompt does**, 195.8 s to 386.4 s,
because it re-reads every layer once per token. That is the working set argued for
in 0.14.0 (7.1 GB per token against 816 MB per layer), now visible in the shape of
the curve rather than inferred from it.

Nothing changed in the code for this release; what changed is that the
documentation no longer hedges. Every performance claim in the README has now been
measured on this machine.

### Documentation

- README, ABOUT and `LlamaBackend::with_prefill_chunk` carry the measurement instead
  of the argument.
- The remaining-work list is honest about what is left: prefill on a larger-than-RAM
  checkpoint is fast now, but a *single* sequence decoding still touches every layer
  per token, so a 25 GB model decodes at well under a token a second on this machine.
  Batching across concurrent requests amortises that; one lonely request cannot be
  helped without speculative decoding or keeping more of the model resident.

## [0.19.0] - 2026-07-29

### Changed

- **The prefill chunk is measured, not a constant.** 0.18.0 fed prompts in at a fixed
  32 tokens per scheduler iteration, which is imperceptible on a small model and
  seconds on a large one — the wrong shape for a knob that exists to bound a latency
  spike. The scheduler now tracks what a decode step costs and what a prefill token
  costs, both as exponential moving averages, and sizes each chunk so its work
  matches one decode step. A newly admitted request then gets roughly half the
  machine while the requests already streaming keep the other half, whatever the
  model.

  Same experiment as 0.18.0 — two clients streaming while a 1474-token prompt
  arrives, three runs:

  | | streamer worst gap | long-prompt request |
  |---|---|---|
  | fixed 32 | 641 / 456 / 636 ms | 14.32 / 14.78 / 15.20 s |
  | measured | 252 / 149 / 225 ms | 13.96 / 14.59 / 15.23 s |

  So ~2.6× less stall for the clients already streaming, and **no measurable cost**
  to the arriving request — the chunk lands around two or three tokens on this model,
  and the extra scheduling overhead is nothing next to a forward pass. On a model
  where those two costs are comparable the trade would show up; the bounds (1 to 512
  tokens) are what keep it from going anywhere silly.

### Tests

- The pacing arithmetic is pinned directly: a 30 ms step against 10 ms prefill tokens
  asks for three, a model ten times faster per token asks for thirty, and absurd
  ratios in either direction are clamped rather than believed. Also that one outlier
  sample moves the average part of the way rather than all of it.

## [0.18.0] - 2026-07-29

Chunked prefill. Continuous batching made one problem sharper: a newly admitted
request absorbed its whole prompt before the next decode step, so one long prompt
froze every client already streaming for as long as it took.

### Added

- **Prefill goes in a piece at a time, interleaved with decoding.** A request now
  enters a pending state and absorbs `PREFILL_TOKENS_PER_STEP` (32) prompt tokens per
  scheduler iteration while the decode batch keeps running, joining it once the
  prompt is in. With nothing else decoding the whole prompt goes in at once — there
  is no one to stall.

  Measured with two clients streaming while a 1474-token prompt arrives (that prompt
  alone takes 14.5 s to prefill), worst inter-token gap the streamers see, three runs:

  | | worst gap |
  |---|---|
  | before | 14.0 s / 13.2 s / 12.9 s |
  | after | 0.57 s / 0.52 s / 0.55 s |

  So roughly **24× less**, and what remains is about one chunk's worth of work, as
  expected. The constant trades that residual hitch against scheduling overhead and
  against how much a layer's weights are amortised inside the backend's own prefill
  batching; 32 keeps most of both.

- `InferenceRuntime::start_incremental` / `advance_prefill` / `finish_prefill`, the
  pieces that make a prompt interruptible. `start` is now written in terms of them,
  so the two paths cannot drift.

### Changed

- Prefill uses `hidden` rather than `logits`. The prefix's logits were always thrown
  away, so the output head was a `vocab x d_model` matmul computed for nothing.
- Cancellation and timeouts are checked between prefill chunks too, so a client that
  hangs up during a long prompt stops it instead of paying for the rest.

### Tests

- Absorbing a prompt three tokens at a time must build exactly the session absorbing
  it in one pass builds. The test uses two runtimes deliberately: sharing one would
  let the first run populate the prompt cache and the second hit it, testing nothing
  — which is what an assertion in the test caught when it was first written.

## [0.17.0] - 2026-07-29

Continuous batching. Concurrent requests now decode in one pass over the weights
instead of one pass each, which is what `logits_batch` was built for in 0.16.0.

### Changed

- **The scheduler drives one batch, not one task per request.** It admits up to
  `max_concurrent` sequences and steps them together, so N concurrent requests cost
  roughly one forward pass per token rather than N. Measured end to end through the
  HTTP API on a 620 MB MoE checkpoint, 24 tokens each, three runs:

  | concurrent | before | after |
  |---|---|---|
  | 1 | 22.3 tok/s | 23.8 tok/s |
  | 2 | 72.7 tok/s | 73.6 tok/s |
  | 4 | 23–83 tok/s | 65–113 tok/s |
  | 8 | 67–76 tok/s | 111–137 tok/s |

  So ~1.6–1.8× aggregate throughput at 8 concurrent, and median latency roughly
  halved (2.5 s → 1.6 s). The old path's numbers swing widely because N independent
  forward passes each fan out across rayon and then fight each other for the same
  cores; the batched loop fans out once. Note it *degraded* from 4 to 8 concurrent
  for that reason, where the batched loop keeps improving.

  With `max_concurrent = 1` the batch is always one sequence, i.e. exactly the old
  behaviour.

- **A queued request whose client has already gone is dropped immediately**, instead
  of waiting for a decode slot to free up first. Its user's concurrency permit lives
  inside the entry, so holding a dead request was holding a live caller's slot.

- `InferenceRuntime::next_token_batch` steps several sessions at once. They are
  independent — separate caches, samplers and budgets — so it produces exactly what
  stepping each alone produces. If the batched forward fails it retries the live
  sequences one at a time, so one bad sequence is not reported as everyone's failure.

- `InferenceBackend::logits_batch` takes `&mut [&mut SeqState]` rather than
  `&mut [SeqState]`. The runtime holds whole `Session`s, so borrowed caches are what
  it can actually hand over; the previous shape could not be called from the one
  place that needed it.

### Tests

- Concurrent requests decoding together must each get their own answer: a sequence
  decoded inside a batch has to match the same request decoded alone, and different
  prompts must not collapse to one output.
- The disconnect test no longer assumes a permit returns within a single yield. A
  `RateLimit` while the user already has `max_concurrent_per_user` in flight is the
  documented answer, and how fast a permit comes back is a timing detail — the bug it
  pins is never recovering, which it still asserts.

## [0.16.0] - 2026-07-29

Groundwork for batching the *decode* step across concurrent requests. A single
sequence decoding alone reads the whole model to produce one token and has no way to
amortise that; several sequences stepping together can share the pass.

### Added

- **`InferenceBackend::logits_batch`**, running several independent sequences in one
  call. It is defaulted — the default calls `logits` per sequence — so existing
  backends keep working and only one that can share work across the batch needs to
  override it. `LlamaBackend` does: only the attention read stays per sequence, while
  the projections, the router and the experts all see the batch at once.

  Measured on a 620 MB MoE checkpoint, 8 decode steps per sequence, best of three:

  | sequences | throughput | per token |
  |---|---|---|
  | 1 | 65.8 tok/s | 15.20 ms |
  | 2 | 65.7 tok/s | 15.22 ms |
  | 4 | 75.7 tok/s | 13.22 ms |
  | 8 | 103.8 tok/s | 9.64 ms |
  | 16 | 128.8 tok/s | 7.76 ms |

  The gain starts around four sequences and reaches ~2× at sixteen, which is what the
  routing predicts: with top-2 of 8 experts, a batch has to reach `n_experts/top_k`
  before an expert serves more than one token per pass. Below that there is nothing
  to share and the numbers say so.

- **Batched attention projections.** `wq`/`wk`/`wv`/`wo` now go through one matmul per
  layer instead of one matvec per token, which also speeds up the prefill path added
  in 0.14.0. Attention itself is unchanged and still strictly per token: it is causal,
  and each sequence reads its own cache.

- `SeqState::max_positions`, so a caller can check whether work fits without taking a
  mutable borrow of the cache it is asking about.

### Not yet wired

The scheduler still drives each request independently, so nothing calls `logits_batch`
outside its tests yet. Turning it on means restructuring the scheduler from one task
per request into a loop that collects ready sequences and steps them together, while
keeping the cancellation, timeout, priority and per-user guarantees that its tests
pin. That is deliberately a separate change: this one is verifiable on its own.

### Tests

- A batched decode step must equal decoding each sequence alone — different lengths
  and contents in the batch, so a crossed index would show.
- A ragged batch (sequences with different amounts to catch up on) falls back rather
  than producing something wrong.
- A refused batch — mismatched lengths, an out-of-vocabulary token — leaves every
  cache untouched instead of half-advanced.

## [0.15.0] - 2026-07-29

Every quantised type now has an integer kernel. Under `mmap`, no quantised checkpoint
takes the dequantise-to-`f32` path any more.

### Added

- **Integer kernels for `Q2_K`, `Q3_K` and `Q5_K`**, completing the set. Each dots the
  packed row straight against an int8-quantised activation without expanding it, the
  way `Q8_0`/`Q4_K`/`Q6_K` already did. Measured at Mixtral's own FFN row width
  (14336×4096), three runs, against dequantise-then-dot:

  | | speedup (3 runs) |
  |---|---|
  | `Q8_0` | 3.9× / 6.2× / 7.4× |
  | `Q2_K` | 5.6× / 5.5× / 4.5× |
  | `Q3_K` | 8.0× / 4.9× / 6.0× |
  | `Q4_K` | 7.7× / 9.8× / 8.0× |
  | `Q5_K` | 6.0× / 5.9× / 6.8× |
  | `Q6_K` | 15.5× / 10.2× / 12.3× |

  The spread is run-to-run noise on a laptop, not a property of the types; the honest
  claim is "several times faster each", not any single figure. p90 relative error
  against the `f32` reference stays under 0.034 for all six.

  `Q2_K` needed one thing the others did not: its scale-and-min pair covers 16 weights
  rather than 32, so its affine min-term sums the activation over runs of 16.
  `Q3_K`'s 6-bit signed scale unpacking is now shared with the dequantiser instead of
  duplicated — get that word-juggling wrong and every weight in the tensor is scaled
  by the wrong number.

### Changed

- The per-type `matvec_q8_0`/`matvec_q4_k`/`matvec_q6_k` wrappers are replaced by one
  `matvec_int8` plus an `int8_dot` dispatcher, so adding a format means writing its
  row kernel and nothing else. Both `matvec` and `matmul` route through it, so the
  new kernels benefit batched prefill as well.

### Tests

- **A unit-activation test.** With an all-ones activation the int8 quantisation is
  exact, so kernel and reference must agree to `f32` rounding — which pins the format
  unpacking on its own, separately from quantisation noise. This is what identified a
  3.5% gap on `Q5_K` as noise rather than a bug.
- **The quantised-activation tolerance is now derived, not chosen.** Each activation
  element is off by at most half its block's quantisation step, so the dot is off by
  at most `sum|w| * step/2`. Asserting that bound keeps the test tight enough to catch
  a misread format, instead of a percentage picked until it passed.

## [0.14.0] - 2026-07-28

Prefill was structured so that every prompt token paid for the whole model. It now
batches, and within a layer groups tokens by the expert they routed to — **~2x faster
prefill even on a checkpoint that fits entirely in RAM**, and it cuts the working set
from the whole model per token to a single layer, which is what a checkpoint larger
than RAM actually needs.

### Added

- **Batched, expert-grouped prefill (`model.prefill_batch`).** Prefill drove each
  token through all N layers before starting the next, so every layer's weights were
  re-read once per token. Two things changed: a chunk of tokens now goes through one
  layer before moving on, and within that layer the tokens are grouped by expert, so
  an expert's packed rows are decoded once for every token that chose it instead of
  once per token.

  Measured on a 620 MB MoE checkpoint resident in RAM, 128-token prefill, warm cache,
  best of three — the case with *nothing* to gain from the page cache, so this is the
  decode saving alone: mmapped 2.68 s → 1.34 s (2.0×), expanded to `f32` 2.12 s →
  1.28 s (1.7×). At 512 tokens, 10.6 s → 5.8 s.

  On top of that the prefill working set drops from the whole model per token —
  7.1 GB for Mixtral-8x7B Q4_K_M — to one layer with all eight experts, 816 MB. Above
  what the page cache holds, that is the difference between streaming the model off
  disk once per token and once per chunk. That second effect is reasoned from the
  working-set sizes rather than measured: demonstrating it needs a checkpoint larger
  than the host's RAM, which this machine could not do safely.

  Output is unchanged and tested that way — a batched prefill must equal feeding the
  same tokens one at a time, including across a chunk boundary. Every benchmark
  configuration above produced identical checksums.

- **`quant::matmul` and `simd::matmul`**, which decode a row once and dot it against
  `n` activation vectors. Tested to agree exactly with `n` separate `matvec` calls
  for all nine supported types.

### Changed

- **The expert prefetcher uses `madvise(MADV_WILLNEED)`** instead of reading one byte
  per page. Touching pages by hand faulted them in one at a time — a Mixtral-sized
  expert is ~99 MB, over six thousand separate faults at a 16 KB page size — and
  dragged every page through the CPU, evicting cache lines the forward pass was still
  using. Measured cold over 400 MB, twice with the order reversed: 217 ms / 232 ms for
  the advice against 1.12 s / 1.43 s for the byte loop, so **5-6× faster**. It is not
  asynchronous on macOS, which the comment now says rather than assumes.
- Attention is unchanged and still strictly token-by-token: it is causal, so a token's
  keys and values must be in the cache before the next one attends to them. Only the
  feed-forward is batched.
- An over-long prefill is refused before any layer runs, instead of failing partway
  and leaving the layers at different lengths.
- The single-token `block`/`feed_forward`/`expert` paths are gone rather than kept
  beside the batched ones: a batch of one is the same code, and two paths that are
  supposed to agree are two paths that can drift.

## [0.13.0] - 2026-07-28

### Added

- **A fused int8 kernel for `Q6_K`**, the second tensor type a `Q4_K_M` checkpoint is
  made of. Like the `Q4_K` kernel it dots the packed row straight against an
  int8-quantised activation, never expanding the row to `f32`: **9–10× faster** than
  dequantise-then-dot at Mixtral's own FFN row width (9.1 ms → 0.9–1.0 ms for a
  14336×4096 matvec; repeat runs measured 9.4× and 10.3×), p50 relative error 0.005.
  `Q6_K` is symmetric, so unlike `Q4_K` there is no per-block activation sum to carry
  — only the dot products.

  With `Q4_K` and `Q6_K` both on the integer path, almost every byte of a `Q4_K_M`
  file now avoids the `f32` round trip. `Q2_K`/`Q3_K`/`Q5_K` still take the slower
  path.

### Tests

- The `Q6_K` kernel is checked against the dequantise-then-dot reference, and the
  check calls the kernel directly as well as through `matvec` — comparing only via
  `matvec` would silently become a comparison of the reference against itself if the
  dispatch were ever dropped.
- A `Weight::Packed` row-addressing test over a k-quant. An F32 row is 4 bytes an
  element; a `Q6_K` row is 210 bytes per 256, so `row_start * row_bytes` does
  genuinely different arithmetic — and a stacked MoE expert tensor is addressed
  entirely through it, where an off-by-one would quietly read a neighbouring expert.

## [0.12.0] - 2026-07-28

A full re-read of the codebase against its own documentation. Most of what it turned
up was the same shape: a guarantee the README describes, upheld on the OpenAI routes
and quietly missing everywhere else.

### Fixed

- **`X-Garuda-User` now works on every protocol.** Each adapter hardcoded its own
  protocol name as the caller id — `"ollama"`, `"anthropic"`, `"llamacpp"`, `"tgi"`,
  `"ws"` — so every Ollama client on the machine shared a single
  `max_concurrent_per_user` bucket, and the ninth concurrent one got a `429` while
  the decode slots and the queue sat empty. Worse, it was silent: the README
  documented the header as a general fairness knob. All six front ends now resolve
  their caller through one `api::user_id`, falling back to the documented
  `anonymous` bucket.
- **Streamed token counts are the engine's, not a frame tally.** `eval_count`,
  `tokens_predicted`, `generated_tokens` and Anthropic's `output_tokens` were each
  incremented once per emitted frame. The streaming decoder holds back bytes that do
  not yet form a whole character, so frames and tokens do not correspond — on the
  test fixture, 26 frames for 32 tokens, a 19% undercount that grows with every
  multi-byte character in the reply. `session::Piece::Done` now carries the real
  count and every adapter reports it.
- **A tied output head no longer loads the embedding matrix twice.** A checkpoint
  that omits `output.weight` uses `token_embd.weight` as its head; the loader called
  the tensor reader again for it, so the non-mmap path held two independent `f32`
  copies of a `vocab × d_model` matrix — about 1 GB of pure duplication on a tied
  model with a 128k vocabulary. The two now share one `Arc`.
- **`tokenizer.ggml.add_bos_token` is honoured.** BOS was prepended unconditionally,
  so a checkpoint trained without one saw a token stream that did not match its
  training.
- **A NaN beside a finite maximum no longer survives `softmax`.** The existing guard
  caught an all-`-inf` distribution but not a single NaN among finite logits: the
  sum went NaN, the `sum > 0.0` check simply skipped normalising, and the NaNs
  reached the sampler's comparator. It now falls back to uniform, as the masked case
  already did.

### Added

- **WebSockets can authenticate from a browser.** The `WebSocket` constructor cannot
  set request headers, so once `server.api_keys` was set no page could reach
  `/v1/ws` at all. A key may now arrive as the `garuda.api-key.<key>` subprotocol,
  the convention the Kubernetes API server uses, and the handshake echoes it back so
  the browser accepts the connection. Deliberately not a query parameter: URLs end up
  in access logs, and this server logs every request URI.
- **A startup warning when the KV cache would thrash.** Spilling only pays off with
  `model.sliding_window` set. Under full attention every step reads the whole prefix,
  so `ensure_resident` pulls back everything the previous `append` spilled — the
  spill is undone and redone once per token, making disk I/O quadratic in sequence
  length. The default configuration never reaches it; a hand-lowered
  `kv_resident_blocks` does, and now says so.

### Changed

- **`/v1/embeddings` is bounded.** The endpoint does not go through the scheduler —
  there is no decode loop to drive — so it inherited none of the scheduler's
  protections: no input cap, no deadline, no cancellation, one blocking task running
  an arbitrarily long array serially while holding one of four slots. Requests are
  now capped at 256 inputs and give up their slot within one forward pass of
  `request_timeout`.
- **Sampling selects the top-k instead of sorting the vocabulary.** Every sampled
  token sorted all 32k candidates to keep 40 of them; it now partitions in one pass.
  The comparator breaks ties on token id, which is unique, so the ordering stays a
  strict total order and seeded output is bit-identical to before.
- **GGUF counts are bounded by what the remaining bytes could describe**, not by the
  raw file length. A descriptor costing 24 bytes on disk costs far more in memory, so
  a count that passed the old check could still drive an allocation orders of
  magnitude larger than the file — reserved before any bounds-checked read ran.

### Documentation

- Corrected claims that had gone stale as the code moved past them: `InferenceBackend`
  described `MoeEngine` as its only implementation two paragraphs above naming both;
  the GGUF reader said the k-quants were "still rejected"; the byte tokenizer said a
  GGUF-backed tokenizer "does not exist here", eight lines above `pub mod spm`;
  `weights` claimed a real checkpoint would replace it, when `LlamaBackend` bypasses
  it entirely; `api::user_id` said "there is no auth". README and this file said the
  chat page keeps its API key in `localStorage` — it uses `sessionStorage`, on
  purpose, so a credential does not outlive the tab the way saved conversations do.

## [0.11.0] - 2026-07-16

Multiple conversations in the built-in chat page — a sidebar to hold more than one
at a time, matching what every other chat UI already does.

### Added

- A conversation sidebar: "+ New chat", a switchable list of past conversations
  (title auto-derived from the first message), per-conversation delete, and a
  collapse toggle. Conversations persist in the browser's `localStorage` — a page
  reload restores the full list and whichever one was open.

### Fixed

- A conversation switch (or "New chat") while a reply was still streaming used to
  be able to save that reply onto whichever conversation the user switched *to*,
  since the in-flight request read and wrote the same shared, mutable
  `history`/`activeId` state that the switch itself reassigned out from under it.
  `send()` now captures its own conversation id and a local copy of the message
  list up front and closes over them for its whole lifetime, so a switch mid-stream
  can no longer cross-contaminate two conversations. Caught by a Playwright test
  that starts a reply, switches away before it finishes, and asserts neither
  conversation's saved messages mention the other's content.

Verified with Playwright against a live server: create/switch/delete conversations,
title derivation, empty "New chat" not littering the list with blanks, persistence
across a full page reload, the sidebar collapse toggle, and the mid-stream-switch
fix above — all against the real rendered page, not simulated.

## [0.10.0] - 2026-07-15

API key authentication — off by default, one config key away from on.

### Added

- **`server.api_keys`**: a list of shared secrets. When set, every request except
  `GET /health` and `GET /` (the chat page's static HTML) must present one, as
  `Authorization: Bearer <key>` or `x-api-key: <key>` — whichever a client's own
  ecosystem convention sends (OpenAI/llama.cpp/Ollama clients default to the
  former, Anthropic clients to the latter), so nothing about picking a wire
  protocol changes because auth is on. Keys compare in constant time
  (`constant_time_eq`), and a request with a present-but-empty key never matches
  even a misconfigured empty-string key — `AppConfig::validate` rejects those
  outright.
- The built-in chat page gained an API key field under Settings, stored in
  `sessionStorage`. A 401 opens Settings automatically and shows a clear error
  instead of hanging; the model badge shows "needs API key" until one is set.
- `auth::require_key`, an axum middleware wrapping the whole merged router (every
  protocol front end, not just the OpenAI-shaped one), so no adapter needed its own
  auth logic.

Verified: unit tests cover missing/wrong/correct keys, both header styles, multiple
configured keys, the exempt paths, and the empty-key edge case; a Playwright pass
against the real chat page confirms the full flow — blocked, prompted, unblocked,
and that the key survives a reload.

## [0.9.0] - 2026-07-15

A real MoE at scale, finally: Mixtral-8x7B (Q4_K_M, 26 GB) now loads and runs on a
16 GB machine, plus the integer-kernel and prefetch work that made it fast enough to
be worth doing.

### Added

- **The older per-expert GGUF tensor layout.** The Llama backend only recognised the
  merged `..._exps` tensor layout newer llama.cpp conversions use. Older ones —
  including the original TheBloke Mixtral quantisations — store each expert as its
  own tensor (`blk.0.ffn_gate.3.weight`); `ExpertWeight::Split` handles that layout
  now, alongside the existing `Stacked` one. Verified: a test builder emits both
  layouts from the same underlying numbers and asserts the logits match exactly.
- **An integer (NEON `i8`) kernel for `Q4_K`** (`quant::matvec_q4_k`), the same trick
  0.6.1 used for `Q8_0`: quantise the activation to int8 once per matvec, then dot
  each row's nibbles against it directly, without ever expanding to `f32`. 0.6.1
  guessed this would be a smaller win than `Q8_0` because of the nibble-unpack cost —
  measured at Mixtral's own row width (14336×4096) it's actually the bigger win,
  **~10× faster**, p90 relative error 2.9% against the exact `f32` path. `Q4_K` is
  the dominant tensor type in a real checkpoint (833 of ~1000 tensors in the Mixtral
  file), so this is the main matvec cost for a real model.
- **Prefetch against a real checkpoint**, not just the synthetic MoE.
  `GgufPagePrefetcher` "loads" an expert by touching its mmap pages on a background
  rayon worker instead of materialising an `Expert`, so the page fault happens ahead
  of the forward pass needing it. Each of a real model's 32 layers routes
  independently, so routing history moved from the flat `SeqState.last_experts` /
  `last_predicted` (fine for the synthetic MoE's single block) to per-layer fields on
  `KVCacheState`. Verified: attaching the engine doesn't change a single logit across
  a multi-step decode, and its launched/skipped counters prove it actually predicts
  and warms rather than sitting inert.
- **A built-in chat page** (`GET /`) — a single dependency-free HTML/JS page that
  streams against the existing `/v1/chat/completions` SSE endpoint, same origin, no
  separate frontend to build or deploy.
- `mixtral.toml` — an example config for exactly this scenario: a checkpoint far
  larger than RAM, `mmap = true`, one sequence at a time.

### Changed

- `garuda inspect` used to `std::fs::read` the whole file just to print its metadata
  — for a checkpoint larger than RAM, that alone could exhaust it. It mmaps now
  (peak RSS ~8 MB measured against the 26 GB Mixtral file), and its loadability check
  now verifies a MoE model's expert tensors actually exist under either layout,
  instead of only checking quant-type support.
- The crate-level doc comment said garuda "cannot load a trained checkpoint" and that
  this was "the gap between this and a usable runtime" — stale since the `llama`
  module was added. Corrected to describe both backends. The `Cargo.toml` package
  description had the same problem ("no trained weights") and is fixed too.

### Not done

- `Q2_K`/`Q3_K`/`Q5_K`/`Q6_K` still take the slower dequantise-to-`f32` path; the
  same int8-kernel trick likely applies, just not done yet.

Verified end to end against the real file: `garuda serve -c mixtral.toml` loads
Mixtral-8x7B Q4_K_M in ~20 ms (mmap, nothing to expand), `prefetch=true` in the log,
and generates real, coherent text — RSS stays around 6–7 GB, well under the 16 GB
machine's budget, the whole point of the mmap-streaming path this release finally
exercises against a model that actually needs it.

## [0.8.0] - 2026-07-14

Two more API front ends — llama.cpp and TGI — and a shared engine core so the
adapters stop reimplementing the same thing.

### Added

- **llama.cpp-compatible API** (`llamacpp` module): `POST /completion`, speaking
  `llama-server`'s shape — `n_predict`, a single `{"content": …}` object, and SSE
  `{"content": …, "stop": false}` frames ending in a `"stop": true` frame with
  `tokens_predicted` / `tokens_evaluated` / `stopped_eos`.
- **TGI-compatible API** (`tgi` module): `POST /generate` (`{"generated_text": …}`,
  optional `details`) and `POST /generate_stream` (per-token SSE `token` events; the
  terminal event carries `generated_text` and `details.finish_reason`).

### Changed

- **New `session` module — one engine-facing core shared by every front end.** The
  five adapters (`api`, `ollama`, `anthropic`, `llamacpp`, `tgi`) previously each
  reimplemented submit → collect-tokens → decode and the streaming decoder loop. That
  logic now lives once in `session` (`render_chat`, `submit`, `collect`, and a
  format-agnostic `pieces` stream of decoded text); each adapter is pure translation —
  parse the request, format the reply. No behaviour change: the 12 OpenAI integration
  tests and all endpoint shapes are unchanged, verified end to end.

All four HTTP front ends and the WebSocket path were re-verified live (streaming and
non-streaming) after the refactor.

- **Moved to the Rust 2024 edition; MSRV is now 1.85** (was edition 2021 / 1.82). The
  automated migration touched only test code (`gen` is a reserved word in 2024; an
  `expr` macro matcher pinned to `expr_2021`). The 2024 tail-expression drop-order
  change was reviewed at each streaming site and is behaviour-neutral here — the only
  side-effectful `Drop` is the request `Handle` (cancellation), which is a named local,
  not a reordered temporary. Verified: full suite and every endpoint pass on both 1.85
  and 1.97.

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

# Installing Garuda

Garuda is a single Rust binary with no system dependencies beyond a Rust
toolchain. Everything below is verified against a clean checkout.

For a quick orientation to what Garuda is (and its real limits — no GPU backend,
authentication off by default), read the [README](README.md) first.

## Requirements

| | |
|---|---|
| **Rust** | 1.85 or newer (2024 edition). Check with `rustc --version`. |
| **Platform** | Linux or macOS (x86-64 or ARM64). Portable Rust; no platform-specific code. |
| **Disk** | ~1 GB for the build (the `target/` directory). |
| **Network** | Only to fetch crates on first build, and to download a model if you want one. |

If you do not have Rust, install it with [rustup](https://rustup.rs):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Build from source

```bash
git clone https://github.com/hanumanit/garuda.git
cd garuda/garuda          # the crate lives in the garuda/ subdirectory
cargo build --release     # first build fetches crates and takes a few minutes
```

The binary is then at `target/release/garuda`. Verify it:

```bash
./target/release/garuda --version
./target/release/garuda --help
```

## Install onto your PATH (optional)

```bash
# From inside the garuda/ crate directory:
cargo install --path .
```

`cargo install` places `garuda` in `~/.cargo/bin` (already on your PATH if you
used rustup). After this you can run `garuda` from anywhere:

```bash
garuda --help
```

To uninstall: `cargo uninstall garuda`.

## Run it

Garuda starts in **synthetic mode** by default — a real engine over untrained
weights, so it serves requests but the generated text is meaningless. This
verifies the install end to end without downloading anything:

```bash
garuda serve                     # listens on 127.0.0.1:8080

# in another terminal:
curl -s localhost:8080/health
curl -s localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":16}'
```

### Run a real model

Download a checkpoint and point the config at it — F32, F16, Q4_0, Q8_0, and every
k-quant from `Q2_K` to `Q6_K` all load:

```bash
curl -L https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf \
  -o stories260K.gguf

# Inspect what you downloaded:
garuda inspect stories260K.gguf
```

Create a `config.toml` (or edit the one in the repo) with:

```toml
[model]
gguf = "stories260K.gguf"
context = 512
```

Then `garuda serve` and ask for a story:

```bash
curl -s localhost:8080/v1/completions \
  -H 'content-type: application/json' \
  -d '{"prompt":"Once upon a time","max_tokens":60,"temperature":0}'
```

Or open `http://localhost:8080/` for the built-in chat page instead of `curl`.

### Run a model larger than RAM

For a checkpoint bigger than the machine — a full-size Mixtral, say — set
`mmap = true`. The weights stay packed on disk and are dequantised a row at a time
instead of expanding to `f32`, so the process uses roughly the file's on-disk size.
[`garuda/mixtral.toml`](garuda/mixtral.toml) is a ready-made config for exactly
that: Mixtral-8x7B Instruct Q4_K_M (~26 GB) on a 16 GB machine, with a 2048-token
window, one sequence at a time, and a 900-second request timeout.

```bash
cd garuda        # the crate directory, where mixtral.toml lives

# ~26 GB
curl -L https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf \
  -o mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf

garuda inspect mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf
```

`model.gguf` in that file is a `/path/to/…` placeholder — edit it to point at the
file you just downloaded, then:

```bash
garuda --config mixtral.toml serve      # this config listens on 8090, not 8080

curl -s localhost:8090/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Name three prime numbers."}],"max_tokens":32}'
```

Be patient with it: on a 16 GB machine the first token takes 33–51 s and each one
after it 5–12 s, because every token is another pass over 26 GB of memory-mapped
weights. That is also why `sampling.max_tokens` is pinned to 48 there, and why a
benchmark against this config wants a few iterations rather than the default 64:
`garuda --config mixtral.toml benchmark --iterations 3 --tokens 8`.

### Run Qwen3.8-27B

Qwen3.8-27B is a **hybrid** model: three of every four blocks are gated delta nets
(linear attention with a fixed-size recurrent state) and the fourth is grouped-query
attention. Garuda loads it from the same GGUF path as anything else — the
architecture is read out of the file — and
[`garuda/qwen3.8.toml`](garuda/qwen3.8.toml) is a ready-made config for the Q4_K_M
quantisation (~19 GB) on a 16 GB machine.

```bash
cd garuda        # the crate directory, where qwen3.8.toml lives

# ~19 GB
curl -L https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-Q4_K_M.gguf \
  -o Qwen3.8-27B-Q4_K_M.gguf

# Prints the hybrid split and whether the file bundles prediction blocks
garuda inspect Qwen3.8-27B-Q4_K_M.gguf
```

Point `model.gguf` at the file you downloaded — it ships as a `/path/to/…`
placeholder — then:

```bash
garuda --config qwen3.8.toml serve      # this config listens on 8091

curl -s localhost:8091/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the capital of Thailand?"}],"max_tokens":40}'
```

Three things differ from a Llama-family checkpoint of this size:

- **A sequence carries a fixed ~144 MB of recurrent state**, the same at ten tokens
  as at a hundred thousand, on top of a KV cache that only the 16 attention blocks
  contribute to. If `memory.prompt_cache` is smaller than that state, no prompt can
  ever be cached, and startup says so.
- **`model.draft_gguf` is refused**, with a message explaining why: a recurrent state
  summarises every token it has read, so a rejected guess cannot be taken back out,
  and speculative decoding needs exactly that. Prompt lookup is off here too.
- **Its chat template opens a `<think>` block.** By default Garuda closes it
  immediately, so replies are answers. `model.thinking = true` leaves it open — the
  checkpoint's own default — and the reasoning then arrives as content ahead of the
  answer, so raise `sampling.max_tokens` to suit.

The vision half of this model is not supported: the image tower ships as a separate
`mmproj-*.gguf` and there is no image input path. Text in, text out.

To try the architecture without downloading 19 GB, `Qwen3.5-0.8B` is the same
arithmetic at 24 blocks:

```bash
curl -L https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf \
  -o Qwen3.5-0.8B-Q8_0.gguf
garuda inspect Qwen3.5-0.8B-Q8_0.gguf
```

## Configuration

`garuda serve` reads `config.toml` from the working directory if present, and uses
built-in defaults otherwise. Point it elsewhere with `--config path/to/file.toml`,
or override the address with `--host` / `--port`. Every key is documented inline in
[`garuda/config.toml`](garuda/config.toml); an unknown key is a startup error rather
than being silently ignored.

These three are top-level flags, so they go **before** the subcommand:

```bash
garuda --config mixtral.toml serve      # correct
garuda serve --config mixtral.toml      # error: unexpected argument '--config' found
```

Set the log level with the `GARUDA_LOG` environment variable (e.g.
`GARUDA_LOG=debug garuda serve`).

## Authentication

Off by default — anyone who can reach the port can use the server. Before exposing
it beyond `localhost`, set one or more keys:

```toml
[server]
api_keys = ["sk-change-me"]
```

Every request needs one then, sent as `Authorization: Bearer sk-change-me` or
`x-api-key: sk-change-me` — except `GET /health` (so a load balancer can probe it)
and `GET /` (the built-in chat page, which has its own API key field under Settings).

## Run the tests

```bash
cd garuda        # the crate directory
cargo test
```

The suite runs with no checkpoint on the machine. One file wants a real one and skips
without it — point it at any `qwen35` GGUF to run the Qwen3.5-family tests against
real weights:

```bash
GARUDA_QWEN35_GGUF=Qwen3.5-0.8B-Q8_0.gguf \
  cargo test --release --test qwen35_real -- --nocapture
```

## Troubleshooting

- **`error: package requires rustc 1.82`** — update your toolchain: `rustup update`.
- **`Address already in use`** — another process holds the port; pick another with
  `garuda --port 8081 serve` (the flag goes before the subcommand).
- **`error: unexpected argument '--config' found`** — `--config`, `--port` and
  `--host` belong to `garuda`, not to `serve`: write `garuda --config f.toml serve`.
- **`tensor type … is not supported`** — the model uses a format Garuda doesn't decode
  yet: the `*_1` linear quants or an IQ imatrix quant. F32, F16, Q4_0, Q8_0, and
  `Q2_K`–`Q6_K` all load.
- **`this checkpoint's MoE tensor layout is not recognised`** (from `garuda inspect`)
  — a mixture-of-experts model whose expert tensors are named differently from either
  layout Garuda knows: the merged `blk.N.ffn_gate_exps.weight` or the per-expert
  `blk.N.ffn_gate.0.weight` style.
- **`configuration error: … unknown field`** — a key in your `config.toml` is
  misspelled or unsupported; the message names it.
- **`architecture 'x' is not supported`** — Garuda loads `llama` and `qwen35`
  checkpoints. `garuda inspect <file>` names the architecture a file declares.
- **`architecture 'qwen35moe' is not supported`** — the mixture-of-experts sibling of
  Qwen3.5 (the `-A3B` models) is a different architecture and is refused rather than
  half-run. The dense checkpoints — Qwen3.8-27B, Qwen3.6-27B, Qwen3.5-0.8B…27B — load.
- **`model.draft_gguf is set … but this architecture cannot verify speculated
  tokens`** — a Qwen3.5-family checkpoint cannot speculate (see above). Clear the key.
- **`no prompt will ever be cached`** (a startup warning) — one sequence's recurrent
  state is larger than `memory.prompt_cache`. Raise the budget past the figure in the
  message, or accept that a repeated prompt re-runs prefill.

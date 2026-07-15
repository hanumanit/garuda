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

For a checkpoint larger than RAM — a full-size Mixtral, say — set `mmap = true`. The
weights stay packed on disk and are dequantised a row at a time instead of expanding
to `f32`, so the process uses roughly the file's on-disk size. See
[`garuda/mixtral.toml`](garuda/mixtral.toml) for a worked example.

## Configuration

`garuda serve` reads `config.toml` from the working directory if present, and uses
built-in defaults otherwise. Point it elsewhere with `--config path/to/file.toml`,
or override the address with `--host` / `--port`. Every key is documented inline in
[`garuda/config.toml`](garuda/config.toml); an unknown key is a startup error rather
than being silently ignored.

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

## Troubleshooting

- **`error: package requires rustc 1.82`** — update your toolchain: `rustup update`.
- **`Address already in use`** — another process holds the port; pick another with
  `garuda serve --port 8090`.
- **`tensor type … is not supported`** — the model uses a format Garuda doesn't decode
  yet: the `*_1` linear quants or an IQ imatrix quant. F32, F16, Q4_0, Q8_0, and
  `Q2_K`–`Q6_K` all load.
- **`this checkpoint's MoE tensor layout is not recognised`** (from `garuda inspect`)
  — a mixture-of-experts model whose expert tensors are named differently from either
  layout Garuda knows: the merged `blk.N.ffn_gate_exps.weight` or the per-expert
  `blk.N.ffn_gate.0.weight` style.
- **`configuration error: … unknown field`** — a key in your `config.toml` is
  misspelled or unsupported; the message names it.

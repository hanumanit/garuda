# Installing Garuda

Garuda is a single Rust binary with no system dependencies beyond a Rust
toolchain. Everything below is verified against a clean checkout.

For a quick orientation to what Garuda is (and its one real limit — only F32/F16
checkpoints load), read the [README](README.md) first.

## Requirements

| | |
|---|---|
| **Rust** | 1.82 or newer (2021 edition). Check with `rustc --version`. |
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

Download a small F32 checkpoint and point the config at it:

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

> Only F32/F16 checkpoints load. Quantised models (`Q4_K`, `Q8_0`, …) are rejected
> with a clear error — the dequantiser is not written yet. See the README.

## Configuration

`garuda serve` reads `config.toml` from the working directory if present, and uses
built-in defaults otherwise. Point it elsewhere with `--config path/to/file.toml`,
or override the address with `--host` / `--port`. Every key is documented inline in
[`garuda/config.toml`](garuda/config.toml); an unknown key is a startup error rather
than being silently ignored.

Set the log level with the `GARUDA_LOG` environment variable (e.g.
`GARUDA_LOG=debug garuda serve`).

## Run the tests

```bash
cd garuda        # the crate directory
cargo test
```

## Troubleshooting

- **`error: package requires rustc 1.82`** — update your toolchain: `rustup update`.
- **`Address already in use`** — another process holds the port; pick another with
  `garuda serve --port 8090`.
- **`gguf: tensor '…' has ggml type N; only F32 and F16 are supported`** — the model
  is quantised. Use an F32/F16 checkpoint, or wait for a dequantiser.
- **`configuration error: … unknown field`** — a key in your `config.toml` is
  misspelled or unsupported; the message names it.

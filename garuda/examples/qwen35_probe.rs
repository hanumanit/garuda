//! Probe a `qwen35` checkpoint: one forward pass, the most likely next tokens.
//!
//! One pass over a 19 GB checkpoint costs a minute or two, so this exists to ask a
//! question that a single pass answers — "what does the model think comes next?" —
//! rather than waiting for a dozen decode steps to find out.
//!
//! ```bash
//! cargo run --release --example qwen35_probe -- <model.gguf> "The capital of France is"
//! ```

use garuda::cache::{KvConfig, SeqState};
use garuda::core::InferenceBackend;
use garuda::gguf::Gguf;
use garuda::qwen35::Qwen35Backend;
use garuda::tokenizer::{Tokenize, bpe::BpeTokenizer};
use std::sync::Arc;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(|| {
        eprintln!("usage: qwen35_probe <model.gguf> [prompt] [tokens]");
        std::process::exit(2);
    });
    let prompt = args
        .next()
        .unwrap_or_else(|| "The capital of France is".to_string());
    let steps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1);

    let file = std::fs::File::open(&path)?;
    // Safety: read-only, held for the process lifetime, never mutated.
    let map = Arc::new(unsafe { memmap2::Mmap::map(&file) }?);
    let gguf = Gguf::parse(&map)?;
    let tk = BpeTokenizer::from_gguf(&gguf)?;
    // GARUDA_PIN=9GB holds that much of the checkpoint in buffers this process owns,
    // instead of leaving every byte to the page cache.
    let pin = std::env::var("GARUDA_PIN")
        .ok()
        .and_then(|v| garuda::config::parse_size(&v).ok())
        .unwrap_or(0);
    if pin > 0 {
        println!("pinning up to {} MB", pin / 1_048_576);
    }
    let backend = Qwen35Backend::from_gguf_pinned(&gguf, &map, Some(map.clone()), pin)?
        .with_prefill_chunk(64);
    if pin > 0 {
        println!("pinned {} MB", backend.pinned_bytes() / 1_048_576);
    }

    // GARUDA_PREFETCH=1 warms each block while the previous one computes. The point of
    // having it here is the A/B: same process, same file, one flag apart.
    let workers: usize = std::env::var("GARUDA_PREFETCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let backend = if workers > 0 {
        let pf = Arc::new(garuda::prefetch::LayerPrefetcher::with_workers(
            map.clone(),
            backend.layer_spans().to_vec(),
            workers,
        ));
        println!(
            "prefetch on: {workers} workers over {} blocks",
            backend.layer_spans().len()
        );
        backend.with_prefetch(pf)
    } else {
        backend
    };
    let cfg = backend.config();

    let mut ctx = tk.encode(&prompt);
    println!("prompt {prompt:?} -> {} tokens {:?}", ctx.len(), ctx);

    let mut seq = SeqState::new(
        KvConfig {
            dims: backend.dims(),
            kv_dim: cfg.kv_dim(),
            n_layers: cfg.n_layers,
            kv_dims: Some(cfg.kv_dims()),
            max_positions: ctx.len() + steps + 8,
            max_resident_blocks: 4096,
            sliding_window: None,
            storage: None,
        },
        1,
    );

    for step in 0..steps {
        let start = Instant::now();
        let logits = backend.logits(&ctx, &mut seq)?;
        let mut ranked: Vec<(usize, f32)> = logits.data().iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        println!("step {step} in {:.1}s", start.elapsed().as_secs_f32());
        for (id, score) in ranked.iter().take(5) {
            let text = tk.decode(&[*id as u32]).unwrap_or_default();
            println!("  {score:8.3}  {id:>7}  {text:?}");
        }
        ctx.push(ranked[0].0 as u32);
    }
    println!(
        "continuation: {:?}",
        tk.decode(&ctx[ctx.len() - steps..]).unwrap_or_default()
    );
    Ok(())
}

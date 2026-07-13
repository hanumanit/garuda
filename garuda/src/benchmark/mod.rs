//! Microbenchmarks.
//!
//! Every number printed here is measured. The previous version printed
//! `Cache Hit  >95%  100.0%  PASS` as a string literal — it never looked at the
//! cache. Where a figure cannot be measured, it is not printed.
//!
//! Throughput figures describe *this* model: a one-block, 128-dim MoE with
//! untrained weights. They say something about the runtime's overhead and nothing
//! at all about how a real 8x7B checkpoint would perform.

use crate::config::AppConfig;
use crate::core::ExpertLoader;
use crate::runtime::{SamplingParams, StopReason};
use crate::server::{Backend, Engine};
use std::time::{Duration, Instant};

pub fn run(config: &AppConfig, iterations: usize, tokens_per_iter: usize) -> anyhow::Result<()> {
    let iterations = iterations.max(1);
    let tokens_per_iter = tokens_per_iter.max(1);

    println!("=== Garuda benchmark ===");

    let t0 = Instant::now();
    let engine = Engine::build(config)?;
    let startup = t0.elapsed();

    match &engine.backend {
        Backend::SyntheticMoe => println!(
            "model: synthetic MoE, {} experts, top-{}, d_model {} (untrained weights)",
            config.model.experts, config.model.top_k, engine.dims.d_model,
        ),
        Backend::Gguf { path, layers } => println!(
            "model: {path} ({layers} layers, d_model {}, vocab {})",
            engine.dims.d_model, engine.dims.vocab_size,
        ),
    }
    println!();
    println!("startup                {startup:>12.2?}");

    // The tiered expert cache only exists for the synthetic MoE.
    if let Some(memory) = &engine.memory {
        // Cold: nothing in L1 and no expert file yet — synthesis plus a write to L2.
        let t = Instant::now();
        memory.load(0)?;
        let cold = t.elapsed();

        let t = Instant::now();
        memory.load(0)?;
        let warm = t.elapsed();

        // Evict it, then measure the real L2 read path.
        memory.unload(0);
        let t = Instant::now();
        memory.load(0)?;
        let from_l2 = t.elapsed();

        println!("expert load (cold)     {cold:>12.2?}");
        println!("expert load (L2 read)  {from_l2:>12.2?}");
        println!("expert load (L1 hit)   {warm:>12.2?}");
    }
    println!();

    let params = SamplingParams {
        max_tokens: tokens_per_iter,
        seed: Some(0xBEEF),
        ..config.sampling()?
    };

    let prompt = engine
        .runtime
        .tokenizer
        .encode("Garuda is an inference runtime with tiered expert storage.");

    // Warm the pipeline so the first iteration's expert synthesis does not skew the
    // decode numbers.
    {
        let mut s = engine.runtime.start(&prompt, &params)?;
        while engine.runtime.next_token(&mut s, &params).is_ok() {}
    }

    let mut total = Duration::ZERO;
    let mut generated = 0usize;
    let mut first_token = Duration::ZERO;
    let mut stops = (0usize, 0usize);

    for _ in 0..iterations {
        let start = Instant::now();
        let mut session = engine.runtime.start(&prompt, &params)?;

        let mut n = 0;
        loop {
            let step = Instant::now();
            match engine.runtime.next_token(&mut session, &params) {
                Ok(_) => {
                    if n == 0 {
                        first_token += step.elapsed();
                    }
                    n += 1;
                }
                Err(StopReason::Eos) => {
                    stops.0 += 1;
                    break;
                }
                Err(_) => {
                    stops.1 += 1;
                    break;
                }
            }
        }
        total += start.elapsed();
        generated += n;
    }

    let per_token = if generated > 0 {
        total / generated as u32
    } else {
        Duration::ZERO
    };
    let tok_per_sec = generated as f64 / total.as_secs_f64();

    println!("prompt                 {:>12} tokens", prompt.len());
    println!("generated              {generated:>12} tokens over {iterations} runs");
    println!(
        "time to first token    {:>12.2?}",
        first_token / iterations as u32
    );
    println!("per-token latency      {per_token:>12.2?}");
    println!("throughput             {tok_per_sec:>12.1} tok/s");
    println!(
        "stopped on eos/length  {:>12}",
        format!("{} / {}", stops.0, stops.1)
    );
    println!();

    if let Some(memory) = &engine.memory {
        let l1 = memory.l1_stats();
        let tiers = memory.tier_counts();
        println!("--- expert cache (measured) ---");
        println!("l1 hit ratio           {:>12.1}%", l1.hit_ratio() * 100.0);
        println!(
            "l1 hits / misses       {:>12}",
            format!("{} / {}", l1.hits, l1.misses)
        );
        println!("l1 evictions           {:>12}", l1.evictions);
        println!(
            "l1 resident            {:>12}",
            format!("{} experts, {} KiB", l1.entries, l1.bytes / 1024)
        );
        println!(
            "loads by tier          {:>12}",
            format!(
                "l1 {} / l2 {} / l3 {} / synth {}",
                tiers.l1, tiers.l2, tiers.l3, tiers.synthesised
            )
        );
    }

    let prompt_cache = engine.runtime.prompt_cache_stats();
    println!(
        "prompt cache           {:>12}",
        format!(
            "{:.0}% hit ({} / {})",
            prompt_cache.hit_ratio() * 100.0,
            prompt_cache.hits,
            prompt_cache.hits + prompt_cache.misses
        )
    );

    if let Some(pf) = &engine.prefetch {
        let s = pf.predictor_stats();
        println!();
        println!("--- prefetch (measured) ---");
        println!(
            "launched / skipped     {:>12}",
            format!("{} / {}", pf.launched(), pf.skipped())
        );
        if s.correct + s.wasted > 0 {
            println!("precision              {:>12.1}%", s.precision() * 100.0);
            println!("recall                 {:>12.1}%", s.recall() * 100.0);
        } else {
            println!("precision              {:>12}", "n/a (no predictions made)");
        }
    } else {
        println!("prefetch               {:>12}", "disabled");
    }

    Ok(())
}

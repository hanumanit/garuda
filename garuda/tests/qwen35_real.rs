//! Runs a real Qwen3.5-family checkpoint, when one is on hand.
//!
//! Set `GARUDA_QWEN35_GGUF` to a `qwen35` GGUF file and these run; without it they
//! skip, so the suite stays runnable on a machine with no 20 GB checkpoint. The
//! synthetic tests in `qwen35::tests` pin the arithmetic and the shapes; only a real
//! checkpoint can show the two things they cannot — that the file's own vocabulary is
//! being applied the way it was trained, and that the result is language.
//!
//! ```bash
//! GARUDA_QWEN35_GGUF=~/models/qwen3.8-27b-Q4_K_M.gguf \
//!   cargo test --release --test qwen35_real -- --nocapture
//! ```

use garuda::cache::{KvConfig, SeqState};
use garuda::core::{InferenceBackend, Token};
use garuda::gguf::Gguf;
use garuda::qwen35::Qwen35Backend;
use garuda::tokenizer::{Tokenize, bpe::BpeTokenizer};
use std::sync::Arc;

struct Loaded {
    backend: Qwen35Backend,
    tokenizer: BpeTokenizer,
    _map: Arc<memmap2::Mmap>,
}

fn load() -> Option<Loaded> {
    let path = std::env::var("GARUDA_QWEN35_GGUF").ok()?;
    let file = std::fs::File::open(&path).expect("opening the checkpoint named by the env var");
    // Safety: read-only, held for the test's lifetime, never mutated.
    let map = Arc::new(unsafe { memmap2::Mmap::map(&file) }.expect("mmapping the checkpoint"));
    let gguf = Gguf::parse(&map).expect("parsing the checkpoint");
    let tokenizer = BpeTokenizer::from_gguf(&gguf).expect("loading the vocabulary");
    let backend = Qwen35Backend::from_gguf(&gguf, &map, Some(map.clone()))
        .expect("loading the checkpoint")
        .with_prefill_chunk(64);
    Some(Loaded {
        backend,
        tokenizer,
        _map: map,
    })
}

fn seq_for(b: &Qwen35Backend, max_positions: usize) -> SeqState {
    let cfg = b.config();
    SeqState::new(
        KvConfig {
            dims: b.dims(),
            kv_dim: cfg.kv_dim(),
            n_layers: cfg.n_layers,
            kv_dims: Some(cfg.kv_dims()),
            max_positions,
            max_resident_blocks: max_positions,
            sliding_window: None,
            storage: None,
        },
        1,
    )
}

/// Greedy decode, which is the strongest signal available without a second
/// implementation to compare against: every step picks the most likely token, so a
/// sign error anywhere in the delta net or the rotation shows up as word salad.
fn greedy(l: &Loaded, prompt: &str, n: usize) -> String {
    let mut ctx: Vec<Token> = l.tokenizer.encode(prompt);
    assert!(!ctx.is_empty(), "the prompt tokenised to nothing");
    let mut seq = seq_for(&l.backend, ctx.len() + n + 8);
    let mut out = Vec::new();
    for _ in 0..n {
        let logits = l.backend.logits(&ctx, &mut seq).expect("a decode step");
        let (best, _) =
            logits
                .data()
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |acc, (i, &v)| {
                    if v > acc.1 { (i, v) } else { acc }
                });
        let tok = best as Token;
        if tok == l.tokenizer.eos() {
            break;
        }
        ctx.push(tok);
        out.push(tok);
    }
    l.tokenizer.decode(&out).expect("decoding the reply")
}

#[test]
fn a_real_checkpoint_reports_the_shape_the_file_declares() {
    let Some(l) = load() else {
        eprintln!("GARUDA_QWEN35_GGUF is not set; skipping");
        return;
    };
    let cfg = l.backend.config();
    let recurrent = cfg.recurrent.iter().filter(|&&r| r).count();

    eprintln!(
        "blocks {} ({recurrent} recurrent, {} attention), d_model {}, heads {}x{} (kv {}), \
         ff {}, vocab {}, context {}, rope base {}, n_rot {}, delta net {}x{} k / {}x{} v, \
         conv kernel {}, recurrent state {} MB",
        cfg.n_layers,
        cfg.n_layers - recurrent,
        cfg.d_model,
        cfg.n_heads,
        cfg.head_dim,
        cfg.n_kv_heads,
        cfg.d_ff,
        cfg.vocab,
        cfg.context,
        cfg.rope_theta,
        cfg.n_rot,
        cfg.n_k_heads,
        cfg.key_head_dim,
        cfg.n_v_heads,
        cfg.value_head_dim,
        cfg.conv_kernel,
        cfg.linear_state_bytes() / 1_048_576,
    );

    assert!(cfg.n_layers > 0 && cfg.d_model > 0);
    assert_eq!(cfg.recurrent.len(), cfg.n_layers);
    assert!(recurrent > 0, "a qwen35 checkpoint has recurrent blocks");
    assert_eq!(cfg.vocab, l.tokenizer.vocab_size());
    l.backend.dims().validate().unwrap();
}

/// The vocabulary, applied the way the checkpoint was trained: round-trips, one token
/// for a common word with its leading space, one token per digit.
#[test]
fn the_checkpoints_own_vocabulary_round_trips() {
    let Some(l) = load() else {
        eprintln!("GARUDA_QWEN35_GGUF is not set; skipping");
        return;
    };
    let tk = &l.tokenizer;

    for text in [
        "The capital of France is Paris.",
        "def add(a, b):\n    return a + b\n",
        "ราชอาณาจักรไทย มีกรุงเทพมหานครเป็นเมืองหลวง",
        "混合专家模型",
        "emoji: 🐦‍🔥 and a tab\tand  double  spaces",
    ] {
        let ids = tk.encode(text);
        let back = tk.decode(&ids).unwrap();
        assert_eq!(back, text, "round trip failed for {text:?} via {ids:?}");
    }

    // Digits are split one at a time by the Qwen pre-tokenizer, and a word carries its
    // leading space. Both are properties of the split, not of the merge list, so they
    // pin the part this crate implements by hand.
    assert_eq!(tk.encode("1234").len(), 4);
    assert_eq!(tk.encode(" world").len(), 1);
    assert!(tk.token_id("<|im_end|>").is_some());
    assert!(
        !tk.encode("<|im_end|>")
            .contains(&tk.token_id("<|im_end|>").unwrap()),
        "text that looks like a control token must not become one"
    );

    // Streaming must agree with the batch decode, including mid-character splits.
    let ids = tk.encode("ราชอาณาจักรไทย 🐦‍🔥");
    let mut dec = tk.stream_decoder();
    let streamed: String = ids.iter().map(|&t| dec.push(t)).collect::<String>() + &dec.finish();
    assert_eq!(streamed, tk.decode(&ids).unwrap());
}

/// Greedy continuation of a prompt whose answer is not in doubt.
#[test]
fn a_real_checkpoint_continues_a_prompt_with_language() {
    let Some(l) = load() else {
        eprintln!("GARUDA_QWEN35_GGUF is not set; skipping");
        return;
    };

    let reply = greedy(&l, "The capital of France is", 12);
    eprintln!("completion: {reply:?}");
    assert!(
        reply.contains("Paris"),
        "expected the obvious continuation, got {reply:?}"
    );
}

/// The same prompt through the chat template the checkpoint names, which is the path a
/// served request takes.
#[test]
fn a_real_checkpoint_answers_a_chat_turn() {
    let Some(l) = load() else {
        eprintln!("GARUDA_QWEN35_GGUF is not set; skipping");
        return;
    };
    let file = std::env::var("GARUDA_QWEN35_GGUF").unwrap();
    let bytes = std::fs::read(&file).ok();
    let template = bytes.as_ref().and_then(|b| {
        Gguf::parse(b).ok().and_then(|g| {
            g.get("tokenizer.chat_template")
                .and_then(garuda::gguf::Value::as_str)
                .map(str::to_owned)
        })
    });
    let fmt = garuda::chat::ChatFormat::detect(template.as_deref());
    eprintln!("chat format: {}", fmt.as_str());
    assert_eq!(fmt.as_str(), "qwen3.5 (thinking off)");

    let prompt = garuda::chat::encode_chat(
        fmt,
        &l.tokenizer,
        [("user", "What is the capital of France? Answer in one word.")],
    );
    let mut seq = seq_for(&l.backend, prompt.len() + 24);
    let mut ctx = prompt;
    let mut out = Vec::new();
    for _ in 0..16 {
        let logits = l.backend.logits(&ctx, &mut seq).unwrap();
        let (best, _) =
            logits
                .data()
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |acc, (i, &v)| {
                    if v > acc.1 { (i, v) } else { acc }
                });
        let tok = best as Token;
        if tok == l.tokenizer.eos() || Some(tok) == l.tokenizer.token_id("<|im_end|>") {
            break;
        }
        ctx.push(tok);
        out.push(tok);
    }
    let reply = l.tokenizer.decode(&out).unwrap();
    eprintln!("chat reply: {reply:?}");
    assert!(
        reply.contains("Paris"),
        "expected Paris in a one-word answer, got {reply:?}"
    );
}

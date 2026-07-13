//! Dump a GGUF file's metadata and tensor list. Scratch tool for understanding a
//! checkpoint's architecture.
//!
//!   cargo run --example gguf_dump -- path/to/model.gguf

use garuda::gguf::{Gguf, Value};

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: gguf_dump <file.gguf>");
    let bytes = std::fs::read(&path)?;
    let g = Gguf::parse(&bytes)?;

    println!("== metadata ==");
    for (k, v) in &g.metadata {
        println!("{k:40} = {}", short(v));
    }

    if let (Some(Value::Array(toks)), Some(Value::Array(scores)), Some(Value::Array(types))) = (
        g.metadata.get("tokenizer.ggml.tokens"),
        g.metadata.get("tokenizer.ggml.scores"),
        g.metadata.get("tokenizer.ggml.token_type"),
    ) {
        println!("\n== token sample ==");
        for i in [0usize, 1, 2, 3, 4, 259, 260, 270, 300, 400, 500] {
            if i < toks.len() {
                println!(
                    "  [{i:3}] {:14} score={:8} type={}",
                    one(&toks[i]),
                    one(scores.get(i).unwrap_or(&Value::F32(0.0))),
                    one(types.get(i).unwrap_or(&Value::I32(0)))
                );
            }
        }
    }

    println!("\n== tensors ({}) ==", g.tensors.len());
    for t in g.tensors.iter().take(3) {
        println!(
            "{:40} dims={:?} type={} offset={}",
            t.name, t.dims, t.ggml_type, t.offset
        );
    }
    Ok(())
}

fn short(v: &Value) -> String {
    match v {
        Value::Array(a) => {
            let head: Vec<String> = a.iter().take(6).map(one).collect();
            format!(
                "[{} items] {}{}",
                a.len(),
                head.join(", "),
                if a.len() > 6 { ", …" } else { "" }
            )
        }
        other => one(other),
    }
}

fn one(v: &Value) -> String {
    match v {
        Value::U8(x) => x.to_string(),
        Value::I8(x) => x.to_string(),
        Value::U16(x) => x.to_string(),
        Value::I16(x) => x.to_string(),
        Value::U32(x) => x.to_string(),
        Value::I32(x) => x.to_string(),
        Value::U64(x) => x.to_string(),
        Value::I64(x) => x.to_string(),
        Value::F32(x) => x.to_string(),
        Value::F64(x) => x.to_string(),
        Value::Bool(x) => x.to_string(),
        Value::String(s) => format!("{s:?}"),
        Value::Array(_) => "[array]".into(),
    }
}

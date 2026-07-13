//! SentencePiece (llama-style) tokenizer loaded from a GGUF vocabulary.
//!
//! Encoding follows llama.cpp's SPM: every input character starts as its own
//! symbol, then the adjacent pair whose concatenation is the highest-scoring token
//! is merged, repeatedly, until nothing merges. Symbols that never became a known
//! token fall back to their raw bytes (`<0x41>` …). This is the resegmentation the
//! model was trained with, so the token stream matches — which is what makes the
//! output coherent rather than noise.

use crate::core::{GarudaError, Token};
use crate::gguf::{Gguf, Value};
use crate::tokenizer::{StreamDecode, Tokenize};
use std::collections::HashMap;
use std::sync::Arc;

// llama token types.
const TYPE_CONTROL: i64 = 3;
const TYPE_BYTE: i64 = 6;

/// The U+2581 "lower one eighth block" SentencePiece uses to mark a space.
const SPACE_MARK: char = '\u{2581}';

/// The decode-side tables, shared with each stream decoder.
struct Vocab {
    tokens: Vec<String>,
    /// `Some(byte)` for the 256 byte-fallback tokens, `None` otherwise.
    byte_value: Vec<Option<u8>>,
    is_control: Vec<bool>,
    bos: Token,
    eos: Token,
}

impl Vocab {
    /// Append one token's raw bytes: byte-fallback tokens yield their byte, normal
    /// tokens yield their UTF-8 with the space marker turned back into a space.
    fn token_bytes(&self, id: Token, out: &mut Vec<u8>) {
        let i = id as usize;
        if i >= self.tokens.len() || self.is_control[i] || id == self.bos || id == self.eos {
            return;
        }
        if let Some(b) = self.byte_value[i] {
            out.push(b);
            return;
        }
        for ch in self.tokens[i].chars() {
            if ch == SPACE_MARK {
                out.push(b' ');
            } else {
                let mut buf = [0u8; 4];
                out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
            }
        }
    }
}

pub struct SpmTokenizer {
    vocab: Arc<Vocab>,
    scores: Vec<f32>,
    lookup: HashMap<String, Token>,
    byte_token: [Token; 256],
    unk: Token,
    add_bos: bool,
}

impl SpmTokenizer {
    /// Load the vocabulary from a parsed GGUF file.
    pub fn from_gguf(g: &Gguf) -> Result<Self, GarudaError> {
        let model = g
            .get("tokenizer.ggml.model")
            .and_then(Value::as_str)
            .unwrap_or("");
        if model != "llama" {
            return Err(GarudaError::Model(format!(
                "tokenizer model '{model}' is not supported (only llama SPM)"
            )));
        }

        let tok_arr = g
            .get("tokenizer.ggml.tokens")
            .and_then(Value::as_array)
            .ok_or_else(|| GarudaError::Model("gguf has no tokenizer.ggml.tokens".into()))?;
        let score_arr = g.get("tokenizer.ggml.scores").and_then(Value::as_array);
        let type_arr = g.get("tokenizer.ggml.token_type").and_then(Value::as_array);

        let n = tok_arr.len();
        let mut tokens = Vec::with_capacity(n);
        let mut scores = Vec::with_capacity(n);
        let mut byte_value = Vec::with_capacity(n);
        let mut is_control = Vec::with_capacity(n);
        let mut lookup = HashMap::with_capacity(n);
        let mut byte_token = [0 as Token; 256];

        for (i, tv) in tok_arr.iter().enumerate() {
            let s = tv
                .as_str()
                .ok_or_else(|| GarudaError::Model("a token is not a string".into()))?
                .to_owned();
            let score = score_arr
                .and_then(|a| a.get(i))
                .and_then(Value::as_f32)
                .unwrap_or(0.0);
            let ttype = type_arr
                .and_then(|a| a.get(i))
                .and_then(Value::as_u64)
                .map(|v| v as i64)
                .unwrap_or(1);

            let bval = if ttype == TYPE_BYTE {
                parse_byte_token(&s)
            } else {
                None
            };
            if let Some(b) = bval {
                byte_token[b as usize] = i as Token;
            }

            lookup.entry(s.clone()).or_insert(i as Token);
            tokens.push(s);
            scores.push(score);
            byte_value.push(bval);
            is_control.push(ttype == TYPE_CONTROL);
        }

        let id = |key: &str, default: Token| {
            g.get(key)
                .and_then(Value::as_u64)
                .map(|v| v as Token)
                .unwrap_or(default)
        };

        Ok(Self {
            vocab: Arc::new(Vocab {
                tokens,
                byte_value,
                is_control,
                bos: id("tokenizer.ggml.bos_token_id", 1),
                eos: id("tokenizer.ggml.eos_token_id", 2),
            }),
            scores,
            lookup,
            byte_token,
            unk: id("tokenizer.ggml.unknown_token_id", 0),
            add_bos: true,
        })
    }

    pub fn bos(&self) -> Token {
        self.vocab.bos
    }

    fn byte_to_token(&self, b: u8) -> Token {
        let t = self.byte_token[b as usize];
        if t == 0 {
            self.unk
        } else {
            t
        }
    }

    /// SPM encode: prepend a space marker, then merge greedily by score.
    fn encode_spm(&self, text: &str) -> Vec<Token> {
        let mut normalized = String::with_capacity(text.len() + 3);
        normalized.push(SPACE_MARK);
        for ch in text.chars() {
            if ch == ' ' {
                normalized.push(SPACE_MARK);
            } else {
                normalized.push(ch);
            }
        }

        // One symbol per character, threaded as a doubly linked list over `syms`.
        let mut syms: Vec<Sym> = Vec::new();
        for (start, ch) in normalized.char_indices() {
            let idx = syms.len() as i32;
            syms.push(Sym {
                start,
                len: ch.len_utf8(),
                prev: idx - 1,
                next: idx + 1,
            });
        }
        if let Some(last) = syms.last_mut() {
            last.next = -1;
        }

        let mut heap: std::collections::BinaryHeap<Bigram> = std::collections::BinaryHeap::new();
        for i in 1..syms.len() as i32 {
            self.try_bigram(&normalized, &syms, i - 1, i, &mut heap);
        }

        while let Some(bg) = heap.pop() {
            let (l, r) = (bg.left as usize, bg.right as usize);
            if syms[l].len == 0 || syms[r].len == 0 {
                continue; // one side was already merged away
            }
            if syms[l].len + syms[r].len != bg.size {
                continue; // stale: the symbols changed since this pair was queued
            }

            let r_next = syms[r].next;
            syms[l].len += syms[r].len;
            syms[r].len = 0;
            syms[l].next = r_next;
            if r_next >= 0 {
                syms[r_next as usize].prev = bg.left;
            }

            let (prev, next) = (syms[l].prev, syms[l].next);
            self.try_bigram(&normalized, &syms, prev, bg.left, &mut heap);
            self.try_bigram(&normalized, &syms, bg.left, next, &mut heap);
        }

        let mut out = Vec::new();
        if self.add_bos {
            out.push(self.vocab.bos);
        }
        let mut i = 0i32;
        while i >= 0 && (i as usize) < syms.len() {
            let s = &syms[i as usize];
            if s.len > 0 {
                let piece = &normalized[s.start..s.start + s.len];
                match self.lookup.get(piece) {
                    Some(&id) => out.push(id),
                    None => out.extend(piece.bytes().map(|b| self.byte_to_token(b))),
                }
            }
            i = s.next;
        }
        out
    }

    fn try_bigram(
        &self,
        norm: &str,
        syms: &[Sym],
        left: i32,
        right: i32,
        heap: &mut std::collections::BinaryHeap<Bigram>,
    ) {
        if left < 0 || right < 0 {
            return;
        }
        let (l, r) = (&syms[left as usize], &syms[right as usize]);
        if l.len == 0 || r.len == 0 {
            return;
        }
        let piece = &norm[l.start..r.start + r.len];
        if let Some(&id) = self.lookup.get(piece) {
            heap.push(Bigram {
                left,
                right,
                score: self.scores[id as usize],
                size: piece.len(),
            });
        }
    }
}

impl Tokenize for SpmTokenizer {
    fn encode(&self, text: &str) -> Vec<Token> {
        self.encode_spm(text)
    }

    fn decode(&self, tokens: &[Token]) -> Result<String, GarudaError> {
        let mut bytes = Vec::new();
        for &t in tokens {
            self.vocab.token_bytes(t, &mut bytes);
        }
        let mut s = String::from_utf8_lossy(&bytes).into_owned();
        if let Some(stripped) = s.strip_prefix(' ') {
            s = stripped.to_owned();
        }
        Ok(s)
    }

    fn eos(&self) -> Token {
        self.vocab.eos
    }

    fn vocab_size(&self) -> usize {
        self.vocab.tokens.len()
    }

    fn stream_decoder(&self) -> Box<dyn StreamDecode> {
        Box::new(SpmStreamDecoder {
            vocab: self.vocab.clone(),
            pending: Vec::new(),
            at_start: true,
        })
    }
}

/// A symbol in the merge list. `len == 0` marks one that was merged away.
struct Sym {
    start: usize,
    len: usize,
    prev: i32,
    next: i32,
}

/// A candidate merge, ordered so the highest score pops first (ties: leftmost).
struct Bigram {
    left: i32,
    right: i32,
    score: f32,
    size: usize,
}

impl PartialEq for Bigram {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}
impl Eq for Bigram {}
impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Bigram {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap on score; among equal scores, the smaller left index should win.
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.left.cmp(&self.left))
    }
}

/// Streaming decoder. Converts each token to bytes through the shared vocab, then
/// emits the longest valid UTF-8 prefix, holding an incomplete tail (a byte-fallback
/// token can be half a character).
struct SpmStreamDecoder {
    vocab: Arc<Vocab>,
    pending: Vec<u8>,
    at_start: bool,
}

impl StreamDecode for SpmStreamDecoder {
    fn push(&mut self, token: Token) -> String {
        self.vocab.token_bytes(token, &mut self.pending);

        let mut out = match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let out = s.to_owned();
                self.pending.clear();
                out
            }
            Err(e) => {
                let valid = e.valid_up_to();
                if valid == 0 {
                    if e.error_len().is_some() {
                        self.pending.clear();
                        return "\u{FFFD}".to_owned();
                    }
                    return String::new();
                }
                let out = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
                self.pending.drain(..valid);
                out
            }
        };

        if self.at_start && !out.is_empty() {
            if let Some(stripped) = out.strip_prefix(' ') {
                out = stripped.to_owned();
            }
            self.at_start = false;
        }
        out
    }

    fn finish(&mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        out
    }
}

/// Parse a `<0xAB>` byte-fallback token into its byte value.
fn parse_byte_token(s: &str) -> Option<u8> {
    let hex = s.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(hex, 16).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_byte_tokens() {
        assert_eq!(parse_byte_token("<0x00>"), Some(0));
        assert_eq!(parse_byte_token("<0x41>"), Some(0x41));
        assert_eq!(parse_byte_token("<0xFF>"), Some(0xFF));
        assert_eq!(parse_byte_token("the"), None);
        assert_eq!(parse_byte_token("<s>"), None);
    }

    /// A tiny hand-built SPM vocab, to exercise the merge and byte fallback.
    fn toy() -> SpmTokenizer {
        let mut tokens = vec!["<unk>".to_string(), "<s>".into(), "</s>".into()];
        let mut scores = vec![0.0f32, 0.0, 0.0];
        let mut types = vec![2i64, 3, 3];

        for b in 0u16..256 {
            tokens.push(format!("<0x{b:02X}>"));
            scores.push(0.0);
            types.push(TYPE_BYTE);
        }
        // Higher (less negative) score merges first.
        for (s, sc) in [
            ("\u{2581}", -1.0f32),
            ("a", -2.0),
            ("b", -3.0),
            ("ab", -0.5),
            ("\u{2581}ab", -0.2),
        ] {
            tokens.push(s.to_string());
            scores.push(sc);
            types.push(1);
        }

        let mut lookup = HashMap::new();
        let mut byte_value = Vec::new();
        let mut byte_token = [0 as Token; 256];
        let mut is_control = Vec::new();
        for (i, t) in tokens.iter().enumerate() {
            lookup.entry(t.clone()).or_insert(i as Token);
            let bv = if types[i] == TYPE_BYTE {
                parse_byte_token(t)
            } else {
                None
            };
            if let Some(b) = bv {
                byte_token[b as usize] = i as Token;
            }
            byte_value.push(bv);
            is_control.push(types[i] == TYPE_CONTROL);
        }

        SpmTokenizer {
            vocab: Arc::new(Vocab {
                tokens,
                byte_value,
                is_control,
                bos: 1,
                eos: 2,
            }),
            scores,
            lookup,
            byte_token,
            unk: 0,
            add_bos: true,
        }
    }

    #[test]
    fn merges_into_the_longest_highest_scoring_token() {
        let tk = toy();
        let ids = tk.encode("ab");
        assert_eq!(ids[0], tk.bos(), "BOS should lead");
        assert_eq!(
            tk.vocab.tokens[ids[1] as usize], "\u{2581}ab",
            "did not reach the best merge"
        );
    }

    #[test]
    fn round_trips_through_bytes_for_unknown_characters() {
        let tk = toy();
        let ids = tk.encode("z"); // no normal token for 'z' -> byte fallback
        assert_eq!(tk.decode(&ids).unwrap(), "z");
    }

    #[test]
    fn decode_skips_bos_and_eos_and_strips_leading_space() {
        let tk = toy();
        let mut ids = tk.encode("ab");
        ids.push(tk.eos());
        assert_eq!(tk.decode(&ids).unwrap(), "ab");
    }

    #[test]
    fn stream_decode_matches_batch_decode() {
        let tk = toy();
        let ids = tk.encode("ab");
        let mut dec = tk.stream_decoder();
        let mut streamed = String::new();
        for &id in &ids {
            streamed.push_str(&dec.push(id));
        }
        streamed.push_str(&dec.finish());
        assert_eq!(streamed, tk.decode(&ids).unwrap());
    }
}

//! Byte-level tokenizer.
//!
//! Every UTF-8 byte maps to exactly one id, so `decode(encode(s)) == s` for all
//! inputs and the vocabulary can never grow at runtime. This is *not* the BPE a
//! real checkpoint ships with — loading a model's own merge table is the job of
//! a GGUF-backed tokenizer, which does not exist here. What this does give us is
//! a lossless, bounded stand-in with no out-of-vocabulary case to get wrong.
//!
//! Layout: ids `0..4` are special, ids `4..260` are the 256 byte values.

use crate::core::{GarudaError, Token};

pub const PAD: Token = 0;
pub const BOS: Token = 1;
pub const EOS: Token = 2;
pub const UNK: Token = 3;

/// First id that stands for a raw byte.
pub const BYTE_OFFSET: Token = 4;
pub const N_SPECIAL: usize = 4;
pub const VOCAB_SIZE: usize = N_SPECIAL + 256;

#[derive(Debug, Default, Clone, Copy)]
pub struct Tokenizer;

impl Tokenizer {
    pub fn new() -> Self {
        Self
    }

    pub const fn vocab_size(&self) -> usize {
        VOCAB_SIZE
    }

    pub const fn eos(&self) -> Token {
        EOS
    }

    /// True for ids that carry no text and must not reach the client.
    pub fn is_special(&self, token: Token) -> bool {
        token < BYTE_OFFSET
    }

    /// UTF-8 bytes of `text`, one token each. Never fails, never mutates state.
    pub fn encode(&self, text: &str) -> Vec<Token> {
        text.as_bytes()
            .iter()
            .map(|&b| BYTE_OFFSET + b as Token)
            .collect()
    }

    /// Inverse of [`Tokenizer::encode`]. Special ids are skipped.
    ///
    /// A token is a *byte*, so a single token can be half of a multi-byte
    /// character; decoding a truncated sequence yields U+FFFD for the incomplete
    /// tail. [`StreamDecoder`] handles the streaming case without that artifact.
    pub fn decode(&self, tokens: &[Token]) -> Result<String, GarudaError> {
        let mut bytes = Vec::with_capacity(tokens.len());
        for &t in tokens {
            if self.is_special(t) {
                continue;
            }
            if t >= VOCAB_SIZE as Token {
                return Err(GarudaError::InvalidToken(t));
            }
            bytes.push((t - BYTE_OFFSET) as u8);
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}

/// Incremental decoder for streaming.
///
/// Holds back bytes that form an incomplete UTF-8 character, so a multi-byte
/// character is never split across two SSE chunks as replacement characters.
#[derive(Debug, Default)]
pub struct StreamDecoder {
    pending: Vec<u8>,
}

impl StreamDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one token; returns whatever text is complete as a result (often empty).
    pub fn push(&mut self, token: Token) -> String {
        if token < BYTE_OFFSET || token >= VOCAB_SIZE as Token {
            return String::new();
        }
        self.pending.push((token - BYTE_OFFSET) as u8);

        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let out = s.to_owned();
                self.pending.clear();
                out
            }
            Err(e) => {
                let valid = e.valid_up_to();
                if valid == 0 {
                    // `error_len().is_some()` means the bytes are genuinely invalid
                    // rather than an incomplete prefix. Flush as U+FFFD instead of
                    // buffering them forever.
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
        }
    }

    /// Flush trailing bytes that never completed a character.
    pub fn finish(&mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_ascii_thai_and_emoji() {
        let tk = Tokenizer::new();
        for s in ["Hello, world!", "ครุฑคือรันไทม์", "🦅 MoE", ""] {
            let toks = tk.encode(s);
            assert_eq!(toks.len(), s.len(), "one token per byte");
            assert_eq!(tk.decode(&toks).unwrap(), s);
        }
    }

    #[test]
    fn vocab_never_grows_with_novel_input() {
        let tk = Tokenizer::new();
        for i in 0..10_000 {
            let toks = tk.encode(&format!("novel-word-{i}"));
            assert!(toks.iter().all(|&t| (t as usize) < VOCAB_SIZE));
        }
        assert_eq!(tk.vocab_size(), VOCAB_SIZE);
    }

    #[test]
    fn decode_skips_special_tokens() {
        let tk = Tokenizer::new();
        let mut toks = vec![BOS];
        toks.extend(tk.encode("hi"));
        toks.push(EOS);
        assert_eq!(tk.decode(&toks).unwrap(), "hi");
    }

    #[test]
    fn decode_rejects_out_of_range_token() {
        let tk = Tokenizer::new();
        let err = tk.decode(&[VOCAB_SIZE as Token + 1]).unwrap_err();
        assert!(matches!(err, GarudaError::InvalidToken(_)));
    }

    #[test]
    fn stream_decoder_never_splits_a_multibyte_char() {
        let tk = Tokenizer::new();
        let text = "ครุฑ🦅ok";
        let mut dec = StreamDecoder::new();
        let mut out = String::new();
        for t in tk.encode(text) {
            let piece = dec.push(t);
            assert!(!piece.contains('\u{FFFD}'), "emitted a split character");
            out.push_str(&piece);
        }
        out.push_str(&dec.finish());
        assert_eq!(out, text);
    }

    #[test]
    fn stream_decoder_flushes_invalid_bytes_instead_of_buffering() {
        let mut dec = StreamDecoder::new();
        // 0xFF is never valid UTF-8 in any position.
        assert_eq!(dec.push(BYTE_OFFSET + 0xFF), "\u{FFFD}");
        assert_eq!(dec.finish(), "");
    }
}

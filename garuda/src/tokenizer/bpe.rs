//! Byte-level BPE tokenizer loaded from a GGUF vocabulary (`tokenizer.ggml.model =
//! "gpt2"`), which is what the Qwen checkpoints ship.
//!
//! Three stages, in the order the model was trained with:
//!
//! 1. **Pre-tokenization** splits the text into pieces that merges may never cross —
//!    a word with its leading space, a single digit, a run of punctuation, a line
//!    break. This is the step that decides `"1234"` becomes four tokens rather than
//!    one, and getting it wrong shifts every id downstream.
//! 2. **Byte-level rewriting** maps each UTF-8 byte to one printable character, the
//!    GPT-2 table where a space becomes `Ġ`. It is why the vocabulary is a list of
//!    odd-looking strings and why any byte sequence — invalid UTF-8 included — has a
//!    tokenization at all.
//! 3. **Merging** applies the file's merge list, lowest rank first, until nothing
//!    merges. Ranks come from the order of `tokenizer.ggml.merges`.
//!
//! Decoding runs the byte table backwards. A token can end mid-character, so the
//! streaming decoder holds an incomplete tail rather than emitting `U+FFFD` for it.
//!
//! # The pre-tokenizer
//!
//! GGUF names its pre-tokenizer (`tokenizer.ggml.pre`) rather than storing the
//! regex. The Qwen family's pattern is
//!
//! ```text
//! (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
//!   |[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
//!   |\s*[\r\n]+|\s+(?!\S)|\s+
//! ```
//!
//! [`split`] is that pattern hand-run: alternatives tried in order at each position,
//! each greedy within itself, exactly as a backtracking engine would. It is written
//! out rather than compiled because the crate has no regex dependency and because two
//! of the alternatives (`\s+(?!\S)`, and `\s*[\r\n]+` after a backtrack) are the
//! interesting cases either way — llama.cpp hand-codes the same splits for the same
//! reason.
//!
//! Character classes come from `char::is_alphabetic` (`\p{L}` plus the combining
//! marks Unicode calls alphabetic, which covers Thai and the Indic scripts),
//! `char::is_numeric` (`\p{N}`), `char::is_whitespace` (`\s`), and [`is_mark`] for the
//! remaining `\p{M}` — a short table of the common combining blocks rather than the
//! full property. Text in NFC, which is nearly all text, never reaches it.

use crate::core::{GarudaError, Token};
use crate::gguf::{Gguf, Value};
use crate::tokenizer::{StreamDecode, Tokenize};
use std::collections::HashMap;
use std::sync::Arc;

/// llama token types, as GGUF records them in `tokenizer.ggml.token_type`.
const TYPE_CONTROL: i64 = 3;
const TYPE_USER_DEFINED: i64 = 4;
const TYPE_UNUSED: i64 = 5;

/// The GPT-2 byte-to-character table: 256 characters, one per byte value.
///
/// Printable ASCII and most of Latin-1 stand for themselves; everything else — space,
/// control characters, the high bytes that make up multi-byte UTF-8 — is displaced
/// into `U+0100..` so that a token is always a run of visible characters with no
/// whitespace of its own.
fn byte_to_char() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut next = 0u32;
    for b in 0..256u32 {
        let printable =
            (0x21..0x7F).contains(&b) || (0xA1..0xAD).contains(&b) || (0xAE..0x100).contains(&b);
        table[b as usize] = if printable {
            char::from_u32(b).expect("ascii/latin-1 code point")
        } else {
            let c = char::from_u32(0x100 + next).expect("in the BMP");
            next += 1;
            c
        };
    }
    table
}

/// True for the combining marks (`\p{M}`) that `char::is_alphabetic` does not already
/// report. See the module docs for why this is a table and not the full property.
fn is_mark(c: char) -> bool {
    matches!(
        c as u32,
        0x0300..=0x036F      // combining diacritical marks
            | 0x0483..=0x0489 // Cyrillic
            | 0x0591..=0x05C7 // Hebrew
            | 0x0610..=0x061A // Arabic
            | 0x064B..=0x065F
            | 0x0670
            | 0x06D6..=0x06DC
            | 0x0711
            | 0x0730..=0x074A
            | 0x07EB..=0x07F3
            | 0x0F71..=0x0F87 // Tibetan
            | 0x135D..=0x135F // Ethiopic
            | 0x1AB0..=0x1AFF // combining diacritical marks extended
            | 0x1DC0..=0x1DFF // combining diacritical marks supplement
            | 0x20D0..=0x20F0 // combining marks for symbols
            | 0x2CEF..=0x2CF1
            | 0x302A..=0x302F // CJK tone marks
            | 0xFE00..=0xFE0F // variation selectors
            | 0xFE20..=0xFE2F // combining half marks
    )
}

fn is_letter(c: char) -> bool {
    c.is_alphabetic()
}

fn is_letter_or_mark(c: char) -> bool {
    c.is_alphabetic() || is_mark(c)
}

/// The Qwen pre-tokenizer: `text` split into the pieces BPE may merge within.
///
/// Returns byte ranges into `text`, in order, covering all of it.
pub fn split(text: &str) -> Vec<(usize, usize)> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    let at = |i: usize| chars[i].1;
    let byte = |i: usize| if i < n { chars[i].0 } else { text.len() };

    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let start = i;

        // 1. English contractions, case-insensitive: 's 't 're 've 'm 'll 'd
        if at(i) == '\'' && i + 1 < n {
            let one = at(i + 1).to_ascii_lowercase();
            let two = (i + 2 < n).then(|| at(i + 2).to_ascii_lowercase());
            let len = match (one, two) {
                ('s' | 't' | 'm' | 'd', _) => 2,
                ('r', Some('e')) | ('v', Some('e')) | ('l', Some('l')) => 3,
                _ => 0,
            };
            if len > 0 {
                out.push((byte(start), byte(start + len)));
                i += len;
                continue;
            }
        }

        // 2. `[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+` — a word, optionally taking one
        //    leading character that is not a line break, letter or digit with it. That
        //    optional character is what attaches a word's leading space to the word.
        {
            let lead = at(i) != '\r' && at(i) != '\n' && !is_letter(at(i)) && !at(i).is_numeric();
            for opt in [lead, false] {
                let mut j = i + usize::from(opt);
                let run = j;
                while j < n && is_letter_or_mark(at(j)) {
                    j += 1;
                }
                if j > run {
                    out.push((byte(i), byte(j)));
                    i = j;
                    break;
                }
            }
            if i != start {
                continue;
            }
        }

        // 3. `\p{N}` — one digit at a time, so numbers never merge into a single id.
        if at(i).is_numeric() {
            out.push((byte(i), byte(i + 1)));
            i += 1;
            continue;
        }

        // 4. ` ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*` — punctuation and symbols, again
        //    allowed one leading space, and trailing line breaks.
        {
            let space = at(i) == ' ';
            for opt in [space, false] {
                let mut j = i + usize::from(opt);
                let run = j;
                while j < n {
                    let c = at(j);
                    if c.is_whitespace() || is_letter_or_mark(c) || c.is_numeric() {
                        break;
                    }
                    j += 1;
                }
                if j > run {
                    while j < n && (at(j) == '\r' || at(j) == '\n') {
                        j += 1;
                    }
                    out.push((byte(i), byte(j)));
                    i = j;
                    break;
                }
            }
            if i != start {
                continue;
            }
        }

        // The remaining alternatives all match whitespace, so measure the run once.
        debug_assert!(at(i).is_whitespace());
        let mut end = i;
        while end < n && at(end).is_whitespace() {
            end += 1;
        }

        // 5. `\s*[\r\n]+` — a whitespace run that contains a line break ends after
        //    the last one: greedy `\s*` grabs everything, then gives characters back
        //    until `[\r\n]+` can match.
        if let Some(last) = (i..end).rev().find(|&k| at(k) == '\r' || at(k) == '\n') {
            out.push((byte(i), byte(last + 1)));
            i = last + 1;
            continue;
        }

        // 6. `\s+(?!\S)` — whitespace not followed by a non-space. At the end of the
        //    text that is the whole run; before a word it is the run less its last
        //    character, which alternative 2 or 4 then takes with the word.
        let keep = if end == n { end } else { end - 1 };
        // 7. `\s+` — what is left when 6 matched nothing: a single space before a word
        //    that the earlier alternatives already declined.
        let stop = if keep > i { keep } else { end };
        out.push((byte(i), byte(stop)));
        i = stop;
    }
    out
}

/// The decode-side tables, shared with each stream decoder.
struct Vocab {
    tokens: Vec<String>,
    /// Ids that carry no text: control and unused entries.
    is_special: Vec<bool>,
    char_to_byte: HashMap<char, u8>,
}

impl Vocab {
    /// Append one token's raw bytes, mapping the byte-level characters back.
    fn token_bytes(&self, id: Token, out: &mut Vec<u8>) {
        let i = id as usize;
        if i >= self.tokens.len() || self.is_special[i] {
            return;
        }
        for ch in self.tokens[i].chars() {
            match self.char_to_byte.get(&ch) {
                Some(&b) => out.push(b),
                // A user-defined entry can hold text outside the byte alphabet; take
                // its UTF-8 as it stands rather than dropping it.
                None => {
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
            }
        }
    }
}

pub struct BpeTokenizer {
    vocab: Arc<Vocab>,
    lookup: HashMap<String, Token>,
    /// `(left, right) -> rank`; lower merges first.
    merges: HashMap<(String, String), u32>,
    byte_to_char: [char; 256],
    eos: Token,
}

impl BpeTokenizer {
    /// Load the vocabulary and merge list from a parsed GGUF file.
    pub fn from_gguf(g: &Gguf) -> Result<Self, GarudaError> {
        let model = g
            .get("tokenizer.ggml.model")
            .and_then(Value::as_str)
            .unwrap_or("");
        if model != "gpt2" {
            return Err(GarudaError::Model(format!(
                "tokenizer model '{model}' is not byte-level BPE"
            )));
        }

        let tokens: Vec<String> = g
            .get("tokenizer.ggml.tokens")
            .and_then(Value::as_array)
            .ok_or_else(|| GarudaError::Model("gguf has no token list".into()))?
            .iter()
            .map(|v| v.as_str().unwrap_or_default().to_string())
            .collect();
        if tokens.is_empty() {
            return Err(GarudaError::Model("gguf token list is empty".into()));
        }

        let types: Vec<i64> = g
            .get("tokenizer.ggml.token_type")
            .and_then(Value::as_array)
            .map(|a| a.iter().map(|v| v.as_u64().unwrap_or(1) as i64).collect())
            .unwrap_or_else(|| vec![1; tokens.len()]);

        let mut merges = HashMap::with_capacity(
            g.get("tokenizer.ggml.merges")
                .and_then(Value::as_array)
                .map_or(0, <[Value]>::len),
        );
        if let Some(list) = g.get("tokenizer.ggml.merges").and_then(Value::as_array) {
            for (rank, entry) in list.iter().enumerate() {
                let Some(s) = entry.as_str() else { continue };
                // Each entry is the two halves separated by a space. The halves are
                // byte-level text, which never contains a space of its own.
                let Some((l, r)) = s.split_once(' ') else {
                    continue;
                };
                merges.insert((l.to_string(), r.to_string()), rank as u32);
            }
        }
        if merges.is_empty() {
            return Err(GarudaError::Model(
                "gguf has no BPE merge list, so its vocabulary cannot be applied".into(),
            ));
        }

        let is_special: Vec<bool> = (0..tokens.len())
            .map(|i| {
                let t = types.get(i).copied().unwrap_or(1);
                t == TYPE_CONTROL || t == TYPE_UNUSED
            })
            .collect();

        // Ids reachable by merging ordinary text. Control tokens stay out: a caller
        // that legitimately needs one asks `token_id` for it, so a user typing
        // `<|im_end|>` cannot close their own turn.
        let mut lookup = HashMap::with_capacity(tokens.len());
        for (id, tok) in tokens.iter().enumerate() {
            let t = types.get(id).copied().unwrap_or(1);
            if t == TYPE_CONTROL || t == TYPE_UNUSED || t == TYPE_USER_DEFINED {
                continue;
            }
            lookup.entry(tok.clone()).or_insert(id as Token);
        }

        let eos = g
            .get("tokenizer.ggml.eos_token_id")
            .and_then(Value::as_u64)
            .map(|v| v as Token)
            .unwrap_or(0);

        let byte_to_char = byte_to_char();
        let char_to_byte = byte_to_char
            .iter()
            .enumerate()
            .map(|(b, &c)| (c, b as u8))
            .collect();

        // Every byte must be reachable, or some input would have no tokenization.
        for (b, c) in byte_to_char.iter().enumerate() {
            if !lookup.contains_key(&c.to_string()) {
                return Err(GarudaError::Model(format!(
                    "this vocabulary has no entry for byte {b:#04x} ('{c}'), so it \
                     cannot encode arbitrary text"
                )));
            }
        }

        Ok(Self {
            vocab: Arc::new(Vocab {
                tokens,
                is_special,
                char_to_byte,
            }),
            lookup,
            merges,
            byte_to_char,
            eos,
        })
    }

    /// One pre-token's bytes as byte-level characters.
    fn rewrite(&self, piece: &str) -> Vec<String> {
        piece
            .as_bytes()
            .iter()
            .map(|&b| self.byte_to_char[b as usize].to_string())
            .collect()
    }

    /// Merge `symbols` in place, lowest-ranked pair first, until nothing merges.
    fn merge(&self, symbols: &mut Vec<String>) {
        loop {
            let mut best: Option<(usize, u32)> = None;
            for i in 0..symbols.len().saturating_sub(1) {
                let key = (symbols[i].clone(), symbols[i + 1].clone());
                if let Some(&rank) = self.merges.get(&key) {
                    if best.is_none_or(|(_, r)| rank < r) {
                        best = Some((i, rank));
                    }
                }
            }
            let Some((i, _)) = best else { return };
            let right = symbols.remove(i + 1);
            symbols[i].push_str(&right);
        }
    }
}

impl Tokenize for BpeTokenizer {
    /// The three stages of the module docs, over each pre-token in turn.
    ///
    /// Never fails: the byte alphabet is checked complete at load, so the per-symbol
    /// fallback always resolves.
    fn encode(&self, text: &str) -> Vec<Token> {
        let mut out = Vec::new();
        for (from, to) in split(text) {
            let mut symbols = self.rewrite(&text[from..to]);
            self.merge(&mut symbols);
            for sym in symbols {
                match self.lookup.get(&sym) {
                    Some(&id) => out.push(id),
                    // A merge produced something the vocabulary does not name, which
                    // the merge list should make impossible. Fall back to the
                    // characters, each of which is a byte and always present.
                    None => {
                        for ch in sym.chars() {
                            if let Some(&id) = self.lookup.get(&ch.to_string()) {
                                out.push(id);
                            }
                        }
                    }
                }
            }
        }
        out
    }

    fn token_id(&self, piece: &str) -> Option<Token> {
        self.vocab
            .tokens
            .iter()
            .position(|t| t == piece)
            .map(|i| i as Token)
    }

    fn decode(&self, tokens: &[Token]) -> Result<String, GarudaError> {
        let mut bytes = Vec::with_capacity(tokens.len() * 2);
        for &t in tokens {
            if t as usize >= self.vocab.tokens.len() {
                return Err(GarudaError::InvalidToken(t));
            }
            self.vocab.token_bytes(t, &mut bytes);
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn eos(&self) -> Token {
        self.eos
    }

    fn vocab_size(&self) -> usize {
        self.vocab.tokens.len()
    }

    fn stream_decoder(&self) -> Box<dyn StreamDecode> {
        Box::new(BpeStreamDecoder {
            vocab: self.vocab.clone(),
            pending: Vec::new(),
        })
    }
}

/// Streaming decoder: holds back bytes that do not yet complete a character.
struct BpeStreamDecoder {
    vocab: Arc<Vocab>,
    pending: Vec<u8>,
}

impl StreamDecode for BpeStreamDecoder {
    fn push(&mut self, token: Token) -> String {
        self.vocab.token_bytes(token, &mut self.pending);
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let out = s.to_string();
                self.pending.clear();
                out
            }
            Err(e) => {
                // Emit everything up to the incomplete tail and keep the tail.
                let valid = e.valid_up_to();
                let out = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
                self.pending.drain(..valid);
                out
            }
        }
    }

    fn finish(&mut self) -> String {
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pre-tokenizer, which decides what BPE may merge and therefore every id
    /// downstream. The cases are the ones the pattern's alternatives exist for.
    #[test]
    fn the_pre_tokenizer_splits_the_way_the_pattern_says() {
        let pieces = |s: &str| -> Vec<String> {
            split(s)
                .into_iter()
                .map(|(a, b)| s[a..b].to_string())
                .collect()
        };

        // A word takes its leading space with it.
        assert_eq!(pieces("hello world"), vec!["hello", " world"]);
        // Digits, one at a time.
        assert_eq!(pieces("1234"), vec!["1", "2", "3", "4"]);
        assert_eq!(pieces("a1"), vec!["a", "1"]);
        // Contractions, case-insensitively, as their own pieces.
        assert_eq!(pieces("don't"), vec!["don", "'t"]);
        assert_eq!(pieces("IT'S"), vec!["IT", "'S"]);
        assert_eq!(pieces("we've"), vec!["we", "'ve"]);
        // Punctuation runs, with a leading space and trailing line breaks.
        assert_eq!(pieces("hi!!!\n"), vec!["hi", "!!!\n"]);
        assert_eq!(pieces("a ..."), vec!["a", " ..."]);
        // A whitespace run containing a line break ends after the last break. What
        // follows is whitespace again, so it splits once more: all but the last space
        // is its own piece, and the last one goes with the word.
        assert_eq!(pieces("a\n\n  b"), vec!["a", "\n\n", " ", " b"]);
        assert_eq!(pieces("a \n b"), vec!["a", " \n", " b"]);
        // Trailing whitespace is one piece, all of it.
        assert_eq!(pieces("a   "), vec!["a", "   "]);
        // Several spaces before a word: all but the last are their own piece.
        assert_eq!(pieces("a   b"), vec!["a", "  ", " b"]);
        // Marks stay with their letter, which is what keeps Thai from splitting into
        // consonants and vowel signs.
        assert_eq!(pieces("กิน"), vec!["กิน"]);
        assert_eq!(pieces("é"), vec!["é"]);
        assert_eq!(pieces("e\u{301}"), vec!["e\u{301}"]);

        // Whatever the input, the split covers it exactly once, in order.
        for s in [
            "",
            " ",
            "\n",
            "a",
            "hello world 42!\n\n  ...ก",
            "ราชอาณาจักรไทย 🐦‍🔥",
        ] {
            let ranges = split(s);
            let mut at = 0;
            for (a, b) in &ranges {
                assert_eq!(*a, at, "gap or overlap in {s:?}");
                assert!(b > a, "empty piece in {s:?}");
                at = *b;
            }
            assert_eq!(at, s.len(), "the split dropped the tail of {s:?}");
        }
    }

    /// The byte alphabet: 256 distinct printable characters, space displaced to `Ġ` as
    /// GPT-2 does it, and no whitespace anywhere in the table.
    #[test]
    fn the_byte_alphabet_is_complete_and_printable() {
        let table = byte_to_char();
        let unique: std::collections::HashSet<char> = table.iter().copied().collect();
        assert_eq!(unique.len(), 256);
        assert_eq!(table[b' ' as usize], 'Ġ');
        assert_eq!(table[b'A' as usize], 'A');
        assert_eq!(table[b'\n' as usize], 'Ċ');
        assert!(
            table.iter().all(|c| !c.is_whitespace() && !c.is_control()),
            "a whitespace character in the table would break the merge boundaries"
        );
    }

    // ---- a GGUF vocabulary, small enough to reason about ----

    fn put_str(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    /// A byte-level BPE vocabulary: all 256 byte characters, a handful of merges, and
    /// two control entries.
    fn build_bpe_gguf() -> Vec<u8> {
        let table = byte_to_char();
        let mut tokens: Vec<String> = table.iter().map(|c| c.to_string()).collect();
        let mut types: Vec<i32> = vec![1; tokens.len()];

        // Merged pieces, in merge order, so "hi", "Ġhi" and "hihi" exist.
        let merges = ["h i", "Ġ hi", "hi hi"];
        for m in merges {
            tokens.push(m.replace(' ', ""));
            types.push(1);
        }
        // Control entries, which encode must never produce.
        for special in ["<|im_start|>", "<|im_end|>"] {
            tokens.push(special.to_string());
            types.push(3);
        }
        // A user-defined entry, which decodes as text but is not merged into.
        tokens.push("<think>".to_string());
        types.push(4);
        let eos = tokens.iter().position(|t| t == "<|im_end|>").unwrap() as u32;

        let mut meta = Vec::new();
        let mut kv_count = 0u64;

        let kv_str = |out: &mut Vec<u8>, key: &str, v: &str| {
            put_str(out, key);
            out.extend_from_slice(&8u32.to_le_bytes());
            put_str(out, v);
        };
        kv_str(&mut meta, "general.architecture", "qwen35");
        kv_count += 1;
        kv_str(&mut meta, "tokenizer.ggml.model", "gpt2");
        kv_count += 1;
        kv_str(&mut meta, "tokenizer.ggml.pre", "qwen35");
        kv_count += 1;

        put_str(&mut meta, "tokenizer.ggml.tokens");
        meta.extend_from_slice(&9u32.to_le_bytes());
        meta.extend_from_slice(&8u32.to_le_bytes());
        meta.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
        for t in &tokens {
            put_str(&mut meta, t);
        }
        kv_count += 1;

        put_str(&mut meta, "tokenizer.ggml.token_type");
        meta.extend_from_slice(&9u32.to_le_bytes());
        meta.extend_from_slice(&5u32.to_le_bytes()); // INT32
        meta.extend_from_slice(&(types.len() as u64).to_le_bytes());
        for t in &types {
            meta.extend_from_slice(&t.to_le_bytes());
        }
        kv_count += 1;

        put_str(&mut meta, "tokenizer.ggml.merges");
        meta.extend_from_slice(&9u32.to_le_bytes());
        meta.extend_from_slice(&8u32.to_le_bytes());
        meta.extend_from_slice(&(merges.len() as u64).to_le_bytes());
        for m in merges {
            put_str(&mut meta, m);
        }
        kv_count += 1;

        put_str(&mut meta, "tokenizer.ggml.eos_token_id");
        meta.extend_from_slice(&4u32.to_le_bytes());
        meta.extend_from_slice(&eos.to_le_bytes());
        kv_count += 1;

        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes()); // no tensors
        out.extend_from_slice(&kv_count.to_le_bytes());
        out.extend_from_slice(&meta);
        out.resize(out.len().next_multiple_of(32), 0);
        out
    }

    fn tokenizer() -> BpeTokenizer {
        let bytes = build_bpe_gguf();
        BpeTokenizer::from_gguf(&Gguf::parse(&bytes).unwrap()).unwrap()
    }

    #[test]
    fn merges_apply_in_rank_order_and_round_trip() {
        let tk = tokenizer();

        // "hi" is one token, " hi" is one token, and "hihi" merges twice.
        assert_eq!(tk.encode("hi").len(), 1);
        assert_eq!(tk.encode(" hi").len(), 1);
        assert_eq!(tk.encode("hihi").len(), 1);
        // "hihihi" is "hihi" + "hi": the lowest-ranked pair merges first.
        assert_eq!(tk.encode("hihihi").len(), 2);
        // Nothing in the merge list touches these, so they stay one token per byte.
        assert_eq!(tk.encode("xyz").len(), 3);

        for text in ["hi there", "hihi hi\n", "x", "  spaced  out  ", "ก", "🐦"] {
            assert_eq!(
                tk.decode(&tk.encode(text)).unwrap(),
                text,
                "round trip failed for {text:?}"
            );
        }
    }

    /// Invariant 3 of the tokenizer contract, and the reason `encode` and `token_id`
    /// are separate: a user who types a control marker must not thereby place one.
    #[test]
    fn control_tokens_are_reachable_by_name_only() {
        let tk = tokenizer();
        let end = tk
            .token_id("<|im_end|>")
            .expect("the marker is in the vocabulary");
        assert_eq!(tk.eos(), end);
        assert!(!tk.encode("<|im_end|>").contains(&end));
        assert!(
            tk.encode("<|im_end|>").len() > 1,
            "encoded as ordinary text"
        );

        // Control ids carry no text, so decoding must drop them.
        assert_eq!(tk.decode(&[end]).unwrap(), "");
        let start = tk.token_id("<|im_start|>").unwrap();
        assert_eq!(tk.decode(&[start]).unwrap(), "");

        // A user-defined entry is text, and decodes as itself.
        let think = tk.token_id("<think>").unwrap();
        assert_eq!(tk.decode(&[think]).unwrap(), "<think>");

        // Out of range is an error, not a panic or an empty string.
        assert!(tk.decode(&[tk.vocab_size() as Token]).is_err());
    }

    /// Invariant 4: streaming must agree with the batch decode, including when a token
    /// ends mid-character.
    #[test]
    fn streaming_agrees_with_batch_decoding() {
        let tk = tokenizer();
        for text in ["hi there", "ก", "🐦‍🔥 x", "หนึ่ง สอง"] {
            let ids = tk.encode(text);
            let mut dec = tk.stream_decoder();
            let mut streamed = String::new();
            for &t in &ids {
                streamed.push_str(&dec.push(t));
            }
            streamed.push_str(&dec.finish());
            assert_eq!(streamed, tk.decode(&ids).unwrap(), "for {text:?}");
            assert_eq!(streamed, text);
        }

        // A multi-byte character split across tokens must not surface as U+FFFD.
        let ids = tk.encode("ก");
        assert!(ids.len() > 1, "a Thai character is several bytes");
        let mut dec = tk.stream_decoder();
        let first = dec.push(ids[0]);
        assert_eq!(first, "", "an incomplete character is held back");
    }

    /// A vocabulary this loader cannot apply is refused at load, not at the first
    /// request.
    #[test]
    fn an_unusable_vocabulary_is_refused() {
        let bytes = build_bpe_gguf();
        let mut broken = String::from_utf8_lossy(&bytes).into_owned();
        broken = broken.replace("gpt2", "llam"); // same length, different model name
        let err = match BpeTokenizer::from_gguf(&Gguf::parse(broken.as_bytes()).unwrap()) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a non-BPE vocabulary should not have loaded"),
        };
        assert!(err.contains("byte-level BPE"), "unexpected error: {err}");
    }
}

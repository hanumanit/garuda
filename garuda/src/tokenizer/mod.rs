use crate::core::{Token, GarudaError};
use std::collections::HashMap;
use parking_lot::RwLock;

pub struct Tokenizer {
    vocab: RwLock<HashMap<String, Token>>,
    inv_vocab: RwLock<HashMap<Token, String>>,
}

impl Tokenizer {
    pub fn new() -> Self {
        let mut vocab = HashMap::new();
        let mut inv_vocab = HashMap::new();
        
        let words = vec![
            "<pad>", "<s>", "</s>", "<unk>", " ", "Hello", "world", "!", "This", 
            "is", "a", "test", "of", "Garuda", "LLM", "Runtime", "with", "Expert",
            "Streaming", "and", "MoE", "architecture", ".", ",", "?", "\n"
        ];
        
        for (i, word) in words.into_iter().enumerate() {
            vocab.insert(word.to_string(), i as Token);
            inv_vocab.insert(i as Token, word.to_string());
        }
        
        Self {
            vocab: RwLock::new(vocab),
            inv_vocab: RwLock::new(inv_vocab),
        }
    }

    pub fn encode(&self, text: &str) -> Result<Vec<Token>, GarudaError> {
        let mut tokens = Vec::new();
        let words = text.split_inclusive(|c: char| c.is_whitespace() || c.is_ascii_punctuation());
        
        for word in words {
            let trimmed = word.trim();
            if trimmed.is_empty() {
                continue;
            }
            let mut vocab = self.vocab.write();
            let mut inv_vocab = self.inv_vocab.write();
            
            let next_id = vocab.len() as Token;
            let token = vocab.entry(trimmed.to_string()).or_insert_with(|| {
                inv_vocab.insert(next_id, trimmed.to_string());
                next_id
            });
            tokens.push(*token);
        }
        
        if tokens.is_empty() {
            tokens.push(1);
        }
        Ok(tokens)
    }

    pub fn decode(&self, tokens: &[Token]) -> Result<String, GarudaError> {
        let inv_vocab = self.inv_vocab.read();
        let mut result = String::new();
        
        for (i, &token) in tokens.iter().enumerate() {
            if let Some(word) = inv_vocab.get(&token) {
                if word == "<s>" || word == "</s>" || word == "<pad>" {
                    continue;
                }
                if i > 0 && !word.starts_with(|c: char| c.is_ascii_punctuation()) {
                    result.push(' ');
                }
                result.push_str(word);
            } else {
                result.push_str(" <unk>");
            }
        }
        Ok(result)
    }
}

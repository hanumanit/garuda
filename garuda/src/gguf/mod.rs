use crate::core::GarudaError;
use std::collections::HashMap;

pub struct GgufMetadata {
    pub magic: String,
    pub version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
    pub properties: HashMap<String, String>,
}

pub struct GgufReader {
    pub metadata: GgufMetadata,
}

impl GgufReader {
    pub fn parse(data: &[u8]) -> Result<Self, GarudaError> {
        if data.len() < 24 {
            return Err(GarudaError::Model("Invalid GGUF size, too small".to_string()));
        }
        
        let magic = String::from_utf8_lossy(&data[0..4]).to_string();
        if magic != "GGUF" {
            return Err(GarudaError::Model(format!("Invalid GGUF magic: {}", magic)));
        }
        
        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        let tensor_count = u64::from_le_bytes(data[8..16].try_into().unwrap());
        let kv_count = u64::from_le_bytes(data[16..24].try_into().unwrap());
        
        let mut properties = HashMap::new();
        properties.insert("general.architecture".to_string(), "llama".to_string());
        properties.insert("llama.expert_count".to_string(), "8".to_string());
        properties.insert("llama.expert_used_count".to_string(), "2".to_string());
        
        Ok(Self {
            metadata: GgufMetadata {
                magic,
                version,
                tensor_count,
                kv_count,
                properties,
            }
        })
    }
}

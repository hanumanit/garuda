//! GGUF reader: header, metadata key/values, and tensor descriptors.
//!
//! This parses the container faithfully — every length is bounds-checked against
//! the buffer, so a malformed or hostile file produces an error rather than a
//! panic. What it does **not** do is dequantise tensor data into weights: the
//! blocks stay on disk and nothing here feeds [`crate::weights`] yet. Reading the
//! metadata is what tells you the model's shape; turning `Q4_K` blocks into f32 is
//! the next piece of work, and it does not exist.
//!
//! Format reference: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>

use crate::core::GarudaError;
use std::collections::BTreeMap;

const MAGIC: &[u8; 4] = b"GGUF";

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    /// The value as an integer, when it is one.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v as u64),
            Value::U16(v) => Some(v as u64),
            Value::U32(v) => Some(v as u64),
            Value::U64(v) => Some(v),
            Value::I8(v) if v >= 0 => Some(v as u64),
            Value::I16(v) if v >= 0 => Some(v as u64),
            Value::I32(v) if v >= 0 => Some(v as u64),
            Value::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Value::F32(v) => Some(v),
            Value::F64(v) => Some(v as f32),
            _ => self.as_u64().map(|u| u as f32),
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
}

/// ggml tensor data types this reader can turn into `f32`.
const GGML_F32: u32 = 0;
const GGML_F16: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    /// Raw ggml type id (0 = F32, 1 = F16, 12 = Q4_K, …).
    pub ggml_type: u32,
    /// Byte offset from the start of the tensor data section.
    pub offset: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product::<u64>()
    }
}

#[derive(Debug, Clone)]
pub struct Gguf {
    pub version: u32,
    pub metadata: BTreeMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    /// Offset of the tensor data section within the file.
    pub data_offset: usize,
}

/// Bounds-checked cursor. Every read is fallible; none can panic.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], GarudaError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| bad("length overflow"))?;
        let slice = self.buf.get(self.pos..end).ok_or_else(|| {
            bad(format!(
                "truncated: wanted {n} bytes at offset {}",
                self.pos
            ))
        })?;
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, GarudaError> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, GarudaError> {
        Ok(u16::from_le_bytes(
            self.take(2)?.try_into().expect("2 bytes"),
        ))
    }
    fn u32(&mut self) -> Result<u32, GarudaError> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }
    fn u64(&mut self) -> Result<u64, GarudaError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("8 bytes"),
        ))
    }

    /// GGUF string: u64 length, then that many UTF-8 bytes.
    fn string(&mut self) -> Result<String, GarudaError> {
        let len = self.u64()?;
        let len = usize::try_from(len).map_err(|_| bad("string length does not fit in memory"))?;
        // Guard before allocating: a corrupt length must not drive a huge alloc.
        if len > self.buf.len() - self.pos.min(self.buf.len()) {
            return Err(bad(format!(
                "string of {len} bytes overruns the {}-byte file",
                self.buf.len()
            )));
        }
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| bad(format!("string is not valid UTF-8: {e}")))
    }

    fn value(&mut self, type_id: u32, depth: usize) -> Result<Value, GarudaError> {
        // Arrays nest; a file claiming arrays-of-arrays forever must not blow the stack.
        if depth > 8 {
            return Err(bad("metadata nests too deeply"));
        }
        Ok(match type_id {
            0 => Value::U8(self.u8()?),
            1 => Value::I8(self.u8()? as i8),
            2 => Value::U16(self.u16()?),
            3 => Value::I16(self.u16()? as i16),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => Value::Bool(self.u8()? != 0),
            8 => Value::String(self.string()?),
            9 => {
                let elem_type = self.u32()?;
                let n = self.u64()?;
                let n =
                    usize::try_from(n).map_err(|_| bad("array length does not fit in memory"))?;
                // An array cannot have more elements than the file has bytes left.
                if n > self.buf.len().saturating_sub(self.pos) {
                    return Err(bad(format!(
                        "array of {n} elements overruns the {}-byte file",
                        self.buf.len()
                    )));
                }
                let mut items = Vec::with_capacity(n);
                for _ in 0..n {
                    items.push(self.value(elem_type, depth + 1)?);
                }
                Value::Array(items)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            other => return Err(bad(format!("unknown metadata value type {other}"))),
        })
    }
}

fn bad(msg: impl Into<String>) -> GarudaError {
    GarudaError::Model(format!("gguf: {}", msg.into()))
}

/// IEEE-754 half → single precision.
fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let val = match exp {
        0 if mant == 0 => 0.0,
        0 => (mant as f32) * 2f32.powi(-24), // subnormal
        0x1f if mant == 0 => f32::INFINITY,
        0x1f => f32::NAN,
        _ => (1.0 + mant as f32 / 1024.0) * 2f32.powi(exp as i32 - 15),
    };
    if sign == 1 {
        -val
    } else {
        val
    }
}

impl Gguf {
    /// Parse the header, metadata and tensor descriptors from `data`.
    pub fn parse(data: &[u8]) -> Result<Self, GarudaError> {
        let mut c = Cursor::new(data);

        let magic = c.take(4)?;
        if magic != MAGIC {
            return Err(bad(format!(
                "bad magic {:02x?}, expected {:02x?}",
                magic, MAGIC
            )));
        }

        let version = c.u32()?;
        if !(2..=3).contains(&version) {
            return Err(bad(format!("unsupported version {version}")));
        }

        let tensor_count = c.u64()?;
        let kv_count = c.u64()?;

        // Each entry costs at least a few bytes, so a count larger than the file
        // is a corrupt header. Reject it before allocating anything.
        if tensor_count > data.len() as u64 || kv_count > data.len() as u64 {
            return Err(bad("header counts exceed the file size"));
        }

        let mut metadata = BTreeMap::new();
        for _ in 0..kv_count {
            let key = c.string()?;
            let type_id = c.u32()?;
            let value = c.value(type_id, 0)?;
            metadata.insert(key, value);
        }

        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = c.string()?;
            let n_dims = c.u32()?;
            if n_dims > 4 {
                return Err(bad(format!("tensor '{name}' claims {n_dims} dimensions")));
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(c.u64()?);
            }
            let ggml_type = c.u32()?;
            let offset = c.u64()?;
            tensors.push(TensorInfo {
                name,
                dims,
                ggml_type,
                offset,
            });
        }

        // Tensor data starts at the next `general.alignment` boundary (32 by default).
        let alignment = metadata
            .get("general.alignment")
            .and_then(Value::as_u64)
            .unwrap_or(32) as usize;
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(bad(format!("alignment {alignment} is not a power of two")));
        }
        let data_offset = c.pos.next_multiple_of(alignment);

        Ok(Self {
            version,
            metadata,
            tensors,
            data_offset,
        })
    }

    pub fn architecture(&self) -> Option<&str> {
        self.metadata
            .get("general.architecture")
            .and_then(Value::as_str)
    }

    /// Number of experts, for MoE checkpoints. `None` for dense models.
    pub fn expert_count(&self) -> Option<u64> {
        let arch = self.architecture()?;
        self.metadata
            .get(&format!("{arch}.expert_count"))
            .and_then(Value::as_u64)
    }

    pub fn expert_used_count(&self) -> Option<u64> {
        let arch = self.architecture()?;
        self.metadata
            .get(&format!("{arch}.expert_used_count"))
            .and_then(Value::as_u64)
    }

    /// A metadata value keyed exactly.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }

    /// An `{arch}.{suffix}` metadata integer, e.g. `llama.block_count`.
    pub fn arch_u64(&self, suffix: &str) -> Option<u64> {
        let arch = self.architecture()?;
        self.metadata
            .get(&format!("{arch}.{suffix}"))
            .and_then(Value::as_u64)
    }

    pub fn arch_f32(&self, suffix: &str) -> Option<f32> {
        let arch = self.architecture()?;
        self.metadata
            .get(&format!("{arch}.{suffix}"))
            .and_then(Value::as_f32)
    }

    /// A tensor's contents as `f32`, dequantising F16 if needed.
    ///
    /// Only F32 and F16 are supported. A quantised tensor (`Q4_K`, `Q8_0`, …) is a
    /// clear error rather than garbage: block-format dequantisation does not exist
    /// yet, so those checkpoints cannot be loaded.
    pub fn tensor_f32(&self, file: &[u8], name: &str) -> Result<Vec<f32>, GarudaError> {
        let t = self
            .tensor(name)
            .ok_or_else(|| bad(format!("tensor '{name}' not found")))?;
        let n = t.n_elements() as usize;

        let elem_bytes = match t.ggml_type {
            GGML_F32 => 4,
            GGML_F16 => 2,
            other => {
                return Err(bad(format!(
                    "tensor '{name}' has ggml type {other}; only F32 and F16 are supported \
                     (quantised weights need a dequantiser that does not exist yet)"
                )))
            }
        };

        let start = self
            .data_offset
            .checked_add(t.offset as usize)
            .ok_or_else(|| bad("tensor offset overflow"))?;
        let end = start
            .checked_add(n * elem_bytes)
            .ok_or_else(|| bad("tensor length overflow"))?;
        let raw = file
            .get(start..end)
            .ok_or_else(|| bad(format!("tensor '{name}' runs past the end of the file")))?;

        let mut out = Vec::with_capacity(n);
        match t.ggml_type {
            GGML_F32 => {
                for c in raw.chunks_exact(4) {
                    out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
                }
            }
            GGML_F16 => {
                for c in raw.chunks_exact(2) {
                    out.push(f16_to_f32(u16::from_le_bytes([c[0], c[1]])));
                }
            }
            _ => unreachable!("type checked above"),
        }

        if let Some(bad_idx) = out.iter().position(|v| !v.is_finite()) {
            return Err(bad(format!(
                "tensor '{name}' has a non-finite value at {bad_idx}"
            )));
        }
        Ok(out)
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal GGUF writer, so the tests exercise the parser against bytes it did
    /// not produce itself.
    #[derive(Default)]
    struct Builder {
        kv: Vec<u8>,
        kv_count: u64,
        tensors: Vec<u8>,
        tensor_count: u64,
    }

    impl Builder {
        fn str_bytes(out: &mut Vec<u8>, s: &str) {
            out.extend_from_slice(&(s.len() as u64).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }

        fn kv_string(mut self, key: &str, value: &str) -> Self {
            Self::str_bytes(&mut self.kv, key);
            self.kv.extend_from_slice(&8u32.to_le_bytes());
            Self::str_bytes(&mut self.kv, value);
            self.kv_count += 1;
            self
        }

        fn kv_u32(mut self, key: &str, value: u32) -> Self {
            Self::str_bytes(&mut self.kv, key);
            self.kv.extend_from_slice(&4u32.to_le_bytes());
            self.kv.extend_from_slice(&value.to_le_bytes());
            self.kv_count += 1;
            self
        }

        fn kv_str_array(mut self, key: &str, values: &[&str]) -> Self {
            Self::str_bytes(&mut self.kv, key);
            self.kv.extend_from_slice(&9u32.to_le_bytes()); // ARRAY
            self.kv.extend_from_slice(&8u32.to_le_bytes()); // of STRING
            self.kv
                .extend_from_slice(&(values.len() as u64).to_le_bytes());
            for v in values {
                Self::str_bytes(&mut self.kv, v);
            }
            self.kv_count += 1;
            self
        }

        fn tensor(mut self, name: &str, dims: &[u64], ggml_type: u32, offset: u64) -> Self {
            Self::str_bytes(&mut self.tensors, name);
            self.tensors
                .extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in dims {
                self.tensors.extend_from_slice(&d.to_le_bytes());
            }
            self.tensors.extend_from_slice(&ggml_type.to_le_bytes());
            self.tensors.extend_from_slice(&offset.to_le_bytes());
            self.tensor_count += 1;
            self
        }

        fn build(self) -> Vec<u8> {
            let mut out = Vec::new();
            out.extend_from_slice(MAGIC);
            out.extend_from_slice(&3u32.to_le_bytes());
            out.extend_from_slice(&self.tensor_count.to_le_bytes());
            out.extend_from_slice(&self.kv_count.to_le_bytes());
            out.extend_from_slice(&self.kv);
            out.extend_from_slice(&self.tensors);
            out.resize(out.len().next_multiple_of(32), 0);
            out.extend_from_slice(&[0u8; 64]); // tensor data
            out
        }
    }

    fn mixtral_like() -> Vec<u8> {
        Builder::default()
            .kv_string("general.architecture", "llama")
            .kv_u32("llama.expert_count", 8)
            .kv_u32("llama.expert_used_count", 2)
            .kv_u32("llama.block_count", 32)
            .kv_str_array("tokenizer.ggml.tokens", &["<unk>", "<s>", "hello"])
            .tensor("token_embd.weight", &[4096, 32000], 12, 0)
            .tensor("blk.0.ffn_gate_exps.weight", &[4096, 14336, 8], 12, 1024)
            .build()
    }

    #[test]
    fn parses_a_real_looking_header() {
        let g = Gguf::parse(&mixtral_like()).unwrap();

        assert_eq!(g.version, 3);
        assert_eq!(g.architecture(), Some("llama"));
        assert_eq!(g.expert_count(), Some(8));
        assert_eq!(g.expert_used_count(), Some(2));
        assert_eq!(g.tensors.len(), 2);
        assert_eq!(g.data_offset % 32, 0, "data must start aligned");
    }

    #[test]
    fn reads_tensor_descriptors() {
        let g = Gguf::parse(&mixtral_like()).unwrap();

        let t = g.tensor("blk.0.ffn_gate_exps.weight").unwrap();
        assert_eq!(t.dims, vec![4096, 14336, 8]);
        assert_eq!(t.ggml_type, 12); // Q4_K
        assert_eq!(t.offset, 1024);
        assert_eq!(t.n_elements(), 4096 * 14336 * 8);

        assert!(g.tensor("nonexistent").is_none());
    }

    #[test]
    fn reads_arrays() {
        let g = Gguf::parse(&mixtral_like()).unwrap();
        let Some(Value::Array(tokens)) = g.metadata.get("tokenizer.ggml.tokens") else {
            panic!("token array missing");
        };
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[2].as_str(), Some("hello"));
    }

    #[test]
    fn rejects_a_non_gguf_file() {
        assert!(Gguf::parse(b"not a gguf file at all").is_err());
        assert!(Gguf::parse(&[]).is_err());
        assert!(Gguf::parse(&[0u8; 3]).is_err());
    }

    #[test]
    fn truncation_anywhere_is_an_error_not_a_panic() {
        let full = mixtral_like();
        // The old reader read a 24-byte header and then invented the metadata.
        for cut in 0..full.len() {
            let _ = Gguf::parse(&full[..cut]); // must not panic
        }
        assert!(Gguf::parse(&full[..30]).is_err());
    }

    #[test]
    fn a_lying_length_field_does_not_cause_a_huge_allocation() {
        let mut data = Vec::new();
        data.extend_from_slice(MAGIC);
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // no tensors
        data.extend_from_slice(&1u64.to_le_bytes()); // one kv
        data.extend_from_slice(&u64::MAX.to_le_bytes()); // key length: absurd

        let err = Gguf::parse(&data).unwrap_err();
        assert!(matches!(err, GarudaError::Model(_)));
    }

    #[test]
    fn absurd_header_counts_are_rejected() {
        let mut data = Vec::new();
        data.extend_from_slice(MAGIC);
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&u64::MAX.to_le_bytes()); // tensor_count
        data.extend_from_slice(&0u64.to_le_bytes());
        assert!(Gguf::parse(&data).is_err());
    }

    #[test]
    fn an_unsupported_version_is_rejected() {
        let mut data = mixtral_like();
        data[4..8].copy_from_slice(&99u32.to_le_bytes());
        assert!(Gguf::parse(&data).is_err());
    }
}

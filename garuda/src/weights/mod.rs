//! Model weights: deterministic synthesis, and the on-disk expert format.
//!
//! Garuda has no trained checkpoint. Rather than fake the *math* — which is what
//! a "simulated" forward pass does — it runs the real arithmetic over weights
//! that are pseudo-random but fully deterministic in the seed. Every run of a
//! given `(seed, dims)` produces bit-identical tensors, so benchmarks and tests
//! are reproducible, and the output is meaningless text (untrained weights) by
//! construction rather than by accident.
//!
//! Swapping in a real checkpoint means replacing [`ModelWeights::synthesize`]
//! and [`expert_from_bytes`] with a GGUF tensor reader. Nothing else moves.

use crate::core::{Expert, ExpertId, GarudaError, ModelDims};

/// splitmix64: small, fast, and reproducible across platforms.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `n` values drawn uniformly from `[-bound, bound]`, deterministic in `seed`.
fn seeded(seed: u64, n: usize, bound: f32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            // Top 24 bits give an exactly-representable f32 mantissa in [0, 1).
            let u = (splitmix64(&mut state) >> 40) as f32 / (1u32 << 24) as f32;
            (u * 2.0 - 1.0) * bound
        })
        .collect()
}

/// Xavier-ish bound for a layer with `fan_in` inputs.
fn bound(fan_in: usize) -> f32 {
    1.0 / (fan_in as f32).sqrt()
}

/// Distinct per-tensor seeds, so no two matrices are accidentally identical.
mod seeds {
    pub const EMBED: u64 = 0x6761_7275_6461_0001; // "garuda" + tag
    pub const WQ: u64 = 0x6761_7275_6461_0002;
    pub const WK: u64 = 0x6761_7275_6461_0003;
    pub const WV: u64 = 0x6761_7275_6461_0004;
    pub const WO: u64 = 0x6761_7275_6461_0005;
    pub const ROUTER: u64 = 0x6761_7275_6461_0006;
    pub const EXPERT: u64 = 0x6761_7275_6461_1000;
}

/// The non-expert weights: everything shared across the MoE layer.
#[derive(Debug)]
pub struct ModelWeights {
    pub dims: ModelDims,
    /// `[vocab_size, d_model]`, tied to the output head.
    pub embedding: Vec<f32>,
    /// `[d_model, d_model]` each.
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    /// `[n_experts, d_model]` router gate.
    pub router: Vec<f32>,
}

impl ModelWeights {
    /// Build the shared weights for `dims`. Deterministic; no I/O.
    pub fn synthesize(dims: ModelDims) -> Result<Self, GarudaError> {
        dims.validate()?;
        let d = dims.d_model;
        let b = bound(d);
        Ok(Self {
            embedding: seeded(seeds::EMBED, dims.vocab_size * d, bound(d)),
            wq: seeded(seeds::WQ, d * d, b),
            wk: seeded(seeds::WK, d * d, b),
            wv: seeded(seeds::WV, d * d, b),
            wo: seeded(seeds::WO, d * d, b),
            router: seeded(seeds::ROUTER, dims.n_experts * d, b),
            dims,
        })
    }

    /// Row `token` of the embedding table.
    pub fn embed(&self, token: crate::core::Token) -> Result<&[f32], GarudaError> {
        let d = self.dims.d_model;
        let idx = token as usize;
        if idx >= self.dims.vocab_size {
            return Err(GarudaError::InvalidToken(token));
        }
        Ok(&self.embedding[idx * d..(idx + 1) * d])
    }
}

/// Deterministic weights for one expert. Same `(id, dims)` always yields the
/// same tensors, on any machine.
pub fn synthesize_expert(id: ExpertId, dims: &ModelDims) -> Expert {
    let (d, f) = (dims.d_model, dims.d_ff);
    let seed = seeds::EXPERT ^ ((id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    Expert {
        id,
        gate: seeded(seed, f * d, bound(d)),
        up: seeded(seed ^ 0xAAAA_AAAA, f * d, bound(d)),
        down: seeded(seed ^ 0x5555_5555, d * f, bound(f)),
        dims: *dims,
        loaded_at: std::time::Instant::now(),
    }
}

/// On-disk expert layout: little-endian f32, `gate` then `up` then `down`.
pub fn expert_to_bytes(expert: &Expert) -> Vec<u8> {
    let mut out = Vec::with_capacity(expert.size_bytes());
    for chunk in [&expert.gate, &expert.up, &expert.down] {
        for v in chunk {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// Parse the layout written by [`expert_to_bytes`].
///
/// Rejects any file whose length is not exactly what `dims` requires, rather
/// than silently truncating to whatever happened to fit.
pub fn expert_from_bytes(
    id: ExpertId,
    dims: &ModelDims,
    bytes: &[u8],
) -> Result<Expert, GarudaError> {
    let (d, f) = (dims.d_model, dims.d_ff);
    let n = Expert::n_params(dims);
    let expected = n * std::mem::size_of::<f32>();
    if bytes.len() != expected {
        return Err(GarudaError::Model(format!(
            "expert {id}: expected {expected} bytes for dims {d}x{f}, found {}",
            bytes.len()
        )));
    }

    // `align_to` would be UB-adjacent on a misaligned mmap; decode explicitly.
    let mut vals = Vec::with_capacity(n);
    for c in bytes.chunks_exact(4) {
        vals.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
    if let Some(bad) = vals.iter().position(|v| !v.is_finite()) {
        return Err(GarudaError::Model(format!(
            "expert {id}: non-finite weight at index {bad}"
        )));
    }

    let down = vals.split_off(2 * f * d);
    let up = vals.split_off(f * d);
    let gate = vals;

    Ok(Expert {
        id,
        gate,
        up,
        down,
        dims: *dims,
        loaded_at: std::time::Instant::now(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthesis_is_deterministic_and_distinct_per_expert() {
        let dims = ModelDims::default();
        let a = synthesize_expert(3, &dims);
        let b = synthesize_expert(3, &dims);
        let c = synthesize_expert(4, &dims);

        assert_eq!(a.gate, b.gate, "same id must reproduce bit-identically");
        assert_eq!(a.down, b.down);
        assert_ne!(a.gate, c.gate, "different ids must differ");
        assert_ne!(a.gate, a.up, "tensors within an expert must differ");
    }

    #[test]
    fn shared_weights_have_the_shapes_the_dims_promise() {
        let dims = ModelDims::default();
        let w = ModelWeights::synthesize(dims).unwrap();
        assert_eq!(w.embedding.len(), dims.vocab_size * dims.d_model);
        assert_eq!(w.wq.len(), dims.d_model * dims.d_model);
        assert_eq!(w.router.len(), dims.n_experts * dims.d_model);
        assert!(w.embedding.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn expert_bytes_round_trip() {
        let dims = ModelDims::default();
        let e = synthesize_expert(7, &dims);
        let bytes = expert_to_bytes(&e);
        let back = expert_from_bytes(7, &dims, &bytes).unwrap();
        assert_eq!(e.gate, back.gate);
        assert_eq!(e.up, back.up);
        assert_eq!(e.down, back.down);
    }

    #[test]
    fn truncated_expert_file_is_rejected_not_truncated() {
        let dims = ModelDims::default();
        let e = synthesize_expert(1, &dims);
        let mut bytes = expert_to_bytes(&e);
        bytes.truncate(bytes.len() - 4);
        assert!(matches!(
            expert_from_bytes(1, &dims, &bytes),
            Err(GarudaError::Model(_))
        ));
        // The old loader silently kept the first 100 floats of whatever it found.
        assert!(expert_from_bytes(1, &dims, &[0u8, 1, 2]).is_err());
        assert!(expert_from_bytes(1, &dims, &[]).is_err());
    }

    #[test]
    fn non_finite_weights_are_rejected() {
        let dims = ModelDims::default();
        let e = synthesize_expert(2, &dims);
        let mut bytes = expert_to_bytes(&e);
        bytes[0..4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(expert_from_bytes(2, &dims, &bytes).is_err());
    }

    #[test]
    fn embed_rejects_out_of_vocab_token() {
        let dims = ModelDims::default();
        let w = ModelWeights::synthesize(dims).unwrap();
        assert!(w.embed(0).is_ok());
        assert!(w.embed(dims.vocab_size as u32).is_err());
    }
}

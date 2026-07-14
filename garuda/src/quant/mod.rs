//! GGUF weight dequantisation.
//!
//! GGUF stores quantised tensors in fixed-size blocks: a scale (and sometimes a
//! minimum) shared by a run of low-bit integers. This module turns those blocks
//! back into `f32`. It is the first step toward running the quantised checkpoints
//! people actually download — the ones this runtime previously rejected outright.
//!
//! Supported today: `F32`, `F16`, `Q4_0`, `Q8_0` — the identity and the two
//! simplest linear quants, which cover the legacy `q4_0`/`q8_0` model files whole.
//! The k-quants (`Q4_K`, `Q6_K`, …) that dominate modern downloads use super-blocks
//! with 6-bit sub-scales and are not decoded yet; [`block_layout`] returns `None`
//! for them so the loader errors clearly rather than producing garbage.
//!
//! Today the whole tensor is expanded to `f32` at load. Keeping weights packed and
//! multiplying with an integer kernel — the trick that lets a model larger than RAM
//! run — is a later phase; this module is the correctness foundation it builds on.

use crate::core::GarudaError;

// ggml type ids (a subset of the full enum).
pub const F32: u32 = 0;
pub const F16: u32 = 1;
pub const Q4_0: u32 = 2;
pub const Q8_0: u32 = 8;

/// Elements per block for the quantised types.
const QK: usize = 32;

/// `(elements_per_block, bytes_per_block)` for a supported type, or `None`.
///
/// `F32`/`F16` are treated as one-element "blocks" so the same length arithmetic
/// covers every supported type.
pub fn block_layout(ggml_type: u32) -> Option<(usize, usize)> {
    match ggml_type {
        F32 => Some((1, 4)),
        F16 => Some((1, 2)),
        // block_q4_0: f16 scale + 32 4-bit quants packed into 16 bytes.
        Q4_0 => Some((QK, 2 + QK / 2)),
        // block_q8_0: f16 scale + 32 int8 quants.
        Q8_0 => Some((QK, 2 + QK)),
        _ => None,
    }
}

pub fn is_supported(ggml_type: u32) -> bool {
    block_layout(ggml_type).is_some()
}

/// Human name for a ggml type, for error messages.
pub fn type_name(ggml_type: u32) -> &'static str {
    match ggml_type {
        F32 => "F32",
        F16 => "F16",
        Q4_0 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        Q8_0 => "Q8_0",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        _ => "unknown",
    }
}

/// The exact number of bytes `n` elements of `ggml_type` occupy on disk.
///
/// Errors if `ggml_type` is unsupported, or if `n` is not a whole number of blocks.
pub fn byte_size(ggml_type: u32, n: usize) -> Result<usize, GarudaError> {
    let (elems, bytes) = block_layout(ggml_type).ok_or_else(|| unsupported(ggml_type))?;
    if n % elems != 0 {
        return Err(GarudaError::Model(format!(
            "{} tensor has {n} elements, not a multiple of the block size {elems}",
            type_name(ggml_type)
        )));
    }
    Ok((n / elems) * bytes)
}

/// Decode exactly `n` elements of `ggml_type` from `raw` into `f32`.
///
/// `raw` must be exactly [`byte_size`] long. Non-finite results are rejected, so a
/// corrupt block is an error rather than a `NaN` that silently poisons inference.
pub fn dequantize(ggml_type: u32, raw: &[u8], n: usize) -> Result<Vec<f32>, GarudaError> {
    let expected = byte_size(ggml_type, n)?;
    if raw.len() != expected {
        return Err(GarudaError::Model(format!(
            "{} tensor: expected {expected} bytes for {n} elements, got {}",
            type_name(ggml_type),
            raw.len()
        )));
    }

    let out = match ggml_type {
        F32 => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        F16 => raw
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        Q8_0 => dequant_q8_0(raw, n),
        Q4_0 => dequant_q4_0(raw, n),
        _ => return Err(unsupported(ggml_type)),
    };

    if let Some(bad) = out.iter().position(|v| !v.is_finite()) {
        return Err(GarudaError::Model(format!(
            "{} tensor produced a non-finite value at index {bad}",
            type_name(ggml_type)
        )));
    }
    Ok(out)
}

/// `block_q8_0`: `[ f16 scale | int8 q[0..32] ]`, dequant `y = scale * q`.
fn dequant_q8_0(raw: &[u8], n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    for block in raw.chunks_exact(2 + QK) {
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for &q in &block[2..] {
            out.push(scale * (q as i8) as f32);
        }
    }
    out
}

/// `block_q4_0`: `[ f16 scale | u8 q[0..16] ]`. Each byte holds two 4-bit weights;
/// the low nibble is element `i`, the high nibble element `i + 16`, and each is
/// centred by subtracting 8. `y = scale * (nibble - 8)`.
fn dequant_q4_0(raw: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for (b, block) in raw.chunks_exact(2 + QK / 2).enumerate() {
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let base = b * QK;
        for (i, &byte) in block[2..].iter().enumerate() {
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = (byte >> 4) as i32 - 8;
            out[base + i] = scale * lo as f32;
            out[base + i + QK / 2] = scale * hi as f32;
        }
    }
    out
}

/// IEEE-754 half → single precision.
pub fn f16_to_f32(h: u16) -> f32 {
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

fn unsupported(ggml_type: u32) -> GarudaError {
    GarudaError::Model(format!(
        "tensor type {} ({ggml_type}) is not supported; only F32, F16, Q4_0 and Q8_0 \
         decode today (the k-quants Q4_K/Q6_K/… need a super-block decoder that does \
         not exist yet)",
        type_name(ggml_type)
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Little-endian f16 bytes for a few exact values.
    fn f16(v: f32) -> [u8; 2] {
        let bits: u16 = if v == 0.5 {
            0x3800
        } else if v == 1.0 {
            0x3C00
        } else if v == 2.0 {
            0x4000
        } else {
            panic!("add the f16 encoding for {v}")
        };
        bits.to_le_bytes()
    }

    #[test]
    fn f16_decoder_matches_known_values() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x3800), 0.5);
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
    }

    #[test]
    fn block_layouts_are_correct() {
        assert_eq!(block_layout(Q8_0), Some((32, 34)));
        assert_eq!(block_layout(Q4_0), Some((32, 18)));
        assert_eq!(block_layout(F16), Some((1, 2)));
        assert_eq!(block_layout(12), None); // Q4_K unsupported
        assert!(!is_supported(14)); // Q6_K
    }

    #[test]
    fn q8_0_dequantises_to_scale_times_q() {
        // one block: scale 0.5, q = 0,1,-2,3, then zeros
        let mut raw = Vec::new();
        raw.extend_from_slice(&f16(0.5));
        let mut qs = [0i8; 32];
        qs[0] = 0;
        qs[1] = 1;
        qs[2] = -2;
        qs[3] = 3;
        raw.extend(qs.iter().map(|&q| q as u8));

        let y = dequantize(Q8_0, &raw, 32).unwrap();
        assert_eq!(y[0], 0.0);
        assert_eq!(y[1], 0.5);
        assert_eq!(y[2], -1.0);
        assert_eq!(y[3], 1.5);
    }

    #[test]
    fn q4_0_uses_the_low_then_high_nibble_layout() {
        // scale 1.0; byte 0 = 0x80 -> low nibble 0 (->-8), high nibble 8 (->0)
        let mut raw = Vec::new();
        raw.extend_from_slice(&f16(1.0));
        let mut qs = [0u8; 16];
        qs[0] = 0x80; // element 0 <- 0x0, element 16 <- 0x8
        qs[1] = 0x0F; // element 1 <- 0xF (->7), element 17 <- 0x0 (->-8)
        raw.extend_from_slice(&qs);

        let y = dequantize(Q4_0, &raw, 32).unwrap();
        assert_eq!(y[0], -8.0); // (0 - 8) * 1.0
        assert_eq!(y[16], 0.0); // (8 - 8) * 1.0
        assert_eq!(y[1], 7.0); // (15 - 8)
        assert_eq!(y[17], -8.0); // (0 - 8)
    }

    #[test]
    fn q8_0_round_trips_within_quantisation_error() {
        // Quantise a known vector the way llama.cpp does, then dequantise it back.
        let orig: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.37).collect();
        let amax = orig.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let scale = amax / 127.0;

        let mut raw = Vec::new();
        // encode scale as f16 (approx via bit twiddling is overkill; use 2.0-scaled known)
        // Instead store scale exactly by round-tripping through f32->f16 is complex here,
        // so assert the *shape* of the error using the scale we used.
        raw.extend_from_slice(&half_bits(scale).to_le_bytes());
        let decoded_scale = f16_to_f32(half_bits(scale));
        for &v in &orig {
            let q = (v / scale).round().clamp(-127.0, 127.0) as i8;
            raw.push(q as u8);
        }

        let y = dequantize(Q8_0, &raw, 32).unwrap();
        for (a, b) in orig.iter().zip(y.iter()) {
            // error bounded by one quantisation step of the (f16-rounded) scale
            assert!((a - b).abs() <= decoded_scale + 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn wrong_length_and_unsupported_type_are_errors() {
        assert!(dequantize(Q8_0, &[0u8; 10], 32).is_err()); // too short
        assert!(dequantize(12, &[0u8; 100], 32).is_err()); // Q4_K unsupported
        assert!(byte_size(Q8_0, 33).is_err()); // not a whole block
    }

    /// Minimal f32 -> f16 for the round-trip test (round-to-nearest, normals only).
    fn half_bits(v: f32) -> u16 {
        if v == 0.0 {
            return 0;
        }
        let bits = v.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        let mant = bits & 0x7f_ffff;
        if exp <= 0 {
            return sign; // flush tiny values to zero — fine for this test
        }
        if exp >= 0x1f {
            return sign | 0x7c00;
        }
        sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
    }
}

//! GGUF weight dequantisation.
//!
//! GGUF stores quantised tensors in fixed-size blocks: a scale (and sometimes a
//! minimum) shared by a run of low-bit integers. This module turns those blocks
//! back into `f32`. It is the first step toward running the quantised checkpoints
//! people actually download — the ones this runtime previously rejected outright.
//!
//! Supported today: `F32`, `F16`, the linear quants `Q4_0`/`Q8_0`, and the k-quants
//! `Q4_K`/`Q6_K` — the latter two are the mix a `*_K_M` GGUF uses, so those files
//! load whole. The remaining k-quants (`Q2_K`, `Q3_K`, `Q5_K`) and the `*_1` linear
//! quants are not decoded yet; [`block_layout`] returns `None` for them so the loader
//! errors clearly rather than producing garbage.
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
pub const Q4_K: u32 = 12;
pub const Q6_K: u32 = 14;

/// Elements per block for the legacy linear quants.
const QK: usize = 32;
/// Elements per super-block for the k-quants.
const QK_K: usize = 256;

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
        // block_q4_K: 2×f16 (scale, min) + 12 packed 6-bit sub-scales + 128 4-bit quants.
        Q4_K => Some((QK_K, 2 + 2 + 12 + QK_K / 2)),
        // block_q6_K: 128 low-nibble + 64 high-2-bit + 16 int8 sub-scales + f16 scale.
        Q6_K => Some((QK_K, QK_K / 2 + QK_K / 4 + QK_K / 16 + 2)),
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
        Q4_K => dequant_q4_k(raw, n),
        Q6_K => dequant_q6_k(raw, n),
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

/// Unpack one 6-bit sub-scale and sub-min from a `block_q4_K` `scales[12]` array.
///
/// The 8 scales and 8 mins are packed 6 bits each across 12 bytes; the first four
/// of each sit in their own byte, the last four borrow the top 2 bits of an earlier
/// byte. This mirrors ggml's `get_scale_min_k4` exactly — get it wrong and the whole
/// tensor is garbage.
fn get_scale_min_k4(j: usize, s: &[u8]) -> (u8, u8) {
    if j < 4 {
        (s[j] & 63, s[j + 4] & 63)
    } else {
        let d = (s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4);
        let m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
        (d, m)
    }
}

/// `block_q4_K` (super-block of 256): `[ f16 d | f16 dmin | u8 scales[12] | u8 qs[128] ]`.
///
/// Eight 32-element sub-blocks, each with its own 6-bit scale and min. A weight is
/// `d·scale·q − dmin·min`, where `q` is a 4-bit quant. `min` is subtracted, so this
/// is an affine (not symmetric) quant.
fn dequant_q4_k(raw: &[u8], n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    for block in raw.chunks_exact(144) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let qs = &block[16..144];

        let mut is = 0;
        for chunk in qs.chunks_exact(32) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let (d1, mn1) = (d * sc1 as f32, dmin * m1 as f32);
            let (d2, mn2) = (d * sc2 as f32, dmin * m2 as f32);
            for &q in chunk {
                out.push(d1 * (q & 0x0F) as f32 - mn1);
            }
            for &q in chunk {
                out.push(d2 * (q >> 4) as f32 - mn2);
            }
            is += 2;
        }
    }
    out
}

/// `block_q6_K` (super-block of 256): `[ u8 ql[128] | u8 qh[64] | i8 scales[16] | f16 d ]`.
///
/// Sixteen 16-element sub-blocks. Each 6-bit quant is assembled from 4 low bits in
/// `ql` and 2 high bits in `qh`, centred by subtracting 32, then scaled by its int8
/// sub-scale times the super-block `d`. The interleaving follows ggml's
/// `dequantize_row_q6_K`.
fn dequant_q6_k(raw: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for (bi, block) in raw.chunks_exact(210).enumerate() {
        let ql = &block[0..128];
        let qh = &block[128..192];
        let sc = &block[192..208];
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        let sb = bi * QK_K;

        // Two halves of 128 elements each.
        for half in 0..2 {
            let ql = &ql[half * 64..];
            let qh = &qh[half * 32..];
            let sc = &sc[half * 8..];
            let y = sb + half * 128;
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0x0F) | ((qh[l] & 3) << 4)) as i32 - 32;
                let q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                out[y + l] = d * (sc[is] as i8) as f32 * q1 as f32;
                out[y + l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                out[y + l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                out[y + l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
            }
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
        "tensor type {} ({ggml_type}) is not supported; F32, F16, Q4_0, Q8_0, Q4_K and \
         Q6_K decode today (Q2_K/Q3_K/Q5_K and the *_1 quants need their own decoders)",
        type_name(ggml_type)
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Little-endian f16 bytes for a few exact values.
    fn f16(v: f32) -> [u8; 2] {
        let bits: u16 = if v == 0.0 {
            0x0000
        } else if v == 0.5 {
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
        assert_eq!(block_layout(Q4_K), Some((256, 144)));
        assert_eq!(block_layout(Q6_K), Some((256, 210)));
        assert_eq!(block_layout(13), None); // Q5_K not decoded yet
        assert!(!is_supported(10)); // Q2_K
    }

    #[test]
    fn q6_k_assembles_6bit_quants_centred_at_32() {
        // d = 1.0, all int8 sub-scales = 1. With all quant bits zero, every 6-bit
        // value is 0, so every output is (0 - 32) * 1 * 1 = -32.
        let mut raw = vec![0u8; 210];
        for s in raw[192..208].iter_mut() {
            *s = 1;
        }
        raw[208..210].copy_from_slice(&f16(1.0));

        // Set element 0's low nibble to 5 -> value 5, output = 5 - 32 = -27.
        raw[0] = 0x05;

        let y = dequantize(Q6_K, &raw, 256).unwrap();
        assert_eq!(y[0], -27.0);
        assert!(y[1..].iter().all(|&v| v == -32.0));
    }

    #[test]
    fn q4_k_reads_nibbles_scaled_by_the_subblock_scale() {
        // d = 1.0, dmin = 0.0 (so the min term drops out), sub-block scale = 1.
        let mut raw = vec![0u8; 144];
        raw[0..2].copy_from_slice(&f16(1.0)); // d
        raw[2..4].copy_from_slice(&f16(0.0)); // dmin
        raw[4] = 1; // scales[0]: get_scale_min_k4(0) -> sc = 1
        raw[5] = 1; // scales[1]: sc = 1 for the second sub-block too
                    // qs[0] = 0x30: low nibble 0 (element 0), high nibble 3 (element 32)
        raw[16] = 0x30;

        let y = dequantize(Q4_K, &raw, 256).unwrap();
        assert_eq!(y[0], 0.0); // 1 * 0
        assert_eq!(y[32], 3.0); // 1 * 3 (high nibble, second 32-run of the group)
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

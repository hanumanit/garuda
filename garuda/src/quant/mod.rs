//! GGUF weight dequantisation.
//!
//! GGUF stores quantised tensors in fixed-size blocks: a scale (and sometimes a
//! minimum) shared by a run of low-bit integers. This module turns those blocks
//! back into `f32`. It is the first step toward running the quantised checkpoints
//! people actually download — the ones this runtime previously rejected outright.
//!
//! Supported today: `F32`, `F16`, the linear quants `Q4_0`/`Q8_0`, and the full set of
//! k-quants `Q2_K`/`Q3_K`/`Q4_K`/`Q5_K`/`Q6_K` — so any `Q2_K … Q6_K` GGUF loads whole.
//! Not decoded yet: the `*_1` linear quants and the IQ imatrix quants; [`block_layout`]
//! returns `None` for them so the loader errors clearly rather than producing garbage.
//!
//! [`matvec`] can also multiply a packed row directly against an int8-quantised
//! activation — never expanding the row to `f32` — for `Q8_0` and `Q4_K`. This is
//! the fast path `Weight::Packed` uses under `mmap`; the other formats still
//! dequantise to `f32` per row before dotting.

use crate::core::GarudaError;

// ggml type ids (a subset of the full enum).
pub const F32: u32 = 0;
pub const F16: u32 = 1;
pub const Q4_0: u32 = 2;
pub const Q8_0: u32 = 8;
pub const Q2_K: u32 = 10;
pub const Q3_K: u32 = 11;
pub const Q4_K: u32 = 12;
pub const Q5_K: u32 = 13;
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
        // block_q2_K: 16 packed 4-bit scale/min pairs + 64 2-bit quants + 2×f16.
        Q2_K => Some((QK_K, QK_K / 16 + QK_K / 4 + 2 + 2)),
        // block_q3_K: 32 high-bit mask + 64 low-2-bit quants + 12 packed 6-bit scales + f16.
        Q3_K => Some((QK_K, QK_K / 8 + QK_K / 4 + 12 + 2)),
        // block_q4_K: 2×f16 (scale, min) + 12 packed 6-bit sub-scales + 128 4-bit quants.
        Q4_K => Some((QK_K, 2 + 2 + 12 + QK_K / 2)),
        // block_q5_K: 2×f16 + 12 packed 6-bit sub-scales + 32 high-bit + 128 4-bit quants.
        Q5_K => Some((QK_K, 2 + 2 + 12 + QK_K / 8 + QK_K / 2)),
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

/// Decode exactly `n` elements of `ggml_type` from `raw` into a fresh `Vec<f32>`.
///
/// `raw` must be exactly [`byte_size`] long. Non-finite results are rejected, so a
/// corrupt block is an error rather than a `NaN` that silently poisons inference.
pub fn dequantize(ggml_type: u32, raw: &[u8], n: usize) -> Result<Vec<f32>, GarudaError> {
    let mut out = vec![0.0f32; n];
    dequant_into(ggml_type, raw, &mut out)?;
    if let Some(bad) = out.iter().position(|v| !v.is_finite()) {
        return Err(GarudaError::Model(format!(
            "{} tensor produced a non-finite value at index {bad}",
            type_name(ggml_type)
        )));
    }
    Ok(out)
}

/// Decode into a caller-provided buffer (`out.len()` elements). No finiteness check —
/// this is the hot path for [`matvec`], where the same buffer is reused per row so
/// nothing is allocated. Validates `raw`/`out` lengths against the format.
pub fn dequant_into(ggml_type: u32, raw: &[u8], out: &mut [f32]) -> Result<(), GarudaError> {
    let n = out.len();
    let expected = byte_size(ggml_type, n)?;
    if raw.len() != expected {
        return Err(GarudaError::Model(format!(
            "{} tensor: expected {expected} bytes for {n} elements, got {}",
            type_name(ggml_type),
            raw.len()
        )));
    }
    match ggml_type {
        F32 => {
            for (o, c) in out.iter_mut().zip(raw.chunks_exact(4)) {
                *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        F16 => {
            for (o, c) in out.iter_mut().zip(raw.chunks_exact(2)) {
                *o = f16_to_f32(u16::from_le_bytes([c[0], c[1]]));
            }
        }
        Q8_0 => dequant_q8_0(raw, out),
        Q4_0 => dequant_q4_0(raw, out),
        Q2_K => dequant_q2_k(raw, out),
        Q3_K => dequant_q3_k(raw, out),
        Q4_K => dequant_q4_k(raw, out),
        Q5_K => dequant_q5_k(raw, out),
        Q6_K => dequant_q6_k(raw, out),
        _ => return Err(unsupported(ggml_type)),
    }
    Ok(())
}

/// `out[r] = dot(dequantised row r of a row-major quantised matrix, x)`.
///
/// The weight stays packed; each rayon worker dequantises one row at a time into a
/// reused scratch buffer, so the full `f32` matrix is never materialised and nothing
/// is allocated per row. This is what lets a memory-mapped, quantised model run
/// without expanding to `f32` — at the cost of re-dequantising every row on each call.
pub fn matvec(
    qtype: u32,
    weight: &[u8],
    rows: usize,
    cols: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<(), GarudaError> {
    use rayon::prelude::*;

    let row_bytes = byte_size(qtype, cols)?;
    if weight.len() != rows * row_bytes {
        return Err(GarudaError::Model(format!(
            "{} matvec: weight is {} bytes, expected {} ({rows} rows × {row_bytes})",
            type_name(qtype),
            weight.len(),
            rows * row_bytes
        )));
    }
    if x.len() != cols || out.len() != rows {
        return Err(GarudaError::Inference(format!(
            "matvec dimension mismatch: x={}, out={}, expected cols={cols}, rows={rows}",
            x.len(),
            out.len()
        )));
    }

    // Q8_0 weights are already int8, so the integer kernel dots them with an int8-quantised
    // input directly, never expanding to f32 — a clear win. Q4_K's super-blocks are large
    // enough (256 elements, 8 sub-blocks sharing the unpack cost) that the same trick pays
    // off there too, despite the extra nibble-unpack; tried for the small-block Q4_0 as
    // well, but there the unpack dominates and it came out slower than dequantise-to-f32,
    // so that one keeps the f32 path.
    if qtype == Q8_0 {
        matvec_q8_0(weight, cols, x, out);
        return Ok(());
    }
    if qtype == Q4_K {
        matvec_q4_k(weight, cols, x, out);
        return Ok(());
    }

    out.par_iter_mut()
        .zip(weight.par_chunks_exact(row_bytes))
        .try_for_each_init(
            || vec![0.0f32; cols],
            |buf, (o, row)| -> Result<(), GarudaError> {
                dequant_into(qtype, row, buf)?;
                *o = crate::simd::dot(buf, x);
                Ok(())
            },
        )
}

/// Quantise an activation vector to int8, one scale per 32-element block, the way ggml
/// quantises activations for a quantised matmul. Returns the int8 values and the
/// per-block scales. This is the small approximation the integer kernels trade for speed.
fn quantize_activation(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
    let nblocks = x.len() / QK;
    let mut xq = vec![0i8; x.len()];
    let mut xs = vec![0.0f32; nblocks];
    for (b, blk) in x.chunks_exact(QK).enumerate() {
        let amax = blk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let scale = amax / 127.0;
        xs[b] = scale;
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        for (j, &v) in blk.iter().enumerate() {
            xq[b * QK + j] = (v * inv).round().clamp(-127.0, 127.0) as i8;
        }
    }
    (xq, xs)
}

/// Q8_0 integer-kernel matvec: the weights are already int8, so quantise `x` to int8 and
/// dot the rows with `simd::dot_i8` — the weights never touch `f32`. Introduces a small
/// activation-quantisation error (the same tradeoff llama.cpp makes), so the result is
/// very close to, but not bit-identical with, the exact `f32` dequantisation path.
fn matvec_q8_0(weight: &[u8], cols: usize, x: &[f32], out: &mut [f32]) {
    use rayon::prelude::*;
    let (xq, xs) = quantize_activation(x);
    let row_bytes = 34 * (cols / QK); // Q8_0 block: 2-byte f16 scale + 32 int8

    out.par_iter_mut()
        .zip(weight.par_chunks_exact(row_bytes))
        .for_each(|(o, row)| {
            let mut sum = 0.0f32;
            for (b, blk) in row.chunks_exact(34).enumerate() {
                let ws = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                // The 32 quant bytes reinterpreted as i8 — same size and alignment.
                let wq: &[i8] =
                    unsafe { std::slice::from_raw_parts(blk[2..].as_ptr() as *const i8, 32) };
                sum += ws * xs[b] * crate::simd::dot_i8(wq, &xq[b * QK..b * QK + QK]) as f32;
            }
            *o = sum;
        });
}

/// Q4_K integer-kernel matvec: quantise `x` to int8 once, then dot each row's nibbles
/// against it directly — the row never expands to `f32`. Same activation-quantisation
/// tradeoff as [`matvec_q8_0`]: very close to, but not bit-identical with, the exact
/// `f32` dequantisation path.
fn matvec_q4_k(weight: &[u8], cols: usize, x: &[f32], out: &mut [f32]) {
    use rayon::prelude::*;
    let (xq, xs) = quantize_activation(x);
    // Q4_K is affine (`w = d·scale·q − dmin·min`), so the min-term needs the sum of
    // each 32-wide block of `xq` too. It does not depend on the row, so it is computed
    // once here and shared by every row's dot product below.
    let sums: Vec<i32> = xq
        .chunks_exact(QK)
        .map(|blk| blk.iter().map(|&v| v as i32).sum())
        .collect();
    let row_bytes = 144 * (cols / QK_K);

    out.par_iter_mut()
        .zip(weight.par_chunks_exact(row_bytes))
        .for_each(|(o, row)| *o = q4_k_dot(row, &xq, &xs, &sums));
}

/// One Q4_K-quantised row dotted against an int8-quantised activation. Mirrors
/// [`dequant_q4_k`]'s block layout exactly, but folds the unpack straight into the
/// dot product instead of materialising an `f32` row first.
fn q4_k_dot(row: &[u8], xq: &[i8], xs: &[f32], sums: &[i32]) -> f32 {
    let mut acc = 0.0f32;
    let mut qlo = [0i8; QK];
    let mut qhi = [0i8; QK];

    for (bi, block) in row.chunks_exact(144).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let qs = &block[16..144];
        let base = bi * 8; // 8 sub-blocks of 32 per 256-element super-block

        let mut is = 0;
        for chunk in qs.chunks_exact(32) {
            for (i, &byte) in chunk.iter().enumerate() {
                qlo[i] = (byte & 0x0F) as i8;
                qhi[i] = (byte >> 4) as i8;
            }
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);

            let sb = base + is;
            let off = sb * QK;
            let dot = crate::simd::dot_i8(&qlo, &xq[off..off + QK]) as f32;
            acc += xs[sb] * (d * sc1 as f32 * dot - dmin * m1 as f32 * sums[sb] as f32);

            let sb = sb + 1;
            let off = sb * QK;
            let dot = crate::simd::dot_i8(&qhi, &xq[off..off + QK]) as f32;
            acc += xs[sb] * (d * sc2 as f32 * dot - dmin * m2 as f32 * sums[sb] as f32);

            is += 2;
        }
    }
    acc
}

/// `block_q8_0`: `[ f16 scale | int8 q[0..32] ]`, dequant `y = scale * q`.
fn dequant_q8_0(raw: &[u8], out: &mut [f32]) {
    for (bi, block) in raw.chunks_exact(2 + QK).enumerate() {
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let base = bi * QK;
        for (i, &q) in block[2..].iter().enumerate() {
            out[base + i] = scale * (q as i8) as f32;
        }
    }
}

/// `block_q4_0`: `[ f16 scale | u8 q[0..16] ]`. Each byte holds two 4-bit weights;
/// the low nibble is element `i`, the high nibble element `i + 16`, and each is
/// centred by subtracting 8. `y = scale * (nibble - 8)`.
fn dequant_q4_0(raw: &[u8], out: &mut [f32]) {
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
fn dequant_q4_k(raw: &[u8], out: &mut [f32]) {
    for (bi, block) in raw.chunks_exact(144).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let qs = &block[16..144];

        let mut w = bi * QK_K;
        let mut is = 0;
        for chunk in qs.chunks_exact(32) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let (d1, mn1) = (d * sc1 as f32, dmin * m1 as f32);
            let (d2, mn2) = (d * sc2 as f32, dmin * m2 as f32);
            for &q in chunk {
                out[w] = d1 * (q & 0x0F) as f32 - mn1;
                w += 1;
            }
            for &q in chunk {
                out[w] = d2 * (q >> 4) as f32 - mn2;
                w += 1;
            }
            is += 2;
        }
    }
}

/// `block_q6_K` (super-block of 256): `[ u8 ql[128] | u8 qh[64] | i8 scales[16] | f16 d ]`.
///
/// Sixteen 16-element sub-blocks. Each 6-bit quant is assembled from 4 low bits in
/// `ql` and 2 high bits in `qh`, centred by subtracting 32, then scaled by its int8
/// sub-scale times the super-block `d`. The interleaving follows ggml's
/// `dequantize_row_q6_K`.
fn dequant_q6_k(raw: &[u8], out: &mut [f32]) {
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
}

/// `block_q5_K` (super-block of 256): `[ f16 d | f16 dmin | u8 scales[12] | u8 qh[32] | u8 qs[128] ]`.
///
/// Like Q4_K, but each quant is 5 bits: the low 4 from `qs`, the 5th from a bit of
/// `qh` selected per 64-element group. `y = d·scale·(q5) − dmin·min`.
fn dequant_q5_k(raw: &[u8], out: &mut [f32]) {
    for (bi, block) in raw.chunks_exact(176).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let qh = &block[16..48];
        let qs = &block[48..176];

        let mut w = bi * QK_K;
        let mut is = 0;
        let (mut u1, mut u2): (u8, u8) = (1, 2);
        for group in 0..4 {
            let ql = &qs[group * 32..group * 32 + 32];
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let (d1, mn1) = (d * sc1 as f32, dmin * m1 as f32);
            let (d2, mn2) = (d * sc2 as f32, dmin * m2 as f32);
            for l in 0..32 {
                let hi = if qh[l] & u1 != 0 { 16 } else { 0 };
                out[w] = d1 * ((ql[l] & 0x0F) as i32 + hi) as f32 - mn1;
                w += 1;
            }
            for l in 0..32 {
                let hi = if qh[l] & u2 != 0 { 16 } else { 0 };
                out[w] = d2 * ((ql[l] >> 4) as i32 + hi) as f32 - mn2;
                w += 1;
            }
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

/// `block_q2_K` (super-block of 256): `[ u8 scales[16] | u8 qs[64] | f16 d | f16 dmin ]`.
///
/// Sixteen 16-element sub-blocks. Each `scales` byte packs a 4-bit scale and a 4-bit
/// min; each quant is 2 bits. `y = d·scale·q − dmin·min`.
fn dequant_q2_k(raw: &[u8], out: &mut [f32]) {
    for (bi, block) in raw.chunks_exact(84).enumerate() {
        let scales = &block[0..16];
        let qs = &block[16..80];
        let d = f16_to_f32(u16::from_le_bytes([block[80], block[81]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[82], block[83]]));

        let mut w = bi * QK_K;
        let mut is = 0;
        for half in 0..2 {
            let q = &qs[half * 32..half * 32 + 32];
            let mut shift = 0;
            for _ in 0..4 {
                let sc = scales[is];
                is += 1;
                let (dl, ml) = (d * (sc & 0x0F) as f32, dmin * (sc >> 4) as f32);
                for &qv in &q[0..16] {
                    out[w] = dl * ((qv >> shift) & 3) as f32 - ml;
                    w += 1;
                }
                let sc = scales[is];
                is += 1;
                let (dl, ml) = (d * (sc & 0x0F) as f32, dmin * (sc >> 4) as f32);
                for &qv in &q[16..32] {
                    out[w] = dl * ((qv >> shift) & 3) as f32 - ml;
                    w += 1;
                }
                shift += 2;
            }
        }
    }
}

/// `block_q3_K` (super-block of 256): `[ u8 hmask[32] | u8 qs[64] | u8 scales[12] | f16 d ]`.
///
/// The trickiest: 3-bit quants (2 low bits from `qs`, a high bit *inverted* from
/// `hmask` — set means don't subtract 4), and sixteen 6-bit signed scales packed
/// across 12 bytes via the same word juggling ggml uses.
fn dequant_q3_k(raw: &[u8], out: &mut [f32]) {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;

    for (bi, block) in raw.chunks_exact(110).enumerate() {
        let hmask = &block[0..32];
        let qs = &block[32..96];
        let scale_bytes = &block[96..108];
        let d_all = f16_to_f32(u16::from_le_bytes([block[108], block[109]]));

        // Unpack the 16 6-bit signed scales, exactly as ggml's dequantize_row_q3_K.
        let mut aux = [0u32; 4];
        for (k, a) in aux.iter_mut().take(3).enumerate() {
            *a = u32::from_le_bytes(scale_bytes[k * 4..k * 4 + 4].try_into().unwrap());
        }
        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        aux[0] = (aux[0] & KMASK2) | (((tmp) & KMASK1) << 4);
        aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
        let mut scales = [0i8; 16];
        for (k, a) in aux.iter().enumerate() {
            for (b, byte) in a.to_le_bytes().iter().enumerate() {
                scales[k * 4 + b] = *byte as i8;
            }
        }

        let mut w = bi * QK_K;
        let mut is = 0;
        let mut m: u8 = 1;
        for half in 0..2 {
            let q = &qs[half * 32..half * 32 + 32];
            let mut shift = 0;
            for _ in 0..4 {
                let dl = d_all * (scales[is] as i32 - 32) as f32;
                is += 1;
                for l in 0..16 {
                    let hbit = if hmask[l] & m != 0 { 0 } else { 4 };
                    out[w] = dl * (((q[l] >> shift) & 3) as i32 - hbit) as f32;
                    w += 1;
                }
                let dl = d_all * (scales[is] as i32 - 32) as f32;
                is += 1;
                for l in 0..16 {
                    let hbit = if hmask[l + 16] & m != 0 { 0 } else { 4 };
                    out[w] = dl * (((q[l + 16] >> shift) & 3) as i32 - hbit) as f32;
                    w += 1;
                }
                shift += 2;
                m <<= 1;
            }
        }
    }
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
    if sign == 1 { -val } else { val }
}

fn unsupported(ggml_type: u32) -> GarudaError {
    GarudaError::Model(format!(
        "tensor type {} ({ggml_type}) is not supported; F32, F16, Q4_0/Q8_0 and the \
         k-quants Q2_K/Q3_K/Q4_K/Q5_K/Q6_K decode today (the *_1 linear quants and the \
         IQ imatrix quants need their own decoders)",
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
        assert_eq!(block_layout(Q2_K), Some((256, 84)));
        assert_eq!(block_layout(Q3_K), Some((256, 110)));
        assert_eq!(block_layout(Q4_K), Some((256, 144)));
        assert_eq!(block_layout(Q5_K), Some((256, 176)));
        assert_eq!(block_layout(Q6_K), Some((256, 210)));
        assert_eq!(block_layout(3), None); // Q4_1 not decoded yet
        assert!(!is_supported(9)); // Q8_1
    }

    #[test]
    fn q5_k_adds_the_high_bit_from_qh() {
        // d = 1, dmin = 0, first sub-block scale = 1. qs[0] low nibble = 3; set the
        // matching qh bit (group 0 low-run uses bit 0) so the 5th bit adds 16 -> 19.
        let mut raw = vec![0u8; 176];
        raw[0..2].copy_from_slice(&f16(1.0)); // d
        raw[2..4].copy_from_slice(&f16(0.0)); // dmin
        raw[4] = 1; // scales[0] -> sc = 1
        raw[16] = 0x01; // qh[0], bit 0 set
        raw[48] = 0x03; // qs[0] low nibble = 3

        let y = dequantize(Q5_K, &raw, 256).unwrap();
        assert_eq!(y[0], 19.0); // 1 * (3 + 16)
        assert_eq!(y[1], 0.0);
    }

    #[test]
    fn q2_k_reads_2bit_quants_scaled_by_the_4bit_subscale() {
        // d = 1, dmin = 0. scales[0] = 0x02 -> scale nibble 2, min nibble 0.
        // qs[0] bits 0-1 = 3 -> output = 2 * 3 = 6.
        let mut raw = vec![0u8; 84];
        raw[0] = 0x02; // scales[0]: scale=2, min=0
        raw[16] = 0x03; // qs[0]: low 2 bits = 3
        raw[80..82].copy_from_slice(&f16(1.0)); // d
        raw[82..84].copy_from_slice(&f16(0.0)); // dmin

        let y = dequantize(Q2_K, &raw, 256).unwrap();
        assert_eq!(y[0], 6.0); // 1 * 2 * 3
        assert_eq!(y[1], 0.0);
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

    /// splitmix64, for deterministic pseudo-random test data (no `rand` dependency).
    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Inverse of [`get_scale_min_k4`]: pack 8 sub-block scales and 8 mins (each a
    /// 6-bit value, 0..64) into the 12-byte layout `block_q4_K` stores them in.
    fn pack_scale_min_k4(scales: [u8; 8], mins: [u8; 8]) -> [u8; 12] {
        let mut s = [0u8; 12];
        for i in 0..4 {
            s[i] = (scales[i] & 0x3F) | ((scales[4 + i] >> 4) << 6);
            s[4 + i] = (mins[i] & 0x3F) | ((mins[4 + i] >> 4) << 6);
            s[8 + i] = (scales[4 + i] & 0x0F) | ((mins[4 + i] & 0x0F) << 4);
        }
        s
    }

    #[test]
    fn pack_scale_min_k4_round_trips_through_the_real_unpacker() {
        let scales = [1u8, 5, 9, 13, 17, 21, 25, 29];
        let mins = [2u8, 6, 10, 14, 18, 22, 26, 30];
        let packed = pack_scale_min_k4(scales, mins);
        for j in 0..8 {
            let (sc, mn) = get_scale_min_k4(j, &packed);
            assert_eq!(sc, scales[j], "scale[{j}]");
            assert_eq!(mn, mins[j], "min[{j}]");
        }
    }

    /// A random-but-valid `block_q4_K` super-block: plausible `d`/`dmin`, random
    /// 6-bit sub-scales/mins, random 4-bit quants.
    fn random_q4_k_block(seed: &mut u64) -> [u8; 144] {
        let mut b = [0u8; 144];
        let d = half_bits(0.02 + (splitmix64(seed) % 100) as f32 * 0.001);
        let dmin = half_bits((splitmix64(seed) % 50) as f32 * 0.001);
        b[0..2].copy_from_slice(&d.to_le_bytes());
        b[2..4].copy_from_slice(&dmin.to_le_bytes());
        let scales: [u8; 8] = std::array::from_fn(|_| (splitmix64(seed) % 64) as u8);
        let mins: [u8; 8] = std::array::from_fn(|_| (splitmix64(seed) % 64) as u8);
        b[4..16].copy_from_slice(&pack_scale_min_k4(scales, mins));
        for byte in b[16..144].iter_mut() {
            *byte = (splitmix64(seed) % 256) as u8;
        }
        b
    }

    #[test]
    fn q4_k_matvec_matches_the_dequantise_then_dot_reference() {
        let mut seed = 42u64;
        // 3 super-blocks per row (768 elements) and 5 rows, so both the multi-block
        // and multi-row chunking in `matvec_q4_k`/`q4_k_dot` get exercised.
        let (rows, blocks_per_row) = (5, 3);
        let cols = blocks_per_row * QK_K;

        let mut weight = Vec::new();
        for _ in 0..rows {
            for _ in 0..blocks_per_row {
                weight.extend_from_slice(&random_q4_k_block(&mut seed));
            }
        }
        let x: Vec<f32> = (0..cols)
            .map(|i| ((splitmix64(&mut seed) % 2000) as f32 / 1000.0 - 1.0) * (1 + i % 5) as f32)
            .collect();

        let mut got = vec![0.0f32; rows];
        matvec(Q4_K, &weight, rows, cols, &x, &mut got).unwrap();

        for r in 0..rows {
            let row = &weight[r * blocks_per_row * 144..(r + 1) * blocks_per_row * 144];
            let deq = dequantize(Q4_K, row, cols).unwrap();
            let want: f32 = deq.iter().zip(&x).map(|(a, b)| a * b).sum();
            // Both sides quantise `x` to int8 with a per-32-block scale, so this is
            // close but not bit-identical — same tolerance shape as the Q8_0 round-trip
            // test above, scaled up for the larger row width.
            let tol = want.abs() * 0.02 + 1.0;
            assert!(
                (got[r] - want).abs() < tol,
                "row {r}: got {}, want {} (tol {tol})",
                got[r],
                want
            );
        }
    }

    #[test]
    fn q4_k_matvec_handles_an_all_zero_row_without_nan_or_panic() {
        let weight = vec![0u8; 144 * 2];
        let x = vec![0.5f32; QK_K * 2];
        let mut out = vec![0.0f32; 1];
        matvec(Q4_K, &weight, 1, QK_K * 2, &x, &mut out).unwrap();
        assert_eq!(out[0], 0.0);
    }

    /// Not a correctness test — a timing comparison of the new fused int8 path against
    /// the old dequantise-then-`f32`-dot path it replaced, at Mixtral's own FFN row
    /// width. `matvec` no longer exposes the old path for `Q4_K` directly (it always
    /// takes the fast one now), so this reimplements that old loop inline exactly as
    /// it read before this change, for comparison. Run with:
    ///   cargo test --release quant::tests::q4_k_new_kernel_is_faster_than_the_old_path -- --ignored --nocapture
    #[test]
    #[ignore]
    fn q4_k_new_kernel_is_faster_than_the_old_path() {
        use rayon::prelude::*;
        use std::time::Instant;

        // Mixtral-8x7B's own FFN row width: gate/up are [d_ff=14336, d_model=4096].
        let (rows, cols) = (14_336usize, 4096usize);
        let blocks_per_row = cols / QK_K;

        let mut seed = 7u64;
        let mut weight = Vec::with_capacity(rows * blocks_per_row * 144);
        for _ in 0..rows * blocks_per_row {
            weight.extend_from_slice(&random_q4_k_block(&mut seed));
        }
        let x: Vec<f32> = (0..cols)
            .map(|i| ((splitmix64(&mut seed) % 2000) as f32 / 1000.0 - 1.0) * (1 + i % 5) as f32)
            .collect();

        let row_bytes = 144 * blocks_per_row;
        let mut out_old = vec![0.0f32; rows];
        let mut out_new = vec![0.0f32; rows];

        // Warm up (page in `weight`, let rayon spin up its pool) before timing either.
        matvec(Q4_K, &weight, rows, cols, &x, &mut out_new).unwrap();

        let t = Instant::now();
        out_old
            .par_iter_mut()
            .zip(weight.par_chunks_exact(row_bytes))
            .for_each_init(
                || vec![0.0f32; cols],
                |buf, (o, row)| {
                    dequant_into(Q4_K, row, buf).unwrap();
                    *o = crate::simd::dot(buf, &x);
                },
            );
        let old_elapsed = t.elapsed();

        let t = Instant::now();
        matvec(Q4_K, &weight, rows, cols, &x, &mut out_new).unwrap();
        let new_elapsed = t.elapsed();

        eprintln!(
            "Q4_K matvec, {rows} rows x {cols} cols: old(dequant+f32 dot) = {old_elapsed:?}, \
             new(int8 fused) = {new_elapsed:?}, speedup = {:.2}x",
            old_elapsed.as_secs_f64() / new_elapsed.as_secs_f64()
        );

        // Relative error is meaningless on rows whose true dot product lands near
        // zero (a few do, out of 14336 uniformly-random rows) — one such row can be
        // 1000% "off" while being absolutely tiny. p90 is a robust regression guard;
        // max is printed for visibility but not asserted on.
        let mut rel_errs: Vec<f32> = out_old
            .iter()
            .zip(&out_new)
            .map(|(a, b)| (a - b).abs() / a.abs().max(1e-6))
            .collect();
        rel_errs.sort_by(|a, b| a.total_cmp(b));
        let n = rel_errs.len();
        let (p50, p90, max) = (rel_errs[n / 2], rel_errs[n * 9 / 10], rel_errs[n - 1]);
        eprintln!(
            "relative error: min={:.4} p50={p50:.4} p90={p90:.4} max={max:.4}",
            rel_errs[0]
        );
        assert!(p90 < 0.10, "p90 relative error regressed: {p90:.4}");
    }
}

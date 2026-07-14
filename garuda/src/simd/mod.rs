//! Dense math kernels.
//!
//! The f32 kernels are plain Rust written so LLVM auto-vectorises them (chunked, no
//! early exits); `matvec` fans rows out across rayon when the matrix is big enough to
//! pay for the split. The one place with explicit intrinsics is [`dot_i8`], which uses
//! aarch64 NEON for the integer quantised matmul path.

use rayon::prelude::*;

/// Below this many rows, rayon's split costs more than it saves.
const PAR_ROW_THRESHOLD: usize = 64;

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Integer dot product of two equal-length `i8` slices, returning the exact `i32` sum.
///
/// On aarch64 (every Apple Silicon core) it uses baseline NEON — a widening `i8×i8→i16`
/// multiply and a pairwise accumulate into `i32`, 16 lanes at a time. Everywhere else it
/// is a scalar loop. This is the primitive an integer quantised matmul is built on.
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    // NEON is baseline on aarch64, so no runtime detection is needed.
    #[cfg(target_arch = "aarch64")]
    let r = unsafe { dot_i8_neon(a, b) };
    #[cfg(not(target_arch = "aarch64"))]
    let r = dot_i8_scalar(a, b);
    r
}

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn dot_i8_scalar(a: &[i8], b: &[i8]) -> i32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x as i32 * y as i32)
        .sum()
}

#[cfg(target_arch = "aarch64")]
unsafe fn dot_i8_neon(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::aarch64::*;
    let n = a.len();
    let chunks = n / 16;
    let mut acc = vdupq_n_s32(0);
    for i in 0..chunks {
        let va = vld1q_s8(a.as_ptr().add(i * 16));
        let vb = vld1q_s8(b.as_ptr().add(i * 16));
        // i8×i8 → i16 for the low and high 8 lanes, then pairwise-accumulate into i32.
        let lo = vmull_s8(vget_low_s8(va), vget_low_s8(vb));
        let hi = vmull_s8(vget_high_s8(va), vget_high_s8(vb));
        acc = vpadalq_s16(acc, lo);
        acc = vpadalq_s16(acc, hi);
    }
    let mut sum = vaddvq_s32(acc);
    for j in chunks * 16..n {
        sum += a[j] as i32 * b[j] as i32;
    }
    sum
}

/// `out[r] = dot(m[r], x)` for a row-major `[rows, cols]` matrix.
///
/// # Panics
/// If the slice lengths disagree with `rows`/`cols`. Callers hold the dims, so a
/// mismatch is a bug in the caller rather than bad input.
pub fn matvec(m: &[f32], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    assert_eq!(m.len(), rows * cols, "matrix len does not match rows*cols");
    assert_eq!(x.len(), cols, "input len does not match cols");
    assert_eq!(out.len(), rows, "output len does not match rows");

    if rows >= PAR_ROW_THRESHOLD {
        out.par_iter_mut()
            .zip(m.par_chunks_exact(cols))
            .for_each(|(o, row)| *o = dot(row, x));
    } else {
        for (o, row) in out.iter_mut().zip(m.chunks_exact(cols)) {
            *o = dot(row, x);
        }
    }
}

/// Numerically stable softmax, in place.
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        // Fully masked (all -inf) or a NaN crept in. Fall back to uniform rather
        // than emitting NaNs that would poison the rest of the forward pass.
        if !x.is_empty() {
            let uniform = 1.0 / x.len() as f32;
            x.iter_mut().for_each(|v| *v = uniform);
        }
        return;
    }
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        x.iter_mut().for_each(|v| *v *= inv);
    }
}

/// SiLU / swish: `x * sigmoid(x)`, in place.
pub fn silu(x: &mut [f32]) {
    x.iter_mut().for_each(|v| *v /= 1.0 + (-*v).exp());
}

/// Root-mean-square norm, in place.
pub fn rmsnorm(x: &mut [f32], eps: f32) {
    if x.is_empty() {
        return;
    }
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    x.iter_mut().for_each(|v| *v *= scale);
}

/// `a += b`
pub fn add_assign(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    a.iter_mut().zip(b.iter()).for_each(|(x, y)| *x += y);
}

/// `a += b * k`
pub fn add_scaled(a: &mut [f32], b: &[f32], k: f32) {
    debug_assert_eq!(a.len(), b.len());
    a.iter_mut().zip(b.iter()).for_each(|(x, y)| *x += y * k);
}

/// `a *= b`, elementwise
pub fn mul_assign(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    a.iter_mut().zip(b.iter()).for_each(|(x, y)| *x *= y);
}

/// Rotary position embedding, applied in place to one head's vector.
///
/// Rotates each `(2i, 2i+1)` pair of `v` by `pos * theta^(-2i/head_dim)`.
pub fn rope(v: &mut [f32], pos: usize, theta: f32) {
    let head_dim = v.len();
    for i in 0..head_dim / 2 {
        let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
        let angle = pos as f32 * freq;
        let (sin, cos) = angle.sin_cos();
        let (a, b) = (v[2 * i], v[2 * i + 1]);
        v[2 * i] = a * cos - b * sin;
        v[2 * i + 1] = a * sin + b * cos;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_i8_matches_naive_and_scalar_matches_hardware() {
        // Lengths that exercise the 16-wide body and a ragged tail.
        for n in [0usize, 1, 15, 16, 31, 32, 64, 100] {
            let a: Vec<i8> = (0..n).map(|i| (i as i32 % 200 - 100) as i8).collect();
            let b: Vec<i8> = (0..n).map(|i| (i as i32 * 7 % 200 - 100) as i8).collect();
            let naive: i32 = a.iter().zip(&b).map(|(&x, &y)| x as i32 * y as i32).sum();
            assert_eq!(dot_i8_scalar(&a, &b), naive, "scalar n={n}");
            // dot_i8 uses the hardware path on this machine when available; it must
            // still equal the exact integer result.
            assert_eq!(dot_i8(&a, &b), naive, "dispatched n={n}");
        }
    }

    #[test]
    fn dot_i8_handles_extremes_without_overflow() {
        // 128 × (-128 × -128) fits comfortably in i32.
        let a = vec![-128i8; 128];
        let b = vec![-128i8; 128];
        assert_eq!(dot_i8(&a, &b), 128 * 128 * 128);
    }

    #[test]
    fn matvec_matches_naive_above_and_below_par_threshold() {
        for rows in [8usize, 128] {
            let cols = 7;
            let m: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.01).collect();
            let x: Vec<f32> = (0..cols).map(|i| i as f32).collect();
            let mut out = vec![0.0; rows];
            matvec(&m, rows, cols, &x, &mut out);

            for r in 0..rows {
                let expect: f32 = (0..cols).map(|c| m[r * cols + c] * x[c]).sum();
                assert!((out[r] - expect).abs() < 1e-4, "rows={rows} row={r}");
            }
        }
    }

    #[test]
    fn softmax_sums_to_one_and_survives_large_inputs() {
        let mut x = vec![1000.0, 1000.5, 999.0];
        softmax(&mut x);
        assert!(x.iter().all(|v| v.is_finite()));
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(x[1] > x[0] && x[0] > x[2]);
    }

    #[test]
    fn softmax_of_empty_and_fully_masked_is_not_nan() {
        let mut empty: Vec<f32> = vec![];
        softmax(&mut empty);

        let mut masked = vec![f32::NEG_INFINITY; 3];
        softmax(&mut masked);
        assert!(masked.iter().all(|v| v.is_finite()));
        assert!((masked.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn rope_is_norm_preserving() {
        let mut v = vec![0.3, -1.2, 0.7, 2.0];
        let before: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        rope(&mut v, 5, 10_000.0);
        let after: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((before - after).abs() < 1e-5);
    }

    #[test]
    fn rope_at_position_zero_is_identity() {
        let orig = vec![0.3, -1.2, 0.7, 2.0];
        let mut v = orig.clone();
        rope(&mut v, 0, 10_000.0);
        for (a, b) in v.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn rmsnorm_of_empty_does_not_panic() {
        let mut x: Vec<f32> = vec![];
        rmsnorm(&mut x, 1e-5);
    }
}

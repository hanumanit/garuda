//! Dense math kernels.
//!
//! These are plain Rust written so LLVM auto-vectorises them (chunked, no early
//! exits). There are no intrinsics and no hand-written assembly here — the name
//! refers to the codegen, not to hand-rolled SIMD. `matvec` fans rows out across
//! rayon when the matrix is big enough to pay for the split.

use rayon::prelude::*;

/// Below this many rows, rayon's split costs more than it saves.
const PAR_ROW_THRESHOLD: usize = 64;

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
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

//! SIMD-accelerated operations for QSM processing
//!
//! This module provides vectorized versions of common operations used in
//! iterative algorithms like MEDI, TV, and TGV. When the `simd` feature is
//! enabled, these use 128-bit SIMD (f32x4) which is compatible with both
//! native SSE/NEON and WASM SIMD.
//!
//! All operations have scalar fallbacks when SIMD is disabled.

#[cfg(feature = "simd")]
use wide::f32x4;

/// SIMD lane width (4 for f32x4)
#[cfg(feature = "simd")]
pub const SIMD_WIDTH: usize = 4;

#[cfg(not(feature = "simd"))]
pub const SIMD_WIDTH: usize = 1;

// ============================================================================
// Dot Product Operations
// ============================================================================

/// Compute dot product: sum(a[i] * b[i])
#[cfg(feature = "simd")]
#[inline]
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / SIMD_WIDTH;
    let remainder = n % SIMD_WIDTH;

    let mut sum = f32x4::ZERO;

    // Process 4 elements at a time
    for i in 0..chunks {
        let idx = i * SIMD_WIDTH;
        let va = f32x4::from(&a[idx..idx + SIMD_WIDTH]);
        let vb = f32x4::from(&b[idx..idx + SIMD_WIDTH]);
        sum += va * vb;
    }

    // Horizontal sum of SIMD register
    let mut result = sum.reduce_add();

    // Handle remainder
    let start = chunks * SIMD_WIDTH;
    for i in 0..remainder {
        result += a[start + i] * b[start + i];
    }

    result
}

#[cfg(not(feature = "simd"))]
#[inline]
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
}

/// Compute squared norm: sum(a[i]^2)
#[cfg(feature = "simd")]
#[inline]
pub fn norm_squared_f32(a: &[f32]) -> f32 {
    let n = a.len();
    let chunks = n / SIMD_WIDTH;
    let remainder = n % SIMD_WIDTH;

    let mut sum = f32x4::ZERO;

    for i in 0..chunks {
        let idx = i * SIMD_WIDTH;
        let va = f32x4::from(&a[idx..idx + SIMD_WIDTH]);
        sum += va * va;
    }

    let mut result = sum.reduce_add();

    let start = chunks * SIMD_WIDTH;
    for i in 0..remainder {
        result += a[start + i] * a[start + i];
    }

    result
}

#[cfg(not(feature = "simd"))]
#[inline]
pub fn norm_squared_f32(a: &[f32]) -> f32 {
    a.iter().map(|&ai| ai * ai).sum()
}

// ============================================================================
// Fused Multiply-Add Operations
// ============================================================================

/// Compute a[i] = a[i] + alpha * b[i] (axpy operation)
#[cfg(feature = "simd")]
#[inline]
pub fn axpy_f32(a: &mut [f32], alpha: f32, b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / SIMD_WIDTH;
    let remainder = n % SIMD_WIDTH;

    let valpha = f32x4::splat(alpha);

    for i in 0..chunks {
        let idx = i * SIMD_WIDTH;
        let va = f32x4::from(&a[idx..idx + SIMD_WIDTH]);
        let vb = f32x4::from(&b[idx..idx + SIMD_WIDTH]);
        let result = va + valpha * vb;
        a[idx..idx + SIMD_WIDTH].copy_from_slice(result.as_array_ref());
    }

    let start = chunks * SIMD_WIDTH;
    for i in 0..remainder {
        a[start + i] += alpha * b[start + i];
    }
}

#[cfg(not(feature = "simd"))]
#[inline]
pub fn axpy_f32(a: &mut [f32], alpha: f32, b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        a[i] += alpha * b[i];
    }
}

/// Compute a[i] = b[i] + beta * a[i] (used in CG for p update)
#[cfg(feature = "simd")]
#[inline]
pub fn xpby_f32(a: &mut [f32], b: &[f32], beta: f32) {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / SIMD_WIDTH;
    let remainder = n % SIMD_WIDTH;

    let vbeta = f32x4::splat(beta);

    for i in 0..chunks {
        let idx = i * SIMD_WIDTH;
        let va = f32x4::from(&a[idx..idx + SIMD_WIDTH]);
        let vb = f32x4::from(&b[idx..idx + SIMD_WIDTH]);
        let result = vb + vbeta * va;
        a[idx..idx + SIMD_WIDTH].copy_from_slice(result.as_array_ref());
    }

    let start = chunks * SIMD_WIDTH;
    for i in 0..remainder {
        a[start + i] = b[start + i] + beta * a[start + i];
    }
}

#[cfg(not(feature = "simd"))]
#[inline]
pub fn xpby_f32(a: &mut [f32], b: &[f32], beta: f32) {
    debug_assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        a[i] = b[i] + beta * a[i];
    }
}

// ============================================================================
// Element-wise Operations for MEDI
// ============================================================================

/// Apply per-direction gradient weights: out[i] = mx[i] * p[i] * mx[i] * gx[i]
/// This is the core operation in MEDI's regularization term
#[cfg(feature = "simd")]
#[inline]
pub fn apply_gradient_weights_f32(
    out_x: &mut [f32], out_y: &mut [f32], out_z: &mut [f32],
    mx: &[f32], my: &[f32], mz: &[f32],
    p: &[f32],
    gx: &[f32], gy: &[f32], gz: &[f32],
) {
    let n = out_x.len();
    debug_assert!(out_y.len() == n && out_z.len() == n);
    debug_assert!(mx.len() == n && my.len() == n && mz.len() == n);
    debug_assert!(p.len() == n && gx.len() == n && gy.len() == n && gz.len() == n);

    let chunks = n / SIMD_WIDTH;
    let remainder = n % SIMD_WIDTH;

    for i in 0..chunks {
        let idx = i * SIMD_WIDTH;

        let vmx = f32x4::from(&mx[idx..idx + SIMD_WIDTH]);
        let vmy = f32x4::from(&my[idx..idx + SIMD_WIDTH]);
        let vmz = f32x4::from(&mz[idx..idx + SIMD_WIDTH]);
        let vp = f32x4::from(&p[idx..idx + SIMD_WIDTH]);
        let vgx = f32x4::from(&gx[idx..idx + SIMD_WIDTH]);
        let vgy = f32x4::from(&gy[idx..idx + SIMD_WIDTH]);
        let vgz = f32x4::from(&gz[idx..idx + SIMD_WIDTH]);

        // out = m * p * m * g = m^2 * p * g
        let rx = vmx * vp * vmx * vgx;
        let ry = vmy * vp * vmy * vgy;
        let rz = vmz * vp * vmz * vgz;

        out_x[idx..idx + SIMD_WIDTH].copy_from_slice(rx.as_array_ref());
        out_y[idx..idx + SIMD_WIDTH].copy_from_slice(ry.as_array_ref());
        out_z[idx..idx + SIMD_WIDTH].copy_from_slice(rz.as_array_ref());
    }

    let start = chunks * SIMD_WIDTH;
    for i in 0..remainder {
        let idx = start + i;
        out_x[idx] = mx[idx] * p[idx] * mx[idx] * gx[idx];
        out_y[idx] = my[idx] * p[idx] * my[idx] * gy[idx];
        out_z[idx] = mz[idx] * p[idx] * mz[idx] * gz[idx];
    }
}

#[cfg(not(feature = "simd"))]
#[inline]
pub fn apply_gradient_weights_f32(
    out_x: &mut [f32], out_y: &mut [f32], out_z: &mut [f32],
    mx: &[f32], my: &[f32], mz: &[f32],
    p: &[f32],
    gx: &[f32], gy: &[f32], gz: &[f32],
) {
    let n = out_x.len();
    for i in 0..n {
        out_x[i] = mx[i] * p[i] * mx[i] * gx[i];
        out_y[i] = my[i] * p[i] * my[i] * gy[i];
        out_z[i] = mz[i] * p[i] * mz[i] * gz[i];
    }
}

/// Compute P = 1 / sqrt(ux^2 + uy^2 + uz^2 + beta)
/// where ux = mx * gx, uy = my * gy, uz = mz * gz
#[cfg(feature = "simd")]
#[inline]
pub fn compute_p_weights_f32(
    p: &mut [f32],
    mx: &[f32], my: &[f32], mz: &[f32],
    gx: &[f32], gy: &[f32], gz: &[f32],
    beta: f32,
) {
    let n = p.len();
    let chunks = n / SIMD_WIDTH;
    let remainder = n % SIMD_WIDTH;

    let vbeta = f32x4::splat(beta);

    for i in 0..chunks {
        let idx = i * SIMD_WIDTH;

        let vmx = f32x4::from(&mx[idx..idx + SIMD_WIDTH]);
        let vmy = f32x4::from(&my[idx..idx + SIMD_WIDTH]);
        let vmz = f32x4::from(&mz[idx..idx + SIMD_WIDTH]);
        let vgx = f32x4::from(&gx[idx..idx + SIMD_WIDTH]);
        let vgy = f32x4::from(&gy[idx..idx + SIMD_WIDTH]);
        let vgz = f32x4::from(&gz[idx..idx + SIMD_WIDTH]);

        let ux = vmx * vgx;
        let uy = vmy * vgy;
        let uz = vmz * vgz;

        let norm_sq = ux * ux + uy * uy + uz * uz + vbeta;
        let result = norm_sq.sqrt().recip();

        p[idx..idx + SIMD_WIDTH].copy_from_slice(result.as_array_ref());
    }

    let start = chunks * SIMD_WIDTH;
    for i in 0..remainder {
        let idx = start + i;
        let ux = mx[idx] * gx[idx];
        let uy = my[idx] * gy[idx];
        let uz = mz[idx] * gz[idx];
        p[idx] = 1.0 / (ux * ux + uy * uy + uz * uz + beta).sqrt();
    }
}

#[cfg(not(feature = "simd"))]
#[inline]
pub fn compute_p_weights_f32(
    p: &mut [f32],
    mx: &[f32], my: &[f32], mz: &[f32],
    gx: &[f32], gy: &[f32], gz: &[f32],
    beta: f32,
) {
    let n = p.len();
    for i in 0..n {
        let ux = mx[i] * gx[i];
        let uy = my[i] * gy[i];
        let uz = mz[i] * gz[i];
        p[i] = 1.0 / (ux * ux + uy * uy + uz * uz + beta).sqrt();
    }
}

/// Combine regularization and data terms: out[i] = lambda * reg[i] + data[i]
/// Matches MATLAB MEDI: y = D + R where D is data term, R = lambda * reg term
#[cfg(feature = "simd")]
#[inline]
pub fn combine_terms_f32(out: &mut [f32], reg: &[f32], data: &[f32], lambda: f32) {
    let n = out.len();
    let chunks = n / SIMD_WIDTH;
    let remainder = n % SIMD_WIDTH;

    let vlambda = f32x4::splat(lambda);

    for i in 0..chunks {
        let idx = i * SIMD_WIDTH;
        let vreg = f32x4::from(&reg[idx..idx + SIMD_WIDTH]);
        let vdata = f32x4::from(&data[idx..idx + SIMD_WIDTH]);
        let result = vlambda * vreg + vdata;
        out[idx..idx + SIMD_WIDTH].copy_from_slice(result.as_array_ref());
    }

    let start = chunks * SIMD_WIDTH;
    for i in 0..remainder {
        out[start + i] = lambda * reg[start + i] + data[start + i];
    }
}

#[cfg(not(feature = "simd"))]
#[inline]
pub fn combine_terms_f32(out: &mut [f32], reg: &[f32], data: &[f32], lambda: f32) {
    for i in 0..out.len() {
        out[i] = lambda * reg[i] + data[i];
    }
}

/// Negate array in place: a[i] = -a[i]
#[cfg(feature = "simd")]
#[inline]
pub fn negate_f32(a: &mut [f32]) {
    let n = a.len();
    let chunks = n / SIMD_WIDTH;
    let remainder = n % SIMD_WIDTH;

    for i in 0..chunks {
        let idx = i * SIMD_WIDTH;
        let va = f32x4::from(&a[idx..idx + SIMD_WIDTH]);
        let result = -va;
        a[idx..idx + SIMD_WIDTH].copy_from_slice(result.as_array_ref());
    }

    let start = chunks * SIMD_WIDTH;
    for i in 0..remainder {
        a[start + i] = -a[start + i];
    }
}

#[cfg(not(feature = "simd"))]
#[inline]
pub fn negate_f32(a: &mut [f32]) {
    for val in a.iter_mut() {
        *val = -*val;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0f32, 3.0, 4.0, 5.0, 6.0];

        let result = dot_product_f32(&a, &b);
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_norm_squared() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

        let result = norm_squared_f32(&a);
        let expected: f32 = a.iter().map(|x| x * x).sum();

        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_axpy() {
        let mut a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0f32, 3.0, 4.0, 5.0, 6.0];
        let alpha = 0.5f32;

        let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + alpha * y).collect();
        axpy_f32(&mut a, alpha, &b);

        for (r, e) in a.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-6);
        }
    }

    #[test]
    fn test_negate() {
        let mut a = vec![1.0f32, -2.0, 3.0, -4.0, 5.0];
        let expected = vec![-1.0f32, 2.0, -3.0, 4.0, -5.0];

        negate_f32(&mut a);

        for (r, e) in a.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-6);
        }
    }

    #[test]
    fn test_gradient_weights() {
        let n = 5;
        let mx = vec![1.0f32; n];
        let my = vec![0.5f32; n];
        let mz = vec![0.8f32; n];
        let p = vec![2.0f32; n];
        let gx = vec![0.1f32; n];
        let gy = vec![0.2f32; n];
        let gz = vec![0.3f32; n];

        let mut out_x = vec![0.0f32; n];
        let mut out_y = vec![0.0f32; n];
        let mut out_z = vec![0.0f32; n];

        apply_gradient_weights_f32(&mut out_x, &mut out_y, &mut out_z, &mx, &my, &mz, &p, &gx, &gy, &gz);

        // Check first element manually
        let ex = 1.0 * 2.0 * 1.0 * 0.1;  // mx * p * mx * gx
        let ey = 0.5 * 2.0 * 0.5 * 0.2;  // my * p * my * gy
        let ez = 0.8 * 2.0 * 0.8 * 0.3;  // mz * p * mz * gz

        assert!((out_x[0] - ex).abs() < 1e-6);
        assert!((out_y[0] - ey).abs() < 1e-6);
        assert!((out_z[0] - ez).abs() < 1e-6);
    }
}

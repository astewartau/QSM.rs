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
// Scale Operations
// ============================================================================

/// Scale array in place: a[i] = alpha * a[i]
#[cfg(feature = "simd")]
#[inline]
pub fn scale_f32(a: &mut [f32], alpha: f32) {
    let n = a.len();
    let chunks = n / SIMD_WIDTH;
    let remainder = n % SIMD_WIDTH;

    let valpha = f32x4::splat(alpha);

    for i in 0..chunks {
        let idx = i * SIMD_WIDTH;
        let va = f32x4::from(&a[idx..idx + SIMD_WIDTH]);
        let result = valpha * va;
        a[idx..idx + SIMD_WIDTH].copy_from_slice(result.as_array_ref());
    }

    let start = chunks * SIMD_WIDTH;
    for i in 0..remainder {
        a[start + i] *= alpha;
    }
}

#[cfg(not(feature = "simd"))]
#[inline]
pub fn scale_f32(a: &mut [f32], alpha: f32) {
    for val in a.iter_mut() {
        *val *= alpha;
    }
}

/// Scale array in place: a[i] = alpha * a[i] (f64 version)
#[inline]
pub fn scale_f64(a: &mut [f64], alpha: f64) {
    for val in a.iter_mut() {
        *val *= alpha;
    }
}

// ============================================================================
// Subtract Operations
// ============================================================================

/// Subtract arrays element-wise: out[i] = a[i] - b[i]
#[cfg(feature = "simd")]
#[inline]
pub fn subtract_f32(out: &mut [f32], a: &[f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(out.len(), a.len());
    let n = a.len();
    let chunks = n / SIMD_WIDTH;
    let remainder = n % SIMD_WIDTH;

    for i in 0..chunks {
        let idx = i * SIMD_WIDTH;
        let va = f32x4::from(&a[idx..idx + SIMD_WIDTH]);
        let vb = f32x4::from(&b[idx..idx + SIMD_WIDTH]);
        let result = va - vb;
        out[idx..idx + SIMD_WIDTH].copy_from_slice(result.as_array_ref());
    }

    let start = chunks * SIMD_WIDTH;
    for i in 0..remainder {
        out[start + i] = a[start + i] - b[start + i];
    }
}

#[cfg(not(feature = "simd"))]
#[inline]
pub fn subtract_f32(out: &mut [f32], a: &[f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(out.len(), a.len());
    for i in 0..out.len() {
        out[i] = a[i] - b[i];
    }
}

/// Subtract arrays element-wise: out[i] = a[i] - b[i] (f64 version)
#[inline]
pub fn subtract_f64(out: &mut [f64], a: &[f64], b: &[f64]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(out.len(), a.len());
    for i in 0..out.len() {
        out[i] = a[i] - b[i];
    }
}

// ============================================================================
// F64 Versions of Core Operations
// ============================================================================

/// Compute dot product: sum(a[i] * b[i]) (f64 version)
#[inline]
pub fn dot_product_f64(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
}

/// Compute squared norm: sum(a[i]^2) (f64 version)
#[inline]
pub fn norm_squared_f64(a: &[f64]) -> f64 {
    a.iter().map(|&ai| ai * ai).sum()
}

/// Compute a[i] = a[i] + alpha * b[i] (axpy operation, f64 version)
#[inline]
pub fn axpy_f64(a: &mut [f64], alpha: f64, b: &[f64]) {
    debug_assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        a[i] += alpha * b[i];
    }
}

/// Negate array in place: a[i] = -a[i] (f64 version)
#[inline]
pub fn negate_f64(a: &mut [f64]) {
    for val in a.iter_mut() {
        *val = -*val;
    }
}

/// Apply per-direction gradient weights (f64 version)
#[inline]
pub fn apply_gradient_weights_f64(
    out_x: &mut [f64], out_y: &mut [f64], out_z: &mut [f64],
    mx: &[f64], my: &[f64], mz: &[f64],
    p: &[f64],
    gx: &[f64], gy: &[f64], gz: &[f64],
) {
    let n = out_x.len();
    for i in 0..n {
        out_x[i] = mx[i] * p[i] * mx[i] * gx[i];
        out_y[i] = my[i] * p[i] * my[i] * gy[i];
        out_z[i] = mz[i] * p[i] * mz[i] * gz[i];
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

    // ====================================================================
    // F64 versions and new operations
    // ====================================================================

    #[test]
    fn test_dot_product_f64() {
        let a = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0f64, 3.0, 4.0, 5.0, 6.0];

        let result = dot_product_f64(&a, &b);
        let expected: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        assert!((result - expected).abs() < 1e-12);
    }

    #[test]
    fn test_norm_squared_f64() {
        let a = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];

        let result = norm_squared_f64(&a);
        let expected: f64 = a.iter().map(|x| x * x).sum();

        assert!((result - expected).abs() < 1e-12);
    }

    #[test]
    fn test_axpy_f64() {
        let mut a = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0f64, 3.0, 4.0, 5.0, 6.0];
        let alpha = 0.5f64;

        let expected: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x + alpha * y).collect();
        axpy_f64(&mut a, alpha, &b);

        for (r, e) in a.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-12);
        }
    }

    #[test]
    fn test_negate_f64() {
        let mut a = vec![1.0f64, -2.0, 3.0, -4.0, 5.0];
        let expected = vec![-1.0f64, 2.0, -3.0, 4.0, -5.0];

        negate_f64(&mut a);

        for (r, e) in a.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-12);
        }
    }

    #[test]
    fn test_scale_f32() {
        let mut a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let alpha = 2.5f32;

        let expected: Vec<f32> = a.iter().map(|x| x * alpha).collect();
        scale_f32(&mut a, alpha);

        for (r, e) in a.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-6, "scale_f32 mismatch: got {}, expected {}", r, e);
        }
    }

    #[test]
    fn test_subtract_f32() {
        let a = vec![5.0f32, 4.0, 3.0, 2.0, 1.0, 0.5, 0.25];
        let b = vec![1.0f32, 1.5, 2.0, 2.5, 3.0, 0.1, 0.05];
        let mut out = vec![0.0f32; 7];

        let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
        subtract_f32(&mut out, &a, &b);

        for (r, e) in out.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-6, "subtract_f32 mismatch: got {}, expected {}", r, e);
        }
    }

    #[test]
    fn test_scale_f64() {
        let mut a = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let alpha = 3.14f64;

        let expected: Vec<f64> = a.iter().map(|x| x * alpha).collect();
        scale_f64(&mut a, alpha);

        for (r, e) in a.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-12, "scale_f64 mismatch: got {}, expected {}", r, e);
        }
    }

    #[test]
    fn test_subtract_f64() {
        let a = vec![10.0f64, 20.0, 30.0, 40.0, 50.0];
        let b = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let mut out = vec![0.0f64; 5];

        let expected: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
        subtract_f64(&mut out, &a, &b);

        for (r, e) in out.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-12, "subtract_f64 mismatch: got {}, expected {}", r, e);
        }
    }

    #[test]
    fn test_gradient_weights_f64() {
        let n = 5;
        let mx = vec![1.0f64; n];
        let my = vec![0.5f64; n];
        let mz = vec![0.8f64; n];
        let p = vec![2.0f64; n];
        let gx = vec![0.1f64; n];
        let gy = vec![0.2f64; n];
        let gz = vec![0.3f64; n];

        let mut out_x = vec![0.0f64; n];
        let mut out_y = vec![0.0f64; n];
        let mut out_z = vec![0.0f64; n];

        apply_gradient_weights_f64(&mut out_x, &mut out_y, &mut out_z, &mx, &my, &mz, &p, &gx, &gy, &gz);

        let ex = 1.0 * 2.0 * 1.0 * 0.1;
        let ey = 0.5 * 2.0 * 0.5 * 0.2;
        let ez = 0.8 * 2.0 * 0.8 * 0.3;

        for i in 0..n {
            assert!((out_x[i] - ex).abs() < 1e-12);
            assert!((out_y[i] - ey).abs() < 1e-12);
            assert!((out_z[i] - ez).abs() < 1e-12);
        }
    }

    // ====================================================================
    // Large-array f32 tests for SIMD coverage
    // ====================================================================

    #[test]
    fn test_dot_product_f32_large() {
        // Use 128+ elements so SIMD loop executes multiple iterations
        let n = 256;
        let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..n).map(|i| 1.0 - (i as f32) * 0.005).collect();

        let result = dot_product_f32(&a, &b);
        let expected: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();

        assert!(
            (result - expected).abs() < 1e-2,
            "dot_product_f32 large: got {}, expected {}",
            result, expected
        );
    }

    #[test]
    fn test_dot_product_f32_large_exact_multiple() {
        // Array size that is an exact multiple of SIMD_WIDTH (no remainder)
        let n = 128;
        let a = vec![2.0f32; n];
        let b = vec![3.0f32; n];

        let result = dot_product_f32(&a, &b);
        let expected = 2.0 * 3.0 * n as f32;

        assert!(
            (result - expected).abs() < 1e-3,
            "dot_product_f32 exact multiple: got {}, expected {}",
            result, expected
        );
    }

    #[test]
    fn test_norm_squared_f32_large() {
        let n = 256;
        let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();

        let result = norm_squared_f32(&a);
        let expected: f32 = a.iter().map(|&x| x * x).sum();

        assert!(
            (result - expected).abs() < 1.0,
            "norm_squared_f32 large: got {}, expected {}",
            result, expected
        );
    }

    #[test]
    fn test_norm_squared_f32_large_exact_multiple() {
        let n = 128;
        let a = vec![0.5f32; n];

        let result = norm_squared_f32(&a);
        let expected = 0.25 * n as f32;

        assert!(
            (result - expected).abs() < 1e-3,
            "norm_squared_f32 exact: got {}, expected {}",
            result, expected
        );
    }

    #[test]
    fn test_axpy_f32_large() {
        let n = 256;
        let mut a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.05).collect();
        let alpha = 2.5f32;

        let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| x + alpha * y).collect();
        axpy_f32(&mut a, alpha, &b);

        for i in 0..n {
            assert!(
                (a[i] - expected[i]).abs() < 1e-3,
                "axpy_f32 large mismatch at {}: got {}, expected {}",
                i, a[i], expected[i]
            );
        }
    }

    #[test]
    fn test_axpy_f32_large_exact_multiple() {
        let n = 128;
        let mut a = vec![1.0f32; n];
        let b = vec![2.0f32; n];
        let alpha = 3.0f32;

        axpy_f32(&mut a, alpha, &b);

        for &val in &a {
            assert!(
                (val - 7.0).abs() < 1e-6, // 1 + 3*2 = 7
                "axpy exact: got {}, expected 7.0",
                val
            );
        }
    }

    #[test]
    fn test_negate_f32_large() {
        let n = 256;
        let original: Vec<f32> = (0..n).map(|i| (i as f32) - 128.0).collect();
        let mut a = original.clone();

        negate_f32(&mut a);

        for i in 0..n {
            assert!(
                (a[i] + original[i]).abs() < 1e-6,
                "negate_f32 large: a[{}] = {}, expected {}",
                i, a[i], -original[i]
            );
        }
    }

    #[test]
    fn test_negate_f32_large_exact_multiple() {
        let n = 128;
        let mut a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let expected: Vec<f32> = a.iter().map(|&x| -x).collect();

        negate_f32(&mut a);

        for i in 0..n {
            assert!(
                (a[i] - expected[i]).abs() < 1e-6,
                "negate_f32 exact: a[{}] = {}, expected {}",
                i, a[i], expected[i]
            );
        }
    }

    #[test]
    fn test_apply_gradient_weights_f32_large() {
        let n = 256;
        let mx: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.001).collect();
        let my: Vec<f32> = (0..n).map(|i| 0.3 + (i as f32) * 0.002).collect();
        let mz: Vec<f32> = (0..n).map(|i| 0.8 - (i as f32) * 0.001).collect();
        let p: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.01).collect();
        let gx: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
        let gy: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).cos()).collect();
        let gz: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).sin()).collect();

        let mut out_x = vec![0.0f32; n];
        let mut out_y = vec![0.0f32; n];
        let mut out_z = vec![0.0f32; n];

        apply_gradient_weights_f32(
            &mut out_x, &mut out_y, &mut out_z,
            &mx, &my, &mz, &p, &gx, &gy, &gz,
        );

        // Verify against scalar computation
        for i in 0..n {
            let ex = mx[i] * p[i] * mx[i] * gx[i];
            let ey = my[i] * p[i] * my[i] * gy[i];
            let ez = mz[i] * p[i] * mz[i] * gz[i];

            assert!(
                (out_x[i] - ex).abs() < 1e-4,
                "gradient_weights_f32 x[{}]: got {}, expected {}",
                i, out_x[i], ex
            );
            assert!(
                (out_y[i] - ey).abs() < 1e-4,
                "gradient_weights_f32 y[{}]: got {}, expected {}",
                i, out_y[i], ey
            );
            assert!(
                (out_z[i] - ez).abs() < 1e-4,
                "gradient_weights_f32 z[{}]: got {}, expected {}",
                i, out_z[i], ez
            );
        }
    }

    #[test]
    fn test_scale_f32_large() {
        let n = 256;
        let mut a: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let alpha = 3.14f32;

        let expected: Vec<f32> = a.iter().map(|&x| x * alpha).collect();
        scale_f32(&mut a, alpha);

        for i in 0..n {
            assert!(
                (a[i] - expected[i]).abs() < 1e-3,
                "scale_f32 large: a[{}] = {}, expected {}",
                i, a[i], expected[i]
            );
        }
    }

    #[test]
    fn test_subtract_f32_large() {
        let n = 256;
        let a: Vec<f32> = (0..n).map(|i| (i as f32) * 2.0).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
        let mut out = vec![0.0f32; n];

        let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect();
        subtract_f32(&mut out, &a, &b);

        for i in 0..n {
            assert!(
                (out[i] - expected[i]).abs() < 1e-3,
                "subtract_f32 large: out[{}] = {}, expected {}",
                i, out[i], expected[i]
            );
        }
    }

    #[test]
    fn test_xpby_f32_large() {
        let n = 256;
        let mut a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.05).collect();
        let beta = 0.7f32;

        let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(&ai, &bi)| bi + beta * ai).collect();
        xpby_f32(&mut a, &b, beta);

        for i in 0..n {
            assert!(
                (a[i] - expected[i]).abs() < 1e-3,
                "xpby_f32 large: a[{}] = {}, expected {}",
                i, a[i], expected[i]
            );
        }
    }

    #[test]
    fn test_combine_terms_f32_large() {
        let n = 256;
        let reg: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let data: Vec<f32> = (0..n).map(|i| 1.0 - (i as f32) * 0.005).collect();
        let mut out = vec![0.0f32; n];
        let lambda = 2.5f32;

        let expected: Vec<f32> = reg.iter().zip(data.iter())
            .map(|(&r, &d)| lambda * r + d).collect();
        combine_terms_f32(&mut out, &reg, &data, lambda);

        for i in 0..n {
            assert!(
                (out[i] - expected[i]).abs() < 1e-3,
                "combine_terms_f32 large: out[{}] = {}, expected {}",
                i, out[i], expected[i]
            );
        }
    }

    #[test]
    fn test_compute_p_weights_f32_large() {
        let n = 256;
        let mx: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.001).collect();
        let my: Vec<f32> = (0..n).map(|i| 0.3 + (i as f32) * 0.002).collect();
        let mz: Vec<f32> = (0..n).map(|i| 0.8 - (i as f32) * 0.001).collect();
        let gx: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
        let gy: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).cos()).collect();
        let gz: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).sin()).collect();
        let beta = 1e-4f32;

        let mut p = vec![0.0f32; n];
        compute_p_weights_f32(&mut p, &mx, &my, &mz, &gx, &gy, &gz, beta);

        // Verify against scalar computation
        for i in 0..n {
            let ux = mx[i] * gx[i];
            let uy = my[i] * gy[i];
            let uz = mz[i] * gz[i];
            let expected = 1.0 / (ux * ux + uy * uy + uz * uz + beta).sqrt();

            assert!(
                (p[i] - expected).abs() < 1e-3,
                "compute_p_weights large: p[{}] = {}, expected {}",
                i, p[i], expected
            );
        }
    }

    #[test]
    fn test_dot_product_f32_with_remainder() {
        // Array size that is NOT a multiple of SIMD_WIDTH to test remainder path
        let n = 131; // 131 = 32*4 + 3 remainder
        let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..n).map(|i| 1.0 - (i as f32) * 0.005).collect();

        let result = dot_product_f32(&a, &b);
        let expected: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();

        assert!(
            (result - expected).abs() < 1.0,
            "dot_product_f32 remainder: got {}, expected {}",
            result, expected
        );
    }

    #[test]
    fn test_norm_squared_f32_with_remainder() {
        let n = 131;
        let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();

        let result = norm_squared_f32(&a);
        let expected: f32 = a.iter().map(|&x| x * x).sum();

        assert!(
            (result - expected).abs() < 1.0,
            "norm_squared_f32 remainder: got {}, expected {}",
            result, expected
        );
    }
}

//! Small numeric helpers shared across algorithms: soft-thresholding (L1 prox),
//! vector norms, and convergence checks. Kept in their own module so that
//! `utils/mod.rs` stays purely structural (module declarations + re-exports) —
//! adding a new util module then no longer looks like a behaviour change.

/// Soft thresholding (shrinkage) operator for L1 regularization.
/// shrink(x, t) = sign(x) * max(|x| - t, 0)
#[inline]
pub fn shrink(x: f64, threshold: f64) -> f64 {
    if x > threshold {
        x - threshold
    } else if x < -threshold {
        x + threshold
    } else {
        0.0
    }
}

/// Weighted soft thresholding operator.
/// Equivalent to shrink(x, threshold * weight).
#[inline]
pub fn weighted_shrink(x: f64, threshold: f64, weight: f64) -> f64 {
    shrink(x, threshold * weight)
}

/// Vector 2-norm (L2 norm).
#[inline]
pub fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Compute relative change ||x_new - x_prev||₂ / (||x_new||₂ + ε) for convergence checking.
#[inline]
pub fn relative_change(x: &[f64], x_prev: &[f64]) -> f64 {
    let (mut norm_diff_sq, mut norm_x_sq) = (0.0, 0.0);
    for i in 0..x.len() {
        let diff = x[i] - x_prev[i];
        norm_diff_sq += diff * diff;
        norm_x_sq += x[i] * x[i];
    }
    norm_diff_sq.sqrt() / (norm_x_sq.sqrt() + 1e-20)
}

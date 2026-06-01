//! Utility functions for QSM processing
//!
//! This module provides common utility functions:
//! - Gradient operators (forward/backward differences)
//! - Padding utilities
//! - Mask operations
//! - Multi-echo field mapping (phase offset removal, B0 estimation)
//! - Bias field correction (makehomogeneous)
//! - Frangi vesselness filter for vessel detection
//! - Surface curvature calculation
//! - Vasculature mask generation
//! - QSMART offset adjustment and utilities
//! - SIMD-accelerated operations (optional, with `simd` feature)

pub mod gradient;
pub mod padding;
pub mod multi_echo;
pub mod bias_correction;
pub mod frangi;
pub mod curvature;
pub mod vasculature;
pub mod qsmart;
pub mod threshold;
pub mod mask;
pub mod simd_ops;
pub mod r2star;

pub use gradient::*;
pub use padding::*;
pub use multi_echo::*;
pub use bias_correction::*;
pub use frangi::*;
pub use curvature::*;
pub use vasculature::*;
pub use qsmart::*;
pub use threshold::*;
pub use mask::*;
pub use simd_ops::*;
pub use r2star::*;

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

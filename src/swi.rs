//! Susceptibility Weighted Imaging (SWI)
//!
//! SWI enhances susceptibility contrast by combining magnitude and phase
//! information. Phase is high-pass filtered, converted to a [0, 1] mask,
//! and multiplied with magnitude.
//!
//! Reference:
//! Eckstein, K., et al. (2021). "Computationally efficient combination of
//! multi-channel phase data from multi-echo acquisitions (ASPIRE)."
//! Magnetic Resonance in Medicine, 79:2996-3006.
//! https://doi.org/10.1002/mrm.26963
//!
//! Reference implementation: https://github.com/korbinian90/CLEARSWI.jl

use crate::utils::gaussian_smooth_3d;

/// Phase mask scaling type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PhaseScaling {
    /// Sigmoid weighting: `(1 + tanh(1 - x/m)) / 2`
    /// where `m = median(positive_phase) * 10 / strength`
    Tanh,
    /// Negate phase first, then apply Tanh
    NegativeTanh,
    /// Traditional SWI: positive phase suppressed, negative → 1
    Positive,
    /// Traditional SWI: negative phase suppressed, positive → 1
    Negative,
    /// Both positive and negative phase suppressed
    Triangular,
}

/// High-pass filter by subtracting Gaussian-smoothed version
///
/// # Arguments
/// * `data` - Input data (e.g. unwrapped phase)
/// * `mask` - Binary mask (1 = inside, 0 = outside)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `sigma` - Gaussian sigma for each dimension in voxels (e.g. [4, 4, 0])
///
/// # Returns
/// High-pass filtered data
pub fn highpass_filter(
    data: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    sigma: [f64; 3],
) -> Vec<f64> {
    let nbox = 4; // masked smoothing uses nbox=4 in MriResearchTools
    let smoothed = gaussian_smooth_3d(data, sigma, Some(mask), None, nbox, nx, ny, nz);
    let n_total = nx * ny * nz;
    let mut result = vec![0.0; n_total];
    for i in 0..n_total {
        if mask[i] == 1 {
            result[i] = data[i] - smoothed[i];
        }
    }
    result
}

/// Create phase mask from filtered phase values
///
/// Converts phase to a [0, 1] weighting mask using the specified scaling.
///
/// # Arguments
/// * `phase` - High-pass filtered phase
/// * `mask` - Binary mask
/// * `scaling` - Phase scaling type
/// * `strength` - Scaling strength (higher = stronger phase contrast)
///
/// # Returns
/// Phase mask with values in [0, 1]
pub fn create_phase_mask(
    phase: &[f64],
    mask: &[u8],
    scaling: PhaseScaling,
    strength: f64,
) -> Vec<f64> {
    let n = phase.len();
    let mut result = vec![0.0; n];

    // Copy phase into result, zeroing outside mask
    for i in 0..n {
        if mask[i] == 1 {
            result[i] = phase[i];
        }
    }

    // Handle NegativeTanh by negating first
    let effective_scaling = if scaling == PhaseScaling::NegativeTanh {
        for v in result.iter_mut() {
            *v = -*v;
        }
        PhaseScaling::Tanh
    } else {
        scaling
    };

    match effective_scaling {
        PhaseScaling::Tanh => {
            // m = median(positive phase in mask) * 10 / strength
            let mut positives: Vec<f64> = (0..n)
                .filter(|&i| mask[i] == 1 && result[i] > 0.0)
                .map(|i| result[i])
                .collect();

            let m = if positives.is_empty() {
                1.0
            } else {
                positives.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                let mid = positives.len() / 2;
                let median = if positives.len().is_multiple_of(2) {
                    (positives[mid - 1] + positives[mid]) / 2.0
                } else {
                    positives[mid]
                };
                median * 10.0 / strength
            };

            for v in result.iter_mut() {
                *v = (1.0 + (1.0 - *v / m).tanh()) / 2.0;
            }
        }
        PhaseScaling::Positive => {
            // Positive phase: rescale to [1,0] then ^strength; negative → 1
            let (min_pos, max_pos) = positive_range(&result, mask);
            for i in 0..n {
                if result[i] > 0.0 && mask[i] == 1 {
                    result[i] = rescale(result[i], min_pos, max_pos, 1.0, 0.0).powf(strength);
                } else {
                    result[i] = 1.0;
                }
            }
        }
        PhaseScaling::Negative => {
            // Negative phase: rescale to [0,1] then ^strength; positive → 1
            let (min_neg, max_neg) = negative_range(&result, mask);
            for i in 0..n {
                if result[i] <= 0.0 && mask[i] == 1 {
                    result[i] = rescale(result[i], min_neg, max_neg, 0.0, 1.0).powf(strength);
                } else {
                    result[i] = 1.0;
                }
            }
        }
        PhaseScaling::Triangular => {
            // Both directions suppressed
            let (min_pos, max_pos) = positive_range(&result, mask);
            let (min_neg, max_neg) = negative_range(&result, mask);
            for i in 0..n {
                if mask[i] == 0 {
                    result[i] = 0.0;
                } else if result[i] > 0.0 {
                    result[i] = rescale(result[i], min_pos, max_pos, 1.0, 0.0).powf(strength);
                } else {
                    result[i] = rescale(result[i], min_neg, max_neg, 0.0, 1.0).powf(strength);
                }
            }
        }
        PhaseScaling::NegativeTanh => unreachable!(),
    }

    // Clamp to [0, 1]
    for v in &mut result {
        if *v < 0.0 {
            *v = 0.0;
        }
    }

    // Zero outside mask
    for i in 0..n {
        if mask[i] == 0 {
            result[i] = 0.0;
        }
    }

    result
}

/// Calculate SWI from unwrapped phase and magnitude
///
/// Pipeline: high-pass filter phase → create phase mask → multiply with magnitude.
///
/// # Arguments
/// * `phase` - Unwrapped phase (single echo or combined)
/// * `magnitude` - Magnitude image (single echo or combined)
/// * `mask` - Binary brain mask
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm (unused, reserved for consistency)
/// * `hp_sigma` - High-pass filter sigma in voxels [x, y, z]
/// * `scaling` - Phase scaling type
/// * `strength` - Phase scaling strength
///
/// # Returns
/// SWI image (magnitude × phase mask)
#[allow(clippy::too_many_arguments)]
pub fn calculate_swi(
    phase: &[f64],
    magnitude: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    _vsx: f64, _vsy: f64, _vsz: f64,
    hp_sigma: [f64; 3],
    scaling: PhaseScaling,
    strength: f64,
) -> Vec<f64> {
    let n_total = nx * ny * nz;

    // High-pass filter phase
    let filtered = highpass_filter(phase, mask, nx, ny, nz, hp_sigma);

    // Create phase mask
    let phase_mask = create_phase_mask(&filtered, mask, scaling, strength);

    // SWI = magnitude × phase_mask
    let mut swi = vec![0.0; n_total];
    for i in 0..n_total {
        swi[i] = magnitude[i] * phase_mask[i];
    }

    swi
}

/// Calculate SWI with default parameters
///
/// Defaults: sigma=[4,4,0], Tanh scaling, strength=4
#[allow(clippy::too_many_arguments)]
pub fn calculate_swi_default(
    phase: &[f64],
    magnitude: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> Vec<f64> {
    calculate_swi(
        phase, magnitude, mask,
        nx, ny, nz,
        vsx, vsy, vsz,
        [4.0, 4.0, 0.0],
        PhaseScaling::Tanh,
        4.0,
    )
}

/// Minimum intensity projection along the z-axis
///
/// For each (x, y) position, takes the minimum value over a sliding window
/// of `window` slices along z.
///
/// # Arguments
/// * `data` - 3D volume (Fortran order)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `window` - Number of slices in the projection window
///
/// # Returns
/// MIP volume with dimensions `nx × ny × (nz - window + 1)`.
/// Returns empty vec if `window > nz`.
pub fn create_mip(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    window: usize,
) -> Vec<f64> {
    if window > nz || window == 0 {
        return vec![];
    }

    let nz_out = nz - window + 1;
    let nxy = nx * ny;
    let mut mip = vec![0.0; nxy * nz_out];

    for k_out in 0..nz_out {
        for j in 0..ny {
            for i in 0..nx {
                let idx_xy = i + j * nx;
                let mut min_val = data[idx_xy + k_out * nxy];
                for kw in 1..window {
                    let val = data[idx_xy + (k_out + kw) * nxy];
                    if val < min_val {
                        min_val = val;
                    }
                }
                mip[idx_xy + k_out * nxy] = min_val;
            }
        }
    }

    mip
}

/// Minimum intensity projection with default window of 7 slices
pub fn create_mip_default(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
) -> Vec<f64> {
    create_mip(data, nx, ny, nz, 7)
}

/// Softplus magnitude scaling for enhanced contrast
///
/// Applies a shifted softplus function: `softplus(x) - softplus(0)` where
/// `softplus(x) = (log(1 + exp(-|f*(x-offset)|)) + max(0, f*(x-offset))) / f`
/// and `f = factor / offset`.
///
/// # Arguments
/// * `magnitude` - Input magnitude data
/// * `offset` - Softplus offset (controls transition point)
/// * `factor` - Steepness factor (default 2.0)
///
/// # Returns
/// Scaled magnitude
pub fn softplus_scaling(
    magnitude: &[f64],
    offset: f64,
    factor: f64,
) -> Vec<f64> {
    if offset.abs() < 1e-20 {
        return magnitude.to_vec();
    }

    let f = factor / offset;

    // softplus(0) for baseline subtraction
    let arg0 = f * (0.0 - offset);
    let sp0 = ((1.0 + (-arg0.abs()).exp()).ln() + arg0.max(0.0)) / f;

    magnitude.iter().map(|&val| {
        let arg = f * (val - offset);
        let sp = ((1.0 + (-arg.abs()).exp()).ln() + arg.max(0.0)) / f;
        sp - sp0
    }).collect()
}

// ---- Helpers ----

/// Get min/max of positive values within mask
fn positive_range(data: &[f64], mask: &[u8]) -> (f64, f64) {
    let mut min_val = f64::MAX;
    let mut max_val = f64::MIN;
    for i in 0..data.len() {
        if mask[i] == 1 && data[i] > 0.0 {
            if data[i] < min_val { min_val = data[i]; }
            if data[i] > max_val { max_val = data[i]; }
        }
    }
    if min_val > max_val {
        (0.0, 1.0) // fallback
    } else {
        (min_val, max_val)
    }
}

/// Get min/max of non-positive values within mask
fn negative_range(data: &[f64], mask: &[u8]) -> (f64, f64) {
    let mut min_val = f64::MAX;
    let mut max_val = f64::MIN;
    for i in 0..data.len() {
        if mask[i] == 1 && data[i] <= 0.0 {
            if data[i] < min_val { min_val = data[i]; }
            if data[i] > max_val { max_val = data[i]; }
        }
    }
    if min_val > max_val {
        (-1.0, 0.0) // fallback
    } else {
        (min_val, max_val)
    }
}

/// Linear rescale from [old_min, old_max] to [new_min, new_max]
#[inline]
fn rescale(val: f64, old_min: f64, old_max: f64, new_min: f64, new_max: f64) -> f64 {
    let range = old_max - old_min;
    if range.abs() < 1e-20 {
        return (new_min + new_max) / 2.0;
    }
    let t = (val - old_min) / range;
    // Clamp t to [0, 1] for robustness
    let t = t.clamp(0.0, 1.0);
    new_min + t * (new_max - new_min)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_swi_zero_phase() {
        let n = 8;
        let nn = n * n * n;
        let phase = vec![0.0; nn];
        let magnitude = vec![1.0; nn];
        let mask = vec![1u8; nn];

        let swi = calculate_swi_default(&phase, &magnitude, &mask, n, n, n, 1.0, 1.0, 1.0);

        // With zero phase, tanh mask gives (1 + tanh(1)) / 2 ≈ 0.88
        for &v in &swi {
            assert!(v.is_finite(), "SWI values should be finite");
            assert!(v >= 0.0, "SWI values should be non-negative");
        }
    }

    #[test]
    fn test_calculate_swi_mask() {
        let n = 8;
        let nn = n * n * n;
        let phase = vec![0.1; nn];
        let magnitude = vec![1.0; nn];
        let mut mask = vec![1u8; nn];
        mask[0] = 0;
        mask[1] = 0;

        let swi = calculate_swi_default(&phase, &magnitude, &mask, n, n, n, 1.0, 1.0, 1.0);

        assert_eq!(swi[0], 0.0, "Outside mask should be 0");
        assert_eq!(swi[1], 0.0, "Outside mask should be 0");
    }

    #[test]
    fn test_phase_mask_range() {
        let n = 10;
        let nn = n * n * n;
        let phase: Vec<f64> = (0..nn).map(|i| (i as f64 * 0.01) - 5.0).collect();
        let mask = vec![1u8; nn];

        for scaling in &[
            PhaseScaling::Tanh,
            PhaseScaling::NegativeTanh,
            PhaseScaling::Positive,
            PhaseScaling::Negative,
            PhaseScaling::Triangular,
        ] {
            let pm = create_phase_mask(&phase, &mask, *scaling, 4.0);
            for (i, &v) in pm.iter().enumerate() {
                assert!(v >= 0.0, "{:?}: value at {} = {} < 0", scaling, i, v);
                assert!(v <= 1.0 + 1e-10, "{:?}: value at {} = {} > 1", scaling, i, v);
            }
        }
    }

    #[test]
    fn test_highpass_filter_constant() {
        // Constant input should give zero output (constant is its own smooth)
        let n = 16;
        let nn = n * n * n;
        let data = vec![5.0; nn];
        let mask = vec![1u8; nn];

        let result = highpass_filter(&data, &mask, n, n, n, [2.0, 2.0, 0.0]);

        for &v in &result {
            assert!(v.abs() < 1.0, "High-pass of constant should be near zero, got {}", v);
        }
    }

    #[test]
    fn test_mip_basic() {
        // 3x3x5 volume, mIP with window=3 → 3x3x3 output
        let (nx, ny, nz) = (3, 3, 5);
        let mut data = vec![10.0; nx * ny * nz];
        // Place a low value at slice 2
        let idx = 1 + 1 * nx + 2 * nx * ny; // (1,1,2)
        data[idx] = 1.0;

        let mip = create_mip(&data, nx, ny, nz, 3);
        assert_eq!(mip.len(), nx * ny * 3);

        // The minimum at (1,1) should appear in slices that include z=2
        // Window starting at z=0: slices 0,1,2 → includes the 1.0
        let mip_idx_0 = 1 + 1 * nx + 0 * nx * ny;
        assert_eq!(mip[mip_idx_0], 1.0);
        // Window starting at z=1: slices 1,2,3 → includes the 1.0
        let mip_idx_1 = 1 + 1 * nx + 1 * nx * ny;
        assert_eq!(mip[mip_idx_1], 1.0);
        // Window starting at z=2: slices 2,3,4 → includes the 1.0
        let mip_idx_2 = 1 + 1 * nx + 2 * nx * ny;
        assert_eq!(mip[mip_idx_2], 1.0);
    }

    #[test]
    fn test_mip_window_too_large() {
        let mip = create_mip(&[1.0; 27], 3, 3, 3, 10);
        assert!(mip.is_empty());
    }

    #[test]
    fn test_softplus_scaling() {
        let mag = vec![0.0, 0.5, 1.0, 2.0];
        let result = softplus_scaling(&mag, 1.0, 2.0);

        // softplus(0, offset=1, factor=2) should be 0 (baseline subtracted)
        assert!(result[0].abs() < 1e-10, "softplus(0) should be ~0, got {}", result[0]);
        // Values should increase monotonically
        for i in 1..result.len() {
            assert!(result[i] >= result[i - 1], "softplus should be monotonically increasing");
        }
    }

    #[test]
    fn test_rescale() {
        assert!((rescale(0.0, 0.0, 10.0, 0.0, 1.0) - 0.0).abs() < 1e-10);
        assert!((rescale(5.0, 0.0, 10.0, 0.0, 1.0) - 0.5).abs() < 1e-10);
        assert!((rescale(10.0, 0.0, 10.0, 0.0, 1.0) - 1.0).abs() < 1e-10);
        // Inverted rescale
        assert!((rescale(0.0, 0.0, 10.0, 1.0, 0.0) - 1.0).abs() < 1e-10);
        assert!((rescale(10.0, 0.0, 10.0, 1.0, 0.0) - 0.0).abs() < 1e-10);
    }
}

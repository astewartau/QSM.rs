//! Spatially Dependent Filtering (SDF) for QSMART
//!
//! SDF is the background field removal method used in QSMART. It uses variable-radius
//! Gaussian filtering where the kernel size depends on the proximity to the brain boundary.
//! This allows for aggressive filtering in the brain interior while preserving details
//! near the surface.
//!
//! The algorithm includes optional curvature-based weighting to further reduce artifacts
//! at highly curved brain regions.
//!
//! ## Reference pipeline
//!
//! QSMART runs SDF twice on its echo-fit total field (`QSMART_SDF.m`): stage 1 on the BET mask
//! × `R_0` with no vessels and `sdf_sigma1` (10) passed as **both** widths (l. 12), so σ =
//! √(10² + 10²) ≈ 14.1 (`sdf_curvature.m` l. 8); stage 2 on the field × mask × `R_0` with the
//! vessel mask, σ1 = 8, σ2 = 2 (l. 20). Both use `lowerLim` 0.6, curvature constant 500 and
//! an indent mask from closing the brain mask with `strel('sphere', 8)` voxels
//! (`Demo_QSMART.m` l. 51–57). The paper instead lists σ_C = 10, σ_V = 2, σ_I = 10, σ = 10,
//! C_s = 500 (Yaghmaie et al. 2021, §3.4); [`SdfParams`] follows the toolbox. The mask is not
//! eroded first (paper §1, §5). Parameters were tuned on 7T human data at 0.6 mm; the Gaussian
//! widths are in voxels (`imgaussfilt3`), so they are not rescaled for other voxel sizes, and
//! the cortical proximity map uses `[σ1, 2σ1, 2σ1]` (`sdf_curvature.m` l. 12): half the width
//! along the first voxel axis, whatever the orientation.
//!
//! Reference:
//! Yaghmaie, N., Syeda, W., et al. (2021).
//! "QSMART: Quantitative Susceptibility Mapping Artifact Reduction Technique."
//! NeuroImage, 231:117701. https://doi.org/10.1016/j.neuroimage.2020.117701
//!
//! Reference implementation: https://github.com/wtsyeda/QSMART
//!
//! # Agreement with the original
//!
//! `tests/qsmart_qsmci_parity.rs` (measured 2026-10-30) runs QSMART's stages on the inputs the
//! verified QSM-CI `qsmart-matlab` v3 container (QSMxT/QSM-CI#296, bit-identical to licensed
//! MATLAB) gave its own, on a 36×36×28 vessel phantom at 3 T; the fixture is regenerated in Actions
//! by `tests/data/qsmart_qsmci_small.py`. Given the toolbox's total field and `mask·R_0` (and stage
//! 2 its vessel mask), [`sdf`] is **10.9 %** from `QSMART_SDF` stage 1 (r 0.9939, max |Δ| 4.6e-3
//! ppm) and **2.1 %** from stage 2 (r 0.99977): an open gap (astewartau/QSM.rs#320), not rounding.
//! Stage 1's widths are QSM.rs's reading of the toolbox (σ1 = σ2 = 10, above); with σ2 = 0, as
//! `recon.m`'s `s1.sdf_sigma2`, it is 149 % off, so the gap is elsewhere.

use crate::Grid;
use crate::utils::curvature::calculate_curvature_proximity;
use crate::utils::portable_math as pm;

/// Parameters for SDF background field removal
///
/// There is no indent/closing radius: the reference `QSMART_SDF.m` builds an indent mask from
/// `sdf_sp_radius` but never passes it to `sdf_curvature.m`, so the radius has no effect.
#[derive(Clone, Debug)]
pub struct SdfParams {
    /// Sigma parameter for initial proximity map (default 10 for stage 1, 8 for stage 2)
    pub sigma1: f64,
    /// Sigma parameter for vasculature proximity (default 0 for stage 1, 2 for stage 2)
    pub sigma2: f64,
    /// Lower limit for clamping proximity values (default 0.6)
    pub lower_lim: f64,
    /// Scaling constant for curvature (default 500)
    pub curv_constant: f64,
    /// Whether to use curvature-based edge refinement
    pub use_curvature: bool,
}

impl Default for SdfParams {
    fn default() -> Self {
        Self {
            sigma1: 10.0,
            // MATLAB passes sigma1 for both sigma args in stage 1.
            // sigma2 affects the combined sigma even though vasculature
            // proximity is skipped when vasc_only is all-ones.
            sigma2: 10.0,
            lower_lim: 0.6,
            curv_constant: 500.0,
            use_curvature: true,
        }
    }
}

impl SdfParams {
    /// Create parameters for QSMART Stage 1
    /// MATLAB passes sigma1 for both sigma args in stage 1.
    pub fn stage1() -> Self {
        Self {
            sigma1: 10.0,
            sigma2: 10.0,
            lower_lim: 0.6,
            curv_constant: 500.0,
            use_curvature: true,
        }
    }

    /// Create parameters for QSMART Stage 2
    pub fn stage2() -> Self {
        Self {
            sigma1: 8.0,
            sigma2: 2.0,
            lower_lim: 0.6,
            curv_constant: 500.0,
            use_curvature: true,
        }
    }
}

/// SDF background field removal
///
/// Removes background field from total field shift using spatially dependent filtering.
///
/// # Arguments
/// * `tfs` - Total field shift (unwrapped phase / ppm)
/// * `mask` - Binary brain mask (weighted by reliability if R_0 is incorporated)
/// * `vasc_only` - Vasculature-only mask (1 = tissue, 0 = vessel). Pass all-ones for stage 1.
/// * `grid` - Volume dimensions and voxel sizes
/// * `params` - SDF parameters
/// * `progress` - Progress callback (alpha_index, total_alphas)
///
/// # Returns
/// Local field shift (background removed)
pub fn sdf(
    tfs: &[f64],
    mask: &[f64],
    vasc_only: &[f64],
    grid: &Grid,
    params: &SdfParams,
    progress: impl Fn(usize, usize),
) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let n_total = nx * ny * nz;

    // Convert mask to binary for morphological operations
    let mask_binary: Vec<u8> = mask.iter().map(|&v| if v > 0.0 { 1 } else { 0 }).collect();

    // Combined sigma for n calculation
    let sigma = (params.sigma1 * params.sigma1 + params.sigma2 * params.sigma2).sqrt();
    let n = if sigma > 0.0 { -pm::ln(sigma) / pm::ln(0.5) } else { 0.0 };

    // Calculate initial proximity map (prox1)
    // Gaussian smoothing of mask with anisotropic kernel [sigma1, 2*sigma1, 2*sigma1]
    let prox1 = if params.sigma1 > 0.0 {
        gaussian_smooth_3d_masked_f64(mask, mask, nx, ny, nz, &[params.sigma1, 2.0 * params.sigma1, 2.0 * params.sigma1])
    } else {
        mask.to_vec()
    };

    // Calculate curvature-based proximity if enabled
    let prox = if params.use_curvature {
        let (prox_curv, _curv_i) = calculate_curvature_proximity(
            &mask_binary,
            &prox1,
            params.lower_lim,
            params.curv_constant,
            params.sigma1,
            grid,
        );
        prox_curv
    } else {
        // Even without curvature, clamp proximity to lower_lim to prevent
        // filter sizes from getting too small at the edges
        // (matching calculate_curvature.m line 45: prox(prox < lowerLim & prox ~= 0) = lowerLim)
        prox1.iter()
            .zip(mask.iter())
            .map(|(&p, &m)| {
                if m > 0.0 && p > 0.0 && p < params.lower_lim {
                    params.lower_lim
                } else {
                    p
                }
            })
            .collect()
    };

    // Calculate vasculature proximity (prox2) for stage 2
    let prox_final = if params.sigma2 > 0.0 {
        let prox2 = gaussian_smooth_3d_masked_f64(vasc_only, mask, nx, ny, nz, &[params.sigma2, params.sigma2, params.sigma2]);
        // Multiply prox * prox2
        prox.iter().zip(prox2.iter()).map(|(&p, &p2)| p * p2).collect()
    } else {
        prox
    };

    // alpha = sigma*round(prox.^n,2).*mask (sdf_curvature.m): each voxel's smoothing width
    let mut alpha: Vec<f64> = prox_final.iter()
        .zip(mask.iter())
        .map(|(&p, &m)| if m > 0.0 { smoothing_width(p, n, sigma) } else { 0.0 })
        .collect();

    // Set alpha=1 for vessel regions within mask
    // (vasc_only=0 means vessel, matching sdf_curvature.m line 27)
    for i in 0..n_total {
        if mask[i] > 0.0 && vasc_only[i] == 0.0 {
            alpha[i] = 1.0;
        }
    }

    // Get unique alpha values and sort
    let mut unique_alphas: Vec<f64> = alpha.iter()
        .filter(|&&a| a > 0.0)
        .copied()
        .collect();
    unique_alphas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    unique_alphas.dedup();

    let total_alphas = unique_alphas.len();

    // Create index map: for each voxel, which alpha index does it belong to?
    let mut alpha_index: Vec<usize> = vec![0; n_total];
    for i in 0..n_total {
        if alpha[i] > 0.0 {
            // Find index in unique_alphas
            let idx = unique_alphas.iter().position(|&a| (a - alpha[i]).abs() < 1e-10).unwrap_or(0);
            alpha_index[i] = idx + 1; // 1-indexed to distinguish from background
        }
    }

    // Apply spatially dependent filtering
    // For each unique alpha, smooth and assign to corresponding voxels
    let mut background = vec![0.0f64; n_total];

    // Pre-compute filter size
    let filter_size = 2 * (2.0 * sigma).ceil() as usize + 1;

    for (alpha_idx, &current_alpha) in unique_alphas.iter().enumerate() {
        progress(alpha_idx, total_alphas);

        // Compute smoothed field for this alpha
        let smoothed: Vec<f64> = if current_alpha > 0.0 {
            // Smooth tfs * mask with Gaussian kernel of size current_alpha
            let weighted_tfs: Vec<f64> = tfs.iter()
                .zip(mask.iter())
                .map(|(&t, &m)| t * m)
                .collect();

            let num = gaussian_smooth_3d_with_filter_size(&weighted_tfs, nx, ny, nz, current_alpha, filter_size);
            let denom = gaussian_smooth_3d_with_filter_size(mask, nx, ny, nz, current_alpha, filter_size);

            // Divide: num / denom
            num.iter()
                .zip(denom.iter())
                .map(|(&n, &d)| if d.abs() > 1e-10 { n / d } else { 0.0 })
                .collect()
        } else {
            // alpha=0: just use tfs*mask
            tfs.iter().zip(mask.iter()).map(|(&t, &m)| t * m).collect()
        };

        // Assign to voxels with this alpha
        for i in 0..n_total {
            if alpha_index[i] == alpha_idx + 1 {
                background[i] = smoothed[i];
            }
        }
    }

    // sdf_curvature.m filters alpha == 0 as `PhiK = tfs.*mask`: where the rounded proximity is
    // 0, the background is the whole field and the local field is 0.
    for i in 0..n_total {
        if mask[i] > 0.0 && alpha[i] == 0.0 {
            background[i] = tfs[i] * mask[i];
        }
    }

    progress(total_alphas, total_alphas);

    // Compute local field: (tfs - background) * mask
    let local_field: Vec<f64> = tfs.iter()
        .zip(background.iter())
        .zip(mask.iter())
        .map(|((&t, &b), &m)| (t - b) * m)
        .collect();

    local_field
}

/// The Gaussian width SDF smooths a voxel with: `sigma*round(prox^n, 2)`, as `sdf_curvature.m`
/// computes it.
///
/// Rounding to two decimals makes the width a discrete choice, which has to come out the same on
/// every target (or the browser and `qsmxt run` smooth a voxel on a rounding boundary with
/// different widths, and the inner inversion spreads the difference) and the same as MATLAB's:
///
/// - `prox^n` comes from [`portable_math::pow`](crate::utils::portable_math::pow), and the
///   proximity map it is computed from uses only portable functions too (the Gaussian kernels
///   here and the curvature's angles). The platform's `powf` differs between targets in the last
///   bit for about one input in ten, and a one-ulp difference flips the rounding whenever the
///   value sits within an ulp of a hundredth's midpoint.
/// - The operations are MATLAB's, in MATLAB's order: `round(x, 2)` is `round(x*100)/100`, with
///   halves rounded away from zero (as [`f64::round`] does), then multiplied by `sigma`. With
///   QSMART's own proximity map (`prox.nii` from its 3 T reference run) as input, this gives its
///   `alpha.nii` bit for bit, for both stages. `sigma*round(x*100)/100` evaluated left to right
///   (as before) differs in the last bit in a quarter of the brain.
///
/// No libm is correctly rounded, MATLAB's included, so a value within an ulp of a midpoint can
/// still round differently from MATLAB; away from midpoints every `pow` within an ulp of the
/// exact value gives MATLAB's width.
fn smoothing_width(prox: f64, n: f64, sigma: f64) -> f64 {
    sigma * ((pm::pow(prox, n) * 100.0).round() / 100.0)
}

/// SDF with curvature-based weighting (full QSMART pipeline)
///
/// This is the main entry point matching QSMART's sdf_curvature function.
pub fn sdf_curvature(
    tfs: &[f64],
    mask: &[f64],
    vasc_only: &[f64],
    grid: &Grid,
    params: &SdfParams,
) -> Vec<f64> {
    // Ensure curvature is enabled
    let params_with_curv = SdfParams {
        use_curvature: true,
        ..params.clone()
    };

    sdf(tfs, mask, vasc_only, grid, &params_with_curv, |_, _| {})
}

/// 3D Gaussian smoothing with specified filter size
fn gaussian_smooth_3d_with_filter_size(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    sigma: f64,
    filter_size: usize,
) -> Vec<f64> {
    if sigma <= 0.0 {
        return data.to_vec();
    }

    // Create 1D Gaussian kernel
    let kernel_radius = (filter_size - 1) / 2;
    let mut kernel = vec![0.0f64; filter_size];

    let mut sum = 0.0;
    for i in 0..filter_size {
        let x = i as f64 - kernel_radius as f64;
        kernel[i] = pm::exp(-x * x / (2.0 * sigma * sigma));
        sum += kernel[i];
    }

    // Normalize
    for k in kernel.iter_mut() {
        *k /= sum;
    }

    // Apply separable convolution
    let smoothed_x = convolve_1d_direction(data, nx, ny, nz, &kernel, 'x');
    let smoothed_xy = convolve_1d_direction(&smoothed_x, nx, ny, nz, &kernel, 'y');
    convolve_1d_direction(&smoothed_xy, nx, ny, nz, &kernel, 'z')
}

/// Gaussian smoothing with anisotropic sigma and mask
fn gaussian_smooth_3d_masked_f64(
    data: &[f64],
    mask: &[f64],
    nx: usize, ny: usize, nz: usize,
    sigmas: &[f64; 3],
) -> Vec<f64> {
    // Apply separable 1D convolutions
    let smoothed_x = convolve_1d_direction_sigma(data, nx, ny, nz, sigmas[0], 'x');
    let smoothed_xy = convolve_1d_direction_sigma(&smoothed_x, nx, ny, nz, sigmas[1], 'y');
    let smoothed_xyz = convolve_1d_direction_sigma(&smoothed_xy, nx, ny, nz, sigmas[2], 'z');

    // Apply mask
    smoothed_xyz.iter()
        .zip(mask.iter())
        .map(|(&v, &m)| if m > 0.0 { v } else { 0.0 })
        .collect()
}

/// 1D convolution along specified axis with replicate padding
/// Matches MATLAB's imgaussfilt3 default behavior
fn convolve_1d_direction(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    kernel: &[f64],
    direction: char,
) -> Vec<f64> {
    let n_total = nx * ny * nz;
    let mut result = vec![0.0f64; n_total];
    let kernel_radius = (kernel.len() - 1) / 2;

    let idx = |i: usize, j: usize, k: usize| i + j * nx + k * nx * ny;

    // Helper to clamp index for replicate padding
    let clamp_x = |x: isize| -> usize { x.max(0).min(nx as isize - 1) as usize };
    let clamp_y = |y: isize| -> usize { y.max(0).min(ny as isize - 1) as usize };
    let clamp_z = |z: isize| -> usize { z.max(0).min(nz as isize - 1) as usize };

    match direction {
        'x' => {
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let mut sum = 0.0;

                        for ki in 0..kernel.len() {
                            let offset = ki as isize - kernel_radius as isize;
                            let ni = clamp_x(i as isize + offset);
                            sum += data[idx(ni, j, k)] * kernel[ki];
                        }

                        result[idx(i, j, k)] = sum;
                    }
                }
            }
        }
        'y' => {
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let mut sum = 0.0;

                        for ki in 0..kernel.len() {
                            let offset = ki as isize - kernel_radius as isize;
                            let nj = clamp_y(j as isize + offset);
                            sum += data[idx(i, nj, k)] * kernel[ki];
                        }

                        result[idx(i, j, k)] = sum;
                    }
                }
            }
        }
        'z' => {
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let mut sum = 0.0;

                        for ki in 0..kernel.len() {
                            let offset = ki as isize - kernel_radius as isize;
                            let nk = clamp_z(k as isize + offset);
                            sum += data[idx(i, j, nk)] * kernel[ki];
                        }

                        result[idx(i, j, k)] = sum;
                    }
                }
            }
        }
        _ => panic!("Invalid convolution direction"),
    }

    result
}

/// 1D convolution with specified sigma
fn convolve_1d_direction_sigma(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    sigma: f64,
    direction: char,
) -> Vec<f64> {
    if sigma <= 0.0 {
        return data.to_vec();
    }

    // Create 1D Gaussian kernel
    // Match MATLAB's imgaussfilt3 default: filterSize = 2*ceil(2*sigma)+1
    let kernel_radius = (2.0 * sigma).ceil() as usize;
    let kernel_size = 2 * kernel_radius + 1;
    let mut kernel = vec![0.0f64; kernel_size];

    let mut sum = 0.0;
    for i in 0..kernel_size {
        let x = i as f64 - kernel_radius as f64;
        kernel[i] = pm::exp(-x * x / (2.0 * sigma * sigma));
        sum += kernel[i];
    }

    // Normalize
    for k in kernel.iter_mut() {
        *k /= sum;
    }

    convolve_1d_direction(data, nx, ny, nz, &kernel, direction)
}

/// Simple SDF without curvature (faster, for testing)
pub fn sdf_simple(
    tfs: &[f64],
    mask: &[f64],
    grid: &Grid,
    sigma1: f64,
) -> Vec<f64> {
    let vasc_only = vec![1.0f64; mask.len()];
    let params = SdfParams {
        sigma1,
        sigma2: 0.0,
        use_curvature: false,
        ..Default::default()
    };

    sdf(tfs, mask, &vasc_only, grid, &params, |_, _| {})
}

#[cfg(test)]
mod tests {
    #[test]
    fn zero_alpha_leaves_no_local_field() {
        // One tissue voxel amid vessel: its vessel proximity (the Gaussian of the tissue mask,
        // sigma2 = 4) is ~1e-3, so alpha = sigma * round(prox^n, 2) is 0 there, and
        // sdf_curvature.m takes the whole field as its background: local field 0.
        let (nx, ny, nz) = (16, 16, 16);
        let n = nx * ny * nz;
        let grid = Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);
        let mask = vec![1.0; n];
        let centre = 8 + 8 * nx + 8 * nx * ny;
        let mut vasc = vec![0.0; n];
        vasc[centre] = 1.0;
        let tfs: Vec<f64> = (0..n).map(|i| 1.0 + 0.01 * (i % nx) as f64).collect();
        let params = SdfParams { sigma1: 1.0, sigma2: 4.0, use_curvature: false, ..SdfParams::default() };
        let lfs = sdf(&tfs, &mask, &vasc, &grid, &params, |_, _| {});
        assert_eq!(lfs[centre], 0.0);
    }

    use super::*;

    #[test]
    fn test_sdf_simple() {
        // Simple test: constant field should give zero local field
        let n = 10;
        let n_total = n * n * n;

        let tfs = vec![1.0f64; n_total];
        let mask = vec![1.0f64; n_total];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        let lfs = sdf_simple(&tfs, &mask, &grid, 2.0);

        // Local field should be near zero for constant total field
        let max_lfs = lfs.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        assert!(max_lfs < 0.1, "Max LFS was {}", max_lfs);
    }

    #[test]
    fn test_gaussian_smooth_constant() {
        // Smoothing constant field should give same constant
        let data = vec![5.0f64; 27];
        let smoothed = gaussian_smooth_3d_with_filter_size(&data, 3, 3, 3, 1.0, 5);

        for &v in &smoothed {
            assert!((v - 5.0).abs() < 0.1);
        }
    }
}

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

use crate::Grid;
use crate::utils::{gaussian_smooth_3d, apply_mask_zero};

/// SWI algorithm parameters
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct SwiParams {
    /// High-pass filter sigma in voxels [x, y, z]
    pub hp_sigma: [f64; 3],
    /// Phase scaling type
    pub scaling: PhaseScaling,
    /// Phase scaling strength
    pub strength: f64,
    /// MIP window size in slices
    pub mip_window: usize,
}

impl Default for SwiParams {
    fn default() -> Self {
        Self {
            hp_sigma: [4.0, 4.0, 0.0],
            scaling: PhaseScaling::Tanh,
            strength: 4.0,
            mip_window: 7,
        }
    }
}

/// Phase mask scaling type
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
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
/// * `grid` - Volume grid (dimensions and voxel sizes)
/// * `sigma` - Gaussian sigma for each dimension in voxels (e.g. [4, 4, 0])
///
/// # Returns
/// High-pass filtered data
pub fn highpass_filter(
    data: &[f64],
    mask: &[u8],
    grid: &Grid,
    sigma: [f64; 3],
) -> Vec<f64> {
    let nbox = 4; // masked smoothing uses nbox=4 in MriResearchTools
    let smoothed = gaussian_smooth_3d(data, sigma, Some(mask), None, nbox, grid);
    let n_total = grid.n_total();
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

    apply_mask_zero(&mut result, mask);

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
/// * `grid` - Volume grid (dimensions and voxel sizes)
/// * `params` - SWI algorithm parameters
///
/// # Returns
/// SWI image (magnitude × phase mask)
pub fn calculate_swi(
    phase: &[f64],
    magnitude: &[f64],
    mask: &[u8],
    grid: &Grid,
    params: &SwiParams,
) -> Vec<f64> {
    let n_total = grid.n_total();

    // High-pass filter phase
    let filtered = highpass_filter(phase, mask, grid, params.hp_sigma);

    // Create phase mask
    let phase_mask = create_phase_mask(&filtered, mask, params.scaling, params.strength);

    // SWI = magnitude × phase_mask
    let mut swi = vec![0.0; n_total];
    for i in 0..n_total {
        swi[i] = magnitude[i] * phase_mask[i];
    }

    swi
}

/// A minimum-intensity projection, and the geometry it actually lives on.
///
/// A projection is not the volume it came from: it is shorter along the slice axis, and each of
/// its slices stands for a slab rather than a plane. Both facts travel with the data here so a
/// caller cannot write it out under the source volume's geometry — which produces a file whose
/// header promises more voxels than it holds, and whose slices are misplaced in world space.
#[derive(Debug, Clone)]
pub struct Mip {
    /// Projection values, Fortran order, `grid.n_total()` of them.
    pub data: Vec<f64>,
    /// Dimensions and voxel sizes of the projection.
    pub grid: Grid,
    /// Row-major 4×4 voxel→world affine, with the origin at the centre of the first slab.
    pub affine: [f64; 16],
}

/// Geometry of a sliding projection over `window` slices, without computing one.
///
/// Returns the projection's grid and affine. Use it to size or place a projection ahead of
/// time — to validate a window before doing the work, or to allocate an output file.
///
/// The slice axis loses `window - 1` slices, since a sliding window of `window` slices has
/// `nz - window + 1` positions. Each output slice represents the **centre** of its slab, so the
/// origin moves `(window - 1) / 2` slices along the slice direction. That shift follows the
/// affine's third column rather than world z, which keeps it correct for an oblique
/// acquisition; an even window lands the origin on a half-slice offset, as the convention
/// implies.
///
/// # Errors
/// If `window` is zero, or deeper than the volume.
pub fn mip_geometry(
    grid: &Grid,
    affine: &[f64; 16],
    window: usize,
) -> Result<(Grid, [f64; 16]), String> {
    let (nx, ny, nz) = grid.dims;
    if window == 0 {
        return Err("MIP window must be at least 1 slice".to_string());
    }
    if window > nz {
        return Err(format!(
            "MIP window of {} slices is deeper than the {}-slice volume",
            window, nz,
        ));
    }

    let (vsx, vsy, vsz) = grid.voxel_size;
    let mip_grid = Grid::new(nx, ny, nz - window + 1, vsx, vsy, vsz);

    let mut mip_affine = *affine;
    let slabs = (window - 1) as f64 / 2.0;
    for row in 0..3 {
        // Row-major 4×4: column 2 is the slice direction, column 3 the origin.
        mip_affine[row * 4 + 3] = affine[row * 4 + 3] + affine[row * 4 + 2] * slabs;
    }

    Ok((mip_grid, mip_affine))
}

/// Minimum intensity projection along the z-axis.
///
/// For each (x, y) position, takes the minimum value over a sliding window of `window` slices
/// along z. The result is `window - 1` slices shorter than `data` and sits half a slab further
/// along the slice direction, so it comes back as a [`Mip`] carrying its own grid and affine —
/// see [`mip_geometry`] for the convention.
///
/// # Arguments
/// * `data` - 3D volume (Fortran order)
/// * `grid` - Volume grid (dimensions and voxel sizes)
/// * `affine` - Row-major 4×4 voxel→world affine of `data`. Pass an identity affine if the
///   caller has no world geometry; the projection's affine is then identity-with-origin-shift.
/// * `window` - Number of slices in the projection window
///
/// # Errors
/// If `window` is zero or deeper than the volume, or if `data` does not match `grid`.
pub fn create_mip(
    data: &[f64],
    grid: &Grid,
    affine: &[f64; 16],
    window: usize,
) -> Result<Mip, String> {
    let (nx, ny, nz) = grid.dims;
    if data.len() != grid.n_total() {
        return Err(format!(
            "MIP input is {} values but its {}x{}x{} grid has {} voxels",
            data.len(), nx, ny, nz, grid.n_total(),
        ));
    }
    let (mip_grid, mip_affine) = mip_geometry(grid, affine, window)?;

    let nz_out = mip_grid.nz();
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

    debug_assert_eq!(mip.len(), mip_grid.n_total());
    Ok(Mip { data: mip, grid: mip_grid, affine: mip_affine })
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
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        let swi = calculate_swi(&phase, &magnitude, &mask, &grid, &SwiParams::default());

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
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        let swi = calculate_swi(&phase, &magnitude, &mask, &grid, &SwiParams::default());

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
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        let result = highpass_filter(&data, &mask, &grid, [2.0, 2.0, 0.0]);

        for &v in &result {
            assert!(v.abs() < 1.0, "High-pass of constant should be near zero, got {}", v);
        }
    }

    const IDENTITY: [f64; 16] = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];

    #[test]
    fn test_mip_basic() {
        // 3x3x5 volume, mIP with window=3 → 3x3x3 output
        let (nx, ny, nz) = (3, 3, 5);
        let grid = Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);
        let mut data = vec![10.0; nx * ny * nz];
        // Place a low value at slice 2
        let idx = 1 + 1 * nx + 2 * nx * ny; // (1,1,2)
        data[idx] = 1.0;

        let mip = create_mip(&data, &grid, &IDENTITY, 3).unwrap();
        assert_eq!(mip.data.len(), nx * ny * 3);

        // The minimum at (1,1) should appear in slices that include z=2
        // Window starting at z=0: slices 0,1,2 → includes the 1.0
        let mip_idx_0 = 1 + 1 * nx + 0 * nx * ny;
        assert_eq!(mip.data[mip_idx_0], 1.0);
        // Window starting at z=1: slices 1,2,3 → includes the 1.0
        let mip_idx_1 = 1 + 1 * nx + 1 * nx * ny;
        assert_eq!(mip.data[mip_idx_1], 1.0);
        // Window starting at z=2: slices 2,3,4 → includes the 1.0
        let mip_idx_2 = 1 + 1 * nx + 2 * nx * ny;
        assert_eq!(mip.data[mip_idx_2], 1.0);
    }

    /// The projection must describe itself: a caller that trusts the source grid writes a file
    /// whose header promises more voxels than it holds (QSMxT#211).
    #[test]
    fn test_mip_reports_its_own_dimensions() {
        let grid = Grid::new(32, 32, 32, 1.0, 1.0, 1.0);
        let mip = create_mip(&vec![1.0; grid.n_total()], &grid, &IDENTITY, 7).unwrap();
        assert_eq!(mip.grid.dims, (32, 32, 26));
        assert_eq!(mip.data.len(), mip.grid.n_total());
        // Voxel sizes are unchanged by a projection.
        assert_eq!(mip.grid.voxel_size, grid.voxel_size);
    }

    #[test]
    fn test_mip_origin_moves_to_the_slab_centre() {
        let grid = Grid::new(4, 4, 16, 1.0, 1.0, 2.0);
        let mut affine = IDENTITY;
        affine[10] = 2.0; // 2 mm slices
        affine[11] = 10.0;
        let (mip_grid, mip_affine) = mip_geometry(&grid, &affine, 7).unwrap();
        assert_eq!(mip_grid.dims, (4, 4, 10));
        // Three slices of 2 mm past the first slab's first slice.
        assert_eq!(mip_affine[11], 16.0);
        // Nothing but the origin moves.
        assert_eq!(mip_affine[..11], affine[..11]);
    }

    #[test]
    fn test_mip_origin_follows_an_oblique_slice_direction() {
        // Slice direction tilted 45° in the y/z plane, 2 mm slices.
        let s = 2.0 / 2f64.sqrt();
        let affine = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, s, 0.0,
            0.0, 0.0, s, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let grid = Grid::new(8, 8, 20, 1.0, 1.0, 2.0);
        let (mip_grid, mip_affine) = mip_geometry(&grid, &affine, 7).unwrap();
        assert_eq!(mip_grid.dims, (8, 8, 14));
        // Both y and z move: three slice steps along the tilted direction, not along world z.
        assert!((mip_affine[7] - 3.0 * s).abs() < 1e-12, "{:?}", mip_affine);
        assert!((mip_affine[11] - 3.0 * s).abs() < 1e-12, "{:?}", mip_affine);
        assert_eq!(mip_affine[3], 0.0);
    }

    #[test]
    fn test_mip_even_window_lands_on_a_half_slice() {
        let grid = Grid::new(4, 4, 16, 1.0, 1.0, 1.0);
        let (_, mip_affine) = mip_geometry(&grid, &IDENTITY, 4).unwrap();
        assert_eq!(mip_affine[11], 1.5);
    }

    #[test]
    fn test_mip_window_of_one_is_the_volume_itself() {
        let grid = Grid::new(3, 3, 5, 1.0, 1.0, 1.0);
        let data: Vec<f64> = (0..grid.n_total()).map(|i| i as f64).collect();
        let mip = create_mip(&data, &grid, &IDENTITY, 1).unwrap();
        assert_eq!(mip.grid.dims, grid.dims);
        assert_eq!(mip.data, data);
        assert_eq!(mip.affine, IDENTITY);
    }

    #[test]
    fn test_mip_window_of_full_depth_leaves_one_slice() {
        let grid = Grid::new(3, 3, 5, 1.0, 1.0, 1.0);
        let mip = create_mip(&vec![1.0; grid.n_total()], &grid, &IDENTITY, 5).unwrap();
        assert_eq!(mip.grid.dims, (3, 3, 1));
        assert_eq!(mip.affine[11], 2.0);
    }

    #[test]
    fn test_mip_window_too_large() {
        let grid = Grid::new(3, 3, 3, 1.0, 1.0, 1.0);
        let err = create_mip(&[1.0; 27], &grid, &IDENTITY, 10).unwrap_err();
        assert!(err.contains("deeper than the 3-slice volume"), "{}", err);
    }

    #[test]
    fn test_mip_zero_window() {
        let grid = Grid::new(3, 3, 3, 1.0, 1.0, 1.0);
        let err = create_mip(&[1.0; 27], &grid, &IDENTITY, 0).unwrap_err();
        assert!(err.contains("at least 1 slice"), "{}", err);
    }

    #[test]
    fn test_mip_rejects_data_that_does_not_match_the_grid() {
        let grid = Grid::new(3, 3, 3, 1.0, 1.0, 1.0);
        let err = create_mip(&[1.0; 26], &grid, &IDENTITY, 3).unwrap_err();
        assert!(err.contains("26 values"), "{}", err);
        assert!(err.contains("27 voxels"), "{}", err);
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

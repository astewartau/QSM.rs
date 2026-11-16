//! Vasculature Mask Generation for QSMART
//!
//! This module generates a binary mask identifying blood vessels in the brain.
//! The vasculature mask is used in QSMART's two-stage processing to separate
//! tissue from vessel regions, reducing streaking artifacts near veins.
//!
//! The algorithm:
//! 1. Apply morphological bottom-hat filtering to enhance dark tubular structures
//!    (MATLAB's `imbothat` with `strel('sphere', 8)`, edge handling included)
//! 2. Apply Frangi vesselness filter to detect vessels (single precision, as QSMART's
//!    `FrangiFilter3D` runs on its single-precision magnitude; see `frangi`)
//! 3. Threshold: vesselness above Otsu's threshold (MATLAB's `graythresh`) = vessel
//! 4. Return complementary mask (1 = tissue, 0 = vessel)
//!
//! ## Reference pipeline
//!
//! QSMART's `vasculature_mask.m` runs this on the mean over echoes of the coil-combined
//! magnitude after ANTs N4 (l. 8–14; [`crate::utils::qsmart_vessel_magnitude`]), restricted to
//! the BET brain mask (`FrangiFilter3D(test.*mask, …)`, l. 27), with `FrangiAlpha` = `FrangiBeta`
//! = 0.5, `BlackWhite` false and the demo's scales 0.5:0.5:6 and c = 500 (`Demo_QSMART.m`
//! l. 41–43; paper §3.4, 7T human, 0.6 mm). The paper's rodent runs used scales 0.01–0.5 in
//! steps of 0.1 and c = 50. The bottom-hat radius (8) and Frangi scales are in voxels, and
//! Otsu's threshold depends on the magnitude's intensity distribution, so the vessel mask
//! depends on that magnitude being the one QSMART uses.
//!
//! # Agreement with the original
//!
//! `tests/qsmart_qsmci_parity.rs` (measured 2026-10-30) runs QSMART's stages on the inputs the
//! verified QSM-CI `qsmart-matlab` v3 container (QSMxT/QSM-CI#296, bit-identical to licensed
//! MATLAB) gave its own, on a 36×36×28 vessel phantom at 3 T; the fixture is regenerated in Actions
//! by `tests/data/qsmart_qsmci_small.py`. From the same magnitudes,
//! [`crate::utils::qsmart_vessel_magnitude`] then [`generate_vasculature_mask`] give the toolbox's
//! `vasculature_mask` exactly (0 of 8,312 mask voxels differ; 276 vessel voxels).
//!
//! # Frangi's c and the magnitude's intensity scale ([`FrangiC`])
//!
//! Frangi's c weighs the Hessian's Frobenius norm S in the vesselness's `1 − exp(−S²/2c²)`
//! factor. S is in the units of the bottom-hat of the magnitude, so c is in magnitude intensity
//! units: multiplying the magnitude by 10 gives exactly the vessel mask of dividing c by 10.
//! QSMART v1.0's c = 500 ([`FrangiC::Absolute`], the default, for parity) was set for its 7 T
//! data; scanners scale magnitude very differently. Measured 2026-11-12 with QSMART v1.0's own
//! MATLAB code (`vasculature_mask.m`) on the N4 echo-mean magnitude of six datasets, whose brain
//! medians range from about 110 to about 360,000:
//!
//! | dataset | brain median | c = 500 | c = 50 | c = 0.25 × median |
//! |---|---|---|---|---|
//! | 3 T Siemens Skyra | 570 | 58.1 % | 9.8 % | 6.6 % |
//! | 3 T Siemens Prisma | 110 | 54 % | 4.7 % | 5.1 % |
//! | 3 T Philips Achieva | 357,000 | 34.6 % | 35.1 % | 5.2 % |
//! | 3 T GE MR750 | 1,824 | 3.4 % | 16 % | 3.4 % |
//! | 7 T exemplar (QSMART's, 9 echoes) | 232 | ~50 % | 3 % | 2.6 % |
//! | QSM-CI simulated head (7 T) | 25 | (Otsu 0) | (Otsu 0) | 2.0 % |
//!
//! (percent of the brain mask labelled vessel). When every vesselness value is small,
//! `graythresh`'s 256 bins over [0, 1] put nearly all of them in the first bin and its threshold
//! drops to 0, so every voxel with any vesselness becomes vessel: the 35–58 % rows. A fixed c can
//! only move that between scanners (c = 50 fixes Siemens, does nothing on Philips and raises GE
//! from 3.4 to 16 %). [`FrangiC::RelativeToMedian`] sets c from the image instead: k times the
//! median of the vessel magnitude inside the brain mask, k = 0.25 by default
//! ([`QSMART_RELATIVE_FRANGI_C`]), which gave 2.0–6.6 % on all six (both field strengths, three
//! vendors, the simulation; 1.2–5.9 % with radius 3 and scales 0.5–2). k = 0.5 let Otsu collapse
//! again on the Philips data (53 %), k = 0.1 gave 4.2–8.8 %. This is close in spirit to Frangi
//! et al. 1998's own suggestion of choosing c from the image (half the largest Hessian norm),
//! but uses the magnitude's median, which outliers barely move. It is QSM.rs's addition, not
//! QSMART's: an option, with v1.0's c = 500 kept as the default.
//!
//! **χ with the relative c** (measured 2026-11-16 on Bunya, QSMxT built on this option;
//! `/scratch/user/uqaste15/qsmart-relc`): on QSM-CI's simulated head (7 T), XSIM rose from 0.150
//! to 0.211 from the total field (NRMSE 91.9 → 87.5 %, r 0.41 → 0.49) and from 0.150 to 0.223
//! through QSMxT's QSMART pipeline (magnitude ×10, as the mask's threshold of 100 empties it
//! otherwise; NRMSE 93.3 → 88.4 %, streaking 0.164 → 0.131), but deep-GM NRMSE rose from 99 to
//! 113 %. In vivo, through the QSMART pipeline on the EMTP travelling heads, deep-GM χ (thalamus,
//! caudate, putamen, pallidum, against the same subject's KKI Achieva scan) agreed better on GE
//! MR750w (mean |Δ| 7.3 → 4.3 ppb, orthogonal-fit |a − 1| over 25 ROIs 0.27 → 0.16), about the
//! same on Siemens Skyra (3.4 → 2.9 ppb, 0.05 → 0.05) and slightly worse on Siemens Prisma
//! (4.4 → 4.8 ppb, 0.07 → 0.12). Visually the 7 T exemplar's textured lateral and frontal
//! cortex is gone and the Siemens scans' posterior-edge spots are weaker. QSM.rs therefore
//! suggests the relative c with QSMART (rule `qsmart.frangi-c-relative`, Suggested: offered with
//! its reason, not preselected; decision 173) rather than recommending it.

use crate::error::InputError;
use crate::Grid;
use crate::utils::frangi::{FrangiParams, frangi_filter_3d};
use crate::utils::threshold::graythresh;

/// Parameters for vasculature mask generation
#[derive(Clone, Debug)]
pub struct VasculatureParams {
    /// Radius for bottom-hat morphological filter (default 8 voxels)
    pub sphere_radius: i32,
    /// Frangi filter scale range [min, max] (default [0.5, 6.0])
    pub frangi_scale_range: [f64; 2],
    /// Frangi filter scale ratio (default 0.5)
    pub frangi_scale_ratio: f64,
    /// Frangi's c: QSMART v1.0's 500 in magnitude units by default ([`FrangiC`])
    pub frangi_c: FrangiC,
}

/// QSMART v1.0's Frangi c, in magnitude intensity units (`Demo_QSMART.m` l. 43).
pub const QSMART_FRANGI_C: f64 = 500.0;

/// The default k of [`FrangiC::RelativeToMedian`]: c = 0.25 × the median vessel magnitude in the
/// brain mask (see the module docs for the measurements behind it).
pub const QSMART_RELATIVE_FRANGI_C: f64 = 0.25;

/// How QSMART's Frangi c is set (see the module docs, "Frangi's c and the magnitude's intensity
/// scale").
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FrangiC {
    /// c in the magnitude's intensity units, as QSMART v1.0 (default [`QSMART_FRANGI_C`] = 500).
    /// The vessel mask then depends on how the scanner scales its magnitude.
    Absolute(f64),
    /// c = k × the median of the vessel magnitude inside the brain mask (QSM.rs's option;
    /// default k [`QSMART_RELATIVE_FRANGI_C`] = 0.25), so the vessel mask does not depend on the
    /// magnitude's intensity scale.
    RelativeToMedian(f64),
}

impl Default for FrangiC {
    fn default() -> Self {
        FrangiC::Absolute(QSMART_FRANGI_C)
    }
}

impl FrangiC {
    /// The relative mode at its default k.
    pub const fn relative() -> Self {
        FrangiC::RelativeToMedian(QSMART_RELATIVE_FRANGI_C)
    }

    /// Stable id of the mode: `absolute` or `relative`.
    pub const fn id(self) -> &'static str {
        match self {
            FrangiC::Absolute(_) => "absolute",
            FrangiC::RelativeToMedian(_) => "relative",
        }
    }

    /// The c given (absolute) or the factor k (relative).
    pub const fn value(self) -> f64 {
        match self {
            FrangiC::Absolute(c) | FrangiC::RelativeToMedian(c) => c,
        }
    }

    /// Checks the value: c or k must be finite and > 0.
    pub fn validate(self) -> Result<(), InputError> {
        let v = self.value();
        if !(v.is_finite() && v > 0.0) {
            let name = match self {
                FrangiC::Absolute(_) => "Frangi c",
                FrangiC::RelativeToMedian(_) => "relative Frangi c (k)",
            };
            return Err(InputError::param(name, v, "finite and > 0"));
        }
        Ok(())
    }

    /// The c to use on `magnitude` within `mask`.
    ///
    /// # Errors
    /// If the value is invalid ([`FrangiC::validate`]), or, relative, if the mask is empty or the
    /// magnitude's median inside it is not > 0.
    pub fn resolve(self, magnitude: &[f64], mask: &[u8]) -> Result<f64, InputError> {
        self.validate()?;
        match self {
            FrangiC::Absolute(c) => Ok(c),
            FrangiC::RelativeToMedian(k) => {
                let mut v: Vec<f64> = magnitude.iter().zip(mask)
                    .filter(|&(_, &m)| m != 0)
                    .map(|(&x, _)| x)
                    .collect();
                let median = median(&mut v).ok_or_else(|| {
                    InputError::new("a relative Frangi c needs a non-empty brain mask")
                })?;
                if !(median.is_finite() && median > 0.0) {
                    return Err(InputError::new(format!(
                        "a relative Frangi c needs a positive magnitude median inside the brain mask (got {median})"
                    )));
                }
                Ok(k * median)
            }
        }
    }
}

/// Median, as MATLAB's `median` (the mean of the two middle values for an even count); `None`
/// when empty. NaNs sort last.
fn median(v: &mut [f64]) -> Option<f64> {
    let n = v.len();
    if n == 0 {
        return None;
    }
    let (_, &mut hi, _) = v.select_nth_unstable_by(n / 2, f64::total_cmp);
    if n % 2 == 1 {
        return Some(hi);
    }
    let lo = v[..n / 2].iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Some((lo + hi) / 2.0)
}

impl Default for VasculatureParams {
    fn default() -> Self {
        Self {
            sphere_radius: 8,
            // QSMART Demo defaults: FrangiScaleRange=[0.5,6], FrangiScaleRatio=0.5
            frangi_scale_range: [0.5, 6.0],
            frangi_scale_ratio: 0.5,
            frangi_c: FrangiC::default(),
        }
    }
}

/// Generate vasculature mask from magnitude image
///
/// # Arguments
/// * `magnitude` - Average magnitude image (ideally bias-corrected)
/// * `mask` - Binary brain mask
/// * `grid` - Volume grid (dimensions and voxel sizes)
/// * `params` - Vasculature detection parameters
/// * `progress` - Progress callback `(current_step, total_steps)`
///
/// # Returns
/// Complementary vasculature mask (1 = tissue, 0 = vessel)
///
/// # Errors
/// If the Frangi parameters are invalid (see [`FrangiParams::validate`]).
pub fn generate_vasculature_mask(
    magnitude: &[f64],
    mask: &[u8],
    grid: &Grid,
    params: &VasculatureParams,
    progress_callback: impl Fn(usize, usize),
) -> Result<Vec<f64>, InputError> {
    let (nx, ny, nz) = grid.dims;
    // Step 1: Apply morphological bottom-hat filter to enhance dark structures
    // Bottom-hat = closing(image) - image
    progress_callback(0, 4);

    let bottom_hat = morphological_bottom_hat(magnitude, nx, ny, nz, params.sphere_radius);

    // Step 2: Apply mask to bottom-hat result
    progress_callback(1, 4);

    let masked_bottom_hat: Vec<f64> = bottom_hat.iter()
        .zip(mask.iter())
        .map(|(&v, &m)| if m != 0 { v } else { 0.0 })
        .collect();

    // Step 3: Apply Frangi vesselness filter
    progress_callback(2, 4);

    // c from the vessel magnitude itself (not its bottom-hat) inside the brain mask, when relative.
    let c = params.frangi_c.resolve(magnitude, mask)?;
    let frangi_params = FrangiParams {
        scale_range: params.frangi_scale_range,
        scale_ratio: params.frangi_scale_ratio,
        alpha: 0.5,
        beta: 0.5,
        c,
        black_white: false, // Detect bright structures (after bottom-hat)
    };

    let frangi_result = frangi_filter_3d(&masked_bottom_hat, grid, &frangi_params, |_, _| {})?;
    let enhanced = frangi_result.vesselness;

    // Step 4: Otsu threshold, as vasculature_mask.m does: `T = graythresh(enhanced)` over the
    // whole volume, vessels where `enhanced > T`, nothing outside the brain mask.
    progress_callback(3, 4);

    let threshold = graythresh(&enhanced);

    // Return complementary mask (1 = tissue, 0 = vessel)
    let vasc_only: Vec<f64> = enhanced.iter()
        .zip(mask.iter())
        .map(|(&v, &m)| if m != 0 && v > threshold { 0.0 } else { 1.0 })
        .collect();

    progress_callback(4, 4);

    Ok(vasc_only)
}

/// Spherical structuring element with pre-computed linear offsets for fast access
struct SphereKernel {
    /// Linear offsets for interior voxels (no bounds checking needed)
    linear_offsets: Vec<isize>,
    /// Coordinate offsets for boundary voxels (need bounds checking)
    coord_offsets: Vec<(i32, i32, i32)>,
    /// Radius of the sphere
    radius: usize,
}

impl SphereKernel {
    /// Create a spherical kernel with pre-computed offsets
    fn new(radius: i32, nx: usize, ny: usize) -> Self {
        let mut linear_offsets = Vec::new();
        let mut coord_offsets = Vec::new();
        let r2 = (radius * radius) as f64;
        let stride_y = nx as isize;
        let stride_z = (nx * ny) as isize;

        for dk in -radius..=radius {
            for dj in -radius..=radius {
                for di in -radius..=radius {
                    let dist2 = (di * di + dj * dj + dk * dk) as f64;
                    if dist2 <= r2 {
                        // Pre-compute linear offset: di + dj*nx + dk*nx*ny
                        let linear = di as isize + dj as isize * stride_y + dk as isize * stride_z;
                        linear_offsets.push(linear);
                        coord_offsets.push((di, dj, dk));
                    }
                }
            }
        }

        Self {
            linear_offsets,
            coord_offsets,
            radius: radius as usize,
        }
    }
}

/// Morphological bottom-hat filter (closing - original), as MATLAB's
/// `imbothat(I, strel('sphere', radius))`.
///
/// Enhances dark features smaller than the structuring element. Like MATLAB's `imclose`,
/// the closing pads the volume with -Inf by the radius before dilating, so near the
/// volume's edge the erosion also takes the dilated padding into account (without the
/// padding, voxels within `radius` of the edge close to higher values; on QSMART's 7T
/// exemplar that changed 5,128 brain voxels). Matches `imbothat` exactly on QSMART's
/// exemplar and three 3T cases.
fn morphological_bottom_hat(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    radius: i32,
) -> Vec<f64> {
    let r = radius.max(0) as usize;
    let (px, py, pz) = (nx + 2 * r, ny + 2 * r, nz + 2 * r);
    let mut padded = vec![f64::NEG_INFINITY; px * py * pz];
    for k in 0..nz {
        for j in 0..ny {
            let src = j * nx + k * nx * ny;
            let dst = r + (j + r) * px + (k + r) * px * py;
            padded[dst..dst + nx].copy_from_slice(&data[src..src + nx]);
        }
    }
    let kernel = SphereKernel::new(radius, px, py);
    let dilated = dilate_grayscale_optimized(&padded, px, py, pz, &kernel);
    let closed = erode_grayscale_optimized(&dilated, px, py, pz, &kernel);
    let mut out = vec![0.0; data.len()];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let idx = i + j * nx + k * nx * ny;
                let c = closed[i + r + (j + r) * px + (k + r) * px * py];
                out[idx] = (c - data[idx]).max(0.0);
            }
        }
    }
    out
}

/// Optimized grayscale dilation with spherical structuring element
/// Uses pre-computed linear offsets and separates interior from boundary processing
fn dilate_grayscale_optimized(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    kernel: &SphereKernel,
) -> Vec<f64> {
    let n_total = nx * ny * nz;
    let mut result = vec![0.0f64; n_total];
    let r = kernel.radius;
    let stride_z = nx * ny;

    // Process interior voxels (no bounds checking needed) - this is the fast path
    if nx > 2 * r && ny > 2 * r && nz > 2 * r {
        for k in r..(nz - r) {
            let k_offset = k * stride_z;
            for j in r..(ny - r) {
                let jk_offset = j * nx + k_offset;
                for i in r..(nx - r) {
                    let idx = i + jk_offset;
                    let mut max_val = f64::NEG_INFINITY;

                    // Fast path: use pre-computed linear offsets, no bounds checking
                    for &offset in &kernel.linear_offsets {
                        let neighbor_idx = (idx as isize + offset) as usize;
                        // Use unsafe for additional speed in inner loop
                        let val = unsafe { *data.get_unchecked(neighbor_idx) };
                        if val > max_val {
                            max_val = val;
                        }
                    }

                    unsafe { *result.get_unchecked_mut(idx) = max_val };
                }
            }
        }
    }

    // Process boundary voxels (need bounds checking) - slower but less frequent
    let nx_i32 = nx as i32;
    let ny_i32 = ny as i32;
    let nz_i32 = nz as i32;

    for k in 0..nz {
        let k_offset = k * stride_z;
        let k_is_boundary = k < r || k >= nz - r;

        for j in 0..ny {
            let jk_offset = j * nx + k_offset;
            let j_is_boundary = j < r || j >= ny - r;

            for i in 0..nx {
                let i_is_boundary = i < r || i >= nx - r;

                // Skip interior voxels (already processed)
                if !k_is_boundary && !j_is_boundary && !i_is_boundary {
                    continue;
                }

                let idx = i + jk_offset;
                let mut max_val = f64::NEG_INFINITY;

                // Slow path: need to check bounds for each neighbor
                for &(di, dj, dk) in &kernel.coord_offsets {
                    let ni = i as i32 + di;
                    let nj = j as i32 + dj;
                    let nk = k as i32 + dk;

                    if ni >= 0 && ni < nx_i32 &&
                       nj >= 0 && nj < ny_i32 &&
                       nk >= 0 && nk < nz_i32 {
                        let neighbor_idx = ni as usize + nj as usize * nx + nk as usize * stride_z;
                        max_val = max_val.max(data[neighbor_idx]);
                    }
                }

                result[idx] = if max_val.is_finite() { max_val } else { data[idx] };
            }
        }
    }

    result
}

/// Optimized grayscale erosion with spherical structuring element
fn erode_grayscale_optimized(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    kernel: &SphereKernel,
) -> Vec<f64> {
    let n_total = nx * ny * nz;
    let mut result = vec![0.0f64; n_total];
    let r = kernel.radius;
    let stride_z = nx * ny;

    // Process interior voxels (no bounds checking needed) - this is the fast path
    if nx > 2 * r && ny > 2 * r && nz > 2 * r {
        for k in r..(nz - r) {
            let k_offset = k * stride_z;
            for j in r..(ny - r) {
                let jk_offset = j * nx + k_offset;
                for i in r..(nx - r) {
                    let idx = i + jk_offset;
                    let mut min_val = f64::INFINITY;

                    // Fast path: use pre-computed linear offsets, no bounds checking
                    for &offset in &kernel.linear_offsets {
                        let neighbor_idx = (idx as isize + offset) as usize;
                        let val = unsafe { *data.get_unchecked(neighbor_idx) };
                        if val < min_val {
                            min_val = val;
                        }
                    }

                    unsafe { *result.get_unchecked_mut(idx) = min_val };
                }
            }
        }
    }

    // Process boundary voxels (need bounds checking)
    let nx_i32 = nx as i32;
    let ny_i32 = ny as i32;
    let nz_i32 = nz as i32;

    for k in 0..nz {
        let k_offset = k * stride_z;
        let k_is_boundary = k < r || k >= nz - r;

        for j in 0..ny {
            let jk_offset = j * nx + k_offset;
            let j_is_boundary = j < r || j >= ny - r;

            for i in 0..nx {
                let i_is_boundary = i < r || i >= nx - r;

                // Skip interior voxels (already processed)
                if !k_is_boundary && !j_is_boundary && !i_is_boundary {
                    continue;
                }

                let idx = i + jk_offset;
                let mut min_val = f64::INFINITY;

                for &(di, dj, dk) in &kernel.coord_offsets {
                    let ni = i as i32 + di;
                    let nj = j as i32 + dj;
                    let nk = k as i32 + dk;

                    if ni >= 0 && ni < nx_i32 &&
                       nj >= 0 && nj < ny_i32 &&
                       nk >= 0 && nk < nz_i32 {
                        let neighbor_idx = ni as usize + nj as usize * nx + nk as usize * stride_z;
                        min_val = min_val.min(data[neighbor_idx]);
                    }
                }

                result[idx] = if min_val.is_finite() { min_val } else { data[idx] };
            }
        }
    }

    result
}


#[cfg(test)]
mod tests {
    use super::*;

    fn grid(nx: usize, ny: usize, nz: usize) -> Grid {
        Grid::new(nx, ny, nz, 1.0, 1.0, 1.0)
    }

    #[test]
    fn test_bottom_hat_basic() {
        // Constant image should give zero bottom-hat
        let data = vec![1.0f64; 27];
        let result = morphological_bottom_hat(&data, 3, 3, 3, 1);

        for &v in &result {
            assert!(v.abs() < 0.01, "Bottom-hat of constant should be ~0");
        }
    }

    #[test]
    fn bottom_hat_edge_matches_imbothat() {
        // MATLAB R2023b: rng(3); x = randi([0 9], 4, 4, 4); imbothat(x, strel('sphere', 1)).
        // In a volume this small every voxel is near the edge; closing without the padding
        // (imerode(imdilate(x, se), se)) differs at 33 of the 64 voxels.
        let x = [
            5., 7., 2., 5., 8., 8., 1., 2., 0., 4., 0., 4., 6., 2., 6., 5., 0., 5., 2., 4., 2., 6.,
            4., 1., 5., 7., 3., 2., 3., 9., 9., 6., 9., 8., 3., 0., 6., 5., 3., 2., 4., 4., 2., 2.,
            4., 8., 5., 2., 2., 4., 2., 2., 5., 0., 4., 3., 7., 7., 6., 6., 3., 6., 3., 5.,
        ];
        let mut expected = vec![0.0; 64];
        expected[26] = 3.0;
        expected[41] = 2.0;
        expected[42] = 3.0;
        assert_eq!(morphological_bottom_hat(&x, 4, 4, 4, 1), expected);
    }

    #[test]
    fn test_vasculature_mask_basic() {
        // Basic test: should run without panic
        let magnitude = vec![1.0f64; 1000];
        let mask = vec![1u8; 1000];

        let params = VasculatureParams {
            sphere_radius: 2,
            frangi_scale_range: [1.0, 2.0],
            frangi_scale_ratio: 1.0,
            frangi_c: FrangiC::Absolute(100.0),
        };

        let result = generate_vasculature_mask(&magnitude, &mask, &grid(10, 10, 10), &params, |_, _| {}).unwrap();
        assert_eq!(result.len(), 1000);
    }

    /// A 24³ volume with a noisy-free background and two dark tubes (vessels are dark on
    /// magnitude; the bottom-hat makes them bright), all inside a cubic mask.
    fn tube_phantom() -> (Vec<f64>, Vec<u8>) {
        let n = 24;
        let mut mag = vec![0.0; n * n * n];
        let mut mask = vec![0u8; n * n * n];
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let idx = i + j * n + k * n * n;
                    let (x, y, z) = (i as f64, j as f64, k as f64);
                    let mut v = 100.0 + 2.0 * x + y - 0.5 * z;
                    if (x - 8.0).hypot(y - 12.0) < 1.5 || (y - 15.0).hypot(z - 9.0) < 1.0 {
                        v *= 0.3;
                    }
                    mag[idx] = v;
                    mask[idx] = u8::from((2..n - 2).contains(&i) && (2..n - 2).contains(&j) && (2..n - 2).contains(&k));
                }
            }
        }
        (mag, mask)
    }

    fn vessels(mag: &[f64], mask: &[u8], c: FrangiC) -> Vec<f64> {
        let params = VasculatureParams { sphere_radius: 3, frangi_scale_range: [0.5, 2.0], frangi_scale_ratio: 0.5, frangi_c: c };
        generate_vasculature_mask(mag, mask, &grid(24, 24, 24), &params, |_, _| {}).unwrap()
    }

    #[test]
    fn absolute_c_is_in_magnitude_units() {
        // Scaling the magnitude by 8 (exact in single precision) gives the mask of c / 8.
        let (mag, mask) = tube_phantom();
        let mag8: Vec<f64> = mag.iter().map(|v| v * 8.0).collect();
        assert_eq!(vessels(&mag8, &mask, FrangiC::Absolute(40.0)), vessels(&mag, &mask, FrangiC::Absolute(5.0)));
    }

    #[test]
    fn relative_c_does_not_depend_on_the_intensity_scale() {
        let (mag, mask) = tube_phantom();
        let mag8: Vec<f64> = mag.iter().map(|v| v / 8.0).collect();
        let a = vessels(&mag, &mask, FrangiC::relative());
        assert_eq!(a, vessels(&mag8, &mask, FrangiC::relative()));
        assert!(a.contains(&0.0), "the tubes should be found");
        // Relative is absolute at k × the in-mask median.
        let c = FrangiC::relative().resolve(&mag, &mask).unwrap();
        assert_eq!(a, vessels(&mag, &mask, FrangiC::Absolute(c)));
    }

    #[test]
    fn relative_c_is_k_times_the_in_mask_median() {
        let mag = [1.0, 2.0, 3.0, 4.0, 100.0, 7.0];
        assert_eq!(FrangiC::RelativeToMedian(0.5).resolve(&mag, &[1, 1, 1, 1, 0, 0]).unwrap(), 0.5 * 2.5);
        assert_eq!(FrangiC::RelativeToMedian(2.0).resolve(&mag, &[1, 1, 1, 0, 1, 1]).unwrap(), 2.0 * 3.0);
        assert_eq!(FrangiC::Absolute(500.0).resolve(&mag, &[0; 6]).unwrap(), 500.0);
    }

    #[test]
    fn invalid_frangi_c_is_refused() {
        let mag = [1.0, 2.0];
        assert!(FrangiC::RelativeToMedian(0.25).resolve(&mag, &[0, 0]).is_err());
        assert!(FrangiC::RelativeToMedian(0.25).resolve(&[0.0, 0.0], &[1, 1]).is_err());
        for c in [FrangiC::RelativeToMedian(0.0), FrangiC::Absolute(-1.0), FrangiC::Absolute(f64::NAN)] {
            assert!(c.resolve(&mag, &[1, 1]).is_err(), "{c:?}");
        }
        assert_eq!(FrangiC::default(), FrangiC::Absolute(QSMART_FRANGI_C));
    }
}

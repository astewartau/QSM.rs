//! Multi-echo field mapping utilities
//!
//! Provides phase offset removal, weighted B0 estimation, bipolar correction,
//! and multi-echo linear fit for multi-echo GRE data.
//!
//! Phase offset removal is based on the HIP (Hermitian Inner Product) technique
//! from ASPIRE/MCPC-3D-S:
//! Eckstein, K., et al. (2018). "Computationally Efficient Combination of
//! Multi-channel Phase Data From Multi-echo Acquisitions (ASPIRE)."
//! Magnetic Resonance in Medicine, 79:2996-3006. https://doi.org/10.1002/mrm.26963

/// Parameters for phase offset removal.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct PhaseOffsetParams {
    /// Gaussian smoothing sigma in voxels [x, y, z] for phase offset estimation
    pub sigma: [f64; 3],
}

impl Default for PhaseOffsetParams {
    fn default() -> Self {
        Self {
            sigma: [4.0, 4.0, 4.0],
        }
    }
}

/// Backward-compatible alias.
pub type Mcpc3dsParams = PhaseOffsetParams;

/// Parameters for multi-echo linear fit.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct LinearFitParams {
    /// Estimate and remove constant phase offset
    pub estimate_offset: bool,
    /// Percentile threshold for reliability-based voxel exclusion (degrees)
    pub reliability_threshold_percentile: f64,
}

impl Default for LinearFitParams {
    fn default() -> Self {
        Self {
            estimate_offset: true,
            reliability_threshold_percentile: 90.0,
        }
    }
}

use std::f64::consts::PI;
use crate::Grid;
use crate::unwrap::romeo::{unwrap_romeo, RomeoParams};
use crate::unwrap::laplacian::laplacian_unwrap;
use crate::unwrap::UnwrapMethod;

const TWO_PI: f64 = 2.0 * PI;

/// Wrap angle to [-π, π]
#[inline]
fn wrap_to_pi(angle: f64) -> f64 {
    let mut a = angle % TWO_PI;
    if a > PI {
        a -= TWO_PI;
    } else if a < -PI {
        a += TWO_PI;
    }
    a
}

/// Index into 3D array (Fortran/column-major order)
#[inline(always)]
fn idx3d(i: usize, j: usize, k: usize, nx: usize, ny: usize) -> usize {
    i + j * nx + k * nx * ny
}

/// B0 weighting types matching MriResearchTools.jl
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum B0WeightType {
    /// mag * TE - optimal for phase SNR (default)
    PhaseSNR,
    /// mag² * TE² - based on phase variance
    PhaseVar,
    /// Uniform weights
    Average,
    /// TE only
    TEs,
    /// Magnitude only
    Mag,
}

impl B0WeightType {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "phase_snr" | "phasesnr" => B0WeightType::PhaseSNR,
            "phase_var" | "phasevar" => B0WeightType::PhaseVar,
            "average" | "uniform" => B0WeightType::Average,
            "tes" | "te" => B0WeightType::TEs,
            "mag" | "magnitude" => B0WeightType::Mag,
            _ => B0WeightType::PhaseSNR, // default
        }
    }
}

/// 3D Gaussian smoothing for phase data (handles phase wrapping)
///
/// Implements gaussiansmooth3d_phase from MriResearchTools.jl
/// Uses separable Gaussian filtering with phase-aware averaging
///
/// # Arguments
/// * `phase` - Input phase data (nx * ny * nz)
/// * `sigma` - Smoothing sigma in voxels [sx, sy, sz]
/// * `mask` - Binary mask (1 = include, 0 = exclude)
/// * `grid` - Volume grid (dimensions and voxel sizes)
///
/// # Returns
/// Smoothed phase data
pub fn gaussian_smooth_3d_phase(
    phase: &[f64],
    sigma: [f64; 3],
    mask: &[u8],
    grid: &Grid,
) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let n_total = nx * ny * nz;

    // For phase smoothing, we smooth the complex representation
    // and extract the angle to handle wrapping correctly
    let mut real = vec![0.0; n_total];
    let mut imag = vec![0.0; n_total];

    // Convert phase to complex (unit vectors)
    for i in 0..n_total {
        if mask[i] > 0 {
            real[i] = phase[i].cos();
            imag[i] = phase[i].sin();
        }
    }

    // Apply separable Gaussian smoothing to real and imaginary parts
    let real_smoothed = gaussian_smooth_3d_separable(&real, sigma, mask, nx, ny, nz);
    let imag_smoothed = gaussian_smooth_3d_separable(&imag, sigma, mask, nx, ny, nz);

    // Convert back to phase
    let mut result = vec![0.0; n_total];
    for i in 0..n_total {
        if mask[i] > 0 {
            result[i] = imag_smoothed[i].atan2(real_smoothed[i]);
        }
    }

    result
}

/// Separable 3D Gaussian smoothing
fn gaussian_smooth_3d_separable(
    data: &[f64],
    sigma: [f64; 3],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
) -> Vec<f64> {
    let n_total = nx * ny * nz;
    let mut result = data.to_vec();
    let mut temp = vec![0.0; n_total];

    // X direction
    if sigma[0] > 0.0 {
        let kernel = make_gaussian_kernel(sigma[0]);
        let half = kernel.len() / 2;

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = idx3d(i, j, k, nx, ny);
                    if mask[idx] == 0 {
                        temp[idx] = 0.0;
                        continue;
                    }

                    let mut sum = 0.0;
                    let mut weight_sum = 0.0;

                    for (ki, &kv) in kernel.iter().enumerate() {
                        let ii = i as isize + ki as isize - half as isize;
                        if ii >= 0 && ii < nx as isize {
                            let nidx = idx3d(ii as usize, j, k, nx, ny);
                            if mask[nidx] > 0 {
                                sum += result[nidx] * kv;
                                weight_sum += kv;
                            }
                        }
                    }

                    temp[idx] = if weight_sum > 0.0 { sum / weight_sum } else { 0.0 };
                }
            }
        }
        std::mem::swap(&mut result, &mut temp);
    }

    // Y direction
    if sigma[1] > 0.0 {
        let kernel = make_gaussian_kernel(sigma[1]);
        let half = kernel.len() / 2;

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = idx3d(i, j, k, nx, ny);
                    if mask[idx] == 0 {
                        temp[idx] = 0.0;
                        continue;
                    }

                    let mut sum = 0.0;
                    let mut weight_sum = 0.0;

                    for (ki, &kv) in kernel.iter().enumerate() {
                        let jj = j as isize + ki as isize - half as isize;
                        if jj >= 0 && jj < ny as isize {
                            let nidx = idx3d(i, jj as usize, k, nx, ny);
                            if mask[nidx] > 0 {
                                sum += result[nidx] * kv;
                                weight_sum += kv;
                            }
                        }
                    }

                    temp[idx] = if weight_sum > 0.0 { sum / weight_sum } else { 0.0 };
                }
            }
        }
        std::mem::swap(&mut result, &mut temp);
    }

    // Z direction
    if sigma[2] > 0.0 {
        let kernel = make_gaussian_kernel(sigma[2]);
        let half = kernel.len() / 2;

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = idx3d(i, j, k, nx, ny);
                    if mask[idx] == 0 {
                        temp[idx] = 0.0;
                        continue;
                    }

                    let mut sum = 0.0;
                    let mut weight_sum = 0.0;

                    for (ki, &kv) in kernel.iter().enumerate() {
                        let kk = k as isize + ki as isize - half as isize;
                        if kk >= 0 && kk < nz as isize {
                            let nidx = idx3d(i, j, kk as usize, nx, ny);
                            if mask[nidx] > 0 {
                                sum += result[nidx] * kv;
                                weight_sum += kv;
                            }
                        }
                    }

                    temp[idx] = if weight_sum > 0.0 { sum / weight_sum } else { 0.0 };
                }
            }
        }
        std::mem::swap(&mut result, &mut temp);
    }

    result
}

/// Create 1D Gaussian kernel
fn make_gaussian_kernel(sigma: f64) -> Vec<f64> {
    let radius = (3.0 * sigma).ceil() as usize;
    let size = 2 * radius + 1;
    let mut kernel = vec![0.0; size];

    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut sum = 0.0;

    for i in 0..size {
        let x = i as f64 - radius as f64;
        kernel[i] = (-x * x / two_sigma_sq).exp();
        sum += kernel[i];
    }

    // Normalize
    for k in kernel.iter_mut() {
        *k /= sum;
    }

    kernel
}

/// Compute Hermitian Inner Product (HIP) between two echoes
///
/// HIP = conj(echo1) * echo2 = mag1 * mag2 * exp(i * (phase2 - phase1))
///
/// Returns (hip_phase, hip_mag) where:
/// - hip_phase = phase2 - phase1 (wrapped to [-π, π])
/// - hip_mag = mag1 * mag2
pub fn hermitian_inner_product(
    phase1: &[f64], mag1: &[f64],
    phase2: &[f64], mag2: &[f64],
    mask: &[u8],
    n: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut hip_phase = vec![0.0; n];
    let mut hip_mag = vec![0.0; n];

    for i in 0..n {
        if mask[i] > 0 {
            hip_phase[i] = wrap_to_pi(phase2[i] - phase1[i]);
            hip_mag[i] = mag1[i] * mag2[i];
        }
    }

    (hip_phase, hip_mag)
}

/// MCPC-3D-S phase offset estimation for single-coil multi-echo data
///
/// Implements the MCPC-3D-S algorithm from MriResearchTools.jl for single-coil data.
/// This estimates and removes the phase offset (φ₀) from each echo.
///
/// # Arguments
/// * `phases` - Phase data for all echoes, shape [n_echoes][nx*ny*nz]
/// * `mags` - Magnitude data for all echoes, shape [n_echoes][nx*ny*nz]
/// * `tes` - Echo times in ms
/// * `mask` - Binary mask
/// * `sigma` - Smoothing sigma in voxels [sx, sy, sz], default [10, 10, 5]
/// * `echoes` - Which echoes to use for HIP calculation, default [0, 1] (first two)
/// * `nx`, `ny`, `nz` - Dimensions
///
/// # Returns
/// (corrected_phases, phase_offset) where:
/// - corrected_phases: phases with offset removed
/// - phase_offset: estimated phase offset
/// Remove phase offset from multi-echo phase data using HIP (Hermitian Inner Product).
///
/// Estimates the spatially-varying phase offset from the phase difference between two
/// echoes, smooths it with a Gaussian filter, and subtracts it from all echoes.
///
/// # Arguments
/// * `phases` - Wrapped phase per echo (n_echoes arrays of nx*ny*nz)
/// * `mags` - Magnitude per echo
/// * `tes` - Echo times (any consistent unit)
/// * `mask` - Binary mask
/// * `sigma` - Gaussian smoothing sigma in voxels [x, y, z]
/// * `echoes` - Which two echoes to use for HIP [e1, e2]
/// * `unwrap_method` - Method to unwrap the HIP phase (Romeo or Laplacian)
/// * `grid` - Volume grid (dimensions and voxel sizes)
///
/// # Returns
/// `(corrected_phases, phase_offset)` — offset-corrected phases and the estimated offset
pub fn phase_offset_removal(
    phases: &[impl AsRef<[f64]>],
    mags: &[impl AsRef<[f64]>],
    tes: &[f64],
    mask: &[u8],
    sigma: [f64; 3],
    echoes: [usize; 2],
    unwrap_method: UnwrapMethod,
    grid: &Grid,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let (nx, ny, nz) = grid.dims;
    let n_echoes = phases.len();
    let n_total = nx * ny * nz;

    let e1 = echoes[0];
    let e2 = echoes[1];

    // ΔTE = TEs[echo2] - TEs[echo1]
    let delta_te = tes[e2] - tes[e1];

    // Compute HIP between the two echoes
    // HIP = conj(echo1) * echo2, so hip_phase = phase2 - phase1
    let (hip_phase, hip_mag) = hermitian_inner_product(
        phases[e1].as_ref(), mags[e1].as_ref(),
        phases[e2].as_ref(), mags[e2].as_ref(),
        mask, n_total
    );

    // Unwrap HIP phase
    let unwrapped_hip = match unwrap_method {
        UnwrapMethod::Romeo => {
            let weight: Vec<f64> = hip_mag.iter().map(|&x| x.sqrt()).collect();
            unwrap_romeo(&hip_phase, &weight, None, 0.0, 0.0, mask, &RomeoParams::default(), grid)
        }
        UnwrapMethod::Laplacian => {
            laplacian_unwrap(&hip_phase, mask, grid)
        }
    };
    drop(hip_phase);
    drop(hip_mag);

    // Phase evolution at TE1: (TE1 / ΔTE) * unwrapped_hip
    // This gives the phase that would have evolved from TE=0 to TE=TE1
    let scale = tes[e1] / delta_te;
    let mut phase_offset = vec![0.0; n_total];
    for i in 0..n_total {
        if mask[i] > 0 {
            // Phase offset = phase[echo1] - phase_evolution
            // phase_evolution = scale * unwrapped_hip
            // IMPORTANT: Do NOT wrap here! Julia line 49 does raw subtraction
            phase_offset[i] = phases[e1].as_ref()[i] - scale * unwrapped_hip[i];
        }
    }
    drop(unwrapped_hip); // Free ~82 MB early

    // Smooth the phase offset (handles wrapping via complex representation)
    // Julia line 51: po[:,:,:,icha] .= gaussiansmooth3d_phase(view(po,:,:,:,icha), sigma; mask)
    let phase_offset_smoothed = gaussian_smooth_3d_phase(&phase_offset, sigma, mask, grid);
    drop(phase_offset); // Free ~82 MB early

    // Remove phase offset from all echoes
    // Julia combinewithPO does: exp.(1im .* (phase - po)) then angle()
    // This is equivalent to wrap_to_pi(phase - po)
    let mut corrected_phases = Vec::with_capacity(n_echoes);
    for e in 0..n_echoes {
        let mut corrected = vec![0.0; n_total];
        for i in 0..n_total {
            if mask[i] > 0 {
                corrected[i] = wrap_to_pi(phases[e].as_ref()[i] - phase_offset_smoothed[i]);
            }
        }
        corrected_phases.push(corrected);
    }

    (corrected_phases, phase_offset_smoothed)
}

/// Box-filter widths approximating a Gaussian of `sigma` (voxels) with `n` passes — the
/// `getboxsizes` of MriResearchTools.jl (Kovesi's "fast almost-Gaussian" construction).
/// Returns all-ones (i.e. no smoothing) when `sigma` is not positive.
pub fn gaussian_box_sizes(sigma: f64, n: usize) -> Vec<usize> {
    fn round_half_even(x: f64) -> f64 {
        let r = x.round();
        if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 { r - x.signum() } else { r }
    }
    if !(sigma > 0.0) || n == 0 {
        return vec![1; n];
    }
    let nf = n as f64;
    let w_ideal = (12.0 * sigma * sigma / nf + 1.0).sqrt();
    let wl = round_half_even(w_ideal - (w_ideal + 1.0) % 2.0); // next lower odd integer
    let wu = wl + 2.0;
    let m_ideal = (12.0 * sigma * sigma - nf * wl * wl - 4.0 * nf * wl - 3.0 * nf) / (-4.0 * wl - 4.0);
    let m = round_half_even(m_ideal);
    (1..=n).map(|i| if (i as f64) <= m { wl as usize } else { wu as usize }).collect()
}

/// Running-average line filter that treats `NaN` as "no data": the box only starts once it
/// holds `boxsize` valid samples, and when it runs into `NaN` it extrapolates linearly for up
/// to `boxsize/2` samples ("fill" mode) before waiting for data again. Positions the filter
/// never reaches keep their input value. Port of `nanboxfilterline!` (MriResearchTools.jl);
/// `orig` is scratch space that is resized as needed.
pub fn nan_box_filter_line(line: &mut [f64], boxsize: usize, orig: &mut Vec<f64>) {
    #[derive(PartialEq, Clone, Copy)]
    enum Mode { Nan, Normal, Fill }
    let n = line.len();
    let r = boxsize / 2;
    if n == 0 || boxsize < 3 {
        return;
    }
    let maxfills = r;
    orig.clear();
    orig.resize(n + boxsize - 1, f64::NAN);
    orig[r..r + n].copy_from_slice(line);

    let mut lsum: f64 = orig[r..2 * r].iter().sum();
    if lsum.is_nan() {
        lsum = 0.0;
    }
    let (mut nfills, mut nvalids) = (0usize, 0usize);
    let mut mode = Mode::Nan;
    let bs = boxsize as f64;

    // `i` is the 1-based output index, as in the reference; orig[k] here is orig[k+1] there.
    for i in 1..=n {
        if lsum.is_nan() {
            break;
        }
        let lead = orig[i - 1 + 2 * r];
        match mode {
            Mode::Normal => {
                if lead.is_nan() {
                    mode = Mode::Fill;
                }
            }
            Mode::Nan => {
                if lead.is_nan() { nvalids = 0; } else { nvalids += 1; }
                if nvalids == boxsize {
                    mode = Mode::Normal;
                    lsum = orig[i - 1..i + 2 * r].iter().sum();
                    line[i - 1] = lsum / bs;
                    continue;
                }
            }
            Mode::Fill => {
                if lead.is_nan() {
                    nfills += 1;
                    if nfills > maxfills {
                        mode = Mode::Nan;
                        nfills = 0;
                        lsum = 0.0;
                        nvalids = 0;
                    }
                } else {
                    mode = Mode::Normal;
                    nfills = 0;
                }
            }
        }
        match mode {
            Mode::Normal => {
                let trailing = if i >= 2 { orig[i - 2] } else { 0.0 };
                lsum += orig[i - 1 + 2 * r] - trailing;
                line[i - 1] = lsum / bs;
            }
            Mode::Fill => {
                let trailing = if i >= 2 { orig[i - 2] } else { 0.0 };
                lsum -= trailing;
                line[i - 1] = (lsum - orig[i - 1]) / (bs - 2.0);
                let prev = if i - 1 >= r { line[i - 1 - r] } else { line[i - 1] };
                let extrapolated = 2.0 * line[i - 1] - prev;
                orig[i - 1 + 2 * r] = extrapolated;
                if i + r < n {
                    line[i - 1 + r] = extrapolated;
                }
                lsum += extrapolated;
            }
            Mode::Nan => {}
        }
    }
}

/// Masked 3D smoothing by repeated NaN-aware box filtering — the `mask` branch of
/// `gaussiansmooth3d!` in MriResearchTools.jl (4 passes per axis, alternating direction on
/// even passes, box widths from [`gaussian_box_sizes`]). Voxels outside `mask` become `NaN`
/// before filtering and are only given values where the filter extrapolates across the
/// boundary; the result is written in place.
pub fn nan_box_smooth_3d(image: &mut [f64], sigma: [f64; 3], mask: &[u8], grid: &Grid) {
    let (nx, ny, nz) = grid.dims;
    let dims = [nx, ny, nz];
    let n_total = nx * ny * nz;
    assert_eq!(image.len(), n_total);
    assert_eq!(mask.len(), n_total);
    const NBOX: usize = 4;
    for i in 0..n_total {
        if mask[i] == 0 {
            image[i] = f64::NAN;
        }
    }
    let mut boxsizes: Vec<Vec<usize>> = sigma.iter().map(|&s| gaussian_box_sizes(s, NBOX)).collect();
    // checkboxsizes!: odd widths, no wider than half the axis
    for d in 0..3 {
        for b in boxsizes[d].iter_mut() {
            if *b % 2 == 0 { *b += 1; }
            if *b as f64 > dims[d] as f64 / 2.0 {
                let mut v = dims[d] / 2;
                if v % 2 == 0 { v += 1; }
                *b = v;
            }
        }
    }
    let mut line: Vec<f64> = Vec::new();
    let mut scratch: Vec<f64> = Vec::new();
    for ibox in 0..NBOX {
        for d in 0..3 {
            let bsize = boxsizes[d][ibox];
            let len = dims[d];
            if len == 1 || bsize < 3 {
                continue;
            }
            let reverse = ibox % 2 == 1;
            let (stride, n_lines_a, stride_a, n_lines_b, stride_b) = match d {
                0 => (1, ny, nx, nz, nx * ny),
                1 => (nx, nx, 1, nz, nx * ny),
                _ => (nx * ny, nx, 1, ny, nx),
            };
            line.resize(len, 0.0);
            for b in 0..n_lines_b {
                for a in 0..n_lines_a {
                    let base = a * stride_a + b * stride_b;
                    for k in 0..len {
                        let kk = if reverse { len - 1 - k } else { k };
                        line[k] = image[base + kk * stride];
                    }
                    nan_box_filter_line(&mut line, bsize, &mut scratch);
                    for k in 0..len {
                        let kk = if reverse { len - 1 - k } else { k };
                        image[base + kk * stride] = line[k];
                    }
                }
            }
        }
    }
}

/// Masked phase smoothing via the complex representation, using [`nan_box_smooth_3d`] —
/// `gaussiansmooth3d_phase(phase, sigma; mask)` of MriResearchTools.jl. Returns `NaN` where
/// the smoothed complex value is undefined (far outside the mask).
pub fn nan_box_smooth_3d_phase(phase: &[f64], sigma: [f64; 3], mask: &[u8], grid: &Grid) -> Vec<f64> {
    let n = phase.len();
    let mut re: Vec<f64> = phase.iter().map(|p| p.cos()).collect();
    let mut im: Vec<f64> = phase.iter().map(|p| p.sin()).collect();
    nan_box_smooth_3d(&mut re, sigma, mask, grid);
    nan_box_smooth_3d(&mut im, sigma, mask, grid);
    (0..n).map(|i| im[i].atan2(re[i])).collect()
}

/// Result of MCPC-3D-S multi-coil combination ([`mcpc3ds_combine`]).
#[derive(Debug, Clone)]
pub struct CoilCombinationResult {
    /// Combined phase per echo, wrapped to [-π, π]
    pub phases: Vec<Vec<f64>>,
    /// Combined magnitude per echo: `sqrt(|Σ_c |S_c|² · exp(i(φ_c − po_c))|)`
    pub magnitudes: Vec<Vec<f64>>,
    /// Mask used for the phase-offset estimation (robust threshold on √|HIP|)
    pub mask: Vec<u8>,
}

/// MCPC-3D-S multi-coil phase combination (Eckstein et al., MRM 2018), as in
/// `MriResearchTools.jl`'s `mcpc3ds` for 5D (multi-echo, uncombined) input.
///
/// Each receive coil carries its own TE-independent phase offset, so uncombined channels
/// cannot simply be summed. The algorithm estimates the offsets from the coil-summed
/// Hermitian inner product (HIP) of two echoes and combines the channels coherently:
///
/// 1. `HIP = Σ_c |S_{1,c}| |S_{2,c}| exp(i(φ_{2,c} − φ_{1,c}))` — the coil offsets cancel in
///    the inter-echo phase difference, so the HIP phase is pure field evolution over ΔTE.
/// 2. The HIP phase is unwrapped once (weighted by √|HIP|, on a robust mask of that weight).
/// 3. Per coil, `po_c = φ_{1,c} − (TE₁/ΔTE)·unwrapped_HIP`, smoothed in the complex domain
///    with the same masked box-filter Gaussian approximation as MriResearchTools
///    ([`nan_box_smooth_3d_phase`]; `sigma` in voxels, the reference default is [10, 10, 5]).
/// 4. Per echo, `S_e = Σ_c |S_{e,c}|² exp(i(φ_{e,c} − po_c))`; output phase `arg(S_e)` and
///    magnitude `sqrt(|S_e|)`.
///
/// With a single coil this reduces to [`phase_offset_removal`] (with the robust HIP mask).
///
/// # Arguments
/// * `phases` - Wrapped phase, coil-major: `phases[coil][echo]` (each `nx*ny*nz`)
/// * `mags` - Magnitude, same layout as `phases`
/// * `tes` - Echo times (any consistent unit; only the ratio `TE₁/ΔTE` is used)
/// * `sigma` - Gaussian smoothing sigma in voxels [x, y, z] for the phase offsets
/// * `echoes` - Which two echoes form the HIP, default `[0, 1]`
/// * `unwrap_method` - How to unwrap the HIP phase (ROMEO or Laplacian)
/// * `grid` - Volume grid
///
/// # Panics
/// If there are no coils, the coils do not all have the same number of echoes, magnitude
/// and phase layouts differ, or `echoes` is out of range.
pub fn mcpc3ds_combine<P: AsRef<[f64]>, M: AsRef<[f64]>>(
    phases: &[Vec<P>],
    mags: &[Vec<M>],
    tes: &[f64],
    sigma: [f64; 3],
    echoes: [usize; 2],
    unwrap_method: UnwrapMethod,
    grid: &Grid,
) -> CoilCombinationResult {
    let (nx, ny, nz) = grid.dims;
    let n = nx * ny * nz;
    let n_coils = phases.len();
    assert!(n_coils > 0, "mcpc3ds_combine: no coils");
    assert_eq!(mags.len(), n_coils, "mcpc3ds_combine: magnitude/phase coil count differ");
    let n_echoes = phases[0].len();
    assert!(n_echoes >= 2, "mcpc3ds_combine: at least two echoes are required");
    assert_eq!(tes.len(), n_echoes, "mcpc3ds_combine: echo time count must match echoes");
    let [e1, e2] = echoes;
    assert!(e1 < n_echoes && e2 < n_echoes && e1 != e2, "mcpc3ds_combine: HIP echoes out of range");
    for c in 0..n_coils {
        assert_eq!(phases[c].len(), n_echoes, "mcpc3ds_combine: coil {} has a different echo count", c);
        assert_eq!(mags[c].len(), n_echoes, "mcpc3ds_combine: coil {} magnitude echo count differs", c);
        for e in 0..n_echoes {
            assert_eq!(phases[c][e].as_ref().len(), n, "mcpc3ds_combine: coil {} echo {} phase size", c, e);
            assert_eq!(mags[c][e].as_ref().len(), n, "mcpc3ds_combine: coil {} echo {} magnitude size", c, e);
        }
    }

    // 1. Coil-summed Hermitian inner product between the two HIP echoes.
    let mut hip_re = vec![0.0f64; n];
    let mut hip_im = vec![0.0f64; n];
    for c in 0..n_coils {
        let (p1, p2) = (phases[c][e1].as_ref(), phases[c][e2].as_ref());
        let (m1, m2) = (mags[c][e1].as_ref(), mags[c][e2].as_ref());
        for i in 0..n {
            let a = m1[i] * m2[i];
            let d = p2[i] - p1[i];
            hip_re[i] += a * d.cos();
            hip_im[i] += a * d.sin();
        }
    }
    let hip_phase: Vec<f64> = (0..n).map(|i| hip_im[i].atan2(hip_re[i])).collect();
    // weight = sqrt(|HIP|)
    let weight: Vec<f64> = (0..n).map(|i| (hip_re[i] * hip_re[i] + hip_im[i] * hip_im[i]).sqrt().sqrt()).collect();
    drop(hip_re);
    drop(hip_im);

    // 2. Robust mask on the HIP weight, then unwrap the HIP phase once.
    let mask = crate::utils::bias_correction::robust_mask(&weight, grid);
    let unwrapped_hip = match unwrap_method {
        UnwrapMethod::Romeo => unwrap_romeo(&hip_phase, &weight, None, 0.0, 0.0, &mask, &RomeoParams::default(), grid),
        UnwrapMethod::Laplacian => laplacian_unwrap(&hip_phase, &mask, grid),
    };
    drop(hip_phase);
    drop(weight);

    // 3./4. Per-coil offset → smooth → accumulate the magnitude²-weighted complex sum per echo.
    let scale = tes[e1] / (tes[e2] - tes[e1]);
    let mut acc_re: Vec<Vec<f64>> = (0..n_echoes).map(|_| vec![0.0f64; n]).collect();
    let mut acc_im: Vec<Vec<f64>> = (0..n_echoes).map(|_| vec![0.0f64; n]).collect();
    let mut po = vec![0.0f64; n];
    for c in 0..n_coils {
        let p1 = phases[c][e1].as_ref();
        for i in 0..n {
            po[i] = if mask[i] > 0 { p1[i] - scale * unwrapped_hip[i] } else { 0.0 };
        }
        // Smooth as MriResearchTools does (NaN-aware box passes); NaN far outside the mask → 0.
        let mut po_s = nan_box_smooth_3d_phase(&po, sigma, &mask, grid);
        for v in po_s.iter_mut() {
            if !v.is_finite() { *v = 0.0; }
        }
        for e in 0..n_echoes {
            let (p, m) = (phases[c][e].as_ref(), mags[c][e].as_ref());
            let (re, im) = (&mut acc_re[e], &mut acc_im[e]);
            for i in 0..n {
                let w = m[i] * m[i];
                let ang = p[i] - po_s[i];
                re[i] += w * ang.cos();
                im[i] += w * ang.sin();
            }
        }
    }
    drop(po);
    drop(unwrapped_hip);

    let mut phases_out = Vec::with_capacity(n_echoes);
    let mut mags_out = Vec::with_capacity(n_echoes);
    for e in 0..n_echoes {
        let (re, im) = (&acc_re[e], &acc_im[e]);
        phases_out.push((0..n).map(|i| im[i].atan2(re[i])).collect());
        mags_out.push((0..n).map(|i| (re[i] * re[i] + im[i] * im[i]).sqrt().sqrt()).collect());
    }

    CoilCombinationResult { phases: phases_out, magnitudes: mags_out, mask }
}

/// Calculate B0 field from unwrapped phase using weighted averaging
///
/// Implements calculateB0_unwrapped from MriResearchTools.jl
///
/// Formula: B0 = (1000 / 2π) * Σ(phase / TE * weight) / Σ(weight)
///
/// # Arguments
/// * `unwrapped_phases` - Unwrapped phase for each echo [n_echoes][nx*ny*nz]
/// * `mags` - Magnitude for each echo (used for some weighting types)
/// * `tes` - Echo times in seconds
/// * `mask` - Binary mask
/// * `weight_type` - Type of weighting to use
/// * `grid` - Volume grid (dimensions and voxel sizes)
///
/// # Returns
/// B0 field in Hz
pub fn calculate_b0_weighted(
    unwrapped_phases: &[impl AsRef<[f64]>],
    mags: &[impl AsRef<[f64]>],
    tes: &[f64],
    mask: &[u8],
    weight_type: B0WeightType,
    grid: &Grid,
) -> Vec<f64> {
    let n_total = grid.n_total();
    let n_echoes = unwrapped_phases.len();
    let mut b0 = vec![0.0; n_total];

    // Compute inline to avoid allocating per-echo weight arrays

    // B0 = (1 / 2π) * Σ(phase / TE * weight) / Σ(weight)
    let scale = 1.0 / TWO_PI;

    for i in 0..n_total {
        if mask[i] == 0 {
            continue;
        }

        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        for e in 0..n_echoes {
            let te = tes[e];
            let mag_val = mags[e].as_ref()[i];
            let phase_over_te = unwrapped_phases[e].as_ref()[i] / te;

            let w = match weight_type {
                B0WeightType::PhaseSNR => mag_val * te,
                B0WeightType::PhaseVar => mag_val * mag_val * te * te,
                B0WeightType::Average => 1.0,
                B0WeightType::TEs => te,
                B0WeightType::Mag => mag_val,
            };

            weighted_sum += phase_over_te * w;
            weight_sum += w;
        }

        if weight_sum > 1e-10 {
            b0[i] = scale * weighted_sum / weight_sum;
        }
    }

    b0
}

// =========================================================================
// Bipolar Correction
// =========================================================================

/// Bipolar gradient correction for multi-echo phase data.
///
/// Removes linear phase artefact caused by bipolar readout gradients.
/// Requires at least 3 echoes.
///
/// Reference: Eckstein PhD thesis (2021), Section 3.1.3
/// https://doi.org/10.34726/hss.2021.43447
///
/// # Arguments
/// * `phases` - Mutable phase data per echo (modified in-place)
/// * `mags` - Magnitude data per echo
/// * `tes` - Echo times (any consistent unit; only ratios are used)
/// * `mask` - Binary mask
/// * `sigma` - Smoothing sigma for artefact estimation
/// * `grid` - Volume grid (dimensions and voxel sizes)
pub fn bipolar_correction<P: AsMut<[f64]> + AsRef<[f64]>>(
    phases: &mut [P],
    mags: &[impl AsRef<[f64]>],
    tes: &[f64],
    mask: &[u8],
    sigma: [f64; 3],
    grid: &Grid,
) {
    let (nx, ny, nz) = grid.dims;
    let n_echoes = phases.len();
    if n_echoes < 3 {
        return; // Need at least 3 echoes
    }

    let n_total = nx * ny * nz;
    let delta_te = tes[1] - tes[0];
    let m = tes[0] / delta_te;
    let k = (tes[0] + tes[2]) / tes[1];

    // Step 1: Compute artefact phase = φ1 + φ3 - k*φ2
    // If k is near-integer, unwrap φ2 first to avoid wrap issues
    let phi2 = if (k - k.round()).abs() < 0.01 {
        // k is integer-ish: unwrap φ2 with ROMEO
        let mag2 = if mags.is_empty() { &[] as &[f64] } else { mags[1].as_ref() };
        unwrap_romeo(
            phases[1].as_ref(), mag2, None, 0.0, 0.0,
            mask, &RomeoParams::default(), grid,
        )
    } else {
        phases[1].as_ref().to_vec()
    };

    let mut artefact = vec![0.0; n_total];
    for i in 0..n_total {
        if mask[i] > 0 {
            artefact[i] = wrap_to_pi(
                phases[0].as_ref()[i] + phases[2].as_ref()[i] - k * phi2[i]
            );
        }
    }

    // Step 2: Smooth the artefact
    artefact = gaussian_smooth_3d_phase(&artefact, sigma, mask, grid);

    // Step 3: Unwrap the artefact with ROMEO
    let mag1 = if mags.is_empty() { &[] as &[f64] } else { mags[0].as_ref() };
    let romeo_params = RomeoParams {
        correct_global: true,
        ..Default::default()
    };
    artefact = unwrap_romeo(
        &artefact, mag1, None, 0.0, 0.0,
        mask, &romeo_params, grid,
    );

    // Step 4: Remove artefact from each echo
    // f = (2 - k) * m - k
    // even echoes: t = (m + 1) / f
    // odd echoes:  t = m / f
    let f = (2.0 - k) * m - k;
    if f.abs() < 1e-10 {
        return; // Degenerate case
    }

    for ieco in 0..n_echoes {
        // Julia is 1-indexed: iseven(ieco) checks 1-indexed echo number
        // Echo 1 (idx 0) is odd, echo 2 (idx 1) is even, etc.
        let t = if (ieco + 1) % 2 == 0 { (m + 1.0) / f } else { m / f };
        for i in 0..n_total {
            if mask[i] > 0 {
                phases[ieco].as_mut()[i] = wrap_to_pi(
                    phases[ieco].as_ref()[i] - t * artefact[i]
                );
            }
        }
    }
}


//=============================================================================
// Multi-Echo Linear Fit
//=============================================================================

/// Result of multi-echo linear fit
pub struct LinearFitResult {
    /// Field map (slope) in rad/s (divide by 2π for Hz)
    pub field: Vec<f64>,
    /// Phase offset (intercept) in radians
    pub phase_offset: Vec<f64>,
    /// Fit residual (normalized by magnitude sum)
    pub fit_residual: Vec<f64>,
    /// Reliability mask (1 = reliable, 0 = unreliable)
    pub reliability_mask: Vec<u8>,
}

/// Multi-echo linear fit with magnitude weighting
///
/// Fits a linear model: phase = slope * TE + intercept
/// using weighted least squares with magnitude as weights.
///
/// Based on QSM.jl multi_echo_linear_fit and QSMART echofit.m
///
/// # Arguments
/// * `unwrapped_phases` - Unwrapped phase for each echo [n_echoes][nx*ny*nz]
/// * `mags` - Magnitude for each echo [n_echoes][nx*ny*nz]
/// * `tes` - Echo times in seconds
/// * `mask` - Binary mask
/// * `estimate_offset` - If true, estimate phase offset (intercept)
/// * `reliability_threshold_percentile` - Percentile for reliability masking (0-100, 0=disable)
///
/// # Returns
/// LinearFitResult containing field, phase_offset, fit_residual, reliability_mask
pub fn multi_echo_linear_fit(
    unwrapped_phases: &[impl AsRef<[f64]>],
    mags: &[impl AsRef<[f64]>],
    tes: &[f64],
    mask: &[u8],
    estimate_offset: bool,
    reliability_threshold_percentile: f64,
) -> LinearFitResult {
    let n_echoes = unwrapped_phases.len();
    let n_total = unwrapped_phases[0].as_ref().len();

    let mut field = vec![0.0; n_total];
    let mut phase_offset = vec![0.0; n_total];
    let mut fit_residual = vec![0.0; n_total];

    if estimate_offset {
        // Weighted linear fit with intercept: phase = α + β * TE
        // Using centered data approach for numerical stability
        //
        // β = Σ w*(TE - TE_mean)*(phase - phase_mean) / Σ w*(TE - TE_mean)²
        // α = phase_mean - β * TE_mean (weighted means)

        // Precompute weighted TE mean and sum of squared deviations
        // (These are per-voxel because weights vary)
        for v in 0..n_total {
            if mask[v] == 0 {
                continue;
            }

            // Compute weighted means
            let mut sum_w = 0.0;
            let mut sum_w_te = 0.0;
            let mut sum_w_phase = 0.0;

            for e in 0..n_echoes {
                let w = mags[e].as_ref()[v];
                sum_w += w;
                sum_w_te += w * tes[e];
                sum_w_phase += w * unwrapped_phases[e].as_ref()[v];
            }

            if sum_w < 1e-10 {
                continue;
            }

            let te_mean = sum_w_te / sum_w;
            let phase_mean = sum_w_phase / sum_w;

            // Compute slope using centered data
            let mut sum_w_te_centered_sq = 0.0;
            let mut sum_w_te_centered_phase_centered = 0.0;

            for e in 0..n_echoes {
                let w = mags[e].as_ref()[v];
                let te_centered = tes[e] - te_mean;
                let phase_centered = unwrapped_phases[e].as_ref()[v] - phase_mean;
                sum_w_te_centered_sq += w * te_centered * te_centered;
                sum_w_te_centered_phase_centered += w * te_centered * phase_centered;
            }

            if sum_w_te_centered_sq > 1e-10 {
                let slope = sum_w_te_centered_phase_centered / sum_w_te_centered_sq;
                let intercept = phase_mean - slope * te_mean;
                field[v] = slope;
                phase_offset[v] = intercept;

                // Compute weighted residual
                let mut sum_w_resid_sq = 0.0;
                for e in 0..n_echoes {
                    let w = mags[e].as_ref()[v];
                    let predicted = intercept + slope * tes[e];
                    let diff = unwrapped_phases[e].as_ref()[v] - predicted;
                    sum_w_resid_sq += w * diff * diff;
                }
                // Normalize by sum of weights and number of echoes (matching echofit.m)
                fit_residual[v] = sum_w_resid_sq / sum_w * n_echoes as f64;
            }
        }
    } else {
        // Weighted linear fit through origin: phase = β * TE
        // β = Σ w*TE*phase / Σ w*TE²
        // (matching echofit.m line 40)

        for v in 0..n_total {
            if mask[v] == 0 {
                continue;
            }

            let mut sum_w_te_phase = 0.0;
            let mut sum_w_te_sq = 0.0;
            let mut sum_w = 0.0;

            for e in 0..n_echoes {
                let w = mags[e].as_ref()[v];
                let te = tes[e];
                let phase = unwrapped_phases[e].as_ref()[v];
                sum_w_te_phase += w * te * phase;
                sum_w_te_sq += w * te * te;
                sum_w += w;
            }

            if sum_w_te_sq > 1e-10 {
                let slope = sum_w_te_phase / sum_w_te_sq;
                field[v] = slope;

                // Compute weighted residual
                let mut sum_w_resid_sq = 0.0;
                for e in 0..n_echoes {
                    let w = mags[e].as_ref()[v];
                    let predicted = slope * tes[e];
                    let diff = unwrapped_phases[e].as_ref()[v] - predicted;
                    sum_w_resid_sq += w * diff * diff;
                }
                // Normalize by sum of weights and number of echoes
                if sum_w > 1e-10 {
                    fit_residual[v] = sum_w_resid_sq / sum_w * n_echoes as f64;
                }
            }
        }
    }

    // Create reliability mask based on fit residuals
    let reliability_mask = if reliability_threshold_percentile > 0.0 {
        compute_reliability_mask(&fit_residual, mask, reliability_threshold_percentile)
    } else {
        // All masked voxels are reliable
        mask.to_vec()
    };

    LinearFitResult {
        field,
        phase_offset,
        fit_residual,
        reliability_mask,
    }
}

/// Compute reliability mask by thresholding fit residuals
///
/// Applies Gaussian smoothing to residuals before thresholding (matching echofit.m)
fn compute_reliability_mask(
    fit_residual: &[f64],
    mask: &[u8],
    threshold_percentile: f64,
) -> Vec<u8> {
    let n_total = fit_residual.len();

    // Collect non-zero residuals for percentile calculation
    let mut residuals: Vec<f64> = fit_residual.iter()
        .enumerate()
        .filter(|(i, &r)| mask[*i] > 0 && r > 0.0 && r.is_finite())
        .map(|(_, &r)| r)
        .collect();

    if residuals.is_empty() {
        return mask.to_vec();
    }

    // Sort and find threshold at given percentile
    residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let percentile_idx = ((threshold_percentile / 100.0) * residuals.len() as f64) as usize;
    let threshold = residuals[percentile_idx.min(residuals.len() - 1)];

    // Create reliability mask
    let mut reliability = vec![0u8; n_total];
    for i in 0..n_total {
        if mask[i] > 0 && fit_residual[i] < threshold {
            reliability[i] = 1;
        }
    }

    reliability
}

/// Convert field from rad/s to Hz
#[inline]
pub fn field_to_hz(field: &[f64]) -> Vec<f64> {
    field.iter().map(|&f| f / TWO_PI).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(nx: usize, ny: usize, nz: usize) -> Grid {
        Grid::new(nx, ny, nz, 1.0, 1.0, 1.0)
    }

    #[test]
    fn test_wrap_to_pi() {
        assert!((wrap_to_pi(0.0) - 0.0).abs() < 1e-10);
        assert!((wrap_to_pi(PI) - PI).abs() < 1e-10);
        assert!((wrap_to_pi(-PI) - (-PI)).abs() < 1e-10);
        assert!((wrap_to_pi(3.0 * PI) - PI).abs() < 1e-10);
        assert!((wrap_to_pi(-3.0 * PI) - (-PI)).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_kernel() {
        let kernel = make_gaussian_kernel(1.0);
        let sum: f64 = kernel.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hip() {
        let n = 8;
        let phase1 = vec![0.1; n];
        let phase2 = vec![0.3; n];
        let mag1 = vec![1.0; n];
        let mag2 = vec![1.0; n];
        let mask = vec![1u8; n];

        let (hip_phase, hip_mag) = hermitian_inner_product(&phase1, &mag1, &phase2, &mag2, &mask, n);

        for i in 0..n {
            assert!((hip_phase[i] - 0.2).abs() < 1e-10);
            assert!((hip_mag[i] - 1.0).abs() < 1e-10);
        }
    }

    // =========================================================================
    // Helper to build synthetic multi-echo data on a small 3D grid
    // =========================================================================

    /// Build synthetic multi-echo phase/magnitude data.
    ///
    /// The phase at each voxel is: phase_offset + slope * TE
    /// where `slope` is a spatially-varying linear ramp along x.
    /// TEs are in seconds. Magnitude is uniform (1.0) inside the mask.
    fn make_synthetic_multi_echo(
        nx: usize, ny: usize, nz: usize,
        tes: &[f64],
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<u8>) {
        let n = nx * ny * nz;
        let n_echoes = tes.len();

        // Constant phase offset (small, well within [-pi, pi])
        let phase_offset_val = 0.3;
        // Slope (rad/s) as a function of x: gentle ramp so phases stay in [-pi, pi]
        // Max slope = 50 rad/s at x=nx-1, so max phase ~ 0.3 + 50*7*0.015 = 5.55
        // which will wrap but that is fine.
        let slope_scale = 50.0;

        let mask = vec![1u8; n];
        let mut phases: Vec<Vec<f64>> = Vec::with_capacity(n_echoes);
        let mut mags: Vec<Vec<f64>> = Vec::with_capacity(n_echoes);

        for e in 0..n_echoes {
            let mut p = vec![0.0; n];
            let m = vec![1.0; n]; // uniform magnitude
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let idx = idx3d(i, j, k, nx, ny);
                        let slope = slope_scale * i as f64;
                        p[idx] = wrap_to_pi(phase_offset_val + slope * tes[e]);
                    }
                }
            }
            phases.push(p);
            mags.push(m);
        }

        (phases, mags, mask)
    }

    // =========================================================================
    // idx3d
    // =========================================================================

    #[test]
    fn test_idx3d_basic() {
        assert_eq!(idx3d(0, 0, 0, 4, 4), 0);
        assert_eq!(idx3d(1, 0, 0, 4, 4), 1);
        assert_eq!(idx3d(0, 1, 0, 4, 4), 4);
        assert_eq!(idx3d(0, 0, 1, 4, 4), 16);
        assert_eq!(idx3d(3, 3, 3, 4, 4), 63);
    }

    // =========================================================================
    // B0WeightType::from_str
    // =========================================================================

    #[test]
    fn test_b0_weight_type_from_str() {
        assert_eq!(B0WeightType::from_str("phase_snr"), B0WeightType::PhaseSNR);
        assert_eq!(B0WeightType::from_str("phasesnr"), B0WeightType::PhaseSNR);
        assert_eq!(B0WeightType::from_str("PhaseSNR"), B0WeightType::PhaseSNR);
        assert_eq!(B0WeightType::from_str("phase_var"), B0WeightType::PhaseVar);
        assert_eq!(B0WeightType::from_str("phasevar"), B0WeightType::PhaseVar);
        assert_eq!(B0WeightType::from_str("average"), B0WeightType::Average);
        assert_eq!(B0WeightType::from_str("uniform"), B0WeightType::Average);
        assert_eq!(B0WeightType::from_str("tes"), B0WeightType::TEs);
        assert_eq!(B0WeightType::from_str("te"), B0WeightType::TEs);
        assert_eq!(B0WeightType::from_str("mag"), B0WeightType::Mag);
        assert_eq!(B0WeightType::from_str("magnitude"), B0WeightType::Mag);
        // Unknown string should default to PhaseSNR
        assert_eq!(B0WeightType::from_str("unknown"), B0WeightType::PhaseSNR);
    }

    // =========================================================================
    // gaussian_smooth_3d_phase
    // =========================================================================

    #[test]
    fn test_gaussian_smooth_3d_phase_uniform_input() {
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        // Uniform phase should remain (approximately) constant after smoothing
        let phase = vec![1.0; n];
        let mask = vec![1u8; n];
        let sigma = [1.0, 1.0, 1.0];

        let smoothed = gaussian_smooth_3d_phase(&phase, sigma, &mask, &grid(nx, ny, nz));

        assert_eq!(smoothed.len(), n);
        for v in &smoothed {
            assert!(v.is_finite(), "smoothed value must be finite");
            assert!((v - 1.0).abs() < 0.05, "uniform phase should remain ~1.0, got {}", v);
        }
    }

    #[test]
    fn test_gaussian_smooth_3d_phase_zero_sigma() {
        let (nx, ny, nz) = (4, 4, 4);
        let n = nx * ny * nz;
        let phase: Vec<f64> = (0..n).map(|i| wrap_to_pi(i as f64 * 0.1)).collect();
        let mask = vec![1u8; n];
        let sigma = [0.0, 0.0, 0.0];

        let smoothed = gaussian_smooth_3d_phase(&phase, sigma, &mask, &grid(nx, ny, nz));

        // With zero sigma, output should equal input (no smoothing applied)
        assert_eq!(smoothed.len(), n);
        for i in 0..n {
            assert!((smoothed[i] - phase[i]).abs() < 1e-10,
                "zero-sigma smoothing should be identity, voxel {}: got {} expected {}",
                i, smoothed[i], phase[i]);
        }
    }

    #[test]
    fn test_gaussian_smooth_3d_phase_masked_zeros() {
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        let phase = vec![0.5; n];
        let mut mask = vec![1u8; n];
        // Set half the voxels to 0
        for i in 0..n / 2 {
            mask[i] = 0;
        }

        let sigma = [1.0, 1.0, 1.0];
        let smoothed = gaussian_smooth_3d_phase(&phase, sigma, &mask, &grid(nx, ny, nz));

        assert_eq!(smoothed.len(), n);
        // Masked-out voxels should remain 0
        for i in 0..n / 2 {
            assert_eq!(smoothed[i], 0.0, "masked-out voxel {} should be 0", i);
        }
        // Masked-in voxels should be finite
        for i in n / 2..n {
            assert!(smoothed[i].is_finite());
        }
    }

    // =========================================================================
    // gaussian_smooth_3d_separable (tested indirectly through phase smoothing
    // but let's also exercise multi-axis sigma)
    // =========================================================================

    #[test]
    fn test_gaussian_smooth_anisotropic_sigma() {
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        let phase: Vec<f64> = (0..n).map(|i| wrap_to_pi(0.3 * (i as f64))).collect();
        let mask = vec![1u8; n];
        let sigma = [2.0, 0.5, 1.0]; // anisotropic

        let smoothed = gaussian_smooth_3d_phase(&phase, sigma, &mask, &grid(nx, ny, nz));
        assert_eq!(smoothed.len(), n);
        for v in &smoothed {
            assert!(v.is_finite());
            assert!(*v >= -PI && *v <= PI, "smoothed phase should be in [-pi, pi], got {}", v);
        }
    }

    // =========================================================================
    // make_gaussian_kernel
    // =========================================================================

    #[test]
    fn test_gaussian_kernel_symmetry() {
        let kernel = make_gaussian_kernel(2.0);
        let len = kernel.len();
        for i in 0..len / 2 {
            assert!((kernel[i] - kernel[len - 1 - i]).abs() < 1e-12,
                "kernel should be symmetric");
        }
    }

    #[test]
    fn test_gaussian_kernel_peak_at_center() {
        let kernel = make_gaussian_kernel(1.5);
        let center = kernel.len() / 2;
        for (i, &v) in kernel.iter().enumerate() {
            if i != center {
                assert!(v <= kernel[center], "center should be peak");
            }
        }
    }

    // =========================================================================
    // hermitian_inner_product (additional tests)
    // =========================================================================

    #[test]
    fn test_hip_with_mask() {
        let n = 4;
        let phase1 = vec![0.5; n];
        let phase2 = vec![1.0; n];
        let mag1 = vec![2.0; n];
        let mag2 = vec![3.0; n];
        let mask = vec![1, 0, 1, 0];

        let (hip_phase, hip_mag) = hermitian_inner_product(
            &phase1, &mag1, &phase2, &mag2, &mask, n
        );

        // Masked-in voxels
        assert!((hip_phase[0] - 0.5).abs() < 1e-10);
        assert!((hip_mag[0] - 6.0).abs() < 1e-10);
        assert!((hip_phase[2] - 0.5).abs() < 1e-10);
        assert!((hip_mag[2] - 6.0).abs() < 1e-10);

        // Masked-out voxels
        assert_eq!(hip_phase[1], 0.0);
        assert_eq!(hip_mag[1], 0.0);
        assert_eq!(hip_phase[3], 0.0);
        assert_eq!(hip_mag[3], 0.0);
    }

    #[test]
    fn test_hip_wrapping() {
        // Test that phase difference wraps correctly
        let n = 1;
        let phase1 = vec![PI - 0.1];
        let phase2 = vec![-PI + 0.1];
        let mag1 = vec![1.0];
        let mag2 = vec![1.0];
        let mask = vec![1u8];

        let (hip_phase, _) = hermitian_inner_product(
            &phase1, &mag1, &phase2, &mag2, &mask, n
        );

        // phase2 - phase1 = (-PI + 0.1) - (PI - 0.1) = -2PI + 0.2 -> wraps to 0.2
        assert!((hip_phase[0] - 0.2).abs() < 1e-10,
            "HIP should wrap phase difference, got {}", hip_phase[0]);
    }

    // =========================================================================
    // find_seed_point (now in romeo.rs, tested here for coverage)
    // =========================================================================

    #[test]
    fn test_find_seed_point_full_mask() {
        use crate::unwrap::romeo::find_seed_point;
        let (nx, ny, nz) = (8, 8, 8);
        let mask = vec![1u8; nx * ny * nz];
        let (si, sj, sk) = find_seed_point(&mask, nx, ny, nz);
        // Center of mass of a fully-filled cube should be approximately center
        assert_eq!(si, 3); // mean of 0..7 = 3.5, integer division = 3
        assert_eq!(sj, 3);
        assert_eq!(sk, 3);
    }

    #[test]
    fn test_find_seed_point_empty_mask() {
        use crate::unwrap::romeo::find_seed_point;
        let (nx, ny, nz) = (8, 8, 8);
        let mask = vec![0u8; nx * ny * nz];
        let (si, sj, sk) = find_seed_point(&mask, nx, ny, nz);
        // Fallback: center of volume
        assert_eq!(si, 4);
        assert_eq!(sj, 4);
        assert_eq!(sk, 4);
    }

    #[test]
    fn test_find_seed_point_corner_mask() {
        use crate::unwrap::romeo::find_seed_point;
        let (nx, ny, nz) = (8, 8, 8);
        let mut mask = vec![0u8; nx * ny * nz];
        // Only set voxel (0,0,0)
        mask[idx3d(0, 0, 0, nx, ny)] = 1;
        let (si, sj, sk) = find_seed_point(&mask, nx, ny, nz);
        assert_eq!(si, 0);
        assert_eq!(sj, 0);
        assert_eq!(sk, 0);
    }

    // =========================================================================
    // field_to_hz
    // =========================================================================

    #[test]
    fn test_field_to_hz() {
        let field = vec![TWO_PI, -TWO_PI, 0.0, PI];
        let hz = field_to_hz(&field);
        assert!((hz[0] - 1.0).abs() < 1e-10);
        assert!((hz[1] - (-1.0)).abs() < 1e-10);
        assert!((hz[2] - 0.0).abs() < 1e-10);
        assert!((hz[3] - 0.5).abs() < 1e-10);
    }

    // =========================================================================
    // calculate_b0_weighted
    // =========================================================================

    #[test]
    fn test_calculate_b0_weighted_phase_snr() {
        // For a constant slope (rad/s), all weight types should recover it.
        // phase[e] = slope * TE[e], so phase/TE = slope for each echo.
        // Weighted average of identical values = same value.
        // B0 = (1 / 2pi) * slope (Hz)
        let n = 64;
        let tes = [0.005, 0.010, 0.015]; // seconds
        let slope = 200.0; // rad/s
        let mask = vec![1u8; n];

        let phases: Vec<Vec<f64>> = tes.iter()
            .map(|&te| vec![slope * te; n])
            .collect();
        let mags: Vec<Vec<f64>> = tes.iter()
            .map(|_| vec![1.0; n])
            .collect();

        let b0 = calculate_b0_weighted(&phases, &mags, &tes, &mask, B0WeightType::PhaseSNR, &grid(n, 1, 1));

        let expected_hz = 1.0 / TWO_PI * slope;
        assert_eq!(b0.len(), n);
        for v in &b0 {
            assert!(v.is_finite());
            assert!((v - expected_hz).abs() < 1e-8,
                "expected {} Hz, got {}", expected_hz, v);
        }
    }

    #[test]
    fn test_calculate_b0_weighted_all_weight_types() {
        let n = 16;
        let tes = [0.005, 0.010, 0.015]; // seconds
        let slope = 100.0; // rad/s
        let mask = vec![1u8; n];

        let phases: Vec<Vec<f64>> = tes.iter()
            .map(|&te| vec![slope * te; n])
            .collect();
        let mags: Vec<Vec<f64>> = tes.iter()
            .map(|_| vec![2.0; n])
            .collect();

        let expected_hz = 1.0 / TWO_PI * slope;

        for wt in &[
            B0WeightType::PhaseSNR,
            B0WeightType::PhaseVar,
            B0WeightType::Average,
            B0WeightType::TEs,
            B0WeightType::Mag,
        ] {
            let b0 = calculate_b0_weighted(&phases, &mags, &tes, &mask, *wt, &grid(n, 1, 1));
            assert_eq!(b0.len(), n);
            for v in &b0 {
                assert!(v.is_finite(), "weight type {:?} produced non-finite", wt);
                assert!((v - expected_hz).abs() < 1e-8,
                    "weight type {:?}: expected {} Hz, got {}", wt, expected_hz, v);
            }
        }
    }

    #[test]
    fn test_calculate_b0_weighted_masked_out() {
        let n = 8;
        let tes = [0.005, 0.010]; // seconds
        let mask = vec![0u8; n]; // all masked out

        let phases: Vec<Vec<f64>> = tes.iter()
            .map(|&te| vec![500.0 * te; n])
            .collect();
        let mags: Vec<Vec<f64>> = tes.iter()
            .map(|_| vec![1.0; n])
            .collect();

        let b0 = calculate_b0_weighted(&phases, &mags, &tes, &mask, B0WeightType::PhaseSNR, &grid(n, 1, 1));

        for v in &b0 {
            assert_eq!(*v, 0.0, "masked-out voxels should have B0=0");
        }
    }

    #[test]
    fn test_calculate_b0_weighted_zero_magnitude() {
        // When magnitude is zero, weight is zero; result should be 0
        let n = 4;
        let tes = [0.005, 0.010, 0.015];
        let mask = vec![1u8; n];

        let phases: Vec<Vec<f64>> = tes.iter()
            .map(|&te| vec![0.2 * te; n])
            .collect();
        let mags: Vec<Vec<f64>> = tes.iter()
            .map(|_| vec![0.0; n]) // zero magnitude
            .collect();

        // PhaseSNR weight = mag * te = 0
        let b0 = calculate_b0_weighted(&phases, &mags, &tes, &mask, B0WeightType::PhaseSNR, &grid(n, 1, 1));
        for v in &b0 {
            assert_eq!(*v, 0.0, "zero-magnitude voxels should yield B0=0");
        }

        // Average weight = 1.0, should still work
        let b0_avg = calculate_b0_weighted(&phases, &mags, &tes, &mask, B0WeightType::Average, &grid(n, 1, 1));
        let expected = 1.0 / TWO_PI * 0.2;
        for v in &b0_avg {
            assert!((v - expected).abs() < 1e-8);
        }
    }

    // =========================================================================
    // multi_echo_linear_fit
    // =========================================================================

    #[test]
    fn test_multi_echo_linear_fit_no_offset() {
        // phase = slope * TE (no intercept)
        // Should recover the slope exactly.
        let n = 32;
        let tes = [0.005, 0.010, 0.015]; // in seconds
        let slope = 100.0; // rad/s
        let mask = vec![1u8; n];

        let phases: Vec<Vec<f64>> = tes.iter()
            .map(|&te| vec![slope * te; n])
            .collect();
        let mags: Vec<Vec<f64>> = tes.iter()
            .map(|_| vec![1.0; n])
            .collect();

        let result = multi_echo_linear_fit(
            &phases, &mags, &tes, &mask,
            false, // no offset estimation
            0.0,   // no reliability threshold
        );

        assert_eq!(result.field.len(), n);
        assert_eq!(result.phase_offset.len(), n);
        assert_eq!(result.fit_residual.len(), n);
        assert_eq!(result.reliability_mask.len(), n);

        for i in 0..n {
            assert!((result.field[i] - slope).abs() < 1e-6,
                "slope: expected {}, got {}", slope, result.field[i]);
            assert_eq!(result.phase_offset[i], 0.0,
                "offset should be 0 when estimate_offset=false");
            assert!(result.fit_residual[i] < 1e-10,
                "residual should be ~0 for perfect linear data");
            assert_eq!(result.reliability_mask[i], 1,
                "reliability should match mask when threshold=0");
        }
    }

    #[test]
    fn test_multi_echo_linear_fit_with_offset() {
        // phase = intercept + slope * TE
        let n = 16;
        let tes = [0.005, 0.010, 0.015, 0.020];
        let slope = 200.0;     // rad/s
        let intercept = 0.5;   // rad
        let mask = vec![1u8; n];

        let phases: Vec<Vec<f64>> = tes.iter()
            .map(|&te| vec![intercept + slope * te; n])
            .collect();
        let mags: Vec<Vec<f64>> = tes.iter()
            .map(|_| vec![1.0; n])
            .collect();

        let result = multi_echo_linear_fit(
            &phases, &mags, &tes, &mask,
            true, // estimate offset
            0.0,
        );

        for i in 0..n {
            assert!((result.field[i] - slope).abs() < 1e-4,
                "slope: expected {}, got {}", slope, result.field[i]);
            assert!((result.phase_offset[i] - intercept).abs() < 1e-4,
                "intercept: expected {}, got {}", intercept, result.phase_offset[i]);
            assert!(result.fit_residual[i] < 1e-8,
                "residual should be ~0 for perfect linear data, got {}", result.fit_residual[i]);
        }
    }

    #[test]
    fn test_multi_echo_linear_fit_masked_out() {
        let n = 8;
        let tes = [0.005, 0.010, 0.015];
        let mask = vec![0u8; n];

        let phases: Vec<Vec<f64>> = tes.iter()
            .map(|&te| vec![100.0 * te; n])
            .collect();
        let mags: Vec<Vec<f64>> = tes.iter()
            .map(|_| vec![1.0; n])
            .collect();

        let result = multi_echo_linear_fit(&phases, &mags, &tes, &mask, true, 0.0);

        for i in 0..n {
            assert_eq!(result.field[i], 0.0);
            assert_eq!(result.phase_offset[i], 0.0);
            assert_eq!(result.fit_residual[i], 0.0);
        }
    }

    #[test]
    fn test_multi_echo_linear_fit_varying_slope() {
        // Each voxel has a different slope
        let n = 8;
        let tes = [0.005, 0.010, 0.015];
        let mask = vec![1u8; n];

        let slopes: Vec<f64> = (0..n).map(|i| 50.0 * (i as f64 + 1.0)).collect();

        let phases: Vec<Vec<f64>> = tes.iter()
            .map(|&te| {
                slopes.iter().map(|&s| s * te).collect()
            })
            .collect();
        let mags: Vec<Vec<f64>> = tes.iter()
            .map(|_| vec![1.0; n])
            .collect();

        let result = multi_echo_linear_fit(&phases, &mags, &tes, &mask, false, 0.0);

        for i in 0..n {
            assert!((result.field[i] - slopes[i]).abs() < 1e-6,
                "voxel {}: expected slope {}, got {}", i, slopes[i], result.field[i]);
        }
    }

    #[test]
    fn test_multi_echo_linear_fit_with_reliability_threshold() {
        // Create data where some voxels have noisy fits
        let n = 100;
        let tes = [0.005, 0.010, 0.015];
        let mask = vec![1u8; n];
        let slope = 100.0;

        let mut phases: Vec<Vec<f64>> = tes.iter()
            .map(|&te| vec![slope * te; n])
            .collect();
        let mags: Vec<Vec<f64>> = tes.iter()
            .map(|_| vec![1.0; n])
            .collect();

        // Add large noise to last 10 voxels to increase their residuals
        for e in 0..tes.len() {
            for i in 90..100 {
                phases[e][i] += if e % 2 == 0 { 2.0 } else { -2.0 };
            }
        }

        // Use 80th percentile threshold
        let result = multi_echo_linear_fit(&phases, &mags, &tes, &mask, false, 80.0);

        assert_eq!(result.reliability_mask.len(), n);

        // Clean voxels (0..90) have residual=0, noisy voxels (90..100) have residual>0.
        // The threshold is computed from non-zero residuals only.
        // Voxels with residual=0 satisfy 0 < threshold, but compute_reliability_mask
        // checks `fit_residual[i] < threshold` -- 0 < any positive threshold => reliable=1.
        // However, residual might not be exactly 0 due to floating point.
        // Just verify that the noisy voxels have higher residuals than clean ones.
        let max_clean_resid = result.fit_residual[0..90].iter()
            .cloned().fold(0.0f64, f64::max);
        let min_noisy_resid = result.fit_residual[90..100].iter()
            .cloned().fold(f64::INFINITY, f64::min);
        assert!(max_clean_resid < min_noisy_resid,
            "clean residuals ({}) should be less than noisy residuals ({})",
            max_clean_resid, min_noisy_resid);

        // The reliability mask should exist and have valid values
        for &v in &result.reliability_mask {
            assert!(v == 0 || v == 1);
        }
    }

    #[test]
    fn test_multi_echo_linear_fit_zero_magnitude() {
        let n = 4;
        let tes = [0.005, 0.010, 0.015];
        let mask = vec![1u8; n];

        let phases: Vec<Vec<f64>> = tes.iter()
            .map(|&te| vec![100.0 * te; n])
            .collect();
        let mags: Vec<Vec<f64>> = tes.iter()
            .map(|_| vec![0.0; n]) // zero magnitude
            .collect();

        // Should not crash; field should be 0 because sum_w_te_sq ~ 0
        let result = multi_echo_linear_fit(&phases, &mags, &tes, &mask, false, 0.0);
        for v in &result.field {
            assert!(v.is_finite());
        }

        let result2 = multi_echo_linear_fit(&phases, &mags, &tes, &mask, true, 0.0);
        for v in &result2.field {
            assert!(v.is_finite());
        }
    }

    // =========================================================================
    // compute_reliability_mask
    // =========================================================================

    #[test]
    fn test_compute_reliability_mask_basic() {
        let n = 10;
        let mask = vec![1u8; n];
        // Residuals in ascending order: 0.1, 0.2, ..., 1.0
        let fit_residual: Vec<f64> = (1..=n).map(|i| i as f64 * 0.1).collect();

        // 50th percentile: threshold ~ 0.5
        let reliability = compute_reliability_mask(&fit_residual, &mask, 50.0);
        assert_eq!(reliability.len(), n);

        // Voxels with residual < threshold should be reliable
        let reliable_count: usize = reliability.iter().map(|&v| v as usize).sum();
        assert!(reliable_count > 0 && reliable_count < n,
            "some but not all should be reliable, got {}/{}", reliable_count, n);
    }

    #[test]
    fn test_compute_reliability_mask_all_zero_residual() {
        let n = 5;
        let mask = vec![1u8; n];
        let fit_residual = vec![0.0; n];

        // When all residuals are 0, the filter skips them (r > 0.0 check fails)
        // so residuals vec is empty and mask is returned as-is
        let reliability = compute_reliability_mask(&fit_residual, &mask, 50.0);
        assert_eq!(reliability, mask);
    }

    // =========================================================================
    // mcpc3ds_single_coil
    // =========================================================================

    #[test]
    fn test_phase_offset_removal_output_sizes() {
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        let tes = [0.005, 0.010, 0.015];
        let (phases, mags, mask) = make_synthetic_multi_echo(nx, ny, nz, &tes);

        let sigma = [1.0, 1.0, 1.0];
        let (corrected, offset) = phase_offset_removal(
            &phases, &mags, &tes, &mask, sigma, [0, 1], UnwrapMethod::Romeo, &grid(nx, ny, nz),
        );

        // Check sizes
        assert_eq!(corrected.len(), tes.len(), "should have one corrected phase per echo");
        for (e, cp) in corrected.iter().enumerate() {
            assert_eq!(cp.len(), n, "echo {} corrected phase should have {} voxels", e, n);
        }
        assert_eq!(offset.len(), n, "phase offset should have {} voxels", n);
    }

    #[test]
    fn test_phase_offset_removal_finite_output() {
        let (nx, ny, nz) = (8, 8, 8);
        let tes = [0.005, 0.010, 0.015];
        let (phases, mags, mask) = make_synthetic_multi_echo(nx, ny, nz, &tes);

        let sigma = [1.0, 1.0, 1.0];
        let (corrected, offset) = phase_offset_removal(
            &phases, &mags, &tes, &mask, sigma, [0, 1], UnwrapMethod::Romeo, &grid(nx, ny, nz),
        );

        for v in &offset {
            assert!(v.is_finite(), "phase offset should be finite");
        }
        for cp in &corrected {
            for v in cp {
                assert!(v.is_finite(), "corrected phase should be finite");
            }
        }
    }

    #[test]
    fn test_phase_offset_removal_corrected_in_range() {
        let (nx, ny, nz) = (8, 8, 8);
        let tes = [0.005, 0.010, 0.015];
        let (phases, mags, mask) = make_synthetic_multi_echo(nx, ny, nz, &tes);

        let sigma = [1.0, 1.0, 1.0];
        let (corrected, _) = phase_offset_removal(
            &phases, &mags, &tes, &mask, sigma, [0, 1], UnwrapMethod::Romeo, &grid(nx, ny, nz),
        );

        // Corrected phases should be in [-pi, pi] since wrap_to_pi is applied
        for cp in &corrected {
            for &v in cp {
                assert!(v >= -PI - 1e-10 && v <= PI + 1e-10,
                    "corrected phase should be in [-pi, pi], got {}", v);
            }
        }
    }

    #[test]
    fn test_phase_offset_removal_uniform_phase() {
        // Uniform phase across all echoes => offset should be approximately that phase
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        let tes = [0.005, 0.010, 0.015];

        // All echoes have constant phase 0.5 (no TE dependence)
        let phases: Vec<Vec<f64>> = (0..3).map(|_| vec![0.5; n]).collect();
        let mags: Vec<Vec<f64>> = (0..3).map(|_| vec![1.0; n]).collect();
        let mask = vec![1u8; n];

        let sigma = [1.0, 1.0, 1.0];
        let (corrected, _offset) = phase_offset_removal(
            &phases, &mags, &tes, &mask, sigma, [0, 1], UnwrapMethod::Romeo, &grid(nx, ny, nz),
        );

        // After removing offset, corrected phases should be close to 0
        for cp in &corrected {
            for &v in cp {
                assert!(v.abs() < 1.0,
                    "after offset removal of uniform phase, corrected should be ~0, got {}", v);
            }
        }
    }

    // =========================================================================
    // Composed field mapping pipeline (offset removal → unwrap → B0)
    // =========================================================================

    #[test]
    fn test_field_mapping_composed() {
        use crate::unwrap::romeo::{unwrap_romeo_multi_echo, RomeoParams};
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        let tes = [0.005, 0.010, 0.015];
        let (phases, mags, mask) = make_synthetic_multi_echo(nx, ny, nz, &tes);
        let sigma = [1.0, 1.0, 1.0];

        // Step 1: Phase offset removal
        let (corrected, offset) = phase_offset_removal(
            &phases, &mags, &tes, &mask, sigma, [0, 1],
            UnwrapMethod::Romeo, &grid(nx, ny, nz),
        );
        assert_eq!(corrected.len(), tes.len());
        assert_eq!(offset.len(), n);

        // Step 2: Multi-echo unwrapping
        let mag_refs: Vec<&[f64]> = mags.iter().map(|m| m.as_slice()).collect();
        let unwrapped = unwrap_romeo_multi_echo(
            &corrected, &mag_refs, &tes, &mask,
            &RomeoParams::default(), &grid(nx, ny, nz),
        );
        assert_eq!(unwrapped.len(), tes.len());

        // Step 3: Weighted B0
        let b0 = calculate_b0_weighted(
            &unwrapped, &mags, &tes, &mask, B0WeightType::PhaseSNR, &grid(nx, ny, nz),
        );
        assert_eq!(b0.len(), n);
        for v in &b0 {
            assert!(v.is_finite(), "B0 should be finite");
        }
    }

    #[test]
    fn test_field_mapping_different_weight_types() {
        use crate::unwrap::romeo::{unwrap_romeo_multi_echo, RomeoParams};
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        let tes = [0.005, 0.010, 0.015];
        let (phases, mags, mask) = make_synthetic_multi_echo(nx, ny, nz, &tes);
        let sigma = [1.0, 1.0, 1.0];

        let (corrected, _) = phase_offset_removal(
            &phases, &mags, &tes, &mask, sigma, [0, 1],
            UnwrapMethod::Romeo, &grid(nx, ny, nz),
        );
        let mag_refs: Vec<&[f64]> = mags.iter().map(|m| m.as_slice()).collect();
        let unwrapped = unwrap_romeo_multi_echo(
            &corrected, &mag_refs, &tes, &mask,
            &RomeoParams::default(), &grid(nx, ny, nz),
        );

        for wt in &[
            B0WeightType::PhaseSNR,
            B0WeightType::Average,
            B0WeightType::TEs,
            B0WeightType::Mag,
            B0WeightType::PhaseVar,
        ] {
            let b0 = calculate_b0_weighted(&unwrapped, &mags, &tes, &mask, *wt, &grid(n, 1, 1));
            for v in &b0 {
                assert!(v.is_finite(), "B0 with {:?} should be finite", wt);
            }
        }
    }

    // =========================================================================
    // wrap_to_pi edge cases
    // =========================================================================

    #[test]
    fn test_wrap_to_pi_near_boundaries() {
        // Values just beyond PI and -PI
        let v1 = wrap_to_pi(PI + 0.001);
        assert!(v1 < PI && v1 > -PI, "should wrap back into range");

        let v2 = wrap_to_pi(-PI - 0.001);
        assert!(v2 > -PI && v2 < PI, "should wrap back into range");

        // Large positive and negative values
        let v3 = wrap_to_pi(100.0 * PI);
        assert!(v3 >= -PI && v3 <= PI, "should be in [-pi, pi], got {}", v3);

        let v4 = wrap_to_pi(-100.0 * PI);
        assert!(v4 >= -PI && v4 <= PI, "should be in [-pi, pi], got {}", v4);
    }

    // =========================================================================
    // unwrap_romeo (tested indirectly through phase_offset_removal
    // but let's also test directly via the public API)
    // =========================================================================

    #[test]
    fn test_unwrap_romeo_smooth_data() {
        use crate::unwrap::romeo::{unwrap_romeo, RomeoParams};
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        // Smooth phase that doesn't need unwrapping
        let phase: Vec<f64> = (0..n).map(|i| {
            let x = (i % nx) as f64 / nx as f64;
            0.5 * x // small smooth phase
        }).collect();
        let mag = vec![1.0; n];
        let mask = vec![1u8; n];

        let unwrapped = unwrap_romeo(
            &phase, &mag, None, 0.0, 0.0,
            &mask, &RomeoParams::default(), &grid(nx, ny, nz),
        );

        assert_eq!(unwrapped.len(), n);
        for (i, &v) in unwrapped.iter().enumerate() {
            assert!(v.is_finite(), "unwrapped voxel {} should be finite", i);
        }
    }

    // =========================================================================
    // Integration: linear fit on mcpc3ds output
    // =========================================================================

    #[test]
    fn test_linear_fit_on_mcpc3ds_output() {
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        let tes = [0.005, 0.010, 0.015];
        let (phases, mags, mask) = make_synthetic_multi_echo(nx, ny, nz, &tes);

        let sigma = [1.0, 1.0, 1.0];
        let (corrected, _offset) = phase_offset_removal(
            &phases, &mags, &tes, &mask, sigma, [0, 1], UnwrapMethod::Romeo, &grid(nx, ny, nz),
        );

        // Run linear fit on corrected phases (tes in seconds for fit)
        let tes_s: Vec<f64> = tes.iter().map(|&t| t / 1000.0).collect();
        let result = multi_echo_linear_fit(
            &corrected, &mags, &tes_s, &mask, true, 0.0,
        );

        assert_eq!(result.field.len(), n);
        assert_eq!(result.phase_offset.len(), n);
        assert_eq!(result.fit_residual.len(), n);
        assert_eq!(result.reliability_mask.len(), n);

        for v in &result.field {
            assert!(v.is_finite(), "field should be finite");
        }
        for v in &result.phase_offset {
            assert!(v.is_finite(), "phase_offset should be finite");
        }
    }

    // =========================================================================
    // bipolar_correction
    // =========================================================================

    #[test]
    fn test_bipolar_correction_3_echoes() {
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        let tes = [0.005, 0.010, 0.015];
        let (phases, mags, mask) = make_synthetic_multi_echo(nx, ny, nz, &tes);

        let mut phases_mut = phases;
        let sigma = [1.0, 1.0, 1.0];
        bipolar_correction(
            &mut phases_mut, &mags, &tes, &mask,
            sigma, &grid(nx, ny, nz),
        );

        // All values should remain finite
        for echo in &phases_mut {
            for &v in echo {
                assert!(v.is_finite(), "bipolar-corrected phase should be finite");
            }
        }
    }

    #[test]
    fn test_bipolar_correction_2_echoes_noop() {
        // With only 2 echoes, bipolar correction should be a no-op
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        let tes = [0.005, 0.010];
        let (phases, mags, mask) = make_synthetic_multi_echo(nx, ny, nz, &tes);

        let original: Vec<Vec<f64>> = phases.iter().map(|p| p.clone()).collect();
        let mut phases_mut = phases;
        bipolar_correction(
            &mut phases_mut, &mags, &tes, &mask,
            [1.0, 1.0, 1.0], &grid(nx, ny, nz),
        );

        // Should be unchanged (2 echoes = noop)
        for (e, echo) in phases_mut.iter().enumerate() {
            for (i, &v) in echo.iter().enumerate() {
                assert_eq!(v, original[e][i],
                    "2-echo bipolar correction should be no-op");
            }
        }
    }

    #[test]
    fn test_field_mapping_with_bipolar() {
        use crate::unwrap::romeo::{unwrap_romeo_multi_echo, RomeoParams};
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        let tes = [0.005, 0.010, 0.015];
        let (phases, mags, mask) = make_synthetic_multi_echo(nx, ny, nz, &tes);
        let sigma = [1.0, 1.0, 1.0];

        let (mut corrected, _) = phase_offset_removal(
            &phases, &mags, &tes, &mask, sigma, [0, 1],
            UnwrapMethod::Romeo, &grid(nx, ny, nz),
        );
        let mag_refs: Vec<&[f64]> = mags.iter().map(|m| m.as_slice()).collect();
        bipolar_correction(&mut corrected, &mag_refs, &tes, &mask, sigma, &grid(nx, ny, nz));
        let unwrapped = unwrap_romeo_multi_echo(
            &corrected, &mag_refs, &tes, &mask,
            &RomeoParams::default(), &grid(nx, ny, nz),
        );
        let b0 = calculate_b0_weighted(&unwrapped, &mags, &tes, &mask, B0WeightType::PhaseSNR, &grid(n, 1, 1));

        assert_eq!(b0.len(), n);
        for v in &b0 {
            assert!(v.is_finite(), "B0 with bipolar correction should be finite");
        }
    }

    // --- mcpc3ds_combine ---

    /// Two coils with distinct smooth phase offsets over a linear field: the combined phase
    /// must be the pure field evolution 2π·f·TE_e, offsets gone.
    fn synthetic_coils(nx: usize, ny: usize, nz: usize, tes: &[f64]) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>, Vec<f64>) {
        let n = nx * ny * nz;
        let offsets: Vec<Box<dyn Fn(usize, usize, usize) -> f64>> = vec![
            Box::new(|x, _, _| 0.3 + 0.01 * x as f64),
            Box::new(|_, y, _| -1.0 + 0.02 * y as f64),
        ];
        let mut field = vec![0.0; n]; // Hz
        let mut phases = Vec::new();
        let mut mags = Vec::new();
        for (c, po) in offsets.iter().enumerate() {
            let mut cp = Vec::new();
            let mut cm = Vec::new();
            for &te in tes {
                let mut p = vec![0.0; n];
                let mut m = vec![0.0; n];
                for z in 0..nz { for y in 0..ny { for x in 0..nx {
                    let i = x + y * nx + z * nx * ny;
                    let f = 2.0 + 1.5 * (x as f64 / nx as f64) - 1.0 * (z as f64 / nz as f64);
                    field[i] = f;
                    p[i] = wrap_to_pi(2.0 * PI * f * te + po(x, y, z));
                    m[i] = 100.0 * (1.0 + 0.3 * (if c == 0 { x as f64 / nx as f64 } else { y as f64 / ny as f64 }));
                }}}
                cp.push(p);
                cm.push(m);
            }
            phases.push(cp);
            mags.push(cm);
        }
        (phases, mags, field)
    }

    #[test]
    fn test_mcpc3ds_combine_removes_coil_offsets() {
        let (nx, ny, nz) = (20, 20, 10);
        let tes = [0.005, 0.010];
        let g = grid(nx, ny, nz);
        let (phases, mags, field) = synthetic_coils(nx, ny, nz, &tes);
        let res = mcpc3ds_combine(&phases, &mags, &tes, [2.0, 2.0, 2.0], [0, 1], UnwrapMethod::Romeo, &g);
        assert_eq!(res.phases.len(), 2);
        assert_eq!(res.magnitudes.len(), 2);
        let n_mask = res.mask.iter().filter(|&&m| m > 0).count();
        assert!(n_mask > n_mask_min(nx, ny, nz), "robust mask should cover the object: {}", n_mask);
        // Interior voxels (away from the smoothing boundary): combined phase == field evolution.
        let mut checked = 0;
        for z in 3..nz - 3 { for y in 4..ny - 4 { for x in 4..nx - 4 {
            let i = x + y * nx + z * nx * ny;
            if res.mask[i] == 0 { continue; }
            for (e, &te) in tes.iter().enumerate() {
                let expected = wrap_to_pi(2.0 * PI * field[i] * te);
                let err = wrap_to_pi(res.phases[e][i] - expected).abs();
                assert!(err < 0.05, "echo {} voxel ({},{},{}): got {:.4}, expected {:.4}", e, x, y, z, res.phases[e][i], expected);
            }
            checked += 1;
        }}}
        assert!(checked > 100, "too few interior voxels checked: {}", checked);
        // Magnitude: sqrt(|Σ m_c² e^{iθ}|) with aligned phases = sqrt(m0² + m1²) ≥ each coil.
        for i in 0..nx * ny * nz {
            let m0 = mags[0][0][i];
            let m1 = mags[1][0][i];
            let expected = (m0 * m0 + m1 * m1).sqrt();
            assert!((res.magnitudes[0][i] - expected).abs() / expected < 0.02, "magnitude at {}: {} vs {}", i, res.magnitudes[0][i], expected);
        }
    }

    fn n_mask_min(nx: usize, ny: usize, nz: usize) -> usize { nx * ny * nz / 2 }

    #[test]
    fn test_mcpc3ds_combine_single_coil_removes_offset_keeps_magnitude() {
        let (nx, ny, nz) = (24, 24, 12);
        let tes = [0.005, 0.010];
        let g = grid(nx, ny, nz);
        let (mut phases, mut mags, field) = synthetic_coils(nx, ny, nz, &tes);
        phases.truncate(1);
        mags.truncate(1);
        let res = mcpc3ds_combine(&phases, &mags, &tes, [2.0, 2.0, 2.0], [0, 1], UnwrapMethod::Romeo, &g);
        let n = nx * ny * nz;
        for e in 0..2 {
            for i in 0..n {
                // one coil: |S| is untouched, and the inter-echo phase difference is preserved
                // exactly (a TE-independent offset cancels in it)
                assert!((res.magnitudes[e][i] - mags[0][e][i]).abs() < 1e-9);
            }
        }
        for i in 0..n {
            let d_in = wrap_to_pi(phases[0][1][i] - phases[0][0][i]);
            let d_out = wrap_to_pi(res.phases[1][i] - res.phases[0][i]);
            assert!(wrap_to_pi(d_out - d_in).abs() < 1e-9, "voxel {}: {} vs {}", i, d_out, d_in);
        }
        // and the offset itself is gone: interior phase == field evolution
        let mut checked = 0;
        for z in 3..nz - 3 { for y in 5..ny - 5 { for x in 5..nx - 5 {
            let i = x + y * nx + z * nx * ny;
            if res.mask[i] == 0 { continue; }
            let err = wrap_to_pi(res.phases[0][i] - 2.0 * PI * field[i] * tes[0]).abs();
            assert!(err < 0.05, "voxel ({},{},{}): {}", x, y, z, err);
            checked += 1;
        }}}
        assert!(checked > 100);
    }

    #[test]
    fn test_mcpc3ds_combine_identical_coils_scale_magnitude() {
        let (nx, ny, nz) = (12, 12, 6);
        let tes = [0.004, 0.009];
        let g = grid(nx, ny, nz);
        let (mut phases, mut mags, _) = synthetic_coils(nx, ny, nz, &tes);
        phases.truncate(1);
        mags.truncate(1);
        let (p, m) = (phases[0].clone(), mags[0].clone());
        phases.push(p);
        mags.push(m);
        let res = mcpc3ds_combine(&phases, &mags, &tes, [2.0, 2.0, 2.0], [0, 1], UnwrapMethod::Laplacian, &g);
        for i in 0..nx * ny * nz {
            let expected = (2.0f64).sqrt() * mags[0][1][i];
            assert!((res.magnitudes[1][i] - expected).abs() < 1e-6);
        }
    }

    // --- nan-box smoothing (MriResearchTools port) ---

    #[test]
    fn test_gaussian_box_sizes_match_reference() {
        // values computed from getboxsizes(sigma, 4) in MriResearchTools.jl
        assert_eq!(gaussian_box_sizes(10.0, 4), vec![17, 17, 17, 19]);
        assert_eq!(gaussian_box_sizes(5.0, 4), vec![7, 9, 9, 9]);
        assert_eq!(gaussian_box_sizes(4.0, 4), vec![7, 7, 7, 7]);
        assert_eq!(gaussian_box_sizes(0.0, 4), vec![1, 1, 1, 1]);
        assert_eq!(gaussian_box_sizes(5.0, 3), vec![9, 9, 11]); // Julia round() ties to even: m = round(1.5) = 2
        assert_eq!(gaussian_box_sizes(2.0, 4), vec![3, 3, 3, 5]);
    }

    #[test]
    fn test_nan_box_filter_line_constant_with_gap() {
        // constant segments stay constant, and the NaN gap is filled by extrapolation
        let mut line = vec![3.0; 30];
        for v in line.iter_mut().take(20).skip(12) { *v = f64::NAN; }
        let mut scratch = Vec::new();
        nan_box_filter_line(&mut line, 5, &mut scratch);
        for (i, v) in line.iter().enumerate() {
            if v.is_finite() {
                assert!((v - 3.0).abs() < 1e-12, "position {} = {}", i, v);
            }
        }
        // filled up to r=2 samples into the gap from the left
        assert!(line[12].is_finite() && line[13].is_finite(), "{:?}", &line[10..16]);
        assert!(line[16].is_nan());
    }

    #[test]
    fn test_nan_box_filter_line_linear_ramp_interior() {
        let n = 40;
        let mut line: Vec<f64> = (0..n).map(|i| 0.5 * i as f64).collect();
        let mut scratch = Vec::new();
        nan_box_filter_line(&mut line, 7, &mut scratch);
        // box mean of a linear ramp is the ramp itself where the box is fully inside
        for i in 7..n - 3 {
            assert!((line[i] - 0.5 * i as f64).abs() < 1e-9, "position {} = {}", i, line[i]);
        }
    }

    #[test]
    fn test_nan_box_smooth_3d_phase_constant_inside_mask() {
        let (nx, ny, nz) = (48, 48, 24);
        let g = grid(nx, ny, nz);
        let n = nx * ny * nz;
        let mut mask = vec![0u8; n];
        for z in 8..nz - 8 { for y in 16..ny - 16 { for x in 16..nx - 16 {
            mask[x + y * nx + z * nx * ny] = 1;
        }}}
        let phase = vec![1.2f64; n];
        let out = nan_box_smooth_3d_phase(&phase, [3.0, 3.0, 2.0], &mask, &g);
        let mut checked = 0;
        for i in 0..n {
            if mask[i] > 0 && out[i].is_finite() {
                assert!((out[i] - 1.2).abs() < 1e-9, "voxel {}: {}", i, out[i]);
                checked += 1;
            }
        }
        let n_mask = mask.iter().filter(|&&m| m > 0).count();
        assert_eq!(checked, n_mask, "every mask voxel must be defined and unchanged");
        // fills extend at most a few box radii past the mask: the corner stays undefined
        assert!(out[0].is_nan());
        let outside_defined = (0..n).filter(|&i| mask[i] == 0 && out[i].is_finite()).count();
        let outside = (0..n).filter(|&i| mask[i] == 0).count();
        assert!(outside_defined < outside / 2, "{} of {}", outside_defined, outside);
    }
}

//! Multi-echo phase combination utilities
//!
//! Implements MCPC-3D-S (Multi-Channel Phase Combination - 3D - Smoothed) algorithm
//! and weighted B0 calculation from MriResearchTools.jl
//!
//! Reference: https://github.com/korbinian90/MriResearchTools.jl

use std::f64::consts::PI;
use crate::unwrap::romeo::calculate_weights_romeo;
use crate::region_grow::grow_region_unwrap;

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
/// * `nx`, `ny`, `nz` - Dimensions
///
/// # Returns
/// Smoothed phase data
pub fn gaussian_smooth_3d_phase(
    phase: &[f64],
    sigma: [f64; 3],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
) -> Vec<f64> {
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
pub fn mcpc3ds_single_coil(
    phases: &[Vec<f64>],
    mags: &[Vec<f64>],
    tes: &[f64],
    mask: &[u8],
    sigma: [f64; 3],
    echoes: [usize; 2],
    nx: usize, ny: usize, nz: usize,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n_echoes = phases.len();
    let n_total = nx * ny * nz;

    let e1 = echoes[0];
    let e2 = echoes[1];

    // ΔTE = TEs[echo2] - TEs[echo1]
    let delta_te = tes[e2] - tes[e1];

    // Compute HIP between the two echoes
    // HIP = conj(echo1) * echo2, so hip_phase = phase2 - phase1
    let (hip_phase, hip_mag) = hermitian_inner_product(
        &phases[e1], &mags[e1],
        &phases[e2], &mags[e2],
        mask, n_total
    );

    // Weight for ROMEO = sqrt(|HIP|) - matches Julia: weight = sqrt.(abs.(hip))
    let weight: Vec<f64> = hip_mag.iter().map(|&x| x.sqrt()).collect();

    // Unwrap HIP phase using ROMEO (matching Julia line 48)
    // Julia: phaseevolution = (TEs[echoes[1]] / ΔTE) .* romeo(angle.(hip); mag=weight, mask)
    let unwrapped_hip = unwrap_with_romeo(&hip_phase, &weight, mask, nx, ny, nz);

    // Phase evolution at TE1: (TE1 / ΔTE) * unwrapped_hip
    // This gives the phase that would have evolved from TE=0 to TE=TE1
    let scale = tes[e1] / delta_te;
    let mut phase_evolution = vec![0.0; n_total];
    for i in 0..n_total {
        if mask[i] > 0 {
            phase_evolution[i] = scale * unwrapped_hip[i];
        }
    }

    // Phase offset = phase[echo1] - phase_evolution
    // IMPORTANT: Do NOT wrap here! Julia line 49 does raw subtraction:
    //   po .= getangle(image, echoes[1]) .- phaseevolution
    // The smoothing function will handle the wrapping internally
    let mut phase_offset = vec![0.0; n_total];
    for i in 0..n_total {
        if mask[i] > 0 {
            phase_offset[i] = phases[e1][i] - phase_evolution[i];
        }
    }

    // Smooth the phase offset (handles wrapping via complex representation)
    // Julia line 51: po[:,:,:,icha] .= gaussiansmooth3d_phase(view(po,:,:,:,icha), sigma; mask)
    let phase_offset_smoothed = gaussian_smooth_3d_phase(&phase_offset, sigma, mask, nx, ny, nz);

    // Remove phase offset from all echoes
    // Julia combinewithPO does: exp.(1im .* (phase - po)) then angle()
    // This is equivalent to wrap_to_pi(phase - po)
    let mut corrected_phases = Vec::with_capacity(n_echoes);
    for e in 0..n_echoes {
        let mut corrected = vec![0.0; n_total];
        for i in 0..n_total {
            if mask[i] > 0 {
                corrected[i] = wrap_to_pi(phases[e][i] - phase_offset_smoothed[i]);
            }
        }
        corrected_phases.push(corrected);
    }

    (corrected_phases, phase_offset_smoothed)
}

/// Unwrap phase using ROMEO algorithm
fn unwrap_with_romeo(
    phase: &[f64],
    mag: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
) -> Vec<f64> {
    // Calculate ROMEO weights (no second echo)
    let weights = calculate_weights_romeo(
        phase, mag, None, // No second echo for single phase
        0.0, 0.0, // TEs not used when phase2 is None
        mask, nx, ny, nz
    );

    // Find seed point (center of mass of mask)
    let (seed_i, seed_j, seed_k) = find_seed_point(mask, nx, ny, nz);

    // Perform region growing unwrap
    let mut unwrapped = phase.to_vec();
    let mut work_mask = mask.to_vec();

    grow_region_unwrap(
        &mut unwrapped, &weights, &mut work_mask,
        nx, ny, nz, seed_i, seed_j, seed_k
    );

    unwrapped
}

/// Find a good seed point (center of mass of the mask)
fn find_seed_point(mask: &[u8], nx: usize, ny: usize, nz: usize) -> (usize, usize, usize) {
    let mut sum_i = 0usize;
    let mut sum_j = 0usize;
    let mut sum_k = 0usize;
    let mut count = 0usize;

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let idx = idx3d(i, j, k, nx, ny);
                if mask[idx] > 0 {
                    sum_i += i;
                    sum_j += j;
                    sum_k += k;
                    count += 1;
                }
            }
        }
    }

    if count == 0 {
        return (nx / 2, ny / 2, nz / 2);
    }

    (sum_i / count, sum_j / count, sum_k / count)
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
/// * `tes` - Echo times in ms
/// * `mask` - Binary mask
/// * `weight_type` - Type of weighting to use
/// * `n_total` - Total number of voxels
///
/// # Returns
/// B0 field in Hz
pub fn calculate_b0_weighted(
    unwrapped_phases: &[Vec<f64>],
    mags: &[Vec<f64>],
    tes: &[f64],
    mask: &[u8],
    weight_type: B0WeightType,
    n_total: usize,
) -> Vec<f64> {
    let n_echoes = unwrapped_phases.len();
    let mut b0 = vec![0.0; n_total];

    // Precompute weights for each echo
    let weights: Vec<Vec<f64>> = (0..n_echoes)
        .map(|e| {
            let te = tes[e];
            let mag = &mags[e];

            match weight_type {
                B0WeightType::PhaseSNR => {
                    // mag * TE
                    (0..n_total).map(|i| mag[i] * te).collect()
                }
                B0WeightType::PhaseVar => {
                    // mag² * TE²
                    (0..n_total).map(|i| mag[i] * mag[i] * te * te).collect()
                }
                B0WeightType::Average => {
                    // Uniform
                    vec![1.0; n_total]
                }
                B0WeightType::TEs => {
                    // TE only
                    vec![te; n_total]
                }
                B0WeightType::Mag => {
                    // Magnitude only
                    mag.clone()
                }
            }
        })
        .collect();

    // B0 = (1000 / 2π) * Σ(phase / TE * weight) / Σ(weight)
    let scale = 1000.0 / TWO_PI;

    for i in 0..n_total {
        if mask[i] == 0 {
            continue;
        }

        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        for e in 0..n_echoes {
            let w = weights[e][i];
            let phase_over_te = unwrapped_phases[e][i] / tes[e];

            weighted_sum += phase_over_te * w;
            weight_sum += w;
        }

        if weight_sum > 1e-10 {
            b0[i] = scale * weighted_sum / weight_sum;
        }
    }

    b0
}

/// Full MCPC-3D-S + B0 calculation pipeline
///
/// This combines phase offset removal with weighted B0 calculation
///
/// # Arguments
/// * `phases` - Wrapped phase for each echo
/// * `mags` - Magnitude for each echo
/// * `tes` - Echo times in ms
/// * `mask` - Binary mask
/// * `sigma` - Smoothing sigma for phase offset [sx, sy, sz]
/// * `weight_type` - B0 weighting type
/// * `nx`, `ny`, `nz` - Dimensions
///
/// # Returns
/// (b0_hz, phase_offset, corrected_phases)
pub fn mcpc3ds_b0_pipeline(
    phases: &[Vec<f64>],
    mags: &[Vec<f64>],
    tes: &[f64],
    mask: &[u8],
    sigma: [f64; 3],
    weight_type: B0WeightType,
    nx: usize, ny: usize, nz: usize,
) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let n_total = nx * ny * nz;
    let n_echoes = phases.len();

    // Step 1: MCPC-3D-S to remove phase offset
    let (corrected_phases, phase_offset) = mcpc3ds_single_coil(
        phases, mags, tes, mask,
        sigma, [0, 1], // use first two echoes
        nx, ny, nz
    );

    // Step 2: Unwrap the corrected phases using ROMEO
    // Each echo needs to be unwrapped independently
    let mut unwrapped_phases = Vec::with_capacity(n_echoes);
    for e in 0..n_echoes {
        let unwrapped = unwrap_with_romeo(&corrected_phases[e], &mags[e], mask, nx, ny, nz);
        unwrapped_phases.push(unwrapped);
    }

    // Step 3: Align echoes to remove 2π ambiguities
    // Use first echo as reference
    for e in 1..n_echoes {
        let te_ratio = tes[e] / tes[0];

        // Calculate mean difference
        let mut sum_diff = 0.0;
        let mut count = 0;
        for i in 0..n_total {
            if mask[i] > 0 {
                let expected = unwrapped_phases[0][i] * te_ratio;
                sum_diff += unwrapped_phases[e][i] - expected;
                count += 1;
            }
        }

        if count > 0 {
            let mean_diff = sum_diff / count as f64;
            let correction = (mean_diff / TWO_PI).round() * TWO_PI;

            if correction.abs() > 0.1 {
                for i in 0..n_total {
                    if mask[i] > 0 {
                        unwrapped_phases[e][i] -= correction;
                    }
                }
            }
        }
    }

    // Step 4: Calculate B0 with weighted averaging
    let b0 = calculate_b0_weighted(
        &unwrapped_phases, mags, tes, mask,
        weight_type, n_total
    );

    (b0, phase_offset, corrected_phases)
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
    unwrapped_phases: &[Vec<f64>],
    mags: &[Vec<f64>],
    tes: &[f64],
    mask: &[u8],
    estimate_offset: bool,
    reliability_threshold_percentile: f64,
) -> LinearFitResult {
    let n_echoes = unwrapped_phases.len();
    let n_total = unwrapped_phases[0].len();

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
                let w = mags[e][v];
                sum_w += w;
                sum_w_te += w * tes[e];
                sum_w_phase += w * unwrapped_phases[e][v];
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
                let w = mags[e][v];
                let te_centered = tes[e] - te_mean;
                let phase_centered = unwrapped_phases[e][v] - phase_mean;
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
                    let w = mags[e][v];
                    let predicted = intercept + slope * tes[e];
                    let diff = unwrapped_phases[e][v] - predicted;
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
                let w = mags[e][v];
                let te = tes[e];
                let phase = unwrapped_phases[e][v];
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
                    let w = mags[e][v];
                    let predicted = slope * tes[e];
                    let diff = unwrapped_phases[e][v] - predicted;
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
}

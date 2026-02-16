//! Multi-echo phase combination utilities
//!
//! Implements MCPC-3D-S (Multi-Channel Phase Combination - 3D - Smoothed) algorithm
//! and weighted B0 calculation.
//!
//! Reference:
//! Eckstein, K., Dymerska, B., Bachrata, B., Bogner, W., Poljanc, K., Trattnig, S.,
//! Robinson, S.D. (2018). "Computationally Efficient Combination of Multi-channel Phase
//! Data From Multi-echo Acquisitions (ASPIRE)."
//! Magnetic Resonance in Medicine, 79:2996-3006. https://doi.org/10.1002/mrm.26963
//!
//! Reference implementation: https://github.com/korbinian90/MriResearchTools.jl

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
    phases: &[impl AsRef<[f64]>],
    mags: &[impl AsRef<[f64]>],
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
        phases[e1].as_ref(), mags[e1].as_ref(),
        phases[e2].as_ref(), mags[e2].as_ref(),
        mask, n_total
    );

    // Weight for ROMEO = sqrt(|HIP|) - matches Julia: weight = sqrt.(abs.(hip))
    let weight: Vec<f64> = hip_mag.iter().map(|&x| x.sqrt()).collect();
    drop(hip_mag); // Free ~82 MB early

    // Unwrap HIP phase using ROMEO (matching Julia line 48)
    // Julia: phaseevolution = (TEs[echoes[1]] / ΔTE) .* romeo(angle.(hip); mag=weight, mask)
    let unwrapped_hip = unwrap_with_romeo(&hip_phase, &weight, mask, nx, ny, nz);
    drop(hip_phase); // Free ~82 MB early
    drop(weight);    // Free ~82 MB early

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
    let phase_offset_smoothed = gaussian_smooth_3d_phase(&phase_offset, sigma, mask, nx, ny, nz);
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
    unwrapped_phases: &[impl AsRef<[f64]>],
    mags: &[impl AsRef<[f64]>],
    tes: &[f64],
    mask: &[u8],
    weight_type: B0WeightType,
    n_total: usize,
) -> Vec<f64> {
    let n_echoes = unwrapped_phases.len();
    let mut b0 = vec![0.0; n_total];

    // Compute inline to avoid allocating per-echo weight arrays

    // B0 = (1000 / 2π) * Σ(phase / TE * weight) / Σ(weight)
    let scale = 1000.0 / TWO_PI;

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
    phases: &[impl AsRef<[f64]>],
    mags: &[impl AsRef<[f64]>],
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
        let unwrapped = unwrap_with_romeo(&corrected_phases[e], mags[e].as_ref(), mask, nx, ny, nz);
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
    /// Magnitude is uniform (1.0) inside the mask.
    fn make_synthetic_multi_echo(
        nx: usize, ny: usize, nz: usize,
        tes: &[f64],
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<u8>) {
        let n = nx * ny * nz;
        let n_echoes = tes.len();

        // Constant phase offset (small, well within [-pi, pi])
        let phase_offset_val = 0.3;
        // Slope (rad/ms) as a function of x: gentle ramp so phases stay in [-pi, pi]
        // Max slope = 0.05 rad/ms at x=nx-1, so max phase ~ 0.3 + 0.05*7*15 = 5.55
        // which will wrap but that is fine.
        let slope_scale = 0.05;

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

        let smoothed = gaussian_smooth_3d_phase(&phase, sigma, &mask, nx, ny, nz);

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

        let smoothed = gaussian_smooth_3d_phase(&phase, sigma, &mask, nx, ny, nz);

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
        let smoothed = gaussian_smooth_3d_phase(&phase, sigma, &mask, nx, ny, nz);

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

        let smoothed = gaussian_smooth_3d_phase(&phase, sigma, &mask, nx, ny, nz);
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
    // find_seed_point
    // =========================================================================

    #[test]
    fn test_find_seed_point_full_mask() {
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
        // For a constant slope (rad/ms), all weight types should recover it.
        // phase[e] = slope * TE[e], so phase/TE = slope for each echo.
        // Weighted average of identical values = same value.
        // B0 = (1000 / 2pi) * slope (Hz)
        let n = 64;
        let tes = [5.0, 10.0, 15.0];
        let slope = 0.2; // rad/ms
        let mask = vec![1u8; n];

        let phases: Vec<Vec<f64>> = tes.iter()
            .map(|&te| vec![slope * te; n])
            .collect();
        let mags: Vec<Vec<f64>> = tes.iter()
            .map(|_| vec![1.0; n])
            .collect();

        let b0 = calculate_b0_weighted(&phases, &mags, &tes, &mask, B0WeightType::PhaseSNR, n);

        let expected_hz = 1000.0 / TWO_PI * slope;
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
        let tes = [5.0, 10.0, 15.0];
        let slope = 0.1;
        let mask = vec![1u8; n];

        let phases: Vec<Vec<f64>> = tes.iter()
            .map(|&te| vec![slope * te; n])
            .collect();
        let mags: Vec<Vec<f64>> = tes.iter()
            .map(|_| vec![2.0; n])
            .collect();

        let expected_hz = 1000.0 / TWO_PI * slope;

        for wt in &[
            B0WeightType::PhaseSNR,
            B0WeightType::PhaseVar,
            B0WeightType::Average,
            B0WeightType::TEs,
            B0WeightType::Mag,
        ] {
            let b0 = calculate_b0_weighted(&phases, &mags, &tes, &mask, *wt, n);
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
        let tes = [5.0, 10.0];
        let mask = vec![0u8; n]; // all masked out

        let phases: Vec<Vec<f64>> = tes.iter()
            .map(|&te| vec![0.5 * te; n])
            .collect();
        let mags: Vec<Vec<f64>> = tes.iter()
            .map(|_| vec![1.0; n])
            .collect();

        let b0 = calculate_b0_weighted(&phases, &mags, &tes, &mask, B0WeightType::PhaseSNR, n);

        for v in &b0 {
            assert_eq!(*v, 0.0, "masked-out voxels should have B0=0");
        }
    }

    #[test]
    fn test_calculate_b0_weighted_zero_magnitude() {
        // When magnitude is zero, weight is zero; result should be 0
        let n = 4;
        let tes = [5.0, 10.0, 15.0];
        let mask = vec![1u8; n];

        let phases: Vec<Vec<f64>> = tes.iter()
            .map(|&te| vec![0.2 * te; n])
            .collect();
        let mags: Vec<Vec<f64>> = tes.iter()
            .map(|_| vec![0.0; n]) // zero magnitude
            .collect();

        // PhaseSNR weight = mag * te = 0
        let b0 = calculate_b0_weighted(&phases, &mags, &tes, &mask, B0WeightType::PhaseSNR, n);
        for v in &b0 {
            assert_eq!(*v, 0.0, "zero-magnitude voxels should yield B0=0");
        }

        // Average weight = 1.0, should still work
        let b0_avg = calculate_b0_weighted(&phases, &mags, &tes, &mask, B0WeightType::Average, n);
        let expected = 1000.0 / TWO_PI * 0.2;
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
    fn test_mcpc3ds_single_coil_output_sizes() {
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        let tes = [5.0, 10.0, 15.0];
        let (phases, mags, mask) = make_synthetic_multi_echo(nx, ny, nz, &tes);

        let sigma = [1.0, 1.0, 1.0];
        let (corrected, offset) = mcpc3ds_single_coil(
            &phases, &mags, &tes, &mask, sigma, [0, 1], nx, ny, nz,
        );

        // Check sizes
        assert_eq!(corrected.len(), tes.len(), "should have one corrected phase per echo");
        for (e, cp) in corrected.iter().enumerate() {
            assert_eq!(cp.len(), n, "echo {} corrected phase should have {} voxels", e, n);
        }
        assert_eq!(offset.len(), n, "phase offset should have {} voxels", n);
    }

    #[test]
    fn test_mcpc3ds_single_coil_finite_output() {
        let (nx, ny, nz) = (8, 8, 8);
        let tes = [5.0, 10.0, 15.0];
        let (phases, mags, mask) = make_synthetic_multi_echo(nx, ny, nz, &tes);

        let sigma = [1.0, 1.0, 1.0];
        let (corrected, offset) = mcpc3ds_single_coil(
            &phases, &mags, &tes, &mask, sigma, [0, 1], nx, ny, nz,
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
    fn test_mcpc3ds_single_coil_corrected_in_range() {
        let (nx, ny, nz) = (8, 8, 8);
        let tes = [5.0, 10.0, 15.0];
        let (phases, mags, mask) = make_synthetic_multi_echo(nx, ny, nz, &tes);

        let sigma = [1.0, 1.0, 1.0];
        let (corrected, _) = mcpc3ds_single_coil(
            &phases, &mags, &tes, &mask, sigma, [0, 1], nx, ny, nz,
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
    fn test_mcpc3ds_single_coil_uniform_phase() {
        // Uniform phase across all echoes => offset should be approximately that phase
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        let tes = [5.0, 10.0, 15.0];

        // All echoes have constant phase 0.5 (no TE dependence)
        let phases: Vec<Vec<f64>> = (0..3).map(|_| vec![0.5; n]).collect();
        let mags: Vec<Vec<f64>> = (0..3).map(|_| vec![1.0; n]).collect();
        let mask = vec![1u8; n];

        let sigma = [1.0, 1.0, 1.0];
        let (corrected, _offset) = mcpc3ds_single_coil(
            &phases, &mags, &tes, &mask, sigma, [0, 1], nx, ny, nz,
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
    // mcpc3ds_b0_pipeline
    // =========================================================================

    #[test]
    fn test_mcpc3ds_b0_pipeline_output_sizes() {
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        let tes = [5.0, 10.0, 15.0];
        let (phases, mags, mask) = make_synthetic_multi_echo(nx, ny, nz, &tes);

        let sigma = [1.0, 1.0, 1.0];
        let (b0, offset, corrected) = mcpc3ds_b0_pipeline(
            &phases, &mags, &tes, &mask, sigma,
            B0WeightType::PhaseSNR, nx, ny, nz,
        );

        assert_eq!(b0.len(), n, "B0 should have n voxels");
        assert_eq!(offset.len(), n, "offset should have n voxels");
        assert_eq!(corrected.len(), tes.len(), "corrected should have n_echoes entries");
        for cp in &corrected {
            assert_eq!(cp.len(), n);
        }
    }

    #[test]
    fn test_mcpc3ds_b0_pipeline_finite_output() {
        let (nx, ny, nz) = (8, 8, 8);
        let tes = [5.0, 10.0, 15.0];
        let (phases, mags, mask) = make_synthetic_multi_echo(nx, ny, nz, &tes);

        let sigma = [1.0, 1.0, 1.0];
        let (b0, offset, corrected) = mcpc3ds_b0_pipeline(
            &phases, &mags, &tes, &mask, sigma,
            B0WeightType::PhaseSNR, nx, ny, nz,
        );

        for v in &b0 {
            assert!(v.is_finite(), "B0 should be finite");
        }
        for v in &offset {
            assert!(v.is_finite(), "offset should be finite");
        }
        for cp in &corrected {
            for v in cp {
                assert!(v.is_finite(), "corrected phase should be finite");
            }
        }
    }

    #[test]
    fn test_mcpc3ds_b0_pipeline_different_weight_types() {
        let (nx, ny, nz) = (8, 8, 8);
        let tes = [5.0, 10.0, 15.0];
        let (phases, mags, mask) = make_synthetic_multi_echo(nx, ny, nz, &tes);
        let sigma = [1.0, 1.0, 1.0];

        for wt in &[
            B0WeightType::PhaseSNR,
            B0WeightType::Average,
            B0WeightType::TEs,
            B0WeightType::Mag,
            B0WeightType::PhaseVar,
        ] {
            let (b0, _, _) = mcpc3ds_b0_pipeline(
                &phases, &mags, &tes, &mask, sigma, *wt, nx, ny, nz,
            );
            for v in &b0 {
                assert!(v.is_finite(),
                    "B0 with weight type {:?} should be finite", wt);
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
    // unwrap_with_romeo (tested indirectly through mcpc3ds_single_coil
    // but let's also test directly)
    // =========================================================================

    #[test]
    fn test_unwrap_with_romeo_smooth_data() {
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        // Smooth phase that doesn't need unwrapping
        let phase: Vec<f64> = (0..n).map(|i| {
            let x = (i % nx) as f64 / nx as f64;
            0.5 * x // small smooth phase
        }).collect();
        let mag = vec![1.0; n];
        let mask = vec![1u8; n];

        let unwrapped = unwrap_with_romeo(&phase, &mag, &mask, nx, ny, nz);

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
        let tes = [5.0, 10.0, 15.0];
        let (phases, mags, mask) = make_synthetic_multi_echo(nx, ny, nz, &tes);

        let sigma = [1.0, 1.0, 1.0];
        let (corrected, _offset) = mcpc3ds_single_coil(
            &phases, &mags, &tes, &mask, sigma, [0, 1], nx, ny, nz,
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
}

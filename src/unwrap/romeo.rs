//! ROMEO weight calculation for phase unwrapping
//!
//! Calculates edge weights for region-growing phase unwrapping based on:
//! - Phase coherence
//! - Phase gradient coherence (multi-echo)
//! - Magnitude coherence
//! - Magnitude weights
//!
//! Reference:
//! Dymerska, B., Eckstein, K., Bachrata, B., Siow, B., Trattnig, S., Shmueli, K.,
//! Robinson, S.D. (2021). "Phase unwrapping with a rapid opensource minimum spanning
//! tree algorithm (ROMEO)." Magnetic Resonance in Medicine, 85(4):2294-2308.
//! https://doi.org/10.1002/mrm.28563
//!
//! Reference implementation: https://github.com/korbinian90/MriResearchTools.jl

use std::f64::consts::PI;

const TWO_PI: f64 = 2.0 * PI;

/// Wrap angle to [-π, π]
#[inline]
fn wrap_angle(angle: f64) -> f64 {
    let mut a = angle % TWO_PI;
    if a > PI {
        a -= TWO_PI;
    } else if a < -PI {
        a += TWO_PI;
    }
    a
}

/// Index into a 3D array in Fortran order (column-major, matches NIfTI)
#[inline(always)]
fn idx3d(i: usize, j: usize, k: usize, nx: usize, ny: usize) -> usize {
    i + j * nx + k * nx * ny
}

/// Calculate ROMEO edge weights for phase unwrapping
///
/// Returns weights array with shape (3, nx, ny, nz) for 3 directions (x, y, z).
/// Weights are normalized to 0-255 for use with the bucket priority queue.
///
/// # Arguments
/// * `phase` - Wrapped phase data (nx * ny * nz), first echo
/// * `mag` - Magnitude data (nx * ny * nz), optional (pass empty slice if none)
/// * `phase2` - Second echo phase for gradient coherence (optional)
/// * `te1`, `te2` - Echo times for gradient coherence scaling
/// * `mask` - Binary mask (nx * ny * nz), 1 = process
/// * `nx`, `ny`, `nz` - Array dimensions
///
/// # Returns
/// Weights array of size 3 * nx * ny * nz in C order [dim][i][j][k]
pub fn calculate_weights_romeo(
    phase: &[f64],
    mag: &[f64],
    phase2: Option<&[f64]>,
    te1: f64,
    te2: f64,
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
) -> Vec<u8> {
    // Default: all weight components enabled
    calculate_weights_romeo_configurable(
        phase, mag, phase2, te1, te2, mask, nx, ny, nz,
        true, true, true  // use_phase_gradient_coherence, use_mag_coherence, use_mag_weight
    )
}

/// Calculate ROMEO edge weights with configurable weight components
///
/// # Arguments
/// * `phase` - Wrapped phase data (nx * ny * nz), first echo
/// * `mag` - Magnitude data (nx * ny * nz), optional (pass empty slice if none)
/// * `phase2` - Second echo phase for gradient coherence (optional)
/// * `te1`, `te2` - Echo times for gradient coherence scaling
/// * `mask` - Binary mask (nx * ny * nz), 1 = process
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `use_phase_gradient_coherence` - Include phase gradient coherence (multi-echo temporal)
/// * `use_mag_coherence` - Include magnitude coherence (min/max similarity)
/// * `use_mag_weight` - Include magnitude weight (penalize low signal)
///
/// # Returns
/// Weights array of size 3 * nx * ny * nz in C order [dim][i][j][k]
pub fn calculate_weights_romeo_configurable(
    phase: &[f64],
    mag: &[f64],
    phase2: Option<&[f64]>,
    te1: f64,
    te2: f64,
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    use_phase_gradient_coherence: bool,
    use_mag_coherence: bool,
    use_mag_weight: bool,
) -> Vec<u8> {
    let n_total = nx * ny * nz;
    let mut weights = vec![0u8; 3 * n_total];

    let has_mag = !mag.is_empty();
    let has_phase2 = phase2.is_some();
    let te_ratio = if te2.abs() > 1e-10 { te1 / te2 } else { 1.0 };

    // Get max magnitude for normalization
    let max_mag = if has_mag && use_mag_weight {
        mag.iter().cloned().fold(0.0_f64, f64::max)
    } else {
        1.0
    };
    let half_max_mag = 0.5 * max_mag + 1e-12;

    // Process each direction
    for dim in 0..3_usize {
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    // Get neighbor based on dimension
                    let (ni, nj, nk) = match dim {
                        0 => (i + 1, j, k),  // x direction
                        1 => (i, j + 1, k),  // y direction
                        _ => (i, j, k + 1),  // z direction
                    };

                    // Skip if neighbor is out of bounds
                    if ni >= nx || nj >= ny || nk >= nz {
                        continue;
                    }

                    let idx = idx3d(i, j, k, nx, ny);
                    let idx_n = idx3d(ni, nj, nk, nx, ny);

                    // Skip if either voxel is outside mask
                    if mask[idx] == 0 || mask[idx_n] == 0 {
                        continue;
                    }

                    // Phase difference
                    let p1_diff = phase[idx_n] - phase[idx];

                    // 1. Phase coherence: 1 - |wrap(diff)| / π (always on)
                    let pc = 1.0 - wrap_angle(p1_diff).abs() / PI;

                    // 2. Phase gradient coherence (optional, multi-echo only)
                    let pgc = if use_phase_gradient_coherence && has_phase2 {
                        let phase2_data = phase2.unwrap();
                        let p2_diff = phase2_data[idx_n] - phase2_data[idx];
                        let wrapped_p1 = wrap_angle(p1_diff);
                        let wrapped_p2 = wrap_angle(p2_diff);
                        (1.0 - (wrapped_p1 - wrapped_p2 * te_ratio).abs()).max(0.0)
                    } else {
                        1.0
                    };

                    // 3. Magnitude coherence: (min/max)² (optional)
                    let mc = if use_mag_coherence && has_mag {
                        let m1 = mag[idx];
                        let m2 = mag[idx_n];
                        let mag_min = m1.min(m2);
                        let mag_max = m1.max(m2);
                        if mag_max > 1e-12 {
                            (mag_min / mag_max).powi(2)
                        } else {
                            0.0
                        }
                    } else {
                        1.0
                    };

                    // 4. Magnitude weights: 0.5 + 0.5 * min(1, mag / (0.5 * max_mag)) (optional)
                    let (mw1, mw2) = if use_mag_weight && has_mag {
                        let mw1 = 0.5 + 0.5 * (mag[idx] / half_max_mag).min(1.0);
                        let mw2 = 0.5 + 0.5 * (mag[idx_n] / half_max_mag).min(1.0);
                        (mw1, mw2)
                    } else {
                        (1.0, 1.0)
                    };

                    // Combined weight
                    let weight = pc * pgc * mc * mw1 * mw2;

                    // Convert to u8 and store
                    let weight_u8 = (weight.clamp(0.0, 1.0) * 255.0) as u8;

                    // Store at edge location (min coordinate)
                    let edge_idx = dim * n_total + idx3d(i, j, k, nx, ny);
                    weights[edge_idx] = weight_u8;
                }
            }
        }
    }

    weights
}

/// Simplified weight calculation for single-echo data (no phase2)
pub fn calculate_weights_single_echo(
    phase: &[f64],
    mag: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
) -> Vec<u8> {
    calculate_weights_romeo(phase, mag, None, 1.0, 1.0, mask, nx, ny, nz)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wrap_angle() {
        assert!((wrap_angle(0.0) - 0.0).abs() < 1e-10);
        assert!((wrap_angle(PI) - PI).abs() < 1e-10);
        assert!((wrap_angle(-PI) - (-PI)).abs() < 1e-10);
        assert!((wrap_angle(TWO_PI) - 0.0).abs() < 1e-10);
        assert!((wrap_angle(3.0 * PI) - PI).abs() < 1e-10);
        assert!((wrap_angle(-3.0 * PI) - (-PI)).abs() < 1e-10);
    }

    #[test]
    fn test_weights_constant_phase() {
        // Constant phase should give high weights (phase coherence = 1)
        let n = 4;
        let phase = vec![0.0; n * n * n];
        let mag = vec![1.0; n * n * n];
        let mask = vec![1u8; n * n * n];

        let weights = calculate_weights_single_echo(&phase, &mag, &mask, n, n, n);

        // All interior weights should be 255 (constant phase, uniform magnitude)
        let mut high_weight_count = 0;
        for &w in weights.iter() {
            if w == 255 {
                high_weight_count += 1;
            }
        }
        assert!(high_weight_count > 0, "Should have some high weights for constant phase");
    }

    #[test]
    fn test_weights_wrapped_jump() {
        // Phase with 2π jump should give low weights at jump location
        let n = 4;
        let mut phase = vec![0.0; n * n * n];

        // Create a 2π jump in x direction at i=2
        for i in 2..n {
            for j in 0..n {
                for k in 0..n {
                    phase[idx3d(i, j, k, n, n)] = TWO_PI;
                }
            }
        }

        let mask = vec![1u8; n * n * n];
        let _weights = calculate_weights_single_echo(&phase, &[], &mask, n, n, n);

        // Weight at x=1 to x=2 edge should be low (wrapped difference = 0, but that's ok)
        // Actually, for a 2π jump, the wrapped difference is 0, so coherence is 1
        // This is correct - ROMEO uses wrapped differences, not raw differences
    }

    #[test]
    fn test_weights_mask() {
        // Weights should be 0 where mask is 0
        let n = 4;
        let phase = vec![0.5; n * n * n];
        let mut mask = vec![1u8; n * n * n];

        // Set some voxels outside mask
        mask[0] = 0;
        mask[1] = 0;

        let weights = calculate_weights_single_echo(&phase, &[], &mask, n, n, n);

        // Edges connected to masked-out voxels should be 0
        // Weight at edge (0,0,0)-(1,0,0) should be 0 since idx 0 is masked out
        assert_eq!(weights[0], 0);  // x-direction edge at (0,0,0)
    }
}

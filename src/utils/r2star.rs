//! R2* mapping from multi-echo magnitude data
//!
//! Provides R2* estimation using the ARLO (Auto-Regression on Linear Operations)
//! algorithm for equi-spaced echo times, with a log-linear fallback for edge cases.
//!
//! Reference:
//! Pei, M., et al. (2015). "Algorithm for fast monoexponential fitting based on
//! Auto-Regression on Linear Operations (ARLO) of data."
//! Magnetic Resonance in Medicine, 73(2):843-850.

/// Check if echo times are approximately equi-spaced (suitable for ARLO).
///
/// Requires at least 3 echo times with spacing deviations below `tolerance`.
///
/// # Arguments
/// * `echo_times` - Echo times in seconds
/// * `tolerance` - Maximum allowed deviation from uniform spacing (seconds, e.g. 0.1e-3)
pub fn use_arlo(echo_times: &[f64], tolerance: f64) -> bool {
    if echo_times.len() < 3 {
        return false;
    }

    let mut te_sorted: Vec<f64> = echo_times.to_vec();
    te_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let diffs: Vec<f64> = te_sorted.windows(2).map(|w| w[1] - w[0]).collect();
    let mean_diff: f64 = diffs.iter().sum::<f64>() / diffs.len() as f64;
    let max_dev = diffs.iter().map(|&d| (d - mean_diff).abs()).fold(0.0_f64, f64::max);

    max_dev <= tolerance
}

/// R2* mapping using ARLO for equi-spaced echo times (≥3 echoes).
///
/// For each masked voxel, computes consecutive signal ratios to estimate the
/// mono-exponential decay rate. Falls back to log-linear fitting for voxels
/// with unreliable ratio estimates.
///
/// # Arguments
/// * `magnitude` - Multi-echo magnitude data, flattened as `[voxel0_echo0, voxel0_echo1, ..., voxel1_echo0, ...]`
///   i.e. shape `(n_voxels, n_echoes)` in row-major order, where `n_voxels = nx*ny*nz`
/// * `mask` - Binary brain mask `[nx*ny*nz]` (1 = process, 0 = skip)
/// * `echo_times` - Echo times in seconds `[n_echoes]` (must be equi-spaced)
/// * `nx`, `ny`, `nz` - Volume dimensions
///
/// # Returns
/// `(r2star_map, s0_map)` - R2* in Hz and S0 (proton density), both `[nx*ny*nz]`
///
/// # Panics
/// Panics if `echo_times.len() < 3` or echo times are not equi-spaced.
#[allow(clippy::too_many_arguments)]
pub fn r2star_arlo(
    magnitude: &[f64],
    mask: &[u8],
    echo_times: &[f64],
    nx: usize, ny: usize, nz: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n_echoes = echo_times.len();
    assert!(n_echoes >= 3, "ARLO requires at least 3 echo times");
    let n_voxels = nx * ny * nz;
    assert_eq!(magnitude.len(), n_voxels * n_echoes,
        "magnitude length must be n_voxels * n_echoes");
    assert_eq!(mask.len(), n_voxels, "mask length must be n_voxels");

    // Sort echo times and get the uniform spacing
    let mut te_sorted: Vec<f64> = echo_times.to_vec();
    let sort_indices: Vec<usize> = {
        let mut idx: Vec<usize> = (0..n_echoes).collect();
        idx.sort_by(|&a, &b| echo_times[a].partial_cmp(&echo_times[b]).unwrap());
        idx
    };
    te_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let delta_te = te_sorted[1] - te_sorted[0];

    let mut r2star_map = vec![0.0_f64; n_voxels];
    let mut s0_map = vec![0.0_f64; n_voxels];

    for v in 0..n_voxels {
        if mask[v] == 0 {
            continue;
        }

        // Extract signal for this voxel, sorted by echo time
        let signal: Vec<f64> = sort_indices.iter()
            .map(|&ei| magnitude[v * n_echoes + ei])
            .collect();

        // Skip if signal is too small
        let sig_max = signal.iter().cloned().fold(0.0_f64, f64::max);
        if sig_max < 1e-10 {
            continue;
        }

        // ARLO: compute consecutive signal ratios
        let mut alphas = Vec::new();
        for i in 0..(n_echoes - 1) {
            if signal[i] > 1e-10 {
                let alpha = signal[i + 1] / signal[i];
                if alpha > 0.0 && alpha <= 1.0 {
                    alphas.push(alpha);
                }
            }
        }

        if alphas.len() >= 2 {
            let alpha_mean: f64 = alphas.iter().sum::<f64>() / alphas.len() as f64;
            let r2star_val = -alpha_mean.ln() / delta_te;

            if r2star_val >= 0.0 && r2star_val <= 500.0 {
                r2star_map[v] = r2star_val;
                s0_map[v] = signal[0] * (r2star_val * te_sorted[0]).exp();
                continue;
            }
        }

        // Fallback: log-linear fit
        let (r2, s0) = log_linear_fit(&signal, &te_sorted);
        r2star_map[v] = r2.max(0.0);
        s0_map[v] = s0.max(0.0);
    }

    (r2star_map, s0_map)
}

/// Log-linear R2* fit: log(S) = log(S0) - R2* * TE
/// Solves via normal equations.
fn log_linear_fit(signal: &[f64], echo_times: &[f64]) -> (f64, f64) {
    let n = signal.len();
    let mut sum_t = 0.0;
    let mut sum_tt = 0.0;
    let mut sum_y = 0.0;
    let mut sum_ty = 0.0;
    let mut count = 0;

    for i in 0..n {
        if signal[i] > 1e-10 {
            let y = signal[i].ln();
            let t = echo_times[i];
            sum_t += t;
            sum_tt += t * t;
            sum_y += y;
            sum_ty += t * y;
            count += 1;
        }
    }

    if count < 2 {
        return (0.0, 0.0);
    }

    let n_f = count as f64;
    let denom = n_f * sum_tt - sum_t * sum_t;
    if denom.abs() < 1e-15 {
        return (0.0, 0.0);
    }

    let slope = (n_f * sum_ty - sum_t * sum_y) / denom;
    let intercept = (sum_y - slope * sum_t) / n_f;

    let r2star = -slope;  // S = S0 * exp(-R2* * TE) → log(S) = log(S0) - R2* * TE
    let s0 = intercept.exp();

    (r2star, s0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_use_arlo_equispaced() {
        let te = vec![0.005, 0.010, 0.015, 0.020];
        assert!(use_arlo(&te, 0.1e-3));
    }

    #[test]
    fn test_use_arlo_not_equispaced() {
        let te = vec![0.005, 0.008, 0.015];
        assert!(!use_arlo(&te, 0.1e-3));
    }

    #[test]
    fn test_use_arlo_too_few() {
        let te = vec![0.005, 0.010];
        assert!(!use_arlo(&te, 0.1e-3));
    }

    #[test]
    fn test_r2star_arlo_synthetic() {
        // Synthetic mono-exponential decay: S(TE) = S0 * exp(-R2* * TE)
        let r2star_true: f64 = 30.0; // Hz
        let s0_true: f64 = 1000.0;
        let te = vec![0.005, 0.010, 0.015, 0.020, 0.025];
        let n_echoes = te.len();

        // Single voxel in a 1x1x1 volume
        let mask = vec![1u8];
        let magnitude: Vec<f64> = te.iter()
            .map(|&t| s0_true * (-r2star_true * t).exp())
            .collect();

        let (r2star_map, s0_map) = r2star_arlo(&magnitude, &mask, &te, 1, 1, 1);

        let r2star_err = (r2star_map[0] - r2star_true).abs() / r2star_true;
        let s0_err = (s0_map[0] - s0_true).abs() / s0_true;

        assert!(r2star_err < 0.01, "R2* error {:.4}% should be < 1%", r2star_err * 100.0);
        assert!(s0_err < 0.01, "S0 error {:.4}% should be < 1%", s0_err * 100.0);
    }

    #[test]
    fn test_r2star_arlo_masked_voxel() {
        let te = vec![0.005, 0.010, 0.015];
        let mask = vec![0u8; 1]; // masked out
        let magnitude = vec![100.0, 80.0, 60.0];

        let (r2star_map, s0_map) = r2star_arlo(&magnitude, &mask, &te, 1, 1, 1);
        assert_eq!(r2star_map[0], 0.0);
        assert_eq!(s0_map[0], 0.0);
    }

    #[test]
    fn test_r2star_arlo_multiple_voxels() {
        let te = vec![0.005, 0.010, 0.015, 0.020];
        let n_echoes = te.len();
        let nx = 2; let ny = 2; let nz = 1;
        let n_voxels = nx * ny * nz;

        let r2star_values = vec![20.0, 40.0, 60.0, 80.0];
        let s0: f64 = 500.0;

        let mask = vec![1u8; n_voxels];
        let mut magnitude = vec![0.0_f64; n_voxels * n_echoes];
        for v in 0..n_voxels {
            for (e, &t) in te.iter().enumerate() {
                magnitude[v * n_echoes + e] = s0 * f64::exp(-r2star_values[v] * t);
            }
        }

        let (r2star_map, _s0_map) = r2star_arlo(&magnitude, &mask, &te, nx, ny, nz);

        for v in 0..n_voxels {
            let err = (r2star_map[v] - r2star_values[v]).abs() / r2star_values[v];
            assert!(err < 0.01, "Voxel {} R2* error {:.4}% should be < 1%", v, err * 100.0);
        }
    }
}

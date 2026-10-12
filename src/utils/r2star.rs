//! R2* and T2* mapping from multi-echo magnitude data
//!
//! Provides R2* estimation using the ARLO (Auto-Regression on Linear Operations)
//! algorithm for equi-spaced echo times, with a log-linear fallback for edge cases.
//! T2* maps can be computed as the reciprocal of R2* via [`t2star_from_r2star`].
//!
//! Reference:
//! Pei, M., et al. (2015). "Algorithm for fast monoexponential fitting based on
//! Auto-Regression on Linear Operations (ARLO) of data."
//! Magnetic Resonance in Medicine, 73(2):843-850.

/// How far each echo spacing may deviate from the mean spacing, in seconds, for
/// [`r2star_arlo`] to accept the echo times as equally spaced (0.1 ms). It covers the rounding
/// of echo times in DICOM and BIDS sidecars, not genuinely unequal spacing.
pub const ARLO_TE_SPACING_TOLERANCE_S: f64 = 1e-4;

/// Whether ARLO can be used on these echo times: at least 3, all finite, distinct, and equally
/// spaced to within `tolerance` seconds (each spacing within `tolerance` of the mean spacing).
///
/// [`r2star_arlo`] applies this with [`ARLO_TE_SPACING_TOLERANCE_S`]; a host can call it with
/// the same constant to reject a run before any work starts.
///
/// # Arguments
/// * `echo_times` - Echo times in seconds, in any order
/// * `tolerance` - Maximum allowed deviation from uniform spacing in seconds
pub fn use_arlo(echo_times: &[f64], tolerance: f64) -> bool {
    arlo_echo_time_problem(echo_times, tolerance).is_none()
}

/// Why ARLO cannot use these echo times, or `None` if it can. See [`use_arlo`].
fn arlo_echo_time_problem(echo_times: &[f64], tolerance: f64) -> Option<String> {
    if echo_times.len() < 3 {
        return Some(format!("ARLO needs at least 3 echoes, got {}", echo_times.len()));
    }
    if echo_times.iter().any(|t| !t.is_finite()) {
        return Some(format!("ARLO needs finite echo times, got {echo_times:?}"));
    }

    let mut te_sorted: Vec<f64> = echo_times.to_vec();
    te_sorted.sort_by(f64::total_cmp);

    let diffs: Vec<f64> = te_sorted.windows(2).map(|w| w[1] - w[0]).collect();
    let mean_diff: f64 = diffs.iter().sum::<f64>() / diffs.len() as f64;
    let max_dev = diffs.iter().map(|&d| (d - mean_diff).abs()).fold(0.0_f64, f64::max);

    if diffs.iter().any(|&d| d <= 0.0) {
        Some(format!("ARLO needs distinct echo times, got {echo_times:?} s"))
    } else if max_dev > tolerance {
        Some(format!(
            "ARLO needs equally spaced echo times, got {echo_times:?} s (spacings deviate from \
             their mean by up to {:.3} ms, tolerance {:.3} ms)",
            max_dev * 1e3, tolerance * 1e3
        ))
    } else {
        None
    }
}

/// R2* mapping using ARLO for equi-spaced echo times (≥3 echoes).
///
/// For each masked voxel, uses the integral-based ARLO estimator (Pei et al.
/// 2015) — a least-squares fit exploiting `s_n - s_{n+2} = R2*·∫S dt` with a
/// Simpson-rule integral — to estimate the mono-exponential decay rate. Falls
/// back to log-linear fitting where the estimate is unreliable.
///
/// # Arguments
/// * `magnitude` - Multi-echo magnitude data, flattened as `[voxel0_echo0, voxel0_echo1, ..., voxel1_echo0, ...]`
///   i.e. shape `(n_voxels, n_echoes)` in row-major order, where `n_voxels = nx*ny*nz`
/// * `mask` - Binary brain mask `[nx*ny*nz]` (1 = process, 0 = skip)
/// * `echo_times` - Echo times in seconds `[n_echoes]` (must be equi-spaced)
/// * `grid` - Volume grid (dimensions and voxel sizes)
///
/// # Returns
/// `(r2star_map, s0_map)` - R2* in Hz and S0 (proton density), both `[nx*ny*nz]`
///
/// # Errors
/// Fewer than 3 echo times, or echo times that are not equally spaced to within
/// [`ARLO_TE_SPACING_TOLERANCE_S`] (see [`use_arlo`]): the Simpson integral uses one spacing
/// for every step, so unequal spacing gives a wrong R2* rather than a noisy one. Also
/// `magnitude` or `mask` lengths that do not match `grid`.
pub fn r2star_arlo(
    magnitude: &[f64],
    mask: &[u8],
    echo_times: &[f64],
    grid: &crate::Grid,
) -> Result<(Vec<f64>, Vec<f64>), crate::error::InputError> {
    use crate::error::{check_len, InputError};
    let n_echoes = echo_times.len();
    if let Some(problem) = arlo_echo_time_problem(echo_times, ARLO_TE_SPACING_TOLERANCE_S) {
        return Err(InputError(problem));
    }
    let n_voxels = grid.n_total();
    check_len("magnitude (n_voxels * n_echoes)", magnitude.len(), n_voxels * n_echoes)?;
    check_len("mask", mask.len(), n_voxels)?;

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

        // ARLO (Pei et al. 2015): integral-based auto-regression on linear
        // operations. From S'(t) = -R2*·S(t), integrating over [t_n, t_{n+2}]:
        //   s_n - s_{n+2} = R2* · ∫_{t_n}^{t_{n+2}} S dt
        // Estimate the integral with Simpson's rule (y_n) and the drop (x_n),
        // then R2* = Σ x_n·y_n / Σ y_n² (least-squares fit of x = R2*·y).
        // This has no ad-hoc ratio selection, so it does not bias on noisy data.
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for i in 0..(n_echoes - 2) {
            let integ = (delta_te / 3.0) * (signal[i] + 4.0 * signal[i + 1] + signal[i + 2]);
            let diff = signal[i] - signal[i + 2];
            num += diff * integ;
            den += integ * integ;
        }

        if den > 1e-30 {
            let r2star_val = num / den;
            if (0.0..=500.0).contains(&r2star_val) {
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

    Ok((r2star_map, s0_map))
}

/// Convert an R2* map (Hz) to a T2* map (seconds).
///
/// T2* = 1 / R2* for voxels where R2* > 0; zero otherwise.
pub fn t2star_from_r2star(r2star: &[f64]) -> Vec<f64> {
    r2star.iter().map(|&r| if r > 0.0 { 1.0 / r } else { 0.0 }).collect()
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
    use crate::Grid;

    fn grid(nx: usize, ny: usize, nz: usize) -> Grid {
        Grid::new(nx, ny, nz, 1.0, 1.0, 1.0)
    }

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

        // Single voxel in a 1x1x1 volume
        let mask = vec![1u8];
        let magnitude: Vec<f64> = te.iter()
            .map(|&t| s0_true * (-r2star_true * t).exp())
            .collect();

        let (r2star_map, s0_map) = r2star_arlo(&magnitude, &mask, &te, &grid(1, 1, 1)).unwrap();

        let r2star_err = (r2star_map[0] - r2star_true).abs() / r2star_true;
        let s0_err = (s0_map[0] - s0_true).abs() / s0_true;

        assert!(r2star_err < 0.01, "R2* error {:.4}% should be < 1%", r2star_err * 100.0);
        assert!(s0_err < 0.01, "S0 error {:.4}% should be < 1%", s0_err * 100.0);
    }

    /// Unequal spacing used to be accepted silently and give a wrong R2*; now it is refused,
    /// as are too few, repeated or non-finite echo times, and mismatched buffers (#273).
    #[test]
    fn r2star_arlo_refuses_echo_times_it_cannot_use() {
        let g = grid(1, 1, 1);
        let run = |te: &[f64]| {
            let mag: Vec<f64> = te.iter().map(|t| (-30.0 * t).exp()).collect();
            r2star_arlo(&mag, &[1], te, &g)
        };
        assert!(run(&[0.005, 0.010, 0.015, 0.020]).is_ok());
        // rounding-level jitter is fine
        assert!(run(&[0.005, 0.01002, 0.01499, 0.020]).is_ok());
        for (te, why) in [
            (&[0.005, 0.010][..], "at least 3"),
            (&[0.005, 0.010, 0.020][..], "equally spaced"),
            (&[0.004, 0.008, 0.012, 0.030][..], "equally spaced"),
            (&[0.005, 0.005, 0.005][..], "distinct"),
            (&[0.005, f64::NAN, 0.015][..], "finite"),
        ] {
            let err = run(te).unwrap_err().to_string();
            assert!(err.contains(why), "{te:?}: {err}");
            assert!(!use_arlo(te, ARLO_TE_SPACING_TOLERANCE_S), "{te:?}");
        }
        let te = [0.005, 0.010, 0.015];
        assert!(r2star_arlo(&[1.0; 2], &[1], &te, &g).is_err());
        assert!(r2star_arlo(&[1.0; 3], &[1, 1], &te, &g).is_err());
    }

    #[test]
    fn test_t2star_from_r2star() {
        let r2star = vec![50.0, 0.0, 100.0, 25.0];
        let t2star = t2star_from_r2star(&r2star);
        assert!((t2star[0] - 0.02).abs() < 1e-10);
        assert_eq!(t2star[1], 0.0);
        assert!((t2star[2] - 0.01).abs() < 1e-10);
        assert!((t2star[3] - 0.04).abs() < 1e-10);
    }

    #[test]
    fn test_r2star_arlo_masked_voxel() {
        let te = vec![0.005, 0.010, 0.015];
        let mask = vec![0u8; 1]; // masked out
        let magnitude = vec![100.0, 80.0, 60.0];

        let (r2star_map, s0_map) = r2star_arlo(&magnitude, &mask, &te, &grid(1, 1, 1)).unwrap();
        assert_eq!(r2star_map[0], 0.0);
        assert_eq!(s0_map[0], 0.0);
    }

    #[test]
    fn test_r2star_arlo_multiple_voxels() {
        let te = vec![0.005, 0.010, 0.015, 0.020];
        let n_echoes = te.len();
        let nx = 2; let ny = 2; let nz = 1;
        let n_voxels = nx * ny * nz;

        let r2star_values = [20.0, 40.0, 60.0, 80.0];
        let s0: f64 = 500.0;

        let mask = vec![1u8; n_voxels];
        let mut magnitude = vec![0.0_f64; n_voxels * n_echoes];
        for v in 0..n_voxels {
            for (e, &t) in te.iter().enumerate() {
                magnitude[v * n_echoes + e] = s0 * f64::exp(-r2star_values[v] * t);
            }
        }

        let (r2star_map, _s0_map) = r2star_arlo(&magnitude, &mask, &te, &grid(nx, ny, nz)).unwrap();

        for v in 0..n_voxels {
            let err = (r2star_map[v] - r2star_values[v]).abs() / r2star_values[v];
            assert!(err < 0.01, "Voxel {} R2* error {:.4}% should be < 1%", v, err * 100.0);
        }
    }
}

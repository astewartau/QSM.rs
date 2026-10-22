//! R2* and T2* mapping from multi-echo magnitude data
//!
//! [`arlo`] is a port of `arlo.m` from the MEDI toolbox (release 2024.11.26, Cornell MRI
//! Research Lab), the ARLO estimator of Pei et al. (2015) as its authors' group implements it.
//! T2* maps are the reciprocal of R2* ([`t2star_from_r2star`]).
//!
//! Reference:
//! Pei, M., Nguyen, T.D., Thimmappa, N.D., Salustri, C., Dong, F., Cooper, M.A., Li, J.,
//! Prince, M.R., Wang, Y. (2015). "Algorithm for fast monoexponential fitting based on
//! Auto-Regression on Linear Operations (ARLO) of data."
//! Magnetic Resonance in Medicine, 73(2):843-850. https://doi.org/10.1002/mrm.25137

use crate::error::{check_len, InputError};

/// Why [`arlo`] cannot use these echo times, or `None` if it can: it needs at least three
/// finite, strictly increasing echo times. A host can call this to reject a run before any work
/// starts.
pub fn arlo_echo_times_problem(echo_times: &[f64]) -> Option<String> {
    if echo_times.len() < 3 {
        Some(format!("ARLO needs at least 3 echoes, got {}", echo_times.len()))
    } else if echo_times.iter().any(|t| !t.is_finite()) {
        Some(format!("ARLO needs finite echo times, got {echo_times:?}"))
    } else if echo_times.windows(2).any(|w| w[1] <= w[0]) {
        Some(format!("ARLO needs strictly increasing echo times, got {echo_times:?} s"))
    } else {
        None
    }
}

/// R2* by ARLO: `arlo.m` (MEDI toolbox 2024.11.26), line for line.
///
/// Over each consecutive echo triplet `j, j+1, j+2`, `y` is `arlo.m`'s integral of the signal
/// (Simpson's rule for even spacing, its general form otherwise) and `x = s_j − s_{j+2}`; then
/// `R2* = (Σ y·x + Σ β·x²) / (Σ β·y·x + Σ y²)`. Echo spacing may be uneven. Every voxel is
/// fitted, with no upper clip and no fallback, and NaN or infinite results are 0, as in
/// `arlo.m`.
///
/// **Deviations from `arlo.m`:**
/// - Negative estimates (noise-only or rising-signal voxels) are set to 0: a negative decay rate
///   has no physical meaning, and QSM.rs does not output negative relaxation rates. Every
///   non-negative estimate is `arlo.m`'s value unchanged. (MEDI's `extract_CSF` thresholds R2*
///   at 5 Hz, so its CSF mask is the same either way.)
/// - `arlo.m` returns `[]` for one echo and 0 everywhere for two (no triplet); this returns an
///   error for fewer than three echoes, and for echo times that are not finite and strictly
///   increasing (where `arlo.m` divides by zero or fits the wrong order).
///
/// # Arguments
/// * `magnitude` - The echo magnitudes, `echo_times.len()` volumes of `n_voxels` stacked with
///   the echo slowest (the order of a 4D NIfTI)
/// * `echo_times` - Echo times in seconds
/// * `n_voxels` - Voxels per echo
///
/// # Returns
/// R2* in Hz per voxel
///
/// # Errors
/// See [`arlo_echo_times_problem`]; also a `magnitude` whose length is not
/// `n_voxels * echo_times.len()`.
pub fn arlo(magnitude: &[f64], echo_times: &[f64], n_voxels: usize) -> Result<Vec<f64>, InputError> {
    let mut r2 = arlo_unclipped(magnitude, echo_times, n_voxels)?;
    // (deviation) no negative R2*
    for v in r2.iter_mut() {
        *v = v.max(0.0);
    }
    Ok(r2)
}

/// `arlo.m` exactly, negative estimates included (NaN and infinite results are 0, as in
/// `arlo.m`). [`arlo`] clips the negatives; this is for ports whose result depends on them, such
/// as mSMV's vessel mask (`fibermetric` on the R2* map, see [`msmv`](mod@crate::inversion::msmv)).
/// Same arguments and errors as [`arlo`].
pub fn arlo_unclipped(magnitude: &[f64], echo_times: &[f64], n_voxels: usize) -> Result<Vec<f64>, InputError> {
    if let Some(problem) = arlo_echo_times_problem(echo_times) {
        return Err(InputError(problem));
    }
    let nte = echo_times.len();
    check_len("magnitude (n_echoes * n_voxels)", magnitude.len(), n_voxels * nte)?;
    let te = echo_times;
    let mut yy = vec![0.0f64; n_voxels];
    let mut yx = vec![0.0f64; n_voxels];
    let mut beta_yx = vec![0.0f64; n_voxels];
    let mut beta_xx = vec![0.0f64; n_voxels];
    for j in 0..nte - 2 {
        let alpha = (te[j + 2] - te[j]) * (te[j + 2] - te[j]) / 2.0 / (te[j + 1] - te[j]);
        let tmp = (2.0 * te[j + 2] * te[j + 2] - te[j] * te[j + 2] - te[j] * te[j]
            + 3.0 * te[j] * te[j + 1] - 3.0 * te[j + 1] * te[j + 2]) / 6.0;
        let beta = tmp / (te[j + 2] - te[j + 1]);
        let gamma = tmp / (te[j + 1] - te[j]);
        let (c0, c1, c2) = (te[j + 2] - te[j] - alpha + gamma, alpha - beta - gamma, beta);
        let (s0, s1, s2) = (
            &magnitude[j * n_voxels..(j + 1) * n_voxels],
            &magnitude[(j + 1) * n_voxels..(j + 2) * n_voxels],
            &magnitude[(j + 2) * n_voxels..(j + 3) * n_voxels],
        );
        for v in 0..n_voxels {
            let y1 = s0[v] * c0 + s1[v] * c1 + s2[v] * c2;
            let x1 = s0[v] - s2[v];
            yy[v] += y1 * y1;
            yx[v] += y1 * x1;
            beta_yx[v] += beta * y1 * x1;
            beta_xx[v] += beta * x1 * x1;
        }
    }
    Ok((0..n_voxels)
        .map(|v| {
            // r2(isnan(r2)) = 0; r2(isinf(r2)) = 0
            let r2 = (yx[v] + beta_xx[v]) / (beta_yx[v] + yy[v]);
            if r2.is_finite() { r2 } else { 0.0 }
        })
        .collect())
}

/// Convert an R2* map (Hz) to a T2* map (seconds).
///
/// T2* = 1 / R2* for voxels where R2* > 0; zero otherwise. R2* = 0 (no measurable decay, or no
/// fit) has no finite T2*; 0 marks it as "no value" rather than a cap such as 1 s, so a T2*
/// map is never larger than the decay it can measure.
pub fn t2star_from_r2star(r2star: &[f64]) -> Vec<f64> {
    r2star.iter().map(|&r| if r > 0.0 { 1.0 / r } else { 0.0 }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arlo_matches_the_closed_form() {
        // Noise-free monoexponential decay at even and uneven spacing: ARLO is close to exact.
        for te in [vec![0.005, 0.011, 0.017, 0.023, 0.029], vec![0.004, 0.009, 0.016, 0.030]] {
            let r2: [f64; 3] = [5.0, 30.0, 80.0];
            let n = r2.len();
            let mag: Vec<f64> = te.iter().flat_map(|&t| r2.iter().map(move |&r| 100.0 * (-r * t).exp())).collect();
            let est = arlo(&mag, &te, n).unwrap();
            for v in 0..n {
                assert!((est[v] - r2[v]).abs() / r2[v] < 0.02, "{te:?}: {} vs {}", est[v], r2[v]);
            }
        }
        // even spacing: arlo_even.m's closed form
        let (esp, s) = (0.006, [90.0, 70.0, 61.0, 40.0]);
        let te: Vec<f64> = (0..4).map(|i| 0.005 + esp * i as f64).collect();
        let (mut xy, mut xx, mut yy) = (0.0, 0.0, 0.0);
        for j in 0..2 {
            let y = esp / 3.0 * (s[j] + 4.0 * s[j + 1] + s[j + 2]);
            let x = s[j] - s[j + 2];
            xy += x * y;
            xx += x * x;
            yy += y * y;
        }
        let want = (esp / 3.0 * xx + xy) / (yy + esp / 3.0 * xy);
        assert!((arlo(&s, &te, 1).unwrap()[0] - want).abs() < 1e-9 * want);
        // zero signal: 0/0 -> 0
        assert_eq!(arlo(&[0.0; 3], &[0.005, 0.01, 0.015], 1).unwrap(), vec![0.0]);
        // a rising signal gives a negative arlo.m estimate; clamped to 0
        assert_eq!(arlo(&[10.0, 20.0, 40.0], &[0.005, 0.01, 0.015], 1).unwrap(), vec![0.0]);
    }

    #[test]
    fn arlo_layout_is_echo_slowest() {
        let te: [f64; 4] = [0.005, 0.010, 0.015, 0.020];
        let r2: [f64; 4] = [20.0, 40.0, 60.0, 80.0];
        let mag: Vec<f64> = te.iter().flat_map(|&t| r2.iter().map(move |&r| 500.0 * (-r * t).exp())).collect();
        let est = arlo(&mag, &te, 4).unwrap();
        for v in 0..4 {
            assert!((est[v] - r2[v]).abs() / r2[v] < 0.01, "voxel {v}: {}", est[v]);
        }
    }

    #[test]
    fn arlo_refuses_echo_times_it_cannot_use() {
        let run = |te: &[f64]| {
            let mag: Vec<f64> = te.iter().map(|t| (-30.0 * t).exp()).collect();
            arlo(&mag, te, 1)
        };
        assert!(run(&[0.005, 0.010, 0.015, 0.020]).is_ok());
        assert!(run(&[0.004, 0.008, 0.012, 0.030]).is_ok(), "uneven spacing is arlo.m's general form");
        for (te, why) in [
            (&[0.005][..], "at least 3"),
            (&[0.005, 0.010][..], "at least 3"),
            (&[0.005, 0.005, 0.005][..], "strictly increasing"),
            (&[0.015, 0.010, 0.005][..], "strictly increasing"),
            (&[0.005, f64::NAN, 0.015][..], "finite"),
        ] {
            let err = run(te).unwrap_err().to_string();
            assert!(err.contains(why), "{te:?}: {err}");
            assert!(arlo_echo_times_problem(te).is_some());
        }
        assert!(arlo(&[1.0; 2], &[0.005, 0.010, 0.015], 1).is_err());
    }

    #[test]
    fn test_t2star_from_r2star() {
        let r2star = vec![50.0, 0.0, 100.0, 25.0, -3.0];
        let t2star = t2star_from_r2star(&r2star);
        assert!((t2star[0] - 0.02).abs() < 1e-10);
        assert_eq!(t2star[1], 0.0);
        assert!((t2star[2] - 0.01).abs() < 1e-10);
        assert!((t2star[3] - 0.04).abs() < 1e-10);
        assert_eq!(t2star[4], 0.0);
    }
}

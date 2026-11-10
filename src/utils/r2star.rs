//! R2* and T2* mapping from multi-echo magnitude data
//!
//! [`arlo`] is a port of `arlo.m` from the MEDI toolbox (release 2024.11.26, Cornell MRI
//! Research Lab), the ARLO estimator of Pei et al. (2015) as its authors' group implements it.
//! [`chisep_nnls`] is a port of `r2star_nnls` from the SNU χ-separation toolbox v1.1.3, a bounded
//! mono-exponential fit per voxel by `lsqnonlin`. T2* maps are the reciprocal of R2*
//! ([`t2star_from_r2star`]).
//!
//! # Reference pipeline
//! MEDI's pipeline script (`README.m:133` of the 2024.11.26 release) runs `arlo(TE, abs(iField))`
//! on the magnitudes of all echoes, unmasked, and uses the result only for the ventricular CSF
//! mask (`extract_CSF(R2s, Mask, voxel_size)`, R2* < 5 Hz within 30 mm of the mask centroid,
//! after a 10 mm SMV erosion; `extract_CSF.m:8-37`) and, in the 2024 release, for the
//! source-separation initialisation. `arlo.m` returns no map for one echo (`arlo.m:22-26`).
//! Nothing in it depends on the field strength.
//!
//! # The χ-separation toolbox's NNLS fit
//!
//! `Chisep_script.m` (toolbox v1.1.3, l. 87-93) computes R2* with `r2star_arlo` if `use_arlo(TE)`
//! and `r2star_nnls` otherwise, and as written always takes `r2star_nnls` (see
//! [`chi_sep_medi`](mod@crate::separation::chi_sep_medi), *Reference pipeline*). The toolbox ships
//! it as P-code (`functions/supportfunc/r2star_nnls.p`). Its algorithm was read off in licensed
//! MATLAB R2023b Update 5 (23.2.0.2459199) on 2026-11-10, by running it with logging stand-ins for
//! `lsqnonlin` and `optimoptions` ahead of MATLAB's on the path and no parallel pool (so its
//! `parfor` runs in the client and calls them): every call's objective (`func2str` and its
//! captured workspace), start point, bounds, options and result. It is:
//!
//! - `imgvec`: `|mag| ./ |mag(:,:,:,1)|` (equal bit for bit to the logged one), reshaped to
//!   (voxels per slice, slices, echoes); a `parfor` over slices and a loop over the voxels of
//!   each where the mask is true (`logical` of it: 0.5, 2 and −1 count as inside) and the
//!   normalised first echo is not NaN (a first echo of 0, ±Inf or NaN: not fitted, R2* 0).
//! - `sig = @(x) abs(x(1))*exp(-TE*x(2)/1000)` with `TE` in ms (the script passes `TE*1000`; a
//!   row or a column gives the same result) and the objective
//!   `@(x) abs(sig(x) - sqz(imgvec(ii,slice,:))')`: the **absolute** residual.
//! - `lsqnonlin(fun, [1 40], [0 0], [2 1000], optimset('Display','off','TolFun',1e-5,'TolX',1e-5))`,
//!   one output (`x`). `optimset` is the legacy form: its `TolFun` sets the function and the
//!   first-order optimality tolerances (on the fixture below, optimality at 1e-5 reproduces 294
//!   of 300 voxels' iterations and 1e-6 only 197). The algorithm is `lsqnonlin`'s default,
//!   trust-region-reflective, with forward differences ([`crate::solvers::lsq_trr`]).
//! - R2* (Hz) = `x(2)` as `lsqnonlin` returns it (so within [0, 1000]); voxels not fitted are 0.
//!   A NaN or Inf ratio at a later echo of a fitted voxel (a NaN or Inf magnitude there, or a
//!   subnormal first echo) stops the toolbox ("Objective function is returning Inf or NaN values
//!   at initial point"); [`chisep_nnls`] returns an error there.
//!
//! The amplitude bound of 2 caps the R2* the fit can reach at short first echoes: a voxel that
//! decays by more than half between `TE = 0` and the first echo cannot be fitted exactly (a
//! noise-free 200 Hz decay at TE₁ = 5 ms gives 156.3 Hz, in MATLAB and here). The tolerances
//! stop a flat signal at 0.066 Hz, not 0. For the CSF mask (R2* < 5 Hz) neither matters.
//!
//! **Agreement** (measured 2026-11-10). The residual is computed with MATLAB's `exp`
//! ([`crate::utils::fdlibm`]; bit-identical to MATLAB's on the probed points), and the
//! finite-difference Jacobian at a point MATLAB also visits is bit-identical to the one
//! `lsqnonlin` returns. What differs is the first trial step, by about an ulp (the floor of
//! [`crate::solvers::lsq_trr`]: MATLAB's step solve rounds differently); forward differences then
//! amplify that by ~1/√eps, so later iterates differ at ~1e-8 relative and the fits stop, at the
//! 1e-5 tolerances, a little apart.
//! - 300 noisy mono-exponential voxels (R2* 0-2400 Hz, SNR 2-∞, TE 5:6:29 ms;
//!   `tests/data/r2star_nnls_chisep_matlab.txt`, made by `tests/data/r2star_nnls_chisep_matlab.m`):
//!   294 take the same iterations, function evaluations and exit flag; R2* max |Δ| 3.8e-5 Hz,
//!   median 3.3e-8 Hz, above 1 Hz at most 2.0e-5 relative (median 2.1e-9). The other six stop
//!   at R2* ≈ 0 (< 1e-5 Hz) after a different number of iterations.
//! - In vivo (EMTP s005001, Siemens Prisma 3 T, TE 5:6:29 ms, 1.55 M voxels in the toolbox's BET
//!   mask; the toolbox's own R2* from QSM-CI's `chisep-medi` reference run, against this port
//!   on a NumPy re-implementation of the toolbox's Tukey-windowed magnitude): NRMSE 1.1e-6,
//!   max |Δ| 0.029 Hz, median 4e-7 Hz; no voxel crosses `extract_CSF`'s 5 Hz threshold, so the
//!   CSF mask is the toolbox's. [`chisep_nnls`] takes 1.7 s on 6 threads.
//! - Edge cases (`chisep_nnls_edge_cases_match_the_toolbox`): to 1e-6 relative, with one
//!   exception, a voxel with one echo 10⁶ times the others, where MATLAB stops at 1.5e-6 Hz and
//!   this port at 2.0 Hz.
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

/// `r2star_nnls`'s start point `[S0/S(TE₁), R2*]` (R2* in Hz).
pub const CHISEP_NNLS_X0: [f64; 2] = [1.0, 40.0];
/// `r2star_nnls`'s lower bounds.
pub const CHISEP_NNLS_LB: [f64; 2] = [0.0, 0.0];
/// `r2star_nnls`'s upper bounds: the normalised amplitude at most 2, R2* at most 1000 Hz.
pub const CHISEP_NNLS_UB: [f64; 2] = [2.0, 1000.0];
/// `r2star_nnls`'s `optimset` tolerances (`TolFun` and `TolX`).
pub const CHISEP_NNLS_TOL: f64 = 1e-5;

/// Why [`chisep_nnls`] cannot use these echo times, or `None` if it can: it needs at least two
/// finite echo times (two unknowns per voxel; `lsqnonlin`'s trust-region-reflective algorithm
/// needs at least as many residuals as unknowns).
pub fn chisep_nnls_echo_times_problem(echo_times: &[f64]) -> Option<String> {
    if echo_times.len() < 2 {
        Some(format!("the χ-separation toolbox's NNLS R2* needs at least 2 echoes, got {}", echo_times.len()))
    } else if echo_times.iter().any(|t| !t.is_finite()) {
        Some(format!("the χ-separation toolbox's NNLS R2* needs finite echo times, got {echo_times:?}"))
    } else {
        None
    }
}

/// The `lsqnonlin` options `r2star_nnls` passes: `optimset('Display','off','TolFun',1e-5,
/// 'TolX',1e-5)`, everything else `lsqnonlin`'s default.
pub fn chisep_nnls_options() -> crate::solvers::LsqTrrOptions {
    crate::solvers::LsqTrrOptions {
        function_tolerance: CHISEP_NNLS_TOL,
        step_tolerance: CHISEP_NNLS_TOL,
        optimality_tolerance: CHISEP_NNLS_TOL,
        ..Default::default()
    }
}

/// R2* as the χ-separation toolbox's `r2star_nnls` computes it (v1.1.3,
/// `functions/supportfunc/r2star_nnls.p`): a bounded mono-exponential least-squares fit per
/// voxel by `lsqnonlin`'s trust-region-reflective method ([`crate::solvers::lsq_trr`]).
///
/// For each voxel where `mask` is true:
/// 1. `y = |S| / |S(TE₁)|`, the echo magnitudes normalised by the first echo's. A voxel whose
///    normalised first echo is NaN (first echo 0, NaN or infinite) is not fitted: its R2* is 0.
/// 2. `lsqnonlin(@(x) abs(abs(x(1))·exp(−TE·x(2)/1000) − y), [1 40], [0 0], [2 1000], opts)`
///    with `TE` in ms and `opts = optimset('Display','off','TolFun',1e-5,'TolX',1e-5)`
///    ([`chisep_nnls_options`]). The residual is the **absolute** difference, as the toolbox
///    writes it.
/// 3. R2* (Hz) is the fitted `x(2)`, as `lsqnonlin` returns it (bounded to [0, 1000] Hz).
///
/// Voxels outside the mask are 0. See the module docs (*The χ-separation toolbox's NNLS fit*)
/// for how this was established and how closely it matches the toolbox.
///
/// # Arguments
/// * `magnitude` - The echo magnitudes, `echo_times.len()` volumes of `n_voxels` stacked with
///   the echo slowest (the order of a 4D NIfTI)
/// * `echo_times` - Echo times in seconds (multiplied by 1000 to the toolbox's ms, as
///   `Chisep_script.m` passes `TE*1000`)
/// * `mask` - The voxels to fit (`n_voxels`; `Chisep_script.m` passes BET's brain mask)
///
/// # Returns
/// R2* in Hz per voxel
///
/// # Errors
/// See [`chisep_nnls_echo_times_problem`]; also mismatched lengths, and a fitted voxel whose
/// normalised magnitude is not finite at a later echo (a NaN or infinite magnitude there, or one
/// that overflows when divided by the first echo's): `lsqnonlin` stops the toolbox with
/// "Objective function is returning Inf or NaN values at initial point".
pub fn chisep_nnls(magnitude: &[f64], echo_times: &[f64], mask: &[bool]) -> Result<Vec<f64>, InputError> {
    use crate::solvers::lsq_trr::LsqTrrWorkspace;
    #[cfg(feature = "parallel")]
    use rayon::prelude::*;

    if let Some(problem) = chisep_nnls_echo_times_problem(echo_times) {
        return Err(InputError(problem));
    }
    let n_voxels = mask.len();
    let ne = echo_times.len();
    check_len("magnitude (n_echoes * n_voxels)", magnitude.len(), n_voxels * ne)?;
    // Chisep_script.m: r2star_nnls(mag_multi_echo, TE*1000, mask_brain)
    let te_ms: Vec<f64> = echo_times.iter().map(|&t| t * 1000.0).collect();
    let opts = chisep_nnls_options();
    let voxels: Vec<usize> = (0..n_voxels).filter(|&v| mask[v]).collect();

    struct Scratch {
        y: Vec<f64>,
        ws: LsqTrrWorkspace,
    }
    let fits: Vec<Result<(usize, f64), InputError>> = crate::maybe_par_map_init!(
        voxels,
        || Scratch { y: vec![0.0; ne], ws: LsqTrrWorkspace::new(ne, 2) },
        |sc: &mut Scratch, &v: &usize| {
            let s1 = magnitude[v].abs();
            for e in 0..ne {
                sc.y[e] = magnitude[e * n_voxels + v].abs() / s1;
            }
            if sc.y[0].is_nan() {
                return Ok((v, 0.0));
            }
            if let Some(e) = sc.y.iter().position(|y| !y.is_finite()) {
                return Err(InputError(format!(
                    "voxel {v}: the magnitude at echo {} divided by the first echo's is {} \
                     (r2star_nnls's lsqnonlin cannot fit a non-finite value)",
                    e + 1,
                    sc.y[e]
                )));
            }
            Ok((v, chisep_nnls_fit(&sc.y, &te_ms, &opts, &mut sc.ws).0[1]))
        }
    )
    .collect();
    let mut r2 = vec![0.0; n_voxels];
    for fit in fits {
        let (v, r) = fit?;
        r2[v] = r;
    }
    Ok(r2)
}

/// One voxel of [`chisep_nnls`]: `lsqnonlin` on the normalised magnitudes `y` (finite), echo
/// times in ms. Returns `x = [S0/S(TE₁), R2* (Hz)]` and the solver's report.
fn chisep_nnls_fit(
    y: &[f64],
    te_ms: &[f64],
    opts: &crate::solvers::LsqTrrOptions,
    ws: &mut crate::solvers::lsq_trr::LsqTrrWorkspace,
) -> ([f64; 2], crate::solvers::LsqTrrReport) {
    use crate::utils::fdlibm;
    let mut x = CHISEP_NNLS_X0;
    let report = crate::solvers::lsq_trr::lsq_trust_region_reflective(
        |x: &[f64], out: &mut [f64]| {
            // abs(sig(x) - y), sig = @(x) abs(x(1))*exp(-TE*x(2)/1000)
            for (o, (&t, &yv)) in out.iter_mut().zip(te_ms.iter().zip(y)) {
                *o = (x[0].abs() * fdlibm::exp(-t * x[1] / 1000.0) - yv).abs();
            }
        },
        &mut x,
        &CHISEP_NNLS_LB,
        &CHISEP_NNLS_UB,
        opts,
        ws,
        |_| {},
    );
    (x, report)
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
    fn chisep_nnls_matches_the_toolbox() {
        // tests/data/r2star_nnls_chisep_matlab.txt: r2star_nnls in licensed MATLAB on 300 noisy voxels.
        let text = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/r2star_nnls_chisep_matlab.txt"));
        let te = [0.005, 0.011, 0.017, 0.023, 0.029];
        let rows: Vec<Vec<f64>> = text
            .lines()
            .filter(|l| !l.starts_with('#'))
            .map(|l| l.split_whitespace().map(|t| t.parse().unwrap()).collect())
            .collect();
        let n = rows.len();
        let mag: Vec<f64> = (0..5).flat_map(|e| rows.iter().map(move |r| r[e])).collect();
        let r2 = chisep_nnls(&mag, &te, &vec![true; n]).unwrap();
        let te_ms: Vec<f64> = te.iter().map(|t| t * 1000.0).collect();
        let mut ws = crate::solvers::lsq_trr::LsqTrrWorkspace::new(5, 2);
        let (mut same_path, mut worst, mut worst_rel) = (0, 0.0f64, 0.0f64);
        for (v, row) in rows.iter().enumerate() {
            let y: Vec<f64> = (0..5).map(|e| row[e].abs() / row[0].abs()).collect();
            let (x, rep) = chisep_nnls_fit(&y, &te_ms, &chisep_nnls_options(), &mut ws);
            assert_eq!(x[1], r2[v]);
            let path = rep.exit.matlab_exitflag() == row[8] as i32
                && rep.iterations == row[9] as usize
                && rep.func_count == row[10] as usize;
            same_path += path as usize;
            let d = (r2[v] - row[6]).abs();
            worst = worst.max(d);
            if row[6] > 1.0 {
                worst_rel = worst_rel.max(d / row[6]);
            }
            if std::env::var("CHISEP_NNLS_DUMP").is_ok() {
                eprintln!("{v} {path} {} {} | {} {} {} {} | {:e}", r2[v], row[6], rep.exit.matlab_exitflag(), row[8], rep.iterations, row[9], d);
            }
        }
        eprintln!("same path {same_path}/{n}, max |ΔR2*| {worst:e} Hz, max relative (R2* > 1 Hz) {worst_rel:e}");
        // Measured 2026-11-10: 294 of 300 take MATLAB's iterations, function evaluations and exit
        // flag; max |ΔR2*| 3.8e-5 Hz, 2.0e-5 relative (module docs, *Agreement*).
        assert!(same_path >= 294, "{same_path}/{n}");
        assert!(worst < 5e-5 && worst_rel < 3e-5, "{worst:e} {worst_rel:e}");
    }

    /// The cases probed one voxel at a time in licensed MATLAB (2026-11-10, toolbox v1.1.3), TE
    /// [5 11 17 23 29] ms: R2* as `r2star_nnls` returned it, or that it stopped with an error.
    #[test]
    fn chisep_nnls_edge_cases_match_the_toolbox() {
        let te = [0.005, 0.011, 0.017, 0.023, 0.029];
        let one = |s: [f64; 5]| chisep_nnls(&s, &te, &[true]);
        let nan = f64::NAN;
        let inf = f64::INFINITY;
        for (s, want) in [
            ([10.0, 5.0, 3.0, 2.0, 1.0], 99.59012838789927),
            // magnitudes enter as |S|, and only their ratio to the first echo matters
            ([-10.0, -5.0, -2.0, -1.0, -0.5], 125.35960623944304),
            ([1e-300, 5e-301, 2e-301, 1e-301, 5e-302], 125.35960623944304),
            // the tolerances stop the fit short of the exact answer (0 Hz)
            ([5.0, 5.0, 5.0, 5.0, 5.0], 0.06597025437079528),
            ([10.0, 0.0, 0.0, 0.0, 0.0], 187.44475404899214),
            // a first echo of 0, NaN or Inf makes y(1) NaN: not fitted
            ([0.0, 5.0, 3.0, 2.0, 1.0], 0.0),
            ([0.0; 5], 0.0),
            ([nan, 5.0, 3.0, 2.0, 1.0], 0.0),
            ([inf, 5.0, 3.0, 2.0, 1.0], 0.0),
            ([-inf, 5.0, 3.0, 2.0, 1.0], 0.0),
            ([inf, inf, 1.0, 1.0, 1.0], 0.0),
            ([1e300, 1e-300, 1e-300, 1e-300, 1e-300], 187.44475404899214),
        ] {
            let got = one(s).unwrap()[0];
            assert!((got - want).abs() <= 1e-6 * want.max(1.0), "{s:?}: {got} vs MATLAB {want}");
        }
        // One echo a million times the others: MATLAB stops at 1.5e-6 Hz, this port at 2.0 Hz.
        // Both are within the bounds; the path is decided by the last bits of the first step
        // (module docs, *Agreement*).
        let odd = one([1.0, 1e6, 1.0, 1.0, 1.0]).unwrap()[0];
        assert!((0.0..=1000.0).contains(&odd), "{odd}");
        // lsqnonlin: "Objective function is returning Inf or NaN values at initial point"
        // (a subnormal first echo overflows the later ratios)
        for s in [[10.0, nan, 5.0, 3.0, 2.0], [10.0, inf, 5.0, 3.0, 2.0], [10.0, 9.0, 8.0, 7.0, nan], [1e-320, 5.0, 3.0, 2.0, 1.0]] {
            assert!(one(s).is_err(), "{s:?}");
        }
        // Noise-free decays (S0 = 100): 30 Hz to the tolerance; 200 Hz would need S0/S(TE1) = e,
        // above the bound of 2, so the fit stops at the bound (MATLAB: 156.29546150101228).
        let decay = |r: f64| te.map(|t| 100.0 * (-r * t).exp());
        assert!((one(decay(30.0)).unwrap()[0] - 30.00000018782443).abs() < 1e-7);
        assert!((one(decay(200.0)).unwrap()[0] - 156.29546150101228).abs() < 1e-6);
        // outside the mask: 0, and not checked
        assert_eq!(chisep_nnls(&[10.0, nan, 5.0, 3.0, 2.0], &te, &[false]).unwrap(), vec![0.0]);
    }

    #[test]
    fn chisep_nnls_layout_and_echo_times() {
        // Echo-slowest layout, and the mask picks the voxels (MATLAB: [99.59 93.40 88.62] Hz for
        // these three voxels, with the third masked out here).
        let vox = [[10.0, 5.0, 3.0, 2.0, 1.0], [10.0, 6.0, 3.0, 2.0, 1.0], [10.0, 7.0, 3.0, 2.0, 1.0]];
        let mag: Vec<f64> = (0..5).flat_map(|e| vox.iter().map(move |v| v[e])).collect();
        let te = [0.005, 0.011, 0.017, 0.023, 0.029];
        let r = chisep_nnls(&mag, &te, &[true, true, false]).unwrap();
        assert!((r[0] - 99.59012838789927).abs() < 1e-6 * 99.6 && (r[1] - 93.39809322011298).abs() < 1e-6 * 93.4, "{r:?}");
        assert_eq!(r[2], 0.0);
        // Two echoes are enough for two unknowns; one is not. Echo spacing may be anything.
        assert!(chisep_nnls(&[10.0, 5.0], &[0.005, 0.011], &[true]).is_ok());
        assert!(chisep_nnls(&[10.0, 5.0, 4.0], &[0.005, 0.017, 0.011], &[true]).is_ok());
        assert!(chisep_nnls(&[10.0], &[0.005], &[true]).unwrap_err().to_string().contains("at least 2"));
        assert!(chisep_nnls(&[10.0, 5.0], &[0.005, nan_te()], &[true]).unwrap_err().to_string().contains("finite"));
        assert!(chisep_nnls(&[10.0, 5.0, 1.0], &[0.005, 0.011], &[true]).is_err());
        fn nan_te() -> f64 {
            f64::NAN
        }
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

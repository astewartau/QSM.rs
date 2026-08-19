//! DECOMPOSE-QSM: signal-domain paramagnetic/diamagnetic source separation.
//!
//! DECOMPOSE (Chen et al., NeuroImage 2021) separates paramagnetic (χ+, iron) and
//! diamagnetic (χ−, myelin·calcium) susceptibility by fitting a three-compartment
//! complex multi-echo gradient-echo signal per voxel. Each voxel's signal is
//!
//! ```text
//!   S(t) = C+ · exp(−( a·χ+  + R2*₀ + i·(2/3)·χ+·γ·B0)·t)
//!        + C− · exp(−(−a·χ−  + R2*₀ + i·(2/3)·χ−·γ·B0)·t)
//!        + C₀ · exp(−R2*₀·t)
//! ```
//!
//! with `γ = 42.58·2π`, `a = 2π·γ·B0 / (9√3)` the static-dephasing broadening
//! coefficient, χ+ ≥ 0, χ− ≤ 0 (ppm-scale). C+, C−, C₀ are the paramagnetic,
//! diamagnetic and neutral compartment amplitudes and R2*₀ the baseline decay.
//!
//! Like the QSM-CI reference, the *phase* is synthesized from the provided
//! conventional QSM (χ_total) rather than re-derived from raw multi-echo phase —
//! this isolates DECOMPOSE's separation step. The observed complex data per echo
//! is `y = |mag_norm| · exp(−i·(2/3)·χ_total·γ·B0·TE)` (magnitude normalised by its
//! global maximum).
//!
//! The per-voxel fit is a 3-stage alternating bounded least-squares, repeated
//! `n_inner` times: (1) amplitudes `[C+,C−,C₀]` against the linear signal, then
//! (2) `R2*₀` and (3) `[χ+,χ−]` against `log(signal)` (complex log). The residual
//! is the complex difference packed as `[Re; −Im]`, matching the reference.
//!
//! Each source is reconstructed from the fitted parameters via a per-compartment
//! phase accumulation, `−Σ angle(model) / ((2/3)·γ·B0·ΣTE)`: the paramagnetic
//! sub-model (`pscModel`) gives χ+ and the diamagnetic sub-model (`dscModel`)
//! gives |χ−|. χ+ is returned ≥ 0, χ− ≤ 0.
//!
//! **Output-mapping note.** The QSM-CI reference *comment* claims a para/dia
//! output swap (χ+ = |DSC|); on the qsm-forward phantom that swap anti-correlates
//! with the ground truth, while the physically-consistent mapping used here
//! (χ+ = |PSC|, from the paramagnetic sub-model) correlates strongly. We therefore
//! do not replicate the reference's swap.
//!
//! Reference:
//! Chen, J., et al. (2021). "Decompose quantitative susceptibility mapping (QSM)
//! to sub-voxel diamagnetic and paramagnetic components based on gradient-echo MRI
//! data." NeuroImage 242:118735. https://doi.org/10.1016/j.neuroimage.2021.118735
//! Reference implementation (QSM-CI port of Tim Ho's open MATLAB DECOMPOSE-QSM).

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use std::f64::consts::PI;

/// Gyromagnetic constant used by the DECOMPOSE reference: `42.58 · 2π`.
const GAMMA: f64 = 42.58 * 2.0 * PI;

/// Parameters for [`decompose`].
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct DecomposeParams {
    /// Main field strength in Tesla.
    pub b0: f64,
    /// Number of alternating 3-stage fit passes per voxel (reference default: 10).
    pub n_inner: usize,
    /// Upper bound on |χ| in the fit, ppm (reference default: 0.5).
    pub chi_bound: f64,
    /// Max Levenberg–Marquardt iterations per fit stage (reference lsqcurvefit
    /// defaults to a trust-region solve; 30 is ample for these 1–3 parameter fits).
    pub max_lm_iter: usize,
}

impl Default for DecomposeParams {
    fn default() -> Self {
        Self {
            b0: 7.0,
            n_inner: 10,
            chi_bound: 0.5,
            max_lm_iter: 30,
        }
    }
}

/// DECOMPOSE source separation from a provided QSM and multi-echo magnitude.
///
/// # Arguments
/// * `chi_total` — Conventional QSM χ_total in **ppm** (`n_voxels`), used to
///   synthesize the per-echo phase.
/// * `magnitude` — Multi-echo magnitude, flattened as `(n_voxels, n_echoes)` in
///   row-major order (echo fastest per voxel) — the same layout as
///   [`crate::separation::r2star_qsm_from_magnitude`]. Normalised internally by
///   its global maximum.
/// * `echo_times` — Echo times in **seconds** (`n_echoes`).
/// * `mask` — Binary brain mask (`n_voxels`, 1 = inside).
/// * `params` — See [`DecomposeParams`].
/// * `progress` — Progress callback `(voxels_done, voxels_total)`, called
///   periodically over the fitted (masked) voxels.
///
/// # Returns
/// `(chi_pos, chi_neg, chi_total)` in ppm, restricted to `mask` — matching the
/// [`chi_sep_ilsqr`](super::chi_sep_ilsqr)/[`chi_sep_medi`](super::chi_sep_medi)
/// convention: `chi_pos` ≥ 0 (paramagnetic), `chi_neg` ≤ 0 (diamagnetic, signed),
/// and `chi_total = chi_pos + chi_neg`.
pub fn decompose(
    chi_total: &[f64],
    magnitude: &[f64],
    echo_times: &[f64],
    mask: &[u8],
    params: &DecomposeParams,
    mut progress: impl FnMut(usize, usize),
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = chi_total.len();
    let ne = echo_times.len();
    assert_eq!(
        magnitude.len(),
        n * ne,
        "magnitude must be n_voxels * n_echoes"
    );
    assert_eq!(mask.len(), n, "mask length must match chi_total");
    assert!(ne >= 3, "DECOMPOSE needs at least 3 echoes");

    let b0 = params.b0;
    let te = echo_times;

    // Global-max magnitude normalisation (matches the reference).
    let gmax = magnitude.iter().cloned().fold(0.0_f64, f64::max).max(1e-30);

    // Reconstruction denominator: (2/3)·γ·B0·ΣTE.
    let den = (2.0 / 3.0) * GAMMA * b0 * te.iter().sum::<f64>();

    // Masked voxel indices (only these are fit).
    let voxels: Vec<usize> = (0..n).filter(|&i| mask[i] != 0).collect();
    let total = voxels.len();

    // Per-voxel fit → (index, chi_pos, chi_neg). Embarrassingly parallel.
    let results: Vec<(usize, f64, f64)> = maybe_par_iter!(voxels)
        .map(|&i| {
            // Build the complex data y[e] = mag_norm · exp(-i·(2/3)·χ_total·γ·B0·TE).
            let chi = chi_total[i];
            let mut y = vec![(0.0_f64, 0.0_f64); ne];
            let mut all_zero = true;
            for e in 0..ne {
                let m = magnitude[i * ne + e] / gmax;
                if m > 0.0 {
                    all_zero = false;
                }
                let ph = -(2.0 / 3.0) * chi * GAMMA * b0 * te[e];
                y[e] = (m * ph.cos(), m * ph.sin());
            }
            if all_zero {
                return (i, 0.0, 0.0);
            }
            // log(y) for stages 2 & 3.
            let logy: Vec<(f64, f64)> = y.iter().map(|&z| clog(z)).collect();

            let (cp, cm, c0, r0, chip, chim) = fit_voxel(te, &y, &logy, b0, params);

            // Reconstruct χ+ (from dscModel) and χ− (from pscModel), both as the
            // per-compartment phase accumulation −Σangle/den.
            let dsc = recon_phase(te, b0, den, |t| {
                dsc_model(t, cp, cm, c0, chip, chim, r0, b0)
            });
            let psc = recon_phase(te, b0, den, |t| {
                psc_model(t, cp, cm, c0, chip, chim, r0, b0)
            });
            // pscModel recovers the paramagnetic term (∝ χ+), dscModel the
            // diamagnetic term (∝ |χ−|). We use this physically-correct mapping
            // (χ+ = |PSC|, χ− = −|DSC|), which correlates positively with the
            // qsm-forward phantom's ground truth. Note: the QSM-CI reference
            // *comment* claims the opposite ("χ+ = |DSC|"); that swap anti-
            // correlates with this GT, so we do not replicate it. See module docs.
            (i, psc.abs(), -dsc.abs())
        })
        .collect();

    let mut chi_pos = vec![0.0_f64; n];
    let mut chi_neg = vec![0.0_f64; n];
    let mut chi_out = vec![0.0_f64; n];
    for (i, pos, neg) in results {
        chi_pos[i] = pos;
        chi_neg[i] = neg;
        chi_out[i] = pos + neg;
    }
    progress(total, total);
    (chi_pos, chi_neg, chi_out)
}

/// The three-stage alternating fit for a single voxel. Returns
/// `(C+, C−, C₀, R2*₀, χ+, χ−)`.
#[allow(clippy::too_many_arguments)]
fn fit_voxel(
    te: &[f64],
    y: &[(f64, f64)],
    logy: &[(f64, f64)],
    b0: f64,
    params: &DecomposeParams,
) -> (f64, f64, f64, f64, f64, f64) {
    let ne = te.len();
    // Initial values (reference).
    let mut c = [0.3_f64, 0.3, 0.4]; // [C+, C−, C₀]
    let mut r0 = 25.0_f64;
    let mut chi = [0.05_f64, -0.05]; // [χ+, χ−]

    let ub = params.chi_bound;
    let inf = f64::INFINITY;
    let maxit = params.max_lm_iter;

    for _ in 0..params.n_inner {
        // Stage 1: amplitudes C against the linear signal y.
        {
            let resid = |x: &[f64]| -> Vec<f64> {
                let mut out = Vec::with_capacity(2 * ne);
                let mut im = Vec::with_capacity(ne);
                for e in 0..ne {
                    let m = signal_model(te[e], x[0], x[1], x[2], chi[0], chi[1], r0, b0);
                    out.push(m.0 - y[e].0);
                    im.push(-(m.1 - y[e].1));
                }
                out.extend(im);
                out
            };
            let x = lm_bounded(&resid, &c, &[0.0, 0.0, 0.0], &[inf, inf, inf], maxit);
            c = [x[0], x[1], x[2]];
        }
        // Stage 2: R2*₀ against log(y).
        {
            let resid = |x: &[f64]| -> Vec<f64> {
                let mut out = Vec::with_capacity(2 * ne);
                let mut im = Vec::with_capacity(ne);
                for e in 0..ne {
                    let m = clog(signal_model(
                        te[e], c[0], c[1], c[2], chi[0], chi[1], x[0], b0,
                    ));
                    out.push(m.0 - logy[e].0);
                    im.push(-(m.1 - logy[e].1));
                }
                out.extend(im);
                out
            };
            let x = lm_bounded(&resid, &[r0], &[0.0], &[inf], maxit);
            r0 = x[0];
        }
        // Stage 3: [χ+, χ−] against log(y).
        {
            let resid = |x: &[f64]| -> Vec<f64> {
                let mut out = Vec::with_capacity(2 * ne);
                let mut im = Vec::with_capacity(ne);
                for e in 0..ne {
                    let m = clog(signal_model(te[e], c[0], c[1], c[2], x[0], x[1], r0, b0));
                    out.push(m.0 - logy[e].0);
                    im.push(-(m.1 - logy[e].1));
                }
                out.extend(im);
                out
            };
            let x = lm_bounded(&resid, &chi, &[0.0, -ub], &[ub, 0.0], maxit);
            chi = [x[0], x[1]];
        }
    }
    (c[0], c[1], c[2], r0, chi[0], chi[1])
}

/// Complex DECOMPOSE signal model at echo time `t` (seconds). Returns `(re, im)`.
#[allow(clippy::too_many_arguments)]
#[inline]
fn signal_model(
    t: f64,
    cp: f64,
    cm: f64,
    c0: f64,
    chip: f64,
    chim: f64,
    r0: f64,
    b0: f64,
) -> (f64, f64) {
    let a = (2.0 * PI * GAMMA * b0) / (9.0 * 3.0_f64.sqrt());
    // term1: paramagnetic
    let d1 = a * chip + r0;
    let w1 = (2.0 / 3.0) * chip * GAMMA * b0;
    let t1 = cexp_decay(cp, d1, w1, t);
    // term2: diamagnetic (χ− ≤ 0, so −a·χ− ≥ 0 adds decay)
    let d2 = -a * chim + r0;
    let w2 = (2.0 / 3.0) * chim * GAMMA * b0;
    let t2 = cexp_decay(cm, d2, w2, t);
    // term3: neutral
    let t3 = (c0 * (-r0 * t).exp(), 0.0);
    (t1.0 + t2.0 + t3.0, t1.1 + t2.1 + t3.1)
}

/// Paramagnetic-only reconstruction sub-model (`pscModel.m`).
#[allow(clippy::too_many_arguments)]
#[inline]
fn psc_model(
    t: f64,
    cp: f64,
    cm: f64,
    c0: f64,
    chip: f64,
    _chim: f64,
    r0: f64,
    b0: f64,
) -> (f64, f64) {
    let a = (2.0 * PI * GAMMA * b0) / (9.0 * 3.0_f64.sqrt());
    let d1 = a * chip + r0;
    let w1 = (2.0 / 3.0) * chip * GAMMA * b0;
    let t1 = cexp_decay(cp, d1, w1, t);
    let t3 = ((c0 + cm) * (-r0 * t).exp(), 0.0);
    (t1.0 + t3.0, t1.1 + t3.1)
}

/// Diamagnetic-only reconstruction sub-model (`dscModel.m`). Note the flipped
/// imaginary sign `(2/3)·(−χ−)·γ·B0` relative to [`signal_model`].
#[allow(clippy::too_many_arguments)]
#[inline]
fn dsc_model(
    t: f64,
    cp: f64,
    cm: f64,
    c0: f64,
    _chip: f64,
    chim: f64,
    r0: f64,
    b0: f64,
) -> (f64, f64) {
    let a = (2.0 * PI * GAMMA * b0) / (9.0 * 3.0_f64.sqrt());
    let d2 = -a * chim + r0;
    let w2 = (2.0 / 3.0) * (-chim) * GAMMA * b0; // flipped sign
    let t2 = cexp_decay(cm, d2, w2, t);
    let t3 = ((c0 + cp) * (-r0 * t).exp(), 0.0);
    (t2.0 + t3.0, t2.1 + t3.1)
}

/// `amp · exp(−(decay + i·freq)·t)` as `(re, im)`.
#[inline]
fn cexp_decay(amp: f64, decay: f64, freq: f64, t: f64) -> (f64, f64) {
    let mag = amp * (-decay * t).exp();
    let ang = -freq * t;
    (mag * ang.cos(), mag * ang.sin())
}

/// Complex natural log: `log(z) = log|z| + i·angle(z)`.
#[inline]
fn clog(z: (f64, f64)) -> (f64, f64) {
    let mag2 = z.0 * z.0 + z.1 * z.1;
    (0.5 * mag2.max(1e-300).ln(), z.1.atan2(z.0))
}

/// Reconstruct a source value: `−Σ angle(model(TE)) / den`.
fn recon_phase(te: &[f64], _b0: f64, den: f64, model: impl Fn(f64) -> (f64, f64)) -> f64 {
    let mut s = 0.0;
    for &t in te {
        let z = model(t);
        s += z.1.atan2(z.0);
    }
    -s / den
}

// ---------------------------------------------------------------------------
// Bounded Levenberg–Marquardt for small (n ≤ 3) least-squares problems.
// ---------------------------------------------------------------------------

/// Minimise ‖resid(x)‖² over the box `[lb, ub]` by Levenberg–Marquardt with a
/// forward finite-difference Jacobian and per-step projection onto the box.
/// Sized for the 1–3 parameter DECOMPOSE stages.
fn lm_bounded(
    resid: &dyn Fn(&[f64]) -> Vec<f64>,
    x0: &[f64],
    lb: &[f64],
    ub: &[f64],
    max_iter: usize,
) -> Vec<f64> {
    let n = x0.len();
    let mut x: Vec<f64> = x0
        .iter()
        .zip(lb)
        .zip(ub)
        .map(|((&v, &l), &u)| v.clamp(l, u))
        .collect();
    let mut r = resid(&x);
    let m = r.len();
    let mut cost = dot(&r, &r);
    let mut lambda = 1e-3;
    let ftol = 1e-10;

    for _ in 0..max_iter {
        // Forward-difference Jacobian J (m×n), column-major in `jac`.
        let mut jac = vec![0.0_f64; m * n];
        for j in 0..n {
            let h = 1e-6 * x[j].abs().max(1e-3);
            let mut xp = x.clone();
            xp[j] = (x[j] + h).clamp(lb[j], ub[j]);
            let hj = xp[j] - x[j];
            let step = if hj.abs() < 1e-30 { h } else { hj };
            if hj.abs() < 1e-30 {
                xp[j] = x[j] + h; // allow out-of-box probe when pinned at a bound
            }
            let rp = resid(&xp);
            for k in 0..m {
                jac[j * m + k] = (rp[k] - r[k]) / step;
            }
        }
        // Normal equations A = JᵀJ (n×n), g = Jᵀr (n).
        let mut a = vec![0.0_f64; n * n];
        let mut g = vec![0.0_f64; n];
        for jc in 0..n {
            for jr in 0..n {
                let mut s = 0.0;
                for k in 0..m {
                    s += jac[jr * m + k] * jac[jc * m + k];
                }
                a[jr * n + jc] = s;
            }
            let mut s = 0.0;
            for k in 0..m {
                s += jac[jc * m + k] * r[k];
            }
            g[jc] = s;
        }

        // Inner loop: inflate lambda until a step decreases the cost.
        let mut improved = false;
        for _ in 0..30 {
            let mut al = a.clone();
            for d in 0..n {
                al[d * n + d] += lambda * a[d * n + d].max(1e-12);
            }
            let Some(dx) = solve_small(&al, &g, n) else {
                lambda *= 2.5;
                continue;
            };
            let mut xn = vec![0.0_f64; n];
            for d in 0..n {
                xn[d] = (x[d] - dx[d]).clamp(lb[d], ub[d]);
            }
            let rn = resid(&xn);
            let cn = dot(&rn, &rn);
            if cn < cost {
                let rel = (cost - cn) / cost.max(1e-300);
                x = xn;
                r = rn;
                cost = cn;
                lambda = (lambda * 0.4).max(1e-12);
                improved = true;
                if rel < ftol {
                    return x;
                }
                break;
            } else {
                lambda *= 2.5;
                if lambda > 1e12 {
                    break;
                }
            }
        }
        if !improved {
            break;
        }
    }
    x
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

/// Solve `A x = b` for small `n` (≤ 3) by Gaussian elimination with partial
/// pivoting. `a` is row-major `n×n`. Returns `None` if singular.
fn solve_small(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut m = a.to_vec();
    let mut y = b.to_vec();
    for col in 0..n {
        // Partial pivot.
        let mut piv = col;
        let mut best = m[col * n + col].abs();
        for r in (col + 1)..n {
            let v = m[r * n + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-300 {
            return None;
        }
        if piv != col {
            for c in 0..n {
                m.swap(col * n + c, piv * n + c);
            }
            y.swap(col, piv);
        }
        let d = m[col * n + col];
        for r in (col + 1)..n {
            let f = m[r * n + col] / d;
            if f != 0.0 {
                for c in col..n {
                    m[r * n + c] -= f * m[col * n + c];
                }
                y[r] -= f * y[col];
            }
        }
    }
    let mut x = vec![0.0_f64; n];
    for col in (0..n).rev() {
        let mut s = y[col];
        for c in (col + 1)..n {
            s -= m[col * n + c] * x[c];
        }
        x[col] = s / m[col * n + col];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lm_recovers_linear_fit() {
        // Fit y = m·t + c via LM on residual [m·t+c − data].
        let t = [0.0, 1.0, 2.0, 3.0];
        let data = [1.0, 3.0, 5.0, 7.0]; // m=2, c=1
        let resid = |x: &[f64]| -> Vec<f64> {
            t.iter()
                .zip(data)
                .map(|(&ti, d)| x[0] * ti + x[1] - d)
                .collect()
        };
        let x = lm_bounded(&resid, &[0.0, 0.0], &[-10.0, -10.0], &[10.0, 10.0], 50);
        assert!((x[0] - 2.0).abs() < 1e-6, "slope {}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-6, "intercept {}", x[1]);
    }

    #[test]
    fn solve_small_3x3() {
        // A x = b with a known solution.
        let a = [2.0, 1.0, 1.0, 1.0, 3.0, 2.0, 1.0, 0.0, 0.0];
        let x_true = [1.0, 2.0, 3.0];
        let b = [
            a[0] * x_true[0] + a[1] * x_true[1] + a[2] * x_true[2],
            a[3] * x_true[0] + a[4] * x_true[1] + a[5] * x_true[2],
            a[6] * x_true[0] + a[7] * x_true[1] + a[8] * x_true[2],
        ];
        let x = solve_small(&a, &b, 3).unwrap();
        for i in 0..3 {
            assert!((x[i] - x_true[i]).abs() < 1e-9, "x{i} = {}", x[i]);
        }
    }

    /// Sign/plumbing check on a synthetic voxel. Verifies the output sign
    /// convention and the physically-correct mapping: χ+ = |PSC| (paramagnetic
    /// sub-model), χ− = −|DSC| (diamagnetic sub-model). A voxel with a large
    /// paramagnetic fit amplitude/χ therefore yields a dominant χ+.
    #[test]
    fn decompose_signs_on_synthetic_voxel() {
        let te = [0.004, 0.012, 0.020, 0.028];
        let b0 = 7.0;
        let params = DecomposeParams {
            b0,
            n_inner: 12,
            chi_bound: 0.5,
            max_lm_iter: 40,
        };

        // Forward: paramagnetic-dominant fit parameters.
        let (cp, cm, c0, chip, chim, r0) = (0.5, 0.1, 0.4, 0.10, -0.02, 20.0);
        let sig: Vec<(f64, f64)> = te
            .iter()
            .map(|&t| signal_model(t, cp, cm, c0, chip, chim, r0, b0))
            .collect();
        let chi_total_val = chip + chim;
        let n = 1;
        let ne = te.len();
        let mut mag = vec![0.0_f64; n * ne];
        for e in 0..ne {
            mag[e] = (sig[e].0 * sig[e].0 + sig[e].1 * sig[e].1).sqrt();
        }
        let chi_total = vec![chi_total_val; n];
        let mask = vec![1u8; n];
        let (pos, neg, tot) = decompose(&chi_total, &mag, &te, &mask, &params, |_, _| {});
        // Sign convention + invariant.
        assert!(pos[0] >= 0.0, "χ+ must be ≥ 0");
        assert!(neg[0] <= 0.0, "χ− must be ≤ 0");
        assert!(
            (tot[0] - (pos[0] + neg[0])).abs() < 1e-12,
            "χ_total = χ+ + χ−"
        );
        // Physically-correct mapping: paramagnetic fit → χ+ (|PSC|) dominates.
        assert!(
            pos[0] > neg[0].abs(),
            "expected χ+ dominant: χ+={} χ−={}",
            pos[0],
            neg[0]
        );
    }
}

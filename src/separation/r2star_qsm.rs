//! R2*-QSM: susceptibility source separation from gradient-echo data alone.
//!
//! Dimov et al. separate paramagnetic (χ+, iron) and diamagnetic (χ−, myelin)
//! susceptibility from GRE data alone — R2* from the multi-echo magnitude plus a
//! conventional QSM (χ_total) from the phase — with **no** separate R2/R2'
//! measurement. Per voxel the model is
//!
//! ```text
//!   R2*      = 𝓇·(|χ+| + |χ−|)      (a single relaxometric constant for both sources)
//!   χ_total  = χ+ + χ−               (χ+ ≥ 0, χ− ≤ 0)
//! ```
//!
//! which inverts in closed form to
//!
//! ```text
//!   χ+   = (χ_total + R2*/𝓇) / 2
//!   |χ−| = (R2*/𝓇 − χ_total) / 2
//! ```
//!
//! with the physical constraints `χ+ ≥ 0`, `|χ−| ≥ 0` enforced by clipping.
//!
//! **Relaxometric constant.** Dimov calibrated `𝓇 = 274 Hz/ppm` at 3 T (a single
//! value for both sources). R2* susceptibility-induced decay scales linearly with
//! B0 while χ does not, so at the acquisition field `𝓇 = 274·(B0/3)` Hz/ppm.
//!
//! This is the paper's voxel-level solve; the full method adds a field
//! data-consistency term (auto-satisfied when χ_total is supplied) and an
//! edge-masked L1 regularisation, both omitted here.
//!
//! Reference:
//! Dimov, A.V., et al. (2022). "Magnetic susceptibility source separation solely
//! from gradient echo data: histological validation." Tomography, 8(3):1544-1551
//! (and J. Neuroimaging 2022, https://doi.org/10.1111/jon.13014). R2*
//! susceptibility model: Yablonskiy & Haacke, Magn. Reson. Med. 1994.

/// Parameters for [`r2star_qsm`] / [`r2star_qsm_from_magnitude`].
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct R2starQsmParams {
    /// Main field strength in Tesla (the constant is scaled by `B0/3`).
    pub b0: f64,
    /// Relaxometric constant at 3 T in Hz/ppm (Dimov et al.: 274).
    pub r_const_3t: f64,
}

impl Default for R2starQsmParams {
    fn default() -> Self {
        Self { b0: 3.0, r_const_3t: 274.0 }
    }
}

/// Closed-form R2*-QSM separation from a QSM and an R2* map.
///
/// # Arguments
/// * `chi_total` — Conventional QSM χ_total in **ppm** (`n` voxels).
/// * `r2star` — R2* map in **Hz** (`n` voxels), e.g. from [`crate::r2star::r2star_arlo`].
/// * `mask` — Binary brain mask (`n` voxels, 1 = inside).
/// * `params` — See [`R2starQsmParams`].
///
/// # Returns
/// `(chi_pos, chi_neg, chi_total)` in ppm, restricted to `mask` — matching the
/// [`chi_sep_ilsqr`](super::chi_sep_ilsqr)/[`chi_sep_medi`](super::chi_sep_medi)
/// convention: `chi_pos` ≥ 0 (paramagnetic), `chi_neg` ≤ 0 (diamagnetic, signed),
/// and `chi_total = chi_pos + chi_neg`.
pub fn r2star_qsm(
    chi_total: &[f64],
    r2star: &[f64],
    mask: &[u8],
    params: &R2starQsmParams,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = chi_total.len();
    assert_eq!(r2star.len(), n, "r2star length must match chi_total");
    assert_eq!(mask.len(), n, "mask length must match chi_total");

    // Field-scaled relaxometric constant (Hz/ppm).
    let r = params.r_const_3t * (params.b0 / 3.0);

    let mut chi_pos = vec![0.0_f64; n];
    let mut chi_neg = vec![0.0_f64; n];
    let mut chi_out = vec![0.0_f64; n];
    for i in 0..n {
        if mask[i] == 0 {
            continue;
        }
        let chi = chi_total[i];
        let s = r2star[i] / r; // R2*/𝓇 = |χ+| + |χ−|
        let pos = ((chi + s) / 2.0).max(0.0); // χ+
        let dia_mag = ((s - chi) / 2.0).max(0.0); // |χ−|
        chi_pos[i] = pos;
        chi_neg[i] = -dia_mag; // signed χ− ≤ 0
        chi_out[i] = pos - dia_mag; // χ_total = χ+ + χ−
    }
    (chi_pos, chi_neg, chi_out)
}

/// R2*-QSM separation fitting R2* from multi-echo magnitude, then separating.
///
/// Fits R2* by magnitude-weighted log-linear least squares (weight ∝ magnitude²),
/// matching the QSM-CI reference implementation, then calls [`r2star_qsm`].
///
/// # Arguments
/// * `magnitude` — Multi-echo magnitude, flattened as `(n_voxels, n_echoes)` in
///   row-major order (echo fastest per voxel) — the same layout as
///   [`crate::r2star::r2star_arlo`].
/// * `echo_times` — Echo times in **seconds** (`n_echoes`).
/// * `chi_total` — Conventional QSM χ_total in **ppm** (`n_voxels`).
/// * `mask` — Binary brain mask (`n_voxels`).
/// * `params` — See [`R2starQsmParams`].
///
/// # Returns
/// `(chi_pos, chi_neg, chi_total)` as in [`r2star_qsm`].
pub fn r2star_qsm_from_magnitude(
    magnitude: &[f64],
    echo_times: &[f64],
    chi_total: &[f64],
    mask: &[u8],
    params: &R2starQsmParams,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = chi_total.len();
    let ne = echo_times.len();
    assert_eq!(magnitude.len(), n * ne, "magnitude must be n_voxels * n_echoes");
    let r2star = fit_r2star_weighted_loglinear(magnitude, echo_times, mask, n);
    r2star_qsm(chi_total, &r2star, mask, params)
}

/// R2* (Hz) by magnitude-weighted log-linear least squares over echoes.
///
/// Weight ∝ magnitude² (SNR) so noisy late echoes don't dominate — the QSM-CI
/// reference's robust stand-in for ARLO. `R2* = −slope of log(magnitude) vs TE`,
/// clamped to ≥ 0.
fn fit_r2star_weighted_loglinear(
    magnitude: &[f64],
    echo_times: &[f64],
    mask: &[u8],
    n_voxels: usize,
) -> Vec<f64> {
    let ne = echo_times.len();
    let mut r2s = vec![0.0_f64; n_voxels];
    for v in 0..n_voxels {
        if mask[v] == 0 {
            continue;
        }
        // Weighted means over echoes.
        let mut sw = 0.0;
        let mut swt = 0.0;
        let mut swl = 0.0;
        for e in 0..ne {
            let m = magnitude[v * ne + e].max(1e-9);
            let w = m * m;
            sw += w;
            swt += w * echo_times[e];
            swl += w * m.ln();
        }
        if sw <= 0.0 {
            continue;
        }
        let tbar = swt / sw;
        let lbar = swl / sw;
        // Weighted covariance / variance.
        let mut cov = 0.0;
        let mut var = 0.0;
        for e in 0..ne {
            let m = magnitude[v * ne + e].max(1e-9);
            let w = m * m;
            let dt = echo_times[e] - tbar;
            cov += w * dt * (m.ln() - lbar);
            var += w * dt * dt;
        }
        if var > 0.0 {
            r2s[v] = (-cov / var).max(0.0);
        }
    }
    r2s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_form_recovers_sources() {
        // Build χ_total and R2* from known χ+ / |χ−| via the forward model, then
        // check the closed form inverts them exactly.
        let params = R2starQsmParams { b0: 3.0, r_const_3t: 274.0 };
        let r = params.r_const_3t; // B0 = 3 -> r = 274
        let chi_pos = vec![0.10, 0.05, 0.00, 0.20];
        let chi_neg_mag = vec![0.02, 0.10, 0.15, 0.00]; // |χ−|
        let n = chi_pos.len();
        let chi_total: Vec<f64> = (0..n).map(|i| chi_pos[i] - chi_neg_mag[i]).collect();
        let r2star: Vec<f64> = (0..n).map(|i| r * (chi_pos[i] + chi_neg_mag[i])).collect();
        let mask = vec![1u8; n];

        let (para, neg, total) = r2star_qsm(&chi_total, &r2star, &mask, &params);
        for i in 0..n {
            assert!((para[i] - chi_pos[i]).abs() < 1e-9, "χ+ voxel {i}");
            assert!((-neg[i] - chi_neg_mag[i]).abs() < 1e-9, "|χ−| voxel {i}");
            assert!((total[i] - chi_total[i]).abs() < 1e-9, "χ_total voxel {i}");
            assert!(neg[i] <= 0.0, "χ− must be ≤ 0 at voxel {i}");
        }
    }

    #[test]
    fn field_scaling_and_masking() {
        // At B0 = 6, r doubles, so the same R2* yields half the source magnitude.
        let p3 = R2starQsmParams { b0: 3.0, r_const_3t: 274.0 };
        let p6 = R2starQsmParams { b0: 6.0, r_const_3t: 274.0 };
        let chi_total = vec![0.0, 0.0];
        let r2star = vec![274.0, 274.0];
        let mask = vec![1u8, 0u8];

        let (para3, neg3, _t3) = r2star_qsm(&chi_total, &r2star, &mask, &p3);
        let (para6, _neg6, _t6) = r2star_qsm(&chi_total, &r2star, &mask, &p6);
        // voxel 0: χ_total=0, R2*/r = 1 at 3T -> χ+ = |χ−| = 0.5 (χ− = −0.5)
        assert!((para3[0] - 0.5).abs() < 1e-9);
        assert!((neg3[0] + 0.5).abs() < 1e-9);
        // at 6T r = 548 -> R2*/r = 0.5 -> χ+ = 0.25
        assert!((para6[0] - 0.25).abs() < 1e-9);
        // masked-out voxel stays zero
        assert_eq!(para3[1], 0.0);
        assert_eq!(neg3[1], 0.0);
    }

    #[test]
    fn r2star_fit_from_magnitude() {
        // Mono-exponential decay with known R2*, single voxel.
        let r2star_true = 40.0_f64;
        let s0 = 1000.0_f64;
        let te = vec![0.004, 0.012, 0.020, 0.028];
        let mag: Vec<f64> = te.iter().map(|&t| s0 * (-r2star_true * t).exp()).collect();
        let mask = vec![1u8];
        let r2s = fit_r2star_weighted_loglinear(&mag, &te, &mask, 1);
        assert!((r2s[0] - r2star_true).abs() / r2star_true < 1e-6, "R2* fit {}", r2s[0]);
    }
}

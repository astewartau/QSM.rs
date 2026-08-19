//! HC-ChiSep: hollow-cylinder χ-separation with signal-derived fiber orientation.
//!
//! HC-ChiSep separates paramagnetic (χ+, iron) and diamagnetic (χ−, myelin)
//! susceptibility from a conventional QSM (χ_total), an R2' map, and multi-echo
//! GRE magnitude — deriving white-matter fibre orientation from the magnitude's
//! multi-compartment interference pattern (Wharton & Bowtell hollow-cylinder
//! model) rather than requiring DTI.
//!
//! The pipeline (headline mode) is:
//! 1. **Dr+ self-calibration** — the paramagnetic relaxivity is estimated from the
//!    5th percentile of `R2'/χ_total` over confidently-paramagnetic voxels,
//!    falling back to the field-scaled empirical `137·B0/3` Hz/ppm.
//! 2. **Closed-form two-source solve** — `χ+ = R2'/Dr+`, `χ− = χ+ − χ_total`
//!    (Ridani convention: no diamagnetic relaxivity outside myelinated WM),
//!    clamped to keep χ+ ≥ 0, |χ−| ≥ 0. This is the non-WM branch.
//! 3. **WM-likeness (beat) weighting** — a soft weight `w` from a hollow-cylinder
//!    vs mono-exponential model-selection test on the magnitude decay, gated by
//!    χ_total sign. Voxels whose magnitude shows the multi-compartment "beat" and
//!    are diamagnetic-leaning get `w → 1`.
//! 4. **θ + MWF grid fit** — for supported voxels, fibre angle θ and myelin-water
//!    fraction (MWF) are found by an R2'-anchored grid search over the hollow-
//!    cylinder magnitude library, then spatially regularised.
//! 5. **Separation** — in WM, |χ−| = K_χ·MWF (myelin-content ↔ MWF anchor,
//!    0.038 ppm ↔ MWF 0.12), χ+ = χ_total + |χ−| shrunk toward a self-calibrated
//!    median; the two branches are blended by `w`.
//!
//! All the reference's env-gated optional regularisers (TV/guided joint inversion,
//! R2' denoise, hard constraint) are **off** in headline mode and are omitted here.
//!
//! Inputs are in the project's standard units — χ_total in ppm, R2' in Hz, TE in
//! seconds, B0 in Tesla — and outputs follow the χ-separation sign convention
//! (χ+ ≥ 0, χ− ≤ 0, χ_total = χ+ + χ−).
//!
//! Reference:
//! Wharton, S. & Bowtell, R. (2012). "Fiber orientation-dependent white matter
//! contrast in gradient echo MRI." PNAS 109(45):18559-18564. HC-ChiSep is the
//! QSM-CI submission building on this biophysical model.

use crate::Grid;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use std::f64::consts::PI;

/// Reduced gyromagnetic ratio (Hz/T) used by the hollow-cylinder model.
const GAMMA_BAR: f64 = 42.577e6;

// --- Hollow-cylinder white-matter model constants (Wharton & Bowtell Table 3) ---
const CHI_I: f64 = -0.06e-6; // isotropic susceptibility
const CHI_A: f64 = -0.10e-6; // anisotropic susceptibility
const E_EXCH: f64 = 0.02e-6; // isotropic exchange offset
const G_RATIO: f64 = 0.7;
const T2_M: f64 = 10e-3; // myelin-water T2 (s)
const T2_A: f64 = 64e-3; // axonal-water T2 (s)
const T2_E: f64 = 48e-3; // extra-axonal-water T2 (s)
const F_AXON: f64 = 0.55; // axonal fraction of non-myelin water

// --- Myelin-content anchor & MWF bounds ---
const CHI_NEG_REF: f64 = 0.038; // ppm |χ−| at reference MWF
const MWF_REF: f64 = 0.12;
const MWF_MIN: f64 = 0.03;
const MWF_MAX: f64 = 0.25;

// --- Locked hyperparameters (headline mode) ---
const SMOOTH_CLS: [f64; 2] = [0.75, 1.0]; // classification smoothing sigmas
const SMOOTH_FIT: f64 = 1.0;
const W_CENTER: f64 = 1.5;
const W_SCALE: f64 = 0.3;
const CHI_GATE_C: f64 = 0.01;
const CHI_GATE_S: f64 = 0.01;
const LAM: f64 = 0.7;
const NCONV_SIGMA: f64 = 1.5;
const W_SUPPORT: f64 = 0.02;
const NO_BEAT_FRAC: f64 = 0.01;

/// `K_χ`: ppm of |χ−| per unit MWF (= CHI_NEG_REF / MWF_REF).
const K_CHI: f64 = CHI_NEG_REF / MWF_REF;

// θ and MWF grids.
const NT: usize = 61; // 0 .. 90 step 1.5
const NM: usize = 45; // 0.03 .. 0.25 step 0.005

/// Parameters for [`hc_chisep`].
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct HcChisepParams {
    /// Main field strength in Tesla.
    pub b0: f64,
    /// Spin-echo echo times in **seconds** (empty = no SE evidence used).
    pub se_echo_times: Vec<f64>,
    /// Paramagnetic relaxivity at 3 T in Hz/ppm (empirical Shin 2021: 137).
    pub dr_pos_3t: f64,
    /// R2' bin width (Hz) for the anchored grid search (reference default 0.25).
    pub bin_hz: f64,
}

impl Default for HcChisepParams {
    fn default() -> Self {
        Self {
            b0: 7.0,
            se_echo_times: Vec::new(),
            dr_pos_3t: 137.0,
            bin_hz: 0.25,
        }
    }
}

/// HC-ChiSep source separation from a QSM, an R2' map and multi-echo magnitude.
///
/// # Arguments
/// * `chi_total` — Conventional QSM χ_total in **ppm** (`nx·ny·nz`, column-major).
/// * `r2prime` — R2' map in **Hz** (`nx·ny·nz`).
/// * `magnitude` — Multi-echo GRE magnitude, flattened as `(n_voxels, n_echoes)`
///   in row-major order (echo fastest per voxel). Normalised per voxel internally.
/// * `echo_times` — GRE echo times in **seconds** (`n_echoes`).
/// * `se_magnitude` — Optional multi-echo spin-echo magnitude, `(n_voxels,
///   n_se_echoes)`; used as soft MWF/pool-T2 evidence when present (with
///   `params.se_echo_times`).
/// * `mask` — Binary brain mask (`nx·ny·nz`, 1 = inside).
/// * `grid` — Volume dimensions and voxel sizes.
/// * `params` — See [`HcChisepParams`].
/// * `progress` — Progress callback `(stage_done, stage_total)`.
///
/// # Returns
/// `(chi_pos, chi_neg, chi_total)` in ppm, restricted to `mask` — matching the
/// χ-separation convention: `chi_pos` ≥ 0 (paramagnetic), `chi_neg` ≤ 0
/// (diamagnetic, signed), and `chi_total = chi_pos + chi_neg`.
#[allow(clippy::too_many_arguments)]
pub fn hc_chisep(
    chi_total: &[f64],
    r2prime: &[f64],
    magnitude: &[f64],
    echo_times: &[f64],
    se_magnitude: Option<&[f64]>,
    mask: &[u8],
    grid: &Grid,
    params: &HcChisepParams,
    mut progress: impl FnMut(usize, usize),
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let dims = grid.dims;
    let n = dims.0 * dims.1 * dims.2;
    let ne = echo_times.len();
    assert_eq!(chi_total.len(), n, "chi_total length must match grid");
    assert_eq!(r2prime.len(), n, "r2prime length must match grid");
    assert_eq!(
        magnitude.len(),
        n * ne,
        "magnitude must be n_voxels * n_echoes"
    );
    assert_eq!(mask.len(), n, "mask length must match grid");

    let b0 = params.b0;

    // Convert the voxel-major public inputs (i*ne + e) to echo-major contiguous
    // per-echo volumes (e*n + i) so the spatial filters and gathers are simple.
    let mag_em = to_echo_major(magnitude, n, ne);
    let se_e_len = params.se_echo_times.len();
    let se_em: Option<Vec<f64>> = match se_magnitude {
        Some(s) if se_e_len >= 2 && s.len() == n * se_e_len => Some(to_echo_major(s, n, se_e_len)),
        _ => None,
    };

    let n_stages = 5;
    let mut stage = 0usize;
    let mut tick = |s: &mut usize, p: &mut dyn FnMut(usize, usize)| {
        *s += 1;
        p(*s, n_stages);
    };

    // --- Stage 1: Dr+ self-calibration ---------------------------------------
    let dr_default = params.dr_pos_3t * b0 / 3.0;
    let dr_pos = {
        let ratios: Vec<f64> = (0..n)
            .filter(|&i| mask[i] != 0 && chi_total[i] > 0.02)
            .map(|i| r2prime[i] / chi_total[i])
            .collect();
        if ratios.len() > 5000 {
            let p5 = percentile(&ratios, 5.0);
            if (0.3 * dr_default..=3.0 * dr_default).contains(&p5) {
                p5
            } else {
                dr_default
            }
        } else {
            dr_default
        }
    };
    tick(&mut stage, &mut progress);

    // --- Stage 2: closed-form two-source solve (non-WM branch) ---------------
    // cf_pos = clip(R2'/Dr+, 0), cf_neg = clip(cf_pos - χ_total, 0) with the
    // χ_total-consistency fix where the pair is inconsistent. cf_neg is a
    // positive magnitude here.
    let mut cf_pos = vec![0.0_f64; n];
    let mut cf_neg = vec![0.0_f64; n];
    for i in 0..n {
        let p = (r2prime[i] / dr_pos).max(0.0);
        let neg = p - chi_total[i];
        if neg < 0.0 {
            cf_pos[i] = chi_total[i].max(0.0);
        } else {
            cf_pos[i] = p;
        }
        cf_neg[i] = (cf_pos[i] - chi_total[i]).max(0.0);
    }
    tick(&mut stage, &mut progress);

    // Build the anchored fitter (hollow-cylinder magnitude library).
    let fitter = AnchoredFitter::new(echo_times, b0, &params.se_echo_times);
    let have_se = se_em.is_some() && fitter.seml.is_some();
    let se_e = params.se_echo_times.len();

    // Brain voxel indices.
    let bii: Vec<usize> = (0..n).filter(|&i| mask[i] != 0).collect();
    if bii.is_empty() {
        return finish(&cf_pos, &cf_neg, mask, n);
    }
    let rho_b: Vec<f64> = bii.iter().map(|&i| r2prime[i]).collect();

    // --- Stage 3: WM-likeness (beat) weighting -------------------------------
    // For each classification smoothing scale, fit the HC library and compare its
    // SSE to a mono-exponential fit; take the elementwise-best log-ratio.
    let maskf: Vec<f64> = mask.iter().map(|&m| m as f64).collect();
    let mut lr_best: Option<Vec<f64>> = None;
    for &s in &SMOOTH_CLS {
        let m_s = smooth_stack(&mag_em, dims, ne, s);
        let sn = gather_normalised(&m_s, &bii, ne);
        let (_t, _m, sse_hc) = fitter.fit(&sn, &rho_b, None, None, params.bin_hz);
        let mut num = sse_hc;
        let mut den: Vec<f64> = (0..bii.len())
            .map(|v| mono_sse(&sn[v * ne..v * ne + ne], echo_times))
            .collect();
        if have_se {
            let se_s = smooth_stack(se_em.as_ref().unwrap(), dims, se_e, s);
            let sen = gather_normalised(&se_s, &bii, se_e);
            let seml = fitter.seml.as_ref().unwrap();
            for v in 0..bii.len() {
                // min over MWF of SE pool SSE
                let mut best = f64::INFINITY;
                for mi in 0..NM {
                    let mut acc = 0.0;
                    for e in 0..se_e {
                        let d = sen[v * se_e + e] - seml[mi * se_e + e];
                        acc += d * d;
                    }
                    best = best.min(acc);
                }
                num[v] += best;
                den[v] += mono_sse(&sen[v * se_e..v * se_e + se_e], &params.se_echo_times);
            }
        }
        let lr: Vec<f64> = (0..bii.len())
            .map(|v| (num[v] / den[v].max(1e-12)).max(1e-6).log10())
            .collect();
        lr_best = Some(match lr_best {
            None => lr,
            Some(prev) => (0..bii.len()).map(|v| prev[v].min(lr[v])).collect(),
        });
    }
    let lr_best = lr_best.unwrap();

    // lr_med = median_filter(volmap(lr_best, fill=10.0), 3)
    let lr_vol = volmap(&lr_best, &bii, n, 10.0);
    let lr_med = median_filter_3(&lr_vol, dims);

    // chs = NC-smoothed χ_total (gaussian(χ·mask,1)/gaussian(mask,1)).
    let chi_masked: Vec<f64> = (0..n).map(|i| chi_total[i] * maskf[i]).collect();
    let chs_num = gaussian_filter_3d(&chi_masked, dims, 1.0);
    let chs_den = gaussian_filter_3d(&maskf, dims, 1.0);
    let chs: Vec<f64> = (0..n).map(|i| chs_num[i] / chs_den[i].max(1e-6)).collect();

    // w = sigmoid(lr) * sigmoid(chi), masked.
    let mut w = vec![0.0_f64; n];
    for i in 0..n {
        let s1 = 1.0 / (1.0 + ((lr_med[i] - W_CENTER) / W_SCALE).exp());
        let s2 = 1.0 / (1.0 + ((chs[i] - CHI_GATE_C) / CHI_GATE_S).exp());
        w[i] = s1 * s2 * maskf[i];
    }
    let beat_frac = {
        let cnt = bii.iter().filter(|&&i| w[i] > 0.5).count();
        cnt as f64 / bii.len() as f64
    };
    tick(&mut stage, &mut progress);
    if beat_frac < NO_BEAT_FRAC {
        // No detectable beat anywhere → closed-form everywhere.
        return finish(&cf_pos, &cf_neg, mask, n);
    }

    // --- Stage 4: θ + MWF grid fit on supported voxels -----------------------
    let sii: Vec<usize> = bii.iter().cloned().filter(|&i| w[i] > W_SUPPORT).collect();
    let rho_s: Vec<f64> = sii.iter().map(|&i| r2prime[i]).collect();

    let m_fit = smooth_stack(&mag_em, dims, ne, SMOOTH_FIT);
    let sn_sm = gather_normalised(&m_fit, &sii, ne);
    let sn_raw = gather_normalised(&mag_em, &sii, ne);
    let (sen_sm, sen_raw) = if have_se {
        let se = se_em.as_ref().unwrap();
        let se_fit = smooth_stack(se, dims, se_e, SMOOTH_FIT);
        (
            Some(gather_normalised(&se_fit, &sii, se_e)),
            Some(gather_normalised(se, &sii, se_e)),
        )
    } else {
        (None, None)
    };

    // θ from the smoothed magnitude, spatially regularised (median, fill 45°).
    let (th_sm, _m_sm, _s_sm) = fitter.fit(&sn_sm, &rho_s, sen_sm.as_deref(), None, params.bin_hz);
    // Non-supported voxels filled with 45° before the median (matches the ref).
    let th_fill = volmap(&th_sm, &sii, n, 45.0);
    let th_reg = median_filter_3(&th_fill, dims);
    let th_pin: Vec<f64> = sii.iter().map(|&i| th_reg[i]).collect();

    // MWF from raw and smoothed magnitude, θ pinned; average with the physics bound.
    let (_t1, mw_raw, _s1) = fitter.fit(
        &sn_raw,
        &rho_s,
        sen_raw.as_deref(),
        Some(&th_pin),
        params.bin_hz,
    );
    let (_t2, mw_sm, _s2) = fitter.fit(
        &sn_sm,
        &rho_s,
        sen_sm.as_deref(),
        Some(&th_pin),
        params.bin_hz,
    );
    let mw_est: Vec<f64> = (0..sii.len())
        .map(|v| {
            let bnd = (-chi_total[sii[v]] / K_CHI).clamp(MWF_MIN, MWF_MAX);
            0.5 * (mw_raw[v].max(bnd) + mw_sm[v].max(bnd))
        })
        .collect();
    let mw_vol = volmap(&mw_est, &sii, n, 0.0);

    // Confidence-weighted normalised convolution of MWF.
    let wv: Vec<f64> = w.iter().map(|&x| x.clamp(0.0, 1.0)).collect();
    let mw_w: Vec<f64> = (0..n).map(|i| mw_vol[i] * wv[i]).collect();
    let mw_num = gaussian_filter_3d(&mw_w, dims, NCONV_SIGMA);
    let mw_den = gaussian_filter_3d(&wv, dims, NCONV_SIGMA);
    let mw_reg: Vec<f64> = (0..n).map(|i| mw_num[i] / mw_den[i].max(1e-3)).collect();
    tick(&mut stage, &mut progress);

    // --- Stage 5: separation + blend -----------------------------------------
    let route: Vec<f64> = (0..n).map(|i| chi_total[i] + K_CHI * mw_reg[i]).collect();
    let conf: Vec<usize> = (0..n).filter(|&i| w[i] > 0.5).collect();
    let c0 = if conf.len() > 100 {
        let vals: Vec<f64> = conf.iter().map(|&i| route[i].max(0.0)).collect();
        median(&vals)
    } else {
        0.005
    };
    let mut chi_pos = vec![0.0_f64; n];
    let mut chi_neg = vec![0.0_f64; n];
    let mut chi_out = vec![0.0_f64; n];
    for i in 0..n {
        if mask[i] == 0 {
            continue;
        }
        let wm_pos = (LAM * route[i] + (1.0 - LAM) * c0).max(0.0);
        let wm_neg = (wm_pos - chi_total[i]).max(0.0);
        let pos = w[i] * wm_pos + (1.0 - w[i]) * cf_pos[i];
        let neg = w[i] * wm_neg + (1.0 - w[i]) * cf_neg[i];
        chi_pos[i] = pos;
        chi_neg[i] = -neg; // signed χ− ≤ 0
        chi_out[i] = pos - neg;
    }
    tick(&mut stage, &mut progress);
    (chi_pos, chi_neg, chi_out)
}

/// Emit the closed-form branch as the final result (χ+ ≥ 0, χ− ≤ 0 signed).
fn finish(cf_pos: &[f64], cf_neg: &[f64], mask: &[u8], n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut chi_pos = vec![0.0_f64; n];
    let mut chi_neg = vec![0.0_f64; n];
    let mut chi_out = vec![0.0_f64; n];
    for i in 0..n {
        if mask[i] != 0 {
            chi_pos[i] = cf_pos[i];
            chi_neg[i] = -cf_neg[i];
            chi_out[i] = cf_pos[i] - cf_neg[i];
        }
    }
    (chi_pos, chi_neg, chi_out)
}

// ---------------------------------------------------------------------------
// Hollow-cylinder model + anchored grid-search fitter.
// ---------------------------------------------------------------------------

/// Compartment frequencies (Hz) for fibre-to-B0 angle `theta` (radians).
/// Returns `(Δf_myelin, Δf_axon, Δf_extra)`; extra-axonal is exactly 0.
fn hc_compartment_freqs(theta: f64, b0: f64) -> (f64, f64) {
    let s2 = theta.sin() * theta.sin();
    let w0 = GAMMA_BAR * b0;
    let ln_term = 0.75 * CHI_A * (1.0 / G_RATIO).ln() * s2;
    let f_my = w0
        * (CHI_I * (2.0 / 3.0 - s2) / 2.0
            + CHI_A * (1.0 / 12.0 - 5.0 / 12.0 * s2)
            + ln_term
            + E_EXCH);
    let f_ax = w0 * ln_term;
    (f_my, f_ax)
}

/// |complex hollow-cylinder GRE magnitude| at echo time `te` (s), angle `theta`
/// (rad), myelin-water fraction `mwf`, and mesoscopic reversible rate `r2p_meso`.
fn hc_wm_signal_mag(te: f64, theta: f64, b0: f64, mwf: f64, r2p_meso: f64) -> f64 {
    let f_m = mwf;
    let rest = 1.0 - f_m;
    let f_a = rest * F_AXON;
    let f_e = rest * (1.0 - F_AXON);
    let (dfm, dfa) = hc_compartment_freqs(theta, b0);
    let (mut re, mut im) = (0.0, 0.0);
    let terms = [(f_m, T2_M, dfm), (f_a, T2_A, dfa), (f_e, T2_E, 0.0)];
    for (f, t2, df) in terms {
        let amp = f * (-te / t2).exp();
        let ph = 2.0 * PI * df * te;
        re += amp * ph.cos();
        im += amp * ph.sin();
    }
    let mag = (re * re + im * im).sqrt();
    mag * (-r2p_meso * te).exp()
}

/// Spin-echo WM magnitude factor (offsets refocused → real pool-T2 mixture).
fn hc_wm_se_signal(te: f64, mwf: f64) -> f64 {
    let rest = 1.0 - mwf;
    mwf * (-te / T2_M).exp()
        + rest * F_AXON * (-te / T2_A).exp()
        + rest * (1.0 - F_AXON) * (-te / T2_E).exp()
}

/// Mono-exp-equivalent reversible rate the beat itself contributes (Hz):
/// weighted-LS slope of `log(|GRE|/SE)` vs centred TE, negated and clamped ≥ 0.
fn hc_wm_r2prime(theta: f64, mwf: f64, tes: &[f64], b0: f64) -> f64 {
    let tbar = tes.iter().sum::<f64>() / tes.len() as f64;
    let denom: f64 = tes.iter().map(|&t| (t - tbar) * (t - tbar)).sum();
    let mut acc = 0.0;
    for &te in tes {
        let gre = hc_wm_signal_mag(te, theta, b0, mwf, 0.0);
        let se = hc_wm_se_signal(te, mwf).max(1e-12);
        let ratio = (gre / se).max(1e-12);
        acc += ratio.ln() * (te - tbar);
    }
    (-acc / denom).max(0.0)
}

/// Precomputed hollow-cylinder magnitude library over the (θ, MWF) grid.
struct AnchoredFitter {
    e: usize,
    /// `Pn[ (ti*NM + mi)*E + e ]` — first-echo-normalised magnitude library.
    pn: Vec<f64>,
    /// `H[ ti*NM + mi ]` — R2'_hc per grid point (Hz).
    h: Vec<f64>,
    /// `dte[e] = TE[e] - TE[0]`.
    dte: Vec<f64>,
    /// Optional SE library `SEml[ mi*se_e + e ]` (first-SE-normalised).
    seml: Option<Vec<f64>>,
}

impl AnchoredFitter {
    fn new(tes: &[f64], b0: f64, se_tes: &[f64]) -> Self {
        let e = tes.len();
        let mut pn = vec![0.0_f64; NT * NM * e];
        let mut h = vec![0.0_f64; NT * NM];
        for ti in 0..NT {
            let theta = (ti as f64 * 1.5).to_radians();
            for mi in 0..NM {
                let mwf = MWF_MIN + mi as f64 * 0.005;
                let base = (ti * NM + mi) * e;
                let first = hc_wm_signal_mag(tes[0], theta, b0, mwf, 0.0).max(1e-30);
                for (k, &te) in tes.iter().enumerate() {
                    pn[base + k] = hc_wm_signal_mag(te, theta, b0, mwf, 0.0) / first;
                }
                h[ti * NM + mi] = hc_wm_r2prime(theta, mwf, tes, b0);
            }
        }
        let dte: Vec<f64> = tes.iter().map(|&t| t - tes[0]).collect();
        let seml = if se_tes.len() >= 2 {
            let se_e = se_tes.len();
            let mut m = vec![0.0_f64; NM * se_e];
            for mi in 0..NM {
                let mwf = MWF_MIN + mi as f64 * 0.005;
                let first = hc_wm_se_signal(se_tes[0], mwf).max(1e-30);
                for (k, &te) in se_tes.iter().enumerate() {
                    m[mi * se_e + k] = hc_wm_se_signal(te, mwf) / first;
                }
            }
            Some(m)
        } else {
            None
        };
        Self {
            e,
            pn,
            h,
            dte,
            seml,
        }
    }

    /// Anchored grid search. `sig_n` is `(N, E)` first-echo-normalised; `rho` is
    /// R2' per voxel (Hz). Returns `(theta_deg, mwf, sse)` per voxel.
    fn fit(
        &self,
        sig_n: &[f64],
        rho: &[f64],
        se_n: Option<&[f64]>,
        theta_pin: Option<&[f64]>,
        bin_hz: f64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let e = self.e;
        let n = rho.len();
        let l = NT * NM;

        // Per-voxel SE SSE (N, NM), if SE evidence is present.
        let se_e = se_n.map(|s| s.len() / n).unwrap_or(0);
        let sse_se: Option<Vec<f64>> = match (se_n, &self.seml) {
            (Some(sn), Some(seml)) => {
                let mut out = vec![0.0_f64; n * NM];
                for v in 0..n {
                    for mi in 0..NM {
                        let mut acc = 0.0;
                        for k in 0..se_e {
                            let d = sn[v * se_e + k] - seml[mi * se_e + k];
                            acc += d * d;
                        }
                        out[v * NM + mi] = acc;
                    }
                }
                Some(out)
            }
            _ => None,
        };

        // Group voxel indices by rho bin.
        use std::collections::HashMap;
        let mut groups: HashMap<i64, Vec<usize>> = HashMap::new();
        for (v, &r) in rho.iter().enumerate() {
            let b = (r.clamp(0.0, 200.0) / bin_hz).round() as i64;
            groups.entry(b).or_default().push(v);
        }

        let group_vec: Vec<(i64, Vec<usize>)> = groups.into_iter().collect();
        // Solve each bin (parallel), scatter results back.
        let per_bin: Vec<Vec<(usize, f64, f64, f64)>> = maybe_par_iter!(group_vec)
            .map(|(b, vs)| {
                // Candidate curves for this bin: Pn · exp(-max(b·bin_hz - H, 0)·dte).
                let mut flat = vec![0.0_f64; l * e];
                let mut m2 = vec![0.0_f64; l];
                let mut bad = vec![false; l];
                let rho_bin = *b as f64 * bin_hz;
                for li in 0..l {
                    let meso = rho_bin - self.h[li];
                    bad[li] = meso < -1.0;
                    let decay = meso.max(0.0);
                    let mut acc = 0.0;
                    for k in 0..e {
                        let val = self.pn[li * e + k] * (-decay * self.dte[k]).exp();
                        flat[li * e + k] = val;
                        acc += val * val;
                    }
                    m2[li] = 0.5 * acc;
                }
                let all_bad = bad.iter().all(|&x| x);

                let mut out = Vec::with_capacity(vs.len());
                for &v in vs {
                    let sig = &sig_n[v * e..v * e + e];
                    let se_row = sse_se.as_ref().map(|s| &s[v * NM..v * NM + NM]);
                    let pin_row = theta_pin.map(|tp| {
                        ((tp[v] / 1.5).round() as isize).clamp(0, NT as isize - 1) as usize
                    });

                    let (mut best_score, mut best_l) = (f64::NEG_INFINITY, 0usize);
                    let (lo, hi) = match pin_row {
                        Some(r) => (r * NM, r * NM + NM),
                        None => (0, l),
                    };
                    for li in lo..hi {
                        if bad[li] && !all_bad {
                            continue;
                        }
                        let mut s = -m2[li];
                        for k in 0..e {
                            s += sig[k] * flat[li * e + k];
                        }
                        if let Some(se_row) = se_row {
                            s -= 0.5 * se_row[li % NM];
                        }
                        if s > best_score {
                            best_score = s;
                            best_l = li;
                        }
                    }
                    // True SSE at the winner.
                    let mut sse = 0.0;
                    for k in 0..e {
                        let d = sig[k] - flat[best_l * e + k];
                        sse += d * d;
                    }
                    let theta_deg = (best_l / NM) as f64 * 1.5;
                    let mwf = MWF_MIN + (best_l % NM) as f64 * 0.005;
                    out.push((v, theta_deg, mwf, sse));
                }
                out
            })
            .collect();

        let mut out_t = vec![0.0_f64; n];
        let mut out_m = vec![MWF_REF; n];
        let mut out_s = vec![f64::INFINITY; n];
        for bin in per_bin {
            for (v, t, m, s) in bin {
                out_t[v] = t;
                out_m[v] = m;
                out_s[v] = s;
            }
        }
        (out_t, out_m, out_s)
    }
}

/// Mono-exponential fit SSE for one voxel's first-echo-normalised signal.
fn mono_sse(sig: &[f64], tes: &[f64]) -> f64 {
    let e = tes.len();
    let tbar = tes.iter().sum::<f64>() / e as f64;
    let logs: Vec<f64> = sig.iter().map(|&s| s.max(1e-9).ln()).collect();
    let den: f64 = tes.iter().map(|&t| (t - tbar) * (t - tbar)).sum();
    let b: f64 = (0..e).map(|k| logs[k] * (tes[k] - tbar)).sum::<f64>() / den;
    let a: f64 = logs.iter().sum::<f64>() / e as f64;
    (0..e)
        .map(|k| {
            let pred = (a + b * (tes[k] - tbar)).exp();
            let d = sig[k] - pred;
            d * d
        })
        .sum()
}

/// Convert a voxel-major multi-echo stack `vm[i*ne + e]` to echo-major
/// (contiguous per-echo volumes) `em[e*n + i]`.
fn to_echo_major(vm: &[f64], n: usize, ne: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; n * ne];
    for i in 0..n {
        for e in 0..ne {
            out[e * n + i] = vm[i * ne + e];
        }
    }
    out
}

/// Gather `(len, E)` first-echo-normalised signals for the given voxel indices
/// from an echo-major volume stack `stack[e*n + i]`.
fn gather_normalised(stack: &[f64], idx: &[usize], ne: usize) -> Vec<f64> {
    let n = stack.len() / ne;
    let mut out = vec![0.0_f64; idx.len() * ne];
    for (k, &i) in idx.iter().enumerate() {
        let first = stack[i].max(1e-9); // e = 0
        for e in 0..ne {
            out[k * ne + e] = stack[e * n + i] / first;
        }
    }
    out
}

/// Scatter per-voxel values back into a full volume, filling the rest with `fill`.
fn volmap(vals: &[f64], idx: &[usize], n: usize, fill: f64) -> Vec<f64> {
    let mut out = vec![fill; n];
    for (k, &i) in idx.iter().enumerate() {
        out[i] = vals[k];
    }
    out
}

// ---------------------------------------------------------------------------
// Spatial filters (scipy-matching: reflect boundary).
// ---------------------------------------------------------------------------

/// Reflect an index into `[0, len)` using scipy's `reflect` mode (edge repeated).
#[inline]
fn reflect_index(mut p: isize, len: usize) -> usize {
    let l = len as isize;
    loop {
        if p < 0 {
            p = -p - 1;
        } else if p >= l {
            p = 2 * l - p - 1;
        } else {
            return p as usize;
        }
    }
}

/// Smooth each echo of an echo-major stack `stack[e*n + i]` by a 3D Gaussian.
fn smooth_stack(stack: &[f64], dims: (usize, usize, usize), ne: usize, sigma: f64) -> Vec<f64> {
    let n = dims.0 * dims.1 * dims.2;
    let mut out = vec![0.0_f64; n * ne];
    for e in 0..ne {
        let sm = gaussian_filter_3d(&stack[e * n..e * n + n], dims, sigma);
        out[e * n..e * n + n].copy_from_slice(&sm);
    }
    out
}

/// 3D separable Gaussian filter, scipy `gaussian_filter` semantics: `reflect`
/// boundary, `truncate = 4.0`.
fn gaussian_filter_3d(data: &[f64], dims: (usize, usize, usize), sigma: f64) -> Vec<f64> {
    if sigma <= 0.0 {
        return data.to_vec();
    }
    let (nx, ny, nz) = dims;
    let radius = (4.0 * sigma + 0.5) as usize;
    let mut kernel = vec![0.0_f64; 2 * radius + 1];
    let mut ksum = 0.0;
    for (k, w) in kernel.iter_mut().enumerate() {
        let x = k as f64 - radius as f64;
        *w = (-0.5 * (x / sigma) * (x / sigma)).exp();
        ksum += *w;
    }
    for w in kernel.iter_mut() {
        *w /= ksum;
    }

    let idx = |x: usize, y: usize, z: usize| (z * ny + y) * nx + x;
    let mut a = data.to_vec();
    let mut b = vec![0.0_f64; a.len()];

    // X axis.
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let mut s = 0.0;
                for (kk, &w) in kernel.iter().enumerate() {
                    let xi = reflect_index(x as isize + kk as isize - radius as isize, nx);
                    s += w * a[idx(xi, y, z)];
                }
                b[idx(x, y, z)] = s;
            }
        }
    }
    std::mem::swap(&mut a, &mut b);
    // Y axis.
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let mut s = 0.0;
                for (kk, &w) in kernel.iter().enumerate() {
                    let yi = reflect_index(y as isize + kk as isize - radius as isize, ny);
                    s += w * a[idx(x, yi, z)];
                }
                b[idx(x, y, z)] = s;
            }
        }
    }
    std::mem::swap(&mut a, &mut b);
    // Z axis.
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let mut s = 0.0;
                for (kk, &w) in kernel.iter().enumerate() {
                    let zi = reflect_index(z as isize + kk as isize - radius as isize, nz);
                    s += w * a[idx(x, y, zi)];
                }
                b[idx(x, y, z)] = s;
            }
        }
    }
    b
}

/// 3D median filter with a 3×3×3 window (scipy `median_filter` size 3, `reflect`).
fn median_filter_3(data: &[f64], dims: (usize, usize, usize)) -> Vec<f64> {
    let (nx, ny, nz) = dims;
    let idx = |x: usize, y: usize, z: usize| (z * ny + y) * nx + x;
    let mut out = vec![0.0_f64; data.len()];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let mut win = [0.0_f64; 27];
                let mut c = 0;
                for dz in -1..=1_isize {
                    let zi = reflect_index(z as isize + dz, nz);
                    for dy in -1..=1_isize {
                        let yi = reflect_index(y as isize + dy, ny);
                        for dx in -1..=1_isize {
                            let xi = reflect_index(x as isize + dx, nx);
                            win[c] = data[idx(xi, yi, zi)];
                            c += 1;
                        }
                    }
                }
                win.sort_by(|a, b| a.partial_cmp(b).unwrap());
                out[idx(x, y, z)] = win[13];
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Small statistics helpers.
// ---------------------------------------------------------------------------

/// numpy-style linear-interpolation percentile (`q` in [0, 100]).
fn percentile(vals: &[f64], q: f64) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    let mut v = vals.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let rank = q / 100.0 * (v.len() as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        v[lo]
    } else {
        let frac = rank - lo as f64;
        v[lo] * (1.0 - frac) + v[hi] * frac
    }
}

fn median(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    let mut v = vals.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = v.len() / 2;
    if v.len() % 2 == 1 {
        v[m]
    } else {
        0.5 * (v[m - 1] + v[m])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_index_matches_scipy() {
        // scipy 'reflect' (edge repeated): [-2,-1,0,1,2,3,4,5] over len=4
        // -> [1,0,0,1,2,3,3,2]
        let len = 4;
        let got: Vec<usize> = (-2..6).map(|p| reflect_index(p, len)).collect();
        assert_eq!(got, vec![1, 0, 0, 1, 2, 3, 3, 2]);
    }

    #[test]
    fn gaussian_preserves_constant() {
        let dims = (8, 8, 8);
        let n = 8 * 8 * 8;
        let data = vec![3.0_f64; n];
        let out = gaussian_filter_3d(&data, dims, 1.0);
        for &v in &out {
            assert!((v - 3.0).abs() < 1e-9, "constant not preserved: {v}");
        }
    }

    #[test]
    fn median_removes_impulse() {
        let dims = (5, 5, 5);
        let n = 125;
        let mut data = vec![1.0_f64; n];
        let idx = |x: usize, y: usize, z: usize| (z * 5 + y) * 5 + x;
        data[idx(2, 2, 2)] = 100.0; // single impulse
        let out = median_filter_3(&data, dims);
        assert!(
            (out[idx(2, 2, 2)] - 1.0).abs() < 1e-9,
            "impulse survived median"
        );
    }

    #[test]
    fn percentile_and_median_basic() {
        let v = [1.0, 2.0, 3.0, 4.0];
        assert!((percentile(&v, 0.0) - 1.0).abs() < 1e-12);
        assert!((percentile(&v, 100.0) - 4.0).abs() < 1e-12);
        assert!((median(&v) - 2.5).abs() < 1e-12);
        let v2 = [1.0, 2.0, 3.0];
        assert!((median(&v2) - 2.0).abs() < 1e-12);
    }

    /// The R2'-anchored fitter should recover a planted (θ, MWF) from a synthetic
    /// hollow-cylinder magnitude, given the matching R2'.
    #[test]
    fn fitter_recovers_planted_theta_mwf() {
        let tes = [0.004, 0.012, 0.020, 0.028];
        let b0 = 7.0;
        let fitter = AnchoredFitter::new(&tes, b0, &[]);

        let theta_deg = 30.0_f64;
        let mwf = 0.10_f64;
        let theta = theta_deg.to_radians();
        // R2' = H(θ,mwf) + a chosen mesoscopic rate.
        let meso = 8.0;
        let h = hc_wm_r2prime(theta, mwf, &tes, b0);
        let rho = h + meso;
        // Build the first-echo-normalised magnitude with that meso decay.
        let first = hc_wm_signal_mag(tes[0], theta, b0, mwf, meso);
        let sig: Vec<f64> = tes
            .iter()
            .map(|&t| hc_wm_signal_mag(t, theta, b0, mwf, meso) / first)
            .collect();

        let (t, m, _s) = fitter.fit(&sig, &[rho], None, None, 0.25);
        // MWF is well-determined; θ is only weakly identifiable from magnitude at a
        // few echoes (near-ties across neighbouring angles), which is why the full
        // algorithm heavily regularises and pins θ — so tolerate a loose θ here.
        assert!((m[0] - mwf).abs() <= 0.02, "mwf {} vs {}", m[0], mwf);
        assert!(
            (t[0] - theta_deg).abs() <= 12.0,
            "theta {} vs {}",
            t[0],
            theta_deg
        );
    }

    /// Build a synthetic phantom: voxels with `x < wm_x_max` are WM-like (hollow-
    /// cylinder beat magnitude, diamagnetic χ_total, R2' = H+meso so the anchored
    /// fit matches), the rest GM-like (mono-exponential magnitude, paramagnetic
    /// χ_total, R2' = Dr+·χ). Returns `(chi_total, r2prime, mag_vm, se_mag_vm, mask)`
    /// with magnitudes in voxel-major `(n, ne)` layout.
    fn synth_phantom(
        dims: (usize, usize, usize),
        wm_x_max: usize,
        b0: f64,
        tes: &[f64],
        se_tes: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<u8>) {
        let (nx, ny, nz) = dims;
        let n = nx * ny * nz;
        let ne = tes.len();
        let se_e = se_tes.len();
        let dr = 137.0 * b0 / 3.0;
        let theta = 40.0_f64.to_radians();
        let mwf = 0.12;
        let meso = 6.0;
        let h = hc_wm_r2prime(theta, mwf, tes, b0);
        let r2star = 25.0;

        let mut chi = vec![0.0_f64; n];
        let mut r2p = vec![0.0_f64; n];
        let mut mag = vec![0.0_f64; n * ne];
        let mut se = vec![0.0_f64; n * se_e];
        let mask = vec![1u8; n];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = (k * ny + j) * nx + i;
                    if i < wm_x_max {
                        chi[idx] = -0.03;
                        r2p[idx] = h + meso;
                        for (e, &t) in tes.iter().enumerate() {
                            mag[idx * ne + e] = hc_wm_signal_mag(t, theta, b0, mwf, meso);
                        }
                        for (e, &t) in se_tes.iter().enumerate() {
                            se[idx * se_e + e] = hc_wm_se_signal(t, mwf);
                        }
                    } else {
                        chi[idx] = 0.05;
                        r2p[idx] = dr * 0.05;
                        for (e, &t) in tes.iter().enumerate() {
                            mag[idx * ne + e] = (-r2star * t).exp();
                        }
                        for (e, &t) in se_tes.iter().enumerate() {
                            se[idx * se_e + e] = (-r2star * t).exp();
                        }
                    }
                }
            }
        }
        (chi, r2p, mag, se, mask)
    }

    /// Full pipeline (with SE evidence): WM slab triggers the beat branch (stages
    /// 3-5), GM uses the closed form; exercises Dr+ self-calibration on a >5000
    /// paramagnetic-voxel pool.
    #[test]
    fn hc_chisep_end_to_end_wm_and_gm() {
        let tes = [0.004, 0.012, 0.020, 0.028];
        let se_tes = [0.01, 0.03, 0.05, 0.07];
        let dims = (20, 20, 20);
        let n = dims.0 * dims.1 * dims.2;
        let (chi, r2p, mag, se, mask) = synth_phantom(dims, 6, 7.0, &tes, &se_tes);
        let grid = Grid::new(dims.0, dims.1, dims.2, 1.0, 1.0, 1.0);
        let params = HcChisepParams { b0: 7.0, se_echo_times: se_tes.to_vec(), ..Default::default() };

        let mut stages = 0;
        let (pos, neg, tot) = hc_chisep(
            &chi, &r2p, &mag, &tes, Some(&se), &mask, &grid, &params, |i, _| stages = i,
        );
        assert_eq!(stages, 5, "all pipeline stages ran (beat detected)");

        // Sign convention + invariant everywhere.
        for i in 0..n {
            assert!(pos[i] >= 0.0 && neg[i] <= 0.0, "signs at {i}");
            assert!((tot[i] - (pos[i] + neg[i])).abs() < 1e-9, "invariant at {i}");
        }
        // Separation happened: paramagnetic in GM, diamagnetic in WM.
        assert!(pos.iter().any(|&v| v > 1e-3), "some χ+ present");
        assert!(neg.iter().any(|&v| v < -1e-3), "some χ− present (WM)");
    }

    /// All-GM phantom: no magnitude beat → `beat_frac` below threshold → the
    /// closed-form branch is returned for every voxel.
    #[test]
    fn hc_chisep_no_beat_falls_back_to_closed_form() {
        let tes = [0.004, 0.012, 0.020, 0.028];
        let dims = (10, 10, 10);
        let n = dims.0 * dims.1 * dims.2;
        let (chi, r2p, mag, _se, mask) = synth_phantom(dims, 0, 7.0, &tes, &[]);
        let grid = Grid::new(dims.0, dims.1, dims.2, 1.0, 1.0, 1.0);
        let params = HcChisepParams { b0: 7.0, ..Default::default() };

        let (pos, neg, _tot) =
            hc_chisep(&chi, &r2p, &mag, &tes, None, &mask, &grid, &params, |_, _| {});
        // Closed form on this phantom: χ+ = R2'/Dr+ = 0.05, χ− = 0.
        for i in 0..n {
            assert!((pos[i] - 0.05).abs() < 1e-6, "closed-form χ+ at {i}: {}", pos[i]);
            assert!(neg[i].abs() < 1e-9, "closed-form χ− at {i}: {}", neg[i]);
        }
    }

    /// Empty mask returns all zeros via the early exit.
    #[test]
    fn hc_chisep_empty_mask_returns_zero() {
        let tes = [0.004, 0.012, 0.020, 0.028];
        let dims = (6, 6, 6);
        let n = dims.0 * dims.1 * dims.2;
        let (chi, r2p, mag, _se, _m) = synth_phantom(dims, 0, 7.0, &tes, &[]);
        let mask = vec![0u8; n];
        let grid = Grid::new(dims.0, dims.1, dims.2, 1.0, 1.0, 1.0);
        let params = HcChisepParams { b0: 7.0, ..Default::default() };

        let (pos, neg, tot) =
            hc_chisep(&chi, &r2p, &mag, &tes, None, &mask, &grid, &params, |_, _| {});
        assert!(pos.iter().all(|&v| v == 0.0), "χ+ all zero");
        assert!(neg.iter().all(|&v| v == 0.0), "χ− all zero");
        assert!(tot.iter().all(|&v| v == 0.0), "χ_total all zero");
    }
}

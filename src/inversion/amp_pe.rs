//! AMP-PE: Approximate Message Passing with built-in Parameter Estimation for QSM.
//!
//! Nonlinear dipole inversion solved with Generalized Approximate Message Passing
//! (GAMP) over a linearized wrapped-phase (complex-exponential) forward model,
//! using a Laplace sparse-wavelet prior and a Gaussian-mixture noise model that
//! absorbs phase outliers.
//!
//! Ported from the reference MATLAB implementation
//! (<https://github.com/EmoryCN2L/QSM_AMP_PE>, Huang et al., *Magn. Reson. Med.*
//! 2023) as packaged for the QSM-CI `dipole` stage (`recon.m` +
//! `amp_pe_mri_qsm_awgn.m` / `amp_pe_mri_qsm_awgn_mix.m`). This is the
//! dipole-inversion stage only: the local (tissue) field is provided, turned into
//! a simulated single-echo phase, and fed to the two-step AMP-PE solve
//! (single-Gaussian warm-up → Gaussian-mixture final).
//!
//! # Design notes / fidelity
//! * The forward dipole operator, wavelet transform (periodic db1/db2), and
//!   `erfcx` all match the MATLAB reference (see `utils::wavelet`,
//!   `utils::special`).
//! * The reference estimates operator Frobenius norms with 2 random probes. Here
//!   they are computed **exactly** (the wavelet transform is orthonormal, so its
//!   norm is `sqrt(coef_len)`; the weighted-dipole norm has a closed form). This
//!   is deterministic and a strict improvement; the verification harness injects
//!   the same exact values into MATLAB so the two agree to numerical precision.
//! * Input local field is **ppm-scale** (crate convention); `mut_cst` converts
//!   ppm to radians at the simulated echo time.
//! * The L2 (chiL2) seed is computed on the field **masked to the ROI** (as
//!   recon.m does) — background outside the mask must not enter the seed.
//! * Verified against the MATLAB reference to ~2e-10 on the real 164x205x205
//!   phantom when both use a double-precision seed. The reference's
//!   `dipole_kernel_angulated` casts the seed kernel to `single`; in
//!   phase-wrapping regions (|phase|>pi) the nonlinear solve is multistable, so
//!   that ~1e-7 seed perturbation can select a different local solution there.
//!   This crate uses full double precision (a strict improvement).

use crate::fft::Fft3dWorkspace;
use crate::inversion::admm::prepare_fansi_spectral;
use crate::utils::special::{erfc, erfcx};
use crate::utils::wavelet::WaveletPlan;
use crate::Grid;
use num_complex::Complex64;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

const EPS: f64 = 2.220446049250313e-16;

/// Chunk length for the deterministic parallel reductions in the GAMP parameter
/// estimation. Fixed size ⇒ the result is independent of the thread count.
const PARAM_CHUNK: usize = 1 << 16;

/// Parameters for the AMP-PE inversion.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct AmpPeParams {
    /// Daubechies wavelet order: 1 (db1, best for straight B0) or 2 (db2).
    pub wave_order: usize,
    /// Wavelet decomposition levels.
    pub nlevel: usize,
    /// Morphology-mask retention fraction (cumulative-energy threshold).
    pub wave_pec: f64,
    /// Simulated echo time (s) used to turn the field into phase.
    pub simulated_te: f64,
    /// Linearization iterations for each of the two stages.
    pub max_linearization_ite: usize,
    /// Field strength (Tesla).
    pub b0: f64,
    /// Gyromagnetic ratio (MHz/T); matches the reference's `42.58`.
    pub gyro_ratio: f64,
    /// Damping rate for the GAMP signal updates (`damp_rate`).
    pub damp_rate_sig: f64,
    /// Learning rate for parameter estimation (`kappa`).
    pub damp_rate_par: f64,
    /// Inner sparse-reconstruction iterations per GAMP call.
    pub max_pe_spar_ite: usize,
    /// Inner parameter-estimation iterations per GAMP call.
    pub max_pe_est_ite: usize,
    /// Convergence threshold for the GAMP inner loop.
    pub cvg_thd: f64,
    /// Tikhonov regularization weight for the L2 seed (`chiL2`).
    pub tikhonov_beta: f64,
}

impl Default for AmpPeParams {
    fn default() -> Self {
        Self {
            wave_order: 1,
            nlevel: 3,
            wave_pec: 0.85,
            simulated_te: 8e-3,
            max_linearization_ite: 25,
            b0: 3.0,
            gyro_ratio: 42.58,
            damp_rate_sig: 0.01,
            damp_rate_par: 0.1,
            max_pe_spar_ite: 5,
            max_pe_est_ite: 5,
            cvg_thd: 1e-6,
            tikhonov_beta: 2e-2,
        }
    }
}

/// Sample variance (MATLAB `var`, normalized by `n-1`).
fn var(v: &[f64]) -> f64 {
    let n = v.len();
    if n < 2 {
        return 0.0;
    }
    let mean = v.iter().sum::<f64>() / n as f64;
    v.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0)
}

/// Sample variance of a complex vector (MATLAB `var` on complex data:
/// `mean(|x - mean|^2) * n/(n-1)`).
fn var_complex(v: &[Complex64]) -> f64 {
    let n = v.len();
    if n < 2 {
        return 0.0;
    }
    let mean = v.iter().sum::<Complex64>() / n as f64;
    v.iter().map(|&x| (x - mean).norm_sqr()).sum::<f64>() / (n as f64 - 1.0)
}

/// Dipole forward/adjoint operator over a padded grid, with mask gather/scatter.
struct DipoleOp {
    fft_ws: Fft3dWorkspace,
    kernel: Vec<f64>,
    n: usize,
    mask_idx: Vec<usize>,
    buf: Vec<Complex64>,
    full: Vec<f64>,
    /// Reused scatter buffer for the adjoint. Non-mask positions are only ever
    /// written to zero (at construction) and never touched again, so the mask
    /// scatter needs no per-call clear.
    scatter: Vec<f64>,
}

impl DipoleOp {
    fn new(grid: &Grid, bdir: (f64, f64, f64), mask_idx: Vec<usize>) -> Self {
        let (fft_ws, kernel, _ee2) = prepare_fansi_spectral(grid, bdir);
        let n = grid.n_total();
        Self {
            fft_ws,
            kernel,
            n,
            mask_idx,
            buf: vec![Complex64::new(0.0, 0.0); n],
            full: vec![0.0; n],
            scatter: vec![0.0; n],
        }
    }

    /// `real(ifftn(D .* fftn(x)))` for a full-volume real input.
    fn apply(&mut self, x: &[f64], out: &mut [f64]) {
        for (b, &xi) in self.buf.iter_mut().zip(x) {
            *b = Complex64::new(xi, 0.0);
        }
        self.fft_ws.fft3d(&mut self.buf);
        for (b, &k) in self.buf.iter_mut().zip(&self.kernel) {
            *b *= k;
        }
        self.fft_ws.ifft3d(&mut self.buf);
        for (o, b) in out.iter_mut().zip(&self.buf) {
            *o = b.re;
        }
    }

    /// Masked dipole field: `A.times(x)` — gather the dipole of `x_full` at the mask.
    fn field_masked(&mut self, x_full: &[f64]) -> Vec<f64> {
        let mut full = std::mem::take(&mut self.full);
        self.apply(x_full, &mut full);
        let out = self.mask_idx.iter().map(|&j| full[j]).collect();
        self.full = full;
        out
    }

    /// Adjoint: scatter masked real vector to full volume, apply dipole (self-adjoint).
    fn adjoint_full(&mut self, y_masked: &[f64]) -> Vec<f64> {
        let mut scattered = std::mem::take(&mut self.scatter);
        for (k, &j) in self.mask_idx.iter().enumerate() {
            scattered[j] = y_masked[k];
        }
        let mut out = vec![0.0; self.n];
        self.apply(&scattered, &mut out);
        self.scatter = scattered;
        out
    }
}

/// AMP-PE nonlinear dipole inversion.
///
/// * `local_field` — local (tissue) field, ppm-scale, full volume (`nx*ny*nz`).
/// * `mask` — binary ROI mask (non-zero = inside).
/// * `magnitude` — optional data-fidelity weight (single combined volume, e.g.
///   RSS over echoes). When `None`, uniform weights and no morphology mask.
/// * `grid` — volume grid (dims + voxel size).
/// * `bdir` — B0 direction.
/// * `params` — see [`AmpPeParams`].
/// * `progress` — callback `(stage_iter, total)` where total counts both stages.
///
/// Returns the susceptibility map (ppm-scale), masked to the ROI, on the input grid.
pub fn amp_pe(
    local_field: &[f64],
    mask: &[u8],
    magnitude: Option<&[f64]>,
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &AmpPeParams,
    mut progress: impl FnMut(usize, usize),
) -> Vec<f64> {
    let orig = (grid.nx(), grid.ny(), grid.nz());
    let pad_mult = 1usize << params.nlevel;
    let pad = (
        orig.0.div_ceil(pad_mult) * pad_mult,
        orig.1.div_ceil(pad_mult) * pad_mult,
        orig.2.div_ceil(pad_mult) * pad_mult,
    );
    let pgrid = Grid::new(
        pad.0,
        pad.1,
        pad.2,
        grid.voxel_size.0,
        grid.voxel_size.1,
        grid.voxel_size.2,
    );

    // Pad field / mask / magnitude to the wavelet-friendly grid.
    let field_p = pad_volume(local_field, orig, pad);
    let mask_f: Vec<f64> = mask.iter().map(|&m| if m != 0 { 1.0 } else { 0.0 }).collect();
    let mask_p = pad_volume(&mask_f, orig, pad);
    let imag_p = match magnitude {
        Some(mag) => pad_volume(&mag.iter().map(|&v| v.abs()).collect::<Vec<_>>(), orig, pad),
        None => mask_p.clone(),
    };
    let have_mag = magnitude.is_some();

    let n = pad.0 * pad.1 * pad.2;
    let mask_idx: Vec<usize> = (0..n).filter(|&i| mask_p[i] > 0.5).collect();
    let m = mask_idx.len();

    let te = params.simulated_te;
    let mut_cst = params.gyro_ratio * params.b0 * 2.0 * std::f64::consts::PI * te;

    // Measurement: simulated single-echo tissue phase (radians) at the mask.
    let phase_image: Vec<f64> = mask_idx.iter().map(|&j| field_p[j] * mut_cst).collect();

    // Data-fidelity weights, normalized to unit mean within the mask.
    let weight_vect: Vec<f64> = if have_mag {
        let wv: Vec<f64> = mask_idx.iter().map(|&j| imag_p[j]).collect();
        let mean = (wv.iter().sum::<f64>() / m as f64).max(EPS);
        wv.iter().map(|&v| v / mean).collect()
    } else {
        vec![1.0; m]
    };

    let mut dip = DipoleOp::new(&pgrid, bdir, mask_idx.clone());
    // Kernel energy for the exact A_qsm Frobenius norm: ||g||^2 = mean(D^2).
    let kernel_energy = dip.kernel.iter().map(|&d| d * d).sum::<f64>() / n as f64;

    // Wavelet plan (orthonormal; coef_len == n).
    let plan = WaveletPlan::new(params.wave_order, pad, params.nlevel);
    debug_assert_eq!(plan.coef_len(), n);

    // --- L2 (Tikhonov) seed for distribution initialization only ---
    // Seed on the field masked to the ROI (recon.m builds `phs_tissue` from the
    // masked field): background outside the mask must not enter the seed.
    let field_masked: Vec<f64> = (0..n).map(|i| if mask_p[i] > 0.5 { field_p[i] } else { 0.0 }).collect();
    let chi_l2 = chi_l2_seed(&field_masked, &mask_p, &dip.kernel, params.tikhonov_beta, pad);
    let x_init_par: Vec<f64> = (0..n).map(|i| if mask_p[i] > 0.5 { chi_l2[i] } else { 0.0 }).collect();
    let x_init_par_psi = plan.forward(&x_init_par);

    // Wavelet morphology mask: pass the largest magnitude wavelet coefficients through.
    let wave_mask = build_wave_mask(&plan, &imag_p, &mask_p, have_mag, params.wave_pec, n);

    // --- GAMP shared state ---
    let mut st = GampState {
        x_hat_meas: vec![0.0; n],
        tau_x_meas: var(&x_init_par),
        s_hat_meas: vec![Complex64::new(0.0, 0.0); m],
        x_hat_psi: vec![0.0; n],
        p_hat_psi: vec![0.0; n],
        tau_x_hat_psi: var(&x_init_par_psi),
        tau_p_psi: var(&x_init_par_psi), // A_wav.multSq(tau_x_hat_psi) = identity
        damp: params.damp_rate_sig,
        damp_ceiling: params.damp_rate_sig,
    };
    let abs_psi: Vec<f64> = x_init_par_psi.iter().map(|v| v.abs()).collect();
    let mut lambda = 1.0 / (var(&abs_psi) / 2.0).sqrt();

    let cfg = GampCfg {
        m,
        n,
        mut_cst,
        kernel_energy,
        max_pe_spar_ite: params.max_pe_spar_ite,
        max_pe_est_ite: params.max_pe_est_ite,
        cvg_thd: params.cvg_thd,
        kappa: params.damp_rate_par,
        // NOTE: `damp_rate_sig` is not carried here — the damping is adaptive and lives in
        // GampState (`damp` / `damp_ceiling`), seeded from this same parameter.
    };

    let total = 2 * params.max_linearization_ite;
    let mut x_curr = vec![0.0; n];
    let mut tau_w_1 = 1e-12;

    // ===== Step 1: single-Gaussian noise (warm-up) =====
    for it in 0..params.max_linearization_ite {
        progress(it + 1, total);
        let (a_x, der_1st, y_upd) = linearize(&mut dip, &x_curr, &phase_image, &weight_vect, mut_cst);
        tau_w_1 = gamp_awgn(&mut dip, &plan, &cfg, &mut st, &mut lambda, tau_w_1, &der_1st, &weight_vect, &y_upd, &wave_mask);
        x_curr = st.x_hat_meas.clone();
        let _ = a_x;
    }

    // ===== estimate outlier mixture from the step-1 residual =====
    let a_x_init = dip.field_masked(&x_curr);
    let resid: Vec<Complex64> = (0..m)
        .map(|j| {
            let axm = a_x_init[j] * mut_cst;
            weight_vect[j] * (Complex64::new(0.0, axm).exp() - Complex64::new(0.0, phase_image[j]).exp())
        })
        .collect();
    let resid_abs: Vec<f64> = resid.iter().map(|c| c.norm()).collect();
    let resid_std = (resid_abs.iter().map(|v| v * v).sum::<f64>() / m as f64).sqrt();
    let outliers: Vec<Complex64> = resid
        .iter()
        .zip(&resid_abs)
        .filter(|(_, &a)| a > 3.0 * resid_std)
        .map(|(&c, _)| c)
        .collect();
    let gamma_est = outliers.len() as f64 / m as f64;
    let psi_est = var_complex(&outliers);

    // ===== Step 2: Gaussian-mixture noise (final) =====
    let mut mix = MixState {
        theta: 0.0,
        phi: tau_w_1,
        omega: 1.0,
        gamma: gamma_est,
        psi: psi_est,
    };
    for it in 0..params.max_linearization_ite {
        progress(params.max_linearization_ite + it + 1, total);
        let (a_x, der_1st, y_upd) = linearize(&mut dip, &x_curr, &phase_image, &weight_vect, mut_cst);
        gamp_awgn_mix(&mut dip, &plan, &cfg, &mut st, &mut mix, &mut lambda, &der_1st, &weight_vect, &y_upd, &wave_mask);
        x_curr = st.x_hat_meas.clone();
        let _ = a_x;
    }

    // Mask and crop back to the input grid.
    for i in 0..n {
        if mask_p[i] <= 0.5 {
            x_curr[i] = 0.0;
        }
    }
    crop_volume(&x_curr, pad, orig)
}

/// Persistent GAMP state carried across linearization iterations.
struct GampState {
    x_hat_meas: Vec<f64>,       // susceptibility image (length n)
    tau_x_meas: f64,
    s_hat_meas: Vec<Complex64>, // length m
    x_hat_psi: Vec<f64>,        // wavelet coefficients (length n)
    p_hat_psi: Vec<f64>,        // image (length n)
    tau_x_hat_psi: f64,
    tau_p_psi: f64,
    /// CURRENT damping factor for the signal updates. Starts at the configured
    /// `damp_rate_sig` and is only ever reduced when a sweep is found to have made the
    /// data fit worse (see the adaptive-damping note on `gamp_awgn`), then relaxed back
    /// towards the configured value. On data where nothing diverges it never leaves the
    /// configured value, so the iteration is bit-for-bit what it was before.
    damp: f64,
    /// Ceiling that `damp` may relax back up to. RATCHETS DOWN on every rejected sweep: once a
    /// run has shown it diverges at a given damping, letting the damping climb back to that value
    /// simply re-diverges. Without this the guard catches each blow-up but the iteration keeps
    /// re-entering the divergent regime and the result is still unusable.
    damp_ceiling: f64,
}

/// A restore point for backing out of a divergent GAMP sweep.
struct GampSnapshot {
    x_hat_meas: Vec<f64>,
    tau_x_meas: f64,
    s_hat_meas: Vec<Complex64>,
    x_hat_psi: Vec<f64>,
    p_hat_psi: Vec<f64>,
    tau_x_hat_psi: f64,
    tau_p_psi: f64,
    lambda: f64,
    mix: Option<MixState>,
}

impl GampState {
    fn snapshot(&self, lambda: f64, mix: Option<&MixState>) -> GampSnapshot {
        GampSnapshot {
            x_hat_meas: self.x_hat_meas.clone(),
            tau_x_meas: self.tau_x_meas,
            s_hat_meas: self.s_hat_meas.clone(),
            x_hat_psi: self.x_hat_psi.clone(),
            p_hat_psi: self.p_hat_psi.clone(),
            tau_x_hat_psi: self.tau_x_hat_psi,
            tau_p_psi: self.tau_p_psi,
            lambda,
            mix: mix.map(|m| MixState { ..*m }),
        }
    }

    fn restore(&mut self, snap: &GampSnapshot, lambda: &mut f64, mix: Option<&mut MixState>) {
        self.x_hat_meas.copy_from_slice(&snap.x_hat_meas);
        self.tau_x_meas = snap.tau_x_meas;
        self.s_hat_meas.copy_from_slice(&snap.s_hat_meas);
        self.x_hat_psi.copy_from_slice(&snap.x_hat_psi);
        self.p_hat_psi.copy_from_slice(&snap.p_hat_psi);
        self.tau_x_hat_psi = snap.tau_x_hat_psi;
        self.tau_p_psi = snap.tau_p_psi;
        *lambda = snap.lambda;
        if let (Some(dst), Some(src)) = (mix, snap.mix.as_ref()) {
            *dst = MixState { ..*src };
        }
    }
}

// --- Adaptive damping ---------------------------------------------------------------------
// GAMP is only guaranteed to converge for measurement operators close to i.i.d. Gaussian. The
// dipole operator, weighted by the magnitude and composed with the wavelet morphology prior, is
// not, and on some in-vivo data the iteration diverges outright: the estimate grows without
// bound until χ reaches ~1e34. The published remedy is adaptive damping (Vila, Schniter, Rangan
// et al., "Adaptive damping and mean removal for the generalized approximate message passing
// algorithm", ICASSP 2015): watch a cost, and when a step makes it worse, undo the step and take
// a smaller one.
//
// The cost is the plain data fit ||A x_hat - y||^2/m, which `a_qsm_mult` already computes each
// sweep. NOTE it must NOT be the `mse` used for noise-variance estimation: that one includes the
// Onsager correction term and is an estimate of the noise level, not a descent quantity, so it
// fluctuates on perfectly healthy runs.
//
// The trigger is deliberately blunt — an order of magnitude worse than the best fit seen so far,
// or non-finite. Divergence here spans many orders of magnitude, so a loose threshold still
// catches it, while normal GAMP non-monotonicity never comes close. That keeps this a strict
// no-op on data that does not diverge, which is what preserves agreement with the reference
// implementation.
/// Cost blow-up factor (relative to the best fit so far) treated as divergence.
const DAMP_DIVERGE_FACTOR: f64 = 2.0;
/// Factor applied to the damping when a sweep is rejected.
const DAMP_SHRINK: f64 = 0.5;
/// Factor by which damping relaxes back towards the configured value after an accepted sweep.
const DAMP_GROW: f64 = 1.1;
/// Damping floor; below this the iteration is making no progress and we stop with the last good state.
const DAMP_MIN: f64 = 1e-4;
/// Cap on rejected sweeps per GAMP call, so a pathological case cannot spin.
const MAX_BACKTRACKS: usize = 20;

#[cfg(test)]
thread_local! {
    /// Sweeps rejected by the divergence guard, counted per thread (each test gets its own).
    /// The guard runs in the sequential part of the GAMP loop, so this never races with rayon.
    static GUARD_FIRINGS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Record that the divergence guard rejected a sweep. Compiled away outside tests.
#[inline]
fn note_guard_firing() {
    #[cfg(test)]
    GUARD_FIRINGS.with(|c| c.set(c.get() + 1));
}

/// Static GAMP configuration.
struct GampCfg {
    m: usize,
    n: usize,
    mut_cst: f64,
    kernel_energy: f64,
    max_pe_spar_ite: usize,
    max_pe_est_ite: usize,
    cvg_thd: f64,
    kappa: f64,
}

/// Gaussian-mixture noise parameters (single component + outlier component).
struct MixState {
    theta: f64,
    phi: f64,
    omega: f64,
    gamma: f64,
    psi: f64,
}

/// Build the linearized measurement for the current estimate.
///
/// Returns `(A*x (masked, radians), der_1st = i*exp(i*Ax), y_upd)`.
fn linearize(
    dip: &mut DipoleOp,
    x_curr: &[f64],
    phase_image: &[f64],
    weight_vect: &[f64],
    mut_cst: f64,
) -> (Vec<f64>, Vec<Complex64>, Vec<Complex64>) {
    let field = dip.field_masked(x_curr);
    let m = field.len();
    let a_x: Vec<f64> = (0..m).map(|j| field[j] * mut_cst).collect();
    let der_1st: Vec<Complex64> = a_x.iter().map(|&v| Complex64::new(0.0, 1.0) * Complex64::new(0.0, v).exp()).collect();
    let y_upd: Vec<Complex64> = (0..m)
        .map(|j| {
            let e_phase = Complex64::new(0.0, phase_image[j]).exp();
            let e_ax = Complex64::new(0.0, a_x[j]).exp();
            weight_vect[j] * (der_1st[j] * a_x[j] + e_phase - e_ax)
        })
        .collect();
    (a_x, der_1st, y_upd)
}

/// Frobenius norm squared of the weighted-dipole operator `A_qsm`.
///
/// `||A_qsm||_F^2 = mut_cst^2 * ||g||^2 * sum_j |der_j * w_j|^2`, where
/// `||g||^2 = mean(D^2)` (kernel_energy).
fn frob_qsm_sq(cfg: &GampCfg, der_1st: &[Complex64], weight_vect: &[f64]) -> f64 {
    let sw: f64 = der_1st
        .iter()
        .zip(weight_vect)
        .map(|(&d, &w)| (d * w).norm_sqr())
        .sum();
    cfg.mut_cst * cfg.mut_cst * cfg.kernel_energy * sw
}

/// A_qsm forward: image -> masked complex measurement.
fn a_qsm_mult(dip: &mut DipoleOp, x_img: &[f64], der_1st: &[Complex64], weight_vect: &[f64], mut_cst: f64) -> Vec<Complex64> {
    let field = dip.field_masked(x_img);
    maybe_par_iter!(field)
        .zip(maybe_par_iter!(der_1st))
        .zip(maybe_par_iter!(weight_vect))
        .map(|((&fj, &dj), &wj)| dj * wj * (fj * mut_cst))
        .collect()
}

/// real(A_qsm^H * s): masked complex -> full-volume real.
fn a_qsm_multtr_real(dip: &mut DipoleOp, s: &[Complex64], der_1st: &[Complex64], weight_vect: &[f64], mut_cst: f64) -> Vec<f64> {
    // x_tmp = s .* conj(der*w); take real part; adjoint dipole; scale by mut_cst.
    let re: Vec<f64> = maybe_par_iter!(s)
        .zip(maybe_par_iter!(der_1st))
        .zip(maybe_par_iter!(weight_vect))
        .map(|((&sj, &dj), &wj)| (sj * (dj * wj).conj()).re)
        .collect();
    let mut adj = dip.adjoint_full(&re);
    maybe_par_iter_mut!(adj).for_each(|v| *v *= mut_cst);
    adj
}

/// One GAMP call with single-Gaussian (AWGN) noise. Returns updated `tau_w_1`.
#[allow(clippy::too_many_arguments)]
fn gamp_awgn(
    dip: &mut DipoleOp,
    plan: &WaveletPlan,
    cfg: &GampCfg,
    st: &mut GampState,
    lambda: &mut f64,
    tau_w_1_in: f64,
    der_1st: &[Complex64],
    weight_vect: &[f64],
    y: &[Complex64],
    wave_mask: &[bool],
) -> f64 {
    let frob_sq = frob_qsm_sq(cfg, der_1st, weight_vect);
    let mut tau_w_1 = tau_w_1_in;
    let (m, n) = (cfg.m, cfg.n);

    let mut best_cost = f64::INFINITY;
    let mut good: Option<GampSnapshot> = None;
    let mut good_tau_w = tau_w_1;
    let mut accepted = 0usize;
    let mut backtracks = 0usize;

    while accepted < cfg.max_pe_spar_ite && backtracks <= MAX_BACKTRACKS {
        // tau_p_meas_1 = frob^2/M * tau_x_meas
        let tau_p_meas_1 = frob_sq / m as f64 * st.tau_x_meas;
        // p_hat_meas_1 = A_qsm*x_hat_meas - tau_p*s
        let axm = a_qsm_mult(dip, &st.x_hat_meas, der_1st, weight_vect, cfg.mut_cst);
        let p_hat_meas_1: Vec<Complex64> = (0..m).map(|j| axm[j] - tau_p_meas_1 * st.s_hat_meas[j]).collect();

        // parameter estimation for tau_w_1. `mse` is invariant across the inner
        // sweep (p_hat_meas_1 and y are fixed), so compute it once.
        let mse: f64 = maybe_par_chunks!(p_hat_meas_1.as_slice(), PARAM_CHUNK)
            .zip(maybe_par_chunks!(y, PARAM_CHUNK))
            .map(|(pc, yc)| pc.iter().zip(yc).map(|(p, yy)| (p - yy).norm_sqr()).sum::<f64>())
            .collect::<Vec<f64>>()
            .iter()
            .sum::<f64>()
            / m as f64;

        // Data fit of the state the PREVIOUS sweep produced. If it has blown up, that sweep
        // diverged: undo it and take smaller steps from here.
        let fit = data_fit(&axm, y, m);
        if !fit.is_finite() || fit > best_cost * DAMP_DIVERGE_FACTOR {
            note_guard_firing();
            backtracks += 1;
            if let Some(g) = good.as_ref() {
                st.restore(g, lambda, None);
                tau_w_1 = good_tau_w;
            }
            st.damp = (st.damp * DAMP_SHRINK).max(DAMP_MIN);
            st.damp_ceiling = st.damp;
            if st.damp <= DAMP_MIN {
                break; // cannot damp further; keep the last good state
            }
            continue;
        }
        if fit < best_cost {
            best_cost = fit;
            good = Some(st.snapshot(*lambda, None));
            good_tau_w = tau_w_1;
        }
        st.damp = (st.damp * DAMP_GROW).min(st.damp_ceiling);

        let tau_w_new = mse + tau_p_meas_1;
        for _ in 0..cfg.max_pe_est_ite {
            tau_w_1 += cfg.kappa * (tau_w_new - tau_w_1);
        }

        let tau_s_meas_1 = 1.0 / (tau_w_1 + tau_p_meas_1);
        for j in 0..m {
            st.s_hat_meas[j] = (y[j] - p_hat_meas_1[j]) * tau_s_meas_1;
        }

        // tau_r_meas_1 = 1 / (frob^2/N * tau_s_meas_1)
        let tau_r_meas_1 = 1.0 / (frob_sq / n as f64 * tau_s_meas_1);
        let mtr = a_qsm_multtr_real(dip, &st.s_hat_meas, der_1st, weight_vect, cfg.mut_cst);
        let r_hat_meas_1: Vec<f64> = maybe_par_iter!(st.x_hat_meas)
            .zip(maybe_par_iter!(mtr))
            .map(|(&x, &mv)| x + tau_r_meas_1 * mv)
            .collect();

        let cvg = wavelet_block(plan, cfg, st, lambda, tau_r_meas_1, &r_hat_meas_1, wave_mask);
        accepted += 1;
        if cvg < cfg.cvg_thd {
            break;
        }
    }
    tau_w_1
}

/// Plain data fit `||A x - y||^2 / m`, the quantity the divergence guard watches.
fn data_fit(axm: &[Complex64], y: &[Complex64], m: usize) -> f64 {
    maybe_par_chunks!(axm, PARAM_CHUNK)
        .zip(maybe_par_chunks!(y, PARAM_CHUNK))
        .map(|(ac, yc)| ac.iter().zip(yc).map(|(a, yy)| (a - yy).norm_sqr()).sum::<f64>())
        .collect::<Vec<f64>>()
        .iter()
        .sum::<f64>()
        / m as f64
}

/// One GAMP call with Gaussian-mixture noise (final stage).
#[allow(clippy::too_many_arguments)]
fn gamp_awgn_mix(
    dip: &mut DipoleOp,
    plan: &WaveletPlan,
    cfg: &GampCfg,
    st: &mut GampState,
    mix: &mut MixState,
    lambda: &mut f64,
    der_1st: &[Complex64],
    weight_vect: &[f64],
    y: &[Complex64],
    wave_mask: &[bool],
) {
    let frob_sq = frob_qsm_sq(cfg, der_1st, weight_vect);
    let (m, n) = (cfg.m, cfg.n);

    let mut best_cost = f64::INFINITY;
    let mut good: Option<GampSnapshot> = None;
    let mut accepted = 0usize;
    let mut backtracks = 0usize;

    while accepted < cfg.max_pe_spar_ite && backtracks <= MAX_BACKTRACKS {
        let tau_p_meas_1 = frob_sq / m as f64 * st.tau_x_meas;
        let axm = a_qsm_mult(dip, &st.x_hat_meas, der_1st, weight_vect, cfg.mut_cst);
        let p_hat_meas_1: Vec<Complex64> = (0..m).map(|j| axm[j] - tau_p_meas_1 * st.s_hat_meas[j]).collect();
        let r_noise: Vec<Complex64> = (0..m).map(|j| y[j] - p_hat_meas_1[j]).collect();

        // Same divergence guard as the AWGN stage.
        let fit = data_fit(&axm, y, m);
        if !fit.is_finite() || fit > best_cost * DAMP_DIVERGE_FACTOR {
            note_guard_firing();
            backtracks += 1;
            if let Some(g) = good.as_ref() {
                st.restore(g, lambda, Some(mix));
            }
            st.damp = (st.damp * DAMP_SHRINK).max(DAMP_MIN);
            st.damp_ceiling = st.damp;
            if st.damp <= DAMP_MIN {
                break;
            }
            continue;
        }
        if fit < best_cost {
            best_cost = fit;
            good = Some(st.snapshot(*lambda, Some(mix)));
        }
        st.damp = (st.damp * DAMP_GROW).min(st.damp_ceiling);

        for _ in 0..cfg.max_pe_est_ite {
            mix_output_parameter_est(&r_noise, tau_p_meas_1, mix, cfg.kappa);
        }
        let (noise_update, tau_z) = mix_output_function(&r_noise, tau_p_meas_1, mix);
        // z_hat = y - noise_update
        let tau_s_meas_1 = 1.0 / tau_p_meas_1 * (1.0 - tau_z / tau_p_meas_1);
        for j in 0..m {
            let z_hat = y[j] - noise_update[j];
            st.s_hat_meas[j] = 1.0 / tau_p_meas_1 * (z_hat - p_hat_meas_1[j]);
        }

        let tau_r_meas_1 = 1.0 / (frob_sq / n as f64 * tau_s_meas_1);
        let mtr = a_qsm_multtr_real(dip, &st.s_hat_meas, der_1st, weight_vect, cfg.mut_cst);
        let r_hat_meas_1: Vec<f64> = maybe_par_iter!(st.x_hat_meas)
            .zip(maybe_par_iter!(mtr))
            .map(|(&x, &mv)| x + tau_r_meas_1 * mv)
            .collect();

        let cvg = wavelet_block(plan, cfg, st, lambda, tau_r_meas_1, &r_hat_meas_1, wave_mask);
        accepted += 1;
        if cvg < cfg.cvg_thd {
            break;
        }
    }
}

/// Shared wavelet-domain GAMP block (identical between AWGN and mixture stages).
///
/// Returns the relative change in `x_hat_meas` (`||Δx|| / ||x||`) used for the
/// GAMP inner-loop convergence test.
fn wavelet_block(
    plan: &WaveletPlan,
    cfg: &GampCfg,
    st: &mut GampState,
    lambda: &mut f64,
    tau_r_meas_1: f64,
    r_hat_meas_1: &[f64],
    wave_mask: &[bool],
) -> f64 {
    // Current adaptive damping (see the adaptive-damping note above); equals the configured
    // `damp_rate_sig` unless a divergent sweep has forced it down.
    let damp = st.damp;
    let tau_s_psi = 1.0 / (tau_r_meas_1 + st.tau_p_psi);
    let s_hat_psi: Vec<f64> = maybe_par_iter!(r_hat_meas_1)
        .zip(maybe_par_iter!(st.p_hat_psi))
        .map(|(&r, &p)| (r - p) * tau_s_psi)
        .collect();

    // A_wav.multSqTr is identity -> tau_r_psi = 1/tau_s_psi
    let tau_r_psi = 1.0 / tau_s_psi;
    let analysis = plan.forward(&s_hat_psi); // A_wav.multTr
    let r_hat_psi: Vec<f64> = maybe_par_iter!(st.x_hat_psi)
        .zip(maybe_par_iter!(analysis))
        .map(|(&x, &a)| x + tau_r_psi * a)
        .collect();

    let abs_r: Vec<f64> = maybe_par_iter!(r_hat_psi).map(|v| v.abs()).collect();
    for _ in 0..cfg.max_pe_est_ite {
        *lambda = input_parameter_est(&abs_r, tau_r_psi, *lambda, cfg.kappa);
    }

    let (x_hat_psi_new, tau_x_hat_psi) = input_function(&r_hat_psi, tau_r_psi, *lambda, wave_mask);
    st.x_hat_psi = x_hat_psi_new;
    st.tau_x_hat_psi = tau_x_hat_psi;

    st.tau_p_psi = tau_x_hat_psi; // A_wav.multSq identity
    let synth = plan.inverse(&st.x_hat_psi); // A_wav.mult
    let tau_p_psi = st.tau_p_psi;
    maybe_par_iter_mut!(st.p_hat_psi)
        .zip(maybe_par_iter!(synth))
        .zip(maybe_par_iter!(s_hat_psi))
        .for_each(|((p, &sy), &sh)| *p = sy - tau_p_psi * sh);

    let tau_x_meas_pre = st.tau_x_meas;
    let tau_new = (st.tau_p_psi * tau_r_meas_1) / (st.tau_p_psi + tau_r_meas_1);
    st.tau_x_meas = tau_x_meas_pre + damp * (tau_new - tau_x_meas_pre);

    let denom = st.tau_p_psi + tau_r_meas_1;
    let mut change_sq = 0.0;
    let mut norm_sq = 0.0;
    for ((xh, &r), &p) in st.x_hat_meas.iter_mut().zip(r_hat_meas_1).zip(&st.p_hat_psi) {
        let x_new = (st.tau_p_psi * r + tau_r_meas_1 * p) / denom;
        let updated = *xh + damp * (x_new - *xh);
        let d = updated - *xh;
        change_sq += d * d;
        norm_sq += updated * updated;
        *xh = updated;
    }
    change_sq.sqrt() / norm_sq.sqrt().max(EPS)
}

/// Laplace input function: soft-threshold with the morphology mask passing large
/// coefficients through unshrunk. Returns `(x_hat_psi, tau_x)`.
fn input_function(r_hat: &[f64], tau_r: f64, lambda: f64, wave_mask: &[bool]) -> (Vec<f64>, f64) {
    let thresh = lambda * tau_r;
    let mut x0 = vec![0.0; r_hat.len()];
    let mut nnz = 0usize;
    for i in 0..r_hat.len() {
        let v = if wave_mask[i] {
            r_hat[i]
        } else {
            let a = r_hat[i].abs() - thresh;
            if a > 0.0 {
                a * r_hat[i].signum()
            } else {
                0.0
            }
        };
        if v != 0.0 {
            nnz += 1;
        }
        x0[i] = v;
    }
    let tau_x = tau_r * nnz as f64 / r_hat.len() as f64;
    (x0, tau_x)
}

/// EM update of the Laplace scale parameter `lambda` (single cluster).
fn input_parameter_est(r_hat: &[f64], tau_r: f64, lambda: f64, kappa: f64) -> f64 {
    let s = (0.5 / tau_r).sqrt();
    // Deterministic parallel reduction: sequential sum within each fixed-size
    // chunk, chunks combined in index order (thread-count independent).
    let partials: Vec<(f64, f64)> = maybe_par_chunks!(r_hat, PARAM_CHUNK)
        .map(|chunk| {
            let mut s1 = 0.0;
            let mut s2 = 0.0;
            for &r in chunk {
                let arg = tau_r * lambda - r;
                // block0 = lambda/2 * exp(0) * erfc(s*arg)   (block - block_min == 0)
                let b0 = lambda / 2.0 * erfc(s * arg);
                let der = (2.0 * tau_r / std::f64::consts::PI).sqrt() / erfcx(s * arg) + r - tau_r * lambda;
                let fst = 1.0 / lambda - der;
                let scd = -1.0 / (lambda * lambda) + (tau_r + (r - tau_r * lambda) * der) - der * der;
                let w = b0 / (b0 + EPS);
                s1 += w * fst;
                s2 += w * scd;
            }
            (s1, s2)
        })
        .collect();
    let (sum1, sum2) = partials.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0, a.1 + p.1));
    let lambda_new = if sum2 < 0.0 {
        lambda - sum1 / sum2
    } else if sum1 > 0.0 {
        lambda * 1.1
    } else {
        lambda * 0.9
    };
    let lambda_new = lambda_new.max(1e-12);
    lambda + kappa * (lambda_new - lambda)
}

/// Gaussian-mixture output function (single Gaussian + zero-mean outlier Gaussian).
/// Returns `(x_hat, tau_x)` where `x_hat` is the posterior-mean noise estimate.
fn mix_output_function(r_hat: &[Complex64], tau_r: f64, mix: &MixState) -> (Vec<Complex64>, f64) {
    let (omega, theta, phi, gamma, psi) = (mix.omega, mix.theta, mix.phi, mix.gamma, mix.psi);
    let mut x_hat = vec![Complex64::new(0.0, 0.0); r_hat.len()];
    // Per-chunk: write the x_hat slice and accumulate a partial tau sum.
    let partials: Vec<f64> = maybe_par_chunks!(r_hat, PARAM_CHUNK)
        .zip(maybe_par_chunks_mut!(x_hat, PARAM_CHUNK))
        .map(|(rc, xc)| {
            let mut tau_sum = 0.0;
            for (&r, xh_out) in rc.iter().zip(xc.iter_mut()) {
                // component 1
                let diff = theta - r.norm(); // |theta - r_hat| with theta real; matches abs for theta=0
                let e1 = (-(diff / (phi + tau_r).sqrt()).powi(2)).exp();
                let block1 = (1.0 - gamma) * omega * (tau_r / (phi + tau_r)) * e1;
                let mean1 = (theta * tau_r + r * phi) / (phi + tau_r);
                let block_nmr1 = block1 * mean1;
                // outlier component (zero mean)
                let e2 = (-(r.norm() / (psi + tau_r).sqrt()).powi(2)).exp();
                let block2 = gamma * (tau_r / (psi + tau_r)) * e2;
                let mean2 = r * psi / (psi + tau_r);
                let block_nmr2 = block2 * mean2;

                let nmr = block_nmr1 + block_nmr2;
                let dnm = block1 + block2;
                let xh = if dnm == 0.0 { r } else { nmr / dnm };
                *xh_out = xh;

                let nmr_sq1 = block1 * (phi * tau_r / (phi + tau_r) + mean1.norm_sqr());
                let nmr_sq2 = block2 * (psi * tau_r / (psi + tau_r) + mean2.norm_sqr());
                let nmr_sq = nmr_sq1 + nmr_sq2;
                let tau_seq = if dnm == 0.0 { 0.0 } else { nmr_sq / dnm - xh.norm_sqr() };
                tau_sum += tau_seq;
            }
            tau_sum
        })
        .collect();
    let tau_sum: f64 = partials.iter().sum();
    let tau_x = (tau_sum / r_hat.len() as f64).max(1e-12);
    (x_hat, tau_x)
}

/// EM update of the mixture parameters `omega`, `phi`, `psi` (theta, gamma fixed).
fn mix_output_parameter_est(r_hat: &[Complex64], tau_r: f64, mix: &mut MixState, kappa: f64) {
    let (omega, theta, phi, gamma, psi) = (mix.omega, mix.theta, mix.phi, mix.gamma, mix.psi);
    // sums: (block_1, block_2, block_1*|r-theta|^2, block_2*|r|^2)
    let partials: Vec<(f64, f64, f64, f64)> = maybe_par_chunks!(r_hat, PARAM_CHUNK)
        .map(|chunk| {
            let (mut sum_b1, mut sum_b2, mut sum_phi_num, mut sum_psi_num) = (0.0, 0.0, 0.0, 0.0);
            for &r in chunk {
                let d1 = r.norm() - theta; // |r - theta| for real theta and using magnitude; theta=0
                let t1 = (1.0 - gamma) * omega / (tau_r + phi) * (-(d1 / (tau_r + phi).sqrt()).powi(2)).exp();
                let t2 = gamma / (tau_r + psi) * (-(r.norm() / (tau_r + psi).sqrt()).powi(2)).exp();
                let sum = t1 + t2;
                let (b1, b2) = if sum == 0.0 { (0.0, 1.0) } else { (t1 / sum, t2 / sum) };
                sum_b1 += b1;
                sum_b2 += b2;
                sum_phi_num += b1 * d1 * d1;
                sum_psi_num += b2 * r.norm_sqr();
            }
            (sum_b1, sum_b2, sum_phi_num, sum_psi_num)
        })
        .collect();
    let (sum_b1, sum_b2, sum_phi_num, sum_psi_num) = partials
        .iter()
        .fold((0.0, 0.0, 0.0, 0.0), |a, p| (a.0 + p.0, a.1 + p.1, a.2 + p.2, a.3 + p.3));
    // omega: single cluster normalizes to 1.
    let omega_new = 1.0;
    mix.omega = omega + kappa * (omega_new - omega);
    mix.theta = 0.0;
    // phi
    let mut phi_new = if sum_b1 != 0.0 { sum_phi_num / sum_b1 - tau_r } else { phi };
    if !phi_new.is_finite() || phi_new < 0.0 {
        phi_new = phi;
    }
    mix.phi = phi + kappa * (phi_new - phi);
    // psi
    let mut psi_new = if sum_b2 != 0.0 { sum_psi_num / sum_b2 - tau_r } else { psi };
    if psi_new < 0.0 {
        psi_new = psi;
    }
    mix.psi = psi + kappa * (psi_new - psi);
}

/// Direct L2 (Tikhonov, gradient) QSM solver used only to seed distributions.
/// Matches `chiL2.m`: `real(ifftn(conj(K) fftn(phase) / (|K|^2 + beta*E2 + eps))) * mask`.
fn chi_l2_seed(phase: &[f64], mask: &[f64], kernel: &[f64], beta: f64, dims: (usize, usize, usize)) -> Vec<f64> {
    let (nx, ny, nz) = dims;
    let n = nx * ny * nz;
    let tau = std::f64::consts::TAU;
    // E2(k) = |1-exp(2πi kx/N)|^2 + ... (real, DC=0)
    let mut e2 = vec![0.0f64; n];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let idx = i + j * nx + k * nx * ny;
                let ex = Complex64::new(1.0, 0.0) - Complex64::new(0.0, tau * i as f64 / nx as f64).exp();
                let ey = Complex64::new(1.0, 0.0) - Complex64::new(0.0, tau * j as f64 / ny as f64).exp();
                let ez = Complex64::new(1.0, 0.0) - Complex64::new(0.0, tau * k as f64 / nz as f64).exp();
                e2[idx] = ex.norm_sqr() + ey.norm_sqr() + ez.norm_sqr();
            }
        }
    }
    let mut ws = Fft3dWorkspace::new(nx, ny, nz);
    let mut buf: Vec<Complex64> = phase.iter().map(|&p| Complex64::new(p, 0.0)).collect();
    ws.fft3d(&mut buf);
    for i in 0..n {
        let k2 = kernel[i] * kernel[i];
        buf[i] = kernel[i] * buf[i] / (k2 + beta * e2[i] + EPS);
    }
    ws.ifft3d(&mut buf);
    (0..n).map(|i| buf[i].re * mask[i]).collect()
}

/// Build the morphology mask: coefficients whose magnitude exceeds a
/// cumulative-energy threshold of the magnitude image's wavelet coefficients.
fn build_wave_mask(plan: &WaveletPlan, imag: &[f64], mask: &[f64], have_mag: bool, wave_pec: f64, n: usize) -> Vec<bool> {
    if !have_mag {
        return vec![true; n];
    }
    let imag_masked: Vec<f64> = (0..n).map(|i| if mask[i] > 0.5 { imag[i] } else { 0.0 }).collect();
    let magwav = plan.forward(&imag_masked);
    let mut abs_sorted: Vec<f64> = magwav.iter().map(|v| v.abs()).collect();
    abs_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap()); // descending
    let total: f64 = abs_sorted.iter().sum();
    // cumulative sum; count elements while cumsum/total <= wave_pec
    let mut cum = 0.0;
    let mut count = 0usize;
    for &v in &abs_sorted {
        cum += v;
        if cum / total <= wave_pec {
            count += 1;
        } else {
            break;
        }
    }
    let thd = abs_sorted[count.max(1) - 1];
    magwav.iter().map(|v| v.abs() > thd).collect()
}

fn pad_volume(v: &[f64], orig: (usize, usize, usize), pad: (usize, usize, usize)) -> Vec<f64> {
    if orig == pad {
        return v.to_vec();
    }
    let (ox, oy, oz) = orig;
    let (px, py, pz) = pad;
    let mut out = vec![0.0; px * py * pz];
    for k in 0..oz {
        for j in 0..oy {
            for i in 0..ox {
                out[i + j * px + k * px * py] = v[i + j * ox + k * ox * oy];
            }
        }
    }
    out
}

fn crop_volume(v: &[f64], pad: (usize, usize, usize), orig: (usize, usize, usize)) -> Vec<f64> {
    if orig == pad {
        return v.to_vec();
    }
    let (ox, oy, oz) = orig;
    let (px, py, _pz) = pad;
    let mut out = vec![0.0; ox * oy * oz];
    for k in 0..oz {
        for j in 0..oy {
            for i in 0..ox {
                out[i + j * ox + k * ox * oy] = v[i + j * px + k * px * py];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn amp_pe_masks_output() {
        // Non-degenerate structured field; check output is finite and zero outside the mask.
        let n = 16;
        let field: Vec<f64> = (0..n * n * n)
            .map(|i| {
                let (x, y, z) = (i % n, (i / n) % n, i / (n * n));
                0.02 * (((x + 2 * y + 3 * z) as f64) * 0.3).sin()
            })
            .collect();
        let mut mask = vec![0u8; n * n * n];
        // central cube mask
        for z in 4..12 {
            for y in 4..12 {
                for x in 4..12 {
                    mask[x + y * n + z * n * n] = 1;
                }
            }
        }
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = AmpPeParams { max_linearization_ite: 3, ..Default::default() };
        let chi = amp_pe(&field, &mask, None, &grid, (0.0, 0.0, 1.0), &params, |_, _| {});
        assert_eq!(chi.len(), n * n * n);
        for (i, &v) in chi.iter().enumerate() {
            assert!(v.is_finite(), "chi[{i}] not finite: {v}");
            if mask[i] == 0 {
                assert_eq!(v, 0.0, "chi must be zero outside mask at {i}");
            }
        }
    }

    #[test]
    fn amp_pe_ignores_out_of_mask_field() {
        // The result must depend only on the field inside the mask (the chiL2 seed
        // and the data term are masked). Adding arbitrary background outside the
        // mask must not change the output.
        let n = 16;
        let base: Vec<f64> = (0..n * n * n)
            .map(|i| {
                let (x, y, z) = (i % n, (i / n) % n, i / (n * n));
                0.02 * (((x + 2 * y + 3 * z) as f64) * 0.3).sin()
            })
            .collect();
        let mut mask = vec![0u8; n * n * n];
        for z in 4..12 {
            for y in 4..12 {
                for x in 4..12 {
                    mask[x + y * n + z * n * n] = 1;
                }
            }
        }
        // Field with background outside the mask.
        let mut with_bg = base.clone();
        for i in 0..with_bg.len() {
            if mask[i] == 0 {
                with_bg[i] += 0.5 * ((i % 11) as f64 - 5.0);
            }
        }
        // Field zeroed outside the mask.
        let masked: Vec<f64> = (0..base.len()).map(|i| if mask[i] != 0 { base[i] } else { 0.0 }).collect();

        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = AmpPeParams { max_linearization_ite: 4, ..Default::default() };
        let a = amp_pe(&with_bg, &mask, None, &grid, (0.0, 0.0, 1.0), &params, |_, _| {});
        let b = amp_pe(&masked, &mask, None, &grid, (0.0, 0.0, 1.0), &params, |_, _| {});
        let maxerr = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
        assert!(maxerr < 1e-9, "out-of-mask field leaked into result: maxerr={maxerr}");
    }

    /// The adaptive-damping guard must NEVER fire on well-conditioned data. That is the whole
    /// basis for claiming this iteration still matches the MATLAB reference: the guard only
    /// engages when the solve is diverging, so on ordinary data the arithmetic is untouched.
    ///
    /// This is not covered by the other tests in this module. An earlier revision of the guard
    /// used the GAMP noise-variance estimate as its cost — a quantity that fluctuates on healthy
    /// runs rather than descending — so it fired constantly and silently changed the answer. Every
    /// other test here still passed while the output was ~9x too smooth.
    #[test]
    fn damping_guard_is_inert_on_well_conditioned_data() {
        let n = 16;
        let field: Vec<f64> = (0..n * n * n)
            .map(|i| {
                let (x, y, z) = (i % n, (i / n) % n, i / (n * n));
                0.02 * (((x + 2 * y + 3 * z) as f64) * 0.3).sin()
            })
            .collect();
        // Structured magnitude, so the wavelet morphology prior is exercised — that prior is
        // what destabilises the solve on real data, so it must be part of the inertness check.
        let mag: Vec<f64> = (0..n * n * n)
            .map(|i| {
                let (x, y) = ((i % n) as f64, ((i / n) % n) as f64);
                100.0 + 30.0 * (x * 0.4).cos() + 20.0 * (y * 0.25).sin()
            })
            .collect();
        let mut mask = vec![0u8; n * n * n];
        for z in 4..12 {
            for y in 4..12 {
                for x in 4..12 {
                    mask[x + y * n + z * n * n] = 1;
                }
            }
        }
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = AmpPeParams { max_linearization_ite: 4, ..Default::default() };

        GUARD_FIRINGS.with(|c| c.set(0));
        let chi = amp_pe(&field, &mask, Some(&mag), &grid, (0.0, 0.0, 1.0), &params, |_, _| {});
        let fired = GUARD_FIRINGS.with(|c| c.get());

        assert_eq!(
            fired, 0,
            "divergence guard fired {fired}x on well-conditioned data — the trigger has been \
             loosened, and every AMP-PE result will have silently changed"
        );
        assert!(chi.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn amp_pe_finite_on_ramp() {
        let n = 16;
        let field: Vec<f64> = (0..n * n * n).map(|i| ((i % 7) as f64 - 3.0) * 0.01).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = AmpPeParams { max_linearization_ite: 3, ..Default::default() };
        let chi = amp_pe(&field, &mask, None, &grid, (0.0, 0.0, 1.0), &params, |_, _| {});
        for &v in &chi {
            assert!(v.is_finite());
        }
    }
}

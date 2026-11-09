//! Weak-Harmonic QSM (WH-QSM) dipole inversion
//!
//! Jointly estimates the susceptibility map `x` AND a residual harmonic
//! background field `phi_h`, so that any harmonic (Laplacian-null) field
//! remaining after background-field removal is absorbed into `phi_h` instead
//! of corrupting the susceptibility estimate. This makes the reconstruction
//! robust to imperfect background-field removal.
//!
//! The optimization solves a nonlinear total-variation problem via ADMM,
//! with an additional weak-harmonic regularization term coupling `x` and the
//! harmonic field. The data-fidelity term is nonlinear (sine of the phase),
//! and is solved with an inner Newton iteration.
//!
//! Reference:
//! Milovic, C., Bilgic, B., Zhao, B., Acosta-Cabronero, J., Tejos, C. (2019).
//! "Weak-harmonic regularization for quantitative susceptibility mapping."
//! Magnetic Resonance in Medicine, 81(2):1399-1411.
//! <https://doi.org/10.1002/mrm.27483>
//!
//! Ported from FANSI's `WH_nlTV.m` (https://gitlab.com/cmilovic/FANSI-toolbox).
//!
//! # Units
//! Like the other inversions, WH-QSM takes the local field in ppm and returns ppm. `WH_nlTV.m`
//! works on the phase in radians, so the field is converted with `2*pi*gamma*b0*te` first and the
//! result converted back at the end; left in ppm, `sin(x) ~ x` and the data term would not be
//! nonlinear. `b0` and `echo_times` are scan parameters that the pipeline fills from metadata, and
//! [`WhQsmParams::te`] picks the echo time the conversion uses (by default the reference's choice;
//! see there).
//!
//! # Data weight
//! `WH_nlTV.m` squares `params.weight` (recommended: the magnitude). FANSI's WH-QSM example
//! (`script_DICOM_Philips.m`, the only script that calls `WH_nlTV`) sets
//! `weight = m0.*mask./cnoise`, divided by its maximum, where `m0` is the proton density
//! from its nonlinear T2* fit (`nlT2sTV`) and `cnoise` the complex-domain noise from its
//! nonlinear multi-echo field fit (`nlme_L2gradient`). The dipole stage has neither map. The
//! default ([`WhQsmWeight::Flat`]) is `WH_nlTV.m`'s own default when no weight is given: 1 over
//! the whole volume (`W = ones(N)`, lines 70-75), with the mask given separately (it confines
//! `x` and the harmonic penalty `beta`), as the QSM-CI original runs it. The data term outside
//! the mask is part of the method: an earlier port weighted only the mask, which with the
//! continuous kernel left it 21.4 % / 50.7 % (sim / KKI χ NRMSE) from the verified original;
//! a numpy transcription of `WH_nlTV.m` on the original's own inputs with the mask as the
//! weight diverges on sim, and with `ones(N)` and FANSI's kernel it is 0.0006 % from it
//! (QSM.rs#402). The magnitude weights are options:
//! [`WhQsmWeight::MaxNormalised`] is the example's formula with the magnitude for `m0` and no
//! noise term, `mask.*mag / max(mask.*mag)`, and [`WhQsmWeight::InVivo`] FANSI's
//! `script_invivo.m` form, `min(2*mask*mag/max(mag), 1)`, as NDI uses. How the weight and the
//! echo time ([`WhQsmParams::te`]) interact on the phantom is noted there.
//! [`WhQsmWeight::FlatInMask`] (1 in the mask, no magnitude) is an in-vivo option: it drops the
//! data term outside the brain, which in vivo carries residual background (measurements on the
//! variant).
//!
//! # Newton safeguard
//! The inner Newton step for `z2` (`WH_nlTV.m` l. 258-268) minimises, voxel by voxel,
//! `W*(1 - cos(z2 - phase)) + mu2/2*(z2 - rhs/mu2)^2`, whose minimiser lies within `W/mu2` of
//! `rhs/mu2` (where the derivative `W*sin(z2 - phase) + mu2*(z2 - rhs/mu2)` vanishes). With
//! the flat weight, a field that is non-zero outside the mask (QSM-CI's simulated local
//! field, up to ~6 rad) puts some voxels where `W*cos + mu2` nears 0, and the unguarded step
//! bounces: on that sim the original's Newton happened to settle after ~24 inner steps, while
//! the port's (same formula, FFT rounding apart) ran to its 50-step cap at 1e16 and the solve
//! went NaN. The port clamps each Newton iterate to that interval. This changes nothing where
//! the step stays inside it (it does wherever the original converges normally), and the fixed
//! point is the same minimiser; it is not in `WH_nlTV.m`. With it the port is 0.0006 % (sim)
//! and 0.0009 % (KKI, no clamping needed) from the verified original (χ in radians, QSM.rs#402).
//!
//! # Finite differences
//! As in `WH_nlTV.m`, the gradient and Laplacian are unit-spacing differences whatever the
//! voxel size; only the dipole kernel uses it.
//!
//! # Reference pipeline
//! FANSI `script_DICOM_Philips.m` (commit 93044a3) runs `WH_nlTV` on the same local field and
//! TE scaling as nlTV (see [`super::fansi`](mod@super::fansi): SEGUE, multi-echo fit, MSMV background removal,
//! `deltaTE` with "any individual echo time" allowed, L364), with `mask = mask2`, `beta = 200`,
//! `muh = 5` and the advice that beta is "usually ... 20 to 50 times larger than muh" and that
//! alpha1 may be slightly lower than FANSI's (L425-436). This port uses `WH_nlTV.m`'s own
//! defaults for beta (150) and muh (5), not the script's beta of 200. Without `params.mask`,
//! `WH_nlTV.m` takes the mask as `W > 0`; here the mask is always given.
//!
//! # Parity
//! Against the verified original (QSM-CI pair `whqsm`; FANSI-toolbox 93044a3, identical to
//! licensed MATLAB R2023b), final-map χ NRMSE at both sides' defaults on QSM-CI's simulated head
//! and the KKI travelling head (QSM-CI parity run 41403629314, QSM.rs#399–#403): sim 0.0018 %, KKI 0.0038 %, from
//! 21.4 % / 50.7 % (sim / KKI) before. Two causes: the kernel, and `WH_nlTV.m`'s flat weight over the whole volume (see Data weight); the Newton safeguard keeps the sim run from diverging (see Newton safeguard). What remains is
//! single (MATLAB) vs double (here) precision and FANSI's γ, 42.58 against the crate's 42.576
//! (the wrappers keep theirs). Fixture: `tests/fansi_qsmci_parity.rs` (relative L2 below 1e-5
//! on an oblique, anisotropic phantom).
//!
//! # Pipeline rules
//!
//! The facts above that a configuration can depart from are published as rules in
//! [`crate::pipeline::rules::RULES`] (each with its strength, basis, reason and source): `fansi.no-laplacian-multi-echo`, `whqsm.flat-weight-te-spacing`, `whqsm.magnitude-weight-te-spacing`.

use crate::inversion::admm::{fansi_unit_ee2, prepare_fansi_kernel, FansiKernel};
use crate::utils::gradient::{bdiv_inplace, fgrad_inplace};
use crate::utils::{apply_mask_zero, shrink};
use crate::{EchoTime, Grid};
use num_complex::Complex64;

/// The data weight (see the module docs).
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WhQsmWeight {
    /// 1 everywhere, no magnitude: `WH_nlTV.m`'s own default when `params.weight` is not
    /// given (`W = ones(N)`, then squared; lines 70-75). The data term covers the whole
    /// volume; the mask still confines `x` and the harmonic penalty.
    #[default]
    Flat,
    /// `mask.*mag / max(mask.*mag)`: FANSI's WH-QSM example (`script_DICOM_Philips.m`)
    /// without its T2*-fit and noise maps. Needs the magnitude.
    MaxNormalised,
    /// `min(2*mask*mag/max(mag), 1)`: FANSI's `script_invivo.m`, as NDI uses. Needs the
    /// magnitude.
    InVivo,
    /// 1 inside the mask, 0 outside, no magnitude: the flat weight without its data term
    /// outside the brain. Not the original (see the module docs, *Data weight*). In vivo the
    /// whole-volume data term adds smooth shading: on the MGH pilot scans (ROMEO + V-SHARP field)
    /// WH-QSM's correlation with iLSQR rose from 0.58–0.60 to 0.72–0.75 and the scale against it
    /// from 1.29–1.37 to 1.04–1.20. On QSM-CI's simulated head (whose local field is true outside
    /// the mask too) it is 16 % (χ NRMSE) from the original and further from the truth (38.2 %
    /// against 33.6 %, r 0.927 against 0.945). A trade-off, so an option (2026-11-09,
    /// `~/work/emtp-v11/regressions-dipole.md`).
    FlatInMask,
}

impl WhQsmWeight {
    /// Whether this weight is made from the magnitude.
    pub fn needs_magnitude(self) -> bool {
        !matches!(self, WhQsmWeight::Flat | WhQsmWeight::FlatInMask)
    }
}

/// WH-QSM parameters.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct WhQsmParams {
    /// TV regularization weight.
    pub alpha1: f64,
    /// ADMM penalty for the TV splitting (mu).
    pub mu1: f64,
    /// ADMM penalty for the data-fidelity splitting.
    pub mu2: f64,
    /// Weak-harmonic ROI penalty (constrains the harmonic field inside the mask). Default 150,
    /// `WH_nlTV.m`'s own default. FANSI's demo (`script_DICOM_Philips.m` l. 432-433) sets
    /// `beta = 200`, `muh = 5`, as a starting point to "fine tune" ("usually, beta is between 20
    /// to 50 times larger than muh"); the function's default is kept.
    pub beta: f64,
    /// ADMM penalty for the harmonic-field splitting.
    pub muh: f64,
    /// Maximum outer iterations.
    pub max_iter: usize,
    /// Percent-update stopping tolerance on `x`.
    pub tol_update: f64,
    /// Inner Newton stopping tolerance.
    pub tol_delta: f64,
    /// Magnitude normalisation for the data weight.
    pub weight: WhQsmWeight,
    /// Field strength in Tesla (scan parameter, for the radian conversion).
    pub b0: f64,
    /// Echo times in seconds, in acquisition order (scan parameter, for the radian
    /// conversion).
    pub echo_times: Vec<f64>,
    /// Which echo time the radian conversion uses. Default [`EchoTime::First`]. FANSI's
    /// multi-echo example, `script_DICOM_Philips.m`, scales the field by `deltaTE =
    /// TE(echo2) - TE(echo1)` (FANSI-toolbox 93044a3, line 364) before `WH_nlTV` (line 436), but
    /// with a weight from M0 and the fit noise, and its comment on that line says "You may use
    /// also any individual echo time". With the default flat weight ([`WhQsmWeight::Flat`]), the
    /// first echo does better: on QSM-CI's simulated head (7 T, echoes 4/12/20/28 ms, its true
    /// local field and mask; FANSI's kernel, the flat weight over the whole volume, QSM.rs#402)
    /// χ NRMSE 33.6 % and r 0.945 at the first echo against 42.0 % and 0.914 at the spacing.
    /// (Before QSM.rs#402, with the continuous kernel and the weight only in the mask, the
    /// spacing diverged on the qsm-forward phantom, r 0.07, and the first echo gave XSIM 0.754,
    /// NRMSE 35.5 %.) With a magnitude weight the echo spacing does better than
    /// the first echo (echo-1 magnitude, [`WhQsmWeight::MaxNormalised`]; qsm-forward phantom,
    /// integration-test settings, FANSI's kernel, QSM.rs#418; XSIM / NRMSE / r 0.676 / 48.9% /
    /// 0.876 vs 0.638 / 49.9% / 0.873; with the continuous kernel, #276: XSIM 0.673 vs 0.637).
    pub te: EchoTime,
    /// The dipole kernel. Default [`FansiKernel::Discrete`], FANSI's
    /// `dipole_kernel_angulated` as the original is run with; [`FansiKernel::Continuous`] is
    /// the crate's continuous kernel (see [`FansiKernel`] for the decision and the measurements).
    pub kernel: FansiKernel,
}

impl Default for WhQsmParams {
    /// `WH_nlTV.m` defaults (mu2 = 1, muh = 5, beta = 150, 150 iterations, tolUpdate 0.1,
    /// tolDelta 1e-6). `WH_nlTV.m` has no default for alpha1; mu1 is its default 100*alpha1.
    fn default() -> Self {
        Self {
            alpha1: 2e-4,
            mu1: 2e-2,
            mu2: 1.0,
            beta: 150.0,
            muh: 5.0,
            max_iter: 150,
            tol_update: 0.1,
            tol_delta: 1e-6,
            weight: WhQsmWeight::Flat,
            b0: 3.0,
            echo_times: vec![0.02],
            te: EchoTime::First,
            kernel: FansiKernel::Discrete,
        }
    }
}

/// L2 norm of a real vector.
#[inline]
fn norm(v: &[f64]) -> f64 {
    v.iter().map(|&a| a * a).sum::<f64>().sqrt()
}

/// Weak-Harmonic QSM dipole inversion.
///
/// Jointly estimates the susceptibility map and a residual harmonic
/// background field, returning the susceptibility map (`out.x`).
///
/// # Arguments
/// * `local_field` - Local field in ppm (nx * ny * nz).
/// * `mask` - Binary mask (nx * ny * nz), 1 = inside ROI.
/// * `magnitude` - Magnitude image, for a magnitude data weight ([`WhQsmParams::weight`]);
///   ignored by the default flat weight, which weights every voxel in the mask by 1.
/// * `grid` - Volume grid (dimensions and voxel sizes).
/// * `bdir` - B0 field direction.
/// * `params` - WH-QSM parameters.
/// * `progress` - Progress callback `(iteration, max_iter)`.
///
/// # Returns
/// Susceptibility map in ppm.
///
/// # Panics
/// If `params.weight` needs the magnitude and `magnitude` is `None`, or if `params.te` cannot
/// be resolved from `params.echo_times`.
pub fn whqsm(
    local_field: &[f64],
    mask: &[u8],
    magnitude: Option<&[f64]>,
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &WhQsmParams,
    mut progress: impl FnMut(usize, usize),
) -> Vec<f64> {
    let n = grid.n_total();

    let alpha = params.alpha1;
    let mu = params.mu1;
    let mu2 = params.mu2;
    let beta = params.beta;
    let muh = params.muh;
    let alpha_over_mu = alpha / mu;

    // Spectral operators: k = real dipole kernel D(k) (DC=0), ee2 = FANSI's EE2. WH_nlTV.m's
    // E1..E3 are unit-spacing differences whatever the voxel size, so the gradient and
    // divergence below run on a unit-voxel grid too; only the dipole kernel uses the voxel size.
    let (mut fft_ws, k) = prepare_fansi_kernel(grid, bdir, params.kernel);
    let ee2 = fansi_unit_ee2(grid.dims);
    let ugrid = Grid::new(grid.nx(), grid.ny(), grid.nz(), 1.0, 1.0, 1.0);

    let mask_f: Vec<f64> = mask.iter().map(|&m| if m != 0 { 1.0 } else { 0.0 }).collect();

    // W = weight.^2 (WH_nlTV.m line 75), weight as `params.weight` says: ones over the whole
    // volume (WH_nlTV.m's default), or from the magnitude (zero outside the mask).
    assert!(
        !params.weight.needs_magnitude() || magnitude.is_some(),
        "WH-QSM: the {:?} weight needs the magnitude", params.weight
    );
    let magnitude = if params.weight.needs_magnitude() { magnitude } else { None };
    let mag_max = magnitude.map_or(0.0, |m| match params.weight {
        WhQsmWeight::Flat | WhQsmWeight::FlatInMask => 0.0,
        WhQsmWeight::MaxNormalised => (0..n)
            .filter(|&i| mask[i] != 0)
            .map(|i| m[i])
            .fold(0.0f64, f64::max),
        WhQsmWeight::InVivo => m.iter().cloned().fold(0.0f64, f64::max),
    });
    let w: Vec<f64> = (0..n)
        .map(|i| {
            if params.weight == WhQsmWeight::Flat {
                return 1.0;
            }
            if mask[i] == 0 {
                return 0.0;
            }
            let v = match magnitude {
                Some(m) if mag_max > 0.0 => match params.weight {
                    WhQsmWeight::Flat | WhQsmWeight::FlatInMask => 1.0,
                    WhQsmWeight::MaxNormalised => m[i] / mag_max,
                    WhQsmWeight::InVivo => (2.0 * m[i] / mag_max).min(1.0),
                },
                _ => 1.0,
            };
            v * v
        })
        .collect();

    // ppm -> radians, as WH_nlTV.m expects.
    // FANSI's scripts use `gyro = 2*pi*42.58`; QSM.rs uses GAMMA_BAR_MHZ_PER_T, which is closer
    // to the CODATA shielded-proton value (see `crate::constants`).
    let ppm_to_rad = 2.0 * std::f64::consts::PI * crate::constants::GAMMA_BAR_MHZ_PER_T * params.b0 * params.te.seconds(&params.echo_times, "WH-QSM");
    let phase: Vec<f64> = local_field.iter().map(|&f| f * ppm_to_rad).collect();

    // Denominator for x-subproblem: 1e-30 + mu2*k^2 + mu*ee2
    let x_denom: Vec<f64> = (0..n)
        .map(|i| 1e-30 + mu2 * k[i] * k[i] + mu * ee2[i])
        .collect();
    // Denominator for phi_h subproblem: eps + mu2 + muh*ee2^2
    let phi_denom: Vec<f64> = (0..n)
        .map(|i| 1e-30 + mu2 + muh * ee2[i] * ee2[i])
        .collect();

    // State variables.
    let mut x = vec![0.0f64; n];
    let mut z_dx = vec![0.0f64; n];
    let mut z_dy = vec![0.0f64; n];
    let mut z_dz = vec![0.0f64; n];
    let mut s_dx = vec![0.0f64; n];
    let mut s_dy = vec![0.0f64; n];
    let mut s_dz = vec![0.0f64; n];

    let mut phi_h = vec![0.0f64; n];
    let mut z_h = vec![0.0f64; n];
    let mut s_h = vec![0.0f64; n];

    // z2 = phase .* W ; s2 = 0
    let mut z2: Vec<f64> = (0..n).map(|i| phase[i] * w[i]).collect();
    let mut s2 = vec![0.0f64; n];

    // Scratch buffers.
    let mut cbuf = vec![Complex64::new(0.0, 0.0); n]; // general complex scratch
    let mut cbuf2 = vec![Complex64::new(0.0, 0.0); n]; // second complex scratch
    let mut fx = vec![Complex64::new(0.0, 0.0); n]; // Fx = fft(x)
    let mut real_scratch = vec![0.0f64; n];
    let mut gx = vec![0.0f64; n];
    let mut gy = vec![0.0f64; n];
    let mut gz = vec![0.0f64; n];
    let mut dg = vec![0.0f64; n]; // divergence buffer
    let mut dx_real = vec![0.0f64; n]; // Dx = real(ifft(k .* Fx))
    let mut ee2_phi = vec![0.0f64; n]; // real(ifft(EE2 .* Fphi_h))

    for t in 0..params.max_iter {
        progress(t + 1, params.max_iter);

        // --- x-subproblem (masked) ---
        // numerator k-space = mu * fft(bdiv(z_d - s_d)) + K .* fft(z2 - s2 - phi_h)
        // x = mask .* real(ifft( numerator ./ x_denom ))

        // bdiv(z_d - s_d) in real space, then fft.
        for i in 0..n {
            gx[i] = z_dx[i] - s_dx[i];
            gy[i] = z_dy[i] - s_dy[i];
            gz[i] = z_dz[i] - s_dz[i];
        }
        bdiv_inplace(&mut dg, &gx, &gy, &gz, &ugrid);
        for i in 0..n {
            cbuf[i] = Complex64::new(dg[i], 0.0);
        }
        fft_ws.fft3d(&mut cbuf); // cbuf = fft(bdiv(z_d - s_d))

        // Dt_kspace source in real space = z2 - s2 - phi_h ; fft it.
        for i in 0..n {
            fx[i] = Complex64::new(z2[i] - s2[i] - phi_h[i], 0.0);
        }
        fft_ws.fft3d(&mut fx); // fx = fft(z2 - s2 - phi_h)

        // numerator ./ x_denom  (k real -> K.* = k[i] * fx[i]; conj(K)=K since real)
        for i in 0..n {
            // Minus on the gradient term: adjoint of crate `fgrad` is `-bdiv` (matches
            // QSM.rs TV-ADMM). `+bdiv` doubles the effective regularization. See fansi.rs.
            let num = -mu * cbuf[i] + k[i] * fx[i];
            // Guard the dipole null-space (DC/singular bins): x_denom -> ~0 there
            // (both dipole kernel and Laplacian vanish). Zero it instead of
            // dividing FFT round-off by ~1e-30, which blows up and (via the
            // harmonic-field coupling) diverges the whole solve.
            cbuf[i] = if x_denom[i] > 1e-20 { num / x_denom[i] } else { Complex64::new(0.0, 0.0) };
        }
        fft_ws.ifft3d(&mut cbuf);

        // Compute x_update = 100 * norm(x - x_prev) / norm(x) after updating.
        // Keep old x in real_scratch for the diff.
        real_scratch.copy_from_slice(&x);
        for i in 0..n {
            x[i] = mask_f[i] * cbuf[i].re;
        }
        // x_update.
        let mut diff_norm = 0.0;
        for i in 0..n {
            let d = x[i] - real_scratch[i];
            diff_norm += d * d;
        }
        let diff_norm = diff_norm.sqrt();
        // As in WH_nlTV.m: 0/0 is NaN when x is all zero, and NaN < tol is false.
        let x_update = 100.0 * diff_norm / norm(&x);
        if x_update < params.tol_update {
            progress(t + 1, t + 1);
            break;
        }

        if t + 1 < params.max_iter {
            // Fx = fft(x)
            for i in 0..n {
                fx[i] = Complex64::new(x[i], 0.0);
            }
            fft_ws.fft3d(&mut fx);

            // Forward gradient of x (== [real(ifft(E1.*Fx)), ...]).
            fgrad_inplace(&mut gx, &mut gy, &mut gz, &x, &ugrid);

            // z_d = shrink(grad + s_d, alpha_over_mu) ; s_d += grad - z_d.
            for i in 0..n {
                let ax = gx[i] + s_dx[i];
                let ay = gy[i] + s_dy[i];
                let az = gz[i] + s_dz[i];
                z_dx[i] = shrink(ax, alpha_over_mu);
                z_dy[i] = shrink(ay, alpha_over_mu);
                z_dz[i] = shrink(az, alpha_over_mu);
                s_dx[i] += gx[i] - z_dx[i];
                s_dy[i] += gy[i] - z_dy[i];
                s_dz[i] += gz[i] - z_dz[i];
            }

            // Dx = real(ifft(K .* Fx))
            for i in 0..n {
                cbuf[i] = k[i] * fx[i];
            }
            fft_ws.ifft3d(&mut cbuf);
            for i in 0..n {
                dx_real[i] = cbuf[i].re;
            }

            // rhs_z2 = mu2 * (Dx + s2 + phi_h) ; z2 = rhs_z2 / mu2 = Dx + s2 + phi_h.
            // Store rhs_z2 in real_scratch.
            for i in 0..n {
                real_scratch[i] = mu2 * (dx_real[i] + s2[i] + phi_h[i]);
                z2[i] = real_scratch[i] / mu2;
            }

            // Newton iteration on z2 (nonlinear data term).
            let mut delta = f64::INFINITY;
            let mut inn = 0;
            while delta > params.tol_delta && inn < 50 {
                inn += 1;
                let norm_old = norm(&z2);
                // update = (W.*sin(z2-phase) + mu2*z2 - rhs_z2) ./ (W.*cos(z2-phase) + mu2)
                let mut upd_norm2 = 0.0;
                for i in 0..n {
                    let dphi = z2[i] - phase[i];
                    let numr = w[i] * dphi.sin() + mu2 * z2[i] - real_scratch[i];
                    let denr = w[i] * dphi.cos() + mu2;
                    // Safeguard (not in WH_nlTV.m): keep the iterate in the interval that holds
                    // the minimiser, |z2 - rhs/mu2| <= W/mu2 (from W*sin(z2-phase) +
                    // mu2*(z2 - rhs/mu2) = 0). See "Newton safeguard" in the module docs.
                    let r = real_scratch[i] / mu2;
                    let b = w[i] / mu2;
                    let z_new = (z2[i] - numr / denr).clamp(r - b, r + b);
                    let u = z2[i] - z_new;
                    z2[i] = z_new;
                    upd_norm2 += u * u;
                }
                let upd_norm = upd_norm2.sqrt();
                delta = if norm_old > 0.0 {
                    upd_norm / norm_old
                } else {
                    // guard: if z2 became zero, stop.
                    0.0
                };
            }

            // --- Harmonic field update (phi_h) ---
            // Fphi_h = ( muh*EE2.*fft(z_h - s_h) + mu2*fft(z2 - s2) - mu2*K.*Fx )
            //          ./ (eps + mu2 + muh*EE2.^2)
            // fft(z_h - s_h)
            for i in 0..n {
                cbuf[i] = Complex64::new(z_h[i] - s_h[i], 0.0);
            }
            fft_ws.fft3d(&mut cbuf); // cbuf = fft(z_h - s_h)

            // fft(z2 - s2)  -> reuse a fresh complex buffer via dx_real? Need another buffer.
            // Use `phi_h`-sized complex scratch: allocate once via a persistent buffer.
            // We reuse `cbuf` after combining; but we need fft(z2 - s2) simultaneously.
            // Compute fft(z2 - s2) into a second complex buffer.
            // (fx currently holds Fx; we still need Fx for the -mu2*K.*Fx term.)
            for i in 0..n {
                cbuf2[i] = Complex64::new(z2[i] - s2[i], 0.0);
            }
            fft_ws.fft3d(&mut cbuf2); // cbuf2 = fft(z2 - s2)

            // Fphi_h (store into cbuf).
            for i in 0..n {
                let numer =
                    muh * ee2[i] * cbuf[i] + mu2 * cbuf2[i] - mu2 * (k[i] * fx[i]);
                cbuf[i] = numer / phi_denom[i];
            }
            // cbuf now = Fphi_h.

            // EE2_phi = real(ifft(EE2 .* Fphi_h)) : first compute with EE2 multiply.
            for i in 0..n {
                cbuf2[i] = ee2[i] * cbuf[i];
            }
            fft_ws.ifft3d(&mut cbuf2);
            for i in 0..n {
                ee2_phi[i] = cbuf2[i].re;
            }

            // phi_h = real(ifft(Fphi_h))
            fft_ws.ifft3d(&mut cbuf);
            for i in 0..n {
                phi_h[i] = cbuf[i].re;
            }

            // z_h = muh*(EE2_phi + s_h) ./ (muh + beta*mask)   (mask 0/1)
            for i in 0..n {
                z_h[i] = muh * (ee2_phi[i] + s_h[i]) / (muh + beta * mask_f[i]);
            }

            // --- dual updates ---
            // s2 = s2 + real(ifft(K.*Fx)) - z2 + phi_h  (= s2 + Dx - z2 + phi_h)
            for i in 0..n {
                s2[i] = s2[i] + dx_real[i] - z2[i] + phi_h[i];
            }
            // s_h = s_h + EE2_phi - z_h
            for i in 0..n {
                s_h[i] = s_h[i] + ee2_phi[i] - z_h[i];
            }
        }
    }

    // Back to ppm.
    for v in x.iter_mut() {
        *v /= ppm_to_rad;
    }
    apply_mask_zero(&mut x, mask);

    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whqsm_zero_field() {
        // Zero field should give (near) zero susceptibility.
        let n = 8;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = WhQsmParams {
            max_iter: 15,
            ..Default::default()
        };

        let chi = whqsm(&field, &mask, None, &grid, (0.0, 0.0, 1.0), &params, |_, _| {});

        for &val in chi.iter() {
            assert!(
                val.abs() < 1e-6,
                "Zero field should give ~zero chi, got {}",
                val
            );
        }
    }

    #[test]
    fn test_whqsm_finite() {
        // Small ramp field -> all outputs finite.
        let n = 8;
        let field: Vec<f64> = (0..n * n * n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = WhQsmParams {
            max_iter: 15,
            ..Default::default()
        };

        let chi = whqsm(&field, &mask, None, &grid, (0.0, 0.0, 1.0), &params, |_, _| {});

        for (i, &val) in chi.iter().enumerate() {
            assert!(val.is_finite(), "Chi should be finite at index {}", i);
        }
    }

    #[test]
    fn test_whqsm_uniform_magnitude_is_mask_weight() {
        // A uniform magnitude normalises to 1 either way, the same weight as the flat default.
        let n = 8;
        let field: Vec<f64> = (0..n * n * n).map(|i| ((i % 7) as f64 - 3.0) * 0.01).collect();
        let mask = vec![1u8; n * n * n];
        let mag = vec![3.0; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = WhQsmParams { max_iter: 10, ..Default::default() };

        let a = whqsm(&field, &mask, None, &grid, (0.0, 0.0, 1.0), &params, |_, _| {});
        for weight in [WhQsmWeight::MaxNormalised, WhQsmWeight::InVivo] {
            let p = WhQsmParams { weight, ..params.clone() };
            let b = whqsm(&field, &mask, Some(&mag), &grid, (0.0, 0.0, 1.0), &p, |_, _| {});
            for (x, y) in a.iter().zip(&b) {
                assert!((x - y).abs() < 1e-12);
            }
        }
        // the flat default ignores a magnitude it is given
        let mag_varied: Vec<f64> = (0..n * n * n).map(|i| 1.0 + (i % 5) as f64).collect();
        let c = whqsm(&field, &mask, Some(&mag_varied), &grid, (0.0, 0.0, 1.0), &params, |_, _| {});
        assert_eq!(a, c);
    }

    #[test]
    #[should_panic(expected = "needs the magnitude")]
    fn test_whqsm_magnitude_weight_needs_magnitude() {
        let n = 4;
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let p = WhQsmParams { weight: WhQsmWeight::MaxNormalised, max_iter: 1, ..Default::default() };
        whqsm(&[0.0; 64], &[1u8; 64], None, &grid, (0.0, 0.0, 1.0), &p, |_, _| {});
    }

    /// `FlatInMask` needs no magnitude, equals a uniform magnitude weight (1 in the mask, 0
    /// outside), and differs from the whole-volume `Flat` default when the mask is not everything.
    #[test]
    fn test_whqsm_flat_in_mask_weight() {
        assert!(!WhQsmWeight::FlatInMask.needs_magnitude());
        let n = 8;
        let field: Vec<f64> = (0..n * n * n).map(|i| ((i % 7) as f64 - 3.0) * 0.01).collect();
        let mask: Vec<u8> = (0..n * n * n).map(|i| u8::from(i % n > 1 && i % n < n - 2)).collect();
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = WhQsmParams { max_iter: 10, ..Default::default() };
        let run = |weight, mag: Option<&[f64]>| {
            whqsm(&field, &mask, mag, &grid, (0.0, 0.0, 1.0), &WhQsmParams { weight, ..params.clone() }, |_, _| {})
        };
        let in_mask = run(WhQsmWeight::FlatInMask, None);
        let uniform = run(WhQsmWeight::MaxNormalised, Some(&vec![2.0; n * n * n]));
        for (x, y) in in_mask.iter().zip(&uniform) {
            assert!((x - y).abs() < 1e-12);
        }
        assert_ne!(in_mask, run(WhQsmWeight::Flat, None));
    }
}

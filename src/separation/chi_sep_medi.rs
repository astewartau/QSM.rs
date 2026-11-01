//! χ-separation (MEDI): the SNU χ-separation toolbox's `chi_sep_MEDI`
//!
//! [`chi_sep_medi`] implements **`chi_sep_MEDI` of the χ-separation toolbox v1.1.3** (SNU-LIST,
//! Jongho Lee's lab; `functions/corefunc/chi_sep_MEDI.p` and `chi_sep_MEDI_func.p`), as its
//! script `Chisep_script.m` calls it (l. 153-161):
//! `chi_sep_MEDI(mag, local_field_hz, r2prime, N_std, mask, params, option_data)` with
//! `params.lambda = 1`, `params.lambda_CSF = 1`, `option_data.qsm`, `option_data.mask_CSF` and
//! `option_data.N_std`. The toolbox ships as protected P-code; with the authors' permission the
//! algorithm was recovered by black-box probing in licensed MATLAB R2023b: the profiler (which
//! lists the bodies of the P-code's anonymous functions verbatim), logging wrappers on the
//! functions it calls that save the caller's whole workspace (`evalin('caller', 'save ...')`),
//! and replaying the operators captured in those workspaces (their function handles stay
//! callable after `load`). A MATLAB re-implementation built from that reproduces
//! `chi_sep_MEDI` to 3·10⁻¹⁶ relative on six phantoms (see *Agreement with the toolbox*).
//!
//! Earlier QSM.rs versions ported Shin et al. 2021's Supplementary Methods instead, with their
//! own weights, λ and no QSM, CSF mask or noise map; the toolbox's algorithm below differs in
//! every step after the forward model.
//!
//! # The algorithm
//!
//! The unknowns are `x+ = χ+ ≥ 0` and `x− = |χ−| ≥ 0` in ppm. With `D` the dipole convolution in
//! Hz (`(CF/10⁶)·F⁻¹ D F`) and `D_r,±` the relaxometric constants (Hz/ppm):
//!
//! 1. **Defaults** (`chi_sep_MEDI`): `B0 = CF/42.58·10⁶`; `D_r,pos = D_r,neg = 137·(B0/2.9)`
//!    ([`toolbox_default_dr`]); `pad_size` 5, `tol_norm_ratio` 0.03, `max_iter` 30,
//!    `cg_max_iter` 100, `cg_tol` 0.05 (see [`ChiSepMediParams`]); and, not parameters, at least
//!    [`CHI_SEP_MEDI_MIN_ITER`] = 6 iterations, `ε` = 10⁻⁵ and `wei_mag` = 1/(2π).
//! 2. **Edge masks** on the input grid: `wG = gradient_mask(mag, mask, 0.9)` (MEDI's
//!    `gradient_mask.m`: `|∇(mag·mask)|` thresholded so that 90 % of the mask's voxel count of
//!    gradient components lie above the level; 1 = not an edge). The R2′ edge mask `wG_r2p` is
//!    the same magnitude mask.
//! 3. **Padding**: every volume is zero-padded by `pad_size` voxels at both ends of the **third
//!    axis only**; `wG`, `wG_r2p` are padded with 1. Everything below runs on that grid. Exactly
//!    zero R2′ voxels (including the padding) become `eps` (2.2·10⁻¹⁶).
//! 4. **Weights and targets**: `m = dataterm_mask(N_std, mask)` (MEDI's: `mask/N_std`, 0 where not
//!    finite, divided by its mean over the mask); R2′ residual weight `wei_mag·m`, field residual
//!    weight `m`. Targets: `R2′·mask` where `R2′ > 0` (0 elsewhere) and the local field in Hz.
//! 5. **Dipole kernel**: MEDI's `dipole_kernel(N, vs, B0_dir, 'kspace')`: the k-grid
//!    `(−N/2 : N/2−1)/(N·vs)` per axis, `D = 1/3 − (k·B0)²/|k|²` (0 at `k = 0`), `fftshift`ed. On
//!    an axis with an odd number of voxels that grid steps through half-integers and misses
//!    `k = 0`, so the kernel is sampled half a frequency bin off and the convolution has an
//!    imaginary part; it is kept as the toolbox keeps it (complex arithmetic throughout, real part
//!    taken where the toolbox takes it).
//! 6. **Initial guess** from the QSM `χ` (`option_data.qsm`): the voxelwise solution of
//!    `D_r,pos·x+ + D_r,neg·x− = R2′`, `x+ − x− = χ`, each clamped at 0:
//!    `x+ = max((R2′ + D_r,neg·χ)/(D_r,pos + D_r,neg), 0)`, `x− = max((R2′ − D_r,pos·χ)/(…), 0)`
//!    (with the raw R2′, negative values included). The iterate is that, masked. A copy with the
//!    CSF voxels set to 0 is what the first iteration's L1 weights see (a quirk of the toolbox:
//!    the iterate itself keeps the CSF values).
//! 7. **Gauss-Newton** while (`res > tol` or `iter < 6`) and `iter < max_iter`:
//!    - L1 weights `V_s = 1/√(|wG·∇x_s|² + ε)` for `s = +, −`, and `V_t = 1/√(|wG·∇(x+ − x−)|² + ε)`
//!      (∇: MEDI's `fgrad`, forward differences scaled by the voxel size, 0 at the last voxel).
//!    - Gradients `b_s` of
//!      `‖wei_mag·m·(D_r,pos·x+ + D_r,neg·x− − R2′)‖² + ‖m·(D x+ − D x− − f)‖²
//!       + λ Σ_s ‖wG V_s^½ ∇x_s‖² + 2λ ‖wG V_t^½ ∇(x+ − x−)‖² + λ_CSF Σ_s ‖LT(|x_s|)‖²`
//!      (as the toolbox evaluates them: `2·D_rᵀ M²·residual`, `∇ᵀ(wG V wG ∇·)` with `∇ᵀ` the exact
//!      adjoint of `fgrad`), where `LT(v) = M_CSF·(v − mean(v over M_CSF))`.
//!    - One CG solve **per source** (`cgsolve.m`, MEDI's: zero start, last iterate, residual
//!      recomputed every 50 iterations): `A_s dx_s = −b_s`, where `A_s` is the same expression's
//!      linearisation in `x_s` with the other source's step set to zero, **CSF term included**
//!      (`λ_CSF·LT(LT(|dx|))`, nonlinear through the `abs`; the toolbox rebinds its `reg_p`/`reg_n`
//!      to include it before building the CG operators).
//!    - `x_s ← max(x_s + dx_s·mask, 0)`; `res = ‖Δx‖/‖x‖` over both sources inside the mask
//!      eroded by a sphere (`imerode(mask, strel('sphere', erode_size))`, voxels outside the
//!      array counted as inside), with `Δx` the projected step. `erode_size =
//!      min(round(min(effective_mask_size)/10), 5)` voxels, `effective_mask_size` being the
//!      number of slices along each axis that hold any mask voxel (5 on a head); with
//!      `erode_size` 0 the whole padded grid counts. The iterate with the smallest `res` so far is
//!      kept.
//! 8. **Output**: that iterate, cropped back to the input grid: `x_para = x+`, `x_dia = x−` (≥ 0),
//!    `x_tot = x+ − x−`. QSM.rs returns `(χ+, χ− = −x−, χ+ + χ−)`.
//!
//! # Units
//!
//! The toolbox takes the local field in Hz; [`chi_sep_medi`] takes it in ppm and multiplies by
//! `cf/10⁶` (so the result depends on [`ChiSepMediParams::cf`] only through `D_r`'s default and the
//! relative weight of the two residuals). R2′ is in Hz, the QSM and the output in ppm.
//!
//! # Agreement with the toolbox
//!
//! Compared with `chi_sep_MEDI` itself (toolbox v1.1.3 in licensed MATLAB R2023b; QSM.rs in
//! double precision, as the toolbox runs):
//!
//! - **Phantoms** (`tests/chi_sep_medi_snu_parity.rs`, fixture from
//!   `tests/data/chi_sep_medi_snu_small.m`): 40×36×28 at 3 T with a CSF mask; 33×29×21 at 7 T,
//!   0.8×0.9×1.5 mm, oblique B0, with a CSF mask; negative R2′, `D_r` 120/150, λ = 2, no CSF
//!   mask. Same Gauss-Newton iteration counts (7, 6, 6), χ+ and |χ−| to relative L2
//!   ≤ 1.1·10⁻¹⁵, max |Δ| ≤ 8·10⁻¹⁷ ppm.
//! - **The verified QSM-CI `chi-sep-medi` container** (v4, identical to licensed MATLAB;
//!   QSMxT/QSM-CI#351): `option_data.qsm` = χ_total, `N_std` = `Fit_ppm_complex`'s `dp1` from the
//!   echoes' magnitudes, `mask_CSF = []`, `CF` = 42.5774 MHz/T·B0. QSM.rs's `N_std`
//!   ([`crate::utils::fit_ppm_complex_noise`]) equals the container's bit for bit in every case.
//!   - a 24×22×18 phantom whose voxels decay at different rates (`tests/chi_sep_medi_qsmci_parity.rs`,
//!     regenerated in Actions by `tests/data/chi_sep_medi_qsmci_small.py`): χ+ and |χ−| to relative
//!     L2 2.6·10⁻⁸, the container's float32 output;
//!   - in vivo (the EMTP travelling-head scan of QSM-CI's fidelity check, 3 T, 184×256×128,
//!     1.39 M mask voxels; its local field, R2′ and χ): 6 iterations each, the same `res` at every
//!     iteration (to the 4 digits the toolbox prints); χ+ and |χ−| to relative L2 2.5·10⁻⁸, max
//!     |Δ| 3·10⁻⁸ / 9·10⁻⁸ ppm, the float32 output (measured 2026-11-01, `examples/chi_sep_file.rs`);
//!   - QSM-CI's χ-separation phantom (`chisep-mc`, 7 T, 164×205×205, same inputs): 6 iterations
//!     each, `res` equal at the first; χ+ relative L2 1.7·10⁻³ (max |Δ| 8.3·10⁻³ ppm), |χ−|
//!     9.3·10⁻³ (7.9·10⁻³ ppm), r ≥ 0.9999. That residual is rounding, amplified: on this phantom
//!     every inner CG solve runs 42-50 iterations (5-10 in vivo), and the result moves as much under
//!     changes at the last bit. The toolbox itself, run with 4 instead of 8 threads, moves by
//!     3.7·10⁻³ / 1.13·10⁻² (more than QSM.rs differs from it), and QSM.rs moves by 2.1·10⁻³ /
//!     4.5·10⁻³ when `CF` changes by one ulp. QSM-CI's `chisep-medi` / `chisep-medi-qsmrs` parity
//!     pair checks this phantom.
//!
//!   Before QSM.rs#382 these cases ran with `N_std = 1/|mag|` (in vivo relative L2 8·10⁻¹³ against
//!   licensed MATLAB with that input), and the convergence test eroded the mask by 3 voxels
//!   instead of the toolbox's `erode_size` (5 on these scans; see step 7), which changed `res` by
//!   about 5 % but neither the iteration counts nor the results here.

//! # Performance
//!
//! Each CG iteration costs four 3-D FFTs (two dipole convolutions, complex) and two fused
//! gradient-adjoint passes; a Gauss-Newton iteration is two CG solves. In vivo (184×256×138
//! padded, 6 iterations) [`chi_sep_medi`] takes 106-133 s on 6 threads with a 2.7 GB peak (the
//! toolbox: 286 s on 8); on `chisep-mc` (164×205×215, CG solves of ~45 iterations) 282-292 s
//! (the toolbox: 931 s). Memory is about 45 `f64` per padded voxel.
//!
//! # Deviations
//!
//! - **An all-zero CSF mask means no CSF term.** The toolbox applies the CSF term whenever
//!   `option_data.mask_CSF` is not empty; with a mask that has no voxels, the mean over it is
//!   `NaN`, `NaN·λ_CSF` is `NaN` even for `λ_CSF = 0`, and every step is lost (the solver returns its
//!   initial guess). Its own GUI passes `[]` when it has no CSF mask, which turns the term off.
//!   The CSF mask is a required input here, as in `Chisep_script.m`; passing an all-zero mask
//!   reproduces the GUI's no-CSF behaviour (`mask_CSF = []`).
//! - **No wall-clock limit.** The toolbox also stops after `params.time_limit` (default 1800 s) of
//!   iterations. That makes its output depend on the machine; it is not reproduced.
//! - Options of `chi_sep_MEDI` that `Chisep_script.m` does not use are not exposed:
//!   `option_data.wG`/`wG_r2p` (precomputed edge masks), `mask_FastRelax`/`mask_SlowRelax`
//!   (voxels whose R2′ residual weight drops to 0.1; both default to empty), and the TKD
//!   initialisation it uses without `option_data.qsm` (`params.init_option`).
//!
//! # Reference pipeline
//!
//! What `Chisep_script.m` (toolbox v1.1.3) runs before `chi_sep_MEDI`, with its line numbers:
//!
//! - Tukey k-space windowing of the complex multi-echo data, factor 0.4 for Siemens/GE and 0 for
//!   Philips (l. 70-73); `mag` = root-sum-of-squares over echoes (l. 75).
//! - `N_std` from `Preprocessing4Phase(mag, phase)` (l. 76), which is MEDI's `Fit_ppm_complex`
//!   noise map (probed: equal to 3·10⁻¹⁸; QSM.rs [`crate::utils::fit_ppm_complex_noise`]).
//! - Mask: MEDI's `BET` on `mag` (l. 82); R2* by ARLO with more than three equally spaced echoes,
//!   NNLS otherwise (l. 87-93); `mask_CSF = extract_CSF(r2star, mask, voxel_size)` (MEDI's
//!   `extract_CSF.m`, l. 96; QSM.rs [`crate::inversion::extract_csf`]).
//! - ROMEO with phase-offset correction and weighted echo averaging, `W = TE·exp(−TE/40 ms)`
//!   (l. 99-123); V-SHARP `smvsize` 12 (l. 136-139); the local field in Hz (l. 139).
//! - QSM: STI Suite `QSM_iLSQR(..., 'padsize', [12 12 12])` (l. 145), passed as `option_data.qsm`
//!   (l. 158).
//! - R2′ is the user's (the script's input; its README offers a pseudo-R2 map otherwise).
//!
//! The toolbox's GUI (`WrapperChisepMEDI.p`, probed with a stand-in app object) differs: it
//! passes `mask_CSF = []` (no CSF term), the RSS of the Tukey-windowed magnitude, `λ = λ_CSF = 1`,
//! and `option_data.qsm` only when its "QSM initialisation" option is on. Its `N_std` is the
//! script's: `Fit_ppm_complex`'s `dp1` from the multi-echo magnitude, zero outside the mask
//! (QSM.rs#382; probed in licensed R2023b on phantoms whose voxels decay at different rates, where
//! the GUI's and `Preprocessing4Phase`'s `N_std` both equal `dp1` to 1.1·10⁻¹⁵). An earlier reading
//! of the GUI's as `c/|mag|` came from a phantom with uniform decay, where `dp1` is proportional to
//! `1/RSS`; with per-voxel decay `dp1·RSS` varies (0.72-0.87 there).
//!
//! Reference: Shin, H.-G., Lee, J., Yun, Y. H., Yoo, S. H., Jang, J., Oh, S.-H., Nam, Y., Jung,
//! S., Kim, S., Fukunaga, M., Kim, W., Choi, H. J., Lee, J. (2021). "χ-separation: Magnetic
//! susceptibility source separation toward iron and myelin mapping in the brain." NeuroImage,
//! 240:118371. <https://doi.org/10.1016/j.neuroimage.2021.118371>
//!
//! # Pipeline rules
//!
//! The facts above that a configuration can depart from are published as rules in
//! [`crate::pipeline::rules::RULES`] (each with its strength, basis, reason and source):
//! `chi-sep.mask-bet`, `chi-sep.unwrap-romeo`, `chi-sep.echo-combination`, `chi-sep.echo-weights`,
//! `chi-sep.bg-vsharp`, `chi-sep-medi.qsm-ilsqr`.

use crate::error::{check_len, InputError};
use crate::fft::Fft3dWorkspace;
use crate::inversion::medi::{cgsolve, dataterm_mask, fgrad, gradient_mask, par_fill, par_update, reg, CgBuffers};
use crate::Grid;
use num_complex::Complex64;

use super::ChiSepMaps;

/// The gyromagnetic ratio the toolbox's `chi_sep_MEDI` and `chi_sep_iLSQR` use to get the field
/// strength from `params.CF` (their `gyro_`), in Hz/T. It differs from
/// [`crate::constants::GAMMA_BAR_HZ_PER_T`] (42.576 MHz/T); kept so the field-dependent defaults
/// match the toolbox.
pub const TOOLBOX_GYRO_HZ_PER_T: f64 = 42.58e6;

/// The field strength, in tesla, at which the toolbox's default relaxometric constant is
/// [`super::DR_SHIN_2021`] (137 Hz/ppm): `chi_sep_MEDI` and `chi_sep_iLSQR` default to
/// `D_r = 137·(B0/2.9)` for both sources, `B0 = CF/42.58e6`, when `params.Dr_pos`/`Dr_neg` are not
/// given (probed: 141.7241379310345 at `CF` = 127.74 MHz, bit for bit). Shin et al. 2021 measured
/// 137 on a nominal 3 T (2.89 T) scanner; the toolbox scales it linearly with the field.
pub const TOOLBOX_DR_REFERENCE_TESLA: f64 = 2.9;

/// The toolbox's default `D_r` in Hz/ppm at central frequency `cf` (Hz): `137·((cf/42.58e6)/2.9)`,
/// in that order of operations.
pub fn toolbox_default_dr(cf: f64) -> f64 {
    super::DR_SHIN_2021 * ((cf / TOOLBOX_GYRO_HZ_PER_T) / TOOLBOX_DR_REFERENCE_TESLA)
}

/// Gauss-Newton iterations `chi_sep_MEDI` always runs (its `min_iter`, not a parameter), even
/// once the relative change is below [`ChiSepMediParams::tol`].
pub const CHI_SEP_MEDI_MIN_ITER: usize = 6;

/// `ε` in the L1 weights `1/sqrt(|wG·∇χ|² + ε)` (`chi_sep_MEDI_func`'s `eps__`).
const EPS_L1: f64 = 1e-5;
/// Weight of the R2′ residual against the field residual, both in Hz (`wei_mag` = 1/(2π)).
const WEI_MAG: f64 = 1.0 / (2.0 * std::f64::consts::PI);
/// MEDI `gradient_mask`'s `percentage` as the toolbox passes it: the number of gradient components
/// marked as edges, as a fraction of the mask's voxel count.
const EDGE_PERCENTAGE: f64 = 0.9;
/// The largest radius, in voxels, of `chi_sep_MEDI`'s convergence-mask erosion.
const ERODE_MAX: i64 = 5;

/// `chi_sep_MEDI`'s convergence mask (`Mask_erode`): the mask eroded by a sphere of radius
/// `erode_size = min(round(min(effective_mask_size)/10), 5)` voxels, where `effective_mask_size`
/// counts, along each axis, the slices that hold any mask voxel (probed in licensed R2023b: two
/// 10³ cubes far apart give [20 20 20]; voxel sizes and `pad_size` play no part; a head gives 5).
/// A radius of 0 (an axis with fewer than five such slices) makes it the whole padded grid, as in
/// the toolbox. `round` is MATLAB's, halves away from zero (15 slices give 2, 25 give 3), as
/// Rust's.
fn convergence_mask(mask: &[bool], dims: (usize, usize, usize)) -> Vec<bool> {
    let (nx, ny, nz) = dims;
    let mut any = [vec![false; nx], vec![false; ny], vec![false; nz]];
    for (p, _) in mask.iter().enumerate().filter(|(_, &m)| m) {
        any[0][p % nx] = true;
        any[1][(p / nx) % ny] = true;
        any[2][p / (nx * ny)] = true;
    }
    let extent = any.iter().map(|a| a.iter().filter(|&&v| v).count()).min().unwrap_or(0);
    let r = ((extent as f64 / 10.0).round() as i64).min(ERODE_MAX);
    if r == 0 {
        vec![true; mask.len()]
    } else {
        erode_sphere(mask, dims, r)
    }
}

/// Parameters for [`chi_sep_medi`]. The defaults are the toolbox's (see the module docs).
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct ChiSepMediParams {
    /// Central frequency in Hz (`params.CF`; default 123.2 MHz). The field strength the
    /// defaults of [`Self::dr_pos`] and [`Self::dr_neg`] scale with is `cf / 42.58e6`.
    pub cf: f64,
    /// Weight of the edge-masked L1 terms (`params.lambda`; default 1, as `Chisep_script.m` and
    /// the toolbox GUI set it).
    pub lambda: f64,
    /// Weight of the CSF term (`params.lambda_CSF`; default 1, as `Chisep_script.m` and the GUI
    /// set it). Used only where [`ChiSepMediInputs::mask_csf`] has voxels.
    pub lambda_csf: f64,
    /// Paramagnetic relaxometric constant in Hz/ppm (`params.Dr_pos`). `None` (the default):
    /// the toolbox's `137·(B0/2.9)` ([`toolbox_default_dr`]).
    pub dr_pos: Option<f64>,
    /// Diamagnetic relaxometric constant in Hz/ppm (`params.Dr_neg`). `None` (the default): the
    /// same as [`Self::dr_pos`]'s default.
    pub dr_neg: Option<f64>,
    /// Zero padding, in voxels, added at both ends of the third axis only (`params.pad_size`;
    /// default 5).
    pub pad: usize,
    /// Stop once the relative change of `[χ+ |χ−|]` in the eroded mask falls to this, after at
    /// least [`CHI_SEP_MEDI_MIN_ITER`] iterations (`params.tol_norm_ratio`; default 0.03).
    pub tol: f64,
    /// Gauss-Newton iterations at most (`params.max_iter`; default 30).
    pub max_iter: usize,
    /// Conjugate-gradient relative tolerance (`params.cg_tol`; default 0.05).
    pub cg_tol: f64,
    /// Conjugate-gradient iterations at most, per source and Gauss-Newton step
    /// (`params.cg_max_iter`; default 100).
    pub cg_max_iter: usize,
}

impl Default for ChiSepMediParams {
    fn default() -> Self {
        Self {
            cf: 123.2e6,
            lambda: 1.0,
            lambda_csf: 1.0,
            dr_pos: None,
            dr_neg: None,
            pad: 5,
            tol: 0.03,
            max_iter: 30,
            cg_tol: 0.05,
            cg_max_iter: 100,
        }
    }
}

impl ChiSepMediParams {
    /// The relaxometric constants used, `(D_r,pos, D_r,neg)` in Hz/ppm: the explicit values, or
    /// the toolbox's [`toolbox_default_dr`].
    pub fn dr(&self) -> (f64, f64) {
        let default = toolbox_default_dr(self.cf);
        (self.dr_pos.unwrap_or(default), self.dr_neg.unwrap_or(default))
    }

    /// Checks the parameters [`chi_sep_medi`] checks.
    pub fn validate(&self) -> Result<(), InputError> {
        let pos = |name: &str, v: f64| {
            if v.is_finite() && v > 0.0 {
                Ok(())
            } else {
                Err(InputError::param(name, v, "finite and > 0"))
            }
        };
        pos("χ-separation MEDI cf", self.cf)?;
        let (dp, dn) = self.dr();
        pos("χ-separation MEDI dr_pos", dp)?;
        pos("χ-separation MEDI dr_neg", dn)?;
        for (name, v) in [("χ-separation MEDI lambda", self.lambda), ("χ-separation MEDI lambda_csf", self.lambda_csf)] {
            if !(v.is_finite() && v >= 0.0) {
                return Err(InputError::param(name, v, "finite and ≥ 0"));
            }
        }
        for (name, v) in [("χ-separation MEDI tol", self.tol), ("χ-separation MEDI cg_tol", self.cg_tol)] {
            if !(v.is_finite() && v >= 0.0) {
                return Err(InputError::param(name, v, "finite and ≥ 0"));
            }
        }
        if self.max_iter == 0 {
            return Err(InputError::param("χ-separation MEDI max_iter", 0, "≥ 1"));
        }
        Ok(())
    }
}

/// The volumes [`chi_sep_medi`] takes, as `chi_sep_MEDI(mag, local_field_hz, r2prime, N_std,
/// mask, params, option_data)` takes them. All are on the grid, column-major.
#[derive(Clone, Copy)]
pub struct ChiSepMediInputs<'a> {
    /// Magnitude (`mag`; `Chisep_script.m` passes the root-sum-of-squares over echoes). Only its
    /// edges are used (MEDI's `gradient_mask`).
    pub magnitude: &'a [f64],
    /// Local field in ppm (the toolbox takes it in Hz; converted with [`ChiSepMediParams::cf`]).
    pub local_field: &'a [f64],
    /// R2′ in Hz (`r2prime`).
    pub r2prime: &'a [f64],
    /// Noise standard deviation of the field (`N_std`); the data weights are `mask/N_std`
    /// normalised to mean 1, so only its relative values matter. The toolbox (script and GUI)
    /// passes MEDI `Fit_ppm_complex`'s `dp1` from the echoes' magnitudes,
    /// [`crate::utils::fit_ppm_complex_noise`].
    pub n_std: &'a [f64],
    /// Conventional QSM in ppm (`option_data.qsm`), the initial guess.
    pub qsm: &'a [f64],
    /// Brain mask (`mask`), nonzero inside.
    pub mask: &'a [u8],
    /// CSF mask (`option_data.mask_CSF`), nonzero inside; required, as `Chisep_script.m` passes
    /// one. All zero: no CSF term, which reproduces the toolbox GUI (`mask_CSF = []`).
    pub mask_csf: &'a [u8],
}

/// The arrays of one run, on the padded grid.
struct Problem {
    dims: (usize, usize, usize),
    vs: [f64; 3],
    /// R2′ data weight `wei_mag·m` and field data weight `m` (`M_cat`'s two halves)
    w_mag: Vec<f64>,
    w_phs: Vec<f64>,
    /// R2′`.*mask` where R2′ > 0, and the local field in Hz (`target_`'s two halves)
    t_mag: Vec<f64>,
    t_phs: Vec<f64>,
    /// Dipole kernel, MEDI's `dipole_kernel(..., 'kspace')`
    d: Vec<f64>,
    /// MEDI edge mask of the magnitude, 1 = not an edge (`wG`; `wG_r2p` is the same)
    wg: [Vec<f64>; 3],
    csf: Vec<usize>,
    drp: f64,
    drn: f64,
    cf_mhz: f64,
    lambda: f64,
    lambda_csf: f64,
}

/// χ-separation by the SNU χ-separation toolbox's `chi_sep_MEDI` (see the module docs).
///
/// Takes the volumes of [`ChiSepMediInputs`] on `grid` and the B0 direction `bdir` (used as
/// given, as the toolbox uses `params.b0_dir`; pass a unit vector), and returns `(χ+, χ−,
/// χ_total)` in ppm, χ+ ≥ 0, χ− ≤ 0, zero outside the mask. `progress` receives
/// `(iteration, max_iter)` after each Gauss-Newton step.
///
/// # Errors
/// An [`InputError`] for parameters [`ChiSepMediParams::validate`] rejects and for a volume
/// whose length is not the grid's.
pub fn chi_sep_medi(
    inputs: &ChiSepMediInputs,
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &ChiSepMediParams,
    mut progress: impl FnMut(usize, usize),
) -> Result<ChiSepMaps, InputError> {
    params.validate()?;
    let n0 = grid.n_total();
    for (name, len) in [
        ("χ-separation MEDI magnitude", inputs.magnitude.len()),
        ("χ-separation MEDI local_field", inputs.local_field.len()),
        ("χ-separation MEDI r2prime", inputs.r2prime.len()),
        ("χ-separation MEDI n_std", inputs.n_std.len()),
        ("χ-separation MEDI qsm", inputs.qsm.len()),
        ("χ-separation MEDI mask", inputs.mask.len()),
        ("χ-separation MEDI mask_csf", inputs.mask_csf.len()),
    ] {
        check_len(name, len, n0)?;
    }
    let (nx, ny, nz0) = grid.dims;
    let pz = params.pad;
    let nz = nz0 + 2 * pz;
    let dims = (nx, ny, nz);
    let n = nx * ny * nz;
    let nxy = nx * ny;
    let vs = [grid.vsx(), grid.vsy(), grid.vsz()];
    // padarray(a, [0 0 pad_size], fill, 'both')
    let pad = |a: &dyn Fn(usize) -> f64, fill: f64| -> Vec<f64> {
        let mut out = vec![fill; n];
        out[pz * nxy..(pz + nz0) * nxy].iter_mut().enumerate().for_each(|(i, v)| *v = a(i));
        out
    };
    let mask0: Vec<bool> = inputs.mask.iter().map(|&m| m != 0).collect();
    // wG = gradient_mask(iMag, Mask, 0.9) on the input grid; wG_r2p is the same. Padded with 1.
    let wg0 = gradient_mask(inputs.magnitude, &mask0, grid, EDGE_PERCENTAGE);
    let wg = [0, 1, 2].map(|c| pad(&|i| wg0[c][i], 1.0));
    let maskf = pad(&|i| f64::from(u8::from(mask0[i])), 0.0);
    let mask: Vec<bool> = maskf.iter().map(|&m| m != 0.0).collect();
    let (drp, drn) = params.dr();
    let cf_mhz = params.cf / 1e6;
    let t_phs = pad(&|i| inputs.local_field[i] * cf_mhz, 0.0);
    // r2prime(r2prime == 0) = eps, after padding
    let mut r2p = pad(&|i| inputs.r2prime[i], 0.0);
    par_update(&mut r2p, |_, v| {
        if *v == 0.0 {
            *v = f64::EPSILON;
        }
    });
    let n_std = pad(&|i| inputs.n_std[i], 0.0);
    let qsm = pad(&|i| inputs.qsm[i], 0.0);
    let csf_mask: Vec<bool> = pad(&|i| f64::from(u8::from(inputs.mask_csf[i] != 0)), 0.0).iter().map(|&v| v != 0.0).collect();
    let csf: Vec<usize> = (0..n).filter(|&i| csf_mask[i]).collect();

    // m = dataterm_mask(1, N_std, Mask); M_cat = [wei_mag.*m.*wei_r2p, m]
    let mut m = vec![0.0; n];
    dataterm_mask(1, &n_std, &mask, &mut m);
    let mut w_mag = vec![0.0; n];
    par_fill(&mut w_mag, |i| WEI_MAG * m[i]);
    // target: R2' where it is positive (the initial guess below takes it as given)
    let mut t_mag = vec![0.0; n];
    par_fill(&mut t_mag, |i| if r2p[i] > 0.0 { r2p[i] * maskf[i] } else { 0.0 });
    let d = toolbox_dipole_kernel(dims, vs, bdir);

    // Initial guess from the voxelwise system Dr_pos·x+ + Dr_neg·x− = R2', x+ − x− = χ.
    let init = |sign: f64, dr_other: f64| -> Vec<f64> {
        let mut v = vec![0.0; n];
        par_fill(&mut v, |i| ((r2p[i] + sign * dr_other * qsm[i]) / (drp + drn)).max(0.0));
        v
    };
    let xp0 = init(1.0, drn);
    let xn0 = init(-1.0, drp);
    // The weights of the first iteration see the guess with the CSF zeroed; the iterate does not.
    let mut x_pos = xp0.clone();
    let mut x_neg = xn0.clone();
    for &i in &csf {
        x_pos[i] = 0.0;
        x_neg[i] = 0.0;
    }
    let mut it_pos = xp0;
    let mut it_neg = xn0;
    par_update(&mut it_pos, |i, v| *v *= maskf[i]);
    par_update(&mut it_neg, |i, v| *v *= maskf[i]);
    let eroded = convergence_mask(&mask, dims);

    let pb = Problem {
        dims, vs, w_mag, w_phs: m, t_mag, t_phs, d, wg, csf,
        drp, drn, cf_mhz, lambda: params.lambda, lambda_csf: params.lambda_csf,
    };
    let mut ws = Workspace::new(dims);

    let mut iter = 0;
    let mut res = f64::INFINITY;
    let mut res_min = f64::INFINITY;
    let (mut best_pos, mut best_neg) = (x_pos.clone(), x_neg.clone());
    let mut cg = CgBuffers::new(n);
    let mut bp = vec![0.0; n];
    let mut bn = vec![0.0; n];
    while (res > params.tol || iter < CHI_SEP_MEDI_MIN_ITER) && iter < params.max_iter {
        iter += 1;
        // Vr = 1./sqrt(abs(wG.*grad(x)).^2 + eps), for each source and for their difference
        let vr = |ws: &mut Workspace, x: &[f64]| -> [Vec<f64>; 3] {
            fgrad(x, dims, vs, &mut ws.g);
            let mut out = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
            for c in 0..3 {
                let (g, w) = (&ws.g[c], &pb.wg[c]);
                par_fill(&mut out[c], |i| {
                    let a = w[i] * g[i];
                    // wG.*(Vr.*(wG.*.)) with wG ∈ {0, 1}
                    w[i] / (a * a + EPS_L1).sqrt()
                });
            }
            out
        };
        let mut diff = vec![0.0; n];
        par_fill(&mut diff, |i| x_pos[i] - x_neg[i]);
        let w_pos = vr(&mut ws, &x_pos);
        let w_neg = vr(&mut ws, &x_neg);
        let w_tot = vr(&mut ws, &diff);
        // bp, bn: the gradients, at (x_pos, x_neg)
        pb.gradients(&mut ws, &x_pos, &x_neg, &w_pos, &w_neg, &w_tot, &mut bp, &mut bn);
        par_update(&mut bp, |_, v| *v = -*v);
        par_update(&mut bn, |_, v| *v = -*v);
        cgsolve(|dx, out| pb.apply(&mut ws, dx, true, &w_pos, &w_tot, out), &bp, params.cg_tol, params.cg_max_iter, &mut cg, |_| {});
        let dxp = cg.x.clone();
        cgsolve(|dx, out| pb.apply(&mut ws, dx, false, &w_neg, &w_tot, out), &bn, params.cg_tol, params.cg_max_iter, &mut cg, |_| {});
        let dxn = &cg.x;
        // x = max(x + dx.*Mask, 0); res = norm(dx_real.*Me)/norm(x.*Me) over both sources
        let mut num = 0.0;
        let mut den = 0.0;
        for (it, x, dx) in [(&mut it_pos, &mut x_pos, &dxp[..]), (&mut it_neg, &mut x_neg, &dxn[..])] {
            for i in 0..n {
                let new = (it[i] + dx[i] * maskf[i]).max(0.0);
                if eroded[i] {
                    let dr = new - it[i];
                    num += dr * dr;
                    den += new * new;
                }
                it[i] = new;
                x[i] = new;
            }
        }
        res = num.sqrt() / den.sqrt();
        if res < res_min {
            res_min = res;
            best_pos.copy_from_slice(&x_pos);
            best_neg.copy_from_slice(&x_neg);
        }
        progress(iter, params.max_iter);
    }

    // Crop the padding; χ− is returned signed.
    let crop = |a: &[f64], sign: f64| -> Vec<f64> {
        (0..n0).map(|i| if mask0[i] { sign * a[pz * nxy + i] } else { 0.0 }).collect()
    };
    let chi_pos = crop(&best_pos, 1.0);
    let chi_neg = crop(&best_neg, -1.0);
    let chi_total = chi_pos.iter().zip(&chi_neg).map(|(p, q)| p + q).collect();
    Ok((chi_pos, chi_neg, chi_total))
}

/// Buffers reused across operator applications.
struct Workspace {
    fft: Fft3dWorkspace,
    a: Vec<Complex64>,
    b: Vec<Complex64>,
    g: [Vec<f64>; 3],
    r: Vec<f64>,
    s: Vec<f64>,
}

impl Workspace {
    fn new((nx, ny, nz): (usize, usize, usize)) -> Self {
        let n = nx * ny * nz;
        let z = Complex64::new(0.0, 0.0);
        Self {
            fft: Fft3dWorkspace::new(nx, ny, nz),
            a: vec![z; n],
            b: vec![z; n],
            g: [vec![0.0; n], vec![0.0; n], vec![0.0; n]],
            r: vec![0.0; n],
            s: vec![0.0; n],
        }
    }
}

impl Problem {
    /// `Dconv_p` in place: `buf ← (CF/1e6)·ifftn(D.*fftn(buf))`
    fn dconv(&self, fft: &mut Fft3dWorkspace, buf: &mut [Complex64]) {
        fft.fft3d(buf);
        par_update(buf, |i, v| *v *= self.d[i]);
        fft.ifft3d(buf);
        let s = self.cf_mhz;
        par_update(buf, |_, v| *v *= s);
    }

    /// `λ_CSF·LT(LT(|x|))` added to `out`, with `LT(v) = Mask_CSF.*(v − mean(v(Mask_CSF)))`
    fn add_csf(&self, x: &[f64], out: &mut [f64]) {
        if self.csf.is_empty() {
            return;
        }
        let k = self.csf.len() as f64;
        let m1 = self.csf.iter().map(|&i| x[i].abs()).sum::<f64>() / k;
        let lt: Vec<f64> = self.csf.iter().map(|&i| x[i].abs() - m1).collect();
        let m2 = lt.iter().sum::<f64>() / k;
        for (&i, &v) in self.csf.iter().zip(&lt) {
            out[i] += self.lambda_csf * (v - m2);
        }
    }

    /// The regulariser of one source: `λ·∇ᵀ(W_single ∇x) + 2λ·∇ᵀ(W_tot ∇(x − y)) + CSF(x)` with
    /// `x − y` given as `diff`, added to `out`.
    #[allow(clippy::too_many_arguments)]
    fn add_reg(&self, ws: &mut Workspace, x: &[f64], diff: &[f64], w_single: &[Vec<f64>; 3], w_tot: &[Vec<f64>; 3], out: &mut [f64]) {
        let lam = self.lambda;
        reg(x, w_single, self.dims, self.vs, &mut ws.r);
        reg(diff, w_tot, self.dims, self.vs, &mut ws.s);
        let (r, s) = (&ws.r, &ws.s);
        par_update(out, |i, v| *v += lam * r[i] + 2.0 * (lam * s[i]));
        self.add_csf(x, out);
    }

    /// The CG operator of one source (`Ap` or `An`): data term plus regulariser at `dx` with the
    /// other source's step zero.
    fn apply(&self, ws: &mut Workspace, dx: &[f64], pos: bool, w_single: &[Vec<f64>; 3], w_tot: &[Vec<f64>; 3], out: &mut [f64]) {
        let dr = if pos { self.drp } else { self.drn };
        // phase: Dconv_p(m.*(m.*Dconv_p(dx)))
        par_fill(&mut ws.a, |i| Complex64::new(dx[i], 0.0));
        self.dconv(&mut ws.fft, &mut ws.a);
        let w = &self.w_phs;
        par_update(&mut ws.a, |i, v| *v = (*v * w[i]) * w[i]);
        self.dconv(&mut ws.fft, &mut ws.a);
        let (a, wm) = (&ws.a, &self.w_mag);
        par_fill(out, |i| 2.0 * (dr * (wm[i] * (wm[i] * (dr * dx[i])))) + 2.0 * a[i].re);
        // reg_p(dx, zero_mat) / reg_n(zero_mat, dx): the difference term sees dx − 0
        self.add_reg(ws, dx, dx, w_single, w_tot, out);
    }

    /// `bp = fidelity_derivative_p + reg_p` and `bn` likewise, at `(xp, xn)`.
    #[allow(clippy::too_many_arguments)]
    fn gradients(
        &self,
        ws: &mut Workspace,
        xp: &[f64],
        xn: &[f64],
        w_pos: &[Vec<f64>; 3],
        w_neg: &[Vec<f64>; 3],
        w_tot: &[Vec<f64>; 3],
        bp: &mut [f64],
        bn: &mut [f64],
    ) {
        let n = xp.len();
        // field residual: Dconv_p(xp) + (−Dconv_p(xn)) − f, weighted twice
        par_fill(&mut ws.a, |i| Complex64::new(xp[i], 0.0));
        self.dconv(&mut ws.fft, &mut ws.a);
        par_fill(&mut ws.b, |i| Complex64::new(xn[i], 0.0));
        self.dconv(&mut ws.fft, &mut ws.b);
        let (b, t, w) = (&ws.b, &self.t_phs, &self.w_phs);
        par_update(&mut ws.a, |i, v| *v = ((*v + (-b[i])) - t[i]) * w[i] * w[i]);
        self.dconv(&mut ws.fft, &mut ws.a);
        // R2' residual: Dr_pos·xp + Dr_neg·xn − r2p.*mask, weighted twice
        let mut rm = vec![0.0; n];
        let (wm, tm) = (&self.w_mag, &self.t_mag);
        par_fill(&mut rm, |i| wm[i] * (wm[i] * ((self.drp * xp[i] + self.drn * xn[i]) - tm[i])));
        let a = &ws.a;
        par_fill(bp, |i| 2.0 * (self.drp * rm[i]) + 2.0 * a[i].re);
        par_fill(bn, |i| 2.0 * (self.drn * rm[i]) - 2.0 * a[i].re);
        let mut diff = vec![0.0; n];
        par_fill(&mut diff, |i| xp[i] - xn[i]);
        self.add_reg(ws, xp, &diff, w_pos, w_tot, bp);
        par_fill(&mut diff, |i| xn[i] - xp[i]);
        self.add_reg(ws, xn, &diff, w_neg, w_tot, bn);
    }
}

/// MEDI's `dipole_kernel(matrix_size, voxel_size, B0_dir, 'kspace')`, as the toolbox builds it:
/// the k-grid `(-N/2 : N/2-1)/(N·vs)` per axis (half-integer steps on an odd axis), `D = 1/3 −
/// (k·B0)²/|k|²`, `NaN` (k = 0) set to 0, then `fftshift`ed (circular shift by `floor(N/2)`).
pub(super) fn toolbox_dipole_kernel((nx, ny, nz): (usize, usize, usize), vs: [f64; 3], b0: (f64, f64, f64)) -> Vec<f64> {
    let coord = |i: usize, n: usize, v: f64| (-(n as f64) / 2.0 + i as f64) / (n as f64 * v);
    let mut d = vec![0.0; nx * ny * nz];
    for k in 0..nz {
        let z = coord(k, nz, vs[2]);
        let kk = (k + nz / 2) % nz;
        for j in 0..ny {
            let y = coord(j, ny, vs[1]);
            let jj = (j + ny / 2) % ny;
            for i in 0..nx {
                let x = coord(i, nx, vs[0]);
                let ii = (i + nx / 2) % nx;
                let s = x * b0.0 + y * b0.1 + z * b0.2;
                let den = x * x + y * y + z * z;
                let v = 1.0 / 3.0 - (s * s) / den;
                d[ii + nx * (jj + ny * kk)] = if v.is_nan() { 0.0 } else { v };
            }
        }
    }
    d
}

/// `imerode(mask, strel('sphere', r))`: a voxel stays if every voxel of the ball of radius `r`
/// voxels around it is in the mask, counting voxels outside the array as in the mask (MATLAB's
/// border rule for erosion).
fn erode_sphere(mask: &[bool], (nx, ny, nz): (usize, usize, usize), r: i64) -> Vec<bool> {
    let mut offs = Vec::new();
    for dz in -r..=r {
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy + dz * dz <= r * r {
                    offs.push((dx, dy, dz));
                }
            }
        }
    }
    let mut out = vec![false; mask.len()];
    par_fill(&mut out, |p| {
        if !mask[p] {
            return false;
        }
        let (i, j, k) = ((p % nx) as i64, ((p / nx) % ny) as i64, (p / (nx * ny)) as i64);
        offs.iter().all(|&(dx, dy, dz)| {
            let (a, b, c) = (i + dx, j + dy, k + dz);
            if a < 0 || b < 0 || c < 0 || a >= nx as i64 || b >= ny as i64 || c >= nz as i64 {
                return true;
            }
            mask[a as usize + nx * (b as usize + ny * c as usize)]
        })
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small separable problem: a χ+ block and a χ− block in an ellipsoid, with R2′ and field
    /// consistent with them (the field from QSM.rs's own kernel; only shapes matter here).
    /// (field ppm, R2′ Hz, χ ppm, magnitude, N_std, mask)
    type Problem6 = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<u8>);

    fn problem(n: (usize, usize, usize)) -> Problem6 {
        let (nx, ny, nz) = n;
        let nt = nx * ny * nz;
        let mut chi = vec![0.0; nt];
        let mut mask = vec![0u8; nt];
        let mut r2p = vec![0.0; nt];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let l = i + nx * (j + ny * k);
                    let (x, y, z) = (i as f64 - nx as f64 / 2.0, j as f64 - ny as f64 / 2.0, k as f64 - nz as f64 / 2.0);
                    if (x / (0.4 * nx as f64)).powi(2) + (y / (0.4 * ny as f64)).powi(2) + (z / (0.4 * nz as f64)).powi(2) <= 1.0 {
                        mask[l] = 1;
                        let (p, q) = (if x > 1.0 { 0.05 } else { 0.0 }, if x < -1.0 { 0.03 } else { 0.0 });
                        chi[l] = p - q;
                        r2p[l] = 137.0 * (p + q) + 2.0;
                    }
                }
            }
        }
        let grid = Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);
        let d = crate::kernels::dipole::dipole_kernel(&grid, (0.0, 0.0, 1.0));
        let mut buf: Vec<Complex64> = chi.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        crate::fft::fft3d(&mut buf, nx, ny, nz);
        buf.iter_mut().zip(&d).for_each(|(b, &dv)| *b *= dv);
        crate::fft::ifft3d(&mut buf, nx, ny, nz);
        let field: Vec<f64> = buf.iter().zip(&mask).map(|(b, &m)| b.re * m as f64).collect();
        let mag: Vec<f64> = (0..nt).map(|l| 100.0 + (l % 7) as f64).collect();
        let n_std: Vec<f64> = mask.iter().map(|&m| 0.01 * m as f64).collect();
        (field, r2p, chi, mag, n_std, mask)
    }

    fn run(
        n: (usize, usize, usize),
        csf: bool,
        params: &ChiSepMediParams,
    ) -> (Result<ChiSepMaps, InputError>, usize, Vec<u8>) {
        let (field, r2p, chi, mag, n_std, mask) = problem(n);
        let mut mask_csf = vec![0u8; mask.len()];
        if csf {
            let c = n.0 / 2 + n.0 * (n.1 / 2 + n.1 * (n.2 / 2));
            mask_csf[c] = 1;
            mask_csf[c + 1] = 1;
        }
        let inputs = ChiSepMediInputs {
            magnitude: &mag, local_field: &field, r2prime: &r2p, n_std: &n_std, qsm: &chi, mask: &mask, mask_csf: &mask_csf,
        };
        let grid = Grid::new(n.0, n.1, n.2, 1.0, 1.0, 1.0);
        let mut iters = 0;
        let r = chi_sep_medi(&inputs, &grid, (0.0, 0.0, 1.0), params, |i, _| iters = i);
        (r, iters, mask)
    }

    #[test]
    fn defaults_are_the_toolbox_values() {
        let p = ChiSepMediParams::default();
        assert_eq!((p.lambda, p.lambda_csf, p.pad, p.tol, p.max_iter, p.cg_tol, p.cg_max_iter), (1.0, 1.0, 5, 0.03, 30, 0.05, 100));
        assert_eq!((p.dr_pos, p.dr_neg), (None, None));
        let (dp, dn) = ChiSepMediParams { cf: 127.74e6, ..p.clone() }.dr();
        assert_eq!((dp, dn), (141.7241379310345, 141.7241379310345));
        let (dp, dn) = ChiSepMediParams { dr_pos: Some(120.0), dr_neg: Some(150.0), ..p }.dr();
        assert_eq!((dp, dn), (120.0, 150.0));
    }

    #[test]
    fn param_errors_name_the_parameter() {
        let bad = [
            ChiSepMediParams { cf: 0.0, ..Default::default() },
            ChiSepMediParams { dr_pos: Some(-1.0), ..Default::default() },
            ChiSepMediParams { dr_neg: Some(f64::NAN), ..Default::default() },
            ChiSepMediParams { lambda: -1.0, ..Default::default() },
            ChiSepMediParams { lambda_csf: f64::INFINITY, ..Default::default() },
            ChiSepMediParams { tol: -0.1, ..Default::default() },
            ChiSepMediParams { cg_tol: f64::NAN, ..Default::default() },
            ChiSepMediParams { max_iter: 0, ..Default::default() },
        ];
        let names = ["cf", "dr_pos", "dr_neg", "lambda", "lambda_csf", "tol", "cg_tol", "max_iter"];
        for (p, name) in bad.iter().zip(names) {
            let e = p.validate().unwrap_err().to_string();
            assert!(e.starts_with(&format!("χ-separation MEDI {name} = ")), "{e}");
        }
        assert!(ChiSepMediParams::default().validate().is_ok());
    }

    #[test]
    fn wrong_buffer_length_is_an_input_error() {
        let n = (8, 8, 6);
        let nt = n.0 * n.1 * n.2;
        let v = vec![0.0; nt];
        let short = vec![0.0; nt - 1];
        let m = vec![1u8; nt];
        let grid = Grid::new(n.0, n.1, n.2, 1.0, 1.0, 1.0);
        let inputs = ChiSepMediInputs { magnitude: &v, local_field: &v, r2prime: &short, n_std: &v, qsm: &v, mask: &m, mask_csf: &m };
        let e = chi_sep_medi(&inputs, &grid, (0.0, 0.0, 1.0), &ChiSepMediParams::default(), |_, _| {}).unwrap_err();
        assert_eq!(e.to_string(), format!("χ-separation MEDI r2prime has {} values, expected {nt}", nt - 1));
    }

    #[test]
    fn sources_have_their_signs_and_runs_at_least_six_iterations() {
        let (r, iters, mask) = run((16, 14, 10), true, &ChiSepMediParams::default());
        let (p, q, t) = r.unwrap();
        assert!(iters >= CHI_SEP_MEDI_MIN_ITER, "{iters}");
        for l in 0..p.len() {
            assert!(p[l] >= 0.0 && q[l] <= 0.0 && t[l] == p[l] + q[l]);
            if mask[l] == 0 {
                assert!(p[l] == 0.0 && q[l] == 0.0);
            }
        }
        assert!(p.iter().any(|&v| v > 0.01) && q.iter().any(|&v| v < -0.01));
        // max_iter below the minimum caps it
        let (_, iters, _) = run((16, 14, 10), true, &ChiSepMediParams { max_iter: 2, ..Default::default() });
        assert_eq!(iters, 2);
    }

    /// The toolbox turns every value NaN with a CSF mask that has no voxels; here it means no CSF
    /// term, and the result is the same as with a mask that never touches a voxel's update.
    /// `nnz(Mask_erode)` as `chi_sep_MEDI` made it (licensed R2023b, read from its workspace) for
    /// box masks on a 64×60×56 grid (66 slices after its padding): the erosion radius follows the
    /// smallest number of mask slices along an axis, rounded to tens.
    #[test]
    fn convergence_mask_matches_the_toolbox() {
        let dims = (64, 60, 66);
        let boxes = |boxes: &[([usize; 3], [usize; 3])]| -> Vec<bool> {
            let mut m = vec![false; dims.0 * dims.1 * dims.2];
            for &(o, n) in boxes {
                for k in o[2]..o[2] + n[2] {
                    for j in o[1]..o[1] + n[1] {
                        for i in o[0]..o[0] + n[0] {
                            m[i + dims.0 * (j + dims.1 * (k + 5))] = true;
                        }
                    }
                }
            }
            m
        };
        let centred = |n: [usize; 3]| boxes(&[([(64 - n[0]) / 2, (60 - n[1]) / 2, (56 - n[2]) / 2], n)]);
        let count = |m: &[bool]| convergence_mask(m, dims).iter().filter(|&&v| v).count();
        for (n, want) in [([10, 10, 10], 512), ([16, 16, 16], 1728), ([24, 24, 24], 8000), ([30, 30, 30], 13824),
                          ([48, 48, 48], 54872), ([40, 20, 10], 5472), ([15, 30, 30], 7436), ([25, 30, 30], 10944),
                          ([4, 30, 30], 64 * 60 * 66), ([56, 56, 50], 84640)] {
            assert_eq!(count(&centred(n)), want, "box {n:?}");
        }
        // Two 10³ cubes far apart count 20 slices per axis: radius 2.
        assert_eq!(count(&boxes(&[([4, 4, 4], [10; 3]), ([44, 39, 39], [10; 3])])), 432);
    }

    #[test]
    fn empty_csf_mask_means_no_csf_term() {
        let (r, _, _) = run((16, 14, 10), false, &ChiSepMediParams::default());
        let (p, q, _) = r.unwrap();
        assert!(p.iter().chain(&q).all(|v| v.is_finite()));
        let (r0, _, _) = run((16, 14, 10), false, &ChiSepMediParams { lambda_csf: 0.0, ..Default::default() });
        assert_eq!(r0.unwrap().0, p);
    }

    /// On an even grid MEDI's k-space kernel is the usual unshifted kernel; on an odd axis its
    /// grid misses k = 0 by half a bin.
    #[test]
    fn toolbox_kernel_is_the_fft_kernel_on_even_grids_and_half_a_bin_off_on_odd() {
        let vs = [0.8, 1.0, 1.5];
        let h = (0.1, -0.2, 0.97);
        let d = toolbox_dipole_kernel((8, 6, 4), vs, h);
        let grid = Grid::new(8, 6, 4, vs[0], vs[1], vs[2]);
        let norm = (h.0 * h.0 + h.1 * h.1 + h.2 * h.2).sqrt();
        let r = crate::kernels::dipole::dipole_kernel(&grid, (h.0 / norm, h.1 / norm, h.2 / norm));
        let rn = toolbox_dipole_kernel((8, 6, 4), vs, (h.0 / norm, h.1 / norm, h.2 / norm));
        assert!(rn.iter().zip(&r).all(|(a, b)| (a - b).abs() < 1e-12));
        assert_eq!(d[0], 0.0);
        let odd = toolbox_dipole_kernel((5, 6, 4), [1.0; 3], (0.0, 0.0, 1.0));
        // no exact zero frequency on the odd axis: no entry is the k = 0 fill value except by
        // coincidence, and index 0 holds k_x = -0.5 + 2 = ... the fftshift of -2.5..1.5
        assert!(odd[0] != 0.0);
    }

    #[test]
    fn erosion_counts_voxels_outside_the_array_as_inside() {
        let dims = (9, 9, 9);
        let full = vec![true; 729];
        assert!(erode_sphere(&full, dims, 3).iter().all(|&v| v));
        let mut holed = full.clone();
        holed[4 + 9 * (4 + 9 * 4)] = false;
        let e = erode_sphere(&holed, dims, 3);
        // every voxel within distance 3 of the hole is removed, the rest stay
        for k in 0..9i64 {
            for j in 0..9i64 {
                for i in 0..9i64 {
                    let d2 = (i - 4).pow(2) + (j - 4).pow(2) + (k - 4).pow(2);
                    assert_eq!(e[(i + 9 * (j + 9 * k)) as usize], d2 > 9, "({i},{j},{k})");
                }
            }
        }
    }
}

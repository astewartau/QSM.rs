//! χ-separation (iLSQR): the SNU χ-separation toolbox's `chi_sep_iLSQR`
//!
//! [`chi_sep_ilsqr`] implements **`chi_sep_iLSQR` of the χ-separation toolbox v1.1.3** (SNU-LIST,
//! Jongho Lee's lab; `functions/corefunc/chi_sep_iLSQR.p` and its `chi_sep_iLSQR_step1_func.p`,
//! `step2_func.p`, `step3_func.p`), as its script `Chisep_script.m` calls it (l. 167-172):
//! `chi_sep_iLSQR(mag, local_field_hz, r2prime, mask, params, option_data)` with `params.b0_dir`,
//! `params.CF`, `params.voxel_size`, `option_data.qsm` and `option_data.N_std`. The toolbox ships as
//! protected P-code; with the authors' permission the algorithm was recovered by black-box probing
//! in licensed MATLAB R2023b, as for [`super::chi_sep_medi`]: the profiler (which lists the bodies
//! of the P-code's anonymous functions verbatim), logging wrappers on the functions it calls that
//! save the caller's whole workspace (`evalin('caller', 'save ...')`), replaying the function
//! handles captured in those workspaces, and step-by-step comparison of each step function's
//! inputs and outputs. A MATLAB re-implementation built from that reproduces `chi_sep_iLSQR` to
//! rounding with the same iteration counts (see *Agreement with the toolbox*).
//!
//! Earlier QSM.rs versions ported Shin et al. 2021's Supplementary Methods instead: an
//! edge-masked-TV Gauss-Newton solve with an SNR weight from the magnitude. The toolbox runs a
//! different algorithm, an iLSQR-style three-step solve (Li et al. 2015) on the two sources; the
//! old port's output differed from the toolbox's by relative L2 0.51 (χ+) / 0.54 (|χ−|) in vivo
//! and 1.06 / 1.30 on QSM-CI's `chisep-mc` phantom (the cases below).
//!
//! # The algorithm
//!
//! The unknowns are `x+ = χ+` and `x− = |χ−|` in ppm. Everything is converted to ppm of the
//! central frequency first: the local field `f`, R2′ as `r = max(R2′/CF·10⁶, 0)` (negative R2′ is
//! clamped, after the conversion) and `d± = D_r,±/CF·10⁶`. With `Dc(x) = real(ifftn(D.*fftn(x)))`:
//!
//! 1. **Defaults** (`chi_sep_iLSQR`): `B0 = CF/42.58·10⁶`; `D_r,pos = D_r,neg = 137·(B0/2.9)`
//!    ([`toolbox_default_dr`]); `pad_size` [16 16 16]; `cg_max_iter_LSQR` 50, `cg_tol_LSQR` 0.02,
//!    `cg_max_iter_SA` 50, `cg_tol_SA` 0.02, `D2_thr` 0.11, `prc_W_I_min`/`max` 60/99.8,
//!    `WG_prc_min`/`max` 50/70 (see [`ChiSepIlsqrParams`]); not parameters: the NNLS tolerance
//!    0.01 and `iter_max` 10 ([`NNLS_TOL`], [`NNLS_ITER_MAX`]).
//! 2. **Padding**: every volume is zero-padded by `pad_size` voxels at both ends of every axis;
//!    the steps below run on that grid and crop their output back.
//! 3. **Dipole kernel**: MEDI's `dipole_kernel(N, vs, B0_dir, 'kspace')`, the same as
//!    `chi_sep_MEDI`'s, including its half-bin offset on odd axes.
//! 4. **Step 1, LSQR** (`chi_sep_iLSQR_step1_func`). Weights: `W_I = clamp((hi − L)/(hi − lo),
//!    0, 1)` with `L = |del2(f, vs(1), vs(2), vs(3))|` and `lo`, `hi` its `prctile`s 60 and 99.8
//!    inside the mask. (`del2` takes its first spacing argument for the second array axis, so on a
//!    grid with `vs(1) ≠ vs(2)` the first two spacings are swapped; kept.) `w_mag = W_I·m`,
//!    `w_phs = 2π·W_I·m` (`m` the mask). With `Ar(x, d) = Dc(w_mag.*(d.*x))`, `ArH(y, d) =
//!    d.*(w_mag.*Dc(y))` and `Af = AfH = Dc(w_phs.*Dc(·))`, MATLAB's `pcg` solves
//!    `[ArH(Ar(x+,d+) + Ar(x−,d−), d+) + AfH(Af(x+ − x−)); ArH(…, d−) − AfH(Af(x+ − x−))] =
//!    [ArH(Dc(w_mag.*r), d+) + AfH(Dc(w_phs.*f)); ArH(Dc(w_mag.*r), d−) − AfH(Dc(w_phs.*f))]` for
//!    the concatenated `[x+ x−]` (tolerance `cg_tol_LSQR`, at most `cg_max_iter_LSQR` iterations).
//!    The start is the voxelwise solution from the QSM `χ`: `x+ = max((r + d−·χ)/(d+ + d−), 0)`,
//!    `x− = max((r − d+·χ)/(d+ + d−), 0)`, masked. (Without a QSM the toolbox uses `χ = 0`.)
//! 5. **Step 2, streaking artefacts** (`chi_sep_iLSQR_step2_func`), per source:
//!    - a fast QSM `χ_fs` of the field: `W = clamp((|D|^0.001 − a)/(b − a), 0, 1)` with `a`, `b`
//!      the `prctile`s 1 and 30 of `|D|^0.001` over the whole grid; `K` the k-space smoothing by a
//!      2.5 mm partial-volume ball (`pv_ball` below); `k1 = ifftshift(fftn(f).*sign(D))`, then
//!      `χ2 = real(ifftn(fftshift(W.*k1 + (1 − W).*K(k1))))`. That multiplies the weight, laid out
//!      for an unshifted spectrum, with the *shifted* spectrum (a quirk of the toolbox, kept).
//!      Then `k = fftn(χ2.*m)`, `χ3 = real(ifftn(W.*k + (1 − W).*K(k))).*m`, and `χ_fs = c₁ + c₂·χ3`
//!      with `(c₁, c₂)` the least-squares fit of `χ3` to the TKD solution (threshold 1/8, masked)
//!      over the whole grid;
//!    - R2′ smoothed in k-space by a Tukey window (`tukeywin(N, 0.5)` per axis, applied as
//!      `fftshift(ifftshift(fftn(r)).*w)`);
//!    - gradient weights `W_G± = clamp((hi − |g±|)/(hi − lo), 0, 1)·m` per axis, `lo`/`hi` the
//!      `prctile`s 50/70 of `|g±|` inside the mask, `g+ = (∇r_s + d−·∇χ_fs)/(d+ + d−)`,
//!      `g− = (∇r_s − d+·∇χ_fs)/(d+ + d−)` (∇: MEDI's `fgrad` with the voxel size, on the masked
//!      volumes);
//!    - the streaking component, by MATLAB's `pcg` on a **complex k-space unknown** `y` restricted
//!      to the cone `M_ic = |D| < D2_thr`: `M_ic.*fftn(∇ᵀ(W_G²·∇ real(ifftn(M_ic.*y)))) =
//!      M_ic.*fftn(∇ᵀ(W_G²·∇x))` (∇ here with unit spacing, ∇ᵀ its exact adjoint; tolerance
//!      `cg_tol_SA`, at most `cg_max_iter_SA` iterations, zero start), and the output is
//!      `(x − real(ifftn(M_ic.*y)).*m).*m`.
//! 6. **Step 3, NNLS** (`chi_sep_iLSQR_step3_func`): repeat step 1 with one `pcg` iteration
//!    starting from the current iterate (not clamped), then step 2, then project both sources onto
//!    `x ≥ 0`. `res = (‖Δx+‖ + ‖Δx−‖)/(‖x+‖ + ‖x−‖)` of the projected iterates; the loop runs while
//!    `res > 0.01` and at most `iter_max + 1` = 11 times (its counter starts at 1 and is tested
//!    with `≤ iter_max + 1`). The output is the iterate with the smallest `res`.
//! 7. **Output**: `x_para = x+`, `x_dia = x−` (≥ 0), `x_tot = x+ − x−`, zero outside the mask.
//!    QSM.rs returns `(χ+, χ− = −x−, χ+ + χ−)`.
//!
//! **MATLAB `pcg`.** The solver is R2023b's `pcg.m` without a preconditioner, line for line: the
//! residual recomputed when the recurrence says converged or stagnates (up to `min(n/50, 5,
//! n − maxit)` extra steps), and, when it does not converge, the iterate with the smallest
//! residual returned instead of the last one (so a one-iteration solve that does not lower the
//! residual returns its start). For the complex unknown of step 2 it uses the conjugate inner
//! product and compares `p'·q ≤ 0` by its real part, as MATLAB does.
//!
//! # Inputs
//!
//! `Chisep_script.m` passes the magnitude, the local field in Hz, R2′, the mask and
//! `option_data.qsm` (STI Suite `QSM_iLSQR`) and `option_data.N_std`. **`chi_sep_iLSQR` never
//! uses `mag` or `N_std`**: neither reaches its step functions, and the re-implementation that
//! ignores them matches the toolbox (and its output is unchanged when they change, see the
//! fixture generator). [`ChiSepIlsqrInputs`] therefore takes the local field, R2′, the QSM and the
//! mask. The QSM is required, as the script passes it. The toolbox's GUI (`WrapperChisepiLSQR.p`,
//! probed with a stand-in app object) passes **no QSM** (`option_data` holds only `N_std` and
//! zero `mask_Unrel_Mag`/`mask_Unrel_Phs`), so its runs start step 1 from zero; QSM.rs does not
//! offer that path.
//!
//! # Units
//!
//! The toolbox takes the local field in Hz and converts it to ppm; [`chi_sep_ilsqr`] takes it in
//! ppm directly. R2′ is in Hz, the QSM and the output in ppm. [`ChiSepIlsqrParams::cf`] sets the
//! ppm scale of R2′ and `D_r` and the default `D_r`.
//!
//! # Agreement with the toolbox
//!
//! Compared with `chi_sep_iLSQR` itself (toolbox v1.1.3 in licensed MATLAB R2023b; QSM.rs in
//! double precision, as the toolbox runs):
//!
//! - **Phantoms** (`tests/chi_sep_ilsqr_snu_parity.rs`, fixture from
//!   `tests/data/chi_sep_ilsqr_snu_small.m`): 40×36×28 at 3 T; 33×29×21 at 7 T, 0.8×0.9×1.5 mm,
//!   oblique B0; negative R2′ with `D_r` 120/150. Same NNLS iteration counts (2, 4, 4); χ+ and
//!   |χ−| to relative L2 ≤ 4·10⁻¹⁰ (3·10⁻¹⁴ on the second). The toolbox itself moves by up to
//!   1.5·10⁻⁹ on these phantoms between 8 threads and 1.
//! - **In vivo** (the EMTP travelling-head scan of QSM-CI's fidelity check, 3 T, 184×256×128;
//!   its local field, R2′ and χ, 1.39 M mask voxels): 4 NNLS iterations each; χ+ relative L2
//!   3.1·10⁻¹¹ (max |Δ| 2.5·10⁻¹⁰ ppm), |χ−| 3.5·10⁻¹¹ (3.2·10⁻¹⁰ ppm).
//! - **QSM-CI's χ-separation phantom** (`chisep-mc`, 7 T, 164×205×205, same inputs):
//!   4 NNLS iterations each; χ+ relative L2 2.5·10⁻¹² (max |Δ| 8.8·10⁻¹² ppm), |χ−| 3.2·10⁻¹²
//!   (7.2·10⁻¹² ppm). The toolbox run with 4 instead of 8 threads moves by 1.1·10⁻¹² / 1.5·10⁻¹²
//!   here and 2.6·10⁻¹² / 2.9·10⁻¹² in vivo.
//!
//! The residual is rounding inside the iterations. QSM.rs evaluates the same expressions with its
//! own FFT and, where that saves work, in a different order (step 1's operator applies `Dc∘Dc`
//! once to the sum of the two sources' R2′ terms instead of to each; the fast QSM's fit uses
//! centred sums instead of a QR solve), so every operation rounds differently, where the
//! toolbox's thread count changes only its reductions; in vivo that puts QSM.rs at about ten
//! times the toolbox's thread-count spread. The solve does not amplify noise in its input: a
//! ±10⁻¹⁵ relative perturbation of the field moves QSM.rs's in-vivo result by 4·10⁻¹⁵.
//!
//! # Performance
//!
//! Each step-1 `pcg` iteration costs six dipole convolutions (12 complex 3-D FFTs) on the padded
//! grid; each step-2 iteration two FFTs; an NNLS iteration is one step-1 iteration (three operator
//! applications) and two step-2 solves. The fixed parts (weights, fast QSM, smoothed R2′) are
//! computed once, where the toolbox recomputes them in every NNLS iteration. In vivo
//! (216×288×160 padded, 4 NNLS iterations) [`chi_sep_ilsqr`] takes 280 s on 6 threads with a
//! 3.5 GB peak (the toolbox: 433 s on 8 threads); on `chisep-mc` (196×237×237) 398 s (the toolbox:
//! 1113 s on 8, 743 s on 4). Memory is about 45 `f64` per padded voxel.
//!
//! # Deviations
//!
//! None in the algorithm. Options of `chi_sep_iLSQR` that `Chisep_script.m` does not use are not
//! exposed: `option_data.mask_Unrel_Mag` / `mask_Unrel_Phs` (masks of unreliable magnitude and
//! phase voxels whose data weights drop; both default to zero), and `params.flag_strong_x_correction`
//! / `strong_x_thresh` (off by default). The start without a QSM (the GUI's) is not offered.
//!
//! # Reference pipeline
//!
//! What `Chisep_script.m` (toolbox v1.1.3) runs before `chi_sep_iLSQR` is what it runs before
//! `chi_sep_MEDI` (see [`super::chi_sep_medi`]'s *Reference pipeline*): Tukey windowing, MEDI's
//! `BET`, ROMEO with weighted echo averaging (`W = TE·exp(−TE/40 ms)`), V-SHARP `smvsize` 12, the
//! local field in Hz (l. 139), and the QSM from STI Suite `QSM_iLSQR(..., 'padsize', [12 12 12])`
//! (l. 145, [`chisep_toolbox_ilsqr`]) passed as `option_data.qsm` (l. 170). R2′ is the user's.
//! `N_std` (l. 76) is passed (l. 171) but not used.
//!
//! Reference: Shin, H.-G., Lee, J., Yun, Y. H., Yoo, S. H., Jang, J., Oh, S.-H., Nam, Y., Jung,
//! S., Kim, S., Fukunaga, M., Kim, W., Choi, H. J., Lee, J. (2021). "χ-separation: Magnetic
//! susceptibility source separation toward iron and myelin mapping in the brain." NeuroImage,
//! 240:118371. <https://doi.org/10.1016/j.neuroimage.2021.118371>. The three steps follow Li, W.,
//! Wang, N., Yu, F., et al. (2015). "A method for estimating and removing streaking artifacts in
//! quantitative susceptibility mapping." NeuroImage, 108:111-122.
//!
//! # Pipeline rules
//!
//! The facts above that a configuration can depart from are published as rules in
//! [`crate::pipeline::rules::RULES`] (each with its strength, basis, reason and source):
//! `chi-sep.mask-bet`, `chi-sep.unwrap-romeo`, `chi-sep.echo-combination`, `chi-sep.echo-weights`,
//! `chi-sep.bg-vsharp`, `chi-sep-ilsqr.qsm-ilsqr`.

use crate::error::{check_len, InputError};
use crate::fft::Fft3dWorkspace;
use crate::inversion::medi::{fgrad, par_fill, par_update};
use crate::inversion::{IlsqrParams, STI_DEFAULT_NITER, STI_GUI_PAD_MM};
use crate::Grid;
use num_complex::Complex64;

use super::chi_sep_medi::{toolbox_default_dr, toolbox_dipole_kernel};
use super::ChiSepMaps;

/// NNLS stopping tolerance of `chi_sep_iLSQR_step3_func` (its `tol_norm_ratio`, not a parameter).
pub const NNLS_TOL: f64 = 0.01;
/// `iter_max` of `chi_sep_iLSQR_step3_func` (not a parameter). The NNLS loop runs at most
/// `NNLS_ITER_MAX + 1` = 11 times: its counter starts at 1 and is tested with `≤ iter_max + 1`.
pub const NNLS_ITER_MAX: usize = 10;
/// Radius, in mm, of the ball the fast QSM smooths k-space with.
const BALL_RADIUS_MM: f64 = 2.5;
/// TKD threshold of the fast QSM's scaling reference.
const TKD_THRESHOLD: f64 = 0.125;
/// Tukey window ratio of the R2′ smoothing.
const TUKEY_RATIO: f64 = 0.5;

/// The iLSQR call of the χ-separation toolbox (`Chisep_script.m` l. 142-145) for the QSM that
/// [`chi_sep_ilsqr`] and [`super::chi_sep_medi`] start from: [`STI_DEFAULT_NITER`] iterations
/// (`QSM_iLSQR`'s own default, as the toolbox calls it without `'niter'`),
/// [`STI_GUI_PAD_MM`] padding, STI's tolerances.
pub fn chisep_toolbox_ilsqr() -> IlsqrParams {
    IlsqrParams { max_iter: STI_DEFAULT_NITER, pad_mm: STI_GUI_PAD_MM, ..IlsqrParams::default() }
}

/// Parameters for [`chi_sep_ilsqr`]. The defaults are the toolbox's (see the module docs); the
/// field names after the toolbox's `params` fields are given with each.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct ChiSepIlsqrParams {
    /// Central frequency in Hz (`params.CF`; default 123.2 MHz). The field strength the
    /// defaults of [`Self::dr_pos`] and [`Self::dr_neg`] scale with is `cf / 42.58e6`.
    pub cf: f64,
    /// Paramagnetic relaxometric constant in Hz/ppm (`params.Dr_pos`). `None` (the default):
    /// the toolbox's `137·(B0/2.9)` ([`toolbox_default_dr`]).
    pub dr_pos: Option<f64>,
    /// Diamagnetic relaxometric constant in Hz/ppm (`params.Dr_neg`). `None` (the default): the
    /// same as [`Self::dr_pos`]'s default.
    pub dr_neg: Option<f64>,
    /// Zero padding, in voxels, added at both ends of each axis (`params.pad_size`; default
    /// [16, 16, 16]).
    pub pad: [usize; 3],
    /// Step 1's `pcg` iterations at most (`params.cg_max_iter_LSQR`; default 50). The NNLS
    /// iterations run one.
    pub cg_max_iter_lsqr: usize,
    /// Step 1's `pcg` relative tolerance (`params.cg_tol_LSQR`; default 0.02).
    pub cg_tol_lsqr: f64,
    /// Step 2's `pcg` iterations at most, per source (`params.cg_max_iter_SA`; default 50).
    pub cg_max_iter_sa: usize,
    /// Step 2's `pcg` relative tolerance (`params.cg_tol_SA`; default 0.02).
    pub cg_tol_sa: f64,
    /// The streaking cone is `|D| < d2_thr` (`params.D2_thr`; default 0.11).
    pub d2_thr: f64,
    /// Percentiles (0-100) of the field's `|del2|` inside the mask at which step 1's weight is 1
    /// and 0 (`params.prc_W_I_min`, `prc_W_I_max`; default 60, 99.8).
    pub prc_w_i: (f64, f64),
    /// Percentiles (0-100) of each gradient component's magnitude inside the mask at which step
    /// 2's weights are 1 and 0 (`params.WG_prc_min`, `WG_prc_max`; default 50, 70).
    pub prc_w_g: (f64, f64),
    /// iLSQR settings for the conventional QSM (`option_data.qsm`) that χ-separation is
    /// initialised with, for hosts that reconstruct it (the pipeline takes the QSM as an input).
    /// Default [`chisep_toolbox_ilsqr`]: 100 iterations, 12 mm padding, as `Chisep_script.m`
    /// (l. 142-145) calls STI Suite's `QSM_iLSQR`: `'padsize', [12 12 12]` and no `'niter'`, so
    /// the function's default of 100. This differs from [`IlsqrParams::default`] (the STI Suite
    /// GUI's 40 iterations).
    pub qsm_ilsqr: IlsqrParams,
}

impl Default for ChiSepIlsqrParams {
    fn default() -> Self {
        Self {
            cf: 123.2e6,
            dr_pos: None,
            dr_neg: None,
            pad: [16; 3],
            cg_max_iter_lsqr: 50,
            cg_tol_lsqr: 0.02,
            cg_max_iter_sa: 50,
            cg_tol_sa: 0.02,
            d2_thr: 0.11,
            prc_w_i: (60.0, 99.8),
            prc_w_g: (50.0, 70.0),
            qsm_ilsqr: chisep_toolbox_ilsqr(),
        }
    }
}

impl ChiSepIlsqrParams {
    /// The relaxometric constants used, `(D_r,pos, D_r,neg)` in Hz/ppm: the explicit values, or
    /// the toolbox's [`toolbox_default_dr`].
    pub fn dr(&self) -> (f64, f64) {
        let default = toolbox_default_dr(self.cf);
        (self.dr_pos.unwrap_or(default), self.dr_neg.unwrap_or(default))
    }

    /// Checks the parameters [`chi_sep_ilsqr`] checks.
    pub fn validate(&self) -> Result<(), InputError> {
        let pos = |name: &str, v: f64| {
            if v.is_finite() && v > 0.0 {
                Ok(())
            } else {
                Err(InputError::param(name, v, "finite and > 0"))
            }
        };
        pos("χ-separation iLSQR cf", self.cf)?;
        let (dp, dn) = self.dr();
        pos("χ-separation iLSQR dr_pos", dp)?;
        pos("χ-separation iLSQR dr_neg", dn)?;
        pos("χ-separation iLSQR d2_thr", self.d2_thr)?;
        for (name, v) in [("χ-separation iLSQR cg_tol_lsqr", self.cg_tol_lsqr), ("χ-separation iLSQR cg_tol_sa", self.cg_tol_sa)] {
            if !(v.is_finite() && v > 0.0 && v < 1.0) {
                return Err(InputError::param(name, v, "in (0, 1)"));
            }
        }
        for (name, (lo, hi)) in [("χ-separation iLSQR prc_w_i", self.prc_w_i), ("χ-separation iLSQR prc_w_g", self.prc_w_g)] {
            if !(lo.is_finite() && hi.is_finite() && (0.0..=100.0).contains(&lo) && (0.0..=100.0).contains(&hi) && lo < hi) {
                return Err(InputError::param(name, format!("({lo}, {hi})"), "0 ≤ min < max ≤ 100"));
            }
        }
        Ok(())
    }
}

/// The volumes [`chi_sep_ilsqr`] takes. `chi_sep_iLSQR(mag, local_field_hz, r2prime, mask, params,
/// option_data)` also takes the magnitude and `option_data.N_std`, but never uses them (see the
/// module docs). All are on the grid, column-major.
#[derive(Clone, Copy)]
pub struct ChiSepIlsqrInputs<'a> {
    /// Local field in ppm (the toolbox takes it in Hz and converts it to ppm).
    pub local_field: &'a [f64],
    /// R2′ in Hz (`r2prime`); negative values are clamped at 0, as the toolbox does.
    pub r2prime: &'a [f64],
    /// Conventional QSM in ppm (`option_data.qsm`), step 1's start; required, as
    /// `Chisep_script.m` passes it.
    pub qsm: &'a [f64],
    /// Brain mask (`mask`), nonzero inside.
    pub mask: &'a [u8],
}

/// χ-separation by the SNU χ-separation toolbox's `chi_sep_iLSQR` (see the module docs).
///
/// Takes the volumes of [`ChiSepIlsqrInputs`] on `grid` and the B0 direction `bdir` (used as
/// given, as the toolbox uses `params.b0_dir`; pass a unit vector), and returns `(χ+, χ−,
/// χ_total)` in ppm, χ+ ≥ 0, χ− ≤ 0, zero outside the mask. `progress` receives
/// `(nnls_iteration, NNLS_ITER_MAX + 1)` after each NNLS iteration.
///
/// # Errors
/// An [`InputError`] for parameters [`ChiSepIlsqrParams::validate`] rejects, for a volume whose
/// length is not the grid's, and for an empty mask.
pub fn chi_sep_ilsqr(
    inputs: &ChiSepIlsqrInputs,
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &ChiSepIlsqrParams,
    mut progress: impl FnMut(usize, usize),
) -> Result<ChiSepMaps, InputError> {
    params.validate()?;
    let n0 = grid.n_total();
    for (name, len) in [
        ("χ-separation iLSQR local_field", inputs.local_field.len()),
        ("χ-separation iLSQR r2prime", inputs.r2prime.len()),
        ("χ-separation iLSQR qsm", inputs.qsm.len()),
        ("χ-separation iLSQR mask", inputs.mask.len()),
    ] {
        check_len(name, len, n0)?;
    }
    if !inputs.mask.iter().any(|&m| m != 0) {
        return Err(InputError("χ-separation iLSQR mask is empty".into()));
    }
    let pb = Problem::new(inputs, grid, bdir, params);
    let n = pb.n;

    // Step 1 from the QSM, then step 2
    let (drp, drn) = (pb.drp, pb.drn);
    let mut x0 = vec![0.0; 2 * n];
    {
        let q = pb.pad_input(inputs.qsm);
        let (a, b) = x0.split_at_mut(n);
        par_fill(a, |i| ((pb.r[i] + drn * q[i]) / (drp + drn)).max(0.0) * pb.m[i]);
        par_fill(b, |i| ((pb.r[i] - drp * q[i]) / (drp + drn)).max(0.0) * pb.m[i]);
    }
    let mut ws = Workspace::new(pb.dims);
    let x = pb.step1(&mut ws, x0, params.cg_tol_lsqr, params.cg_max_iter_lsqr);
    let (mut xp, mut xn) = pb.step2(&mut ws, x, params);

    // Step 3: NNLS
    let max_it = NNLS_ITER_MAX + 1;
    let mut res = f64::INFINITY;
    let mut res_min = f64::INFINITY;
    let mut best = (xp.clone(), xn.clone());
    let mut it = 1;
    while res > NNLS_TOL && it <= max_it {
        let mut x0 = vec![0.0; 2 * n];
        {
            let (a, b) = x0.split_at_mut(n);
            par_fill(a, |i| xp[i] * pb.m[i]);
            par_fill(b, |i| xn[i] * pb.m[i]);
        }
        let x = pb.step1(&mut ws, x0, params.cg_tol_lsqr, 1);
        let (mut a, mut b) = pb.step2(&mut ws, x, params);
        // a.*(a > 0)
        par_update(&mut a, |_, v| *v *= f64::from(u8::from(*v > 0.0)));
        par_update(&mut b, |_, v| *v *= f64::from(u8::from(*v > 0.0)));
        let nrm = |v: &[f64]| det_sum(v.len(), |i| v[i] * v[i]).sqrt();
        let dnrm = |u: &[f64], v: &[f64]| det_sum(u.len(), |i| (u[i] - v[i]) * (u[i] - v[i])).sqrt();
        res = (dnrm(&a, &xp) + dnrm(&b, &xn)) / (nrm(&a) + nrm(&b));
        xp = a;
        xn = b;
        if res < res_min {
            res_min = res;
            best.0.copy_from_slice(&xp);
            best.1.copy_from_slice(&xn);
        }
        progress(it, max_it);
        it += 1;
    }

    let chi_pos = pb.crop(&best.0, 1.0);
    let chi_neg = pb.crop(&best.1, -1.0);
    let chi_total = chi_pos.iter().zip(&chi_neg).map(|(p, q)| p + q).collect();
    Ok((chi_pos, chi_neg, chi_total))
}

/// The arrays of one run, on the padded grid.
struct Problem {
    dims: (usize, usize, usize),
    n: usize,
    /// input grid and padding
    dims0: (usize, usize, usize),
    pad: [usize; 3],
    /// mask (0/1)
    m: Vec<f64>,
    /// R2′ in ppm, clamped at 0
    r: Vec<f64>,
    /// `D_r,±` in ppm
    drp: f64,
    drn: f64,
    /// dipole kernel (`dipole_kernel(..., 'kspace')`)
    d: Vec<f64>,
    /// step 1's data weights
    w_mag: Vec<f64>,
    w_phs: Vec<f64>,
    /// step 1's right-hand side, `[b+; b−]`
    b1: Vec<f64>,
    /// step 2's squared gradient weights `W_G±²` per axis
    wg2: [[Vec<f64>; 3]; 2],
    /// step 2's cone `|D| < D2_thr`
    cone: Vec<bool>,
}

/// FFT buffers reused across operator applications.
struct Workspace {
    fft: Fft3dWorkspace,
    c: Vec<Complex64>,
    g: [Vec<f64>; 3],
    t: Vec<f64>,
    u: Vec<f64>,
}

impl Workspace {
    fn new((nx, ny, nz): (usize, usize, usize)) -> Self {
        let n = nx * ny * nz;
        Self {
            fft: Fft3dWorkspace::new(nx, ny, nz),
            c: vec![Complex64::new(0.0, 0.0); n],
            g: [vec![0.0; n], vec![0.0; n], vec![0.0; n]],
            t: vec![0.0; n],
            u: vec![0.0; n],
        }
    }
}

impl Problem {
    fn new(inputs: &ChiSepIlsqrInputs, grid: &Grid, bdir: (f64, f64, f64), params: &ChiSepIlsqrParams) -> Self {
        let dims0 = grid.dims;
        let pad = params.pad;
        let dims = (dims0.0 + 2 * pad[0], dims0.1 + 2 * pad[1], dims0.2 + 2 * pad[2]);
        let n = dims.0 * dims.1 * dims.2;
        let vs = [grid.vsx(), grid.vsy(), grid.vsz()];
        let mut pb = Self {
            dims,
            n,
            dims0,
            pad,
            m: Vec::new(),
            r: Vec::new(),
            drp: 0.0,
            drn: 0.0,
            d: toolbox_dipole_kernel(dims, vs, bdir),
            w_mag: Vec::new(),
            w_phs: Vec::new(),
            b1: Vec::new(),
            wg2: Default::default(),
            cone: Vec::new(),
        };
        let cf = params.cf;
        let mask: Vec<f64> = inputs.mask.iter().map(|&v| f64::from(u8::from(v != 0))).collect();
        pb.m = pb.pad_input(&mask);
        let mut r = pb.pad_input(inputs.r2prime);
        par_update(&mut r, |_, v| *v = (*v / cf * 1e6).max(0.0));
        pb.r = r;
        let f = pb.pad_input(inputs.local_field);
        let (dp, dn) = params.dr();
        pb.drp = dp / cf * 1e6;
        pb.drn = dn / cf * 1e6;
        let inside: Vec<usize> = (0..n).filter(|&i| pb.m[i] != 0.0).collect();
        let mut ws = Workspace::new(dims);

        // Step 1: weights and right-hand side
        let lap = del2(&f, dims, [vs[1], vs[0], vs[2]]);
        let w_i = ramp_weight(&lap.iter().map(|v| v.abs()).collect::<Vec<_>>(), &inside, params.prc_w_i);
        pb.w_mag = (0..n).map(|i| w_i[i] * pb.m[i]).collect();
        pb.w_phs = (0..n).map(|i| 2.0 * std::f64::consts::PI * w_i[i] * pb.m[i]).collect();
        let mut yr = vec![0.0; n];
        let wr: Vec<f64> = (0..n).map(|i| pb.w_mag[i] * pb.r[i]).collect();
        pb.dc(&mut ws, &wr, &mut yr);
        let mut t = vec![0.0; n];
        pb.dc(&mut ws, &yr, &mut t); // Dc(yr), shared by ArH(yr, d±)
        let mut yf = vec![0.0; n];
        let wf: Vec<f64> = (0..n).map(|i| pb.w_phs[i] * f[i]).collect();
        pb.dc(&mut ws, &wf, &mut yf);
        let mut af = vec![0.0; n];
        pb.af(&mut ws, &yf, &mut af);
        let mut b1 = vec![0.0; 2 * n];
        {
            let (bp, bn) = b1.split_at_mut(n);
            let (w, t, af) = (&pb.w_mag, &t, &af);
            par_fill(bp, |i| dp_mul(pb.drp, w[i] * t[i]) + af[i]);
            par_fill(bn, |i| dp_mul(pb.drn, w[i] * t[i]) - af[i]);
        }
        pb.b1 = b1;

        // Step 2: fast QSM, smoothed R2′, gradient weights, cone
        let x_fs = pb.fast_qsm(&mut ws, &f, vs);
        let r_lpf = pb.tukey_lpf(&mut ws);
        let xm: Vec<f64> = (0..n).map(|i| x_fs[i] * pb.m[i]).collect();
        let rm: Vec<f64> = (0..n).map(|i| r_lpf[i] * pb.m[i]).collect();
        let mut gx = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
        let mut gr = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
        fgrad(&xm, dims, vs, &mut gx);
        fgrad(&rm, dims, vs, &mut gr);
        let (drp, drn) = (pb.drp, pb.drn);
        for c in 0..3 {
            let gp: Vec<f64> = (0..n).map(|i| ((gr[c][i] + drn * gx[c][i]) / (drp + drn)).abs()).collect();
            let gn: Vec<f64> = (0..n).map(|i| ((gr[c][i] - drp * gx[c][i]) / (drp + drn)).abs()).collect();
            for (s, g) in [gp, gn].into_iter().enumerate() {
                let w = ramp_weight(&g, &inside, params.prc_w_g);
                // conj(W).*W with W = W_G.*M
                let wm: Vec<f64> = (0..n).map(|i| w[i] * pb.m[i]).collect();
                pb.wg2[s][c] = wm.iter().map(|v| v * v).collect();
            }
        }
        pb.cone = pb.d.iter().map(|v| v.abs() < params.d2_thr).collect();
        pb
    }

    /// `padarray(a, pad, 0, 'both')` of an input-grid volume
    fn pad_input(&self, a: &[f64]) -> Vec<f64> {
        let (nx, ny, nz) = self.dims;
        let (mx, my, _) = self.dims0;
        let [px, py, pz] = self.pad;
        let mut out = vec![0.0; nx * ny * nz];
        for k in 0..self.dims0.2 {
            for j in 0..my {
                let src = mx * (j + my * k);
                let dst = px + nx * ((j + py) + ny * (k + pz));
                out[dst..dst + mx].copy_from_slice(&a[src..src + mx]);
            }
        }
        out
    }

    /// The input-grid part of a padded volume, times `sign`
    fn crop(&self, a: &[f64], sign: f64) -> Vec<f64> {
        let (nx, ny, _) = self.dims;
        let (mx, my, mz) = self.dims0;
        let [px, py, pz] = self.pad;
        let mut out = Vec::with_capacity(mx * my * mz);
        for k in 0..mz {
            for j in 0..my {
                let src = px + nx * ((j + py) + ny * (k + pz));
                out.extend(a[src..src + mx].iter().map(|v| sign * v));
            }
        }
        out
    }

    /// Zero the padding ring (the toolbox crops each step's output and pads the next step's input)
    fn zero_ring(&self, a: &mut [f64]) {
        let (nx, ny, _) = self.dims;
        let (mx, my, mz) = self.dims0;
        let [px, py, pz] = self.pad;
        par_update(a, |l, v| {
            let (i, j, k) = (l % nx, (l / nx) % ny, l / (nx * ny));
            if i < px || i >= px + mx || j < py || j >= py + my || k < pz || k >= pz + mz {
                *v = 0.0;
            }
        });
    }

    /// `Dc(x) = real(ifftn(D.*fftn(x)))` into `out`
    fn dc(&self, ws: &mut Workspace, x: &[f64], out: &mut [f64]) {
        par_fill(&mut ws.c, |i| Complex64::new(x[i], 0.0));
        ws.fft.fft3d(&mut ws.c);
        let d = &self.d;
        par_update(&mut ws.c, |i, v| *v *= d[i]);
        ws.fft.ifft3d(&mut ws.c);
        let c = &ws.c;
        par_fill(out, |i| c[i].re);
    }

    /// `Af(x) = Dc(w_phs.*Dc(x))` into `out`
    fn af(&self, ws: &mut Workspace, x: &[f64], out: &mut [f64]) {
        let mut t = std::mem::take(&mut ws.t);
        self.dc(ws, x, &mut t);
        let w = &self.w_phs;
        par_update(&mut t, |i, v| *v *= w[i]);
        self.dc(ws, &t, out);
        ws.t = t;
    }

    /// Step 1's normal operator on `[x+; x−]`
    fn a1(&self, ws: &mut Workspace, x: &[f64], out: &mut [f64]) {
        let n = self.n;
        let (xp, xn) = x.split_at(n);
        let (drp, drn) = (self.drp, self.drn);
        // R2′ part: ArH(Ar(x+, d+) + Ar(x−, d−), d±) = d±.*(w_mag.*Dc(Dc(w_mag.*(d+ x+ + d− x−))))
        let mut s = std::mem::take(&mut ws.u);
        let w = &self.w_mag;
        par_fill(&mut s, |i| w[i] * (drp * xp[i]) + w[i] * (drn * xn[i]));
        let mut t1 = vec![0.0; n];
        self.dc(ws, &s, &mut t1);
        self.dc(ws, &t1, &mut s);
        // field part: Af(Af(x+ − x−))
        par_fill(&mut t1, |i| xp[i] - xn[i]);
        let mut v = vec![0.0; n];
        self.af(ws, &t1, &mut v);
        self.af(ws, &v, &mut t1);
        let (op, on) = out.split_at_mut(n);
        let (s_, t_) = (&s, &t1);
        par_fill(op, |i| dp_mul(drp, w[i] * s_[i]) + t_[i]);
        par_fill(on, |i| dp_mul(drn, w[i] * s_[i]) - t_[i]);
        ws.u = s;
    }

    /// Step 1: `pcg` from `x0 = [x+; x−]`; returns the solution with the padding ring zeroed.
    fn step1(&self, ws: &mut Workspace, x0: Vec<f64>, tol: f64, maxit: usize) -> Vec<f64> {
        let o = pcg(|v, o| self.a1(ws, v, o), &self.b1, tol, maxit, Some(x0));
        let mut x = o.x;
        let (a, b) = x.split_at_mut(self.n);
        self.zero_ring(a);
        self.zero_ring(b);
        x
    }

    /// `real(ifftn(M_ic.*y))` into `out`
    fn cone_ifft(&self, ws: &mut Workspace, y: &[Complex64], out: &mut [f64]) {
        let cone = &self.cone;
        par_fill(&mut ws.c, |i| if cone[i] { y[i] } else { Complex64::new(0.0, 0.0) });
        ws.fft.ifft3d(&mut ws.c);
        let c = &ws.c;
        par_fill(out, |i| c[i].re);
    }

    /// `M_ic.*fftn(gT(W2.*fgrad(x, 1), 1))` into `out`
    fn cone_fft_reg(&self, ws: &mut Workspace, x: &[f64], w2: &[Vec<f64>; 3], out: &mut [Complex64]) {
        fgrad(x, self.dims, [1.0; 3], &mut ws.g);
        for c in 0..3 {
            let w = &w2[c];
            par_update(&mut ws.g[c], |i, v| *v *= w[i]);
        }
        let mut t = std::mem::take(&mut ws.t);
        grad_adjoint_unit(&ws.g, self.dims, &mut t);
        par_fill(&mut ws.c, |i| Complex64::new(t[i], 0.0));
        ws.t = t;
        ws.fft.fft3d(&mut ws.c);
        let (cone, c) = (&self.cone, &ws.c);
        par_fill(out, |i| if cone[i] { c[i] } else { Complex64::new(0.0, 0.0) });
    }

    /// Step 2 on step 1's `[x+; x−]`: each source minus its streaking component, masked.
    fn step2(&self, ws: &mut Workspace, x: Vec<f64>, params: &ChiSepIlsqrParams) -> (Vec<f64>, Vec<f64>) {
        let n = self.n;
        let mut out = [vec![0.0; n], vec![0.0; n]];
        for (s, o) in out.iter_mut().enumerate() {
            let xs = &x[s * n..(s + 1) * n];
            let w2 = &self.wg2[s];
            let mut b = vec![Complex64::new(0.0, 0.0); n];
            self.cone_fft_reg(ws, xs, w2, &mut b);
            let mut t = vec![0.0; n];
            let y = pcg(
                |v: &[Complex64], o: &mut [Complex64]| {
                    self.cone_ifft(ws, v, &mut t);
                    self.cone_fft_reg(ws, &t, w2, o);
                },
                &b,
                params.cg_tol_sa,
                params.cg_max_iter_sa,
                None,
            );
            let y = y.x;
            let mut sa = vec![0.0; n];
            self.cone_ifft(ws, &y, &mut sa);
            let m = &self.m;
            par_fill(o, |i| (xs[i] - sa[i] * m[i]) * m[i]);
        }
        let [p, q] = out;
        (p, q)
    }

    /// The fast QSM of step 2 (see the module docs), on the padded grid.
    fn fast_qsm(&self, ws: &mut Workspace, f: &[f64], vs: [f64; 3]) -> Vec<f64> {
        let n = self.n;
        let dims = self.dims;
        let d = &self.d;
        // k-space blending weight from |D|^0.001
        let d1: Vec<f64> = d.iter().map(|v| v.abs().powf(0.001)).collect();
        let a = prctile(&mut d1.clone(), 1.0);
        let b = prctile(&mut d1.clone(), 30.0);
        let w: Vec<f64> = d1.iter().map(|&v| clamp01((v - a) / (b - a))).collect();
        drop(d1);
        // TKD reference: fftn(f)./Dt, |Dt| ≥ 1/8 with sign(0) = 1
        let mut fk: Vec<Complex64> = f.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        ws.fft.fft3d(&mut fk);
        let mut t: Vec<Complex64> = (0..n).map(|i| {
            let s = if d[i] < 0.0 { -1.0 } else { 1.0 };
            let dt = if d[i].abs() < TKD_THRESHOLD { TKD_THRESHOLD * s } else { d[i] };
            fk[i] / dt
        }).collect();
        ws.fft.ifft3d(&mut t);
        let x_tkd: Vec<f64> = (0..n).map(|i| t[i].re * self.m[i]).collect();
        // the ball's k-space smoothing: lpf(z) = ifftn(fftn(z).*fftn(fftshift(S)))
        let ball = pv_ball(dims, BALL_RADIUS_MM, vs);
        let mut kk: Vec<Complex64> = circshift(&ball, dims, true).into_iter().map(|v| Complex64::new(v, 0.0)).collect();
        ws.fft.fft3d(&mut kk);
        let lpf = |ws: &mut Workspace, x: &[Complex64]| -> Vec<Complex64> {
            let mut y = x.to_vec();
            ws.fft.fft3d(&mut y);
            par_update(&mut y, |i, v| *v *= kk[i]);
            ws.fft.ifft3d(&mut y);
            y
        };
        // k1c = ifftshift(fftn(f).*sign(D)); k2 = fftshift(W.*k1c + (1 − W).*lpf(k1c))
        let sk: Vec<Complex64> = (0..n).map(|i| fk[i] * sign0(d[i])).collect();
        let k1c = circshift(&sk, dims, false);
        drop(sk);
        let l1 = lpf(ws, &k1c);
        let blend: Vec<Complex64> = (0..n).map(|i| k1c[i] * w[i] + l1[i] * (1.0 - w[i])).collect();
        drop((k1c, l1));
        let mut k2 = circshift(&blend, dims, true);
        drop(blend);
        ws.fft.ifft3d(&mut k2);
        // km = fftn(x2.*m); x3 = real(ifftn(W.*km + (1 − W).*lpf(km))).*m
        let mut km: Vec<Complex64> = (0..n).map(|i| Complex64::new(k2[i].re * self.m[i], 0.0)).collect();
        drop(k2);
        ws.fft.fft3d(&mut km);
        let l2 = lpf(ws, &km);
        let mut x3c: Vec<Complex64> = (0..n).map(|i| km[i] * w[i] + l2[i] * (1.0 - w[i])).collect();
        drop((km, l2));
        ws.fft.ifft3d(&mut x3c);
        let x3: Vec<f64> = (0..n).map(|i| x3c[i].re * self.m[i]).collect();
        // [ones x3] \ x_tkd over the whole grid
        let nf = n as f64;
        let mx = det_sum(n, |i| x3[i]) / nf;
        let my = det_sum(n, |i| x_tkd[i]) / nf;
        let sxy = det_sum(n, |i| (x3[i] - mx) * (x_tkd[i] - my));
        let sxx = det_sum(n, |i| (x3[i] - mx) * (x3[i] - mx));
        let c2 = sxy / sxx;
        let c1 = my - c2 * mx;
        x3.iter().map(|&v| c2 * v + c1).collect()
    }

    /// R2′ (ppm) smoothed by a Tukey window in k-space:
    /// `real(ifftn(fftshift(ifftshift(fftn(r)).*w)))`
    fn tukey_lpf(&self, ws: &mut Workspace) -> Vec<f64> {
        let (nx, ny, nz) = self.dims;
        let (wx, wy, wz) = (tukeywin(nx, TUKEY_RATIO), tukeywin(ny, TUKEY_RATIO), tukeywin(nz, TUKEY_RATIO));
        let mut c: Vec<Complex64> = self.r.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        ws.fft.fft3d(&mut c);
        let mut s = circshift(&c, self.dims, false);
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    s[i + nx * (j + ny * k)] *= (wx[i] * wy[j]) * wz[k];
                }
            }
        }
        c = circshift(&s, self.dims, true);
        ws.fft.ifft3d(&mut c);
        c.iter().map(|v| v.re).collect()
    }
}

/// `d * v`, named so the operation order of the toolbox's `d.*(w.*·)` stays visible
#[inline]
fn dp_mul(d: f64, v: f64) -> f64 {
    d * v
}

/// MATLAB's `min(max(v, 0), 1)`, which maps NaN to 0 (`clamp` would keep it)
#[allow(clippy::manual_clamp)]
fn clamp01(v: f64) -> f64 {
    v.max(0.0).min(1.0)
}

/// MATLAB's `sign` (0 at 0)
fn sign0(v: f64) -> f64 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// `min(max((hi − v)/(hi − lo), 0), 1)` with `lo`, `hi` the `prctile`s `p` of `v` at `inside`
fn ramp_weight(v: &[f64], inside: &[usize], p: (f64, f64)) -> Vec<f64> {
    let mut sel: Vec<f64> = inside.iter().map(|&i| v[i]).collect();
    let lo = prctile(&mut sel, p.0);
    let hi = prctile(&mut sel, p.1);
    v.iter().map(|&x| clamp01((hi - x) / (hi - lo))).collect()
}

/// MATLAB's `prctile(x, p)` (R2023b, `'exact'`, no NaNs): with `r = (p/100)·n` and `k =
/// floor(r + 0.5)`, `(0.5 + r − k)·x(k+1) + (0.5 − r + k)·x(k)` of the sorted values, clamped to
/// the first and last and copied where the two neighbours are equal. `x` is reordered.
fn prctile(x: &mut [f64], p: f64) -> f64 {
    let n = x.len();
    let r = (p / 100.0) * n as f64;
    let k = (r + 0.5).floor();
    let rr = r - k;
    let k1 = (k as i64).max(1) as usize; // 1-based
    let kp1 = ((k as i64 + 1) as usize).min(n);
    let xk = *x.select_nth_unstable_by(k1 - 1, f64::total_cmp).1;
    let xkp1 = if kp1 == k1 { xk } else { x[k1..].iter().copied().fold(f64::INFINITY, f64::min) };
    if rr == -0.5 || xk == xkp1 {
        return xk;
    }
    (0.5 + rr) * xkp1 + (0.5 - rr) * xk
}

/// MATLAB's `tukeywin(n, ratio)` for `0 < ratio < 1`
fn tukeywin(n: usize, ratio: f64) -> Vec<f64> {
    if n == 1 {
        return vec![1.0];
    }
    let n1 = (n - 1) as f64;
    // linspace(0, 1, n)
    let t: Vec<f64> = (0..n).map(|k| if k + 1 == n { 1.0 } else { k as f64 / n1 }).collect();
    let per = ratio / 2.0;
    let tl = (per * n1).floor() as usize + 1;
    let th = n - tl + 1;
    let pi_per = std::f64::consts::PI / per;
    (0..n)
        .map(|k| {
            if k < tl {
                (1.0 + (pi_per * (t[k] - per)).cos()) / 2.0
            } else if k + 1 >= th {
                (1.0 + (pi_per * (t[k] - 1.0 + per)).cos()) / 2.0
            } else {
                1.0
            }
        })
        .collect()
}

/// `fftshift` (`forward`, shift by `floor(n/2)`) or `ifftshift` (shift by `ceil(n/2)`) per axis
fn circshift<T: Copy + Send + Sync>(a: &[T], (nx, ny, nz): (usize, usize, usize), forward: bool) -> Vec<T> {
    let s = |n: usize| if forward { n / 2 } else { n.div_ceil(2) };
    let (sx, sy, sz) = (s(nx), s(ny), s(nz));
    let mut out = a.to_vec();
    for k in 0..nz {
        let kk = (k + sz) % nz;
        for j in 0..ny {
            let jj = (j + sy) % ny;
            for i in 0..nx {
                out[(i + sx) % nx + nx * (jj + ny * kk)] = a[i + nx * (j + ny * k)];
            }
        }
    }
    out
}

/// MATLAB's `del2(f, hx, hy, hz)` with the spacings already in array-axis order (`h[0]` along
/// the first axis): per axis, half the centred second difference inside and linear
/// extrapolation at the ends, summed over the axes and divided by 3. Coordinates are `h·(1:n)`
/// and their differences are taken as MATLAB takes them.
fn del2(f: &[f64], (nx, ny, nz): (usize, usize, usize), h: [f64; 3]) -> Vec<f64> {
    let n = [nx, ny, nz];
    let stride = [1, nx, nx * ny];
    let mut v = vec![0.0; f.len()];
    for ax in 0..3 {
        let len = n[ax];
        let hs: Vec<f64> = (0..len - 1).map(|i| h[ax] * (i + 2) as f64 - h[ax] * (i + 1) as f64).collect();
        let st = stride[ax];
        let g_at = |p0: usize, t: usize| -> f64 {
            // interior second difference at line position t (1 ≤ t ≤ len − 2)
            let p = p0 + t * st;
            ((f[p + st] - f[p]) / hs[t] - (f[p] - f[p - st]) / hs[t - 1]) / (hs[t] + hs[t - 1])
        };
        par_update(&mut v, |l, out| {
            let pos = (l / st) % len;
            let p0 = l - pos * st;
            let g = if len > 3 {
                if pos == 0 {
                    g_at(p0, 1) * (hs[0] + hs[1]) / hs[1] - g_at(p0, 2) * hs[0] / hs[1]
                } else if pos == len - 1 {
                    -g_at(p0, len - 3) * hs[len - 2] / hs[len - 3] + g_at(p0, len - 2) * (hs[len - 2] + hs[len - 3]) / hs[len - 3]
                } else {
                    g_at(p0, pos)
                }
            } else if len == 3 {
                g_at(p0, 1)
            } else {
                0.0
            };
            *out += g;
        });
    }
    par_update(&mut v, |_, x| *x /= 3.0);
    v
}

/// The exact adjoint of MATLAB-style forward differences with unit spacing (0 at the last
/// voxel): per axis `−g(1)`, `g(i−1) − g(i)`, `g(n−1)`, summed over x, y, z.
fn grad_adjoint_unit(g: &[Vec<f64>; 3], (nx, ny, nz): (usize, usize, usize), out: &mut [f64]) {
    let n = [nx, ny, nz];
    let stride = [1, nx, nx * ny];
    par_fill(out, |l| {
        let mut acc = 0.0;
        for ax in 0..3 {
            let (len, st) = (n[ax], stride[ax]);
            let pos = (l / st) % len;
            let gg = &g[ax];
            let t = if pos == 0 {
                -gg[l]
            } else if pos == len - 1 {
                gg[l - st]
            } else {
                gg[l - st] - gg[l]
            };
            acc = if ax == 0 { t } else { acc + t };
        }
        acc
    });
}

/// The fast QSM's smoothing ball: radius `r` mm on the grid `(−n/2 : n/2−1)·vs` per axis (half
/// integers on an odd axis, as the dipole kernel), 1 where the voxel centre is within `r − h`
/// (`h` the half diagonal), and partial-volume weighted by 20³ sub-samples at offsets
/// `(−0.475:0.05:0.475)·vs` where it is within `r + h`; normalised to sum 1. Centred.
fn pv_ball((nx, ny, nz): (usize, usize, usize), r: f64, vs: [f64; 3]) -> Vec<f64> {
    let coord = |n: usize, v: f64| -> Vec<f64> { (0..n).map(|i| (-(n as f64) / 2.0 + i as f64) * v).collect() };
    let (cx, cy, cz) = (coord(nx, vs[0]), coord(ny, vs[1]), coord(nz, vs[2]));
    let h = ((vs[0] / 2.0).powi(2) + (vs[1] / 2.0).powi(2) + (vs[2] / 2.0).powi(2)).sqrt();
    // -0.475:0.05:0.475 as MATLAB's colon builds it (from both ends)
    let sub: Vec<f64> = (0..20).map(|k| if k < 10 { -0.475 + k as f64 * 0.05 } else { 0.475 - (19 - k) as f64 * 0.05 }).collect();
    let mut s = vec![0.0; nx * ny * nz];
    let r2 = r * r;
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let (x, y, z) = (cx[i], cy[j], cz[k]);
                let d = (x * x + y * y + z * z).sqrt();
                let l = i + nx * (j + ny * k);
                if d <= r - h {
                    s[l] = 1.0;
                } else if d < r + h {
                    let mut cnt = 0usize;
                    for &c in &sub {
                        let zz = z + c * vs[2];
                        for &b in &sub {
                            let yy = y + b * vs[1];
                            for &a in &sub {
                                let xx = x + a * vs[0];
                                if xx * xx + yy * yy + zz * zz <= r2 {
                                    cnt += 1;
                                }
                            }
                        }
                    }
                    s[l] = cnt as f64 / 8000.0;
                }
            }
        }
    }
    let tot: f64 = s.iter().sum();
    s.iter_mut().for_each(|v| *v /= tot);
    s
}

/// A sum over `0..n` in fixed chunks, so the result does not depend on the thread count
fn det_sum<T>(n: usize, f: impl Fn(usize) -> T + Sync) -> T
where
    T: Copy + Send + Sync + std::ops::Add<Output = T> + Default,
{
    const CHUNK: usize = 1 << 15;
    let chunks = n.div_ceil(CHUNK);
    let part = |c: usize| {
        let mut acc = T::default();
        for i in c * CHUNK..((c + 1) * CHUNK).min(n) {
            acc = acc + f(i);
        }
        acc
    };
    #[cfg(feature = "parallel")]
    let parts: Vec<T> = {
        use rayon::prelude::*;
        (0..chunks).into_par_iter().map(part).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let parts: Vec<T> = (0..chunks).map(part).collect();
    parts.into_iter().fold(T::default(), |a, b| a + b)
}

/// An element of a `pcg` vector: real (step 1) or complex (step 2).
trait PcgElem:
    Copy + Send + Sync + Default + std::ops::Add<Output = Self> + std::ops::Sub<Output = Self> + std::ops::Mul<Output = Self> + std::ops::Div<Output = Self>
{
    fn one() -> Self;
    /// `conj(self)·o`
    fn cmul(self, o: Self) -> Self;
    fn abs2(self) -> f64;
    fn abs(self) -> f64;
    fn re(self) -> f64;
    fn is_inf(self) -> bool;
    fn is_zero(self) -> bool;
}

impl PcgElem for f64 {
    fn one() -> Self {
        1.0
    }
    fn cmul(self, o: Self) -> Self {
        self * o
    }
    fn abs2(self) -> f64 {
        self * self
    }
    fn abs(self) -> f64 {
        f64::abs(self)
    }
    fn re(self) -> f64 {
        self
    }
    fn is_inf(self) -> bool {
        self.is_infinite()
    }
    fn is_zero(self) -> bool {
        self == 0.0
    }
}

impl PcgElem for Complex64 {
    fn one() -> Self {
        Complex64::new(1.0, 0.0)
    }
    fn cmul(self, o: Self) -> Self {
        self.conj() * o
    }
    fn abs2(self) -> f64 {
        self.norm_sqr()
    }
    fn abs(self) -> f64 {
        self.norm()
    }
    fn re(self) -> f64 {
        self.re
    }
    fn is_inf(self) -> bool {
        self.re.is_infinite() || self.im.is_infinite()
    }
    fn is_zero(self) -> bool {
        self.re == 0.0 && self.im == 0.0
    }
}

/// What [`pcg`] returns: the solution, MATLAB's `flag` and `iter`.
struct PcgOut<T> {
    x: Vec<T>,
    #[allow(dead_code)]
    flag: u8,
    #[allow(dead_code)]
    iter: usize,
}

/// MATLAB R2023b's `pcg(A, b, tol, maxit, [], [], x0)` (no preconditioner), line for line.
fn pcg<T: PcgElem>(mut a: impl FnMut(&[T], &mut [T]), b: &[T], tol: f64, maxit: usize, x0: Option<Vec<T>>) -> PcgOut<T> {
    let n = b.len();
    let norm = |v: &[T]| det_sum(v.len(), |i| v[i].abs2()).sqrt();
    let dot = |u: &[T], v: &[T]| det_sum(u.len(), |i| u[i].cmul(v[i]));
    let n2b = norm(b);
    if n2b == 0.0 {
        return PcgOut { x: vec![T::default(); n], flag: 0, iter: 0 };
    }
    let mut x = x0.unwrap_or_else(|| vec![T::default(); n]);
    let mut flag = 1u8;
    let mut xmin = x.clone();
    let mut imin = 0;
    let tolb = tol * n2b;
    let mut q = vec![T::default(); n];
    a(&x, &mut q);
    let mut r: Vec<T> = (0..n).map(|i| b[i] - q[i]).collect();
    let mut normr_act = norm(&r);
    if normr_act <= tolb {
        return PcgOut { x, flag: 0, iter: 0 };
    }
    let mut normrmin = normr_act;
    let mut rho = T::one();
    let mut stag = 0;
    let mut moresteps = 0;
    let maxmsteps = ((n / 50) as i64).min(5).min(n as i64 - maxit as i64);
    let maxstagsteps = 3;
    let mut p = vec![T::default(); n];
    let mut ii = 0;
    let mut iter = 0;
    for it in 1..=maxit {
        ii = it;
        let rho1 = rho;
        rho = dot(&r, &r);
        if rho.is_zero() || rho.is_inf() {
            flag = 4;
            break;
        }
        if it == 1 {
            p.copy_from_slice(&r);
        } else {
            let beta = rho / rho1;
            if beta.is_zero() || beta.is_inf() {
                flag = 4;
                break;
            }
            par_update(&mut p, |i, v| *v = r[i] + beta * *v);
        }
        a(&p, &mut q);
        let pq = dot(&p, &q);
        if pq.re() <= 0.0 || pq.is_inf() {
            flag = 4;
            break;
        }
        let alpha = rho / pq;
        if alpha.is_inf() {
            flag = 4;
            break;
        }
        if norm(&p) * alpha.abs() < f64::EPSILON * norm(&x) {
            stag += 1;
        } else {
            stag = 0;
        }
        par_update(&mut x, |i, v| *v = *v + alpha * p[i]);
        par_update(&mut r, |i, v| *v = *v - alpha * q[i]);
        let normr = norm(&r);
        normr_act = normr;
        if normr <= tolb || stag >= maxstagsteps || moresteps > 0 {
            a(&x, &mut q);
            par_fill(&mut r, |i| b[i] - q[i]);
            normr_act = norm(&r);
            if normr_act <= tolb {
                flag = 0;
                iter = it;
                break;
            }
            if stag >= maxstagsteps && moresteps == 0 {
                stag = 0;
            }
            moresteps += 1;
            if moresteps >= maxmsteps {
                flag = 3;
                iter = it;
                break;
            }
        }
        if normr_act < normrmin {
            normrmin = normr_act;
            xmin.copy_from_slice(&x);
            imin = it;
        }
        if stag >= maxstagsteps {
            flag = 3;
            break;
        }
    }
    if flag != 0 {
        a(&xmin, &mut q);
        let r_comp = norm(&(0..n).map(|i| b[i] - q[i]).collect::<Vec<_>>());
        if r_comp <= normr_act {
            x = xmin;
            iter = imin;
        } else {
            iter = ii;
        }
    }
    PcgOut { x, flag, iter }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_the_toolbox_values() {
        let p = ChiSepIlsqrParams::default();
        assert_eq!((p.dr_pos, p.dr_neg, p.pad), (None, None, [16; 3]));
        assert_eq!((p.cg_max_iter_lsqr, p.cg_tol_lsqr, p.cg_max_iter_sa, p.cg_tol_sa), (50, 0.02, 50, 0.02));
        assert_eq!((p.d2_thr, p.prc_w_i, p.prc_w_g), (0.11, (60.0, 99.8), (50.0, 70.0)));
        assert_eq!((NNLS_TOL, NNLS_ITER_MAX), (0.01, 10));
        let (dp, dn) = ChiSepIlsqrParams { cf: 127.74e6, ..p.clone() }.dr();
        assert_eq!((dp, dn), (141.7241379310345, 141.7241379310345));
        assert_eq!(p.qsm_ilsqr.max_iter, STI_DEFAULT_NITER);
    }

    #[test]
    fn param_errors_name_the_parameter() {
        let bad = [
            ChiSepIlsqrParams { cf: 0.0, ..Default::default() },
            ChiSepIlsqrParams { dr_pos: Some(-1.0), ..Default::default() },
            ChiSepIlsqrParams { dr_neg: Some(f64::NAN), ..Default::default() },
            ChiSepIlsqrParams { d2_thr: 0.0, ..Default::default() },
            ChiSepIlsqrParams { cg_tol_lsqr: 1.0, ..Default::default() },
            ChiSepIlsqrParams { cg_tol_sa: 0.0, ..Default::default() },
            ChiSepIlsqrParams { prc_w_i: (99.8, 60.0), ..Default::default() },
            ChiSepIlsqrParams { prc_w_g: (50.0, 101.0), ..Default::default() },
        ];
        let names = ["cf", "dr_pos", "dr_neg", "d2_thr", "cg_tol_lsqr", "cg_tol_sa", "prc_w_i", "prc_w_g"];
        for (p, name) in bad.iter().zip(names) {
            let e = p.validate().unwrap_err().to_string();
            assert!(e.starts_with(&format!("χ-separation iLSQR {name} = ")), "{e}");
        }
        assert!(ChiSepIlsqrParams::default().validate().is_ok());
    }

    #[test]
    fn wrong_buffer_length_and_empty_mask_are_input_errors() {
        let nt = 8 * 8 * 6;
        let v = vec![0.0; nt];
        let short = vec![0.0; nt - 1];
        let m = vec![1u8; nt];
        let grid = Grid::new(8, 8, 6, 1.0, 1.0, 1.0);
        let inputs = ChiSepIlsqrInputs { local_field: &v, r2prime: &short, qsm: &v, mask: &m };
        let e = chi_sep_ilsqr(&inputs, &grid, (0.0, 0.0, 1.0), &ChiSepIlsqrParams::default(), |_, _| {}).unwrap_err();
        assert_eq!(e.to_string(), format!("χ-separation iLSQR r2prime has {} values, expected {nt}", nt - 1));
        let z = vec![0u8; nt];
        let inputs = ChiSepIlsqrInputs { local_field: &v, r2prime: &v, qsm: &v, mask: &z };
        assert!(chi_sep_ilsqr(&inputs, &grid, (0.0, 0.0, 1.0), &ChiSepIlsqrParams::default(), |_, _| {}).is_err());
    }

    /// Values from MATLAB R2023b.
    #[test]
    fn matlab_helpers() {
        let mut x: Vec<f64> = vec![5.0, 1.0, 4.0, 2.0, 3.0];
        // prctile(1:5, [1 30 60 99.8]) = 1, 2, 3.5, 5
        assert_eq!(prctile(&mut x.clone(), 1.0), 1.0);
        assert_eq!(prctile(&mut x.clone(), 30.0), 2.0);
        assert_eq!(prctile(&mut x.clone(), 60.0), 3.5);
        assert_eq!(prctile(&mut x, 99.8), 5.0);
        // tukeywin(9, 0.5)' = [0 0.5 1 1 1 1 1 0.5 0]
        let w = tukeywin(9, 0.5);
        let want = [0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0];
        assert!(w.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-15), "{w:?}");
        // del2 of x.^2 along the first axis is 2/6 everywhere (exact quadratic, extrapolated ends)
        let dims = (6, 4, 5);
        let f: Vec<f64> = (0..120).map(|l| ((l % 6) as f64).powi(2)).collect();
        assert!(del2(&f, dims, [1.0; 3]).iter().all(|v| (v - 2.0 / 6.0).abs() < 1e-13));
        // ifftshift then fftshift is the identity on odd and even axes
        let a: Vec<usize> = (0..7 * 4 * 3).collect();
        assert_eq!(circshift(&circshift(&a, (7, 4, 3), false), (7, 4, 3), true), a);
        assert_eq!(circshift(&[0, 1, 2, 3, 4], (5, 1, 1), true), vec![3, 4, 0, 1, 2]);
    }

    /// MATLAB's `pcg` returns its start when one iteration does not lower the residual, and the
    /// exact solution of a small SPD system otherwise.
    #[test]
    fn pcg_follows_matlab() {
        let m = [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
        let apply = |x: &[f64], o: &mut [f64]| {
            for i in 0..3 {
                o[i] = (0..3).map(|j| m[i][j] * x[j]).sum();
            }
        };
        let b = [1.0, 2.0, 3.0];
        let out = pcg(apply, &b, 1e-12, 10, None);
        let mut ax = [0.0; 3];
        apply(&out.x, &mut ax);
        assert!(ax.iter().zip(&b).all(|(a, b)| (a - b).abs() < 1e-10), "{:?}", out.x);
        assert_eq!(out.flag, 0);
        // complex unknown with an R-linear operator: Re-part projection keeps CG well defined
        let bc = [Complex64::new(1.0, 0.5), Complex64::new(-1.0, 0.0)];
        let out = pcg(|x: &[Complex64], o: &mut [Complex64]| { o[0] = x[0] * 2.0; o[1] = x[1] * 3.0; }, &bc, 1e-12, 5, None);
        assert!((out.x[0] - Complex64::new(0.5, 0.25)).norm() < 1e-14 && (out.x[1] - Complex64::new(-1.0 / 3.0, 0.0)).norm() < 1e-14);
    }

    #[test]
    fn sources_have_their_signs() {
        let (nx, ny, nz) = (20, 18, 14);
        let nt = nx * ny * nz;
        let mut mask = vec![0u8; nt];
        let mut chi = vec![0.0; nt];
        let mut r2p = vec![0.0; nt];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let l = i + nx * (j + ny * k);
                    let (x, y, z) = (i as f64 - 10.0, j as f64 - 9.0, k as f64 - 7.0);
                    if (x / 8.0).powi(2) + (y / 7.0).powi(2) + (z / 5.5).powi(2) <= 1.0 {
                        mask[l] = 1;
                        let (p, q) = (if x > 1.0 { 0.05 } else { 0.0 }, if x < -1.0 { 0.03 } else { 0.0 });
                        chi[l] = p - q;
                        r2p[l] = 141.0 * (p + q) + 2.0;
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
        let inputs = ChiSepIlsqrInputs { local_field: &field, r2prime: &r2p, qsm: &chi, mask: &mask };
        let mut iters = 0;
        let (p, q, t) = chi_sep_ilsqr(&inputs, &grid, (0.0, 0.0, 1.0), &ChiSepIlsqrParams { pad: [4; 3], ..Default::default() }, |i, _| iters = i).unwrap();
        assert!((1..=NNLS_ITER_MAX + 1).contains(&iters));
        for l in 0..nt {
            assert!(p[l] >= 0.0 && q[l] <= 0.0 && t[l] == p[l] + q[l]);
            if mask[l] == 0 {
                assert!(p[l] == 0.0 && q[l] == 0.0);
            }
        }
        assert!(p.iter().any(|&v| v > 0.01) && q.iter().any(|&v| v < -0.01));
    }
}

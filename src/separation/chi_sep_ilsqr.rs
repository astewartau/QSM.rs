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
//! (216×288×160 padded, 4 NNLS iterations) the toolbox takes 433 s on 8 threads and on
//! `chisep-mc` (196×237×237) 1113 s (743 s on 4); QSM.rs took 280 s on 6 threads and 398 s with
//! the memory layout before the one below, which is faster again (155 s → 91 s on 12 threads on
//! a phantom of the in-vivo size).
//!
//! **Memory.** The arithmetic is the toolbox's, in the order above; what is stored is cut to what
//! each step needs, without changing a single operation:
//!
//! - the dipole kernel is evaluated where it is used, from per-axis tables, by the same
//!   expressions as `chi_sep_medi`'s kernel (bit for bit, tested);
//! - step 1's operator chains its convolutions through one complex buffer (each convolution's
//!   real part is the next one's input, as before) and keeps its R2′ term in its output;
//! - `W_I` and the NNLS iterates are stored at the mask's voxels only (they are zero elsewhere);
//!   step 2's weights `W_G²` are recomputed where used from the masked fast QSM and smoothed R2′
//!   (kept on the mask's bounding box) and their percentiles, by the same expressions; the
//!   gradient of step 2's regulariser is fused into its adjoint;
//! - step 2's complex `pcg` stores only the cone's voxels (19 % of k-space at the default
//!   `D2_thr`): its vectors are zero outside the cone, and its sums add the same terms in the
//!   same chunks;
//! - `pcg` copies its best iterate (`xmin`) only once it differs from the current one, and the
//!   NNLS iterations' start is rebuilt from the iterates rather than stored;
//! - the k-space shifts run in place.
//!
//! The peak is step 1's first solve: `pcg`'s `x`, `xmin`, `q`, `r`, `p` (10 `f64` per padded
//! voxel), the right-hand side (2) and the FFT buffer (2), plus what the NNLS iterations keep
//! (the mask, `W_I`, the two volumes `W_G²` is computed from, the cone): about **14.25 `f64` per padded voxel +
//! 1.5 per mask voxel + 2 per voxel of the mask's bounding box**. Later, with the current and best
//! iterates held (4 per mask voxel), it is 12.25 + 5.5 per mask voxel + 2 per box voxel, which is
//! more only if the mask fills over half the padded grid; a bound for any mask is 14.25 per padded
//! voxel + 7.5 per input voxel. A larger `D2_thr` widens the cone and step 2's share (it exceeds
//! step 1's above about 0.25). Measured on the phantom of `examples/chi_sep_ilsqr_memory.rs` at
//! the in-vivo size (184×256×128, 16 % of the padded grid in the mask): a 1.20 GB heap peak, 15.1
//! `f64` per padded voxel (1.34 GB resident), where the earlier layout needed 3.43 GB (43 per padded
//! voxel). The results are the same numbers: χ+, χ− and χ_total equal the earlier layout's bit for
//! bit inside the mask (on the parity fixtures and that phantom, with and without threads); outside
//! it both are zero, now +0 throughout where the earlier one left some −0.
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
use crate::inversion::medi::{par_fill, par_update};
use crate::inversion::{IlsqrParams, STI_DEFAULT_NITER, STI_GUI_PAD_MM};
use crate::Grid;
use num_complex::Complex64;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::chi_sep_medi::toolbox_default_dr;
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
/// length is not the grid's, for an empty mask, and for a padded grid of more than 2³² voxels.
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
    let padded: usize = (0..3).map(|a| [grid.dims.0, grid.dims.1, grid.dims.2][a] + 2 * params.pad[a]).product();
    if padded > u32::MAX as usize {
        return Err(InputError(format!("χ-separation iLSQR padded grid has {padded} voxels, more than 2^32")));
    }
    let (pb, mut ws, r) = Problem::new(inputs, grid, bdir, params);
    let n = pb.n;

    // Step 1 from the QSM, then step 2
    let (drp, drn) = (pb.drp, pb.drn);
    let mut x0 = vec![0.0; 2 * n];
    {
        let q = pb.pad_input(inputs.qsm);
        let (a, b) = x0.split_at_mut(n);
        let m = &pb.m;
        par_fill(a, |i| ((r[i] + drn * q[i]) / (drp + drn)).max(0.0) * f64::from(m[i]));
        par_fill(b, |i| ((r[i] - drp * q[i]) / (drp + drn)).max(0.0) * f64::from(m[i]));
    }
    drop(r);
    let mut x = pb.step1(&mut ws, Start::Owned(x0), params.cg_tol_lsqr, params.cg_max_iter_lsqr);
    pb.step2(&mut ws, &mut x, params);
    // The iterates live on the mask's voxels only (they are zero elsewhere): `[x+; x−]` there.
    let mut cur = pb.on_mask(&x, false);
    drop(x);

    // Step 3: NNLS
    let ni = pb.inside.len();
    let max_it = NNLS_ITER_MAX + 1;
    let mut res = f64::INFINITY;
    let mut res_min = f64::INFINITY;
    let mut best = cur.clone();
    let mut it = 1;
    while res > NNLS_TOL && it <= max_it {
        // the start [x+.*m; x−.*m], rebuilt by `pcg` when it needs it again
        let start = |x: &mut [f64]| pb.padded_from_mask(&cur, x);
        let mut x = pb.step1(&mut ws, Start::Fill(&start), params.cg_tol_lsqr, 1);
        pb.step2(&mut ws, &mut x, params);
        // a.*(a > 0)
        let new = pb.on_mask(&x, true);
        drop(x);
        let sums = |f: &(dyn Fn(usize) -> f64 + Sync)| pb.inside.sum(f).sqrt();
        let (np, nn) = (&new[..ni], &new[ni..]);
        let (cp, cn) = (&cur[..ni], &cur[ni..]);
        res = (sums(&|j| (np[j] - cp[j]) * (np[j] - cp[j])) + sums(&|j| (nn[j] - cn[j]) * (nn[j] - cn[j])))
            / (sums(&|j| np[j] * np[j]) + sums(&|j| nn[j] * nn[j]));
        cur = new;
        if res < res_min {
            res_min = res;
            best.copy_from_slice(&cur);
        }
        progress(it, max_it);
        it += 1;
    }
    drop(cur);

    let (chi_pos, chi_neg) = pb.crop_from_mask(&best);
    let chi_total = chi_pos.iter().zip(&chi_neg).map(|(p, q)| p + q).collect();
    Ok((chi_pos, chi_neg, chi_total))
}

/// Chunk length of [`det_sum`]'s partial sums.
const CHUNK: usize = 1 << 15;

/// A set of voxels of the padded grid, for vectors stored on those voxels only (the voxels of
/// the full vector outside it are zero): their indices, ascending, and for each [`CHUNK`] of the
/// full grid where its entries start. Sums over such a vector ([`Self::sum`]) add the same terms
/// in the same chunks as [`det_sum`] over the full vector, less the zeros, so they are equal.
struct Support {
    idx: Vec<u32>,
    bounds: Vec<usize>,
}

impl Support {
    /// The voxels of `0..n` where `keep`
    fn new(n: usize, keep: impl Fn(usize) -> bool) -> Self {
        let mut idx = Vec::new();
        let mut bounds = Vec::with_capacity(n.div_ceil(CHUNK) + 1);
        for l in 0..n {
            if l.is_multiple_of(CHUNK) {
                bounds.push(idx.len());
            }
            if keep(l) {
                idx.push(l as u32);
            }
        }
        bounds.push(idx.len());
        idx.shrink_to_fit();
        Self { idx, bounds }
    }

    fn len(&self) -> usize {
        self.idx.len()
    }

    /// [`det_sum`] of `f` over the full vector, `f(j)` the term of the set's `j`-th voxel
    fn sum<T>(&self, f: impl Fn(usize) -> T + Sync) -> T
    where
        T: Copy + Send + Sync + std::ops::Add<Output = T> + Default,
    {
        chunked_sum(self.bounds.len() - 1, |c| self.bounds[c]..self.bounds[c + 1], f)
    }

    /// `f(l, Some(j))` for the set's `j`-th voxel `l` and `f(l, None)` for the others, over the
    /// voxels of chunk `c` of a grid of `n`, in order
    fn walk(&self, c: usize, n: usize, mut f: impl FnMut(usize, Option<usize>)) {
        let (mut j, end) = (self.bounds[c], self.bounds[c + 1]);
        for l in c * CHUNK..((c + 1) * CHUNK).min(n) {
            if j < end && self.idx[j] as usize == l {
                f(l, Some(j));
                j += 1;
            } else {
                f(l, None);
            }
        }
    }
}

/// `f(c, chunk)` for each [`CHUNK`] of `a` (over threads with the `parallel` feature)
fn par_chunked<T: Send>(a: &mut [T], f: impl Fn(usize, &mut [T]) + Sync + Send) {
    #[cfg(feature = "parallel")]
    a.par_chunks_mut(CHUNK).enumerate().for_each(|(c, s)| f(c, s));
    #[cfg(not(feature = "parallel"))]
    a.chunks_mut(CHUNK).enumerate().for_each(|(c, s)| f(c, s));
}

/// [`par_chunked`] over two volumes at once
fn par_chunked2<A: Send, B: Send>(a: &mut [A], b: &mut [B], f: impl Fn(usize, &mut [A], &mut [B]) + Sync + Send) {
    #[cfg(feature = "parallel")]
    a.par_chunks_mut(CHUNK).zip(b.par_chunks_mut(CHUNK)).enumerate().for_each(|(c, (s, t))| f(c, s, t));
    #[cfg(not(feature = "parallel"))]
    a.chunks_mut(CHUNK).zip(b.chunks_mut(CHUNK)).enumerate().for_each(|(c, (s, t))| f(c, s, t));
}

/// `f(j, k, l0, row)` for each x-row of a volume (`l0` its first index; over threads with the
/// `parallel` feature)
fn par_rows<T: Send>(a: &mut [T], (nx, ny, _): (usize, usize, usize), f: impl Fn(usize, usize, usize, &mut [T]) + Sync + Send) {
    #[cfg(feature = "parallel")]
    a.par_chunks_mut(nx).with_min_len(64).enumerate().for_each(|(r, row)| f(r % ny, r / ny, r * nx, row));
    #[cfg(not(feature = "parallel"))]
    a.chunks_mut(nx).enumerate().for_each(|(r, row)| f(r % ny, r / ny, r * nx, row));
}

/// The toolbox's dipole kernel (`chi_sep_medi::toolbox_dipole_kernel`) evaluated where it is used, from its
/// per-axis k coordinates: the same expressions on the same values, so the same numbers, without
/// a volume of it.
struct Kernel {
    /// k coordinate of each array index, per axis, after the kernel's `fftshift`
    k: [Vec<f64>; 3],
    b0: (f64, f64, f64),
}

impl Kernel {
    fn new(dims: (usize, usize, usize), vs: [f64; 3], b0: (f64, f64, f64)) -> Self {
        let axis = |n: usize, v: f64| {
            let mut t = vec![0.0; n];
            for i in 0..n {
                t[(i + n / 2) % n] = (-(n as f64) / 2.0 + i as f64) / (n as f64 * v);
            }
            t
        };
        Self { k: [axis(dims.0, vs[0]), axis(dims.1, vs[1]), axis(dims.2, vs[2])], b0 }
    }

    /// `D` at array index `(i, j, k)`
    #[inline]
    fn at(&self, i: usize, j: usize, k: usize) -> f64 {
        let (x, y, z) = (self.k[0][i], self.k[1][j], self.k[2][k]);
        let s = x * self.b0.0 + y * self.b0.1 + z * self.b0.2;
        let den = x * x + y * y + z * z;
        let v = 1.0 / 3.0 - (s * s) / den;
        if v.is_nan() {
            0.0
        } else {
            v
        }
    }
}

/// A volume stored on a box of the padded grid, zero outside it.
struct BoxVol {
    lo: [usize; 3],
    len: [usize; 3],
    data: Vec<f64>,
}

impl BoxVol {
    fn new(a: &[f64], (nx, ny, _): (usize, usize, usize), lo: [usize; 3], hi: [usize; 3]) -> Self {
        let len = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
        let mut data = Vec::with_capacity(len[0] * len[1] * len[2]);
        for k in lo[2]..hi[2] {
            for j in lo[1]..hi[1] {
                let s = lo[0] + nx * (j + ny * k);
                data.extend_from_slice(&a[s..s + len[0]]);
            }
        }
        Self { lo, len, data }
    }

    #[inline]
    fn get(&self, p: [usize; 3]) -> f64 {
        let q = [p[0].wrapping_sub(self.lo[0]), p[1].wrapping_sub(self.lo[1]), p[2].wrapping_sub(self.lo[2])];
        if q[0] < self.len[0] && q[1] < self.len[1] && q[2] < self.len[2] {
            self.data[q[0] + self.len[0] * (q[1] + self.len[1] * q[2])]
        } else {
            0.0
        }
    }
}

/// The arrays of one run, on the padded grid. What is fixed across the NNLS iterations is kept
/// in the least memory that gives the same numbers (see *Performance* in the module docs).
struct Problem {
    dims: (usize, usize, usize),
    n: usize,
    /// input grid and padding
    dims0: (usize, usize, usize),
    pad: [usize; 3],
    vs: [f64; 3],
    /// mask (0/1)
    m: Vec<u8>,
    /// the mask's voxels
    inside: Support,
    /// `D_r,±` in ppm
    drp: f64,
    drn: f64,
    /// dipole kernel (`dipole_kernel(..., 'kspace')`)
    d: Kernel,
    /// step 1's data weight `W_I` at the mask's voxels: `w_mag = W_I.*m` there, 0 elsewhere,
    /// and `w_phs = 2π·w_mag`
    w_i: Vec<f64>,
    /// step 1's right-hand side, `[b+; b−]`
    b1: Vec<f64>,
    /// step 2's gradient weights are recomputed where used from the masked fast QSM and smoothed
    /// R2′ (zero outside the mask, so kept on its bounding box) and their percentiles `(lo, hi)`
    /// per source and axis
    xm: BoxVol,
    rm: BoxVol,
    wg_thr: [[(f64, f64); 3]; 2],
    /// step 2's cone `|D| < D2_thr`
    cone: Support,
}

/// FFT buffers reused across operator applications.
struct Workspace {
    fft: Fft3dWorkspace,
    c: Vec<Complex64>,
}

impl Problem {
    /// The problem, the FFT workspace, and R2′ (ppm, clamped) on the padded grid for the start.
    fn new(inputs: &ChiSepIlsqrInputs, grid: &Grid, bdir: (f64, f64, f64), params: &ChiSepIlsqrParams) -> (Self, Workspace, Vec<f64>) {
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
            vs,
            m: Vec::new(),
            inside: Support { idx: Vec::new(), bounds: Vec::new() },
            drp: 0.0,
            drn: 0.0,
            d: Kernel::new(dims, vs, bdir),
            w_i: Vec::new(),
            b1: Vec::new(),
            xm: BoxVol { lo: [0; 3], len: [0; 3], data: Vec::new() },
            rm: BoxVol { lo: [0; 3], len: [0; 3], data: Vec::new() },
            wg_thr: [[(0.0, 0.0); 3]; 2],
            cone: Support { idx: Vec::new(), bounds: Vec::new() },
        };
        let mut ws = Workspace { fft: Fft3dWorkspace::new(dims.0, dims.1, dims.2), c: vec![Complex64::new(0.0, 0.0); n] };
        let cf = params.cf;
        let mask: Vec<u8> = inputs.mask.iter().map(|&v| u8::from(v != 0)).collect();
        pb.m = pb.pad_input(&mask);
        drop(mask);
        pb.inside = Support::new(n, |l| pb.m[l] != 0);
        let mut r = pb.pad_input(inputs.r2prime);
        par_update(&mut r, |_, v| *v = (*v / cf * 1e6).max(0.0));
        let f = pb.pad_input(inputs.local_field);
        let (dp, dn) = params.dr();
        pb.drp = dp / cf * 1e6;
        pb.drn = dn / cf * 1e6;

        // Step 1: weights and right-hand side
        let mut lap = del2(&f, dims, [vs[1], vs[0], vs[2]]);
        par_update(&mut lap, |_, v| *v = v.abs());
        let (lo, hi) = prc_on(&pb.inside, |l| lap[l], params.prc_w_i);
        pb.w_i = pb.inside.idx.iter().map(|&l| clamp01((hi - lap[l as usize]) / (hi - lo))).collect();
        drop(lap);
        let mut b1 = vec![0.0; 2 * n];
        {
            let (bp, bn) = b1.split_at_mut(n);
            // Dc(Dc(w_mag.*r)), shared by ArH(·, d±)
            pb.load_w(&mut ws.c, |l, w| w * r[l]);
            pb.dc_c(&mut ws);
            pb.dc_out(&mut ws, bp);
            // Af(Dc(w_phs.*f))
            pb.load_w(&mut ws.c, |l, w| 2.0 * std::f64::consts::PI * w * f[l]);
            pb.dc_c(&mut ws);
            pb.af_c(&mut ws);
            let c = &ws.c;
            par_fill(bn, |i| c[i].re);
            let (drp, drn) = (pb.drp, pb.drn);
            let (w_i, inside) = (&pb.w_i, &pb.inside);
            par_chunked2(bp, bn, |ch, sp, sn| {
                let o = ch * CHUNK;
                inside.walk(ch, n, |l, j| {
                    let w = j.map_or(0.0, |j| w_i[j]);
                    let (t, af) = (sp[l - o], sn[l - o]);
                    sp[l - o] = dp_mul(drp, w * t) + af;
                    sn[l - o] = dp_mul(drn, w * t) - af;
                });
            });
        }
        pb.b1 = b1;

        // Step 2: fast QSM, smoothed R2′, gradient weights, cone
        let mut xm = pb.fast_qsm(&mut ws, f, vs);
        let mut rm = pb.tukey_lpf(&mut ws, &r);
        let m = &pb.m;
        par_update(&mut xm, |i, v| *v *= f64::from(m[i]));
        par_update(&mut rm, |i, v| *v *= f64::from(m[i]));
        let (mut lo, mut hi) = ([usize::MAX; 3], [0; 3]);
        for &l in &pb.inside.idx {
            let l = l as usize;
            let p = [l % dims.0, (l / dims.0) % dims.1, l / (dims.0 * dims.1)];
            for a in 0..3 {
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a] + 1);
            }
        }
        pb.xm = BoxVol::new(&xm, dims, lo, hi);
        drop(xm);
        pb.rm = BoxVol::new(&rm, dims, lo, hi);
        drop(rm);
        for s in 0..2 {
            for c in 0..3 {
                let at = |l: usize| {
                    let p = [l % dims.0, (l / dims.0) % dims.1, l / (dims.0 * dims.1)];
                    pb.g_pm(s, c, p)
                };
                pb.wg_thr[s][c] = prc_on(&pb.inside, at, params.prc_w_g);
            }
        }
        let (d, (nx, ny, _)) = (&pb.d, dims);
        pb.cone = Support::new(n, |l| d.at(l % nx, (l / nx) % ny, l / (nx * ny)).abs() < params.d2_thr);
        (pb, ws, r)
    }

    /// `padarray(a, pad, 0, 'both')` of an input-grid volume
    fn pad_input<T: Copy + Default>(&self, a: &[T]) -> Vec<T> {
        let (nx, ny, nz) = self.dims;
        let (mx, my, _) = self.dims0;
        let [px, py, pz] = self.pad;
        let mut out = vec![T::default(); nx * ny * nz];
        for k in 0..self.dims0.2 {
            for j in 0..my {
                let src = mx * (j + my * k);
                let dst = px + nx * ((j + py) + ny * (k + pz));
                out[dst..dst + mx].copy_from_slice(&a[src..src + mx]);
            }
        }
        out
    }

    /// `[x+; x−]` of a padded `[x+; x−]` at the mask's voxels, with `a.*(a > 0)` if `clamp`
    fn on_mask(&self, x: &[f64], clamp: bool) -> Vec<f64> {
        let ni = self.inside.len();
        let mut out = vec![0.0; 2 * ni];
        let (n, idx) = (self.n, &self.inside.idx);
        par_fill(&mut out, |j| {
            let v = if j < ni { x[idx[j] as usize] } else { x[n + idx[j - ni] as usize] };
            if clamp {
                v * f64::from(u8::from(v > 0.0))
            } else {
                v
            }
        });
        out
    }

    /// The padded `[x+.*m; x−.*m]` of `[x+; x−]` at the mask's voxels, into `x`
    fn padded_from_mask(&self, v: &[f64], x: &mut [f64]) {
        let (n, ni) = (self.n, self.inside.len());
        let (a, b) = x.split_at_mut(n);
        let inside = &self.inside;
        par_chunked2(a, b, |ch, sa, sb| {
            let o = ch * CHUNK;
            inside.walk(ch, n, |l, j| {
                (sa[l - o], sb[l - o]) = j.map_or((0.0, 0.0), |j| (v[j], v[ni + j]));
            });
        });
    }

    /// `(χ+, χ−)` on the input grid from `[x+; x−]` at the mask's voxels (`χ− = −x−`)
    fn crop_from_mask(&self, v: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let (nx, ny, _) = self.dims;
        let (mx, my, mz) = self.dims0;
        let [px, py, pz] = self.pad;
        let ni = self.inside.len();
        let (mut p, mut q) = (vec![0.0; mx * my * mz], vec![0.0; mx * my * mz]);
        for (j, &l) in self.inside.idx.iter().enumerate() {
            let l = l as usize;
            let (i, jj, k) = (l % nx, (l / nx) % ny, l / (nx * ny));
            let o = (i - px) + mx * ((jj - py) + my * (k - pz));
            p[o] = v[j];
            q[o] = -v[ni + j];
        }
        (p, q)
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

    /// `c ← f(l, w)` as a real volume, with `w = W_I` at the mask's voxels and 0 elsewhere (the
    /// value of `w_mag` there)
    fn load_w(&self, c: &mut [Complex64], f: impl Fn(usize, f64) -> f64 + Sync + Send) {
        let (n, w_i, inside) = (self.n, &self.w_i, &self.inside);
        par_chunked(c, |ch, s| {
            let o = ch * CHUNK;
            inside.walk(ch, n, |l, j| s[l - o] = Complex64::new(f(l, j.map_or(0.0, |j| w_i[j])), 0.0));
        });
    }

    /// `c ← D.*c`
    fn kmul(&self, c: &mut [Complex64]) {
        let d = &self.d;
        par_rows(c, self.dims, |j, k, _, row| {
            for (i, v) in row.iter_mut().enumerate() {
                *v *= d.at(i, j, k);
            }
        });
    }

    /// `c ← ifftn(D.*fftn(c))`
    fn dc_raw(&self, ws: &mut Workspace) {
        ws.fft.fft3d(&mut ws.c);
        self.kmul(&mut ws.c);
        ws.fft.ifft3d(&mut ws.c);
    }

    /// `c ← Dc(c)`, `c` holding a real volume: `real(ifftn(D.*fftn(c)))`, as a real volume
    fn dc_c(&self, ws: &mut Workspace) {
        self.dc_raw(ws);
        par_update(&mut ws.c, |_, v| *v = Complex64::new(v.re, 0.0));
    }

    /// `out ← Dc(c)`, `c` holding a real volume
    fn dc_out(&self, ws: &mut Workspace, out: &mut [f64]) {
        self.dc_raw(ws);
        let c = &ws.c;
        par_fill(out, |i| c[i].re);
    }

    /// `c ← Af(c) = Dc(w_phs.*Dc(c))`, `c` holding a real volume
    fn af_c(&self, ws: &mut Workspace) {
        self.dc_raw(ws);
        let (n, w_i, inside) = (self.n, &self.w_i, &self.inside);
        par_chunked(&mut ws.c, |ch, s| {
            let o = ch * CHUNK;
            inside.walk(ch, n, |l, j| {
                let w = j.map_or(0.0, |j| 2.0 * std::f64::consts::PI * w_i[j]);
                s[l - o] = Complex64::new(s[l - o].re * w, 0.0);
            });
        });
        self.dc_c(ws);
    }

    /// Step 1's normal operator on `[x+; x−]`
    fn a1(&self, ws: &mut Workspace, x: &[f64], out: &mut [f64]) {
        let n = self.n;
        let (xp, xn) = x.split_at(n);
        let (drp, drn) = (self.drp, self.drn);
        let (op, on) = out.split_at_mut(n);
        // R2′ part: ArH(Ar(x+, d+) + Ar(x−, d−), d±) = d±.*(w_mag.*Dc(Dc(w_mag.*(d+ x+ + d− x−)))),
        // the inner part in `op` until the end
        self.load_w(&mut ws.c, |l, w| w * (drp * xp[l]) + w * (drn * xn[l]));
        self.dc_c(ws);
        self.dc_out(ws, op);
        // field part: Af(Af(x+ − x−))
        par_fill(&mut ws.c, |i| Complex64::new(xp[i] - xn[i], 0.0));
        self.af_c(ws);
        self.af_c(ws);
        let (c, w_i, inside) = (&ws.c, &self.w_i, &self.inside);
        par_chunked2(op, on, |ch, sp, sn| {
            let o = ch * CHUNK;
            inside.walk(ch, n, |l, j| {
                let w = j.map_or(0.0, |j| w_i[j]);
                let (s, t) = (sp[l - o], c[l].re);
                sp[l - o] = dp_mul(drp, w * s) + t;
                sn[l - o] = dp_mul(drn, w * s) - t;
            });
        });
    }

    /// Step 1: `pcg` from `start`; returns the solution with the padding ring zeroed.
    fn step1(&self, ws: &mut Workspace, start: Start<f64>, tol: f64, maxit: usize) -> Vec<f64> {
        let o = pcg(|v, o| self.a1(ws, v, o), &self.b1, tol, maxit, start, 2 * self.n, None);
        let mut x = o.x;
        let (a, b) = x.split_at_mut(self.n);
        self.zero_ring(a);
        self.zero_ring(b);
        x
    }

    /// `out ← real(ifftn(M_ic.*y))`, `y` on the cone's voxels
    fn cone_ifft(&self, ws: &mut Workspace, y: &[Complex64], out: &mut [f64]) {
        let (n, cone) = (self.n, &self.cone);
        par_chunked(&mut ws.c, |ch, s| {
            let o = ch * CHUNK;
            cone.walk(ch, n, |l, j| s[l - o] = j.map_or(Complex64::new(0.0, 0.0), |j| y[j]));
        });
        ws.fft.ifft3d(&mut ws.c);
        let c = &ws.c;
        par_fill(out, |i| c[i].re);
    }

    /// The fgrad component `c` (voxel size `vs`) of the masked fast QSM and smoothed R2′ at `p`,
    /// combined into `|g+|` (`s` = 0) or `|g−|` (`s` = 1)
    #[inline]
    fn g_pm(&self, s: usize, c: usize, p: [usize; 3]) -> f64 {
        let len = [self.dims.0, self.dims.1, self.dims.2];
        let (gx, gr) = if p[c] + 1 < len[c] {
            let mut q = p;
            q[c] += 1;
            ((self.xm.get(q) - self.xm.get(p)) / self.vs[c], (self.rm.get(q) - self.rm.get(p)) / self.vs[c])
        } else {
            (0.0, 0.0)
        };
        let (drp, drn) = (self.drp, self.drn);
        if s == 0 {
            ((gr + drn * gx) / (drp + drn)).abs()
        } else {
            ((gr - drp * gx) / (drp + drn)).abs()
        }
    }

    /// `W_G²` of source `s`, axis `c` at voxel `p` (`l`): `conj(W).*W` with `W = W_G.*M`
    #[inline]
    fn wg2(&self, s: usize, c: usize, p: [usize; 3], l: usize) -> f64 {
        if self.m[l] == 0 {
            return 0.0;
        }
        let (lo, hi) = self.wg_thr[s][c];
        let w = clamp01((hi - self.g_pm(s, c, p)) / (hi - lo));
        w * w
    }

    /// `out ← M_ic.*fftn(gT(W_G².*fgrad(x, 1), 1))` on the cone's voxels, for source `s`
    fn cone_fft_reg(&self, ws: &mut Workspace, s: usize, x: &[f64], out: &mut [Complex64]) {
        let dims = self.dims;
        let len = [dims.0, dims.1, dims.2];
        let stride = [1, dims.0, dims.0 * dims.1];
        // W_G².*fgrad(x, 1) along `c` at voxel p (index l)
        let gw = |c: usize, p: [usize; 3], l: usize| -> f64 {
            let g = if p[c] + 1 < len[c] { x[l + stride[c]] - x[l] } else { 0.0 };
            g * self.wg2(s, c, p, l)
        };
        // gT: per axis −g(1), g(i−1) − g(i), g(n−1), summed over x, y, z
        par_rows(&mut ws.c, dims, |j, k, l0, row| {
            for (i, v) in row.iter_mut().enumerate() {
                let (p, l) = ([i, j, k], l0 + i);
                let mut acc = 0.0;
                for c in 0..3 {
                    let (pos, st) = (p[c], stride[c]);
                    let mut q = p;
                    q[c] = pos.wrapping_sub(1);
                    let t = if pos == 0 {
                        -gw(c, p, l)
                    } else if pos == len[c] - 1 {
                        gw(c, q, l - st)
                    } else {
                        gw(c, q, l - st) - gw(c, p, l)
                    };
                    acc = if c == 0 { t } else { acc + t };
                }
                *v = Complex64::new(acc, 0.0);
            }
        });
        ws.fft.fft3d(&mut ws.c);
        let (idx, c) = (&self.cone.idx, &ws.c);
        par_fill(out, |j| c[idx[j] as usize]);
    }

    /// Step 2 on step 1's `[x+; x−]`, in place: each source minus its streaking component, masked.
    fn step2(&self, ws: &mut Workspace, x: &mut [f64], params: &ChiSepIlsqrParams) {
        let n = self.n;
        let mut t = vec![0.0; n];
        for (s, xs) in x.chunks_mut(n).enumerate() {
            let mut b = vec![Complex64::new(0.0, 0.0); self.cone.len()];
            self.cone_fft_reg(ws, s, xs, &mut b);
            let y = pcg(
                |v: &[Complex64], o: &mut [Complex64]| {
                    self.cone_ifft(ws, v, &mut t);
                    self.cone_fft_reg(ws, s, &t, o);
                },
                &b,
                params.cg_tol_sa,
                params.cg_max_iter_sa,
                Start::Zero,
                n,
                Some(&self.cone),
            );
            drop(b);
            self.cone_ifft(ws, &y.x, &mut t);
            let m = &self.m;
            par_update(xs, |i, v| *v = (*v - t[i] * f64::from(m[i])) * f64::from(m[i]));
        }
    }

    /// The fast QSM of step 2 (see the module docs), on the padded grid.
    fn fast_qsm(&self, ws: &mut Workspace, f: Vec<f64>, vs: [f64; 3]) -> Vec<f64> {
        let n = self.n;
        let dims = self.dims;
        let d = &self.d;
        let m = &self.m;
        // k-space blending weight from |D|^0.001
        let mut w = vec![0.0; n];
        par_rows(&mut w, dims, |j, k, _, row| {
            for (i, v) in row.iter_mut().enumerate() {
                *v = d.at(i, j, k).abs().powf(0.001);
            }
        });
        let (a, b) = {
            let mut t = w.clone();
            (prctile(&mut t, 1.0), prctile(&mut t, 30.0))
        };
        par_update(&mut w, |_, v| *v = clamp01((*v - a) / (b - a)));
        // TKD reference: fftn(f)./Dt, |Dt| ≥ 1/8 with sign(0) = 1
        let mut fk: Vec<Complex64> = f.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        drop(f);
        ws.fft.fft3d(&mut fk);
        {
            let fk = &fk;
            par_rows(&mut ws.c, dims, |j, k, l0, row| {
                for (i, v) in row.iter_mut().enumerate() {
                    let di = d.at(i, j, k);
                    let s = if di < 0.0 { -1.0 } else { 1.0 };
                    let dt = if di.abs() < TKD_THRESHOLD { TKD_THRESHOLD * s } else { di };
                    *v = fk[l0 + i] / dt;
                }
            });
        }
        ws.fft.ifft3d(&mut ws.c);
        let mut x_tkd = vec![0.0; n];
        {
            let c = &ws.c;
            par_fill(&mut x_tkd, |i| c[i].re * f64::from(m[i]));
        }
        // the ball's k-space smoothing: lpf(z) = ifftn(fftn(z).*fftn(fftshift(S)))
        let kk = {
            let mut ball = pv_ball(dims, BALL_RADIUS_MM, vs);
            circshift(&mut ball, dims, true);
            let mut kk: Vec<Complex64> = ball.iter().map(|&v| Complex64::new(v, 0.0)).collect();
            drop(ball);
            ws.fft.fft3d(&mut kk);
            kk
        };
        // lpf(z) into `out`
        let lpf = |fft: &mut Fft3dWorkspace, z: &[Complex64], out: &mut [Complex64], kk: &[Complex64]| {
            out.copy_from_slice(z);
            fft.fft3d(out);
            par_update(out, |i, v| *v *= kk[i]);
            fft.ifft3d(out);
        };
        // k1c = ifftshift(fftn(f).*sign(D)); k2 = fftshift(W.*k1c + (1 − W).*lpf(k1c))
        par_rows(&mut fk, dims, |j, k, _, row| {
            for (i, v) in row.iter_mut().enumerate() {
                *v *= sign0(d.at(i, j, k));
            }
        });
        circshift(&mut fk, dims, false);
        let k1c = fk;
        lpf(&mut ws.fft, &k1c, &mut ws.c, &kk);
        par_update(&mut ws.c, |i, v| *v = k1c[i] * w[i] + *v * (1.0 - w[i]));
        let mut l2 = k1c;
        circshift(&mut ws.c, dims, true);
        ws.fft.ifft3d(&mut ws.c);
        // km = fftn(x2.*m); x3 = real(ifftn(W.*km + (1 − W).*lpf(km))).*m
        par_update(&mut ws.c, |i, v| *v = Complex64::new(v.re * f64::from(m[i]), 0.0));
        ws.fft.fft3d(&mut ws.c);
        lpf(&mut ws.fft, &ws.c, &mut l2, &kk);
        drop(kk);
        {
            let km = &ws.c;
            par_update(&mut l2, |i, v| *v = km[i] * w[i] + *v * (1.0 - w[i]));
        }
        drop(w);
        ws.fft.ifft3d(&mut l2);
        let mut x3 = vec![0.0; n];
        par_fill(&mut x3, |i| l2[i].re * f64::from(m[i]));
        drop(l2);
        // [ones x3] \ x_tkd over the whole grid
        let nf = n as f64;
        let mx = det_sum(n, |i| x3[i]) / nf;
        let my = det_sum(n, |i| x_tkd[i]) / nf;
        let sxy = det_sum(n, |i| (x3[i] - mx) * (x_tkd[i] - my));
        let sxx = det_sum(n, |i| (x3[i] - mx) * (x3[i] - mx));
        let c2 = sxy / sxx;
        let c1 = my - c2 * mx;
        par_update(&mut x3, |_, v| *v = c2 * *v + c1);
        x3
    }

    /// R2′ (ppm) smoothed by a Tukey window in k-space:
    /// `real(ifftn(fftshift(ifftshift(fftn(r)).*w)))`
    fn tukey_lpf(&self, ws: &mut Workspace, r: &[f64]) -> Vec<f64> {
        let (nx, ny, nz) = self.dims;
        let (wx, wy, wz) = (tukeywin(nx, TUKEY_RATIO), tukeywin(ny, TUKEY_RATIO), tukeywin(nz, TUKEY_RATIO));
        par_fill(&mut ws.c, |i| Complex64::new(r[i], 0.0));
        ws.fft.fft3d(&mut ws.c);
        circshift(&mut ws.c, self.dims, false);
        par_rows(&mut ws.c, self.dims, |j, k, _, row| {
            for (i, v) in row.iter_mut().enumerate() {
                *v *= (wx[i] * wy[j]) * wz[k];
            }
        });
        circshift(&mut ws.c, self.dims, true);
        ws.fft.ifft3d(&mut ws.c);
        let c = &ws.c;
        let mut out = vec![0.0; nx * ny * nz];
        par_fill(&mut out, |i| c[i].re);
        out
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

/// `(lo, hi)`, the `prctile`s `p` of `v(l)` over the mask's voxels `l`, for the weights
/// `min(max((hi − v)/(hi − lo), 0), 1)`
fn prc_on(inside: &Support, v: impl Fn(usize) -> f64, p: (f64, f64)) -> (f64, f64) {
    let mut sel: Vec<f64> = inside.idx.iter().map(|&l| v(l as usize)).collect();
    let lo = prctile(&mut sel, p.0);
    let hi = prctile(&mut sel, p.1);
    (lo, hi)
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

/// `fftshift` (`forward`, shift by `floor(n/2)`) or `ifftshift` (shift by `ceil(n/2)`) per axis,
/// in place: a rotation of each x-row, of each z-plane by whole rows, and of the volume by whole
/// planes
fn circshift<T: Copy + Send>(a: &mut [T], (nx, ny, nz): (usize, usize, usize), forward: bool) {
    let s = |n: usize| if forward { n / 2 } else { n.div_ceil(2) };
    let (sx, sy, sz) = (s(nx), s(ny), s(nz));
    par_rows(a, (nx, ny, nz), |_, _, _, row| row.rotate_right(sx));
    #[cfg(feature = "parallel")]
    a.par_chunks_mut(nx * ny).for_each(|p| p.rotate_right(sy * nx));
    #[cfg(not(feature = "parallel"))]
    a.chunks_mut(nx * ny).for_each(|p| p.rotate_right(sy * nx));
    a.rotate_right(sz * nx * ny);
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
/// voxel): per axis `−g(1)`, `g(i−1) − g(i)`, `g(n−1)`, summed over x, y, z. The reference for
/// the fused version in `Problem::cone_fft_reg`.
#[cfg(test)]
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
    chunked_sum(n.div_ceil(CHUNK), |c| c * CHUNK..((c + 1) * CHUNK).min(n), f)
}

/// `f` summed over each of `chunks` ranges in order, then the partial sums in order
fn chunked_sum<T>(chunks: usize, range: impl Fn(usize) -> std::ops::Range<usize> + Sync, f: impl Fn(usize) -> T + Sync) -> T
where
    T: Copy + Send + Sync + std::ops::Add<Output = T> + Default,
{
    let part = |c: usize| {
        let mut acc = T::default();
        for i in range(c) {
            acc = acc + f(i);
        }
        acc
    };
    #[cfg(feature = "parallel")]
    let parts: Vec<T> = (0..chunks).into_par_iter().map(part).collect();
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

/// Where [`pcg`] starts.
enum Start<'a, T> {
    Zero,
    Owned(Vec<T>),
    /// A start that writes itself into a buffer (every element), so `pcg` keeps no copy of it.
    Fill(&'a (dyn Fn(&mut [T]) + 'a)),
}

/// Where [`pcg`]'s best iterate so far (MATLAB's `xmin`) is: the start (rewritten by
/// [`Start::Fill`] when needed), the current iterate, or its own copy.
#[derive(Clone, Copy, PartialEq)]
enum XMin {
    Start,
    X,
    Saved,
}

/// `det_sum` of `f` over a vector of `n`, or over its voxels in `on` (the rest zero)
fn psum<T>(on: Option<&Support>, n: usize, f: impl Fn(usize) -> T + Sync) -> T
where
    T: Copy + Send + Sync + std::ops::Add<Output = T> + Default,
{
    match on {
        Some(s) => s.sum(f),
        None => det_sum(n, f),
    }
}

/// MATLAB R2023b's `pcg(A, b, tol, maxit, [], [], x0)` (no preconditioner), line for line.
///
/// The vectors have `n_logical` elements; with `on`, only those in that set are stored (the
/// others are zero in every vector of the solve: `A` maps such vectors to such vectors and `b`
/// is one), and the norms and inner products add the same terms as over the whole vector.
/// MATLAB's `xmin` is copied only when it stops being the current iterate (`XMin`), so the
/// result is the same with one vector less while they coincide; and a returned `iter` of an
/// `xmin` that is the last iterate is `imin` (the two candidates are then the same vector).
fn pcg<T: PcgElem>(
    mut a: impl FnMut(&[T], &mut [T]),
    b: &[T],
    tol: f64,
    maxit: usize,
    start: Start<T>,
    n_logical: usize,
    on: Option<&Support>,
) -> PcgOut<T> {
    let n = b.len();
    let norm = |v: &[T]| psum(on, n, |i| v[i].abs2()).sqrt();
    let dot = |u: &[T], v: &[T]| psum(on, n, |i| u[i].cmul(v[i]));
    let n2b = norm(b);
    if n2b == 0.0 {
        return PcgOut { x: vec![T::default(); n], flag: 0, iter: 0 };
    }
    let (mut x, mut xmin, fill) = match start {
        Start::Zero => (vec![T::default(); n], XMin::X, None),
        Start::Owned(v) => (v, XMin::X, None),
        Start::Fill(f) => {
            let mut x = vec![T::default(); n];
            f(&mut x);
            (x, XMin::Start, Some(f))
        }
    };
    let mut saved: Vec<T> = Vec::new();
    let mut flag = 1u8;
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
    let maxmsteps = ((n_logical / 50) as i64).min(5).min(n_logical as i64 - maxit as i64);
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
        if xmin == XMin::X {
            if saved.is_empty() {
                saved = x.clone();
            } else {
                saved.copy_from_slice(&x);
            }
            xmin = XMin::Saved;
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
            xmin = XMin::X;
            imin = it;
        }
        if stag >= maxstagsteps {
            flag = 3;
            break;
        }
    }
    if flag != 0 {
        if xmin == XMin::X {
            iter = imin;
        } else {
            // r_comp = norm(b − A·xmin); `p` is free now and holds a rebuilt start
            if xmin == XMin::Start {
                (fill.expect("a start to rebuild"))(&mut p);
            }
            let xm: &[T] = if xmin == XMin::Saved { &saved } else { &p };
            a(xm, &mut q);
            let r_comp = psum(on, n, |i| (b[i] - q[i]).abs2()).sqrt();
            if r_comp <= normr_act {
                x = if xmin == XMin::Saved { saved } else { p };
                iter = imin;
            } else {
                iter = ii;
            }
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
        let mut b = a.clone();
        circshift(&mut b, (7, 4, 3), false);
        circshift(&mut b, (7, 4, 3), true);
        assert_eq!(b, a);
        let mut b = vec![0, 1, 2, 3, 4];
        circshift(&mut b, (5, 1, 1), true);
        assert_eq!(b, vec![3, 4, 0, 1, 2]);
        // the in-place shift moves each element where the out-of-place one does
        for dims in [(7, 4, 3), (6, 5, 4), (1, 3, 2), (5, 1, 6)] {
            for forward in [true, false] {
                let a: Vec<usize> = (0..dims.0 * dims.1 * dims.2).collect();
                let mut b = a.clone();
                circshift(&mut b, dims, forward);
                assert_eq!(b, circshift_ref(&a, dims, forward), "{dims:?} {forward}");
            }
        }
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
        let out = pcg(apply, &b, 1e-12, 10, Start::Zero, 3, None);
        let mut ax = [0.0; 3];
        apply(&out.x, &mut ax);
        assert!(ax.iter().zip(&b).all(|(a, b)| (a - b).abs() < 1e-10), "{:?}", out.x);
        assert_eq!(out.flag, 0);
        // complex unknown with an R-linear operator: Re-part projection keeps CG well defined
        let bc = [Complex64::new(1.0, 0.5), Complex64::new(-1.0, 0.0)];
        let out = pcg(|x: &[Complex64], o: &mut [Complex64]| { o[0] = x[0] * 2.0; o[1] = x[1] * 3.0; }, &bc, 1e-12, 5, Start::Zero, 2, None);
        assert!((out.x[0] - Complex64::new(0.5, 0.25)).norm() < 1e-14 && (out.x[1] - Complex64::new(-1.0 / 3.0, 0.0)).norm() < 1e-14);
    }

    /// The out-of-place `fftshift`/`ifftshift` that [`circshift`] replaced
    fn circshift_ref<T: Copy>(a: &[T], (nx, ny, nz): (usize, usize, usize), forward: bool) -> Vec<T> {
        let s = |n: usize| if forward { n / 2 } else { n.div_ceil(2) };
        let (sx, sy, sz) = (s(nx), s(ny), s(nz));
        let mut out = a.to_vec();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    out[(i + sx) % nx + nx * ((j + sy) % ny + ny * ((k + sz) % nz))] = a[i + nx * (j + ny * k)];
                }
            }
        }
        out
    }

    #[test]
    fn kernel_is_the_toolbox_kernel_bit_for_bit() {
        let h = (0.1f64, -0.4, 0.9);
        let norm = (h.0 * h.0 + h.1 * h.1 + h.2 * h.2).sqrt();
        for (dims, vs, b0) in [((8, 6, 4), [1.0; 3], (0.0, 0.0, 1.0)), ((7, 5, 9), [0.8, 0.9, 1.5], (h.0 / norm, h.1 / norm, h.2 / norm))] {
            let want = super::super::chi_sep_medi::toolbox_dipole_kernel(dims, vs, b0);
            let kernel = Kernel::new(dims, vs, b0);
            for l in 0..want.len() {
                let (i, j, k) = (l % dims.0, (l / dims.0) % dims.1, l / (dims.0 * dims.1));
                assert_eq!(kernel.at(i, j, k).to_bits(), want[l].to_bits(), "{dims:?} at {l}");
            }
        }
    }

    /// A vector on a support (zero elsewhere): sums over it equal the full vector's bit for bit,
    /// including a chunk with no voxels of the support.
    #[test]
    fn sums_on_a_support_are_the_full_sums() {
        let n = 3 * CHUNK + 1234;
        let keep = |l: usize| (l / CHUNK != 1) && (l.is_multiple_of(7) || l % 11 == 3);
        let sup = Support::new(n, keep);
        let full: Vec<f64> = (0..n).map(|l| if keep(l) { ((l as f64) * 0.618).sin() * 1e3 } else { 0.0 }).collect();
        let on: Vec<f64> = sup.idx.iter().map(|&l| full[l as usize]).collect();
        assert_eq!(sup.bounds[1], sup.bounds[2]);
        assert_eq!(det_sum(n, |i| full[i] * full[i]).to_bits(), sup.sum(|j| on[j] * on[j]).to_bits());
        assert_eq!(det_sum(n, |i| full[i]).to_bits(), sup.sum(|j| on[j]).to_bits());
        let mut seen = 0;
        for c in 0..n.div_ceil(CHUNK) {
            sup.walk(c, n, |l, j| {
                assert_eq!(j.is_some(), keep(l));
                if let Some(j) = j {
                    assert_eq!(sup.idx[j] as usize, l);
                }
                seen += 1;
            });
        }
        assert_eq!(seen, n);
    }

    /// `pcg` on a support and from a rebuilt start returns what it returns on the whole vector
    /// from a stored start, bit for bit.
    #[test]
    fn pcg_support_and_start_do_not_change_the_result() {
        let n = 2 * CHUNK + 77;
        let keep = |l: usize| l % 5 != 1;
        let sup = Support::new(n, keep);
        let ns = sup.len();
        // an SPD operator coupling the kept voxels in order: 3 on the diagonal, 0.5 to the neighbours
        let op_on = |x: &[f64], o: &mut [f64]| {
            for j in 0..x.len() {
                let prev = if j > 0 { x[j - 1] } else { 0.0 };
                let next = if j + 1 < x.len() { x[j + 1] } else { 0.0 };
                o[j] = 3.0 * x[j] + 0.5 * prev + 0.5 * next;
            }
        };
        let gather = |x: &[f64]| -> Vec<f64> { sup.idx.iter().map(|&l| x[l as usize]).collect() };
        let scatter = |v: &[f64], x: &mut [f64]| {
            x.fill(0.0);
            sup.idx.iter().zip(v).for_each(|(&l, &v)| x[l as usize] = v);
        };
        let op_full = |x: &[f64], o: &mut [f64]| {
            let mut t = vec![0.0; ns];
            op_on(&gather(x), &mut t);
            scatter(&t, o);
        };
        let b_on: Vec<f64> = (0..ns).map(|j| ((j as f64) * 0.37).cos()).collect();
        let mut b = vec![0.0; n];
        scatter(&b_on, &mut b);
        let x0_on: Vec<f64> = (0..ns).map(|j| 0.1 * ((j as f64) * 1.3).sin()).collect();
        let mut x0 = vec![0.0; n];
        scatter(&x0_on, &mut x0);
        for maxit in [1, 3, 30] {
            let full = pcg(op_full, &b, 1e-10, maxit, Start::Owned(x0.clone()), n, None).x;
            let fill = |x: &mut [f64]| scatter(&x0_on, x);
            let rebuilt = pcg(op_full, &b, 1e-10, maxit, Start::Fill(&fill), n, None).x;
            let on = pcg(op_on, &b_on, 1e-10, maxit, Start::Owned(x0_on.clone()), n, Some(&sup)).x;
            assert!(full.iter().zip(&rebuilt).all(|(a, b)| a.to_bits() == b.to_bits()), "maxit {maxit}");
            assert!(gather(&full).iter().zip(&on).all(|(a, b)| a.to_bits() == b.to_bits()), "maxit {maxit}");
        }
    }

    /// A small phantom: an ellipsoid with a paramagnetic and a diamagnetic half.
    fn small_phantom() -> (Grid, Vec<f64>, Vec<f64>, Vec<f64>, Vec<u8>) {
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
        (grid, field, r2p, chi, mask)
    }

    /// Step 2's gradient weights, recomputed where used, and its regulariser, with the gradient
    /// fused into the adjoint, equal the full-volume computation bit for bit.
    #[test]
    fn fused_regulariser_is_the_full_volume_one() {
        use crate::inversion::medi::fgrad;
        let (grid, field, r2p, chi, mask) = small_phantom();
        let inputs = ChiSepIlsqrInputs { local_field: &field, r2prime: &r2p, qsm: &chi, mask: &mask };
        let params = ChiSepIlsqrParams { pad: [4; 3], ..Default::default() };
        let (pb, mut ws, _) = Problem::new(&inputs, &grid, (0.0, 0.0, 1.0), &params);
        let (dims, n) = (pb.dims, pb.n);
        let at = |l: usize| [l % dims.0, (l / dims.0) % dims.1, l / (dims.0 * dims.1)];
        let full = |b: &BoxVol| (0..n).map(|l| b.get(at(l))).collect::<Vec<f64>>();
        let (xm, rm) = (full(&pb.xm), full(&pb.rm));
        // the weights as computed before: fgrad of the volumes, ramp over the mask, (W.*M).^2
        let mut gx = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
        let mut gr = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
        fgrad(&xm, dims, pb.vs, &mut gx);
        fgrad(&rm, dims, pb.vs, &mut gr);
        let (drp, drn) = (pb.drp, pb.drn);
        let inside: Vec<usize> = (0..n).filter(|&l| pb.m[l] != 0).collect();
        let mut wg2 = [[vec![], vec![], vec![]], [vec![], vec![], vec![]]];
        for c in 0..3 {
            let gp: Vec<f64> = (0..n).map(|i| ((gr[c][i] + drn * gx[c][i]) / (drp + drn)).abs()).collect();
            let gn: Vec<f64> = (0..n).map(|i| ((gr[c][i] - drp * gx[c][i]) / (drp + drn)).abs()).collect();
            for (s, g) in [gp, gn].into_iter().enumerate() {
                let mut sel: Vec<f64> = inside.iter().map(|&i| g[i]).collect();
                let (lo, hi) = (prctile(&mut sel, params.prc_w_g.0), prctile(&mut sel, params.prc_w_g.1));
                wg2[s][c] = (0..n).map(|i| clamp01((hi - g[i]) / (hi - lo)) * f64::from(pb.m[i])).map(|v| v * v).collect();
                for l in 0..n {
                    assert_eq!(pb.wg2(s, c, at(l), l).to_bits(), wg2[s][c][l].to_bits(), "W_G² {s} {c} at {l}");
                }
            }
        }
        let x: Vec<f64> = (0..n).map(|l| ((l as f64) * 0.71).sin() * 0.01).collect();
        for s in 0..2 {
            let mut g = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
            fgrad(&x, dims, [1.0; 3], &mut g);
            for c in 0..3 {
                g[c].iter_mut().zip(&wg2[s][c]).for_each(|(v, w)| *v *= w);
            }
            let mut t = vec![0.0; n];
            grad_adjoint_unit(&g, dims, &mut t);
            let mut want: Vec<Complex64> = t.iter().map(|&v| Complex64::new(v, 0.0)).collect();
            ws.fft.fft3d(&mut want);
            let mut got = vec![Complex64::new(0.0, 0.0); pb.cone.len()];
            pb.cone_fft_reg(&mut ws, s, &x, &mut got);
            for (j, &l) in pb.cone.idx.iter().enumerate() {
                let w = want[l as usize];
                assert_eq!((got[j].re.to_bits(), got[j].im.to_bits()), (w.re.to_bits(), w.im.to_bits()), "source {s} at {l}");
            }
        }
    }

    #[test]
    fn sources_have_their_signs() {
        let (grid, field, r2p, chi, mask) = small_phantom();
        let nt = mask.len();
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

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
//! As in the authors' `DECOMPOSE_analysis.m`, the *phase* is synthesized from a QSM of
//! **each echo** (theirs: STAR-QSM per echo, `QSM_multi`): the complex data per echo is
//! `y(TE) = |mag_norm(TE)| · exp(−i·(2/3)·χ(TE)·γ·B0·TE)` (magnitude normalised by its global
//! maximum). [`decompose`] takes that per-echo QSM; producing it is the caller's. A voxel whose
//! `Σ y` is zero is skipped, as theirs is.
//!
//! The per-voxel fit is a 3-stage alternating bounded least-squares (`lsqcurvefit`), repeated
//! `n_inner` times: (1) amplitudes `[C+,C−,C₀]` against the linear signal, then
//! (2) `R2*₀` and (3) `[χ+,χ−]` against `log(signal)` (complex log). The residual
//! is the complex difference packed as `[Re; −Im]`, matching the reference.
//!
//! Each source is reconstructed from the fitted parameters via a per-compartment
//! phase accumulation, `−Σ angle(model) / ((2/3)·γ·B0·ΣTE)`: the paramagnetic
//! sub-model (`pscModel`) gives χ+ and the diamagnetic sub-model (`dscModel`)
//! gives |χ−|. χ+ is returned ≥ 0, χ− ≤ 0.
//!
//! **Phase unwrapping** (off by default: [`DecomposeParams::unwrap_phase`]). The authors' code
//! takes every phase on its principal branch (MATLAB's `log` and `angle`), and so does this
//! port by default. With the option on: every phase the fit and the reconstruction see arrives
//! through `atan2`, on `(−π, π]`. A compartment accumulates `(2/3)·χ·γ·B0·TE`
//! radians, so above `|χ_total| = π / ((2/3)·γ·B0·TE_max)` the late echoes fold
//! back onto that branch — 0.09 ppm at 7 T with a 28 ms last echo, which is
//! ordinary globus pallidus. Left folded, the alternation converges to a
//! degenerate fixed point (χ− pinned at `chi_bound`, R2*₀ and the amplitudes at
//! zero) and returns χ+ = 0 exactly, so the most paramagnetic structure in the
//! volume reads as a hole. Both the stage-2/3 residuals and `recon_phase`
//! therefore walk the echoes in order and keep each phase on the branch nearest
//! its predecessor ([`unwrap_near`]); the synthesized data phase is known in
//! closed form and is used unwrapped rather than round-tripped through `atan2`.
//! The original method, published at 3 T, does not meet this limit in practice —
//! the threshold there is ~0.2 ppm.
//!
//! Unwrapping is only unambiguous while the phase step *between* echoes stays
//! under π, so the χ search is additionally capped at
//! `π / ((2/3)·γ·B0·ΔTE_max)` — 0.31 ppm at 7 T with 8 ms spacing, inside the
//! 0.5 ppm default [`DecomposeParams::chi_bound`]. Past that the model matches a
//! given χ on more than one branch, the stage-3 landscape grows spurious minima,
//! and scattered voxels fall into them; capping the search is what removes the
//! speckle of dropouts that unwrapping alone left in the recovered pallidum.
//!
//! Note that `−Σ angle / den` remains a *compressive* estimator of χ: its gain
//! rises with χ itself (≈0.03 at white-matter χ+, ≈0.87 in the pallidum on the
//! QSM-CI phantom), so it flattens contrast between weak sources. That is
//! inherent to the reference's reconstruction, not to this port — the fitted
//! compartment amplitudes set the gain, and they are only weakly identifiable
//! from data that carries no sub-voxel compartment signature.
//!
//! **Output mapping.** The port fits what the authors fit: their `ydata =
//! squeeze(y_data(i,j,k,:))'` is MATLAB's conjugate transpose, so the fit is to conj(y), in their
//! parameter order and with their bounds and initial values. Conjugating the data mirrors the fit
//! (`conj(S(C+, C−, χ+, χ−)) = S(C−, C+, −χ−, −χ+)`), which is why their script writes DSC to its
//! paramagnetic file (`XPSC`) and PSC to the diamagnetic one; the port writes χ+ = |DSC| and
//! χ− = −|PSC| the same way. (Fitting y itself and writing PSC as χ+ gives the same maps
//! mathematically, as earlier versions of this port did, but not the same rounding, and the
//! solver's path through a near-degenerate voxel follows rounding.)
//!
//! **Solver.** Every stage is MATLAB `lsqcurvefit` with its default options, which the authors
//! use: the trust-region-reflective method, ported in [`crate::solvers::lsq_trr`] and verified
//! against MATLAB R2023b call by call on this fit (its module docs, *Matching MATLAB*: 92.6 % of
//! the fixture's 11340 stage calls take MATLAB's exact path, all stage-2 calls among them). The
//! solver of each stage is selectable ([`DecomposeParams::solvers`]), for the paper's choice.
//! Before that this port used a bounded Levenberg–Marquardt, which left χ± 2.9–3.3 % (NRMSE) from
//! the authors' code on the KKI crop below and 11.1 % / 9.0 % on the full chisep-mc phantom
//! (astewartau/QSM.rs#329).
//!
//! # Agreement with the authors' code
//!
//! Against the QSM-CI `decompose-qsm` container (the authors' `utils/` and fit loop compiled
//! from licensed MATLAB R2023b, verified identical to it, QSMxT/QSM-CI#309), given the same
//! per-echo QSM:
//!
//! | port | data | χ+ NRMSE (r) | χ− NRMSE (r) |
//! |---|---|---|---|
//! | unwrapping and χ cap on, LM 30 iterations, stop at 1e-10 | KKI 24×24×16 crop (3 T, 5 echoes) | 92.0 % (0.587) | 34.9 % (0.933) |
//! | principal-branch phases, LM 30 iterations | KKI crop | 8.2 % (0.9956) | 10.1 % (0.9951) |
//! | LM to 400 iterations, stop at 1e-12 | KKI crop | 2.9 % (0.9994) | 3.3 % (0.9994) |
//! | LM to 400 iterations, stop at 1e-12 | `tests/data/decompose_qsm_small` (378 voxels) | 1.1 % (0.99985) | 11.6 % (0.9915) |
//! | **`lsqcurvefit` trust-region-reflective** (default) | `tests/data/decompose_qsm_small` | **0.24 %** (0.99999) | **0.90 %** (0.99995) |
//!
//! What remains is concentrated in voxels where one compartment collapses (an amplitude or χ
//! within rounding of 0): there the Jacobian column of the vanished parameter is rounding noise,
//! and MATLAB's and Rust's elementary functions round differently (see `lsq_trr`).
//!
//! # Threads
//!
//! Voxels are fitted in parallel (rayon, feature `parallel`), and the result is bit-identical at
//! any thread count. The fit allocates nothing per voxel (see the loop). On a 6-core i5-11500,
//! 12000 synthetic 6-echo voxels, static musl build: 1.37 s on 1 thread, 0.70 s on 2, 0.37 s on
//! 4, 0.26 s on 6, 0.23 s on 12. (With bounded LM to 400 iterations: 28.0 s, 8.0 s, 6.4 s on 1,
//! 4, 6 threads; `lsqcurvefit`'s 1e-6 tolerances end each stage far sooner than LM's 1e-12.)
//!
//! # Reference pipeline
//!
//! Chen et al. 2021 (NeuroImage 242:118735, §3 "Material and methods"; PMC8720043):
//!
//! - **Data:** in vivo at 3 T (GE MR750w), 0.86 × 0.86 × 1.0 mm, 16 echoes,
//!   TE1/ΔTE/TE16 = 2.7/2.9/46.2 ms. The phantom was 1 mm iso, 16 echoes 1.47–25.9 ms; the ex
//!   vivo data were 7 T.
//! - **Mask:** FSL BET. **Unwrapping:** Laplacian. **Background removal:** V-SHARP.
//! - **QSM input:** STAR-QSM computed **for each echo**. The signal model (Eq. 4) is
//!   `y(t) = M(t)·exp(−i·(2/3)·χ(t)·γ·B0·t)` with the per-echo χ(t), not one echo-combined χ.
//!   [`decompose`] takes that per-echo χ. (The authors' script runs STAR-QSM per echo at
//!   `padsize` 12 after `MRPhaseUnwrap` and `V_SHARP(smvsize 25)` of that echo.)
//! - **Referencing:** χ referenced to the whole-brain mean.
//! - **Constants:** `0.5 > |χ+|, |χ−| > 0` (§2.2, [`DecomposeParams::chi_bound`]), and
//!   `a = 323.5 Hz/ppm` at 3 T (§2.1), which the formula above reproduces and which scales with
//!   B0 by construction.
//!
//! [`DecomposeParams`]'s `b0` defaults to 7 T. The pipeline dispatcher overrides it from the scan
//! metadata.
//!
//! Reference:
//! Chen, J., et al. (2021). "Decompose quantitative susceptibility mapping (QSM)
//! to sub-voxel diamagnetic and paramagnetic components based on gradient-echo MRI
//! data." NeuroImage 242:118735. https://doi.org/10.1016/j.neuroimage.2021.118735
//! Reference implementation (QSM-CI port of Tim Ho's open MATLAB DECOMPOSE-QSM).
//!
//! # Pipeline rules
//!
//! The facts above that a configuration can depart from are published as rules in
//! [`crate::pipeline::rules::RULES`] (each with its strength, basis, reason and source): `decompose.mask-bet`, `decompose.unwrap-laplacian`, `decompose.bg-vsharp`, `decompose.qsm-star`, `decompose.reference-mean`.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use std::f64::consts::PI;

use crate::error::{check_len, InputError};
use crate::solvers::lsq_trr::{lsq_trust_region_reflective, LsqTrrOptions, LsqTrrWorkspace};

/// Gyromagnetic constant used by the DECOMPOSE reference: `42.58 · 2π` — the reference
/// implementation's value (`signalModel.m`), not `crate::constants`. Keep it.
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
    /// `MaxIterations` of each stage's `lsqcurvefit` (default 400, MATLAB's default, which the
    /// authors keep). Every other solver setting is `lsqcurvefit`'s default
    /// ([`LsqTrrOptions::default`]).
    pub max_iterations: usize,
    /// Unwrap the phase along the echoes and cap the χ search at what the echo spacing can
    /// represent (see the module docs, *Phase unwrapping*). Default `false`: the authors' code
    /// takes every phase on its principal branch (`log`, `angle`) and searches χ up to
    /// [`chi_bound`](Self::chi_bound).
    pub unwrap_phase: bool,
    /// The solver of each of the three stages (see [`DecomposeStageSolvers`]).
    pub solvers: DecomposeStageSolvers,
}

/// Which solver fits each stage of the alternating fit.
///
/// The reference implementation (Tim Ho's MATLAB DECOMPOSE-QSM, which the QSM-CI `decompose-qsm`
/// container runs) calls `lsqcurvefit` with its default trust-region-reflective algorithm for all
/// three stages, and that is the default here. The paper (Chen et al. 2021, §3.1) describes a
/// different choice: Eq. 5.1 (the amplitudes, with `C+ + C− + C₀ = 1`) and Eq. 5.2 (R2*₀) by
/// `lsqlin`'s interior-point method, and only Eq. 5.3 (χ±) by `lsqnonlin` trust-region-reflective
/// with a supplied Jacobian sparsity pattern. Those solvers are not ported (yet); this is where
/// they would be selected.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecomposeStageSolvers {
    /// Stage 1, the amplitudes `[C+, C−, C₀]`.
    pub amplitudes: DecomposeStageSolver,
    /// Stage 2, `R2*₀`.
    pub r2star: DecomposeStageSolver,
    /// Stage 3, `[χ+, χ−]`.
    pub chi: DecomposeStageSolver,
}

impl Default for DecomposeStageSolvers {
    fn default() -> Self {
        let s = DecomposeStageSolver::LsqcurvefitTrustRegionReflective;
        Self { amplitudes: s, r2star: s, chi: s }
    }
}

/// A stage solver for [`decompose`].
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecomposeStageSolver {
    /// MATLAB `lsqcurvefit` with its default options (trust-region-reflective, finite-difference
    /// Jacobian): [`lsq_trust_region_reflective`]. The reference implementation's solver.
    LsqcurvefitTrustRegionReflective,
}

impl DecomposeParams {
    /// Checks `chi_bound`, the upper bound on |χ| in the fit: it must be ≥ 0 (a negative bound
    /// leaves the fit's lower bound above its upper one).
    pub fn validate(&self) -> Result<(), InputError> {
        if self.chi_bound.is_nan() || self.chi_bound < 0.0 {
            return Err(InputError::param("DECOMPOSE chi_bound", self.chi_bound, ">= 0 ppm"));
        }
        Ok(())
    }
}

impl Default for DecomposeParams {
    fn default() -> Self {
        Self {
            b0: 7.0,
            n_inner: 10,
            chi_bound: 0.5,
            max_iterations: 400,
            unwrap_phase: false,
            solvers: DecomposeStageSolvers::default(),
        }
    }
}

/// DECOMPOSE source separation from per-echo QSM and the multi-echo magnitude.
///
/// # Arguments
/// * `chi_per_echo` — QSM of **each echo** in **ppm**, flattened as `(n_voxels, n_echoes)`
///   (echo fastest per voxel, like `magnitude`): the authors' input (`QSM_multi`, a STAR-QSM
///   per echo in `DECOMPOSE_analysis.m`). It sets each echo's phase,
///   `−(2/3)·χ(TE)·γ·B0·TE`.
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
    chi_per_echo: &[f64],
    magnitude: &[f64],
    echo_times: &[f64],
    mask: &[u8],
    params: &DecomposeParams,
    mut progress: impl FnMut(usize, usize),
) -> Result<super::ChiSepMaps, InputError> {
    params.validate()?;
    let n = mask.len();
    let ne = echo_times.len();
    if ne < 3 {
        return Err(InputError::new(format!("DECOMPOSE needs at least 3 echoes, got {ne}")));
    }
    check_len("DECOMPOSE per-echo QSM (n_voxels * n_echoes)", chi_per_echo.len(), n * ne)?;
    check_len("DECOMPOSE magnitude (n_voxels * n_echoes)", magnitude.len(), n * ne)?;

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
    //
    // The fit allocates nothing: every buffer lives in a `Scratch` that rayon hands out per
    // worker job and reuses across voxels. This matters far beyond the usual allocator overhead:
    // the released binaries are static musl builds, whose allocator serialises concurrent small
    // allocations. Before QSM.rs#93 this loop made ~10⁴ allocations per voxel and got *slower*
    // with every thread added (QSMxT v9.16.0 on 3200 voxels: 2.2 s on 1 thread, 6.3 s on 8);
    // allocation-free, it scales with the physical cores (the module docs, *Threads*).
    let results: Vec<(usize, f64, f64)> = maybe_par_map_init!(
        voxels,
        || Scratch::new(ne),
        |ws: &mut Scratch, &i: &usize| {
            // The authors' `ydata = squeeze(y_data(i,j,k,:))'`: MATLAB's `'` is the conjugate
            // transpose, so they fit conj(y), y[e] = mag_norm · exp(−i·(2/3)·χ(TE_e)·γ·B0·TE_e).
            let (mut sum_re, mut sum_im) = (0.0, 0.0);
            for e in 0..ne {
                let m = magnitude[i * ne + e] / gmax;
                let ph = -(2.0 / 3.0) * chi_per_echo[i * ne + e] * GAMMA * b0 * te[e];
                let y = (m * ph.cos(), m * ph.sin());
                ws.yc[e] = (y.0, -y.1);
                sum_re += ws.yc[e].0;
                sum_im += ws.yc[e].1;
                // log(conj y) for stages 2 & 3: MATLAB's `log`, on the principal branch, unless
                // `unwrap_phase`, where the synthesized phase is used as it is (see
                // `unwrap_near`).
                let arg = if params.unwrap_phase { -ph } else { ws.yc[e].1.atan2(ws.yc[e].0) };
                ws.logyc[e] = (ws.yc[e].0.hypot(ws.yc[e].1).ln(), arg);
            }
            // The authors skip a voxel whose `sum(ydata)` is zero.
            if sum_re == 0.0 && sum_im == 0.0 {
                return (i, 0.0, 0.0);
            }

            let p = fit_voxel(te, &ws.yc, &ws.logyc, b0, params, &mut ws.solver);

            // The authors' reconstruction: PSC = −Σangle(pscModel)/den, DSC = −Σangle(dscModel)/den,
            // written as χ+ = |DSC| (`XPSC`) and |χ−| = |PSC| (`XDSC`). Fitting conj(y) mirrors the
            // compartments, so it is DSC that carries the paramagnetic source (module docs,
            // *Output mapping*).
            let unwrap = params.unwrap_phase;
            let dsc = recon_phase(te, den, unwrap, |t| dsc_model(t, &p, b0));
            let psc = recon_phase(te, den, unwrap, |t| psc_model(t, &p, b0));
            (i, dsc.abs(), -psc.abs())
        }
    )
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
    Ok((chi_pos, chi_neg, chi_out))
}

/// One voxel's fitted parameters, in the authors' order and sign convention (fitted to conj(y)).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Fit {
    /// `C_plus`, `C_minus`, `C_0`.
    c: [f64; 3],
    /// `R_star_2_0`.
    r0: f64,
    /// `chi_plus`, `chi_minus`.
    chi: [f64; 2],
}

/// Reusable scratch for one voxel's fit. Every buffer is fully overwritten before it is read, so
/// a `Scratch` shared across voxels cannot carry state between them.
struct Scratch {
    /// conj(y) per echo.
    yc: Vec<(f64, f64)>,
    /// `log(conj(y))` per echo.
    logyc: Vec<(f64, f64)>,
    solver: StageSolvers,
}

impl Scratch {
    fn new(ne: usize) -> Self {
        Self { yc: vec![(0.0, 0.0); ne], logyc: vec![(0.0, 0.0); ne], solver: StageSolvers::new(2 * ne) }
    }
}

/// `lsqcurvefit` workspaces for the three stages (3, 1 and 2 parameters).
struct StageSolvers {
    s1: LsqTrrWorkspace,
    s2: LsqTrrWorkspace,
    s3: LsqTrrWorkspace,
}

impl StageSolvers {
    fn new(m: usize) -> Self {
        Self { s1: LsqTrrWorkspace::new(m, 3), s2: LsqTrrWorkspace::new(m, 1), s3: LsqTrrWorkspace::new(m, 2) }
    }
}

/// The authors' three-stage alternating fit for one voxel (`DECOMPOSE_analysis.m`):
///
/// ```text
/// chi_0 = [0.05 -0.05]; R0 = 25; C0 = [0.3 0.3 0.4];
/// for it = 1:N_inner
///     C0    = lsqcurvefit(objective1 (C),  C0,    TE, complex_to_real(ydata),      [0 0 0], [inf inf inf]);
///     R0    = lsqcurvefit(objective2 (R0), R0,    TE, complex_to_real(log(ydata)), 0,       inf);
///     chi_0 = lsqcurvefit(objective3 (χ),  chi_0, TE, complex_to_real(log(ydata)), [0 -upper], [upper 0]);
/// end
/// ```
///
/// with `lsqcurvefit`'s default options, which [`lsq_trust_region_reflective`] reproduces.
/// `lsqcurvefit`'s residual is `fun(x, TE) − ydata`, packed by `complex_to_real` as `[Re; −Im]`.
fn fit_voxel(
    te: &[f64],
    yc: &[(f64, f64)],
    logyc: &[(f64, f64)],
    b0: f64,
    params: &DecomposeParams,
    ws: &mut StageSolvers,
) -> Fit {
    let ne = te.len();
    let mut p = Fit { c: [0.3, 0.3, 0.4], r0: 25.0, chi: [0.05, -0.05] };

    // Cap the χ search at the largest |χ| the echo spacing can represent without
    // aliasing. Unwrapping puts each echo's phase on the branch nearest the
    // previous one, which is only unambiguous while the step between echoes stays
    // under π; past that the model can match a given χ on more than one branch,
    // the stage-3 landscape grows spurious minima, and scattered voxels fall into
    // them. At 7 T with 8 ms spacing the limit is 0.31 ppm, well inside the 0.5
    // ppm default — which is what left the recovered pallidum speckled with
    // dropouts even after the phases were unwrapped.
    // Only with `unwrap_phase`: the authors search up to `chi_bound` (`upper`).
    let dte = te.windows(2).map(|w| w[1] - w[0]).fold(te[0], f64::max);
    let upper = if params.unwrap_phase {
        params.chi_bound.min(PI / ((2.0 / 3.0) * GAMMA * b0 * dte))
    } else {
        params.chi_bound
    };
    let unwrap = params.unwrap_phase;
    let inf = f64::INFINITY;
    let opts = LsqTrrOptions { max_iterations: params.max_iterations, ..LsqTrrOptions::default() };

    // `complex_to_real(log(model)) − complex_to_real(log(ydata))`, phases optionally unwrapped
    // along the echoes.
    let log_resid = |q: &Fit, out: &mut [f64]| {
        let mut prev = 0.0;
        for e in 0..ne {
            let z = signal_model(te[e], q, b0);
            let (lr, la) = (z.0.hypot(z.1).ln(), z.1.atan2(z.0));
            let ang = if e == 0 || !unwrap { la } else { unwrap_near(la, prev) };
            prev = ang;
            out[e] = lr - logyc[e].0;
            out[ne + e] = -ang - -logyc[e].1;
        }
    };

    for _ in 0..params.n_inner {
        // Stage 1: objective1, the amplitudes [C+, C−, C₀] against the linear signal.
        let q = p;
        let mut x = p.c;
        match params.solvers.amplitudes {
            DecomposeStageSolver::LsqcurvefitTrustRegionReflective => lsq_trust_region_reflective(
            |x: &[f64], out: &mut [f64]| {
                let q = Fit { c: [x[0], x[1], x[2]], ..q };
                for e in 0..ne {
                    let z = signal_model(te[e], &q, b0);
                    out[e] = z.0 - yc[e].0;
                    out[ne + e] = -z.1 - -yc[e].1;
                }
            },
            &mut x,
            &[0.0; 3],
            &[inf; 3],
            &opts,
            &mut ws.s1,
            |_| {},
        ),
        };
        p.c = x;
        // Stage 2: objective2, R2*₀ against log(ydata).
        let q = p;
        let mut x = [p.r0];
        match params.solvers.r2star {
            DecomposeStageSolver::LsqcurvefitTrustRegionReflective => lsq_trust_region_reflective(
            |x: &[f64], out: &mut [f64]| log_resid(&Fit { r0: x[0], ..q }, out),
            &mut x,
            &[0.0],
            &[inf],
            &opts,
            &mut ws.s2,
            |_| {},
        ),
        };
        p.r0 = x[0];
        // Stage 3: objective3, [χ+, χ−] against log(ydata).
        let q = p;
        let mut x = p.chi;
        match params.solvers.chi {
            DecomposeStageSolver::LsqcurvefitTrustRegionReflective => lsq_trust_region_reflective(
            |x: &[f64], out: &mut [f64]| log_resid(&Fit { chi: [x[0], x[1]], ..q }, out),
            &mut x,
            &[0.0, -upper],
            &[upper, 0.0],
            &opts,
            &mut ws.s3,
            |_| {},
        ),
        };
        p.chi = x;
    }
    p
}

/// `C · exp(−(decay + i·freq)·t)` evaluated as MATLAB does it: `exp` of the complex exponent
/// `(−decay·t, −freq·t)` is `e^{re}·(cos im, sin im)`, then scaled by `C`.
#[inline]
fn compartment(amp: f64, decay: f64, freq: f64, t: f64) -> (f64, f64) {
    let e = (-decay * t).exp();
    let im = -freq * t;
    (amp * (e * im.cos()), amp * (e * im.sin()))
}

/// `a = 2π·γ·B0 / (9√3)` (`signalModel.m`).
#[inline]
fn broadening(b0: f64) -> f64 {
    (2.0 * PI * GAMMA * b0) / (9.0 * 3.0_f64.sqrt())
}

/// The authors' `signalModel.m` at echo time `t` (seconds). Returns `(re, im)`.
#[inline]
fn signal_model(t: f64, p: &Fit, b0: f64) -> (f64, f64) {
    let a = broadening(b0);
    let t1 = compartment(p.c[0], a * p.chi[0] + p.r0, (2.0 / 3.0) * p.chi[0] * GAMMA * b0, t);
    let t2 = compartment(p.c[1], -a * p.chi[1] + p.r0, (2.0 / 3.0) * p.chi[1] * GAMMA * b0, t);
    let t3 = p.c[2] * (-p.r0 * t).exp();
    (t1.0 + t2.0 + t3, t1.1 + t2.1)
}

/// The authors' `pscModel.m`: the `C_plus` compartment, the rest at the baseline decay.
#[inline]
fn psc_model(t: f64, p: &Fit, b0: f64) -> (f64, f64) {
    let a = broadening(b0);
    let t1 = compartment(p.c[0], a * p.chi[0] + p.r0, (2.0 / 3.0) * p.chi[0] * GAMMA * b0, t);
    let t3 = (p.c[2] + p.c[1]) * (-p.r0 * t).exp();
    (t1.0 + t3, t1.1)
}

/// The authors' `dscModel.m`: the `C_minus` compartment with its frequency negated
/// (`(2/3)·(−χ−)·γ·B0`), the rest at the baseline decay.
#[inline]
fn dsc_model(t: f64, p: &Fit, b0: f64) -> (f64, f64) {
    let a = broadening(b0);
    let t2 = compartment(p.c[1], -a * p.chi[1] + p.r0, (2.0 / 3.0) * -p.chi[1] * GAMMA * b0, t);
    let t3 = (p.c[2] + p.c[0]) * (-p.r0 * t).exp();
    (t2.0 + t3, t2.1)
}

/// `2π`, the period of the phase branch cut.
const TAU: f64 = 2.0 * PI;

/// Put `ang` on the 2π-branch nearest `prev` (only with [`DecomposeParams::unwrap_phase`]).
///
/// Every phase here reaches us through `atan2`, which folds onto `(−π, π]`. That
/// fold is not harmless: a compartment accumulates `(2/3)·χ·γ·B0·TE` radians, so
/// at `|χ_total| > π / ((2/3)·γ·B0·TE_max)` — 0.09 ppm at 7 T with a 28 ms last
/// echo, i.e. inside the globus pallidus — the late echoes fold back and the
/// alternating fit chases the fold into a degenerate fixed point (χ− pinned at
/// its bound, R2*₀ and the amplitudes driven to zero, χ+ stuck at 0). Walking the
/// echoes in order and keeping each phase on the branch nearest its predecessor
/// restores the monotone ramp the model assumes.
///
/// The first echo is left on its principal branch: it anchors the series, and at
/// any sane `TE₁` its phase is well inside `(−π, π]` for `|χ| ≤ chi_bound`.
#[inline]
fn unwrap_near(ang: f64, prev: f64) -> f64 {
    ang + ((prev - ang) / TAU).round() * TAU
}

/// Reconstruct a source value: `−Σ angle(model(TE)) / den`, with (if `unwrap`) the phase
/// unwrapped along the echo axis so a strong source is not folded back to zero
/// (see [`unwrap_near`]).
fn recon_phase(te: &[f64], den: f64, unwrap: bool, model: impl Fn(f64) -> (f64, f64)) -> f64 {
    let mut s = 0.0;
    let mut prev = 0.0;
    for (e, &t) in te.iter().enumerate() {
        let z = model(t);
        let raw = z.1.atan2(z.0);
        let ang = if e == 0 || !unwrap { raw } else { unwrap_near(raw, prev) };
        prev = ang;
        s += ang;
    }
    -s / den
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One χ per voxel repeated at every echo: the per-echo QSM `decompose` takes.
    fn per_echo(chi: &[f64], ne: usize) -> Vec<f64> {
        chi.iter().flat_map(|&c| std::iter::repeat_n(c, ne)).collect()
    }

    /// Output mapping on a synthetic voxel: χ+ = |DSC| and χ− = −|PSC| of the parameters fitted
    /// to conj(y), as the authors write them, and on paramagnetic-dominant data χ+ dominates.
    #[test]
    fn outputs_are_the_authors_dsc_and_psc() {
        let te = [0.004, 0.012, 0.020, 0.028];
        let b0 = 7.0;
        let params = DecomposeParams {
            b0,
            n_inner: 12,
            chi_bound: 0.5,
            max_iterations: 400,
            unwrap_phase: false,
            solvers: Default::default(),
        };
        let truth = Fit {
            c: [0.5, 0.1, 0.4],
            r0: 20.0,
            chi: [0.10, -0.02],
        };
        let ne = te.len();
        let mag: Vec<f64> = te
            .iter()
            .map(|&t| {
                let z = signal_model(t, &truth, b0);
                (z.0 * z.0 + z.1 * z.1).sqrt()
            })
            .collect();
        let chi = per_echo(&[truth.chi[0] + truth.chi[1]], ne);
        let (pos, neg, tot) = decompose(&chi, &mag, &te, &[1u8], &params, |_, _| {}).unwrap();

        // The same fit by hand.
        let gmax = mag.iter().cloned().fold(0.0, f64::max);
        let yc: Vec<(f64, f64)> = (0..ne)
            .map(|e| {
                let ph = -(2.0 / 3.0) * chi[e] * GAMMA * b0 * te[e];
                (mag[e] / gmax * ph.cos(), -(mag[e] / gmax * ph.sin()))
            })
            .collect();
        let logyc: Vec<(f64, f64)> = yc
            .iter()
            .map(|&z| (z.0.hypot(z.1).ln(), z.1.atan2(z.0)))
            .collect();
        let mut ws = StageSolvers::new(2 * ne);
        let p = fit_voxel(&te, &yc, &logyc, b0, &params, &mut ws);
        let den = (2.0 / 3.0) * GAMMA * b0 * te.iter().sum::<f64>();
        let dsc = recon_phase(&te, den, false, |t| dsc_model(t, &p, b0));
        let psc = recon_phase(&te, den, false, |t| psc_model(t, &p, b0));
        assert_eq!((pos[0], neg[0]), (dsc.abs(), -psc.abs()));
        assert!(
            pos[0] > neg[0].abs(),
            "χ+ dominant: χ+={} χ−={}",
            pos[0],
            neg[0]
        );
        assert!(
            (tot[0] - (pos[0] + neg[0])).abs() < 1e-12,
            "χ_total = χ+ + χ−"
        );
    }

    /// Each echo's phase comes from that echo's QSM: a per-echo χ that differs between echoes
    /// gives a different result from one repeated χ.
    #[test]
    fn each_echo_uses_its_own_qsm() {
        let te = [0.004, 0.012, 0.020, 0.028];
        let params = DecomposeParams { b0: 3.0, n_inner: 4, ..Default::default() };
        let mag = [1.0, 0.8, 0.6, 0.45];
        let same = decompose(&[0.05; 4], &mag, &te, &[1u8], &params, |_, _| {}).unwrap();
        let varying = decompose(&[0.05, 0.06, 0.07, 0.08], &mag, &te, &[1u8], &params, |_, _| {}).unwrap();
        assert_ne!((same.0[0], same.1[0]), (varying.0[0], varying.1[0]));
    }

    /// Regression: χ+ must survive past the phase-wrap threshold.
    ///
    /// A compartment accumulates `(2/3)·χ·γ·B0·TE` radians, so with `atan2`'s
    /// `(−π, π]` fold and no unwrapping the alternating fit collapses to a
    /// degenerate fixed point above `|χ_total| = π / ((2/3)·γ·B0·TE_max)` and
    /// returns χ+ = 0 exactly. At the phantom's 7 T and 28 ms last echo that
    /// threshold is 0.090 ppm — inside the globus pallidus, which consequently
    /// reconstructed as a hole. Sweep across it and require a response that is
    /// non-zero and increasing, not a dropout.
    #[test]
    fn chi_pos_survives_the_phase_wrap_threshold() {
        let te = [0.004, 0.012, 0.020, 0.028]; // 7 T phantom sampling
        let b0 = 7.0;
        let params = DecomposeParams {
            b0,
            n_inner: 10,
            chi_bound: 0.5,
            max_iterations: 400,
            unwrap_phase: true,
            solvers: Default::default(),
        };
        let threshold = PI / ((2.0 / 3.0) * GAMMA * b0 * te[te.len() - 1]);
        assert!(
            (0.06..0.12).contains(&threshold),
            "test assumes a threshold inside the swept range, got {threshold}"
        );

        // One voxel forward-simulated from the model, for a range of true χ+ that
        // straddles the threshold.
        let (cp, cm, c0, chim, r0) = (0.35, 0.25, 0.40, -0.03, 25.0);
        let mut prev = 0.0_f64;
        for k in 4..=13 {
            let chip = 0.02 * k as f64;
            let mag: Vec<f64> = te
                .iter()
                .map(|&t| {
                    let z = signal_model(t, &Fit { c: [cp, cm, c0], r0, chi: [chip, chim] }, b0);
                    (z.0 * z.0 + z.1 * z.1).sqrt()
                })
                .collect();
            let (pos, _neg, _tot) =
                decompose(&per_echo(&[chip + chim], te.len()), &mag, &te, &[1u8], &params, |_, _| {}).unwrap();
            assert!(
                pos[0] > 0.0,
                "χ+ collapsed to zero at χ+={chip} (χ_total={}, threshold {threshold:.4})",
                chip + chim
            );
            assert!(
                pos[0] > prev,
                "χ+ must increase with the true source: {} -> {} at χ+={chip}",
                prev,
                pos[0]
            );
            prev = pos[0];
        }
    }

    /// Regression: no scattered dropouts across the deep-grey χ+ range.
    ///
    /// Unwrapping alone was not enough. The χ search ran to `chi_bound` (0.5 ppm
    /// by default), past the 0.31 ppm that 8 ms echo spacing can represent
    /// without the per-echo phase step exceeding π. In that aliased region the
    /// stage-3 landscape grows spurious minima, and voxels at particular
    /// (χ, R2*) combinations fell into them — leaving a recovered pallidum that
    /// was bright but speckled with black. Feed single-compartment data built the
    /// way the phantom builds it (magnitude decaying at R2* = R2 + Dr·|χ|, phase
    /// from χ_total) and require every voxel across the band to recover.
    #[test]
    fn chi_pos_has_no_dropouts_across_deep_grey() {
        let te = [0.004, 0.012, 0.020, 0.028];
        let b0 = 7.0;
        let params = DecomposeParams {
            b0,
            n_inner: 10,
            chi_bound: 0.5,
            max_iterations: 400,
            unwrap_phase: true,
            solvers: Default::default(),
        };
        let (dr_pos, dr_neg, r2) = (320.0_f64, 397.0_f64, 30.0_f64);
        let chim = -0.015_f64;
        for k in 3..=30 {
            let chip = 0.01 * k as f64;
            let r2s = r2 + dr_pos * chip + dr_neg * chim.abs();
            let mag: Vec<f64> = te.iter().map(|&t| (-r2s * t).exp()).collect();
            let (pos, _neg, _tot) =
                decompose(&per_echo(&[chip + chim], te.len()), &mag, &te, &[1u8], &params, |_, _| {}).unwrap();
            assert!(
                pos[0] > 0.3 * chip,
                "dropout at χ+={chip:.2}: recovered {:.5} (gain {:.3})",
                pos[0],
                pos[0] / chip
            );
        }
    }

    /// Multi-voxel run exercising the masked-out, all-zero, and progress paths.
    #[test]
    fn decompose_multivoxel_mask_and_zero_paths() {
        let te = [0.004, 0.012, 0.020, 0.028];
        let b0 = 7.0;
        let ne = te.len();
        let params = DecomposeParams { b0, n_inner: 4, chi_bound: 0.5, max_iterations: 400, unwrap_phase: false, solvers: Default::default() };

        // A real signal for the fitted voxels.
        let (cp, cm, c0, chip, chim, r0) = (0.4, 0.2, 0.4, 0.06, -0.05, 22.0);
        let sig: Vec<(f64, f64)> = te
            .iter()
            .map(|&t| signal_model(t, &Fit { c: [cp, cm, c0], r0, chi: [chip, chim] }, b0))
            .collect();
        let smag: Vec<f64> = sig.iter().map(|z| (z.0 * z.0 + z.1 * z.1).sqrt()).collect();

        // 4 voxels: [0]=masked-out, [1]=all-zero magnitude, [2],[3]=real signal.
        let n = 4;
        let mut mag = vec![0.0_f64; n * ne];
        for e in 0..ne {
            mag[2 * ne + e] = smag[e];
            mag[3 * ne + e] = smag[e];
        }
        let chi_total = per_echo(&vec![chip + chim; n], ne);
        let mask = vec![0u8, 1, 1, 1];

        let mut last = (0usize, 0usize);
        let (pos, neg, tot) =
            decompose(&chi_total, &mag, &te, &mask, &params, |i, t| last = (i, t)).unwrap();

        // Masked-out and all-zero voxels stay at zero.
        assert_eq!((pos[0], neg[0], tot[0]), (0.0, 0.0, 0.0), "masked voxel");
        assert_eq!((pos[1], neg[1], tot[1]), (0.0, 0.0, 0.0), "all-zero voxel");
        // Fitted voxels obey the sign convention + invariant.
        for i in [2usize, 3] {
            assert!(pos[i] >= 0.0 && neg[i] <= 0.0, "signs at voxel {i}");
            assert!((tot[i] - (pos[i] + neg[i])).abs() < 1e-12, "invariant at voxel {i}");
        }
        // Progress reported completion over the 3 masked voxels.
        assert_eq!(last, (3, 3), "final progress");
    }

    /// #199: a negative `chi_bound` panicked inside the fit; it is an error now, as are the
    /// shape problems that used to be asserts.
    #[test]
    fn negative_chi_bound_and_bad_shapes_are_errors() {
        let te = [0.004, 0.008, 0.012];
        let mag = [1.0, 0.9, 0.8];
        for bad in [-0.1, f64::NAN] {
            let params = DecomposeParams { chi_bound: bad, ..Default::default() };
            let err = decompose(&[0.0; 3], &mag, &te, &[1u8], &params, |_, _| {}).unwrap_err();
            assert!(err.0.contains("DECOMPOSE chi_bound") && err.0.contains(">= 0"), "{err}");
        }
        let ok = DecomposeParams::default();
        assert!(decompose(&[0.0; 2], &mag[..2], &te[..2], &[1u8], &ok, |_, _| {}).is_err());
        assert!(decompose(&[0.0; 3], &mag[..2], &te, &[1u8], &ok, |_, _| {}).is_err());
        assert!(decompose(&[0.0; 3], &mag, &te, &[1u8, 1], &ok, |_, _| {}).is_err());
        // One χ per voxel where one per echo is needed.
        assert!(decompose(&[0.0], &mag, &te, &[1u8], &ok, |_, _| {}).is_err());
    }

    /// Every `lsqcurvefit` call of the authors' fit, replayed stage by stage against MATLAB
    /// R2023b (`scripts/lsq_trr_matlab/probe_lsq_trr.m` on the conj(y) that
    /// `decompose_qsm_parity::decompose_probe_input` writes): each stage starts from MATLAB's
    /// own x0 and fixed parameters, so a difference is that call's alone.
    /// `DECOMPOSE_TRACE=<trace> DECOMPOSE_PROBE_IN=<input> cargo test --release --lib
    /// decompose_stages_match_matlab -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn decompose_stages_match_matlab() {
        use crate::solvers::lsq_trr_trace::{compare, dump, parse, replay};
        let (Ok(tp), Ok(yp)) = (
            std::env::var("DECOMPOSE_TRACE"),
            std::env::var("DECOMPOSE_PROBE_IN"),
        ) else {
            return;
        };
        let raw: Vec<f64> = std::fs::read(yp)
            .unwrap()
            .chunks(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let (nv, ne, b0) = (raw[0] as usize, raw[1] as usize, raw[2]);
        let te = raw[3..3 + ne].to_vec();
        let yv = |v: usize| -> Vec<(f64, f64)> {
            let o = 3 + ne + v * 2 * ne;
            (0..ne).map(|e| (raw[o + e], raw[o + ne + e])).collect()
        };
        assert!(nv > 0);
        let calls = parse(&std::fs::read_to_string(tp).unwrap());
        let dump_tag = std::env::var("DECOMPOSE_DUMP").ok();
        let (mut ok, mut total) = (0, 0);
        let mut worst: Vec<(f64, String)> = Vec::new();
        let mut by_stage = [[0usize; 2]; 3];
        let only = std::env::var("DECOMPOSE_ONLY").ok();
        for call in calls.iter().filter(|c| c.stage.is_some() && only.as_deref().is_none_or(|o| o == c.tag)) {
            let (v, _, st, fx) = call.stage.unwrap();
            let yc = yv(v);
            let logyc: Vec<(f64, f64)> = yc
                .iter()
                .map(|z| (z.0.hypot(z.1).ln(), z.1.atan2(z.0)))
                .collect();
            let q = Fit {
                c: [fx[3], fx[4], fx[5]],
                r0: fx[2],
                chi: [fx[0], fx[1]],
            };
            let log_resid = |q: &Fit, out: &mut [f64]| {
                for e in 0..ne {
                    let z = signal_model(te[e], q, b0);
                    out[e] = z.0.hypot(z.1).ln() - logyc[e].0;
                    out[ne + e] = -z.1.atan2(z.0) - -logyc[e].1;
                }
            };
            let r = match st {
                1 => replay(call, |x, out| {
                    let q = Fit {
                        c: [x[0], x[1], x[2]],
                        ..q
                    };
                    for e in 0..ne {
                        let z = signal_model(te[e], &q, b0);
                        out[e] = z.0 - yc[e].0;
                        out[ne + e] = -z.1 - -yc[e].1;
                    }
                }),
                2 => replay(call, |x, out| log_resid(&Fit { r0: x[0], ..q }, out)),
                _ => replay(call, |x, out| {
                    log_resid(
                        &Fit {
                            chi: [x[0], x[1]],
                            ..q
                        },
                        out,
                    )
                }),
            };
            let d = compare(call, &r, 1e-6);
            total += 1;
            let good = d.same_counts && d.same_exit && d.first_divergence.is_none();
            by_stage[st - 1][usize::from(good)] += 1;
            if good {
                ok += 1;
            }
            worst.push((d.x_rel, format!("{d:?}")));
            if !good && std::env::var("DECOMPOSE_LIST").is_ok() {
                eprintln!("FAIL {} fd={:?} x_rel={:.1e} M={:?} R={:?}", d.tag, d.first_divergence, d.x_rel, d.matlab, d.rust);
            }
            if dump_tag.as_deref() == Some(call.tag.as_str()) {
                eprintln!("{}", dump(call, &r));
            }
        }
        worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for w in worst.iter().take(15) {
            eprintln!("{}", w.1);
        }
        eprintln!("{ok} of {total} stage calls identical (counts, exit flag, every evaluation to 1e-6); by stage [differ, same]: {by_stage:?}");
    }
}

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
//! degenerate fixed point (χ− pinned at the χ bound, R2*₀ and the amplitudes at
//! zero) and returns χ+ ≈ 0, so the most paramagnetic structure in the
//! volume reads as a hole. Both the stage-2/3 residuals and `recon_phase`
//! therefore walk the echoes in order and keep each phase on the branch nearest
//! its predecessor ([`unwrap_near`]); the synthesized data phase is known in
//! closed form and is used unwrapped rather than round-tripped through `atan2`.
//! The option does only this; it does not change the χ search (that is
//! [`DecomposeParams::chi_bound`]). The original method, published at 3 T, does not meet this
//! limit in practice — the threshold there is ~0.2 ppm.
//!
//! **The χ bound** ([`DecomposeParams::chi_bound`], a tuning value). The authors search
//! `0 ≤ χ+ ≤ 0.5`, `−0.5 ≤ χ− ≤ 0` ppm, a bound "roughly calculated using TE = 25 ms and B0 = 3T
//! to ensure the phase does not exceed 2π" (Chen et al. 2021, §2.2); that is the default
//! ([`DecomposeChiBound::Ppm`]`(0.5)`). [`DecomposeChiBound::EchoSpacing`] instead takes the
//! bound from the scan: `π / ((2/3)·γ·B0·ΔTE_max)`, with `ΔTE_max` the largest phase step the
//! unwrapping walks (`TE₁` from zero, or any echo spacing). Unwrapping along the echoes is
//! unambiguous only while the phase step between echoes stays under π, and this is the largest
//! |χ| for which it does: 0.31 ppm (0.3145) at 7 T with 8 ms spacing, 0.73 ppm at 3 T. Past it the
//! model matches a given χ on more than one branch and the stage-3 landscape grows spurious
//! minima.
//!
//! **At 7 T.** The authors designed the fit for 3 T, and at 7 T the phase wraps between echoes
//! inside ordinary deep grey matter. Measured 2026-10-30 on the QSM-CI 7 T chisep phantom
//! (164×205×205, 1 mm, TE 4/12/20/28 ms; its provided χ given at every echo) with this code at
//! astewartau/QSM.rs@add3d6e (the `lsqcurvefit` port, FDLIBM), `n_inner` 10:
//!
//! ```text
//! QSMCI_CHISEP=<phantom> DECOMPOSE_UNWRAP=<0|1> DECOMPOSE_CHI_BOUND=<ppm|echo-spacing> \
//!   cargo test --release --features parallel --test chisep_relaxometry_qsmci \
//!   test_decompose_qsmci -- --ignored --nocapture
//! ```
//!
//! "Pallidum" is the phantom's globus pallidus (`dseg` label 2, 3683 voxels, true mean χ+
//! 0.137 ppm); "collapsed" counts its voxels whose χ+ is under a quarter of the truth.
//!
//! | `unwrap_phase` | `chi_bound` | χ+ r / XSIM / NRMSE | χ− r / XSIM / NRMSE | pallidum mean χ+ | collapsed |
//! |---|---|---|---|---|---|
//! | off (authors) | 0.5 (authors) | 0.484 / 0.216 / 10.9 % | 0.745 / 0.325 / 4.8 % | 0.002 | 3541 |
//! | on | 0.5 | 0.658 / 0.250 / 10.0 % | 0.751 / 0.324 / 4.7 % | 0.027 | 2932 |
//! | off | echo spacing (0.3145) | 0.587 / 0.367 / 9.7 % | 0.846 / 0.471 / 4.1 % | 0.005 | 3372 |
//! | on | echo spacing (0.3145) | **0.895 / 0.489 / 7.1 %** | **0.865 / 0.477 / 4.0 %** | 0.111 | 17 |
//!
//! So the two help in different ways. The bound alone improves χ− (r 0.75 → 0.85) and χ+'s XSIM,
//! but leaves the pallidum collapsed; unwrapping alone recovers little of it; together they
//! recover it. (The rows were first measured with the bound hard-wired into the unwrapping
//! option, as `min(0.5, π / ((2/3)·γ·B0·ΔTE_max))`; that option, and a plain `chi_bound` of
//! 0.3145 with and without unwrapping, gave the same metrics to the digits shown, and so does
//! [`DecomposeChiBound::EchoSpacing`]. An older note
//! here, from the bounded-LM port, said the bound removed "a speckle of dropouts" that unwrapping
//! alone left; with this solver, unwrapping alone leaves most of the pallidum collapsed, and
//! voxels at exactly χ+ = 0 number 1–8 of 1.31 M in every row.) The defaults stay the authors'
//! (off, 0.5); the pipeline rules `decompose.unwrap-7t` and `decompose.chi-bound-7t` suggest
//! both at 7 T (from 6.5 T, so a Siemens 7 T's 6.98 T counts). An earlier figure, χ+ r 0.59 for
//! the authors' default against 0.87 (QSM.rs#333's CI run 40786263426), compared ports that also
//! differed in the solver, so it isolated neither change.
//!
//! Note that `−Σ angle / den` remains a *compressive* estimator of χ: its gain
//! rises with χ itself (≈0.03 at white-matter χ+, ≈0.87 in the pallidum on the
//! QSM-CI phantom), so it flattens contrast between weak sources. That is
//! inherent to the reference's reconstruction, not to this port — the fitted
//! compartment amplitudes set the gain, and they are only weakly identifiable
//! from data that carries no sub-voxel compartment signature.
//!
//! **Even on its own model the fit underestimates.** Data generated noise-free from the signal
//! model above was given as the per-echo QSM, χ(TE) = −∠S(TE) / ((2/3)·γ·B0·TE) (Eq. 4). That is
//! 400 voxels with C± 0.05–0.4, χ± ±(0.01–0.15) ppm, R2*₀ 15–25 s⁻¹, at 3 T with 16 echoes from
//! 2.7 to 46.2 ms; no phase wraps. Measured 2026-11-01, `n_inner` 10:
//!
//! ```text
//!                   r vs Eq. 6/7 at the truth   median out/truth   slope through 0   exactly 0
//!   χ+ (vs PCS)     0.85                        0.71               0.86              44 / 400
//!   χ− (vs DCS)     0.83                        0.70               0.85              48 / 400
//! ```
//!
//! The fitted C± and χ± correlate with the truth at only r 0.15–0.40. The 10-pass alternation
//! stops at a fixed point short of the global minimum from the reference's start values. On such
//! data that minimum is exact: a global fit of all six parameters recovers the truth. MATLAB R2023b, running
//! the reference loop (the QSM-CI `decompose-qsm` code) on the same voxels, gives the same table.
//! This port is within 0.6 ppb of it in every voxel (NRMSE 0.47 % χ+, 0.29 % χ−). So the
//! underestimate belongs to the method, not to the port.
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
//! # The paper's fit
//!
//! [`DecomposeParams::paper`] is Chen et al. 2021's fit, which the DECOMPOSE pipeline uses
//! (rule `decompose.fit-paper`; decided 2026-11-10); the default stays the reference
//! implementation's, which QSM-CI's `decompose-qsm` container runs.
//!
//! - **Stage 5.1** ([`DecomposeStageSolver::LsqlinSumToOne`]): the amplitudes with
//!   `C+ + C− + C₀ = 1`, `0 ≤ C ≤ 1`, solved exactly on the simplex instead of by `lsqlin`'s
//!   interior-point iterations. The minimum is unique when the three compartments' signals are
//!   independent, so `lsqlin` reaches the same point to its tolerance (to be confirmed against
//!   MATLAB's `lsqlin`).
//! - **Stage 5.2** ([`DecomposeStageSolver::LsqlinLog`]): R2*₀ in closed form, since the log
//!   residual is linear in it.
//! - **Stage 5.3**: `lsqnonlin` trust-region-reflective, which is the `lsqcurvefit` port.
//! - **Scale** ([`DecomposeNormalisation::ExtrapolatedS0`]): the sum-to-one constraint makes the
//!   model 1 at `TE = 0`, so each voxel is divided by its `S(0)`, extrapolated from the first two
//!   echoes. The paper does not say how it scales the data; this is QSM.rs's reading (Ashley,
//!   2026-11-10).
//! - **Initial values and tolerances**: the paper gives none, so they are the reference
//!   implementation's (`C = [0.3 0.3 0.4]`, `R2*₀ = 25`, `χ = [0.05 −0.05]`, MATLAB's default
//!   tolerances).
//!
//! # Reference pipeline
//!
//! Chen et al. 2021 (NeuroImage 242:118477, §3.6 "Image preparation and QSM reconstruction"; PMC8720043):
//!
//! - **Data:** in vivo at 3 T (GE MR750w), 0.86 × 0.86 × 1.0 mm, 16 echoes,
//!   TE1/ΔTE/TE16 = 2.7/2.9/46.2 ms. The phantom was 1 mm iso, 16 echoes 1.47–25.9 ms; the ex
//!   vivo data were 7 T.
//! - **Mask:** FSL BET. **Unwrapping:** Laplacian. **Background removal:** V-SHARP.
//! - **QSM input:** STAR-QSM computed **for each echo** ("Note that QSM of each echo, χ(t), is
//!   used instead of the average QSM across all echoes", §2.2; "it is necessary to use
//!   TE-dependent QSM as the input", §4.1). A host gets it from a per-echo reconstruction
//!   ([`crate::pipeline::per_echo`]). The signal model (Eq. 4) is
//!   `y(t) = M(t)·exp(−i·(2/3)·χ(t)·γ·B0·t)` with the per-echo χ(t), not one echo-combined χ.
//!   [`decompose`] takes that per-echo χ. (The authors' script runs STAR-QSM per echo at
//!   `padsize` 12 after `MRPhaseUnwrap` and `V_SHARP(smvsize 25)` of that echo.)
//! - **Referencing:** χ referenced to the whole-brain mean.
//! - **Constants:** `0.5 > |χ+|, |χ−| > 0` (§2.2, [`DecomposeParams::chi_bound`]'s default), and
//!   `a = 323.5 Hz/ppm` at 3 T (§2.1), which the formula above reproduces and which scales with
//!   B0 by construction.
//!
//! [`DecomposeParams`]'s `b0` defaults to 7 T. The pipeline dispatcher overrides it from the scan
//! metadata.
//!
//! Reference:
//! Chen, J., et al. (2021). "Decompose quantitative susceptibility mapping (QSM)
//! to sub-voxel diamagnetic and paramagnetic components based on gradient-echo MRI
//! data." NeuroImage 242:118477. https://doi.org/10.1016/j.neuroimage.2021.118477
//! Reference implementation (QSM-CI port of Tim Ho's open MATLAB DECOMPOSE-QSM).
//!
//! # Pipeline rules
//!
//! The facts above that a configuration can depart from are published as rules in
//! [`crate::pipeline::rules::RULES`] (each with its strength, basis, reason and source): `decompose.mask-bet`, `decompose.unwrap-laplacian`, `decompose.bg-vsharp`, `decompose.qsm-star`, `decompose.qsm-per-echo` (Recommended: stated by the authors, §2.2 and §4.1), `decompose.fit-paper`, `decompose.reference-mean`, and at 7 T `decompose.unwrap-7t` and `decompose.chi-bound-7t` (Recommended: measured, *At 7 T* above).

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use std::f64::consts::PI;

use crate::error::{check_len, InputError};
use crate::solvers::lsq_trr::{lsq_trust_region_reflective, LsqTrrOptions, LsqTrrWorkspace};
use crate::utils::fdlibm;

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
    /// The bound on |χ+| and |χ−| in the stage-3 search. Default: the authors' 0.5 ppm
    /// ([`DecomposeChiBound::Ppm`]); [`DecomposeChiBound::EchoSpacing`] takes it from the scan
    /// (see the module docs, *The χ bound*).
    pub chi_bound: DecomposeChiBound,
    /// `optimoptions('lsqcurvefit', …)` of every stage: iterations, evaluations, tolerances and
    /// finite-difference settings. Default: MATLAB's defaults ([`LsqTrrOptions::default`]), which
    /// the authors' code keeps (`MaxIterations` 400, the 1e-6 tolerances).
    pub lsqcurvefit: LsqTrrOptions,
    /// Unwrap the phase along the echoes (see the module docs, *Phase unwrapping*). Default
    /// `false`: the authors' code takes every phase on its principal branch (`log`, `angle`).
    /// It leaves the χ search to [`chi_bound`](Self::chi_bound).
    pub unwrap_phase: bool,
    /// The solver of each of the three stages (see [`DecomposeStageSolvers`]).
    pub solvers: DecomposeStageSolvers,
    /// How the magnitude is scaled before the fit (see [`DecomposeNormalisation`]). Default: the
    /// reference implementation's global maximum.
    pub normalisation: DecomposeNormalisation,
}

/// How [`decompose`] scales the magnitude before fitting it.
///
/// The paper's stage 5.1 constrains `C+ + C− + C₀ = 1`, so the model is 1 at `TE = 0` and each
/// voxel's data must be scaled to `S(0) = 1`. The paper does not say how (Chen et al. 2021, §2.2,
/// §3.1); the reference implementation, which has no sum-to-one constraint, divides by the global
/// maximum.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DecomposeNormalisation {
    /// Divide every voxel by the largest magnitude in the volume (`DECOMPOSE_analysis.m`).
    #[default]
    GlobalMax,
    /// Divide each voxel by its own first-echo magnitude (`S(TE₁) = 1`).
    FirstEcho,
    /// Divide each voxel by its signal at `TE = 0`, extrapolated mono-exponentially from the first
    /// two echoes, `S₀ = M₁·(M₁/M₂)^(TE₁/(TE₂−TE₁))`; by `M₁` where that is not a decay
    /// (`M₂ ≥ M₁`, or a zero). QSM.rs's reading of "S(0) = 1", which the paper leaves unstated.
    ExtrapolatedS0,
}

impl DecomposeNormalisation {
    /// Stable id: `global-max`, `first-echo` or `extrapolated-s0`.
    pub fn id(self) -> &'static str {
        match self {
            Self::GlobalMax => "global-max",
            Self::FirstEcho => "first-echo",
            Self::ExtrapolatedS0 => "extrapolated-s0",
        }
    }

    /// The divisor for one voxel's magnitudes `m` at echo times `te`, given the volume's maximum.
    fn divisor(self, m: &[f64], te: &[f64], gmax: f64) -> f64 {
        match self {
            Self::GlobalMax => gmax,
            Self::FirstEcho => m[0],
            Self::ExtrapolatedS0 => {
                let (m1, m2) = (m[0], m[1]);
                if m1 > 0.0 && m2 > 0.0 && m2 < m1 {
                    let s0 = m1 * (m1 / m2).powf(te[0] / (te[1] - te[0]));
                    if s0.is_finite() { s0 } else { m1 }
                } else {
                    m1
                }
            }
        }
    }
}

/// Which solver fits each stage of the alternating fit.
///
/// The reference implementation (Tim Ho's MATLAB DECOMPOSE-QSM, which the QSM-CI `decompose-qsm`
/// container runs) calls `lsqcurvefit` with its default trust-region-reflective algorithm for all
/// three stages, and that is the default here. The paper (Chen et al. 2021, §3.1) describes a
/// different choice: Eq. 5.1 (the amplitudes, with `C+ + C− + C₀ = 1`) and Eq. 5.2 (R2*₀) by
/// `lsqlin`'s interior-point method, and only Eq. 5.3 (χ±) by `lsqnonlin` trust-region-reflective
/// with a supplied Jacobian sparsity pattern. [`DecomposeStageSolvers::paper`] selects those
/// (see each [`DecomposeStageSolver`] for how it reproduces them).
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

impl DecomposeStageSolvers {
    /// The paper's solvers (Chen et al. 2021, §2.2, §3.1): Eq. 5.1 by `lsqlin` with
    /// `C+ + C− + C₀ = 1` and `0 ≤ C ≤ 1`, Eq. 5.2 by `lsqlin` on the log signal, Eq. 5.3 by
    /// `lsqnonlin` trust-region-reflective.
    pub fn paper() -> Self {
        Self {
            amplitudes: DecomposeStageSolver::LsqlinSumToOne,
            r2star: DecomposeStageSolver::LsqlinLog,
            chi: DecomposeStageSolver::LsqcurvefitTrustRegionReflective,
        }
    }

    /// Checks that each stage has a solver it can use: [`DecomposeStageSolver::LsqlinSumToOne`]
    /// only for the amplitudes and [`DecomposeStageSolver::LsqlinLog`] only for R2*₀ (stage 3 is
    /// not linear).
    pub fn validate(&self) -> Result<(), InputError> {
        use DecomposeStageSolver as S;
        let bad = |stage: &str, s: S| InputError::new(format!("DECOMPOSE: {s:?} cannot fit the {stage}"));
        if self.amplitudes == S::LsqlinLog {
            return Err(bad("amplitudes", self.amplitudes));
        }
        if self.r2star == S::LsqlinSumToOne {
            return Err(bad("R2*0 stage", self.r2star));
        }
        if self.chi != S::LsqcurvefitTrustRegionReflective {
            return Err(bad("χ stage", self.chi));
        }
        Ok(())
    }
}

/// A stage solver for [`decompose`].
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecomposeStageSolver {
    /// MATLAB `lsqcurvefit` with its default options (trust-region-reflective, finite-difference
    /// Jacobian): [`lsq_trust_region_reflective`]. The reference implementation's solver. The
    /// paper's stage-5.3 solver, `lsqnonlin` trust-region-reflective with a Jacobian sparsity
    /// pattern, is the same algorithm on the same residual: `lsqcurvefit` is `lsqnonlin` on
    /// `fun(x, TE) − ydata`, and a sparsity pattern only groups the finite differences, which
    /// for two parameters with a dense Jacobian changes nothing.
    LsqcurvefitTrustRegionReflective,
    /// The paper's stage 5.1 (amplitudes only): `lsqlin`, linear least squares of the complex
    /// signal in `[C+, C−, C₀]` with `C+ + C− + C₀ = 1` and `0 ≤ C ≤ 1`.
    ///
    /// Solved exactly instead of by `lsqlin`'s interior-point iterations. The problem is a convex
    /// quadratic on the probability simplex (`0 ≤ C` and `ΣC = 1` imply `C ≤ 1`); every face of
    /// the simplex is tried (the three vertices, three edges and the interior: the
    /// equality-constrained least squares on that face, from its KKT system) and the feasible
    /// solution with the smallest residual is the minimum. When the three compartments' signals
    /// are linearly independent the objective is strictly convex and its minimum unique, so
    /// `lsqlin` converges to this point and differs from it only by its tolerances (to be checked
    /// against MATLAB's `lsqlin`). When they are not (χ+ or χ− at 0 makes that compartment's
    /// signal equal C₀'s), the minimum is a set; interior-point methods tend to its centre, and
    /// a tiny ridge (1e-12 of the trace) makes this the minimum-norm point, which splits equal
    /// compartments equally. The reconstruction (PSC, DSC) does not depend on that split.
    LsqlinSumToOne,
    /// The paper's stage 5.2 (R2*₀ only): `lsqlin` on the log signal, `R2*₀ ≥ 0`. All three
    /// compartments decay with `exp(−R2*₀·t)`, so `log S(t) = −R2*₀·t + log S|_{R2*₀=0}(t)`:
    /// the real part of the log residual is linear in R2*₀ and its imaginary part does not
    /// depend on it. The least-squares R2*₀ is `Σt(L−ℓ)/Σt²` (`L` the model's log magnitude at
    /// R2*₀ = 0, `ℓ` the data's), clamped at 0: the exact minimum of this one-dimensional
    /// convex problem, which `lsqlin` reaches to its tolerance.
    LsqlinLog,
}

/// The bound on |χ| in DECOMPOSE's stage-3 search (`[0, −bound]` to `[bound, 0]`), see the
/// module docs, *The χ bound*.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DecomposeChiBound {
    /// A fixed bound in ppm. The authors' is 0.5 (Chen et al. 2021, §2.2), the default.
    Ppm(f64),
    /// `π / ((2/3)·γ·B0·ΔTE_max)`, the largest |χ| whose phase step between echoes stays under π,
    /// so that unwrapping along the echoes is unambiguous; `ΔTE_max` is the largest of `TE₁` and
    /// the echo spacings. A tuning value: 0.3145 ppm at 7 T with 8 ms spacing.
    EchoSpacing,
}

impl DecomposeParams {
    /// Checks a fixed `chi_bound` ([`DecomposeChiBound::Ppm`]): it must be ≥ 0 ppm (a negative
    /// bound leaves the fit's lower bound above its upper one). An
    /// [`EchoSpacing`](DecomposeChiBound::EchoSpacing) bound depends on the echo times, and
    /// [`decompose`] checks it.
    pub fn validate(&self) -> Result<(), InputError> {
        self.solvers.validate()?;
        if let DecomposeChiBound::Ppm(_) = self.chi_bound {
            self.chi_bound.resolve(self.b0, &[])?;
        }
        Ok(())
    }
}

impl Default for DecomposeChiBound {
    fn default() -> Self {
        DecomposeChiBound::Ppm(0.5)
    }
}

impl DecomposeChiBound {
    /// Stable id: `"ppm"` (whatever its value) or `"echo-spacing"`.
    pub fn id(self) -> &'static str {
        match self {
            DecomposeChiBound::Ppm(_) => "ppm",
            DecomposeChiBound::EchoSpacing => "echo-spacing",
        }
    }

    /// The bound in ppm at field `b0` (T) and `echo_times` (s, in acquisition order). Errors if
    /// it is not ≥ 0 (a negative bound leaves the fit's lower bound above its upper one), or for
    /// [`EchoSpacing`](Self::EchoSpacing) not a finite positive bound (no echoes, a time ≤ 0).
    pub fn resolve(self, b0: f64, echo_times: &[f64]) -> Result<f64, InputError> {
        match self {
            DecomposeChiBound::Ppm(x) if x.is_nan() || x < 0.0 => {
                Err(InputError::param("DECOMPOSE chi_bound", x, ">= 0 ppm"))
            }
            DecomposeChiBound::Ppm(x) => Ok(x),
            DecomposeChiBound::EchoSpacing => {
                let first = echo_times.first().copied().unwrap_or(f64::NAN);
                let dte = echo_times.windows(2).map(|w| w[1] - w[0]).fold(first, f64::max);
                let bound = PI / ((2.0 / 3.0) * GAMMA * b0 * dte);
                if bound.is_finite() && bound > 0.0 {
                    Ok(bound)
                } else {
                    Err(InputError::new(format!(
                        "DECOMPOSE chi_bound from the echo spacing must be a finite bound > 0 ppm, \
                         got {bound} (B0 {b0} T, echo times {echo_times:?})"
                    )))
                }
            }
        }
    }
}

impl Default for DecomposeParams {
    fn default() -> Self {
        Self {
            b0: 7.0,
            n_inner: 10,
            chi_bound: DecomposeChiBound::default(),
            lsqcurvefit: LsqTrrOptions::default(),
            unwrap_phase: false,
            solvers: DecomposeStageSolvers::default(),
            normalisation: DecomposeNormalisation::default(),
        }
    }
}

impl DecomposeParams {
    /// Which fit this is, as rules name it: `paper` ([`Self::paper`]'s solvers and
    /// normalisation), `reference-code` (the reference implementation's, the default) or
    /// `custom` (any other combination).
    pub fn fit_id(&self) -> &'static str {
        let p = Self::paper();
        let d = Self::default();
        if self.solvers == p.solvers && self.normalisation == p.normalisation {
            "paper"
        } else if self.solvers == d.solvers && self.normalisation == d.normalisation {
            "reference-code"
        } else {
            "custom"
        }
    }

    /// The paper's fit (Chen et al. 2021): its stage solvers ([`DecomposeStageSolvers::paper`])
    /// on per-voxel `S(0) = 1` data ([`DecomposeNormalisation::ExtrapolatedS0`]). The paper
    /// gives no initial values or tolerances; those stay the reference implementation's.
    pub fn paper() -> Self {
        Self {
            solvers: DecomposeStageSolvers::paper(),
            normalisation: DecomposeNormalisation::ExtrapolatedS0,
            ..Self::default()
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
    let upper = params.chi_bound.resolve(b0, te)?;

    // The magnitude's scale (`params.normalisation`): the reference's global maximum by default.
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
            let scale = params.normalisation.divisor(&magnitude[i * ne..(i + 1) * ne], te, gmax);
            if scale.is_nan() || scale <= 0.0 {
                return (i, 0.0, 0.0);
            }
            for e in 0..ne {
                let m = magnitude[i * ne + e] / scale;
                let ph = -(2.0 / 3.0) * chi_per_echo[i * ne + e] * GAMMA * b0 * te[e];
                let y = (m * fdlibm::cos(ph), m * fdlibm::sin(ph));
                ws.yc[e] = (y.0, -y.1);
                sum_re += ws.yc[e].0;
                sum_im += ws.yc[e].1;
                // log(conj y) for stages 2 & 3: MATLAB's `log`, on the principal branch, unless
                // `unwrap_phase`, where the synthesized phase is used as it is (see
                // `unwrap_near`).
                let (lr, la) = matlab_log(ws.yc[e]);
                ws.logyc[e] = (lr, if params.unwrap_phase { -ph } else { la });
            }
            // The authors skip a voxel whose `sum(ydata)` is zero.
            if sum_re == 0.0 && sum_im == 0.0 {
                return (i, 0.0, 0.0);
            }

            let p = fit_voxel(te, &ws.yc, &ws.logyc, b0, upper, params, &mut ws.solver);

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
///     chi_0 = lsqcurvefit(objective3 (χ),  chi_0, TE, complex_to_real(log(ydata)), [0 -0.5], [0.5 0]);
/// end
/// ```
///
/// with `lsqcurvefit`'s default options, which [`lsq_trust_region_reflective`] reproduces. The
/// χ bound `upper` is [`DecomposeParams::chi_bound`], resolved (the authors' 0.5 by default).
/// `lsqcurvefit`'s residual is `fun(x, TE) − ydata`, packed by `complex_to_real` as `[Re; −Im]`.
fn fit_voxel(
    te: &[f64],
    yc: &[(f64, f64)],
    logyc: &[(f64, f64)],
    b0: f64,
    upper: f64,
    params: &DecomposeParams,
    ws: &mut StageSolvers,
) -> Fit {
    let ne = te.len();
    let mut p = Fit { c: [0.3, 0.3, 0.4], r0: 25.0, chi: [0.05, -0.05] };

    let unwrap = params.unwrap_phase;
    let inf = f64::INFINITY;
    let opts = &params.lsqcurvefit;

    // `complex_to_real(log(model)) − complex_to_real(log(ydata))`, phases optionally unwrapped
    // along the echoes.
    let log_resid = |q: &Fit, out: &mut [f64]| {
        let mut prev = 0.0;
        for e in 0..ne {
            let z = signal_model(te[e], q, b0);
            let (lr, la) = matlab_log(z);
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
            DecomposeStageSolver::LsqcurvefitTrustRegionReflective => { lsq_trust_region_reflective(
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
            opts,
            &mut ws.s1,
            |_| {},
        ); }
            DecomposeStageSolver::LsqlinSumToOne => x = amplitudes_sum_to_one(te, yc, &q, b0),
            DecomposeStageSolver::LsqlinLog => unreachable!("validated"),
        };
        p.c = x;
        // Stage 2: objective2, R2*₀ against log(ydata).
        let q = p;
        let mut x = [p.r0];
        match params.solvers.r2star {
            DecomposeStageSolver::LsqcurvefitTrustRegionReflective => { lsq_trust_region_reflective(
            |x: &[f64], out: &mut [f64]| log_resid(&Fit { r0: x[0], ..q }, out),
            &mut x,
            &[0.0],
            &[inf],
            opts,
            &mut ws.s2,
            |_| {},
        ); }
            DecomposeStageSolver::LsqlinLog => {
                if let Some(r0) = r2star_log_linear(te, logyc, &q, b0) {
                    x = [r0];
                }
            }
            DecomposeStageSolver::LsqlinSumToOne => unreachable!("validated"),
        };
        p.r0 = x[0];
        // Stage 3: objective3, [χ+, χ−] against log(ydata).
        let q = p;
        let mut x = p.chi;
        match params.solvers.chi {
            DecomposeStageSolver::LsqcurvefitTrustRegionReflective => { lsq_trust_region_reflective(
            |x: &[f64], out: &mut [f64]| log_resid(&Fit { chi: [x[0], x[1]], ..q }, out),
            &mut x,
            &[0.0, -upper],
            &[upper, 0.0],
            opts,
            &mut ws.s3,
            |_| {},
        ); }
            DecomposeStageSolver::LsqlinSumToOne | DecomposeStageSolver::LsqlinLog => unreachable!("validated"),
        };
        p.chi = x;
    }
    p
}

/// Stage 5.1 of the paper, exactly ([`DecomposeStageSolver::LsqlinSumToOne`]): the amplitudes
/// minimising `‖Σ_j C_j·b_j(t) − conj(y)‖²` over `C ≥ 0`, `ΣC = 1`, with `b_j` compartment `j`'s
/// signal at unit amplitude under `q`'s R2*₀ and χ.
fn amplitudes_sum_to_one(te: &[f64], yc: &[(f64, f64)], q: &Fit, b0: f64) -> [f64; 3] {
    // Normal equations G = AᵀA, h = Aᵀb over the stacked [Re; Im] rows (the sign convention of
    // the imaginary rows does not change the sums).
    let mut g = [[0.0f64; 3]; 3];
    let mut h = [0.0f64; 3];
    let mut bb = 0.0;
    for (e, &t) in te.iter().enumerate() {
        let cols: [(f64, f64); 3] = std::array::from_fn(|j| {
            let mut c = [0.0; 3];
            c[j] = 1.0;
            signal_model(t, &Fit { c, ..*q }, b0)
        });
        for j in 0..3 {
            for k in 0..3 {
                g[j][k] += cols[j].0 * cols[k].0 + cols[j].1 * cols[k].1;
            }
            h[j] += cols[j].0 * yc[e].0 + cols[j].1 * yc[e].1;
        }
        bb += yc[e].0 * yc[e].0 + yc[e].1 * yc[e].1;
    }
    let ridge = 1e-12 * (g[0][0] + g[1][1] + g[2][2]).max(f64::MIN_POSITIVE);
    let objective = |c: &[f64; 3]| {
        let mut f = bb;
        for j in 0..3 {
            f -= 2.0 * h[j] * c[j];
            for k in 0..3 {
                f += c[j] * g[j][k] * c[k];
            }
        }
        f
    };
    let mut best: Option<([f64; 3], f64)> = None;
    // Every non-empty support: the face where the other amplitudes are 0.
    for support in 1u8..8 {
        let idx: Vec<usize> = (0..3).filter(|j| support & (1 << j) != 0).collect();
        let m = idx.len();
        // KKT of min cᵀGc − 2hᵀc s.t. Σc = 1 on the support: [2G 1; 1ᵀ 0][c; λ] = [2h; 1].
        let mut a = [[0.0f64; 5]; 4];
        for (r, &j) in idx.iter().enumerate() {
            for (s, &k) in idx.iter().enumerate() {
                a[r][s] = 2.0 * (g[j][k] + if j == k { ridge } else { 0.0 });
            }
            a[r][m] = 1.0;
            a[m][r] = 1.0;
            a[r][m + 1] = 2.0 * h[j];
        }
        a[m][m + 1] = 1.0;
        let Some(sol) = solve_small(&mut a, m + 1) else { continue };
        let mut c = [0.0; 3];
        for (r, &j) in idx.iter().enumerate() {
            c[j] = sol[r];
        }
        if c.iter().any(|&v| v < 0.0 || !v.is_finite()) {
            continue;
        }
        let f = objective(&c);
        if best.is_none_or(|(_, fb)| f < fb) {
            best = Some((c, f));
        }
    }
    best.map_or(q.c, |(c, _)| c)
}

/// Gaussian elimination with partial pivoting on the `n × (n+1)` augmented system in `a`, its
/// right-hand side in column `n` (`n ≤ 4`); `None` if it is singular.
fn solve_small(a: &mut [[f64; 5]; 4], n: usize) -> Option<[f64; 4]> {
    for col in 0..n {
        let piv = (col..n).max_by(|&r, &s| a[r][col].abs().total_cmp(&a[s][col].abs()))?;
        if a[piv][col].abs() < 1e-300 {
            return None;
        }
        a.swap(col, piv);
        for r in col + 1..n {
            let f = a[r][col] / a[col][col];
            for k in col..=n {
                a[r][k] -= f * a[col][k];
            }
        }
    }
    let mut x = [0.0; 4];
    for r in (0..n).rev() {
        let mut v = a[r][n];
        for k in r + 1..n {
            v -= a[r][k] * x[k];
        }
        x[r] = v / a[r][r];
    }
    Some(x)
}

/// Stage 5.2 of the paper, exactly ([`DecomposeStageSolver::LsqlinLog`]): the R2*₀ ≥ 0
/// minimising the log residual, `Σt(L − ℓ)/Σt²` clamped at 0 (see the variant's docs). `None`
/// when the model's log magnitude is not finite at some echo (a zero signal), leaving R2*₀ as it
/// was.
fn r2star_log_linear(te: &[f64], logyc: &[(f64, f64)], q: &Fit, b0: f64) -> Option<f64> {
    let q0 = Fit { r0: 0.0, ..*q };
    let (mut num, mut den) = (0.0, 0.0);
    for (e, &t) in te.iter().enumerate() {
        let z = signal_model(t, &q0, b0);
        let l = fdlibm::log_abs(z.0, z.1);
        if !l.is_finite() {
            return None;
        }
        num += t * (l - logyc[e].0);
        den += t * t;
    }
    Some((num / den).max(0.0))
}

/// MATLAB's complex `log(z)` as `(log|z|, angle z)`: FDLIBM's `atan2` and MATLAB's rule for the
/// real part ([`fdlibm::log_abs`]), bit for bit.
#[inline]
fn matlab_log(z: (f64, f64)) -> (f64, f64) {
    (fdlibm::log_abs(z.0, z.1), fdlibm::atan2(z.1, z.0))
}

/// `C · exp(−(decay + i·freq)·t)` evaluated as MATLAB does it: `exp` of the complex exponent
/// `(−decay·t, −freq·t)` is `e^{re}·(cos im, sin im)`, then scaled by `C`.
#[inline]
fn compartment(amp: f64, decay: f64, freq: f64, t: f64) -> (f64, f64) {
    let e = fdlibm::exp(-decay * t);
    let im = -freq * t;
    (amp * (e * fdlibm::cos(im)), amp * (e * fdlibm::sin(im)))
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
    let t3 = p.c[2] * fdlibm::exp(-p.r0 * t);
    (t1.0 + t2.0 + t3, t1.1 + t2.1)
}

/// The authors' `pscModel.m`: the `C_plus` compartment, the rest at the baseline decay.
#[inline]
fn psc_model(t: f64, p: &Fit, b0: f64) -> (f64, f64) {
    let a = broadening(b0);
    let t1 = compartment(p.c[0], a * p.chi[0] + p.r0, (2.0 / 3.0) * p.chi[0] * GAMMA * b0, t);
    let t3 = (p.c[2] + p.c[1]) * fdlibm::exp(-p.r0 * t);
    (t1.0 + t3, t1.1)
}

/// The authors' `dscModel.m`: the `C_minus` compartment with its frequency negated
/// (`(2/3)·(−χ−)·γ·B0`), the rest at the baseline decay.
#[inline]
fn dsc_model(t: f64, p: &Fit, b0: f64) -> (f64, f64) {
    let a = broadening(b0);
    let t2 = compartment(p.c[1], -a * p.chi[1] + p.r0, (2.0 / 3.0) * -p.chi[1] * GAMMA * b0, t);
    let t3 = (p.c[2] + p.c[0]) * fdlibm::exp(-p.r0 * t);
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
        let raw = fdlibm::atan2(z.1, z.0);
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
            chi_bound: DecomposeChiBound::Ppm(0.5),
            lsqcurvefit: LsqTrrOptions::default(),
            unwrap_phase: false,
            solvers: Default::default(),
            normalisation: Default::default(),
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
                (mag[e] / gmax * fdlibm::cos(ph), -(mag[e] / gmax * fdlibm::sin(ph)))
            })
            .collect();
        let logyc: Vec<(f64, f64)> = yc.iter().map(|&z| matlab_log(z)).collect();
        let mut ws = StageSolvers::new(2 * ne);
        let p = fit_voxel(&te, &yc, &logyc, b0, 0.5, &params, &mut ws);
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
    /// non-zero and increasing, not a dropout, with the configuration the 7 T rules suggest
    /// (`unwrap_phase` and the echo-spacing χ bound; unwrapping with the authors' 0.5 ppm bound
    /// fails here, as on the phantom: module docs, *At 7 T*).
    #[test]
    fn chi_pos_survives_the_phase_wrap_threshold() {
        let te = [0.004, 0.012, 0.020, 0.028]; // 7 T phantom sampling
        let b0 = 7.0;
        let params = DecomposeParams {
            b0,
            n_inner: 10,
            chi_bound: DecomposeChiBound::EchoSpacing,
            lsqcurvefit: LsqTrrOptions::default(),
            unwrap_phase: true,
            solvers: Default::default(),
            normalisation: Default::default(),
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

    /// The 7 T configuration the pipeline rules suggest (`unwrap_phase` with the echo-spacing χ
    /// bound) recovers χ+ across the deep-grey range.
    ///
    /// Single-compartment data built the way the QSM-CI phantom builds it (magnitude decaying at
    /// R2* = R2 + Dr·|χ|, phase from χ_total), swept over χ+ 0.03–0.30 ppm; every voxel must
    /// recover at least 0.3 of its χ+. With the authors' 0.5 ppm bound unwrapping alone fails
    /// here, as on the phantom, where it leaves most of the pallidum collapsed and the
    /// echo-spacing bound (0.3145 ppm here) recovers it (module docs, *At 7 T*).
    #[test]
    fn chi_pos_has_no_dropouts_across_deep_grey() {
        let te = [0.004, 0.012, 0.020, 0.028];
        let b0 = 7.0;
        let params = DecomposeParams {
            b0,
            n_inner: 10,
            chi_bound: DecomposeChiBound::EchoSpacing,
            lsqcurvefit: LsqTrrOptions::default(),
            unwrap_phase: true,
            solvers: Default::default(),
            normalisation: Default::default(),
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
        let params = DecomposeParams { b0, n_inner: 4, chi_bound: DecomposeChiBound::Ppm(0.5), lsqcurvefit: LsqTrrOptions::default(), unwrap_phase: false, solvers: Default::default(), normalisation: Default::default() };

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

    /// The echo-spacing χ bound: `π / ((2/3)·γ·B0·ΔTE_max)`, 0.3145 ppm at 7 T with 8 ms spacing
    /// (the first echo, 4 ms, is a smaller step), set by the largest step (`TE₁` counts), and an
    /// error where it is not a finite positive bound.
    #[test]
    fn echo_spacing_chi_bound() {
        let b = |b0: f64, te: &[f64]| DecomposeChiBound::EchoSpacing.resolve(b0, te);
        let at7 = b(7.0, &[0.004, 0.012, 0.020, 0.028]).unwrap();
        assert!((at7 - 0.31447).abs() < 1e-4, "{at7}");
        assert_eq!(at7, PI / ((2.0 / 3.0) * GAMMA * 7.0 * 0.008));
        // An irregular spacing: the largest step sets it.
        assert_eq!(b(7.0, &[0.004, 0.008, 0.020]).unwrap(), PI / ((2.0 / 3.0) * GAMMA * 7.0 * 0.012));
        // A first echo later than the spacing: TE₁ sets it.
        assert_eq!(b(3.0, &[0.010, 0.013, 0.016]).unwrap(), PI / ((2.0 / 3.0) * GAMMA * 3.0 * 0.010));
        assert!(b(7.0, &[]).is_err() && b(7.0, &[0.0, 0.0, 0.0]).is_err() && b(0.0, &[0.004, 0.008]).is_err());
        assert_eq!(DecomposeChiBound::Ppm(0.2).resolve(7.0, &[]).unwrap(), 0.2);
        assert_eq!(DecomposeParams::default().chi_bound, DecomposeChiBound::Ppm(0.5));
        assert_eq!((DecomposeChiBound::Ppm(0.5).id(), DecomposeChiBound::EchoSpacing.id()), ("ppm", "echo-spacing"));
    }

    /// #199: a negative `chi_bound` panicked inside the fit; it is an error now, as are the
    /// shape problems that used to be asserts.
    #[test]
    fn negative_chi_bound_and_bad_shapes_are_errors() {
        let te = [0.004, 0.008, 0.012];
        let mag = [1.0, 0.9, 0.8];
        for bad in [-0.1, f64::NAN] {
            let params = DecomposeParams { chi_bound: DecomposeChiBound::Ppm(bad), ..Default::default() };
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
        let mut bit_same = [0usize; 3];
        let only = std::env::var("DECOMPOSE_ONLY").ok();
        for call in calls.iter().filter(|c| c.stage.is_some() && only.as_deref().is_none_or(|o| o == c.tag)) {
            let (v, _, st, fx) = call.stage.unwrap();
            let yc = yv(v);
            let logyc: Vec<(f64, f64)> = yc.iter().map(|&z| matlab_log(z)).collect();
            let q = Fit {
                c: [fx[3], fx[4], fx[5]],
                r0: fx[2],
                chi: [fx[0], fx[1]],
            };
            let log_resid = |q: &Fit, out: &mut [f64]| {
                for e in 0..ne {
                    let (lr, la) = matlab_log(signal_model(te[e], q, b0));
                    out[e] = lr - logyc[e].0;
                    out[ne + e] = -la - -logyc[e].1;
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
            if d.first_bit.is_none() {
                bit_same[st - 1] += 1;
            }
            if std::env::var("DECOMPOSE_BITS").is_ok() {
                eprintln!("BITS {} {} {:?} {}", d.tag, st, d.first_bit, good);
            }
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
        eprintln!("bit-identical throughout (every evaluation point and sum of squares) by stage: {bit_same:?}");
    }

    /// Stage 5.1 solved exactly: on noise-free data whose amplitudes sum to one it returns them,
    /// and on any data its residual is no larger than at any point of a fine grid on the simplex.
    #[test]
    fn paper_amplitudes_are_the_simplex_minimum() {
        let te = [0.003, 0.006, 0.009, 0.012, 0.015, 0.018];
        let b0 = 3.0;
        let q = Fit { c: [0.0; 3], r0: 20.0, chi: [0.12, -0.08] };
        let truth = [0.25, 0.15, 0.6];
        let yc: Vec<(f64, f64)> = te.iter().map(|&t| signal_model(t, &Fit { c: truth, ..q }, b0)).collect();
        let c = amplitudes_sum_to_one(&te, &yc, &q, b0);
        for j in 0..3 {
            assert!((c[j] - truth[j]).abs() < 1e-9, "{c:?}");
        }
        // Data outside the model (scaled up and with a phase ramp): compare with a grid search.
        let yc: Vec<(f64, f64)> = yc.iter().enumerate()
            .map(|(e, &(re, im))| { let (s, k) = (1.7, 0.3 * e as f64); (s * (re * k.cos() - im * k.sin()), s * (re * k.sin() + im * k.cos())) })
            .collect();
        let resid = |c: [f64; 3]| -> f64 {
            te.iter().zip(&yc).map(|(&t, y)| { let z = signal_model(t, &Fit { c, ..q }, b0); (z.0 - y.0).powi(2) + (z.1 - y.1).powi(2) }).sum()
        };
        let c = amplitudes_sum_to_one(&te, &yc, &q, b0);
        assert!((c.iter().sum::<f64>() - 1.0).abs() < 1e-12 && c.iter().all(|&v| v >= 0.0), "{c:?}");
        let n = 200;
        for i in 0..=n {
            for j in 0..=n - i {
                let g = [i as f64 / n as f64, j as f64 / n as f64, (n - i - j) as f64 / n as f64];
                assert!(resid(c) <= resid(g) + 1e-12, "{c:?} worse than {g:?}");
            }
        }
    }

    /// Stage 5.2 solved exactly: the closed form is the minimum of the log residual over R2*₀ ≥ 0.
    #[test]
    fn paper_r2star_is_the_log_residual_minimum() {
        let te = [0.003, 0.006, 0.009, 0.012, 0.015, 0.018];
        let b0 = 3.0;
        let q = Fit { c: [0.3, 0.2, 0.5], r0: 0.0, chi: [0.1, -0.05] };
        let logyc: Vec<(f64, f64)> = te.iter().enumerate()
            .map(|(e, &t)| { let z = signal_model(t, &Fit { r0: 31.0, ..q }, b0); let l = matlab_log(z); (l.0 + 0.01 * (e % 2) as f64, l.1) })
            .collect();
        let resid = |r0: f64| -> f64 {
            let mut out = vec![0.0; 2 * te.len()];
            for (e, &t) in te.iter().enumerate() {
                let l = matlab_log(signal_model(t, &Fit { r0, ..q }, b0));
                out[e] = l.0 - logyc[e].0;
                out[te.len() + e] = l.1 - logyc[e].1;
            }
            out.iter().map(|v| v * v).sum()
        };
        let r0 = r2star_log_linear(&te, &logyc, &q, b0).unwrap();
        assert!((r0 - 31.0).abs() < 2.0, "{r0}");
        for k in 0..2000 {
            let g = k as f64 * 0.05;
            assert!(resid(r0) <= resid(g) + 1e-12, "{r0} worse than {g}");
        }
        // Data decaying slower than the model at R2*₀ = 0 clamps at the bound.
        let rising: Vec<(f64, f64)> = logyc.iter().zip(&te).map(|(&(lr, la), &t)| (lr + 40.0 * t, la)).collect();
        assert_eq!(r2star_log_linear(&te, &rising, &q, b0), Some(0.0));
    }

    /// The paper's fit runs end to end, and a solver on a stage it cannot fit is refused.
    #[test]
    fn paper_solvers_run_and_are_validated() {
        let te = [0.0027, 0.0056, 0.0085, 0.0114, 0.0143, 0.0172];
        let b0 = 3.0;
        let truth = Fit { c: [0.2, 0.1, 0.7], r0: 18.0, chi: [0.1, -0.05] };
        let mag: Vec<f64> = te.iter().map(|&t| { let z = signal_model(t, &truth, b0); 800.0 * (z.0 * z.0 + z.1 * z.1).sqrt() }).collect();
        let chi = per_echo(&[0.03], te.len());
        let params = DecomposeParams { b0, ..DecomposeParams::paper() };
        let (pos, neg, _) = decompose(&chi, &mag, &te, &[1u8], &params, |_, _| {}).unwrap();
        assert!(pos[0].is_finite() && neg[0].is_finite() && pos[0] >= 0.0 && neg[0] <= 0.0, "{pos:?} {neg:?}");

        use DecomposeStageSolver as S;
        for solvers in [
            DecomposeStageSolvers { amplitudes: S::LsqlinLog, ..DecomposeStageSolvers::paper() },
            DecomposeStageSolvers { r2star: S::LsqlinSumToOne, ..DecomposeStageSolvers::paper() },
            DecomposeStageSolvers { chi: S::LsqlinLog, ..DecomposeStageSolvers::paper() },
        ] {
            assert!(DecomposeParams { solvers, ..DecomposeParams::paper() }.validate().is_err(), "{solvers:?}");
        }
    }

    /// The per-voxel scales: the first echo, and S(0) extrapolated from the first two echoes
    /// (falling back to the first echo where they do not decay).
    #[test]
    fn normalisation_divisors() {
        let te = [0.005, 0.010, 0.015];
        let m = [600.0, 300.0, 150.0];
        assert_eq!(DecomposeNormalisation::GlobalMax.divisor(&m, &te, 1000.0), 1000.0);
        assert_eq!(DecomposeNormalisation::FirstEcho.divisor(&m, &te, 1000.0), 600.0);
        assert!((DecomposeNormalisation::ExtrapolatedS0.divisor(&m, &te, 1000.0) - 1200.0).abs() < 1e-9);
        assert_eq!(DecomposeNormalisation::ExtrapolatedS0.divisor(&[300.0, 600.0, 1.0], &te, 1000.0), 300.0);
    }

}

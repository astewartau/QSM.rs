//! Pipeline configuration types
//!
//! Defines the configuration structs and enums for the QSM pipeline runner.
//! These are pure algorithm config types (no serde). Consumers that need
//! serialization (e.g. qsmxt-config) provide their own serde wrappers
//! and convert to these types.

use crate::bgremove::{
    IsmvParams, LbvParams, PdfParams, ResharpParams, SharpParams, SdfParams, VsharpParams,
    HarperellaParams, IharperellaParams,
};
use crate::inversion::{
    IlsqrParams, MediParams, NltvParams, RtsParams, TgvParams, TikhonovParams, TkdParams, TvParams,
    NdiParams, FansiParams, L1QsmParams, WhQsmParams, HdQsmParams, TfiParams, AmpPeParams,
    LsqrQsmParams, HeidiParams, StarParams,
};
use crate::separation::{
    ChiSepIlsqrParams, ChiSepNetParams, ChiSepParams, DecomposeParams, HcChisepParams,
    R2starQsmParams, SusepNetParams, WaveSepParams,
};
use crate::unwrap::romeo::RomeoParams;
use crate::unwrap::LaplacianSolver;
use crate::utils::multi_echo::{B0WeightType, EchoQuality, FitPpmComplexParams, LinearFitParams, RobustFitParams};
use crate::utils::QsmartParams;


// =========================================================================
// Selection enums
// =========================================================================

/// Phase unwrapping algorithm
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UnwrappingAlgorithm {
    Romeo,
    Laplacian,
}

/// Whether an algorithm is valid when B0 does not lie along the voxel `+z` axis.
///
/// The dipole relationship is direction-dependent, and the FFT that implements it lives in the
/// voxel grid, so an oblique acquisition has to be handled deliberately. There are three cases,
/// and the difference matters because getting it wrong is silent — the reconstruction completes
/// and the numbers are simply wrong.
///
/// Every algorithm enum reports its case through `orientation_support()`. Hosts should check
/// [`OrientationSupport::requires_axial`] before running an oblique dataset. See
/// [`crate::geometry`] for the B0 direction and for resampling to a cardinal grid,
/// [`crate::registration`] for recovering the rigid transform between volumes (which is how a
/// multi-orientation set gets onto one grid, and where its per-orientation B0 directions come
/// from, since the object rotates and B0 does not), and [`crate::motion`] for the same within
/// one acquisition, which reports one B0 direction per volume.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrientationSupport {
    /// The B0 direction is an explicit parameter — the dipole kernel is built from it — so the
    /// method reconstructs correctly on the grid the data was acquired on, whatever its
    /// orientation. See [`crate::geometry::b0_direction_from_affine`].
    Arbitrary,
    /// The method never uses B0. The SMV-family background removals rest on the spherical mean
    /// value property of harmonic fields, which is rotation-invariant.
    NotApplicable,
    /// Assumes B0 along `+z` and offers no way to say otherwise. Deep-learning methods learned
    /// the dipole relationship from axially-acquired training data, so there is no direction to
    /// rotate; oblique data must be resampled to a cardinal grid first
    /// ([`crate::geometry::resample_complex_to_axial`]).
    ///
    /// Unrolled networks with a physics data-consistency term (LPCNN, MoDL-QSM, NeXtQSM) build
    /// that term from the true direction, so they are partly corrected — but their learned prior
    /// is still axial, so they belong here.
    AxialOnly,
}

impl OrientationSupport {
    /// Whether oblique data must be resampled before this method can be trusted.
    pub fn requires_axial(&self) -> bool {
        matches!(self, OrientationSupport::AxialOnly)
    }
}

/// What a dipole inversion reconstructs from, and so which earlier stages a host has to run.
///
/// Reported by [`InversionConfig::input`], the one place this is decided. A host should route
/// from it rather than keep its own lists of methods that skip background removal.
///
/// The pipeline's own background-removal methods ([`BgRemovalAlgorithm`]) all take the total
/// field in ppm and produce a local field, and so does iQFM ([`run_iqfm`](super::run_iqfm)) from
/// wrapped phase. Each of them is therefore only useful in front of a
/// [`InversionInput::LocalField`] inversion: with the others it would have no effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InversionInput {
    /// The background-removed local field in ppm: field mapping, then background removal,
    /// then [`run_dipole_inversion`](super::run_dipole_inversion) with the eroded mask the
    /// background removal returned.
    LocalField,
    /// The total field in ppm, straight from field mapping with **no** background removal,
    /// and the brain mask. The method removes or models the background itself.
    TotalField,
    /// The per-echo wrapped phase (and magnitude): no field mapping and no background removal.
    WrappedPhase,
}

impl InversionInput {
    /// Whether a background-removal stage (or iQFM) belongs in front of the inversion.
    pub fn needs_bg_removal(&self) -> bool {
        matches!(self, InversionInput::LocalField)
    }

    /// Whether the echoes have to be field-mapped (unwrapped and combined) first.
    pub fn needs_field_mapping(&self) -> bool {
        !matches!(self, InversionInput::WrappedPhase)
    }
}

/// Background field removal algorithm
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BgRemovalAlgorithm {
    Vsharp,
    Pdf,
    Lbv,
    Ismv,
    Sharp,
    Resharp,
    Harperella,
    Iharperella,
    /// BFRnet deep-learning background removal (requires the `onnx` feature and
    /// the `bfrnet` model weights; see [`crate::models`]).
    Bfrnet,
}

/// Dipole inversion algorithm
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InversionAlgorithm {
    Tkd,
    Tsvd,
    Tikhonov,
    Tv,
    Rts,
    Nltv,
    Medi,
    /// Preconditioned Total Field Inversion (single-step, total field).
    Tfi,
    Ilsqr,
    /// STAR-QSM (STI Suite 3.0's `QSM_star`): k-space initial guess refined by SpaRSA.
    Star,
    Tgv,
    Qsmart,
    /// Nonlinear Dipole Inversion (FANSI ndi.m).
    Ndi,
    /// FANSI nonlinear Total Variation (nlTV).
    Fansi,
    /// FANSI nonlinear Total Generalized Variation (nlTGV).
    FansiTgv,
    /// L1 data-fidelity QSM (FANSI nlL1TV / PI-QSM).
    L1qsm,
    /// Weak-Harmonic QSM (FANSI WH_nlTV).
    Whqsm,
    /// Hybrid two-stage L1→L2 QSM (HD-QSM).
    Hdqsm,
    /// Approximate Message Passing with built-in Parameter Estimation (AMP-PE).
    AmpPe,
    /// Minimally regularised LSQR with residual-field and global-offset unknowns
    /// (Schweser 2010).
    Lsqr,
    /// Homogeneity Enabled Incremental Dipole Inversion (Schweser 2012). Seeds
    /// itself from [`InversionAlgorithm::Lsqr`], then fills the dipole cone under
    /// a field-derived weighted-TV prior.
    Heidi,
    /// xQSM deep-learning dipole inversion (requires the `onnx` feature and the
    /// `xqsm` model weights; see [`crate::models`]).
    Xqsm,
    /// QSMnet deep-learning dipole inversion (requires the `onnx` feature and the
    /// `qsmnet` model weights; see [`crate::models`]).
    Qsmnet,
    /// QSMnet+ deep-learning dipole inversion (susceptibility-scaling augmented;
    /// requires the `onnx` feature and the `qsmnet-plus` weights).
    QsmnetPlus,
    /// AutoQSM single-step reconstruction (requires the `onnx` feature and the
    /// `autoqsm` weights). NOTE: takes the **total** field — it does its own
    /// background removal, so the `local_field_ppm` argument should be the total field.
    Autoqsm,
    /// QSMGAN deep-learning dipole inversion (local field → χ; `onnx` + `qsmgan` weights).
    Qsmgan,
    /// IR2QSM unrolled deep-learning dipole inversion (local field → χ; `onnx` + `ir2qsm` weights).
    Ir2qsm,
    /// LPCNN learned-proximal deep-learning dipole inversion (local field → χ; `onnx` + `lpcnn` weights).
    Lpcnn,
    /// MoDL-QSM model-based deep-learning dipole inversion (local field → χ33/STI component;
    /// `onnx` + `modl-qsm` weights).
    ModlQsm,
    /// NeXtQSM single-step reconstruction (requires `onnx` + the two `nextqsm` weight files).
    /// NOTE: takes the **total** field — it does its own background removal.
    Nextqsm,
    /// iQSM single-step reconstruction from wrapped **phase** (end-to-end: joint unwrap +
    /// background removal + inversion). Not a dipole-inversion-stage option — use
    /// [`super::run_iqsm`]. Rejected by `run_dipole_inversion`.
    Iqsm,
    /// iQSM+ single-step reconstruction from wrapped **phase** with orientation-adaptive
    /// feature editing (uses the B0 direction). End-to-end; use [`super::run_iqsm_plus`].
    IqsmPlus,
}

/// How [`run_field_mapping`](super::run_field_mapping) combines the unwrapped echoes into one
/// B0 map. Each method carries exactly the settings it reads, so a weighting cannot be set on a
/// fit that ignores it.
///
/// Only multi-echo data is combined. A single echo is `φ/TE` whatever is chosen here, except
/// that [`Self::RobustFit`] refuses a series too short for it (see [`RobustFitParams::min_echoes`])
/// and [`Self::ComplexFit`] needs the echo's magnitude (it is also `φ/TE` there, as in MEDI).
#[derive(Clone, Debug, PartialEq)]
pub enum B0EstimationMethod {
    /// Weighted mean of the per-echo frequencies, `B0 = Σ wₑ (φₑ/TEₑ) / Σ wₑ`, with `wₑ` from
    /// `weighting` (default [`B0WeightType::PhaseSNR`]). The default method.
    WeightedAvg {
        /// Per-echo weight; [`B0WeightType::AssumedDecay`] needs a finite, positive T2*.
        weighting: B0WeightType,
    },
    /// Magnitude-weighted least-squares fit of phase against TE
    /// ([`multi_echo_linear_fit`](crate::utils::multi_echo_linear_fit)). The weight is always the
    /// magnitude; there is no `B0WeightType` here.
    LinearFit(LinearFitParams),
    /// Motion-robust echo combination
    /// ([`multi_echo_robust_fit`](crate::utils::multi_echo_robust_fit)): the same
    /// magnitude-weighted fit as [`Self::LinearFit`], iteratively reweighted so that an echo which
    /// departs from a straight line in TE at a voxel stops pulling the slope there. The weight is
    /// magnitude × robust weight; there is no `B0WeightType`, because the robust scale is measured
    /// on magnitude-standardised residuals and another weighting would break that.
    ///
    /// Needs at least [`RobustFitParams::min_echoes`] echoes (4 with `estimate_offset`, else 3);
    /// fewer is [`PipelineError::InvalidConfig`] rather than a silent plain fit. The per-echo
    /// report is returned in [`FieldMappingResult::robust_fit`].
    RobustFit(RobustFitParams),
    /// MEDI's field map: the complex fit of `Fit_ppm_complex.m`
    /// ([`fit_ppm_complex`](crate::utils::fit_ppm_complex)) to the wrapped echoes, then spatial
    /// unwrapping of the fitted phase per echo spacing with `unwrapping_algorithm` (MEDI uses its
    /// region-growing `unwrapPhase`, which is not ported; ROMEO is given MEDI's `iMag`, the
    /// root-sum-of-squares of the echo magnitudes), then division by the echo spacing.
    ///
    /// This replaces the per-echo path rather than combining unwrapped echoes: the fit has its own
    /// phase offset (`p0`), so `phase_offset_removal` does not apply and is not run;
    /// `bipolar_correction` is refused (MEDI's `Fit_ppm_complex_bipolar` is not ported). Needs the
    /// magnitude of every echo and evenly spaced echoes. Also returns the field's noise map
    /// ([`FieldMappingResult::field_noise_ppm`], MEDI's `N_std`).
    ComplexFit(FitPpmComplexParams),
    /// QSMART's echo fit (`echofit.m`, [`qsmart_echofit`](crate::utils::qsmart_echofit)): the
    /// magnitude-weighted fit of phase against TE through the origin, over every voxel of the
    /// unwrapped echoes, with no per-echo constant removed after Laplacian unwrapping (with ROMEO,
    /// whole turns between echoes are aligned, as for the weighted mean). Also returns QSMART's
    /// reliability map `R_0` ([`FieldMappingResult::reliability`]): voxels whose fit residual,
    /// box-filtered over ~1 mm, reaches `fit_threshold` (by default the 99th percentile of the
    /// blurred residual, as QSMART's authors run it; see
    /// [`QsmartFitThreshold`](crate::utils::QsmartFitThreshold)). The field is converted to ppm
    /// with QSMART's γ = 2.675e8 rad/s/T.
    ///
    /// The echoes come from the configured unwrapping, phase-offset removal and bipolar
    /// correction. QSMART unwraps each echo with Laplacian unwrapping on the raw phase
    /// (`unwrap_phase.m`: [`LaplacianSolver::Mirror`],
    /// [`FieldMappingConfig::laplacian_unmasked`]) and does neither of the others; with those
    /// settings this is QSMART's field map, bit for bit. The rules `qsmart.unwrap-laplacian`,
    /// `qsmart.laplacian-solver` and `qsmart.unwrap-unmasked` suggest them for a QSMART run.
    QsmartEchofit {
        /// Residual threshold for `R_0`
        fit_threshold: crate::utils::QsmartFitThreshold,
    },
}

impl Default for B0EstimationMethod {
    fn default() -> Self {
        B0EstimationMethod::WeightedAvg { weighting: B0WeightType::PhaseSNR }
    }
}

/// QSM referencing method
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QsmReference {
    Mean,
    None,
}

// =========================================================================
// Masking types
// =========================================================================

/// Input data source for mask generation
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MaskingInput {
    MagnitudeFirst,
    Magnitude,
    MagnitudeLast,
    PhaseQuality,
}

/// Threshold method for mask generation
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MaskThresholdMethod {
    Otsu,
    Fixed,
    Percentile,
}

/// A single mask operation (generator or refinement)
#[derive(Clone, Debug, PartialEq)]
pub enum MaskOp {
    Threshold {
        method: MaskThresholdMethod,
        value: Option<f64>,
    },
    /// BET brain extraction from the **magnitude** (a generator), with the full
    /// [`BetParams`](crate::bet::BetParams) — fractional intensity, smoothness, gradient
    /// threshold, iterations, subdivisions and voxel scale. `BetParams::default()` is FSL's
    /// defaults on the acquisition's own geometry.
    Bet(crate::bet::BetParams),
    Erode {
        iterations: usize,
    },
    Dilate {
        iterations: usize,
    },
    Close {
        radius: usize,
    },
    FillHoles {
        max_size: usize,
    },
    GaussianSmooth {
        sigma_mm: f64,
    },
    /// Signal-gated erosion: peel only low-signal boundary voxels (skull-base / sinus dropout)
    /// down to a depth cap. See [`crate::utils::morphology::signal_gated_erosion`].
    ///
    /// Gates on the **magnitude** image, never on the section's input: it divides out a
    /// receive-coil bias estimate and compares against the in-mask median, which only means
    /// anything for magnitude. Errors if no magnitude is supplied, so a phase-quality mask input
    /// can still be refined with it as long as the magnitude is available.
    SignalErode(crate::utils::morphology::SignalErosionParams),
    /// HD-BET deep-learning brain extraction from the **magnitude** (a generator, like `Bet`),
    /// whatever the section's input is. Requires the `onnx` feature and the `hd-bet` model
    /// weights; see [`crate::models`].
    HdBet(crate::bet::HdBetParams),
    /// RS2-Net deep-learning **rodent** brain extraction from the **magnitude** (a generator,
    /// like `HdBet`). Requires the `onnx` feature and the `rs2-net` model weights; see
    /// [`crate::models`].
    Rs2Net(crate::bet::Rs2NetParams),
    /// MindGrab deep-learning brain extraction from the **magnitude** (a generator, like
    /// `HdBet`), for any contrast. Needs [`ScanMetadata::affine`], the `onnx` feature and the
    /// `mindgrab` model weights; see [`crate::models`].
    MindGrab(crate::bet::MindGrabParams),
}

impl MaskOp {
    /// Registry id of the deep-learning model this op runs, if any.
    pub fn dl_model_id(&self) -> Option<&'static str> {
        match self {
            Self::HdBet(_) => Some("hd-bet"),
            Self::Rs2Net(_) => Some("rs2-net"),
            Self::MindGrab(_) => Some("mindgrab"),
            Self::Threshold { .. } | Self::Bet(_) | Self::Erode { .. } | Self::Dilate { .. }
            | Self::Close { .. } | Self::FillHoles { .. } | Self::GaussianSmooth { .. }
            | Self::SignalErode(_) => None,
        }
    }
}

/// A mask section: input source + generator + refinements
#[derive(Clone, Debug, PartialEq)]
pub struct MaskSection {
    pub input: MaskingInput,
    pub generator: MaskOp,
    pub refinements: Vec<MaskOp>,
}

impl MaskSection {
    /// Get all operations (generator + refinements) in order
    pub fn all_ops(&self) -> Vec<MaskOp> {
        let mut ops = vec![self.generator.clone()];
        ops.extend(self.refinements.iter().cloned());
        ops
    }
}

// =========================================================================
// Per-stage config structs
// =========================================================================

/// Masking configuration
#[derive(Clone, Debug)]
pub struct MaskingConfig {
    pub inhomogeneity_correction: bool,
    pub homogeneity_sigma_mm: f64,
    pub homogeneity_nbox: usize,
    pub sections: Vec<MaskSection>,
}

impl Default for MaskingConfig {
    fn default() -> Self {
        Self {
            inhomogeneity_correction: true,
            homogeneity_sigma_mm: 7.0,
            homogeneity_nbox: 15,
            sections: vec![MaskSection {
                input: MaskingInput::PhaseQuality,
                generator: MaskOp::Threshold {
                    method: MaskThresholdMethod::Otsu,
                    value: None,
                },
                refinements: vec![
                    MaskOp::Dilate { iterations: 1 },
                    MaskOp::FillHoles { max_size: 0 },
                    MaskOp::Erode { iterations: 1 },
                ],
            }],
        }
    }
}

/// Field mapping configuration
#[derive(Clone, Debug)]
pub struct FieldMappingConfig {
    pub unwrapping_algorithm: UnwrappingAlgorithm,
    pub phase_offset_removal: bool,
    /// Gaussian smoothing sigma of the MCPC-3D-S phase offsets, in **mm** per axis (default
    /// [7, 7, 7]); divided by the voxel size to give the smoothing in voxels
    /// ([`Self::phase_offset_sigma_voxels`]). The default is the ROMEO command line's (what
    /// users run, `romeo -B`): `--phase-offset-smoothing-sigma-mm`, "default: [7,7,7]", divided
    /// by the NIfTI voxel size (ROMEO.jl 08776d4, `ext/RomeoApp/argparse.jl` l. 79-84 and
    /// 217-224, `caller.jl` l. 137-138). The same physical width at every resolution.
    /// MriResearchTools' `mcpc3ds` function itself defaults to `sigma = [10, 10, 5]` in voxels
    /// (`src/mcpc3ds.jl` l. 39), which QSM.rs used before; that is 10/10/5 mm at 1 mm voxels but
    /// a different width at any other resolution.
    ///
    /// Measured on 1 mm data (ROMEO field, default V-SHARP, RTS), where the old default is
    /// 10/10/5 mm: χ moves by at most 0.003 XSIM, on the qsm-forward phantom XSIM 0.254 → 0.256,
    /// NRMSE 85.9 → 85.7 %, and on an in-vivo 3T Philips scan, agreement with the UK Biobank
    /// pipeline's χ, XSIM 0.714 → 0.714, NRMSE 69.7 → 69.4 %.
    pub phase_offset_sigma_mm: [f64; 3],
    pub bipolar_correction: bool,
    /// How the unwrapped echoes are combined, with that method's own settings (default
    /// [`B0EstimationMethod::WeightedAvg`] with [`B0WeightType::PhaseSNR`]).
    pub b0_estimation: B0EstimationMethod,
    /// Poisson solver used when `unwrapping_algorithm` is [`UnwrappingAlgorithm::Laplacian`]
    /// (default [`LaplacianSolver::Dct`]). With [`LaplacianSolver::Fft`] `{ pad: [64; 3] }` and
    /// `b0_estimation` [`B0EstimationMethod::WeightedAvg`] with
    /// [`B0WeightType::AssumedDecay`] at 40 ms, a multi-echo run gives UK Biobank's field map
    /// (see [`run_field_mapping`](super::run_field_mapping)).
    pub laplacian_solver: LaplacianSolver,
    /// Laplacian unwrapping on the raw phase over the whole volume, instead of on the phase
    /// zeroed outside the brain mask (default `false`). Zeroing first keeps the noise outside the
    /// brain out of the global Poisson solve, and is what UK Biobank (`MRPhaseUnwrap(mask .*
    /// phase)`) does. QSM.jl's `unwrap_laplacian` takes the raw phase: its `:dct` solver (this
    /// crate's [`LaplacianSolver::Dct`]) ignores the mask, so it is `laplacian_unmasked: true`
    /// here, while its default `:mgpcg` (not ported) multiplies the Laplacian by the mask and
    /// solves with a Dirichlet boundary on it, which also removes the background (see
    /// [`crate::unwrap::laplacian`]). STI Suite 3.0's `QSM_GUI` and QSMART unwrap the raw phase
    /// (rules `sti.unwrap-unmasked`, `qsmart.unwrap-unmasked`; see [`crate::unwrap::laplacian`]).
    /// With [`B0EstimationMethod::QsmartEchofit`] the unwrapped echoes are also kept over the
    /// whole volume, as QSMART fits them; the other combinations use the voxels inside the mask.
    /// Only with Laplacian unwrapping.
    pub laplacian_unmasked: bool,
    pub romeo_params: RomeoParams,
}

/// The ROMEO command line's default phase-offset smoothing sigma, in mm (the default of
/// [`FieldMappingConfig::phase_offset_sigma_mm`]).
pub const ROMEO_CLI_PHASE_OFFSET_SIGMA_MM: [f64; 3] = [7.0, 7.0, 7.0];

impl FieldMappingConfig {
    /// [`phase_offset_sigma_mm`](Self::phase_offset_sigma_mm) in voxels of `voxel_size`
    /// (mm), as the ROMEO command line converts it.
    pub fn phase_offset_sigma_voxels(&self, voxel_size: (f64, f64, f64)) -> [f64; 3] {
        let s = self.phase_offset_sigma_mm;
        [s[0] / voxel_size.0, s[1] / voxel_size.1, s[2] / voxel_size.2]
    }
}

impl Default for FieldMappingConfig {
    fn default() -> Self {
        Self {
            unwrapping_algorithm: UnwrappingAlgorithm::Romeo,
            phase_offset_removal: true,
            phase_offset_sigma_mm: ROMEO_CLI_PHASE_OFFSET_SIGMA_MM,
            bipolar_correction: false,
            b0_estimation: B0EstimationMethod::default(),
            laplacian_solver: LaplacianSolver::Dct,
            laplacian_unmasked: false,
            romeo_params: RomeoParams::default(),
        }
    }
}

/// Background removal configuration
#[derive(Clone, Debug)]
pub struct BgRemovalConfig {
    pub algorithm: BgRemovalAlgorithm,
    pub vsharp: VsharpParams,
    pub pdf: PdfParams,
    pub lbv: LbvParams,
    pub ismv: IsmvParams,
    pub sharp: SharpParams,
    pub resharp: ResharpParams,
    pub harperella: HarperellaParams,
    pub iharperella: IharperellaParams,
    pub sdf: SdfParams,
    /// Overlap-tiling for BFRnet, as `(core, halo)` in voxels, as [`InversionConfig::tile`] is
    /// for the inversion networks. `None` (default) runs the whole volume, which on a whole-head
    /// scan does not fit a 32-bit (WASM) heap; `Some` runs it patch by patch, an approximation
    /// with bounded memory ([`bfrnet_tiled`](crate::bgremove::bfrnet_tiled)). Ignored by the
    /// classical methods.
    pub tile: Option<(usize, usize)>,
}

impl Default for BgRemovalConfig {
    fn default() -> Self {
        Self {
            algorithm: BgRemovalAlgorithm::Vsharp,
            vsharp: VsharpParams::default(),
            pdf: PdfParams::default(),
            lbv: LbvParams::default(),
            ismv: IsmvParams::default(),
            sharp: SharpParams::default(),
            resharp: ResharpParams::default(),
            harperella: HarperellaParams::default(),
            iharperella: IharperellaParams::default(),
            sdf: SdfParams::default(),
            tile: None,
        }
    }
}

/// Dipole inversion configuration
#[derive(Clone, Debug)]
pub struct InversionConfig {
    pub algorithm: InversionAlgorithm,
    pub tkd: TkdParams,
    pub tsvd: TkdParams,
    pub tikhonov: TikhonovParams,
    pub tv: TvParams,
    pub rts: RtsParams,
    pub nltv: NltvParams,
    pub medi: MediParams,
    pub tfi: TfiParams,
    pub ilsqr: IlsqrParams,
    /// STAR-QSM (`te` and `b0` are overridden from scan metadata by the dispatcher).
    pub star: StarParams,
    pub tgv: TgvParams,
    pub qsmart: QsmartParams,
    pub ndi: NdiParams,
    /// Shared by the `Fansi` (nlTV) and `FansiTgv` (nlTGV) algorithms; the
    /// dispatcher sets `is_tgv` from the selected algorithm.
    pub fansi: FansiParams,
    pub l1qsm: L1QsmParams,
    pub whqsm: WhQsmParams,
    pub hdqsm: HdQsmParams,
    /// AMP-PE (`b0` is overridden from scan metadata by the dispatcher).
    pub amp_pe: AmpPeParams,
    /// Minimally regularised LSQR (`b0` is overridden from scan metadata by the
    /// dispatcher). Also supplies HEIDI's seed map.
    pub lsqr: LsqrQsmParams,
    /// HEIDI cone filling.
    pub heidi: HeidiParams,
    /// Overlap-tiling for the deep-learning inversions, as `(core, halo)` in voxels. `None`
    /// (default) runs the net whole-volume; `Some` runs it patch-by-patch (bounded memory) via the
    /// `*_tiled` variants — an approximation, mainly for memory-constrained targets (e.g. WASM).
    /// Stored as a plain tuple so this config type stays available without the `onnx` feature;
    /// the dispatcher converts it to a [`crate::inversion::TileConfig`]. Ignored by the
    /// natively-patch-based nets (autoqsm/qsmgan) and by non-DL algorithms.
    pub tile: Option<(usize, usize)>,
}

impl InversionConfig {
    /// What the selected inversion reconstructs from, for data with `n_echoes` echoes. See
    /// [`InversionInput`].
    ///
    /// It is a property of the config rather than of [`InversionAlgorithm`] alone, because
    /// **TGV** depends on more than its name:
    ///
    /// - **TGV** ([`run_tgv`](super::run_tgv)) reconstructs multi-echo data from the
    ///   field-mapped total field ([`TgvInput::FieldPpm`](crate::inversion::TgvInput::FieldPpm),
    ///   expressed as phase at the first echo time) and a single echo from its wrapped phase
    ///   ([`TgvInput::WrappedPhase`](crate::inversion::TgvInput::WrappedPhase)). Either way it
    ///   does its own background removal.
    ///
    /// The other total-field methods are TFI (a total-field method by definition: it inverts
    /// the field over the whole FOV under a preconditioner), QSMART
    /// ([`run_qsmart`](super::run_qsmart), whose two SDF stages are its background removal),
    /// AutoQSM and NeXtQSM (networks with their own background removal). iQSM and iQSM+
    /// ([`run_iqsm`](super::run_iqsm), [`run_iqsm_plus`](super::run_iqsm_plus)) take wrapped
    /// phase. Every other method, including WH-QSM (which models only a *residual* harmonic
    /// field) and LSQR (a residual-field term), takes the local field.
    ///
    /// **MEDI** takes the local field with every option: plain MEDI, MEDI-SMV
    /// ([`MediParams::smv`]) and MEDI-mSMV ([`MediParams::msmv`]). That is how the MEDI authors
    /// run it. MEDI toolbox `README.m` runs `RDF = PDF(iFreq, N_std, Mask, ...)` and then
    /// `MEDI_L1('lambda',1000,'lambda_CSF',100,'merit','smv',5)` on `RDF`, which its naming
    /// convention defines as the local field. `MEDI_L1` reads `iFreq` but never uses it. The
    /// mSMV paper (Roberts et al. 2024, Eqs. 4 and 13) defines MEDI-SMV as the SMV high-pass of
    /// the *local* field after PDF, and SEPIA passes MEDI its local field. The SMV step removes
    /// the residual background that a background removal leaves near the brain edge. It is not
    /// a background removal itself. On the total field, MEDI-SMV is a different pipeline. With
    /// the default mask shrinking it gives nearly the same χ, but without shrinking it fails
    /// (see the MEDI module docs). The authors' pipeline uses PDF for the background removal
    /// ([`reference_bg_removal`](Self::reference_bg_removal)).
    pub fn input(&self, n_echoes: usize) -> InversionInput {
        use InversionAlgorithm as A;
        match self.algorithm {
            A::Tgv if n_echoes > 1 => InversionInput::TotalField,
            A::Tgv | A::Iqsm | A::IqsmPlus => InversionInput::WrappedPhase,
            A::Tfi | A::Qsmart | A::Autoqsm | A::Nextqsm => InversionInput::TotalField,
            A::Medi
            | A::Tkd
            | A::Tsvd
            | A::Tikhonov
            | A::Tv
            | A::Rts
            | A::Nltv
            | A::Ilsqr
            | A::Star
            | A::Ndi
            | A::Fansi
            | A::FansiTgv
            | A::L1qsm
            | A::Whqsm
            | A::Hdqsm
            | A::AmpPe
            | A::Lsqr
            | A::Heidi
            | A::Xqsm
            | A::Qsmnet
            | A::QsmnetPlus
            | A::Qsmgan
            | A::Ir2qsm
            | A::Lpcnn
            | A::ModlQsm => InversionInput::LocalField,
        }
    }

    /// The background removal the selected method's authors run in front of it, for a
    /// [`InversionInput::LocalField`] method whose authors' pipeline QSM.rs ports. Hosts can use
    /// it as the default background removal for that method.
    ///
    /// - MEDI (with or without SMV or mSMV): [`BgRemovalAlgorithm::Pdf`]. The MEDI toolbox's
    ///   `README.m` runs `PDF.m` (ported as [`BgRemovalAlgorithm::Pdf`], with the same
    ///   `Fit_ppm_complex` noise weighting) and gives `MEDI_L1` its result. The mSMV paper
    ///   (Roberts et al. 2024) also runs MEDI-SMV and MEDI-mSMV after PDF.
    ///
    /// `None` when no such pipeline is recorded for the method. It is then the host's choice.
    pub fn reference_bg_removal(&self) -> Option<BgRemovalAlgorithm> {
        match self.algorithm {
            InversionAlgorithm::Medi => Some(BgRemovalAlgorithm::Pdf),
            _ => None,
        }
    }
}

impl Default for InversionConfig {
    fn default() -> Self {
        Self {
            algorithm: InversionAlgorithm::Rts,
            tkd: TkdParams::default(),
            tsvd: TkdParams::default(),
            tikhonov: TikhonovParams::default(),
            tv: TvParams::default(),
            rts: RtsParams::default(),
            nltv: NltvParams::default(),
            medi: MediParams::default(),
            tfi: TfiParams::default(),
            ilsqr: IlsqrParams::default(),
            star: StarParams::default(),
            tgv: TgvParams::default(),
            qsmart: QsmartParams::default(),
            ndi: NdiParams::default(),
            fansi: FansiParams::default(),
            l1qsm: L1QsmParams::default(),
            whqsm: WhQsmParams::default(),
            hdqsm: HdQsmParams::default(),
            amp_pe: AmpPeParams::default(),
            lsqr: LsqrQsmParams::default(),
            heidi: HeidiParams::default(),
            tile: None,
        }
    }
}

/// Susceptibility source-separation algorithm (χ+ / χ−).
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SeparationAlgorithm {
    /// Shin 2021 projected-CG, QSM-initialized (local field + R2' + magnitude + QSM).
    ChiSepIlsqr,
    /// MEDI-based Gauss-Newton (local field + R2' + magnitude).
    ChiSepMedi,
    /// Closed-form from a QSM + R2* (Dimov 2022; R2* fit from magnitude if absent).
    R2starQsm,
    /// Wavelet-L1 proximal-gradient from a QSM + R2' (Fang 2023).
    WaveSep,
    /// Signal-domain per-voxel fit from a QSM + multi-echo magnitude (Chen 2021).
    Decompose,
    /// Hollow-cylinder fit from a QSM + R2' + multi-echo magnitude (Wharton & Bowtell).
    HcChisep,
    /// SUSEP-Net deep-learning separation from QSM + R2' + local field (requires
    /// the `onnx` feature and the `susep-net` weights; see [`crate::models`]).
    SusepNet,
    /// χ-sepnet (SNU-LIST) deep-learning separation from local field + QSM + R2'
    /// (requires the `onnx` feature and the `chi-sepnet` weights; see [`crate::models`]).
    ChiSepNet,
}

impl BgRemovalAlgorithm {
    /// How this method behaves when B0 is not along voxel `+z`.
    ///
    /// Only PDF projects onto dipole fields and therefore needs the direction; every other
    /// method here is harmonic/SMV-based and is direction-independent. BFRnet is a network, but
    /// background removal has no dipole orientation to get wrong in the way inversion does — it
    /// separates harmonic from non-harmonic content, so it is treated as direction-independent.
    pub fn orientation_support(&self) -> OrientationSupport {
        match self {
            BgRemovalAlgorithm::Pdf => OrientationSupport::Arbitrary,
            BgRemovalAlgorithm::Vsharp
            | BgRemovalAlgorithm::Lbv
            | BgRemovalAlgorithm::Ismv
            | BgRemovalAlgorithm::Sharp
            | BgRemovalAlgorithm::Resharp
            | BgRemovalAlgorithm::Harperella
            | BgRemovalAlgorithm::Iharperella
            | BgRemovalAlgorithm::Bfrnet => OrientationSupport::NotApplicable,
        }
    }
}

impl InversionAlgorithm {
    /// How this method behaves when B0 is not along voxel `+z`.
    ///
    /// Every classical inversion builds its dipole kernel from the supplied direction, directly
    /// or through the shared ADMM/FANSI spectral setup, so all of them handle oblique data on
    /// the acquired grid. The deep-learning methods take no direction at all.
    pub fn orientation_support(&self) -> OrientationSupport {
        match self {
            InversionAlgorithm::Tkd
            | InversionAlgorithm::Tsvd
            | InversionAlgorithm::Tikhonov
            | InversionAlgorithm::Tv
            | InversionAlgorithm::Rts
            | InversionAlgorithm::Nltv
            | InversionAlgorithm::Medi
            | InversionAlgorithm::Tfi
            | InversionAlgorithm::Ilsqr
            | InversionAlgorithm::Star
            | InversionAlgorithm::Tgv
            | InversionAlgorithm::Qsmart
            | InversionAlgorithm::Ndi
            | InversionAlgorithm::Fansi
            | InversionAlgorithm::FansiTgv
            | InversionAlgorithm::L1qsm
            | InversionAlgorithm::Whqsm
            | InversionAlgorithm::Hdqsm
            | InversionAlgorithm::AmpPe
            | InversionAlgorithm::Lsqr
            | InversionAlgorithm::Heidi => OrientationSupport::Arbitrary,
            InversionAlgorithm::Xqsm
            | InversionAlgorithm::Qsmnet
            | InversionAlgorithm::QsmnetPlus
            | InversionAlgorithm::Autoqsm
            | InversionAlgorithm::Qsmgan
            | InversionAlgorithm::Ir2qsm
            | InversionAlgorithm::Lpcnn
            | InversionAlgorithm::ModlQsm
            | InversionAlgorithm::Nextqsm
            | InversionAlgorithm::Iqsm
            | InversionAlgorithm::IqsmPlus => OrientationSupport::AxialOnly,
        }
    }
}

impl SeparationAlgorithm {
    /// How this method behaves when B0 is not along voxel `+z`.
    ///
    /// The model-based separations carry the direction through their field terms. The rest
    /// either consume an already-reconstructed χ map and R2\* (so orientation was settled
    /// upstream) or are networks trained on axial data.
    pub fn orientation_support(&self) -> OrientationSupport {
        match self {
            SeparationAlgorithm::ChiSepIlsqr | SeparationAlgorithm::ChiSepMedi => {
                OrientationSupport::Arbitrary
            }
            SeparationAlgorithm::R2starQsm
            | SeparationAlgorithm::WaveSep
            | SeparationAlgorithm::Decompose
            | SeparationAlgorithm::HcChisep => OrientationSupport::NotApplicable,
            SeparationAlgorithm::SusepNet | SeparationAlgorithm::ChiSepNet => {
                OrientationSupport::AxialOnly
            }
        }
    }
}

// ─── Deep-learning model-registry mapping ───
//
// These map each stage enum's deep-learning variants to their [`crate::models`]
// registry id. Exhaustive matches: adding a variant forces a decision here, and the
// `registry_models_are_pipeline_wired` test asserts every registry model of a pipeline
// stage is claimed by some variant — so a model added to the registry but not wired
// into an enum fails the build's tests (registry↔pipeline drift guard).

impl BgRemovalAlgorithm {
    /// Model-registry id for deep-learning variants, else `None` (classical methods).
    pub fn dl_model_id(self) -> Option<&'static str> {
        match self {
            Self::Bfrnet => Some("bfrnet"),
            Self::Vsharp | Self::Pdf | Self::Lbv | Self::Ismv | Self::Sharp
            | Self::Resharp | Self::Harperella | Self::Iharperella => None,
        }
    }
    /// Every variant, for exhaustiveness in tests/tools.
    pub const VARIANTS: &'static [Self] = &[
        Self::Vsharp, Self::Pdf, Self::Lbv, Self::Ismv, Self::Sharp,
        Self::Resharp, Self::Harperella, Self::Iharperella, Self::Bfrnet,
    ];
}

impl InversionAlgorithm {
    /// Model-registry id for deep-learning variants, else `None` (classical methods).
    pub fn dl_model_id(self) -> Option<&'static str> {
        match self {
            Self::Xqsm => Some("xqsm"),
            Self::Qsmnet => Some("qsmnet"),
            Self::QsmnetPlus => Some("qsmnet-plus"),
            Self::Autoqsm => Some("autoqsm"),
            Self::Qsmgan => Some("qsmgan"),
            Self::Ir2qsm => Some("ir2qsm"),
            Self::Lpcnn => Some("lpcnn"),
            Self::ModlQsm => Some("modl-qsm"),
            Self::Nextqsm => Some("nextqsm"),
            Self::Iqsm => Some("iqsm"),
            Self::IqsmPlus => Some("iqsm-plus"),
            Self::Tkd | Self::Tsvd | Self::Tikhonov | Self::Tv | Self::Rts | Self::Nltv
            | Self::Medi | Self::Tfi | Self::Ilsqr | Self::Star | Self::Tgv | Self::Qsmart | Self::Ndi
            | Self::Fansi | Self::FansiTgv | Self::L1qsm | Self::Whqsm | Self::Hdqsm
            | Self::AmpPe | Self::Lsqr | Self::Heidi => None,
        }
    }
    pub const VARIANTS: &'static [Self] = &[
        Self::Tkd, Self::Tsvd, Self::Tikhonov, Self::Tv, Self::Rts, Self::Nltv, Self::Medi,
        Self::Tfi, Self::Ilsqr, Self::Star, Self::Tgv, Self::Qsmart, Self::Ndi, Self::Fansi, Self::FansiTgv,
        Self::L1qsm, Self::Whqsm, Self::Hdqsm, Self::AmpPe, Self::Lsqr, Self::Heidi,
        Self::Xqsm, Self::Qsmnet,
        Self::QsmnetPlus, Self::Autoqsm, Self::Qsmgan, Self::Ir2qsm, Self::Lpcnn, Self::ModlQsm,
        Self::Nextqsm, Self::Iqsm, Self::IqsmPlus,
    ];
}

impl SeparationAlgorithm {
    /// Model-registry id for deep-learning variants, else `None` (classical methods).
    pub fn dl_model_id(self) -> Option<&'static str> {
        match self {
            Self::SusepNet => Some("susep-net"),
            Self::ChiSepNet => Some("chi-sepnet"),
            Self::ChiSepIlsqr | Self::ChiSepMedi | Self::R2starQsm | Self::WaveSep
            | Self::Decompose | Self::HcChisep => None,
        }
    }
    pub const VARIANTS: &'static [Self] = &[
        Self::ChiSepIlsqr, Self::ChiSepMedi, Self::R2starQsm, Self::WaveSep,
        Self::Decompose, Self::HcChisep, Self::SusepNet, Self::ChiSepNet,
    ];
}

// ─── Stable choice ids ───
//
// Each selection enum names its variants by a stable kebab-case id, used by the pipeline rules
// ([`super::rules`]) to name a choice in data and JSON. They are the names QSMxT's config uses for
// the same choices, so a host's mapping is the identity wherever the two agree.

impl UnwrappingAlgorithm {
    /// Stable id (`"romeo"`, `"laplacian"`).
    pub fn id(self) -> &'static str {
        match self {
            Self::Romeo => "romeo",
            Self::Laplacian => "laplacian",
        }
    }
}

impl B0EstimationMethod {
    /// Stable id of the method (its settings are not part of it).
    pub fn id(&self) -> &'static str {
        match self {
            Self::WeightedAvg { .. } => "weighted-avg",
            Self::LinearFit(_) => "linear-fit",
            Self::RobustFit(_) => "robust-fit",
            Self::ComplexFit(_) => "complex-fit",
            Self::QsmartEchofit { .. } => "qsmart-echofit",
        }
    }
}

impl MaskOp {
    /// Stable id of the operation (its parameters are not part of it).
    pub fn id(&self) -> &'static str {
        match self {
            Self::Threshold { .. } => "threshold",
            Self::Bet(_) => "bet",
            Self::Erode { .. } => "erode",
            Self::Dilate { .. } => "dilate",
            Self::Close { .. } => "close",
            Self::FillHoles { .. } => "fill-holes",
            Self::GaussianSmooth { .. } => "gaussian-smooth",
            Self::SignalErode(_) => "signal-erode",
            Self::HdBet(_) => "hd-bet",
            Self::Rs2Net(_) => "rs2-net",
            Self::MindGrab(_) => "mindgrab",
        }
    }
}

impl BgRemovalAlgorithm {
    /// Stable id (`"vsharp"`, `"pdf"`, …).
    pub fn id(self) -> &'static str {
        match self {
            Self::Vsharp => "vsharp",
            Self::Pdf => "pdf",
            Self::Lbv => "lbv",
            Self::Ismv => "ismv",
            Self::Sharp => "sharp",
            Self::Resharp => "resharp",
            Self::Harperella => "harperella",
            Self::Iharperella => "iharperella",
            Self::Bfrnet => "bfrnet",
        }
    }
}

impl InversionAlgorithm {
    /// Stable id (`"rts"`, `"medi"`, `"fansi-tgv"`, …).
    pub fn id(self) -> &'static str {
        match self {
            Self::Tkd => "tkd",
            Self::Tsvd => "tsvd",
            Self::Tikhonov => "tikhonov",
            Self::Tv => "tv",
            Self::Rts => "rts",
            Self::Nltv => "nltv",
            Self::Medi => "medi",
            Self::Tfi => "tfi",
            Self::Ilsqr => "ilsqr",
            Self::Star => "star",
            Self::Tgv => "tgv",
            Self::Qsmart => "qsmart",
            Self::Ndi => "ndi",
            Self::Fansi => "fansi",
            Self::FansiTgv => "fansi-tgv",
            Self::L1qsm => "l1qsm",
            Self::Whqsm => "whqsm",
            Self::Hdqsm => "hdqsm",
            Self::AmpPe => "amp-pe",
            Self::Lsqr => "lsqr",
            Self::Heidi => "heidi",
            Self::Xqsm => "xqsm",
            Self::Qsmnet => "qsmnet",
            Self::QsmnetPlus => "qsmnet-plus",
            Self::Autoqsm => "autoqsm",
            Self::Qsmgan => "qsmgan",
            Self::Ir2qsm => "ir2qsm",
            Self::Lpcnn => "lpcnn",
            Self::ModlQsm => "modl-qsm",
            Self::Nextqsm => "nextqsm",
            Self::Iqsm => "iqsm",
            Self::IqsmPlus => "iqsm-plus",
        }
    }
}

impl SeparationAlgorithm {
    /// Stable id (`"chi-sep-ilsqr"`, `"decompose"`, …).
    pub fn id(self) -> &'static str {
        match self {
            Self::ChiSepIlsqr => "chi-sep-ilsqr",
            Self::ChiSepMedi => "chi-sep-medi",
            Self::R2starQsm => "r2star-qsm",
            Self::WaveSep => "wavesep",
            Self::Decompose => "decompose",
            Self::HcChisep => "hc-chisep",
            Self::SusepNet => "susep-net",
            Self::ChiSepNet => "chi-sepnet",
        }
    }
}

/// Configuration for the χ-separation stage.
#[derive(Clone, Debug)]
pub struct SeparationConfig {
    pub algorithm: SeparationAlgorithm,
    /// `cf` on the chi-sep params is overridden from scan metadata by the dispatcher.
    pub chi_sep_ilsqr: ChiSepIlsqrParams,
    pub chi_sep_medi: ChiSepParams,
    /// `b0` overridden from scan metadata by the dispatcher.
    pub r2star_qsm: R2starQsmParams,
    pub wavesep: WaveSepParams,
    /// `b0` overridden from scan metadata by the dispatcher.
    pub decompose: DecomposeParams,
    /// `b0` overridden from scan metadata; `se_echo_times` supplies the SE echoes.
    pub hc_chisep: HcChisepParams,
    /// χ-sepnet sliding-window patch (default: the authors' 192×192×128).
    pub chi_sepnet: ChiSepNetParams,
    /// SUSEP-Net whole volume (default) or sliding-window patch.
    pub susep_net: SusepNetParams,
}

impl Default for SeparationConfig {
    fn default() -> Self {
        Self {
            algorithm: SeparationAlgorithm::ChiSepIlsqr,
            chi_sep_ilsqr: ChiSepIlsqrParams::default(),
            chi_sep_medi: ChiSepParams::default(),
            r2star_qsm: R2starQsmParams::default(),
            wavesep: WaveSepParams::default(),
            decompose: DecomposeParams::default(),
            hc_chisep: HcChisepParams::default(),
            chi_sepnet: ChiSepNetParams::default(),
            susep_net: SusepNetParams::default(),
        }
    }
}

// =========================================================================
// Top-level pipeline config
// =========================================================================

/// Complete QSM pipeline configuration.
///
/// Contains all per-stage configs. Consumers call individual stage functions
/// (e.g. `run_field_mapping`, `run_bg_removal`) passing the relevant section.
/// Masking config is used when the consumer needs to generate a mask.
#[derive(Clone, Debug)]
pub struct QsmPipelineConfig {
    pub masking: MaskingConfig,
    pub field_mapping: FieldMappingConfig,
    pub bg_removal: BgRemovalConfig,
    pub inversion: InversionConfig,
    pub reference: QsmReference,
}

impl Default for QsmPipelineConfig {
    fn default() -> Self {
        Self {
            masking: MaskingConfig::default(),
            field_mapping: FieldMappingConfig::default(),
            bg_removal: BgRemovalConfig::default(),
            inversion: InversionConfig::default(),
            reference: QsmReference::Mean,
        }
    }
}

// =========================================================================
// Scan metadata and stage result types
// =========================================================================

/// Metadata about the scan
#[derive(Clone, Debug)]
pub struct ScanMetadata {
    /// Volume dimensions (nx, ny, nz)
    pub dims: (usize, usize, usize),
    /// Voxel size in mm (vsx, vsy, vsz)
    pub voxel_size: (f64, f64, f64),
    /// Echo times in seconds
    pub echo_times: Vec<f64>,
    /// Main field strength in Tesla
    pub field_strength: f64,
    /// B0 direction as unit vector in voxel coordinates
    pub b0_direction: (f64, f64, f64),
    /// Slice thickness and slice axis, when the acquisition recorded them.
    ///
    /// `voxel_size` along the slice axis is the slice-to-slice *pitch*; this supplies the
    /// thickness, and the two together say whether the sampled volume has holes in it. A 2D
    /// multi-slice acquisition often leaves a gap, and a gapped volume cannot be reconstructed
    /// at all — the FFT dipole and spherical-mean kernels are defined on a uniform grid, so on
    /// one with holes they compute a different convolution and produce a map that is wrong
    /// while still looking like a susceptibility map.
    ///
    /// `None` means the acquisition did not say. BIDS only *recommends* `SliceThickness`, so
    /// this is common, and a gap then cannot be detected by this crate or by anything else
    /// reading the NIfTI alone. The stages run anyway rather than refusing the vast majority of
    /// data that has no gap — so a host that can read the field should set it, and a host that
    /// cannot should tell its user that this check could not run. See
    /// [`Grid::require_contiguous_slices_if_known`](crate::Grid::require_contiguous_slices_if_known).
    pub slice_geometry: Option<crate::grid::SliceGeometry>,
    /// The image's voxel→world affine (row-major 4×4, as in a NIfTI sform/qform), when the host
    /// has one. Only the steps that need the anatomical orientation read it — currently
    /// [`MaskOp::MindGrab`], which errors without it; everything else works in voxel space.
    pub affine: Option<[f64; 16]>,
}

impl ScanMetadata {
    /// Get a Grid from this metadata's dimensions and voxel sizes.
    #[inline]
    pub fn grid(&self) -> crate::Grid {
        crate::Grid {
            dims: self.dims,
            voxel_size: self.voxel_size,
        }
    }

    /// Refuse an acquisition whose slices are not contiguous.
    ///
    /// Every stage that convolves with a kernel defined on the voxel grid calls this first —
    /// background removal, dipole inversion, TGV, chi-separation. They call it unconditionally
    /// rather than per algorithm, because every one of those is three-dimensional: a complete
    /// background removal needs a 3D stage to have the mean value property at all, and the
    /// dipole kernel is inherently 3D, so there is no variant for which a gap is harmless.
    ///
    /// Phase unwrapping deliberately does **not** call this. A gap degrades 3D unwrapping too,
    /// but the remedy there is to unwrap in plane ([`crate::unwrap::unwrap_slicewise`]) rather
    /// than to refuse, and refusing would block the one path that handles the case.
    ///
    /// Returns [`PipelineError::InvalidInput`] carrying the measured thickness, pitch and gap.
    pub fn require_contiguous_slices(&self) -> Result<(), PipelineError> {
        self.grid()
            .require_contiguous_slices_if_known(self.slice_geometry)
            .map_err(|e| PipelineError::InvalidInput(e.to_string()))
    }
}

/// Results from field mapping stage
pub struct FieldMappingResult {
    /// B0 field map in ppm
    pub b0_field_ppm: Vec<f64>,
    /// Phase offset map (if phase offset removal was used)
    pub phase_offset: Option<Vec<f64>>,
    /// What the robust fit found, when [`B0EstimationMethod::RobustFit`] combined the echoes;
    /// `None` for every other method.
    pub robust_fit: Option<RobustFitReport>,
    /// QSMART's reliability map `R_0` (1 = reliable), when [`B0EstimationMethod::QsmartEchofit`]
    /// combined the echoes; `None` for every other method. Pass it to
    /// [`run_qsmart`](crate::pipeline::run_qsmart).
    pub reliability: Option<Vec<u8>>,
    /// The blurred echo-fit residual [`reliability`](Self::reliability) was cut at, when
    /// [`B0EstimationMethod::QsmartEchofit`] combined the echoes: the percentile's value for
    /// [`QsmartFitThreshold::Percentile`](crate::utils::QsmartFitThreshold::Percentile), the
    /// value itself for a fixed threshold (`echofit.m` prints it as "Setting adaptive echo-fit
    /// threshold at %2.2f"). `None` for every other method.
    pub qsmart_fit_threshold: Option<f64>,
    /// Standard deviation of `b0_field_ppm` per voxel, in ppm, for the methods that estimate it:
    /// [`B0EstimationMethod::ComplexFit`] (`Fit_ppm_complex.m`'s `dp1`, MEDI's `N_std`, which
    /// MEDI passes to PDF and MEDI_L1). `None` for every other method.
    pub field_noise_ppm: Option<Vec<f64>>,
}

/// Per-echo evidence from [`B0EstimationMethod::RobustFit`]: which echoes the fit stopped
/// trusting, and where. Diagnostics only; the field map has already been computed with these
/// weights and nothing here drops an echo.
#[derive(Clone, Debug)]
pub struct RobustFitReport {
    /// Per-echo summary, including [`EchoQuality::flagged`] (the echoes that are out of line
    /// with the rest) and [`EchoQuality::voxels_examined`] (check it before trusting a clean
    /// report).
    pub quality: EchoQuality,
    /// Final robust weight per echo per voxel, on `[0, 1]`, before the magnitude is multiplied
    /// in: `robust_weights[echo][voxel]`, same voxel order as the input. `1.0` where the loss
    /// never fired, including
    /// everywhere outside the mask, which is never examined.
    pub robust_weights: Vec<Vec<f64>>,
}

/// Results from background removal stage
pub struct BgRemovalResult {
    /// Local field in ppm
    pub local_field_ppm: Vec<f64>,
    /// Eroded mask
    pub eroded_mask: Vec<u8>,
}

/// Pipeline error type
#[derive(Debug)]
pub enum PipelineError {
    /// Invalid configuration
    InvalidConfig(String),
    /// Invalid input data
    InvalidInput(String),
    /// Algorithm failure
    AlgorithmError(String),
    /// Dimension mismatch
    DimensionMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid config: {}", msg),
            Self::InvalidInput(msg) => write!(f, "invalid input: {}", msg),
            Self::AlgorithmError(msg) => write!(f, "algorithm error: {}", msg),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {}, got {}", expected, got)
            }
        }
    }
}

impl std::error::Error for PipelineError {}

impl From<crate::error::InputError> for PipelineError {
    fn from(e: crate::error::InputError) -> Self {
        Self::InvalidInput(e.0)
    }
}

#[cfg(test)]
mod orientation_tests {
    use super::*;

    /// Every classical inversion builds its kernel from the supplied direction, so it must claim
    /// Arbitrary. If a new one is added without wiring bdir through, this is where it shows up.
    #[test]
    fn classical_inversions_take_a_direction() {
        for a in [
            InversionAlgorithm::Tkd, InversionAlgorithm::Tsvd, InversionAlgorithm::Tikhonov,
            InversionAlgorithm::Tv, InversionAlgorithm::Rts, InversionAlgorithm::Nltv,
            InversionAlgorithm::Medi, InversionAlgorithm::Tfi, InversionAlgorithm::Ilsqr,
            InversionAlgorithm::Star, InversionAlgorithm::Tgv, InversionAlgorithm::Qsmart, InversionAlgorithm::Ndi,
            InversionAlgorithm::Fansi, InversionAlgorithm::FansiTgv, InversionAlgorithm::L1qsm,
            InversionAlgorithm::Whqsm, InversionAlgorithm::Hdqsm, InversionAlgorithm::AmpPe,
            InversionAlgorithm::Lsqr, InversionAlgorithm::Heidi,
        ] {
            assert_eq!(a.orientation_support(), OrientationSupport::Arbitrary, "{a:?}");
            assert!(!a.orientation_support().requires_axial(), "{a:?}");
        }
    }

    /// Networks have no direction input, so oblique data must be resampled for them.
    #[test]
    fn learned_inversions_need_axial_data() {
        for a in [
            InversionAlgorithm::Xqsm, InversionAlgorithm::Qsmnet, InversionAlgorithm::QsmnetPlus,
            InversionAlgorithm::Autoqsm, InversionAlgorithm::Qsmgan, InversionAlgorithm::Ir2qsm,
            InversionAlgorithm::Lpcnn, InversionAlgorithm::ModlQsm, InversionAlgorithm::Nextqsm,
            InversionAlgorithm::Iqsm, InversionAlgorithm::IqsmPlus,
        ] {
            assert!(a.orientation_support().requires_axial(), "{a:?}");
        }
    }

    /// SMV-family background removal is harmonic and rotation-invariant; only PDF cares.
    #[test]
    fn background_removal_is_direction_independent_except_pdf() {
        assert_eq!(BgRemovalAlgorithm::Pdf.orientation_support(), OrientationSupport::Arbitrary);
        for a in [
            BgRemovalAlgorithm::Vsharp, BgRemovalAlgorithm::Sharp, BgRemovalAlgorithm::Resharp,
            BgRemovalAlgorithm::Ismv, BgRemovalAlgorithm::Lbv, BgRemovalAlgorithm::Harperella,
            BgRemovalAlgorithm::Iharperella, BgRemovalAlgorithm::Bfrnet,
        ] {
            assert_eq!(a.orientation_support(), OrientationSupport::NotApplicable, "{a:?}");
            assert!(!a.orientation_support().requires_axial(), "{a:?}");
        }
    }

    /// No background removal or separation method should ever force a resample on its own.
    #[test]
    fn only_learned_methods_force_a_resample() {
        for a in [SeparationAlgorithm::ChiSepIlsqr, SeparationAlgorithm::ChiSepMedi] {
            assert_eq!(a.orientation_support(), OrientationSupport::Arbitrary, "{a:?}");
        }
        for a in [SeparationAlgorithm::SusepNet, SeparationAlgorithm::ChiSepNet] {
            assert!(a.orientation_support().requires_axial(), "{a:?}");
        }
        for a in [
            SeparationAlgorithm::R2starQsm, SeparationAlgorithm::WaveSep,
            SeparationAlgorithm::Decompose, SeparationAlgorithm::HcChisep,
        ] {
            assert!(!a.orientation_support().requires_axial(), "{a:?}");
        }
    }
}

#[cfg(test)]
mod inversion_input_tests {
    use super::*;

    fn input(algorithm: InversionAlgorithm, smv: bool, n_echoes: usize) -> InversionInput {
        let mut c = InversionConfig { algorithm, ..Default::default() };
        c.medi.smv = smv;
        c.input(n_echoes)
    }

    /// Every variant, spelled out, so a new algorithm cannot be added without deciding its input.
    #[test]
    fn every_inversion_declares_its_input() {
        use InversionAlgorithm as A;
        use InversionInput::*;
        let expected: &[(A, InversionInput)] = &[
            (A::Tkd, LocalField), (A::Tsvd, LocalField), (A::Tikhonov, LocalField),
            (A::Tv, LocalField), (A::Rts, LocalField), (A::Nltv, LocalField),
            (A::Tfi, TotalField), (A::Ilsqr, LocalField), (A::Star, LocalField), (A::Qsmart, TotalField),
            (A::Ndi, LocalField), (A::Fansi, LocalField), (A::FansiTgv, LocalField),
            (A::L1qsm, LocalField), (A::Whqsm, LocalField), (A::Hdqsm, LocalField),
            (A::AmpPe, LocalField), (A::Lsqr, LocalField), (A::Heidi, LocalField),
            (A::Xqsm, LocalField), (A::Qsmnet, LocalField), (A::QsmnetPlus, LocalField),
            (A::Autoqsm, TotalField), (A::Qsmgan, LocalField), (A::Ir2qsm, LocalField),
            (A::Lpcnn, LocalField), (A::ModlQsm, LocalField), (A::Nextqsm, TotalField),
            (A::Iqsm, WrappedPhase), (A::IqsmPlus, WrappedPhase), (A::Medi, LocalField),
        ];
        // TGV depends on more than the name; checked below.
        let special = [A::Tgv];
        for v in A::VARIANTS {
            if special.contains(v) {
                continue;
            }
            let (_, want) = expected.iter().find(|(a, _)| a == v)
                .unwrap_or_else(|| panic!("{v:?} is missing from this test"));
            for n_echoes in [1, 4] {
                for smv in [false, true] {
                    assert_eq!(input(*v, smv, n_echoes), *want, "{v:?}, {n_echoes} echoes, smv {smv}");
                }
            }
        }
        assert_eq!(expected.len() + special.len(), A::VARIANTS.len());
    }

    /// MEDI takes a background removal's local field with every SMV option, as the MEDI
    /// toolbox's README.m pipeline and the mSMV paper run it (#287).
    #[test]
    fn medi_takes_the_local_field_with_every_smv_option() {
        for smv in [true, false] {
            for shrink in [true, false] {
                for msmv in [None, Some(crate::inversion::MsmvParams::default())] {
                    let mut c = InversionConfig { algorithm: InversionAlgorithm::Medi, ..Default::default() };
                    c.medi.smv = smv;
                    c.medi.smv_shrink_mask = shrink;
                    c.medi.msmv = msmv;
                    for n in [1, 4] {
                        assert_eq!(c.input(n), InversionInput::LocalField);
                    }
                }
            }
        }
    }

    /// The MEDI toolbox's pipeline runs PDF before MEDI_L1.
    #[test]
    fn medi_reference_bg_removal_is_pdf() {
        let c = InversionConfig { algorithm: InversionAlgorithm::Medi, ..Default::default() };
        assert_eq!(c.reference_bg_removal(), Some(BgRemovalAlgorithm::Pdf));
        let c = InversionConfig { algorithm: InversionAlgorithm::Rts, ..Default::default() };
        assert_eq!(c.reference_bg_removal(), None);
    }

    #[test]
    fn tgv_takes_the_field_for_multi_echo_and_phase_for_one_echo() {
        assert_eq!(input(InversionAlgorithm::Tgv, true, 1), InversionInput::WrappedPhase);
        for n in [2, 4] {
            assert_eq!(input(InversionAlgorithm::Tgv, true, n), InversionInput::TotalField);
        }
    }

    #[test]
    fn routing_predicates() {
        assert!(InversionInput::LocalField.needs_bg_removal());
        assert!(!InversionInput::TotalField.needs_bg_removal());
        assert!(!InversionInput::WrappedPhase.needs_bg_removal());
        assert!(InversionInput::LocalField.needs_field_mapping());
        assert!(InversionInput::TotalField.needs_field_mapping());
        assert!(!InversionInput::WrappedPhase.needs_field_mapping());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{all_models, ModelStage};

    /// Registry↔pipeline drift guard: every deep-learning model whose `stage` maps to a
    /// pipeline stage enum must be claimed by some enum variant's `dl_model_id()`. A model
    /// added to the registry but not wired into an enum (so no pipeline consumer could
    /// select it) fails here. `PhaseToField` (iQFM) is intentionally exempt — it's exposed
    /// as the standalone `run_iqfm` field-preparation building block, not a stage enum.
    #[test]
    fn registry_models_are_pipeline_wired() {
        let inv: Vec<&str> = InversionAlgorithm::VARIANTS.iter().filter_map(|v| v.dl_model_id()).collect();
        let bfr: Vec<&str> = BgRemovalAlgorithm::VARIANTS.iter().filter_map(|v| v.dl_model_id()).collect();
        let sep: Vec<&str> = SeparationAlgorithm::VARIANTS.iter().filter_map(|v| v.dl_model_id()).collect();
        for m in all_models() {
            let (wired, enum_name) = match m.stage {
                ModelStage::BackgroundRemoval => (bfr.contains(&m.id), "BgRemovalAlgorithm"),
                ModelStage::DipoleInversion | ModelStage::SingleStep => (inv.contains(&m.id), "InversionAlgorithm"),
                ModelStage::ChiSeparation => (sep.contains(&m.id), "SeparationAlgorithm"),
                ModelStage::BrainExtraction => {
                    let ops = [
                        MaskOp::HdBet(Default::default()),
                        MaskOp::Rs2Net(Default::default()),
                        MaskOp::MindGrab(Default::default()),
                    ];
                    (ops.iter().any(|op| op.dl_model_id() == Some(m.id)), "MaskOp")
                }
                // iQFM: standalone run_iqfm, no stage enum.
                ModelStage::PhaseToField => continue,
                // SynthSeg: standalone segment::synthseg, an analysis step rather than a
                // reconstruction stage. Remove this arm if a segmentation stage is added.
                ModelStage::Segmentation => continue,
                // R2PRIMEnet: standalone relaxometry::r2primenet, an input-preparation step
                // that feeds the R2'-consuming separation methods rather than a stage of its own.
                ModelStage::R2PrimeGeneration => continue,
            };
            assert!(wired, "registry model '{}' (stage {:?}) has no {} variant — wire it or add a dl_model_id mapping", m.id, m.stage, enum_name);
        }
    }

    #[test]
    fn test_default_config() {
        let config = QsmPipelineConfig::default();
        assert_eq!(config.field_mapping.unwrapping_algorithm, UnwrappingAlgorithm::Romeo);
        assert_eq!(config.bg_removal.algorithm, BgRemovalAlgorithm::Vsharp);
        assert_eq!(config.inversion.algorithm, InversionAlgorithm::Rts);
        assert_eq!(config.reference, QsmReference::Mean);
        assert!(config.field_mapping.phase_offset_removal);
        assert!(!config.field_mapping.bipolar_correction);
    }

    #[test]
    fn test_default_masking_config() {
        let config = MaskingConfig::default();
        assert_eq!(config.sections.len(), 1);
        assert_eq!(config.sections[0].input, MaskingInput::PhaseQuality);
        assert_eq!(config.sections[0].refinements.len(), 3);
    }

    #[test]
    fn test_mask_section_all_ops() {
        let section = MaskSection {
            input: MaskingInput::Magnitude,
            generator: MaskOp::Threshold { method: MaskThresholdMethod::Otsu, value: None },
            refinements: vec![MaskOp::Erode { iterations: 1 }, MaskOp::Dilate { iterations: 2 }],
        };
        let ops = section.all_ops();
        assert_eq!(ops.len(), 3);
        assert!(matches!(ops[0], MaskOp::Threshold { .. }));
        assert!(matches!(ops[1], MaskOp::Erode { iterations: 1 }));
        assert!(matches!(ops[2], MaskOp::Dilate { iterations: 2 }));
    }
}

#[cfg(test)]
mod slice_gap_tests {
    use super::*;

    // What gates the slice-gap wiring, recorded so the sweep can be reconstructed. Each of
    // these was applied and confirmed to fail at least one assertion in this module,
    // `slice_gap_coverage`, or the stage tests: the check removed from each of background
    // removal, dipole inversion, chi-separation and `run_nextqsm`; the helper swallowing the
    // refusal and returning `Ok`; the refusal downgraded to `InvalidConfig`; the tolerance
    // widened to half the pitch; a non-positive thickness accepted; `None` refused instead of
    // allowed; the `Option`-aware check ignoring the declared geometry; `gap()` sign-flipped;
    // `spacing()` reading the wrong axis; `SliceGeometry::contiguous` ignoring its axis; and
    // the pinned stage count made stale.
    //
    // Worth knowing: removing the *dispatcher's* check does not fail the NeXtQSM route test,
    // because that route is defended twice. `slice_gap_coverage` is what pins both call sites.
    use crate::grid::SliceGeometry;

    fn meta(thickness: Option<f64>) -> ScanMetadata {
        ScanMetadata {
            dims: (8, 8, 4),
            voxel_size: (1.0, 1.0, 3.0),
            echo_times: vec![0.005],
            field_strength: 3.0,
            b0_direction: (0.0, 0.0, 1.0),
            slice_geometry: thickness.map(|t| SliceGeometry { thickness: t, axis: 2 }),
            affine: None,
        }
    }

    #[test]
    fn a_declared_gap_is_refused_with_the_numbers_in_the_message() {
        let err = meta(Some(2.0)).require_contiguous_slices().unwrap_err();
        let PipelineError::InvalidInput(msg) = &err else {
            panic!("a gap should be invalid *input*, not a config or algorithm error: {err:?}");
        };
        // the host has to be able to tell its user what was wrong with the data
        assert!(msg.contains("2.0000"), "no thickness in the message: {msg}");
        assert!(msg.contains("3.0000"), "no pitch in the message: {msg}");
        assert!(msg.contains("1.0000"), "no gap in the message: {msg}");
        assert!(msg.contains("not contiguous"), "no explanation in the message: {msg}");
    }

    #[test]
    fn a_declared_contiguous_acquisition_passes() {
        assert!(meta(Some(3.0)).require_contiguous_slices().is_ok());
    }

    #[test]
    fn an_undeclared_acquisition_passes_because_nothing_can_be_concluded() {
        // Not a silent skip: the thickness is genuinely absent from the input, and refusing
        // every acquisition that omitted a recommended BIDS field would block almost all data.
        assert!(meta(None).require_contiguous_slices().is_ok());
    }

    #[test]
    fn the_slice_axis_is_honoured() {
        let mut m = meta(None);
        // 1 mm slices stacked along x on a 1 x 1 x 3 mm grid: contiguous
        m.slice_geometry = Some(SliceGeometry { thickness: 1.0, axis: 0 });
        assert!(m.require_contiguous_slices().is_ok());
        m.slice_geometry = Some(SliceGeometry { thickness: 0.5, axis: 0 });
        assert!(m.require_contiguous_slices().is_err());
    }
}

/// Every pipeline stage that convolves on the voxel grid has to refuse a gapped volume, and a
/// stage added later must not be able to skip that by omission. These tests read the stage
/// sources and require each `pub fn` taking a `&ScanMetadata` to either call
/// [`ScanMetadata::require_contiguous_slices`] or appear on an allow-list with a reason —
/// the same spirit as the orientation tests above, which enumerate every algorithm variant so a
/// new one cannot be added without declaring what it supports.
#[cfg(test)]
mod slice_gap_coverage {
    /// Stages that legitimately do not check, and why.
    const EXEMPT: &[(&str, &str)] = &[
        // Unwrapping is degraded by a gap, not invalidated by it, and the remedy is
        // unwrap::unwrap_slicewise rather than refusal. Refusing here would block the one path
        // that handles the case.
        ("run_field_mapping", "unwrapping: the remedy is slice-wise, not refusal"),
        // Masking is thresholding and morphology on the voxel grid. It reads no kernel and
        // produces no field, so a gap changes nothing about whether its output is meaningful —
        // and the stages that consume the mask refuse the gap themselves.
        ("run_masking", "no kernel, no field: a gap is irrelevant"),
        ("build_mask_section", "no kernel, no field: a gap is irrelevant"),
        ("resolve_masking_input", "no kernel, no field: a gap is irrelevant"),
        ("section_magnitude", "no kernel, no field: a gap is irrelevant"),
        ("apply_mask_ops", "no kernel, no field: a gap is irrelevant"),
        // Registration and resampling are interpolation on the sampled grid, not a convolution
        // with a kernel that assumes one. A gap makes the through-plane interpolation cross
        // tissue that was never measured, which costs accuracy bounded by how smooth the field
        // is — it does not turn the operation into a different operation. The output stays on
        // the reference echo's grid, so `slice_geometry` still describes it, and the stages that
        // genuinely cannot cope refuse before any kernel runs.
        ("run_motion_correction", "interpolation, not a kernel: a gap degrades rather than invalidates"),
    ];

    const SOURCES: &[(&str, &str)] = &[
        ("bg_removal.rs", include_str!("bg_removal.rs")),
        ("inversion.rs", include_str!("inversion.rs")),
        ("separation.rs", include_str!("separation.rs")),
        ("qsmart.rs", include_str!("qsmart.rs")),
        ("field_mapping.rs", include_str!("field_mapping.rs")),
        ("masking.rs", include_str!("masking.rs")),
        ("motion.rs", include_str!("motion.rs")),
        ("two_pass.rs", include_str!("two_pass.rs")),
        ("referencing.rs", include_str!("referencing.rs")),
        ("phase_utils.rs", include_str!("phase_utils.rs")),
    ];

    /// Every `pub fn` in a stage source, with its signature and body, excluding test modules.
    fn public_stage_fns(src: &str) -> Vec<(String, String)> {
        // stop at the test module so test helpers taking metadata are not scanned
        let code = match src.find("\nmod tests {") {
            Some(i) => &src[..i],
            None => src,
        };
        let mut out = Vec::new();
        let mut rest = code;
        while let Some(i) = rest.find("\npub fn ") {
            let after = &rest[i + "\npub fn ".len()..];
            let name: String = after.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
            // the body runs to the next top-level `pub fn`, which is enough to see the call
            let body_end = after.find("\npub fn ").unwrap_or(after.len());
            out.push((name, after[..body_end].to_string()));
            rest = &rest[i + 1..];
        }
        out
    }

    #[test]
    fn every_stage_taking_scan_metadata_refuses_a_gapped_volume() {
        let mut checked = 0usize;
        for (file, src) in SOURCES {
            for (name, body) in public_stage_fns(src) {
                // The cfg(not(feature = "onnx")) stubs name it `_metadata` and return an error
                // without touching the data, so they have nothing to check. Match on the
                // leading delimiter: "_metadata: &ScanMetadata" contains the unprefixed string.
                let takes_metadata = body.contains(" metadata: &ScanMetadata")
                    || body.contains("(metadata: &ScanMetadata");
                let returns_result = body.contains("PipelineError");
                if !takes_metadata || !returns_result {
                    continue;
                }
                if let Some((_, why)) = EXEMPT.iter().find(|(n, _)| *n == name) {
                    assert!(
                        !body.contains("require_contiguous_slices()"),
                        "{file}: {name} is on the slice-gap exemption list ({why}) but calls \
                         require_contiguous_slices anyway - remove it from EXEMPT"
                    );
                    continue;
                }
                assert!(
                    body.contains("metadata.require_contiguous_slices()?"),
                    "{file}: {name} takes a &ScanMetadata and returns a PipelineError but never \
                     calls metadata.require_contiguous_slices(). Either call it, or add it to \
                     EXEMPT with a reason a gap cannot invalidate it."
                );
                checked += 1;
            }
        }
        // if the scan stops finding stages the test has quietly stopped testing anything
        // Background removal, dipole inversion, the four single-step DL reconstructions
        // (NeXtQSM, iQSM, iQSM+, iQFM), TGV, chi-separation and QSMART: nine stages convolve on
        // the grid and must refuse a gap. If this count drops, the source scan has stopped
        // seeing stages rather than the stages having stopped needing the check.
        assert_eq!(
            checked, 9,
            "expected 9 checking stages, scanned {checked}; either a stage was added without \
             the check and without an exemption, or the source scan has broken"
        );
    }

    #[test]
    fn the_exemption_list_names_stages_that_exist() {
        for (name, _) in EXEMPT {
            assert!(
                SOURCES.iter().any(|(_, src)| src.contains(&format!("pub fn {name}("))),
                "EXEMPT names {name}, which no longer exists - stale exemption"
            );
        }
    }
}

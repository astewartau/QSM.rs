//! Pipeline configuration types
//!
//! Defines the configuration structs and enums for the QSM pipeline runner.
//! These are pure algorithm config types (no serde). Consumers that need
//! serialization (e.g. qsmxt-config) provide their own serde wrappers
//! and convert to these types.

use crate::bgremove::{
    IsmvParams, LbvParams, PdfParams, ResharpParams, SharpParams, SdfParams, VsharpParams,
    HarperellaParams, MsmvParams,
};
use crate::inversion::{
    IlsqrParams, MediParams, NltvParams, RtsParams, TgvParams, TikhonovParams, TkdParams, TvParams,
    NdiParams, FansiParams, L1QsmParams, WhQsmParams, HdQsmParams, TfiParams, AmpPeParams,
    LsqrQsmParams, HeidiParams,
};
use crate::separation::{
    ChiSepIlsqrParams, ChiSepParams, DecomposeParams, HcChisepParams, R2starQsmParams,
    WaveSepParams,
};
use crate::unwrap::romeo::RomeoParams;
use crate::utils::multi_echo::{B0WeightType, LinearFitParams};
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

/// B0 estimation method
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum B0EstimationMethod {
    WeightedAvg,
    LinearFit,
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
    Bet {
        fractional_intensity: f64,
        /// Factor applied to the voxel sizes before BET runs — see
        /// [`BetParams::voxel_scale`](crate::bet::BetParams::voxel_scale). 1.0 is the
        /// acquisition's own geometry; 10.0 is the usual preclinical setting for mouse data.
        voxel_scale: f64,
    },
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
    /// down to a depth cap. See [`crate::utils::signal_gated_erosion`].
    ///
    /// Gates on the **magnitude** image, never on the section's input: it divides out a
    /// receive-coil bias estimate and compares against the in-mask median, which only means
    /// anything for magnitude. Errors if no magnitude is supplied, so a phase-quality mask input
    /// can still be refined with it as long as the magnitude is available.
    SignalErode(crate::utils::SignalErosionParams),
    /// HD-BET deep-learning brain extraction from the **magnitude** (a generator, like `Bet`),
    /// whatever the section's input is. Requires the `onnx` feature and the `hd-bet` model
    /// weights; see [`crate::models`].
    HdBet(crate::bet::HdBetParams),
}

impl MaskOp {
    /// Registry id of the deep-learning model this op runs, if any.
    pub fn dl_model_id(&self) -> Option<&'static str> {
        match self {
            Self::HdBet(_) => Some("hd-bet"),
            Self::Threshold { .. } | Self::Bet { .. } | Self::Erode { .. } | Self::Dilate { .. }
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
    pub phase_offset_sigma: [f64; 3],
    pub bipolar_correction: bool,
    pub b0_estimation: B0EstimationMethod,
    pub b0_weight_type: B0WeightType,
    pub romeo_params: RomeoParams,
    pub linear_fit_params: LinearFitParams,
}

impl Default for FieldMappingConfig {
    fn default() -> Self {
        Self {
            unwrapping_algorithm: UnwrappingAlgorithm::Romeo,
            phase_offset_removal: true,
            phase_offset_sigma: [10.0, 10.0, 5.0],
            bipolar_correction: false,
            b0_estimation: B0EstimationMethod::WeightedAvg,
            b0_weight_type: B0WeightType::PhaseSNR,
            romeo_params: RomeoParams::default(),
            linear_fit_params: LinearFitParams::default(),
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
    pub sdf: SdfParams,
    /// mSMV refinement parameters (`b0`/`te` are overridden from scan metadata by
    /// the dispatcher). Used by the `msmv_refine` post-step.
    pub msmv: MsmvParams,
    /// Apply mSMV boundary-shadow refinement after the primary BFR (Roberts 2024).
    /// mSMV is a refinement, not a standalone primary remover, so it is exposed
    /// only as this post-step (redundant after `Ismv`, which is already SMV-based).
    pub msmv_refine: bool,
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
            sdf: SdfParams::default(),
            msmv: MsmvParams::default(),
            msmv_refine: false,
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
            | Self::Medi | Self::Tfi | Self::Ilsqr | Self::Tgv | Self::Qsmart | Self::Ndi
            | Self::Fansi | Self::FansiTgv | Self::L1qsm | Self::Whqsm | Self::Hdqsm
            | Self::AmpPe | Self::Lsqr | Self::Heidi => None,
        }
    }
    pub const VARIANTS: &'static [Self] = &[
        Self::Tkd, Self::Tsvd, Self::Tikhonov, Self::Tv, Self::Rts, Self::Nltv, Self::Medi,
        Self::Tfi, Self::Ilsqr, Self::Tgv, Self::Qsmart, Self::Ndi, Self::Fansi, Self::FansiTgv,
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
}

/// Results from field mapping stage
pub struct FieldMappingResult {
    /// B0 field map in ppm
    pub b0_field_ppm: Vec<f64>,
    /// Phase offset map (if phase offset removal was used)
    pub phase_offset: Option<Vec<f64>>,
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
            InversionAlgorithm::Tgv, InversionAlgorithm::Qsmart, InversionAlgorithm::Ndi,
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
                    (MaskOp::HdBet(Default::default()).dl_model_id() == Some(m.id), "MaskOp")
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

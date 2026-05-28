//! Pipeline configuration types
//!
//! Defines the configuration structs and enums for the QSM pipeline runner.
//! These are pure algorithm config types (no serde). Consumers that need
//! serialization (e.g. qsmxt-config) provide their own serde wrappers
//! and convert to these types.

use crate::bgremove::{
    IsmvParams, LbvParams, PdfParams, ResharpParams, SharpParams, SdfParams, VsharpParams,
};
use crate::inversion::{
    IlsqrParams, MediParams, NltvParams, RtsParams, TgvParams, TikhonovParams, TkdParams, TvParams,
};
use crate::pipeline::HarperellaParams;
use crate::unwrap::romeo::RomeoParams;
use crate::utils::multi_echo::{B0WeightType, LinearFitParams};
use crate::utils::QsmartParams;
use crate::bet::BetParams;

// =========================================================================
// Selection enums
// =========================================================================

/// Phase unwrapping algorithm
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UnwrappingAlgorithm {
    Romeo,
    Laplacian,
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
}

/// Dipole inversion algorithm
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InversionAlgorithm {
    Tkd,
    Tsvd,
    Tikhonov,
    Tv,
    Rts,
    Nltv,
    Medi,
    Ilsqr,
    Tgv,
    Qsmart,
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
    pub ilsqr: IlsqrParams,
    pub tgv: TgvParams,
    pub qsmart: QsmartParams,
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
            ilsqr: IlsqrParams::default(),
            tgv: TgvParams::default(),
            qsmart: QsmartParams::default(),
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
mod tests {
    use super::*;

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

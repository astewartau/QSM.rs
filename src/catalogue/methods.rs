//! The catalogue's entries, and the reference code they follow.
//!
//! A `reference` repeats what the method's module docs say it follows (toolbox, release or
//! commit); those docs say how, and where QSM.rs departs from it. An entry with none either
//! follows its paper alone or is QSM.rs's own (see the module docs). A network's weights and
//! upstream repository are in its [`ModelSpec`](crate::models::ModelSpec).

use super::citations::*;
use super::Stage::*;
use super::{Method, Reference};

// ---- Reference code ---------------------------------------------------------------------------

const QSM_JL: Reference = Reference {
    name: "QSM.jl", version: Some("fde4b92"), url: Some("https://github.com/kamesy/QSM.jl"),
};
const MEDI_TOOLBOX: Reference = Reference {
    name: "MEDI toolbox", version: Some("2024.11.26"), url: Some("http://pre.weill.cornell.edu/mri/pages/qsm.html"),
};
const MSMV_MEDI: Reference = Reference {
    name: "mSMV (MEDI toolbox prerelease MEDI_toolbox-mSMV100125)", version: Some("42a1f18"),
    url: Some("https://github.com/agr78/mSMV"),
};
const STI_SUITE: Reference = Reference { name: "STI Suite", version: Some("3.0"), url: None };
const FANSI: Reference = Reference {
    name: "FANSI toolbox", version: Some("93044a3"), url: Some("https://gitlab.com/cmilovic/FANSI-toolbox"),
};
const MRI_RESEARCH_TOOLS: Reference = Reference {
    name: "MriResearchTools.jl", version: Some("771eadf"), url: Some("https://github.com/korbinian90/MriResearchTools.jl"),
};
const QSMART_TOOLBOX: Reference = Reference {
    name: "QSMART toolbox", version: Some("v1.0"), url: Some("https://github.com/wtsyeda/QSMART"),
};
const QSM_HEIDI: Reference = Reference {
    name: "qsm_heidi", version: Some("4592631"), url: Some("https://gitlab.com/R01NS114227/qsm_heidi"),
};
const CHI_SEPARATION_TOOLBOX: Reference = Reference {
    name: "χ-separation toolbox", version: Some("v1.1.3"), url: Some("https://github.com/SNU-LIST/chi-separation"),
};
const QSM_CI: Reference = Reference { name: "QSM-CI", version: None, url: Some("https://github.com/QSMxT/QSM-CI") };

// ---- Entries ----------------------------------------------------------------------------------

/// Every method, grouped by stage in pipeline order. See the [module docs](super) for adding one.
pub static METHODS: &[Method] = &[
    // ---- Masking: generators and refinements (`MaskOp`) ----
    Method { id: "threshold", stage: Masking, name: "Thresholding", citations: &[], reference: None, models: &[] },
    Method { id: "bet", stage: Masking, name: "BET", citations: &[SMITH2002],
        reference: Some(Reference { name: "FSL-BET2", version: Some("d5acd7f"), url: Some("https://github.com/Bostrix/FSL-BET2") }),
        models: &[] },
    Method { id: "hd-bet", stage: Masking, name: "HD-BET", citations: &[ISENSEE2019],
        reference: Some(Reference { name: "HD-BET", version: Some("678e44d"), url: Some("https://github.com/MIC-DKFZ/HD-BET") }),
        models: &["hd-bet"] },
    Method { id: "rs2-net", stage: Masking, name: "RS2-Net", citations: &[LIN2024],
        reference: Some(Reference { name: "Rodent-Skull-Stripping", version: None, url: Some("https://github.com/VitoLin21/Rodent-Skull-Stripping") }),
        models: &["rs2-net"] },
    Method { id: "mindgrab", stage: Masking, name: "MindGrab", citations: &[FANI2025],
        reference: Some(Reference { name: "brainchop-cli", version: None, url: Some("https://github.com/neuroneural/brainchop-cli") }),
        models: &["mindgrab"] },
    Method { id: "erode", stage: Masking, name: "Erosion", citations: &[], reference: None, models: &[] },
    Method { id: "dilate", stage: Masking, name: "Dilation", citations: &[], reference: None, models: &[] },
    Method { id: "close", stage: Masking, name: "Closing", citations: &[], reference: None, models: &[] },
    Method { id: "min-intensity", stage: Masking, name: "Minimum intensity", citations: &[], reference: Some(QSMART_TOOLBOX), models: &[] },
    Method { id: "majority", stage: Masking, name: "Majority filter", citations: &[], reference: Some(QSMART_TOOLBOX), models: &[] },
    Method { id: "fill-holes", stage: Masking, name: "Hole filling", citations: &[], reference: None, models: &[] },
    Method { id: "largest-component", stage: Masking, name: "Largest connected component", citations: &[], reference: Some(STI_SUITE), models: &[] },
    Method { id: "imfill", stage: Masking, name: "Hole filling (imfill)", citations: &[], reference: None, models: &[] },
    Method { id: "gaussian-smooth", stage: Masking, name: "Gaussian smoothing", citations: &[], reference: None, models: &[] },
    Method { id: "signal-erode", stage: Masking, name: "Signal-gated erosion", citations: &[],
        reference: Some(Reference { name: "QSM-CI hd-bet-qsmci submission", version: None, url: Some("https://github.com/QSMxT/QSM-CI") }),
        models: &[] },
    // Not a `MaskOp`: UK Biobank's trims around V-SHARP (`utils::phase_reliability`).
    Method { id: "phase-reliability-trim", stage: Masking, name: "UK Biobank phase-reliability trimming", citations: &[WANG2022],
        reference: Some(Reference { name: "UK Biobank QSM pipeline", version: None, url: Some("https://git.fmrib.ox.ac.uk/cwang/uk_biobank_qsm_pipeline") }),
        models: &[] },
    // ---- Mask thresholds (`MaskThresholdMethod`) ----
    Method { id: "otsu", stage: MaskThreshold, name: "Otsu", citations: &[OTSU1979], reference: None, models: &[] },
    Method { id: "fixed", stage: MaskThreshold, name: "Fixed threshold", citations: &[], reference: None, models: &[] },
    Method { id: "percentile", stage: MaskThreshold, name: "Percentile threshold", citations: &[], reference: None, models: &[] },
    Method { id: "sti-gui", stage: MaskThreshold, name: "STI Suite GUI threshold", citations: &[], reference: Some(STI_SUITE), models: &[] },
    // ---- Receive-field bias correction (`InhomogeneityCorrection`) ----
    Method { id: "makehomogeneous", stage: InhomogeneityCorrection, name: "makehomogeneous", citations: &[ECKSTEIN2019],
        reference: Some(MRI_RESEARCH_TOOLS), models: &[] },
    Method { id: "n4", stage: InhomogeneityCorrection, name: "N4", citations: &[TUSTISON2010],
        reference: Some(Reference { name: "ANTs", version: None, url: Some("https://github.com/ANTsX/ANTs") }),
        models: &[] },
    // ---- Phase corrections before unwrapping (`FieldMappingConfig` switches) ----
    Method { id: "mcpc-3d-s", stage: PhaseCorrection, name: "MCPC-3D-S", citations: &[ECKSTEIN2018],
        reference: Some(MRI_RESEARCH_TOOLS), models: &[] },
    Method { id: "bipolar-correction", stage: PhaseCorrection, name: "Bipolar correction", citations: &[ECKSTEIN2021PHD],
        reference: Some(MRI_RESEARCH_TOOLS), models: &[] },
    // ---- Phase unwrapping (`UnwrappingAlgorithm`) ----
    Method { id: "romeo", stage: Unwrapping, name: "ROMEO", citations: &[DYMERSKA2021],
        reference: Some(Reference { name: "ROMEO.jl", version: Some("08776d4"), url: Some("https://github.com/korbinian90/ROMEO.jl") }),
        models: &[] },
    Method { id: "laplacian", stage: Unwrapping, name: "Laplacian", citations: &[SCHOFIELD2003], reference: Some(QSM_JL), models: &[] },
    Method { id: "prelude", stage: Unwrapping, name: "PRELUDE", citations: &[JENKINSON2003],
        reference: Some(Reference { name: "FSL prelude (fsl-fugue)", version: Some("2603.0"), url: None }),
        models: &[] },
    Method { id: "medi-region-growing", stage: Unwrapping, name: "MEDI region growing", citations: &[LIU2013NONLINEAR],
        reference: Some(MEDI_TOOLBOX), models: &[] },
    // ---- Echo combination into a B0 map (`B0EstimationMethod`) ----
    Method { id: "weighted-avg", stage: EchoCombination, name: "Weighted average", citations: &[],
        reference: Some(MRI_RESEARCH_TOOLS), models: &[] },
    Method { id: "linear-fit", stage: EchoCombination, name: "Linear fit", citations: &[], reference: Some(QSM_JL), models: &[] },
    Method { id: "robust-fit", stage: EchoCombination, name: "Robust fit", citations: &[], reference: None, models: &[] },
    Method { id: "complex-fit", stage: EchoCombination, name: "Nonlinear complex fit", citations: &[LIU2013NONLINEAR],
        reference: Some(MEDI_TOOLBOX), models: &[] },
    Method { id: "qsmart-echofit", stage: EchoCombination, name: "QSMART echo fit", citations: &[YAGHMAIE2021],
        reference: Some(QSMART_TOOLBOX), models: &[] },
    // ---- Background field removal (`BgRemovalAlgorithm`) ----
    Method { id: "vsharp", stage: BackgroundRemoval, name: "V-SHARP", citations: &[WU2012], reference: Some(QSM_JL), models: &[] },
    Method { id: "pdf", stage: BackgroundRemoval, name: "PDF", citations: &[LIU2011PDF], reference: Some(MEDI_TOOLBOX), models: &[] },
    Method { id: "lbv", stage: BackgroundRemoval, name: "LBV", citations: &[ZHOU2014], reference: Some(MEDI_TOOLBOX), models: &[] },
    Method { id: "ismv", stage: BackgroundRemoval, name: "iSMV", citations: &[WEN2014],
        reference: Some(Reference { name: "mSMV", version: Some("42a1f18"), url: Some("https://github.com/agr78/mSMV") }),
        models: &[] },
    Method { id: "sharp", stage: BackgroundRemoval, name: "SHARP", citations: &[SCHWESER2011],
        reference: Some(Reference { name: "SEPIA", version: Some("db939cc"), url: Some("https://github.com/kschan0214/sepia") }),
        models: &[] },
    Method { id: "resharp", stage: BackgroundRemoval, name: "RESHARP", citations: &[SUN2014],
        reference: Some(Reference { name: "sunhongfu/QSM", version: Some("c75b138"), url: Some("https://github.com/sunhongfu/QSM") }),
        models: &[] },
    Method { id: "harperella", stage: BackgroundRemoval, name: "HARPERELLA", citations: &[LI2014], reference: None, models: &[] },
    Method { id: "iharperella", stage: BackgroundRemoval, name: "iHARPERELLA", citations: &[LI2015IHARP], reference: Some(STI_SUITE), models: &[] },
    Method { id: "bfrnet", stage: BackgroundRemoval, name: "BFRnet", citations: &[ZHU2023BFRNET], reference: None, models: &["bfrnet"] },
    // Not a `BgRemovalAlgorithm`: run by `pipeline::run_iqfm` from wrapped phase.
    Method { id: "iqfm", stage: BackgroundRemoval, name: "iQFM", citations: &[GAO2022IQSM], reference: None, models: &["iqfm"] },
    // ---- Dipole inversion (`InversionAlgorithm`) ----
    Method { id: "tkd", stage: Inversion, name: "TKD", citations: &[SHMUELI2009], reference: Some(QSM_JL), models: &[] },
    Method { id: "tsvd", stage: Inversion, name: "TSVD", citations: &[SHMUELI2009], reference: Some(QSM_JL), models: &[] },
    Method { id: "tikhonov", stage: Inversion, name: "Tikhonov", citations: &[BILGIC2014L2], reference: Some(QSM_JL), models: &[] },
    Method { id: "tv", stage: Inversion, name: "TV-ADMM", citations: &[BILGIC2014TV], reference: Some(QSM_JL), models: &[] },
    Method { id: "rts", stage: Inversion, name: "RTS", citations: &[KAMES2018], reference: Some(QSM_JL), models: &[] },
    Method { id: "nltv", stage: Inversion, name: "NLTV", citations: &[MILOVIC2018], reference: Some(QSM_JL), models: &[] },
    Method { id: "medi", stage: Inversion, name: "MEDI", citations: &[LIU2011MEDI], reference: Some(MEDI_TOOLBOX), models: &[] },
    Method { id: "tfi", stage: Inversion, name: "TFI", citations: &[LIU2017TFI], reference: Some(MEDI_TOOLBOX), models: &[] },
    Method { id: "ilsqr", stage: Inversion, name: "iLSQR", citations: &[LI2015], reference: Some(STI_SUITE), models: &[] },
    Method { id: "star", stage: Inversion, name: "STAR-QSM", citations: &[WEI2015STAR], reference: Some(STI_SUITE), models: &[] },
    Method { id: "tgv", stage: Inversion, name: "TGV-QSM", citations: &[LANGKAMMER2015],
        reference: Some(Reference {
            name: "QuantitativeSusceptibilityMappingTGV.jl", version: Some("6c0886a"),
            url: Some("https://github.com/korbinian90/QuantitativeSusceptibilityMappingTGV.jl"),
        }),
        models: &[] },
    Method { id: "qsmart", stage: Inversion, name: "QSMART", citations: &[YAGHMAIE2021], reference: Some(QSMART_TOOLBOX), models: &[] },
    Method { id: "ndi", stage: Inversion, name: "NDI", citations: &[POLAK2020], reference: Some(FANSI), models: &[] },
    Method { id: "fansi", stage: Inversion, name: "FANSI nonlinear TV", citations: &[MILOVIC2018], reference: Some(FANSI), models: &[] },
    Method { id: "fansi-tgv", stage: Inversion, name: "FANSI nonlinear TGV", citations: &[MILOVIC2018], reference: Some(FANSI), models: &[] },
    Method { id: "l1qsm", stage: Inversion, name: "L1-QSM", citations: &[MILOVIC2022], reference: Some(FANSI), models: &[] },
    Method { id: "whqsm", stage: Inversion, name: "WH-QSM", citations: &[MILOVIC2019], reference: Some(FANSI), models: &[] },
    Method { id: "hdqsm", stage: Inversion, name: "HD-QSM", citations: &[LAMBERT2022],
        reference: Some(Reference { name: "HD-QSM", version: Some("c456a94"), url: Some("https://github.com/mglambert/HD-QSM") }),
        models: &[] },
    Method { id: "amp-pe", stage: Inversion, name: "AMP-PE", citations: &[HUANG2023AMPPE],
        reference: Some(Reference { name: "QSM_AMP_PE", version: Some("15bf8dc"), url: Some("https://github.com/EmoryCN2L/QSM_AMP_PE") }),
        models: &[] },
    Method { id: "lsqr", stage: Inversion, name: "LSQR", citations: &[SCHWESER2010], reference: Some(QSM_HEIDI), models: &[] },
    Method { id: "xqsm", stage: Inversion, name: "xQSM", citations: &[GAO2021XQSM], reference: None, models: &["xqsm"] },
    Method { id: "qsmnet", stage: Inversion, name: "QSMnet", citations: &[YOON2018QSMNET], reference: None, models: &["qsmnet"] },
    Method { id: "qsmnet-plus", stage: Inversion, name: "QSMnet+", citations: &[JUNG2020QSMNETPLUS], reference: None, models: &["qsmnet-plus"] },
    Method { id: "qsmnetme", stage: Inversion, name: "QSMnetME", citations: &[KIM2025CHISEPNET],
        reference: Some(CHI_SEPARATION_TOOLBOX), models: &["qsmnetme"] },
    Method { id: "autoqsm", stage: Inversion, name: "AutoQSM", citations: &[WEI2019AUTOQSM], reference: None, models: &["autoqsm"] },
    Method { id: "qsmgan", stage: Inversion, name: "QSMGAN", citations: &[CHEN2020QSMGAN], reference: None, models: &["qsmgan"] },
    Method { id: "ir2qsm", stage: Inversion, name: "IR2QSM", citations: &[LI2025IR2QSM], reference: None, models: &["ir2qsm"] },
    Method { id: "lpcnn", stage: Inversion, name: "LPCNN", citations: &[LAI2020LPCNN], reference: None, models: &["lpcnn"] },
    Method { id: "modl-qsm", stage: Inversion, name: "MoDL-QSM", citations: &[FENG2021MODLQSM], reference: None, models: &["modl-qsm"] },
    Method { id: "nextqsm", stage: Inversion, name: "NeXtQSM", citations: &[COGNOLATO2023NEXTQSM], reference: None, models: &["nextqsm"] },
    Method { id: "iqsm", stage: Inversion, name: "iQSM", citations: &[GAO2022IQSM], reference: None, models: &["iqsm"] },
    Method { id: "iqsm-plus", stage: Inversion, name: "iQSM+", citations: &[GAO2024IQSMPLUS], reference: None, models: &["iqsm-plus"] },
    // ---- Additions to an inversion ----
    Method { id: "medi-zero-reference", stage: InversionAddition, name: "MEDI+0", citations: &[LIU2018MEDI0],
        reference: Some(MEDI_TOOLBOX), models: &[] },
    Method { id: "msmv", stage: InversionAddition, name: "mSMV", citations: &[ROBERTS2024], reference: Some(MSMV_MEDI), models: &[] },
    Method { id: "heidi", stage: InversionAddition, name: "HEIDI", citations: &[SCHWESER2012], reference: Some(QSM_HEIDI), models: &[] },
    Method { id: "two-pass", stage: InversionAddition, name: "Two-pass artefact reduction", citations: &[STEWART2022],
        reference: None, models: &[] },
    // ---- Relaxometry ----
    Method { id: "arlo", stage: Relaxometry, name: "ARLO", citations: &[PEI2015], reference: Some(MEDI_TOOLBOX), models: &[] },
    Method { id: "epg", stage: Relaxometry, name: "EPG R2 mapping", citations: &[WEIGEL2015], reference: None, models: &[] },
    Method { id: "r2primenet", stage: Relaxometry, name: "R2PRIMEnet", citations: &[KIM2025CHISEPNET],
        reference: Some(CHI_SEPARATION_TOOLBOX), models: &["r2primenet"] },
    // ---- Susceptibility source separation (`SeparationAlgorithm`) ----
    Method { id: "chi-sep-ilsqr", stage: Separation, name: "χ-separation (iLSQR)", citations: &[SHIN2021CHISEP],
        reference: Some(CHI_SEPARATION_TOOLBOX), models: &[] },
    Method { id: "chi-sep-medi", stage: Separation, name: "χ-separation (MEDI)", citations: &[SHIN2021CHISEP],
        reference: Some(CHI_SEPARATION_TOOLBOX), models: &[] },
    Method { id: "r2star-qsm", stage: Separation, name: "R2*-QSM", citations: &[DIMOV2022, DIMOV2022TOMOGRAPHY],
        reference: Some(QSM_CI), models: &[] },
    Method { id: "wavesep", stage: Separation, name: "WaveSep", citations: &[FANG2023WAVESEP],
        reference: Some(Reference { name: "WaveSep", version: None, url: Some("https://github.com/ZhenghanFang/WaveSep") }),
        models: &[] },
    Method { id: "decompose", stage: Separation, name: "DECOMPOSE-QSM", citations: &[CHEN2021DECOMPOSE],
        reference: Some(Reference { name: "DECOMPOSE-QSM (QSM-CI port)", version: None, url: Some("https://github.com/QSMxT/QSM-CI") }),
        models: &[] },
    Method { id: "hc-chisep", stage: Separation, name: "HC-ChiSep", citations: &[STEWART2026, WHARTON2012],
        reference: Some(Reference { name: "QSM-CI hc-chisep-qsmci submission", version: None, url: Some("https://github.com/QSMxT/QSM-CI") }),
        models: &[] },
    Method { id: "susep-net", stage: Separation, name: "SUSEP-Net", citations: &[LI2025SUSEPNET],
        reference: Some(Reference { name: "SUSEP-Net", version: None, url: Some("https://github.com/YangGaoUQ/SUSEP-Net") }),
        models: &["susep-net"] },
    Method { id: "chi-sepnet", stage: Separation, name: "χ-sepnet", citations: &[KIM2025CHISEPNET],
        reference: Some(CHI_SEPARATION_TOOLBOX), models: &["chi-sepnet", "qsmnetme", "r2primenet"] },
    // ---- Susceptibility-weighted imaging ----
    Method { id: "clear-swi", stage: Swi, name: "CLEAR-SWI", citations: &[ECKSTEIN2021],
        reference: Some(Reference { name: "CLEARSWI.jl", version: Some("0949bc5"), url: Some("https://github.com/korbinian90/CLEARSWI.jl") }),
        models: &[] },
    Method { id: "smwi", stage: Swi, name: "SMWI", citations: &[GHO2014],
        reference: Some(Reference { name: "SEPIA", version: None, url: Some("https://github.com/kschan0214/sepia") }),
        models: &[] },
    // ---- Segmentation (`SynthSegVersion`) ----
    Method { id: "synthseg", stage: Segmentation, name: "SynthSeg 1.0", citations: &[BILLOT2023],
        reference: Some(Reference { name: "SynthSeg", version: Some("2a2aa3b"), url: Some("https://github.com/BBillot/SynthSeg") }),
        models: &["synthseg"] },
    Method { id: "synthseg-2.0", stage: Segmentation, name: "SynthSeg 2.0", citations: &[BILLOT2023, BILLOT2023PNAS],
        reference: Some(Reference { name: "SynthSeg", version: Some("2a2aa3b"), url: Some("https://github.com/BBillot/SynthSeg") }),
        models: &["synthseg-2.0"] },
    // ---- Multiple orientations ----
    Method { id: "cosmos", stage: MultiOrientation, name: "COSMOS", citations: &[LIU2009COSMOS], reference: None, models: &[] },
    Method { id: "sti", stage: MultiOrientation, name: "STI", citations: &[LIU2010STI], reference: None, models: &[] },
];

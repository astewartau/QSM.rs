//! The papers the catalogue cites, one constant per paper.
//!
//! Each is the authors' publication of a method, checked against the publisher's record (Crossref
//! or DataCite) for its DOI, venue, volume, issue and pages. `authors` is written as a methods
//! section cites it: every author for a short list, otherwise the first one or few and "et al.".
//! A title is the published title; acronyms a front end adds for its readers are not part of it.
//! `key` is a citation key in the usual `surnameYEARsuffix` form; hosts use it to collect and
//! de-duplicate references, so a paper keeps its key once published here.

use super::Citation;

/// The empty citation, for struct-update syntax: every entry sets `key`, `authors`, `year` and
/// `title`, and whichever optional fields its paper has.
const BLANK: Citation = Citation {
    key: "",
    authors: "",
    year: 0,
    title: "",
    venue: None,
    volume: None,
    issue: None,
    pages: None,
    doi: None,
    url: None,
};

// ---- Masking and bias correction ------------------------------------------------------------

pub(crate) const SMITH2002: Citation = Citation {
    key: "smith2002", authors: "Smith, S.M.", year: 2002,
    title: "Fast robust automated brain extraction",
    venue: Some("Human Brain Mapping"), volume: Some("17"), issue: Some("3"), pages: Some("143-155"),
    doi: Some("10.1002/hbm.10062"), ..BLANK
};

pub(crate) const ISENSEE2019: Citation = Citation {
    key: "isensee2019", authors: "Isensee, F., Schell, M., Pflueger, I., et al.", year: 2019,
    title: "Automated brain extraction of multisequence MRI using artificial neural networks",
    venue: Some("Human Brain Mapping"), volume: Some("40"), issue: Some("17"), pages: Some("4952-4964"),
    doi: Some("10.1002/hbm.24750"), ..BLANK
};

pub(crate) const LIN2024: Citation = Citation {
    key: "lin2024", authors: "Lin, Y., Ding, Y., Chang, S., Ge, X., Sui, X., Jiang, Y.", year: 2024,
    title: "RS2-Net: An end-to-end deep learning framework for rodent skull stripping in multi-center brain MRI",
    venue: Some("NeuroImage"), volume: Some("298"), pages: Some("120769"),
    doi: Some("10.1016/j.neuroimage.2024.120769"), ..BLANK
};

pub(crate) const FANI2025: Citation = Citation {
    key: "fani2025", authors: "Fani, A., Doan, M., Le, I., et al.", year: 2025,
    title: "MindGrab for BrainChop: Fast and Accurate Skull Stripping for Command Line and Browser",
    venue: Some("arXiv"), doi: Some("10.48550/arXiv.2506.11860"), ..BLANK
};

pub(crate) const OTSU1979: Citation = Citation {
    key: "otsu1979", authors: "Otsu, N.", year: 1979,
    title: "A Threshold Selection Method from Gray-Level Histograms",
    venue: Some("IEEE Transactions on Systems, Man, and Cybernetics"), volume: Some("9"), issue: Some("1"),
    pages: Some("62-66"), doi: Some("10.1109/TSMC.1979.4310076"), ..BLANK
};

pub(crate) const WANG2022: Citation = Citation {
    key: "wang2022", authors: "Wang, C., et al.", year: 2022,
    title: "Phenotypic and genetic associations of quantitative magnetic susceptibility in UK Biobank brain imaging",
    venue: Some("Nature Neuroscience"), volume: Some("25"), issue: Some("6"), pages: Some("818-831"),
    doi: Some("10.1038/s41593-022-01074-w"), ..BLANK
};

pub(crate) const ECKSTEIN2019: Citation = Citation {
    key: "eckstein2019", authors: "Eckstein, K., Trattnig, S., Robinson, S.D.", year: 2019,
    title: "A Simple Homogeneity Correction for Neuroimaging at 7T",
    venue: Some("Proc. ISMRM 27th Annual Meeting"), ..BLANK
};

pub(crate) const TUSTISON2010: Citation = Citation {
    key: "tustison2010",
    authors: "Tustison, N.J., Avants, B.B., Cook, P.A., Zheng, Y., Egan, A., Yushkevich, P.A., Gee, J.C.",
    year: 2010, title: "N4ITK: Improved N3 Bias Correction",
    venue: Some("IEEE Transactions on Medical Imaging"), volume: Some("29"), issue: Some("6"),
    pages: Some("1310-1320"), doi: Some("10.1109/TMI.2010.2046908"), ..BLANK
};

// ---- Phase correction, unwrapping and echo combination --------------------------------------

pub(crate) const ECKSTEIN2018: Citation = Citation {
    key: "eckstein2018", authors: "Eckstein, K., et al.", year: 2018,
    title: "Computationally Efficient Combination of Multi-channel Phase Data From Multi-echo Acquisitions (ASPIRE)",
    venue: Some("Magnetic Resonance in Medicine"), volume: Some("79"), pages: Some("2996-3006"),
    doi: Some("10.1002/mrm.26963"), ..BLANK
};

pub(crate) const ECKSTEIN2021PHD: Citation = Citation {
    key: "eckstein2021phd", authors: "Eckstein, K.", year: 2021,
    title: "Improved Methods for Ultra-High Field Magnetic Susceptibility Imaging",
    venue: Some("PhD thesis, TU Wien"), doi: Some("10.34726/hss.2021.43447"), ..BLANK
};

pub(crate) const DYMERSKA2021: Citation = Citation {
    key: "dymerska2021", authors: "Dymerska, B., et al.", year: 2021,
    title: "Phase unwrapping with a rapid opensource minimum spanning tree algorithm (ROMEO)",
    venue: Some("Magnetic Resonance in Medicine"), volume: Some("85"), issue: Some("4"), pages: Some("2294-2308"),
    doi: Some("10.1002/mrm.28563"), ..BLANK
};

pub(crate) const SCHOFIELD2003: Citation = Citation {
    key: "schofield2003", authors: "Schofield, M.A., Zhu, Y.", year: 2003,
    title: "Fast phase unwrapping algorithm for interferometric applications",
    venue: Some("Optics Letters"), volume: Some("28"), issue: Some("14"), pages: Some("1194-1196"),
    doi: Some("10.1364/OL.28.001194"), ..BLANK
};

pub(crate) const JENKINSON2003: Citation = Citation {
    key: "jenkinson2003", authors: "Jenkinson, M.", year: 2003,
    title: "Fast, automated, N-dimensional phase-unwrapping algorithm",
    venue: Some("Magnetic Resonance in Medicine"), volume: Some("49"), issue: Some("1"), pages: Some("193-197"),
    doi: Some("10.1002/mrm.10354"), ..BLANK
};

pub(crate) const LIU2013NONLINEAR: Citation = Citation {
    key: "liu2013nonlinear", authors: "Liu, T., et al.", year: 2013,
    title: "Nonlinear formulation of the magnetic field to source relationship for robust quantitative susceptibility mapping",
    venue: Some("Magnetic Resonance in Medicine"), volume: Some("69"), issue: Some("2"), pages: Some("467-476"),
    doi: Some("10.1002/mrm.24272"), ..BLANK
};

pub(crate) const YAGHMAIE2021: Citation = Citation {
    key: "yaghmaie2021", authors: "Yaghmaie, N., Syeda, W., et al.", year: 2021,
    title: "QSMART: Quantitative Susceptibility Mapping Artifact Reduction Technique",
    venue: Some("NeuroImage"), volume: Some("231"), pages: Some("117701"),
    doi: Some("10.1016/j.neuroimage.2020.117701"), ..BLANK
};

// ---- Background field removal ---------------------------------------------------------------

pub(crate) const WU2012: Citation = Citation {
    key: "wu2012", authors: "Wu, B., et al.", year: 2012,
    title: "Whole brain susceptibility mapping using compressed sensing",
    venue: Some("Magnetic Resonance in Medicine"), volume: Some("67"), issue: Some("1"), pages: Some("137-147"),
    doi: Some("10.1002/mrm.23000"), ..BLANK
};

pub(crate) const SCHWESER2011: Citation = Citation {
    key: "schweser2011", authors: "Schweser, F., et al.", year: 2011,
    title: "Quantitative imaging of intrinsic magnetic tissue properties using MRI signal phase: An approach to in vivo brain iron metabolism?",
    venue: Some("NeuroImage"), volume: Some("54"), issue: Some("4"), pages: Some("2789-2807"),
    doi: Some("10.1016/j.neuroimage.2010.10.070"), ..BLANK
};

pub(crate) const LIU2011PDF: Citation = Citation {
    key: "liu2011pdf", authors: "Liu, T., et al.", year: 2011,
    title: "A novel background field removal method for MRI using projection onto dipole fields (PDF)",
    venue: Some("NMR in Biomedicine"), volume: Some("24"), issue: Some("9"), pages: Some("1129-1136"),
    doi: Some("10.1002/nbm.1670"), ..BLANK
};

pub(crate) const WEN2014: Citation = Citation {
    key: "wen2014", authors: "Wen, Y., et al.", year: 2014,
    title: "An iterative spherical mean value method for background field removal in MRI",
    venue: Some("Magnetic Resonance in Medicine"), volume: Some("72"), issue: Some("4"), pages: Some("1065-1071"),
    doi: Some("10.1002/mrm.24998"), ..BLANK
};

pub(crate) const ZHOU2014: Citation = Citation {
    key: "zhou2014", authors: "Zhou, D., et al.", year: 2014,
    title: "Background field removal by solving the Laplacian boundary value problem",
    venue: Some("NMR in Biomedicine"), volume: Some("27"), issue: Some("3"), pages: Some("312-319"),
    doi: Some("10.1002/nbm.3064"), ..BLANK
};

pub(crate) const SUN2014: Citation = Citation {
    key: "sun2014", authors: "Sun, H., Wilman, A.H.", year: 2014,
    title: "Background field removal using spherical mean value filtering and Tikhonov regularization",
    venue: Some("Magnetic Resonance in Medicine"), volume: Some("71"), issue: Some("3"), pages: Some("1151-1157"),
    doi: Some("10.1002/mrm.24765"), ..BLANK
};

pub(crate) const LI2014: Citation = Citation {
    key: "li2014", authors: "Li, W., Avram, A.V., Wu, B., Xiao, X., Liu, C.", year: 2014,
    title: "Integrated Laplacian-based phase unwrapping and background phase removal for quantitative susceptibility mapping",
    venue: Some("NMR in Biomedicine"), volume: Some("27"), issue: Some("2"), pages: Some("219-227"),
    doi: Some("10.1002/nbm.3056"), ..BLANK
};

pub(crate) const LI2015IHARP: Citation = Citation {
    key: "li2015iharp", authors: "Li, W., Wu, B., Liu, C.", year: 2015,
    title: "iHARPERELLA: an improved method for integrated 3D phase unwrapping and background phase removal",
    venue: Some("Proc. ISMRM"), volume: Some("23"), pages: Some("3313"), ..BLANK
};

pub(crate) const ZHU2023BFRNET: Citation = Citation {
    key: "zhu2023bfrnet", authors: "Zhu, X., et al.", year: 2023,
    title: "BFRnet: A deep learning-based MR background field removal method for QSM of the brain containing significant pathological susceptibility sources",
    venue: Some("Zeitschrift für Medizinische Physik"), volume: Some("33"), issue: Some("4"), pages: Some("578-590"),
    doi: Some("10.1016/j.zemedi.2022.08.001"), ..BLANK
};

pub(crate) const GAO2022IQSM: Citation = Citation {
    key: "gao2022iqsm", authors: "Gao, Y., et al.", year: 2022,
    title: "Instant tissue field and magnetic susceptibility mapping from MRI raw phase using Laplacian enhanced deep neural networks",
    venue: Some("NeuroImage"), volume: Some("259"), pages: Some("119410"),
    doi: Some("10.1016/j.neuroimage.2022.119410"), ..BLANK
};

// ---- Dipole inversion -----------------------------------------------------------------------

pub(crate) const SHMUELI2009: Citation = Citation {
    key: "shmueli2009", authors: "Shmueli, K., et al.", year: 2009,
    title: "Magnetic susceptibility mapping of brain tissue in vivo using MRI phase data",
    venue: Some("Magnetic Resonance in Medicine"), volume: Some("62"), issue: Some("6"), pages: Some("1510-1522"),
    doi: Some("10.1002/mrm.22135"), ..BLANK
};

pub(crate) const BILGIC2014L2: Citation = Citation {
    key: "bilgic2014l2", authors: "Bilgic, B., et al.", year: 2014,
    title: "Fast image reconstruction with L2-regularization",
    venue: Some("Journal of Magnetic Resonance Imaging"), volume: Some("40"), issue: Some("1"), pages: Some("181-191"),
    doi: Some("10.1002/jmri.24365"), ..BLANK
};

pub(crate) const BILGIC2014TV: Citation = Citation {
    key: "bilgic2014tv", authors: "Bilgic, B., et al.", year: 2014,
    title: "Fast quantitative susceptibility mapping with L1-regularization and automatic parameter selection",
    venue: Some("Magnetic Resonance in Medicine"), volume: Some("72"), issue: Some("5"), pages: Some("1444-1459"),
    doi: Some("10.1002/mrm.25029"), ..BLANK
};

pub(crate) const KAMES2018: Citation = Citation {
    key: "kames2018", authors: "Kames, C., Wiggermann, V., Rauscher, A.", year: 2018,
    title: "Rapid two-step dipole inversion for susceptibility mapping with sparsity priors",
    venue: Some("NeuroImage"), volume: Some("167"), pages: Some("276-283"),
    doi: Some("10.1016/j.neuroimage.2017.11.018"), ..BLANK
};

pub(crate) const MILOVIC2018: Citation = Citation {
    key: "milovic2018", authors: "Milovic, C., Bilgic, B., Zhao, B., Acosta-Cabronero, J., Tejos, C.", year: 2018,
    title: "Fast nonlinear susceptibility inversion with variational regularization",
    venue: Some("Magnetic Resonance in Medicine"), volume: Some("80"), issue: Some("2"), pages: Some("814-821"),
    doi: Some("10.1002/mrm.27073"), ..BLANK
};

pub(crate) const LIU2011MEDI: Citation = Citation {
    key: "liu2011medi", authors: "Liu, T., et al.", year: 2011,
    title: "Morphology enabled dipole inversion (MEDI) from a single-angle acquisition: Comparison with COSMOS in human brain imaging",
    venue: Some("Magnetic Resonance in Medicine"), volume: Some("66"), issue: Some("3"), pages: Some("777-783"),
    doi: Some("10.1002/mrm.22816"), ..BLANK
};

pub(crate) const LIU2018MEDI0: Citation = Citation {
    key: "liu2018medi0", authors: "Liu, Z., et al.", year: 2018,
    title: "MEDI+0: Morphology enabled dipole inversion with automatic uniform cerebrospinal fluid zero reference for quantitative susceptibility mapping",
    venue: Some("Magnetic Resonance in Medicine"), volume: Some("79"), issue: Some("5"), pages: Some("2795-2803"),
    doi: Some("10.1002/mrm.26946"), ..BLANK
};

pub(crate) const ROBERTS2024: Citation = Citation {
    key: "roberts2024", authors: "Roberts, A.G., et al.", year: 2024,
    title: "Maximum spherical mean value filtering for whole-brain QSM",
    venue: Some("Magnetic Resonance in Medicine"), volume: Some("91"), issue: Some("4"), pages: Some("1586-1597"),
    doi: Some("10.1002/mrm.29963"), ..BLANK
};

pub(crate) const LIU2017TFI: Citation = Citation {
    key: "liu2017tfi", authors: "Liu, Z., et al.", year: 2017,
    title: "Preconditioned total field inversion (TFI) method for quantitative susceptibility mapping",
    venue: Some("Magnetic Resonance in Medicine"), volume: Some("78"), issue: Some("1"), pages: Some("303-315"),
    doi: Some("10.1002/mrm.26331"), ..BLANK
};

pub(crate) const LI2015: Citation = Citation {
    key: "li2015", authors: "Li, W., et al.", year: 2015,
    title: "A method for estimating and removing streaking artifacts in quantitative susceptibility mapping",
    venue: Some("NeuroImage"), volume: Some("108"), pages: Some("111-122"),
    doi: Some("10.1016/j.neuroimage.2014.12.043"), ..BLANK
};

pub(crate) const WEI2015STAR: Citation = Citation {
    key: "wei2015star", authors: "Wei, H., Dibb, R., Zhou, Y., Sun, Y., Xu, J., Wang, N., Liu, C.", year: 2015,
    title: "Streaking artifact reduction for quantitative susceptibility mapping of sources with large dynamic range",
    venue: Some("NMR in Biomedicine"), volume: Some("28"), issue: Some("10"), pages: Some("1294-1303"),
    doi: Some("10.1002/nbm.3383"), ..BLANK
};

pub(crate) const LANGKAMMER2015: Citation = Citation {
    key: "langkammer2015", authors: "Langkammer, C., et al.", year: 2015,
    title: "Fast quantitative susceptibility mapping using 3D EPI and total generalized variation",
    venue: Some("NeuroImage"), volume: Some("111"), pages: Some("622-630"),
    doi: Some("10.1016/j.neuroimage.2015.02.041"), ..BLANK
};

pub(crate) const POLAK2020: Citation = Citation {
    key: "polak2020", authors: "Polak, D., et al.", year: 2020,
    title: "Nonlinear dipole inversion (NDI) enables robust quantitative susceptibility mapping (QSM)",
    venue: Some("NMR in Biomedicine"), volume: Some("33"), issue: Some("12"), pages: Some("e4271"),
    doi: Some("10.1002/nbm.4271"), ..BLANK
};

pub(crate) const MILOVIC2022: Citation = Citation {
    key: "milovic2022", authors: "Milovic, C., et al.", year: 2022,
    title: "Streaking artifact suppression of quantitative susceptibility mapping reconstructions via L1-norm data fidelity optimization (L1-QSM)",
    venue: Some("Magnetic Resonance in Medicine"), volume: Some("87"), issue: Some("1"), pages: Some("457-473"),
    doi: Some("10.1002/mrm.28957"), ..BLANK
};

pub(crate) const MILOVIC2019: Citation = Citation {
    key: "milovic2019", authors: "Milovic, C., et al.", year: 2019,
    title: "Weak-harmonic regularization for quantitative susceptibility mapping",
    venue: Some("Magnetic Resonance in Medicine"), volume: Some("81"), issue: Some("2"), pages: Some("1399-1411"),
    doi: Some("10.1002/mrm.27483"), ..BLANK
};

pub(crate) const LAMBERT2022: Citation = Citation {
    key: "lambert2022", authors: "Lambert, M., et al.", year: 2022,
    title: "Hybrid data fidelity term approach for quantitative susceptibility mapping",
    venue: Some("Magnetic Resonance in Medicine"), volume: Some("88"), issue: Some("2"), pages: Some("962-972"),
    doi: Some("10.1002/mrm.29218"), ..BLANK
};

pub(crate) const HUANG2023AMPPE: Citation = Citation {
    key: "huang2023amppe", authors: "Huang, S., et al.", year: 2023,
    title: "Robust quantitative susceptibility mapping via approximate message passing with parameter estimation",
    venue: Some("Magnetic Resonance in Medicine"), volume: Some("90"), issue: Some("4"), pages: Some("1414-1430"),
    doi: Some("10.1002/mrm.29722"), ..BLANK
};

pub(crate) const SCHWESER2010: Citation = Citation {
    key: "schweser2010", authors: "Schweser, F., Deistung, A., Lehr, B.W., Reichenbach, J.R.", year: 2010,
    title: "Differentiation between diamagnetic and paramagnetic cerebral lesions based on magnetic susceptibility mapping",
    venue: Some("Medical Physics"), volume: Some("37"), issue: Some("10"), pages: Some("5165-5178"),
    doi: Some("10.1118/1.3481505"), ..BLANK
};

pub(crate) const SCHWESER2012: Citation = Citation {
    key: "schweser2012", authors: "Schweser, F., Sommer, K., Deistung, A., Reichenbach, J.R.", year: 2012,
    title: "Quantitative susceptibility mapping for investigating subtle susceptibility variations in the human brain",
    venue: Some("NeuroImage"), volume: Some("62"), issue: Some("3"), pages: Some("2083-2100"),
    doi: Some("10.1016/j.neuroimage.2012.05.067"), ..BLANK
};

pub(crate) const STEWART2022: Citation = Citation {
    key: "stewart2022", authors: "Stewart, A.W., Robinson, S.D., O'Brien, K., et al.", year: 2022,
    title: "QSMxT: Robust masking and artifact reduction for quantitative susceptibility mapping",
    venue: Some("Magnetic Resonance in Medicine"), volume: Some("87"), issue: Some("3"), pages: Some("1289-1300"),
    doi: Some("10.1002/mrm.29048"), ..BLANK
};

pub(crate) const GAO2021XQSM: Citation = Citation {
    key: "gao2021xqsm", authors: "Gao, Y., et al.", year: 2021,
    title: "xQSM: quantitative susceptibility mapping with octave convolutional and noise-regularized neural networks",
    venue: Some("NMR in Biomedicine"), volume: Some("34"), issue: Some("3"), pages: Some("e4461"),
    doi: Some("10.1002/nbm.4461"), ..BLANK
};

pub(crate) const YOON2018QSMNET: Citation = Citation {
    key: "yoon2018qsmnet", authors: "Yoon, J., et al.", year: 2018,
    title: "Quantitative susceptibility mapping using deep neural network: QSMnet",
    venue: Some("NeuroImage"), volume: Some("179"), pages: Some("199-206"),
    doi: Some("10.1016/j.neuroimage.2018.06.030"), ..BLANK
};

pub(crate) const JUNG2020QSMNETPLUS: Citation = Citation {
    key: "jung2020qsmnetplus", authors: "Jung, W., et al.", year: 2020,
    title: "Exploring linearity of deep neural network trained QSM: QSMnet+",
    venue: Some("NeuroImage"), volume: Some("211"), pages: Some("116619"),
    doi: Some("10.1016/j.neuroimage.2020.116619"), ..BLANK
};

pub(crate) const WEI2019AUTOQSM: Citation = Citation {
    key: "wei2019autoqsm", authors: "Wei, H., et al.", year: 2019,
    title: "Learning-based single-step quantitative susceptibility mapping reconstruction without brain extraction",
    venue: Some("NeuroImage"), volume: Some("202"), pages: Some("116064"),
    doi: Some("10.1016/j.neuroimage.2019.116064"), ..BLANK
};

pub(crate) const CHEN2020QSMGAN: Citation = Citation {
    key: "chen2020qsmgan", authors: "Chen, Y., et al.", year: 2020,
    title: "QSMGAN: improved quantitative susceptibility mapping using 3D generative adversarial networks with increased receptive field",
    venue: Some("NeuroImage"), volume: Some("207"), pages: Some("116389"),
    doi: Some("10.1016/j.neuroimage.2019.116389"), ..BLANK
};

pub(crate) const LI2025IR2QSM: Citation = Citation {
    key: "li2025ir2qsm", authors: "Li, M., et al.", year: 2025,
    title: "Quantitative susceptibility mapping via deep neural networks with iterative reverse concatenations and recurrent modules",
    venue: Some("Medical Physics"), volume: Some("52"), issue: Some("6"), pages: Some("4341-4354"),
    doi: Some("10.1002/mp.17747"), ..BLANK
};

pub(crate) const LAI2020LPCNN: Citation = Citation {
    key: "lai2020lpcnn", authors: "Lai, K.-W., et al.", year: 2020,
    title: "Learned proximal networks for quantitative susceptibility mapping",
    venue: Some("MICCAI 2020"), volume: Some("LNCS 12262"), pages: Some("125-135"),
    doi: Some("10.1007/978-3-030-59713-9_13"), ..BLANK
};

pub(crate) const FENG2021MODLQSM: Citation = Citation {
    key: "feng2021modlqsm", authors: "Feng, R., et al.", year: 2021,
    title: "MoDL-QSM: model-based deep learning for quantitative susceptibility mapping",
    venue: Some("NeuroImage"), volume: Some("240"), pages: Some("118376"),
    doi: Some("10.1016/j.neuroimage.2021.118376"), ..BLANK
};

pub(crate) const COGNOLATO2023NEXTQSM: Citation = Citation {
    key: "cognolato2023nextqsm", authors: "Cognolato, F., et al.", year: 2023,
    title: "NeXtQSM — a complete deep learning pipeline for data-consistent quantitative susceptibility mapping trained with hybrid data",
    venue: Some("Medical Image Analysis"), volume: Some("84"), pages: Some("102700"),
    doi: Some("10.1016/j.media.2022.102700"), ..BLANK
};

pub(crate) const GAO2024IQSMPLUS: Citation = Citation {
    key: "gao2024iqsmplus", authors: "Gao, Y., et al.", year: 2024,
    title: "Plug-and-play latent feature editing for orientation-adaptive quantitative susceptibility mapping neural networks",
    venue: Some("Medical Image Analysis"), volume: Some("94"), pages: Some("103160"),
    doi: Some("10.1016/j.media.2024.103160"), ..BLANK
};

// ---- Source separation and relaxometry ------------------------------------------------------

pub(crate) const SHIN2021CHISEP: Citation = Citation {
    key: "shin2021chisep", authors: "Shin, H.-G., et al.", year: 2021,
    title: "χ-separation: Magnetic susceptibility source separation toward iron and myelin mapping in the brain",
    venue: Some("NeuroImage"), volume: Some("240"), pages: Some("118371"),
    doi: Some("10.1016/j.neuroimage.2021.118371"), ..BLANK
};

pub(crate) const DIMOV2022: Citation = Citation {
    key: "dimov2022", authors: "Dimov, A. V., et al.", year: 2022,
    title: "Susceptibility source separation from gradient echo data using magnitude decay modeling",
    venue: Some("Journal of Neuroimaging"), volume: Some("32"), issue: Some("5"), pages: Some("852-859"),
    doi: Some("10.1111/jon.13014"), ..BLANK
};

pub(crate) const DIMOV2022TOMOGRAPHY: Citation = Citation {
    key: "dimov2022tomography", authors: "Dimov, A. V., et al.", year: 2022,
    title: "Magnetic Susceptibility Source Separation Solely from Gradient Echo Data: Histological Validation",
    venue: Some("Tomography"), volume: Some("8"), issue: Some("3"), pages: Some("1544-1551"),
    doi: Some("10.3390/tomography8030127"), ..BLANK
};

pub(crate) const FANG2023WAVESEP: Citation = Citation {
    key: "fang2023wavesep", authors: "Fang, Z., Shin, H.-G., van Zijl, P., Li, X., Sulam, J.", year: 2023,
    title: "WaveSep: A Flexible Wavelet-Based Approach for Source Separation in Susceptibility Imaging",
    venue: Some("Machine Learning in Clinical Neuroimaging (MLCN 2023)"),
    pages: Some("56-66"), doi: Some("10.1007/978-3-031-44858-4_6"), ..BLANK
};

pub(crate) const CHEN2021DECOMPOSE: Citation = Citation {
    key: "chen2021decompose", authors: "Chen, J., et al.", year: 2021,
    title: "Decompose quantitative susceptibility mapping (QSM) to sub-voxel diamagnetic and paramagnetic components based on gradient-echo MRI data",
    venue: Some("NeuroImage"), volume: Some("242"), pages: Some("118477"),
    doi: Some("10.1016/j.neuroimage.2021.118477"), ..BLANK
};

/// HC-ChiSep has no published paper yet: its method is the QSM-CI submission (the catalogue
/// entry's `reference`), on the hollow-cylinder model of [`WHARTON2012`].
pub(crate) const STEWART2026: Citation = Citation {
    key: "stewart2026", authors: "Stewart, A., et al.", year: 2026,
    title: "Hollow-cylinder susceptibility source separation (HC-ChiSep)", ..BLANK
};

pub(crate) const WHARTON2012: Citation = Citation {
    key: "wharton2012", authors: "Wharton, S., Bowtell, R.", year: 2012,
    title: "Fiber orientation-dependent white matter contrast in gradient echo MRI",
    venue: Some("Proceedings of the National Academy of Sciences"), volume: Some("109"), issue: Some("45"),
    pages: Some("18559-18564"), doi: Some("10.1073/pnas.1211075109"), ..BLANK
};

pub(crate) const LI2025SUSEPNET: Citation = Citation {
    key: "li2025susepnet", authors: "Li, M., Chen, C., Li, Z., et al.", year: 2025,
    title: "SUSEP-Net: Simulation-Supervised and Contrastive Learning-based Deep Neural Networks for Susceptibility Source Separation",
    venue: Some("arXiv"), doi: Some("10.48550/arXiv.2506.13293"), ..BLANK
};

/// χ-sepnet, and the R2PRIMEnet and QSMnetME networks of its pipeline.
pub(crate) const KIM2025CHISEPNET: Citation = Citation {
    key: "kim2025chisepnet", authors: "Kim, M., et al.", year: 2025,
    title: "χ-sepnet: Deep neural network for magnetic susceptibility source separation",
    venue: Some("Human Brain Mapping"), volume: Some("46"), issue: Some("2"), pages: Some("e70136"),
    doi: Some("10.1002/hbm.70136"), ..BLANK
};

pub(crate) const PEI2015: Citation = Citation {
    key: "pei2015", authors: "Pei, M., et al.", year: 2015,
    title: "Algorithm for fast monoexponential fitting based on Auto-Regression on Linear Operations (ARLO) of data",
    venue: Some("Magnetic Resonance in Medicine"), volume: Some("73"), issue: Some("2"), pages: Some("843-850"),
    doi: Some("10.1002/mrm.25137"), ..BLANK
};

pub(crate) const WEIGEL2015: Citation = Citation {
    key: "weigel2015", authors: "Weigel, M.", year: 2015,
    title: "Extended phase graphs: dephasing, RF pulses, and echoes - pure and simple",
    venue: Some("Journal of Magnetic Resonance Imaging"), volume: Some("41"), issue: Some("2"), pages: Some("266-295"),
    doi: Some("10.1002/jmri.24619"), ..BLANK
};

// ---- SWI, segmentation and multiple orientations --------------------------------------------

pub(crate) const ECKSTEIN2021: Citation = Citation {
    key: "eckstein2021",
    authors: "Eckstein, K., Bachrata, B., Hangel, G., Widhalm, G., Enzinger, C., Barth, M., Trattnig, S., Robinson, S.D.",
    year: 2021,
    title: "Improved susceptibility weighted imaging at ultra-high field using bipolar multi-echo acquisition and optimized image processing: CLEAR-SWI",
    venue: Some("NeuroImage"), volume: Some("237"), pages: Some("118175"),
    doi: Some("10.1016/j.neuroimage.2021.118175"), ..BLANK
};

pub(crate) const GHO2014: Citation = Citation {
    key: "gho2014", authors: "Gho, S.-M., Liu, C., Li, W., et al.", year: 2014,
    title: "Susceptibility map-weighted imaging (SMWI) for neuroimaging",
    venue: Some("Magnetic Resonance in Medicine"), volume: Some("72"), issue: Some("2"), pages: Some("337-346"),
    doi: Some("10.1002/mrm.24920"), ..BLANK
};

pub(crate) const BILLOT2023: Citation = Citation {
    key: "billot2023", authors: "Billot, B., Greve, D.N., Puonti, O., et al.", year: 2023,
    title: "SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining",
    venue: Some("Medical Image Analysis"), volume: Some("86"), pages: Some("102789"),
    doi: Some("10.1016/j.media.2023.102789"), ..BLANK
};

pub(crate) const BILLOT2023PNAS: Citation = Citation {
    key: "billot2023pnas", authors: "Billot, B., Magdamo, C., Cheng, Y., et al.", year: 2023,
    title: "Robust machine learning segmentation for large-scale analysis of heterogeneous clinical brain MRI datasets",
    venue: Some("Proceedings of the National Academy of Sciences"), volume: Some("120"), issue: Some("9"),
    pages: Some("e2216399120"), doi: Some("10.1073/pnas.2216399120"), ..BLANK
};

pub(crate) const LIU2009COSMOS: Citation = Citation {
    key: "liu2009cosmos", authors: "Liu, T., Spincemaille, P., de Rochefort, L., Kressler, B., Wang, Y.", year: 2009,
    title: "Calculation of susceptibility through multiple orientation sampling (COSMOS): a method for conditioning the inverse problem from measured magnetic field map to susceptibility source image in MRI",
    venue: Some("Magnetic Resonance in Medicine"), volume: Some("61"), pages: Some("196-204"),
    doi: Some("10.1002/mrm.21828"), ..BLANK
};

pub(crate) const LIU2010STI: Citation = Citation {
    key: "liu2010sti", authors: "Liu, C.", year: 2010,
    title: "Susceptibility tensor imaging",
    venue: Some("Magnetic Resonance in Medicine"), volume: Some("63"), pages: Some("1471-1477"),
    doi: Some("10.1002/mrm.22482"), ..BLANK
};

# QSM.rs

A Rust library for Quantitative Susceptibility Mapping (QSM) of the brain.

QSM.rs provides a complete set of algorithms for reconstructing magnetic susceptibility maps from MRI phase data, including brain extraction, phase unwrapping, background field removal, and dipole inversion.

**[API Documentation](https://astewartau.github.io/QSM.rs/qsm_core/)**

## Usage

Add `qsm-core` to your `Cargo.toml`:

```toml
[dependencies]
qsm-core = { git = "https://github.com/astewartau/QSM.rs" }
```

Optional features: `parallel` (Rayon multi-threading) and `simd`:

```toml
[dependencies]
qsm-core = { git = "https://github.com/astewartau/QSM.rs", features = ["parallel"] }
```

There are two ways to use the crate. Both are demonstrated as runnable examples
([`examples/pipeline_highlevel.rs`](examples/pipeline_highlevel.rs),
[`examples/pipeline_lowlevel.rs`](examples/pipeline_lowlevel.rs)):

```
cargo run --release --example pipeline_highlevel
cargo run --release --example pipeline_lowlevel
```

### High-level pipeline (recommended)

Describe the scan once with a `ScanMetadata`, then run the stages. The `run_*`
functions dispatch to the configured algorithm and handle unit conversions
internally.

```rust,no_run
use qsm_core::pipeline::{
    ScanMetadata, FieldMappingConfig, BgRemovalConfig, InversionConfig, QsmReference,
    run_field_mapping, run_bg_removal, run_dipole_inversion, apply_reference,
};

# fn run(phase: Vec<f64>, magnitude: Vec<f64>, mask: Vec<u8>) -> Result<(), qsm_core::pipeline::PipelineError> {
let meta = ScanMetadata {
    dims: (128, 128, 64),
    voxel_size: (1.0, 1.0, 1.0),
    echo_times: vec![0.020],        // seconds
    field_strength: 3.0,            // Tesla
    b0_direction: (0.0, 0.0, 1.0),
};

let phases: Vec<&[f64]> = vec![&phase];
let mags: Vec<&[f64]> = vec![&magnitude];

let field = run_field_mapping(&phases, Some(&mags), &mask, &meta,
    &FieldMappingConfig::default(), &mut |_, _| {})?;
let bg = run_bg_removal(&field.b0_field_ppm, &mask, &meta,
    &BgRemovalConfig::default(), &mut |_, _| {})?;
let chi = run_dipole_inversion(&bg.local_field_ppm, &bg.eroded_mask, &meta,
    &InversionConfig::default(), Some(&magnitude), &mut |_, _| {})?;
let chi = apply_reference(&chi, &bg.eroded_mask, QsmReference::Mean);
# let _ = chi; Ok(())
# }
```

### Low-level building blocks

Call the individual algorithm functions directly when you want to wire stages
together yourself. Each takes a `Grid`, a `*Params` struct (all implement
`Default`), and — for iterative methods — a progress callback.

```rust,no_run
use qsm_core::{Grid, bet, unwrap, bgremove, inversion};
use qsm_core::bet::BetParams;
use qsm_core::bgremove::VsharpParams;
use qsm_core::inversion::TvParams;

# fn run(phase: &[f64], magnitude: &[f64]) {
let grid = Grid::new(128, 128, 64, 1.0, 1.0, 1.0);
let bdir = (0.0, 0.0, 1.0); // B0 direction

let mask = bet::run_bet(magnitude, &grid, &BetParams::default(), |_, _| {});
let unwrapped = unwrap::laplacian_unwrap(phase, &mask, &grid);
let (local, eroded) = bgremove::vsharp(&unwrapped, &mask, &grid, &VsharpParams::default(), |_, _| {});
let chi = inversion::tv_admm(&local, &eroded, &grid, bdir, &TvParams::default(), |_, _| {});
# let _ = chi;
# }
```

Load and save NIfTI volumes with [`qsm_core::io`](src/io.rs).

## Algorithms

### Brain Extraction

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **BET** | Brain Extraction Tool — region-growing brain masking with mesh evolution | Smith, S.M. (2002). "Fast robust automated brain extraction." *Human Brain Mapping*, 17(3):143-155. [DOI](https://doi.org/10.1002/hbm.10062) |

### Phase Unwrapping

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **ROMEO** | Region-growing with quality-guided ordering using magnitude and gradient coherence weighting | Dymerska, B., et al. (2021). "Phase unwrapping with a rapid opensource minimum spanning tree algorithm (ROMEO)." *Magnetic Resonance in Medicine*, 85(4):2294-2308. [DOI](https://doi.org/10.1002/mrm.28563) |
| **Laplacian** | FFT-based Poisson solver for fast, robust unwrapping | Schofield, M.A., Zhu, Y. (2003). "Fast phase unwrapping algorithm for interferometric applications." *Optics Letters*, 28(14):1194-1196. [DOI](https://doi.org/10.1364/OL.28.001194) |

### Background Field Removal

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **V-SHARP** | Variable-radius Sophisticated Harmonic Artifact Reduction for Phase data — multi-scale deconvolution for robust background removal | Wu, B., et al. (2012). "Whole brain susceptibility mapping using compressed sensing." *Magnetic Resonance in Medicine*, 67(1):137-147. [DOI](https://doi.org/10.1002/mrm.23000) |
| **SHARP** | Sophisticated Harmonic Artifact Reduction for Phase data — deconvolution-based harmonic field removal | Schweser, F., et al. (2011). "Quantitative imaging of intrinsic magnetic tissue properties using MRI signal phase." *NeuroImage*, 54(4):2789-2807. [DOI](https://doi.org/10.1016/j.neuroimage.2010.10.070) |
| **RESHARP** | Regularized SHARP — uses Tikhonov regularization instead of TSVD truncation for more robust SMV deconvolution | Sun, H. and Wilman, A.H. (2013). "Background field removal using spherical mean value filtering and Tikhonov regularization." *Magn Reson Med*, 71(3):1151-1157. [DOI](https://doi.org/10.1002/mrm.24765) |
| **SMV** | Simple Spherical Mean Value — subtracts the spherical mean of the field for basic background removal | Schweser, F., et al. (2011). "Quantitative imaging of intrinsic magnetic tissue properties using MRI signal phase." *NeuroImage*, 54(4):2789-2807. [DOI](https://doi.org/10.1016/j.neuroimage.2010.10.070) |
| **PDF** | Projection onto Dipole Fields — orthogonal projection approach | Liu, T., et al. (2011). "A novel background field removal method for MRI using projection onto dipole fields." *NMR in Biomedicine*, 24(9):1129-1136. [DOI](https://doi.org/10.1002/nbm.1670) |
| **iSMV** | Iterative Spherical Mean Value — iterative deconvolution-based method | Wen, Y., et al. (2014). "An iterative spherical mean value method for background field removal in MRI." *Magnetic Resonance in Medicine*, 72(4):1065-1071. [DOI](https://doi.org/10.1002/mrm.24998) |
| **LBV** | Laplacian Boundary Value — boundary value problem approach | Zhou, D., et al. (2014). "Background field removal by solving the Laplacian boundary value problem." *NMR in Biomedicine*, 27(3):312-319. [DOI](https://doi.org/10.1002/nbm.3064) |
| **SDF** | Spatially Dependent Filtering — used in the QSMART pipeline | Yaghmaie, N., Syeda, W., et al. (2021). "QSMART: Quantitative Susceptibility Mapping Artifact Reduction Technique." *NeuroImage*, 231:117701. [DOI](https://doi.org/10.1016/j.neuroimage.2020.117701) |

### Combined Phase Unwrapping + Background Removal

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **HARPERELLA** | Integrated Laplacian-based phase unwrapping and background phase removal — estimates exterior Laplacian via SMV uniformity | Li, W., et al. (2014). "Integrated Laplacian-based phase unwrapping and background phase removal for quantitative susceptibility mapping." *NMR in Biomedicine*, 27(2):219-227. [DOI](https://doi.org/10.1002/nbm.3056) |
| **iHARPERELLA** | Improved HARPERELLA — estimates exterior Laplacian by directly minimizing weighted phase for more robust low-frequency suppression | Li, W., Wu, B., Liu, C. (2015). "iHARPERELLA: an improved method for integrated 3D phase unwrapping and background phase removal." *Proc. ISMRM* 23, p.3313. |

### Dipole Inversion

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **TKD** | Truncated K-space Division — fast closed-form solution with k-space thresholding | Shmueli, K., et al. (2009). "Magnetic susceptibility mapping of brain tissue in vivo using MRI phase data." *Magnetic Resonance in Medicine*, 62(6):1510-1522. [DOI](https://doi.org/10.1002/mrm.22135) |
| **TSVD** | Truncated Singular Value Decomposition — zeros out small dipole kernel values instead of truncating | Shmueli, K., et al. (2009). "Magnetic susceptibility mapping of brain tissue in vivo using MRI phase data." *Magnetic Resonance in Medicine*, 62(6):1510-1522. [DOI](https://doi.org/10.1002/mrm.22135) |
| **Tikhonov** | L2-regularized inversion with configurable kernels (identity, gradient, Laplacian) | Bilgic, B., et al. (2014). "Fast image reconstruction with L2-regularization." *Journal of Magnetic Resonance Imaging*, 40(1):181-191. [DOI](https://doi.org/10.1002/jmri.24365) |
| **TV** | Total Variation via ADMM — edge-preserving L1 regularization | Bilgic, B., et al. (2014). "Fast quantitative susceptibility mapping with L1-regularization and automatic parameter selection." *Magnetic Resonance in Medicine*, 72(5):1444-1459. [DOI](https://doi.org/10.1002/mrm.25029) |
| **NLTV** | Nonlinear Total Variation — nonlinear data fidelity with iterative reweighting | Kames, C., Wiggermann, V., Rauscher, A. (2018). "Rapid two-step dipole inversion for susceptibility mapping with sparsity priors." *NeuroImage*, 167:276-283. [DOI](https://doi.org/10.1016/j.neuroimage.2017.11.018) |
| **RTS** | Rapid Two-Step — LSMR solve followed by TV refinement | Kames, C., Wiggermann, V., Rauscher, A. (2018). "Rapid two-step dipole inversion for susceptibility mapping with sparsity priors." *NeuroImage*, 167:276-283. [DOI](https://doi.org/10.1016/j.neuroimage.2017.11.018) |
| **MEDI** | Morphology Enabled Dipole Inversion — L1 regularization with gradient and SNR weighting | Liu, T., et al. (2011). "Morphology enabled dipole inversion (MEDI) from a single-angle acquisition." *Magnetic Resonance in Medicine*, 66(3):777-783. [DOI](https://doi.org/10.1002/mrm.22816) |
| **iLSQR** | Iterative LSQR with streaking artifact removal | Li, W., et al. (2015). "A method for estimating and removing streaking artifacts in quantitative susceptibility mapping." *NeuroImage*, 108:111-122. [DOI](https://doi.org/10.1016/j.neuroimage.2014.12.043) |

### End-to-End QSM

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **TGV** | Total Generalized Variation — single-step QSM from wrapped phase, combining unwrapping, background removal, and dipole inversion | Langkammer, C., et al. (2015). "Fast quantitative susceptibility mapping using 3D EPI and total generalized variation." *NeuroImage*, 111:622-630. [DOI](https://doi.org/10.1016/j.neuroimage.2015.02.041) |
| **QSMART** | Two-stage QSM artifact reduction using SDF background removal, TKD inversion, and Frangi vesselness-based tissue/vasculature separation | Yaghmaie, N., Syeda, W., et al. (2021). "QSMART: Quantitative Susceptibility Mapping Artifact Reduction Technique." *NeuroImage*, 231:117701. [DOI](https://doi.org/10.1016/j.neuroimage.2020.117701) |

### SWI Processing

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **CLEAR-SWI** | Susceptibility Weighted Imaging — phase mask weighting with high-pass filtering and minimum intensity projection | Eckstein, K., et al. (2024). "CLEAR-SWI: Computational Efficient T2* Weighted Imaging." *Proc. ISMRM*. |

### Multi-Echo Processing

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **MCPC-3D-S** | Multi-Channel Phase Combination (ASPIRE) — removes phase offsets across echoes | Eckstein, K., et al. (2018). "Computationally Efficient Combination of Multi-channel Phase Data From Multi-echo Acquisitions (ASPIRE)." *Magnetic Resonance in Medicine*, 79:2996-3006. [DOI](https://doi.org/10.1002/mrm.26963) |
| **R2\*/T2\* (ARLO)** | R2* mapping from multi-echo magnitude using Auto-Regression on Linear Operations; T2* = 1/R2* | Pei, M., et al. (2015). "Algorithm for fast monoexponential fitting based on Auto-Regression on Linear Operations (ARLO) of data." *Magnetic Resonance in Medicine*, 73(2):843-850. [DOI](https://doi.org/10.1002/mrm.25137) |

### Preprocessing

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **Bias Correction** | Homogeneity correction for high-field MRI | Eckstein, K., Trattnig, S., Robinson, S.D. (2019). "A Simple Homogeneity Correction for Neuroimaging at 7T." *Proc. ISMRM 27th Annual Meeting*. |

### Susceptibility Source Separation

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **χ-separation** | Gauss-Newton optimization separating total susceptibility into paramagnetic (iron) and diamagnetic (myelin) components using coupled field and R2' constraints | Shin, H., et al. (2021). "χ-separation: Magnetic susceptibility source separation toward iron and myelin mapping in the brain." *NeuroImage*, 240:118371. [DOI](https://doi.org/10.1016/j.neuroimage.2021.118371) |

### Utilities

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **Otsu Thresholding** | Automatic threshold selection for bimodal histograms | Otsu, N. (1979). "A Threshold Selection Method from Gray-Level Histograms." *IEEE Transactions on Systems, Man, and Cybernetics*, 9(1):62-66. [DOI](https://doi.org/10.1109/TSMC.1979.4310076) |
| **Frangi Filter** | 3D multi-scale vesselness enhancement filter | Frangi, A.F., et al. (1998). "Multiscale vessel enhancement filtering." *MICCAI'98*, LNCS vol 1496, 130-137. [DOI](https://doi.org/10.1007/BFb0056195) |
| **Surface Curvature** | Discrete differential geometry operators for triangulated meshes | Meyer, M., et al. (2003). "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds." *Visualization and Mathematics III*, 35-57. [DOI](https://doi.org/10.1007/978-3-662-05105-4_2) |

## Reference Implementations

This library was developed with reference to the following open-source implementations:

| Repository | Algorithms | Language |
|------------|------------|----------|
| [QSM.jl](https://github.com/kamesy/QSM.jl) | SHARP, V-SHARP, SMV, PDF, iSMV, LBV, Laplacian unwrap, TKD, TSVD, Tikhonov, TV, RTS, NLTV | Julia |
| [QSM.m](https://github.com/kamesy/QSM.m) | iLSQR | MATLAB |
| [QuantitativeSusceptibilityMappingTGV.jl](https://github.com/korbinian90/QuantitativeSusceptibilityMappingTGV.jl) | TGV | Julia |
| [MriResearchTools.jl](https://github.com/korbinian90/MriResearchTools.jl) | ROMEO, MCPC-3D-S, R2*/T2*, bias correction | Julia |
| [MEDI_toolbox](https://github.com/huawu02/MEDI_toolbox) | MEDI | MATLAB |
| [FSL-BET2](https://github.com/Bostrix/FSL-BET2) | BET | C++ |
| [QSMART](https://github.com/wtsyeda/QSMART) | SDF, QSMART pipeline, Frangi filter, curvature | MATLAB |
| [CLEARSWI.jl](https://github.com/korbinian90/CLEARSWI.jl) | CLEAR-SWI | Julia |
| [chi-separation](https://github.com/SNU-LIST/chi-separation) | Chi-separation | MATLAB |

## License

This project is licensed under the [MIT License](LICENSE).

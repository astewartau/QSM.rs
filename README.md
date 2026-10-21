# QSM.rs

A Rust library for Quantitative Susceptibility Mapping (QSM) of the brain.

QSM.rs provides a complete set of algorithms for reconstructing magnetic susceptibility maps from MRI phase data, including brain extraction, phase unwrapping, background field removal, and dipole inversion.

**[Website](https://astewartau.github.io/QSM.rs/)** · **[API Documentation](https://astewartau.github.io/QSM.rs/api/qsm_core/)**

QSM.rs is the shared reconstruction engine behind the wider QSM ecosystem: the
[QSMxT](https://qsmxt.github.io/QSMxT/) command-line pipeline and the
[QSMbly](https://qsmbly.neurodesk.org/) in-browser app both call directly into this
library, and [QSM-CI](https://qsmxt.github.io/QSM-CI/) benchmarks these methods against
phantom ground truth. See the [ecosystem hub](https://qsmxt.github.io/) for an overview.

## Installation

Add `qsm-core` to your `Cargo.toml`:

```toml
[dependencies]
qsm-core = { git = "https://github.com/astewartau/QSM.rs" }
```

### Feature flags

| Feature | Enables |
|---------|---------|
| `parallel` | Rayon multi-threading (and multithreaded ONNX kernels with `onnx`) |
| `simd` | SIMD-accelerated inner loops via `wide` |
| `onnx` | Deep-learning methods, run through the pure-Rust `tract` engine (native and WASM) |
| `download` | Native download and SHA-256-checked caching of model weights (see `qsm_core::models`) |
| `introspection` | `Serialize` on the `*Params` structs, for tooling that inspects parameter names |

```toml
[dependencies]
qsm-core = { git = "https://github.com/astewartau/QSM.rs", features = ["parallel", "onnx"] }
```

Model weights are not bundled: deep-learning functions take the ONNX bytes from the caller.

## Usage

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
let unwrapped = unwrap::laplacian_unwrap(phase, &mask, &grid, unwrap::LaplacianSolver::Dct);
let (local, eroded) = bgremove::vsharp(&unwrapped, &mask, &grid, &VsharpParams::default(), |_, _| {});
let chi = inversion::tv_admm(&local, &eroded, &grid, bdir, &TvParams::default(), |_, _| {});
# let _ = chi;
# }
```

Load and save NIfTI volumes with [`qsm_core::io`](src/io.rs).

## Algorithms

Methods marked † need the `onnx` feature and model weights.

### Brain Extraction and Segmentation

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **BET** | Brain extraction by surface-mesh evolution (`run_bet`) | Smith, S.M. (2002). "Fast robust automated brain extraction." *Human Brain Mapping*, 17(3):143-155. [DOI](https://doi.org/10.1002/hbm.10062) |
| **HD-BET** † | Deep-learning brain extraction from a magnitude image (`hd_bet`) | Isensee, F., et al. (2019). "Automated brain extraction of multisequence MRI using artificial neural networks." *Human Brain Mapping*, 40(17):4952-4964. [DOI](https://doi.org/10.1002/hbm.24750) |
| **RS2-Net** † | Deep-learning rodent brain extraction (`rs2_net`) | Lin, Y., et al. (2024). "RS2-Net: An end-to-end deep learning framework for rodent skull stripping in multi-center brain MRI." *NeuroImage*, 298:120769. [DOI](https://doi.org/10.1016/j.neuroimage.2024.120769) |
| **MindGrab** † | Deep-learning brain extraction for any contrast (`mindgrab`) | Fani, A., et al. (2025). "MindGrab for BrainChop: Fast and Accurate Skull Stripping for Command Line and Browser." *arXiv*:2506.11860. [DOI](https://doi.org/10.48550/arXiv.2506.11860) |
| **SynthSeg** † | Contrast-agnostic whole-brain segmentation into FreeSurfer labels (`segment::synthseg`) | Billot, B., et al. (2023). "SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining." *Medical Image Analysis*, 86:102789. [DOI](https://doi.org/10.1016/j.media.2023.102789) |
| **Threshold masking** | Otsu, fixed or percentile threshold of the magnitude or of the ROMEO phase-quality map (`MaskOp::Threshold`, `otsu_threshold`) | Otsu, N. (1979). "A Threshold Selection Method from Gray-Level Histograms." *IEEE Transactions on Systems, Man, and Cybernetics*, 9(1):62-66. [DOI](https://doi.org/10.1109/TSMC.1979.4310076) |

### Mask Operations

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **Erosion / dilation** | Iterated 6-connected erosion and dilation (`erode_mask`, `dilate_mask`) | — |
| **Closing** | Dilation then erosion with a spherical structuring element (`morphological_close`) | — |
| **Hole filling** | Fills enclosed background components up to a size limit (`fill_holes`) | — |
| **Gaussian smoothing** | Smooths the mask and re-thresholds it at 0.5 (`MaskOp::GaussianSmooth`) | — |
| **Signal-gated erosion** | Mask refinement that erodes only low-signal boundary voxels (`signal_gated_erosion`) | QSM-CI harmonization masking, [QSMxT/QSM-CI](https://github.com/QSMxT/QSM-CI) |
| **Connected components** | 6-connected component labelling, as `scipy.ndimage.label` (internal: signal-gated erosion, PRELUDE, SynthSeg) | — |

### Coil Combination and Field Mapping

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **Real/imaginary input** | Converts real/imaginary echoes to magnitude/phase, detecting and undoing alternate-slice π chopping with one decision for all echoes, and optionally negating the phase for GE's opposite sign convention (`complex_to_mag_phase`, `ComplexToMagPhaseParams`) | — |
| **MCPC-3D-S** | Coil combination and phase-offset removal (`mcpc3ds_combine`, `phase_offset_removal`) | Eckstein, K., et al. (2018). "Computationally Efficient Combination of Multi-channel Phase Data From Multi-echo Acquisitions (ASPIRE)." *Magnetic Resonance in Medicine*, 79:2996-3006. [DOI](https://doi.org/10.1002/mrm.26963) |
| **Bipolar correction** | Removes the linear phase artefact of bipolar readout gradients; needs three or more echoes (`bipolar_correction`) | Eckstein, K. (2021). PhD thesis, TU Wien, Section 3.1.3. [DOI](https://doi.org/10.34726/hss.2021.43447) |
| **B0 weighted average** | Weighted average of unwrapped phase / TE over echoes; weights by phase SNR, phase variance, magnitude, TE, uniform, or an assumed T2* decay (`calculate_b0_weighted`, `B0WeightType`) | — |
| **Linear fit** | Magnitude-weighted least-squares fit of phase against TE, with optional phase offset and reliability mask (`multi_echo_linear_fit`) | — |
| **QSMART echo fit** | QSMART's magnitude-weighted fit of phase against TE through the origin, on echoes unwrapped over the whole volume, with its reliability map `R_0` from the box-filtered fit residual (`qsmart_echofit`, `B0EstimationMethod::QsmartEchofit`) | Yaghmaie, N., Syeda, W., et al. (2021). *NeuroImage*, 231:117701. [DOI](https://doi.org/10.1016/j.neuroimage.2020.117701) |
| **Robust echo combination** | Robust-loss multi-echo field fit with per-echo outlier scores (`multi_echo_robust_fit`, `B0EstimationMethod::RobustFit`) | Biondetti, E., et al. (2022). *Magnetic Resonance in Medicine*. [PMC9545116](https://pmc.ncbi.nlm.nih.gov/articles/PMC9545116/) |

### Phase Unwrapping

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **ROMEO** | Quality-guided region growing (`unwrap_romeo`) | Dymerska, B., et al. (2021). "Phase unwrapping with a rapid opensource minimum spanning tree algorithm (ROMEO)." *Magnetic Resonance in Medicine*, 85(4):2294-2308. [DOI](https://doi.org/10.1002/mrm.28563) |
| **Laplacian** | Poisson-equation unwrapping of the total field, DCT, zero-padded FFT or mirrored FFT (`lapunwrap.m`, as in QSMART) solver (`laplacian_unwrap`) | Ghiglia, D.C., Romero, L.A. (1994). *JOSA A*, 11(1):107-117. [DOI](https://doi.org/10.1364/JOSAA.11.000107); Schofield, M.A., Zhu, Y. (2003). "Fast phase unwrapping algorithm for interferometric applications." *Optics Letters*, 28(14):1194-1196. [DOI](https://doi.org/10.1364/OL.28.001194) |
| **Best path** | 3D-SRNCP reliability-ordered edge merging (`unwrap_bestpath`) | Abdul-Rahman, H., et al. (2007). "Fast and robust three-dimensional best path phase unwrapping algorithm." *Applied Optics*, 46(26):6623-6635. [DOI](https://doi.org/10.1364/AO.46.006623) |
| **PRELUDE** | Region merging over phase-range bins (`unwrap_prelude`) | Jenkinson, M. (2003). "Fast, automated, N-dimensional phase-unwrapping algorithm." *Magnetic Resonance in Medicine*, 49(1):193-197. [DOI](https://doi.org/10.1002/mrm.10354) |
| **Slice-wise (2D)** | Any of the above per slice, for 2D multi-slice or thick-slice data (`unwrap_slicewise`, `unwrap_slicewise_multi_echo`) | — |

### Background Field Removal

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **V-SHARP** | Variable-radius SHARP | Wu, B., et al. (2012). "Whole brain susceptibility mapping using compressed sensing." *Magnetic Resonance in Medicine*, 67(1):137-147. [DOI](https://doi.org/10.1002/mrm.23000) |
| **SHARP** | Spherical-mean-value deconvolution with TSVD | Schweser, F., et al. (2011). "Quantitative imaging of intrinsic magnetic tissue properties using MRI signal phase." *NeuroImage*, 54(4):2789-2807. [DOI](https://doi.org/10.1016/j.neuroimage.2010.10.070) |
| **RESHARP** | SHARP with Tikhonov regularization | Sun, H., Wilman, A.H. (2014). "Background field removal using spherical mean value filtering and Tikhonov regularization." *Magnetic Resonance in Medicine*, 71(3):1151-1157. [DOI](https://doi.org/10.1002/mrm.24765) |
| **SMV** | Spherical mean value filtering | Schweser, F., et al. (2011). *NeuroImage*, 54(4):2789-2807. [DOI](https://doi.org/10.1016/j.neuroimage.2010.10.070) |
| **iSMV** | Iterative spherical mean value | Wen, Y., et al. (2014). "An iterative spherical mean value method for background field removal in MRI." *Magnetic Resonance in Medicine*, 72(4):1065-1071. [DOI](https://doi.org/10.1002/mrm.24998) |
| **mSMV** | Maximum spherical mean value boundary correction, standalone or after another method | Roberts, A.G., et al. (2024). "Maximum spherical mean value (mSMV) filtering for whole-brain quantitative susceptibility mapping." *Magnetic Resonance in Medicine*, 91(4):1586-1597. [DOI](https://doi.org/10.1002/mrm.29963) |
| **PDF** | Projection onto dipole fields | Liu, T., et al. (2011). "A novel background field removal method for MRI using projection onto dipole fields." *NMR in Biomedicine*, 24(9):1129-1136. [DOI](https://doi.org/10.1002/nbm.1670) |
| **LBV** | Laplacian boundary value | Zhou, D., et al. (2014). "Background field removal by solving the Laplacian boundary value problem." *NMR in Biomedicine*, 27(3):312-319. [DOI](https://doi.org/10.1002/nbm.3064) |
| **SDF** | Spatially dependent filtering (QSMART) | Yaghmaie, N., Syeda, W., et al. (2021). "QSMART: Quantitative Susceptibility Mapping Artifact Reduction Technique." *NeuroImage*, 231:117701. [DOI](https://doi.org/10.1016/j.neuroimage.2020.117701) |
| **BFRnet** † | Deep-learning background field prediction | Zhu, X., et al. (2023). "BFRnet: A deep learning-based MR background field removal method for QSM of the brain containing significant pathological susceptibility sources." *Zeitschrift für Medizinische Physik*. [DOI](https://doi.org/10.1016/j.zemedi.2022.08.001) |

### Combined Phase Unwrapping and Background Removal

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **HARPERELLA** | Integrated Laplacian unwrapping and background removal, wrapped phase or total field (ppm) → local field (`harperella`, `HarperellaInput`) | Li, W., et al. (2014). "Integrated Laplacian-based phase unwrapping and background phase removal for quantitative susceptibility mapping." *NMR in Biomedicine*, 27(2):219-227. [DOI](https://doi.org/10.1002/nbm.3056) |
| **iHARPERELLA** | Improved HARPERELLA (`iharperella`) | Li, W., Wu, B., Liu, C. (2015). "iHARPERELLA: an improved method for integrated 3D phase unwrapping and background phase removal." *Proc. ISMRM* 23, p.3313. |
| **iQFM** † | Deep-learning wrapped phase → local field (`run_iqfm`) | Gao, Y., et al. (2022). "Instant tissue field and magnetic susceptibility mapping from MRI raw phase using Laplacian enhanced deep neural networks." *NeuroImage*, 259:119410. [DOI](https://doi.org/10.1016/j.neuroimage.2022.119410) |

### Dipole Inversion

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **TKD** | Truncated k-space division | Shmueli, K., et al. (2009). "Magnetic susceptibility mapping of brain tissue in vivo using MRI phase data." *Magnetic Resonance in Medicine*, 62(6):1510-1522. [DOI](https://doi.org/10.1002/mrm.22135) |
| **TSVD** | Truncated SVD of the dipole kernel | Shmueli, K., et al. (2009). *Magnetic Resonance in Medicine*, 62(6):1510-1522. [DOI](https://doi.org/10.1002/mrm.22135) |
| **Tikhonov** | Closed-form L2 regularization (identity, gradient or Laplacian) | Bilgic, B., et al. (2014). "Fast image reconstruction with L2-regularization." *Journal of Magnetic Resonance Imaging*, 40(1):181-191. [DOI](https://doi.org/10.1002/jmri.24365) |
| **TV** | Total variation via ADMM (`tv_admm`) | Bilgic, B., et al. (2014). "Fast quantitative susceptibility mapping with L1-regularization and automatic parameter selection." *Magnetic Resonance in Medicine*, 72(5):1444-1459. [DOI](https://doi.org/10.1002/mrm.25029) |
| **NLTV** | Nonlinear-fidelity TV with iterative reweighting | Kames, C., Wiggermann, V., Rauscher, A. (2018). "Rapid two-step dipole inversion for susceptibility mapping with sparsity priors." *NeuroImage*, 167:276-283. [DOI](https://doi.org/10.1016/j.neuroimage.2017.11.018) |
| **RTS** | Rapid two-step: LSMR solve then TV refinement | Kames, C., Wiggermann, V., Rauscher, A. (2018). *NeuroImage*, 167:276-283. [DOI](https://doi.org/10.1016/j.neuroimage.2017.11.018) |
| **MEDI** | Morphology-enabled dipole inversion | Liu, T., et al. (2011). "Morphology enabled dipole inversion (MEDI) from a single-angle acquisition." *Magnetic Resonance in Medicine*, 66(3):777-783. [DOI](https://doi.org/10.1002/mrm.22816) |
| **iLSQR** | LSQR with streaking-artifact estimation and removal | Li, W., et al. (2015). "A method for estimating and removing streaking artifacts in quantitative susceptibility mapping." *NeuroImage*, 108:111-122. [DOI](https://doi.org/10.1016/j.neuroimage.2014.12.043) |
| **STAR-QSM** | Streaking artifact reduction: clamped k-space division refined by L1-regularised SpaRSA (STI Suite 3.0 `QSM_star`) | Wei, H., et al. (2015). "Streaking artifact reduction for quantitative susceptibility mapping of sources with large dynamic range." *NMR in Biomedicine*, 28(10):1294-1303. [DOI](https://doi.org/10.1002/nbm.3383) |
| **NDI** | Nonlinear dipole inversion by gradient descent | Polak, D., et al. (2020). "Nonlinear dipole inversion (NDI) enables robust quantitative susceptibility mapping (QSM)." *NMR in Biomedicine*, 33(12):e4271. [DOI](https://doi.org/10.1002/nbm.4271) |
| **FANSI (nlTV / nlTGV)** | Nonlinear TV or TGV via ADMM | Milovic, C., et al. (2018). "Fast nonlinear susceptibility inversion with variational regularization." *Magnetic Resonance in Medicine*, 80(2):814-821. [DOI](https://doi.org/10.1002/mrm.27073) |
| **L1-QSM** | L1-norm nonlinear data fidelity with TV | Milovic, C., et al. (2022). "Streaking artifact suppression of quantitative susceptibility mapping reconstructions via L1-norm data fidelity optimization (L1-QSM)." *Magnetic Resonance in Medicine*, 87(1):457-473. [DOI](https://doi.org/10.1002/mrm.28957) |
| **WH-QSM** | Joint susceptibility and residual harmonic background | Milovic, C., et al. (2019). "Weak-harmonic regularization for quantitative susceptibility mapping." *Magnetic Resonance in Medicine*, 81(2):1399-1411. [DOI](https://doi.org/10.1002/mrm.27483) |
| **HD-QSM** | Two-stage L1 then L2 hybrid data fidelity | Lambert, M., et al. (2022). "Hybrid data fidelity term approach for quantitative susceptibility mapping." *Magnetic Resonance in Medicine*, 88(4):1567-1583. [DOI](https://doi.org/10.1002/mrm.29218) |
| **AMP-PE** | Approximate message passing with automatic parameter estimation | Huang, S., et al. (2023). "Approximate Message Passing with Parameter Estimation: a probabilistic Bayesian dipole inversion." *Magnetic Resonance in Medicine*, 90(4):1414-1430. [DOI](https://doi.org/10.1002/mrm.29722) |
| **LSQR** | Minimally regularized LSQR with residual-field and offset unknowns | Schweser, F., et al. (2010). "Differentiation between diamagnetic and paramagnetic cerebral lesions based on magnetic susceptibility mapping." *Medical Physics*, 37(10):5165-5178. [DOI](https://doi.org/10.1118/1.3481505) |
| **HEIDI** | LSQR seed plus weighted-TV recovery of the dipole cone | Schweser, F., et al. (2012). "Quantitative susceptibility mapping for investigating subtle susceptibility variations in the human brain." *NeuroImage*, 62(3):2083-2100. [DOI](https://doi.org/10.1016/j.neuroimage.2012.05.067) |
| **xQSM** † | Octave-convolution U-Net | Gao, Y., et al. (2021). "xQSM: quantitative susceptibility mapping with octave convolutional and noise-regularized neural networks." *NMR in Biomedicine*, 34(3):e4461. [DOI](https://doi.org/10.1002/nbm.4461) |
| **QSMnet / QSMnet+** † | 3D U-Net | Yoon, J., et al. (2018). *NeuroImage*, 179:199-206. [DOI](https://doi.org/10.1016/j.neuroimage.2018.06.030); Jung, W., et al. (2020). *NeuroImage*, 211:116579. [DOI](https://doi.org/10.1016/j.neuroimage.2020.116579) |
| **QSMGAN** † | U-Net generator refined by a Wasserstein GAN | Chen, Y., et al. (2020). "QSMGAN: Improved quantitative susceptibility mapping using 3D generative adversarial networks with increased receptive field." *NeuroImage*, 207:116389. [DOI](https://doi.org/10.1016/j.neuroimage.2019.116389) |
| **LPCNN** † | Unrolled proximal gradient descent with a learned proximal operator | Lai, K.-W., Aggarwal, M., van Zijl, P., Li, X., Sulam, J. (2020). "Learned proximal networks for quantitative susceptibility mapping." *MICCAI 2020*, LNCS 12262. [DOI](https://doi.org/10.1007/978-3-030-59713-9_13) |
| **MoDL-QSM** † | Unrolled model-based gradient descent with a learned prior | Feng, R., et al. (2021). "MoDL-QSM: Model-based deep learning for quantitative susceptibility mapping." *NeuroImage*, 240:118376. [DOI](https://doi.org/10.1016/j.neuroimage.2021.118376) |
| **IR2QSM** † | Iterative reverse-concatenation recurrent U-Net | Li, M., et al. (2025). *Medical Physics*. [DOI](https://doi.org/10.1002/mp.17747); [arXiv:2406.12300](https://arxiv.org/abs/2406.12300) |

### Single-Step QSM

| Algorithm | Input | Description | Reference |
|-----------|-------|-------------|-----------|
| **TGV** | Total field (ppm), or one wrapped echo with its TE and B0 | Single-step total generalized variation (`tgv_qsm`, `TgvInput`) | Langkammer, C., et al. (2015). "Fast quantitative susceptibility mapping using 3D EPI and total generalized variation." *NeuroImage*, 111:622-630. [DOI](https://doi.org/10.1016/j.neuroimage.2015.02.041) |
| **TFI** | Total field (ppm) | Preconditioned total field inversion | Liu, Z., et al. (2017). "Preconditioned total field inversion (TFI) algorithm for quantitative susceptibility mapping." *Magnetic Resonance in Medicine*, 78(1):303-315. [DOI](https://doi.org/10.1002/mrm.26946) |
| **QSMART** | Total field (ppm), optional reliability map `R_0` | SDF background removal, inversion, and vasculature separation | Yaghmaie, N., Syeda, W., et al. (2021). *NeuroImage*, 231:117701. [DOI](https://doi.org/10.1016/j.neuroimage.2020.117701) |
| **AutoQSM** † | Total field (ppm) | Patch V-Net from total field to χ | Wei, H., et al. (2019). "Learning-based single-step quantitative susceptibility mapping reconstruction without brain extraction." *NeuroImage*, 202:116064. [DOI](https://doi.org/10.1016/j.neuroimage.2019.116064) |
| **NeXtQSM** † | Total field (ppm) | Background-removal U-Net followed by a variational inversion network | Cognolato, F., et al. (2023). "NeXtQSM—A complete deep learning pipeline for data-consistent Quantitative Susceptibility Mapping trained with hybrid data." *Medical Image Analysis*, 84:102700. [DOI](https://doi.org/10.1016/j.media.2022.102700) |
| **iQSM** † | Wrapped phase | Wrapped phase → χ (`run_iqsm`) | Gao, Y., et al. (2022). *NeuroImage*, 259:119410. [DOI](https://doi.org/10.1016/j.neuroimage.2022.119410) |
| **iQSM+** † | Wrapped phase | Orientation-adaptive iQSM (`run_iqsm_plus`) | Gao, Y., et al. (2024). *Medical Image Analysis*. [DOI](https://doi.org/10.1016/j.media.2024.103160) |

### Multi-Orientation

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **COSMOS** | Closed-form multi-orientation inversion (`cosmos`, `cosmos_weighted`) | Liu, T., et al. (2009). "Calculation of susceptibility through multiple orientation sampling (COSMOS)." *Magnetic Resonance in Medicine*, 61(1):196-204. [DOI](https://doi.org/10.1002/mrm.21828) |
| **STI** | Susceptibility tensor imaging from six or more orientations (`sti`) | Liu, C. (2010). "Susceptibility tensor imaging." *Magnetic Resonance in Medicine*, 63(6):1471-1477. [DOI](https://doi.org/10.1002/mrm.22482) |

### Relaxometry

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **R2\*/T2\* (ARLO)** | R2* mapping from multi-echo magnitude (`r2star_arlo`) | Pei, M., et al. (2015). "Algorithm for fast monoexponential fitting based on Auto-Regression on Linear Operations (ARLO) of data." *Magnetic Resonance in Medicine*, 73(2):843-850. [DOI](https://doi.org/10.1002/mrm.25137) |
| **R2 (EPG)** | R2 mapping from multi-echo spin echo by extended-phase-graph matching (`r2_epg`) | Weigel, M. (2015). "Extended phase graphs: dephasing, RF pulses, and echoes — pure and simple." *Journal of Magnetic Resonance Imaging*, 41(2):266-295. [DOI](https://doi.org/10.1002/jmri.24619) |
| **R2′** | R2* − R2 from paired gradient- and spin-echo data (`r2prime`) | Yablonskiy, D.A., Haacke, E.M. (1994). "Theory of NMR signal behavior in magnetically inhomogeneous tissues: the static dephasing regime." *Magnetic Resonance in Medicine*, 32(6):749-763. [DOI](https://doi.org/10.1002/mrm.1910320610) |
| **R2PRIMEnet** † | Deep-learning R2* → R2′ for GRE-only data (3 T) | Kim, M., et al. (2025). "χ-sepnet: Deep neural network for magnetic susceptibility source separation." *Human Brain Mapping*, 46(4):e70136. [DOI](https://doi.org/10.1002/hbm.70136) |

### Susceptibility Source Separation

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **χ-separation** | Paramagnetic/diamagnetic separation from local field, R2′ and magnitude; iLSQR-style (QSM-initialized) and MEDI-style variants | Shin, H., et al. (2021). "χ-separation: Magnetic susceptibility source separation toward iron and myelin mapping in the brain." *NeuroImage*, 240:118371. [DOI](https://doi.org/10.1016/j.neuroimage.2021.118371) |
| **HC-ChiSep** | Hollow-cylinder separation with fibre angle estimated from multi-echo magnitude | Wharton, S., Bowtell, R. (2012). "Fiber orientation-dependent white matter contrast in gradient echo MRI." *PNAS*, 109(45):18559-18564. [DOI](https://doi.org/10.1073/pnas.1211075109) |
| **R2\*-QSM** | Closed-form separation from QSM and R2* | Dimov, A.V., et al. (2022). "Magnetic susceptibility source separation solely from gradient echo data: histological validation." *Tomography*, 8(3):1544-1551. [DOI](https://doi.org/10.3390/tomography8030128) |
| **WaveSep** | Wavelet-domain L1 separation from QSM and R2′ | Fang, Z., et al. (2023). "Wavelet-based single-step susceptibility source separation." *Proc. ISMRM*. |
| **DECOMPOSE** | Per-voxel three-compartment fit of the complex multi-echo signal | Chen, J., et al. (2021). "Decompose quantitative susceptibility mapping (QSM) to sub-voxel diamagnetic and paramagnetic components based on gradient-echo MRI data." *NeuroImage*, 242:118477. [DOI](https://doi.org/10.1016/j.neuroimage.2021.118477) |
| **χ-sepnet** † | 3D U-Net separation | Kim, M., et al. (2025). *Human Brain Mapping*, 46(4):e70136. [DOI](https://doi.org/10.1002/hbm.70136) |
| **SUSEP-Net** † | 3D U-Net separation | Li, Z., Gao, Y., Sun, H., et al. (2025). [arXiv:2506.13293](https://arxiv.org/abs/2506.13293) |

### SWI Processing

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **CLEAR-SWI** | Multi-echo SWI: combined, bias-corrected magnitude × phase mask from the echo-combined, high-pass filtered phase; minimum intensity projection | Eckstein, K., et al. (2021). "Improved susceptibility weighted imaging at ultra-high field using bipolar multi-echo acquisition and optimized image processing: CLEAR-SWI." *NeuroImage*, 237:118175. [doi:10.1016/j.neuroimage.2021.118175](https://doi.org/10.1016/j.neuroimage.2021.118175) |
| **SMWI** | Magnitude weighted by a susceptibility-map mask | Gho, S.-M., et al. (2014). "Susceptibility map-weighted imaging (SMWI) for neuroimaging." *Magnetic Resonance in Medicine*, 72:337-346. [DOI](https://doi.org/10.1002/mrm.24920) |

### Preprocessing and Artefact Reduction

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **Bias correction** | Receive-field homogeneity correction (`makehomogeneous`) | Eckstein, K., Trattnig, S., Robinson, S.D. (2019). "A Simple Homogeneity Correction for Neuroimaging at 7T." *Proc. ISMRM 27th Annual Meeting*. |
| **N4 bias correction** | N4ITK bias field correction as ANTs' `N4BiasFieldCorrection` (`n4_bias_correction`) | Tustison, N.J., et al. (2010). "N4ITK: Improved N3 Bias Correction." *IEEE Transactions on Medical Imaging*, 29(6):1310-1320. [DOI](https://doi.org/10.1109/TMI.2010.2046908) |
| **MP-PCA denoising** | Marchenko–Pastur PCA denoising across echoes or volumes (`mppca_denoise`) | Veraart, J., et al. (2016). "Denoising of diffusion MRI using random matrix theory." *NeuroImage*, 142:394-406. [DOI](https://doi.org/10.1016/j.neuroimage.2016.08.016) |
| **Gibbs unringing** | Local subvoxel-shift Gibbs-ringing removal (`gibbs_unring`) | Kellner, E., et al. (2016). "Gibbs-ringing artifact removal based on local subvoxel-shifts." *Magnetic Resonance in Medicine*, 76(5):1574-1581. [DOI](https://doi.org/10.1002/mrm.26054) |
| **Anisotropic diffusion** | Edge-preserving gradient anisotropic diffusion smoothing (`gradient_anisotropic_diffusion`) | Perona, P., Malik, J. (1990). "Scale-space and edge detection using anisotropic diffusion." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 12(7):629-639. [DOI](https://doi.org/10.1109/34.56205) |
| **Motion correction** | Rigid co-registration of a series in the complex domain, with a B0 direction per volume (`motion`) | — |
| **Two-pass** | Reconstructs on a hole-preserving and a filled mask and combines them, keeping streaking from strong sources out of the brain | Stewart, A.W., et al. (2022). "QSMxT: Robust masking and artifact reduction for quantitative susceptibility mapping." *Magnetic Resonance in Medicine*, 87(3):1289-1300. [DOI](https://doi.org/10.1002/mrm.29048) |

### Utilities

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **Frangi filter** | Multi-scale 3D vesselness (`frangi_filter_3d`) | Frangi, A.F., et al. (1998). "Multiscale vessel enhancement filtering." *MICCAI'98*, LNCS vol 1496, 130-137. [DOI](https://doi.org/10.1007/BFb0056195) |
| **Vasculature mask** | Bottom-hat filtering then Frangi vesselness, for QSMART (`generate_vasculature_mask`), on QSMART's mean-echo, N4-corrected magnitude (`qsmart_vessel_magnitude`) | Yaghmaie, N., Syeda, W., et al. (2021). *NeuroImage*, 231:117701. [DOI](https://doi.org/10.1016/j.neuroimage.2020.117701) |
| **Surface curvature** | Discrete differential-geometry operators on triangle meshes (`calculate_gaussian_curvature`) | Meyer, M., et al. (2003). "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds." *Visualization and Mathematics III*, 35-57. [DOI](https://doi.org/10.1007/978-3-662-05105-4_2) |
| **Rigid registration** | 6-DOF registration by normalised cross-correlation, with the B0 direction of each orientation (`register_rigid`) | — |
| **Resampling / reorientation** | Affine-derived B0 direction and obliquity, reorientation, and resampling (complex-domain for phase) to an axial or another grid (`geometry`) | — |

## Reference Implementations

This library was developed with reference to the following open-source implementations:

| Repository | Algorithms | Language |
|------------|------------|----------|
| [QSM.jl](https://github.com/kamesy/QSM.jl) | SHARP, V-SHARP, SMV, PDF, iSMV, LBV, Laplacian unwrap, TKD, TSVD, Tikhonov, TV, RTS, NLTV, multi-echo linear fit | Julia |
| [FANSI-toolbox](https://gitlab.com/cmilovic/FANSI-toolbox) | NDI, FANSI (nlTV/nlTGV), L1-QSM, WH-QSM | MATLAB |
| [HD-QSM](https://github.com/mglambert/HD-QSM) | HD-QSM | MATLAB |
| [QSM_AMP_PE](https://github.com/EmoryCN2L/QSM_AMP_PE) | AMP-PE | MATLAB |
| [qsm_heidi](https://gitlab.com/R01NS114227/qsm_heidi) | LSQR, HEIDI | MATLAB |
| [QuantitativeSusceptibilityMappingTGV.jl](https://github.com/korbinian90/QuantitativeSusceptibilityMappingTGV.jl) | TGV | Julia |
| [MriResearchTools.jl](https://github.com/korbinian90/MriResearchTools.jl) | ROMEO, MCPC-3D-S, B0 weighted average, R2*/T2*, bias correction, hole filling | Julia |
| [MEDI_toolbox](https://github.com/huawu02/MEDI_toolbox) | MEDI | MATLAB |
| [FSL-BET2](https://github.com/Bostrix/FSL-BET2) | BET | C++ |
| [Rodent-Skull-Stripping](https://github.com/VitoLin21/Rodent-Skull-Stripping) | RS2-Net | Python |
| [brainchop-cli](https://github.com/neuroneural/brainchop-cli) | MindGrab | Python |
| [niimath](https://github.com/rordenlab/niimath) | MindGrab conform and reslice | C |
| [QSMART](https://github.com/wtsyeda/QSMART) | SDF, QSMART pipeline, Frangi filter, vasculature mask, curvature | MATLAB |
| [CLEARSWI.jl](https://github.com/korbinian90/CLEARSWI.jl) | CLEAR-SWI | Julia |
| [SEPIA](https://github.com/kschan0214/sepia) | SMWI | MATLAB |
| [QSM-CI](https://github.com/QSMxT/QSM-CI) | Signal-gated mask erosion | Python |
| [ITK](https://github.com/InsightSoftwareConsortium/ITK) | Gradient anisotropic diffusion, N4 bias correction | C++ |
| [ANTs](https://github.com/ANTsX/ANTs) | N4 bias correction (command-line defaults) | C++ |
| [chi-separation](https://github.com/SNU-LIST/chi-separation) | Chi-separation | MATLAB |

## License

This project is licensed under the [MIT License](LICENSE).

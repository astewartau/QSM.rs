# QSM.rs

A Rust library for Quantitative Susceptibility Mapping (QSM) of the brain.

QSM.rs provides a complete set of algorithms for reconstructing magnetic susceptibility maps from MRI phase data, including brain extraction, phase unwrapping, background field removal, and dipole inversion.

**[Website](https://astewartau.github.io/QSM.rs/)** · **[API Documentation](https://astewartau.github.io/QSM.rs/api/qsm_core/)**

QSM.rs is the shared reconstruction engine behind the wider QSM ecosystem: the
[QSMxT](https://qsmxt.github.io/QSMxT/) command-line pipeline and the
[QSMbly](https://qsmbly.neurodesk.org/) in-browser app both call directly into this
library, and [QSM-CI](https://qsmxt.github.io/QSM-CI/) benchmarks these methods against
phantom ground truth. See the [ecosystem hub](https://qsmxt.github.io/) for an overview.

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
let unwrapped = unwrap::laplacian_unwrap(phase, &mask, &grid, unwrap::LaplacianSolver::Dct);
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
| **BET** | Brain Extraction Tool — region-growing brain masking with mesh evolution. `voxel_scale` inflates the voxel sizes so BET's human-scale surface model can be pointed at preclinical data (10 for mouse), though `rs2_net` masks rodent brains better where its weights are available | Smith, S.M. (2002). "Fast robust automated brain extraction." *Human Brain Mapping*, 17(3):143-155. [DOI](https://doi.org/10.1002/hbm.10062) |
| **RS2-Net** | Deep-learning **rodent** (mouse, rat) brain extraction from the magnitude: a Swin-UNETR trained on 1,142 MRIs from 89 centres, run through a Rust port of its nnU-Net pipeline. Handles thick slices and the thin rodent skull, where BET leaks into muscle (`onnx` feature) | Lin, Y., Ding, Y., Chang, S., Ge, X., Sui, X., Jiang, Y. (2024). "RS2-Net: An end-to-end deep learning framework for rodent skull stripping in multi-center brain MRI." *NeuroImage*, 298:120769. [DOI](https://doi.org/10.1016/j.neuroimage.2024.120769) |
| **Signal-gated erosion** | Mask refinement for any mask: peels only low-signal boundary voxels (sinus / skull-base T2* dropout) down to a depth cap, after dividing out the receive-coil bias; interior dark structures are kept | QSM-CI harmonization masking (`hd-bet-qsmci`), [QSMxT/QSM-CI](https://github.com/QSMxT/QSM-CI) |

### Phase Unwrapping

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **ROMEO** | Region-growing with quality-guided ordering using magnitude and gradient coherence weighting | Dymerska, B., et al. (2021). "Phase unwrapping with a rapid opensource minimum spanning tree algorithm (ROMEO)." *Magnetic Resonance in Medicine*, 85(4):2294-2308. [DOI](https://doi.org/10.1002/mrm.28563) |
| **Laplacian** | Poisson-equation unwrapping that leaves the background field alone, so the result is a total field (`laplacian_unwrap`), with two solvers: `LaplacianSolver::Dct` (default; unweighted least squares — finite differences of wrapped differences, Neumann/DCT solve, as QSM.jl `:dct`; what `UnwrapMethod::Laplacian` selects) and `LaplacianSolver::Fft { pad }` (the sin/cos method with an FFT solve on the zero-padded volume, which reproduces STI Suite 3.0's `MRPhaseUnwrap`; its zero padding adds a harmonic error to the total field, not to the local field). `laplacian_unwrap_multi_echo` builds the per-echo unwrap + weighted echo average field map (`TE·exp(−TE/T2*)` weights, among others). | Ghiglia, D.C., Romero, L.A. (1994). *JOSA A*, 11(1):107-117. [DOI](https://doi.org/10.1364/JOSAA.11.000107); Schofield, M.A., Zhu, Y. (2003). "Fast phase unwrapping algorithm for interferometric applications." *Optics Letters*, 28(14):1194-1196. [DOI](https://doi.org/10.1364/OL.28.001194) |
| **Best path** | 3D-SRNCP — every edge sorted by reliability and merged in that order (`unwrap_bestpath`) | Abdul-Rahman, H., et al. (2007). "Fast and robust three-dimensional best path phase unwrapping algorithm." *Applied Optics*, 46(26):6623-6635. [DOI](https://doi.org/10.1364/AO.46.006623) |
| **PRELUDE** | Best-pair-first region merging: the phase range is split into bins, each bin's connected components become regions, and interfaces are merged in order of how costly getting their offset wrong would be (`unwrap_prelude`). A clean-room implementation from the technical report — FSL's source is non-commercial-licensed and was not consulted. Kept as the baseline the literature benchmarks against. Published comparisons report FSL's implementation as the slowest option; this one is not — on our phantom it runs in 0.19-0.34 s per echo against ROMEO's 0.29-0.34 s, and the four unwrappers' local fields agree to within 0.02 correlation | Jenkinson, M. (2003). "Fast, automated, N-dimensional phase-unwrapping algorithm." *Magnetic Resonance in Medicine*, 49(1):193-197. [DOI](https://doi.org/10.1002/mrm.10354); [FMRIB TR01MJ1](https://www.fmrib.ox.ac.uk/datasets/techrep/tr01mj1/tr01mj1.pdf) |
| **Slice-wise (2D)** | Any of the above run on each slice independently, for 2D multi-slice acquisitions whose slices carry independent receive phase offsets (`unwrap_slicewise`). Also worth using on **thick slices with no offsets at all**, where 3 mm slices alone mislead a 3D region grower (0.936 against 0.983). Makes the unwrapped **phase** usable on such data; for a `linear-fit` B0 map alone, `enforce_inter_echo_consistency` is the part that matters and works on any unwrapper's output — see below. | — |

### 2D multi-slice acquisitions

Every unwrapper above reads across slice boundaries, and a 2D multi-slice acquisition excites
each slice separately, so each one carries its own constant receive phase offset. Unwrapping
slice by slice avoids that, at the cost of fixing each slice only up to a multiple of 2π.

Multi-echo data closes that gap: the offset is echo-independent, so it sits in the intercept of
`phi(TE) = phi0 + gamma*dB*TE` and a fit across echoes returns the field from the slope. That
only works while the leftover 2π is echo-independent too, which is why
`unwrap_slicewise_multi_echo` enforces it rather than leaving it to the B0 fit — the correction
needs the *wrapped* input phase, and the unwrapper is the only stage holding both that and its
own output. The division of labour is: this module makes each echo of a voxel differ from the
last by its measured phase evolution and nothing else; the B0 fit discards the leftover wrap
from the intercept, together with the physical slice offset.

That correction is per voxel, not per slice, and the difference is not cosmetic. A region grower
with no route through the slice axis has to cross every in-plane fringe head-on, and where one
is ambiguous it leaves *part* of a slice offset by 2π. On the qsm-forward 2D phantom the wrap
count varies within a single slice of a single echo, so a per-slice correction has nothing it
can do. Measured during development, per voxel took the fitted field from r = -0.44 to
r = 0.98 against ground truth; the per-slice variant was not kept, so that figure is a record
of the decision rather than something the tests reproduce.

The 3D unwrappers are not a safe fallback *unaided*. On clean data their quality weighting defers
the incoherent z edges, so each slice does come out a whole number of wraps from the truth — but
*which* whole number depends on the echo, because the jump being rounded is
`offset_step + field_step * TE`. `correct_multi_echo_wraps` cannot repair it: it corrects the
whole volume at once and the error is per slice.

**The two halves fix different things, and it is worth knowing which you need.** Measured on the
phantom's 3 mm interleaved-offset session:

| offsets session | linear-fit B0 correlation | in-plane 2π jumps, echo 1 | echo 4 |
|---|---|---|---|
| wrapped input, nothing applied | -0.01 | 1432 | 1433 |
| 3D ROMEO alone | 0.20 | — | — |
| 3D ROMEO + `enforce_inter_echo_consistency` | **0.983** | — | — |
| raw wrapped + `enforce_inter_echo_consistency` | **0.983** | 1432 | 2142 |
| **slice-wise + `enforce_inter_echo_consistency`** | **0.983** | **0** | **796** |

The echo-1 column is the cleanest evidence, and clean *by construction*:
`enforce_inter_echo_consistency` only ever writes echoes 1.. (0-indexed), so it cannot touch the
first echo, and the 1432 → 1432 identity confirms it. Any difference in that column belongs to the
spatial unwrapping and nothing else.

A fitted slope is blind to a constant per voxel, so `enforce_inter_echo_consistency` is what
rescues a `linear-fit` B0 map — from *any* starting point, including no spatial unwrapping at all.
It is public and works on any unwrapper's output, so **if a linear-fit field map is all you want,
you do not need the slice-wise mode**; you need the pass.

What the slice-wise mode fixes is the phase itself. The pass propagates echo 0's wrap state
faithfully into every later echo, so on its own it leaves *more* in-plane 2π discontinuities than
not unwrapping at all. Anything that reads the unwrapped phase rather than its TE-slope — a
single-echo field map, a non-linear B0 estimator, phase fed straight to background removal —
needs the spatial unwrapping, and on 2D multi-slice data that means per slice.

#### Thick slices need this even without per-slice offsets

The slice-wise mode was built for acquisitions whose slices carry independent receive phase
offsets, and it turns out not to need them. On the `ses-thick` session, which has **uniform
receive phase and differs from a 3D acquisition only in being 3 mm thick**, 3D ROMEO reaches
0.936 against slice-wise's 0.983.

3 mm slices by themselves produce through-slice phase steps large enough to mislead a region
grower. Nothing about the receive chain is involved, so the trigger is slice thickness rather
than 2D multi-slice acquisition as such, and the remedy is the same: do not differentiate along
a direction the data is poorly sampled in.

This was found by an assertion failing. An earlier revision of the no-offsets control required
3D to reach 0.95, on the assumption that there is nothing to fix when the receive phase is
uniform. That assumption is wrong, and the test now pins the ordering instead.

Slice-wise Laplacian reaches only 0.81 on the fit, because its Poisson solution is not a whole
number of wraps from the truth to begin with and the pass cannot re-seat it cleanly; prefer
ROMEO or best path.

Background field removal stays **3D**, and so does the dipole inversion. That is not a gap in
the implementation, it is the physics: SHARP and its variants rest on the mean value property of
harmonic functions, which is three-dimensional, and a disc has no equivalent, so V-SHARP run
slice by slice would remove only the in-plane component of a field that is not separable that
way. The dipole kernel is likewise inherently 3D. A 2D multi-slice acquisition still assembles
into a 3D volume, so there is nothing to replace here, only anisotropic voxels to handle, which
the kernels already do through their frequency axes.

A 2D V-SHARP stage chained into 3D PDF, as the EPI-QSM literature uses, was implemented and
measured against this phantom and removed. The numbers are kept here so the case does not have
to be re-implemented to be re-litigated.

On a synthetic background with every source outside the ROI, where the ground truth is exactly
zero (residual RMS, lower is better):

| | slab coverage | enclosed ROI |
|---|---|---|
| 2D V-SHARP | 1.18 | 0.47 |
| 3D V-SHARP | 1.69 | **0.14** |
| 2D V-SHARP → 3D PDF | **0.99** | 0.24 |

On the qsm-forward 2D phantom's `ses-thick` session, correlation against the local field
(higher is better), using the whole head for the enclosed case and a 16-slice slab cut so the
tissue reaches its end slices for the other:

| | 48 mm slab | enclosed ROI |
|---|---|---|
| 2D V-SHARP | 0.9792 | **0.9597** |
| 3D V-SHARP | **0.9820** | 0.9506 |
| 2D V-SHARP → 3D PDF | 0.9791 | 0.8963 |

**The real-phantom differences are not large enough to decide anything**, and on the enclosed
ROI the disc edges the sphere out. What does carry is that the chain, which is the arrangement
actually proposed for 2D data, is the worst of the three on the enclosed ROI and does not
improve on the 2D stage alone on the slab. Together with the synthetic case, where the ground
truth is exact and 3D wins the enclosed ROI by a factor of three, and with the theory (SHARP
rests on a three-dimensional mean value property that a disc does not have), there was no
demonstrated benefit to ship.

The open question is whether a real acquisition has a slab thin enough relative to a useful
kernel radius for the disc to earn its place. See astewartau/QSM.rs#75.


A slice **gap**, where the excited slabs are thinner than the slice pitch, makes the sampled
volume non-contiguous. The FFT dipole kernel and the SHARP family's spherical kernels are defined
on a uniform grid, so on gapped data they compute a different convolution and the result is wrong
rather than approximate. It is therefore refused, not warned about.

Set `ScanMetadata::slice_geometry` from the acquisition's `SliceThickness` and every pipeline
stage that convolves on the voxel grid will refuse a gapped volume before doing any work — all
nine of background removal, dipole inversion, TGV, the four single-step DL reconstructions
(NeXtQSM, iQSM, iQSM+, iQFM), chi-separation and QSMART — returning `PipelineError::InvalidInput`
with the thickness, pitch and gap in the message.
The thickness has to be supplied because nothing in a NIfTI records it: the spacing is the
*pitch*, so 3 mm slices every 3 mm and 2 mm slices every 3 mm are indistinguishable. `None` means
the acquisition did not say, in which case nothing can be concluded and the stages run; BIDS only
*recommends* `SliceThickness`, so hosts that cannot read it should tell the user that this check
could not run.

Masking and field mapping are exempt, by design. Masking reads no kernel and produces no field.
Unwrapping is degraded by a gap rather than invalidated by it, and the remedy is to unwrap in
plane — refusing there would block the one path that handles the case. The exemptions are a list
in `pipeline::config` that a test cross-checks against the stage sources, so a stage added later
cannot skip the check by omission.

Below the pipeline, `Grid::require_contiguous_slices(slice_thickness, slice_axis)` is the
primitive and `require_contiguous_slices_if_known` is the `Option`-aware form. Purely in-plane
work is unaffected, which is why slice-wise unwrapping does not call it.

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
| **Laplacian (ROI-masked)** | FFT-based Poisson solver with the Laplacian zeroed outside the mask — discards sources outside the ROI, so unwrapping and harmonic background removal happen together and the result is **not** a total field (`laplacian_unwrap_bfr`) | Schofield, M.A., Zhu, Y. (2003). "Fast phase unwrapping algorithm for interferometric applications." *Optics Letters*, 28(14):1194-1196. [DOI](https://doi.org/10.1364/OL.28.001194); Zhou, D., et al. (2014). "Background field removal by solving the Laplacian boundary value problem." *NMR in Biomedicine*, 27(3):312-319. [DOI](https://doi.org/10.1002/nbm.3064) |

> The two Laplacian entries are the same unwrapping method under different boundary
> conditions, and they are **not** interchangeable. `laplacian_unwrap_bfr` removes the harmonic
> background as a side effect of masking the Laplacian; pairing it with a separate
> background-removal stage removes background twice. `laplacian_unwrap` unwraps
> only. See the `unwrap::laplacian` module docs.
>
> The combined function zeroes ∇² outside the mask — deleting the exterior sources that
> generate the background, since a field produced outside the ROI is harmonic inside it —
> and solves under a homogeneous Dirichlet condition on the ROI. On the test data it reaches
> **r = 0.887** against the ground-truth local field, against 0.909 for `bgremove::lbv` and
> 0.879 for V-SHARP. `UnwrapMethod::Laplacian` still selects the plain variant, since the
> pipeline removes background as a later stage.

### Grid size and reconstruction cost

FFT-based stages cost `O(N log N)` in the whole grid, not in the brain, and `rustfft` is much
faster on sizes whose prime factors are small. [`crop`](src/crop.rs) provides both levers:

- `fft_pad_box` grows each axis outward to the next 7-smooth size. Nothing is discarded and the
  periodic boundary moves *away* from the object. An axially-resampled UK Biobank grid of
  272×339×77 (2⁴·17, 3·113, 7·11) pads to 280×343×80 — 8% more voxels, and the transform drops
  from 131 ms to 71 ms.
- `crop_box_for_mask` shrinks to the mask plus a margin in millimetres, rounding each axis to a
  friendly size. This one moves the boundary *closer*, so it changes the reconstruction: on real
  data a crop that removed voxels shifted χ by ~0.6% of its dynamic range at the median and ~4%
  at the 99th percentile. Validate before relying on it.

### Acquisition orientation

The dipole relationship depends on which way B0 points, and the FFT that implements it lives in
the voxel grid, so an oblique acquisition has to be handled deliberately. Each algorithm reports
what it can do via `orientation_support()`:

| | meaning | methods |
|---|---|---|
| `Arbitrary` | B0 direction is an explicit parameter, so oblique data reconstructs correctly on its acquired grid | every classical dipole inversion, PDF, `chi-sep` iLSQR/MEDI |
| `NotApplicable` | never uses B0 — the SMV family is harmonic and rotation-invariant | SHARP, V-SHARP, RESHARP, iSMV, LBV, HARPERELLA, iHARPERELLA, BFRnet, and the separations that consume an existing χ map |
| `AxialOnly` | assumes B0 along `+z` with no way to say otherwise; oblique data must be resampled first | every deep-learning inversion and separation |

Getting this wrong is silent — the reconstruction completes and the values are simply wrong — so
hosts should check `orientation_support().requires_axial()` before running an oblique dataset.
See [`geometry`](src/geometry.rs) for the direction itself and for resampling to a cardinal grid,
and [`registration`](src/registration.rs) for recovering the rigid transform between two volumes —
which is both how a multi-orientation set gets onto one common grid and where its per-orientation
B0 directions come from, since the object rotates and B0 does not. The same applies within a
single acquisition once the subject moves: [`motion`](src/motion.rs) applies it along a series of
repeats, runs or echoes, and reports one B0 direction per volume for the same reason.

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
| **NDI** | Nonlinear Dipole Inversion — gradient-descent solve of a nonlinear (wrapped-phase) data-fidelity term; effectively tuning-free | Polak, D., et al. (2020). "Nonlinear dipole inversion (NDI) enables robust quantitative susceptibility mapping (QSM)." *NMR in Biomedicine*, 33(12):e4271. [DOI](https://doi.org/10.1002/nbm.4271) |
| **FANSI (nlTV / nlTGV)** | Fast Nonlinear Susceptibility Inversion — nonlinear total-variation and total-generalized-variation regularization via ADMM | Milovic, C., et al. (2018). "Fast nonlinear susceptibility inversion with variational regularization." *Magnetic Resonance in Medicine*, 80(2):814-821. [DOI](https://doi.org/10.1002/mrm.27073) |
| **L1-QSM** | L1-norm data-fidelity QSM (PI-QSM) — robust to phase inconsistencies via an L1 fidelity term with TV regularization | Milovic, C., et al. (2022). "Comparison of parameter optimization methods for quantitative susceptibility mapping." *Magnetic Resonance in Medicine*, 87(3):1517-1531. [DOI](https://doi.org/10.1002/mrm.28957) |
| **WH-QSM** | Weak-Harmonic QSM — jointly estimates susceptibility and a residual harmonic background field, correcting imperfect background-field removal | Milovic, C., et al. (2019). "Weak-harmonic regularization for quantitative susceptibility mapping." *Magnetic Resonance in Medicine*, 81(2):1399-1411. [DOI](https://doi.org/10.1002/mrm.27483) |
| **HD-QSM** | Hybrid Data-fidelity QSM — two-stage linear inversion where an L1 stage produces a discrepancy map that reweights a second L2 stage | Lambert, M., et al. (2022). "Hybrid data fidelity term approach for quantitative susceptibility mapping." *Magnetic Resonance in Medicine*, 88(4):1567-1583. [DOI](https://doi.org/10.1002/mrm.29218) |
| **AMP-PE** | Approximate Message Passing with Parameter Estimation — generalized approximate message passing over a linearized wrapped-phase model with a sparse-wavelet prior and a Gaussian-mixture noise model; regularization and noise parameters are estimated automatically | Huang, S., et al. (2023). "Approximate Message Passing with Parameter Estimation: a probabilistic Bayesian dipole inversion." *Magnetic Resonance in Medicine*, 90(4):1414-1430. [DOI](https://doi.org/10.1002/mrm.29722) |
| **LSQR** | Minimally regularized least-squares inversion — solves for χ jointly with a weighted residual field and a global offset, relying on LSQR's minimum-norm property rather than an explicit penalty | Schweser, F., et al. (2010). "Differentiation between diamagnetic and paramagnetic cerebral lesions based on magnetic susceptibility mapping." *Medical Physics*, 37(10):5165-5178. [DOI](https://doi.org/10.1118/1.3481505) |
| **HEIDI** | Homogeneity Enabled Incremental Dipole Inversion — keeps the well-conditioned k-space coefficients of an LSQR map as exact constraints and recovers the ill-conditioned dipole cone under a weighted-TV prior whose per-direction weights come from the field map's own gradient and Laplacian | Schweser, F., et al. (2012). "Quantitative susceptibility mapping for investigating subtle susceptibility variations in the human brain." *NeuroImage*, 62(3):2083-2100. [DOI](https://doi.org/10.1016/j.neuroimage.2012.05.067) |

### End-to-End QSM

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **TGV** | Total Generalized Variation — single-step QSM from wrapped phase, combining unwrapping, background removal, and dipole inversion | Langkammer, C., et al. (2015). "Fast quantitative susceptibility mapping using 3D EPI and total generalized variation." *NeuroImage*, 111:622-630. [DOI](https://doi.org/10.1016/j.neuroimage.2015.02.041) |
| **QSMART** | Two-stage QSM artifact reduction using SDF background removal, TKD inversion, and Frangi vesselness-based tissue/vasculature separation | Yaghmaie, N., Syeda, W., et al. (2021). "QSMART: Quantitative Susceptibility Mapping Artifact Reduction Technique." *NeuroImage*, 231:117701. [DOI](https://doi.org/10.1016/j.neuroimage.2020.117701) |

### Artefact Reduction

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **Two-pass** | Reconstructs twice — once on a mask whose holes around strong susceptibility sources are left intact, once on the filled mask — and keeps the hole-preserving reconstruction wherever it is defined, so the streaking those sources cause stays out of the rest of the brain | Stewart, A.W., et al. (2022). "QSMxT: Robust masking and artifact reduction for quantitative susceptibility mapping." *Magnetic Resonance in Medicine*, 87(3):1289-1300. [DOI](https://doi.org/10.1002/mrm.29048) |

### SWI Processing

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **CLEAR-SWI** | Susceptibility Weighted Imaging — phase mask weighting with high-pass filtering and minimum intensity projection | Eckstein, K., et al. (2024). "CLEAR-SWI: Computational Efficient T2* Weighted Imaging." *Proc. ISMRM*. |
| **SMWI** | Susceptibility Map-Weighted Imaging — magnitude weighted by a mask built from the susceptibility map rather than filtered phase, giving paramagnetic and diamagnetic contrasts free of phase blooming; one mask is broadcast over all echoes for multi-echo SMWI | Gho, S.-M., et al. (2014). "Susceptibility map-weighted imaging (SMWI) for neuroimaging." *Magnetic Resonance in Medicine*, 72:337-346. [DOI](https://doi.org/10.1002/mrm.24920) |

### Multi-Echo Processing

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **MCPC-3D-S** | Multi-Channel Phase Combination (ASPIRE) — combines uncombined receive-coil channels by estimating each coil's phase offset from the coil-summed inter-echo Hermitian inner product (`mcpc3ds_combine`), and removes the residual TE-independent offset from already-combined multi-echo phase (`phase_offset_removal`) | Eckstein, K., et al. (2018). "Computationally Efficient Combination of Multi-channel Phase Data From Multi-echo Acquisitions (ASPIRE)." *Magnetic Resonance in Medicine*, 79:2996-3006. [DOI](https://doi.org/10.1002/mrm.26963) |
| **R2\*/T2\* (ARLO)** | R2* mapping from multi-echo magnitude using Auto-Regression on Linear Operations; T2* = 1/R2* | Pei, M., et al. (2015). "Algorithm for fast monoexponential fitting based on Auto-Regression on Linear Operations (ARLO) of data." *Magnetic Resonance in Medicine*, 73(2):843-850. [DOI](https://doi.org/10.1002/mrm.25137) |
| **R2 (EPG)** | R2 mapping from multi-echo spin-echo (MESE) via Extended Phase Graph dictionary matching; models imperfect refocusing (B1 < 180°) to remove the stimulated-echo bias a mono-exponential fit suffers | Weigel, M. (2015). "Extended phase graphs: dephasing, RF pulses, and echoes — pure and simple." *Journal of Magnetic Resonance Imaging*, 41(2):266-295. [DOI](https://doi.org/10.1002/jmri.24619) |
| **R2′ (R2\* − R2)** | Reversible transverse relaxation from paired gradient-echo (R2*) and spin-echo (R2) acquisitions — the input required by χ-separation. Returns the region R2 was measured over alongside the map, since a spin-echo companion is routinely a slab rather than the whole head | Yablonskiy, D.A., Haacke, E.M. (1994). "Theory of NMR signal behavior in magnetically inhomogeneous tissues: the static dephasing regime." *Magnetic Resonance in Medicine*, 32(6):749-763. [DOI](https://doi.org/10.1002/mrm.1910320610) |
| **R2PRIMEnet** | Learned R2*→R2′ conversion for the **GRE-only** condition, where no spin-echo R2 is acquired: a 3D U-Net predicts the reversible component from the GRE-derived R2* map, supplying the R2′ that every χ-separation method needs. **3 T only** — the weights are trained at 3 T and R2* scales with field strength, so the call refuses outside that band rather than returning a confident wrong answer. Sliding-window patches, size configurable so 32-bit hosts can trade a little accuracy for memory (`onnx` feature) | Kim, M., Ji, S., Lee, J., et al. (2025). "χ-sepnet: Deep neural network for magnetic susceptibility source separation." *Human Brain Mapping*, 46(4):e70136. [DOI](https://doi.org/10.1002/hbm.70136) |

### Preprocessing

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **Bias Correction** | Homogeneity correction for high-field MRI | Eckstein, K., Trattnig, S., Robinson, S.D. (2019). "A Simple Homogeneity Correction for Neuroimaging at 7T." *Proc. ISMRM 27th Annual Meeting*. |
| **MP-PCA Denoising** | Marchenko–Pastur PCA denoising of multi-echo / multi-volume data; removes random noise along the volume dimension with a parameter-free (random-matrix) threshold, edge-preserving | Veraart, J., et al. (2016). "Denoising of diffusion MRI using random matrix theory." *NeuroImage*, 142:394-406. [DOI](https://doi.org/10.1016/j.neuroimage.2016.08.016) |
| **Gibbs Unringing** | Removal of k-space truncation (Gibbs) ringing via local subvoxel shifts, generalised to full 3D | Kellner, E., et al. (2016). "Gibbs-ringing artifact removal based on local subvoxel-shifts." *Magnetic Resonance in Medicine*, 76(5):1574-1581. [DOI](https://doi.org/10.1002/mrm.26054) |
| **Motion-Robust Echo Combination** | Iteratively reweighted least squares around the multi-echo fit, with a robust loss on each echo's residual: an echo that is corrupted in a way no rigid transform can undo — spin-history dropout, a mistimed shot, phase that is simply not on the line the others define — is pulled out of the fit where it misbehaves instead of dragging the slope everywhere it touches. Reports a per-echo score and names the offender; it reports more reliably than it repairs, and deliberately drops nothing. The companion to motion correction, for what registration cannot reach | Biondetti, E., et al. (2022). *Magnetic Resonance in Medicine* — found weighted and nonlinear-fit echo combination clearly outperforms unweighted averaging or plain linear fitting. [PMC9545116](https://pmc.ncbi.nlm.nih.gov/articles/PMC9545116/) |
| **Motion Correction** | Retrospective and image-domain: rigidly co-registers the volumes of a series — repeats, runs or echoes — onto one of them, moves each magnitude/phase pair through the complex domain so wrapped phase survives the interpolation, and re-derives B0 per volume because the head rotates and B0 does not. Average the registered series, or pass it downstream per volume with its own direction. The best QSM motion correction in the literature is *prospective*, navigator-driven and needs raw k-space plus a modified sequence; none of that is reachable from reconstructed NIfTI, which is the boundary this draws | "Effects of Motion in Ultrashort Echo Time QSM for Musculoskeletal Imaging" (2025) — registers every echo to the first TE as fixed reference and reports significantly reduced streaking, with a residual boundary effect from motion it could not correct. [PMC12565248](https://pmc.ncbi.nlm.nih.gov/articles/PMC12565248/) |

### Susceptibility Source Separation

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **χ-separation** | Gauss-Newton optimization separating total susceptibility into paramagnetic (iron) and diamagnetic (myelin) components using coupled field and R2' constraints | Shin, H., et al. (2021). "χ-separation: Magnetic susceptibility source separation toward iron and myelin mapping in the brain." *NeuroImage*, 240:118371. [DOI](https://doi.org/10.1016/j.neuroimage.2021.118371) |
| **HC-ChiSep** | Hollow-cylinder χ-separation with the white-matter fibre-to-B0 angle derived from the multi-echo GRE magnitude's multi-compartment interference rather than imported from DTI; R2′-anchored θ/MWF fit, myelin-content↔MWF anchor for χ− | Wharton, S., Bowtell, R. (2012). "Fiber orientation-dependent white matter contrast in gradient echo MRI." *PNAS*, 109(45):18559-18564. [DOI](https://doi.org/10.1073/pnas.1211075109) |
| **R2\*-QSM** | Closed-form separation from gradient-echo data alone — a QSM plus R2*, with one relaxometric constant for both sources and no R2′ measurement | Dimov, A.V., et al. (2022). "Magnetic susceptibility source separation solely from gradient echo data: histological validation." *Tomography*, 8(3):1544-1551. [DOI](https://doi.org/10.3390/tomography8030128) |
| **WaveSep** | Wavelet-domain L1 proximal-gradient separation from a QSM and R2′ | Fang, Z., et al. (2023). "Wavelet-based single-step susceptibility source separation." *Proc. ISMRM*. |
| **DECOMPOSE** | Signal-domain three-compartment fit of the complex multi-echo gradient-echo signal per voxel (paramagnetic / diamagnetic / neutral) | Chen, J., et al. (2021). "Decompose quantitative susceptibility mapping (QSM) to sub-voxel diamagnetic and paramagnetic components based on gradient-echo MRI data." *NeuroImage*, 242:118477. [DOI](https://doi.org/10.1016/j.neuroimage.2021.118477) |
| **χ-sepnet / SUSEP-Net** | Trained 3D U-Nets mapping [QSM, local field, R2′] to the two source maps (`onnx` feature). Both run as sliding windows with a configurable patch, so a 32-bit host can trade a little accuracy for bounded memory | Kim, M., et al. (2025). *Human Brain Mapping*, 46(4):e70136. [DOI](https://doi.org/10.1002/hbm.70136) · Li, Z., Gao, Y., Sun, H., et al. (2025). arXiv:2506.13293 |

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
| [FANSI-toolbox](https://gitlab.com/cmilovic/FANSI-toolbox) | NDI, FANSI (nlTV/nlTGV), L1-QSM, WH-QSM | MATLAB |
| [HD-QSM](https://github.com/mglambert/HD-QSM) | HD-QSM | MATLAB |
| [QSM_AMP_PE](https://github.com/EmoryCN2L/QSM_AMP_PE) | AMP-PE | MATLAB |
| [qsm_heidi](https://gitlab.com/R01NS114227/qsm_heidi) | LSQR, HEIDI | MATLAB |
| [QuantitativeSusceptibilityMappingTGV.jl](https://github.com/korbinian90/QuantitativeSusceptibilityMappingTGV.jl) | TGV | Julia |
| [MriResearchTools.jl](https://github.com/korbinian90/MriResearchTools.jl) | ROMEO, MCPC-3D-S, R2*/T2*, bias correction | Julia |
| [MEDI_toolbox](https://github.com/huawu02/MEDI_toolbox) | MEDI | MATLAB |
| [FSL-BET2](https://github.com/Bostrix/FSL-BET2) | BET | C++ |
| [Rodent-Skull-Stripping](https://github.com/VitoLin21/Rodent-Skull-Stripping) | RS2-Net | Python |
| [QSMART](https://github.com/wtsyeda/QSMART) | SDF, QSMART pipeline, Frangi filter, curvature | MATLAB |
| [CLEARSWI.jl](https://github.com/korbinian90/CLEARSWI.jl) | CLEAR-SWI | Julia |
| [SEPIA](https://github.com/kschan0214/sepia) | SMWI | MATLAB |
| [QSM-CI](https://github.com/QSMxT/QSM-CI) | Signal-gated mask erosion | Python |
| [chi-separation](https://github.com/SNU-LIST/chi-separation) | Chi-separation | MATLAB |

## License

This project is licensed under the [MIT License](LICENSE).

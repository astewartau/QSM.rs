//! mSMV (maximum spherical mean value) filtering, the field step of MEDI-mSMV
//!
//! mSMV removes residual background field near the brain boundary, the main source of shadow
//! artifacts, without eroding the mask (Roberts et al. 2024). It is **not a background field
//! removal step on its own**: the authors run it inside MEDI, as `MEDI_L1(..., 'msmv', radius)`,
//! on the local field from a conventional background removal (PDF, V-SHARP or LBV in the
//! paper), and the inversion then fits the SMV-filtered field with the SMV-filtered dipole
//! kernel `(δ − S_r1) d` (the paper's Eq. 12, "MEDI-mSMV"). Its output is a high-passed field,
//! `(δ − S_r1)` applied to the local field plus a boundary correction, so inverting it with the
//! plain dipole kernel `d`, as a background removal output would be, is the wrong forward
//! model. In QSM.rs it is therefore an option of [`medi`](fn@super::medi):
//! [`MediParams::msmv`](super::MediParams::msmv). This module holds the filter itself, a port of
//! `msmv.m` and `kernel_lim.m`, for MEDI and for parity checks.
//!
//! # The authors' method, as their code runs it
//! Reference code: agr78/mSMV (commit 42a1f18), `prerelease/MEDI_toolbox-mSMV100125.zip`, the
//! newest MEDI toolbox with mSMV: `functions/msmv.m`, `kernel_lim.m`, `MaskErode.m`, `SMV.m`,
//! `sphere_kernel.m`, `MEDI_L1.m` and `parse_QSM_input.m`. `MEDI_L1(..., 'msmv', r)` (the
//! `'msmv'` keyword takes the radius, like `'smv'`, and turns SMV on):
//!
//! 1. reads `R2s` from the input file and **errors if it is missing**;
//! 2. does **not** erode the mask (`smv_shrink_mask` is forced off) and calls
//!    `RDF = msmv(RDF, Mask, R2s, voxel_size, r, B0_mag, prefilter, maxk, vessel_radius)`;
//! 3. inverts with `D = (1 − SphereK).*D` and the SMV-propagated noise
//!    `tempn = sqrt(SMV(tempn.^2) + tempn.^2)`, exactly as MEDI-SMV does, over the full mask.
//!
//! `msmv.m` itself, in MEDI's units (the field in radians over the echo spacing `delta_TE`):
//!
//! - `SphereK = single(sphere_kernel(..., radius))`, the partial-volume sphere of MEDI;
//!   `Mne = SMV(Mask, SphereK) > 0.999`, `Me = Mask − Mne` (the boundary band of width ~radius).
//! - Prefilter (default on): `RDF_s = Mask.*(RDF − SMV(RDF, SphereK))` (the paper's Eq. 4). With
//!   `prefilter = 0` the input is used as it is (not masked); the README says this is
//!   "typically used with SHARP, RESHARP, VSHARP", i.e. background removals that already
//!   applied an SMV. The paper ran it with the prefilter on after PDF, V-SHARP and LBV.
//! - Threshold `t = kernel_lim(RDF, ...)` on the **unfiltered** input: starting from a rendered
//!   sphere of half the smallest voxel, the radius grows in 1e-3 mm steps and `t` is the first
//!   `max|Mask.*(RDF − SMV(RDF, K))|` that reaches `tmin = 0.01·B0_mag/3` rad. At half a voxel
//!   the rendered sphere is the centre voxel alone (the residual is 0); the residual then grows
//!   in steps as sub-voxel samples of the neighbours enter the sphere, so `t` is either just
//!   above `tmin` (smooth fields) or the first nonzero step, which strong local fields put well
//!   above it (0.138 rad against `tmin` = 0.023 rad on the 7 T qsm-forward phantom after PDF):
//!   the code's form of the paper's `max(t_min, lim_{r'→0} ...)`.
//! - Vessel mask: `Mask_ev = Mask − MaskErode(Mask, radius + 1)`,
//!   `Mv = imbinarize(fibermetric(Mask_ev.*R2s, 1:vessel_radius, 'ObjectPolarity', 'bright'), 0)`
//!   with `vessel_radius = max(15, 8·max(voxel_size))` by default. `fibermetric` takes the
//!   thicknesses in voxels; the threshold 0 keeps every voxel with any bright-tubular response at
//!   any of the 15+ scales (the Frangi sign condition `λ2, λ3 < 0`), not only veins: 70 % of the
//!   band on the qsm-forward phantom (in MATLAB as here), so the correction acts on the rest.
//! - `Mb = imbinarize(|Me.*RDF_s|, t) & ~Mv`; while `sum(Mb)/sum(Mask) > 1e-6`: recompute `Mb`
//!   from the current field, `RDF_s = Mask.*(RDF_s − SMV(Mb.*RDF_s, r2))` with the minimum
//!   radius `r2 = min(voxel_size)/2 + 0.05` mm, stop after `maxk − 1` passes (default `maxk` 5,
//!   so at most 4 passes). The loop test uses the `Mb` of the previous pass.
//!
//! # Where the paper and the code differ (the code is followed)
//! - **Threshold.** The paper (Eq. 7) defines `t = max(t_min, lim_{r'→0} max_M |b_L0 − S_r' b_L0|)`
//!   on the prefiltered field with `t_min = 0.3 Hz` at 3 T, "scaled accordingly for other field
//!   strengths". `kernel_lim.m` uses the unfiltered input field and `tmin = 0.01·B0/3` **radians
//!   over `delta_TE`**, which is 0.3 Hz only for `delta_TE` ≈ 5.3 ms (0.49 Hz at the paper's
//!   in-vivo 3.25 ms). The field strength comes from `B0_mag`, which `MEDI_L1` reads from the
//!   input file and otherwise leaves at `msmv.m`'s default of 3 T. QSM.rs passes the scan's
//!   field strength, as the paper's scaling asks; at 3 T this is the code's value.
//! - **Iterations.** The paper's `i_max = 5`; the code's `maxk = 5` gives at most 4 passes.
//! - **Vessel mask.** The paper says "a Frangi filter ... followed by thresholding" without the
//!   scales or threshold; the code's are above.
//!
//! # Ported here
//! [`msmv`] follows `msmv.m` line for line, with MEDI's rendered `sphere_kernel` for every SMV
//! (the mSMV release's version, which takes the real part of the kernel's FFT), `kernel_lim.m`
//! as written, `MaskErode.m`, and MATLAB's `fibermetric` (R2023b, default
//! `StructureSensitivity` 0.01 for double input, ported in
//! [`fibermetric`](mod@crate::utils::fibermetric)). R2* is a required input, as it is for
//! `MEDI_L1`'s mSMV path; MEDI's pipeline (and the authors' SEPIA patch) computes it with ARLO
//! from the echo magnitudes, which is what [`medi`](fn@super::medi) does
//! ([`arlo_unclipped`](crate::utils::arlo_unclipped): `arlo.m` exactly, negative values kept,
//! since the vessel filter sees them).
//!
//! Differences from MATLAB, none of which changes the method:
//! - Arithmetic is in double. `msmv.m` stores `SphereK` as single, which makes `SMV(RDF, SphereK)`
//!   and, with the prefilter on, the whole filtered field single precision; here the kernel is
//!   rounded to single as MATLAB stores it but the field stays double. Voxels within single
//!   rounding of the threshold `t` can be classified differently.
//! - SMVs with small kernels (`kernel_lim`'s growing sphere and the minimum-radius `r2` kernel)
//!   are applied as a direct circular convolution with the same kernel instead of by FFT (same
//!   operator, equal to rounding; `kernel_lim` runs dozens to hundreds of them).
//! - `fibermetric` is evaluated only on the boundary band `Me`, the only place `Mv` is used
//!   (`Mb` is zero outside `Me`), so [`MsmvOutput::vessel_mask`] is false outside it.
//! - `kernel_lim.m` loops forever if the residual never reaches `tmin` (a field below `tmin/2`
//!   everywhere); here the threshold is then `tmin`, which flags no voxel (see [`kernel_lim`]).
//!
//! # Parity with the authors' MATLAB
//! Against `msmv.m` and `MEDI_L1(..., 'msmv', 5)` (R2023b, the prerelease above):
//! - Small phantom (`tests/msmv_medi_parity.rs`, 32×36×28, anisotropic voxels): `t` identical,
//!   `Me` and the vessel mask identical (5615 of 6272 band voxels), the filtered field to 5e-8
//!   rad (relative 1e-7; single rounding) with the prefilter and to 1e-16 without; one MEDI-mSMV
//!   Gauss-Newton step to 3e-7 ppm (relative 6e-7).
//! - qsm-forward phantom after MATLAB's `PDF.m` (164×206×206 once padded, 7 T): `t` 0.138 rad
//!   equal to 1e-13 relative, `Me` (321,821 voxels) and `Mv` (225,330) identical, the field to
//!   1.5e-7 relative with the prefilter and 3e-16 without. MEDI-mSMV with MEDI's defaults agrees
//!   to 4e-3 relative L2 (1e-2 ppm max) and one step solved to `cg_tol` 1e-4 to 2.4e-4 (4e-4 ppm
//!   max), as close as MEDI-SMV gets in the same setup (1.9e-4): MATLAB's MEDI runs in single
//!   precision, and at the default `cg_tol` 0.01 CG can stop one iteration apart, which moves χ
//!   by ~1e-3 (see the parity test's docs).
//!
//! # Measured
//! QSM.rs MEDI with its defaults (λ 1000, MERIT, MEDI+0), after QSM.rs PDF, V-SHARP and LBV,
//! comparing MEDI without SMV (the paper's Eq. 14), MEDI-SMV (Eq. 13, mask eroded by 5 mm) and
//! MEDI-mSMV (Eq. 12). "GM variance" is the paper's shadow score, the χ variance in grey matter
//! (×1e-4 ppm²; lower means less shadow, down to the true variation).
//!
//! qsm-forward phantom (7 T, TE 4/12/20/28 ms, 1 mm; ground-truth GM variance 1.61), XSIM and
//! NRMSE over the brain mask (MEDI-SMV is zero on the third of the mask it erodes):
//!
//! | field | BFR | MEDI XSIM / NRMSE / GM var | MEDI-SMV | MEDI-mSMV |
//! |---|---|---|---|---|
//! | noise-free | PDF | 0.597 / 65 % / 3.64 | 0.563 / 63 % / 2.37 | 0.561 / 65 % / 2.92 |
//! | noise-free | V-SHARP | 0.548 / 63 % / 2.88 | 0.510 / 66 % / 2.46 | 0.547 / 63 % / 2.42 |
//! | noise-free | LBV | 0.612 / 59 % / 3.32 | 0.529 / 65 % / 2.40 | 0.606 / 59 % / 2.46 |
//! | SNR 50 | PDF | 0.305 / 103 % / 9.20 | 0.352 / 76 % / 3.55 | 0.305 / 90 % / 5.58 |
//! | SNR 50 | V-SHARP | 0.321 / 80 % / 3.87 | 0.335 / 77 % / 3.12 | 0.333 / 78 % / 3.07 |
//! | SNR 50 | LBV | 0.296 / 98 % / 9.84 | 0.336 / 77 % / 3.48 | 0.358 / 78 % / 3.52 |
//!
//! In vivo, four 3 T multi-echo scans (EMTP travelling heads in QSM-CI: Philips Achieva and
//! Ingenia, Siemens Skyra and Prisma), GM variance in the SynthSeg cortex, mean of the four
//! (MEDI-SMV covers only 55–67 % of the cortex; MEDI and MEDI-mSMV 93–100 %):
//!
//! | BFR | MEDI | MEDI-SMV | MEDI-mSMV |
//! |---|---|---|---|
//! | PDF | 15.1 | 6.0 | 7.3 |
//! | V-SHARP | 7.0 | 5.5 | 4.5 |
//! | LBV | 17.0 | 5.8 | 6.2 |
//!
//! MEDI-mSMV lowered the GM variance against plain MEDI for every background removal and every
//! scan (PDF by half, LBV to a third, V-SHARP by a third, the paper's direction and size), to
//! about MEDI-SMV's level without its erosion, and lowered the χ spread in the 5 mm edge band.
//! On the phantom it did not raise XSIM in general: noise-free it was level with MEDI after
//! V-SHARP and LBV and lower after PDF (0.561 vs 0.597); at SNR 50 it was level after PDF and
//! higher after V-SHARP (0.333 vs 0.321) and LBV (0.358 vs 0.296), with lower NRMSE. It helps
//! most where the local field carries residual background at the edge (PDF, LBV, noisy data)
//! and little where V-SHARP's erosion already removed it.
//!
//! **Prefilter off** (`msmv.m`'s `prefilter = 0`) breaks the forward model in `MEDI_L1`: the
//! field is then the background removal's local field, unfiltered, but `MEDI_L1` still fits it
//! with `(δ − S) d`. On the phantom that gave NRMSE 460 % (PDF), 280 % (V-SHARP) and 740 % (LBV),
//! and the authors' MATLAB gives the same (481 % for PDF + `MEDI_L1(..., 'msmv', 5)` with
//! `prefilter = 0`). It is only consistent for a field that is already high-passed by the same
//! SMV; V-SHARP's deconvolved output is not. The option is kept because the authors' code has
//! it; keep the default.
//!
//! # Use with other inversions
//! Only MEDI is offered. The authors made and tested MEDI-mSMV only (with PDF, V-SHARP and LBV
//! before it). An inversion whose forward model is an explicit dipole convolution could in
//! principle fit the mSMV field with `(δ − S_r1) d` instead of `d`, but none was made or
//! evaluated by the authors, and the result would be a new method, so QSM.rs does not offer one.
//!
//! Reference:
//! Roberts, A.G., Romano, D.J., Şişman, M., Dimov, A.V., Nguyen, T.D., Kovanlikaya, I.,
//! Gauthier, S.A., Wang, Y., Spincemaille, P. (2024). "Maximum spherical mean value filtering for
//! whole-brain QSM." Magnetic Resonance in Medicine, 91(4):1586-1597.
//! https://doi.org/10.1002/mrm.29963
//!
//! Reference implementation: https://github.com/agr78/mSMV (`prerelease/MEDI_toolbox-mSMV100125.zip`)

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::medi::{sphere_kernel_real, sphere_kernel_single, sphere_kernel_sparse};
use crate::fft::apply_real_kernel;
use crate::utils::fibermetric::{fibermetric, FiberPolarity, FIBERMETRIC_DEFAULT_C};
use crate::Grid;

/// mSMV options (`msmv.m`'s `prefilter`, `maxk` and `vessel_radius`, with its defaults). The
/// prefilter radius is MEDI's SMV radius and the field strength the scan's (see
/// [`MediParams::msmv`](super::MediParams::msmv)).
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct MsmvParams {
    /// SMV-filter the field first, `Mask.*(RDF − SMV(RDF))` (default on, as in `msmv.m` and the
    /// paper). The authors' README suggests turning it off after background removals that
    /// already applied an SMV ("typically used with SHARP, RESHARP, VSHARP"); the paper kept it
    /// on after PDF, V-SHARP and LBV. In MEDI-mSMV, off is only consistent for a field already
    /// high-passed by the same 5 mm SMV: MEDI still fits it with the SMV-filtered dipole kernel,
    /// and after PDF, V-SHARP or LBV that diverges (NRMSE 280–740 % on the phantom, in the
    /// authors' MATLAB too; see the module docs).
    pub prefilter: bool,
    /// `maxk`: the boundary correction makes at most `maxk − 1` passes (default 5)
    pub maxk: usize,
    /// Largest `fibermetric` thickness in voxels; `None` is `msmv.m`'s default
    /// `max(15, 8·max(voxel_size))`. The thicknesses are `1:vessel_radius`.
    pub vessel_radius: Option<f64>,
}

impl Default for MsmvParams {
    fn default() -> Self {
        Self { prefilter: true, maxk: 5, vessel_radius: None }
    }
}

/// [`msmv`]'s output with the intermediate results, for parity checks and evaluation
#[derive(Clone, Debug)]
pub struct MsmvOutput {
    /// The filtered field (`msmv.m`'s return value), in the input's unit (radians)
    pub field: Vec<f64>,
    /// `t` from `kernel_lim.m`, radians
    pub threshold: f64,
    /// `Mv` on the boundary band `Me` (false elsewhere, see the module docs)
    pub vessel_mask: Vec<bool>,
    /// The boundary band `Me = Mask − (SMV(Mask) > 0.999)`, as `Me != 0`
    pub boundary: Vec<bool>,
    /// Boundary-correction passes made
    pub passes: usize,
}

/// `msmv.m`: mSMV-filter a local field (see the module docs).
///
/// # Arguments
/// * `rdf` - The local field in **radians over the echo spacing**, as MEDI's `RDF` (`nx·ny·nz`)
/// * `mask` - The brain mask the field is defined on (1 = brain), not eroded
/// * `r2s` - R2* in Hz (`arlo.m`), for the vessel mask
/// * `grid` - Dimensions and voxel sizes (mm); each axis needs at least 3 voxels
/// * `radius` - Prefilter / boundary radius in mm (MEDI's SMV radius, default 5)
/// * `b0` - `B0_mag` in tesla: sets `tmin = 0.01·b0/3` rad
/// * `params` - [`MsmvParams`]
pub fn msmv(
    rdf: &[f64],
    mask: &[u8],
    r2s: &[f64],
    grid: &Grid,
    radius: f64,
    b0: f64,
    params: &MsmvParams,
) -> MsmvOutput {
    let (nx, ny, nz) = grid.dims;
    let n = grid.n_total();
    assert_eq!(rdf.len(), n, "mSMV: field length does not match the grid");
    assert_eq!(mask.len(), n, "mSMV: mask length does not match the grid");
    assert_eq!(r2s.len(), n, "mSMV: R2* length does not match the grid");
    assert!(radius.is_finite() && radius > 0.0, "mSMV: radius must be finite and > 0 mm");
    assert!(b0.is_finite() && b0 > 0.0, "mSMV: field strength must be finite and > 0 T");
    let vs = [grid.vsx(), grid.vsy(), grid.vsz()];
    let maskf: Vec<f64> = mask.iter().map(|&m| f64::from(m)).collect();

    // r2 = min(voxel_size(:))/2+0.05; SphereK = single(sphere_kernel(matrix_size,voxel_size,radius))
    let r2 = vs.iter().cloned().fold(f64::INFINITY, f64::min) / 2.0 + 0.05;
    let sphere_k = sphere_kernel_single(grid, radius);

    // Mne = SMV(Mask,SphereK) > 0.999; Me = Mask-Mne
    let smv_mask = apply_real_kernel(&maskf, &sphere_k, nx, ny, nz);
    let me: Vec<f64> = (0..n).map(|i| maskf[i] - f64::from(u8::from(smv_mask[i] > 0.999))).collect();

    // RDF_s = Mask.*(RDF-SMV(RDF,SphereK)), or RDF itself without the prefilter
    let mut rdf_s: Vec<f64> = if params.prefilter {
        let s = apply_real_kernel(rdf, &sphere_k, nx, ny, nz);
        (0..n).map(|i| maskf[i] * (rdf[i] - s[i])).collect()
    } else {
        rdf.to_vec()
    };

    // t = kernel_lim(RDF,voxel_size,matrix_size,Mask,B0_mag)
    let t = kernel_lim(rdf, &maskf, grid, b0);

    // Mask_ev = Mask-MaskErode(Mask,matrix_size,voxel_size,radius+1)
    let erode_k = sphere_kernel_real(grid, radius + 1.0);
    let smv_ev = apply_real_kernel(&maskf, &erode_k, nx, ny, nz);
    let mask_ev: Vec<f64> = (0..n).map(|i| maskf[i] - f64::from(u8::from(smv_ev[i] > 0.999))).collect();

    // Mv = imbinarize(fibermetric(Mask_ev.*R2s, 1:vessel_radius, 'ObjectPolarity','bright'), 0),
    // needed only where Me is nonzero
    let vessel_radius = params
        .vessel_radius
        .unwrap_or_else(|| (8.0 * vs.iter().cloned().fold(0.0, f64::max)).max(15.0));
    let thickness: Vec<usize> = (1..=vessel_radius.floor() as usize).collect();
    let boundary: Vec<bool> = me.iter().map(|&v| v != 0.0).collect();
    let ev_r2s: Vec<f64> = (0..n).map(|i| mask_ev[i] * r2s[i]).collect();
    let vessel_mask: Vec<bool> = if thickness.is_empty() {
        vec![false; n]
    } else {
        fibermetric(&ev_r2s, grid.dims, &thickness, FIBERMETRIC_DEFAULT_C, FiberPolarity::Bright, Some(&boundary))
            .into_iter()
            .map(|v| v > 0.0)
            .collect()
    };

    // Mb = imbinarize(abs(Me.*RDF_s0),t); Mb = Mb & ~Mv
    let flagged = |f: &[f64]| -> Vec<bool> { (0..n).map(|i| (me[i] * f[i]).abs() > t && !vessel_mask[i]).collect() };
    let mut mb = flagged(&rdf_s);
    let mask_sum: f64 = maskf.iter().sum();
    let small = SparseKernel::new(grid, r2);
    let mut passes = 0usize;
    let mut k = 1usize;
    while mb.iter().filter(|&&b| b).count() as f64 / mask_sum > 1e-6 {
        mb = flagged(&rdf_s);
        // RDF_s = Mask.*(RDF_s-SMV(Mb.*RDF_s,matrix_size,voxel_size,r2))
        let masked: Vec<f64> = (0..n).map(|i| if mb[i] { rdf_s[i] } else { 0.0 }).collect();
        let s = small.apply(&masked, grid);
        for i in 0..n {
            rdf_s[i] = maskf[i] * (rdf_s[i] - s[i]);
        }
        passes += 1;
        k += 1;
        if k + 1 > params.maxk {
            break;
        }
    }

    MsmvOutput { field: rdf_s, threshold: t, vessel_mask, boundary, passes }
}

/// `kernel_lim.m`: the shadow threshold `t` in radians (see the module docs).
///
/// `mask` is the brain mask as 0/1. Where `kernel_lim.m` would loop forever (the residual can
/// never reach `tmin`) this returns `tmin`, which flags no voxel, as any value `kernel_lim.m`
/// could return would not; and the search stops once the sphere is larger than the volume.
pub fn kernel_lim(rdf: &[f64], mask: &[f64], grid: &Grid, b0: f64) -> f64 {
    let vs = [grid.vsx(), grid.vsy(), grid.vsz()];
    let tmin = 0.01 * b0 / 3.0;
    let mut k = 0.5 * vs.iter().cloned().fold(f64::INFINITY, f64::min);
    let mut t = 0.001;
    let eta = 1e-3;
    let extent = {
        let (nx, ny, nz) = grid.dims;
        ((nx as f64 * vs[0]).powi(2) + (ny as f64 * vs[1]).powi(2) + (nz as f64 * vs[2]).powi(2)).sqrt()
    };
    // |RDF − SMV(RDF)| <= 2 max|RDF| (the sphere averages with positive weights), so below
    // tmin/2 kernel_lim.m never returns. No voxel can exceed tmin then, so tmin itself is a
    // threshold that flags nothing, as any value kernel_lim.m could have returned would.
    let max_abs = rdf.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if t < tmin && 2.0 * max_abs < tmin {
        return tmin;
    }
    while t < tmin {
        // RDF_smv = Mask.*(RDF-SMV(RDF,matrix_size,voxel_size,K)); t = max(abs(RDF_smv(:)))
        let s = SparseKernel::new(grid, k).apply(rdf, grid);
        t = (0..rdf.len()).map(|i| (mask[i] * (rdf[i] - s[i])).abs()).fold(0.0, f64::max);
        k += eta;
        if k > extent {
            break;
        }
    }
    t
}

/// `SMV(x, matrix_size, voxel_size, r)` for a small `r`: MEDI's rendered sphere applied as a
/// circular convolution with `real(fftn(...))`'s kernel, i.e. the sphere symmetrised
/// (`(S(o) + S(−o))/2`), which is the sphere itself on even grids. Large spheres go through the
/// FFT.
enum SparseKernel {
    Direct(Vec<([isize; 3], f64)>),
    Fft(Vec<f64>),
}

impl SparseKernel {
    fn new(grid: &Grid, radius: f64) -> Self {
        let sparse = sphere_kernel_sparse(grid, radius);
        if sparse.len() > 512 {
            return SparseKernel::Fft(sphere_kernel_real(grid, radius));
        }
        let (nx, ny, nz) = grid.dims;
        let signed = |i: usize, n: usize| if 2 * i > n { i as isize - n as isize } else { i as isize };
        let mut taps: Vec<([isize; 3], f64)> = Vec::with_capacity(2 * sparse.len());
        for (p, v) in sparse {
            let (i, j, k) = (p % nx, (p / nx) % ny, p / (nx * ny));
            let o = [signed(i, nx), signed(j, ny), signed(k, nz)];
            taps.push((o, v / 2.0));
            taps.push(([-o[0], -o[1], -o[2]], v / 2.0));
        }
        SparseKernel::Direct(taps)
    }

    fn apply(&self, x: &[f64], grid: &Grid) -> Vec<f64> {
        let (nx, ny, nz) = grid.dims;
        match self {
            SparseKernel::Fft(k) => apply_real_kernel(x, k, nx, ny, nz),
            SparseKernel::Direct(taps) => {
                let wrap = |a: usize, o: isize, n: usize| (a as isize + o).rem_euclid(n as isize) as usize;
                let mut out = vec![0.0f64; x.len()];
                let fill = |(p, v): (usize, &mut f64)| {
                    let (i, j, k) = (p % nx, (p / nx) % ny, p / (nx * ny));
                    let mut acc = 0.0;
                    for &(o, w) in taps {
                        acc += w * x[wrap(i, o[0], nx) + nx * (wrap(j, o[1], ny) + ny * wrap(k, o[2], nz))];
                    }
                    *v = acc;
                };
                #[cfg(feature = "parallel")]
                out.par_iter_mut().enumerate().with_min_len(1 << 12).for_each(fill);
                #[cfg(not(feature = "parallel"))]
                out.iter_mut().enumerate().for_each(fill);
                out
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The direct convolution equals `real(ifftn(fftn(x).*K))` with the FFT kernel.
    #[test]
    fn direct_smv_matches_fft() {
        for dims in [(12usize, 10usize, 8usize), (11, 9, 7)] {
            let grid = Grid::new(dims.0, dims.1, dims.2, 1.0, 0.9, 1.3);
            let n = grid.n_total();
            let x: Vec<f64> = (0..n).map(|i| ((i * 37 % 101) as f64 - 50.0) * 0.01).collect();
            for r in [0.5, 0.55, 0.7, 1.2] {
                let direct = SparseKernel::new(&grid, r).apply(&x, &grid);
                let fft = apply_real_kernel(&x, &sphere_kernel_real(&grid, r), dims.0, dims.1, dims.2);
                let err = direct.iter().zip(&fft).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
                assert!(err < 1e-13, "dims {dims:?} r {r}: {err}");
            }
        }
    }

    /// At half a voxel the rendered sphere is the centre voxel alone, so kernel_lim's first
    /// residual is 0 and the threshold is the first residual at or above `tmin`.
    #[test]
    fn kernel_lim_reaches_tmin() {
        let grid = Grid::new(16, 16, 16, 1.0, 1.0, 1.0);
        let n = grid.n_total();
        let rdf: Vec<f64> = (0..n).map(|i| ((i % 16) as f64 * 0.7).sin() * 0.5).collect();
        let mask = vec![1.0; n];
        let t = kernel_lim(&rdf, &mask, &grid, 3.0);
        assert!((0.01..0.02).contains(&t), "t = {t}");
        let t7 = kernel_lim(&rdf, &mask, &grid, 7.0);
        assert!(t7 >= 0.01 * 7.0 / 3.0, "t7 = {t7}");
    }

    /// A field that can never reach tmin ends the search instead of looping.
    #[test]
    fn kernel_lim_terminates_on_small_field() {
        let grid = Grid::new(8, 8, 8, 2.0, 2.0, 2.0);
        let n = grid.n_total();
        assert_eq!(kernel_lim(&vec![0.0; n], &vec![1.0; n], &grid, 3.0), 0.01);
        assert_eq!(kernel_lim(&vec![1e-3; n], &vec![1.0; n], &grid, 3.0), 0.01);
    }

    fn ball(nn: usize, r: f64) -> (Grid, Vec<u8>) {
        let grid = Grid::new(nn, nn, nn, 1.0, 1.0, 1.0);
        let c = nn as f64 / 2.0;
        let mut mask = vec![0u8; grid.n_total()];
        for k in 0..nn {
            for j in 0..nn {
                for i in 0..nn {
                    let d = ((i as f64 - c).powi(2) + (j as f64 - c).powi(2) + (k as f64 - c).powi(2)).sqrt();
                    mask[i + nn * (j + nn * k)] = u8::from(d < r);
                }
            }
        }
        (grid, mask)
    }

    /// With the prefilter off and a field below the threshold everywhere, nothing is flagged and
    /// the field comes back unchanged (and unmasked, as in `msmv.m`).
    #[test]
    fn no_prefilter_below_threshold_is_identity() {
        let (grid, mask) = ball(24, 9.0);
        let n = grid.n_total();
        let rdf: Vec<f64> = (0..n).map(|i| 1e-4 * ((i % 24) as f64 * 0.3).sin()).collect();
        let r2s = vec![20.0; n];
        let p = MsmvParams { prefilter: false, ..MsmvParams::default() };
        let out = msmv(&rdf, &mask, &r2s, &grid, 4.0, 3.0, &p);
        assert_eq!(out.passes, 0);
        assert_eq!(out.field, rdf);
    }

    /// A large field on the boundary band is reduced, the mask is not eroded, and every voxel
    /// the correction touches is in the band.
    #[test]
    fn corrects_the_boundary_band() {
        let (grid, mask) = ball(32, 12.0);
        let n = grid.n_total();
        let c = 16.0;
        // A harmonic-like residual growing towards the edge: x^2 - y^2 in radians
        let rdf: Vec<f64> = (0..n)
            .map(|p| {
                let (i, j) = ((p % 32) as f64 - c, ((p / 32) % 32) as f64 - c);
                if mask[p] != 0 { 2e-3 * (i * i - j * j) } else { 0.0 }
            })
            .collect();
        let r2s = vec![0.0; n];
        let out = msmv(&rdf, &mask, &r2s, &grid, 4.0, 3.0, &MsmvParams::default());
        assert!(out.passes >= 1 && out.passes <= 4, "passes {}", out.passes);
        assert!(out.vessel_mask.iter().all(|&v| !v), "flat R2* has no vessels");
        let band_max = |f: &[f64]| (0..n).filter(|&i| out.boundary[i]).map(|i| f[i].abs()).fold(0.0, f64::max);
        // the prefiltered field before correction
        let maskf: Vec<f64> = mask.iter().map(|&m| f64::from(m)).collect();
        let s = apply_real_kernel(&rdf, &sphere_kernel_single(&grid, 4.0), 32, 32, 32);
        let pre: Vec<f64> = (0..n).map(|i| maskf[i] * (rdf[i] - s[i])).collect();
        assert!(band_max(&out.field) < band_max(&pre), "{} vs {}", band_max(&out.field), band_max(&pre));
        assert!((0..n).all(|i| mask[i] != 0 || out.field[i] == 0.0));
    }
}

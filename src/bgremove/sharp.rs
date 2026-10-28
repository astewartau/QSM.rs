//! SHARP background field removal
//!
//! Sophisticated Harmonic Artifact Reduction for Phase data (Schweser et al. 2011). A harmonic
//! function equals its spherical mean, so the high-pass `f − S∗f` (S a normalised sphere) has no
//! harmonic (background) part. SHARP takes it inside the mask eroded by the sphere, then undoes
//! the high-pass by deconvolving `1 − Ŝ`, with its small values truncated (TSVD).
//!
//! # Which implementation
//!
//! The SHARP authors (Schweser et al. 2011) published no code. This module follows Tian Liu's
//! `SHARP.m` (Cornell, the MEDI group; 2010, last modified 2011-02-01) as SEPIA distributes and
//! runs it: kschan0214/sepia `db939cc` (v1.3.0.1), `misc/background_removal/SHARP/SHARP.m`,
//! called by `wrapper/WrapperBFR/Wrapper_BFR_SHARP.m`, which puts the MEDI toolbox on the path for
//! `SMV.m`, `SMV_kernel.m` and `sphere_kernel.m`. Those are MEDI 2024.11.26's (byte-identical to
//! the copies in AMP-PE and other MEDI redistributions; they were last changed in 2013, so they are
//! also what SEPIA's README's MEDI of Jan 2020 has). The defaults are SEPIA's (below). Decided
//! 2026-10-28. QSM.jl's port, which this module followed before, is kept as
//! [`SharpVariant::QsmJl`] for the QSM.jl parity pair.
//!
//! `SHARP.m`, step by step, as SEPIA runs it (`BackgroundRemovalMacro.m` with the settings SEPIA's
//! GUI exports for SHARP):
//!
//! 0. SEPIA zero-pads every odd axis by one slice at its end (`zeropad_odd_dimension(…, 'pre')`,
//!    `padarray(…, 'post')`) and crops the result back afterwards. Without it MEDI's sphere, built
//!    on the grid `-N/2 : N/2-1` and `fftshift`ed, sits half a voxel off the origin on an odd
//!    axis, which shifts the output.
//! 1. `S` = `sphere_kernel(matrix_size, voxel_size, radius)`: a sphere of `radius` **mm**
//!    (coordinates are scaled by the voxel size), voxels wholly inside are 1, voxels the surface
//!    cuts get their volume fraction from 20³ sub-voxel samples, normalised to sum 1,
//!    `fftn(fftshift(Sphere))`.
//! 2. Erode: `Mask = SMV(Mask, matrix_size, voxel_size, radius) > 0.999`, i.e. keep the voxels
//!    whose sphere lies in the mask to 0.1 % of its weight. The convolution is periodic on the
//!    (padded) field of view.
//! 3. `kernel = SMV_kernel(…) = 1 − S`; `iFreq = ifftn(fftn(iFreq).*kernel)`, times `Mask`.
//! 4. `kRDF = fftn(·)./kernel`; `kRDF(abs(kernel) <= threshold) = 0`; NaN and Inf set to 0.
//! 5. `RDF = ifftn(kRDF).*Mask`. Every FFT is on the whole (padded) field of view: no cropping
//!    and no further padding.
//!
//! `SHARP.m` returns only the local field; this port also returns step 2's eroded mask (SEPIA's
//! own next step takes `imfill(RDF ≠ 0, 'holes')` as the mask for the dipole inversion).
//! `SHARP.m` does not take `real()`: MATLAB's `ifftn` returns a real array because its input is
//! exactly conjugate-symmetric (checked in licensed R2023b: `isreal(RDF)` on even, odd and
//! padded grids), so the real part taken here is the same thing.
//!
//! # Defaults
//!
//! `radius` 4 mm and `threshold` 0.03: SEPIA's GUI (`gui_func/panel/BFR_method/
//! sepia_handle_panel_bkgRemoval_SHARP.m`, `defaultRadius = 4; defaultthreshold = 0.03`), which is
//! what SEPIA's exported config and pipeline pass to `SHARP.m`. Confirmed by running SEPIA
//! headless in licensed MATLAB R2023b: the GUI, with SHARP selected, exports
//! `algorParam.bfr.radius = 4; algorParam.bfr.threshold = 0.03` (refine 'None', no extra erosion),
//! and with those `BackgroundRemovalMacro` calls `SHARP(…, 4, 0.03)` and its output is identical
//! to `SHARP.m` called directly on the padded data. Two traps in SEPIA:
//!
//! - The GUI labels the radius "SMV radius (voxel)" and the wrapper prints `Radius(voxel)`, but
//!   the number is passed unchanged to `sphere_kernel`, which compares `(X·voxel_size)²` with
//!   `radius²`: it is in **mm**. QSM.rs's `radius` is in mm too, so SEPIA's 4 is 4 mm here, on any
//!   voxel size.
//! - `Wrapper_BFR_SHARP.m`'s fallbacks for missing fields are `radius = 0.1` and
//!   `threshold = 30` (swapped, and unusable: no voxel survives a 0.1 mm sphere's erosion). They
//!   apply only when the config lacks the fields, which the GUI never writes.
//!
//! `SHARP.m`'s own defaults, used by nobody here, are `threshold = 0` (no truncation, only the
//! NaN/Inf guard) and `radius = round(6/max(voxel_size))·max(voxel_size)` (6 mm at 1 mm).
//!
//! Other implementations, for the record:
//!
//! | implementation | radius | TSVD threshold |
//! |---|---|---|
//! | SEPIA + Tian Liu's `SHARP.m` (followed) | 4 mm (GUI) | 0.03 (GUI) |
//! | QSM.jl `sharp` (kamesy/QSM.jl fde4b92; [`SharpVariant::QsmJl`]) | `18·min(vsz)` mm | 0.05 |
//! | QSM.m `sharp.m` (kamesy/QSM.m f22dc40, `src/bgremove/sharp.m:13-15,33-34`) | 9 mm | 0.05 |
//! | Hongfu Sun's `sharp.m` (sunhongfu/QSM c75b138, `background_field_removal/sharp.m`) | 4 mm if omitted, 5 mm in its header's example; his pipelines pass 2–5 mm, mostly 3 | none (required); header example 0.05; pipelines 0.05–0.1, mostly 0.1 |
//!
//! QSM.jl's 18 mm is not used: SHARP erodes the mask by its radius, so an 18 mm sphere removes
//! most of the cortex.
//!
//! # Tian Liu's `SHARP.m` and Hongfu Sun's `sharp.m`
//!
//! | | Liu `SHARP.m` (+ MEDI) | Sun `sharp.m` |
//! |---|---|---|
//! | radius units | mm | mm, rounded to whole voxels per axis: `r_i = max(1, round(r/vox_i))` |
//! | kernel | sphere of radius r mm, partial-volume edge voxels (20³ sub-samples), normalised | binary ellipsoid `Σ x_i²/r_i² ≤ 1` in voxels, normalised; no partial volume |
//! | erosion | `S∗mask > 0.999`, periodic (FFT) convolution | `convn(mask, ker, 'same') > 1 − 1/|ball|` (the whole ball fits), linear convolution, zero outside the volume |
//! | TSVD | zero where `|1 − Ŝ| ≤ thr`; NaN/Inf zeroed | zero where `|1 − Ŝ| < thr` |
//! | FFT domain | the whole volume, odd axes padded to even by SEPIA | the whole volume, no padding (the kernel is placed at the corner and the result `circshift`ed back, so the ball stays centred on odd axes) |
//! | output | `ifftn(·).*Mask` (real) | `real(ifftn(·)).*mask_ero`, plus the eroded mask |
//! | defaults | `thr` 0, `r` 6 mm rounded to the coarsest voxel; SEPIA passes 4 mm, 0.03 | `r` 4 mm; `thr` none |
//!
//! Neither is clearly better reasoned. Liu's kernel is the more careful discretisation: a true
//! sphere in mm on anisotropic voxels, with partial-volume weights, so the kernel varies smoothly
//! with the radius. Sun's rounds the radius per axis, so on anisotropic or non-integer radii the
//! "sphere" is an ellipsoid of whole voxels, and the radius changes the kernel only in steps.
//! Sun's erosion is the stricter and the better defined (the ball must fit; no wrap-around at the
//! field of view's edge), where Liu's 0.999 is an arbitrary tolerance and its periodic convolution
//! can let a mask touching the volume's edge see the opposite side. Liu's code needs SEPIA's
//! padding to be right on odd axes; Sun's is right on any size. The TSVD rules differ only in
//! `<` against `≤`. These are judgement calls, not errors, on either side.
//!
//! # Agreement with `SHARP.m`
//!
//! `tests/sharp_sepia_parity.rs` checks this port against SEPIA's output in licensed MATLAB R2023b
//! (`tests/data/sharp_sepia_small.m`, the generator, run in SEPIA's `BackgroundRemovalMacro`) on
//! small phantoms: at SEPIA's defaults on a 1 mm grid, at SEPIA's defaults on an anisotropic grid
//! with odd dimensions, and at 2.5 mm / 0.1 on another anisotropic grid; the eroded masks are
//! identical and the fields agree to float rounding.
//!
//! On QSM-CI's fidelity cases, against the licensed-MATLAB run of the QSM-CI `sharp-sepia`
//! container's `recon.m` (QSMxT/QSM-CI `algorithms/sharp-sepia`, the same SEPIA path; the
//! container is bit-identical to it), both in float64, at SEPIA's defaults, with identical eroded
//! masks: the in-silico head (QSM Challenge 2.0 simulation, 1 mm, 164×205×205, so two odd axes
//! padded; ground-truth total field) NRMSE 2.8e-11 %, max |Δ| 1.3e-14 ppm; the in-vivo
//! travelling head (Siemens Prisma, 1 mm, 184×256×128, ROMEO total field) NRMSE 1.2e-11 %, max
//! |Δ| 1.4e-14 ppm.
//!
//! # [`SharpVariant::QsmJl`]: QSM.jl's `sharp`
//!
//! A port of QSM.jl's `sharp` (kamesy/QSM.jl fde4b92, `src/bgremove/sharp.jl`), kept for the
//! QSM.jl parity pair ([`SharpParams::qsmjl`] gives its defaults). Line by line:
//!
//! 1. Crop the field and the mask to the mask's bounding box (`crop_indices`) and zero-pad them
//!    to `fastfftsize(box, (p, p, p); rfft = true)` with `p = ⌈r / min(vsz)⌉`
//!    ([`crate::utils::padding::qsmjl_fast_fft_size`]); every FFT below is
//!    periodic on that padded box.
//! 2. `S` = the FFT of the normalised ball of voxel centres within `r` mm (`_smv_kernel!`).
//! 3. Erode the mask to the voxels where `S ∗ mask > 1 − √eps` (eps of `Float64`), i.e. where
//!    the whole ball fits in the mask.
//! 4. High-pass `(1 − S)·F(f)`, back to image space, multiply by the eroded mask.
//! 5. Deconvolve: `F⁻¹[F(·)/(1 − S)]` with the frequencies where `|1 − S| < thr` set to 0
//!    (truncated SVD), and multiply by the eroded mask again.
//!
//! QSM.jl uses real-to-complex FFTs, this port complex ones; otherwise every step is QSM.jl's. On
//! the QSM-CI in-vivo travelling-head scan (Siemens Prisma 2.89 T, 1 mm, 184×256×128, ROMEO total
//! field and brain mask), QSM.jl's `sharp` in the QSM-CI `julia-qsm` image (`sha256:b3c048ac…`)
//! and this variant with the same `r` and `thr` agree to float rounding (both outputs saved as
//! float32), with identical eroded masks: at `r` 6 mm NRMSE 6.6e-10 %, max |Δ| 1.2e-10 ppm; at
//! QSM.jl's default 18 mm NRMSE 7.8e-12 %, max |Δ| 9.1e-13 ppm. `tests/sharp_qsmjl_parity.rs`
//! checks it against QSM.jl's output on small phantoms. V-SHARP's filtering
//! ([`super::vsharp()`]) shares this variant's box and edge test.
//!
//! Reference:
//! Schweser, F., Deistung, A., Lehr, B.W., Reichenbach, J.R. (2011).
//! "Quantitative imaging of intrinsic magnetic tissue properties using MRI signal phase."
//! NeuroImage, 54(4):2789-2807. <https://doi.org/10.1016/j.neuroimage.2010.10.070>
//!
//! Reference implementation: <https://github.com/kschan0214/sepia> (`SHARP.m`, db939cc) with the
//! MEDI toolbox (<http://pre.weill.cornell.edu/mri/pages/qsm.html>, 2024.11.26)

use num_complex::Complex64;
use crate::Grid;
use crate::fft::{fft3d, ifft3d, fft_real_kernel, Fft3dWorkspace};
use crate::kernels::smv::{smv_kernel, erode_mask_smv};
use crate::utils::padding::qsmjl_fast_fft_size;

/// QSM.jl's threshold for "the whole ball fits in the mask": `S ∗ mask > 1 − √eps(Float64)`
/// (`δ` in `src/bgremove/sharp.jl` `_sharp!`, fde4b92). Shared by [`SharpVariant::QsmJl`] and
/// V-SHARP.
pub(crate) const QSMJL_SMV_DELTA: f64 = 1.0 - 1.490_116_119_384_765_6e-8; // 1 - sqrt(f64::EPSILON)

/// `SHARP.m`'s erosion threshold: `Mask = SMV(Mask, …, radius) > 0.999`.
const SEPIA_SMV_DELTA: f64 = 0.999;

/// Which implementation of SHARP to reproduce. See the [module docs](self).
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SharpVariant {
    /// Tian Liu's `SHARP.m` with MEDI's kernels, as SEPIA runs it (odd axes padded to even):
    /// the reference.
    #[default]
    Sepia,
    /// QSM.jl's `sharp` (fde4b92): its own ball, `1 − √eps` edge test, `<` TSVD rule, and the
    /// mask's bounding box padded to a fast FFT size. For the QSM.jl parity pair.
    QsmJl,
}

/// SHARP parameters. The defaults are SEPIA's (see the [module docs](self#defaults)).
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct SharpParams {
    /// Deconvolution (TSVD) threshold (default 0.03, SEPIA's GUI). [`SharpVariant::Sepia`] sets
    /// the frequencies where `|1 − Ŝ| ≤ threshold` to 0 (`SHARP.m`), [`SharpVariant::QsmJl`]
    /// those where `|1 − Ŝ| < threshold`.
    pub threshold: f64,
    /// Sphere radius in mm (default 4, SEPIA's GUI, whose "voxel" label is wrong: `SHARP.m`
    /// takes mm). Before 2026-10-28 this followed QSM.jl (`18 · min(voxel size)`), and before
    /// that defaulted to 6 mm, which had no reference.
    pub radius: f64,
    /// The implementation to reproduce (default [`SharpVariant::Sepia`]).
    pub variant: SharpVariant,
}

impl Default for SharpParams {
    fn default() -> Self {
        Self {
            threshold: 0.03,
            radius: 4.0,
            variant: SharpVariant::Sepia,
        }
    }
}

impl SharpParams {
    /// QSM.jl `sharp`'s default kernel radius as a multiple of the smallest voxel dimension.
    pub const QSMJL_RADIUS_VOXELS: f64 = 18.0;

    /// QSM.jl `sharp` at its own defaults on `grid`: [`SharpVariant::QsmJl`], `r` =
    /// `18 · min(voxel size)` mm, `thr` 0.05.
    pub fn qsmjl(grid: &Grid) -> Self {
        let (a, b, c) = grid.voxel_size;
        Self {
            threshold: 0.05,
            radius: Self::QSMJL_RADIUS_VOXELS * a.min(b).min(c),
            variant: SharpVariant::QsmJl,
        }
    }
}

/// SHARP background field removal (see the [module docs](self))
///
/// # Arguments
/// * `field` - Unwrapped total field (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz), 1 = inside ROI
/// * `grid` - Volume dimensions and voxel sizes
/// * `params` - SHARP parameters (threshold, radius in mm, variant)
///
/// # Returns
/// (local_field, eroded_mask)
pub fn sharp(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    params: &SharpParams,
) -> (Vec<f64>, Vec<u8>) {
    match params.variant {
        SharpVariant::Sepia => sharp_sepia(field, mask, grid, params.threshold, params.radius),
        SharpVariant::QsmJl => sharp_core(field, mask, grid, params.threshold, params.radius),
    }
}

/// Tian Liu's `SHARP.m` as SEPIA runs it (steps 0-5 of the module docs).
fn sharp_sepia(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    threshold: f64,
    radius: f64,
) -> (Vec<f64>, Vec<u8>) {
    let (nx, ny, nz) = grid.dims;
    // 0. SEPIA's zeropad_odd_dimension(…, 'pre'): one zero slice at the end of each odd axis.
    let (px, py, pz) = (nx + nx % 2, ny + ny % 2, nz + nz % 2);
    let pgrid = Grid::new(px, py, pz, grid.vsx(), grid.vsy(), grid.vsz());
    let n = px * py * pz;
    let at = |i: usize, j: usize, k: usize| i + px * (j + py * k);
    let pad = |x: &dyn Fn(usize) -> f64| {
        let mut out = vec![Complex64::new(0.0, 0.0); n];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    out[at(i, j, k)] = Complex64::new(x(i + nx * (j + ny * k)), 0.0);
                }
            }
        }
        out
    };
    let mut fft = Fft3dWorkspace::new(px, py, pz);

    // 1. S = sphere_kernel(matrix_size, voxel_size, radius) = fftn(fftshift(Sphere))
    let mut s = vec![Complex64::new(0.0, 0.0); n];
    for (p, v) in crate::inversion::medi::sphere_kernel_sparse(&pgrid, radius) {
        s[p] = Complex64::new(v, 0.0);
    }
    fft.fft3d(&mut s);

    // 2. Mask = SMV(Mask, matrix_size, voxel_size, radius) > 0.999 (MATLAB's > compares real parts)
    let mut m = pad(&|l| mask[l] as f64);
    fft.fft3d(&mut m);
    for (x, k) in m.iter_mut().zip(&s) {
        *x *= k;
    }
    fft.ifft3d(&mut m);
    let eroded: Vec<bool> = m.iter().map(|x| x.re > SEPIA_SMV_DELTA).collect();
    drop(m);

    // 3. kernel = 1 - S; iFreq = ifftn(fftn(iFreq).*kernel).*Mask
    let mut f = pad(&|l| field[l]);
    fft.fft3d(&mut f);
    for (x, k) in f.iter_mut().zip(&s) {
        *x *= Complex64::new(1.0, 0.0) - k;
    }
    fft.ifft3d(&mut f);
    for (x, &e) in f.iter_mut().zip(&eroded) {
        if !e {
            *x = Complex64::new(0.0, 0.0);
        }
    }

    // 4. kRDF = fftn(.)./kernel; kRDF(abs(kernel) <= threshold) = 0; NaN and Inf -> 0
    fft.fft3d(&mut f);
    for (x, k) in f.iter_mut().zip(&s) {
        let kernel = Complex64::new(1.0, 0.0) - k;
        let q = *x / kernel;
        *x = if kernel.norm() <= threshold || !q.is_finite() { Complex64::new(0.0, 0.0) } else { q };
    }

    // 5. RDF = ifftn(kRDF).*Mask (real: see the module docs), cropped back
    fft.ifft3d(&mut f);
    let mut local = vec![0.0; nx * ny * nz];
    let mut out_mask = vec![0u8; nx * ny * nz];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let p = at(i, j, k);
                if eroded[p] {
                    let l = i + nx * (j + ny * k);
                    local[l] = f[p].re;
                    out_mask[l] = 1;
                }
            }
        }
    }
    (local, out_mask)
}

/// The box QSM.jl filters in (`_sharp!`, `src/bgremove/sharp.jl`, fde4b92): the mask's bounding
/// box (`crop_indices(mask)`), zero-padded to `fastfftsize(box, (p, p, p); rfft = true)` with
/// `p = ⌈r_max / min(vsz)⌉` (`padfastfft`). QSM.jl centres the box in the padding; here it sits
/// at the origin, which is the same for periodic convolution, masking and the inverse filter.
pub(crate) struct SmvBox {
    lo: [usize; 3],
    n: [usize; 3],
    /// Padded box dimensions.
    pub np: (usize, usize, usize),
    dims: (usize, usize, usize),
}

impl SmvBox {
    /// The box for `mask` on `grid` and the largest kernel radius `r_max` (mm); `None` for an
    /// empty mask.
    pub fn new(mask: &[u8], grid: &Grid, r_max: f64) -> Option<Self> {
        let (nx, ny, nz) = grid.dims;
        let mut lo = [usize::MAX; 3];
        let mut hi = [0usize; 3];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    if mask[i + j * nx + k * nx * ny] != 0 {
                        for (d, v) in [i, j, k].into_iter().enumerate() {
                            lo[d] = lo[d].min(v);
                            hi[d] = hi[d].max(v);
                        }
                    }
                }
            }
        }
        if lo[0] == usize::MAX {
            return None;
        }
        let n = [0, 1, 2].map(|d| hi[d] - lo[d] + 1);
        let (a, b, c) = grid.voxel_size;
        // QSM.jl: pad = Int(cld(r, minimum(vsz)))
        let p = (r_max / a.min(b).min(c)).ceil() as i64;
        let [px, py, pz] = qsmjl_fast_fft_size(n, [p; 3]);
        Some(Self { lo, n, np: (px, py, pz), dims: grid.dims })
    }

    /// The padded box's grid (same voxel size as `grid`).
    pub fn grid(&self, grid: &Grid) -> Grid {
        Grid::new(self.np.0, self.np.1, self.np.2, grid.vsx(), grid.vsy(), grid.vsz())
    }

    /// `x` cropped to the box and zero-padded.
    pub fn crop<T: Copy + Default>(&self, x: &[T]) -> Vec<T> {
        let (nx, ny, _) = self.dims;
        let (px, py, pz) = self.np;
        let mut out = vec![T::default(); px * py * pz];
        for k in 0..self.n[2] {
            for j in 0..self.n[1] {
                let src = self.lo[0] + (j + self.lo[1]) * nx + (k + self.lo[2]) * nx * ny;
                let dst = j * px + k * px * py;
                out[dst..dst + self.n[0]].copy_from_slice(&x[src..src + self.n[0]]);
            }
        }
        out
    }

    /// The box of the padded `xp` put back on the full grid, zero outside it.
    pub fn uncrop<T: Copy + Default>(&self, xp: &[T]) -> Vec<T> {
        let (nx, ny, nz) = self.dims;
        let (px, py, _) = self.np;
        let mut out = vec![T::default(); nx * ny * nz];
        for k in 0..self.n[2] {
            for j in 0..self.n[1] {
                let dst = self.lo[0] + (j + self.lo[1]) * nx + (k + self.lo[2]) * nx * ny;
                let src = j * px + k * px * py;
                out[dst..dst + self.n[0]].copy_from_slice(&xp[src..src + self.n[0]]);
            }
        }
        out
    }
}

/// QSM.jl's SHARP ([`SharpVariant::QsmJl`]) with an explicit kernel radius (mm).
///
/// Internal entry point shared with V-SHARP; the public [`sharp`] wrapper
/// dispatches to it for [`SharpVariant::QsmJl`].
pub(crate) fn sharp_core(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    threshold: f64,
    radius: f64,
) -> (Vec<f64>, Vec<u8>) {
    let Some(bx) = SmvBox::new(mask, grid, radius) else {
        return (vec![0.0; field.len()], vec![0; mask.len()]);
    };
    let (local, eroded) = sharp_padded(&bx.crop(field), &bx.crop(mask), &bx.grid(grid), threshold, radius);
    (bx.uncrop(&local), bx.uncrop(&eroded))
}

/// SHARP on QSM.jl's padded box (steps 2-5 of the module docs).
fn sharp_padded(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    threshold: f64,
    radius: f64,
) -> (Vec<f64>, Vec<u8>) {
    let (nx, ny, nz) = grid.dims;
    let n_total = nx * ny * nz;

    // Generate SMV kernel and FFT it
    let s_kernel = smv_kernel(grid, radius);
    let s_fft = fft_real_kernel(&s_kernel, nx, ny, nz);

    // Erode mask via SMV convolution
    let eroded_mask = erode_mask_smv(mask, &s_fft, grid, QSMJL_SMV_DELTA);

    // FFT of field
    let mut field_complex: Vec<Complex64> = field.iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();
    fft3d(&mut field_complex, nx, ny, nz);

    // High-pass filter: multiply by (1-S)
    for i in 0..n_total {
        field_complex[i] *= 1.0 - s_fft[i];
    }

    // IFFT
    ifft3d(&mut field_complex, nx, ny, nz);

    // Apply eroded mask
    for i in 0..n_total {
        if eroded_mask[i] == 0 {
            field_complex[i] = Complex64::new(0.0, 0.0);
        }
    }

    // FFT again for deconvolution
    fft3d(&mut field_complex, nx, ny, nz);

    // Deconvolution: divide by (1-S) with threshold
    for i in 0..n_total {
        let one_minus_s = 1.0 - s_fft[i];
        if one_minus_s.abs() < threshold {
            field_complex[i] = Complex64::new(0.0, 0.0);
        } else {
            field_complex[i] /= one_minus_s;
        }
    }

    // Final IFFT
    ifft3d(&mut field_complex, nx, ny, nz);

    // Apply eroded mask and extract real part
    let local_field: Vec<f64> = field_complex.iter()
        .enumerate()
        .map(|(i, c)| if eroded_mask[i] == 1 { c.re } else { 0.0 })
        .collect();

    (local_field, eroded_mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(variant: SharpVariant) -> SharpParams {
        SharpParams { threshold: 0.05, radius: 2.0, variant }
    }

    #[test]
    fn test_sharp_zero_field() {
        // Zero field should give zero local field
        let n = 16;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        for variant in [SharpVariant::Sepia, SharpVariant::QsmJl] {
            let (local, _) = sharp(&field, &mask, &grid, &params(variant));
            for &val in local.iter() {
                assert!(val.abs() < 1e-8, "{variant:?}: zero field should give zero local field, got {val}");
            }
        }
    }

    #[test]
    fn test_sharp_finite() {
        // Result should be finite (no NaN or Inf), on an odd grid too (SEPIA pads it)
        let (nx, ny, nz) = (16, 15, 13);
        let field: Vec<f64> = (0..nx * ny * nz).map(|i| (i as f64) * 0.01).collect();
        let mut mask = vec![0u8; nx * ny * nz];
        for k in 2..nz - 2 {
            for j in 2..ny - 2 {
                for i in 2..nx - 2 {
                    mask[i + nx * (j + ny * k)] = 1;
                }
            }
        }
        let grid = Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);

        for variant in [SharpVariant::Sepia, SharpVariant::QsmJl] {
            let (local, eroded) = sharp(&field, &mask, &grid, &params(variant));
            assert_eq!(local.len(), field.len());
            for (i, &val) in local.iter().enumerate() {
                assert!(val.is_finite(), "{variant:?}: local field should be finite at index {i}");
                assert!(eroded[i] == 1 || val == 0.0, "{variant:?}: field outside the eroded mask at {i}");
            }
            let eroded_count: usize = eroded.iter().map(|&m| m as usize).sum();
            assert!(eroded_count > 0, "{variant:?}: eroded mask should have some voxels");
            assert!((0..mask.len()).all(|i| eroded[i] <= mask[i]), "{variant:?}: eroded mask outside the mask");
        }
    }

    #[test]
    fn test_sharp_defaults_are_sepias() {
        let p = SharpParams::default();
        assert_eq!((p.radius, p.threshold, p.variant), (4.0, 0.03, SharpVariant::Sepia));
        let q = SharpParams::qsmjl(&Grid::new(8, 8, 8, 0.8, 1.0, 1.2));
        assert_eq!((q.threshold, q.variant), (0.05, SharpVariant::QsmJl));
        assert!((q.radius - 14.4).abs() < 1e-12);
    }
}

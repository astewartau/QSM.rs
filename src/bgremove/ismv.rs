//! Iterative Spherical Mean Value (iSMV) background field removal
//!
//! A harmonic (background) field equals its own spherical mean. iSMV holds the field fixed on a
//! thin band at the edge of the mask and, inside the mask eroded by the sphere, repeatedly
//! replaces the field by its spherical mean (SMV), `F ← SMV(F)·M + (mask − M)·field`. That
//! converges to the harmonic field matching the edge band, i.e. the background; the local field
//! is `(field − F)·M`. No deconvolution follows, so the sphere can be small and the mask is
//! eroded less than by SHARP.
//!
//! # Which implementation (decided 2026-10-29, astewartau/QSM.rs#322)
//!
//! The default, [`IsmvVariant::Wen2014`], is **the authors' paper and code together**:
//!
//! - **Code:** the iSMV authors' lab (Cornell, Yi Wang's group) ships `iSMV.m` with its kernel
//!   `tdcsmv.m` in github.com/agr78/mSMV (42a1f18, `code/dependencies/MEDI_functions/`). This
//!   module ports its loop, kernel and erosion line by line (see *`iSMV.m`, step by step*). Nothing
//!   in that repository (or in MEDI 2024.11.26, which ships no iSMV) calls it, so the code fixes
//!   no radius, and its hard-coded tolerance, 0.001, under-converges (below).
//! - **Paper:** Wen, Zhou, Liu, Spincemaille, Wang 2014 (*Methods*, PMC4026360) stop "when the
//!   relative change between successive iterates falls below 5 × 10⁻⁵" and compare radii of 1
//!   and 6 voxels, recommending neither ("much less dependent on the choice of radius"). So the
//!   defaults are the paper's: `tol` 5 × 10⁻⁵ on `iSMV.m`'s relative change, and a radius of
//!   **6 voxels**. The paper's other radius, 1 voxel, degenerates with `tdcsmv.m`: its in-plane
//!   loop runs over `-(n−1):(n−1)` with `n = ceil(r/vs₁) = 1`, so the "sphere" is a 3-voxel
//!   column along the third axis.
//! - **6 voxels in mm** is `6 × the smallest voxel dimension` ([`IsmvParams::radius_mm`]): the
//!   paper's in-vivo voxels were 0.94 × 0.94 × 2 mm, and `tdcsmv.m` counts the sphere's
//!   in-plane extent in units of `voxel_size(1)`, the in-plane (smallest) voxel size. On
//!   isotropic voxels the two readings agree. `iSMV.m` itself takes the radius in mm.
//! - **No iteration cap** by default ([`IsmvParams::max_iter`] `None`), as in `iSMV.m` (the paper
//!   reports none). The iteration is a contraction on the eroded mask (the SMV of every voxel of
//!   `M` reaches the fixed edge band or the zero outside the mask), so it always terminates for
//!   `tol > 0`; a cap is available.
//!
//! QSM.jl's `ismv`, which this module followed from 2026-10-28 (the designated implementation of
//! Kames's SMV methods when the authors have no code), is kept as [`IsmvVariant::QsmJl`] for the
//! QSM.jl parity pair ([`IsmvParams::qsmjl`]); see *QSM.jl's `ismv`*.
//!
//! ## Why the paper's tolerance (measured, 2026-10-28)
//!
//! The stopping tolerance matters more than the radius. At `tol` 1e-3 (`iSMV.m` as shipped, or
//! QSM.jl's default) the iteration stops after 9–15 iterations and leaves background in the
//! field, which shrinks deep-grey-matter contrast about threefold in vivo. CI phantom
//! (qsm-forward, 1 mm; local field scored over the brain mask as CI scores it, and over the
//! 941,844 voxels every setting keeps; RTS χ over those voxels) and the QSM-CI travelling heads
//! (10 acquisitions of 4 subjects, ROMEO field, SynthSeg regions; |slope − 1| of each scan's
//! regional χ against the subject's mean across scanners):
//!
//! | setting | phantom field, CI scoring: NRMSE % / r | common voxels: NRMSE % / r | RTS χ: NRMSE % / r | in vivo median \|slope−1\| | pallidum χ (ppb) |
//! |---|---|---|---|---|---|
//! | QSM.jl: 2 × max voxel, 1e-3 of the first residual, 500 | 60.9 / 0.800 | 53.3 / 0.853 | 68.4 / 0.740 | 0.111 | 41 |
//! | `iSMV.m` as shipped, 2 voxels, 0.001 | 63.0 / 0.778 | 53.7 / 0.847 | 68.9 / 0.731 | 0.106 | 44 |
//! | `iSMV.m` as shipped, 6 voxels, 0.001 | 57.9 / 0.817 | 36.2 / 0.934 | 59.9 / 0.806 | 0.055 | 97 |
//! | **`iSMV.m`, 6 voxels, 5e-5 (Wen 2014; the default)** | 53.0 / 0.848 | **25.2 / 0.968** | 53.6 / 0.848 | 0.036 | 129 |
//! | QSM.m `ismv.m`: 3 mm, √eps of the first residual, 500 | 48.1 / 0.878 | 27.4 / 0.963 | 51.2 / 0.862 | 0.038 | 134 |
//! | QSM.rs before 2026-10-28: 5 mm, 1e-6, 50 (no reference) | 51.4 / 0.858 | 26.2 / 0.966 | 54.2 / 0.845 | 0.024 | 124 |
//!
//! The default took 94 iterations on the phantom. Its cost is coverage: a 6-voxel sphere keeps
//! fewer edge voxels (in vivo, on average 1.06 M brain voxels, against 1.25 M at 3 mm and 1.33 M
//! at 2 voxels).
//!
//! ## `iSMV.m`, step by step
//!
//! 0. **Odd sizes (QSM.rs's fix).** Every odd axis is zero-padded by one slice at its end and the
//!    result cropped back, as SEPIA does before its FFT-based background removal
//!    (`zeropad_odd_dimension(…, 'pre')`, kschan0214/sepia db939cc; see [`super::sharp`](mod@super::sharp)).
//!    `tdcsmv.m` places the kernel at `f(end/2+1+k, …)`, which is not an integer index on an
//!    odd axis, so `iSMV.m` errors there; padding is the unequivocal fix, and on even sizes it is
//!    a no-op.
//! 1. **Kernel** `a = tdcsmv(matrix_size, voxel_size, radius)`: a sphere of `radius` mm with
//!    partial-volume weights from sub-voxel samples ([`tdcsmv`]), normalised to sum 1,
//!    `fftn(fftshift(SMV))`.
//! 2. **Erosion** `M = ifftn(fftn(Mask).*a) > .999` (MATLAB's `>` compares real parts), periodic
//!    on the (padded) array. `BV = (Mask − M).*iFreq`.
//! 3. **Loop** from `F1 = iFreq` (the whole input array, also outside the mask, as `iSMV.m`):
//!    `F2 = ifftn(fftn(F1).*a).*M + BV`; `d = norm(F1(:)−F2(:))/norm(F1(:))`; `F1 = F2`; while
//!    `d ≥ tol` (`iSMV.m`: `while (d>=.001)`; here [`IsmvParams::tol`], default 5e-5). The
//!    iterates are kept complex, as MATLAB's are (their imaginary part is rounding, ~1e-17).
//!    A NaN `d` (an all-zero input) stops the loop, as in MATLAB.
//! 4. **Output** `L = (iFreq − F2).*M` (its real part; `iSMV.m` returns the complex array) and
//!    `M`. The local field is zero outside the eroded mask `M`, not only outside the input mask.
//!
//! `tdcsmv.m` has two more places where it cannot run as written; both changes leave every
//! input it handles unchanged:
//!
//! - It uses `voxel_size(1)` for the second axis too (sub-voxel grid, offsets and extent), so on
//!   voxels with `vs₁ ≠ vs₂` its sphere has the wrong size along that axis. Here the second axis
//!   uses `voxel_size(2)`; with `vs₁ = vs₂` (every case `tdcsmv.m` is right for) nothing
//!   changes.
//! - It picks the sub-voxel factor `w` as the smallest integer with `vs₁·w` integral in single
//!   precision (`while (int16(g*w)~=single(g*w))`), unless `vs₁ = vs₃` (then 10), then doubles
//!   it. For a voxel size with no such `w` (a header value like 0.93750006) that loop never ends
//!   in MATLAB (`int16` saturates at 32767). Here the search stops at
//!   [`TDCSMV_MAX_SUBVOXEL`] and falls back to `w = 10`, `tdcsmv.m`'s own isotropic choice.
//!
//! A sphere that does not fit the array (MATLAB would grow `f` and then fail on the size
//! mismatch) wraps around it periodically.
//!
//! # QSM.jl's `ismv` ([`IsmvVariant::QsmJl`])
//!
//! A port of QSM.jl's `ismv` (kamesy/QSM.jl fde4b92, `src/bgremove/ismv.jl`), with its defaults
//! from [`IsmvParams::qsmjl`]: radius `r = 2·max(vsz)` mm, stop when the residual `‖fᵢ − fᵢ₋₁‖`
//! falls to `tol = 1e-3` of the first one, at most `maxit = 500` iterations. Its ball is the
//! voxel centres within `r` (no partial volume), its erosion `S ∗ mask > 1 − √eps(Float64)`, and
//! it returns `mask·(field − f)` over the input mask with the eroded mask.
//!
//! QSM.jl crops to the mask's bounding box and zero-pads to a fast FFT size before filtering;
//! this port filters on the whole array, periodically. iSMV only applies the SMV kernel
//! forwards, at voxels whose ball lies inside the mask, so the two are the same unless the mask
//! reaches two opposite faces of the array (then a ball can wrap around here and count as
//! fitting). QSM.jl's edge test is `S ∗ mask > 1 − √eps(Float64)`, this port's `> 1 − 1e-10`:
//! the same for any ball of fewer than 6·10⁷ voxels.
//!
//! On the QSM-CI in-vivo travelling-head scan (Siemens Prisma 2.89 T, 1 mm, 184×256×128, ROMEO
//! total field and brain mask), QSM.jl's `ismv` in the QSM-CI `julia-qsm` image
//! (`sha256:b3c048ac…`, QSM.jl fde4b92) and this port agree to float rounding (both outputs
//! saved as float32), with identical masks: at QSM.jl's defaults NRMSE 1.4e-9 %, max |Δ|
//! 2.3e-10 ppm; at radius 5 mm, `tol` 1e-6, 50 iterations NRMSE 3.1e-9 %, max |Δ| 4.7e-10 ppm.
//!
//! # Other implementations, for the record
//!
//! | implementation | radius | stopping rule | cap |
//! |---|---|---|---|
//! | Wen et al. 2014 + lab `iSMV.m` (the default, [`IsmvVariant::Wen2014`]) | 6 voxels (paper; also 1) | relative change < 5e-5 (paper); `iSMV.m`: < 0.001 | none |
//! | QSM.jl `ismv` (fde4b92; [`IsmvVariant::QsmJl`]) | `2·max(vsz)` mm | residual < 1e-3 of the first | 500 |
//! | QSM.m `ismv.m` (kamesy/QSM.m f22dc40, `src/bgremove/ismv.m:12-17,37-39`) | 3 mm | residual < √eps of the first | 500 |
//!
//! SEPIA does not offer iSMV.
//!
//! # Tests
//!
//! `tests/ismv_wen2014_parity.rs` checks the default against the lab's `iSMV.m`/`tdcsmv.m` run
//! in GNU Octave (`tests/data/ismv_wen2014_small.m`): at the paper's defaults on 1 mm and
//! anisotropic voxels, `iSMV.m` as shipped (tol 0.001), and an odd-sized array padded as above.
//! `tests/ismv_qsmjl_parity.rs` checks [`IsmvVariant::QsmJl`] against QSM.jl's output.
//!
//! Reference:
//! Wen, Y., Zhou, D., Liu, T., Spincemaille, P., Wang, Y. (2014).
//! "An iterative spherical mean value method for background field removal in MRI."
//! Magnetic Resonance in Medicine, 72(4):1065-1071. <https://doi.org/10.1002/mrm.24998>
//!
//! Reference implementation: <https://github.com/agr78/mSMV> (`iSMV.m`, `tdcsmv.m`, 42a1f18);
//! QSM.jl's: <https://github.com/kamesy/QSM.jl>

use num_complex::Complex64;
use crate::Grid;
use crate::error::InputError;
use crate::fft::{fft3d, ifft3d, fft_real_kernel, Fft3dWorkspace};
use crate::kernels::smv::{smv_kernel, erode_mask_smv};
use crate::utils::vec_norm;

/// Largest sub-voxel factor `tdcsmv.m`'s search for an integral `voxel_size(1)·w` tries before
/// falling back to 10 (MATLAB's loop would not end). See the [module docs](self#ismvm-step-by-step).
pub const TDCSMV_MAX_SUBVOXEL: u32 = 64;

/// `iSMV.m`'s erosion threshold: `M = ifftn(fftn(Mask).*a) > .999`.
const LAB_ERODE_THRESHOLD: f64 = 0.999;

/// Which implementation of iSMV to reproduce. See the [module docs](self).
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IsmvVariant {
    /// The authors' paper and code: the lab's `iSMV.m` + `tdcsmv.m` (agr78/mSMV 42a1f18) with
    /// Wen et al. 2014's stopping rule, odd axes padded to even. The default.
    #[default]
    Wen2014,
    /// QSM.jl's `ismv` (fde4b92): its ball, `1 − √eps` erosion and residual-based stopping
    /// rule. For the QSM.jl parity pair.
    QsmJl,
}

/// iSMV parameters. The defaults are Wen et al. 2014's with the lab's `iSMV.m` (see the
/// [module docs](self)); [`IsmvParams::qsmjl`] gives QSM.jl's.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct IsmvParams {
    /// The implementation to reproduce (default [`IsmvVariant::Wen2014`]).
    pub variant: IsmvVariant,
    /// Stopping tolerance; finite and > 0. [`IsmvVariant::Wen2014`]: stop when the relative
    /// change between successive iterates, `‖F₁ − F₂‖/‖F₁‖`, falls below it (default 5e-5,
    /// Wen et al. 2014; `iSMV.m` hard-codes 0.001). [`IsmvVariant::QsmJl`]: stop when the
    /// residual falls to this fraction of the first one (QSM.jl's `tol`, 1e-3).
    pub tol: f64,
    /// Iteration cap; `None` (the default) is none, as in `iSMV.m`. QSM.jl's `maxit` is 500.
    pub max_iter: Option<usize>,
    /// Sphere radius in mm, finite and > 0; `None` (the default) is the variant's default:
    /// 6 voxels for [`IsmvVariant::Wen2014`] (Wen et al. 2014), `2 · max(voxel size)` for
    /// [`IsmvVariant::QsmJl`]. See [`IsmvParams::radius_mm`]. Before 2026-10-29 the default was
    /// QSM.jl's, and before 2026-10-28 5 mm with `tol` 1e-6 and 50 iterations, which had no
    /// reference.
    pub radius: Option<f64>,
}

impl Default for IsmvParams {
    fn default() -> Self {
        Self { variant: IsmvVariant::Wen2014, tol: Self::WEN2014_TOL, max_iter: None, radius: None }
    }
}

impl IsmvParams {
    /// Wen et al. 2014's stopping tolerance on the relative change between iterates.
    pub const WEN2014_TOL: f64 = 5e-5;
    /// Wen et al. 2014's radius in voxels (of the smallest voxel dimension).
    pub const WEN2014_RADIUS_VOXELS: f64 = 6.0;
    /// QSM.jl `ismv`'s default kernel radius as a multiple of the largest voxel dimension.
    pub const QSMJL_RADIUS_VOXELS: f64 = 2.0;

    /// QSM.jl `ismv` at its own defaults: [`IsmvVariant::QsmJl`], `tol` 1e-3, `maxit` 500,
    /// `r = 2 · max(voxel size)`.
    pub fn qsmjl() -> Self {
        Self { variant: IsmvVariant::QsmJl, tol: 1e-3, max_iter: Some(500), radius: None }
    }

    /// The sphere radius in mm on `grid`: [`radius`](Self::radius), or the variant's default
    /// (`6 · min(voxel size)` for [`IsmvVariant::Wen2014`], `2 · max(voxel size)` for
    /// [`IsmvVariant::QsmJl`]).
    pub fn radius_mm(&self, grid: &Grid) -> f64 {
        let (a, b, c) = grid.voxel_size;
        self.radius.unwrap_or(match self.variant {
            IsmvVariant::Wen2014 => Self::WEN2014_RADIUS_VOXELS * a.min(b).min(c),
            IsmvVariant::QsmJl => Self::QSMJL_RADIUS_VOXELS * a.max(b).max(c),
        })
    }

    /// Checks what iSMV cannot run with: `tol` and the radius must be finite and > 0 (at
    /// `tol` 0 and no cap the iteration would not end).
    pub fn validate(&self) -> Result<(), InputError> {
        if !(self.tol.is_finite() && self.tol > 0.0) {
            return Err(InputError::param("iSMV tol", self.tol, "finite and > 0"));
        }
        if let Some(r) = self.radius {
            if !(r.is_finite() && r > 0.0) {
                return Err(InputError::param("iSMV radius", r, "finite and > 0 mm"));
            }
        }
        Ok(())
    }
}

/// iSMV background field removal (see the [module docs](self))
///
/// # Arguments
/// * `field` - Total field (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz), 1 = brain, 0 = background
/// * `grid` - Volume dimensions and voxel sizes
/// * `params` - iSMV parameters
/// * `progress` - Progress callback (iteration, expected iterations; with no cap the second is
///   an estimate from the convergence rate so far)
///
/// # Returns
/// (local field, eroded mask). [`IsmvVariant::Wen2014`]'s local field is zero outside the
/// eroded mask (`iSMV.m`), [`IsmvVariant::QsmJl`]'s outside the input mask (QSM.jl).
pub fn ismv(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    params: &IsmvParams,
    progress: impl FnMut(usize, usize),
) -> Result<(Vec<f64>, Vec<u8>), InputError> {
    params.validate()?;
    let n = grid.dims.0 * grid.dims.1 * grid.dims.2;
    if field.len() != n || mask.len() != n {
        return Err(InputError::new(format!(
            "iSMV: field ({}) and mask ({}) must have {n} voxels ({:?})", field.len(), mask.len(), grid.dims,
        )));
    }
    let radius = params.radius_mm(grid);
    Ok(match params.variant {
        IsmvVariant::Wen2014 => ismv_wen2014(field, mask, grid, radius, params.tol, params.max_iter, progress),
        IsmvVariant::QsmJl => ismv_qsmjl(field, mask, grid, radius, params.tol, params.max_iter, progress),
    })
}

/// `tdcsmv.m` (agr78/mSMV 42a1f18): the SMV sphere of `radius` mm on a `dims` array, centred
/// at the FFT origin, normalised to sum 1 (image space; `tdcsmv.m` returns its `fftn`).
///
/// Each voxel within the sphere's reach gets the fraction of its `w`-per-edge sub-voxel samples
/// inside the sphere. Two changes, unchanged on every input `tdcsmv.m` runs on (see the
/// [module docs](self#ismvm-step-by-step)): the second axis uses `voxel_size.1` (`tdcsmv.m`
/// uses `voxel_size(1)` for it), and the search for `w` stops at [`TDCSMV_MAX_SUBVOXEL`].
/// Placement wraps around the array (`tdcsmv.m` places at `end/2+1+offset` and `fftshift`s,
/// which is the same on even sizes).
pub fn tdcsmv(dims: (usize, usize, usize), voxel_size: (f64, f64, f64), radius: f64) -> Vec<f64> {
    let (nx, ny, nz) = dims;
    let (v1, v2, v3) = voxel_size;
    // if (voxel_size(1)==voxel_size(3)) w = 10; else smallest w with int16(g*w) == single(g*w)
    let mut w: u32 = if v1 == v3 {
        10
    } else {
        let int16 = |x: f64| x.round().clamp(-32768.0, 32767.0);
        (1..=TDCSMV_MAX_SUBVOXEL)
            .find(|&w| { let gw = v1 * w as f64; int16(gw) == (gw as f32) as f64 })
            .unwrap_or(10)
    };
    w *= 2;
    let wf = w as f64;
    // vs = voxel_size*w; R = radius*w; dx = vs(1); dz = vs(3)
    let (dx, dy, dz) = (v1 * wf, v2 * wf, v3 * wf);
    let r2 = (radius * wf) * (radius * wf);
    // n = ceil(radius/voxel_size(1)) (per in-plane axis here); zmax = floor((radius+vs(3)/2)/vs(3))
    let n1 = (radius / v1).ceil() as i64;
    let n2 = (radius / v2).ceil() as i64;
    let zmax = ((radius + v3 / 2.0) / v3).floor() as i64;
    // meshgrid(-d/2+d1 : ps : d/2-d2) with ps = 1, d1 = d2 = 0.5 (MATLAB colon semantics)
    let colon = |d: f64| -> Vec<f64> {
        let (a, b) = (-d / 2.0 + 0.5, d / 2.0 - 0.5);
        if b < a {
            return Vec::new();
        }
        let tol = 3.0 * f64::EPSILON * a.abs().max(b.abs());
        let count = ((b - a) + tol).floor() as usize + 1;
        (0..count).map(|i| a + i as f64).collect()
    };
    let (px, py, pz) = (colon(dx), colon(dy), colon(dz));
    let wrap = |o: i64, m: usize| o.rem_euclid(m as i64) as usize;
    let mut f = vec![0.0f64; nx * ny * nz];
    let norm = wf * wf * wf; // (w/ps)^3
    for zof in -zmax..=zmax {
        let zz: Vec<f64> = pz.iter().map(|&z| (z + dz * zof as f64).powi(2)).collect();
        for j in (-n2 + 1)..=(n2 - 1) {
            let yy: Vec<f64> = py.iter().map(|&y| (y + dy * j as f64).powi(2)).collect();
            for k in (-n1 + 1)..=(n1 - 1) {
                let xx: Vec<f64> = px.iter().map(|&x| (x + dx * k as f64).powi(2)).collect();
                // sphere = ((x+dx*k).^2 + (y+dx*j).^2 + (z+dz*zof).^2) <= R^2, summed
                let mut count = 0usize;
                for &z2 in &zz {
                    for &y2 in &yy {
                        count += xx.iter().filter(|&&x2| x2 + y2 + z2 <= r2).count();
                    }
                }
                // f(end/2+1+k, end/2+1+j, end/2+zof+1) = sum(sphere(:))/((w/ps)^3)
                f[wrap(k, nx) + nx * (wrap(j, ny) + ny * wrap(zof, nz))] += count as f64 / norm;
            }
        }
    }
    let s: f64 = f.iter().sum();
    f.iter_mut().for_each(|v| *v /= s);
    f
}

/// The lab's `iSMV.m` with Wen et al. 2014's stopping rule (steps 0–4 of the module docs).
fn ismv_wen2014(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    radius: f64,
    tol: f64,
    max_iter: Option<usize>,
    mut progress: impl FnMut(usize, usize),
) -> (Vec<f64>, Vec<u8>) {
    let (nx, ny, nz) = grid.dims;
    // 0. SEPIA's zeropad_odd_dimension(…, 'pre'): one zero slice at the end of each odd axis.
    let (px, py, pz) = (nx + nx % 2, ny + ny % 2, nz + nz % 2);
    let n = px * py * pz;
    let at = |i: usize, j: usize, k: usize| i + px * (j + py * k);
    let pad = |x: &dyn Fn(usize) -> f64| {
        let mut out = vec![0.0; n];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    out[at(i, j, k)] = x(i + nx * (j + ny * k));
                }
            }
        }
        out
    };
    let ifreq = pad(&|l| field[l]);
    let m0 = pad(&|l| (mask[l] != 0) as u8 as f64);
    let mut fft = Fft3dWorkspace::new(px, py, pz);
    let to_c = |x: &[f64]| -> Vec<Complex64> { x.iter().map(|&v| Complex64::new(v, 0.0)).collect() };

    // 1. a = tdcsmv(matrix_size, voxel_size, radius)
    let mut a = to_c(&tdcsmv((px, py, pz), grid.voxel_size, radius));
    fft.fft3d(&mut a);

    // 2. M = ifftn(fftn(Mask).*a) > .999; BV = (Mask - M).*iFreq
    let mut c = to_c(&m0);
    fft.fft3d(&mut c);
    c.iter_mut().zip(&a).for_each(|(x, k)| *x *= k);
    fft.ifft3d(&mut c);
    let m: Vec<f64> = c.iter().map(|x| (x.re > LAB_ERODE_THRESHOLD) as u8 as f64).collect();
    let bv: Vec<f64> = (0..n).map(|l| (m0[l] - m[l]) * ifreq[l]).collect();

    // 3. F1 = iFreq; while d >= tol: F2 = ifftn(fftn(F1).*a).*M + BV; d = ‖F1-F2‖/‖F1‖; F1 = F2
    let mut f1 = to_c(&ifreq);
    let mut f2 = vec![Complex64::new(0.0, 0.0); n];
    let mut iter = 0usize;
    let mut d_prev = f64::NAN;
    loop {
        f2.copy_from_slice(&f1);
        fft.fft3d(&mut f2);
        f2.iter_mut().zip(&a).for_each(|(x, k)| *x *= k);
        fft.ifft3d(&mut f2);
        let (mut num, mut den) = (0.0, 0.0);
        for l in 0..n {
            f2[l] = f2[l] * m[l] + bv[l];
            num += (f1[l] - f2[l]).norm_sqr();
            den += f1[l].norm_sqr();
        }
        let d = num.sqrt() / den.sqrt();
        iter += 1;
        std::mem::swap(&mut f1, &mut f2); // F1 = F2 (f1 now holds the newest iterate)
        let capped = max_iter.is_some_and(|c| iter >= c);
        // `while (d>=tol)`: a NaN d ends the loop, as in MATLAB.
        if d < tol || d.is_nan() || capped {
            progress(iter, iter);
            break;
        }
        progress(iter, max_iter.unwrap_or_else(|| expected_iterations(iter, d, d_prev, tol)));
        d_prev = d;
    }

    // 4. L = (F0 - F2).*M, cropped back
    let mut local = vec![0.0; nx * ny * nz];
    let mut eroded = vec![0u8; nx * ny * nz];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let (l, p) = (i + nx * (j + ny * k), at(i, j, k));
                local[l] = ((Complex64::new(ifreq[p], 0.0) - f1[p]) * m[p]).re;
                eroded[l] = m[p] as u8;
            }
        }
    }
    (local, eroded)
}

/// Iterations an uncapped geometric convergence from `d_prev` to `d` needs to reach `tol`, for
/// the progress callback (at least one more than `iter`).
fn expected_iterations(iter: usize, d: f64, d_prev: f64, tol: f64) -> usize {
    let rate = d / d_prev;
    if !(rate.is_finite() && rate > 0.0 && rate < 1.0) {
        return iter + 1;
    }
    let more = ((tol / d).ln() / rate.ln()).ceil();
    iter + if more.is_finite() && more >= 1.0 { (more as usize).min(1_000_000) } else { 1 }
}

/// QSM.jl's `ismv` with an explicit radius (mm); see the [module docs](self).
fn ismv_qsmjl(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    radius: f64,
    tol: f64,
    max_iter: Option<usize>,
    mut progress: impl FnMut(usize, usize),
) -> (Vec<f64>, Vec<u8>) {
    let (nx, ny, nz) = grid.dims;
    let n_total = nx * ny * nz;
    let max_iter = max_iter.unwrap_or(usize::MAX);

    // Generate SMV kernel
    let smv = smv_kernel(grid, radius);
    let smv_fft_real = fft_real_kernel(&smv, nx, ny, nz);

    // Complex FFT needed for iteration loop
    let mut smv_fft: Vec<Complex64> = smv.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    fft3d(&mut smv_fft, nx, ny, nz);

    let m0: Vec<f64> = mask.iter().map(|&m| if m != 0 { 1.0 } else { 0.0 }).collect();

    // Erode mask using SMV
    let eroded_mask: Vec<f64> =
        erode_mask_smv(mask, &smv_fft_real, grid, 1.0 - 1e-10).iter().map(|&m| m as f64).collect();

    // Boundary correction: bc = (mask - eroded) * field
    let bc: Vec<f64> = (0..n_total).map(|i| (m0[i] - eroded_mask[i]) * field[i]).collect();

    let mut f: Vec<f64> = field.to_vec();
    // f0 = eroded_mask * field (for residual calculation)
    let mut f0: Vec<f64> = field.iter().zip(&eroded_mask).map(|(&fi, &m)| fi * m).collect();

    let mut nr = vec_norm(&f0);
    let eps = tol * nr;

    let mut iter = 0;
    while iter < max_iter {
        progress(iter + 1, if max_iter == usize::MAX { iter + 2 } else { max_iter });
        if nr <= eps {
            progress(iter + 1, iter + 1);
            break;
        }
        // f = eroded_mask * SMV(f) + bc
        let mut f_complex: Vec<Complex64> = f.iter().map(|&x| Complex64::new(x, 0.0)).collect();
        fft3d(&mut f_complex, nx, ny, nz);
        f_complex.iter_mut().zip(&smv_fft).for_each(|(x, k)| *x *= k);
        ifft3d(&mut f_complex, nx, ny, nz);
        for i in 0..n_total {
            f[i] = eroded_mask[i] * f_complex[i].re + bc[i];
        }
        // residual ||f0 - f||
        let mut residual_sq = 0.0;
        for i in 0..n_total {
            let diff = f0[i] - f[i];
            residual_sq += diff * diff;
            f0[i] = f[i];
        }
        nr = residual_sq.sqrt();
        iter += 1;
    }

    // local field: mask * (field - f)
    let local_field: Vec<f64> =
        (0..n_total).map(|i| if mask[i] != 0 { field[i] - f[i] } else { 0.0 }).collect();
    let eroded_mask_u8: Vec<u8> = eroded_mask.iter().map(|&m| (m > 0.5) as u8).collect();
    (local_field, eroded_mask_u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ball_mask(n: usize, r: f64) -> Vec<u8> {
        let c = (n as f64 - 1.0) / 2.0;
        let mut m = vec![0u8; n * n * n];
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let d2 = (i as f64 - c).powi(2) + (j as f64 - c).powi(2) + (k as f64 - c).powi(2);
                    m[i + n * (j + n * k)] = (d2 <= r * r) as u8;
                }
            }
        }
        m
    }

    fn both() -> [IsmvParams; 2] {
        [IsmvParams::default(), IsmvParams::qsmjl()]
    }

    #[test]
    fn defaults_are_the_papers_and_qsmjls() {
        let p = IsmvParams::default();
        assert_eq!(p.variant, IsmvVariant::Wen2014);
        assert_eq!(p.tol, 5e-5);
        assert_eq!(p.max_iter, None);
        let grid = Grid::new(8, 8, 8, 0.9, 0.9, 2.0);
        assert!((p.radius_mm(&grid) - 5.4).abs() < 1e-12);
        let q = IsmvParams::qsmjl();
        assert_eq!((q.variant, q.tol, q.max_iter), (IsmvVariant::QsmJl, 1e-3, Some(500)));
        assert_eq!(q.radius_mm(&grid), 4.0);
        assert_eq!(IsmvParams { radius: Some(3.0), ..q }.radius_mm(&grid), 3.0);
    }

    #[test]
    fn validate_refuses_unusable_tol_and_radius() {
        for tol in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(IsmvParams { tol, ..Default::default() }.validate().is_err(), "tol {tol}");
        }
        for r in [0.0, -2.0, f64::NAN] {
            assert!(IsmvParams { radius: Some(r), ..Default::default() }.validate().is_err(), "radius {r}");
        }
        let grid = Grid::new(4, 4, 4, 1.0, 1.0, 1.0);
        assert!(ismv(&[0.0; 64], &[1; 64], &grid, &IsmvParams { tol: 0.0, ..Default::default() }, |_, _| {}).is_err());
        assert!(ismv(&[0.0; 63], &[1; 64], &grid, &IsmvParams::default(), |_, _| {}).is_err());
    }

    #[test]
    fn tdcsmv_is_normalised_symmetric_and_degenerates_at_one_voxel() {
        let k = tdcsmv((16, 16, 16), (1.0, 1.0, 1.0), 3.0);
        assert!((k.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let at = |i: i64, j: i64, l: i64| k[(i.rem_euclid(16) + 16 * (j.rem_euclid(16) + 16 * l.rem_euclid(16))) as usize];
        for (i, j, l) in [(1, 0, 0), (2, 1, 0), (1, 1, 2)] {
            assert!((at(i, j, l) - at(-i, -j, -l)).abs() < 1e-15);
            assert!((at(i, j, l) - at(j, i, l)).abs() < 1e-15);
        }
        // tdcsmv.m's in-plane extent is ±(ceil(r/vs)-1) voxels, its z extent floor(r/vs + 1/2)
        assert_eq!(at(3, 0, 0), 0.0);
        assert!(at(0, 0, 3) > 0.0);
        // r = 1 voxel: a 3-voxel column along the third axis (the paper's other radius)
        let k1 = tdcsmv((8, 8, 8), (1.0, 1.0, 1.0), 1.0);
        let nonzero: Vec<usize> = (0..512).filter(|&l| k1[l] != 0.0).collect();
        assert_eq!(nonzero, vec![0, 64, 7 * 64]);
    }

    #[test]
    fn tdcsmv_second_axis_uses_its_own_voxel_size() {
        // vs₂ = 2·vs₁: the sphere reaches half as many voxels along the second axis
        let k = tdcsmv((16, 16, 16), (1.0, 2.0, 1.0), 4.0);
        let at = |i: i64, j: i64| k[(i.rem_euclid(16) + 16 * j.rem_euclid(16)) as usize];
        assert!(at(3, 0) > 0.0 && at(0, 1) > 0.0);
        assert_eq!(at(0, 2), 0.0);
        // a voxel size with no integral w (single precision) still builds a kernel
        let k = tdcsmv((16, 16, 16), (0.937_500_06, 0.937_500_06, 2.0), 4.0);
        assert!((k.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn harmonic_field_is_removed() {
        // A linear field is harmonic: iSMV's local field vanishes on the eroded mask.
        let n = 24;
        let field: Vec<f64> = (0..n * n * n).map(|l| 0.01 * (l / (n * n)) as f64 + 0.002 * (l % n) as f64).collect();
        let mask = ball_mask(n, 10.0);
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        for p in [IsmvParams { radius: Some(3.0), ..Default::default() }, IsmvParams::qsmjl()] {
            let p = IsmvParams { tol: 1e-9, max_iter: Some(5000), ..p };
            let (local, eroded) = ismv(&field, &mask, &grid, &p, |_, _| {}).unwrap();
            let kept: usize = eroded.iter().map(|&m| m as usize).sum();
            assert!(kept > 0 && kept < mask.iter().map(|&m| m as usize).sum::<usize>(), "{:?}", p.variant);
            // `iSMV.m`'s erosion keeps voxels whose sphere has up to 0.1 % of its weight outside
            // the mask, where the iterate is 0, so its fixed point is off by up to 1e-3 × the
            // field; QSM.jl's ball fits exactly.
            let bound = match p.variant { IsmvVariant::Wen2014 => 1e-3 * 0.25, IsmvVariant::QsmJl => 1e-6 };
            let worst = (0..local.len()).filter(|&l| eroded[l] != 0).map(|l| local[l].abs()).fold(0.0, f64::max);
            assert!(worst < bound, "{:?}: max |local| {worst}", p.variant);
            for l in 0..local.len() {
                assert!(eroded[l] == 0 || mask[l] != 0, "eroded mask within the mask");
                if mask[l] == 0 {
                    assert_eq!(local[l], 0.0);
                }
            }
        }
    }

    #[test]
    fn wen2014_is_zero_outside_the_eroded_mask_and_handles_odd_sizes() {
        let n = 21; // odd: padded to 22 internally
        let field: Vec<f64> = (0..n * n * n).map(|l| ((l * 7919) % 101) as f64 * 1e-3).collect();
        let mask = ball_mask(n, 9.0);
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let (local, eroded) = ismv(&field, &mask, &grid, &IsmvParams { radius: Some(3.0), ..Default::default() }, |_, _| {}).unwrap();
        assert_eq!((local.len(), eroded.len()), (n * n * n, n * n * n));
        assert!(eroded.iter().any(|&m| m != 0));
        for l in 0..local.len() {
            assert!(local[l].is_finite());
            if eroded[l] == 0 {
                assert_eq!(local[l], 0.0);
            }
        }
    }

    #[test]
    fn zero_field_and_progress() {
        let n = 12;
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let mask = ball_mask(n, 5.0);
        for p in both() {
            let p = IsmvParams { radius: Some(2.0), ..p };
            let mut calls = Vec::new();
            let (local, _) = ismv(&vec![0.0; n * n * n], &mask, &grid, &p, |i, m| calls.push((i, m))).unwrap();
            assert!(local.iter().all(|&v| v == 0.0), "{:?}", p.variant);
            assert!(!calls.is_empty());
            let (i, m) = *calls.last().unwrap();
            assert_eq!(i, m, "{:?}: the last progress call is complete", p.variant);
        }
    }

    #[test]
    fn iteration_cap_is_honoured() {
        let n = 16;
        let field: Vec<f64> = (0..n * n * n).map(|l| ((l * 31) % 17) as f64 * 1e-3).collect();
        let mask = ball_mask(n, 7.0);
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        for p in both() {
            let p = IsmvParams { radius: Some(2.0), tol: 1e-14, max_iter: Some(3), ..p };
            let mut last = 0;
            ismv(&field, &mask, &grid, &p, |i, _| last = i).unwrap();
            assert!(last <= 4, "{:?}: {last}", p.variant);
        }
    }

    #[test]
    fn larger_radius_erodes_more() {
        let n = 20;
        let field = vec![0.1; n * n * n];
        let mask = ball_mask(n, 8.0);
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        for p in both() {
            let count = |r: f64| {
                let (_, e) = ismv(&field, &mask, &grid, &IsmvParams { radius: Some(r), ..p.clone() }, |_, _| {}).unwrap();
                e.iter().map(|&m| m as usize).sum::<usize>()
            };
            assert!(count(4.0) < count(2.0), "{:?}", p.variant);
        }
    }

    #[test]
    fn expected_iterations_extrapolates_geometric_convergence() {
        assert_eq!(expected_iterations(3, 1e-3, f64::NAN, 1e-5), 4);
        assert_eq!(expected_iterations(3, 1e-3, 1e-2, 1e-5), 5);
        assert_eq!(expected_iterations(3, 1e-3, 1e-3, 1e-5), 4);
    }
}

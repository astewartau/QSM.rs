//! SHARP background field removal
//!
//! Sophisticated Harmonic Artifact Reduction for Phase data.
//! Uses the spherical mean value property of harmonic functions to
//! separate local from background fields.
//!
//! # Which implementation
//!
//! The SHARP authors (Schweser et al. 2011) published no code. Where the authors' code does not
//! exist, QSM.rs follows one designated implementation per family; for Kames's SMV methods
//! (SHARP, V-SHARP's filtering, iSMV) and inversions that is QSM.jl, his maintained successor
//! to QSM.m. So this module is a port of QSM.jl's `sharp` (kamesy/QSM.jl fde4b92,
//! `src/bgremove/sharp.jl`), with its defaults. Other implementations, for reference:
//!
//! | implementation | radius | TSVD threshold |
//! |---|---|---|
//! | QSM.jl `sharp` (followed) | `18·min(vsz)` mm | 0.05 |
//! | QSM.m `sharp.m` (kamesy/QSM.m f22dc40, `src/bgremove/sharp.m:13-15,33-34`) | 9 mm | 0.05 |
//! | Hongfu Sun's `sharp.m` (sunhongfu/QSM c75b138, `background_field_removal/sharp.m`) | 4 mm if omitted (`ker_rad`, rounded to voxels per axis); his drivers pass 2–5 mm, mostly 3 | no default; his drivers pass 0.05–0.1, mostly 0.1 |
//!
//! QSM.jl, line by line:
//!
//! 1. Crop the field and the mask to the mask's bounding box (`crop_indices`) and zero-pad them
//!    to `fastfftsize(box, (p, p, p); rfft = true)` with `p = ⌈r / min(vsz)⌉`
//!    ([`qsmjl_fast_fft_size`](crate::utils::padding::qsmjl_fast_fft_size)); every FFT below is
//!    periodic on that padded box.
//! 2. `S` = the FFT of the normalised ball of voxel centres within `r` mm (`_smv_kernel!`).
//! 3. Erode the mask to the voxels where `S ∗ mask > 1 − √eps` (eps of `Float64`), i.e. where
//!    the whole ball fits in the mask. (This module used `1 − √1e-7` before, which also accepts
//!    balls missing a few voxels once a ball has more than ~3,200.)
//! 4. High-pass `(1 − S)·F(f)`, back to image space, multiply by the eroded mask.
//! 5. Deconvolve: `F⁻¹[F(·)/(1 − S)]` with the frequencies where `|1 − S| < thr` set to 0
//!    (truncated SVD), and multiply by the eroded mask again.
//!
//! QSM.jl uses real-to-complex FFTs, this port complex ones; otherwise every step is QSM.jl's.
//!
//! # Agreement with QSM.jl
//!
//! On the QSM-CI in-vivo travelling-head scan (Siemens Prisma 2.89 T, 1 mm, 184×256×128, ROMEO
//! total field and brain mask), QSM.jl's `sharp` in the QSM-CI `julia-qsm` image
//! (`sha256:b3c048ac…`, QSM.jl fde4b92) and this port with the same `r` and `thr` agree to
//! float rounding (both outputs saved as float32), with identical eroded masks: at `r` 6 mm
//! NRMSE 6.6e-10 %, max |Δ| 1.2e-10 ppm; at QSM.jl's default 18 mm NRMSE 7.8e-12 %, max |Δ|
//! 9.1e-13 ppm. Before this module used QSM.jl's crop and padding (the
//! deconvolution was periodic on the whole field of view instead), the same comparison at
//! `r` = 6 mm gave NRMSE 12.6 %. `tests/sharp_qsmjl_parity.rs` checks the port against QSM.jl's
//! output on small phantoms.
//!
//! # Reference pipeline
//!
//! QSM.jl's README pipeline uses V-SHARP, not SHARP (see [`super::vsharp`]). Hongfu Sun's
//! `sharp.m` keeps voxels where the normalised SMV of the mask exceeds `1 − 1/|ball|`, and his
//! driver `Siemens_3T/qsm_r2s_prisma.m` runs it on a BET mask (`bet2 -f 0.4 -w 2`) times his
//! echo-fit reliability map.
//!
//! Reference:
//! Schweser, F., Deistung, A., Lehr, B.W., Reichenbach, J.R. (2011).
//! "Quantitative imaging of intrinsic magnetic tissue properties using MRI signal phase."
//! NeuroImage, 54(4):2789-2807. https://doi.org/10.1016/j.neuroimage.2010.10.070
//!
//! Reference implementation: https://github.com/kamesy/QSM.jl

use num_complex::Complex64;
use crate::Grid;
use crate::fft::{fft3d, ifft3d, fft_real_kernel};
use crate::kernels::smv::{smv_kernel, erode_mask_smv};
use crate::utils::padding::qsmjl_fast_fft_size;

/// QSM.jl's threshold for "the whole ball fits in the mask": `S ∗ mask > 1 − √eps(Float64)`
/// (`δ` in `src/bgremove/sharp.jl` `_sharp!`, fde4b92). Shared by SHARP and V-SHARP.
pub(crate) const QSMJL_SMV_DELTA: f64 = 1.0 - 1.490_116_119_384_765_6e-8; // 1 - sqrt(f64::EPSILON)

/// SHARP algorithm parameters (QSM.jl `sharp` keywords and defaults)
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct SharpParams {
    /// Deconvolution (TSVD) threshold: frequencies where `|1 − S| < threshold` are set to 0
    /// (default: 0.05, QSM.jl `sharp`'s `thr`, `src/bgremove/sharp.jl`, fde4b92)
    pub threshold: f64,
    /// SMV kernel radius in mm; `None` (the default) is QSM.jl `sharp`'s default `r`,
    /// 18 × the smallest voxel dimension (18 mm at 1 mm). See [`SharpParams::radius_mm`].
    /// (Before it followed QSM.jl this defaulted to 6 mm, which had no reference.)
    pub radius: Option<f64>,
}

impl Default for SharpParams {
    fn default() -> Self {
        Self {
            threshold: 0.05,
            radius: None,
        }
    }
}

impl SharpParams {
    /// QSM.jl `sharp`'s default kernel radius as a multiple of the smallest voxel dimension.
    pub const QSMJL_RADIUS_VOXELS: f64 = 18.0;

    /// The kernel radius in mm on `grid`: [`radius`](Self::radius), or QSM.jl's default
    /// `18 · min(voxel size)`.
    pub fn radius_mm(&self, grid: &Grid) -> f64 {
        let (a, b, c) = grid.voxel_size;
        self.radius.unwrap_or(Self::QSMJL_RADIUS_VOXELS * a.min(b).min(c))
    }
}

/// SHARP background field removal (QSM.jl `sharp`; see the [module docs](self))
///
/// # Arguments
/// * `field` - Unwrapped total field (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz), 1 = inside ROI
/// * `grid` - Volume dimensions and voxel sizes
/// * `params` - SHARP parameters (threshold, radius in mm)
///
/// # Returns
/// (local_field, eroded_mask)
pub fn sharp(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    params: &SharpParams,
) -> (Vec<f64>, Vec<u8>) {
    sharp_core(field, mask, grid, params.threshold, params.radius_mm(grid))
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

/// SHARP with an explicit absolute kernel radius (mm).
///
/// Internal entry point shared with V-SHARP; the public [`sharp`] wrapper
/// derives the radius from [`SharpParams`].
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

    #[test]
    fn test_sharp_zero_field() {
        // Zero field should give zero local field
        let n = 16;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        // Use small radius for small test array
        let (local, _) = sharp(&field, &mask, &grid, &SharpParams { threshold: 0.05, radius: Some(2.0) });

        for &val in local.iter() {
            assert!(val.abs() < 1e-8, "Zero field should give zero local field, got {}", val);
        }
    }

    #[test]
    fn test_sharp_finite() {
        // Result should be finite (no NaN or Inf)
        let n = 16;
        let field: Vec<f64> = (0..n*n*n).map(|i| (i as f64) * 0.01).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        let (local, eroded) = sharp(&field, &mask, &grid, &SharpParams { threshold: 0.05, radius: Some(2.0) });

        for (i, &val) in local.iter().enumerate() {
            assert!(val.is_finite(), "Local field should be finite at index {}", i);
        }

        // Eroded mask should have at least some voxels
        let eroded_count: usize = eroded.iter().map(|&m| m as usize).sum();
        assert!(eroded_count > 0, "Eroded mask should have some voxels");
    }
}

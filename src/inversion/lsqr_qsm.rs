//! Minimally regularised LSQR dipole inversion (Schweser 2010)
//!
//! Solves the dipole inversion as one large least-squares system and leans on
//! LSQR's minimum-norm property instead of an explicit regularisation term —
//! hence "minimally regularised".
//!
//! Two extra sets of unknowns make that work:
//!
//! * a **residual field** `r`, added to the data equation with weight `w`, which
//!   absorbs field contributions the dipole model cannot explain (unmodelled
//!   background, noise, non-local sources). Because LSQR converges to the
//!   minimum-norm solution, `‖r‖` is implicitly penalised, and `w` sets how
//!   cheaply the fit may spend field on `r` rather than on χ.
//! * a **global offset** `c`, one scalar, absorbing any constant field shift
//!   left over from background removal.
//!
//! With χ over the whole (cropped and padded) grid, `r` and the data over the mask only:
//!
//! ```text
//! minimise ‖ W · ( M[ F⁻¹(D · F χ) + c ] + w·r − b ) ‖₂   over  [χ; r; c]
//! ```
//!
//! `W` is the optional magnitude image (used as is, as upstream does) and `M` restricts to
//! the mask.
//!
//! Ported from the authors' code, `inversion_lsqr.m` and `computesusceptibility_lsqr.m` of
//! qsm_heidi (<https://gitlab.com/R01NS114227/qsm_heidi>), with its settings:
//!
//! * the grid: field, mask and magnitude are cropped to the mask's bounding box, each odd
//!   extent grown by one voxel, zero-padded by 20 voxels per side and the last voxel dropped
//!   (`zealouscrop`, `prepareforconvolution` with its 40-voxel anti-aliasing frame), so every
//!   extent is odd;
//! * the dipole kernel: the unit dipole response `(3cos²θ − 1)/(4πr³)` sampled in the
//!   **spatial domain** on that grid and Fourier transformed (`generateunitdipoleresponse`
//!   with `isFourierDomainFormula = false`), see [`spatial_dipole_kernel`];
//! * `w = 0.2/9.4 · B0`, the global offset, tolerance 1e-5 and at most 400 iterations of
//!   MATLAB's `lsqr` (its stopping rules, via [`super::ilsqr::lsqr_matlab`]).
//!
//! # Units
//! The problem is linear and homogeneous in the field: scaling the field scales χ, `r` and `c`
//! alike and leaves LSQR's iterates and stopping tests unchanged. Upstream solves for the
//! phase in radians and converts χ to ppm at the end; here the field goes in ppm and χ comes
//! out in ppm, which is the same solution (on a 3T scan the two agree to rounding).
//!
//! This map is what HEIDI consumes: [`super::heidi`] keeps its well-conditioned
//! k-space coefficients and re-derives the rest under a weighted-TV prior.
//!
//! Reference:
//! Schweser, F., Deistung, A., Lehr, B.W., Reichenbach, J.R. (2010).
//! "Differentiation between diamagnetic and paramagnetic cerebral lesions based
//! on magnetic susceptibility mapping." Medical Physics, 37(10):5165-5178.
//! https://doi.org/10.1118/1.3481505

use num_complex::Complex64;

use crate::fft::Fft3dWorkspace;
use crate::utils::apply_mask_zero;
use crate::Grid;

use super::ilsqr::lsqr_matlab;

/// Residual weighting per tesla: upstream's `0.2 / 9.4 * B0` ("0.2 optimal in QUASAR paper").
pub const RESIDUAL_WEIGHTING_PER_TESLA: f64 = 0.2 / 9.4;

/// Width of upstream's anti-aliasing frame (`DEFAULT_ANTIALIASINGFRAMEWIDTH`), in voxels,
/// split evenly over both sides of each axis.
pub const ANTIALIASING_FRAME: usize = 40;

/// Parameters for [`lsqr_qsm`].
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct LsqrQsmParams {
    /// Field strength in tesla. A scan parameter, not a user knob — the pipeline
    /// dispatcher overrides it from scan metadata.
    pub b0: f64,
    /// Weight `w` on the residual-field unknowns. `None` uses
    /// `RESIDUAL_WEIGHTING_PER_TESLA * b0`; set to `Some(0.0)` to drop the
    /// residual term entirely and solve the plain masked system.
    pub residual_weighting: Option<f64>,
    /// Fit a single global field offset alongside χ.
    pub fit_global_offset: bool,
    /// LSQR convergence tolerance (MATLAB `lsqr`'s `tol`).
    pub tol: f64,
    /// Maximum LSQR iterations.
    pub max_iter: usize,
    /// Zero the returned χ outside the mask. HEIDI wants this off — it low-passes
    /// χ through the dipole cone, and a hard mask edge rings.
    pub mask_output: bool,
}

impl Default for LsqrQsmParams {
    /// `inversion_lsqr.m`'s settings.
    fn default() -> Self {
        Self {
            b0: 3.0,
            residual_weighting: None,
            fit_global_offset: true,
            tol: 1e-5,
            max_iter: 400,
            mask_output: true,
        }
    }
}

impl LsqrQsmParams {
    /// Effective residual weight `w`.
    pub fn effective_residual_weighting(&self) -> f64 {
        self.residual_weighting
            .unwrap_or(RESIDUAL_WEIGHTING_PER_TESLA * self.b0)
    }
}

/// Per axis: (first voxel of the crop, its extent, the padded extent). See [`crop_grid`].
pub(crate) type CropBox = [(usize, usize, usize); 3];

/// The grid upstream solves on: the mask's bounding box (`zealouscrop`), each extent made even
/// by one trailing voxel and then grown by the anti-aliasing frame less one voxel
/// (`prepareforconvolution`: `makeeven`, 20 voxels per side, `makeodd`).
///
/// Returns `(start, extent, padded)` per axis, or `None` for an empty mask. Placing the crop
/// at the corner of the padded grid rather than 20 voxels in gives the same solution: every
/// operator is a circular convolution or a pointwise product.
pub(crate) fn crop_grid(mask: &[u8], dims: (usize, usize, usize)) -> Option<CropBox> {
    let (nx, ny, nz) = dims;
    let (mut lo, mut hi) = ([usize::MAX; 3], [0usize; 3]);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                if mask[i + nx * (j + ny * k)] != 0 {
                    for (d, c) in [i, j, k].into_iter().enumerate() {
                        lo[d] = lo[d].min(c);
                        hi[d] = hi[d].max(c);
                    }
                }
            }
        }
    }
    if lo[0] == usize::MAX {
        return None;
    }
    Some(std::array::from_fn(|d| {
        let e = hi[d] - lo[d] + 1;
        (lo[d], e, e + e % 2 + ANTIALIASING_FRAME - 1)
    }))
}

/// Copy the `crop` box of `x` (on `dims`) to the corner of a zero grid of the padded size.
pub(crate) fn crop_pad<T: Copy + Default>(x: &[T], dims: (usize, usize, usize), crop: &CropBox) -> Vec<T> {
    let (nx, ny, _) = dims;
    let (px, py, pz) = (crop[0].2, crop[1].2, crop[2].2);
    let mut out = vec![T::default(); px * py * pz];
    for k in 0..crop[2].1 {
        for j in 0..crop[1].1 {
            let src = crop[0].0 + nx * (crop[1].0 + j + ny * (crop[2].0 + k));
            let dst = px * (j + py * k);
            out[dst..dst + crop[0].1].copy_from_slice(&x[src..src + crop[0].1]);
        }
    }
    out
}

/// Inverse of [`crop_pad`]: the crop box back in place on `dims`, zero elsewhere.
pub(crate) fn uncrop(x: &[f64], dims: (usize, usize, usize), crop: &CropBox) -> Vec<f64> {
    let (nx, ny, nz) = dims;
    let (px, py) = (crop[0].2, crop[1].2);
    let mut out = vec![0.0; nx * ny * nz];
    for k in 0..crop[2].1 {
        for j in 0..crop[1].1 {
            let dst = crop[0].0 + nx * (crop[1].0 + j + ny * (crop[2].0 + k));
            let src = px * (j + py * k);
            out[dst..dst + crop[0].1].copy_from_slice(&x[src..src + crop[0].1]);
        }
    }
    out
}

/// The dipole kernel in k-space (FFT order, real) as qsm_heidi's
/// `generateunitdipoleresponse(dims, true, Options)` builds it with
/// `isFourierDomainFormula = false` and a voxel size given: the unit dipole response
/// `(3(b·r̂)² − 1)/(4π|r|³)` (zero at `r = 0`) sampled on an isotropic grid of the smallest
/// voxel size, `2⌊n·a/2⌋ + 1` voxels along an axis of `n` voxels and aspect `a = vs/min(vs)`,
/// Fourier transformed, cropped to the central `n` frequencies, real part, DC set to zero.
/// For isotropic voxels the fine grid is the grid itself.
///
/// Every extent must be odd (upstream refuses even grids).
pub fn spatial_dipole_kernel(grid: &Grid, bdir: (f64, f64, f64)) -> Vec<f64> {
    let dims = [grid.dims.0, grid.dims.1, grid.dims.2];
    let vs = [grid.voxel_size.0, grid.voxel_size.1, grid.voxel_size.2];
    assert!(dims.iter().all(|&n| n % 2 == 1), "the spatial dipole kernel needs odd extents, got {dims:?}");
    let vmin = vs.iter().cloned().fold(f64::INFINITY, f64::min);
    let fine: [usize; 3] = std::array::from_fn(|d| {
        let x = dims[d] as f64 * vs[d] / vmin;
        2 * (x / 2.0).floor() as usize + 1
    });
    let bn = (bdir.0 * bdir.0 + bdir.1 * bdir.1 + bdir.2 * bdir.2).sqrt();
    let b = [bdir.0 / bn, bdir.1 / bn, bdir.2 / bn];
    // FFT order: index i holds coordinate i (i <= (n-1)/2) or i - n.
    let coord = |i: usize, n: usize| if i <= (n - 1) / 2 { i as f64 } else { i as f64 - n as f64 };
    let (fx, fy, fz) = (fine[0], fine[1], fine[2]);
    let mut k: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); fx * fy * fz];
    for iz in 0..fz {
        let z = coord(iz, fz);
        for iy in 0..fy {
            let y = coord(iy, fy);
            for ix in 0..fx {
                let x = coord(ix, fx);
                let r2 = x * x + y * y + z * z;
                if r2 == 0.0 {
                    continue;
                }
                let r = r2.sqrt();
                let c = (b[0] * x + b[1] * y + b[2] * z) / r;
                k[ix + fx * (iy + fy * iz)] = Complex64::new((3.0 * c * c - 1.0) / (r2 * r * 4.0 * std::f64::consts::PI), 0.0);
            }
        }
    }
    Fft3dWorkspace::new(fx, fy, fz).fft3d(&mut k);
    // Frequency index m of the coarse grid sits at m mod fine on the fine one.
    let fidx = |i: usize, n: usize, f: usize| if i <= (n - 1) / 2 { i } else { f + i - n };
    let (nx, ny, nz) = grid.dims;
    let mut d = vec![0.0; nx * ny * nz];
    for iz in 0..nz {
        let jz = fidx(iz, nz, fz);
        for iy in 0..ny {
            let jy = fidx(iy, ny, fy);
            for ix in 0..nx {
                d[ix + nx * (iy + ny * iz)] = k[fidx(ix, nx, fx) + fx * (jy + fy * jz)].re;
            }
        }
    }
    d[0] = 0.0;
    d
}

/// The solve on upstream's grid. Returns the cropped-and-padded grid, its crop box, and χ on
/// it (unmasked), or `None` for an empty mask.
pub(crate) fn lsqr_qsm_padded(
    local_field: &[f64],
    mask: &[u8],
    magnitude: Option<&[f64]>,
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &LsqrQsmParams,
    progress: impl FnMut(usize, usize),
) -> Option<(Grid, CropBox, Vec<f64>)> {
    let n_total = grid.n_total();
    assert_eq!(local_field.len(), n_total, "field length must match grid");
    assert_eq!(mask.len(), n_total, "mask length must match grid");
    if let Some(m) = magnitude {
        assert_eq!(m.len(), n_total, "magnitude length must match grid");
    }
    let crop = crop_grid(mask, grid.dims)?;
    let (vsx, vsy, vsz) = grid.voxel_size;
    let pgrid = Grid::new(crop[0].2, crop[1].2, crop[2].2, vsx, vsy, vsz);
    let field = crop_pad(local_field, grid.dims, &crop);
    let pmask = crop_pad(mask, grid.dims, &crop);
    let pmag = magnitude.map(|m| crop_pad(m, grid.dims, &crop));
    let chi = lsqr_qsm_core(&field, &pmask, pmag.as_deref(), &pgrid, bdir, params, progress);
    Some((pgrid, crop, chi))
}

/// Minimally regularised LSQR dipole inversion.
///
/// # Arguments
/// * `local_field` - Local field in ppm (`nx * ny * nz`, Fortran order)
/// * `mask` - Binary mask; the data equation is written over these voxels only
/// * `magnitude` - Optional magnitude image, the row weight `W` (used as is; LSQR's solution
///   does not depend on its scale)
/// * `grid` - Volume grid
/// * `bdir` - B0 direction
/// * `params` - Algorithm parameters
/// * `progress` - Progress callback `(iteration, max_iter)`
///
/// # Returns
/// Susceptibility map in ppm (zero outside the mask's bounding box, and outside the mask
/// with `mask_output`).
pub fn lsqr_qsm(
    local_field: &[f64],
    mask: &[u8],
    magnitude: Option<&[f64]>,
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &LsqrQsmParams,
    progress: impl FnMut(usize, usize),
) -> Vec<f64> {
    let Some((_, crop, chi)) = lsqr_qsm_padded(local_field, mask, magnitude, grid, bdir, params, progress) else {
        return vec![0.0; grid.n_total()];
    };
    let mut chi = uncrop(&chi, grid.dims, &crop);
    if params.mask_output {
        apply_mask_zero(&mut chi, mask);
    }
    chi
}

fn lsqr_qsm_core(
    local_field: &[f64],
    mask: &[u8],
    magnitude: Option<&[f64]>,
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &LsqrQsmParams,
    progress: impl FnMut(usize, usize),
) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let n_total = grid.n_total();

    // Voxels carrying a data row.
    let mask_idx: Vec<usize> = (0..n_total).filter(|&i| mask[i] != 0).collect();
    let m = mask_idx.len();

    let w_res = params.effective_residual_weighting();
    let use_residual = w_res != 0.0;
    let n_res = if use_residual { m } else { 0 };
    let n_offset = usize::from(params.fit_global_offset);
    let n_unknown = n_total + n_res + n_offset;

    // Row weights: the magnitude inside the mask, as upstream's `spatialWeightingMatrix`.
    let row_weight: Vec<f64> = match magnitude {
        Some(mag) => mask_idx.iter().map(|&i| mag[i]).collect(),
        None => vec![1.0; m],
    };

    // Right-hand side: the field over the mask, row-weighted. Non-finite values are set to 0,
    // as upstream does.
    let b: Vec<Complex64> = mask_idx
        .iter()
        .zip(&row_weight)
        .map(|(&i, &w)| {
            let v = local_field[i] * w;
            Complex64::new(if v.is_finite() { v } else { 0.0 }, 0.0)
        })
        .collect();

    let d = spatial_dipole_kernel(grid, bdir);
    let mut fft = Fft3dWorkspace::new(nx, ny, nz);
    let mut buf = vec![Complex64::new(0.0, 0.0); n_total];
    let mut apply_dipole = |x: &[Complex64], buf: &mut Vec<Complex64>| {
        for (o, v) in buf.iter_mut().zip(x) {
            *o = Complex64::new(v.re, 0.0);
        }
        fft.fft3d(buf);
        for (o, &dk) in buf.iter_mut().zip(&d) {
            *o *= dk;
        }
        fft.ifft3d(buf);
        // upstream takes the real part of the convolution
        for o in buf.iter_mut() {
            o.im = 0.0;
        }
    };

    let mut progress = progress;
    let max_iter = params.max_iter;
    let mut calls = 0usize;
    let buf_cell = std::cell::RefCell::new(&mut buf);
    let apply_dipole_cell = std::cell::RefCell::new(&mut apply_dipole);

    let apply_a = |x: &[Complex64]| -> Vec<Complex64> {
        calls += 1;
        progress(calls.min(max_iter), max_iter);
        let mut buf = buf_cell.borrow_mut();
        (apply_dipole_cell.borrow_mut())(&x[..n_total], &mut buf);
        let offset = if n_offset == 1 { x[n_total + n_res].re } else { 0.0 };
        mask_idx
            .iter()
            .zip(&row_weight)
            .enumerate()
            .map(|(row, (&idx, &w))| {
                let mut v = buf[idx].re + offset;
                if use_residual {
                    v += w_res * x[n_total + row].re;
                }
                Complex64::new(w * v, 0.0)
            })
            .collect()
    };

    let apply_at = |y: &[Complex64]| -> Vec<Complex64> {
        // Scatter the weighted residual back onto the grid.
        let mut scattered = vec![Complex64::new(0.0, 0.0); n_total];
        for (row, (&idx, &w)) in mask_idx.iter().zip(&row_weight).enumerate() {
            scattered[idx] = Complex64::new(y[row].re * w, 0.0);
        }
        let mut out = vec![Complex64::new(0.0, 0.0); n_unknown];
        if use_residual {
            for (row, &idx) in mask_idx.iter().enumerate() {
                out[n_total + row] = scattered[idx] * w_res;
            }
        }
        if n_offset == 1 {
            out[n_total + n_res] = Complex64::new(mask_idx.iter().map(|&i| scattered[i].re).sum(), 0.0);
        }
        // The dipole kernel is real and even, so it is its own adjoint.
        let mut buf = buf_cell.borrow_mut();
        (apply_dipole_cell.borrow_mut())(&scattered, &mut buf);
        out[..n_total].copy_from_slice(&buf[..]);
        out
    };

    let (solution, _) = lsqr_matlab(apply_a, apply_at, &b, params.tol, params.max_iter);
    solution[..n_total].iter().map(|c| c.re).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::dipole::dipole_kernel;

    fn test_grid(n: usize) -> Grid {
        Grid::new(n, n, n, 1.0, 1.0, 1.0)
    }

    #[test]
    fn zero_field_gives_zero_chi() {
        let g = test_grid(12);
        let n = g.n_total();
        let chi = lsqr_qsm(
            &vec![0.0; n],
            &vec![1u8; n],
            None,
            &g,
            (0.0, 0.0, 1.0),
            &LsqrQsmParams::default(),
            |_, _| {},
        );
        assert!(chi.iter().all(|v| v.abs() < 1e-12), "expected an all-zero map");
    }

    #[test]
    fn empty_mask_is_handled() {
        let g = test_grid(8);
        let n = g.n_total();
        let chi = lsqr_qsm(
            &vec![0.1; n],
            &vec![0u8; n],
            None,
            &g,
            (0.0, 0.0, 1.0),
            &LsqrQsmParams::default(),
            |_, _| {},
        );
        assert_eq!(chi.len(), n);
        assert!(chi.iter().all(|v| *v == 0.0));
    }

    /// A prime-factored grid takes the fast-FFT padding path.
    #[test]
    fn padding_path_preserves_dimensions() {
        let (nx, ny, nz) = (7usize, 11usize, 13usize);
        let g = Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);
        let n_total = g.n_total();
        let field: Vec<f64> = (0..n_total).map(|i| ((i as f64) * 0.21).sin() * 0.02).collect();
        let chi = lsqr_qsm(
            &field,
            &vec![1u8; n_total],
            None,
            &g,
            (0.0, 0.0, 1.0),
            &LsqrQsmParams { max_iter: 20, ..Default::default() },
            |_, _| {},
        );
        assert_eq!(chi.len(), n_total);
        assert!(chi.iter().all(|v| v.is_finite()));
        assert!(chi.iter().any(|v| v.abs() > 0.0));
    }

    /// Forward-simulate a sphere, invert, and check we land near the truth.
    #[test]
    fn recovers_a_sphere() {
        let n = 32;
        let g = test_grid(n);
        let n_total = g.n_total();
        let d = dipole_kernel(&g, (0.0, 0.0, 1.0));

        let c = n as f64 / 2.0;
        let mut chi_true = vec![0.0; n_total];
        let mut mask = vec![0u8; n_total];
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let idx = i + j * n + k * n * n;
                    let r = ((i as f64 - c).powi(2)
                        + (j as f64 - c).powi(2)
                        + (k as f64 - c).powi(2))
                    .sqrt();
                    if r < 4.0 {
                        chi_true[idx] = 0.1;
                    }
                    if r < 11.0 {
                        mask[idx] = 1;
                    }
                }
            }
        }

        // Forward model: field = real(ifft(D * fft(chi)))
        let mut ws = Fft3dWorkspace::new(n, n, n);
        let mut buf: Vec<Complex64> =
            chi_true.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        ws.fft3d(&mut buf);
        for (b, &dk) in buf.iter_mut().zip(&d) {
            *b *= dk;
        }
        ws.ifft3d(&mut buf);
        let field: Vec<f64> = buf.iter().map(|c| c.re).collect();

        let params = LsqrQsmParams {
            residual_weighting: Some(0.0),
            fit_global_offset: false,
            max_iter: 120,
            ..Default::default()
        };
        let chi = lsqr_qsm(&field, &mask, None, &g, (0.0, 0.0, 1.0), &params, |_, _| {});

        // The sphere comes back positive and with the right shape. A masked,
        // cone-degenerate system solved for its minimum-norm solution
        // under-estimates amplitude — ~0.07 for a true 0.1 here — so this
        // checks structure (correlation) and sign, not absolute recovery.
        let (sum, count) = chi_true.iter().zip(&chi).fold((0.0, 0usize), |(s, c), (t, r)| {
            if *t > 0.0 { (s + r, c + 1) } else { (s, c) }
        });
        let mean = sum / count as f64;
        assert!(
            (0.04..0.12).contains(&mean),
            "sphere mean {mean} is outside the expected range for a min-norm solve"
        );

        let n_vox = chi.len() as f64;
        let mean_true: f64 = chi_true.iter().sum::<f64>() / n_vox;
        let mean_rec: f64 = chi.iter().sum::<f64>() / n_vox;
        let (mut cov, mut var_t, mut var_r) = (0.0, 0.0, 0.0);
        for (t, r) in chi_true.iter().zip(&chi) {
            let (a, b) = (t - mean_true, r - mean_rec);
            cov += a * b;
            var_t += a * a;
            var_r += b * b;
        }
        let corr = cov / (var_t.sqrt() * var_r.sqrt());
        assert!(corr > 0.8, "recovered map correlates only {corr} with the truth");
    }

    /// Upstream's grid: the mask's bounding box, made even, plus 39 voxels, so always odd.
    #[test]
    fn crop_grid_matches_prepareforconvolution() {
        let dims = (20, 21, 22);
        let mut mask = vec![0u8; 20 * 21 * 22];
        // box x 3..=9 (7), y 2..=11 (10), z 5..=5 (1)
        for j in 2..=11 {
            for i in 3..=9 {
                mask[i + 20 * (j + 21 * 5)] = 1;
            }
        }
        let c = crop_grid(&mask, dims).unwrap();
        assert_eq!(c, [(3, 7, 47), (2, 10, 49), (5, 1, 41)]);
        assert!(crop_grid(&vec![0u8; 20 * 21 * 22], dims).is_none());
        let x: Vec<f64> = (0..mask.len()).map(|i| i as f64).collect();
        let back = uncrop(&crop_pad(&x, dims, &c), dims, &c);
        for (i, (&a, &b)) in x.iter().zip(&back).enumerate() {
            let (ii, jj, kk) = (i % 20, (i / 20) % 21, i / 420);
            let inside = (3..=9).contains(&ii) && (2..=11).contains(&jj) && kk == 5;
            assert_eq!(b, if inside { a } else { 0.0 });
        }
    }

    /// The spatial-domain kernel is real, zero at DC, symmetric, and close to the k-space
    /// formula 1/3 - (k·b)²/k² away from the highest frequencies.
    #[test]
    fn spatial_dipole_kernel_approximates_the_kspace_kernel() {
        let n = 33;
        let g = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let d = spatial_dipole_kernel(&g, (0.0, 0.0, 1.0));
        assert_eq!(d[0], 0.0);
        let at = |i: usize, j: usize, k: usize| d[i + n * (j + n * k)];
        // symmetric under k -> -k
        assert!((at(3, 5, 7) - at(n - 3, n - 5, n - 7)).abs() < 1e-12);
        // along B0 (kz axis) ~ -2/3, perpendicular ~ 1/3, at low frequencies
        assert!((at(0, 0, 2) + 2.0 / 3.0).abs() < 0.05, "{}", at(0, 0, 2));
        assert!((at(2, 0, 0) - 1.0 / 3.0).abs() < 0.05, "{}", at(2, 0, 0));
        // an anisotropic grid is built on a finer isotropic one: 2*floor(9*2/2)+1 = 19 along z
        let ga = Grid::new(9, 9, 9, 1.0, 1.0, 2.0);
        let da = spatial_dipole_kernel(&ga, (0.0, 0.0, 1.0));
        assert_eq!(da.len(), 729);
        assert_eq!(da[0], 0.0);
        assert!(da.iter().all(|v| v.is_finite()));
    }

    #[test]
    #[should_panic(expected = "odd extents")]
    fn spatial_dipole_kernel_rejects_even_grids() {
        spatial_dipole_kernel(&Grid::new(8, 9, 9, 1.0, 1.0, 1.0), (0.0, 0.0, 1.0));
    }

    /// The magnitude weight is used as is; its scale does not change the solution (a power of
    /// two keeps the floating-point arithmetic exact, so the maps are bit-identical).
    #[test]
    fn magnitude_scale_does_not_matter() {
        let n = 10;
        let g = test_grid(n);
        let nt = g.n_total();
        let field: Vec<f64> = (0..nt).map(|i| ((i as f64) * 0.37).sin() * 0.02).collect();
        let mag: Vec<f64> = (0..nt).map(|i| 1.0 + ((i as f64) * 0.11).cos().abs()).collect();
        let mag2: Vec<f64> = mag.iter().map(|v| v * 1024.0).collect();
        let mask = vec![1u8; nt];
        let p = LsqrQsmParams { max_iter: 15, ..Default::default() };
        let a = lsqr_qsm(&field, &mask, Some(&mag), &g, (0.0, 0.0, 1.0), &p, |_, _| {});
        let b = lsqr_qsm(&field, &mask, Some(&mag2), &g, (0.0, 0.0, 1.0), &p, |_, _| {});
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.to_bits(), y.to_bits(), "{x} vs {y}");
        }
    }

    #[test]
    fn residual_weighting_scales_with_field_strength() {
        let p = LsqrQsmParams { b0: 9.4, residual_weighting: None, ..Default::default() };
        assert!((p.effective_residual_weighting() - 0.2).abs() < 1e-12);

        let p3 = LsqrQsmParams { b0: 3.0, residual_weighting: None, ..Default::default() };
        assert!((p3.effective_residual_weighting() - 0.2 / 9.4 * 3.0).abs() < 1e-12);

        let fixed = LsqrQsmParams { residual_weighting: Some(0.5), ..Default::default() };
        assert_eq!(fixed.effective_residual_weighting(), 0.5);
    }
}

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
//! With χ over the whole grid, `r` and the data over the mask only:
//!
//! ```text
//! minimise ‖ W · ( M[ F⁻¹(D · F χ) + c ] + w·r − b ) ‖₂   over  [χ; r; c]
//! ```
//!
//! `W` is an optional per-voxel row weight (the magnitude image, SNR weighting)
//! and `M` restricts to the mask.
//!
//! This map is what HEIDI consumes: [`super::heidi`] keeps its well-conditioned
//! k-space coefficients and re-derives the rest under a weighted-TV prior.
//!
//! Reference:
//! Schweser, F., Deistung, A., Lehr, B.W., Reichenbach, J.R. (2010).
//! "Differentiation between diamagnetic and paramagnetic cerebral lesions based
//! on magnetic susceptibility mapping." Medical Physics, 37(10):5165-5178.
//! https://doi.org/10.1118/1.3481505

use std::cell::RefCell;

use num_complex::Complex64;

use crate::fft::Fft3dWorkspace;
use crate::kernels::dipole::dipole_kernel;
use crate::utils::apply_mask_zero;
use crate::utils::padding::{next_fast_fft_size, pad3d, unpad3d};
use crate::Grid;

use super::ilsqr::lsqr;

/// Residual weighting per tesla, from the QUASAR paper's optimum of 0.2 at 9.4 T.
pub const RESIDUAL_WEIGHTING_PER_TESLA: f64 = 0.2 / 9.4;

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
    ///
    /// This term is the only thing regularising the solve: with `w = 0` the
    /// masked inverse is badly ill-posed and the map streaks severely. The
    /// trade-off runs the other way for the χ that HEIDI keeps, though — a
    /// smaller `w` leaves more real signal in χ rather than in `r`, and HEIDI
    /// discards the streak-ridden cone anyway.
    pub residual_weighting: Option<f64>,
    /// Fit a single global field offset alongside χ.
    pub fit_global_offset: bool,
    /// LSQR convergence tolerance.
    pub tol: f64,
    /// Maximum LSQR iterations.
    pub max_iter: usize,
    /// Zero the returned χ outside the mask. HEIDI wants this off — it low-passes
    /// χ through the dipole cone, and a hard mask edge rings.
    pub mask_output: bool,
}

impl Default for LsqrQsmParams {
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

/// Minimally regularised LSQR dipole inversion.
///
/// # Arguments
/// * `local_field` - Local field in ppm (`nx * ny * nz`, Fortran order)
/// * `mask` - Binary mask; the data equation is written over these voxels only
/// * `magnitude` - Optional magnitude image used as an SNR row weight. It is
///   normalised to unit mean inside the mask, which leaves the LSQR iterates
///   unchanged under a rescaling of the magnitude units but keeps `w` meaningful.
/// * `grid` - Volume grid
/// * `bdir` - B0 direction
/// * `params` - Algorithm parameters
/// * `progress` - Progress callback `(iteration, max_iter)`
///
/// # Returns
/// Susceptibility map in ppm.
pub fn lsqr_qsm(
    local_field: &[f64],
    mask: &[u8],
    magnitude: Option<&[f64]>,
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &LsqrQsmParams,
    progress: impl FnMut(usize, usize),
) -> Vec<f64> {
    // Every LSQR iteration is two FFT round trips, so pay a little extra volume
    // for a transform size that factors into small primes.
    let dims = grid.dims;
    let fast = (
        next_fast_fft_size(dims.0),
        next_fast_fft_size(dims.1),
        next_fast_fft_size(dims.2),
    );
    if fast == dims {
        return lsqr_qsm_core(local_field, mask, magnitude, grid, bdir, params, progress);
    }
    let (vsx, vsy, vsz) = grid.voxel_size;
    let pgrid = Grid::new(fast.0, fast.1, fast.2, vsx, vsy, vsz);
    let padded_mag = magnitude.map(|m| pad3d(m, dims, fast));
    let chi = lsqr_qsm_core(
        &pad3d(local_field, dims, fast),
        &pad3d(mask, dims, fast),
        padded_mag.as_deref(),
        &pgrid,
        bdir,
        params,
        progress,
    );
    unpad3d(&chi, fast, dims)
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
    assert_eq!(local_field.len(), n_total, "field length must match grid");
    assert_eq!(mask.len(), n_total, "mask length must match grid");

    // Voxels carrying a data row.
    let mask_idx: Vec<usize> = (0..n_total).filter(|&i| mask[i] != 0).collect();
    let m = mask_idx.len();
    if m == 0 {
        return vec![0.0; n_total];
    }

    let w_res = params.effective_residual_weighting();
    let use_residual = w_res != 0.0;
    let n_res = if use_residual { m } else { 0 };
    let n_offset = usize::from(params.fit_global_offset);
    let n_unknown = n_total + n_res + n_offset;

    // Row weights: magnitude normalised to unit mean over the mask.
    let row_weight: Vec<f64> = match magnitude {
        Some(mag) => {
            assert_eq!(mag.len(), n_total, "magnitude length must match grid");
            let sum: f64 = mask_idx.iter().map(|&i| mag[i]).sum();
            let scale = if sum > 0.0 { m as f64 / sum } else { 1.0 };
            mask_idx.iter().map(|&i| mag[i] * scale).collect()
        }
        None => vec![1.0; m],
    };

    // Right-hand side: the field over the mask, row-weighted. Non-finite field
    // values are dropped rather than poisoning the whole solve.
    let b: Vec<f64> = mask_idx
        .iter()
        .zip(&row_weight)
        .map(|(&i, &w)| {
            let v = local_field[i];
            if v.is_finite() { v * w } else { 0.0 }
        })
        .collect();

    let d = dipole_kernel(grid, bdir);

    // LSQR's operators are `Fn`, so the mutable FFT workspace and the progress
    // callback live behind RefCells. One `apply_a` call per LSQR iteration makes
    // that the natural place to report progress.
    let fft = RefCell::new(Fft3dWorkspace::new(nx, ny, nz));
    let progress = RefCell::new(progress);
    let iter_count = std::cell::Cell::new(0usize);
    let max_iter = params.max_iter;

    let apply_dipole = |x: &[f64]| -> Vec<f64> {
        let mut buf: Vec<Complex64> = x.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        let mut ws = fft.borrow_mut();
        ws.fft3d(&mut buf);
        for (b, &dk) in buf.iter_mut().zip(&d) {
            *b *= dk;
        }
        ws.ifft3d(&mut buf);
        buf.iter().map(|c| c.re).collect()
    };

    let apply_a = |x: &[f64]| -> Vec<f64> {
        let n = iter_count.get() + 1;
        iter_count.set(n);
        (progress.borrow_mut())(n.min(max_iter), max_iter);

        let field = apply_dipole(&x[..n_total]);
        let offset = if n_offset == 1 { x[n_total + n_res] } else { 0.0 };

        let mut out = vec![0.0; m];
        for (row, (&idx, &w)) in mask_idx.iter().zip(&row_weight).enumerate() {
            let mut v = field[idx] + offset;
            if use_residual {
                v += w_res * x[n_total + row];
            }
            out[row] = w * v;
        }
        out
    };

    let apply_at = |y: &[f64]| -> Vec<f64> {
        // Scatter the weighted residual back onto the grid.
        let mut scattered = vec![0.0; n_total];
        for (row, (&idx, &w)) in mask_idx.iter().zip(&row_weight).enumerate() {
            scattered[idx] = y[row] * w;
        }

        let mut out = vec![0.0; n_unknown];

        // The dipole kernel is real and even, so it is its own adjoint.
        let chi_part = apply_dipole(&scattered);
        out[..n_total].copy_from_slice(&chi_part);

        if use_residual {
            for (row, &idx) in mask_idx.iter().enumerate() {
                out[n_total + row] = scattered[idx] * w_res;
            }
        }
        if n_offset == 1 {
            out[n_total + n_res] = mask_idx.iter().map(|&i| scattered[i]).sum();
        }
        out
    };

    let solution = lsqr(apply_a, apply_at, &b, params.tol, params.max_iter);

    let mut chi = solution[..n_total].to_vec();
    if params.mask_output {
        apply_mask_zero(&mut chi, mask);
    }
    chi
}

#[cfg(test)]
mod tests {
    use super::*;

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

//! Projection onto Dipole Fields (PDF) background field removal, a port of `PDF.m` from the MEDI
//! toolbox (release 2024.11.26)
//!
//! PDF fits the field inside the mask with the field of dipoles outside it, by weighted least
//! squares,
//!
//! min_x || W (f − D M_bg x) ||²₂  over the mask,  W = 1 / N_std,
//!
//! and returns what the fit does not explain, `(f − D M_bg x)` in the mask, as the local field.
//! `D` is the dipole kernel, `M_bg` restricts the sources to outside the mask, and `N_std` is the
//! noise standard deviation of the field.
//!
//! The port follows `PDF.m` and its helpers line for line:
//!
//! - the volume is cropped to the bounding box of the mask and zero padded after it to
//!   `floor((extent − 1 + n_pad) / 2) · 2` voxels per axis (`n_pad` = 40), so the background
//!   sources live in the box plus the padding, and the result is put back into the full volume;
//! - `W = 1/N_std`, set to 0 where `N_std` is 0, and to 0 outside the mask;
//! - the normal equations `M_bg D W² D M_bg x = M_bg D W² f` (`dipole_term.m`) are solved by
//!   conjugate gradients (`cgsolve.m`, starting from 0, at most `n_cg` = 30 iterations,
//!   returning the last iterate; more iterations are a tuning option, see
//!   [`PdfParams::n_cg`]), stopping when the relative residual falls below
//!   `tol · ||D W outside the mask|| / ||M_bg D W² f||` with `tol` = 0.1;
//! - the dipole kernel is `dipole_kernel.m`'s k-space kernel when B0 is along an axis
//!   (`norm(B0_dir, 1) < 1.01`). For an oblique B0 it is the kernel named by `space`, by default
//!   `dipole_kernel.m`'s `'imagespace'` kernel, the FFT of the sampled dipole field
//!   `(3 (r·B0)² − |r|²) / (4π |r|⁵)` (0 at r = 0).
//!
//! # Units and the noise map
//! Like the other background removals, `pdf` takes the total field in ppm and returns the local
//! field in ppm. `PDF.m` works in whatever unit the field is in (MEDI's pipeline passes radians
//! over the echo spacing), and the weighting is relative, but the absolute scale of `N_std`
//! still matters: the CG tolerance is `tol · ||D W|| / ||D W² f||`, which scales with
//! `N_std / f`. So `n_std` must be in the same unit as the field, here ppm. MEDI's pipeline
//! (README.m) takes `N_std` from `Fit_ppm_complex.m`
//! ([`fit_ppm_complex_noise`](crate::utils::fit_ppm_complex_noise), radians over the echo
//! spacing); divide it by `2π · γ · B0 · ΔTE · 1e-6` for ppm, as the pipeline does.
//!
//! # Deviation from `PDF.m`
//! With `n_pad = 0`, an axis with an odd number of voxels is not padded to an even size, and
//! `dipole_kernel.m` then builds its grid as `-N/2:N/2-1`, half a voxel off the sample points.
//! `pdf` uses the kernel centred on the voxel grid instead. With `n_pad > 0` (the default)
//! every axis is even and the kernels are `dipole_kernel.m`'s.
//!
//! # Reference pipeline
//! MEDI's pipeline script (`README.m` of the 2024.11.26 release, line 124) runs
//! `PDF(iFreq, N_std, Mask, matrix_size, voxel_size, B0_dir)` with `PDF.m`'s defaults (`tol`
//! 0.1, `n_CG` 30, `'imagespace'`, `n_pad` 40) on the field from `Fit_ppm_complex` unwrapped by
//! `unwrapPhase`, inside the BET mask of the root-sum-of-squares magnitude (README.m:118, BET
//! defaults), and passes the local field to `MEDI_L1`. The mSMV authors (same group) run
//! `PDF(..., tol = 0, n_CG = 100)` after ROMEO unwrapping (github.com/agr78/mSMV, README.md
//! and `data/healthy_subjects/proc.m:15`). Nothing in `PDF.m` depends on the field strength.
//!
//! Reference:
//! Liu, T., Khalidov, I., de Rochefort, L., Spincemaille, P., Liu, J., Tsiouris, A.J.,
//! Wang, Y. (2011). "A novel background field removal method for MRI using projection
//! onto dipole fields." NMR in Biomedicine, 24(9):1129-1136. https://doi.org/10.1002/nbm.1670
//!
//! de Rochefort, L., Liu, T., Kressler, B., Liu, J., Spincemaille, P., Lebon, V., Wu, J.,
//! Wang, Y. (2010). "Quantitative susceptibility map reconstruction from MR phase data using
//! bayesian regularization: validation and application to brain imaging." Magnetic Resonance
//! in Medicine, 63(1):194-206.
//!
//! Reference implementation: MEDI toolbox release 2024.11.26 (Cornell MRI Research Lab,
//! `functions/PDF.m`, `dipole_term.m`, `cgsolve.m`, `dipole_kernel.m`)
//!
//! # Pipeline rules
//!
//! The facts above that a configuration can depart from are published as rules in
//! [`crate::pipeline::rules::RULES`] (each with its strength, basis, reason and source): `pdf.mask-bet`, `pdf.field-complex-fit`, and the measured suggestion `pdf.cg-iterations`
//! (see [`PdfParams::n_cg`]).

use num_complex::Complex64;
use crate::error::{check_len, InputError};
use crate::fft::Fft3dWorkspace;
use crate::kernels::dipole::dipole_kernel;
use crate::Grid;

/// Which dipole kernel `pdf` uses when B0 is oblique (`PDF.m`'s `space` argument). With B0
/// along an axis the k-space kernel is used whatever this says, as in `PDF.m`.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DipoleKernelSpace {
    /// The FFT of the dipole field sampled on the voxel grid (`'imagespace'`)
    #[default]
    ImageSpace,
    /// The analytic k-space kernel `1/3 − (k·B0)²/|k|²` (`'kspace'`)
    KSpace,
}

/// PDF parameters (`PDF.m` defaults)
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct PdfParams {
    /// Tolerance (`tol`): CG stops when its relative residual falls below
    /// `tol · ||D W outside the mask|| / ||M_bg D W² f||`
    pub tol: f64,
    /// Maximum CG iterations (`n_CG`), by default `PDF.m`'s 30 (its default argument, which
    /// MEDI's `README.m:124` pipeline uses). Decided 2026-10-30: the default follows the
    /// reference, and the iteration count is tuning. Until then QSM.rs defaulted to 300.
    ///
    /// More iterations measured better, so 300 is suggested by the rule `pdf.cg-iterations`
    /// (Recommended, measured; [`crate::pipeline::rules::RULES`]). At 30 the solve can stop far
    /// from convergence:
    /// - on a Philips 3 T travelling-head scan the local field kept a large residual background
    ///   (correlation with V-SHARP 0.54 at 30 iterations, 0.83 at 300), while a GE scan was
    ///   unaffected (0.85 at 30, 0.84 at 300); neither scan reached `tol` within 300;
    /// - on QSM-CI's simulated head (QSM Challenge 2.0, 7 T), MEDI's own `PDF.m` at its
    ///   defaults (the `pdf-cornell` container, identical to licensed MATLAB) scored about
    ///   100 % NRMSE against the true local field, and QSM.rs's PDF at 300 iterations 44 %
    ///   (QSMxT/QSM-CI#322).
    ///
    /// 300 costs about 30 s more per 1 mm whole-brain scan. The mSMV authors run 100 (with
    /// `tol` 0; see the module docs, Reference pipeline).
    pub n_cg: usize,
    /// Kernel for an oblique B0 (`space`)
    pub space: DipoleKernelSpace,
    /// Zero padding around the mask's bounding box (`n_pad`), in voxels per axis; 0 runs on the
    /// full volume. 1 is refused, as `PDF.m` cannot run with it on odd extents.
    pub n_pad: usize,
}

impl Default for PdfParams {
    fn default() -> Self {
        Self { tol: 0.1, n_cg: 30, space: DipoleKernelSpace::ImageSpace, n_pad: 40 }
    }
}

/// PDF background field removal (`PDF.m`)
///
/// # Arguments
/// * `field` - Unwrapped total field in ppm (nx * ny * nz)
/// * `n_std` - Noise standard deviation of the field, in ppm (see module docs); 0 = no weight
/// * `mask` - Binary mask (nx * ny * nz), 1 = brain
/// * `grid` - Volume dimensions and voxel sizes
/// * `bdir` - B0 field direction (normalised here)
/// * `params` - PDF parameters
/// * `progress` - Progress callback `(cg_iteration, n_cg)`
///
/// # Returns
/// Local field in ppm, zero outside the mask
pub fn pdf(
    field: &[f64],
    n_std: &[f64],
    mask: &[u8],
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &PdfParams,
    mut progress: impl FnMut(usize, usize),
) -> Result<Vec<f64>, InputError> {
    let (nx, ny, nz) = grid.dims;
    let n = grid.n_total();
    check_len("field", field.len(), n)?;
    check_len("n_std", n_std.len(), n)?;
    check_len("mask", mask.len(), n)?;
    if params.n_pad == 1 {
        return Err(InputError::new("PDF n_pad must be 0 or at least 2"));
    }
    let bnorm = (bdir.0 * bdir.0 + bdir.1 * bdir.1 + bdir.2 * bdir.2).sqrt();
    if !(bnorm.is_finite() && bnorm > 0.0) {
        return Err(InputError::new(format!("PDF B0 direction {bdir:?} has no length")));
    }
    let b = (bdir.0 / bnorm, bdir.1 / bnorm, bdir.2 / bnorm);

    // d1first:d1last etc.: the bounding box of the mask
    let (mut lo, mut hi) = ([usize::MAX; 3], [0usize; 3]);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                if mask[i + nx * (j + ny * k)] != 0 {
                    for (a, v) in [i, j, k].into_iter().enumerate() {
                        lo[a] = lo[a].min(v);
                        hi[a] = hi[a].max(v);
                    }
                }
            }
        }
    }
    if lo[0] == usize::MAX {
        return Err(InputError::new("PDF mask is empty"));
    }

    // Crop to the box and pad after it to floor((last - first + n_pad)/2)*2, or keep the
    // whole volume when n_pad = 0.
    let (origin, ext, dims) = if params.n_pad > 0 {
        let ext = [hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1];
        let m = ext.map(|e| (e - 1 + params.n_pad) / 2 * 2);
        (lo, ext, m)
    } else {
        ([0; 3], [nx, ny, nz], [nx, ny, nz])
    };
    let [mx, my, mz] = dims;
    let m_total = mx * my * mz;
    let crop = |v: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0; m_total];
        for k in 0..ext[2] {
            for j in 0..ext[1] {
                let src = origin[0] + nx * (origin[1] + j + ny * (origin[2] + k));
                let dst = mx * (j + my * k);
                out[dst..dst + ext[0]].copy_from_slice(&v[src..src + ext[0]]);
            }
        }
        out
    };
    let f = crop(field);
    let noise = crop(n_std);
    let mask_f: Vec<f64> = mask.iter().map(|&m| f64::from(u8::from(m != 0))).collect();
    let inside: Vec<bool> = crop(&mask_f).iter().map(|&m| m > 0.0).collect();

    // W = 1./N_std; W(isinf(W)) = 0; W = W.*(Mask>0); W_var = W.^2
    let w: Vec<f64> = (0..m_total)
        .map(|i| {
            let v = 1.0 / noise[i];
            if !inside[i] || v.is_infinite() { 0.0 } else { v }
        })
        .collect();
    let w_var: Vec<f64> = w.iter().map(|&v| v * v).collect();

    let pgrid = Grid::new(mx, my, mz, grid.vsx(), grid.vsy(), grid.vsz());
    let oblique = b.0.abs() + b.1.abs() + b.2.abs() >= 1.01;
    let d: Vec<Complex64> = if oblique && params.space == DipoleKernelSpace::ImageSpace {
        image_space_dipole_kernel(&pgrid, b)
    } else {
        dipole_kernel(&pgrid, b).into_iter().map(|v| Complex64::new(v, 0.0)).collect()
    };

    let mut fft = Fft3dWorkspace::new(mx, my, mz);
    let mut buf = vec![Complex64::new(0.0, 0.0); m_total];
    // real(ifftn(D.*fftn(x)))
    let mut conv = |x: &[f64], out: &mut [f64]| {
        for (c, &v) in buf.iter_mut().zip(x) {
            *c = Complex64::new(v, 0.0);
        }
        fft.fft3d(&mut buf);
        for (c, dk) in buf.iter_mut().zip(&d) {
            *c *= dk;
        }
        fft.ifft3d(&mut buf);
        for (o, c) in out.iter_mut().zip(&buf) {
            *o = c.re;
        }
    };

    // b = real(ifftn(D.*fftn(W_var.*iFreq))) outside the mask. Vectors over the background
    // voxels are kept as full volumes that are zero inside the mask.
    let mut tmp: Vec<f64> = (0..m_total).map(|i| w_var[i] * f[i]).collect();
    let mut rhs = vec![0.0; m_total];
    conv(&tmp, &mut rhs);
    zero_inside(&mut rhs, &inside);

    // E_noise_level = real(ifftn(D.*fftn(W_std))); cg_tol = tol*norm(E outside)/norm(b)
    let mut e = vec![0.0; m_total];
    conv(&w, &mut e);
    zero_inside(&mut e, &inside);
    let cg_tol = params.tol * dot(&e, &e).sqrt() / dot(&rhs, &rhs).sqrt();

    // dipole_term: Ax = real(ifftn(D.*fftn(W_var.*real(ifftn(D.*fftn(x)))))) outside the mask
    let mut tmp2 = vec![0.0; m_total];
    let mut apply = |x: &[f64], out: &mut [f64]| {
        conv(x, &mut tmp);
        for i in 0..m_total {
            tmp[i] *= w_var[i];
        }
        conv(&tmp, &mut tmp2);
        out.copy_from_slice(&tmp2);
        zero_inside(out, &inside);
    };
    let x = cgsolve(&mut apply, &rhs, cg_tol, params.n_cg, &mut progress);

    // p_dipole = real(ifftn(D.*fftn(xx))); p_final = (iFreq - p_dipole).*Mask
    let mut p_dipole = vec![0.0; m_total];
    conv(&x, &mut p_dipole);
    let mut local = vec![0.0; n];
    for k in 0..ext[2] {
        for j in 0..ext[1] {
            for i in 0..ext[0] {
                let src = i + mx * (j + my * k);
                if inside[src] {
                    let dst = origin[0] + i + nx * (origin[1] + j + ny * (origin[2] + k));
                    local[dst] = f[src] - p_dipole[src];
                }
            }
        }
    }
    Ok(local)
}

fn zero_inside(v: &mut [f64], inside: &[bool]) {
    for (x, &m) in v.iter_mut().zip(inside) {
        if m {
            *x = 0.0;
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `cgsolve.m` (Romberg; as modified for MEDI) from x0 = 0: conjugate gradients on `A x = b`,
/// stopping after `maxiter` iterations or when `||r|| / ||b|| <= tol`, with the residual
/// recomputed from `x` every 50 iterations. Returns the last iterate, as `cgsolve.m` does.
fn cgsolve(
    a: &mut impl FnMut(&[f64], &mut [f64]),
    b: &[f64],
    tol: f64,
    maxiter: usize,
    progress: &mut impl FnMut(usize, usize),
) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    let mut r = b.to_vec();
    let mut d = r.clone();
    let mut q = vec![0.0; n];
    let mut delta = dot(&r, &r);
    let delta0 = dot(b, b);
    let mut numiter = 0;
    while numiter < maxiter && delta > tol * tol * delta0 {
        a(&d, &mut q);
        let alpha = delta / dot(&d, &q);
        for i in 0..n {
            x[i] += alpha * d[i];
        }
        if (numiter + 1) % 50 == 0 {
            a(&x, &mut q);
            for i in 0..n {
                r[i] = b[i] - q[i];
            }
        } else {
            for i in 0..n {
                r[i] -= alpha * q[i];
            }
        }
        let deltaold = delta;
        delta = dot(&r, &r);
        let beta = delta / deltaold;
        for i in 0..n {
            d[i] = r[i] + beta * d[i];
        }
        numiter += 1;
        progress(numiter, maxiter);
    }
    x
}

/// `dipole_kernel.m`'s `'imagespace'` kernel: the dipole field
/// `(3 (r·B0)² − |r|²) / (4π |r|⁵)` sampled at the voxel positions r (mm), 0 at r = 0, with the
/// origin at index 0 (`fftshift` of the `-N/2:N/2-1` grid), Fourier transformed.
fn image_space_dipole_kernel(grid: &Grid, b: (f64, f64, f64)) -> Vec<Complex64> {
    let (nx, ny, nz) = grid.dims;
    // Position of index j on the wrapped grid: j for j < ceil(n/2), else j - n. For even n this
    // is fftshift(-n/2:n/2-1); for odd n it is centred on the voxels (see module docs).
    let coord = |j: usize, n: usize, vs: f64| -> f64 {
        let c = if j < n.div_ceil(2) { j as f64 } else { j as f64 - n as f64 };
        c * vs
    };
    let mut d = vec![Complex64::new(0.0, 0.0); nx * ny * nz];
    for k in 0..nz {
        let z = coord(k, nz, grid.vsz());
        for j in 0..ny {
            let y = coord(j, ny, grid.vsy());
            for i in 0..nx {
                let x = coord(i, nx, grid.vsx());
                let r2 = x * x + y * y + z * z;
                if r2 > 0.0 {
                    let rb = x * b.0 + y * b.1 + z * b.2;
                    let v = (3.0 * rb * rb - r2) / (4.0 * std::f64::consts::PI * r2.powf(2.5));
                    d[i + nx * (j + ny * k)] = Complex64::new(v, 0.0);
                }
            }
        }
    }
    Fft3dWorkspace::new(nx, ny, nz).fft3d(&mut d);
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(n: usize, c: (f64, f64, f64), r: f64) -> Vec<u8> {
        let mut m = vec![0u8; n * n * n];
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let (dx, dy, dz) = (i as f64 - c.0, j as f64 - c.1, k as f64 - c.2);
                    m[i + n * (j + n * k)] = u8::from(dx * dx + dy * dy + dz * dz <= r * r);
                }
            }
        }
        m
    }

    /// Field of a susceptibility distribution on an `n`³ grid (k-space kernel, B0 along z).
    fn forward(chi: &[f64], n: usize) -> Vec<f64> {
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let d = dipole_kernel(&grid, (0.0, 0.0, 1.0));
        let mut c: Vec<Complex64> = chi.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        let mut fft = Fft3dWorkspace::new(n, n, n);
        fft.fft3d(&mut c);
        for (v, dk) in c.iter_mut().zip(&d) {
            *v *= dk;
        }
        fft.ifft3d(&mut c);
        c.iter().map(|v| v.re).collect()
    }

    fn params() -> PdfParams {
        PdfParams { n_pad: 8, ..PdfParams::default() }
    }

    fn maxdiff(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
    }

    #[test]
    fn zero_field_gives_zero() {
        let n = 12;
        let mask = sphere(n, (6.0, 6.0, 6.0), 3.0);
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let out = pdf(&vec![0.0; n * n * n], &vec![1.0; n * n * n], &mask, &grid, (0.0, 0.0, 1.0), &params(), |_, _| {}).unwrap();
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn removes_a_background_source_and_zeroes_outside() {
        // A source outside the mask: its field inside the mask is background, and PDF should
        // remove most of it.
        let n = 24;
        let mask = sphere(n, (12.0, 12.0, 12.0), 5.0);
        let mut chi = vec![0.0; n * n * n];
        for k in 2..5 {
            for j in 10..14 {
                for i in 10..14 {
                    chi[i + n * (j + n * k)] = 1.0;
                }
            }
        }
        let field = forward(&chi, n);
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let p = PdfParams { tol: 1e-6, n_cg: 300, ..params() };
        let out = pdf(&field, &vec![1.0; n * n * n], &mask, &grid, (0.0, 0.0, 1.0), &p, |_, _| {}).unwrap();
        let (mut before, mut after) = (0.0, 0.0);
        for i in 0..n * n * n {
            if mask[i] != 0 {
                before += field[i] * field[i];
                after += out[i] * out[i];
            } else {
                assert_eq!(out[i], 0.0);
            }
        }
        assert!(after < 0.05 * before, "background energy {before} -> {after}");
    }

    #[test]
    fn unweighted_voxels_do_not_affect_the_fit() {
        // N_std = 0 gives W = 0 (1/0 = inf -> 0), so the field there does not enter the fit:
        // changing it changes the output at that voxel only.
        let n = 16;
        let mask = sphere(n, (8.0, 8.0, 8.0), 4.0);
        let field: Vec<f64> = (0..n * n * n).map(|i| ((i * 7919) % 101) as f64 * 1e-3).collect();
        let mut noise = vec![1.0; n * n * n];
        let c = 8 + n * (8 + n * 8);
        noise[c] = 0.0;
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let run = |f: &[f64]| pdf(f, &noise, &mask, &grid, (0.0, 0.0, 1.0), &params(), |_, _| {}).unwrap();
        let a = run(&field);
        let mut field2 = field.clone();
        field2[c] += 5.0;
        let b = run(&field2);
        for i in 0..n * n * n {
            let expect = if i == c { 5.0 } else { 0.0 };
            assert!((b[i] - a[i] - expect).abs() < 1e-9, "voxel {i}");
        }
    }

    #[test]
    fn noise_scale_sets_the_stopping_point_only() {
        // The weights are relative, but the CG tolerance scales with N_std / field. With the
        // tolerance out of play (all n_cg iterations run) the result does not depend on the
        // scale of N_std.
        let n = 16;
        let mask = sphere(n, (8.0, 8.0, 8.0), 4.0);
        let field: Vec<f64> = (0..n * n * n).map(|i| ((i * 31) % 17) as f64 * 1e-2).collect();
        let noise: Vec<f64> = (0..n * n * n).map(|i| 1.0 + (i % 5) as f64 * 0.1).collect();
        let noise3: Vec<f64> = noise.iter().map(|v| v * 3.0).collect();
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let p = PdfParams { tol: 0.0, n_cg: 10, ..params() };
        let a = pdf(&field, &noise, &mask, &grid, (0.0, 0.0, 1.0), &p, |_, _| {}).unwrap();
        let b = pdf(&field, &noise3, &mask, &grid, (0.0, 0.0, 1.0), &p, |_, _| {}).unwrap();
        assert!(maxdiff(&a, &b) < 1e-9);
        // With the default tolerance, a larger N_std stops CG earlier.
        let (mut ia, mut ib) = (0, 0);
        pdf(&field, &noise, &mask, &grid, (0.0, 0.0, 1.0), &params(), |i, _| ia = i).unwrap();
        pdf(&field, &noise3, &mask, &grid, (0.0, 0.0, 1.0), &params(), |i, _| ib = i).unwrap();
        assert!(ib <= ia, "{ib} > {ia}");
    }

    #[test]
    fn result_follows_the_mask_not_the_volume() {
        // PDF crops to the mask's bounding box, so moving the volume around the mask gives the
        // same local field, moved.
        let n = 16;
        let mask = sphere(n, (7.0, 7.0, 7.0), 3.0);
        let field: Vec<f64> = (0..n * n * n).map(|i| ((i * 13) % 23) as f64 * 1e-2).collect();
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let a = pdf(&field, &vec![1.0; n * n * n], &mask, &grid, (0.0, 0.0, 1.0), &params(), |_, _| {}).unwrap();
        let sh = |v: &[f64]| -> Vec<f64> {
            let mut o = vec![0.0; n * n * n];
            for k in 0..n - 2 {
                for j in 0..n - 1 {
                    for i in 0..n - 3 {
                        o[(i + 3) + n * ((j + 1) + n * (k + 2))] = v[i + n * (j + n * k)];
                    }
                }
            }
            o
        };
        let mask_s: Vec<u8> = sh(&mask.iter().map(|&m| m as f64).collect::<Vec<_>>()).iter().map(|&v| v as u8).collect();
        let b = pdf(&sh(&field), &vec![1.0; n * n * n], &mask_s, &grid, (0.0, 0.0, 1.0), &params(), |_, _| {}).unwrap();
        assert!(maxdiff(&sh(&a), &b) < 1e-12);
    }

    #[test]
    fn oblique_b0_uses_the_chosen_kernel_axial_does_not() {
        let n = 16;
        let mask = sphere(n, (8.0, 8.0, 8.0), 4.0);
        let field: Vec<f64> = (0..n * n * n).map(|i| ((i * 31) % 17) as f64 * 1e-2).collect();
        // low noise, so CG runs (with N_std ~ the field, the tolerance stops it at once)
        let noise = vec![1e-4; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let run = |b, space| {
            let p = PdfParams { space, ..params() };
            pdf(&field, &noise, &mask, &grid, b, &p, |_, _| {}).unwrap()
        };
        let axial = (0.0, 0.0, 1.0);
        assert_eq!(maxdiff(&run(axial, DipoleKernelSpace::ImageSpace), &run(axial, DipoleKernelSpace::KSpace)), 0.0);
        let tilted = (0.1, 0.0, 1.0);
        assert!(maxdiff(&run(tilted, DipoleKernelSpace::ImageSpace), &run(tilted, DipoleKernelSpace::KSpace)) > 1e-6);
    }

    #[test]
    fn image_space_kernel_is_the_fft_of_the_sampled_dipole() {
        // fftn(fftshift(d)) on a 4x4x4 grid of 1 mm voxels, B0 along z: D at k = 0 is the sum
        // of the sampled field over the grid -2:1 per axis.
        let grid = Grid::new(4, 4, 4, 1.0, 1.0, 1.0);
        let d = image_space_dipole_kernel(&grid, (0.0, 0.0, 1.0));
        let mut sum = 0.0;
        for z in -2..2 {
            for y in -2..2 {
                for x in -2i32..2 {
                    let r2 = f64::from(x * x + y * y + z * z);
                    if r2 > 0.0 {
                        sum += (3.0 * f64::from(z * z) - r2) / (4.0 * std::f64::consts::PI * r2.powf(2.5));
                    }
                }
            }
        }
        assert!((d[0].re - sum).abs() < 1e-12);
    }

    #[test]
    fn cgsolve_solves_a_small_spd_system() {
        let m = [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
        let mut a = |x: &[f64], out: &mut [f64]| {
            for r in 0..3 {
                out[r] = (0..3).map(|c| m[r][c] * x[c]).sum();
            }
        };
        let b = [1.0, 2.0, 3.0];
        let x = cgsolve(&mut a, &b, 1e-12, 10, &mut |_, _| {});
        let mut ax = [0.0; 3];
        a(&x, &mut ax);
        for r in 0..3 {
            assert!((ax[r] - b[r]).abs() < 1e-10);
        }
    }

    #[test]
    fn refuses_bad_input() {
        let n = 8;
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let f = vec![0.0; n * n * n];
        let mask = sphere(n, (4.0, 4.0, 4.0), 2.0);
        let one = vec![1.0; n * n * n];
        let z = (0.0, 0.0, 1.0);
        assert!(pdf(&f, &one[1..], &mask, &grid, z, &params(), |_, _| {}).is_err());
        assert!(pdf(&f, &one, &vec![0u8; n * n * n], &grid, z, &params(), |_, _| {}).is_err());
        assert!(pdf(&f, &one, &mask, &grid, z, &PdfParams { n_pad: 1, ..params() }, |_, _| {}).is_err());
        assert!(pdf(&f, &one, &mask, &grid, (0.0, 0.0, 0.0), &params(), |_, _| {}).is_err());
    }
}

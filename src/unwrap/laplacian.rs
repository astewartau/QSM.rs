//! Laplacian-based phase unwrapping
//!
//! The Laplacian of the wrapped phase equals the Laplacian of the true phase wherever
//! neighbouring samples differ by less than π, so the true phase can be recovered by
//! solving a Poisson equation. Path-independent and fast, unlike region-growing methods.
//!
//! [`laplacian_unwrap`] unwraps only; the harmonic (background) component survives and the
//! result is a total field. It has two solvers, named as in QSM.jl's `unwrap_laplacian`:
//!
//! | [`LaplacianSolver`] | ∇²φ estimate | Poisson solve | Reference |
//! |---|---|---|---|
//! | [`Dct`](LaplacianSolver::Dct) (default) | finite differences of wrapped neighbour differences | DCT, Neumann boundary on the array | Ghiglia & Romero (1994); QSM.jl `:dct` |
//! | [`Fft { pad }`](LaplacianSolver::Fft) | `cos φ·∇²(sin φ) − sin φ·∇²(cos φ)`, spectral ∇² (`|k|²`) | FFT, periodic, on the volume zero-padded by `pad` | Schofield & Zhu (2003); STI Suite 3.0 `MRPhaseUnwrap` |
//!
//! `Dct` is unweighted least-squares unwrapping: the Poisson equation for the wrapped phase
//! differences, solved under a Neumann condition with a DCT (Ghiglia & Romero 1994). It
//! reproduces QSM.jl's `:dct` solver exactly (QSM.jl v0.5.4 on byte-identical input: r =
//! 1.000000, rms difference 0.0).
//!
//! `Fft` is the literal Schofield & Zhu method: the Laplacian of the true phase from the sin/cos
//! identity with spectral derivatives, then an FFT Poisson solve. It reproduces STI Suite 3.0's
//! `MRPhaseUnwrap(phase, 'voxelsize', vs, 'padsize', pad)`, which is this algorithm, to
//! ≈1e-14 rad inside the mask (pad 64; in-vivo 3 T data and synthetic volumes at several voxel
//! sizes). Steps: zero-pad by `pad` per side (an odd padded dimension gets one more zero plane at
//! its end); `K = |k|²` on the padded FFT grid, `k_d = m / (M_d·h_d)`, no 4π² factor;
//! `L = cos φ·F⁻¹[K·F(sin φ)] − sin φ·F⁻¹[K·F(cos φ)]`; `φ_u = F⁻¹[F(L)/K]` with DC → 0; crop.
//!
//! The two give nearly the same local field after background removal (simulated head: NRMSE
//! 79.5 % for `Fft` against 81.3 % for `Dct`; on a 150-subject in-vivo test-retest cohort the
//! same iron-region ICC, 0.924 against 0.925). They differ in the total field: zero padding puts
//! a step wherever the phase at the array edge is not zero, which `Fft` turns into a harmonic
//! error that `Dct` does not have. Hence `Dct` is the default.
//!
//! Multi-echo field maps are built by [`crate::pipeline::run_field_mapping`], whose
//! `laplacian_solver` setting selects the solver: each echo is masked, unwrapped and combined
//! with the configured [`B0WeightType`](crate::utils::B0WeightType). UK Biobank's field map is
//! `Fft { pad: [64; 3] }` with [`B0WeightType::AssumedDecay`](crate::utils::B0WeightType::AssumedDecay)
//! at 40 ms.
//!
//! [`UnwrapMethod::Laplacian`](super::UnwrapMethod::Laplacian) selects [`laplacian_unwrap`] with
//! [`LaplacianSolver::Dct`], since the pipeline removes background as a later stage.
//!
//! # References
//!
//! Ghiglia, D.C., Romero, L.A. (1994). "Robust two-dimensional weighted and unweighted phase
//! unwrapping that uses fast transforms and iterative methods." Journal of the Optical Society
//! of America A, 11(1):107-117. <https://doi.org/10.1364/JOSAA.11.000107>
//!
//! Schofield, M.A., Zhu, Y. (2003). "Fast phase unwrapping algorithm for interferometric
//! applications." Optics Letters, 28(14):1194-1196. <https://doi.org/10.1364/OL.28.001194>
//!
//! Li, W., Wu, B., Liu, C. (2011). "Quantitative susceptibility mapping of human brain reflects
//! spatial variation in tissue composition." NeuroImage, 55(4):1645-1656 (STI Suite's use of
//! the Schofield & Zhu method). <https://doi.org/10.1016/j.neuroimage.2010.11.088>
//!
//! Reference implementation: <https://github.com/kamesy/QSM.jl> (`unwrap_laplacian`).

use std::f64::consts::PI;
use num_complex::Complex64;
#[cfg(test)]
use crate::fft::{fft3d, ifft3d};
use crate::fft::{fftfreq, Fft3dWorkspace};
use crate::Grid;

/// Wrap angle to [-π, π]
#[inline]
pub(crate) fn wrap(x: f64) -> f64 {
    let mut y = x % (2.0 * PI);
    if y > PI {
        y -= 2.0 * PI;
    } else if y < -PI {
        y += 2.0 * PI;
    }
    y
}

/// Compute wrapped Laplacian of phase with periodic boundary conditions
///
/// Uses second-order central finite differences on wrapped phase differences.
#[cfg(test)]
pub(crate) fn wrapped_laplacian_periodic(
    phase: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> Vec<f64> {
    periodic_laplacian_of_differences(phase, nx, ny, nz, vsx, vsy, vsz, wrap)
}

/// Second-order central Laplacian with periodic boundaries, where each neighbour difference
/// is passed through `wrap` first (the 2π wrap for phase, the identity for a field).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn periodic_laplacian_of_differences(
    phase: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    wrap: impl Fn(f64) -> f64,
) -> Vec<f64> {
    let n_total = nx * ny * nz;
    let mut d2u = vec![0.0; n_total];

    let dx2 = 1.0 / (vsx * vsx);
    let dy2 = 1.0 / (vsy * vsy);
    let dz2 = 1.0 / (vsz * vsz);

    for k in 0..nz {
        let km1 = if k == 0 { nz - 1 } else { k - 1 };
        let kp1 = if k + 1 >= nz { 0 } else { k + 1 };

        for j in 0..ny {
            let jm1 = if j == 0 { ny - 1 } else { j - 1 };
            let jp1 = if j + 1 >= ny { 0 } else { j + 1 };

            for i in 0..nx {
                let im1 = if i == 0 { nx - 1 } else { i - 1 };
                let ip1 = if i + 1 >= nx { 0 } else { i + 1 };

                let idx = i + j * nx + k * nx * ny;
                let u_ijk = phase[idx];

                let idx_im1 = im1 + j * nx + k * nx * ny;
                let idx_ip1 = ip1 + j * nx + k * nx * ny;
                let idx_jm1 = i + jm1 * nx + k * nx * ny;
                let idx_jp1 = i + jp1 * nx + k * nx * ny;
                let idx_km1 = i + j * nx + km1 * nx * ny;
                let idx_kp1 = i + j * nx + kp1 * nx * ny;

                let lap_x = (wrap(phase[idx_ip1] - u_ijk) - wrap(u_ijk - phase[idx_im1])) * dx2;
                let lap_y = (wrap(phase[idx_jp1] - u_ijk) - wrap(u_ijk - phase[idx_jm1])) * dy2;
                let lap_z = (wrap(phase[idx_kp1] - u_ijk) - wrap(u_ijk - phase[idx_km1])) * dz2;

                d2u[idx] = lap_x + lap_y + lap_z;
            }
        }
    }

    d2u
}

/// Solve Poisson equation using FFT (periodic boundary conditions).
///
/// Only the test-only even-extension oracle uses this now; the library solves under
/// Neumann via [`solve_poisson_dct`].
#[cfg(test)]
pub(crate) fn solve_poisson_fft(
    f: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> Vec<f64> {
    let mut f_complex: Vec<Complex64> = f.iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();
    fft3d(&mut f_complex, nx, ny, nz);

    let idx2 = 1.0 / (vsx * vsx);
    let idy2 = 1.0 / (vsy * vsy);
    let idz2 = 1.0 / (vsz * vsz);

    for k in 0..nz {
        let fk = if k <= nz / 2 { k as f64 / nz as f64 } else { (k as f64 - nz as f64) / nz as f64 };
        let lam_z = 2.0 * ((2.0 * PI * fk).cos() - 1.0) * idz2;

        for j in 0..ny {
            let fj = if j <= ny / 2 { j as f64 / ny as f64 } else { (j as f64 - ny as f64) / ny as f64 };
            let lam_y = 2.0 * ((2.0 * PI * fj).cos() - 1.0) * idy2;

            for i in 0..nx {
                let fi = if i <= nx / 2 { i as f64 / nx as f64 } else { (i as f64 - nx as f64) / nx as f64 };
                let lam_x = 2.0 * ((2.0 * PI * fi).cos() - 1.0) * idx2;

                let lam = lam_x + lam_y + lam_z;
                let idx = i + j * nx + k * nx * ny;

                if lam.abs() > 1e-20 {
                    f_complex[idx] /= lam;
                } else {
                    f_complex[idx] = Complex64::new(0.0, 0.0);
                }
            }
        }
    }

    ifft3d(&mut f_complex, nx, ny, nz);
    f_complex.iter().map(|c| c.re).collect()
}

/// Wrapped Laplacian under a Neumann boundary: at each array face the missing neighbour is
/// the sample itself, so the wrapped difference across the face is zero. This is exactly
/// the half-sample even extension the DCT-II assumes, so pairing it with
/// [`solve_poisson_dct`] reproduces the even-extended periodic solve without building the
/// 2x-per-axis extension.
pub(crate) fn wrapped_laplacian_neumann(
    phase: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> Vec<f64> {
    let n_total = nx * ny * nz;
    let mut d2u = vec![0.0; n_total];
    let (dx2, dy2, dz2) = (1.0 / (vsx * vsx), 1.0 / (vsy * vsy), 1.0 / (vsz * vsz));

    for k in 0..nz {
        let km1 = k.saturating_sub(1);
        let kp1 = if k + 1 >= nz { k } else { k + 1 };
        for j in 0..ny {
            let jm1 = j.saturating_sub(1);
            let jp1 = if j + 1 >= ny { j } else { j + 1 };
            for i in 0..nx {
                let im1 = i.saturating_sub(1);
                let ip1 = if i + 1 >= nx { i } else { i + 1 };
                let idx = i + j * nx + k * nx * ny;
                let u = phase[idx];
                let lap_x = (wrap(phase[ip1 + j * nx + k * nx * ny] - u) - wrap(u - phase[im1 + j * nx + k * nx * ny])) * dx2;
                let lap_y = (wrap(phase[i + jp1 * nx + k * nx * ny] - u) - wrap(u - phase[i + jm1 * nx + k * nx * ny])) * dy2;
                let lap_z = (wrap(phase[i + j * nx + kp1 * nx * ny] - u) - wrap(u - phase[i + j * nx + km1 * nx * ny])) * dz2;
                d2u[idx] = lap_x + lap_y + lap_z;
            }
        }
    }
    d2u
}

/// Solve ∇²u = f under a Neumann boundary condition on the array, in place, via DCT-II.
///
/// The DCT-II of a length-N signal is the FFT of its even extension restricted to the
/// original samples, so this is the padded periodic solve with the extension never
/// materialised: working memory is the volume itself plus one axis-length buffer,
/// against 8x the volume in complex doubles for the explicit extension.
///
/// Eigenvalues of the second difference under this basis are `2(cos(πk/N) − 1)/h²`. The DC
/// mode is set to zero, as in the periodic solve — the result is defined up to a constant.
pub(crate) fn solve_poisson_dct(
    f: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> Vec<f64> {
    use rustdct::DctPlanner;
    let nxy = nx * ny;
    let mut data = f.to_vec();
    let mut planner = DctPlanner::<f64>::new();

    // Forward: DCT-II along each axis.
    let dct2_x = planner.plan_dct2(nx);
    let dct2_y = planner.plan_dct2(ny);
    let dct2_z = planner.plan_dct2(nz);
    {
        let mut scratch = vec![0.0; dct2_x.get_scratch_len()];
        for row in data.chunks_mut(nx) {
            dct2_x.process_dct2_with_scratch(row, &mut scratch);
        }
    }
    {
        let mut buf = vec![0.0; ny];
        let mut scratch = vec![0.0; dct2_y.get_scratch_len()];
        for k in 0..nz {
            for i in 0..nx {
                for j in 0..ny { buf[j] = data[i + j * nx + k * nxy]; }
                dct2_y.process_dct2_with_scratch(&mut buf, &mut scratch);
                for j in 0..ny { data[i + j * nx + k * nxy] = buf[j]; }
            }
        }
    }
    {
        let mut buf = vec![0.0; nz];
        let mut scratch = vec![0.0; dct2_z.get_scratch_len()];
        for j in 0..ny {
            for i in 0..nx {
                for k in 0..nz { buf[k] = data[i + j * nx + k * nxy]; }
                dct2_z.process_dct2_with_scratch(&mut buf, &mut scratch);
                for k in 0..nz { data[i + j * nx + k * nxy] = buf[k]; }
            }
        }
    }

    // Divide by the Laplacian eigenvalue of each DCT-II mode.
    let (ix2, iy2, iz2) = (1.0 / (vsx * vsx), 1.0 / (vsy * vsy), 1.0 / (vsz * vsz));
    let lam_x: Vec<f64> = (0..nx).map(|i| 2.0 * ((PI * i as f64 / nx as f64).cos() - 1.0) * ix2).collect();
    let lam_y: Vec<f64> = (0..ny).map(|j| 2.0 * ((PI * j as f64 / ny as f64).cos() - 1.0) * iy2).collect();
    let lam_z: Vec<f64> = (0..nz).map(|k| 2.0 * ((PI * k as f64 / nz as f64).cos() - 1.0) * iz2).collect();
    for (k, &lz) in lam_z.iter().enumerate() {
        for (j, &ly) in lam_y.iter().enumerate() {
            let row = j * nx + k * nxy;
            for (i, &lx) in lam_x.iter().enumerate() {
                let lam = lx + ly + lz;
                let idx = row + i;
                data[idx] = if lam.abs() > 1e-20 { data[idx] / lam } else { 0.0 };
            }
        }
    }

    // Inverse: DCT-III along each axis. Unnormalised DCT-III∘DCT-II scales by N/2 per axis.
    let dct3_x = planner.plan_dct3(nx);
    let dct3_y = planner.plan_dct3(ny);
    let dct3_z = planner.plan_dct3(nz);
    {
        let mut buf = vec![0.0; nz];
        let mut scratch = vec![0.0; dct3_z.get_scratch_len()];
        for j in 0..ny {
            for i in 0..nx {
                for k in 0..nz { buf[k] = data[i + j * nx + k * nxy]; }
                dct3_z.process_dct3_with_scratch(&mut buf, &mut scratch);
                for k in 0..nz { data[i + j * nx + k * nxy] = buf[k]; }
            }
        }
    }
    {
        let mut buf = vec![0.0; ny];
        let mut scratch = vec![0.0; dct3_y.get_scratch_len()];
        for k in 0..nz {
            for i in 0..nx {
                for j in 0..ny { buf[j] = data[i + j * nx + k * nxy]; }
                dct3_y.process_dct3_with_scratch(&mut buf, &mut scratch);
                for j in 0..ny { data[i + j * nx + k * nxy] = buf[j]; }
            }
        }
    }
    {
        let mut scratch = vec![0.0; dct3_x.get_scratch_len()];
        for row in data.chunks_mut(nx) {
            dct3_x.process_dct3_with_scratch(row, &mut scratch);
        }
    }
    let norm = 8.0 / (nx as f64 * ny as f64 * nz as f64);
    for v in data.iter_mut() { *v *= norm; }
    data
}

/// The original even-extension implementation of [`laplacian_unwrap`] (`Dct`), kept only as the
/// oracle for the DCT solve: the DCT-II is the FFT of the even extension, so the two must
/// agree to rounding. Not compiled into the library.
#[cfg(test)]
fn laplacian_unwrap_even_extended_reference(
    phase: &[f64],
    mask: &[u8],
    grid: &Grid,
) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let (vsx, vsy, vsz) = grid.voxel_size;
    let n_total = nx * ny * nz;

    // Even extension: continuous across the seam, so the periodic FFT solve realises a
    // Neumann condition on the original array instead of wrapping a discontinuity.
    let (px, py, pz) = (2 * nx, 2 * ny, 2 * nz);
    let mut ext = vec![0.0f64; px * py * pz];
    for k in 0..pz {
        let sk = if k < nz { k } else { 2 * nz - 1 - k };
        for j in 0..py {
            let sj = if j < ny { j } else { 2 * ny - 1 - j };
            for i in 0..px {
                let si = if i < nx { i } else { 2 * nx - 1 - i };
                ext[i + j * px + k * px * py] = phase[si + sj * nx + sk * nx * ny];
            }
        }
    }

    // Laplacian is taken on the extended array, so the wrapped differences never straddle
    // the original boundary — doing it before the extension reintroduces the seam.
    let d2u = wrapped_laplacian_periodic(&ext, px, py, pz, vsx, vsy, vsz);
    let solved = solve_poisson_fft(&d2u, px, py, pz, vsx, vsy, vsz);

    let mut result = vec![0.0; n_total];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let dst = i + j * nx + k * nx * ny;
                if mask[dst] != 0 {
                    result[dst] = solved[i + j * px + k * px * py];
                }
            }
        }
    }
    result
}

/// Poisson solver for [`laplacian_unwrap`] (names as in QSM.jl's `unwrap_laplacian`).
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LaplacianSolver {
    /// Unweighted least squares (Ghiglia & Romero 1994): finite differences of wrapped
    /// neighbour differences, Poisson solve under a Neumann condition via DCT. Matches QSM.jl
    /// `:dct`. The default.
    #[default]
    Dct,
    /// Schofield & Zhu (2003): ∇²φ from the sin/cos identity with spectral derivatives, FFT
    /// Poisson solve on the volume zero-padded by `pad` voxels per side. Reproduces STI Suite
    /// 3.0's `MRPhaseUnwrap` with `padsize = pad` (its default is 12; UK Biobank uses 64).
    Fft {
        /// Zero-padding added to both sides of each axis, in voxels.
        pad: [usize; 3],
    },
    /// Schofield & Zhu (2003) as Hongfu Sun's `lapunwrap.m` does it (QSMART's Laplacian
    /// unwrapping): ∇²φ from the sin/cos identity with spectral derivatives and an FFT Poisson
    /// solve, on the volume mirrored to twice its size along each axis (so no seam at the edge)
    /// rather than zero-padded. DC → 0 (`lapunwrap.m` divides it by `eps`, a constant that
    /// depends only on rounding).
    Mirror,
}

impl LaplacianSolver {
    /// [`LaplacianSolver::Fft`] with STI Suite's default padding, 12 voxels per side.
    pub const FFT_DEFAULT_PAD: LaplacianSolver = LaplacianSolver::Fft { pad: [12, 12, 12] };
}

/// Laplacian phase unwrapping, **without** background field removal.
///
/// Solves the Poisson equation over the whole array with the chosen [`LaplacianSolver`] (see
/// the [module docs](self) for both). Nothing is masked out of the input, so field sources
/// anywhere in the FOV are retained and the harmonic (background) component survives: the
/// result is an unwrapped **total** field, suitable for a subsequent background-removal stage.
/// (With [`LaplacianSolver::Fft`] the total field carries a harmonic error from the zero
/// padding wherever the phase at the array edge is not zero; the local field is unaffected.)
///
/// Because the whole array participates, this is sensitive to phase quality *outside* the
/// ROI. STI Suite-based pipelines zero the phase outside the brain first (`mask · φ`), which
/// [`run_field_mapping`](crate::pipeline::run_field_mapping) does on its multi-echo Laplacian
/// path; otherwise, where the phase outside the object is
/// noise, or wraps faster than one radian per voxel, prefer ROMEO
/// ([`super::romeo::unwrap_romeo`]).
///
/// # Arguments
/// * `phase` - Wrapped phase (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz); applied to the *output* only
/// * `grid` - Volume grid (dimensions and voxel sizes)
/// * `solver` - [`LaplacianSolver::Dct`] (default) or [`LaplacianSolver::Fft`]
///
/// # Returns
/// Unwrapped phase, zero outside `mask`.
///
/// # References
/// Ghiglia & Romero (1994) for `Dct`; Schofield & Zhu (2003) for `Fft`. See the module docs.
pub fn laplacian_unwrap(
    phase: &[f64],
    mask: &[u8],
    grid: &Grid,
    solver: LaplacianSolver,
) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let (vsx, vsy, vsz) = grid.voxel_size;
    let u = match solver {
        LaplacianSolver::Dct => {
            let d2u = wrapped_laplacian_neumann(phase, nx, ny, nz, vsx, vsy, vsz);
            solve_poisson_dct(&d2u, nx, ny, nz, vsx, vsy, vsz)
        }
        LaplacianSolver::Fft { pad } => laplacian_unwrap_fft_padded(phase, grid, pad),
        LaplacianSolver::Mirror => laplacian_unwrap_fft_mirrored(phase, grid),
    };
    u.iter().zip(mask).map(|(&v, &m)| if m != 0 { v } else { 0.0 }).collect()
}

/// Schofield & Zhu sin/cos unwrapping on the zero-padded volume (STI Suite's `MRPhaseUnwrap`).
/// Returns the whole volume, unmasked.
fn laplacian_unwrap_fft_padded(phase: &[f64], grid: &Grid, pad: [usize; 3]) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let (vsx, vsy, vsz) = grid.voxel_size;
    assert_eq!(phase.len(), nx * ny * nz, "phase length does not match grid");
    let [px, py, pz] = pad;
    let even = |n: usize| n + (n % 2);
    let (mx, my, mz) = (even(nx + 2 * px), even(ny + 2 * py), even(nz + 2 * pz));
    let mxy = mx * my;

    let k2 = |n: usize, h: f64| -> Vec<f64> { fftfreq(n, h).into_iter().map(|f| f * f).collect() };
    let (kx, ky, kz) = (k2(mx, vsx), k2(my, vsy), k2(mz, vsz));
    // Padded phase as (sin φ, cos φ); outside the input φ = 0.
    let sincos = |i: usize, j: usize, k: usize| -> (f64, f64) {
        match (i.checked_sub(px), j.checked_sub(py), k.checked_sub(pz)) {
            (Some(i), Some(j), Some(k)) if i < nx && j < ny && k < nz => phase[i + j * nx + k * nx * ny].sin_cos(),
            _ => (0.0, 1.0),
        }
    };

    let mut ws = Fft3dWorkspace::new(mx, my, mz);
    let mut buf = vec![Complex64::new(0.0, 0.0); mxy * mz];

    // K is real and even in k, so F⁻¹[K·F(x)] is real for real x: transform sin φ + i·cos φ
    // once and read F⁻¹[K·F(sin φ)] and F⁻¹[K·F(cos φ)] off the real and imaginary parts.
    for k in 0..mz {
        for j in 0..my {
            for i in 0..mx {
                let (s, c) = sincos(i, j, k);
                buf[i + j * mx + k * mxy] = Complex64::new(s, c);
            }
        }
    }
    ws.fft3d(&mut buf);
    apply_k2(&mut buf, &kx, &ky, &kz, false);
    ws.ifft3d(&mut buf);

    // L = cos φ · F⁻¹[K F sin φ] − sin φ · F⁻¹[K F cos φ]  (= −∇²φ / 4π²)
    for k in 0..mz {
        for j in 0..my {
            for i in 0..mx {
                let idx = i + j * mx + k * mxy;
                let (s, c) = sincos(i, j, k);
                let v = buf[idx];
                buf[idx] = Complex64::new(c * v.re - s * v.im, 0.0);
            }
        }
    }

    // Poisson inverse: φ = F⁻¹[F(L) / K], DC → 0.
    ws.fft3d(&mut buf);
    apply_k2(&mut buf, &kx, &ky, &kz, true);
    ws.ifft3d(&mut buf);

    let mut out = vec![0.0; nx * ny * nz];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                out[i + j * nx + k * nx * ny] = buf[(i + px) + (j + py) * mx + (k + pz) * mxy].re;
            }
        }
    }
    out
}

/// Schofield & Zhu sin/cos unwrapping on the volume mirrored to `2n` along each axis
/// (`lapunwrap.m`). Returns the whole volume, unmasked.
fn laplacian_unwrap_fft_mirrored(phase: &[f64], grid: &Grid) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let (vsx, vsy, vsz) = grid.voxel_size;
    assert_eq!(phase.len(), nx * ny * nz, "phase length does not match grid");
    let (mx, my, mz) = (2 * nx, 2 * ny, 2 * nz);
    let mxy = mx * my;
    let k2 = |n: usize, h: f64| -> Vec<f64> { fftfreq(n, h).into_iter().map(|f| f * f).collect() };
    let (kx, ky, kz) = (k2(mx, vsx), k2(my, vsy), k2(mz, vsz));
    let fold = |i: usize, n: usize| if i < n { i } else { 2 * n - 1 - i };
    let sincos = |i: usize, j: usize, k: usize| phase[fold(i, nx) + fold(j, ny) * nx + fold(k, nz) * nx * ny].sin_cos();

    let mut ws = Fft3dWorkspace::new(mx, my, mz);
    let mut buf = vec![Complex64::new(0.0, 0.0); mxy * mz];
    for k in 0..mz {
        for j in 0..my {
            for i in 0..mx {
                let (s, c) = sincos(i, j, k);
                buf[i + j * mx + k * mxy] = Complex64::new(s, c);
            }
        }
    }
    ws.fft3d(&mut buf);
    apply_k2(&mut buf, &kx, &ky, &kz, false);
    ws.ifft3d(&mut buf);
    for k in 0..mz {
        for j in 0..my {
            for i in 0..mx {
                let idx = i + j * mx + k * mxy;
                let (s, c) = sincos(i, j, k);
                let v = buf[idx];
                buf[idx] = Complex64::new(c * v.re - s * v.im, 0.0);
            }
        }
    }
    ws.fft3d(&mut buf);
    apply_k2(&mut buf, &kx, &ky, &kz, true);
    ws.ifft3d(&mut buf);

    let mut out = vec![0.0; nx * ny * nz];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                out[i + j * nx + k * nx * ny] = buf[i + j * mx + k * mxy].re;
            }
        }
    }
    out
}

/// Multiply (or, with `invert`, divide) a spectrum by `K = kx² + ky² + kz²`; DC → 0 on divide.
fn apply_k2(buf: &mut [Complex64], kx: &[f64], ky: &[f64], kz: &[f64], invert: bool) {
    let (mx, my) = (kx.len(), ky.len());
    let plane = |(k, slab): (usize, &mut [Complex64])| {
        for j in 0..my {
            let kyz = ky[j] + kz[k];
            for (v, &kxi) in slab[j * mx..(j + 1) * mx].iter_mut().zip(kx) {
                let kk = kxi + kyz;
                if invert {
                    *v = if kk == 0.0 { Complex64::new(0.0, 0.0) } else { *v / kk };
                } else {
                    *v *= kk;
                }
            }
        }
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        buf.par_chunks_mut(mx * my).enumerate().for_each(plane);
    }
    #[cfg(not(feature = "parallel"))]
    buf.chunks_mut(mx * my).enumerate().for_each(plane);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mirror_matches_lapunwrap() {
        // Hongfu Sun's lapunwrap.m (as QSMART calls it) in MATLAB R2023b on the same wrapped
        // phase, both made zero-mean (lapunwrap's DC is a rounding-dependent constant).
        let (nx, ny, nz) = (8, 6, 5);
        let grid = Grid::new(nx, ny, nz, 1.0, 1.5, 2.0);
        let phase: Vec<f64> = (0..nx * ny * nz)
            .map(|i| {
                let (a, b, c) = ((i % nx) as f64, ((i / nx) % ny) as f64, (i / (nx * ny)) as f64);
                let t = 0.9 * a * a / 4.0 + 1.7 * b - 0.6 * c * c + 0.3 * (i as f64).sin();
                t.sin().atan2(t.cos())
            })
            .collect();
        let mut u = laplacian_unwrap(&phase, &vec![1; phase.len()], &grid, LaplacianSolver::Mirror);
        let mean = u.iter().sum::<f64>() / u.len() as f64;
        u.iter_mut().for_each(|v| *v -= mean);
        let close = |a: f64, b: f64| assert!((a - b).abs() <= 1e-10 * b.abs().max(1.0), "{a} vs {b}");
        close(u.iter().map(|v| v.abs()).sum(), 809.9262085647304);
        close(u.iter().map(|v| v * v).sum(), 4077.856347779972);
        close(u[0], -5.778520101297622);
        close(u[7], 2.016668604865187);
        close(u[100], -4.332690703087126);
    }

    fn grid(n: usize) -> Grid {
        Grid::new(n, n, n, 1.0, 1.0, 1.0)
    }

    #[test]
    fn test_wrap() {
        assert!((wrap(0.0) - 0.0).abs() < 1e-10);
        assert!((wrap(PI) - PI).abs() < 1e-10);
        assert!((wrap(-PI) - (-PI)).abs() < 1e-10);
        assert!((wrap(2.0 * PI) - 0.0).abs() < 1e-10);
        assert!((wrap(3.0 * PI) - PI).abs() < 1e-10);
        assert!((wrap(-3.0 * PI) - (-PI)).abs() < 1e-10);
    }

    /// Pearson r and the slope of `want` regressed on `obs`, inside `mask`.
    fn corr_slope(obs: &[f64], want: &[f64], mask: &[u8]) -> (f64, f64) {
        let (mut sa, mut sb, mut n) = (0.0, 0.0, 0usize);
        for i in 0..obs.len() {
            if mask[i] != 0 { sa += obs[i]; sb += want[i]; n += 1; }
        }
        let nf = n as f64;
        let (ma, mb) = (sa / nf, sb / nf);
        let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
        for i in 0..obs.len() {
            if mask[i] == 0 { continue; }
            let (da, db) = (obs[i] - ma, want[i] - mb);
            cov += da * db;
            va += da * da;
            vb += db * db;
        }
        (cov / (va.sqrt() * vb.sqrt()), cov / va)
    }
    /// `(wrapped, truth, ramp, blob, mask)`.
    type RampPlusBlob = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<u8>);


    /// A field split into a harmonic part (a linear ramp, ∇² = 0) and a non-harmonic part
    /// (a Gaussian blob), wrapped hard enough that unwrapping is doing real work.
    /// Returns (wrapped, truth, ramp, blob, mask).
    fn ramp_plus_blob(n: usize) -> RampPlusBlob {
        let c = n as f64 / 2.0;
        let total = n * n * n;
        let (mut truth, mut ramp, mut blob) = (vec![0.0; total], vec![0.0; total], vec![0.0; total]);
        let mut mask = vec![0u8; total];
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let idx = i + j * n + k * n * n;
                    let (x, y, z) = (i as f64 - c, j as f64 - c, k as f64 - c);
                    let r = (x * x + y * y + z * z).sqrt();
                    ramp[idx] = 0.35 * x + 0.20 * y + 0.15 * z;
                    blob[idx] = 12.0 * (-(r * r) / (2.0 * 8.0f64.powi(2))).exp();
                    truth[idx] = ramp[idx] + blob[idx];
                    if r < 22.0 { mask[idx] = 1; }
                }
            }
        }
        let wrapped: Vec<f64> = truth.iter().map(|&v| wrap(v)).collect();
        (wrapped, truth, ramp, blob, mask)
    }

    #[test]
    fn laplacian_unwrap_recovers_the_whole_field_including_background() {
        let n = 64;
        let (wrapped, truth, _, _, mask) = ramp_plus_blob(n);

        let out = laplacian_unwrap(&wrapped, &mask, &grid(n), LaplacianSolver::Dct);
        let (r, slope) = corr_slope(&out, &truth, &mask);

        assert!(r > 0.99, "should recover the total field, got r = {r}");
        assert!((slope - 1.0).abs() < 0.05, "expected unit slope, got {slope}");
    }

    #[test]
    fn dct_solve_matches_the_even_extended_solve() {
        // The DCT-II is the FFT of the even extension, so these are the same computation
        // with and without materialising the extension. Agreement should be to rounding.
        let n = 64;
        let (wrapped, _, _, _, mask) = ramp_plus_blob(n);
        let padded = laplacian_unwrap_even_extended_reference(&wrapped, &mask, &grid(n));
        let dct = laplacian_unwrap(&wrapped, &mask, &grid(n), LaplacianSolver::Dct);
        let scale = padded.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
        let max_diff = padded.iter().zip(&dct).fold(0.0f64, |m, (&a, &b)| m.max((a - b).abs()));
        assert!(max_diff < 1e-9 * scale, "DCT and padded solves differ: max |diff| = {max_diff} (scale {scale})");
    }

    // ---- LaplacianSolver::Fft (Schofield & Zhu; STI Suite's MRPhaseUnwrap) and multi-echo ----

    /// `Fft` unwrap of the whole volume (all-ones mask, so nothing is zeroed).
    fn fft(p: &[f64], grid: &Grid, pad: [usize; 3]) -> Vec<f64> {
        laplacian_unwrap(p, &vec![1u8; p.len()], grid, LaplacianSolver::Fft { pad })
    }

    #[test]
    fn dct_is_the_default_solver() {
        assert_eq!(LaplacianSolver::default(), LaplacianSolver::Dct);
    }

    /// The 8×8×6 test volume used for the STI golden values (0-based indices here,
    /// 1-based `i+1` etc. in the MATLAB that produced them).
    fn sti_golden_input() -> (Vec<f64>, Grid) {
        let (nx, ny, nz) = (8, 8, 6);
        let mut p = vec![0.0; nx * ny * nz];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let (a, b, c) = ((i + 1) as f64, (j + 1) as f64, (k + 1) as f64);
                    let t = 2.1 * (0.7 * a).sin() + 1.3 * (0.4 * b * c).cos() + 0.9 * a - 0.6 * c + 0.05 * a * b;
                    p[i + j * nx + k * nx * ny] = t.sin().atan2(t.cos());
                }
            }
        }
        (p, Grid::new(nx, ny, nz, 1.0, 1.0, 2.0))
    }

    #[test]
    fn fft_matches_sti_golden_values() {
        // STI Suite 3.0 MRPhaseUnwrap(phi, 'voxelsize', [1 1 2], 'padsize', [2 3 1]) in STI Suite 3.0
        // (R2023b, the values in STI_GOLDEN).
        let (p, grid) = sti_golden_input();
        let u = fft(&p, &grid, [2, 3, 1]);
        let sum: f64 = u.iter().sum();
        let sumsq: f64 = u.iter().map(|v| v * v).sum();
        assert!((sum - STI_GOLDEN.0).abs() < 1e-9, "sum {sum} vs STI {}", STI_GOLDEN.0);
        assert!((sumsq - STI_GOLDEN.1).abs() < 1e-9, "sum sq {sumsq} vs STI {}", STI_GOLDEN.1);
        for &(idx, want) in STI_GOLDEN.2 {
            assert!((u[idx] - want).abs() < 1e-12, "voxel {idx}: {} vs STI {want}", u[idx]);
        }
    }

    /// (sum, sum of squares, [(0-based linear index, value)]) of STI's output on golden_input().
    #[allow(clippy::type_complexity, clippy::excessive_precision)]
    const STI_GOLDEN: (f64, f64, &[(usize, f64)]) = (
        -47.110605048249184,
        278.05516319999214,
        &[
            (0, 0.035944170373255008), (1, 0.54603271528608766), (8, -0.12289609617968449),
            (64, 0.17121148831819896), (99, -0.58598953742943161), (199, 0.35334342671042662),
            (332, -0.18693470845945748), (383, 1.1196905106581743),
        ],
    );

    #[test]
    fn fft_invariant_to_two_pi_jumps() {
        // Only sin φ and cos φ enter, so adding 2π anywhere changes nothing.
        let (p, grid) = sti_golden_input();
        let jumped: Vec<f64> = p.iter().enumerate()
            .map(|(i, &v)| v + 2.0 * PI * ((i * 7919) % 5) as f64 - 4.0 * PI)
            .collect();
        let a = fft(&p, &grid, [12; 3]);
        let b = fft(&jumped, &grid, [12; 3]);
        let d = a.iter().zip(&b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
        assert!(d < 1e-12, "max |diff| {d}");
    }

    /// Smooth blob, wrapped many times, inside a box of zeros.
    fn wrapped_blob_vs(n: (usize, usize, usize), vs: (f64, f64, f64)) -> (Vec<f64>, Vec<f64>, Grid) {
        let grid = Grid::new(n.0, n.1, n.2, vs.0, vs.1, vs.2);
        let c = (n.0 as f64 / 2.0 * vs.0, n.1 as f64 / 2.0 * vs.1, n.2 as f64 / 2.0 * vs.2);
        let mut truth = vec![0.0; grid.n_total()];
        for k in 0..n.2 {
            for j in 0..n.1 {
                for i in 0..n.0 {
                    let (x, y, z) = (i as f64 * vs.0 - c.0, j as f64 * vs.1 - c.1, k as f64 * vs.2 - c.2);
                    truth[i + j * n.0 + k * n.0 * n.1] = 20.0 * (-(x * x + y * y + z * z) / (2.0 * 9.0f64.powi(2))).exp();
                }
            }
        }
        let wrapped = truth.iter().map(|&v| wrap(v)).collect();
        (wrapped, truth, grid)
    }

    fn max_dev_after_mean(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let off = a.iter().zip(b).map(|(x, y)| x - y).sum::<f64>() / n;
        a.iter().zip(b).fold(0.0f64, |m, (x, y)| m.max((x - y - off).abs()))
    }

    #[test]
    fn fft_unwraps_a_smooth_field() {
        // Peak 20 rad (≈ 3 wraps), decays to ~0 at the edge: the periodic, zero-padded solve
        // should return it up to a constant. Steepest change ≈ 1.35 rad/mm, so at most ~1.6 rad
        // per voxel here; the sin/cos identity degrades well before π per voxel (at 2 mm,
        // 2.7 rad/voxel, the error is ~0.2 rad).
        for vs in [(1.0, 1.0, 1.0), (0.8, 0.8, 1.2), (1.2, 0.9, 1.0)] {
            let (wrapped, truth, grid) = wrapped_blob_vs((64, 64, 48), vs);
            let u = fft(&wrapped, &grid, [12; 3]);
            let d = max_dev_after_mean(&u, &truth);
            assert!(d < 0.05, "voxel size {vs:?}: max deviation {d} rad");
        }
    }

    #[test]
    fn fft_odd_dimensions_keep_the_input_size() {
        let (wrapped, truth, grid) = wrapped_blob_vs((61, 63, 47), (1.0, 1.0, 1.0));
        for pad in [[0, 0, 0], [3, 4, 5]] {
            let u = fft(&wrapped, &grid, pad);
            assert_eq!(u.len(), grid.n_total());
            assert!(max_dev_after_mean(&u, &truth) < 0.05);
        }
    }

    #[test]
    fn fft_dc_of_the_padded_box_is_zero() {
        // The Poisson solve zeroes the mean over the padded box, not over the input.
        let (wrapped, _, grid) = wrapped_blob_vs((32, 32, 24), (1.0, 1.0, 1.0));
        let u = fft(&wrapped, &grid, [0, 0, 0]);
        let mean = u.iter().sum::<f64>() / u.len() as f64;
        assert!(mean.abs() < 1e-12, "mean {mean}");
    }

    #[test]
    fn fft_zero_padding_costs_total_field_fidelity() {
        // A harmonic ramp that is far from zero at the array faces: the zero padding puts a
        // step there, and the periodic solve turns it into a harmonic error inside. The Neumann
        // solve has no such step. This is the documented reason `Fft` is not the
        // default; if it starts passing the other way, update the module docs.
        let n = 48;
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let c = n as f64 / 2.0;
        let mut truth = vec![0.0; n * n * n];
        for k in 0..n { for j in 0..n { for i in 0..n {
            let (x, y, z) = (i as f64 - c, j as f64 - c, k as f64 - c);
            truth[i + j * n + k * n * n] = 0.35 * x + 0.2 * y + 0.15 * z
                + 12.0 * (-(x * x + y * y + z * z) / 128.0).exp();
        }}}
        let wrapped: Vec<f64> = truth.iter().map(|&v| wrap(v)).collect();
        let mask = vec![1u8; truth.len()];
        let r = |u: &[f64]| {
            let m = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            let (mu, mt) = (m(u), m(&truth));
            let (mut cov, mut vu, mut vt) = (0.0, 0.0, 0.0);
            for (a, b) in u.iter().zip(&truth) { cov += (a - mu) * (b - mt); vu += (a - mu).powi(2); vt += (b - mt).powi(2); }
            cov / (vu * vt).sqrt()
        };
        let sti = fft(&wrapped, &grid, [12; 3]);
        let neumann = laplacian_unwrap(&wrapped, &mask, &grid, LaplacianSolver::Dct);
        assert!(r(&neumann) > 0.99, "Neumann r = {}", r(&neumann));
        assert!(r(&sti) < 0.9, "STI r = {}", r(&sti));
    }
}

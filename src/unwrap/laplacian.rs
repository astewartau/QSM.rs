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
//! [`laplacian_unwrap_multi_echo`] unwraps each echo and averages the unwrapped phases with
//! per-echo weights ([`EchoWeighting`], e.g. `TE·exp(−TE/T2*)`) and a matching weighted echo
//! time: the field map STI Suite-based pipelines build (with `Fft { pad: [64; 3] }` and
//! T2\* = 40 ms, UK Biobank's).
//!
//! [`laplacian_unwrap_bfr`] (deprecated) is a different algorithm: finite-difference ∇² masked
//! to the ROI and solved under a Dirichlet condition on it. Zeroing ∇²φ outside the mask
//! discards every field source outside the ROI. Background fields are harmonic inside the ROI,
//! and ∇²(harmonic) = 0 carries no information about them, so they cannot be recovered
//! afterwards — the function returns a partially background-removed field, not a total field.
//! That is the same combination HARPERELLA and iHARPERELLA perform, and is why it is
//! categorised with them rather than with ROMEO. Pair it with a separate background-removal
//! stage only deliberately: doing so removes background twice, by an amount that is not
//! controlled. On the project's test data it reaches r = 0.887 against the ground-truth local
//! field, against r = 0.909 for [`crate::bgremove::lbv`] on the same field and 0.879 for V-SHARP,
//! and it matches QSM.jl's `:mgpcg` to r = 0.999983 (Gauss-Seidel against
//! multigrid-preconditioned CG on the same equation).
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
//! Zhou, D., Liu, T., Spincemaille, P., Wang, Y. (2014). "Background field removal by
//! solving the Laplacian boundary value problem." NMR in Biomedicine, 27(3):312-319 (the
//! boundary-value half of [`laplacian_unwrap_bfr`]). <https://doi.org/10.1002/nbm.3064>
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
pub(crate) fn wrapped_laplacian_periodic(
    phase: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
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

/// Laplacian phase unwrapping **combined with background field removal**.
///
/// Solves the Poisson equation with the Laplacian zeroed outside `mask`. That discards the
/// field sources outside the ROI, and a field generated outside the ROI is harmonic inside
/// it — so the background component is removed along with the wraps.
///
/// **The result is not a total field.** It is unwrapped *and* partially background-removed,
/// by an amount that depends on the mask and the field geometry. Following this with a
/// separate background-removal stage (V-SHARP, PDF, …) removes background twice.
///
/// Use [`laplacian_unwrap`] to unwrap without removing background.
///
/// Because ∇²(harmonic) = 0, the discarded component leaves no trace in the input to the
/// Poisson solve and cannot be restored afterwards.
///
/// # Intended input
/// One wrapped phase volume. On the project's test data it then matches
/// [`crate::bgremove::lbv`] on the same field (r = 0.887 against 0.909, identical residual
/// smooth content). It takes *wrapped phase*, so in a multi-echo pipeline the only way to
/// apply it is to each echo before combining; that usage is not what the algorithm
/// describes, has not been validated, and leaves visibly more background than either
/// unwrapping then a field-map background removal or this function on a single volume.
/// For multi-echo data use [`laplacian_unwrap`] or ROMEO, combine, then a background
/// removal from [`crate::bgremove`].
///
/// # Echo time
/// Accuracy falls off with the amount of phase to unwrap. On the 7 T test data, against
/// the ground-truth local field: r = 0.84 at TE = 4 ms, 0.70 at 8 ms, 0.50 at 12 ms —
/// where unwrapping then [`crate::bgremove::lbv`] gives 0.86, 0.84, 0.69 and
/// [`crate::bgremove::vsharp`] holds near 0.82 throughout. Prefer the earliest echo.
///
/// # Arguments
/// * `phase` - Wrapped phase (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz), 1 = inside ROI
/// * `grid` - Volume grid (dimensions and voxel sizes)
///
/// # Returns
/// Unwrapped, partially background-removed phase, zero outside `mask`.
///
/// # References
/// Schofield & Zhu (2003) for the unwrapping; Zhou et al. (2014) for the
/// boundary-value formulation of the background removal. See the module docs.
/// Solve ∇²u = f inside `mask` with u = 0 outside it (homogeneous Dirichlet on the ROI),
/// by Gauss-Seidel with successive over-relaxation.
///
/// Masking the source term is only half of the ROI formulation: the solution has to be
/// constrained at the ROI boundary too, or it picks up an arbitrary harmonic component.
/// Solving the masked source over the whole volume with a periodic FFT does not constrain
/// it, and that component is large — which is what this replaces.
///
/// Mirrors the solver in [`crate::bgremove::lbv`], which solves the homogeneous case
/// (`f = 0`) with boundary values taken from the field.
fn solve_poisson_dirichlet_roi(
    f: &[f64],
    mask: &[u8],
    grid: &Grid,
    tol: f64,
    max_iter: usize,
) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let (vsx, vsy, vsz) = grid.voxel_size;
    let (dx2, dy2, dz2) = (1.0 / (vsx * vsx), 1.0 / (vsy * vsy), 1.0 / (vsz * vsz));
    let diag = -2.0 * (dx2 + dy2 + dz2);
    let omega = 1.5;

    let mut u = vec![0.0f64; nx * ny * nz];

    // Relative criterion, so convergence does not depend on the units of `f`.
    let scale = f.iter().map(|v| v.abs()).fold(0.0f64, f64::max).max(1e-30);
    let scaled_tol = tol * scale / diag.abs();

    for _ in 0..max_iter {
        let mut max_change = 0.0f64;
        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    let idx = i + j * nx + k * nx * ny;
                    if mask[idx] == 0 {
                        continue; // stays 0: this is the Dirichlet condition
                    }
                    let sum = dx2 * (u[idx - 1] + u[idx + 1])
                            + dy2 * (u[idx - nx] + u[idx + nx])
                            + dz2 * (u[idx - nx * ny] + u[idx + nx * ny]);
                    // ∇²u = sum + diag*u = f  =>  u = (f - sum) / diag
                    let target = (f[idx] - sum) / diag;
                    let old = u[idx];
                    let next = old + omega * (target - old);
                    max_change = max_change.max((next - old).abs());
                    u[idx] = next;
                }
            }
        }
        if max_change < scaled_tol {
            break;
        }
    }
    u
}

#[deprecated(
    since = "0.35.0",
    note = "unwrap with `laplacian_unwrap` and then remove background with a `bgremove` \
            method (`lbv` reproduces this; `vsharp` is more robust at long TE). That route \
            is more accurate at every echo time measured and composes with any background \
            removal. Kept for parity with QSM.jl's `unwrap_laplacian(solver = :mgpcg)`."
)]
pub fn laplacian_unwrap_bfr(
    phase: &[f64],
    mask: &[u8],
    grid: &Grid,
) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let (vsx, vsy, vsz) = grid.voxel_size;
    let n_total = nx * ny * nz;

    let d2u = wrapped_laplacian_periodic(phase, nx, ny, nz, vsx, vsy, vsz);

    let d2u_masked: Vec<f64> = d2u.iter()
        .enumerate()
        .map(|(i, &val)| if mask[i] != 0 { val } else { 0.0 })
        .collect();

    // Dirichlet on the ROI, matching QSM.jl's `:mgpcg` path. Masking the source and then
    // solving over the whole volume with a periodic FFT leaves the harmonic component
    // unconstrained, which showed up as a large spurious background field.
    let max_iter = (3 * nx.max(ny).max(nz)).min(500);
    let unwrapped = solve_poisson_dirichlet_roi(&d2u_masked, mask, grid, 1e-6, max_iter);

    let mut result = vec![0.0; n_total];
    for i in 0..n_total {
        if mask[i] != 0 {
            result[i] = unwrapped[i];
        }
    }

    result
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
/// Use [`laplacian_unwrap_bfr`] if you want unwrapping and background removal together.
///
/// Because the whole array participates, this is sensitive to phase quality *outside* the
/// ROI. STI Suite-based pipelines zero the phase outside the brain first (`mask · φ`), which
/// [`laplacian_unwrap_multi_echo`] does; otherwise, where the phase outside the object is
/// noise, or wraps faster than one radian per voxel, prefer ROMEO
/// ([`super::romeo::unwrap_romeo`]) or the masked variant.
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

/// How echoes are weighted in [`laplacian_unwrap_multi_echo`]. One weight per echo; echo
/// times and `t2star` in the same unit.
#[derive(Clone, Debug, PartialEq)]
pub enum EchoWeighting {
    /// `w = TE · exp(−TE / T2*)`: the phase-SNR-optimal weight for a single-exponential decay
    /// with the given T2\*. UK Biobank uses T2\* = 40 ms.
    T2Star { t2star: f64 },
    /// `w = TE`.
    EchoTime,
    /// `w = 1`.
    Uniform,
    /// Explicit per-echo weights.
    Custom(Vec<f64>),
}

impl EchoWeighting {
    /// `TE · exp(−TE / 40 ms)` (UK Biobank's weighting), for echo times in seconds.
    pub fn t2star_40ms() -> Self {
        EchoWeighting::T2Star { t2star: 0.040 }
    }

    /// Per-echo weights for the given echo times.
    pub fn weights(&self, tes: &[f64]) -> Vec<f64> {
        match self {
            EchoWeighting::T2Star { t2star } => tes.iter().map(|&te| te * (-te / t2star).exp()).collect(),
            EchoWeighting::EchoTime => tes.to_vec(),
            EchoWeighting::Uniform => vec![1.0; tes.len()],
            EchoWeighting::Custom(w) => {
                assert_eq!(w.len(), tes.len(), "one custom weight per echo");
                w.clone()
            }
        }
    }
}

/// Result of [`laplacian_unwrap_multi_echo`].
#[derive(Clone, Debug)]
pub struct EchoAverage {
    /// `Σ wᵢ φᵢ / Σ wᵢ`: weighted mean of the unwrapped echo phases (radians), zero outside the mask.
    pub phase: Vec<f64>,
    /// `Σ wᵢ TEᵢ / Σ wᵢ`: the echo time `phase` corresponds to (unit of the input TEs).
    pub te_eff: f64,
    /// The per-echo weights used.
    pub weights: Vec<f64>,
}

impl EchoAverage {
    /// Field in rad per unit of TE (rad/s for TEs in seconds): `phase / te_eff`.
    pub fn field_rad(&self) -> Vec<f64> {
        self.phase.iter().map(|&p| p / self.te_eff).collect()
    }

    /// Field in Hz, for TEs in seconds: `phase / (2π · te_eff)`.
    pub fn field_hz(&self) -> Vec<f64> {
        let s = 1.0 / (std::f64::consts::TAU * self.te_eff);
        self.phase.iter().map(|&p| p * s).collect()
    }
}

/// Multi-echo field map from per-echo Laplacian unwrapping and a weighted echo average.
///
/// Each echo is zeroed outside `mask`, unwrapped on its own with [`laplacian_unwrap`] and the
/// given solver, and the unwrapped phases are averaged:
/// `phase = Σ wᵢ φᵢ / Σ wᵢ`, `te_eff = Σ wᵢ TEᵢ / Σ wᵢ`, field = `phase / te_eff`
/// (equivalently, a mean of `φᵢ / TEᵢ` weighted by `wᵢ·TEᵢ`).
///
/// With `LaplacianSolver::Fft { pad: [64; 3] }` and [`EchoWeighting::t2star_40ms`] this is the
/// field map of STI Suite-based pipelines such as UK Biobank's (`MRPhaseUnwrap(mask .* phase)`
/// per echo, then the T2\*-weighted mean), to ≈1e-14 rad inside the mask.
///
/// The phases should be free of a per-echo phase offset (e.g. coil-combined with MCPC-3D-S);
/// any offset left in passes through scaled by `1 / te_eff`.
///
/// # Arguments
/// * `phases` - Wrapped phase per echo (radians)
/// * `tes` - Echo times (any unit; the field is per that unit)
/// * `mask` - Binary mask: applied to each echo before unwrapping and to the output
/// * `grid` - Volume grid
/// * `solver` - Poisson solver for each echo
/// * `weighting` - Echo weights
pub fn laplacian_unwrap_multi_echo<P: AsRef<[f64]>>(
    phases: &[P],
    tes: &[f64],
    mask: &[u8],
    grid: &Grid,
    solver: LaplacianSolver,
    weighting: &EchoWeighting,
) -> EchoAverage {
    assert!(!phases.is_empty(), "no echoes");
    assert_eq!(phases.len(), tes.len(), "one echo time per echo");
    let n = grid.n_total();
    assert_eq!(mask.len(), n, "mask length does not match grid");
    let weights = weighting.weights(tes);
    let wsum: f64 = weights.iter().sum();
    assert!(wsum.abs() > 0.0 && wsum.is_finite(), "echo weights sum to {wsum}");
    let te_eff = weights.iter().zip(tes).map(|(w, t)| w * t).sum::<f64>() / wsum;

    let mut phase = vec![0.0; n];
    for (p, &w) in phases.iter().zip(&weights) {
        let p = p.as_ref();
        assert_eq!(p.len(), n, "phase length does not match grid");
        let masked: Vec<f64> = p.iter().zip(mask).map(|(&v, &b)| if b != 0 { v } else { 0.0 }).collect();
        for (acc, v) in phase.iter_mut().zip(laplacian_unwrap(&masked, mask, grid, solver)) {
            *acc += w * v;
        }
    }
    for v in phase.iter_mut() {
        *v /= wsum;
    }
    EchoAverage { phase, te_eff, weights }
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;

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

    #[test]
    fn test_laplacian_unwrap_bfr_constant() {
        let n = 8;
        let phase = vec![1.0; n * n * n];
        let mask = vec![1u8; n * n * n];

        let unwrapped = laplacian_unwrap_bfr(&phase, &mask, &grid(n));

        let mean: f64 = unwrapped.iter().sum::<f64>() / (n * n * n) as f64;
        for &val in unwrapped.iter() {
            assert!((val - mean).abs() < 1e-6, "Constant phase should unwrap to constant");
        }
    }

    #[test]
    fn test_laplacian_unwrap_bfr_smooth() {
        let n = 16;
        let mut phase = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];

        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let idx = i + j * n + k * n * n;
                    phase[idx] = 0.5 * (2.0 * PI * i as f64 / n as f64).sin();
                }
            }
        }

        let unwrapped = laplacian_unwrap_bfr(&phase, &mask, &grid(n));

        // This function solves under a Dirichlet condition on the ROI, so it does *not*
        // round-trip its input — the harmonic part that would be needed to match at the
        // boundary is exactly what it removes. What must hold is that the output is finite,
        // stays bounded by the input, and is zero where the ROI is not.
        //
        // The original assertion here was `< 1.0` against a signal of amplitude 0.5, which
        // passed for an all-zero output. `laplacian_unwrap_bfr_removes_the_harmonic_component`
        // is the test that pins what this function actually does.
        assert!(unwrapped.iter().all(|v| v.is_finite()), "output must be finite");
        let peak = unwrapped.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
        assert!(peak <= 0.5 + 1e-6, "output should not exceed the input amplitude, got {peak}");
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
    fn laplacian_unwrap_bfr_removes_the_harmonic_component() {
        // Pins the documented contract: laplacian_unwrap_bfr is unwrapping + background
        // removal. It reproduces the non-harmonic field faithfully and returns the
        // harmonic (background) component as zero, because ∇²(harmonic) = 0 leaves no
        // trace in the input to the Poisson solve.
        //
        // If the harmonic assertion starts failing, the limitation was lifted — update
        // the module docs and the README categorisation along with it.
        let n = 64;
        let (wrapped, _, ramp, blob, mask) = ramp_plus_blob(n);

        let out = laplacian_unwrap_bfr(&wrapped, &mask, &grid(n));

        let (r_blob, slope_blob) = corr_slope(&out, &blob, &mask);
        assert!(r_blob > 0.99, "non-harmonic part should survive, got r = {r_blob}");
        assert!((slope_blob - 1.0).abs() < 0.05, "expected unit slope, got {slope_blob}");

        let (r_ramp, _) = corr_slope(&out, &ramp, &mask);
        assert!(r_ramp.abs() < 0.05, "harmonic part should be discarded, got r = {r_ramp}");
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

    #[test]
    fn the_two_variants_disagree_by_the_background_field() {
        // The difference between them is the harmonic component, which is what makes
        // them different algorithms rather than two settings of one.
        let n = 64;
        let (wrapped, _, ramp, _, mask) = ramp_plus_blob(n);

        let pure = laplacian_unwrap(&wrapped, &mask, &grid(n), LaplacianSolver::Dct);
        let combined = laplacian_unwrap_bfr(&wrapped, &mask, &grid(n));
        let diff: Vec<f64> = pure.iter().zip(&combined).map(|(a, b)| a - b).collect();

        let (r, _) = corr_slope(&diff, &ramp, &mask);
        assert!(r > 0.95, "their difference should be the harmonic field, got r = {r}");
    }

    #[test]
    fn test_laplacian_unwrap_bfr_finite() {
        let n = 8;
        let phase: Vec<f64> = (0..n*n*n).map(|i| wrap((i as f64) * 0.1)).collect();
        let mask = vec![1u8; n * n * n];

        let unwrapped = laplacian_unwrap_bfr(&phase, &mask, &grid(n));

        for (i, &val) in unwrapped.iter().enumerate() {
            assert!(val.is_finite(), "Unwrapped phase should be finite at index {}", i);
        }
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

    #[test]
    fn t2star_weights_and_effective_te() {
        let tes = [9.42e-3, 19.7e-3];
        let w = EchoWeighting::t2star_40ms().weights(&tes);
        assert!((w[0] - 9.42e-3 * (-9.42f64 / 40.0).exp()).abs() < 1e-15);
        assert!((w[1] - 19.7e-3 * (-19.7f64 / 40.0).exp()).abs() < 1e-15);
        // te_eff for UK Biobank's echoes: 15.772350 ms (from the MATLAB run)
        let te_eff = (w[0] * tes[0] + w[1] * tes[1]) / (w[0] + w[1]);
        assert!((te_eff - 15.772350e-3).abs() < 1e-9, "{te_eff}");
        assert_eq!(EchoWeighting::Uniform.weights(&tes), vec![1.0, 1.0]);
        assert_eq!(EchoWeighting::EchoTime.weights(&tes), tes.to_vec());
    }

    #[test]
    fn multi_echo_average_recovers_the_frequency() {
        // Phase linear in TE (no offset): every weighting gives the same field.
        let (_, truth, grid) = wrapped_blob_vs((64, 64, 48), (1.0, 1.0, 1.0));
        let tes = [0.004, 0.009, 0.014];
        // truth is the phase at the last echo, TE = 14 ms
        let phases: Vec<Vec<f64>> = tes.iter()
            .map(|&te| truth.iter().map(|&v| wrap(v * te / 0.014)).collect())
            .collect();
        let reference: Vec<f64> = truth.iter().map(|&v| v / 0.014).collect();
        for wt in [EchoWeighting::t2star_40ms(), EchoWeighting::Uniform, EchoWeighting::EchoTime,
                   EchoWeighting::Custom(vec![1.0, 2.0, 3.0])] {
            let ones = vec![1u8; grid.n_total()];
            let avg = laplacian_unwrap_multi_echo(&phases, &tes, &ones, &grid, LaplacianSolver::FFT_DEFAULT_PAD, &wt);
            let f = avg.field_rad();
            let d = max_dev_after_mean(&f, &reference) * 0.014;
            assert!(d < 0.05, "{wt:?}: max deviation {d} rad at 14 ms");
            let hz = avg.field_hz();
            assert!((hz[100] * std::f64::consts::TAU - f[100]).abs() < 1e-9);
        }
    }

    #[test]
    fn multi_echo_mask_is_applied_to_input_and_output() {
        let (wrapped, _, grid) = wrapped_blob_vs((32, 32, 24), (1.0, 1.0, 1.0));
        let mask: Vec<u8> = (0..grid.n_total()).map(|i| (i % 3 != 0) as u8).collect();
        let masked: Vec<f64> = wrapped.iter().zip(&mask).map(|(&v, &m)| v * m as f64).collect();
        let solver = LaplacianSolver::Fft { pad: [4, 4, 4] };
        let a = laplacian_unwrap_multi_echo(&[&wrapped], &[1.0], &mask, &grid, solver, &EchoWeighting::Uniform);
        let b = laplacian_unwrap(&masked, &mask, &grid, solver);
        assert_eq!(a.te_eff, 1.0);
        let d = a.phase.iter().zip(&b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
        assert!(d < 1e-14, "{d}");
    }
}

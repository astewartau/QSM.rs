//! Shared ADMM iteration infrastructure for TV-based inversion methods.
//!
//! Provides pre-allocated buffers and a generic iteration step used by
//! TV-ADMM, NLTV, and RTS algorithms.

use num_complex::Complex64;
use crate::fft::Fft3dWorkspace;
use crate::Grid;
use crate::kernels::dipole::dipole_kernel;
use crate::kernels::laplacian::laplacian_kernel;
use crate::utils::gradient::{bdiv_inplace, fgrad_inplace};
use crate::utils::relative_change;

/// Pre-allocated buffers for ADMM-based inversion algorithms.
///
/// All buffers are sized to `n_total` voxels. Zero allocations occur
/// during iteration when using these buffers.
pub struct AdmmBuffers {
    /// Current solution
    pub x: Vec<f64>,
    /// Previous solution (for convergence)
    pub x_prev: Vec<f64>,
    /// Dual variables (x/y/z components)
    pub ux: Vec<f64>,
    pub uy: Vec<f64>,
    pub uz: Vec<f64>,
    /// Gradient / (z-u) buffers (dual-purpose)
    pub gx: Vec<f64>,
    pub gy: Vec<f64>,
    pub gz: Vec<f64>,
    /// Divergence buffer
    pub div_buf: Vec<f64>,
    /// Complex FFT work buffer
    pub work_complex: Vec<Complex64>,
}

impl AdmmBuffers {
    /// Allocate all ADMM buffers for a given volume size.
    pub fn new(n_total: usize) -> Self {
        Self {
            x: vec![0.0; n_total],
            x_prev: vec![0.0; n_total],
            ux: vec![0.0; n_total],
            uy: vec![0.0; n_total],
            uz: vec![0.0; n_total],
            gx: vec![0.0; n_total],
            gy: vec![0.0; n_total],
            gz: vec![0.0; n_total],
            div_buf: vec![0.0; n_total],
            work_complex: vec![Complex64::new(0.0, 0.0); n_total],
        }
    }
}

/// Run one ADMM iteration step.
///
/// Performs the x-subproblem (spectral solve), convergence check,
/// and fused z-subproblem + dual variable update.
///
/// The `shrink_fn(vx, vy, vz, index) -> (zx, zy, zz)` closure captures
/// per-algorithm differences in the proximal operator.
///
/// # Arguments
/// * `buf` - Pre-allocated buffers (modified in place)
/// * `fft_ws` - FFT workspace
/// * `f_hat` - Pre-computed constant RHS in frequency domain
/// * `inv_a` - Pre-computed inverse operator
/// * `rho` - ADMM penalty multiplier for divergence term
/// * `grid` - Volume grid
/// * `tol` - Convergence tolerance
/// * `shrink_fn` - Proximal operator for z-subproblem
///
/// # Returns
/// `true` if converged (relative change < tol)
#[inline]
pub fn admm_step<S>(
    buf: &mut AdmmBuffers,
    fft_ws: &mut Fft3dWorkspace,
    f_hat: &[Complex64],
    inv_a: &[f64],
    rho: f64,
    grid: &Grid,
    tol: f64,
    shrink_fn: S,
) -> bool
where
    S: Fn(f64, f64, f64, usize) -> (f64, f64, f64),
{
    let n_total = grid.n_total();

    // Swap x and x_prev (no allocation, just pointer swap)
    std::mem::swap(&mut buf.x, &mut buf.x_prev);

    // === x-subproblem: solve in frequency domain ===

    // Compute div(z - u) — gx/gy/gz hold (z-u) from previous step
    bdiv_inplace(&mut buf.div_buf, &buf.gx, &buf.gy, &buf.gz, grid);

    // FFT of divergence
    for i in 0..n_total {
        buf.work_complex[i] = Complex64::new(buf.div_buf[i], 0.0);
    }
    fft_ws.fft3d(&mut buf.work_complex);

    // x_hat = f_hat - rho * FFT(div) * inv_a
    for i in 0..n_total {
        buf.work_complex[i] = f_hat[i] - rho * buf.work_complex[i] * inv_a[i];
    }

    // IFFT to spatial domain
    fft_ws.ifft3d(&mut buf.work_complex);
    for i in 0..n_total {
        buf.x[i] = buf.work_complex[i].re;
    }

    // === Convergence check ===
    if relative_change(&buf.x, &buf.x_prev) < tol {
        return true;
    }

    // === Fused z-subproblem + u-update ===

    // Compute gradient of x
    fgrad_inplace(&mut buf.gx, &mut buf.gy, &mut buf.gz, &buf.x, grid);

    // Apply proximal operator and update duals
    for i in 0..n_total {
        let vx = buf.gx[i] + buf.ux[i];
        let vy = buf.gy[i] + buf.uy[i];
        let vz = buf.gz[i] + buf.uz[i];

        let (zx, zy, zz) = shrink_fn(vx, vy, vz, i);

        buf.ux[i] = vx - zx;
        buf.uy[i] = vy - zy;
        buf.uz[i] = vz - zz;

        // Store (2z - v) = (z - u_new) for next iteration's divergence
        buf.gx[i] = 2.0 * zx - vx;
        buf.gy[i] = 2.0 * zy - vy;
        buf.gz[i] = 2.0 * zz - vz;
    }

    false
}

/// Pre-compute ADMM spectral operators (dipole kernel, inverse operator, and RHS).
///
/// Shared by TV-ADMM and NLTV. Returns (fft_workspace, inv_a, f_hat).
pub fn prepare_admm_spectral(
    local_field: &[f64],
    grid: &Grid,
    bdir: (f64, f64, f64),
    rho: f64,
) -> (Fft3dWorkspace, Vec<f64>, Vec<Complex64>) {
    let n_total = grid.n_total();
    let mut fft_ws = Fft3dWorkspace::new(grid.nx(), grid.ny(), grid.nz());
    let d_kernel = dipole_kernel(grid, bdir);
    let l_kernel = laplacian_kernel(grid, true);
    let mut l_complex: Vec<Complex64> = l_kernel.iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();
    fft_ws.fft3d(&mut l_complex);
    let mut inv_a: Vec<f64> = vec![0.0; n_total];
    for i in 0..n_total {
        let a = d_kernel[i] * d_kernel[i] + rho * l_complex[i].re;
        inv_a[i] = if a.abs() > 1e-20 { 1.0 / a } else { 0.0 };
    }
    let f_hat = &mut l_complex;
    for i in 0..n_total {
        f_hat[i] = Complex64::new(local_field[i], 0.0);
    }
    fft_ws.fft3d(f_hat);
    for i in 0..n_total {
        f_hat[i] = f_hat[i] * d_kernel[i] * inv_a[i];
    }
    (fft_ws, inv_a, l_complex)
}

/// Pre-compute spectral operators shared by the FANSI-family nonlinear solvers
/// (NDI, nlTV, nlTGV, L1-QSM, WH-QSM, HD-QSM).
///
/// Unlike [`prepare_admm_spectral`], this does not bake the data term into an RHS,
/// because those solvers carry a separately-updated data auxiliary variable. It
/// returns the reusable primitives instead:
///
/// * `Fft3dWorkspace` — reusable FFT plans (unnormalized forward `fft3d`,
///   normalized inverse `ifft3d`, matching MATLAB `fftn`/`ifftn`).
/// * `k_kernel` — the real k-space dipole kernel `D(k)` (same convention as the
///   rest of the crate: continuous kernel, includes voxel-size scaling, DC = 0).
/// * `ee2` — the real spectral Laplacian `EE2(k) = |E1|² + |E2|² + |E3|²`, i.e.
///   the frequency response of `bdiv ∘ fgrad`. Use as the regularization term in
///   the x-subproblem denominator so it stays consistent with real-space
///   `fgrad_inplace`/`bdiv_inplace` (which share the same voxel-size scaling).
///
/// The x-subproblem denominator for these solvers is therefore
/// `mu2 * k_kernel² + mu * ee2` (all real), matching FANSI's
/// `mu2*abs(Kernel).^2 + mu*EE2`.
pub fn prepare_fansi_spectral(
    grid: &Grid,
    bdir: (f64, f64, f64),
) -> (Fft3dWorkspace, Vec<f64>, Vec<f64>) {
    let mut fft_ws = Fft3dWorkspace::new(grid.nx(), grid.ny(), grid.nz());
    let k_kernel = dipole_kernel(grid, bdir);
    let l_kernel = laplacian_kernel(grid, true);
    let mut l_complex: Vec<Complex64> = l_kernel.iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();
    fft_ws.fft3d(&mut l_complex);
    let ee2: Vec<f64> = l_complex.iter().map(|c| c.re).collect();
    (fft_ws, k_kernel, ee2)
}

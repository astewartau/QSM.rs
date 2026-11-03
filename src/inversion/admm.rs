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
/// For TV-ADMM (QSM.jl's `tv`): both are the kernels QSM.jl's real-to-complex FFT implies
/// ([`crate::fft::c2r_kernel`]). Returns (fft_workspace, inv_a, f_hat).
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
    // The Laplacian's DC term is 0 (its stencil sums to zero), but its FFT is only zero to
    // rounding on anisotropic voxels (e.g. 4.4e-16 at 0.9 × 1.0 × 1.3 mm), and inverting that
    // puts a rounding-dependent constant into χ. Zero it, so `a` at DC is exactly 0 and χ's DC is
    // 0, as it is on isotropic voxels. QSM.jl's `tv` keeps its rounded value (see `tv`'s docs).
    l_complex[0] = Complex64::new(0.0, 0.0);
    let mut inv_a: Vec<f64> = vec![0.0; n_total];
    for i in 0..n_total {
        let a = d_kernel[i] * d_kernel[i] + rho * l_complex[i].re;
        inv_a[i] = if a.abs() > 1e-20 { 1.0 / a } else { 0.0 };
    }
    // QSM.jl's `tv` applies `iA` and `iA·D` on its real-to-complex half spectrum; give the full
    // FFT the kernels `irfft` implies (astewartau/QSM.rs#329), so tilted B0 matches it too.
    let (nx, ny, nz) = grid.dims;
    let rhs: Vec<f64> = (0..n_total).map(|i| d_kernel[i] * inv_a[i]).collect();
    let rhs = crate::fft::c2r_kernel(&rhs, nx, ny, nz);
    let inv_a = crate::fft::c2r_kernel(&inv_a, nx, ny, nz);
    let f_hat = &mut l_complex;
    for i in 0..n_total {
        f_hat[i] = Complex64::new(local_field[i], 0.0);
    }
    fft_ws.fft3d(f_hat);
    for i in 0..n_total {
        f_hat[i] *= rhs[i];
    }
    (fft_ws, inv_a, l_complex)
}

/// Pre-compute the continuous-kernel spectral operators of the nonlinear solvers that are
/// not FANSI's (QSM.jl's NLTV, AMP-PE). The FANSI-family ports take their kernel from
/// [`prepare_fansi_kernel`] instead.
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

/// The dipole kernel a FANSI-family solver (NDI, nlTV, nlTGV, L1-QSM, WH-QSM, HD-QSM) uses.
///
/// FANSI's solvers take the kernel as an input (`params.K`), and every FANSI script, like the
/// QSM-CI wrappers of the verified originals (FANSI-toolbox 93044a3, HD-QSM c456a94), builds it
/// with `dipole_kernel_angulated(N, voxel_size, B0_dir)`, whose default `mode = 1` is the
/// discrete kernel [`FansiKernel::Discrete`]. HDQSM.m ships no kernel and is "based on" FANSI;
/// its wrapper uses the same function. The discrete kernel is consistent with the unit
/// forward differences these solvers regularise with (both come from the same `E_j`).
///
/// Decision (Ashley, 2026-11-02; QSM.rs#399–#403): the FANSI-family ports use FANSI's own
/// discrete kernel by default, as the originals do; the continuous kernel the rest of the crate
/// uses is kept as an option. Measured with the continuous kernel against the verified originals
/// (QSM-CI parity run 41384710780, sim / KKI χ NRMSE): NDI 10.6 / 9.5 %, nlTV 11.0 / 9.5 %,
/// nlTGV 11.0 / 9.6 %, L1-QSM 14.4 / 15.5 %, WH-QSM 21.4 / 50.7 %, HD-QSM 9.1 / 7.1 %; a numpy
/// transcription of `ndi.m` on the original's own inputs is 0.0001 % from it with the discrete
/// kernel and 10.6 % with the continuous one.
/// With the discrete kernel every pair is at rounding (each method's module docs, "Parity"). The
/// phantom measurements quoted in the methods' `te` docs were re-measured with the discrete kernel
/// (QSM.rs#418); the echo-time choices they support did not change.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FansiKernel {
    /// FANSI's `dipole_kernel_angulated` mode 1 (FANSI-toolbox 93044a3): the kernel of unit
    /// forward differences, `E_j = (1 - exp(2*pi*i*k_j/N_j)) / r_j` with the voxel size `r`
    /// normalised by its maximum, `D = 1/3 - |sum_j E_j*b_j|^2 / sum_j |E_j|^2`, `D(0) = 0`
    /// ([`fansi_dipole_kernel`]).
    #[default]
    Discrete,
    /// The continuous kernel `1/3 - (k.b)^2/|k|^2` of [`crate::kernels::dipole::dipole_kernel`]
    /// (`dipole_kernel_angulated` mode 0 up to its `k` normalisation).
    Continuous,
}

impl FansiKernel {
    /// The kernel in FFT order, `i + nx*(j + ny*k)`.
    pub fn build(self, grid: &Grid, bdir: (f64, f64, f64)) -> Vec<f64> {
        match self {
            FansiKernel::Discrete => fansi_dipole_kernel(grid, bdir),
            FansiKernel::Continuous => dipole_kernel(grid, bdir),
        }
    }
}

/// FANSI's discrete dipole kernel, `dipole_kernel_angulated(N, spatial_res, B0_dir)` with its
/// default `mode = 1` (FANSI-toolbox 93044a3):
///
/// ```matlab
/// spatial_res = spatial_res/max(spatial_res(:));
/// E_j = (1 - exp(2i*pi*k_j/N_j)) / spatial_res(j);    % k_j = 0..N_j-1
/// kernel = 1/3 - |E1*B0(1) + E2*B0(2) + E3*B0(3)|^2 ./ (|E1|^2 + |E2|^2 + |E3|^2);
/// kernel(1,1,1) = 0;
/// ```
///
/// `bdir` is normalised first (the QSM-CI wrappers pass `B0_dir/norm(B0_dir)`). MATLAB builds
/// it in single precision (`N = single(N)`); this is double, as the crate's other kernels.
/// FFT order, `i + nx*(j + ny*k)`.
pub fn fansi_dipole_kernel(grid: &Grid, bdir: (f64, f64, f64)) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let (vx, vy, vz) = grid.voxel_size;
    let vmax = vx.max(vy).max(vz);
    let (bx, by, bz) = bdir;
    let bn = (bx * bx + by * by + bz * bz).sqrt();
    let b = [bx / bn, by / bn, bz / bn];
    let e = |k: usize, n: usize, r: f64| {
        let a = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
        Complex64::new(1.0 - a.cos(), -a.sin()) / (r / vmax)
    };
    let ex: Vec<Complex64> = (0..nx).map(|i| e(i, nx, vx)).collect();
    let ey: Vec<Complex64> = (0..ny).map(|j| e(j, ny, vy)).collect();
    let ez: Vec<Complex64> = (0..nz).map(|k| e(k, nz, vz)).collect();
    let mut d = Vec::with_capacity(nx * ny * nz);
    for e3 in &ez {
        for e2 in &ey {
            for e1 in &ex {
                let lap = e1.norm_sqr() + e2.norm_sqr() + e3.norm_sqr();
                let proj = (e1 * b[0] + e2 * b[1] + e3 * b[2]).norm_sqr();
                d.push(if lap > 0.0 { 1.0 / 3.0 - proj / lap } else { 0.0 });
            }
        }
    }
    d
}

/// [`Fft3dWorkspace`] and the FANSI-family dipole kernel `kernel` selects (see [`FansiKernel`]).
pub fn prepare_fansi_kernel(
    grid: &Grid,
    bdir: (f64, f64, f64),
    kernel: FansiKernel,
) -> (Fft3dWorkspace, Vec<f64>) {
    (Fft3dWorkspace::new(grid.nx(), grid.ny(), grid.nz()), kernel.build(grid, bdir))
}

/// FANSI's `EE2 = |E1|^2 + |E2|^2 + |E3|^2` with `E_j = 1 - exp(2*pi*i*k_j/N_j)`: the
/// spectral Laplacian of unit-spacing forward differences, as FANSI's nlTV.m, nlTGV.m and
/// nlL1TV.m build it (whatever the voxel size). Pair it with `fgrad_inplace`/`bdiv_inplace`
/// on a unit-voxel grid. FFT order, `i + nx*(j + ny*k)`.
pub fn fansi_unit_ee2(dims: (usize, usize, usize)) -> Vec<f64> {
    let (nx, ny, nz) = dims;
    let e = |k: usize, n: usize| 2.0 - 2.0 * (2.0 * std::f64::consts::PI * k as f64 / n as f64).cos();
    let mut ee2 = Vec::with_capacity(nx * ny * nz);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                ee2.push(e(i, nx) + e(j, ny) + e(k, nz));
            }
        }
    }
    ee2
}

/// ppm -> radians factor `2*pi*gamma*b0*te` (gamma = [`GAMMA_BAR_MHZ_PER_T`](crate::constants::GAMMA_BAR_MHZ_PER_T)),
/// for the FANSI solvers, which work on the phase in radians while the crate's inversions take
/// ppm. FANSI's scripts use `gyro = 2*pi*42.58`; QSM.rs uses the shared constant, which is
/// closer to the CODATA shielded-proton value (see [`crate::constants`]).
pub fn fansi_ppm_to_rad(b0: f64, te: f64) -> f64 {
    2.0 * std::f64::consts::PI * crate::constants::GAMMA_BAR_MHZ_PER_T * b0 * te
}

/// FANSI's data weight from the magnitude, as in its in vivo example (script_invivo.m):
/// `min(2*mask*mag/max(mag), 1)`.
pub fn fansi_magnitude_weight(mask: &[u8], magnitude: &[f64]) -> Vec<f64> {
    let mag_max = magnitude.iter().cloned().fold(0.0f64, f64::max);
    (0..mask.len())
        .map(|i| {
            if mask[i] == 0 || mag_max <= 0.0 {
                0.0
            } else {
                (2.0 * magnitude[i] / mag_max).min(1.0)
            }
        })
        .collect()
}

#[cfg(test)]
mod fansi_kernel_tests {
    use super::*;

    /// `dipole_kernel_angulated` mode 1, written out for one bin: B0 oblique, anisotropic voxels
    /// (normalised by the largest), DC 0, and the continuous kernel differs.
    #[test]
    fn discrete_kernel_matches_dipole_kernel_angulated() {
        let grid = Grid::new(6, 5, 4, 0.5, 1.0, 2.0);
        let b = (0.3, 0.1, 0.9);
        let d = fansi_dipole_kernel(&grid, b);
        assert_eq!(d[0], 0.0);
        let bn = (0.3f64 * 0.3 + 0.1 * 0.1 + 0.9 * 0.9).sqrt();
        let (i, j, k) = (2usize, 3usize, 1usize);
        let ee = |kk: usize, n: usize, r: f64| {
            let a = 2.0 * std::f64::consts::PI * kk as f64 / n as f64;
            (Complex64::new(1.0, 0.0) - Complex64::new(0.0, a).exp()) / (r / 2.0)
        };
        let (e1, e2, e3) = (ee(i, 6, 0.5), ee(j, 5, 1.0), ee(k, 4, 2.0));
        let lap = e1.norm_sqr() + e2.norm_sqr() + e3.norm_sqr();
        let proj = (e1 * (0.3 / bn) + e2 * (0.1 / bn) + e3 * (0.9 / bn)).norm_sqr();
        let want = 1.0 / 3.0 - proj / lap;
        let got = d[i + 6 * (j + 5 * k)];
        assert!((got - want).abs() < 1e-14, "{got} vs {want}");
        let c = FansiKernel::Continuous.build(&grid, b);
        assert!((c[i + 6 * (j + 5 * k)] - got).abs() > 1e-3);
        assert_eq!(FansiKernel::default(), FansiKernel::Discrete);
    }
}


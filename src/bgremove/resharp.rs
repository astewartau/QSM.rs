//! RESHARP background field removal
//!
//! Regularized SHARP — uses Tikhonov regularization instead of TSVD
//! truncation for more robust deconvolution of the SMV-filtered field.
//!
//! Solves: argmin ||M·ifft(C·fft(x)) - M·ifft(C·fft(y))||² + λ||x||²
//!
//! Where C = (1 - S) is the high-pass (delta minus SMV) kernel in k-space,
//! M is the eroded mask, and λ is the Tikhonov regularization parameter.
//!
//! ## Reference pipeline
//!
//! The first author's code, `background_field_removal/resharp.m` in github sunhongfu/QSM
//! (also shipped in QSMART's toolbox), defaults to a 3 mm kernel (`round(r/vox)` per axis, at
//! least 2 voxels, so ellipsoidal on anisotropic voxels), λ = 1e-4, and MATLAB `cgs` at
//! tolerance 1e-6 for at most 200 iterations. It zero-pads by the kernel radius and keeps
//! voxels where the normalised SMV of the mask exceeds 0.999999. Its driver
//! `Siemens_3T/qsm_r2s_prisma.m` masks with FSL `bet2 -f 0.4 -w 2` on the first-echo
//! magnitude, unwraps each echo (bestpath by default), takes the total field from
//! `echofit.m` (magnitude-weighted, zero-intercept), multiplies the mask by the echo-fit
//! reliability map (`fit_thr` 40), and calls RESHARP with 3 mm, λ = 1e-3 and 500 iterations,
//! followed by TV (`tvdi`) or iLSQR. [`ResharpParams::default`] follows `resharp.m`'s own
//! defaults (3 mm, λ = 1e-4, at most 200 iterations), not this driver's λ and iterations.
//!
//! # Parity
//! Against `resharp.m` as the verified QSM-CI `resharp` container runs it (identical to licensed
//! MATLAB R2023b): the fixture `tests/resharp_sunhongfu_parity.rs` (anisotropic voxels; converged,
//! another kernel and λ, and a 5-iteration cap) agrees to 1e-14 relative with identical eroded
//! masks, and the QSM-CI `resharp` pair (QSM-CI's simulated head and the KKI travelling head)
//! agrees at the container's float32 output (KKI 2.5e-6 % NRMSE, astewartau/QSM.rs#404). Until
//! #404 the port solved by conjugate gradients, which stops unconverged at 200 iterations where
//! `cgs` converges (3.06 % on KKI); it also used a ball of radius `r` mm instead of `resharp.m`'s
//! `round(r/voxel)`-voxel ellipsoid, and did not pad.
//!
//! Reference:
//! Sun, H. and Wilman, A.H. (2013).
//! "Background field removal using spherical mean value filtering and Tikhonov regularization."
//! Magn Reson Med, 71(3):1151-1157. https://doi.org/10.1002/mrm.24765
//!
//! # Pipeline rules
//!
//! The facts above that a configuration can depart from are published as rules in
//! [`crate::pipeline::rules::RULES`] (each with its strength, basis, reason and source): `resharp.radius`, `resharp.iterations`.

use num_complex::Complex64;
use crate::Grid;
use crate::fft::{fft3d, ifft3d, fft_real_kernel};
use crate::kernels::smv::erode_mask_smv;
use crate::solvers::cgs_solve_with_progress;

/// RESHARP algorithm parameters.
///
/// The defaults are the first author's code: `background_field_removal/resharp.m` in
/// github.com/sunhongfu/QSM (c75b138), whose own defaults are `ker_rad = 3`, `tik_reg = 1e-4`,
/// `iter_num = 200` (l. 24-34) and `cgs(@Afun, b, 1e-6, iter_num)` (l. 92). Hongfu Sun's
/// scanner drivers in the same repository override them inconsistently (3T Siemens drivers
/// 3 mm / 1e-3 / 500, Philips 3T 3 mm / 1e-4 / 200, 1.5T and 4.7T drivers 4-5 mm / 5e-4, GE
/// variants 2-4 mm), so the function's defaults are the reference here; the radius, 3 mm, is
/// also what every 3T driver uses.
///
/// Measured against the previous defaults (6 mm, 30 iterations), ROMEO field then RTS, on
/// the voxels both keep: on the qsm-forward phantom χ XSIM 0.283 → 0.303, NRMSE 80.4 → 79.7 %,
/// r 0.597 → 0.606; on an in-vivo 3T Philips scan, agreement with the UK Biobank pipeline's
/// χ (STI Suite V-SHARP + iLSQR) XSIM 0.647 → 0.688, NRMSE 63.2 → 62.2 %, r 0.833 → 0.838. The
/// 3 mm kernel also keeps ~18 % more voxels at the brain edge, where agreement is lower (in
/// vivo on its own mask, XSIM 0.653, NRMSE 71.1 %). The radius drives the in-vivo gain
/// (3 mm / 30 iterations: 0.727) and the iterations the phantom gain (6 mm / 200: 0.302).
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct ResharpParams {
    /// SMV kernel radius in mm (default 3, `resharp.m`'s `ker_rad`); per axis it is
    /// `round(radius/voxel)` voxels, at least 2 ([`resharp_kernel_radii`]), so the kernel is an
    /// ellipsoid on anisotropic voxels. QSM.rs previously used 6 mm, which had no source.
    pub radius: f64,
    /// Tikhonov regularization parameter (default 1e-4, `resharp.m`'s `tik_reg`)
    pub tik_reg: f64,
    /// Relative residual tolerance of the `cgs` solve (default 1e-6, as `resharp.m` passes it)
    pub tol: f64,
    /// Maximum `cgs` iterations (default 200, `resharp.m`'s `iter_num`), each making two operator
    /// calls as MATLAB's `cgs` does. QSM.rs previously defaulted to 30, which had no source and
    /// stopped well short of the tolerance.
    pub max_iter: usize,
}

impl Default for ResharpParams {
    fn default() -> Self {
        Self {
            radius: 3.0,
            tik_reg: 1e-4,
            tol: 1e-6,
            max_iter: 200,
        }
    }
}

/// The kernel radius in voxels per axis, as `resharp.m` (l. 37-42): `round(radius/voxel)`, at
/// least 2.
pub fn resharp_kernel_radii(grid: &Grid, radius: f64) -> [usize; 3] {
    let (vx, vy, vz) = grid.voxel_size;
    // MATLAB's round: half away from zero, as f64::round.
    [vx, vy, vz].map(|v| ((radius / v).round() as usize).max(2))
}

/// `resharp.m`'s kernel (l. 43-45): the voxels of the `rx×ry×rz` ellipsoid
/// (`X²/rx² + Y²/ry² + Z²/rz² ≤ 1`, in voxels), normalised to sum 1, centred at index (0,0,0)
/// of an `n` grid for FFT use (Fortran order).
fn resharp_kernel(n: (usize, usize, usize), r: [usize; 3]) -> Vec<f64> {
    let (nx, ny, nz) = n;
    let [rx, ry, rz] = r.map(|v| v as i64);
    let mut s = vec![0.0; nx * ny * nz];
    let mut count = 0.0;
    let wrap = |d: i64, n: usize| if d < 0 { (d + n as i64) as usize } else { d as usize };
    for z in -rz..=rz {
        for y in -ry..=ry {
            for x in -rx..=rx {
                let e = (x * x) as f64 / (rx * rx) as f64 + (y * y) as f64 / (ry * ry) as f64
                    + (z * z) as f64 / (rz * rz) as f64;
                if e <= 1.0 {
                    s[wrap(x, nx) + nx * (wrap(y, ny) + ny * wrap(z, nz))] = 1.0;
                    count += 1.0;
                }
            }
        }
    }
    s.iter_mut().for_each(|v| *v /= count);
    s
}

/// RESHARP background field removal, as `resharp.m` (sunhongfu/QSM c75b138).
///
/// Like `resharp.m`, it zero-pads the field and mask by the kernel radius (l. 52-53), erodes
/// the mask to the voxels whose kernel-weighted mean of the mask exceeds 0.999999 (l. 57-60),
/// solves `(H'H + λ)x = H'H y` (`H = M·ifft(DKER·fft(·))`, `DKER = δ − kernel`) with MATLAB's
/// `cgs` at tolerance 1e-6 ([`crate::solvers::cgs_solve_with_progress`]: conjugate gradients
/// squared with `cgs`'s stopping rules and returned iterate, l. 92), masks the result and crops
/// the padding. `resharp.m` builds `DKER` from the kernel stored at index `(r+1)` and undoes the
/// shift with `circshift`; this centres the kernel at index 0, the same operator.
///
/// Parity (QSMxT/QSM-CI `resharp` pair, run in astewartau/QSM.rs#404): the port previously ran
/// conjugate gradients, which a numpy `resharp.m` showed to be exactly the 3.06 % (KKI) /
/// 0.65 % (sim) gap: CG stops unconverged at 200 iterations where `cgs` converges in 199.
///
/// # Arguments
/// * `field` - Unwrapped total field (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz), 1 = inside ROI
/// * `grid` - Volume dimensions and voxel sizes
/// * `params` - RESHARP algorithm parameters
/// * `progress` - Progress callback (current_iteration, max_iterations)
///
/// # Returns
/// (local_field, eroded_mask)
pub fn resharp(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    params: &ResharpParams,
    mut progress: impl FnMut(usize, usize),
) -> (Vec<f64>, Vec<u8>) {
    let (nx, ny, nz) = grid.dims;
    let r = resharp_kernel_radii(grid, params.radius);
    let (px, py, pz) = (nx + 2 * r[0], ny + 2 * r[1], nz + 2 * r[2]);
    let n_pad = px * py * pz;
    let pidx = |i: usize, j: usize, k: usize| (i + r[0]) + px * ((j + r[1]) + py * (k + r[2]));

    let mut field_p = vec![0.0; n_pad];
    let mut mask_p = vec![0u8; n_pad];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let s = i + nx * (j + ny * k);
                field_p[pidx(i, j, k)] = field[s];
                mask_p[pidx(i, j, k)] = mask[s];
            }
        }
    }
    let pgrid = Grid::new(px, py, pz, grid.voxel_size.0, grid.voxel_size.1, grid.voxel_size.2);

    let s_kernel = resharp_kernel((px, py, pz), r);
    let s_fft = fft_real_kernel(&s_kernel, px, py, pz);
    // DKER = 1 - S (delta minus kernel) in k-space
    let dker: Vec<f64> = s_fft.iter().map(|&s| 1.0 - s).collect();

    // resharp.m: mask_ero(convn(mask, ker, 'same') > 0.999999) = 1
    let eroded_p = erode_mask_smv(&mask_p, &s_fft, &pgrid, 0.999999);
    let eroded_f64: Vec<f64> = eroded_p.iter().map(|&m| m as f64).collect();

    // b = H'(H(field))
    let b = apply_ht_h(&dker, &eroded_f64, &field_p, px, py, pz);

    let tik_reg = params.tik_reg;
    let x0 = vec![0.0; n_pad];
    let (x, _report) = cgs_solve_with_progress(
        |x_vec| {
            let mut result = apply_ht_h(&dker, &eroded_f64, x_vec, px, py, pz);
            for (ri, &xi) in result.iter_mut().zip(x_vec) {
                *ri += tik_reg * xi;
            }
            result
        },
        &b,
        &x0,
        params.tol,
        params.max_iter,
        &mut progress,
    );

    let mut local_field = vec![0.0; nx * ny * nz];
    let mut eroded_mask = vec![0u8; nx * ny * nz];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let p = pidx(i, j, k);
                let s = i + nx * (j + ny * k);
                if eroded_p[p] == 1 {
                    local_field[s] = x[p];
                    eroded_mask[s] = 1;
                }
            }
        }
    }
    (local_field, eroded_mask)
}

/// Apply H'H to a vector: ifft(DKER * fft(M * ifft(DKER * fft(x))))
///
/// H(x)  = M * ifft(DKER * fft(x))
/// H'(y) = ifft(DKER * fft(M * y))
fn apply_ht_h(
    dker: &[f64],
    mask: &[f64],
    x: &[f64],
    nx: usize, ny: usize, nz: usize,
) -> Vec<f64> {
    let n_total = nx * ny * nz;

    // Step 1: H(x) = M * ifft(DKER * fft(x))
    let mut tmp: Vec<Complex64> = x.iter()
        .map(|&v| Complex64::new(v, 0.0))
        .collect();
    fft3d(&mut tmp, nx, ny, nz);

    for i in 0..n_total {
        tmp[i] *= dker[i];
    }
    ifft3d(&mut tmp, nx, ny, nz);

    // Apply mask
    for i in 0..n_total {
        tmp[i] *= mask[i];
    }

    // Step 2: H'(Hx) = ifft(DKER * fft(M * Hx))
    fft3d(&mut tmp, nx, ny, nz);

    for i in 0..n_total {
        tmp[i] *= dker[i];
    }
    ifft3d(&mut tmp, nx, ny, nz);

    tmp.iter().map(|c| c.re).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resharp_zero_field() {
        let n = 16;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = ResharpParams { radius: 2.0, tik_reg: 1e-4, tol: 1e-6, max_iter: 50 };

        let (local, _) = resharp(&field, &mask, &grid, &params, |_, _| {});

        for &val in local.iter() {
            assert!(val.abs() < 1e-8, "Zero field should give zero local field, got {}", val);
        }
    }

    #[test]
    fn test_resharp_finite() {
        let n = 16;
        let field: Vec<f64> = (0..n*n*n).map(|i| (i as f64) * 0.01).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = ResharpParams { radius: 2.0, tik_reg: 1e-4, tol: 1e-6, max_iter: 50 };

        let (local, eroded) = resharp(&field, &mask, &grid, &params, |_, _| {});

        for (i, &val) in local.iter().enumerate() {
            assert!(val.is_finite(), "Local field should be finite at index {}", i);
        }

        let eroded_count: usize = eroded.iter().map(|&m| m as usize).sum();
        assert!(eroded_count > 0, "Eroded mask should have some voxels");
    }
}

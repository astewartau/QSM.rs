//! Gradient operators for QSM
//!
//! Forward difference gradient and backward divergence operators
//! used in TV regularization and other algorithms.

/// Forward difference gradient operator (in-place)
///
/// Computes forward differences along each axis with periodic boundary conditions.
/// Writes results directly into pre-allocated output buffers.
///
/// # Arguments
/// * `gx`, `gy`, `gz` - Output gradient components (must be pre-allocated to nx*ny*nz)
/// * `x` - Input array (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes
#[inline]
pub fn fgrad_inplace(
    gx: &mut [f64], gy: &mut [f64], gz: &mut [f64],
    x: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) {
    let hx = 1.0 / vsx;
    let hy = 1.0 / vsy;
    let hz = 1.0 / vsz;

    // Fortran order: index = i + j*nx + k*nx*ny
    for k in 0..nz {
        let kp1 = if k + 1 < nz { k + 1 } else { 0 };
        let k_offset = k * nx * ny;
        let kp1_offset = kp1 * nx * ny;

        for j in 0..ny {
            let jp1 = if j + 1 < ny { j + 1 } else { 0 };
            let j_offset = j * nx;
            let jp1_offset = jp1 * nx;

            for i in 0..nx {
                let ip1 = if i + 1 < nx { i + 1 } else { 0 };

                let idx = i + j_offset + k_offset;
                let idx_xp = ip1 + j_offset + k_offset;
                let idx_yp = i + jp1_offset + k_offset;
                let idx_zp = i + j_offset + kp1_offset;

                let x_val = x[idx];
                gx[idx] = (x[idx_xp] - x_val) * hx;
                gy[idx] = (x[idx_yp] - x_val) * hy;
                gz[idx] = (x[idx_zp] - x_val) * hz;
            }
        }
    }
}

/// Backward divergence operator (in-place)
///
/// Computes backward divergence with periodic boundary conditions.
/// Writes result directly into pre-allocated output buffer.
///
/// # Arguments
/// * `div` - Output divergence (must be pre-allocated to nx*ny*nz)
/// * `gx`, `gy`, `gz` - Gradient components
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes
#[inline]
pub fn bdiv_inplace(
    div: &mut [f64],
    gx: &[f64], gy: &[f64], gz: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) {
    let hx = 1.0 / vsx;
    let hy = 1.0 / vsy;
    let hz = 1.0 / vsz;

    // Fortran order: index = i + j*nx + k*nx*ny
    for k in 0..nz {
        let km1 = if k == 0 { nz - 1 } else { k - 1 };
        let k_offset = k * nx * ny;
        let km1_offset = km1 * nx * ny;

        for j in 0..ny {
            let jm1 = if j == 0 { ny - 1 } else { j - 1 };
            let j_offset = j * nx;
            let jm1_offset = jm1 * nx;

            for i in 0..nx {
                let im1 = if i == 0 { nx - 1 } else { i - 1 };

                let idx = i + j_offset + k_offset;
                let idx_xm = im1 + j_offset + k_offset;
                let idx_ym = i + jm1_offset + k_offset;
                let idx_zm = i + j_offset + km1_offset;

                // Negative divergence (adjoint of forward gradient)
                div[idx] = (gx[idx] - gx[idx_xm]) * hx
                         + (gy[idx] - gy[idx_ym]) * hy
                         + (gz[idx] - gz[idx_zm]) * hz;
            }
        }
    }
}

/// Forward difference gradient operator
///
/// Computes forward differences along each axis with periodic boundary conditions.
///
/// # Arguments
/// * `x` - Input array (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes
///
/// # Returns
/// Tuple of (gx, gy, gz) gradient components
pub fn fgrad(
    x: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n_total = nx * ny * nz;
    let mut gx = vec![0.0; n_total];
    let mut gy = vec![0.0; n_total];
    let mut gz = vec![0.0; n_total];

    let hx = 1.0 / vsx;
    let hy = 1.0 / vsy;
    let hz = 1.0 / vsz;

    // Fortran order: index = i + j*nx + k*nx*ny
    for k in 0..nz {
        let kp1 = (k + 1) % nz;
        for j in 0..ny {
            let jp1 = (j + 1) % ny;
            for i in 0..nx {
                let ip1 = (i + 1) % nx;  // Periodic BC

                let idx = i + j * nx + k * nx * ny;
                let idx_xp = ip1 + j * nx + k * nx * ny;
                let idx_yp = i + jp1 * nx + k * nx * ny;
                let idx_zp = i + j * nx + kp1 * nx * ny;

                gx[idx] = (x[idx_xp] - x[idx]) * hx;
                gy[idx] = (x[idx_yp] - x[idx]) * hy;
                gz[idx] = (x[idx_zp] - x[idx]) * hz;
            }
        }
    }

    (gx, gy, gz)
}

/// Backward divergence operator (negative adjoint of forward gradient)
///
/// Computes backward divergence with periodic boundary conditions.
///
/// # Arguments
/// * `gx`, `gy`, `gz` - Gradient components
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes
///
/// # Returns
/// Divergence (negative)
pub fn bdiv(
    gx: &[f64], gy: &[f64], gz: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> Vec<f64> {
    let n_total = nx * ny * nz;
    let mut div = vec![0.0; n_total];

    let hx = 1.0 / vsx;
    let hy = 1.0 / vsy;
    let hz = 1.0 / vsz;

    // Fortran order: index = i + j*nx + k*nx*ny
    for k in 0..nz {
        let km1 = if k == 0 { nz - 1 } else { k - 1 };
        for j in 0..ny {
            let jm1 = if j == 0 { ny - 1 } else { j - 1 };
            for i in 0..nx {
                let im1 = if i == 0 { nx - 1 } else { i - 1 };  // Periodic BC

                let idx = i + j * nx + k * nx * ny;
                let idx_xm = im1 + j * nx + k * nx * ny;
                let idx_ym = i + jm1 * nx + k * nx * ny;
                let idx_zm = i + j * nx + km1 * nx * ny;

                // Negative divergence (adjoint of forward gradient)
                div[idx] = (gx[idx] - gx[idx_xm]) * hx
                         + (gy[idx] - gy[idx_ym]) * hy
                         + (gz[idx] - gz[idx_zm]) * hz;
            }
        }
    }

    div
}

/// Compute gradient magnitude squared: |∇x|² = gx² + gy² + gz²
pub fn grad_magnitude_squared(
    gx: &[f64], gy: &[f64], gz: &[f64]
) -> Vec<f64> {
    gx.iter().zip(gy.iter()).zip(gz.iter())
        .map(|((&gxi, &gyi), &gzi)| gxi * gxi + gyi * gyi + gzi * gzi)
        .collect()
}

// ============================================================================
// F32 (Single Precision) Gradient Functions
// ============================================================================

/// Forward difference gradient operator (in-place, f32)
#[inline]
pub fn fgrad_inplace_f32(
    gx: &mut [f32], gy: &mut [f32], gz: &mut [f32],
    x: &[f32],
    nx: usize, ny: usize, nz: usize,
    vsx: f32, vsy: f32, vsz: f32,
) {
    let hx = 1.0 / vsx;
    let hy = 1.0 / vsy;
    let hz = 1.0 / vsz;

    for k in 0..nz {
        let k_offset = k * nx * ny;

        for j in 0..ny {
            let j_offset = j * nx;

            for i in 0..nx {
                let idx = i + j_offset + k_offset;
                let x_val = x[idx];

                // Forward difference with zero boundary (matching Julia)
                gx[idx] = if i + 1 < nx {
                    (x[idx + 1] - x_val) * hx
                } else {
                    0.0
                };

                gy[idx] = if j + 1 < ny {
                    (x[i + (j + 1) * nx + k_offset] - x_val) * hy
                } else {
                    0.0
                };

                gz[idx] = if k + 1 < nz {
                    (x[i + j_offset + (k + 1) * nx * ny] - x_val) * hz
                } else {
                    0.0
                };
            }
        }
    }
}

/// Backward divergence operator (in-place, f32)
/// Uses zero boundary conditions (matching Julia)
#[inline]
pub fn bdiv_inplace_f32(
    div: &mut [f32],
    gx: &[f32], gy: &[f32], gz: &[f32],
    nx: usize, ny: usize, nz: usize,
    vsx: f32, vsy: f32, vsz: f32,
) {
    let hx = 1.0 / vsx;
    let hy = 1.0 / vsy;
    let hz = 1.0 / vsz;

    for k in 0..nz {
        let k_offset = k * nx * ny;

        for j in 0..ny {
            let j_offset = j * nx;

            for i in 0..nx {
                let idx = i + j_offset + k_offset;

                // Zero at boundary (matching Julia)
                let gx_xm = if i > 0 { gx[(i - 1) + j_offset + k_offset] } else { 0.0 };
                let gy_ym = if j > 0 { gy[i + (j - 1) * nx + k_offset] } else { 0.0 };
                let gz_zm = if k > 0 { gz[i + j_offset + (k - 1) * nx * ny] } else { 0.0 };

                div[idx] = (gx[idx] - gx_xm) * hx
                         + (gy[idx] - gy_ym) * hy
                         + (gz[idx] - gz_zm) * hz;
            }
        }
    }
}

/// Forward difference gradient operator (allocating, f32)
pub fn fgrad_f32(
    x: &[f32],
    nx: usize, ny: usize, nz: usize,
    vsx: f32, vsy: f32, vsz: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n_total = nx * ny * nz;
    let mut gx = vec![0.0f32; n_total];
    let mut gy = vec![0.0f32; n_total];
    let mut gz = vec![0.0f32; n_total];
    fgrad_inplace_f32(&mut gx, &mut gy, &mut gz, x, nx, ny, nz, vsx, vsy, vsz);
    (gx, gy, gz)
}

// ============================================================================
// Symmetric Gradient (for TGV)
// ============================================================================

/// Symmetric gradient operator for TGV regularization (in-place, f32)
///
/// Computes the symmetric gradient tensor from a vector field w = (wx, wy, wz).
/// The output is a 6-component symmetric tensor:
///   q[0] = ∂wx/∂x (Sxx)
///   q[1] = (∂wx/∂y + ∂wy/∂x) / 2 (Sxy)
///   q[2] = (∂wx/∂z + ∂wz/∂x) / 2 (Sxz)
///   q[3] = ∂wy/∂y (Syy)
///   q[4] = (∂wy/∂z + ∂wz/∂y) / 2 (Syz)
///   q[5] = ∂wz/∂z (Szz)
#[inline]
pub fn symgrad_inplace_f32(
    sxx: &mut [f32], sxy: &mut [f32], sxz: &mut [f32],
    syy: &mut [f32], syz: &mut [f32], szz: &mut [f32],
    wx: &[f32], wy: &[f32], wz: &[f32],
    nx: usize, ny: usize, nz: usize,
    vsx: f32, vsy: f32, vsz: f32,
) {
    let hx = 1.0 / vsx;
    let hy = 1.0 / vsy;
    let hz = 1.0 / vsz;

    for k in 0..nz {
        let k_offset = k * nx * ny;

        for j in 0..ny {
            let j_offset = j * nx;

            for i in 0..nx {
                let idx = i + j_offset + k_offset;

                let wx0 = wx[idx];
                let wy0 = wy[idx];
                let wz0 = wz[idx];

                // X derivatives (zero at boundary, matching Julia)
                if i + 1 < nx {
                    let idx_xp = (i + 1) + j_offset + k_offset;
                    sxx[idx] = (wx[idx_xp] - wx0) * hx;
                    // Contributions to off-diagonal terms
                    let dwy_dx = (wy[idx_xp] - wy0) * hx;
                    let dwz_dx = (wz[idx_xp] - wz0) * hx;
                    sxy[idx] = dwy_dx * 0.5;
                    sxz[idx] = dwz_dx * 0.5;
                } else {
                    sxx[idx] = 0.0;
                    sxy[idx] = 0.0;
                    sxz[idx] = 0.0;
                }

                // Y derivatives (zero at boundary)
                if j + 1 < ny {
                    let idx_yp = i + (j + 1) * nx + k_offset;
                    syy[idx] = (wy[idx_yp] - wy0) * hy;
                    let dwx_dy = (wx[idx_yp] - wx0) * hy;
                    let dwz_dy = (wz[idx_yp] - wz0) * hy;
                    sxy[idx] += dwx_dy * 0.5;
                    syz[idx] = dwz_dy * 0.5;
                } else {
                    syy[idx] = 0.0;
                    syz[idx] = 0.0;
                }

                // Z derivatives (zero at boundary)
                if k + 1 < nz {
                    let idx_zp = i + j_offset + (k + 1) * nx * ny;
                    szz[idx] = (wz[idx_zp] - wz0) * hz;
                    let dwx_dz = (wx[idx_zp] - wx0) * hz;
                    let dwy_dz = (wy[idx_zp] - wy0) * hz;
                    sxz[idx] += dwx_dz * 0.5;
                    syz[idx] += dwy_dz * 0.5;
                } else {
                    szz[idx] = 0.0;
                }
            }
        }
    }
}

/// Divergence of symmetric tensor field (adjoint of symgrad)
///
/// Computes the divergence of a 6-component symmetric tensor field,
/// producing a 3-component vector field.
/// This is the adjoint of symgrad_inplace_f32.
/// Uses zero boundary conditions (matching Julia).
#[inline]
pub fn symdiv_inplace_f32(
    divx: &mut [f32], divy: &mut [f32], divz: &mut [f32],
    sxx: &[f32], sxy: &[f32], sxz: &[f32],
    syy: &[f32], syz: &[f32], szz: &[f32],
    nx: usize, ny: usize, nz: usize,
    vsx: f32, vsy: f32, vsz: f32,
) {
    let hx = 1.0 / vsx;
    let hy = 1.0 / vsy;
    let hz = 1.0 / vsz;

    for k in 0..nz {
        let k_offset = k * nx * ny;

        for j in 0..ny {
            let j_offset = j * nx;

            for i in 0..nx {
                let idx = i + j_offset + k_offset;

                // Divergence of first row of tensor: div([Sxx, Sxy, Sxz])
                // Using backward difference with zero at boundary
                let sxx_xm = if i > 0 { sxx[(i - 1) + j_offset + k_offset] } else { 0.0 };
                let sxy_ym = if j > 0 { sxy[i + (j - 1) * nx + k_offset] } else { 0.0 };
                let sxz_zm = if k > 0 { sxz[i + j_offset + (k - 1) * nx * ny] } else { 0.0 };

                divx[idx] = (sxx[idx] - sxx_xm) * hx
                          + (sxy[idx] - sxy_ym) * hy
                          + (sxz[idx] - sxz_zm) * hz;

                // Divergence of second row: div([Sxy, Syy, Syz])
                let sxy_xm = if i > 0 { sxy[(i - 1) + j_offset + k_offset] } else { 0.0 };
                let syy_ym = if j > 0 { syy[i + (j - 1) * nx + k_offset] } else { 0.0 };
                let syz_zm = if k > 0 { syz[i + j_offset + (k - 1) * nx * ny] } else { 0.0 };

                divy[idx] = (sxy[idx] - sxy_xm) * hx
                          + (syy[idx] - syy_ym) * hy
                          + (syz[idx] - syz_zm) * hz;

                // Divergence of third row: div([Sxz, Syz, Szz])
                let sxz_xm = if i > 0 { sxz[(i - 1) + j_offset + k_offset] } else { 0.0 };
                let syz_ym = if j > 0 { syz[i + (j - 1) * nx + k_offset] } else { 0.0 };
                let szz_zm = if k > 0 { szz[i + j_offset + (k - 1) * nx * ny] } else { 0.0 };

                divz[idx] = (sxz[idx] - sxz_xm) * hx
                          + (syz[idx] - syz_ym) * hy
                          + (szz[idx] - szz_zm) * hz;
            }
        }
    }
}

/// Forward difference gradient operator (in-place, f32) - masked version
/// Only computes gradient where mask is non-zero
#[inline]
pub fn fgrad_masked_inplace_f32(
    gx: &mut [f32], gy: &mut [f32], gz: &mut [f32],
    x: &[f32],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f32, vsy: f32, vsz: f32,
) {
    let hx = 1.0 / vsx;
    let hy = 1.0 / vsy;
    let hz = 1.0 / vsz;

    for k in 0..nz {
        let kp1 = if k + 1 < nz { k + 1 } else { 0 };
        let k_offset = k * nx * ny;
        let kp1_offset = kp1 * nx * ny;

        for j in 0..ny {
            let jp1 = if j + 1 < ny { j + 1 } else { 0 };
            let j_offset = j * nx;
            let jp1_offset = jp1 * nx;

            for i in 0..nx {
                let ip1 = if i + 1 < nx { i + 1 } else { 0 };

                let idx = i + j_offset + k_offset;

                if mask[idx] == 0 {
                    gx[idx] = 0.0;
                    gy[idx] = 0.0;
                    gz[idx] = 0.0;
                    continue;
                }

                let idx_xp = ip1 + j_offset + k_offset;
                let idx_yp = i + jp1_offset + k_offset;
                let idx_zp = i + j_offset + kp1_offset;

                let x_val = x[idx];
                gx[idx] = (x[idx_xp] - x_val) * hx;
                gy[idx] = (x[idx_yp] - x_val) * hy;
                gz[idx] = (x[idx_zp] - x_val) * hz;
            }
        }
    }
}

/// Backward divergence operator (in-place, f32) - masked version
#[inline]
pub fn bdiv_masked_inplace_f32(
    div: &mut [f32],
    gx: &[f32], gy: &[f32], gz: &[f32],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f32, vsy: f32, vsz: f32,
) {
    let hx = 1.0 / vsx;
    let hy = 1.0 / vsy;
    let hz = 1.0 / vsz;

    for k in 0..nz {
        let k_offset = k * nx * ny;

        for j in 0..ny {
            let j_offset = j * nx;

            for i in 0..nx {
                let idx = i + j_offset + k_offset;

                if mask[idx] == 0 {
                    div[idx] = 0.0;
                    continue;
                }

                // Julia: div = mask[i]*g[i] - mask[i-1]*g[i-1] (0 if i<=1)
                let m = if mask[idx] != 0 { 1.0 } else { 0.0 };

                let gx_term = m * gx[idx] * hx - if i > 0 {
                    let idx_xm = (i - 1) + j_offset + k_offset;
                    let m_xm = if mask[idx_xm] != 0 { 1.0 } else { 0.0 };
                    m_xm * gx[idx_xm] * hx
                } else {
                    0.0
                };

                let gy_term = m * gy[idx] * hy - if j > 0 {
                    let idx_ym = i + (j - 1) * nx + k_offset;
                    let m_ym = if mask[idx_ym] != 0 { 1.0 } else { 0.0 };
                    m_ym * gy[idx_ym] * hy
                } else {
                    0.0
                };

                let gz_term = m * gz[idx] * hz - if k > 0 {
                    let idx_zm = i + j_offset + (k - 1) * nx * ny;
                    let m_zm = if mask[idx_zm] != 0 { 1.0 } else { 0.0 };
                    m_zm * gz[idx_zm] * hz
                } else {
                    0.0
                };

                div[idx] = gx_term + gy_term + gz_term;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grad_constant() {
        // Gradient of constant should be zero
        let n = 4;
        let x = vec![1.0; n * n * n];

        let (gx, gy, gz) = fgrad(&x, n, n, n, 1.0, 1.0, 1.0);

        for i in 0..n*n*n {
            assert!(gx[i].abs() < 1e-10);
            assert!(gy[i].abs() < 1e-10);
            assert!(gz[i].abs() < 1e-10);
        }
    }

    #[test]
    fn test_div_grad_adjoint() {
        // Check that <grad(x), h> = <x, -div(h)> (adjoint relationship)
        let n = 4;
        let x: Vec<f64> = (0..n*n*n).map(|i| (i as f64) * 0.1).collect();

        // Create an arbitrary vector field h
        let hx: Vec<f64> = (0..n*n*n).map(|i| ((i as f64) * 0.2).sin()).collect();
        let hy: Vec<f64> = (0..n*n*n).map(|i| ((i as f64) * 0.3).cos()).collect();
        let hz: Vec<f64> = (0..n*n*n).map(|i| ((i as f64) * 0.1).sin()).collect();

        let (gx, gy, gz) = fgrad(&x, n, n, n, 1.0, 1.0, 1.0);
        let div_h = bdiv(&hx, &hy, &hz, n, n, n, 1.0, 1.0, 1.0);

        // <grad(x), h> should equal <x, -div(h)>
        let lhs: f64 = gx.iter().zip(hx.iter())
            .chain(gy.iter().zip(hy.iter()))
            .chain(gz.iter().zip(hz.iter()))
            .map(|(&a, &b)| a * b)
            .sum();

        // Note: bdiv returns div, not -div, so we need to negate
        let rhs: f64 = x.iter().zip(div_h.iter())
            .map(|(&xi, &di)| -xi * di)
            .sum();

        let rel_err = (lhs - rhs).abs() / (lhs.abs() + rhs.abs() + 1e-10);
        assert!(rel_err < 1e-10, "Adjoint property failed: lhs={}, rhs={}, rel_err={}", lhs, rhs, rel_err);
    }

    #[test]
    fn test_fgrad_inplace_f32() {
        // Linear ramp in x: x[i,j,k] = i
        let nx = 4;
        let ny = 3;
        let nz = 3;
        let n = nx * ny * nz;
        let mut x = vec![0.0f32; n];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    x[i + j * nx + k * nx * ny] = i as f32;
                }
            }
        }

        let mut gx = vec![0.0f32; n];
        let mut gy = vec![0.0f32; n];
        let mut gz = vec![0.0f32; n];
        fgrad_inplace_f32(&mut gx, &mut gy, &mut gz, &x, nx, ny, nz, 1.0, 1.0, 1.0);

        // Interior x-gradient should be 1.0
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = i + j * nx + k * nx * ny;
                    if i + 1 < nx {
                        assert!(
                            (gx[idx] - 1.0).abs() < 1e-6,
                            "gx[{},{},{}] = {}, expected 1.0",
                            i, j, k, gx[idx]
                        );
                    } else {
                        // Boundary: zero BC
                        assert!(
                            gx[idx].abs() < 1e-6,
                            "gx at boundary should be 0, got {}",
                            gx[idx]
                        );
                    }
                    // y and z gradients should be 0 everywhere (x doesn't vary along y or z)
                    if j + 1 < ny {
                        assert!(gy[idx].abs() < 1e-6, "gy should be 0, got {}", gy[idx]);
                    }
                    if k + 1 < nz {
                        assert!(gz[idx].abs() < 1e-6, "gz should be 0, got {}", gz[idx]);
                    }
                }
            }
        }

        // Test with non-unit voxel size
        let vsx = 2.0f32;
        fgrad_inplace_f32(&mut gx, &mut gy, &mut gz, &x, nx, ny, nz, vsx, 1.0, 1.0);
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..(nx - 1) {
                    let idx = i + j * nx + k * nx * ny;
                    assert!(
                        (gx[idx] - 0.5).abs() < 1e-6,
                        "gx with vsx=2 should be 0.5, got {}",
                        gx[idx]
                    );
                }
            }
        }
    }

    #[test]
    fn test_symgrad_f32() {
        // Linear vector field: wx = y, wy = 0, wz = 0
        // => Sxx=0, Syy=0, Szz=0, Sxy = (dwx/dy + dwy/dx)/2 = (1+0)/2 = 0.5
        // Sxz=0, Syz=0
        let nx = 4;
        let ny = 4;
        let nz = 4;
        let n = nx * ny * nz;

        let mut wx = vec![0.0f32; n];
        let wy = vec![0.0f32; n];
        let wz = vec![0.0f32; n];

        // wx = j (y coordinate)
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    wx[i + j * nx + k * nx * ny] = j as f32;
                }
            }
        }

        let mut sxx = vec![0.0f32; n];
        let mut sxy = vec![0.0f32; n];
        let mut sxz = vec![0.0f32; n];
        let mut syy = vec![0.0f32; n];
        let mut syz = vec![0.0f32; n];
        let mut szz = vec![0.0f32; n];

        symgrad_inplace_f32(
            &mut sxx, &mut sxy, &mut sxz, &mut syy, &mut syz, &mut szz,
            &wx, &wy, &wz,
            nx, ny, nz, 1.0, 1.0, 1.0,
        );

        // Check interior points (away from boundaries)
        for k in 0..(nz - 1) {
            for j in 0..(ny - 1) {
                for i in 0..(nx - 1) {
                    let idx = i + j * nx + k * nx * ny;
                    assert!(
                        sxx[idx].abs() < 1e-6,
                        "sxx[{},{},{}] = {}, expected 0",
                        i, j, k, sxx[idx]
                    );
                    assert!(
                        (sxy[idx] - 0.5).abs() < 1e-6,
                        "sxy[{},{},{}] = {}, expected 0.5",
                        i, j, k, sxy[idx]
                    );
                    assert!(
                        sxz[idx].abs() < 1e-6,
                        "sxz[{},{},{}] = {}, expected 0",
                        i, j, k, sxz[idx]
                    );
                    assert!(
                        syy[idx].abs() < 1e-6,
                        "syy[{},{},{}] = {}, expected 0",
                        i, j, k, syy[idx]
                    );
                    assert!(
                        syz[idx].abs() < 1e-6,
                        "syz[{},{},{}] = {}, expected 0",
                        i, j, k, syz[idx]
                    );
                    assert!(
                        szz[idx].abs() < 1e-6,
                        "szz[{},{},{}] = {}, expected 0",
                        i, j, k, szz[idx]
                    );
                }
            }
        }
    }

    #[test]
    fn test_symdiv_f32() {
        // Constant tensor field => divergence should be zero at interior
        let nx = 4;
        let ny = 4;
        let nz = 4;
        let n = nx * ny * nz;

        let sxx = vec![1.0f32; n];
        let sxy = vec![0.5f32; n];
        let sxz = vec![0.0f32; n];
        let syy = vec![1.0f32; n];
        let syz = vec![0.0f32; n];
        let szz = vec![1.0f32; n];

        let mut divx = vec![0.0f32; n];
        let mut divy = vec![0.0f32; n];
        let mut divz = vec![0.0f32; n];

        symdiv_inplace_f32(
            &mut divx, &mut divy, &mut divz,
            &sxx, &sxy, &sxz, &syy, &syz, &szz,
            nx, ny, nz, 1.0, 1.0, 1.0,
        );

        // For a constant tensor field, backward differences give 0 at interior points
        // but non-zero at boundaries (i=0, j=0, k=0) due to zero BC
        for k in 1..nz {
            for j in 1..ny {
                for i in 1..nx {
                    let idx = i + j * nx + k * nx * ny;
                    assert!(
                        divx[idx].abs() < 1e-6,
                        "divx[{},{},{}] = {}, expected 0",
                        i, j, k, divx[idx]
                    );
                    assert!(
                        divy[idx].abs() < 1e-6,
                        "divy[{},{},{}] = {}, expected 0",
                        i, j, k, divy[idx]
                    );
                    assert!(
                        divz[idx].abs() < 1e-6,
                        "divz[{},{},{}] = {}, expected 0",
                        i, j, k, divz[idx]
                    );
                }
            }
        }

        // At i=0, j=0, k=0 the backward BC (0.0 for i-1) means
        // divx[0,0,0] = (sxx[0]-0)*1 + (sxy[0]-0)*1 + (sxz[0]-0)*1 = 1 + 0.5 + 0 = 1.5
        let idx_origin = 0;
        assert!(
            (divx[idx_origin] - 1.5).abs() < 1e-5,
            "divx at origin expected 1.5, got {}",
            divx[idx_origin]
        );
    }

    #[test]
    fn test_bdiv_masked_f32() {
        let nx = 4;
        let ny = 4;
        let nz = 4;
        let n = nx * ny * nz;

        // Create a mask that is 1 for the inner 2x2x2 cube
        let mut mask = vec![0u8; n];
        for k in 1..3 {
            for j in 1..3 {
                for i in 1..3 {
                    mask[i + j * nx + k * nx * ny] = 1;
                }
            }
        }

        // Constant gradient field
        let gx = vec![1.0f32; n];
        let gy = vec![1.0f32; n];
        let gz = vec![1.0f32; n];

        let mut div = vec![0.0f32; n];
        bdiv_masked_inplace_f32(&mut div, &gx, &gy, &gz, &mask, nx, ny, nz, 1.0, 1.0, 1.0);

        // Outside the mask, divergence should be zero
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = i + j * nx + k * nx * ny;
                    if mask[idx] == 0 {
                        assert!(
                            div[idx].abs() < 1e-6,
                            "div outside mask at [{},{},{}] = {}, expected 0",
                            i, j, k, div[idx]
                        );
                    }
                }
            }
        }

        // Inside the mask with a neighbor also in mask: constant field gives 0 divergence
        // At (2,2,2): mask[2,2,2]=1, mask[1,2,2]=1 => gx_term = 1*1 - 1*1 = 0, same for y,z
        let idx_222 = 2 + 2 * nx + 2 * nx * ny;
        assert!(
            div[idx_222].abs() < 1e-6,
            "div at inner point (2,2,2) should be 0 for constant field, got {}",
            div[idx_222]
        );

        // At (1,1,1): mask[1,1,1]=1, mask[0,1,1]=0
        // gx_term = 1*1*1 - 0 = 1 (since mask at i-1 is 0)
        // gy_term = 1*1*1 - 0 = 1
        // gz_term = 1*1*1 - 0 = 1
        // total = 3
        let idx_111 = 1 + 1 * nx + 1 * nx * ny;
        assert!(
            (div[idx_111] - 3.0).abs() < 1e-5,
            "div at boundary-of-mask (1,1,1) expected 3.0, got {}",
            div[idx_111]
        );
    }

    #[test]
    fn test_fgrad_bdiv_adjoint_f64() {
        // More thorough adjoint test with non-uniform voxels and larger grid
        let nx = 6;
        let ny = 7;
        let nz = 5;
        let n = nx * ny * nz;
        let vsx = 1.5;
        let vsy = 0.8;
        let vsz = 2.0;

        // Create non-trivial scalar field
        let x: Vec<f64> = (0..n)
            .map(|idx| {
                let i = idx % nx;
                let j = (idx / nx) % ny;
                let k = idx / (nx * ny);
                (i as f64 * 0.3).sin() + (j as f64 * 0.7).cos() + (k as f64 * 0.2)
            })
            .collect();

        // Create non-trivial vector field
        let hx: Vec<f64> = (0..n)
            .map(|idx| {
                let i = idx % nx;
                let j = (idx / nx) % ny;
                ((i as f64 + 1.0) * (j as f64 + 1.0)).sqrt()
            })
            .collect();
        let hy: Vec<f64> = (0..n)
            .map(|idx| {
                let k = idx / (nx * ny);
                (k as f64 * 0.5 + 0.1).sin()
            })
            .collect();
        let hz: Vec<f64> = (0..n)
            .map(|idx| {
                let i = idx % nx;
                let k = idx / (nx * ny);
                (i as f64 * 0.4 - k as f64 * 0.2).cos()
            })
            .collect();

        let (gx, gy, gz) = fgrad(&x, nx, ny, nz, vsx, vsy, vsz);
        let div_h = bdiv(&hx, &hy, &hz, nx, ny, nz, vsx, vsy, vsz);

        // <grad(x), h> = sum(gx*hx + gy*hy + gz*hz)
        let lhs: f64 = gx.iter().zip(hx.iter())
            .chain(gy.iter().zip(hy.iter()))
            .chain(gz.iter().zip(hz.iter()))
            .map(|(&a, &b)| a * b)
            .sum();

        // <x, -div(h)> = -sum(x * div_h)
        let rhs: f64 = x.iter().zip(div_h.iter())
            .map(|(&xi, &di)| -xi * di)
            .sum();

        let rel_err = (lhs - rhs).abs() / (lhs.abs().max(rhs.abs()) + 1e-15);
        assert!(
            rel_err < 1e-10,
            "Adjoint property failed for non-uniform voxels: lhs={}, rhs={}, rel_err={}",
            lhs, rhs, rel_err
        );
    }

    // ====================================================================
    // f32 gradient tests on larger arrays
    // ====================================================================

    #[test]
    fn test_fgrad_f32_allocating() {
        // Test the allocating version
        let nx = 8;
        let ny = 8;
        let nz = 8;
        let n = nx * ny * nz;

        let mut x = vec![0.0f32; n];
        // Linear ramp in x
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    x[i + j * nx + k * nx * ny] = i as f32;
                }
            }
        }

        let (gx, gy, gz) = fgrad_f32(&x, nx, ny, nz, 1.0, 1.0, 1.0);

        assert_eq!(gx.len(), n);
        assert_eq!(gy.len(), n);
        assert_eq!(gz.len(), n);

        // Interior x-gradient should be 1.0
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..(nx - 1) {
                    let idx = i + j * nx + k * nx * ny;
                    assert!(
                        (gx[idx] - 1.0).abs() < 1e-5,
                        "fgrad_f32 gx[{},{},{}] = {}, expected 1.0",
                        i, j, k, gx[idx]
                    );
                }
            }
        }
    }

    #[test]
    fn test_fgrad_inplace_f32_larger() {
        // Test with larger 8x8x8 volume (512 elements)
        let nx = 8;
        let ny = 8;
        let nz = 8;
        let n = nx * ny * nz;

        let mut x = vec![0.0f32; n];
        // Linear ramp in y
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    x[i + j * nx + k * nx * ny] = j as f32;
                }
            }
        }

        let mut gx = vec![0.0f32; n];
        let mut gy = vec![0.0f32; n];
        let mut gz = vec![0.0f32; n];
        fgrad_inplace_f32(&mut gx, &mut gy, &mut gz, &x, nx, ny, nz, 1.0, 1.0, 1.0);

        // Interior y-gradient should be 1.0
        for k in 0..nz {
            for j in 0..(ny - 1) {
                for i in 0..nx {
                    let idx = i + j * nx + k * nx * ny;
                    assert!(
                        (gy[idx] - 1.0).abs() < 1e-5,
                        "fgrad_inplace_f32 gy at interior should be 1.0, got {}",
                        gy[idx]
                    );
                }
            }
        }

        // gx should be 0 (no variation in x for y-ramp)
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..(nx - 1) {
                    let idx = i + j * nx + k * nx * ny;
                    assert!(
                        gx[idx].abs() < 1e-5,
                        "gx should be 0 for y-ramp, got {}",
                        gx[idx]
                    );
                }
            }
        }
    }

    #[test]
    fn test_fgrad_inplace_f32_z_ramp() {
        let nx = 8;
        let ny = 8;
        let nz = 8;
        let n = nx * ny * nz;

        let mut x = vec![0.0f32; n];
        // Linear ramp in z
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    x[i + j * nx + k * nx * ny] = k as f32;
                }
            }
        }

        let mut gx = vec![0.0f32; n];
        let mut gy = vec![0.0f32; n];
        let mut gz = vec![0.0f32; n];
        fgrad_inplace_f32(&mut gx, &mut gy, &mut gz, &x, nx, ny, nz, 1.0, 1.0, 1.0);

        // Interior z-gradient should be 1.0
        for k in 0..(nz - 1) {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = i + j * nx + k * nx * ny;
                    assert!(
                        (gz[idx] - 1.0).abs() < 1e-5,
                        "gz at interior should be 1.0, got {}",
                        gz[idx]
                    );
                }
            }
        }
    }

    #[test]
    fn test_bdiv_inplace_f32_larger() {
        // Test bdiv on 8x8x8 volume
        let nx = 8;
        let ny = 8;
        let nz = 8;
        let n = nx * ny * nz;

        // Constant gradient field
        let gx = vec![1.0f32; n];
        let gy = vec![1.0f32; n];
        let gz = vec![1.0f32; n];

        let mut div = vec![0.0f32; n];
        bdiv_inplace_f32(&mut div, &gx, &gy, &gz, nx, ny, nz, 1.0, 1.0, 1.0);

        // For constant gradient, backward differences give 0 at interior (i>0,j>0,k>0)
        for k in 1..nz {
            for j in 1..ny {
                for i in 1..nx {
                    let idx = i + j * nx + k * nx * ny;
                    assert!(
                        div[idx].abs() < 1e-5,
                        "bdiv interior should be 0 for constant gradient, got {}",
                        div[idx]
                    );
                }
            }
        }

        // At boundary (i=0 or j=0 or k=0), should be non-zero
        assert!(
            div[0].abs() > 0.1,
            "bdiv at origin should be non-zero for constant gradient"
        );
    }

    #[test]
    fn test_fgrad_masked_inplace_f32_larger() {
        let nx = 8;
        let ny = 8;
        let nz = 8;
        let n = nx * ny * nz;

        // Create a mask covering inner 6x6x6
        let mut mask = vec![0u8; n];
        for k in 1..7 {
            for j in 1..7 {
                for i in 1..7 {
                    mask[i + j * nx + k * nx * ny] = 1;
                }
            }
        }

        // Linear ramp in x
        let mut x = vec![0.0f32; n];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    x[i + j * nx + k * nx * ny] = i as f32;
                }
            }
        }

        let mut gx = vec![0.0f32; n];
        let mut gy = vec![0.0f32; n];
        let mut gz = vec![0.0f32; n];
        fgrad_masked_inplace_f32(
            &mut gx, &mut gy, &mut gz, &x, &mask,
            nx, ny, nz, 1.0, 1.0, 1.0,
        );

        // Outside mask should be zero
        for i in 0..n {
            if mask[i] == 0 {
                assert_eq!(gx[i], 0.0);
                assert_eq!(gy[i], 0.0);
                assert_eq!(gz[i], 0.0);
            }
        }

        // Inside mask, x-gradient should be ~1.0 (for interior)
        for k in 1..7 {
            for j in 1..7 {
                for i in 1..6 { // not at boundary of mask
                    let idx = i + j * nx + k * nx * ny;
                    if mask[idx] != 0 {
                        assert!(
                            (gx[idx] - 1.0).abs() < 1e-5,
                            "masked gx inside should be 1.0, got {} at ({},{},{})",
                            gx[idx], i, j, k
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_bdiv_masked_inplace_f32_larger() {
        let nx = 8;
        let ny = 8;
        let nz = 8;
        let n = nx * ny * nz;

        let mut mask = vec![0u8; n];
        for k in 1..7 {
            for j in 1..7 {
                for i in 1..7 {
                    mask[i + j * nx + k * nx * ny] = 1;
                }
            }
        }

        let gx = vec![1.0f32; n];
        let gy = vec![1.0f32; n];
        let gz = vec![1.0f32; n];

        let mut div = vec![0.0f32; n];
        bdiv_masked_inplace_f32(&mut div, &gx, &gy, &gz, &mask, nx, ny, nz, 1.0, 1.0, 1.0);

        // Outside mask should be zero
        for i in 0..n {
            if mask[i] == 0 {
                assert_eq!(div[i], 0.0, "Outside mask div should be zero");
            }
        }
    }

    #[test]
    fn test_fgrad_inplace_f64() {
        // Test the f64 in-place gradient
        let nx = 8;
        let ny = 8;
        let nz = 8;
        let n = nx * ny * nz;

        let mut x = vec![0.0f64; n];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    x[i + j * nx + k * nx * ny] = i as f64;
                }
            }
        }

        let mut gx = vec![0.0f64; n];
        let mut gy = vec![0.0f64; n];
        let mut gz = vec![0.0f64; n];
        fgrad_inplace(&mut gx, &mut gy, &mut gz, &x, nx, ny, nz, 1.0, 1.0, 1.0);

        // Interior gradient in x should be 1
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..(nx - 1) {
                    let idx = i + j * nx + k * nx * ny;
                    assert!(
                        (gx[idx] - 1.0).abs() < 1e-10,
                        "fgrad_inplace gx should be 1.0 at interior, got {}",
                        gx[idx]
                    );
                }
            }
        }
    }

    #[test]
    fn test_bdiv_inplace_f64() {
        let nx = 8;
        let ny = 8;
        let nz = 8;
        let n = nx * ny * nz;

        let gx = vec![1.0f64; n];
        let gy = vec![1.0f64; n];
        let gz = vec![1.0f64; n];

        let mut div = vec![0.0f64; n];
        bdiv_inplace(&mut div, &gx, &gy, &gz, nx, ny, nz, 1.0, 1.0, 1.0);

        // Interior divergence of constant field should be 0
        for k in 1..nz {
            for j in 1..ny {
                for i in 1..nx {
                    let idx = i + j * nx + k * nx * ny;
                    assert!(
                        div[idx].abs() < 1e-10,
                        "bdiv_inplace interior should be 0, got {}",
                        div[idx]
                    );
                }
            }
        }
    }

    #[test]
    fn test_grad_magnitude_squared() {
        let n = 64;
        let gx: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let gy: Vec<f64> = (0..n).map(|i| i as f64 * 0.2).collect();
        let gz: Vec<f64> = (0..n).map(|i| i as f64 * 0.3).collect();

        let mag_sq = grad_magnitude_squared(&gx, &gy, &gz);

        assert_eq!(mag_sq.len(), n);
        for i in 0..n {
            let expected = gx[i] * gx[i] + gy[i] * gy[i] + gz[i] * gz[i];
            assert!(
                (mag_sq[i] - expected).abs() < 1e-10,
                "grad_magnitude_squared[{}] = {}, expected {}",
                i, mag_sq[i], expected
            );
        }
    }

    #[test]
    fn test_fgrad_f32_nonunit_voxels() {
        let nx = 8;
        let ny = 8;
        let nz = 8;
        let n = nx * ny * nz;
        let vsx = 2.0f32;
        let vsy = 0.5f32;
        let vsz = 3.0f32;

        // Linear ramp in each direction simultaneously
        let mut x = vec![0.0f32; n];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    x[i + j * nx + k * nx * ny] = i as f32 + j as f32 + k as f32;
                }
            }
        }

        let mut gx = vec![0.0f32; n];
        let mut gy = vec![0.0f32; n];
        let mut gz = vec![0.0f32; n];
        fgrad_inplace_f32(&mut gx, &mut gy, &mut gz, &x, nx, ny, nz, vsx, vsy, vsz);

        // Interior x-gradient should be 1/vsx = 0.5
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..(nx - 1) {
                    let idx = i + j * nx + k * nx * ny;
                    assert!(
                        (gx[idx] - 1.0 / vsx).abs() < 1e-5,
                        "gx should be {}, got {}",
                        1.0 / vsx, gx[idx]
                    );
                }
            }
        }

        // Interior y-gradient should be 1/vsy = 2.0
        for k in 0..nz {
            for j in 0..(ny - 1) {
                for i in 0..nx {
                    let idx = i + j * nx + k * nx * ny;
                    assert!(
                        (gy[idx] - 1.0 / vsy).abs() < 1e-5,
                        "gy should be {}, got {}",
                        1.0 / vsy, gy[idx]
                    );
                }
            }
        }

        // Interior z-gradient should be 1/vsz
        for k in 0..(nz - 1) {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = i + j * nx + k * nx * ny;
                    assert!(
                        (gz[idx] - 1.0 / vsz).abs() < 1e-5,
                        "gz should be {}, got {}",
                        1.0 / vsz, gz[idx]
                    );
                }
            }
        }
    }

    #[test]
    fn test_symgrad_symdiv_basics_f32() {
        // Test basic properties of symgrad and symdiv
        let nx = 4;
        let ny = 4;
        let nz = 4;
        let n = nx * ny * nz;

        // Zero vector field => zero symgrad
        let wx = vec![0.0f32; n];
        let wy = vec![0.0f32; n];
        let wz = vec![0.0f32; n];

        let mut sxx = vec![0.0f32; n];
        let mut sxy = vec![0.0f32; n];
        let mut sxz = vec![0.0f32; n];
        let mut syy = vec![0.0f32; n];
        let mut syz = vec![0.0f32; n];
        let mut szz = vec![0.0f32; n];

        symgrad_inplace_f32(
            &mut sxx, &mut sxy, &mut sxz, &mut syy, &mut syz, &mut szz,
            &wx, &wy, &wz, nx, ny, nz, 1.0, 1.0, 1.0,
        );

        for i in 0..n {
            assert_eq!(sxx[i], 0.0);
            assert_eq!(sxy[i], 0.0);
            assert_eq!(sxz[i], 0.0);
            assert_eq!(syy[i], 0.0);
            assert_eq!(syz[i], 0.0);
            assert_eq!(szz[i], 0.0);
        }

        // Constant vector field => zero symgrad
        let wx = vec![5.0f32; n];
        let wy = vec![3.0f32; n];
        let wz = vec![7.0f32; n];

        symgrad_inplace_f32(
            &mut sxx, &mut sxy, &mut sxz, &mut syy, &mut syz, &mut szz,
            &wx, &wy, &wz, nx, ny, nz, 1.0, 1.0, 1.0,
        );

        // Interior points (not at boundary) should have zero symgrad
        for k in 0..(nz - 1) {
            for j in 0..(ny - 1) {
                for i in 0..(nx - 1) {
                    let idx = i + j * nx + k * nx * ny;
                    assert!(
                        sxx[idx].abs() < 1e-6,
                        "Constant field: sxx should be 0 at interior"
                    );
                    assert!(
                        sxy[idx].abs() < 1e-6,
                        "Constant field: sxy should be 0 at interior"
                    );
                }
            }
        }

        // Test symdiv of zero tensor => zero
        let qxx = vec![0.0f32; n];
        let qxy = vec![0.0f32; n];
        let qxz = vec![0.0f32; n];
        let qyy = vec![0.0f32; n];
        let qyz = vec![0.0f32; n];
        let qzz = vec![0.0f32; n];

        let mut divx = vec![0.0f32; n];
        let mut divy = vec![0.0f32; n];
        let mut divz = vec![0.0f32; n];
        symdiv_inplace_f32(
            &mut divx, &mut divy, &mut divz,
            &qxx, &qxy, &qxz, &qyy, &qyz, &qzz,
            nx, ny, nz, 1.0, 1.0, 1.0,
        );

        for i in 0..n {
            assert_eq!(divx[i], 0.0);
            assert_eq!(divy[i], 0.0);
            assert_eq!(divz[i], 0.0);
        }
    }

    #[test]
    fn test_symgrad_nonuniform_voxels_f32() {
        // Test symgrad with non-unit voxel sizes
        let nx = 6;
        let ny = 6;
        let nz = 6;
        let n = nx * ny * nz;
        let vsx = 2.0f32;
        let vsy = 0.5f32;
        let vsz = 1.5f32;

        // wx = x coordinate (linear in i)
        let mut wx = vec![0.0f32; n];
        let wy = vec![0.0f32; n];
        let wz = vec![0.0f32; n];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    wx[i + j * nx + k * nx * ny] = i as f32;
                }
            }
        }

        let mut sxx = vec![0.0f32; n];
        let mut sxy = vec![0.0f32; n];
        let mut sxz = vec![0.0f32; n];
        let mut syy = vec![0.0f32; n];
        let mut syz = vec![0.0f32; n];
        let mut szz = vec![0.0f32; n];

        symgrad_inplace_f32(
            &mut sxx, &mut sxy, &mut sxz, &mut syy, &mut syz, &mut szz,
            &wx, &wy, &wz, nx, ny, nz, vsx, vsy, vsz,
        );

        // sxx should be dwx/dx = 1/vsx at interior
        for k in 0..(nz - 1) {
            for j in 0..(ny - 1) {
                for i in 0..(nx - 1) {
                    let idx = i + j * nx + k * nx * ny;
                    assert!(
                        (sxx[idx] - 1.0 / vsx).abs() < 1e-5,
                        "sxx should be {}, got {}",
                        1.0 / vsx, sxx[idx]
                    );
                }
            }
        }
    }
}

//! Laplacian kernel for QSM processing
//!
//! Discrete 7-point stencil Laplacian kernel used for Laplacian-based
//! phase unwrapping and regularization.

/// Generate Laplacian kernel in image space
///
/// Creates a 7-point stencil Laplacian centered at (0,0,0) for FFT compatibility.
/// The stencil is: [-2(1/dx² + 1/dy² + 1/dz²)] at center,
/// [1/dx²] at ±1 in x, [1/dy²] at ±1 in y, [1/dz²] at ±1 in z.
///
/// # Arguments
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `negative` - If true, return negative Laplacian
///
/// # Returns
/// Flattened Laplacian kernel array of size nx*ny*nz in C order
pub fn laplacian_kernel(
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    negative: bool,
) -> Vec<f64> {
    let n_total = nx * ny * nz;
    let mut l = vec![0.0; n_total];

    // Stencil coefficients
    let hx = 1.0 / (vsx * vsx);
    let hy = 1.0 / (vsy * vsy);
    let hz = 1.0 / (vsz * vsz);
    let center = -2.0 * (hx + hy + hz);

    let sign = if negative { -1.0 } else { 1.0 };

    // Fortran order: idx(i, j, k) = i + j*nx + k*nx*ny

    // Center (0, 0, 0)
    l[0] = sign * center;

    // x neighbors: (1, 0, 0) and (nx-1, 0, 0)
    if nx > 1 {
        l[1] = sign * hx;  // (1, 0, 0)
        l[nx - 1] = sign * hx;  // (nx-1, 0, 0) wraps to -1
    }

    // y neighbors: (0, 1, 0) and (0, ny-1, 0)
    if ny > 1 {
        l[nx] = sign * hy;  // (0, 1, 0)
        l[(ny - 1) * nx] = sign * hy;  // (0, ny-1, 0)
    }

    // z neighbors: (0, 0, 1) and (0, 0, nz-1)
    if nz > 1 {
        l[nx * ny] = sign * hz;  // (0, 0, 1)
        l[(nz - 1) * nx * ny] = sign * hz;  // (0, 0, nz-1)
    }

    l
}

/// Generate negative Laplacian kernel (commonly used for gradient regularization)
pub fn negative_laplacian_kernel(
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> Vec<f64> {
    laplacian_kernel(nx, ny, nz, vsx, vsy, vsz, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_laplacian_kernel_sum() {
        // Laplacian kernel should sum to 0 (for periodic BC)
        let l = laplacian_kernel(8, 8, 8, 1.0, 1.0, 1.0, false);
        let sum: f64 = l.iter().sum();
        assert!(sum.abs() < 1e-10, "Laplacian should sum to 0, got {}", sum);
    }

    #[test]
    fn test_laplacian_kernel_center() {
        let l = laplacian_kernel(8, 8, 8, 1.0, 1.0, 1.0, false);
        // With voxel size 1, center should be -6
        assert!((l[0] - (-6.0)).abs() < 1e-10, "Center should be -6, got {}", l[0]);
    }

    #[test]
    fn test_negative_laplacian() {
        let l_pos = laplacian_kernel(8, 8, 8, 1.0, 1.0, 1.0, false);
        let l_neg = laplacian_kernel(8, 8, 8, 1.0, 1.0, 1.0, true);

        for (p, n) in l_pos.iter().zip(l_neg.iter()) {
            assert!((p + n).abs() < 1e-10, "Negative should be opposite sign");
        }
    }
}

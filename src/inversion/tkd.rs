//! Truncated k-space division (TKD) / TSVD for QSM
//!
//! TKD is the simplest dipole inversion method. It directly divides the
//! field in k-space by the dipole kernel, with truncation to avoid
//! division by small values near the magic angle.
//!
//! Reference:
//! Shmueli, K., de Zwart, J.A., van Gelderen, P., Li, T.Q., Dodd, S.J., Duyn, J.H. (2009).
//! "Magnetic susceptibility mapping of brain tissue in vivo using MRI phase data."
//! Magnetic Resonance in Medicine, 62(6):1510-1522. https://doi.org/10.1002/mrm.22135
//!
//! Reference implementation: https://github.com/kamesy/QSM.jl

use num_complex::Complex64;
use crate::fft::{fft3d, ifft3d};
use crate::kernels::dipole::dipole_kernel;

/// Truncated k-space division (TKD) for dipole inversion
///
/// Computes susceptibility map from local field using direct k-space division
/// with threshold-based truncation.
///
/// # Arguments
/// * `local_field` - Local field values (unwrapped phase / TE / gamma / B0)
/// * `mask` - Binary mask (1 = inside ROI, 0 = outside)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `bdir` - B0 field direction (bx, by, bz)
/// * `threshold` - Truncation threshold (typically 0.1-0.2)
///
/// # Returns
/// Susceptibility map (same size as input)
pub fn tkd(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bdir: (f64, f64, f64),
    threshold: f64,
) -> Vec<f64> {
    let n_total = nx * ny * nz;

    // Generate dipole kernel
    let d = dipole_kernel(nx, ny, nz, vsx, vsy, vsz, bdir);

    // Compute inverse dipole kernel with truncation
    // TKD: if |D| <= threshold, use sign(D)/threshold; else use 1/D
    let inv_threshold = 1.0 / threshold;
    let inv_d: Vec<f64> = d.iter().map(|&dval| {
        if dval.abs() <= threshold {
            // Truncate: use sign(D)/threshold
            if dval >= 0.0 { inv_threshold } else { -inv_threshold }
        } else {
            // Normal inverse
            1.0 / dval
        }
    }).collect();

    // Convert local field to complex
    let mut field_complex: Vec<Complex64> = local_field.iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();

    // FFT of local field
    fft3d(&mut field_complex, nx, ny, nz);

    // Multiply by inverse dipole kernel
    for i in 0..n_total {
        field_complex[i] *= inv_d[i];
    }

    // IFFT to get susceptibility
    ifft3d(&mut field_complex, nx, ny, nz);

    // Extract real part and apply mask
    let mut chi: Vec<f64> = field_complex.iter()
        .map(|c| c.re)
        .collect();

    // Apply mask
    for i in 0..n_total {
        if mask[i] == 0 {
            chi[i] = 0.0;
        }
    }

    chi
}

/// TKD with default B direction (0, 0, 1) and threshold 0.15
pub fn tkd_default(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> Vec<f64> {
    tkd(local_field, mask, nx, ny, nz, vsx, vsy, vsz, (0.0, 0.0, 1.0), 0.15)
}

/// Truncated singular value decomposition (TSVD) variant
///
/// Similar to TKD but zeros out values below threshold instead of truncating.
/// This produces smoother results but may have more artifacts at the magic angle.
pub fn tsvd(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bdir: (f64, f64, f64),
    threshold: f64,
) -> Vec<f64> {
    let n_total = nx * ny * nz;

    // Generate dipole kernel
    let d = dipole_kernel(nx, ny, nz, vsx, vsy, vsz, bdir);

    // Compute inverse dipole kernel with TSVD truncation
    // TSVD: if |D| <= threshold, use 0; else use 1/D
    let inv_d: Vec<f64> = d.iter().map(|&dval| {
        if dval.abs() <= threshold {
            0.0  // Zero out small values
        } else {
            1.0 / dval
        }
    }).collect();

    // Convert local field to complex
    let mut field_complex: Vec<Complex64> = local_field.iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();

    // FFT of local field
    fft3d(&mut field_complex, nx, ny, nz);

    // Multiply by inverse dipole kernel
    for i in 0..n_total {
        field_complex[i] *= inv_d[i];
    }

    // IFFT to get susceptibility
    ifft3d(&mut field_complex, nx, ny, nz);

    // Extract real part and apply mask
    let mut chi: Vec<f64> = field_complex.iter()
        .map(|c| c.re)
        .collect();

    // Apply mask
    for i in 0..n_total {
        if mask[i] == 0 {
            chi[i] = 0.0;
        }
    }

    chi
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tkd_zero_field() {
        // Zero field should give zero susceptibility
        let n = 8;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];

        let chi = tkd_default(&field, &mask, n, n, n, 1.0, 1.0, 1.0);

        for val in chi.iter() {
            assert!(val.abs() < 1e-10, "Zero field should give zero chi");
        }
    }

    #[test]
    fn test_tkd_mask() {
        // Values outside mask should be zero
        let n = 8;
        let field = vec![1.0; n * n * n];
        let mut mask = vec![1u8; n * n * n];

        // Set some voxels outside mask
        mask[0] = 0;
        mask[1] = 0;

        let chi = tkd_default(&field, &mask, n, n, n, 1.0, 1.0, 1.0);

        assert_eq!(chi[0], 0.0, "Outside mask should be 0");
        assert_eq!(chi[1], 0.0, "Outside mask should be 0");
    }

    #[test]
    fn test_tkd_finite() {
        // Result should be finite (no NaN or Inf)
        let n = 8;
        let field: Vec<f64> = (0..n*n*n).map(|i| (i as f64) * 0.01).collect();
        let mask = vec![1u8; n * n * n];

        let chi = tkd_default(&field, &mask, n, n, n, 1.0, 1.0, 1.0);

        for (i, val) in chi.iter().enumerate() {
            assert!(val.is_finite(), "Chi should be finite at index {}", i);
        }
    }

    #[test]
    fn test_tsvd_vs_tkd() {
        // TSVD and TKD should give different results near magic angle
        let n = 16;
        let field: Vec<f64> = (0..n*n*n).map(|i| ((i as f64) * 0.1).sin()).collect();
        let mask = vec![1u8; n * n * n];

        let chi_tkd = tkd(&field, &mask, n, n, n, 1.0, 1.0, 1.0, (0.0, 0.0, 1.0), 0.15);
        let chi_tsvd = tsvd(&field, &mask, n, n, n, 1.0, 1.0, 1.0, (0.0, 0.0, 1.0), 0.15);

        // They should be different
        let diff: f64 = chi_tkd.iter().zip(chi_tsvd.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(diff > 1e-10, "TKD and TSVD should give different results");
    }
}

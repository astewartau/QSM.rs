//! HARPERELLA and iHARPERELLA — integrated phase unwrapping and background field removal
//!
//! Both algorithms simultaneously unwrap phase and remove background field by
//! estimating the phase Laplacian outside the brain. The background phase is
//! harmonic inside the brain (∇²φ_bg = 0), so the wrapped Laplacian inside
//! the brain contains only tissue sources.
//!
//! **HARPERELLA** estimates the exterior Laplacian by making the SMV of the total
//! Laplacian uniform across the FOV (Eq [3] in the paper):
//!   min ||S(∇²φ_E) + S(∇²φ_brain) - δ||₂
//!
//! **iHARPERELLA** instead directly minimizes the weighted resulting phase
//! (Eq [3] in the ISMRM abstract):
//!   min ||W · inv_lap(∇²φ_brain + ∇²φ_out)||₂
//! providing more robust low-frequency background suppression.
//!
//! References:
//! - HARPERELLA: Li, W., et al. (2014). "Integrated Laplacian-based phase
//!   unwrapping and background phase removal for quantitative susceptibility
//!   mapping." NMR in Biomedicine, 27(2):219-227. doi:10.1002/nbm.3056
//! - iHARPERELLA: Li, W., Wu, B., Liu, C. (2015). "iHARPERELLA: an improved
//!   method for integrated 3D phase unwrapping and background phase removal."
//!   Proc. ISMRM 23, p.3313.

use num_complex::Complex64;
use crate::fft::{fft3d, ifft3d};
use crate::kernels::smv::smv_kernel;
use crate::unwrap::laplacian::wrapped_laplacian_periodic;

/// HARPERELLA / iHARPERELLA algorithm parameters
#[derive(Clone, Debug)]
pub struct HarperellaParams {
    /// SMV kernel radius in mm (paper uses 10mm for in vivo)
    pub radius: f64,
    /// Maximum CG iterations for exterior Laplacian estimation
    pub max_iter: usize,
    /// CG convergence tolerance
    pub tol: f64,
}

/// Type alias for backwards compatibility
pub type IharperellaParams = HarperellaParams;

impl Default for HarperellaParams {
    fn default() -> Self {
        Self {
            radius: 10.0,
            max_iter: 40,
            tol: 1e-6,
        }
    }
}

// =============================================================================
// HARPERELLA (Li et al., NMR Biomed 2014)
// Exterior estimation via SMV uniformity in Laplacian domain
// =============================================================================

/// HARPERELLA — integrated phase unwrapping and background removal
///
/// Estimates exterior Laplacian by enforcing uniform SMV across the FOV.
///
/// # Arguments
/// * `phase` - Wrapped phase in radians (nx * ny * nz)
/// * `mask` - Binary brain mask (nx * ny * nz), 1 = inside
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `radius` - SMV kernel radius in mm
/// * `max_iter` - Maximum CG iterations
///
/// # Returns
/// (tissue_phase, mask) — unwrapped background-free tissue phase and brain mask
pub fn harperella(
    phase: &[f64], mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    radius: f64, max_iter: usize,
) -> (Vec<f64>, Vec<u8>) {
    harperella_with_progress(phase, mask, nx, ny, nz, vsx, vsy, vsz,
                             radius, max_iter, 1e-6, |_, _| {})
}

/// HARPERELLA with default parameters
pub fn harperella_default(
    phase: &[f64], mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> (Vec<f64>, Vec<u8>) {
    let p = HarperellaParams::default();
    harperella_with_progress(phase, mask, nx, ny, nz, vsx, vsy, vsz,
                             p.radius, p.max_iter, p.tol, |_, _| {})
}

/// HARPERELLA with progress callback
pub fn harperella_with_progress<F>(
    phase: &[f64], mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    radius: f64, max_iter: usize, tol: f64,
    mut callback: F,
) -> (Vec<f64>, Vec<u8>)
where F: FnMut(usize, usize),
{
    let n_total = nx * ny * nz;
    let (lap_brain, interior_mask, exterior_mask) =
        prepare_laplacian(phase, mask, nx, ny, nz, vsx, vsy, vsz);

    // SMV kernel
    let s_fft = compute_smv_fft(nx, ny, nz, vsx, vsy, vsz, radius);

    // Compute δ = mean of S(∇²φ) over trustable interior I (Eq [4])
    let smv_lap = apply_smv(&lap_brain, &s_fft, nx, ny, nz);
    let mut delta_sum = 0.0;
    let mut delta_count = 0usize;
    for i in 0..n_total {
        if interior_mask[i] == 1 {
            delta_sum += smv_lap[i];
            delta_count += 1;
        }
    }
    let delta_val = if delta_count > 0 { delta_sum / delta_count as f64 } else { 0.0 };

    // Solve: min ||S(∇²φ_E) + S(∇²φ_brain) - δ||₂  (Eq [3])
    // Normal equations: ext * S(S(ext * x)) = ext * S(δ - S(lap_brain))
    let rhs: Vec<f64> = (0..n_total).map(|i| delta_val - smv_lap[i]).collect();
    let smv_rhs = apply_smv(&rhs, &s_fft, nx, ny, nz);
    let atb: Vec<f64> = (0..n_total).map(|i| exterior_mask[i] * smv_rhs[i]).collect();

    let lap_ext = cg_solve_masked(
        &atb, &exterior_mask, max_iter, tol, &mut callback,
        |x| {
            let masked: Vec<f64> = (0..n_total).map(|i| exterior_mask[i] * x[i]).collect();
            let sx = apply_smv(&masked, &s_fft, nx, ny, nz);
            let ssx = apply_smv(&sx, &s_fft, nx, ny, nz);
            (0..n_total).map(|i| exterior_mask[i] * ssx[i]).collect()
        },
    );

    // Combine and inverse Laplacian
    let mut lap_fov: Vec<f64> = lap_brain;
    for i in 0..n_total { lap_fov[i] += exterior_mask[i] * lap_ext[i]; }
    let tissue_phase = solve_poisson(&lap_fov, nx, ny, nz, vsx, vsy, vsz);

    let result: Vec<f64> = tissue_phase.iter().enumerate()
        .map(|(i, &v)| if mask[i] != 0 { v } else { 0.0 }).collect();
    (result, mask.to_vec())
}

// =============================================================================
// iHARPERELLA (Li et al., ISMRM 2015 #3313)
// Exterior estimation via direct phase minimization
// =============================================================================

/// iHARPERELLA — improved integrated phase unwrapping and background removal
///
/// More robust low-frequency background suppression than HARPERELLA.
/// Estimates exterior Laplacian by directly minimizing the weighted resulting
/// phase rather than enforcing SMV uniformity.
pub fn iharperella(
    phase: &[f64], mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    radius: f64, max_iter: usize,
) -> (Vec<f64>, Vec<u8>) {
    iharperella_with_progress(phase, mask, nx, ny, nz, vsx, vsy, vsz,
                              radius, max_iter, 1e-6, |_, _| {})
}

/// iHARPERELLA with default parameters
pub fn iharperella_default(
    phase: &[f64], mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> (Vec<f64>, Vec<u8>) {
    let p = HarperellaParams::default();
    iharperella_with_progress(phase, mask, nx, ny, nz, vsx, vsy, vsz,
                              p.radius, p.max_iter, p.tol, |_, _| {})
}

/// iHARPERELLA with progress callback
pub fn iharperella_with_progress<F>(
    phase: &[f64], mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    radius: f64, max_iter: usize, tol: f64,
    mut callback: F,
) -> (Vec<f64>, Vec<u8>)
where F: FnMut(usize, usize),
{
    let n_total = nx * ny * nz;
    let (lap_brain, interior_mask, exterior_mask) =
        prepare_laplacian(phase, mask, nx, ny, nz, vsx, vsy, vsz);

    // W_Brain = eroded interior mask (weighting for phase minimization)
    let w_brain: Vec<f64> = interior_mask.iter().map(|&m| m as f64).collect();

    // Pre-compute Laplacian eigenvalues for inverse Laplacian
    let lap_eig = compute_laplacian_eigenvalues(nx, ny, nz, vsx, vsy, vsz);

    // Solve: min ||W · inv_lap(∇²φ_brain + ∇²φ_out · (1-M))||₂  (Eq [3])
    //
    // A(x) = W · inv_lap(x · (1-M))
    // A'(y) = (1-M) · inv_lap(W · y)    [inv_lap is self-adjoint]
    // b = -W · inv_lap(∇²φ_brain)
    //
    // Normal equations: A'A(x) = A'(-b)
    // (1-M) · inv_lap(W² · inv_lap((1-M) · x)) = (1-M) · inv_lap(W² · inv_lap(∇²φ_brain))

    // Compute A'b = (1-M) · inv_lap(W² · inv_lap(∇²φ_brain))
    let phase_from_brain = apply_inv_lap(&lap_brain, &lap_eig, nx, ny, nz);
    let w2_phase: Vec<f64> = (0..n_total).map(|i| w_brain[i] * w_brain[i] * phase_from_brain[i]).collect();
    let atb_raw = apply_inv_lap(&w2_phase, &lap_eig, nx, ny, nz);
    let atb: Vec<f64> = (0..n_total).map(|i| exterior_mask[i] * atb_raw[i]).collect();

    let lap_ext = cg_solve_masked(
        &atb, &exterior_mask, max_iter, tol, &mut callback,
        |x| {
            // A'A(x) = (1-M) · inv_lap(W² · inv_lap((1-M) · x))
            let masked: Vec<f64> = (0..n_total).map(|i| exterior_mask[i] * x[i]).collect();
            let inv1 = apply_inv_lap(&masked, &lap_eig, nx, ny, nz);
            let w2_inv1: Vec<f64> = (0..n_total).map(|i| w_brain[i] * w_brain[i] * inv1[i]).collect();
            let inv2 = apply_inv_lap(&w2_inv1, &lap_eig, nx, ny, nz);
            (0..n_total).map(|i| exterior_mask[i] * inv2[i]).collect()
        },
    );

    // Combine and inverse Laplacian
    let mut lap_fov = lap_brain;
    for i in 0..n_total { lap_fov[i] += exterior_mask[i] * lap_ext[i]; }
    let tissue_phase = solve_poisson(&lap_fov, nx, ny, nz, vsx, vsy, vsz);

    let result: Vec<f64> = tissue_phase.iter().enumerate()
        .map(|(i, &v)| if mask[i] != 0 { v } else { 0.0 }).collect();
    (result, mask.to_vec())
}

// =============================================================================
// Shared infrastructure
// =============================================================================

/// Compute wrapped Laplacian, erode mask, prepare interior/exterior masks
fn prepare_laplacian(
    phase: &[f64], mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> (Vec<f64>, Vec<u8>, Vec<f64>) {
    let n_total = nx * ny * nz;

    // Wrapped Laplacian (Eq [1])
    let lap = wrapped_laplacian_periodic(phase, nx, ny, nz, vsx, vsy, vsz);

    // Erode mask by ~3 voxels
    let erode_radius = 3.0 * vsx.min(vsy).min(vsz);
    let erode_fft = compute_smv_fft(nx, ny, nz, vsx, vsy, vsz, erode_radius);

    let mask_f64: Vec<f64> = mask.iter().map(|&m| m as f64).collect();
    let mut mask_conv: Vec<Complex64> = mask_f64.iter()
        .map(|&x| Complex64::new(x, 0.0)).collect();
    fft3d(&mut mask_conv, nx, ny, nz);
    for i in 0..n_total { mask_conv[i] *= erode_fft[i]; }
    ifft3d(&mut mask_conv, nx, ny, nz);

    let delta_thresh = 1.0 - 1e-7_f64.sqrt();
    let interior_mask: Vec<u8> = mask_conv.iter()
        .map(|c| if c.re > delta_thresh { 1 } else { 0 }).collect();
    let exterior_mask: Vec<f64> = mask.iter()
        .map(|&m| if m == 0 { 1.0 } else { 0.0 }).collect();

    // Keep Laplacian only at trustable interior voxels
    let mut lap_brain = vec![0.0; n_total];
    for i in 0..n_total {
        if interior_mask[i] == 1 { lap_brain[i] = lap[i]; }
    }

    (lap_brain, interior_mask, exterior_mask)
}

/// Compute FFT of SMV kernel (returns real part, since kernel is symmetric)
fn compute_smv_fft(
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64, radius: f64,
) -> Vec<f64> {
    let s_kernel = smv_kernel(nx, ny, nz, vsx, vsy, vsz, radius);
    let mut c: Vec<Complex64> = s_kernel.iter()
        .map(|&x| Complex64::new(x, 0.0)).collect();
    fft3d(&mut c, nx, ny, nz);
    c.iter().map(|v| v.re).collect()
}

/// Apply SMV filter in k-space: S(x) = ifft(s_fft * fft(x))
fn apply_smv(x: &[f64], s_fft: &[f64], nx: usize, ny: usize, nz: usize) -> Vec<f64> {
    let n_total = nx * ny * nz;
    let mut c: Vec<Complex64> = x.iter()
        .map(|&v| Complex64::new(v, 0.0)).collect();
    fft3d(&mut c, nx, ny, nz);
    for i in 0..n_total { c[i] *= s_fft[i]; }
    ifft3d(&mut c, nx, ny, nz);
    c.iter().map(|v| v.re).collect()
}

/// Compute discrete Laplacian eigenvalues for inverse Laplacian
fn compute_laplacian_eigenvalues(
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> Vec<f64> {
    use std::f64::consts::PI;
    let n_total = nx * ny * nz;
    let idx2 = 1.0 / (vsx * vsx);
    let idy2 = 1.0 / (vsy * vsy);
    let idz2 = 1.0 / (vsz * vsz);
    let mut eig = vec![0.0; n_total];

    for k in 0..nz {
        let fk = if k <= nz / 2 { k as f64 / nz as f64 } else { (k as f64 - nz as f64) / nz as f64 };
        let lz = 2.0 * ((2.0 * PI * fk).cos() - 1.0) * idz2;
        for j in 0..ny {
            let fj = if j <= ny / 2 { j as f64 / ny as f64 } else { (j as f64 - ny as f64) / ny as f64 };
            let ly = 2.0 * ((2.0 * PI * fj).cos() - 1.0) * idy2;
            for i in 0..nx {
                let fi = if i <= nx / 2 { i as f64 / nx as f64 } else { (i as f64 - nx as f64) / nx as f64 };
                let lx = 2.0 * ((2.0 * PI * fi).cos() - 1.0) * idx2;
                eig[i + j * nx + k * nx * ny] = lx + ly + lz;
            }
        }
    }
    eig
}

/// Apply inverse Laplacian using pre-computed eigenvalues
fn apply_inv_lap(f: &[f64], eig: &[f64], nx: usize, ny: usize, nz: usize) -> Vec<f64> {
    let n_total = nx * ny * nz;
    let mut c: Vec<Complex64> = f.iter()
        .map(|&x| Complex64::new(x, 0.0)).collect();
    fft3d(&mut c, nx, ny, nz);
    for i in 0..n_total {
        if eig[i].abs() > 1e-20 {
            c[i] /= eig[i];
        } else {
            c[i] = Complex64::new(0.0, 0.0);
        }
    }
    ifft3d(&mut c, nx, ny, nz);
    c.iter().map(|v| v.re).collect()
}

/// Solve Poisson equation via FFT: ∇²u = f → u
fn solve_poisson(
    f: &[f64], nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> Vec<f64> {
    let eig = compute_laplacian_eigenvalues(nx, ny, nz, vsx, vsy, vsz);
    apply_inv_lap(f, &eig, nx, ny, nz)
}

/// Generic CG solver for masked normal equations
fn cg_solve_masked<F, Op>(
    b: &[f64],
    _mask: &[f64],
    max_iter: usize,
    tol: f64,
    callback: &mut F,
    apply_ata: Op,
) -> Vec<f64>
where
    F: FnMut(usize, usize),
    Op: Fn(&[f64]) -> Vec<f64>,
{
    let n = b.len();
    let mut x = vec![0.0; n];
    let mut r = b.to_vec();
    let mut p = r.clone();
    let mut rsold: f64 = r.iter().map(|&v| v * v).sum();
    let b_norm: f64 = b.iter().map(|&v| v * v).sum::<f64>().sqrt();

    if b_norm < 1e-20 { return x; }

    for iter in 0..max_iter {
        callback(iter + 1, max_iter);

        let ap = apply_ata(&p);
        let pap: f64 = p.iter().zip(ap.iter()).map(|(&pi, &api)| pi * api).sum();

        if pap.abs() < 1e-20 { break; }

        let alpha = rsold / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        let rsnew: f64 = r.iter().map(|&v| v * v).sum();
        if rsnew.sqrt() < tol * b_norm { break; }

        let beta = rsnew / rsold;
        for i in 0..n { p[i] = r[i] + beta * p[i]; }
        rsold = rsnew;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unwrap::laplacian::wrap;

    #[test]
    fn test_harperella_zero_phase() {
        let n = 16;
        let phase = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];
        let (tissue, _) = harperella(&phase, &mask, n, n, n, 1.0, 1.0, 1.0, 2.0, 10);
        for &val in tissue.iter() {
            assert!(val.abs() < 1e-8, "Zero phase should give zero tissue phase, got {}", val);
        }
    }

    #[test]
    fn test_harperella_finite() {
        let n = 16;
        let phase: Vec<f64> = (0..n*n*n).map(|i| wrap((i as f64) * 0.1)).collect();
        let mask = vec![1u8; n * n * n];
        let (tissue, _) = harperella(&phase, &mask, n, n, n, 1.0, 1.0, 1.0, 2.0, 10);
        for (i, &val) in tissue.iter().enumerate() {
            assert!(val.is_finite(), "Tissue phase should be finite at index {}", i);
        }
    }

    #[test]
    fn test_iharperella_zero_phase() {
        let n = 16;
        let phase = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];
        let (tissue, _) = iharperella(&phase, &mask, n, n, n, 1.0, 1.0, 1.0, 2.0, 10);
        for &val in tissue.iter() {
            assert!(val.abs() < 1e-8, "Zero phase should give zero tissue phase, got {}", val);
        }
    }

    #[test]
    fn test_iharperella_finite() {
        let n = 16;
        let phase: Vec<f64> = (0..n*n*n).map(|i| wrap((i as f64) * 0.1)).collect();
        let mask = vec![1u8; n * n * n];
        let (tissue, _) = iharperella(&phase, &mask, n, n, n, 1.0, 1.0, 1.0, 2.0, 10);
        for (i, &val) in tissue.iter().enumerate() {
            assert!(val.is_finite(), "Tissue phase should be finite at index {}", i);
        }
    }
}

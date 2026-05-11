//! HARPERELLA / iHARPERELLA — integrated phase unwrapping and background field removal
//!
//! Simultaneously unwraps phase and removes background field in a single step
//! by estimating the phase Laplacian outside the brain. The background phase
//! is harmonic inside the brain (∇²φ_bg = 0), so the wrapped Laplacian inside
//! the brain contains only tissue sources. By estimating a consistent exterior
//! Laplacian via LSQR, the inverse Laplacian yields background-free tissue phase.
//!
//! Algorithm:
//! 1. Compute wrapped Laplacian inside brain: ∇²φ = cosφ·∇²sinφ - sinφ·∇²cosφ
//! 2. Zero the Laplacian at boundary voxels (unreliable near mask edge)
//! 3. Estimate exterior Laplacian ∇²φ_E by minimizing:
//!    ||S(∇²φ_E) + S(∇²φ_brain) - δ||₂  (via CG on normal equations)
//!    ensuring the SMV of the total Laplacian is uniform (no boundary sources)
//! 4. Inverse Laplacian of (interior + exterior) gives tissue phase
//!
//! Reference:
//! Li, W., Avram, A.V., Wu, B., Xiao, X., Liu, C. (2014).
//! "Integrated Laplacian-based phase unwrapping and background phase removal
//! for quantitative susceptibility mapping."
//! NMR in Biomedicine, 27(2):219-227. https://doi.org/10.1002/nbm.3056

use num_complex::Complex64;
use crate::fft::{fft3d, ifft3d};
use crate::kernels::smv::smv_kernel;
use crate::unwrap::laplacian::wrapped_laplacian_periodic;

/// iHARPERELLA algorithm parameters
#[derive(Clone, Debug)]
pub struct IharperellaParams {
    /// SMV kernel radius in mm (paper uses 10mm for in vivo)
    pub radius: f64,
    /// Maximum CG/LSQR iterations for exterior Laplacian estimation
    pub max_iter: usize,
    /// CG convergence tolerance
    pub tol: f64,
}

impl Default for IharperellaParams {
    fn default() -> Self {
        Self {
            radius: 10.0,
            max_iter: 40,
            tol: 1e-6,
        }
    }
}

/// iHARPERELLA — integrated phase unwrapping and background removal
///
/// # Arguments
/// * `phase` - Wrapped phase in radians (nx * ny * nz)
/// * `mask` - Binary brain mask (nx * ny * nz), 1 = inside
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `radius` - SMV kernel radius in mm
/// * `max_iter` - Maximum CG iterations for exterior estimation
///
/// # Returns
/// (tissue_phase, mask) — unwrapped background-free tissue phase and brain mask
pub fn iharperella(
    phase: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    radius: f64,
    max_iter: usize,
) -> (Vec<f64>, Vec<u8>) {
    iharperella_with_progress(phase, mask, nx, ny, nz, vsx, vsy, vsz,
                              radius, max_iter, 1e-6, |_, _| {})
}

/// iHARPERELLA with default parameters
pub fn iharperella_default(
    phase: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> (Vec<f64>, Vec<u8>) {
    let p = IharperellaParams::default();
    iharperella_with_progress(phase, mask, nx, ny, nz, vsx, vsy, vsz,
                              p.radius, p.max_iter, p.tol, |_, _| {})
}

/// iHARPERELLA with progress callback
///
/// Callback receives (current_iteration, max_iterations) during CG solve.
pub fn iharperella_with_progress<F>(
    phase: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    radius: f64,
    max_iter: usize,
    tol: f64,
    mut callback: F,
) -> (Vec<f64>, Vec<u8>)
where
    F: FnMut(usize, usize),
{
    let n_total = nx * ny * nz;

    // =========================================================================
    // Step 1: Compute wrapped Laplacian inside brain (Eq [1])
    // ∇²φ = cosφ·∇²sinφ - sinφ·∇²cosφ
    // Our wrapped_laplacian_periodic is equivalent (uses wrapped finite diffs)
    // =========================================================================
    let lap = wrapped_laplacian_periodic(phase, nx, ny, nz, vsx, vsy, vsz);

    // =========================================================================
    // Step 2: Build interior mask (erode by ~3 voxels) and exterior mask
    // The paper zeros "large Laplacian values near the boundary of the brain
    // (less than three voxels from the boundary)" as these are inaccurate.
    // =========================================================================

    // Erode mask by ~3 voxels using SMV convolution with small radius
    let erode_radius = 3.0 * vsx.min(vsy).min(vsz);
    let erode_kernel = smv_kernel(nx, ny, nz, vsx, vsy, vsz, erode_radius);
    let mut erode_complex: Vec<Complex64> = erode_kernel.iter()
        .map(|&x| Complex64::new(x, 0.0)).collect();
    fft3d(&mut erode_complex, nx, ny, nz);
    let erode_fft: Vec<f64> = erode_complex.iter().map(|c| c.re).collect();

    let mask_f64: Vec<f64> = mask.iter().map(|&m| m as f64).collect();
    let mut mask_conv: Vec<Complex64> = mask_f64.iter()
        .map(|&x| Complex64::new(x, 0.0)).collect();
    fft3d(&mut mask_conv, nx, ny, nz);
    for i in 0..n_total {
        mask_conv[i] *= erode_fft[i];
    }
    ifft3d(&mut mask_conv, nx, ny, nz);

    let delta_thresh = 1.0 - 1e-7_f64.sqrt();
    // Interior mask I: trustable region (eroded brain)
    let interior_mask: Vec<u8> = mask_conv.iter()
        .map(|c| if c.re > delta_thresh { 1 } else { 0 })
        .collect();
    // Exterior mask E: everything outside the brain mask
    // (includes true exterior and boundary region O)
    let exterior_mask: Vec<f64> = mask.iter()
        .map(|&m| if m == 0 { 1.0 } else { 0.0 })
        .collect();

    // Laplacian inside brain (I+O), zeroed at boundary and outside
    // Zero boundary voxels (in O but not in I) for reliability
    let mut lap_brain = vec![0.0; n_total];
    for i in 0..n_total {
        if interior_mask[i] == 1 {
            lap_brain[i] = lap[i];
        }
    }

    // =========================================================================
    // Step 3: Pre-compute SMV kernel for the exterior estimation
    // =========================================================================
    let s_kernel = smv_kernel(nx, ny, nz, vsx, vsy, vsz, radius);
    let mut s_complex: Vec<Complex64> = s_kernel.iter()
        .map(|&x| Complex64::new(x, 0.0)).collect();
    fft3d(&mut s_complex, nx, ny, nz);
    let s_fft: Vec<f64> = s_complex.iter().map(|c| c.re).collect();

    // =========================================================================
    // Step 4: Compute δ = mean of S(∇²φ) over trustable interior I (Eq [4])
    // =========================================================================
    let smv_lap_brain = apply_smv(&lap_brain, &s_fft, nx, ny, nz);

    let mut delta_sum = 0.0;
    let mut delta_count = 0usize;
    for i in 0..n_total {
        if interior_mask[i] == 1 {
            delta_sum += smv_lap_brain[i];
            delta_count += 1;
        }
    }
    let delta_val = if delta_count > 0 { delta_sum / delta_count as f64 } else { 0.0 };

    // =========================================================================
    // Step 5: Solve for exterior Laplacian (Eq [3])
    // min_{∇²φ_E} ||S(∇²φ_E) + S(∇²φ_brain) - δ||₂
    //
    // Let A(x) = S(x * exterior_mask)  [SMV of exterior values]
    // b = δ - S(∇²φ_brain)            [target for A(x)]
    //
    // Normal equations: A'A(x) = A'b
    // A'(y) = exterior_mask * S(y)     [SMV is self-adjoint for real symmetric kernel]
    // A'A(x) = exterior_mask * S(S(exterior_mask * x))
    // =========================================================================

    // RHS: b = δ - S(lap_brain)
    let rhs: Vec<f64> = (0..n_total)
        .map(|i| delta_val - smv_lap_brain[i])
        .collect();

    // A'b = exterior_mask * S(rhs)
    let smv_rhs = apply_smv(&rhs, &s_fft, nx, ny, nz);
    let atb: Vec<f64> = (0..n_total)
        .map(|i| exterior_mask[i] * smv_rhs[i])
        .collect();

    // Solve A'A(x) = A'b with CG
    let lap_ext = cg_exterior(
        &atb, &exterior_mask, &s_fft,
        nx, ny, nz, tol, max_iter, &mut callback,
    );

    // =========================================================================
    // Step 6: Combine and inverse Laplacian (Eq [5], [6])
    // ∇²φ_FOV = ∇²φ_brain + ∇²φ_E
    // φ = inverse_laplacian(∇²φ_FOV)
    // =========================================================================
    let mut lap_fov = lap_brain;
    for i in 0..n_total {
        lap_fov[i] += exterior_mask[i] * lap_ext[i];
    }

    // Inverse Laplacian via FFT (same as solve_poisson_fft)
    let tissue_phase = solve_poisson(&lap_fov, nx, ny, nz, vsx, vsy, vsz);

    // Mask to brain
    let result: Vec<f64> = tissue_phase.iter()
        .enumerate()
        .map(|(i, &v)| if mask[i] != 0 { v } else { 0.0 })
        .collect();

    (result, mask.to_vec())
}

/// Apply SMV filter in k-space: S(x) = ifft(s_fft * fft(x))
fn apply_smv(x: &[f64], s_fft: &[f64], nx: usize, ny: usize, nz: usize) -> Vec<f64> {
    let n_total = nx * ny * nz;
    let mut c: Vec<Complex64> = x.iter()
        .map(|&v| Complex64::new(v, 0.0)).collect();
    fft3d(&mut c, nx, ny, nz);
    for i in 0..n_total {
        c[i] *= s_fft[i];
    }
    ifft3d(&mut c, nx, ny, nz);
    c.iter().map(|v| v.re).collect()
}

/// CG solver for the exterior Laplacian normal equations
/// Solves: exterior_mask * S(S(exterior_mask * x)) = rhs
fn cg_exterior<F>(
    b: &[f64],
    ext_mask: &[f64],
    s_fft: &[f64],
    nx: usize, ny: usize, nz: usize,
    tol: f64,
    max_iter: usize,
    callback: &mut F,
) -> Vec<f64>
where
    F: FnMut(usize, usize),
{
    let n = b.len();

    // A'A operator: x -> ext_mask * S(S(ext_mask * x))
    let apply_ata = |x: &[f64]| -> Vec<f64> {
        // Mask to exterior
        let masked: Vec<f64> = (0..n).map(|i| ext_mask[i] * x[i]).collect();
        // Apply S twice
        let sx = apply_smv(&masked, s_fft, nx, ny, nz);
        let ssx = apply_smv(&sx, s_fft, nx, ny, nz);
        // Mask to exterior again
        (0..n).map(|i| ext_mask[i] * ssx[i]).collect()
    };

    // CG iteration
    let mut x = vec![0.0; n];
    let mut r = b.to_vec();
    let mut p = r.clone();
    let mut rsold: f64 = r.iter().map(|&v| v * v).sum();
    let b_norm: f64 = b.iter().map(|&v| v * v).sum::<f64>().sqrt();

    if b_norm < 1e-20 {
        return x;
    }

    for iter in 0..max_iter {
        callback(iter + 1, max_iter);

        let ap = apply_ata(&p);
        let pap: f64 = p.iter().zip(ap.iter()).map(|(&pi, &api)| pi * api).sum();

        if pap.abs() < 1e-20 {
            break;
        }

        let alpha = rsold / pap;

        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        let rsnew: f64 = r.iter().map(|&v| v * v).sum();

        if rsnew.sqrt() < tol * b_norm {
            break;
        }

        let beta = rsnew / rsold;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }

        rsold = rsnew;
    }

    x
}

/// Solve Poisson equation via FFT: ∇²u = f → u
fn solve_poisson(
    f: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> Vec<f64> {
    use std::f64::consts::PI;
    let n_total = nx * ny * nz;

    let mut c: Vec<Complex64> = f.iter()
        .map(|&x| Complex64::new(x, 0.0)).collect();
    fft3d(&mut c, nx, ny, nz);

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
                    c[idx] /= lam;
                } else {
                    c[idx] = Complex64::new(0.0, 0.0);
                }
            }
        }
    }

    ifft3d(&mut c, nx, ny, nz);
    c.iter().map(|v| v.re).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unwrap::laplacian::wrap;

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

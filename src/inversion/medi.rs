//! MEDI (Morphology Enabled Dipole Inversion) L1 regularization
//!
//! Gauss-Newton optimization with L1 TV regularization and
//! morphology-based gradient weighting from magnitude images.
//!
//! Features:
//! - **Per-direction gradient masks** (mx, my, mz) matching the original MEDI formulation
//! - Adaptive edge detection with configurable percentage threshold (default: 30% edges)
//! - SNR-based data weighting using noise standard deviation maps
//! - Optional SMV (Spherical Mean Value) preprocessing
//! - Optional merit-based outlier adjustment (MERIT)
//! - **Optimized with f32 single precision for WASM performance**
//! - **Buffer reuse to minimize allocations**
//! - **Standard CG convergence (relative tolerance)**
//! - **Linear extrapolation boundary conditions** matching MATLAB's gradf
//!
//! Reference:
//! Liu T, Liu J, de Rochefort L, Spincemaille P, Khalidov I, Ledoux JR,
//! Wang Y. Morphology enabled dipole inversion (MEDI) from a single-angle
//! acquisition: comparison with COSMOS in human brain imaging.
//! Magnetic resonance in medicine. 2011 Aug;66(3):777-83.
//!
//! Liu J, Liu T, de Rochefort L, Ledoux J, Khalidov I, Chen W, Tsiouris AJ,
//! Wisnieff C, Spincemaille P, Prince MR, Wang Y. Morphology enabled dipole
//! inversion for quantitative susceptibility mapping using structural
//! consistency between the magnitude image and the susceptibility map.
//! Neuroimage. 2012 Feb 1;59(3):2560-8.

use num_complex::Complex32;
use crate::fft::Fft3dWorkspaceF32;
use crate::kernels::dipole::dipole_kernel_f32;
use crate::kernels::smv::smv_kernel_f32;
use crate::utils::simd_ops::{
    dot_product_f32, norm_squared_f32, axpy_f32, xpby_f32,
    apply_gradient_weights_f32, compute_p_weights_f32, combine_terms_f32, negate_f32,
};
// Note: Uses fgrad_periodic_inplace_f32 / bdiv_periodic_inplace_f32 (periodic BCs)
// for the MEDI inner loop matching MATLAB's gradfp_mex / gradfp_adj_mex.
// Uses fgrad_linext_inplace_f32 (linear extrapolation BCs) only for gradient mask
// computation, matching MATLAB's gradf_mex.

/// Workspace for MEDI operations - holds all reusable buffers (f32 version)
/// Uses single precision for ~2x speedup on WASM
pub struct MediWorkspace {
    pub n_total: usize,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub vsx: f32,
    pub vsy: f32,
    pub vsz: f32,

    // FFT workspace with cached plans (f32)
    pub fft_ws: Fft3dWorkspaceF32,

    // Gradient buffers (3 components)
    pub gx: Vec<f32>,
    pub gy: Vec<f32>,
    pub gz: Vec<f32>,

    // Weighted gradient buffers
    pub reg_x: Vec<f32>,
    pub reg_y: Vec<f32>,
    pub reg_z: Vec<f32>,

    // Divergence buffer
    pub div_buf: Vec<f32>,

    // Complex buffer for FFT operations
    pub complex_buf: Vec<Complex32>,
    pub complex_buf2: Vec<Complex32>,

    // Real buffer for dipole result
    pub dipole_buf: Vec<f32>,

    // CG solver buffers
    pub cg_r: Vec<f32>,
    pub cg_p: Vec<f32>,
    pub cg_ap: Vec<f32>,
}

impl MediWorkspace {
    /// Create a new MEDI workspace for the given dimensions
    pub fn new(nx: usize, ny: usize, nz: usize, vsx: f32, vsy: f32, vsz: f32) -> Self {
        let n_total = nx * ny * nz;

        Self {
            n_total,
            nx, ny, nz,
            vsx, vsy, vsz,
            fft_ws: Fft3dWorkspaceF32::new(nx, ny, nz),
            gx: vec![0.0; n_total],
            gy: vec![0.0; n_total],
            gz: vec![0.0; n_total],
            reg_x: vec![0.0; n_total],
            reg_y: vec![0.0; n_total],
            reg_z: vec![0.0; n_total],
            div_buf: vec![0.0; n_total],
            complex_buf: vec![Complex32::new(0.0, 0.0); n_total],
            complex_buf2: vec![Complex32::new(0.0, 0.0); n_total],
            dipole_buf: vec![0.0; n_total],
            cg_r: vec![0.0; n_total],
            cg_p: vec![0.0; n_total],
            cg_ap: vec![0.0; n_total],
        }
    }
}

/// Apply dipole convolution: out = real(ifft(D * fft(x)))
#[inline]
fn apply_dipole_conv(
    fft_ws: &mut Fft3dWorkspaceF32,
    x: &[f32],
    d_kernel: &[f32],
    out: &mut [f32],
    complex_buf: &mut [Complex32],
) {
    fft_ws.apply_dipole_inplace(x, d_kernel, out, complex_buf);
}

/// MEDI operator buffers - separate struct to allow split borrowing
struct MediOpBuffers<'a> {
    gx: &'a mut [f32],
    gy: &'a mut [f32],
    gz: &'a mut [f32],
    reg_x: &'a mut [f32],
    reg_y: &'a mut [f32],
    reg_z: &'a mut [f32],
    div_buf: &'a mut [f32],
    dipole_buf: &'a mut [f32],
    complex_buf: &'a mut [Complex32],
    complex_buf2: &'a mut [Complex32],
}

/// Apply MEDI operator in-place: out = fidelity(dx) + lambda*reg(dx)
/// This is the hot path - called many times per Gauss-Newton iteration
/// Uses per-direction gradient masks (mx, my, mz) matching MATLAB MEDI
/// SIMD-accelerated for element-wise operations
#[inline]
fn apply_medi_operator_core(
    fft_ws: &mut Fft3dWorkspaceF32,
    bufs: &mut MediOpBuffers,
    n: usize,
    nx: usize, ny: usize, nz: usize,
    vsx: f32, vsy: f32, vsz: f32,
    dx: &[f32],
    w: &[Complex32],
    d_kernel: &[f32],
    mx: &[f32],  // Per-direction gradient mask for x
    my: &[f32],  // Per-direction gradient mask for y
    mz: &[f32],  // Per-direction gradient mask for z
    vr: &[f32],
    lambda: f32,
    out: &mut [f32],
) {
    // 1. Compute gradient of dx (in-place into gx, gy, gz) - periodic BCs matching MATLAB gradfp_mex
    fgrad_periodic_inplace_f32(bufs.gx, bufs.gy, bufs.gz, dx, nx, ny, nz, vsx, vsy, vsz);

    // 2. Apply per-direction weights: reg_i = m_i * P * m_i * g_i (SIMD accelerated)
    // MATLAB: ux = mx .* P .* mx .* ux; uy = my .* P .* my .* uy; uz = mz .* P .* mz .* uz;
    apply_gradient_weights_f32(
        bufs.reg_x, bufs.reg_y, bufs.reg_z,
        mx, my, mz, vr,
        bufs.gx, bufs.gy, bufs.gz,
    );

    // 3. Compute divergence (in-place into div_buf) - periodic BCs matching MATLAB gradfp_adj_mex
    bdiv_periodic_inplace_f32(bufs.div_buf, bufs.reg_x, bufs.reg_y, bufs.reg_z, nx, ny, nz, vsx, vsy, vsz);

    // 4. Fidelity term: D^T(|w|^2 * D(dx))
    apply_dipole_conv(fft_ws, dx, d_kernel, bufs.dipole_buf, bufs.complex_buf);

    // Multiply by |w|^2 and convert to complex
    for i in 0..n {
        let w_mag_sq = w[i].norm_sqr();
        bufs.complex_buf2[i] = Complex32::new(bufs.dipole_buf[i] * w_mag_sq, 0.0);
    }

    // Apply D^T (which is D for real symmetric kernel)
    fft_ws.fft3d(bufs.complex_buf2);
    for i in 0..n {
        bufs.complex_buf2[i] *= d_kernel[i];
    }
    fft_ws.ifft3d(bufs.complex_buf2);

    // 5. Combine: out = lambda*div_buf + real(complex_buf2) (matching MATLAB: y = D + R)
    // Extract real parts for SIMD operation
    for i in 0..n {
        bufs.dipole_buf[i] = bufs.complex_buf2[i].re;
    }
    combine_terms_f32(out, bufs.div_buf, bufs.dipole_buf, lambda);
}

/// Conjugate gradient solver with buffer reuse
/// Solves Ax = b where A is the MEDI operator
///
/// The optional progress callback receives (cg_iter, max_iter) for each CG iteration.
/// Uses per-direction gradient masks (mx, my, mz) matching MATLAB MEDI.
#[inline]
fn cg_solve_medi<F>(
    ws: &mut MediWorkspace,
    w: &[Complex32],
    d_kernel: &[f32],
    mx: &[f32],  // Per-direction gradient mask for x
    my: &[f32],  // Per-direction gradient mask for y
    mz: &[f32],  // Per-direction gradient mask for z
    vr: &[f32],
    lambda: f32,
    b: &[f32],
    x: &mut [f32],
    tol: f32,
    max_iter: usize,
    mut progress_callback: F,
) where
    F: FnMut(usize, usize),
{
    let n = ws.n_total;
    let (nx, ny, nz) = (ws.nx, ws.ny, ws.nz);
    let (vsx, vsy, vsz) = (ws.vsx, ws.vsy, ws.vsz);

    // Initialize x to zero
    x.fill(0.0);

    // r = b - A*x = b (since x=0)
    ws.cg_r.copy_from_slice(b);

    // p = r
    ws.cg_p.copy_from_slice(&ws.cg_r);

    // rsold = r·r (SIMD accelerated)
    let mut rsold: f32 = norm_squared_f32(&ws.cg_r);

    // b_norm for relative tolerance (SIMD accelerated)
    let b_norm: f32 = norm_squared_f32(b).sqrt();
    if b_norm < 1e-10 {
        return; // b is zero, x=0 is the solution
    }

    // Buffer for p (to avoid borrow conflict)
    let mut p_copy = vec![0.0f32; n];

    for cg_iter in 0..max_iter {
        // Report CG progress
        progress_callback(cg_iter + 1, max_iter);

        // Copy p to avoid borrow conflict
        p_copy.copy_from_slice(&ws.cg_p);

        // ap = A*p - use split borrowing
        {
            let mut bufs = MediOpBuffers {
                gx: &mut ws.gx,
                gy: &mut ws.gy,
                gz: &mut ws.gz,
                reg_x: &mut ws.reg_x,
                reg_y: &mut ws.reg_y,
                reg_z: &mut ws.reg_z,
                div_buf: &mut ws.div_buf,
                dipole_buf: &mut ws.dipole_buf,
                complex_buf: &mut ws.complex_buf,
                complex_buf2: &mut ws.complex_buf2,
            };
            apply_medi_operator_core(
                &mut ws.fft_ws, &mut bufs, n, nx, ny, nz, vsx, vsy, vsz,
                &p_copy, w, d_kernel, mx, my, mz, vr, lambda, &mut ws.cg_ap
            );
        }

        // pap = p·ap (SIMD accelerated)
        let pap: f32 = dot_product_f32(&ws.cg_p, &ws.cg_ap);

        if pap.abs() < 1e-15 {
            break;
        }

        let alpha = rsold / pap;

        // x = x + alpha*p (SIMD accelerated)
        axpy_f32(x, alpha, &ws.cg_p);

        // r = r - alpha*ap (SIMD accelerated)
        axpy_f32(&mut ws.cg_r, -alpha, &ws.cg_ap);

        // rsnew = r·r (SIMD accelerated)
        let rsnew: f32 = norm_squared_f32(&ws.cg_r);
        let residual = rsnew.sqrt();

        // Check convergence
        if residual < tol * b_norm {
            break;
        }

        let beta = rsnew / rsold;

        // p = r + beta*p (SIMD accelerated)
        xpby_f32(&mut ws.cg_p, &ws.cg_r, beta);

        rsold = rsnew;
    }
}

/// MEDI L1 dipole inversion with full options (OPTIMIZED f32 VERSION)
///
/// # Arguments
/// * `local_field` - Local field/phase (RDF) in radians (nx * ny * nz)
/// * `n_std` - Noise standard deviation map (same size as local_field)
/// * `magnitude` - Magnitude image for gradient weighting (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz), 1 = brain
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `lambda` - Regularization parameter (default: 7.5e-5, matching MATLAB MEDI)
/// * `bdir` - B0 field direction (default: (0, 0, 1))
/// * `merit` - Enable iterative merit-based outlier adjustment (default: false)
/// * `smv` - Enable SMV preprocessing within MEDI (default: false)
/// * `smv_radius` - SMV radius in mm (default: 5.0)
/// * `data_weighting` - Data weighting mode: 0=uniform, 1=SNR (default: 1)
/// * `percentage` - Fraction of voxels considered edges (default: 0.3 = 30%, matching MATLAB gpct=30)
/// * `cg_tol` - CG solver tolerance (default: 0.01)
/// * `cg_max_iter` - CG maximum iterations (default: 10, matching MATLAB)
/// * `max_iter` - Maximum Gauss-Newton iterations (default: 30, matching MATLAB)
/// * `tol` - Convergence tolerance (default: 0.1)
///
/// # Returns
/// Susceptibility map (in same units as input field)
#[allow(clippy::too_many_arguments)]
pub fn medi_l1(
    local_field: &[f64],
    n_std: &[f64],
    magnitude: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    lambda: f64,
    bdir: (f64, f64, f64),
    merit: bool,
    smv: bool,
    smv_radius: f64,
    data_weighting: i32,
    percentage: f64,
    cg_tol: f64,
    cg_max_iter: usize,
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let n_total = nx * ny * nz;

    // Convert to f32 for internal computation (much faster on WASM)
    let vsx_f32 = vsx as f32;
    let vsy_f32 = vsy as f32;
    let vsz_f32 = vsz as f32;
    let lambda_f32 = lambda as f32;
    let bdir_f32 = (bdir.0 as f32, bdir.1 as f32, bdir.2 as f32);
    let smv_radius_f32 = smv_radius as f32;
    let percentage_f32 = percentage as f32;
    let cg_tol_f32 = cg_tol as f32;
    let tol_f32 = tol as f32;

    // Convert input arrays to f32
    let local_field_f32: Vec<f32> = local_field.iter().map(|&v| v as f32).collect();
    let n_std_f32: Vec<f32> = n_std.iter().map(|&v| v as f32).collect();
    let magnitude_f32: Vec<f32> = magnitude.iter().map(|&v| v as f32).collect();

    // Create workspace - this allocates all buffers ONCE
    let mut ws = MediWorkspace::new(nx, ny, nz, vsx_f32, vsy_f32, vsz_f32);

    // Working copies that may be modified by SMV preprocessing
    let mut rdf: Vec<f32> = local_field_f32.clone();
    let mut work_mask: Vec<u8> = mask.to_vec();
    let mut tempn: Vec<f32> = n_std_f32.clone();

    // Apply mask to N_std
    for i in 0..n_total {
        if mask[i] == 0 {
            tempn[i] = 0.0;
        }
    }

    // Generate dipole kernel
    let mut d_kernel = dipole_kernel_f32(nx, ny, nz, vsx_f32, vsy_f32, vsz_f32, bdir_f32);

    // SMV preprocessing (optional)
    let sphere_k = if smv {
        let sk = smv_kernel_f32(nx, ny, nz, vsx_f32, vsy_f32, vsz_f32, smv_radius_f32);

        // FFT of sphere kernel for convolution
        let mut sk_fft: Vec<Complex32> = sk.iter()
            .map(|&v| Complex32::new(v, 0.0))
            .collect();
        ws.fft_ws.fft3d(&mut sk_fft);

        // Erode mask: SMV(mask) > 0.999
        let mask_f32: Vec<f32> = work_mask.iter().map(|&m| m as f32).collect();
        let smv_mask = apply_smv_kernel_ws(&mask_f32, &sk_fft, &mut ws);
        for i in 0..n_total {
            work_mask[i] = if smv_mask[i] > 0.999 { 1 } else { 0 };
        }

        // Modify dipole kernel: D = (1 - SphereK) * D
        for i in 0..n_total {
            d_kernel[i] *= 1.0 - sk[i];
        }

        // Modify RDF: RDF = RDF - SMV(RDF)
        let smv_rdf = apply_smv_kernel_ws(&rdf, &sk_fft, &mut ws);
        for i in 0..n_total {
            rdf[i] -= smv_rdf[i];
            if work_mask[i] == 0 {
                rdf[i] = 0.0;
            }
        }

        // Modify noise: tempn = sqrt(SMV(tempn^2) + tempn^2)
        let tempn_sq: Vec<f32> = tempn.iter().map(|&t| t * t).collect();
        let smv_tempn_sq = apply_smv_kernel_ws(&tempn_sq, &sk_fft, &mut ws);
        for i in 0..n_total {
            tempn[i] = (smv_tempn_sq[i] + tempn_sq[i]).sqrt();
        }

        Some(sk_fft)
    } else {
        None
    };

    // Compute data weighting
    let mut m = dataterm_mask_f32(data_weighting, &tempn, &work_mask);

    // b0 = m * exp(i * RDF)
    let mut b0: Vec<Complex32> = rdf.iter()
        .zip(m.iter())
        .map(|(&f, &mi)| {
            let phase = Complex32::new(0.0, f);
            mi * phase.exp()
        })
        .collect();

    // Compute per-direction gradient weighting masks from magnitude edges
    // Returns (mx, my, mz) - separate masks for each gradient direction (matching MATLAB MEDI)
    let (w_gx, w_gy, w_gz) = gradient_mask_f32(&magnitude_f32, &work_mask, nx, ny, nz, vsx_f32, vsy_f32, vsz_f32, percentage_f32);

    // Fallback: if any mask is all zeros, use magnitude image (matching MATLAB)
    let w_gx = if w_gx.iter().any(|&v| v != 0.0) { w_gx } else { magnitude_f32.clone() };
    let w_gy = if w_gy.iter().any(|&v| v != 0.0) { w_gy } else { magnitude_f32.clone() };
    let w_gz = if w_gz.iter().any(|&v| v != 0.0) { w_gz } else { magnitude_f32.clone() };

    // Initialize susceptibility
    let mut chi = vec![0.0f32; n_total];
    let mut dx = vec![0.0f32; n_total];  // Reusable buffer for CG solution
    let mut rhs = vec![0.0f32; n_total]; // Reusable buffer for RHS
    let mut vr = vec![0.0f32; n_total];  // Reusable buffer for Vr (P in MATLAB)
    let mut w: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); n_total]; // Reusable buffer for w
    let mut chi_prev = vec![0.0f32; n_total]; // Reusable buffer for convergence check
    let mut badpoint = vec![0.0f32; n_total];
    let mut n_std_work: Vec<f32> = n_std_f32.clone();

    // MATLAB: beta = sqrt(eps(class(f))) where eps for f64 ≈ 2.22e-16, so sqrt(eps) ≈ 1.49e-8.
    // This is a regularization parameter for the P weight denominator, not a precision limit.
    // Using the same value as MATLAB (1.49e-8) is fine in f32 (representable, well above f32 eps).
    let beta = 1.49e-8_f32;

    // Gauss-Newton iterations
    for _iter in 0..max_iter {
        // Save chi_prev for convergence check
        chi_prev.copy_from_slice(&chi);

        // Compute P = 1 / sqrt(|m * grad(chi)|^2 + beta) using per-direction masks (SIMD accelerated)
        // MATLAB: P = 1 ./ sqrt(ux.*ux + uy.*uy + uz.*uz + beta);
        // where ux = mx .* grad_x(chi), uy = my .* grad_y(chi), uz = mz .* grad_z(chi)
        // Uses periodic BCs matching MATLAB's grad_ (which calls gradfp_mex)
        fgrad_periodic_inplace_f32(
            &mut ws.gx, &mut ws.gy, &mut ws.gz,
            &chi, nx, ny, nz, vsx_f32, vsy_f32, vsz_f32,
        );

        compute_p_weights_f32(&mut vr, &w_gx, &w_gy, &w_gz, &ws.gx, &ws.gy, &ws.gz, beta);

        // Compute w = m * exp(i * D*chi) using workspace
        apply_dipole_conv(&mut ws.fft_ws, &chi, &d_kernel, &mut ws.dipole_buf, &mut ws.complex_buf);
        for i in 0..n_total {
            let phase = Complex32::new(0.0, ws.dipole_buf[i]);
            w[i] = m[i] * phase.exp();
        }

        // Compute right-hand side using workspace
        compute_rhs_inplace(&chi, &w, &b0, &d_kernel, &w_gx, &w_gy, &w_gz, &vr, lambda_f32, &mut rhs, &mut ws);

        // Negate for CG (solving A*dx = -b) (SIMD accelerated)
        negate_f32(&mut rhs);

        // Solve A*dx = rhs using optimized CG with buffer reuse (no progress reporting)
        cg_solve_medi(&mut ws, &w, &d_kernel, &w_gx, &w_gy, &w_gz, &vr, lambda_f32, &rhs, &mut dx, cg_tol_f32, cg_max_iter, |_, _| {});

        // Update: chi = chi + dx (SIMD accelerated)
        axpy_f32(&mut chi, 1.0, &dx);

        // Check convergence (SIMD accelerated)
        let norm_dx_sq = norm_squared_f32(&dx);
        let norm_chi_sq = norm_squared_f32(&chi_prev);
        let rel_change = norm_dx_sq.sqrt() / (norm_chi_sq.sqrt() + 1e-6);

        // Merit adjustment (optional)
        if merit {
            // Compute residual: wres = m * exp(i * D*chi) - b0
            apply_dipole_conv(&mut ws.fft_ws, &chi, &d_kernel, &mut ws.dipole_buf, &mut ws.complex_buf);
            let mut wres: Vec<Complex32> = ws.dipole_buf.iter()
                .zip(m.iter())
                .zip(b0.iter())
                .map(|((&dc, &mi), &b0i)| {
                    let phase = Complex32::new(0.0, dc);
                    mi * phase.exp() - b0i
                })
                .collect();

            // Subtract mean over mask
            let mask_count = work_mask.iter().filter(|&&m| m != 0).count() as f32;
            if mask_count > 0.0 {
                let mean_wres: Complex32 = wres.iter()
                    .zip(work_mask.iter())
                    .filter(|(_, &m)| m != 0)
                    .map(|(w, _)| w)
                    .sum::<Complex32>() / mask_count;

                for i in 0..n_total {
                    if work_mask[i] != 0 {
                        wres[i] -= mean_wres;
                    }
                }
            }

            // Compute factor = std(abs(wres[mask])) * 6
            let abs_wres: Vec<f32> = wres.iter()
                .zip(work_mask.iter())
                .filter(|(_, &m)| m != 0)
                .map(|(w, _)| w.norm())
                .collect();

            if !abs_wres.is_empty() {
                let mean_abs: f32 = abs_wres.iter().sum::<f32>() / abs_wres.len() as f32;
                let var: f32 = abs_wres.iter()
                    .map(|&v| (v - mean_abs).powi(2))
                    .sum::<f32>() / abs_wres.len() as f32;
                let factor = var.sqrt() * 6.0;

                if factor > 1e-10 {
                    // Normalize wres by factor
                    let mut wres_norm: Vec<f32> = wres.iter()
                        .map(|w| w.norm() / factor)
                        .collect();

                    // Clamp values < 1 to 1
                    for v in wres_norm.iter_mut() {
                        if *v < 1.0 {
                            *v = 1.0;
                        }
                    }

                    // Mark bad points and update noise
                    for i in 0..n_total {
                        if wres_norm[i] > 1.0 {
                            badpoint[i] = 1.0;
                        }
                        if work_mask[i] != 0 {
                            n_std_work[i] *= wres_norm[i].powi(2);
                        }
                    }

                    // Recompute tempn
                    tempn = n_std_work.clone();
                    if let Some(ref sk_fft) = sphere_k {
                        let tempn_sq: Vec<f32> = tempn.iter().map(|&t| t * t).collect();
                        let smv_tempn_sq = apply_smv_kernel_ws(&tempn_sq, sk_fft, &mut ws);
                        for i in 0..n_total {
                            tempn[i] = (smv_tempn_sq[i] + tempn_sq[i]).sqrt();
                        }
                    }

                    // Recompute data weighting and b0
                    m = dataterm_mask_f32(data_weighting, &tempn, &work_mask);
                    b0 = rdf.iter()
                        .zip(m.iter())
                        .map(|(&f, &mi)| {
                            let phase = Complex32::new(0.0, f);
                            mi * phase.exp()
                        })
                        .collect();
                }
            }
        }

        if rel_change < tol_f32 {
            break;
        }
    }

    // Suppress unused variable warning
    let _ = badpoint;

    // Apply mask and convert back to f64
    chi.iter()
        .zip(mask.iter())
        .map(|(&c, &m)| if m == 0 { 0.0 } else { c as f64 })
        .collect()
}

/// Apply SMV kernel using workspace buffers (f32)
fn apply_smv_kernel_ws(
    x: &[f32],
    sk_fft: &[Complex32],
    ws: &mut MediWorkspace,
) -> Vec<f32> {
    let n_total = ws.n_total;

    // Copy to complex buffer
    for (c, &r) in ws.complex_buf.iter_mut().zip(x.iter()) {
        *c = Complex32::new(r, 0.0);
    }

    ws.fft_ws.fft3d(&mut ws.complex_buf);

    for i in 0..n_total {
        ws.complex_buf[i] *= sk_fft[i];
    }

    ws.fft_ws.ifft3d(&mut ws.complex_buf);

    ws.complex_buf.iter().map(|c| c.re).collect()
}

/// Compute RHS in-place using workspace buffers (f32)
/// Uses per-direction gradient masks (mx, my, mz) matching MATLAB MEDI
/// SIMD-accelerated for element-wise operations
fn compute_rhs_inplace(
    chi: &[f32],
    w: &[Complex32],
    b0: &[Complex32],
    d_kernel: &[f32],
    mx: &[f32],  // Per-direction gradient mask for x
    my: &[f32],  // Per-direction gradient mask for y
    mz: &[f32],  // Per-direction gradient mask for z
    vr: &[f32],
    lambda: f32,
    rhs: &mut [f32],
    ws: &mut MediWorkspace,
) {
    let n = ws.n_total;

    // Regularization term: div(m * P * m * grad(chi)) for each direction
    // MATLAB: b = lam .* gradAdj_(ux, uy, uz, vsz);
    // where ux = mx .* P .* mx .* grad_x(chi), etc.
    // Uses periodic BCs matching MATLAB's gradfp_mex / gradfp_adj_mex
    fgrad_periodic_inplace_f32(
        &mut ws.gx, &mut ws.gy, &mut ws.gz,
        chi, ws.nx, ws.ny, ws.nz,
        ws.vsx, ws.vsy, ws.vsz,
    );

    // Apply per-direction weights: ux = mx * P * mx * gx (SIMD accelerated)
    apply_gradient_weights_f32(
        &mut ws.reg_x, &mut ws.reg_y, &mut ws.reg_z,
        mx, my, mz, vr,
        &ws.gx, &ws.gy, &ws.gz,
    );

    bdiv_periodic_inplace_f32(
        &mut ws.div_buf,
        &ws.reg_x, &ws.reg_y, &ws.reg_z,
        ws.nx, ws.ny, ws.nz,
        ws.vsx, ws.vsy, ws.vsz,
    );

    // Data term: D^T(conj(w) * (-i) * (w - b0))
    // MATLAB: b = b + real(ifft3(conj(D) .* fft3(1i .* w2 .* (exp(1i.*(f - Dx)) - 1))));
    for i in 0..n {
        let diff = w[i] - b0[i];
        let conj_w = w[i].conj();
        let neg_i = Complex32::new(0.0, -1.0);
        ws.complex_buf2[i] = conj_w * neg_i * diff;
    }

    // Apply D^T (which is D for real symmetric kernel)
    ws.fft_ws.fft3d(&mut ws.complex_buf2);

    for i in 0..n {
        ws.complex_buf2[i] *= d_kernel[i];
    }

    ws.fft_ws.ifft3d(&mut ws.complex_buf2);

    // Extract real parts for SIMD combine operation
    for i in 0..n {
        ws.dipole_buf[i] = ws.complex_buf2[i].re;
    }

    // Combine terms: rhs = lambda * reg_term + data_term (SIMD accelerated, matching MATLAB)
    combine_terms_f32(rhs, &ws.div_buf, &ws.dipole_buf, lambda);
}

/// Generate data weighting mask (f32)
///
/// # Arguments
/// * `mode` - 0 for uniform weighting, 1 for SNR weighting
/// * `n_std` - Noise standard deviation
/// * `mask` - Binary mask
fn dataterm_mask_f32(mode: i32, n_std: &[f32], mask: &[u8]) -> Vec<f32> {
    let n = n_std.len();

    if mode == 0 {
        // Uniform weighting
        mask.iter().map(|&m| if m != 0 { 1.0 } else { 0.0 }).collect()
    } else {
        // SNR weighting: w = mask / N_std, normalized so mean over ROI = 1
        let mut w: Vec<f32> = n_std.iter()
            .zip(mask.iter())
            .map(|(&n, &m)| {
                if m != 0 && n > 1e-10 {
                    1.0 / n
                } else {
                    0.0
                }
            })
            .collect();

        // Compute mean over ROI
        let mask_count = mask.iter().filter(|&&m| m != 0).count() as f32;
        if mask_count > 0.0 {
            let sum: f32 = w.iter()
                .zip(mask.iter())
                .filter(|(_, &m)| m != 0)
                .map(|(&wi, _)| wi)
                .sum();
            let mean = sum / mask_count;

            if mean > 1e-10 {
                // Normalize so mean = 1
                for i in 0..n {
                    w[i] /= mean;
                }
            }
        }

        // Ensure zeros outside mask
        for i in 0..n {
            if mask[i] == 0 {
                w[i] = 0.0;
            }
        }

        w
    }
}

/// Generate per-direction gradient weighting masks (f32)
///
/// Computes separate edge masks for each gradient direction from magnitude image,
/// matching the MATLAB MEDI implementation (gradientMaskMedi.m).
/// Returns (mx, my, mz) where each mask is 1 (regularize) for non-edges, 0 for edges.
///
/// # Arguments
/// * `magnitude` - Magnitude image
/// * `mask` - Binary mask
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes
/// * `percentage` - Percentage of voxels considered to be edges (0.0-1.0, e.g., 0.3 = 30% edges)
///
/// # Returns
/// Tuple of (mx, my, mz) per-direction binary gradient masks
fn gradient_mask_f32(
    magnitude: &[f32],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f32, vsy: f32, vsz: f32,
    percentage: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n_total = nx * ny * nz;

    // Normalize magnitude by max value within mask (matching MATLAB)
    let mag_max = magnitude.iter()
        .zip(mask.iter())
        .filter(|(_, &m)| m != 0)
        .map(|(&v, _)| v.abs())
        .fold(0.0_f32, f32::max);

    let mag_normalized: Vec<f32> = magnitude.iter()
        .zip(mask.iter())
        .map(|(&m, &msk)| {
            if msk != 0 && mag_max > 1e-10 {
                m / mag_max
            } else {
                0.0
            }
        })
        .collect();

    // Compute gradient of normalized magnitude (using linear extrapolation BCs)
    let (gx, gy, gz) = fgrad_linext_f32(&mag_normalized, nx, ny, nz, vsx, vsy, vsz);

    // Take absolute values of each gradient direction
    let abs_gx: Vec<f32> = gx.iter().map(|&v| v.abs()).collect();
    let abs_gy: Vec<f32> = gy.iter().map(|&v| v.abs()).collect();
    let abs_gz: Vec<f32> = gz.iter().map(|&v| v.abs()).collect();

    // Collect all gradient values within mask for threshold computation
    let mut all_grads: Vec<f32> = Vec::with_capacity(3 * n_total);
    for i in 0..n_total {
        if mask[i] != 0 {
            all_grads.push(abs_gx[i]);
            all_grads.push(abs_gy[i]);
            all_grads.push(abs_gz[i]);
        }
    }

    if all_grads.is_empty() {
        return (vec![1.0; n_total], vec![1.0; n_total], vec![1.0; n_total]);
    }

    // Sort to find percentile threshold (100 - percentage)
    // MATLAB: thr = prctile([mx(mask); my(mask); mz(mask)], 100 - p);
    all_grads.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let percentile_idx = ((1.0 - percentage) * (all_grads.len() - 1) as f32) as usize;
    let threshold = all_grads[percentile_idx.min(all_grads.len() - 1)];

    // Create per-direction masks: 1 where gradient < threshold (non-edges), 0 at edges
    // MATLAB: mx = mx < thr; my = my < thr; mz = mz < thr;
    let mx: Vec<f32> = abs_gx.iter()
        .zip(mask.iter())
        .map(|(&g, &m)| if m != 0 && g < threshold { 1.0 } else { 0.0 })
        .collect();

    let my: Vec<f32> = abs_gy.iter()
        .zip(mask.iter())
        .map(|(&g, &m)| if m != 0 && g < threshold { 1.0 } else { 0.0 })
        .collect();

    let mz: Vec<f32> = abs_gz.iter()
        .zip(mask.iter())
        .map(|(&g, &m)| if m != 0 && g < threshold { 1.0 } else { 0.0 })
        .collect();

    (mx, my, mz)
}

/// Forward difference gradient with linear extrapolation boundary conditions (f32)
/// Matches MATLAB's gradf behavior: dx(end) = dx(end-1)
fn fgrad_linext_f32(
    x: &[f32],
    nx: usize, ny: usize, nz: usize,
    vsx: f32, vsy: f32, vsz: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n_total = nx * ny * nz;
    let mut gx = vec![0.0f32; n_total];
    let mut gy = vec![0.0f32; n_total];
    let mut gz = vec![0.0f32; n_total];
    fgrad_linext_inplace_f32(&mut gx, &mut gy, &mut gz, x, nx, ny, nz, vsx, vsy, vsz);
    (gx, gy, gz)
}

/// Forward difference gradient with linear extrapolation boundary conditions (f32, in-place)
/// Matches MATLAB's gradf behavior: dx(end) = dx(end-1)
#[inline]
fn fgrad_linext_inplace_f32(
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

                // Forward difference with linear extrapolation at boundary
                // MATLAB: dx(end,:,:) = dx(end-1,:,:)
                if i + 1 < nx {
                    gx[idx] = (x[idx + 1] - x_val) * hx;
                } else if i > 0 {
                    // Copy from previous (linear extrapolation)
                    gx[idx] = gx[idx - 1];
                } else {
                    gx[idx] = 0.0;
                }

                if j + 1 < ny {
                    gy[idx] = (x[i + (j + 1) * nx + k_offset] - x_val) * hy;
                } else if j > 0 {
                    gy[idx] = gy[i + (j - 1) * nx + k_offset];
                } else {
                    gy[idx] = 0.0;
                }

                if k + 1 < nz {
                    gz[idx] = (x[i + j_offset + (k + 1) * nx * ny] - x_val) * hz;
                } else if k > 0 {
                    gz[idx] = gz[i + j_offset + (k - 1) * nx * ny];
                } else {
                    gz[idx] = 0.0;
                }
            }
        }
    }
}

/// Backward divergence with linear extrapolation boundary conditions (f32, in-place)
/// Adjoint of fgrad_linext_inplace_f32
/// Matches MATLAB's gradAdj_ behavior
#[inline]
fn bdiv_linext_inplace_f32(
    div: &mut [f32],
    gx: &[f32], gy: &[f32], gz: &[f32],
    nx: usize, ny: usize, nz: usize,
    vsx: f32, vsy: f32, vsz: f32,
) {
    let hx = -1.0 / vsx;  // Negative for adjoint
    let hy = -1.0 / vsy;
    let hz = -1.0 / vsz;

    for k in 0..nz {
        let k_offset = k * nx * ny;

        for j in 0..ny {
            let j_offset = j * nx;

            for i in 0..nx {
                let idx = i + j_offset + k_offset;

                // Adjoint of forward difference with linear extrapolation BC
                // MATLAB: dx = dx - circshift(dx, [1,0,0]); dx = ih(1) .* dx;
                // With boundary: dx(1) = dx(1), others = dx(i) - dx(i-1)
                let gx_term = if i > 0 {
                    (gx[idx] - gx[idx - 1]) * hx
                } else {
                    gx[idx] * hx
                };

                let gy_term = if j > 0 {
                    (gy[idx] - gy[i + (j - 1) * nx + k_offset]) * hy
                } else {
                    gy[idx] * hy
                };

                let gz_term = if k > 0 {
                    (gz[idx] - gz[i + j_offset + (k - 1) * nx * ny]) * hz
                } else {
                    gz[idx] * hz
                };

                div[idx] = gx_term + gy_term + gz_term;
            }
        }
    }
}

/// Forward difference gradient with periodic boundary conditions (f32, in-place)
/// Matches MATLAB's gradfp_mex used inside MEDI iterations.
/// At boundaries, wraps around: dx(end) = (x(1) - x(end)) / h
#[inline]
fn fgrad_periodic_inplace_f32(
    gx: &mut [f32], gy: &mut [f32], gz: &mut [f32],
    x: &[f32],
    nx: usize, ny: usize, nz: usize,
    vsx: f32, vsy: f32, vsz: f32,
) {
    let hx = 1.0 / vsx;
    let hy = 1.0 / vsy;
    let hz = 1.0 / vsz;
    let nxny = nx * ny;

    for k in 0..nz {
        let k_offset = k * nxny;

        for j in 0..ny {
            let j_offset = j * nx;

            for i in 0..nx {
                let idx = i + j_offset + k_offset;
                let x_val = x[idx];

                // x-direction: periodic wrap at i = nx-1
                let x_next = if i + 1 < nx { x[idx + 1] } else { x[j_offset + k_offset] };
                gx[idx] = (x_next - x_val) * hx;

                // y-direction: periodic wrap at j = ny-1
                let y_next = if j + 1 < ny { x[i + (j + 1) * nx + k_offset] } else { x[i + k_offset] };
                gy[idx] = (y_next - x_val) * hy;

                // z-direction: periodic wrap at k = nz-1
                let z_next = if k + 1 < nz { x[i + j_offset + (k + 1) * nxny] } else { x[i + j_offset] };
                gz[idx] = (z_next - x_val) * hz;
            }
        }
    }
}

/// Backward divergence with periodic boundary conditions (f32, in-place)
/// Adjoint of fgrad_periodic_inplace_f32, matching MATLAB's gradfp_adj_mex.
/// At boundaries, wraps around: at i=0, uses gx(end) instead of zero.
#[inline]
fn bdiv_periodic_inplace_f32(
    div: &mut [f32],
    gx: &[f32], gy: &[f32], gz: &[f32],
    nx: usize, ny: usize, nz: usize,
    vsx: f32, vsy: f32, vsz: f32,
) {
    let hx = -1.0 / vsx;  // Negative for adjoint
    let hy = -1.0 / vsy;
    let hz = -1.0 / vsz;
    let nxny = nx * ny;

    for k in 0..nz {
        let k_offset = k * nxny;

        for j in 0..ny {
            let j_offset = j * nx;

            for i in 0..nx {
                let idx = i + j_offset + k_offset;

                // x-direction: at i=0, wrap to gx[nx-1,j,k]
                let gx_prev = if i > 0 { gx[idx - 1] } else { gx[(nx - 1) + j_offset + k_offset] };
                let gx_term = (gx[idx] - gx_prev) * hx;

                // y-direction: at j=0, wrap to gy[i,ny-1,k]
                let gy_prev = if j > 0 { gy[i + (j - 1) * nx + k_offset] } else { gy[i + (ny - 1) * nx + k_offset] };
                let gy_term = (gy[idx] - gy_prev) * hy;

                // z-direction: at k=0, wrap to gz[i,j,nz-1]
                let gz_prev = if k > 0 { gz[i + j_offset + (k - 1) * nxny] } else { gz[i + j_offset + (nz - 1) * nxny] };
                let gz_term = (gz[idx] - gz_prev) * hz;

                div[idx] = gx_term + gy_term + gz_term;
            }
        }
    }
}

/// MEDI L1 with progress callback (OPTIMIZED f32 VERSION)
///
/// Same as `medi_l1` but calls `progress_callback(iteration, max_iter)` each iteration.
#[allow(clippy::too_many_arguments)]
pub fn medi_l1_with_progress<F>(
    local_field: &[f64],
    n_std: &[f64],
    magnitude: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    lambda: f64,
    bdir: (f64, f64, f64),
    merit: bool,
    smv: bool,
    smv_radius: f64,
    data_weighting: i32,
    percentage: f64,
    cg_tol: f64,
    cg_max_iter: usize,
    max_iter: usize,
    tol: f64,
    mut progress_callback: F,
) -> Vec<f64>
where
    F: FnMut(usize, usize),
{
    let n_total = nx * ny * nz;

    // Convert to f32 for internal computation
    let vsx_f32 = vsx as f32;
    let vsy_f32 = vsy as f32;
    let vsz_f32 = vsz as f32;
    let lambda_f32 = lambda as f32;
    let bdir_f32 = (bdir.0 as f32, bdir.1 as f32, bdir.2 as f32);
    let smv_radius_f32 = smv_radius as f32;
    let percentage_f32 = percentage as f32;
    let cg_tol_f32 = cg_tol as f32;
    let tol_f32 = tol as f32;

    // Convert input arrays to f32
    let local_field_f32: Vec<f32> = local_field.iter().map(|&v| v as f32).collect();
    let n_std_f32: Vec<f32> = n_std.iter().map(|&v| v as f32).collect();
    let magnitude_f32: Vec<f32> = magnitude.iter().map(|&v| v as f32).collect();

    // Create workspace - allocates all buffers ONCE
    let mut ws = MediWorkspace::new(nx, ny, nz, vsx_f32, vsy_f32, vsz_f32);

    // Working copies that may be modified by SMV preprocessing
    let mut rdf: Vec<f32> = local_field_f32.clone();
    let mut work_mask: Vec<u8> = mask.to_vec();
    let mut tempn: Vec<f32> = n_std_f32.clone();

    // Apply mask to N_std
    for i in 0..n_total {
        if mask[i] == 0 {
            tempn[i] = 0.0;
        }
    }

    // Generate dipole kernel
    let mut d_kernel = dipole_kernel_f32(nx, ny, nz, vsx_f32, vsy_f32, vsz_f32, bdir_f32);

    // SMV preprocessing (optional)
    let sphere_k = if smv {
        let sk = smv_kernel_f32(nx, ny, nz, vsx_f32, vsy_f32, vsz_f32, smv_radius_f32);

        // FFT of sphere kernel for convolution
        let mut sk_fft: Vec<Complex32> = sk.iter()
            .map(|&v| Complex32::new(v, 0.0))
            .collect();
        ws.fft_ws.fft3d(&mut sk_fft);

        // Erode mask: SMV(mask) > 0.999
        let mask_f32: Vec<f32> = work_mask.iter().map(|&m| m as f32).collect();
        let smv_mask = apply_smv_kernel_ws(&mask_f32, &sk_fft, &mut ws);
        for i in 0..n_total {
            work_mask[i] = if smv_mask[i] > 0.999 { 1 } else { 0 };
        }

        // Modify dipole kernel: D = (1 - SphereK) * D
        for i in 0..n_total {
            d_kernel[i] *= 1.0 - sk[i];
        }

        // Modify RDF: RDF = RDF - SMV(RDF)
        let smv_rdf = apply_smv_kernel_ws(&rdf, &sk_fft, &mut ws);
        for i in 0..n_total {
            rdf[i] -= smv_rdf[i];
            if work_mask[i] == 0 {
                rdf[i] = 0.0;
            }
        }

        // Modify noise: tempn = sqrt(SMV(tempn^2) + tempn^2)
        let tempn_sq: Vec<f32> = tempn.iter().map(|&t| t * t).collect();
        let smv_tempn_sq = apply_smv_kernel_ws(&tempn_sq, &sk_fft, &mut ws);
        for i in 0..n_total {
            tempn[i] = (smv_tempn_sq[i] + tempn_sq[i]).sqrt();
        }

        Some(sk_fft)
    } else {
        None
    };

    // Compute data weighting
    let mut m = dataterm_mask_f32(data_weighting, &tempn, &work_mask);

    // b0 = m * exp(i * RDF)
    let mut b0: Vec<Complex32> = rdf.iter()
        .zip(m.iter())
        .map(|(&f, &mi)| {
            let phase = Complex32::new(0.0, f);
            mi * phase.exp()
        })
        .collect();

    // Compute per-direction gradient weighting masks from magnitude edges
    // Returns (mx, my, mz) - separate masks for each gradient direction (matching MATLAB MEDI)
    let (w_gx, w_gy, w_gz) = gradient_mask_f32(&magnitude_f32, &work_mask, nx, ny, nz, vsx_f32, vsy_f32, vsz_f32, percentage_f32);

    // Fallback: if any mask is all zeros, use magnitude image (matching MATLAB)
    let w_gx = if w_gx.iter().any(|&v| v != 0.0) { w_gx } else { magnitude_f32.clone() };
    let w_gy = if w_gy.iter().any(|&v| v != 0.0) { w_gy } else { magnitude_f32.clone() };
    let w_gz = if w_gz.iter().any(|&v| v != 0.0) { w_gz } else { magnitude_f32.clone() };

    // Initialize susceptibility and reusable buffers
    let mut chi = vec![0.0f32; n_total];
    let mut dx = vec![0.0f32; n_total];
    let mut rhs = vec![0.0f32; n_total];
    let mut vr = vec![0.0f32; n_total];
    let mut w: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); n_total];
    let mut chi_prev = vec![0.0f32; n_total];
    let mut badpoint = vec![0.0f32; n_total];
    let mut n_std_work: Vec<f32> = n_std_f32.clone();

    // MATLAB: beta = sqrt(eps(class(f))) where eps for f64 ≈ 2.22e-16, so sqrt(eps) ≈ 1.49e-8.
    // This is a regularization parameter for the P weight denominator, not a precision limit.
    // Using the same value as MATLAB (1.49e-8) is fine in f32 (representable, well above f32 eps).
    let beta = 1.49e-8_f32;

    // Total progress = GN iterations * CG iterations per GN
    let total_steps = max_iter * cg_max_iter;

    // Gauss-Newton iterations
    for iter in 0..max_iter {
        chi_prev.copy_from_slice(&chi);

        // Compute P = 1 / sqrt(|m * grad(chi)|^2 + beta) using per-direction masks (SIMD accelerated)
        // MATLAB: P = 1 ./ sqrt(ux.*ux + uy.*uy + uz.*uz + beta);
        // Uses periodic BCs matching MATLAB's grad_ (which calls gradfp_mex)
        fgrad_periodic_inplace_f32(
            &mut ws.gx, &mut ws.gy, &mut ws.gz,
            &chi, nx, ny, nz, vsx_f32, vsy_f32, vsz_f32,
        );

        compute_p_weights_f32(&mut vr, &w_gx, &w_gy, &w_gz, &ws.gx, &ws.gy, &ws.gz, beta);

        // Compute w = m * exp(i * D*chi)
        apply_dipole_conv(&mut ws.fft_ws, &chi, &d_kernel, &mut ws.dipole_buf, &mut ws.complex_buf);
        for i in 0..n_total {
            let phase = Complex32::new(0.0, ws.dipole_buf[i]);
            w[i] = m[i] * phase.exp();
        }

        // Compute right-hand side
        compute_rhs_inplace(&chi, &w, &b0, &d_kernel, &w_gx, &w_gy, &w_gz, &vr, lambda_f32, &mut rhs, &mut ws);

        // Negate for CG (solving A*dx = -b) (SIMD accelerated)
        negate_f32(&mut rhs);

        // Solve A*dx = rhs using optimized CG with combined progress reporting
        // Progress = (gn_iter * cg_max_iter + cg_iter) / (max_iter * cg_max_iter)
        let gn_iter = iter;
        cg_solve_medi(
            &mut ws, &w, &d_kernel, &w_gx, &w_gy, &w_gz, &vr, lambda_f32, &rhs, &mut dx, cg_tol_f32, cg_max_iter,
            |cg_iter, cg_total| {
                let current = gn_iter * cg_total + cg_iter;
                progress_callback(current, total_steps);
            }
        );

        // Update: chi = chi + dx (SIMD accelerated)
        axpy_f32(&mut chi, 1.0, &dx);

        // Check convergence (SIMD accelerated)
        let norm_dx_sq = norm_squared_f32(&dx);
        let norm_chi_sq = norm_squared_f32(&chi_prev);
        let rel_change = norm_dx_sq.sqrt() / (norm_chi_sq.sqrt() + 1e-6);

        // Merit adjustment (optional)
        if merit {
            // Compute residual: wres = m * exp(i * D*chi) - b0
            apply_dipole_conv(&mut ws.fft_ws, &chi, &d_kernel, &mut ws.dipole_buf, &mut ws.complex_buf);
            let mut wres: Vec<Complex32> = ws.dipole_buf.iter()
                .zip(m.iter())
                .zip(b0.iter())
                .map(|((&dc, &mi), &b0i)| {
                    let phase = Complex32::new(0.0, dc);
                    mi * phase.exp() - b0i
                })
                .collect();

            // Subtract mean over mask
            let mask_count = work_mask.iter().filter(|&&m| m != 0).count() as f32;
            if mask_count > 0.0 {
                let mean_wres: Complex32 = wres.iter()
                    .zip(work_mask.iter())
                    .filter(|(_, &m)| m != 0)
                    .map(|(w, _)| w)
                    .sum::<Complex32>() / mask_count;

                for i in 0..n_total {
                    if work_mask[i] != 0 {
                        wres[i] -= mean_wres;
                    }
                }
            }

            // Compute factor = std(abs(wres[mask])) * 6
            let abs_wres: Vec<f32> = wres.iter()
                .zip(work_mask.iter())
                .filter(|(_, &m)| m != 0)
                .map(|(w, _)| w.norm())
                .collect();

            if !abs_wres.is_empty() {
                let mean_abs: f32 = abs_wres.iter().sum::<f32>() / abs_wres.len() as f32;
                let var: f32 = abs_wres.iter()
                    .map(|&v| (v - mean_abs).powi(2))
                    .sum::<f32>() / abs_wres.len() as f32;
                let factor = var.sqrt() * 6.0;

                if factor > 1e-10 {
                    // Normalize wres by factor
                    let mut wres_norm: Vec<f32> = wres.iter()
                        .map(|w| w.norm() / factor)
                        .collect();

                    // Clamp values < 1 to 1
                    for v in wres_norm.iter_mut() {
                        if *v < 1.0 {
                            *v = 1.0;
                        }
                    }

                    // Mark bad points and update noise
                    for i in 0..n_total {
                        if wres_norm[i] > 1.0 {
                            badpoint[i] = 1.0;
                        }
                        if work_mask[i] != 0 {
                            n_std_work[i] *= wres_norm[i].powi(2);
                        }
                    }

                    // Recompute tempn
                    tempn = n_std_work.clone();
                    if let Some(ref sk_fft) = sphere_k {
                        let tempn_sq: Vec<f32> = tempn.iter().map(|&t| t * t).collect();
                        let smv_tempn_sq = apply_smv_kernel_ws(&tempn_sq, sk_fft, &mut ws);
                        for i in 0..n_total {
                            tempn[i] = (smv_tempn_sq[i] + tempn_sq[i]).sqrt();
                        }
                    }

                    // Recompute data weighting and b0
                    m = dataterm_mask_f32(data_weighting, &tempn, &work_mask);
                    b0 = rdf.iter()
                        .zip(m.iter())
                        .map(|(&f, &mi)| {
                            let phase = Complex32::new(0.0, f);
                            mi * phase.exp()
                        })
                        .collect();
                }
            }
        }

        if rel_change < tol_f32 {
            // Report completion on early convergence
            progress_callback(total_steps, total_steps);
            break;
        }
    }

    // Suppress unused variable warning
    let _ = badpoint;

    // Apply mask and convert back to f64
    chi.iter()
        .zip(mask.iter())
        .map(|(&c, &m)| if m == 0 { 0.0 } else { c as f64 })
        .collect()
}

/// MEDI with default parameters (backward compatible)
pub fn medi_l1_default(
    local_field: &[f64],
    mask: &[u8],
    magnitude: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> Vec<f64> {
    // Create uniform noise std (no SNR weighting)
    let n_std = vec![1.0; local_field.len()];

    medi_l1(
        local_field,
        &n_std,
        magnitude,
        mask,
        nx, ny, nz,
        vsx, vsy, vsz,
        7.5e-5,            // lambda (matching MATLAB default)
        (0.0, 0.0, 1.0),   // bdir
        false,             // merit
        false,             // smv
        5.0,               // smv_radius
        1,                 // data_weighting (SNR mode)
        0.3,               // percentage (30% edges, matching MATLAB gpct=30)
        0.01,              // cg_tol
        10,                // cg_max_iter (matching MATLAB default)
        30,                // max_iter (matching MATLAB default)
        0.1,               // tol
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataterm_mask_uniform() {
        let n_std = vec![1.0f32; 27];
        let mask = vec![1u8; 27];

        let w = dataterm_mask_f32(0, &n_std, &mask);

        for &wi in w.iter() {
            assert!((wi - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dataterm_mask_snr() {
        let n_std = vec![2.0f32; 27];
        let mask = vec![1u8; 27];

        let w = dataterm_mask_f32(1, &n_std, &mask);

        // Mean should be 1
        let mean: f32 = w.iter().sum::<f32>() / 27.0;
        assert!((mean - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_gradient_mask_constant() {
        // Constant magnitude should have no edges (all gradients are zero)
        let mag = vec![1.0f32; 8 * 8 * 8];
        let mask = vec![1u8; 8 * 8 * 8];

        let (mx, my, mz) = gradient_mask_f32(&mag, &mask, 8, 8, 8, 1.0, 1.0, 1.0, 0.3);

        // All should be binary masks (0 or 1)
        for i in 0..(8 * 8 * 8) {
            assert!(mx[i] == 0.0 || mx[i] == 1.0, "mx should be binary, got {}", mx[i]);
            assert!(my[i] == 0.0 || my[i] == 1.0, "my should be binary, got {}", my[i]);
            assert!(mz[i] == 0.0 || mz[i] == 1.0, "mz should be binary, got {}", mz[i]);
        }
    }

    #[test]
    fn test_medi_zero_field() {
        let n = 8;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];
        let mag = vec![1.0; n * n * n];
        let n_std = vec![1.0; n * n * n];

        let chi = medi_l1(
            &field, &n_std, &mag, &mask, n, n, n, 1.0, 1.0, 1.0,
            1000.0, (0.0, 0.0, 1.0), false, false, 5.0, 1, 0.9, 0.1, 10, 3, 0.1
        );

        for &val in chi.iter() {
            assert!(val.abs() < 1e-4, "Zero field should give near-zero chi, got {}", val);
        }
    }

    #[test]
    fn test_medi_finite() {
        let n = 8;
        let field: Vec<f64> = (0..n*n*n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];
        let mag = vec![1.0; n * n * n];
        let n_std = vec![1.0; n * n * n];

        let chi = medi_l1(
            &field, &n_std, &mag, &mask, n, n, n, 1.0, 1.0, 1.0,
            1000.0, (0.0, 0.0, 1.0), false, false, 5.0, 1, 0.9, 0.1, 10, 3, 0.1
        );

        for (i, &val) in chi.iter().enumerate() {
            assert!(val.is_finite(), "Chi should be finite at index {}", i);
        }
    }

    #[test]
    fn test_medi_with_smv() {
        let n = 8;
        let field: Vec<f64> = (0..n*n*n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];
        let mag = vec![1.0; n * n * n];
        let n_std = vec![1.0; n * n * n];

        // Test with SMV enabled
        let chi = medi_l1(
            &field, &n_std, &mag, &mask, n, n, n, 1.0, 1.0, 1.0,
            1000.0, (0.0, 0.0, 1.0), false, true, 2.0, 1, 0.9, 0.1, 10, 3, 0.1
        );

        for (i, &val) in chi.iter().enumerate() {
            assert!(val.is_finite(), "Chi with SMV should be finite at index {}", i);
        }
    }

    #[test]
    fn test_medi_mask() {
        let n = 8;
        let field: Vec<f64> = (0..n*n*n).map(|i| (i as f64) * 0.001).collect();
        let mut mask = vec![1u8; n * n * n];
        let mag = vec![1.0; n * n * n];
        let n_std = vec![1.0; n * n * n];
        mask[0] = 0;
        mask[10] = 0;

        let chi = medi_l1(
            &field, &n_std, &mag, &mask, n, n, n, 1.0, 1.0, 1.0,
            1000.0, (0.0, 0.0, 1.0), false, false, 5.0, 1, 0.9, 0.1, 10, 3, 0.1
        );

        assert_eq!(chi[0], 0.0, "Masked voxel should be zero");
        assert_eq!(chi[10], 0.0, "Masked voxel should be zero");
    }

    /// Debug test: run MEDI step-by-step on real data, saving intermediates
    /// for comparison with Octave reference (other/medi_debug_octave.m).
    /// Run with: cargo test --release test_medi_debug -- --ignored --nocapture
    #[test]
    #[ignore]
    fn test_medi_debug() {
        let data_path = "/home/ashley/OUT/2bgRemoved.nii";
        if !std::path::Path::new(data_path).exists() {
            eprintln!("Skipping: {} not found", data_path);
            return;
        }

        let outdir = "/home/ashley/OUT/debug";
        std::fs::create_dir_all(outdir).ok();

        // Read NIfTI
        let bytes = std::fs::read(data_path).unwrap();
        let nifti_data = crate::nifti_io::load_nifti(&bytes).unwrap();
        let (nx, ny, nz) = nifti_data.dims;
        let (vsx, vsy, vsz) = nifti_data.voxel_size;

        let n_total = nx * ny * nz;
        let vsx_f32 = vsx as f32;
        let vsy_f32 = vsy as f32;
        let vsz_f32 = vsz as f32;

        eprintln!("Data: {}x{}x{}, voxel: {}x{}x{}", nx, ny, nz, vsx, vsy, vsz);

        // Convert to f32
        let local_field: Vec<f32> = nifti_data.data.iter().map(|&v| v as f32).collect();

        // Create mask from non-zero voxels
        let mask: Vec<u8> = local_field.iter()
            .map(|&v| if v.abs() > 1e-10 { 1 } else { 0 })
            .collect();
        let mask_count: usize = mask.iter().filter(|&&m| m != 0).count();
        eprintln!("Mask voxels: {} / {}", mask_count, n_total);

        // Save inputs
        save_f32_raw(&local_field, &format!("{}/f_rust.raw", outdir));
        let mask_f32: Vec<f32> = mask.iter().map(|&m| m as f32).collect();
        save_f32_raw(&mask_f32, &format!("{}/mask_rust.raw", outdir));

        // Parameters (matching MATLAB defaults)
        let lambda: f32 = 7.5e-5;
        let beta: f32 = 1.49e-8;
        let bdir = (0.0f32, 0.0f32, 1.0f32);
        let cg_tol: f32 = 0.1;  // Match MATLAB tolcg default
        let cg_max_iter: usize = 10;

        // Dipole kernel
        let d_kernel = crate::kernels::dipole::dipole_kernel_f32(
            nx, ny, nz, vsx_f32, vsy_f32, vsz_f32, bdir,
        );
        save_f32_raw(&d_kernel, &format!("{}/D_rust.raw", outdir));
        eprintln!("D: min={} max={} D[0]={}", fmin(&d_kernel), fmax(&d_kernel), d_kernel[0]);

        // Workspace
        let mut ws = MediWorkspace::new(nx, ny, nz, vsx_f32, vsy_f32, vsz_f32);

        // Data weighting: uniform m = mask (matching w=ones in Octave)
        let m: Vec<f32> = mask.iter().map(|&m| if m != 0 { 1.0 } else { 0.0 }).collect();

        // b0 = m * exp(i * f)
        let b0: Vec<Complex32> = local_field.iter().zip(m.iter())
            .map(|(&f, &mi)| {
                let phase = Complex32::new(0.0, f);
                mi * phase.exp()
            })
            .collect();

        // Gradient mask: uniform magnitude -> mx=my=mz=mask
        let w_gx: Vec<f32> = m.clone();
        let w_gy: Vec<f32> = m.clone();
        let w_gz: Vec<f32> = m.clone();

        // ===== ITERATION 1 (chi = 0) =====
        eprintln!("\n=== Iteration 1 ===");
        let mut chi = vec![0.0f32; n_total];
        let mut dx = vec![0.0f32; n_total];
        let mut rhs = vec![0.0f32; n_total];
        let mut vr = vec![0.0f32; n_total];
        let mut w: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); n_total];

        // P weights (chi=0 -> gradient=0 -> P = 1/sqrt(beta))
        fgrad_periodic_inplace_f32(
            &mut ws.gx, &mut ws.gy, &mut ws.gz,
            &chi, nx, ny, nz, vsx_f32, vsy_f32, vsz_f32,
        );
        compute_p_weights_f32(&mut vr, &w_gx, &w_gy, &w_gz, &ws.gx, &ws.gy, &ws.gz, beta);
        save_f32_raw(&vr, &format!("{}/P1_rust.raw", outdir));
        eprintln!("P1: min={} max={} mean={}", fmin(&vr), fmax(&vr), fmean(&vr));

        // w = m * exp(i * D*chi) = m (since chi=0)
        apply_dipole_conv(&mut ws.fft_ws, &chi, &d_kernel, &mut ws.dipole_buf, &mut ws.complex_buf);
        for i in 0..n_total {
            let phase = Complex32::new(0.0, ws.dipole_buf[i]);
            w[i] = m[i] * phase.exp();
        }

        // RHS
        compute_rhs_inplace(
            &chi, &w, &b0, &d_kernel,
            &w_gx, &w_gy, &w_gz, &vr, lambda,
            &mut rhs, &mut ws,
        );
        save_f32_raw(&rhs, &format!("{}/rhs1_rust.raw", outdir));
        eprintln!("RHS1: min={} max={} norm={}", fmin(&rhs), fmax(&rhs), fnorm(&rhs));

        // Negate for CG
        negate_f32(&mut rhs);

        // CG solve (with iteration-level residual logging)
        let mut cg_residuals: Vec<f32> = Vec::new();
        {
            // Manual CG to capture residuals (matching cg_solve_medi but with logging)
            let n = ws.n_total;
            let (nx, ny, nz) = (ws.nx, ws.ny, ws.nz);
            let (vsx, vsy, vsz) = (ws.vsx, ws.vsy, ws.vsz);

            dx.fill(0.0);
            ws.cg_r.copy_from_slice(&rhs);
            ws.cg_p.copy_from_slice(&ws.cg_r);
            let mut rsold: f32 = norm_squared_f32(&ws.cg_r);
            let b_norm: f32 = norm_squared_f32(&rhs).sqrt();

            let mut p_copy = vec![0.0f32; n];
            let mut prev_residual = rsold.sqrt();

            for cg_iter in 0..cg_max_iter {
                let residual_before = rsold.sqrt();
                cg_residuals.push(residual_before);

                p_copy.copy_from_slice(&ws.cg_p);
                {
                    let mut bufs = MediOpBuffers {
                        gx: &mut ws.gx, gy: &mut ws.gy, gz: &mut ws.gz,
                        reg_x: &mut ws.reg_x, reg_y: &mut ws.reg_y, reg_z: &mut ws.reg_z,
                        div_buf: &mut ws.div_buf, dipole_buf: &mut ws.dipole_buf,
                        complex_buf: &mut ws.complex_buf, complex_buf2: &mut ws.complex_buf2,
                    };
                    apply_medi_operator_core(
                        &mut ws.fft_ws, &mut bufs, n, nx, ny, nz, vsx, vsy, vsz,
                        &p_copy, &w, &d_kernel, &w_gx, &w_gy, &w_gz, &vr, lambda, &mut ws.cg_ap,
                    );
                }

                let pap: f32 = dot_product_f32(&ws.cg_p, &ws.cg_ap);
                if pap.abs() < 1e-15 { break; }
                let alpha = rsold / pap;

                axpy_f32(&mut dx, alpha, &ws.cg_p);
                axpy_f32(&mut ws.cg_r, -alpha, &ws.cg_ap);

                let rsnew: f32 = norm_squared_f32(&ws.cg_r);
                let residual = rsnew.sqrt();

                eprintln!("  CG iter {}: res={:.6e}, alpha={:.6e}, pap={:.6e}",
                    cg_iter + 1, residual, alpha, pap);

                if residual < cg_tol * b_norm { break; }

                // No stall detection in this debug version (matching MATLAB)

                let beta_cg = rsnew / rsold;
                xpby_f32(&mut ws.cg_p, &ws.cg_r, beta_cg);
                rsold = rsnew;
                prev_residual = residual;
            }
        }
        save_f32_raw(&dx, &format!("{}/dx1_rust.raw", outdir));
        eprintln!("dx1: min={} max={} norm={}", fmin(&dx), fmax(&dx), fnorm(&dx));

        // Update chi
        axpy_f32(&mut chi, 1.0, &dx);
        save_f32_raw(&chi, &format!("{}/chi1_rust.raw", outdir));
        eprintln!("chi1: min={} max={} norm={}", fmin(&chi), fmax(&chi), fnorm(&chi));

        // ===== ITERATION 2 =====
        eprintln!("\n=== Iteration 2 ===");

        // P weights
        fgrad_periodic_inplace_f32(
            &mut ws.gx, &mut ws.gy, &mut ws.gz,
            &chi, nx, ny, nz, vsx_f32, vsy_f32, vsz_f32,
        );
        compute_p_weights_f32(&mut vr, &w_gx, &w_gy, &w_gz, &ws.gx, &ws.gy, &ws.gz, beta);
        save_f32_raw(&vr, &format!("{}/P2_rust.raw", outdir));
        eprintln!("P2: min={} max={} mean={}", fmin(&vr), fmax(&vr), fmean(&vr));

        // w = m * exp(i * D*chi)
        apply_dipole_conv(&mut ws.fft_ws, &chi, &d_kernel, &mut ws.dipole_buf, &mut ws.complex_buf);
        for i in 0..n_total {
            let phase = Complex32::new(0.0, ws.dipole_buf[i]);
            w[i] = m[i] * phase.exp();
        }

        // RHS
        compute_rhs_inplace(
            &chi, &w, &b0, &d_kernel,
            &w_gx, &w_gy, &w_gz, &vr, lambda,
            &mut rhs, &mut ws,
        );
        save_f32_raw(&rhs, &format!("{}/rhs2_rust.raw", outdir));
        eprintln!("RHS2: min={} max={} norm={}", fmin(&rhs), fmax(&rhs), fnorm(&rhs));

        negate_f32(&mut rhs);

        // CG solve (no stall detection)
        {
            let n = ws.n_total;
            let (nx, ny, nz) = (ws.nx, ws.ny, ws.nz);
            let (vsx, vsy, vsz) = (ws.vsx, ws.vsy, ws.vsz);

            dx.fill(0.0);
            ws.cg_r.copy_from_slice(&rhs);
            ws.cg_p.copy_from_slice(&ws.cg_r);
            let mut rsold: f32 = norm_squared_f32(&ws.cg_r);
            let b_norm: f32 = norm_squared_f32(&rhs).sqrt();
            let mut p_copy = vec![0.0f32; n];

            for cg_iter in 0..cg_max_iter {
                p_copy.copy_from_slice(&ws.cg_p);
                {
                    let mut bufs = MediOpBuffers {
                        gx: &mut ws.gx, gy: &mut ws.gy, gz: &mut ws.gz,
                        reg_x: &mut ws.reg_x, reg_y: &mut ws.reg_y, reg_z: &mut ws.reg_z,
                        div_buf: &mut ws.div_buf, dipole_buf: &mut ws.dipole_buf,
                        complex_buf: &mut ws.complex_buf, complex_buf2: &mut ws.complex_buf2,
                    };
                    apply_medi_operator_core(
                        &mut ws.fft_ws, &mut bufs, n, nx, ny, nz, vsx, vsy, vsz,
                        &p_copy, &w, &d_kernel, &w_gx, &w_gy, &w_gz, &vr, lambda, &mut ws.cg_ap,
                    );
                }
                let pap: f32 = dot_product_f32(&ws.cg_p, &ws.cg_ap);
                if pap.abs() < 1e-15 { break; }
                let alpha = rsold / pap;
                axpy_f32(&mut dx, alpha, &ws.cg_p);
                axpy_f32(&mut ws.cg_r, -alpha, &ws.cg_ap);
                let rsnew: f32 = norm_squared_f32(&ws.cg_r);
                let residual = rsnew.sqrt();
                eprintln!("  CG iter {}: res={:.6e}", cg_iter + 1, residual);
                if residual < cg_tol * b_norm { break; }
                let beta_cg = rsnew / rsold;
                xpby_f32(&mut ws.cg_p, &ws.cg_r, beta_cg);
                rsold = rsnew;
            }
        }
        save_f32_raw(&dx, &format!("{}/dx2_rust.raw", outdir));
        eprintln!("dx2: min={} max={} norm={}", fmin(&dx), fmax(&dx), fnorm(&dx));

        axpy_f32(&mut chi, 1.0, &dx);
        save_f32_raw(&chi, &format!("{}/chi2_rust.raw", outdir));
        eprintln!("chi2: min={} max={} norm={}", fmin(&chi), fmax(&chi), fnorm(&chi));

        eprintln!("\nDone. Intermediates saved to {}", outdir);
    }

    fn save_f32_raw(data: &[f32], path: &str) {
        use std::io::Write;
        let mut file = std::fs::File::create(path).unwrap();
        for &val in data {
            file.write_all(&val.to_le_bytes()).unwrap();
        }
    }

    fn fmin(data: &[f32]) -> f32 { data.iter().cloned().fold(f32::MAX, f32::min) }
    fn fmax(data: &[f32]) -> f32 { data.iter().cloned().fold(f32::MIN, f32::max) }
    fn fmean(data: &[f32]) -> f32 { data.iter().sum::<f32>() / data.len() as f32 }
    fn fnorm(data: &[f32]) -> f32 { data.iter().map(|&v| v * v).sum::<f32>().sqrt() }
}

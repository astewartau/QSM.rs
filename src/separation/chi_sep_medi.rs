//! Chi-separation using MEDI-based coupled optimization
//!
//! Separates total susceptibility into paramagnetic (chi+) and diamagnetic (chi-)
//! components using chi_pos + chi_neg formulation:
//!   chi_pos >= 0 (paramagnetic, iron), in Hz internally
//!   chi_neg <= 0 (diamagnetic, myelin), in Hz internally
//!   chi_total = chi_pos + chi_neg
//!
//! Forward model (all in Hz):
//!   field = D * (chi_pos + chi_neg)
//!   R2'(Hz) = dr_p_eff * chi_pos + dr_q_eff * (-chi_neg)
//!           = dr_p_eff * |chi_pos| + dr_q_eff * |chi_neg|
//!   where dr_eff = ppm_factor * Dr (dimensionless effective relaxivity)
//!
//! The constraints (chi_pos >= 0, chi_neg <= 0) naturally break the gauge
//! freedom of the chi_pos + chi_neg formulation. In most voxels, either
//! chi_pos = 0 or chi_neg = 0, which pins one variable to its constraint
//! boundary and prevents correlated drift.
//!
//! Reference:
//! Shin, H., et al. (2021). "chi-separation: Magnetic susceptibility source
//! separation toward iron and myelin mapping in the brain." NeuroImage, 240:118371.

use num_complex::Complex32;
use crate::fft::Fft3dWorkspaceF32;
use crate::kernels::dipole::dipole_kernel_f32;
use crate::inversion::medi::{
    gradient_mask_f32,
    fgrad_periodic_inplace_f32,
    bdiv_periodic_inplace_f32,
};
use crate::utils::simd_ops::{
    dot_product_f32, norm_squared_f32, axpy_f32, xpby_f32,
    apply_gradient_weights_f32, compute_p_weights_f32,
};

/// Workspace for chi-separation — holds all reusable buffers (f32).
struct ChiSepWorkspace {
    n: usize,
    nx: usize, ny: usize, nz: usize,
    vsx: f32, vsy: f32, vsz: f32,

    fft_ws: Fft3dWorkspaceF32,

    gx: Vec<f32>,
    gy: Vec<f32>,
    gz: Vec<f32>,

    reg_x: Vec<f32>,
    reg_y: Vec<f32>,
    reg_z: Vec<f32>,

    div_buf: Vec<f32>,

    complex_buf: Vec<Complex32>,
    dipole_buf: Vec<f32>,

    tmp: Vec<f32>,
}

impl ChiSepWorkspace {
    fn new(nx: usize, ny: usize, nz: usize, vsx: f32, vsy: f32, vsz: f32) -> Self {
        let n = nx * ny * nz;
        Self {
            n, nx, ny, nz, vsx, vsy, vsz,
            fft_ws: Fft3dWorkspaceF32::new(nx, ny, nz),
            gx: vec![0.0; n],
            gy: vec![0.0; n],
            gz: vec![0.0; n],
            reg_x: vec![0.0; n],
            reg_y: vec![0.0; n],
            reg_z: vec![0.0; n],
            div_buf: vec![0.0; n],
            complex_buf: vec![Complex32::new(0.0, 0.0); n],
            dipole_buf: vec![0.0; n],
            tmp: vec![0.0; n],
        }
    }
}

/// Chi-separation using MEDI-based coupled optimization.
///
/// # Arguments
/// * `local_field` - Local field map in Hz
/// * `r2prime` - R2' map in Hz
/// * `magnitude` - Magnitude image for edge weighting
/// * `mask` - Binary brain mask, 1 = brain
/// * `cf` - Central frequency in Hz (e.g. 123.2e6 for 3T)
/// * `dr_pos` - Paramagnetic relaxivity in Hz/ppm (default: 114.0)
/// * `dr_neg` - Diamagnetic relaxivity in Hz/ppm (default: 30.0)
///
/// # Returns
/// `(chi_pos, chi_neg, chi_total)` — susceptibility maps in ppm
#[allow(clippy::too_many_arguments)]
pub fn chi_sep_medi(
    local_field: &[f64],
    r2prime: &[f64],
    magnitude: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bdir: (f64, f64, f64),
    cf: f64,
    lambda_para: f64,
    lambda_dia: f64,
    lambda_cpl: f64,
    dr_pos: f64,
    dr_neg: f64,
    percentage: f64,
    cg_tol: f64,
    cg_max_iter: usize,
    max_iter: usize,
    tol: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    chi_sep_medi_with_progress(
        local_field, r2prime, magnitude, mask,
        nx, ny, nz, vsx, vsy, vsz, bdir, cf,
        lambda_para, lambda_dia, lambda_cpl,
        dr_pos, dr_neg, percentage,
        cg_tol, cg_max_iter, max_iter, tol,
        |_, _| {},
    )
}

/// Chi-separation with progress callback.
#[allow(clippy::too_many_arguments)]
pub fn chi_sep_medi_with_progress<F>(
    local_field: &[f64],
    r2prime: &[f64],
    magnitude: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bdir: (f64, f64, f64),
    cf: f64,
    lambda_para: f64,
    lambda_dia: f64,
    lambda_cpl: f64,
    dr_pos: f64,
    dr_neg: f64,
    percentage: f64,
    cg_tol: f64,
    cg_max_iter: usize,
    max_iter: usize,
    tol: f64,
    mut progress_callback: F,
) -> (Vec<f64>, Vec<f64>, Vec<f64>)
where
    F: FnMut(usize, usize),
{
    let n = nx * ny * nz;
    let ppm_factor = (1.0e6 / cf) as f32;

    let vsx_f32 = vsx as f32;
    let vsy_f32 = vsy as f32;
    let vsz_f32 = vsz as f32;
    let bdir_f32 = (bdir.0 as f32, bdir.1 as f32, bdir.2 as f32);
    let lambda_para_f32 = lambda_para as f32;
    let lambda_dia_f32 = lambda_dia as f32;
    let lambda_cpl_f32 = lambda_cpl as f32;
    let cg_tol_f32 = cg_tol as f32;
    let tol_f32 = tol as f32;

    // Effective relaxivities: Dr(Hz/ppm) * ppm_factor(ppm/Hz) = dimensionless
    // R2'(Hz) = Dr_pos * chi_ppm = Dr_pos * (chi_Hz * ppm_factor) = dr_p_eff * chi_Hz
    let dr_p_eff = ppm_factor * dr_pos as f32;
    let dr_q_eff = ppm_factor * dr_neg as f32;

    // Auto-tune R2' normalization for field-strength-independent gauge breaking.
    //
    // The chi-sep gauge mode (χ+ grows, χ- shrinks equally) lives in the null space
    // of the field Hessian and is ONLY constrained by R2'. The gauge eigenvalue is
    // λ_cpl * (dr_p + dr_q)² which at 7T is only ~23 vs ~1000 for TV modes.
    //
    // We scale the R2' equation (both data and relaxivities) by r2_scale so that the
    // gauge eigenvalue = max(λ_para, λ_dia), matching the TV regularization strength.
    // This doesn't change the solution (R2' residual is still 0 at truth).
    //
    // r2_scale = sqrt(target / (λ_cpl * dr_sum²))
    // → eigenvalue = λ_cpl * (r2_scale * dr_sum)² = target
    let dr_sum = dr_p_eff + dr_q_eff;
    let target_eig = 10.0 * lambda_para_f32.max(lambda_dia_f32);
    let r2_scale = (target_eig / (lambda_cpl_f32 * dr_sum * dr_sum)).sqrt();
    let dr_p_use = dr_p_eff * r2_scale;
    let dr_q_use = dr_q_eff * r2_scale;

    let field_f32: Vec<f32> = local_field.iter()
        .zip(mask.iter())
        .map(|(&v, &m)| if m != 0 { v as f32 } else { 0.0 })
        .collect();
    let r2p_f32: Vec<f32> = r2prime.iter()
        .zip(mask.iter())
        .map(|(&v, &m)| if m != 0 { (v as f32) * r2_scale } else { 0.0 })
        .collect();
    let mag_f32: Vec<f32> = magnitude.iter().map(|&v| v as f32).collect();

    let mut ws = ChiSepWorkspace::new(nx, ny, nz, vsx_f32, vsy_f32, vsz_f32);
    let d_kernel = dipole_kernel_f32(nx, ny, nz, vsx_f32, vsy_f32, vsz_f32, bdir_f32);
    let (mx, my, mz) = gradient_mask_f32(
        &mag_f32, mask, nx, ny, nz, vsx_f32, vsy_f32, vsz_f32, percentage as f32,
    );

    // chi_pos >= 0 (paramagnetic), chi_neg <= 0 (diamagnetic), both in Hz
    let mut chi_pos = vec![0.0f32; n];
    let mut chi_neg = vec![0.0f32; n];

    let mut vr_pos = vec![0.0f32; n];
    let mut vr_neg = vec![0.0f32; n];
    let mut vr_sum = vec![0.0f32; n];
    let n2 = 2 * n;
    let mut dx = vec![0.0f32; n2];
    let mut rhs = vec![0.0f32; n2];
    let mut chi_sum_buf = vec![0.0f32; n];

    // TV weight on (chi+ + chi-) sum — paper uses 2*lambda for sum term
    // Disabled for now (0.0) pending parameter tuning; the sum TV can over-couple components
    let lambda_sum_f32 = 0.0_f32;

    let eps = 1.0e-6_f32;

    for iter in 0..max_iter {
        progress_callback(iter + 1, max_iter);

        // --- TV reweighting ---
        fgrad_periodic_inplace_f32(&mut ws.gx, &mut ws.gy, &mut ws.gz,
            &chi_pos, nx, ny, nz, vsx_f32, vsy_f32, vsz_f32);
        compute_p_weights_f32(&mut vr_pos, &mx, &my, &mz, &ws.gx, &ws.gy, &ws.gz, eps);

        fgrad_periodic_inplace_f32(&mut ws.gx, &mut ws.gy, &mut ws.gz,
            &chi_neg, nx, ny, nz, vsx_f32, vsy_f32, vsz_f32);
        compute_p_weights_f32(&mut vr_neg, &mx, &my, &mz, &ws.gx, &ws.gy, &ws.gz, eps);

        // TV weights for the sum (chi+ + chi-)
        for i in 0..n {
            chi_sum_buf[i] = chi_pos[i] + chi_neg[i];
        }
        fgrad_periodic_inplace_f32(&mut ws.gx, &mut ws.gy, &mut ws.gz,
            &chi_sum_buf, nx, ny, nz, vsx_f32, vsy_f32, vsz_f32);
        compute_p_weights_f32(&mut vr_sum, &mx, &my, &mz, &ws.gx, &ws.gy, &ws.gz, eps);

        // --- Residuals ---
        // field_residual = field - D*(chi_pos + chi_neg)
        for i in 0..n {
            ws.tmp[i] = chi_pos[i] + chi_neg[i];
        }
        let chi_sum = ws.tmp.clone();
        ws.fft_ws.apply_dipole_inplace(&chi_sum, &d_kernel, &mut ws.dipole_buf, &mut ws.complex_buf);
        let field_residual: Vec<f32> = field_f32.iter()
            .zip(ws.dipole_buf.iter())
            .map(|(&f, &d)| f - d)
            .collect();

        // R2' residual (normalized): r2_norm - dr_p_use * chi_pos + dr_q_use * chi_neg
        // where r2_norm = R2' / (dr_p_eff + dr_q_eff), dr_p_use = dr_p_eff / dr_sum
        let r2_residual: Vec<f32> = (0..n).map(|i| {
            r2p_f32[i] - dr_p_use * chi_pos[i] + dr_q_use * chi_neg[i]
        }).collect();

        // --- Build gradient (MEDI convention: b_orig = gradient, then negate) ---
        //
        // Gradient derived from first principles:
        //   J = λ_para*TV(χ+) + λ_dia*TV(χ-) + λ_cpl/2*||field_res||² + λ_cpl/2*||r2_res||²
        //
        // ∂J/∂χ+ = λ_para*TV_grad(χ+) - λ_cpl*D(field_res) - λ_cpl*r2_res*dr_p_eff
        // ∂J/∂χ- = λ_dia*TV_grad(χ-)  - λ_cpl*D(field_res) + λ_cpl*r2_res*dr_q_eff
        //
        // TV_grad(χ) = bdiv(wG*Vr*wG*fgrad(χ)) [MEDI convention, IS the gradient]
        //   because bdiv = -div, so bdiv(wG*Vr*wG*∇χ) = -div(wG*Vr*wG*∇χ) = ∂TV/∂χ
        //
        // Field: ∂||field_res||²/∂χ+ = -2*D(field_res), with 1/2 → -D(field_res)
        //   Same for χ- since chi_total = χ+ + χ-, ∂/∂χ- has same sign
        //
        // R2': r2_res = R2' - dr_p*χ+ + dr_q*χ- (since |χ-| = -χ-)
        //   ∂r2_res/∂χ+ = -dr_p → ∂||r2_res||²/∂χ+ = -2*r2_res*dr_p, with 1/2 → -r2_res*dr_p
        //   ∂r2_res/∂χ- = +dr_q → ∂||r2_res||²/∂χ- = +2*r2_res*dr_q, with 1/2 → +r2_res*dr_q

        // TV gradient for chi_pos
        fgrad_periodic_inplace_f32(&mut ws.gx, &mut ws.gy, &mut ws.gz,
            &chi_pos, nx, ny, nz, vsx_f32, vsy_f32, vsz_f32);
        apply_gradient_weights_f32(&mut ws.reg_x, &mut ws.reg_y, &mut ws.reg_z,
            &mx, &my, &mz, &vr_pos, &ws.gx, &ws.gy, &ws.gz);
        bdiv_periodic_inplace_f32(&mut ws.div_buf,
            &ws.reg_x, &ws.reg_y, &ws.reg_z, nx, ny, nz, vsx_f32, vsy_f32, vsz_f32);
        for i in 0..n {
            rhs[i] = lambda_para_f32 * ws.div_buf[i];
        }

        // TV gradient for chi_neg
        fgrad_periodic_inplace_f32(&mut ws.gx, &mut ws.gy, &mut ws.gz,
            &chi_neg, nx, ny, nz, vsx_f32, vsy_f32, vsz_f32);
        apply_gradient_weights_f32(&mut ws.reg_x, &mut ws.reg_y, &mut ws.reg_z,
            &mx, &my, &mz, &vr_neg, &ws.gx, &ws.gy, &ws.gz);
        bdiv_periodic_inplace_f32(&mut ws.div_buf,
            &ws.reg_x, &ws.reg_y, &ws.reg_z, nx, ny, nz, vsx_f32, vsy_f32, vsz_f32);
        for i in 0..n {
            rhs[n + i] = lambda_dia_f32 * ws.div_buf[i];
        }

        // TV gradient for (chi+ + chi-) sum — applied equally to both components
        for i in 0..n {
            chi_sum_buf[i] = chi_pos[i] + chi_neg[i];
        }
        fgrad_periodic_inplace_f32(&mut ws.gx, &mut ws.gy, &mut ws.gz,
            &chi_sum_buf, nx, ny, nz, vsx_f32, vsy_f32, vsz_f32);
        apply_gradient_weights_f32(&mut ws.reg_x, &mut ws.reg_y, &mut ws.reg_z,
            &mx, &my, &mz, &vr_sum, &ws.gx, &ws.gy, &ws.gz);
        bdiv_periodic_inplace_f32(&mut ws.div_buf,
            &ws.reg_x, &ws.reg_y, &ws.reg_z, nx, ny, nz, vsx_f32, vsy_f32, vsz_f32);
        for i in 0..n {
            let tv_sum = lambda_sum_f32 * ws.div_buf[i];
            rhs[i] += tv_sum;
            rhs[n + i] += tv_sum;
        }

        // Field fidelity gradient: -λ_cpl * D(field_res), SAME sign for both
        ws.fft_ws.apply_dipole_inplace(&field_residual, &d_kernel,
            &mut ws.dipole_buf, &mut ws.complex_buf);
        for i in 0..n {
            let fg = lambda_cpl_f32 * ws.dipole_buf[i];
            rhs[i] -= fg;
            rhs[n + i] -= fg;
        }

        // R2' fidelity gradient (normalized):
        //   ∂J/∂χ+ contribution: -λ_cpl * r2_res * dr_p_use
        //   ∂J/∂χ- contribution: +λ_cpl * r2_res * dr_q_use
        for i in 0..n {
            if mask[i] == 0 { continue; }
            rhs[i] -= lambda_cpl_f32 * r2_residual[i] * dr_p_use;
            rhs[n + i] += lambda_cpl_f32 * r2_residual[i] * dr_q_use;
        }

        // Negate for CG: b = -gradient
        for v in rhs.iter_mut() {
            *v = -*v;
        }

        // --- CG solve ---
        cg_solve_chisep(
            &mut ws, &d_kernel,
            &mx, &my, &mz,
            &vr_pos, &vr_neg, &vr_sum,
            lambda_para_f32, lambda_dia_f32, lambda_sum_f32, lambda_cpl_f32,
            dr_p_use, dr_q_use,
            mask,
            &rhs, &mut dx,
            cg_tol_f32, cg_max_iter,
        );

        // --- Update (half Newton step for stability with sign constraints) ---
        for i in 0..n {
            chi_pos[i] += 0.5 * dx[i];
            chi_neg[i] += 0.5 * dx[n + i];
        }

        // --- Enforce constraints: chi_pos >= 0, chi_neg <= 0 ---
        for i in 0..n {
            if mask[i] == 0 {
                chi_pos[i] = 0.0;
                chi_neg[i] = 0.0;
            } else {
                chi_pos[i] = chi_pos[i].max(0.0);
                chi_neg[i] = chi_neg[i].min(0.0);
            }
        }

        // --- Convergence check ---
        let update_norm = norm_squared_f32(&dx).sqrt();
        let sol_norm = (norm_squared_f32(&chi_pos) + norm_squared_f32(&chi_neg)).sqrt();
        let ratio = update_norm / (sol_norm + 1e-6);

        if ratio < tol_f32 {
            break;
        }
    }

    // Convert Hz -> ppm
    let chi_pos_out: Vec<f64> = chi_pos.iter()
        .zip(mask.iter())
        .map(|(&v, &m)| if m == 0 { 0.0 } else { (v * ppm_factor) as f64 })
        .collect();
    let chi_neg_out: Vec<f64> = chi_neg.iter()
        .zip(mask.iter())
        .map(|(&v, &m)| if m == 0 { 0.0 } else { (v * ppm_factor) as f64 })
        .collect();
    let chi_total: Vec<f64> = chi_pos_out.iter()
        .zip(chi_neg_out.iter())
        .map(|(&p, &n)| p + n)
        .collect();

    (chi_pos_out, chi_neg_out, chi_total)
}

/// Apply chi-sep Hessian operator A to doubled vector dx = [d_pos; d_neg].
///
/// A_pos = λ_para * TV_hess(d_pos) + λ_sum * TV_hess_sum(d_pos+d_neg)
///       + λ_cpl * D²(d_pos + d_neg) + λ_cpl * dr_p * (dr_p * d_pos - dr_q * d_neg)
/// A_neg = λ_dia * TV_hess(d_neg) + λ_sum * TV_hess_sum(d_pos+d_neg)
///       + λ_cpl * D²(d_pos + d_neg) - λ_cpl * dr_q * (dr_p * d_pos - dr_q * d_neg)
///
/// TV_hessian uses IRLS: bdiv(wG*Vr*wG*fgrad(.)), positive semi-definite.
/// Field fidelity: D², same for both, positive semi-definite.
/// R2': rank-1 structure [dr_p, -dr_q]^T * [dr_p, -dr_q], positive semi-definite.
#[allow(clippy::too_many_arguments)]
fn apply_chisep_operator(
    ws: &mut ChiSepWorkspace,
    d_kernel: &[f32],
    mx: &[f32], my: &[f32], mz: &[f32],
    vr_pos: &[f32], vr_neg: &[f32], vr_sum: &[f32],
    lambda_para: f32, lambda_dia: f32, lambda_sum: f32, lambda_cpl: f32,
    dr_p: f32, dr_q: f32,
    mask: &[u8],
    dx: &[f32],
    out: &mut [f32],
) {
    let n = ws.n;
    let (nx, ny, nz) = (ws.nx, ws.ny, ws.nz);
    let (vsx, vsy, vsz) = (ws.vsx, ws.vsy, ws.vsz);

    let d_pos = &dx[..n];
    let d_neg = &dx[n..];

    // TV for chi_pos: λ_para * bdiv(wG*Vr_pos*wG*fgrad(d_pos))
    fgrad_periodic_inplace_f32(&mut ws.gx, &mut ws.gy, &mut ws.gz,
        d_pos, nx, ny, nz, vsx, vsy, vsz);
    apply_gradient_weights_f32(&mut ws.reg_x, &mut ws.reg_y, &mut ws.reg_z,
        mx, my, mz, vr_pos, &ws.gx, &ws.gy, &ws.gz);
    bdiv_periodic_inplace_f32(&mut ws.div_buf,
        &ws.reg_x, &ws.reg_y, &ws.reg_z, nx, ny, nz, vsx, vsy, vsz);
    for i in 0..n {
        out[i] = lambda_para * ws.div_buf[i];
    }

    // TV for chi_neg: λ_dia * bdiv(wG*Vr_neg*wG*fgrad(d_neg))
    fgrad_periodic_inplace_f32(&mut ws.gx, &mut ws.gy, &mut ws.gz,
        d_neg, nx, ny, nz, vsx, vsy, vsz);
    apply_gradient_weights_f32(&mut ws.reg_x, &mut ws.reg_y, &mut ws.reg_z,
        mx, my, mz, vr_neg, &ws.gx, &ws.gy, &ws.gz);
    bdiv_periodic_inplace_f32(&mut ws.div_buf,
        &ws.reg_x, &ws.reg_y, &ws.reg_z, nx, ny, nz, vsx, vsy, vsz);
    for i in 0..n {
        out[n + i] = lambda_dia * ws.div_buf[i];
    }

    // TV for sum (d_pos + d_neg): λ_sum * bdiv(wG*Vr_sum*wG*fgrad(d_pos+d_neg))
    // Applied equally to both components
    for i in 0..n {
        ws.tmp[i] = d_pos[i] + d_neg[i];
    }
    let sum_for_tv = ws.tmp.clone();
    fgrad_periodic_inplace_f32(&mut ws.gx, &mut ws.gy, &mut ws.gz,
        &sum_for_tv, nx, ny, nz, vsx, vsy, vsz);
    apply_gradient_weights_f32(&mut ws.reg_x, &mut ws.reg_y, &mut ws.reg_z,
        mx, my, mz, vr_sum, &ws.gx, &ws.gy, &ws.gz);
    bdiv_periodic_inplace_f32(&mut ws.div_buf,
        &ws.reg_x, &ws.reg_y, &ws.reg_z, nx, ny, nz, vsx, vsy, vsz);
    for i in 0..n {
        let tv_s = lambda_sum * ws.div_buf[i];
        out[i] += tv_s;
        out[n + i] += tv_s;
    }

    // Field fidelity: λ_cpl * D²(d_pos + d_neg), SAME for both
    for i in 0..n {
        ws.tmp[i] = d_pos[i] + d_neg[i];
    }
    let sum_copy = ws.tmp.clone();
    ws.fft_ws.apply_dipole_inplace(&sum_copy, d_kernel, &mut ws.dipole_buf, &mut ws.complex_buf);
    let d1_copy = ws.dipole_buf.clone();
    ws.fft_ws.apply_dipole_inplace(&d1_copy, d_kernel, &mut ws.dipole_buf, &mut ws.complex_buf);

    for i in 0..n {
        let ff = lambda_cpl * ws.dipole_buf[i];
        out[i] += ff;
        out[n + i] += ff;
    }

    // R2' fidelity: rank-1 Hessian [dr_p, -dr_q]^T * [dr_p, -dr_q]
    // r2_lin = dr_p * d_pos - dr_q * d_neg
    // out_pos += λ_cpl * dr_p * r2_lin
    // out_neg -= λ_cpl * dr_q * r2_lin
    for i in 0..n {
        if mask[i] == 0 { continue; }
        let r2_lin = dr_p * d_pos[i] - dr_q * d_neg[i];
        out[i] += lambda_cpl * dr_p * r2_lin;
        out[n + i] -= lambda_cpl * dr_q * r2_lin;
    }
}

/// CG solver for the doubled chi-sep system.
#[allow(clippy::too_many_arguments)]
fn cg_solve_chisep(
    ws: &mut ChiSepWorkspace,
    d_kernel: &[f32],
    mx: &[f32], my: &[f32], mz: &[f32],
    vr_pos: &[f32], vr_neg: &[f32], vr_sum: &[f32],
    lambda_para: f32, lambda_dia: f32, lambda_sum: f32, lambda_cpl: f32,
    dr_p: f32, dr_q: f32,
    mask: &[u8],
    b: &[f32],
    x: &mut [f32],
    tol: f32,
    max_iter: usize,
) {
    let n2 = 2 * ws.n;
    x.fill(0.0);

    let mut cg_r = vec![0.0f32; n2];
    let mut cg_p = vec![0.0f32; n2];
    let mut cg_ap = vec![0.0f32; n2];

    cg_r.copy_from_slice(&b[..n2]);
    cg_p.copy_from_slice(&cg_r);

    let mut rsold = dot_product_f32(&cg_r, &cg_r);
    let b_norm = dot_product_f32(b, b).sqrt();

    if b_norm < 1e-10 {
        return;
    }

    for _cg_iter in 0..max_iter {
        apply_chisep_operator(
            ws, d_kernel, mx, my, mz,
            vr_pos, vr_neg, vr_sum,
            lambda_para, lambda_dia, lambda_sum, lambda_cpl,
            dr_p, dr_q,
            mask,
            &cg_p, &mut cg_ap,
        );

        let pap = dot_product_f32(&cg_p, &cg_ap);
        if pap.abs() < 1e-15 {
            break;
        }

        let alpha = rsold / pap;
        axpy_f32(x, alpha, &cg_p);
        axpy_f32(&mut cg_r, -alpha, &cg_ap);

        let rsnew = dot_product_f32(&cg_r, &cg_r);
        if rsnew.sqrt() < tol * b_norm {
            break;
        }

        let beta_cg = rsnew / rsold;
        xpby_f32(&mut cg_p, &cg_r, beta_cg);
        rsold = rsnew;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::dipole::dipole_kernel;
    use crate::fft::{fft3d_real, ifft3d_real};

    fn make_sphere(nx: usize, ny: usize, nz: usize, cx: f64, cy: f64, cz: f64, r: f64) -> Vec<f64> {
        let mut vol = vec![0.0; nx * ny * nz];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let dx = i as f64 - cx;
                    let dy = j as f64 - cy;
                    let dz = k as f64 - cz;
                    if dx * dx + dy * dy + dz * dz <= r * r {
                        vol[i + j * nx + k * nx * ny] = 1.0;
                    }
                }
            }
        }
        vol
    }

    #[test]
    fn test_chi_sep_medi_basic() {
        let (nx, ny, nz) = (32, 32, 32);
        let n = nx * ny * nz;
        let (vsx, vsy, vsz) = (1.0, 1.0, 1.0);
        let bdir = (0.0, 0.0, 1.0);
        let cf: f64 = 123.2e6; // 3T

        let chi_pos_true_ppm = 0.05;
        let chi_neg_true_ppm = -0.03;

        let sphere_inner = make_sphere(nx, ny, nz, 16.0, 16.0, 16.0, 4.0);
        let sphere_outer = make_sphere(nx, ny, nz, 16.0, 16.0, 16.0, 8.0);
        let brain_mask = make_sphere(nx, ny, nz, 16.0, 16.0, 16.0, 12.0);

        let hz_per_ppm = cf / 1.0e6;
        let mut chi_pos_ppm = vec![0.0f64; n];
        let mut chi_neg_ppm = vec![0.0f64; n];
        for i in 0..n {
            if sphere_inner[i] > 0.5 {
                chi_pos_ppm[i] = chi_pos_true_ppm;
            }
            if sphere_outer[i] > 0.5 && sphere_inner[i] < 0.5 {
                chi_neg_ppm[i] = chi_neg_true_ppm;
            }
        }

        // Forward model: field_Hz = D * chi_total_Hz
        let chi_total_hz: Vec<f64> = chi_pos_ppm.iter()
            .zip(chi_neg_ppm.iter())
            .map(|(&p, &n)| (p + n) * hz_per_ppm)
            .collect();
        let d = dipole_kernel(nx, ny, nz, vsx, vsy, vsz, bdir);
        let chi_fft = fft3d_real(&chi_total_hz, nx, ny, nz);
        let field_fft: Vec<_> = chi_fft.iter()
            .zip(d.iter())
            .map(|(&c, &dk)| c * dk)
            .collect();
        let local_field = ifft3d_real(&field_fft, nx, ny, nz);

        // R2'(Hz) = Dr_pos * |chi+_ppm| + Dr_neg * |chi-_ppm|
        let dr_pos: f64 = 114.0;
        let dr_neg: f64 = 30.0;
        let r2prime: Vec<f64> = (0..n).map(|i| {
            dr_pos * chi_pos_ppm[i].abs() + dr_neg * chi_neg_ppm[i].abs()
        }).collect();

        let mask: Vec<u8> = brain_mask.iter()
            .map(|&v| if v > 0.5 { 1 } else { 0 })
            .collect();

        let magnitude: Vec<f64> = (0..n).map(|i| {
            if mask[i] == 0 { return 0.0; }
            let base = 100.0;
            if sphere_inner[i] > 0.5 {
                base * 1.5
            } else if sphere_outer[i] > 0.5 {
                base * 0.7
            } else {
                base
            }
        }).collect();

        let (chi_pos_out, chi_neg_out, chi_total_out) = chi_sep_medi(
            &local_field, &r2prime, &magnitude, &mask,
            nx, ny, nz, vsx, vsy, vsz, bdir, cf,
            1000.0, 1000.0, 100.0,
            dr_pos, dr_neg,
            0.3, 0.01, 100, 10, 0.1,
        );

        // chi+ should be non-negative, chi- non-positive
        for i in 0..n {
            if mask[i] != 0 {
                assert!(chi_pos_out[i] >= -1e-10,
                    "chi+ should be non-negative at voxel {}, got {}", i, chi_pos_out[i]);
                assert!(chi_neg_out[i] <= 1e-10,
                    "chi- should be non-positive at voxel {}, got {}", i, chi_neg_out[i]);
            }
        }

        for i in 0..n {
            let diff = (chi_total_out[i] - chi_pos_out[i] - chi_neg_out[i]).abs();
            assert!(diff < 1e-10, "chi_total != chi+ + chi- at voxel {}", i);
        }

        let pos_max = chi_pos_out.iter().cloned().fold(0.0_f64, f64::max);
        let neg_min = chi_neg_out.iter().cloned().fold(0.0_f64, f64::min);
        assert!(pos_max > 0.0, "chi+ should have positive values, max={}", pos_max);
        assert!(neg_min < 0.0, "chi- should have negative values, min={}", neg_min);
    }
}

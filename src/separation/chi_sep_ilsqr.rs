//! χ-separation via the original Shin 2021 projected-CG algorithm.
//!
//! Faithful implementation of the algorithm in the Supplementary Methods of
//! Shin et al. 2021 (the SNU-LIST toolbox ships it compiled as `chi_sep_iLSQR.p`
//! / `chi_sep_MEDI.p`, so this is a paper-based port). The minimization is
//!
//! ```text
//! argmin_{χ+, χ−} ‖ Wr·{R2' − (Dr,pos·|χ+| + Dr,neg·|χ−|)}
//!                  + i·2π·Wf·{f − Df ∗ (χ+ + χ−)} ‖₂² + reg(χ+, χ−)
//!                  subject to χ+ ≥ 0, χ− ≤ 0
//! ```
//!
//! where `Wf` is an SNR weight from the GRE magnitude and `Wr = Wf/10` where
//! R2' is unreliable (R2' > 30 Hz or < 1 Hz). Because the R2' term is real and
//! the field term imaginary, the complex norm decouples into two independent
//! least-squares terms; with the sign constraints enforced (|χ+| = χ+,
//! |χ−| = −χ−) both are linear in (χ+, χ−). The regularization is edge-masked
//! L1 total variation (MEDI-style, IRLS-linearized here):
//!
//! ```text
//! reg = 2λ1‖M_Mag ∇(χ+ + χ−)‖₁ + λ1‖M_R2' ∇χ+‖₁ + λ1‖M_R2' ∇χ−‖₁
//! ```
//!
//! with `M_Mag` a binary edge mask from the magnitude and `M_R2'` one from the
//! R2' map. The solution is initialized from the voxelwise 2×2 system
//! `{Dr,pos·χ+ − Dr,neg·χ− = R2',  χ+ + χ− = χ_QSM}` using a conventional QSM
//! reconstruction — the toolbox's `chi_sep_iLSQR` name refers to feeding it an
//! iLSQR QSM ([`crate::inversion::ilsqr`]); per the QSM-CI diagnostic the QSM
//! must be reconstructed from the same local field with matching conventions,
//! not supplied externally. Iteration is Gauss-Newton/CG with sign projection
//! each step; it stops when ‖χⁿ⁺¹ − χⁿ‖/‖χⁿ‖ < tol on χ_total or at max_iter.
//!
//! Reference: Shin, H., et al. (2021). "χ-separation: Magnetic susceptibility
//! source separation toward iron and myelin mapping in the brain." NeuroImage,
//! 240:118371 (Supplementary Methods, "Algorithm for χ-separation").

use crate::fft::Fft3dWorkspaceF32;
use crate::inversion::medi::{
    bdiv_periodic_inplace_f32, fgrad_periodic_inplace_f32, gradient_mask_f32,
};
use crate::kernels::dipole::dipole_kernel_f32;
use crate::utils::simd_ops::{
    apply_gradient_weights_f32, axpy_f32, compute_p_weights_f32, dot_product_f32, xpby_f32,
};
use crate::Grid;
use num_complex::Complex32;

const TWO_PI: f32 = std::f32::consts::TAU;

/// Parameters for [`chi_sep_ilsqr`].
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct ChiSepIlsqrParams {
    /// Central frequency in Hz (e.g. 123.2e6 at 3T, 298.0e6 at 7T)
    pub cf: f64,
    /// Paramagnetic relaxometric constant in Hz/ppm (Shin 2021: 137)
    pub dr_pos: f64,
    /// Diamagnetic relaxometric constant in Hz/ppm (Shin 2021: same as dr_pos)
    pub dr_neg: f64,
    /// L1 edge-masked TV weight λ1 (the total-χ term carries 2λ1)
    pub lambda1: f64,
    /// Edge-mask keep fraction for M_Mag / M_R2' (MEDI convention, 0.9)
    pub percentage: f64,
    /// R2' reliability window: Wr = Wf/10 outside [r2p_min, r2p_max] Hz
    pub r2p_min: f64,
    /// Upper bound of the reliable R2' window in Hz
    pub r2p_max: f64,
    /// Outer Gauss-Newton max iterations (paper: 30)
    pub max_iter: usize,
    /// Outer relative-change tolerance on χ_total (paper: 0.01)
    pub tol: f64,
    /// Inner conjugate-gradient max iterations
    pub cg_max_iter: usize,
    /// Inner conjugate-gradient relative tolerance
    pub cg_tol: f64,
}

impl Default for ChiSepIlsqrParams {
    fn default() -> Self {
        Self {
            cf: 123.2e6,
            dr_pos: 137.0,
            dr_neg: 137.0,
            lambda1: 1.0,
            percentage: 0.9,
            r2p_min: 1.0,
            r2p_max: 30.0,
            max_iter: 30,
            tol: 0.01,
            cg_max_iter: 30,
            cg_tol: 0.05,
        }
    }
}

/// Shared buffers for the operator applications (all length n unless noted).
struct Ws {
    n: usize,
    nx: usize,
    ny: usize,
    nz: usize,
    vsx: f32,
    vsy: f32,
    vsz: f32,
    fft_ws: Fft3dWorkspaceF32,
    gx: Vec<f32>,
    gy: Vec<f32>,
    gz: Vec<f32>,
    wx: Vec<f32>,
    wy: Vec<f32>,
    wz: Vec<f32>,
    div: Vec<f32>,
    dip: Vec<f32>,
    cbuf: Vec<Complex32>,
    sum: Vec<f32>,
}

impl Ws {
    fn new(grid: &Grid) -> Self {
        let (nx, ny, nz) = grid.dims;
        let (vsx, vsy, vsz) = grid.voxel_size;
        let n = nx * ny * nz;
        Self {
            n,
            nx,
            ny,
            nz,
            vsx: vsx as f32,
            vsy: vsy as f32,
            vsz: vsz as f32,
            fft_ws: Fft3dWorkspaceF32::new(nx, ny, nz),
            gx: vec![0.0; n],
            gy: vec![0.0; n],
            gz: vec![0.0; n],
            wx: vec![0.0; n],
            wy: vec![0.0; n],
            wz: vec![0.0; n],
            div: vec![0.0; n],
            dip: vec![0.0; n],
            cbuf: vec![Complex32::new(0.0, 0.0); n],
            sum: vec![0.0; n],
        }
    }

    /// out (div) = bdiv( M · Vr · M · ∇x ), the IRLS-linearized TV Hessian apply.
    #[allow(clippy::too_many_arguments)]
    fn tv_apply(&mut self, x: &[f32], mx: &[f32], my: &[f32], mz: &[f32], vr: &[f32]) {
        fgrad_periodic_inplace_f32(
            &mut self.gx, &mut self.gy, &mut self.gz, x, self.nx, self.ny, self.nz, self.vsx,
            self.vsy, self.vsz,
        );
        apply_gradient_weights_f32(
            &mut self.wx, &mut self.wy, &mut self.wz, mx, my, mz, vr, &self.gx, &self.gy, &self.gz,
        );
        bdiv_periodic_inplace_f32(
            &mut self.div, &self.wx, &self.wy, &self.wz, self.nx, self.ny, self.nz, self.vsx,
            self.vsy, self.vsz,
        );
    }

    /// dip = D(k) applied to x (unitless dipole convolution).
    fn dipole_apply(&mut self, x: &[f32], d_kernel: &[f32]) {
        self.fft_ws
            .apply_dipole_inplace(x, d_kernel, &mut self.dip, &mut self.cbuf);
    }
}

/// IRLS weights Vr = 1/sqrt(|M·∇χ|² + eps) for one TV term.
fn irls_weights(ws: &mut Ws, x: &[f32], mx: &[f32], my: &[f32], mz: &[f32], vr: &mut [f32]) {
    let eps = 1.0e-6_f32;
    fgrad_periodic_inplace_f32(
        &mut ws.gx, &mut ws.gy, &mut ws.gz, x, ws.nx, ws.ny, ws.nz, ws.vsx, ws.vsy, ws.vsz,
    );
    compute_p_weights_f32(vr, mx, my, mz, &ws.gx, &ws.gy, &ws.gz, eps);
}

/// χ-separation (Shin 2021, projected Gauss-Newton/CG).
///
/// # Arguments
/// * `local_field` - Local (tissue) field map in Hz `[nx*ny*nz]`
/// * `r2prime` - R2' map in Hz `[nx*ny*nz]`
/// * `magnitude` - GRE magnitude (echo-combined) for the SNR weight and edge mask
/// * `qsm` - Conventional QSM in ppm for initialization (use [`crate::inversion::ilsqr`]
///   on the same local field — see module docs)
/// * `mask` - Binary brain mask `[nx*ny*nz]`
/// * `grid` - Volume grid
/// * `bdir` - B0 direction unit vector
/// * `params` - Algorithm parameters (see [`ChiSepIlsqrParams`])
/// * `progress` - Progress callback `(iteration, max_iterations)`
///
/// # Returns
/// `(chi_pos, chi_neg, chi_total)` in ppm; `chi_neg` is ≤ 0.
#[allow(clippy::too_many_arguments)]
pub fn chi_sep_ilsqr<F>(
    local_field: &[f64],
    r2prime: &[f64],
    magnitude: &[f64],
    qsm: &[f64],
    mask: &[u8],
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &ChiSepIlsqrParams,
    mut progress: F,
) -> (Vec<f64>, Vec<f64>, Vec<f64>)
where
    F: FnMut(usize, usize),
{
    let (nx, ny, nz) = grid.dims;
    let n = nx * ny * nz;
    assert_eq!(local_field.len(), n, "local_field length must match grid");
    assert_eq!(r2prime.len(), n, "r2prime length must match grid");
    assert_eq!(magnitude.len(), n, "magnitude length must match grid");
    assert_eq!(qsm.len(), n, "qsm length must match grid");
    assert_eq!(mask.len(), n, "mask length must match grid");

    let cf_ppm = (params.cf * 1.0e-6) as f32; // Hz per ppm
    let dr_p = params.dr_pos as f32;
    let dr_q = params.dr_neg as f32;
    let lambda1 = params.lambda1 as f32;
    let tol = params.tol as f32;

    let mut ws = Ws::new(grid);
    let d_kernel = dipole_kernel_f32(grid, (bdir.0 as f32, bdir.1 as f32, bdir.2 as f32));

    // --- SNR weights: Wf = magnitude / mean(magnitude in mask), 0 outside ---
    let mut mag_mean = 0.0_f64;
    let mut n_mask = 0usize;
    for i in 0..n {
        if mask[i] != 0 {
            mag_mean += magnitude[i];
            n_mask += 1;
        }
    }
    assert!(n_mask > 0, "mask is empty");
    mag_mean /= n_mask as f64;
    let wf2: Vec<f32> = (0..n)
        .map(|i| {
            if mask[i] == 0 {
                0.0
            } else {
                let w = (magnitude[i] / mag_mean) as f32;
                w * w
            }
        })
        .collect();
    // Wr = Wf/10 where R2' is unreliable → Wr² = Wf²/100 there.
    let wr2: Vec<f32> = (0..n)
        .map(|i| {
            if mask[i] == 0 {
                0.0
            } else if r2prime[i] > params.r2p_max || r2prime[i] < params.r2p_min {
                wf2[i] / 100.0
            } else {
                wf2[i]
            }
        })
        .collect();

    // --- Edge masks: M_Mag from magnitude, M_R2' from the R2' map ---
    let mag_f32: Vec<f32> = magnitude.iter().map(|&v| v as f32).collect();
    let r2p_f32: Vec<f32> = r2prime
        .iter()
        .zip(mask.iter())
        .map(|(&v, &m)| if m != 0 { v as f32 } else { 0.0 })
        .collect();
    let pct = params.percentage as f32;
    let (vsx, vsy, vsz) = (ws.vsx, ws.vsy, ws.vsz);
    let (mmx, mmy, mmz) = gradient_mask_f32(&mag_f32, mask, nx, ny, nz, vsx, vsy, vsz, pct);
    let (mrx, mry, mrz) = gradient_mask_f32(&r2p_f32, mask, nx, ny, nz, vsx, vsy, vsz, pct);

    let field_f32: Vec<f32> = local_field
        .iter()
        .zip(mask.iter())
        .map(|(&v, &m)| if m != 0 { v as f32 } else { 0.0 })
        .collect();

    // --- Initialization: voxelwise 2×2 solve of
    //   Dr,pos·χ+ − Dr,neg·χ− = R2'   and   χ+ + χ− = χ_QSM,
    // then sign projection (supplementary methods). All in ppm. ---
    let dr_sum = dr_p + dr_q;
    let mut chi_pos = vec![0.0_f32; n];
    let mut chi_neg = vec![0.0_f32; n];
    for i in 0..n {
        if mask[i] == 0 {
            continue;
        }
        let q = qsm[i] as f32;
        let r = r2prime[i] as f32;
        chi_pos[i] = ((dr_q * q + r) / dr_sum).max(0.0);
        chi_neg[i] = ((dr_p * q - r) / dr_sum).min(0.0);
    }

    // Field-term scale: residuals in Hz enter as i·2π·Wf·(·), so the quadratic
    // carries (2π)²; the dipole operator maps ppm → Hz via cf_ppm·D(k).
    let field_w = TWO_PI * TWO_PI;

    let mut vr_tot = vec![0.0_f32; n];
    let mut vr_pos = vec![0.0_f32; n];
    let mut vr_neg = vec![0.0_f32; n];
    let n2 = 2 * n;
    let mut rhs = vec![0.0_f32; n2];
    let mut dx = vec![0.0_f32; n2];
    let mut prev_total = vec![0.0_f32; n];
    for i in 0..n {
        prev_total[i] = chi_pos[i] + chi_neg[i];
    }

    for iter in 0..params.max_iter {
        progress(iter + 1, params.max_iter);

        // --- IRLS reweighting for the three TV terms ---
        for i in 0..n {
            ws.sum[i] = chi_pos[i] + chi_neg[i];
        }
        let total = ws.sum.clone();
        irls_weights(&mut ws, &total, &mmx, &mmy, &mmz, &mut vr_tot);
        irls_weights(&mut ws, &chi_pos, &mrx, &mry, &mrz, &mut vr_pos);
        irls_weights(&mut ws, &chi_neg, &mrx, &mry, &mrz, &mut vr_neg);

        // --- Gradient of the cost (1/2‖·‖² convention), then b = −grad ---
        // Field residual r_f = f − cf_ppm·D(χ+ + χ−)   [Hz]
        ws.dipole_apply(&total, &d_kernel);
        let rf: Vec<f32> = (0..n)
            .map(|i| field_f32[i] - cf_ppm * ws.dip[i])
            .collect();
        // grad_field(both) = −(2π)²·cf_ppm·Dᴴ(Wf²·r_f)
        let wrf: Vec<f32> = (0..n).map(|i| wf2[i] * rf[i]).collect();
        ws.dipole_apply(&wrf, &d_kernel);
        for i in 0..n {
            let g = -field_w * cf_ppm * ws.dip[i];
            rhs[i] = g;
            rhs[n + i] = g;
        }

        // R2' residual r_r = R2' − Dr,pos·χ+ + Dr,neg·χ−   [Hz]
        // grad_pos = −Dr,pos·Wr²·r_r ; grad_neg = +Dr,neg·Wr²·r_r
        for i in 0..n {
            if mask[i] == 0 {
                continue;
            }
            let rr = r2p_f32[i] - dr_p * chi_pos[i] + dr_q * chi_neg[i];
            let wrr = wr2[i] * rr;
            rhs[i] -= dr_p * wrr;
            rhs[n + i] += dr_q * wrr;
        }

        // TV gradients: 2λ1 on the total (both components), λ1 per component.
        ws.tv_apply(&total, &mmx, &mmy, &mmz, &vr_tot);
        for i in 0..n {
            let g = 2.0 * lambda1 * ws.div[i];
            rhs[i] += g;
            rhs[n + i] += g;
        }
        let chi_pos_snapshot = chi_pos.clone();
        ws.tv_apply(&chi_pos_snapshot, &mrx, &mry, &mrz, &vr_pos);
        for i in 0..n {
            rhs[i] += lambda1 * ws.div[i];
        }
        let chi_neg_snapshot = chi_neg.clone();
        ws.tv_apply(&chi_neg_snapshot, &mrx, &mry, &mrz, &vr_neg);
        for i in 0..n {
            rhs[n + i] += lambda1 * ws.div[i];
        }

        for v in rhs.iter_mut() {
            *v = -*v;
        }

        // --- Inner CG on the Gauss-Newton system ---
        cg_solve(
            &mut ws, &d_kernel, &wf2, &wr2, &mmx, &mmy, &mmz, &mrx, &mry, &mrz, &vr_tot, &vr_pos,
            &vr_neg, lambda1, field_w, cf_ppm, dr_p, dr_q, mask, &rhs, &mut dx,
            params.cg_tol as f32, params.cg_max_iter,
        );

        // --- Update + sign projection (violations forced to zero) ---
        for i in 0..n {
            if mask[i] == 0 {
                continue;
            }
            chi_pos[i] = (chi_pos[i] + dx[i]).max(0.0);
            chi_neg[i] = (chi_neg[i] + dx[n + i]).min(0.0);
        }

        // --- Convergence on χ_total (paper: ‖χⁿ⁺¹ − χⁿ‖/‖χⁿ‖ < 0.01) ---
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for i in 0..n {
            let t = chi_pos[i] + chi_neg[i];
            let d = (t - prev_total[i]) as f64;
            num += d * d;
            den += (prev_total[i] as f64) * (prev_total[i] as f64);
            prev_total[i] = t;
        }
        if den > 0.0 && (num / den).sqrt() < tol as f64 {
            break;
        }
    }

    let chi_pos_out: Vec<f64> = chi_pos.iter().map(|&v| v as f64).collect();
    let chi_neg_out: Vec<f64> = chi_neg.iter().map(|&v| v as f64).collect();
    let chi_total: Vec<f64> = (0..n).map(|i| chi_pos_out[i] + chi_neg_out[i]).collect();
    (chi_pos_out, chi_neg_out, chi_total)
}

/// Apply the Gauss-Newton operator to `dx = [d+; d−]`:
///
/// ```text
/// A± = 2λ1·TVtot(d+ + d−) + λ1·TV±(d±) + (2π)²·cf²·Dᴴ Wf² D (d+ + d−)
///      ± Dr,± · Wr² · (Dr,pos·d+ − Dr,neg·d−)
/// ```
#[allow(clippy::too_many_arguments)]
fn apply_operator(
    ws: &mut Ws,
    d_kernel: &[f32],
    wf2: &[f32],
    wr2: &[f32],
    mmx: &[f32],
    mmy: &[f32],
    mmz: &[f32],
    mrx: &[f32],
    mry: &[f32],
    mrz: &[f32],
    vr_tot: &[f32],
    vr_pos: &[f32],
    vr_neg: &[f32],
    lambda1: f32,
    field_w: f32,
    cf_ppm: f32,
    dr_p: f32,
    dr_q: f32,
    mask: &[u8],
    dx: &[f32],
    out: &mut [f32],
) {
    let n = ws.n;
    let (d_pos, d_neg) = dx.split_at(n);

    // Field: (2π)²·cf²·Dᴴ Wf² D applied to the sum, same for both components.
    for i in 0..n {
        ws.sum[i] = d_pos[i] + d_neg[i];
    }
    let dsum = ws.sum.clone();
    ws.dipole_apply(&dsum, d_kernel);
    let wd: Vec<f32> = (0..n).map(|i| wf2[i] * ws.dip[i]).collect();
    ws.dipole_apply(&wd, d_kernel);
    let scale = field_w * cf_ppm * cf_ppm;
    for i in 0..n {
        let f = scale * ws.dip[i];
        out[i] = f;
        out[n + i] = f;
    }

    // Total-TV (2λ1) on the sum, applied to both components.
    ws.tv_apply(&dsum, mmx, mmy, mmz, vr_tot);
    for i in 0..n {
        let g = 2.0 * lambda1 * ws.div[i];
        out[i] += g;
        out[n + i] += g;
    }

    // Component TVs (λ1) with the R2' edge mask.
    ws.tv_apply(d_pos, mrx, mry, mrz, vr_pos);
    for i in 0..n {
        out[i] += lambda1 * ws.div[i];
    }
    ws.tv_apply(d_neg, mrx, mry, mrz, vr_neg);
    for i in 0..n {
        out[n + i] += lambda1 * ws.div[i];
    }

    // R2': rank-1 per voxel, [Dr,pos, −Dr,neg]ᵀ Wr² [Dr,pos, −Dr,neg].
    for i in 0..n {
        if mask[i] == 0 {
            continue;
        }
        let lin = wr2[i] * (dr_p * d_pos[i] - dr_q * d_neg[i]);
        out[i] += dr_p * lin;
        out[n + i] -= dr_q * lin;
    }
}

/// Standard CG on the doubled system.
#[allow(clippy::too_many_arguments)]
fn cg_solve(
    ws: &mut Ws,
    d_kernel: &[f32],
    wf2: &[f32],
    wr2: &[f32],
    mmx: &[f32],
    mmy: &[f32],
    mmz: &[f32],
    mrx: &[f32],
    mry: &[f32],
    mrz: &[f32],
    vr_tot: &[f32],
    vr_pos: &[f32],
    vr_neg: &[f32],
    lambda1: f32,
    field_w: f32,
    cf_ppm: f32,
    dr_p: f32,
    dr_q: f32,
    mask: &[u8],
    b: &[f32],
    x: &mut [f32],
    tol: f32,
    max_iter: usize,
) {
    let n2 = 2 * ws.n;
    x.fill(0.0);
    let mut r = b.to_vec();
    let mut p = r.clone();
    let mut ap = vec![0.0_f32; n2];

    let b_norm = dot_product_f32(b, b).sqrt();
    if b_norm < 1e-12 {
        return;
    }
    let mut rsold = dot_product_f32(&r, &r);

    for _ in 0..max_iter {
        apply_operator(
            ws, d_kernel, wf2, wr2, mmx, mmy, mmz, mrx, mry, mrz, vr_tot, vr_pos, vr_neg, lambda1,
            field_w, cf_ppm, dr_p, dr_q, mask, &p, &mut ap,
        );
        let pap = dot_product_f32(&p, &ap);
        if pap.abs() < 1e-20 {
            break;
        }
        let alpha = rsold / pap;
        axpy_f32(x, alpha, &p);
        axpy_f32(&mut r, -alpha, &ap);
        let rsnew = dot_product_f32(&r, &r);
        if rsnew.sqrt() < tol * b_norm {
            break;
        }
        xpby_f32(&mut p, &r, rsnew / rsold);
        rsold = rsnew;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft::{fft3d_real, ifft3d_real};
    use crate::kernels::dipole::dipole_kernel;

    fn make_sphere(nx: usize, ny: usize, nz: usize, c: f64, r: f64) -> Vec<f64> {
        let mut vol = vec![0.0; nx * ny * nz];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let (dx, dy, dz) = (i as f64 - c, j as f64 - c, k as f64 - c);
                    if dx * dx + dy * dy + dz * dz <= r * r {
                        vol[i + j * nx + k * nx * ny] = 1.0;
                    }
                }
            }
        }
        vol
    }

    /// Two-shell phantom: paramagnetic core, diamagnetic shell. With consistent
    /// field/R2'/QSM inputs the algorithm must recover both components.
    #[test]
    fn test_chi_sep_ilsqr_recovers_two_shell_phantom() {
        let (nx, ny, nz) = (32, 32, 32);
        let n = nx * ny * nz;
        let grid = Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);
        let bdir = (0.0, 0.0, 1.0);
        let cf = 123.2e6_f64;
        let hz_per_ppm = cf * 1.0e-6;

        let inner = make_sphere(nx, ny, nz, 16.0, 4.0);
        let outer = make_sphere(nx, ny, nz, 16.0, 8.0);
        let brain = make_sphere(nx, ny, nz, 16.0, 12.0);

        let mut chi_pos_t = vec![0.0_f64; n];
        let mut chi_neg_t = vec![0.0_f64; n];
        for i in 0..n {
            if inner[i] > 0.5 {
                chi_pos_t[i] = 0.08;
            } else if outer[i] > 0.5 {
                chi_neg_t[i] = -0.05;
            }
        }
        let chi_total_t: Vec<f64> = (0..n).map(|i| chi_pos_t[i] + chi_neg_t[i]).collect();

        // field_Hz = cf_ppm * D(k) * chi_total_ppm
        let d = dipole_kernel(&grid, bdir);
        let cf_fft = fft3d_real(&chi_total_t, nx, ny, nz);
        let f_fft: Vec<_> = cf_fft.iter().zip(d.iter()).map(|(&c, &dk)| c * dk).collect();
        let field_hz: Vec<f64> = ifft3d_real(&f_fft, nx, ny, nz)
            .iter()
            .map(|&v| v * hz_per_ppm)
            .collect();

        let dr = 137.0_f64;
        let r2prime: Vec<f64> = (0..n)
            .map(|i| dr * (chi_pos_t[i].abs() + chi_neg_t[i].abs()))
            .collect();
        let mask: Vec<u8> = brain.iter().map(|&v| (v > 0.5) as u8).collect();
        let magnitude: Vec<f64> = (0..n)
            .map(|i| if mask[i] != 0 { 100.0 } else { 0.0 })
            .collect();

        let params = ChiSepIlsqrParams {
            cf,
            ..ChiSepIlsqrParams::default()
        };
        // Ideal conventional QSM = true chi_total (unit test isolates the separation).
        let (chi_pos, chi_neg, chi_total) = chi_sep_ilsqr(
            &field_hz, &r2prime, &magnitude, &chi_total_t, &mask, &grid, bdir, &params, |_, _| {},
        );

        for i in 0..n {
            assert!(chi_pos[i] >= 0.0, "chi+ must be non-negative");
            assert!(chi_neg[i] <= 0.0, "chi- must be non-positive");
            assert!((chi_total[i] - chi_pos[i] - chi_neg[i]).abs() < 1e-10);
        }

        // Region means should recover the assigned values well: the init is exact
        // here, so the iterations must not walk away from the solution.
        let mean_in = |v: &Vec<f64>, region: &Vec<f64>| {
            let (mut s, mut c) = (0.0, 0usize);
            for i in 0..n {
                if region[i] > 0.5 {
                    s += v[i];
                    c += 1;
                }
            }
            s / c as f64
        };
        let pos_core = mean_in(&chi_pos, &inner);
        let shell: Vec<f64> = (0..n)
            .map(|i| if outer[i] > 0.5 && inner[i] < 0.5 { 1.0 } else { 0.0 })
            .collect();
        let neg_shell = mean_in(&chi_neg, &shell);
        assert!(
            (pos_core - 0.08).abs() < 0.02,
            "chi+ core mean {:.4} vs true 0.08",
            pos_core
        );
        assert!(
            (neg_shell + 0.05).abs() < 0.02,
            "chi- shell mean {:.4} vs true -0.05",
            neg_shell
        );
    }

    /// The voxelwise init must solve the 2×2 system exactly when QSM and R2'
    /// are consistent, before any sign clipping is needed.
    #[test]
    fn test_chi_sep_ilsqr_init_solves_linear_system() {
        let grid = Grid::new(2, 1, 1, 1.0, 1.0, 1.0);
        // Voxel 0: pure paramagnetic 0.1 ppm; voxel 1: mixed +0.06 / -0.04.
        let qsm = vec![0.1, 0.02];
        let dr = 137.0;
        let r2prime = vec![dr * 0.1, dr * (0.06 + 0.04)];
        let field = vec![0.0, 0.0]; // irrelevant at max_iter = 0
        let magnitude = vec![1.0, 1.0];
        let mask = vec![1u8, 1];
        let params = ChiSepIlsqrParams {
            max_iter: 0,
            ..ChiSepIlsqrParams::default()
        };
        let (p, q, _) = chi_sep_ilsqr(
            &field, &r2prime, &magnitude, &qsm, &mask, &grid, (0.0, 0.0, 1.0), &params, |_, _| {},
        );
        assert!((p[0] - 0.1).abs() < 1e-6 && q[0].abs() < 1e-6, "pure para voxel");
        assert!(
            (p[1] - 0.06).abs() < 1e-6 && (q[1] + 0.04).abs() < 1e-6,
            "mixed voxel: got ({:.4}, {:.4})",
            p[1],
            q[1]
        );
    }
}

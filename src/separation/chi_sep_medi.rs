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
//! The two residuals are weighted as in Shin et al. 2021, the field term by (2π)² against
//! the R2' term's 1 (both in Hz), under the coupling weight `lambda_cpl`; edge-masked L1 TV
//! on each source (`lambda_para`, `lambda_dia`) regularises.
//!
//! The constraints (chi_pos >= 0, chi_neg <= 0) naturally break the gauge
//! freedom of the chi_pos + chi_neg formulation. In most voxels, either
//! chi_pos = 0 or chi_neg = 0, which pins one variable to its constraint
//! boundary and prevents correlated drift.
//!
//! # Reference pipeline
//!
//! The SNU-LIST toolbox's `chi_sep_MEDI` is driven by the same script as `chi_sep_iLSQR`
//! (`Chisep_script.m`), so the preprocessing is the one listed in
//! [`chi_sep_ilsqr`](mod@super::chi_sep_ilsqr): BET mask, ROMEO + weighted echo averaging, V-SHARP
//! `smvsize = 12`, STI Suite iLSQR QSM. How this function differs from the toolbox call:
//!
//! - The toolbox call is `chi_sep_MEDI(mag, local_field_hz, r2prime, N_std, mask, params,
//!   option_data)` with `params.lambda = 1`, `params.lambda_CSF = 1`, `option_data.qsm = QSM`
//!   (the iLSQR QSM) and `option_data.mask_CSF` (`Chisep_script.m:150-161`). Kim et al. 2025
//!   (Hum Brain Mapp 46(4):e70136, §2.3) also ran it with "MEDI regularization with a
//!   regularization parameter of 1". This function takes no QSM, no CSF mask and no noise map
//!   `N_std` (its SNR weight comes from the magnitude). Its λ defaults (30/30/100) were measured
//!   on the QSM-CI phantom (see the comment in `ChiSepParams`'s `Default`), not taken from the
//!   toolbox.
//! - D_r: the toolbox's value is inside the compiled `.p`. QSM.rs uses [`DR_SHIN_2021`] at every
//!   field strength (see [`chi_sep_ilsqr`](mod@super::chi_sep_ilsqr) for the 7 T question).
//!
//! Reference:
//! Shin, H., et al. (2021). "chi-separation: Magnetic susceptibility source
//! separation toward iron and myelin mapping in the brain." NeuroImage, 240:118371.
//!
//! # Pipeline rules
//!
//! The facts above that a configuration can depart from are published as rules in
//! [`crate::pipeline::rules::RULES`] (each with its strength, basis, reason and source): `chi-sep.mask-bet`, `chi-sep.unwrap-romeo`, `chi-sep.echo-combination`, `chi-sep.echo-weights`, `chi-sep.bg-vsharp`.

use num_complex::Complex32;
use crate::Grid;
use super::DR_SHIN_2021;
use crate::fft::Fft3dWorkspaceF32;
use crate::kernels::dipole::dipole_kernel_f32;
use crate::inversion::medi::{gradient_mask_f32, fgrad_periodic_inplace_f32};
#[cfg(any(test, doc))]
use crate::inversion::medi::bdiv_periodic_inplace_f32;
use crate::utils::padding::{next_fast_fft_size, pad3d, unpad3d};
use crate::utils::simd_ops::{dot_product_f32, norm_squared_f32, compute_p_weights_f32};
#[cfg(any(test, doc))]
use crate::utils::simd_ops::apply_gradient_weights_f32;
#[cfg(doc)]
use crate::utils::simd_ops::{axpy_f32, xpby_f32};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Workspace for chi-separation — holds all reusable buffers (f32), allocated once per call.
struct ChiSepWorkspace {
    n: usize,
    nx: usize, ny: usize, nz: usize,
    vsx: f32, vsy: f32, vsz: f32,

    fft_ws: Fft3dWorkspaceF32,

    gx: Vec<f32>,
    gy: Vec<f32>,
    gz: Vec<f32>,

    div_buf: Vec<f32>,

    complex_buf: Vec<Complex32>,
    dipole_buf: Vec<f32>,
    /// d_pos + d_neg inside the operator
    stage: Vec<f32>,

    cg_r: Vec<f32>,
    cg_p: Vec<f32>,
    cg_ap: Vec<f32>,
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
            div_buf: vec![0.0; n],
            complex_buf: vec![Complex32::new(0.0, 0.0); n],
            dipole_buf: vec![0.0; n],
            stage: vec![0.0; n],
            cg_r: vec![0.0; 2 * n],
            cg_p: vec![0.0; 2 * n],
            cg_ap: vec![0.0; 2 * n],
        }
    }
}

/// Per-axis TV weights `wG·Vr·wG` (`m·p·m` in [`apply_gradient_weights_f32`]'s terms), formed
/// once per Gauss-Newton iteration and multiplied in the same order, so `w·∇x` has the bits of
/// `m·p·m·∇x`.
type TvWeights = [Vec<f32>; 3];

fn tv_weights(mx: &[f32], my: &[f32], mz: &[f32], vr: &[f32], w: &mut TvWeights) {
    let [wx, wy, wz] = w;
    maybe_par_iter_mut!(wx)
        .zip(maybe_par_iter_mut!(wy))
        .zip(maybe_par_iter_mut!(wz))
        .enumerate()
        .for_each(|(i, ((a, b), c))| {
            *a = mx[i] * vr[i] * mx[i];
            *b = my[i] * vr[i] * my[i];
            *c = mz[i] * vr[i] * mz[i];
        });
}

/// `out = λ · bdiv(w ⊙ fgrad(x))` with periodic boundaries: the IRLS TV term, the same values
/// as [`fgrad_periodic_inplace_f32`], the weighting and [`bdiv_periodic_inplace_f32`] in turn
/// (bitwise; see the test), but a few z planes at a time so the gradients stay in cache and no
/// full-volume temporaries are written.
#[allow(clippy::too_many_arguments)]
fn tv_term(
    x: &[f32],
    w: &TvWeights,
    lambda: f32,
    nx: usize, ny: usize, nz: usize,
    vsx: f32, vsy: f32, vsz: f32,
    out: &mut [f32],
) {
    let nxy = nx * ny;
    let (hx, hy, hz) = (1.0 / vsx, 1.0 / vsy, 1.0 / vsz);
    let (ax, ay, az) = (-1.0 / vsx, -1.0 / vsy, -1.0 / vsz);
    // Weighted periodic forward differences of plane k: z into gz, and x, y when asked.
    let grad = |k: usize, gxy: Option<(&mut [f32], &mut [f32])>, gz: &mut [f32]| {
        let base = k * nxy;
        let znext = if k + 1 < nz { base + nxy } else { 0 };
        for q in 0..nxy {
            let p = base + q;
            gz[q] = w[2][p] * ((x[znext + q] - x[p]) * hz);
        }
        if let Some((gx, gy)) = gxy {
            for j in 0..ny {
                let row = base + j * nx;
                let yrow = if j + 1 < ny { row + nx } else { base };
                for i in 0..nx {
                    let q = i + j * nx;
                    let p = row + i;
                    let xn = if i + 1 < nx { x[p + 1] } else { x[row] };
                    gx[q] = w[0][p] * ((xn - x[p]) * hx);
                    gy[q] = w[1][p] * ((x[yrow + i] - x[p]) * hy);
                }
            }
        }
    };
    let planes = |k0: usize, o: &mut [f32], bufs: &mut [Vec<f32>; 4]| {
        let [gx, gy, gz, gz_prev] = bufs;
        grad(if k0 > 0 { k0 - 1 } else { nz - 1 }, None, gz_prev);
        for (dk, op) in o.chunks_mut(nxy).enumerate() {
            grad(k0 + dk, Some((gx, gy)), gz);
            for j in 0..ny {
                for i in 0..nx {
                    let q = i + j * nx;
                    let gx_prev = if i > 0 { gx[q - 1] } else { gx[nx - 1 + j * nx] };
                    let gy_prev = if j > 0 { gy[q - nx] } else { gy[i + (ny - 1) * nx] };
                    let div = (gx[q] - gx_prev) * ax + (gy[q] - gy_prev) * ay + (gz[q] - gz_prev[q]) * az;
                    op[q] = lambda * div;
                }
            }
            std::mem::swap(gz, gz_prev);
        }
    };
    let new_bufs = || [vec![0.0f32; nxy], vec![0.0f32; nxy], vec![0.0f32; nxy], vec![0.0f32; nxy]];
    // A few planes per task: each task recomputes one plane of z differences.
    const PLANES: usize = 4;
    #[cfg(feature = "parallel")]
    out.par_chunks_mut(nxy * PLANES)
        .enumerate()
        .for_each_init(new_bufs, |bufs, (c, o)| planes(c * PLANES, o, bufs));
    #[cfg(not(feature = "parallel"))]
    {
        let mut bufs = new_bufs();
        out.chunks_mut(nxy * PLANES).enumerate().for_each(|(c, o)| planes(c * PLANES, o, &mut bufs));
    }
}

/// `a = b + beta·a` elementwise ([`xpby_f32`]'s values, on rayon threads under `parallel`)
fn xpby(a: &mut [f32], b: &[f32], beta: f32) {
    maybe_par_iter_mut!(a).zip(maybe_par_iter!(b)).for_each(|(a, &b)| *a = b + beta * *a);
}

/// Chi-separation algorithm parameters.
///
/// `cf`, `dr_pos`, and `dr_neg` are acquisition/tissue dependent and should be
/// set for your scan; the remaining fields are optimization knobs with sensible
/// defaults.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct ChiSepParams {
    /// Central frequency in Hz (e.g. 123.2e6 for 3T)
    pub cf: f64,
    /// Paramagnetic L1 regularization weight
    pub lambda_para: f64,
    /// Diamagnetic L1 regularization weight
    pub lambda_dia: f64,
    /// Field/R2' coupling weight
    pub lambda_cpl: f64,
    /// Paramagnetic relaxivity in Hz/ppm
    pub dr_pos: f64,
    /// Diamagnetic relaxivity in Hz/ppm
    pub dr_neg: f64,
    /// Edge-mask percentage for morphology weighting
    pub percentage: f64,
    /// Inner conjugate-gradient tolerance
    pub cg_tol: f64,
    /// Inner conjugate-gradient max iterations
    pub cg_max_iter: usize,
    /// Outer Gauss-Newton max iterations
    pub max_iter: usize,
    /// Outer convergence tolerance
    pub tol: f64,
}

impl Default for ChiSepParams {
    fn default() -> Self {
        Self {
            cf: 123.2e6,
            // 30 against λ_cpl = 100, with the R2' term at Shin 2021's weight. On the QSM-CI
            // chisep phantom χ+/|χ−| correlate 0.88/0.66 with truth at 30 (10: 0.91/0.57,
            // 100: 0.83/0.65, 300: 0.73/0.58, 1000: 0.64/0.37). The old default of 1000 was
            // tuned with the R2' term weighted ~5000x heavier, where no λ separated the sources.
            lambda_para: 30.0,
            lambda_dia: 30.0,
            lambda_cpl: 100.0,
            // Same paper, same forward model, same single-orientation QSM as `chi_sep_ilsqr`, so
            // the same relaxivity. These read 114/30 until QSM.rs#130: 114 is the COSMOS-
            // referenced constant ([`DR_KIM_2025_COSMOS`](super::DR_KIM_2025_COSMOS)) borrowed
            // from the χ-sepnet networks, which this method is not, and 30 appears in no
            // publication — neither Shin 2021 nor Kim 2025 fits the two sources separately. The
            // integration test in `tests/chisep_qsmci.rs` already overrode both to 137, so the
            // measured behaviour is unchanged; only the default moved to match it.
            dr_pos: DR_SHIN_2021,
            dr_neg: DR_SHIN_2021,
            percentage: 0.3,
            cg_tol: 0.01,
            cg_max_iter: 100,
            max_iter: 10,
            tol: 0.1,
        }
    }
}

/// Chi-separation using MEDI-based coupled optimization.
///
/// # Arguments
/// * `local_field` - Local field map in ppm (same units convention as the
///   dipole-inversion algorithms; converted to Hz internally via `params.cf`)
/// * `r2prime` - R2' map in Hz
/// * `magnitude` - Magnitude image for edge weighting
/// * `mask` - Binary brain mask, 1 = brain
/// * `grid` - Volume grid (dimensions and voxel sizes)
/// * `bdir` - B0 field direction
/// * `params` - Chi-separation parameters (see [`ChiSepParams`])
/// * `progress` - Progress callback: `(iteration, total_iterations)`
///
/// # Returns
/// `(chi_pos, chi_neg, chi_total)` — susceptibility maps in ppm
///
/// Volumes whose dimensions are not FFT-friendly (2ᵃ·3ᵇ·5ᶜ) are transparently
/// zero-padded to the next fast size for the internal FFTs and cropped back
/// (see [`chi_sep_ilsqr`](crate::separation::chi_sep_ilsqr) for details).
///
/// # Performance
///
/// On the QSM-CI χ-separation phantom (164×205×205, padded to 180×216×216; 30 Gauss-Newton
/// iterations, ~1900 CG steps), AMD EPYC 9454: 4 threads 720 s → 432 s, 16 threads
/// 839 s → 178 s, without `parallel` 883 s → 688 s, WASM (Node, one thread) 2584 s → 2316 s
/// (QSM.rs#296). The output is bit-for-bit what it was before. Most of the time is the CG
/// operator's two dipole FFT pairs, its two TV terms and CG's two dot products, which stay
/// sequential so their sums do not change.
pub fn chi_sep_medi<F>(
    local_field: &[f64],
    r2prime: &[f64],
    magnitude: &[f64],
    mask: &[u8],
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &ChiSepParams,
    progress: F,
) -> (Vec<f64>, Vec<f64>, Vec<f64>)
where
    F: FnMut(usize, usize),
{
    let dims = grid.dims;
    let fast = (
        next_fast_fft_size(dims.0),
        next_fast_fft_size(dims.1),
        next_fast_fft_size(dims.2),
    );
    if fast == dims {
        return chi_sep_medi_core(local_field, r2prime, magnitude, mask, grid, bdir, params, progress);
    }
    let (vsx, vsy, vsz) = grid.voxel_size;
    let pgrid = Grid::new(fast.0, fast.1, fast.2, vsx, vsy, vsz);
    let (chi_pos, chi_neg, chi_total) = chi_sep_medi_core(
        &pad3d(local_field, dims, fast),
        &pad3d(r2prime, dims, fast),
        &pad3d(magnitude, dims, fast),
        &pad3d(mask, dims, fast),
        &pgrid,
        bdir,
        params,
        progress,
    );
    (
        unpad3d(&chi_pos, fast, dims),
        unpad3d(&chi_neg, fast, dims),
        unpad3d(&chi_total, fast, dims),
    )
}

#[allow(clippy::too_many_arguments)]
fn chi_sep_medi_core<F>(
    local_field: &[f64],
    r2prime: &[f64],
    magnitude: &[f64],
    mask: &[u8],
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &ChiSepParams,
    mut progress: F,
) -> (Vec<f64>, Vec<f64>, Vec<f64>)
where
    F: FnMut(usize, usize),
{
    let cf = params.cf;
    let lambda_para = params.lambda_para;
    let lambda_dia = params.lambda_dia;
    let lambda_cpl = params.lambda_cpl;
    let dr_pos = params.dr_pos;
    let dr_neg = params.dr_neg;
    let percentage = params.percentage;
    let cg_tol = params.cg_tol;
    let cg_max_iter = params.cg_max_iter;
    let max_iter = params.max_iter;
    let tol = params.tol;
    let (nx, ny, nz) = grid.dims;
    let (vsx, vsy, vsz) = grid.voxel_size;
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

    // Data-term weighting from Shin et al. 2021: the cost is
    //   ‖Wr·(R2' − Dr·(|χ+| + |χ−|))‖² + ‖i·2π·Wf·(f − D(χ+ + χ−))‖²
    // with both residuals in Hz, so the field term carries (2π)² against the R2' term's 1
    // (the same weighting `chi_sep_ilsqr` uses). Here λ_cpl multiplies the field term, so the
    // R2' equation (data and relaxivities) is scaled by 1/(2π).
    //
    // This used to be auto-tuned to put the χ+/χ− gauge mode's eigenvalue at 10·max(λ_para,
    // λ_dia), which weighted the R2' term ~120x above the field term at 3-7 T instead of ~40x
    // below it. The solver then met R2' alone and returned χ+ ≈ |χ−| ≈ R2'/2Dr: the same map
    // twice (corr(χ+, |χ−|) 0.99 on the QSM-CI chisep phantom, |χ−| at 0.07 against its truth).
    let r2_scale = 1.0 / std::f32::consts::TAU;
    let dr_p_use = dr_p_eff * r2_scale;
    let dr_q_use = dr_q_eff * r2_scale;

    // Local field arrives in ppm (library-wide convention); the solver works in
    // Hz internally, so convert on entry.
    let field_f32: Vec<f32> = local_field.iter()
        .zip(mask.iter())
        .map(|(&v, &m)| if m != 0 { (v * cf * 1.0e-6) as f32 } else { 0.0 })
        .collect();
    let r2p_f32: Vec<f32> = r2prime.iter()
        .zip(mask.iter())
        .map(|(&v, &m)| if m != 0 { (v as f32) * r2_scale } else { 0.0 })
        .collect();
    let mag_f32: Vec<f32> = magnitude.iter().map(|&v| v as f32).collect();

    let mut ws = ChiSepWorkspace::new(nx, ny, nz, vsx_f32, vsy_f32, vsz_f32);
    let d_kernel = dipole_kernel_f32(grid, bdir_f32);
    let (mx, my, mz) = gradient_mask_f32(
        &mag_f32, mask, nx, ny, nz, vsx_f32, vsy_f32, vsz_f32, percentage as f32,
    );

    // chi_pos >= 0 (paramagnetic), chi_neg <= 0 (diamagnetic), both in Hz
    let mut chi_pos = vec![0.0f32; n];
    let mut chi_neg = vec![0.0f32; n];

    let mut vr = vec![0.0f32; n];
    let new_w = || -> TvWeights { [vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]] };
    let mut w_pos = new_w();
    let mut w_neg = new_w();
    let n2 = 2 * n;
    let mut dx = vec![0.0f32; n2];
    let mut rhs = vec![0.0f32; n2];
    let mut chi_sum_buf = vec![0.0f32; n];
    let mut field_residual = vec![0.0f32; n];

    // TV weight on (chi+ + chi-) sum — paper uses 2*lambda for sum term
    // Disabled for now (0.0) pending parameter tuning; the sum TV can over-couple components.
    // Every use of the term is skipped while its weight is exactly 0 (it would add 0).
    let lambda_sum_f32 = 0.0_f32;
    let mut w_sum = (lambda_sum_f32 != 0.0).then(new_w);

    let eps = 1.0e-6_f32;

    for iter in 0..max_iter {
        progress(iter + 1, max_iter);

        // --- TV reweighting: Vr = 1/sqrt(|wG∇χ|² + ε), then the weights wG·Vr·wG ---
        fgrad_periodic_inplace_f32(&mut ws.gx, &mut ws.gy, &mut ws.gz,
            &chi_pos, nx, ny, nz, vsx_f32, vsy_f32, vsz_f32);
        compute_p_weights_f32(&mut vr, &mx, &my, &mz, &ws.gx, &ws.gy, &ws.gz, eps);
        tv_weights(&mx, &my, &mz, &vr, &mut w_pos);

        fgrad_periodic_inplace_f32(&mut ws.gx, &mut ws.gy, &mut ws.gz,
            &chi_neg, nx, ny, nz, vsx_f32, vsy_f32, vsz_f32);
        compute_p_weights_f32(&mut vr, &mx, &my, &mz, &ws.gx, &ws.gy, &ws.gz, eps);
        tv_weights(&mx, &my, &mz, &vr, &mut w_neg);

        maybe_par_iter_mut!(chi_sum_buf).enumerate().for_each(|(i, s)| *s = chi_pos[i] + chi_neg[i]);

        // TV weights for the sum (chi+ + chi-)
        if let Some(w_sum) = w_sum.as_mut() {
            fgrad_periodic_inplace_f32(&mut ws.gx, &mut ws.gy, &mut ws.gz,
                &chi_sum_buf, nx, ny, nz, vsx_f32, vsy_f32, vsz_f32);
            compute_p_weights_f32(&mut vr, &mx, &my, &mz, &ws.gx, &ws.gy, &ws.gz, eps);
            tv_weights(&mx, &my, &mz, &vr, w_sum);
        }

        // --- Residuals (chi_sum_buf holds chi_pos + chi_neg) ---
        // field_residual = field - D*(chi_pos + chi_neg)
        ws.fft_ws.apply_dipole_inplace(&chi_sum_buf, &d_kernel, &mut ws.dipole_buf, &mut ws.complex_buf);
        maybe_par_iter_mut!(field_residual).enumerate().for_each(|(i, r)| *r = field_f32[i] - ws.dipole_buf[i]);

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
        let (rhs_pos, rhs_neg) = rhs.split_at_mut(n);
        tv_term(&chi_pos, &w_pos, lambda_para_f32, nx, ny, nz, vsx_f32, vsy_f32, vsz_f32, rhs_pos);
        tv_term(&chi_neg, &w_neg, lambda_dia_f32, nx, ny, nz, vsx_f32, vsy_f32, vsz_f32, rhs_neg);

        // TV gradient for (chi+ + chi-) sum — applied equally to both components
        if let Some(w_sum) = w_sum.as_ref() {
            tv_term(&chi_sum_buf, w_sum, lambda_sum_f32, nx, ny, nz, vsx_f32, vsy_f32, vsz_f32, &mut ws.div_buf);
            add_to_both(rhs_pos, rhs_neg, &ws.div_buf);
        }

        // Field fidelity gradient: -λ_cpl * D(field_res), SAME sign for both
        ws.fft_ws.apply_dipole_inplace(&field_residual, &d_kernel,
            &mut ws.dipole_buf, &mut ws.complex_buf);

        // Then the R2' fidelity gradient on the mask, with the R2' residual scaled by r2_scale:
        //   r2_res = R2'·s - dr_p_use * chi_pos + dr_q_use * chi_neg   (dr_*_use = dr_*_eff·s)
        //   ∂J/∂χ+ contribution: -λ_cpl * r2_res * dr_p_use
        //   ∂J/∂χ- contribution: +λ_cpl * r2_res * dr_q_use
        // and the negation for CG: b = -gradient.
        let dip = &ws.dipole_buf;
        maybe_par_iter_mut!(rhs_pos).zip(maybe_par_iter_mut!(rhs_neg)).enumerate().for_each(|(i, (rp, rn))| {
            let fg = lambda_cpl_f32 * dip[i];
            let (mut p, mut q) = (*rp - fg, *rn - fg);
            if mask[i] != 0 {
                let r2_residual = r2p_f32[i] - dr_p_use * chi_pos[i] + dr_q_use * chi_neg[i];
                p -= lambda_cpl_f32 * r2_residual * dr_p_use;
                q += lambda_cpl_f32 * r2_residual * dr_q_use;
            }
            *rp = -p;
            *rn = -q;
        });

        // --- CG solve ---
        let op = ChiSepOperator {
            d_kernel: &d_kernel,
            w_pos: &w_pos, w_neg: &w_neg, w_sum: w_sum.as_ref(),
            lambda_para: lambda_para_f32, lambda_dia: lambda_dia_f32,
            lambda_sum: lambda_sum_f32, lambda_cpl: lambda_cpl_f32,
            dr_p: dr_p_use, dr_q: dr_q_use,
            mask,
        };
        cg_solve_chisep(&mut ws, &op, &rhs, &mut dx, cg_tol_f32, cg_max_iter);

        // --- Update (half Newton step for stability with sign constraints), then
        // --- enforce the constraints chi_pos >= 0, chi_neg <= 0 ---
        maybe_par_iter_mut!(chi_pos).zip(maybe_par_iter_mut!(chi_neg)).enumerate().for_each(|(i, (p, q))| {
            if mask[i] == 0 {
                *p = 0.0;
                *q = 0.0;
            } else {
                *p = (*p + 0.5 * dx[i]).max(0.0);
                *q = (*q + 0.5 * dx[n + i]).min(0.0);
            }
        });

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

/// `a += t`, `b += t` elementwise
fn add_to_both(a: &mut [f32], b: &mut [f32], t: &[f32]) {
    maybe_par_iter_mut!(a).zip(maybe_par_iter_mut!(b)).enumerate().for_each(|(i, (a, b))| {
        *a += t[i];
        *b += t[i];
    });
}

/// The fixed parts of the chi-sep Hessian operator for one Gauss-Newton iteration.
struct ChiSepOperator<'a> {
    /// The dipole kernel D
    d_kernel: &'a [f32],
    w_pos: &'a TvWeights,
    w_neg: &'a TvWeights,
    /// `None` while `lambda_sum` is 0
    w_sum: Option<&'a TvWeights>,
    lambda_para: f32,
    lambda_dia: f32,
    lambda_sum: f32,
    lambda_cpl: f32,
    dr_p: f32,
    dr_q: f32,
    mask: &'a [u8],
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
/// The λ_sum term is skipped while λ_sum is 0.
fn apply_chisep_operator(ws: &mut ChiSepWorkspace, op: &ChiSepOperator, dx: &[f32], out: &mut [f32]) {
    let n = ws.n;
    let (nx, ny, nz) = (ws.nx, ws.ny, ws.nz);
    let (vsx, vsy, vsz) = (ws.vsx, ws.vsy, ws.vsz);

    let (d_pos, d_neg) = dx.split_at(n);
    let (out_pos, out_neg) = out.split_at_mut(n);

    // TV for chi_pos: λ_para * bdiv(wG*Vr_pos*wG*fgrad(d_pos))
    tv_term(d_pos, op.w_pos, op.lambda_para, nx, ny, nz, vsx, vsy, vsz, out_pos);
    // TV for chi_neg: λ_dia * bdiv(wG*Vr_neg*wG*fgrad(d_neg))
    tv_term(d_neg, op.w_neg, op.lambda_dia, nx, ny, nz, vsx, vsy, vsz, out_neg);

    // `stage` holds d_pos + d_neg for the sum TV and the field-fidelity term.
    maybe_par_iter_mut!(ws.stage).enumerate().for_each(|(i, s)| *s = d_pos[i] + d_neg[i]);

    // TV for sum (d_pos + d_neg): λ_sum * bdiv(wG*Vr_sum*wG*fgrad(d_pos+d_neg)),
    // applied equally to both components
    if let Some(w_sum) = op.w_sum {
        tv_term(&ws.stage, w_sum, op.lambda_sum, nx, ny, nz, vsx, vsy, vsz, &mut ws.div_buf);
        add_to_both(out_pos, out_neg, &ws.div_buf);
    }

    // Field fidelity: λ_cpl * D²(d_pos + d_neg), SAME for both
    ws.fft_ws.apply_dipole_inplace(&ws.stage, op.d_kernel, &mut ws.dipole_buf, &mut ws.complex_buf);
    ws.stage.copy_from_slice(&ws.dipole_buf);
    ws.fft_ws.apply_dipole_inplace(&ws.stage, op.d_kernel, &mut ws.dipole_buf, &mut ws.complex_buf);

    // R2' fidelity: rank-1 Hessian [dr_p, -dr_q]^T * [dr_p, -dr_q]
    // r2_lin = dr_p * d_pos - dr_q * d_neg
    // out_pos += λ_cpl * dr_p * r2_lin
    // out_neg -= λ_cpl * dr_q * r2_lin
    let (lambda_cpl, dr_p, dr_q, mask, dip) = (op.lambda_cpl, op.dr_p, op.dr_q, op.mask, &ws.dipole_buf);
    maybe_par_iter_mut!(out_pos).zip(maybe_par_iter_mut!(out_neg)).enumerate().for_each(|(i, (a, b))| {
        let ff = lambda_cpl * dip[i];
        *a += ff;
        *b += ff;
        if mask[i] != 0 {
            let r2_lin = dr_p * d_pos[i] - dr_q * d_neg[i];
            *a += lambda_cpl * dr_p * r2_lin;
            *b -= lambda_cpl * dr_q * r2_lin;
        }
    });
}

/// CG solver for the doubled chi-sep system (vectors in the workspace).
fn cg_solve_chisep(
    ws: &mut ChiSepWorkspace,
    op: &ChiSepOperator,
    b: &[f32],
    x: &mut [f32],
    tol: f32,
    max_iter: usize,
) {
    x.fill(0.0);

    let mut cg_r = std::mem::take(&mut ws.cg_r);
    let mut cg_p = std::mem::take(&mut ws.cg_p);
    let mut cg_ap = std::mem::take(&mut ws.cg_ap);

    cg_r.copy_from_slice(b);
    cg_p.copy_from_slice(&cg_r);

    let mut rsold = dot_product_f32(&cg_r, &cg_r);
    let b_norm = dot_product_f32(b, b).sqrt();

    if b_norm >= 1e-10 {
        for _cg_iter in 0..max_iter {
            apply_chisep_operator(ws, op, &cg_p, &mut cg_ap);

            let pap = dot_product_f32(&cg_p, &cg_ap);
            if pap.abs() < 1e-15 {
                break;
            }

            let alpha = rsold / pap;
            // x += α·p, r −= α·Ap (as `axpy_f32` computes them), in one pass
            maybe_par_iter_mut!(x).zip(maybe_par_iter_mut!(cg_r)).enumerate().for_each(|(i, (x, r))| {
                *x += alpha * cg_p[i];
                *r += -alpha * cg_ap[i];
            });

            let rsnew = dot_product_f32(&cg_r, &cg_r);
            if rsnew.sqrt() < tol * b_norm {
                break;
            }

            let beta_cg = rsnew / rsold;
            xpby(&mut cg_p, &cg_r, beta_cg);
            rsold = rsnew;
        }
    }

    ws.cg_r = cg_r;
    ws.cg_p = cg_p;
    ws.cg_ap = cg_ap;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Grid;
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
        let grid = Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);
        let bdir = (0.0, 0.0, 1.0);
        let cf: f64 = 123.2e6; // 3T

        let chi_pos_true_ppm = 0.05;
        let chi_neg_true_ppm = -0.03;

        let sphere_inner = make_sphere(nx, ny, nz, 16.0, 16.0, 16.0, 4.0);
        let sphere_outer = make_sphere(nx, ny, nz, 16.0, 16.0, 16.0, 8.0);
        let brain_mask = make_sphere(nx, ny, nz, 16.0, 16.0, 16.0, 12.0);

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

        // Forward model: field_ppm = D * chi_total_ppm (library units convention)
        let chi_total_ppm: Vec<f64> = chi_pos_ppm.iter()
            .zip(chi_neg_ppm.iter())
            .map(|(&p, &n)| p + n)
            .collect();
        let d = dipole_kernel(&grid, bdir);
        let chi_fft = fft3d_real(&chi_total_ppm, nx, ny, nz);
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

        let params = ChiSepParams {
            cf,
            lambda_para: 1000.0, lambda_dia: 1000.0, lambda_cpl: 100.0,
            dr_pos, dr_neg,
            percentage: 0.3, cg_tol: 0.01, cg_max_iter: 100, max_iter: 10, tol: 0.1,
        };
        let (chi_pos_out, chi_neg_out, chi_total_out) = chi_sep_medi(
            &local_field, &r2prime, &magnitude, &mask,
            &grid, bdir, &params,
            |_, _| {},
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

    /// The fused TV term equals the three-pass `fgrad`, weighting, `bdiv` it replaced, bitwise,
    /// on a grid with unequal sides and voxel sizes (and a single-plane one).
    #[test]
    fn tv_term_equals_unfused_passes() {
        for &(nx, ny, nz) in &[(13usize, 9usize, 7usize), (6, 5, 1), (1, 4, 9)] {
            let n = nx * ny * nz;
            let (vsx, vsy, vsz) = (0.9f32, 1.1f32, 1.3f32);
            let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
            let vr: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32 * 0.21).cos().abs()).collect();
            let m = |s: usize| -> Vec<f32> { (0..n).map(|i| !(i * s).is_multiple_of(5) as u8 as f32).collect() };
            let (mx, my, mz) = (m(3), m(7), m(11));
            let lambda = 30.0f32;

            let (mut gx, mut gy, mut gz) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
            let (mut rx, mut ry, mut rz) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
            let mut div = vec![0.0f32; n];
            fgrad_periodic_inplace_f32(&mut gx, &mut gy, &mut gz, &x, nx, ny, nz, vsx, vsy, vsz);
            apply_gradient_weights_f32(&mut rx, &mut ry, &mut rz, &mx, &my, &mz, &vr, &gx, &gy, &gz);
            bdiv_periodic_inplace_f32(&mut div, &rx, &ry, &rz, nx, ny, nz, vsx, vsy, vsz);
            let want: Vec<f32> = div.iter().map(|&d| lambda * d).collect();

            let mut w: TvWeights = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
            tv_weights(&mx, &my, &mz, &vr, &mut w);
            let mut got = vec![f32::NAN; n];
            tv_term(&x, &w, lambda, nx, ny, nz, vsx, vsy, vsz, &mut got);
            assert_eq!(got, want, "{:?}", (nx, ny, nz));
        }
    }

    fn corr(a: &[f64], b: &[f64], mask: &[u8]) -> f64 {
        let idx: Vec<usize> = (0..a.len()).filter(|&i| mask[i] != 0).collect();
        let n = idx.len() as f64;
        let ma = idx.iter().map(|&i| a[i]).sum::<f64>() / n;
        let mb = idx.iter().map(|&i| b[i]).sum::<f64>() / n;
        let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
        for &i in &idx {
            let (x, y) = (a[i] - ma, b[i] - mb);
            ab += x * y;
            aa += x * x;
            bb += y * y;
        }
        ab / (aa * bb).sqrt()
    }

    /// The two sources must come back as two different maps, each where it was put.
    ///
    /// A paramagnetic and a diamagnetic sphere side by side at 7 T. Before the R2' term carried
    /// Shin 2021's weight (1 against the field term's (2π)²) it was auto-scaled ~120x above the
    /// field term, and the solver met R2' alone: χ+ and |χ−| both came back ≈ R2'/2Dr, i.e. the
    /// same map, with |χ−| bright in the paramagnetic sphere.
    #[test]
    fn test_chi_sep_medi_separates_sources() {
        let (nx, ny, nz) = (32, 32, 32);
        let n = nx * ny * nz;
        let grid = Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);
        let bdir = (0.0, 0.0, 1.0);
        let cf: f64 = crate::constants::GAMMA_BAR_HZ_PER_T * 7.0;
        let dr = super::DR_SHIN_2021;

        let para = make_sphere(nx, ny, nz, 10.0, 16.0, 16.0, 4.0);
        let dia = make_sphere(nx, ny, nz, 22.0, 16.0, 16.0, 4.0);
        let brain = make_sphere(nx, ny, nz, 16.0, 16.0, 16.0, 13.0);
        let mask: Vec<u8> = brain.iter().map(|&v| (v > 0.5) as u8).collect();
        let gt_pos: Vec<f64> = para.iter().map(|&v| if v > 0.5 { 0.1 } else { 0.0 }).collect();
        let gt_neg: Vec<f64> = dia.iter().map(|&v| if v > 0.5 { -0.05 } else { 0.0 }).collect();

        let total: Vec<f64> = gt_pos.iter().zip(&gt_neg).map(|(p, q)| p + q).collect();
        let d = dipole_kernel(&grid, bdir);
        let f: Vec<_> = fft3d_real(&total, nx, ny, nz).iter().zip(&d).map(|(&c, &k)| c * k).collect();
        let field = ifft3d_real(&f, nx, ny, nz);
        let r2prime: Vec<f64> = (0..n).map(|i| dr * (gt_pos[i] - gt_neg[i])).collect();
        let magnitude: Vec<f64> = mask.iter().map(|&m| 100.0 * m as f64).collect();

        let params = ChiSepParams { cf, ..ChiSepParams::default() };
        let (chi_pos, chi_neg, _) =
            chi_sep_medi(&field, &r2prime, &magnitude, &mask, &grid, bdir, &params, |_, _| {});
        let dia_mag: Vec<f64> = chi_neg.iter().map(|v| -v).collect();
        let gt_dia: Vec<f64> = gt_neg.iter().map(|v| -v).collect();

        let mean_in = |v: &[f64], region: &[f64]| {
            let idx: Vec<usize> = (0..n).filter(|&i| region[i] > 0.5).collect();
            idx.iter().map(|&i| v[i]).sum::<f64>() / idx.len() as f64
        };
        let same = corr(&chi_pos, &dia_mag, &mask);
        let pos_own = corr(&chi_pos, &gt_pos, &mask);
        let neg_own = corr(&dia_mag, &gt_dia, &mask);
        println!("corr(χ+,|χ−|) {same:.3}  χ+ vs truth {pos_own:.3}  |χ−| vs truth {neg_own:.3}");
        println!(
            "χ+ in para/dia sphere {:.4}/{:.4}  |χ−| in para/dia sphere {:.4}/{:.4}",
            mean_in(&chi_pos, &para), mean_in(&chi_pos, &dia),
            mean_in(&dia_mag, &para), mean_in(&dia_mag, &dia)
        );
        assert!(same < 0.3, "χ+ and |χ−| are the same map: corr {same:.3}");
        assert!(pos_own > 0.7, "χ+ vs its truth {pos_own:.3}");
        assert!(neg_own > 0.7, "|χ−| vs its truth {neg_own:.3}");
        assert!(mean_in(&chi_pos, &para) > 3.0 * mean_in(&chi_pos, &dia), "χ+ not concentrated in the paramagnetic sphere");
        assert!(mean_in(&dia_mag, &dia) > 3.0 * mean_in(&dia_mag, &para), "|χ−| not concentrated in the diamagnetic sphere");
    }
}

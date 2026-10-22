//! MEDI (Morphology Enabled Dipole Inversion), a port of `MEDI_L1.m` from the MEDI toolbox
//! (release 2024.11.26)
//!
//! Solves
//!
//! min_χ ||wG ∇χ||₁ + 2λ ||m (exp(i Dχ) − exp(i f))||²₂
//!
//! by Gauss-Newton, with the L1 term handled by lagged diffusivity and each Gauss-Newton step
//! solved by conjugate gradients. `f` is the local field in radians, `m` the data weight
//! (`dataterm_mask.m`) and `wG` the binary edge mask from the magnitude (`gradient_mask.m`): 0 at
//! the strongest magnitude edges, 1 elsewhere, per gradient direction.
//!
//! The port follows `MEDI_L1.m` and its helpers line for line: `fgrad.m`/`bdiv.m` (forward
//! differences, zero at the last voxel, and their adjoint), the per-direction weight
//! `1/sqrt(|wG ∇χ|² + 1e-6)`, `cgsolve.m`, `sphere_kernel.m` (the partial-volume sphere) for the
//! SMV step, and the MERIT reweighting. Defaults are those of `MEDI_L1.m` and
//! `parse_QSM_input.m`: λ = 1000, 10 Gauss-Newton iterations, 100 CG iterations, tolerances 0.1
//! and 0.01, `percentage` 0.9, SNR data weighting, SMV (radius 5 mm, shrinking the mask), MERIT
//! and the CSF zero reference (MEDI+0, `lambda_CSF` 100) on. Not ported: the preconditioner
//! `P`, the linear formulation and the `Lreg` term (all off unless requested).
//!
//! # MEDI+0: CSF zero reference
//! MEDI's pipeline (README.m) computes R2* from the echo magnitudes with `arlo.m`
//! ([`arlo`](crate::utils::r2star::arlo)), the ventricular CSF mask from it with
//! `extract_CSF.m` ([`extract_csf`]), and passes the mask to `MEDI_L1`, which then adds
//! `reg_CSF.m`'s term `lambda_CSF/2 · ||M_CSF (χ − mean_CSF(χ))||²` to the cost and subtracts
//! the CSF mean from the result. `medi` does the same from the echo magnitudes and echo times
//! it requires ([`MediParams::csf`], default on, as `parse_QSM_input` sets `lambda_CSF` = 100).
//! Differences from MATLAB, which only matter where MATLAB fails or gives no answer:
//! - Fewer than three echoes: `arlo.m` gives no R2* for one echo (so MEDI runs without the CSF
//!   term) and 0 everywhere for two (which would make the whole central brain "CSF"). `medi`
//!   runs without the CSF term in both cases.
//! - A CSF mask with no voxels makes `MEDI_L1` return NaN (the mean over no voxels); `medi`
//!   runs without the CSF term instead.
//! - `extract_CSF.m` errors when the central sphere holds fewer than three low-R2* regions;
//!   [`extract_csf`] uses those it finds.
//! - `MEDI_L1` subtracts the CSF mean from every voxel, so the result is `−mean` outside the
//!   mask; `medi` keeps the result zero outside the mask, like the other inversions. Inside the
//!   mask the values are the same.
//!
//! # Units
//! Like the other inversions, `medi` takes the local field in ppm and returns ppm. MEDI works on
//! the field in radians accumulated over the echo spacing (`delta_TE`, `TE(2) − TE(1)` for
//! multi-echo data and `TE` for one echo, as MEDI's DICOM readers set it) at the centre frequency
//! `CF = γ·B0`. The field is converted with `2π·CF·delta_TE·1e-6` first and the result converted
//! back at the end, as `MEDI_L1.m` does. The result depends on this scale (the data term is
//! nonlinear and λ is fixed), so `b0` and `echo_times` are scan parameters that the pipeline
//! fills from metadata.
//!
//! # Magnitude and data weight
//! `MEDI_L1.m` weights the data by `1/N_std`, the noise standard deviation of the field
//! (normalised to mean 1 in the mask), and builds the edge mask from `iMag`. MEDI's pipeline
//! (README.m) takes both from the complex echoes: `iMag = sqrt(sum(abs(iField).^2,4))` and
//! `N_std` from `Fit_ppm_complex.m`, which depends only on the echo magnitudes
//! ([`fit_ppm_complex_noise`](crate::utils::fit_ppm_complex_noise)). `medi` requires the echo
//! magnitudes ([`EchoMagnitudes`]) and their echo times ([`MediParams::echo_times`]) and does
//! the same, so it always weights the data as
//! `MEDI_L1` does in MEDI's pipeline; one echo is fitted as `Fit_ppm_complex.m` fits one echo.
//! `Fit_ppm_complex.m` assumes evenly spaced echoes.
//!
//! # Zero padding
//! `zeropad` pads the volume at the end of each axis before the inversion and crops the result,
//! as `MEDI_L1`'s `'zeropad'` option does.
//!
//! **Deliberate deviation from `MEDI_L1.m`:** with SMV on, `sphere_kernel.m` builds its grid as
//! `-N/2:N/2-1`, which along an axis with an odd number of voxels centres the sphere half a voxel
//! off the grid. The SMV operator is then not symmetric, and `MEDI_L1` diverges (in MATLAB as
//! here: NRMSE above 10000% on a 164×205×205 phantom). So when SMV is on, `medi` pads every axis
//! that is still odd after `zeropad` by one more voxel and crops the result back. On grids that
//! are even after `zeropad`, and with SMV off, nothing changes and the result is `MEDI_L1`'s.
//!
//! Reference:
//! Liu, T., Liu, J., de Rochefort, L., Spincemaille, P., Khalidov, I., Ledoux, J.R.,
//! Wang, Y. (2011). "Morphology enabled dipole inversion (MEDI) from a single-angle
//! acquisition: comparison with COSMOS in human brain imaging."
//! Magnetic Resonance in Medicine, 66(3):777-783. https://doi.org/10.1002/mrm.22816
//!
//! Liu, J., Liu, T., de Rochefort, L., Ledoux, J., Khalidov, I., Chen, W., Tsiouris, A.J.,
//! Wisnieff, C., Spincemaille, P., Prince, M.R., Wang, Y. (2012).
//! "Morphology enabled dipole inversion for quantitative susceptibility mapping using
//! structural consistency between the magnitude image and the susceptibility map."
//! NeuroImage, 59(3):2560-2568.
//!
//! Reference implementation: MEDI toolbox release 2024.11.26 (Cornell MRI Research Lab,
//! `functions/MEDI_L1.m`)
//!
//! The single-precision helpers at the end of this file (`MediWorkspace`, `gradient_mask_f32`,
//! the periodic gradients, ...) are not used by `medi`. They follow the MEDI variant in the
//! chi-separation toolbox and Cornell's TFI code, and are shared by [`tfi`](super::tfi) and the
//! chi-separation solvers.

use num_complex::{Complex32, Complex64};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use crate::fft::{Box3, Fft3dWorkspace, Fft3dWorkspaceF32, FftOpts};
use crate::kernels::dipole::dipole_kernel;
use crate::utils::simd_ops::{apply_gradient_weights_f32, combine_terms_f32};
use crate::Grid;

/// MEDI parameters (`MEDI_L1.m` / `parse_QSM_input.m` defaults)
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct MediParams {
    /// Weight of the data-fidelity term (`'lambda'`): larger fits the data more closely
    pub lambda: f64,
    /// MERIT: reweight the noise by the data residual after each iteration (`'merit'`)
    pub merit: bool,
    /// Apply the spherical mean value high-pass to the field and the dipole kernel (`'smv'`)
    pub smv: bool,
    /// SMV radius in mm
    pub smv_radius: f64,
    /// With SMV, erode the mask to where the SMV of the mask exceeds 0.999 (`smv_shrink_mask`)
    pub smv_shrink_mask: bool,
    /// Data weighting: 0 uniform (1 everywhere), 1 SNR (`mask / n_std`, mean 1 in the mask)
    pub data_weighting: i32,
    /// Edge fraction for the gradient mask (`'percentage'`): the threshold is set so that the
    /// number of gradient components (all three directions) above it is this fraction of the
    /// number of voxels in the mask
    pub percentage: f64,
    /// CG relative residual tolerance
    pub cg_tol: f64,
    /// Maximum CG iterations per Gauss-Newton step
    pub cg_max_iter: usize,
    /// Maximum Gauss-Newton iterations
    pub max_iter: usize,
    /// Stop when ||dχ|| / ||χ|| falls below this
    pub tol: f64,
    /// Field strength in Tesla (scan parameter, for the radian conversion)
    pub b0: f64,
    /// Echo times in seconds, one per echo magnitude (scan parameter). The radian conversion
    /// uses MEDI's `delta_TE`: `TE(2) − TE(1)`, or `TE` for one echo; the CSF mask uses R2*
    /// from all of them (`arlo.m`).
    pub echo_times: Vec<f64>,
    /// MEDI+0: zero-reference to the ventricular CSF (`extract_CSF.m` on `arlo.m` R2*, and
    /// `MEDI_L1`'s `lambda_CSF` term). On by default, as in MEDI's pipeline. Needs at least
    /// three echoes; with fewer, MEDI runs without it (see the module docs).
    pub csf: bool,
    /// Weight of the CSF term (`'lambda_CSF'`)
    pub lambda_csf: f64,
    /// `extract_CSF.m`'s `thresh_R2s`: voxels below this R2* (Hz) are CSF candidates
    pub csf_r2s_threshold: f64,
    /// `extract_CSF.m`'s `flag_erode`: erode the mask by a 10 mm SMV before finding the CSF
    pub csf_erode: bool,
    /// Voxels of zeros appended along x, y and z before the inversion (`'zeropad'`). With SMV,
    /// axes that are still odd get one more (see the module docs).
    pub zeropad: (usize, usize, usize),
}

/// The magnitude of each echo, `n_echoes` volumes stacked with the echo slowest. MEDI's edge
/// mask uses their root sum of squares and its data weight the `Fit_ppm_complex.m` noise.
#[derive(Clone, Copy, Debug)]
pub struct EchoMagnitudes<'a> {
    /// `n_echoes` magnitude volumes, echo slowest
    pub data: &'a [f64],
    /// Number of echoes (evenly spaced)
    pub n_echoes: usize,
}

impl Default for MediParams {
    fn default() -> Self {
        Self {
            lambda: 1000.0,
            merit: true,
            smv: true,
            smv_radius: 5.0,
            smv_shrink_mask: true,
            data_weighting: 1,
            percentage: 0.9,
            cg_tol: 0.01,
            cg_max_iter: 100,
            max_iter: 10,
            tol: 0.1,
            b0: 3.0,
            echo_times: vec![0.005],
            csf: true,
            lambda_csf: 100.0,
            csf_r2s_threshold: 5.0,
            csf_erode: true,
            zeropad: (0, 0, 0),
        }
    }
}

/// MEDI dipole inversion (`MEDI_L1.m`)
///
/// # Arguments
/// * `local_field` - Local field in ppm (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz), 1 = brain
/// * `magnitude` - The echo magnitudes (see module docs)
/// * `grid` - Volume grid (dimensions and voxel sizes)
/// * `bdir` - B0 field direction
/// * `params` - MEDI parameters, including field strength and echo spacing
/// * `progress` - Progress callback `(current_step, total_steps)` over CG iterations
///
/// # Returns
/// Susceptibility in ppm, zero outside the mask (the SMV-eroded mask when `smv` is on)
pub fn medi(
    local_field: &[f64],
    mask: &[u8],
    magnitude: EchoMagnitudes,
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &MediParams,
    progress: impl FnMut(usize, usize),
) -> Vec<f64> {
    let n = grid.n_total();
    // iMag = sqrt(sum(abs(iField).^2,4)); [~, N_std] = Fit_ppm_complex(iField)
    let EchoMagnitudes { data: m, n_echoes } = magnitude;
    assert!(n_echoes > 0 && m.len() == n * n_echoes, "magnitude is not n_echoes volumes");
    let imag: Vec<f64> = (0..n)
        .map(|i| (0..n_echoes).map(|e| m[i + n * e].powi(2)).sum::<f64>().sqrt())
        .collect();
    let n_std = crate::utils::fit_ppm_complex_noise(m, n_echoes);
    let te = &params.echo_times;
    assert_eq!(te.len(), n_echoes, "echo_times does not match the echo magnitudes");
    // R2s = arlo(TE, abs(iField)); Mask_CSF = extract_CSF(R2s, Mask, voxel_size)
    let csf: Option<Vec<bool>> = if params.csf && n_echoes >= 3 {
        let r2s = crate::utils::r2star::arlo(m, te, n).expect("three or more increasing echo times");
        let c = extract_csf(&r2s, mask, grid, params.csf_r2s_threshold, params.csf_erode);
        c.iter().any(|&v| v != 0).then(|| c.iter().map(|&v| v != 0).collect())
    } else {
        None
    };

    let (nx, ny, nz) = grid.dims;
    let (zx, zy, zz) = params.zeropad;
    // Deviation (see module docs): with SMV, pad axes that are still odd by one voxel.
    let even = |n: usize, z: usize| z + usize::from(params.smv && (n + z) % 2 == 1);
    let (px, py, pz) = (even(nx, zx), even(ny, zy), even(nz, zz));
    if (px, py, pz) == (0, 0, 0) {
        return medi_l1(local_field, mask, &imag, &n_std, csf.as_deref(), grid, bdir, params, progress);
    }
    // padarray(..., pad, 'post'), then x = x(1:matrix_size0(1), ...)
    let (mx, my, mz) = (nx + px, ny + py, nz + pz);
    let pad = |v: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0; mx * my * mz];
        for k in 0..nz {
            for j in 0..ny {
                let (src, dst) = (nx * (j + ny * k), mx * (j + my * k));
                out[dst..dst + nx].copy_from_slice(&v[src..src + nx]);
            }
        }
        out
    };
    let mask_f: Vec<f64> = mask.iter().map(|&m| m as f64).collect();
    let mask_p: Vec<u8> = pad(&mask_f).iter().map(|&v| v as u8).collect();
    let pgrid = Grid::new(mx, my, mz, grid.vsx(), grid.vsy(), grid.vsz());
    let csf_p: Option<Vec<bool>> = csf.map(|c| {
        let f: Vec<f64> = c.iter().map(|&b| b as u8 as f64).collect();
        pad(&f).iter().map(|&v| v != 0.0).collect()
    });
    let chi = medi_l1(
        &pad(local_field), &mask_p, &pad(&imag), &pad(&n_std), csf_p.as_deref(),
        &pgrid, bdir, params, progress,
    );
    let mut out = vec![0.0; n];
    for k in 0..nz {
        for j in 0..ny {
            let (dst, src) = (nx * (j + ny * k), mx * (j + my * k));
            out[dst..dst + nx].copy_from_slice(&chi[src..src + nx]);
        }
    }
    out
}

/// `MEDI_L1.m` on an `iMag`, an `N_std` and, for MEDI+0, a `Mask_CSF`
#[allow(clippy::too_many_arguments)]
fn medi_l1(
    local_field: &[f64],
    mask: &[u8],
    magnitude: &[f64],
    n_std: &[f64],
    mask_csf: Option<&[bool]>,
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &MediParams,
    mut progress: impl FnMut(usize, usize),
) -> Vec<f64> {
    let n = grid.n_total();
    let vs = [grid.vsx(), grid.vsy(), grid.vsz()];
    let mut fft = Fft3dWorkspace::new(grid.nx(), grid.ny(), grid.nz());
    let mut buf = vec![Complex64::new(0.0, 0.0); n];

    // ppm -> radians over delta_TE; MEDI_L1 converts back with x/(2*pi*delta_TE*CF)*1e6.
    let cf = 42.576e6 * params.b0;
    let te = &params.echo_times;
    let delta_te = if te.len() == 1 { te[0] } else { te[1] - te[0] };
    let ppm_to_rad = 2.0 * std::f64::consts::PI * delta_te * cf * 1e-6;
    let mut rdf = vec![0.0f64; n];
    par_fill(&mut rdf, |i| local_field[i] * ppm_to_rad);

    // N_std = N_std.*Mask; tempn = N_std
    let mut mask: Vec<bool> = mask.iter().map(|&m| m != 0).collect();
    let mut n_std: Vec<f64> = (0..n)
        .map(|i| if mask[i] { n_std[i] } else { 0.0 })
        .collect();
    let mut tempn = n_std.clone();

    // D = dipole_kernel(...); complex because the SMV step multiplies it by SphereK.
    let mut d: Vec<Complex64> = dipole_kernel(grid, bdir).iter().map(|&v| Complex64::new(v, 0.0)).collect();

    let sphere_k = if params.smv {
        let k = sphere_kernel(grid, params.smv_radius, &mut fft);
        // if opts.smv_shrink_mask, Mask = SMV(Mask, SphereK) > 0.999
        if params.smv_shrink_mask {
            let m: Vec<f64> = mask.iter().map(|&b| b as u8 as f64).collect();
            smv(&m, &k, &mut fft, &mut buf);
            par_fill(&mut mask, |i| buf[i].re > 0.999);
        }
        // D = (1-SphereK).*D
        par_update(&mut d, |i, v| *v *= Complex64::new(1.0, 0.0) - k[i]);
        // RDF = RDF - SMV(RDF, SphereK); RDF = RDF.*Mask
        smv(&rdf, &k, &mut fft, &mut buf);
        par_update(&mut rdf, |i, v| *v = if mask[i] { *v - buf[i].re } else { 0.0 });
        // tempn = sqrt(SMV(tempn.^2, SphereK) + tempn.^2)
        smv_noise(&mut tempn, &k, &mut fft, &mut buf);
        Some(k)
    } else {
        None
    };

    let mut m = vec![0.0f64; n];
    dataterm_mask(params.data_weighting, &tempn, &mask, &mut m);
    let mut b0 = vec![Complex64::new(0.0, 0.0); n];
    par_fill(&mut b0, |i| m[i] * Complex64::new(0.0, rdf[i]).exp());
    let wg = gradient_mask(magnitude, &mask, grid, params.percentage);
    // The data weight m, and with it w and conj(w).*w, is zero outside the mask, so the
    // Dconv(dx) in A(dx) is only needed, and the Dconv after the weighting only has input,
    // inside the mask's bounding box.
    let mbox = if params.data_weighting == 0 {
        [0..grid.nx(), 0..grid.ny(), 0..grid.nz()]
    } else {
        bounding_box(&mask, grid.dims)
    };

    let lambda2 = 2.0 * params.lambda;
    // reg_CSF.m: opHop(x) = lambda_CSF * LT(LT(x)), LT(x) = Mask_CSF.*(x - mean(x(Mask_CSF)))
    let csf_idx: Vec<usize> = mask_csf.map_or(Vec::new(), |c| (0..n).filter(|&i| c[i]).collect());
    let reg_csf = |x: &[f64], acc: &mut [f64]| {
        if csf_idx.is_empty() {
            return;
        }
        let len = csf_idx.len() as f64;
        let mean = csf_idx.iter().map(|&i| x[i]).sum::<f64>() / len;
        let t1: Vec<f64> = csf_idx.iter().map(|&i| x[i] - mean).collect();
        let mean = t1.iter().sum::<f64>() / len;
        for (q, &i) in csf_idx.iter().enumerate() {
            acc[i] += params.lambda_csf * (t1[q] - mean);
        }
    };
    let e = 1e-6;
    let dims = grid.dims;
    let mut x = vec![0.0f64; n];
    // wG.*Vr.*wG per direction, Vr = 1./sqrt(abs(wG.*grad(real(x))).^2 + e)
    let mut wvw = [vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n]];
    let mut g = [vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n]];
    let mut w = vec![Complex64::new(0.0, 0.0); n];
    let mut w2 = vec![0.0f64; n];
    let mut b = vec![0.0f64; n];
    let mut t = vec![0.0f64; n];
    let mut rv = vec![0.0f64; n];
    let mut cg = CgBuffers::new(n);
    let mask_idx: Vec<usize> = (0..n).filter(|&i| mask[i]).collect();
    let total = params.max_iter * params.cg_max_iter;

    let mut res_norm_ratio = f64::INFINITY;
    let mut iter = 0;
    while res_norm_ratio > params.tol && iter < params.max_iter {
        iter += 1;
        // Vr = 1./sqrt(abs(wG.*grad(real(x))).^2 + e), per direction; kept as wG.*Vr.*wG
        fgrad(&x, dims, vs, &mut g);
        for c in 0..3 {
            let (wgc, gc) = (&wg[c], &g[c]);
            par_fill(&mut wvw[c], |i| {
                let u = wgc[i] * gc[i];
                let vr = 1.0 / (u * u + e).sqrt();
                wgc[i] * vr * wgc[i]
            });
        }
        // w = W(m, x) = m.*exp(1i*Dconv(x))
        dconv(&x, &d, &mut fft, &mut buf, &mut t);
        par_fill(&mut w, |i| m[i] * Complex64::new(0.0, t[i]).exp());
        par_fill(&mut w2, |i| w[i].norm_sqr());

        // b = reg(x) + 2*lambda*Dconv(real(conj(w).*conj(1i).*(w-b0)))
        reg(&x, &wvw, dims, vs, &mut b);
        reg_csf(&x, &mut b);
        par_fill(&mut rv, |i| (w[i].conj() * Complex64::new(0.0, -1.0) * (w[i] - b0[i])).re);
        dconv(&rv, &d, &mut fft, &mut buf, &mut t);
        par_update(&mut b, |i, v| *v = -(*v + lambda2 * t[i]));

        // A = @(dx) reg(dx) + 2*lambda*Dconv(conj(w).*w.*Dconv(dx))
        let base = (iter - 1) * params.cg_max_iter;
        cgsolve(
            |p: &[f64], out: &mut [f64]| {
                reg(p, &wvw, dims, vs, out);
                reg_csf(p, out);
                dconv_weighted(p, &d, &w2, &mbox, &mut fft, &mut buf);
                par_update(out, |i, v| *v += lambda2 * buf[i].re);
            },
            &b, params.cg_tol, params.cg_max_iter, &mut cg,
            |k| progress(base + k, total),
        );
        let dx = &cg.x;

        // res_norm_ratio = norm(dx)/norm(x); x = x + dx
        let ndx = dx.iter().map(|v| v * v).sum::<f64>().sqrt();
        let nrm_x = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        res_norm_ratio = ndx / nrm_x;
        par_update(&mut x, |i, v| *v += dx[i]);

        if params.merit {
            // wres = m.*exp(1i*real(ifftn(D.*fftn(x)))) - b0, used inside the mask only
            dconv(&x, &d, &mut fft, &mut buf, &mut t);
            let wres: Vec<Complex64> = mask_idx
                .iter()
                .map(|&i| m[i] * Complex64::new(0.0, t[i]).exp() - b0[i])
                .collect();
            let mean = wres.iter().copied().sum::<Complex64>() / mask_idx.len() as f64;
            // factor = std(abs(a))*6, MATLAB's std (normalised by N-1)
            let a: Vec<f64> = wres.iter().map(|&v| (v - mean).norm()).collect();
            let am = a.iter().sum::<f64>() / a.len() as f64;
            let sd = (a.iter().map(|v| (v - am).powi(2)).sum::<f64>() / (a.len() as f64 - 1.0)).sqrt();
            let factor = sd * 6.0;
            for (q, &i) in mask_idx.iter().enumerate() {
                let r = (a[q] / factor).max(1.0);
                n_std[i] *= r * r;
            }
            tempn.copy_from_slice(&n_std);
            if let Some(k) = &sphere_k {
                smv_noise(&mut tempn, k, &mut fft, &mut buf);
            }
            dataterm_mask(params.data_weighting, &tempn, &mask, &mut m);
            par_fill(&mut b0, |i| m[i] * Complex64::new(0.0, rdf[i]).exp());
        }
    }
    progress(total, total);

    // x = x/(2*pi*delta_TE*CF)*1e6.*Mask
    par_update(&mut x, |i, v| *v = if mask[i] { *v / ppm_to_rad } else { 0.0 });
    // if flag_CSF, x = x - mean(x(Mask_CSF)); kept zero outside the mask (see the module docs)
    if !csf_idx.is_empty() {
        let mean = csf_idx.iter().map(|&i| x[i]).sum::<f64>() / csf_idx.len() as f64;
        par_update(&mut x, |i, v| {
            if mask[i] {
                *v -= mean;
            }
        });
    }
    x
}

/// `out[i] = f(i)` for every `i` (over threads with the `parallel` feature)
fn par_fill<T: Send>(out: &mut [T], f: impl Fn(usize) -> T + Sync + Send) {
    #[cfg(feature = "parallel")]
    out.par_iter_mut().with_min_len(1 << 14).enumerate().for_each(|(i, v)| *v = f(i));
    #[cfg(not(feature = "parallel"))]
    out.iter_mut().enumerate().for_each(|(i, v)| *v = f(i));
}

/// `f(i, &mut out[i])` for every `i` (over threads with the `parallel` feature)
fn par_update<T: Send>(out: &mut [T], f: impl Fn(usize, &mut T) + Sync + Send) {
    #[cfg(feature = "parallel")]
    out.par_iter_mut().with_min_len(1 << 14).enumerate().for_each(|(i, v)| f(i, v));
    #[cfg(not(feature = "parallel"))]
    out.iter_mut().enumerate().for_each(|(i, v)| f(i, v));
}

/// The smallest box holding every `true` voxel (the whole grid if there is none)
fn bounding_box(mask: &[bool], (nx, ny, nz): (usize, usize, usize)) -> Box3 {
    let (mut lo, mut hi) = ([usize::MAX; 3], [0usize; 3]);
    for (p, _) in mask.iter().enumerate().filter(|(_, &m)| m) {
        let c = [p % nx, (p / nx) % ny, p / (nx * ny)];
        for a in 0..3 {
            lo[a] = lo[a].min(c[a]);
            hi[a] = hi[a].max(c[a] + 1);
        }
    }
    if lo[0] == usize::MAX {
        return [0..nx, 0..ny, 0..nz];
    }
    [lo[0]..hi[0], lo[1]..hi[1], lo[2]..hi[2]]
}

/// Loads a real array into the transform's rows as `complex(x, 0)`
fn load_real(x: &[f64]) -> impl Fn(usize, &mut [Complex64]) + Sync + '_ {
    move |l, row| {
        for (c, &v) in row.iter_mut().zip(&x[l..]) {
            *c = Complex64::new(v, 0.0);
        }
    }
}

/// Multiplies the transform's rows by `k` as they are loaded (`K.*buf`, fused into the first
/// pass of the next transform)
fn mul_kernel(k: &[Complex64]) -> impl Fn(usize, &mut [Complex64]) + Sync + '_ {
    move |l, row| {
        for (c, &kv) in row.iter_mut().zip(&k[l..]) {
            *c *= kv;
        }
    }
}

/// `Dconv`: `out = real(ifftn(D.*fftn(x)))`
fn dconv(x: &[f64], d: &[Complex64], fft: &mut Fft3dWorkspace, buf: &mut [Complex64], out: &mut [f64]) {
    let load = load_real(x);
    fft.fft3d_with(buf, &FftOpts { load: Some(&load), ..FftOpts::default() });
    fft.ifft3d_with(buf, &FftOpts { load: Some(&mul_kernel(d)), ..FftOpts::default() });
    par_fill(out, |i| buf[i].re);
}

/// `Dconv(w2.*Dconv(p))`, left in the real part of `buf`. `w2` is zero outside `mbox`, so the
/// inner `Dconv` is computed only on `mbox` and the outer one reads input only there; the
/// values are those of the two plain `Dconv`s (see [`Fft3dWorkspace::fft3d_with`]).
fn dconv_weighted(
    p: &[f64],
    d: &[Complex64],
    w2: &[f64],
    mbox: &Box3,
    fft: &mut Fft3dWorkspace,
    buf: &mut [Complex64],
) {
    let load = load_real(p);
    fft.fft3d_with(buf, &FftOpts { load: Some(&load), ..FftOpts::default() });
    // t = real(ifftn(D.*...)).*w2 on the box (the imaginary part is dropped on the way back in)
    let mul_d = mul_kernel(d);
    fft.ifft3d_with(buf, &FftOpts { needed: Some(mbox), post: Some(w2), load: Some(&mul_d), ..FftOpts::default() });
    let drop_imag = |_: usize, row: &mut [Complex64]| {
        for c in row.iter_mut() {
            c.im = 0.0;
        }
    };
    fft.fft3d_with(buf, &FftOpts { support: Some(mbox), load: Some(&drop_imag), ..FftOpts::default() });
    fft.ifft3d_with(buf, &FftOpts { load: Some(&mul_d), ..FftOpts::default() });
}

/// `fgrad.m`: forward differences, zero at the last voxel along each axis
fn fgrad(x: &[f64], dims: (usize, usize, usize), vs: [f64; 3], g: &mut [Vec<f64>; 3]) {
    let (nx, ny, nz) = dims;
    let nxy = nx * ny;
    let [g0, g1, g2] = g;
    let plane = |k: usize, ((a, b), c): ((&mut [f64], &mut [f64]), &mut [f64])| {
        for j in 0..ny {
            for i in 0..nx {
                let q = i + nx * j;
                let p = q + nxy * k;
                a[q] = if i + 1 < nx { (x[p + 1] - x[p]) / vs[0] } else { 0.0 };
                b[q] = if j + 1 < ny { (x[p + nx] - x[p]) / vs[1] } else { 0.0 };
                c[q] = if k + 1 < nz { (x[p + nxy] - x[p]) / vs[2] } else { 0.0 };
            }
        }
    };
    #[cfg(feature = "parallel")]
    g0.par_chunks_mut(nxy)
        .zip(g1.par_chunks_mut(nxy))
        .zip(g2.par_chunks_mut(nxy))
        .enumerate()
        .for_each(|(k, s)| plane(k, s));
    #[cfg(not(feature = "parallel"))]
    g0.chunks_mut(nxy)
        .zip(g1.chunks_mut(nxy))
        .zip(g2.chunks_mut(nxy))
        .enumerate()
        .for_each(|(k, s)| plane(k, s));
}

/// `bdiv.m`: the adjoint of `fgrad` (minus the backward-difference divergence). [`reg`] computes
/// the same with the gradients fused in; this is its reference in the tests.
#[cfg(test)]
fn bdiv(g: &[Vec<f64>; 3], dims: (usize, usize, usize), vs: [f64; 3], out: &mut [f64]) {
    let (nx, ny, nz) = dims;
    let nxy = nx * ny;
    // Dx = [Gx(1:end-1); 0] - [0; Gx(1:end-1)]; div = -(Dx/vsx + Dy/vsy + Dz/vsz)
    let term = |gc: &[f64], p: usize, i: usize, len: usize, stride: usize| -> f64 {
        let cur = if i + 1 < len { gc[p] } else { 0.0 };
        let prev = if i > 0 { gc[p - stride] } else { 0.0 };
        cur - prev
    };
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let p = i + nx * j + nxy * k;
                out[p] = -(term(&g[0], p, i, nx, 1) / vs[0]
                    + term(&g[1], p, j, ny, nx) / vs[1]
                    + term(&g[2], p, k, nz, nxy) / vs[2]);
            }
        }
    }
}

/// `reg(dx) = div(wG.*(Vr.*(wG.*grad(real(dx)))))` into `out`, with `wvw = wG.*Vr.*wG`: `bdiv`
/// of the weighted `fgrad`, a z plane at a time so the gradients never leave the cache. Each
/// value is computed with the same operations as `fgrad` followed by `bdiv`.
fn reg(x: &[f64], wvw: &[Vec<f64>; 3], dims: (usize, usize, usize), vs: [f64; 3], out: &mut [f64]) {
    let (nx, ny, nz) = dims;
    let nxy = nx * ny;
    // Weighted forward differences of plane k: x and y into gx, gy (when given), z into gz.
    let grad = |k: usize, gx: Option<(&mut [f64], &mut [f64])>, gz: &mut [f64]| {
        let base = nxy * k;
        for q in 0..nxy {
            let p = base + q;
            gz[q] = (if k + 1 < nz { (x[p + nxy] - x[p]) / vs[2] } else { 0.0 }) * wvw[2][p];
        }
        if let Some((gx, gy)) = gx {
            for j in 0..ny {
                for i in 0..nx {
                    let q = i + nx * j;
                    let p = base + q;
                    gx[q] = (if i + 1 < nx { (x[p + 1] - x[p]) / vs[0] } else { 0.0 }) * wvw[0][p];
                    gy[q] = (if j + 1 < ny { (x[p + nx] - x[p]) / vs[1] } else { 0.0 }) * wvw[1][p];
                }
            }
        }
    };
    // Dx = [Gx(1:end-1); 0] - [0; Gx(1:end-1)]; div = -(Dx/vsx + Dy/vsy + Dz/vsz)
    let planes = |k0: usize, o: &mut [f64], bufs: &mut [Vec<f64>; 4]| {
        let [gx, gy, gz, gz_prev] = bufs;
        if k0 > 0 {
            grad(k0 - 1, None, gz_prev);
        }
        for (dk, op) in o.chunks_mut(nxy).enumerate() {
            let k = k0 + dk;
            grad(k, Some((gx, gy)), gz);
            for j in 0..ny {
                for i in 0..nx {
                    let q = i + nx * j;
                    let tx = (if i + 1 < nx { gx[q] } else { 0.0 }) - (if i > 0 { gx[q - 1] } else { 0.0 });
                    let ty = (if j + 1 < ny { gy[q] } else { 0.0 }) - (if j > 0 { gy[q - nx] } else { 0.0 });
                    let tz = (if k + 1 < nz { gz[q] } else { 0.0 }) - (if k > 0 { gz_prev[q] } else { 0.0 });
                    op[q] = -(tx / vs[0] + ty / vs[1] + tz / vs[2]);
                }
            }
            std::mem::swap(gz, gz_prev);
        }
    };
    let new_bufs = || [vec![0.0f64; nxy], vec![0.0f64; nxy], vec![0.0f64; nxy], vec![0.0f64; nxy]];
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

/// `cgsolve.m`'s vectors, reused across Gauss-Newton iterations
struct CgBuffers {
    x: Vec<f64>,
    r: Vec<f64>,
    d: Vec<f64>,
    q: Vec<f64>,
}

impl CgBuffers {
    fn new(n: usize) -> Self {
        Self { x: vec![0.0; n], r: vec![0.0; n], d: vec![0.0; n], q: vec![0.0; n] }
    }
}

/// `cgsolve.m` from a zero start; the last iterate is left in `cg.x`. `progress` receives the
/// iteration count. The dot products are summed in index order, as before.
fn cgsolve(
    mut a: impl FnMut(&[f64], &mut [f64]),
    b: &[f64],
    tol: f64,
    max_iter: usize,
    cg: &mut CgBuffers,
    mut progress: impl FnMut(usize),
) {
    let dot = |u: &[f64], v: &[f64]| u.iter().zip(v).map(|(a, b)| a * b).sum::<f64>();
    let CgBuffers { x, r, d, q } = cg;
    x.fill(0.0);
    r.copy_from_slice(b);
    d.copy_from_slice(b);
    let mut delta = dot(r, r);
    let delta0 = dot(b, b);
    let mut numiter = 0;
    while numiter < max_iter && delta > tol * tol * delta0 {
        a(d, q);
        let alpha = delta / dot(d, q);
        if (numiter + 1) % 50 == 0 {
            par_update(x, |i, v| *v += alpha * d[i]);
            // r = b - A*x, recomputed every 50 iterations
            a(x, q);
            par_fill(r, |i| b[i] - q[i]);
        } else {
            #[cfg(feature = "parallel")]
            x.par_iter_mut().zip(r.par_iter_mut()).with_min_len(1 << 14).enumerate().for_each(|(i, (xv, rv))| {
                *xv += alpha * d[i];
                *rv -= alpha * q[i];
            });
            #[cfg(not(feature = "parallel"))]
            x.iter_mut().zip(r.iter_mut()).enumerate().for_each(|(i, (xv, rv))| {
                *xv += alpha * d[i];
                *rv -= alpha * q[i];
            });
        }
        let delta_old = delta;
        delta = dot(r, r);
        let beta = delta / delta_old;
        let rr = &*r;
        par_update(d, |i, v| *v = rr[i] + beta * *v);
        numiter += 1;
        progress(numiter);
    }
}

/// `dataterm_mask.m` into `w`: 0 gives 1 everywhere; 1 gives `Mask./N_std` (0 where that is
/// not finite) normalised to mean 1 over the mask.
fn dataterm_mask(mode: i32, n_std: &[f64], mask: &[bool], w: &mut [f64]) {
    if mode == 0 {
        w.fill(1.0);
        return;
    }
    par_fill(w, |i| {
        let (s, m) = (n_std[i], mask[i]);
        let v = (m as u8 as f64) / s;
        if m && v.is_finite() { v } else { 0.0 }
    });
    let count = mask.iter().filter(|&&m| m).count() as f64;
    let mean = w.iter().zip(mask).filter(|(_, &m)| m).map(|(v, _)| v).sum::<f64>() / count;
    par_update(w, |_, v| *v /= mean);
}

/// `gradient_mask.m` (binary mode): 1 where `|∇(iMag.*Mask)|` is at or below a threshold, per
/// direction, over the whole volume. The threshold starts at 1% of `max(iMag)` and is scaled by
/// 1.05 or 0.95 until the number of gradient components above it, over the number of voxels in
/// the mask, crosses `percentage`.
fn gradient_mask(mag: &[f64], mask: &[bool], grid: &Grid, percentage: f64) -> [Vec<f64>; 3] {
    let vs = [grid.vsx(), grid.vsy(), grid.vsz()];
    let n = mag.len();
    let masked: Vec<f64> = mag.iter().zip(mask).map(|(&v, &m)| if m { v } else { 0.0 }).collect();
    let mut g = [vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n]];
    fgrad(&masked, grid.dims, vs, &mut g);
    for c in g.iter_mut() {
        par_update(c, |_, v| *v = v.abs());
    }
    // Counting `wG > level` by binary search over the sorted values gives the same counts as
    // MATLAB's sum(wG(:) > level).
    let mut sorted: Vec<f64> = g.iter().flat_map(|c| c.iter().copied()).collect();
    let cmp = |a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
    #[cfg(feature = "parallel")]
    sorted.par_sort_unstable_by(cmp);
    #[cfg(not(feature = "parallel"))]
    sorted.sort_unstable_by(cmp);
    let above = |level: f64| sorted.len() - sorted.partition_point(|&v| v <= level);
    let denominator = mask.iter().filter(|&&m| m).count() as f64;
    let mut level = 0.01 * mag.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut numerator = above(level) as f64;
    if numerator / denominator > percentage {
        while numerator / denominator > percentage {
            level *= 1.05;
            numerator = above(level) as f64;
        }
    } else {
        // MATLAB loops forever if the fraction cannot reach `percentage` (e.g. a constant
        // magnitude, whose only edges are at the mask boundary). Stop once the level is below
        // the smallest nonzero gradient, where the count can no longer change.
        let min_positive = sorted.iter().copied().find(|&v| v > 0.0).unwrap_or(0.0);
        while numerator / denominator < percentage && level >= min_positive && level > 0.0 {
            level *= 0.95;
            numerator = above(level) as f64;
        }
    }
    for c in g.iter_mut() {
        par_update(c, |_, v| *v = if *v <= level { 1.0 } else { 0.0 });
    }
    g
}

/// `sphere_kernel.m`: a sphere of `radius` mm with partial-volume edge voxels (each split into
/// 20³ sub-voxels), normalised to sum 1, returned as its FFT (`fftn(fftshift(Sphere))`).
fn sphere_kernel(grid: &Grid, radius: f64, fft: &mut Fft3dWorkspace) -> Vec<Complex64> {
    let (nx, ny, nz) = grid.dims;
    let vs = [grid.vsx(), grid.vsy(), grid.vsz()];
    let dims = [nx, ny, nz];
    let r2 = radius * radius;
    // MATLAB's -N/2 : N/2-1 grid, then fftshift (circshift by floor(N/2)).
    let coord = |i: usize, a: usize| (i as f64 - dims[a] as f64 / 2.0) * vs[a];
    let split = 10usize;
    let sub: Vec<f64> = (0..2 * split).map(|s| (s as f64 - split as f64 + 0.5) / (2 * split) as f64).collect();
    let mut sphere = vec![0.0f64; nx * ny * nz];
    let mut total = 0.0;
    for k in 0..nz {
        let z = coord(k, 2);
        for j in 0..ny {
            let y = coord(j, 1);
            for i in 0..nx {
                let x = coord(i, 0);
                let out = (x.abs() - 0.5 * vs[0]).max(0.0).powi(2)
                    + (y.abs() - 0.5 * vs[1]).max(0.0).powi(2)
                    + (z.abs() - 0.5 * vs[2]).max(0.0).powi(2)
                    > r2;
                let inside = (x.abs() + 0.5 * vs[0]).powi(2)
                    + (y.abs() + 0.5 * vs[1]).powi(2)
                    + (z.abs() + 0.5 * vs[2]).powi(2)
                    <= r2;
                let v = if inside {
                    1.0
                } else if out {
                    0.0
                } else {
                    let mut occupied = 0usize;
                    for &sz in &sub {
                        for &sy in &sub {
                            for &sx in &sub {
                                let (px, py, pz) = (x + sx * vs[0], y + sy * vs[1], z + sz * vs[2]);
                                if px * px + py * py + pz * pz <= r2 {
                                    occupied += 1;
                                }
                            }
                        }
                    }
                    occupied as f64 / sub.len().pow(3) as f64
                };
                if v != 0.0 {
                    let (si, sj, sk) = ((i + nx / 2) % nx, (j + ny / 2) % ny, (k + nz / 2) % nz);
                    sphere[si + nx * (sj + ny * sk)] = v;
                    total += v;
                }
            }
        }
    }
    let mut k: Vec<Complex64> = sphere.iter().map(|&v| Complex64::new(v / total, 0.0)).collect();
    fft.fft3d(&mut k);
    k
}

/// `SMV.m` with a precomputed kernel: `ifftn(fftn(x).*K)`, left in `buf`
fn smv(x: &[f64], k: &[Complex64], fft: &mut Fft3dWorkspace, buf: &mut [Complex64]) {
    let load = load_real(x);
    fft.fft3d_with(buf, &FftOpts { load: Some(&load), ..FftOpts::default() });
    fft.ifft3d_with(buf, &FftOpts { load: Some(&mul_kernel(k)), ..FftOpts::default() });
}

/// The SMV noise propagation in `MEDI_L1.m`, in place: `tempn = sqrt(SMV(tempn.^2, SphereK) + tempn.^2)`
fn smv_noise(tempn: &mut [f64], k: &[Complex64], fft: &mut Fft3dWorkspace, buf: &mut [Complex64]) {
    let t = &*tempn;
    let load = |l: usize, row: &mut [Complex64]| {
        for (c, &v) in row.iter_mut().zip(&t[l..]) {
            *c = Complex64::new(v * v, 0.0);
        }
    };
    fft.fft3d_with(buf, &FftOpts { load: Some(&load), ..FftOpts::default() });
    fft.ifft3d_with(buf, &FftOpts { load: Some(&mul_kernel(k)), ..FftOpts::default() });
    par_update(tempn, |i, v| *v = (buf[i].re + *v * *v).sqrt());
}

/// `extract_CSF.m` (MEDI toolbox 2024.11.26): the ventricular CSF mask for MEDI+0.
///
/// Low-R2* voxels (`r2s < r2s_threshold`) form 6-connected regions twice: inside a 30 mm sphere
/// at the mask's centroid, and inside the mask (eroded first, when `erode`, to where the 10 mm
/// `SMV.m` of the mask exceeds 0.999). The CSF mask is every region of the second kind that
/// touches one of the three largest regions of the first kind. `extract_CSF.m` errors when
/// there are fewer than three central regions; this uses those there are.
///
/// # Arguments
/// * `r2s` - R2* in Hz (`arlo.m`, [`arlo`](crate::utils::r2star::arlo))
/// * `mask` - Brain mask, 1 = brain
/// * `grid` - Volume grid (dimensions and voxel sizes in mm)
/// * `r2s_threshold` - `thresh_R2s` (default 5 Hz)
/// * `erode` - `flag_erode` (default on)
///
/// # Returns
/// The CSF mask, 1 = CSF
pub fn extract_csf(r2s: &[f64], mask: &[u8], grid: &Grid, r2s_threshold: f64, erode: bool) -> Vec<u8> {
    let (nx, ny, nz) = grid.dims;
    let n = grid.n_total();
    assert!(r2s.len() == n && mask.len() == n, "r2s and mask must match the grid");
    let vs = [grid.vsx(), grid.vsy(), grid.vsz()];
    let pos = |p: usize| {
        let (i, j, k) = (p % nx, (p / nx) % ny, p / (nx * ny));
        [(i + 1) as f64 * vs[0], (j + 1) as f64 * vs[1], (k + 1) as f64 * vs[2]]
    };
    // Center region (sphere): radius 30 mm about the centroid of the mask
    let inside: Vec<usize> = (0..n).filter(|&p| mask[p] > 0).collect();
    let mut cen = [0.0f64; 3];
    for &p in &inside {
        let x = pos(p);
        for a in 0..3 {
            cen[a] += x[a];
        }
    }
    let count = inside.len() as f64;
    let cen = cen.map(|c| c / count);
    let mask_cen: Vec<bool> = (0..n)
        .map(|p| {
            let x = pos(p);
            ((x[0] - cen[0]).powi(2) + (x[1] - cen[1]).powi(2) + (x[2] - cen[2]).powi(2)).sqrt() <= 30.0
        })
        .collect();
    // if flag_erode, Mask = SMV(Mask, matrix_size, voxel_size, 10) > 0.999
    let mask: Vec<bool> = if erode {
        let mut fft = Fft3dWorkspace::new(nx, ny, nz);
        let k = sphere_kernel(grid, 10.0, &mut fft);
        let m: Vec<f64> = mask.iter().map(|&v| v as f64).collect();
        let mut buf = vec![Complex64::new(0.0, 0.0); n];
        smv(&m, &k, &mut fft, &mut buf);
        buf.iter().map(|v| v.re > 0.999).collect()
    } else {
        mask.iter().map(|&v| v != 0).collect()
    };
    let low: Vec<bool> = r2s.iter().map(|&r| r < r2s_threshold).collect();

    // ROIs_region_cen: the three largest 6-connected regions of (R2s < thresh) in the sphere
    let raw1: Vec<bool> = (0..n).map(|p| low[p] && mask_cen[p]).collect();
    let (lab1, sizes1) = label6(&raw1, grid.dims);
    let mut order: Vec<usize> = (0..sizes1.len()).collect();
    order.sort_by(|&a, &b| sizes1[b].cmp(&sizes1[a])); // stable, as MATLAB's sort
    let mut central = vec![false; sizes1.len()];
    for &c in order.iter().take(3) {
        central[c] = true;
    }
    // ROIs_region: 6-connected regions of (R2s < thresh) in the (eroded) mask; keep those that
    // overlap a central region
    let raw2: Vec<bool> = (0..n).map(|p| low[p] && mask[p]).collect();
    let (lab2, sizes2) = label6(&raw2, grid.dims);
    let mut keep = vec![false; sizes2.len()];
    for p in 0..n {
        if let (Some(a), Some(b)) = (lab1[p], lab2[p]) {
            if central[a] {
                keep[b] = true;
            }
        }
    }
    // Mask_ROI_CSF(Mask == 0) = 0
    (0..n).map(|p| u8::from(mask[p] && lab2[p].is_some_and(|b| keep[b]))).collect()
}

/// 6-connected components (`bwconncomp(·, 6)`), numbered in the order of their first voxel in
/// memory order (x fastest, as MATLAB's column-major order). Returns each voxel's component and
/// the component sizes.
fn label6(fg: &[bool], dims: (usize, usize, usize)) -> (Vec<Option<usize>>, Vec<usize>) {
    let (nx, ny, nz) = dims;
    let mut label = vec![None; fg.len()];
    let mut sizes = Vec::new();
    let mut stack = Vec::new();
    for start in 0..fg.len() {
        if !fg[start] || label[start].is_some() {
            continue;
        }
        let c = sizes.len();
        let mut size = 0;
        label[start] = Some(c);
        stack.push(start);
        while let Some(p) = stack.pop() {
            size += 1;
            let (i, j, k) = (p % nx, (p / nx) % ny, p / (nx * ny));
            let mut visit = |q: usize| {
                if fg[q] && label[q].is_none() {
                    label[q] = Some(c);
                    stack.push(q);
                }
            };
            if i > 0 { visit(p - 1); }
            if i + 1 < nx { visit(p + 1); }
            if j > 0 { visit(p - nx); }
            if j + 1 < ny { visit(p + nx); }
            if k > 0 { visit(p - nx * ny); }
            if k + 1 < nz { visit(p + nx * ny); }
        }
        sizes.push(size);
    }
    (label, sizes)
}

// ============================================================================
// Single-precision MEDI-style helpers shared by TFI and the chi-separation solvers
// (not used by `medi` above)
// ============================================================================

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
    /// Create a new MEDI workspace for the given grid
    pub fn new(grid: &Grid) -> Self {
        let (nx, ny, nz) = grid.dims;
        let (vsx, vsy, vsz) = (grid.vsx() as f32, grid.vsy() as f32, grid.vsz() as f32);
        let n_total = grid.n_total();

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
pub(crate) fn apply_dipole_conv(
    fft_ws: &mut Fft3dWorkspaceF32,
    x: &[f32],
    d_kernel: &[f32],
    out: &mut [f32],
    complex_buf: &mut [Complex32],
) {
    fft_ws.apply_dipole_inplace(x, d_kernel, out, complex_buf);
}

/// MEDI operator buffers - separate struct to allow split borrowing
pub(crate) struct MediOpBuffers<'a> {
    pub gx: &'a mut [f32],
    pub gy: &'a mut [f32],
    pub gz: &'a mut [f32],
    pub reg_x: &'a mut [f32],
    pub reg_y: &'a mut [f32],
    pub reg_z: &'a mut [f32],
    pub div_buf: &'a mut [f32],
    pub dipole_buf: &'a mut [f32],
    pub complex_buf: &'a mut [Complex32],
    pub complex_buf2: &'a mut [Complex32],
}

/// Apply MEDI operator in-place: out = fidelity(dx) + lambda*reg(dx)
/// This is the hot path - called many times per Gauss-Newton iteration
/// Uses per-direction gradient masks (mx, my, mz) matching MATLAB MEDI
/// SIMD-accelerated for element-wise operations
#[inline]
pub(crate) fn apply_medi_operator_core(
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

/// Compute RHS in-place using workspace buffers (f32)
/// Uses per-direction gradient masks (mx, my, mz) matching MATLAB MEDI
/// SIMD-accelerated for element-wise operations
pub(crate) fn compute_rhs_inplace(
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
pub(crate) fn dataterm_mask_f32(mode: i32, n_std: &[f32], mask: &[u8]) -> Vec<f32> {
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
pub(crate) fn gradient_mask_f32(
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
pub(crate) fn fgrad_linext_f32(
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
pub(crate) fn fgrad_linext_inplace_f32(
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


/// Forward difference gradient with periodic boundary conditions (f32, in-place)
/// Matches MATLAB's gradfp_mex used inside MEDI iterations.
/// At boundaries, wraps around: dx(end) = (x(1) - x(end)) / h
///
/// Parallelised one rayon task per z-slab (disjoint output chunks); `x` is only
/// read, including across slab boundaries for the z-difference.
pub(crate) fn fgrad_periodic_inplace_f32(
    gx: &mut [f32], gy: &mut [f32], gz: &mut [f32],
    x: &[f32],
    nx: usize, ny: usize, nz: usize,
    vsx: f32, vsy: f32, vsz: f32,
) {
    let hx = 1.0 / vsx;
    let hy = 1.0 / vsy;
    let hz = 1.0 / vsz;
    let nxny = nx * ny;

    maybe_par_chunks_mut!(gx, nxny)
        .zip(maybe_par_chunks_mut!(gy, nxny))
        .zip(maybe_par_chunks_mut!(gz, nxny))
        .enumerate()
        .for_each(|(k, ((gx_s, gy_s), gz_s))| {
            let k_offset = k * nxny;

            for j in 0..ny {
                let j_offset = j * nx;

                for i in 0..nx {
                    let local = i + j_offset;
                    let idx = local + k_offset;
                    let x_val = x[idx];

                    // x-direction: periodic wrap at i = nx-1
                    let x_next = if i + 1 < nx { x[idx + 1] } else { x[j_offset + k_offset] };
                    gx_s[local] = (x_next - x_val) * hx;

                    // y-direction: periodic wrap at j = ny-1
                    let y_next = if j + 1 < ny { x[i + (j + 1) * nx + k_offset] } else { x[i + k_offset] };
                    gy_s[local] = (y_next - x_val) * hy;

                    // z-direction: periodic wrap at k = nz-1
                    let z_next = if k + 1 < nz { x[i + j_offset + (k + 1) * nxny] } else { x[i + j_offset] };
                    gz_s[local] = (z_next - x_val) * hz;
                }
            }
        });
}

/// Backward divergence with periodic boundary conditions (f32, in-place)
/// Adjoint of fgrad_periodic_inplace_f32, matching MATLAB's gradfp_adj_mex.
/// At boundaries, wraps around: at i=0, uses gx(end) instead of zero.
///
/// Parallelised one rayon task per z-slab (disjoint output chunks); the
/// gradient inputs are only read, including across slab boundaries.
pub(crate) fn bdiv_periodic_inplace_f32(
    div: &mut [f32],
    gx: &[f32], gy: &[f32], gz: &[f32],
    nx: usize, ny: usize, nz: usize,
    vsx: f32, vsy: f32, vsz: f32,
) {
    let hx = -1.0 / vsx;  // Negative for adjoint
    let hy = -1.0 / vsy;
    let hz = -1.0 / vsz;
    let nxny = nx * ny;

    maybe_par_chunks_mut!(div, nxny).enumerate().for_each(|(k, div_s)| {
        let k_offset = k * nxny;

        for j in 0..ny {
            let j_offset = j * nx;

            for i in 0..nx {
                let local = i + j_offset;
                let idx = local + k_offset;

                // x-direction: at i=0, wrap to gx[nx-1,j,k]
                let gx_prev = if i > 0 { gx[idx - 1] } else { gx[(nx - 1) + j_offset + k_offset] };
                let gx_term = (gx[idx] - gx_prev) * hx;

                // y-direction: at j=0, wrap to gy[i,ny-1,k]
                let gy_prev = if j > 0 { gy[i + (j - 1) * nx + k_offset] } else { gy[i + (ny - 1) * nx + k_offset] };
                let gy_term = (gy[idx] - gy_prev) * hy;

                // z-direction: at k=0, wrap to gz[i,j,nz-1]
                let gz_prev = if k > 0 { gz[i + j_offset + (k - 1) * nxny] } else { gz[i + j_offset + (nz - 1) * nxny] };
                let gz_term = (gz[idx] - gz_prev) * hz;

                div_s[local] = gx_term + gy_term + gz_term;
            }
        }
    });
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

    fn small_params() -> MediParams {
        MediParams { cg_max_iter: 20, max_iter: 3, smv: false, merit: false, ..MediParams::default() }
    }

    /// A sphere of susceptibility in a cube, its field in ppm, a magnitude with an edge at the
    /// sphere, and a spherical mask.
    fn phantom(n: usize) -> (Grid, Vec<f64>, Vec<f64>, Vec<f64>, Vec<u8>) {
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let c = n as f64 / 2.0;
        let r2 = |i: usize| {
            let (x, y, z) = ((i % n) as f64 - c, ((i / n) % n) as f64 - c, (i / (n * n)) as f64 - c);
            x * x + y * y + z * z
        };
        let chi: Vec<f64> = (0..n * n * n).map(|i| if r2(i) <= 9.0 { 0.1 } else { 0.0 }).collect();
        let d = dipole_kernel(&grid, (0.0, 0.0, 1.0));
        let dc: Vec<Complex64> = d.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        let mut fft = Fft3dWorkspace::new(n, n, n);
        let mut buf = vec![Complex64::new(0.0, 0.0); n * n * n];
        let mut field = vec![0.0; n * n * n];
        dconv(&chi, &dc, &mut fft, &mut buf, &mut field);
        let mag: Vec<f64> = (0..n * n * n).map(|i| if r2(i) <= 9.0 { 0.5 } else { 1.0 }).collect();
        let mask: Vec<u8> = (0..n * n * n).map(|i| (r2(i) <= 49.0) as u8).collect();
        (grid, chi, field, mag, mask)
    }

    #[test]
    fn defaults_are_medi_l1s() {
        let p = MediParams::default();
        assert_eq!((p.lambda, p.percentage, p.cg_tol, p.tol), (1000.0, 0.9, 0.01, 0.1));
        assert_eq!((p.cg_max_iter, p.max_iter, p.data_weighting), (100, 10, 1));
        assert!(p.smv && p.merit && p.smv_shrink_mask);
        assert_eq!(p.smv_radius, 5.0);
        // MEDI+0 as in MEDI's README: lambda_CSF 100, extract_CSF's defaults
        assert!(p.csf && p.csf_erode);
        assert_eq!((p.lambda_csf, p.csf_r2s_threshold), (100.0, 5.0));
    }

    #[test]
    fn bdiv_is_the_adjoint_of_fgrad() {
        let grid = Grid::new(5, 4, 3, 1.0, 0.8, 1.5);
        let n = grid.n_total();
        let vs = [grid.vsx(), grid.vsy(), grid.vsz()];
        let x: Vec<f64> = (0..n).map(|i| ((i * 37) % 11) as f64 - 5.0).collect();
        let g: [Vec<f64>; 3] = [0, 1, 2].map(|c| (0..n).map(|i| ((i * 13 + c * 7) % 9) as f64 - 4.0).collect());
        let mut gx = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
        fgrad(&x, grid.dims, vs, &mut gx);
        let lhs: f64 = (0..3).map(|c| gx[c].iter().zip(&g[c]).map(|(a, b)| a * b).sum::<f64>()).sum();
        let mut div = vec![0.0; n];
        bdiv(&g, grid.dims, vs, &mut div);
        let rhs: f64 = x.iter().zip(&div).map(|(a, b)| a * b).sum::<f64>();
        assert!((lhs - rhs).abs() < 1e-9, "<fgrad x, g> = {lhs}, <x, bdiv g> = {rhs}");
    }

    #[test]
    fn dconv_weighted_equals_two_plain_dconvs() {
        // The pruned Dconv(w2.*Dconv(p)) of A(dx) gives the values of the two plain Dconvs.
        let grid = Grid::new(12, 10, 8, 1.0, 1.0, 1.2);
        let n = grid.n_total();
        let (nx, ny) = (12, 10);
        let d: Vec<Complex64> = dipole_kernel(&grid, (0.1, 0.2, 0.97))
            .iter()
            .enumerate()
            .map(|(i, &v)| Complex64::new(v, 0.01 * ((i % 7) as f64 - 3.0)))
            .collect();
        let p: Vec<f64> = (0..n).map(|i| ((i as f64 * 12.9898).sin() * 43758.5453).fract()).collect();
        let mask: Vec<bool> = (0..n)
            .map(|i| (2..9).contains(&(i % nx)) && (3..8).contains(&((i / nx) % ny)) && (1..6).contains(&(i / (nx * ny))))
            .collect();
        let w2: Vec<f64> = (0..n).map(|i| if mask[i] { 1.0 + (i % 5) as f64 } else { 0.0 }).collect();
        let mut fft = Fft3dWorkspace::new(12, 10, 8);
        let mut buf = vec![Complex64::new(0.0, 0.0); n];
        let (mut t, mut f) = (vec![0.0; n], vec![0.0; n]);
        dconv(&p, &d, &mut fft, &mut buf, &mut t);
        let t: Vec<f64> = (0..n).map(|i| t[i] * w2[i]).collect();
        dconv(&t, &d, &mut fft, &mut buf, &mut f);
        let mbox = bounding_box(&mask, grid.dims);
        assert_eq!(mbox, [2..9, 3..8, 1..6]);
        dconv_weighted(&p, &d, &w2, &mbox, &mut fft, &mut buf);
        for i in 0..n {
            assert_eq!(buf[i].re, f[i], "voxel {i}");
        }
    }

    #[test]
    fn reg_equals_bdiv_of_weighted_fgrad() {
        // 9 z planes: tasks of 4 planes, so plane boundaries inside and between tasks
        let grid = Grid::new(7, 6, 9, 1.0, 0.8, 1.5);
        let n = grid.n_total();
        let vs = [grid.vsx(), grid.vsy(), grid.vsz()];
        let x: Vec<f64> = (0..n).map(|i| ((i as f64 * 12.9898).sin() * 43758.5453).fract()).collect();
        let wvw: [Vec<f64>; 3] = [0, 1, 2].map(|c| (0..n).map(|i| ((i * 13 + c * 7) % 5) as f64 * 0.3).collect());
        let mut g = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
        fgrad(&x, grid.dims, vs, &mut g);
        for c in 0..3 {
            for i in 0..n {
                g[c][i] *= wvw[c][i];
            }
        }
        let (mut want, mut got) = (vec![0.0; n], vec![0.0; n]);
        bdiv(&g, grid.dims, vs, &mut want);
        reg(&x, &wvw, grid.dims, vs, &mut got);
        assert_eq!(want, got);
    }

    #[test]
    fn fgrad_is_zero_at_the_last_voxel() {
        let grid = Grid::new(4, 4, 4, 1.0, 1.0, 1.0);
        let x: Vec<f64> = (0..64).map(|i| (i * i) as f64).collect();
        let mut g = [vec![1.0; 64], vec![1.0; 64], vec![1.0; 64]];
        fgrad(&x, grid.dims, [1.0; 3], &mut g);
        for j in 0..4 {
            for k in 0..4 {
                assert_eq!(g[0][3 + 4 * (j + 4 * k)], 0.0);
            }
        }
        assert_eq!(g[0][0], x[1] - x[0]);
    }

    #[test]
    fn gradient_mask_marks_about_percentage_of_mask_voxels_as_edges() {
        let n = 16;
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let mag: Vec<f64> = (0..n * n * n).map(|i| ((i as f64 * 12.9898).sin() * 43758.5453).fract().abs()).collect();
        let mask = vec![true; n * n * n];
        let wg = gradient_mask(&mag, &mask, &grid, 0.9);
        let edges: usize = wg.iter().map(|c| c.iter().filter(|&&v| v == 0.0).count()).sum();
        let frac = edges as f64 / (n * n * n) as f64;
        assert!((frac - 0.9).abs() < 0.02, "edge count / mask voxels = {frac}");
    }

    #[test]
    fn sphere_kernel_is_normalised_and_real() {
        let grid = Grid::new(16, 16, 16, 1.0, 1.0, 1.0);
        let mut fft = Fft3dWorkspace::new(16, 16, 16);
        let k = sphere_kernel(&grid, 3.0, &mut fft);
        assert!((k[0].re - 1.0).abs() < 1e-12, "DC = sum of the sphere = {}", k[0].re);
        assert!(k.iter().all(|v| v.im.abs() < 1e-12));
    }

    #[test]
    fn zero_field_gives_zero() {
        let (grid, _, field, mag, mask) = phantom(16);
        let zero = vec![0.0; field.len()];
        let chi = medi_l1(&zero, &mask, &mag, &vec![1.0; mag.len()], None, &grid, (0.0, 0.0, 1.0), &small_params(), |_, _| {});
        assert!(chi.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn recovers_a_sphere_in_ppm() {
        let (grid, chi_true, field, mag, mask) = phantom(24);
        let chi = medi_l1(&field, &mask, &mag, &vec![1.0; mag.len()], None, &grid, (0.0, 0.0, 1.0), &small_params(), |_, _| {});
        let c = 12 + 24 * (12 + 24 * 12);
        assert!((chi[c] - chi_true[c]).abs() < 0.03, "centre: {} vs {}", chi[c], chi_true[c]);
        for (i, &m) in mask.iter().enumerate() {
            if m == 0 {
                assert_eq!(chi[i], 0.0);
            }
        }
    }

    #[test]
    fn uniform_noise_map_equals_none() {
        let (grid, _, field, mag, mask) = phantom(16);
        let p = small_params();
        let a = medi_l1(&field, &mask, &mag, &vec![1.0; mag.len()], None, &grid, (0.0, 0.0, 1.0), &p, |_, _| {});
        let b = medi_l1(&field, &mask, &mag, &vec![3.0; field.len()], None, &grid, (0.0, 0.0, 1.0), &p, |_, _| {});
        let err = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
        assert!(err < 1e-9, "max |diff| {err}");
    }

    #[test]
    fn smv_and_merit_run() {
        let (grid, _, field, mag, mask) = phantom(16);
        let p = MediParams { smv: true, smv_radius: 2.0, merit: true, ..small_params() };
        let mut calls = 0;
        let chi = medi_l1(&field, &mask, &mag, &vec![1.0; mag.len()], None, &grid, (0.0, 0.0, 1.0), &p, |_, _| calls += 1);
        assert!(chi.iter().all(|v| v.is_finite()));
        assert!(chi.iter().any(|&v| v != 0.0));
        assert!(calls > 0);
    }

    #[test]
    fn zeropad_pads_at_the_end_and_crops() {
        let (grid, _, field, mag, mask) = phantom(15);
        let p = MediParams { zeropad: (1, 0, 1), ..small_params() };
        let chi = medi(&field, &mask, EchoMagnitudes { data: &mag, n_echoes: 1 }, &grid, (0.0, 0.0, 1.0), &p, |_, _| {});
        // the same inversion on a volume padded by hand
        let (n, m) = (15usize, 16usize);
        let pad = |v: &[f64], nyp: usize| {
            let mut out = vec![0.0; m * nyp * m];
            for k in 0..n { for j in 0..n { for i in 0..n {
                out[i + m * (j + nyp * k)] = v[i + n * (j + n * k)];
            }}}
            out
        };
        let mask_f: Vec<f64> = mask.iter().map(|&v| v as f64).collect();
        let mask_p: Vec<u8> = pad(&mask_f, n).iter().map(|&v| v as u8).collect();
        let pgrid = Grid::new(m, n, m, 1.0, 1.0, 1.0);
        let padded = medi_l1(&pad(&field, n), &mask_p, &pad(&mag, n), &pad(&crate::utils::fit_ppm_complex_noise(&mag, 1), n), None, &pgrid, (0.0, 0.0, 1.0), &small_params(), |_, _| {});
        for k in 0..n { for j in 0..n { for i in 0..n {
            assert_eq!(chi[i + n * (j + n * k)], padded[i + m * (j + n * k)]);
        }}}
    }

    #[test]
    fn echo_magnitudes_give_rss_and_the_fit_noise() {
        let (grid, _, field, mag, mask) = phantom(16);
        let n = mag.len();
        let echoes: Vec<f64> = mag.iter().copied().chain(mag.iter().map(|v| 0.5 * v)).collect();
        let p = MediParams { echo_times: vec![0.005, 0.010], ..small_params() };
        let a = medi(&field, &mask, EchoMagnitudes { data: &echoes, n_echoes: 2 }, &grid, (0.0, 0.0, 1.0), &p, |_, _| {});
        let rss: Vec<f64> = (0..n).map(|i| (echoes[i].powi(2) + echoes[i + n].powi(2)).sqrt()).collect();
        let n_std = crate::utils::fit_ppm_complex_noise(&echoes, 2);
        let b = medi_l1(&field, &mask, &rss, &n_std, None, &grid, (0.0, 0.0, 1.0), &p, |_, _| {});
        assert_eq!(a, b);
    }

    #[test]
    fn smv_pads_odd_axes_to_even() {
        let (grid, _, field, mag, mask) = phantom(15);
        let p = MediParams { smv: true, smv_radius: 2.0, merit: false, ..small_params() };
        let auto = medi(&field, &mask, EchoMagnitudes { data: &mag, n_echoes: 1 }, &grid, (0.0, 0.0, 1.0), &p, |_, _| {});
        let explicit = MediParams { zeropad: (1, 1, 1), ..p.clone() };
        let manual = medi(&field, &mask, EchoMagnitudes { data: &mag, n_echoes: 1 }, &grid, (0.0, 0.0, 1.0), &explicit, |_, _| {});
        assert_eq!(auto, manual);
        // an explicit pad that leaves an axis odd gets one more voxel
        let odd = MediParams { zeropad: (2, 0, 0), ..p.clone() };
        let more = MediParams { zeropad: (3, 1, 1), ..p };
        let a = medi(&field, &mask, EchoMagnitudes { data: &mag, n_echoes: 1 }, &grid, (0.0, 0.0, 1.0), &odd, |_, _| {});
        let b = medi(&field, &mask, EchoMagnitudes { data: &mag, n_echoes: 1 }, &grid, (0.0, 0.0, 1.0), &more, |_, _| {});
        assert_eq!(a, b);
    }

    /// A 100³ head-sized phantom: a 48 mm-radius brain at R2* 20 Hz with low-R2* regions.
    fn csf_phantom() -> (Grid, Vec<f64>, Vec<u8>, impl Fn(usize) -> [f64; 3]) {
        let n = 100;
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let at = move |p: usize| [(p % n) as f64 - 50.0, ((p / n) % n) as f64 - 50.0, (p / (n * n)) as f64 - 50.0];
        let mask: Vec<u8> = (0..n * n * n).map(|p| u8::from(dist(at(p), [0.0; 3]) <= 48.0)).collect();
        let r2s: Vec<f64> = (0..n * n * n)
            .map(|p| {
                let x = at(p);
                if is_ventricle(x) {
                    2.0
                // low R2* inside the eroded mask but 35 mm from the centre
                } else if dist(x, [0.0, 35.0, 0.0]) <= 2.0 {
                    1.0
                // a rim the 10 mm erosion removes, and the background
                } else if dist(x, [0.0; 3]) > 44.0 {
                    0.0
                } else {
                    20.0
                }
            })
            .collect();
        (grid, r2s, mask, at)
    }

    fn dist(x: [f64; 3], c: [f64; 3]) -> f64 {
        ((x[0] - c[0]).powi(2) + (x[1] - c[1]).powi(2) + (x[2] - c[2]).powi(2)).sqrt()
    }

    /// Two central "ventricles"
    fn is_ventricle(x: [f64; 3]) -> bool {
        dist(x, [-5.0, 0.0, 0.0]) <= 3.0 || dist(x, [5.0, 0.0, 0.0]) <= 2.0
    }

    #[test]
    fn extract_csf_keeps_central_low_r2s_regions_in_the_eroded_mask() {
        let (grid, r2s, mask, at) = csf_phantom();
        let csf = extract_csf(&r2s, &mask, &grid, 5.0, true);
        for (p, &c) in csf.iter().enumerate() {
            assert_eq!(c == 1, is_ventricle(at(p)), "voxel {:?}", at(p));
        }
        // Without the erosion the rim is a candidate region, but it does not touch the 30 mm
        // central sphere, so the result is the same.
        assert_eq!(extract_csf(&r2s, &mask, &grid, 5.0, false), csf);
        // a threshold below the ventricles' R2* finds nothing there
        assert!(extract_csf(&r2s, &mask, &grid, 1.5, true).iter().all(|&v| v == 0));
    }

    #[test]
    fn extract_csf_takes_the_three_largest_central_regions() {
        let n = 24;
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let mask = vec![1u8; n * n * n];
        // four separate low-R2* single-row regions of 4, 3, 2 and 1 voxels near the centre
        let mut r2s = vec![20.0; n * n * n];
        for (len, j) in [(4usize, 6usize), (3, 10), (2, 14), (1, 18)] {
            for i in 0..len {
                r2s[(10 + i) + n * (j + n * 12)] = 0.0;
            }
        }
        let csf = extract_csf(&r2s, &mask, &grid, 5.0, false);
        assert_eq!(csf.iter().filter(|&&v| v == 1).count(), 4 + 3 + 2);
        assert_eq!(csf[10 + n * (18 + n * 12)], 0);
    }

    #[test]
    fn csf_term_zero_references_the_result() {
        let (grid, _, field, mag, mask) = phantom(16);
        let n = mag.len();
        let csf: Vec<bool> = (0..n).map(|i| mask[i] == 1 && i % 7 == 0).collect();
        let p = small_params();
        let off = medi_l1(&field, &mask, &mag, &vec![1.0; n], None, &grid, (0.0, 0.0, 1.0), &p, |_, _| {});
        let on = medi_l1(&field, &mask, &mag, &vec![1.0; n], Some(&csf), &grid, (0.0, 0.0, 1.0), &p, |_, _| {});
        let mean = |x: &[f64]| (0..n).filter(|&i| csf[i]).map(|i| x[i]).sum::<f64>() / csf.iter().filter(|&&c| c).count() as f64;
        assert!(mean(&on).abs() < 1e-12, "CSF mean {}", mean(&on));
        assert!(mean(&off).abs() > 1e-6);
        for i in 0..n {
            if mask[i] == 0 {
                assert_eq!(on[i], 0.0);
            }
        }
        // with lambda_CSF = 0 the result is the plain result shifted by its CSF mean
        let p0 = MediParams { lambda_csf: 0.0, ..p };
        let shifted = medi_l1(&field, &mask, &mag, &vec![1.0; n], Some(&csf), &grid, (0.0, 0.0, 1.0), &p0, |_, _| {});
        let m = mean(&off);
        for i in 0..n {
            let want = if mask[i] == 1 { off[i] - m } else { 0.0 };
            assert!((shifted[i] - want).abs() < 1e-12);
        }
        // and the CSF term changes the solution itself
        let d = (0..n).map(|i| (on[i] - shifted[i]).abs()).fold(0.0, f64::max);
        assert!(d > 1e-6, "lambda_CSF had no effect");
    }

    #[test]
    fn fewer_than_three_echoes_run_without_the_csf_term() {
        let (grid, _, field, mag, mask) = phantom(16);
        let echoes: Vec<f64> = mag.iter().copied().chain(mag.iter().map(|v| 0.5 * v)).collect();
        let p = MediParams { echo_times: vec![0.005, 0.010], ..small_params() };
        let on = medi(&field, &mask, EchoMagnitudes { data: &echoes, n_echoes: 2 }, &grid, (0.0, 0.0, 1.0), &p, |_, _| {});
        let off = medi(&field, &mask, EchoMagnitudes { data: &echoes, n_echoes: 2 }, &grid, (0.0, 0.0, 1.0), &MediParams { csf: false, ..p }, |_, _| {});
        assert_eq!(on, off);
    }
}

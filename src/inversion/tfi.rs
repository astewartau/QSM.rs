//! TFI (preconditioned Total Field Inversion)
//!
//! Single-step QSM: inverts the total field, before background removal, over the whole field of
//! view. Background sources outside the mask are part of the solution, made cheap to reach by a
//! preconditioner `P`, and the susceptibility is `χ = P·y`:
//!
//! min_y ||wG ∇(P y)||₁ + 2λ ||m (D P y − f)||²₂
//!
//! # What follows which source
//!
//! Cornell's own TFI script (`TFI_L1.m`) is not public. The 2024.11.26 MEDI toolbox has the TFI
//! solver inside `MEDI_L1.m`: when `RDF.mat` holds a preconditioner `P` (`parse_QSM_input.m`),
//! `MEDI_L1` solves for `y` with `χ = P·y`. This port follows that code line for line:
//!
//! - **2024 `MEDI_L1.m` and its helpers** (`fgrad.m`, `bdiv.m`, `gradient_mask.m`,
//!   `dataterm_mask.m`, `cgsolve.m`, `dipole_kernel.m`): the `'linear'` data term (and the
//!   toolbox's default exponential one, as an option), the data weight `m` from `N_std`, the edge mask `wG` and then `wG.*Mask`
//!   (no regularisation outside the mask), the lagged-diffusivity weight
//!   `1/sqrt(|wG ∇(P y)|² + 1e-6·P²)` (epsilon scaled by `P²`), the operator
//!   `PH(reg(P dx)) + 2λ·PH(Dconv(|w|² Dconv(P dx)))`, the right-hand side, the Gauss-Newton
//!   stop `||dy||/||y||`, MERIT, the output `P y` in ppm masked by `Mask`, and the defaults of
//!   `parse_QSM_input.m` (λ = 1000, MERIT on, 10 Gauss-Newton and 100 CG iterations, tolerances
//!   0.1 and 0.01, `percentage` 0.9).
//! - **Liu et al. 2017**, for what the toolbox does not settle for TFI: `P = 1` inside the mask
//!   and `P_B = 30` outside (eq. 3; 30 is the paper's in-vivo value); the input is the total
//!   field over the whole field of view, so there is no SMV step (an SMV high-pass would remove
//!   the background TFI is meant to fit); and the data term is the linear one of eq. 5, which is
//!   `MEDI_L1`'s `'linear'` option rather than its exponential default. (With the exponential
//!   term, `MEDI_L1` itself fails on in-vivo total fields in radians: the field wraps inside
//!   `exp(i·)`. Set `linear: false` to run the toolbox default.)
//!
//! Not ported: the R2*-based preconditioner (eq. 4), zero padding and the CSF zero reference.
//!
//! # Units
//! Like the other inversions, `tfi` takes the field in ppm and returns ppm. `MEDI_L1` works on the
//! field in radians accumulated over the echo spacing (`delta_TE`) at `CF = γ·B0`; the field is
//! converted with `2π·CF·delta_TE·1e-6` first and back at the end, as `MEDI_L1.m` does. The
//! result depends on this scale, so `b0` and `echo_times` are scan parameters
//! that the pipeline fills from metadata, and [`TfiParams::te`] picks the echo time the
//! conversion uses (by default the reference's choice; see there).
//!
//! # Magnitude and data weight
//! As for [`medi`](super::medi), the echo magnitudes ([`EchoMagnitudes`]) are required: the edge
//! mask uses their root sum of squares (`iMag = sqrt(sum(abs(iField).^2,4))`) and the data weight
//! `m = Mask./N_std` (mean 1 in the mask) uses `Fit_ppm_complex.m`'s noise map
//! ([`fit_ppm_complex_noise`](crate::utils::fit_ppm_complex_noise)), as MEDI's pipeline makes
//! them, so TFI always weights the data as the reference does.
//!
//! Reference:
//! Liu, Z., Kee, Y., Zhou, D., Wang, Y., Spincemaille, P. (2017). "Preconditioned total field
//! inversion (TFI) method for quantitative susceptibility mapping." Magnetic Resonance in
//! Medicine, 78(1):303-315. https://doi.org/10.1002/mrm.26331
//!
//! Reference implementation: MEDI toolbox 2024.11.26 (Weill Cornell), `functions/MEDI_L1.m`.

use num_complex::Complex64;
use crate::fft::Fft3dWorkspace;
use crate::kernels::dipole::dipole_kernel;
use crate::{EchoTime, Grid};

use super::medi::{
    bounding_box, cgsolve, dataterm_mask, dconv, dconv_weighted, fgrad, gradient_mask, par_fill,
    par_update, reg, CgBuffers, EchoMagnitudes,
};

/// TFI parameters (2024 `MEDI_L1.m` / `parse_QSM_input.m` defaults; `P_B` and the linear data
/// term from Liu et al. 2017)
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct TfiParams {
    /// Weight of the data-fidelity term (`'lambda'`): larger fits the data more closely
    pub lambda: f64,
    /// Preconditioner outside the mask, `P_B` (1 inside); Liu et al. 2017 use 30 in vivo
    pub precond: f64,
    /// MERIT: reweight the noise by the data residual after each iteration (`'merit'`)
    pub merit: bool,
    /// Linear data term `||m (D P y − f)||²` (`'linear'`, Liu et al. 2017 eq. 5); `false` gives
    /// `MEDI_L1`'s default exponential term `||m (exp(i D P y) − exp(i f))||²`
    pub linear: bool,
    /// Data weighting: 0 uniform (1 everywhere), 1 SNR (`mask / n_std`, mean 1 in the mask)
    pub data_weighting: i32,
    /// Edge fraction for the gradient mask (`'percentage'`)
    pub percentage: f64,
    /// CG relative residual tolerance
    pub cg_tol: f64,
    /// Maximum CG iterations per Gauss-Newton step
    pub cg_max_iter: usize,
    /// Maximum Gauss-Newton iterations
    pub max_iter: usize,
    /// Stop when ||dy|| / ||y|| falls below this
    pub tol: f64,
    /// Field strength in Tesla (scan parameter, for the radian conversion)
    pub b0: f64,
    /// Echo times in seconds, in acquisition order (scan parameter, for the radian
    /// conversion).
    pub echo_times: Vec<f64>,
    /// Which echo time the radian conversion uses. Default [`EchoTime::Spacing`]: the MEDI
    /// toolbox's `MEDI_L1.m` works on the field in radians over `delta_TE`
    /// (`x/(2*pi*delta_TE*CF)*1e6`), which its readers set to `TE(2) - TE(1)`, or TE for one echo
    /// (`Read_Siemens_DICOM.m` lines 176-178).
    pub te: EchoTime,
}

impl Default for TfiParams {
    fn default() -> Self {
        Self {
            lambda: 1000.0,
            precond: 30.0,
            merit: true,
            linear: true,
            data_weighting: 1,
            percentage: 0.9,
            cg_tol: 0.01,
            cg_max_iter: 100,
            max_iter: 10,
            tol: 0.1,
            b0: 3.0,
            echo_times: vec![0.005],
            te: EchoTime::Spacing,
        }
    }
}

/// Preconditioned total field inversion (2024 `MEDI_L1.m` with a preconditioner)
///
/// # Arguments
/// * `total_field` - Total field in ppm, before background removal (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz), 1 = brain: `P = 1` and the data are fitted here
/// * `magnitude` - Echo magnitudes, or a combined magnitude (see module docs)
/// * `grid` - Volume grid (dimensions and voxel sizes)
/// * `bdir` - B0 field direction
/// * `params` - TFI parameters, including field strength and echo spacing
/// * `progress` - Progress callback `(current_step, total_steps)` over CG iterations
///
/// # Returns
/// Susceptibility `P·y` in ppm, zero outside the mask
pub fn tfi(
    total_field: &[f64],
    mask: &[u8],
    magnitude: EchoMagnitudes,
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &TfiParams,
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
    tfi_l1(total_field, mask, &imag, &n_std, grid, bdir, params, progress)
}

/// `MEDI_L1.m` with the preconditioner, on an `iMag` and an `N_std`
#[allow(clippy::too_many_arguments)]
fn tfi_l1(
    total_field: &[f64],
    mask: &[u8],
    magnitude: &[f64],
    n_std: &[f64],
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &TfiParams,
    mut progress: impl FnMut(usize, usize),
) -> Vec<f64> {
    let n = grid.n_total();
    let vs = [grid.vsx(), grid.vsy(), grid.vsz()];
    let dims = grid.dims;
    let mut fft = Fft3dWorkspace::new(grid.nx(), grid.ny(), grid.nz());
    let mut buf = vec![Complex64::new(0.0, 0.0); n];

    // ppm -> radians over delta_TE; MEDI_L1 converts back with x/(2*pi*delta_TE*CF)*1e6.
    let cf = crate::constants::GAMMA_BAR_HZ_PER_T * params.b0;
    let ppm_to_rad = 2.0 * std::f64::consts::PI * params.te.seconds(&params.echo_times, "TFI") * cf * 1e-6;
    let mut rdf = vec![0.0f64; n];
    par_fill(&mut rdf, |i| total_field[i] * ppm_to_rad);

    let mask: Vec<bool> = mask.iter().map(|&m| m != 0).collect();
    // P = 1*Mask + P_B*(1-Mask) (Liu et al. 2017, eq. 3). P is real, so PH = P.
    let mut p = vec![0.0f64; n];
    par_fill(&mut p, |i| if mask[i] { 1.0 } else { params.precond });

    // N_std = N_std.*Mask; tempn = N_std
    let mut n_std: Vec<f64> = (0..n).map(|i| if mask[i] { n_std[i] } else { 0.0 }).collect();

    let d: Vec<Complex64> = dipole_kernel(grid, bdir).iter().map(|&v| Complex64::new(v, 0.0)).collect();
    let mut m = vec![0.0f64; n];
    dataterm_mask(params.data_weighting, &n_std, &mask, &mut m);
    // opts.wG = gradient_mask(...); with a preconditioner, opts.wG = wG.*Mask
    let mut wg = gradient_mask(magnitude, &mask, grid, params.percentage);
    for c in wg.iter_mut() {
        par_update(c, |i, v| {
            if !mask[i] {
                *v = 0.0;
            }
        });
    }
    // m, and with it w2, is zero outside the mask (SNR weighting), so the Dconv(P dx) in A(dx)
    // is only needed, and the Dconv after the weighting only has input, inside its bounding box.
    let mbox = if params.data_weighting == 0 {
        [0..grid.nx(), 0..grid.ny(), 0..grid.nz()]
    } else {
        bounding_box(&mask, dims)
    };

    let linear = params.linear;
    // B = m.*exp(1i*RDF), or m.*RDF with 'linear'
    let data = |m: &[f64], b0: &mut [Complex64]| {
        par_fill(b0, |i| {
            if linear { Complex64::new(m[i] * rdf[i], 0.0) } else { m[i] * Complex64::new(0.0, rdf[i]).exp() }
        });
    };
    // W = m.*exp(1i*Dconv(x)), or m.*Dconv(x) with 'linear'
    let model = |m: f64, dpx: f64| -> Complex64 {
        if linear { Complex64::new(m * dpx, 0.0) } else { m * Complex64::new(0.0, dpx).exp() }
    };
    let mut b0 = vec![Complex64::new(0.0, 0.0); n];
    data(&m, &mut b0);

    // e = 0.000001, scaled with P: e*PH(P(ones(matrix_size)))
    let e: Vec<f64> = p.iter().map(|&v| 1e-6 * v * v).collect();
    let lambda2 = 2.0 * params.lambda;

    let mut x = vec![0.0f64; n];
    let mut px = vec![0.0f64; n];
    let mut pq = vec![0.0f64; n];
    // wG.*Vr.*wG per direction
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
        // w = W(opts.m, P(x))
        par_fill(&mut px, |i| p[i] * x[i]);
        dconv(&px, &d, &mut fft, &mut buf, &mut t);
        par_fill(&mut w, |i| model(m[i], t[i]));
        // reg_MEDI(opts, P(x), PHP): Vr = 1./sqrt(abs(wG.*grad(real(P(x)))).^2 + e)
        fgrad(&px, dims, vs, &mut g);
        for c in 0..3 {
            let (wgc, gc) = (&wg[c], &g[c]);
            par_fill(&mut wvw[c], |i| {
                let u = wgc[i] * gc[i];
                let vr = 1.0 / (u * u + e[i]).sqrt();
                wgc[i] * vr * wgc[i]
            });
        }
        // fidelity weight: conj(opts.m).*opts.m (linear) or conj(w).*w
        if linear {
            par_fill(&mut w2, |i| m[i] * m[i]);
        } else {
            par_fill(&mut w2, |i| w[i].norm_sqr());
        }

        // b = PH(reg(P(x))) + 2*lambda*PH(Dconv(real(conj(w).*conj(1i).*(w-b0))))
        // or, with 'linear', b = PH(reg(P(x))) + 2*lambda*PH(Dconv(real(conj(opts.m).*(w-b0))))
        par_fill(&mut rv, |i| {
            if linear {
                m[i] * (w[i] - b0[i]).re
            } else {
                (w[i].conj() * Complex64::new(0.0, -1.0) * (w[i] - b0[i])).re
            }
        });
        dconv(&rv, &d, &mut fft, &mut buf, &mut t);
        reg(&px, &wvw, dims, vs, &mut b);
        par_update(&mut b, |i, v| *v = -(p[i] * *v + lambda2 * p[i] * t[i]));

        // A = @(dx) PH(reg(P(dx))) + 2*lambda*PH(Dconv(w2.*Dconv(P(dx)))); dx = cgsolve(A, -b)
        let base = (iter - 1) * params.cg_max_iter;
        cgsolve(
            |q: &[f64], out: &mut [f64]| {
                par_fill(&mut pq, |i| p[i] * q[i]);
                reg(&pq, &wvw, dims, vs, out);
                dconv_weighted(&pq, &d, &w2, &mbox, &mut fft, &mut buf);
                par_update(out, |i, v| *v = p[i] * *v + lambda2 * p[i] * buf[i].re);
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
            // wres = W(opts.m, P(x)) - b0, used inside the mask only
            par_fill(&mut px, |i| p[i] * x[i]);
            dconv(&px, &d, &mut fft, &mut buf, &mut t);
            let wres: Vec<Complex64> = mask_idx.iter().map(|&i| model(m[i], t[i]) - b0[i]).collect();
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
            dataterm_mask(params.data_weighting, &n_std, &mask, &mut m);
            data(&m, &mut b0);
        }
    }
    progress(total, total);

    // x = P(x)/(2*pi*delta_TE*CF)*1e6.*Mask
    par_update(&mut x, |i, v| *v = if mask[i] { p[i] * *v / ppm_to_rad } else { 0.0 });
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> TfiParams {
        TfiParams { cg_max_iter: 10, max_iter: 3, ..TfiParams::default() }
    }

    fn cube_mask(n: usize, lo: usize, hi: usize) -> Vec<u8> {
        (0..n * n * n)
            .map(|i| {
                let (x, y, z) = (i % n, (i / n) % n, i / (n * n));
                u8::from((lo..hi).contains(&x) && (lo..hi).contains(&y) && (lo..hi).contains(&z))
            })
            .collect()
    }

    #[test]
    fn zero_field_gives_zero() {
        let n = 8;
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let mask = cube_mask(n, 2, 6);
        let mag: Vec<f64> = (0..n * n * n).map(|i| 1.0 + (i % 5) as f64).collect();
        let chi = tfi_l1(&vec![0.0; n * n * n], &mask, &mag, &vec![1.0; mag.len()], &grid, (0.0, 0.0, 1.0), &small(), |_, _| {});
        assert!(chi.iter().all(|v| v.abs() < 1e-12));
    }

    #[test]
    fn output_is_masked_and_finite() {
        let n = 8;
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let mask = cube_mask(n, 2, 6);
        let field: Vec<f64> = (0..n * n * n).map(|i| 0.01 * ((i % 7) as f64 - 3.0)).collect();
        let mag: Vec<f64> = (0..n * n * n).map(|i| 1.0 + (i % 5) as f64).collect();
        for linear in [false, true] {
            let p = TfiParams { linear, ..small() };
            let chi = tfi_l1(&field, &mask, &mag, &vec![1.0; mag.len()], &grid, (0.0, 0.0, 1.0), &p, |_, _| {});
            assert!(chi.iter().any(|&v| v != 0.0));
            for i in 0..chi.len() {
                assert!(chi[i].is_finite());
                if mask[i] == 0 {
                    assert_eq!(chi[i], 0.0);
                }
            }
        }
    }

    /// The field scale matters (λ is fixed and, with the exponential term, the data term is
    /// nonlinear), so `b0` and `te` change the result.
    #[test]
    fn units_follow_b0_and_te() {
        let n = 8;
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let mask = cube_mask(n, 2, 6);
        let field: Vec<f64> = (0..n * n * n).map(|i| 0.05 * ((i % 7) as f64 - 3.0)).collect();
        let mag: Vec<f64> = (0..n * n * n).map(|i| 1.0 + (i % 5) as f64).collect();
        let a = tfi_l1(&field, &mask, &mag, &vec![1.0; mag.len()], &grid, (0.0, 0.0, 1.0), &small(), |_, _| {});
        let p7 = TfiParams { b0: 7.0, ..small() };
        let b = tfi_l1(&field, &mask, &mag, &vec![1.0; mag.len()], &grid, (0.0, 0.0, 1.0), &p7, |_, _| {});
        assert_ne!(a, b);
    }

    /// Echo magnitudes give the RSS edge mask and the Fit_ppm_complex noise map.
    #[test]
    fn echo_magnitudes_give_rss_and_the_fit_noise() {
        let n = 8;
        let nt = n * n * n;
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let mask = cube_mask(n, 2, 6);
        let field: Vec<f64> = (0..nt).map(|i| 0.01 * ((i % 7) as f64 - 3.0)).collect();
        let mag: Vec<f64> = (0..nt).map(|i| 1.0 + (i % 5) as f64).collect();
        let echoes: Vec<f64> = mag.iter().copied().chain(mag.iter().map(|v| 0.5 * v + (v % 2.0))).collect();
        let p = small();
        let a = tfi(&field, &mask, EchoMagnitudes { data: &echoes, n_echoes: 2 }, &grid, (0.0, 0.0, 1.0), &p, |_, _| {});
        let rss: Vec<f64> = (0..nt).map(|i| (echoes[i].powi(2) + echoes[i + nt].powi(2)).sqrt()).collect();
        let n_std = crate::utils::fit_ppm_complex_noise(&echoes, 2);
        let b = tfi_l1(&field, &mask, &rss, &n_std, &grid, (0.0, 0.0, 1.0), &p, |_, _| {});
        assert_eq!(a, b);
    }
}

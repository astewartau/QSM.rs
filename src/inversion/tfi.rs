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
//! # Parity
//! Against TFI as the verified QSM-CI `tfi-cornell` container runs it (MEDI 2024.11.26 `MEDI_L1.m`
//! with a preconditioner, identical to licensed MATLAB R2023b), astewartau/QSM.rs#407. Two
//! departures are deliberate, each with the original's behaviour as an option:
//! - **Dipole kernel on odd grids** ([`TfiKernel`]). `dipole_kernel.m` builds its k grid as
//!   `-N/2:N/2-1`, half a frequency step off the DFT grid on an odd axis, so it is not the dipole
//!   kernel there. On QSM-CI's simulated head (164×205×205) that alone made the port's χ 0.77× the
//!   original's (42 % NRMSE); with [`TfiKernel::MediOddGrid`] 2.5 %. The DFT-grid kernel fits the
//!   true χ better (demeaned, in the mask: NRMSE 62.1 % against 71.4 %, r 0.785 against 0.718), so it
//!   stays the default. On even grids the two are the same.
//! - **Precision** ([`TfiPrecision`]). `MEDI_L1.m` runs in single precision (`tempn =
//!   single(N_std)` makes the data weight single, and MATLAB's mixed arithmetic carries that into
//!   everything). On the fixture's small odd phantom (`tests/tfi_cornell_parity.rs`) the port with
//!   both options is 2.9e-6 relative after one Gauss-Newton step and 4.7e-4 with all defaults; in
//!   double precision 4.7e-3.
//!
//! Beyond that the original is not reproducible at full size: its 100-iteration CG solves at
//! `cg_tol` 0.01 in single precision amplify rounding. In the same container, 1e-7 relative noise
//! on the simulated head's total field moves its χ by 1.6 % NRMSE, and the container on another
//! CPU (QSM-CI's runner against this VM) differs from itself by 2.8 % on the KKI travelling head.
//! On the reference's own inputs the port with both options is 3.6 % (sim) and 1.26 % (KKI) from
//! it; with the DFT-grid kernel in double precision, 42 % and 1.25 %.
//!
//! # Reference pipeline
//! From Liu et al. 2017 (Methods, in-vivo experiments); not stated above:
//! - Brain mask (the `M` where `P = 1`): BET. Field map: nonlinear (complex) field estimation
//!   followed by graph-cut phase unwrapping. No background removal or high-pass before TFI.
//! - In vivo the paper uses the R2*-based preconditioner of eq. 4, `P = 1` in `M ∩ {R2* < 30 s⁻¹}`
//!   and `P_B` elsewhere, with `P_B` = 30 (10 for the phantom). Only eq. 3 is ported here.
//! - Data: 3 T, 6 echoes, TE₁ 3.5 ms, ΔTE 3.6 ms, 1×1×3 mm³, monopolar readout.
//! - The paper's solver settings differ from `MEDI_L1`'s defaults, which this port uses: λ = 1e-3
//!   on the L1 term (not the data term), Gauss-Newton stop `||dy||/||y|| < 0.01`, up to 300 CG
//!   iterations. CSF is the reference for reported values.
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
//!
//! # Pipeline rules
//!
//! The facts above that a configuration can depart from are published as rules in
//! [`crate::pipeline::rules::RULES`] (each with its strength, basis, reason and source): `tfi.mask-bet`, `tfi.te-spacing`.

use num_complex::Complex64;
use crate::fft::Fft3dWorkspace;
use crate::kernels::dipole::{dipole_kernel, dipole_kernel_medi};
use crate::{EchoTime, Grid};

use super::medi::{
    bounding_box, cgsolve, dataterm_mask, dconv, dconv_weighted, fgrad, gradient_mask, par_fill,
    par_update, reg, CgBuffers, EchoMagnitudes,
};

/// Which dipole kernel [`tfi`] uses: they differ only on grids with an odd number of voxels
/// along some axis.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TfiKernel {
    /// `1/3 − (k·B0)²/|k|²` sampled on the DFT frequency grid ([`dipole_kernel`]), zero at DC,
    /// on every grid. The default.
    #[default]
    Dft,
    /// `dipole_kernel.m` exactly as MATLAB builds it ([`dipole_kernel_medi`]): on an odd axis its
    /// `-N/2:N/2-1` grid samples half a frequency step off the DFT grid, with no zero at DC. The
    /// parity option for the MEDI toolbox's TFI on odd grids (astewartau/QSM.rs#407).
    MediOddGrid,
}

/// Arithmetic of [`tfi`]'s Gauss-Newton iterations.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TfiPrecision {
    /// Double precision throughout. The default.
    #[default]
    Double,
    /// MATLAB's single precision, as `MEDI_L1.m` ends up running: it makes the data weight from
    /// `tempn = single(N_std)`, and MATLAB's mixed arithmetic then makes the weight, the data,
    /// every operator output, every CG vector and the iterate single (an array op with one
    /// single operand is single). Here: every array is f32, with f32 FFTs, and the double
    /// operands (`P`, the edge mask, `ε`, after MERIT the reweighted data weight) are rounded to
    /// single where they meet a single one; inner products are accumulated in double and
    /// rounded. The parity option for the MEDI toolbox's TFI (astewartau/QSM.rs#407).
    Single,
}

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
    /// Dipole kernel (default [`TfiKernel::Dft`]). `MEDI_L1.m` calls `dipole_kernel.m`, whose
    /// k-space grid is half a step off on axes with an odd number of voxels; there it is not the
    /// dipole field. On QSM-CI's simulated head (164×205×205) that scales χ by ~1.3 against the
    /// DFT-grid kernel and fits the true χ worse (in the mask, demeaned: NRMSE 71.4 % against
    /// 62.1 %, r 0.718 against 0.785), so the DFT-grid kernel is the default and
    /// [`TfiKernel::MediOddGrid`] the parity option. On even grids they are the same.
    pub kernel: TfiKernel,
    /// Arithmetic (default [`TfiPrecision::Double`]). `MEDI_L1.m` runs in single precision, by
    /// accident of `tempn = single(N_std)` (see [`TfiPrecision::Single`]); the CG solves at
    /// `cg_tol` 0.01 then stop at different points from double precision, which moves χ by about
    /// 1 % (KKI travelling head) from the same inputs. Double precision solves the same equations
    /// more accurately, so it is the default and single precision the parity option.
    pub precision: TfiPrecision,
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
            kernel: TfiKernel::Dft,
            precision: TfiPrecision::Double,
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

    // D = dipole_kernel(matrix_size, voxel_size, B0_dir): see TfiKernel
    let dk = match params.kernel {
        TfiKernel::Dft => dipole_kernel(grid, bdir),
        TfiKernel::MediOddGrid => {
            let n = (bdir.0 * bdir.0 + bdir.1 * bdir.1 + bdir.2 * bdir.2).sqrt();
            dipole_kernel_medi(grid, (bdir.0 / n, bdir.1 / n, bdir.2 / n))
        }
    };
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
    if params.precision == TfiPrecision::Single {
        return tfi_single(&rdf, &mask, &p, n_std, &dk, &wg, grid, ppm_to_rad, params, progress);
    }
    // m, and with it w2, is zero outside the mask (SNR weighting), so the Dconv(P dx) in A(dx)
    // is only needed, and the Dconv after the weighting only has input, inside its bounding box.
    let d: Vec<Complex64> = dk.iter().map(|&v| Complex64::new(v, 0.0)).collect();
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

/// [`tfi_l1`] in MATLAB's single precision ([`TfiPrecision::Single`]): `MEDI_L1.m` line for line
/// with MATLAB's types. `tempn = single(N_std)` makes the data weight `m`, and with it the data
/// `b0 = m.*RDF`, single; from there every product with a single operand is single, so the
/// operator, the right-hand side, `cgsolve.m`'s vectors and scalars, `dx` and the iterate `x`
/// (`x + dx`) are single. Double operands (`P`, the edge mask, `ε`, the kernel, and after MERIT
/// the data weight and data, which `dataterm_mask` remakes from `double(N_std)`) are rounded to
/// single where they meet a single one. FFTs are single precision.
#[allow(clippy::too_many_arguments)]
fn tfi_single(
    rdf: &[f64],
    mask: &[bool],
    p: &[f64],
    mut n_std: Vec<f64>,
    dk: &[f64],
    wg: &[Vec<f64>; 3],
    grid: &Grid,
    ppm_to_rad: f64,
    params: &TfiParams,
    mut progress: impl FnMut(usize, usize),
) -> Vec<f64> {
    use num_complex::Complex32;
    let n = grid.n_total();
    let (nx, ny, nz) = grid.dims;
    let vs = [grid.vsx() as f32, grid.vsy() as f32, grid.vsz() as f32];
    let mut fft = crate::fft::Fft3dWorkspaceF32::new(nx, ny, nz);
    let mut buf = vec![Complex32::new(0.0, 0.0); n];
    let d: Vec<f32> = dk.iter().map(|&v| v as f32).collect();
    let p32: Vec<f32> = p.iter().map(|&v| v as f32).collect();
    let wg32: Vec<Vec<f32>> = wg.iter().map(|c| c.iter().map(|&v| v as f32).collect()).collect();
    // e = 0.000001*PH(P(ones)) (reg_MEDI.m), double, rounded where it meets single
    let e: Vec<f32> = p.iter().map(|&v| (1e-6 * v * v) as f32).collect();
    let lambda2 = (2.0 * params.lambda) as f32;
    let linear = params.linear;

    // Dconv = @(dx) real(ifftn(D.*fftn(dx))), single
    let mut dconv = |x: &[f32], out: &mut [f32]| {
        for (c, &v) in buf.iter_mut().zip(x) {
            *c = Complex32::new(v, 0.0);
        }
        fft.fft3d(&mut buf);
        for (c, &dv) in buf.iter_mut().zip(&d) {
            *c *= dv;
        }
        fft.ifft3d(&mut buf);
        for (o, c) in out.iter_mut().zip(&buf) {
            *o = c.re;
        }
    };
    let nxy = nx * ny;
    // fgrad.m, single
    let fgrad = |x: &[f32], g: &mut [Vec<f32>; 3]| {
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let q = i + nx * j + nxy * k;
                    g[0][q] = if i + 1 < nx { (x[q + 1] - x[q]) / vs[0] } else { 0.0 };
                    g[1][q] = if j + 1 < ny { (x[q + nx] - x[q]) / vs[1] } else { 0.0 };
                    g[2][q] = if k + 1 < nz { (x[q + nxy] - x[q]) / vs[2] } else { 0.0 };
                }
            }
        }
    };
    // bdiv.m, single
    let bdiv = |g: &[Vec<f32>; 3], out: &mut [f32]| {
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let q = i + nx * j + nxy * k;
                    let tx = (if i + 1 < nx { g[0][q] } else { 0.0 }) - (if i > 0 { g[0][q - 1] } else { 0.0 });
                    let ty = (if j + 1 < ny { g[1][q] } else { 0.0 }) - (if j > 0 { g[1][q - nx] } else { 0.0 });
                    let tz = (if k + 1 < nz { g[2][q] } else { 0.0 }) - (if k > 0 { g[2][q - nxy] } else { 0.0 });
                    out[q] = -(tx / vs[0] + ty / vs[1] + tz / vs[2]);
                }
            }
        }
    };
    // inner products: accumulated in double, returned single
    let dot = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum::<f64>() as f32 };

    // m = dataterm_mask(mode, single(N_std), Mask): single
    let dataterm_single = |n_std: &[f64]| -> Vec<f32> {
        if params.data_weighting == 0 {
            return vec![1.0; n];
        }
        let mut w: Vec<f32> = (0..n)
            .map(|i| {
                let v = (mask[i] as u8 as f32) / (n_std[i] as f32);
                if mask[i] && v.is_finite() { v } else { 0.0 }
            })
            .collect();
        let count = mask.iter().filter(|&&m| m).count() as f64;
        let mean = (w.iter().zip(mask).filter(|(_, &m)| m).map(|(&v, _)| v as f64).sum::<f64>() / count) as f32;
        w.iter_mut().for_each(|v| *v /= mean);
        w
    };
    let mut m: Vec<f32> = dataterm_single(&n_std);
    // b0 = B(m, RDF): m.*RDF (linear) or m.*exp(1i*RDF), single
    let data = |m: &[f32]| -> Vec<Complex32> {
        (0..n)
            .map(|i| {
                if linear { Complex32::new(m[i] * rdf[i] as f32, 0.0) } else { m[i] * Complex32::new(0.0, rdf[i] as f32).exp() }
            })
            .collect()
    };
    let mut b0 = data(&m);

    let mut x = vec![0.0f32; n];
    let mut px = vec![0.0f32; n];
    let mut t = vec![0.0f32; n];
    let mut t2 = vec![0.0f32; n];
    let mut g = [vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]];
    let mut wvw = [vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]];
    let mut w = vec![Complex32::new(0.0, 0.0); n];
    let mut w2 = vec![0.0f32; n];
    let mut b = vec![0.0f32; n];
    let mut rv = vec![0.0f32; n];
    let (mut cx, mut cr, mut cd, mut cq) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
    let mut pq = vec![0.0f32; n];
    let mut regout = vec![0.0f32; n];
    let model = |m: f32, dpx: f32| -> Complex32 {
        if linear { Complex32::new(m * dpx, 0.0) } else { m * Complex32::new(0.0, dpx).exp() }
    };
    let mask_idx: Vec<usize> = (0..n).filter(|&i| mask[i]).collect();
    let total = params.max_iter * params.cg_max_iter;
    let mut res_norm_ratio = f32::INFINITY;
    let mut iter = 0;
    while res_norm_ratio > params.tol as f32 && iter < params.max_iter {
        iter += 1;
        for i in 0..n {
            px[i] = p32[i] * x[i];
        }
        dconv(&px, &mut t);
        for i in 0..n {
            w[i] = model(m[i], t[i]);
        }
        // reg_MEDI: Vr = 1./sqrt(abs(wG.*grad(real(P(x)))).^2 + e)
        fgrad(&px, &mut g);
        for c in 0..3 {
            for i in 0..n {
                let u = wg32[c][i] * g[c][i];
                let vr = 1.0 / (u * u + e[i]).sqrt();
                wvw[c][i] = wg32[c][i] * vr * wg32[c][i];
            }
        }
        if linear {
            for i in 0..n {
                w2[i] = m[i] * m[i];
            }
        } else {
            for i in 0..n {
                w2[i] = w[i].norm_sqr();
            }
        }
        // reg(dx) = PH(div(wG.*Vr.*wG.*grad(real(P(dx)))))
        let regop = |q: &[f32], out: &mut [f32], pq: &mut [f32], g: &mut [Vec<f32>; 3]| {
            for i in 0..n {
                pq[i] = p32[i] * q[i];
            }
            fgrad(pq, g);
            for c in 0..3 {
                for i in 0..n {
                    g[c][i] *= wvw[c][i];
                }
            }
            bdiv(g, out);
            for i in 0..n {
                out[i] *= p32[i];
            }
        };
        // b = reg(x) + 2*lambda*PH(Dconv(real(conj(m).*(w-b0)))) (or the exponential term's)
        for i in 0..n {
            rv[i] = if linear {
                m[i] * (w[i] - b0[i]).re
            } else {
                (w[i].conj() * Complex32::new(0.0, -1.0) * (w[i] - b0[i])).re
            };
        }
        dconv(&rv, &mut t);
        regop(&x, &mut b, &mut pq, &mut g);
        for i in 0..n {
            b[i] = -(b[i] + lambda2 * (p32[i] * t[i]));
        }
        // A = @(dx) reg(dx) + 2*lambda*PH(Dconv(w2.*Dconv(P(dx)))); dx = cgsolve(A, -b)
        let mut apply = |q: &[f32], out: &mut [f32]| {
            regop(q, &mut regout, &mut pq, &mut g);
            for i in 0..n {
                pq[i] = p32[i] * q[i];
            }
            dconv(&pq, &mut t);
            for i in 0..n {
                t[i] *= w2[i];
            }
            dconv(&t, &mut t2);
            for i in 0..n {
                out[i] = regout[i] + lambda2 * (p32[i] * t2[i]);
            }
        };
        // cgsolve.m, single
        let base = (iter - 1) * params.cg_max_iter;
        let tol = params.cg_tol as f32;
        cx.fill(0.0);
        cr.copy_from_slice(&b);
        cd.copy_from_slice(&b);
        let mut delta = dot(&cr, &cr);
        let delta0 = dot(&b, &b);
        let mut numiter = 0;
        while numiter < params.cg_max_iter && delta > tol * tol * delta0 {
            apply(&cd, &mut cq);
            let alpha = delta / dot(&cd, &cq);
            for i in 0..n {
                cx[i] += alpha * cd[i];
            }
            if (numiter + 1) % 50 == 0 {
                apply(&cx, &mut cq);
                for i in 0..n {
                    cr[i] = b[i] - cq[i];
                }
            } else {
                for i in 0..n {
                    cr[i] -= alpha * cq[i];
                }
            }
            let delta_old = delta;
            delta = dot(&cr, &cr);
            let beta = delta / delta_old;
            for i in 0..n {
                cd[i] = cr[i] + beta * cd[i];
            }
            numiter += 1;
            progress(base + numiter, total);
        }
        // res_norm_ratio = norm(dx)/norm(x); x = x + dx
        res_norm_ratio = dot(&cx, &cx).sqrt() / dot(&x, &x).sqrt();
        for i in 0..n {
            x[i] += cx[i];
        }

        if params.merit {
            for i in 0..n {
                px[i] = p32[i] * x[i];
            }
            dconv(&px, &mut t);
            // wres = W(m, P(x)) - b0, single; mean and std over the mask in single
            let wres: Vec<Complex32> = mask_idx.iter().map(|&i| model(m[i], t[i]) - b0[i]).collect();
            let cnt = mask_idx.len() as f64;
            let mean = {
                let s = wres.iter().fold(num_complex::Complex64::new(0.0, 0.0), |a, v| a + num_complex::Complex64::new(v.re as f64, v.im as f64));
                Complex32::new((s.re / cnt) as f32, (s.im / cnt) as f32)
            };
            let a: Vec<f32> = wres.iter().map(|&v| (v - mean).norm()).collect();
            let am = (a.iter().map(|&v| v as f64).sum::<f64>() / cnt) as f32;
            let sd = ((a.iter().map(|&v| ((v - am) as f64).powi(2)).sum::<f64>() / (cnt - 1.0)).sqrt()) as f32;
            let factor = sd * 6.0;
            // N_std(Mask==1) = N_std(Mask==1).*wres(Mask==1).^2: single product, stored double
            for (q, &i) in mask_idx.iter().enumerate() {
                let r = (a[q] / factor).max(1.0);
                n_std[i] = ((n_std[i] as f32) * (r * r)) as f64;
            }
            // tempn = double(N_std); m = dataterm_mask(...) in double, b0 = B(m, RDF) in double,
            // both rounded to single where they meet the single iterate
            let mut md = vec![0.0f64; n];
            dataterm_mask(params.data_weighting, &n_std, mask, &mut md);
            m = md.iter().map(|&v| v as f32).collect();
            b0 = (0..n)
                .map(|i| {
                    if linear {
                        Complex32::new((md[i] * rdf[i]) as f32, 0.0)
                    } else {
                        let z = md[i] * num_complex::Complex64::new(0.0, rdf[i]).exp();
                        Complex32::new(z.re as f32, z.im as f32)
                    }
                })
                .collect();
        }
    }
    progress(total, total);
    let scale = ppm_to_rad as f32;
    (0..n).map(|i| if mask[i] { (p32[i] * x[i] / scale) as f64 } else { 0.0 }).collect()
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

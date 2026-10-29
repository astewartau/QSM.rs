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
//! fills from metadata, and [`MediParams::te`] picks the echo time (by default `delta_TE`).
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
//! # Input: the local field
//! `medi` takes the local field from a background removal, with or without SMV or mSMV. The
//! MEDI toolbox's pipeline (`README.m`) runs `RDF = PDF(iFreq, N_std, Mask, ...)` and then
//! `MEDI_L1('lambda',1000,'lambda_CSF',100,'merit','smv',5)` on `RDF`. Its naming convention
//! calls `RDF` the "Relative Difference Field, aka local field", and `MEDI_L1` reads `iFreq`
//! (the total field) without using it. The mSMV paper (Roberts et al. 2024, Eqs. 4 and 13)
//! defines MEDI-SMV on the local field after PDF, `b_L0 = b̂_L − S·M·b̂_L` on the eroded mask,
//! and SEPIA's MEDI wrapper passes MEDI its local field. MEDI's SMV step only removes the
//! residual background that a background removal leaves near the edge. The pipeline therefore
//! routes MEDI through a background removal
//! ([`InversionConfig::input`](crate::pipeline::InversionConfig::input)), PDF in the authors'
//! pipeline ([`reference_bg_removal`](crate::pipeline::InversionConfig::reference_bg_removal)).
//!
//! Until #287, QSM.rs gave MEDI-SMV the total field and the brain mask instead. The two routes
//! were compared with MEDI's defaults (λ 1000, MERIT, MEDI+0) and QSM.rs's PDF. Data: the
//! qsm-forward 7 T head phantom (TE 4/12/20/28 ms), noise-free from its true total field and at
//! SNR 50 through QSM.rs's field mapping, scored over the brain mask; and four 3 T EMTP
//! travelling-head scans in QSM-CI (Philips Achieva, Siemens Skyra, Siemens Prisma, Philips
//! Ingenia), scored by the χ variance in the SynthSeg cortex (the mSMV paper's shadow score):
//!
//! | input → MEDI | phantom, noise-free: XSIM / NRMSE / corr | phantom, SNR 50 | in vivo cortex χ variance (×1e-4 ppm²), 4 scans |
//! |---|---|---|---|
//! | total field → MEDI-SMV (before #287) | 0.563 / 62.6 % / 0.780 | 0.352 / 76.2 % / 0.657 | 4.76 / 8.99 / 4.12 / 5.71 |
//! | PDF → MEDI-SMV (authors) | 0.563 / 62.6 % / 0.780 | 0.352 / 76.2 % / 0.658 | 4.86 / 9.09 / 4.14 / 5.76 |
//! | total field → MEDI-SMV, no shrinking | 0.233 / 140.6 % / 0.458 | 0.079 / 315.0 % / 0.207 | 850 / 383 / 875 / 983 |
//! | PDF → MEDI-SMV, no shrinking | 0.558 / 65.1 % / 0.760 | 0.292 / 94.0 % / 0.511 | 6.9 / 12.7 / 9.1 / 11.4 |
//!
//! With the default mask shrinking the two routes give nearly the same χ: the SMV high-pass
//! removes any harmonic field inside the eroded mask, and the total field differs from PDF's
//! local field by a field that is harmonic there. The maps differ by 0.5 ppb RMS on the phantom
//! (relative L2 1.3 %) and 1.5–4.9 ppb in vivo (4–14 %). Without shrinking, the edge band keeps
//! the background field on the total-field route, and the result is unusable. The authors'
//! route costs a PDF run, 10–80 s on these volumes.
//!
//! # SMV options
//! What `MEDI_L1.m` does with each combination of `'smv'`/`'nosmv'`, `smv_shrink_mask` and
//! `'msmv'` (`SphereK` is `sphere_kernel.m`'s kernel, `SMV(x) = ifftn(SphereK·fftn(x))`):
//!
//! | options | mask | field `RDF` | kernel `D` | noise `tempn` |
//! |---|---|---|---|---|
//! | `'nosmv'` | unchanged | unchanged | `D` | `N_std` |
//! | `'smv'`, shrink on (default) | `SMV(Mask) > 0.999` | `Mask·(RDF − SMV(RDF))` | `(1 − SphereK)·D` | `sqrt(SMV(tempn²) + tempn²)` |
//! | `'smv'`, shrink off, release 2024.11.26 | unchanged | `Mask·(RDF − SMV(RDF))` | `(1 − SphereK)·D` | `sqrt(SMV(tempn²) + tempn²)` |
//! | `'smv'`, shrink off, mSMV prerelease | unchanged | **unchanged** | `(1 − SphereK)·D` | `sqrt(SMV(tempn²) + tempn²)` |
//! | `'msmv'` (prerelease only; forces shrink off) | unchanged | `msmv(RDF, Mask, R2s, ...)` | `(1 − SphereK)·D` | `sqrt(SMV(tempn²) + tempn²)` |
//!
//! `medi` follows the 2024.11.26 release for the first three rows ([`MediParams::smv`],
//! [`MediParams::smv_shrink_mask`]) and the prerelease for `'msmv'` ([`MediParams::msmv`]),
//! which only the prerelease has. The prerelease (`MEDI_toolbox-mSMV100125.zip`) moved the
//! field high-pass inside `if opts.smv_shrink_mask && ~opts.msmv`, so with SMV on and shrinking
//! off it fits the unfiltered field with the SMV-filtered kernel. Its comments and the mSMV
//! paper do not describe this case: the paper's MEDI-SMV (Eq. 13) is the shrinking one. The
//! two releases agree in every other row and, apart from the mSMV code, in the rest of
//! `MEDI_L1.m`. In both, `'msmv'`/`'smv'` set `smv = 1`. In the prerelease, `'nosmv'` given
//! after `'msmv'` turns SMV off and with it mSMV, a combination `medi` does not have:
//! [`MediParams::msmv`] implies SMV.
//!
//! Checked against both toolboxes in MATLAB R2023b (`tests/medi_smv_parity.rs`, a 32×36×28
//! phantom, no CSF term). The two `MEDI_L1.m` versions give identical χ with `'nosmv'` and with
//! `'smv'` and shrinking. With shrinking off they differ by 101 % relative L2 after one step
//! and 176 % with all defaults. `medi` against the release, χ over the mask:
//!
//! | options | one Gauss-Newton step: max abs diff (ppm), rel. L2 | all defaults (10 steps) |
//! |---|---|---|
//! | `'nosmv'` | 2.5e-7, 1.1e-6 | 1.4e-3, 1.9e-3 |
//! | `'smv'`, shrink on | 1.1e-8, 2.1e-6 | 2.9e-8, 5.8e-7 |
//! | `'smv'`, shrink off | 1.7e-7, 5.7e-7 | 2.0e-4, 2.1e-4 |
//!
//! After one step the differences are single (MATLAB) against double (here) rounding. Over
//! ten steps they also include CG stopping one iteration apart at `cg_tol` 0.01.
//!
//! # MEDI-mSMV
//! [`MediParams::msmv`] is `MEDI_L1(..., 'msmv', radius)` from the mSMV authors' MEDI toolbox
//! (agr78/mSMV, `prerelease/MEDI_toolbox-mSMV100125.zip`; Roberts et al. 2024, Eq. 12): SMV is
//! on with radius [`MediParams::smv_radius`], but instead of eroding the mask and high-passing
//! the field, `MEDI_L1` passes the local field and the full mask to `msmv.m`
//! ([`msmv`](super::msmv::msmv)), then inverts the result with the SMV-filtered dipole kernel
//! `(1 − SphereK)·D` and the SMV-propagated noise, as MEDI-SMV does. The mask stays whole, so the
//! result reaches the brain edge. `msmv.m` needs R2* for its vessel mask, and `MEDI_L1` errors
//! without it; `medi` computes it from the echo magnitudes with `arlo.m`
//! ([`arlo_unclipped`](crate::utils::arlo_unclipped)), as MEDI's pipeline and the authors' SEPIA
//! patch do, so MEDI-mSMV needs at least three echoes (and panics with fewer; the pipeline
//! returns an error). `msmv.m`'s field strength `B0_mag` is [`MediParams::b0`]. MEDI-mSMV takes
//! the local field from a background removal (the paper: PDF, V-SHARP and LBV), not the total
//! field. See [`msmv`](mod@super::msmv) for what the filter does and how it compares with the
//! authors' MATLAB.
//!
//! **Deliberate deviation from `MEDI_L1.m`:** with SMV on, `sphere_kernel.m` builds its grid as
//! `-N/2:N/2-1`, which along an axis with an odd number of voxels centres the sphere half a voxel
//! off the grid. The SMV operator is then not symmetric, and `MEDI_L1` diverges (in MATLAB as
//! here: NRMSE above 10000% on a 164×205×205 phantom). So when SMV is on, `medi` pads every axis
//! that is still odd after `zeropad` by one more voxel and crops the result back. On grids that
//! are even after `zeropad`, and with SMV off, nothing changes and the result is `MEDI_L1`'s.
//!
//! # Reference pipeline
//! What MEDI's own pipeline script (`README.m` of the 2024.11.26 release; the release has no
//! `MEDI_full_process.m`) runs around `MEDI_L1`. Inputs, units and the CSF term are above.
//! - Field map: `Fit_ppm_complex(iField)` on the complex echoes (evenly spaced; README.m:87),
//!   which also gives `N_std`; `Fit_ppm_complex_bipolar` for bipolar readouts (README.m:98).
//!   Unwrapping: `unwrapPhase(iMag, iFreq_raw, matrix_size)`, MEDI's region-growing unwrapper
//!   (README.m:93; graph-cut and Laplacian unwrapping are commented alternatives).
//! - Brain mask: FSL BET via the bundled `bet2` MEX, `BET(iMag, matrix_size, voxel_size)` on the
//!   root-sum-of-squares magnitude (README.m:118), with BET's defaults (fractional threshold 0.5,
//!   gradient threshold 0; `functions/bet2/bet2.cpp:136`). `genMask.m` (magnitude threshold) is a
//!   commented alternative.
//! - Background removal: PDF, `PDF(iFreq, N_std, Mask, matrix_size, voxel_size, B0_dir)` with
//!   `PDF.m`'s defaults (README.m:124). LBV with `tol` 0.005 is the commented alternative
//!   (README.m:129). `MEDI_L1` then inverts this local field (`RDF`), so with SMV on, the SMV
//!   high-pass is applied on top of the PDF local field, not to the total field.
//! - Inversion: `MEDI_L1('lambda',1000,'lambda_CSF',100,'merit','smv',5)` (README.m:144), which
//!   equals the 2024 defaults (`parse_QSM_input.m:10-20`: MERIT and SMV on, radius 5 mm). The
//!   2020 release's `parse_QSM_input.m` defaulted to MERIT and SMV off.
//! - ppm conversion: `delta_TE` is `TE(2) − TE(1)`, or `TE` for one echo (`Read_DICOM.m:166-168`),
//!   and `CF` is read from the DICOM header; QSM.rs uses `CF = γ·B0`.
//! - Field strength: nothing in `MEDI_L1`, `PDF`, `extract_CSF` or `arlo` depends on it beyond
//!   `CF`; λ = 1000 and `extract_CSF`'s 5 Hz R2* threshold are the same at 3 T and 7 T.
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
//! The double-precision helpers below are shared with [`tfi`](super::tfi), which is `MEDI_L1.m`
//! run with a preconditioner, and with the χ-separation ports.
//!
//! # Pipeline rules
//!
//! The facts above that a configuration can depart from are published as rules in
//! [`crate::pipeline::rules::RULES`] (each with its strength, basis, reason and source): `medi.mask-bet`, `medi.field-complex-fit`, `medi.te-spacing`, `medi-msmv.three-echoes`, `medi-msmv.bg-removal`, `medi-msmv.prefilter`, `medi.bg-pdf`.

use num_complex::Complex64;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use crate::fft::{Box3, Fft3dWorkspace, FftOpts};
use crate::kernels::dipole::dipole_kernel;
use crate::{EchoTime, Grid};

/// MEDI parameters (`MEDI_L1.m` / `parse_QSM_input.m` defaults)
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct MediParams {
    /// Weight of the data-fidelity term (`'lambda'`): larger fits the data more closely
    pub lambda: f64,
    /// MERIT: reweight the noise by the data residual after each iteration (`'merit'`)
    pub merit: bool,
    /// MEDI-SMV (`'smv'`; `false` is `'nosmv'`): high-pass the local field with the spherical
    /// mean value kernel, `RDF − SMV(RDF)`, and fit it with the SMV-filtered dipole kernel
    /// `(1 − SphereK)·D` and the SMV-propagated noise `sqrt(SMV(N²) + N²)`. This removes the
    /// residual background a background removal leaves near the edge. It is applied to the
    /// local field, not the total field (see the module docs). On by default, as in
    /// `MEDI_L1.m`.
    pub smv: bool,
    /// SMV radius in mm
    pub smv_radius: f64,
    /// With SMV, erode the mask to where the SMV of the mask exceeds 0.999 (`smv_shrink_mask`,
    /// default on). With it off, the field is still high-passed, the kernel filtered and the
    /// noise propagated, on the full mask, as the 2024.11.26 `MEDI_L1.m` does. The mSMV
    /// prerelease's `MEDI_L1.m` skips the field high-pass in this case and keeps the rest; `medi`
    /// follows the release (table in the module docs). The edge band within the SMV radius then
    /// holds `RDF − SMV(RDF)` of a sphere that reaches outside the mask. Ignored without SMV,
    /// and with [`msmv`](Self::msmv), which keeps the mask, as `MEDI_L1` does.
    pub smv_shrink_mask: bool,
    /// MEDI-mSMV (`MEDI_L1(..., 'msmv', smv_radius)`): replace the SMV step's mask erosion and
    /// field high-pass with [`msmv`](super::msmv::msmv) on the full mask, keeping the
    /// SMV-filtered dipole kernel and noise. Takes the local field from a background removal
    /// (the authors used PDF, V-SHARP and LBV) on its mask; implies SMV (radius
    /// [`smv_radius`](Self::smv_radius)). Needs at least three echoes: the vessel mask uses R2*
    /// from `arlo.m` on the echo magnitudes, as MEDI's pipeline computes it. `None` (default) is
    /// plain MEDI or MEDI-SMV. See the module docs.
    pub msmv: Option<super::msmv::MsmvParams>,
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
    /// picks one with [`te`](Self::te); the CSF mask uses R2* from all of them (`arlo.m`).
    pub echo_times: Vec<f64>,
    /// Which echo time the radian conversion uses. Default [`EchoTime::Spacing`], MEDI's
    /// `delta_TE`: `MEDI_L1.m` works on the field in radians over `delta_TE`
    /// (`x/(2*pi*delta_TE*CF)*1e6`), which its readers set to `TE(2) − TE(1)`, or `TE` for one
    /// echo (`Read_Siemens_DICOM.m` lines 176-178).
    pub te: EchoTime,
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
            msmv: None,
            data_weighting: 1,
            percentage: 0.9,
            cg_tol: 0.01,
            cg_max_iter: 100,
            max_iter: 10,
            tol: 0.1,
            b0: 3.0,
            echo_times: vec![0.005],
            te: EchoTime::Spacing,
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
/// Susceptibility in ppm, zero outside the mask (the SMV-eroded mask when `smv` is on without
/// `msmv`)
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
    // MEDI-mSMV's vessel mask needs R2s (MEDI_L1 errors without it); MEDI's pipeline and the
    // authors' SEPIA patch compute it as arlo(TE, abs(iField)), negative values and all.
    let r2s_msmv: Option<Vec<f64>> = params.msmv.as_ref().map(|_| {
        crate::utils::r2star::arlo_unclipped(m, te, n)
            .unwrap_or_else(|e| panic!("MEDI-mSMV needs R2* from three or more echoes: {e}"))
    });
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
    let smv_on = params.smv || params.msmv.is_some();
    let even = |n: usize, z: usize| z + usize::from(smv_on && (n + z) % 2 == 1);
    let (px, py, pz) = (even(nx, zx), even(ny, zy), even(nz, zz));
    if (px, py, pz) == (0, 0, 0) {
        return medi_l1(
            local_field, mask, &imag, &n_std, csf.as_deref(), r2s_msmv.as_deref(), grid, bdir, params, progress,
        );
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
    let r2s_p = r2s_msmv.as_deref().map(pad);
    let chi = medi_l1(
        &pad(local_field), &mask_p, &pad(&imag), &pad(&n_std), csf_p.as_deref(), r2s_p.as_deref(),
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
    r2s: Option<&[f64]>,
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &MediParams,
    mut progress: impl FnMut(usize, usize),
) -> Vec<f64> {
    let n = grid.n_total();
    let vs = [grid.vsx(), grid.vsy(), grid.vsz()];
    let mut fft = Fft3dWorkspace::new(grid.nx(), grid.ny(), grid.nz());
    let mut buf = vec![Complex64::new(0.0, 0.0); n];

    // ppm -> radians over delta_TE (by default); MEDI_L1 converts back with
    // x/(2*pi*delta_TE*CF)*1e6.
    let cf = crate::constants::GAMMA_BAR_HZ_PER_T * params.b0;
    let delta_te = params.te.seconds(&params.echo_times, "MEDI");
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

    let sphere_k = if params.smv || params.msmv.is_some() {
        let k = sphere_kernel(grid, params.smv_radius, &mut fft);
        if let Some(mp) = &params.msmv {
            // if opts.msmv: RDF = msmv(RDF, Mask, opts.R2s, voxel_size, radius, B0_mag, ...),
            // on the full mask (smv_shrink_mask is off). B0_mag: the scan's field strength
            // (see the msmv module docs).
            let r2s = r2s.expect("MEDI-mSMV needs R2*");
            let mask_u8: Vec<u8> = mask.iter().map(|&b| u8::from(b)).collect();
            rdf = super::msmv::msmv(&rdf, &mask_u8, r2s, grid, params.smv_radius, params.b0, mp).field;
        } else {
            // if opts.smv_shrink_mask, Mask = SMV(Mask, SphereK) > 0.999
            if params.smv_shrink_mask {
                let m: Vec<f64> = mask.iter().map(|&b| b as u8 as f64).collect();
                smv(&m, &k, &mut fft, &mut buf);
                par_fill(&mut mask, |i| buf[i].re > 0.999);
            }
            // RDF = RDF - SMV(RDF, SphereK); RDF = RDF.*Mask, with or without shrinking, as
            // in the 2024.11.26 release. The mSMV prerelease does this only when shrinking
            // (see the SMV options table in the module docs).
            smv(&rdf, &k, &mut fft, &mut buf);
            par_update(&mut rdf, |i, v| *v = if mask[i] { *v - buf[i].re } else { 0.0 });
        }
        // D = (1-SphereK).*D
        par_update(&mut d, |i, v| *v *= Complex64::new(1.0, 0.0) - k[i]);
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
pub(crate) fn par_fill<T: Send>(out: &mut [T], f: impl Fn(usize) -> T + Sync + Send) {
    #[cfg(feature = "parallel")]
    out.par_iter_mut().with_min_len(1 << 14).enumerate().for_each(|(i, v)| *v = f(i));
    #[cfg(not(feature = "parallel"))]
    out.iter_mut().enumerate().for_each(|(i, v)| *v = f(i));
}

/// `f(i, &mut out[i])` for every `i` (over threads with the `parallel` feature)
pub(crate) fn par_update<T: Send>(out: &mut [T], f: impl Fn(usize, &mut T) + Sync + Send) {
    #[cfg(feature = "parallel")]
    out.par_iter_mut().with_min_len(1 << 14).enumerate().for_each(|(i, v)| f(i, v));
    #[cfg(not(feature = "parallel"))]
    out.iter_mut().enumerate().for_each(|(i, v)| f(i, v));
}

/// The smallest box holding every `true` voxel (the whole grid if there is none)
pub(super) fn bounding_box(mask: &[bool], (nx, ny, nz): (usize, usize, usize)) -> Box3 {
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
pub(super) fn dconv(x: &[f64], d: &[Complex64], fft: &mut Fft3dWorkspace, buf: &mut [Complex64], out: &mut [f64]) {
    let load = load_real(x);
    fft.fft3d_with(buf, &FftOpts { load: Some(&load), ..FftOpts::default() });
    fft.ifft3d_with(buf, &FftOpts { load: Some(&mul_kernel(d)), ..FftOpts::default() });
    par_fill(out, |i| buf[i].re);
}

/// `Dconv(w2.*Dconv(p))`, left in the real part of `buf`. `w2` is zero outside `mbox`, so the
/// inner `Dconv` is computed only on `mbox` and the outer one reads input only there; the
/// values are those of the two plain `Dconv`s (see [`Fft3dWorkspace::fft3d_with`]).
pub(super) fn dconv_weighted(
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
pub(crate) fn fgrad(x: &[f64], dims: (usize, usize, usize), vs: [f64; 3], g: &mut [Vec<f64>; 3]) {
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
pub(crate) fn reg(x: &[f64], wvw: &[Vec<f64>; 3], dims: (usize, usize, usize), vs: [f64; 3], out: &mut [f64]) {
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
pub(crate) struct CgBuffers {
    pub(crate) x: Vec<f64>,
    r: Vec<f64>,
    d: Vec<f64>,
    q: Vec<f64>,
}

impl CgBuffers {
    pub(crate) fn new(n: usize) -> Self {
        Self { x: vec![0.0; n], r: vec![0.0; n], d: vec![0.0; n], q: vec![0.0; n] }
    }
}

/// `cgsolve.m` from a zero start; the last iterate is left in `cg.x`. `progress` receives the
/// iteration count. The dot products are summed in index order, as before.
pub(crate) fn cgsolve(
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
pub(crate) fn dataterm_mask(mode: i32, n_std: &[f64], mask: &[bool], w: &mut [f64]) {
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
pub(crate) fn gradient_mask(mag: &[f64], mask: &[bool], grid: &Grid, percentage: f64) -> [Vec<f64>; 3] {
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
        // the smallest nonzero gradient, where the count can no longer change; with no nonzero
        // gradient at all, at once (0.95·level stalls at the smallest subnormal, never 0).
        let min_positive = sorted.iter().copied().find(|&v| v > 0.0).unwrap_or(f64::INFINITY);
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

/// `sphere_kernel.m`'s sphere in image space: a sphere of `radius` mm with partial-volume edge
/// voxels (each split into 20³ sub-voxels), on MATLAB's `-N/2:N/2-1` grid, `fftshift`ed
/// (circshift by `floor(N/2)`) and normalised to sum 1. Returned sparsely as `(flat index,
/// value)` for the nonzero voxels, in column-major order. Only voxels within one voxel of the
/// sphere are visited, so small radii (mSMV's `kernel_lim.m` and its minimum-radius kernel) are
/// cheap; the values are those of the full-grid loop.
pub(crate) fn sphere_kernel_sparse(grid: &Grid, radius: f64) -> Vec<(usize, f64)> {
    let (nx, ny, nz) = grid.dims;
    let vs = [grid.vsx(), grid.vsy(), grid.vsz()];
    let dims = [nx, ny, nz];
    let r2 = radius * radius;
    // MATLAB's -N/2 : N/2-1 grid, then fftshift (circshift by floor(N/2)).
    let coord = |i: usize, a: usize| (i as f64 - dims[a] as f64 / 2.0) * vs[a];
    // Voxels whose centre is further than radius + one voxel from the origin are outside.
    let range = |a: usize| {
        let reach = (radius / vs[a]).ceil() + 2.0;
        let c = dims[a] as f64 / 2.0;
        let lo = (c - reach).floor().max(0.0) as usize;
        let hi = ((c + reach).ceil() as usize).min(dims[a] - 1);
        lo..=hi
    };
    let split = 10usize;
    let sub: Vec<f64> = (0..2 * split).map(|s| (s as f64 - split as f64 + 0.5) / (2 * split) as f64).collect();
    let mut out = Vec::new();
    for k in range(2) {
        let z = coord(k, 2);
        for j in range(1) {
            let y = coord(j, 1);
            for i in range(0) {
                let x = coord(i, 0);
                let outside = (x.abs() - 0.5 * vs[0]).max(0.0).powi(2)
                    + (y.abs() - 0.5 * vs[1]).max(0.0).powi(2)
                    + (z.abs() - 0.5 * vs[2]).max(0.0).powi(2)
                    > r2;
                let inside = (x.abs() + 0.5 * vs[0]).powi(2)
                    + (y.abs() + 0.5 * vs[1]).powi(2)
                    + (z.abs() + 0.5 * vs[2]).powi(2)
                    <= r2;
                let v = if inside {
                    1.0
                } else if outside {
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
                    out.push((si + nx * (sj + ny * sk), v));
                }
            }
        }
    }
    let total: f64 = out.iter().map(|&(_, v)| v).sum();
    for e in out.iter_mut() {
        e.1 /= total;
    }
    out
}

/// `sphere_kernel(matrix_size, voxel_size, radius)` with the mSMV release's `sphere_kernel.m`
/// (`real(fftn(fftshift(Sphere)))`): the real part of the FFT of [`sphere_kernel_sparse`], in
/// double, as `SMV.m` builds it from a radius (`SMV(x, matrix_size, voxel_size, r)`).
pub(crate) fn sphere_kernel_real(grid: &Grid, radius: f64) -> Vec<f64> {
    let mut full = vec![0.0f64; grid.n_total()];
    for (p, v) in sphere_kernel_sparse(grid, radius) {
        full[p] = v;
    }
    crate::fft::fft_real_kernel(&full, grid.nx(), grid.ny(), grid.nz())
}

/// [`sphere_kernel_real`] rounded to single precision: `single(sphere_kernel(...))`, as
/// `msmv.m` (and `MEDI_L1.m`) store the prefilter kernel `SphereK`.
pub(crate) fn sphere_kernel_single(grid: &Grid, radius: f64) -> Vec<f64> {
    sphere_kernel_real(grid, radius).into_iter().map(|v| v as f32 as f64).collect()
}

/// `sphere_kernel.m` as `MEDI_L1.m` and `extract_CSF.m` use it here: `fftn(fftshift(Sphere))`
/// in double, complex (the MEDI toolbox's `sphere_kernel.m` before the mSMV release added
/// `real(...)`; the two differ only on odd grids, which `medi` pads away)
fn sphere_kernel(grid: &Grid, radius: f64, fft: &mut Fft3dWorkspace) -> Vec<Complex64> {
    let mut k = vec![Complex64::new(0.0, 0.0); grid.n_total()];
    for (p, v) in sphere_kernel_sparse(grid, radius) {
        k[p] = Complex64::new(v, 0.0);
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A constant magnitude has no gradient anywhere: the level search must stop (MATLAB's loops
    /// forever) and nothing is an edge.
    #[test]
    fn gradient_mask_of_a_constant_magnitude_terminates_with_no_edges() {
        let grid = Grid::new(6, 6, 6, 1.0, 1.0, 1.0);
        let g = gradient_mask(&[1.0; 216], &[true; 216], &grid, 0.9);
        assert!(g.iter().all(|c| c.iter().all(|&v| v == 1.0)));
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
        let chi = medi_l1(&zero, &mask, &mag, &vec![1.0; mag.len()], None, None, &grid, (0.0, 0.0, 1.0), &small_params(), |_, _| {});
        assert!(chi.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn recovers_a_sphere_in_ppm() {
        let (grid, chi_true, field, mag, mask) = phantom(24);
        let chi = medi_l1(&field, &mask, &mag, &vec![1.0; mag.len()], None, None, &grid, (0.0, 0.0, 1.0), &small_params(), |_, _| {});
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
        let a = medi_l1(&field, &mask, &mag, &vec![1.0; mag.len()], None, None, &grid, (0.0, 0.0, 1.0), &p, |_, _| {});
        let b = medi_l1(&field, &mask, &mag, &vec![3.0; field.len()], None, None, &grid, (0.0, 0.0, 1.0), &p, |_, _| {});
        let err = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
        assert!(err < 1e-9, "max |diff| {err}");
    }

    #[test]
    fn smv_and_merit_run() {
        let (grid, _, field, mag, mask) = phantom(16);
        let p = MediParams { smv: true, smv_radius: 2.0, merit: true, ..small_params() };
        let mut calls = 0;
        let chi = medi_l1(&field, &mask, &mag, &vec![1.0; mag.len()], None, None, &grid, (0.0, 0.0, 1.0), &p, |_, _| calls += 1);
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
        let padded = medi_l1(&pad(&field, n), &mask_p, &pad(&mag, n), &pad(&crate::utils::fit_ppm_complex_noise(&mag, 1), n), None, None, &pgrid, (0.0, 0.0, 1.0), &small_params(), |_, _| {});
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
        let b = medi_l1(&field, &mask, &rss, &n_std, None, None, &grid, (0.0, 0.0, 1.0), &p, |_, _| {});
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
        let off = medi_l1(&field, &mask, &mag, &vec![1.0; n], None, None, &grid, (0.0, 0.0, 1.0), &p, |_, _| {});
        let on = medi_l1(&field, &mask, &mag, &vec![1.0; n], Some(&csf), None, &grid, (0.0, 0.0, 1.0), &p, |_, _| {});
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
        let shifted = medi_l1(&field, &mask, &mag, &vec![1.0; n], Some(&csf), None, &grid, (0.0, 0.0, 1.0), &p0, |_, _| {});
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

//! Truncated k-space division (TKD) / TSVD for QSM
//!
//! TKD is the simplest dipole inversion method. It directly divides the
//! field in k-space by the dipole kernel, with truncation to avoid
//! division by small values near the magic angle.
//!
//! Reference:
//! Shmueli, K., de Zwart, J.A., van Gelderen, P., Li, T.Q., Dodd, S.J., Duyn, J.H. (2009).
//! "Magnetic susceptibility mapping of brain tissue in vivo using MRI phase data."
//! Magnetic Resonance in Medicine, 62(6):1510-1522. https://doi.org/10.1002/mrm.22135
//!
//! Reference implementation: https://github.com/kamesy/QSM.jl
//!
//! QSM.jl is this method's reference: its authors released no code, and QSM.jl, Kames's maintained successor to
//! QSM.m, is the implementation designated for this family (QSM.rs's reference rule: the authors' code where it
//! exists, otherwise one designated implementation per family, the others' values recorded).
//!
//! QSM.jl pads the field to its fast FFT size first (`pad = (0, 0, 0)`, `padfastfft`,
//! `src/inversion/direct.jl:182-184`), and so does this port by default ([`TkdParams::pad`]).
//!
//! # Agreement with QSM.jl
//!
//! On the QSM-CI in-vivo travelling-head scan (Siemens Prisma 2.89 T, 1 mm, 184×256×128, V-SHARP
//! local field and brain mask), against QSM.jl in the QSM-CI `julia-qsm` image (`sha256:b3c048ac…`,
//! QSM.jl fde4b92), both outputs saved as float32:
//! identical (NRMSE 0) for `tkd` and `tsvd` at QSM.jl's defaults and with padding off on both.
//! Before this port padded, its defaults differed from QSM.jl's by NRMSE 7.5 % (TKD) and 15.6 %
//! (TSVD). `tests/inversion_qsmjl_parity.rs` checks both against QSM.jl's output on small
//! phantoms, including tilted B0 on even sizes.
//!
//! # Half spectrum (tilted B0)
//!
//! QSM.jl applies its k-space multipliers on FFTW's real-to-complex half spectrum and inverts
//! with `irfft`. With B0 tilted from the grid axes the dipole kernel is not even on the Nyquist
//! planes of an even axis (`D(k) ≠ D(−k)` there, since the index of −½ maps to itself), and
//! `irfft` resolves that differently from a full complex FFT and its real part. This port uses
//! the multiplier `irfft` implies ([`crate::fft::c2r_kernel`]/[`crate::fft::c2r_spectrum`],
//! astewartau/QSM.rs#329), which reproduces QSM.jl there too. Before (2026-10-29) the full-FFT
//! port differed from QSM.jl on the oblique KKI travelling head (Philips Achieva, B0 tilted) by
//! 0.5–1.2 % NRMSE for TKD, TSVD, Tikhonov and RTS (QSMxT/QSM-CI parity workflow);
//! `tests/inversion_qsmjl_parity.rs` now has even-sized oblique cases.
//!
//! # Reference pipeline
//! QSM.jl (commit fde4b92) `tkd`/`tsvd` (`src/inversion/direct.jl` L1-96), threshold 0.15 for
//! both: TKD replaces `1/D` by `sign(D)/thr` where `|D| <= thr`, TSVD by 0. This port follows
//! that truncation exactly (Shmueli 2009 gives no code; QSM.m's `tkd` is the same). QSM.jl's README
//! pipeline (L35-52: BET on the last echo, 5-voxel erosion, Laplacian unwrapping, V-SHARP) is the
//! one its inversions are shown with; see [`super::rts`](mod@super::rts).
//!
//! # Default threshold: 0.25 for TKD (QSM.jl: 0.15), 0.15 for TSVD
//!
//! The threshold is tuning, not algorithm: [`TkdParams::default`] uses 0.25 (QSMxT decision 125,
//! 2026-11-12), and the QSM.jl parity tests pass QSM.jl's 0.15 explicitly ([`QSMJL_THRESHOLD`]).
//! TSVD keeps QSM.jl's 0.15 ([`TsvdParams::default`]): only TKD was re-tuned.
//!
//! Evidence. QSMxT's qsm-qc visual ratings preferred 0.25; the default was changed only after
//! 0.25 also scored better on the QSM challenge data (QSMxT `invert`, local field given, threshold
//! swept 0.1–0.4; QSM-CI `qsm_eval`; XSIM primary). XSIM / NRMSE (%) at 0.15 → 0.25:
//!
//! | Data (reference)                          | XSIM          | NRMSE         | DGM `|1−slope|` |
//! |-------------------------------------------|---------------|---------------|-----------------|
//! | 2016 challenge (χ33)                      | 0.222 → 0.303 | 123.6 → 97.6  | –               |
//! | 2016 challenge (COSMOS)                   | 0.352 → 0.469 | 99.3 → 74.7   | –               |
//! | Challenge 2.0 Sim1 SNR1                   | 0.431 → 0.516 | 98.7 → 78.2   | 0.104 → 0.200   |
//! | Challenge 2.0 Sim1 SNR2                   | 0.575 → 0.629 | 78.3 → 65.7   | 0.103 → 0.199   |
//! | Challenge 2.0 Sim2 SNR1                   | 0.570 → 0.598 | 68.9 → 60.0   | 0.110 → 0.184   |
//! | Challenge 2.0 Sim2 SNR2                   | 0.638 → 0.643 | 60.3 → 55.3   | 0.106 → 0.181   |
//! | QSM-CI simulation (2019 head model, 7 T)  | 0.786 → 0.743 | 36.3 → 39.1   | 0.123 → 0.193   |
//!
//! Caveats. A larger threshold underestimates χ more (TKD's known bias): the deep-grey-matter
//! regression slope moves further from 1 on every simulated set (last column), so for
//! quantitative ROI values a smaller threshold (or a regularised method) is the better choice.
//! The near-noise-free QSM-CI phantom prefers 0.15 on every metric. Beyond 0.25 XSIM keeps rising
//! on the 2016 data and Sim1 but falls on Sim2 (0.3: 0.591/0.628), and the slope bias grows
//! (0.4: 0.35–0.37), so 0.25 is the compromise. Figures and metrics: QSMxT tune-check, 2026-11-12
//! (`qsmxt-ecosystem/tune-check/`).
//!
//! With HEIDI's cone fill after TKD or TSVD the threshold must be at most HEIDI's cone threshold
//! (0.1); the `heidi.tkd-cone`/`heidi.tsvd-cone` rules require it, whatever the default.

use num_complex::Complex64;
use crate::fft::{fft3d, ifft3d};
use crate::kernels::dipole::dipole_kernel;
use crate::utils::apply_mask_zero;
use crate::Grid;

/// QSM.jl's default threshold for both `tkd` and `tsvd` (`direct.jl`), for parity runs.
pub const QSMJL_THRESHOLD: f64 = 0.15;

/// TKD algorithm parameters
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct TkdParams {
    /// Truncation threshold on `|D|` (default 0.25; QSM.jl's is 0.15, see the module docs)
    pub threshold: f64,
    /// Zero padding per axis, QSM.jl's `pad` (default `[0, 0, 0]`, QSM.jl's): `< 0` does not
    /// pad that axis, `≥ 0` pads it by at least that many voxels, rounded up to QSM.jl's fast
    /// FFT size (see [`with_qsmjl_padding`](crate::utils::padding::with_qsmjl_padding)).
    pub pad: [i64; 3],
}

impl Default for TkdParams {
    fn default() -> Self {
        Self { threshold: 0.25, pad: crate::utils::padding::QSMJL_PAD }
    }
}

/// TSVD algorithm parameters
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct TsvdParams {
    /// Truncation threshold on `|D|` (default 0.15, QSM.jl's)
    pub threshold: f64,
    /// Zero padding per axis, as [`TkdParams::pad`] (default `[0, 0, 0]`, QSM.jl's)
    pub pad: [i64; 3],
}

impl Default for TsvdParams {
    fn default() -> Self {
        Self { threshold: QSMJL_THRESHOLD, pad: crate::utils::padding::QSMJL_PAD }
    }
}

/// Truncated k-space division (TKD) for dipole inversion
///
/// Computes susceptibility map from local field using direct k-space division
/// with threshold-based truncation.
///
/// # Arguments
/// * `local_field` - Local field values (unwrapped phase / TE / gamma / B0)
/// * `mask` - Binary mask (1 = inside ROI, 0 = outside)
/// * `grid` - Volume grid (dimensions and voxel sizes)
/// * `bdir` - B0 field direction (bx, by, bz)
/// * `params` - TKD parameters (truncation threshold)
///
/// # Returns
/// Susceptibility map (same size as input)
pub fn tkd(
    local_field: &[f64],
    mask: &[u8],
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &TkdParams,
) -> Vec<f64> {
    crate::utils::padding::with_qsmjl_padding(local_field, mask, grid, params.pad, |f, m, g| {
        tkd_unpadded(f, m, g, bdir, params)
    })
}

/// [`tkd`] on the already padded grid.
fn tkd_unpadded(
    local_field: &[f64],
    mask: &[u8],
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &TkdParams,
) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let n_total = grid.n_total();
    let threshold = params.threshold;

    // Generate dipole kernel
    let d = dipole_kernel(grid, bdir);

    // Compute inverse dipole kernel with truncation
    // TKD: if |D| <= threshold, use sign(D)/threshold; else use 1/D
    let inv_threshold = 1.0 / threshold;
    let inv_d: Vec<f64> = d.iter().map(|&dval| {
        if dval.abs() <= threshold {
            // Truncate: use sign(D)/threshold
            if dval >= 0.0 { inv_threshold } else { -inv_threshold }
        } else {
            // Normal inverse
            1.0 / dval
        }
    }).collect();

    // QSM.jl applies this multiplier on its real-to-complex half spectrum; the kernel `irfft`
    // implies, so the full FFT reproduces it with tilted B0 (astewartau/QSM.rs#329).
    let inv_d = crate::fft::c2r_kernel(&inv_d, nx, ny, nz);

    // Convert local field to complex
    let mut field_complex: Vec<Complex64> = local_field.iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();

    // FFT of local field
    fft3d(&mut field_complex, nx, ny, nz);

    // Multiply by inverse dipole kernel
    for i in 0..n_total {
        field_complex[i] *= inv_d[i];
    }

    // IFFT to get susceptibility
    ifft3d(&mut field_complex, nx, ny, nz);

    // Extract real part and apply mask
    let mut chi: Vec<f64> = field_complex.iter()
        .map(|c| c.re)
        .collect();

    apply_mask_zero(&mut chi, mask);

    chi
}

/// Truncated singular value decomposition (TSVD) variant
///
/// Similar to TKD but zeros out values below threshold instead of truncating.
/// This produces smoother results but may have more artifacts at the magic angle.
pub fn tsvd(
    local_field: &[f64],
    mask: &[u8],
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &TsvdParams,
) -> Vec<f64> {
    crate::utils::padding::with_qsmjl_padding(local_field, mask, grid, params.pad, |f, m, g| {
        tsvd_unpadded(f, m, g, bdir, params)
    })
}

/// [`tsvd`] on the already padded grid.
fn tsvd_unpadded(
    local_field: &[f64],
    mask: &[u8],
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &TsvdParams,
) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let n_total = grid.n_total();
    let threshold = params.threshold;

    // Generate dipole kernel
    let d = dipole_kernel(grid, bdir);

    // Compute inverse dipole kernel with TSVD truncation
    // TSVD: if |D| <= threshold, use 0; else use 1/D
    let inv_d: Vec<f64> = d.iter().map(|&dval| {
        if dval.abs() <= threshold {
            0.0  // Zero out small values
        } else {
            1.0 / dval
        }
    }).collect();

    // QSM.jl applies this multiplier on its real-to-complex half spectrum; the kernel `irfft`
    // implies, so the full FFT reproduces it with tilted B0 (astewartau/QSM.rs#329).
    let inv_d = crate::fft::c2r_kernel(&inv_d, nx, ny, nz);

    // Convert local field to complex
    let mut field_complex: Vec<Complex64> = local_field.iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();

    // FFT of local field
    fft3d(&mut field_complex, nx, ny, nz);

    // Multiply by inverse dipole kernel
    for i in 0..n_total {
        field_complex[i] *= inv_d[i];
    }

    // IFFT to get susceptibility
    ifft3d(&mut field_complex, nx, ny, nz);

    // Extract real part and apply mask
    let mut chi: Vec<f64> = field_complex.iter()
        .map(|c| c.re)
        .collect();

    apply_mask_zero(&mut chi, mask);

    chi
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tkd_zero_field() {
        // Zero field should give zero susceptibility
        let n = 8;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        let chi = tkd(&field, &mask, &grid, (0.0, 0.0, 1.0), &TkdParams { threshold: 0.15, ..Default::default() });

        for val in chi.iter() {
            assert!(val.abs() < 1e-10, "Zero field should give zero chi");
        }
    }

    #[test]
    fn test_tkd_mask() {
        // Values outside mask should be zero
        let n = 8;
        let field = vec![1.0; n * n * n];
        let mut mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        // Set some voxels outside mask
        mask[0] = 0;
        mask[1] = 0;

        let chi = tkd(&field, &mask, &grid, (0.0, 0.0, 1.0), &TkdParams { threshold: 0.15, ..Default::default() });

        assert_eq!(chi[0], 0.0, "Outside mask should be 0");
        assert_eq!(chi[1], 0.0, "Outside mask should be 0");
    }

    #[test]
    fn test_tkd_finite() {
        // Result should be finite (no NaN or Inf)
        let n = 8;
        let field: Vec<f64> = (0..n*n*n).map(|i| (i as f64) * 0.01).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        let chi = tkd(&field, &mask, &grid, (0.0, 0.0, 1.0), &TkdParams { threshold: 0.15, ..Default::default() });

        for (i, val) in chi.iter().enumerate() {
            assert!(val.is_finite(), "Chi should be finite at index {}", i);
        }
    }

    #[test]
    fn test_tsvd_vs_tkd() {
        // TSVD and TKD should give different results near magic angle
        let n = 16;
        let field: Vec<f64> = (0..n*n*n).map(|i| ((i as f64) * 0.1).sin()).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        let chi_tkd = tkd(&field, &mask, &grid, (0.0, 0.0, 1.0), &TkdParams { threshold: 0.15, ..Default::default() });
        let chi_tsvd = tsvd(&field, &mask, &grid, (0.0, 0.0, 1.0), &TsvdParams { threshold: 0.15, ..Default::default() });

        // They should be different
        let diff: f64 = chi_tkd.iter().zip(chi_tsvd.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(diff > 1e-10, "TKD and TSVD should give different results");
    }
}

//! # QSM-Core
//!
//! A Rust library for Quantitative Susceptibility Mapping (QSM) of the brain.
//!
//! QSM-Core reconstructs magnetic susceptibility maps from MRI phase data:
//! brain extraction, phase unwrapping, background field removal, dipole
//! inversion, and susceptibility source separation.
//!
//! ## Which API should I use?
//!
//! The crate offers two entry points at different levels:
//!
//! - **[`pipeline`] — the high-level API.** Describe a scan with [`ScanMetadata`]
//!   and a [`QsmPipelineConfig`], then call the `run_*` stage functions. This is
//!   the easiest way to go from phase data to a susceptibility map and the
//!   recommended starting point.
//! - **Algorithm building blocks — the low-level API.** Each algorithm
//!   ([`bgremove::vsharp()`], [`inversion::tv_admm()`], …) is a plain function taking
//!   a [`Grid`], a `*Params` struct, and (for iterative methods) a progress
//!   callback. Use these when you want to wire stages together yourself.
//!
//! ```no_run
//! use qsm_core::{Grid, bet, unwrap, bgremove, inversion};
//! use qsm_core::bet::BetParams;
//! use qsm_core::bgremove::VsharpParams;
//! use qsm_core::inversion::TvParams;
//!
//! # fn run(phase: &[f64], magnitude: &[f64]) {
//! let grid = Grid::new(128, 128, 64, 1.0, 1.0, 1.0);
//! let bdir = (0.0, 0.0, 1.0);
//!
//! let mask = bet::run_bet(magnitude, &grid, &BetParams::default(), |_, _| {});
//! let unwrapped = unwrap::laplacian_unwrap(phase, &mask, &grid);
//! let (local, eroded) = bgremove::vsharp(&unwrapped, &mask, &grid, &VsharpParams::default(), |_, _| {});
//! let chi = inversion::tv_admm(&local, &eroded, &grid, bdir, &TvParams::default(), |_, _| {});
//! # let _ = chi;
//! # }
//! ```
//!
//! ## Modules
//!
//! **High-level pipeline**
//! - [`pipeline`] — config-driven full reconstruction
//!
//! **Algorithm building blocks**
//! - [`bet`] — brain extraction (BET)
//! - [`unwrap`] — phase unwrapping (ROMEO, Laplacian)
//! - [`bgremove`] — background field removal (V-SHARP, SHARP, RESHARP, PDF, iSMV, LBV, HARPERELLA)
//! - [`inversion`] — dipole inversion (TKD, TSVD, Tikhonov, TV, NLTV, RTS, MEDI, iLSQR, TGV)
//! - [`separation`] — paramagnetic/diamagnetic source separation (χ-separation)
//! - [`swi`] — susceptibility weighted imaging (CLEAR-SWI)
//! - [`fieldmap`] — multi-echo phase combination and B0 field mapping
//! - [`r2star`] — R2\*/T2\* mapping (ARLO)
//! - [`mask`] — mask thresholding and morphology
//! - [`homogeneity`] — receive-field bias correction
//!
//! **Core types & I/O**
//! - [`Grid`] — 3D volume descriptor shared by every algorithm
//! - [`io`] — NIfTI read/write
//!
//! ## Feature Flags
//!
//! - **`parallel`** — enables [Rayon](https://docs.rs/rayon)-based multi-threading for FFT and iterative solvers
//! - **`simd`** — enables SIMD acceleration via the [`wide`](https://docs.rs/wide) crate
//!
//! ## Algorithms
//!
//! | Stage | Methods |
//! |-------|---------|
//! | Brain extraction | BET |
//! | Phase unwrapping | ROMEO, Laplacian |
//! | Background removal | V-SHARP, SHARP, RESHARP, PDF, iSMV, LBV, SDF |
//! | Dipole inversion | TKD, TSVD, Tikhonov, TV-ADMM, NLTV, RTS, MEDI, TGV, iLSQR |
//! | Combined unwrap+BFR | HARPERELLA, iHARPERELLA |
//! | SWI | CLEAR-SWI |
//! | Separation | Chi-separation (MEDI-based) |
//! | Multi-echo | MCPC-3D-S, R2\*/T2\* (ARLO), bias correction |
//! | Utilities | Frangi vesselness, surface curvature, Otsu thresholding, QSMART |

// ============================================================================
// Internal plumbing — public so advanced/downstream code can reach it, but
// hidden from the documented API surface. No stability guarantees.
// ============================================================================
#[macro_use]
#[doc(hidden)]
pub mod par;
#[doc(hidden)]
pub mod grid;
#[doc(hidden)]
pub mod fft;
#[doc(hidden)]
pub mod priority_queue;
#[doc(hidden)]
pub mod region_grow;
#[doc(hidden)]
pub mod kernels;
#[doc(hidden)]
pub mod solvers;
#[doc(hidden)]
pub mod utils;

// ============================================================================
// Core types
// ============================================================================
pub use grid::Grid;

// ============================================================================
// High-level pipeline
// ============================================================================
pub mod pipeline;

// ============================================================================
// Algorithm building blocks
// ============================================================================
pub mod bet;
pub mod unwrap;
pub mod bgremove;
pub mod inversion;
pub mod separation;
pub mod swi;

/// Multi-echo phase combination and B0 field mapping.
///
/// Building blocks for turning multi-echo wrapped phase into a B0 field map:
/// phase-offset removal (MCPC-3D-S / ASPIRE), weighted B0 estimation, and
/// linear multi-echo fitting.
pub mod fieldmap {
    pub use crate::utils::multi_echo::{
        phase_offset_removal, calculate_b0_weighted, multi_echo_linear_fit,
        bipolar_correction, field_to_hz,
        PhaseOffsetParams, LinearFitParams, LinearFitResult, B0WeightType,
    };
}

/// R2\*/T2\* mapping from multi-echo magnitude (ARLO).
pub mod r2star {
    pub use crate::utils::r2star::{
        r2star_arlo, t2star_from_r2star, use_arlo,
    };
}

/// Brain-mask thresholding and morphology.
///
/// Mask generation ([`otsu_threshold`](crate::mask::otsu_threshold)) plus
/// morphological operations (erode/dilate/close/fill-holes) and sphere masks.
pub mod mask {
    pub use crate::utils::mask::{
        create_sphere_mask, apply_mask_zero, erode_mask, dilate_mask,
    };
    pub use crate::utils::threshold::otsu_threshold;
    pub use crate::utils::curvature::morphological_close;
    pub use crate::utils::bias_correction::fill_holes;
}

/// Receive-field (B1−) bias correction for magnitude images.
pub mod homogeneity {
    pub use crate::utils::bias_correction::{
        makehomogeneous, get_sensitivity, HomogeneityParams,
    };
}

// ============================================================================
// I/O
// ============================================================================
pub mod io;

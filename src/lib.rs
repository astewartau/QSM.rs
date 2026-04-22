//! # QSM-Core
//!
//! A Rust library for Quantitative Susceptibility Mapping (QSM) of the brain.
//!
//! QSM-Core provides a complete set of algorithms for reconstructing magnetic
//! susceptibility maps from MRI phase data, including brain extraction, phase
//! unwrapping, background field removal, dipole inversion, and susceptibility
//! source separation.
//!
//! ## Pipeline Overview
//!
//! A typical QSM pipeline follows these steps:
//!
//! 1. **Brain extraction** ([`bet`]) — mask the brain region from magnitude images
//! 2. **Phase unwrapping** ([`unwrap`]) — remove 2π wraps from MRI phase data
//! 3. **Background field removal** ([`bgremove`]) — isolate the local field from background sources
//! 4. **Dipole inversion** ([`inversion`]) — reconstruct susceptibility from the local field
//!
//! Additional processing modules:
//!
//! - **SWI** ([`swi`]) — Susceptibility Weighted Imaging (CLEAR-SWI)
//! - **Chi-separation** ([`separation`]) — decompose susceptibility into paramagnetic/diamagnetic components
//! - **Multi-echo processing** ([`utils`]) — R2\*/T2\* mapping, phase combination (MCPC-3D-S), bias correction
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
//! | Background removal | V-SHARP, SHARP, PDF, iSMV, LBV, SDF |
//! | Dipole inversion | TKD, TSVD, Tikhonov, TV-ADMM, NLTV, RTS, MEDI, TGV, iLSQR |
//! | SWI | CLEAR-SWI |
//! | Separation | Chi-separation (MEDI-based) |
//! | Multi-echo | MCPC-3D-S, R2\*/T2\* (ARLO), bias correction |
//! | Utilities | Frangi vesselness, surface curvature, Otsu thresholding, QSMART |

// Conditional parallelism (must be first for macro visibility)
#[macro_use]
pub mod par;

// Core modules
pub mod fft;
pub mod priority_queue;
pub mod region_grow;

// Algorithm modules
pub mod kernels;
pub mod unwrap;
pub mod bgremove;
pub mod inversion;
pub mod separation;
pub mod solvers;
pub mod utils;
pub mod swi;

// I/O modules
pub mod nifti_io;

// Brain extraction
pub mod bet;

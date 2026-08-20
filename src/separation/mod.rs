//! Susceptibility source separation
//!
//! This module provides algorithms for separating total magnetic susceptibility
//! into paramagnetic (χ+, primarily iron) and diamagnetic (χ-, primarily myelin)
//! components using local field maps and R2' relaxation data.
//!
//! # Methods
//! - `chi_sep_ilsqr`: the original Shin 2021 projected-CG algorithm (SNU-LIST
//!   toolbox `chi_sep_iLSQR`), initialized from a conventional QSM
//! - `chi_sep_medi`: MEDI-based Gauss-Newton optimization with coupled field + R2' constraints
//! - `r2star_qsm`: closed-form separation from a QSM + R2* (Dimov 2022, no R2' needed)
//! - `wavesep`: wavelet-L1 proximal-gradient separation from a QSM + R2' (Fang 2023)
//! - `decompose`: signal-domain 3-compartment per-voxel fit from a QSM + multi-echo
//!   magnitude (Chen 2021)
//! - `hc_chisep`: hollow-cylinder χ-separation with signal-derived fiber orientation
//!   from a QSM + R2' + multi-echo magnitude (Wharton & Bowtell model)
//!
//! # Reference
//! Shin, H., et al. (2021). "χ-separation: Magnetic susceptibility source separation
//! toward iron and myelin mapping in the brain." NeuroImage, 240:118371.

pub mod chi_sep_ilsqr;
pub mod chi_sep_medi;
pub mod decompose;
pub mod hc_chisep;
pub mod r2star_qsm;
pub mod wavesep;
#[cfg(feature = "onnx")]
pub mod susep_net;
#[cfg(feature = "onnx")]
pub mod chisepnet;

pub use chi_sep_ilsqr::{chi_sep_ilsqr, ChiSepIlsqrParams};
pub use chi_sep_medi::{chi_sep_medi, ChiSepParams};
pub use decompose::{decompose, DecomposeParams};
pub use hc_chisep::{hc_chisep, HcChisepParams};
pub use r2star_qsm::{r2star_qsm, r2star_qsm_from_magnitude, R2starQsmParams};
pub use wavesep::{wavesep, WaveSepParams};
#[cfg(feature = "onnx")]
pub use susep_net::{susep_net, SusepNetNorm};
#[cfg(feature = "onnx")]
pub use chisepnet::{chisepnet, ChiSepNetNorm};

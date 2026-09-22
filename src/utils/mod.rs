//! Utility functions for QSM processing
//!
//! This module provides common utility functions:
//! - Gradient operators (forward/backward differences)
//! - Padding utilities
//! - Mask operations (incl. signal-gated erosion)
//! - R2*/R2 relaxometry and learned R2* -> R2' conversion (`onnx`)
//! - Multi-echo field mapping (phase offset removal, B0 estimation)
//! - Bias field correction (makehomogeneous)
//! - Frangi vesselness filter for vessel detection
//! - Surface curvature calculation
//! - Vasculature mask generation
//! - QSMART offset adjustment and utilities
//! - SIMD-accelerated operations (optional, with `simd` feature)

pub mod gradient;
pub mod padding;
pub mod multi_echo;
pub mod bias_correction;
pub mod frangi;
pub mod curvature;
pub mod vasculature;
pub mod qsmart;
pub mod threshold;
pub mod mask;
pub mod connected;
pub mod signal_erosion;
pub mod resample;
pub mod simd_ops;
pub mod r2star;
pub mod epg;
#[cfg(feature = "onnx")]
pub mod r2primenet;
#[cfg(feature = "onnx")]
pub(crate) mod sliding;
pub mod denoise;
pub mod anisotropic_diffusion;
pub mod gibbs;
pub mod ops;
pub mod special;
pub mod wavelet;

pub use gradient::*;
pub use padding::*;
pub use multi_echo::*;
pub use bias_correction::*;
pub use frangi::*;
pub use curvature::*;
pub use vasculature::*;
pub use qsmart::*;
pub use threshold::*;
pub use mask::*;
pub use signal_erosion::{signal_gated_erosion, SignalErosionParams};
pub use simd_ops::*;
pub use r2star::*;
pub use epg::*;
#[cfg(feature = "onnx")]
pub use r2primenet::{
    r2primenet, r2primenet_from_magnitude, R2PrimeNetNorm, R2PrimeNetParams,
    AUTHORS_PATCH, WASM_PATCH,
};
pub use denoise::*;
pub use anisotropic_diffusion::{gradient_anisotropic_diffusion, AnisotropicDiffusionParams};
pub use gibbs::*;
pub use ops::*;

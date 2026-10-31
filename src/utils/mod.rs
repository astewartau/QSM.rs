//! Utility functions for QSM processing
//!
//! This module provides common utility functions:
//! - Gradient operators (forward/backward differences)
//! - Padding utilities
//! - Mask creation and morphology (erosion, dilation, closing, hole filling, signal-gated erosion)
//! - R2*/R2 relaxometry and learned R2* -> R2' conversion (`onnx`)
//! - Real/imaginary to magnitude/phase conversion with slice-chopping correction
//! - Multi-echo field mapping (phase offset removal, B0 estimation)
//! - Bias field correction (makehomogeneous; N4 as in ANTs)
//! - Frangi vesselness filter for vessel detection, and MATLAB's `fibermetric`
//! - Surface curvature calculation
//! - Vasculature mask generation
//! - QSMART offset adjustment and utilities
//! - SIMD-accelerated operations (optional, with `simd` feature)

pub mod gradient;
pub mod echo_time;
pub mod padding;
pub mod complex;
pub mod multi_echo;
pub mod bias_correction;
pub mod n4;
pub mod frangi;
pub mod fibermetric;
pub mod curvature;
pub mod vasculature;
pub mod itk_geometry;
pub mod qsmart;
pub mod threshold;
pub mod mask;
pub mod connected;
pub mod union_find;
pub mod morphology;
pub mod resample;
pub mod simd_ops;
pub mod r2star;
pub mod epg;
pub mod r2primenet;
#[cfg(feature = "onnx")]
pub(crate) mod sliding;
pub mod denoise;
pub mod anisotropic_diffusion;
pub mod gibbs;
pub mod unring3d;
pub mod ops;
pub mod special;
pub mod wavelet;
pub mod fdlibm;
pub mod portable_math;

pub use gradient::*;
pub use padding::*;
pub use multi_echo::*;
pub use bias_correction::*;
pub use n4::{n4_bias_correction, N4Output, N4Params};
pub use itk_geometry::ItkGeometry;
pub use frangi::*;
pub use fibermetric::{fibermetric, FiberPolarity, FIBERMETRIC_DEFAULT_C};
pub use curvature::*;
pub use vasculature::*;
pub use qsmart::*;
pub use threshold::*;
pub use mask::*;
pub use morphology::*;
pub use simd_ops::*;
pub use r2star::*;
pub use epg::*;
pub use r2primenet::{
    R2PrimeNetNorm, R2PrimeNetParams, AUTHORS_PATCH, WASM_PATCH, B0_TOLERANCE_T, TRAINED_B0_T,
};
#[cfg(feature = "onnx")]
pub use r2primenet::{r2primenet, r2primenet_from_magnitude};
pub use denoise::*;
pub use anisotropic_diffusion::{gradient_anisotropic_diffusion, AnisotropicDiffusionParams};
pub use gibbs::*;
pub use ops::*;

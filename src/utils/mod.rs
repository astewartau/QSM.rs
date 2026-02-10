//! Utility functions for QSM processing
//!
//! This module provides common utility functions:
//! - Gradient operators (forward/backward differences)
//! - Padding utilities
//! - Mask operations
//! - Multi-echo phase combination (MCPC-3D-S)
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
pub mod simd_ops;

pub use gradient::*;
pub use padding::*;
pub use multi_echo::*;
pub use bias_correction::*;
pub use frangi::*;
pub use curvature::*;
pub use vasculature::*;
pub use qsmart::*;
pub use simd_ops::*;

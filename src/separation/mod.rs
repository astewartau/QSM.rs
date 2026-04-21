//! Susceptibility source separation
//!
//! This module provides algorithms for separating total magnetic susceptibility
//! into paramagnetic (χ+, primarily iron) and diamagnetic (χ-, primarily myelin)
//! components using local field maps and R2' relaxation data.
//!
//! # Methods
//! - `chi_sep_medi`: MEDI-based Gauss-Newton optimization with coupled field + R2' constraints
//!
//! # Reference
//! Shin, H., et al. (2021). "χ-separation: Magnetic susceptibility source separation
//! toward iron and myelin mapping in the brain." NeuroImage, 240:118371.

pub mod chi_sep_medi;

pub use chi_sep_medi::*;

//! Background field removal methods
//!
//! This module provides methods to separate local from background field:
//! - SMV: Simple spherical mean value (basic subtraction)
//! - SHARP: Sophisticated harmonic artifact reduction for phase data
//! - V-SHARP: Variable kernel SHARP
//! - PDF: Projection onto dipole fields
//! - iSMV: Iterative spherical mean value
//! - LBV: Laplacian boundary value
//! - SDF: Spatially Dependent Filtering (QSMART)

pub mod smv;
pub mod sharp;
pub mod vsharp;
pub mod pdf;
pub mod ismv;
pub mod lbv;
pub mod sdf;

pub use smv::{smv, smv_default};
pub use sharp::{sharp, sharp_default};
pub use vsharp::{vsharp, vsharp_default};
pub use pdf::{pdf, pdf_default};
pub use ismv::{ismv, ismv_default};
pub use lbv::{lbv, lbv_default, lbv_with_progress};
pub use sdf::{sdf, sdf_curvature, sdf_simple, SdfParams};

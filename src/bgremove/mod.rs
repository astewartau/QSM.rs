//! Background field removal methods
//!
//! This module provides methods to separate local from background field:
//! - SHARP: Sophisticated harmonic artifact reduction for phase data
//! - RESHARP: Regularized SHARP with Tikhonov regularization
//! - V-SHARP: Variable kernel SHARP
//! - PDF: Projection onto dipole fields
//! - iSMV: Iterative spherical mean value
//! - LBV: Laplacian boundary value
//! - SDF: Spatially Dependent Filtering (QSMART)
//!
//! Note: The `smv` module provides simple spherical mean value filtering,
//! used internally by SHARP, V-SHARP, and iSMV. It is not recommended as
//! a standalone background removal method (it lacks the deconvolution step
//! that SHARP provides).

pub mod smv;
pub mod sharp;
pub mod resharp;
pub mod vsharp;
pub mod pdf;
pub mod ismv;
pub mod lbv;
pub mod sdf;
pub mod iharperella;

pub use smv::smv;
pub use sharp::{sharp, SharpParams};
pub use resharp::{resharp, ResharpParams};
pub use vsharp::{vsharp, VsharpParams};
pub use pdf::{pdf, PdfParams};
pub use ismv::{ismv, IsmvParams};
pub use lbv::{lbv, LbvParams};
pub use sdf::{sdf, sdf_curvature, sdf_simple, SdfParams};
pub use iharperella::{
    harperella, iharperella, iharperella_with_weights,
    HarperellaParams, IharperellaParams,
};

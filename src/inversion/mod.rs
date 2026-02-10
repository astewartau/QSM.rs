//! Dipole inversion methods for QSM
//!
//! This module provides various methods to solve the inverse problem
//! of estimating magnetic susceptibility from local field measurements.
//!
//! Methods include:
//! - TKD: Truncated k-space division (fast, simple)
//! - TSVD: Truncated SVD (zeros small values)
//! - Tikhonov: L2 regularization (closed-form)
//! - TV: Total variation regularization via ADMM (iterative)
//! - NLTV: Nonlinear TV with iterative reweighting
//! - RTS: Rapid two-step method
//! - MEDI: Morphology-enabled dipole inversion
//! - TGV: Total Generalized Variation (single-step from wrapped phase)

pub mod tkd;
pub mod tikhonov;
pub mod tv;
pub mod nltv;
pub mod rts;
pub mod medi;
pub mod tgv;
pub mod ilsqr;

pub use tkd::*;
pub use tikhonov::*;
pub use tv::*;
pub use nltv::*;
pub use rts::*;
pub use medi::*;
pub use tgv::*;
pub use ilsqr::*;

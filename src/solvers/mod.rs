//! Iterative solvers for QSM
//!
//! This module provides iterative solvers used by various QSM algorithms:
//! - CG: Conjugate gradient
//! - LSMR: Least squares minimal residual

pub mod cg;
pub mod lsmr;

pub use cg::*;
pub use lsmr::*;

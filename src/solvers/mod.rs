//! Iterative solvers for QSM
//!
//! This module provides iterative solvers used by various QSM algorithms:
//! - CG: Conjugate gradient

pub mod cg;

pub use cg::*;

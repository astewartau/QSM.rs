//! Iterative solvers for QSM
//!
//! This module provides iterative solvers used by various QSM algorithms:
//! - CG: Conjugate gradient
//! - [`lsq_trr`]: bound-constrained nonlinear least squares by the subspace trust-region
//!   reflective method (MATLAB `lsqcurvefit`/`lsqnonlin`'s default algorithm)

pub mod cg;
pub mod lsq_trr;
#[cfg(test)]
pub(crate) mod lsq_trr_trace;

pub use cg::*;
pub use lsq_trr::{
    lsq_trust_region_reflective, LsqTrrExit, LsqTrrIterate, LsqTrrOptions, LsqTrrReport,
    LsqTrrWorkspace,
};

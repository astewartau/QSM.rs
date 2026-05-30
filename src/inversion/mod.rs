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

pub mod admm;
pub mod tkd;
pub mod tikhonov;
pub mod tv;
pub mod nltv;
pub mod rts;
pub mod medi;
pub mod tgv;
pub mod ilsqr;

pub use tkd::{tkd, tkd_default, tsvd, TkdParams};
pub use tikhonov::{tikhonov, tikhonov_default, TikhonovParams, Regularization};
pub use tv::{tv_admm, tv_admm_with_progress, tv_admm_default, TvParams};
pub use nltv::{nltv, nltv_with_progress, nltv_default, NltvParams};
pub use rts::{rts, rts_with_progress, rts_default, RtsParams};
pub use medi::{medi_l1, medi_l1_with_progress, medi_l1_default, MediParams, MediWorkspace};
pub use tgv::{tgv_qsm, tgv_qsm_with_progress, TgvParams, get_default_alpha, get_default_iterations};
pub use ilsqr::{ilsqr, ilsqr_simple, ilsqr_with_progress, IlsqrParams, lsqr, lsqr_complex, lsmr};

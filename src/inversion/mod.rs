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
pub mod ndi;
pub mod fansi;
pub mod l1qsm;
pub mod whqsm;
pub mod hdqsm;

pub use tkd::{tkd, tsvd, TkdParams};
pub use tikhonov::{tikhonov, TikhonovParams, Regularization};
pub use tv::{tv_admm, TvParams};
pub use nltv::{nltv, NltvParams};
pub use rts::{rts, RtsParams};
pub use medi::{medi, MediParams, MediWorkspace};
pub use tgv::{tgv_qsm, TgvParams, get_default_alpha, get_default_iterations};
pub use ilsqr::{ilsqr, IlsqrParams};
pub use ndi::{ndi, NdiParams};
pub use fansi::{fansi, FansiParams};
pub use l1qsm::{l1qsm, L1QsmParams};
pub use whqsm::{whqsm, WhQsmParams};
pub use hdqsm::{hdqsm, HdQsmParams};

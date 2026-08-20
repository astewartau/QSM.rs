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
pub mod tfi;
pub mod tgv;
pub mod ilsqr;
pub mod ndi;
pub mod fansi;
pub mod l1qsm;
pub mod whqsm;
pub mod hdqsm;
pub mod amp_pe;
#[cfg(feature = "onnx")]
pub mod xqsm;
#[cfg(feature = "onnx")]
pub mod qsmnet;
#[cfg(feature = "onnx")]
pub mod autoqsm;
#[cfg(feature = "onnx")]
pub mod iqsm;
#[cfg(feature = "onnx")]
pub mod iqsm_plus;
#[cfg(feature = "onnx")]
pub mod nextqsm;

pub use tkd::{tkd, tsvd, TkdParams};
pub use tikhonov::{tikhonov, TikhonovParams, Regularization};
pub use tv::{tv_admm, TvParams};
pub use nltv::{nltv, NltvParams};
pub use rts::{rts, RtsParams};
pub use medi::{medi, MediParams, MediWorkspace};
pub use tfi::{tfi, TfiParams};
pub use tgv::{tgv_qsm, TgvParams, get_default_alpha, get_default_iterations};
pub use ilsqr::{ilsqr, IlsqrParams};
pub use ndi::{ndi, NdiParams};
pub use fansi::{fansi, FansiParams};
pub use l1qsm::{l1qsm, L1QsmParams};
pub use whqsm::{whqsm, WhQsmParams};
pub use hdqsm::{hdqsm, HdQsmParams};
pub use amp_pe::{amp_pe, AmpPeParams};
#[cfg(feature = "onnx")]
pub use xqsm::xqsm;
#[cfg(feature = "onnx")]
pub use qsmnet::{qsmnet, QsmnetNorm};
#[cfg(feature = "onnx")]
pub use autoqsm::autoqsm;
#[cfg(feature = "onnx")]
pub use iqsm::{iqsm, iqsm_multi_echo};
#[cfg(feature = "onnx")]
pub use iqsm_plus::{iqsm_plus, iqsm_plus_multi_echo};
#[cfg(feature = "onnx")]
pub use nextqsm::{nextqsm, nextqsm_padded, NEXTQSM_LAMBDAS};

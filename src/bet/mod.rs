//! Brain extraction.
//!
//! - **BET** ([`run_bet`]): mesh-evolution Brain Extraction Tool.
//!   Smith, S.M. (2002). "Fast robust automated brain extraction."
//!   Human Brain Mapping, 17(3):143-155. https://doi.org/10.1002/hbm.10062.
//!   Reference implementation: https://github.com/Bostrix/FSL-BET2
//! - **HD-BET** (`hd_bet`, `onnx` feature): nnU-Net deep-learning brain extraction; see
//!   [`hdbet`].

mod icosphere;
mod mesh;
mod evolution;
pub mod hdbet;

pub use evolution::{run_bet, BetParams};
#[cfg(feature = "onnx")]
pub use hdbet::hd_bet;
pub use hdbet::HdBetParams;

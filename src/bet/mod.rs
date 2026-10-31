//! Brain extraction.
//!
//! - **BET** ([`run_bet`]): mesh-evolution Brain Extraction Tool.
//!   Smith, S.M. (2002). "Fast robust automated brain extraction."
//!   Human Brain Mapping, 17(3):143-155. https://doi.org/10.1002/hbm.10062.
//!   Reference implementation: https://github.com/Bostrix/FSL-BET2
//! - **HD-BET** (`hd_bet`, `onnx` feature): nnU-Net deep-learning brain extraction; see
//!   [`hdbet`].
//! - **RS2-Net** (`rs2_net`, `onnx` feature): deep-learning *rodent* brain extraction (mouse,
//!   rat); see [`rs2net`].
//! - **MindGrab** (`mindgrab`, `onnx` feature): deep-learning brain extraction for any
//!   contrast; see [`mod@mindgrab`].

mod icosphere;
mod mesh;
mod evolution;
pub mod hdbet;
pub mod mindgrab;
pub mod rs2net;

pub use evolution::{run_bet, BetParams, FslLeftRightOrder};
#[cfg(feature = "onnx")]
pub use hdbet::hd_bet;
pub use hdbet::{hd_bet_memory_bytes, hd_bet_network_input, HdBetInference, HdBetParams};
#[cfg(feature = "onnx")]
pub use rs2net::rs2_net;
pub use rs2net::{Rs2NetParams, RS2_NET_PATCH};
#[cfg(feature = "onnx")]
pub use mindgrab::mindgrab;
pub use mindgrab::MindGrabParams;

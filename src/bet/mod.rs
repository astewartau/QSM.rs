//! BET (Brain Extraction Tool) Implementation
//!
//! Based on: Smith, S.M. (2002) "Fast robust automated brain extraction"
//! Human Brain Mapping, 17(3):143-155
//!
//! Ported from Python vectorized implementation.

mod icosphere;
mod mesh;
mod evolution;

pub use evolution::{run_bet, run_bet_with_progress};

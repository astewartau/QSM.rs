//! BET (Brain Extraction Tool) Implementation
//!
//! Reference:
//! Smith, S.M. (2002). "Fast robust automated brain extraction."
//! Human Brain Mapping, 17(3):143-155. https://doi.org/10.1002/hbm.10062
//!
//! Reference implementation: https://github.com/Bostrix/FSL-BET2

mod icosphere;
mod mesh;
mod evolution;

pub use evolution::{run_bet, run_bet_with_progress};

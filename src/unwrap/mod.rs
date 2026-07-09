//! Phase unwrapping methods
//!
//! Removes 2π wraps from MRI phase data.
//!
//! - [`unwrap_romeo`] / [`unwrap_romeo_multi_echo`] — ROMEO, region growing with
//!   quality-guided ordering ([`RomeoParams`])
//! - [`laplacian_unwrap`] — fast FFT-based Laplacian unwrapping

pub mod romeo;
pub mod laplacian;

pub use romeo::{
    unwrap_romeo, unwrap_romeo_multi_echo, correct_multi_echo_wraps,
    calculate_weights_romeo, voxel_quality_romeo,
    RomeoParams, RomeoWeightType,
};
pub use laplacian::laplacian_unwrap;

/// Phase unwrapping method selection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UnwrapMethod {
    Romeo,
    Laplacian,
}

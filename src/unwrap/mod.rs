//! Phase unwrapping methods
//!
//! Removes 2π wraps from MRI phase data.
//!
//! - [`unwrap_romeo`] / [`unwrap_romeo_multi_echo`] — ROMEO, region growing with
//!   quality-guided ordering ([`RomeoParams`])
//! - [`laplacian_unwrap`] — Laplacian unwrapping (Neumann BC on the array)
//! - [`laplacian_unwrap_bfr`] — Laplacian unwrapping **+ background field removal**
//!   (∇² masked to the ROI); see the [`laplacian`] module docs for why these differ

pub mod romeo;
pub mod laplacian;

pub use romeo::{
    unwrap_romeo, unwrap_romeo_multi_echo, correct_multi_echo_wraps,
    calculate_weights_romeo, voxel_quality_romeo,
    RomeoParams, RomeoWeightType,
};
pub use laplacian::{laplacian_unwrap_bfr, laplacian_unwrap};

/// Phase unwrapping method selection.
///
/// These select unwrappers only. [`laplacian_unwrap_bfr`], which unwraps *and* removes the
/// harmonic background, is deliberately not reachable here — call it directly if that
/// combination is what you want, and read the [`laplacian`] module docs first.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UnwrapMethod {
    Romeo,
    /// Laplacian unwrapping under a Neumann boundary condition; see
    /// [`laplacian_unwrap`]. Unwraps only — the background field is left alone.
    Laplacian,
}

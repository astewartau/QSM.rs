//! Phase unwrapping methods
//!
//! Removes 2π wraps from MRI phase data.
//!
//! - [`unwrap_romeo`] / [`unwrap_romeo_multi_echo`] — ROMEO, region growing with
//!   quality-guided ordering ([`RomeoParams`])
//! - [`unwrap_bestpath`] — best path (3D-SRNCP), globally sorted edge merging
//!   ([`BestPathParams`])
//! - [`laplacian_unwrap`] — Laplacian unwrapping (Neumann BC on the array)
//! - [`laplacian_unwrap_bfr`] — Laplacian unwrapping **+ background field removal**
//!   (∇² masked to the ROI); see the [`laplacian`] module docs for why these differ
//! - [`laplacian_unwrap_sti`] — Laplacian unwrapping as STI Suite 3.0 computes it (spectral
//!   sin/cos Laplacian, zero-padded periodic solve), and
//!   [`laplacian_unwrap_sti_multi_echo`] for the per-echo unwrap + weighted echo average
//!   field map built on it; see the [`laplacian_sti`] module docs

pub mod romeo;
pub mod bestpath;
pub mod laplacian;
pub mod laplacian_sti;

pub use romeo::{
    unwrap_romeo, unwrap_romeo_multi_echo, correct_multi_echo_wraps,
    calculate_weights_romeo, voxel_quality_romeo,
    RomeoParams, RomeoWeightType,
};
pub use bestpath::{unwrap_bestpath, BestPathParams};
#[allow(deprecated)]
pub use laplacian::laplacian_unwrap_bfr;
pub use laplacian::laplacian_unwrap;
pub use laplacian_sti::{
    laplacian_unwrap_sti, laplacian_unwrap_sti_multi_echo, EchoAverage, EchoWeighting,
    LaplacianStiParams,
};

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
    /// STI Suite's formulation is [`laplacian_unwrap_sti`] (not selectable here).
    Laplacian,
    /// Best path (3D-SRNCP); see [`unwrap_bestpath`]. The original Abdul-Rahman
    /// algorithm, not ROMEO run with best-path weights — for that, set
    /// [`RomeoParams::bestpath`].
    BestPath,
}

//! Phase unwrapping methods
//!
//! Removes 2π wraps from MRI phase data.
//!
//! - [`unwrap_romeo`] / [`unwrap_romeo_multi_echo`] — ROMEO, region growing with
//!   quality-guided ordering ([`RomeoParams`])
//! - [`unwrap_bestpath`] — best path (3D-SRNCP), globally sorted edge merging
//!   ([`BestPathParams`])
//! - [`unwrap_prelude`] — PRELUDE, best-pair-first region merging ([`PreludeParams`]);
//!   the literature's reference unwrapper, kept as a baseline rather than a default
//! - [`unwrap_medi_region_growing`] — MEDI's `unwrapPhase` (`mexUnwrap.c`, Tian Liu):
//!   magnitude-ordered region growing from the middle of the field of view, no mask, single
//!   precision, bit-identical to the MEDI toolbox
//! - [`laplacian_unwrap`] — Laplacian unwrapping, with a [`LaplacianSolver`]: `Dct` (default;
//!   unweighted least squares, Neumann BC on the array), `Fft { pad }` (Schofield & Zhu sin/cos
//!   method on the zero-padded volume, which reproduces STI Suite 3.0's `MRPhaseUnwrap`),
//!   `Mirror` (the same on the mirrored volume, QSMART's `lapunwrap.m`) or `Mgpcg` (QSM.jl's
//!   default: Dirichlet on the mask, multigrid-preconditioned CG; also removes the background)
//!
//! All of the above read across slice boundaries. For a 2D multi-slice acquisition, where
//! each slice carries its own receive phase offset, run one of them per slice with
//! [`unwrap_slicewise`] / [`unwrap_slicewise_multi_echo`] ([`SliceWiseParams`]).

pub mod romeo;
pub mod bestpath;
pub mod laplacian;
pub mod prelude;
pub mod medi_region_growing;
pub mod slicewise;

pub use romeo::{
    unwrap_romeo, unwrap_romeo_multi_echo, correct_multi_echo_wraps,
    calculate_weights_romeo, voxel_quality_romeo,
    RomeoParams, RomeoWeightType,
};
pub use bestpath::{unwrap_bestpath, BestPathParams};
pub use prelude::{unwrap_prelude, PreludeLabelling, PreludeParams};
pub use medi_region_growing::{unwrap_medi_region_growing, MEDI_NOISE_RATIO};
pub use slicewise::{
    unwrap_slicewise, unwrap_slicewise_multi_echo, enforce_inter_echo_consistency,
    SliceWiseParams,
};
pub use laplacian::{laplacian_unwrap, LaplacianSolver};

/// Phase unwrapping method selection.
///
/// These select unwrappers only. Methods that unwrap *and* remove the background field
/// (HARPERELLA, iHARPERELLA) live in [`crate::bgremove`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UnwrapMethod {
    Romeo,
    /// Laplacian unwrapping with [`LaplacianSolver::Dct`] (Neumann boundary on the array); see
    /// [`laplacian_unwrap`]. Unwraps only — the background field is left alone. Call
    /// [`laplacian_unwrap`] directly for the other [`LaplacianSolver`]s.
    Laplacian,
    /// Best path (3D-SRNCP); see [`unwrap_bestpath`]. The original Abdul-Rahman
    /// algorithm, not ROMEO run with best-path weights — for that, set
    /// [`RomeoParams::bestpath`].
    BestPath,
    /// PRELUDE (Jenkinson) with its settings; see [`unwrap_prelude`]. The unwrapper the
    /// literature benchmarks against. It carries its [`PreludeParams`] so that every caller
    /// that takes an `UnwrapMethod` (MCPC-3D-S's HIP unwrapping, the slice-wise unwrapper) runs
    /// it with the caller's settings, not silently with the defaults.
    Prelude(PreludeParams),
    /// MEDI's region growing; see [`unwrap_medi_region_growing`]. The mask is not used (MEDI's
    /// `unwrapPhase` takes none) and the magnitude is its quality map, so the callers that take
    /// an `UnwrapMethod` pass theirs: MCPC-3D-S's HIP unwrapping its weight `sqrt(|HIP|)`, the
    /// slice-wise unwrapper each slice's magnitude (required there).
    MediRegionGrowing,
}

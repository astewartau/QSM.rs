//! Background field removal methods
//!
//! This module provides methods to separate local from background field:
//! - SHARP: Sophisticated harmonic artifact reduction for phase data
//! - RESHARP: Regularized SHARP with Tikhonov regularization
//! - V-SHARP: Variable kernel SHARP
//! - PDF: Projection onto dipole fields
//! - iSMV: Iterative spherical mean value
//! - LBV: Laplacian boundary value
//! - SDF: Spatially Dependent Filtering (QSMART)
//!
//! mSMV (maximum spherical mean value) is not here: it is part of MEDI-mSMV, which fits its
//! SMV-filtered field with the SMV-filtered dipole kernel, so its output is not a local field for
//! the other inversions. See [`crate::inversion::msmv`](mod@crate::inversion::msmv).
//!
//! Note: The `smv` module provides simple spherical mean value filtering,
//! used internally by SHARP, V-SHARP, and iSMV. It is not recommended as
//! a standalone background removal method (it lacks the deconvolution step
//! that SHARP provides).
//!
//! ## 2D multi-slice data: background removal stays 3D
//!
//! Background field removal is **3D** even for 2D multi-slice acquisitions, and so is the
//! dipole inversion. SHARP and its variants rest on the mean value property of harmonic
//! functions, which is three-dimensional; a disc has no equivalent, so V-SHARP run slice by
//! slice would remove only the in-plane component of a field that is not separable that way.
//! A 2D multi-slice acquisition still assembles into a 3D volume, so there is nothing to
//! replace here, only anisotropic voxels to handle, which the kernels already do through their
//! frequency axes. (Phase *unwrapping* is the stage that does go per slice; see
//! [`crate::unwrap::slicewise`].)
//!
//! A 2D V-SHARP stage chained into 3D PDF, as the EPI-QSM literature uses, was implemented,
//! measured and removed. The numbers are kept here so the case does not have to be
//! re-implemented to be re-litigated.
//!
//! On a synthetic background with every source outside the ROI, where the ground truth is
//! exactly zero (residual RMS, lower is better):
//!
//! | | slab coverage | enclosed ROI |
//! |---|---|---|
//! | 2D V-SHARP | 1.18 | 0.47 |
//! | 3D V-SHARP | 1.69 | **0.14** |
//! | 2D V-SHARP → 3D PDF | **0.99** | 0.24 |
//!
//! On the qsm-forward 2D phantom's `ses-thick` session, correlation against the local field
//! (higher is better), using the whole head for the enclosed case and a 16-slice slab cut so
//! the tissue reaches its end slices for the other:
//!
//! | | 48 mm slab | enclosed ROI |
//! |---|---|---|
//! | 2D V-SHARP | 0.9792 | **0.9597** |
//! | 3D V-SHARP | **0.9820** | 0.9506 |
//! | 2D V-SHARP → 3D PDF | 0.9791 | 0.8963 |
//!
//! The real-phantom differences are not large enough to decide anything, and on the enclosed
//! ROI the disc edges the sphere out. What does carry is that the chain, the arrangement
//! actually proposed for 2D data, is the worst of the three on the enclosed ROI and does not
//! improve on the 2D stage alone on the slab. Together with the synthetic case, where the
//! ground truth is exact and 3D wins the enclosed ROI by a factor of three, and with the
//! theory, there was no demonstrated benefit to ship. The open question is whether a real
//! acquisition has a slab thin enough relative to a useful kernel radius for the disc to earn
//! its place; see astewartau/QSM.rs#75.
//!
//! A slice **gap** (excited slabs thinner than the slice pitch) is refused rather than
//! approximated, because the spherical kernels are defined on a uniform grid; see
//! [`crate::Grid::require_contiguous_slices`].

pub mod smv;
pub mod sharp;
pub mod resharp;
pub mod vsharp;
pub mod pdf;
pub mod ismv;
pub mod lbv;
pub mod sdf;
pub mod iharperella;
#[cfg(feature = "onnx")]
pub mod bfrnet;

pub use smv::smv;
pub use sharp::{sharp, SharpParams};
pub use resharp::{resharp, ResharpParams};
pub use vsharp::{vsharp, VsharpGrid, VsharpParams};
pub use pdf::{pdf, DipoleKernelSpace, PdfParams};
pub use ismv::{ismv, IsmvParams};
pub use lbv::{lbv, LbvParams};
pub use sdf::{sdf, sdf_curvature, sdf_simple, SdfParams};
pub use iharperella::{
    harperella, iharperella,
    HarperellaInput, HarperellaParams, IharperellaParams,
};
#[cfg(feature = "onnx")]
pub use bfrnet::{bfrnet, bfrnet_tiled};

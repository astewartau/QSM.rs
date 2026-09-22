//! Anatomical segmentation.
//!
//! - **SynthSeg** ([`synthseg`], `onnx` feature): contrast-agnostic whole-brain segmentation
//!   into FreeSurfer labels. Because it is trained on synthesised contrasts it runs directly on
//!   a GRE magnitude image, so susceptibility can be reported per structure without acquiring
//!   and registering a separate T1w scan. See [`synthseg`](self::synthseg) for the caveats.
//!
//! Brain extraction (whole-head → brain mask) lives in [`crate::bet`].

pub mod labels;
pub mod synthseg;

pub use labels::{SynthSegLabels, SynthSegVersion};
#[cfg(feature = "onnx")]
pub use synthseg::synthseg;
pub use synthseg::{SynthSegParams, SynthSegResult};

//! Phase unwrapping methods
//!
//! This module provides various phase unwrapping algorithms:
//! - ROMEO: Region growing with quality-guided ordering
//! - Laplacian: Laplacian-based unwrapping

pub mod romeo;
pub mod laplacian;

pub use romeo::*;
pub use romeo::correct_multi_echo_wraps;
pub use laplacian::*;

/// Phase unwrapping method selection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UnwrapMethod {
    Romeo,
    Laplacian,
}

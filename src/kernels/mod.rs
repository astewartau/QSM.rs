//! Kernel functions for QSM processing
//!
//! This module provides various kernel implementations used in QSM algorithms:
//! - Dipole kernel for susceptibility-to-field relationship
//! - SMV kernel for spherical mean value filtering
//! - Laplacian kernel for second derivative operations

pub mod dipole;
pub mod smv;
pub mod laplacian;

pub use dipole::*;
pub use smv::*;
pub use laplacian::*;

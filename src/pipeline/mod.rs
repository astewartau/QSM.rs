//! Combined pipeline algorithms
//!
//! Algorithms that span multiple QSM pipeline stages (e.g., combined
//! phase unwrapping and background field removal).
//!
//! - HARPERELLA: SMV-based exterior Laplacian estimation (Li et al., 2014)
//! - iHARPERELLA: Phase-domain exterior estimation with improved low-freq suppression (Li et al., 2015)

pub mod iharperella;

pub use iharperella::{
    harperella, harperella_default, harperella_with_progress,
    iharperella, iharperella_default, iharperella_with_progress,
    HarperellaParams, IharperellaParams,
};

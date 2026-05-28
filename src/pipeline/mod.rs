//! QSM pipeline stages and utilities
//!
//! Shared stage functions that both qsmxt.rs and qsmbly call to ensure
//! identical processing. Each consumer calls the stages individually with
//! its own I/O and caching layer on top.
//!
//! ## Pipeline stages (typical order)
//!
//! 1. [`run_field_mapping`] — multi-echo phase → B0 field map (ppm)
//! 2. [`run_bg_removal`] — total field → local field (ppm)
//! 3. [`run_dipole_inversion`] — local field → susceptibility (ppm)
//! 4. [`apply_reference`] — mean subtraction
//!
//! For TGV, use [`run_tgv`] which combines steps 1-4 internally.
//!
//! ## Combined algorithms
//!
//! - HARPERELLA: SMV-based exterior Laplacian estimation (Li et al., 2014)
//! - iHARPERELLA: Phase-domain exterior estimation with improved low-freq suppression (Li et al., 2015)

// Combined algorithms
pub mod iharperella;

pub use iharperella::{
    harperella, harperella_default, harperella_with_progress,
    iharperella, iharperella_default, iharperella_with_progress,
    iharperella_with_weights,
    HarperellaParams, IharperellaParams,
};

// Pipeline stage modules
pub mod config;
pub mod phase_utils;
pub mod referencing;
pub mod field_mapping;
pub mod bg_removal;
pub mod inversion;

pub use config::*;
pub use phase_utils::{
    scale_phase_to_pi, hz_to_ppm, rads_to_ppm, rss_combine,
    erode_mask, dilate_mask,
};
pub use referencing::apply_reference;
pub use field_mapping::run_field_mapping;
pub use bg_removal::run_bg_removal;
pub use inversion::{run_dipole_inversion, run_tgv};

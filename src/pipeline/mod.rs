//! QSM pipeline stages and utilities
//!
//! Shared stage functions that both qsmxt.rs and qsmbly call to ensure
//! identical processing. Each consumer calls the stages individually with
//! its own I/O and caching layer on top.
//!
//! ## Pipeline stages (typical order)
//!
//! 0. [`run_motion_correction`] — optional: co-register the echo series before fitting across it
//! 1. [`run_field_mapping`] — multi-echo phase → B0 field map (ppm)
//! 2. [`run_bg_removal`] — total field → local field (ppm)
//! 3. [`run_dipole_inversion`] — local field → susceptibility (ppm)
//! 4. [`apply_reference`] — mean subtraction
//!
//! Two-pass artefact reduction ([`two_pass`]) runs stages 2-3 a second time against a mask whose
//! holes are intact, then folds the two reconstructions together.
//!
//! [`per_echo`] runs stages 1-4 once per echo instead ([`QsmPipelineConfig::qsm_per_echo`]): a
//! QSM for each echo, and their mean.
//!
//! For TGV, use [`run_tgv`] which combines steps 1-4 internally.
//!
//! [`run_qsmart`] reports its progress and intermediates (vessel mask, per-stage local fields and
//! χ) to an [`Observer`] as borrowed slices, so a host can show or save them without the stage
//! keeping copies.
//!
//! [`rules`] says what each method requires, recommends and discourages of the rest of the
//! configuration, with sources, and checks a configuration against it ([`rules::evaluate`]).
//!
//! ## Combined algorithms
//!
//! - HARPERELLA: SMV-based exterior Laplacian estimation (Li et al., 2014)
//! - iHARPERELLA: Phase-domain exterior estimation with improved low-freq suppression (Li et al., 2015)

// HARPERELLA / iHARPERELLA live in `bgremove` (their single canonical home);
// the pipeline uses them via `crate::bgremove::...`.

// Pipeline stage modules
pub mod config;
pub mod phase_utils;
pub mod referencing;
pub mod masking;
pub mod motion;
pub mod field_mapping;
pub mod bg_removal;
pub mod inversion;
pub mod separation;
pub mod observer;
pub mod qsmart;
pub mod two_pass;
pub mod per_echo;
pub mod rules;

pub use config::*;
pub use phase_utils::{
    scale_phase_to_pi, hz_to_ppm, rads_to_ppm, ppm_to_hz, ppm_to_rads, rss_combine,
};
pub use referencing::apply_reference;
pub use masking::{apply_mask_ops, build_mask_section, resolve_masking_input, run_masking, section_magnitude};
pub use motion::{run_motion_correction, MotionCorrectionConfig, MotionCorrectionResult};
pub use field_mapping::run_field_mapping;
pub use bg_removal::run_bg_removal;
pub use inversion::{run_dipole_inversion, run_tgv, run_nextqsm, run_iqsm, run_iqsm_plus, run_iqfm};
pub use separation::{run_separation, SeparationInputs, SeparationResult};
pub use observer::{Observer, VolumeRef};
pub use qsmart::run_qsmart;
pub use two_pass::{combine_two_pass, default_reliable_sections, restrict_reliable_mask};

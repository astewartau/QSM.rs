//! Physical constants shared across the crate.
//!
//! Use these wherever QSM.rs converts between radians, Hz and ppm itself, and wherever a host
//! converts QSM.rs's outputs (e.g. ppm → Hz for a BIDS `fieldmap`), so both agree.
//!
//! Ported methods that hard-code a different value in their authors' code (DECOMPOSE's 42.58,
//! TGV-QSM's 42.5781, HEIDI's 2.67522128e8 rad/s/T, QSMART's 2.675e8, ...) keep that value
//! locally, so the port matches the reference. Those sites say so; do not point them here.

/// Proton reduced gyromagnetic ratio γ̄ = γ/2π, in Hz/T.
///
/// 42.576 MHz/T, the value QSM.rs has always used and the one the QSM-CI contract states for
/// Hz ↔ ppm.
///
/// CODATA 2018 reference values:
/// - shielded proton in water (25 °C, spherical sample), γ'ₚ/2π = 42.576 384 74 MHz/T, which is
///   the relevant one for water-proton MRI;
/// - free proton, γₚ/2π = 42.577 478 518 MHz/T.
///
/// 42.576 is within 1e-5 (relative) of the shielded value, closer to both than the 42.58 some
/// toolboxes use. It is not set to the CODATA figure exactly because that would change every
/// output; that is a separate decision.
pub const GAMMA_BAR_HZ_PER_T: f64 = 42.576e6;

/// [`GAMMA_BAR_HZ_PER_T`] in MHz/T, for formulas written as `2π · γ̄[MHz/T] · B0 · TE` (ppm → rad).
pub const GAMMA_BAR_MHZ_PER_T: f64 = 42.576;

/// Proton gyromagnetic ratio γ = 2π·γ̄, in rad/s/T.
pub const GAMMA_RAD_PER_S_PER_T: f64 = 2.0 * std::f64::consts::PI * GAMMA_BAR_HZ_PER_T;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn units_agree() {
        assert!((GAMMA_BAR_MHZ_PER_T * 1e6 - GAMMA_BAR_HZ_PER_T).abs() < 1e-6);
        assert!((GAMMA_RAD_PER_S_PER_T - 2.675_128_97e8).abs() < 1.0);
    }
}

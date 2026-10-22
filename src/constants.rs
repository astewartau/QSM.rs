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
/// Hz ↔ ppm. CODATA 2018 gives 42.577 478 MHz/T for the free proton and 42.576 385 MHz/T for the
/// proton shielded in water, which is what tissue protons are closer to; the differences are far
/// below anything a field map resolves, and changing the value would change every output.
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
        assert_eq!(GAMMA_BAR_MHZ_PER_T * 1e6, GAMMA_BAR_HZ_PER_T);
        assert!((GAMMA_RAD_PER_S_PER_T - 2.675_164_4e8).abs() < 1e2);
    }
}

//! Susceptibility source separation
//!
//! This module provides algorithms for separating total magnetic susceptibility
//! into paramagnetic (χ+, primarily iron) and diamagnetic (χ-, primarily myelin)
//! components using local field maps and R2' relaxation data.
//!
//! # Methods
//! - `chi_sep_ilsqr`: the original Shin 2021 projected-CG algorithm (SNU-LIST
//!   toolbox `chi_sep_iLSQR`), initialized from a conventional QSM
//! - `chi_sep_medi`: MEDI-based Gauss-Newton optimization with coupled field + R2' constraints
//! - `r2star_qsm`: closed-form separation from a QSM + R2* (Dimov 2022, no R2' needed)
//! - `wavesep`: wavelet-L1 proximal-gradient separation from a QSM + R2' (Fang 2023)
//! - `decompose`: signal-domain 3-compartment per-voxel fit from a QSM + multi-echo
//!   magnitude (Chen 2021)
//! - `hc_chisep`: hollow-cylinder χ-separation with signal-derived fiber orientation
//!   from a QSM + R2' + multi-echo magnitude (Wharton & Bowtell model)
//!
//! # Reference
//! Shin, H., et al. (2021). "χ-separation: Magnetic susceptibility source separation
//! toward iron and myelin mapping in the brain." NeuroImage, 240:118371.

/// χ-separation's relaxometric constant D_r, in Hz/ppm, as published by Shin et al. 2021.
///
/// The forward model is `R2' = D_r,pos·|χ+| + D_r,neg·|χ−|`, and both published derivations give
/// the two sources the *same* relaxivity — neither paper fits them separately. 137 Hz/ppm comes
/// from linear regression of R2' against **single-orientation** QSM in iron-rich deep grey matter
/// (Shin et al. 2021, NeuroImage 240:118371), which is the reconstruction every analytic method
/// in this module is given, so it is their default.
///
/// [`DR_KIM_2025_COSMOS`] is the same quantity re-measured against COSMOS. Which is correct for a
/// given dataset is an open question in the field; what is not open is that a value measured
/// against one kind of QSM belongs with that kind of QSM. Changing one method's default without
/// the others makes two variants of the same method incomparable on the same data — which is the
/// state QSM.rs#130 was filed about — so these are shared constants rather than four literals.
pub const DR_SHIN_2021: f64 = 137.0;

/// χ-separation's relaxometric constant D_r re-measured against COSMOS, in Hz/ppm.
///
/// Kim et al. 2025 (Human Brain Mapping 46(4):e70136) repeated Shin's regression using
/// COSMOS-reconstructed χ maps instead of single-orientation QSM and obtained 114 Hz/ppm,
/// attributing the difference to the QSM algorithm behind the training data. The SNU-LIST
/// toolbox's own script carries the same note against this value.
///
/// It is the right constant for a COSMOS-referenced χ — which is what the χ-sepnet family of
/// networks was trained on, and why
/// [`ChiSepNetNorm`](chisepnet::ChiSepNetNorm) and
/// [`R2PrimeNetNorm`](crate::relaxometry::R2PrimeNetNorm) bake 114 in as a fixed training
/// normalisation rather than a tunable. Those are deliberately *not* defined in terms of this
/// constant: a network's normalisation cannot move when the field's best estimate of the physical
/// relaxivity does, and tying them together would invite exactly that edit.
pub const DR_KIM_2025_COSMOS: f64 = 114.0;

pub mod chi_sep_ilsqr;
pub mod chi_sep_medi;
pub mod decompose;
pub mod hc_chisep;
pub mod r2star_qsm;
pub mod wavesep;
#[cfg(feature = "onnx")]
pub mod susep_net;
#[cfg(feature = "onnx")]
pub mod chisepnet;

pub use chi_sep_ilsqr::{chi_sep_ilsqr, ChiSepIlsqrParams};
pub use chi_sep_medi::{chi_sep_medi, ChiSepParams};
pub use decompose::{decompose, DecomposeParams};
pub use hc_chisep::{hc_chisep, HcChisepParams};
pub use r2star_qsm::{r2star_qsm, r2star_qsm_from_magnitude, R2starQsmParams};
pub use wavesep::{wavesep, WaveSepParams};
#[cfg(feature = "onnx")]
pub use susep_net::{susep_net, SusepNetNorm, SusepNetParams};
#[cfg(feature = "onnx")]
pub use chisepnet::{chisepnet, ChiSepNetNorm, ChiSepNetParams};

#[cfg(test)]
mod tests {
    use super::*;

    /// The four analytic methods must default to the same relaxivity.
    ///
    /// D_r is a physical constant of the forward model, not a tuning knob: when two variants of
    /// χ-separation default to different values, the same data analysed with each is not
    /// comparable, and nothing in the output says why. `chi_sep_medi` sat at 114/30 against the
    /// others' 137/137 until QSM.rs#130 — undetected because no test asserted they agreed.
    #[test]
    fn analytic_methods_share_one_relaxivity() {
        let ilsqr = ChiSepIlsqrParams::default();
        let medi = ChiSepParams::default();
        let wave = WaveSepParams::default();
        let hc = HcChisepParams::default();

        for (name, dr) in [
            ("chi_sep_ilsqr.dr_pos", ilsqr.dr_pos),
            ("chi_sep_ilsqr.dr_neg", ilsqr.dr_neg),
            ("chi_sep_medi.dr_pos", medi.dr_pos),
            ("chi_sep_medi.dr_neg", medi.dr_neg),
            ("wavesep.dr_pos", wave.dr_pos),
            ("wavesep.dr_neg", wave.dr_neg),
            ("hc_chisep.dr_pos_3t", hc.dr_pos_3t),
        ] {
            assert_eq!(
                dr, DR_SHIN_2021,
                "{name} defaults to {dr} Hz/ppm, not the shared {DR_SHIN_2021}. If that is \
                 deliberate, say which paper it comes from here; if it is not, it makes this \
                 method's output incomparable with the others' on the same data."
            );
        }
    }

    /// The two published values are distinct and neither drifted.
    ///
    /// They are not interchangeable: 137 was regressed against single-orientation QSM, 114
    /// against COSMOS. A future edit that "unifies" them would silently re-reference every
    /// analytic method's χ+/χ− to a reconstruction its input is not.
    #[test]
    fn published_relaxivities_are_distinct() {
        assert_eq!(DR_SHIN_2021, 137.0, "Shin et al. 2021, NeuroImage 240:118371");
        assert_eq!(DR_KIM_2025_COSMOS, 114.0, "Kim et al. 2025, Hum Brain Mapp 46(4):e70136");
        assert_ne!(DR_SHIN_2021, DR_KIM_2025_COSMOS);
    }

    /// The χ-sepnet family's 114 is a baked training normalisation, not the physical relaxivity,
    /// so it must keep its own literal rather than following [`DR_SHIN_2021`] when that moves.
    #[cfg(feature = "onnx")]
    #[test]
    fn network_normalisation_is_not_the_analytic_default() {
        assert_eq!(ChiSepNetNorm::default().dr, DR_KIM_2025_COSMOS);
        assert_eq!(crate::relaxometry::R2PrimeNetNorm::default().dr, DR_KIM_2025_COSMOS);
    }
}

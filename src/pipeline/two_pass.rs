//! Two-pass artefact reduction.
//!
//! Strong susceptibility sources (air–tissue interfaces, haemorrhage, implants) leave the phase
//! unrecoverable around them, and a phase-quality mask drops those voxels — leaving holes. Filling
//! the holes and reconstructing once puts unreliable field values back into the inversion, and the
//! streaking that follows spreads well beyond the hole. Dropping them instead leaves nothing to
//! report where the source actually is.
//!
//! Two-pass does both and keeps the better half of each: reconstruct once on the mask with its
//! holes intact ("reliable"), once on the mask the user configured ("filled"), then take the
//! reliable reconstruction wherever it is defined and the filled one everywhere else.
//!
//! Stewart AW, Robinson SD, O'Brien K, et al. QSMxT: Robust masking and artifact reduction for
//! quantitative susceptibility mapping. Magn Reson Med. 2022;87(3):1289-1300.
//! <https://doi.org/10.1002/mrm.29048>
//!
//! The orchestration — running the two passes, caching them, naming their outputs — belongs to
//! the consumer, since each drives the pipeline stages itself. What lives here is the part both
//! consumers must agree on: the reliable-mask recipes ([`reliable_sections`]), where the
//! reliable pass's local field comes from ([`ReliableBackground`]), the domain the reliable mask
//! is restricted to ([`restrict_reliable_mask`]), and the combination itself.
//!
//! ## Evidence behind the options (QSM QC, Shi subject-6 streak phantom, November 2026)
//!
//! Measured on the Shi et al. subject-6 phantom with 24 strong sources (6 calcifications,
//! 3 haemorrhages, 15 microbleeds) and 4 veins, 6 echoes and a single-echo copy (echo 3,
//! TE 18.8 ms), V-SHARP 12–2 mm and iLSQR; RMSE in ppb against the true χ after matching means
//! more than 25 mm from any source; "near" is within 12 mm of a source. The reliable pass was
//! emulated with the stages run on the run's own total field; the emulation reproduces the real
//! reliable pass exactly.
//!
//! - **The Otsu reliable mask is full of tiny holes.** Otsu over the whole field of view lands
//!   at the brain/background valley of the phase-quality histogram (40–43 on every phantom map)
//!   and leaves 1.5–4 % of brain voxels below it, mostly isolated voxels (5,259 of 6,815 hole
//!   components on 6 echoes, 5,480 of 7,521 on one echo are single voxels).
//! - **Re-running background removal on that mask is what costs deep grey matter**
//!   ([`ReliableBackground::Separate`]). V-SHARP keeps a voxel only where its smallest ball fits
//!   inside the mask, so every hole grows by about 2 mm and the kernel radius near holes falls
//!   from 12 mm towards 2 mm. The reliable pass covers 80–84 % of the brain instead of 97–98 %,
//!   and the pallidum, entirely inside the reliable mask, goes from 38 to 66 ppb (6 echoes) and
//!   42 to 70 ppb (one echo), mean bias −35 → −62 ppb, because V-SHARP's per-voxel radius in deep
//!   grey matter falls to about 7 mm and its local-field contrast in the nuclei falls by 30–55 %.
//!   The same holes applied only at inversion, on the main pass's local field
//!   ([`ReliableBackground::Shared`]), give pallidum 38.8 / 42.2 ppb, equal to a single pass, take
//!   far-from-source error below a single pass, remove the 2 mm single-pass rim around each hole,
//!   and keep most of the near-source gain (one echo: near 140 single pass, 123 separate, 112
//!   shared; 6 echoes: 86, 70, 73). Non-eroding background removal (PDF) shows no penalty either
//!   way (pallidum 27.9 single pass, 27.0 two-pass). Caveat of the shared field: near a hole,
//!   the main pass's V-SHARP balls include the excluded voxels, so unreliable field can leak up to
//!   the kernel radius; on the phantom this did no measurable harm.
//! - **A fixed, deep phase-quality threshold picks the holes better than Otsu**
//!   ([`ReliableThreshold::Fixed`], [`RELIABLE_FIXED_THRESHOLD`]). Phantom sources sit at
//!   phase quality 2–9, deep grey nuclei at 57–92, Otsu at about 40. A threshold of 15 leaves a
//!   wide margin below anything the pallidum is likely to reach and makes far fewer holes (one
//!   echo: 2,098 holes, 10.4 ml, against Otsu's 7,521, 43.9 ml). With the shared field it gives the
//!   best one-echo map (near 99, pallidum 41.9) and 6-echo near 69 with pallidum at single-pass
//!   level (38.7); the large haemorrhage's interior partly stays in the reliable pass on 6
//!   echoes. With the separate field it still erodes (one echo: pallidum 54.5).
//! - Matching an offset between the passes (QSMART's `adjust_offset`) changed near or rim error
//!   by less than 1 ppb in every case, so the combination stays offset-free.
//!
//! Untested: real lesions and real pallidum, 7 T, bipolar readouts, anisotropic voxels, other
//! inversions (TV, TKD), one phantom seed. The defaults therefore stay the published algorithm
//! (Otsu, separate background removal) until in-vivo results decide.
//!
//! A future option, not implemented: shape-selective holes from vesselness, as QSMART chooses
//! its stage-2 exclusions (Frangi vesselness on the black top-hat of the N4 mean-echo magnitude,
//! [`crate::utils::vasculature`]). On the phantom it takes veins (about 86 % of each), every
//! calcification and microbleed and none of the pallidum, but only 10 % of a 1.6 ml haemorrhage
//! (the 8-voxel top-hat misses large blobs), so it would complement a phase-quality threshold
//! (their union) rather than replace it.

use super::config::{
    MaskOp, MaskSection, MaskThresholdMethod, MaskingInput, PipelineError,
};

/// The phase-quality level of [`ReliableThreshold::Fixed`]'s recommended recipe, on ROMEO's
/// 0–100 voxel-quality scale: voxels at or below it are holes. Chosen from the phantom
/// measurements in the module docs: strong sources sit at 2–9, deep grey nuclei at 57–92.
pub const RELIABLE_FIXED_THRESHOLD: f64 = 15.0;

/// How the reliable-pass mask picks its holes from the ROMEO phase-quality map.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ReliableThreshold {
    /// Otsu's threshold over the whole field of view: the published recipe and the default. It
    /// lands at the brain/background valley, so it also drops scattered low-quality brain
    /// voxels as single-voxel holes (module docs).
    Otsu,
    /// A fixed level on the 0–100 phase-quality scale ([`RELIABLE_FIXED_THRESHOLD`] is the one
    /// the evidence supports): holes only where the phase is far below anything tissue reaches,
    /// whatever the scan's histogram.
    Fixed(f64),
}

/// The reliable-pass mask recipe: a threshold on the ROMEO phase-quality map, and nothing else.
///
/// The refinements the `robust-threshold` recipe applies — dilate, fill-holes, erode — are
/// deliberately absent: the holes *are* the signal here, and filling them would collapse this
/// pass onto the other one.
pub fn reliable_sections(threshold: ReliableThreshold) -> Vec<MaskSection> {
    let (method, value) = match threshold {
        ReliableThreshold::Otsu => (MaskThresholdMethod::Otsu, None),
        ReliableThreshold::Fixed(level) => (MaskThresholdMethod::Fixed, Some(level)),
    };
    vec![MaskSection {
        input: MaskingInput::PhaseQuality,
        generator: MaskOp::Threshold { method, value },
        refinements: vec![],
    }]
}

/// The reliable-pass mask recipe used when the consumer does not supply one: Otsu on phase
/// quality ([`ReliableThreshold::Otsu`]), the published recipe.
pub fn default_reliable_sections() -> Vec<MaskSection> {
    reliable_sections(ReliableThreshold::Otsu)
}

/// Where the reliable pass's local field comes from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReliableBackground {
    /// Background removal is run again, on the reliable mask, and the inversion inverts that
    /// local field: the published algorithm, and the default. Eroding background removal (V-SHARP,
    /// SHARP, RESHARP) grows every hole by its smallest kernel and shrinks the kernel near holes,
    /// which lowers deep-grey-matter contrast when the reliable mask has many small holes (module
    /// docs).
    #[default]
    Separate,
    /// Background removal runs once, for the main pass, and the reliable pass inverts that local
    /// field inside the reliable mask: the holes are applied at inversion only. The reliable mask
    /// is restricted to where the main pass's local field is defined
    /// ([`restrict_reliable_mask`]), and the reliable pass is defined on all of it.
    Shared,
}

impl ReliableBackground {
    /// Stable id, as hosts name it.
    pub fn id(self) -> &'static str {
        match self {
            Self::Separate => "separate",
            Self::Shared => "shared",
        }
    }

    /// Whether the reliable pass runs background removal of its own.
    pub fn reruns_background_removal(self) -> bool {
        self == Self::Separate
    }
}

/// Restrict the reliable mask to the region being reconstructed.
///
/// The reliable mask decides where the reliable pass's values are *preferred*, so anywhere it
/// reaches past the main mask it would promote a reconstruction over a region the caller excluded
/// outright. Being a subset of the reconstruction domain is a property of the method rather than
/// a setting, so this is applied whatever recipe the two masks came from.
///
/// `main` is the main pass's mask with [`ReliableBackground::Separate`], and the mask the main
/// pass's background removal left (where its local field is defined) with
/// [`ReliableBackground::Shared`]: the reliable pass inverts that field, so it can be defined
/// nowhere else.
pub fn restrict_reliable_mask(reliable: &[u8], main: &[u8]) -> Result<Vec<u8>, PipelineError> {
    if reliable.len() != main.len() {
        return Err(PipelineError::DimensionMismatch {
            expected: main.len(),
            got: reliable.len(),
        });
    }
    Ok(reliable.iter().zip(main).map(|(&r, &m)| r & m).collect())
}

/// Combine the two passes: the reliable reconstruction where it is defined, the filled one
/// elsewhere.
///
/// `support` is where `reliable` carries a reconstruction — not the mask the pass was *asked*
/// for. Background removal erodes, so the two differ by a rim of voxels in which `reliable` is
/// zero; selecting on the requested mask would keep those zeros and ring the holes with a seam of
/// empty voxels. Passing the support the inversion actually produced makes the seam impossible.
pub fn combine_two_pass(
    reliable: &[f64],
    filled: &[f64],
    support: &[u8],
) -> Result<Vec<f64>, PipelineError> {
    if reliable.len() != filled.len() {
        return Err(PipelineError::DimensionMismatch {
            expected: reliable.len(),
            got: filled.len(),
        });
    }
    if support.len() != reliable.len() {
        return Err(PipelineError::DimensionMismatch {
            expected: reliable.len(),
            got: support.len(),
        });
    }

    Ok(reliable
        .iter()
        .zip(filled)
        .zip(support)
        .map(|((&r, &f), &s)| if s != 0 { r } else { f })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_recipe_leaves_the_holes_alone() {
        let sections = default_reliable_sections();
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].input, MaskingInput::PhaseQuality);
        assert!(matches!(
            sections[0].generator,
            MaskOp::Threshold { method: MaskThresholdMethod::Otsu, value: None }
        ));
        // Filling or closing here would defeat the purpose of the pass.
        assert!(sections[0].refinements.is_empty());
    }

    #[test]
    fn a_fixed_threshold_recipe_keeps_its_level_and_its_holes() {
        let sections = reliable_sections(ReliableThreshold::Fixed(RELIABLE_FIXED_THRESHOLD));
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].input, MaskingInput::PhaseQuality);
        assert!(matches!(
            sections[0].generator,
            MaskOp::Threshold { method: MaskThresholdMethod::Fixed, value: Some(v) } if v == 15.0
        ));
        assert!(sections[0].refinements.is_empty());
        assert_eq!(default_reliable_sections(), reliable_sections(ReliableThreshold::Otsu));
    }

    #[test]
    fn the_published_background_handling_is_the_default() {
        assert_eq!(ReliableBackground::default(), ReliableBackground::Separate);
        assert!(ReliableBackground::Separate.reruns_background_removal());
        assert!(!ReliableBackground::Shared.reruns_background_removal());
        assert_eq!(ReliableBackground::Shared.id(), "shared");
    }

    #[test]
    fn combine_picks_each_pass_where_it_belongs() {
        let reliable = [1.0, 2.0, 3.0, 4.0];
        let filled = [10.0, 20.0, 30.0, 40.0];
        let support = [1u8, 0, 1, 0];
        let out = combine_two_pass(&reliable, &filled, &support).unwrap();
        assert_eq!(out, vec![1.0, 20.0, 3.0, 40.0]);
    }

    /// A zero inside the reliable pass's support is a real reconstructed value, not a gap: the
    /// filled pass must not overwrite it.
    #[test]
    fn a_reconstructed_zero_is_kept() {
        let out = combine_two_pass(&[0.0], &[9.0], &[1]).unwrap();
        assert_eq!(out, vec![0.0]);
    }

    /// The eroded rim belongs to the filled pass — that is the point of selecting on the support
    /// rather than on the mask the pass was asked for.
    #[test]
    fn the_eroded_rim_comes_from_the_filled_pass() {
        // Mask asked for [1,1,1]; background removal eroded it to [0,1,0].
        let reliable = [0.0, 5.0, 0.0];
        let filled = [7.0, 8.0, 9.0];
        let out = combine_two_pass(&reliable, &filled, &[0, 1, 0]).unwrap();
        assert_eq!(out, vec![7.0, 5.0, 9.0]);
    }

    #[test]
    fn combine_rejects_mismatched_lengths() {
        assert!(combine_two_pass(&[1.0, 2.0], &[1.0], &[1, 1]).is_err());
        assert!(combine_two_pass(&[1.0, 2.0], &[1.0, 2.0], &[1]).is_err());
    }

    #[test]
    fn restrict_intersects() {
        let out = restrict_reliable_mask(&[1, 1, 0, 1], &[1, 0, 1, 0]).unwrap();
        assert_eq!(out, vec![1, 0, 0, 0]);
    }

    #[test]
    fn restrict_rejects_mismatched_lengths() {
        assert!(restrict_reliable_mask(&[1, 1], &[1]).is_err());
    }
}

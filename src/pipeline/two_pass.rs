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
//! consumers must agree on: the default reliable-mask recipe, and the combination itself.

use super::config::{
    MaskOp, MaskSection, MaskThresholdMethod, MaskingInput, PipelineError,
};

/// The reliable-pass mask recipe used when the consumer does not supply one.
///
/// An Otsu threshold on the ROMEO phase-quality map, and nothing else. The refinements the
/// `robust-threshold` recipe applies — dilate, fill-holes, erode — are deliberately absent: the
/// holes *are* the signal here, and filling them would collapse this pass onto the other one.
pub fn default_reliable_sections() -> Vec<MaskSection> {
    vec![MaskSection {
        input: MaskingInput::PhaseQuality,
        generator: MaskOp::Threshold {
            method: MaskThresholdMethod::Otsu,
            value: None,
        },
        refinements: vec![],
    }]
}

/// Restrict the reliable mask to the region being reconstructed.
///
/// The reliable mask decides where the reliable pass's values are *preferred*, so anywhere it
/// reaches past the main mask it would promote a reconstruction over a region the caller excluded
/// outright. Being a subset of the reconstruction domain is a property of the method rather than
/// a setting, so this is applied whatever recipe the two masks came from.
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

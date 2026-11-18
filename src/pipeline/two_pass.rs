//! Two-pass artefact reduction.
//!
//! Strong susceptibility sources (haemorrhage, calcification, large veins, implants) leave the
//! phase unrecoverable around them, and a phase-quality mask drops those voxels — leaving holes.
//! Filling the holes and reconstructing once puts unreliable field values back into the
//! inversion, and the streaking that follows spreads well beyond the hole. Dropping them instead
//! leaves nothing to report where the source actually is.
//!
//! Two-pass does both: reconstruct once on the mask the user configured (the "one-pass" or
//! "filled" map, χ₁), once more on a mask with holes at the unreliable voxels (the "second" or
//! "reliable" pass, χ₂), then merge the two. There are two ways to do this ([`TwoPassMode`]):
//!
//! - [`TwoPassMode::Original`], the published algorithm (Stewart AW, Robinson SD, O'Brien K, et
//!   al. QSMxT: Robust masking and artifact reduction for quantitative susceptibility mapping.
//!   Magn Reson Med. 2022;87(3):1289-1300, <https://doi.org/10.1002/mrm.29048>): the holes are
//!   every voxel below an Otsu threshold of ROMEO phase quality ([`default_reliable_sections`]),
//!   background removal and inversion are run again on the holed mask, and the second pass is
//!   taken wherever it has a value ([`combine_two_pass`]).
//! - [`TwoPassMode::Local`], the default: only holes with a core of very low phase quality are
//!   kept ([`select_source_holes`]), the second pass's background removal does not leave an
//!   eroded rim around them ([`fill_eroded_rim`]), and the second pass is used only near its
//!   holes, feathered into the one-pass map ([`local_weights`], [`combine_two_pass_weighted`]).
//!
//! The orchestration — running the two passes, caching them, naming their outputs — belongs to
//! the consumer, since each drives the pipeline stages itself. What lives here is the part both
//! consumers must agree on: the reliable-mask recipe, the hole selection, the rim fill and the
//! merge.
//!
//! ## Why `Local` replaced `Original` as the default (QSMxT v12.1, decision 174)
//!
//! Source: QSM QC finding `docs/findings/2026-11-15-two-pass-reliable-mask.md` (qsm-qc repo),
//! sections "Few, real holes" and "Recipe for QSM.rs". Ashley judged the variant tested there as
//! "G" better than the published two-pass and asked for it as the default, with the published
//! algorithm kept as an option.
//!
//! **What was wrong with `Original`.** Otsu over the field of view lands at the brain/background
//! valley (phase quality 40-44 on every scan tested), so it leaves thousands of hole components
//! inside the brain (clinical single echo: 1,876-2,696 inside the one-pass support, 20-37 ml;
//! six-echo MULTIMS 325-510). V-SHARP keeps a voxel only where its smallest (2 mm) ball fits in the
//! mask, so each hole grows by ~2 mm (second pass on 75-81 % of the brain instead of 93-96 %), and
//! near holes its kernel radius drops from 12 mm towards 2 mm, which attenuates the local field in
//! the deep grey nuclei. In vivo the damage comes mostly from the large vein and sinus holes within
//! V-SHARP's reach of the deep grey matter: `Original` lowers mean pallidal χ by 19-63 ppb in 10 of
//! 11 scans, including six-echo controls with almost no holes. On the real sources it does help:
//! the dark halo 1-8 mm around a lesion is pulled towards zero.
//!
//! **What `Local` does, step by step** (as tested; the constants are [`LocalTwoPassParams`]):
//! 1. Reliable mask: ROMEO phase quality, Otsu over the field of view (the recipe is unchanged,
//!    [`default_reliable_sections`]), then holes of up to 27 voxels filled (as the
//!    `fill-holes:27` refinement). Holes = main mask ∩ one-pass support ∖ that mask.
//! 2. Opening of the holes with the 6-neighbour cross, 2 iterations (cuts vessels up to about 4
//!    voxels wide); 6-connected components of what is left; a component is kept if at least 20 of
//!    its voxels have phase quality < 5 (a "deep core": normal pallidal holes held 0-16 such
//!    voxels, real sources more). The final holes are the original holes within 2 voxels
//!    (6-neighbour dilation × 2) of a kept component, so each source gets back its own rim. No
//!    segmentation is needed. Holes outside the one-pass support stay holes: they join the
//!    brain-edge rim and do not change the merged map.
//! 3. Second-pass background removal on the main mask minus the holes. V-SHARP (or whichever
//!    method is configured) as the main pass; where its eroded support has no value but the mask
//!    does (V-SHARP's ~2 mm rim around each hole), LBV (default parameters, peel 0) on the same
//!    mask and total field. Inversion on that support ∩ the one-pass support.
//! 4. Merge: weight w = 1 within 6 mm of a hole, falling linearly to 0 at 12 mm (V-SHARP's largest
//!    radius); χ = w·χ₂ + (1 − w)·χ₁ where the second pass has a value, χ₁ in the holes and
//!    everywhere else. No offset between the passes (a QSMART-style offset fit changed near- or
//!    rim-error by < 1 ppb in every case). Referencing then runs over the one-pass support.
//!
//! **Evidence** (iLSQR unless noted; the second pass was emulated with `qsmxt bgremove` /
//! `qsmxt invert` on each run's own total field and checked against QSMxT's own two-pass to
//! ≤ 0.001 ppb):
//!
//! | Measure | One pass | `Original` | `Local` |
//! |---|---|---|---|
//! | Phantom (Shi subject 6, 24 strong sources), single echo TE 18.8 ms, RMSE ppb near / rim 0.5-3 mm / pallidum / far | 140 / 392 / 42 / 14 | 123 / 390 / 70 / 15 | 114 / 380 / 44 / 14 |
//! | Same phantom, six echoes | 86 / 231 / 38 / 13 | 70 / 226 / 66 / 15 | 63 / 206 / 38 / 13 |
//! | In vivo, 11 scans (8 clinical 3T single echo TE 20 ms, 3 six-echo MULTIMS): pallidal mean χ minus one pass | — | −10 to −63 ppb (≥ 19 ppb low in 10 of 11) | within 4 ppb in 9 of 11 (z05 −10, z04 −20) |
//! | Mean χ 1-8 mm from the three real lesions (z01 / z05 / z07), ppb | −95 / −63 / −20 | −61 / −29 / −10 | −37 / −15 / +2 |
//! | Spatial SD there (z01 / z05 / z07), ppb | 204 / 140 / 102 | 188 / 124 / 88 | 176 / 113 / 85 |
//! | TV (λ 2e-4), z01 / z05: pallidum minus one pass; near-lesion mean | 152 / 114; −106 / −69 | −77 / −52; −69 / −30 | −4 / −10; −40 / −11 |
//!
//! Real sources stay holes (z01 92 %, z04 lesion 90 %, z05 96 % of the source region); veins and
//! sinuses no longer are (0-15 %); holes inside the one-pass support drop from thousands to 0-9
//! per clinical scan (0 on the six-echo scans, where two-pass then equals one pass).
//!
//! **Known limits.** No variant tested was better than `Original` on every case and measure:
//! - z04, whose pallidum is itself a hole with a deep core: pallidum −20 ppb (`Original` −10), as
//!   the LBV rim then sits inside the nucleus. z05's pallidum, within ~12 mm of its lesion, is
//!   −10 ppb (`Original` −44).
//! - Veins are no longer holes, so the dark halos around them are the one-pass ones again (z03,
//!   z10: 8-9 ppb darker near veins than `Original`, as one pass).
//! - The dark band right around a lesion is the phase-quality hole itself (1-2 mm past the bright
//!   core, filled with one-pass values); `Local` narrows it but cannot remove it.
//! - The deep-core numbers (phase quality < 5, 20 voxels, opening × 2) and the 6-12 mm feathering
//!   were set from these 11 scans and one phantom. Untested: 7T, other echo times, bipolar
//!   readouts, anisotropic voxels, other background-removal methods than V-SHARP.
//!
//! Other variants tested and not taken: holes applied at inversion only, on the main pass's local
//! field (fixes the pallidum but loses most of the near-source gain, and does nothing with
//! inversions whose data term ignores the mask, e.g. TV); a fixed phase-quality threshold of 15
//! (does not keep real pallida out on single echo, lets half of each real source in); hole
//! filling alone (`fill-holes:8/27/64`: leaves the in-vivo pallidal penalty almost unchanged);
//! PDF or LBV for the whole second pass (far-field error 31-61 ppb against 12-15 for V-SHARP).
//!
//! ## Which inversions the second pass changes
//!
//! With either mode, when background removal is a separate step it is run again on the holed
//! mask, so the second pass's local field is zero in the holes and the inversion gets the holed
//! support as its mask. How an inversion treats the holes then depends on its data term:
//! - **holes carry no data** (the mask weights the data term): iLSQR, LSQR, NLTV, NDI, FANSI,
//!   L1-QSM, HD-QSM, MEDI, STAR-QSM, AMP-PE, and WH-QSM with a masked weight;
//! - **holes are read as a measured zero field** (data term over the whole grid, the mask only
//!   applied to the output): TKD, TSVD, Tikhonov, TV, RTS, WH-QSM's default flat weight, and
//!   NeXtQSM's unrolled data term. The second pass still differs from the first (QSM QC measured
//!   TV above), but the hole is not excluded from the fit;
//! - **deep-learning inversions of the local field** (xQSM, QSMnet, QSMnet-ME, QSMGAN, IR2QSM,
//!   LPCNN, MoDL-QSM) see the zeroed field.
//!
//! The total-field methods (TGV, QSMART, TFI, AutoQSM, NeXtQSM, iQSM, iQSM+, iQFM) remove the
//! background themselves, so there is no separate background removal and no rim to fill
//! ([`fill_eroded_rim`] does not apply). The reliable mask is their domain (TGV, eroded), the
//! domain of their inner iLSQR (QSMART), their data weight (TFI), or a network input channel
//! (iQSM, iQSM+, iQFM). AutoQSM applies the mask only to its output, so its second pass equals
//! the first inside the mask and two-pass changes nothing.

use super::config::{
    MaskOp, MaskSection, MaskThresholdMethod, MaskingInput, PipelineError, ScanMetadata,
};
use crate::utils::connected::{label_components, Connectivity};
use crate::utils::distance::distance_to_set_mm;
use crate::utils::morphology::{dilate_mask, erode_mask, fill_holes};
use crate::Grid;

/// How the two passes are made and merged (see the module docs for both and the evidence).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum TwoPassMode {
    /// Holes only at strong sources ([`select_source_holes`]), no eroded rim around them
    /// ([`fill_eroded_rim`]), second pass used only within 6-12 mm of its holes
    /// ([`combine_two_pass_weighted`]). The default since decision 174.
    #[default]
    Local,
    /// The published algorithm (Stewart et al. 2022): every voxel below the reliable mask's
    /// threshold is a hole, and the second pass is taken wherever it has a value
    /// ([`combine_two_pass`]).
    Original,
}

impl TwoPassMode {
    /// Every mode, the default first.
    pub const ALL: [TwoPassMode; 2] = [Self::Local, Self::Original];

    /// The mode's name in configuration files and on command lines.
    pub fn id(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Original => "original",
        }
    }

    /// The mode named `id` ([`Self::id`]).
    pub fn from_id(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.id() == id)
    }
}

/// The constants of [`TwoPassMode::Local`], as tested by QSM QC (module docs). They were set from
/// 11 in-vivo scans and one phantom; they are fields so they can be studied, not user settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalTwoPassParams {
    /// Holes of the reliable mask of at most this many voxels are filled first (6-connected,
    /// not touching the volume edge: [`fill_holes`]). 27.
    pub fill_holes: usize,
    /// Iterations of the opening (6-neighbour cross) that cuts thin vessels off the holes. 2.
    pub opening_iterations: usize,
    /// Phase quality (0-100, ROMEO voxel quality) below which a voxel counts towards a deep core. 5.
    pub core_quality: f64,
    /// Voxels below [`core_quality`](Self::core_quality) an opened hole component needs to be
    /// kept. 20.
    pub core_voxels: usize,
    /// Each kept component gets back the original hole voxels within this many 6-neighbour
    /// dilations of it. 2.
    pub rim_dilations: usize,
    /// Distance from a hole (mm) within which the second pass is used alone. 6.
    pub full_weight_mm: f64,
    /// Distance from a hole (mm) at which the second pass's weight reaches 0. 12, V-SHARP's
    /// default largest radius: beyond it the holes cannot change V-SHARP's local field.
    pub zero_weight_mm: f64,
}

impl Default for LocalTwoPassParams {
    fn default() -> Self {
        Self {
            fill_holes: 27,
            opening_iterations: 2,
            core_quality: 5.0,
            core_voxels: 20,
            rim_dilations: 2,
            full_weight_mm: 6.0,
            zero_weight_mm: 12.0,
        }
    }
}

/// The reliable-pass mask recipe used when the consumer does not supply one.
///
/// An Otsu threshold on the ROMEO phase-quality map, and nothing else, in both modes. The
/// refinements the `robust-threshold` recipe applies — dilate, fill-holes, erode — are
/// deliberately absent: the holes *are* the signal here, and filling them would collapse this
/// pass onto the other one. [`TwoPassMode::Local`] fills holes of up to 27 voxels itself, as the
/// first step of [`select_source_holes`], so the recipe is the same for both modes.
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

/// The reliable mask [`select_source_holes`] made, and what it kept.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceHoles {
    /// The reliable mask: the main mask minus the kept holes and minus the holes outside the
    /// one-pass support.
    pub reliable: Vec<u8>,
    /// Hole voxels inside the one-pass support before selection (after hole filling).
    pub candidate_voxels: usize,
    /// Opened hole components with a deep core: the sources kept as holes.
    pub sources: usize,
    /// Hole voxels inside the one-pass support after selection.
    pub hole_voxels: usize,
}

/// Keep only the holes at strong sources ([`TwoPassMode::Local`], steps 1-2 in the module docs).
///
/// * `reliable` — the reliable-mask recipe's mask over the field of view, before it is restricted
///   to `main` (hole filling must see the field of view, as the `fill-holes` refinement does);
/// * `main` — the mask the main pass ran on;
/// * `support` — where the main pass's map is defined (its background removal's eroded mask);
/// * `quality` — the ROMEO phase-quality map (0-100) the deep cores are measured on.
///
/// The result is a subset of `main`, ready to be the second pass's mask.
pub fn select_source_holes(
    reliable: &[u8],
    main: &[u8],
    support: &[u8],
    quality: &[f64],
    grid: &Grid,
    params: &LocalTwoPassParams,
) -> Result<SourceHoles, PipelineError> {
    let n = grid.n_total();
    for (name, len) in [("reliable", reliable.len()), ("main", main.len()), ("support", support.len()), ("quality", quality.len())] {
        if len != n {
            return Err(PipelineError::InvalidInput(format!(
                "two-pass hole selection: {name} has {len} voxels, the grid has {n}"
            )));
        }
    }
    let binary: Vec<u8> = reliable.iter().map(|&v| u8::from(v != 0)).collect();
    let filled = fill_holes(&binary, grid, params.fill_holes);
    // Holes of the filled mask inside the main mask, split by the one-pass support.
    let holes: Vec<bool> = main.iter().zip(&filled).map(|(&m, &f)| m != 0 && f == 0).collect();
    let inside: Vec<u8> = holes.iter().zip(support).map(|(&h, &s)| u8::from(h && s != 0)).collect();
    let opened = dilate_mask(
        &erode_mask(&inside, grid, params.opening_iterations),
        grid,
        params.opening_iterations,
    );
    let (labels, sizes) = label_components(&opened, grid.dims, Connectivity::Face);
    let mut cores = vec![0usize; sizes.len()];
    for (&l, &q) in labels.iter().zip(quality) {
        if l != 0 && q < params.core_quality {
            cores[l as usize] += 1;
        }
    }
    let keep: Vec<bool> = cores.iter().enumerate().map(|(l, &c)| l != 0 && c >= params.core_voxels).collect();
    let blobs: Vec<u8> = labels.iter().map(|&l| u8::from(keep[l as usize])).collect();
    let reach = dilate_mask(&blobs, grid, params.rim_dilations);
    let mut hole_voxels = 0;
    let out: Vec<u8> = (0..n)
        .map(|i| {
            if main[i] == 0 {
                return 0;
            }
            let hole = holes[i] && (support[i] == 0 || reach[i] != 0);
            hole_voxels += usize::from(hole && support[i] != 0);
            u8::from(!hole)
        })
        .collect();
    Ok(SourceHoles {
        reliable: out,
        candidate_voxels: inside.iter().filter(|&&v| v != 0).count(),
        sources: keep.iter().filter(|&&k| k).count(),
        hole_voxels,
    })
}

/// The second pass's local field with V-SHARP's eroded rim around the holes filled by LBV
/// ([`TwoPassMode::Local`], step 3 in the module docs).
///
/// `local` and `local_mask` are the second pass's background removal on `reliable` (any method;
/// a method that does not erode leaves no rim and LBV is not run). Where `reliable` has a voxel
/// that `local_mask` lacks, LBV (default parameters, peel 0) on `reliable` and `total_field` gives
/// the value, where LBV's own interior reaches. Everything is restricted to `support`, the one-pass
/// support.
///
/// Returns the local field (zero outside the returned mask) and the mask the inversion should
/// run on.
pub fn fill_eroded_rim(
    total_field: &[f64],
    reliable: &[u8],
    local: &[f64],
    local_mask: &[u8],
    support: &[u8],
    metadata: &ScanMetadata,
    progress: impl FnMut(usize, usize),
) -> Result<(Vec<f64>, Vec<u8>), PipelineError> {
    // LBV's Laplacian assumes a uniform grid, as every background-removal stage does.
    metadata.require_contiguous_slices()?;
    let grid = metadata.grid();
    let n = grid.n_total();
    for (name, len) in [("total field", total_field.len()), ("reliable", reliable.len()), ("local field", local.len()),
                        ("local-field mask", local_mask.len()), ("support", support.len())] {
        if len != n {
            return Err(PipelineError::InvalidInput(format!(
                "two-pass rim fill: {name} has {len} voxels, the grid has {n}"
            )));
        }
    }
    let rim = (0..n).any(|i| reliable[i] != 0 && local_mask[i] == 0 && support[i] != 0);
    let lbv = if rim {
        let params = crate::bgremove::LbvParams {
            b0: metadata.field_strength,
            echo_times: super::inversion::scan_echo_times("LBV", crate::bgremove::LbvParams::default().te, metadata)?,
            ..Default::default()
        };
        Some(crate::bgremove::lbv(total_field, reliable, &grid, &params, progress)?)
    } else {
        None
    };
    let mut field = vec![0.0; n];
    let mut mask = vec![0u8; n];
    for i in 0..n {
        if support[i] == 0 {
            continue;
        }
        if local_mask[i] != 0 {
            field[i] = local[i];
            mask[i] = 1;
        } else if let Some((lf, interior)) = &lbv {
            if reliable[i] != 0 && interior[i] != 0 {
                field[i] = lf[i];
                mask[i] = 1;
            }
        }
    }
    Ok((field, mask))
}

/// The second pass's weight in the merge ([`TwoPassMode::Local`], step 4): 1 within
/// [`full_weight_mm`](LocalTwoPassParams::full_weight_mm) of a hole, falling linearly to 0 at
/// [`zero_weight_mm`](LocalTwoPassParams::zero_weight_mm). A hole is a voxel of `main` and of
/// `support` (the one-pass support) that `reliable` lacks; distances are in mm on `grid`. With no
/// holes every weight is 0 and the merge returns the one-pass map.
pub fn local_weights(
    reliable: &[u8],
    main: &[u8],
    support: &[u8],
    grid: &Grid,
    params: &LocalTwoPassParams,
) -> Result<Vec<f64>, PipelineError> {
    let n = grid.n_total();
    if reliable.len() != n || main.len() != n || support.len() != n {
        return Err(PipelineError::DimensionMismatch { expected: n, got: reliable.len().min(main.len()).min(support.len()) });
    }
    let holes: Vec<u8> = (0..n).map(|i| u8::from(main[i] != 0 && support[i] != 0 && reliable[i] == 0)).collect();
    let span = params.zero_weight_mm - params.full_weight_mm;
    Ok(distance_to_set_mm(&holes, grid)
        .into_iter()
        .map(|d| if d.is_finite() { ((params.zero_weight_mm - d) / span).clamp(0.0, 1.0) } else { 0.0 })
        .collect())
}

/// Merge the passes with weights ([`TwoPassMode::Local`], step 4): `w·reliable + (1 − w)·filled`
/// where `support` (the second pass's support, as for [`combine_two_pass`]) is set, `filled`
/// elsewhere. `weights` are [`local_weights`].
pub fn combine_two_pass_weighted(
    reliable: &[f64],
    filled: &[f64],
    support: &[u8],
    weights: &[f64],
) -> Result<Vec<f64>, PipelineError> {
    let n = reliable.len();
    for len in [filled.len(), support.len(), weights.len()] {
        if len != n {
            return Err(PipelineError::DimensionMismatch { expected: n, got: len });
        }
    }
    Ok((0..n)
        .map(|i| if support[i] != 0 { weights[i] * reliable[i] + (1.0 - weights[i]) * filled[i] } else { filled[i] })
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
    fn idx(x: usize, y: usize, z: usize, n: usize) -> usize {
        x + n * (y + n * z)
    }

    /// One compact source with a deep core (kept), a thin vessel with very low quality (cut by
    /// the opening), a compact hole of moderate quality (no deep core: dropped), a single-voxel
    /// hole (filled) and a hole outside the one-pass support (stays a hole).
    #[test]
    fn only_holes_with_a_deep_core_stay_holes() {
        let n = 32;
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let nv = n * n * n;
        let main = vec![1u8; nv];
        let mut support = vec![1u8; nv];
        let mut reliable = vec![1u8; nv];
        let mut quality = vec![90.0; nv];
        let hole = |x0: usize, x1: usize, y0: usize, y1: usize, z0: usize, z1: usize, q: f64, rel: &mut [u8], qu: &mut [f64]| {
            for z in z0..z1 { for y in y0..y1 { for x in x0..x1 {
                rel[idx(x, y, z, n)] = 0;
                qu[idx(x, y, z, n)] = q;
            }}}
        };
        hole(4, 11, 4, 11, 4, 11, 2.0, &mut reliable, &mut quality); // source, 343 voxels
        hole(16, 28, 20, 22, 20, 22, 1.0, &mut reliable, &mut quality); // vessel, 2 voxels wide
        hole(18, 25, 4, 11, 4, 11, 30.0, &mut reliable, &mut quality); // low quality, no deep core
        hole(25, 26, 25, 26, 25, 26, 1.0, &mut reliable, &mut quality); // single voxel
        hole(4, 11, 18, 25, 18, 25, 2.0, &mut reliable, &mut quality); // like the source ...
        for z in 18..25 { for y in 18..25 { for x in 4..11 { support[idx(x, y, z, n)] = 0; } } } // ... outside the support
        let out = select_source_holes(&reliable, &main, &support, &quality, &grid, &LocalTwoPassParams::default()).unwrap();
        assert_eq!(out.sources, 1);
        // The opening leaves the cube's 3³ centre; two dilations give back the faces but not the
        // corners (6-neighbour distance > 4 from the centre).
        assert_eq!(out.hole_voxels, 311);
        assert_eq!(out.reliable[idx(7, 7, 7, n)], 0, "the source stays a hole");
        assert_eq!(out.reliable[idx(4, 7, 7, n)], 0, "with its own rim");
        assert_eq!(out.reliable[idx(4, 4, 4, n)], 1, "but not its corners");
        assert_eq!(out.reliable[idx(20, 21, 21, n)], 1, "the vessel is cut");
        assert_eq!(out.reliable[idx(21, 7, 7, n)], 1, "no deep core");
        assert_eq!(out.reliable[idx(25, 25, 25, n)], 1, "filled");
        assert_eq!(out.reliable[idx(7, 21, 21, n)], 0, "outside the support");
        assert_eq!(out.reliable[idx(0, 0, 0, n)], 1);
    }

    #[test]
    fn the_hole_selection_stays_in_the_main_mask() {
        let grid = Grid::new(4, 4, 4, 1.0, 1.0, 1.0);
        let main: Vec<u8> = (0..64).map(|i| u8::from(i % 3 != 0)).collect();
        let out = select_source_holes(&[1; 64], &main, &[1; 64], &[90.0; 64], &grid, &LocalTwoPassParams::default()).unwrap();
        assert_eq!(out.reliable, main);
        assert!(select_source_holes(&[1; 63], &main, &[1; 64], &[90.0; 64], &grid, &LocalTwoPassParams::default()).is_err());
    }

    #[test]
    fn weights_fall_from_six_to_twelve_mm() {
        // 1 x 1 x 40 line, 0.5 mm along z: hole at z = 0.
        let grid = Grid::new(1, 1, 40, 1.0, 1.0, 0.5);
        let mut reliable = vec![1u8; 40];
        reliable[0] = 0;
        let w = local_weights(&reliable, &[1; 40], &[1; 40], &grid, &LocalTwoPassParams::default()).unwrap();
        assert_eq!(w[12], 1.0); // 6 mm
        assert!((w[18] - 0.5).abs() < 1e-12); // 9 mm
        assert_eq!(w[24], 0.0); // 12 mm
        assert_eq!(w[39], 0.0);
        // A hole outside the one-pass support does not count.
        let mut support = vec![1u8; 40];
        support[0] = 0;
        let w = local_weights(&reliable, &[1; 40], &support, &grid, &LocalTwoPassParams::default()).unwrap();
        assert!(w.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn the_weighted_merge_feathers_the_second_pass() {
        let out = combine_two_pass_weighted(&[1.0, 1.0, 1.0, 1.0], &[3.0, 3.0, 3.0, 3.0], &[1, 1, 1, 0], &[1.0, 0.5, 0.0, 1.0]).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 3.0]);
        assert!(combine_two_pass_weighted(&[1.0], &[1.0], &[1], &[]).is_err());
    }

    fn rim_meta(n: usize) -> ScanMetadata {
        ScanMetadata { dims: (n, n, n), voxel_size: (1.0, 1.0, 1.0), echo_times: vec![0.02], field_strength: 3.0,
                       b0_direction: (0.0, 0.0, 1.0), slice_geometry: None, affine: None }
    }

    /// Without a rim (the method did not erode at the holes) LBV is not needed: the local field
    /// is passed through, restricted to the one-pass support.
    #[test]
    fn no_rim_passes_the_local_field_through() {
        let n = 8;
        let nv = n * n * n;
        let local: Vec<f64> = (0..nv).map(|i| i as f64).collect();
        let mut support = vec![1u8; nv];
        support[5] = 0;
        let (field, mask) = fill_eroded_rim(&vec![0.0; nv], &vec![1; nv], &local, &vec![1; nv], &support, &rim_meta(n), |_, _| {}).unwrap();
        assert_eq!(mask[5], 0);
        assert_eq!(field[5], 0.0);
        assert_eq!(field[6], 6.0);
        assert_eq!(mask.iter().filter(|&&m| m != 0).count(), nv - 1);
    }

    /// The eroded rim gets LBV's value of a harmonic (background) field: about zero, and it joins
    /// the mask; voxels the main method kept keep its value.
    #[test]
    fn the_rim_is_filled_by_lbv() {
        let n = 16;
        let nv = n * n * n;
        let mut reliable = vec![0u8; nv];
        let mut local_mask = vec![0u8; nv];
        for z in 2..14 { for y in 2..14 { for x in 2..14 {
            reliable[idx(x, y, z, n)] = 1;
            if (5..11).contains(&x) && (5..11).contains(&y) && (5..11).contains(&z) {
                local_mask[idx(x, y, z, n)] = 1;
            }
        }}}
        // A linear (harmonic) total field: LBV removes it entirely.
        let total: Vec<f64> = (0..nv).map(|i| 0.01 * (i % n) as f64).collect();
        let local = vec![7.0; nv];
        let (field, mask) = fill_eroded_rim(&total, &reliable, &local, &local_mask, &vec![1; nv], &rim_meta(n), |_, _| {}).unwrap();
        assert_eq!(field[idx(7, 7, 7, n)], 7.0);
        let rim = idx(3, 7, 7, n);
        assert_eq!(mask[rim], 1);
        assert!(field[rim].abs() < 1e-3, "{}", field[rim]);
        assert_eq!(mask[idx(2, 7, 7, n)], 0, "LBV's interior leaves out the mask's own boundary");
        assert_eq!(mask[idx(0, 0, 0, n)], 0);
    }

    #[test]
    fn modes_have_stable_ids() {
        assert_eq!(TwoPassMode::default(), TwoPassMode::Local);
        for m in TwoPassMode::ALL {
            assert_eq!(TwoPassMode::from_id(m.id()), Some(m));
        }
        assert_eq!(TwoPassMode::Original.id(), "original");
    }

}

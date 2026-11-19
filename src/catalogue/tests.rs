use super::*;
use crate::models::all_models;
use crate::pipeline::{
    B0EstimationMethod, BgRemovalAlgorithm, InhomogeneityCorrection, InversionAlgorithm, MaskOp,
    MaskThresholdMethod, SeparationAlgorithm, UnwrappingAlgorithm,
};
use crate::segment::SynthSegVersion;
use std::collections::HashSet;

/// Every variant of a field-less enum, listed once. The match makes the list exhaustive: a new
/// variant fails to compile here until it is added.
macro_rules! every_variant {
    ($ty:ident: $($v:ident),+ $(,)?) => {{
        let _exhaustive = |x: $ty| match x { $($ty::$v)|+ => () };
        vec![$($ty::$v),+]
    }};
}

/// Every [`MaskOp`], one value per variant. The match fails to compile on a new variant; add it
/// to the list below it too.
fn every_mask_op() -> Vec<MaskOp> {
    let _exhaustive = |op: &MaskOp| match op {
        MaskOp::Threshold { .. } | MaskOp::Bet(_) | MaskOp::Erode { .. } | MaskOp::Dilate { .. }
        | MaskOp::Close { .. } | MaskOp::MinIntensity { .. } | MaskOp::Majority
        | MaskOp::FillHoles { .. } | MaskOp::LargestComponent | MaskOp::Imfill
        | MaskOp::GaussianSmooth { .. } | MaskOp::SignalErode(_) | MaskOp::HdBet(_)
        | MaskOp::Rs2Net(_) | MaskOp::MindGrab(_) => (),
    };
    vec![
        MaskOp::Threshold { method: MaskThresholdMethod::Otsu, value: None },
        MaskOp::Bet(Default::default()),
        MaskOp::Erode { iterations: 1 },
        MaskOp::Dilate { iterations: 1 },
        MaskOp::Close { radius: 1 },
        MaskOp::MinIntensity { value: 0.0 },
        MaskOp::Majority,
        MaskOp::FillHoles { max_size: 0 },
        MaskOp::LargestComponent,
        MaskOp::Imfill,
        MaskOp::GaussianSmooth { sigma_mm: 1.0 },
        MaskOp::SignalErode(Default::default()),
        MaskOp::HdBet(Default::default()),
        MaskOp::Rs2Net(Default::default()),
        MaskOp::MindGrab(Default::default()),
    ]
}

/// Every [`B0EstimationMethod`], one value per variant (as [`every_mask_op`]).
fn every_b0_method() -> Vec<B0EstimationMethod> {
    let _exhaustive = |m: &B0EstimationMethod| match m {
        B0EstimationMethod::WeightedAvg { .. } | B0EstimationMethod::LinearFit(_)
        | B0EstimationMethod::RobustFit(_) | B0EstimationMethod::ComplexFit(_)
        | B0EstimationMethod::QsmartEchofit { .. } => (),
    };
    vec![
        B0EstimationMethod::default(),
        B0EstimationMethod::LinearFit(Default::default()),
        B0EstimationMethod::RobustFit(Default::default()),
        B0EstimationMethod::ComplexFit(Default::default()),
        B0EstimationMethod::QsmartEchofit { fit_threshold: Default::default() },
    ]
}

fn bg_removal_variants() -> Vec<BgRemovalAlgorithm> {
    let all = every_variant!(BgRemovalAlgorithm:
        Vsharp, Pdf, Lbv, Ismv, Sharp, Resharp, Harperella, Iharperella, Bfrnet);
    assert_eq!(all, BgRemovalAlgorithm::VARIANTS, "BgRemovalAlgorithm::VARIANTS is incomplete");
    all
}

fn inversion_variants() -> Vec<InversionAlgorithm> {
    let all = every_variant!(InversionAlgorithm:
        Tkd, Tsvd, Tikhonov, Tv, Rts, Nltv, Medi, Tfi, Ilsqr, Star, Tgv, Qsmart, Ndi, Fansi, FansiTgv,
        L1qsm, Whqsm, Hdqsm, AmpPe, Lsqr, Xqsm, Qsmnet, QsmnetPlus, Qsmnetme, Autoqsm, Qsmgan, Ir2qsm,
        Lpcnn, ModlQsm, Nextqsm, Iqsm, IqsmPlus);
    assert_eq!(all, InversionAlgorithm::VARIANTS, "InversionAlgorithm::VARIANTS is incomplete");
    all
}

fn separation_variants() -> Vec<SeparationAlgorithm> {
    let all = every_variant!(SeparationAlgorithm:
        ChiSepIlsqr, ChiSepMedi, R2starQsm, WaveSep, Decompose, HcChisep, SusepNet, ChiSepNet);
    assert_eq!(all, SeparationAlgorithm::VARIANTS, "SeparationAlgorithm::VARIANTS is incomplete");
    all
}

/// Each dispatched stage with the ids of all its enum's variants.
fn dispatched_stages() -> Vec<(Stage, Vec<&'static str>)> {
    vec![
        (Stage::Masking, every_mask_op().iter().map(MaskOp::id).collect()),
        (Stage::MaskThreshold, every_variant!(MaskThresholdMethod: Otsu, Fixed, Percentile, StiGui)
            .into_iter().map(MaskThresholdMethod::id).collect()),
        // `None` is the absence of a correction, not a method.
        (Stage::InhomogeneityCorrection, every_variant!(InhomogeneityCorrection: None, Makehomogeneous, N4)
            .into_iter().filter(|c| *c != InhomogeneityCorrection::None).map(InhomogeneityCorrection::id).collect()),
        (Stage::Unwrapping, every_variant!(UnwrappingAlgorithm: Romeo, Laplacian, Prelude, MediRegionGrowing)
            .into_iter().map(UnwrappingAlgorithm::id).collect()),
        (Stage::EchoCombination, every_b0_method().iter().map(B0EstimationMethod::id).collect()),
        (Stage::BackgroundRemoval, bg_removal_variants().into_iter().map(BgRemovalAlgorithm::id).collect()),
        (Stage::Inversion, inversion_variants().into_iter().map(InversionAlgorithm::id).collect()),
        (Stage::Separation, separation_variants().into_iter().map(SeparationAlgorithm::id).collect()),
        (Stage::Segmentation, every_variant!(SynthSegVersion: V1, V2)
            .into_iter().map(SynthSegVersion::model_id).collect()),
    ]
}

/// Entries of a dispatched stage that are not one of its enum's variants, with the entry point
/// that runs them.
const OUTSIDE_THE_ENUM: &[(Stage, &str, &str)] = &[
    (Stage::Masking, "phase-reliability-trim", "BgRemovalConfig::ukb_mask_trim"),
    (Stage::BackgroundRemoval, "iqfm", "pipeline::run_iqfm"),
];

#[test]
fn every_dispatcher_variant_has_exactly_one_entry_in_its_stage() {
    for (stage, ids) in dispatched_stages() {
        for id in &ids {
            let n = METHODS.iter().filter(|m| m.stage == stage && m.id == *id).count();
            assert_eq!(n, 1, "{stage:?} variant {id:?} has {n} catalogue entries");
            assert!(METHODS.iter().all(|m| m.id != *id || m.stage == stage),
                "{id:?} is filed under another stage as well as {stage:?}");
        }
    }
}

#[test]
fn a_dispatched_stage_lists_only_its_variants_and_the_named_exceptions() {
    for (stage, ids) in dispatched_stages() {
        for m in in_stage(stage) {
            let exception = OUTSIDE_THE_ENUM.iter().any(|(s, id, _)| *s == stage && *id == m.id);
            assert!(ids.contains(&m.id) || exception,
                "{stage:?} entry {:?} is not a variant of the stage's enum; add it to OUTSIDE_THE_ENUM \
                 with its entry point, or give it another stage", m.id);
        }
    }
    for (stage, id, _) in OUTSIDE_THE_ENUM {
        assert!(find(*stage, id).is_some(), "OUTSIDE_THE_ENUM names {id:?}, which has no entry");
    }
}

#[test]
fn ids_are_unique_within_a_stage_and_kebab_case() {
    let mut seen = HashSet::new();
    for m in METHODS {
        assert!(seen.insert((m.stage, m.id)), "duplicate {:?} id {:?}", m.stage, m.id);
        assert!(!m.id.is_empty() && !m.id.starts_with('-') && !m.id.ends_with('-'), "{:?}", m.id);
        assert!(m.id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '.'),
            "id {:?} is not kebab-case", m.id);
        assert!(!m.name.trim().is_empty() && m.name.trim() == m.name, "{}: bad name {:?}", m.id, m.name);
    }
}

#[test]
fn every_model_belongs_to_a_method_and_every_method_model_exists() {
    let used: HashSet<&str> = METHODS.iter().flat_map(|m| m.models.iter().copied()).collect();
    for spec in all_models() {
        assert!(used.contains(spec.id), "registry model {:?} belongs to no catalogue method", spec.id);
    }
    for m in METHODS {
        for id in m.models {
            assert!(crate::models::find_model(id).is_some(), "{}: no registry model {id:?}", m.id);
        }
    }
}

#[test]
fn a_variants_network_is_among_its_entrys_models() {
    let check = |stage, id: &str, model: Option<&str>| {
        if let Some(model) = model {
            let m = find(stage, id).unwrap();
            assert!(m.models.contains(&model), "{id}: runs {model:?}, entry lists {:?}", m.models);
        }
    };
    for op in every_mask_op() { check(Stage::Masking, op.id(), op.dl_model_id()); }
    for a in bg_removal_variants() { check(Stage::BackgroundRemoval, a.id(), a.dl_model_id()); }
    for a in inversion_variants() { check(Stage::Inversion, a.id(), a.dl_model_id()); }
    for a in separation_variants() { check(Stage::Separation, a.id(), a.dl_model_id()); }
}

fn all_citations() -> impl Iterator<Item = (&'static str, &'static Citation)> {
    METHODS.iter().flat_map(|m| m.citations.iter().map(move |c| (m.id, c)))
}

#[test]
fn dois_are_bare_and_well_formed() {
    for (method, c) in all_citations() {
        let Some(doi) = c.doi else { continue };
        assert!(doi.starts_with("10.") && doi.contains('/'), "{method}: {:?} is not a DOI", doi);
        assert!(!doi.contains("doi.org") && !doi.starts_with("doi:") && !doi.contains("://"),
            "{method}: DOI {doi:?} carries a prefix");
        assert!(!doi.chars().any(char::is_whitespace), "{method}: DOI {doi:?} has whitespace");
        assert!(!doi.ends_with(['.', ',', ';', ')', ']']), "{method}: DOI {doi:?} ends in punctuation");
    }
}

#[test]
fn citations_are_complete_and_one_key_is_one_work() {
    let mut by_key: std::collections::HashMap<&str, &Citation> = Default::default();
    for (method, c) in all_citations() {
        assert!(!c.key.is_empty() && !c.authors.is_empty() && !c.title.is_empty(), "{method}: {c:?}");
        assert!((1900..=2100).contains(&c.year), "{method}: year {}", c.year);
        assert!(!c.title.ends_with('.'), "{method}: title {:?} ends in a full stop", c.title);
        assert!(c.doi.is_none() || c.url.is_none(), "{method}: {} has both a DOI and a URL", c.key);
        if let Some(url) = c.url {
            assert!(url.starts_with("https://"), "{method}: {url:?}");
        }
        let fields = [Some(c.key), Some(c.authors), Some(c.title), c.venue, c.volume, c.issue, c.pages];
        for s in fields.into_iter().flatten() {
            assert_eq!(s.trim(), s, "{method}: {} has untrimmed {s:?}", c.key);
        }
        if let Some(other) = by_key.insert(c.key, c) {
            assert_eq!(other, c, "two different works share the key {:?}", c.key);
        }
    }
    let mut by_doi: std::collections::HashMap<&str, &str> = Default::default();
    for c in by_key.values() {
        if let Some(doi) = c.doi {
            if let Some(key) = by_doi.insert(doi, c.key) {
                panic!("{key:?} and {:?} are the same work ({doi})", c.key);
            }
        }
    }
}

#[test]
fn references_are_named_and_linked() {
    for m in METHODS {
        let Some(r) = m.reference else { continue };
        assert!(!r.name.is_empty(), "{}: unnamed reference", m.id);
        if let Some(url) = r.url {
            assert!(url.starts_with("https://") || url.starts_with("http://"), "{}: {url:?}", m.id);
        }
    }
}

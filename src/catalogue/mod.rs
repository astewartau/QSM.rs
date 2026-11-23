//! The method catalogue: what each method QSM.rs implements is called, and whose work it is.
//!
//! One [`Method`] per method a host can choose: its stable id (the one the pipeline's enums
//! return from `id()`), its stage, its display name, a one-sentence summary for users, what it
//! needs from the scan ([`Need`]), the stages it does the work of ([`Method::skips`]), the
//! authors' paper(s) ([`Citation`]), the reference code QSM.rs's implementation follows
//! ([`Reference`]), and the ids of the deep-learning models it runs ([`crate::models`]). Front
//! ends take names, summaries and citations from here instead of restating them, so a methods
//! section, an algorithm table and a settings screen agree with each other and with the code.
//!
//! The catalogue holds identity, provenance and what a host needs to list a method. How a method
//! works, its defaults and their sources, its deviations from the reference and what was
//! measured are in its module docs; what it requires or recommends of the other settings is in
//! [`crate::pipeline::rules`].
//!
//! ```
//! use qsm_core::catalogue::{self, Stage};
//! use qsm_core::pipeline::InversionAlgorithm;
//!
//! let rts = catalogue::find(Stage::Inversion, InversionAlgorithm::Rts.id()).unwrap();
//! assert_eq!(rts.name, "RTS");
//! assert_eq!(rts.citations[0].doi, Some("10.1016/j.neuroimage.2017.11.018"));
//! assert_eq!(rts.reference.unwrap().name, "QSM.jl");
//! ```
//!
//! # Citations
//!
//! A [`Citation`] is structured, not formatted: hosts format it in their own style. Every field
//! is the published record's, checked against Crossref or DataCite by DOI. `authors` is written as
//! a methods section cites it (all authors of a short list, otherwise the first few and "et al.").
//! A paper cited by several methods is one constant, so its `key` identifies it.
//!
//! # Adding a method
//!
//! A method a host can choose gets one entry in [`METHODS`], in its stage's group:
//!
//! 1. `id`: the variant's `id()` where the stage has a dispatcher enum (see [`Stage`]); otherwise
//!    a new kebab-case id.
//! 2. `name`: the name the authors use.
//! 3. `summary`: one plain sentence for users: what the method does, in words a user without
//!    the paper follows. No history, no comparison with QSM.rs's other methods, no settings.
//! 4. `needs`: what it needs beyond what every method has ([`Need`]).
//! 5. `skips`: the stages whose work it does itself ([`Method::skips`]).
//! 6. `citations`: the authors' paper(s), as constants in `citations.rs`, with the DOI checked
//!    against its record. A method with no paper (QSM.rs's own, or a generic operation) has none.
//! 7. `reference`: the code the implementation follows, as its module docs name it (toolbox,
//!    release or commit, URL); `None` if it follows the paper alone or is QSM.rs's own.
//! 8. `models`: the [`ModelSpec`](crate::models::ModelSpec) ids it runs, if any.
//!
//! The tests fail if a dispatcher variant has no entry, if an entry names a model the registry
//! lacks, if a registry model belongs to no method, if a DOI is malformed, if a summary is empty
//! or not one sentence, or if `needs` or `skips` disagree with what the pipeline does (an
//! inversion's [`InversionInput`](crate::pipeline::InversionInput), a method's
//! [`OrientationSupport`](crate::pipeline::OrientationSupport)). Adding a variant to a
//! dispatcher enum also fails to compile the test's list of variants until the variant joins it.

pub(crate) mod citations;
mod methods;

pub use methods::METHODS;

/// A stage of reconstruction, as a host groups its choices.
///
/// Where the pipeline dispatches the stage through an enum, the catalogue has exactly one entry
/// per variant, under the variant's `id()`, and the tests hold it to that.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    /// Mask generators and refinements: [`MaskOp`](crate::pipeline::MaskOp), plus UK Biobank's
    /// phase-reliability trimming ([`crate::utils::phase_reliability`]).
    Masking,
    /// How a threshold generator picks its level:
    /// [`MaskThresholdMethod`](crate::pipeline::MaskThresholdMethod).
    MaskThreshold,
    /// Receive-field bias correction of the magnitude:
    /// [`InhomogeneityCorrection`](crate::pipeline::InhomogeneityCorrection), except `None`.
    InhomogeneityCorrection,
    /// Phase corrections before unwrapping: MCPC-3D-S phase-offset removal and bipolar
    /// correction ([`FieldMappingConfig`](crate::pipeline::FieldMappingConfig) switches).
    PhaseCorrection,
    /// Phase unwrapping: [`UnwrappingAlgorithm`](crate::pipeline::UnwrappingAlgorithm).
    Unwrapping,
    /// Combining echoes into one B0 map:
    /// [`B0EstimationMethod`](crate::pipeline::B0EstimationMethod).
    EchoCombination,
    /// Background field removal: [`BgRemovalAlgorithm`](crate::pipeline::BgRemovalAlgorithm),
    /// plus iQFM ([`run_iqfm`](crate::pipeline::run_iqfm)), which makes the local field from
    /// wrapped phase.
    BackgroundRemoval,
    /// Dipole inversion, including single-step methods:
    /// [`InversionAlgorithm`](crate::pipeline::InversionAlgorithm).
    Inversion,
    /// A step or option added to an inversion and cited on its own: MEDI's CSF zero reference
    /// and mSMV, HEIDI, and two-pass artefact reduction.
    InversionAddition,
    /// R2\*, T2\*, R2 and R2′ maps.
    Relaxometry,
    /// Susceptibility source separation:
    /// [`SeparationAlgorithm`](crate::pipeline::SeparationAlgorithm).
    Separation,
    /// Susceptibility-weighted imaging ([`crate::swi`]).
    Swi,
    /// Anatomical segmentation: [`SynthSegVersion`](crate::segment::SynthSegVersion), under its
    /// `model_id()`.
    Segmentation,
    /// Reconstruction from several head orientations
    /// ([`inversion::cosmos`](mod@crate::inversion::cosmos),
    /// [`inversion::sti`](mod@crate::inversion::sti)).
    MultiOrientation,
}

/// One method QSM.rs implements.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Method {
    /// Stable kebab-case id, unique within the stage; for a dispatched stage, the variant's
    /// `id()`.
    pub id: &'static str,
    pub stage: Stage,
    /// Display name, as the authors write it (`"RTS"`, `"χ-separation (iLSQR)"`).
    pub name: &'static str,
    /// One plain sentence for users: what the method does. A host shows it next to the name
    /// (a settings list, a methods index); the method's module docs hold the details.
    pub summary: &'static str,
    /// What it needs beyond what every method has; see [`Need`].
    pub needs: &'static [Need],
    /// The stages whose work it does itself, so a pipeline running it leaves them out: a
    /// total-field inversion (TFI, QSMART) skips [`Stage::BackgroundRemoval`], a
    /// wrapped-phase one (iQSM) also [`Stage::Unwrapping`] and [`Stage::EchoCombination`]. Only
    /// what it skips whatever the scan: TGV skips unwrapping on a single echo only, so its entry
    /// lists background removal alone. Empty for most methods.
    pub skips: &'static [Stage],
    /// The authors' paper(s), the method's own first. Empty for a method with none.
    pub citations: &'static [Citation],
    /// The code QSM.rs's implementation follows, where it follows one.
    pub reference: Option<Reference>,
    /// [`ModelSpec`](crate::models::ModelSpec) ids of the networks it runs.
    pub models: &'static [&'static str],
}

/// Something a method needs beyond what every method has.
///
/// Every reconstruction has the phase of each echo, the echo times, the field strength (to
/// express the field in ppm), the voxel grid with its B0 direction, and a brain mask, so none of
/// those is listed. A method that only uses something when it is there (ROMEO's magnitude
/// weights, QSMART's vessel magnitude) does not list it either.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Need {
    /// The magnitude image (of every echo, for a multi-echo method): the method fails without it.
    Magnitude,
    /// At least this many echoes.
    Echoes(u8),
    /// Evenly spaced echo times (MEDI's complex fit models the phase by echo index:
    /// [`echo_spacing_problem`](crate::fieldmap::echo_spacing_problem)).
    EvenEchoSpacing,
    /// B0 along the slice axis: an oblique acquisition is resampled to axial first, because the
    /// method takes no B0 direction (a network trained on axial data). The same as
    /// [`OrientationSupport::AxialOnly`](crate::pipeline::OrientationSupport::AxialOnly).
    Axial,
    /// An R2′ map in Hz: R2* minus R2 from a multi-echo spin-echo scan, or one the user
    /// supplies, or one R2PRIMEnet predicts from R2*.
    R2Prime,
    /// A multi-echo spin-echo acquisition.
    SpinEcho,
    /// Scans of the head in at least this many orientations.
    Orientations(u8),
}

/// A published work, as structured fields; hosts format it.
///
/// Serializable without the `introspection` feature, because a
/// [`ReferencePipeline`](crate::pipeline::reference_pipelines::ReferencePipeline) carries its
/// citations and is always serializable.
#[derive(serde::Serialize)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Citation {
    /// Citation key (`"kames2018"`), the same for every method citing this work.
    pub key: &'static str,
    /// Authors as cited: `"Kames, C., Wiggermann, V., Rauscher, A."`, `"Wu, B., et al."`.
    pub authors: &'static str,
    /// Year of the issue it appeared in (or of the preprint or thesis).
    pub year: u16,
    /// Published title, without a closing full stop.
    pub title: &'static str,
    /// Journal, proceedings, `"arXiv"`, or a thesis's degree and university; `None` if
    /// unpublished.
    pub venue: Option<&'static str>,
    pub volume: Option<&'static str>,
    pub issue: Option<&'static str>,
    /// Page range, or the article number for journals that number articles (`"e4271"`).
    pub pages: Option<&'static str>,
    /// The DOI itself (`"10.1002/mrm.28563"`), without `https://doi.org/`.
    pub doi: Option<&'static str>,
    /// A link, only for a work with no DOI.
    pub url: Option<&'static str>,
}

/// The code an implementation follows.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reference {
    /// Toolbox or repository (`"STI Suite"`, `"QSM.jl"`).
    pub name: &'static str,
    /// The release or commit compared against.
    pub version: Option<&'static str>,
    pub url: Option<&'static str>,
}

/// The entry for `id` in `stage`.
pub fn find(stage: Stage, id: &str) -> Option<&'static Method> {
    METHODS.iter().find(|m| m.stage == stage && m.id == id)
}

/// The entries of `stage`, in catalogue order.
pub fn in_stage(stage: Stage) -> impl Iterator<Item = &'static Method> {
    METHODS.iter().filter(move |m| m.stage == stage)
}

#[cfg(test)]
mod tests;

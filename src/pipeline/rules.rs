//! Pipeline rules: what each method requires, recommends and discourages, with sources.
//!
//! Many methods only work as intended with certain other methods, settings or values (the
//! pipeline their authors ran them in), and some combinations are known to fail. This module
//! publishes that knowledge as data, so a host can check a configuration and offer the change
//! instead of letting a user silently run something a method was never meant for.
//!
//! A [`Rule`] reads:
//!
//! > **when** every [`Condition`] in [`Rule::when`] holds, **then** [`Rule::setting`] should pass
//! > [`Rule::expect`] (one of a set of values, none of a set, or a range), with
//! > [`Rule::default`] as the value to apply, at a [`Strength`], for a [`Rule::reason`], citing a
//! > [`Rule::source`].
//!
//! [`evaluate`] checks a configuration ([`RuleInput`]) against every rule in [`RULES`], given what
//! is known about the scan ([`ScanFacts`]), and returns a [`Finding`] for each rule whose
//! conditions hold and whose setting does not pass. [`RULES`] is the catalogue; both it and the
//! findings serialise to JSON (`serde::Serialize`) for hosts that are not written in Rust.
//!
//! # Decisions
//!
//! - **No source, no rule.** A rule says "the method's reference pipeline does this" (the authors'
//!   code, file and line, or their paper) or "QSM.rs measured this" (and the measurement is
//!   recorded in that method's QSM.rs docs). It never encodes a preference. Where the reference is
//!   ambiguous, there is no rule; the open question is recorded in the method's docs instead.
//! - **Scoped to the triggering choice.** The first condition in [`Rule::when`] is the choice
//!   that triggers the rule (usually a method being selected), and the rule is about what that
//!   choice needs. "MEDI recommends PDF" fires only when MEDI is chosen, never as a general
//!   statement about background removal. This keeps two rules from disagreeing about one setting
//!   except where two chosen methods genuinely want different things, and then the host sees both
//!   findings with their triggers.
//! - **Strength**, and the basis it rests on ([`Rule::basis`], a [`Basis`]):
//!   - [`Strength::Required`]: the method cannot run correctly otherwise (e.g. it does not fit
//!     the heap). Usually measured.
//!   - [`Strength::Recommended`]: the authors explicitly recommend it (their paper, README or a
//!     comment in their code says so; [`Basis::Stated`]), or QSM.rs measured a benefit and
//!     recorded it in the method's docs ([`Basis::Measured`]).
//!   - [`Strength::Discouraged`]: a combination known to fail, stated by the authors or measured
//!     (or a setting QSM.rs ignores in that combination, [`Basis::Implementation`]).
//!   - [`Strength::Reference`]: what the reference pipeline used, where the authors never said
//!     it is needed or better: their code or paper simply does it
//!     ([`Basis::ReferenceChoice`]), e.g. QSMART's `bet2 -f 0.2` in `brainmask.m`, or a network
//!     trained on V-SHARP fields.
//!
//!   Precedence is `Required > Recommended > Discouraged > Reference` (`Strength: Ord`). A
//!   statement or measurement outranks a bare reference choice; a recommendation outranks a
//!   discouragement because a recommendation names the value to use, while a discouragement
//!   only names one to avoid. This module reports every finding and leaves resolving them to
//!   the host.
//! - **Intended front-end handling** (QSMxT/QSMxT#419, astewartau/qsmbly#194):
//!   - `Required`: keeps the host's "forces" behaviour; the run does not proceed as configured.
//!   - `Recommended` and `Discouraged`: a modal when a change creates the finding (apply the
//!     default, choose another allowed value, or ignore), a badge on the setting afterwards.
//!   - `Reference`: **never a modal**, only an info marker on the setting. Still recorded in
//!     the run record, and listed in the methods' Note section in a softer form ("differs from
//!     the reference pipeline"), separately from departures from recommendations.
//! - **Rules that restate a default.** Since #293 most methods' own defaults are their reference
//!   values, so many `Reference` rules (STAR's padding and TE, RESHARP's radius and iterations,
//!   iLSQR's iterations and padding, QSMART's iLSQR settings, the reference echo times) hold on
//!   the default config and fire only when a user departs from it. They are kept: they say what
//!   the departure is from and why, which the methods Note needs. A host that also shows
//!   "differs from default" should show one or the other for these, not both.
//! - **Data, not code.** Conditions and expectations are values ([`Condition`], [`Test`],
//!   [`Value`]) over named settings and scan facts, not closures, so the JSON form says
//!   everything the Rust evaluation does. The only code is [`current`] (a setting's value in a
//!   config) and [`ScanFacts::fact`] (a fact's value for a scan).
//! - **Settings are QSM.rs's own.** A setting is a dotted path into QSM.rs's config types:
//!   `masking.*`, `field_mapping.*`, `bg_removal.*`, `inversion.*` ([`QsmPipelineConfig`]),
//!   `separation.*` ([`SeparationConfig`]) and `segmentation.*` (SynthSeg's
//!   [`SynthSegParams`]). Enum choices are named by their `id()` (e.g.
//!   [`InversionAlgorithm::id`](super::config::InversionAlgorithm::id)). A host maps these to its own config paths. [`SETTINGS`] lists
//!   every path a rule may use.
//! - **Settings not in play have no value.** A setting of a stage that does not run (background
//!   removal ahead of a total-field inversion; field mapping ahead of a wrapped-phase one), or of
//!   a method that is not selected anywhere (HD-BET's parameters with no HD-BET step in the mask
//!   recipe), has no current value, and a rule on it does not fire. A setting with several values
//!   (one per mask section) triggers a condition if any value does, and passes a rule's
//!   expectation only if every value does.
//! - **Scan facts may be unknown.** A host checks a config before it has seen data. A condition
//!   on a fact that is not known (`None` in [`ScanFacts`]) does not hold, so scan-dependent rules
//!   stay silent until the scan is known; evaluate again at plan time with the scan's facts.
//!   [`Rule::needs_scan`] says which rules those are.
//! - **Values with scan-relative meaning** are stated in the unit the setting uses ([`Unit`]):
//!   millimetres, voxels, bytes. A rule never converts mm to voxels itself; where that matters it
//!   conditions on the voxel size.
//! - **Inclusive ranges.** [`Test::Range`] bounds are inclusive. Byte and voxel counts are
//!   integers, so "more than the heap" is `min: heap + 1`.

use serde::Serialize;

use super::config::{QsmPipelineConfig, SeparationConfig};
use crate::bet::{hd_bet_memory_bytes, HdBetInference, HdBetParams};
use crate::models::Heap;
use crate::segment::synthseg::{synthseg_network_grid, SynthSegInference, SynthSegParams};

// =========================================================================
// Types
// =========================================================================

/// How strongly a rule applies. Ordered `Reference < Discouraged < Recommended < Required`, so
/// `max` is the finding that takes precedence on a setting (see the module docs, *Strength*).
#[derive(Serialize, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Strength {
    /// What the method's reference pipeline used, where the authors never said it is needed or
    /// better (their code or paper simply does it). Informational.
    Reference,
    /// A combination known to fail (measured, or stated by the authors).
    Discouraged,
    /// The authors explicitly recommend it (paper, README or code comment), or QSM.rs measured a
    /// benefit and recorded it in the method's docs.
    Recommended,
    /// The method cannot run correctly otherwise.
    Required,
}

/// What a rule rests on.
#[derive(Serialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Basis {
    /// The authors say so: their paper, README or a comment in their code.
    Stated,
    /// QSM.rs measured it (memory, accuracy), and the measurement is in the method's docs.
    Measured,
    /// The authors' code or paper does it, without saying it is needed or better.
    ReferenceChoice,
    /// What QSM.rs's own implementation does (recorded in its docs), e.g. a setting it ignores
    /// in some combination.
    Implementation,
}

/// A setting's (or a scan fact's) value. Serialised untagged: `true`, `3.0`, `"pdf"`,
/// `[8.0, 8.0, 20.0]`.
#[derive(Serialize, Clone, Copy, Debug, PartialEq)]
#[serde(untagged)]
pub enum Value {
    Bool(bool),
    Number(f64),
    /// An enum choice, by its `id()` (e.g. `"pdf"`, `"medi"`, `"sliding-window"`).
    Choice(&'static str),
    /// A fixed-length numeric setting (per-axis padding, kernel radii).
    Vector(&'static [f64]),
}

/// The current value of a setting, owned (a config's vectors are not `'static`).
#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(untagged)]
pub enum Current {
    Bool(bool),
    Number(f64),
    Choice(&'static str),
    Vector(Vec<f64>),
}

impl Current {
    fn number(&self) -> Option<f64> {
        match self {
            Current::Number(x) => Some(*x),
            _ => None,
        }
    }

    fn equals(&self, v: &Value) -> bool {
        match (self, v) {
            (Current::Bool(a), Value::Bool(b)) => a == b,
            (Current::Number(a), Value::Number(b)) => a == b,
            (Current::Choice(a), Value::Choice(b)) => a == b,
            (Current::Vector(a), Value::Vector(b)) => a.as_slice() == *b,
            _ => false,
        }
    }
}

/// A test on a value: what a condition requires of a setting or fact, or what a rule expects of
/// its setting.
#[derive(Serialize, Clone, Copy, Debug, PartialEq)]
#[serde(tag = "test", rename_all = "snake_case")]
pub enum Test {
    /// Equal to one of these.
    OneOf { values: &'static [Value] },
    /// Equal to none of these.
    NoneOf { values: &'static [Value] },
    /// A number within `[min, max]` (inclusive; either bound may be open).
    Range { min: Option<f64>, max: Option<f64> },
}

impl Test {
    fn passes(&self, v: &Current) -> bool {
        match self {
            Test::OneOf { values } => values.iter().any(|x| v.equals(x)),
            Test::NoneOf { values } => !values.iter().any(|x| v.equals(x)),
            Test::Range { min, max } => v.number().is_some_and(|x| {
                min.is_none_or(|m| x >= m) && max.is_none_or(|m| x <= m)
            }),
        }
    }
}

/// A fact about the scan or the host, as opposed to a setting.
#[derive(Serialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Fact {
    /// Main field strength, T.
    FieldStrength,
    /// Number of echoes.
    EchoCount,
    /// Smallest voxel edge, mm.
    MinVoxelSize,
    /// Largest voxel edge, mm.
    MaxVoxelSize,
    /// Where the pipeline runs: `"native"` or `"wasm"` ([`Target::id`]).
    Target,
    /// Peak bytes of the configured HD-BET step on this scan in a `wasm32` heap
    /// ([`hd_bet_memory_bytes`] with [`Heap::Wasm32`]; the largest over HD-BET steps).
    HdBetWasmBytes,
    /// Voxels of the grid SynthSeg's network sees for this scan ([`synthseg_network_grid`]).
    SynthSegNetworkVoxels,
}

impl Fact {
    /// Whether the fact depends on the scan (and so is unknown before the scan is).
    pub fn needs_scan(self) -> bool {
        !matches!(self, Fact::Target)
    }
}

/// One condition of a rule.
#[derive(Serialize, Clone, Copy, Debug, PartialEq)]
#[serde(tag = "on", rename_all = "snake_case")]
pub enum Condition {
    /// A setting in the config passes `is`. With several current values (one per mask section)
    /// it holds if any of them passes: a method chosen anywhere triggers its rules. A setting
    /// not in play never holds.
    Setting { setting: &'static str, is: Test },
    /// A scan or host fact passes `is` (an unknown fact never holds).
    Fact { fact: Fact, is: Test },
}

/// The unit of a rule's setting, for numbers and vectors.
#[derive(Serialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Unit {
    /// A choice, a flag or a dimensionless number.
    None,
    Mm,
    Voxels,
    Seconds,
    Bytes,
}

/// A rule: when [`Self::when`] holds, [`Self::setting`] should pass [`Self::expect`].
#[derive(Serialize, Clone, Copy, Debug, PartialEq)]
pub struct Rule {
    /// Stable identifier, `<method>.<what>` in kebab case.
    pub id: &'static str,
    /// Conditions, all of which must hold. The first is the triggering choice the rule is scoped
    /// to (see the module docs).
    pub when: &'static [Condition],
    /// The setting the rule is about (a path in [`SETTINGS`]).
    pub setting: &'static str,
    /// What the setting should pass.
    pub expect: Test,
    /// The value to apply to resolve the finding (one that passes `expect`).
    pub default: Value,
    /// Unit of `default` and of any range in `expect`.
    pub unit: Unit,
    pub strength: Strength,
    /// What the rule rests on: a statement by the authors, a QSM.rs measurement, the reference's
    /// choice, or QSM.rs's implementation.
    pub basis: Basis,
    /// One line: why.
    pub reason: &'static str,
    /// The authors' code (file, line) or paper (section), or the QSM.rs docs recording the
    /// measurement.
    pub source: &'static str,
}

impl Rule {
    /// The triggering choice (the first condition).
    pub fn trigger(&self) -> &'static Condition {
        &self.when[0]
    }

    /// Whether the rule can only fire once the scan is known.
    pub fn needs_scan(&self) -> bool {
        self.when.iter().any(|c| matches!(c, Condition::Fact { fact, .. } if fact.needs_scan()))
    }
}

/// A rule that fires on a configuration: its conditions hold and its setting does not pass.
#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct Finding {
    #[serde(flatten)]
    pub rule: &'static Rule,
    /// The setting's current value(s), one per place it is set (e.g. one per mask section).
    pub current: Vec<Current>,
}

/// Where the pipeline runs. A WASM host has a 32-bit heap ([`WASM32_HEAP_BYTES`]).
#[derive(Serialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum Target {
    #[default]
    Native,
    Wasm,
}

impl Target {
    /// The target this build runs on.
    pub fn current() -> Self {
        if cfg!(target_arch = "wasm32") { Target::Wasm } else { Target::Native }
    }

    pub fn id(self) -> &'static str {
        match self {
            Target::Native => "native",
            Target::Wasm => "wasm",
        }
    }
}

/// The hard ceiling of a 32-bit WASM linear memory: 4 GiB. A step whose estimated peak exceeds
/// it cannot run in a browser at all.
pub const WASM32_HEAP_BYTES: u64 = 1 << 32;

/// What is known about the scan and the host. Every scan fact is optional: a host checks a config
/// before it has seen data, and a rule conditioned on an unknown fact does not fire.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ScanFacts {
    pub field_strength: Option<f64>,
    pub echo_count: Option<usize>,
    /// Voxel size in mm.
    pub voxel_size: Option<(f64, f64, f64)>,
    /// Matrix size in voxels.
    pub dims: Option<(usize, usize, usize)>,
    pub target: Target,
}

impl ScanFacts {
    /// Every fact of a scan, on `target`.
    pub fn from_metadata(m: &super::config::ScanMetadata, target: Target) -> Self {
        Self {
            field_strength: Some(m.field_strength),
            echo_count: Some(m.echo_times.len()),
            voxel_size: Some(m.voxel_size),
            dims: Some(m.dims),
            target,
        }
    }

    fn grid(&self) -> Option<crate::Grid> {
        Some(crate::Grid { dims: self.dims?, voxel_size: self.voxel_size? })
    }

    /// The fact's value for this scan and config, if known.
    pub fn fact(&self, fact: Fact, input: &RuleInput) -> Option<Current> {
        let vs = |f: fn(f64, f64) -> f64| self.voxel_size.map(|(x, y, z)| f(f(x, y), z));
        match fact {
            Fact::FieldStrength => self.field_strength.map(Current::Number),
            Fact::EchoCount => self.echo_count.map(|n| Current::Number(n as f64)),
            Fact::MinVoxelSize => vs(f64::min).map(Current::Number),
            Fact::MaxVoxelSize => vs(f64::max).map(Current::Number),
            Fact::Target => Some(Current::Choice(self.target.id())),
            Fact::HdBetWasmBytes => {
                let grid = self.grid()?;
                hd_bet_steps(input.pipeline)
                    .map(|p| hd_bet_memory_bytes(&grid, p, Heap::Wasm32))
                    .max()
                    .map(|b| Current::Number(b as f64))
            }
            Fact::SynthSegNetworkVoxels => {
                let grid = self.grid()?;
                let d = synthseg_network_grid(&grid, input.segmentation?);
                Some(Current::Number(d.iter().map(|&v| v as f64).product()))
            }
        }
    }
}

/// The configuration a host checks: the QSM pipeline, and the optional stages it runs.
#[derive(Clone, Copy, Debug)]
pub struct RuleInput<'a> {
    pub pipeline: &'a QsmPipelineConfig,
    /// The χ-separation stage's config, when it runs.
    pub separation: Option<&'a SeparationConfig>,
    /// SynthSeg's parameters, when segmentation runs.
    pub segmentation: Option<&'a SynthSegParams>,
}

impl<'a> RuleInput<'a> {
    /// Just the QSM pipeline: no separation, no segmentation.
    pub fn pipeline(pipeline: &'a QsmPipelineConfig) -> Self {
        Self { pipeline, separation: None, segmentation: None }
    }
}

// =========================================================================
// Settings
// =========================================================================

/// Every setting path a rule may name, with what it is. [`current`] reads each of them.
pub const SETTINGS: &[(&str, &str)] = &[
    ("masking.generator", "Each mask section's generator (MaskOp::id)"),
    ("masking.hd_bet.inference", "Each HD-BET step's inference: full-volume, sliding-window (HdBetInference::SLIDING_WINDOW), low-memory (HdBetInference::LOW_MEMORY) or custom-window"),
    ("field_mapping.unwrapping_algorithm", "Phase unwrapping (UnwrappingAlgorithm::id)"),
    ("field_mapping.b0_estimation", "Echo combination (B0EstimationMethod::id)"),
    ("field_mapping.b0_estimation.weighting", "Weighted average's per-echo weight (B0WeightType, by `weight_id`); only with weighted-avg"),
    ("field_mapping.phase_offset_removal", "MCPC-3D-S phase-offset removal before unwrapping (bool)"),
    ("field_mapping.phase_offset_sigma_mm", "MCPC-3D-S smoothing sigma per axis, mm; only with phase-offset removal"),
    ("field_mapping.bipolar_correction", "Bipolar readout correction (bool)"),
    ("field_mapping.laplacian_solver", "Laplacian unwrapping's solver: dct, fft-12 (LaplacianSolver::FFT_DEFAULT_PAD), fft (other padding) or mirror; only with Laplacian unwrapping"),
    ("bg_removal.algorithm", "Background field removal (BgRemovalAlgorithm::id)"),
    ("bg_removal.vsharp.deconvolve", "V-SHARP's TSVD deconvolution (bool); only with V-SHARP"),
    ("bg_removal.vsharp.max_radius", "V-SHARP's largest kernel radius, mm; only with V-SHARP"),
    ("bg_removal.vsharp.min_radius", "V-SHARP's smallest kernel radius, mm; only with V-SHARP"),
    ("bg_removal.vsharp.radius_step", "V-SHARP's step between kernel radii, mm; only with V-SHARP"),
    ("bg_removal.vsharp.grid", "V-SHARP's working grid: native, isotropic-finest, isotropic-1mm or isotropic (other spacing); only with V-SHARP"),
    ("bg_removal.resharp.radius", "RESHARP's kernel radius, mm; only with RESHARP"),
    ("bg_removal.resharp.max_iter", "RESHARP's CG iteration cap; only with RESHARP"),
    ("inversion.algorithm", "Dipole inversion (InversionAlgorithm::id)"),
    ("inversion.star.pad", "STAR-QSM padding per side, voxels (StarParams::pad)"),
    ("inversion.ilsqr.max_iter", "iLSQR's LSQR iteration cap (IlsqrParams::max_iter); only with iLSQR"),
    ("inversion.ilsqr.pad_mm", "iLSQR's padding, mm (IlsqrParams::pad_mm); only with iLSQR"),
    ("inversion.whqsm.weight", "WH-QSM's data weight: flat, max-normalised or in-vivo (WhQsmWeight); only with WH-QSM"),
    ("inversion.star.te", "Echo time STAR-QSM converts at (EchoTime: first, last, spacing, sum, seconds); only with STAR"),
    ("inversion.fansi.te", "Echo time FANSI nlTV/nlTGV convert at (EchoTime); only with fansi or fansi-tgv"),
    ("inversion.ndi.te", "Echo time NDI converts at (EchoTime); only with NDI"),
    ("inversion.l1qsm.te", "Echo time L1-QSM converts at (EchoTime); only with L1-QSM"),
    ("inversion.whqsm.te", "Echo time WH-QSM converts at (EchoTime); only with WH-QSM"),
    ("inversion.medi.msmv", "MEDI-mSMV: MediParams::msmv is set (bool); only with MEDI"),
    ("inversion.medi.msmv.prefilter", "MEDI-mSMV's SMV prefilter (MsmvParams::prefilter, bool); only with MEDI-mSMV"),
    ("inversion.medi.te", "Echo time MEDI converts at (EchoTime); only with MEDI"),
    ("inversion.tfi.te", "Echo time TFI converts at (EchoTime); only with TFI"),
    ("inversion.heidi.te", "Echo time HEIDI converts at (EchoTime); only with HEIDI"),
    ("inversion.qsmart.inversion", "QSMART's inner inversion (InversionAlgorithm::id); only with QSMART"),
    ("inversion.qsmart.ilsqr_max_iter", "QSMART's iLSQR iteration count; only with QSMART"),
    ("inversion.qsmart.ilsqr_pad_mm", "QSMART's iLSQR padding, mm; only with QSMART"),
    ("reference", "QSM referencing (QsmReference: mean or none)"),
    ("separation.algorithm", "χ-separation (SeparationAlgorithm::id)"),
    ("segmentation.inference", "SynthSeg inference: full-volume, low-memory (SynthSegInference::LOW_MEMORY) or sliding-window"),
    ("segmentation.inference.patch", "SynthSeg window edge, voxels at 1 mm (SlidingWindow::patch)"),
];

/// Stable id of an [`EchoTime`](crate::EchoTime) choice (`Seconds` is `"seconds"`, whatever its
/// value).
pub fn echo_time_id(t: crate::EchoTime) -> &'static str {
    match t {
        crate::EchoTime::First => "first",
        crate::EchoTime::Last => "last",
        crate::EchoTime::Spacing => "spacing",
        crate::EchoTime::Sum => "sum",
        crate::EchoTime::Seconds(_) => "seconds",
    }
}

/// Stable id of a weighted average's per-echo weight.
pub fn weight_id(w: &crate::utils::multi_echo::B0WeightType) -> &'static str {
    use crate::utils::multi_echo::B0WeightType as W;
    match w {
        W::PhaseSNR => "phase-snr",
        W::PhaseVar => "phase-var",
        W::Average => "average",
        W::TEs => "tes",
        W::Mag => "mag",
        W::AssumedDecay { .. } => "assumed-decay",
    }
}

/// Every HD-BET step in the mask recipe.
fn hd_bet_steps(c: &QsmPipelineConfig) -> impl Iterator<Item = &HdBetParams> {
    c.masking.sections.iter().flat_map(|s| std::iter::once(&s.generator).chain(&s.refinements)).filter_map(
        |op| match op {
            super::config::MaskOp::HdBet(p) => Some(p),
            _ => None,
        },
    )
}

/// The current value(s) of `setting` in `input`: empty when the setting is not in play (its
/// stage does not run, or its method is not selected), one value per place it is set otherwise.
/// `None` for a path that is not in [`SETTINGS`].
///
/// Whether background removal and field mapping run depends on the inversion's input
/// ([`InversionConfig::input`](super::config::InversionConfig::input)), which for TGV depends on
/// the echo count; that difference is only between two inputs that both skip background removal,
/// so one echo is assumed when the count is not known.
pub fn current(input: &RuleInput, facts: &ScanFacts, setting: &str) -> Option<Vec<Current>> {
    let c = input.pipeline;
    let inv_input = c.inversion.input(facts.echo_count.unwrap_or(1));
    let one = |v: Current| vec![v];
    let gated = |on: bool, v: Current| if on { vec![v] } else { vec![] };
    let bg = |a: super::config::BgRemovalAlgorithm| inv_input.needs_bg_removal() && c.bg_removal.algorithm == a;
    let vsharp = bg(super::config::BgRemovalAlgorithm::Vsharp);
    let resharp = bg(super::config::BgRemovalAlgorithm::Resharp);
    use super::config::InversionAlgorithm as A;
    let alg = |a: &[A]| a.contains(&c.inversion.algorithm);
    let qsmart = alg(&[A::Qsmart]);
    Some(match setting {
        "masking.generator" => {
            c.masking.sections.iter().map(|s| Current::Choice(s.generator.id())).collect()
        }
        "masking.hd_bet.inference" => hd_bet_steps(c)
            .map(|p| {
                Current::Choice(match p.inference {
                    HdBetInference::FullVolume => "full-volume",
                    i if i == HdBetInference::SLIDING_WINDOW => "sliding-window",
                    i if i == HdBetInference::LOW_MEMORY => "low-memory",
                    HdBetInference::SlidingWindow { .. } => "custom-window",
                })
            })
            .collect(),
        "field_mapping.unwrapping_algorithm" => gated(
            inv_input.needs_field_mapping(),
            Current::Choice(c.field_mapping.unwrapping_algorithm.id()),
        ),
        "field_mapping.b0_estimation" => gated(
            inv_input.needs_field_mapping(),
            Current::Choice(c.field_mapping.b0_estimation.id()),
        ),
        "field_mapping.b0_estimation.weighting" => match &c.field_mapping.b0_estimation {
            super::config::B0EstimationMethod::WeightedAvg { weighting } if inv_input.needs_field_mapping() => {
                one(Current::Choice(weight_id(weighting)))
            }
            _ => vec![],
        },
        "field_mapping.phase_offset_removal" => gated(
            inv_input.needs_field_mapping(),
            Current::Bool(c.field_mapping.phase_offset_removal),
        ),
        "field_mapping.bipolar_correction" => gated(
            inv_input.needs_field_mapping(),
            Current::Bool(c.field_mapping.bipolar_correction),
        ),
        "reference" => one(Current::Choice(match c.reference {
            super::config::QsmReference::Mean => "mean",
            super::config::QsmReference::None => "none",
        })),
        "field_mapping.laplacian_solver" => {
            use crate::unwrap::LaplacianSolver as L;
            gated(
                inv_input.needs_field_mapping()
                    && c.field_mapping.unwrapping_algorithm == super::config::UnwrappingAlgorithm::Laplacian,
                Current::Choice(match c.field_mapping.laplacian_solver {
                    L::Dct => "dct",
                    s if s == L::FFT_DEFAULT_PAD => "fft-12",
                    L::Fft { .. } => "fft",
                    L::Mirror => "mirror",
                }),
            )
        }
        "bg_removal.vsharp.deconvolve" => gated(vsharp, Current::Bool(c.bg_removal.vsharp.deconvolve)),
        "bg_removal.vsharp.max_radius" => gated(vsharp, Current::Number(c.bg_removal.vsharp.max_radius)),
        "bg_removal.vsharp.min_radius" => gated(vsharp, Current::Number(c.bg_removal.vsharp.min_radius)),
        "bg_removal.vsharp.radius_step" => gated(vsharp, Current::Number(c.bg_removal.vsharp.radius_step)),
        "field_mapping.phase_offset_sigma_mm" => gated(
            inv_input.needs_field_mapping() && c.field_mapping.phase_offset_removal,
            Current::Vector(c.field_mapping.phase_offset_sigma_mm.to_vec()),
        ),
        "inversion.ilsqr.max_iter" => gated(alg(&[A::Ilsqr]), Current::Number(c.inversion.ilsqr.max_iter as f64)),
        "inversion.ilsqr.pad_mm" => gated(alg(&[A::Ilsqr]), Current::Number(c.inversion.ilsqr.pad_mm)),
        "bg_removal.vsharp.grid" => gated(
            vsharp,
            Current::Choice(match c.bg_removal.vsharp.grid {
                crate::bgremove::VsharpGrid::Native => "native",
                crate::bgremove::VsharpGrid::IsotropicFinest => "isotropic-finest",
                crate::bgremove::VsharpGrid::Isotropic { spacing_mm: 1.0 } => "isotropic-1mm",
                crate::bgremove::VsharpGrid::Isotropic { .. } => "isotropic",
            }),
        ),
        "bg_removal.resharp.radius" => gated(resharp, Current::Number(c.bg_removal.resharp.radius)),
        "bg_removal.resharp.max_iter" => gated(resharp, Current::Number(c.bg_removal.resharp.max_iter as f64)),
        "inversion.whqsm.weight" => gated(
            alg(&[A::Whqsm]),
            Current::Choice(match c.inversion.whqsm.weight {
                crate::inversion::WhQsmWeight::Flat => "flat",
                crate::inversion::WhQsmWeight::MaxNormalised => "max-normalised",
                crate::inversion::WhQsmWeight::InVivo => "in-vivo",
            }),
        ),
        "inversion.star.te" => gated(alg(&[A::Star]), Current::Choice(echo_time_id(c.inversion.star.te))),
        "inversion.fansi.te" => {
            gated(alg(&[A::Fansi, A::FansiTgv]), Current::Choice(echo_time_id(c.inversion.fansi.te)))
        }
        "inversion.ndi.te" => gated(alg(&[A::Ndi]), Current::Choice(echo_time_id(c.inversion.ndi.te))),
        "inversion.l1qsm.te" => gated(alg(&[A::L1qsm]), Current::Choice(echo_time_id(c.inversion.l1qsm.te))),
        "inversion.whqsm.te" => gated(alg(&[A::Whqsm]), Current::Choice(echo_time_id(c.inversion.whqsm.te))),
        "inversion.medi.msmv" => gated(alg(&[A::Medi]), Current::Bool(c.inversion.medi.msmv.is_some())),
        "inversion.medi.msmv.prefilter" => match &c.inversion.medi.msmv {
            Some(m) if alg(&[A::Medi]) => one(Current::Bool(m.prefilter)),
            _ => vec![],
        },
        "inversion.medi.te" => gated(alg(&[A::Medi]), Current::Choice(echo_time_id(c.inversion.medi.te))),
        "inversion.tfi.te" => gated(alg(&[A::Tfi]), Current::Choice(echo_time_id(c.inversion.tfi.te))),
        "inversion.heidi.te" => gated(alg(&[A::Heidi]), Current::Choice(echo_time_id(c.inversion.heidi.te))),
        "inversion.qsmart.inversion" => gated(qsmart, Current::Choice(c.inversion.qsmart.inversion.id())),
        "inversion.qsmart.ilsqr_max_iter" => {
            gated(qsmart, Current::Number(c.inversion.qsmart.ilsqr_max_iter as f64))
        }
        "inversion.qsmart.ilsqr_pad_mm" => gated(qsmart, Current::Number(c.inversion.qsmart.ilsqr_pad_mm)),
        "bg_removal.algorithm" => gated(
            inv_input.needs_bg_removal(),
            Current::Choice(c.bg_removal.algorithm.id()),
        ),
        "inversion.algorithm" => one(Current::Choice(c.inversion.algorithm.id())),
        "inversion.star.pad" => gated(
            c.inversion.algorithm == super::config::InversionAlgorithm::Star,
            Current::Vector(c.inversion.star.pad.iter().map(|&p| p as f64).collect()),
        ),
        "separation.algorithm" => {
            input.separation.map(|s| Current::Choice(s.algorithm.id())).into_iter().collect()
        }
        "segmentation.inference" => input
            .segmentation
            .map(|s| {
                Current::Choice(match s.inference {
                    SynthSegInference::FullVolume => "full-volume",
                    i if i == SynthSegInference::LOW_MEMORY => "low-memory",
                    SynthSegInference::SlidingWindow { .. } => "sliding-window",
                })
            })
            .into_iter()
            .collect(),
        "segmentation.inference.patch" => input
            .segmentation
            .and_then(|s| match s.inference {
                SynthSegInference::SlidingWindow { patch, .. } => Some(Current::Number(patch as f64)),
                SynthSegInference::FullVolume => None,
            })
            .into_iter()
            .collect(),
        _ => return None,
    })
}

// =========================================================================
// Evaluation
// =========================================================================

fn holds(cond: &Condition, input: &RuleInput, facts: &ScanFacts) -> bool {
    match cond {
        Condition::Setting { setting, is } => {
            current(input, facts, setting).unwrap_or_default().iter().any(|v| is.passes(v))
        }
        Condition::Fact { fact, is } => facts.fact(*fact, input).is_some_and(|v| is.passes(&v)),
    }
}

/// Check one rule: `Some` finding when its conditions hold and its setting does not pass.
pub fn check(rule: &'static Rule, input: &RuleInput, facts: &ScanFacts) -> Option<Finding> {
    if !rule.when.iter().all(|c| holds(c, input, facts)) {
        return None;
    }
    let current = current(input, facts, rule.setting).unwrap_or_default();
    if current.is_empty() || current.iter().all(|v| rule.expect.passes(v)) {
        return None;
    }
    Some(Finding { rule, current })
}

/// Every finding for `input` on a scan with `facts`, in [`RULES`] order.
pub fn evaluate(input: &RuleInput, facts: &ScanFacts) -> Vec<Finding> {
    RULES.iter().filter_map(|r| check(r, input, facts)).collect()
}


// =========================================================================
// The rules
// =========================================================================

use Condition as C;
use Value::{Bool, Choice, Number};

const fn method(setting: &'static str, values: &'static [Value]) -> Condition {
    C::Setting { setting, is: Test::OneOf { values } }
}
const fn inv(values: &'static [Value]) -> Condition {
    method("inversion.algorithm", values)
}
const fn bg(values: &'static [Value]) -> Condition {
    method("bg_removal.algorithm", values)
}
const fn sep(values: &'static [Value]) -> Condition {
    method("separation.algorithm", values)
}
const fn fact_at_least(fact: Fact, min: f64) -> Condition {
    C::Fact { fact, is: Test::Range { min: Some(min), max: None } }
}
const fn fact_at_most(fact: Fact, max: f64) -> Condition {
    C::Fact { fact, is: Test::Range { min: None, max: Some(max) } }
}
const ON_WASM: Condition = C::Fact { fact: Fact::Target, is: Test::OneOf { values: &[Choice("wasm")] } };
const MULTI_ECHO: Condition = fact_at_least(Fact::EchoCount, 2.0);

/// A choice of `strength` on `basis`: `setting` should be one of `values`, the first being the
/// default.
const fn one_of(
    id: &'static str,
    when: &'static [Condition],
    setting: &'static str,
    values: &'static [Value],
    strength: Strength,
    basis: Basis,
    reason: &'static str,
    source: &'static str,
) -> Rule {
    Rule {
        id,
        when,
        setting,
        expect: Test::OneOf { values },
        default: values[0],
        unit: Unit::None,
        strength,
        basis,
        reason,
        source,
    }
}

/// What the authors explicitly recommend.
const fn recommend(
    id: &'static str,
    when: &'static [Condition],
    setting: &'static str,
    values: &'static [Value],
    reason: &'static str,
    source: &'static str,
) -> Rule {
    one_of(id, when, setting, values, Strength::Recommended, Basis::Stated, reason, source)
}

/// What QSM.rs measured to be better (recorded in the method's docs).
const fn measured_better(
    id: &'static str,
    when: &'static [Condition],
    setting: &'static str,
    values: &'static [Value],
    reason: &'static str,
    source: &'static str,
) -> Rule {
    one_of(id, when, setting, values, Strength::Recommended, Basis::Measured, reason, source)
}

/// What the reference pipeline used, without the authors saying it is needed or better.
const fn reference(
    id: &'static str,
    when: &'static [Condition],
    setting: &'static str,
    values: &'static [Value],
    reason: &'static str,
    source: &'static str,
) -> Rule {
    one_of(id, when, setting, values, Strength::Reference, Basis::ReferenceChoice, reason, source)
}

/// A discouraged choice: `setting` should be none of `avoid`; `default` replaces it.
const fn discourage(
    id: &'static str,
    when: &'static [Condition],
    setting: &'static str,
    avoid: &'static [Value],
    default: Value,
    basis: Basis,
    reason: &'static str,
    source: &'static str,
) -> Rule {
    Rule {
        id,
        when,
        setting,
        expect: Test::NoneOf { values: avoid },
        default,
        unit: Unit::None,
        strength: Strength::Discouraged,
        basis,
        reason,
        source,
    }
}

const BET: &[Value] = &[Choice("bet")];
const PDF: &[Value] = &[Choice("pdf")];
const VSHARP: &[Value] = &[Choice("vsharp")];
const LAPLACIAN: &[Value] = &[Choice("laplacian")];
const ROMEO: &[Value] = &[Choice("romeo")];
const COMPLEX_FIT: &[Value] = &[Choice("complex-fit")];
const WEIGHTED_AVG: &[Value] = &[Choice("weighted-avg")];
const CHI_SEP_TOOLBOX: &[Value] = &[Choice("chi-sep-ilsqr"), Choice("chi-sep-medi"), Choice("chi-sepnet")];
const STI: &[Value] = &[Choice("star"), Choice("ilsqr")];
const FANSI_FAMILY: &[Value] =
    &[Choice("ndi"), Choice("fansi"), Choice("fansi-tgv"), Choice("l1qsm"), Choice("whqsm")];

/// The catalogue. Each rule's reason and source are also recorded in the docs of the method it
/// is about (its "Reference pipeline" section). Rules that depend on work not yet merged are
/// listed in issue #289, not here.
pub static RULES: &[Rule] = &[
    // ---------------------------------------------------------------- Brain masking
    reference("medi.mask-bet", &[inv(&[Choice("medi")])], "masking.generator", BET,
        "MEDI's pipeline masks with FSL BET (bet2, f 0.5) on the root-sum-of-squares magnitude",
        "MEDI toolbox 2024.11.26 README.m:80,118; functions/bet2/bet2.cpp:136"),
    reference("tfi.mask-bet", &[inv(&[Choice("tfi")])], "masking.generator", BET,
        "TFI's in-vivo ROI is a BET brain mask",
        "Liu et al. 2017 MRM 79:2950 (TFI), Methods"),
    reference("pdf.mask-bet", &[bg(PDF)], "masking.generator", BET,
        "MEDI's PDF runs inside the BET mask",
        "MEDI toolbox 2024.11.26 README.m:118,124"),
    reference("amp-pe.mask-bet", &[inv(&[Choice("amp-pe")])], "masking.generator", BET,
        "AMP-PE's pipeline masks with BET (0.5) on the RSS magnitude, then erodes 2 mm",
        "QSM_AMP_PE qsm_multi_echo_combined.m L71-81"),
    recommend("qsmnet.mask-bet", &[inv(&[Choice("qsmnet"), Choice("qsmnet-plus")])], "masking.generator", BET,
        "QSMnet's README instructs BET (FSL) brain extraction before its input is made",
        "SNU-LIST/QSMnet README.md:64-68 (Process flow)"),
    reference("qsmgan.mask-bet", &[inv(&[Choice("qsmgan")])], "masking.generator", BET,
        "QSMGAN's training inputs were masked with FSL BET",
        "Chen et al. 2020 NeuroImage 207:116389 (QSMGAN), Methods"),
    reference("lpcnn.mask-bet", &[inv(&[Choice("lpcnn")])], "masking.generator", BET,
        "LPCNN's data were masked with FSL BET",
        "Lai et al. MICCAI 2020 (LPCNN, arXiv:2008.05024), Dataset Acquisition and Preprocessing"),
    reference("nextqsm.mask-bet", &[inv(&[Choice("nextqsm")])], "masking.generator", BET,
        "NeXtQSM's in-vivo pipeline masks with FSL BET",
        "Cognolato et al. 2023 Med. Image Anal. 84:102700 (NeXtQSM), §3.2"),
    reference("iqsm.mask-bet", &[inv(&[Choice("iqsm"), Choice("iqsm-plus")])], "masking.generator", BET,
        "iQSM's default mask is bet2 -f 0.5 on the first-echo magnitude",
        "sunhongfu/iQSM run.py:499-506, bet2_utils.py:47,64"),
    reference("bfrnet.mask-bet", &[bg(&[Choice("bfrnet")])], "masking.generator", BET,
        "BFRnet's in-vivo masks are bet2 -f 0.2 -w 2, eroded 3 voxels",
        "Zhu et al. 2023 Z. Med. Phys. (BFRnet), Methods"),
    reference("chi-sep.mask-bet", &[sep(CHI_SEP_TOOLBOX)], "masking.generator", BET,
        "The χ-separation toolbox and χ-sepnet mask with BET",
        "χ-separation toolbox Chisep_script.m:79-82; Kim et al. 2025 (χ-sepnet, arXiv:2409.14030) §2.1.1"),
    reference("qsmart.mask-bet", &[inv(&[Choice("qsmart")])], "masking.generator", BET,
        "QSMART's authors mask with FSL bet2 -f 0.2 on the N4-corrected first-echo magnitude",
        "Yaghmaie et al. 2021 NeuroImage 231:117701 (QSMART) §3.1; QSMART brainmask.m l. 7-33"),
    reference("decompose.mask-bet", &[sep(&[Choice("decompose")])], "masking.generator", BET,
        "DECOMPOSE's pipeline masks with FSL BET",
        "Chen et al. 2021 NeuroImage 242:118735 (DECOMPOSE), §3"),
    Rule {
        id: "mindgrab.wasm-heap",
        when: &[method("masking.generator", &[Choice("mindgrab")]), ON_WASM],
        setting: "masking.generator",
        expect: Test::NoneOf { values: &[Choice("mindgrab")] },
        default: Choice("bet"),
        unit: Unit::None,
        strength: Strength::Required,
        basis: Basis::Measured,
        reason: "MindGrab always runs a whole 256³ volume and cannot be tiled; its WASM peak \
                 (about 5.6 GB) exceeds a 4 GiB heap",
        source: "QSM.rs models registry (mindgrab memory figures) and bet::mindgrab docs",
    },
    Rule {
        id: "hd-bet.wasm-heap",
        when: &[
            method("masking.generator", &[Choice("hd-bet")]),
            ON_WASM,
            fact_at_least(Fact::HdBetWasmBytes, (WASM32_HEAP_BYTES + 1) as f64),
        ],
        setting: "masking.hd_bet.inference",
        expect: Test::OneOf { values: &[Choice("low-memory")] },
        default: Choice("low-memory"),
        unit: Unit::None,
        strength: Strength::Required,
        basis: Basis::Measured,
        reason: "The configured HD-BET step needs more than a 4 GiB WASM heap on this scan; \
                 (128, 128, 64) sliding windows peak at about 1.6 GB and agree with the trained \
                 patch at Dice ≈0.99",
        source: "QSM.rs bet::hdbet, HdBetParams::low_memory and hd_bet_memory_bytes (#280)",
    },
    // ---------------------------------------------------------------- Segmentation
    Rule {
        id: "synthseg.wasm-windows",
        when: &[method("segmentation.inference", &[Choice("full-volume")]), ON_WASM],
        setting: "segmentation.inference",
        expect: Test::NoneOf { values: &[Choice("full-volume")] },
        default: Choice("low-memory"),
        unit: Unit::None,
        strength: Strength::Required,
        basis: Basis::Measured,
        reason: "SynthSeg's whole-volume pass needs 6–12 GB on 1 mm head scans; no whole-volume \
                 setting fits a 4 GB WASM heap",
        source: "QSM.rs segment::synthseg module docs and SynthSegInference::LOW_MEMORY (#281)",
    },
    Rule {
        id: "synthseg.wasm-128-window",
        when: &[
            C::Setting {
                setting: "segmentation.inference.patch",
                is: Test::Range { min: Some(128.0), max: None },
            },
            ON_WASM,
            fact_at_least(Fact::SynthSegNetworkVoxels, (192 * 256 * 128 + 1) as f64),
        ],
        setting: "segmentation.inference.patch",
        expect: Test::Range { min: None, max: Some(96.0) },
        default: Number(96.0),
        unit: Unit::Voxels,
        strength: Strength::Required,
        basis: Basis::Measured,
        reason: "128³ SynthSeg windows took a 192×256×128 grid to 3.83 GB of the 4 GB WASM heap; \
                 a larger grid does not fit, 96³ windows do",
        source: "QSM.rs SynthSegInference::LOW_MEMORY docs (#281)",
    },
    // ---------------------------------------------------------------- Phase unwrapping
    discourage("fansi.no-laplacian-multi-echo", &[inv(FANSI_FAMILY), MULTI_ECHO],
        "field_mapping.unwrapping_algorithm", LAPLACIAN, Choice("romeo"), Basis::Stated,
        "FANSI's authors advise against Laplacian unwrapping for multi-echo data: it removes \
         harmonic components of the phase",
        "FANSI-toolbox script_DICOM_Philips.m L196-197 (93044a3)"),
    recommend("qsmnet.unwrap-laplacian", &[inv(&[Choice("qsmnet"), Choice("qsmnet-plus")])],
        "field_mapping.unwrapping_algorithm", LAPLACIAN,
        "QSMnet's README instructs Laplacian phase unwrapping (STI Suite)",
        "SNU-LIST/QSMnet README.md:64-68 (Process flow)"),
    reference("autoqsm.unwrap-laplacian", &[inv(&[Choice("autoqsm")])],
        "field_mapping.unwrapping_algorithm", LAPLACIAN,
        "AutoQSM was trained on Laplacian-unwrapped total phase",
        "Wei et al. 2019 NeuroImage 201:116064 (AutoQSM), Methods"),
    reference("qsmgan.unwrap-laplacian", &[inv(&[Choice("qsmgan")])],
        "field_mapping.unwrapping_algorithm", LAPLACIAN,
        "QSMGAN's training inputs were Laplacian-unwrapped",
        "Chen et al. 2020 NeuroImage 207:116389 (QSMGAN), Methods"),
    reference("decompose.unwrap-laplacian", &[sep(&[Choice("decompose")])],
        "field_mapping.unwrapping_algorithm", LAPLACIAN,
        "DECOMPOSE's pipeline uses Laplacian unwrapping",
        "Chen et al. 2021 NeuroImage 242:118735 (DECOMPOSE), §3"),
    reference("chi-sep.unwrap-romeo", &[sep(CHI_SEP_TOOLBOX)],
        "field_mapping.unwrapping_algorithm", ROMEO,
        "The χ-separation toolbox and χ-sepnet unwrap each echo with ROMEO",
        "χ-separation toolbox Chisep_script.m:99-111; Kim et al. 2025 (χ-sepnet) §2.1.1"),
    reference("sti.unwrap-laplacian", &[inv(STI)], "field_mapping.unwrapping_algorithm", LAPLACIAN,
        "STI Suite's QSM_GUI unwraps each echo with MRPhaseUnwrap (Laplacian)",
        "STI Suite 3.0 QSM_GUI, probed headless (recorded in QSM.rs laplacian.rs docs)"),
    reference("sti.laplacian-solver",
        &[inv(STI), method("field_mapping.unwrapping_algorithm", LAPLACIAN)],
        "field_mapping.laplacian_solver", &[Choice("fft-12")],
        "MRPhaseUnwrap is the FFT Schofield-Zhu solver with 12 voxels of padding",
        "STI Suite 3.0 MRPhaseUnwrap, probed (recorded in QSM.rs LaplacianSolver::Fft docs)"),
    reference("qsmart.unwrap-laplacian", &[inv(&[Choice("qsmart")])],
        "field_mapping.unwrapping_algorithm", LAPLACIAN,
        "QSMART unwraps each echo with Laplacian unwrapping (lapunwrap.m)",
        "QSMART Demo_QSMART.m l. 32; unwrap_phase.m l. 5-10"),
    reference("qsmart.laplacian-solver",
        &[inv(&[Choice("qsmart")]), method("field_mapping.unwrapping_algorithm", LAPLACIAN)],
        "field_mapping.laplacian_solver", &[Choice("mirror")],
        "QSMART's lapunwrap.m solves on the mirrored volume",
        "QSMART unwrap_phase.m l. 5-10, lapunwrap.m"),
    // ---------------------------------------------------------------- Field mapping
    reference("qsmart.field-echofit", &[inv(&[Choice("qsmart")])],
        "field_mapping.b0_estimation", &[Choice("qsmart-echofit")],
        "QSMART's total field and reliability map R_0 come from echofit.m",
        "QSMART QSMART.m l. 26; echofit.m"),
    one_of("qsmart-echofit.unwrap-laplacian",
        &[method("field_mapping.b0_estimation", &[Choice("qsmart-echofit")])],
        "field_mapping.unwrapping_algorithm", LAPLACIAN, Strength::Required, Basis::Implementation,
        "QSMART's echo fit unwraps each echo over the whole volume with Laplacian unwrapping; the \
         pipeline refuses other unwrapping with it",
        "QSMART unwrap_phase.m l. 5-10, echofit.m; QSM.rs B0EstimationMethod::QsmartEchofit docs"),
    reference("sti.echo-combination", &[inv(STI), MULTI_ECHO],
        "field_mapping.b0_estimation", WEIGHTED_AVG,
        "STI Suite's QSM_GUI sums the unwrapped echoes and divides by the summed echo times",
        "STI Suite 3.0 QSM_GUI, probed headless (recorded in QSM.rs star.rs docs)"),
    reference("sti.echo-weights", &[inv(STI), MULTI_ECHO],
        "field_mapping.b0_estimation.weighting", &[Choice("tes")],
        "Σφ/ΣTE is the TE-weighted mean of φ/TE",
        "STI Suite 3.0 QSM_GUI, probed headless (recorded in QSM.rs star.rs docs)"),
    recommend("field-mapping.phase-offsets",
        &[method("field_mapping.b0_estimation", WEIGHTED_AVG),
          method("field_mapping.unwrapping_algorithm", ROMEO), MULTI_ECHO],
        "field_mapping.phase_offset_removal", &[Bool(true)],
        "The B0 calculation's authors state that phase offsets have to be removed before it; \
         ROMEO removes them (MCPC-3D-S) whenever it computes B0 from multi-echo data",
        "MriResearchTools.jl romeofunctions.jl:3-7; ROMEO.jl caller.jl:80-91"),
    Rule {
        id: "field-mapping.phase-offset-sigma",
        when: &[method("field_mapping.phase_offset_removal", &[Bool(true)])],
        setting: "field_mapping.phase_offset_sigma_mm",
        expect: Test::OneOf { values: &[Value::Vector(&super::config::ROMEO_CLI_PHASE_OFFSET_SIGMA_MM)] },
        default: Value::Vector(&super::config::ROMEO_CLI_PHASE_OFFSET_SIGMA_MM),
        unit: Unit::Mm,
        strength: Strength::Reference,
        basis: Basis::ReferenceChoice,
        reason: "The ROMEO command line smooths the MCPC-3D-S phase offsets with σ = 7 mm per axis",
        source: "ROMEO.jl 08776d4 argparse.jl l. 79-84, 217-224; caller.jl l. 137-138 (#293)",
    },
    discourage("field-mapping.bipolar-needs-three-echoes",
        &[method("field_mapping.bipolar_correction", &[Bool(true)]), fact_at_most(Fact::EchoCount, 2.0)],
        "field_mapping.bipolar_correction", &[Bool(true)], Bool(false), Basis::Stated,
        "Bipolar correction needs echoes 1–3; with fewer it is skipped",
        "ROMEO.jl argparse.jl:71-76; MriResearchTools.jl mcpc3ds.jl:110-119"),
    discourage("field-mapping.bipolar-not-with-laplacian",
        &[method("field_mapping.bipolar_correction", &[Bool(true)]),
          method("field_mapping.unwrapping_algorithm", LAPLACIAN)],
        "field_mapping.bipolar_correction", &[Bool(true)], Bool(false), Basis::Implementation,
        "Bipolar correction is not applied on the Laplacian path",
        "QSM.rs pipeline::field_mapping docs"),
    reference("medi.field-complex-fit", &[inv(&[Choice("medi")])],
        "field_mapping.b0_estimation", COMPLEX_FIT,
        "MEDI's field map and its noise weight N_std come from Fit_ppm_complex",
        "MEDI toolbox 2024.11.26 README.m:87,140; dataterm_mask.m:23-27"),
    reference("pdf.field-complex-fit", &[bg(PDF)],
        "field_mapping.b0_estimation", COMPLEX_FIT,
        "PDF is weighted by Fit_ppm_complex's noise map N_std",
        "MEDI toolbox 2024.11.26 README.m:87,124; PDF.m:74-76"),
    reference("r2star-qsm.field-complex-fit", &[sep(&[Choice("r2star-qsm")])],
        "field_mapping.b0_estimation", COMPLEX_FIT,
        "R2*-QSM's field was a nonlinear complex fit",
        "Dimov et al. 2022 Tomography 8:1544, Methods"),
    reference("amp-pe.echo-combination", &[inv(&[Choice("amp-pe")]), MULTI_ECHO],
        "field_mapping.b0_estimation", WEIGHTED_AVG,
        "AMP-PE's pipeline averages φ/TE over the echoes",
        "QSM_AMP_PE qsm_multi_echo_combined.m L100-104"),
    reference("amp-pe.echo-weights", &[inv(&[Choice("amp-pe")]), MULTI_ECHO],
        "field_mapping.b0_estimation.weighting", &[Choice("average")],
        "AMP-PE's average of φ/TE is unweighted",
        "QSM_AMP_PE qsm_multi_echo_combined.m L100-104"),
    reference("qsmnet.echo-combination", &[inv(&[Choice("qsmnet"), Choice("qsmnet-plus")]), MULTI_ECHO],
        "field_mapping.b0_estimation", WEIGHTED_AVG,
        "QSMnet's input is the summed phase over the summed echo times",
        "SNU-LIST/QSMnet save_input_data_for_QSMnet.m:22"),
    reference("qsmnet.echo-weights", &[inv(&[Choice("qsmnet"), Choice("qsmnet-plus")]), MULTI_ECHO],
        "field_mapping.b0_estimation.weighting", &[Choice("tes")],
        "Σφ/ΣTE is the TE-weighted mean of φ/TE",
        "SNU-LIST/QSMnet save_input_data_for_QSMnet.m:22"),
    reference("autoqsm.echo-combination", &[inv(&[Choice("autoqsm")]), MULTI_ECHO],
        "field_mapping.b0_estimation", WEIGHTED_AVG,
        "AutoQSM's input is ψ = Σφ/(γB0ΣTE)",
        "Wei et al. 2019 NeuroImage 201:116064 (AutoQSM), Methods"),
    reference("autoqsm.echo-weights", &[inv(&[Choice("autoqsm")]), MULTI_ECHO],
        "field_mapping.b0_estimation.weighting", &[Choice("tes")],
        "Σφ/ΣTE is the TE-weighted mean of φ/TE",
        "Wei et al. 2019 NeuroImage 201:116064 (AutoQSM), Methods"),
    reference("chi-sep.echo-combination", &[sep(CHI_SEP_TOOLBOX), MULTI_ECHO],
        "field_mapping.b0_estimation", WEIGHTED_AVG,
        "The χ-separation toolbox combines the unwrapped echoes by a weighted average",
        "χ-separation toolbox Chisep_script.m:113-123; Kim et al. 2025 (χ-sepnet) §2.1.1"),
    reference("chi-sep.echo-weights", &[sep(CHI_SEP_TOOLBOX), MULTI_ECHO],
        "field_mapping.b0_estimation.weighting", &[Choice("assumed-decay")],
        "The toolbox weights the echoes by TE·exp(−TE/T2*) with T2* = 40 ms (AssumedDecay, 0.04 s)",
        "χ-separation toolbox Chisep_script.m:113-123"),
    // ---------------------------------------------------------------- Background field removal
    reference("hdqsm.bg-pdf", &[inv(&[Choice("hdqsm")])], "bg_removal.algorithm", PDF,
        "HD-QSM's in-vivo pipeline removes the background with PDF",
        "Lambert et al. 2022 MRM 88:1600 (HD-QSM), §3.3"),
    reference("amp-pe.bg-pdf", &[inv(&[Choice("amp-pe")])], "bg_removal.algorithm", PDF,
        "AMP-PE's pipeline removes the background with PDF",
        "QSM_AMP_PE README (Overview, step 4); qsm_multi_echo_combined.m L112-139"),
    reference("lsqr.bg-sharp", &[inv(&[Choice("lsqr"), Choice("heidi")])], "bg_removal.algorithm",
        &[Choice("sharp")],
        "The authors' LSQR/HEIDI pipeline inverts a SHARP local field",
        "qsm_heidi makefile L14-20, def.sh (4592631)"),
    reference("rts.bg-vsharp", &[inv(&[Choice("rts")])], "bg_removal.algorithm", VSHARP,
        "RTS's author's toolbox (QSM.jl) runs V-SHARP before it",
        "QSM.jl README.md:49 (fde4b92)"),
    reference("xqsm.bg-resharp", &[inv(&[Choice("xqsm")])], "bg_removal.algorithm", &[Choice("resharp")],
        "xQSM's in-vivo pipeline removes the background with RESHARP",
        "Gao et al. 2021 NMR Biomed 34:e4461 (xQSM), in-vivo experiments"),
    recommend("qsmnet.bg-vsharp", &[inv(&[Choice("qsmnet"), Choice("qsmnet-plus")])], "bg_removal.algorithm",
        VSHARP, "QSMnet's README instructs 3D V-SHARP (STI Suite) background removal",
        "SNU-LIST/QSMnet README.md:64-68 (Process flow), :75"),
    reference("qsmgan.bg-vsharp", &[inv(&[Choice("qsmgan")])], "bg_removal.algorithm", VSHARP,
        "QSMGAN was trained on V-SHARP local fields",
        "Chen et al. 2020 NeuroImage 207:116389 (QSMGAN), Methods; checkpoint config input_algo"),
    reference("lpcnn.bg-vsharp", &[inv(&[Choice("lpcnn")])], "bg_removal.algorithm", VSHARP,
        "LPCNN's local fields were from V-SHARP",
        "Lai et al. MICCAI 2020 (LPCNN), Dataset Acquisition and Preprocessing"),
    reference("chi-sep.bg-vsharp", &[sep(CHI_SEP_TOOLBOX)], "bg_removal.algorithm", VSHARP,
        "The χ-separation toolbox and χ-sepnet remove the background with V-SHARP",
        "χ-separation toolbox Chisep_script.m:136-139; Kim et al. 2025 (χ-sepnet) §2.1.1"),
    reference("r2star-qsm.bg-pdf", &[sep(&[Choice("r2star-qsm")])], "bg_removal.algorithm", PDF,
        "R2*-QSM's local field was from PDF",
        "Dimov et al. 2022 Tomography 8:1544, Methods"),
    reference("decompose.bg-vsharp", &[sep(&[Choice("decompose")])], "bg_removal.algorithm", VSHARP,
        "DECOMPOSE's pipeline removes the background with V-SHARP",
        "Chen et al. 2021 NeuroImage 242:118735 (DECOMPOSE), §3"),
    reference("sti.bg-vsharp", &[inv(STI)], "bg_removal.algorithm", VSHARP,
        "STI Suite's QSM_GUI feeds iLSQR and STAR-QSM the V-SHARP local field",
        "STI Suite 3.0 QSM_GUI, probed headless (recorded in QSM.rs star.rs and ilsqr.rs docs)"),
    reference("sti.vsharp-no-deconvolve", &[inv(STI), bg(VSHARP)], "bg_removal.vsharp.deconvolve",
        &[Bool(false)],
        "STI Suite's V_SHARP does not deconvolve",
        "STI Suite 3.0 V_SHARP, probed (recorded in QSM.rs vsharp.rs docs)"),
    Rule {
        id: "sti.vsharp-radius",
        when: &[inv(STI), bg(VSHARP)],
        setting: "bg_removal.vsharp.max_radius",
        expect: Test::OneOf { values: &[Number(12.0)] },
        default: Number(12.0),
        unit: Unit::Mm,
        strength: Strength::Reference,
        basis: Basis::ReferenceChoice,
        reason: "STI Suite's QSM_GUI runs V_SHARP with smvsize 12 mm (also V_SHARP's default)",
        source: "STI Suite 3.0 QSM_GUI and V_SHARP, probed (recorded in QSM.rs vsharp.rs docs)",
    },
    reference("sti.vsharp-grid", &[inv(STI), bg(VSHARP)], "bg_removal.vsharp.grid",
        &[Choice("isotropic-1mm")],
        "STI Suite's V_SHARP estimates the background on a 1 mm grid",
        "STI Suite 3.0 V_SHARP (recorded in QSM.rs vsharp.rs docs)"),
    Rule {
        id: "sti.vsharp-min-radius",
        when: &[inv(STI), bg(VSHARP)],
        setting: "bg_removal.vsharp.min_radius",
        expect: Test::OneOf { values: &[Number(crate::bgremove::vsharp::STI_MIN_RADIUS_MM)] },
        default: Number(crate::bgremove::vsharp::STI_MIN_RADIUS_MM),
        unit: Unit::Mm,
        strength: Strength::Reference,
        basis: Basis::ReferenceChoice,
        reason: "STI Suite's V_SHARP drops the voxels only its 1 mm ball fits, so its radii in \
                 effect run down to 2 mm",
        source: "STI Suite 3.0 V_SHARP, probed (ukb-qsmxt docs/08 step 4; QSM.rs VsharpParams, #293)",
    },
    Rule {
        id: "sti.vsharp-radius-step",
        when: &[inv(STI), bg(VSHARP)],
        setting: "bg_removal.vsharp.radius_step",
        expect: Test::OneOf { values: &[Number(crate::bgremove::vsharp::STI_RADIUS_STEP_MM)] },
        default: Number(crate::bgremove::vsharp::STI_RADIUS_STEP_MM),
        unit: Unit::Mm,
        strength: Strength::Reference,
        basis: Basis::ReferenceChoice,
        reason: "STI Suite's V_SHARP steps its kernel radius by 1 mm",
        source: "STI Suite 3.0 V_SHARP, probed (ukb-qsmxt docs/08; QSM.rs VsharpParams, #293)",
    },
    Rule {
        id: "sti.ilsqr-iterations",
        when: &[inv(&[Choice("ilsqr")])],
        setting: "inversion.ilsqr.max_iter",
        expect: Test::OneOf { values: &[Number(crate::inversion::STI_GUI_NITER as f64)] },
        default: Number(crate::inversion::STI_GUI_NITER as f64),
        unit: Unit::None,
        strength: Strength::Reference,
        basis: Basis::ReferenceChoice,
        reason: "STI Suite's QSM_GUI calls QSM_iLSQR with niter 40",
        source: "STI Suite 3.0 QSM_GUI, probed headless (QSM.rs IlsqrParams::max_iter, #293)",
    },
    Rule {
        id: "sti.ilsqr-pad",
        when: &[inv(&[Choice("ilsqr")])],
        setting: "inversion.ilsqr.pad_mm",
        expect: Test::OneOf { values: &[Number(crate::inversion::STI_GUI_PAD_MM)] },
        default: Number(crate::inversion::STI_GUI_PAD_MM),
        unit: Unit::Mm,
        strength: Strength::Reference,
        basis: Basis::ReferenceChoice,
        reason: "STI Suite's QSM_GUI calls QSM_iLSQR with padsize [12 12 12]",
        source: "STI Suite 3.0 QSM_GUI, probed headless (QSM.rs IlsqrParams::pad_mm, #293)",
    },
    reference("resharp.radius", &[bg(&[Choice("resharp")])], "bg_removal.resharp.radius",
        &[Number(3.0)],
        "RESHARP's author's code defaults to a 3 mm kernel",
        "sunhongfu/QSM background_field_removal/resharp.m l. 24-26"),
    Rule {
        id: "resharp.iterations",
        when: &[bg(&[Choice("resharp")])],
        setting: "bg_removal.resharp.max_iter",
        expect: Test::Range { min: Some(200.0), max: None },
        default: Number(200.0),
        unit: Unit::None,
        strength: Strength::Reference,
        basis: Basis::ReferenceChoice,
        reason: "RESHARP's author's code allows its CG solve up to 200 iterations",
        source: "sunhongfu/QSM background_field_removal/resharp.m l. 32-34, 92",
    },
    // ---------------------------------------------------------------- Dipole inversion
    Rule {
        id: "star.padding",
        when: &[inv(&[Choice("star")])],
        setting: "inversion.star.pad",
        expect: Test::OneOf { values: &[Value::Vector(&[8.0, 8.0, 20.0])] },
        default: Value::Vector(&[8.0, 8.0, 20.0]),
        unit: Unit::Voxels,
        strength: Strength::Reference,
        basis: Basis::ReferenceChoice,
        reason: "STI Suite 3.0's QSM_GUI passes padsize [8 8 20] to QSM_star on every grid",
        source: "STI Suite 3.0 QSM_GUI (probed headless, recorded in QSM.rs StarParams::pad, #283)",
    },
    reference("qsmart.inner-inversion", &[inv(&[Choice("qsmart")])], "inversion.qsmart.inversion",
        &[Choice("ilsqr")],
        "QSMART's two stages invert with STI Suite's iLSQR",
        "Yaghmaie et al. 2021 §3.4; QSMART QSMART.m l. 32-39"),
    Rule {
        id: "qsmart.ilsqr-iterations",
        when: &[inv(&[Choice("qsmart")])],
        setting: "inversion.qsmart.ilsqr_max_iter",
        expect: Test::OneOf { values: &[Number(50.0)] },
        default: Number(50.0),
        unit: Unit::None,
        strength: Strength::Reference,
        basis: Basis::ReferenceChoice,
        reason: "QSMART calls QSM_iLSQR with 'niter', 50",
        source: "QSMART QSMART.m l. 32-33; Yaghmaie et al. 2021 §3.4",
    },
    Rule {
        id: "qsmart.ilsqr-pad",
        when: &[inv(&[Choice("qsmart")])],
        setting: "inversion.qsmart.ilsqr_pad_mm",
        expect: Test::OneOf { values: &[Number(crate::inversion::STI_DEFAULT_PAD_MM)] },
        default: Number(crate::inversion::STI_DEFAULT_PAD_MM),
        unit: Unit::Mm,
        strength: Strength::Reference,
        basis: Basis::ReferenceChoice,
        reason: "QSMART calls QSM_iLSQR without 'padsize', so STI Suite's default padding applies",
        source: "QSMART QSMART.m l. 32; STI Suite QSM_iLSQR default, probed (QSM.rs ilsqr.rs docs)",
    },
    // ---------------------------------------------------------------- MEDI (#288)
    reference("medi.bg-pdf",
        &[inv(&[Choice("medi")]), method("inversion.medi.msmv", &[Bool(false)])],
        "bg_removal.algorithm", PDF,
        "MEDI's pipeline inverts the PDF local field (also InversionConfig::reference_bg_removal)",
        "MEDI toolbox 2024.11.26 README.m:124,144"),
    // ---------------------------------------------------------------- MEDI-mSMV (#285)
    Rule {
        id: "medi-msmv.three-echoes",
        when: &[inv(&[Choice("medi")]), method("inversion.medi.msmv", &[Bool(true)]),
                fact_at_most(Fact::EchoCount, 2.0)],
        setting: "inversion.medi.msmv",
        expect: Test::OneOf { values: &[Bool(false)] },
        default: Bool(false),
        unit: Unit::None,
        strength: Strength::Required,
        basis: Basis::Implementation,
        reason: "MEDI-mSMV's vessel mask needs R2* from arlo.m, which needs at least three echoes; \
                 the pipeline refuses fewer",
        source: "MEDI toolbox arlo.m:22-26; QSM.rs MediParams::msmv docs (#285)",
    },
    one_of("medi-msmv.bg-removal",
        &[inv(&[Choice("medi")]), method("inversion.medi.msmv", &[Bool(true)])],
        "bg_removal.algorithm", &[Choice("pdf"), Choice("vsharp"), Choice("lbv")],
        Strength::Reference, Basis::ReferenceChoice,
        "The mSMV authors run it on a local field from PDF, V-SHARP or LBV (their README's \
         example uses PDF; it lists these as examples, \"by PDF, VSHARP, LBV, etc.\", without \
         recommending one)",
        "agr78/mSMV README (in_file; MEDI example: PDF(...,0,100)); Roberts et al. 2024 MRM, Methods"),
    discourage("medi-msmv.prefilter",
        &[inv(&[Choice("medi")]), method("inversion.medi.msmv", &[Bool(true)])],
        "inversion.medi.msmv.prefilter", &[Bool(false)], Bool(true), Basis::Measured,
        "With the prefilter off, MEDI fits the unfiltered field with the SMV-filtered kernel: \
         NRMSE 280-740 % on the phantom after PDF, V-SHARP or LBV, and 481 % in the authors' MATLAB",
        "QSM.rs inversion::msmv docs and MsmvParams::prefilter (#285)"),
    // ---------------------------------------------------------------- Echo time of the ppm↔rad conversion (#276)
    // With one echo every choice is the same time, so these need several.
    measured_better("fansi.te-spacing", &[inv(&[Choice("fansi"), Choice("fansi-tgv")]), MULTI_ECHO],
        "inversion.fansi.te", &[Choice("spacing")],
        "FANSI's multi-echo script converts at a TE difference; at the echo spacing nlTV and nlTGV \
         did better than at the first echo on every metric on the phantom",
        "FANSI-toolbox script_DICOM_Philips.m L364, L370; QSM.rs FansiParams::te docs (#276)"),
    measured_better("ndi.te-first", &[inv(&[Choice("ndi")]), MULTI_ECHO],
        "inversion.ndi.te", &[Choice("first")],
        "FANSI's script allows any single echo time; the first echo gave NDI its best NRMSE and \
         correlation on the phantom",
        "FANSI-toolbox script_DICOM_Philips.m L364; QSM.rs NdiParams::te docs (#276)"),
    measured_better("l1qsm.te-first", &[inv(&[Choice("l1qsm")]), MULTI_ECHO],
        "inversion.l1qsm.te", &[Choice("first")],
        "FANSI's script allows any single echo time; the first echo did better than the spacing \
         on the phantom (XSIM 0.782 vs 0.776, NRMSE 37.4 vs 40.2 %)",
        "FANSI-toolbox script_DICOM_Philips.m L364; QSM.rs L1QsmParams::te docs (#276)"),
    discourage("whqsm.flat-weight-te-spacing",
        &[inv(&[Choice("whqsm")]), method("inversion.whqsm.weight", &[Choice("flat")]), MULTI_ECHO],
        "inversion.whqsm.te", &[Choice("spacing")], Choice("first"), Basis::Measured,
        "With the flat weight, WH-QSM at the echo spacing diverges (r 0.07 on the phantom): its \
         Newton step fails once the phase nears ±π",
        "QSM.rs WhQsmParams::te docs (#276)"),
    measured_better("whqsm.magnitude-weight-te-spacing",
        &[inv(&[Choice("whqsm")]),
          method("inversion.whqsm.weight", &[Choice("max-normalised"), Choice("in-vivo")]), MULTI_ECHO],
        "inversion.whqsm.te", &[Choice("spacing")],
        "With a magnitude weight, the echo spacing did better than the first echo on the phantom \
         (XSIM 0.673 vs 0.637); FANSI's script converts at a TE difference with a magnitude-derived weight",
        "QSM.rs WhQsmParams::te docs (#276); FANSI-toolbox script_DICOM_Philips.m L364, L436"),
    reference("star.te-sum", &[inv(&[Choice("star")]), MULTI_ECHO], "inversion.star.te", &[Choice("sum")],
        "STI Suite's QSM_GUI gives QSM_star the summed phase with TE = ΣTE",
        "STI Suite 3.0 QSM_GUI, probed (recorded in QSM.rs StarParams::te, #276)"),
    reference("medi.te-spacing", &[inv(&[Choice("medi")]), MULTI_ECHO], "inversion.medi.te", &[Choice("spacing")],
        "MEDI_L1 converts with delta_TE = TE(2) − TE(1)",
        "MEDI toolbox 2024.11.26 MEDI_L1.m:202; Read_DICOM.m:166-168"),
    reference("tfi.te-spacing", &[inv(&[Choice("tfi")]), MULTI_ECHO], "inversion.tfi.te", &[Choice("spacing")],
        "TFI runs through MEDI_L1, which converts with delta_TE",
        "MEDI toolbox 2024.11.26 MEDI_L1.m:202 (QSM.rs TfiParams::te, #276)"),
    reference("heidi.te-last", &[inv(&[Choice("heidi")]), MULTI_ECHO], "inversion.heidi.te", &[Choice("last")],
        "The authors' HEIDI pipeline converts at the last echo time",
        "qsm_heidi phase2susceptibilityfactor.m L70-71 (QSM.rs HeidiParams::te, #276)"),
    // ---------------------------------------------------------------- χ-separation inputs
    reference("chi-sep-ilsqr.qsm-ilsqr", &[sep(&[Choice("chi-sep-ilsqr")])], "inversion.algorithm",
        &[Choice("ilsqr")],
        "The χ-separation toolbox feeds it STI Suite iLSQR QSM",
        "χ-separation toolbox Chisep_script.m:142-145,170"),
    reference("r2star-qsm.qsm-medi", &[sep(&[Choice("r2star-qsm")])], "inversion.algorithm",
        &[Choice("medi")],
        "R2*-QSM's χ was reconstructed with MEDI",
        "Dimov et al. 2022 Tomography 8:1544, Methods"),
    reference("decompose.qsm-star", &[sep(&[Choice("decompose")])], "inversion.algorithm",
        &[Choice("star")],
        "DECOMPOSE's χ was reconstructed with STAR-QSM",
        "Chen et al. 2021 NeuroImage 242:118735 (DECOMPOSE), §3"),
    reference("decompose.reference-mean", &[sep(&[Choice("decompose")])], "reference",
        &[Choice("mean")],
        "DECOMPOSE's χ is referenced to the whole-brain mean",
        "Chen et al. 2021 NeuroImage 242:118735 (DECOMPOSE), §3"),
    reference("susep-net.qsm-iqsm-plus", &[sep(&[Choice("susep-net")])], "inversion.algorithm",
        &[Choice("iqsm-plus")],
        "SUSEP-Net's in-vivo QSM input is from iQSM+",
        "SUSEP-Net (arXiv:2506.13293) §3.2"),
    discourage("chi-sepnet.min-resolution",
        &[sep(&[Choice("chi-sepnet")]), fact_at_most(Fact::MinVoxelSize, 0.6)],
        "separation.algorithm", &[Choice("chi-sepnet")], Choice("chi-sep-ilsqr"), Basis::Stated,
        "χ-sepnet's authors state the networks process resolutions coarser than 0.6 mm",
        "SNU-LIST/chi-separation README; Kim et al. 2024 (arXiv:2410.12239) §4"),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::config::*;

    fn facts() -> ScanFacts {
        ScanFacts {
            field_strength: Some(3.0),
            echo_count: Some(4),
            voxel_size: Some((1.0, 1.0, 1.0)),
            dims: Some((192, 256, 160)),
            target: Target::Native,
        }
    }

    /// A configuration under test: the pipeline, the optional stages, and the scan.
    #[derive(Clone)]
    struct Case {
        cfg: QsmPipelineConfig,
        sep: Option<SeparationConfig>,
        seg: Option<SynthSegParams>,
        facts: ScanFacts,
    }

    impl Case {
        fn new(f: impl FnOnce(&mut Case)) -> Case {
            let mut c = Case { cfg: QsmPipelineConfig::default(), sep: None, seg: None, facts: facts() };
            f(&mut c);
            c
        }
        fn input(&self) -> RuleInput<'_> {
            RuleInput { pipeline: &self.cfg, separation: self.sep.as_ref(), segmentation: self.seg.as_ref() }
        }
        fn ids(&self) -> Vec<&'static str> {
            evaluate(&self.input(), &self.facts).iter().map(|f| f.rule.id).collect()
        }
    }

    fn inversion(a: InversionAlgorithm) -> impl FnOnce(&mut Case) {
        move |c| c.cfg.inversion.algorithm = a
    }
    fn separation(a: SeparationAlgorithm) -> impl FnOnce(&mut Case) {
        move |c| c.sep = Some(SeparationConfig { algorithm: a, ..Default::default() })
    }
    fn background(a: BgRemovalAlgorithm) -> impl FnOnce(&mut Case) {
        move |c| c.cfg.bg_removal.algorithm = a
    }
    fn linear_fit(c: &mut Case) {
        c.cfg.field_mapping.b0_estimation = B0EstimationMethod::LinearFit(Default::default());
    }

    /// A configuration that violates each rule, keyed by rule id.
    fn violations() -> Vec<(&'static str, Case)> {
        use BgRemovalAlgorithm as B;
        use InversionAlgorithm as I;
        use SeparationAlgorithm as S;
        vec![
            ("medi.mask-bet", Case::new(inversion(I::Medi))),
            ("tfi.mask-bet", Case::new(inversion(I::Tfi))),
            ("pdf.mask-bet", Case::new(background(B::Pdf))),
            ("amp-pe.mask-bet", Case::new(inversion(I::AmpPe))),
            ("qsmnet.mask-bet", Case::new(inversion(I::QsmnetPlus))),
            ("qsmgan.mask-bet", Case::new(inversion(I::Qsmgan))),
            ("lpcnn.mask-bet", Case::new(inversion(I::Lpcnn))),
            ("nextqsm.mask-bet", Case::new(inversion(I::Nextqsm))),
            ("iqsm.mask-bet", Case::new(inversion(I::Iqsm))),
            ("bfrnet.mask-bet", Case::new(background(B::Bfrnet))),
            ("chi-sep.mask-bet", Case::new(separation(S::ChiSepMedi))),
            ("decompose.mask-bet", Case::new(separation(S::Decompose))),
            ("mindgrab.wasm-heap", Case::new(|c| {
                c.cfg.masking.sections[0].generator = MaskOp::MindGrab(Default::default());
                c.facts.target = Target::Wasm;
            })),
            ("hd-bet.wasm-heap", Case::new(|c| {
                c.cfg.masking.sections[0].generator = MaskOp::HdBet(crate::bet::HdBetParams::default());
                c.facts.target = Target::Wasm;
            })),
            ("synthseg.wasm-windows", Case::new(|c| {
                c.seg = Some(SynthSegParams::default());
                c.facts.target = Target::Wasm;
            })),
            ("synthseg.wasm-128-window", Case::new(|c| {
                c.seg = Some(SynthSegParams {
                    inference: SynthSegInference::SlidingWindow { patch: 128, overlap: 48 },
                    ..SynthSegParams::low_memory()
                });
                c.facts.target = Target::Wasm;
            })),
            ("fansi.no-laplacian-multi-echo", Case::new(|c| {
                c.cfg.inversion.algorithm = I::Ndi;
                c.cfg.field_mapping.unwrapping_algorithm = UnwrappingAlgorithm::Laplacian;
            })),
            ("qsmnet.unwrap-laplacian", Case::new(inversion(I::Qsmnet))),
            ("autoqsm.unwrap-laplacian", Case::new(inversion(I::Autoqsm))),
            ("qsmgan.unwrap-laplacian", Case::new(inversion(I::Qsmgan))),
            ("decompose.unwrap-laplacian", Case::new(separation(S::Decompose))),
            ("chi-sep.unwrap-romeo", Case::new(|c| {
                separation(S::ChiSepIlsqr)(c);
                c.cfg.field_mapping.unwrapping_algorithm = UnwrappingAlgorithm::Laplacian;
            })),
            ("field-mapping.phase-offsets", Case::new(|c| c.cfg.field_mapping.phase_offset_removal = false)),
            ("field-mapping.bipolar-needs-three-echoes", Case::new(|c| {
                c.cfg.field_mapping.bipolar_correction = true;
                c.facts.echo_count = Some(2);
            })),
            ("field-mapping.bipolar-not-with-laplacian", Case::new(|c| {
                c.cfg.field_mapping.bipolar_correction = true;
                c.cfg.field_mapping.unwrapping_algorithm = UnwrappingAlgorithm::Laplacian;
            })),
            ("medi.field-complex-fit", Case::new(inversion(I::Medi))),
            ("pdf.field-complex-fit", Case::new(background(B::Pdf))),
            ("r2star-qsm.field-complex-fit", Case::new(separation(S::R2starQsm))),
            ("amp-pe.echo-combination", Case::new(|c| { inversion(I::AmpPe)(c); linear_fit(c) })),
            ("amp-pe.echo-weights", Case::new(inversion(I::AmpPe))),
            ("qsmnet.echo-combination", Case::new(|c| { inversion(I::Qsmnet)(c); linear_fit(c) })),
            ("qsmnet.echo-weights", Case::new(inversion(I::Qsmnet))),
            ("autoqsm.echo-combination", Case::new(|c| { inversion(I::Autoqsm)(c); linear_fit(c) })),
            ("autoqsm.echo-weights", Case::new(inversion(I::Autoqsm))),
            ("chi-sep.echo-combination", Case::new(|c| { separation(S::ChiSepNet)(c); linear_fit(c) })),
            ("chi-sep.echo-weights", Case::new(separation(S::ChiSepIlsqr))),
            ("hdqsm.bg-pdf", Case::new(inversion(I::Hdqsm))),
            ("amp-pe.bg-pdf", Case::new(inversion(I::AmpPe))),
            ("lsqr.bg-sharp", Case::new(inversion(I::Heidi))),
            ("rts.bg-vsharp", Case::new(background(B::Pdf))),
            ("xqsm.bg-resharp", Case::new(inversion(I::Xqsm))),
            ("qsmnet.bg-vsharp", Case::new(|c| { inversion(I::Qsmnet)(c); background(B::Lbv)(c) })),
            ("qsmgan.bg-vsharp", Case::new(|c| { inversion(I::Qsmgan)(c); background(B::Lbv)(c) })),
            ("lpcnn.bg-vsharp", Case::new(|c| { inversion(I::Lpcnn)(c); background(B::Lbv)(c) })),
            ("chi-sep.bg-vsharp", Case::new(|c| { separation(S::ChiSepIlsqr)(c); background(B::Pdf)(c) })),
            ("r2star-qsm.bg-pdf", Case::new(separation(S::R2starQsm))),
            ("decompose.bg-vsharp", Case::new(|c| { separation(S::Decompose)(c); background(B::Pdf)(c) })),
            ("star.padding", Case::new(|c| {
                c.cfg.inversion.algorithm = I::Star;
                c.cfg.inversion.star.pad = [12; 3];
            })),
            ("chi-sep-ilsqr.qsm-ilsqr", Case::new(separation(S::ChiSepIlsqr))),
            ("medi.bg-pdf", Case::new(inversion(I::Medi))),
            ("sti.vsharp-min-radius", Case::new(|c| { inversion(I::Ilsqr)(c); c.cfg.bg_removal.vsharp.min_radius = 1.0 })),
            ("sti.vsharp-radius-step", Case::new(|c| { inversion(I::Star)(c); c.cfg.bg_removal.vsharp.radius_step = 2.0 })),
            ("sti.ilsqr-iterations", Case::new(|c| { inversion(I::Ilsqr)(c); c.cfg.inversion.ilsqr.max_iter = 100 })),
            ("sti.ilsqr-pad", Case::new(|c| { inversion(I::Ilsqr)(c); c.cfg.inversion.ilsqr.pad_mm = 6.0 })),
            ("field-mapping.phase-offset-sigma", Case::new(|c| c.cfg.field_mapping.phase_offset_sigma_mm = [10.0, 10.0, 5.0])),
            ("medi-msmv.three-echoes", Case::new(|c| {
                inversion(I::Medi)(c);
                c.cfg.inversion.medi.msmv = Some(Default::default());
                c.facts.echo_count = Some(2);
            })),
            ("medi-msmv.bg-removal", Case::new(|c| {
                inversion(I::Medi)(c);
                c.cfg.inversion.medi.msmv = Some(Default::default());
                background(B::Sharp)(c);
            })),
            ("medi-msmv.prefilter", Case::new(|c| {
                inversion(I::Medi)(c);
                c.cfg.inversion.medi.msmv =
                    Some(crate::inversion::MsmvParams { prefilter: false, ..Default::default() });
            })),
            ("fansi.te-spacing", Case::new(|c| { inversion(I::FansiTgv)(c); c.cfg.inversion.fansi.te = crate::EchoTime::First })),
            ("ndi.te-first", Case::new(|c| { inversion(I::Ndi)(c); c.cfg.inversion.ndi.te = crate::EchoTime::Spacing })),
            ("l1qsm.te-first", Case::new(|c| { inversion(I::L1qsm)(c); c.cfg.inversion.l1qsm.te = crate::EchoTime::Sum })),
            ("whqsm.flat-weight-te-spacing", Case::new(|c| { inversion(I::Whqsm)(c); c.cfg.inversion.whqsm.te = crate::EchoTime::Spacing })),
            ("whqsm.magnitude-weight-te-spacing", Case::new(|c| {
                inversion(I::Whqsm)(c);
                c.cfg.inversion.whqsm.weight = crate::inversion::WhQsmWeight::InVivo;
            })),
            ("star.te-sum", Case::new(|c| { inversion(I::Star)(c); c.cfg.inversion.star.te = crate::EchoTime::First })),
            ("medi.te-spacing", Case::new(|c| { inversion(I::Medi)(c); c.cfg.inversion.medi.te = crate::EchoTime::First })),
            ("tfi.te-spacing", Case::new(|c| { inversion(I::Tfi)(c); c.cfg.inversion.tfi.te = crate::EchoTime::Last })),
            ("heidi.te-last", Case::new(|c| { inversion(I::Heidi)(c); c.cfg.inversion.heidi.te = crate::EchoTime::First })),
            ("sti.bg-vsharp", Case::new(|c| { inversion(I::Star)(c); background(B::Pdf)(c) })),
            ("sti.vsharp-no-deconvolve", Case::new(|c| { inversion(I::Ilsqr)(c); c.cfg.bg_removal.vsharp.deconvolve = true })),
            ("sti.vsharp-radius", Case::new(|c| { inversion(I::Ilsqr)(c); c.cfg.bg_removal.vsharp.max_radius = 8.0 })),
            ("sti.vsharp-grid", Case::new(|c| { inversion(I::Star)(c); c.cfg.bg_removal.vsharp.grid = crate::bgremove::VsharpGrid::Native })),
            ("resharp.radius", Case::new(|c| { background(B::Resharp)(c); c.cfg.bg_removal.resharp.radius = 6.0 })),
            ("resharp.iterations", Case::new(|c| { background(B::Resharp)(c); c.cfg.bg_removal.resharp.max_iter = 30 })),
            ("sti.unwrap-laplacian", Case::new(inversion(I::Ilsqr))),
            ("sti.laplacian-solver", Case::new(|c| {
                inversion(I::Star)(c);
                c.cfg.field_mapping.unwrapping_algorithm = UnwrappingAlgorithm::Laplacian;
            })),
            ("qsmart.unwrap-laplacian", Case::new(inversion(I::Qsmart))),
            ("qsmart.laplacian-solver", Case::new(|c| {
                inversion(I::Qsmart)(c);
                c.cfg.field_mapping.unwrapping_algorithm = UnwrappingAlgorithm::Laplacian;
            })),
            ("qsmart.field-echofit", Case::new(inversion(I::Qsmart))),
            ("qsmart-echofit.unwrap-laplacian", Case::new(|c| {
                c.cfg.field_mapping.b0_estimation =
                    B0EstimationMethod::QsmartEchofit { fit_threshold: Default::default() };
            })),
            ("sti.echo-combination", Case::new(|c| { inversion(I::Star)(c); linear_fit(c) })),
            ("sti.echo-weights", Case::new(inversion(I::Ilsqr))),
            ("qsmart.mask-bet", Case::new(inversion(I::Qsmart))),
            ("qsmart.inner-inversion", Case::new(|c| {
                inversion(I::Qsmart)(c);
                c.cfg.inversion.qsmart.inversion = I::Star;
            })),
            ("qsmart.ilsqr-iterations", Case::new(|c| { inversion(I::Qsmart)(c); c.cfg.inversion.qsmart.ilsqr_max_iter = 100 })),
            ("qsmart.ilsqr-pad", Case::new(|c| { inversion(I::Qsmart)(c); c.cfg.inversion.qsmart.ilsqr_pad_mm = 12.0 })),
            ("r2star-qsm.qsm-medi", Case::new(separation(S::R2starQsm))),
            ("decompose.qsm-star", Case::new(separation(S::Decompose))),
            ("decompose.reference-mean", Case::new(|c| {
                separation(S::Decompose)(c);
                c.cfg.reference = QsmReference::None;
            })),
            ("susep-net.qsm-iqsm-plus", Case::new(separation(S::SusepNet))),
            ("chi-sepnet.min-resolution", Case::new(|c| {
                separation(S::ChiSepNet)(c);
                c.facts.voxel_size = Some((0.5, 0.5, 0.5));
            })),
        ]
    }

    /// Every path a rule names is one [`current`] reads, every default passes its own test, and
    /// every id is unique.
    #[test]
    fn rules_are_well_formed() {
        let cfg = QsmPipelineConfig::default();
        let input = RuleInput::pipeline(&cfg);
        let facts = ScanFacts::default();
        let known = |p: &str| SETTINGS.iter().any(|(s, _)| *s == p);
        let mut ids = std::collections::HashSet::new();
        for r in RULES {
            assert!(ids.insert(r.id), "duplicate rule id {}", r.id);
            assert!(!r.when.is_empty(), "{}: no trigger", r.id);
            assert!(matches!(r.trigger(), Condition::Setting { .. }), "{}: the trigger is a choice", r.id);
            assert!(known(r.setting), "{}: {} not in SETTINGS", r.id, r.setting);
            for c in r.when {
                if let Condition::Setting { setting, .. } = c {
                    assert!(known(setting), "{}: {setting} not in SETTINGS", r.id);
                }
            }
            let d = match r.default {
                Value::Bool(b) => Current::Bool(b),
                Value::Number(x) => Current::Number(x),
                Value::Choice(s) => Current::Choice(s),
                Value::Vector(v) => Current::Vector(v.to_vec()),
            };
            assert!(r.expect.passes(&d), "{}: default does not pass its own test", r.id);
            assert!(!r.reason.is_empty() && !r.source.is_empty(), "{}: reason and source", r.id);
        }
        for (p, _) in SETTINGS {
            assert!(current(&input, &facts, p).is_some(), "{p} in SETTINGS but unreadable");
        }
    }

    /// The default pipeline (with SynthSeg's defaults, natively) raises nothing, with or without
    /// scan facts. χ-separation is not part of the default pipeline: its default method's rules
    /// fire on the default QSM settings, because the χ-separation toolbox's pipeline differs
    /// from them.
    #[test]
    fn default_config_raises_nothing() {
        for facts in [ScanFacts::default(), facts()] {
            let c = Case { seg: Some(SynthSegParams::default()), facts, ..Case::new(|_| {}) };
            assert!(c.ids().is_empty(), "{:?}", c.ids());
        }
    }

    /// Each rule fires on a configuration that violates it, and every rule has such a case.
    #[test]
    fn each_rule_fires_on_a_violation() {
        let cases = violations();
        for r in RULES {
            assert!(cases.iter().any(|(id, _)| *id == r.id), "{}: no violating case", r.id);
        }
        for (id, case) in &cases {
            assert!(RULES.iter().any(|r| r.id == *id), "{id}: not a rule");
            assert!(case.ids().contains(id), "{id} did not fire: {:?}", case.ids());
        }
    }

    /// Precedence, and the basis matching each strength.
    #[test]
    fn strengths_order_and_rest_on_their_basis() {
        use Strength::*;
        assert!(Required > Recommended && Recommended > Discouraged && Discouraged > Reference);
        for r in RULES {
            match r.strength {
                Reference => assert_eq!(r.basis, Basis::ReferenceChoice, "{}", r.id),
                Recommended => assert!(matches!(r.basis, Basis::Stated | Basis::Measured), "{}", r.id),
                Required | Discouraged => assert_ne!(r.basis, Basis::ReferenceChoice, "{}", r.id),
            }
        }
    }

    /// Scan-dependent rules stay silent until the scan is known.
    #[test]
    fn unknown_facts_do_not_fire() {
        for (id, case) in violations() {
            let r = RULES.iter().find(|r| r.id == id).unwrap();
            let blind = Case { facts: ScanFacts { target: case.facts.target, ..Default::default() }, ..case };
            if r.needs_scan() {
                assert!(!blind.ids().contains(&id), "{id} fired without scan facts");
            }
        }
    }

    #[test]
    fn hd_bet_low_memory_resolves_it() {
        let c = Case::new(|c| {
            c.cfg.masking.sections[0].generator = MaskOp::HdBet(crate::bet::HdBetParams::low_memory());
            c.facts.target = Target::Wasm;
        });
        assert!(!c.ids().contains(&"hd-bet.wasm-heap"));
    }

    #[test]
    fn synthseg_window_scales_with_the_grid() {
        let c = Case::new(|c| {
            c.seg = Some(SynthSegParams {
                inference: SynthSegInference::SlidingWindow { patch: 128, overlap: 48 },
                ..SynthSegParams::low_memory()
            });
            c.facts.target = Target::Wasm;
            c.facts.dims = Some((184, 256, 128));
        });
        assert!(!c.ids().contains(&"synthseg.wasm-128-window"));
        let low = Case { seg: Some(SynthSegParams::low_memory()), ..c };
        assert!(low.ids().is_empty(), "{:?}", low.ids());
    }

    /// Rules are scoped to their trigger: STAR's padding is not reported when STAR is not chosen,
    /// and background-removal rules are silent when the inversion skips background removal.
    #[test]
    fn rules_are_scoped() {
        let c = Case::new(|c| c.cfg.inversion.star.pad = [12; 3]);
        assert!(c.ids().is_empty(), "{:?}", c.ids());
        // AutoQSM takes the total field: LBV is never run, so no BFR rule fires for it.
        let c = Case::new(|c| {
            c.cfg.inversion.algorithm = InversionAlgorithm::Autoqsm;
            c.cfg.bg_removal.algorithm = BgRemovalAlgorithm::Lbv;
        });
        assert!(!c.ids().iter().any(|id| id.contains(".bg-")), "{:?}", c.ids());
        // Fixed per the defaults, the findings go away.
        let c = Case::new(|c| {
            c.cfg.inversion.algorithm = InversionAlgorithm::Hdqsm;
            c.cfg.bg_removal.algorithm = BgRemovalAlgorithm::Pdf;
            c.cfg.masking.sections[0].generator = MaskOp::Bet(Default::default());
            c.cfg.field_mapping.b0_estimation = B0EstimationMethod::ComplexFit(Default::default());
        });
        assert!(c.ids().is_empty(), "{:?}", c.ids());
    }

    /// The JSON form carries everything the evaluation uses.
    #[test]
    fn serialises_to_json() {
        let json = serde_json::to_value(RULES).unwrap();
        let star = json.as_array().unwrap().iter().find(|r| r["id"] == "star.padding").unwrap();
        assert_eq!(star["strength"], "reference");
        assert_eq!(star["basis"], "reference_choice");
        assert_eq!(star["setting"], "inversion.star.pad");
        assert_eq!(star["unit"], "voxels");
        assert_eq!(star["when"][0]["on"], "setting");
        assert_eq!(star["when"][0]["is"]["test"], "one_of");
        assert_eq!(star["when"][0]["is"]["values"][0], "star");
        assert_eq!(star["expect"]["test"], "one_of");
        assert_eq!(star["default"], serde_json::json!([8.0, 8.0, 20.0]));
        let wasm = json.as_array().unwrap().iter().find(|r| r["id"] == "mindgrab.wasm-heap").unwrap();
        assert_eq!(wasm["when"][1]["on"], "fact");
        assert_eq!(wasm["when"][1]["fact"], "target");

        let c = Case::new(|c| {
            c.cfg.inversion.algorithm = InversionAlgorithm::Star;
            c.cfg.inversion.star.pad = [12; 3];
        });
        let j = serde_json::to_value(evaluate(&c.input(), &c.facts)).unwrap();
        let pad = j.as_array().unwrap().iter().find(|f| f["id"] == "star.padding").unwrap();
        assert_eq!(pad["current"], serde_json::json!([[12.0, 12.0, 12.0]]));
        assert_eq!(pad["default"], serde_json::json!([8.0, 8.0, 20.0]));
    }
}

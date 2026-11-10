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
//! - **Two kinds of rule** ([`Rule::kind`], a [`Kind`]; decided 2026-11-04). A host must be able
//!   to tell "this is what the method's authors did" from "this is better, for this reason",
//!   because only the second is advice:
//!   - [`Kind::Reference`], **the original authors' setting**: their code, paper or README does
//!     or instructs it, and nobody has shown that it is needed or better. Following it reproduces
//!     their pipeline; departing from it is a legitimate choice. E.g. QSMART's `bet2 -f 0.2` in
//!     `brainmask.m`, MEDI's region-growing unwrapper after its complex fit, a network trained on
//!     V-SHARP fields, or a toolbox default such as `Chisep_script.m`'s Tukey window, even where
//!     the code's comment calls it a "Recommendation".
//!   - [`Kind::Reasoned`], **a reasoned recommendation**: there is a stated reason why the
//!     setting is needed or better (the method cannot run, a combination is known to fail, the
//!     authors explain why, or QSM.rs measured it), and the reason is in [`Rule::reason`]. E.g.
//!     HEIDI's LSQR seed (the cone fill keeps everything outside the cone), PDF's `n_cg` (measured).
//!
//!   The kind is fixed by the [`Strength`]: [`Strength::Reference`] is the reference kind, every
//!   other strength is reasoned. It is one field rather than two so that the two cannot
//!   disagree. The JSON form carries both (`"kind"` and `"strength"`).
//! - **Strength**, and the basis it rests on ([`Rule::basis`], a [`Basis`]):
//!   - [`Strength::Required`]: the method cannot run correctly otherwise (e.g. it does not fit
//!     the heap). Usually measured.
//!   - [`Strength::Recommended`]: the authors say why it is needed or better (their paper, README
//!     or a comment in their code gives the reason; [`Basis::Stated`]), or QSM.rs measured a
//!     benefit and recorded it in the method's docs ([`Basis::Measured`]).
//!   - [`Strength::Discouraged`]: a combination known to fail, stated by the authors or measured
//!     (or a setting QSM.rs ignores in that combination, [`Basis::Implementation`]).
//!   - [`Strength::Suggested`]: QSM.rs measured a benefit on some data but not all (better in
//!     vivo, slightly worse on QSM-CI's simulated head), so it is offered with its reason but
//!     not preselected ([`Basis::Measured`]; decided 2026-11-10). E.g. NDI's in-mask weight
//!     normaliser and WH-QSM's in-mask flat weight.
//!   - [`Strength::Reference`]: the original authors' setting ([`Kind::Reference`]): their code
//!     or paper simply does it ([`Basis::ReferenceChoice`]), or their README or a code comment
//!     instructs it without saying why ([`Basis::Stated`]).
//!
//!   Precedence is `Required > Recommended > Discouraged > Suggested > Reference` (`Strength: Ord`). A
//!   reason outranks a bare reference choice; a recommendation outranks a discouragement because
//!   a recommendation names the value to use, while a discouragement only names one to avoid.
//!   This module reports every finding and leaves resolving them to the host.
//! - **Intended front-end handling** (QSMxT/QSMxT#419, astewartau/qsmbly#194; 2026-11-04):
//!   - `Required`: keeps the host's "forces" behaviour; the run does not proceed as configured.
//!   - `Recommended` and `Discouraged` (reasoned): offered as a change, labelled as a
//!     recommendation with its reason, with the recommended value **preselected**; a host's
//!     "accept recommendations" applies them. A badge on the setting afterwards.
//!   - `Suggested` (reasoned): offered and labelled as a recommendation with its reason, like
//!     `Recommended`, but **not preselected** and not applied by "accept recommendations"; a host
//!     applies it only when the user picks it ([`Strength::preselected`]).
//!   - `Reference`: shown (e.g. when its method is chosen), labelled as the original authors'
//!     setting, but **never preselected** and never applied by "accept recommendations"; a host
//!     applies it only when the user picks it or asks for the authors' settings explicitly.
//!     Recorded in the run record, and listed in the methods' Note section in a softer form
//!     ("differs from the reference pipeline"), separately from departures from recommendations.
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

/// How strongly a rule applies. Ordered `Reference < Suggested < Discouraged < Recommended < Required`, so
/// `max` is the finding that takes precedence on a setting (see the module docs, *Strength*).
#[derive(Serialize, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Strength {
    /// The original authors' setting: what the method's reference pipeline does or instructs,
    /// where nobody has said why it is needed or better ([`Kind::Reference`]). Never preselected.
    Reference,
    /// Better for a measured reason on some data but not all (e.g. better in vivo, slightly worse
    /// on QSM-CI's simulated head): offered with its reason, but never preselected
    /// ([`Strength::preselected`]).
    Suggested,
    /// A combination known to fail (measured, or stated by the authors with the reason).
    Discouraged,
    /// Better for a stated reason: the authors say why (paper, README or code comment), or
    /// QSM.rs measured a benefit and recorded it in the method's docs.
    Recommended,
    /// The method cannot run correctly otherwise.
    Required,
}

impl Strength {
    /// The kind of suggestion a rule of this strength is: [`Kind::Reference`] for
    /// [`Strength::Reference`], [`Kind::Reasoned`] for every other.
    pub const fn kind(self) -> Kind {
        match self {
            Strength::Reference => Kind::Reference,
            Strength::Suggested | Strength::Discouraged | Strength::Recommended | Strength::Required => {
                Kind::Reasoned
            }
        }
    }

    /// Whether a host offers this rule's change preselected, and its "accept recommendations"
    /// applies it: every reasoned strength but [`Strength::Suggested`].
    pub const fn preselected(self) -> bool {
        matches!(self, Strength::Discouraged | Strength::Recommended | Strength::Required)
    }
}

/// Whether a rule is the original authors' setting or a reasoned recommendation (see the module
/// docs, *Two kinds of rule*). Fixed by the rule's [`Strength`] ([`Strength::kind`]).
#[derive(Serialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// The original authors' setting: their code, paper or README does or instructs it, without
    /// a reason why it is needed or better. A host shows it, labelled as the authors' setting,
    /// and never preselects or bulk-applies it.
    Reference,
    /// Needed or better for the reason in [`Rule::reason`]: a hard requirement, a known failure,
    /// the authors' stated reason, or a QSM.rs measurement. A host preselects it, except a
    /// [`Strength::Suggested`] one ([`Strength::preselected`]).
    Reasoned,
}

/// What a rule rests on.
#[derive(Serialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Basis {
    /// The authors say so: their paper, README or a comment in their code. With
    /// [`Strength::Reference`], they instruct it without saying why; otherwise they give the
    /// reason.
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
    /// The scanner's manufacturer ([`Manufacturer::id`]: `"siemens"`, `"ge"`, `"philips"` or
    /// `"other"`), from the BIDS sidecar's `Manufacturer` (DICOM (0008,0070)).
    Manufacturer,
}

/// A scanner manufacturer, as far as a rule distinguishes them.
#[derive(Serialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Manufacturer {
    Siemens,
    Ge,
    Philips,
    /// Any other (Canon, Toshiba, United Imaging, Hitachi, Bruker, ...).
    Other,
}

impl Manufacturer {
    /// The manufacturer a BIDS `Manufacturer` value (DICOM (0008,0070)) names, as dcm2niix and the
    /// scanners spell it: `"Siemens"`, `"SIEMENS"`, `"Siemens Healthineers"`; `"GE"`,
    /// `"GE MEDICAL SYSTEMS"`, `"General Electric"`; `"Philips"`, `"Philips Medical Systems"`,
    /// `"Philips Healthcare"`. Case and surrounding spaces do not matter; anything else is
    /// [`Manufacturer::Other`].
    pub fn from_name(name: &str) -> Self {
        let m = name.trim().to_uppercase();
        if m.starts_with("SIEMENS") {
            Manufacturer::Siemens
        } else if m == "GE" || m.starts_with("GE ") || m.contains("GENERAL ELECTRIC") {
            Manufacturer::Ge
        } else if m.starts_with("PHILIPS") {
            Manufacturer::Philips
        } else {
            Manufacturer::Other
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            Manufacturer::Siemens => "siemens",
            Manufacturer::Ge => "ge",
            Manufacturer::Philips => "philips",
            Manufacturer::Other => "other",
        }
    }
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
    /// No value of a setting passes `is`: the [`Condition::Setting`] it names does not hold. A
    /// setting not in play has no value, so this holds. It scopes a rule out of a choice that has
    /// its own value for the setting: STI Suite's field-mapping rules do not apply under the
    /// χ-separation toolbox, which runs STI Suite's iLSQR but maps the field its own way
    /// (`sti.unwrap-laplacian`). Never a rule's trigger.
    Unless { setting: &'static str, is: Test },
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
///
/// Serialised with every field, its [`Kind`] (`"kind"`) and whether a host preselects it
/// (`"preselected"`), both from the strength.
#[derive(Clone, Copy, Debug, PartialEq)]
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

    /// The original authors' setting, or a reasoned recommendation ([`Strength::kind`]).
    pub fn kind(&self) -> Kind {
        self.strength.kind()
    }

    /// Whether the rule can only fire once the scan is known.
    pub fn needs_scan(&self) -> bool {
        self.when.iter().any(|c| matches!(c, Condition::Fact { fact, .. } if fact.needs_scan()))
    }
}

impl Serialize for Rule {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut m = s.serialize_struct("Rule", 12)?;
        m.serialize_field("id", self.id)?;
        m.serialize_field("when", self.when)?;
        m.serialize_field("setting", self.setting)?;
        m.serialize_field("expect", &self.expect)?;
        m.serialize_field("default", &self.default)?;
        m.serialize_field("unit", &self.unit)?;
        m.serialize_field("kind", &self.kind())?;
        m.serialize_field("strength", &self.strength)?;
        m.serialize_field("preselected", &self.strength.preselected())?;
        m.serialize_field("basis", &self.basis)?;
        m.serialize_field("reason", self.reason)?;
        m.serialize_field("source", self.source)?;
        m.end()
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
    /// The scanner's manufacturer, from the sidecar ([`Manufacturer::from_name`]).
    pub manufacturer: Option<Manufacturer>,
    pub target: Target,
}

impl ScanFacts {
    /// Every fact of a scan's metadata, on `target`. [`ScanMetadata`](super::config::ScanMetadata)
    /// does not carry the manufacturer; a host that read it sets [`Self::manufacturer`].
    pub fn from_metadata(m: &super::config::ScanMetadata, target: Target) -> Self {
        Self {
            field_strength: Some(m.field_strength),
            echo_count: Some(m.echo_times.len()),
            voxel_size: Some(m.voxel_size),
            dims: Some(m.dims),
            manufacturer: None,
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
            Fact::Manufacturer => self.manufacturer.map(|m| Current::Choice(m.id())),
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
    ("masking.input", "Each mask section's input image (MaskingInput::id): magnitude-first, magnitude (RSS over echoes), magnitude-last or phase-quality"),
    ("masking.inhomogeneity_correction", "The host's correction of the magnitude before masking (InhomogeneityCorrection::id): none, makehomogeneous or n4"),
    ("masking.bet.fractional_intensity", "Each BET step's fractional intensity threshold (BetParams::fractional_intensity)"),
    ("masking.erosion", "Each mask section's erosion: the voxels its Erode refinements take off in total (0 = none)"),
    ("masking.hd_bet.inference", "Each HD-BET step's inference: full-volume, sliding-window (HdBetInference::SLIDING_WINDOW), low-memory (HdBetInference::LOW_MEMORY) or custom-window"),
    ("field_mapping.unwrapping_algorithm", "Phase unwrapping (UnwrappingAlgorithm::id)"),
    ("field_mapping.b0_estimation", "Echo combination (B0EstimationMethod::id)"),
    ("field_mapping.b0_estimation.weighting", "Weighted average's per-echo weight (B0WeightType, by `weight_id`); only with weighted-avg"),
    ("field_mapping.b0_estimation.qsmart_fit_threshold", "QSMART echo fit's residual threshold for R_0 (QsmartFitThreshold): absolute-40 (QSMART_toolbox_v1.0), absolute (another value), percentile-99 (the October 2020 demo) or percentile (another); only with qsmart-echofit"),
    ("field_mapping.phase_offset_removal", "MCPC-3D-S phase-offset removal before unwrapping (bool)"),
    ("field_mapping.phase_offset_sigma_mm", "MCPC-3D-S smoothing sigma per axis, mm; only with phase-offset removal"),
    ("field_mapping.bipolar_correction", "Bipolar readout correction (bool)"),
    ("field_mapping.laplacian_solver", "Laplacian unwrapping's solver: dct, fft-12 (LaplacianSolver::FFT_DEFAULT_PAD), fft (other padding) or mirror; only with Laplacian unwrapping"),
    ("field_mapping.laplacian_unmasked", "Laplacian unwrapping on the raw phase over the whole volume, not the phase zeroed outside the mask (bool); only with Laplacian unwrapping"),
    ("bg_removal.algorithm", "Background field removal (BgRemovalAlgorithm::id)"),
    ("bg_removal.vsharp.deconvolve", "V-SHARP's TSVD deconvolution (bool); only with V-SHARP"),
    ("bg_removal.vsharp.max_radius", "V-SHARP's largest kernel radius, mm; only with V-SHARP"),
    ("bg_removal.vsharp.min_radius", "V-SHARP's smallest kernel radius, mm; only with V-SHARP"),
    ("bg_removal.vsharp.radius_step", "V-SHARP's step between kernel radii, mm; only with V-SHARP"),
    ("bg_removal.vsharp.grid", "V-SHARP's working grid: native, isotropic-finest, isotropic-1mm or isotropic (other spacing); only with V-SHARP"),
    ("bg_removal.ismv.variant", "iSMV's implementation: wen2014 or qsmjl (IsmvVariant::id); only with iSMV"),
    ("bg_removal.ismv.tol", "iSMV's stopping tolerance (IsmvParams::tol); only with iSMV"),
    ("bg_removal.ismv.max_iter", "iSMV's iteration cap (IsmvParams::max_iter); no cap reads as +∞ (null in JSON); only with iSMV"),
    ("bg_removal.resharp.radius", "RESHARP's kernel radius, mm; only with RESHARP"),
    ("bg_removal.resharp.max_iter", "RESHARP's cgs iteration cap; only with RESHARP"),
    ("bg_removal.pdf.n_cg", "PDF's CG iteration cap (PdfParams::n_cg); only with PDF"),
    ("bg_removal.iharperella.rim", "iHARPERELLA's boundary-rim rule: sti or whole (HarperellaRim::id); only with iHARPERELLA"),
    ("bg_removal.lbv.tol", "LBV's coarsest-grid stopping tolerance, radians over the echo spacing (LbvParams::tol); only with LBV"),
    ("inversion.algorithm", "Dipole inversion (InversionAlgorithm::id)"),
    ("inversion.star.pad", "STAR-QSM padding per side, voxels (StarParams::pad)"),
    ("inversion.ilsqr.max_iter", "iLSQR's LSQR iteration cap (IlsqrParams::max_iter); only with iLSQR"),
    ("inversion.ilsqr.pad_mm", "iLSQR's padding, mm (IlsqrParams::pad_mm); only with iLSQR"),
    ("inversion.ilsqr.precision", "iLSQR's arithmetic: double or single (IlsqrPrecision); with iLSQR, or QSMART inverting with iLSQR"),
    ("inversion.whqsm.weight", "WH-QSM's data weight: flat, flat-in-mask, max-normalised or in-vivo (WhQsmWeight); only with WH-QSM"),
    ("inversion.ndi.weight_norm", "NDI's magnitude-weight normaliser: global-max or mask-p999 (NdiWeightNorm::id); only with NDI"),
    ("inversion.star.te", "Echo time STAR-QSM converts at (EchoTime: first, last, spacing, sum, seconds); only with STAR"),
    ("inversion.fansi.te", "Echo time FANSI nlTV/nlTGV convert at (EchoTime); only with fansi or fansi-tgv"),
    ("inversion.ndi.te", "Echo time NDI converts at (EchoTime); only with NDI"),
    ("inversion.l1qsm.te", "Echo time L1-QSM converts at (EchoTime); only with L1-QSM"),
    ("inversion.whqsm.te", "Echo time WH-QSM converts at (EchoTime); only with WH-QSM"),
    ("inversion.medi.msmv", "MEDI-mSMV: MediParams::msmv is set (bool); only with MEDI"),
    ("inversion.medi.msmv.prefilter", "MEDI-mSMV's SMV prefilter (MsmvParams::prefilter, bool); only with MEDI-mSMV"),
    ("inversion.medi.te", "Echo time MEDI converts at (EchoTime); only with MEDI"),
    ("inversion.tfi.te", "Echo time TFI converts at (EchoTime); only with TFI"),
    ("inversion.apply_heidi", "HEIDI's cone fill after the inversion (InversionConfig::apply_heidi, bool)"),
    ("inversion.heidi.cone_threshold", "HEIDI's cone threshold, |D| (HeidiParams::cone_threshold); only with HEIDI"),
    ("inversion.heidi.te", "Echo time HEIDI converts at (EchoTime); only with HEIDI"),
    ("inversion.heidi.denoise_itk", "HEIDI's anisotropic diffusion in ITK's float arithmetic (AnisotropicDiffusionParams::reference_itk, bool); only with HEIDI's denoising on"),
    ("inversion.tkd.threshold", "TKD's truncation threshold, |D| (TkdParams::threshold); only with TKD"),
    ("inversion.tsvd.threshold", "TSVD's truncation threshold, |D| (TkdParams::threshold); only with TSVD"),
    ("inversion.qsmart.inversion", "QSMART's inner inversion (InversionAlgorithm::id); only with QSMART"),
    ("inversion.qsmart.ilsqr_max_iter", "QSMART's iLSQR iteration count; only with QSMART"),
    ("inversion.qsmart.ilsqr_pad_mm", "QSMART's iLSQR padding, mm; only with QSMART"),
    ("inversion.qsmart.offset_grid", "QSMART's offset-adjustment dipole grid: fft or qsmart-reference (QsmartOffsetGrid::id); only with QSMART"),
    ("reference", "QSM referencing (QsmReference: mean or none)"),
    ("qsm_per_echo", "A QSM per echo, and their mean, instead of one QSM from the echo-combined field (QsmPipelineConfig::qsm_per_echo, bool)"),
    ("tukey_window", "Taper ratio of the χ-separation toolbox's k-space Tukey window on the complex echoes, 0 = none (QsmPipelineConfig::tukey_window)"),
    ("separation.algorithm", "χ-separation (SeparationAlgorithm::id)"),
    ("separation.chi_sepnet.route", "χ-sepnet's input route: toolbox or native (ChiSepNetRoute::id); only with χ-sepnet"),
    ("separation.decompose.unwrap_phase", "DECOMPOSE's phase unwrapping along the echoes (DecomposeParams::unwrap_phase, bool); only with DECOMPOSE"),
    ("separation.decompose.fit", "DECOMPOSE's fit (DecomposeParams::fit_id): reference-code (Tim Ho's script: lsqcurvefit for every stage, global-max scale), paper (Chen et al. 2021's solvers on per-voxel S(0) = 1 data) or custom; only with DECOMPOSE"),
    ("separation.decompose.chi_bound", "DECOMPOSE's bound on |χ| in the fit (DecomposeChiBound::id): ppm (a fixed bound, the authors' 0.5) or echo-spacing (π / ((2/3)·γ·B0·ΔTE_max), computed from the scan); only with DECOMPOSE"),
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
///
/// A per-echo reconstruction ([`QsmPipelineConfig::qsm_per_echo`]) runs every stage on one echo
/// at a time, so its inversion's input is the one-echo input, and the settings that combine or
/// compare echoes (the echo combination and its weighting, phase-offset removal and its sigma,
/// bipolar correction) are not in play.
pub fn current(input: &RuleInput, facts: &ScanFacts, setting: &str) -> Option<Vec<Current>> {
    let c = input.pipeline;
    let inv_input = c.inversion.input(if c.qsm_per_echo { 1 } else { facts.echo_count.unwrap_or(1) });
    // Field mapping that combines or compares echoes: not on a per-echo reconstruction.
    let combines = inv_input.needs_field_mapping() && !c.qsm_per_echo;
    let one = |v: Current| vec![v];
    let gated = |on: bool, v: Current| if on { vec![v] } else { vec![] };
    let bg = |a: super::config::BgRemovalAlgorithm| inv_input.needs_bg_removal() && c.bg_removal.algorithm == a;
    let vsharp = bg(super::config::BgRemovalAlgorithm::Vsharp);
    let resharp = bg(super::config::BgRemovalAlgorithm::Resharp);
    let ismv = bg(super::config::BgRemovalAlgorithm::Ismv);
    let pdf = bg(super::config::BgRemovalAlgorithm::Pdf);
    let lbv = bg(super::config::BgRemovalAlgorithm::Lbv);
    use super::config::InversionAlgorithm as A;
    let alg = |a: &[A]| a.contains(&c.inversion.algorithm);
    let qsmart = alg(&[A::Qsmart]);
    Some(match setting {
        "masking.generator" => {
            c.masking.sections.iter().map(|s| Current::Choice(s.generator.id())).collect()
        }
        "masking.input" => c.masking.sections.iter().map(|s| Current::Choice(s.input.id())).collect(),
        "masking.inhomogeneity_correction" => one(Current::Choice(c.masking.inhomogeneity_correction.id())),
        "masking.bet.fractional_intensity" => c
            .masking
            .sections
            .iter()
            .flat_map(|s| std::iter::once(&s.generator).chain(&s.refinements))
            .filter_map(|op| match op {
                super::config::MaskOp::Bet(p) => Some(Current::Number(p.fractional_intensity)),
                _ => None,
            })
            .collect(),
        "masking.erosion" => c
            .masking
            .sections
            .iter()
            .map(|s| {
                let n: usize = s
                    .refinements
                    .iter()
                    .map(|op| match op {
                        super::config::MaskOp::Erode { iterations } => *iterations,
                        _ => 0,
                    })
                    .sum();
                Current::Number(n as f64)
            })
            .collect(),
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
        "field_mapping.b0_estimation" => gated(combines, Current::Choice(c.field_mapping.b0_estimation.id())),
        "field_mapping.b0_estimation.weighting" => match &c.field_mapping.b0_estimation {
            super::config::B0EstimationMethod::WeightedAvg { weighting } if combines => {
                one(Current::Choice(weight_id(weighting)))
            }
            _ => vec![],
        },
        "field_mapping.b0_estimation.qsmart_fit_threshold" => match &c.field_mapping.b0_estimation {
            super::config::B0EstimationMethod::QsmartEchofit { fit_threshold } if combines => {
                use crate::utils::QsmartFitThreshold as T;
                one(Current::Choice(match *fit_threshold {
                    T::Absolute(t) if t == crate::utils::QSMART_FIT_THRESHOLD => "absolute-40",
                    T::Absolute(_) => "absolute",
                    T::Percentile(p) if p == crate::utils::QSMART_FIT_PERCENTILE => "percentile-99",
                    T::Percentile(_) => "percentile",
                }))
            }
            _ => vec![],
        },
        "field_mapping.phase_offset_removal" => gated(combines, Current::Bool(c.field_mapping.phase_offset_removal)),
        "field_mapping.bipolar_correction" => gated(combines, Current::Bool(c.field_mapping.bipolar_correction)),
        "qsm_per_echo" => one(Current::Bool(c.qsm_per_echo)),
        "tukey_window" => one(Current::Number(c.tukey_window)),
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
        "field_mapping.laplacian_unmasked" => gated(
            inv_input.needs_field_mapping()
                && c.field_mapping.unwrapping_algorithm == super::config::UnwrappingAlgorithm::Laplacian,
            Current::Bool(c.field_mapping.laplacian_unmasked),
        ),
        "bg_removal.vsharp.deconvolve" => gated(vsharp, Current::Bool(c.bg_removal.vsharp.deconvolve)),
        "bg_removal.vsharp.max_radius" => gated(vsharp, Current::Number(c.bg_removal.vsharp.max_radius)),
        "bg_removal.vsharp.min_radius" => gated(vsharp, Current::Number(c.bg_removal.vsharp.min_radius)),
        "bg_removal.vsharp.radius_step" => gated(vsharp, Current::Number(c.bg_removal.vsharp.radius_step)),
        "field_mapping.phase_offset_sigma_mm" => gated(
            combines && c.field_mapping.phase_offset_removal,
            Current::Vector(c.field_mapping.phase_offset_sigma_mm.to_vec()),
        ),
        "inversion.ilsqr.max_iter" => gated(alg(&[A::Ilsqr]), Current::Number(c.inversion.ilsqr.max_iter as f64)),
        "inversion.ilsqr.pad_mm" => gated(alg(&[A::Ilsqr]), Current::Number(c.inversion.ilsqr.pad_mm)),
        // QSMART's two stages take their iLSQR precision from `inversion.ilsqr` (`run_qsmart`).
        "inversion.ilsqr.precision" => gated(
            alg(&[A::Ilsqr]) || (qsmart && c.inversion.qsmart.inversion == A::Ilsqr),
            Current::Choice(match c.inversion.ilsqr.precision {
                crate::inversion::IlsqrPrecision::Double => "double",
                crate::inversion::IlsqrPrecision::Single => "single",
            }),
        ),
        "bg_removal.vsharp.grid" => gated(
            vsharp,
            Current::Choice(match c.bg_removal.vsharp.grid {
                crate::bgremove::VsharpGrid::Native => "native",
                crate::bgremove::VsharpGrid::IsotropicFinest => "isotropic-finest",
                crate::bgremove::VsharpGrid::Isotropic { spacing_mm: 1.0 } => "isotropic-1mm",
                crate::bgremove::VsharpGrid::Isotropic { .. } => "isotropic",
            }),
        ),
        "bg_removal.ismv.variant" => gated(ismv, Current::Choice(c.bg_removal.ismv.variant.id())),
        "bg_removal.ismv.tol" => gated(ismv, Current::Number(c.bg_removal.ismv.tol)),
        "bg_removal.ismv.max_iter" => gated(
            ismv,
            Current::Number(c.bg_removal.ismv.max_iter.map_or(f64::INFINITY, |n| n as f64)),
        ),
        "bg_removal.resharp.radius" => gated(resharp, Current::Number(c.bg_removal.resharp.radius)),
        "bg_removal.resharp.max_iter" => gated(resharp, Current::Number(c.bg_removal.resharp.max_iter as f64)),
        "bg_removal.pdf.n_cg" => gated(pdf, Current::Number(c.bg_removal.pdf.n_cg as f64)),
        "bg_removal.lbv.tol" => gated(lbv, Current::Number(c.bg_removal.lbv.tol)),
        "bg_removal.iharperella.rim" => gated(
            bg(super::config::BgRemovalAlgorithm::Iharperella),
            Current::Choice(c.bg_removal.iharperella.rim.id()),
        ),
        "inversion.ndi.weight_norm" => gated(alg(&[A::Ndi]), Current::Choice(c.inversion.ndi.weight_norm.id())),
        "inversion.whqsm.weight" => gated(
            alg(&[A::Whqsm]),
            Current::Choice(match c.inversion.whqsm.weight {
                crate::inversion::WhQsmWeight::Flat => "flat",
                crate::inversion::WhQsmWeight::MaxNormalised => "max-normalised",
                crate::inversion::WhQsmWeight::InVivo => "in-vivo",
                crate::inversion::WhQsmWeight::FlatInMask => "flat-in-mask",
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
        "inversion.apply_heidi" => one(Current::Bool(c.inversion.apply_heidi)),
        "inversion.heidi.cone_threshold" => gated(c.inversion.apply_heidi, Current::Number(c.inversion.heidi.cone_threshold)),
        "inversion.heidi.te" => gated(c.inversion.apply_heidi, Current::Choice(echo_time_id(c.inversion.heidi.te))),
        "inversion.heidi.denoise_itk" => match &c.inversion.heidi.denoise {
            Some(d) if c.inversion.apply_heidi => one(Current::Bool(d.reference_itk)),
            _ => vec![],
        },
        "inversion.tkd.threshold" => gated(alg(&[A::Tkd]), Current::Number(c.inversion.tkd.threshold)),
        "inversion.tsvd.threshold" => gated(alg(&[A::Tsvd]), Current::Number(c.inversion.tsvd.threshold)),
        "inversion.qsmart.inversion" => gated(qsmart, Current::Choice(c.inversion.qsmart.inversion.id())),
        "inversion.qsmart.ilsqr_max_iter" => {
            gated(qsmart, Current::Number(c.inversion.qsmart.ilsqr_max_iter as f64))
        }
        "inversion.qsmart.ilsqr_pad_mm" => gated(qsmart, Current::Number(c.inversion.qsmart.ilsqr_pad_mm)),
        "inversion.qsmart.offset_grid" => gated(qsmart, Current::Choice(c.inversion.qsmart.adjust_offset_grid.id())),
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
        "separation.chi_sepnet.route" => input
            .separation
            .filter(|s| s.algorithm == super::config::SeparationAlgorithm::ChiSepNet)
            .map(|s| Current::Choice(s.chi_sepnet.route.id()))
            .into_iter()
            .collect(),
        "separation.decompose.unwrap_phase" => input
            .separation
            .filter(|s| s.algorithm == super::config::SeparationAlgorithm::Decompose)
            .map(|s| Current::Bool(s.decompose.unwrap_phase))
            .into_iter()
            .collect(),
        "separation.decompose.fit" => input
            .separation
            .filter(|s| s.algorithm == super::config::SeparationAlgorithm::Decompose)
            .map(|s| Current::Choice(s.decompose.fit_id()))
            .into_iter()
            .collect(),
        "separation.decompose.chi_bound" => input
            .separation
            .filter(|s| s.algorithm == super::config::SeparationAlgorithm::Decompose)
            .map(|s| Current::Choice(s.decompose.chi_bound.id()))
            .into_iter()
            .collect(),
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
        Condition::Unless { setting, is } => !holds(&Condition::Setting { setting, is: *is }, input, facts),
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
/// None of these χ-separation methods is chosen (χ-separation off holds too).
const fn unless_sep(values: &'static [Value]) -> Condition {
    C::Unless { setting: "separation.algorithm", is: Test::OneOf { values } }
}
const fn fact_at_least(fact: Fact, min: f64) -> Condition {
    C::Fact { fact, is: Test::Range { min: Some(min), max: None } }
}
const fn fact_at_most(fact: Fact, max: f64) -> Condition {
    C::Fact { fact, is: Test::Range { min: None, max: Some(max) } }
}
const fn manufacturer(values: &'static [Value]) -> Condition {
    C::Fact { fact: Fact::Manufacturer, is: Test::OneOf { values } }
}
/// A known manufacturer other than these (an unknown one does not hold).
const fn not_manufacturer(values: &'static [Value]) -> Condition {
    C::Fact { fact: Fact::Manufacturer, is: Test::NoneOf { values } }
}
const ON_WASM: Condition = C::Fact { fact: Fact::Target, is: Test::OneOf { values: &[Choice("wasm")] } };
const MULTI_ECHO: Condition = fact_at_least(Fact::EchoCount, 2.0);
/// The field strength from which `decompose.unwrap-7t` and `decompose.chi-bound-7t` fire: 7 T scanners, including those that
/// report 6.98 T (Siemens), and nothing at 3 T or below.
const DECOMPOSE_UNWRAP_FROM_T: f64 = 6.5;

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

/// A reasoned recommendation the authors make: they say why it is needed or better, and
/// `reason` says it.
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

/// What QSM.rs measured to be better on some data but not all ([`Strength::Suggested`]): offered
/// with its reason, never preselected.
const fn suggest(
    id: &'static str,
    when: &'static [Condition],
    setting: &'static str,
    values: &'static [Value],
    reason: &'static str,
    source: &'static str,
) -> Rule {
    one_of(id, when, setting, values, Strength::Suggested, Basis::Measured, reason, source)
}

/// The original authors' setting: what their code or paper does, without saying it is needed or
/// better.
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

/// The original authors' setting as their README or a code comment instructs it ("do X", "we
/// recommend X"), without saying why it is needed or better: still [`Kind::Reference`].
const fn reference_stated(
    id: &'static str,
    when: &'static [Condition],
    setting: &'static str,
    values: &'static [Value],
    reason: &'static str,
    source: &'static str,
) -> Rule {
    one_of(id, when, setting, values, Strength::Reference, Basis::Stated, reason, source)
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
const MEDI_MASK_SOURCE: &str = "MEDI toolbox README.m (iMag = sqrt(sum(abs(iField).^2,4)); Mask = BET(iMag, ...)), \
    l. 92 and 111 in the 2020 fork at ~/code/ref/medi; BET.m calls the bet2 mex";
const DECOMPOSE_MASK_SOURCE: &str = "Chen et al. 2021 NeuroImage 242:118477 (DECOMPOSE), §3.6 (\"The GRE \
    magnitude images of the first echo was used to mask and extract the brain tissue using ... BET\"); \
    BET's f, erosion and correction are not stated";
const CHI_SEP_MASK_SOURCE: &str = "χ-separation toolbox v1.1.3 Chisep_script.m:75 (mag = sqrt(sum(abs(mag_multi_echo).^2,4))), :82 (BET)";
/// The CG iteration count `pdf.cg-iterations` suggests (measured; see `PdfParams::n_cg`).
const PDF_SUGGESTED_N_CG: f64 = 300.0;
const PDF: &[Value] = &[Choice("pdf")];
const LBV: &[Value] = &[Choice("lbv")];
const VSHARP: &[Value] = &[Choice("vsharp")];
const LAPLACIAN: &[Value] = &[Choice("laplacian")];
/// Background removal that keeps the brain's edge, so an error in the field map's rim reaches the
/// whole brain (`laplacian.non-eroding-bfr`). iSMV (which holds its boundary band fixed) is not
/// included: on QSM-CI's simulated head Laplacian + iSMV was much worse than ROMEO + iSMV (χ XSIM
/// 0.12 against 0.24), but on 14 travelling-head scans with the fixed kernel (#436) it was as
/// reproducible (median |slope − 1| 2.8 % against 3.3 %); revisit with the full re-run.
const RIM_SENSITIVE_BFR: &[Value] = &[Choice("pdf"), Choice("lbv")];
const ROMEO: &[Value] = &[Choice("romeo")];
const COMPLEX_FIT: &[Value] = &[Choice("complex-fit")];
const WEIGHTED_AVG: &[Value] = &[Choice("weighted-avg")];
const CHI_SEP_TOOLBOX: &[Value] = &[Choice("chi-sep-ilsqr"), Choice("chi-sep-medi"), Choice("chi-sepnet")];
/// χ-sepnet's toolbox route (`ChiSepNetRoute::Toolbox`).
const CHI_SEPNET_TOOLBOX_ROUTE: Condition = method("separation.chi_sepnet.route", &[Choice("toolbox")]);
/// The χ-separation toolbox's taper ratio (`tukey_fac`): its script's value, which its comment
/// leaves for every manufacturer but Philips (0).
pub const CHISEP_TUKEY: f64 = 0.4;
const TUKEY_SOURCE: &str = "χ-separation toolbox v1.1.3 Chisep_script.m:69-73 \
    (\"tukey_fac = 0.4; % Recommendation: Siemens, GE: 0.4, Philips: 0\"); QSM.rs utils::tukey_window docs";
const TUKEY_NOT_PHILIPS_REASON: &str = "The χ-separation toolbox Tukey-windows the complex echoes in \
    k-space before everything else, with a taper ratio of 0.4; its comment sets 0 for Philips only";
const TUKEY_PHILIPS_REASON: &str = "The χ-separation toolbox uses no Tukey windowing (a taper ratio of 0) \
    for Philips data";
/// The χ-separation toolbox's iterative methods (χ-sepnet has two routes; only its toolbox route
/// is the toolbox's pipeline).
const CHI_SEP_TOOLBOX_ITERATIVE: &[Value] = &[Choice("chi-sep-ilsqr"), Choice("chi-sep-medi")];
/// QSMnetME: the χ-separation toolbox's QSM network, trained on that toolbox's pipeline.
const QSMNETME: &[Value] = &[Choice("qsmnetme")];
const STI: &[Value] = &[Choice("star"), Choice("ilsqr")];
/// HEIDI's cone fill is on (`inversion.apply_heidi`): the trigger of the `heidi.*` rules.
const HEIDI_ON: Condition = method("inversion.apply_heidi", &[Bool(true)]);
/// Inversions with no local field to fill the cone from (the total-field and phase methods).
const HEIDI_NO_LOCAL_FIELD: &[Value] = &[
    Choice("tfi"), Choice("tgv"), Choice("qsmart"), Choice("autoqsm"), Choice("nextqsm"),
    Choice("iqsm"), Choice("iqsm-plus"),
];
/// Seeds whose region outside the cone is near data-consistent (the HEIDI module docs, *Any
/// seed*), but for LSQR (the reference) and RTS (`heidi.seed-rts`).
const HEIDI_UNREGULARISED_SEEDS: &[Value] =
    &[Choice("tkd"), Choice("tsvd"), Choice("tikhonov"), Choice("ndi")];
/// Seeds whose regularisation outside the cone HEIDI would keep.
const HEIDI_REGULARISED_SEEDS: &[Value] = &[
    Choice("medi"), Choice("tv"), Choice("nltv"), Choice("fansi"), Choice("fansi-tgv"),
    Choice("l1qsm"), Choice("whqsm"), Choice("hdqsm"), Choice("amp-pe"), Choice("ilsqr"),
    Choice("star"), Choice("xqsm"), Choice("qsmnet"), Choice("qsmnet-plus"), Choice("qsmnetme"),
    Choice("qsmgan"), Choice("ir2qsm"), Choice("lpcnn"), Choice("modl-qsm"),
];
/// Upstream's error when a truncating seed's threshold is above the cone's.
const HEIDI_CONE_SOURCE: &str = "qsm_heidi computesusceptibility_heidi.m L922-923 (4592631); QSM.rs heidi.rs docs (Any seed)";

/// HEIDI's default cone threshold: the `heidi.*-cone` rules are exact for it;
/// [`heidi_seed_problem`](super::heidi_seed_problem) checks any.
const HEIDI_DEFAULT_CONE: Condition = C::Setting {
    setting: "inversion.heidi.cone_threshold",
    is: Test::Range { min: Some(0.1), max: Some(0.1) },
};

/// `heidi.<method>-cone`: a truncating seed's threshold must not be above HEIDI's cone threshold.
const fn heidi_truncation(id: &'static str, when: &'static [Condition], setting: &'static str) -> Rule {
    Rule {
        id,
        when,
        setting,
        expect: Test::Range { min: None, max: Some(0.1) },
        default: Number(0.1),
        unit: Unit::None,
        strength: Strength::Required,
        basis: Basis::Stated,
        reason: "Coefficients the seed truncated (|D| below its threshold) would be kept as data \
                 where they lie outside HEIDI's cone; upstream stops with an error",
        source: HEIDI_CONE_SOURCE,
    }
}

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
    reference_stated("qsmnet.mask-bet", &[inv(&[Choice("qsmnet"), Choice("qsmnet-plus")])], "masking.generator", BET,
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
    reference("qsmnetme.mask-bet", &[inv(QSMNETME)], "masking.generator", BET,
        "QSMnetME runs in the χ-separation toolbox, which masks with BET",
        "χ-separation toolbox Chisep_script.m:79-82; Kim et al. 2025 (χ-sepnet) §2.1.1"),
    reference("qsmart.mask-bet", &[inv(&[Choice("qsmart")])], "masking.generator", BET,
        "QSMART's authors mask with FSL bet2 -f 0.2 on the N4-corrected first-echo magnitude",
        "Yaghmaie et al. 2021 NeuroImage 231:117701 (QSMART) §3.1; QSMART brainmask.m l. 7-33"),
    reference("decompose.mask-bet", &[sep(&[Choice("decompose")])], "masking.generator", BET,
        "DECOMPOSE's pipeline masks with FSL BET",
        "Chen et al. 2021 NeuroImage 242:118477 (DECOMPOSE), §3.6"),
    // The rest of each toolbox's mask recipe: what BET reads, how that image is corrected, and
    // what is done to the mask after. QSMxT's `bet` preset (BET on the corrected RSS, then a
    // 2-voxel erosion) does none of these toolboxes' recipes.
    reference("medi.mask-input", &[inv(&[Choice("medi")])], "masking.input", &[Choice("magnitude")],
        "The MEDI toolbox runs BET on the root-sum-of-squares magnitude over echoes",
        MEDI_MASK_SOURCE),
    reference("medi.mask-no-correction", &[inv(&[Choice("medi")])], "masking.inhomogeneity_correction",
        &[Choice("none")], "The MEDI toolbox runs BET on the magnitude as read, with no bias-field correction",
        MEDI_MASK_SOURCE),
    reference("medi.mask-no-erosion", &[inv(&[Choice("medi")])], "masking.erosion", &[Number(0.0)],
        "The MEDI toolbox uses BET's mask as it is, without eroding it", MEDI_MASK_SOURCE),
    reference("chi-sep.mask-input", &[sep(CHI_SEP_TOOLBOX)], "masking.input", &[Choice("magnitude")],
        "The χ-separation toolbox runs MEDI's BET on the root-sum-of-squares magnitude over echoes",
        CHI_SEP_MASK_SOURCE),
    reference("chi-sep.mask-no-correction", &[sep(CHI_SEP_TOOLBOX)], "masking.inhomogeneity_correction",
        &[Choice("none")], "The χ-separation toolbox runs BET on the magnitude as read, with no bias-field correction",
        CHI_SEP_MASK_SOURCE),
    reference("chi-sep.mask-no-erosion", &[sep(CHI_SEP_TOOLBOX)], "masking.erosion", &[Number(0.0)],
        "The χ-separation toolbox uses BET's mask as it is, without eroding it", CHI_SEP_MASK_SOURCE),
    // DECOMPOSE's paper describes BET on the first echo's magnitude and nothing more: no
    // correction before it and no erosion after it are stated, so none is what its pipeline runs.
    reference("decompose.mask-input", &[sep(&[Choice("decompose")])], "masking.input",
        &[Choice("magnitude-first")], "DECOMPOSE's authors run BET on the first echo's GRE magnitude",
        DECOMPOSE_MASK_SOURCE),
    reference("decompose.mask-no-correction", &[sep(&[Choice("decompose")])],
        "masking.inhomogeneity_correction", &[Choice("none")],
        "DECOMPOSE's authors run BET on the first echo's magnitude; no bias-field correction is described",
        DECOMPOSE_MASK_SOURCE),
    reference("decompose.mask-no-erosion", &[sep(&[Choice("decompose")])], "masking.erosion", &[Number(0.0)],
        "DECOMPOSE's authors extract the brain with BET; no erosion of its mask is described",
        DECOMPOSE_MASK_SOURCE),
    reference("qsmart.mask-input", &[inv(&[Choice("qsmart")])], "masking.input", &[Choice("magnitude-first")],
        "QSMART runs BET on the first echo's magnitude (root-sum-of-squares over coils)",
        "QSMART brainmask.m l. 7"),
    reference("qsmart.mask-n4", &[inv(&[Choice("qsmart")])], "masking.inhomogeneity_correction",
        &[Choice("n4")], "QSMART corrects the first echo's magnitude with ANTs' N4BiasFieldCorrection before BET",
        "QSMART brainmask.m l. 10-11"),
    reference("qsmart.bet-f", &[inv(&[Choice("qsmart")])], "masking.bet.fractional_intensity",
        &[Number(0.2)], "QSMART runs bet2 with -f 0.2", "QSMART brainmask.m l. 14"),
    reference("qsmart.mask-no-erosion", &[inv(&[Choice("qsmart")])], "masking.erosion", &[Number(0.0)],
        "QSMART does not erode its mask: avoiding erosion is what its spatially dependent filtering is for",
        "QSMART brainmask.m l. 7-33; Yaghmaie et al. 2021 NeuroImage 231:117701 §1, §5"),
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
    reference_stated("qsmnet.unwrap-laplacian", &[inv(&[Choice("qsmnet"), Choice("qsmnet-plus")])],
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
        "Chen et al. 2021 NeuroImage 242:118477 (DECOMPOSE), §3.6"),
    reference("chi-sep.unwrap-romeo", &[sep(CHI_SEP_TOOLBOX)],
        "field_mapping.unwrapping_algorithm", ROMEO,
        "The χ-separation toolbox and χ-sepnet unwrap each echo with ROMEO",
        "χ-separation toolbox Chisep_script.m:99-111; Kim et al. 2025 (χ-sepnet) §2.1.1"),
    reference("qsmnetme.unwrap-romeo", &[inv(QSMNETME)],
        "field_mapping.unwrapping_algorithm", ROMEO,
        "QSMnetME's toolbox unwraps each echo with ROMEO",
        "χ-separation toolbox Chisep_script.m:99-111; Kim et al. 2025 (χ-sepnet) §2.1.1"),
    // Scoped out of the χ-separation toolbox: it runs STI Suite's iLSQR (`chi-sep-*.qsm-ilsqr`)
    // but unwraps with ROMEO (`chi-sep.unwrap-romeo`), so this rule is not its pipeline's.
    reference("sti.unwrap-laplacian", &[inv(STI), unless_sep(CHI_SEP_TOOLBOX)],
        "field_mapping.unwrapping_algorithm", LAPLACIAN,
        "STI Suite's QSM_GUI unwraps each echo with MRPhaseUnwrap (Laplacian)",
        "STI Suite 3.0 QSM_GUI, probed headless (recorded in QSM.rs laplacian.rs docs)"),
    reference("sti.laplacian-solver",
        &[inv(STI), method("field_mapping.unwrapping_algorithm", LAPLACIAN)],
        "field_mapping.laplacian_solver", &[Choice("fft-12")],
        "MRPhaseUnwrap is the FFT Schofield-Zhu solver with 12 voxels of padding",
        "STI Suite 3.0 MRPhaseUnwrap, probed (recorded in QSM.rs LaplacianSolver::Fft docs)"),
    reference("sti.unwrap-unmasked",
        &[inv(STI), method("field_mapping.unwrapping_algorithm", LAPLACIAN)],
        "field_mapping.laplacian_unmasked", &[Bool(true)],
        "STI Suite's QSM_GUI gives MRPhaseUnwrap the raw wrapped phase, not the phase masked first",
        "STI Suite 3.0 QSM_GUI, probed headless (recorded in QSM.rs laplacian.rs docs)"),
    reference("qsmart.unwrap-laplacian", &[inv(&[Choice("qsmart")])],
        "field_mapping.unwrapping_algorithm", LAPLACIAN,
        "QSMART unwraps each echo with Laplacian unwrapping (lapunwrap.m)",
        "QSMART Demo_QSMART.m l. 32; unwrap_phase.m l. 5-10"),
    reference("qsmart.laplacian-solver",
        &[inv(&[Choice("qsmart")]), method("field_mapping.unwrapping_algorithm", LAPLACIAN)],
        "field_mapping.laplacian_solver", &[Choice("mirror")],
        "QSMART's lapunwrap.m solves on the mirrored volume",
        "QSMART unwrap_phase.m l. 5-10, lapunwrap.m"),
    reference("qsmart.unwrap-unmasked",
        &[inv(&[Choice("qsmart")]), method("field_mapping.unwrapping_algorithm", LAPLACIAN)],
        "field_mapping.laplacian_unmasked", &[Bool(true)],
        "QSMART unwraps each echo's raw phase over the whole volume, and its echo fit fits every voxel",
        "QSMART unwrap_phase.m l. 5-10; echofit.m"),
    // ---------------------------------------------------------------- Field mapping
    reference("qsmart.field-echofit", &[inv(&[Choice("qsmart")])],
        "field_mapping.b0_estimation", &[Choice("qsmart-echofit")],
        "QSMART's total field and reliability map R_0 come from echofit.m",
        "QSMART QSMART.m l. 26; echofit.m"),
    reference("qsmart.fit-threshold",
        &[inv(&[Choice("qsmart")]), method("field_mapping.b0_estimation", &[Choice("qsmart-echofit")])],
        "field_mapping.b0_estimation.qsmart_fit_threshold", &[Choice("absolute-40")],
        "QSMART_toolbox_v1.0's demo cuts R_0 at a fixed echo-fit residual of 40 (adaptive_threshold 0)",
        "QSMART Demo_QSMART.m l. 38, 46; echofit.m l. 96-101"),
    reference("sti.echo-combination", &[inv(STI), MULTI_ECHO],
        "field_mapping.b0_estimation", WEIGHTED_AVG,
        "STI Suite's QSM_GUI sums the unwrapped echoes and divides by the summed echo times",
        "STI Suite 3.0 QSM_GUI, probed headless (recorded in QSM.rs star.rs docs)"),
    // Scoped out of the χ-separation toolbox, which weights the echoes its own way
    // (`chi-sep.echo-weights`).
    reference("sti.echo-weights", &[inv(STI), MULTI_ECHO, unless_sep(CHI_SEP_TOOLBOX)],
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
    // QSM.rs#439. Measured 2026-11-09: QSM-CI simulated head (RTS χ XSIM, Laplacian / ROMEO
    // field) and the travelling heads (inter-scanner median |a−1|, Laplacian / ROMEO).
    discourage("laplacian.non-eroding-bfr", &[bg(RIM_SENSITIVE_BFR)],
        "field_mapping.unwrapping_algorithm", LAPLACIAN, Choice("romeo"), Basis::Measured,
        "The Laplacian field map of the masked phase has a large error in the brain's outer three \
         voxels. PDF and LBV keep that rim, so the error spreads into the brain: on QSM-CI's \
         simulated head χ XSIM fell from 0.29 to 0.13 (PDF) and 0.28 to 0.11 (LBV) against ROMEO's \
         field, and across scanners (RTS, travelling heads) regional χ agreed far less: median \
         |slope − 1| 40 % against 7 % (PDF) and 15 % against 8 % (LBV). ROMEO, a background \
         removal that erodes the rim (SHARP, V-SHARP, RESHARP), or eroding the mask by a few voxels \
         avoids it",
        "QSM.rs#439 (QSM-CI simulated head and harmonization travelling heads, v11)"),
    discourage("field-mapping.bipolar-not-with-laplacian",
        &[method("field_mapping.bipolar_correction", &[Bool(true)]),
          method("field_mapping.unwrapping_algorithm", LAPLACIAN)],
        "field_mapping.bipolar_correction", &[Bool(true)], Bool(false), Basis::Implementation,
        "Bipolar correction is not applied on the Laplacian path",
        "QSM.rs pipeline::field_mapping docs"),
    // MEDI's README.m unwraps Fit_ppm_complex's field with unwrapPhase. ROMEO stays the default
    // unwrapper (QSMxT's too); this is only the authors' pairing, offered when the complex fit is
    // chosen and never preselected (2026-11-04).
    reference("complex-fit.unwrap-region-growing", &[method("field_mapping.b0_estimation", COMPLEX_FIT)],
        "field_mapping.unwrapping_algorithm", &[Choice("medi-region-growing")],
        "MEDI's pipeline unwraps Fit_ppm_complex's field with its region-growing unwrapPhase \
         (mexUnwrap.c), on the root-sum-of-squares magnitude",
        "MEDI toolbox 2024.11.26 README.m:80-95; functions/unwrapPhase.m, mexUnwrap.c (QSM.rs \
         unwrap::medi_region_growing docs)"),
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
    reference("qsmnetme.echo-combination", &[inv(QSMNETME), MULTI_ECHO],
        "field_mapping.b0_estimation", WEIGHTED_AVG,
        "QSMnetME's toolbox combines the unwrapped echoes by a weighted average",
        "χ-separation toolbox Chisep_script.m:113-123; Kim et al. 2025 (χ-sepnet) §2.1.1"),
    reference("qsmnetme.echo-weights", &[inv(QSMNETME), MULTI_ECHO],
        "field_mapping.b0_estimation.weighting", &[Choice("assumed-decay")],
        "QSMnetME's toolbox weights the echoes by TE·exp(−TE/T2*) with T2* = 40 ms (AssumedDecay, 0.04 s)",
        "χ-separation toolbox Chisep_script.m:113-123"),
    // ---------------------------------------------------------------- Background field removal
    reference("hdqsm.bg-pdf", &[inv(&[Choice("hdqsm")])], "bg_removal.algorithm", PDF,
        "HD-QSM's in-vivo pipeline removes the background with PDF",
        "Lambert et al. 2022 MRM 88:1600 (HD-QSM), §3.3"),
    reference("amp-pe.bg-pdf", &[inv(&[Choice("amp-pe")])], "bg_removal.algorithm", PDF,
        "AMP-PE's pipeline removes the background with PDF",
        "QSM_AMP_PE README (Overview, step 4); qsm_multi_echo_combined.m L112-139"),
    reference("lsqr.bg-sharp", &[inv(&[Choice("lsqr")])], "bg_removal.algorithm",
        &[Choice("sharp")],
        "The authors' LSQR/HEIDI pipeline inverts a SHARP local field",
        "qsm_heidi makefile L16-23, def.sh (4592631)"),
    reference("rts.bg-vsharp", &[inv(&[Choice("rts")])], "bg_removal.algorithm", VSHARP,
        "RTS's author's toolbox (QSM.jl) runs V-SHARP before it",
        "QSM.jl README.md:49 (fde4b92)"),
    reference("xqsm.bg-resharp", &[inv(&[Choice("xqsm")])], "bg_removal.algorithm", &[Choice("resharp")],
        "xQSM's in-vivo pipeline removes the background with RESHARP",
        "Gao et al. 2021 NMR Biomed 34:e4461 (xQSM), in-vivo experiments"),
    reference_stated("qsmnet.bg-vsharp", &[inv(&[Choice("qsmnet"), Choice("qsmnet-plus")])], "bg_removal.algorithm",
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
    reference("qsmnetme.bg-vsharp", &[inv(QSMNETME)], "bg_removal.algorithm", VSHARP,
        "QSMnetME's toolbox removes the background with V-SHARP",
        "χ-separation toolbox Chisep_script.m:136-139; Kim et al. 2025 (χ-sepnet) §2.1.1"),
    reference("r2star-qsm.bg-pdf", &[sep(&[Choice("r2star-qsm")])], "bg_removal.algorithm", PDF,
        "R2*-QSM's local field was from PDF",
        "Dimov et al. 2022 Tomography 8:1544, Methods"),
    reference("decompose.bg-vsharp", &[sep(&[Choice("decompose")])], "bg_removal.algorithm", VSHARP,
        "DECOMPOSE's pipeline removes the background with V-SHARP",
        "Chen et al. 2021 NeuroImage 242:118477 (DECOMPOSE), §3.6"),
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
        source: "STI Suite 3.0 V_SHARP, probed (QSM.rs VsharpParams, #293)",
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
        source: "STI Suite 3.0 V_SHARP, probed (QSM.rs VsharpParams, #293)",
    },
    // STI Suite's pipeline is its QSM_GUI's (decided 2026-11-10), not STI_Templates.m's: 40
    // iterations. The χ-separation toolbox calls QSM_iLSQR without 'niter' (100), so this rule is
    // scoped out of it (`chi-sep.ilsqr-iterations`).
    Rule {
        id: "sti.ilsqr-iterations",
        when: &[inv(&[Choice("ilsqr")]), unless_sep(CHI_SEP_TOOLBOX)],
        setting: "inversion.ilsqr.max_iter",
        expect: Test::OneOf { values: &[Number(crate::inversion::STI_GUI_NITER as f64)] },
        default: Number(crate::inversion::STI_GUI_NITER as f64),
        unit: Unit::None,
        strength: Strength::Reference,
        basis: Basis::ReferenceChoice,
        reason: "STI Suite's QSM_GUI calls QSM_iLSQR with niter 40",
        source: "STI Suite 3.0 QSM_GUI, probed headless (QSM.rs ilsqr.rs docs, Reference pipeline; STI_GUI_NITER)",
    },
    Rule {
        id: "chi-sep.ilsqr-iterations",
        when: &[sep(CHI_SEP_TOOLBOX_ITERATIVE), inv(&[Choice("ilsqr")])],
        setting: "inversion.ilsqr.max_iter",
        expect: Test::OneOf { values: &[Number(crate::inversion::STI_DEFAULT_NITER as f64)] },
        default: Number(crate::inversion::STI_DEFAULT_NITER as f64),
        unit: Unit::None,
        strength: Strength::Reference,
        basis: Basis::ReferenceChoice,
        reason: "The χ-separation toolbox calls QSM_iLSQR without niter, so it runs 100 iterations",
        source: "χ-separation toolbox v1.1.3 Chisep_script.m:145; STI Suite 3.0 QSM_iLSQR default, probed (bitwise equal to niter 100)",
    },
    // Single precision is STI Suite's own arithmetic, so it is part of STI Suite's pipeline and of
    // QSMART's (which calls STI's QSM_iLSQR), and of nothing else: the χ-separation toolbox also
    // calls QSM_iLSQR, but its pipeline keeps QSM.rs's double precision (decided 2026-11-10).
    reference("sti.ilsqr-single", &[inv(&[Choice("ilsqr")]), unless_sep(CHI_SEP_TOOLBOX)],
        "inversion.ilsqr.precision", &[Choice("single")],
        "STI Suite's QSM_iLSQR runs both LSQR solves in single precision",
        "STI Suite 3.0 QSM_iLSQR, probed (QSM.rs ilsqr.rs docs, Performance and precision)"),
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
        reason: "RESHARP's author's code allows its cgs solve up to 200 iterations",
        source: "sunhongfu/QSM background_field_removal/resharp.m l. 32-34, 92",
    },
    Rule {
        id: "pdf.cg-iterations",
        when: &[bg(PDF)],
        setting: "bg_removal.pdf.n_cg",
        expect: Test::Range { min: Some(PDF_SUGGESTED_N_CG), max: None },
        default: Number(PDF_SUGGESTED_N_CG),
        unit: Unit::None,
        strength: Strength::Recommended,
        basis: Basis::Measured,
        reason: "A tuning value: PDF.m's 30 CG iterations can stop far from convergence. At 300, \
                 a Philips 3 T scan's local field correlated 0.83 with V-SHARP's, against 0.54 \
                 at 30 (a GE scan: 0.84 against 0.85), and Philips scans lost a residual \
                 background that Siemens and GE scans did not have (so scanners can agree less \
                 at 300 than at 30, where they share it). On QSM-CI's simulated head the local \
                 field's NRMSE fell from 101 % to 84 % (58 % with a tolerance of 1e-5 as well). \
                 Costs about 30 s more per 1 mm whole-brain scan",
        source: "QSM.rs PdfParams::n_cg docs (travelling-head scans and the simulated head, \
                 2026-11-09); PDF.m default n_CG 30, MEDI toolbox 2024.11.26",
    },
    // A settings file from before the multigrid port can hold the SOR solver's 1e-6, a relative
    // tolerance; the setting is kept as given and this suggests LBV.m's own (QSM.rs#406).
    reference("lbv.tolerance", &[bg(LBV)], "bg_removal.lbv.tol", &[Number(0.01)],
        "LBV.m stops its coarsest grid's relaxation when the summed absolute update falls to 0.01 \
         (radians over the echo spacing)",
        "MEDI toolbox 2024.11.26 functions/_LBV/LBV.m:29-31; QSM.rs LbvParams::tol"),
    reference("ismv-qsmjl.tolerance", &[method("bg_removal.ismv.variant", &[Choice("qsmjl")])],
        "bg_removal.ismv.tol", &[Number(1e-3)],
        "QSM.jl's ismv stops when the residual falls to 1e-3 of the first",
        "QSM.jl src/bgremove/ismv.jl:39 (fde4b92); QSM.rs IsmvParams::qsmjl"),
    reference("ismv-qsmjl.max-iterations", &[method("bg_removal.ismv.variant", &[Choice("qsmjl")])],
        "bg_removal.ismv.max_iter", &[Number(500.0)],
        "QSM.jl's ismv stops after at most 500 iterations",
        "QSM.jl src/bgremove/ismv.jl:40 (fde4b92); QSM.rs IsmvParams::qsmjl"),
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
    reference("qsmart.ilsqr-single",
        &[inv(&[Choice("qsmart")]), method("inversion.qsmart.inversion", &[Choice("ilsqr")])],
        "inversion.ilsqr.precision", &[Choice("single")],
        "QSMART's two stages call STI Suite's QSM_iLSQR, which runs both LSQR solves in single precision",
        "QSMART QSMART.m l. 32-39; STI Suite QSM_iLSQR, probed (QSM.rs ilsqr.rs docs, Performance and precision)"),
    reference("qsmart.offset-grid", &[inv(&[Choice("qsmart")])], "inversion.qsmart.offset_grid",
        &[Choice("qsmart-reference")],
        "QSMART's adjust_offset.m builds its dipole kernel on its own grid (-N/2:N/2-1), half a \
         step off the FFT grid on odd matrix sizes",
        "QSMART adjust_offset.m; QSM.rs QsmartOffsetGrid::QsmartReference docs (#386)"),
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
    // ---------------------------------------------------------------- Boundary rim and in-vivo weights (v11, #437, #438)
    measured_better("iharperella.whole-rim", &[bg(&[Choice("iharperella")])],
        "bg_removal.iharperella.rim", &[Choice("whole")],
        "On a field map the brain's boundary rim holds the step at the mask edge; zeroing the whole \
         rim rather than STI's 20 % largest values made iHARPERELLA more reproducible across \
         scanners (travelling heads, inter-scanner median |slope - 1| 7.1 vs 3.8 %) and better on \
         the simulated head from a ROMEO field (XSIM 0.295 vs 0.244)",
        "QSM.rs bgremove::harperella HarperellaRim docs (#438)"),
    suggest("ndi.weight-norm-in-vivo", &[inv(&[Choice("ndi")])],
        "inversion.ndi.weight_norm", &[Choice("mask-p999")],
        "In vivo the magnitude's maximum is often outside the brain (2.3-5x the brain's 99.9th \
         percentile), so FANSI's max-normalised weight, and NDI's scale, vary from scan to scan; \
         normalising by the in-mask 99.9th percentile gave scales 1.04-1.07 against iLSQR (0.43-0.91). \
         Slightly worse on the simulated head (NRMSE 53.9 vs 52.0 %), whose maximum is in the brain",
        "QSM.rs inversion::ndi NdiWeightNorm docs (#437)"),
    suggest("whqsm.flat-in-mask-in-vivo",
        &[inv(&[Choice("whqsm")]), method("inversion.whqsm.weight", &[Choice("flat")])],
        "inversion.whqsm.weight", &[Choice("flat-in-mask")],
        "In vivo the flat weight's data term outside the brain adds smooth shading; the weight \
         inside the mask only gave r 0.72-0.75 with iLSQR (0.58-0.60). Slightly worse on the \
         simulated head (NRMSE 38.2 vs 33.6 %), whose local field is true outside the mask too",
        "QSM.rs inversion::whqsm WhQsmWeight::FlatInMask docs (#437)"),
    // ---------------------------------------------------------------- Echo time of the ppm↔rad conversion (#276)
    // With one echo every choice is the same time, so these need several.
    measured_better("fansi.te-spacing", &[inv(&[Choice("fansi"), Choice("fansi-tgv")]), MULTI_ECHO],
        "inversion.fansi.te", &[Choice("spacing")],
        "FANSI's multi-echo script converts at a TE difference; at the echo spacing nlTV and nlTGV \
         did better than at the first echo on every metric on the phantom",
        "FANSI-toolbox script_DICOM_Philips.m L364, L370; QSM.rs FansiParams::te docs (#276, #418)"),
    measured_better("ndi.te-first", &[inv(&[Choice("ndi")]), MULTI_ECHO],
        "inversion.ndi.te", &[Choice("first")],
        "FANSI's script allows any single echo time; the first echo gave NDI its best NRMSE and \
         correlation on the phantom",
        "FANSI-toolbox script_DICOM_Philips.m L364; QSM.rs NdiParams::te docs (#276, #418)"),
    measured_better("l1qsm.te-first", &[inv(&[Choice("l1qsm")]), MULTI_ECHO],
        "inversion.l1qsm.te", &[Choice("first")],
        "FANSI's script allows any single echo time; the first echo did better than the spacing \
         on the phantom (XSIM 0.776 vs 0.764, NRMSE 38.9 vs 46.9 %)",
        "FANSI-toolbox script_DICOM_Philips.m L364; QSM.rs L1QsmParams::te docs (#276, #418)"),
    discourage("whqsm.flat-weight-te-spacing",
        &[inv(&[Choice("whqsm")]), method("inversion.whqsm.weight", &[Choice("flat")]), MULTI_ECHO],
        "inversion.whqsm.te", &[Choice("spacing")], Choice("first"), Basis::Measured,
        "With the flat weight, WH-QSM did worse at the echo spacing than at the first echo on the \
         simulated head (NRMSE 42.0 vs 33.6 %, r 0.914 vs 0.945)",
        "QSM.rs WhQsmParams::te docs (#276)"),
    measured_better("whqsm.magnitude-weight-te-spacing",
        &[inv(&[Choice("whqsm")]),
          method("inversion.whqsm.weight", &[Choice("max-normalised"), Choice("in-vivo")]), MULTI_ECHO],
        "inversion.whqsm.te", &[Choice("spacing")],
        "With a magnitude weight, the echo spacing did better than the first echo on the phantom \
         (XSIM 0.676 vs 0.638); FANSI's script converts at a TE difference with a magnitude-derived weight",
        "QSM.rs WhQsmParams::te docs (#276, #418); FANSI-toolbox script_DICOM_Philips.m L364, L436"),
    reference("star.te-sum", &[inv(&[Choice("star")]), MULTI_ECHO], "inversion.star.te", &[Choice("sum")],
        "STI Suite's QSM_GUI gives QSM_star the summed phase with TE = ΣTE",
        "STI Suite 3.0 QSM_GUI, probed (recorded in QSM.rs StarParams::te, #276)"),
    reference("medi.te-spacing", &[inv(&[Choice("medi")]), MULTI_ECHO], "inversion.medi.te", &[Choice("spacing")],
        "MEDI_L1 converts with delta_TE = TE(2) − TE(1)",
        "MEDI toolbox 2024.11.26 MEDI_L1.m:202; Read_DICOM.m:166-168"),
    reference("tfi.te-spacing", &[inv(&[Choice("tfi")]), MULTI_ECHO], "inversion.tfi.te", &[Choice("spacing")],
        "TFI runs through MEDI_L1, which converts with delta_TE",
        "MEDI toolbox 2024.11.26 MEDI_L1.m:202 (QSM.rs TfiParams::te, #276)"),
    reference("heidi.te-last", &[HEIDI_ON, MULTI_ECHO], "inversion.heidi.te", &[Choice("last")],
        "The authors' HEIDI pipeline converts at the last echo time",
        "qsm_heidi phase2susceptibilityfactor.m L70-71 (QSM.rs HeidiParams::te, #276)"),
    // ---------------------------------------------------------------- HEIDI after the inversion
    Rule {
        id: "heidi.local-field",
        when: &[HEIDI_ON],
        setting: "inversion.algorithm",
        expect: Test::NoneOf { values: HEIDI_NO_LOCAL_FIELD },
        default: Choice("lsqr"),
        unit: Unit::None,
        strength: Strength::Required,
        basis: Basis::Implementation,
        reason: "HEIDI's homogeneity weights and cone fill take the local field the seed was \
                 inverted from; this method has none",
        source: "QSM.rs heidi.rs docs (Any seed); run_heidi_cone_fill",
    },
    // The seed rules name the seeds they are about in `expect` (none of them), not in `when`, so
    // each fires on its own seeds only and is resolved by the seed it names. Every seed but LSQR
    // is offered LSQR: recommended for the unregularised seeds and RTS, discouraged (with the
    // reason) for the regularised ones, required where there is no local field.
    Rule {
        id: "heidi.seed-lsqr",
        when: &[HEIDI_ON],
        setting: "inversion.algorithm",
        expect: Test::NoneOf { values: HEIDI_UNREGULARISED_SEEDS },
        default: Choice("lsqr"),
        unit: Unit::None,
        strength: Strength::Recommended,
        basis: Basis::Stated,
        reason: "HEIDI as published keeps the well-conditioned k-space of a minimally regularised \
                 LSQR solution and fills only the cone",
        source: "Schweser et al. 2012 NeuroImage 62:2083 (HEIDI); qsm_heidi makefile, inversion_lsqr.m then inversion_heidi.m (4592631)",
    },
    Rule {
        id: "heidi.seed-rts",
        when: &[HEIDI_ON],
        setting: "inversion.algorithm",
        expect: Test::NoneOf { values: &[Choice("rts")] },
        default: Choice("lsqr"),
        unit: Unit::None,
        strength: Strength::Recommended,
        basis: Basis::Stated,
        reason: "HEIDI's reference seed is LSQR; RTS already splits k-space at the cone and fills it \
                 with TV of its own",
        source: "Schweser et al. 2012 NeuroImage 62:2083 (HEIDI); Kames et al. 2018 NeuroImage 167:276 (RTS)",
    },
    reference("heidi.denoise-itk-float", &[HEIDI_ON, inv(&[Choice("lsqr")])], "inversion.heidi.denoise_itk",
        &[Bool(true)],
        "The HEIDI package denoises the field with ITK's GradientAnisotropicDiffusion executable, \
         which works in float",
        "qsm_heidi computesusceptibility_heidi.m L1439, diffusionimagefilter.m L129-134 (4592631); QSM.rs heidi.rs docs (AnisotropicDiffusionParams::reference_itk)"),
    discourage("heidi.seed-regularised", &[HEIDI_ON], "inversion.algorithm",
        HEIDI_REGULARISED_SEEDS, Choice("lsqr"), Basis::Stated,
        "HEIDI replaces only the dipole cone, so this method's regularisation outside the cone is \
         kept; HEIDI takes that region from a minimally regularised, near data-consistent solution",
        "Schweser et al. 2012 NeuroImage 62:2083 (HEIDI); QSM.rs heidi.rs docs (Any seed)"),
    heidi_truncation("heidi.tkd-cone", &[HEIDI_ON, inv(&[Choice("tkd")]), HEIDI_DEFAULT_CONE],
        "inversion.tkd.threshold"),
    heidi_truncation("heidi.tsvd-cone", &[HEIDI_ON, inv(&[Choice("tsvd")]), HEIDI_DEFAULT_CONE],
        "inversion.tsvd.threshold"),
    // ---------------------------------------------------------------- χ-separation inputs
    reference("chi-sep-ilsqr.qsm-ilsqr", &[sep(&[Choice("chi-sep-ilsqr")])], "inversion.algorithm",
        &[Choice("ilsqr")],
        "The χ-separation toolbox feeds it STI Suite iLSQR QSM",
        "χ-separation toolbox v1.1.3 Chisep_script.m:145,170"),
    reference("chi-sep-medi.qsm-ilsqr", &[sep(&[Choice("chi-sep-medi")])], "inversion.algorithm",
        &[Choice("ilsqr")],
        "The χ-separation toolbox initialises χ-separation-MEDI from STI Suite iLSQR QSM",
        "χ-separation toolbox v1.1.3 Chisep_script.m:145,158"),
    reference("r2star-qsm.qsm-medi", &[sep(&[Choice("r2star-qsm")])], "inversion.algorithm",
        &[Choice("medi")],
        "R2*-QSM's χ was reconstructed with MEDI",
        "Dimov et al. 2022 Tomography 8:1544, Methods"),
    reference("decompose.qsm-star", &[sep(&[Choice("decompose")])], "inversion.algorithm",
        &[Choice("star")],
        "DECOMPOSE's χ of each echo was reconstructed with STAR-QSM",
        "Chen et al. 2021 NeuroImage 242:118477 (DECOMPOSE), §3.6"),
    recommend("decompose.qsm-per-echo", &[sep(&[Choice("decompose")])], "qsm_per_echo", &[Bool(true)],
        "DECOMPOSE's authors reconstruct a QSM for each echo, χ(t), rather than one QSM from all \
         echoes combined, and state that a TE-dependent QSM input is necessary",
        "Chen et al. 2021 NeuroImage 242:118477 (DECOMPOSE), §2.2, §4.1; STAR-QSM per echo §3.6"),
    measured_better("decompose.unwrap-7t",
        &[sep(&[Choice("decompose")]), fact_at_least(Fact::FieldStrength, DECOMPOSE_UNWRAP_FROM_T)],
        "separation.decompose.unwrap_phase", &[Bool(true)],
        "DECOMPOSE was designed for 3 T (its χ bound keeps the phase under 2π at TE 25 ms and \
         3 T); at 7 T the phase wraps between echoes and χ+ partly collapses unless the phase is \
         unwrapped along the echoes (with decompose.chi-bound-7t's bound)",
        "Chen et al. 2021 NeuroImage 242:118477 (DECOMPOSE), §2.2; measured on the QSM-CI 7 T \
         chisep phantom, χ+ r 0.48 off, 0.66 on, 0.90 on with the echo-spacing χ bound (QSM.rs \
         separation::decompose docs, At 7 T)"),
    measured_better("decompose.chi-bound-7t",
        &[sep(&[Choice("decompose")]), fact_at_least(Fact::FieldStrength, DECOMPOSE_UNWRAP_FROM_T)],
        "separation.decompose.chi_bound", &[Choice("echo-spacing")],
        "A tuning value: bound the χ search at π / ((2/3)·γ·B0·ΔTE_max), computed from the scan \
         (0.31 ppm at 7 T with 8 ms spacing), the largest |χ| whose phase step between echoes \
         stays under π so that unwrapping along the echoes is unambiguous; the authors' 0.5 ppm \
         was set for TE 25 ms at 3 T",
        "Chen et al. 2021 NeuroImage 242:118477 (DECOMPOSE), §2.2; measured on the QSM-CI 7 T \
         chisep phantom, χ− r 0.75 → 0.85 with this bound alone, χ+ r 0.66 → 0.90 with \
         unwrapping (QSM.rs separation::decompose docs, At 7 T)"),
    // DECOMPOSE's pipeline is its paper's (decided 2026-11-10), not Tim Ho's script.
    reference("decompose.fit-paper", &[sep(&[Choice("decompose")])], "separation.decompose.fit",
        &[Choice("paper")],
        "DECOMPOSE's paper fits the amplitudes with C+ + C− + C₀ = 1 and the amplitudes and R2*₀ by \
         linear least squares (lsqlin), χ± by lsqnonlin; the data are scaled to S(0) = 1 per voxel \
         (QSM.rs's reading: the paper does not say how)",
        "Chen et al. 2021 NeuroImage 242:118477 (DECOMPOSE), §2.2 Eq. 5.1-5.3, §3.1; QSM.rs \
         separation::decompose docs (DecomposeStageSolver, DecomposeNormalisation)"),
    reference("decompose.reference-mean", &[sep(&[Choice("decompose")])], "reference",
        &[Choice("mean")],
        "DECOMPOSE's χ is referenced to the whole-brain mean",
        "Chen et al. 2021 NeuroImage 242:118477 (DECOMPOSE), §3.6"),
    // ---------------------------------------------------------------- Referencing
    // Toolboxes whose pipeline ends at the inversion's own output: nothing is subtracted after it.
    reference("medi.reference-none", &[inv(&[Choice("medi")])], "reference", &[Choice("none")],
        "MEDI's pipeline references χ inside MEDI_L1 (MEDI+0 subtracts the mean over the \
         ventricular CSF; with fewer than three echoes there is no CSF mask and nothing is \
         subtracted), and nothing after it",
        "MEDI toolbox 2024.11.26 MEDI_L1.m:204-207; arlo.m:22-26 (QSM.rs inversion::medi docs, MEDI+0)"),
    reference("qsmart.reference-none", &[inv(&[Choice("qsmart")])], "reference", &[Choice("none")],
        "QSMART's result is adjust_offset's combined χ, which nothing references afterwards",
        "QSMART QSMART.m l. 43-45; adjust_offset.m"),
    reference("heidi.reference-none", &[HEIDI_ON, inv(&[Choice("lsqr")])], "reference", &[Choice("none")],
        "The authors' LSQR/HEIDI pipeline saves HEIDI's χ in ppm without referencing it",
        "qsm_heidi inversion_heidi.m L55-65 (4592631)"),
    reference("susep-net.qsm-iqsm-plus", &[sep(&[Choice("susep-net")])], "inversion.algorithm",
        &[Choice("iqsm-plus")],
        "SUSEP-Net's in-vivo QSM input is from iQSM+",
        "SUSEP-Net (arXiv:2506.13293) §3.2"),
    discourage("chi-sepnet.min-resolution",
        &[sep(&[Choice("chi-sepnet")]), fact_at_most(Fact::MinVoxelSize, 0.6)],
        "separation.algorithm", &[Choice("chi-sepnet")], Choice("chi-sep-ilsqr"), Basis::Stated,
        "χ-sepnet's authors state the networks process resolutions coarser than 0.6 mm",
        "SNU-LIST/chi-separation README; Kim et al. 2024 (arXiv:2410.12239) §4"),
    // The toolbox's Tukey window, before all three of its χ-separation methods (Chisep_script.m
    // l. 69-73 run ahead of the χ-sepnet, χ-sep-MEDI and χ-sep-iLSQR calls). The script's comment
    // calls the vendor values a "Recommendation" but gives no reason: the authors' setting. The
    // script's value is 0.4 ("Siemens, GE: 0.4, Philips: 0"), so every known manufacturer but
    // Philips gets 0.4 (Canon and the others too: 0.4 is what the script runs unless the user
    // changes it for Philips). An unknown manufacturer fires neither rule, and the window stays
    // at the host's default.
    reference_stated("chi-sep.tukey-not-philips",
        &[sep(CHI_SEP_TOOLBOX_ITERATIVE), not_manufacturer(&[Choice("philips")])],
        "tukey_window", &[Number(CHISEP_TUKEY)], TUKEY_NOT_PHILIPS_REASON, TUKEY_SOURCE),
    reference_stated("chi-sep.tukey-philips",
        &[sep(CHI_SEP_TOOLBOX_ITERATIVE), manufacturer(&[Choice("philips")])],
        "tukey_window", &[Number(0.0)], TUKEY_PHILIPS_REASON, TUKEY_SOURCE),
    reference_stated("chi-sepnet.tukey-not-philips",
        &[sep(&[Choice("chi-sepnet")]), CHI_SEPNET_TOOLBOX_ROUTE, not_manufacturer(&[Choice("philips")])],
        "tukey_window", &[Number(CHISEP_TUKEY)], TUKEY_NOT_PHILIPS_REASON, TUKEY_SOURCE),
    reference_stated("chi-sepnet.tukey-philips",
        &[sep(&[Choice("chi-sepnet")]), CHI_SEPNET_TOOLBOX_ROUTE, manufacturer(&[Choice("philips")])],
        "tukey_window", &[Number(0.0)], TUKEY_PHILIPS_REASON, TUKEY_SOURCE),
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
            manufacturer: Some(Manufacturer::Siemens),
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

    /// A suggestion is reasoned (it gives a reason) but never preselected; every other reasoned
    /// strength is, and the JSON says so.
    #[test]
    fn suggested_is_reasoned_and_not_preselected() {
        assert_eq!(Strength::Suggested.kind(), Kind::Reasoned);
        assert!(!Strength::Suggested.preselected() && !Strength::Reference.preselected());
        assert!(Strength::Discouraged.preselected() && Strength::Recommended.preselected() && Strength::Required.preselected());
        assert!(Strength::Reference < Strength::Suggested && Strength::Suggested < Strength::Discouraged);
        let ndi = RULES.iter().find(|r| r.id == "ndi.weight-norm-in-vivo").unwrap();
        let json = serde_json::to_value(ndi).unwrap();
        assert_eq!((json["kind"].as_str(), json["strength"].as_str(), json["preselected"].as_bool()),
            (Some("reasoned"), Some("suggested"), Some(false)));
        let rim = RULES.iter().find(|r| r.id == "iharperella.whole-rim").unwrap();
        assert_eq!(serde_json::to_value(rim).unwrap()["preselected"].as_bool(), Some(true));
    }

    /// With HEIDI on, every inversion but LSQR gets a finding that offers LSQR: recommended,
    /// discouraged (with its reason) or required, never only information.
    #[test]
    fn heidi_offers_lsqr_for_every_other_seed() {
        for &a in InversionAlgorithm::VARIANTS {
            let mut case = Case::new(|c| { inversion(a)(c); c.cfg.inversion.apply_heidi = true });
            // TKD/TSVD at the cone threshold, so their truncation rule does not fire too.
            case.cfg.inversion.tkd.threshold = 0.1;
            case.cfg.inversion.tsvd.threshold = 0.1;
            let lsqr = evaluate(&case.input(), &case.facts).into_iter().filter(|f| {
                f.rule.setting == "inversion.algorithm" && f.rule.id.starts_with("heidi.")
                    && f.rule.default == Choice("lsqr") && f.rule.strength > Strength::Reference
            }).count();
            assert_eq!(lsqr > 0, a != InversionAlgorithm::Lsqr, "{a:?}: {:?}", case.ids());
        }
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
            ("qsmnetme.mask-bet", Case::new(inversion(I::Qsmnetme))),
            ("decompose.mask-bet", Case::new(separation(S::Decompose))),
            // The default mask (phase quality, makehomogeneous, Otsu, erode 1) departs from each.
            ("medi.mask-input", Case::new(inversion(I::Medi))),
            ("medi.mask-no-correction", Case::new(inversion(I::Medi))),
            ("medi.mask-no-erosion", Case::new(inversion(I::Medi))),
            ("chi-sep.mask-input", Case::new(separation(S::ChiSepIlsqr))),
            ("chi-sep.mask-no-correction", Case::new(separation(S::ChiSepNet))),
            ("chi-sep.mask-no-erosion", Case::new(separation(S::ChiSepMedi))),
            ("decompose.mask-input", Case::new(separation(S::Decompose))),
            ("decompose.mask-no-correction", Case::new(separation(S::Decompose))),
            ("decompose.mask-no-erosion", Case::new(separation(S::Decompose))),
            ("qsmart.mask-input", Case::new(inversion(I::Qsmart))),
            ("qsmart.mask-n4", Case::new(inversion(I::Qsmart))),
            ("qsmart.bet-f", Case::new(|c| {
                inversion(I::Qsmart)(c);
                c.cfg.masking.sections[0].generator = MaskOp::Bet(Default::default());
            })),
            ("qsmart.mask-no-erosion", Case::new(inversion(I::Qsmart))),
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
            ("qsmnetme.unwrap-romeo", Case::new(|c| {
                inversion(I::Qsmnetme)(c);
                c.cfg.field_mapping.unwrapping_algorithm = UnwrappingAlgorithm::Laplacian;
            })),
            ("field-mapping.phase-offsets", Case::new(|c| c.cfg.field_mapping.phase_offset_removal = false)),
            ("field-mapping.bipolar-needs-three-echoes", Case::new(|c| {
                c.cfg.field_mapping.bipolar_correction = true;
                c.facts.echo_count = Some(2);
            })),
            ("laplacian.non-eroding-bfr", Case::new(|c| {
                background(B::Lbv)(c);
                c.cfg.field_mapping.unwrapping_algorithm = UnwrappingAlgorithm::Laplacian;
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
            ("qsmnetme.echo-combination", Case::new(|c| { inversion(I::Qsmnetme)(c); linear_fit(c) })),
            ("qsmnetme.echo-weights", Case::new(inversion(I::Qsmnetme))),
            ("hdqsm.bg-pdf", Case::new(inversion(I::Hdqsm))),
            ("amp-pe.bg-pdf", Case::new(inversion(I::AmpPe))),
            ("lsqr.bg-sharp", Case::new(inversion(I::Lsqr))),
            ("heidi.local-field", Case::new(|c| { inversion(I::Tfi)(c); c.cfg.inversion.apply_heidi = true })),
            ("heidi.seed-lsqr", Case::new(|c| {
                inversion(I::Tkd)(c);
                c.cfg.inversion.apply_heidi = true;
                c.cfg.inversion.tkd.threshold = 0.1;
            })),
            ("heidi.seed-rts", Case::new(|c| { inversion(I::Rts)(c); c.cfg.inversion.apply_heidi = true })),
            ("heidi.seed-regularised", Case::new(|c| { inversion(I::Medi)(c); c.cfg.inversion.apply_heidi = true })),
            ("heidi.tkd-cone", Case::new(|c| { inversion(I::Tkd)(c); c.cfg.inversion.apply_heidi = true })),
            ("heidi.tsvd-cone", Case::new(|c| { inversion(I::Tsvd)(c); c.cfg.inversion.apply_heidi = true })),
            ("rts.bg-vsharp", Case::new(background(B::Pdf))),
            ("xqsm.bg-resharp", Case::new(inversion(I::Xqsm))),
            ("qsmnet.bg-vsharp", Case::new(|c| { inversion(I::Qsmnet)(c); background(B::Lbv)(c) })),
            ("qsmgan.bg-vsharp", Case::new(|c| { inversion(I::Qsmgan)(c); background(B::Lbv)(c) })),
            ("lpcnn.bg-vsharp", Case::new(|c| { inversion(I::Lpcnn)(c); background(B::Lbv)(c) })),
            ("chi-sep.bg-vsharp", Case::new(|c| { separation(S::ChiSepIlsqr)(c); background(B::Pdf)(c) })),
            ("qsmnetme.bg-vsharp", Case::new(|c| { inversion(I::Qsmnetme)(c); background(B::Pdf)(c) })),
            ("r2star-qsm.bg-pdf", Case::new(separation(S::R2starQsm))),
            ("decompose.fit-paper", Case::new(separation(S::Decompose))),
            ("decompose.bg-vsharp", Case::new(|c| { separation(S::Decompose)(c); background(B::Pdf)(c) })),
            ("star.padding", Case::new(|c| {
                c.cfg.inversion.algorithm = I::Star;
                c.cfg.inversion.star.pad = [12; 3];
            })),
            ("chi-sep-ilsqr.qsm-ilsqr", Case::new(separation(S::ChiSepIlsqr))),
            ("chi-sep-medi.qsm-ilsqr", Case::new(separation(S::ChiSepMedi))),
            ("medi.bg-pdf", Case::new(inversion(I::Medi))),
            ("sti.vsharp-min-radius", Case::new(|c| { inversion(I::Ilsqr)(c); c.cfg.bg_removal.vsharp.min_radius = 1.0 })),
            ("sti.vsharp-radius-step", Case::new(|c| { inversion(I::Star)(c); c.cfg.bg_removal.vsharp.radius_step = 2.0 })),
            ("sti.ilsqr-iterations", Case::new(inversion(I::Ilsqr))),
            ("chi-sep.ilsqr-iterations", Case::new(|c| {
                separation(S::ChiSepIlsqr)(c);
                inversion(I::Ilsqr)(c);
                c.cfg.inversion.ilsqr.max_iter = 40;
            })),
            ("sti.ilsqr-single", Case::new(inversion(I::Ilsqr))),
            ("qsmart.ilsqr-single", Case::new(inversion(I::Qsmart))),
            ("qsmart.offset-grid", Case::new(inversion(I::Qsmart))),
            ("qsmart.fit-threshold", Case::new(|c| {
                inversion(I::Qsmart)(c);
                c.cfg.field_mapping.b0_estimation = B0EstimationMethod::QsmartEchofit {
                    fit_threshold: crate::utils::QsmartFitThreshold::Percentile(99.0),
                };
            })),
            ("heidi.denoise-itk-float", Case::new(|c| { inversion(I::Lsqr)(c); c.cfg.inversion.apply_heidi = true })),
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
            ("iharperella.whole-rim", Case::new(background(B::Iharperella))),
            ("ndi.weight-norm-in-vivo", Case::new(inversion(I::Ndi))),
            ("whqsm.flat-in-mask-in-vivo", Case::new(inversion(I::Whqsm))),
            ("l1qsm.te-first", Case::new(|c| { inversion(I::L1qsm)(c); c.cfg.inversion.l1qsm.te = crate::EchoTime::Sum })),
            ("whqsm.flat-weight-te-spacing", Case::new(|c| { inversion(I::Whqsm)(c); c.cfg.inversion.whqsm.te = crate::EchoTime::Spacing })),
            ("whqsm.magnitude-weight-te-spacing", Case::new(|c| {
                inversion(I::Whqsm)(c);
                c.cfg.inversion.whqsm.weight = crate::inversion::WhQsmWeight::InVivo;
            })),
            ("star.te-sum", Case::new(|c| { inversion(I::Star)(c); c.cfg.inversion.star.te = crate::EchoTime::First })),
            ("medi.te-spacing", Case::new(|c| { inversion(I::Medi)(c); c.cfg.inversion.medi.te = crate::EchoTime::First })),
            ("tfi.te-spacing", Case::new(|c| { inversion(I::Tfi)(c); c.cfg.inversion.tfi.te = crate::EchoTime::Last })),
            ("heidi.te-last", Case::new(|c| {
                inversion(I::Lsqr)(c);
                c.cfg.inversion.apply_heidi = true;
                c.cfg.inversion.heidi.te = crate::EchoTime::First;
            })),
            ("sti.bg-vsharp", Case::new(|c| { inversion(I::Star)(c); background(B::Pdf)(c) })),
            ("sti.vsharp-no-deconvolve", Case::new(|c| { inversion(I::Ilsqr)(c); c.cfg.bg_removal.vsharp.deconvolve = true })),
            ("sti.vsharp-radius", Case::new(|c| { inversion(I::Ilsqr)(c); c.cfg.bg_removal.vsharp.max_radius = 8.0 })),
            ("sti.vsharp-grid", Case::new(|c| { inversion(I::Star)(c); c.cfg.bg_removal.vsharp.grid = crate::bgremove::VsharpGrid::Native })),
            ("resharp.radius", Case::new(|c| { background(B::Resharp)(c); c.cfg.bg_removal.resharp.radius = 6.0 })),
            ("resharp.iterations", Case::new(|c| { background(B::Resharp)(c); c.cfg.bg_removal.resharp.max_iter = 30 })),
            ("pdf.cg-iterations", Case::new(background(B::Pdf))),
            ("lbv.tolerance", Case::new(|c| { background(B::Lbv)(c); c.cfg.bg_removal.lbv.tol = 1e-6 })),
            ("ismv-qsmjl.tolerance", Case::new(|c| { background(B::Ismv)(c); c.cfg.bg_removal.ismv.variant = crate::bgremove::IsmvVariant::QsmJl })),
            ("ismv-qsmjl.max-iterations", Case::new(|c| { background(B::Ismv)(c); c.cfg.bg_removal.ismv.variant = crate::bgremove::IsmvVariant::QsmJl })),
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
            ("qsmart.unwrap-unmasked", Case::new(|c| {
                inversion(I::Qsmart)(c);
                c.cfg.field_mapping.unwrapping_algorithm = UnwrappingAlgorithm::Laplacian;
            })),
            ("sti.unwrap-unmasked", Case::new(|c| {
                inversion(I::Ilsqr)(c);
                c.cfg.field_mapping.unwrapping_algorithm = UnwrappingAlgorithm::Laplacian;
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
            ("decompose.qsm-per-echo", Case::new(separation(S::Decompose))),
            ("decompose.unwrap-7t", Case::new(|c| {
                separation(S::Decompose)(c);
                c.facts.field_strength = Some(7.0);
            })),
            ("decompose.chi-bound-7t", Case::new(|c| {
                separation(S::Decompose)(c);
                c.facts.field_strength = Some(7.0);
            })),
            ("decompose.reference-mean", Case::new(|c| {
                separation(S::Decompose)(c);
                c.cfg.reference = QsmReference::None;
            })),
            ("susep-net.qsm-iqsm-plus", Case::new(separation(S::SusepNet))),
            ("chi-sepnet.min-resolution", Case::new(|c| {
                separation(S::ChiSepNet)(c);
                c.facts.voxel_size = Some((0.5, 0.5, 0.5));
            })),
            ("medi.reference-none", Case::new(inversion(I::Medi))),
            ("qsmart.reference-none", Case::new(inversion(I::Qsmart))),
            ("heidi.reference-none", Case::new(|c| { inversion(I::Lsqr)(c); c.cfg.inversion.apply_heidi = true })),
            ("chi-sepnet.tukey-not-philips", Case::new(separation(S::ChiSepNet))),
            ("chi-sep.tukey-not-philips", Case::new(separation(S::ChiSepMedi))),
            ("chi-sep.tukey-philips", Case::new(|c| {
                separation(S::ChiSepIlsqr)(c);
                c.facts.manufacturer = Some(Manufacturer::Philips);
                c.cfg.tukey_window = 0.4;
            })),
            ("complex-fit.unwrap-region-growing", Case::new(|c| {
                c.cfg.field_mapping.b0_estimation = B0EstimationMethod::ComplexFit(Default::default());
            })),
            ("chi-sepnet.tukey-philips", Case::new(|c| {
                separation(S::ChiSepNet)(c);
                c.facts.manufacturer = Some(Manufacturer::Philips);
                c.cfg.tukey_window = 0.4;
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
                if let Condition::Setting { setting, .. } | Condition::Unless { setting, .. } = c {
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

    /// Each rule's kind follows its strength: only `Reference` rules are the authors' setting,
    /// and the JSON form says which.
    #[test]
    fn kinds_follow_strengths() {
        for r in RULES {
            assert_eq!(r.kind() == Kind::Reference, r.strength == Strength::Reference, "{}", r.id);
        }
        let json = serde_json::to_value(RULES).unwrap();
        for (r, j) in RULES.iter().zip(json.as_array().unwrap()) {
            assert_eq!(j["kind"], serde_json::to_value(r.kind()).unwrap(), "{}", r.id);
        }
        let kind = |id: &str| RULES.iter().find(|r| r.id == id).unwrap().kind();
        // Ashley's examples (2026-11-04): the authors' pairing vs better for a stated reason.
        assert_eq!(kind("complex-fit.unwrap-region-growing"), Kind::Reference);
        assert_eq!(kind("chi-sepnet.tukey-not-philips"), Kind::Reference);
        assert_eq!(kind("qsmnet.bg-vsharp"), Kind::Reference);
        assert_eq!(kind("heidi.seed-lsqr"), Kind::Reasoned);
        assert_eq!(kind("pdf.cg-iterations"), Kind::Reasoned);
    }

    /// Precedence, and the basis matching each strength.
    #[test]
    fn strengths_order_and_rest_on_their_basis() {
        use Strength::*;
        assert!(Required > Recommended && Recommended > Discouraged && Discouraged > Suggested && Suggested > Reference);
        for r in RULES {
            match r.strength {
                Reference => assert!(matches!(r.basis, Basis::ReferenceChoice | Basis::Stated), "{}", r.id),
                Recommended => assert!(matches!(r.basis, Basis::Stated | Basis::Measured), "{}", r.id),
                Suggested => assert_eq!(r.basis, Basis::Measured, "{}", r.id),
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
            c.cfg.field_mapping.unwrapping_algorithm = UnwrappingAlgorithm::MediRegionGrowing;
            c.cfg.bg_removal.pdf.n_cg = 300;
        });
        assert!(c.ids().is_empty(), "{:?}", c.ids());
    }

    /// PDF's iteration count: the default is PDF.m's 30, and 300 (or more) is a measured
    /// suggestion, silent once followed and when PDF is not run.
    #[test]
    fn pdf_suggests_more_iterations() {
        let id = "pdf.cg-iterations";
        let r = RULES.iter().find(|r| r.id == id).unwrap();
        assert_eq!((r.strength, r.basis, r.default), (Strength::Recommended, Basis::Measured, Number(300.0)));
        assert_eq!(crate::bgremove::PdfParams::default().n_cg, 30);
        let with = |n: usize| Case::new(|c| {
            c.cfg.bg_removal.algorithm = BgRemovalAlgorithm::Pdf;
            c.cfg.bg_removal.pdf.n_cg = n;
        }).ids().contains(&id);
        assert!(with(30) && with(299));
        assert!(!with(300) && !with(1000));
        // Not in play without PDF, nor when the inversion skips background removal.
        assert!(!Case::new(|c| c.cfg.bg_removal.pdf.n_cg = 30).ids().contains(&id));
        let total_field = Case::new(|c| {
            c.cfg.inversion.algorithm = InversionAlgorithm::Autoqsm;
            c.cfg.bg_removal.algorithm = BgRemovalAlgorithm::Pdf;
        });
        assert!(!total_field.ids().contains(&id), "{:?}", total_field.ids());
    }

    /// χ-sepnet's toolbox route, χ-sep-MEDI and χ-sep-iLSQR suggest the toolbox's Tukey window by
    /// manufacturer (the authors' setting, never preselected): 0 for Philips, the script's 0.4
    /// for every other known manufacturer; nothing on χ-sepnet's native route, with other
    /// separation methods, for an unknown manufacturer, or once followed.
    #[test]
    fn chi_sepnet_suggests_the_vendor_tukey_window() {
        use crate::separation::ChiSepNetRoute;
        let ids = |m: Option<Manufacturer>, route: ChiSepNetRoute, tukey: f64| Case::new(|c| {
            separation(SeparationAlgorithm::ChiSepNet)(c);
            c.sep.as_mut().unwrap().chi_sepnet.route = route;
            c.facts.manufacturer = m;
            c.cfg.tukey_window = tukey;
        }).ids().into_iter().filter(|id| id.starts_with("chi-sepnet.tukey")).collect::<Vec<_>>();
        let (sg, ph) = ("chi-sepnet.tukey-not-philips", "chi-sepnet.tukey-philips");
        use Manufacturer::*;
        use ChiSepNetRoute::*;
        assert_eq!(ids(Some(Siemens), Toolbox, 0.0), [sg]);
        assert_eq!(ids(Some(Ge), Toolbox, 0.2), [sg]);
        assert!(ids(Some(Ge), Toolbox, 0.4).is_empty());
        assert_eq!(ids(Some(Philips), Toolbox, 0.4), [ph]);
        assert!(ids(Some(Philips), Toolbox, 0.0).is_empty());
        assert_eq!(ids(Some(Other), Toolbox, 0.0), [sg]);
        assert!(ids(None, Toolbox, 0.0).is_empty());
        assert!(ids(Some(Siemens), Native, 0.0).is_empty());
        let r = RULES.iter().find(|r| r.id == sg).unwrap();
        assert_eq!((r.strength, r.basis, r.default), (Strength::Reference, Basis::Stated, Number(0.4)));
        assert!(r.needs_scan());
        // χ-sep-MEDI and χ-sep-iLSQR: the same values, under their own rule ids.
        let tukey = |a: SeparationAlgorithm, m: Manufacturer, t: f64| Case::new(|c| {
            separation(a)(c);
            c.facts.manufacturer = Some(m);
            c.cfg.tukey_window = t;
        }).ids().into_iter().filter(|id| id.contains(".tukey")).collect::<Vec<_>>();
        for a in [SeparationAlgorithm::ChiSepMedi, SeparationAlgorithm::ChiSepIlsqr] {
            for m in [Siemens, Ge, Other] {
                assert_eq!(tukey(a, m, 0.0), ["chi-sep.tukey-not-philips"], "{a:?} {m:?}");
                assert!(tukey(a, m, 0.4).is_empty(), "{a:?} {m:?}");
            }
            assert_eq!(tukey(a, Philips, 0.4), ["chi-sep.tukey-philips"], "{a:?}");
            assert!(tukey(a, Philips, 0.0).is_empty(), "{a:?}");
        }
        // Not with the other separation methods.
        for a in [SeparationAlgorithm::Decompose, SeparationAlgorithm::R2starQsm, SeparationAlgorithm::SusepNet] {
            assert!(tukey(a, Siemens, 0.0).is_empty(), "{a:?}");
        }
        for id in ["chi-sep.tukey-not-philips", "chi-sep.tukey-philips", "chi-sepnet.tukey-philips"] {
            assert_eq!(RULES.iter().find(|r| r.id == id).unwrap().kind(), Kind::Reference, "{id}");
        }
    }

    /// The toolboxes that reference nothing after their inversion (MEDI, QSMART, LSQR with HEIDI)
    /// ask for no referencing; HEIDI only with its own seed, and DECOMPOSE keeps the mean.
    #[test]
    fn toolboxes_that_reference_nothing_after_the_inversion() {
        let fires = |c: Case, id: &str| c.ids().contains(&id);
        let with = |a: InversionAlgorithm, heidi: bool, reference: QsmReference| Case::new(|c| {
            inversion(a)(c);
            c.cfg.inversion.apply_heidi = heidi;
            c.cfg.reference = reference;
        });
        use InversionAlgorithm as I;
        for (id, a) in [("medi.reference-none", I::Medi), ("qsmart.reference-none", I::Qsmart)] {
            assert!(fires(with(a, false, QsmReference::Mean), id), "{id}");
            assert!(!fires(with(a, false, QsmReference::None), id), "{id}");
        }
        assert!(fires(with(I::Lsqr, true, QsmReference::Mean), "heidi.reference-none"));
        assert!(!fires(with(I::Lsqr, true, QsmReference::None), "heidi.reference-none"));
        assert!(!fires(with(I::Tkd, true, QsmReference::Mean), "heidi.reference-none"));
        assert!(!fires(with(I::Lsqr, false, QsmReference::Mean), "heidi.reference-none"));
        for id in ["medi.reference-none", "qsmart.reference-none", "heidi.reference-none"] {
            let r = RULES.iter().find(|r| r.id == id).unwrap();
            assert_eq!((r.kind(), r.default), (Kind::Reference, Choice("none")), "{id}");
        }
        let decompose = Case::new(|c| { separation(SeparationAlgorithm::Decompose)(c); c.cfg.reference = QsmReference::None });
        assert!(fires(decompose, "decompose.reference-mean"));
    }

    /// The reference rules that apply to `case` (their conditions hold and their setting is in
    /// play), whether or not it follows them.
    fn applying(case: &Case) -> Vec<&'static Rule> {
        RULES.iter().filter(|r| r.strength == Strength::Reference).filter(|r| {
            r.when.iter().all(|c| holds(c, &case.input(), &case.facts))
                && !current(&case.input(), &case.facts, r.setting).unwrap_or_default().is_empty()
        }).collect()
    }

    /// Pairs of reference rules that apply to `case` and want different values of one setting.
    fn conflicts(case: &Case) -> Vec<(&'static str, &'static str)> {
        let rules = applying(case);
        let d = |r: &Rule| match r.default {
            Value::Bool(b) => Current::Bool(b),
            Value::Number(x) => Current::Number(x),
            Value::Choice(s) => Current::Choice(s),
            Value::Vector(v) => Current::Vector(v.to_vec()),
        };
        let mut out = Vec::new();
        for (i, a) in rules.iter().enumerate() {
            for b in &rules[i + 1..] {
                if a.setting == b.setting && !(b.expect.passes(&d(a)) && a.expect.passes(&d(b))) {
                    out.push((a.id, b.id));
                }
            }
        }
        out
    }

    /// The χ-separation toolbox runs STI Suite's iLSQR but maps the field its own way, so under
    /// its methods STI Suite's field-mapping rules do not apply: no two reference rules disagree,
    /// and once the toolbox's field mapping is followed no STI Suite rule is departed from (a
    /// host's methods text would otherwise say ROMEO differs from iLSQR's reference). STI Suite's
    /// V-SHARP rules still apply: the toolbox calls STI Suite's V_SHARP.
    #[test]
    fn chi_sep_toolbox_methods_and_sti_suite_agree() {
        use crate::separation::ChiSepNetRoute;
        let toolbox = |a: SeparationAlgorithm, inv_a: InversionAlgorithm, followed: bool| Case::new(move |c| {
            separation(a)(c);
            c.sep.as_mut().unwrap().chi_sepnet.route = ChiSepNetRoute::Toolbox;
            inversion(inv_a)(c);
            if followed {
                c.cfg.field_mapping.unwrapping_algorithm = UnwrappingAlgorithm::Romeo;
                c.cfg.field_mapping.b0_estimation = B0EstimationMethod::WeightedAvg {
                    weighting: crate::utils::multi_echo::B0WeightType::assumed_decay_default(),
                };
                c.cfg.field_mapping.phase_offset_removal = true;
                c.cfg.bg_removal.algorithm = BgRemovalAlgorithm::Vsharp;
            }
        });
        for a in [SeparationAlgorithm::ChiSepIlsqr, SeparationAlgorithm::ChiSepMedi, SeparationAlgorithm::ChiSepNet] {
            for inv_a in [InversionAlgorithm::Ilsqr, InversionAlgorithm::Star] {
                for followed in [false, true] {
                    let case = toolbox(a, inv_a, followed);
                    assert!(conflicts(&case).is_empty(), "{a:?} {inv_a:?}: {:?}", conflicts(&case));
                    if followed {
                        let sti: Vec<_> = case.ids().into_iter().filter(|id| id.starts_with("sti.")).collect();
                        assert!(sti.is_empty(), "{a:?} {inv_a:?}: {sti:?}");
                    }
                }
            }
            let case = toolbox(a, InversionAlgorithm::Ilsqr, true);
            assert!(applying(&case).iter().any(|r| r.id == "sti.vsharp-radius"), "{a:?}");
            assert!(!applying(&case).iter().any(|r| r.id == "sti.echo-weights"), "{a:?}");
        }
        // Without χ-separation, and under DECOMPOSE (STAR-QSM with Laplacian, as STI Suite), they apply.
        assert!(Case::new(inversion(InversionAlgorithm::Ilsqr)).ids().contains(&"sti.unwrap-laplacian"));
        let decompose = Case::new(|c| { separation(SeparationAlgorithm::Decompose)(c); inversion(InversionAlgorithm::Star)(c) });
        assert!(applying(&decompose).iter().any(|r| r.id == "sti.unwrap-laplacian"));
        assert!(conflicts(&decompose).is_empty(), "{:?}", conflicts(&decompose));
    }

    /// `Unless` holds when the setting is not in play, and serialises with its tag.
    #[test]
    fn unless_holds_when_the_setting_is_not_in_play() {
        let cond = unless_sep(CHI_SEP_TOOLBOX);
        let plain = Case::new(|_| {});
        assert!(holds(&cond, &plain.input(), &plain.facts));
        let chisep = Case::new(separation(SeparationAlgorithm::ChiSepMedi));
        assert!(!holds(&cond, &chisep.input(), &chisep.facts));
        let decompose = Case::new(separation(SeparationAlgorithm::Decompose));
        assert!(holds(&cond, &decompose.input(), &decompose.facts));
        assert_eq!(serde_json::to_value(cond).unwrap()["on"], "unless");
    }

    /// The complex fit (MEDI's Fit_ppm_complex) suggests MEDI's region-growing unwrapper as the
    /// authors' setting: a reference rule, silent once followed or without the complex fit, and
    /// ROMEO stays the default unwrapper.
    #[test]
    fn complex_fit_suggests_medi_region_growing_as_the_authors_setting() {
        let id = "complex-fit.unwrap-region-growing";
        let r = RULES.iter().find(|r| r.id == id).unwrap();
        assert_eq!((r.kind(), r.basis, r.default), (Kind::Reference, Basis::ReferenceChoice, Choice("medi-region-growing")));
        let with = |b0: B0EstimationMethod, u: UnwrappingAlgorithm| Case::new(|c| {
            c.cfg.field_mapping.b0_estimation = b0;
            c.cfg.field_mapping.unwrapping_algorithm = u;
        }).ids().contains(&id);
        let fit = || B0EstimationMethod::ComplexFit(Default::default());
        assert!(with(fit(), UnwrappingAlgorithm::Romeo) && with(fit(), UnwrappingAlgorithm::Laplacian));
        assert!(!with(fit(), UnwrappingAlgorithm::MediRegionGrowing));
        assert!(!with(B0EstimationMethod::default(), UnwrappingAlgorithm::Romeo));
        assert_eq!(FieldMappingConfig::default().unwrapping_algorithm, UnwrappingAlgorithm::Romeo);
    }

    #[test]
    fn manufacturers_are_read_as_dcm2niix_writes_them() {
        use Manufacturer::*;
        for (name, m) in [
            ("Siemens", Siemens), ("SIEMENS", Siemens), ("Siemens Healthineers", Siemens),
            ("GE", Ge), ("GE MEDICAL SYSTEMS", Ge), ("General Electric", Ge), (" ge ", Ge),
            ("Philips", Philips), ("Philips Medical Systems", Philips), ("Philips Healthcare", Philips),
            ("Canon_MEC", Other), ("TOSHIBA", Other), ("UIH", Other), ("GEMINI", Other), ("", Other),
        ] {
            assert_eq!(Manufacturer::from_name(name), m, "{name:?}");
        }
    }

    /// The JSON form carries everything the evaluation uses.
    #[test]
    fn serialises_to_json() {
        let json = serde_json::to_value(RULES).unwrap();
        let star = json.as_array().unwrap().iter().find(|r| r["id"] == "star.padding").unwrap();
        assert_eq!(star["kind"], "reference");
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

    /// DECOMPOSE's per-echo QSM: recommended with DECOMPOSE (stated by its authors), satisfied
    /// by a per-echo reconstruction, and silent without DECOMPOSE.
    #[test]
    fn decompose_wants_a_qsm_per_echo() {
        let id = "decompose.qsm-per-echo";
        let r = RULES.iter().find(|r| r.id == id).unwrap();
        assert_eq!((r.strength, r.basis, r.default), (Strength::Recommended, Basis::Stated, Bool(true)));
        assert!(Case::new(separation(SeparationAlgorithm::Decompose)).ids().contains(&id));
        let on = Case::new(|c| { separation(SeparationAlgorithm::Decompose)(c); c.cfg.qsm_per_echo = true });
        assert!(!on.ids().contains(&id), "{:?}", on.ids());
        assert!(!Case::new(separation(SeparationAlgorithm::ChiSepIlsqr)).ids().contains(&id));
    }

    /// DECOMPOSE's unwrapping is suggested at 7 T (a Siemens 7 T reports 6.98 T) and not at 3 T,
    /// nor when it is already on, nor before the field strength is known.
    #[test]
    fn decompose_unwrapping_at_7t_only() {
        let id = "decompose.unwrap-7t";
        let at = |b0: Option<f64>, unwrap: bool| Case::new(|c| {
            separation(SeparationAlgorithm::Decompose)(c);
            c.sep.as_mut().unwrap().decompose.unwrap_phase = unwrap;
            c.facts.field_strength = b0;
        }).ids().contains(&id);
        assert!(at(Some(7.0), false) && at(Some(6.98), false) && at(Some(9.4), false));
        assert!(!at(Some(3.0), false) && !at(Some(2.89), false) && !at(Some(1.5), false));
        assert!(!at(Some(7.0), true));
        assert!(!at(None, false));
        let r = RULES.iter().find(|r| r.id == id).unwrap();
        assert_eq!((r.strength, r.basis), (Strength::Recommended, Basis::Measured));
        assert!(r.needs_scan());
        // The authors' default stays: DecomposeParams leaves unwrapping off.
        assert!(!crate::separation::DecomposeParams::default().unwrap_phase);
    }

    /// DECOMPOSE's echo-spacing χ bound is suggested at 7 T and not at 3 T, nor when it is already
    /// chosen, nor before the field strength is known; a fixed bound of any value reads as "ppm".
    #[test]
    fn decompose_chi_bound_at_7t_only() {
        use crate::separation::DecomposeChiBound as B;
        let id = "decompose.chi-bound-7t";
        let at = |b0: Option<f64>, bound: B| Case::new(|c| {
            separation(SeparationAlgorithm::Decompose)(c);
            c.sep.as_mut().unwrap().decompose.chi_bound = bound;
            c.facts.field_strength = b0;
        }).ids().contains(&id);
        assert!(at(Some(7.0), B::Ppm(0.5)) && at(Some(6.98), B::Ppm(0.3)) && at(Some(9.4), B::Ppm(0.5)));
        assert!(!at(Some(3.0), B::Ppm(0.5)) && !at(Some(1.5), B::Ppm(0.5)));
        assert!(!at(Some(7.0), B::EchoSpacing));
        assert!(!at(None, B::Ppm(0.5)));
        let r = RULES.iter().find(|r| r.id == id).unwrap();
        assert_eq!((r.strength, r.basis), (Strength::Recommended, Basis::Measured));
        assert_eq!(r.default, Value::Choice("echo-spacing"));
        assert!(r.needs_scan());
        // The authors' default stays: 0.5 ppm.
        assert_eq!(crate::separation::DecomposeParams::default().chi_bound, B::Ppm(0.5));
    }

    /// QSM.jl's iSMV variant suggests QSM.jl's tolerance and cap; its own params pass, the
    /// default variant raises nothing, and "no cap" reads as +∞ (null in JSON).
    #[test]
    fn ismv_qsmjl_suggests_qsmjl_values() {
        use crate::bgremove::{IsmvParams, IsmvVariant};
        let ids = ["ismv-qsmjl.tolerance", "ismv-qsmjl.max-iterations"];
        let with = |p: IsmvParams| Case::new(|c| {
            c.cfg.bg_removal.algorithm = BgRemovalAlgorithm::Ismv;
            c.cfg.bg_removal.ismv = p;
        });
        let ismv_ids = |c: &Case| -> Vec<&'static str> { c.ids().into_iter().filter(|id| id.starts_with("ismv")).collect() };
        let variant_only = with(IsmvParams { variant: IsmvVariant::QsmJl, ..Default::default() });
        assert!(ids.iter().all(|id| variant_only.ids().contains(id)), "{:?}", variant_only.ids());
        let f = evaluate(&variant_only.input(), &variant_only.facts);
        let cap = f.iter().find(|f| f.rule.id == ids[1]).unwrap();
        assert_eq!(cap.current, vec![Current::Number(f64::INFINITY)]);
        assert!(serde_json::to_value(cap).unwrap()["current"][0].is_null());
        assert!(ismv_ids(&with(IsmvParams::qsmjl())).is_empty());
        assert!(ismv_ids(&with(IsmvParams::default())).is_empty());
        let p = IsmvParams::qsmjl();
        for r in RULES.iter().filter(|r| ids.contains(&r.id)) {
            assert_eq!(r.strength, Strength::Reference, "{}", r.id);
            let v = if r.setting == "bg_removal.ismv.tol" { p.tol } else { p.max_iter.unwrap() as f64 };
            assert_eq!(r.default, Number(v), "{}", r.id);
        }
        // Not in play without iSMV.
        let off = Case::new(|c| c.cfg.bg_removal.ismv.variant = IsmvVariant::QsmJl);
        assert!(off.ids().is_empty(), "{:?}", off.ids());
    }

    /// LBV's tolerance is kept as given: at LBV.m's 0.01 nothing is said; another (the SOR
    /// solver's old 1e-6) gets LBV.m's value suggested; without LBV it is not in play.
    #[test]
    fn lbv_tolerance_is_suggested_not_rewritten() {
        let with = |tol: f64| Case::new(move |c| {
            c.cfg.bg_removal.algorithm = BgRemovalAlgorithm::Lbv;
            c.cfg.bg_removal.lbv.tol = tol;
        });
        assert!(!with(0.01).ids().contains(&"lbv.tolerance"));
        let old = with(1e-6);
        assert!(old.ids().contains(&"lbv.tolerance"), "{:?}", old.ids());
        let f = evaluate(&old.input(), &old.facts);
        let r = f.iter().find(|f| f.rule.id == "lbv.tolerance").unwrap();
        assert_eq!(r.current, vec![Current::Number(1e-6)]);
        assert_eq!(r.rule.strength, Strength::Reference);
        let off = Case::new(|c| c.cfg.bg_removal.lbv.tol = 1e-6);
        assert!(!off.ids().contains(&"lbv.tolerance"));
    }

    /// On a per-echo reconstruction the settings that combine or compare echoes are not in play.
    #[test]
    fn per_echo_takes_echo_combination_out_of_play() {
        let combining = ["field_mapping.b0_estimation", "field_mapping.phase_offset_removal",
            "field_mapping.phase_offset_sigma_mm", "field_mapping.bipolar_correction",
            "field_mapping.b0_estimation.weighting"];
        let mut c = Case::new(|c| c.cfg.field_mapping.phase_offset_removal = true);
        for p in combining {
            assert!(!current(&c.input(), &c.facts, p).unwrap().is_empty(), "{p}");
        }
        c.cfg.qsm_per_echo = true;
        for p in combining {
            assert!(current(&c.input(), &c.facts, p).unwrap().is_empty(), "{p}");
        }
        assert!(!current(&c.input(), &c.facts, "field_mapping.unwrapping_algorithm").unwrap().is_empty());
        assert_eq!(current(&c.input(), &c.facts, "qsm_per_echo").unwrap(), vec![Current::Bool(true)]);
        // STI Suite's echo weighting is about how echoes are combined: silent per echo.
        let combined = Case::new(inversion(InversionAlgorithm::Star));
        assert!(combined.ids().contains(&"sti.echo-weights"), "{:?}", combined.ids());
        let per_echo = Case { cfg: QsmPipelineConfig { qsm_per_echo: true, ..combined.cfg.clone() }, ..combined.clone() };
        assert!(!per_echo.ids().contains(&"sti.echo-weights"), "{:?}", per_echo.ids());
    }
}

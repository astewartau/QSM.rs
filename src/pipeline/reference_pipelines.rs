//! Toolbox pipelines: the original toolboxes' whole pipelines, each as the method choices that
//! start it.
//!
//! A host offers these so a user can start from a toolbox's pipeline (the MEDI toolbox's, STI
//! Suite's, QSMART's) instead of choosing a method and then taking its reference settings one by
//! one. A [`ReferencePipeline`] is only its starting choices ([`ReferencePipeline::choose`]): the
//! rest of the pipeline is the [`Strength::Reference`](super::rules::Strength::Reference) rules
//! those choices trigger, in [`RULES`](super::rules::RULES), so the catalogue and the rules can
//! never disagree.
//!
//! # Decisions
//!
//! - **A pipeline is its choices plus the reference rules they trigger, followed through.** A
//!   host sets the choices, then applies every `reference` rule whose trigger the pipeline has
//!   set, and repeats until nothing changes: MEDI triggers `medi.bg-pdf`, PDF triggers
//!   `pdf.field-complex-fit`, the complex fit triggers `complex-fit.unwrap-region-growing`. A
//!   rule whose trigger the pipeline did not set (one the user's other settings trigger) is not
//!   part of it. QSM.rs has no setter by setting path (the rules are data over a host's config),
//!   so the host does the applying; QSMxT's `qsmxt_engine::pipelines` is the reference host.
//! - **A pipeline starts from the defaults.** Since #293 each method's defaults are its
//!   reference values, so every reconstruction stage is reset to the defaults before the rules
//!   apply, whichever stages the pipeline touches: a value the user changed (MEDI's λ), or a
//!   method a pipeline chosen before set (DECOMPOSE), is not silently kept under the toolbox's
//!   name. What is not reconstruction (the input, which outputs to make, how to run) stays.
//! - **Reasoned rules are not part of a pipeline, unless the authors ran them.** A pipeline is
//!   what the authors ran. A reasoned rule its choices trigger (PDF's measured
//!   `pdf.cg-iterations`) is still reported by [`evaluate`](super::rules::evaluate), and the host
//!   shows it, but the pipeline keeps the authors' value. A reasoned rule that is itself what the
//!   authors ran, with their reason, is part of it: either as one of the pipeline's choices
//!   (DECOMPOSE's per-echo QSM, `decompose.qsm-per-echo`, a method choice that starts it), or,
//!   when it is a detail of the pipeline rather than a choice that starts it, in
//!   [`ReferencePipeline::reasoned`], whose rules a host follows like the reference rules (QSMART's
//!   Laplacian unwrapping, echo fit and unreduced mask, recommended with QSMART since
//!   2026-11-13). A departure from those is then a departure from the pipeline, not a different
//!   pipeline.
//! - **Scan-dependent rules apply on the scan.** A rule with a scan condition (MEDI's echo-time
//!   choice needs more than one echo) is part of the pipeline only when the scan is known to meet
//!   it, as everywhere in [`rules`](super::rules).
//! - **Parity is checked, not declared.** Whether a pipeline reproduces its toolbox is measured
//!   by QSM-CI's whole-pipeline parity track (QSMxT/QSM-CI#421), against each toolbox's own
//!   pipeline run end to end; a QSMxT release waits on it. A pipeline names its QSM-CI pair
//!   ([`ReferencePipeline::qsmci_pair`]) so a host can show that pair's measured status; nothing
//!   here claims a pipeline reproduces its toolbox, and nothing lists what it does not.
//! - **Differences are deliberate, and each has its reason** (23 Nov 2026, decision 217). A
//!   pipeline lists in [`ReferencePipeline::differences`] only what it does differently from the
//!   toolbox on purpose, each with why, and [`DIFFERENCES`] what pipelines share (except where
//!   the toolbox already does it: the centre frequency, decisions 31 and 149). A gap, to be closed, goes
//!   in [`ReferencePipeline::note`] with the issue that tracks it. Rounding that the toolbox's
//!   own single-precision or FFT arithmetic introduces (STI Suite's V-SHARP mask, QSMART's
//!   Laplacian offsets) is not copied; it is listed as a difference, and QSM-CI sets its parity
//!   limits from it.
//! - **Which pipelines.** A toolbox whose own code runs a whole pipeline around its method, with
//!   rules for it: the MEDI toolbox, STI Suite (its iLSQR and STAR-QSM), QSMART, the HEIDI
//!   package, the χ-separation toolbox (its three methods), DECOMPOSE, and UK Biobank's QSM
//!   pipeline (its own script around STI Suite's functions). A method that is only
//!   trained on a pipeline (QSMnet's V-SHARP) keeps its reference settings as a method's, not as
//!   a pipeline of its own.
//! - **UK Biobank is chosen by name, never by its method.** Its inversion is STI Suite's iLSQR,
//!   which also starts STI Suite's own pipeline, so iLSQR cannot be its trigger. Its trigger is
//!   what only UK Biobank does, its phase-reliability mask trimming
//!   ([`BgRemovalConfig::ukb_mask_trim`](super::config::BgRemovalConfig::ukb_mask_trim)): the
//!   `ukb.*` rules follow from it, and STI Suite's GUI rules that UK Biobank departs from are
//!   scoped out of it. Choosing iLSQR alone still gives STI Suite's (13 Nov 2026, decisions 145,
//!   146 and 148).
//! - **UK Biobank's mask is HD-BET's, on purpose.** UK Biobank masks with its T1 structural brain
//!   mask (BET) registered to the SWI, which a run from the SWI alone does not have. On UK Biobank
//!   data, HD-BET on the magnitude came closest to that mask among the SWI-only masks: about 3.4 mm
//!   higher at the brain's inferior edge, and otherwise close. That HD-BET ran in 192×192×96
//!   sliding windows (step 0.5, no test-time augmentation) on the inhomogeneity-corrected
//!   magnitude, followed by signal-gated erosion (threshold 0.8, depth cap 5, 1 global erosion,
//!   bias sigma 12, smallest component 1000), QSMxT's `hd-bet` mask preset of October 2026; the
//!   preset has since run HD-BET on the whole volume, whose mask was 0.5-1 % smaller, mostly at
//!   the inferior edge, where the comparison's head-size bias came from. So the pipeline follows
//!   the measured rules `ukb.mask-hd-bet` and `ukb.mask-hd-bet-window` (the sliding window; in
//!   [`ReferencePipeline::reasoned`]), its `differences` say so, and
//!   it runs a deep-learning model ([`ReferencePipeline::models`]) (20 Nov 2026, decision 202).

use serde::Serialize;

use super::rules::Value::{self, Bool, Choice};
use crate::catalogue::citations::*;
use crate::catalogue::Citation;

/// A deliberate difference between a pipeline and its toolbox's own: what this pipeline does, and
/// why. Plain user-facing text, one sentence each, shown with the pipeline (QSMxT's pipeline
/// pages and Pipeline tab).
#[derive(Serialize, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Difference {
    /// What this pipeline does that the toolbox does not, or does otherwise.
    pub what: &'static str,
    /// Why.
    pub why: &'static str,
}

/// A difference that several pipelines share, because every QSM.rs reconstruction does it, and
/// the pipelines it does not apply to (their toolbox already does the same).
#[derive(Serialize, Clone, Copy, Debug, PartialEq, Eq)]
pub struct SharedDifference {
    pub difference: Difference,
    /// Ids of the pipelines whose toolbox already does this, so it is no difference there.
    pub not_for: &'static [&'static str],
}

/// Differences that every pipeline has unless its id is in `not_for`; shown with a pipeline
/// before its own ([`ReferencePipeline::all_differences`]).
pub static DIFFERENCES: &[SharedDifference] = &[
    // Decisions 31 and 149: a host sets `ScanMetadata::field_strength` from ImagingFrequency
    // (QSMxT does, for every run), toolbox pipelines and QSM-CI's references included. Not a
    // difference where the toolbox itself takes the centre frequency: the MEDI toolbox reads it
    // from the DICOM (`Read_DICOM.m`: CF = ImagingFrequency). The χ-separation toolbox v1.1.3's
    // `Chisep_script.m` takes "CF (central frequency)" as a user input (l. 60), with the example
    // `CF = 123200000; in Hz unit` (l. 63): a Siemens 3 T scanner's actual frequency, not γ·3 T
    // (127.7 MHz). Its separation steps convert with it (`params.CF = CF`, l. 154 and 168), and
    // so does χ-sepnet (`chi_sepnet_general(..., CF, ...)`, l. 194). The script's QSM step for
    // χ-separation-iLSQR and -MEDI takes a separate `B0_strength` instead (l. 145), which is
    // CHISEP_QSM_FIELD below.
    SharedDifference {
        difference: Difference {
            what: "Where the original toolbox converts phase to ppm with the nominal field strength (such as 3 T), this \
                   pipeline uses the scanner's centre frequency (ImagingFrequency) whenever the scan's metadata records it.",
            why: "The centre frequency is the field the scanner actually ran at, while the nominal value can be several per \
                  cent off (Siemens' 3 T systems run at about 2.89 T), and every susceptibility value scales with it.",
        },
        not_for: &["medi-toolbox", "chi-sep-ilsqr", "chi-sep-medi", "chi-sepnet"],
    },
];

/// STI Suite's GUI takes B0 along the slice axis unless its user types a direction in
/// (ilsqr.rs and star.rs docs, *B0 direction*); QSM.rs's iLSQR and STAR-QSM take the scan's.
const STI_B0_DIRECTION: Difference = Difference {
    what: "The B0 direction is taken from the scan's orientation.",
    why: "STI Suite's GUI uses B0 along the slice axis unless its user types a direction in, and its manual asks for the \
          scan's direction on oblique scans; on axial scans the two are the same.",
};

/// STI Suite's V_SHARP drops a few interior voxels from its eroded mask through single-precision
/// FFT rounding; QSM.rs's test is exact and keeps them (vsharp.rs docs, *STI's fit test*;
/// QSM.rs#463). On QSM-CI's s005001 Prisma scan: 4 of 1,276,574 voxels, the whole χ gap of
/// QSM-CI's `toolbox-pipeline-sti` pair (NRMSE 0.30 %).
const STI_VSHARP_MASK: Difference = Difference {
    what: "V-SHARP's eroded mask is computed exactly, so it keeps a few isolated voxels inside the brain that STI Suite \
           leaves out.",
    why: "STI Suite leaves them out through rounding in its single-precision FFTs, which no other FFT matches and which \
          does not depend on the data; on QSM-CI's in-vivo scan it was 4 of about 1.3 million voxels.",
};

/// One toolbox pipeline.
#[derive(Serialize, Clone, Copy, Debug, PartialEq)]
pub struct ReferencePipeline {
    /// Stable id, in kebab-case (`medi-toolbox`): a host's CLI value and config label.
    pub id: &'static str,
    /// What a host calls it ("MEDI toolbox"); with a method when one toolbox has several
    /// ("STI Suite · iLSQR").
    pub name: &'static str,
    /// The toolbox and the version its pipeline is taken from ("MEDI toolbox 2024.11.26"), as a
    /// methods sentence names it ("Processing followed the MEDI toolbox 2024.11.26 pipeline").
    pub toolbox: &'static str,
    /// A known gap between this pipeline and the toolbox's own, in a sentence or two a host shows
    /// with the pipeline's details (QSMxT's Pipeline tab), or `None`. Plain user-facing text: what
    /// the toolbox does that this pipeline does not yet, and what it does instead. A gap is noted
    /// here only until it is closed, and the issue that tracks it is cited where the note is set.
    /// A deliberate difference goes in [`differences`](Self::differences) instead.
    pub note: Option<&'static str>,
    /// The issue that tracks the [`note`](Self::note)'s gap (`"astewartau/QSM.rs#491"`), set
    /// exactly when there is a note; a host links it after the note.
    pub note_issue: Option<&'static str>,
    /// What this pipeline does differently from the toolbox on purpose, each with its reason
    /// (besides the [`DIFFERENCES`] it shares; [`all_differences`](Self::all_differences) gives both). Only deliberate differences: never a
    /// claim that the rest reproduces the toolbox; QSM-CI measures that
    /// ([`qsmci_pair`](Self::qsmci_pair)).
    pub differences: &'static [Difference],
    /// When to choose it, in one plain sentence.
    pub use_when: &'static str,
    /// The papers to cite for it, the main method's first.
    pub citations: &'static [Citation],
    /// The QSM-CI parity pair (`parity/thresholds.yml`) that checks this pipeline against the
    /// toolbox's own, whose measured status a host shows; `None` where QSM-CI has none yet
    /// (`sti-suite-star`: QSMxT/QSM-CI#469; `chi-sepnet`: QSMxT/QSM-CI#470).
    pub qsmci_pair: Option<&'static str>,
    /// The issue that asks QSM-CI for the missing pair (`"QSMxT/QSM-CI#469"`), set exactly when
    /// [`qsmci_pair`](Self::qsmci_pair) is `None`.
    pub qsmci_pair_issue: Option<&'static str>,
    /// The method choices that start it, as rule settings and values ([`SETTINGS`](super::rules::SETTINGS)).
    pub choose: &'static [(&'static str, Value)],
    /// Why each choice in [`choose`](Self::choose), in the same order: one plain sentence (no
    /// full stop, as a rule's reason) a host shows on the choice's row.
    pub choose_reasons: &'static [&'static str],
    /// Where the pipeline is written down, for developers: the toolbox's script and lines, the
    /// QSM.rs docs that record what was probed. Not for users; see [`based_on`](Self::based_on).
    pub source: &'static str,
    /// What it is based on, for users: the toolbox's pipeline and its paper, in one short line
    /// ("UK Biobank's QSM pipeline (UKBiobank_QSM.m) · Wang et al. 2022").
    pub based_on: &'static str,
    /// Reasoned rules (by id) that are also what the authors ran: a host follows them as it
    /// follows the reference rules the choices trigger, in [`RULES`](super::rules::RULES) order.
    pub reasoned: &'static [&'static str],
}

/// `Chisep_script.m` v1.1.3 l. 145 gives its QSM (STI Suite's `QSM_iLSQR`, the separation's
/// `option_data.qsm`) `'B0', B0_strength`, a field strength the user types apart from `CF`
/// (l. 60; no example value), while the separation converts with `CF`. QSM.rs converts both with
/// the one field strength the host sets from the centre frequency.
const CHISEP_QSM_FIELD: Difference = Difference {
    what: "The QSM that the separation starts from is converted with the same centre frequency as the separation itself.",
    why: "The toolbox's script gives that QSM step (STI Suite's iLSQR) a field strength typed separately from its centre \
          frequency, which can be the nominal one; one value keeps the two steps consistent.",
};

/// QSMART's deliberate differences (QSM.rs `pipeline::qsmart` docs, *Agreement with the original*).
const QSMART_DIFFERENCES: &[Difference] = &[
    Difference {
        what: "One gyromagnetic ratio is used throughout.",
        why: "QSMART converts the field to rad/s with 2.675e8 rad/s/T and STI Suite's iLSQR inside it converts back with \
              2π·42.575 MHz/T, which lowers its susceptibility by 0.002 %; the mismatch is not part of the method.",
    },
    Difference {
        what: "Each echo's Laplacian-unwrapped phase has no arbitrary constant added.",
        why: "QSMART's Laplacian unwrapping leaves each echo a small constant set by FFT rounding, which cannot be matched \
              and is not part of the method; through the echo fit it moved the total field by 0.24 % on QSM-CI's in-vivo scan.",
    },
];

/// DECOMPOSE follows its paper (decision 15(1), 10 Nov 2026): QSM.rs `separation::decompose`
/// docs, *Reference pipeline*.
const DECOMPOSE_DIFFERENCES: &[Difference] = &[Difference {
    what: "The steps before the fit follow the paper (Laplacian unwrapping, V-SHARP, STAR-QSM of each echo) at the \
           functions' own settings, without the extra settings of the authors' analysis script.",
    why: "The paper is the method as published and gives no settings for these steps; the authors' analysis script, which \
          pads STAR-QSM by 12 voxels and runs V-SHARP with a 25 mm kernel on each echo, is a separate choice of settings.",
}];

/// UK Biobank's mask (decision 202, module docs, *UK Biobank's mask is HD-BET's, on purpose*).
const UKB_DIFFERENCES: &[Difference] = &[Difference {
    what: "The brain mask is HD-BET on the magnitude, in 192×192×96 sliding windows, followed by signal-gated erosion.",
    why: "UK Biobank masks with its T1 structural brain mask registered to the SWI, which a run from the SWI alone does not \
          have; on UK Biobank data HD-BET came closest to it of the masks an SWI alone can give (about 3.4 mm higher at the \
          brain's inferior edge, otherwise close).",
}];

/// The catalogue, in the order a host lists it.
pub static REFERENCE_PIPELINES: &[ReferencePipeline] = &[
    ReferencePipeline {
        id: "medi-toolbox",
        name: "MEDI toolbox",
        toolbox: "MEDI toolbox 2024.11.26",
        // README.m:98 fits bipolar data with Fit_ppm_complex_bipolar; QSMxT can't tell a scan is
        // bipolar yet, and bipolar correction isn't supported with the complex fit
        // (QSMxT/QSMxT#564).
        note: Some(
            "For bipolar readouts, the MEDI toolbox uses a bipolar echo fit, which is not supported yet. \
             This pipeline always uses the monopolar fit.",
        ),
        differences: &[],
        use_when: "You want results comparable with the MEDI toolbox's own pipeline: its complex echo fit, PDF and MEDI.",
        citations: &[LIU2011MEDI, LIU2011PDF],
        qsmci_pair: Some("toolbox-pipeline-medi-cornell"),
        note_issue: Some("QSMxT/QSMxT#564"),
        qsmci_pair_issue: None,
        based_on: "The MEDI toolbox's example pipeline (README.m) · Liu et al. 2011",
        choose: &[("inversion.algorithm", Choice("medi"))],
        choose_reasons: &["The MEDI toolbox's pipeline inverts the field with MEDI"],
        source: "MEDI toolbox 2024.11.26 README.m:80-144",
        reasoned: &[],
    },
    ReferencePipeline {
        id: "sti-suite-ilsqr",
        name: "STI Suite · iLSQR",
        toolbox: "STI Suite 3.0",
        note: None,
        differences: &[STI_B0_DIRECTION, STI_VSHARP_MASK],
        use_when: "You want results comparable with STI Suite's QSM_GUI run with iLSQR.",
        citations: &[LI2015, WU2012],
        qsmci_pair: Some("toolbox-pipeline-sti"),
        note_issue: None,
        qsmci_pair_issue: None,
        based_on: "STI Suite's QSM_GUI, iLSQR path · Li et al. 2015",
        choose: &[("inversion.algorithm", Choice("ilsqr"))],
        choose_reasons: &["STI Suite's GUI inverts the field with iLSQR"],
        source: "STI Suite 3.0 QSM_GUI, probed headless (QSM.rs ilsqr.rs and vsharp.rs docs)",
        reasoned: &[],
    },
    ReferencePipeline {
        id: "sti-suite-star",
        name: "STI Suite · STAR-QSM",
        toolbox: "STI Suite 3.0",
        // STI Suite's one-button STAR-QSM path masks with `Getmasking`, P-code that calls no
        // traceable function (rules.rs, the `sti.mask-*` rules are iLSQR's only); not ported yet
        // (astewartau/QSM.rs#493).
        note: Some(
            "STI Suite's STAR-QSM path makes its brain mask with its own routine, which is not available yet. \
             This pipeline uses QSMxT's default mask instead.",
        ),
        differences: &[STI_B0_DIRECTION, STI_VSHARP_MASK],
        use_when: "You want results comparable with STI Suite's QSM_GUI run with STAR-QSM.",
        citations: &[WEI2015STAR, WU2012],
        // QSM-CI has a pair for STI Suite's iLSQR pipeline (`toolbox-pipeline-sti`) and for STAR-QSM
        // alone (`star`), none yet for this pipeline end to end (QSMxT/QSM-CI#469).
        qsmci_pair: None,
        note_issue: Some("astewartau/QSM.rs#493"),
        qsmci_pair_issue: Some("QSMxT/QSM-CI#469"),
        based_on: "STI Suite's QSM_GUI, STAR-QSM path · Wei et al. 2015",
        choose: &[("inversion.algorithm", Choice("star"))],
        choose_reasons: &["STI Suite's GUI inverts the field with STAR-QSM"],
        source: "STI Suite 3.0 QSM_GUI, probed headless (QSM.rs star.rs and vsharp.rs docs)",
        reasoned: &[],
    },
    ReferencePipeline {
        id: "qsmart",
        name: "QSMART",
        toolbox: "QSMART",
        note: None,
        differences: QSMART_DIFFERENCES,
        use_when: "You want results comparable with QSMART's own pipeline, which treats the cortex next to the skull and the veins separately.",
        citations: &[YAGHMAIE2021],
        qsmci_pair: Some("toolbox-pipeline-qsmart"),
        note_issue: None,
        qsmci_pair_issue: None,
        based_on: "The QSMART toolbox (QSMART.m) · Yaghmaie et al. 2021",
        choose: &[("inversion.algorithm", Choice("qsmart"))],
        choose_reasons: &["QSMART inverts the total field in two stages, one for the brain's edge and veins and one for the rest"],
        source: "QSMART QSMART.m; Yaghmaie et al. 2021 NeuroImage 231:117701",
        // Recommended with QSMART (QSM.rs pipeline::qsmart docs), and what its authors ran.
        reasoned: &["qsmart.mask-no-erosion", "qsmart.unwrap-laplacian", "qsmart.field-echofit"],
    },
    ReferencePipeline {
        id: "heidi",
        name: "HEIDI package",
        toolbox: "HEIDI package (qsm_heidi 4592631)",
        note: None,
        differences: &[],
        use_when: "You want results comparable with the HEIDI package: LSQR, then HEIDI, from a SHARP local field.",
        citations: &[SCHWESER2012, SCHWESER2010],
        // The package covers only local field to χ, so its pair is the method pair (`heidi`); the
        // rest of the pipeline is QSMxT wiring, tested there.
        qsmci_pair: Some("heidi"),
        note_issue: None,
        qsmci_pair_issue: None,
        based_on: "The HEIDI package (qsm_heidi), LSQR then HEIDI · Schweser et al. 2012",
        choose: &[("inversion.algorithm", Choice("lsqr")), ("inversion.apply_heidi", Bool(true))],
        choose_reasons: &["The HEIDI package first inverts the local field with LSQR, a least-squares fit with almost no regularisation, which keeps the measured data that HEIDI builds on", "HEIDI then fills in the ill-conditioned part of the data from its surroundings, which reduces streaking"],
        source: "qsm_heidi makefile and def.sh (4592631); Schweser et al. 2012 NeuroImage 62:2083",
        reasoned: &[],
    },
    ReferencePipeline {
        id: "chi-sep-ilsqr",
        name: "χ-separation toolbox · iLSQR",
        toolbox: "χ-separation toolbox 1.1.3",
        note: None,
        differences: &[CHISEP_QSM_FIELD],
        use_when: "You want paramagnetic and diamagnetic maps comparable with the χ-separation toolbox's script, solved with iLSQR, and have an R2′ map or a multi-echo spin-echo scan.",
        citations: &[SHIN2021CHISEP],
        qsmci_pair: Some("toolbox-pipeline-chisep-ilsqr"),
        note_issue: None,
        qsmci_pair_issue: None,
        based_on: "The χ-separation toolbox's script (Chisep_script.m) with iLSQR · Shin et al. 2021",
        choose: &[("separation.algorithm", Choice("chi-sep-ilsqr"))],
        choose_reasons: &["The χ-separation toolbox's script solves the separation with iLSQR"],
        source: "χ-separation toolbox v1.1.3 Chisep_script.m",
        reasoned: &[],
    },
    ReferencePipeline {
        id: "chi-sep-medi",
        name: "χ-separation toolbox · MEDI",
        toolbox: "χ-separation toolbox 1.1.3",
        note: None,
        differences: &[CHISEP_QSM_FIELD],
        use_when: "You want paramagnetic and diamagnetic maps comparable with the χ-separation toolbox's script, solved with MEDI, and have an R2′ map or a multi-echo spin-echo scan.",
        citations: &[SHIN2021CHISEP],
        qsmci_pair: Some("toolbox-pipeline-chisep-medi"),
        note_issue: None,
        qsmci_pair_issue: None,
        based_on: "The χ-separation toolbox's script (Chisep_script.m) with MEDI · Shin et al. 2021",
        choose: &[("separation.algorithm", Choice("chi-sep-medi"))],
        choose_reasons: &["The χ-separation toolbox's script solves the separation with MEDI"],
        source: "χ-separation toolbox v1.1.3 Chisep_script.m",
        reasoned: &[],
    },
    ReferencePipeline {
        id: "chi-sepnet",
        name: "χ-separation toolbox · χ-sepnet",
        toolbox: "χ-separation toolbox 1.1.3",
        note: None,
        differences: &[],
        use_when: "You want paramagnetic and diamagnetic maps from the χ-separation toolbox's network, from 3 T gradient-echo data alone or with a measured R2′.",
        citations: &[KIM2025CHISEPNET, SHIN2021CHISEP],
        // QSM-CI's `chisepnet` pair checks the network's route (`chi_sepnet_general`), not the
        // toolbox's field chain before it; there is no whole-pipeline pair yet (QSMxT/QSM-CI#470).
        qsmci_pair: None,
        note_issue: None,
        qsmci_pair_issue: Some("QSMxT/QSM-CI#470"),
        based_on: "The χ-separation toolbox's script (Chisep_script.m) with χ-sepnet · Kim et al. 2025",
        choose: &[("separation.algorithm", Choice("chi-sepnet")), ("separation.chi_sepnet.route", Choice("toolbox"))],
        choose_reasons: &["The χ-separation toolbox's script separates the sources with its χ-sepnet network", "χ-sepnet runs as the toolbox runs it, turning the inputs to B0 along z and resampling them to 1 mm"],
        source: "χ-separation toolbox v1.1.3 Chisep_script.m; Kim et al. 2025 (χ-sepnet)",
        reasoned: &[],
    },
    ReferencePipeline {
        id: "decompose",
        name: "DECOMPOSE",
        toolbox: "DECOMPOSE",
        note: None,
        // A QSM per echo is what its authors ran (§2.2, §4.1: a TE-dependent QSM input is
        // necessary), so it is one of the pipeline's choices, though its rule
        // (`decompose.qsm-per-echo`) is a reasoned one: see the decision above.
        differences: DECOMPOSE_DIFFERENCES,
        use_when: "You want paramagnetic and diamagnetic maps from multi-echo gradient-echo data alone, with DECOMPOSE as its paper describes it.",
        citations: &[CHEN2021DECOMPOSE],
        qsmci_pair: Some("toolbox-pipeline-decompose"),
        note_issue: None,
        qsmci_pair_issue: None,
        based_on: "The DECOMPOSE paper · Chen et al. 2021",
        choose: &[("separation.algorithm", Choice("decompose")), ("qsm_per_echo", Bool(true))],
        choose_reasons: &["DECOMPOSE separates the sources by fitting a three-compartment model to the multi-echo signal", "DECOMPOSE's authors reconstruct a susceptibility map for each echo, and state that this is necessary"],
        source: "Chen et al. 2021 NeuroImage 242:118477 (DECOMPOSE), §2.2, §3.6, §4.1",
        reasoned: &[],
    },
    ReferencePipeline {
        id: "uk-biobank",
        name: "UK Biobank",
        toolbox: "UK Biobank QSM",
        // The mask is a deliberate difference (UKB_DIFFERENCES, decision 202), not a gap. The gap:
        // on UK Biobank's own geometry (256×288×48, 0.8×0.8×3 mm, B0 tilt 12–26°, 2 echoes) the
        // local field after V-SHARP is 0.05–0.08 % NRMSE from STI Suite's and χ 0.28–0.41 %, about
        // 30–40× QSM-CI's limit on its near-isotropic scan (astewartau/QSM.rs#491).
        note: Some(
            "On UK Biobank's own scans (3 mm slices, oblique), V-SHARP's local field differs slightly from STI Suite's, \
             which iLSQR amplifies to about 0.3-0.4 % of the susceptibility map.",
        ),
        differences: UKB_DIFFERENCES,
        use_when: "You want results comparable with UK Biobank's released susceptibility maps, from the SWI scan alone.",
        citations: &[WANG2022, LI2015, ISENSEE2019],
        // Checked on QSM-CI's near-isotropic, non-oblique bridge scan only; a UK Biobank-geometry case
        // (anisotropic, oblique, 2 echoes) is QSMxT/QSM-CI#467.
        qsmci_pair: Some("toolbox-pipeline-ukb"),
        note_issue: Some("astewartau/QSM.rs#491"),
        qsmci_pair_issue: None,
        based_on: "UK Biobank's QSM pipeline (UKBiobank_QSM.m) · Wang et al. 2022",
        choose: &[("inversion.algorithm", Choice("ilsqr")), ("bg_removal.ukb_mask_trim", Bool(true))],
        choose_reasons: &["UK Biobank inverts its trimmed local field with STI Suite's iLSQR", "UK Biobank trims its mask to voxels with reliable phase, before and after background removal"],
        source: "UK Biobank UKBiobank_QSM.m (git.fmrib.ox.ac.uk/cwang/uk_biobank_qsm_pipeline); Wang et al. 2022 Nat. Neurosci.; \
                 HD-BET in place of its registered T1 brain mask (decision 202)",
        reasoned: &["ukb.mask-hd-bet", "ukb.mask-hd-bet-window"],
    },
];

impl ReferencePipeline {
    /// Its differences from the toolbox: the shared ones that apply to it ([`DIFFERENCES`]),
    /// then its own ([`differences`](Self::differences)).
    pub fn all_differences(&self) -> Vec<Difference> {
        DIFFERENCES.iter().filter(|d| !d.not_for.contains(&self.id)).map(|d| d.difference)
            .chain(self.differences.iter().copied()).collect()
    }

    /// The deep-learning models it runs (registry ids, `hd-bet`): those its choices and its
    /// reasoned rules choose, so a host built without them can say so before the pipeline is
    /// chosen rather than fail inside a run.
    pub fn models(&self) -> Vec<&'static str> {
        use super::config::{BgRemovalAlgorithm, InversionAlgorithm, MaskOp, SeparationAlgorithm};
        let reasoned = self.reasoned.iter().filter_map(|id| super::rules::RULES.iter().find(|r| r.id == *id))
            .map(|r| (r.setting, r.default));
        self.choose.iter().copied().chain(reasoned).filter_map(|(setting, value)| {
            let Choice(id) = value else { return None };
            match setting {
                "masking.generator" => [MaskOp::HdBet(Default::default()), MaskOp::Rs2Net(Default::default()), MaskOp::MindGrab(Default::default())]
                    .into_iter().find(|op| op.id() == id).and_then(|op| op.dl_model_id()),
                "inversion.algorithm" => InversionAlgorithm::VARIANTS.iter().find(|a| a.id() == id).and_then(|a| a.dl_model_id()),
                "bg_removal.algorithm" => BgRemovalAlgorithm::VARIANTS.iter().find(|a| a.id() == id).and_then(|a| a.dl_model_id()),
                "separation.algorithm" => SeparationAlgorithm::VARIANTS.iter().find(|a| a.id() == id).and_then(|a| a.dl_model_id()),
                _ => None,
            }
        }).collect()
    }
}

/// The pipeline with this id.
pub fn reference_pipeline(id: &str) -> Option<&'static ReferencePipeline> {
    REFERENCE_PIPELINES.iter().find(|p| p.id == id)
}

#[cfg(test)]
mod tests {
    use super::super::rules::{Condition, Strength, Test, RULES, SETTINGS};
    use super::*;

    /// QSMART's pipeline follows its recommended group (Laplacian unwrapping, the echo fit, no
    /// erosion), which is what its authors ran.
    #[test]
    fn qsmart_follows_its_recommended_group() {
        let p = reference_pipeline("qsmart").unwrap();
        for id in ["qsmart.mask-no-erosion", "qsmart.unwrap-laplacian", "qsmart.field-echofit"] {
            let r = RULES.iter().find(|r| r.id == id).unwrap();
            assert_eq!(r.strength, Strength::Recommended, "{id}");
            assert!(p.reasoned.contains(&id), "{id}");
        }
    }

    /// Ids are unique kebab-case, every choice is a rule setting, every pipeline's choices
    /// trigger at least one of the authors' settings (else it would be a method, not a pipeline),
    /// and its reasoned rules are reasoned rules its choices trigger.
    #[test]
    fn pipelines_are_well_formed() {
        let mut ids = std::collections::HashSet::new();
        for p in REFERENCE_PIPELINES {
            assert!(ids.insert(p.id), "duplicate pipeline id {}", p.id);
            assert!(p.id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'), "{}: not kebab-case", p.id);
            assert!(!p.choose.is_empty(), "{}: no choices", p.id);
            assert!(p.note.is_none_or(|n| !n.trim().is_empty() && n.ends_with('.')), "{}: note must be a sentence", p.id);
            for (setting, _) in p.choose {
                assert!(SETTINGS.iter().any(|(s, _)| s == setting), "{}: {setting} not in SETTINGS", p.id);
            }
            let triggered = RULES.iter().filter(|r| r.strength == Strength::Reference).any(|r| {
                let Condition::Setting { setting, is: Test::OneOf { values } } = r.trigger() else { return false };
                p.choose.iter().any(|(s, v)| s == setting && values.contains(v))
            });
            assert!(triggered, "{}: its choices trigger no reference rule", p.id);
        }
        for p in REFERENCE_PIPELINES {
            for id in p.reasoned {
                let r = RULES.iter().find(|r| r.id == *id).unwrap_or_else(|| panic!("{}: no rule {id}", p.id));
                assert_ne!(r.strength, Strength::Reference, "{}: {id} is a reference rule, followed anyway", p.id);
                let Condition::Setting { setting, is: Test::OneOf { values } } = r.trigger() else { panic!("{id}") };
                assert!(p.choose.iter().any(|(s, v)| s == setting && values.contains(v)), "{}: its choices do not trigger {id}", p.id);
            }
        }
        assert_eq!(reference_pipeline("medi-toolbox").map(|p| p.name), Some("MEDI toolbox"));
        assert!(reference_pipeline("nope").is_none());
    }

    fn is_sentence(s: &str) -> bool {
        s.trim() == s && !s.is_empty() && s.ends_with('.') && !s[..s.len() - 1].contains(". ")
            && s.starts_with(|c: char| !c.is_lowercase())
    }

    /// Every pipeline has a one-sentence use-when, citations, its differences (each a sentence
    /// with a reason), and a QSM-CI pair slug where it has one; the global differences too.
    #[test]
    fn every_pipeline_has_its_documentation_fields() {
        let check = |id: &str, d: &Difference| {
            assert!(is_sentence(d.what), "{id}: difference {:?} is not a sentence", d.what);
            assert!(is_sentence(d.why), "{id}: reason {:?} is not a sentence", d.why);
        };
        assert!(!DIFFERENCES.is_empty());
        for d in DIFFERENCES {
            check("all pipelines", &d.difference);
            for id in d.not_for {
                assert!(reference_pipeline(id).is_some(), "not_for names {id:?}, which is no pipeline");
            }
        }
        for p in REFERENCE_PIPELINES {
            assert!(is_sentence(p.use_when), "{}: use_when {:?} is not one sentence", p.id, p.use_when);
            assert!(!p.citations.is_empty(), "{}: no citations", p.id);
            for d in p.differences { check(p.id, d); }
            assert_eq!(p.choose_reasons.len(), p.choose.len(), "{}: one reason per choice", p.id);
            for r in p.choose_reasons {
                assert_eq!(super::super::rules::reason_problem(r), None, "{}: choice reason {r:?}", p.id);
            }
            assert_eq!(p.note.is_some(), p.note_issue.is_some(), "{}: a note needs its issue, and only a note", p.id);
            assert_eq!(p.qsmci_pair.is_none(), p.qsmci_pair_issue.is_some(), "{}: a missing pair needs its issue", p.id);
            for issue in p.note_issue.into_iter().chain(p.qsmci_pair_issue) {
                let (repo, n) = issue.split_once('#').unwrap_or_else(|| panic!("{}: {issue:?}", p.id));
                assert!(repo.split('/').count() == 2 && n.parse::<u32>().is_ok(), "{}: {issue:?} is not owner/repo#n", p.id);
            }
            // based_on is for users: short, no file paths, no QSM.rs docs, no repeated difference.
            let b = p.based_on;
            assert!(!b.is_empty() && b.len() <= 100 && b.contains(" · "), "{}: based_on {b:?}", p.id);
            for bad in [".rs", "probed", "docs", "/", "difference", ":"] {
                assert!(!b.contains(bad), "{}: based_on {b:?} contains {bad:?}", p.id);
            }
            assert!(p.note.is_none_or(is_sentence_or_two), "{}: note", p.id);
            if let Some(pair) = p.qsmci_pair {
                assert!(!pair.is_empty() && pair.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'),
                    "{}: pair {pair:?} is not a slug", p.id);
            }
            let text = [p.use_when, p.based_on].into_iter().chain(p.note).chain(p.differences.iter().flat_map(|d| [d.what, d.why]));
            for t in text {
                // Parity is QSM-CI's to report (decision 217): no claims in the text.
                assert!(!t.to_lowercase().contains("reproduce"), "{}: {t:?} claims parity", p.id);
            }
        }
    }

    fn is_sentence_or_two(n: &str) -> bool {
        n.trim() == n && !n.is_empty() && n.ends_with('.')
    }

    /// The centre-frequency difference is not listed where the toolbox already takes the centre
    /// frequency (MEDI, the χ-separation toolbox), and is listed first elsewhere.
    #[test]
    fn shared_differences_skip_the_pipelines_they_do_not_apply_to() {
        let cf = DIFFERENCES[0].difference;
        assert!(!reference_pipeline("medi-toolbox").unwrap().all_differences().contains(&cf));
        let ukb = reference_pipeline("uk-biobank").unwrap().all_differences();
        assert_eq!(ukb.first(), Some(&cf));
        assert_eq!(ukb.len(), 1 + reference_pipeline("uk-biobank").unwrap().differences.len());
    }

    /// The pipelines that run a deep-learning model: χ-sepnet's, and UK Biobank's, which masks
    /// with HD-BET. The others run none.
    #[test]
    fn the_models_a_pipeline_runs() {
        for p in REFERENCE_PIPELINES {
            let want: &[&str] = match p.id {
                "uk-biobank" => &["hd-bet"],
                "chi-sepnet" => &["chi-sepnet"],
                _ => &[],
            };
            assert_eq!(p.models(), want, "{}", p.id);
        }
    }

    /// Choosing iLSQR alone is STI Suite's pipeline; UK Biobank's own choices replace the STI
    /// GUI's settings it departs from, and keep what STI Suite's functions themselves do.
    #[test]
    fn uk_biobank_is_not_chosen_by_ilsqr() {
        use super::super::rules::{evaluate, RuleInput, ScanFacts};
        use super::super::config::{QsmPipelineConfig, InversionAlgorithm};
        let facts = ScanFacts { echo_count: Some(2), ..Default::default() };
        let ids = |cfg: &QsmPipelineConfig| -> Vec<&'static str> {
            evaluate(&RuleInput::pipeline(cfg), &facts).iter().map(|f| f.rule.id).collect()
        };
        let mut sti = QsmPipelineConfig::default();
        sti.inversion.algorithm = InversionAlgorithm::Ilsqr;
        let found = ids(&sti);
        assert!(found.contains(&"sti.ilsqr-iterations") && !found.iter().any(|id| id.starts_with("ukb.")), "{found:?}");
        let mut ukb = sti.clone();
        ukb.bg_removal.ukb_mask_trim = true;
        let found = ids(&ukb);
        for id in ["ukb.unwrap-laplacian", "ukb.echo-weights", "ukb.ilsqr-pad", "ukb.reference-none", "sti.ilsqr-single"] {
            assert!(found.contains(&id), "{id} in {found:?}");
        }
        for id in ["sti.ilsqr-iterations", "sti.ilsqr-pad", "sti.vsharp-polish-mask", "sti.mask-threshold-gui", "sti.unwrap-unmasked", "sti.echo-weights"] {
            assert!(!found.contains(&id), "{id} in {found:?}");
        }
    }

    /// DECOMPOSE's pipeline is a QSM per echo, as its authors ran it: the value its
    /// (reasoned) rule asks for.
    #[test]
    fn decompose_runs_a_qsm_per_echo() {
        let p = reference_pipeline("decompose").unwrap();
        let rule = RULES.iter().find(|r| r.id == "decompose.qsm-per-echo").expect("the rule");
        assert_ne!(rule.strength, Strength::Reference);
        let chosen = p.choose.iter().find(|(s, _)| *s == rule.setting).map(|(_, v)| v);
        assert!(matches!((&rule.expect, chosen), (Test::OneOf { values }, Some(v)) if values.contains(v)), "{chosen:?}");
    }
}

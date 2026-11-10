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
//!   authors ran, with their reason (DECOMPOSE's per-echo QSM, `decompose.qsm-per-echo`), is
//!   one of the pipeline's choices.
//! - **Scan-dependent rules apply on the scan.** A rule with a scan condition (MEDI's echo-time
//!   choice needs more than one echo) is part of the pipeline only when the scan is known to meet
//!   it, as everywhere in [`rules`](super::rules).
//! - **Parity is checked, not declared.** Whether a pipeline reproduces its toolbox is measured
//!   by QSM-CI's whole-pipeline parity track (QSMxT/QSM-CI#421), against each toolbox's own
//!   pipeline run end to end; a QSMxT release waits on it.
//! - **Which pipelines.** A toolbox whose own code runs a whole pipeline around its method, with
//!   rules for it: the MEDI toolbox, STI Suite (its iLSQR and STAR-QSM), QSMART, the HEIDI
//!   package, the χ-separation toolbox (its three methods) and DECOMPOSE. A method that is only
//!   trained on a pipeline (QSMnet's V-SHARP) keeps its reference settings as a method's, not as
//!   a pipeline of its own.

use serde::Serialize;

use super::rules::Value::{self, Bool, Choice};

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
    /// The method choices that start it, as rule settings and values ([`SETTINGS`](super::rules::SETTINGS)).
    pub choose: &'static [(&'static str, Value)],
    /// Where the pipeline is written down: the toolbox's script, README or paper.
    pub source: &'static str,
}

/// The catalogue, in the order a host lists it.
pub static REFERENCE_PIPELINES: &[ReferencePipeline] = &[
    ReferencePipeline {
        id: "medi-toolbox",
        name: "MEDI toolbox",
        toolbox: "MEDI toolbox 2024.11.26",
        choose: &[("inversion.algorithm", Choice("medi"))],
        source: "MEDI toolbox 2024.11.26 README.m:80-144",
    },
    ReferencePipeline {
        id: "sti-suite-ilsqr",
        name: "STI Suite · iLSQR",
        toolbox: "STI Suite 3.0",
        choose: &[("inversion.algorithm", Choice("ilsqr"))],
        source: "STI Suite 3.0 QSM_GUI, probed headless (QSM.rs ilsqr.rs and vsharp.rs docs)",
    },
    ReferencePipeline {
        id: "sti-suite-star",
        name: "STI Suite · STAR-QSM",
        toolbox: "STI Suite 3.0",
        choose: &[("inversion.algorithm", Choice("star"))],
        source: "STI Suite 3.0 QSM_GUI, probed headless (QSM.rs star.rs and vsharp.rs docs)",
    },
    ReferencePipeline {
        id: "qsmart",
        name: "QSMART",
        toolbox: "QSMART",
        choose: &[("inversion.algorithm", Choice("qsmart"))],
        source: "QSMART QSMART.m; Yaghmaie et al. 2021 NeuroImage 231:117701",
    },
    ReferencePipeline {
        id: "heidi",
        name: "HEIDI package",
        toolbox: "HEIDI package (qsm_heidi 4592631)",
        choose: &[("inversion.algorithm", Choice("lsqr")), ("inversion.apply_heidi", Bool(true))],
        source: "qsm_heidi makefile and def.sh (4592631); Schweser et al. 2012 NeuroImage 62:2083",
    },
    ReferencePipeline {
        id: "chi-sep-ilsqr",
        name: "χ-separation toolbox · iLSQR",
        toolbox: "χ-separation toolbox 1.1.3",
        choose: &[("separation.algorithm", Choice("chi-sep-ilsqr"))],
        source: "χ-separation toolbox v1.1.3 Chisep_script.m",
    },
    ReferencePipeline {
        id: "chi-sep-medi",
        name: "χ-separation toolbox · MEDI",
        toolbox: "χ-separation toolbox 1.1.3",
        choose: &[("separation.algorithm", Choice("chi-sep-medi"))],
        source: "χ-separation toolbox v1.1.3 Chisep_script.m",
    },
    ReferencePipeline {
        id: "chi-sepnet",
        name: "χ-separation toolbox · χ-sepnet",
        toolbox: "χ-separation toolbox 1.1.3",
        choose: &[("separation.algorithm", Choice("chi-sepnet")), ("separation.chi_sepnet.route", Choice("toolbox"))],
        source: "χ-separation toolbox v1.1.3 Chisep_script.m; Kim et al. 2025 (χ-sepnet)",
    },
    ReferencePipeline {
        id: "decompose",
        name: "DECOMPOSE",
        toolbox: "DECOMPOSE",
        // A QSM per echo is what its authors ran (§2.2, §4.1: a TE-dependent QSM input is
        // necessary), so it is one of the pipeline's choices, though its rule
        // (`decompose.qsm-per-echo`) is a reasoned one: see the decision above.
        choose: &[("separation.algorithm", Choice("decompose")), ("qsm_per_echo", Bool(true))],
        source: "Chen et al. 2021 NeuroImage 242:118477 (DECOMPOSE), §2.2, §3.6, §4.1",
    },
];

/// The pipeline with this id.
pub fn reference_pipeline(id: &str) -> Option<&'static ReferencePipeline> {
    REFERENCE_PIPELINES.iter().find(|p| p.id == id)
}

#[cfg(test)]
mod tests {
    use super::super::rules::{Condition, Strength, Test, RULES, SETTINGS};
    use super::*;

    /// Ids are unique kebab-case, every choice is a rule setting, and every pipeline's choices
    /// trigger at least one of the authors' settings (else it would be a method, not a pipeline).
    #[test]
    fn pipelines_are_well_formed() {
        let mut ids = std::collections::HashSet::new();
        for p in REFERENCE_PIPELINES {
            assert!(ids.insert(p.id), "duplicate pipeline id {}", p.id);
            assert!(p.id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'), "{}: not kebab-case", p.id);
            assert!(!p.choose.is_empty(), "{}: no choices", p.id);
            for (setting, _) in p.choose {
                assert!(SETTINGS.iter().any(|(s, _)| s == setting), "{}: {setting} not in SETTINGS", p.id);
            }
            let triggered = RULES.iter().filter(|r| r.strength == Strength::Reference).any(|r| {
                let Condition::Setting { setting, is: Test::OneOf { values } } = r.trigger() else { return false };
                p.choose.iter().any(|(s, v)| s == setting && values.contains(v))
            });
            assert!(triggered, "{}: its choices trigger no reference rule", p.id);
        }
        assert_eq!(reference_pipeline("medi-toolbox").map(|p| p.name), Some("MEDI toolbox"));
        assert!(reference_pipeline("nope").is_none());
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

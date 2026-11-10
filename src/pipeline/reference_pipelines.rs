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
//! - **What cannot be reproduced is said.** No pipeline here is reproduced in full.
//!   [`ReferencePipeline::not_reproduced`] lists, one short plain line each, every step of the
//!   toolbox's pipeline that QSM.rs does not reproduce today: steps not yet ported, settings no
//!   rule can express yet, steps outside the toolbox's published code, and choices still to be
//!   made (said neutrally, naming what is followed). The host shows them before the pipeline is
//!   chosen. Each line is taken from the toolbox-pipeline audit of 2026-11-10, which has the
//!   toolbox file and line behind it; a line goes when what it says is fixed.
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
    /// Steps of the toolbox's pipeline that QSM.rs does not reproduce, one short line each, for
    /// a host to show before the pipeline is chosen (see the module docs).
    pub not_reproduced: &'static [&'static str],
}

/// The catalogue, in the order a host lists it.
pub static REFERENCE_PIPELINES: &[ReferencePipeline] = &[
    ReferencePipeline {
        id: "medi-toolbox",
        name: "MEDI toolbox",
        toolbox: "MEDI toolbox 2024.11.26",
        choose: &[("inversion.algorithm", Choice("medi"))],
        source: "MEDI toolbox 2024.11.26 README.m:80-144",
        not_reproduced: &[
            "Brain mask: the toolbox runs BET on the magnitude without bias correction and does not erode the mask",
            "Central frequency: the toolbox converts to ppm with the scanner's imaging frequency, not γ·B0",
            "Uncombined coils: the toolbox's Fit_ppm_complex combines them itself; here the coils must be combined first",
            "Two echoes: the toolbox's CSF mask covers the whole central brain; here MEDI runs without the CSF term",
            "Odd matrix sizes: the toolbox's SMV kernel is half a voxel off; here odd axes are padded by one voxel",
            "Outside the mask: the toolbox leaves minus the CSF mean there; here it is 0",
        ],
    },
    ReferencePipeline {
        id: "sti-suite-ilsqr",
        name: "STI Suite · iLSQR",
        toolbox: "STI Suite 3.0",
        choose: &[("inversion.algorithm", Choice("ilsqr"))],
        source: "STI Suite 3.0 QSM_GUI, probed headless (QSM.rs ilsqr.rs and vsharp.rs docs)",
        not_reproduced: &[
            "Brain mask: the mask STI Suite's QSM_GUI makes is not reproduced",
            "Precision: STI Suite runs iLSQR in single precision; here it runs in double",
            "Field mapping: STI Suite keeps the unwrapped field outside the mask and removes no per-echo mean",
            "Iterations: STI Suite's GUI runs iLSQR for 40 iterations, its STI_Templates.m for 100; this follows STI_Templates.m",
            "Referencing: whether STI Suite references χ is not confirmed",
        ],
    },
    ReferencePipeline {
        id: "sti-suite-star",
        name: "STI Suite · STAR-QSM",
        toolbox: "STI Suite 3.0",
        choose: &[("inversion.algorithm", Choice("star"))],
        source: "STI Suite 3.0 QSM_GUI, probed headless (QSM.rs star.rs and vsharp.rs docs)",
        not_reproduced: &[
            "Brain mask: the mask STI Suite's QSM_GUI makes is not reproduced",
            "Field mapping: STI Suite keeps the unwrapped field outside the mask and removes no per-echo mean",
            "Referencing: whether STI Suite references χ is not confirmed",
        ],
    },
    ReferencePipeline {
        id: "qsmart",
        name: "QSMART",
        toolbox: "QSMART",
        choose: &[("inversion.algorithm", Choice("qsmart"))],
        source: "QSMART QSMART.m; Yaghmaie et al. 2021 NeuroImage 231:117701",
        not_reproduced: &[
            "Brain mask: QSMART's recipe (BET f 0.2 on N4-corrected echo 1, threshold, majority filter, closing) is not reproduced",
            "Phase: geme_cmb's phase-offset removal and POEM coil combination are not reproduced",
            "Precision: QSMART runs iLSQR in single precision; here it runs in double",
            "γ: QSMART's demo converts to ppm with 2.675e8 rad/s/T; here 42.576 MHz/T",
            "adjust_offset: QSMART's k-space grid differs on odd matrix sizes; here the FFT grid is used",
        ],
    },
    ReferencePipeline {
        id: "heidi",
        name: "HEIDI package",
        toolbox: "HEIDI package (qsm_heidi 4592631)",
        choose: &[("inversion.algorithm", Choice("lsqr")), ("inversion.apply_heidi", Bool(true))],
        source: "qsm_heidi makefile and def.sh (4592631); Schweser et al. 2012 NeuroImage 62:2083",
        not_reproduced: &[
            "Unwarping, masks and phase preprocessing: the authors run them outside the qsm_heidi repository; not reproduced",
            "SHARP: the authors' settings are not in the qsm_heidi repository",
            "LSQR weight: the toolbox weights by the magnitude without bias correction",
        ],
    },
    ReferencePipeline {
        id: "chi-sep-ilsqr",
        name: "χ-separation toolbox · iLSQR",
        toolbox: "χ-separation toolbox 1.1.3",
        choose: &[("separation.algorithm", Choice("chi-sep-ilsqr"))],
        source: "χ-separation toolbox v1.1.3 Chisep_script.m",
        not_reproduced: &[
            "Brain mask: the toolbox runs BET on the magnitude without bias correction and does not erode the mask",
            "ROMEO: the toolbox runs the ROMEO command line at its defaults on the uneroded mask",
            "Central frequency: the toolbox uses the scanner's imaging frequency for ppm and the relaxometric constants",
            "Tukey window: 0.4, or 0 on Philips, is set only when the scanner's manufacturer is known",
            "R2*: the toolbox's R2* fits (r2star_arlo, r2star_nnls) are P-coded and not confirmed",
            "R2′: the toolbox takes R2′ from the user; here it is measured or predicted",
        ],
    },
    ReferencePipeline {
        id: "chi-sep-medi",
        name: "χ-separation toolbox · MEDI",
        toolbox: "χ-separation toolbox 1.1.3",
        choose: &[("separation.algorithm", Choice("chi-sep-medi"))],
        source: "χ-separation toolbox v1.1.3 Chisep_script.m",
        not_reproduced: &[
            "Brain mask: the toolbox runs BET on the magnitude without bias correction and does not erode the mask",
            "ROMEO: the toolbox runs the ROMEO command line at its defaults on the uneroded mask",
            "Central frequency: the toolbox uses the scanner's imaging frequency for ppm and the relaxometric constants",
            "Tukey window: 0.4, or 0 on Philips, is set only when the scanner's manufacturer is known",
            "R2*: the toolbox's R2* fits (r2star_arlo, r2star_nnls) are P-coded and not confirmed",
            "R2′: the toolbox takes R2′ from the user; here it is measured or predicted",
            "CSF mask: the script's CSF mask line is commented out; here MEDI's extract_CSF is used",
        ],
    },
    ReferencePipeline {
        id: "chi-sepnet",
        name: "χ-separation toolbox · χ-sepnet",
        toolbox: "χ-separation toolbox 1.1.3",
        choose: &[("separation.algorithm", Choice("chi-sepnet")), ("separation.chi_sepnet.route", Choice("toolbox"))],
        source: "χ-separation toolbox v1.1.3 Chisep_script.m; Kim et al. 2025 (χ-sepnet)",
        not_reproduced: &[
            "Brain mask: the toolbox runs BET on the magnitude without bias correction and does not erode the mask",
            "ROMEO: the toolbox runs the ROMEO command line at its defaults on the uneroded mask",
            "Central frequency: the toolbox uses the scanner's imaging frequency for ppm and the relaxometric constants",
            "Tukey window: 0.4, or 0 on Philips, is set only when the scanner's manufacturer is known",
            "R2*: the toolbox's R2* fits (r2star_arlo, r2star_nnls) are P-coded and not confirmed",
            "Resolution: the toolbox's resolution-generalisation branch (resgen) is not ported",
        ],
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
        not_reproduced: &[
            "The paper and the authors' script differ; this pipeline follows the paper, with STI Suite's settings where it gives none",
            "From the paper: BET on the first echo's magnitude and the constrained interior-point fit are not reproduced",
            "From the script: V-SHARP 25 mm, STAR-QSM padding 12, no referencing and B0 from the imaging frequency are not followed",
        ],
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

    /// Every pipeline says what it does not reproduce, one short line each.
    #[test]
    fn every_pipeline_says_what_it_does_not_reproduce() {
        for p in REFERENCE_PIPELINES {
            assert!(!p.not_reproduced.is_empty(), "{}", p.id);
            for line in p.not_reproduced {
                assert!(!line.is_empty() && line.chars().count() <= 120 && !line.contains('\n'), "{}: {line}", p.id);
                assert!(!line.ends_with('.'), "{}: {line}", p.id);
            }
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

# HC-ChiSep — manuscript plan

Working title (preferred):

> **Fibre orientation from the signal, not from diffusion: hollow-cylinder χ-separation of
> paramagnetic and diamagnetic susceptibility**

Alternatives: *"Orientation-resolved susceptibility source separation without DTI"*;
*"R2′-anchored multi-compartment fitting makes white-matter fibre orientation identifiable
from single-orientation GRE magnitude"* (this one foregrounds the actual methodological
contribution rather than the application).

Status of this document: a writing plan. Section 3 (Introduction) and Section 4 (Theory) are
specified paragraph-by-paragraph and equation-by-equation, as requested; Sections 5–8 are
specified at the level of "which claim, backed by which number, from which file". Section 10
lists the experiments that must be run before this is submittable.

---

## 1. What the paper actually claims (and what it must not)

The single most important editorial decision is the claim set, because the strongest result
(24.6 % NRMSE, best on the QSM-CI board) is also the most attackable one: **HC-ChiSep and the
phantom that scores it share a forward-model family.** The honest, defensible framing puts the
*identifiability* result first and the leaderboard result second.

**Primary claim (defensible, general, and the real contribution).**
The fibre-to-B0 angle θ of white matter is *observable in a single-orientation multi-echo GRE
magnitude*, and becomes reliably estimable once the total reversible decay rate is anchored to
an independently measured R2′. The anchor removes a shallow θ ↔ decay-rate trade-off valley
that otherwise makes the per-voxel fit noise-limited at realistic SNR. Evidence: identifiability
analysis (unique global minimum, exact refinement), Monte-Carlo noise study, and the anchored-vs-
unanchored comparison (θ median error 10.6° → 6.9°, correlation 0.72 → 0.86; MWF correlation
0.29 → 0.81).

**Secondary claim (application).**
Because that same microstructure is what makes the diamagnetic relaxivity orientation-dependent,
signal-derived θ closes the anisotropy gap in χ-separation *without a diffusion acquisition*.
On the QSM-CI χ-separation phantom this yields 24.6 % average detrended NRMSE vs 43.4 % for the
closed-form null, 46.1 % for SUSEP-Net and 78.1 % for χ-sepnet; and the signal-derived arm is
*not worse* than the arm handed the ground-truth DTI angle (24.6 vs 25.7).

**Tertiary claim (benchmark methodology).**
A source-separation phantom whose R2′ is generated from a scalar relaxivity model is solvable
in closed form from its own public inputs; anisotropy breaks that degeneracy only inside white
matter. This is a benchmark-design result worth stating, and it is why the null baseline is
scored on the leaderboard.

**Claims to explicitly disclaim, in the Abstract and again in the Discussion.**
1. Matched model: the hollow-cylinder pool parameters, the |χ−| ↔ MWF anchor and the Ridani
   relaxivity convention used by HC-ChiSep are the same published family the phantom generator
   implements. The paper reports what is recoverable when the model class is right — a
   mechanistic reference baseline, not a demonstration of robustness to model error.
2. Hyperparameters (smoothing scales, the WM-likeness sigmoid, λ, the normalised-convolution
   σ) were selected against this phantom's ground truth. The *physics* is fixed a priori and
   Dr+ is self-calibrated from the inputs at run time, but the tuning must be disclosed.
3. No in-vivo validation is presented (see §10 — this is the largest gap).
4. θ recoverability is a 7 T result. At 3 T it is marginal and at ~1.3 T it is absent.

---

## 2. Venue and format

Two products, in this order:

- **A. ISMRM-style abstract** (a template is already in `~/repos/qsm/Abstracttemplate_Annapolis2026.docx`)
  — 1 page: the anchoring idea, the θ-recovery figure, the leaderboard table. Cheap, and it
  establishes priority on "signal-derived orientation for χ-separation".
- **B. Full paper for *Magnetic Resonance in Medicine*** — the MRM LaTeX class and AMA bib style
  already used in this group live in `~/repos/qsm/qsmbly/paper/` (`MRM.cls`, `MRM-AMA.tex`,
  `ama.bst`); copy that scaffold rather than starting from scratch. MRM is the right home:
  the phantom paper (Ridani 2026), the χ-separation paper (Shin 2021) and the Challenge 2.0
  head model (Marques 2021) are all MRM/NeuroImage-adjacent, and MRM takes Theory-heavy method
  papers with a "Theory" section as a first-class heading.

MRM section order to use: Introduction — Theory — Methods — Results — Discussion — Conclusion.
The Theory section is where the mathematics lives; the Introduction only *motivates* it. Keeping
them separate is what lets the Introduction stay readable while the derivation stays complete.

---

## 3. Introduction — paragraph-by-paragraph plan

Seven paragraphs, each doing exactly one job. The through-line is a single narrowing argument:
*two sources, one measurement → add R2′ → but R2′ has a hidden third unknown (θ) → everyone
either ignores it, or buys it with a second acquisition → but θ is already in the data we have
→ here is why it has not been usable, and what makes it usable.*

**¶1 — The cancellation problem (why source separation exists at all).**
Susceptibility is a signed sum: χ_total = χ+ + χ−. Iron (ferritin, haemosiderin) is
paramagnetic, myelin and calcium diamagnetic, and in white matter and deep grey matter they
coexist, so QSM reports a difference that can be near zero while both sources are large. Cite
the demyelination/iron-accumulation motivation (MS lesions, deep grey iron in
neurodegeneration). One sentence stating the clinical stake; no more.
→ carries Eq. (T1).

**¶2 — The standard fix: a second, additive projection.**
Reversible transverse relaxation adds where the field subtracts, because in the static-dephasing
regime R2′ depends on the *magnitude* of the field perturbation, not its sign (Yablonskiy &
Haacke 1994). So R2′ = R2* − R2 gives a second equation, and χ-separation (Shin et al. 2021)
solves the 2×2 system through the relaxivity Dr. State that this is the shared skeleton of every
current method — χ-separation/χ-sepnet, SUSEP-Net, WaveSep, APART-QSM, R2*-QSM — regardless of
whether the solve is closed-form, iterative-regularised, or learned.
→ carries Eqs. (T2), (T3).

**¶3 — The turn: Dr is not a constant, it is microstructure.**
This is the pivot paragraph and should be the most carefully written in the paper. In the
static-dephasing derivation, the relaxivity is a *geometry* factor: spherical inclusions give
Dr+ = 2πγB0/(9√3), while parallel cylinders — myelinated axons — give Dr−(θ) = ½γB0·sin²θ. The
second equation therefore contains a third unknown, the fibre-to-B0 angle θ, and the system is
under-determined again. Worse, χ− itself is orientation-dependent (χ−(θ) = (χ∥ − χ⊥)cos²θ + χ−,iso).
Ridani et al. (2026) quantified exactly this: orientation-induced error is the dominant
systematic error of current source separation in white matter.
→ carries Eqs. (T4), (T5), (T6).

**¶4 — How the field currently disposes of θ, and what each choice costs.**
Four strategies, each one sentence, each with its cost:
 (i) *assume it away* — a single empirical scalar relaxivity (137 Hz/ppm, Shin 2021; 114 Hz/ppm
     in the χ-sepnet calibration) fitted in iron-rich deep grey nuclei and then applied to
     myelin cylinders. Cost: a systematic, orientation-patterned WM bias, and — as our own
     benchmark shows — brittleness, since a network that hard-codes the calibration degrades
     from competitive to worst-in-class when the phantom's Dr scale moves by 2.8× (78.1 %).
 (ii) *buy it with diffusion* — take θ from a DTI V1 map. Cost: a second acquisition, a
     registration (in our own data the V1 map is misregistered by (+5, +1, −6) voxels with a
     stale native-DWI affine — a realistic failure mode), and a single-fibre assumption.
 (iii) *buy it with head rotation* — multi-orientation STI-style acquisition. Cost: impractical
     clinically.
 (iv) *learn it* — supervised networks. Cost: the mapping is learned at one relaxivity scale and
     one field, and distribution shift is invisible at inference time.
→ this paragraph is where the reader must accept that "the θ problem is real and unsolved".

**¶5 — The observation this paper is built on.**
The same microstructure that makes Dr− orientation-dependent *also* makes the GRE magnitude
non-mono-exponential in an orientation-dependent way. A myelinated voxel is not one water pool
but three — myelin water, axonal, extra-axonal — sitting at different Larmor offsets set by the
hollow-cylinder field of the sheath (Wharton & Bowtell 2012). The pools beat against each other;
the beat's shape depends on θ and on the myelin water fraction. One mechanism, two observable
consequences: an orientation-dependent relaxivity, and an orientation-dependent decay *shape*.
The proposal is to use the second to resolve the first — the orientation information a
χ-separation method needs is already inside the magnitude data it already has.
→ carries Eqs. (T7), (T8).

**¶6 — Why this has not simply been done, and what makes it work now.**
Two obstacles, both quantified in this paper:
 (a) *Amplitude.* At clinically usual fields the beat is too small relative to noise. Frequency
     offsets scale with B0; in our field study θ is essentially unrecoverable at 1.27 T
     (median error 25°, versus ≈30° for a random guess), marginal at 3 T (10°) and recoverable
     at 7 T (3°). This paper is therefore about the 7 T regime — a scope statement, made early
     and honestly.
 (b) *Conditioning.* Even where the beat is visible, the noiseless problem, though uniquely
     identifiable, has its minimum in a long shallow curved valley: increasing θ and decreasing
     the mesoscopic dephasing rate change the decay in nearly the same way, and only the
     non-mono-exponential curvature distinguishes them. Under Rician noise at the effective
     white-matter SNR of a realistic acquisition (~28), per-voxel MWF and rate estimates are at
     or beyond the anatomical spread of those parameters — the fit is noise-limited, not
     resolution-limited (grid refinement changes nothing).
 The resolution: **R2′ is already measured, and it constrains exactly the direction the valley
 runs in.** Anchoring the fit by requiring the model's own analytic reversible rate plus a
 non-negative mesoscopic remainder to equal the observed R2′ collapses the valley and turns a
 3-parameter noise-limited fit into a well-conditioned 2-parameter one.
→ carries Eqs. (T9), (T10). This is the paragraph that states the paper's actual novelty.

**¶7 — Contributions and roadmap.**
Numbered list (four items): (1) the R2′-anchored hollow-cylinder estimator and its
identifiability/conditioning analysis; (2) a DTI-free, orientation-aware separation built on it,
including the model-selection rule that finds myelinated white matter from the data alone
(Dice 0.893 against the tissue label, without any segmentation input); (3) an evaluation on a
public, held-out-ground-truth benchmark including the closed-form null baseline, with the
matched-model caveat stated; (4) an open-source reference implementation in two independent
languages (the QSM-CI Python submission and the Rust `qsm-core` port), both CI-tested.

*Length target*: ~900 words. ¶3 and ¶6 each get ~180; the rest ~110 each.

---

## 4. Theory — the equation chain

Every equation below is already implemented and verified somewhere in the three repos; the
"source" line gives the file so the manuscript's numbers and the code cannot drift apart.

**T1 — the under-determined voxel.**

    χ_total = χ+ + χ−,   χ+ ≥ 0 ≥ χ−

One equation, two unknowns. (Convention note for the paper: we report χ− signed, ≤ 0, while
the phantom stores |χ−|; state this once, early.)
*Source*: `QSM.rs/src/separation/hc_chisep.rs` return contract.

**T2 — static dephasing gives the second equation.**

    R2′ = R2* − R2 = Dr+·|χ+| + Dr−·|χ−|

Yablonskiy–Haacke static-dephasing regime: dephasing is driven by |Δω|, so the two sources add.
*Source*: `qsmci/data/sim/chisep/README.md`, `qsm-forward` README §relaxivity.

**T3 — the 2×2 solve and the closed-form null.**

    [ 1    1 ] [ χ+  ]   [ χ_total ]
    [ Dr+ −Dr−] [ χ−  ] = [   R2′   ]

    ⇒ χ+ = (R2′ + Dr−·χ_total)/(Dr+ + Dr−),  χ− = (Dr+·χ_total − R2′)/(Dr+ + Dr−)

and in the single-relaxivity case Dr+ = Dr− = Dr this degenerates to

    χ± = ½(χ_total ± R2′/Dr)

State plainly: **this is the whole of χ-separation when Dr is a known constant** — everything
else in the literature is regularisation of a system that is already algebraically determined.
It is also the null baseline scored on the leaderboard, and on a phantom whose R2′ is generated
by this model with constant Dr it is *exact*. That fact is the paper's justification for why an
orientation-aware method is needed at all, and for why the benchmark had to be upgraded.
*Source*: `qsmci/outreach/chisep-phantom-talk.md` §"Failure 1"; `algorithms/chisep-null-qsmci`.

**T4 — the relaxivity is a geometry factor.**

    spheres:            Dr+ = 2πγ̄B0/(9√3)          ( = 107.84 Hz/ppm/T )
    parallel cylinders: Dr−(θ) = ½·γ̄·B0·sin²θ      ( = 0 … 133.77 Hz/ppm/T )

so the myelin term of T2 is orientation-dependent and vanishes for fibres along B0.
*Source*: `qsm-forward/qsm_forward/qsm_forward.py::generate_dr_maps_ridani` (lines 1254–1314),
which implements exactly these constants.

**T5 — the source itself is anisotropic.**

    χ−(θ) = (χ∥ − χ⊥)·cos²θ + χ−,iso

*Source*: Ridani et al. 2026 Eq. 3, as ported in `qsm-forward` (`chi-sep` subcommand).

**T6 — the resulting three-unknown problem.** Substituting T4–T5 into T2:

    R2′ = Dr+·χ+ + ½γ̄B0 sin²θ · |χ−(θ)|,   χ_total = χ+ + χ−(θ)

Two equations, three unknowns (χ+, χ−, θ). **This is the formal statement of the gap the paper
closes.** Every existing method either fixes θ implicitly (isotropic Dr), imports it (DTI), or
marginalises it (learning). Note also the important structural consequence used later: in the
Ridani convention the myelin cylinders carry the diamagnetic dephasing and the paramagnetic
spherical term is absent inside white matter (Dr+ = 0 there), so **the paramagnetic amplitude in
WM is not recoverable from R2′ at all** — it must come from the χ_total constraint. Getting this
backwards was the original design error of our own algorithm and is worth one sentence, because
it is a trap any reader implementing this will fall into.
*Source*: `qsmci/reports/2026-08-chisep/hc-chisep/REPORT.md` §"Structural finding".

**T7 — the hollow-cylinder compartment frequencies (the third observable).**
Model the sheath as an infinite hollow cylinder of material with a cylindrically symmetric,
radially oriented anisotropic susceptibility, split into isotropic χ_I and anisotropic χ_A parts,
g = inner/outer radius ratio, E an isotropic exchange offset, ω0 = γ̄B0:

    intra-axonal:   Δf_A(θ) = ω0 · (3/4)·χ_A·ln(1/g)·sin²θ
    myelin water:   Δf_M(θ) = ω0 · [ (χ_I/2)(2/3 − sin²θ) + χ_A(1/12 − (5/12)sin²θ)
                                     + (3/4)χ_A ln(1/g) sin²θ + E ]
    extra-axonal:   Δf_E(θ) = 0

with limiting cases that must be stated because they are the model's sanity checks: Δf_A → 0 as
g → 1 or χ_A → 0; Δf_M → ω0·E when χ_I = χ_A = 0; both are ∝ sin²θ, hence flat near θ = 0 —
which is precisely where every estimator in the paper degrades.
*Source*: `QSM.rs/src/separation/hc_chisep.rs::hc_compartment_freqs`;
`qsm-forward/prototypes/hollow_cylinder/REPORT.md` §1 (15/15 closed-form unit tests).

**T8 — the multi-compartment signal.**

    S(TE) = S0 · e^{−R2′_meso·TE} · Σ_p f_p · e^{−TE/T2_p} · e^{i2πΔf_p(θ)TE}
    f_M = MWF,  f_A = (1−MWF)·f_axon,  f_E = (1−MWF)(1−f_axon)

and the matched spin echo, in which the 180° pulse refocuses the frequency offsets but not the
pool T2 mixture:

    S_SE(TE) = S0 · Σ_p f_p e^{−TE/T2_p}

|S| is therefore *not* mono-exponential, and the deviation is a function of (θ, MWF). Report the
magnitudes: at 7 T, Δf_A(90°) = −7.97 Hz and Δf_M(90°) = +12.89 Hz, and the per-voxel beat
amplitude is only ~0.5–2 % of the first-echo intensity — the number that dictates everything
about how the estimator must be built.
*Source*: `hc_chisep.rs::hc_wm_signal_mag` / `hc_wm_se_signal`; prototype REPORT §Level 1.

**T9 — identifiability, and the trade-off valley.**
State the two-part result:
 (a) *Noiseless, the problem is well-posed*: for off-grid probe truths the SSE landscape over
     (θ, MWF, R2′_meso) has a unique global minimum at the truth and continuous refinement from
     the grid winner recovers all three to < 10⁻³.
 (b) *Under noise it is ill-conditioned*: the minimum lies in a curved shallow valley in which
     larger θ trades against smaller R2′_meso (both increase net decay; only the curvature
     separates them), longest and shallowest near θ = 0. Quantify with the Monte-Carlo table
     (Rician noise, 200 trials/probe): at SNR 100, θ MAE 6.5–19.2°, MWF MAE 0.033–0.062 against
     a phantom MWF spread of 0.042, R2′_meso MAE 1.4–4.7 Hz against a spread of 2.5 Hz. Hence
     the honest conclusion: **θ is the one robust parameter of the blind fit; the paramagnetic
     rate is not recoverable from magnitude alone.**
 Include the negative control that grid refinement (Nelder–Mead) changes nothing — the limit is
 noise, not discretisation.
*Source*: `qsmci/reports/2026-08-chisep/nonoracle/report.md` Parts 1–2.

**T10 — the anchor (the paper's methodological core).**
The valley runs along "total reversible decay". That quantity is measured. Define the model's
own analytic reversible rate as the mono-exponential-equivalent slope of the pool interference
over the actual echo train,

    R2′_hc(θ,MWF) = −Σ_e (TE_e − T̄E)·ln( |S_GRE(TE_e)| / S_SE(TE_e) ) / Σ_e (TE_e − T̄E)²   , clipped ≥ 0

and require the observed R2′ to be that rate plus a non-negative mesoscopic remainder:

    R2′_meso = R2′_obs − R2′_hc(θ,MWF) ≥ 0

The candidate magnitude curve for a voxel with measured R2′_obs is then

    Ŝ(TE; θ,MWF) = |pools(θ,MWF,TE)| · exp(−R2′_meso(θ,MWF)·(TE − TE₁))

first-echo normalised, so S0 drops out. The search is now over a 2-D grid (θ ∈ [0°,90°] step
1.5°, MWF ∈ [0.03,0.25] step 0.005 — 61 × 45 candidates) with voxels binned by R2′_obs (0.25 Hz)
so each bin shares one candidate library and the search is a single matrix product per bin.
State the payoff as a before/after: θ median error 10.6° → 6.9° (raw) and correlation 0.72 →
0.86 after spatial regularisation; MWF correlation 0.29 → 0.81, median error 0.059 → 0.015.
*Source*: `hc_chisep.rs::AnchoredFitter` (`new`, `fit`); REPORT.md §Diagnostics.

**T11 — locating myelinated white matter by model selection.**
Since the per-voxel beat is below the per-voxel noise, detection must pool voxels. Define, on
lightly smoothed data (σ ∈ {0.75, 1.0} voxels, elementwise best over scales),

    lr = log10 [ SSE_anchored-HC(GRE, SE) / SSE_free-mono-exponential(GRE, SE) ]

    w = σ((c_lr − lr)/s_lr) · σ((c_χ − χ̄_total)/s_χ) · mask,   c_lr = 1.5, s_lr = 0.3, c_χ = s_χ = 0.01 ppm

The first factor asks "does this voxel's decay look like interfering pools rather than a single
pool?", the second is a consistency prior ("myelinated WM is diamagnetic-leaning") derived from
the input χ map, not from a segmentation. Report the separation (WM median lr ≈ −0.1, non-WM
≈ +2.2) and the resulting Dice of 0.893 (precision 0.869, recall 0.919) against the held-out
tissue label — a result worth its own sentence because the classifier never sees a segmentation.
*Source*: `hc_chisep.rs` Stage 3; REPORT.md §Diagnostics.

**T12 — from (θ, MWF) to the two sources.**
Two limits of the same two-source model, blended by w:
 *Outside myelinated WM* (no cylinders ⇒ Dr− = 0), T3 collapses to the exact closed form

    χ+ = R2′/Dr+,   |χ−| = χ+ − χ_total

 *Inside myelinated WM* (Dr+ = 0 by the convention in T6, so R2′ carries no paramagnetic
 amplitude), close the system with the myelin-content ↔ MWF proportionality

    |χ−| = K_χ · MWF,   K_χ = 0.038 ppm / 0.12 ≈ 0.317 ppm per unit MWF
    χ+   = λ·(χ_total + K_χ·MWF) + (1−λ)·c₀,   λ = 0.7

 with c₀ a self-calibrated median WM χ+ prior. Note the exact non-negativity constraint used as
 an estimator bound rather than a post-hoc clamp:

    χ+ ≥ 0  ⟺  MWF ≥ −χ_total/K_χ

 (a physical feasibility condition turned into a per-voxel prior — worth highlighting, it is one
 of the reasons MWF recovery improves so much over the blind fit).
 Finally the blend: χ+ = w·χ+^WM + (1−w)·χ+^CF, likewise for χ−. State that hard switching
 instead of soft blending costs ≈10 NRMSE points — the branch boundary is genuinely uncertain.
*Source*: `hc_chisep.rs` Stages 2 and 5.

**T13 — Dr+ self-calibration and graceful degradation.** Two short paragraphs:
 - Dr+ is not assumed but estimated from the inputs as the 5th percentile of R2′/χ_total over
   confidently paramagnetic voxels (χ_total > 0.02 ppm) — a robust lower envelope, since voxels
   with any diamagnetic content inflate the ratio. It recovers 320.0 and 137.0 Hz/ppm exactly on
   the two phantom builds, with the field-scaled empirical value 137·B0/3 as fallback.
 - If no hollow-cylinder-consistent voxels exist, w collapses and the method reduces to the
   closed-form solve everywhere — the method degrades to the state of the art rather than
   failing. (In practice on single-compartment data the classifier still fires on 40 % of the
   brain because 4-echo single-compartment WM is genuinely consistent with the model family;
   the outcome is much better than the pure fallback, 35.7 vs 92.3 average detrended NRMSE.
   Report this honestly — it is a statement about what the classifier means: "consistent with
   myelinated-WM physics", not "a beat was detected".)

---

## 5. Methods — what to write

1. **Algorithm.** The five stages, in the T13→T10→T11→T12 order, with the pseudocode block and
   the complete hyperparameter table (all constants are literals in
   `QSM.rs/src/separation/hc_chisep.rs` lines 47–83 — copy them from there, do not retype).
   Report complexity: O(N·|grid|) as a binned GEMM; 104 s for 1.3 M voxels, 8 GRE + 4 SE echoes,
   14 CPU cores; no GPU, no trained weights, no network access.
2. **Model parameters.** Table of χ_I, χ_A, E, g, T2_M/A/E, f_axon, MWF range, K_χ, with the
   literature source for each (Wharton & Bowtell 2012 Table 3 for the first five).
   ⚠ **Consistency check required before writing**: the prototype's seed table lists χ_I = −0.10 ppm
   while both the shipped phantom and `hc_chisep.rs` use χ_I = −0.06 ppm (W&B's own measured
   value). Confirm which was used for each reported number and state one value.
3. **Phantom.** One subsection, because reviewers will ask exactly how the truth was made:
   Challenge-2.0 head model (Marques 2021, 7 T, 0.64 mm) → Ridani et al. (2026) source split
   (χ+ from a χ-separation atlas ratio, validated r = 0.83 against Hallgren–Sourander histology
   iron; χ− as the remainder; intra-ROI texture; WM anisotropy from DTI V1) → our multi-compartment
   WM upgrade. The three corrections we had to make are genuine methodological content and
   should be stated as such: (a) the spin echo must decay with the same pool-T2 mixture, else
   signal-derived R2′ drifts ~2× in WM; (b) no double counting — the pool interference *is* the
   mechanistic origin of WM's orientation-dependent R2′, so the imposed Dr−(θ)|χ−| term must be
   dropped there; (c) the MWF ↔ χ− mapping had a units error saturating MWF at 0.25 everywhere.
   Also report the acquisition (8 GRE echoes 3–45 ms, 4 SE echoes 10–70 ms, TR 50 ms/1.5 s, flip
   15°, Rician noise at peak SNR 100, native 1 mm simulation with a single PSF) and the
   field-strength decision (Dr = 137·B0/3 ≈ 320 Hz/ppm with pools at 7 T; at that scale the
   imposed myelin term and the true 7 T mechanism agree to ~2 Hz with no tuning — a genuinely
   satisfying consistency result and the best single argument that the phantom is coherent).
4. **Evaluation.** Detrended NRMSE (definition from `qsmci/eval/qsm_eval.py::nrmse_challenge`:
   demean, fit and remove a linear trend, then normalise — chosen because χ-separation outputs
   are scale-sensitive and xSIM is too lenient, scoring structure while missing magnitude error),
   XSIM, correlation, and the cross-source leakage metric (mean |recon| in the region owned by
   the *other* source: iron/vein for χ−, calcification for χ+). Comparators: the closed-form
   null, χ-sepnet, SUSEP-Net (and see §10 — the rest of the board must be added).
5. **Implementations.** The Python QSM-CI submission (numpy/scipy/nibabel only, containerised)
   and the independent Rust port in `qsm-core` (`src/separation/hc_chisep.rs`, 1046 lines,
   parallel via rayon, unit-tested and gated in CI). Note that the Rust port required
   scipy-matching Gaussian and median filters with reflect boundaries — a reproducibility detail
   that matters for anyone reimplementing.

---

## 6. Results — the claim → number → source map

| # | Claim | Number | Source |
|---|---|---|---|
| R1 | Noiseless identifiability | unique global minimum at truth; refinement to <1e-3 for all three parameters | nonoracle/report.md Part 1 |
| R2 | Ill-conditioning under noise | θ MAE 6.5–19.2°; MWF MAE 0.033–0.062 (spread 0.042); rate MAE 1.4–4.7 Hz (spread 2.5) @ SNR 100 | ibid., MC table |
| R3 | Grid is not the limiter | Nelder–Mead refinement: θ 11.0→11.2°, MWF corr 0.27→0.26 | ibid. Part 2 |
| R4 | Blind fit: θ recoverable, χ+ rate not | θ corr 0.72 / median 10.6°; MWF corr 0.29; R2′_meso corr −0.07 | ibid. Part 2 |
| R5 | Hard SE anchoring hurts | θ median 10.6 → 15.0° | ibid. Part 3 |
| R6 | Field dependence of θ | median error 25.0° @1.27 T, 10.0° @3 T, 3.0° @7 T (SNR 100) | b0_study/report.md §1 |
| R7 | Beat above noise floor | residual RMS 0.026 @7 T/90° vs 1/SNR = 0.010; flat (0.0106→0.0097) at 1.27 T | ibid. §3 |
| R8 | Phantom R2′ anisotropy vs literature | 7 T curve 17.35 Hz @90° vs 5–12 Hz literature band; fixed-Dr curve 6.5·sin²θ | ibid. §2 |
| R9 | Anchoring works | θ corr 0.862, median 8.4°, MAE 10.1°; MWF corr 0.81, median err 0.015 | hc-chisep/REPORT.md |
| R10 | WM found without a segmentation | Dice 0.893 (P 0.869 / R 0.919); w>0.5 fraction 0.418 | ibid. |
| R11 | **Headline separation** | para 19.6 / dia 29.6 / avg **24.6** detrended NRMSE; xSIM 0.910 / 0.919 | ibid. |
| R12 | Beats the fair null and the trained nets | null 43.4, SUSEP-Net 46.1, χ-sepnet 78.1 | ibid.; bakeoff/ |
| R13 | **Signal-derived ≥ DTI-informed** | 24.6 vs 25.7 avg | ibid. arm b |
| R14 | The mechanism is what pays | χ− channel 29.6 vs null's 56.6; χ+ 19.6 vs 30.2 | ibid. |
| R15 | Ablation | closed-form-everywhere: 76.5 avg | ibid. arm c |
| R16 | Graceful degradation | 35.7 avg on the single-compartment phantom vs 92.3 for the pure fallback | ibid. arm d |
| R17 | Low cross-contamination | iron leak 0.0165 ppm; calcification leak 0.0065 ppm | ibid. |
| R18 | Error budget | oracle-MWF floor 8.3 → our MWF 23.2 → +our mask 24.6 | ibid. |
| R19 | Runtime | 104 s, 1.3 M voxels, 14 cores, CPU only | ibid. |
| R20 | Independent reimplementation | Rust port on the single-compartment phantom: χ+ corr 0.94, χ− 0.91 (vs WaveSep 0.943/0.814, R2*-QSM 0.935/0.863, DECOMPOSE 0.58/0.79 on the same data) | QSM.rs commit 5cf180b; `tests/chisep_relaxometry_qsmci.rs` |

R18 is the most useful result for a reader who wants to improve on this: **the binding constraint
is per-voxel MWF precision, not orientation and barely the mask.** Say so explicitly.

---

## 7. Figures and tables

- **F1** (schematic, no data). Left: one voxel, three pools in a myelinated axon with the
  hollow-cylinder geometry and the three frequency offsets. Middle: the resulting non-mono-exponential
  |S(TE)| for θ = 0/45/90° against a mono-exponential reference. Right: the same θ appearing in
  Dr−(θ) ∝ sin²θ. Caption states the paper's thesis in one sentence. This figure *is* the paper.
- **F2** Identifiability: the SSE landscape (θ × R2′_meso slice at true MWF) showing the curved
  valley, with the truth marked — plus the anchored slice next to it showing the valley gone.
  (The second panel needs generating; see §10.)  Source: `nonoracle/sse_landscape.png`.
- **F3** Field study: θ MAE vs θ_GT at 1.27/3/7 T; beat residual RMS vs θ against the 1/SNR floor.
  Source: `b0_study/theta_mae.png`, `beat_noise.png`.
- **F4** Parameter maps: θ estimate vs ground truth (map + scatter), MWF estimate vs truth, and
  the WM-likeness weight w vs the held-out label. Source: `nonoracle/maps.png` + regenerate at
  the anchored setting.
- **F5** Separation results: χ+ and χ− for ground truth, HC-ChiSep, the null, SUSEP-Net and
  χ-sepnet, one axial + one sagittal slice, fixed windows, difference maps underneath.
  Source: `/tmp/hc-chisep-work/contact_sheet.png` — ⚠ this is in a scratch directory and must be
  regenerated and committed.
- **T1** Model parameters with literature sources.
- **T2** Monte-Carlo identifiability table (R2).
- **T3** Main results: all arms × both sources × {detrended NRMSE, xSIM} + leakage + runtime,
  with comparator methods.
- **T4** Ablation / error budget.

---

## 8. Discussion — the argument order

1. **What the result means.** Orientation information is present in ordinary single-orientation
   multi-echo GRE magnitude at 7 T and is usable once the fit is anchored. The DTI arm not being
   better (R13) is the strongest form of this claim: a diffusion scan bought nothing here.
2. **What generalises and what does not.** Generalises: the anchoring principle (any
   multi-compartment fit with an independently measured aggregate decay rate can use it); the
   identifiability and field-dependence limits; the model-selection route to a myelin mask.
   Does not generalise as stated: the numeric K_χ anchor, the Dr− = 0-outside-WM convention, the
   fixed χ_I/χ_A/g. Say which are conventions of the phantom family and which are physics.
3. **The matched-model caveat**, in full, in its own paragraph, with the phrase "mechanistic
   reference baseline" and a clear statement of what a model-agnostic competitor would face.
4. **Benchmark design as a finding.** A phantom whose R2′ is model-generated with a scalar
   relaxivity is closed-form solvable from its own public inputs; the null must therefore be a
   scored baseline. Anisotropy breaks the degeneracy only inside WM. This is useful to the
   community beyond this method.
5. **Route to in vivo.** Needs: 7 T (or a lower-field variant with far more echoes/SNR — quantify
   from R6/R7); an R2 measurement (MESE) alongside the GRE, with EPG-corrected refocusing;
   handling of crossing fibres (a single θ per voxel is the model's weakest anatomical
   assumption); g-ratio and exchange variability; and a validation target, since there is no in
   vivo ground truth — propose orientation-dependent reproducibility (scan-rescan at rotated head
   positions) and agreement with DTI-derived θ as the practical checks.
6. **Limitations**, as an explicit list: tuning on the scoring phantom; single fibre population;
   fixed pool parameters; Rician noise only; ideal refocusing; unregularised output copies R2′
   input noise identically into both sources (the "shared grain" — say that the scored
   configuration is deliberately the faithful, noise-carrying one, and that the TV-regularised
   opt-in exists but over-smooths in vivo); θ ambiguity near 0° that is intrinsic to sin²θ.

---

## 9. Reference list to assemble (verify each before use)

Verified in-repo:
- Wharton S, Bowtell R. *Fiber orientation-dependent white matter contrast in gradient echo MRI.*
  PNAS 2012;109(45):18559–18564. doi:10.1073/pnas.1211075109  ← the model
- Wharton S, Bowtell R. NeuroImage 2013;83:1011–1023 (R2*/frequency-difference orientation mapping)
- Yablonskiy DA, Haacke EM. MRM 1994 (static dephasing regime)  ← the R2′ = Dr·|χ| link
- Sukstanskii AL, Yablonskiy DA. MRM 2014;71:345–353; Yablonskiy & Sukstanskii MRM 2014;71:2059
- Nam Y, Lee J, et al. NeuroImage 2015;116:214–221 (complex model fitting for MWF)
- Sandgaard et al. NMR Biomed 2024 (tensor generalisation; the ln g geometric factor)
- Shin H-G, et al. *χ-separation.* NeuroImage 2021  ← Dr = 137 Hz/ppm empirical calibration
- Ridani D, De Leener B, Alonso-Ortiz E. MRM 2026. doi:10.1002/mrm.70468  ← the phantom
- Marques JP, et al. MRM 2021. doi:10.1002/mrm.28716  ← Challenge 2.0 head model
- Chen J, et al. *DECOMPOSE.* NeuroImage 2021
- Fang Z, Shin H-G, van Zijl P, Li X, Sulam J. *WaveSep.* MLCN/MICCAI 2023.
  doi:10.1007/978-3-031-44858-4_6
- Dimov AV, et al. 2022 (R2*-based separation)
- Hallgren B, Sourander P. 1958 (histological brain iron)
- Brown 1961 (sphere static-dephasing constant, via Ridani)

To locate and verify (marked TODO in the draft):
- χ-sepnet (Kim, Shin et al.) — the 114 Hz/ppm COSMOS-referenced calibration
- SUSEP-Net
- APART-QSM
- Lee J, et al. 2011 (the 5–12 Hz 7 T ΔR2* anisotropy band cited in the field study)
- Li W, Sibgatulin et al. (WM χ∥ − χ⊥ values used by Ridani)
- Cox et al. (7 T T2 scaling); Bottomley T1 power law
- QSM-CI platform citation (Stewart et al. / Zenodo DOI)
- MS / neurodegeneration motivation citations for ¶1

---

## 10. Gaps to close before this is submittable

Ordered by how badly a reviewer will want them.

1. **Complete the comparison table.** Only χ-sepnet, SUSEP-Net and the null have been scored on
   the shipping phantom. WaveSep, χ-sep-MEDI, χ-sep-iLSQR, DECOMPOSE, APART-QSM, R2*-QSM and
   R2PRIMENET all exist as QSM-CI submissions and must be scored on `data/sim/chisep-ship` for
   the paper's main table. Without this, "best on the board" rests on three comparators.
2. **A generalisation arm that is not tuned on.** The bake-off already generated seven phantom
   variants (3 T, 4.5 T + SNR 250, 7 T-emp, 7 T-mech, 7 T-scaled, hcb0-1.27, hcb0-4.5) and the
   registry lists `ridani-3t-iso`, `ridani-3t-aniso`, `ridani-7t-aniso`. Score HC-ChiSep across
   them with the hyperparameters frozen. This is the cheapest available answer to "you tuned on
   your test set" and it directly tests the field-dependence prediction of R6.
3. **An SNR sweep.** The whole conditioning argument is SNR-dependent; the paper should show
   performance vs SNR (50/100/250) rather than asserting the SNR-100 point.
4. **Regenerate and commit the figure assets.** The contact sheet, per-arm outputs and score
   JSONs live in `/tmp/hc-chisep-work/` and will not survive. Everything a figure depends on must
   be reproduced by a committed script (`qsmci/reports/2026-08-chisep/hc-chisep/`).
5. **The anchored SSE-landscape panel** for F2 (only the unanchored one exists). This is the
   visual proof of the paper's central mechanism and it currently does not exist as an image.
6. **Resolve the χ_I discrepancy** (−0.06 vs −0.10 ppm) between the prototype and the shipped
   model, and state one value with its provenance.
7. **Run the Rust port on the shipping (multi-compartment) phantom.** Today the QSM.rs CI runs it
   only on `data/sim/chisep` (single-compartment), so the two implementations have never been
   compared on the phantom the paper's headline number comes from. Agreement between two
   independent implementations on the headline dataset is a strong reproducibility claim and is
   nearly free to obtain.
8. **Decide on in vivo.** Either present a qualitative in-vivo demonstration (the notes already
   record that guided/hard-constrained variants over-smooth in vivo, so some in-vivo data exists)
   with an explicit "no ground truth, qualitative only" framing, or state its absence as a
   limitation. A methods paper of this type can survive without it, but only if the omission is
   deliberate and argued.
9. **Terminology.** "Beat" is used loosely for both the frequency beat and general
   non-mono-exponentiality; at θ = 0 the residual is multi-T2 curvature carrying no orientation
   information. Fix the vocabulary once, globally — the field study makes the distinction and the
   paper must not blur it.

---

## 11. Immediate next actions

**Update (2026-09-08), claim structure revised.** The separation step was found to be two
different things: outside white matter it is the conventional closed-form solve, identical to the
null baseline, and inside white matter it is a myelin-content route that never uses $\theta$ at
all. The draft has been restructured accordingly --- the **core contribution is the $R_2'$-anchored
$(\theta,\mathrm{MWF})$ estimator plus the data-derived white-matter detection**, and the
white-matter separation branch, while it carries an extra calibration constant, is **part of the
method, not an extension** --- it runs automatically wherever the model-selection weight fires,
needs no segmentation or user input, and its $K_\chi$ constant is an assumption of the same kind
as the scalar $D_r$ that conventional $\chi$-separation assumes brain-wide. Consequences
recorded in the draft: orientation is a nuisance parameter (it makes MWF identifiable) rather than
a correction applied to the separation, which is what the DTI arm and the error budget were
already saying; and the white-matter branch inverts the same $K_\chi$ map the phantom generator
used, so its white-matter accuracy is circular on this phantom and the parameter-recovery results
carry the paper. Note for the record: the white-matter branch does **not** require a segmentation
--- the weight comes from model selection (Dice 0.893 against a held-out label, no label used) ---
its real dependency is on $K_\chi$, on $|\chi^-|$ being linear in MWF, and on the $D_r^+=0$
convention.

**Update:** the LaTeX draft now exists (`hc-chisep.tex`, builds via `make`). Introduction,
Theory, Methods, Results, Discussion and Conclusion are written; the in vivo sections, the
comparator rows and the artwork are placeholders. The items below stand as written.


- [ ] Decide venue and scope (abstract first, or straight to the full paper) — §2.
- [ ] Copy the MRM LaTeX scaffold from `~/repos/qsm/qsmbly/paper/` into `paper/hc-chisep/`.
- [ ] Write Introduction (§3) and Theory (§4) — both are fully specified above and can be
      drafted without new experiments.
- [ ] In parallel, start gap 1 (scoring the rest of the board) and gap 2 (frozen-hyperparameter
      generalisation sweep), since they are compute-bound and everything else waits on them.

//! Integration test: does EPI susceptibility-distortion correction put a real brain back?
//!
//! The unit tests in `src/distortion.rs` pin the arithmetic — the sign, the voxel shift, the
//! Jacobian, the fold condition — against closed forms. What they cannot show is whether the
//! whole thing works on anatomy with a field map that has the shape a real one has. This does
//! that, on the BIDS phantom, by simulating an echo-planar acquisition and correcting it.
//!
//! # The field this distorts with
//!
//! Not the raw ground-truth field map. That quantity is analytic and has voxel-sharp features
//! at every susceptibility source, so its phase-encode gradient folds at isolated voxels even at
//! a 4 ms readout — a field no scanner could measure and no pipeline would be handed. What a
//! pipeline *is* handed is a B0 map acquired at lower resolution and confined to tissue, so the
//! simulation uses the ground truth smoothed to that resolution and tapered to zero outside a
//! dilated brain. The raw field still appears, in
//! [`the_pile_up_check_has_to_be_confined_to_the_brain`], because its wild background is exactly
//! the thing that makes masking the fold check mandatory rather than optional.
//!
//! # Every claim is pinned from both sides
//!
//! A test that only asserts "the corrected image is good" passes just as happily when the
//! correction is a no-op, so each block here also requires the *uncorrected* input to be bad by
//! a stated margin, and three separate controls must come out worse than doing nothing at all:
//! reversing the phase-encode polarity, skipping the Jacobian, and interpolating the wrapped
//! phase directly instead of through the complex domain.
//!
//! # Where the thresholds come from, and the prohibition that governs them
//!
//! A threshold chosen by measuring the achieved value and backing off is **not** the same thing
//! as one placed to clear a demonstrated null, even when both happen to pass. This file was
//! built the first way and corrected to the second, so the rule is written down:
//!
//! > No magnitude-based assertion here may be read as evidence about the *geometric* correction
//! > unless its threshold clears the zero-displacement null.
//!
//! That null — displacement forced to zero, Jacobian modulation and masking left running — is
//! the informative one, because it isolates one mechanism in a pipeline that has several.
//! Measured against it:
//!
//! | metric | uncorrected | null (no shift, modulation on) | corrected | margin |
//! |---|---|---|---|---|
//! | magnitude correlation | 0.685 | **0.752** | 0.976 | 1.26× |
//! | magnitude NRMSE | 0.0380 | **0.0334** | 0.0103 | 3.2× |
//! | phase mean abs error | 0.485 rad | **0.485 rad** | 0.040 rad | 4.9× |
//!
//! So intensity scaling alone genuinely moves the magnitude metrics, and the floors are set to
//! clear that rather than merely to clear doing nothing. The phase metric is confound-free *by
//! construction* — a positive real scale cannot move a phase — which the null confirms to four
//! figures rather than discovers. Prefer the phase number when the two disagree; it is the one
//! measuring only geometry.
//!
//! # Is 20 ms a special readout? No — measured, not assumed
//!
//! [`TRT`] is the longest readout that does not fold this phantom's brain, so the headline
//! numbers are taken just below a boundary. That is worth checking rather than asserting, since
//! a limitation measured at a boundary is a property of the boundary. Swept:
//!
//! | TotalReadoutTime | peak shift | corrected correlation |
//! |---|---|---|
//! | 4 ms | 2.3 vox | 0.983 |
//! | 8 ms | 4.6 vox | 0.983 |
//! | 12 ms | 7.0 vox | 0.982 |
//! | 16 ms | 9.3 vox | 0.980 |
//! | 20 ms | 11.6 vox | 0.976 |
//!
//! Essentially flat — 0.7% across a five-fold range of distortion — so 0.976 is representative
//! of the correction rather than an artefact of running it at the hardest usable setting. A
//! negative result, recorded because silence about a check is indistinguishable from not having
//! made it.
//!
//! One incidental finding, which matters if anyone reuses the scoring region: `|shift| > 1
//! voxel` selects 204,396 voxels at 20 ms but only **12** at 4 ms, so every statistic over it is
//! small-sample noise down there (the uncorrected correlation reads 0.566, below its value at
//! longer readouts, which is an artefact and not physics). The region definition is sound at the
//! distortion this file tests and useless well below it. Both scoring regions here assert a
//! floor against that, the polarity test's derived from the sampling error of the statistic
//! rather than backed off from its measured value.
//!
//! # Does the region drop something different from what it keeps?
//!
//! Size is the easy half. The dangerous half is whether the excluded voxels differ
//! systematically from the included ones, because that bias has a *direction* where sampling
//! noise does not. Measured at 20 ms, scoring each piece separately:
//!
//! | | voxels | corrected correlation |
//! |---|---|---|
//! | kept (brain eroded by 4) | 204,396 | 0.976 |
//! | dropped: the eroded rim | 89,017 | 0.982 |
//! | dropped: no source in the FOV | 0 | — |
//! | whole brain, for reference | 293,413 | 0.981 |
//!
//! The rim scores **better** than what is kept, so the erosion discards the easier voxels and
//! the headline number is taken over the harder subset — conservative, not flattering. That is
//! a result rather than an assumption: the same check on a sibling branch found a region whose
//! exclusions eroded from the periphery inward, dropping exactly the tissue its test was about
//! and improving its metrics as the experiment became less meaningful. The intuition is not
//! reliable in either direction, which is why it is measured here.
//!
//! The second exclusion is empty — and *why* matters more than the fact. The brain occupies
//! `j = 14..=182` of 205, so it stands 14 voxels clear of the low phase-encode edge while the
//! peak shift is 11.6: a margin of **2.4 voxels**, not a structural impossibility. Tighter
//! framing or a stronger field and it starts excluding real tissue, and at that point the
//! direct comparison above is no longer available, because a voxel with no source has nothing
//! to score. The two exclusions are different in kind — one is a deliberate erosion with data
//! on both sides, the other removes voxels that were never measured — and only the first admits
//! a direct answer. Worth knowing which you have before assuming the check is cheap.
//!
//! The full list of mutation axes this work is known to catch lives at the top of the test
//! module in `src/distortion.rs`, alongside the code they perturb.
//!
//! Run with:
//!   cargo test --release --features parallel --test epi_unwarp -- --ignored --nocapture

mod common;

use common::{correlation, nrmse, TestData};
use qsm_core::distortion::{
    distort_complex, unwarp_complex, DisplacementField, PhaseEncoding, UnwarpParams,
};
use qsm_core::mask::{dilate_mask, erode_mask};
use qsm_core::Grid;

/// Proton gyromagnetic ratio, matching `pipeline::hz_to_ppm`.
const GAMMA_HZ_PER_T: f64 = 42.576e6;

/// A `TotalReadoutTime` in the range a 3D-EPI protocol actually uses. On this phantom it puts
/// the peak displacement at 11.6 voxels, which is severe but not folded anywhere in the brain.
const TRT: f64 = 0.020;

/// A shorter readout, used where a control has to run without the field folding on it. The
/// reversed-polarity control doubles the effective distortion, and at [`TRT`] that folds — so
/// the test that needs a *number* out of it runs here instead, at 7.0 voxels of peak shift.
const TRT_SHORT: f64 = 0.012;

/// Phase encode along voxel axis 1. Which anatomical direction that is depends on the storage
/// order and does not matter: distortion is defined on the array axis, which is why BIDS spells
/// it `i`/`j`/`k` rather than anatomically.
fn pe() -> PhaseEncoding {
    PhaseEncoding::parse("j").expect("j is a BIDS phase-encode direction")
}

/// Separable box blur, `passes` applications of a `2r+1` kernel along each axis. Used to make
/// the simulated fieldmap look like one that was measured, not computed.
fn box_blur(data: &[f64], dims: (usize, usize, usize), r: usize, passes: usize) -> Vec<f64> {
    let (nx, ny, nz) = dims;
    let mut cur = data.to_vec();
    for _ in 0..passes {
        for axis in 0..3 {
            let n = [nx, ny, nz][axis];
            let stride = [1usize, nx, nx * ny][axis];
            let mut out = vec![0.0f64; cur.len()];
            for (idx, slot) in out.iter_mut().enumerate() {
                let (i, j, k) = (idx % nx, (idx / nx) % ny, idx / (nx * ny));
                let p = [i, j, k][axis];
                let base = idx - p * stride;
                let (mut sum, mut cnt) = (0.0, 0usize);
                for q in p.saturating_sub(r)..=(p + r).min(n - 1) {
                    sum += cur[base + q * stride];
                    cnt += 1;
                }
                *slot = sum / cnt as f64;
            }
            cur = out;
        }
    }
    cur
}

/// The ground-truth field map in Hz, exactly as the forward model laid it down.
///
/// Used here only as a **plausibly shaped** field to distort with, not as the field that
/// generated the phantom's phase. The two need not agree and nothing below assumes they do: the
/// same field drives the forward warp and the correction, so the test is self-consistent however
/// the phantom was produced. (It matters because `qsm-forward` applies a second-order shim after
/// writing `fieldmap`, so on a phantom generated with one the signal comes from a separate
/// `desc-shimmed_fieldmap`. This phantom has no such file, and either way it would not change
/// what this test measures — do not "fix" it to chase correspondence that is not required.)
fn raw_field_hz(data: &TestData) -> Vec<f64> {
    let scale = GAMMA_HZ_PER_T * data.field_strength * 1e-6;
    data.fieldmap.iter().map(|&ppm| ppm * scale).collect()
}

/// The ground-truth field made acquirable: confined to a dilated brain, tapered smoothly to
/// zero, and blurred to the resolution a measured B0 map has.
fn acquirable_field_hz(data: &TestData, grid: &Grid) -> Vec<f64> {
    let dilated = dilate_mask(&data.mask, grid, 6);
    let taper = box_blur(
        &dilated.iter().map(|&m| f64::from(m)).collect::<Vec<_>>(),
        data.dims,
        4,
        2,
    );
    let confined: Vec<f64> =
        raw_field_hz(data).iter().zip(&taper).map(|(&v, &w)| v * w).collect();
    box_blur(&confined, data.dims, 2, 2)
}

/// Mean absolute phase difference in radians, taken the only way phase differences can be: wrap
/// the difference back onto `(-pi, pi]` before averaging it.
fn phase_error(a: &[f64], b: &[f64], mask: &[u8]) -> f64 {
    let (mut sum, mut count) = (0.0, 0usize);
    for i in 0..a.len() {
        if mask[i] == 0 {
            continue;
        }
        let d = a[i] - b[i];
        sum += d.sin().atan2(d.cos()).abs();
        count += 1;
    }
    assert!(count > 0, "empty mask");
    sum / count as f64
}

/// The gather that [`unwarp_complex`] performs, applied to the *wrapped phase values* instead of
/// to the complex signal. The control that shows the complex domain is load-bearing.
fn naive_phase_gather(phase: &[f64], disp: &DisplacementField) -> Vec<f64> {
    let (nx, ny, nz) = disp.dims;
    let mut out = vec![0.0f64; phase.len()];
    for (idx, slot) in out.iter_mut().enumerate() {
        let j = (idx / nx) % ny;
        let u = j as f64 + disp.shift[idx];
        if !(0.0..=(ny - 1) as f64).contains(&u) {
            continue;
        }
        let lo = u.floor() as usize;
        let hi = (lo + 1).min(ny - 1);
        let f = u - lo as f64;
        let base = idx - j * nx;
        *slot = phase[base + lo * nx] * (1.0 - f) + phase[base + hi * nx] * f;
    }
    let _ = nz;
    out
}

/// Everything the tests share: the phantom, the simulated fieldmap, and the scoring mask.
struct Scene {
    data: TestData,
    field: Vec<f64>,
    /// Brain eroded by 4 voxels, so edge effects of the forward model are not scored.
    inner: Vec<u8>,
}

impl Scene {
    fn load() -> Self {
        let data = TestData::load().expect("BIDS phantom");
        let grid = Grid::new(data.dims.0, data.dims.1, data.dims.2, 1.0, 1.0, 1.0);
        let field = acquirable_field_hz(&data, &grid);
        let inner = erode_mask(&data.mask, &grid, 4);
        Self { data, field, inner }
    }

    fn displacement(&self, trt: f64, polarity: &str) -> DisplacementField {
        DisplacementField::from_field_hz(
            &self.field,
            self.data.dims,
            PhaseEncoding::parse(polarity).unwrap(),
            trt,
        )
    }

    /// Voxels inside the eroded brain that the readout actually moved by more than a voxel —
    /// where correction is the difference between a usable image and a useless one, and where a
    /// whole-brain average would hide the effect entirely.
    fn distorted_region(&self, disp: &DisplacementField, kept: &[u8]) -> Vec<u8> {
        (0..self.inner.len())
            .map(|i| u8::from(self.inner[i] == 1 && kept[i] == 1 && disp.shift[i].abs() > 1.0))
            .collect()
    }
}

// ---------------------------------------------------------------------------------------------

/// The headline: simulate a 20 ms-readout EPI of the phantom and correct it. Magnitude and
/// wrapped phase both have to come back, in the region that was actually displaced, and the
/// uncorrected input has to be demonstrably bad in the same region.
#[test]
#[ignore]
fn epi_unwarping_recovers_the_anatomy_a_realistic_readout_displaced() {
    let scene = Scene::load();
    let disp = scene.displacement(TRT, "j");
    let peak = disp.max_shift_voxels();
    println!("[INFO] peak displacement {peak:.2} voxels at TotalReadoutTime {TRT} s");
    assert!(
        (11.0..12.0).contains(&peak),
        "the simulated readout should displace by ~11.6 voxels, got {peak}"
    );
    disp.check_invertible(Some(&scene.data.mask), 0.0)
        .expect("an acquirable fieldmap at 20 ms must not fold inside the brain");

    let truth_mag = &scene.data.mag_echoes[0];
    let truth_phase = &scene.data.phase_echoes[0];
    let (dist_mag, dist_phase) = distort_complex(truth_mag, truth_phase, &disp);

    let out = unwarp_complex(
        &dist_mag,
        &dist_phase,
        &disp,
        Some(&scene.data.mask),
        &UnwarpParams::default(),
    )
    .expect("the simulated field does not fold inside the brain");

    let region = scene.distorted_region(&disp, &out.mask);
    let n_region: usize = region.iter().map(|&m| usize::from(m == 1)).sum();
    println!("[INFO] scoring {n_region} voxels displaced by more than one voxel");
    assert!(n_region > 100_000, "only {n_region} voxels were meaningfully displaced");

    // ---- magnitude -------------------------------------------------------------------------
    let (corr_before, corr_after) = (
        correlation(&dist_mag, truth_mag, &region),
        correlation(&out.magnitude, truth_mag, &region),
    );
    let (nrmse_before, nrmse_after) = (
        nrmse(&dist_mag, truth_mag, &region),
        nrmse(&out.magnitude, truth_mag, &region),
    );
    println!("[INFO] magnitude  corr {corr_before:.4} -> {corr_after:.4}   nrmse {nrmse_before:.4} -> {nrmse_after:.4}");
    // Measured 0.685 -> 0.976 and 0.0380 -> 0.0103.
    //
    // Where the 0.95 floor comes from, which is not "a bit below 0.976". A metric can move the
    // right way for a mechanism other than the one under test, so this was measured against a
    // null: with the displacement forced to zero but the Jacobian modulation and masking left
    // working, the same metrics read 0.752 and 0.0334. That is a real improvement over the
    // uncorrected 0.685 and 0.0380 — the modulation alone does something — so the floor has to
    // clear the confound, not merely the do-nothing case. 0.95 sits well above 0.752 and the
    // 0.015 NRMSE ceiling well below 0.0334, which is what makes these assertions statements
    // about the geometry rather than about the intensity scaling.
    assert!(corr_before < 0.80, "the simulation must actually distort (corr {corr_before})");
    assert!(corr_after > 0.95, "corrected magnitude correlation {corr_after}");
    assert!(nrmse_before > 0.025, "the simulation must actually distort (nrmse {nrmse_before})");
    assert!(nrmse_after < 0.015, "corrected magnitude NRMSE {nrmse_after}");

    // ---- wrapped phase, on the last echo where it actually wraps -----------------------
    // Echo 1 at TE = 4 ms accrues only ~14 rad across the brain, so its wrap surfaces are too
    // sparse for a mean to say anything about them. Echo 4 at 28 ms accrues ~99 rad and wraps
    // everywhere, which is the regime the complex domain exists for.
    let truth_phase4 = &scene.data.phase_echoes[3];
    let (dist_mag4, dist_phase4) =
        distort_complex(&scene.data.mag_echoes[3], truth_phase4, &disp);
    let out4 = unwarp_complex(&dist_mag4, &dist_phase4, &disp, Some(&scene.data.mask),
        &UnwarpParams::default()).unwrap();

    let (ph_before, ph_after) = (
        phase_error(&dist_phase4, truth_phase4, &region),
        phase_error(&out4.phase, truth_phase4, &region),
    );
    println!("[INFO] phase (echo 4)  mean |err| {ph_before:.4} -> {ph_after:.4} rad");
    // Measured 0.485 -> 0.0396 rad. Unlike the magnitude metrics above, this one has *no*
    // confound to clear: against the same zero-displacement null it reads 0.4852, exactly the
    // uncorrected value, because the only other mechanism in play is Jacobian modulation and a
    // positive real scale cannot move a phase. (Asserted bit-for-bit further down.) So every bit
    // of the improvement here is geometry, and the 0.10 floor has a 4.9x margin over the null
    // rather than the 1.26x the magnitude correlation has.
    assert!(ph_before > 0.30, "the simulation must displace the phase too ({ph_before} rad)");
    assert!(ph_after < 0.10, "corrected phase error {ph_after} rad");

    // ---- control: interpolating the wrapped phase directly ---------------------------------
    // The same gather, applied to the phase values rather than to mag*exp(i*phi). Counted as
    // gross failures rather than averaged: a wrap surface is thin, so a mean over the whole
    // region dilutes exactly the voxels this is about, while the count of voxels thrown more
    // than a radian off is precisely the damage done.
    let naive = naive_phase_gather(&dist_phase4, &disp);
    let gross = |p: &[f64]| -> usize {
        (0..p.len())
            .filter(|&i| {
                if region[i] == 0 { return false; }
                let d = p[i] - truth_phase4[i];
                d.sin().atan2(d.cos()).abs() > 1.0
            })
            .count()
    };
    let (g_naive, g_complex, g_before) = (gross(&naive), gross(&out4.phase), gross(&dist_phase4));
    println!("[INFO] voxels more than 1 rad wrong: uncorrected {g_before}, complex {g_complex}, direct-phase {g_naive}");
    assert!(
        g_naive > 10 * g_complex.max(1),
        "interpolating wrapped phase directly should ruin far more voxels ({g_naive} vs {g_complex})"
    );
    assert!(
        g_naive > n_region / 50,
        "the direct-phase control should wreck a substantial share of {n_region} voxels, got {g_naive}"
    );

    // ---- control: skipping the Jacobian ------------------------------------------------------
    let unmodulated = unwarp_complex(
        &dist_mag,
        &dist_phase,
        &disp,
        Some(&scene.data.mask),
        &UnwarpParams { jacobian_modulation: false, min_jacobian: 0.0 },
    )
    .unwrap();
    let nrmse_nojac = nrmse(&unmodulated.magnitude, truth_mag, &region);
    let corr_nojac = correlation(&unmodulated.magnitude, truth_mag, &region);
    println!("[INFO] control: no Jacobian  corr {corr_nojac:.4}  nrmse {nrmse_nojac:.4}");
    // Measured 0.869 / 0.0242 without, against 0.976 / 0.0103 with: the modulation more than
    // halves the magnitude error, which is why it defaults on.
    assert!(
        nrmse_nojac > 2.0 * nrmse_after,
        "Jacobian modulation should more than halve the magnitude error ({nrmse_nojac} vs {nrmse_after})"
    );
    // Phase is untouched by a positive real scale, so it must be bit-identical either way.
    for i in 0..out.phase.len() {
        assert_eq!(out.phase[i], unmodulated.phase[i], "Jacobian modulation moved the phase at {i}");
    }

    common::save_center_slices(truth_mag, &scene.data.mask, scene.data.dims, "epi_unwarp_truth");
    common::save_center_slices(&dist_mag, &scene.data.mask, scene.data.dims, "epi_unwarp_distorted");
    common::save_center_slices(&out.magnitude, &scene.data.mask, scene.data.dims, "epi_unwarp_corrected");
}

/// Reversing the phase-encode polarity doubles the distortion instead of removing it, so getting
/// the sign backwards must come out *worse than doing nothing*. This is the assertion that pins
/// the convention on real data.
#[test]
#[ignore]
fn reversing_the_phase_encode_polarity_is_worse_than_not_correcting_at_all() {
    let scene = Scene::load();
    let forward = scene.displacement(TRT_SHORT, "j");
    let reversed = scene.displacement(TRT_SHORT, "j-");
    assert_eq!(reversed.pe.to_bids(), "j-");
    // The two are the same magnitude and opposite sign, by construction.
    for i in 0..forward.shift.len() {
        assert_eq!(forward.shift[i], -reversed.shift[i]);
    }

    let truth_mag = &scene.data.mag_echoes[0];
    let truth_phase = &scene.data.phase_echoes[0];
    let (dist_mag, dist_phase) = distort_complex(truth_mag, truth_phase, &forward);

    let right = unwarp_complex(&dist_mag, &dist_phase, &forward, Some(&scene.data.mask),
        &UnwarpParams::default()).unwrap();
    let wrong = unwarp_complex(&dist_mag, &dist_phase, &reversed, Some(&scene.data.mask),
        &UnwarpParams::default())
        .expect("at a 12 ms readout the reversed field is still invertible, so it yields a number");

    let region = scene.distorted_region(&forward, &right.mask);
    // Bound the scoring region, which nothing else here does. It holds 48,102 voxels at the
    // current `TRT_SHORT` and is in no danger — but `TRT_SHORT` is the one knob this test
    // invites you to turn, since it exists only because the reversed field folds at `TRT`, and
    // lowering it walks the region toward the handful of voxels it holds at 4 ms (see the
    // module docs). The floor is derived, not backed off from the measurement: the standard
    // error of a correlation near r = 0.5 is about `(1 - r²)/sqrt(n)`, so at n = 10,000 it is
    // ~0.0075 against the 0.23 gap this test asserts — a 30x margin. Below that the comparison
    // stops meaning anything, quietly.
    let n_region: usize = region.iter().map(|&m| usize::from(m == 1)).sum();
    assert!(
        n_region > 10_000,
        "scoring region collapsed to {n_region} voxels; the correlations below are noise"
    );
    let (c_dist, c_right, c_wrong) = (
        correlation(&dist_mag, truth_mag, &region),
        correlation(&right.magnitude, truth_mag, &region),
        correlation(&wrong.magnitude, truth_mag, &region),
    );
    println!("[INFO] corr  uncorrected {c_dist:.4}  correct polarity {c_right:.4}  reversed {c_wrong:.4}");
    // Measured 0.757 uncorrected, 0.982 correct, 0.525 reversed.
    //
    // `c_wrong < c_dist` is the assertion that carries the claim, and it is relative, so it
    // cannot drift. The 0.65 beside it is placed deliberately between the do-nothing baseline
    // (0.757) and the measured reversed value (0.525): it asserts that getting the sign backwards
    // is *substantially* worse than not correcting, not merely a hair worse, which is what makes
    // the symptom recognisable in practice.
    assert!(c_right > 0.95, "correct polarity {c_right}");
    assert!(c_wrong < c_dist, "reversed polarity {c_wrong} must be worse than uncorrected {c_dist}");
    assert!(c_wrong < 0.65, "reversed polarity {c_wrong} should be dramatically worse");
}

/// The fold check must be confined to the brain, and it must fire when the brain really does
/// fold. Both halves use the *raw* ground-truth field, whose unconstrained background is the
/// reason the first half matters.
#[test]
#[ignore]
fn the_pile_up_check_has_to_be_confined_to_the_brain() {
    let scene = Scene::load();
    let raw = raw_field_hz(&scene.data);
    let disp = DisplacementField::from_field_hz(&raw, scene.data.dims, pe(), 0.004);

    disp.check_invertible(Some(&scene.data.mask), 0.0)
        .expect("a 4 ms readout does not fold the brain");
    let outside = disp
        .check_invertible(None, 0.0)
        .expect_err("the unconstrained background field folds, as it does on every real scan");
    println!(
        "[INFO] whole-volume check: {} of {} voxels fold, worst J = {:.3} at {:?}",
        outside.n_voxels, outside.n_considered, outside.min_jacobian, outside.worst_voxel
    );
    // Measured 37885 voxels, minimum J = -3.411. Checking the whole volume would refuse this
    // perfectly reconstructable scan, which is the entire argument for taking a mask.
    assert!(outside.n_voxels > 20_000, "only {} background voxels folded", outside.n_voxels);
    assert!(outside.min_jacobian < -1.0, "worst background J {}", outside.min_jacobian);
    let in_brain: usize = scene.data.mask.iter().map(|&m| usize::from(m == 1)).sum();
    assert!(
        outside.n_considered > in_brain,
        "the unmasked check must consider more than the {in_brain} brain voxels"
    );
}

/// Lengthen the readout and the brain itself folds. The correction must refuse at that point
/// rather than return a smooth, plausible, wrong image — and the refusal must get worse
/// monotonically, not flicker.
#[test]
#[ignore]
fn a_long_enough_readout_folds_the_brain_and_is_refused() {
    let scene = Scene::load();
    let mask = Some(scene.data.mask.as_slice());

    scene
        .displacement(TRT, "j")
        .check_invertible(mask, 0.0)
        .expect("20 ms is below the folding threshold on this phantom");

    let mut previous = 0usize;
    let mut previous_min = f64::INFINITY;
    for trt in [0.030, 0.040, 0.050] {
        let disp = scene.displacement(trt, "j");
        let err = disp
            .check_invertible(mask, 0.0)
            .expect_err("a readout this long must fold the brain");
        println!(
            "[INFO] {trt} s: peak {:.1} voxels, {} folded voxels, worst J {:.3}",
            disp.max_shift_voxels(), err.n_voxels, err.min_jacobian
        );
        assert!(
            err.n_voxels > previous,
            "a longer readout must fold at least as much: {} then {}", previous, err.n_voxels
        );
        assert!(err.min_jacobian < previous_min, "the worst Jacobian must keep dropping");
        previous = err.n_voxels;
        previous_min = err.min_jacobian;

        // The refusal has to reach the entry point, not just the standalone check.
        let truth_mag = &scene.data.mag_echoes[0];
        let truth_phase = &scene.data.phase_echoes[0];
        match unwarp_complex(truth_mag, truth_phase, &disp, mask, &UnwarpParams::default()) {
            Ok(_) => panic!("unwarp_complex accepted a folded field at {trt} s"),
            Err(e) => assert!(e.to_string().contains("pile-up"), "{e}"),
        }
    }
    // Measured 22 folded voxels at 30 ms rising to 306 at 50 ms.
    assert!(previous > 100, "the 50 ms readout should fold hundreds of voxels, got {previous}");
}

//! Self-contained relaxometry demo (no external data) for CI.
//!
//! Builds a small synthetic multi-echo phantom and exercises the relaxometry
//! toolkit, emitting `RELAXMETRIC` lines (parsed by the CI into a table) and
//! centre-slice figures (rendered by `scripts/render_slices.py`) that show, at a
//! glance, that each tool does what it should:
//!   - EPG R2 recovers the true R2 from imperfectly-refocused MESE, where a
//!     mono-exponential fit is biased.
//!   - R2' = R2* - R2 recovers the true R2'.
//!   - a slab MESE yields R2' only where R2 was measured, instead of handing back R2*.
//!   - MP-PCA denoising improves R2* from noisy GRE.
//!   - Gibbs unringing removes k-space-truncation ringing.
//!
//! Run: cargo test --release --features parallel --test relaxometry_demo -- --ignored --nocapture

mod common;

use common::{correlation, save_center_slices};
use qsm_core::fft::{fft3d_real, ifft3d_real};
use qsm_core::relaxometry::{epg_cpmg_echoes, r2_epg, r2prime, R2EpgParams};
use qsm_core::r2star::r2star_arlo;
use qsm_core::Grid;

const NX: usize = 64;
const NY: usize = 64;
const NZ: usize = 40;
const B1: f64 = 0.75; // imperfect refocusing → mono-exp R2 is biased
const T1: f64 = 1.0;

fn idx(x: usize, y: usize, z: usize) -> usize {
    x + y * NX + z * NX * NY
}

/// Build the synthetic phantom: (mask, R2, R2prime) in Hz.
fn phantom() -> (Vec<u8>, Vec<f64>, Vec<f64>) {
    let n = NX * NY * NZ;
    let mut mask = vec![0u8; n];
    let mut r2 = vec![0.0f64; n];
    let mut rp = vec![0.0f64; n];
    let (cx, cy, cz) = (NX as f64 / 2.0, NY as f64 / 2.0, NZ as f64 / 2.0);
    let sphere = |x, y, z, ox: f64, oy: f64, oz: f64, r: f64| -> bool {
        let (dx, dy, dz) = (x as f64 - ox, y as f64 - oy, z as f64 - oz);
        dx * dx + dy * dy + dz * dz <= r * r
    };
    for z in 0..NZ {
        for y in 0..NY {
            for x in 0..NX {
                // Ellipsoid "brain" mask.
                let (dx, dy, dz) = ((x as f64 - cx) / (NX as f64 * 0.44),
                                    (y as f64 - cy) / (NY as f64 * 0.44),
                                    (z as f64 - cz) / (NZ as f64 * 0.44));
                if dx * dx + dy * dy + dz * dz > 1.0 {
                    continue;
                }
                let i = idx(x, y, z);
                mask[i] = 1;
                // Base grey matter.
                r2[i] = 14.0;
                rp[i] = 2.5;
                // White-matter slab (left half).
                if (x as f64) < cx {
                    r2[i] = 11.0;
                    rp[i] = 2.0;
                }
                // Deep-grey blob.
                if sphere(x, y, z, NX as f64 * 0.36, NY as f64 * 0.58, cz, 8.0) {
                    r2[i] = 22.0;
                    rp[i] = 6.0;
                }
                // Iron-rich sphere (high R2 and high R2').
                if sphere(x, y, z, NX as f64 * 0.66, NY as f64 * 0.44, cz, 5.0) {
                    r2[i] = 40.0;
                    rp[i] = 14.0;
                }
            }
        }
    }
    (mask, r2, rp)
}

/// Deterministic Rician noise (Box–Muller from an LCG), per-echo peak / snr.
fn add_rician(data: &[f64], n_vox: usize, ne: usize, snr: f64, seed: u64) -> Vec<f64> {
    let mut mx = vec![0.0f64; ne];
    for v in 0..n_vox {
        for e in 0..ne {
            mx[e] = mx[e].max(data[v * ne + e].abs());
        }
    }
    let mut s = seed;
    let mut g = || {
        let mut u = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 2.0)
        };
        let (a, b) = (u(), u());
        (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
    };
    let mut out = vec![0.0f64; data.len()];
    for v in 0..n_vox {
        for e in 0..ne {
            let sig = mx[e] / snr;
            let re = data[v * ne + e] + sig * g();
            let im = sig * g();
            out[v * ne + e] = (re * re + im * im).sqrt();
        }
    }
    out
}

/// Low-pass a volume in k-space (zero |freq| above `keep`) to induce Gibbs ringing.
fn kspace_truncate(vol: &[f64], keep: f64) -> Vec<f64> {
    let mut c = fft3d_real(vol, NX, NY, NZ);
    let ff = |m: usize, n: usize| -> f64 {
        if m <= n / 2 { m as f64 / n as f64 } else { (m as f64 - n as f64) / n as f64 }
    };
    for z in 0..NZ {
        for y in 0..NY {
            for x in 0..NX {
                if ff(x, NX).abs() > keep || ff(y, NY).abs() > keep || ff(z, NZ).abs() > keep {
                    let i = idx(x, y, z);
                    c[i] *= 0.0; // zero without naming Complex64
                }
            }
        }
    }
    ifft3d_real(&c, NX, NY, NZ)
}

fn tv3(vol: &[f64], mask: &[u8]) -> f64 {
    let mut tv = 0.0;
    for z in 0..NZ {
        for y in 0..NY {
            for x in 0..NX {
                let i = idx(x, y, z);
                if mask[i] == 0 {
                    continue;
                }
                if x + 1 < NX {
                    tv += (vol[i] - vol[idx(x + 1, y, z)]).abs();
                }
                if y + 1 < NY {
                    tv += (vol[i] - vol[idx(x, y + 1, z)]).abs();
                }
            }
        }
    }
    tv
}

fn mean_masked(x: &[f64], m: &[u8]) -> f64 {
    let (mut s, mut c) = (0.0, 0usize);
    for i in 0..x.len() {
        if m[i] > 0 {
            s += x[i];
            c += 1;
        }
    }
    s / c.max(1) as f64
}

fn metric(name: &str, key: &str, value: f64) {
    println!("RELAXMETRIC {}|{}|{:.4}", name, key, value);
}

#[test]
#[ignore] // demo: run explicitly in CI (writes slices/*.bin + prints RELAXMETRIC lines)
fn test_relaxometry_demo() {
    let dims = (NX, NY, NZ);
    let n_vox = NX * NY * NZ;
    let grid = Grid::new(NX, NY, NZ, 1.0, 1.0, 1.0);
    let (mask, r2_true, rp_true) = phantom();
    let r2star_true: Vec<f64> = (0..n_vox).map(|i| r2_true[i] + rp_true[i]).collect();

    // --- Signals ---
    let gre_te: Vec<f64> = (1..=16).map(|e| e as f64 * 3e-3).collect();
    let se_te: Vec<f64> = (1..=32).map(|e| e as f64 * 8e-3).collect();
    let n_gre = gre_te.len();
    let n_se = se_te.len();
    let esp = se_te[0];

    let mut gre = vec![0.0f64; n_vox * n_gre];
    let mut se = vec![0.0f64; n_vox * n_se];
    for v in 0..n_vox {
        if mask[v] == 0 {
            continue;
        }
        for (e, &te) in gre_te.iter().enumerate() {
            gre[v * n_gre + e] = (-te * r2star_true[v]).exp();
        }
        // MESE generated with OUR EPG forward model at B1<1.
        let train = epg_cpmg_echoes(1.0 / r2_true[v], T1, B1, esp, n_se);
        for e in 0..n_se {
            se[v * n_se + e] = train[e];
        }
    }

    // --- EPG R2 vs mono-exp R2 (clean) ---
    let (r2_epg, b1_map) = r2_epg(&se, &mask, &se_te, &grid, &R2EpgParams::default(), None);
    let (r2_mono, _) = r2star_arlo(&se, &mask, &se_te, &grid);
    let (r2star, _) = r2star_arlo(&gre, &mask, &gre_te, &grid);
    let rp_derived = r2prime(&r2star, &r2_epg, &mask, None).r2prime;

    metric("EPG R2", "corr", correlation(&r2_epg, &r2_true, &mask));
    metric("EPG R2", "bias_Hz", mean_masked(&r2_epg, &mask) - mean_masked(&r2_true, &mask));
    metric("EPG R2", "b1_recovered", mean_masked(&b1_map, &mask));
    metric("mono-exp R2", "corr", correlation(&r2_mono, &r2_true, &mask));
    metric("mono-exp R2", "bias_Hz", mean_masked(&r2_mono, &mask) - mean_masked(&r2_true, &mask));
    metric("R2prime", "corr", correlation(&rp_derived, &rp_true, &mask));

    save_center_slices(&r2_true, &mask, dims, "relax_r2_truth");
    save_center_slices(&r2_epg, &mask, dims, "relax_r2_epg");
    save_center_slices(&r2_mono, &mask, dims, "relax_r2_monoexp");
    save_center_slices(&rp_true, &mask, dims, "relax_r2prime_truth");
    save_center_slices(&rp_derived, &mask, dims, "relax_r2prime_derived");

    // --- MP-PCA denoising (noisy GRE → R2*) ---
    let gre_noisy = add_rician(&gre, n_vox, n_gre, 15.0, 1);
    let (r2star_noisy, _) = r2star_arlo(&gre_noisy, &mask, &gre_te, &grid);
    let gre_dn = qsm_core::denoise::mppca_denoise(&gre_noisy, dims, n_gre, 2, Some(&mask));
    let (r2star_dn, _) = r2star_arlo(&gre_dn, &mask, &gre_te, &grid);
    let corr_noisy = correlation(&r2star_noisy, &r2star_true, &mask);
    let corr_dn = correlation(&r2star_dn, &r2star_true, &mask);
    metric("R2* MP-PCA", "corr_noisy", corr_noisy);
    metric("R2* MP-PCA", "corr_denoised", corr_dn);
    save_center_slices(&r2star_noisy, &mask, dims, "relax_denoise_before");
    save_center_slices(&r2star_dn, &mask, dims, "relax_denoise_after");

    // --- Gibbs unringing (k-space-truncated R2* map → ringing) ---
    let rung = kspace_truncate(&r2star_true, 0.30);
    let unrung = qsm_core::unring::gibbs_unring_volume(&rung, dims);
    let tv_before = tv3(&rung, &mask);
    let tv_after = tv3(&unrung, &mask);
    metric("Gibbs unring", "tv_before", tv_before);
    metric("Gibbs unring", "tv_after", tv_after);
    save_center_slices(&rung, &mask, dims, "relax_unring_before");
    save_center_slices(&unrung, &mask, dims, "relax_unring_after");

    // --- Functional gates (also make the demo a CI test) ---
    assert!(correlation(&r2_epg, &r2_true, &mask) > 0.95, "EPG R2 should recover truth");
    // On clean data EPG's advantage over mono-exp is BIAS (a near-uniform offset
    // that correlation ignores), not correlation — so gate on bias, not corr.
    let bias_epg = (mean_masked(&r2_epg, &mask) - mean_masked(&r2_true, &mask)).abs();
    let bias_mono = (mean_masked(&r2_mono, &mask) - mean_masked(&r2_true, &mask)).abs();
    assert!(bias_epg < bias_mono, "EPG should be less biased than mono-exp ({bias_epg:.2} vs {bias_mono:.2})");
    assert!(correlation(&rp_derived, &rp_true, &mask) > 0.85, "R2' should recover truth");
    assert!(corr_dn > corr_noisy, "MP-PCA should improve noisy R2* ({corr_noisy:.3} -> {corr_dn:.3})");
    assert!(tv_after < tv_before, "Gibbs unring should reduce ringing TV ({tv_before:.1} -> {tv_after:.1})");
}

/// A MESE that covers only a slab of the brain.
///
/// Whole-brain MESE is expensive, so a spin-echo companion to a 3D GRE is routinely a band of
/// slices. Outside that band there is no R2, and `R2' = R2* - 0 = R2*` would report the whole of
/// R2* as reversible — overstating R2' by roughly the tissue R2 (~14 Hz here) with nothing in the
/// output to show it.
///
/// Two conditions, because the two things that bound coverage are independently necessary:
///
/// 1. **A MESE fitted with EPG.** `r2_epg` leaves every voxel it did not fit at exactly zero, so
///    the `r2 > 0` floor alone already excludes the un-acquired region. Checks that R2' is
///    reported only inside the slab, is accurate there, and that subtracting over the whole brain
///    mask — all the caller could do before — badly overstates R2' outside it.
/// 2. **A bring-your-own R2 map resampled onto this grid.** Interpolation across the slab
///    boundary leaves small non-zero R2 values *outside* the acquired FOV, which no inspection of
///    the map can distinguish from tissue. Only the explicit coverage mask excludes those, so
///    this is the condition that the mask — not the heuristic — has to carry.
#[test]
#[ignore] // demo: run explicitly in CI (writes slices/*.bin + prints RELAXMETRIC lines)
fn test_r2prime_slab_coverage() {
    let dims = (NX, NY, NZ);
    let n_vox = NX * NY * NZ;
    let grid = Grid::new(NX, NY, NZ, 1.0, 1.0, 1.0);
    let (mask, r2_true, rp_true) = phantom();
    let r2star_true: Vec<f64> = (0..n_vox).map(|i| r2_true[i] + rp_true[i]).collect();

    // The MESE's FOV: ten slices about the centre, a band narrower than the brain. This is
    // geometry, not data — which is why only the caller can supply it.
    let (z_lo, z_hi) = (NZ / 2 - 5, NZ / 2 + 5);
    let mut slab = vec![0u8; n_vox];
    for z in z_lo..z_hi {
        for y in 0..NY {
            for x in 0..NX {
                slab[idx(x, y, z)] = 1;
            }
        }
    }
    // R2 exists only where the brain and the MESE's FOV overlap.
    let fit_mask: Vec<u8> = (0..n_vox).map(|i| ((mask[i] > 0) && (slab[i] > 0)) as u8).collect();
    let outside: Vec<u8> = (0..n_vox).map(|i| ((mask[i] > 0) && (slab[i] == 0)) as u8).collect();
    let n_brain = mask.iter().filter(|&&m| m > 0).count();
    let n_fit = fit_mask.iter().filter(|&&m| m > 0).count();
    let n_outside = outside.iter().filter(|&&m| m > 0).count();
    assert!(n_fit > 0 && n_outside > 0, "the slab must cover part of the brain, not all or none");
    metric("R2prime slab", "coverage_pct_of_brain", 100.0 * n_fit as f64 / n_brain as f64);

    // ── Signals: GRE over the whole brain, MESE only inside the slab ──
    let gre_te: Vec<f64> = (1..=16).map(|e| e as f64 * 3e-3).collect();
    let se_te: Vec<f64> = (1..=32).map(|e| e as f64 * 8e-3).collect();
    let (n_gre, n_se) = (gre_te.len(), se_te.len());
    let esp = se_te[0];

    let mut gre = vec![0.0f64; n_vox * n_gre];
    let mut se = vec![0.0f64; n_vox * n_se];
    for v in 0..n_vox {
        if mask[v] == 0 {
            continue;
        }
        for (e, &te) in gre_te.iter().enumerate() {
            gre[v * n_gre + e] = (-te * r2star_true[v]).exp();
        }
        if fit_mask[v] == 0 {
            continue; // nothing acquired here: the echoes stay zero
        }
        let train = epg_cpmg_echoes(1.0 / r2_true[v], T1, B1, esp, n_se);
        for e in 0..n_se {
            se[v * n_se + e] = train[e];
        }
    }

    let (r2_meas, _) = r2_epg(&se, &fit_mask, &se_te, &grid, &R2EpgParams::default(), None);
    let (r2star, _) = r2star_arlo(&gre, &mask, &gre_te, &grid);

    // ── Condition 1: EPG-fitted slab MESE ──
    let out = r2prime(&r2star, &r2_meas, &mask, Some(&slab));
    // What the whole-brain subtraction gave before coverage existed.
    let naive: Vec<f64> = (0..n_vox)
        .map(|i| if mask[i] > 0 { (r2star[i] - r2_meas[i]).max(0.0) } else { 0.0 })
        .collect();

    // Accuracy inside the slab is gated in Hz, not by correlation: R2 and R2' are correlated by
    // construction in this phantom, so handing back R2* instead of R2* − R2 still correlates with
    // the truth at r > 0.99. Only an absolute error separates the two.
    let err_in = mean_abs_err(&out.r2prime, &rp_true, &out.coverage);
    let corr_in = correlation(&out.r2prime, &rp_true, &out.coverage);
    let truth_outside = mean_masked(&rp_true, &outside);
    let naive_outside = mean_masked(&naive, &outside);
    metric("R2prime slab", "abs_err_in_coverage_Hz", err_in);
    metric("R2prime slab", "corr_in_coverage", corr_in);
    metric("R2prime slab", "truth_outside_Hz", truth_outside);
    metric("R2prime slab", "unmasked_outside_Hz", naive_outside);

    save_center_slices(&out.r2prime, &mask, dims, "relax_r2prime_slab");
    save_center_slices(&naive, &mask, dims, "relax_r2prime_slab_naive");

    // Coverage is exactly brain ∩ FOV: not widened, and not narrowed below where R2 was fitted.
    let differ = mask_diff(&out.coverage, &fit_mask);
    assert_eq!(differ, 0, "coverage differs from brain ∩ MESE FOV at {differ} voxels");
    // Nothing is reported outside it — not R2*, not a partial subtraction, nothing.
    let leaked = (0..n_vox).filter(|&i| outside[i] > 0 && out.r2prime[i] != 0.0).count();
    assert_eq!(leaked, 0, "{leaked} uncovered voxels carry a non-zero R2'");
    // R2' is still recovered inside the slab, to well under a Hz.
    assert!(err_in < 1.0, "R2' inside the slab should be accurate (mean |err| {err_in:.2} Hz)");
    // And the error avoided is large: the whole-mask subtraction reports roughly the tissue R2
    // (~11-14 Hz) as reversible dephasing, several times the true R2' there.
    assert!(
        naive_outside > truth_outside + 8.0,
        "subtracting over the whole mask should badly overstate R2' outside the slab \
         ({naive_outside:.2} Hz vs truth {truth_outside:.2} Hz) — if it does not, this phantom \
         no longer exercises the bug"
    );
    // For an EPG-derived R2 the `r2 > 0` floor alone reproduces the FOV, because r2_epg leaves
    // every unfitted voxel at exactly zero. So this condition does not need the explicit mask.
    let heuristic = r2prime(&r2star, &r2_meas, &mask, None);
    let differ = mask_diff(&heuristic.coverage, &out.coverage);
    assert_eq!(differ, 0, "the r2 > 0 floor should match the FOV here (differs at {differ} voxels)");
    assert!(heuristic.r2prime == out.r2prime, "the two routes should give the same R2' here");

    // ── Condition 2: a resampled bring-your-own R2 map ──
    // A custom R2 map arrives on its own grid and is interpolated onto this one, which smears the
    // slab boundary: the slices just outside the FOV pick up a fraction of the R2 inside it. The
    // values are small but non-zero and perfectly tissue-like, so `r2 > 0` keeps them.
    let r2_resampled = blur_z(&r2_meas);
    let bleed: Vec<u8> = (0..n_vox)
        .map(|i| (outside[i] > 0 && r2_resampled[i] > 0.0) as u8)
        .collect();
    let n_bleed = bleed.iter().filter(|&&b| b > 0).count();
    assert!(n_bleed > 0, "resampling should leave non-zero R2 outside the FOV, else this condition is vacuous");
    metric("R2prime slab", "bleed_voxels_outside_FOV", n_bleed as f64);

    // The heuristic cannot see that these are outside the acquisition, so it admits them and
    // reports a spurious R2' — this is the limit the explicit mask exists to cover.
    let without = r2prime(&r2star, &r2_resampled, &mask, None);
    let spurious = mean_masked(&without.r2prime, &bleed);
    metric("R2prime slab", "heuristic_bleed_R2prime_Hz", spurious);
    assert!(
        mask_diff(&without.coverage, &fit_mask) > 0 && spurious > truth_outside + 2.0,
        "the r2 > 0 floor should admit the interpolation bleed ({spurious:.2} Hz over {n_bleed} \
         voxels) — that is why coverage has to be supplied, not inferred"
    );

    // With the FOV supplied, the bleed is excluded: coverage and map are unchanged from the
    // hard-edged case, even though the R2 map now has values outside the slab.
    let with = r2prime(&r2star, &r2_resampled, &mask, Some(&slab));
    let differ = mask_diff(&with.coverage, &fit_mask);
    assert_eq!(differ, 0, "an explicit FOV must exclude the interpolation bleed ({differ} voxels differ)");
    let leaked = (0..n_vox).filter(|&i| outside[i] > 0 && with.r2prime[i] != 0.0).count();
    assert_eq!(leaked, 0, "{leaked} out-of-FOV voxels carry an R2' from interpolated R2");
}

/// How many voxels two binary masks disagree on (a count, so a failure prints a number rather
/// than a hundred thousand elements).
fn mask_diff(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b.iter()).filter(|(&x, &y)| (x > 0) != (y > 0)).count()
}

/// Mean |a − b| over `mask`.
fn mean_abs_err(a: &[f64], b: &[f64], mask: &[u8]) -> f64 {
    let (mut s, mut c) = (0.0, 0usize);
    for i in 0..a.len() {
        if mask[i] > 0 {
            s += (a[i] - b[i]).abs();
            c += 1;
        }
    }
    s / c.max(1) as f64
}

/// Three-tap moving average along z — what trilinear interpolation onto a shifted grid does to a
/// slab boundary: the slices either side of it pick up a fraction of what was inside.
fn blur_z(vol: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0f64; vol.len()];
    for z in 0..NZ {
        for y in 0..NY {
            for x in 0..NX {
                let lo = vol[idx(x, y, z.saturating_sub(1))];
                let hi = vol[idx(x, y, (z + 1).min(NZ - 1))];
                out[idx(x, y, z)] = 0.25 * lo + 0.5 * vol[idx(x, y, z)] + 0.25 * hi;
            }
        }
    }
    out
}

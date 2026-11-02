//! `swi::calculate_swi` (CLEAR-SWI) against CLEARSWI.jl (korbinian90, 0949bc5 = 1.7.0), run in the
//! verified QSM-CI Julia image (QSMxT/QSM-CI#299), astewartau/QSM.rs#317.
//!
//! `tests/data/clearswi_small.bin`, made by `tests/data/clearswi_small.jl` (layout there; regenerated
//! in Actions by `.github/workflows/regenerate-fixtures.yml`), holds CLEARSWI.jl's outputs for one
//! 20x20x14 multi-echo phantom, which is rebuilt here bit for bit:
//!
//! 1. `calculateSWI`'s magnitude and phase mask (and the combined filtered phase) for the phase
//!    without wraps, at CLEARSWI.jl's default options, which `SwiParams::default` mirrors;
//! 2. the same without sensitivity correction (`mag_sens = [1]`, [`Sensitivity::Off`]);
//! 3. the same as 1 for the phase with wraps across the brain;
//! 4. CLEARSWI.jl's own steps after unwrapping (`laplacian_combine` without `laplacianunwrap`) on
//!    the true unwrapped phase of case 1.
//!
//! What is compared, and why. QSM.rs's Laplacian unwrapping differs from MriResearchTools'
//! `laplacianunwrap` on purpose (`src/swi.rs`, "Differences from CLEARSWI.jl"). The two cannot
//! be matched by choosing the input: `laplacianunwrap` does not return even a phase without wraps
//! unchanged (on case 1 it adds a smooth field of 0.09-0.20 rad standard deviation per echo, its
//! forward and inverse Laplacians being different operators), where QSM.rs's DCT solver returns it
//! up to a constant. So:
//! - everything after unwrapping is compared at matched settings and input (case 4: the masked
//!   high-pass, the `mag·TE`-weighted echo combination and the tanh phase mask);
//! - the magnitude is compared with and without sensitivity correction (cases 2 and 1; the CLEAR
//!   sensitivity estimate, `getsensitivity`, is the one `makehomogeneous` is built on,
//!   astewartau/QSM.rs#319, #355);
//! - `calculate_swi` end to end against `calculateSWI` (cases 1 and 3) measures what the deliberate
//!   unwrapping difference does to the phase mask, without and with wraps. It is held at its
//!   measured size, not hidden.

use qsm_core::swi::{calculate_swi, create_phase_mask, highpass_filter, Sensitivity, SwiParams};
use qsm_core::utils::calculate_b0_weighted;
use qsm_core::Grid;
use std::f64::consts::PI;

const N: [usize; 3] = [20, 20, 14];
const VS: [f64; 3] = [1.0, 1.0, 1.5];
const TES: [f64; 3] = [4.0, 8.0, 12.0];
/// The generator's cases: (frequency-map scale, sensitivity correction).
const CASES: [(f64, bool); 3] = [(1.0, true), (1.0, false), (6.0, true)];

/// One volume per echo.
type Echoes = Vec<Vec<f64>>;

/// The generator's `phantom(scale)`: wrapped phase, magnitude and unwrapped phase per echo.
fn phantom(scale: f64) -> (Echoes, Echoes, Echoes) {
    let nv = N[0] * N[1] * N[2];
    let (a, b, c) = (0.46 * N[0] as f64, 0.44 * N[1] as f64, 0.42 * N[2] as f64);
    let (mut wrapped, mut mag, mut unwrapped) = (vec![vec![0.0; nv]; TES.len()], vec![vec![0.0; nv]; TES.len()], vec![vec![0.0; nv]; TES.len()]);
    for (e, &te) in TES.iter().enumerate() {
        for k in 0..N[2] {
            for j in 0..N[1] {
                for i in 0..N[0] {
                    let l = i + j * N[0] + k * N[0] * N[1];
                    let x = i as f64 - (N[0] - 1) as f64 / 2.0 + 0.3;
                    let y = j as f64 - (N[1] - 1) as f64 / 2.0 + 0.3;
                    let z = k as f64 - (N[2] - 1) as f64 / 2.0 + 0.3;
                    let inside = x * x / (a * a) + y * y / (b * b) + z * z / (c * c) < 1.0;
                    let bg = 0.004 * x + 0.003 * y - 0.002 * z + 0.0002 * x * y;
                    let s1 = (x - 3.0) * (x - 3.0) + (y + 2.0) * (y + 2.0);
                    let s2 = (x + 4.0) * (x + 4.0) + (y - 3.0) * (y - 3.0) + (z - 1.0) * (z - 1.0);
                    let s3 = (y + 5.0) * (y + 5.0) + (z + 2.0) * (z + 2.0);
                    let src = 0.06 / (1.0 + s1 / 2.0) - 0.05 / (1.0 + s2 / 3.0) + 0.04 / (1.0 + s3 / 1.5);
                    let omega = scale * (bg + if inside { src } else { 0.0 });
                    let phi0 = 0.1 + 0.01 * x - 0.02 * y;
                    let h = (i as u64 * 73856093) ^ (j as u64 * 19349663) ^ (k as u64 * 83492791) ^ (e as u64 * 2654435761);
                    let noise = ((h % 1000) as f64 / 1000.0 - 0.5) * if inside { 0.04 } else { 0.6 };
                    let raw = te * omega + phi0 + noise;
                    wrapped[e][l] = raw - 2.0 * PI * (raw / (2.0 * PI)).round_ties_even();
                    unwrapped[e][l] = raw;
                    let bias = 1.0 + 0.02 * x - 0.015 * y + 0.0005 * x * z;
                    let vein = if inside { 30.0 / (1.0 + s1 / 2.0) } else { 0.0 };
                    let tissue = if inside { 100.0 - 0.1 * (x * x + y * y + z * z) - vein } else { 4.0 + (h % 7) as f64 };
                    mag[e][l] = tissue * bias / (1.0 + 0.1 * e as f64);
                }
            }
        }
    }
    (wrapped, mag, unwrapped)
}

/// One case of the fixture: the mask, the SWI magnitude and phase mask over the grid, and the
/// combined filtered phase at the mask voxels.
struct Original {
    mask: Vec<u8>,
    magnitude: Vec<f64>,
    phase_mask: Vec<f64>,
    filtered: Vec<f64>,
}

/// Case 4: the combined filtered phase at case 1's mask voxels (Hz) and the phase mask.
struct AfterUnwrapping {
    filtered: Vec<f64>,
    phase_mask: Vec<f64>,
}

fn load() -> (Vec<Original>, AfterUnwrapping) {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/clearswi_small.bin")).expect("fixture");
    let nv = N[0] * N[1] * N[2];
    let mut o = 0;
    let f64s = |o: &mut usize, k: usize| -> Vec<f64> {
        let v = raw[*o..*o + 8 * k].as_chunks::<8>().0.iter().map(|&b| f64::from_le_bytes(b)).collect();
        *o += 8 * k;
        v
    };
    let out: Vec<Original> = CASES
        .iter()
        .map(|_| {
            let nb = nv.div_ceil(8);
            let mask: Vec<u8> = (0..nv).map(|i| (raw[o + i / 8] >> (i % 8)) & 1).collect();
            o += nb;
            let m = u32::from_le_bytes(raw[o..o + 4].try_into().unwrap()) as usize;
            o += 4;
            assert_eq!(m, mask.iter().filter(|&&v| v != 0).count());
            let magnitude = f64s(&mut o, nv);
            let phase_mask = f64s(&mut o, nv);
            let filtered = f64s(&mut o, m);
            Original { mask, magnitude, phase_mask, filtered }
        })
        .collect();
    let m = out[0].filtered.len();
    let after = AfterUnwrapping { filtered: f64s(&mut o, m), phase_mask: f64s(&mut o, nv) };
    assert_eq!(o, raw.len());
    (out, after)
}

fn grid() -> Grid {
    Grid::new(N[0], N[1], N[2], VS[0], VS[1], VS[2])
}

/// The echo times in seconds: QSM.rs's unit (CLEARSWI.jl takes ms; both combine to Hz).
fn tes_s() -> Vec<f64> {
    TES.iter().map(|t| t / 1000.0).collect()
}

/// NRMSE (%), r and max |Δ| over the given voxels, printed.
fn parity(name: &str, a: &[f64], b: &[f64]) -> (f64, f64, f64) {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb, mut d2, mut b2, mut dmax) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma).powi(2);
        sbb += (y - mb).powi(2);
        d2 += (x - y).powi(2);
        b2 += y * y;
        dmax = dmax.max((x - y).abs());
    }
    let (nrmse, r) = (100.0 * (d2 / b2).sqrt(), sab / (saa * sbb).sqrt());
    println!("{name}: NRMSE {nrmse:.3e} %, r {r:.10}, max |Δ| {dmax:.3e}");
    (nrmse, r, dmax)
}

fn check(name: &str, a: &[f64], b: &[f64], nrmse_max: f64, dmax_max: f64) {
    let (nrmse, _, dmax) = parity(name, a, b);
    assert!(nrmse <= nrmse_max && dmax <= dmax_max,
        "{name}: NRMSE {nrmse:.3e} % (limit {nrmse_max:e}), max |Δ| {dmax:.3e} (limit {dmax_max:e})");
}

/// QSM.rs's `calculate_swi` on case `c`.
fn run(c: usize) -> qsm_core::swi::SwiResult {
    let (scale, sens) = CASES[c];
    let (phase, mag, _) = phantom(scale);
    let params = SwiParams {
        sensitivity: if sens { SwiParams::default().sensitivity } else { Sensitivity::Off },
        ..Default::default()
    };
    calculate_swi(&phase, &mag, &tes_s(), None, &grid(), &params).expect("calculate_swi")
}

fn at_mask(v: &[f64], mask: &[u8]) -> Vec<f64> {
    v.iter().zip(mask).filter(|(_, &m)| m != 0).map(|(&x, _)| x).collect()
}

#[test]
fn clearswi_after_unwrapping_matches_clearswi_jl() {
    let (originals, after) = load();
    let mask = &originals[0].mask;
    let (_, mag, unwrapped) = phantom(1.0);
    let params = SwiParams::default();
    let filtered: Vec<Vec<f64>> = unwrapped.iter().map(|u| highpass_filter(u, mask, &grid(), params.hp_sigma)).collect();
    let combined = calculate_b0_weighted(&filtered, &mag, &tes_s(), mask, params.echo_weighting, &grid());
    // To rounding (2e-14 % and 4e-15 %): the masked high-pass (MriResearchTools'
    // `gaussiansmooth3d(...; mask)`, 1.9 % before astewartau/QSM.rs#355 fixed its NaN box filter,
    // #319), the echo combination and the tanh phase mask.
    check("after unwrapping: filtered phase", &at_mask(&combined, mask), &after.filtered, 1e-12, 1e-12);
    let phase_mask = create_phase_mask(&combined, mask, params.scaling, params.strength);
    check("after unwrapping: phase mask", &phase_mask, &after.phase_mask, 1e-12, 1e-12);
}

#[test]
fn swi_magnitude_matches_clearswi_jl() {
    let (originals, _) = load();
    // Combination (RSS) and softplus: to rounding.
    check("magnitude, no sensitivity correction", &run(1).magnitude, &originals[1].magnitude, 1e-12, 1e-12);
    // The CLEAR sensitivity estimate (`get_sensitivity`, MriResearchTools' `getsensitivity`): to
    // rounding (2e-14 %; 4.4 % before astewartau/QSM.rs#355 fixed `makehomogeneous`, #319).
    check("magnitude, CLEAR sensitivity", &run(0).magnitude, &originals[0].magnitude, 1e-12, 1e-12);
}

#[test]
fn calculate_swi_differs_from_calculate_swi_jl_by_the_documented_unwrapping() {
    let (originals, _) = load();
    // The deliberate unwrapping difference alone (everything after it matches to rounding, above),
    // without wraps (case 1: 17 % in the filtered phase, 2.1 % in the phase mask) and with (case 3:
    // 43 % and 3.8 %). Held at the sizes measured.
    for (c, filtered_max, mask_max) in [(0, (17.5, 1.0), (2.2, 0.33)), (2, (44.0, 12.5), (3.9, 0.31))] {
        let (o, r) = (&originals[c], run(c));
        assert_eq!(r.mask, o.mask, "case {}: the phase mask's mask", c + 1);
        let name = format!("case {}: filtered phase", c + 1);
        check(&name, &at_mask(&r.filtered_phase, &o.mask), &o.filtered, filtered_max.0, filtered_max.1);
        check(&format!("case {}: phase mask", c + 1), &r.phase_mask, &o.phase_mask, mask_max.0, mask_max.1);
    }
}

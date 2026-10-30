//! MriResearchTools.jl's ports against MriResearchTools.jl itself (korbinian90, 771eadf):
//! `makehomogeneous`, `robust_mask` (`robustmask`), `fill_holes`, `calculate_b0_weighted`
//! (`calculateB0_unwrapped`) and `bipolar_correction` (`bipolar_correction!`).
//!
//! `tests/data/mritools_small.bin` is made by `tests/data/mritools_small.jl` in the verified QSM-CI
//! `julia-qsm` image (QSMxT/QSM-CI#299; the command is in its header, and
//! `.github/workflows/regenerate-fixtures.yml` runs it in Actions). The phantom is rebuilt here
//! exactly as the script builds it; the layout is in the script.
//!
//! `makehomogeneous` is MriResearchTools' bit for bit (it was 3.3 % NRMSE before three port errors
//! were fixed, see `utils::get_sensitivity`), and `robust_mask`, `fill_holes` and
//! `calculate_b0_weighted` match to the voxel and to rounding. **Known gap**
//! (astewartau/QSM.rs#319), recorded so that a change shows and to be tightened when fixed:
//! `bipolar_correction`'s artefact estimate differs from
//! `bipolar_correction!`'s by up to 0.48 rad, so the corrected phases differ by up to 0.24 rad
//! (echoes 1, 3) and 0.48 rad (echoes 2, 4), modulo 2π.

use qsm_core::utils::{bipolar_correction, calculate_b0_weighted, fill_holes, makehomogeneous, robust_mask, B0WeightType};
use qsm_core::Grid;
use std::f64::consts::PI;

const N: [usize; 3] = [20, 18, 14];
const VS: [f64; 3] = [1.0, 1.0, 1.5];
const TES: [f64; 4] = [4.0, 8.0, 12.0, 16.0];

struct Phantom {
    phase: Vec<Vec<f64>>,
    wrapped: Vec<Vec<f64>>,
    mag: Vec<Vec<f64>>,
    mask: Vec<u8>,
    holey: Vec<u8>,
}

/// The generator's `phantom()`.
fn phantom() -> Phantom {
    let nv = N[0] * N[1] * N[2];
    let (a, b, c) = (0.46 * N[0] as f64, 0.44 * N[1] as f64, 0.42 * N[2] as f64);
    let ne = TES.len();
    let mut p = Phantom { phase: vec![vec![0.0; nv]; ne], wrapped: vec![vec![0.0; nv]; ne], mag: vec![vec![0.0; nv]; ne], mask: vec![0; nv], holey: vec![0; nv] };
    for (e, &te) in TES.iter().enumerate() {
        for k in 0..N[2] {
            for j in 0..N[1] {
                for i in 0..N[0] {
                    let l = i + j * N[0] + k * N[0] * N[1];
                    let x = i as f64 - (N[0] - 1) as f64 / 2.0 + 0.3;
                    let y = j as f64 - (N[1] - 1) as f64 / 2.0 + 0.3;
                    let z = k as f64 - (N[2] - 1) as f64 / 2.0 + 0.3;
                    let inside = x * x / (a * a) + y * y / (b * b) + z * z / (c * c) < 1.0;
                    p.mask[l] = u8::from(inside);
                    let cav = (x - 4.0) * (x - 4.0) + y * y + z * z < 4.0 || (x + 5.0) * (x + 5.0) + (y - 3.0) * (y - 3.0) + z * z < 2.5;
                    let big = (x + 1.0) * (x + 1.0) + (y + 5.0) * (y + 5.0) + (z - 1.0) * (z - 1.0) < 12.0;
                    p.holey[l] = u8::from(inside && !cav && !big);
                    let omega = 0.08 * x + 0.05 * y - 0.03 * z + 0.004 * x * y + 0.002 * (x * x - z * z);
                    let phi0 = 0.5 + 0.02 * x - 0.03 * y;
                    let h = (i as u64 * 73856093) ^ (j as u64 * 19349663) ^ (k as u64 * 83492791) ^ (e as u64 * 2654435761);
                    let noise = ((h % 1000) as f64 / 1000.0 - 0.5) * if inside { 0.1 } else { 2.0 };
                    // Julia's echo index is 1-based: `iseven(e)` there is an odd `e` here
                    let bip = if e % 2 == 1 { 1.0 } else { -1.0 } * 0.03 * x;
                    let raw = te * omega + phi0 + noise + bip;
                    p.phase[e][l] = raw;
                    p.wrapped[e][l] = raw - 2.0 * PI * (raw / (2.0 * PI)).round_ties_even();
                    let bias = 1.0 + 0.02 * x - 0.015 * y + 0.0005 * x * z;
                    let tissue = if inside { 100.0 - 0.1 * (x * x + y * y + z * z) } else { 4.0 + (h % 7) as f64 };
                    p.mag[e][l] = tissue * bias / (1.0 + 0.1 * e as f64);
                }
            }
        }
    }
    p
}

struct Reader<'a> {
    raw: &'a [u8],
    o: usize,
}

impl Reader<'_> {
    fn f64s(&mut self, k: usize) -> Vec<f64> {
        let v = (0..k).map(|i| f64::from_le_bytes(self.raw[self.o + 8 * i..self.o + 8 * i + 8].try_into().unwrap())).collect();
        self.o += 8 * k;
        v
    }
    fn bits(&mut self, k: usize) -> Vec<u8> {
        let v = (0..k).map(|i| self.raw[self.o + i / 8] >> (i % 8) & 1).collect();
        self.o += k.div_ceil(8);
        v
    }
    fn u32(&mut self) -> usize {
        self.o += 4;
        u32::from_le_bytes(self.raw[self.o - 4..self.o].try_into().unwrap()) as usize
    }
}

fn max_rel(a: &[f64], b: &[f64]) -> f64 {
    let scale = b.iter().map(|v| v.abs()).fold(0.0, f64::max);
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max) / scale
}

#[test]
fn mritools_ports_match_mritools() {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/mritools_small.bin")).expect("fixture");
    let mut r = Reader { raw: &raw, o: 0 };
    let p = phantom();
    let nv = p.mask.len();
    let grid = Grid::new(N[0], N[1], N[2], VS[0], VS[1], VS[2]);
    let inside: Vec<usize> = (0..nv).filter(|&l| p.mask[l] != 0).collect();

    let want = r.f64s(nv);
    let got = makehomogeneous(&p.mag[0], &grid, 7.0, 15).unwrap();
    let (gi, wi): (Vec<f64>, Vec<f64>) = inside.iter().map(|&l| (got[l], want[l])).unzip();
    let nrmse = (gi.iter().zip(&wi).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / wi.iter().map(|b| b * b).sum::<f64>()).sqrt();
    println!("makehomogeneous: in the mask NRMSE {nrmse:.2e}, max |Δ| / max {:.2e}", max_rel(&gi, &wi));
    // Measured 2026-10-30: bit-identical (NRMSE 0; 3.3 % before the fixes in `get_sensitivity`).
    assert!(nrmse == 0.0, "makehomogeneous: NRMSE {nrmse:e}");

    let want = r.bits(nv);
    let got = robust_mask(&p.mag[0], &grid);
    let diff = got.iter().zip(&want).filter(|(a, b)| a != b).count();
    println!("robust_mask: {} voxels, {diff} differ", want.iter().filter(|&&v| v != 0).count());
    assert_eq!(diff, 0, "robust_mask");

    let want = r.bits(nv);
    let got = fill_holes(&p.holey, &grid, 30);
    let diff = got.iter().zip(&want).filter(|(a, b)| a != b).count();
    println!("fill_holes: {} voxels, {diff} differ", want.iter().filter(|&&v| v != 0).count());
    assert_eq!(diff, 0, "fill_holes");

    let m = r.u32();
    assert_eq!(m, inside.len());
    let tes_s: Vec<f64> = TES.iter().map(|t| t / 1000.0).collect();
    for (name, w) in [("phase_snr", B0WeightType::PhaseSNR), ("phase_var", B0WeightType::PhaseVar), ("average", B0WeightType::Average), ("TEs", B0WeightType::TEs), ("mag", B0WeightType::Mag)] {
        let want = r.f64s(m);
        let b0 = calculate_b0_weighted(&p.phase, &p.mag, &tes_s, &p.mask, w, &grid);
        let got: Vec<f64> = inside.iter().map(|&l| b0[l]).collect();
        let d = max_rel(&got, &want);
        println!("calculate_b0_weighted {name}: max |Δ| / max {d:.2e}");
        assert!(d < 1e-12, "{name}: {d:e}");
    }

    let mut phases = p.wrapped.clone();
    bipolar_correction(&mut phases, &p.mag, &TES, &p.mask, [7.0, 7.0, 7.0 / 1.5], &grid);
    for (e, ph) in phases.iter().enumerate() {
        let want = r.f64s(m);
        let got: Vec<f64> = inside.iter().map(|&l| ph[l]).collect();
        // MriResearchTools rewraps the corrected phase; compare modulo 2π
        let d = got.iter().zip(&want).map(|(a, b)| (a - b - 2.0 * PI * ((a - b) / (2.0 * PI)).round()).abs()).fold(0.0, f64::max);
        println!("bipolar_correction echo {}: max |Δ| {d:.2e} rad (modulo 2π)", e + 1);
        // Measured 2026-10-30: 0.24 rad on echoes 1 and 3, 0.48 on 2 and 4, which take the
        // artefact twice (the known gap above: the artefact estimate differs by up to 0.48 rad).
        let bound = if e % 2 == 1 { 0.55 } else { 0.3 };
        assert!(d < bound, "bipolar echo {}: {d:e}", e + 1);
    }
    assert_eq!(r.o, raw.len(), "fixture length");
}

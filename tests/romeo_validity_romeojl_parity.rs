//! MCPC-3D-S where its phase offset is undefined, and ROMEO's multi-echo modes, against the
//! authors' code: MriResearchTools.jl's `mcpc3ds` (v3.8.0), ROMEO.jl's `unwrap` (v1.6.0) and
//! `calculateB0_unwrapped`, on a small three-echo phantom whose brain mask reaches well past
//! MCPC-3D-S's own `robustmask` into a region with no signal and a steep coil-combination offset.
//!
//! There the reference's smoothed offset, and so its corrected phase, is NaN; ROMEO.jl never grows
//! through those voxels and their B0 is 0. QSM.rs must do the same
//! ([`PhaseOffsetRemoval::valid`], [`FieldMappingResult::valid_mask`]); until 2026-11 it set the
//! offset to 0 there and unwrapped the raw phase, which this test also shows to change the turns
//! inside the head (the ring artefacts of the Q0748 audit on GE and Philips data).
//!
//! The inputs are rebuilt here exactly as `tests/data/romeo_validity_romeojl_small.jl` built them
//! (only sums, products, quotients, round and integer xor, so bit for bit); the fixture holds only
//! the references' output. The script ran in the QSM-CI `julia-qsm` image,
//! `sha256:b3c048acd8a88f23b7a04378f2b85dbac433bcc78b3d9cdf8ce58eb8a7478185`, with one Julia
//! thread; the command and the layout of `tests/data/romeo_validity_romeojl_small.bin` are in the
//! script's header.

use qsm_core::pipeline::{
    hz_to_ppm, run_field_mapping, B0EstimationMethod, FieldMappingConfig, FieldMappingResult, ScanMetadata,
};
use qsm_core::unwrap::{unwrap_romeo_multi_echo, RomeoParams, UnwrapMethod};
use qsm_core::utils::{calculate_b0_weighted, phase_offset_removal, B0WeightType, PhaseOffsetRemoval};
use qsm_core::Grid;
use std::f64::consts::PI;

const N: [usize; 3] = [30, 26, 16];
const TES: [f64; 3] = [4.0, 8.0, 12.0]; // ms
const SIGMA: [f64; 3] = [2.0; 3];

/// The generator's `phantom()`: wrapped phase and magnitude per echo, and the brain mask.
fn phantom() -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<u8>) {
    let n = N;
    let nv = n[0] * n[1] * n[2];
    let (a, b, c) = (0.30 * n[0] as f64, 0.40 * n[1] as f64, 0.40 * n[2] as f64);
    let mut phase = vec![vec![0.0; nv]; TES.len()];
    let mut mag = vec![vec![0.0; nv]; TES.len()];
    let mut brain = vec![0u8; nv];
    for (e, &te) in TES.iter().enumerate() {
        for k in 0..n[2] {
            for j in 0..n[1] {
                for i in 0..n[0] {
                    let l = i + j * n[0] + k * n[0] * n[1];
                    let x = i as f64 - 0.40 * (n[0] - 1) as f64 + 0.3;
                    let y = j as f64 - (n[1] - 1) as f64 / 2.0 + 0.3;
                    let z = k as f64 - (n[2] - 1) as f64 / 2.0 + 0.3;
                    let head = x * x / (a * a) + y * y / (b * b) + z * z / (c * c) < 1.0;
                    let slab = x > 0.0 && x < 1.75 * a && y * y / (b * b) + z * z / (c * c) < 0.5;
                    brain[l] = u8::from(head || slab);
                    let omega = 0.08 * x + 0.05 * y - 0.03 * z + 0.004 * x * y + 0.002 * (x * x - z * z);
                    let phi0 = if head { 0.5 + 0.02 * x - 0.03 * y } else { 0.5 + 0.9 * (x - a) * (y + 0.5) - 0.4 * z };
                    let h = (i as u64 * 73856093) ^ (j as u64 * 19349663) ^ (k as u64 * 83492791) ^ (e as u64 * 2654435761);
                    let noise = ((h % 1000) as f64 / 1000.0 - 0.5) * if head { 0.4 } else { 1.5 };
                    let raw = te * omega + phi0 + noise;
                    phase[e][l] = raw - 2.0 * PI * (raw / (2.0 * PI)).round_ties_even();
                    mag[e][l] = if head {
                        (100.0 - 0.1 * (x * x + y * y + z * z)) / (1.0 + 0.1 * e as f64)
                    } else {
                        0.5 + (h % 7) as f64 / 10.0
                    };
                }
            }
        }
    }
    (phase, mag, brain)
}

fn cases() -> [(&'static str, bool, RomeoParams); 6] {
    let individual = RomeoParams { individual: true, correct_global: true, ..RomeoParams::default() };
    let template = RomeoParams { temporal_uncertain_unwrapping: 0.0, ..RomeoParams::default() };
    [
        ("MCPC-3D-S, individual, global correction", true, individual.clone()),
        ("MCPC-3D-S, individual", true, RomeoParams { correct_global: false, ..individual.clone() }),
        ("MCPC-3D-S, template", true, template.clone()),
        ("MCPC-3D-S, template, global correction", true, RomeoParams { correct_global: true, ..template.clone() }),
        ("raw phase, individual, global correction", false, individual),
        ("raw phase, template", false, template),
    ]
}

struct Reader {
    raw: Vec<u8>,
    o: usize,
}

impl Reader {
    fn bytes(&mut self, n: usize) -> &[u8] {
        let s = &self.raw[self.o..self.o + n];
        self.o += n;
        s
    }
    fn f64s(&mut self, n: usize) -> Vec<f64> {
        self.bytes(8 * n).as_chunks::<8>().0.iter().map(|c| f64::from_le_bytes(*c)).collect()
    }
}

fn turns(uw: &[f64], input: &[f64], at: &[usize]) -> Vec<i64> {
    at.iter().map(|&l| ((uw[l] - input[l]) / (2.0 * PI)).round() as i64).collect()
}

#[test]
fn mcpc3ds_validity_and_romeo_modes_match_the_julia_packages() {
    let (phase, mag, brain) = phantom();
    let grid = Grid::new(N[0], N[1], N[2], 1.0, 1.0, 1.0);
    let brain_vox: Vec<usize> = (0..brain.len()).filter(|&l| brain[l] != 0).collect();
    let mut r = Reader {
        raw: std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/romeo_validity_romeojl_small.bin")).expect("fixture"),
        o: 0,
    };
    let mut failed = Vec::new();

    // MCPC-3D-S: where the offset is defined, and the corrected phase there.
    let por: PhaseOffsetRemoval = phase_offset_removal(&phase, &mag, &TES, SIGMA, [0, 1], UnwrapMethod::Romeo, &grid);
    let finite_jl: Vec<u8> = r.bytes(brain_vox.len()).to_vec();
    let valid_rs: Vec<u8> = brain_vox.iter().map(|&l| por.valid[l]).collect();
    let n_invalid = finite_jl.iter().filter(|&&f| f == 0).count();
    println!("mcpc3ds: {n_invalid} of {} brain voxels without an offset", brain_vox.len());
    assert!(n_invalid > 100, "the phantom must put part of the brain mask outside MCPC-3D-S's mask");
    if valid_rs != finite_jl {
        failed.push(format!(
            "valid voxels: {} differ",
            valid_rs.iter().zip(&finite_jl).filter(|(a, b)| a != b).count()
        ));
    }
    let keep: Vec<usize> = brain_vox.iter().zip(&finite_jl).filter(|(_, &f)| f != 0).map(|(&l, _)| l).collect();
    for e in 0..TES.len() {
        let want = r.f64s(keep.len());
        let max_d = keep.iter().zip(&want).map(|(&l, w)| {
            let d = por.phases[e][l] - w;
            (d - 2.0 * PI * (d / (2.0 * PI)).round()).abs()
        }).fold(0.0, f64::max);
        println!("mcpc3ds, echo {}: max |Δ| {max_d:.2e} rad (wrapped)", e + 1);
        if max_d.is_nan() || max_d >= 1e-9 {
            failed.push(format!("mcpc3ds echo {}: max |Δ| {max_d:e}", e + 1));
        }
        let nan_outside = brain_vox.iter().zip(&finite_jl).filter(|(&l, &f)| f == 0 && !por.phases[e][l].is_nan()).count();
        if nan_outside != 0 {
            failed.push(format!("mcpc3ds echo {}: {nan_outside} invalid voxels not NaN", e + 1));
        }
    }

    // ROMEO's multi-echo modes on the corrected (NaN) and on the raw phase, and the B0.
    for (name, offset_removal, params) in cases() {
        let input = if offset_removal { &por.phases } else { &phase };
        let at = if offset_removal { &keep } else { &brain_vox };
        let uw = unwrap_romeo_multi_echo(input, &mag, &TES, &brain, &params, &grid);
        for e in 0..TES.len() {
            let want: Vec<i64> = r.bytes(at.len()).iter().map(|&t| t as i8 as i64).collect();
            let got = turns(&uw[e], &input[e], at);
            let differ = got.iter().zip(&want).filter(|(a, b)| a != b).count();
            println!("{name}, echo {}: {differ} of {} voxels with a different number of turns", e + 1, at.len());
            if differ != 0 {
                failed.push(format!("{name}, echo {}: {differ} voxels", e + 1));
            }
            if offset_removal {
                let reached = brain_vox.iter().filter(|&&l| por.valid[l] == 0 && !uw[e][l].is_nan()).count();
                if reached != 0 {
                    failed.push(format!("{name}, echo {}: ROMEO reached {reached} NaN voxels", e + 1));
                }
            }
        }
        if offset_removal {
            let want_hz = r.f64s(at.len());
            let b0_khz = calculate_b0_weighted(&uw, &mag, &TES, &brain, B0WeightType::PhaseSNR, &grid);
            let max_rel = at.iter().zip(&want_hz)
                .map(|(&l, w)| (b0_khz[l] * 1000.0 - w).abs() / w.abs().max(1.0))
                .fold(0.0, f64::max);
            println!("{name}: B0 max relative |Δ| {max_rel:.2e}");
            if max_rel.is_nan() || max_rel > 1e-12 {
                failed.push(format!("{name}: B0 max relative |Δ| {max_rel:e}"));
            }
        }
    }
    assert_eq!(r.o, r.raw.len());
    assert!(failed.is_empty(), "{failed:#?}");
}

/// The default pipeline (MCPC-3D-S, individual ROMEO with global correction, `:phase_snr`) end to
/// end through [`run_field_mapping`]: the field equals the reference's B0 in `brain ∩ valid`, is 0
/// on the rest of the brain mask, and `valid_mask` says which is which.
#[test]
fn run_field_mapping_returns_the_reference_field_and_its_valid_mask() {
    let (phase, mag, brain) = phantom();
    let tes_s: Vec<f64> = TES.iter().map(|t| t / 1000.0).collect();
    let meta = ScanMetadata {
        dims: (N[0], N[1], N[2]),
        voxel_size: (1.0, 1.0, 1.0),
        echo_times: tes_s,
        field_strength: 3.0,
        b0_direction: (0.0, 0.0, 1.0),
        slice_geometry: None,
        affine: None,
    };
    let cfg = FieldMappingConfig {
        phase_offset_sigma_mm: SIGMA,
        b0_estimation: B0EstimationMethod::WeightedAvg { weighting: B0WeightType::PhaseSNR },
        romeo_params: RomeoParams { individual: true, correct_global: true, ..RomeoParams::default() },
        ..FieldMappingConfig::default()
    };
    let p: Vec<&[f64]> = phase.iter().map(|v| v.as_slice()).collect();
    let m: Vec<&[f64]> = mag.iter().map(|v| v.as_slice()).collect();
    let res: FieldMappingResult = run_field_mapping(&p, Some(&m), &brain, &meta, &cfg, &mut |_, _| {}).unwrap();

    // the fixture's first case
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/romeo_validity_romeojl_small.bin")).unwrap();
    let brain_vox: Vec<usize> = (0..brain.len()).filter(|&l| brain[l] != 0).collect();
    let finite = &raw[..brain_vox.len()];
    let keep: Vec<usize> = brain_vox.iter().zip(finite).filter(|(_, &f)| f != 0).map(|(&l, _)| l).collect();
    let o = brain_vox.len() + 8 * TES.len() * keep.len() + TES.len() * keep.len();
    let want_hz: Vec<f64> = raw[o..o + 8 * keep.len()].as_chunks::<8>().0.iter().map(|c| f64::from_le_bytes(*c)).collect();
    let want_ppm = hz_to_ppm(&want_hz, 3.0);

    for (&l, &f) in brain_vox.iter().zip(finite) {
        assert_eq!(res.valid_mask[l], f, "valid_mask at {l}");
        if f == 0 {
            assert_eq!(res.b0_field_ppm[l], 0.0, "field outside the valid mask at {l}");
        }
    }
    assert!(res.valid_mask.iter().zip(&brain).all(|(&v, &b)| v <= b), "valid_mask must lie in the brain mask");
    let max_rel = keep.iter().zip(&want_ppm)
        .map(|(&l, w)| (res.b0_field_ppm[l] - w).abs() / w.abs().max(1e-3))
        .fold(0.0, f64::max);
    println!("run_field_mapping: max relative |Δ| {max_rel:.2e} over {} voxels", keep.len());
    assert!(max_rel < 1e-9, "max relative |Δ| {max_rel:e}");
    assert!(res.b0_field_ppm.iter().all(|v| v.is_finite()));
    assert!(res.phase_offset.as_ref().unwrap().iter().all(|v| v.is_finite()));
}

/// What the old semantics did on the same phantom: the offset set to 0 where it is undefined (so
/// those voxels keep their raw phase) changes the number of turns ROMEO finds *inside the valid
/// region*, which is the mechanism of the GE/Philips ring artefacts. Guards the phantom: if it
/// ever stops exercising the difference, the parity test above proves less.
#[test]
fn raw_phase_where_the_offset_is_undefined_changes_the_turns() {
    let (phase, mag, brain) = phantom();
    let grid = Grid::new(N[0], N[1], N[2], 1.0, 1.0, 1.0);
    let por = phase_offset_removal(&phase, &mag, &TES, SIGMA, [0, 1], UnwrapMethod::Romeo, &grid);
    let old: Vec<Vec<f64>> = por.phases.iter().zip(&phase)
        .map(|(c, p)| c.iter().zip(p).map(|(&c, &p)| if c.is_finite() { c } else { p }).collect())
        .collect();
    let params = RomeoParams { individual: true, correct_global: true, ..RomeoParams::default() };
    let new_uw = unwrap_romeo_multi_echo(&por.phases, &mag, &TES, &brain, &params, &grid);
    let old_uw = unwrap_romeo_multi_echo(&old, &mag, &TES, &brain, &params, &grid);
    let keep: Vec<usize> = (0..brain.len()).filter(|&l| brain[l] != 0 && por.valid[l] != 0).collect();
    let differ: usize = (0..TES.len())
        .map(|e| keep.iter().filter(|&&l| ((new_uw[e][l] - old_uw[e][l]) / (2.0 * PI)).round() != 0.0).count())
        .sum();
    println!("{differ} echo-voxels in brain ∩ valid change their turns with the old semantics");
    assert!(differ > 0);
}

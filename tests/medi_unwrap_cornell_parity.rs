//! `unwrap::unwrap_medi_region_growing` against the MEDI toolbox 2024.11.26's `unwrapPhase` and
//! its `mexUnwrap.c` (Tian Liu), as the verified QSM-CI `fit-ppm-complex-cornell` container runs
//! them (QSMxT/QSM-CI#300), and the field-mapping stage's complex fit with it against that
//! container's whole field map.
//!
//! `tests/data/medi_unwrap_cornell_small.bin` is made by `tests/data/medi_unwrap_cornell_small.py`
//! with that container, so it can be regenerated in Actions
//! (`.github/workflows/regenerate-fixtures.yml`): two small multi-echo phantoms (even and odd
//! sizes, magnitude plateaus so the heap meets ties, background below the noise level and a
//! bright island cut off from the seed), with recon.m's iMag, iFreq_raw, iFreq and total field.

use qsm_core::fieldmap::FitPpmComplexParams;
use qsm_core::pipeline::{run_field_mapping, B0EstimationMethod, FieldMappingConfig, ScanMetadata, UnwrappingAlgorithm};
use qsm_core::unwrap::unwrap_medi_region_growing;
use qsm_core::Grid;

struct Case {
    dims: (usize, usize, usize),
    tes: Vec<f64>,
    b0: f64,
    mags: Vec<Vec<f64>>,
    phases: Vec<Vec<f64>>,
    imag: Vec<f64>,
    ifreq_raw: Vec<f64>,
    /// mexUnwrap's output (single precision)
    ifreq: Vec<f32>,
    /// recon.m's total field (ppm, float64)
    total: Vec<f64>,
}

fn cases() -> Vec<Case> {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/medi_unwrap_cornell_small.bin")).expect("fixture");
    let ncases = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
    let mut out = Vec::new();
    let mut o = 4;
    let f64s = |o: &mut usize, k: usize| -> Vec<f64> {
        let v = raw[*o..*o + 8 * k].chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
        *o += 8 * k;
        v
    };
    for _ in 0..ncases {
        let h: Vec<usize> = raw[o..o + 16].chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap()) as usize).collect();
        o += 16;
        let (dims, ne) = ((h[0], h[1], h[2]), h[3]);
        let n = h[0] * h[1] * h[2];
        let tes = f64s(&mut o, ne);
        let b0 = f64s(&mut o, 1)[0];
        let (mut mags, mut phases) = (Vec::new(), Vec::new());
        for _ in 0..ne {
            mags.push(f64s(&mut o, n));
            phases.push(f64s(&mut o, n));
        }
        let imag = f64s(&mut o, n);
        let ifreq_raw = f64s(&mut o, n);
        let ifreq = raw[o..o + 4 * n].chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        o += 4 * n;
        let total = f64s(&mut o, n);
        out.push(Case { dims, tes, b0, mags, phases, imag, ifreq_raw, ifreq, total });
    }
    assert_eq!(o, raw.len(), "fixture length");
    out
}

/// On MEDI's own inputs (iMag, iFreq_raw), every voxel matches mexUnwrap bit for bit, including
/// the voxels it never reaches; and the phantoms do exercise the growth (many voxels move by a
/// turn or more) and the voxels it leaves alone.
#[test]
fn bit_identical_to_mexunwrap() {
    for c in cases() {
        let (nx, ny, nz) = c.dims;
        let g = Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);
        let got = unwrap_medi_region_growing(&c.ifreq_raw, &c.imag, &g);
        let differ = got.iter().zip(&c.ifreq).filter(|(a, &b)| **a != b as f64).count();
        assert_eq!(differ, 0, "{:?}: {differ} voxels differ from mexUnwrap", c.dims);
        let moved = c.ifreq.iter().zip(&c.ifreq_raw).filter(|(a, b)| (**a as f64 - **b).abs() > 1.0).count();
        let kept = c.ifreq.iter().zip(&c.ifreq_raw).filter(|(a, b)| **a == **b as f32).count();
        assert!(moved > 300 && kept > 300, "{:?}: {moved} unwrapped by a turn, {kept} unchanged", c.dims);
    }
}

/// The field-mapping stage with the complex fit and MEDI's region growing, from the echoes, is
/// MEDI's field map: QSM.rs's `Fit_ppm_complex` port agrees with MATLAB's to rounding, and with
/// that the region growing makes the same choices at every voxel, and the same float32 values.
#[test]
fn complex_fit_with_region_growing_is_medis_field_map() {
    for c in cases() {
        let (nx, ny, nz) = c.dims;
        let n = nx * ny * nz;
        let meta = ScanMetadata {
            dims: c.dims, voxel_size: (1.0, 1.0, 1.0), echo_times: c.tes.clone(),
            field_strength: c.b0, b0_direction: (0.0, 0.0, 1.0), slice_geometry: None, affine: None,
        };
        let config = FieldMappingConfig {
            unwrapping_algorithm: UnwrappingAlgorithm::MediRegionGrowing,
            b0_estimation: B0EstimationMethod::ComplexFit(FitPpmComplexParams::default()),
            ..Default::default()
        };
        let p: Vec<&[f64]> = c.phases.iter().map(|v| v.as_slice()).collect();
        let m: Vec<&[f64]> = c.mags.iter().map(|v| v.as_slice()).collect();
        // A mask that leaves out most of the array: the unwrapping must not use it.
        let mask: Vec<u8> = (0..n).map(|i| (i % 3 == 0) as u8).collect();
        let got = run_field_mapping(&p, Some(&m), &mask, &meta, &config, &mut |_, _| {}).unwrap();
        // Both in ppm, positive for a phase growing with TE, with γ̄ = 42.576 MHz/T.
        let (mut worst, mut num, mut den) = (0.0f64, 0.0, 0.0);
        for i in 0..n {
            let d = got.b0_field_ppm[i] - c.total[i];
            worst = worst.max(d.abs());
            num += d * d;
            den += c.total[i] * c.total[i];
        }
        let nrmse = (num / den).sqrt();
        eprintln!("{:?}: total field max |Δ| {worst:.2e} ppm, NRMSE {nrmse:.2e}", c.dims);
        // Measured: max |Δ| 8.9e-16 and 4.4e-16 ppm (NRMSE 1.2e-16): the double-precision rounding
        // of the units only. A turn in iFreq is 1e6 / (CF * dTE) ppm, hundreds of ppm here.
        assert!(worst < 1e-12, "{:?}: max |Δ| {worst:e} ppm (NRMSE {nrmse:e})", c.dims);
    }
}

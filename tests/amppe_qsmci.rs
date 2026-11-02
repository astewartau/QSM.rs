//! AMP-PE dipole inversion (Huang et al., MRM 2023): against its original, and on the qsm-forward
//! BIDS phantom.
//!
//! AMP-PE (Approximate Message Passing with Parameter Estimation) is a nonlinear dipole inversion
//! with a sparse-wavelet prior and a Gaussian-mixture noise model.
//!
//! - `amp_pe_matches_the_amp_pe_container` (runs by default): against the verified QSM-CI `amp-pe`
//!   container (QSMxT/QSM-CI#309, bit-identical to licensed MATLAB R2023b), on two small phantoms
//!   (`tests/data/amp_pe_qsmci_small.bin`, made by `tests/data/amp_pe_qsmci_small.py`, which holds
//!   the digest and the layout; regenerated in Actions by `.github/workflows/regenerate-fixtures.yml`).
//! - `amp_pe_bids_quality` (ignored; needs the phantom): end-to-end quality against the
//!   ground-truth susceptibility on real-scale data. AMP-PE returns a near-zero-mean χ within the
//!   mask (the constant offset is unresolvable from the local field alone), so the fair metric is
//!   the offset-invariant masked correlation. Run with (data path defaults to the local
//!   qsm-forward derivatives):
//!   `AMPPE_BIDS=/home/ashley/bids/derivatives/qsm-forward/sub-1/anat cargo test --release --test amppe_qsmci -- --ignored --nocapture`

mod common;

use common::{correlation, load_nifti_file, nrmse};
use qsm_core::inversion::{amp_pe, AmpPeParams};
use qsm_core::Grid;
use std::path::Path;
use std::time::Instant;

fn base_dir() -> String {
    std::env::var("AMPPE_BIDS")
        .unwrap_or_else(|_| "/home/ashley/bids/derivatives/qsm-forward/sub-1/anat".to_string())
}

#[test]
#[ignore] // needs the qsm-forward phantom; run with --release --ignored
fn amp_pe_bids_quality() {
    let base = base_dir();
    if !Path::new(&format!("{}/sub-1_fieldmap-local.nii", base)).exists() {
        common::skip_for_missing_data(&format!("phantom not found at {base}"));
        return;
    }

    let lf = load_nifti_file(&format!("{}/sub-1_fieldmap-local.nii", base)).unwrap();
    let mask_nii = load_nifti_file(&format!("{}/sub-1_mask.nii", base)).unwrap();
    let gt = load_nifti_file(&format!("{}/sub-1_Chimap.nii", base)).unwrap();

    let dims = lf.dims;
    let (vx, vy, vz) = lf.voxel_size;
    let grid = Grid::new(dims.0, dims.1, dims.2, vx, vy, vz);
    let mask: Vec<u8> = mask_nii.data.iter().map(|&m| if m > 0.5 { 1 } else { 0 }).collect();

    // Iteration count is overridable to keep the test tractable; the default is a
    // reduced count that already recovers the structure (full quality uses 25+25).
    let max_lin: usize = std::env::var("AMPPE_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    let params = AmpPeParams {
        wave_order: 1,
        nlevel: 3,
        max_linearization_ite: max_lin,
        b0: 3.0,
        gyro_ratio: 42.58,
        simulated_te: 8e-3,
        ..Default::default()
    };

    let t = Instant::now();
    let chi = amp_pe(&lf.data, &mask, None, None, &grid, (0.0, 0.0, 1.0), &params, |_, _| {}).unwrap();
    let elapsed = t.elapsed();

    let corr = correlation(&chi, &gt.data, &mask);
    let nr = nrmse(&chi, &gt.data, &mask);
    println!(
        "AMP-PE ({max_lin}+{max_lin} iters, {:.1}s): masked corr vs GT = {:.4}, nrmse = {:.4}",
        elapsed.as_secs_f64(),
        corr,
        nr
    );

    assert!(chi.iter().all(|v| v.is_finite()), "chi must be finite");
    // Offset-invariant structural agreement with the ground truth. AMP-PE without
    // magnitude weighting reaches ~0.79 masked-corr on this phantom at 8+8 iters;
    // the threshold is a loose smoke-gate (quality, not a benchmark).
    assert!(corr > 0.70, "AMP-PE corr vs GT too low: {corr:.4}");
}

/// A case of `tests/data/amp_pe_qsmci_small.py`.
struct QsmciCase {
    dims: (usize, usize, usize),
    vs: (f64, f64, f64),
    h: (f64, f64, f64),
    magnitude: bool,
    wave_order: usize,
    nlevel: usize,
}

/// `CASES` of `tests/data/amp_pe_qsmci_small.py`, in order (both with 6 linearisations per stage).
const QSMCI_CASES: [QsmciCase; 2] = [
    QsmciCase { dims: (30, 28, 22), vs: (1.0, 1.0, 1.0), h: (0.0, 0.0, 1.0), magnitude: true, wave_order: 1, nlevel: 3 },
    QsmciCase {
        dims: (32, 32, 24),
        vs: (1.0, 1.0, 1.2),
        h: (0.10101525445522108, -0.4040610178208843, 0.9091372900969897),
        magnitude: false,
        wave_order: 2,
        nlevel: 2,
    },
];

#[test]
fn amp_pe_matches_the_amp_pe_container() {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/amp_pe_qsmci_small.bin")).expect("fixture");
    let mut o = 0;
    for c in &QSMCI_CASES {
        let n = c.dims.0 * c.dims.1 * c.dims.2;
        let mask = raw[o..o + n].to_vec();
        o += n;
        let inside: Vec<usize> = (0..n).filter(|&i| mask[i] != 0).collect();
        let mut field = vec![0.0; n];
        for (k, &i) in inside.iter().enumerate() {
            field[i] = f64::from_le_bytes(raw[o + 8 * k..o + 8 * k + 8].try_into().unwrap());
        }
        o += 8 * inside.len();
        let magnitude: Option<Vec<f64>> = c.magnitude.then(|| {
            let m = raw[o..o + 4 * n].as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b) as f64).collect();
            o += 4 * n;
            m
        });
        let original: Vec<f64> =
            raw[o..o + 4 * inside.len()].as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b) as f64).collect();
        o += 4 * inside.len();

        // The QSM-CI wrapper's call: γ 42.58, B0 3 T, simulated TE 8 ms, the magnitude both as the
        // weight (normalised to unit mean in the mask) and for the morphology mask.
        let params = AmpPeParams {
            wave_order: c.wave_order,
            nlevel: c.nlevel,
            max_linearization_ite: 6,
            b0: 3.0,
            normalize_weight: true,
            ..Default::default()
        };
        let grid = Grid::new(c.dims.0, c.dims.1, c.dims.2, c.vs.0, c.vs.1, c.vs.2);
        let chi = amp_pe(&field, &mask, magnitude.as_deref(), magnitude.as_deref(), &grid, c.h, &params, |_, _| {})
            .expect("amp_pe");
        let got: Vec<f64> = inside.iter().map(|&i| chi[i]).collect();
        let (mut d2, mut b2, mut dmax) = (0.0, 0.0, 0.0f64);
        for (a, b) in got.iter().zip(&original) {
            d2 += (a - b).powi(2);
            b2 += b * b;
            dmax = dmax.max((a - b).abs());
        }
        let nrmse = 100.0 * (d2 / b2).sqrt();
        println!("{:?} db{} nlevel {}: NRMSE {nrmse:.3e} %, max |Δ| {dmax:.3e} ppm", c.dims, c.wave_order, c.nlevel);
        // Not to rounding: 1.09 % and 1.41 % (max |Δ| 3.5e-3, 2.4e-3 ppm), the documented parity
        // (astewartau/QSM.rs#363, #371): QSM.rs computes the two operator norms exactly where the
        // original estimates them with two random probes, and keeps the exact norms by decision
        // (`src/inversion/amp_pe.rs`). The limits hold it at that size.
        let (nrmse_max, dmax_max) = if c.magnitude { (1.3, 4e-3) } else { (1.6, 2.8e-3) };
        assert!(nrmse <= nrmse_max && dmax <= dmax_max, "{:?}: NRMSE {nrmse} % (limit {nrmse_max}), max |Δ| {dmax} (limit {dmax_max})", c.dims);
    }
    assert_eq!(o, raw.len());
}

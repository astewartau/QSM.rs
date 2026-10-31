//! QSMART stage by stage against the authors' MATLAB toolbox (QSMART v1.0), as the verified QSM-CI
//! `qsmart-matlab` v3 container runs it (QSMxT/QSM-CI#296, bit-identical to licensed MATLAB R2023b in
//! every intermediate), astewartau/QSM.rs#317.
//!
//! `tests/data/qsmart_qsmci_small.bin`, made by `tests/data/qsmart_qsmci_small.py` (which builds the
//! phantom and holds the digest; regenerated in Actions by `.github/workflows/regenerate-fixtures.yml`):
//! a 36x36x28 multi-echo vessel phantom at 1 mm and 3 T (TE 4, 8, 12 ms), its wrapped phases and
//! magnitudes, and the container's intermediates. Each stage of QSM.rs's QSMART is given the
//! container's inputs to that stage, so a difference is the stage's own:
//!
//! | stage | QSM.rs | the toolbox | compared with |
//! |---|---|---|---|
//! | vessels | `qsmart_vessel_magnitude`, `generate_vasculature_mask` | `vasculature_mask` | `vasc_only` |
//! | total field | `laplacian_unwrap` (`Mirror`, unmasked), `qsmart_echofit` | `unwrap_phase`, `echofit` | `tfs`, `R_0` |
//! | SDF 1, 2 | `sdf` | `QSMART_SDF` | `lfs1`, `lfs2` |
//! | iLSQR 1, 2 | `ilsqr` (50 iterations, STI's padding) | STI Suite's `QSM_iLSQR` | `chi1`, `chi2` |
//! | combination | `adjust_offset` | `adjust_offset` | `chimap` |
//!
//! and `run_qsmart` end to end from the container's `tfs` and `R_0`. In this phantom `R_0` is 1 in
//! the whole mask (the demo's `fit_threshold` 40 is far above its residuals).

use qsm_core::bgremove::{sdf, SdfParams};
use qsm_core::inversion::{ilsqr, IlsqrParams};
use qsm_core::unwrap::{laplacian_unwrap, LaplacianSolver};
use qsm_core::utils::{
    adjust_offset, generate_vasculature_mask, QsmartOffsetGrid, qsmart_echofit, qsmart_vessel_magnitude, QsmartFitThreshold,
    QsmartParams, VasculatureParams,
};
use qsm_core::Grid;

const DIMS: [usize; 3] = [36, 36, 28];
const B0: f64 = 3.0;
const TES: [f64; 3] = [0.004, 0.008, 0.012];
/// QSMART's `params.ppm`: γ·B0/1e6 with its γ = 2.675e8 rad/s/T.
const PPM: f64 = 2.675e8 * B0 / 1e6;

struct Fixture {
    mask: Vec<u8>,
    phases: Vec<Vec<f64>>,
    magnitudes: Vec<Vec<f64>>,
    /// The container's intermediates: over the grid
    vasc_only: Vec<f64>,
    r_0: Vec<f64>,
    /// at the mask voxels, then put back on the grid (0 outside)
    tfs: Vec<f64>,
    lfs1: Vec<f64>,
    lfs2: Vec<f64>,
    chi1: Vec<f64>,
    chi2: Vec<f64>,
    chi: Vec<f64>,
}

fn grid() -> Grid {
    Grid::new(DIMS[0], DIMS[1], DIMS[2], 1.0, 1.0, 1.0)
}

fn load() -> Fixture {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/qsmart_qsmci_small.bin")).expect("fixture");
    let n = DIMS.iter().product::<usize>();
    let f32s = |b: &[u8]| -> Vec<f64> { b.as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c) as f64).collect() };
    let mask = raw[..n].to_vec();
    let inside: Vec<usize> = (0..n).filter(|&i| mask[i] != 0).collect();
    let m = inside.len();
    let mut o = n;
    let mut grid_f32 = || {
        let v = f32s(&raw[o..o + 4 * n]);
        o += 4 * n;
        v
    };
    let phases: Vec<Vec<f64>> = (0..3).map(|_| grid_f32()).collect();
    let magnitudes: Vec<Vec<f64>> = (0..3).map(|_| grid_f32()).collect();
    let mut u8s = || {
        let v: Vec<f64> = raw[o..o + n].iter().map(|&b| b as f64).collect();
        o += n;
        v
    };
    let (vasc_only, r_0) = (u8s(), u8s());
    let mut at_mask = || {
        let mut v = vec![0.0; n];
        for (k, x) in f32s(&raw[o..o + 4 * m]).into_iter().enumerate() {
            v[inside[k]] = x;
        }
        o += 4 * m;
        v
    };
    let (tfs, lfs1, lfs2, chi1, chi2, chi) = (at_mask(), at_mask(), at_mask(), at_mask(), at_mask(), at_mask());
    assert_eq!(o, raw.len());
    Fixture { mask, phases, magnitudes, vasc_only, r_0, tfs, lfs1, lfs2, chi1, chi2, chi }
}

/// NRMSE (%), r and max |Δ| inside the mask, printed.
fn parity(name: &str, a: &[f64], b: &[f64], mask: &[u8]) -> (f64, f64, f64) {
    let idx: Vec<usize> = (0..mask.len()).filter(|&i| mask[i] != 0).collect();
    let n = idx.len() as f64;
    let (ma, mb) = (idx.iter().map(|&i| a[i]).sum::<f64>() / n, idx.iter().map(|&i| b[i]).sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb, mut d2, mut b2, mut dmax) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0f64);
    for &i in &idx {
        let (x, y) = (a[i], b[i]);
        sab += (x - ma) * (y - mb);
        saa += (x - ma).powi(2);
        sbb += (y - mb).powi(2);
        d2 += (x - y).powi(2);
        b2 += y * y;
        dmax = dmax.max((x - y).abs());
    }
    let (nrmse, r) = (100.0 * (d2 / b2).sqrt(), sab / (saa * sbb).sqrt());
    println!("{name}: NRMSE {nrmse:.3e} %, r {r:.8}, max |Δ| {dmax:.3e}");
    (nrmse, r, dmax)
}

fn assert_parity(name: &str, a: &[f64], b: &[f64], mask: &[u8], nrmse_max: f64, dmax_max: f64) {
    let (nrmse, _, dmax) = parity(name, a, b, mask);
    assert!(nrmse <= nrmse_max && dmax <= dmax_max,
        "{name}: NRMSE {nrmse:.3e} % (limit {nrmse_max:e}), max |Δ| {dmax:.3e} (limit {dmax_max:e})");
}

/// `mask .* R_0`, the region both stages reconstruct.
fn mask_r0(f: &Fixture) -> Vec<f64> {
    f.mask.iter().zip(&f.r_0).map(|(&m, &r)| if m != 0 && r != 0.0 { 1.0 } else { 0.0 }).collect()
}

fn sdf_params(stage: usize) -> SdfParams {
    let q = QsmartParams::for_field_strength(B0);
    let (sigma1, sigma2) = if stage == 1 { (q.sdf_sigma1_stage1, q.sdf_sigma2_stage1) } else { (q.sdf_sigma1_stage2, q.sdf_sigma2_stage2) };
    SdfParams { sigma1, sigma2, lower_lim: q.sdf_lower_lim, curv_constant: q.sdf_curv_constant, use_curvature: true }
}

fn ilsqr_params() -> IlsqrParams {
    let q = QsmartParams::for_field_strength(B0);
    IlsqrParams { max_iter: q.ilsqr_max_iter, pad_mm: q.ilsqr_pad_mm, ..Default::default() }
}

#[test]
fn vessel_mask_matches_qsmart() {
    let f = load();
    let mags: Vec<&[f64]> = f.magnitudes.iter().map(|v| v.as_slice()).collect();
    let vmag = qsmart_vessel_magnitude(&mags, &grid()).unwrap();
    let vasc = generate_vasculature_mask(&vmag, &f.mask, &grid(), &VasculatureParams::default(), |_, _| {}).unwrap();
    let (mut differ, mut vessels) = (0, 0);
    for i in 0..vasc.len() {
        if f.mask[i] != 0 {
            differ += usize::from((vasc[i] > 0.5) != (f.vasc_only[i] > 0.5));
            vessels += usize::from(f.vasc_only[i] < 0.5);
        }
    }
    println!("vessel mask: {differ} of {} mask voxels differ ({vessels} vessel voxels in QSMART's)", f.mask.iter().filter(|&&m| m != 0).count());
    assert_eq!(differ, 0);
}

#[test]
fn total_field_matches_qsmart() {
    let f = load();
    let ones = vec![1u8; f.mask.len()];
    let unwrapped: Vec<Vec<f64>> = f.phases.iter().map(|p| laplacian_unwrap(p, &ones, &grid(), LaplacianSolver::Mirror)).collect();
    let ph: Vec<&[f64]> = unwrapped.iter().map(|v| v.as_slice()).collect();
    let mags: Vec<&[f64]> = f.magnitudes.iter().map(|v| v.as_slice()).collect();
    let fit = qsmart_echofit(&ph, &mags, &TES, &grid(), PPM, QsmartFitThreshold::Absolute(40.0));
    // Laplacian unwrapping leaves a constant free; QSM.rs's and lapunwrap's differ by 4.3e-3 ppm
    // here (the total field's NRMSE is 1.3 % with it). Compare without it, and hold it.
    let inside: Vec<usize> = (0..f.mask.len()).filter(|&i| f.mask[i] != 0).collect();
    let offset = inside.iter().map(|&i| fit.field_ppm[i] - f.tfs[i]).sum::<f64>() / inside.len() as f64;
    println!("tfs: constant offset {offset:.4e} ppm");
    assert!(offset.abs() < 5e-3, "tfs: constant offset {offset} ppm");
    let tfs: Vec<f64> = fit.field_ppm.iter().map(|v| v - offset).collect();
    assert_parity("tfs, offset removed", &tfs, &f.tfs, &f.mask, 3e-2, 6e-4);
    let differ = (0..f.mask.len()).filter(|&i| f.mask[i] != 0 && (fit.reliability[i] != 0) != (f.r_0[i] != 0.0)).count();
    assert_eq!(differ, 0, "R_0 differs at {differ} mask voxels");
}

#[test]
fn sdf_stage1_matches_qsmart() {
    let f = load();
    let ones = vec![1.0; f.mask.len()];
    let lfs1 = sdf(&f.tfs, &mask_r0(&f), &ones, &grid(), &sdf_params(1), |_, _| {});
    // An open gap (astewartau/QSM.rs#320): 10.9 % from the container's own total field. Held at
    // the size measured; tighten once found.
    assert_parity("SDF stage 1", &lfs1, &f.lfs1, &f.mask, 11.5, 5e-3);
}

#[test]
fn sdf_stage2_matches_qsmart() {
    let f = load();
    let w = mask_r0(&f);
    let field: Vec<f64> = f.tfs.iter().zip(&w).map(|(t, m)| t * m).collect();
    let lfs2 = sdf(&field, &w, &f.vasc_only, &grid(), &sdf_params(2), |_, _| {});
    // An open gap (astewartau/QSM.rs#320): 2.1 %, as stage 1.
    assert_parity("SDF stage 2", &lfs2, &f.lfs2, &f.mask, 2.3, 8.5e-3);
}

#[test]
fn ilsqr_stages_match_qsmart() {
    let f = load();
    let w = mask_r0(&f);
    let m1: Vec<u8> = w.iter().map(|&v| u8::from(v > 0.5)).collect();
    let m2: Vec<u8> = w.iter().zip(&f.vasc_only).map(|(&v, &t)| u8::from(v > 0.5 && t > 0.5)).collect();
    // STI Suite's single-precision LSQR at 50 iterations: the gap of `src/inversion/ilsqr.rs`
    // (3.4 % and 1.6 % on tests/ilsqr_sti_parity.rs's phantoms; astewartau/QSM.rs#329).
    let (chi1, ..) = ilsqr(&f.lfs1, &m1, &grid(), (0.0, 0.0, 1.0), &ilsqr_params(), |_, _| {});
    assert_parity("iLSQR stage 1", &chi1, &f.chi1, &m1, 7.0, 6e-3);
    let (chi2, ..) = ilsqr(&f.lfs2, &m2, &grid(), (0.0, 0.0, 1.0), &ilsqr_params(), |_, _| {});
    assert_parity("iLSQR stage 2", &chi2, &f.chi2, &m2, 2.6, 2e-3);
}

#[test]
fn offset_adjustment_matches_qsmart() {
    let f = load();
    let removed: Vec<f64> = mask_r0(&f).iter().zip(&f.vasc_only).map(|(m, v)| m - v).collect();
    let mut chi = adjust_offset(&removed, &f.lfs1, &f.chi1, &f.chi2, &grid(), (0.0, 0.0, 1.0), 1.0, QsmartOffsetGrid::QsmartReference);
    for (c, &m) in chi.iter_mut().zip(&f.mask) {
        if m == 0 {
            *c = 0.0;
        }
    }
    assert_parity("adjust_offset", &chi, &f.chi, &f.mask, 1e-5, 1e-8);
}

#[test]
fn run_qsmart_matches_qsmart_end_to_end() {
    use qsm_core::pipeline::{run_qsmart, InversionAlgorithm, InversionConfig, QsmReference, ScanMetadata};
    let f = load();
    let mags: Vec<&[f64]> = f.magnitudes.iter().map(|v| v.as_slice()).collect();
    let vmag = qsmart_vessel_magnitude(&mags, &grid()).unwrap();
    let r0: Vec<u8> = f.r_0.iter().map(|&v| u8::from(v != 0.0)).collect();
    let meta = ScanMetadata {
        dims: (DIMS[0], DIMS[1], DIMS[2]),
        voxel_size: (1.0, 1.0, 1.0),
        echo_times: TES.to_vec(),
        field_strength: B0,
        b0_direction: (0.0, 0.0, 1.0),
        slice_geometry: None,
        affine: None,
    };
    let config = InversionConfig {
        algorithm: InversionAlgorithm::Qsmart,
        qsmart: QsmartParams::for_field_strength(B0),
        ..Default::default()
    };
    let chi = run_qsmart(&f.tfs, &f.mask, Some(&r0), Some(&vmag), &meta, &config, QsmReference::None, &mut ()).unwrap();
    // SDF's and iLSQR's gaps compounded (above): 14.7 %. Held at the size measured.
    assert_parity("run_qsmart", &chi, &f.chi, &f.mask, 15.5, 3.5e-2);
}

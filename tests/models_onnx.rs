//! The deep-learning ports against their originals (`onnx` feature).
//!
//! # Fixture tests
//!
//! Each port runs on a small box of real data and is compared with its original's output on the
//! same box, made by the original's verified QSM-CI entry (QSMxT/QSM-CI#316: the authors' code and
//! weights, pinned by digest, bit-identical run to run), astewartau/QSM.rs#317:
//!
//! - `tests/data/dl_invivo_small.bin`: the inputs, a 36x32x28 box of the public travelling-head
//!   scan `prisma-bridge-run1` (Siemens Prisma, 2.89 T, 5 echoes, 1 mm) across the brain's edge,
//!   cut by `scripts/dl_fixture_inputs.py` (provenance and layout there).
//! - `tests/data/dl_<group>_qsmci_small.bin`, one per group of originals, made by the generator
//!   next to it (`dl_<group>_qsmci_small.py`, which runs the containers and holds their digests),
//!   so it can be regenerated in Actions (`.github/workflows/regenerate-fixtures.yml`). Each
//!   group's tests are a module of this binary, in `tests/models_onnx/`: `dipole` (inversions
//!   from the local field), `field` (from the total field or the phase) and `separation`
//!   (χ-separation, R2′ and brain extraction).
//!
//! The weights are the registry's (fetched with `download`, or from `$QSM_MODEL_DIR`). Each test
//! reports NRMSE, Pearson r and max |Δ| inside the mask, and asserts limits set from what was
//! measured, which each port's module docs record. These run in CI as the `models_onnx` unit of
//! integration-tests.yml (`scripts/ci_shards.py`), from the `onnx` build; locally:
//!
//! ```bash
//! cargo test --release --features "onnx download" --test models_onnx -- --nocapture
//! ```
//!
//! # Manual tests
//!
//! The `#[ignore]` tests below are the ones without a fixture yet: they read references from local
//! paths (env vars), or measure memory and speed.
#![cfg(feature = "onnx")]

use qsm_core::Grid;

/// The box of `tests/data/dl_invivo_small.bin` (`scripts/dl_fixture_inputs.py`).
const DIMS: [usize; 3] = [36, 32, 28];
/// Echo times (s), field strength (T) and B0 direction of the box's scan.
#[allow(dead_code)] // each group's tests read some of these
const TES: [f64; 5] = [0.005, 0.011, 0.017, 0.023, 0.029];
#[allow(dead_code)] // as TES
const B0: f64 = 2.8946;
#[allow(dead_code)] // as TES
const B0_DIR: (f64, f64, f64) = (0.0, 0.0, 1.0);

/// The volumes of `tests/data/dl_invivo_small.bin`, column-major over [`DIMS`] at 1 mm.
#[allow(dead_code)] // each group's tests read some of them
struct Invivo {
    mask: Vec<u8>,
    /// Total and local field (ppm), χ (ppm) and R2′ (s⁻¹)
    total_field: Vec<f64>,
    local_field: Vec<f64>,
    chi: Vec<f64>,
    r2prime: Vec<f64>,
    /// Per echo, in [`TES`] order: magnitude and phase (rad)
    magnitude: Vec<Vec<f64>>,
    phase: Vec<Vec<f64>>,
}

fn data_file(name: &str) -> Vec<u8> {
    let p = format!("{}/tests/data/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&p).unwrap_or_else(|e| panic!("{p}: {e}"))
}

fn f32s(raw: &[u8]) -> Vec<f64> {
    raw.as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b) as f64).collect()
}

fn invivo() -> Invivo {
    let raw = data_file("dl_invivo_small.bin");
    let n = DIMS.iter().product::<usize>();
    assert_eq!(raw.len(), n + 4 * n * 14);
    let vol = |i: usize| f32s(&raw[n + 4 * n * i..n + 4 * n * (i + 1)]);
    Invivo {
        mask: raw[..n].to_vec(),
        total_field: vol(0),
        local_field: vol(1),
        chi: vol(2),
        r2prime: vol(3),
        magnitude: (0..5).map(|e| vol(4 + e)).collect(),
        phase: (0..5).map(|e| vol(9 + e)).collect(),
    }
}

fn grid() -> Grid {
    Grid::new(DIMS[0], DIMS[1], DIMS[2], 1.0, 1.0, 1.0)
}

/// A group fixture's outputs: `count` volumes, each float32 at the mask voxels.
fn originals(name: &str, mask: &[u8], count: usize) -> Vec<Vec<f64>> {
    let raw = data_file(name);
    let m = mask.iter().filter(|&&v| v != 0).count();
    assert_eq!(raw.len(), 4 * m * count, "{name}: expected {count} volumes of {m} voxels");
    raw.chunks_exact(4 * m).map(f32s).collect()
}

/// The registry model's first weight file.
fn weights(id: &str) -> qsm_core::models::WeightBytes {
    qsm_core::models::primary_weight(id)
        .unwrap_or_else(|e| panic!("{id} weights ({e}): build with `download` or set $QSM_MODEL_DIR"))
}

/// NRMSE (%), Pearson r and max |Δ| of `port` against `original` (at the mask voxels), printed.
fn parity(name: &str, port: &[f64], mask: &[u8], original: &[f64]) -> (f64, f64, f64) {
    let a: Vec<f64> = port.iter().zip(mask).filter(|(_, &m)| m != 0).map(|(&v, _)| v).collect();
    assert_eq!(a.len(), original.len());
    assert!(a.iter().all(|v| v.is_finite()), "{name}: non-finite output");
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, original.iter().sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb, mut d2, mut b2, mut dmax) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0f64);
    for (&x, &y) in a.iter().zip(original) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma).powi(2);
        sbb += (y - mb).powi(2);
        d2 += (x - y).powi(2);
        b2 += y * y;
        dmax = dmax.max((x - y).abs());
    }
    let (nrmse, r) = (100.0 * (d2 / b2).sqrt(), sab / (saa * sbb).sqrt());
    println!("{name}: NRMSE {nrmse:.2e} %, r {r:.8}, max |Δ| {dmax:.2e}");
    (nrmse, r, dmax)
}

/// Asserts the measured parity: NRMSE (%) and max |Δ| within these limits. Where a port matches
/// its original to float32 rounding, the limits are about 3-5 × what was measured, for other CPUs'
/// rounding.
fn assert_parity(name: &str, port: &[f64], mask: &[u8], original: &[f64], nrmse_max: f64, dmax_max: f64) {
    let (nrmse, _, dmax) = parity(name, port, mask, original);
    assert!(nrmse <= nrmse_max && dmax <= dmax_max,
        "{name}: NRMSE {nrmse:.3e} % (limit {nrmse_max:e}), max |Δ| {dmax:.3e} (limit {dmax_max:e})");
}

#[path = "models_onnx/dipole.rs"]
mod dipole;
#[path = "models_onnx/field.rs"]
mod field;
#[path = "models_onnx/separation.rs"]
mod separation;

// ---------------------------------------------------------------------------------------------
// Manual tests.

use qsm_core::models::onnx::{OnnxModel, Tensor};

/// Memory probe: load `PROBE_ONNX`, run one op at `PROBE_SHAPE` (`c,d,h,w`),
/// print peak RSS (VmHWM). Used to check whether tract streams a given op or
/// materializes a giant im2col buffer at full resolution.
#[test]
#[ignore]
fn onnx_op_memory_probe() {
    fn vmhwm_gb() -> f64 {
        let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                let kb: f64 = rest.split_whitespace().next().unwrap_or("0").parse().unwrap_or(0.0);
                return kb / 1024.0 / 1024.0;
            }
        }
        0.0
    }
    let path = std::env::var("PROBE_ONNX").expect("PROBE_ONNX");
    let shape: Vec<usize> = std::env::var("PROBE_SHAPE")
        .expect("PROBE_SHAPE e.g. 32,192,256,256")
        .split(',').map(|s| s.parse().unwrap()).collect();
    let (c, d, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let bytes = std::fs::read(&path).expect("onnx");
    let model = OnnxModel::load(&bytes).expect("load");
    let n = c * d * h * w;
    let data: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
    println!("probing {path} at [1,{c},{d},{h},{w}] ...");
    match model.run_single(&Tensor::new(vec![1, c, d, h, w], data)) {
        Ok(out) => println!("PROBE OK: out {:?}  peak_RSS={:.2} GB", out.shape, vmhwm_gb()),
        Err(e) => println!("PROBE FAILED: {e}  peak_RSS={:.2} GB", vmhwm_gb()),
    }
}

/// Wall-clock benchmark of the full `inversion::nextqsm` (internal /64 padding,
/// BFR + 6-step unroll) on a real-resolution volume. Reads `BENCH_FIELD` /
/// `BENCH_MASK` niftis and the two onnx from `NEXTQSM_DIR`. Prints load vs
/// inference timing. Not a correctness assertion.
///
/// ```bash
/// BENCH_FIELD=/tmp/bench_field.nii.gz BENCH_MASK=/tmp/bench_mask.nii.gz \
///   cargo test --release --features onnx --test models_onnx nextqsm_bench -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn nextqsm_bench_realsize() {
    use qsm_core::io::read_nifti_file;
    use std::path::Path;
    use std::time::Instant;

    let dir = std::env::var("NEXTQSM_DIR").unwrap_or("/tmp/nextqsm_export".to_string());
    let field = read_nifti_file(Path::new(&std::env::var("BENCH_FIELD").unwrap())).expect("field");
    let mask_nii = read_nifti_file(Path::new(&std::env::var("BENCH_MASK").unwrap())).expect("mask");
    let bf = std::fs::read(format!("{dir}/nextqsm-bf.onnx")).expect("bf onnx");
    let vjp = std::fs::read(format!("{dir}/nextqsm-vjp.onnx")).expect("vjp onnx");
    let grid = qsm_core::Grid { dims: field.dims, voxel_size: (1.0, 1.0, 1.0) };
    let mask: Vec<u8> = mask_nii.data.iter().map(|&v| (v > 0.5) as u8).collect();
    let (dx, dy, dz) = grid.dims;
    let pad = |s: usize| s.div_ceil(64) * 64;
    println!(
        "input {:?} ({} vox) -> padded ({},{},{}) ({} vox)",
        grid.dims, field.data.len(), pad(dx), pad(dy), pad(dz), pad(dx) * pad(dy) * pad(dz)
    );

    let t0 = Instant::now();
    let chi = qsm_core::inversion::nextqsm(&field.data, &mask, &grid, (0.0, 0.0, 1.0), &bf, &vjp)
        .expect("nextqsm");
    let dt = t0.elapsed().as_secs_f64();
    let finite = chi.iter().filter(|v| v.is_finite()).count();
    println!("NeXtQSM (tract) end-to-end on real volume: {dt:.1} s  (finite out: {finite}/{})", chi.len());
}

/// End-to-end weight fetch: with the `download` feature, download a model from
/// its hosted URL into a fresh cache and verify SHA-256. Exercises the real
/// "download on use" path (QSMxT scenario). Needs network.
///
/// The download → cache → checksum plumbing is **model-agnostic**, so this only
/// fetches one model (the smallest, `xqsm` ~20 MB) — re-downloading every hosted
/// model each run would move ~1 GB for no extra coverage. Each model's hosted
/// URL + hash is verified once at upload time. To check a specific model's live
/// URL, set `QSM_DL_MODEL=<id>`.
///
/// ```bash
/// cargo test --features "onnx download" --test models_onnx osf_download -- --ignored --nocapture
/// ```
#[cfg(feature = "download")]
#[test]
#[ignore]
fn osf_download_and_verify() {
    use qsm_core::models::{find_model, primary_weight_bytes, sha256_hex};

    let id = std::env::var("QSM_DL_MODEL").unwrap_or("xqsm".to_string());

    // Fresh cache dir and no bring-your-own override, so this really hits HTTP.
    let tmp = std::env::temp_dir().join(format!("qsm_dl_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::env::set_var("QSM_MODEL_CACHE", &tmp);
    std::env::remove_var("QSM_MODEL_DIR");

    let spec = find_model(&id).unwrap_or_else(|| panic!("unknown model {id}"));
    assert!(spec.is_available(), "{id} should be Available");
    let bytes = primary_weight_bytes(spec).unwrap_or_else(|e| panic!("{id} download: {e}"));
    let f = &spec.files[0];
    assert_eq!(bytes.len() as u64, f.bytes, "{id} size mismatch");
    assert_eq!(sha256_hex(&bytes), f.sha256, "{id} sha mismatch");
    assert!(tmp.join(f.name).is_file(), "{id} should be cached");
    println!("{id}: downloaded {} bytes, sha ok, cached", bytes.len());

    let _ = std::fs::remove_dir_all(&tmp);
}

/// RS2-Net end-to-end parity: `bet::rs2_net` (tract + the Rust nnU-Net pipeline) vs RS2-Net's
/// own nnU-Net pipeline running the same exported graph under ONNX Runtime
/// (`scripts/onnx-export/ref_rs2net.py`, which writes `mag.nii` and `ref_mask.nii.gz`), on an
/// in-vivo mouse GRE magnitude with 0.17×0.20×0.8 mm voxels (separate-z resampling).
///
/// ```bash
/// RS2NET_ONNX=/path/rs2-net.onnx RS2_REF_DIR=/path/ref \
///   cargo test --release --features onnx --test models_onnx rs2net -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn rs2net_matches_python_reference() {
    use qsm_core::bet::{rs2_net, Rs2NetParams};
    use qsm_core::io::read_nifti_file;
    use std::path::Path;

    let onnx = std::fs::read(std::env::var("RS2NET_ONNX").unwrap_or("/tmp/rs2net_export/rs2-net.onnx".into()))
        .expect("RS2NET_ONNX");
    let dir = std::env::var("RS2_REF_DIR").unwrap_or("/tmp/rs2net_ref".into());
    let mag = read_nifti_file(Path::new(&format!("{dir}/mag.nii"))).expect("magnitude");
    let want = read_nifti_file(Path::new(&format!("{dir}/ref_mask.nii.gz"))).expect("reference mask");
    let (nx, ny, nz) = mag.dims;
    let (vx, vy, vz) = mag.voxel_size;
    let grid = qsm_core::Grid::new(nx, ny, nz, vx, vy, vz);
    let t = std::time::Instant::now();
    let mask = rs2_net(&mag.data, &grid, &onnx, &Rs2NetParams::default(), |_, _| {}).expect("rs2_net");
    let (mut inter, mut a, mut b, mut diff) = (0usize, 0usize, 0usize, 0usize);
    for (&m, &r) in mask.iter().zip(&want.data) {
        let (m, r) = (m != 0, r > 0.5);
        inter += (m && r) as usize;
        a += m as usize;
        b += r as usize;
        diff += (m != r) as usize;
    }
    let dice = 2.0 * inter as f64 / (a + b) as f64;
    println!("{:?} @ {:?} mm  Dice vs reference {dice:.6}  differing voxels {diff}  ({:.1}s)",
        mag.dims, mag.voxel_size, t.elapsed().as_secs_f64());
    assert!(dice > 0.999, "Dice {dice}");
}

/// MindGrab end-to-end parity: `bet::mindgrab` (tract + the Rust conform / normalise /
/// largest-component / reslice pipeline) vs `brainchop --skull-strip`'s pipeline (niimath
/// conform / bwlabel / reslice, the same graph under ONNX Runtime) on an in-vivo GRE magnitude.
/// `scripts/onnx-export/ref_mindgrab.py` writes `mag.nii` and `ref_mask.nii.gz`.
///
/// ```bash
/// MINDGRAB_ONNX=/path/mindgrab.onnx MINDGRAB_REF_DIR=/path/ref \
///   cargo test --release --features onnx --test models_onnx mindgrab -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn mindgrab_matches_brainchop() {
    use qsm_core::bet::{mindgrab, MindGrabParams};
    use qsm_core::io::read_nifti_file;
    use std::path::Path;

    let onnx = std::fs::read(std::env::var("MINDGRAB_ONNX").unwrap_or("/tmp/mindgrab_export/mindgrab.onnx".into()))
        .expect("MINDGRAB_ONNX");
    let dir = std::env::var("MINDGRAB_REF_DIR").unwrap_or("/tmp/mindgrab_ref".into());
    let mag = read_nifti_file(Path::new(&format!("{dir}/mag.nii"))).expect("magnitude");
    let (nx, ny, nz) = mag.dims;
    let (vx, vy, vz) = mag.voxel_size;
    let grid = qsm_core::Grid::new(nx, ny, nz, vx, vy, vz);
    let t = std::time::Instant::now();
    let mask = mindgrab(&mag.data, &grid, &mag.affine, &onnx, &MindGrabParams::default(), |_, _| {})
        .expect("mindgrab");
    println!("mindgrab: {:.1}s, {} brain voxels", t.elapsed().as_secs_f64(), mask.iter().filter(|&&m| m != 0).count());
    let want = read_nifti_file(Path::new(&format!("{dir}/ref_mask.nii.gz"))).expect("reference mask");
    let (mut inter, mut a, mut b, mut diff) = (0usize, 0usize, 0usize, 0usize);
    for (&m, &r) in mask.iter().zip(&want.data) {
        let (m, r) = (m != 0, r > 0.5);
        inter += (m && r) as usize;
        a += m as usize;
        b += r as usize;
        diff += (m != r) as usize;
    }
    let dice = 2.0 * inter as f64 / (a + b) as f64;
    println!("{:?} @ {:?} mm  Dice vs reference {dice:.6}  differing voxels {diff}  ({:.1}s)",
        mag.dims, mag.voxel_size, t.elapsed().as_secs_f64());
    assert!(dice > 0.999, "Dice {dice}");
}

/// Parity: `separation::susep_net` (tract; column-major↔NCDHW repack, pad-to-8, de-normalise)
/// must match the authors' ONNX-Runtime recipe on the same inputs, run whole-volume as they do.
/// The second half measures what the sliding-window patch — the only option on a 32-bit host,
/// where whole-volume activations do not fit — costs against that.
///
/// Generate the fixtures first with `scripts/onnx-export/ref_susep_net.py`.
///
/// ```bash
/// SUSEPNET_ONNX=<...>/susep-net.onnx \
///   cargo test --release --features onnx --test models_onnx susep_net_matches_ref -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn susep_net_matches_ref_and_tiles_closely() {
    use qsm_core::io::read_nifti_file;
    use std::path::Path;

    let base = std::env::var("SUSEPNET_REF").unwrap_or("/tmp/susep_net_ref".into());
    let rd = |p: String| read_nifti_file(Path::new(&p)).expect("nii");
    let qsm = rd(format!("{base}/qsm.nii.gz"));
    let lfs = rd(format!("{base}/lfs.nii.gz"));
    let r2p = rd(format!("{base}/r2prime.nii.gz"));
    let ref_pos = rd(format!("{base}/chi_pos_ref.nii.gz"));
    let ref_neg = rd(format!("{base}/chi_neg_ref.nii.gz"));
    let mask_nii = rd(std::env::var("SUSEPNET_MASK").unwrap_or(
        "/home/ashley/repos/qsm/QSM.rs/TEST_DATA/QSM_Dat08c_Mask.nii.gz".into(),
    ));
    let onnx_bytes = std::fs::read(
        std::env::var("SUSEPNET_ONNX").expect("set SUSEPNET_ONNX to a local susep-net.onnx"),
    )
    .expect("onnx");

    let grid = qsm_core::Grid { dims: qsm.dims, voxel_size: qsm.voxel_size };
    let mask: Vec<u8> = mask_nii.data.iter().map(|&v| (v > 0.5) as u8).collect();
    let norm = qsm_core::separation::SusepNetNorm::default();
    let run = |patch| {
        qsm_core::separation::susep_net(
            &lfs.data, &qsm.data, &r2p.data, &mask, &grid, &onnx_bytes, &norm,
            &qsm_core::separation::SusepNetParams { patch },
            |done, total| {
                if done > 0 {
                    println!("  patch {done}/{total}");
                }
            },
        )
        .expect("susep_net")
    };

    // Agreement on the masked brain: correlation, max |Δ|, and relative L2.
    let stats = |a: &[f64], b: &[f64]| -> (f64, f64, f64) {
        let (mut sx, mut sy, mut sxx, mut syy, mut sxy, mut n) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0f64);
        let (mut maxd, mut num, mut den) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..a.len() {
            if mask[i] == 0 {
                continue;
            }
            sx += a[i]; sy += b[i]; sxx += a[i] * a[i]; syy += b[i] * b[i]; sxy += a[i] * b[i];
            n += 1.0;
            maxd = maxd.max((a[i] - b[i]).abs());
            num += (a[i] - b[i]) * (a[i] - b[i]); den += a[i] * a[i];
        }
        let corr = (sxy - sx * sy / n) / ((sxx - sx * sx / n).sqrt() * (syy - sy * sy / n).sqrt());
        (corr, maxd, 100.0 * (num / den).sqrt())
    };

    // Whole volume — must reproduce the Python reference (f32 NIfTI storage rounding aside).
    let (pos, neg, _tot) = run(None);
    let neg_mag: Vec<f64> = neg.iter().map(|&v| -v).collect();
    let (cp, mp, _) = stats(&pos, &ref_pos.data);
    let (cn, mn, _) = stats(&neg_mag, &ref_neg.data);
    println!("SUSEP-Net whole-volume vs Python: χ+ corr={cp:.6} max|Δ|={mp:.3e} | χ− corr={cn:.6} max|Δ|={mn:.3e}");
    assert!(cp > 0.999999 && cn > 0.999999, "correlation too low: χ+={cp} χ−={cn}");
    assert!(mp < 5e-5 && mn < 5e-5, "max abs diff too high: χ+={mp} χ−={mn} ppm");

    // Sliding window — an approximation, and this pins how much of one.
    let (tpos, tneg, _) = run(Some(qsm_core::separation::susep_net::WASM_PATCH));
    let tneg_mag: Vec<f64> = tneg.iter().map(|&v| -v).collect();
    let (tcp, _, tnp) = stats(&pos, &tpos);
    let (tcn, _, tnn) = stats(&neg_mag, &tneg_mag);
    println!("SUSEP-Net tiled vs whole-volume: χ+ corr={tcp:.4} NRMSE={tnp:.2}% | χ− corr={tcn:.4} NRMSE={tnn:.2}%");
    assert!(tcp > 0.95 && tcn > 0.95, "tiled correlation regressed: χ+={tcp} χ−={tcn}");
}

/// Tiled vs whole-volume BFRnet, iQFM, iQSM and iQSM+ on an in-vivo scan, through the pipeline
/// dispatchers, at the tile a browser host runs (`(56, 4)`, 64³ patches) and at the native
/// default (`(128, 8)`), or at the tiles in `QSM_TILES` (`core,halo;core,halo…`). Tiling these
/// networks is an approximation; this pins how much of one. On the 184×256×128 example scan the
/// worst correlation with the whole volume was 0.91 (iQSM, `(48, 8)`).
///
/// `QSM_EXAMPLE_DIR` holds `phase.nii.gz` / `magnitude.nii.gz` (4D, multi-echo), `mask.nii.gz`
/// and `params.json` (`TE` in s, `B0` in T), as in the example scan QSMxT ships; `QSM_WEIGHTS_DIR`
/// the weights (`scripts/fetch_model_weights.py`).
///
/// ```bash
/// QSM_EXAMPLE_DIR=… QSM_WEIGHTS_DIR=… cargo test --release --features "parallel onnx" \
///   --test models_onnx tiled_phase_nets_and_bfrnet -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn tiled_phase_nets_and_bfrnet_track_whole_volume() {
    use qsm_core::io::{load_nifti_4d, read_nifti_file};
    use qsm_core::pipeline::*;
    use std::path::Path;

    let dir = std::env::var("QSM_EXAMPLE_DIR").expect("QSM_EXAMPLE_DIR");
    let weights = std::env::var("QSM_WEIGHTS_DIR").expect("QSM_WEIGHTS_DIR");
    for id in ["bfrnet", "iqfm", "iqsm", "iqsm-plus"] {
        for f in qsm_core::models::find_model(id).unwrap().files {
            let bytes = std::fs::read(format!("{weights}/{}", f.name)).expect("weights");
            qsm_core::models::provide_weights(f.name, bytes).unwrap();
        }
    }
    let read4 = |name: &str| load_nifti_4d(&std::fs::read(format!("{dir}/{name}")).expect(name)).expect(name);
    let (phase, (nx, ny, nz, ne), voxel, _) = read4("phase.nii.gz");
    let (mag, ..) = read4("magnitude.nii.gz");
    let mask: Vec<u8> =
        read_nifti_file(Path::new(&format!("{dir}/mask.nii.gz"))).expect("mask").data.iter().map(|&v| (v > 0.5) as u8).collect();
    let params: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/params.json")).expect("params")).unwrap();
    let tes: Vec<f64> = params["TE"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let b0 = params["B0"].as_f64().unwrap();
    assert_eq!(tes.len(), ne);
    let n = nx * ny * nz;
    let echo = |v: &[f64], e: usize| v[e * n..(e + 1) * n].to_vec();
    let mut phases: Vec<Vec<f64>> = (0..ne).map(|e| echo(&phase, e)).collect();
    for p in &mut phases {
        if p.iter().any(|v| v.abs() > 3.2) {
            scale_phase_to_pi(p);
        }
    }
    let mags: Vec<Vec<f64>> = (0..ne).map(|e| echo(&mag, e)).collect();
    let meta = ScanMetadata {
        dims: (nx, ny, nz),
        voxel_size: voxel,
        echo_times: tes.clone(),
        field_strength: b0,
        b0_direction: (0.0, 0.0, 1.0),
        slice_geometry: None,
        affine: None,
    };

    let stats = |a: &[f64], b: &[f64], m: &[u8]| -> (f64, f64) {
        let (mut sx, mut sy, mut sxx, mut syy, mut sxy, mut k) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0f64);
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for i in 0..a.len() {
            if m[i] != 0 {
                sx += a[i]; sy += b[i]; sxx += a[i] * a[i]; syy += b[i] * b[i]; sxy += a[i] * b[i];
                k += 1.0;
                num += (a[i] - b[i]).powi(2); den += a[i] * a[i];
            }
        }
        let corr = (sxy - sx * sy / k) / ((sxx - sx * sx / k).sqrt() * (syy - sy * sy / k).sqrt());
        (corr, 100.0 * (num / den).sqrt())
    };
    let tiles: Vec<(usize, usize)> = std::env::var("QSM_TILES").ok().map(|s| s.split(';').map(|t| { let (c, h) = t.split_once(',').unwrap(); (c.parse().unwrap(), h.parse().unwrap()) }).collect()).unwrap_or(vec![(56, 4), (128, 8)]);
    let mut worst = f64::INFINITY;
    let mut report = |what: &str, whole: &[f64], run: &dyn Fn(Option<(usize, usize)>) -> Vec<f64>, m: &[u8]| {
        for &tile in &tiles {
            let t = std::time::Instant::now();
            let tiled = run(Some(tile));
            let (corr, nrmse) = stats(whole, &tiled, m);
            println!("{what} tile {tile:?}: corr {corr:.4}, NRMSE {nrmse:.1}% ({:.0} s)", t.elapsed().as_secs_f64());
            worst = worst.min(corr);
        }
    };

    // One echo for the phase networks: the tiling error is per echo.
    let e = 1;
    let m1 = ScanMetadata { echo_times: vec![tes[e]], ..meta.clone() };
    let (ph1, mg1): (Vec<&[f64]>, Vec<&[f64]>) = (vec![&phases[e]], vec![&mags[e]]);
    let iqfm = |tile| run_iqfm(&ph1, &mg1, &mask, &m1, tile, Default::default()).expect("iqfm");
    let iqsm = |tile| run_iqsm(&ph1, &mg1, &mask, &m1, QsmReference::None, tile, Default::default()).expect("iqsm");
    let iqsm_plus = |tile| run_iqsm_plus(&ph1, &mg1, &mask, &m1, QsmReference::None, tile).expect("iqsm+");
    // The networks erode the mask by 3 voxels; compare where they wrote.
    let compare_mask = |v: &[f64]| -> Vec<u8> { v.iter().map(|&x| (x != 0.0) as u8).collect() };
    for (what, run) in [
        ("iQFM", &iqfm as &dyn Fn(Option<(usize, usize)>) -> Vec<f64>),
        ("iQSM", &iqsm),
        ("iQSM+", &iqsm_plus),
    ] {
        let whole = run(None);
        let m = compare_mask(&whole);
        report(what, &whole, run, &m);
    }

    // BFRnet on the total field from the default field mapping.
    let prefs: Vec<&[f64]> = phases.iter().map(Vec::as_slice).collect();
    let mrefs: Vec<&[f64]> = mags.iter().map(Vec::as_slice).collect();
    let field = run_field_mapping(&prefs, Some(&mrefs), &mask, &meta, &FieldMappingConfig::default(), &mut |_, _| {})
        .expect("field mapping")
        .b0_field_ppm;
    let bfrnet = |tile| {
        let config = BgRemovalConfig { algorithm: BgRemovalAlgorithm::Bfrnet, tile, ..Default::default() };
        run_bg_removal(&field, &mask, &meta, &config, None, &mut |_, _| {}).expect("bfrnet").local_field_ppm
    };
    let whole = bfrnet(None);
    report("BFRnet", &whole, &bfrnet, &mask);
    assert!(worst > 0.85, "a tiled run no longer tracks the whole volume (worst corr {worst})");
}

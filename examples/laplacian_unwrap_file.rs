//! Laplacian unwrapping of NIfTI phase, one echo or several, with either solver.
//!
//! ```text
//! cargo run --release --features parallel --example laplacian_unwrap_file -- \
//!     --out out/sub-1 [--solver dct|fft] [--pad 12] [--te 0.00942,0.0197] [--mask mask.nii.gz] \
//!     [--b0-weight-type phase_snr|assumed-decay|tes|...] [--assumed-t2star 0.040] [--b0 3.0] \
//!     [--vs 0.8,0.8,3] echo1_phase.nii.gz [echo2_phase.nii.gz ...]
//! ```
//!
//! Each echo is zeroed outside the mask (if one is given), unwrapped, and zeroed outside the
//! mask again; written as `<out>_echo-<n>_unwrapped.nii.gz`. With `--te` and two or more echoes,
//! also runs [`run_field_mapping`] with Laplacian unwrapping, the same solver and the given
//! `--b0-weight-type` (default `phase_snr`; magnitude is not read, so magnitude-based weights
//! see uniform magnitude) and writes `<out>_fieldmap-ppm.nii.gz`.
//! `--solver fft --pad 64 --b0-weight-type assumed-decay --assumed-t2star 0.040` reproduces STI
//! Suite 3.0's `MRPhaseUnwrap` per echo and UK Biobank's echo combination (up to a global
//! constant in the field map). `--vs` overrides the voxel size read from the header (e.g. with
//! the affine's column norms, as MATLAB computes it).

use qsm_core::io::{read_nifti_file, save_nifti_to_file};
use qsm_core::pipeline::{run_field_mapping, FieldMappingConfig, ScanMetadata, UnwrappingAlgorithm};
use qsm_core::unwrap::{laplacian_unwrap, LaplacianSolver};
use qsm_core::utils::B0WeightType;
use qsm_core::Grid;
use std::path::Path;

fn list(s: &str) -> Vec<f64> {
    s.split(',').map(|v| v.trim().parse().expect("number")).collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut out, mut mask_path, mut tes, mut vs) = (None, None, None, None);
    let (mut solver, mut pad, mut weighting, mut t2star, mut b0) =
        (String::from("dct"), 12usize, String::from("phase_snr"), B0WeightType::DEFAULT_ASSUMED_T2STAR_S, 3.0f64);
    let mut inputs = Vec::new();
    while let Some(a) = args.next() {
        let mut val = || args.next().unwrap_or_else(|| panic!("{a} needs a value"));
        match a.as_str() {
            "--out" => out = Some(val()),
            "--mask" => mask_path = Some(val()),
            "--te" => tes = Some(list(&val())),
            "--vs" => vs = Some(list(&val())),
            "--solver" => solver = val(),
            "--pad" => pad = val().parse().expect("pad"),
            "--b0-weight-type" => weighting = val(),
            "--assumed-t2star" => t2star = val().parse().expect("assumed T2* (s)"),
            "--b0" => b0 = val().parse().expect("field strength (T)"),
            _ => inputs.push(a),
        }
    }
    let out = out.expect("--out <prefix> is required");
    assert!(!inputs.is_empty(), "no input phase files");
    let solver = match solver.as_str() {
        "dct" => LaplacianSolver::Dct,
        "fft" => LaplacianSolver::Fft { pad: [pad; 3] },
        other => panic!("unknown solver {other} (dct|fft)"),
    };

    let echoes: Vec<_> = inputs.iter().map(|p| read_nifti_file(Path::new(p)).expect("read phase")).collect();
    let h = &echoes[0];
    let v = vs.map(|v| (v[0], v[1], v[2])).unwrap_or(h.voxel_size);
    let grid = Grid::new(h.dims.0, h.dims.1, h.dims.2, v.0, v.1, v.2);
    let mask: Vec<u8> = match mask_path {
        Some(p) => read_nifti_file(Path::new(&p)).expect("read mask").data.iter().map(|&x| (x > 0.5) as u8).collect(),
        None => vec![1; grid.n_total()],
    };
    let save = |suffix: &str, data: &[f64]| {
        let p = format!("{out}_{suffix}.nii.gz");
        save_nifti_to_file(Path::new(&p), data, h.dims, h.voxel_size, &h.affine).expect("write");
        println!("wrote {p}");
    };

    let t = std::time::Instant::now();
    for (e, ph) in echoes.iter().enumerate() {
        let input: Vec<f64> = ph.data.iter().zip(&mask).map(|(&x, &b)| if b != 0 { x } else { 0.0 }).collect();
        save(&format!("echo-{}_unwrapped", e + 1), &laplacian_unwrap(&input, &mask, &grid, solver));
    }
    if let Some(tes) = tes.filter(|_| echoes.len() > 1) {
        let b0_weight_type = match B0WeightType::from_str(&weighting) {
            B0WeightType::AssumedDecay { .. } => B0WeightType::AssumedDecay { t2star_s: t2star },
            w => w,
        };
        let config = FieldMappingConfig {
            unwrapping_algorithm: UnwrappingAlgorithm::Laplacian,
            laplacian_solver: solver,
            b0_weight_type,
            ..Default::default()
        };
        let meta = ScanMetadata {
            dims: grid.dims,
            voxel_size: grid.voxel_size,
            echo_times: tes,
            field_strength: b0,
            b0_direction: (0.0, 0.0, 1.0),
            slice_geometry: None,
        };
        let phases: Vec<&[f64]> = echoes.iter().map(|n| n.data.as_slice()).collect();
        let field = run_field_mapping(&phases, None, &mask, &meta, &config, &mut |_, _| {})
            .expect("field mapping");
        println!("B0 weighting {b0_weight_type:?}");
        save("fieldmap-ppm", &field.b0_field_ppm);
    }
    println!("grid {:?} voxel size {:?} solver {solver:?}: {:.1} s", grid.dims, grid.voxel_size, t.elapsed().as_secs_f64());
}

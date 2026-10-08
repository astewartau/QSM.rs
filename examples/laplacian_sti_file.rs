//! STI Suite-style Laplacian unwrapping of NIfTI phase, one echo or several.
//!
//! ```text
//! cargo run --release --features parallel --example laplacian_sti_file -- \
//!     --out out/sub-1 --te 0.00942,0.0197 [--mask mask.nii.gz] [--pad 64] \
//!     [--weighting t2star|te|uniform] [--t2star 0.040] [--vs 0.8,0.8,3] \
//!     echo1_phase.nii.gz echo2_phase.nii.gz
//! ```
//!
//! Writes `<out>_echo-<n>_unwrapped.nii.gz` per echo (each echo multiplied by the mask first,
//! if one is given, and not masked afterwards — as STI's callers do). With `--te`, also
//! `<out>_phase-avg.nii.gz` (weighted mean of the unwrapped phases, rad) and
//! `<out>_fieldmap-hz.nii.gz` (`phase-avg / (2π · TE_eff)`), and prints TE_eff.
//! `--vs` overrides the voxel size read from the header (e.g. to use the affine's column
//! norms, as MATLAB scripts often do).

use qsm_core::io::{read_nifti_file, save_nifti_to_file};
use qsm_core::unwrap::{laplacian_unwrap_sti, laplacian_unwrap_sti_multi_echo, EchoWeighting, LaplacianStiParams};
use qsm_core::Grid;
use std::path::Path;

fn list(s: &str) -> Vec<f64> {
    s.split(',').map(|v| v.trim().parse().expect("number")).collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut out, mut mask_path, mut tes, mut vs) = (None, None, None, None);
    let (mut pad, mut weighting, mut t2star) = (12usize, String::from("t2star"), 0.040f64);
    let mut inputs = Vec::new();
    while let Some(a) = args.next() {
        let mut val = || args.next().unwrap_or_else(|| panic!("{a} needs a value"));
        match a.as_str() {
            "--out" => out = Some(val()),
            "--mask" => mask_path = Some(val()),
            "--te" => tes = Some(list(&val())),
            "--vs" => vs = Some(list(&val())),
            "--pad" => pad = val().parse().expect("pad"),
            "--weighting" => weighting = val(),
            "--t2star" => t2star = val().parse().expect("t2star"),
            _ => inputs.push(a),
        }
    }
    let out = out.expect("--out <prefix> is required");
    assert!(!inputs.is_empty(), "no input phase files");

    let echoes: Vec<_> = inputs.iter().map(|p| read_nifti_file(Path::new(p)).expect("read phase")).collect();
    let h = &echoes[0];
    let v = vs.map(|v| (v[0], v[1], v[2])).unwrap_or(h.voxel_size);
    let grid = Grid::new(h.dims.0, h.dims.1, h.dims.2, v.0, v.1, v.2);
    let mask: Option<Vec<u8>> = mask_path.map(|p| {
        read_nifti_file(Path::new(&p)).expect("read mask").data.iter().map(|&x| (x > 0.5) as u8).collect()
    });
    let params = LaplacianStiParams { pad: [pad; 3] };
    let save = |suffix: &str, data: &[f64]| {
        let p = format!("{out}_{suffix}.nii.gz");
        save_nifti_to_file(Path::new(&p), data, h.dims, h.voxel_size, &h.affine).expect("write");
        println!("wrote {p}");
    };

    let t = std::time::Instant::now();
    for (e, ph) in echoes.iter().enumerate() {
        let input: Vec<f64> = match &mask {
            Some(m) => ph.data.iter().zip(m).map(|(&x, &b)| if b != 0 { x } else { 0.0 }).collect(),
            None => ph.data.clone(),
        };
        save(&format!("echo-{}_unwrapped", e + 1), &laplacian_unwrap_sti(&input, &grid, &params));
    }
    if let Some(tes) = tes {
        let w = match weighting.as_str() {
            "t2star" => EchoWeighting::T2Star { t2star },
            "te" => EchoWeighting::EchoTime,
            "uniform" => EchoWeighting::Uniform,
            other => panic!("unknown weighting {other}"),
        };
        let phases: Vec<&[f64]> = echoes.iter().map(|n| n.data.as_slice()).collect();
        let avg = laplacian_unwrap_sti_multi_echo(&phases, &tes, mask.as_deref(), &grid, &params, &w);
        println!("weights {:?}  TE_eff {}", avg.weights, avg.te_eff);
        save("phase-avg", &avg.phase);
        save("fieldmap-hz", &avg.field_hz());
    }
    println!("grid {:?} voxel size {:?} pad {pad}: {:.1} s", grid.dims, grid.voxel_size, t.elapsed().as_secs_f64());
}

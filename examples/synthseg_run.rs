//! Run SynthSeg on a NIfTI magnitude image and, optionally, compare against a reference
//! segmentation (e.g. one produced by the upstream Python implementation).
//!
//! cargo run --release --features "parallel onnx download" --example synthseg_run -- \
//!     <image.nii[.gz]> <synthseg.onnx|auto> <out.nii.gz> [reference.nii.gz] [--v1|--v2] [--fast] \
//!     [--low-memory] [--patch=N] [--overlap=N] [--head-crop=N] [--no-head-crop] [--f16|--f32]
//!     [--volumes=out.csv]
//!
//! `--low-memory` starts from `SynthSegParams::low_memory()`; `--patch`/`--overlap` set sliding
//! windows, `--head-crop`/`--no-head-crop` the head crop's margin or turn it off, `--f16`/`--f32`
//! the posteriors' precision.
//!
//! Passing `auto` for the model fetches and caches the registry weights (needs `download`).
use qsm_core::io::{read_nifti_file, save_nifti_gz};
use qsm_core::segment::{synthseg, PosteriorPrecision, SynthSegInference, SynthSegParams, SynthSegVersion};
use qsm_core::Grid;
use std::path::Path;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let positional: Vec<&String> = args[1..].iter().filter(|a| !a.starts_with("--")).collect();
    let flag = |f: &str| args.iter().any(|a| a == f);
    let value = |f: &str| -> Option<String> {
        args.iter().find_map(|a| a.strip_prefix(&format!("{f}=")).map(str::to_string))
    };
    let number = |f: &str| value(f).map(|v| v.parse::<usize>().unwrap_or_else(|_| panic!("{f} takes a number")));

    let img = read_nifti_file(Path::new(positional[0])).expect("read image");
    let onnx = if positional[1] == "auto" {
        println!("resolving synthseg weights...");
        qsm_core::models::primary_weight("synthseg").expect("synthseg weights")
    } else {
        std::fs::read(positional[1]).expect("read onnx").into()
    };
    let grid = Grid::new(
        img.dims.0, img.dims.1, img.dims.2,
        img.voxel_size.0, img.voxel_size.1, img.voxel_size.2,
    );
    let base = if flag("--low-memory") { SynthSegParams::low_memory() } else { SynthSegParams::default() };
    let mut params = SynthSegParams {
        version: if flag("--v1") { SynthSegVersion::V1 } else { SynthSegVersion::V2 },
        ..base
    };
    if flag("--fast") {
        params.flip_averaging = false;
        params.topology_cleanup = false;
    }
    if let Some(patch) = number("--patch") {
        params.inference = SynthSegInference::SlidingWindow { patch, overlap: number("--overlap").unwrap_or(patch / 2) };
    }
    if let Some(margin) = number("--head-crop") {
        params.head_crop = Some(margin);
    }
    if flag("--f16") {
        params.posteriors = PosteriorPrecision::F16;
    }
    if flag("--f32") {
        params.posteriors = PosteriorPrecision::F32;
    }
    if flag("--no-head-crop") {
        params.head_crop = None;
    }
    println!("image {:?} @ {:?} mm, {:?}", img.dims, img.voxel_size, params);

    let t = Instant::now();
    let out = synthseg(&img.data, &grid, &img.affine, &onnx, &params, |d, n| {
        println!("  pass {d}/{n} ({:?})", t.elapsed())
    })
    .expect("synthseg");
    println!("total {:?}", t.elapsed());
    println!("SECONDS {:.3}", t.elapsed().as_secs_f64());

    let labels = params.version.labels();
    for (c, name) in labels.names.iter().enumerate().skip(1) {
        if out.volumes[c] > 0.0 {
            println!("  {:<32} {:>10.1} mm3", name, out.volumes[c]);
        }
    }

    if let Some(csv) = value("--volumes") {
        let mut text = String::from("label,name,volume_mm3\n");
        for (c, name) in labels.names.iter().enumerate().skip(1) {
            text += &format!("{},{},{}\n", labels.ids[c], name, out.volumes[c]);
        }
        std::fs::write(&csv, text).expect("write volumes");
    }

    let as_f64: Vec<f64> = out.labels.iter().map(|&v| v as f64).collect();
    let bytes = save_nifti_gz(&as_f64, img.dims, img.voxel_size, &img.affine).expect("encode");
    std::fs::write(positional[2], bytes).expect("write");
    println!("wrote {}", positional[2]);

    if let Some(refp) = positional.get(3) {
        let r = read_nifti_file(Path::new(refp)).expect("read reference");
        assert_eq!(r.dims, img.dims, "reference is on a different grid");
        let same = r.data.iter().zip(&out.labels).filter(|(a, b)| **a as i32 == **b).count();
        let n = r.data.len();
        println!("\nvs {refp}:\n  exact voxel agreement: {:.6}% ({} differ)",
                 100.0 * same as f64 / n as f64, n - same);
        let mut worst: Vec<(f64, i32)> = labels.ids[1..]
            .iter()
            .map(|&l| {
                let (mut inter, mut a, mut b) = (0usize, 0usize, 0usize);
                for (rv, ov) in r.data.iter().zip(&out.labels) {
                    let (x, y) = (*rv as i32 == l, *ov == l);
                    inter += (x && y) as usize;
                    a += x as usize;
                    b += y as usize;
                }
                (if a + b > 0 { 2.0 * inter as f64 / (a + b) as f64 } else { 1.0 }, l)
            })
            .collect();
        worst.sort_by(|x, y| x.0.total_cmp(&y.0));
        let mean = worst.iter().map(|w| w.0).sum::<f64>() / worst.len() as f64;
        println!("  mean Dice: {mean:.6}");
        println!("  lowest:    {:?}", &worst[..worst.len().min(5)]);
    }
}

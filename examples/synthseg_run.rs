//! Run SynthSeg on a NIfTI magnitude image and, optionally, compare against a reference
//! segmentation (e.g. one produced by the upstream Python implementation).
//!
//! cargo run --release --features "parallel onnx" --example synthseg_run -- \
//!     <image.nii[.gz]> <synthseg.onnx> <out.nii.gz> [reference.nii.gz] [--v1|--v2] [--fast]
use qsm_core::io::{read_nifti_file, save_nifti_gz};
use qsm_core::segment::{synthseg, SynthSegParams, SynthSegVersion};
use qsm_core::Grid;
use std::path::Path;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let positional: Vec<&String> = args[1..].iter().filter(|a| !a.starts_with("--")).collect();
    let flag = |f: &str| args.iter().any(|a| a == f);

    let img = read_nifti_file(Path::new(positional[0])).expect("read image");
    let onnx = std::fs::read(positional[1]).expect("read onnx");
    let grid = Grid::new(
        img.dims.0, img.dims.1, img.dims.2,
        img.voxel_size.0, img.voxel_size.1, img.voxel_size.2,
    );
    let params = SynthSegParams {
        version: if flag("--v1") { SynthSegVersion::V1 } else { SynthSegVersion::V2 },
        ..if flag("--fast") { SynthSegParams::fast() } else { SynthSegParams::default() }
    };
    println!("image {:?} @ {:?} mm, {:?}", img.dims, img.voxel_size, params.version);

    let t = Instant::now();
    let out = synthseg(&img.data, &grid, &img.affine, &onnx, &params, |d, n| {
        println!("  pass {d}/{n} ({:?})", t.elapsed())
    })
    .expect("synthseg");
    println!("total {:?}", t.elapsed());

    let labels = params.version.labels();
    for (c, name) in labels.names.iter().enumerate().skip(1) {
        if out.volumes[c] > 0.0 {
            println!("  {:<32} {:>10.1} mm3", name, out.volumes[c]);
        }
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

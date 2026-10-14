//! Run HD-BET on a NIfTI magnitude image (3D; for multi-echo GRE pass the root-sum-of-squares).
//!
//! usage: hdbet_file <magnitude.nii[.gz]> <mask_out.nii[.gz]> [full|window|window:<px>x<py>x<pz>[:<step>]] [tta]
//! - `full` (default): one network pass over the whole volume
//! - `window`: nnU-Net's sliding window, 192x192x96 patches at 50 % overlap (or the given ones)
//! - `tta`: 8-fold mirroring test-time augmentation
//!
//! The weights come from the model registry (`$QSM_MODEL_DIR` or the qsm-rs cache). Prints the
//! number of network evaluations and the run time.
use qsm_core::bet::{hd_bet, HdBetInference, HdBetParams};
use qsm_core::io::{read_nifti_file, save_nifti_to_file};
use qsm_core::Grid;
use std::path::Path;

fn parse_inference(s: &str) -> HdBetInference {
    if s == "full" {
        return HdBetInference::FullVolume;
    }
    let mut parts = s.split(':');
    assert_eq!(parts.next(), Some("window"), "mode must be full or window[:<px>x<py>x<pz>[:<step>]], got {s}");
    let HdBetInference::SlidingWindow { mut patch, mut tile_step } = HdBetInference::SLIDING_WINDOW else { unreachable!() };
    if let Some(p) = parts.next() {
        let d: Vec<usize> = p.split('x').map(|v| v.parse().expect("patch dims")).collect();
        assert_eq!(d.len(), 3, "patch must be <px>x<py>x<pz>");
        patch = (d[0], d[1], d[2]);
    }
    if let Some(st) = parts.next() {
        tile_step = st.parse().expect("step");
    }
    HdBetInference::SlidingWindow { patch, tile_step }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if !(3..=5).contains(&a.len()) {
        eprintln!("usage: {} <magnitude.nii[.gz]> <mask_out.nii[.gz]> [full|window[:<px>x<py>x<pz>[:<step>]]] [tta]", a[0]);
        std::process::exit(2);
    }
    let params = HdBetParams {
        inference: a.get(3).map_or(HdBetInference::FullVolume, |s| parse_inference(s)),
        mirror_tta: a.get(4).is_some_and(|s| s == "tta"),
    };
    let m = read_nifti_file(Path::new(&a[1])).expect("read magnitude");
    let grid = Grid::new(m.dims.0, m.dims.1, m.dims.2, m.voxel_size.0, m.voxel_size.1, m.voxel_size.2);
    let weights = qsm_core::models::primary_weight("hd-bet").expect("hd-bet weights");
    let t = std::time::Instant::now();
    let mut evals = 0;
    let mask = hd_bet(&m.data, &grid, &weights, &params, |_, total| evals = total).expect("hd_bet");
    eprintln!(
        "hd_bet: {:?} @ {:?} mm, {:?}, tta {}, {} network evaluations, {} brain voxels, {:.2}s",
        m.dims, m.voxel_size, params.inference, params.mirror_tta, evals,
        mask.iter().filter(|&&v| v != 0).count(), t.elapsed().as_secs_f64()
    );
    let mask: Vec<f64> = mask.iter().map(|&v| v as f64).collect();
    save_nifti_to_file(Path::new(&a[2]), &mask, m.dims, m.voxel_size, &m.affine).expect("write mask");
}

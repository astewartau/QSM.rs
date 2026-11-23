//! Audit harness (not for commit): QSM.rs QSMART on Warda's 7T exemplar intermediates.
//! usage: warda_parity <case_dir> <out_dir> <mode,...>   env: LOWER_LIM, FRANGI="min,max,ratio", FIT_THR
use qsm_core::bgremove::{sdf, SdfParams};
use qsm_core::inversion::{ilsqr, IlsqrParams, STI_DEFAULT_PAD_MM};
use qsm_core::io::{read_nifti_file, save_nifti_to_file};
use qsm_core::pipeline::{run_qsmart, InversionAlgorithm, InversionConfig, QsmReference, ScanMetadata};
use qsm_core::utils::{adjust_offset, generate_vasculature_mask, QsmartParams, VasculatureParams};
use qsm_core::Grid;
use std::path::Path;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (d, o, modes) = (&a[1], &a[2], &a[3]);
    let want = |m: &str| modes.split(',').any(|x| x == m);
    let rd = |f: &str| read_nifti_file(Path::new(&format!("{d}/{f}.nii"))).unwrap_or_else(|e| panic!("{f}: {e}"));
    let m = rd("mask_used");
    let (dims, vs, aff) = (m.dims, m.voxel_size, m.affine.clone());
    let sv = |f: &str, v: &[f64]| save_nifti_to_file(Path::new(&format!("{o}/{f}.nii")), v, dims, vs, &aff).unwrap();
    let grid = Grid::new(dims.0, dims.1, dims.2, vs.0, vs.1, vs.2);
    let n = m.data.len();
    // QSMART's readComplexDicoms: z_prjs = [IOP(3), IOP(6), cross(IOP(1:3), IOP(4:6))(3)]
    let (r, c) = ([0.99964246883302, 0.02144096400688, 0.01597559288029], [-0.0174488245093, 0.97582856279732, -0.2178397451159]);
    let bdir = (r[2], c[2], r[0] * c[1] - r[1] * c[0]);
    let b0 = 7.0;
    let env = |k: &str, dflt: &str| std::env::var(k).unwrap_or_else(|_| dflt.into());
    let lower_lim: f64 = env("LOWER_LIM", "0.6").parse().unwrap();
    let fr: Vec<f64> = env("FRANGI", "0.5,6,0.5").split(',').map(|x| x.parse().unwrap()).collect();
    let fit_thr: f64 = env("FIT_THR", "40").parse().unwrap();
    eprintln!("bdir {bdir:?} lower_lim {lower_lim} frangi {fr:?} fit_thr {fit_thr}");
    let mask: Vec<u8> = m.data.iter().map(|&v| (v > 0.5) as u8).collect();
    let blur = rd("fit_residual_blur").data;
    let r0: Vec<u8> = blur.iter().map(|&b| u8::from(b < fit_thr || b.is_nan())).collect();
    let mr0: Vec<f64> = (0..n).map(|i| (mask[i] * r0[i]) as f64).collect();
    let tfs = rd("tfs").data;
    let mag = rd("AvgEcho_n4").data;
    let vp = VasculatureParams { sphere_radius: 8, frangi_scale_range: [fr[0], fr[1]], frangi_scale_ratio: fr[2], frangi_c: 500.0 };
    let mut q = QsmartParams::for_field_strength(b0);
    q.sdf_lower_lim = lower_lim;
    q.vasc_sphere_radius = 8;
    q.frangi_scale_range = [fr[0], fr[1]];
    q.frangi_scale_ratio = fr[2];
    let t = std::time::Instant::now();
    if want("vasc") {
        let v = generate_vasculature_mask(&mag, &mask, &grid, &vp, |_, _| {}); sv("vasc_only", &v);
        eprintln!("vasc {:.0}s", t.elapsed().as_secs_f64());
    }
    if want("sdf1") {
        let p = SdfParams { sigma1: 10.0, sigma2: 10.0, lower_lim, curv_constant: 500.0, use_curvature: true };
        sv("lfs_sdf_1", &sdf(&tfs, &mr0, &vec![1.0; n], &grid, &p, |_, _| {}));
        eprintln!("sdf1 {:.0}s", t.elapsed().as_secs_f64());
    }
    if want("sdf2") {
        let vasc = rd("vasc_only").data; // hers, to isolate SDF
        let tw: Vec<f64> = tfs.iter().zip(&mr0).map(|(a, b)| a * b).collect();
        let p = SdfParams { sigma1: 8.0, sigma2: 2.0, lower_lim, curv_constant: 500.0, use_curvature: true };
        sv("lfs_sdf_2", &sdf(&tw, &mr0, &vasc, &grid, &p, |_, _| {}));
        eprintln!("sdf2 {:.0}s", t.elapsed().as_secs_f64());
    }
    if want("ilsqr") {
        let ip = IlsqrParams { max_iter: 50, pad_mm: STI_DEFAULT_PAD_MM, ..IlsqrParams::default() };
        let m1: Vec<u8> = mr0.iter().map(|&v| (v > 0.5) as u8).collect();
        let (c1, ..) = ilsqr(&rd("lfs_sdf_1").data, &m1, &grid, bdir, &ip, |_, _| {}); sv("QSM_1_from_her_lfs", &c1);
        let vasc = rd("vasc_only").data;
        let m2: Vec<u8> = (0..n).map(|i| m1[i] * (vasc[i] > 0.5) as u8).collect();
        let (c2, ..) = ilsqr(&rd("lfs_sdf_2").data, &m2, &grid, bdir, &ip, |_, _| {}); sv("QSM_2_from_her_lfs", &c2);
        eprintln!("ilsqr {:.0}s", t.elapsed().as_secs_f64());
    }
    if want("offset") {
        let vasc = rd("vasc_only").data;
        let rem: Vec<f64> = (0..n).map(|i| mr0[i] - vasc[i]).collect();
        let c = adjust_offset(&rem, &rd("lfs_sdf_1").data, &rd("QSM_1").data, &rd("QSM_2").data, &grid, bdir, 1.0);
        sv("QSMART_adjusted_offset_from_her_chis", &c);
    }
    if want("chain") {
        // QSM.rs from her total field with HER vessel mask: SDF (both stages), iLSQR, offset
        let vasc = rd("vasc_only").data;
        let ip = IlsqrParams { max_iter: 50, pad_mm: STI_DEFAULT_PAD_MM, ..IlsqrParams::default() };
        let l1 = sdf(&tfs, &mr0, &vec![1.0; n], &grid, &SdfParams { sigma1: 10.0, sigma2: 10.0, lower_lim, curv_constant: 500.0, use_curvature: true }, |_, _| {});
        let tw: Vec<f64> = tfs.iter().zip(&mr0).map(|(a, b)| a * b).collect();
        let l2 = sdf(&tw, &mr0, &vasc, &grid, &SdfParams { sigma1: 8.0, sigma2: 2.0, lower_lim, curv_constant: 500.0, use_curvature: true }, |_, _| {});
        let m1: Vec<u8> = mr0.iter().map(|&v| (v > 0.5) as u8).collect();
        let m2: Vec<u8> = (0..n).map(|i| m1[i] * (vasc[i] > 0.5) as u8).collect();
        let (c1, ..) = ilsqr(&l1, &m1, &grid, bdir, &ip, |_, _| {});
        let (c2, ..) = ilsqr(&l2, &m2, &grid, bdir, &ip, |_, _| {});
        let rem: Vec<f64> = (0..n).map(|i| mr0[i] - vasc[i]).collect();
        let mut c = adjust_offset(&rem, &l1, &c1, &c2, &grid, bdir, 1.0);
        for i in 0..n { if mask[i] == 0 { c[i] = 0.0; } }
        sv("full_local_field_stage1", &l1); sv("full_local_field_stage2", &l2); sv("full_chi_stage1", &c1); sv("full_chi_stage2", &c2);
        sv("QSMART_qsmrs", &c);
        eprintln!("chain {:.0}s", t.elapsed().as_secs_f64());
    }
    if want("full") {
        let meta = ScanMetadata { dims, voxel_size: vs, echo_times: vec![0.0051], field_strength: b0, b0_direction: bdir, slice_geometry: None, affine: None };
        let cfg = InversionConfig { algorithm: InversionAlgorithm::Qsmart, qsmart: q.clone(), ..Default::default() };
        let mut obs = Saver { o: o.clone(), dims, vs, aff: aff.clone() };
        let chi = run_qsmart(&tfs, &mask, Some(&r0), Some(&mag), &meta, &cfg, QsmReference::None, &mut obs).unwrap();
        sv("QSMART_qsmrs", &chi);
        eprintln!("full {:.0}s", t.elapsed().as_secs_f64());
    }
}

struct Saver { o: String, dims: (usize, usize, usize), vs: (f64, f64, f64), aff: [f64; 16] }
impl qsm_core::pipeline::observer::Observer for Saver {
    fn progress(&mut self, _: usize, _: usize) {}
    fn output(&mut self, name: &'static str, v: qsm_core::pipeline::observer::VolumeRef<'_>) {
        let data: Vec<f64> = match v {
            qsm_core::pipeline::observer::VolumeRef::Real(x) => x.to_vec(),
            qsm_core::pipeline::observer::VolumeRef::Mask(x) => x.iter().map(|&b| b as f64).collect(),
        };
        save_nifti_to_file(Path::new(&format!("{}/full_{name}.nii", self.o)), &data, self.dims, self.vs, &self.aff).unwrap();
    }
}

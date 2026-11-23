//! Audit harness (not for commit): QSM.rs QSMART components fed the original MATLAB's inputs.
use qsm_core::bgremove::{sdf, SdfParams};
use qsm_core::inversion::{ilsqr, IlsqrParams, STI_DEFAULT_PAD_MM};
use qsm_core::io::{load_nifti_4d, read_nifti_file, save_nifti_to_file};
use qsm_core::pipeline::{run_qsmart, InversionAlgorithm, InversionConfig, QsmReference, ScanMetadata};
use qsm_core::utils::{adjust_offset, calculate_curvature_proximity, frangi_filter_3d, generate_vasculature_mask,
    FrangiParams, QsmartParams, VasculatureParams};
use qsm_core::Grid;
use qsm_core::utils::{makehomogeneous, HomogeneityParams};
use std::path::Path;

fn gauss1(d: &[f64], n: (usize, usize, usize), s: f64, ax: usize) -> Vec<f64> {
    let r = (2.0 * s).ceil() as isize;
    let mut k: Vec<f64> = (-r..=r).map(|x| (-(x * x) as f64 / (2.0 * s * s)).exp()).collect();
    let t: f64 = k.iter().sum(); k.iter_mut().for_each(|v| *v /= t);
    let (nx, ny, nz) = n; let mut o = vec![0.0; d.len()];
    let len = [nx, ny, nz][ax] as isize; let st = [1, nx, nx * ny][ax];
    for z in 0..nz { for y in 0..ny { for x in 0..nx {
        let i = x + y * nx + z * nx * ny; let c = [x, y, z][ax] as isize; let mut acc = 0.0;
        for (j, w) in k.iter().enumerate() { let p = (c + j as isize - r).clamp(0, len - 1); acc += w * d[(i as isize + (p - c) * st as isize) as usize]; }
        o[i] = acc; }}}
    o
}

fn read4(p: &str) -> Vec<f64> { load_nifti_4d(&std::fs::read(p).unwrap()).unwrap().0 }

fn main() {
    let e = std::env::args().nth(1).unwrap();
    let rd = |f: &str| read_nifti_file(Path::new(&format!("{e}/mat/{f}.nii"))).unwrap_or_else(|x| panic!("{f}: {x}"));
    let m = read_nifti_file(Path::new(&format!("{e}/data/mask.nii"))).unwrap();
    let dims = m.dims; let vs = m.voxel_size; let aff = m.affine.clone();
    let sv = |f: &str, d: &[f64]| save_nifti_to_file(Path::new(&format!("{e}/rs/{f}.nii")), d, dims, vs, &aff).unwrap();
    let grid = Grid::new(dims.0, dims.1, dims.2, vs.0, vs.1, vs.2);
    let p: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{e}/data/params.json")).unwrap()).unwrap();
    let h: Vec<f64> = p["B0_dir"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let nrm = (h[0] * h[0] + h[1] * h[1] + h[2] * h[2]).sqrt(); let bdir = (h[0] / nrm, h[1] / nrm, h[2] / nrm);
    let b0 = p["B0"].as_f64().unwrap();
    let mask: Vec<u8> = m.data.iter().map(|&v| (v > 0.5) as u8).collect();
    let r0 = rd("R_0").data; let tfs = rd("tfs").data; let avg = rd("avgecho_n4").data;
    let bothat = rd("bothat").data; let vasc_m = rd("vasc_only").data;
    let mr0: Vec<f64> = mask.iter().zip(&r0).map(|(&a, &b)| a as f64 * b).collect();
    let mr0u: Vec<u8> = mr0.iter().map(|&v| (v > 0.5) as u8).collect();
    let n = mask.len();
    let only = std::env::args().nth(2).unwrap_or_default();
    let t = std::time::Instant::now();
    let want = |s: &str| only.is_empty() || only.split(',').any(|x| x == s);

    if want("vasc") {
        let vp = VasculatureParams::default();
        let v = generate_vasculature_mask(&avg, &mask, &grid, &vp, |_, _| {}); sv("vasc_only", &v);
        let masked: Vec<f64> = bothat.iter().zip(&mask).map(|(&b, &m)| b * m as f64).collect();
        let fr = frangi_filter_3d(&masked, &grid, &FrangiParams::default(), |_, _| {}); sv("enhanced_from_mat_bothat", &fr.vesselness);
        eprintln!("vasc done {:.0}s", t.elapsed().as_secs_f64());
    }
    if want("curv") {
        let s = 10.0; let mut p1 = gauss1(&mr0, dims, s, 0); p1 = gauss1(&p1, dims, 2.0 * s, 1); p1 = gauss1(&p1, dims, 2.0 * s, 2);
        let p1: Vec<f64> = p1.iter().zip(&mr0).map(|(a, b)| a * b).collect();
        let (prox, curv) = calculate_curvature_proximity(&mr0u, &p1, 0.6, 500.0, s, &grid);
        sv("prox1_s1", &p1); sv("prox_s1", &prox); sv("curvature_s1", &curv);
        eprintln!("curv done {:.0}s", t.elapsed().as_secs_f64());
    }
    if want("sdf") {
        let ones = vec![1.0; n];
        let l1 = sdf(&tfs, &mr0, &ones, &grid, &SdfParams { sigma1: 10.0, sigma2: 10.0, ..SdfParams::default() }, |_, _| {});
        sv("lfs1_ppm", &l1);
        let tw: Vec<f64> = tfs.iter().zip(&mr0).map(|(a, b)| a * b).collect();
        let l2 = sdf(&tw, &mr0, &vasc_m, &grid, &SdfParams::stage2(), |_, _| {});
        sv("lfs2_ppm", &l2);
        eprintln!("sdf done {:.0}s", t.elapsed().as_secs_f64());
    }
    if want("ilsqr") {
        let ip = IlsqrParams { max_iter: 50, pad_mm: STI_DEFAULT_PAD_MM, ..IlsqrParams::default() };
        let l1 = rd("lfs1_ppm").data; let l2 = rd("lfs2_ppm").data;
        let (c1, ..) = ilsqr(&l1, &mr0u, &grid, bdir, &ip, |_, _| {}); sv("chi1_from_mat_lfs", &c1);
        let m2: Vec<u8> = mr0u.iter().zip(&vasc_m).map(|(&a, &v)| a * (v > 0.5) as u8).collect();
        let (c2, ..) = ilsqr(&l2, &m2, &grid, bdir, &ip, |_, _| {}); sv("chi2_from_mat_lfs", &c2);
        // STI is fed rad/s with TE=1 s: same thing in Rust, scaled back like STI does
        let ppm = 2.675e8 * b0 / 1e6; let sti = 1.0 / (2.0 * std::f64::consts::PI * 42.575 * b0);
        let l1r: Vec<f64> = l1.iter().map(|v| v * ppm).collect();
        let (c1r, ..) = ilsqr(&l1r, &mr0u, &grid, bdir, &ip, |_, _| {});
        sv("chi1_from_mat_lfs_radps", &c1r.iter().map(|v| v * sti).collect::<Vec<_>>());
        eprintln!("ilsqr done {:.0}s", t.elapsed().as_secs_f64());
    }
    if want("offset") {
        let c1 = rd("chi1").data; let c2 = rd("chi2").data; let l1 = rd("lfs1_ppm").data;
        let rem: Vec<f64> = mr0.iter().zip(&vasc_m).map(|(a, v)| a - v).collect();
        let c = adjust_offset(&rem, &l1, &c1, &c2, &grid, bdir, 1.0); sv("chi_offset_from_mat_chis", &c);
    }
    if want("full") {
        let meta = ScanMetadata { dims, voxel_size: vs, echo_times: vec![0.0055], field_strength: b0, b0_direction: bdir, slice_geometry: None, affine: None };
        let cfg = InversionConfig { algorithm: InversionAlgorithm::Qsmart, qsmart: QsmartParams::for_field_strength(b0), ..Default::default() };
        // as faithful as run_qsmart's signature allows: mask = mask*R_0, N4 mean-echo magnitude
        let chi = run_qsmart(&tfs, &mr0u, None, Some(&avg), &meta, &cfg, QsmReference::None, &mut ()).unwrap(); sv("chi_full_maskR0", &chi);
        // as QSMxT wires it: brain mask only, RSS magnitude
        let mag = read4(&format!("{e}/data/magnitude.nii"));
        let ne = mag.len() / n; assert!(ne > 1);
        let rss: Vec<f64> = (0..n).map(|i| (0..ne).map(|k| mag[i + k * n].powi(2)).sum::<f64>().sqrt()).collect();
        let chi = run_qsmart(&tfs, &mask, None, Some(&rss), &meta, &cfg, QsmReference::None, &mut ()).unwrap(); sv("chi_full_qsmxt_wiring", &chi);
        eprintln!("full done {:.0}s", t.elapsed().as_secs_f64());
    }
    if want("mags") {
        let meta = ScanMetadata { dims, voxel_size: vs, echo_times: vec![0.0055], field_strength: b0, b0_direction: bdir, slice_geometry: None, affine: None };
        let cfg = InversionConfig { algorithm: InversionAlgorithm::Qsmart, qsmart: QsmartParams::for_field_strength(b0), ..Default::default() };
        let hp = HomogeneityParams::default();
        let mag = read4(&format!("{e}/data/magnitude.nii"));
        let ne = mag.len() / n; assert!(ne > 1);
        let mean: Vec<f64> = (0..n).map(|i| (0..ne).map(|k| mag[i + k * n]).sum::<f64>() / ne as f64).collect();
        let rss: Vec<f64> = (0..n).map(|i| (0..ne).map(|k| mag[i + k * n].powi(2)).sum::<f64>().sqrt()).collect();
        let vp = VasculatureParams::default();
        for (name, m) in [("n4mean", avg.clone()), ("mhmean", makehomogeneous(&mean, &grid, hp.sigma_mm, hp.nbox)), ("mhrss", makehomogeneous(&rss, &grid, hp.sigma_mm, hp.nbox))] {
            sv(&format!("mag_{name}"), &m);
            let v = generate_vasculature_mask(&m, &mask, &grid, &vp, |_, _| {}); sv(&format!("vasc_{name}"), &v);
            let chi = run_qsmart(&tfs, &mr0u, None, Some(&m), &meta, &cfg, QsmReference::None, &mut ()).unwrap(); sv(&format!("chi_{name}"), &chi);
            eprintln!("{name} done {:.0}s", t.elapsed().as_secs_f64());
        }
    }
    if want("unwrap") {
        let ph = read4(&format!("{e}/data/phase.nii"));
        let ne = ph.len() / n; assert!(ne > 1); let ones = vec![1u8; n];
        for k in 0..ne {
            let u = qsm_core::unwrap::laplacian_unwrap(&ph[k * n..(k + 1) * n], &ones, &grid, qsm_core::unwrap::LaplacianSolver::Mirror);
            sv(&format!("unph_mirror_e{}", k + 1), &u);
        }
    }
    if want("scaled") {
        // makehomogeneous(mean) rescaled so its in-mask median equals the N4 magnitude's
        let meta = ScanMetadata { dims, voxel_size: vs, echo_times: vec![0.0055], field_strength: b0, b0_direction: bdir, slice_geometry: None, affine: None };
        let cfg = InversionConfig { algorithm: InversionAlgorithm::Qsmart, qsmart: QsmartParams::for_field_strength(b0), ..Default::default() };
        let mh = read_nifti_file(Path::new(&format!("{e}/rs/mag_mhmean.nii"))).unwrap().data;
        let med = |v: &[f64]| { let mut x: Vec<f64> = v.iter().zip(&mask).filter(|(_, &m)| m != 0).map(|(&a, _)| a).collect(); x.sort_by(|a, b| a.partial_cmp(b).unwrap()); x[x.len() / 2] };
        let k = med(&avg) / med(&mh); eprintln!("scale {k}");
        let m: Vec<f64> = mh.iter().map(|v| v * k).collect();
        let v = generate_vasculature_mask(&m, &mask, &grid, &VasculatureParams::default(), |_, _| {}); sv("vasc_mhscaled", &v);
        let chi = run_qsmart(&tfs, &mr0u, None, Some(&m), &meta, &cfg, QsmReference::None, &mut ()).unwrap(); sv("chi_mhscaled", &chi);
    }
    if want("route") {
        use qsm_core::pipeline::{run_field_mapping, B0EstimationMethod, FieldMappingConfig, UnwrappingAlgorithm};
        let ph = read4(&format!("{e}/data/phase.nii")); let mg = read4(&format!("{e}/data/magnitude.nii"));
        let ne = ph.len() / n; assert!(ne > 1);
        let tes: Vec<f64> = p["TE"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let meta = ScanMetadata { dims, voxel_size: vs, echo_times: tes.clone(), field_strength: b0, b0_direction: bdir, slice_geometry: None, affine: None };
        let pr: Vec<&[f64]> = (0..ne).map(|k| &ph[k * n..(k + 1) * n]).collect();
        let mr: Vec<&[f64]> = (0..ne).map(|k| &mg[k * n..(k + 1) * n]).collect();
        let fm = FieldMappingConfig { unwrapping_algorithm: UnwrappingAlgorithm::Laplacian, laplacian_solver: qsm_core::unwrap::LaplacianSolver::Mirror,
            b0_estimation: B0EstimationMethod::QsmartEchofit { fit_threshold: 40.0 }, ..Default::default() };
        let f = run_field_mapping(&pr, Some(&mr), &mask, &meta, &fm, &mut |_, _| {}).unwrap();
        sv("route_tfs", &f.b0_field_ppm);
        let r0 = f.reliability.unwrap(); sv("route_R0", &r0.iter().map(|&v| v as f64).collect::<Vec<_>>());
        let uw: Vec<Vec<f64>> = pr.iter().map(|x| qsm_core::unwrap::laplacian_unwrap(x, &vec![1; n], &grid, qsm_core::unwrap::LaplacianSolver::Mirror)).collect();
        let uwr: Vec<&[f64]> = uw.iter().map(|v| v.as_slice()).collect();
        let ef = qsm_core::utils::qsmart_echofit(&uwr, &mr, &tes, &grid, 2.675e8 * b0 / 1e6, 40.0); sv("route_blur", &ef.fit_residual_blur);
        let cfg = InversionConfig { algorithm: InversionAlgorithm::Qsmart, qsmart: QsmartParams::for_field_strength(b0), ..Default::default() };
        let chi = run_qsmart(&f.b0_field_ppm, &mask, Some(&r0), Some(&avg), &meta, &cfg, QsmReference::None, &mut ()).unwrap(); sv("route_chi", &chi);
        if std::env::var("PURE").is_ok() {
            // QSMART's vessel magnitude made in QSM.rs too: mean over echoes, then N4 (no mask, as QSMART calls it)
            let mean: Vec<f64> = (0..n).map(|i| (0..ne).map(|k| mg[i + k * n]).sum::<f64>() / ne as f64).collect();
            let n4 = qsm_core::utils::n4_bias_correction(&mean, None, &grid, &qsm_core::utils::N4Params::default(), |_, _| {}).corrected;
            sv("pure_mag", &n4);
            let v = generate_vasculature_mask(&n4, &mask, &grid, &VasculatureParams::default(), |_, _| {}); sv("pure_vasc", &v);
            let chi = run_qsmart(&f.b0_field_ppm, &mask, Some(&r0), Some(&n4), &meta, &cfg, QsmReference::None, &mut ()).unwrap(); sv("pure_chi", &chi);
        }
        eprintln!("route done {:.0}s", t.elapsed().as_secs_f64());
    }
}

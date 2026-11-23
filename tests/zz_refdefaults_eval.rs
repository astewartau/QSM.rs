//! Untracked evaluation harness for the reference-defaults PR (not committed).
//! cargo test --release --features parallel --test zz_refdefaults_eval -- --ignored --nocapture
#![allow(dead_code)]
mod common;
use common::*;
use qsm_core::bgremove::{ResharpParams, VsharpGrid, VsharpParams};
use qsm_core::inversion::IlsqrParams;
use qsm_core::pipeline::*;
use std::time::Instant;

struct Case {
    name: String,
    phases: Vec<Vec<f64>>,
    mags: Vec<Vec<f64>>,
    mask: Vec<u8>,
    meta: ScanMetadata,
    /// ground truth (phantom) or reference reconstruction (in vivo)
    chi_ref: Vec<f64>,
    local_ref: Option<Vec<f64>>,
    truth: bool,
}

fn and(a: &[u8], b: &[u8]) -> Vec<u8> { a.iter().zip(b).map(|(x, y)| (*x > 0 && *y > 0) as u8).collect() }

fn metrics(tag: &str, x: &[f64], r: &[f64], m: &[u8], dims: (usize, usize, usize)) {
    let (nr, _) = nrmse_challenge(x, r, m);
    println!("    {:<34} XSIM {:.4}  NRMSE {:6.2} %  r {:.4}  (n={})", tag, xsim(x, r, m, dims), nr, correlation(x, r, m),
        m.iter().filter(|&&v| v > 0).count());
}

fn bg_cfg(alg: BgRemovalAlgorithm, f: impl FnOnce(&mut BgRemovalConfig)) -> BgRemovalConfig {
    let mut c = BgRemovalConfig { algorithm: alg, ..Default::default() };
    f(&mut c);
    c
}

fn run_case(c: &Case) {
    println!("\n==================== {} ====================", c.name);
    let dims = c.meta.dims;
    let ph: Vec<&[f64]> = c.phases.iter().map(|v| v.as_slice()).collect();
    let mg: Vec<&[f64]> = c.mags.iter().map(|v| v.as_slice()).collect();
    let t = Instant::now();
    let field = run_field_mapping(&ph, Some(&mg), &c.mask, &c.meta, &FieldMappingConfig::default(), &mut |_, _| {})
        .expect("field mapping");
    println!("field mapping {:.1?}", t.elapsed());
    let mag_last = c.mags.last().unwrap();
    let vs = c.meta.voxel_size;

    // MCPC-3D-S sigma: old [10,10,5] voxels vs new ROMEO CLI 7 mm, through the default pipeline
    println!("\n--- MCPC-3D-S sigma (field map, then default V-SHARP + RTS) ---");
    let old_fm = FieldMappingConfig { phase_offset_sigma_mm: [10.0 * vs.0, 10.0 * vs.1, 5.0 * vs.2], ..Default::default() };
    for (fname, fm) in [("old: [10,10,5] voxels", old_fm), ("new: 7 mm", FieldMappingConfig::default())] {
        let f = run_field_mapping(&ph, Some(&mg), &c.mask, &c.meta, &fm, &mut |_, _| {}).expect("fm");
        let bg = run_bg_removal(&f.b0_field_ppm, &c.mask, &c.meta, &BgRemovalConfig::default(), None, &mut |_, _| {}).expect("bg");
        let chi = run_dipole_inversion(&bg.local_field_ppm, &bg.eroded_mask, &c.meta, &InversionConfig::default(), Some(mag_last), &mut |_, _| {}).expect("rts");
        let chi = apply_reference(&chi, &bg.eroded_mask, QsmReference::Mean);
        println!("  {}", fname);
        if c.truth {
            let tf: Vec<f64> = c.chi_ref.iter().map(|_| 0.0).collect();
            let _ = tf;
        }
        metrics("chi RTS", &chi, &c.chi_ref, &and(&bg.eroded_mask, &c.mask), dims);
    }

    let old_vsharp = VsharpParams { deconvolve: true, threshold: 0.2, grid: VsharpGrid::Native, min_radius: 1.0, radius_step: 1.0, ..VsharpParams::default() };
    let new_vsharp = VsharpParams::default();
    let groups: Vec<(&str, Vec<(&str, BgRemovalConfig)>)> = vec![
        ("V-SHARP", vec![
            ("old: deconv, native", bg_cfg(BgRemovalAlgorithm::Vsharp, |c| c.vsharp = old_vsharp.clone())),
            ("no deconv, native, 12->1 mm", bg_cfg(BgRemovalAlgorithm::Vsharp, |c| c.vsharp = VsharpParams { grid: VsharpGrid::Native, min_radius: 1.0, ..new_vsharp.clone() })),
            ("no deconv, 1 mm grid, 12->1 mm", bg_cfg(BgRemovalAlgorithm::Vsharp, |c| c.vsharp = VsharpParams { min_radius: 1.0, ..new_vsharp.clone() })),
            ("new: no deconv, 1 mm, 12->2 mm", bg_cfg(BgRemovalAlgorithm::Vsharp, |c| c.vsharp = new_vsharp.clone())),
        ]),
        ("RESHARP", vec![
            ("old: 6 mm, 30 it", bg_cfg(BgRemovalAlgorithm::Resharp, |c| c.resharp = ResharpParams { radius: 6.0, max_iter: 30, ..ResharpParams::default() })),
            ("3 mm, 30 it", bg_cfg(BgRemovalAlgorithm::Resharp, |c| c.resharp = ResharpParams { max_iter: 30, ..ResharpParams::default() })),
            ("6 mm, 200 it", bg_cfg(BgRemovalAlgorithm::Resharp, |c| c.resharp = ResharpParams { radius: 6.0, ..ResharpParams::default() })),
            ("new: 3 mm, 200 it", bg_cfg(BgRemovalAlgorithm::Resharp, |c| c.resharp = ResharpParams::default())),
        ]),
    ];
    let mut vsharp_locals: Vec<(String, Vec<f64>, Vec<u8>)> = Vec::new();
    for (gname, variants) in &groups {
        println!("\n--- {} (local field, then RTS) ---", gname);
        let mut outs = Vec::new();
        for (vname, cfg) in variants {
            let t = Instant::now();
            let bg = run_bg_removal(&field.b0_field_ppm, &c.mask, &c.meta, cfg, None, &mut |_, _| {}).expect("bg");
            let tb = t.elapsed();
            let chi = run_dipole_inversion(&bg.local_field_ppm, &bg.eroded_mask, &c.meta, &InversionConfig::default(), Some(mag_last), &mut |_, _| {}).expect("rts");
            let chi = apply_reference(&chi, &bg.eroded_mask, QsmReference::Mean);
            println!("  {:<28} bg {:.1?}, mask {}", vname, tb, bg.eroded_mask.iter().filter(|&&v| v > 0).count());
            if *gname == "V-SHARP" { vsharp_locals.push((vname.to_string(), bg.local_field_ppm.clone(), bg.eroded_mask.clone())); }
            outs.push((vname, bg, chi));
        }
        let mut common_m = c.mask.clone();
        for (_, bg, _) in &outs { common_m = and(&common_m, &bg.eroded_mask); }
        for (vname, bg, chi) in &outs {
            println!("  {}", vname);
            if let Some(lr) = &c.local_ref {
                metrics("local field (common mask)", &bg.local_field_ppm, lr, &common_m, dims);
            }
            metrics("chi RTS (common mask)", chi, &c.chi_ref, &common_m, dims);
            metrics("chi RTS (own mask)", chi, &c.chi_ref, &and(&bg.eroded_mask, &c.mask), dims);
        }
    }

    // L1-QSM lambda 1.0 vs 0.8
    println!("\n--- L1-QSM lambda (old 1.0 vs new 0.8) ---");
    {
        let mut ins: Vec<(String, Vec<f64>, Vec<u8>)> = vsharp_locals.iter().filter(|(n, _, _)| n.starts_with("new")).cloned().collect();
        if c.truth { if let Some(lr) = &c.local_ref { ins.push(("true local field".into(), lr.clone(), c.mask.clone())); } }
        for (iname, local, m) in &ins {
            println!("  input: {}", iname);
            for lam in [1.0, 0.8] {
                let mut cfg = InversionConfig { algorithm: InversionAlgorithm::L1qsm, ..Default::default() };
                cfg.l1qsm.lambda = lam;
                let chi = run_dipole_inversion(local, m, &c.meta, &cfg, Some(mag_last), &mut |_, _| {}).expect("l1");
                let chi = apply_reference(&chi, m, QsmReference::Mean);
                metrics(&format!("lambda {lam}"), &chi, &c.chi_ref, &and(m, &c.mask), dims);
            }
        }
    }

    // iLSQR: old function defaults vs GUI defaults, on V-SHARP old/new local fields (+ GT local)
    println!("\n--- iLSQR (old 100 it / 6 mm vs new 40 it / 12 mm) ---");
    let mut inputs: Vec<(String, Vec<f64>, Vec<u8>)> = vsharp_locals.into_iter()
        .filter(|(n, _, _)| n.starts_with("old") || n.starts_with("new")).collect();
    if c.truth { if let Some(lr) = &c.local_ref { inputs.push(("true local field".into(), lr.clone(), c.mask.clone())); } }
    for (iname, local, m) in &inputs {
        println!("  input: V-SHARP {}", iname);
        for (pname, p) in [("old: 100 it, 6 mm", IlsqrParams { max_iter: 100, pad_mm: 6.0, ..IlsqrParams::default() }),
                           ("40 it, 6 mm", IlsqrParams { pad_mm: 6.0, ..IlsqrParams::default() }),
                           ("100 it, 12 mm", IlsqrParams { max_iter: 100, ..IlsqrParams::default() }),
                           ("new: 40 it, 12 mm", IlsqrParams::default())] {
            let cfg = InversionConfig { algorithm: InversionAlgorithm::Ilsqr, ilsqr: p, ..Default::default() };
            let t = Instant::now();
            let chi = run_dipole_inversion(local, m, &c.meta, &cfg, Some(mag_last), &mut |_, _| {}).expect("ilsqr");
            let chi = apply_reference(&chi, m, QsmReference::Mean);
            println!("    [{:.1?}]", t.elapsed());
            metrics(pname, &chi, &c.chi_ref, &and(m, &c.mask), dims);
        }
    }
}

#[test]
#[ignore]
fn eval_phantom() {
    let d = TestData::load().expect("phantom");
    let meta = ScanMetadata { dims: d.dims, voxel_size: d.voxel_size, echo_times: d.echo_times.clone(),
        field_strength: d.field_strength, b0_direction: d.b0_dir, slice_geometry: None, affine: None };
    println!("phantom dims {:?} vs {:?} TE {:?} B0 {}", d.dims, d.voxel_size, d.echo_times, d.field_strength);
    run_case(&Case { name: "qsm-forward phantom".into(), phases: d.phase_echoes, mags: d.mag_echoes, mask: d.mask,
        meta, chi_ref: d.chi, local_ref: Some(d.fieldmap_local), truth: true });
}

fn rescale_phase(v: &mut [f64]) {
    let (mn, mx) = v.iter().fold((f64::MAX, f64::MIN), |(a, b), &x| (a.min(x), b.max(x)));
    if mx - mn > 2.0 * std::f64::consts::PI + 0.1 {
        for x in v.iter_mut() { *x = (*x - mn) / (mx - mn) * 2.0 * std::f64::consts::PI - std::f64::consts::PI; }
    }
}

#[test]
#[ignore]
fn eval_invivo() {
    let acq = std::env::var("ACQ").unwrap_or("s002001-kki-achieva-bridge-run1".into());
    let d = format!("/scratch/user/uqaste15/src/QSM-CI/data/harmonization/{acq}/inputs");
    let rd4 = |p: String| qsm_core::io::load_nifti_4d(&std::fs::read(&p).expect(&p)).expect(&p);
    let (ph, (nx, ny, nz, ne), vs, affine) = rd4(format!("{d}/phase.nii.gz"));
    let (mg, _, _, _) = rd4(format!("{d}/magnitude.nii.gz"));
    let n = nx * ny * nz;
    let mut phases: Vec<Vec<f64>> = (0..ne).map(|e| ph[e * n..(e + 1) * n].to_vec()).collect();
    for p in phases.iter_mut() { rescale_phase(p); }
    let mags: Vec<Vec<f64>> = (0..ne).map(|e| mg[e * n..(e + 1) * n].to_vec()).collect();
    let rd = |p: String| qsm_core::io::read_nifti_file(std::path::Path::new(&p)).expect(&p);
    let mask: Vec<u8> = rd(format!("{d}/mask.nii.gz")).data.iter().map(|&v| (v > 0.5) as u8).collect();
    let chi_ref = rd(format!("/scratch/user/uqaste15/qsmci-repro/results/ukb-qsm-cmp-{acq}/recon.nii.gz")).data;
    let js: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{d}/params.json")).unwrap()).unwrap();
    let tes: Vec<f64> = js["TE"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let bd: Vec<f64> = js["B0_dir"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let meta = ScanMetadata { dims: (nx, ny, nz), voxel_size: vs, echo_times: tes.clone(), field_strength: js["B0"].as_f64().unwrap(),
        b0_direction: (bd[0], bd[1], bd[2]), slice_geometry: None, affine: Some(affine) };
    println!("in vivo {acq} dims {:?} vs {:?} TE {:?} bdir {:?}", meta.dims, vs, tes, meta.b0_direction);
    run_case(&Case { name: format!("in vivo {acq} (reference: UK Biobank pipeline, STI V_SHARP + iLSQR)"), phases, mags, mask, meta,
        chi_ref, local_ref: None, truth: false });
}

/// χ-sep iLSQR with its QSM from iLSQR at main's default (100 it / 6 mm), the GUI's (40 / 12)
/// and the toolbox's call (100 / 12, the new `qsm_ilsqr`).
#[test]
#[ignore]
fn eval_chisep() {
    use qsm_core::inversion::ilsqr;
    use qsm_core::separation::{chi_sep_ilsqr, ChiSepIlsqrParams};
    let ph = load_chisep_phantom().expect("chisep phantom");
    for (name, p) in [("main: 100 it, 6 mm", IlsqrParams { max_iter: 100, pad_mm: 6.0, ..IlsqrParams::default() }),
                      ("GUI: 40 it, 12 mm", IlsqrParams::default()),
                      ("new qsm_ilsqr: 100 it, 12 mm", ChiSepIlsqrParams::default().qsm_ilsqr)] {
        println!("\n--- chi-sep iLSQR, QSM from iLSQR {} ---", name);
        let (qsm, _, _, _) = ilsqr(&ph.local_field_ppm, &ph.mask, &ph.grid, ph.bdir, &p, |_, _| {});
        let params = ChiSepIlsqrParams { cf: ph.cf, ..ChiSepIlsqrParams::default() };
        let t = Instant::now();
        let (pos, neg, _) = chi_sep_ilsqr(&ph.local_field_ppm, &ph.r2prime, &ph.magnitude_rss, &qsm, &ph.mask, &ph.grid, ph.bdir, &params, |_, _| {});
        let secs = t.elapsed().as_secs_f64();
        let dia: Vec<f64> = neg.iter().map(|&v| -v).collect();
        chisep_score("chi+", &pos, &ph.gt_para, &ph.mask, ph.dims, secs);
        chisep_score("chi-", &dia, &ph.gt_dia, &ph.mask, ph.dims, secs);
    }
}

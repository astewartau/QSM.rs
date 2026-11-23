//! Scratch (exp/vsharp-ablation): V-SHARP ingredient ablation between the published V-SHARP
//! (origin/main) and STI Suite 3.0's V_SHARP. Not for merging.
//!
//! cargo test --release --features parallel --test vsharp_ablation -- --ignored --nocapture
//! env: ABL_OUT (tsv path), ABL_SPECS (comma list; default = full factorial + extras),
//!      ABL_ILSQR (comma list of specs that also get iLSQR), ABL_REAL_DIR (probe dir),
//!      ABL_INPUT (truth = ground-truth total field [default]; romeo = field estimated from the
//!      simulated echoes with the CI's ROMEO pipeline), ABL_ZBIN (k: average k slices in z, e.g. 3
//!      gives a 1 x 1 x 3 mm phantom like UKB's 0.8 x 0.8 x 3 mm; masks need all k inside)

mod common;

use common::{correlation, nrmse_challenge, xsim, ChallengeMetrics, TestData};
use qsm_core::bgremove::vsharp::ablation::{run, Opts};
use qsm_core::bgremove::{vsharp, VsharpParams};
use qsm_core::inversion::{ilsqr, tkd, IlsqrParams, TkdParams};
use qsm_core::io::read_nifti_file;
use qsm_core::unwrap::romeo::{unwrap_romeo_multi_echo, RomeoParams};
use qsm_core::unwrap::UnwrapMethod;
use qsm_core::utils::{calculate_b0_weighted, phase_offset_removal, B0WeightType};
use qsm_core::Grid;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

fn factorial() -> Vec<String> {
    let mut v = Vec::new();
    for bits in 0..64u32 {
        let b = |k: u32| (bits >> k) & 1;
        v.push(format!("d{}g{}b{}r{}p{}e{}", b(5), b(4), b(3), b(2), b(1), b(0)));
    }
    // the two halves of the STI ball, from each end
    for x in ["q1", "c1"] { v.push(format!("d1g0b0r0p0e0{x}")); v.push(format!("d0g1b0r1p1e1{x}")); }
    for t in ["t05", "t1", "t3"] {
        v.push(format!("d1g0b0r0p0e0{t}"));
        v.push(format!("d1g1b1r1p1e1{t}"));
    }
    v
}

struct Real {
    field: Vec<f64>,
    mask: Vec<u8>,
    grid: Grid,
    sti_local: Vec<f64>,
    sti_mask: Vec<u8>,
}

fn load_real(dir: &str) -> Option<Real> {
    let f = read_nifti_file(Path::new(&format!("{dir}/real_phs.nii.gz"))).ok()?;
    let m = read_nifti_file(Path::new(&format!("{dir}/real_mask.nii.gz"))).ok()?;
    let sl = read_nifti_file(Path::new(&format!("{dir}/sti_smv12_local.nii"))).ok()?;
    let sm = read_nifti_file(Path::new(&format!("{dir}/sti_smv12_mask.nii"))).ok()?;
    let grid = Grid::new(f.dims.0, f.dims.1, f.dims.2, f.voxel_size.0, f.voxel_size.1, f.voxel_size.2);
    Some(Real {
        mask: m.data.iter().map(|&v| (v > 0.5) as u8).collect(),
        field: f.data,
        grid,
        sti_local: sl.data,
        sti_mask: sm.data.iter().map(|&v| (v > 0.5) as u8).collect(),
    })
}

/// (r over both masks, 99.9th pct |Δ|, rms Δ / rms STI, mask voxels only in ours, only in STI)
fn vs_sti(r: &Real, local: &[f64], mask: &[u8]) -> (f64, f64, f64, usize, usize) {
    let both: Vec<u8> = mask.iter().zip(&r.sti_mask).map(|(&a, &b)| (a != 0 && b != 0) as u8).collect();
    let corr = correlation(local, &r.sti_local, &both);
    let mut d: Vec<f64> = (0..local.len()).filter(|&i| both[i] != 0).map(|i| (local[i] - r.sti_local[i]).abs()).collect();
    d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p999 = if d.is_empty() { f64::NAN } else { d[((d.len() as f64 * 0.999) as usize).min(d.len() - 1)] };
    let rms_d = (d.iter().map(|v| v * v).sum::<f64>() / d.len().max(1) as f64).sqrt();
    let rms_s = ((0..local.len()).filter(|&i| both[i] != 0).map(|i| r.sti_local[i].powi(2)).sum::<f64>() / d.len().max(1) as f64).sqrt();
    let ours = mask.iter().zip(&r.sti_mask).filter(|(&a, &b)| a != 0 && b == 0).count();
    let theirs = mask.iter().zip(&r.sti_mask).filter(|(&a, &b)| a == 0 && b != 0).count();
    (corr, p999, rms_d / rms_s, ours, theirs)
}

#[test]
#[ignore]
fn vsharp_ablation() {
    let out = std::env::var("ABL_OUT").unwrap_or_else(|_| "vsharp_ablation.tsv".into());
    let specs: Vec<String> = match std::env::var("ABL_SPECS") {
        Ok(s) if !s.is_empty() => s.split(',').map(String::from).collect(),
        _ => factorial(),
    };
    let ilsqr_specs: Vec<String> = std::env::var("ABL_ILSQR").unwrap_or_default().split(',').filter(|s| !s.is_empty()).map(String::from).collect();
    let real_dir = std::env::var("ABL_REAL_DIR").unwrap_or_else(|_| "/scratch/user/uqaste15/vsharp_probe".into());

    let mut data = TestData::load().expect("test data");
    let input = std::env::var("ABL_INPUT").unwrap_or_else(|_| "truth".into());
    if input == "romeo" {
        let (nx, ny, nz) = data.dims;
        let (vx, vy, vz) = data.voxel_size;
        let g = Grid::new(nx, ny, nz, vx, vy, vz);
        let (ph, _) = phase_offset_removal(&data.phase_echoes, &data.mag_echoes, &data.echo_times, &data.mask,
            [10.0, 10.0, 5.0], [0, 1], UnwrapMethod::Romeo, &g);
        let mags: Vec<&[f64]> = data.mag_echoes.iter().map(|v| v.as_slice()).collect();
        let uw = unwrap_romeo_multi_echo(&ph, &mags, &data.echo_times, &data.mask, &RomeoParams::default(), &g);
        let hz = calculate_b0_weighted(&uw, &data.mag_echoes, &data.echo_times, &data.mask, B0WeightType::PhaseSNR, &g);
        let scale = 1e6 / (42.576e6 * data.field_strength);
        data.fieldmap = hz.iter().zip(&data.mask).map(|(&v, &m)| if m != 0 { v * scale } else { 0.0 }).collect();
    } else {
        assert_eq!(input, "truth", "ABL_INPUT must be truth or romeo");
    }
    let zbin: usize = std::env::var("ABL_ZBIN").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    if zbin > 1 {
        let (nx, ny, nz) = data.dims;
        let nzb = nz / zbin;
        let avg = |v: &[f64]| -> Vec<f64> {
            let mut o = vec![0.0; nx * ny * nzb];
            for k in 0..nzb { for q in 0..zbin { for j in 0..ny { for i in 0..nx {
                o[i + j * nx + k * nx * ny] += v[i + j * nx + (k * zbin + q) * nx * ny] / zbin as f64;
            }}}}
            o
        };
        let all = |v: &[u8]| -> Vec<u8> {
            (0..nx * ny * nzb).map(|d| {
                let (i, j, k) = (d % nx, (d / nx) % ny, d / (nx * ny));
                (0..zbin).all(|q| v[i + j * nx + (k * zbin + q) * nx * ny] != 0) as u8
            }).collect()
        };
        let centre = |v: &[u8]| -> Vec<u8> {
            (0..nx * ny * nzb).map(|d| {
                let (i, j, k) = (d % nx, (d / nx) % ny, d / (nx * ny));
                v[i + j * nx + (k * zbin + zbin / 2) * nx * ny]
            }).collect()
        };
        data.fieldmap = avg(&data.fieldmap);
        data.fieldmap_local = avg(&data.fieldmap_local);
        data.chi = avg(&data.chi);
        data.mask = all(&data.mask);
        data.segmentation = centre(&data.segmentation);
        data.dims = (nx, ny, nzb);
        data.voxel_size.2 *= zbin as f64;
    }
    println!("input {input}, zbin {zbin}");
    let (nx, ny, nz) = data.dims;
    let (vx, vy, vz) = data.voxel_size;
    let grid = Grid::new(nx, ny, nz, vx, vy, vz);
    let real = if std::env::var("ABL_NO_REAL").is_ok() { None } else { load_real(&real_dir) };
    println!("phantom {:?} {:?}; real field {}", data.dims, data.voxel_size, if real.is_some() { "loaded" } else { "MISSING" });

    // common interior: where every variant has output (most eroded: strict fit, min radius 2)
    let mut inner = data.mask.clone();
    for spec in ["d0g0b0r1p0e1", "d0g0b1r1p0e1", "d0g1b0r1p1e1", "d0g1b1r1p1e1"] {
        let (_, m) = run(&data.fieldmap, &data.mask, &grid, Opts::from_spec(spec).unwrap());
        for (a, b) in inner.iter_mut().zip(&m) { *a &= *b; }
    }
    println!("inner mask {} of {}", inner.iter().filter(|&&v| v != 0).count(), data.mask.iter().filter(|&&v| v != 0).count());

    let mut f = std::fs::File::create(&out).unwrap();
    writeln!(f, "spec\tmask_vox\tnrmse\tr\txsim\tnrmse_inner\tr_inner\txsim_inner\ttkd_nrmse\ttkd_r\ttkd_xsim\tilsqr_nrmse\tilsqr_r\tilsqr_xsim\ttkd_nrmse_in\ttkd_r_in\tilsqr_nrmse_in\tilsqr_r_in\tilsqr_xsim_in\tilsqr_dgm_in\tilsqr_dgmlin_in\treal_r_sti\treal_p999\treal_relrms\treal_only_ours\treal_only_sti\tsecs").unwrap();

    // reference implementations: origin/main's code (default) and 93b15a1's STI path
    let refs: Vec<(String, Box<dyn Fn(&[f64], &[u8], &Grid) -> (Vec<f64>, Vec<u8>)>)> = vec![
        ("REF-main".into(), Box::new(|fl: &[f64], m: &[u8], g: &Grid| vsharp(fl, m, g, &VsharpParams::default(), |_, _| {}))),
        ("REF-sti".into(), Box::new(|fl: &[f64], m: &[u8], g: &Grid| vsharp(fl, m, g, &VsharpParams { deconvolve: false, ..Default::default() }, |_, _| {}))),
    ];
    let mut ref_out: Vec<(Vec<f64>, Vec<u8>, Option<(Vec<f64>, Vec<u8>)>)> = Vec::new();

    let mut rows: Vec<(String, Box<dyn Fn(&[f64], &[u8], &Grid) -> (Vec<f64>, Vec<u8>)>)> = refs;
    for s in &specs {
        let o = Opts::from_spec(s).unwrap_or_else(|e| panic!("{e}"));
        rows.push((s.clone(), Box::new(move |fl: &[f64], m: &[u8], g: &Grid| run(fl, m, g, o))));
    }

    for (name, fun) in &rows {
        let t0 = Instant::now();
        let (local, m) = fun(&data.fieldmap, &data.mask, &grid);
        let nr = nrmse_challenge(&local, &data.fieldmap_local, &data.mask).0;
        let r = correlation(&local, &data.fieldmap_local, &data.mask);
        let xs = xsim(&local, &data.fieldmap_local, &data.mask, data.dims);
        let nr_i = nrmse_challenge(&local, &data.fieldmap_local, &inner).0;
        let r_i = correlation(&local, &data.fieldmap_local, &inner);
        let xs_i = xsim(&local, &data.fieldmap_local, &inner, data.dims);
        let chi = tkd(&local, &m, &grid, data.b0_dir, &TkdParams { threshold: 0.2 });
        let ct = ChallengeMetrics::compute("TKD", &chi, &data.chi, &m, &data.segmentation, data.dims);
        let inm: Vec<u8> = m.iter().zip(&inner).map(|(&a, &b)| a & b).collect();
        let cti = ChallengeMetrics::compute("TKD", &chi, &data.chi, &inm, &data.segmentation, data.dims);
        let il = if ilsqr_specs.iter().any(|s| s == name || s == "all") || name.starts_with("REF") {
            let chi = ilsqr(&local, &m, &grid, data.b0_dir, &IlsqrParams::default(), |_, _| {}).0;
            let c = ChallengeMetrics::compute("iLSQR", &chi, &data.chi, &m, &data.segmentation, data.dims);
            let ci = ChallengeMetrics::compute("iLSQR", &chi, &data.chi, &inm, &data.segmentation, data.dims);
            (c.nrmse, c.correlation, c.xsim, ci.nrmse, ci.correlation, ci.xsim, ci.nrmse_dgm, ci.dgm_linearity)
        } else {
            (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN)
        };
        let real_out = real.as_ref().map(|re| fun(&re.field, &re.mask, &re.grid));
        let rv = match (&real, &real_out) {
            (Some(re), Some((l, rm))) => vs_sti(re, l, rm),
            _ => (f64::NAN, f64::NAN, f64::NAN, 0, 0),
        };
        let secs = t0.elapsed().as_secs_f64();
        let mv = m.iter().filter(|&&v| v != 0).count();
        let line = format!(
            "{name}\t{mv}\t{nr:.3}\t{r:.5}\t{xs:.4}\t{nr_i:.3}\t{r_i:.5}\t{xs_i:.4}\t{:.3}\t{:.5}\t{:.4}\t{:.3}\t{:.5}\t{:.4}\t{:.3}\t{:.5}\t{:.3}\t{:.5}\t{:.4}\t{:.3}\t{:.4}\t{:.8}\t{:.3e}\t{:.3e}\t{}\t{}\t{secs:.1}",
            ct.nrmse, ct.correlation, ct.xsim, il.0, il.1, il.2, cti.nrmse, cti.correlation, il.3, il.4, il.5, il.6, il.7, rv.0, rv.1, rv.2, rv.3, rv.4
        );
        println!("{line}");
        writeln!(f, "{line}").unwrap();
        f.flush().unwrap();

        // prototype vs reference: does the prototype reproduce main / STI?
        if name.starts_with("REF") {
            ref_out.push((local, m, real_out));
        } else if name == "d1g0b0r0p0e0" || name == "d0g1b1r1p1e1" {
            let k = if name == "d1g0b0r0p0e0" { 0 } else { 1 };
            let (rl, rm, rr) = &ref_out[k];
            let dmax = local.iter().zip(rl).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            let mdiff = m.iter().zip(rm).filter(|(a, b)| a != b).count();
            let (rdmax, rmdiff) = match (&real_out, rr) {
                (Some((l1, m1)), Some((l2, m2))) => (
                    l1.iter().zip(l2).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max),
                    m1.iter().zip(m2).filter(|(a, b)| a != b).count(),
                ),
                _ => (f64::NAN, 0),
            };
            let msg = format!("# CHECK {name} vs {}: phantom max|d| {dmax:.3e} mask diff {mdiff}; real max|d| {rdmax:.3e} mask diff {rmdiff}", if k == 0 { "REF-main" } else { "REF-sti" });
            println!("{msg}");
            writeln!(f, "{msg}").unwrap();
        }
    }
}

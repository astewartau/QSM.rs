//! Run `chi_sep_medi` or `chi_sep_ilsqr` on NIfTI inputs, for parity checks against the SNU
//! χ-separation toolbox (QSM-CI's `chisep-medi` and `chisep-ilsqr` containers, licensed MATLAB).
//!
//! usage: chi_sep_file <medi|ilsqr> <local_field_ppm> <r2prime_hz> <qsm_ppm> <mask> <B0_T>
//!                     <bx,by,bz> <out_prefix> [mag1,mag2,... (medi)] [csf_mask (medi)]
//!
//! `CHISEP_GAMMA` (Hz/T, default 42.5774e6, the QSM-CI wrappers' value) sets `CF = γ·B0`. For
//! MEDI, `N_std` is MEDI `Fit_ppm_complex`'s `dp1` from the magnitude echoes
//! ([`qsm_core::utils::fit_ppm_complex_noise`]), zero outside the mask, the magnitude the
//! root-sum-of-squares over echoes, and the CSF mask all zero (the toolbox GUI's `mask_CSF = []`)
//! unless one is given.
//!
//! Writes `<out_prefix>_chipos.f64`, `_chidia.f64` (|χ−|) and, for MEDI, `_nstd.f64`, raw
//! little-endian f64 in NIfTI voxel order.
use qsm_core::io::read_nifti_file;
use qsm_core::separation::{chi_sep_ilsqr, chi_sep_medi, ChiSepIlsqrInputs, ChiSepIlsqrParams, ChiSepMediInputs, ChiSepMediParams};
use qsm_core::Grid;
use std::path::Path;

fn write(path: String, v: &[f64]) {
    std::fs::write(path, v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).expect("write");
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 9 {
        eprintln!("usage: {} <medi|ilsqr> <field> <r2p> <qsm> <mask> <B0> <bx,by,bz> <out> [mags] [csf]", a[0]);
        std::process::exit(2);
    }
    let rd = |p: &str| read_nifti_file(Path::new(p)).expect("read");
    let (field, r2p, qsm, m) = (rd(&a[2]), rd(&a[3]), rd(&a[4]), rd(&a[5]));
    let mask: Vec<u8> = m.data.iter().map(|&v| u8::from(v > 0.5)).collect();
    let b0: f64 = a[6].parse().unwrap();
    let h: Vec<f64> = a[7].split(',').map(|v| v.parse().unwrap()).collect();
    let gamma: f64 = std::env::var("CHISEP_GAMMA").map(|v| v.parse().unwrap()).unwrap_or(42.5774e6);
    let cf = gamma * b0;
    let (nx, ny, nz) = field.dims;
    let (vx, vy, vz) = field.voxel_size;
    let grid = Grid::new(nx, ny, nz, vx, vy, vz);
    let out = &a[8];
    let t = std::time::Instant::now();
    let (pos, neg, _) = match a[1].as_str() {
        "medi" => {
            let mags: Vec<_> = a[9].split(',').map(rd).collect();
            let n = field.data.len();
            let echo_major: Vec<f64> = mags.iter().flat_map(|v| v.data.iter().copied()).collect();
            let mut n_std = qsm_core::utils::fit_ppm_complex_noise(&echo_major, mags.len());
            for (v, &k) in n_std.iter_mut().zip(&mask) {
                if k == 0 {
                    *v = 0.0;
                }
            }
            write(format!("{out}_nstd.f64"), &n_std);
            let rss: Vec<f64> = (0..n).map(|i| mags.iter().map(|v| v.data[i] * v.data[i]).sum::<f64>().sqrt()).collect();
            let csf: Vec<u8> = match a.get(10) {
                Some(p) => rd(p).data.iter().map(|&v| u8::from(v > 0.5)).collect(),
                None => vec![0; n],
            };
            let inputs = ChiSepMediInputs {
                magnitude: &rss, local_field: &field.data, r2prime: &r2p.data, n_std: &n_std, qsm: &qsm.data,
                mask: &mask, mask_csf: &csf,
            };
            let params = ChiSepMediParams { cf, ..ChiSepMediParams::default() };
            chi_sep_medi(&inputs, &grid, (h[0], h[1], h[2]), &params, |i, n| eprintln!("medi {i}/{n}")).expect("medi")
        }
        "ilsqr" => {
            let inputs = ChiSepIlsqrInputs { local_field: &field.data, r2prime: &r2p.data, qsm: &qsm.data, mask: &mask };
            let params = ChiSepIlsqrParams { cf, ..ChiSepIlsqrParams::default() };
            chi_sep_ilsqr(&inputs, &grid, (h[0], h[1], h[2]), &params, |i, n| eprintln!("ilsqr {i}/{n}")).expect("ilsqr")
        }
        m => panic!("unknown method {m}"),
    };
    eprintln!("{}: {:.1}s", a[1], t.elapsed().as_secs_f64());
    write(format!("{out}_chipos.f64"), &pos);
    write(format!("{out}_chidia.f64"), &neg.iter().map(|v| -v).collect::<Vec<_>>());
}

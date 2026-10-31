//! `unring::gibbs_unring_3d` (`GibbsMode::Bautista3d`, the default) against MRtrix3's
//! `mrdegibbs -dimensionality 3`, the authors' code of Bautista et al.'s 3-D subvoxel-shift
//! unringing (Proc. ISMRM 2021, 29, 3535).
//!
//! `tests/data/gibbs_mrtrix3d_small.bin` is made by `tests/data/gibbs_mrtrix3d_small.py`, which
//! builds `mrdegibbs` from MRtrix3 `dev` at a pinned commit in a pinned container, so it can be
//! regenerated in Actions (`.github/workflows/regenerate-fixtures.yml`): two phantoms ringing
//! along all three axes (24×24×16, and 21×26×15 for odd axes), and `mrdegibbs`'s float64 output
//! at its defaults.

use num_complex::Complex64;
use qsm_core::unring::{gibbs_unring_3d, gibbs_unring_3d_complex, MrtrixUnringParams};

const CASES: [(usize, usize, usize); 2] = [(24, 24, 16), (21, 26, 15)];

#[test]
fn gibbs_unring_3d_against_mrdegibbs() {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/gibbs_mrtrix3d_small.bin")).expect("fixture");
    let mut off = 0;
    let mut read = |n: usize| -> Vec<f64> {
        let v = (0..n).map(|i| f64::from_le_bytes(raw[off + 8 * i..off + 8 * i + 8].try_into().unwrap())).collect();
        off += 8 * n;
        v
    };
    for dims in CASES {
        let n = dims.0 * dims.1 * dims.2;
        let (input, mrtrix) = (read(n), read(n));
        let c: Vec<Complex64> = input.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        let out: Vec<f64> = gibbs_unring_3d_complex(&c, dims, MrtrixUnringParams::default()).iter().map(|v| v.re).collect();
        let norm = |v: &mut dyn Iterator<Item = f64>| v.map(|x| x * x).sum::<f64>().sqrt();
        let nrmse = norm(&mut out.iter().zip(&mrtrix).map(|(a, b)| a - b)) / norm(&mut mrtrix.iter().copied());
        let max = out.iter().zip(&mrtrix).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let change = norm(&mut input.iter().zip(&mrtrix).map(|(a, b)| a - b)) / norm(&mut input.iter().copied());
        println!("{dims:?}: vs mrdegibbs NRMSE {nrmse:.2e}, max |Δ| {max:.2e}; mrdegibbs changes the input by {:.2} %", 100.0 * change);
        // Measured 2026-10-31: ~1e-16 (FFT rounding); mrdegibbs changes the input by a few %.
        assert!(nrmse < 1e-12, "NRMSE {nrmse}");
        // The real-input wrapper is the real part with negatives set to 0.
        let real = gibbs_unring_3d(&input, dims, MrtrixUnringParams::default());
        assert!(real.iter().zip(&mrtrix).all(|(a, b)| (a - b.max(0.0)).abs() < 1e-9));
    }
}

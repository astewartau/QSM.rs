//! `unring::gibbs_unring_2d` against MRtrix3's `mrdegibbs` in its default 2-D mode, Kellner et
//! al.'s slice-wise subvoxel-shift unringing (MRM 2016, 76, 1574).
//!
//! `tests/data/gibbs_mrtrix2d_small.bin` is made by `tests/data/gibbs_mrtrix2d_small.py`, which
//! builds `mrdegibbs` from MRtrix3 `dev` at a pinned commit in a pinned container (as
//! `tests/data/gibbs_mrtrix3d_small.py`), so it can be regenerated in Actions
//! (`.github/workflows/regenerate-fixtures.yml`): a 24×24×16 phantom in slices normal to z, and a
//! 21×26×15 one (odd axes) in slices normal to z, y and x, with `mrdegibbs`'s float64 output at
//! its defaults.

use num_complex::Complex64;
use qsm_core::unring::{gibbs_unring_2d, gibbs_unring_2d_complex, MrtrixUnringParams};

/// (dims, slice axis): `mrdegibbs -axes 0,1`, `0,1`, `0,2`, `1,2`.
const CASES: [((usize, usize, usize), usize); 4] =
    [((24, 24, 16), 2), ((21, 26, 15), 2), ((21, 26, 15), 1), ((21, 26, 15), 0)];

#[test]
fn gibbs_unring_2d_against_mrdegibbs() {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/gibbs_mrtrix2d_small.bin")).expect("fixture");
    let mut off = 0;
    let mut read = |n: usize| -> Vec<f64> {
        let v = (0..n).map(|i| f64::from_le_bytes(raw[off + 8 * i..off + 8 * i + 8].try_into().unwrap())).collect();
        off += 8 * n;
        v
    };
    for (dims, slice_axis) in CASES {
        let n = dims.0 * dims.1 * dims.2;
        let (input, mrtrix) = (read(n), read(n));
        let c: Vec<Complex64> = input.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        let out: Vec<f64> =
            gibbs_unring_2d_complex(&c, dims, slice_axis, MrtrixUnringParams::default()).iter().map(|v| v.re).collect();
        let norm = |v: &mut dyn Iterator<Item = f64>| v.map(|x| x * x).sum::<f64>().sqrt();
        let nrmse = norm(&mut out.iter().zip(&mrtrix).map(|(a, b)| a - b)) / norm(&mut mrtrix.iter().copied());
        let max = out.iter().zip(&mrtrix).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let change = norm(&mut input.iter().zip(&mrtrix).map(|(a, b)| a - b)) / norm(&mut input.iter().copied());
        println!(
            "{dims:?}, slices normal to axis {slice_axis}: vs mrdegibbs NRMSE {nrmse:.2e}, max |Δ| {max:.2e}; mrdegibbs changes the input by {:.2} %",
            100.0 * change
        );
        // Measured 2026-10-31: ~1e-16 (FFT rounding); mrdegibbs changes the input by a few %.
        assert!(nrmse < 1e-12, "NRMSE {nrmse}");
        // The real-input wrapper is the real part with negatives set to 0.
        let real = gibbs_unring_2d(&input, dims, slice_axis, MrtrixUnringParams::default());
        assert!(real.iter().zip(&mrtrix).all(|(a, b)| (a - b.max(0.0)).abs() < 1e-9));
    }
}

//! `utils::tukey_window` against the SNU χ-separation toolbox's `tukey_windowing` (v1.1.3).
//!
//! `tests/data/tukey_window_snu_small.bin` holds only the toolbox's output, made by
//! `tests/data/tukey_window_snu_small.m` (its header names the toolbox files, their hashes and
//! the MATLAB release). The inputs are rebuilt here exactly as that script builds them: real and
//! imaginary parts in [−1, 1) from an integer LCG over the linear index. Layout (little-endian),
//! per case in [`CASES`] order: the windowed volume's real parts, then its imaginary parts, over
//! all voxels and echoes in column-major order (x fastest, echo slowest), f64.

use num_complex::Complex64;
use qsm_core::utils::tukey_window::{tukey_window_complex, tukey_window_mag_phase};

/// `(dims, echoes, factor)`, as the script's `cases`.
const CASES: [((usize, usize, usize), usize, f64); 4] = [
    ((10, 9, 7), 3, 0.4),
    ((16, 12, 8), 1, 0.25),
    ((7, 8, 1), 2, 1.0),
    ((6, 5, 4), 2, 0.0),
];

/// The script's `input_volume(n, seed)`.
fn input(len: usize, seed: usize) -> Vec<Complex64> {
    let u = |s: usize, l: usize| ((1103515245.0 * ((l + s * 7919) as f64) + 12345.0) % 2f64.powi(31)) / 2f64.powi(31);
    (0..len).map(|l| Complex64::new(2.0 * u(2 * seed, l) - 1.0, 2.0 * u(2 * seed + 1, l) - 1.0)).collect()
}

#[test]
fn tukey_window_matches_the_toolbox() {
    let bytes = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/tukey_window_snu_small.bin")).unwrap();
    let f64s: Vec<f64> = bytes.as_chunks::<8>().0.iter().map(|&b| f64::from_le_bytes(b)).collect();
    let mut at = 0;
    for (c, &(dims, echoes, factor)) in CASES.iter().enumerate() {
        let n = dims.0 * dims.1 * dims.2;
        let len = n * echoes;
        let mut x = input(len, c + 1);
        let (re, im) = (&f64s[at..at + len], &f64s[at + len..at + 2 * len]);
        at += 2 * len;
        let x0 = x.clone();
        for echo in x.chunks_exact_mut(n) {
            tukey_window_complex(echo, dims, factor);
        }
        let (mut max_abs, mut max_ref) = (0f64, 0f64);
        for i in 0..len {
            max_abs = max_abs.max((x[i] - Complex64::new(re[i], im[i])).norm());
            max_ref = max_ref.max(Complex64::new(re[i], im[i]).norm());
        }
        println!("case {}: {dims:?} x {echoes} echoes, factor {factor}: max |Δ| {max_abs:.2e} (max |y| {max_ref:.2})", c + 1);
        assert!(max_abs < 1e-13, "case {}: max |Δ| {max_abs:e}", c + 1);

        // The magnitude/phase form, as the script calls it (abs and angle of the windowed image).
        if factor > 0.0 {
            let (mut mag, mut phase): (Vec<f64>, Vec<f64>) = x0.iter().map(|z| (z.norm(), z.arg())).unzip();
            for (m, p) in mag.chunks_exact_mut(n).zip(phase.chunks_exact_mut(n)) {
                tukey_window_mag_phase(m, p, dims, factor);
            }
            for i in 0..len {
                let y = Complex64::new(re[i], im[i]);
                assert!((mag[i] - y.norm()).abs() < 1e-12, "case {}: magnitude at {i}", c + 1);
                if y.norm() > 1e-6 {
                    let d = (phase[i] - y.arg()).rem_euclid(2.0 * std::f64::consts::PI);
                    assert!(d.min(2.0 * std::f64::consts::PI - d) < 1e-9, "case {}: phase at {i}", c + 1);
                }
            }
        }
    }
    assert_eq!(at, f64s.len(), "the fixture holds more than these cases");
}

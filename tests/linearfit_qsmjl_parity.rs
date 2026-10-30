//! `utils::multi_echo_linear_fit` against QSM.jl's `multi_echo_linear_fit` (kamesy/QSM.jl
//! fde4b92), with and without the phase offset.
//!
//! `tests/data/linearfit_qsmjl_small.bin` is made by `tests/data/linearfit_qsmjl_small.jl` in the
//! verified QSM-CI `julia-qsm` image (QSMxT/QSM-CI#299; the command is in its header, and
//! `.github/workflows/regenerate-fixtures.yml` runs it in Actions). The phantom is rebuilt here
//! exactly as the script builds it. QSM.rs weights each echo by its magnitude; QSM.jl's `W` is the
//! square root of the weights, so it is given `sqrt(mag)`.

use qsm_core::utils::multi_echo_linear_fit;

const N: [usize; 3] = [14, 12, 10];
const TES: [f64; 4] = [0.004, 0.009, 0.014, 0.019];

/// (unwrapped phase per echo, magnitude per echo, mask), as the generator builds them.
fn phantom() -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<u8>) {
    let n = N.iter().product();
    let (mut phas, mut mag, mut mask) = (vec![vec![0.0; n]; TES.len()], vec![vec![0.0; n]; TES.len()], vec![0u8; n]);
    for k in 1..=N[2] {
        for j in 1..=N[1] {
            for i in 1..=N[0] {
                let l = (i - 1) + N[0] * ((j - 1) + N[1] * (k - 1));
                let (x, y, z) = (i as f64 - 7.3, j as f64 - 6.1, k as f64 - 5.4);
                mask[l] = u8::from(x * x / 36.0 + y * y / 30.0 + z * z / 20.0 < 1.0);
                let f = 300.0 + 40.0 * x - 25.0 * y + 3.0 * x * z;
                let o = 0.4 - 0.05 * y + 0.01 * x * x;
                let r2 = 25.0 + 2.0 * x + 1.5 * z;
                for (e, &t) in TES.iter().enumerate() {
                    let d = 0.02 * (((i * 7 + j * 3 + k * 5 + (e + 1) * 11) % 13) as f64 - 6.0) / 6.0;
                    phas[e][l] = o + f * t + d;
                    mag[e][l] = (1.0 + 0.1 * z / 5.0) / (1.0 + r2 * t);
                }
            }
        }
    }
    (phas, mag, mask)
}

#[test]
fn linear_fit_matches_qsmjl() {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/linearfit_qsmjl_small.bin")).expect("fixture");
    let (phas, mag, mask) = phantom();
    let inside: Vec<usize> = (0..mask.len()).filter(|&l| mask[l] != 0).collect();
    let m = inside.len();
    let mut o = 0;
    let f64s = |o: &mut usize, k: usize| -> Vec<f64> {
        let v = (0..k).map(|i| f64::from_le_bytes(raw[*o + 8 * i..*o + 8 * i + 8].try_into().unwrap())).collect();
        *o += 8 * k;
        v
    };
    let rel = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| ((x - y) / y).abs()).fold(0.0, f64::max);
    for offset in [true, false] {
        assert_eq!(u32::from_le_bytes(raw[o..o + 4].try_into().unwrap()) as usize, m);
        o += 4;
        let slope = f64s(&mut o, m);
        let off = if offset { f64s(&mut o, m) } else { vec![] };
        let r = multi_echo_linear_fit(&phas, &mag, &TES, &mask, offset);
        let got: Vec<f64> = inside.iter().map(|&l| r.field[l]).collect();
        let d = rel(&got, &slope);
        println!("offset {offset}: slope max relative |Δ| {d:.2e}");
        assert!(d < 1e-12, "offset {offset}: slope differs by {d:e}");
        if offset {
            let got: Vec<f64> = inside.iter().map(|&l| r.phase_offset[l]).collect();
            let d = got.iter().zip(&off).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            println!("offset: max |Δ| {d:.2e} rad");
            assert!(d < 1e-12, "offset differs by {d:e}");
        }
    }
    assert_eq!(o, raw.len(), "fixture length");
}

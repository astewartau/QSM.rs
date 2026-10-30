//! The small synthetic phantom of the MEDI parity tests against MATLAB (`msmv_medi_parity.rs`,
//! `medi_smv_parity.rs`, `medi_cornell_parity.rs`). It is built from `+ − × ÷` only (no
//! transcendental functions, no FFT), so it is the same on every platform; [`dump`] writes it for
//! the MATLAB scripts in those files, and `tests/data/medi_cornell_small.py` rebuilds it in Python.

#![allow(dead_code)] // each test binary uses only part of it

use qsm_core::Grid;

pub const DIMS: [usize; 3] = [32, 36, 28];
pub const VS: [f64; 3] = [1.0, 1.0, 1.2];
pub const TE: [f64; 4] = [0.004, 0.008, 0.012, 0.016];
pub const B0: f64 = 3.0;

/// Deterministic value in [-1, 1) from a voxel index (xorshift, integer arithmetic only)
fn hash(i: usize, salt: u64) -> f64 {
    let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ salt.wrapping_mul(0xD1B5_4A32_D192_ED03);
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    (x >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

/// A rational bump `a / (1 + d²/w)`
fn bump(a: f64, d2: f64, w: f64) -> f64 {
    a / (1.0 + d2 / w)
}

pub struct Phantom {
    /// Local field in radians over the echo spacing (zero outside the mask)
    pub rdf: Vec<f64>,
    pub mask: Vec<u8>,
    /// R2* (Hz) for `msmv.m` alone
    pub r2s: Vec<f64>,
    /// Four echo magnitudes, echo slowest
    pub mag: Vec<f64>,
}

/// An ellipsoidal "brain" with a harmonic residual background (largest at the edge), a vein-like
/// tube and blobs in the field, bright tubes in R2* near the edge, and hashed texture; the
/// magnitudes decay as `M0/(1 + R2*·TE)`.
pub fn phantom() -> Phantom {
    build(false)
}

/// [`phantom`] with three CSF-like blobs near the centre (R2* ≈ 2 Hz), so MEDI+0's
/// `extract_CSF` finds ventricles.
pub fn phantom_csf() -> Phantom {
    build(true)
}

fn build(csf: bool) -> Phantom {
    let [nx, ny, nz] = DIMS;
    let n = nx * ny * nz;
    let (mut rdf, mut mask, mut r2s, mut mag) = (vec![0.0; n], vec![0u8; n], vec![0.0; n], vec![0.0; 4 * n]);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let p = i + nx * (j + ny * k);
                let (x, y, z) = (i as f64 - 16.0, j as f64 - 18.0, (k as f64 - 14.0) * VS[2]);
                let inside = x * x / 169.0 + y * y / 225.0 + z * z / 144.0 < 1.0;
                let tube = (x - 9.0) * (x - 9.0) + (y - 6.0) * (y - 6.0);
                let tube2 = (y + 10.0) * (y + 10.0) + (z - 4.0) * (z - 4.0);
                let blob = (x + 4.0) * (x + 4.0) + (y - 3.0) * (y - 3.0) + z * z;
                let mut r2 = 20.0 + bump(45.0, tube, 1.5) + bump(35.0, tube2 + 0.25 * (x + 2.0) * (x + 2.0), 1.0)
                    + bump(25.0, blob, 6.0) + 3.0 * hash(p, 1);
                let ventricle = [(0.0, -5.0, 0.0), (0.0, 5.0, 0.0), (-3.0, 0.0, 4.0)]
                    .iter()
                    .map(|&(cx, cy, cz): &(f64, f64, f64)| (x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz))
                    .fold(f64::INFINITY, f64::min);
                if csf && ventricle < 6.0 {
                    r2 = 2.0 + 0.5 * hash(p, 5);
                }
                r2s[p] = r2;
                let m0 = if inside { 1.0 + 0.05 * hash(p, 2) } else { 0.05 * (1.0 + hash(p, 3)) };
                for (e, &te) in TE.iter().enumerate() {
                    mag[p + n * e] = m0 / (1.0 + r2 * te) * (1.0 + 0.01 * hash(p, 10 + e as u64));
                }
                if inside {
                    mask[p] = 1;
                    rdf[p] = 0.002 * (x * x - y * y) + 0.0012 * x * z + bump(0.3, tube, 2.0)
                        - bump(0.15, tube2, 1.5) + bump(0.1, blob, 9.0) + 0.004 * hash(p, 4);
                }
            }
        }
    }
    Phantom { rdf, mask, r2s, mag }
}

pub fn grid() -> Grid {
    Grid::new(DIMS[0], DIMS[1], DIMS[2], VS[0], VS[1], VS[2])
}

/// Writes the phantom for the MATLAB scripts: `rdf.f64`, `r2s.f64`, `mag.f64` and `mask.u8`
/// (column-major, little-endian) in `dir`
pub fn dump(dir: &str) {
    let p = phantom();
    let f64s = |v: &[f64]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    std::fs::write(format!("{dir}/rdf.f64"), f64s(&p.rdf)).unwrap();
    std::fs::write(format!("{dir}/r2s.f64"), f64s(&p.r2s)).unwrap();
    std::fs::write(format!("{dir}/mag.f64"), f64s(&p.mag)).unwrap();
    std::fs::write(format!("{dir}/mask.u8"), &p.mask).unwrap();
}

/// MEDI's ppm → radians over `delta_TE` at this phantom's field strength
pub fn ppm_to_rad() -> f64 {
    2.0 * std::f64::consts::PI * (TE[1] - TE[0]) * 42.576e6 * B0 * 1e-6
}

/// (max |a − b|, relative L2 of a − b to b)
pub fn diff(a: &[f64], b: &[f64]) -> (f64, f64) {
    let max = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    let rel = (a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / b.iter().map(|y| y * y).sum::<f64>()).sqrt();
    (max, rel)
}

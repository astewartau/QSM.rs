//! Deterministic AMP-PE run, for bit-for-bit comparison between builds.
use qsm_core::inversion::{amp_pe, AmpPeParams};
use qsm_core::Grid;

fn main() {
    let (nx, ny, nz) = (32usize, 32, 24);
    let grid = Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);
    let n = nx * ny * nz;
    // deterministic pseudo-random field + a couple of blobs, no RNG crate needed
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut rnd = || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
                       ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5 };
    let mut field = vec![0.0f64; n];
    let mut mag = vec![0.0f64; n];
    let mut mask = vec![0u8; n];
    for k in 0..nz { for j in 0..ny { for i in 0..nx {
        let idx = (k * ny + j) * nx + i;
        let (dx, dy, dz) = (i as f64 - 16.0, j as f64 - 16.0, k as f64 - 12.0);
        let r = (dx*dx + dy*dy + dz*dz).sqrt();
        if r < 11.0 { mask[idx] = 1; }
        field[idx] = 0.02 * (dx / 16.0) + 0.01 * rnd();
        if r < 4.0 { field[idx] += 0.05; }
        mag[idx] = 100.0 + 40.0 * (dy / 16.0) + 10.0 * rnd();
    }}}
    let mut p = AmpPeParams::default();
    p.max_linearization_ite = 4;
    let chi = amp_pe(&field, &mask, Some(&mag), &grid, (0.0, 0.0, 1.0), &p, |_, _| {});
    // bitwise checksum so any numerical difference at all shows up
    let mut h = 1469598103934665603u64;
    for v in &chi { for b in v.to_bits().to_le_bytes() { h ^= b as u64; h = h.wrapping_mul(1099511628211); } }
    let inm: Vec<f64> = chi.iter().zip(&mask).filter(|(_, &m)| m != 0).map(|(v, _)| *v).collect();
    let mean = inm.iter().sum::<f64>() / inm.len() as f64;
    let sd = (inm.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / inm.len() as f64).sqrt();
    println!("hash=0x{:016x} n={} mean={:.17e} sd={:.17e}", h, inm.len(), mean, sd);
}

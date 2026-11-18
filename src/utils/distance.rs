//! Exact Euclidean distance transform on anisotropic grids.
//!
//! [`distance_to_set_mm`] is `scipy.ndimage.distance_transform_edt(~set, sampling=voxel_size)`:
//! for every voxel, the distance in mm to the nearest voxel of `set` (0 inside it). It uses the
//! separable algorithm of Felzenszwalb & Huttenlocher (Theory of Computing 8:415-428, 2012), one
//! axis at a time, with the squared voxel spacing of that axis.

use crate::utils::bias_correction::check_volume;
use crate::Grid;

/// Distance in mm from each voxel to the nearest non-zero voxel of `set`, on `grid`'s spacing.
/// Voxels of `set` get 0. With an empty `set` every distance is `f64::INFINITY` (scipy has no
/// answer for that case).
pub fn distance_to_set_mm(set: &[u8], grid: &Grid) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let n = check_volume("distance_to_set_mm: set", set.len(), grid);
    if !set.iter().any(|&v| v != 0) {
        return vec![f64::INFINITY; n];
    }
    // Large but finite, so the parabola intersections stay finite.
    const FAR: f64 = 1e30;
    let mut f: Vec<f64> = set.iter().map(|&v| if v != 0 { 0.0 } else { FAR }).collect();
    let (vx, vy, vz) = grid.voxel_size;
    let mut line = Vec::new();
    let mut out = Vec::new();
    let mut ws = Workspace::default();
    let mut pass = |f: &mut [f64], base: usize, stride: usize, len: usize, s2: f64| {
        line.clear();
        line.extend((0..len).map(|i| f[base + i * stride]));
        edt_1d(&line, s2, &mut out, &mut ws);
        for (i, &v) in out.iter().enumerate() {
            f[base + i * stride] = v;
        }
    };
    for z in 0..nz {
        for y in 0..ny {
            pass(&mut f, nx * (y + ny * z), 1, nx, vx * vx);
        }
    }
    for z in 0..nz {
        for x in 0..nx {
            pass(&mut f, x + nx * ny * z, nx, ny, vy * vy);
        }
    }
    for y in 0..ny {
        for x in 0..nx {
            pass(&mut f, x + nx * y, nx * ny, nz, vz * vz);
        }
    }
    f.iter().map(|&v| if v >= FAR { f64::INFINITY } else { v.sqrt() }).collect()
}

#[derive(Default)]
struct Workspace {
    v: Vec<usize>,
    z: Vec<f64>,
}

/// 1D squared distance transform `d(q) = min_p s2·(q−p)² + f(p)`.
fn edt_1d(f: &[f64], s2: f64, d: &mut Vec<f64>, ws: &mut Workspace) {
    let n = f.len();
    d.clear();
    d.resize(n, 0.0);
    if n == 0 {
        return;
    }
    ws.v.clear();
    ws.v.resize(n, 0);
    ws.z.clear();
    ws.z.resize(n + 1, 0.0);
    let (v, z) = (&mut ws.v, &mut ws.z);
    let inter = |q: usize, p: usize| {
        let (qf, pf) = (q as f64, p as f64);
        ((f[q] + s2 * qf * qf) - (f[p] + s2 * pf * pf)) / (2.0 * s2 * (qf - pf))
    };
    let mut k = 0usize;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    for q in 1..n {
        let mut s = inter(q, v[k]);
        while s <= z[k] {
            k -= 1;
            s = inter(q, v[k]);
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = f64::INFINITY;
    }
    k = 0;
    for (q, dq) in d.iter_mut().enumerate() {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let dx = q as f64 - v[k] as f64;
        *dq = s2 * dx * dx + f[v[k]];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute(set: &[u8], g: &Grid) -> Vec<f64> {
        let (nx, ny, nz) = g.dims;
        let pts: Vec<(usize, usize, usize)> = (0..set.len())
            .filter(|&i| set[i] != 0)
            .map(|i| (i % nx, (i / nx) % ny, i / (nx * ny)))
            .collect();
        (0..nx * ny * nz)
            .map(|i| {
                let (x, y, z) = (i % nx, (i / nx) % ny, i / (nx * ny));
                pts.iter()
                    .map(|&(a, b, c)| {
                        let (sx, sy, sz) = g.voxel_size;
                        let dx = (x as f64 - a as f64) * sx;
                        let dy = (y as f64 - b as f64) * sy;
                        let dz = (z as f64 - c as f64) * sz;
                        (dx * dx + dy * dy + dz * dz).sqrt()
                    })
                    .fold(f64::INFINITY, f64::min)
            })
            .collect()
    }

    #[test]
    fn matches_brute_force_on_an_anisotropic_grid() {
        let g = Grid::new(9, 7, 6, 0.8, 1.0, 2.5);
        let mut set = vec![0u8; 9 * 7 * 6];
        for &i in &[3usize, 100, 177, 250, 377] {
            set[i] = 1;
        }
        let d = distance_to_set_mm(&set, &g);
        for (a, b) in d.iter().zip(brute(&set, &g)) {
            assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        }
    }

    #[test]
    fn an_empty_set_is_infinitely_far() {
        let g = Grid::new(3, 3, 3, 1.0, 1.0, 1.0);
        assert!(distance_to_set_mm(&[0; 27], &g).iter().all(|d| d.is_infinite()));
    }
}

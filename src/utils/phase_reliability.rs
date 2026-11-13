//! UK Biobank's phase-reliability mask trimming.
//!
//! UK Biobank's QSM pipeline (Wang, Tendler & Miller; Wang et al. 2022, Nat. Neurosci.,
//! doi:10.1038/s41593-022-01074-w; code at <https://git.fmrib.ox.ac.uk/cwang/uk_biobank_qsm_pipeline>,
//! Apache-2.0) removes the unreliable phase near the sinuses from its brain mask twice: once before
//! V-SHARP and once after it, before iLSQR. Both trims use two phase-reliability maps, one from each
//! of UK Biobank's two echoes (`phasevariance_nonlin_v2.m`). This module ports those maps
//! ([`ukb_phase_reliability`]) and the trimming ([`ukb_trim_mask`]) exactly. How a pipeline uses
//! them is [`BgRemovalConfig::ukb_mask_trim`](crate::pipeline::BgRemovalConfig::ukb_mask_trim).
//!
//! # The reliability map (`phasevariance_nonlin_v2(mask, phase, 2, voxsz)`)
//!
//! The magnitude of the masked unit phasor `M·e^{iφ}`, convolved with a ball of radius 2 mm, over
//! the convolved mask: 1 where the phase is uniform over the ball, lower where it varies. Outside
//! the mask the map is 1. In the authors' code, step by step:
//!
//! 1. Per axis, a factor `f = round(10·voxel size)` (tenths of a millimetre; 8 for UK Biobank's
//!    0.8 mm, 30 for its 3 mm), and `f_xy = round(f_y/f_x)`, `f_xz = round(f_z/f_x)`.
//! 2. The volume is zero-padded at its end, per axis, to a multiple of that axis's factor.
//! 3. A ball of radius `10·r` (20 units) on a fine grid of `n_x × n_y·f_xy × n_z·f_xz` points
//!    with integer steps from `−n/2`, so that one unit is `1/f_x` of an x voxel. The ball holds
//!    the points with `x² + y² + z² ≤ (10·r)²`.
//! 4. Averaged over blocks of `f_x × f_y × f_z` fine points: a partial-volume kernel of
//!    `n_x/f_x × n_y/f_y·f_xy × n_z/f_z·f_xz` voxels. (The fine grid's y and z steps are only
//!    0.1 mm when `f_xy` and `f_xz` are 1, as on UK Biobank's 0.8 × 0.8 mm slices; the kernel is
//!    the authors', whatever the voxel size.)
//! 5. The kernel is placed in a zero volume of the padded size, starting at `(n − n_k)/2 + 1`
//!    (MATLAB, 1-based) per axis. That start is a half-integer on most grids (e.g. 1 mm voxels),
//!    which MATLAB indexes with a warning ("Integer operands are required for colon operator
//!    when used as index") by rounding up: the start used here is `ceil((n − n_k)/2)` (0-based).
//!    Measured against the verified QSM-CI `ukb-qsm` container (MATLAB R2023b): rounding up
//!    reproduces its maps to 2e-15 on 1 mm, 1 × 1 × 1.5 mm and 0.8 × 0.8 × 3 mm phantoms; rounding
//!    down moves them by up to 0.65.
//! 6. `fftshift(ifftn(fftn(·)·fftn(kernel)))`, a circular convolution, of the masked phasor and
//!    of the mask; their magnitudes divided (`|·| / (|·| + eps)`), 1 outside the mask, cropped back.
//!
//! The kernel's placement and the `fftshift` decide where the ball is centred, and with the
//! half-voxel offsets above it is not exactly on the voxel. That is the authors' code, followed as
//! it is (it is how UK Biobank's maps are made); QSM-CI's `ukb-qsm` README calls it out.
//!
//! # The trim
//!
//! [`ukb_trim_mask`] is, in the authors' code (`UKBiobank_QSM.m`):
//!
//! ```matlab
//! mask(map < 0.6) = 0;  mask(map2 < 0.5) = 0;          % before V-SHARP; 0.7 / 0.6 after
//! for ii = 1:dim(3)
//!     mask(:, :, ii) =  bwareaopen(mask(:, :, ii), 300);   % 200 after
//!     mask(:, :, ii) =~ bwareaopen(~mask(:, :, ii), 50);   % 30 after
//! end
//! mask = imfill(mask, 26, 'holes');
//! ```
//!
//! `bwareaopen` on a slice keeps the 8-connected components of at least that many pixels; on the
//! slice's complement it fills every 8-connected background region smaller than that, holes and
//! regions at the slice's edge alike. `imfill(·, 26, 'holes')` then fills every background voxel
//! that a 26-connected path through the background cannot join to the array's border. [`UKB_TRIM_FIELD`]
//! and [`UKB_TRIM_LOCAL`] are the two trims' settings. Checked voxel for voxel against the
//! container's trimmed masks (`tests/ukb_qsm_parity.rs`).
//!
//! # Several echoes
//!
//! UK Biobank acquires two echoes: `map` comes from the first, `map2` from the second. With more
//! echoes, `map2` comes from the last (as QSM-CI's `ukb-qsm` generalises it), so two echoes are
//! exactly the authors' pipeline.

use num_complex::Complex64;

use crate::fft::{fft3d, ifft3d};
use crate::Grid;

/// The radius of the reliability map's ball, in mm: UK Biobank's `phasevariance_nonlin_v2(·, ·, 2, ·)`.
pub const UKB_RELIABILITY_RADIUS_MM: f64 = 2.0;

/// One of UK Biobank's two mask trims (see the module docs, *The trim*).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UkbTrim {
    /// Voxels whose first-echo reliability is below this are removed.
    pub first_echo: f64,
    /// Voxels whose last-echo reliability is below this are removed.
    pub last_echo: f64,
    /// Per slice, 8-connected mask components smaller than this many pixels are removed.
    pub min_area: usize,
    /// Per slice, 8-connected background regions smaller than this many pixels are filled.
    pub min_hole_area: usize,
}

/// The trim before V-SHARP: reliability 0.6 / 0.5, components of 300 pixels, holes of 50.
pub const UKB_TRIM_FIELD: UkbTrim = UkbTrim { first_echo: 0.6, last_echo: 0.5, min_area: 300, min_hole_area: 50 };
/// The trim of V-SHARP's mask, before iLSQR: reliability 0.7 / 0.6, components of 200 pixels,
/// holes of 30.
pub const UKB_TRIM_LOCAL: UkbTrim = UkbTrim { first_echo: 0.7, last_echo: 0.6, min_area: 200, min_hole_area: 30 };

/// The two reliability maps a trim reads: from the first echo and from the last.
#[derive(Clone, Debug, PartialEq)]
pub struct UkbPhaseReliability {
    /// From the first echo's phase (UK Biobank's `map`).
    pub first_echo: Vec<f64>,
    /// From the last echo's phase (UK Biobank's `map2`).
    pub last_echo: Vec<f64>,
}

impl UkbPhaseReliability {
    /// Both maps, from the wrapped phase of each echo (radians) inside the brain `mask` that the
    /// trims will start from (UK Biobank computes them on its untrimmed mask).
    pub fn compute(phases: &[&[f64]], mask: &[u8], grid: &Grid) -> Result<Self, String> {
        let (first, last) = match (phases.first(), phases.last()) {
            (Some(f), Some(l)) => (*f, *l),
            _ => return Err("UK Biobank phase reliability: no echoes".into()),
        };
        Ok(Self {
            first_echo: ukb_phase_reliability(mask, first, grid, UKB_RELIABILITY_RADIUS_MM)?,
            last_echo: ukb_phase_reliability(mask, last, grid, UKB_RELIABILITY_RADIUS_MM)?,
        })
    }

    /// This trim of `mask` ([`ukb_trim_mask`]).
    pub fn trim(&self, mask: &[u8], grid: &Grid, trim: &UkbTrim) -> Vec<u8> {
        ukb_trim_mask(mask, &self.first_echo, &self.last_echo, grid, trim)
    }
}

/// UK Biobank's phase-reliability map, `phasevariance_nonlin_v2(mask, phase, radius_mm, voxsz)`
/// (module docs). `phase` in radians (wrapped or not: only `e^{iφ}` is used).
pub fn ukb_phase_reliability(mask: &[u8], phase: &[f64], grid: &Grid, radius_mm: f64) -> Result<Vec<f64>, String> {
    let (n0x, n0y, n0z) = grid.dims;
    let n0 = n0x * n0y * n0z;
    if mask.len() != n0 || phase.len() != n0 {
        return Err(format!("UK Biobank phase reliability: {} mask and {} phase voxels for a {n0}-voxel grid", mask.len(), phase.len()));
    }
    let (rx, ry, rz) = grid.voxel_size;
    // MATLAB round: half away from zero, which f64::round is.
    let factor = |r: f64| (r * 10.0).round() as i64;
    let (fx, fy, fz) = (factor(rx), factor(ry), factor(rz));
    if fx < 1 || fy < 1 || fz < 1 {
        return Err(format!("UK Biobank phase reliability: voxel size {:?} mm is below 0.05 mm", grid.voxel_size));
    }
    let fxy = (fy as f64 / fx as f64).round() as i64;
    let fxz = (fz as f64 / fx as f64).round() as i64;
    let padded = |n: usize, f: i64| -> usize {
        let f = f as usize;
        n.div_ceil(f) * f
    };
    let (nx, ny, nz) = (padded(n0x, fx), padded(n0y, fy), padded(n0z, fz));
    let n = nx * ny * nz;

    // Kernel: the ball on the fine grid, averaged over f_x × f_y × f_z blocks. Only the fine
    // points inside the ball are non-zero, so each block's sum counts those (exact, as MATLAB's
    // mean of ones and zeros is).
    let (kx, ky, kz) = (nx / fx as usize, ny / fy as usize * fxy as usize, nz / fz as usize * fxz as usize);
    let fine = [nx as i64, ny as i64 * fxy, nz as i64 * fxz];
    let r_units = radius_mm * 10.0;
    let r2 = r_units * r_units;
    let mut counts = vec![0u64; kx * ky * kz];
    // Fine coordinate of index i on an axis of m points: −m/2 + i (MATLAB's −m/2 : m/2 − 1).
    let range = |m: i64| -> (usize, usize) {
        let start = -(m as f64) / 2.0;
        let lo = (-r_units - start).ceil().max(0.0) as usize;
        let hi = ((r_units - start).floor() as i64).min(m - 1).max(-1);
        (lo, (hi + 1) as usize)
    };
    let (ix, iy, iz) = (range(fine[0]), range(fine[1]), range(fine[2]));
    for k in iz.0..iz.1 {
        let z = -(fine[2] as f64) / 2.0 + k as f64;
        let bz = k / fz as usize;
        for j in iy.0..iy.1 {
            let y = -(fine[1] as f64) / 2.0 + j as f64;
            let by = j / fy as usize;
            for i in ix.0..ix.1 {
                let x = -(fine[0] as f64) / 2.0 + i as f64;
                if x * x + y * y + z * z <= r2 {
                    counts[i / fx as usize + kx * (by + ky * bz)] += 1;
                }
            }
        }
    }
    let block = (fx * fy * fz) as f64;
    // Placed from MATLAB's 1-based (n − n_k)/2 + 1, which MATLAB rounds up when it is a
    // half-integer (module docs, step 5): 0-based ceil((n − n_k)/2).
    let start = |m: usize, k: usize| -> Result<usize, String> {
        if k > m {
            return Err(format!("UK Biobank phase reliability: kernel of {k} voxels on an axis of {m}"));
        }
        Ok((m - k).div_ceil(2))
    };
    let (sx, sy, sz) = (start(nx, kx)?, start(ny, ky)?, start(nz, kz)?);
    let mut kernel = vec![Complex64::new(0.0, 0.0); n];
    for c in 0..kz {
        for b in 0..ky {
            for a in 0..kx {
                let v = counts[a + kx * (b + ky * c)];
                if v != 0 {
                    kernel[(sx + a) + nx * ((sy + b) + ny * (sz + c))] = Complex64::new(v as f64 / block, 0.0);
                }
            }
        }
    }
    fft3d(&mut kernel, nx, ny, nz);

    // The masked phasor and the mask on the padded grid.
    let mut phasor = vec![Complex64::new(0.0, 0.0); n];
    let mut m = vec![Complex64::new(0.0, 0.0); n];
    for k in 0..n0z {
        for j in 0..n0y {
            for i in 0..n0x {
                let l0 = i + n0x * (j + n0y * k);
                if mask[l0] != 0 {
                    let l = i + nx * (j + ny * k);
                    phasor[l] = Complex64::new(0.0, phase[l0]).exp();
                    m[l] = Complex64::new(1.0, 0.0);
                }
            }
        }
    }
    let convolve = |v: &mut Vec<Complex64>| {
        fft3d(v, nx, ny, nz);
        for (a, b) in v.iter_mut().zip(&kernel) {
            *a *= b;
        }
        ifft3d(v, nx, ny, nz);
    };
    convolve(&mut phasor);
    convolve(&mut m);

    // fftshift: output index (i + ⌊n/2⌋) mod n holds input i; read back for the cropped grid.
    let unshift = |o: usize, len: usize| (o + len - len / 2) % len;
    let mut map = vec![1.0; n0];
    for k in 0..n0z {
        let kk = unshift(k, nz);
        for j in 0..n0y {
            let jj = unshift(j, ny);
            for i in 0..n0x {
                let l0 = i + n0x * (j + n0y * k);
                if mask[l0] == 0 {
                    continue;
                }
                let l = unshift(i, nx) + nx * (jj + ny * kk);
                map[l0] = phasor[l].norm() / (m[l].norm() + f64::EPSILON);
            }
        }
    }
    Ok(map)
}

/// One of UK Biobank's mask trims (module docs, *The trim*): `mask` less the voxels whose
/// reliability is below the trim's thresholds, then per axial slice small components removed and
/// small background regions filled (8-connected), then holes filled in 3-D (26-connected).
pub fn ukb_trim_mask(mask: &[u8], first_echo: &[f64], last_echo: &[f64], grid: &Grid, trim: &UkbTrim) -> Vec<u8> {
    let (nx, ny, nz) = grid.dims;
    let nxy = nx * ny;
    // `mask(map < t) = 0`: only a value below the threshold removes a voxel (a NaN would not).
    let below = |v: f64, t: f64| v < t;
    let mut m: Vec<bool> = (0..mask.len())
        .map(|l| mask[l] != 0 && !below(first_echo[l], trim.first_echo) && !below(last_echo[l], trim.last_echo))
        .collect();
    for k in 0..nz {
        let slice = &mut m[k * nxy..(k + 1) * nxy];
        remove_small_2d(slice, nx, ny, true, trim.min_area);
        remove_small_2d(slice, nx, ny, false, trim.min_hole_area);
    }
    fill_enclosed_background_26(&mut m, grid.dims);
    m.into_iter().map(u8::from).collect()
}

/// `bwareaopen` on one slice, 8-connected: the components of pixels equal to `value` that are
/// smaller than `min_area` are set to `!value`.
fn remove_small_2d(slice: &mut [bool], nx: usize, ny: usize, value: bool, min_area: usize) {
    let mut seen = vec![false; slice.len()];
    let mut component = Vec::new();
    let mut stack = Vec::new();
    for seed in 0..slice.len() {
        if slice[seed] != value || seen[seed] {
            continue;
        }
        component.clear();
        seen[seed] = true;
        stack.push(seed);
        while let Some(p) = stack.pop() {
            component.push(p);
            let (x, y) = (p % nx, p / nx);
            for yy in y.saturating_sub(1)..(y + 2).min(ny) {
                for xx in x.saturating_sub(1)..(x + 2).min(nx) {
                    let q = xx + yy * nx;
                    if !seen[q] && slice[q] == value {
                        seen[q] = true;
                        stack.push(q);
                    }
                }
            }
        }
        if component.len() < min_area {
            for &p in &component {
                slice[p] = !value;
            }
        }
    }
}

/// `imfill(mask, 26, 'holes')`: every background voxel that no 26-connected background path joins
/// to the array's border becomes foreground.
fn fill_enclosed_background_26(m: &mut [bool], dims: (usize, usize, usize)) {
    let (nx, ny, nz) = dims;
    let nxy = nx * ny;
    let mut outside = vec![false; m.len()];
    let mut stack = Vec::new();
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let border = i == 0 || j == 0 || k == 0 || i + 1 == nx || j + 1 == ny || k + 1 == nz;
                let l = i + nx * (j + ny * k);
                if border && !m[l] && !outside[l] {
                    outside[l] = true;
                    stack.push(l);
                }
            }
        }
    }
    while let Some(l) = stack.pop() {
        let (x, y, z) = (l % nx, (l / nx) % ny, l / nxy);
        for zz in z.saturating_sub(1)..(z + 2).min(nz) {
            for yy in y.saturating_sub(1)..(y + 2).min(ny) {
                for xx in x.saturating_sub(1)..(x + 2).min(nx) {
                    let q = xx + nx * (yy + ny * zz);
                    if !m[q] && !outside[q] {
                        outside[q] = true;
                        stack.push(q);
                    }
                }
            }
        }
    }
    for (v, o) in m.iter_mut().zip(outside) {
        if !*v && !o {
            *v = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_phase_is_fully_reliable_and_noise_is_trimmed() {
        let g = Grid::new(24, 24, 12, 1.0, 1.0, 1.0);
        let n = 24 * 24 * 12;
        let mask: Vec<u8> = (0..n).map(|l| {
            let (i, j, k) = ((l % 24) as f64 - 11.5, ((l / 24) % 24) as f64 - 11.5, (l / 576) as f64 - 5.5);
            u8::from(i * i / 100.0 + j * j / 100.0 + k * k / 25.0 < 1.0)
        }).collect();
        let flat = vec![0.7; n];
        let map = ukb_phase_reliability(&mask, &flat, &g, 2.0).unwrap();
        assert!(map.iter().all(|&v| (v - 1.0).abs() < 1e-12), "uniform phase");
        // phase alternating by π between neighbours is unreliable everywhere inside
        let checker: Vec<f64> = (0..n).map(|l| std::f64::consts::PI * ((l % 24 + (l / 24) % 24 + l / 576) % 2) as f64).collect();
        let r = UkbPhaseReliability::compute(&[&checker, &checker], &mask, &g).unwrap();
        assert!(r.trim(&mask, &g, &UKB_TRIM_FIELD).iter().all(|&v| v == 0));
        assert_eq!(r.first_echo.iter().zip(&mask).filter(|(_, &m)| m == 0).count(), mask.iter().filter(|&&m| m == 0).count());
    }

    #[test]
    fn imfill_26_leaves_a_diagonally_open_cavity() {
        // A cavity whose only way out is a diagonal (corner) step is not a hole for imfill(·, 26).
        let dims = (5, 5, 5);
        let mut m = vec![true; 125];
        let at = |i: usize, j: usize, k: usize| i + 5 * (j + 5 * k);
        m[at(2, 2, 2)] = false; // enclosed
        let mut open = m.clone();
        open[at(1, 1, 1)] = false; // joined diagonally to (2,2,2) and to (0,0,0)'s neighbourhood…
        open[at(0, 0, 0)] = false; // …and the border
        fill_enclosed_background_26(&mut m, dims);
        assert!(m[at(2, 2, 2)]);
        fill_enclosed_background_26(&mut open, dims);
        assert!(!open[at(2, 2, 2)]);
    }
}

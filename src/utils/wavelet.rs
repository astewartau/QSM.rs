//! Separable 3D periodic Daubechies wavelet transform (db1/db2).
//!
//! A faithful reimplementation of MATLAB's `wavedec3`/`waverec3` in **periodic
//! (`'per'`) mode** — the sparsifying basis used by the AMP-PE QSM inversion.
//!
//! The transform is orthonormal, so synthesis is the exact adjoint of analysis
//! (perfect reconstruction). Arrays are flat `Vec<f64>` in column-major
//! (Fortran) order — `index = i0 + i1*n0 + i2*n0*n1` — matching MATLAB, so the
//! flattened coefficient vector lines up element-for-element with
//! `wavedec3` + the toolbox's coefficient concatenation.
//!
//! ## Conventions (reverse-engineered from MATLAB `wavedec3` `'mode','per'`)
//! One analysis step along an axis of length `N` (even) with filter `f` (length
//! `lf`) produces `N/2` coefficients:
//! ```text
//!   c[k] = sum_{n=0}^{lf-1} f[n] * x[(2k - n + 1) mod N]
//! ```
//! with the low-pass filter `LoD` giving the approximation and the high-pass
//! `HiD[k] = (-1)^(k+1) * LoD[lf-1-k]` giving the detail. The 3D step applies
//! this along axis 0, then 1, then 2; the 8 subbands are ordered by
//! `4*a2 + 2*a0 + a1` (a=0 low, a=1 high on that axis). Multilevel decomposition
//! recurses on the all-low (approximation) subband. The coefficient vector is
//! `[coarsest approximation, coarsest 7 details, ..., finest 7 details]`.

/// Daubechies analysis low-pass filter `LoD` for the given order (1 or 2).
fn lod_filter(order: usize) -> Vec<f64> {
    match order {
        1 => {
            let s = std::f64::consts::FRAC_1_SQRT_2;
            vec![s, s]
        }
        2 => vec![
            -0.12940952255126037,
            0.2241438680420134,
            0.8365163037378079,
            0.48296291314453416,
        ],
        _ => panic!("wavelet: only db1 and db2 (order 1 or 2) are supported, got {order}"),
    }
}

/// High-pass analysis filter from the low-pass one: `HiD[k] = (-1)^(k+1) LoD[lf-1-k]`.
fn hid_from_lod(lod: &[f64]) -> Vec<f64> {
    let lf = lod.len();
    (0..lf)
        .map(|k| {
            let sign = if (k + 1) % 2 == 0 { 1.0 } else { -1.0 };
            sign * lod[lf - 1 - k]
        })
        .collect()
}

/// Subband ordering: (a0, a1, a2) detail flags for the 8 subbands of one 3D step,
/// in MATLAB `wavedec3` order (index = `4*a2 + 2*a0 + a1`).
const SUBBANDS: [(usize, usize, usize); 8] = [
    (0, 0, 0), // AAA (approximation)
    (0, 1, 0), // ADA
    (1, 0, 0), // DAA
    (1, 1, 0), // DDA
    (0, 0, 1), // AAD
    (0, 1, 1), // ADD
    (1, 0, 1), // DAD
    (1, 1, 1), // DDD
];

/// Analyze a 3D array along one axis, splitting it into low/high halves.
///
/// `dims` are the current dimensions; the returned arrays have `dims[axis]`
/// halved. Column-major layout throughout.
fn analyze_axis(
    inp: &[f64],
    dims: (usize, usize, usize),
    axis: usize,
    lod: &[f64],
    hid: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let (n0, n1, n2) = dims;
    let n = [n0, n1, n2][axis];
    let half = n / 2;
    let lf = lod.len();

    let (stride, num_lines, _) = axis_geometry(dims, axis);
    let out_dims = halve_dim(dims, axis);
    let out_len = out_dims.0 * out_dims.1 * out_dims.2;
    let mut low = vec![0.0; out_len];
    let mut high = vec![0.0; out_len];

    let mut line = vec![0.0f64; n];
    let (os, _, _) = axis_geometry(out_dims, axis);

    for l in 0..num_lines {
        let base = line_base(l, dims, axis);
        for (t, slot) in line.iter_mut().enumerate() {
            *slot = inp[base + t * stride];
        }
        let obase = line_base(l, out_dims, axis);
        for k in 0..half {
            let center = 2 * k + 1;
            let mut ca = 0.0;
            let mut cd = 0.0;
            for nn in 0..lf {
                // (center - nn) mod n, with center = 2k+1
                let idx = (center as isize - nn as isize).rem_euclid(n as isize) as usize;
                let v = line[idx];
                ca += lod[nn] * v;
                cd += hid[nn] * v;
            }
            low[obase + k * os] = ca;
            high[obase + k * os] = cd;
        }
    }
    (low, high)
}

/// Synthesis along one axis: the exact adjoint of `analyze_axis`. Combines
/// low/high halves (dims already halved along `axis`) back to full length.
fn synthesize_axis(
    low: &[f64],
    high: &[f64],
    dims_half: (usize, usize, usize),
    axis: usize,
    lod: &[f64],
    hid: &[f64],
) -> Vec<f64> {
    let half = [dims_half.0, dims_half.1, dims_half.2][axis];
    let n = half * 2;
    let lf = lod.len();
    let out_dims = double_dim(dims_half, axis);
    let out_len = out_dims.0 * out_dims.1 * out_dims.2;
    let mut out = vec![0.0; out_len];

    let (in_stride, num_lines, _) = axis_geometry(dims_half, axis);
    let (out_stride, _, _) = axis_geometry(out_dims, axis);

    let mut line = vec![0.0f64; n];
    for l in 0..num_lines {
        for slot in line.iter_mut() {
            *slot = 0.0;
        }
        let ibase = line_base(l, dims_half, axis);
        for k in 0..half {
            let ca = low[ibase + k * in_stride];
            let cd = high[ibase + k * in_stride];
            let center = 2 * k + 1;
            for nn in 0..lf {
                let idx = (center as isize - nn as isize).rem_euclid(n as isize) as usize;
                line[idx] += lod[nn] * ca + hid[nn] * cd;
            }
        }
        let obase = line_base(l, out_dims, axis);
        for (t, &v) in line.iter().enumerate() {
            out[obase + t * out_stride] = v;
        }
    }
    out
}

// --- column-major axis geometry helpers ---

/// Returns (stride along axis, number of lines, unused marker).
fn axis_geometry(dims: (usize, usize, usize), axis: usize) -> (usize, usize, ()) {
    let (n0, n1, n2) = dims;
    let total = n0 * n1 * n2;
    let n = [n0, n1, n2][axis];
    let stride = match axis {
        0 => 1,
        1 => n0,
        2 => n0 * n1,
        _ => unreachable!(),
    };
    (stride, total / n, ())
}

/// Base flat offset of line index `l` for the given axis (column-major).
fn line_base(l: usize, dims: (usize, usize, usize), axis: usize) -> usize {
    let (n0, n1, _n2) = dims;
    match axis {
        // lines run along axis 0 (stride 1); there are n1*n2 of them, indexed by (i1,i2)
        0 => l * n0,
        // lines run along axis 1 (stride n0); indexed by (i0,i2): base = i0 + i2*n0*n1
        1 => {
            let i0 = l % n0;
            let i2 = l / n0;
            i0 + i2 * n0 * n1
        }
        // lines run along axis 2 (stride n0*n1); indexed by (i0,i1): base = i0 + i1*n0
        2 => l,
        _ => unreachable!(),
    }
}

fn halve_dim(dims: (usize, usize, usize), axis: usize) -> (usize, usize, usize) {
    let mut d = [dims.0, dims.1, dims.2];
    d[axis] /= 2;
    (d[0], d[1], d[2])
}
fn double_dim(dims: (usize, usize, usize), axis: usize) -> (usize, usize, usize) {
    let mut d = [dims.0, dims.1, dims.2];
    d[axis] *= 2;
    (d[0], d[1], d[2])
}

/// One 3D analysis step: returns the 8 subbands (in `SUBBANDS` order) each of
/// halved dimensions.
fn decompose_one(
    a: &[f64],
    dims: (usize, usize, usize),
    lod: &[f64],
    hid: &[f64],
) -> ([Vec<f64>; 8], (usize, usize, usize)) {
    // axis 0
    let (l0, h0) = analyze_axis(a, dims, 0, lod, hid);
    let d1 = halve_dim(dims, 0);
    // axis 1
    let (ll, lh) = analyze_axis(&l0, d1, 1, lod, hid);
    let (hl, hh) = analyze_axis(&h0, d1, 1, lod, hid);
    let d2 = halve_dim(d1, 1);
    // axis 2 — produce (a0,a1,a2) subbands
    let (lll, llh) = analyze_axis(&ll, d2, 2, lod, hid); // a0=0,a1=0
    let (lhl, lhh) = analyze_axis(&lh, d2, 2, lod, hid); // a0=0,a1=1
    let (hll, hlh) = analyze_axis(&hl, d2, 2, lod, hid); // a0=1,a1=0
    let (hhl, hhh) = analyze_axis(&hh, d2, 2, lod, hid); // a0=1,a1=1
    let d3 = halve_dim(d2, 2);

    // map (a0,a1,a2) -> array
    let get = |a0: usize, a1: usize, a2: usize| -> Vec<f64> {
        match (a0, a1, a2) {
            (0, 0, 0) => lll.clone(),
            (0, 0, 1) => llh.clone(),
            (0, 1, 0) => lhl.clone(),
            (0, 1, 1) => lhh.clone(),
            (1, 0, 0) => hll.clone(),
            (1, 0, 1) => hlh.clone(),
            (1, 1, 0) => hhl.clone(),
            (1, 1, 1) => hhh.clone(),
            _ => unreachable!(),
        }
    };
    let subs = std::array::from_fn(|i| {
        let (a0, a1, a2) = SUBBANDS[i];
        get(a0, a1, a2)
    });
    (subs, d3)
}

/// One 3D synthesis step: adjoint of `decompose_one`. `subs` are the 8 subbands
/// in `SUBBANDS` order, each of dims `dims_sub`; returns the parent array.
fn reconstruct_one(
    subs: &[Vec<f64>; 8],
    dims_sub: (usize, usize, usize),
    lod: &[f64],
    hid: &[f64],
) -> Vec<f64> {
    // index helper for (a0,a1,a2)
    let idx = |a0: usize, a1: usize, a2: usize| 4 * a2 + 2 * a0 + a1;
    // combine along axis 2 first
    let d2 = double_dim(dims_sub, 2);
    let ll = synthesize_axis(&subs[idx(0, 0, 0)], &subs[idx(0, 0, 1)], dims_sub, 2, lod, hid);
    let lh = synthesize_axis(&subs[idx(0, 1, 0)], &subs[idx(0, 1, 1)], dims_sub, 2, lod, hid);
    let hl = synthesize_axis(&subs[idx(1, 0, 0)], &subs[idx(1, 0, 1)], dims_sub, 2, lod, hid);
    let hh = synthesize_axis(&subs[idx(1, 1, 0)], &subs[idx(1, 1, 1)], dims_sub, 2, lod, hid);
    // combine along axis 1
    let d1 = double_dim(d2, 1);
    let l0 = synthesize_axis(&ll, &lh, d2, 1, lod, hid);
    let h0 = synthesize_axis(&hl, &hh, d2, 1, lod, hid);
    // combine along axis 0
    synthesize_axis(&l0, &h0, d1, 0, lod, hid)
}

/// Layout of one coefficient cell (subband) inside the flat coefficient vector.
#[derive(Clone, Copy)]
struct Cell {
    dims: (usize, usize, usize),
    offset: usize,
    len: usize,
}

/// A reusable plan for the periodic 3D Daubechies transform of a fixed grid.
pub struct WaveletPlan {
    dims: (usize, usize, usize),
    nlevel: usize,
    lod: Vec<f64>,
    hid: Vec<f64>,
    cells: Vec<Cell>,
    coef_len: usize,
}

impl WaveletPlan {
    /// Build a plan for `db{order}` at `nlevel` levels over `dims`.
    ///
    /// Each dimension must be divisible by `2^nlevel` (periodic exact halving);
    /// pad beforehand otherwise (as recon.m does).
    pub fn new(order: usize, dims: (usize, usize, usize), nlevel: usize) -> Self {
        let (n0, n1, n2) = dims;
        let p = 1usize << nlevel;
        assert!(
            n0 % p == 0 && n1 % p == 0 && n2 % p == 0,
            "wavelet: each dim ({n0},{n1},{n2}) must be divisible by 2^nlevel={p}"
        );
        let lod = lod_filter(order);
        let hid = hid_from_lod(&lod);

        // Coefficient layout: [coarsest approx, coarsest 7 details, ..., finest 7 details].
        let mut level_dims = vec![dims];
        let mut cur = dims;
        for _ in 0..nlevel {
            cur = (cur.0 / 2, cur.1 / 2, cur.2 / 2);
            level_dims.push(cur);
        }
        // coarsest subband dims = level_dims[nlevel]
        let mut cells = Vec::with_capacity(7 * nlevel + 1);
        let mut offset = 0;
        let push = |dims: (usize, usize, usize), cells: &mut Vec<Cell>, offset: &mut usize| {
            let len = dims.0 * dims.1 * dims.2;
            cells.push(Cell { dims, offset: *offset, len });
            *offset += len;
        };
        // approximation at coarsest level
        push(level_dims[nlevel], &mut cells, &mut offset);
        // details from coarsest (level nlevel) to finest (level 1)
        for lev in (1..=nlevel).rev() {
            let sub_dims = level_dims[lev]; // detail subbands at this level have these dims
            for _ in 0..7 {
                push(sub_dims, &mut cells, &mut offset);
            }
        }
        let coef_len = offset;
        Self { dims, nlevel, lod, hid, cells, coef_len }
    }

    /// Length of the flattened coefficient vector.
    pub fn coef_len(&self) -> usize {
        self.coef_len
    }

    /// Grid dimensions this plan operates on.
    pub fn dims(&self) -> (usize, usize, usize) {
        self.dims
    }

    /// Forward transform: image (`prod(dims)`) -> coefficient vector (`coef_len`).
    pub fn forward(&self, image: &[f64]) -> Vec<f64> {
        assert_eq!(image.len(), self.dims.0 * self.dims.1 * self.dims.2);
        let mut coef = vec![0.0; self.coef_len];
        let mut approx = image.to_vec();
        let mut adims = self.dims;

        // For each level, decompose the current approximation; store the 7 details.
        // Detail cells for level `lev` (1-based, finest=1) live at a known offset.
        // We fill from finest to coarsest as we descend.
        // For each descent step (finest first), which 7 cells to write. The
        // coefficient vector orders details coarsest-first, so descent step 0
        // (the finest level) writes the last group.
        let detail_cell_ranges: Vec<[usize; 7]> = (0..self.nlevel)
            .map(|step| {
                let group_start = 1 + 7 * (self.nlevel - (step + 1));
                std::array::from_fn(|j| group_start + j)
            })
            .collect();

        for cell_ids in &detail_cell_ranges {
            let (subs, sub_dims) = decompose_one(&approx, adims, &self.lod, &self.hid);
            // subs[0] is AAA (approx); subs[1..8] are the 7 details in SUBBANDS order
            for (j, &cid) in cell_ids.iter().enumerate() {
                let cell = self.cells[cid];
                debug_assert_eq!(cell.dims, sub_dims);
                coef[cell.offset..cell.offset + cell.len].copy_from_slice(&subs[j + 1]);
            }
            approx = subs[0].clone();
            adims = sub_dims;
        }
        // final approximation -> cell 0
        let cell0 = self.cells[0];
        coef[cell0.offset..cell0.offset + cell0.len].copy_from_slice(&approx);
        coef
    }

    /// Inverse transform: coefficient vector (`coef_len`) -> image (`prod(dims)`).
    pub fn inverse(&self, coef: &[f64]) -> Vec<f64> {
        assert_eq!(coef.len(), self.coef_len);
        // Start from the coarsest approximation and ascend.
        let cell0 = self.cells[0];
        let mut approx = coef[cell0.offset..cell0.offset + cell0.len].to_vec();
        let mut adims = cell0.dims;

        // Ascend: coarsest level first. Coarsest detail group is cells[1..8].
        for lev in (1..=self.nlevel).rev() {
            // detail group for this level
            let group_start = 1 + 7 * (self.nlevel - lev);
            let mut subs: [Vec<f64>; 8] = std::array::from_fn(|_| Vec::new());
            subs[0] = approx.clone();
            for j in 0..7 {
                let cell = self.cells[group_start + j];
                subs[j + 1] = coef[cell.offset..cell.offset + cell.len].to_vec();
            }
            approx = reconstruct_one(&subs, adims, &self.lod, &self.hid);
            adims = double_dim(double_dim(double_dim(adims, 0), 1), 2);
        }
        approx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms(a: &[f64]) -> f64 {
        (a.iter().map(|v| v * v).sum::<f64>() / a.len() as f64).sqrt()
    }

    #[test]
    fn perfect_reconstruction_db1() {
        let dims = (8, 8, 8);
        let plan = WaveletPlan::new(1, dims, 3);
        let img: Vec<f64> = (0..512).map(|i| ((i * 7 % 13) as f64 - 6.0).sin()).collect();
        let coef = plan.forward(&img);
        let rec = plan.inverse(&coef);
        let err: f64 = img
            .iter()
            .zip(&rec)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-10, "db1 PR err={err}");
    }

    #[test]
    fn perfect_reconstruction_db2() {
        let dims = (16, 8, 24);
        let plan = WaveletPlan::new(2, dims, 3);
        let n = dims.0 * dims.1 * dims.2;
        let img: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.1).cos() + (i % 5) as f64).collect();
        let coef = plan.forward(&img);
        let rec = plan.inverse(&coef);
        let err: f64 = img
            .iter()
            .zip(&rec)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-9, "db2 PR err={err}");
    }

    #[test]
    fn orthonormal_norm_preserved() {
        // Orthonormal transform: ||coef|| == ||image||.
        let dims = (8, 16, 8);
        for order in [1, 2] {
            let plan = WaveletPlan::new(order, dims, 2);
            let n = dims.0 * dims.1 * dims.2;
            let img: Vec<f64> = (0..n).map(|i| ((i * 3 % 17) as f64) - 8.0).collect();
            let coef = plan.forward(&img);
            assert_eq!(coef.len(), n, "orthonormal transform preserves length");
            assert!((rms(&coef) - rms(&img)).abs() / rms(&img) < 1e-10, "order {order} norm");
        }
    }
}

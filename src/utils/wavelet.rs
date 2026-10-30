//! Separable 3D periodic Daubechies wavelet transform (db1–db4).
//!
//! A faithful reimplementation of the two periodic transforms the ported methods use:
//! MATLAB's `wavedec3`/`waverec3` in **periodic (`'per'`) mode** (AMP-PE, a MATLAB original) and
//! PyWavelets' `wavedecn`/`waverecn` in **`mode="periodization"`** (WaveSep, a Python original).
//! The two differ only in where the filter sits on the input ([`Periodization`]), which gives
//! differently aligned bases for every order but db1.
//!
//! The transform is orthonormal, so synthesis is the exact adjoint of analysis
//! (perfect reconstruction). Arrays are flat `Vec<f64>` in column-major
//! (Fortran) order — `index = i0 + i1*n0 + i2*n0*n1` — matching MATLAB, so the
//! flattened coefficient vector lines up element-for-element with
//! `wavedec3` + the toolbox's coefficient concatenation.
//!
//! ## Conventions
//! One analysis step along an axis of length `N` (even) with filter `f` (length
//! `lf`) produces `N/2` coefficients:
//! ```text
//!   c[k] = sum_{n=0}^{lf-1} f[n] * x[(2k + s - n) mod N]
//! ```
//! with the low-pass filter `LoD` giving the approximation and the high-pass
//! `HiD[k] = (-1)^(k+1) * LoD[lf-1-k]` giving the detail (PyWavelets' `dec_hi`). The shift `s` is:
//! - [`Periodization::Matlab`]: `s = 1` (reverse-engineered from MATLAB `wavedec3` `'mode','per'`);
//! - [`Periodization::PyWavelets`]: `s = lf/2`. PyWavelets'
//!   `downsampling_convolution_periodization` (`convolution.template.c`) starts its output at input
//!   index `F/2` and steps by 2. Checked against `pywt.dwt(..., mode="periodization")` 1.x for
//!   db1–db4 (2026-10-30, the verified QSM-CI `wavesep` image): every coefficient matches with
//!   `s = lf/2` and no other shift.
//!
//! The 3D step applies this along axis 0, then 1, then 2; the 8 subbands are ordered by
//! `4*a2 + 2*a0 + a1` (a=0 low, a=1 high on that axis). Multilevel decomposition
//! recurses on the all-low (approximation) subband. The coefficient vector is
//! `[coarsest approximation, coarsest 7 details, ..., finest 7 details]` (MATLAB's order; the
//! ported methods only threshold or mask coefficients elementwise, so PyWavelets' dict layout
//! needs no separate ordering).
//!
//! **Why this matters (astewartau/QSM.rs#327):** WaveSep used the MATLAB alignment until
//! 2026-10-30. Its db4 periodized soft-threshold step then differed from PyWavelets' by 3.4 %
//! relative on a smooth 32³ field, which put WaveSep 10 % NRMSE from the original on a 24³ case.
//! With [`Periodization::PyWavelets`] the step matches PyWavelets to rounding.
//!
//! Each 1D-along-axis pass is parallelized over its independent "lines" (with the
//! `parallel` feature). Distinct lines write disjoint strided index sets of the
//! output, so per-line raw-pointer writes are data-race free.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Where a periodic analysis step centres its filter: the two originals' conventions (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Periodization {
    /// MATLAB `wavedec3`/`waverec3` with `'mode','per'`: `c[k] = Σ f[n] x[(2k + 1 − n) mod N]`.
    Matlab,
    /// PyWavelets `wavedecn`/`waverecn` with `mode="periodization"`:
    /// `c[k] = Σ f[n] x[(2k + lf/2 − n) mod N]`.
    PyWavelets,
}

impl Periodization {
    /// The shift `s` of the module docs for a filter of length `lf`.
    fn shift(self, lf: usize) -> usize {
        match self {
            Periodization::Matlab => 1,
            Periodization::PyWavelets => lf / 2,
        }
    }
}

/// Daubechies orders this module implements (db1–db4). Anything else has no filter.
pub const SUPPORTED_ORDERS: std::ops::RangeInclusive<usize> = 1..=4;

/// `Ok` if `order` is in [`SUPPORTED_ORDERS`], otherwise an error naming `param` (e.g.
/// `"AMP-PE wave_order"`), the value and the range. Callers that take a wavelet order check it
/// with this before building a [`WaveletPlan`], which panics on an unsupported order.
pub fn check_order(param: &str, order: usize) -> Result<(), crate::error::InputError> {
    if SUPPORTED_ORDERS.contains(&order) {
        Ok(())
    } else {
        Err(crate::error::InputError::param(
            param,
            order,
            &format!("from {} to {} (db1–db4)", SUPPORTED_ORDERS.start(), SUPPORTED_ORDERS.end()),
        ))
    }
}

/// Daubechies analysis low-pass filter `LoD` for the given order (1..=4), as the original's library
/// has it.
///
/// - [`Periodization::PyWavelets`]: PyWavelets' `Wavelet(f"db{order}").dec_lo`, printed with `repr`
///   (PyWavelets 1.8.0 in the verified QSM-CI `wavesep` image, 2026-10-30).
/// - [`Periodization::Matlab`]: the taps QSM.rs has used for AMP-PE since its port (MATLAB
///   `wfilters` order). For db1 and db2 they are PyWavelets' to the last bit; for db3 and db4 they
///   differ from PyWavelets' by ~1e-12 (not checked against MATLAB's `wfilters` itself; AMP-PE's
///   authors use db1 and db2).
fn lod_filter(order: usize, periodization: Periodization) -> Vec<f64> {
    match (order, periodization) {
        (1, _) => {
            let s = std::f64::consts::FRAC_1_SQRT_2;
            vec![s, s]
        }
        (2, _) => vec![
            -0.12940952255126037,
            0.2241438680420134,
            0.8365163037378079,
            0.48296291314453416,
        ],
        (3, Periodization::Matlab) => vec![
            0.035226291882100656,
            -0.08544127388224149,
            -0.13501102001039084,
            0.4598775021193313,
            0.8068915093133388,
            0.3326705529509569,
        ],
        (3, Periodization::PyWavelets) => vec![
            0.03522629188570953,
            -0.08544127388202666,
            -0.13501102001025458,
            0.45987750211849154,
            0.8068915093110925,
            0.33267055295008263,
        ],
        (4, Periodization::Matlab) => vec![
            -0.010597401784997278,
            0.032883011666982945,
            0.030841381835986965,
            -0.18703481171888114,
            -0.02798376941698385,
            0.6308807679295904,
            0.7148465705525415,
            0.23037781330885523,
        ],
        (4, Periodization::PyWavelets) => vec![
            -0.010597401785069032,
            0.0328830116668852,
            0.030841381835560764,
            -0.18703481171909309,
            -0.027983769416859854,
            0.6308807679298589,
            0.7148465705529157,
            0.2303778133088965,
        ],
        _ => panic!("wavelet: only db1..db4 (order 1..4) are supported, got {order}"),
    }
}

/// Maximum useful decomposition level for a periodic transform, matching
/// `pywt.dwt_max_level(data_len, filter_len)`.
///
/// `filter_len` is the analysis filter length (`2*order` for `db{order}`).
/// Returns `floor(log2(data_len / (filter_len - 1)))`, clamped at 0.
pub fn dwt_max_level(data_len: usize, filter_len: usize) -> usize {
    if filter_len <= 1 || data_len < filter_len {
        return 0;
    }
    let ratio = data_len as f64 / (filter_len - 1) as f64;
    ratio.log2().floor().max(0.0) as usize
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
    shift: usize,
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

    let (os, _, _) = axis_geometry(out_dims, axis);
    // Wraparound (mod n) is only needed near the boundaries (`wrap_range`); the bulk
    // uses direct indexing, avoiding a division per tap.
    let (k_lo, k_hi) = wrap_range(half, lf, shift);

    // Raw output pointers as usize (Send). Distinct lines `l` write disjoint
    // index sets {obase(l) + k*os}, so the writes below never race.
    let low_ptr = low.as_mut_ptr() as usize;
    let high_ptr = high.as_mut_ptr() as usize;
    let process = |l: usize, line: &mut [f64]| {
        let base = line_base(l, dims, axis);
        for (t, slot) in line.iter_mut().enumerate() {
            *slot = inp[base + t * stride];
        }
        let obase = line_base(l, out_dims, axis);
        let lp = low_ptr as *mut f64;
        let hp = high_ptr as *mut f64;
        for k in 0..half {
            let center = 2 * k + shift;
            let mut ca = 0.0;
            let mut cd = 0.0;
            if k < k_lo || k >= k_hi {
                for nn in 0..lf {
                    let idx = (center as isize - nn as isize).rem_euclid(n as isize) as usize;
                    let v = line[idx];
                    ca += lod[nn] * v;
                    cd += hid[nn] * v;
                }
            } else {
                for nn in 0..lf {
                    let v = line[center - nn];
                    ca += lod[nn] * v;
                    cd += hid[nn] * v;
                }
            }
            // SAFETY: `obase + k*os` is in-bounds for `out_len` and unique to this
            // line `l`, so no two concurrent invocations touch the same index.
            unsafe {
                *lp.add(obase + k * os) = ca;
                *hp.add(obase + k * os) = cd;
            }
        }
    };
    run_lines(num_lines, n, process);
    (low, high)
}

/// Coefficients `k` in `k_lo..k_hi` have every tap `2k + shift − nn` (`nn < lf`) inside
/// `0..2·half`; the others need the periodic wrap.
fn wrap_range(half: usize, lf: usize, shift: usize) -> (usize, usize) {
    let n = 2 * half;
    // 2k + shift ≥ lf − 1  ⇔  k ≥ ceil((lf − 1 − shift) / 2)
    let k_lo = ((lf - 1).saturating_sub(shift)).div_ceil(2).min(half);
    // 2k + shift ≤ n − 1  ⇔  k < ceil((n − shift) / 2)
    let k_hi = n.saturating_sub(shift).div_ceil(2).clamp(k_lo, half);
    (k_lo, k_hi)
}

/// Run a per-line closure over `num_lines`, giving each invocation a scratch
/// buffer of length `n`. Parallel over lines with the `parallel` feature.
#[cfg(feature = "parallel")]
fn run_lines(num_lines: usize, n: usize, process: impl Fn(usize, &mut [f64]) + Sync + Send) {
    (0..num_lines)
        .into_par_iter()
        .for_each_init(|| vec![0.0f64; n], |line, l| process(l, line));
}

#[cfg(not(feature = "parallel"))]
fn run_lines(num_lines: usize, n: usize, mut process: impl FnMut(usize, &mut [f64])) {
    let mut line = vec![0.0f64; n];
    for l in 0..num_lines {
        process(l, &mut line);
    }
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
    shift: usize,
) -> Vec<f64> {
    let half = [dims_half.0, dims_half.1, dims_half.2][axis];
    let n = half * 2;
    let lf = lod.len();
    let out_dims = double_dim(dims_half, axis);
    let out_len = out_dims.0 * out_dims.1 * out_dims.2;
    let mut out = vec![0.0; out_len];

    let (in_stride, num_lines, _) = axis_geometry(dims_half, axis);
    let (out_stride, _, _) = axis_geometry(out_dims, axis);

    // Wraparound is only needed for the boundary coefficients (mirrors
    // `analyze_axis`); the rest scatter with direct indexing.
    let (k_lo, k_hi) = wrap_range(half, lf, shift);
    // Distinct lines write disjoint output index sets {obase(l) + t*out_stride}.
    let out_ptr = out.as_mut_ptr() as usize;
    let process = |l: usize, line: &mut [f64]| {
        for slot in line.iter_mut() {
            *slot = 0.0;
        }
        let ibase = line_base(l, dims_half, axis);
        for k in 0..half {
            let ca = low[ibase + k * in_stride];
            let cd = high[ibase + k * in_stride];
            let center = 2 * k + shift;
            if k < k_lo || k >= k_hi {
                for nn in 0..lf {
                    let idx = (center as isize - nn as isize).rem_euclid(n as isize) as usize;
                    line[idx] += lod[nn] * ca + hid[nn] * cd;
                }
            } else {
                for nn in 0..lf {
                    line[center - nn] += lod[nn] * ca + hid[nn] * cd;
                }
            }
        }
        let obase = line_base(l, out_dims, axis);
        let op = out_ptr as *mut f64;
        for (t, &v) in line.iter().enumerate() {
            // SAFETY: `obase + t*out_stride` is in-bounds and unique to line `l`.
            unsafe {
                *op.add(obase + t * out_stride) = v;
            }
        }
    };
    run_lines(num_lines, n, process);
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
    sh: usize,
) -> ([Vec<f64>; 8], (usize, usize, usize)) {
    // axis 0
    let (l0, h0) = analyze_axis(a, dims, 0, lod, hid, sh);
    let d1 = halve_dim(dims, 0);
    // axis 1
    let (ll, lh) = analyze_axis(&l0, d1, 1, lod, hid, sh);
    let (hl, hh) = analyze_axis(&h0, d1, 1, lod, hid, sh);
    let d2 = halve_dim(d1, 1);
    // axis 2 — produce (a0,a1,a2) subbands
    let (lll, llh) = analyze_axis(&ll, d2, 2, lod, hid, sh); // a0=0,a1=0
    let (lhl, lhh) = analyze_axis(&lh, d2, 2, lod, hid, sh); // a0=0,a1=1
    let (hll, hlh) = analyze_axis(&hl, d2, 2, lod, hid, sh); // a0=1,a1=0
    let (hhl, hhh) = analyze_axis(&hh, d2, 2, lod, hid, sh); // a0=1,a1=1
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
    sh: usize,
) -> Vec<f64> {
    // index helper for (a0,a1,a2)
    let idx = |a0: usize, a1: usize, a2: usize| 4 * a2 + 2 * a0 + a1;
    // combine along axis 2 first
    let d2 = double_dim(dims_sub, 2);
    let ll = synthesize_axis(&subs[idx(0, 0, 0)], &subs[idx(0, 0, 1)], dims_sub, 2, lod, hid, sh);
    let lh = synthesize_axis(&subs[idx(0, 1, 0)], &subs[idx(0, 1, 1)], dims_sub, 2, lod, hid, sh);
    let hl = synthesize_axis(&subs[idx(1, 0, 0)], &subs[idx(1, 0, 1)], dims_sub, 2, lod, hid, sh);
    let hh = synthesize_axis(&subs[idx(1, 1, 0)], &subs[idx(1, 1, 1)], dims_sub, 2, lod, hid, sh);
    // combine along axis 1
    let d1 = double_dim(d2, 1);
    let l0 = synthesize_axis(&ll, &lh, d2, 1, lod, hid, sh);
    let h0 = synthesize_axis(&hl, &hh, d2, 1, lod, hid, sh);
    // combine along axis 0
    synthesize_axis(&l0, &h0, d1, 0, lod, hid, sh)
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
    shift: usize,
    cells: Vec<Cell>,
    coef_len: usize,
}

impl WaveletPlan {
    /// Build a plan for `db{order}` at `nlevel` levels over `dims`, with the original's
    /// periodization convention (`Matlab` for a MATLAB original, `PyWavelets` for a Python one).
    ///
    /// Each dimension must be divisible by `2^nlevel` (periodic exact halving);
    /// pad beforehand otherwise (as recon.m does).
    pub fn new(order: usize, dims: (usize, usize, usize), nlevel: usize, periodization: Periodization) -> Self {
        let (n0, n1, n2) = dims;
        let p = 1usize << nlevel;
        assert!(
            n0 % p == 0 && n1 % p == 0 && n2 % p == 0,
            "wavelet: each dim ({n0},{n1},{n2}) must be divisible by 2^nlevel={p}"
        );
        let lod = lod_filter(order, periodization);
        let hid = hid_from_lod(&lod);
        let shift = periodization.shift(lod.len());

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
        Self { dims, nlevel, lod, hid, shift, cells, coef_len }
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
            let (subs, sub_dims) = decompose_one(&approx, adims, &self.lod, &self.hid, self.shift);
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
            approx = reconstruct_one(&subs, adims, &self.lod, &self.hid, self.shift);
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
        let plan = WaveletPlan::new(1, dims, 3, Periodization::Matlab);
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
        let plan = WaveletPlan::new(2, dims, 3, Periodization::Matlab);
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
    fn perfect_reconstruction_db4() {
        let dims = (16, 16, 32);
        let plan = WaveletPlan::new(4, dims, 2, Periodization::Matlab);
        let n = dims.0 * dims.1 * dims.2;
        let img: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.07).sin() + (i % 7) as f64).collect();
        let coef = plan.forward(&img);
        let rec = plan.inverse(&coef);
        let err: f64 = img
            .iter()
            .zip(&rec)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-9, "db4 PR err={err}");
    }

    /// PyWavelets 1.8.0 (the verified QSM-CI `wavesep` image, 2026-10-30) on
    /// `x = sin(0.37 i0) + cos(0.21 i1 + 0.5 i2) + 0.1 ((7 i0 + 3 i1 + 5 i2) mod 11)`, 16×16×32,
    /// `pywt.wavedecn(x, f"db{order}", mode="periodization", level=2)`, then every coefficient
    /// soft-thresholded at 0.05 (`pywt.threshold`) and `waverecn`. Values printed with `repr`.
    #[test]
    fn pywavelets_periodization_matches_pywt() {
        let dims = (16, 16, 32);
        let x: Vec<f64> = (0..dims.0 * dims.1 * dims.2)
            .map(|i| {
                let (i0, i1, i2) = ((i % 16) as f64, ((i / 16) % 16) as f64, (i / 256) as f64);
                (0.37 * i0).sin() + (0.21 * i1 + 0.5 * i2).cos() + 0.1 * ((7.0 * i0 + 3.0 * i1 + 5.0 * i2) % 11.0)
            })
            .collect();
        // (order, Σ|c|, approx [0,0,0] [1,2,3] [3,1,7], level-2 'dad' [1,2,3], level-1 'add' [5,6,7],
        //  thresholded round trip [0,0,0] [5,9,20] [15,15,31], ‖y − x‖)
        #[allow(clippy::type_complexity)]
        let refs: [(usize, f64, [f64; 3], f64, f64, [f64; 3], f64); 2] = [
            (2, 3878.236296169174, [-0.13178516932194556, 16.474964093457814, -3.862965195840758],
             0.01828616653364562, 0.2807617040724214,
             [0.989116521921333, 2.512318887842404, 1.0605870412950476], 4.4182932347541986),
            (4, 3491.3456776299936, [-8.195945496463047, -8.61732149960479, 10.034554937571587],
             -0.005590978474005846, 0.34047362050470464,
             [1.0289256074515492, 2.5090971698624047, 1.0523496272652415], 4.315652189415173),
        ];
        let close = |a: f64, b: f64, what: &str| assert!((a - b).abs() <= 1e-12 * b.abs().max(1.0), "{what}: {a} vs pywt {b}");
        for (order, l1, approx, dad, add, rec, dist) in refs {
            let plan = WaveletPlan::new(order, dims, 2, Periodization::PyWavelets);
            let c = plan.forward(&x);
            close(c.iter().map(|v| v.abs()).sum(), l1, "Σ|c|");
            let at = |cell: usize, (i0, i1, i2): (usize, usize, usize)| {
                let Cell { dims: (d0, d1, _), offset, .. } = plan.cells[cell];
                c[offset + i0 + i1 * d0 + i2 * d0 * d1]
            };
            for (p, v) in [(0, 0, 0), (1, 2, 3), (3, 1, 7)].into_iter().zip(approx) {
                close(at(0, p), v, "approximation");
            }
            // 'dad' = (a0, a1, a2) = (1, 0, 1): SUBBANDS[6]; coarsest group starts at cell 1.
            close(at(1 + 5, (1, 2, 3)), dad, "level-2 dad");
            // 'add' = (0, 1, 1): SUBBANDS[5]; finest group starts at cell 8.
            close(at(8 + 4, (5, 6, 7)), add, "level-1 add");
            let st: Vec<f64> = c.iter().map(|&v| v.signum() * (v.abs() - 0.05).max(0.0)).collect();
            let y = plan.inverse(&st);
            for ((i0, i1, i2), v) in [(0, 0, 0), (5, 9, 20), (15, 15, 31)].into_iter().zip(rec) {
                close(y[i0 + 16 * i1 + 256 * i2], v, "round trip");
            }
            close(y.iter().zip(&x).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt(), dist, "‖y − x‖");
            // The MATLAB alignment is a different basis for db2/db4.
            let m = WaveletPlan::new(order, dims, 2, Periodization::Matlab).forward(&x);
            assert!((m.iter().map(|v| v.abs()).sum::<f64>() - l1).abs() > 1e-6 * l1, "db{order}: MATLAB alignment should differ");
        }
    }

    #[test]
    fn dwt_max_level_matches_pywt() {
        // pywt.dwt_max_level reference values: (data_len, filter_len) -> level
        assert_eq!(dwt_max_level(256, 8), 5); // floor(log2(256/7)) = 5
        assert_eq!(dwt_max_level(16, 8), 1); // floor(log2(16/7)) = 1
        assert_eq!(dwt_max_level(7, 8), 0); // data_len < filter_len
        assert_eq!(dwt_max_level(64, 4), 4); // floor(log2(64/3)) = 4
    }

    #[test]
    fn orthonormal_norm_preserved() {
        // Orthonormal transform: ||coef|| == ||image||.
        let dims = (8, 16, 8);
        for order in [1, 2, 3, 4] {
            for per in [Periodization::Matlab, Periodization::PyWavelets] {
                let plan = WaveletPlan::new(order, dims, 2, per);
                let n = dims.0 * dims.1 * dims.2;
                let img: Vec<f64> = (0..n).map(|i| ((i * 3 % 17) as f64) - 8.0).collect();
                let coef = plan.forward(&img);
                assert_eq!(coef.len(), n, "orthonormal transform preserves length");
                assert!((rms(&coef) - rms(&img)).abs() / rms(&img) < 1e-10, "order {order} {per:?} norm");
                let rec = plan.inverse(&coef);
                let err = img.iter().zip(&rec).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
                assert!(err < 1e-9, "order {order} {per:?} PR err={err}");
            }
        }
    }

    #[test]
    fn check_order_accepts_db1_to_db4_only() {
        for o in 1..=4 {
            assert!(check_order("x", o).is_ok());
            assert!(!lod_filter(o, Periodization::Matlab).is_empty() && !lod_filter(o, Periodization::PyWavelets).is_empty());
        }
        for o in [0, 5, 40] {
            let err = check_order("WaveSep wavelet_order", o).unwrap_err();
            assert!(err.0.contains(&format!("WaveSep wavelet_order = {o}")) && err.0.ends_with("must be from 1 to 4 (db1–db4)"), "{err}");
        }
    }
}

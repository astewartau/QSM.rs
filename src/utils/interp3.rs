//! MATLAB `interp3` on a uniform grid, at scattered query points: `'spline'` and `'linear'`.
//!
//! The χ-separation toolbox's `chi_sepnet_general` resamples its inputs onto a rotated 1 mm grid
//! with `interp3(X, Y, Z, V, Xq, Yq, Zq, method)` (field and R2′/R2* `'spline'`, mask `'linear'`)
//! and maps its outputs back with `'spline'`; [`crate::utils::chisep_frame`] reproduces that.
//! Outside the sample grid the two methods differ, as in MATLAB: `'spline'` extrapolates (the end
//! cells' cubics continued), `'linear'` gives `NaN`. Measured against MATLAB R2023b's `interp3` on
//! the toolbox's own rotated, partly extrapolating resampling of a 64×60×48 volume at
//! 0.8×0.9×1.1 mm: this spline matches it to 6e-14 relative (QSM.rs#377, 2026-11-03).
//!
//! **`'spline'`** is the tensor-product cubic spline with MATLAB `spline`'s end conditions
//! (not-a-knot) along each axis. It is linear in the data and separable, so it is built from the
//! second-derivative arrays of the 1-D splines: with `M_d` the operator that maps the samples of
//! every line along axis `d` to their spline's second derivatives at the nodes, the eight arrays
//! `V, M_x V, M_y V, M_x M_y V, M_z V, …, M_x M_y M_z V` define the 3-D spline exactly, and on a
//! cell it is the product of the 1-D cubic pieces
//! `s(t) = (1−t)·v_j + t·v_{j+1} + h²/6·[((1−t)³ − (1−t))·M_j + (t³ − t)·M_{j+1}]`
//! (64 terms per query, 8 per array). The arrays are made and used one at a time, so the memory
//! is a few copies of the input rather than eight. Lines of 3 samples get the parabola through
//! them and lines of 2 a straight line, as `spline` does. MATLAB stores the spline in another
//! basis (`pp` coefficients), so the two agree to rounding, not bit for bit.
//!
//! **`'linear'`** is trilinear interpolation in the cell that holds the query.
//!
//! Coordinates: axis `d` has nodes `origin[d] + i·step[d]`, `i = 0..n[d]`, and the data are
//! column-major `(n[0], n[1], n[2])` as everywhere in the crate. (MATLAB's `interp3` names its
//! first coordinate `X` but pairs it with the *second* array dimension; a caller porting a
//! MATLAB call maps those itself.)

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// A uniform sample grid: `n[d]` nodes at `origin[d] + i·step[d]` along axis `d`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UniformAxes {
    pub n: [usize; 3],
    pub origin: [f64; 3],
    pub step: [f64; 3],
}

impl UniformAxes {
    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.n[0] * self.n[1] * self.n[2]
    }

    /// Whether the grid has no nodes.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The coordinates of node `(i, j, k)`.
    pub fn node(&self, i: usize, j: usize, k: usize) -> [f64; 3] {
        [
            self.origin[0] + i as f64 * self.step[0],
            self.origin[1] + j as f64 * self.step[1],
            self.origin[2] + k as f64 * self.step[2],
        ]
    }

    /// The coordinates of node `idx` in column-major order.
    pub fn node_at(&self, idx: usize) -> [f64; 3] {
        let (n0, n1) = (self.n[0], self.n[1]);
        self.node(idx % n0, (idx / n0) % n1, idx / (n0 * n1))
    }

    /// Fractional node index of coordinate `q` along axis `d`.
    fn index(&self, d: usize, q: f64) -> f64 {
        (q - self.origin[d]) / self.step[d]
    }

    /// The cell along axis `d` nearest coordinate `q` and the fraction across it, which falls
    /// outside `[0, 1]` beyond the grid's ends (spline extrapolation continues the end cubics).
    fn cell_extrapolated(&self, d: usize, q: f64) -> (usize, f64) {
        let n = self.n[d];
        let u = self.index(d, q);
        if n == 1 {
            return (0, 0.0);
        }
        let j = (u.floor().max(0.0) as usize).min(n - 2);
        (j, u - j as f64)
    }

    /// The cell along axis `d` that holds coordinate `q`, and the fraction across it; `None`
    /// outside `[origin, origin + (n−1)·step]`.
    fn cell(&self, d: usize, q: f64) -> Option<(usize, f64)> {
        let u = self.index(d, q);
        if !(u >= 0.0 && u <= (self.n[d] - 1) as f64) {
            return None;
        }
        Some(self.cell_extrapolated(d, q))
    }
}

/// Second derivatives at the nodes of the not-a-knot cubic spline through `v` (spacing `h`):
/// MATLAB `spline`'s end conditions. Writes into `m` (same length as `v`).
fn not_a_knot_second_derivatives(v: &[f64], h: f64, m: &mut [f64]) {
    let n = v.len();
    match n {
        0..=2 => m.iter_mut().for_each(|x| *x = 0.0),
        3 => {
            // The parabola through three points: constant second derivative.
            let c = (v[0] - 2.0 * v[1] + v[2]) / (h * h);
            m.iter_mut().for_each(|x| *x = c);
        }
        _ => {
            // Interior rows: M_{i−1} + 4 M_i + M_{i+1} = r_i, r_i = 6 (v_{i−1} − 2 v_i + v_{i+1}) / h².
            // Not-a-knot (third derivative continuous at nodes 1 and n−2): M_0 − 2 M_1 + M_2 = 0 and
            // M_{n−3} − 2 M_{n−2} + M_{n−1} = 0. Substituting M_0 = 2 M_1 − M_2 into row 1 leaves
            // 6 M_1 = r_1, and likewise 6 M_{n−2} = r_{n−2}; rows 2..n−3 are then tridiagonal.
            let s = 6.0 / (h * h);
            let r = |i: usize| s * (v[i - 1] - 2.0 * v[i] + v[i + 1]);
            m[1] = r(1) / 6.0;
            m[n - 2] = r(n - 2) / 6.0;
            if n > 4 {
                // Thomas algorithm on rows 2..=n−3 (diagonal 4, off-diagonals 1), in place in m.
                let k = n - 4;
                let mut cp = vec![0.0; k];
                for (q, i) in (2..n - 2).enumerate() {
                    let mut rhs = r(i);
                    if i == 2 {
                        rhs -= m[1];
                    }
                    if i == n - 3 {
                        rhs -= m[n - 2];
                    }
                    let (den, prev) = if q == 0 { (4.0, 0.0) } else { (4.0 - cp[q - 1], m[i - 1]) };
                    cp[q] = 1.0 / den;
                    m[i] = (rhs - prev) / den;
                }
                for q in (0..k - 1).rev() {
                    let i = q + 2;
                    m[i] -= cp[q] * m[i + 1];
                }
            }
            m[0] = 2.0 * m[1] - m[2];
            m[n - 1] = 2.0 * m[n - 2] - m[n - 3];
        }
    }
}

/// Apply the second-derivative operator along `axis` to every line of `src`.
fn second_derivative_along(src: &[f64], n: [usize; 3], axis: usize, h: f64) -> Vec<f64> {
    let len = n[axis];
    let stride = [1, n[0], n[0] * n[1]][axis];
    let (oa, ob) = [(1, 2), (0, 2), (0, 1)][axis];
    let mut out = vec![0.0; src.len()];
    let mut line = vec![0.0; len];
    let mut m = vec![0.0; len];
    for b in 0..n[ob] {
        for a in 0..n[oa] {
            let mut idx = [0usize; 3];
            idx[oa] = a;
            idx[ob] = b;
            let base = idx[0] + n[0] * (idx[1] + n[1] * idx[2]);
            for (i, l) in line.iter_mut().enumerate() {
                *l = src[base + i * stride];
            }
            not_a_knot_second_derivatives(&line, h, &mut m);
            for (i, &mv) in m.iter().enumerate() {
                out[base + i * stride] = mv;
            }
        }
    }
    out
}

/// A query point source: the coordinates of point `q`.
pub type Query<'a> = dyn Fn(usize) -> [f64; 3] + Sync + 'a;

/// Add the terms of the spline that read array `bits` (bit `d` set: the second-derivative
/// operator applied along axis `d`) at every query.
fn accumulate(acc: &mut [f64], array: &[f64], axes: &UniformAxes, bits: usize, query: &Query) {
    let n = axes.n;
    let h2: [f64; 3] = std::array::from_fn(|d| axes.step[d] * axes.step[d] / 6.0);
    let term = |q: usize, out: &mut f64| {
        let p = query(q);
        if !p.iter().all(|v| v.is_finite()) {
            *out = f64::NAN;
            return;
        }
        // Per axis, the two nodes' weights for this array's kind (value or second derivative).
        let mut j = [0usize; 3];
        let mut w = [[0.0f64; 2]; 3];
        for d in 0..3 {
            let (jd, t) = axes.cell_extrapolated(d, p[d]);
            let u = 1.0 - t;
            j[d] = jd;
            w[d] = if bits & (1 << d) == 0 { [u, t] } else { [h2[d] * (u * u * u - u), h2[d] * (t * t * t - t)] };
        }
        let mut s = 0.0;
        for (dz, wz) in w[2].iter().enumerate() {
            for (dy, wy) in w[1].iter().enumerate() {
                let base = n[0] * ((j[1] + dy) + n[1] * (j[2] + dz)) + j[0];
                s += wy * wz * (w[0][0] * array[base] + w[0][1] * array[base + 1]);
            }
        }
        *out += s;
    };
    #[cfg(feature = "parallel")]
    acc.par_iter_mut().enumerate().for_each(|(q, o)| term(q, o));
    #[cfg(not(feature = "parallel"))]
    acc.iter_mut().enumerate().for_each(|(q, o)| term(q, o));
}

/// MATLAB `interp3(…, 'spline')` of `data` (on `axes`) at `n_query` points, `query(q)` giving the
/// coordinates of point `q`; extrapolates beyond the grid as MATLAB does. A non-finite coordinate
/// gives `NaN`.
///
/// # Panics
/// If `data.len()` does not match `axes`, or an axis has fewer than 2 samples.
pub fn spline_resample(data: &[f64], axes: &UniformAxes, n_query: usize, query: &Query) -> Vec<f64> {
    let n = axes.n;
    assert_eq!(data.len(), axes.len(), "data length must match the grid");
    assert!(n.iter().all(|&k| k >= 2), "spline interpolation needs at least 2 samples per axis");
    let h = axes.step;
    let mut acc = vec![0.0; n_query];
    // Each array is made from one already in hand and dropped as soon as nothing needs it:
    // V → M_x V → M_y M_x V → M_z M_y M_x V; M_x V → M_z M_x V; V → M_y V → M_z M_y V; V → M_z V.
    accumulate(&mut acc, data, axes, 0b000, query);
    {
        let mx = second_derivative_along(data, n, 0, h[0]);
        accumulate(&mut acc, &mx, axes, 0b001, query);
        {
            let mxy = second_derivative_along(&mx, n, 1, h[1]);
            accumulate(&mut acc, &mxy, axes, 0b011, query);
            let mxyz = second_derivative_along(&mxy, n, 2, h[2]);
            accumulate(&mut acc, &mxyz, axes, 0b111, query);
        }
        let mxz = second_derivative_along(&mx, n, 2, h[2]);
        accumulate(&mut acc, &mxz, axes, 0b101, query);
    }
    {
        let my = second_derivative_along(data, n, 1, h[1]);
        accumulate(&mut acc, &my, axes, 0b010, query);
        let myz = second_derivative_along(&my, n, 2, h[2]);
        accumulate(&mut acc, &myz, axes, 0b110, query);
    }
    let mz = second_derivative_along(data, n, 2, h[2]);
    accumulate(&mut acc, &mz, axes, 0b100, query);
    acc
}

/// MATLAB `interp3(…, 'linear')` of `data` (on `axes`) at `n_query` points; `NaN` outside the
/// grid.
pub fn linear_resample(data: &[f64], axes: &UniformAxes, n_query: usize, query: &Query) -> Vec<f64> {
    assert_eq!(data.len(), axes.len(), "data length must match the grid");
    let n = axes.n;
    let at = |i: usize, j: usize, k: usize| data[i.min(n[0] - 1) + n[0] * (j.min(n[1] - 1) + n[1] * k.min(n[2] - 1))];
    let eval = |q: usize| -> f64 {
        let p = query(q);
        let mut c = [(0usize, 0.0f64); 3];
        for d in 0..3 {
            match axes.cell(d, p[d]) {
                Some(v) => c[d] = v,
                None => return f64::NAN,
            }
        }
        let mut v = 0.0;
        for (dk, gz) in [(0, 1.0 - c[2].1), (1, c[2].1)] {
            for (dj, gy) in [(0, 1.0 - c[1].1), (1, c[1].1)] {
                for (di, gx) in [(0, 1.0 - c[0].1), (1, c[0].1)] {
                    let g = gx * gy * gz;
                    if g != 0.0 {
                        v += g * at(c[0].0 + di, c[1].0 + dj, c[2].0 + dk);
                    }
                }
            }
        }
        v
    };
    let mut out = vec![0.0; n_query];
    #[cfg(feature = "parallel")]
    out.par_iter_mut().enumerate().for_each(|(q, o)| *o = eval(q));
    #[cfg(not(feature = "parallel"))]
    out.iter_mut().enumerate().for_each(|(q, o)| *o = eval(q));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn axes(n: [usize; 3]) -> UniformAxes {
        UniformAxes { n, origin: [-1.0, 0.5, 2.0], step: [0.7, 1.1, 0.9] }
    }

    fn fill(a: &UniformAxes, f: impl Fn(f64, f64, f64) -> f64) -> Vec<f64> {
        (0..a.len()).map(|i| { let p = a.node_at(i); f(p[0], p[1], p[2]) }).collect()
    }

    fn at(v: &[f64], a: &UniformAxes, qs: &[[f64; 3]], spline: bool) -> Vec<f64> {
        let qs = qs.to_vec();
        let n = qs.len();
        let q = move |i: usize| qs[i];
        if spline { spline_resample(v, a, n, &q) } else { linear_resample(v, a, n, &q) }
    }

    /// Not-a-knot splines reproduce cubics exactly, inside the grid and beyond it, so a
    /// tensor-product cubic comes back exactly anywhere.
    #[test]
    fn spline_reproduces_tensor_cubics() {
        let f = |x: f64, y: f64, z: f64| {
            (1.0 + x - 0.3 * x * x + 0.05 * x * x * x) * (2.0 - y + 0.2 * y * y * y) * (0.5 + z * z - 0.1 * z * z * z)
        };
        for n in [[7, 6, 5], [4, 9, 4], [12, 5, 8]] {
            let a = axes(n);
            let qs = [[0.13, 1.7, 2.4], [2.9, 4.0, 4.1], [-1.0, 0.5, 2.0], [-1.6, 9.3, 1.2]];
            let got = at(&fill(&a, f), &a, &qs, true);
            for (q, g) in qs.iter().zip(got) {
                let e = f(q[0], q[1], q[2]);
                assert!((g - e).abs() < 1e-9 * e.abs().max(1.0), "{n:?} at {q:?}: {g} vs {e}");
            }
        }
    }

    #[test]
    fn linear_is_nan_outside() {
        let a = axes([5, 5, 5]);
        let v = fill(&a, |x, y, z| x + 2.0 * y - z);
        let hi = a.node(4, 4, 4);
        let qs = [hi, [hi[0] + 1e-9, hi[1], hi[2]], a.node(0, 0, 0), [0.1, 1.0, 3.0]];
        let got = at(&v, &a, &qs, false);
        assert!((got[0] - (hi[0] + 2.0 * hi[1] - hi[2])).abs() < 1e-12);
        assert!(got[1].is_nan());
        assert!((got[3] - (0.1 + 2.0 - 3.0)).abs() < 1e-12);
    }

    /// Short lines: 3 samples give the parabola, 2 a straight line, as MATLAB's `spline` does.
    #[test]
    fn short_axes() {
        let a = UniformAxes { n: [3, 2, 4], origin: [0.0; 3], step: [1.0; 3] };
        let f = |x: f64, y: f64, z: f64| (x * x) * (1.0 + y) * (1.0 + z * z * z);
        let v = fill(&a, f);
        let qs = [[0.6, 0.3, 2.2], [1.9, 0.0, 0.1], [0.0, 1.0, 3.0], [1.2, 0.5, 1.5]];
        for (q, g) in qs.iter().zip(at(&v, &a, &qs, true)) {
            assert!((g - f(q[0], q[1], q[2])).abs() < 1e-10, "{q:?}");
        }
    }
}

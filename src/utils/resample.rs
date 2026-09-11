//! Spline resampling of 3D volumes to a new grid shape.
//!
//! [`resize`] reproduces `skimage.transform.resize(image, shape, order, mode='edge',
//! anti_aliasing=False)` — i.e. `scipy.ndimage.zoom(..., grid_mode=True, mode='nearest')` plus
//! skimage's clip to the input's value range. That is the resampler nnU-Net (and so HD-BET) uses,
//! and matching it exactly keeps ported deep-learning preprocessing faithful to the reference.
//!
//! Coordinates are half-pixel ("grid mode"): output sample `o` of an axis resized from `n` to `m`
//! reads input position `(o + 0.5) · n/m − 0.5`, clamped to the edge. Order 3 first converts the
//! samples to cubic B-spline coefficients along each axis (after scipy's 12-voxel edge pre-pad),
//! so the interpolant passes through the data. Because both the prefilter and the tensor-product
//! spline are separable, the volume is resampled one axis at a time — exact, not an approximation.
//!
//! Volumes are C-order `[d0, d1, d2]` (last axis fastest). QSM.rs's column-major `(nx, ny, nz)`
//! layout is exactly C-order `[nz, ny, nx]`, so pass `[nz, ny, nx]` for crate volumes.

/// Resize a C-order 3D volume `[d0, d1, d2]` to `new_dims` with a spline of `order` 0, 1 or 3.
///
/// Matches `skimage.transform.resize(image, new_dims, order, mode='edge', anti_aliasing=False)`
/// (including its clip of the result to the input's min/max). A same-shape resize returns the
/// input unchanged.
pub fn resize(data: &[f64], dims: [usize; 3], new_dims: [usize; 3], order: usize) -> Vec<f64> {
    assert_eq!(data.len(), dims.iter().product::<usize>(), "data length does not match dims");
    assert!(matches!(order, 0 | 1 | 3), "resize supports spline orders 0, 1 and 3");
    if dims == new_dims {
        return data.to_vec();
    }
    let mut out = data.to_vec();
    let mut cur = dims;
    for axis in 0..3 {
        if cur[axis] != new_dims[axis] || order == 3 {
            // Order 3 still passes through a same-length axis: the prefilter+interpolation at the
            // original sample positions is the identity up to rounding, as in scipy.
            (out, cur) = resize_axis(&out, cur, axis, new_dims[axis], order);
        }
    }
    if order > 1 {
        let (lo, hi) = data.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
        out.iter_mut().for_each(|v| *v = v.clamp(lo, hi));
    }
    out
}

/// Resample one axis of a C-order volume to `new_len` samples.
pub(crate) fn resize_axis(
    data: &[f64],
    dims: [usize; 3],
    axis: usize,
    new_len: usize,
    order: usize,
) -> (Vec<f64>, [usize; 3]) {
    let n = dims[axis];
    let mut new_dims = dims;
    new_dims[axis] = new_len;
    let stride = |d: [usize; 3], a: usize| d[a + 1..].iter().product::<usize>();
    let (s_in, s_out) = (stride(dims, axis), stride(new_dims, axis));
    // Lines along `axis` are indexed by (outer, inner): outer spans axes before it, inner after.
    let inner = s_in;
    let outer: usize = dims[..axis].iter().product();
    let mut out = vec![0.0; new_dims.iter().product()];
    let plan = LinePlan::new(n, new_len, order);
    let mut line = vec![0.0; n];
    let mut res = vec![0.0; new_len];
    let mut scratch = Vec::new();
    for o in 0..outer {
        for i in 0..inner {
            let base_in = o * n * inner + i;
            for (k, v) in line.iter_mut().enumerate() {
                *v = data[base_in + k * s_in];
            }
            plan.apply(&line, &mut res, &mut scratch);
            let base_out = o * new_len * inner + i;
            for (k, &v) in res.iter().enumerate() {
                out[base_out + k * s_out] = v;
            }
        }
    }
    (out, new_dims)
}

/// Precomputed sample positions for resampling a length-`n` line to `m` samples.
struct LinePlan {
    n: usize,
    order: usize,
    /// Per output sample: input coordinate (edge-clamped for orders 0/1; offset by the pre-pad
    /// for order 3).
    coords: Vec<f64>,
}

/// scipy pre-pads by this many edge voxels before the order-3 prefilter (`mode='nearest'`).
const NPAD: usize = 12;

impl LinePlan {
    fn new(n: usize, m: usize, order: usize) -> Self {
        let zoom = n as f64 / m as f64;
        let coords = (0..m)
            .map(|o| {
                let cc = (o as f64 + 0.5) * zoom - 0.5;
                if order == 3 { cc + NPAD as f64 } else { cc.clamp(0.0, (n - 1) as f64) }
            })
            .collect();
        Self { n, order, coords }
    }

    fn apply(&self, line: &[f64], out: &mut [f64], scratch: &mut Vec<f64>) {
        let n = self.n;
        match self.order {
            0 => {
                for (o, &cc) in out.iter_mut().zip(&self.coords) {
                    *o = line[((cc + 0.5).floor() as usize).min(n - 1)];
                }
            }
            1 => {
                for (o, &cc) in out.iter_mut().zip(&self.coords) {
                    let i = cc.floor() as usize;
                    let t = cc - i as f64;
                    *o = if i + 1 < n { (1.0 - t) * line[i] + t * line[i + 1] } else { line[n - 1] };
                }
            }
            _ => {
                scratch.clear();
                scratch.extend(std::iter::repeat_n(line[0], NPAD));
                scratch.extend_from_slice(line);
                scratch.extend(std::iter::repeat_n(line[n - 1], NPAD));
                cubic_prefilter(scratch);
                let len = scratch.len() as isize;
                for (o, &cc) in out.iter_mut().zip(&self.coords) {
                    let f = cc.floor();
                    let t = cc - f;
                    let w = cubic_weights(t);
                    let start = f as isize - 1;
                    *o = (0..4)
                        .map(|k| w[k] * scratch[(start + k as isize).clamp(0, len - 1) as usize])
                        .sum();
                }
            }
        }
    }
}

/// Cubic B-spline weights for the 4 taps `floor(x)-1 ..= floor(x)+2`, `t = x - floor(x)`.
#[inline]
fn cubic_weights(t: f64) -> [f64; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    let u = 1.0 - t;
    [u * u * u / 6.0, (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0, (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0, t3 / 6.0]
}

/// In-place conversion of samples to cubic B-spline coefficients (scipy `spline_filter1d`,
/// order 3, with the `nearest`/`reflect` boundary initialisation scipy uses for `mode='nearest'`).
fn cubic_prefilter(c: &mut [f64]) {
    let n = c.len();
    if n < 2 {
        return;
    }
    let z = 3f64.sqrt() - 2.0;
    let gain = (1.0 - z) * (1.0 - 1.0 / z);
    c.iter_mut().for_each(|v| *v *= gain);

    // causal init (half-sample symmetric boundary)
    let z_n = z.powi(n as i32);
    let c0 = c[0];
    let mut acc = c[0] + z_n * c[n - 1];
    let mut z_i = z;
    for i in 1..n {
        acc += z_i * (c[i] + z_n * c[n - 1 - i]);
        z_i *= z;
    }
    c[0] = acc * z / (1.0 - z_n * z_n) + c0;
    for i in 1..n {
        c[i] += z * c[i - 1];
    }
    // anti-causal init + recursion
    c[n - 1] *= z / (z - 1.0);
    for i in (0..n - 1).rev() {
        c[i] = z * (c[i + 1] - c[i]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ground truth from `np.clip(scipy.ndimage.zoom(a, out/in, order, mode='nearest',
    /// grid_mode=True), a.min(), a.max())` (scipy 1.17) on a random 3×4×5 volume.
    #[test]
    fn matches_scipy_zoom() {
        const A: &[f64] = &[0.126, -0.132, 0.64, 0.105, -0.536, 0.362, 1.304, 0.947, -0.704, -1.265, -0.623, 0.041, -2.325, -0.219, -1.246, -0.732, -0.544, -0.316, 0.412, 1.043, -0.129, 1.366, -0.665, 0.352, 0.903, 0.094, -0.743, -0.922, -0.458, 0.22, -1.01, -0.209, -0.159, 0.541, 0.215, 0.355, -0.654, -0.13, 0.784, 1.493, -1.259, 1.514, 1.346, 0.781, 0.264, -0.314, 1.458, 1.96, 1.802, 1.315, 0.357, -1.208, -0.004, 0.656, -1.288, 0.395, 0.43, 0.696, -1.184, -0.662];
        let cases: &[([usize; 3], usize, &[f64])] = &[
            ([5, 3, 6], 3, &[0.20691140736889094, -0.05544255056061204, 0.6796879053626795, 0.7623692162882375, -0.2977192633551938, -0.7618094213178997, -0.11131853234591335, 0.9876428139602066, -0.060227925092753536, -0.8913611965662104, -0.7702982003953869, -1.753562296842177, -0.8443882598775497, -0.5272083457200214, -0.7108803964459949, -0.3964863405756097, 0.4592174925613366, 0.6491764274567631, 0.17824497333794903, 0.40787989371436983, 0.2100084605909856, -0.018951514235165518, 0.010448843084948324, -0.03847628459096933, -0.37828977819908854, 0.23964253412996248, -0.42650146505891545, -0.8405209608910014, -0.5583746340088893, -0.9855420490491567, -0.4327945623621795, -0.5389451544017205, -0.6813413684508631, -0.07456620236859524, 0.8905102781220942, 1.162511857644753, -0.12383284958710847, 1.0174319198989874, 0.009061440209348045, -0.6258390283762802, 0.4896940507455213, 0.8338327211253993, -0.5702187099768432, -0.6226396738926097, -0.6568196055101468, -0.3638778874610913, 0.0005964843274519169, 0.018462044488186738, 0.1682645752198659, -0.432376037228785, -0.4287513627321319, 0.2767794529070827, 0.9797916327721143, 1.3449249308206501, -0.8672025551975446, 1.100264627791211, 1.0894895639853939, 0.4102064524038356, 0.7509215607072818, 0.7416579179569649, -0.1730937615571534, -0.4162504605669039, -0.019825779128214586, 0.7894710515024982, 0.8208459195657049, 0.13786380068462353, 0.40817803237808076, -0.09375701470080006, 0.17337223508776486, 0.13277860643650471, -0.27830099537247177, 0.06015791095359369, -1.374713942053433, 0.9920449771042323, 1.96, 1.4116290790894135, 0.8225190927368777, 0.4184315731102499, 0.19911619791357993, -0.004637485725696058, 0.5550191880385396, 1.5746034891687932, 1.3162820730628888, -0.05399620564169662, 0.427893263303175, 0.1463030719209041, 0.5821894755030108, -0.08280076886961928, -1.30904106503382, -1.0185408883657396]),
            ([2, 7, 4], 3, &[0.0885320124509941, 0.1304943788632103, 0.2522357634009382, -0.09311040581273282, 0.2924250532111617, 0.5089582031314358, 0.07183225353051605, -0.4272275176297005, 0.3970632911514789, 0.7848979549132512, -0.31707201311689376, -0.9555524686647726, -0.12609041112323008, 0.06488077021181975, -0.7412364473295877, -1.140416150125761, -0.6770990189156827, -0.7656418379633627, -0.7444254242786972, -0.597884175781794, -0.638972122795987, -0.7535645497972714, -0.1281335857619922, 0.6106505107525216, -0.47714089769322315, -0.541967518465588, 0.3271495724998062, 1.3749245589764694, -0.8230197493052285, 1.5826851247328977, 0.5950547783935175, 0.39668296218681554, -0.5530314925638274, 1.60674867341478, 0.9034775181910113, 0.8860982063511333, -0.17171260221669335, 1.2258824158990915, 1.2981268861309603, 1.2234422468350383, -0.09631135484691188, -0.022928862530882964, 1.2025461646030933, 0.24223512612681786, -0.058079566826018836, -0.8271040528397503, 0.6563417462176621, -0.7912924096534675, 0.23448689122631666, -0.14627786182179292, -0.0249984172042219, -0.5788779263291858, 0.44834805228532854, 0.5492006791409273, -0.3970321541306202, -0.16846821504948714]),
            ([5, 3, 6], 1, &[0.1653333333333333, 0.12183333333333331, 0.44790277777777787, 0.39074999999999976, -0.18674999999999997, -0.6574999999999999, -0.1305, 0.47175, -0.12170833333333353, -0.5942083333333334, -0.66, -1.2555, -0.7138333333333333, -0.5133333333333332, -0.5656944444444448, -0.2518055555555557, 0.39549999999999963, 0.6614999999999993, 0.06246666666666668, 0.36826666666666663, 0.27266388888888893, 0.1054555555555556, 0.03196666666666666, -0.07883333333333342, -0.2615, 0.09445000000000006, -0.2784750000000001, -0.47572499999999995, -0.36180000000000007, -0.6663, -0.3773000000000002, -0.46919999999999995, -0.46751666666666686, -0.05862777777777782, 0.5883499999999997, 0.9088999999999993, -0.09183333333333335, 0.7379166666666668, 0.009805555555555373, -0.32248611111111086, 0.3600416666666667, 0.7891666666666667, -0.458, -0.47150000000000003, -0.513625, -0.2979999999999999, 0.0855, 0.21750000000000003, 0.12749999999999959, -0.40299999999999997, -0.32024999999999987, 0.23113888888888912, 0.8776249999999999, 1.2799999999999998, -0.6976333333333335, 0.8070416666666667, 0.8870055555555556, 0.615713888888889, 0.6379166666666667, 0.5791666666666666, -0.17029999999999995, -0.12912499999999993, 0.1681000000000001, 0.5303500000000003, 0.589275, 0.09509999999999996, 0.28419999999999984, -0.03225000000000017, 0.11391666666666662, 0.07588888888888876, -0.1587, 0.05219999999999963, -1.1015, 0.853125, 1.4718055555555554, 1.2411805555555553, 0.8231666666666667, 0.43916666666666665, 0.02149999999999999, 0.09912499999999999, 0.6225833333333335, 1.0825833333333335, 0.925125, 0.013499999999999934, 0.3886666666666666, 0.21491666666666628, 0.40336111111111084, -0.0276111111111114, -0.8495833333333329, -0.7663333333333336]),
            ([6, 2, 9], 0, &[0.362, 0.362, 1.304, 1.304, 0.947, -0.704, -0.704, -1.265, -1.265, -0.732, -0.732, -0.544, -0.544, -0.316, 0.412, 0.412, 1.043, 1.043, 0.362, 0.362, 1.304, 1.304, 0.947, -0.704, -0.704, -1.265, -1.265, -0.732, -0.732, -0.544, -0.544, -0.316, 0.412, 0.412, 1.043, 1.043, 0.094, 0.094, -0.743, -0.743, -0.922, -0.458, -0.458, 0.22, 0.22, 0.355, 0.355, -0.654, -0.654, -0.13, 0.784, 0.784, 1.493, 1.493, 0.094, 0.094, -0.743, -0.743, -0.922, -0.458, -0.458, 0.22, 0.22, 0.355, 0.355, -0.654, -0.654, -0.13, 0.784, 0.784, 1.493, 1.493, -0.314, -0.314, 1.458, 1.458, 1.96, 1.802, 1.802, 1.315, 1.315, 0.395, 0.395, 0.43, 0.43, 0.696, -1.184, -1.184, -0.662, -0.662, -0.314, -0.314, 1.458, 1.458, 1.96, 1.802, 1.802, 1.315, 1.315, 0.395, 0.395, 0.43, 0.43, 0.696, -1.184, -1.184, -0.662, -0.662]),
        ];
        for (shape, order, want) in cases {
            let got = resize(A, [3, 4, 5], *shape, *order);
            let err = got.iter().zip(*want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(err < 1e-12, "order {order} -> {shape:?}: max |d| = {err:e}");
        }
    }

    #[test]
    fn same_shape_is_identity() {
        let d: Vec<f64> = (0..24).map(|i| i as f64).collect();
        assert_eq!(resize(&d, [2, 3, 4], [2, 3, 4], 3), d);
    }

    #[test]
    fn cubic_interpolates_through_samples() {
        // prefilter + evaluation at the original sample positions is the identity
        let line: Vec<f64> = (0..17).map(|i| ((i * 7) % 5) as f64 - 1.3).collect();
        let plan = LinePlan::new(17, 17, 3);
        let mut out = vec![0.0; 17];
        plan.apply(&line, &mut out, &mut Vec::new());
        for (a, b) in out.iter().zip(&line) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }

    #[test]
    fn linear_upsample_matches_half_pixel_formula() {
        // [0, 1] -> 4 samples at x = -0.25, 0.25, 0.75, 1.25 -> clamped -> 0, .25, .75, 1
        let (out, dims) = resize_axis(&[0.0, 1.0], [1, 1, 2], 2, 4, 1);
        assert_eq!(dims, [1, 1, 4]);
        assert_eq!(out, vec![0.0, 0.25, 0.75, 1.0]);
    }

    #[test]
    fn nearest_rounds_half_up_and_clamps() {
        // 3 -> 2: cc = 0.25, 1.75 -> idx 0, 2
        let (out, _) = resize_axis(&[10.0, 20.0, 30.0], [3, 1, 1], 0, 2, 0);
        assert_eq!(out, vec![10.0, 30.0]);
    }

    #[test]
    fn cubic_output_is_clipped_to_input_range() {
        // A step overshoots under cubic interpolation; skimage clips to the input range.
        let d: Vec<f64> = (0..10).map(|i| if i < 5 { 0.0 } else { 1.0 }).collect();
        let out = resize(&d, [1, 1, 10], [1, 1, 23], 3);
        assert!(out.iter().all(|&v| (0.0..=1.0).contains(&v)));
        assert!(out.iter().any(|&v| v > 0.0 && v < 1.0));
    }
}

//! SynthSeg whole-brain segmentation (`onnx` feature).
//!
//! SynthSeg is a 3D U-Net trained purely on synthetic images sampled from anatomical label maps
//! with fully randomised contrast and resolution. That domain randomisation is what makes it
//! contrast-agnostic: one set of weights segments any MR contrast without retraining, which is
//! why it is the whole-brain parcellation that can be run directly on a GRE magnitude image
//! rather than on a separately acquired T1w scan.
//!
//! This module is a port of the non-robust path of `SynthSeg/predict_synthseg.py` around the
//! exported network (`synthseg.onnx`, see [`crate::models`]):
//!
//! 1. resample to 1 mm isotropic (anti-aliased, linear) if the input is not already there;
//! 2. permute and flip the axes into RAS orientation using the NIfTI affine;
//! 3. optionally centre-crop, then rescale the 0.5–99.5 percentile range to `[0, 1]`;
//! 4. pad to a multiple of 32 (at least 128) and run the network, giving one posterior map per
//!    label;
//! 5. blur the posteriors, and average them with a left-right flipped second pass;
//! 6. keep the largest connected component of the foreground and of each topological class,
//!    renormalise, take the argmax, and undo the crop and the orientation change.
//!
//! **Feed it magnitude, not susceptibility.** On a χ map the white/grey boundary is faint and
//! the brainstem is badly under-segmented (Dice ≈0.70 against the same subject's magnitude
//! segmentation); on the echo-combined GRE magnitude the structure volumes come out in the
//! normal physiological range. For multi-echo data the root-sum-of-squares over echoes works.
//!
//! Memory is the practical constraint: the posteriors alone are `nx·ny·nz·n_labels` floats, and
//! peak usage on a 192×256×128 volume is around 6 GB. [`SynthSegParams::crop`] bounds it for
//! constrained hosts.
//!
//! Reference:
//! Billot, B., Greve, D.N., Puonti, O., et al. (2023). "SynthSeg: Segmentation of brain MRI
//! scans of any contrast and resolution without retraining." Medical Image Analysis, 86:102789.
//! <https://doi.org/10.1016/j.media.2023.102789>
//!
//! Reference implementation: <https://github.com/BBillot/SynthSeg> (Apache-2.0).

// The pre/post-processing is plain Rust; only inference needs `onnx`.
#![cfg_attr(not(feature = "onnx"), allow(dead_code))]

use crate::grid::Grid;
#[cfg(feature = "onnx")]
use crate::models::onnx::{OnnxError, OnnxModel, Tensor};
use crate::segment::labels::SynthSegVersion;
use crate::utils::connected::largest_component;
use crate::utils::resample::resize;
use crate::utils::signal_erosion::gaussian_filter_anisotropic;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Network down-sampling factor: the padded input must be a multiple of this on every axis.
const DIVISOR: usize = 32;
/// Minimum padded size per axis, as in the SynthSeg CLI when `--crop` is not given.
const MIN_PAD: usize = 128;
/// Target resolution (mm).
const TARGET_RES: f64 = 1.0;

/// SynthSeg inference parameters.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct SynthSegParams {
    /// Which generation of weights is being run — this selects the label table, so it must match
    /// the `.onnx` being passed in.
    pub version: SynthSegVersion,
    /// Optional cubic centre-crop, in voxels at 1 mm, applied before inference (rounded up to a
    /// multiple of 32). `None` — the default, and the SynthSeg CLI's — processes the whole
    /// volume. Note that the CLI's `--crop` help text claims a default of 192, but its argparse
    /// default is in fact `None`; matching the CLI means cropping nothing.
    pub crop: Option<usize>,
    /// Average the posteriors with a left-right flipped second pass (SynthSeg's default).
    /// Doubles inference time; `--fast` turns it off.
    pub flip_averaging: bool,
    /// Reset each topological class to its largest connected component (SynthSeg's default).
    /// `--fast` turns it off.
    pub topology_cleanup: bool,
    /// Gaussian blur applied to the posteriors, in voxels (SynthSeg's default is 0.5). Zero
    /// disables it.
    pub sigma_smoothing: f64,
}

impl Default for SynthSegParams {
    fn default() -> Self {
        Self {
            version: SynthSegVersion::default(),
            crop: None,
            flip_averaging: true,
            topology_cleanup: true,
            sigma_smoothing: 0.5,
        }
    }
}

impl SynthSegParams {
    /// SynthSeg's `--fast`: no flip averaging and no topological cleanup. Roughly halves the
    /// runtime for a small accuracy cost.
    pub fn fast() -> Self {
        Self { flip_averaging: false, topology_cleanup: false, ..Self::default() }
    }
}

/// The outcome of a SynthSeg run.
pub struct SynthSegResult {
    /// FreeSurfer label id per voxel on the input grid, column-major `(nx, ny, nz)`.
    pub labels: Vec<i32>,
    /// Volume in mm³ per label, parallel to [`SynthSegVersion::labels`]`().ids`. These are
    /// partial-volume aware — the sum of the soft posteriors, as SynthSeg reports them, not a
    /// count of `labels`. The background entry is always zero.
    pub volumes: Vec<f64>,
}

/// Whole-brain segmentation via SynthSeg.
///
/// * `magnitude` — magnitude image, column-major `(nx, ny, nz)`. Not a susceptibility map; see
///   the module docs.
/// * `affine` — the image's NIfTI affine (row-major 4×4), used to find the RAS orientation.
/// * `model_onnx` — bytes of the exported `synthseg.onnx` (see [`crate::models`]).
/// * `progress(done, total)` — called after each network evaluation.
#[cfg(feature = "onnx")]
pub fn synthseg(
    magnitude: &[f64],
    grid: &Grid,
    affine: &[f64; 16],
    model_onnx: &[u8],
    params: &SynthSegParams,
    mut progress: impl FnMut(usize, usize),
) -> Result<SynthSegResult, OnnxError> {
    let (nx, ny, nz) = grid.dims;
    assert_eq!(magnitude.len(), nx * ny * nz, "magnitude length must match grid");
    if params.sigma_smoothing < 0.0 {
        return Err(OnnxError::Shape("sigma_smoothing must not be negative".into()));
    }
    let n_labels = params.version.n_labels();

    let pre = preprocess(magnitude, grid, affine, params);
    let model = OnnxModel::load(model_onnx)?;

    let passes = 1 + params.flip_averaging as usize;
    let mut done = 0;
    let mut post = run_pass(&model, &pre.data, pre.dims, n_labels, false, params)?;
    done += 1;
    progress(done, passes);

    if params.flip_averaging {
        let flipped = flip_axis0(&pre.data, pre.dims, 1);
        let mut post_f = run_pass(&model, &flipped, pre.dims, n_labels, true, params)?;
        done += 1;
        progress(done, passes);
        // Undo the spatial flip, map each channel back to its left-right partner, and average.
        post_f = flip_axis0(&post_f, pre.dims, n_labels);
        let flip_table = params.version.labels().flip;
        for (out, chunk) in post.chunks_exact_mut(n_labels).zip(post_f.chunks_exact(n_labels)) {
            for c in 0..n_labels {
                out[c] = 0.5 * (out[c] + chunk[flip_table[c]]);
            }
        }
    }

    Ok(postprocess(post, &pre, params))
}

/// The network input plus everything needed to map predictions back onto the caller's grid.
struct Preprocessed {
    /// Normalised, padded volume in RAS orientation, C-order `[d0, d1, d2]`.
    data: Vec<f32>,
    dims: [usize; 3],
    /// Shape of the RAS-aligned volume before cropping and padding.
    aligned_dims: [usize; 3],
    /// Where the padded volume's real data sits: `[start, end)` per axis.
    pad: [(usize, usize); 3],
    /// Where the crop sits inside `aligned_dims`: `[start, end)` per axis.
    crop: [(usize, usize); 3],
    /// `perm[i]` is the pre-alignment axis that became RAS axis `i`.
    perm: [usize; 3],
    /// Whether RAS axis `i` was flipped during alignment.
    flip: [bool; 3],
    /// Shape of the volume the alignment consumed (post-resampling, pre-alignment).
    source_dims: [usize; 3],
    /// Resolution the network saw, mm.
    res: f64,
    /// The caller's grid, for mapping the result back.
    out_dims: (usize, usize, usize),
    /// Whether a resample to 1 mm happened (the label map is returned on the caller's grid
    /// either way).
    resampled: bool,
}

// ----------------------------------------------------------------------- orientation

/// Invert a 3×3 matrix given row-major; returns `None` if it is singular.
fn invert3(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-12 {
        return None;
    }
    let mut inv = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let (a, b) = ((i + 1) % 3, (i + 2) % 3);
            let (c, d) = ((j + 1) % 3, (j + 2) % 3);
            // Transposed cofactor = adjugate.
            inv[j][i] = (m[a][c] * m[b][d] - m[a][d] * m[b][c]) / det;
        }
    }
    Some(inv)
}

/// Which volume axis each of R, A, S runs along — lab2im's `get_ras_axes`.
fn ras_axes(affine: &[f64; 16]) -> [usize; 3] {
    let m = [
        [affine[0], affine[1], affine[2]],
        [affine[4], affine[5], affine[6]],
        [affine[8], affine[9], affine[10]],
    ];
    let inv = match invert3(&m) {
        Some(i) => i,
        None => return [0, 1, 2],
    };
    let mut axes = [0usize; 3];
    for (w, a) in axes.iter_mut().enumerate() {
        let mut best = 0;
        for v in 1..3 {
            if inv[v][w].abs() > inv[best][w].abs() {
                best = v;
            }
        }
        *a = best;
    }
    // A degenerate affine can map two world axes onto the same volume axis; lab2im repairs that
    // by reassigning the last duplicate to whichever axis is missing.
    for i in 0..3 {
        if !axes.contains(&i) {
            let mut counts = [0usize; 3];
            for &a in &axes {
                counts[a] += 1;
            }
            let dup = (0..3).max_by_key(|&v| counts[v]).unwrap();
            if let Some(pos) = axes.iter().rposition(|&a| a == dup) {
                axes[pos] = i;
            }
        }
    }
    axes
}

/// The axis permutation and per-axis flips that take a volume into RAS orientation.
///
/// Returns `(perm, flip)` where RAS axis `i` is the source volume's axis `perm[i]`, reversed
/// when `flip[i]`. Equivalent to lab2im's `align_volume_to_ref(..., aff_ref=eye(4))`.
fn ras_alignment(affine: &[f64; 16]) -> ([usize; 3], [bool; 3]) {
    let perm = ras_axes(affine);
    let m = [
        [affine[0], affine[1], affine[2]],
        [affine[4], affine[5], affine[6]],
        [affine[8], affine[9], affine[10]],
    ];
    let mut flip = [false; 3];
    for i in 0..3 {
        flip[i] = m[i][perm[i]] < 0.0;
    }
    (perm, flip)
}

// ----------------------------------------------------------------------- preprocessing

fn round_up(n: usize, m: usize) -> usize {
    if n % m == 0 { n } else { (n / m + 1) * m }
}

/// `np.percentile(sorted, q)` with linear interpolation between order statistics.
fn percentile(sorted_scratch: &mut [f32], q: f64) -> f64 {
    let n = sorted_scratch.len();
    if n == 0 {
        return 0.0;
    }
    let pos = q / 100.0 * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let frac = pos - lo as f64;
    let a = *sorted_scratch.select_nth_unstable_by(lo, |x, y| x.total_cmp(y)).1 as f64;
    if frac == 0.0 || lo + 1 >= n {
        return a;
    }
    // Everything above index `lo` is already partitioned to the right; the next order statistic
    // is the minimum of that tail.
    let b = sorted_scratch[lo + 1..].iter().copied().fold(f32::INFINITY, f32::min) as f64;
    a + frac * (b - a)
}

fn preprocess(
    magnitude: &[f64],
    grid: &Grid,
    affine: &[f64; 16],
    params: &SynthSegParams,
) -> Preprocessed {
    let (perm, flip) = ras_alignment(affine);
    let out_dims = grid.dims;

    // 1. resample to 1 mm isotropic if needed (SynthSeg's 0.05 mm tolerance).
    let vs = [grid.vsx(), grid.vsy(), grid.vsz()];
    let needs_resample =
        vs.iter().any(|&v| v > TARGET_RES + 0.05 || v < TARGET_RES - 0.05);
    let (source, source_dims_cm) = if needs_resample {
        resample_to_iso(magnitude, grid)
    } else {
        (magnitude.to_vec(), [grid.nx(), grid.ny(), grid.nz()])
    };

    // 2. align to RAS, writing straight into the C-order layout the network wants.
    let aligned_dims = [source_dims_cm[perm[0]], source_dims_cm[perm[1]], source_dims_cm[perm[2]]];
    let mut data = align_to_ras(&source, source_dims_cm, aligned_dims, perm, flip);

    // 3. centre-crop.
    let mut dims = aligned_dims;
    let crop = match params.crop {
        Some(c) => {
            let want = round_up(c, DIVISOR);
            let mut bounds = [(0, 0); 3];
            for i in 0..3 {
                let keep = want.min(dims[i]);
                let start = (dims[i] - keep) / 2;
                bounds[i] = (start, start + keep);
            }
            data = crop_c_order(&data, dims, bounds, 1);
            dims = [bounds[0].1 - bounds[0].0, bounds[1].1 - bounds[1].0, bounds[2].1 - bounds[2].0];
            bounds
        }
        None => [(0, dims[0]), (0, dims[1]), (0, dims[2])],
    };

    // 4. rescale the robust intensity range to [0, 1].
    let mut scratch = data.clone();
    let lo = percentile(&mut scratch, 0.5);
    let mut scratch = data.clone();
    let hi = percentile(&mut scratch, 99.5);
    if hi > lo {
        let scale = 1.0 / (hi - lo);
        for v in data.iter_mut() {
            *v = (((*v as f64).clamp(lo, hi) - lo) * scale) as f32;
        }
    } else {
        data.iter_mut().for_each(|v| *v = 0.0);
    }

    // 5. pad to a multiple of 32, at least 128 per axis.
    let min_pad = params.crop.map(|c| round_up(c, DIVISOR)).unwrap_or(MIN_PAD);
    let padded: [usize; 3] =
        std::array::from_fn(|i| round_up(dims[i], DIVISOR).max(round_up(min_pad, DIVISOR)));
    let mut pad = [(0, 0); 3];
    for i in 0..3 {
        let start = (padded[i] - dims[i]) / 2;
        pad[i] = (start, start + dims[i]);
    }
    let data = pad_c_order(&data, dims, padded, pad, 1);

    Preprocessed {
        data,
        dims: padded,
        aligned_dims,
        pad,
        crop,
        perm,
        flip,
        source_dims: source_dims_cm,
        res: TARGET_RES,
        out_dims,
        resampled: needs_resample,
    }
}

/// Resample a column-major volume to 1 mm isotropic, matching lab2im's `resample_volume`:
/// an anti-aliasing Gaussian when down-sampling, then linear interpolation on a half-pixel grid.
fn resample_to_iso(magnitude: &[f64], grid: &Grid) -> (Vec<f64>, [usize; 3]) {
    let vs = [grid.vsx(), grid.vsy(), grid.vsz()];
    let dims = [grid.nx(), grid.ny(), grid.nz()];
    let factor: [f64; 3] = std::array::from_fn(|i| vs[i] / TARGET_RES);
    let sigma: [f64; 3] =
        std::array::from_fn(|i| if factor[i] > 1.0 { 0.0 } else { 0.25 / factor[i] });
    let blurred = gaussian_filter_anisotropic(magnitude, grid.dims, sigma);
    let new_dims: [usize; 3] =
        std::array::from_fn(|i| ((dims[i] as f64) * factor[i]).ceil().max(1.0) as usize);
    // `resize` works in C-order; a column-major (nx, ny, nz) volume is C-order [nz, ny, nx].
    let out = resize(
        &blurred,
        [dims[2], dims[1], dims[0]],
        [new_dims[2], new_dims[1], new_dims[0]],
        1,
    );
    (out, new_dims)
}

/// Gather a column-major `(nx, ny, nz)` volume into a C-order RAS-oriented `f32` buffer.
fn align_to_ras(
    source: &[f64],
    source_dims: [usize; 3],
    dims: [usize; 3],
    perm: [usize; 3],
    flip: [bool; 3],
) -> Vec<f32> {
    let (nx, ny) = (source_dims[0], source_dims[1]);
    let nxy = nx * ny;
    let strides = [1usize, nx, nxy];
    let mut out = vec![0f32; dims[0] * dims[1] * dims[2]];
    // Source offset contributed by each output axis, and its step.
    let mut base = 0usize;
    let mut step = [0isize; 3];
    for i in 0..3 {
        let s = strides[perm[i]] as isize;
        if flip[i] {
            base += (dims[i] - 1) * strides[perm[i]];
            step[i] = -s;
        } else {
            step[i] = s;
        }
    }
    crate::maybe_par_chunks_mut!(out, dims[1] * dims[2]).enumerate().for_each(|(o0, plane)| {
        let row0 = base as isize + step[0] * o0 as isize;
        for o1 in 0..dims[1] {
            let row = row0 + step[1] * o1 as isize;
            let dst = &mut plane[o1 * dims[2]..(o1 + 1) * dims[2]];
            for (o2, d) in dst.iter_mut().enumerate() {
                *d = source[(row + step[2] * o2 as isize) as usize] as f32;
            }
        }
    });
    out
}

/// Scatter a C-order RAS-oriented volume back to column-major `(nx, ny, nz)`.
fn unalign_from_ras<T: Copy>(
    aligned: &[T],
    dims: [usize; 3],
    out_dims: [usize; 3],
    perm: [usize; 3],
    flip: [bool; 3],
    out: &mut [T],
) {
    let (nx, ny) = (out_dims[0], out_dims[1]);
    let nxy = nx * ny;
    let strides = [1usize, nx, nxy];
    let mut base = 0usize;
    let mut step = [0isize; 3];
    for i in 0..3 {
        let s = strides[perm[i]] as isize;
        if flip[i] {
            base += (dims[i] - 1) * strides[perm[i]];
            step[i] = -s;
        } else {
            step[i] = s;
        }
    }
    for o0 in 0..dims[0] {
        let row0 = base as isize + step[0] * o0 as isize;
        for o1 in 0..dims[1] {
            let row = row0 + step[1] * o1 as isize;
            let src = &aligned[(o0 * dims[1] + o1) * dims[2]..(o0 * dims[1] + o1 + 1) * dims[2]];
            for (o2, &v) in src.iter().enumerate() {
                out[(row + step[2] * o2 as isize) as usize] = v;
            }
        }
    }
}

/// Crop a C-order `[d0, d1, d2]` volume of `n_ch`-element voxels to `bounds`.
fn crop_c_order<T: Copy + Default>(
    data: &[T],
    dims: [usize; 3],
    bounds: [(usize, usize); 3],
    n_ch: usize,
) -> Vec<T> {
    let out_dims: [usize; 3] = std::array::from_fn(|i| bounds[i].1 - bounds[i].0);
    let mut out = vec![T::default(); out_dims.iter().product::<usize>() * n_ch];
    for o0 in 0..out_dims[0] {
        for o1 in 0..out_dims[1] {
            let src = ((o0 + bounds[0].0) * dims[1] + o1 + bounds[1].0) * dims[2] + bounds[2].0;
            let dst = (o0 * out_dims[1] + o1) * out_dims[2];
            out[dst * n_ch..(dst + out_dims[2]) * n_ch]
                .copy_from_slice(&data[src * n_ch..(src + out_dims[2]) * n_ch]);
        }
    }
    out
}

/// Zero-pad a C-order `[d0, d1, d2]` volume of `n_ch`-element voxels so the data lands at `at`.
fn pad_c_order<T: Copy + Default>(
    data: &[T],
    dims: [usize; 3],
    out_dims: [usize; 3],
    at: [(usize, usize); 3],
    n_ch: usize,
) -> Vec<T> {
    let mut out = vec![T::default(); out_dims.iter().product::<usize>() * n_ch];
    for o0 in 0..dims[0] {
        for o1 in 0..dims[1] {
            let src = (o0 * dims[1] + o1) * dims[2];
            let dst = ((o0 + at[0].0) * out_dims[1] + o1 + at[1].0) * out_dims[2] + at[2].0;
            out[dst * n_ch..(dst + dims[2]) * n_ch]
                .copy_from_slice(&data[src * n_ch..(src + dims[2]) * n_ch]);
        }
    }
    out
}

/// Reverse axis 0 of a C-order `[d0, d1, d2]` volume of `n_ch`-element voxels.
fn flip_axis0(data: &[f32], dims: [usize; 3], n_ch: usize) -> Vec<f32> {
    let plane = dims[1] * dims[2] * n_ch;
    let mut out = vec![0f32; data.len()];
    for o0 in 0..dims[0] {
        let src = (dims[0] - 1 - o0) * plane;
        out[o0 * plane..(o0 + 1) * plane].copy_from_slice(&data[src..src + plane]);
    }
    out
}

// ----------------------------------------------------------------------- inference

#[cfg(feature = "onnx")]
fn run_pass(
    model: &OnnxModel,
    data: &[f32],
    dims: [usize; 3],
    n_labels: usize,
    _flipped: bool,
    params: &SynthSegParams,
) -> Result<Vec<f32>, OnnxError> {
    let input = Tensor::new(vec![1, dims[0], dims[1], dims[2], 1], data.to_vec());
    let out = model.run_single(&input)?;
    let expect = [1, dims[0], dims[1], dims[2], n_labels];
    if out.shape != expect {
        return Err(OnnxError::Shape(format!(
            "SynthSeg returned {:?}, expected {:?} — do the weights match params.version?",
            out.shape, expect
        )));
    }
    let mut post = out.data;
    if params.sigma_smoothing > 0.0 {
        blur_posteriors(&mut post, dims, n_labels, params.sigma_smoothing);
    }
    Ok(post)
}

/// Per-channel separable Gaussian blur of a `[d0, d1, d2, n_ch]` posterior map.
///
/// Matches lab2im's `GaussianBlur`: kernel width `2·ceil(2.5σ)/2 + 1`, normalised, applied with
/// zero padding at the borders (TensorFlow's `SAME`) — note that is zero padding, not the
/// reflection [`gaussian_filter_anisotropic`] uses.
fn blur_posteriors(post: &mut [f32], dims: [usize; 3], n_ch: usize, sigma: f64) {
    let width = ((2.5 * sigma).ceil() as usize / 2) * 2 + 1;
    if width <= 1 {
        return;
    }
    let radius = width / 2;
    let mut w: Vec<f32> = (0..width)
        .map(|i| {
            let x = i as f64 - (width - 1) as f64 / 2.0;
            (-(x * x) / (2.0 * sigma * sigma)).exp() as f32
        })
        .collect();
    let s: f32 = w.iter().sum();
    w.iter_mut().for_each(|v| *v /= s);

    // [d0, d1, d2, n_ch] seen as [outer, n, inner] for each axis in turn.
    blur_along(post, 1, dims[0], dims[1] * dims[2] * n_ch, &w, radius);
    blur_along(post, dims[0], dims[1], dims[2] * n_ch, &w, radius);
    blur_along(post, dims[0] * dims[1], dims[2], n_ch, &w, radius);
}

/// Convolve `data`, viewed as `[outer, n, inner]`, along the middle axis with zero padding.
fn blur_along(data: &mut [f32], outer: usize, n: usize, inner: usize, w: &[f32], radius: usize) {
    if n <= 1 {
        return;
    }
    let stride = n * inner;
    crate::maybe_par_chunks_mut!(data, stride).take(outer).for_each(|line| {
        // A rolling copy of the taps that have already been overwritten.
        let mut history = vec![0f32; (2 * radius + 1) * inner];
        let mut filled = 0usize;
        for i in 0..n + radius {
            // Stash the slot about to be consumed, then write the output for `i - radius`.
            if i < n {
                let slot = (i % (2 * radius + 1)) * inner;
                history[slot..slot + inner].copy_from_slice(&line[i * inner..(i + 1) * inner]);
                filled += 1;
            }
            if i < radius || filled == 0 {
                continue;
            }
            let o = i - radius;
            if o >= n {
                break;
            }
            for c in 0..inner {
                let mut acc = 0f32;
                for (t, &wt) in w.iter().enumerate() {
                    let src = o as isize + t as isize - radius as isize;
                    if src < 0 || src as usize >= n {
                        continue; // zero padding
                    }
                    let src = src as usize;
                    acc += wt
                        * if src <= i {
                            history[(src % (2 * radius + 1)) * inner + c]
                        } else {
                            line[src * inner + c]
                        };
                }
                line[o * inner + c] = acc;
            }
        }
    });
}

// ----------------------------------------------------------------------- postprocessing

#[cfg(feature = "onnx")]
fn postprocess(
    mut post: Vec<f32>,
    pre: &Preprocessed,
    params: &SynthSegParams,
) -> SynthSegResult {
    let labels = params.version.labels();
    let n_labels = labels.ids.len();
    let dims = pre.dims;
    let n_vox = dims.iter().product::<usize>();
    // Connected-component work happens on the padded grid; 6-connectivity is symmetric under
    // axis permutation, so the C-order volume can be handed over as column-major reversed.
    let cc_dims = (dims[2], dims[1], dims[0]);

    // Foreground: everything but the background channel, thresholded and reduced to one blob.
    let mut fg = vec![0u8; n_vox];
    for (v, chunk) in fg.iter_mut().zip(post.chunks_exact(n_labels)) {
        *v = (chunk[1..].iter().sum::<f32>() > 0.25) as u8;
    }
    let fg = largest_component(&fg, cc_dims);
    for (keep, chunk) in fg.iter().zip(post.chunks_exact_mut(n_labels)) {
        if *keep == 0 {
            chunk[1..].fill(0.0);
        }
    }

    if params.topology_cleanup {
        let max_class = labels.topology.iter().copied().max().unwrap_or(0);
        for class in 1..=max_class {
            let members: Vec<usize> =
                (0..n_labels).filter(|&c| labels.topology[c] == class).collect();
            if members.is_empty() {
                continue;
            }
            let mut mask = vec![0u8; n_vox];
            for (v, chunk) in mask.iter_mut().zip(post.chunks_exact(n_labels)) {
                *v = members.iter().any(|&c| chunk[c] > 0.25) as u8;
            }
            let mask = largest_component(&mask, cc_dims);
            for (keep, chunk) in mask.iter().zip(post.chunks_exact_mut(n_labels)) {
                if *keep == 0 {
                    for &c in &members {
                        chunk[c] = 0.0;
                    }
                }
            }
        }
    } else {
        // `--fast` keeps only the posteriors that clear a slightly lower bar.
        for chunk in post.chunks_exact_mut(n_labels) {
            for c in 1..n_labels {
                if chunk[c] <= 0.2 {
                    chunk[c] = 0.0;
                }
            }
        }
    }

    // Undo the padding, renormalise, and take the argmax.
    let cropped_dims: [usize; 3] = std::array::from_fn(|i| pre.pad[i].1 - pre.pad[i].0);
    post = crop_c_order(&post, dims, pre.pad, n_labels);

    let mut volumes = vec![0f64; n_labels];
    let mut hard = vec![labels.ids[0]; cropped_dims.iter().product::<usize>()];
    for (voxel, chunk) in hard.iter_mut().zip(post.chunks_exact_mut(n_labels)) {
        let total: f32 = chunk.iter().sum();
        if total > 0.0 {
            chunk.iter_mut().for_each(|v| *v /= total);
        }
        let mut best = 0usize;
        for c in 1..n_labels {
            if chunk[c] > chunk[best] {
                best = c;
            }
        }
        *voxel = labels.ids[best];
        // Accumulate in f64: the reference sums ~10^6 f32 posteriors per structure and loses
        // about 0.4 % to rounding doing so.
        for c in 1..n_labels {
            volumes[c] += chunk[c] as f64;
        }
    }
    let voxel_volume = pre.res * pre.res * pre.res;
    volumes.iter_mut().for_each(|v| *v *= voxel_volume);

    // Undo the crop, then the RAS alignment.
    let aligned = if pre.crop == [(0, pre.aligned_dims[0]), (0, pre.aligned_dims[1]), (0, pre.aligned_dims[2])]
    {
        hard
    } else {
        pad_c_order(&hard, cropped_dims, pre.aligned_dims, pre.crop, 1)
    };

    let mut source = vec![labels.ids[0]; pre.source_dims.iter().product::<usize>()];
    unalign_from_ras(&aligned, pre.aligned_dims, pre.source_dims, pre.perm, pre.flip, &mut source);

    let out_dims = [pre.out_dims.0, pre.out_dims.1, pre.out_dims.2];
    let labels_out = if pre.resampled {
        resample_labels_nearest(&source, pre.source_dims, out_dims)
    } else {
        source
    };

    SynthSegResult { labels: labels_out, volumes }
}

/// Nearest-neighbour resample of a column-major label volume back onto the caller's grid.
fn resample_labels_nearest(src: &[i32], dims: [usize; 3], out_dims: [usize; 3]) -> Vec<i32> {
    let mut out = vec![0i32; out_dims.iter().product()];
    let scale: [f64; 3] = std::array::from_fn(|i| dims[i] as f64 / out_dims[i] as f64);
    for z in 0..out_dims[2] {
        let sz = (((z as f64 + 0.5) * scale[2] - 0.5).round().max(0.0) as usize).min(dims[2] - 1);
        for y in 0..out_dims[1] {
            let sy =
                (((y as f64 + 0.5) * scale[1] - 0.5).round().max(0.0) as usize).min(dims[1] - 1);
            for x in 0..out_dims[0] {
                let sx = (((x as f64 + 0.5) * scale[0] - 0.5).round().max(0.0) as usize)
                    .min(dims[0] - 1);
                out[x + out_dims[0] * (y + out_dims[1] * z)] =
                    src[sx + dims[0] * (sy + dims[1] * sz)];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A radiological-ish affine: +i is left, +j anterior, +k superior (the qsmxt example's).
    const LAS: [f64; 16] = [
        -1.0, 0.0, 0.0, 91.0, 0.0, 1.0, 0.0, -113.5, 0.0, 0.0, 1.0, -39.7, 0.0, 0.0, 0.0, 1.0,
    ];

    #[test]
    fn las_affine_flips_only_the_left_right_axis() {
        let (perm, flip) = ras_alignment(&LAS);
        assert_eq!(perm, [0, 1, 2]);
        assert_eq!(flip, [true, false, false]);
    }

    #[test]
    fn identity_affine_is_already_ras() {
        let mut aff = [0.0; 16];
        aff[0] = 1.0;
        aff[5] = 1.0;
        aff[10] = 1.0;
        aff[15] = 1.0;
        assert_eq!(ras_alignment(&aff), ([0, 1, 2], [false; 3]));
    }

    #[test]
    fn permuted_affine_is_detected() {
        // +i superior, +j left, +k posterior.
        let mut aff = [0.0; 16];
        aff[8] = 1.0; // world S from axis 0
        aff[1] = -1.0; // world R from axis 1, negated
        aff[6] = -1.0; // world A from axis 2, negated
        aff[15] = 1.0;
        let (perm, flip) = ras_alignment(&aff);
        assert_eq!(perm, [1, 2, 0], "R, A, S should come from axes 1, 2, 0");
        assert_eq!(flip, [true, true, false]);
    }

    #[test]
    fn alignment_round_trips() {
        let dims = [3usize, 4, 5];
        let n = dims.iter().product::<usize>();
        let src: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let (perm, flip) = ras_alignment(&LAS);
        let aligned_dims = [dims[perm[0]], dims[perm[1]], dims[perm[2]]];
        let aligned = align_to_ras(&src, dims, aligned_dims, perm, flip);
        let mut back = vec![0f32; n];
        unalign_from_ras(&aligned, aligned_dims, dims, perm, flip, &mut back);
        let expected: Vec<f32> = src.iter().map(|&v| v as f32).collect();
        assert_eq!(back, expected);
    }

    #[test]
    fn alignment_round_trips_under_permutation() {
        let dims = [3usize, 4, 5];
        let n = dims.iter().product::<usize>();
        let src: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let (perm, flip) = ([2usize, 0, 1], [true, false, true]);
        let aligned_dims = [dims[perm[0]], dims[perm[1]], dims[perm[2]]];
        let aligned = align_to_ras(&src, dims, aligned_dims, perm, flip);
        assert_eq!(aligned.len(), n);
        let mut back = vec![0f32; n];
        unalign_from_ras(&aligned, aligned_dims, dims, perm, flip, &mut back);
        let expected: Vec<f32> = src.iter().map(|&v| v as f32).collect();
        assert_eq!(back, expected);
    }

    #[test]
    fn percentile_matches_linear_interpolation() {
        let make = || (0..101).map(|i| i as f32).collect::<Vec<_>>();
        assert!((percentile(&mut make(), 0.0) - 0.0).abs() < 1e-6);
        assert!((percentile(&mut make(), 100.0) - 100.0).abs() < 1e-6);
        assert!((percentile(&mut make(), 50.0) - 50.0).abs() < 1e-6);
        // np.percentile(np.arange(101), 0.5) == 0.5
        assert!((percentile(&mut make(), 0.5) - 0.5).abs() < 1e-6);
        assert!((percentile(&mut make(), 99.5) - 99.5).abs() < 1e-6);
    }

    #[test]
    fn round_up_hits_multiples_of_the_divisor() {
        assert_eq!(round_up(128, DIVISOR), 128);
        assert_eq!(round_up(129, DIVISOR), 160);
        assert_eq!(round_up(184, DIVISOR), 192);
        assert_eq!(round_up(256, DIVISOR), 256);
    }

    #[test]
    fn crop_and_pad_are_inverses() {
        let dims = [4usize, 5, 6];
        let n = dims.iter().product::<usize>();
        let src: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let bounds = [(1, 3), (1, 4), (2, 5)];
        let cropped = crop_c_order(&src, dims, bounds, 1);
        let inner: [usize; 3] = std::array::from_fn(|i| bounds[i].1 - bounds[i].0);
        assert_eq!(cropped.len(), inner.iter().product::<usize>());
        let back = pad_c_order(&cropped, inner, dims, bounds, 1);
        for o0 in bounds[0].0..bounds[0].1 {
            for o1 in bounds[1].0..bounds[1].1 {
                for o2 in bounds[2].0..bounds[2].1 {
                    let i = (o0 * dims[1] + o1) * dims[2] + o2;
                    assert_eq!(back[i], src[i]);
                }
            }
        }
    }

    #[test]
    fn blur_preserves_a_constant_interior() {
        // A 3-tap normalised kernel on a constant field reproduces it away from the borders.
        let dims = [8usize, 8, 8];
        let n_ch = 2;
        let mut post = vec![1f32; dims.iter().product::<usize>() * n_ch];
        blur_posteriors(&mut post, dims, n_ch, 0.5);
        let at = |a: usize, b: usize, c: usize, ch: usize| ((a * dims[1] + b) * dims[2] + c) * n_ch + ch;
        assert!((post[at(4, 4, 4, 0)] - 1.0).abs() < 1e-5);
        assert!((post[at(4, 4, 4, 1)] - 1.0).abs() < 1e-5);
        // Zero padding means the corner loses weight.
        assert!(post[at(0, 0, 0, 0)] < 0.95);
    }

    #[test]
    fn blur_keeps_channels_independent() {
        let dims = [6usize, 6, 6];
        let n_ch = 3;
        let mut post = vec![0f32; dims.iter().product::<usize>() * n_ch];
        let at = |a: usize, b: usize, c: usize, ch: usize| ((a * dims[1] + b) * dims[2] + c) * n_ch + ch;
        post[at(3, 3, 3, 1)] = 1.0;
        blur_posteriors(&mut post, dims, n_ch, 0.5);
        assert!(post[at(3, 3, 3, 1)] > 0.0, "channel 1 should carry the blurred impulse");
        for a in 0..dims[0] {
            for b in 0..dims[1] {
                for c in 0..dims[2] {
                    assert_eq!(post[at(a, b, c, 0)], 0.0, "channel 0 leaked");
                    assert_eq!(post[at(a, b, c, 2)], 0.0, "channel 2 leaked");
                }
            }
        }
    }

    #[test]
    fn flip_axis0_is_an_involution() {
        let dims = [3usize, 2, 2];
        let n_ch = 2;
        let n = dims.iter().product::<usize>() * n_ch;
        let src: Vec<f32> = (0..n).map(|i| i as f32).collect();
        assert_eq!(flip_axis0(&flip_axis0(&src, dims, n_ch), dims, n_ch), src);
    }
}

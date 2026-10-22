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
//! Memory is the practical constraint: the network takes about 0.95 KB per voxel of its input at
//! once (2.2 GB at 128³, 6.0 GB on a 184×256×128 scan padded to 192×256×128, measured natively),
//! more than a 32-bit WASM heap (4 GB) holds. Three settings bound it:
//!
//! - [`SynthSegInference::SlidingWindow`] runs the network on overlapping cubic windows whose
//!   posteriors are blended with a Gaussian weight, as nnU-Net does, so the network's memory
//!   follows the window instead of the volume. Window origins sit on the network's max-pooling
//!   lattice (multiples of 16 voxels from the whole-volume input's origin), so every window pools
//!   the same voxels together that the whole-volume pass does.
//! - [`SynthSegParams::head_crop`] runs everything (network, posteriors, post-processing) on the
//!   head's bounding box plus a margin instead of the padded field of view.
//! - [`PosteriorPrecision::F16`] keeps the posteriors (`n_labels` values per voxel, the one
//!   whole-volume buffer the windows cannot bound) at half precision.
//!
//! [`SynthSegParams::low_memory`] combines them for the browser.
//!
//! Reference:
//! Billot, B., Greve, D.N., Puonti, O., et al. (2023). "SynthSeg: Segmentation of brain MRI
//! scans of any contrast and resolution without retraining." Medical Image Analysis, 86:102789.
//! <https://doi.org/10.1016/j.media.2023.102789>
//!
//! Reference implementation: <https://github.com/BBillot/SynthSeg> (Apache-2.0).

// The pre/post-processing is plain Rust; only inference needs `onnx`.
#![cfg_attr(not(feature = "onnx"), allow(dead_code))]

use crate::geometry::AxisCode;
use crate::grid::Grid;
#[cfg(feature = "onnx")]
use crate::models::onnx::{OnnxError, OnnxModel, Tensor};
use crate::segment::labels::SynthSegVersion;
use crate::utils::connected::{largest_component, Connectivity};
use crate::utils::resample::resize;
use crate::utils::morphology::signal_erosion::gaussian_filter_anisotropic;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Network down-sampling factor: the padded input must be a multiple of this on every axis.
const DIVISOR: usize = 32;
/// The U-Net's max-pooling lattice: four 2× poolings, so a window must start on a multiple of 16
/// voxels from the whole-volume input's origin to pool the same voxels together, and any input
/// to the network must be a multiple of 16 on every axis.
const LATTICE: usize = 16;
/// Minimum padded size per axis, as in the SynthSeg CLI when `--crop` is not given.
const MIN_PAD: usize = 128;
/// Target resolution (mm).
const TARGET_RES: f64 = 1.0;
/// Normalised intensity (the 0.5–99.5 percentile range mapped to `[0, 1]`) above which a voxel
/// counts as head when [`SynthSegParams::head_crop`] looks for the head's bounding box.
const HEAD_THRESHOLD: f32 = 0.1;

/// How the network is applied to the preprocessed (1 mm, RAS, padded) volume.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SynthSegInference {
    /// One network pass over the whole padded volume, as the SynthSeg CLI runs it. Peak memory
    /// grows with the volume (about 0.95 KB per padded voxel).
    FullVolume,
    /// Gaussian-weighted sliding window: the network runs on cubes of `patch` voxels (at least
    /// 64, rounded up to a multiple of 16) that overlap their neighbours by at least `overlap`
    /// voxels, and each voxel's posteriors are the Gaussian-weighted (σ = patch/8) mean over the
    /// windows that cover it. Window origins are on the network's pooling lattice, so the step
    /// between windows is a multiple of 16 no larger than `patch − overlap`, spread evenly over
    /// the axis. Peak memory for the network follows the patch. An axis the volume does not
    /// exceed is covered by one window; a volume one window covers runs as `FullVolume` does.
    SlidingWindow {
        /// Edge of the cubic patch, voxels at 1 mm.
        patch: usize,
        /// Minimum overlap between neighbouring windows, voxels (less than `patch`).
        overlap: usize,
    },
}

impl SynthSegInference {
    /// The memory-bounded window, for a 32-bit WASM heap (see [`SynthSegParams::low_memory`]):
    /// 96³ patches overlapping by at least 48. Against FreeSurfer's `mri_synthseg` (which the
    /// whole volume matches) on three 1 mm scans, mean Dice over the labels 0.984–0.988 (lowest
    /// label 0.93–0.96); 128³ windows reach 0.993–0.996 (lowest 0.98) but, even with
    /// half-precision posteriors, took a 184×256×128 scan to 3.8 GB of a 4 GB heap.
    pub const LOW_MEMORY: Self = Self::SlidingWindow { patch: 96, overlap: 48 };
}

/// Storage for the posteriors, `n_labels` values per voxel of the network's grid.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PosteriorPrecision {
    /// Single precision, as the SynthSeg CLI keeps them.
    #[default]
    F32,
    /// Half precision (IEEE binary16, about 3 significant digits): half the memory. The
    /// arithmetic stays in single precision; only the stored values are rounded. With
    /// `FullVolume` and flip averaging, the first pass's posteriors are rounded to half
    /// precision before the flipped pass runs, so the two passes are never held in full
    /// precision at once.
    F16,
}

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
    /// Whole-volume (default, the CLI's) or sliding-window inference.
    pub inference: SynthSegInference,
    /// Crop the padded volume to the head before inference: the bounding box of the largest
    /// connected region brighter than 10 % of the normalised intensity range, grown by this
    /// many voxels on every side and rounded out to the pooling lattice. Everything outside it
    /// is background. `None` (the default, and the CLI's) runs on the whole padded volume.
    pub head_crop: Option<usize>,
    /// How the posteriors are stored (the default, single precision, is the CLI's).
    pub posteriors: PosteriorPrecision,
}

impl Default for SynthSegParams {
    fn default() -> Self {
        Self {
            version: SynthSegVersion::default(),
            crop: None,
            flip_averaging: true,
            topology_cleanup: true,
            sigma_smoothing: 0.5,
            inference: SynthSegInference::FullVolume,
            head_crop: None,
            posteriors: PosteriorPrecision::F32,
        }
    }
}

impl SynthSegParams {
    /// SynthSeg's `--fast`: no flip averaging and no topological cleanup. Roughly halves the
    /// runtime for a small accuracy cost.
    pub fn fast() -> Self {
        Self { flip_averaging: false, topology_cleanup: false, ..Self::default() }
    }

    /// Memory-bounded setting for constrained hosts (e.g. 32-bit WASM): the defaults with
    /// [`SynthSegInference::LOW_MEMORY`] windows, a 16-voxel head crop and half-precision
    /// posteriors.
    pub fn low_memory() -> Self {
        Self {
            inference: SynthSegInference::LOW_MEMORY,
            head_crop: Some(16),
            posteriors: PosteriorPrecision::F16,
            ..Self::default()
        }
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

    let window = match params.inference {
        SynthSegInference::FullVolume => None,
        SynthSegInference::SlidingWindow { patch, overlap } => {
            if patch < 64 {
                return Err(OnnxError::Shape(format!("the SynthSeg patch must be at least 64 voxels, not {patch}")));
            }
            let patch = round_up(patch, LATTICE);
            if overlap >= patch {
                return Err(OnnxError::Shape(format!(
                    "the SynthSeg window overlap must be less than the patch ({patch}), not {overlap}"
                )));
            }
            Some((patch, overlap))
        }
    };

    let pre = preprocess(magnitude, grid, affine, params);
    let model = OnnxModel::load(model_onnx)?;
    let flip = params.flip_averaging.then(|| params.version.labels().flip);
    match params.posteriors {
        PosteriorPrecision::F32 => infer::<f32>(&model, &pre, n_labels, window, flip, params, &mut progress),
        PosteriorPrecision::F16 => infer::<half::f16>(&model, &pre, n_labels, window, flip, params, &mut progress),
    }
}

/// Network, flip averaging and post-processing, with the posteriors stored as `T`.
#[cfg(feature = "onnx")]
fn infer<T: Posterior>(
    model: &OnnxModel,
    pre: &Preprocessed,
    n_labels: usize,
    window: Option<(usize, usize)>,
    flip: Option<&[usize]>,
    params: &SynthSegParams,
    progress: &mut impl FnMut(usize, usize),
) -> Result<SynthSegResult, OnnxError> {
    // A volume one window covers runs as a whole volume does.
    if let Some((patch, overlap)) = window.filter(|&(patch, _)| pre.dims.iter().any(|&d| d > patch)) {
        let mut post: Vec<T> = run_windows(model, &pre.data, pre.dims, n_labels, patch, overlap, flip, progress)?;
        if params.sigma_smoothing > 0.0 {
            blur_posteriors(&mut post, pre.dims, n_labels, params.sigma_smoothing);
        }
        return Ok(postprocess(post, pre, params));
    }

    let passes = 1 + flip.is_some() as usize;
    let post = run_pass(model, &pre.data, pre.dims, n_labels, params)?;
    progress(1, passes);
    // At half precision the first pass is rounded (and its single-precision copy freed) before
    // the second runs.
    let mut post = T::store(post);
    if let Some(table) = flip {
        let flipped = flip_axis0(&pre.data, pre.dims, 1);
        let post_f = run_pass(model, &flipped, pre.dims, n_labels, params)?;
        drop(flipped);
        progress(2, passes);
        merge_flipped(&mut post, &post_f, pre.dims, n_labels, table);
    }
    Ok(postprocess(post, pre, params))
}

/// A stored posterior value: `f32`, or `half::f16` for [`PosteriorPrecision::F16`]. Arithmetic
/// is in `f32` either way.
trait Posterior: Copy + Default + Send + Sync + 'static {
    fn get(self) -> f32;
    fn put(v: f32) -> Self;
    /// Convert a single-precision buffer (no copy when `Self` is `f32`).
    fn store(v: Vec<f32>) -> Vec<Self>;
}

impl Posterior for f32 {
    #[inline]
    fn get(self) -> f32 {
        self
    }
    #[inline]
    fn put(v: f32) -> Self {
        v
    }
    fn store(v: Vec<f32>) -> Vec<Self> {
        v
    }
}

#[cfg(feature = "onnx")]
impl Posterior for half::f16 {
    #[inline]
    fn get(self) -> f32 {
        self.to_f32()
    }
    #[inline]
    fn put(v: f32) -> Self {
        half::f16::from_f32(v)
    }
    fn store(v: Vec<f32>) -> Vec<Self> {
        v.into_iter().map(half::f16::from_f32).collect()
    }
}

/// Average the posteriors of the left-right flipped pass `flipped` (still mirrored along axis 0)
/// into `post`, mapping each channel to its left-right partner `table[c]`. In place: the
/// mirrored copy is never materialised.
#[cfg(feature = "onnx")]
fn merge_flipped<T: Posterior>(post: &mut [T], flipped: &[f32], dims: [usize; 3], n_labels: usize, table: &[usize]) {
    let plane = dims[1] * dims[2] * n_labels;
    crate::maybe_par_chunks_mut!(post, plane).enumerate().for_each(|(o0, out)| {
        let src = &flipped[(dims[0] - 1 - o0) * plane..(dims[0] - o0) * plane];
        for (o, f) in out.chunks_exact_mut(n_labels).zip(src.chunks_exact(n_labels)) {
            for c in 0..n_labels {
                o[c] = T::put(0.5 * (o[c].get() + f[table[c]]));
            }
        }
    });
}

/// The network input plus everything needed to map predictions back onto the caller's grid.
struct Preprocessed {
    /// Normalised network input in RAS orientation, C-order `[d0, d1, d2]`: the padded volume,
    /// or the head's box in it.
    data: Vec<f32>,
    dims: [usize; 3],
    /// Shape of the RAS-aligned volume before cropping and padding.
    aligned_dims: [usize; 3],
    /// Where the network grid's first voxel sits in the RAS-aligned volume, per axis (negative
    /// where the grid starts in padding).
    origin: [isize; 3],
    /// The network-grid range that holds aligned data, `[start, end)` per axis: inside the
    /// padding, and inside any crop.
    valid: [(usize, usize); 3],
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

/// The axis permutation and per-axis flips that take a volume into RAS orientation.
///
/// Returns `(perm, flip)` where RAS axis `i` is the source volume's axis `perm[i]`, reversed
/// when `flip[i]`. Equivalent to lab2im's `align_volume_to_ref(..., aff_ref=eye(4))`.
fn ras_alignment(affine: &[f64; 16]) -> ([usize; 3], [bool; 3]) {
    crate::geometry::alignment_to(affine, [AxisCode::R, AxisCode::A, AxisCode::S])
}

// ----------------------------------------------------------------------- preprocessing

fn round_up(n: usize, m: usize) -> usize {
    if n.is_multiple_of(m) { n } else { (n / m + 1) * m }
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
    let needs_resample = vs.iter().any(|&v| (v - TARGET_RES).abs() > 0.05);
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
    let mut data = pad_c_order(&data, dims, padded, pad, 1);
    let mut net_dims = padded;
    let mut origin: [isize; 3] = std::array::from_fn(|i| crop[i].0 as isize - pad[i].0 as isize);
    let mut valid = pad;

    // 6. optionally crop to the head. The box's corners are on the pooling lattice of the
    //    padded volume, so windows placed from the box's origin stay in phase with it.
    if let Some(bounds) = params.head_crop.and_then(|margin| head_box(&data, padded, margin)) {
        data = crop_c_order(&data, padded, bounds, 1);
        net_dims = std::array::from_fn(|i| bounds[i].1 - bounds[i].0);
        for i in 0..3 {
            origin[i] += bounds[i].0 as isize;
            let start = pad[i].0.max(bounds[i].0);
            let end = pad[i].1.min(bounds[i].1).max(start);
            valid[i] = (start - bounds[i].0, end - bounds[i].0);
        }
    }

    Preprocessed {
        data,
        dims: net_dims,
        aligned_dims,
        origin,
        valid,
        perm,
        flip,
        source_dims: source_dims_cm,
        res: TARGET_RES,
        out_dims,
        resampled: needs_resample,
    }
}

/// The head's bounding box in the normalised, padded C-order volume `data`: the largest
/// face-connected region above [`HEAD_THRESHOLD`], grown by `margin` voxels on every side and
/// rounded out to multiples of [`LATTICE`] (`dims` are multiples of it already). `None` when
/// nothing is above the threshold or the box is the whole volume.
fn head_box(data: &[f32], dims: [usize; 3], margin: usize) -> Option<[(usize, usize); 3]> {
    let above: Vec<u8> = data.iter().map(|&v| (v > HEAD_THRESHOLD) as u8).collect();
    // 6-connectivity is symmetric under axis permutation: the C-order volume is column-major
    // with the axes reversed.
    let head = largest_component(&above, (dims[2], dims[1], dims[0]), Connectivity::Face);
    drop(above);
    let mut lo = dims;
    let mut hi = [0usize; 3];
    for (l, _) in head.iter().enumerate().filter(|(_, &h)| h != 0) {
        let at = [l / (dims[1] * dims[2]), (l / dims[2]) % dims[1], l % dims[2]];
        for i in 0..3 {
            lo[i] = lo[i].min(at[i]);
            hi[i] = hi[i].max(at[i] + 1);
        }
    }
    if hi.contains(&0) {
        return None;
    }
    let bounds: [(usize, usize); 3] = std::array::from_fn(|i| {
        let start = lo[i].saturating_sub(margin) / LATTICE * LATTICE;
        let end = round_up((hi[i] + margin).min(dims[i]), LATTICE).min(dims[i]);
        (start, end)
    });
    (bounds != [(0, dims[0]), (0, dims[1]), (0, dims[2])]).then_some(bounds)
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

/// Window origins along an axis of `size` voxels for windows of `p` (both multiples of
/// [`LATTICE`]): the first at 0, the last at `size − p`, and the fewest windows whose steps — each a
/// multiple of the lattice, as even as the lattice allows — leave at least `overlap` voxels shared
/// between neighbours.
fn lattice_windows(size: usize, p: usize, overlap: usize) -> Vec<usize> {
    debug_assert!(size.is_multiple_of(LATTICE) && p.is_multiple_of(LATTICE), "{size} {p}");
    if size <= p {
        return vec![0];
    }
    let span = (size - p) / LATTICE;
    let max_step = (p.saturating_sub(overlap) / LATTICE).max(1);
    let steps = span.div_ceil(max_step);
    (0..=steps).map(|i| i * span / steps * LATTICE).collect()
}

/// Sliding-window posteriors (channels-last, `[d0, d1, d2, n_labels]`) of the network input
/// `data` (dims multiples of [`LATTICE`]): the network on each `patch`-cube window (clamped to
/// the volume on an axis it does not exceed), origins on the pooling lattice
/// ([`lattice_windows`]), blended with nnU-Net's Gaussian importance. With `flip` (the label
/// table's left-right partners), each window is also run mirrored along axis 0 and its
/// posteriors, mirrored back with each channel mapped to its partner, averaged in, as the
/// whole-volume flip averaging does. The blur is the caller's, once, on the blended posteriors.
///
/// The blending weights are summed before any window runs, so each window's posteriors go
/// straight into the normalised mean: the only whole-volume buffers are the posteriors (as `T`)
/// and one weight per voxel.
#[cfg(feature = "onnx")]
#[allow(clippy::too_many_arguments)]
fn run_windows<T: Posterior>(
    model: &OnnxModel,
    data: &[f32],
    dims: [usize; 3],
    n_labels: usize,
    patch: usize,
    overlap: usize,
    flip: Option<&[usize]>,
    progress: &mut impl FnMut(usize, usize),
) -> Result<Vec<T>, OnnxError> {
    use crate::bet::hdbet::gaussian_importance;
    let p: [usize; 3] = std::array::from_fn(|a| patch.min(dims[a]));
    let starts: [Vec<usize>; 3] = std::array::from_fn(|a| lattice_windows(dims[a], p[a], overlap));
    let mut origins = Vec::new();
    for &s0 in &starts[0] {
        for &s1 in &starts[1] {
            for &s2 in &starts[2] {
                origins.push([s0, s1, s2]);
            }
        }
    }
    let gauss = gaussian_importance(p);
    let shape = [1, p[0], p[1], p[2], 1];
    let plan = model.plan_for(&[&shape])?;
    let passes = 1 + flip.is_some() as usize;
    let total = origins.len() * passes;
    let mut done = 0;

    // Calls `f(volume offset, window offset)` for the start of each row of the window at `o`.
    let rows = |o: [usize; 3], f: &mut dyn FnMut(usize, usize)| {
        for z in 0..p[0] {
            for y in 0..p[1] {
                f(((o[0] + z) * dims[1] + o[1] + y) * dims[2] + o[2], (z * p[1] + y) * p[2]);
            }
        }
    };

    // 1 / (the summed Gaussian weight of the windows covering each voxel).
    let n: usize = dims.iter().product();
    let mut inv_weight = vec![0f32; n];
    for &o in &origins {
        rows(o, &mut |v, w| {
            for k in 0..p[2] {
                inv_weight[v + k] += gauss[w + k];
            }
        });
    }
    inv_weight.iter_mut().for_each(|w| *w = if *w > 0.0 { 1.0 / *w } else { 0.0 });

    let mut post = vec![T::default(); n * n_labels];
    let mut input = vec![0f32; p.iter().product()];
    let run = |x: Vec<f32>| -> Result<Vec<f32>, OnnxError> {
        let out = plan.run_single(&Tensor::new(shape.to_vec(), x))?;
        let expect = [1, p[0], p[1], p[2], n_labels];
        if out.shape != expect {
            return Err(OnnxError::Shape(format!(
                "SynthSeg returned {:?}, expected {:?} — do the weights match params.version?",
                out.shape, expect
            )));
        }
        Ok(out.data)
    };
    for &o in &origins {
        rows(o, &mut |v, w| input[w..w + p[2]].copy_from_slice(&data[v..v + p[2]]));
        let mut pred = run(input.clone())?;
        done += 1;
        progress(done, total);
        if let Some(table) = flip {
            let back = flip_axis0(&run(flip_axis0(&input, p, 1))?, p, n_labels);
            done += 1;
            progress(done, total);
            for (out, chunk) in pred.chunks_exact_mut(n_labels).zip(back.chunks_exact(n_labels)) {
                for c in 0..n_labels {
                    out[c] = 0.5 * (out[c] + chunk[table[c]]);
                }
            }
        }
        rows(o, &mut |v, w| {
            for k in 0..p[2] {
                let g = gauss[w + k] * inv_weight[v + k];
                let (a, b) = ((v + k) * n_labels, (w + k) * n_labels);
                for c in 0..n_labels {
                    post[a + c] = T::put(post[a + c].get() + g * pred[b + c]);
                }
            }
        });
    }
    Ok(post)
}

/// Per-channel separable Gaussian blur of a `[d0, d1, d2, n_ch]` posterior map.
///
/// Matches lab2im's `GaussianBlur`: kernel width `2·ceil(2.5σ)/2 + 1`, normalised, applied with
/// zero padding at the borders (TensorFlow's `SAME`) — note that is zero padding, not the
/// reflection [`gaussian_filter_anisotropic`] uses.
fn blur_posteriors<T: Posterior>(post: &mut [T], dims: [usize; 3], n_ch: usize, sigma: f64) {
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
fn blur_along<T: Posterior>(data: &mut [T], outer: usize, n: usize, inner: usize, w: &[f32], radius: usize) {
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
                for (h, v) in history[slot..slot + inner].iter_mut().zip(&line[i * inner..(i + 1) * inner]) {
                    *h = v.get();
                }
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
                            line[src * inner + c].get()
                        };
                }
                line[o * inner + c] = T::put(acc);
            }
        }
    });
}

// ----------------------------------------------------------------------- postprocessing

#[cfg(feature = "onnx")]
fn postprocess<T: Posterior>(mut post: Vec<T>, pre: &Preprocessed, params: &SynthSegParams) -> SynthSegResult {
    let labels = params.version.labels();
    let n_labels = labels.ids.len();
    let dims = pre.dims;
    let n_vox = dims.iter().product::<usize>();
    // Connected-component work happens on the network's grid; 6-connectivity is symmetric under
    // axis permutation, so the C-order volume can be handed over as column-major reversed.
    let cc_dims = (dims[2], dims[1], dims[0]);

    // Foreground: everything but the background channel, thresholded and reduced to one blob.
    let mut fg = vec![0u8; n_vox];
    for (v, chunk) in fg.iter_mut().zip(post.chunks_exact(n_labels)) {
        *v = (chunk[1..].iter().map(|p| p.get()).sum::<f32>() > 0.25) as u8;
    }
    let fg = largest_component(&fg, cc_dims, Connectivity::Face);
    for (keep, chunk) in fg.iter().zip(post.chunks_exact_mut(n_labels)) {
        if *keep == 0 {
            chunk[1..].fill(T::default());
        }
    }
    drop(fg);

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
                *v = members.iter().any(|&c| chunk[c].get() > 0.25) as u8;
            }
            let mask = largest_component(&mask, cc_dims, Connectivity::Face);
            for (keep, chunk) in mask.iter().zip(post.chunks_exact_mut(n_labels)) {
                if *keep == 0 {
                    for &c in &members {
                        chunk[c] = T::default();
                    }
                }
            }
        }
    } else {
        // `--fast` keeps only the posteriors that clear a slightly lower bar.
        for chunk in post.chunks_exact_mut(n_labels) {
            for c in chunk.iter_mut().skip(1) {
                if c.get() <= 0.2 {
                    *c = T::default();
                }
            }
        }
    }

    // Over the voxels that hold data (the padding dropped): renormalise, take the argmax, and
    // sum the soft volumes.
    let valid = pre.valid;
    let valid_dims: [usize; 3] = std::array::from_fn(|i| valid[i].1 - valid[i].0);
    let mut volumes = vec![0f64; n_labels];
    let mut hard = Vec::with_capacity(valid_dims.iter().product::<usize>());
    let mut p = vec![0f32; n_labels];
    for o0 in valid[0].0..valid[0].1 {
        for o1 in valid[1].0..valid[1].1 {
            let row = ((o0 * dims[1] + o1) * dims[2] + valid[2].0) * n_labels;
            for chunk in post[row..row + valid_dims[2] * n_labels].chunks_exact(n_labels) {
                p.iter_mut().zip(chunk).for_each(|(p, c)| *p = c.get());
                let total: f32 = p.iter().sum();
                if total > 0.0 {
                    p.iter_mut().for_each(|v| *v /= total);
                }
                let mut best = 0usize;
                for c in 1..n_labels {
                    if p[c] > p[best] {
                        best = c;
                    }
                }
                hard.push(labels.ids[best]);
                // Accumulate in f64: the reference sums ~10^6 f32 posteriors per structure and
                // loses about 0.4 % to rounding doing so.
                for c in 1..n_labels {
                    volumes[c] += p[c] as f64;
                }
            }
        }
    }
    drop(post);
    let voxel_volume = pre.res * pre.res * pre.res;
    volumes.iter_mut().for_each(|v| *v *= voxel_volume);

    // Place the valid block in the RAS-aligned volume (background around it: the crop, and
    // anything outside a head crop), then undo the alignment.
    let at: [(usize, usize); 3] = std::array::from_fn(|i| {
        let start = (valid[i].0 as isize + pre.origin[i]) as usize;
        (start, start + valid_dims[i])
    });
    let aligned = if valid_dims == pre.aligned_dims {
        hard
    } else {
        let mut out = vec![labels.ids[0]; pre.aligned_dims.iter().product()];
        for o0 in 0..valid_dims[0] {
            for o1 in 0..valid_dims[1] {
                let src = (o0 * valid_dims[1] + o1) * valid_dims[2];
                let dst = ((o0 + at[0].0) * pre.aligned_dims[1] + o1 + at[1].0) * pre.aligned_dims[2] + at[2].0;
                out[dst..dst + valid_dims[2]].copy_from_slice(&hard[src..src + valid_dims[2]]);
            }
        }
        out
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

    #[cfg(feature = "onnx")]
    #[test]
    fn a_window_under_64_or_an_overlap_of_the_whole_patch_is_refused() {
        let grid = Grid::new(4, 4, 4, 1.0, 1.0, 1.0);
        let mag = vec![1.0; 64];
        let aff = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        for (patch, overlap) in [(63, 0), (32, 0), (128, 128), (100, 112)] {
            let params = SynthSegParams { inference: SynthSegInference::SlidingWindow { patch, overlap }, ..SynthSegParams::default() };
            let err = synthseg(&mag, &grid, &aff, &[], &params, |_, _| {}).err().expect("refused");
            assert!(matches!(err, OnnxError::Shape(_)), "{patch} {overlap}: {err:?}");
        }
    }

    #[test]
    fn low_memory_is_the_defaults_with_windows_a_head_crop_and_half_precision() {
        let p = SynthSegParams::low_memory();
        assert_eq!(p.inference, SynthSegInference::LOW_MEMORY);
        assert_eq!(p.posteriors, PosteriorPrecision::F16);
        assert!(p.head_crop.is_some());
        let whole = SynthSegParams { inference: SynthSegInference::FullVolume, head_crop: None, posteriors: PosteriorPrecision::F32, ..p };
        assert_eq!(whole, SynthSegParams::default());
    }

    #[test]
    fn windows_sit_on_the_pooling_lattice_and_keep_the_overlap() {
        for (size, p, overlap) in [(256, 96, 48), (256, 96, 64), (192, 96, 48), (256, 128, 48), (256, 128, 64), (176, 128, 32), (240, 112, 0)] {
            let w = lattice_windows(size, p, overlap);
            assert_eq!(w[0], 0);
            assert_eq!(*w.last().unwrap(), size - p, "{size} {p} {overlap}: {w:?}");
            assert!(w.iter().all(|o| o % LATTICE == 0), "{w:?}");
            for pair in w.windows(2) {
                assert!(pair[1] > pair[0] && p - (pair[1] - pair[0]) >= overlap, "{size} {p} {overlap}: {w:?}");
            }
            // No window could be dropped: one fewer step would have to exceed the largest allowed.
            let max_step = (p - overlap) / LATTICE * LATTICE;
            assert!(w.len() == 2 || (size - p) > (w.len() - 2) * max_step, "{size} {p} {overlap}: {w:?}");
        }
        assert_eq!(lattice_windows(256, 96, 48), vec![0, 32, 80, 112, 160]);
        assert_eq!(lattice_windows(192, 96, 48), vec![0, 48, 96]);
        assert_eq!(lattice_windows(256, 128, 48), vec![0, 64, 128]);
        assert_eq!(lattice_windows(96, 96, 48), vec![0]);
        assert_eq!(lattice_windows(64, 96, 48), vec![0]);
    }

    #[test]
    fn the_head_box_is_the_largest_bright_region_grown_and_on_the_lattice() {
        let dims = [64usize, 96, 128];
        let mut data = vec![0f32; dims.iter().product()];
        let at = |a: usize, b: usize, c: usize| (a * dims[1] + b) * dims[2] + c;
        // The head, and a brighter speck elsewhere that is not connected to it.
        for a in 20..40 {
            for b in 30..61 {
                for c in 40..90 {
                    data[at(a, b, c)] = 0.5;
                }
            }
        }
        data[at(2, 2, 2)] = 1.0;
        let bounds = head_box(&data, dims, 4).expect("a box");
        assert_eq!(bounds, [(16, 48), (16, 80), (32, 96)]);
        // A margin that reaches the edges is clamped; a box of the whole volume is no crop.
        assert_eq!(head_box(&data, dims, 40), None);
        assert_eq!(head_box(&vec![0f32; data.len()], dims, 4), None);
    }

    #[test]
    fn blur_at_half_precision_follows_single_precision() {
        let dims = [6usize, 7, 8];
        let n_ch = 3;
        let n = dims.iter().product::<usize>() * n_ch;
        let src: Vec<f32> = (0..n).map(|i| ((i * 7919) % 101) as f32 / 101.0).collect();
        let mut single = src.clone();
        blur_posteriors(&mut single, dims, n_ch, 0.5);
        #[cfg(feature = "onnx")]
        {
            let mut half: Vec<half::f16> = <half::f16 as Posterior>::store(src);
            blur_posteriors(&mut half, dims, n_ch, 0.5);
            for (a, b) in single.iter().zip(&half) {
                assert!((a - b.to_f32()).abs() < 2e-3, "{a} vs {b}");
            }
        }
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn merging_the_flipped_pass_in_place_matches_mirroring_it_first() {
        let dims = [5usize, 3, 2];
        let n_ch = 3;
        let table = [0usize, 2, 1];
        let n = dims.iter().product::<usize>() * n_ch;
        let post: Vec<f32> = (0..n).map(|i| i as f32 * 0.25).collect();
        let flipped: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.5).collect();
        let mut expected = post.clone();
        let mirrored = flip_axis0(&flipped, dims, n_ch);
        for (out, chunk) in expected.chunks_exact_mut(n_ch).zip(mirrored.chunks_exact(n_ch)) {
            for c in 0..n_ch {
                out[c] = 0.5 * (out[c] + chunk[table[c]]);
            }
        }
        let mut got = post;
        merge_flipped(&mut got, &flipped, dims, n_ch, &table);
        assert_eq!(got, expected);
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

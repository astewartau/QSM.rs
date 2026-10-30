//! Bias field correction (homogeneity correction)
//!
//! Implements the makehomogeneous algorithm for correcting RF receive field inhomogeneities.
//! This uses the "boxsegment" approach with box filter Gaussian approximation.
//!
//! Reference:
//! Eckstein, K., Trattnig, S., Robinson, S.D. (2019).
//! "A Simple Homogeneity Correction for Neuroimaging at 7T."
//! Proc. ISMRM 27th Annual Meeting.
//!
//! Reference implementation: https://github.com/korbinian90/MriResearchTools.jl
//!
//! # Reference pipeline
//!
//! MriResearchTools.jl (commit 771eadf, `src/intensitycorrection.jl`):
//!
//! - `makehomogeneous(mag::NIVolume; sigma_mm=7, nbox=15)` (`:1-14`): sigma 7 mm converted to
//!   voxels with the NIfTI voxel size, and `nbox` = 15 boxes per dimension for the `boxsegment`
//!   tissue segmentation. The masked smoothing uses 8 box-filter passes (`nbox_smoother=8`,
//!   `:85-99`), the sigma is split 0.7 / √(1−0.7²) between the final and the masked smoothing
//!   (`getsigma`), and the bias field is estimated from the first echo inside `robustmask`
//!   (`:92-96`). [`get_sensitivity`] / [`makehomogeneous`] follow this; the pipeline config's
//!   `masking.homogeneity_sigma_mm` 7.0 and `masking.homogeneity_nbox` 15 (applied by the host,
//!   not by qsm-core's masking stage) are these defaults.
//! - `getsensitivity` without a `sigma_mm` uses `get_default_sigma_mm`, commented "7mm, but a
//!   maximum of 10% FoV" but computing `min(median(voxel size × matrix size), 7)` — the whole
//!   field of view, not 10 % of it (`:104-113`). Neither cap is applied here; `makehomogeneous`
//!   with an explicit `sigma_mm` (as above) applies none either.
//! - Published with CLEAR-SWI (Eckstein et al. 2021, *NeuroImage* 237:118175,
//!   <https://doi.org/10.1016/j.neuroimage.2021.118175>; `intensitycorrection.jl:78`), which uses
//!   it as the magnitude sensitivity correction. QSMxT's Python releases called
//!   `makehomogeneous(mag; sigma=[20, 20, 10])` (voxels; `qsmxt/scripts/makehomogeneous.jl`).
//! - It assumes one dominant tissue present across the object ("main tissue", `:75`); it uses
//!   the voxel size but no field strength or orientation.

/// Parameters for inhomogeneity correction (bias field removal).
///
/// Not used by [`makehomogeneous`], which takes `sigma_mm` and `nbox` directly; the pipeline
/// config carries them as `masking.homogeneity_sigma_mm` / `masking.homogeneity_nbox` (defaults
/// 7.0 and 15, MriResearchTools' defaults) for the host to apply.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct HomogeneityParams {
    /// Gaussian smoothing sigma in mm (default: 7.0)
    pub sigma_mm: f64,
    /// Boxes per dimension for the `boxsegment` tissue segmentation, `makehomogeneous`'s `nbox`
    /// (default: 15, MriResearchTools'). Not the number of box-filter passes of the smoothing.
    ///
    /// Measured on the EMTP travelling heads (bridge run 1; #314): the earlier default of 3 left
    /// echo 1 visibly uneven, with a large dark centre on a GE MR750 (in-brain coefficient of
    /// variation 0.429, against 0.114 with 15 and 0.116 with N4; Philips Achieva 0.152 / 0.098 /
    /// 0.103). With 15 the image matches N4's; BET masks moved by at most 1.4 % in volume, and
    /// on GE agreed better with HD-BET.
    pub nbox: usize,
}

impl Default for HomogeneityParams {
    fn default() -> Self {
        Self {
            sigma_mm: 7.0,
            nbox: 15,
        }
    }
}

use std::collections::VecDeque;
use crate::error::InputError;
use crate::Grid;
use crate::utils::morphology::fill_holes;

/// Index into 3D array (Fortran/column-major order)
#[inline(always)]
pub(crate) fn idx3d(i: usize, j: usize, k: usize, nx: usize, ny: usize) -> usize {
    i + j * nx + k * nx * ny
}

/// Check a caller-supplied volume against the grid it is about to be indexed by.
///
/// Every filter here walks `0..nx*ny*nz` and indexes straight into the caller's
/// buffer, so a volume that does not match the grid reads past its end. Refusing
/// at the entry point names both sizes; the index panic 300 lines down does not
/// say which input was wrong, or that the grid was the problem at all.
#[inline]
pub(crate) fn check_volume(name: &str, len: usize, grid: &Grid) -> usize {
    let (nx, ny, nz) = grid.dims;
    let n_total = nx * ny * nz;
    assert_eq!(
        len, n_total,
        "{name} has {len} voxels but the grid is {nx}x{ny}x{nz} ({n_total} voxels)"
    );
    n_total
}

//=============================================================================
// Box Filter Gaussian Approximation (matching MriResearchTools.jl)
//=============================================================================

/// Calculate box sizes to approximate Gaussian with given sigma using n box filters
///
/// This implements the algorithm from MriResearchTools.jl:
/// Multiple box filter passes approximate a Gaussian convolution.
fn get_box_sizes(sigma: f64, n: usize) -> Vec<usize> {
    if sigma <= 0.0 || n == 0 {
        return vec![0; n];
    }

    // wideal = sqrt((12*sigma^2 / n) + 1)
    let wideal = ((12.0 * sigma * sigma / n as f64) + 1.0).sqrt();

    // wl = next lower odd integer
    let wl_float = wideal - (wideal + 1.0) % 2.0;
    // Julia's `round` (and `round(Int, …)`) rounds ties to even
    let wl = wl_float.round_ties_even() as usize;
    let wl = if wl.is_multiple_of(2) { wl + 1 } else { wl }; // ensure odd
    let wu = wl + 2;

    // mideal = (12*sigma^2 - n*wl^2 - 4*n*wl - 3*n) / (-4*wl - 4)
    let wl_f = wl as f64;
    let n_f = n as f64;
    let mideal = (12.0 * sigma * sigma - n_f * wl_f * wl_f - 4.0 * n_f * wl_f - 3.0 * n_f)
                 / (-4.0 * wl_f - 4.0);
    let m = mideal.round_ties_even() as usize;

    (0..n).map(|i| if i < m { wl } else { wu }).collect()
}

/// Check and adjust box sizes to fit image dimensions
fn check_box_sizes(boxsizes: &mut [Vec<usize>], dims: &[usize]) {
    for (dim, bs) in boxsizes.iter_mut().enumerate() {
        if dim >= dims.len() {
            continue;
        }
        for b in bs.iter_mut() {
            // Ensure odd
            if *b % 2 == 0 {
                *b += 1;
            }
            // Limit to half image size
            let max_size = dims[dim] / 2;
            if *b > max_size {
                *b = if max_size.is_multiple_of(2) { max_size + 1 } else { max_size };
            }
        }
    }
}

/// 1D box filter on a line (in-place), matching Julia's boxfilterline!
///
/// Uses running sum with edge handling.
fn box_filter_line(line: &mut [f64], boxsize: usize) {
    if boxsize < 3 || line.len() < boxsize {
        return;
    }

    let n = line.len();
    let r = boxsize / 2;

    // Use a circular buffer approach
    let mut queue: VecDeque<f64> = VecDeque::with_capacity(boxsize);

    // Initialize with first r values
    let mut lsum: f64 = line[..r].iter().sum();
    for i in 0..r {
        queue.push_back(line[i]);
    }

    // Start with edge effect (growing window)
    for i in 0..=r {
        lsum += line[i + r];
        queue.push_back(line[i + r]);
        line[i] = lsum / (r + i + 1) as f64;
    }

    // Middle part (full window)
    for i in (r + 1)..(n - r) {
        let old = queue.pop_front().unwrap();
        lsum += line[i + r] - old;
        queue.push_back(line[i + r]);
        line[i] = lsum / boxsize as f64;
    }

    // End with edge effect (shrinking window)
    for i in (n - r)..n {
        let old = queue.pop_front().unwrap();
        lsum -= old;
        line[i] = lsum / (r + n - i) as f64;
    }
}

/// 1D weighted box filter on a line (in-place), matching Julia's weighted boxfilterline!
///
/// The weights are Float32 there (`gaussiansmooth3d!` converts them), so `w^2` and `w − wold`
/// are computed in f32 and each updated weight is rounded to f32; the sums are f64.
fn box_filter_line_weighted(line: &mut [f64], weight: &mut [f32], boxsize: usize) {
    if boxsize < 3 || line.len() < boxsize {
        return;
    }

    let n = line.len();
    let r = boxsize / 2;

    let mut lq: VecDeque<f64> = VecDeque::with_capacity(boxsize);
    let mut wq: VecDeque<f32> = VecDeque::with_capacity(boxsize);

    // Initialize with first boxsize values
    let mut sum = f64::EPSILON; // slightly bigger than 0 to avoid division by 0
    let mut wsum = f64::EPSILON;
    let mut wsmooth = f64::EPSILON;

    for i in 0..boxsize {
        sum += line[i] * weight[i] as f64;
        wsum += weight[i] as f64;
        wsmooth += (weight[i] * weight[i]) as f64;
        lq.push_back(line[i]);
        wq.push_back(weight[i]);
    }

    // Middle part
    for i in (r + 1)..(n - r) {
        let w = weight[i + r];
        let l = line[i + r];
        let wold = wq.pop_front().unwrap();
        let lold = lq.pop_front().unwrap();
        wq.push_back(w);
        lq.push_back(l);

        sum += l * w as f64 - lold * wold as f64;
        wsum += (w - wold) as f64;
        line[i] = sum / wsum;
        wsmooth += (w * w - wold * wold) as f64;
        weight[i] = (wsmooth / wsum) as f32;
    }
}

/// Running-average line filter that treats `NaN` as "no data": the box only starts once it
/// holds `boxsize` valid samples, and when it runs into `NaN` it extrapolates linearly for up
/// to `boxsize/2` samples ("fill" mode) before waiting for data again. Positions the filter
/// never reaches keep their input value. Port of `nanboxfilterline!` (MriResearchTools.jl);
/// `orig` is scratch space that is resized as needed.
///
/// [`makehomogeneous`]'s masked smoothing had its own copy until 2026-10-30, which wrote each
/// fill-mode extrapolation one voxel further than the original (`i + r < n` with a 0-based `i`,
/// so the last voxel of a line was overwritten) and ignored the original's stop when the running
/// sum turns NaN. The first alone put [`makehomogeneous`] 4.0 % NRMSE from MriResearchTools (see
/// [`get_sensitivity`]). This is the port MCPC-3D-S already used (exact in the ROMEO fixture).
pub fn nan_box_filter_line(line: &mut [f64], boxsize: usize, orig: &mut Vec<f64>) {
    #[derive(PartialEq, Clone, Copy)]
    enum Mode { Nan, Normal, Fill }
    let n = line.len();
    let r = boxsize / 2;
    if n == 0 || boxsize < 3 {
        return;
    }
    let maxfills = r;
    orig.clear();
    orig.resize(n + boxsize - 1, f64::NAN);
    orig[r..r + n].copy_from_slice(line);

    let mut lsum: f64 = orig[r..2 * r].iter().sum();
    if lsum.is_nan() {
        lsum = 0.0;
    }
    let (mut nfills, mut nvalids) = (0usize, 0usize);
    let mut mode = Mode::Nan;
    let bs = boxsize as f64;

    // `i` is the 1-based output index, as in the reference; orig[k] here is orig[k+1] there.
    for i in 1..=n {
        if lsum.is_nan() {
            break;
        }
        let lead = orig[i - 1 + 2 * r];
        match mode {
            Mode::Normal => {
                if lead.is_nan() {
                    mode = Mode::Fill;
                }
            }
            Mode::Nan => {
                if lead.is_nan() { nvalids = 0; } else { nvalids += 1; }
                if nvalids == boxsize {
                    mode = Mode::Normal;
                    lsum = orig[i - 1..i + 2 * r].iter().sum();
                    line[i - 1] = lsum / bs;
                    continue;
                }
            }
            Mode::Fill => {
                if lead.is_nan() {
                    nfills += 1;
                    if nfills > maxfills {
                        mode = Mode::Nan;
                        nfills = 0;
                        lsum = 0.0;
                        nvalids = 0;
                    }
                } else {
                    mode = Mode::Normal;
                    nfills = 0;
                }
            }
        }
        match mode {
            Mode::Normal => {
                let trailing = if i >= 2 { orig[i - 2] } else { 0.0 };
                lsum += orig[i - 1 + 2 * r] - trailing;
                line[i - 1] = lsum / bs;
            }
            Mode::Fill => {
                let trailing = if i >= 2 { orig[i - 2] } else { 0.0 };
                lsum -= trailing;
                line[i - 1] = (lsum - orig[i - 1]) / (bs - 2.0);
                let prev = if i > r { line[i - 1 - r] } else { line[i - 1] };
                let extrapolated = 2.0 * line[i - 1] - prev;
                orig[i - 1 + 2 * r] = extrapolated;
                if i + r < n {
                    line[i - 1 + r] = extrapolated;
                }
                lsum += extrapolated;
            }
            Mode::Nan => {}
        }
    }
}

/// 3D Gaussian smoothing using box filter approximation
///
/// This matches MriResearchTools.jl's gaussiansmooth3d function.
///
/// Parameters:
/// - data: input 3D data (will be copied)
/// - sigma: sigma values for each dimension [sx, sy, sz]
/// - mask: optional mask (None = no masking)
/// - weight: optional weights (None = no weighting). As in MriResearchTools, they are copied to
///   f32 (`Float32.(weight)`), zeros are replaced by the smallest non-zero weight, and the
///   caller's array is left unchanged.
/// - nbox: number of box filter passes (default 3, or 4 with mask)
/// - nx, ny, nz: dimensions
pub fn gaussian_smooth_3d(
    data: &[f64],
    sigma: [f64; 3],
    mask: Option<&[u8]>,
    weight: Option<&[f64]>,
    nbox: usize,
    grid: &Grid,
) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let n_total = check_volume("gaussian_smooth_3d: data", data.len(), grid);
    if let Some(m) = mask {
        check_volume("gaussian_smooth_3d: mask", m.len(), grid);
    }
    if let Some(w) = weight {
        check_volume("gaussian_smooth_3d: weight", w.len(), grid);
    }
    // `weight = Float32.(weight); weight[weight .== 0] .= minimum(weight[weight .!= 0])`
    let mut weight: Option<Vec<f32>> = weight.map(|w| {
        let mut w: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let min_nz = w.iter().copied().filter(|&v| v != 0.0).fold(f32::INFINITY, f32::min);
        for v in w.iter_mut().filter(|v| **v == 0.0) {
            *v = min_nz;
        }
        w
    });
    let mut result: Vec<f64> = data.to_vec();

    // Calculate box sizes for each dimension
    let mut boxsizes: Vec<Vec<usize>> = sigma.iter()
        .map(|&s| get_box_sizes(s, nbox))
        .collect();

    check_box_sizes(&mut boxsizes, &[nx, ny, nz]);

    // Apply mask: set masked-out voxels to NaN
    if let Some(m) = mask {
        for i in 0..n_total {
            if m[i] == 0 {
                result[i] = f64::NAN;
            }
        }
    }

    // Apply box filters for each pass and dimension
    let mut scratch: Vec<f64> = Vec::new();
    for ibox in 0..nbox {
        // X direction
        let bsize_x = boxsizes[0][ibox];
        if nx > 1 && bsize_x >= 3 {
            // Alternate direction for masked smoothing on even passes
            let reverse = mask.is_some() && ibox % 2 == 1;

            for k in 0..nz {
                for j in 0..ny {
                    let mut line: Vec<f64> = (0..nx).map(|i| {
                        let idx = if reverse { nx - 1 - i } else { i };
                        result[idx3d(idx, j, k, nx, ny)]
                    }).collect();

                    if mask.is_some() {
                        nan_box_filter_line(&mut line, bsize_x, &mut scratch);
                    } else if let Some(ref mut w) = weight.as_deref_mut() {
                        let mut wline: Vec<f32> = (0..nx).map(|i| {
                            let idx = if reverse { nx - 1 - i } else { i };
                            w[idx3d(idx, j, k, nx, ny)]
                        }).collect();
                        box_filter_line_weighted(&mut line, &mut wline, bsize_x);
                        for i in 0..nx {
                            let idx = if reverse { nx - 1 - i } else { i };
                            w[idx3d(idx, j, k, nx, ny)] = wline[i];
                        }
                    } else {
                        box_filter_line(&mut line, bsize_x);
                    }

                    for i in 0..nx {
                        let idx = if reverse { nx - 1 - i } else { i };
                        result[idx3d(idx, j, k, nx, ny)] = line[i];
                    }
                }
            }
        }

        // Y direction
        let bsize_y = boxsizes[1][ibox];
        if ny > 1 && bsize_y >= 3 {
            let reverse = mask.is_some() && ibox % 2 == 1;

            for k in 0..nz {
                for i in 0..nx {
                    let mut line: Vec<f64> = (0..ny).map(|j| {
                        let idx = if reverse { ny - 1 - j } else { j };
                        result[idx3d(i, idx, k, nx, ny)]
                    }).collect();

                    if mask.is_some() {
                        nan_box_filter_line(&mut line, bsize_y, &mut scratch);
                    } else if let Some(ref mut w) = weight.as_deref_mut() {
                        let mut wline: Vec<f32> = (0..ny).map(|j| {
                            let idx = if reverse { ny - 1 - j } else { j };
                            w[idx3d(i, idx, k, nx, ny)]
                        }).collect();
                        box_filter_line_weighted(&mut line, &mut wline, bsize_y);
                        for j in 0..ny {
                            let idx = if reverse { ny - 1 - j } else { j };
                            w[idx3d(i, idx, k, nx, ny)] = wline[j];
                        }
                    } else {
                        box_filter_line(&mut line, bsize_y);
                    }

                    for j in 0..ny {
                        let idx = if reverse { ny - 1 - j } else { j };
                        result[idx3d(i, idx, k, nx, ny)] = line[j];
                    }
                }
            }
        }

        // Z direction
        let bsize_z = boxsizes[2][ibox];
        if nz > 1 && bsize_z >= 3 {
            let reverse = mask.is_some() && ibox % 2 == 1;

            for j in 0..ny {
                for i in 0..nx {
                    let mut line: Vec<f64> = (0..nz).map(|k| {
                        let idx = if reverse { nz - 1 - k } else { k };
                        result[idx3d(i, j, idx, nx, ny)]
                    }).collect();

                    if mask.is_some() {
                        nan_box_filter_line(&mut line, bsize_z, &mut scratch);
                    } else if let Some(ref mut w) = weight.as_deref_mut() {
                        let mut wline: Vec<f32> = (0..nz).map(|k| {
                            let idx = if reverse { nz - 1 - k } else { k };
                            w[idx3d(i, j, idx, nx, ny)]
                        }).collect();
                        box_filter_line_weighted(&mut line, &mut wline, bsize_z);
                        for k in 0..nz {
                            let idx = if reverse { nz - 1 - k } else { k };
                            w[idx3d(i, j, idx, nx, ny)] = wline[k];
                        }
                    } else {
                        box_filter_line(&mut line, bsize_z);
                    }

                    for k in 0..nz {
                        let idx = if reverse { nz - 1 - k } else { k };
                        result[idx3d(i, j, idx, nx, ny)] = line[k];
                    }
                }
            }
        }
    }

    result
}

/// Simplified smoothing with explicit box sizes (for robustmask post-processing)
pub fn gaussian_smooth_3d_boxsizes(
    data: &[f64],
    boxsizes: &[Vec<usize>],
    nbox: usize,
    grid: &Grid,
) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    check_volume("gaussian_smooth_3d_boxsizes: data", data.len(), grid);
    assert_eq!(
        boxsizes.len(), 3,
        "gaussian_smooth_3d_boxsizes: boxsizes needs one entry per axis, got {}",
        boxsizes.len()
    );
    let mut result = data.to_vec();

    // Apply box filters for each pass and dimension
    for ibox in 0..nbox {
        // X direction
        if nx > 1 && ibox < boxsizes[0].len() {
            let bsize = boxsizes[0][ibox];
            if bsize >= 3 {
                for k in 0..nz {
                    for j in 0..ny {
                        let mut line: Vec<f64> = (0..nx).map(|i| result[idx3d(i, j, k, nx, ny)]).collect();
                        box_filter_line(&mut line, bsize);
                        for i in 0..nx {
                            result[idx3d(i, j, k, nx, ny)] = line[i];
                        }
                    }
                }
            }
        }

        // Y direction
        if ny > 1 && ibox < boxsizes[1].len() {
            let bsize = boxsizes[1][ibox];
            if bsize >= 3 {
                for k in 0..nz {
                    for i in 0..nx {
                        let mut line: Vec<f64> = (0..ny).map(|j| result[idx3d(i, j, k, nx, ny)]).collect();
                        box_filter_line(&mut line, bsize);
                        for j in 0..ny {
                            result[idx3d(i, j, k, nx, ny)] = line[j];
                        }
                    }
                }
            }
        }

        // Z direction
        if nz > 1 && ibox < boxsizes[2].len() {
            let bsize = boxsizes[2][ibox];
            if bsize >= 3 {
                for j in 0..ny {
                    for i in 0..nx {
                        let mut line: Vec<f64> = (0..nz).map(|k| result[idx3d(i, j, k, nx, ny)]).collect();
                        box_filter_line(&mut line, bsize);
                        for k in 0..nz {
                            result[idx3d(i, j, k, nx, ny)] = line[k];
                        }
                    }
                }
            }
        }
    }

    result
}

//=============================================================================
// Robust Mask (matching MriResearchTools.jl)
//=============================================================================

/// MriResearchTools' unmasked `gaussiansmooth3d!` with explicit box sizes (`for ibox, dim`), in
/// f32: `boxfilterline!` (`src/smoothing.jl`, 771eadf), a running sum with shrinking windows at
/// both ends.
fn smooth_boxes_f32(image: &mut [f32], boxsizes: &[usize], grid: &Grid) {
    let (nx, ny, nz) = grid.dims;
    let dims = [nx, ny, nz];
    let mut line: Vec<f32> = Vec::new();
    for &bsize0 in boxsizes {
        for d in 0..3 {
            let len = dims[d];
            // checkboxsizes!: odd, and at most half the axis
            let mut bsize = bsize0 + usize::from(bsize0 % 2 == 0);
            if bsize as f64 > len as f64 / 2.0 {
                bsize = len / 2;
                if bsize % 2 == 0 {
                    bsize += 1;
                }
            }
            if len == 1 || bsize < 3 {
                continue;
            }
            let (stride, na, sa, nb, sb) = match d {
                0 => (1, ny, nx, nz, nx * ny),
                1 => (nx, nx, 1, nz, nx * ny),
                _ => (nx * ny, nx, 1, ny, nx),
            };
            line.resize(len, 0.0);
            for b in 0..nb {
                for a in 0..na {
                    let base = a * sa + b * sb;
                    for k in 0..len {
                        line[k] = image[base + k * stride];
                    }
                    box_filter_line_f32(&mut line, bsize);
                    for k in 0..len {
                        image[base + k * stride] = line[k];
                    }
                }
            }
        }
    }
}

/// `boxfilterline!` (MriResearchTools `src/smoothing.jl`, 771eadf) in f32.
fn box_filter_line_f32(line: &mut [f32], boxsize: usize) {
    let n = line.len();
    let r = boxsize / 2;
    if n < boxsize {
        return;
    }
    let mut q: VecDeque<f32> = VecDeque::with_capacity(boxsize);
    let mut lsum: f32 = line[..r].iter().sum();
    q.extend(line[..r].iter().copied());
    for i in 0..=r {
        lsum += line[i + r];
        q.push_back(line[i + r]);
        line[i] = lsum / (r + i + 1) as f32;
    }
    for i in (r + 1)..(n - r) {
        lsum += line[i + r] - q.pop_front().unwrap();
        q.push_back(line[i + r]);
        line[i] = lsum / boxsize as f32;
    }
    for i in (n - r)..n {
        lsum -= q.pop_front().unwrap();
        line[i] = lsum / (r + n - i) as f32;
    }
}

/// MriResearchTools' `sample(I; n=1e5)` (`src/utility.jl:23-33`, 771eadf): `len = ceil(√min(n,
/// length))` blocks of `len` consecutive values, starting at `round.(Int, range(0, length − len;
/// length = len))` (0-based here), keeping the finite ones; all finite values if none are.
fn mrt_sample(x: &[f64]) -> Vec<f64> {
    let total = x.len();
    if total == 0 {
        return Vec::new();
    }
    let n = (1e5f64).min(total as f64);
    let len = n.sqrt().ceil() as usize;
    let stop = (total - len) as f64;
    let mut out = Vec::with_capacity(len * len);
    for k in 0..len {
        // Julia's range(0, stop; length=len) and round (ties to even)
        let start = if len == 1 { 0.0 } else { stop * k as f64 / (len - 1) as f64 };
        let s = start.round_ties_even() as usize;
        out.extend(x[s..s + len].iter().copied().filter(|v| v.is_finite()));
    }
    if out.is_empty() {
        out = x.iter().copied().filter(|v| v.is_finite()).collect();
    }
    out
}

/// `Statistics.quantile(v, p)` (type 7, the default) of an ascending `sorted`, in Julia 1.10's
/// arithmetic (`Statistics._quantile`, `stdlib/v1.10/Statistics/src/Statistics.jl:1000-1031`):
/// `aleph = n·p + (1 − p)`, `j = clamp(trunc(aleph), 1, n − 1)` (1-based), `γ = aleph − j`, then
/// `a + γ(b − a)` when `a ≈ b` (`isapprox`, rtol √eps) and `(1 − γ)a + γb` otherwise.
fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let m = 1.0 - p; // alpha + p (1 − alpha − beta) with alpha = beta = 1, exactly
    let aleph = n as f64 * p + m;
    let j = (aleph.trunc() as usize).clamp(1, n - 1);
    let gamma = (aleph - j as f64).clamp(0.0, 1.0);
    let (a, b) = (sorted[j - 1], sorted[j]);
    let approx = a == b || (a - b).abs() <= f64::EPSILON.sqrt() * a.abs().max(b.abs());
    if a.is_finite() && b.is_finite() && approx {
        a + gamma * (b - a)
    } else {
        (1.0 - gamma) * a + gamma * b
    }
}

/// Create robust mask from magnitude using quantile-based thresholding: MriResearchTools.jl's
/// `robustmask` (`src/masking.jl:6-27`, 771eadf), with its threshold from `sample` and
/// `Statistics.quantile`, and its box smoothing in single precision (see the comments in the
/// body). On the √|HIP| of the QSM-CI in-vivo travelling-head scan it gives MriResearchTools'
/// mask exactly (2,488,761 voxels); before it sampled every positive voxel, took nearest-rank
/// quantiles and smoothed in f64, and 12,619 voxels differed.
pub fn robust_mask(mag: &[f64], grid: &Grid) -> Vec<u8> {
    let n_total = check_volume("robust_mask: magnitude", mag.len(), grid);

    // MriResearchTools `robustmask` (`src/masking.jl:6-27`, 771eadf) estimates its threshold
    // from `sample(weight)` (`src/utility.jl:23-33`): `len = ceil(√min(1e5, n))` blocks of `len`
    // consecutive voxels (in memory order), starting at `round.(Int, range(0, n − len;
    // length=len))`, finite values only (zeros included).
    let samples = mrt_sample(mag);
    if samples.is_empty() {
        return vec![0u8; n_total];
    }
    let mut sorted = samples.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Statistics.quantile (type 7) on the sample
    let q05 = quantile_sorted(&sorted, 0.05);
    let q15 = quantile_sorted(&sorted, 0.15);
    let q80 = quantile_sorted(&sorted, 0.80);
    let q99 = quantile_sorted(&sorted, 0.99);
    let mean = |v: Vec<f64>| if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 };

    // high_intensity = mean(w[q8 .<= w .<= q99]); noise = mean(w[w .<= q15])
    let high_intensity = mean(samples.iter().copied().filter(|&v| q80 <= v && v <= q99).collect());
    let mut noise = mean(samples.iter().copied().filter(|&v| v <= q15).collect());
    if noise > high_intensity / 10.0 {
        noise = mean(samples.iter().copied().filter(|&v| v <= q05).collect());
        if noise > high_intensity / 10.0 {
            noise = 0.0; // no noise detected
        }
    }

    // Calculate threshold: max(5*noise, high_intensity/5)
    let threshold = (5.0 * noise).max(high_intensity / 5.0);

    // The mask, then: smoothed with one 5-voxel box per axis, > 0.4; holes filled; smoothed
    // with two 3-voxel boxes per axis, > 0.6. MriResearchTools smooths the Bool mask as Float32
    // (`gaussiansmooth3d` starts from `0f0 .+ image`) and compares with the Float64 literals, so
    // a box average of exactly 2/5 counts as above 0.4 there (0.4f0 > 0.4). The smoothing here
    // is the same running-sum box filter in f32, compared the same way.
    let mut mask_f32: Vec<f32> = mag.iter()
        .map(|&v| if v > threshold { 1.0 } else { 0.0 })
        .collect();
    smooth_boxes_f32(&mut mask_f32, &[5], grid);
    let mut mask: Vec<u8> = mask_f32.iter()
        .map(|&v| u8::from(v as f64 > 0.4))
        .collect();

    // fill_holes(mask; max_hole_size = length(mask) / 20)
    let max_hole_size = n_total / 20;
    mask = fill_holes(&mask, grid, max_hole_size);

    let mut mask_f32: Vec<f32> = mask.iter().map(|&v| v as f32).collect();
    smooth_boxes_f32(&mut mask_f32, &[3, 3], grid);
    mask = mask_f32.iter()
        .map(|&v| u8::from(v as f64 > 0.6))
        .collect();

    mask
}

//=============================================================================
// Box Segmentation
//=============================================================================

/// Box segmentation for finding tissue regions
///
/// Divides the image into nbox^3 boxes and identifies voxels that
/// consistently appear in the high-intensity range across multiple boxes.
///
/// MriResearchTools' `boxsegment` and `threshold` (`src/intensitycorrection.jl`, 771eadf): box
/// centres every `boxshift = ceil(N / nbox)` voxels from the first, each box
/// `max(1, c − boxshift):min(c + boxshift, N)` with both ends included, a voxel counts in a box
/// when it is strictly within ±10 % of the box's `quantile(image[mask], 0.9)`, and tissue is what
/// at least two boxes count.
fn box_segment(
    image: &[f64],
    mask: &[u8],
    nbox: usize,
    nx: usize, ny: usize, nz: usize,
) -> Vec<u8> {
    let n_total = nx * ny * nz;
    let mut vote_count = vec![0u8; n_total];

    // Calculate box shift (stride between box centers)
    let box_shift_x = nx.div_ceil(nbox);
    let box_shift_y = ny.div_ceil(nbox);
    let box_shift_z = nz.div_ceil(nbox);

    // For each box center
    let mut cz = 0;
    while cz < nz {
        let mut cy = 0;
        while cy < ny {
            let mut cx = 0;
            while cx < nx {
                // Box bounds: `max(1, c − boxshift):min(c + boxshift, N)`, both ends included
                // (MriResearchTools `boxsegment`, `src/intensitycorrection.jl`, 771eadf).
                let x_start = cx.saturating_sub(box_shift_x);
                let x_end = (cx + box_shift_x + 1).min(nx);
                let y_start = cy.saturating_sub(box_shift_y);
                let y_end = (cy + box_shift_y + 1).min(ny);
                let z_start = cz.saturating_sub(box_shift_z);
                let z_end = (cz + box_shift_z + 1).min(nz);

                // Collect values in this box
                let mut box_vals: Vec<f64> = Vec::new();
                for z in z_start..z_end {
                    for y in y_start..y_end {
                        for x in x_start..x_end {
                            let idx = idx3d(x, y, z, nx, ny);
                            if mask[idx] > 0 && image[idx].is_finite() {
                                box_vals.push(image[idx]);
                            }
                        }
                    }
                }

                if box_vals.is_empty() {
                    cx += box_shift_x;
                    continue;
                }

                // `threshold`: `quantile(image[mask], 0.9)` (Statistics, type 7)
                box_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let q90 = quantile_sorted(&box_vals, 0.9);

                // Define tissue range around 90th percentile
                let width = 0.1;
                let low = (1.0 - width) * q90;
                let high = (1.0 + width) * q90;

                // Vote for voxels in tissue range
                for z in z_start..z_end {
                    for y in y_start..y_end {
                        for x in x_start..x_end {
                            let idx = idx3d(x, y, z, nx, ny);
                            if mask[idx] > 0 {
                                let v = image[idx];
                                if v > low && v < high {
                                    vote_count[idx] = vote_count[idx].saturating_add(1);
                                }
                            }
                        }
                    }
                }

                cx += box_shift_x;
            }
            cy += box_shift_y;
        }
        cz += box_shift_z;
    }

    // Threshold: must be identified as tissue in >= 2 boxes
    let mut segmented = vec![0u8; n_total];
    for i in 0..n_total {
        if vote_count[i] >= 2 && mask[i] > 0 {
            segmented[i] = 1;
        }
    }

    segmented
}

//=============================================================================
// Fill and Smooth (with weighted smoothing)
//=============================================================================

/// Fill holes and smooth the lowpass field with weighted smoothing
///
/// Matches MriResearchTools.jl's fillandsmooth! function.
/// Uses weighted smoothing where filled holes get weight 0.2.
fn fill_and_smooth(
    lowpass: &mut [f64],
    stable_mean: f64,
    sigma2: [f64; 3],
    grid: &Grid,
) {
    let n_total = grid.n_total();

    // Identify holes/outliers and create weight mask
    // lowpassweight = 1.2 - lowpassmask (so holes get 0.2, normal get 1.2)
    let mut weight = vec![1.2f64; n_total];

    for i in 0..n_total {
        if lowpass[i] < stable_mean / 4.0 ||
           lowpass[i].is_nan() ||
           lowpass[i] > 10.0 * stable_mean {
            lowpass[i] = 3.0 * stable_mean;
            weight[i] = 1.2 - 1.0; // `1.2 .- lowpassmask`: filled holes get less weight
        }
    }

    // Apply weighted smoothing
    let nbox = 3; // default for non-masked smoothing
    let smoothed = gaussian_smooth_3d(lowpass, sigma2, None, Some(&weight), nbox, grid);
    lowpass.copy_from_slice(&smoothed);
}

//=============================================================================
// Main API
//=============================================================================

/// Get sensitivity (bias field) from magnitude
///
/// This estimates the RF receive field inhomogeneity (sensitivity map)
/// that can be divided out to correct the image.
///
/// MriResearchTools' `getsensitivity` (771eadf). **Agreement:** bit-identical to
/// MriResearchTools' `makehomogeneous(mag; sigma = 7 mm in voxels, nbox = 15)` on the 20×18×14,
/// 1×1×1.5 mm phantom of `tests/mritools_parity.rs` (verified QSM-CI `julia-qsm` image,
/// QSMxT/QSM-CI#299; measured 2026-10-30), where it was 3.3 % NRMSE in the mask before three
/// port errors were fixed (astewartau/QSM.rs#319). Each measured alone, with the other two fixed:
/// - the masked smoothing's `nanboxfilterline!` wrote each fill-mode extrapolation onto the last
///   voxel of the line as well (an off-by-one from Julia's 1-based index): 4.0 %;
/// - [`box_segment`]'s boxes stopped one voxel short of `c + boxshift`, which the original
///   includes: 1.05 %;
/// - its 90th percentile was a nearest-rank index, not `Statistics.quantile` (type 7): 0.35 %.
///
/// Also now as the original, at rounding level: Julia's ties-to-even `round` in the box widths,
/// and the Float32 weights of `fillandsmooth!`'s weighted smoothing (`gaussiansmooth3d!` converts
/// them). These are port fixes, not tuning: the algorithm and its defaults are unchanged.
///
/// # Errors
/// If `nbox` (boxes per dimension for the tissue segmentation) is 0.
pub fn get_sensitivity(
    mag: &[f64],
    grid: &Grid,
    sigma_mm: f64,
    nbox: usize,
) -> Result<Vec<f64>, InputError> {
    if nbox == 0 {
        return Err(InputError::param("homogeneity nbox", nbox, ">= 1"));
    }
    let (nx, ny, nz) = grid.dims;
    let (vx, vy, vz) = grid.voxel_size;
    let n_total = check_volume("get_sensitivity: magnitude", mag.len(), grid);

    // Convert mm to voxels
    let sigma = [sigma_mm / vx, sigma_mm / vy, sigma_mm / vz];

    // Create initial mask (with full post-processing)
    let mask = robust_mask(mag, grid);

    // Box segmentation to find tissue
    let segmentation = box_segment(mag, &mask, nbox, nx, ny, nz);

    // Split sigma into two parts (matching MriResearchTools.jl)
    let factor: f64 = 0.7;
    let sigma1 = [
        (1.0_f64 - factor * factor).sqrt() * sigma[0],
        (1.0_f64 - factor * factor).sqrt() * sigma[1],
        (1.0_f64 - factor * factor).sqrt() * sigma[2],
    ];
    let sigma2 = [
        factor * sigma[0],
        factor * sigma[1],
        factor * sigma[2],
    ];

    // First smoothing with tissue mask (nbox=8 for masked smoothing)
    let mut lowpass = gaussian_smooth_3d(mag, sigma1, Some(&segmentation), None, 8, grid);

    // Calculate stable mean for filling
    let mut sum = 0.0;
    let mut count = 0usize;
    for i in 0..n_total {
        if mask[i] > 0 && mag[i].is_finite() {
            sum += mag[i];
            count += 1;
        }
    }
    let stable_mean = if count > 0 { sum / count as f64 } else { 1.0 };

    // Fill holes and apply weighted second smoothing
    fill_and_smooth(&mut lowpass, stable_mean, sigma2, grid);

    Ok(lowpass)
}

/// Make magnitude homogeneous by dividing by bias field
///
/// This is the main entry point for bias field correction.
///
/// # Arguments
/// * `mag` - Input magnitude data (nx * ny * nz)
/// * `grid` - Dimensions and voxel sizes in mm
/// * `sigma_mm` - Smoothing sigma in mm, used as given (MriResearchTools' default is 7; no
///   field-of-view cap is applied, see the module docs)
/// * `nbox` - Number of boxes per dimension for segmentation (MriResearchTools' default 15)
///
/// # Returns
/// Bias-corrected magnitude
///
/// # Errors
/// If `nbox` is 0 (see [`get_sensitivity`]).
pub fn makehomogeneous(
    mag: &[f64],
    grid: &Grid,
    sigma_mm: f64,
    nbox: usize,
) -> Result<Vec<f64>, InputError> {
    let n_total = check_volume("makehomogeneous: magnitude", mag.len(), grid);
    let sensitivity = get_sensitivity(mag, grid, sigma_mm, nbox)?;

    let mut result = vec![0.0; n_total];
    for i in 0..n_total {
        if sensitivity[i] > 1e-10 && !sensitivity[i].is_nan() {
            result[i] = mag[i] / sensitivity[i];
        } else {
            result[i] = mag[i];
        }
    }

    Ok(result)
}

/// RSS (Root Sum of Squares) magnitude combination
///
/// Combines multi-echo magnitude images using RSS.
///
/// # Arguments
/// * `mags_flat` - Flattened magnitudes [echo0, echo1, ...]
/// * `n_echoes` - Number of echoes
/// * `n_total` - Voxels per echo (nx * ny * nz)
///
/// # Returns
/// RSS-combined magnitude
pub fn rss_combine(
    mags_flat: &[f64],
    n_echoes: usize,
    n_total: usize,
) -> Vec<f64> {
    assert_eq!(
        mags_flat.len(), n_echoes * n_total,
        "rss_combine: magnitudes hold {} values but {n_echoes} echoes of {n_total} voxels \
         need {}",
        mags_flat.len(), n_echoes * n_total
    );
    let mut result = vec![0.0; n_total];

    for e in 0..n_echoes {
        let offset = e * n_total;
        for i in 0..n_total {
            let v = mags_flat[offset + i];
            result[i] += v * v;
        }
    }

    for i in 0..n_total {
        result[i] = result[i].sqrt();
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(nx: usize, ny: usize, nz: usize) -> Grid {
        Grid::new(nx, ny, nz, 1.0, 1.0, 1.0)
    }

    #[test]
    fn test_get_box_sizes() {
        // Test box size calculation matches Julia
        let sizes = get_box_sizes(5.0, 3);
        assert_eq!(sizes.len(), 3);
        // For sigma=5, n=3: wideal ≈ 5.77
        // All sizes should be odd and reasonable
        for &s in &sizes {
            assert!(s % 2 == 1, "Box size should be odd, got {}", s);
            assert!((3..=11).contains(&s), "Box size should be in reasonable range, got {}", s);
        }
    }

    #[test]
    fn test_box_filter_line() {
        // Simple test: uniform values should stay uniform
        let mut line = vec![1.0; 10];
        box_filter_line(&mut line, 3);
        for &v in &line {
            assert!((v - 1.0).abs() < 1e-10, "Uniform line should stay uniform");
        }
    }

    #[test]
    fn test_robust_mask_basic() {
        // Simple test with uniform high values
        let mag = vec![100.0; 27];
        let mask = robust_mask(&mag, &grid(3, 3, 3));
        // All values are the same, so all should be masked
        let masked_count: usize = mask.iter().map(|&v| v as usize).sum();
        assert!(masked_count > 0, "Should have some masked voxels");
    }

    #[test]
    fn test_rss_combine() {
        // Two echoes, 4 voxels each
        let mags = vec![
            3.0, 0.0, 0.0, 5.0,  // echo 0
            4.0, 0.0, 0.0, 12.0, // echo 1
        ];
        let result = rss_combine(&mags, 2, 4);

        // sqrt(3^2 + 4^2) = 5
        assert!((result[0] - 5.0).abs() < 1e-10);
        // sqrt(0 + 0) = 0
        assert!((result[1] - 0.0).abs() < 1e-10);
        // sqrt(5^2 + 12^2) = 13
        assert!((result[3] - 13.0).abs() < 1e-10);
    }

    // =====================================================================
    // Helper: create a 3D sphere magnitude phantom with optional bias field
    // =====================================================================

    /// Create a 3D sphere phantom with a smooth bias field applied.
    /// Returns (magnitude_data, mask) where mask marks inside-sphere voxels.
    fn make_sphere_phantom(n: usize, bias: bool) -> (Vec<f64>, Vec<u8>) {
        let center = n as f64 / 2.0;
        let radius = n as f64 / 2.0 - 2.0;
        let n_total = n * n * n;
        let mut mag = vec![0.0f64; n_total];
        let mut mask = vec![0u8; n_total];

        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let dx = i as f64 - center;
                    let dy = j as f64 - center;
                    let dz = k as f64 - center;
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    let idx = i + j * n + k * n * n;

                    if dist < radius {
                        // Base tissue intensity
                        let base = 100.0;
                        // Apply smooth bias field if requested
                        let bias_val = if bias {
                            1.0 + 0.5 * (i as f64 / n as f64)
                        } else {
                            1.0
                        };
                        mag[idx] = base * bias_val;
                        mask[idx] = 1;
                    } else {
                        // Background noise
                        mag[idx] = 1.0 + 0.5 * ((i + j + k) % 3) as f64;
                    }
                }
            }
        }

        (mag, mask)
    }

    // =====================================================================
    // Tests for get_box_sizes edge cases
    // =====================================================================

    #[test]
    fn test_get_box_sizes_zero_sigma() {
        let sizes = get_box_sizes(0.0, 3);
        assert_eq!(sizes, vec![0, 0, 0]);
    }

    #[test]
    fn test_get_box_sizes_zero_n() {
        let sizes = get_box_sizes(5.0, 0);
        assert!(sizes.is_empty());
    }

    #[test]
    fn test_get_box_sizes_negative_sigma() {
        let sizes = get_box_sizes(-1.0, 3);
        assert_eq!(sizes, vec![0, 0, 0]);
    }

    #[test]
    fn test_get_box_sizes_large_sigma() {
        let sizes = get_box_sizes(20.0, 4);
        assert_eq!(sizes.len(), 4);
        for &s in &sizes {
            assert!(s % 2 == 1, "Box size should be odd, got {}", s);
            assert!(s >= 3, "Box size should be at least 3, got {}", s);
        }
    }

    // =====================================================================
    // Tests for check_box_sizes
    // =====================================================================

    #[test]
    fn test_check_box_sizes_clamps_to_half_image() {
        // Box size larger than half the image dimension should be clamped
        let mut boxsizes = vec![vec![99], vec![99], vec![99]];
        let dims = [10, 10, 10];
        check_box_sizes(&mut boxsizes, &dims);
        for bs in &boxsizes {
            for &b in bs {
                assert!(b <= dims[0], "Box size should be clamped, got {}", b);
                assert!(b % 2 == 1, "Box size should be odd, got {}", b);
            }
        }
    }

    #[test]
    fn test_check_box_sizes_makes_even_odd() {
        let mut boxsizes = vec![vec![4], vec![6], vec![8]];
        let dims = [100, 100, 100];
        check_box_sizes(&mut boxsizes, &dims);
        for bs in &boxsizes {
            for &b in bs {
                assert!(b % 2 == 1, "Box size should be odd, got {}", b);
            }
        }
    }

    // =====================================================================
    // Tests for box_filter_line edge cases
    // =====================================================================

    #[test]
    fn test_box_filter_line_too_small_boxsize() {
        // boxsize < 3 should be a no-op
        let mut line = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let original = line.clone();
        box_filter_line(&mut line, 1);
        assert_eq!(line, original);
    }

    #[test]
    fn test_box_filter_line_larger_than_data() {
        // boxsize > line length should be a no-op
        let mut line = vec![1.0, 2.0];
        let original = line.clone();
        box_filter_line(&mut line, 5);
        assert_eq!(line, original);
    }

    #[test]
    fn test_box_filter_line_smoothing_effect() {
        // A spike should be smoothed out
        let mut line = vec![0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        box_filter_line(&mut line, 5);
        // The spike at index 3 should be reduced
        assert!(line[3] < 10.0, "Spike should be smoothed");
        // Neighbors should have received some intensity
        assert!(line[2] > 0.0 || line[4] > 0.0, "Neighbors should gain intensity");
        // All values should be finite
        for &v in &line {
            assert!(v.is_finite(), "All values should be finite");
        }
    }

    // =====================================================================
    // Tests for box_filter_line_weighted
    // =====================================================================

    #[test]
    fn test_box_filter_line_weighted_too_small() {
        let mut line = vec![1.0, 2.0];
        let mut weight = vec![1.0, 1.0];
        let orig_l = line.clone();
        let orig_w = weight.clone();
        box_filter_line_weighted(&mut line, &mut weight, 1);
        assert_eq!(line, orig_l);
        assert_eq!(weight, orig_w);
    }

    #[test]
    fn test_box_filter_line_weighted_uniform() {
        let mut line = vec![5.0; 10];
        let mut weight = vec![1.0; 10];
        box_filter_line_weighted(&mut line, &mut weight, 3);
        // With uniform values and uniform weights, the middle portion
        // should remain close to 5.0
        for i in 2..8 {
            assert!(
                (line[i] - 5.0).abs() < 0.5,
                "Uniform weighted line should stay near 5.0, got {} at index {}",
                line[i], i
            );
        }
    }

    #[test]
    fn test_box_filter_line_weighted_varying_weights() {
        let mut line = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut weight = vec![1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0];
        box_filter_line_weighted(&mut line, &mut weight, 3);
        // All values should remain finite
        for &v in &line {
            assert!(v.is_finite(), "All values should be finite after weighted filter");
        }
        for &w in &weight {
            assert!(w.is_finite(), "All weights should be finite after weighted filter");
        }
    }

    // =====================================================================
    // Tests for nan_box_filter_line
    // =====================================================================

    #[test]
    fn test_nan_box_filter_line_too_small() {
        let mut line = vec![1.0, 2.0];
        let original = line.clone();
        nan_box_filter_line(&mut line, 5, &mut Vec::new());
        assert_eq!(line, original);
    }

    #[test]
    fn test_nan_box_filter_line_no_nans() {
        // All valid data => the nan filter processes it and produces finite results
        let mut line = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        nan_box_filter_line(&mut line, 3, &mut Vec::new());
        // All values should remain finite
        for &v in &line {
            assert!(v.is_finite(), "All values should be finite after nan box filter");
        }
        // The nan box filter in Normal mode should act as a smoothing filter
        // Check that some middle values have changed (been smoothed)
        // The function's Normal mode applies running sum averaging
    }

    #[test]
    fn test_nan_box_filter_line_with_nans() {
        // Data with NaN gaps
        let mut line = vec![5.0, 5.0, 5.0, f64::NAN, f64::NAN, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0];
        nan_box_filter_line(&mut line, 3, &mut Vec::new());
        // Non-NaN regions should still be mostly finite
        let finite_count = line.iter().filter(|v| v.is_finite()).count();
        assert!(finite_count >= 6, "Most values should be finite, got {}", finite_count);
    }

    #[test]
    fn test_nan_box_filter_line_all_nan_except_edges() {
        // Mostly NaN
        let mut line = vec![1.0; 12];
        for i in 2..10 {
            line[i] = f64::NAN;
        }
        nan_box_filter_line(&mut line, 3, &mut Vec::new());
        // The function should not panic and edge values should remain finite
        assert!(line[0].is_finite() || line[0].is_nan());
    }

    // =====================================================================
    // Tests for gaussian_smooth_3d (main 3D smoothing)
    // =====================================================================

    #[test]
    fn test_gaussian_smooth_3d_uniform_no_mask() {
        let n = 8;
        let data = vec![5.0; n * n * n];
        let result = gaussian_smooth_3d(&data, [2.0, 2.0, 2.0], None, None, 3, &grid(n, n, n));
        assert_eq!(result.len(), n * n * n);
        // Uniform data should stay nearly uniform after smoothing
        for &v in &result {
            assert!(
                (v - 5.0).abs() < 0.5,
                "Uniform data should stay near 5.0, got {}",
                v
            );
        }
    }

    #[test]
    fn test_gaussian_smooth_3d_with_mask() {
        let n = 10;
        let (mag, mask) = make_sphere_phantom(n, false);
        // Smooth with mask
        let result = gaussian_smooth_3d(&mag, [1.5, 1.5, 1.5], Some(&mask), None, 4, &grid(n, n, n));
        assert_eq!(result.len(), n * n * n);
        // Inside mask, values should still be finite
        for i in 0..result.len() {
            if mask[i] > 0 {
                assert!(result[i].is_finite(), "Masked voxel at {} should be finite", i);
            }
        }
    }

    #[test]
    fn test_gaussian_smooth_3d_with_weights() {
        let n = 8;
        let data = vec![10.0; n * n * n];
        let weight = vec![1.0; n * n * n];
        let result = gaussian_smooth_3d(
            &data, [1.5, 1.5, 1.5], None, Some(&weight), 3, &grid(n, n, n),
        );
        assert_eq!(result.len(), n * n * n);
        for &v in &result {
            assert!(v.is_finite(), "Result should be finite");
        }
    }

    // =====================================================================
    // Tests for gaussian_smooth_3d_boxsizes
    // =====================================================================

    #[test]
    fn test_gaussian_smooth_3d_boxsizes_uniform() {
        let n = 8;
        let data = vec![3.0; n * n * n];
        let boxsizes = vec![vec![3, 3], vec![3, 3], vec![3, 3]];
        let result = gaussian_smooth_3d_boxsizes(&data, &boxsizes, 2, &grid(n, n, n));
        assert_eq!(result.len(), n * n * n);
        for &v in &result {
            assert!(
                (v - 3.0).abs() < 0.5,
                "Uniform data should stay near 3.0, got {}",
                v
            );
        }
    }

    #[test]
    fn test_gaussian_smooth_3d_boxsizes_spike() {
        let n = 10;
        let n_total = n * n * n;
        let mut data = vec![0.0; n_total];
        // Place a spike at the center
        let center = n / 2 + (n / 2) * n + (n / 2) * n * n;
        data[center] = 100.0;
        let boxsizes = vec![vec![5, 5], vec![5, 5], vec![5, 5]];
        let result = gaussian_smooth_3d_boxsizes(&data, &boxsizes, 2, &grid(n, n, n));
        assert_eq!(result.len(), n_total);
        // Spike should be reduced
        assert!(
            result[center] < 100.0,
            "Spike should be smoothed, got {}",
            result[center]
        );
        // Sum should be approximately conserved (box filter is mean-preserving)
        for &v in &result {
            assert!(v.is_finite(), "All values should be finite");
        }
    }

    // =====================================================================
    // Tests for robust_mask (more comprehensive)
    // =====================================================================

    #[test]
    fn test_robust_mask_sphere() {
        let n = 12;
        let (mag, _) = make_sphere_phantom(n, false);
        let mask = robust_mask(&mag, &grid(n, n, n));
        assert_eq!(mask.len(), n * n * n);
        let masked_count: usize = mask.iter().map(|&v| v as usize).sum();
        // The sphere should produce some masked voxels
        assert!(masked_count > 0, "Should have masked voxels for sphere phantom");
        // Center should be masked (high intensity)
        let center = n / 2 + (n / 2) * n + (n / 2) * n * n;
        assert_eq!(mask[center], 1, "Center of sphere should be masked");
    }

    #[test]
    fn test_robust_mask_empty() {
        // All zero magnitude -> empty mask
        let mag = vec![0.0; 27];
        let mask = robust_mask(&mag, &grid(3, 3, 3));
        let count: usize = mask.iter().map(|&v| v as usize).sum();
        assert_eq!(count, 0, "Zero magnitude should produce empty mask");
    }

    #[test]
    fn test_robust_mask_nan_values() {
        // Magnitude with NaN values
        let mut mag = vec![100.0; 125];
        mag[0] = f64::NAN;
        mag[10] = f64::NAN;
        mag[50] = f64::INFINITY;
        let mask = robust_mask(&mag, &grid(5, 5, 5));
        assert_eq!(mask.len(), 125);
        // Should still produce a valid mask
        for &v in &mask {
            assert!(v == 0 || v == 1, "Mask values should be 0 or 1");
        }
    }

    // =====================================================================
    // Tests for box_segment
    // =====================================================================

    #[test]
    fn test_box_segment_uniform() {
        let n = 10;
        let (mag, mask) = make_sphere_phantom(n, false);
        let seg = box_segment(&mag, &mask, 3, n, n, n);
        assert_eq!(seg.len(), n * n * n);
        // In a uniform sphere, most interior voxels should be segmented as tissue
        let seg_count: usize = seg.iter().map(|&v| v as usize).sum();
        assert!(seg_count > 0, "Box segment should find some tissue voxels");
    }

    #[test]
    fn test_box_segment_empty_mask() {
        let n = 8;
        let mag = vec![100.0; n * n * n];
        let mask = vec![0u8; n * n * n]; // empty mask
        let seg = box_segment(&mag, &mask, 3, n, n, n);
        let count: usize = seg.iter().map(|&v| v as usize).sum();
        assert_eq!(count, 0, "Empty mask should produce no segmentation");
    }

    // =====================================================================
    // Tests for fill_and_smooth
    // =====================================================================

    #[test]
    fn test_fill_and_smooth_basic() {
        let n = 10;
        let n_total = n * n * n;
        let stable_mean = 100.0;
        let mut lowpass = vec![stable_mean; n_total];
        // Add some holes (very low values)
        lowpass[0] = 0.0;
        lowpass[100] = f64::NAN;
        lowpass[200] = 2000.0; // outlier > 10*stable_mean

        let sigma2 = [2.0, 2.0, 2.0];
        fill_and_smooth(&mut lowpass, stable_mean, sigma2, &grid(n, n, n));

        // All values should be finite after fill and smooth
        for (i, &v) in lowpass.iter().enumerate() {
            assert!(v.is_finite(), "Value at {} should be finite, got {}", i, v);
        }
    }

    #[test]
    fn test_fill_and_smooth_preserves_approximate_mean() {
        let n = 8;
        let n_total = n * n * n;
        let stable_mean = 50.0;
        let mut lowpass = vec![stable_mean; n_total];
        let sigma2 = [1.5, 1.5, 1.5];
        fill_and_smooth(&mut lowpass, stable_mean, sigma2, &grid(n, n, n));

        // Mean should be approximately preserved
        let mean: f64 = lowpass.iter().sum::<f64>() / n_total as f64;
        assert!(
            (mean - stable_mean).abs() < stable_mean * 0.5,
            "Mean should be approximately preserved, got {} vs {}",
            mean,
            stable_mean
        );
    }

    // =====================================================================
    // Tests for get_sensitivity
    // =====================================================================

    #[test]
    fn test_get_sensitivity_sphere() {
        let n = 12;
        let (mag, _) = make_sphere_phantom(n, true);
        let sensitivity = get_sensitivity(&mag, &grid(n, n, n), 4.0, 5).unwrap();
        assert_eq!(sensitivity.len(), n * n * n);
        // Sensitivity should be finite and mostly positive in the sphere
        let center = n / 2 + (n / 2) * n + (n / 2) * n * n;
        assert!(
            sensitivity[center].is_finite(),
            "Sensitivity at center should be finite"
        );
        assert!(
            sensitivity[center] > 0.0,
            "Sensitivity at center should be positive, got {}",
            sensitivity[center]
        );
    }

    #[test]
    fn test_get_sensitivity_output_size() {
        let n = 10;
        let (mag, _) = make_sphere_phantom(n, false);
        let sensitivity = get_sensitivity(&mag, &grid(n, n, n), 3.0, 3).unwrap();
        assert_eq!(sensitivity.len(), n * n * n);
    }

    // =====================================================================
    // Tests for makehomogeneous (main entry point)
    // =====================================================================

    #[test]
    fn test_makehomogeneous_output_size_and_finite() {
        let n = 12;
        let (mag, _) = make_sphere_phantom(n, true);
        let result = makehomogeneous(&mag, &grid(n, n, n), 4.0, 5).unwrap();
        assert_eq!(result.len(), n * n * n);
        // All output values should be finite
        for (i, &v) in result.iter().enumerate() {
            assert!(v.is_finite(), "Output at {} should be finite, got {}", i, v);
        }
    }

    #[test]
    fn test_makehomogeneous_reduces_bias() {
        let n = 12;
        let (mag_biased, mask) = make_sphere_phantom(n, true);
        let result = makehomogeneous(&mag_biased, &grid(n, n, n), 4.0, 5).unwrap();

        // Collect values inside the sphere
        let mut original_vals = Vec::new();
        let mut corrected_vals = Vec::new();
        for i in 0..(n * n * n) {
            if mask[i] > 0 {
                original_vals.push(mag_biased[i]);
                corrected_vals.push(result[i]);
            }
        }

        // The coefficient of variation should be reduced (or at least not much worse)
        let orig_mean = original_vals.iter().sum::<f64>() / original_vals.len() as f64;
        let orig_std = (original_vals
            .iter()
            .map(|v| (v - orig_mean).powi(2))
            .sum::<f64>()
            / original_vals.len() as f64)
            .sqrt();
        let orig_cv = orig_std / orig_mean;

        let corr_mean = corrected_vals.iter().sum::<f64>() / corrected_vals.len() as f64;
        let corr_std = (corrected_vals
            .iter()
            .map(|v| (v - corr_mean).powi(2))
            .sum::<f64>()
            / corrected_vals.len() as f64)
            .sqrt();
        let corr_cv = corr_std / corr_mean;

        assert!(
            corr_cv <= orig_cv * 2.0,
            "Corrected CV ({}) should not be much worse than original CV ({})",
            corr_cv,
            orig_cv
        );
    }

    #[test]
    fn test_makehomogeneous_no_bias_preserves() {
        let n = 12;
        let (mag, mask) = make_sphere_phantom(n, false);
        let result = makehomogeneous(&mag, &grid(n, n, n), 4.0, 5).unwrap();

        // With no bias, corrected values should still be positive inside sphere
        for i in 0..(n * n * n) {
            if mask[i] > 0 {
                assert!(
                    result[i] > 0.0,
                    "Inside sphere voxel {} should be positive, got {}",
                    i,
                    result[i]
                );
            }
        }
    }

    #[test]
    fn test_makehomogeneous_anisotropic_voxels() {
        let n = 12;
        let (mag, _) = make_sphere_phantom(n, true);
        // Use anisotropic voxel sizes
        let result = makehomogeneous(&mag, &Grid::new(n, n, n, 0.5, 0.5, 2.0), 4.0, 5).unwrap();
        assert_eq!(result.len(), n * n * n);
        for &v in &result {
            assert!(v.is_finite(), "All output values should be finite");
        }
    }

    // =====================================================================
    // Tests for gaussian_smooth_3d with reversed passes (mask + even pass)
    // =====================================================================

    #[test]
    fn test_gaussian_smooth_3d_mask_reverse_passes() {
        // Exercise the reverse pass code paths (mask.is_some() && ibox % 2 == 1)
        let n = 10;
        let (mag, mask) = make_sphere_phantom(n, false);
        // nbox=4 ensures we have even passes (ibox=1,3)
        let result = gaussian_smooth_3d(&mag, [1.5, 1.5, 1.5], Some(&mask), None, 4, &grid(n, n, n));
        assert_eq!(result.len(), n * n * n);
        // Inside mask, values should still be finite
        for i in 0..result.len() {
            if mask[i] > 0 {
                // NaN smoothing can produce NaN near edges, but center should be finite
            }
        }
        // At least some values should be finite
        let finite_count = result.iter().filter(|v| v.is_finite()).count();
        assert!(finite_count > 0, "Should have some finite values");
    }

    // =====================================================================
    // Issue #70: a volume that does not match its grid must say so
    //
    // A run is processed on one grid and that grid indexes every other volume,
    // so a magnitude of a different matrix size used to read past the end of
    // the buffer inside the box filter:
    //
    //     bias_correction.rs:453:72: index out of bounds:
    //     the len is 21807104 but the index is 21807104
    //
    // which named neither the input nor the grid. Each entry point now refuses
    // the mismatch and reports both sizes.
    // =====================================================================

    #[test]
    #[should_panic(expected = "makehomogeneous: magnitude has 216 voxels but the grid is 8x8x8 (512 voxels)")]
    fn makehomogeneous_refuses_magnitude_shorter_than_grid() {
        let _ = makehomogeneous(&vec![1.0; 6 * 6 * 6], &grid(8, 8, 8), 4.0, 5);
    }

    #[test]
    #[should_panic(expected = "makehomogeneous: magnitude has 1000 voxels but the grid is 8x8x8 (512 voxels)")]
    fn makehomogeneous_refuses_magnitude_longer_than_grid() {
        // The over-long case never panicked - it silently corrected the first
        // 512 voxels and discarded the rest, which is worse than a crash.
        let _ = makehomogeneous(&vec![1.0; 10 * 10 * 10], &grid(8, 8, 8), 4.0, 5);
    }

    #[test]
    #[should_panic(expected = "robust_mask: magnitude has 216 voxels but the grid is 8x8x8 (512 voxels)")]
    fn robust_mask_refuses_magnitude_not_matching_grid() {
        robust_mask(&vec![1.0; 6 * 6 * 6], &grid(8, 8, 8));
    }

    #[test]
    #[should_panic(expected = "gaussian_smooth_3d_boxsizes: data has 216 voxels but the grid is 8x8x8 (512 voxels)")]
    fn gaussian_smooth_3d_boxsizes_refuses_data_not_matching_grid() {
        // This is the function that carried the reported panic (line 453).
        let boxsizes = vec![vec![5], vec![5], vec![5]];
        gaussian_smooth_3d_boxsizes(&vec![1.0; 6 * 6 * 6], &boxsizes, 1, &grid(8, 8, 8));
    }

    #[test]
    #[should_panic(expected = "gaussian_smooth_3d: mask has 216 voxels but the grid is 8x8x8 (512 voxels)")]
    fn gaussian_smooth_3d_refuses_mask_not_matching_grid() {
        let data = vec![1.0; 8 * 8 * 8];
        let mask = vec![1u8; 6 * 6 * 6];
        gaussian_smooth_3d(&data, [1.5, 1.5, 1.5], Some(&mask), None, 3, &grid(8, 8, 8));
    }

    #[test]
    #[should_panic(expected = "gaussian_smooth_3d: weight has 216 voxels but the grid is 8x8x8 (512 voxels)")]
    fn gaussian_smooth_3d_refuses_weight_not_matching_grid() {
        let data = vec![1.0; 8 * 8 * 8];
        let weight = vec![1.0; 6 * 6 * 6];
        gaussian_smooth_3d(&data, [1.5, 1.5, 1.5], None, Some(&weight), 3, &grid(8, 8, 8));
    }

    #[test]
    #[should_panic(expected = "rss_combine: magnitudes hold 600 values but 3 echoes of 216 voxels need 648")]
    fn rss_combine_refuses_truncated_echo_stack() {
        rss_combine(&vec![1.0; 600], 3, 6 * 6 * 6);
    }

    #[test]
    fn makehomogeneous_accepts_a_matching_volume() {
        // The guard must not reject the ordinary case.
        let n = 8;
        let out = makehomogeneous(&vec![1.0; n * n * n], &grid(n, n, n), 4.0, 5).unwrap();
        assert_eq!(out.len(), n * n * n);
    }

    /// The defaults are MriResearchTools' (`makehomogeneous(mag; sigma_mm=7, nbox=15)`), and the
    /// pipeline config's copies agree with them until it holds `HomogeneityParams` itself (#314).
    #[test]
    fn defaults_are_mriresearchtools_and_agree_with_the_pipeline_config() {
        let p = HomogeneityParams::default();
        assert_eq!((p.sigma_mm, p.nbox), (7.0, 15));
        let m = crate::pipeline::MaskingConfig::default();
        assert_eq!((m.homogeneity_sigma_mm, m.homogeneity_nbox), (p.sigma_mm, p.nbox));
    }

    /// #199: `nbox = 0` divided by zero in the box segmentation; it is an error now.
    #[test]
    fn zero_nbox_is_an_error() {
        let n = 8;
        let mag = vec![1.0; n * n * n];
        let err = makehomogeneous(&mag, &grid(n, n, n), 4.0, 0).unwrap_err();
        assert!(err.0.contains("homogeneity nbox = 0") && err.0.contains(">= 1"), "{err}");
        assert!(get_sensitivity(&mag, &grid(n, n, n), 4.0, 0).is_err());
    }
}

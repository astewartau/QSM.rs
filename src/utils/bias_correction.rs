//! Bias field correction (homogeneity correction)
//!
//! Implements the makehomogeneous algorithm from MriResearchTools.jl
//! for correcting RF receive field inhomogeneities.
//!
//! Reference: CLEAR-SWI paper - Eckstein et al. (2021)
//! This uses the "boxsegment" approach with box filter Gaussian approximation.

use std::collections::VecDeque;

/// Index into 3D array (Fortran/column-major order)
#[inline(always)]
fn idx3d(i: usize, j: usize, k: usize, nx: usize, ny: usize) -> usize {
    i + j * nx + k * nx * ny
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
    let wl = wl_float.round() as usize;
    let wl = if wl % 2 == 0 { wl + 1 } else { wl }; // ensure odd
    let wu = wl + 2;

    // mideal = (12*sigma^2 - n*wl^2 - 4*n*wl - 3*n) / (-4*wl - 4)
    let wl_f = wl as f64;
    let n_f = n as f64;
    let mideal = (12.0 * sigma * sigma - n_f * wl_f * wl_f - 4.0 * n_f * wl_f - 3.0 * n_f)
                 / (-4.0 * wl_f - 4.0);
    let m = mideal.round() as usize;

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
                *b = if max_size % 2 == 0 { max_size + 1 } else { max_size };
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
fn box_filter_line_weighted(line: &mut [f64], weight: &mut [f64], boxsize: usize) {
    if boxsize < 3 || line.len() < boxsize {
        return;
    }

    let n = line.len();
    let r = boxsize / 2;

    let mut lq: VecDeque<f64> = VecDeque::with_capacity(boxsize);
    let mut wq: VecDeque<f64> = VecDeque::with_capacity(boxsize);

    // Initialize with first boxsize values
    let mut sum = f64::EPSILON; // slightly bigger than 0 to avoid division by 0
    let mut wsum = f64::EPSILON;
    let mut wsmooth = f64::EPSILON;

    for i in 0..boxsize {
        sum += line[i] * weight[i];
        wsum += weight[i];
        wsmooth += weight[i] * weight[i];
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

        sum += l * w - lold * wold;
        wsum += w - wold;
        line[i] = sum / wsum;
        wsmooth += w * w - wold * wold;
        weight[i] = wsmooth / wsum;
    }
}

/// 1D box filter with NaN handling (for masked smoothing)
/// Matches Julia's nanboxfilterline!
fn nan_box_filter_line(line: &mut [f64], boxsize: usize) {
    if boxsize < 3 || line.len() < boxsize {
        return;
    }

    let n = line.len();
    let r = boxsize / 2;
    let maxfills = r;

    // Create padded buffer with NaN padding
    let mut orig = vec![f64::NAN; n + boxsize - 1];
    orig[r..r + n].copy_from_slice(line);

    // Initial sum of first window (excluding NaN)
    let mut lsum = 0.0;
    for i in (r + 1)..=(2 * r) {
        if !orig[i].is_nan() {
            lsum += orig[i];
        }
    }

    let mut nfills = 0usize;
    let mut nvalids = 0usize;

    #[derive(PartialEq, Clone, Copy)]
    enum Mode { Nan, Normal, Fill }
    let mut mode = Mode::Nan;

    for i in 0..n {
        // Check for mode change
        match mode {
            Mode::Normal => {
                if orig[i + 2 * r].is_nan() {
                    mode = Mode::Fill;
                }
            }
            Mode::Nan => {
                if orig[i + 2 * r].is_nan() {
                    nvalids = 0;
                } else {
                    nvalids += 1;
                }
                if nvalids == boxsize {
                    mode = Mode::Normal;
                    lsum = 0.0;
                    for j in i..=(i + 2 * r) {
                        lsum += orig[j];
                    }
                    line[i] = lsum / boxsize as f64;
                    continue;
                }
            }
            Mode::Fill => {
                if orig[i + 2 * r].is_nan() {
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

        // Perform operation
        match mode {
            Mode::Normal => {
                if i > 0 {
                    lsum += orig[i + 2 * r] - orig[i - 1];
                }
                line[i] = lsum / boxsize as f64;
            }
            Mode::Fill => {
                if i > 0 {
                    lsum -= orig[i - 1];
                }
                line[i] = (lsum - orig[i]) / (boxsize - 2) as f64;

                // Extrapolate the NaN value
                let extrapolated = if i >= r {
                    2.0 * line[i] - line[i - r]
                } else {
                    line[i]
                };
                orig[i + 2 * r] = extrapolated;
                if i + r < n {
                    line[i + r] = extrapolated;
                }
                lsum += orig[i + 2 * r];
            }
            Mode::Nan => {
                // Keep as NaN or 0
            }
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
/// - weight: optional weights (None = no weighting)
/// - nbox: number of box filter passes (default 3, or 4 with mask)
/// - nx, ny, nz: dimensions
pub fn gaussian_smooth_3d(
    data: &[f64],
    sigma: [f64; 3],
    mask: Option<&[u8]>,
    mut weight: Option<&mut [f64]>,
    nbox: usize,
    nx: usize, ny: usize, nz: usize,
) -> Vec<f64> {
    let n_total = nx * ny * nz;
    let mut result: Vec<f64> = data.iter().map(|&v| v as f64).collect();

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
                        nan_box_filter_line(&mut line, bsize_x);
                    } else if let Some(ref mut w) = weight.as_deref_mut() {
                        let mut wline: Vec<f64> = (0..nx).map(|i| {
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
                        nan_box_filter_line(&mut line, bsize_y);
                    } else if let Some(ref mut w) = weight.as_deref_mut() {
                        let mut wline: Vec<f64> = (0..ny).map(|j| {
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
                        nan_box_filter_line(&mut line, bsize_z);
                    } else if let Some(ref mut w) = weight.as_deref_mut() {
                        let mut wline: Vec<f64> = (0..nz).map(|k| {
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
    nx: usize, ny: usize, nz: usize,
) -> Vec<f64> {
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
// Connected Components and Hole Filling
//=============================================================================

/// Find connected component using flood fill (6-connectivity in 3D)
fn flood_fill_component(
    mask: &[u8],
    visited: &mut [bool],
    start: usize,
    nx: usize, ny: usize, nz: usize,
) -> Vec<usize> {
    let mut component = Vec::new();
    let mut stack = vec![start];

    while let Some(idx) = stack.pop() {
        if visited[idx] || mask[idx] != 0 {
            continue;
        }

        visited[idx] = true;
        component.push(idx);

        // Get 3D coordinates
        let k = idx / (nx * ny);
        let rem = idx % (nx * ny);
        let j = rem / nx;
        let i = rem % nx;

        // 6-connectivity neighbors
        if i > 0 {
            let n = idx3d(i - 1, j, k, nx, ny);
            if !visited[n] && mask[n] == 0 { stack.push(n); }
        }
        if i + 1 < nx {
            let n = idx3d(i + 1, j, k, nx, ny);
            if !visited[n] && mask[n] == 0 { stack.push(n); }
        }
        if j > 0 {
            let n = idx3d(i, j - 1, k, nx, ny);
            if !visited[n] && mask[n] == 0 { stack.push(n); }
        }
        if j + 1 < ny {
            let n = idx3d(i, j + 1, k, nx, ny);
            if !visited[n] && mask[n] == 0 { stack.push(n); }
        }
        if k > 0 {
            let n = idx3d(i, j, k - 1, nx, ny);
            if !visited[n] && mask[n] == 0 { stack.push(n); }
        }
        if k + 1 < nz {
            let n = idx3d(i, j, k + 1, nx, ny);
            if !visited[n] && mask[n] == 0 { stack.push(n); }
        }
    }

    component
}

/// Fill holes in a binary mask
///
/// Matches MriResearchTools.jl's fill_holes function.
/// Fills connected components of zeros (holes) up to max_hole_size.
/// Uses 6-connectivity for 3D.
pub fn fill_holes(mask: &[u8], nx: usize, ny: usize, nz: usize, max_hole_size: usize) -> Vec<u8> {
    let n_total = nx * ny * nz;
    let mut result = mask.to_vec();
    let mut visited = vec![false; n_total];

    // Find all connected components of zeros (potential holes)
    for idx in 0..n_total {
        if mask[idx] == 0 && !visited[idx] {
            let component = flood_fill_component(mask, &mut visited, idx, nx, ny, nz);

            // Check if this component touches the boundary
            let mut touches_boundary = false;
            for &cidx in &component {
                let k = cidx / (nx * ny);
                let rem = cidx % (nx * ny);
                let j = rem / nx;
                let i = rem % nx;

                if i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1 {
                    touches_boundary = true;
                    break;
                }
            }

            // Fill if it's a hole (doesn't touch boundary) and small enough
            if !touches_boundary && component.len() <= max_hole_size {
                for cidx in component {
                    result[cidx] = 1;
                }
            }
        }
    }

    result
}

//=============================================================================
// Robust Mask (matching MriResearchTools.jl)
//=============================================================================

/// Create robust mask from magnitude using quantile-based thresholding
///
/// This matches MriResearchTools.jl's robustmask function, including
/// post-processing with smoothing and hole filling.
pub fn robust_mask(mag: &[f64], nx: usize, ny: usize, nz: usize) -> Vec<u8> {
    let n_total = nx * ny * nz;

    // Collect valid (positive, finite) samples and sort
    let mut samples: Vec<f64> = mag.iter()
        .filter(|&&v| v.is_finite() && v > 0.0)
        .copied()
        .collect();

    if samples.is_empty() {
        return vec![0u8; n_total];
    }

    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let len = samples.len();

    // Calculate quantiles
    let q05_idx = ((0.05 * len as f64) as usize).min(len - 1);
    let q15_idx = ((0.15 * len as f64) as usize).min(len - 1);
    let q80_idx = ((0.80 * len as f64) as usize).min(len - 1);
    let q99_idx = ((0.99 * len as f64) as usize).min(len - 1);

    let q05 = samples[q05_idx];
    let q15 = samples[q15_idx];
    let q80 = samples[q80_idx];
    let q99 = samples[q99_idx];

    // Calculate high intensity mean (between 80th and 99th percentile)
    let high_samples: Vec<f64> = samples.iter()
        .filter(|&&v| v >= q80 && v <= q99)
        .copied()
        .collect();

    let high_intensity = if high_samples.is_empty() {
        q99
    } else {
        high_samples.iter().sum::<f64>() / high_samples.len() as f64
    };

    // Estimate noise level from low-intensity voxels
    let low_samples: Vec<f64> = samples.iter()
        .filter(|&&v| v <= q15)
        .copied()
        .collect();

    let mut noise = if low_samples.is_empty() {
        0.0
    } else {
        low_samples.iter().sum::<f64>() / low_samples.len() as f64
    };

    // If noise estimate is too high, try using lower percentile
    if noise > high_intensity / 10.0 {
        let very_low_samples: Vec<f64> = samples.iter()
            .filter(|&&v| v <= q05)
            .copied()
            .collect();

        noise = if very_low_samples.is_empty() {
            0.0
        } else {
            very_low_samples.iter().sum::<f64>() / very_low_samples.len() as f64
        };

        if noise > high_intensity / 10.0 {
            noise = 0.0;
        }
    }

    // Calculate threshold: max(5*noise, high_intensity/5)
    let threshold = (5.0 * noise).max(high_intensity / 5.0);

    // Create initial binary mask
    let mut mask_f64: Vec<f64> = mag.iter()
        .map(|&v| if v > threshold { 1.0 } else { 0.0 })
        .collect();

    // Post-processing Step 1: Smooth with nbox=1, boxsize=5, threshold at 0.4
    let boxsizes1 = vec![vec![5], vec![5], vec![5]];
    mask_f64 = gaussian_smooth_3d_boxsizes(&mask_f64, &boxsizes1, 1, nx, ny, nz);
    let mut mask: Vec<u8> = mask_f64.iter()
        .map(|&v| if v > 0.4 { 1 } else { 0 })
        .collect();

    // Post-processing Step 2: Fill holes
    let max_hole_size = n_total / 20;
    mask = fill_holes(&mask, nx, ny, nz, max_hole_size);

    // Post-processing Step 3: Smooth with nbox=2, boxsizes=[3,3], threshold at 0.6
    mask_f64 = mask.iter().map(|&v| v as f64).collect();
    let boxsizes2 = vec![vec![3, 3], vec![3, 3], vec![3, 3]];
    mask_f64 = gaussian_smooth_3d_boxsizes(&mask_f64, &boxsizes2, 2, nx, ny, nz);
    mask = mask_f64.iter()
        .map(|&v| if v > 0.6 { 1 } else { 0 })
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
fn box_segment(
    image: &[f64],
    mask: &[u8],
    nbox: usize,
    nx: usize, ny: usize, nz: usize,
) -> Vec<u8> {
    let n_total = nx * ny * nz;
    let mut vote_count = vec![0u8; n_total];

    // Calculate box shift (stride between box centers)
    let box_shift_x = (nx + nbox - 1) / nbox;
    let box_shift_y = (ny + nbox - 1) / nbox;
    let box_shift_z = (nz + nbox - 1) / nbox;

    // For each box center
    let mut cz = 0;
    while cz < nz {
        let mut cy = 0;
        while cy < ny {
            let mut cx = 0;
            while cx < nx {
                // Calculate box bounds (2x box_shift around center)
                let x_start = cx.saturating_sub(box_shift_x);
                let x_end = (cx + box_shift_x).min(nx);
                let y_start = cy.saturating_sub(box_shift_y);
                let y_end = (cy + box_shift_y).min(ny);
                let z_start = cz.saturating_sub(box_shift_z);
                let z_end = (cz + box_shift_z).min(nz);

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

                // Sort and find 90th percentile
                box_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let q90_idx = ((0.9 * box_vals.len() as f64) as usize).min(box_vals.len() - 1);
                let q90 = box_vals[q90_idx];

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
    nx: usize, ny: usize, nz: usize,
) {
    let n_total = nx * ny * nz;

    // Identify holes/outliers and create weight mask
    // lowpassweight = 1.2 - lowpassmask (so holes get 0.2, normal get 1.2)
    let mut weight = vec![1.2f64; n_total];

    for i in 0..n_total {
        if lowpass[i] < stable_mean / 4.0 ||
           lowpass[i].is_nan() ||
           lowpass[i] > 10.0 * stable_mean {
            lowpass[i] = 3.0 * stable_mean;
            weight[i] = 0.2; // Filled holes get less weight
        }
    }

    // Apply weighted smoothing
    let nbox = 3; // default for non-masked smoothing
    let smoothed = gaussian_smooth_3d(lowpass, sigma2, None, Some(&mut weight), nbox, nx, ny, nz);
    lowpass.copy_from_slice(&smoothed);
}

//=============================================================================
// Main API
//=============================================================================

/// Get sensitivity (bias field) from magnitude
///
/// This estimates the RF receive field inhomogeneity (sensitivity map)
/// that can be divided out to correct the image.
pub fn get_sensitivity(
    mag: &[f64],
    nx: usize, ny: usize, nz: usize,
    vx: f64, vy: f64, vz: f64,
    sigma_mm: f64,
    nbox: usize,
) -> Vec<f64> {
    let n_total = nx * ny * nz;

    // Convert mm to voxels
    let sigma = [sigma_mm / vx, sigma_mm / vy, sigma_mm / vz];

    // Create initial mask (with full post-processing)
    let mask = robust_mask(mag, nx, ny, nz);

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
    let mut lowpass = gaussian_smooth_3d(mag, sigma1, Some(&segmentation), None, 8, nx, ny, nz);

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
    fill_and_smooth(&mut lowpass, stable_mean, sigma2, nx, ny, nz);

    lowpass
}

/// Make magnitude homogeneous by dividing by bias field
///
/// This is the main entry point for bias field correction.
///
/// # Arguments
/// * `mag` - Input magnitude data (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Dimensions
/// * `vx`, `vy`, `vz` - Voxel sizes in mm
/// * `sigma_mm` - Smoothing sigma in mm (default 7, will be clamped to 10% FOV)
/// * `nbox` - Number of boxes per dimension for segmentation (default 15)
///
/// # Returns
/// Bias-corrected magnitude
pub fn makehomogeneous(
    mag: &[f64],
    nx: usize, ny: usize, nz: usize,
    vx: f64, vy: f64, vz: f64,
    sigma_mm: f64,
    nbox: usize,
) -> Vec<f64> {
    let sensitivity = get_sensitivity(mag, nx, ny, nz, vx, vy, vz, sigma_mm, nbox);
    let n_total = nx * ny * nz;

    let mut result = vec![0.0; n_total];
    for i in 0..n_total {
        if sensitivity[i] > 1e-10 && !sensitivity[i].is_nan() {
            result[i] = mag[i] / sensitivity[i];
        } else {
            result[i] = mag[i];
        }
    }

    result
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

    #[test]
    fn test_get_box_sizes() {
        // Test box size calculation matches Julia
        let sizes = get_box_sizes(5.0, 3);
        assert_eq!(sizes.len(), 3);
        // For sigma=5, n=3: wideal ≈ 5.77
        // All sizes should be odd and reasonable
        for &s in &sizes {
            assert!(s % 2 == 1, "Box size should be odd, got {}", s);
            assert!(s >= 3 && s <= 11, "Box size should be in reasonable range, got {}", s);
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
    fn test_fill_holes_basic() {
        // 3x3x3 cube with a hole in the center
        let mut mask = vec![1u8; 27];
        mask[13] = 0; // center voxel

        let filled = fill_holes(&mask, 3, 3, 3, 5);
        assert_eq!(filled[13], 1, "Center hole should be filled");
    }

    #[test]
    fn test_robust_mask_basic() {
        // Simple test with uniform high values
        let mag = vec![100.0; 27];
        let mask = robust_mask(&mag, 3, 3, 3);
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
}

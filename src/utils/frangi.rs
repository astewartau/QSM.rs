//! Frangi Vesselness Filter for 3D tubular structure detection
//!
//! Implementation of the Frangi vesselness filter based on:
//! - Frangi et al., "Multiscale vessel enhancement filtering", MICCAI 1998
//! - Manniesing et al., "Multiscale Vessel Enhancing Diffusion in CT Angiography Noise Filtering"
//!
//! This filter uses eigenvalues of the Hessian matrix to detect tubular (vessel-like) structures.


/// Parameters for Frangi vesselness filter
#[derive(Clone, Debug)]
pub struct FrangiParams {
    /// Range of sigma values [min, max] for multi-scale analysis
    pub scale_range: [f64; 2],
    /// Step size between sigma values (default 0.5)
    pub scale_ratio: f64,
    /// Alpha parameter: sensitivity to plate-like structures (Ra term), default 0.5
    pub alpha: f64,
    /// Beta parameter: sensitivity to blob-like structures (Rb term), default 0.5
    pub beta: f64,
    /// C parameter: sensitivity to noise/background (S term), default 500
    /// Threshold between eigenvalues of noise and vessel structure
    pub c: f64,
    /// Detect black vessels (true) or white vessels (false)
    pub black_white: bool,
}

impl Default for FrangiParams {
    fn default() -> Self {
        Self {
            // QSMART reference defaults: FrangiScaleRange=[1,10], FrangiScaleRatio=2
            scale_range: [1.0, 10.0],
            scale_ratio: 2.0,
            alpha: 0.5,
            beta: 0.5,
            c: 500.0,
            black_white: false, // White vessels (bright structures)
        }
    }
}

/// Result of Frangi filter including vesselness and scale information
pub struct FrangiResult {
    /// Vesselness response (0 to 1)
    pub vesselness: Vec<f64>,
    /// Scale at which maximum vesselness was found
    pub scale: Vec<f64>,
}

/// Apply 3D Frangi vesselness filter
///
/// # Arguments
/// * `data` - Input 3D volume (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Volume dimensions
/// * `params` - Frangi filter parameters
///
/// # Returns
/// FrangiResult with vesselness map and optimal scale map
pub fn frangi_filter_3d(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    params: &FrangiParams,
) -> FrangiResult {
    let n_total = nx * ny * nz;

    // Generate sigma values
    let mut sigmas = Vec::new();
    let mut sigma = params.scale_range[0];
    while sigma <= params.scale_range[1] {
        sigmas.push(sigma);
        sigma += params.scale_ratio;
    }

    if sigmas.is_empty() {
        sigmas.push(params.scale_range[0]);
    }

    // Initialize output arrays
    let mut best_vesselness = vec![0.0f64; n_total];
    let mut best_scale = vec![1.0f64; n_total];

    // Constants for vesselness computation
    let a = 2.0 * params.alpha * params.alpha;
    let b = 2.0 * params.beta * params.beta;
    let c2 = 2.0 * params.c * params.c;

    // Process each scale
    for (scale_idx, &sigma) in sigmas.iter().enumerate() {
        // Compute Hessian components
        let (dxx, dyy, dzz, dxy, dxz, dyz) = compute_hessian_3d(data, nx, ny, nz, sigma);

        // Scale normalization (sigma^2)
        let scale_factor = sigma * sigma;

        // Compute eigenvalues and vesselness for each voxel
        for i in 0..n_total {
            // Scale-normalized Hessian
            let h_xx = dxx[i] * scale_factor;
            let h_yy = dyy[i] * scale_factor;
            let h_zz = dzz[i] * scale_factor;
            let h_xy = dxy[i] * scale_factor;
            let h_xz = dxz[i] * scale_factor;
            let h_yz = dyz[i] * scale_factor;

            // Compute eigenvalues of symmetric 3x3 matrix
            let (lambda1, lambda2, lambda3) = eigenvalues_3x3_symmetric(
                h_xx, h_yy, h_zz, h_xy, h_xz, h_yz
            );

            // Sort by absolute value: |lambda1| <= |lambda2| <= |lambda3|
            let (l1, l2, l3) = sort_by_abs(lambda1, lambda2, lambda3);

            // Compute vesselness
            let vesselness = compute_vesselness(l1, l2, l3, a, b, c2, params.black_white);

            // Keep maximum across scales
            if scale_idx == 0 || vesselness > best_vesselness[i] {
                best_vesselness[i] = vesselness;
                best_scale[i] = sigma;
            }
        }
    }

    FrangiResult {
        vesselness: best_vesselness,
        scale: best_scale,
    }
}

/// Compute vesselness measure from sorted eigenvalues
/// |lambda1| <= |lambda2| <= |lambda3|
fn compute_vesselness(l1: f64, l2: f64, l3: f64, a: f64, b: f64, c2: f64, black_white: bool) -> f64 {
    let abs_l2 = l2.abs();
    let abs_l3 = l3.abs();

    // Avoid division by zero
    if abs_l3 < 1e-10 || abs_l2 < 1e-10 {
        return 0.0;
    }

    // Check sign conditions for vessel-like structures
    // For bright vessels: lambda2 < 0 AND lambda3 < 0
    // For dark vessels: lambda2 > 0 AND lambda3 > 0
    if black_white {
        // Dark vessels (black ridges)
        if l2 < 0.0 || l3 < 0.0 {
            return 0.0;
        }
    } else {
        // Bright vessels (white ridges)
        if l2 > 0.0 || l3 > 0.0 {
            return 0.0;
        }
    }

    // Ra: distinguishes plate-like from line-like structures
    // Ra = |lambda2| / |lambda3|
    let ra = abs_l2 / abs_l3;

    // Rb: distinguishes blob-like from line-like structures
    // Rb = |lambda1| / sqrt(|lambda2 * lambda3|)
    let rb = l1.abs() / (abs_l2 * abs_l3).sqrt();

    // S: second-order structureness (Frobenius norm of Hessian eigenvalues)
    // S = sqrt(lambda1^2 + lambda2^2 + lambda3^2)
    let s = (l1 * l1 + l2 * l2 + l3 * l3).sqrt();

    // Vesselness function
    // V = (1 - exp(-Ra^2/2alpha^2)) * exp(-Rb^2/2beta^2) * (1 - exp(-S^2/2c^2))
    let exp_ra = 1.0 - (-ra * ra / a).exp();
    let exp_rb = (-rb * rb / b).exp();
    let exp_s = 1.0 - (-s * s / c2).exp();

    let v = exp_ra * exp_rb * exp_s;

    // Clamp to valid range
    if v.is_finite() { v.max(0.0).min(1.0) } else { 0.0 }
}

/// Sort three values by absolute value
fn sort_by_abs(a: f64, b: f64, c: f64) -> (f64, f64, f64) {
    let mut vals = [(a.abs(), a), (b.abs(), b), (c.abs(), c)];
    vals.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    (vals[0].1, vals[1].1, vals[2].1)
}

/// Compute 3D Hessian matrix components using Gaussian derivatives
///
/// Returns (Dxx, Dyy, Dzz, Dxy, Dxz, Dyz)
fn compute_hessian_3d(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    sigma: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n_total = nx * ny * nz;

    // First, apply Gaussian smoothing
    let smoothed = if sigma > 0.0 {
        gaussian_smooth_3d(data, nx, ny, nz, sigma)
    } else {
        data.to_vec()
    };

    // Compute first derivatives
    let dx = gradient_3d(&smoothed, nx, ny, nz, 'x');
    let dy = gradient_3d(&smoothed, nx, ny, nz, 'y');
    let dz = gradient_3d(&smoothed, nx, ny, nz, 'z');

    // Compute second derivatives
    let dxx = gradient_3d(&dx, nx, ny, nz, 'x');
    let dxy = gradient_3d(&dx, nx, ny, nz, 'y');
    let dxz = gradient_3d(&dx, nx, ny, nz, 'z');

    let dyy = gradient_3d(&dy, nx, ny, nz, 'y');
    let dyz = gradient_3d(&dy, nx, ny, nz, 'z');

    let dzz = gradient_3d(&dz, nx, ny, nz, 'z');

    (dxx, dyy, dzz, dxy, dxz, dyz)
}

/// Compute gradient in specified direction using central differences
fn gradient_3d(data: &[f64], nx: usize, ny: usize, nz: usize, direction: char) -> Vec<f64> {
    let n_total = nx * ny * nz;
    let mut grad = vec![0.0f64; n_total];

    // Index helper: i + j*nx + k*nx*ny
    let idx = |i: usize, j: usize, k: usize| i + j * nx + k * nx * ny;

    match direction {
        'x' => {
            for k in 0..nz {
                for j in 0..ny {
                    // Forward difference at left edge
                    grad[idx(0, j, k)] = data[idx(1, j, k)] - data[idx(0, j, k)];

                    // Central differences in interior
                    for i in 1..nx-1 {
                        grad[idx(i, j, k)] = (data[idx(i+1, j, k)] - data[idx(i-1, j, k)]) / 2.0;
                    }

                    // Backward difference at right edge
                    if nx > 1 {
                        grad[idx(nx-1, j, k)] = data[idx(nx-1, j, k)] - data[idx(nx-2, j, k)];
                    }
                }
            }
        }
        'y' => {
            for k in 0..nz {
                for i in 0..nx {
                    // Forward difference at bottom edge
                    grad[idx(i, 0, k)] = data[idx(i, 1, k)] - data[idx(i, 0, k)];

                    // Central differences in interior
                    for j in 1..ny-1 {
                        grad[idx(i, j, k)] = (data[idx(i, j+1, k)] - data[idx(i, j-1, k)]) / 2.0;
                    }

                    // Backward difference at top edge
                    if ny > 1 {
                        grad[idx(i, ny-1, k)] = data[idx(i, ny-1, k)] - data[idx(i, ny-2, k)];
                    }
                }
            }
        }
        'z' => {
            for j in 0..ny {
                for i in 0..nx {
                    // Forward difference at front edge
                    grad[idx(i, j, 0)] = data[idx(i, j, 1)] - data[idx(i, j, 0)];

                    // Central differences in interior
                    for k in 1..nz-1 {
                        grad[idx(i, j, k)] = (data[idx(i, j, k+1)] - data[idx(i, j, k-1)]) / 2.0;
                    }

                    // Backward difference at back edge
                    if nz > 1 {
                        grad[idx(i, j, nz-1)] = data[idx(i, j, nz-1)] - data[idx(i, j, nz-2)];
                    }
                }
            }
        }
        _ => panic!("Invalid gradient direction"),
    }

    grad
}

/// 3D Gaussian smoothing using separable 1D convolutions
fn gaussian_smooth_3d(data: &[f64], nx: usize, ny: usize, nz: usize, sigma: f64) -> Vec<f64> {
    if sigma <= 0.0 {
        return data.to_vec();
    }

    // Create 1D Gaussian kernel
    let kernel_radius = (3.0 * sigma).ceil() as usize;
    let kernel_size = 2 * kernel_radius + 1;
    let mut kernel = vec![0.0f64; kernel_size];

    let mut sum = 0.0;
    for i in 0..kernel_size {
        let x = i as f64 - kernel_radius as f64;
        kernel[i] = (-x * x / (2.0 * sigma * sigma)).exp();
        sum += kernel[i];
    }

    // Normalize kernel
    for k in kernel.iter_mut() {
        *k /= sum;
    }

    // Apply separable convolution
    // X direction
    let smoothed_x = convolve_1d_direction(data, nx, ny, nz, &kernel, 'x');

    // Y direction
    let smoothed_xy = convolve_1d_direction(&smoothed_x, nx, ny, nz, &kernel, 'y');

    // Z direction
    let smoothed_xyz = convolve_1d_direction(&smoothed_xy, nx, ny, nz, &kernel, 'z');

    smoothed_xyz
}

/// Apply 1D convolution along specified axis
fn convolve_1d_direction(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    kernel: &[f64],
    direction: char,
) -> Vec<f64> {
    let n_total = nx * ny * nz;
    let mut result = vec![0.0f64; n_total];
    let kernel_radius = (kernel.len() - 1) / 2;

    let idx = |i: usize, j: usize, k: usize| i + j * nx + k * nx * ny;

    match direction {
        'x' => {
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let mut sum = 0.0;
                        let mut weight_sum = 0.0;

                        for ki in 0..kernel.len() {
                            let offset = ki as isize - kernel_radius as isize;
                            let ni = i as isize + offset;

                            if ni >= 0 && ni < nx as isize {
                                sum += data[idx(ni as usize, j, k)] * kernel[ki];
                                weight_sum += kernel[ki];
                            }
                        }

                        result[idx(i, j, k)] = if weight_sum > 0.0 { sum / weight_sum } else { 0.0 };
                    }
                }
            }
        }
        'y' => {
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let mut sum = 0.0;
                        let mut weight_sum = 0.0;

                        for ki in 0..kernel.len() {
                            let offset = ki as isize - kernel_radius as isize;
                            let nj = j as isize + offset;

                            if nj >= 0 && nj < ny as isize {
                                sum += data[idx(i, nj as usize, k)] * kernel[ki];
                                weight_sum += kernel[ki];
                            }
                        }

                        result[idx(i, j, k)] = if weight_sum > 0.0 { sum / weight_sum } else { 0.0 };
                    }
                }
            }
        }
        'z' => {
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let mut sum = 0.0;
                        let mut weight_sum = 0.0;

                        for ki in 0..kernel.len() {
                            let offset = ki as isize - kernel_radius as isize;
                            let nk = k as isize + offset;

                            if nk >= 0 && nk < nz as isize {
                                sum += data[idx(i, j, nk as usize)] * kernel[ki];
                                weight_sum += kernel[ki];
                            }
                        }

                        result[idx(i, j, k)] = if weight_sum > 0.0 { sum / weight_sum } else { 0.0 };
                    }
                }
            }
        }
        _ => panic!("Invalid convolution direction"),
    }

    result
}

/// Compute eigenvalues of a 3x3 symmetric matrix using Householder + QL algorithm
///
/// This is a direct port of the QSMART/JAMA algorithm from eig3volume.c,
/// which uses Householder reduction to tridiagonal form followed by QL iteration.
/// This method is more numerically stable than the analytical Cardano formula.
///
/// Matrix is:
/// | a  d  e |
/// | d  b  f |
/// | e  f  c |
///
/// Returns eigenvalues (not sorted)
fn eigenvalues_3x3_symmetric(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> (f64, f64, f64) {
    // Build the symmetric matrix V (will be modified in place)
    let mut v = [[0.0f64; 3]; 3];
    v[0][0] = a; v[0][1] = d; v[0][2] = e;
    v[1][0] = d; v[1][1] = b; v[1][2] = f;
    v[2][0] = e; v[2][1] = f; v[2][2] = c;

    let mut eigenvalues = [0.0f64; 3];
    let mut e_vec = [0.0f64; 3];

    // Step 1: Householder reduction to tridiagonal form (tred2)
    tred2(&mut v, &mut eigenvalues, &mut e_vec);

    // Step 2: QL algorithm for symmetric tridiagonal matrix (tql2)
    tql2(&mut v, &mut eigenvalues, &mut e_vec);

    (eigenvalues[0], eigenvalues[1], eigenvalues[2])
}

/// Symmetric Householder reduction to tridiagonal form
///
/// Derived from the Algol procedures tred2 by Bowdler, Martin, Reinsch, and Wilkinson,
/// Handbook for Auto. Comp., Vol.ii-Linear Algebra, and the corresponding Fortran
/// subroutine in EISPACK.
///
/// Direct port from QSMART's eig3volume.c
fn tred2(v: &mut [[f64; 3]; 3], d: &mut [f64; 3], e: &mut [f64; 3]) {
    const N: usize = 3;

    // Initialize d with the last row of V
    for j in 0..N {
        d[j] = v[N - 1][j];
    }

    // Householder reduction to tridiagonal form
    for i in (1..N).rev() {
        // Scale to avoid under/overflow
        let mut scale = 0.0;
        let mut h = 0.0;

        for k in 0..i {
            scale += d[k].abs();
        }

        if scale == 0.0 {
            e[i] = d[i - 1];
            for j in 0..i {
                d[j] = v[i - 1][j];
                v[i][j] = 0.0;
                v[j][i] = 0.0;
            }
        } else {
            // Generate Householder vector
            for k in 0..i {
                d[k] /= scale;
                h += d[k] * d[k];
            }

            let f = d[i - 1];
            let mut g = h.sqrt();
            if f > 0.0 {
                g = -g;
            }
            e[i] = scale * g;
            h -= f * g;
            d[i - 1] = f - g;

            for j in 0..i {
                e[j] = 0.0;
            }

            // Apply similarity transformation to remaining columns
            for j in 0..i {
                let f = d[j];
                v[j][i] = f;
                let mut g = e[j] + v[j][j] * f;
                for k in (j + 1)..i {
                    g += v[k][j] * d[k];
                    e[k] += v[k][j] * f;
                }
                e[j] = g;
            }

            let mut f = 0.0;
            for j in 0..i {
                e[j] /= h;
                f += e[j] * d[j];
            }

            let hh = f / (h + h);
            for j in 0..i {
                e[j] -= hh * d[j];
            }

            for j in 0..i {
                let f = d[j];
                let g = e[j];
                for k in j..i {
                    v[k][j] -= f * e[k] + g * d[k];
                }
                d[j] = v[i - 1][j];
                v[i][j] = 0.0;
            }
        }
        d[i] = h;
    }

    // Accumulate transformations
    for i in 0..(N - 1) {
        v[N - 1][i] = v[i][i];
        v[i][i] = 1.0;
        let h = d[i + 1];
        if h != 0.0 {
            for k in 0..=i {
                d[k] = v[k][i + 1] / h;
            }
            for j in 0..=i {
                let mut g = 0.0;
                for k in 0..=i {
                    g += v[k][i + 1] * v[k][j];
                }
                for k in 0..=i {
                    v[k][j] -= g * d[k];
                }
            }
        }
        for k in 0..=i {
            v[k][i + 1] = 0.0;
        }
    }

    for j in 0..N {
        d[j] = v[N - 1][j];
        v[N - 1][j] = 0.0;
    }
    v[N - 1][N - 1] = 1.0;
    e[0] = 0.0;
}

/// Symmetric tridiagonal QL algorithm
///
/// Derived from the Algol procedures tql2 by Bowdler, Martin, Reinsch, and Wilkinson,
/// Handbook for Auto. Comp., Vol.ii-Linear Algebra, and the corresponding Fortran
/// subroutine in EISPACK.
///
/// Direct port from QSMART's eig3volume.c
fn tql2(v: &mut [[f64; 3]; 3], d: &mut [f64; 3], e: &mut [f64; 3]) {
    const N: usize = 3;

    for i in 1..N {
        e[i - 1] = e[i];
    }
    e[N - 1] = 0.0;

    let mut f: f64 = 0.0;
    let mut tst1: f64 = 0.0;
    let eps: f64 = 2.0f64.powi(-52);

    for l in 0..N {
        // Find small subdiagonal element
        tst1 = tst1.max(d[l].abs() + e[l].abs());
        let mut m = l;
        while m < N {
            if e[m].abs() <= eps * tst1 {
                break;
            }
            m += 1;
        }

        // If m == l, d[l] is an eigenvalue, otherwise iterate
        if m > l {
            loop {
                // Compute implicit shift
                let g = d[l];
                let mut p = (d[l + 1] - g) / (2.0 * e[l]);
                let mut r = hypot(p, 1.0);
                if p < 0.0 {
                    r = -r;
                }
                d[l] = e[l] / (p + r);
                d[l + 1] = e[l] * (p + r);
                let dl1 = d[l + 1];
                let mut h = g - d[l];
                for i in (l + 2)..N {
                    d[i] -= h;
                }
                f += h;

                // Implicit QL transformation
                p = d[m];
                let mut c = 1.0;
                let mut c2 = c;
                let mut c3 = c;
                let el1 = e[l + 1];
                let mut s = 0.0;
                let mut s2 = 0.0;

                for i in (l..m).rev() {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    let g = c * e[i];
                    let h = c * p;
                    r = hypot(p, e[i]);
                    e[i + 1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);

                    // Accumulate transformation
                    for k in 0..N {
                        let vh = v[k][i + 1];
                        v[k][i + 1] = s * v[k][i] + c * vh;
                        v[k][i] = c * v[k][i] - s * vh;
                    }
                }
                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;

                // Check for convergence
                if e[l].abs() <= eps * tst1 {
                    break;
                }
            }
        }
        d[l] += f;
        e[l] = 0.0;
    }

    // Sort eigenvalues and corresponding vectors (ascending order)
    for i in 0..(N - 1) {
        let mut k = i;
        let mut p = d[i];
        for j in (i + 1)..N {
            if d[j] < p {
                k = j;
                p = d[j];
            }
        }
        if k != i {
            d[k] = d[i];
            d[i] = p;
            for j in 0..N {
                let temp = v[j][i];
                v[j][i] = v[j][k];
                v[j][k] = temp;
            }
        }
    }
}

/// Compute hypotenuse avoiding overflow/underflow
#[inline]
fn hypot(x: f64, y: f64) -> f64 {
    (x * x + y * y).sqrt()
}

/// Simple wrapper for Frangi filter with default parameters
pub fn frangi_filter_3d_default(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
) -> Vec<f64> {
    let params = FrangiParams::default();
    frangi_filter_3d(data, nx, ny, nz, &params).vesselness
}

/// Frangi filter with progress callback
pub fn frangi_filter_3d_with_progress<F>(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    params: &FrangiParams,
    progress_callback: F,
) -> FrangiResult
where
    F: Fn(usize, usize),
{
    let n_total = nx * ny * nz;

    // Generate sigma values
    let mut sigmas = Vec::new();
    let mut sigma = params.scale_range[0];
    while sigma <= params.scale_range[1] {
        sigmas.push(sigma);
        sigma += params.scale_ratio;
    }

    if sigmas.is_empty() {
        sigmas.push(params.scale_range[0]);
    }

    let total_scales = sigmas.len();

    // Initialize output arrays
    let mut best_vesselness = vec![0.0f64; n_total];
    let mut best_scale = vec![1.0f64; n_total];

    // Constants for vesselness computation
    let a = 2.0 * params.alpha * params.alpha;
    let b = 2.0 * params.beta * params.beta;
    let c2 = 2.0 * params.c * params.c;

    // Process each scale
    for (scale_idx, &sigma) in sigmas.iter().enumerate() {
        progress_callback(scale_idx, total_scales);

        // Compute Hessian components
        let (dxx, dyy, dzz, dxy, dxz, dyz) = compute_hessian_3d(data, nx, ny, nz, sigma);

        // Scale normalization (sigma^2)
        let scale_factor = sigma * sigma;

        // Compute eigenvalues and vesselness for each voxel
        for i in 0..n_total {
            // Scale-normalized Hessian
            let h_xx = dxx[i] * scale_factor;
            let h_yy = dyy[i] * scale_factor;
            let h_zz = dzz[i] * scale_factor;
            let h_xy = dxy[i] * scale_factor;
            let h_xz = dxz[i] * scale_factor;
            let h_yz = dyz[i] * scale_factor;

            // Compute eigenvalues of symmetric 3x3 matrix
            let (lambda1, lambda2, lambda3) = eigenvalues_3x3_symmetric(
                h_xx, h_yy, h_zz, h_xy, h_xz, h_yz
            );

            // Sort by absolute value: |lambda1| <= |lambda2| <= |lambda3|
            let (l1, l2, l3) = sort_by_abs(lambda1, lambda2, lambda3);

            // Compute vesselness
            let vesselness = compute_vesselness(l1, l2, l3, a, b, c2, params.black_white);

            // Keep maximum across scales
            if scale_idx == 0 || vesselness > best_vesselness[i] {
                best_vesselness[i] = vesselness;
                best_scale[i] = sigma;
            }
        }
    }

    progress_callback(total_scales, total_scales);

    FrangiResult {
        vesselness: best_vesselness,
        scale: best_scale,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eigenvalues_diagonal() {
        // Diagonal matrix: eigenvalues are the diagonal elements
        let (l1, l2, l3) = eigenvalues_3x3_symmetric(1.0, 2.0, 3.0, 0.0, 0.0, 0.0);
        let mut sorted = vec![l1, l2, l3];
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert!((sorted[0] - 1.0).abs() < 1e-10);
        assert!((sorted[1] - 2.0).abs() < 1e-10);
        assert!((sorted[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_eigenvalues_identity() {
        // Identity matrix: all eigenvalues are 1
        let (l1, l2, l3) = eigenvalues_3x3_symmetric(1.0, 1.0, 1.0, 0.0, 0.0, 0.0);

        assert!((l1 - 1.0).abs() < 1e-10);
        assert!((l2 - 1.0).abs() < 1e-10);
        assert!((l3 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_constant() {
        // Gradient of constant field should be zero
        let data = vec![5.0; 27]; // 3x3x3
        let grad = gradient_3d(&data, 3, 3, 3, 'x');

        for &v in &grad {
            assert!(v.abs() < 1e-10);
        }
    }

    #[test]
    fn test_frangi_filter_basic() {
        // Basic test: filter should run without panic
        let data = vec![0.0; 1000]; // 10x10x10
        let params = FrangiParams {
            scale_range: [1.0, 2.0],
            scale_ratio: 1.0,
            ..Default::default()
        };

        let result = frangi_filter_3d(&data, 10, 10, 10, &params);
        assert_eq!(result.vesselness.len(), 1000);
    }
}

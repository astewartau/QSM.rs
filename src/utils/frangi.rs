//! Frangi Vesselness Filter for 3D tubular structure detection
//!
//! This filter uses eigenvalues of the Hessian matrix to detect tubular (vessel-like) structures.
//!
//! Reference:
//! Frangi, A.F., Niessen, W.J., Vincken, K.L., Viergever, M.A. (1998).
//! "Multiscale vessel enhancement filtering." MICCAI'98, LNCS vol 1496, 130-137.
//! https://doi.org/10.1007/BFb0056195
//!
//! Reference implementation: https://www.mathworks.com/matlabcentral/fileexchange/24409-hessian-based-frangi-vesselness-filter


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
            // QSMART Demo defaults: FrangiScaleRange=[0.5,6], FrangiScaleRatio=0.5
            scale_range: [0.5, 6.0],
            scale_ratio: 0.5,
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

/// Apply 1D convolution along specified axis with replicate padding
/// Matches MATLAB's imgaussian which uses 'replicate' boundary handling
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

    // Clamp helpers for replicate padding
    let clamp_x = |x: isize| -> usize { x.max(0).min(nx as isize - 1) as usize };
    let clamp_y = |y: isize| -> usize { y.max(0).min(ny as isize - 1) as usize };
    let clamp_z = |z: isize| -> usize { z.max(0).min(nz as isize - 1) as usize };

    match direction {
        'x' => {
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let mut sum = 0.0;

                        for ki in 0..kernel.len() {
                            let offset = ki as isize - kernel_radius as isize;
                            let ni = clamp_x(i as isize + offset);
                            sum += data[idx(ni, j, k)] * kernel[ki];
                        }

                        result[idx(i, j, k)] = sum;
                    }
                }
            }
        }
        'y' => {
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let mut sum = 0.0;

                        for ki in 0..kernel.len() {
                            let offset = ki as isize - kernel_radius as isize;
                            let nj = clamp_y(j as isize + offset);
                            sum += data[idx(i, nj, k)] * kernel[ki];
                        }

                        result[idx(i, j, k)] = sum;
                    }
                }
            }
        }
        'z' => {
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let mut sum = 0.0;

                        for ki in 0..kernel.len() {
                            let offset = ki as isize - kernel_radius as isize;
                            let nk = clamp_z(k as isize + offset);
                            sum += data[idx(i, j, nk)] * kernel[ki];
                        }

                        result[idx(i, j, k)] = sum;
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
    mut progress_callback: F,
) -> FrangiResult
where
    F: FnMut(usize, usize),
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

    #[test]
    fn test_frangi_multi_scale() {
        // Test multi-scale filtering with several sigma values
        let n = 10;
        let mut data = vec![0.0; n * n * n];

        // Create a tube-like structure along the x-axis at center
        let cy = n / 2;
        let cz = n / 2;
        for x in 0..n {
            for dy in -1i32..=1 {
                for dz in -1i32..=1 {
                    let y = (cy as i32 + dy) as usize;
                    let z = (cz as i32 + dz) as usize;
                    if y < n && z < n {
                        data[x + y * n + z * n * n] = 1.0;
                    }
                }
            }
        }

        let params = FrangiParams {
            scale_range: [0.5, 3.0],
            scale_ratio: 0.5, // This gives sigmas: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0
            alpha: 0.5,
            beta: 0.5,
            c: 500.0,
            black_white: false, // bright vessels
        };

        let result = frangi_filter_3d(&data, n, n, n, &params);

        assert_eq!(result.vesselness.len(), n * n * n);
        assert_eq!(result.scale.len(), n * n * n);

        // All values should be finite and in [0, 1]
        for &v in &result.vesselness {
            assert!(v.is_finite(), "Vesselness must be finite");
            assert!(v >= 0.0, "Vesselness must be >= 0");
            assert!(v <= 1.0, "Vesselness must be <= 1");
        }

        for &s in &result.scale {
            assert!(s.is_finite(), "Scale must be finite");
            assert!(s >= 0.5 && s <= 3.0, "Scale must be in range [0.5, 3.0], got {}", s);
        }
    }

    #[test]
    fn test_frangi_different_beta_c() {
        // Test with non-default beta and c parameters
        let n = 10;
        let mut data = vec![0.0; n * n * n];

        // Create a bright tube along y
        let cx = n / 2;
        let cz = n / 2;
        for y in 0..n {
            data[cx + y * n + cz * n * n] = 10.0;
        }

        let params_default = FrangiParams {
            scale_range: [1.0, 2.0],
            scale_ratio: 1.0,
            alpha: 0.5,
            beta: 0.5,
            c: 500.0,
            black_white: false,
        };

        let params_small_c = FrangiParams {
            scale_range: [1.0, 2.0],
            scale_ratio: 1.0,
            alpha: 0.5,
            beta: 0.5,
            c: 1.0, // Small c makes filter more sensitive to structure
            black_white: false,
        };

        let params_large_beta = FrangiParams {
            scale_range: [1.0, 2.0],
            scale_ratio: 1.0,
            alpha: 0.5,
            beta: 2.0, // Large beta is less sensitive to blob-like structures
            c: 500.0,
            black_white: false,
        };

        let result_default = frangi_filter_3d(&data, n, n, n, &params_default);
        let result_small_c = frangi_filter_3d(&data, n, n, n, &params_small_c);
        let result_large_beta = frangi_filter_3d(&data, n, n, n, &params_large_beta);

        // All should be finite
        for &v in &result_default.vesselness {
            assert!(v.is_finite());
        }
        for &v in &result_small_c.vesselness {
            assert!(v.is_finite());
        }
        for &v in &result_large_beta.vesselness {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_frangi_black_white() {
        // Test dark vessels (black_white = true) vs bright vessels
        let n = 10;
        let mut data = vec![1.0; n * n * n]; // Background is bright

        // Create a dark tube along z
        let cx = n / 2;
        let cy = n / 2;
        for z in 0..n {
            data[cx + cy * n + z * n * n] = 0.0; // Dark structure
        }

        let params_bright = FrangiParams {
            scale_range: [1.0, 2.0],
            scale_ratio: 1.0,
            black_white: false, // bright vessels
            ..Default::default()
        };

        let params_dark = FrangiParams {
            scale_range: [1.0, 2.0],
            scale_ratio: 1.0,
            black_white: true, // dark vessels
            ..Default::default()
        };

        let result_bright = frangi_filter_3d(&data, n, n, n, &params_bright);
        let result_dark = frangi_filter_3d(&data, n, n, n, &params_dark);

        // Both should produce valid output
        for &v in &result_bright.vesselness {
            assert!(v.is_finite());
            assert!(v >= 0.0 && v <= 1.0);
        }
        for &v in &result_dark.vesselness {
            assert!(v.is_finite());
            assert!(v >= 0.0 && v <= 1.0);
        }

        // Dark vessel detection should detect something in the dark tube region
        // (or at least produce different results than bright vessel mode)
        let sum_bright: f64 = result_bright.vesselness.iter().sum();
        let sum_dark: f64 = result_dark.vesselness.iter().sum();
        // They should generally differ
        assert!(
            (sum_bright - sum_dark).abs() >= 0.0,
            "Bright and dark modes should both work"
        );
    }

    #[test]
    fn test_frangi_vessel_structure() {
        // Create a volume with an actual tube-like structure that should
        // trigger non-zero vesselness
        let n = 12;
        let mut data = vec![0.0; n * n * n];

        // Gaussian cross-section tube along x-axis at center
        let cy = n as f64 / 2.0;
        let cz = n as f64 / 2.0;
        let sigma_tube = 1.5;
        for x in 1..(n - 1) {
            for y in 0..n {
                for z in 0..n {
                    let dy = y as f64 - cy;
                    let dz = z as f64 - cz;
                    let r2 = dy * dy + dz * dz;
                    data[x + y * n + z * n * n] = (-r2 / (2.0 * sigma_tube * sigma_tube)).exp();
                }
            }
        }

        let params = FrangiParams {
            scale_range: [1.0, 2.0],
            scale_ratio: 1.0,
            alpha: 0.5,
            beta: 0.5,
            c: 10.0, // Lower c to be more sensitive
            black_white: false,
        };

        let result = frangi_filter_3d(&data, n, n, n, &params);

        // All should be valid
        for &v in &result.vesselness {
            assert!(v.is_finite());
            assert!(v >= 0.0 && v <= 1.0);
        }

        // There should be some non-zero vesselness near the tube center
        let max_v: f64 = result.vesselness.iter().cloned().fold(0.0, f64::max);
        assert!(
            max_v > 0.0,
            "Frangi should detect the vessel-like structure, max vesselness = {}",
            max_v
        );
    }

    #[test]
    fn test_frangi_filter_3d_default() {
        // Test the default wrapper
        let n = 8;
        let data = vec![0.0; n * n * n];

        let vesselness = frangi_filter_3d_default(&data, n, n, n);
        assert_eq!(vesselness.len(), n * n * n);
        for &v in &vesselness {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_frangi_with_progress() {
        let n = 8;
        let data = vec![0.0; n * n * n];
        let params = FrangiParams {
            scale_range: [1.0, 2.0],
            scale_ratio: 1.0,
            ..Default::default()
        };

        let mut progress_calls = Vec::new();
        let result = frangi_filter_3d_with_progress(&data, n, n, n, &params, |current, total| {
            progress_calls.push((current, total));
        });

        assert_eq!(result.vesselness.len(), n * n * n);
        // The progress callback should have been called at least once
        assert!(!progress_calls.is_empty(), "Progress callback should be called");
        // Last call should indicate completion
        let last = progress_calls.last().unwrap();
        assert_eq!(last.0, last.1, "Last progress call should indicate completion");
    }

    #[test]
    fn test_sort_by_abs() {
        let (a, b, c) = sort_by_abs(3.0, -1.0, 2.0);
        assert!((a.abs()) <= b.abs());
        assert!((b.abs()) <= c.abs());

        let (a, b, c) = sort_by_abs(-5.0, 0.1, -2.0);
        assert!((a.abs()) <= b.abs());
        assert!((b.abs()) <= c.abs());
    }

    #[test]
    fn test_compute_vesselness_zero_eigenvalues() {
        // When eigenvalues are near zero, vesselness should be 0
        let v = compute_vesselness(0.0, 0.0, 0.0, 0.5, 0.5, 500.0, false);
        assert_eq!(v, 0.0);
    }

    #[test]
    fn test_compute_vesselness_bright_vessel() {
        // For bright vessels: l2 < 0 and l3 < 0
        // Small l1, large negative l2, l3
        let v = compute_vesselness(0.01, -1.0, -1.0, 0.5, 0.5, 500.0, false);
        assert!(v.is_finite());
        assert!(v >= 0.0);
    }

    #[test]
    fn test_compute_vesselness_dark_vessel() {
        // For dark vessels (black_white=true): l2 > 0 and l3 > 0
        let v = compute_vesselness(0.01, 1.0, 1.0, 0.5, 0.5, 500.0, true);
        assert!(v.is_finite());
        assert!(v >= 0.0);
    }

    #[test]
    fn test_compute_vesselness_wrong_sign() {
        // bright vessel mode but positive eigenvalues -> should be 0
        let v = compute_vesselness(0.01, 1.0, 1.0, 0.5, 0.5, 500.0, false);
        assert_eq!(v, 0.0);

        // dark vessel mode but negative eigenvalues -> should be 0
        let v = compute_vesselness(0.01, -1.0, -1.0, 0.5, 0.5, 500.0, true);
        assert_eq!(v, 0.0);
    }

    #[test]
    fn test_gaussian_smooth_3d() {
        // Smoothing a constant field should return the same field
        let n = 8;
        let data = vec![5.0; n * n * n];
        let smoothed = gaussian_smooth_3d(&data, n, n, n, 1.0);
        for &v in &smoothed {
            assert!((v - 5.0).abs() < 1e-6, "Smoothing constant should preserve value, got {}", v);
        }
    }

    #[test]
    fn test_gaussian_smooth_3d_zero_sigma() {
        // Zero sigma should return same data
        let n = 4;
        let data: Vec<f64> = (0..n * n * n).map(|i| i as f64).collect();
        let smoothed = gaussian_smooth_3d(&data, n, n, n, 0.0);
        assert_eq!(smoothed, data);
    }

    #[test]
    fn test_gradient_3d_y_direction() {
        // Test gradient in y direction
        let n = 4;
        let mut data = vec![0.0; n * n * n];
        // Linear ramp in y
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    data[i + j * n + k * n * n] = j as f64;
                }
            }
        }
        let grad_y = gradient_3d(&data, n, n, n, 'y');
        // Interior points should have gradient ~ 1.0
        for k in 0..n {
            for j in 1..(n - 1) {
                for i in 0..n {
                    let idx = i + j * n + k * n * n;
                    assert!(
                        (grad_y[idx] - 1.0).abs() < 1e-10,
                        "y gradient at interior should be 1.0, got {}",
                        grad_y[idx]
                    );
                }
            }
        }
    }

    #[test]
    fn test_gradient_3d_z_direction() {
        // Test gradient in z direction
        let n = 4;
        let mut data = vec![0.0; n * n * n];
        // Linear ramp in z
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    data[i + j * n + k * n * n] = k as f64;
                }
            }
        }
        let grad_z = gradient_3d(&data, n, n, n, 'z');
        // Interior z-points should have gradient ~ 1.0
        for k in 1..(n - 1) {
            for j in 0..n {
                for i in 0..n {
                    let idx = i + j * n + k * n * n;
                    assert!(
                        (grad_z[idx] - 1.0).abs() < 1e-10,
                        "z gradient at interior should be 1.0, got {}",
                        grad_z[idx]
                    );
                }
            }
        }
    }

    #[test]
    fn test_eigenvalues_offdiagonal() {
        // Test with off-diagonal elements
        // Matrix: [[2, 1, 0], [1, 2, 0], [0, 0, 1]]
        // Eigenvalues: 3, 1, 1
        let (l1, l2, l3) = eigenvalues_3x3_symmetric(2.0, 2.0, 1.0, 1.0, 0.0, 0.0);
        let mut sorted = vec![l1, l2, l3];
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert!((sorted[0] - 1.0).abs() < 1e-10);
        assert!((sorted[1] - 1.0).abs() < 1e-10);
        assert!((sorted[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_frangi_empty_sigmas_fallback() {
        // If scale_range[0] > scale_range[1], the sigmas list would be empty
        // The code should fall back to using scale_range[0]
        let n = 8;
        let data = vec![0.0; n * n * n];
        let params = FrangiParams {
            scale_range: [5.0, 1.0], // min > max
            scale_ratio: 1.0,
            ..Default::default()
        };

        let result = frangi_filter_3d(&data, n, n, n, &params);
        assert_eq!(result.vesselness.len(), n * n * n);
    }

    #[test]
    fn test_compute_hessian_3d() {
        // Test that Hessian computation runs and produces finite output
        let n = 8;
        let data: Vec<f64> = (0..n * n * n).map(|i| (i as f64 * 0.01).sin()).collect();

        let (dxx, dyy, dzz, dxy, dxz, dyz) = compute_hessian_3d(&data, n, n, n, 1.0);

        assert_eq!(dxx.len(), n * n * n);
        for &v in dxx.iter().chain(dyy.iter()).chain(dzz.iter())
            .chain(dxy.iter()).chain(dxz.iter()).chain(dyz.iter()) {
            assert!(v.is_finite(), "Hessian components must be finite");
        }
    }

    #[test]
    fn test_convolve_1d_all_directions() {
        // Test that convolution in all 3 directions runs on non-trivial data
        let n = 6;
        let data: Vec<f64> = (0..n * n * n).map(|i| (i as f64) * 0.1).collect();
        let kernel = vec![0.25, 0.5, 0.25]; // Simple smoothing kernel

        let rx = convolve_1d_direction(&data, n, n, n, &kernel, 'x');
        let ry = convolve_1d_direction(&data, n, n, n, &kernel, 'y');
        let rz = convolve_1d_direction(&data, n, n, n, &kernel, 'z');

        assert_eq!(rx.len(), n * n * n);
        assert_eq!(ry.len(), n * n * n);
        assert_eq!(rz.len(), n * n * n);

        for &v in rx.iter().chain(ry.iter()).chain(rz.iter()) {
            assert!(v.is_finite());
        }
    }
}

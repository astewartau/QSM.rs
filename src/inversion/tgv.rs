//! TGV-QSM: Total Generalized Variation for Quantitative Susceptibility Mapping
//!
//! Single-step QSM reconstruction from wrapped phase data using TGV regularization.
//! Based on Langkammer et al. (2015) "Fast quantitative susceptibility mapping using 3D EPI"
//! and Bredies et al. (2014) "Single-step quantitative susceptibility mapping using TGV".
//!
//! The algorithm solves:
//!   min_χ ||∇²(phase) - D*χ||₂² + α₁||∇χ - w||₁ + α₀||ε(w)||₁
//!
//! where:
//! - χ is the susceptibility map
//! - w is an auxiliary vector field (velocity)
//! - ε(w) is the symmetric gradient of w
//! - D is the dipole kernel
//! - α₀, α₁ are TGV regularization parameters
//!
//! Optimizations:
//! - Bounding box reduction: only process the region containing the mask
//! - Pre-allocated buffers: all temporary arrays allocated once outside the loop
//! - Early termination: convergence check every 100 iterations

use std::f32::consts::PI;

/// TGV parameters
#[derive(Clone, Debug)]
pub struct TgvParams {
    /// First-order TGV weight (gradient term)
    pub alpha1: f32,
    /// Second-order TGV weight (symmetric gradient term)
    pub alpha0: f32,
    /// Number of primal-dual iterations
    pub iterations: usize,
    /// Number of mask erosions
    pub erosions: usize,
    /// Primal step size multiplier (larger = faster but less stable)
    pub step_size: f32,
    /// Field strength in Tesla
    pub fieldstrength: f32,
    /// Echo time in seconds
    pub te: f32,
    /// Convergence tolerance (relative change in chi)
    pub tol: f32,
}

impl Default for TgvParams {
    fn default() -> Self {
        Self {
            alpha1: 0.003,
            alpha0: 0.002,
            iterations: 1000,
            erosions: 3,
            step_size: 3.0,
            fieldstrength: 3.0,
            te: 0.020, // 20ms
            tol: 1e-5,
        }
    }
}

/// Get default alpha values based on regularization level (1-4)
pub fn get_default_alpha(regularization: u8) -> (f32, f32) {
    let reg = regularization.clamp(1, 4) as f32;
    let alpha0 = 0.001 + 0.001 * (reg - 1.0);
    let alpha1 = 0.001 + 0.002 * (reg - 1.0);
    (alpha0.max(0.0), alpha1.max(0.0))
}

/// Get default number of iterations based on voxel size and step size.
///
/// Matches the Julia reference: `max(1000, 3200 / prod(res)^0.42) / step_size^0.6`
pub fn get_default_iterations(res: (f32, f32, f32), step_size: f32) -> usize {
    let prod_res = res.0 * res.1 * res.2;
    let it = (1000.0_f32).max(3200.0 / prod_res.powf(0.42)) / step_size.powf(0.6);
    it.round() as usize
}

/// Bounding box for mask region
#[derive(Clone, Debug)]
struct BoundingBox {
    i_min: usize,
    i_max: usize,
    j_min: usize,
    j_max: usize,
    k_min: usize,
    k_max: usize,
}

impl BoundingBox {
    /// Find minimal bounding box containing the mask with padding
    fn from_mask(mask: &[u8], nx: usize, ny: usize, nz: usize, padding: usize) -> Self {
        let mut i_min = nx;
        let mut i_max = 0;
        let mut j_min = ny;
        let mut j_max = 0;
        let mut k_min = nz;
        let mut k_max = 0;

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    if mask[i + j * nx + k * nx * ny] != 0 {
                        i_min = i_min.min(i);
                        i_max = i_max.max(i);
                        j_min = j_min.min(j);
                        j_max = j_max.max(j);
                        k_min = k_min.min(k);
                        k_max = k_max.max(k);
                    }
                }
            }
        }

        // Add padding and clamp to bounds
        let i_min = i_min.saturating_sub(padding);
        let j_min = j_min.saturating_sub(padding);
        let k_min = k_min.saturating_sub(padding);
        let i_max = (i_max + padding + 1).min(nx);
        let j_max = (j_max + padding + 1).min(ny);
        let k_max = (k_max + padding + 1).min(nz);

        Self { i_min, i_max, j_min, j_max, k_min, k_max }
    }

    fn dims(&self) -> (usize, usize, usize) {
        (self.i_max - self.i_min, self.j_max - self.j_min, self.k_max - self.k_min)
    }

    fn total(&self) -> usize {
        let (bx, by, bz) = self.dims();
        bx * by * bz
    }
}

/// Extract sub-volume from full volume
fn extract_subvolume<T: Copy + Default>(
    full: &[T],
    bbox: &BoundingBox,
    nx: usize, ny: usize, _nz: usize,
) -> Vec<T> {
    let (bx, by, bz) = bbox.dims();
    let mut sub = vec![T::default(); bx * by * bz];

    for k in 0..bz {
        for j in 0..by {
            for i in 0..bx {
                let full_idx = (bbox.i_min + i) + (bbox.j_min + j) * nx + (bbox.k_min + k) * nx * ny;
                let sub_idx = i + j * bx + k * bx * by;
                sub[sub_idx] = full[full_idx];
            }
        }
    }
    sub
}

/// Insert sub-volume back into full volume
fn insert_subvolume<T: Copy>(
    full: &mut [T],
    sub: &[T],
    bbox: &BoundingBox,
    nx: usize, ny: usize, _nz: usize,
) {
    let (bx, by, bz) = bbox.dims();

    for k in 0..bz {
        for j in 0..by {
            for i in 0..bx {
                let full_idx = (bbox.i_min + i) + (bbox.j_min + j) * nx + (bbox.k_min + k) * nx * ny;
                let sub_idx = i + j * bx + k * bx * by;
                full[full_idx] = sub[sub_idx];
            }
        }
    }
}

/// Compute dipole stencil (27-point spatial kernel)
pub fn compute_dipole_stencil(
    res: (f32, f32, f32),
    b0_dir: (f32, f32, f32),
) -> [[[f32; 3]; 3]; 3] {
    let (dx, dy, dz) = res;
    let (bx, by, bz) = b0_dir;

    // Normalize B0 direction
    let b_norm = (bx * bx + by * by + bz * bz).sqrt();
    let bx = bx / b_norm;
    let by = by / b_norm;
    let bz = bz / b_norm;

    let mut stencil = [[[0.0f32; 3]; 3]; 3];

    let hx2 = 1.0 / (dx * dx);
    let hy2 = 1.0 / (dy * dy);
    let hz2 = 1.0 / (dz * dz);
    let factor = 1.0 / 3.0;

    // Center (i=1, j=1, k=1)
    stencil[1][1][1] = -2.0 * (hx2 + hy2 + hz2) * factor
                     + 2.0 * (bx * bx * hx2 + by * by * hy2 + bz * bz * hz2);

    // X neighbors
    stencil[0][1][1] = hx2 * factor - bx * bx * hx2;
    stencil[2][1][1] = hx2 * factor - bx * bx * hx2;

    // Y neighbors
    stencil[1][0][1] = hy2 * factor - by * by * hy2;
    stencil[1][2][1] = hy2 * factor - by * by * hy2;

    // Z neighbors
    stencil[1][1][0] = hz2 * factor - bz * bz * hz2;
    stencil[1][1][2] = hz2 * factor - bz * bz * hz2;

    // Cross terms for oblique B0
    let hxy = 1.0 / (dx * dy);
    let hxz = 1.0 / (dx * dz);
    let hyz = 1.0 / (dy * dz);

    let xy_factor = -bx * by * hxy;
    stencil[0][0][1] = xy_factor;
    stencil[2][2][1] = xy_factor;
    stencil[0][2][1] = -xy_factor;
    stencil[2][0][1] = -xy_factor;

    let xz_factor = -bx * bz * hxz;
    stencil[0][1][0] = xz_factor;
    stencil[2][1][2] = xz_factor;
    stencil[0][1][2] = -xz_factor;
    stencil[2][1][0] = -xz_factor;

    let yz_factor = -by * bz * hyz;
    stencil[1][0][0] = yz_factor;
    stencil[1][2][2] = yz_factor;
    stencil[1][0][2] = -yz_factor;
    stencil[1][2][0] = -yz_factor;

    stencil
}

/// Compute SVD-fitted oblique 27-point dipole stencil via DST-based Poisson solves.
///
/// Matches the Julia reference implementation from QuantitativeSusceptibilityMappingTGV.jl.
/// For each unique centrosymmetric pair of the 26 off-center stencil positions, a Poisson
/// equation is solved using DST-I to obtain Green's function values. These are then fitted
/// to the analytical dipole field via least squares, and resolution-weighted to produce
/// the final stencil.
///
/// Note: The delta function for offsets (di,dj,dk) and (-di,-dj,-dk) are identical
/// (both place +1 at mid±I and -2 at mid), so we only compute 13 unique Poisson
/// solutions and solve a full-rank 13×13 system. This avoids the rank deficiency
/// that would occur with all 26 positions.
pub fn compute_oblique_stencil(
    res: (f32, f32, f32),
    b0_dir: (f32, f32, f32),
) -> [[[f32; 3]; 3]; 3] {
    let n: usize = 64;
    let singularity_cutout = 4.0_f64;
    let mid = n / 2; // 32 (0-indexed center)
    let n2 = n * n;
    let n3 = n * n * n;

    let (dx, dy, dz) = (res.0 as f64, res.1 as f64, res.2 as f64);

    // Normalize B0 direction
    let b_norm = ((b0_dir.0 * b0_dir.0 + b0_dir.1 * b0_dir.1 + b0_dir.2 * b0_dir.2) as f64).sqrt();
    let bdir = (b0_dir.0 as f64 / b_norm, b0_dir.1 as f64 / b_norm, b0_dir.2 as f64 / b_norm);

    // Compute dipole field on 64³ grid
    let mut d = vec![f64::NAN; n3];
    let mut d_mask = vec![false; n3];

    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let x = i as f64 - mid as f64;
                let y = j as f64 - mid as f64;
                let z = k as f64 - mid as f64;
                let r = (x * x + y * y + z * z).sqrt();

                let idx = i + j * n + k * n2;
                if r < singularity_cutout {
                    // d[idx] remains NAN, d_mask[idx] remains false
                } else {
                    let xz = (bdir.0 * x + bdir.1 * y + bdir.2 * z) / r;
                    let kappa = (3.0 * xz * xz - 1.0) / (4.0 * std::f64::consts::PI * r * r * r);
                    d[idx] = kappa;
                    d_mask[idx] = true;
                }
            }
        }
    }

    // DST-I eigenvalue components: coord2[k] = 2 * sin(π*(k+1) / (2*(N+1)))
    let coord2_sq: Vec<f64> = (0..n)
        .map(|k| {
            let v = 2.0 * (std::f64::consts::PI * (k as f64 + 1.0) / (2.0 * (n as f64 + 1.0))).sin();
            v * v
        })
        .collect();

    // 3D eigenvalue grid: coord2_grid[i,j,k] = coord2[i]² + coord2[j]² + coord2[k]²
    let mut coord2_grid = vec![0.0_f64; n3];
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                coord2_grid[i + j * n + k * n2] = coord2_sq[i] + coord2_sq[j] + coord2_sq[k];
            }
        }
    }

    // Pre-compute DST-I sin table
    let sin_table = dst_sin_table(n);
    // Inverse DST-I scale: our DST computes Σ x[n]*sin(...) (no factor of 2),
    // so the inverse is (2/(N+1))^3 for 3D (unlike FFTW which uses 1/(2*(N+1))^3).
    let idst_scale = (2.0 / (n as f64 + 1.0)).powi(3);

    // Enumerate all 26 stencil positions (excluding center), column-major order
    let mut stencil_positions: Vec<(i32, i32, i32)> = Vec::with_capacity(26);
    for dk in -1..=1_i32 {
        for dj in -1..=1_i32 {
            for di in -1..=1_i32 {
                if di == 0 && dj == 0 && dk == 0 {
                    continue;
                }
                stencil_positions.push((di, dj, dk));
            }
        }
    }

    // Collect valid dipole point indices
    let valid_indices: Vec<usize> = d_mask
        .iter()
        .enumerate()
        .filter(|(_, &m)| m)
        .map(|(idx, _)| idx)
        .collect();

    // Exploit centrosymmetry: delta for (di,dj,dk) is identical to (-di,-dj,-dk),
    // so only compute 13 unique Poisson solutions (one per centrosymmetric pair).
    // Pair p corresponds to stencil_positions[p] and stencil_positions[25-p].
    let num_pairs = 13;
    let mut a_rows: Vec<Vec<f64>> = Vec::with_capacity(num_pairs);

    for p in 0..num_pairs {
        let (di, dj, dk) = stencil_positions[p];

        // Create delta function: delta[mid+I] = 1, delta[mid-I] = 1, delta[mid] = -2
        let mut delta = vec![0.0_f64; n3];
        let pi = (mid as i32 + di) as usize;
        let pj = (mid as i32 + dj) as usize;
        let pk = (mid as i32 + dk) as usize;
        let mi = (mid as i32 - di) as usize;
        let mj = (mid as i32 - dj) as usize;
        let mk = (mid as i32 - dk) as usize;

        delta[pi + pj * n + pk * n2] = 1.0;
        delta[mi + mj * n + mk * n2] += 1.0; // += handles coincident positions
        delta[mid + mid * n + mid * n2] = -2.0;

        // Forward 3D DST-I
        let mut fdelta = dst3d(&delta, n, &sin_table);

        // Poisson solve: divide by -coord2_grid
        for idx in 0..n3 {
            fdelta[idx] /= -coord2_grid[idx];
        }

        // Inverse 3D DST-I (= forward DST-I * scale)
        let vdelta = dst3d(&fdelta, n, &sin_table);

        // Extract values at valid dipole points, applying inverse scale
        let row: Vec<f64> = valid_indices
            .iter()
            .map(|&idx| vdelta[idx] * idst_scale)
            .collect();
        a_rows.push(row);
    }

    // Extract dipole values at valid points
    let d_valid: Vec<f64> = valid_indices.iter().map(|&idx| d[idx]).collect();

    // Solve least squares: B^T * y ≈ d_valid via normal equations (13 unknowns, full rank)
    // G = B * B^T (13×13), h = B * d_valid (13×1)
    let mut g = vec![vec![0.0_f64; num_pairs]; num_pairs];
    for i in 0..num_pairs {
        for j in 0..=i {
            let dot: f64 = a_rows[i]
                .iter()
                .zip(a_rows[j].iter())
                .map(|(&a, &b)| a * b)
                .sum();
            g[i][j] = dot;
            g[j][i] = dot;
        }
    }

    let mut h = vec![0.0_f64; num_pairs];
    for i in 0..num_pairs {
        h[i] = a_rows[i]
            .iter()
            .zip(d_valid.iter())
            .map(|(&a, &b)| a * b)
            .sum();
    }

    // Solve G * y = h via eigendecomposition with thresholding (handles rank deficiency
    // from centrosymmetry and other spatial symmetries like x-y equivalence)
    let y = solve_symmetric_pseudoinverse(&g, &h);

    // Assemble stencil: assign each pair's coefficient to both positions
    // y[p] = x[p] + x[25-p] = 2*x[p] (by symmetry), so stencil[I] = 2*x[I] = y[p]
    let mut result = [[[0.0f32; 3]; 3]; 3];
    for p in 0..num_pairs {
        let coeff = y[p] as f32;

        // First member of pair
        let (di, dj, dk) = stencil_positions[p];
        result[(di + 1) as usize][(dj + 1) as usize][(dk + 1) as usize] = coeff;

        // Second member (centrosymmetric partner)
        let (di2, dj2, dk2) = stencil_positions[25 - p];
        result[(di2 + 1) as usize][(dj2 + 1) as usize][(dk2 + 1) as usize] = coeff;
    }

    // Apply resolution weights: w = (i²/dx² + j²/dy² + k²/dz²) / (i² + j² + k²)
    for dk in -1..=1_i32 {
        for dj in -1..=1_i32 {
            for di in -1..=1_i32 {
                if di == 0 && dj == 0 && dk == 0 {
                    continue;
                }
                let si = (di + 1) as usize;
                let sj = (dj + 1) as usize;
                let sk = (dk + 1) as usize;

                let i2 = (di * di) as f64;
                let j2 = (dj * dj) as f64;
                let k2 = (dk * dk) as f64;
                let weight = (i2 / (dx * dx) + j2 / (dy * dy) + k2 / (dz * dz)) / (i2 + j2 + k2);
                result[si][sj][sk] *= weight as f32;
            }
        }
    }

    // Center = -sum(all others)
    let mut total = 0.0f32;
    for dk in 0..3 {
        for dj in 0..3 {
            for di in 0..3 {
                if !(di == 1 && dj == 1 && dk == 1) {
                    total += result[di][dj][dk];
                }
            }
        }
    }
    result[1][1][1] = -total;

    result
}

/// Build sin table for DST-I of given length.
/// sin_table[n][k] = sin(π*(n+1)*(k+1)/(N+1))
fn dst_sin_table(n: usize) -> Vec<Vec<f64>> {
    let scale = std::f64::consts::PI / (n as f64 + 1.0);
    let mut table = vec![vec![0.0_f64; n]; n];
    for j in 0..n {
        for k in 0..n {
            table[j][k] = ((j as f64 + 1.0) * (k as f64 + 1.0) * scale).sin();
        }
    }
    table
}

/// 3D separable DST-I (Type I Discrete Sine Transform).
fn dst3d(input: &[f64], n: usize, sin_table: &[Vec<f64>]) -> Vec<f64> {
    let n2 = n * n;
    let mut data = input.to_vec();
    let mut buf_in = vec![0.0_f64; n];
    let mut buf_out = vec![0.0_f64; n];

    // Transform along x (contiguous dimension)
    for k in 0..n {
        for j in 0..n {
            let base = j * n + k * n2;
            buf_in.copy_from_slice(&data[base..base + n]);
            dst1(&buf_in, sin_table, &mut buf_out);
            data[base..base + n].copy_from_slice(&buf_out);
        }
    }

    // Transform along y (stride = n)
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                buf_in[j] = data[i + j * n + k * n2];
            }
            dst1(&buf_in, sin_table, &mut buf_out);
            for j in 0..n {
                data[i + j * n + k * n2] = buf_out[j];
            }
        }
    }

    // Transform along z (stride = n*n)
    for j in 0..n {
        for i in 0..n {
            for k in 0..n {
                buf_in[k] = data[i + j * n + k * n2];
            }
            dst1(&buf_in, sin_table, &mut buf_out);
            for k in 0..n {
                data[i + j * n + k * n2] = buf_out[k];
            }
        }
    }

    data
}

/// 1D DST-I: X[k] = Σ x[n] * sin(π*(n+1)*(k+1)/(N+1))
fn dst1(input: &[f64], sin_table: &[Vec<f64>], output: &mut [f64]) {
    let n = input.len();
    for k in 0..n {
        let mut sum = 0.0_f64;
        for j in 0..n {
            sum += input[j] * sin_table[j][k];
        }
        output[k] = sum;
    }
}

/// Solve a symmetric positive semi-definite system via eigendecomposition with thresholding.
///
/// Computes the pseudo-inverse solution: x = V * diag(1/λ, thresholded) * V^T * h
/// where G = V Λ V^T is the eigendecomposition. Small eigenvalues (< threshold * max_eigenvalue)
/// are zeroed, matching Julia's SVD-based approach for handling rank-deficient systems.
fn solve_symmetric_pseudoinverse(g: &[Vec<f64>], h: &[f64]) -> Vec<f64> {
    let n = h.len();

    // Copy g for eigendecomposition (Jacobi method modifies in place)
    let mut a: Vec<Vec<f64>> = g.to_vec();
    let mut v = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        v[i][i] = 1.0;
    }

    // Jacobi eigendecomposition for symmetric matrices
    let max_sweeps = 100;
    let tol = 1e-15;

    for _ in 0..max_sweeps {
        // Find largest off-diagonal element
        let mut max_off = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                max_off = max_off.max(a[i][j].abs());
            }
        }
        if max_off < tol {
            break;
        }

        // Sweep all off-diagonal pairs
        for p in 0..n {
            for q in (p + 1)..n {
                if a[p][q].abs() < tol {
                    continue;
                }

                // Compute Givens rotation angle to zero a[p][q]
                let app = a[p][p];
                let aqq = a[q][q];
                let apq = a[p][q];
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Update matrix A' = G^T A G (only rows/cols p, q change)
                // First update off-diagonal rows
                for i in 0..n {
                    if i == p || i == q {
                        continue;
                    }
                    let aip = a[i][p];
                    let aiq = a[i][q];
                    a[i][p] = c * aip - s * aiq;
                    a[p][i] = a[i][p];
                    a[i][q] = s * aip + c * aiq;
                    a[q][i] = a[i][q];
                }

                // Update 2×2 block
                a[p][p] = c * c * app - 2.0 * c * s * apq + s * s * aqq;
                a[q][q] = s * s * app + 2.0 * c * s * apq + c * c * aqq;
                a[p][q] = 0.0;
                a[q][p] = 0.0;

                // Accumulate eigenvectors: V' = V * G
                for i in 0..n {
                    let vip = v[i][p];
                    let viq = v[i][q];
                    v[i][p] = c * vip - s * viq;
                    v[i][q] = s * vip + c * viq;
                }
            }
        }
    }

    // Eigenvalues are on diagonal
    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    let max_eigen = eigenvalues.iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs()));
    let threshold = 1e-10 * max_eigen;

    // Solve: x = V * diag(1/λ_thresholded) * V^T * h
    let mut vt_h = vec![0.0_f64; n];
    for i in 0..n {
        for j in 0..n {
            vt_h[i] += v[j][i] * h[j];
        }
    }

    for i in 0..n {
        if eigenvalues[i].abs() > threshold {
            vt_h[i] /= eigenvalues[i];
        } else {
            vt_h[i] = 0.0;
        }
    }

    let mut x = vec![0.0_f64; n];
    for i in 0..n {
        for j in 0..n {
            x[i] += v[i][j] * vt_h[j];
        }
    }

    x
}

/// Apply dipole stencil to a 3D volume
/// Uses Neumann BC at boundaries (matching Julia's wave_local)
fn apply_stencil(
    output: &mut [f32],
    input: &[f32],
    stencil: &[[[f32; 3]; 3]; 3],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
) {
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let idx = i + j * nx + k * nx * ny;

                if mask[idx] == 0 {
                    output[idx] = 0.0;
                    continue;
                }

                // Julia's wave_local only computes if not at boundary
                // If at boundary, result is 0
                if i == 0 || j == 0 || k == 0 || i + 1 >= nx || j + 1 >= ny || k + 1 >= nz {
                    output[idx] = 0.0;
                    continue;
                }

                let mut sum = 0.0f32;

                for dk in 0..3i32 {
                    for dj in 0..3i32 {
                        for di in 0..3i32 {
                            let ni = (i as i32 + di - 1) as usize;
                            let nj = (j as i32 + dj - 1) as usize;
                            let nk = (k as i32 + dk - 1) as usize;

                            let nidx = ni + nj * nx + nk * nx * ny;
                            sum += stencil[di as usize][dj as usize][dk as usize] * input[nidx];
                        }
                    }
                }

                output[idx] = sum;
            }
        }
    }
}

/// Compute Laplacian of wrapped phase using the DEL method
pub fn compute_phase_laplacian(
    phase: &[f32],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f32, vsy: f32, vsz: f32,
) -> Vec<f32> {
    let n_total = nx * ny * nz;

    let sin_phase: Vec<f32> = phase.iter().map(|&p| p.sin()).collect();
    let cos_phase: Vec<f32> = phase.iter().map(|&p| p.cos()).collect();

    let lap_sin = compute_laplacian(&sin_phase, nx, ny, nz, vsx, vsy, vsz);
    let lap_cos = compute_laplacian(&cos_phase, nx, ny, nz, vsx, vsy, vsz);

    let mut laplacian = vec![0.0f32; n_total];
    for i in 0..n_total {
        if mask[i] != 0 {
            laplacian[i] = lap_sin[i] * cos_phase[i] - lap_cos[i] * sin_phase[i];
        }
    }

    laplacian
}

/// Compute discrete Laplacian of a 3D array
fn compute_laplacian(
    input: &[f32],
    nx: usize, ny: usize, nz: usize,
    vsx: f32, vsy: f32, vsz: f32,
) -> Vec<f32> {
    let n_total = nx * ny * nz;
    let mut output = vec![0.0f32; n_total];

    let hx2 = 1.0 / (vsx * vsx);
    let hy2 = 1.0 / (vsy * vsy);
    let hz2 = 1.0 / (vsz * vsz);
    let center = -2.0 * (hx2 + hy2 + hz2);

    for k in 0..nz {
        let km1 = if k == 0 { 0 } else { k - 1 };
        let kp1 = if k + 1 >= nz { nz - 1 } else { k + 1 };

        for j in 0..ny {
            let jm1 = if j == 0 { 0 } else { j - 1 };
            let jp1 = if j + 1 >= ny { ny - 1 } else { j + 1 };

            for i in 0..nx {
                let im1 = if i == 0 { 0 } else { i - 1 };
                let ip1 = if i + 1 >= nx { nx - 1 } else { i + 1 };

                let idx = i + j * nx + k * nx * ny;

                output[idx] = center * input[idx]
                    + hx2 * (input[im1 + j * nx + k * nx * ny] + input[ip1 + j * nx + k * nx * ny])
                    + hy2 * (input[i + jm1 * nx + k * nx * ny] + input[i + jp1 * nx + k * nx * ny])
                    + hz2 * (input[i + j * nx + km1 * nx * ny] + input[i + j * nx + kp1 * nx * ny]);
            }
        }
    }

    output
}

/// Apply Laplacian with mask
fn apply_laplacian_inplace(
    output: &mut [f32],
    input: &[f32],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f32, vsy: f32, vsz: f32,
) {
    let hx2 = 1.0 / (vsx * vsx);
    let hy2 = 1.0 / (vsy * vsy);
    let hz2 = 1.0 / (vsz * vsz);

    for k in 0..nz {
        let k_offset = k * nx * ny;

        for j in 0..ny {
            let j_offset = j * nx;

            for i in 0..nx {
                let idx = i + j_offset + k_offset;

                if mask[idx] == 0 {
                    output[idx] = 0.0;
                    continue;
                }

                let a0 = input[idx];

                // Neumann BC: use center value at boundary (matching Julia)
                let a_xm = if i > 0 { input[(i - 1) + j_offset + k_offset] } else { a0 };
                let a_xp = if i + 1 < nx { input[(i + 1) + j_offset + k_offset] } else { a0 };
                let a_ym = if j > 0 { input[i + (j - 1) * nx + k_offset] } else { a0 };
                let a_yp = if j + 1 < ny { input[i + (j + 1) * nx + k_offset] } else { a0 };
                let a_zm = if k > 0 { input[i + j_offset + (k - 1) * nx * ny] } else { a0 };
                let a_zp = if k + 1 < nz { input[i + j_offset + (k + 1) * nx * ny] } else { a0 };

                // Laplacian: sum of second derivatives
                output[idx] = hx2 * (a_xm - 2.0 * a0 + a_xp)
                            + hy2 * (a_ym - 2.0 * a0 + a_yp)
                            + hz2 * (a_zm - 2.0 * a0 + a_zp);
            }
        }
    }
}

/// Erode mask by one voxel (6-connected)
pub fn erode_mask(mask: &[u8], nx: usize, ny: usize, nz: usize) -> Vec<u8> {
    let n_total = nx * ny * nz;
    let mut eroded = vec![0u8; n_total];

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let idx = i + j * nx + k * nx * ny;

                if mask[idx] == 0 {
                    continue;
                }

                // Voxels at the boundary of the volume are always eroded
                if i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1 {
                    continue;
                }

                // Check 6-connected neighbors
                let all_neighbors = mask[idx - 1] != 0
                    && mask[idx + 1] != 0
                    && mask[idx - nx] != 0
                    && mask[idx + nx] != 0
                    && mask[idx - nx * ny] != 0
                    && mask[idx + nx * ny] != 0;

                eroded[idx] = if all_neighbors { 1 } else { 0 };
            }
        }
    }

    eroded
}

/// Compute gradient norm squared
#[inline]
fn grad_norm_sq(res: (f32, f32, f32)) -> f32 {
    let (dx, dy, dz) = res;
    4.0 * (1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz))
}

/// Compute the squared spectral norm of the TGV operator matrix via power iteration
///
/// The operator matrix M is:
/// [0,   g,  1 ]
/// [0,   0,  g ]
/// [g², w,  0 ]
///
/// We compute the largest eigenvalue of M^T * M using power iteration.
fn compute_operator_norm_sqr(g: f32, g2: f32, w: f32) -> f32 {
    // M^T * M = [g⁴,   g²w,    0    ]
    //           [g²w,  g²+w²,  g    ]
    //           [0,    g,      g²+1 ]
    let g4 = g2 * g2;
    let g2w = g2 * w;
    let g2_w2 = g2 + w * w;
    let g2_1 = g2 + 1.0;

    // Power iteration to find largest eigenvalue
    let mut v = [1.0f32, 1.0, 1.0];

    for _ in 0..20 {
        // Matrix-vector multiply: y = (M^T * M) * v
        let y0 = g4 * v[0] + g2w * v[1];
        let y1 = g2w * v[0] + g2_w2 * v[1] + g * v[2];
        let y2 = g * v[1] + g2_1 * v[2];

        // Compute norm
        let norm = (y0 * y0 + y1 * y1 + y2 * y2).sqrt();
        if norm < 1e-10 {
            break;
        }

        // Normalize
        v[0] = y0 / norm;
        v[1] = y1 / norm;
        v[2] = y2 / norm;
    }

    // Rayleigh quotient: eigenvalue = v^T * (M^T * M) * v
    let y0 = g4 * v[0] + g2w * v[1];
    let y1 = g2w * v[0] + g2_w2 * v[1] + g * v[2];
    let y2 = g * v[1] + g2_1 * v[2];

    v[0] * y0 + v[1] * y1 + v[2] * y2
}

/// L2 norm of 3-component vector
#[inline]
fn norm3(x: f32, y: f32, z: f32) -> f32 {
    (x * x + y * y + z * z).sqrt()
}

/// Frobenius norm of symmetric 3x3 tensor (6 components)
#[inline]
fn frobenius_norm(sxx: f32, sxy: f32, sxz: f32, syy: f32, syz: f32, szz: f32) -> f32 {
    (sxx * sxx + syy * syy + szz * szz + 2.0 * (sxy * sxy + sxz * sxz + syz * syz)).sqrt()
}

/// L-infinity projection for 3-component vector
#[inline]
fn project_linf3(px: &mut f32, py: &mut f32, pz: &mut f32, threshold: f32) {
    let norm = norm3(*px, *py, *pz);
    if norm > threshold {
        let scale = threshold / norm;
        *px *= scale;
        *py *= scale;
        *pz *= scale;
    }
}

/// L-infinity projection for 6-component symmetric tensor
#[inline]
fn project_linf6(
    qxx: &mut f32, qxy: &mut f32, qxz: &mut f32,
    qyy: &mut f32, qyz: &mut f32, qzz: &mut f32,
    threshold: f32,
) {
    let norm = frobenius_norm(*qxx, *qxy, *qxz, *qyy, *qyz, *qzz);
    if norm > threshold {
        let scale = threshold / norm;
        *qxx *= scale;
        *qxy *= scale;
        *qxz *= scale;
        *qyy *= scale;
        *qyz *= scale;
        *qzz *= scale;
    }
}

/// Compute relative change for convergence check
fn compute_relative_change(chi: &[f32], chi_prev: &[f32], mask: &[u8]) -> f32 {
    let mut diff_sq = 0.0f32;
    let mut norm_sq = 0.0f32;

    for i in 0..chi.len() {
        if mask[i] != 0 {
            let d = chi[i] - chi_prev[i];
            diff_sq += d * d;
            norm_sq += chi[i] * chi[i];
        }
    }

    if norm_sq > 1e-10 {
        (diff_sq / norm_sq).sqrt()
    } else {
        1.0
    }
}

/// Pre-allocated workspace for TGV iteration
struct TgvWorkspace {
    // Primal variables
    chi: Vec<f32>,
    chi_: Vec<f32>,
    chi_prev: Vec<f32>,  // For convergence check
    phi: Vec<f32>,
    phi_: Vec<f32>,
    wx: Vec<f32>,
    wy: Vec<f32>,
    wz: Vec<f32>,
    wx_: Vec<f32>,
    wy_: Vec<f32>,
    wz_: Vec<f32>,

    // Dual variables
    eta: Vec<f32>,
    px: Vec<f32>,
    py: Vec<f32>,
    pz: Vec<f32>,
    qxx: Vec<f32>,
    qxy: Vec<f32>,
    qxz: Vec<f32>,
    qyy: Vec<f32>,
    qyz: Vec<f32>,
    qzz: Vec<f32>,

    // Temporary buffers
    temp1: Vec<f32>,
    temp2: Vec<f32>,
    gx: Vec<f32>,
    gy: Vec<f32>,
    gz: Vec<f32>,

    // Symmetric gradient buffers (reused)
    sxx: Vec<f32>,
    sxy: Vec<f32>,
    sxz: Vec<f32>,
    syy: Vec<f32>,
    syz: Vec<f32>,
    szz: Vec<f32>,

    // Divergence buffers (reused)
    divqx: Vec<f32>,
    divqy: Vec<f32>,
    divqz: Vec<f32>,
}

impl TgvWorkspace {
    fn new(n: usize) -> Self {
        Self {
            chi: vec![0.0; n],
            chi_: vec![0.0; n],
            chi_prev: vec![0.0; n],
            phi: vec![0.0; n],
            phi_: vec![0.0; n],
            wx: vec![0.0; n],
            wy: vec![0.0; n],
            wz: vec![0.0; n],
            wx_: vec![0.0; n],
            wy_: vec![0.0; n],
            wz_: vec![0.0; n],
            eta: vec![0.0; n],
            px: vec![0.0; n],
            py: vec![0.0; n],
            pz: vec![0.0; n],
            qxx: vec![0.0; n],
            qxy: vec![0.0; n],
            qxz: vec![0.0; n],
            qyy: vec![0.0; n],
            qyz: vec![0.0; n],
            qzz: vec![0.0; n],
            temp1: vec![0.0; n],
            temp2: vec![0.0; n],
            gx: vec![0.0; n],
            gy: vec![0.0; n],
            gz: vec![0.0; n],
            sxx: vec![0.0; n],
            sxy: vec![0.0; n],
            sxz: vec![0.0; n],
            syy: vec![0.0; n],
            syz: vec![0.0; n],
            szz: vec![0.0; n],
            divqx: vec![0.0; n],
            divqy: vec![0.0; n],
            divqz: vec![0.0; n],
        }
    }
}

/// Main TGV-QSM reconstruction
pub fn tgv_qsm(
    phase: &[f32],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f32, vsy: f32, vsz: f32,
    params: &TgvParams,
    b0_dir: (f32, f32, f32),
) -> Vec<f32> {
    tgv_qsm_with_progress(phase, mask, nx, ny, nz, vsx, vsy, vsz, params, b0_dir, |_, _| {})
}

/// TGV-QSM with progress callback (optimized version)
pub fn tgv_qsm_with_progress<F>(
    phase: &[f32],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f32, vsy: f32, vsz: f32,
    params: &TgvParams,
    b0_dir: (f32, f32, f32),
    progress: F,
) -> Vec<f32>
where
    F: Fn(usize, usize),
{
    let n_total = nx * ny * nz;
    let res = (vsx, vsy, vsz);

    // Erode mask
    let mut mask_eroded = mask.to_vec();
    for _ in 0..params.erosions {
        mask_eroded = erode_mask(&mask_eroded, nx, ny, nz);
    }

    // Create mask0 (one more erosion for internal computations)
    let mask0 = erode_mask(&mask_eroded, nx, ny, nz);

    // Find bounding box (with padding of 2 voxels)
    let bbox = BoundingBox::from_mask(&mask0, nx, ny, nz, 2);
    let (bx, by, bz) = bbox.dims();
    let b_total = bbox.total();

    // Extract sub-volumes for the bounding box region
    let phase_sub = extract_subvolume(phase, &bbox, nx, ny, nz);
    let mask0_sub = extract_subvolume(&mask0, &bbox, nx, ny, nz);
    let mask_eroded_sub = extract_subvolume(&mask_eroded, &bbox, nx, ny, nz);

    // Compute phase Laplacian on sub-volume
    let mut laplace_phi0 = compute_phase_laplacian(&phase_sub, &mask0_sub, bx, by, bz, vsx, vsy, vsz);

    // Subtract mean within mask
    let (sum, count): (f32, usize) = laplace_phi0.iter().zip(mask0_sub.iter())
        .filter(|(_, &m)| m != 0)
        .fold((0.0, 0), |(s, c), (&v, _)| (s + v, c + 1));
    if count > 0 {
        let mean = sum / count as f32;
        for (v, &m) in laplace_phi0.iter_mut().zip(mask0_sub.iter()) {
            if m != 0 {
                *v -= mean;
            }
        }
    }

    // Compute SVD-fitted oblique dipole stencil (matching Julia reference)
    let stencil = compute_oblique_stencil(res, b0_dir);

    // Compute step sizes for convergence (matching Julia implementation)
    let grad_norm_squared = grad_norm_sq(res);
    let grad_norm = grad_norm_squared.sqrt();
    let wave_norm: f32 = stencil.iter().flatten().flatten().map(|x| x.abs()).sum();
    let norm_sqr = compute_operator_norm_sqr(grad_norm, grad_norm_squared, wave_norm);

    let tau = 1.0 / norm_sqr.sqrt();
    let sigma = tau;

    // step_size is applied selectively in the updates:
    // - eta, phi: use base sigma/tau
    // - p, q, chi, w: use sigma * step_size / tau * step_size
    let sigma_step = sigma * params.step_size;
    let tau_step = tau * params.step_size;

    // Projection thresholds are alpha values (NOT 1/alpha!)
    // Julia: projects p to ||p|| <= alpha1, q to ||q|| <= alpha0
    let alpha = (params.alpha0, params.alpha1);

    // Pre-allocate all workspace buffers
    let mut ws = TgvWorkspace::new(b_total);

    let mut _converged = false;
    let mut final_iter = params.iterations;

    // Main iteration loop
    for iter in 0..params.iterations {
        progress(iter, params.iterations);

        // Convergence check every 100 iterations
        if iter > 0 && iter % 100 == 0 {
            let rel_change = compute_relative_change(&ws.chi, &ws.chi_prev, &mask0_sub);
            if rel_change < params.tol {
                _converged = true;
                final_iter = iter;
                break;
            }
            // Save current chi for next convergence check
            ws.chi_prev.copy_from_slice(&ws.chi);
        }

        // === DUAL UPDATE ===

        // 1. Update eta (data term dual)
        apply_laplacian_inplace(&mut ws.temp1, &ws.phi_, &mask0_sub, bx, by, bz, vsx, vsy, vsz);
        apply_stencil(&mut ws.temp2, &ws.chi_, &stencil, &mask0_sub, bx, by, bz);

        for i in 0..b_total {
            if mask0_sub[i] != 0 {
                ws.eta[i] += sigma * (-ws.temp1[i] + ws.temp2[i] - laplace_phi0[i]);
            }
        }

        // 2. Update p (gradient dual)
        // Julia: p += mask0 * sigma * grad(chi) - mask * sigma * w
        // Compute unmasked gradient first
        crate::utils::gradient::fgrad_inplace_f32(
            &mut ws.gx, &mut ws.gy, &mut ws.gz, &ws.chi_, bx, by, bz, vsx, vsy, vsz
        );

        for i in 0..b_total {
            let in_mask0 = mask0_sub[i] != 0;
            let in_mask = mask_eroded_sub[i] != 0;

            if in_mask0 || in_mask {
                // gradient term scaled by mask0, w term scaled by mask
                let sigmaw0 = if in_mask0 { sigma_step } else { 0.0 };
                let sigmaw = if in_mask { sigma_step } else { 0.0 };

                ws.px[i] += sigmaw0 * ws.gx[i] - sigmaw * ws.wx_[i];
                ws.py[i] += sigmaw0 * ws.gy[i] - sigmaw * ws.wy_[i];
                ws.pz[i] += sigmaw0 * ws.gz[i] - sigmaw * ws.wz_[i];

                project_linf3(&mut ws.px[i], &mut ws.py[i], &mut ws.pz[i], alpha.1);
            }
        }

        // 3. Update q (symmetric gradient dual)
        crate::utils::gradient::symgrad_inplace_f32(
            &mut ws.sxx, &mut ws.sxy, &mut ws.sxz, &mut ws.syy, &mut ws.syz, &mut ws.szz,
            &ws.wx_, &ws.wy_, &ws.wz_, bx, by, bz, vsx, vsy, vsz
        );

        for i in 0..b_total {
            if mask0_sub[i] != 0 {
                ws.qxx[i] += sigma_step * ws.sxx[i];
                ws.qxy[i] += sigma_step * ws.sxy[i];
                ws.qxz[i] += sigma_step * ws.sxz[i];
                ws.qyy[i] += sigma_step * ws.syy[i];
                ws.qyz[i] += sigma_step * ws.syz[i];
                ws.qzz[i] += sigma_step * ws.szz[i];

                project_linf6(
                    &mut ws.qxx[i], &mut ws.qxy[i], &mut ws.qxz[i],
                    &mut ws.qyy[i], &mut ws.qyz[i], &mut ws.qzz[i],
                    alpha.0
                );
            }
        }

        // === VARIABLE SWAP ===
        std::mem::swap(&mut ws.phi, &mut ws.phi_);
        std::mem::swap(&mut ws.chi, &mut ws.chi_);
        std::mem::swap(&mut ws.wx, &mut ws.wx_);
        std::mem::swap(&mut ws.wy, &mut ws.wy_);
        std::mem::swap(&mut ws.wz, &mut ws.wz_);

        // === PRIMAL UPDATE ===

        // 1. Update phi
        for i in 0..b_total {
            ws.temp1[i] = if mask0_sub[i] != 0 { ws.eta[i] } else { 0.0 };
        }
        apply_laplacian_inplace(&mut ws.temp2, &ws.temp1, &mask0_sub, bx, by, bz, vsx, vsy, vsz);

        for i in 0..b_total {
            let denom = 1.0 + if mask_eroded_sub[i] != 0 { tau } else { 0.0 };
            ws.phi[i] = (ws.phi_[i] + tau * ws.temp2[i]) / denom;
        }

        // 2. Update chi
        crate::utils::gradient::bdiv_masked_inplace_f32(
            &mut ws.temp1, &ws.px, &ws.py, &ws.pz, &mask0_sub, bx, by, bz, vsx, vsy, vsz
        );

        for i in 0..b_total {
            ws.gx[i] = if mask0_sub[i] != 0 { ws.eta[i] } else { 0.0 };
        }
        apply_stencil(&mut ws.temp2, &ws.gx, &stencil, &mask0_sub, bx, by, bz);

        for i in 0..b_total {
            ws.chi[i] = ws.chi_[i] + tau_step * (ws.temp1[i] - ws.temp2[i]);
        }

        // 3. Update w
        for i in 0..b_total {
            let m = if mask0_sub[i] != 0 { 1.0 } else { 0.0 };
            ws.sxx[i] = ws.qxx[i] * m;
            ws.sxy[i] = ws.qxy[i] * m;
            ws.sxz[i] = ws.qxz[i] * m;
            ws.syy[i] = ws.qyy[i] * m;
            ws.syz[i] = ws.qyz[i] * m;
            ws.szz[i] = ws.qzz[i] * m;
        }

        crate::utils::gradient::symdiv_inplace_f32(
            &mut ws.divqx, &mut ws.divqy, &mut ws.divqz,
            &ws.sxx, &ws.sxy, &ws.sxz, &ws.syy, &ws.syz, &ws.szz,
            bx, by, bz, vsx, vsy, vsz
        );

        // Julia: w_dest = w; if mask: w_dest += tau*(p + div(mask0*q))
        for i in 0..b_total {
            ws.wx[i] = ws.wx_[i];
            ws.wy[i] = ws.wy_[i];
            ws.wz[i] = ws.wz_[i];
            if mask_eroded_sub[i] != 0 {
                ws.wx[i] += tau_step * (ws.px[i] + ws.divqx[i]);
                ws.wy[i] += tau_step * (ws.py[i] + ws.divqy[i]);
                ws.wz[i] += tau_step * (ws.pz[i] + ws.divqz[i]);
            }
        }

        // === EXTRAGRADIENT UPDATE ===
        for i in 0..b_total {
            ws.phi_[i] = 2.0 * ws.phi[i] - ws.phi_[i];
            ws.chi_[i] = 2.0 * ws.chi[i] - ws.chi_[i];
            ws.wx_[i] = 2.0 * ws.wx[i] - ws.wx_[i];
            ws.wy_[i] = 2.0 * ws.wy[i] - ws.wy_[i];
            ws.wz_[i] = 2.0 * ws.wz[i] - ws.wz_[i];
        }
    }

    progress(final_iter, params.iterations);

    // Scale to susceptibility (ppm)
    let gamma = 42.5781f32;  // Hz/T
    let scale = 1.0 / (2.0 * PI * params.te * params.fieldstrength * gamma);

    // Create full-size result and insert sub-volume
    let mut result = vec![0.0f32; n_total];

    // Scale chi in sub-volume and apply mask
    let mut chi_scaled = vec![0.0f32; b_total];
    for i in 0..b_total {
        if mask_eroded_sub[i] != 0 {
            chi_scaled[i] = ws.chi[i] * scale;
        }
    }

    // Insert back into full volume
    insert_subvolume(&mut result, &chi_scaled, &bbox, nx, ny, nz);

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dipole_stencil() {
        let stencil = compute_dipole_stencil((1.0, 1.0, 1.0), (0.0, 0.0, 1.0));

        let mut sum = 0.0f32;
        for k in 0..3 {
            for j in 0..3 {
                for i in 0..3 {
                    sum += stencil[i][j][k];
                }
            }
        }
        assert!(sum.abs() < 1e-6, "Stencil sum should be ~0, got {}", sum);
    }

    #[test]
    fn test_phase_laplacian() {
        let nx = 4;
        let ny = 4;
        let nz = 4;
        let n = nx * ny * nz;

        let phase = vec![1.0f32; n];
        let mask = vec![1u8; n];

        let lap = compute_phase_laplacian(&phase, &mask, nx, ny, nz, 1.0, 1.0, 1.0);

        let max_val = lap.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(max_val < 1e-5, "Laplacian of constant should be ~0, got max {}", max_val);
    }

    #[test]
    fn test_erode_mask() {
        let nx = 5;
        let ny = 5;
        let nz = 5;

        let mask = vec![1u8; nx * ny * nz];
        let eroded = erode_mask(&mask, nx, ny, nz);

        let center = 2 + 2 * nx + 2 * nx * ny;
        assert_eq!(eroded[center], 1);
        assert_eq!(eroded[0], 0);
    }

    #[test]
    fn test_default_alpha() {
        let (a0, a1) = get_default_alpha(2);
        assert!((a0 - 0.002).abs() < 1e-6);
        assert!((a1 - 0.003).abs() < 1e-6);
    }

    #[test]
    fn test_bounding_box() {
        let nx = 10;
        let ny = 10;
        let nz = 10;
        let mut mask = vec![0u8; nx * ny * nz];

        // Set a small region in the center
        for k in 3..7 {
            for j in 3..7 {
                for i in 3..7 {
                    mask[i + j * nx + k * nx * ny] = 1;
                }
            }
        }

        let bbox = BoundingBox::from_mask(&mask, nx, ny, nz, 1);

        // Should be 3-1=2 to 6+1+1=8 (with padding 1)
        assert_eq!(bbox.i_min, 2);
        assert_eq!(bbox.i_max, 8);
        assert_eq!(bbox.j_min, 2);
        assert_eq!(bbox.j_max, 8);
    }
}

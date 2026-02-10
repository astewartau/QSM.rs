//! Surface Curvature Calculation for QSMART
//!
//! This module computes Gaussian and mean curvatures at the surface of a 3D binary mask,
//! based on the discrete differential geometry approach from:
//! Meyer, M., Desbrun, M., Schröder, P., & Barr, A. H. (2003).
//! "Discrete differential-geometry operators for triangulated 2-manifolds."
//!
//! The curvatures are used in QSMART to weight the spatially-dependent filtering
//! near brain boundaries to reduce artifacts.
//!
//! Uses 2D Delaunay triangulation (via delaunator crate) matching MATLAB's approach:
//! `tri = delaunay(x, y)` - triangulates on x,y coordinates with z as height.

use std::f64::consts::PI;
use delaunator::{triangulate, Point};

/// Result of curvature calculation
pub struct CurvatureResult {
    /// Gaussian curvature at surface voxels (full volume, 0 for non-surface)
    pub gaussian_curvature: Vec<f64>,
    /// Mean curvature at surface voxels (full volume, 0 for non-surface)
    pub mean_curvature: Vec<f64>,
    /// Indices of surface voxels
    pub surface_indices: Vec<usize>,
}

/// Simple 3D point structure
#[derive(Clone, Copy, Debug)]
struct Point3D {
    x: f64,
    y: f64,
    z: f64,
}

impl Point3D {
    fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    fn sub(&self, other: &Point3D) -> Point3D {
        Point3D::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    fn dot(&self, other: &Point3D) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn cross(&self, other: &Point3D) -> Point3D {
        Point3D::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    fn norm(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    fn normalize(&self) -> Point3D {
        let n = self.norm();
        if n > 1e-10 {
            Point3D::new(self.x / n, self.y / n, self.z / n)
        } else {
            Point3D::new(0.0, 0.0, 0.0)
        }
    }

    fn scale(&self, s: f64) -> Point3D {
        Point3D::new(self.x * s, self.y * s, self.z * s)
    }

    fn add(&self, other: &Point3D) -> Point3D {
        Point3D::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

/// Triangle structure
#[derive(Clone, Copy, Debug)]
struct Triangle {
    v0: usize,
    v1: usize,
    v2: usize,
}

/// Extract surface voxels from a binary mask
///
/// Surface voxels are mask voxels with at least one non-mask neighbor (6-connectivity)
fn extract_surface_voxels(
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
) -> Vec<usize> {
    let idx = |i: usize, j: usize, k: usize| i + j * nx + k * nx * ny;

    let mut surface = Vec::new();

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let center_idx = idx(i, j, k);
                if mask[center_idx] == 0 {
                    continue;
                }

                // Check 6-connectivity neighbors
                let mut is_surface = false;

                // Check each neighbor
                if i > 0 && mask[idx(i - 1, j, k)] == 0 { is_surface = true; }
                if !is_surface && i < nx - 1 && mask[idx(i + 1, j, k)] == 0 { is_surface = true; }
                if !is_surface && j > 0 && mask[idx(i, j - 1, k)] == 0 { is_surface = true; }
                if !is_surface && j < ny - 1 && mask[idx(i, j + 1, k)] == 0 { is_surface = true; }
                if !is_surface && k > 0 && mask[idx(i, j, k - 1)] == 0 { is_surface = true; }
                if !is_surface && k < nz - 1 && mask[idx(i, j, k + 1)] == 0 { is_surface = true; }

                // Also check boundary
                if !is_surface && (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1) {
                    is_surface = true;
                }

                if is_surface {
                    surface.push(center_idx);
                }
            }
        }
    }

    surface
}

/// 2D Delaunay triangulation of surface points
///
/// This matches MATLAB's approach: `tri = delaunay(x, y)`
/// Triangulates on x,y coordinates, treating z as a height field.
/// This is appropriate for brain surfaces which are mostly single-valued when
/// viewed from above/below.
fn triangulate_surface(
    points: &[Point3D],
    _nx: usize, _ny: usize, _nz: usize,
) -> Vec<Triangle> {
    if points.len() < 3 {
        return Vec::new();
    }

    // Convert to delaunator's Point format (2D: x, y only)
    let coords: Vec<Point> = points.iter()
        .map(|p| Point { x: p.x, y: p.y })
        .collect();

    // Run 2D Delaunay triangulation
    let result = triangulate(&coords);

    // Convert triangles back to our format
    // delaunator returns triangles as flat array: [t0_v0, t0_v1, t0_v2, t1_v0, t1_v1, t1_v2, ...]
    let mut triangles = Vec::with_capacity(result.triangles.len() / 3);

    for i in (0..result.triangles.len()).step_by(3) {
        let v0 = result.triangles[i];
        let v1 = result.triangles[i + 1];
        let v2 = result.triangles[i + 2];

        // Filter out degenerate triangles (edges too long for voxel surface)
        // Max edge length ~5 voxels to avoid connecting distant surface regions
        let p0 = &points[v0];
        let p1 = &points[v1];
        let p2 = &points[v2];

        let d01 = p0.sub(p1).norm();
        let d12 = p1.sub(p2).norm();
        let d20 = p2.sub(p0).norm();

        let max_edge = 5.0;  // Maximum edge length in voxels
        if d01 <= max_edge && d12 <= max_edge && d20 <= max_edge {
            triangles.push(Triangle { v0, v1, v2 });
        }
    }

    triangles
}

/// Compute Gaussian and mean curvatures using discrete differential geometry
///
/// Based on Meyer et al., "Discrete differential-geometry operators for triangulated 2-manifolds"
fn compute_curvatures_from_mesh(
    points: &[Point3D],
    triangles: &[Triangle],
) -> (Vec<f64>, Vec<f64>) {
    let n_points = points.len();
    let mut gaussian_curvature = vec![0.0f64; n_points];
    let mut mean_curvature = vec![0.0f64; n_points];
    let mut angle_sum = vec![0.0f64; n_points];
    let mut area_mixed = vec![0.0f64; n_points];
    let mut mean_curv_vec = vec![Point3D::new(0.0, 0.0, 0.0); n_points];
    let mut normal_vec = vec![Point3D::new(0.0, 0.0, 0.0); n_points];

    // Process each triangle
    for tri in triangles {
        let p0 = &points[tri.v0];
        let p1 = &points[tri.v1];
        let p2 = &points[tri.v2];

        // Edge vectors
        let e01 = p1.sub(p0); // v0 -> v1
        let e12 = p2.sub(p1); // v1 -> v2
        let e20 = p0.sub(p2); // v2 -> v0

        let l01 = e01.norm();
        let l12 = e12.norm();
        let l20 = e20.norm();

        if l01 < 1e-10 || l12 < 1e-10 || l20 < 1e-10 {
            continue;
        }

        // Triangle area
        let cross = e01.cross(&e12.scale(-1.0));
        let area = 0.5 * cross.norm();
        if area < 1e-10 {
            continue;
        }

        // Triangle normal
        let face_normal = cross.normalize();

        // Angles at each vertex
        let cos_a0 = e01.normalize().dot(&e20.scale(-1.0).normalize());
        let cos_a1 = e01.scale(-1.0).normalize().dot(&e12.normalize());
        let cos_a2 = e12.scale(-1.0).normalize().dot(&e20.normalize());

        let a0 = cos_a0.clamp(-1.0, 1.0).acos();
        let a1 = cos_a1.clamp(-1.0, 1.0).acos();
        let a2 = cos_a2.clamp(-1.0, 1.0).acos();

        // Accumulate angle sums for Gaussian curvature
        angle_sum[tri.v0] += a0;
        angle_sum[tri.v1] += a1;
        angle_sum[tri.v2] += a2;

        // Compute cotangent weights for mean curvature
        let cot_a0 = cos_a0 / (1.0 - cos_a0 * cos_a0).sqrt().max(1e-10);
        let cot_a1 = cos_a1 / (1.0 - cos_a1 * cos_a1).sqrt().max(1e-10);
        let cot_a2 = cos_a2 / (1.0 - cos_a2 * cos_a2).sqrt().max(1e-10);

        // Compute A_mixed for each vertex
        // Check if any angle is obtuse
        let obtuse_0 = a0 > PI / 2.0;
        let obtuse_1 = a1 > PI / 2.0;
        let obtuse_2 = a2 > PI / 2.0;

        // Add contribution to A_mixed for each vertex
        if obtuse_0 {
            area_mixed[tri.v0] += area / 2.0;
        } else if obtuse_1 || obtuse_2 {
            area_mixed[tri.v0] += area / 4.0;
        } else {
            area_mixed[tri.v0] += (l20 * l20 * cot_a1 + l01 * l01 * cot_a2) / 8.0;
        }

        if obtuse_1 {
            area_mixed[tri.v1] += area / 2.0;
        } else if obtuse_0 || obtuse_2 {
            area_mixed[tri.v1] += area / 4.0;
        } else {
            area_mixed[tri.v1] += (l01 * l01 * cot_a2 + l12 * l12 * cot_a0) / 8.0;
        }

        if obtuse_2 {
            area_mixed[tri.v2] += area / 2.0;
        } else if obtuse_0 || obtuse_1 {
            area_mixed[tri.v2] += area / 4.0;
        } else {
            area_mixed[tri.v2] += (l12 * l12 * cot_a0 + l20 * l20 * cot_a1) / 8.0;
        }

        // Mean curvature vector contribution
        // For vertex 0: (cot(a2) * e01 + cot(a1) * e20) / (4 * A_mixed)
        mean_curv_vec[tri.v0] = mean_curv_vec[tri.v0].add(&e01.scale(cot_a2).add(&e20.scale(-cot_a1)));
        mean_curv_vec[tri.v1] = mean_curv_vec[tri.v1].add(&e12.scale(cot_a0).add(&e01.scale(-cot_a2)));
        mean_curv_vec[tri.v2] = mean_curv_vec[tri.v2].add(&e20.scale(cot_a1).add(&e12.scale(-cot_a0)));

        // Accumulate face normal for vertex normal
        let weight = 1.0 / (p0.sub(&points[tri.v0]).norm() + 1.0);
        normal_vec[tri.v0] = normal_vec[tri.v0].add(&face_normal.scale(weight));
        normal_vec[tri.v1] = normal_vec[tri.v1].add(&face_normal.scale(weight));
        normal_vec[tri.v2] = normal_vec[tri.v2].add(&face_normal.scale(weight));
    }

    // Compute final curvature values
    for i in 0..n_points {
        if area_mixed[i] > 1e-10 {
            // Gaussian curvature: K = (2π - Σθ) / A_mixed
            gaussian_curvature[i] = (2.0 * PI - angle_sum[i]) / area_mixed[i];

            // Mean curvature: H = |mean_curv_vec| / (4 * A_mixed)
            let mc_vec = mean_curv_vec[i].scale(0.25 / area_mixed[i]);
            let mc_mag = mc_vec.norm();

            // Determine sign from dot product with normal
            let n_vec = normal_vec[i].normalize();
            let sign = if mc_vec.dot(&n_vec) < 0.0 { -1.0 } else { 1.0 };

            mean_curvature[i] = sign * mc_mag;
        }
    }

    (gaussian_curvature, mean_curvature)
}

/// Calculate proximity maps using curvature at the brain surface
///
/// This is the main entry point matching QSMART's calculate_curvature function.
///
/// # Arguments
/// * `mask` - Binary brain mask
/// * `prox1` - Initial proximity map from Gaussian smoothing
/// * `lower_lim` - Clamping value for proximity (default 0.6)
/// * `curv_constant` - Scaling constant for curvature (default 500)
/// * `sigma` - Kernel size for smoothing curvature
/// * `nx`, `ny`, `nz` - Volume dimensions
///
/// # Returns
/// Modified proximity map incorporating curvature-based edge weighting
pub fn calculate_curvature_proximity(
    mask: &[u8],
    prox1: &[f64],
    lower_lim: f64,
    curv_constant: f64,
    sigma: f64,
    nx: usize, ny: usize, nz: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n_total = nx * ny * nz;

    // Extract surface voxels
    let surface_indices = extract_surface_voxels(mask, nx, ny, nz);

    if surface_indices.is_empty() {
        return (prox1.to_vec(), vec![1.0; n_total]);
    }

    // Convert surface indices to 3D points
    let points: Vec<Point3D> = surface_indices
        .iter()
        .map(|&idx| {
            let i = idx % nx;
            let j = (idx / nx) % ny;
            let k = idx / (nx * ny);
            Point3D::new(i as f64, j as f64, k as f64)
        })
        .collect();

    // Triangulate surface
    let triangles = triangulate_surface(&points, nx, ny, nz);

    // Compute curvatures
    let (gc, _mc) = compute_curvatures_from_mesh(&points, &triangles);

    // Create full curvature volume
    let mut curv_i = vec![1.0f64; n_total];

    // Find max negative curvature for scaling
    let max_neg_gc = gc.iter()
        .filter(|&&v| v < 0.0)
        .map(|&v| v.abs())
        .fold(1.0f64, |a, b| a.max(b));

    // Scale and assign curvature values
    for (point_idx, &vol_idx) in surface_indices.iter().enumerate() {
        let g = gc[point_idx];
        let scaled = if g < 0.0 {
            // Scale negative curvatures by curv_constant
            g / max_neg_gc * curv_constant
        } else {
            // Positive curvatures (convex regions) get 1.0
            1.0
        };
        curv_i[vol_idx] = scaled.max(-curv_constant).min(1.0);
    }

    // Smooth the curvature map
    let sigmas = [sigma, 2.0 * sigma, 2.0 * sigma];
    let prox3 = gaussian_smooth_3d_masked(&curv_i, mask, nx, ny, nz, &sigmas);

    // Clamp prox3 values
    let prox3_clamped: Vec<f64> = prox3.iter().enumerate()
        .map(|(i, &v)| {
            if mask[i] == 0 {
                0.0
            } else if v < 0.5 && v != 0.0 {
                0.5
            } else {
                v
            }
        })
        .collect();

    // Multiply with initial proximity
    let mut prox: Vec<f64> = prox1.iter()
        .zip(prox3_clamped.iter())
        .map(|(&p1, &p3)| p1 * p3)
        .collect();

    // Edge proximity calculation (prox4)
    // Surface voxels get their prox value, dilated region gets 0
    let surface_mask = create_surface_mask(mask, nx, ny, nz);
    let dilated_mask = dilate_mask(mask, nx, ny, nz, 5);

    let mut prox4 = vec![1.0f64; n_total];
    for i in 0..n_total {
        if surface_mask[i] != 0 {
            prox4[i] = prox[i];
        }
        if dilated_mask[i] != 0 && mask[i] == 0 {
            prox4[i] = 0.0;
        }
    }

    // Smooth prox4
    let prox4_smooth = gaussian_smooth_3d_masked(&prox4, &vec![1u8; n_total], nx, ny, nz, &[5.0, 10.0, 10.0]);

    // Clamp proximity values
    for i in 0..n_total {
        if mask[i] == 0 {
            prox[i] = 0.0;
        } else if prox[i] < lower_lim && prox[i] != 0.0 {
            prox[i] = lower_lim;
        }
    }

    // Edge refinement
    for i in 0..n_total {
        prox[i] *= prox4_smooth[i];
    }

    (prox, curv_i)
}

/// Create a surface mask (boundary voxels)
fn create_surface_mask(mask: &[u8], nx: usize, ny: usize, nz: usize) -> Vec<u8> {
    let eroded = erode_mask(mask, nx, ny, nz, 1);
    let mut surface = vec![0u8; mask.len()];

    for i in 0..mask.len() {
        if mask[i] != 0 && eroded[i] == 0 {
            surface[i] = 1;
        }
    }

    surface
}

/// Erode a binary mask using spherical structuring element
fn erode_mask(mask: &[u8], nx: usize, ny: usize, nz: usize, radius: i32) -> Vec<u8> {
    let n_total = nx * ny * nz;
    let mut eroded = vec![0u8; n_total];

    let idx = |i: usize, j: usize, k: usize| i + j * nx + k * nx * ny;

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                if mask[idx(i, j, k)] == 0 {
                    continue;
                }

                let mut all_inside = true;

                'outer: for dz in -radius..=radius {
                    for dy in -radius..=radius {
                        for dx in -radius..=radius {
                            let dist2 = dx * dx + dy * dy + dz * dz;
                            if dist2 > radius * radius {
                                continue;
                            }

                            let ni = i as i32 + dx;
                            let nj = j as i32 + dy;
                            let nk = k as i32 + dz;

                            if ni < 0 || ni >= nx as i32 ||
                               nj < 0 || nj >= ny as i32 ||
                               nk < 0 || nk >= nz as i32 {
                                all_inside = false;
                                break 'outer;
                            }

                            if mask[idx(ni as usize, nj as usize, nk as usize)] == 0 {
                                all_inside = false;
                                break 'outer;
                            }
                        }
                    }
                }

                if all_inside {
                    eroded[idx(i, j, k)] = 1;
                }
            }
        }
    }

    eroded
}

/// Dilate a binary mask using spherical structuring element
fn dilate_mask(mask: &[u8], nx: usize, ny: usize, nz: usize, radius: i32) -> Vec<u8> {
    let n_total = nx * ny * nz;
    let mut dilated = vec![0u8; n_total];

    let idx = |i: usize, j: usize, k: usize| i + j * nx + k * nx * ny;

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                if mask[idx(i, j, k)] != 0 {
                    // Set all neighbors within radius
                    for dz in -radius..=radius {
                        for dy in -radius..=radius {
                            for dx in -radius..=radius {
                                let dist2 = dx * dx + dy * dy + dz * dz;
                                if dist2 > radius * radius {
                                    continue;
                                }

                                let ni = i as i32 + dx;
                                let nj = j as i32 + dy;
                                let nk = k as i32 + dz;

                                if ni >= 0 && ni < nx as i32 &&
                                   nj >= 0 && nj < ny as i32 &&
                                   nk >= 0 && nk < nz as i32 {
                                    dilated[idx(ni as usize, nj as usize, nk as usize)] = 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    dilated
}

/// Morphological closing (dilation followed by erosion)
pub fn morphological_close(mask: &[u8], nx: usize, ny: usize, nz: usize, radius: i32) -> Vec<u8> {
    let dilated = dilate_mask(mask, nx, ny, nz, radius);
    erode_mask(&dilated, nx, ny, nz, radius)
}

/// 3D Gaussian smoothing with anisotropic sigma
fn gaussian_smooth_3d_masked(
    data: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    sigmas: &[f64; 3],
) -> Vec<f64> {
    // Apply separable 1D convolutions
    let smoothed_x = convolve_1d_direction_masked(data, mask, nx, ny, nz, sigmas[0], 'x');
    let smoothed_xy = convolve_1d_direction_masked(&smoothed_x, mask, nx, ny, nz, sigmas[1], 'y');
    let smoothed_xyz = convolve_1d_direction_masked(&smoothed_xy, mask, nx, ny, nz, sigmas[2], 'z');

    // Apply mask
    smoothed_xyz.iter()
        .enumerate()
        .map(|(i, &v)| if mask[i] != 0 { v } else { 0.0 })
        .collect()
}

/// 1D convolution with Gaussian kernel along specified axis
/// Uses replicate padding to match MATLAB's imgaussfilt3 behavior
fn convolve_1d_direction_masked(
    data: &[f64],
    _mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    sigma: f64,
    direction: char,
) -> Vec<f64> {
    if sigma <= 0.0 {
        return data.to_vec();
    }

    let n_total = nx * ny * nz;
    let mut result = vec![0.0f64; n_total];

    // Create 1D Gaussian kernel
    let kernel_radius = (3.0 * sigma).ceil() as i32;
    let kernel_size = 2 * kernel_radius + 1;
    let mut kernel = vec![0.0f64; kernel_size as usize];

    let mut sum = 0.0;
    for i in 0..kernel_size {
        let x = (i - kernel_radius) as f64;
        kernel[i as usize] = (-x * x / (2.0 * sigma * sigma)).exp();
        sum += kernel[i as usize];
    }

    // Normalize
    for k in kernel.iter_mut() {
        *k /= sum;
    }

    let idx = |i: usize, j: usize, k: usize| i + j * nx + k * nx * ny;

    // Helper functions for replicate padding (clamp to valid range)
    let clamp_x = |x: i32| -> usize { x.max(0).min(nx as i32 - 1) as usize };
    let clamp_y = |y: i32| -> usize { y.max(0).min(ny as i32 - 1) as usize };
    let clamp_z = |z: i32| -> usize { z.max(0).min(nz as i32 - 1) as usize };

    match direction {
        'x' => {
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let mut conv_sum = 0.0;

                        for ki in 0..kernel_size {
                            let offset = ki - kernel_radius;
                            let ni = clamp_x(i as i32 + offset);
                            conv_sum += data[idx(ni, j, k)] * kernel[ki as usize];
                        }

                        result[idx(i, j, k)] = conv_sum;
                    }
                }
            }
        }
        'y' => {
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let mut conv_sum = 0.0;

                        for ki in 0..kernel_size {
                            let offset = ki - kernel_radius;
                            let nj = clamp_y(j as i32 + offset);
                            conv_sum += data[idx(i, nj, k)] * kernel[ki as usize];
                        }

                        result[idx(i, j, k)] = conv_sum;
                    }
                }
            }
        }
        'z' => {
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let mut conv_sum = 0.0;

                        for ki in 0..kernel_size {
                            let offset = ki - kernel_radius;
                            let nk = clamp_z(k as i32 + offset);
                            conv_sum += data[idx(i, j, nk)] * kernel[ki as usize];
                        }

                        result[idx(i, j, k)] = conv_sum;
                    }
                }
            }
        }
        _ => panic!("Invalid convolution direction"),
    }

    result
}

/// Simple Gaussian curvature calculation for mask boundary
/// Returns full volume with curvature values at surface voxels
pub fn calculate_gaussian_curvature(
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
) -> CurvatureResult {
    let n_total = nx * ny * nz;

    // Extract surface voxels
    let surface_indices = extract_surface_voxels(mask, nx, ny, nz);

    if surface_indices.is_empty() {
        return CurvatureResult {
            gaussian_curvature: vec![0.0; n_total],
            mean_curvature: vec![0.0; n_total],
            surface_indices: Vec::new(),
        };
    }

    // Convert surface indices to 3D points
    let points: Vec<Point3D> = surface_indices
        .iter()
        .map(|&idx| {
            let i = idx % nx;
            let j = (idx / nx) % ny;
            let k = idx / (nx * ny);
            Point3D::new(i as f64, j as f64, k as f64)
        })
        .collect();

    // Triangulate surface
    let triangles = triangulate_surface(&points, nx, ny, nz);

    // Compute curvatures
    let (gc_points, mc_points) = compute_curvatures_from_mesh(&points, &triangles);

    // Create full volumes
    let mut gaussian_curvature = vec![0.0f64; n_total];
    let mut mean_curvature = vec![0.0f64; n_total];

    for (point_idx, &vol_idx) in surface_indices.iter().enumerate() {
        gaussian_curvature[vol_idx] = gc_points[point_idx];
        mean_curvature[vol_idx] = mc_points[point_idx];
    }

    CurvatureResult {
        gaussian_curvature,
        mean_curvature,
        surface_indices,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_surface_basic() {
        // 3x3x3 cube with center filled
        let mut mask = vec![0u8; 27];
        mask[13] = 1; // Center voxel

        let surface = extract_surface_voxels(&mask, 3, 3, 3);
        assert_eq!(surface.len(), 1);
        assert_eq!(surface[0], 13);
    }

    #[test]
    fn test_erode_mask() {
        // 5x5x5 solid cube
        let mask = vec![1u8; 125];
        let eroded = erode_mask(&mask, 5, 5, 5, 1);

        // Center 3x3x3 should remain
        let count: usize = eroded.iter().map(|&v| v as usize).sum();
        assert!(count > 0);
        assert!(count < 125);
    }

    #[test]
    fn test_dilate_mask() {
        // Single center voxel in 5x5x5
        let mut mask = vec![0u8; 125];
        mask[62] = 1; // Center

        let dilated = dilate_mask(&mask, 5, 5, 5, 1);

        // Should expand to 6-connectivity
        let count: usize = dilated.iter().map(|&v| v as usize).sum();
        assert!(count >= 7); // At least 7 voxels (center + 6 neighbors)
    }
}

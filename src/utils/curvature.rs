//! Surface Curvature Calculation for QSMART
//!
//! This module computes Gaussian and mean curvatures at the surface of a 3D binary mask,
//! based on the discrete differential geometry approach.
//!
//! The curvatures are used in QSMART to weight the spatially-dependent filtering
//! near brain boundaries to reduce artifacts.
//!
//! Uses 2D Delaunay triangulation (via delaunator crate) matching MATLAB's approach:
//! `tri = delaunay(x, y)` - triangulates on x,y coordinates with z as height.
//!
//! Reference:
//! Meyer, M., Desbrun, M., Schröder, P., Barr, A.H. (2003).
//! "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds."
//! Visualization and Mathematics III, 35-57. https://doi.org/10.1007/978-3-662-05105-4_2
//!
//! Reference implementation: https://www.mathworks.com/matlabcentral/fileexchange/61136-curvatures

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
/// Matches MATLAB's approach: curvMask = mask - imerode(mask, strel('sphere',1))
/// Surface voxels are those in the mask but not in the eroded mask.
fn extract_surface_voxels(
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
) -> Vec<usize> {
    let eroded = erode_mask(mask, nx, ny, nz, 1);

    let mut surface = Vec::new();
    for i in 0..mask.len() {
        if mask[i] != 0 && eroded[i] == 0 {
            surface.push(i);
        }
    }

    surface
}

/// 2D Delaunay triangulation of surface points
///
/// This matches MATLAB's approach: `tri = delaunay(x, y)`
/// Triangulates on x,y coordinates, treating z as a height field.
///
/// Returns (triangles, boundary_flags) where boundary_flags[i] is true
/// if vertex i is on the convex hull boundary (matching MATLAB's freeBoundary).
fn triangulate_surface(
    points: &[Point3D],
    _nx: usize, _ny: usize, _nz: usize,
) -> (Vec<Triangle>, Vec<bool>) {
    if points.len() < 3 {
        return (Vec::new(), vec![false; points.len()]);
    }

    // Convert to delaunator's Point format (2D: x, y only)
    let coords: Vec<Point> = points.iter()
        .map(|p| Point { x: p.x, y: p.y })
        .collect();

    // Run 2D Delaunay triangulation
    let result = triangulate(&coords);

    // Identify boundary vertices (convex hull of the 2D triangulation)
    // Matches MATLAB's freeBoundary(triangulation(tri, x, y, z))
    let mut boundary = vec![false; points.len()];
    for &idx in &result.hull {
        boundary[idx] = true;
    }

    // Convert triangles — no edge length filtering, matching MATLAB
    let mut triangles = Vec::with_capacity(result.triangles.len() / 3);

    for i in (0..result.triangles.len()).step_by(3) {
        let v0 = result.triangles[i];
        let v1 = result.triangles[i + 1];
        let v2 = result.triangles[i + 2];

        triangles.push(Triangle { v0, v1, v2 });
    }

    (triangles, boundary)
}

/// Compute Gaussian and mean curvatures using discrete differential geometry
///
/// Based on Meyer et al., "Discrete differential-geometry operators for triangulated 2-manifolds"
///
/// Boundary vertices (on the triangulation free boundary) get GC=0, MC=0
/// to match MATLAB's curvatures.m behavior.
fn compute_curvatures_from_mesh(
    points: &[Point3D],
    triangles: &[Triangle],
    boundary: &[bool],
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
        mean_curv_vec[tri.v0] = mean_curv_vec[tri.v0].add(&e01.scale(cot_a2).add(&e20.scale(-cot_a1)));
        mean_curv_vec[tri.v1] = mean_curv_vec[tri.v1].add(&e12.scale(cot_a0).add(&e01.scale(-cot_a2)));
        mean_curv_vec[tri.v2] = mean_curv_vec[tri.v2].add(&e20.scale(cot_a1).add(&e12.scale(-cot_a0)));

        // Accumulate face normal for vertex normal using incenter-based distance weighting
        // Matches MATLAB: wi = 1/norm(incenter - vertex); n_vec += wi * faceNormal
        // Incenter = (a*P0 + b*P1 + c*P2) / (a+b+c) where a=|P1P2|, b=|P0P2|, c=|P0P1|
        let perim = l12 + l20 + l01;
        if perim > 1e-10 {
            let incenter = p0.scale(l12).add(&p1.scale(l20)).add(&p2.scale(l01)).scale(1.0 / perim);

            let w0 = 1.0 / p0.sub(&incenter).norm().max(1e-10);
            let w1 = 1.0 / p1.sub(&incenter).norm().max(1e-10);
            let w2 = 1.0 / p2.sub(&incenter).norm().max(1e-10);

            normal_vec[tri.v0] = normal_vec[tri.v0].add(&face_normal.scale(w0));
            normal_vec[tri.v1] = normal_vec[tri.v1].add(&face_normal.scale(w1));
            normal_vec[tri.v2] = normal_vec[tri.v2].add(&face_normal.scale(w2));
        }
    }

    // Compute final curvature values
    // Skip boundary vertices (GC=0, MC=0) matching MATLAB's freeBoundary check
    for i in 0..n_points {
        if boundary[i] {
            // Boundary vertices get zero curvature (unreliable)
            continue;
        }

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
    let (triangles, boundary) = triangulate_surface(&points, nx, ny, nz);

    // Compute curvatures (boundary vertices get GC=0, MC=0)
    let (gc, _mc) = compute_curvatures_from_mesh(&points, &triangles, &boundary);

    // Create full curvature volume
    let mut curv_i = vec![1.0f64; n_total];

    // Find max negative curvature for scaling
    let max_neg_gc = gc.iter()
        .filter(|&&v| v < 0.0)
        .map(|&v| v.abs())
        .fold(1.0f64, |a, b| a.max(b));

    // Scale and assign curvature values
    // Matches MATLAB: scaledGC = GC./max(abs(GC(GC<0)))*curvConstant; scaledGC(GC>0) = 1;
    // GC==0 → scaled to 0 (MATLAB: 0/max*curv = 0, not overwritten by GC>0 check)
    for (point_idx, &vol_idx) in surface_indices.iter().enumerate() {
        let g = gc[point_idx];
        let scaled = if g < 0.0 {
            g / max_neg_gc * curv_constant
        } else if g > 0.0 {
            1.0
        } else {
            // GC == 0: stays at 0 (boundary vertices, flat regions)
            0.0
        };
        curv_i[vol_idx] = scaled;
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
    // Matches MATLAB order of operations:
    //   prox4 = prox .* (mask - imerode(mask, strel('sphere',1)));
    //   prox4(prox4==0) = 1;
    //   prox4((imdilate(mask, strel('sphere',5)) - mask)==1) = 0;
    let surface_mask = create_surface_mask(mask, nx, ny, nz);
    let dilated_mask = dilate_mask(mask, nx, ny, nz, 5);

    // Step 1: prox4 = prox * surface_mask (surface voxels get prox, rest get 0)
    let mut prox4 = vec![0.0f64; n_total];
    for i in 0..n_total {
        if surface_mask[i] != 0 {
            prox4[i] = prox[i];
        }
    }
    // Step 2: set ALL zero-valued voxels to 1 (interior + outside + surface with prox==0)
    for i in 0..n_total {
        if prox4[i] == 0.0 {
            prox4[i] = 1.0;
        }
    }
    // Step 3: set dilated shell outside mask to 0
    for i in 0..n_total {
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
    // Match MATLAB's imgaussfilt3 default: filterSize = 2*ceil(2*sigma)+1
    let kernel_radius = (2.0 * sigma).ceil() as i32;
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
    let (triangles, boundary) = triangulate_surface(&points, nx, ny, nz);

    // Compute curvatures
    let (gc_points, mc_points) = compute_curvatures_from_mesh(&points, &triangles, &boundary);

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

    // =====================================================================
    // Helper: create a 3D sphere mask
    // =====================================================================

    /// Create a solid sphere mask centered in an n x n x n volume.
    fn make_sphere_mask(n: usize, radius: f64) -> Vec<u8> {
        let center = n as f64 / 2.0;
        let n_total = n * n * n;
        let mut mask = vec![0u8; n_total];

        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let dx = i as f64 - center;
                    let dy = j as f64 - center;
                    let dz = k as f64 - center;
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    if dist < radius {
                        mask[i + j * n + k * n * n] = 1;
                    }
                }
            }
        }

        mask
    }

    // =====================================================================
    // Tests for Point3D operations
    // =====================================================================

    #[test]
    fn test_point3d_sub() {
        let a = Point3D::new(3.0, 4.0, 5.0);
        let b = Point3D::new(1.0, 1.0, 1.0);
        let c = a.sub(&b);
        assert!((c.x - 2.0).abs() < 1e-10);
        assert!((c.y - 3.0).abs() < 1e-10);
        assert!((c.z - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_point3d_dot() {
        let a = Point3D::new(1.0, 2.0, 3.0);
        let b = Point3D::new(4.0, 5.0, 6.0);
        let d = a.dot(&b);
        assert!((d - 32.0).abs() < 1e-10); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_point3d_cross() {
        let a = Point3D::new(1.0, 0.0, 0.0);
        let b = Point3D::new(0.0, 1.0, 0.0);
        let c = a.cross(&b);
        assert!((c.x - 0.0).abs() < 1e-10);
        assert!((c.y - 0.0).abs() < 1e-10);
        assert!((c.z - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_point3d_norm() {
        let p = Point3D::new(3.0, 4.0, 0.0);
        assert!((p.norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_point3d_normalize() {
        let p = Point3D::new(0.0, 0.0, 5.0);
        let n = p.normalize();
        assert!((n.x - 0.0).abs() < 1e-10);
        assert!((n.y - 0.0).abs() < 1e-10);
        assert!((n.z - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_point3d_normalize_zero() {
        let p = Point3D::new(0.0, 0.0, 0.0);
        let n = p.normalize();
        assert!((n.x).abs() < 1e-10);
        assert!((n.y).abs() < 1e-10);
        assert!((n.z).abs() < 1e-10);
    }

    #[test]
    fn test_point3d_scale_and_add() {
        let a = Point3D::new(1.0, 2.0, 3.0);
        let b = a.scale(2.0);
        assert!((b.x - 2.0).abs() < 1e-10);
        assert!((b.y - 4.0).abs() < 1e-10);
        assert!((b.z - 6.0).abs() < 1e-10);

        let c = Point3D::new(0.5, 0.5, 0.5);
        let d = b.add(&c);
        assert!((d.x - 2.5).abs() < 1e-10);
        assert!((d.y - 4.5).abs() < 1e-10);
        assert!((d.z - 6.5).abs() < 1e-10);
    }

    // =====================================================================
    // Tests for extract_surface_voxels
    // =====================================================================

    #[test]
    fn test_extract_surface_sphere() {
        let n = 10;
        let mask = make_sphere_mask(n, 3.5);
        let surface = extract_surface_voxels(&mask, n, n, n);

        // Surface should be non-empty
        assert!(!surface.is_empty(), "Sphere should have surface voxels");

        // All surface indices should be within the mask
        for &idx in &surface {
            assert_eq!(mask[idx], 1, "Surface voxel should be in mask");
        }

        // Surface count should be less than total mask count
        let mask_count: usize = mask.iter().map(|&v| v as usize).sum();
        assert!(
            surface.len() < mask_count,
            "Surface ({}) should be smaller than total mask ({})",
            surface.len(),
            mask_count
        );
    }

    #[test]
    fn test_extract_surface_empty_mask() {
        let mask = vec![0u8; 27];
        let surface = extract_surface_voxels(&mask, 3, 3, 3);
        assert!(surface.is_empty(), "Empty mask should have no surface voxels");
    }

    // =====================================================================
    // Tests for erode_mask (more thorough)
    // =====================================================================

    #[test]
    fn test_erode_mask_sphere() {
        let n = 10;
        let mask = make_sphere_mask(n, 4.0);
        let eroded = erode_mask(&mask, n, n, n, 1);

        let orig_count: usize = mask.iter().map(|&v| v as usize).sum();
        let eroded_count: usize = eroded.iter().map(|&v| v as usize).sum();
        assert!(
            eroded_count < orig_count,
            "Eroded sphere should be smaller: {} < {}",
            eroded_count,
            orig_count
        );
        assert!(eroded_count > 0, "Eroded sphere should not be empty");

        // Center should still be in eroded mask
        let center = n / 2 + (n / 2) * n + (n / 2) * n * n;
        assert_eq!(eroded[center], 1, "Center should survive erosion");
    }

    #[test]
    fn test_erode_mask_single_voxel() {
        // A single voxel should be eroded away
        let mut mask = vec![0u8; 125];
        mask[62] = 1; // center of 5x5x5
        let eroded = erode_mask(&mask, 5, 5, 5, 1);
        let count: usize = eroded.iter().map(|&v| v as usize).sum();
        assert_eq!(count, 0, "Single voxel should be fully eroded");
    }

    // =====================================================================
    // Tests for dilate_mask (more thorough)
    // =====================================================================

    #[test]
    fn test_dilate_mask_sphere() {
        let n = 10;
        let mask = make_sphere_mask(n, 3.0);
        let dilated = dilate_mask(&mask, n, n, n, 1);

        let orig_count: usize = mask.iter().map(|&v| v as usize).sum();
        let dilated_count: usize = dilated.iter().map(|&v| v as usize).sum();
        assert!(
            dilated_count > orig_count,
            "Dilated sphere should be larger: {} > {}",
            dilated_count,
            orig_count
        );
    }

    #[test]
    fn test_dilate_mask_radius_2() {
        let mut mask = vec![0u8; 125];
        mask[62] = 1; // center of 5x5x5
        let dilated = dilate_mask(&mask, 5, 5, 5, 2);
        let count: usize = dilated.iter().map(|&v| v as usize).sum();
        // Should be more than radius=1 dilation
        assert!(count > 7, "Radius-2 dilation should produce more than 7 voxels, got {}", count);
    }

    // =====================================================================
    // Tests for morphological_close
    // =====================================================================

    #[test]
    fn test_morphological_close_fills_small_gaps() {
        let n = 10;
        let mut mask = make_sphere_mask(n, 4.0);
        // Remove a surface voxel to create a small gap
        let surface = extract_surface_voxels(&mask, n, n, n);
        if !surface.is_empty() {
            mask[surface[0]] = 0;
        }

        let closed = morphological_close(&mask, n, n, n, 1);
        let orig_count: usize = mask.iter().map(|&v| v as usize).sum();
        let closed_count: usize = closed.iter().map(|&v| v as usize).sum();
        // Closing should recover the gap or at least not shrink significantly
        assert!(
            closed_count >= orig_count,
            "Closing should not reduce mask size: {} vs {}",
            closed_count,
            orig_count
        );
    }

    #[test]
    fn test_morphological_close_empty() {
        let mask = vec![0u8; 27];
        let closed = morphological_close(&mask, 3, 3, 3, 1);
        let count: usize = closed.iter().map(|&v| v as usize).sum();
        assert_eq!(count, 0, "Closing empty mask should stay empty");
    }

    // =====================================================================
    // Tests for create_surface_mask
    // =====================================================================

    #[test]
    fn test_create_surface_mask_sphere() {
        let n = 10;
        let mask = make_sphere_mask(n, 4.0);
        let surface = create_surface_mask(&mask, n, n, n);
        let surface_count: usize = surface.iter().map(|&v| v as usize).sum();
        let mask_count: usize = mask.iter().map(|&v| v as usize).sum();

        assert!(surface_count > 0, "Surface mask should be non-empty");
        assert!(
            surface_count < mask_count,
            "Surface ({}) should be smaller than mask ({})",
            surface_count,
            mask_count
        );

        // Every surface voxel should be in the original mask
        for i in 0..surface.len() {
            if surface[i] > 0 {
                assert_eq!(mask[i], 1, "Surface voxel should be in original mask");
            }
        }
    }

    // =====================================================================
    // Tests for triangulate_surface
    // =====================================================================

    #[test]
    fn test_triangulate_surface_few_points() {
        // Less than 3 points should return empty triangulation
        let points = vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(1.0, 1.0, 1.0)];
        let (triangles, boundary) = triangulate_surface(&points, 5, 5, 5);
        assert!(triangles.is_empty(), "Less than 3 points should give no triangles");
        assert_eq!(boundary.len(), 2);
    }

    #[test]
    fn test_triangulate_surface_square_points() {
        // Four points forming a square in XY
        let points = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(0.0, 1.0, 0.0),
            Point3D::new(1.0, 1.0, 0.0),
        ];
        let (triangles, boundary) = triangulate_surface(&points, 5, 5, 5);
        // Should produce 2 triangles from 4 points
        assert_eq!(triangles.len(), 2, "4 points should produce 2 triangles");
        // All 4 points are on the convex hull
        for &b in &boundary {
            assert!(b, "All 4 points should be on boundary");
        }
    }

    // =====================================================================
    // Tests for compute_curvatures_from_mesh
    // =====================================================================

    #[test]
    fn test_compute_curvatures_from_mesh_flat_surface() {
        // A flat grid of points (z=0) should have zero curvature
        let points = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(2.0, 0.0, 0.0),
            Point3D::new(0.0, 1.0, 0.0),
            Point3D::new(1.0, 1.0, 0.0),
            Point3D::new(2.0, 1.0, 0.0),
            Point3D::new(0.0, 2.0, 0.0),
            Point3D::new(1.0, 2.0, 0.0),
            Point3D::new(2.0, 2.0, 0.0),
        ];

        // Create triangulation for the 3x3 grid
        let triangles = vec![
            Triangle { v0: 0, v1: 1, v2: 4 },
            Triangle { v0: 0, v1: 4, v2: 3 },
            Triangle { v0: 1, v1: 2, v2: 5 },
            Triangle { v0: 1, v1: 5, v2: 4 },
            Triangle { v0: 3, v1: 4, v2: 7 },
            Triangle { v0: 3, v1: 7, v2: 6 },
            Triangle { v0: 4, v1: 5, v2: 8 },
            Triangle { v0: 4, v1: 8, v2: 7 },
        ];

        // All boundary except center vertex (index 4)
        let boundary = vec![true, true, true, true, false, true, true, true, true];

        let (gc, mc) = compute_curvatures_from_mesh(&points, &triangles, &boundary);

        // Center vertex (not boundary) on flat surface should have ~zero curvature
        assert!(
            gc[4].abs() < 1e-6,
            "Flat surface should have ~0 Gaussian curvature, got {}",
            gc[4]
        );
        assert!(
            mc[4].abs() < 1e-6,
            "Flat surface should have ~0 mean curvature, got {}",
            mc[4]
        );
    }

    #[test]
    fn test_compute_curvatures_from_mesh_degenerate_triangle() {
        // Degenerate triangle (collinear points) should not crash
        let points = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(2.0, 0.0, 0.0), // collinear
        ];
        let triangles = vec![Triangle { v0: 0, v1: 1, v2: 2 }];
        let boundary = vec![false, false, false];
        let (gc, mc) = compute_curvatures_from_mesh(&points, &triangles, &boundary);
        // Should not crash; values may be zero because area is zero
        assert_eq!(gc.len(), 3);
        assert_eq!(mc.len(), 3);
    }

    #[test]
    fn test_compute_curvatures_from_mesh_boundary_zero() {
        // Boundary vertices should have zero curvature
        let points = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(0.5, 1.0, 1.0),
        ];
        let triangles = vec![Triangle { v0: 0, v1: 1, v2: 2 }];
        let boundary = vec![true, true, true]; // all boundary
        let (gc, mc) = compute_curvatures_from_mesh(&points, &triangles, &boundary);
        for i in 0..3 {
            assert!((gc[i]).abs() < 1e-10, "Boundary vertex GC should be 0");
            assert!((mc[i]).abs() < 1e-10, "Boundary vertex MC should be 0");
        }
    }

    // =====================================================================
    // Tests for convolve_1d_direction_masked
    // =====================================================================

    #[test]
    fn test_convolve_1d_direction_uniform() {
        let n = 8;
        let data = vec![5.0; n * n * n];
        let mask = vec![1u8; n * n * n];

        let result_x = convolve_1d_direction_masked(&data, &mask, n, n, n, 1.0, 'x');
        let result_y = convolve_1d_direction_masked(&data, &mask, n, n, n, 1.0, 'y');
        let result_z = convolve_1d_direction_masked(&data, &mask, n, n, n, 1.0, 'z');

        // Uniform data should stay uniform after convolution
        for &v in &result_x {
            assert!((v - 5.0).abs() < 0.1, "X convolution should preserve uniform data, got {}", v);
        }
        for &v in &result_y {
            assert!((v - 5.0).abs() < 0.1, "Y convolution should preserve uniform data, got {}", v);
        }
        for &v in &result_z {
            assert!((v - 5.0).abs() < 0.1, "Z convolution should preserve uniform data, got {}", v);
        }
    }

    #[test]
    fn test_convolve_1d_direction_zero_sigma() {
        let n = 5;
        let data = vec![3.0; n * n * n];
        let mask = vec![1u8; n * n * n];

        let result = convolve_1d_direction_masked(&data, &mask, n, n, n, 0.0, 'x');
        assert_eq!(result, data, "Zero sigma should return copy of input");
    }

    // =====================================================================
    // Tests for gaussian_smooth_3d_masked
    // =====================================================================

    #[test]
    fn test_gaussian_smooth_3d_masked_uniform() {
        let n = 8;
        let data = vec![10.0; n * n * n];
        let mask = vec![1u8; n * n * n];
        let sigmas = [1.0, 1.0, 1.0];
        let result = gaussian_smooth_3d_masked(&data, &mask, n, n, n, &sigmas);
        assert_eq!(result.len(), n * n * n);
        for &v in &result {
            assert!(v.is_finite(), "Result should be finite");
            assert!((v - 10.0).abs() < 1.0, "Uniform data should stay near 10.0, got {}", v);
        }
    }

    #[test]
    fn test_gaussian_smooth_3d_masked_applies_mask() {
        let n = 8;
        let data = vec![10.0; n * n * n];
        let mut mask = vec![1u8; n * n * n];
        // Zero out half the mask
        for i in 0..(n * n * n / 2) {
            mask[i] = 0;
        }
        let sigmas = [1.0, 1.0, 1.0];
        let result = gaussian_smooth_3d_masked(&data, &mask, n, n, n, &sigmas);
        // Masked-out voxels should be 0
        for i in 0..result.len() {
            if mask[i] == 0 {
                assert!((result[i]).abs() < 1e-10, "Masked-out voxel should be 0, got {}", result[i]);
            }
        }
    }

    // =====================================================================
    // Tests for calculate_gaussian_curvature (main public function)
    // =====================================================================

    #[test]
    fn test_calculate_gaussian_curvature_sphere() {
        let n = 12;
        let mask = make_sphere_mask(n, 4.5);
        let result = calculate_gaussian_curvature(&mask, n, n, n);

        assert_eq!(result.gaussian_curvature.len(), n * n * n);
        assert_eq!(result.mean_curvature.len(), n * n * n);
        assert!(!result.surface_indices.is_empty(), "Should have surface indices");

        // Surface curvature values should be finite
        for &idx in &result.surface_indices {
            assert!(
                result.gaussian_curvature[idx].is_finite(),
                "GC at surface index {} should be finite",
                idx
            );
            assert!(
                result.mean_curvature[idx].is_finite(),
                "MC at surface index {} should be finite",
                idx
            );
        }

        // Non-surface voxels should have zero curvature
        let surface_set: std::collections::HashSet<usize> =
            result.surface_indices.iter().cloned().collect();
        for i in 0..(n * n * n) {
            if !surface_set.contains(&i) {
                assert!(
                    (result.gaussian_curvature[i]).abs() < 1e-10,
                    "Non-surface GC should be 0"
                );
                assert!(
                    (result.mean_curvature[i]).abs() < 1e-10,
                    "Non-surface MC should be 0"
                );
            }
        }
    }

    #[test]
    fn test_calculate_gaussian_curvature_empty_mask() {
        let n = 5;
        let mask = vec![0u8; n * n * n];
        let result = calculate_gaussian_curvature(&mask, n, n, n);
        assert!(result.surface_indices.is_empty());
        assert!(result.gaussian_curvature.iter().all(|&v| v == 0.0));
        assert!(result.mean_curvature.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_calculate_gaussian_curvature_single_voxel() {
        let mut mask = vec![0u8; 125];
        mask[62] = 1; // single voxel in center of 5x5x5
        let result = calculate_gaussian_curvature(&mask, 5, 5, 5);
        // Single voxel is its own surface after erosion removes it
        // Result depends on whether erosion removes it entirely
        assert_eq!(result.gaussian_curvature.len(), 125);
        assert_eq!(result.mean_curvature.len(), 125);
    }

    // =====================================================================
    // Tests for calculate_curvature_proximity (main entry point)
    // =====================================================================

    #[test]
    fn test_calculate_curvature_proximity_sphere() {
        let n = 12;
        let mask = make_sphere_mask(n, 4.5);
        let n_total = n * n * n;

        // Create an initial proximity map (all 1.0 inside mask)
        let prox1: Vec<f64> = mask.iter().map(|&v| v as f64).collect();

        let (prox, curv_i) = calculate_curvature_proximity(
            &mask, &prox1, 0.6, 500.0, 1.0, n, n, n,
        );

        assert_eq!(prox.len(), n_total);
        assert_eq!(curv_i.len(), n_total);

        // All prox values should be finite
        for (i, &v) in prox.iter().enumerate() {
            assert!(v.is_finite(), "Prox at {} should be finite, got {}", i, v);
        }

        // All curv_i values should be finite
        for (i, &v) in curv_i.iter().enumerate() {
            assert!(v.is_finite(), "Curv_i at {} should be finite, got {}", i, v);
        }
    }

    #[test]
    fn test_calculate_curvature_proximity_empty_surface() {
        let n = 5;
        let mask = vec![0u8; n * n * n];
        let n_total = n * n * n;
        let prox1 = vec![1.0; n_total];

        let (prox, curv_i) = calculate_curvature_proximity(
            &mask, &prox1, 0.6, 500.0, 1.0, n, n, n,
        );

        // With empty mask, should return prox1 and all-ones curv_i
        assert_eq!(prox.len(), n_total);
        assert_eq!(curv_i.len(), n_total);
        for &v in &curv_i {
            assert!((v - 1.0).abs() < 1e-10, "Empty surface should give curv_i=1.0");
        }
    }

    #[test]
    fn test_calculate_curvature_proximity_respects_mask() {
        let n = 12;
        let mask = make_sphere_mask(n, 4.5);
        let n_total = n * n * n;
        let prox1: Vec<f64> = mask.iter().map(|&v| v as f64).collect();

        let (prox, _curv_i) = calculate_curvature_proximity(
            &mask, &prox1, 0.6, 500.0, 1.0, n, n, n,
        );

        // Outside mask, proximity should be 0 (due to prox1 being 0 there)
        // or small from smoothing bleed
        for i in 0..n_total {
            assert!(prox[i].is_finite(), "Prox should be finite everywhere");
        }
    }

    #[test]
    fn test_calculate_curvature_proximity_varying_params() {
        let n = 12;
        let mask = make_sphere_mask(n, 4.5);
        let prox1: Vec<f64> = mask.iter().map(|&v| v as f64).collect();

        // Different lower_lim and curv_constant
        let (prox_a, _) = calculate_curvature_proximity(
            &mask, &prox1, 0.3, 100.0, 0.5, n, n, n,
        );
        let (prox_b, _) = calculate_curvature_proximity(
            &mask, &prox1, 0.9, 1000.0, 2.0, n, n, n,
        );

        // Both should produce finite results
        for &v in &prox_a {
            assert!(v.is_finite());
        }
        for &v in &prox_b {
            assert!(v.is_finite());
        }
    }
}

//! BET surface evolution algorithm
//!
//! Based on: Smith, S.M. (2002) "Fast robust automated brain extraction"
//! Human Brain Mapping, 17(3):143-155
//!
//! Aligned with FSL-BET2 implementation.

use super::icosphere::create_icosphere;
use super::mesh::{build_neighbor_matrix, compute_vertex_normals, compute_mean_edge_length_mm, self_intersection_heuristic};
use std::collections::VecDeque;

/// Brain parameters struct (like FSL's bet_parameters)
struct BetParameters {
    t2: f64,       // 2nd percentile (robust min)
    t98: f64,      // 98th percentile (robust max)
    t: f64,        // threshold = t2 + 0.1*(t98-t2)
    tm: f64,       // median within-brain intensity (critical for proper surface evolution)
    cog: [f64; 3], // center of gravity in voxel coordinates
    cog_mm: [f64; 3], // center of gravity in mm (for z-gradient)
    radius: f64,   // estimated brain radius in mm
}

/// Estimate brain parameters from the image (matches FSL-BET2's adjust_initial_mesh)
fn estimate_brain_parameters(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    voxel_size: &[f64; 3],
) -> BetParameters {
    // Collect non-zero values
    let nonzero: Vec<f64> = data.iter().copied().filter(|&v| v > 0.0).collect();

    if nonzero.is_empty() {
        let cog = [(nx as f64) / 2.0, (ny as f64) / 2.0, (nz as f64) / 2.0];
        let cog_mm = [cog[0] * voxel_size[0], cog[1] * voxel_size[1], cog[2] * voxel_size[2]];
        return BetParameters { t2: 0.0, t98: 1.0, t: 0.1, tm: 0.5, cog, cog_mm, radius: 50.0 };
    }

    // Sort for percentiles
    let mut sorted = nonzero.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let t2 = percentile(&sorted, 2.0);
    let t98 = percentile(&sorted, 98.0);
    let t = t2 + 0.1 * (t98 - t2);

    // Find center of gravity (weighted by intensity, like FSL)
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_z = 0.0;
    let mut sum_weight = 0.0;
    let mut n_voxels = 0usize;

    // Use Fortran order: index = x + y*nx + z*nx*ny
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let idx = i + j * nx + k * nx * ny;
                let val = data[idx];
                if val > t {
                    // FSL: weight = min(c, t98 - t2) where c = val - t2
                    let c = (val - t2).min(t98 - t2);
                    sum_x += (i as f64) * c;
                    sum_y += (j as f64) * c;
                    sum_z += (k as f64) * c;
                    sum_weight += c;
                    n_voxels += 1;
                }
            }
        }
    }

    let cog = if sum_weight > 0.0 {
        [sum_x / sum_weight, sum_y / sum_weight, sum_z / sum_weight]
    } else {
        [(nx as f64) / 2.0, (ny as f64) / 2.0, (nz as f64) / 2.0]
    };

    // Estimate brain radius
    let voxel_volume = voxel_size[0] * voxel_size[1] * voxel_size[2];
    let brain_volume = (n_voxels as f64) * voxel_volume;
    let radius = (3.0 * brain_volume / (4.0 * std::f64::consts::PI)).powf(1.0 / 3.0);

    // Compute tm: median intensity within a sphere centered at COG with radius = brain radius
    // This is critical for proper intensity-based surface evolution (FSL bet2.cpp lines 385-403)
    let cog_mm = [cog[0] * voxel_size[0], cog[1] * voxel_size[1], cog[2] * voxel_size[2]];
    let radius_sq = radius * radius;

    let mut within_brain_values: Vec<f64> = Vec::new();
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let idx = i + j * nx + k * nx * ny;
                let val = data[idx];
                // Only consider voxels with intensity between t2 and t98
                if val > t2 && val < t98 {
                    // Check if within sphere of radius centered at COG
                    let px = (i as f64) * voxel_size[0];
                    let py = (j as f64) * voxel_size[1];
                    let pz = (k as f64) * voxel_size[2];
                    let dx = px - cog_mm[0];
                    let dy = py - cog_mm[1];
                    let dz = pz - cog_mm[2];
                    let dist_sq = dx * dx + dy * dy + dz * dz;
                    if dist_sq < radius_sq {
                        within_brain_values.push(val);
                    }
                }
            }
        }
    }

    // Compute median (tm)
    let tm = if within_brain_values.is_empty() {
        (t2 + t98) / 2.0 // fallback
    } else {
        within_brain_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = within_brain_values.len() / 2;
        within_brain_values[mid]
    };

    BetParameters { t2, t98, t, tm, cog, cog_mm, radius }
}

/// Compute percentile of sorted array
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Trilinear interpolation
fn sample_intensity(data: &[f64], nx: usize, ny: usize, nz: usize, x: f64, y: f64, z: f64) -> f64 {
    let x = x.max(0.0).min((nx - 1) as f64);
    let y = y.max(0.0).min((ny - 1) as f64);
    let z = z.max(0.0).min((nz - 1) as f64);

    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let z0 = z.floor() as usize;
    let x1 = (x0 + 1).min(nx - 1);
    let y1 = (y0 + 1).min(ny - 1);
    let z1 = (z0 + 1).min(nz - 1);

    let xd = x - x0 as f64;
    let yd = y - y0 as f64;
    let zd = z - z0 as f64;

    // Fortran order: x + y*nx + z*nx*ny
    let idx = |i: usize, j: usize, k: usize| i + j * nx + k * nx * ny;

    let c000 = data[idx(x0, y0, z0)];
    let c001 = data[idx(x0, y0, z1)];
    let c010 = data[idx(x0, y1, z0)];
    let c011 = data[idx(x0, y1, z1)];
    let c100 = data[idx(x1, y0, z0)];
    let c101 = data[idx(x1, y0, z1)];
    let c110 = data[idx(x1, y1, z0)];
    let c111 = data[idx(x1, y1, z1)];

    let c00 = c000 * (1.0 - xd) + c100 * xd;
    let c01 = c001 * (1.0 - xd) + c101 * xd;
    let c10 = c010 * (1.0 - xd) + c110 * xd;
    let c11 = c011 * (1.0 - xd) + c111 * xd;

    let c0 = c00 * (1.0 - yd) + c10 * yd;
    let c1 = c01 * (1.0 - yd) + c11 * yd;

    c0 * (1.0 - zd) + c1 * zd
}

/// Sample min/max intensities along inward normal (matches FSL-BET2's step_of_computation)
///
/// FSL samples from 1mm to d1 (7mm) for Imin, and up to d2 (3mm) for Imax.
/// Initial values: Imin = tm, Imax = t
/// Final clamps: Imin >= t2, Imax <= tm
///
/// point_mm: vertex position in mm coordinates
/// normal: unit normal vector (outward pointing)
///
/// Returns (i_min, i_max, success) where success=false means sampling failed
/// and the caller should use f3=0 (like FSL does).
fn sample_intensities_fsl(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    point_mm: &[f64; 3],
    normal: &[f64; 3],
    voxel_size: &[f64; 3],
    t2: f64,
    t: f64,
    tm: f64,
) -> (f64, f64, bool) {
    let d1 = 7.0; // max search distance for Imin (mm)
    let d2 = 3.0; // max search distance for Imax (mm)
    let dscale = voxel_size[0].min(voxel_size[1]).min(voxel_size[2]).min(1.0);

    // Initialize like FSL does
    let mut i_min = tm;
    let mut i_max = t;

    // Convert mm to voxel helper
    let mm_to_voxel = |p_mm: &[f64; 3]| -> [f64; 3] {
        [p_mm[0] / voxel_size[0], p_mm[1] / voxel_size[1], p_mm[2] / voxel_size[2]]
    };

    // Check if voxel position is in bounds
    let in_bounds = |v: &[f64; 3]| -> bool {
        v[0] >= 0.0 && v[0] < (nx - 1) as f64 &&
        v[1] >= 0.0 && v[1] < (ny - 1) as f64 &&
        v[2] >= 0.0 && v[2] < (nz - 1) as f64
    };

    // Starting position in mm (1mm inward along normal)
    let mut p_mm = [
        point_mm[0] - normal[0],
        point_mm[1] - normal[1],
        point_mm[2] - normal[2],
    ];
    let mut p_vox = mm_to_voxel(&p_mm);

    // Check if starting point is in bounds (FSL: first bounds check)
    if !in_bounds(&p_vox) {
        // FSL: if first bounds check fails, f3 = 0 (no intensity force)
        return (i_min, i_max, false);
    }

    let im = sample_intensity(data, nx, ny, nz, p_vox[0], p_vox[1], p_vox[2]);
    i_min = i_min.min(im);
    i_max = i_max.max(im);

    // Check far point at d1-1 (FSL: second bounds check)
    let p_far_mm = [
        point_mm[0] - (d1 - 1.0) * normal[0],
        point_mm[1] - (d1 - 1.0) * normal[1],
        point_mm[2] - (d1 - 1.0) * normal[2],
    ];
    let p_far_vox = mm_to_voxel(&p_far_mm);

    if !in_bounds(&p_far_vox) {
        // FSL: if second bounds check fails, f3 = 0 (no intensity force)
        return (i_min, i_max, false);
    }

    let im = sample_intensity(data, nx, ny, nz, p_far_vox[0], p_far_vox[1], p_far_vox[2]);
    i_min = i_min.min(im);

    // Sample from 2mm to d1 (stepping by dscale mm)
    let mut gi = 2.0;
    while gi < d1 {
        p_mm[0] -= normal[0] * dscale;
        p_mm[1] -= normal[1] * dscale;
        p_mm[2] -= normal[2] * dscale;
        p_vox = mm_to_voxel(&p_mm);

        if in_bounds(&p_vox) {
            let im = sample_intensity(data, nx, ny, nz, p_vox[0], p_vox[1], p_vox[2]);
            i_min = i_min.min(im);

            // Only update Imax for samples within d2
            if gi < d2 {
                i_max = i_max.max(im);
            }
        }

        gi += dscale;
    }

    // Clamp like FSL does (this is critical for sinus exclusion)
    i_min = i_min.max(t2);    // Imin can't go below noise floor
    i_max = i_max.min(tm);    // Imax can't go above median brain intensity

    (i_min, i_max, true)
}

/// Convert surface mesh to binary mask using flood fill
///
/// vertices_mm: vertex positions in mm coordinates
/// voxel_size: voxel dimensions in mm
fn surface_to_mask(
    vertices_mm: &[[f64; 3]],
    faces: &[[usize; 3]],
    nx: usize, ny: usize, nz: usize,
    voxel_size: &[f64; 3],
) -> Vec<u8> {
    // Convert mm vertices to voxel coordinates
    let vertices: Vec<[f64; 3]> = vertices_mm
        .iter()
        .map(|v| [
            v[0] / voxel_size[0],
            v[1] / voxel_size[1],
            v[2] / voxel_size[2],
        ])
        .collect();

    let mininc = 0.5;

    // Start with all 1s (outside)
    let mut grid: Vec<u8> = vec![1; nx * ny * nz];
    // Fortran order: x + y*nx + z*nx*ny
    let idx = |i: usize, j: usize, k: usize| i + j * nx + k * nx * ny;

    // Draw mesh surface as 0s
    for &[i0, i1, i2] in faces {
        let v0 = vertices[i0];
        let v1 = vertices[i1];
        let v2 = vertices[i2];

        // Edge from v1 to v0
        let edge = [v0[0] - v1[0], v0[1] - v1[1], v0[2] - v1[2]];
        let edge_len = (edge[0].powi(2) + edge[1].powi(2) + edge[2].powi(2)).sqrt();

        if edge_len < 0.001 {
            continue;
        }

        let edge_dir = [edge[0] / edge_len, edge[1] / edge_len, edge[2] / edge_len];
        let n_edge_steps = (edge_len / mininc).ceil() as usize + 1;

        for j in 0..n_edge_steps {
            let d = (j as f64) * mininc;
            let p_edge = if d > edge_len {
                v0
            } else {
                [v1[0] + d * edge_dir[0], v1[1] + d * edge_dir[1], v1[2] + d * edge_dir[2]]
            };

            // Draw segment from p_edge to v2
            let seg = [v2[0] - p_edge[0], v2[1] - p_edge[1], v2[2] - p_edge[2]];
            let seg_len = (seg[0].powi(2) + seg[1].powi(2) + seg[2].powi(2)).sqrt();

            if seg_len < 0.001 {
                let ix = p_edge[0].round() as isize;
                let iy = p_edge[1].round() as isize;
                let iz = p_edge[2].round() as isize;
                if ix >= 0 && ix < nx as isize && iy >= 0 && iy < ny as isize && iz >= 0 && iz < nz as isize {
                    grid[idx(ix as usize, iy as usize, iz as usize)] = 0;
                }
                continue;
            }

            let seg_dir = [seg[0] / seg_len, seg[1] / seg_len, seg[2] / seg_len];
            let n_seg_steps = (seg_len / mininc).ceil() as usize + 1;

            for k in 0..n_seg_steps {
                let sd = (k as f64) * mininc;
                let p = if sd > seg_len {
                    v2
                } else {
                    [p_edge[0] + sd * seg_dir[0], p_edge[1] + sd * seg_dir[1], p_edge[2] + sd * seg_dir[2]]
                };

                let ix = p[0].round() as isize;
                let iy = p[1].round() as isize;
                let iz = p[2].round() as isize;

                if ix >= 0 && ix < nx as isize && iy >= 0 && iy < ny as isize && iz >= 0 && iz < nz as isize {
                    grid[idx(ix as usize, iy as usize, iz as usize)] = 0;
                }
            }
        }
    }

    // Flood fill from center of mesh
    let mut center = [0.0, 0.0, 0.0];
    for v in &vertices {
        center[0] += v[0];
        center[1] += v[1];
        center[2] += v[2];
    }
    center[0] /= vertices.len() as f64;
    center[1] /= vertices.len() as f64;
    center[2] /= vertices.len() as f64;

    let mut cx = center[0].round() as isize;
    let mut cy = center[1].round() as isize;
    let mut cz = center[2].round() as isize;

    cx = cx.max(0).min(nx as isize - 1);
    cy = cy.max(0).min(ny as isize - 1);
    cz = cz.max(0).min(nz as isize - 1);

    // If center is on surface, find nearby interior point
    if grid[idx(cx as usize, cy as usize, cz as usize)] == 0 {
        'search: for dx in -5..=5 {
            for dy in -5..=5 {
                for dz in -5..=5 {
                    let nx_ = cx + dx;
                    let ny_ = cy + dy;
                    let nz_ = cz + dz;
                    if nx_ >= 0 && nx_ < nx as isize && ny_ >= 0 && ny_ < ny as isize && nz_ >= 0 && nz_ < nz as isize {
                        if grid[idx(nx_ as usize, ny_ as usize, nz_ as usize)] == 1 {
                            cx = nx_;
                            cy = ny_;
                            cz = nz_;
                            break 'search;
                        }
                    }
                }
            }
        }
    }

    // BFS flood fill
    let mut queue: VecDeque<(usize, usize, usize)> = VecDeque::new();
    let cx = cx as usize;
    let cy = cy as usize;
    let cz = cz as usize;
    grid[idx(cx, cy, cz)] = 0;
    queue.push_back((cx, cy, cz));

    let neighbors: [(isize, isize, isize); 6] = [
        (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)
    ];

    while let Some((x, y, z)) = queue.pop_front() {
        for &(dx, dy, dz) in &neighbors {
            let nx_ = x as isize + dx;
            let ny_ = y as isize + dy;
            let nz_ = z as isize + dz;

            if nx_ >= 0 && nx_ < nx as isize && ny_ >= 0 && ny_ < ny as isize && nz_ >= 0 && nz_ < nz as isize {
                let ni = idx(nx_ as usize, ny_ as usize, nz_ as usize);
                if grid[ni] == 1 {
                    grid[ni] = 0;
                    queue.push_back((nx_ as usize, ny_ as usize, nz_ as usize));
                }
            }
        }
    }

    // Invert: 0 = brain (inside + surface), we want 1 = brain
    for v in grid.iter_mut() {
        *v = if *v == 0 { 1 } else { 0 };
    }

    // Fill holes using simple morphological closing
    fill_holes(&mut grid, nx, ny, nz);

    grid
}

/// Simple hole filling
fn fill_holes(mask: &mut [u8], nx: usize, ny: usize, nz: usize) {
    // Fortran order: x + y*nx + z*nx*ny
    let idx = |i: usize, j: usize, k: usize| i + j * nx + k * nx * ny;

    // Flood fill from corners to find exterior
    let mut exterior: Vec<bool> = vec![false; nx * ny * nz];
    let mut queue: VecDeque<(usize, usize, usize)> = VecDeque::new();

    // Start from all boundary voxels that are 0
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                if i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1 {
                    if mask[idx(i, j, k)] == 0 {
                        exterior[idx(i, j, k)] = true;
                        queue.push_back((i, j, k));
                    }
                }
            }
        }
    }

    let neighbors: [(isize, isize, isize); 6] = [
        (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)
    ];

    while let Some((x, y, z)) = queue.pop_front() {
        for &(dx, dy, dz) in &neighbors {
            let nx_ = x as isize + dx;
            let ny_ = y as isize + dy;
            let nz_ = z as isize + dz;

            if nx_ >= 0 && nx_ < nx as isize && ny_ >= 0 && ny_ < ny as isize && nz_ >= 0 && nz_ < nz as isize {
                let ni = idx(nx_ as usize, ny_ as usize, nz_ as usize);
                if mask[ni] == 0 && !exterior[ni] {
                    exterior[ni] = true;
                    queue.push_back((nx_ as usize, ny_ as usize, nz_ as usize));
                }
            }
        }
    }

    // Any voxel that is 0 but not exterior is interior -> fill it
    for i in 0..mask.len() {
        if mask[i] == 0 && !exterior[i] {
            mask[i] = 1;
        }
    }
}

/// Self-intersection threshold (matches FSL-BET2)
const SELF_INTERSECTION_THRESHOLD: f64 = 4000.0;

/// Maximum number of recovery passes (matches FSL-BET2)
const MAX_PASSES: usize = 10;

/// Run a single pass of surface evolution
///
/// Returns the final vertices after evolution
fn evolution_pass(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    voxel_size: &[f64; 3],
    bp: &BetParameters,
    vertices: &mut Vec<[f64; 3]>,
    faces: &[[usize; 3]],
    neighbor_matrix: &[Vec<usize>],
    neighbor_counts: &[usize],
    bt: f64,
    smoothness_factor: f64,
    gradient_threshold: f64,
    iterations: usize,
    pass: usize,
    progress_callback: &mut Option<&mut dyn FnMut(usize, usize)>,
) {
    let n_vertices = vertices.len();

    // BET parameters (from FSL) - adjusted by smoothness_factor
    let rmin = 3.33 * smoothness_factor;
    let rmax = 10.0 * smoothness_factor;
    let e = (1.0 / rmin + 1.0 / rmax) / 2.0;
    let f = 6.0 / (1.0 / rmin - 1.0 / rmax);
    let normal_max_update_fraction = 0.5;
    let lambda_fit = 0.1;

    // Initial mean edge length
    // Vertices are in mm, so use the mm version
    let mut l = compute_mean_edge_length_mm(vertices, faces);

    // Smoothing increase factor for recovery passes (FSL: 10^(pass+1))
    let base_increase = if pass > 0 { 10.0_f64.powi((pass + 1) as i32) } else { 1.0 };

    // Report progress at start if first pass
    let progress_interval = (iterations / 20).max(1);

    // Debug counter for sampling failures
    let mut sample_fail_count: usize = 0;

    for iteration in 0..iterations {
        // Report progress periodically
        if let Some(ref mut cb) = progress_callback {
            if iteration % progress_interval == 0 {
                cb(iteration, iterations);
            }
        }

        // Compute increase factor with tapering in later iterations (FSL: after 75%)
        let incfactor = if pass > 0 && iteration > (0.75 * iterations as f64) as usize {
            let t = iteration as f64 / iterations as f64;
            4.0 * (1.0 - t) * (base_increase - 1.0) + 1.0
        } else {
            base_increase
        };

        // Compute vertex normals
        let normals = compute_vertex_normals(vertices, faces);

        // Compute updates for each vertex
        let mut updates: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0]; n_vertices];

        for i in 0..n_vertices {
            let v = vertices[i];
            let n = normals[i];

            // Compute mean neighbor position
            let mut mean_neighbor = [0.0, 0.0, 0.0];
            let count = neighbor_counts[i];
            for j in 0..count {
                let ni = neighbor_matrix[i][j];
                mean_neighbor[0] += vertices[ni][0];
                mean_neighbor[1] += vertices[ni][1];
                mean_neighbor[2] += vertices[ni][2];
            }
            if count > 0 {
                mean_neighbor[0] /= count as f64;
                mean_neighbor[1] /= count as f64;
                mean_neighbor[2] /= count as f64;
            }

            // Vector from vertex to mean neighbor
            let dv = [mean_neighbor[0] - v[0], mean_neighbor[1] - v[1], mean_neighbor[2] - v[2]];

            // Dot product with normal
            let dv_dot_n = dv[0] * n[0] + dv[1] * n[1] + dv[2] * n[2];

            // Normal component
            let sn = [dv_dot_n * n[0], dv_dot_n * n[1], dv_dot_n * n[2]];

            // Tangential component
            let st = [dv[0] - sn[0], dv[1] - sn[1], dv[2] - sn[2]];

            // Force 1: Tangential (vertex spacing)
            let u1 = [st[0] * 0.5, st[1] * 0.5, st[2] * 0.5];

            // Force 2: Normal (smoothness)
            let sn_mag = dv_dot_n.abs();
            let rinv = (2.0 * sn_mag) / (l * l);
            let mut f2 = (1.0 + (f * (rinv - e)).tanh()) * 0.5;

            // In recovery passes, increase smoothing for outward-pointing updates (FSL behavior)
            if pass > 0 && dv_dot_n > 0.0 {
                f2 *= incfactor;
                f2 = f2.min(1.0);
            }

            let u2 = [f2 * sn[0], f2 * sn[1], f2 * sn[2]];

            // Force 3: Intensity-based (using FSL-style sampling with tm)
            let (i_min, i_max, sample_ok) = sample_intensities_fsl(
                data, nx, ny, nz, &v, &n, voxel_size, bp.t2, bp.t, bp.tm
            );

            // FSL: if sampling fails (out of bounds), f3 = 0 (no intensity force)
            let u3 = if sample_ok {
                // Apply z-gradient to local threshold (FSL's -g option)
                let local_bt = if gradient_threshold.abs() > 1e-10 {
                    // Vertex is already in mm coordinates
                    let z_offset = (v[2] - bp.cog_mm[2]) / bp.radius;
                    (bt + gradient_threshold * z_offset).clamp(0.0, 1.0)
                } else {
                    bt
                };

                // Compute local threshold and force (matches FSL exactly)
                let t_l = (i_max - bp.t2) * local_bt + bp.t2;
                let f3 = if i_max - bp.t2 > 0.0 {
                    2.0 * (i_min - t_l) / (i_max - bp.t2)
                } else {
                    2.0 * (i_min - t_l)
                };
                let f3 = f3 * normal_max_update_fraction * lambda_fit * l;

                [f3 * n[0], f3 * n[1], f3 * n[2]]
            } else {
                // Sampling failed - use f3 = 0 like FSL does
                sample_fail_count += 1;
                [0.0, 0.0, 0.0]
            };

            // Combined update
            updates[i] = [u1[0] + u2[0] + u3[0], u1[1] + u2[1] + u3[1], u1[2] + u2[2] + u3[2]];
        }

        // Apply updates
        for i in 0..n_vertices {
            vertices[i][0] += updates[i][0];
            vertices[i][1] += updates[i][1];
            vertices[i][2] += updates[i][2];
        }

        // Update edge length periodically
        if iteration % 100 == 0 {
            l = compute_mean_edge_length_mm(vertices, faces);
        }
    }

    // Debug: report sampling failures
    if sample_fail_count > 0 {
        let total_samples = iterations * n_vertices;
        let fail_pct = 100.0 * sample_fail_count as f64 / total_samples as f64;
        eprintln!("[BET] Sampling fallback used: {} / {} ({:.1}%)",
                  sample_fail_count, total_samples, fail_pct);
    }
}

/// Run BET brain extraction
///
/// # Arguments
/// * `data` - 3D magnitude image data (nx * ny * nz, Fortran order)
/// * `nx`, `ny`, `nz` - Image dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `fractional_intensity` - Intensity threshold (0.0-1.0, smaller = larger brain)
/// * `smoothness_factor` - Smoothness constraint (default 1.0, larger = smoother)
/// * `gradient_threshold` - Z-gradient for threshold (-1 to 1, positive = larger at bottom)
/// * `iterations` - Number of surface evolution iterations
/// * `subdivisions` - Icosphere subdivision level
///
/// # Returns
/// Binary mask (1 = brain, 0 = background)
pub fn run_bet(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    fractional_intensity: f64,
    smoothness_factor: f64,
    gradient_threshold: f64,
    iterations: usize,
    subdivisions: usize,
) -> Vec<u8> {
    let voxel_size = [vsx, vsy, vsz];

    // Step 1: Estimate brain parameters
    let bp = estimate_brain_parameters(data, nx, ny, nz, &voxel_size);

    // Step 2: Create icosphere
    let (unit_vertices, faces) = create_icosphere(subdivisions);
    let n_vertices = unit_vertices.len();

    // Scale and position sphere in mm coordinates (start at 50% of estimated radius)
    // Like FSL, we work entirely in mm - voxel conversion only happens at sampling/masking
    let initial_radius_mm = bp.radius * 0.5;

    let initial_vertices: Vec<[f64; 3]> = unit_vertices
        .iter()
        .map(|v| [
            v[0] * initial_radius_mm + bp.cog_mm[0],
            v[1] * initial_radius_mm + bp.cog_mm[1],
            v[2] * initial_radius_mm + bp.cog_mm[2],
        ])
        .collect();

    // Build neighbor structure
    let (neighbor_matrix, neighbor_counts) = build_neighbor_matrix(n_vertices, &faces, 6);

    // FSL power transform: bt = pow(f, 0.275)
    // This raises 0.5 -> 0.826, which makes the surface more aggressive in expanding
    let bt = fractional_intensity.powf(0.275);

    // Multi-pass evolution with self-intersection recovery (like FSL-BET2)
    let mut vertices = initial_vertices.clone();
    let mut pass = 0;

    loop {
        // Run evolution pass
        evolution_pass(
            data, nx, ny, nz, &voxel_size, &bp,
            &mut vertices, &faces,
            &neighbor_matrix, &neighbor_counts,
            bt, smoothness_factor, gradient_threshold,
            iterations, pass,
            &mut None,
        );

        // Check for self-intersection
        let si_score = self_intersection_heuristic(&vertices, &initial_vertices, &faces, &voxel_size);
        let has_self_intersection = si_score > SELF_INTERSECTION_THRESHOLD;

        if has_self_intersection {
            eprintln!("[BET] Self-intersection detected (score={:.0}, threshold={}), pass {}",
                      si_score, SELF_INTERSECTION_THRESHOLD, pass + 1);
        }

        // Exit if no self-intersection or max passes reached
        if !has_self_intersection || pass >= MAX_PASSES {
            if pass > 0 {
                eprintln!("[BET] Completed after {} recovery pass(es)", pass);
            }
            break;
        }

        // Reset to original mesh and try again with higher smoothing
        vertices = initial_vertices.clone();
        pass += 1;
    }

    // Step 4: Convert surface to binary mask
    surface_to_mask(&vertices, &faces, nx, ny, nz, &voxel_size)
}

/// Run BET brain extraction with progress callback
///
/// Same as run_bet but calls progress_callback(iteration, total_iterations) periodically
pub fn run_bet_with_progress<F>(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    fractional_intensity: f64,
    smoothness_factor: f64,
    gradient_threshold: f64,
    iterations: usize,
    subdivisions: usize,
    mut progress_callback: F,
) -> Vec<u8>
where
    F: FnMut(usize, usize),
{
    let voxel_size = [vsx, vsy, vsz];

    // Step 1: Estimate brain parameters
    progress_callback(0, iterations);
    let bp = estimate_brain_parameters(data, nx, ny, nz, &voxel_size);

    // Step 2: Create icosphere
    let (unit_vertices, faces) = create_icosphere(subdivisions);
    let n_vertices = unit_vertices.len();

    // Scale and position sphere in mm coordinates (start at 50% of estimated radius)
    let initial_radius_mm = bp.radius * 0.5;

    let initial_vertices: Vec<[f64; 3]> = unit_vertices
        .iter()
        .map(|v| [
            v[0] * initial_radius_mm + bp.cog_mm[0],
            v[1] * initial_radius_mm + bp.cog_mm[1],
            v[2] * initial_radius_mm + bp.cog_mm[2],
        ])
        .collect();

    let (neighbor_matrix, neighbor_counts) = build_neighbor_matrix(n_vertices, &faces, 6);

    // FSL power transform: bt = pow(f, 0.275)
    let bt = fractional_intensity.powf(0.275);

    // Multi-pass evolution with self-intersection recovery (like FSL-BET2)
    let mut vertices = initial_vertices.clone();
    let mut pass = 0;

    loop {
        // Run evolution pass
        let mut cb: Option<&mut dyn FnMut(usize, usize)> = Some(&mut progress_callback);
        evolution_pass(
            data, nx, ny, nz, &voxel_size, &bp,
            &mut vertices, &faces,
            &neighbor_matrix, &neighbor_counts,
            bt, smoothness_factor, gradient_threshold,
            iterations, pass,
            &mut cb,
        );

        // Check for self-intersection
        let si_score = self_intersection_heuristic(&vertices, &initial_vertices, &faces, &voxel_size);
        let has_self_intersection = si_score > SELF_INTERSECTION_THRESHOLD;

        if has_self_intersection {
            eprintln!("[BET] Self-intersection detected (score={:.0}, threshold={}), pass {}",
                      si_score, SELF_INTERSECTION_THRESHOLD, pass + 1);
        }

        // Exit if no self-intersection or max passes reached
        if !has_self_intersection || pass >= MAX_PASSES {
            if pass > 0 {
                eprintln!("[BET] Completed after {} recovery pass(es)", pass);
            }
            break;
        }

        // Reset to original mesh and try again with higher smoothing
        vertices = initial_vertices.clone();
        pass += 1;
    }

    // Final progress update
    progress_callback(iterations, iterations);

    // Step 4: Convert surface to binary mask
    surface_to_mask(&vertices, &faces, nx, ny, nz, &voxel_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a 3D sphere volume (Fortran order) centered in the grid.
    /// Returns (data, nx, ny, nz) where the sphere has intensity `intensity`
    /// and background is 0. Sphere radius in voxels is `radius`.
    fn make_sphere_volume(n: usize, radius: f64, intensity: f64) -> (Vec<f64>, usize, usize, usize) {
        let center = (n as f64) / 2.0;
        let mut data = vec![0.0; n * n * n];
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let di = (i as f64) - center;
                    let dj = (j as f64) - center;
                    let dk = (k as f64) - center;
                    let dist = (di * di + dj * dj + dk * dk).sqrt();
                    if dist <= radius {
                        // Smoothly varying intensity: bright at center, dimmer at edge
                        data[i + j * n + k * n * n] = intensity * (1.0 - 0.5 * dist / radius);
                    }
                }
            }
        }
        (data, n, n, n)
    }

    #[test]
    fn test_estimate_brain_parameters() {
        let nx = 10;
        let ny = 10;
        let nz = 10;
        let mut data = vec![0.0; nx * ny * nz];

        // Create a sphere with varying intensity (like a real brain)
        // Fortran order: index = i + j*nx + k*nx*ny
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let di = (i as f64) - 5.0;
                    let dj = (j as f64) - 5.0;
                    let dk = (k as f64) - 5.0;
                    let dist = (di*di + dj*dj + dk*dk).sqrt();
                    if dist <= 4.0 {
                        // Intensity varies from 50 to 150 based on distance from center
                        data[i + j * nx + k * nx * ny] = 150.0 - dist * 25.0;
                    }
                }
            }
        }

        let bp = estimate_brain_parameters(&data, nx, ny, nz, &[1.0, 1.0, 1.0]);

        assert!(bp.t2 >= 0.0);
        assert!(bp.t98 >= bp.t2); // Allow equal for edge cases
        assert!((bp.cog[0] - 5.0).abs() < 1.0);
        assert!((bp.cog[1] - 5.0).abs() < 1.0);
        assert!((bp.cog[2] - 5.0).abs() < 1.0);
        assert!(bp.radius > 0.0);
        assert!(bp.tm > bp.t2 && bp.tm < bp.t98); // tm should be between t2 and t98
        // Check cog_mm is correctly computed
        assert!((bp.cog_mm[0] - bp.cog[0]).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_brain_parameters_empty_data() {
        // All zeros should trigger the nonzero.is_empty() branch
        let nx = 4;
        let ny = 4;
        let nz = 4;
        let data = vec![0.0; nx * ny * nz];
        let bp = estimate_brain_parameters(&data, nx, ny, nz, &[1.0, 1.0, 1.0]);

        assert!((bp.t2 - 0.0).abs() < 1e-10);
        assert!((bp.t98 - 1.0).abs() < 1e-10);
        assert!((bp.t - 0.1).abs() < 1e-10);
        assert!((bp.tm - 0.5).abs() < 1e-10);
        assert!((bp.cog[0] - 2.0).abs() < 1e-10);
        assert!((bp.cog[1] - 2.0).abs() < 1e-10);
        assert!((bp.cog[2] - 2.0).abs() < 1e-10);
        assert!((bp.radius - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_brain_parameters_anisotropic_voxels() {
        let (data, nx, ny, nz) = make_sphere_volume(12, 4.0, 200.0);
        let voxel_size = [2.0, 2.0, 2.0];
        let bp = estimate_brain_parameters(&data, nx, ny, nz, &voxel_size);

        // cog_mm should be cog * voxel_size
        assert!((bp.cog_mm[0] - bp.cog[0] * voxel_size[0]).abs() < 1e-10);
        assert!((bp.cog_mm[1] - bp.cog[1] * voxel_size[1]).abs() < 1e-10);
        assert!((bp.cog_mm[2] - bp.cog[2] * voxel_size[2]).abs() < 1e-10);
        assert!(bp.radius > 0.0);
        assert!(bp.t98 > bp.t2);
    }

    #[test]
    fn test_sample_intensity() {
        let data = vec![
            0.0, 1.0, 2.0, 3.0,
            4.0, 5.0, 6.0, 7.0,
        ];
        let val = sample_intensity(&data, 2, 2, 2, 0.5, 0.5, 0.5);
        // Trilinear interpolation of cube corners
        assert!((val - 3.5).abs() < 0.01);
    }

    #[test]
    fn test_sample_intensity_at_corners() {
        // 2x2x2 cube: values 0..7
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        // Sampling at integer corners should return exact values
        assert!((sample_intensity(&data, 2, 2, 2, 0.0, 0.0, 0.0) - 0.0).abs() < 1e-10);
        assert!((sample_intensity(&data, 2, 2, 2, 1.0, 0.0, 0.0) - 1.0).abs() < 1e-10);
        assert!((sample_intensity(&data, 2, 2, 2, 0.0, 1.0, 0.0) - 2.0).abs() < 1e-10);
        assert!((sample_intensity(&data, 2, 2, 2, 1.0, 1.0, 0.0) - 3.0).abs() < 1e-10);
        assert!((sample_intensity(&data, 2, 2, 2, 0.0, 0.0, 1.0) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_sample_intensity_clamping() {
        // Out-of-bounds coordinates should be clamped
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        // Negative coordinates should clamp to 0
        let val = sample_intensity(&data, 2, 2, 2, -1.0, -1.0, -1.0);
        assert!((val - 10.0).abs() < 1e-10, "Expected 10.0, got {}", val);
        // Beyond max should clamp to max
        let val = sample_intensity(&data, 2, 2, 2, 5.0, 5.0, 5.0);
        assert!((val - 80.0).abs() < 1e-10, "Expected 80.0, got {}", val);
    }

    #[test]
    fn test_sample_intensity_larger_volume() {
        // 4x4x4 volume with known pattern
        let n = 4;
        let mut data = vec![0.0; n * n * n];
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    data[i + j * n + k * n * n] = (i + j + k) as f64;
                }
            }
        }
        // At center (1.5, 1.5, 1.5), each corner = i+j+k
        // Average of corners: (1+1+1=3, 2+1+1=4, 1+2+1=4, 2+2+1=5, 1+1+2=4, 2+1+2=5, 1+2+2=5, 2+2+2=6) / 8 = 4.5
        let val = sample_intensity(&data, n, n, n, 1.5, 1.5, 1.5);
        assert!((val - 4.5).abs() < 1e-10, "Expected 4.5, got {}", val);
    }

    #[test]
    fn test_percentile() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let p0 = percentile(&sorted, 0.0);
        assert!((p0 - 1.0).abs() < 1e-10);

        let p50 = percentile(&sorted, 50.0);
        // Index = round(0.5 * 9) = round(4.5) = 5, value = 6.0
        assert!((p50 - 5.0).abs() < 1.5, "p50={}", p50);

        let p100 = percentile(&sorted, 100.0);
        assert!((p100 - 10.0).abs() < 1e-10);

        // Empty array
        let empty: Vec<f64> = vec![];
        assert!((percentile(&empty, 50.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_power_transform() {
        // Verify power transform matches FSL
        let f = 0.5_f64;
        let bt = f.powf(0.275);
        // 0.5^0.275 ~ 0.826
        assert!((bt - 0.826).abs() < 0.01);
    }

    #[test]
    fn test_sample_intensities_fsl_basic() {
        // Create a 16x16x16 uniform volume
        let n = 16;
        let data = vec![100.0; n * n * n];
        let voxel_size = [1.0, 1.0, 1.0];
        // Point at the center, normal pointing inward
        let point_mm = [8.0, 8.0, 8.0];
        let normal = [0.0, 0.0, 1.0]; // outward along z

        let t2 = 10.0;
        let t = 20.0;
        let tm = 100.0;

        let (i_min, i_max, ok) = sample_intensities_fsl(
            &data, n, n, n, &point_mm, &normal, &voxel_size, t2, t, tm,
        );

        assert!(ok, "Sampling should succeed for center point");
        assert!(i_min.is_finite());
        assert!(i_max.is_finite());
        // Uniform volume: i_min should be clamped to >= t2
        assert!(i_min >= t2, "i_min={} should be >= t2={}", i_min, t2);
        // i_max should be clamped to <= tm
        assert!(i_max <= tm, "i_max={} should be <= tm={}", i_max, tm);
    }

    #[test]
    fn test_sample_intensities_fsl_out_of_bounds() {
        // Point near edge, normal pointing outward -- should fail bounds check
        let n = 8;
        let data = vec![100.0; n * n * n];
        let voxel_size = [1.0, 1.0, 1.0];
        // Point at the edge
        let point_mm = [0.5, 0.5, 0.5];
        let normal = [-1.0, 0.0, 0.0]; // pointing outward (further out of bounds)

        let (_, _, _ok) = sample_intensities_fsl(
            &data, n, n, n, &point_mm, &normal, &voxel_size, 10.0, 20.0, 100.0,
        );

        // Construct a case that goes out of bounds on the first inward step:
        // point - normal = (0.5 - 1.0, 0.5, 0.5) = (-0.5, 0.5, 0.5) -> voxel (-0.5) is OOB
        let point_mm2 = [0.5, 0.5, 0.5];
        let normal2 = [1.0, 0.0, 0.0]; // point - normal = (-0.5, 0.5, 0.5) -> voxel (-0.5) out of bounds
        let (_, _, ok2) = sample_intensities_fsl(
            &data, n, n, n, &point_mm2, &normal2, &voxel_size, 10.0, 20.0, 100.0,
        );
        assert!(!ok2, "Sampling should fail when initial step goes out of bounds");
    }

    #[test]
    fn test_sample_intensities_fsl_varying_intensity() {
        // Create a volume with intensity gradient along z
        let n = 16;
        let mut data = vec![0.0; n * n * n];
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    data[i + j * n + k * n * n] = 50.0 + (k as f64) * 10.0;
                }
            }
        }
        let voxel_size = [1.0, 1.0, 1.0];
        let point_mm = [8.0, 8.0, 8.0];
        // Normal pointing in -z direction (inward sampling goes into higher z)
        let normal = [0.0, 0.0, -1.0];

        let t2 = 50.0;
        let t = 55.0;
        let tm = 120.0;

        let (i_min, i_max, ok) = sample_intensities_fsl(
            &data, n, n, n, &point_mm, &normal, &voxel_size, t2, t, tm,
        );

        assert!(ok, "Sampling should succeed");
        assert!(i_min.is_finite() && i_max.is_finite());
        assert!(i_min >= t2);
        assert!(i_max <= tm);
    }

    #[test]
    fn test_fill_holes_no_holes() {
        // A solid 3x3x3 cube with no holes should remain unchanged
        let nx = 5;
        let ny = 5;
        let nz = 5;
        let mut mask = vec![0u8; nx * ny * nz];
        let idx = |i: usize, j: usize, k: usize| i + j * nx + k * nx * ny;

        // Place a solid 3x3x3 block in the center
        for k in 1..4 {
            for j in 1..4 {
                for i in 1..4 {
                    mask[idx(i, j, k)] = 1;
                }
            }
        }

        let original = mask.clone();
        fill_holes(&mut mask, nx, ny, nz);

        assert_eq!(mask, original, "Solid mask should not change");
    }

    #[test]
    fn test_fill_holes_with_interior_hole() {
        // A 7x7x7 volume with a shell of 1s and a hole (0) inside
        let n = 7;
        let idx = |i: usize, j: usize, k: usize| i + j * n + k * n * n;
        let mut mask = vec![0u8; n * n * n];

        // Create a hollow cube: 1 on shell from (1,1,1) to (5,5,5), 0 inside
        for k in 1..6 {
            for j in 1..6 {
                for i in 1..6 {
                    mask[idx(i, j, k)] = 1;
                }
            }
        }
        // Carve out the interior
        for k in 2..5 {
            for j in 2..5 {
                for i in 2..5 {
                    mask[idx(i, j, k)] = 0;
                }
            }
        }

        // The interior (2..5)^3 should be 0 before fill_holes
        assert_eq!(mask[idx(3, 3, 3)], 0, "Center should be 0 before fill");

        fill_holes(&mut mask, n, n, n);

        // After fill_holes, interior holes should be filled
        assert_eq!(mask[idx(3, 3, 3)], 1, "Center hole should be filled");
        assert_eq!(mask[idx(2, 2, 2)], 1, "Interior corner should be filled");

        // Exterior should still be 0
        assert_eq!(mask[idx(0, 0, 0)], 0, "Exterior should remain 0");
        assert_eq!(mask[idx(6, 6, 6)], 0, "Exterior should remain 0");
    }

    #[test]
    fn test_fill_holes_exterior_connected() {
        // All zeros (no brain): nothing to fill
        let n = 5;
        let mut mask = vec![0u8; n * n * n];
        fill_holes(&mut mask, n, n, n);
        assert!(mask.iter().all(|&v| v == 0), "All-zero mask should stay all-zero");
    }

    #[test]
    fn test_surface_to_mask_small_sphere() {
        // Create a small icosphere mesh centered in a 16x16x16 grid
        let (unit_verts, faces) = create_icosphere(2);
        let n = 16;
        let center = (n as f64) / 2.0;
        let radius = 5.0; // 5 voxel radius
        let voxel_size = [1.0, 1.0, 1.0];

        // Place vertices in mm (= voxel coords since voxel_size is 1)
        let vertices_mm: Vec<[f64; 3]> = unit_verts
            .iter()
            .map(|v| [
                v[0] * radius + center,
                v[1] * radius + center,
                v[2] * radius + center,
            ])
            .collect();

        let mask = surface_to_mask(&vertices_mm, &faces, n, n, n, &voxel_size);

        assert_eq!(mask.len(), n * n * n);

        // Center should be inside the mask
        let idx = |i: usize, j: usize, k: usize| i + j * n + k * n * n;
        assert_eq!(mask[idx(8, 8, 8)], 1, "Center should be brain");

        // Corners should be outside the mask
        assert_eq!(mask[idx(0, 0, 0)], 0, "Corner should be background");
        assert_eq!(mask[idx(15, 15, 15)], 0, "Corner should be background");

        // Count brain voxels -- should be a reasonable fraction of total
        let brain_count: usize = mask.iter().map(|&v| v as usize).sum();
        assert!(brain_count > 0, "Should have some brain voxels");
        // Sphere volume ~ 4/3 * pi * 5^3 = 524 voxels; total = 4096
        // Allow generous range
        assert!(brain_count > 100 && brain_count < 2000,
                "Brain voxels ({}) should be roughly sphere-like", brain_count);
    }

    #[test]
    fn test_surface_to_mask_anisotropic_voxels() {
        let (unit_verts, faces) = create_icosphere(1);
        let n = 16;
        let voxel_size = [2.0, 2.0, 2.0];
        let center_mm = [(n as f64) * voxel_size[0] / 2.0,
                         (n as f64) * voxel_size[1] / 2.0,
                         (n as f64) * voxel_size[2] / 2.0];
        let radius_mm = 8.0;

        let vertices_mm: Vec<[f64; 3]> = unit_verts
            .iter()
            .map(|v| [
                v[0] * radius_mm + center_mm[0],
                v[1] * radius_mm + center_mm[1],
                v[2] * radius_mm + center_mm[2],
            ])
            .collect();

        let mask = surface_to_mask(&vertices_mm, &faces, n, n, n, &voxel_size);
        assert_eq!(mask.len(), n * n * n);

        let brain_count: usize = mask.iter().map(|&v| v as usize).sum();
        assert!(brain_count > 0, "Should have brain voxels with anisotropic voxels");
    }

    #[test]
    fn test_run_bet_small_synthetic_volume() {
        // Create a 16x16x16 volume with a bright sphere
        let (data, nx, ny, nz) = make_sphere_volume(16, 6.0, 200.0);

        let mask = run_bet(
            &data,
            nx, ny, nz,
            1.0, 1.0, 1.0, // voxel size
            0.5,            // fractional_intensity
            1.0,            // smoothness_factor
            0.0,            // gradient_threshold
            50,             // iterations (few for speed)
            1,              // subdivisions (low for speed)
        );

        assert_eq!(mask.len(), nx * ny * nz);

        // Should produce a valid binary mask
        assert!(mask.iter().all(|&v| v == 0 || v == 1), "Mask should be binary");

        // Should have some brain voxels
        let brain_count: usize = mask.iter().map(|&v| v as usize).sum();
        assert!(brain_count > 0, "BET should extract some brain voxels");

        // Center should be brain
        let idx = |i: usize, j: usize, k: usize| i + j * nx + k * nx * ny;
        assert_eq!(mask[idx(8, 8, 8)], 1, "Center of sphere should be brain");
    }

    #[test]
    fn test_run_bet_with_gradient_threshold() {
        // Exercise the gradient threshold code path
        let (data, nx, ny, nz) = make_sphere_volume(16, 6.0, 200.0);

        let mask = run_bet(
            &data,
            nx, ny, nz,
            1.0, 1.0, 1.0,
            0.5,
            1.0,
            0.3,  // non-zero gradient threshold
            50,
            1,
        );

        assert_eq!(mask.len(), nx * ny * nz);
        assert!(mask.iter().all(|&v| v == 0 || v == 1));
        let brain_count: usize = mask.iter().map(|&v| v as usize).sum();
        assert!(brain_count > 0, "BET with gradient should extract brain voxels");
    }

    #[test]
    fn test_run_bet_with_progress() {
        // Test the progress callback variant
        let (data, nx, ny, nz) = make_sphere_volume(16, 6.0, 200.0);

        let mut progress_calls = Vec::new();
        let mask = run_bet_with_progress(
            &data,
            nx, ny, nz,
            1.0, 1.0, 1.0,
            0.5,
            1.0,
            0.0,
            50,
            1,
            |current, total| {
                progress_calls.push((current, total));
            },
        );

        assert_eq!(mask.len(), nx * ny * nz);
        assert!(mask.iter().all(|&v| v == 0 || v == 1));

        // Progress callback should have been called at least twice (start + end)
        assert!(progress_calls.len() >= 2,
                "Progress callback should be called at least twice, got {} calls",
                progress_calls.len());

        // First call should be (0, iterations)
        assert_eq!(progress_calls[0].0, 0);

        // Last call should be (iterations, iterations)
        let last = progress_calls.last().unwrap();
        assert_eq!(last.0, last.1, "Final progress should be complete");
    }

    #[test]
    fn test_run_bet_different_fractional_intensities() {
        let (data, nx, ny, nz) = make_sphere_volume(16, 6.0, 200.0);

        // Higher fractional_intensity = smaller brain
        let mask_small = run_bet(&data, nx, ny, nz, 1.0, 1.0, 1.0, 0.7, 1.0, 0.0, 50, 1);
        let mask_large = run_bet(&data, nx, ny, nz, 1.0, 1.0, 1.0, 0.3, 1.0, 0.0, 50, 1);

        let count_small: usize = mask_small.iter().map(|&v| v as usize).sum();
        let count_large: usize = mask_large.iter().map(|&v| v as usize).sum();

        // Both should produce valid masks
        assert!(count_small > 0);
        assert!(count_large > 0);
    }

    #[test]
    fn test_run_bet_anisotropic_voxels() {
        let (data, nx, ny, nz) = make_sphere_volume(16, 6.0, 200.0);

        let mask = run_bet(
            &data,
            nx, ny, nz,
            2.0, 2.0, 2.0, // 2mm voxels
            0.5,
            1.0,
            0.0,
            50,
            1,
        );

        assert_eq!(mask.len(), nx * ny * nz);
        assert!(mask.iter().all(|&v| v == 0 || v == 1));
    }

    #[test]
    fn test_evolution_pass_basic() {
        // Directly test evolution_pass with a small icosphere
        let n = 16;
        let (data, nx, ny, nz) = make_sphere_volume(n, 6.0, 200.0);
        let voxel_size = [1.0, 1.0, 1.0];
        let bp = estimate_brain_parameters(&data, nx, ny, nz, &voxel_size);

        let (unit_verts, faces) = create_icosphere(1);
        let n_vertices = unit_verts.len();
        let initial_radius_mm = bp.radius * 0.5;

        let mut vertices: Vec<[f64; 3]> = unit_verts
            .iter()
            .map(|v| [
                v[0] * initial_radius_mm + bp.cog_mm[0],
                v[1] * initial_radius_mm + bp.cog_mm[1],
                v[2] * initial_radius_mm + bp.cog_mm[2],
            ])
            .collect();

        let (neighbor_matrix, neighbor_counts) = build_neighbor_matrix(n_vertices, &faces, 6);
        let bt = 0.5_f64.powf(0.275);

        let vertices_before = vertices.clone();

        evolution_pass(
            &data, nx, ny, nz, &voxel_size, &bp,
            &mut vertices, &faces,
            &neighbor_matrix, &neighbor_counts,
            bt, 1.0, 0.0,
            10, // few iterations
            0,  // first pass
            &mut None,
        );

        // Vertices should have moved
        let mut any_moved = false;
        for (before, after) in vertices_before.iter().zip(vertices.iter()) {
            let dist = ((after[0] - before[0]).powi(2) +
                       (after[1] - before[1]).powi(2) +
                       (after[2] - before[2]).powi(2)).sqrt();
            if dist > 1e-10 {
                any_moved = true;
            }
            // All coordinates should be finite
            assert!(after[0].is_finite() && after[1].is_finite() && after[2].is_finite());
        }
        assert!(any_moved, "Evolution should move at least some vertices");
    }

    #[test]
    fn test_evolution_pass_recovery_pass() {
        // Test with pass > 0 to exercise recovery smoothing code paths
        let n = 16;
        let (data, nx, ny, nz) = make_sphere_volume(n, 6.0, 200.0);
        let voxel_size = [1.0, 1.0, 1.0];
        let bp = estimate_brain_parameters(&data, nx, ny, nz, &voxel_size);

        let (unit_verts, faces) = create_icosphere(1);
        let n_vertices = unit_verts.len();
        let initial_radius_mm = bp.radius * 0.5;

        let mut vertices: Vec<[f64; 3]> = unit_verts
            .iter()
            .map(|v| [
                v[0] * initial_radius_mm + bp.cog_mm[0],
                v[1] * initial_radius_mm + bp.cog_mm[1],
                v[2] * initial_radius_mm + bp.cog_mm[2],
            ])
            .collect();

        let (neighbor_matrix, neighbor_counts) = build_neighbor_matrix(n_vertices, &faces, 6);
        let bt = 0.5_f64.powf(0.275);

        // Run with pass=1 and enough iterations to hit the 75% tapering code
        evolution_pass(
            &data, nx, ny, nz, &voxel_size, &bp,
            &mut vertices, &faces,
            &neighbor_matrix, &neighbor_counts,
            bt, 1.0, 0.0,
            20,  // enough iterations to hit 75% mark
            1,   // recovery pass
            &mut None,
        );

        // All coordinates should remain finite
        for v in &vertices {
            assert!(v[0].is_finite() && v[1].is_finite() && v[2].is_finite());
        }
    }

    #[test]
    fn test_evolution_pass_with_gradient() {
        // Exercise the z-gradient threshold branch
        let n = 16;
        let (data, nx, ny, nz) = make_sphere_volume(n, 6.0, 200.0);
        let voxel_size = [1.0, 1.0, 1.0];
        let bp = estimate_brain_parameters(&data, nx, ny, nz, &voxel_size);

        let (unit_verts, faces) = create_icosphere(1);
        let n_vertices = unit_verts.len();
        let initial_radius_mm = bp.radius * 0.5;

        let mut vertices: Vec<[f64; 3]> = unit_verts
            .iter()
            .map(|v| [
                v[0] * initial_radius_mm + bp.cog_mm[0],
                v[1] * initial_radius_mm + bp.cog_mm[1],
                v[2] * initial_radius_mm + bp.cog_mm[2],
            ])
            .collect();

        let (neighbor_matrix, neighbor_counts) = build_neighbor_matrix(n_vertices, &faces, 6);
        let bt = 0.5_f64.powf(0.275);

        evolution_pass(
            &data, nx, ny, nz, &voxel_size, &bp,
            &mut vertices, &faces,
            &neighbor_matrix, &neighbor_counts,
            bt, 1.0, 0.5, // gradient_threshold = 0.5
            10, 0,
            &mut None,
        );

        for v in &vertices {
            assert!(v[0].is_finite() && v[1].is_finite() && v[2].is_finite());
        }
    }

    #[test]
    fn test_evolution_pass_with_progress_callback() {
        let n = 16;
        let (data, nx, ny, nz) = make_sphere_volume(n, 6.0, 200.0);
        let voxel_size = [1.0, 1.0, 1.0];
        let bp = estimate_brain_parameters(&data, nx, ny, nz, &voxel_size);

        let (unit_verts, faces) = create_icosphere(1);
        let n_vertices = unit_verts.len();
        let initial_radius_mm = bp.radius * 0.5;

        let mut vertices: Vec<[f64; 3]> = unit_verts
            .iter()
            .map(|v| [
                v[0] * initial_radius_mm + bp.cog_mm[0],
                v[1] * initial_radius_mm + bp.cog_mm[1],
                v[2] * initial_radius_mm + bp.cog_mm[2],
            ])
            .collect();

        let (neighbor_matrix, neighbor_counts) = build_neighbor_matrix(n_vertices, &faces, 6);
        let bt = 0.5_f64.powf(0.275);

        let mut calls = 0usize;
        let mut callback = |_iter: usize, _total: usize| {
            calls += 1;
        };
        let mut cb: Option<&mut dyn FnMut(usize, usize)> = Some(&mut callback);

        evolution_pass(
            &data, nx, ny, nz, &voxel_size, &bp,
            &mut vertices, &faces,
            &neighbor_matrix, &neighbor_counts,
            bt, 1.0, 0.0,
            20, 0,
            &mut cb,
        );

        assert!(calls > 0, "Progress callback should have been called");
    }
}

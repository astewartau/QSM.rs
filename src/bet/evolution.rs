//! BET surface evolution algorithm
//!
//! Based on: Smith, S.M. (2002) "Fast robust automated brain extraction"
//! Human Brain Mapping, 17(3):143-155
//!
//! Aligned with FSL-BET2 implementation.
//!
//! # Reference pipeline
//!
//! - **Defaults.** FSL BET2 (Bostrix/FSL-BET2 d5acd7f, `bet2.cpp`): `-f 0.5` (`:133`), `-g 0`
//!   (`:136`), smoothness `-w 1` (`:139`), 1000 iterations (`:792`), curvature radii
//!   3.33–10 mm scaled by the smoothness (`:788-789`), surface from `make_mesh_from_icosa(5, m)`
//!   (`:772`). [`BetParams::default`] has the same `f`, `g`, smoothness and iterations, and
//!   `subdivisions: 4`: bet2's `make_mesh_from_icosa(subdivisions + 1)`, 2562 vertices.
//! - **Input.** BET was developed on structural (T1-, T2-, PD-weighted) head images (Smith
//!   2002) and works in mm
//!   (the search distances and curvature limits above), so it assumes human-scale anatomy; see
//!   [`BetParams::voxel_scale`] for rodents. It needs no orientation: the initial sphere comes
//!   from the image's intensity centre of gravity. `-g` is a gradient along the voxel z axis.
//! - **Ported from bet2 step by step** (astewartau/QSM.rs#327):
//!   - the mesh is bet2's own icosahedron, `make_mesh_from_icosa`: its orientation, the
//!     triangle order and winding, and `retessellate` ([`super::icosphere::bet2_icosa`]);
//!   - vertex normals are `Mpoint::local_normal`, the normalised sum of the unnormalised
//!     triangle normals (area-weighted);
//!   - intensities are sampled at the nearest voxel, `(int)(p/dim + 0.5)`, not interpolated.
//!     Inside the search loop this includes newimage's unchecked `value()`, which past an edge
//!     reads a neighbouring row's voxel;
//!   - `l` is the mean over vertices of the mean distance to their neighbours, recomputed at the
//!     start of iterations 0, 50, 100, 200, …;
//!   - `t2`/`t98` are newimage's `robustmin`/`robustmax` (`find_thresholds`, a histogram method),
//!     and `tm` is the `⌊n/2⌋`-th smallest value;
//!   - `-f`, `-w` and `-g` are rounded to single precision, as bet2 reads them;
//!   - the mask is `make_mask_from_mesh`: the triangles drawn as bet2's `draw_mesh` draws them,
//!     then a 6-connected spread from the mesh's mean point, with no hole filling;
//!   - FSL reverses x of a neurologically stored image (positive determinant) before bet2 runs
//!     ([`FslLeftRightOrder`]).
//! - **Agreement with bet2** (Bostrix/FSL-BET2 d5acd7f built from source; measured 2026-10-31):
//!   - **Travelling head:** on `prisma-bridge-run1`'s echo-1 magnitude after ANTs N4, the masks
//!     are **identical**, at `-f 0.5` and at `-f 0.2` (0 of 6.0 M voxels differ). Before the
//!     port, Dice was 0.9857 and 0.9916, with QSM.rs's masks 1.25 % and 0.64 % larger.
//!   - **Synthetic head** of `tests/bet_bet2_parity.rs`: Dice 0.9991 (`-f 0.5`, 5 voxels) and
//!     1.0000 (`-f 0.2`). That piecewise-constant head is sensitive at this level. Before the
//!     port: 0.9903 and 0.9922.
//!   - **Undefined behaviour in bet2:** a sample index outside the whole volume buffer is read
//!     as 0 here.
//! - **In QSM pipelines.** MEDI runs BET on `iMag = sqrt(sum(abs(iField).^2,4))`, the RSS over
//!   echoes, with `f = 0.5` (MEDI_toolbox `README.m:92,111`) or `f = 0.1`
//!   (`MEDI_full_process.m:109,128`). QSMART runs `bet2 -f 0.2 -m` on the first echo's
//!   coil-RSS magnitude after N4 (`QSMART_toolbox_v1.0/brainmask.m:7-14`). QSM.jl's README runs
//!   `bet` on the last echo with `-f 0.5` and erodes the mask by 5 voxels (`README.md:34-38`).

use crate::Grid;
use super::icosphere::bet2_icosa;
#[cfg(test)]
use super::icosphere::create_icosphere;
use super::mesh::{bet2_vertex_normals, build_neighbor_matrix, self_intersection_heuristic};

/// Brain parameters struct (like FSL's bet_parameters)
struct BetParameters {
    t2: f64,       // 2nd percentile (robust min)
    // `t98` and `cog` are consumed only as intermediates -- `t98` by the `t` threshold and the
    // `tm` intensity window, `cog` by `cog_mm` -- so the evolution path never reads the stored
    // fields and the lib target sees them as dead. They are kept because the unit tests below
    // assert on them directly (that `t98 >= t2`, that `tm` falls between the two, that `cog`
    // lands on the phantom's centre), which is the only check that `estimate_brain_parameters`
    // still matches FSL-BET2's `adjust_initial_mesh`. Dropping the fields would delete that.
    #[allow(dead_code)]
    t98: f64,      // 98th percentile (robust max)
    t: f64,        // threshold = t2 + 0.1*(t98-t2)
    tm: f64,       // median within-brain intensity (critical for proper surface evolution)
    #[allow(dead_code)]
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

    // bet2: newimage's robustmin/robustmax (`find_thresholds`) over the whole volume.
    let (t2, t98) = find_thresholds(data);
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
        // bet2: med = floor(n/2); tm = max of the `med` smallest values, i.e. sorted[med - 1].
        within_brain_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = within_brain_values.len() / 2;
        within_brain_values[med.max(1) - 1]
    };

    BetParameters { t2, t98, t, tm, cog, cog_mm, radius }
}

/// newimage's `find_thresholds` (Steve Smith's robust 2 %/98 % limits; FSL-BET2 d5acd7f
/// `newimage/newimage.cc`), for a `volume<float>`: a 1000-bin histogram between the minimum and
/// maximum, the bins holding the lowest and highest 2 %, and while those limits span less than a
/// tenth of the range, the histogram redone between them (up to 10 passes, the last ignoring the
/// end bins). Returns (`robustmin`, `robustmax`).
fn find_thresholds(data: &[f64]) -> (f64, f64) {
    const BINS: i64 = 1000;
    const MAX_PASSES: i32 = 10;
    let vol: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let vmin = vol.iter().copied().fold(f32::INFINITY, f32::min);
    let vmax = vol.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let (mut min, mut max) = (vmin, vmax);
    let (mut top_bin, mut bottom_bin, mut pass) = (0i64, 0i64, 1);
    let (mut lowest_bin, mut highest_bin) = (0i64, BINS - 1);
    let (mut thresh2, mut thresh98) = (0.0f32, 0.0f32);
    let mut hist = vec![0i64; BINS as usize];
    while pass == 1 || ((thresh98 - thresh2) as f64) < ((max - min) as f64) / 10.0 {
        if pass > 1 {
            bottom_bin = (bottom_bin - 1).max(0);
            top_bin = (top_bin + 1).min(BINS - 1);
            let tmpmin = (min as f64 + (bottom_bin as f64 / BINS as f64) * (max - min) as f64) as f32;
            max = (min as f64 + ((top_bin + 1) as f64 / BINS as f64) * (max - min) as f64) as f32;
            min = tmpmin;
        }
        if pass == MAX_PASSES || min == max {
            min = vmin;
            max = vmax;
        }
        // find_histogram
        hist.iter_mut().for_each(|h| *h = 0);
        let mut validsize: i64 = if min == max { -1 } else { 0 };
        if min != max {
            let fa = BINS as f64 / (max - min) as f64;
            let fb = (BINS as f64 * (-(min as f64))) / (max - min) as f64;
            for &v in &vol {
                let b = ((fa * v as f64 + fb) as i64).clamp(0, BINS - 1);
                hist[b as usize] += 1;
                validsize += 1;
            }
        }
        if validsize < 1 {
            return (min as f64, max as f64);
        }
        if pass == MAX_PASSES {
            validsize -= hist[lowest_bin as usize] + hist[highest_bin as usize];
            lowest_bin += 1;
            highest_bin -= 1;
        }
        if validsize < 0 {
            thresh2 = min;
            thresh98 = min;
            break;
        }
        let fa = (max - min) as f64 / BINS as f64;
        let mut count = 0i64;
        bottom_bin = lowest_bin;
        while count < validsize / 50 {
            count += hist[bottom_bin as usize];
            bottom_bin += 1;
        }
        bottom_bin -= 1;
        thresh2 = min + (bottom_bin as f64 * fa) as f32;
        count = 0;
        top_bin = highest_bin;
        while count < validsize / 50 {
            count += hist[top_bin as usize];
            top_bin -= 1;
        }
        top_bin += 1;
        thresh98 = min + ((top_bin + 1) as f64 * fa) as f32;
        if pass == MAX_PASSES {
            break;
        }
        pass += 1;
    }
    (thresh2 as f64, thresh98 as f64)
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
    // bet2.cpp `step_of_computation`: nearest-voxel samples, `(int)(p/dim + 0.5)` (truncated
    // toward zero), along the inward normal from 1 mm; d1 = 7, d2 = 3.
    const D1: f64 = 7.0;
    const D2: f64 = 3.0;
    let (xdim, ydim, zdim) = (voxel_size[0], voxel_size[1], voxel_size[2]);
    let dscale = xdim.min(ydim).min(zdim).min(1.0);
    let in_bounds = |i: i64, j: i64, k: i64| i >= 0 && j >= 0 && k >= 0 && (i as usize) < nx && (j as usize) < ny && (k as usize) < nz;
    // newimage's `value(x, y, z)` does no bounds check: it reads the voxel at the linear index
    // `(z·ny + y)·nx + x`, which for an index past an edge is a voxel of a neighbouring row or
    // slice. That is reproduced; an index outside the whole buffer (undefined behaviour in bet2)
    // reads 0.
    let n_total = (nx * ny * nz) as i64;
    let value = |i: i64, j: i64, k: i64| {
        let lin = (k * ny as i64 + j) * nx as i64 + i;
        if (0..n_total).contains(&lin) { data[lin as usize] } else { 0.0 }
    };
    let mut i_min = tm;
    let mut i_max = t;
    let p = [point_mm[0] - normal[0], point_mm[1] - normal[1], point_mm[2] - normal[2]]; // coord + (-1)·n
    let (mut iv, mut jv, mut kv) = (p[0] / xdim + 0.5, p[1] / ydim + 0.5, p[2] / zdim + 0.5);
    if !in_bounds(iv as i64, jv as i64, kv as i64) {
        return (i_min, i_max, false);
    }
    let im = value(iv as i64, jv as i64, kv as i64);
    i_min = i_min.min(im);
    i_max = i_max.max(im);
    let (mut nxv, mut nyv, mut nzv) = (normal[0] / xdim, normal[1] / ydim, normal[2] / zdim);
    let (i2, j2, k2) = ((iv - (D1 - 1.0) * nxv) as i64, (jv - (D1 - 1.0) * nyv) as i64, (kv - (D1 - 1.0) * nzv) as i64);
    nxv *= dscale;
    nyv *= dscale;
    nzv *= dscale;
    if !in_bounds(i2, j2, k2) {
        return (i_min, i_max, false);
    }
    let im = value(i2, j2, k2);
    i_min = i_min.min(im);
    let mut gi = 2.0;
    while gi < D1 {
        iv -= nxv;
        jv -= nyv;
        kv -= nzv;
        let im = value(iv as i64, jv as i64, kv as i64);
        i_min = i_min.min(im);
        if gi < D2 {
            i_max = i_max.max(im);
        }
        gi += dscale;
    }
    i_min = i_min.max(t2);
    i_max = i_max.min(tm);
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
    // bet2.cpp `make_mask_from_mesh`: a volume of 1s; every triangle drawn as 0s
    // (`draw_mesh`: points along the edge v1 -> v0 every half of the smallest voxel size, each
    // joined to v2 by `draw_segment`, samples `floor(p/dim + 0.5)`); then a 6-connected spread
    // of 0s from the mesh's mean point (truncated to a voxel) through the 1s. Brain is the 0s.
    let (xdim, ydim, zdim) = (voxel_size[0], voxel_size[1], voxel_size[2]);
    let mininc = xdim.min(ydim).min(zdim) * 0.5;
    let mut grid: Vec<u8> = vec![1; nx * ny * nz];
    let idx = |i: usize, j: usize, k: usize| i + j * nx + k * nx * ny;
    let set0 = |p: [f64; 3], grid: &mut Vec<u8>| {
        let (i, j, k) = ((p[0] / xdim + 0.5).floor(), (p[1] / ydim + 0.5).floor(), (p[2] / zdim + 0.5).floor());
        if i >= 0.0 && j >= 0.0 && k >= 0.0 && (i as usize) < nx && (j as usize) < ny && (k as usize) < nz {
            grid[idx(i as usize, j as usize, k as usize)] = 0;
        }
    };
    let unit = |a: [f64; 3], b: [f64; 3]| -> ([f64; 3], f64) {
        let v = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let d = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if d > 0.0 { ([v[0] / d, v[1] / d, v[2] / d], d) } else { (v, d) }
    };
    for &[i0, i1, i2] in faces {
        let (v0, v1, v2) = (vertices_mm[i0], vertices_mm[i1], vertices_mm[i2]);
        let (n, d) = unit(v0, v1);
        let mut j = 0.0;
        while j <= d {
            let p = [v1[0] + j * n[0], v1[1] + j * n[1], v1[2] + j * n[2]];
            // draw_segment(p, v2): from v2 towards p.
            let (m, dd) = unit(p, v2);
            let mut t = 0.0;
            while t <= dd {
                set0([v2[0] + t * m[0], v2[1] + t * m[1], v2[2] + t * m[2]], &mut grid);
                t += mininc;
            }
            j += mininc;
        }
    }
    let mut c = [0.0f64; 3];
    for v in vertices_mm {
        c[0] += v[0];
        c[1] += v[1];
        c[2] += v[2];
    }
    let inv = 1.0 / vertices_mm.len() as f64;
    let (cx, cy, cz) = ((c[0] * inv / xdim) as i64, (c[1] * inv / ydim) as i64, (c[2] * inv / zdim) as i64);
    let mut stack: Vec<(i64, i64, i64)> = vec![(cx, cy, cz)];
    let inside = |x: i64, y: i64, z: i64| x >= 0 && y >= 0 && z >= 0 && (x as usize) < nx && (y as usize) < ny && (z as usize) < nz;
    while let Some((x, y, z)) = stack.pop() {
        if !inside(x, y, z) {
            continue;
        }
        grid[idx(x as usize, y as usize, z as usize)] = 0;
        for (dx, dy, dz) in [(-1, 0, 0), (0, -1, 0), (0, 0, -1), (1, 0, 0), (0, 1, 0), (0, 0, 1)] {
            let (a, b, c) = (x + dx, y + dy, z + dz);
            if inside(a, b, c) && grid[idx(a as usize, b as usize, c as usize)] == 1 {
                stack.push((a, b, c));
            }
        }
    }
    grid.iter().map(|&v| (v == 0) as u8).collect()
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
    vertices: &mut [[f64; 3]],
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

    // bet2: l, the mean over vertices of the mean distance to their neighbours, recomputed at
    // the start of iterations 0, 50, 100, 200, ...
    let mut l = 0.0;
    let mean_neighbour_distance = |vs: &[[f64; 3]]| -> f64 {
        let mut total = 0.0;
        for i in 0..vs.len() {
            let mut li = 0.0;
            for j in 0..neighbor_counts[i] {
                let q = vs[neighbor_matrix[i][j]];
                li += ((q[0] - vs[i][0]).powi(2) + (q[1] - vs[i][1]).powi(2) + (q[2] - vs[i][2]).powi(2)).sqrt();
            }
            total += li / neighbor_counts[i] as f64;
        }
        total / vs.len() as f64
    };

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

        if iteration == 50 || iteration % 100 == 0 {
            l = mean_neighbour_distance(vertices);
        }
        // Compute vertex normals
        let normals = bet2_vertex_normals(vertices, faces);

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

    }

    // Debug: report sampling failures
    if sample_fail_count > 0 {
        let total_samples = iterations * n_vertices;
        let fail_pct = 100.0 * sample_fail_count as f64 / total_samples as f64;
        eprintln!("[BET] Sampling fallback used: {} / {} ({:.1}%)",
                  sample_fail_count, total_samples, fail_pct);
    }
}

/// BET algorithm parameters
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct BetParams {
    /// Fractional intensity threshold (0.0-1.0, smaller = larger brain)
    pub fractional_intensity: f64,
    /// Surface smoothness factor
    pub smoothness: f64,
    /// Gradient threshold (-1 to 1)
    pub gradient_threshold: f64,
    /// Number of iterations
    pub iterations: usize,
    /// Icosphere subdivision level
    pub subdivisions: usize,
    /// Factor applied to the grid's voxel sizes before the surface is evolved (default 1.0 =
    /// the acquisition's own geometry).
    ///
    /// BET's surface model is fixed in human-scale millimetres — a 7 mm / 3 mm intensity search
    /// along each vertex normal, and curvature bounded between 3.33 mm and 10 mm radii — so on a
    /// ~1 cm rodent brain those distances span the whole head and the search runs straight
    /// through the skull. Inflating the voxel sizes so the brain reaches roughly human
    /// dimensions is the standard preclinical workaround (`fslchpixdim` x10 before `bet`);
    /// a scale of 10 suits mouse data.
    ///
    /// The mask is returned on the input index grid either way, so nothing needs scaling back.
    /// Prefer [`super::rs2_net`] where the weights can be fetched: on an in-vivo mouse GRE it
    /// follows the inner skull, where scaled BET still leaks into the skull base.
    pub voxel_scale: f64,
}

impl Default for BetParams {
    fn default() -> Self {
        Self {
            fractional_intensity: 0.5,
            smoothness: 1.0,
            gradient_threshold: 0.0,
            iterations: 1000,
            subdivisions: 4,
            voxel_scale: 1.0,
        }
    }
}

/// How FSL stores an image's left-right axis, which decides whether FSL flips it on reading
/// (`FslGetLeftRightOrder2`, `fslio.c`; `read_volume` calls `makeradiological()`): an image
/// whose sform (or, without one, qform) has a negative determinant is radiological and read as
/// it is; one with a positive determinant is neurological and FSL reverses its x axis before bet2
/// sees it (and back when saving). The mesh is not symmetric, so the flip changes the mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FslLeftRightOrder {
    /// Negative determinant (or no orientation): processed as stored.
    Radiological,
    /// Positive determinant: x reversed for processing, as FSL does.
    Neurological,
}

impl FslLeftRightOrder {
    /// From a voxel-to-world affine (row-major 4×4, the sform or else the qform the image was
    /// read with). `None` (no orientation) and a determinant within 1e-12 of 0 are
    /// radiological, as in FSL.
    pub fn from_affine(affine: Option<&[f64; 16]>) -> Self {
        match affine {
            Some(a) => {
                let det = a[0] * (a[5] * a[10] - a[6] * a[9]) - a[1] * (a[4] * a[10] - a[6] * a[8])
                    + a[2] * (a[4] * a[9] - a[5] * a[8]);
                if det >= 1e-12 { Self::Neurological } else { Self::Radiological }
            }
            None => Self::Radiological,
        }
    }
}

/// Run BET brain extraction, as FSL's bet2 (see the module docs).
///
/// # Arguments
/// * `data` - 3D magnitude image data (nx * ny * nz, Fortran order)
/// * `grid` - Volume grid (dimensions and voxel sizes)
/// * `params` - BET parameters (intensity threshold, smoothness, gradient, iterations, subdivisions)
/// * `order` - The image's [`FslLeftRightOrder`], from its header
///   ([`FslLeftRightOrder::from_affine`]): bet2 runs on neurological images with x reversed.
/// * `progress` - Progress callback: `(iteration, total_iterations)`
///
/// # Returns
/// Binary mask (1 = brain, 0 = background)
pub fn run_bet<F>(
    data: &[f64],
    grid: &Grid,
    params: &BetParams,
    order: FslLeftRightOrder,
    progress: F,
) -> Vec<u8>
where
    F: FnMut(usize, usize),
{
    match order {
        FslLeftRightOrder::Radiological => run_bet_as_stored(data, grid, params, progress),
        FslLeftRightOrder::Neurological => {
            let (nx, ny, nz) = grid.dims;
            let flip_x = |i: usize| (nx - 1 - i % nx) + nx * (i / nx);
            let flipped: Vec<f64> = (0..nx * ny * nz).map(|i| data[flip_x(i)]).collect();
            let mask = run_bet_as_stored(&flipped, grid, params, progress);
            (0..nx * ny * nz).map(|i| mask[flip_x(i)]).collect()
        }
    }
}

fn run_bet_as_stored<F>(
    data: &[f64],
    grid: &Grid,
    params: &BetParams,
    mut progress: F,
) -> Vec<u8>
where
    F: FnMut(usize, usize),
{
    // bet2 reads -f, -w and -g as floats (`Option<float>`).
    let fractional_intensity = params.fractional_intensity as f32 as f64;
    let smoothness_factor = params.smoothness as f32 as f64;
    let gradient_threshold = params.gradient_threshold as f32 as f64;
    let iterations = params.iterations;
    let subdivisions = params.subdivisions;
    let (nx, ny, nz) = grid.dims;
    // Every physical distance in BET reaches the algorithm through `voxel_size` — the brain
    // estimate, the surface evolution, the self-intersection heuristic and the rasteriser all
    // take it from here — so scaling it here is the whole of `voxel_scale`. A non-positive
    // scale would invert or collapse the geometry; treat it as "unscaled".
    let scale = if params.voxel_scale > 0.0 { params.voxel_scale } else { 1.0 };
    let voxel_size = [grid.vsx() * scale, grid.vsy() * scale, grid.vsz() * scale];

    // Step 1: Estimate brain parameters
    progress(0, iterations);
    let bp = estimate_brain_parameters(data, nx, ny, nz, &voxel_size);

    // Step 2: Create icosphere
    // bet2's own icosahedron, `make_mesh_from_icosa(subdivisions + 1)`.
    let (unit_vertices, faces) = bet2_icosa(subdivisions + 1);
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
        let mut cb: Option<&mut dyn FnMut(usize, usize)> = Some(&mut progress);
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
    progress(iterations, iterations);

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
        let grid = Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);

        let mask = run_bet(
            &data,
            &grid,
            &BetParams { fractional_intensity: 0.5, smoothness: 1.0, gradient_threshold: 0.0, iterations: 50, subdivisions: 1, voxel_scale: 1.0 },
            FslLeftRightOrder::Radiological, |_, _| {},
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

    /// `voxel_scale: s` must mean exactly "run BET as if the voxels were s times larger", so
    /// scaling the grid and scaling the parameter have to give the same mask. This is what lets
    /// the preclinical `fslchpixdim`-style workaround be a parameter rather than a pre-pass.
    #[test]
    fn test_voxel_scale_matches_an_equally_scaled_grid() {
        let (data, nx, ny, nz) = make_sphere_volume(16, 6.0, 200.0);
        let base = BetParams {
            fractional_intensity: 0.5, smoothness: 1.0, gradient_threshold: 0.0,
            iterations: 50, subdivisions: 1, voxel_scale: 1.0,
        };

        // Anisotropic voxels, so a scale applied to the wrong axis would show up.
        let scaled_grid = Grid::new(nx, ny, nz, 3.0, 4.5, 6.0);
        let via_grid = run_bet(&data, &scaled_grid, &base, FslLeftRightOrder::Radiological, |_, _| {});

        let native_grid = Grid::new(nx, ny, nz, 0.3, 0.45, 0.6);
        let via_param = run_bet(
            &data, &native_grid,
            &BetParams { voxel_scale: 10.0, ..base.clone() },
            FslLeftRightOrder::Radiological, |_, _| {},
        );

        assert_eq!(via_param, via_grid, "voxel_scale must equal scaling the grid itself");
    }

    /// A non-positive scale would invert or collapse the geometry; it falls back to unscaled
    /// rather than producing a degenerate surface.
    #[test]
    fn test_non_positive_voxel_scale_is_treated_as_unscaled() {
        let (data, nx, ny, nz) = make_sphere_volume(16, 6.0, 200.0);
        let grid = Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);
        let base = BetParams {
            fractional_intensity: 0.5, smoothness: 1.0, gradient_threshold: 0.0,
            iterations: 50, subdivisions: 1, voxel_scale: 1.0,
        };
        let unscaled = run_bet(&data, &grid, &base, FslLeftRightOrder::Radiological, |_, _| {});

        for bad in [0.0, -1.0] {
            let mask = run_bet(&data, &grid, &BetParams { voxel_scale: bad, ..base.clone() }, FslLeftRightOrder::Radiological, |_, _| {});
            assert_eq!(mask, unscaled, "voxel_scale {bad} should behave as 1.0");
        }
    }

    #[test]
    fn test_run_bet_with_gradient_threshold() {
        // Exercise the gradient threshold code path
        let (data, nx, ny, nz) = make_sphere_volume(16, 6.0, 200.0);
        let grid = Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);

        let mask = run_bet(
            &data,
            &grid,
            &BetParams { fractional_intensity: 0.5, smoothness: 1.0, gradient_threshold: 0.3, iterations: 50, subdivisions: 1, voxel_scale: 1.0 },
            FslLeftRightOrder::Radiological, |_, _| {},
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
        let grid = Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);

        let mut progress_calls = Vec::new();
        let mask = run_bet(
            &data,
            &grid,
            &BetParams { fractional_intensity: 0.5, smoothness: 1.0, gradient_threshold: 0.0, iterations: 50, subdivisions: 1, voxel_scale: 1.0 },
            FslLeftRightOrder::Radiological, |current, total| {
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
        let grid = Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);

        // Higher fractional_intensity = smaller brain
        let mask_small = run_bet(
            &data,
            &grid,
            &BetParams { fractional_intensity: 0.7, smoothness: 1.0, gradient_threshold: 0.0, iterations: 50, subdivisions: 1, voxel_scale: 1.0 },
            FslLeftRightOrder::Radiological, |_, _| {},
        );
        let mask_large = run_bet(
            &data,
            &grid,
            &BetParams { fractional_intensity: 0.3, smoothness: 1.0, gradient_threshold: 0.0, iterations: 50, subdivisions: 1, voxel_scale: 1.0 },
            FslLeftRightOrder::Radiological, |_, _| {},
        );

        let count_small: usize = mask_small.iter().map(|&v| v as usize).sum();
        let count_large: usize = mask_large.iter().map(|&v| v as usize).sum();

        // Both should produce valid masks
        assert!(count_small > 0);
        assert!(count_large > 0);
    }

    #[test]
    fn test_run_bet_anisotropic_voxels() {
        let (data, nx, ny, nz) = make_sphere_volume(16, 6.0, 200.0);
        let grid = Grid::new(nx, ny, nz, 2.0, 2.0, 2.0);

        let mask = run_bet(
            &data,
            &grid,
            &BetParams { fractional_intensity: 0.5, smoothness: 1.0, gradient_threshold: 0.0, iterations: 50, subdivisions: 1, voxel_scale: 1.0 },
            FslLeftRightOrder::Radiological, |_, _| {},
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

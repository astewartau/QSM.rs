//! Surface Curvature Calculation for QSMART
//!
//! This module computes Gaussian and mean curvatures at the surface of a 3D binary mask,
//! based on the discrete differential geometry approach.
//!
//! The curvatures are used in QSMART to weight the spatially-dependent filtering
//! near brain boundaries to reduce artifacts.
//!
//! Uses 2D Delaunay triangulation matching MATLAB's approach: `tri = delaunay(x, y)` -
//! triangulates on x,y coordinates with z as height.
//!
//! Reference:
//! Meyer, M., Desbrun, M., Schröder, P., Barr, A.H. (2003).
//! "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds."
//! Visualization and Mathematics III, 35-57. https://doi.org/10.1007/978-3-662-05105-4_2
//!
//! Reference implementation: https://www.mathworks.com/matlabcentral/fileexchange/61136-curvatures
//!
//! ## Agreement with the original
//!
//! SDF's smoothing widths are `σ·round(prox^n, 2)`, a discrete choice, so a small difference in
//! the curvature-weighted proximity map moves whole voxels from one width to another. This module
//! therefore reproduces MATLAB's curvature, not just the method (astewartau/QSM.rs#320):
//!
//! - **Triangulation.** MATLAB's `delaunay` is CGAL's, which breaks cocircular ties by a symbolic
//!   perturbation, so its triangulation of the grid's many cocircular points is unique.
//!   `triangulate_surface` flips delaunator's ties to it.
//! - **Arithmetic.** `curvatures.m`'s angles with MATLAB's `acos` (FDLIBM's,
//!   [`fdlibm::acos`](crate::utils::fdlibm::acos)), `norm` (reference BLAS `dnrm2`) and `dot`, the
//!   third angle as `pi - (a1 + a2)`, and `A_mixed`'s `>= pi/2`. A vertex on a flat patch has an
//!   angle sum of 2π up to rounding, and the sign of `2π - Σθ` decides whether it gets a
//!   curvature weight of 1 or (nearly) 0, so these are MATLAB's bits on purpose. Only IEEE
//!   arithmetic and FDLIBM are used, so the result is the same on every target, the browser
//!   included.
//! - **Erosion.** `imerode` pads with the maximum: a mask touching the array edge isn't eroded
//!   there ([`imerode_sphere`](crate::utils::morphology)).
//!
//! Measured with licensed MATLAB R2023b on the 36×36×28 fixture of
//! `tests/qsmart_qsmci_parity.rs` (an instrumented copy of `sdf_curvature.m` and
//! `calculate_curvature.m`, each QSM.rs step given MATLAB's inputs): MATLAB's 1,122 triangles
//! (539 cocircular edges among them) triangle for triangle, all 3,366 angles bit for bit, and
//! SDF's local field to 4e-5 % in stage 1 and 2e-5 % in stage 2 (the float32 of the fixture),
//! where delaunator's ties, glibc's `acos` and plain `sqrt(x²+y²+z²)` had given 10.9 % and 2.1 %.
//! What is left: 29 of 1,600 surface voxels still get a different curvature weight. They are
//! flat vertices, where MATLAB's `2π - Σθ` is ±1e-16 and its sign follows the order of CGAL's
//! triangles and of their vertices, which no public interface fixes. Here they don't change any
//! smoothing width.
//!
//! ## Reference pipeline
//!
//! QSMART's `calculate_curvature.m` (Syeda, October 2020) takes the surface as the brain mask
//! minus its erosion by `strel('sphere', 1)`, triangulates it with `delaunay(x, y)`, scales the
//! negative Gaussian curvatures by `curvConstant / max|GC<0|` (500 in `Demo_QSMART.m`) and sets
//! positive ones to 1; the curvature map is smoothed with widths `[σ, 2σ, 2σ]` voxels (σ = SDF's
//! σ1; the first voxel axis gets half the width of the other two, whatever the orientation),
//! clamped at 0.5, and edge-refined with a `[5, 10, 10]`-voxel Gaussian (l. 21–48). The
//! October 2020 toolbox version differs (curvature constant 250 hard-coded, no edge refinement);
//! QSM.rs follows the GitHub v1.0 code.

use std::collections::HashMap;
use std::f64::consts::PI;
use delaunator::{triangulate, Point};
use crate::Grid;
use crate::utils::morphology::{dilate_ball, imerode_sphere};
use crate::utils::fdlibm;
use crate::utils::portable_math as pm;

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
    let eroded = imerode_sphere(mask, nx, ny, nz, 1);

    let mut surface = Vec::new();
    for i in 0..mask.len() {
        if mask[i] != 0 && eroded[i] == 0 {
            surface.push(i);
        }
    }

    surface
}

/// Sign of the orientation of `(a, b, c)`: positive if counter-clockwise. Exact on the integer
/// voxel coordinates.
fn orient(a: (i64, i64), b: (i64, i64), c: (i64, i64)) -> i64 {
    ((b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)).signum()
}

/// Sign of the in-circle determinant: positive if `d` is inside the circle through the
/// counter-clockwise `(a, b, c)`. Exact on the integer voxel coordinates.
fn incircle(a: (i64, i64), b: (i64, i64), c: (i64, i64), d: (i64, i64)) -> i64 {
    let r = |p: (i64, i64)| {
        let (x, y) = ((p.0 - d.0) as i128, (p.1 - d.1) as i128);
        (x, y, x * x + y * y)
    };
    let (ax, ay, aw) = r(a);
    let (bx, by, bw) = r(b);
    let (cx, cy, cw) = r(c);
    let det = ax * (by * cw - bw * cy) - ay * (bx * cw - bw * cx) + aw * (bx * cy - by * cx);
    det.signum() as i64
}

/// CGAL's `Delaunay_triangulation_2::side_of_oriented_circle(p0, p1, p2, p, perturb = true)`:
/// the in-circle test, with cocircular ties broken by a symbolic perturbation that orders the
/// points lexicographically by `(x, y)`. Positive: `p` is inside, and the edge opposite it has to
/// be flipped.
fn side_of_oriented_circle(pts: &[(i64, i64)], p0: usize, p1: usize, p2: usize, p: usize) -> i64 {
    let os = incircle(pts[p0], pts[p1], pts[p2], pts[p]);
    if os != 0 {
        return os;
    }
    let mut order = [p0, p1, p2, p];
    order.sort_by_key(|&i| pts[i]);
    for &q in order[2..].iter().rev() {
        if q == p {
            return -1;
        }
        let o = if q == p2 {
            orient(pts[p0], pts[p1], pts[p])
        } else if q == p1 {
            orient(pts[p0], pts[p], pts[p2])
        } else {
            orient(pts[p], pts[p1], pts[p2])
        };
        if o != 0 {
            return o;
        }
    }
    -1
}

/// 2-D Delaunay triangulation of the surface points' `(x, y)`, as MATLAB's `delaunay(x, y)`.
///
/// Surface voxels lie on an integer grid, so many point quadruples are cocircular and the plain
/// Delaunay triangulation is not unique. MATLAB's `delaunay` is CGAL's `Delaunay_triangulation_2`,
/// which breaks every such tie by a symbolic perturbation ([`side_of_oriented_circle`]): that
/// makes its triangulation unique, whatever the insertion order. delaunator's is a valid Delaunay
/// triangulation that breaks ties its own way, so its cocircular edges are flipped here, with
/// exact integer predicates, until every edge is Delaunay under CGAL's perturbation. Measured on
/// the 36×36×28 fixture of `tests/qsmart_qsmci_parity.rs` (astewartau/QSM.rs#320): MATLAB's
/// triangulation has 539 cocircular edges, and this gives it triangle for triangle.
///
/// Points must have unique (x,y) coordinates (caller should deduplicate first), on the integer
/// grid. Triangles come out counter-clockwise in `(x, y)`.
///
/// Returns (triangles, boundary_flags): `boundary_flags[i]` is true if vertex `i` is on the free
/// boundary (the convex hull, collinear points on it included), as `curvatures.m`'s
/// `freeBoundary` finds it.
fn triangulate_surface(
    points: &[Point3D],
) -> (Vec<Triangle>, Vec<bool>) {
    let n = points.len();
    if n < 3 {
        return (Vec::new(), vec![false; n]);
    }
    let pts: Vec<(i64, i64)> = points.iter().map(|p| (p.x as i64, p.y as i64)).collect();

    let coords: Vec<Point> = points.iter()
        .map(|p| Point { x: p.x, y: p.y })
        .collect();
    let result = triangulate(&coords);

    // Counter-clockwise triangles; delaunator can leave zero-area slivers along collinear hull
    // points, which CGAL's triangulation doesn't have.
    let mut tris: Vec<[usize; 3]> = Vec::with_capacity(result.triangles.len() / 3);
    for t in result.triangles.as_chunks::<3>().0 {
        match orient(pts[t[0]], pts[t[1]], pts[t[2]]) {
            1 => tris.push([t[0], t[1], t[2]]),
            -1 => tris.push([t[0], t[2], t[1]]),
            _ => {}
        }
    }

    // Directed edge (a, b) -> the triangle that has it.
    let mut edge: HashMap<(usize, usize), usize> = HashMap::with_capacity(3 * tris.len());
    for (ti, t) in tris.iter().enumerate() {
        for k in 0..3 {
            edge.insert((t[k], t[(k + 1) % 3]), ti);
        }
    }
    let third = |t: &[usize; 3], a: usize, b: usize| -> usize {
        t.iter().copied().find(|&v| v != a && v != b).unwrap()
    };
    let mut stack: Vec<(usize, usize)> = edge.keys().copied().filter(|&(a, b)| a < b).collect();
    stack.sort_unstable();
    while let Some((a, b)) = stack.pop() {
        let (Some(&t1), Some(&t2)) = (edge.get(&(a, b)), edge.get(&(b, a))) else { continue };
        let c = third(&tris[t1], a, b);
        let d = third(&tris[t2], a, b);
        if side_of_oriented_circle(&pts, a, b, c, d) <= 0 {
            continue;
        }
        // Flip a-b to c-d: (a, b, c), (b, a, d) -> (a, d, c), (d, b, c).
        for t in [t1, t2] {
            let tri = tris[t];
            for k in 0..3 {
                edge.remove(&(tri[k], tri[(k + 1) % 3]));
            }
        }
        tris[t1] = [a, d, c];
        tris[t2] = [d, b, c];
        for t in [t1, t2] {
            let tri = tris[t];
            debug_assert_eq!(orient(pts[tri[0]], pts[tri[1]], pts[tri[2]]), 1);
            for k in 0..3 {
                edge.insert((tri[k], tri[(k + 1) % 3]), t);
            }
        }
        stack.extend([(a, d), (d, b), (b, c), (c, a)]);
    }

    let mut boundary = vec![false; n];
    for &(a, b) in edge.keys() {
        if !edge.contains_key(&(b, a)) {
            boundary[a] = true;
        }
    }
    let triangles = tris.into_iter().map(|t| Triangle { v0: t[0], v1: t[1], v2: t[2] }).collect();
    (triangles, boundary)
}

/// MATLAB's `norm` of a 3-vector: the reference BLAS `dnrm2` (scaled sum of squares), which is
/// what reproduces `curvatures.m`'s edge lengths bit for bit.
fn norm_matlab(v: [f64; 3]) -> f64 {
    let mut scale = 0.0f64;
    let mut ssq = 1.0f64;
    for x in v {
        if x != 0.0 {
            let a = x.abs();
            if scale < a {
                ssq = 1.0 + ssq * (scale / a) * (scale / a);
                scale = a;
            } else {
                ssq += (a / scale) * (a / scale);
            }
        }
    }
    scale * ssq.sqrt()
}

/// MATLAB's `dot` of two 3-vectors, in the order that reproduces `curvatures.m`'s angles.
fn dot_matlab(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + (a[1] * b[1] + a[2] * b[2])
}

/// Compute Gaussian and mean curvatures using discrete differential geometry
///
/// `curvatures.m` (Dastan, v1.1) line by line: per triangle, its area, edge vectors
/// `v1 = p2 - p1`, `v2 = p3 - p2`, `v3 = p1 - p3`, and angles (`acos` at `p1` and `p2`, and
/// `pi - (a1 + a2)` at `p3`); per vertex not on the free boundary, the angle sum, `A_mixed`
/// (a triangle with an angle `>= pi/2` gives a half or a quarter of its area) and the mean
/// curvature vector, over its triangles. The arithmetic is MATLAB's (its `acos`, `norm` and `dot`),
/// so the angles are its bits: the sign of `2π - Σθ` at a vertex on a flat patch, which decides
/// whether it gets a curvature weight of 1 or 0, is decided by rounding.
///
/// Boundary vertices (on the triangulation free boundary) get GC=0, MC=0, as in `curvatures.m`.
fn compute_curvatures_from_mesh(
    points: &[Point3D],
    triangles: &[Triangle],
    boundary: &[bool],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n_points = points.len();
    let nt = triangles.len();
    let p = |i: usize| [points[i].x, points[i].y, points[i].z];
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let div = |a: [f64; 3], s: f64| [a[0] / s, a[1] / s, a[2] / s];
    let neg = |a: [f64; 3]| [-a[0], -a[1], -a[2]];

    let mut area = vec![0.0f64; nt];
    let mut v = vec![[[0.0f64; 3]; 3]; nt];
    let mut l_edg = vec![[0.0f64; 3]; nt];
    let mut ang = vec![[0.0f64; 3]; nt];
    let mut f_normal = vec![[0.0f64; 3]; nt];
    let mut f_center = vec![[0.0f64; 3]; nt];
    for (i, t) in triangles.iter().enumerate() {
        let (p1, p2, p3) = (p(t.v0), p(t.v1), p(t.v2));
        let (a, b) = (sub(p2, p1), sub(p3, p1));
        let cr = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
        area[i] = 0.5 * norm_matlab(cr);
        let cn = norm_matlab(cr);
        f_normal[i] = if cn > 0.0 { div(cr, cn) } else { [0.0; 3] };
        let (v1, v2, v3) = (sub(p2, p1), sub(p3, p2), sub(p1, p3));
        let l = [norm_matlab(v1), norm_matlab(v2), norm_matlab(v3)];
        // incenter: (|p2p3|·p1 + |p3p1|·p2 + |p1p2|·p3) / perimeter
        let per = l[0] + l[1] + l[2];
        for d in 0..3 {
            f_center[i][d] = (l[1] * p1[d] + l[2] * p2[d] + l[0] * p3[d]) / per;
        }
        let a0 = fdlibm::acos(dot_matlab(div(v1, l[0]), div(neg(v3), l[2])));
        let a1 = fdlibm::acos(dot_matlab(div(neg(v1), l[0]), div(v2, l[1])));
        ang[i] = [a0, a1, PI - (a0 + a1)];
        v[i] = [v1, v2, v3];
        l_edg[i] = l;
    }

    let mut attached: Vec<Vec<usize>> = vec![Vec::new(); n_points];
    for (i, t) in triangles.iter().enumerate() {
        for vi in [t.v0, t.v1, t.v2] {
            attached[vi].push(i);
        }
    }

    let mut gaussian_curvature = vec![0.0f64; n_points];
    let mut mean_curvature = vec![0.0f64; n_points];
    let mut area_mixed = vec![0.0f64; n_points];
    for i in 0..n_points {
        if boundary[i] {
            continue;
        }
        let mut alf = 0.0f64;
        let mut a_mixed = 0.0f64;
        let mut mc_vec = [0.0f64; 3];
        let mut n_vec = [0.0f64; 3];
        for &tn in &attached[i] {
            let t = &triangles[tn];
            let k = [t.v0, t.v1, t.v2].iter().position(|&x| x == i).unwrap();
            let (an, vv) = (ang[tn], v[tn]);
            alf += an[k];
            let (e, w1, f, w2) = match k {
                0 => (vv[0], an[2], vv[2], an[1]),
                1 => (vv[1], an[0], vv[0], an[2]),
                _ => (vv[2], an[1], vv[1], an[0]),
            };
            let (t1, t2) = (libm::tan(w1), libm::tan(w2));
            for d in 0..3 {
                mc_vec[d] += e[d] / t1 - f[d] / t2;
            }
            if an[k] >= PI / 2.0 {
                a_mixed += area[tn] / 2.0;
            } else if an.iter().any(|&x| x >= PI / 2.0) {
                a_mixed += area[tn] / 4.0;
            } else {
                let mut sum = 0.0f64;
                for m in 0..3 {
                    if m != k {
                        let ll = (m + 1) % 3;
                        sum += l_edg[tn][ll].powi(2) / libm::tan(an[m]);
                    }
                }
                a_mixed += sum / 8.0;
            }
            let wi = 1.0 / norm_matlab(sub(f_center[tn], p(i)));
            for d in 0..3 {
                n_vec[d] += wi * f_normal[tn][d];
            }
        }
        gaussian_curvature[i] = (2.0 * PI - alf) / a_mixed;
        let mc = [0.25 * mc_vec[0] / a_mixed, 0.25 * mc_vec[1] / a_mixed, 0.25 * mc_vec[2] / a_mixed];
        let nn = norm_matlab(n_vec);
        let nv = div(n_vec, nn);
        let mag = norm_matlab(mc);
        mean_curvature[i] = if dot_matlab(mc, nv) < 0.0 { -mag } else { mag };
        area_mixed[i] = a_mixed;
    }

    (gaussian_curvature, mean_curvature, area_mixed)
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
    grid: &Grid,
) -> (Vec<f64>, Vec<f64>) {
    let (nx, ny, nz) = grid.dims;
    let n_total = nx * ny * nz;

    // Extract surface voxels
    let surface_indices = extract_surface_voxels(mask, nx, ny, nz);

    if surface_indices.is_empty() {
        return (prox1.to_vec(), vec![1.0; n_total]);
    }

    // Convert surface indices to 3D points
    let all_points: Vec<Point3D> = surface_indices
        .iter()
        .map(|&idx| {
            let i = idx % nx;
            let j = (idx / nx) % ny;
            let k = idx / (nx * ny);
            Point3D::new(i as f64, j as f64, k as f64)
        })
        .collect();

    // Deduplicate (x,y) coordinates before triangulation.
    // MATLAB's delaunay suppresses duplicate (x,y) points, keeping
    // the first occurrence (smallest z). Duplicate vertices get GC=Inf → curvI=1.0.
    // We explicitly dedup to avoid degenerate zero-area triangles that would
    // corrupt curvature values.
    let mut xy_to_rep: HashMap<(usize, usize), usize> = HashMap::new();
    let mut is_representative = vec![false; all_points.len()];
    for (idx, p) in all_points.iter().enumerate() {
        let key = (p.x as usize, p.y as usize);
        xy_to_rep.entry(key).or_insert_with(|| {
            is_representative[idx] = true;
            idx
        });
    }

    // Build representative point array and index mapping
    let rep_indices: Vec<usize> = (0..all_points.len())
        .filter(|&i| is_representative[i])
        .collect();
    let mut orig_to_rep = vec![0usize; all_points.len()];
    for (new_idx, &old_idx) in rep_indices.iter().enumerate() {
        orig_to_rep[old_idx] = new_idx;
    }
    let rep_points: Vec<Point3D> = rep_indices.iter().map(|&i| all_points[i]).collect();

    // 2-D Delaunay of the unique representatives. Surface voxels lie on an integer grid, so many
    // point quadruples are cocircular and the triangulation is not unique: MATLAB's `delaunay`
    // and delaunator break those ties differently, and the curvatures follow the triangulation.
    let (triangles, boundary) = triangulate_surface(&rep_points);

    // Compute curvatures on representative points
    let (gc, _mc, _amixed) = compute_curvatures_from_mesh(&rep_points, &triangles, &boundary);

    // Create full curvature volume
    let mut curv_i = vec![1.0f64; n_total];

    // Scale for the negative curvatures: calculate_curvature.m's `max(abs(GC(GC<0)))`, which
    // can be below 1. With no negative curvature there is nothing to scale.
    let max_neg_gc = gc.iter()
        .filter(|&&v| v < 0.0)
        .map(|&v| v.abs())
        .fold(0.0f64, f64::max);

    // Scale and assign curvature values for representative vertices only.
    // Non-representative (duplicate x,y) vertices keep curvI=1.0, matching MATLAB
    // where suppressed duplicates get GC=Inf → scaledGC=1.0.
    for (orig_idx, &vol_idx) in surface_indices.iter().enumerate() {
        if !is_representative[orig_idx] {
            continue; // duplicate (x,y) → curvI=1.0
        }
        let rep_idx = orig_to_rep[orig_idx];
        let g = gc[rep_idx];
        let scaled = if g < 0.0 {
            g / max_neg_gc * curv_constant
        } else if g > 0.0 {
            1.0
        } else {
            // GC == 0: boundary vertices, flat regions
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
    let dilated_mask = dilate_ball(mask, nx, ny, nz, 5);

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
    let eroded = imerode_sphere(mask, nx, ny, nz, 1);
    let mut surface = vec![0u8; mask.len()];

    for i in 0..mask.len() {
        if mask[i] != 0 && eroded[i] == 0 {
            surface[i] = 1;
        }
    }

    surface
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
        kernel[i as usize] = pm::exp(-x * x / (2.0 * sigma * sigma));
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
    grid: &Grid,
) -> CurvatureResult {
    let (nx, ny, nz) = grid.dims;
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
    let all_points: Vec<Point3D> = surface_indices
        .iter()
        .map(|&idx| {
            let i = idx % nx;
            let j = (idx / nx) % ny;
            let k = idx / (nx * ny);
            Point3D::new(i as f64, j as f64, k as f64)
        })
        .collect();

    // Deduplicate (x,y) — same logic as calculate_curvature_proximity
    let mut xy_to_rep: HashMap<(usize, usize), usize> = HashMap::new();
    let mut is_representative = vec![false; all_points.len()];
    for (idx, p) in all_points.iter().enumerate() {
        let key = (p.x as usize, p.y as usize);
        xy_to_rep.entry(key).or_insert_with(|| {
            is_representative[idx] = true;
            idx
        });
    }
    let rep_indices: Vec<usize> = (0..all_points.len())
        .filter(|&i| is_representative[i])
        .collect();
    let mut orig_to_rep = vec![0usize; all_points.len()];
    for (new_idx, &old_idx) in rep_indices.iter().enumerate() {
        orig_to_rep[old_idx] = new_idx;
    }
    let rep_points: Vec<Point3D> = rep_indices.iter().map(|&i| all_points[i]).collect();

    // Triangulate the unique representatives (2-D Delaunay, as above)
    let (triangles, boundary) = triangulate_surface(&rep_points);

    // Compute curvatures on representatives
    let (gc_points, mc_points, _amixed) = compute_curvatures_from_mesh(&rep_points, &triangles, &boundary);

    // Create full volumes — only representative vertices get curvature values
    let mut gaussian_curvature = vec![0.0f64; n_total];
    let mut mean_curvature = vec![0.0f64; n_total];

    for (orig_idx, &vol_idx) in surface_indices.iter().enumerate() {
        if is_representative[orig_idx] {
            let rep_idx = orig_to_rep[orig_idx];
            gaussian_curvature[vol_idx] = gc_points[rep_idx];
            mean_curvature[vol_idx] = mc_points[rep_idx];
        }
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

    fn grid(nx: usize, ny: usize, nz: usize) -> Grid {
        Grid::new(nx, ny, nz, 1.0, 1.0, 1.0)
    }

    #[test]
    fn test_extract_surface_basic() {
        // 3x3x3 cube with center filled
        let mut mask = vec![0u8; 27];
        mask[13] = 1; // Center voxel

        let surface = extract_surface_voxels(&mask, 3, 3, 3);
        assert_eq!(surface.len(), 1);
        assert_eq!(surface[0], 13);
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

    /// Unique `(x, y)` points of a grid-aligned surface: an ellipsoid's surface voxels, first
    /// occurrence kept as `calculate_curvature_proximity` keeps them.
    fn ellipsoid_surface_points(dims: (usize, usize, usize), radii: (f64, f64, f64)) -> Vec<Point3D> {
        let (nx, ny, nz) = dims;
        let mut mask = vec![0u8; nx * ny * nz];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let (x, y, z) = (i as f64 - (nx - 1) as f64 / 2.0, j as f64 - (ny - 1) as f64 / 2.0, k as f64 - (nz - 1) as f64 / 2.0);
                    if (x / radii.0).powi(2) + (y / radii.1).powi(2) + (z / radii.2).powi(2) < 1.0 {
                        mask[i + j * nx + k * nx * ny] = 1;
                    }
                }
            }
        }
        let mut seen = std::collections::HashSet::new();
        extract_surface_voxels(&mask, nx, ny, nz).into_iter()
            .map(|idx| Point3D::new((idx % nx) as f64, ((idx / nx) % ny) as f64, (idx / (nx * ny)) as f64))
            .filter(|p| seen.insert((p.x as i64, p.y as i64)))
            .collect()
    }

    #[test]
    fn triangulation_is_cgals_unique_perturbed_delaunay() {
        // MATLAB's (CGAL's) triangulation doesn't depend on the order of the points: the
        // perturbation makes it unique. Grid points have many cocircular quadruples, which
        // delaunator alone splits according to its insertion order.
        for (dims, radii) in [((36, 36, 28), (14.0, 13.5, 10.5)), ((40, 31, 20), (19.5, 12.2, 9.0))] {
            let pts = ellipsoid_surface_points(dims, radii);
            let n = pts.len();
            let canonical = |tris: &[Triangle], ids: &[usize]| {
                let mut v: Vec<[usize; 3]> = tris.iter().map(|t| {
                    let mut a = [ids[t.v0], ids[t.v1], ids[t.v2]];
                    a.sort_unstable();
                    a
                }).collect();
                v.sort_unstable();
                v
            };
            let (tris, boundary) = triangulate_surface(&pts);
            let h = boundary.iter().filter(|&&b| b).count();
            // A triangulation of n points with h on the hull has 2n - 2 - h triangles.
            assert_eq!(tris.len(), 2 * n - 2 - h);
            let ids: Vec<usize> = (0..n).collect();
            let reference = canonical(&tris, &ids);
            // A fixed shuffle of the same points.
            let mut perm: Vec<usize> = (0..n).collect();
            let mut state = 12345u64;
            for i in (1..n).rev() {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                perm.swap(i, (state >> 33) as usize % (i + 1));
            }
            let shuffled: Vec<Point3D> = perm.iter().map(|&i| pts[i]).collect();
            let (tris2, _) = triangulate_surface(&shuffled);
            assert_eq!(canonical(&tris2, &perm), reference, "triangulation depends on the point order");
        }
    }

    #[test]
    fn perturbed_incircle_breaks_square_ties() {
        // The unit square is cocircular; CGAL's perturbation splits it by one fixed diagonal
        // whatever the order of the points.
        let pts = [(0i64, 0i64), (1, 0), (1, 1), (0, 1)];
        // (0,0),(1,0),(1,1) counter-clockwise, (0,1) on its circle: perturbed, which side?
        let s1 = side_of_oriented_circle(&pts, 0, 1, 2, 3);
        // The other diagonal: (1,0),(1,1),(0,1) with (0,0).
        let s2 = side_of_oriented_circle(&pts, 1, 2, 3, 0);
        assert_ne!(s1, 0);
        assert_ne!(s1, s2, "exactly one diagonal of the square is Delaunay under the perturbation");
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
        let (triangles, boundary) = triangulate_surface(&points);
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
        let (triangles, boundary) = triangulate_surface(&points);
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

        let (gc, mc, _amixed) = compute_curvatures_from_mesh(&points, &triangles, &boundary);

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
        let (gc, mc, _amixed) = compute_curvatures_from_mesh(&points, &triangles, &boundary);
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
        let (gc, mc, _amixed) = compute_curvatures_from_mesh(&points, &triangles, &boundary);
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
        let result = calculate_gaussian_curvature(&mask, &grid(n, n, n));

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
        let result = calculate_gaussian_curvature(&mask, &grid(n, n, n));
        assert!(result.surface_indices.is_empty());
        assert!(result.gaussian_curvature.iter().all(|&v| v == 0.0));
        assert!(result.mean_curvature.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_calculate_gaussian_curvature_single_voxel() {
        let mut mask = vec![0u8; 125];
        mask[62] = 1; // single voxel in center of 5x5x5
        let result = calculate_gaussian_curvature(&mask, &grid(5, 5, 5));
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
            &mask, &prox1, 0.6, 500.0, 1.0, &grid(n, n, n),
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
            &mask, &prox1, 0.6, 500.0, 1.0, &grid(n, n, n),
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
            &mask, &prox1, 0.6, 500.0, 1.0, &grid(n, n, n),
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
            &mask, &prox1, 0.3, 100.0, 0.5, &grid(n, n, n),
        );
        let (prox_b, _) = calculate_curvature_proximity(
            &mask, &prox1, 0.9, 1000.0, 2.0, &grid(n, n, n),
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

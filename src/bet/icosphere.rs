//! Icosphere mesh generation

use std::collections::HashMap;

#[cfg(test)]
/// Create a tessellated icosphere mesh
///
/// # Arguments
/// * `subdivisions` - Number of subdivision levels (4 gives 2562 vertices)
///
/// # Returns
/// Tuple of (vertices, faces) where vertices is Vec<[f64; 3]> and faces is Vec<[usize; 3]>
pub fn create_icosphere(subdivisions: usize) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;

    // Initial icosahedron vertices
    let mut vertices: Vec<[f64; 3]> = vec![
        [-1.0,  phi, 0.0], [ 1.0,  phi, 0.0], [-1.0, -phi, 0.0], [ 1.0, -phi, 0.0],
        [ 0.0, -1.0,  phi], [ 0.0,  1.0,  phi], [ 0.0, -1.0, -phi], [ 0.0,  1.0, -phi],
        [ phi, 0.0, -1.0], [ phi, 0.0,  1.0], [-phi, 0.0, -1.0], [-phi, 0.0,  1.0],
    ];

    // Normalize to unit sphere
    let norm = (vertices[0][0].powi(2) + vertices[0][1].powi(2) + vertices[0][2].powi(2)).sqrt();
    for v in vertices.iter_mut() {
        v[0] /= norm;
        v[1] /= norm;
        v[2] /= norm;
    }

    // Initial icosahedron faces
    let mut faces: Vec<[usize; 3]> = vec![
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];

    // Subdivide
    for _ in 0..subdivisions {
        let (new_vertices, new_faces) = subdivide_icosphere(&vertices, &faces);
        vertices = new_vertices;
        faces = new_faces;
    }

    (vertices, faces)
}

#[cfg(test)]
/// Subdivide each triangle into 4 triangles
fn subdivide_icosphere(vertices: &[[f64; 3]], faces: &[[usize; 3]]) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let mut new_vertices: Vec<[f64; 3]> = vertices.to_vec();
    let mut edge_midpoints: HashMap<(usize, usize), usize> = HashMap::new();
    let mut new_faces: Vec<[usize; 3]> = Vec::with_capacity(faces.len() * 4);

    let mut get_midpoint = |i1: usize, i2: usize, verts: &mut Vec<[f64; 3]>| -> usize {
        let key = if i1 < i2 { (i1, i2) } else { (i2, i1) };
        if let Some(&idx) = edge_midpoints.get(&key) {
            return idx;
        }

        let v1 = verts[i1];
        let v2 = verts[i2];
        let mut mid = [
            (v1[0] + v2[0]) / 2.0,
            (v1[1] + v2[1]) / 2.0,
            (v1[2] + v2[2]) / 2.0,
        ];

        // Normalize to unit sphere
        let norm = (mid[0].powi(2) + mid[1].powi(2) + mid[2].powi(2)).sqrt();
        mid[0] /= norm;
        mid[1] /= norm;
        mid[2] /= norm;

        let idx = verts.len();
        verts.push(mid);
        edge_midpoints.insert(key, idx);
        idx
    };

    for &[v0, v1, v2] in faces {
        let m01 = get_midpoint(v0, v1, &mut new_vertices);
        let m12 = get_midpoint(v1, v2, &mut new_vertices);
        let m20 = get_midpoint(v2, v0, &mut new_vertices);

        new_faces.push([v0, m01, m20]);
        new_faces.push([v1, m12, m01]);
        new_faces.push([v2, m20, m12]);
        new_faces.push([m01, m12, m20]);
    }

    (new_vertices, new_faces)
}

/// FSL-BET2's `make_mesh_from_icosa(n, m)` (Bostrix/FSL-BET2 d5acd7f, `meshclass/mesh.cpp`):
/// its icosahedron (vertex coordinates `tau = 0.8506508084`, `one = 0.5257311121`, in its
/// orientation and order), every triangle's last two vertices swapped, then `n − 1` passes of
/// `Mesh::retessellate` (each triangle `(v0, v1, v2)` split at its edge midpoints `p0 = (v1+v2)/2`,
/// `p1 = (v0+v2)/2`, `p2 = (v0+v1)/2` into `(p2, p0, p1)`, `(p1, v0, p2)`, `(p0, v2, p1)`,
/// `(p2, v1, p0)`, a midpoint shared with an earlier triangle reused) with every point scaled
/// back onto the unit sphere by `1/norm` after each pass. `n = 5` (bet2's) gives 2562 vertices.
pub fn bet2_icosa(n: usize) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    const TAU: f64 = 0.8506508084;
    const ONE: f64 = 0.5257311121;
    // ZA ZB ZC ZD YA YB YC YD XA XB XC XD
    let mut pts: Vec<[f64; 3]> = vec![
        [TAU, ONE, 0.0], [-TAU, ONE, 0.0], [-TAU, -ONE, 0.0], [TAU, -ONE, 0.0],
        [ONE, 0.0, TAU], [ONE, 0.0, -TAU], [-ONE, 0.0, -TAU], [-ONE, 0.0, TAU],
        [0.0, TAU, ONE], [0.0, -TAU, ONE], [0.0, -TAU, -ONE], [0.0, TAU, -ONE],
    ];
    let (za, zb, zc, zd, ya, yb, yc, yd, xa, xb, xc, xd) = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
    let mut tris: Vec<[usize; 3]> = vec![
        [ya, xa, yd], [ya, yd, xb], [yb, yc, xd], [yb, xc, yc], [za, ya, zd],
        [za, zd, yb], [zc, yd, zb], [zc, zb, yc], [xa, za, xd], [xa, xd, zb],
        [xb, xc, zd], [xb, zc, xc], [xa, ya, za], [xd, za, yb], [ya, xb, zd],
        [yb, zd, xc], [yd, xa, zb], [yc, zb, xd], [yd, zc, xb], [yc, xc, zc],
    ];
    for t in tris.iter_mut() {
        t.swap(1, 2); // Triangle::swap
    }
    for _ in 1..n {
        let mut added: HashMap<[u64; 3], usize> = HashMap::new();
        let mut new_tris = Vec::with_capacity(tris.len() * 4);
        let mid = |a: [f64; 3], b: [f64; 3]| [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, (a[2] + b[2]) / 2.0];
        for &[v0, v1, v2] in &tris {
            let pt0 = mid(pts[v1], pts[v2]);
            let pt2 = mid(pts[v0], pts[v1]);
            let pt1 = mid(pts[v0], pts[v2]);
            let mut get = |pt: [f64; 3], pts: &mut Vec<[f64; 3]>| -> usize {
                // bet2 matches an earlier midpoint to 1e-8; the same edge gives the same bits.
                let key = [pt[0].to_bits(), pt[1].to_bits(), pt[2].to_bits()];
                *added.entry(key).or_insert_with(|| {
                    pts.push(pt);
                    pts.len() - 1
                })
            };
            let p0 = get(pt0, &mut pts);
            let p1 = get(pt1, &mut pts);
            let p2 = get(pt2, &mut pts);
            new_tris.push([p2, p0, p1]);
            new_tris.push([p1, v0, p2]);
            new_tris.push([p0, v2, p1]);
            new_tris.push([p2, v1, p0]);
        }
        tris = new_tris;
        for p in pts.iter_mut() {
            let norm = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            let inv = 1.0 / norm;
            *p = [p[0] * inv, p[1] * inv, p[2] * inv];
        }
    }
    (pts, tris)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_icosphere_subdivisions() {
        // subdivisions=0: 12 vertices, 20 faces
        let (v0, f0) = create_icosphere(0);
        assert_eq!(v0.len(), 12);
        assert_eq!(f0.len(), 20);

        // subdivisions=1: 42 vertices, 80 faces
        let (v1, f1) = create_icosphere(1);
        assert_eq!(v1.len(), 42);
        assert_eq!(f1.len(), 80);

        // subdivisions=4: 2562 vertices
        let (v4, f4) = create_icosphere(4);
        assert_eq!(v4.len(), 2562);
        assert_eq!(f4.len(), 5120);
    }

    #[test]
    fn test_vertices_on_unit_sphere() {
        let (vertices, _) = create_icosphere(2);
        for v in vertices {
            let norm = (v[0].powi(2) + v[1].powi(2) + v[2].powi(2)).sqrt();
            assert!((norm - 1.0).abs() < 1e-10, "Vertex not on unit sphere: norm = {}", norm);
        }
    }
}

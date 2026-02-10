//! Icosphere mesh generation

use std::collections::HashMap;

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

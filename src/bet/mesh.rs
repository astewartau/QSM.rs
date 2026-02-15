//! Mesh utilities for BET

/// Build neighbor matrix for vectorized neighbor lookups
///
/// Returns (neighbor_matrix, neighbor_counts) where:
/// - neighbor_matrix[i] contains the indices of vertex i's neighbors (padded with usize::MAX)
/// - neighbor_counts[i] is the number of valid neighbors for vertex i
pub fn build_neighbor_matrix(n_vertices: usize, faces: &[[usize; 3]], max_neighbors: usize) -> (Vec<Vec<usize>>, Vec<usize>) {
    let mut neighbor_lists: Vec<Vec<usize>> = vec![Vec::new(); n_vertices];

    for &[v0, v1, v2] in faces {
        // Add edges v0-v1, v1-v2, v2-v0
        for &(a, b) in &[(v0, v1), (v1, v2), (v2, v0)] {
            if !neighbor_lists[a].contains(&b) {
                neighbor_lists[a].push(b);
            }
            if !neighbor_lists[b].contains(&a) {
                neighbor_lists[b].push(a);
            }
        }
    }

    // Find actual max neighbors
    let actual_max = neighbor_lists.iter().map(|n| n.len()).max().unwrap_or(0);
    let padded_max = actual_max.max(max_neighbors);

    // Build padded matrix
    let mut neighbor_matrix: Vec<Vec<usize>> = Vec::with_capacity(n_vertices);
    let mut neighbor_counts: Vec<usize> = Vec::with_capacity(n_vertices);

    for neighs in neighbor_lists {
        neighbor_counts.push(neighs.len());
        let mut row = neighs;
        row.resize(padded_max, usize::MAX);
        neighbor_matrix.push(row);
    }

    (neighbor_matrix, neighbor_counts)
}

/// Compute outward-pointing normals at each vertex
pub fn compute_vertex_normals(vertices: &[[f64; 3]], faces: &[[usize; 3]]) -> Vec<[f64; 3]> {
    let n_vertices = vertices.len();
    let mut normals: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0]; n_vertices];

    for &[i0, i1, i2] in faces {
        let v0 = vertices[i0];
        let v1 = vertices[i1];
        let v2 = vertices[i2];

        // Edge vectors
        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

        // Face normal (cross product)
        let face_normal = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];

        // Normalize
        let norm = (face_normal[0].powi(2) + face_normal[1].powi(2) + face_normal[2].powi(2)).sqrt();
        let face_normal = if norm > 1e-10 {
            [face_normal[0] / norm, face_normal[1] / norm, face_normal[2] / norm]
        } else {
            [0.0, 0.0, 0.0]
        };

        // Accumulate at vertices
        for &idx in &[i0, i1, i2] {
            normals[idx][0] += face_normal[0];
            normals[idx][1] += face_normal[1];
            normals[idx][2] += face_normal[2];
        }
    }

    // Normalize all vertex normals
    for n in normals.iter_mut() {
        let norm = (n[0].powi(2) + n[1].powi(2) + n[2].powi(2)).sqrt();
        if norm > 1e-10 {
            n[0] /= norm;
            n[1] /= norm;
            n[2] /= norm;
        }
    }

    normals
}

/// Compute mean edge length for vertices in voxel coordinates (converts to mm)
pub fn compute_mean_edge_length(vertices: &[[f64; 3]], faces: &[[usize; 3]], voxel_size: &[f64; 3]) -> f64 {
    let mut total_length = 0.0;
    let mut count = 0;

    for &[i0, i1, i2] in faces {
        // Edge v0-v1
        let dx = (vertices[i1][0] - vertices[i0][0]) * voxel_size[0];
        let dy = (vertices[i1][1] - vertices[i0][1]) * voxel_size[1];
        let dz = (vertices[i1][2] - vertices[i0][2]) * voxel_size[2];
        total_length += (dx*dx + dy*dy + dz*dz).sqrt();

        // Edge v1-v2
        let dx = (vertices[i2][0] - vertices[i1][0]) * voxel_size[0];
        let dy = (vertices[i2][1] - vertices[i1][1]) * voxel_size[1];
        let dz = (vertices[i2][2] - vertices[i1][2]) * voxel_size[2];
        total_length += (dx*dx + dy*dy + dz*dz).sqrt();

        // Edge v2-v0
        let dx = (vertices[i0][0] - vertices[i2][0]) * voxel_size[0];
        let dy = (vertices[i0][1] - vertices[i2][1]) * voxel_size[1];
        let dz = (vertices[i0][2] - vertices[i2][2]) * voxel_size[2];
        total_length += (dx*dx + dy*dy + dz*dz).sqrt();

        count += 3;
    }

    if count > 0 {
        total_length / count as f64
    } else {
        1.0
    }
}

/// Compute mean edge length for vertices already in mm coordinates
pub fn compute_mean_edge_length_mm(vertices_mm: &[[f64; 3]], faces: &[[usize; 3]]) -> f64 {
    let mut total_length = 0.0;
    let mut count = 0;

    for &[i0, i1, i2] in faces {
        // Edge v0-v1
        let dx = vertices_mm[i1][0] - vertices_mm[i0][0];
        let dy = vertices_mm[i1][1] - vertices_mm[i0][1];
        let dz = vertices_mm[i1][2] - vertices_mm[i0][2];
        total_length += (dx*dx + dy*dy + dz*dz).sqrt();

        // Edge v1-v2
        let dx = vertices_mm[i2][0] - vertices_mm[i1][0];
        let dy = vertices_mm[i2][1] - vertices_mm[i1][1];
        let dz = vertices_mm[i2][2] - vertices_mm[i1][2];
        total_length += (dx*dx + dy*dy + dz*dz).sqrt();

        // Edge v2-v0
        let dx = vertices_mm[i0][0] - vertices_mm[i2][0];
        let dy = vertices_mm[i0][1] - vertices_mm[i2][1];
        let dz = vertices_mm[i0][2] - vertices_mm[i2][2];
        total_length += (dx*dx + dy*dy + dz*dz).sqrt();

        count += 3;
    }

    if count > 0 {
        total_length / count as f64
    } else {
        1.0
    }
}

/// Compute distance between two vertices (already in mm)
fn vertex_distance(v1: &[f64; 3], v2: &[f64; 3]) -> f64 {
    let dx = v2[0] - v1[0];
    let dy = v2[1] - v1[1];
    let dz = v2[2] - v1[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Compute self-intersection heuristic by comparing vertex distances
/// between current and original mesh.
///
/// This is based on FSL-BET2's self_intersection method which:
/// 1. Computes mean edge length for both meshes (ml, mlo)
/// 2. For vertex pairs that are currently close (< ml apart), checks if they've
///    gotten significantly closer than they were in the original mesh
/// 3. Accumulates squared differences in normalized distances
///
/// Returns a scalar value where > 4000 indicates likely self-intersection.
/// The threshold of 4000 matches FSL-BET2's self_intersection_threshold.
///
/// Note: Vertices are expected to be in mm coordinates.
pub fn self_intersection_heuristic(
    current_vertices: &[[f64; 3]],
    original_vertices: &[[f64; 3]],
    faces: &[[usize; 3]],
    _voxel_size: &[f64; 3], // kept for API compatibility, not used (vertices are in mm)
) -> f64 {
    if current_vertices.len() != original_vertices.len() {
        return f64::MAX;
    }

    let n = current_vertices.len();

    // Compute mean edge length for normalization (like FSL's ml and mlo)
    // Vertices are in mm, so use the mm version
    let ml = compute_mean_edge_length_mm(current_vertices, faces);
    let mlo = compute_mean_edge_length_mm(original_vertices, faces);

    if ml < 1e-10 || mlo < 1e-10 {
        return f64::MAX;
    }

    let ml_sq = ml * ml;
    let mut intersection = 0.0;

    // FSL compares all vertex pairs, but only counts pairs where current distance < ml
    // This detects when non-adjacent vertices have gotten too close (mesh folding)
    // For efficiency, we sample a subset of pairs for large meshes
    let step = if n > 500 { (n / 500).max(1) } else { 1 };

    for i in (0..n).step_by(step) {
        for j in (i + 1..n).step_by(step) {
            // Current distance squared (vertices already in mm)
            let dx = current_vertices[j][0] - current_vertices[i][0];
            let dy = current_vertices[j][1] - current_vertices[i][1];
            let dz = current_vertices[j][2] - current_vertices[i][2];
            let curr_dist_sq = dx * dx + dy * dy + dz * dz;

            // Only consider pairs that are currently close (< ml apart)
            // This is the key insight from FSL - we're looking for folding
            if curr_dist_sq < ml_sq {
                let curr_dist = curr_dist_sq.sqrt();
                let orig_dist = vertex_distance(&original_vertices[i], &original_vertices[j]);

                // Normalize distances
                let dist = curr_dist / ml;
                let disto = orig_dist / mlo;

                // Accumulate squared difference
                let diff = dist - disto;
                intersection += diff * diff;
            }
        }
    }

    intersection
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bet::icosphere::create_icosphere;

    #[test]
    fn test_neighbor_matrix() {
        let (vertices, faces) = create_icosphere(1);
        let (neighbor_matrix, neighbor_counts) = build_neighbor_matrix(vertices.len(), &faces, 6);

        assert_eq!(neighbor_matrix.len(), vertices.len());
        assert_eq!(neighbor_counts.len(), vertices.len());

        // Each vertex should have at least 1 neighbor
        for &count in &neighbor_counts {
            assert!(count >= 1);
        }
    }

    #[test]
    fn test_vertex_normals() {
        let (vertices, faces) = create_icosphere(1);
        let normals = compute_vertex_normals(&vertices, &faces);

        assert_eq!(normals.len(), vertices.len());

        // Normals should be unit length and point outward (same direction as vertex)
        for (v, n) in vertices.iter().zip(normals.iter()) {
            let norm = (n[0].powi(2) + n[1].powi(2) + n[2].powi(2)).sqrt();
            assert!((norm - 1.0).abs() < 1e-6, "Normal not unit length");

            // Dot product with vertex should be positive (outward pointing)
            let dot = v[0] * n[0] + v[1] * n[1] + v[2] * n[2];
            assert!(dot > 0.9, "Normal not pointing outward");
        }
    }

    #[test]
    fn test_neighbor_matrix_symmetry() {
        // Neighbor relationships must be symmetric: if A neighbors B then B neighbors A
        let (vertices, faces) = create_icosphere(2);
        let n = vertices.len();
        let (neighbor_matrix, neighbor_counts) = build_neighbor_matrix(n, &faces, 6);

        for i in 0..n {
            for j in 0..neighbor_counts[i] {
                let ni = neighbor_matrix[i][j];
                // ni should have i in its neighbor list
                let found = (0..neighbor_counts[ni]).any(|k| neighbor_matrix[ni][k] == i);
                assert!(found, "Asymmetric neighbors: {} -> {} but not reverse", i, ni);
            }
        }
    }

    #[test]
    fn test_neighbor_matrix_single_triangle() {
        // Minimal mesh: one triangle with 3 vertices
        let faces = vec![[0, 1, 2]];
        let (neighbor_matrix, neighbor_counts) = build_neighbor_matrix(3, &faces, 6);

        assert_eq!(neighbor_counts.len(), 3);
        // Each vertex in a single triangle should have exactly 2 neighbors
        for &count in &neighbor_counts {
            assert_eq!(count, 2);
        }

        // Padding should be usize::MAX
        for row in &neighbor_matrix {
            for &val in row.iter().skip(2) {
                assert_eq!(val, usize::MAX);
            }
        }
    }

    #[test]
    fn test_neighbor_matrix_max_neighbors_padding() {
        // When max_neighbors > actual neighbors, rows should be padded
        let faces = vec![[0, 1, 2]];
        let (neighbor_matrix, _neighbor_counts) = build_neighbor_matrix(3, &faces, 10);

        // Each row should have at least 10 entries (max_neighbors)
        for row in &neighbor_matrix {
            assert!(row.len() >= 10);
        }
    }

    #[test]
    fn test_compute_mean_edge_length_voxel() {
        // A unit cube triangle (vertices in voxel coords, voxel_size = [2, 2, 2])
        let vertices: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let faces = vec![[0, 1, 2]];
        let voxel_size = [2.0, 2.0, 2.0];

        let mel = compute_mean_edge_length(&vertices, &faces, &voxel_size);
        // Edge lengths in mm: 2.0, 2.0, 2*sqrt(2) = 2.828...
        // Mean = (2 + 2 + 2.828) / 3 = 2.276
        assert!(mel > 2.0 && mel < 3.0, "mean edge length = {}", mel);
        assert!(mel.is_finite());
    }

    #[test]
    fn test_compute_mean_edge_length_mm_unit_triangle() {
        let vertices_mm: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let faces = vec![[0, 1, 2]];

        let mel = compute_mean_edge_length_mm(&vertices_mm, &faces);
        // Edge lengths: 1.0, 1.0, sqrt(2) = 1.414
        // Mean = (1 + 1 + 1.414) / 3 = 1.138
        let expected = (1.0 + 1.0 + 2.0_f64.sqrt()) / 3.0;
        assert!((mel - expected).abs() < 1e-10, "mel={}, expected={}", mel, expected);
    }

    #[test]
    fn test_compute_mean_edge_length_mm_empty_faces() {
        let vertices_mm: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0]];
        let faces: Vec<[usize; 3]> = vec![];

        let mel = compute_mean_edge_length_mm(&vertices_mm, &faces);
        // With no faces, should return fallback of 1.0
        assert!((mel - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_mean_edge_length_empty_faces() {
        let vertices: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0]];
        let faces: Vec<[usize; 3]> = vec![];
        let voxel_size = [1.0, 1.0, 1.0];

        let mel = compute_mean_edge_length(&vertices, &faces, &voxel_size);
        assert!((mel - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_mean_edge_length_mm_icosphere() {
        // Icosphere with subdivision 2, scaled by radius 5.0
        let (unit_verts, faces) = create_icosphere(2);
        let radius = 5.0;
        let vertices_mm: Vec<[f64; 3]> = unit_verts
            .iter()
            .map(|v| [v[0] * radius, v[1] * radius, v[2] * radius])
            .collect();

        let mel = compute_mean_edge_length_mm(&vertices_mm, &faces);
        assert!(mel > 0.0 && mel.is_finite(), "mel should be positive and finite, got {}", mel);
        // For a sphere of radius 5, edges should be a fraction of the radius
        assert!(mel < radius, "Edge length should be smaller than radius");
    }

    #[test]
    fn test_vertex_normals_degenerate_face() {
        // Include a degenerate triangle (zero area) alongside a valid one
        let vertices: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.0], // collinear with v0 and v1
        ];
        let faces = vec![
            [0, 1, 2],  // valid triangle
            [0, 1, 3],  // degenerate (collinear points)
        ];

        let normals = compute_vertex_normals(&vertices, &faces);
        assert_eq!(normals.len(), 4);
        // The valid triangle's normal contribution should still produce finite normals
        for n in &normals {
            assert!(n[0].is_finite() && n[1].is_finite() && n[2].is_finite());
        }
    }

    #[test]
    fn test_vertex_normals_icosphere_subdivision_2() {
        // Higher subdivision level
        let (vertices, faces) = create_icosphere(2);
        let normals = compute_vertex_normals(&vertices, &faces);

        assert_eq!(normals.len(), vertices.len());
        for (v, n) in vertices.iter().zip(normals.iter()) {
            let norm = (n[0].powi(2) + n[1].powi(2) + n[2].powi(2)).sqrt();
            assert!((norm - 1.0).abs() < 1e-6, "Normal not unit length: {}", norm);

            let dot = v[0] * n[0] + v[1] * n[1] + v[2] * n[2];
            assert!(dot > 0.9, "Normal not pointing outward, dot={}", dot);
        }
    }

    #[test]
    fn test_vertex_distance() {
        let v1 = [0.0, 0.0, 0.0];
        let v2 = [3.0, 4.0, 0.0];
        let d = vertex_distance(&v1, &v2);
        assert!((d - 5.0).abs() < 1e-10, "Expected 5.0, got {}", d);

        // Same point
        let d0 = vertex_distance(&v1, &v1);
        assert!((d0 - 0.0).abs() < 1e-10);

        // 3D case
        let v3 = [1.0, 2.0, 3.0];
        let v4 = [4.0, 6.0, 3.0];
        let d34 = vertex_distance(&v3, &v4);
        let expected = (9.0 + 16.0 + 0.0_f64).sqrt();
        assert!((d34 - expected).abs() < 1e-10);
    }

    #[test]
    fn test_self_intersection_heuristic_no_intersection() {
        // An icosphere that hasn't changed should have very low score
        let (vertices, faces) = create_icosphere(1);
        let radius = 5.0;
        let scaled: Vec<[f64; 3]> = vertices
            .iter()
            .map(|v| [v[0] * radius, v[1] * radius, v[2] * radius])
            .collect();

        let score = self_intersection_heuristic(&scaled, &scaled, &faces, &[1.0, 1.0, 1.0]);
        // Identical meshes should yield 0 (no differences)
        assert!((score - 0.0).abs() < 1e-6, "Expected ~0, got {}", score);
    }

    #[test]
    fn test_self_intersection_heuristic_expanded_mesh() {
        // Slightly expand the mesh uniformly -- should not indicate self-intersection
        let (vertices, faces) = create_icosphere(1);
        let radius = 5.0;
        let original: Vec<[f64; 3]> = vertices
            .iter()
            .map(|v| [v[0] * radius, v[1] * radius, v[2] * radius])
            .collect();

        let expanded: Vec<[f64; 3]> = vertices
            .iter()
            .map(|v| [v[0] * radius * 1.1, v[1] * radius * 1.1, v[2] * radius * 1.1])
            .collect();

        let score = self_intersection_heuristic(&expanded, &original, &faces, &[1.0, 1.0, 1.0]);
        assert!(score.is_finite(), "Score should be finite, got {}", score);
        // Uniform expansion should have a very low score (no folding)
        assert!(score < 4000.0, "Uniform expansion should not trigger self-intersection, score={}", score);
    }

    #[test]
    fn test_self_intersection_heuristic_mismatched_vertices() {
        let faces = vec![[0, 1, 2]];
        let v1: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let v2: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]; // wrong size

        let score = self_intersection_heuristic(&v1, &v2, &faces, &[1.0, 1.0, 1.0]);
        assert_eq!(score, f64::MAX);
    }

    #[test]
    fn test_self_intersection_heuristic_collapsed_mesh() {
        // Collapse all vertices to the center -- extreme folding
        let (vertices, faces) = create_icosphere(1);
        let radius = 5.0;
        let original: Vec<[f64; 3]> = vertices
            .iter()
            .map(|v| [v[0] * radius, v[1] * radius, v[2] * radius])
            .collect();

        // Collapse to a single point
        let collapsed: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0]; vertices.len()];

        let score = self_intersection_heuristic(&collapsed, &original, &faces, &[1.0, 1.0, 1.0]);
        // All vertices at same point means ml ~ 0, which should give MAX
        assert_eq!(score, f64::MAX, "Collapsed mesh should have MAX score");
    }

    #[test]
    fn test_compute_mean_edge_length_anisotropic_voxel() {
        // Test with anisotropic voxel sizes
        let vertices: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let faces = vec![[0, 1, 2]];
        let voxel_size = [1.0, 1.0, 3.0]; // z is 3x larger

        let mel = compute_mean_edge_length(&vertices, &faces, &voxel_size);
        // Edge 0->1: (1*1, 0, 0) -> length 1.0
        // Edge 1->2: (-1*1, 0, 1*3) -> length sqrt(1+9) = sqrt(10) = 3.162
        // Edge 2->0: (0, 0, -1*3) -> length 3.0
        // Mean = (1.0 + 3.162 + 3.0) / 3 = 2.387
        let expected = (1.0 + 10.0_f64.sqrt() + 3.0) / 3.0;
        assert!((mel - expected).abs() < 1e-6, "mel={}, expected={}", mel, expected);
    }
}

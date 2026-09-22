//! 6-connected component labelling for binary masks.
//!
//! Matches `scipy.ndimage.label(mask)` with its default structuring element (connectivity 1,
//! i.e. face neighbours only), which is what the reference implementations of the ported
//! deep-learning post-processing use.

/// Label the 6-connected foreground components of `mask`.
///
/// Returns `(labels, sizes)`: `labels[i]` is 0 for background or a 1-based component id, and
/// `sizes[k]` is the voxel count of component `k` (`sizes[0]` is unused and always 0).
pub(crate) fn label_components(
    mask: &[u8],
    dims: (usize, usize, usize),
) -> (Vec<u32>, Vec<usize>) {
    let (nx, ny, nz) = dims;
    let nxy = nx * ny;
    let mut label = vec![0u32; mask.len()];
    let mut sizes = vec![0usize]; // index 0 = background
    let mut stack = Vec::new();
    for seed in 0..mask.len() {
        if mask[seed] == 0 || label[seed] != 0 {
            continue;
        }
        let id = sizes.len() as u32;
        let mut size = 0;
        label[seed] = id;
        stack.push(seed);
        while let Some(i) = stack.pop() {
            size += 1;
            let (x, y, z) = (i % nx, (i / nx) % ny, i / nxy);
            let mut visit = |j: usize| {
                if mask[j] != 0 && label[j] == 0 {
                    label[j] = id;
                    stack.push(j);
                }
            };
            if x > 0 { visit(i - 1); }
            if x + 1 < nx { visit(i + 1); }
            if y > 0 { visit(i - nx); }
            if y + 1 < ny { visit(i + nx); }
            if z > 0 { visit(i - nxy); }
            if z + 1 < nz { visit(i + nxy); }
        }
        sizes.push(size);
    }
    (label, sizes)
}

/// Keep only the largest 6-connected component of `mask`.
///
/// Mirrors lab2im's `get_largest_connected_component`: an all-background mask is returned
/// unchanged, and ties go to the component encountered first in the raster scan.
pub(crate) fn largest_component(mask: &[u8], dims: (usize, usize, usize)) -> Vec<u8> {
    let (label, sizes) = label_components(mask, dims);
    if sizes.len() <= 1 {
        return mask.to_vec();
    }
    let mut best = 1usize;
    for (id, &size) in sizes.iter().enumerate().skip(2) {
        if size > sizes[best] {
            best = id;
        }
    }
    let best = best as u32;
    label.iter().map(|&l| (l == best) as u8).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn largest_component_keeps_only_the_biggest_blob() {
        let dims = (6, 1, 1);
        // two components: {0,1,2} and {4}
        let mask = [1u8, 1, 1, 0, 1, 0];
        assert_eq!(largest_component(&mask, dims), vec![1, 1, 1, 0, 0, 0]);
    }

    #[test]
    fn diagonal_neighbours_are_separate_components() {
        // 6-connectivity: (0,0,0) and (1,1,0) touch only at a corner.
        let dims = (2, 2, 1);
        let mut mask = vec![0u8; 4];
        mask[0] = 1; // (0,0,0)
        mask[3] = 1; // (1,1,0)
        let (_, sizes) = label_components(&mask, dims);
        assert_eq!(sizes.len() - 1, 2, "expected two separate components");
    }

    #[test]
    fn empty_mask_is_returned_unchanged() {
        let mask = vec![0u8; 8];
        assert_eq!(largest_component(&mask, (2, 2, 2)), mask);
    }
}

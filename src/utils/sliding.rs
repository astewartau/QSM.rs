//! Sliding-window tiling for fixed-patch networks.
//!
//! The SNU-LIST nets (χ-sepnet, R2PRIMEnet) are run as an overlapping sliding window with a
//! 0.75 stride and a final flush patch at the far edge, averaging wherever patches overlap.
//! This is the start-position enumeration they share; each model keeps its own patch
//! packing, since their channel counts differ.

/// Overlap stride as a fraction of the patch, matching the authors' inference.
pub const STRIDE_FRACTION: f64 = 0.75;

/// Start positions along one axis: `0, 0.75·patch, …`, plus a final flush start at
/// `size - patch` so the far edge is always covered. A volume at or under the patch size
/// yields a single start (the caller pads).
pub fn starts(size: usize, patch: usize) -> Vec<usize> {
    if size <= patch {
        return vec![0];
    }
    let step = ((patch as f64 * STRIDE_FRACTION) as usize).max(1);
    let mut s: Vec<usize> = (0..=size - patch).step_by(step).collect();
    if *s.last().unwrap() != size - patch {
        s.push(size - patch);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiles_and_flushes() {
        // At or under the patch: one (padded) patch.
        assert_eq!(starts(100, 192), vec![0]);
        assert_eq!(starts(192, 192), vec![0]);
        // Over: stride 144, plus a flush patch ending exactly at the far edge.
        assert_eq!(starts(205, 192), vec![0, 13]);
        assert_eq!(starts(480, 192), vec![0, 144, 288]);
        assert_eq!(starts(500, 192), vec![0, 144, 288, 308]);
        // The flush start is never duplicated when the stride already lands on it.
        assert_eq!(starts(336, 192), vec![0, 144]);
    }

    /// Every voxel is covered by at least one patch — the averaging denominator is never 0.
    #[test]
    fn covers_every_voxel() {
        for size in [64usize, 192, 193, 205, 400, 500] {
            for patch in [64usize, 128, 192] {
                let padded = size.max(patch);
                let mut hits = vec![0u32; padded];
                for s in starts(padded, patch) {
                    for v in hits.iter_mut().skip(s).take(patch) {
                        *v += 1;
                    }
                }
                assert!(hits.iter().all(|&h| h > 0), "gap at size {size} patch {patch}");
            }
        }
    }
}

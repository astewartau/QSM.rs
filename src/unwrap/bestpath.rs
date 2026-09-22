//! Best-path phase unwrapping (3D-SRNCP).
//!
//! Sorting by reliability following a non-continuous path, after
//! H. Abdul-Rahman, M. Gdeisat, D. Burton and M. Lalor, "Fast three-dimensional
//! phase-unwrapping algorithm based on sorting by reliability following a
//! non-continuous path", Proc. SPIE 5856 (2005) 32–40,
//! <https://doi.org/10.1364/AO.46.006623>.
//!
//! The method is a path-following unwrapper that never actually follows a path
//! through the volume. Instead:
//!
//! 1. Every voxel gets a **reliability** — the summed squared second difference
//!    of the wrapped phase over the 13 directions of the 26-neighbourhood. A
//!    *low* value means the local phase is smooth and the voxel is trustworthy.
//! 2. Every 6-connected pair of in-mask voxels becomes an **edge**, whose
//!    reliability is the sum of its two endpoints'. Each edge also records the
//!    integer wrap count implied by the phase difference across it.
//! 3. The edges are **sorted** by reliability, most reliable first, and walked in
//!    that order. Each edge joins the two voxel groups at its ends, offsetting
//!    one group by the wraps the edge implies. Edges whose endpoints already share
//!    a group are skipped.
//!
//! So the unwrapping order is global rather than local: the most trustworthy
//! phase differences in the whole volume are committed first, and noisy regions
//! are reached last, through whatever reliable route survives. That is what makes
//! it robust to residues compared with a seeded flood fill.
//!
//! # Relationship to ROMEO
//!
//! [`calculate_weights_bestpath`](super::romeo::calculate_weights_bestpath) also
//! implements Abdul-Rahman reliabilities, but as *edge weights fed into ROMEO's
//! minimum-spanning-tree growing* (matching ROMEO.jl's `calculateweights_bestpath`,
//! including its inversion to `1/sqrt(D²)` and quantisation to `u8`). That is
//! ROMEO with different weights; this module is the original algorithm, which
//! merges groups over a globally sorted edge list rather than growing a region
//! from a seed.
//!
//! # Deviations from the reference implementation
//!
//! The algorithm here is a reimplementation, informed by the C reference that
//! scikit-image distributes as `skimage/restoration/unwrap_3d_ljmu.c` under the
//! BSD-3-Clause licence (see `THIRD_PARTY_NOTICES.md`). Three deliberate
//! departures:
//!
//! - **Group merging** uses a union-find with integer potentials
//!   ([`UnionFind`](crate::utils::union_find::UnionFind)) rather than the
//!   reference's explicit per-group linked lists. The merge decisions and the
//!   resulting wrap counts are the same; only the bookkeeping cost differs
//!   (O(α) per edge instead of O(size of the smaller group)).
//! - **Masked voxels keep their original wrapped phase** on output. The reference
//!   overwrites them with the minimum of the unwrapped phase, which is a display
//!   convenience; leaving them alone matches [`unwrap_romeo`](super::unwrap_romeo)
//!   and the rest of this crate.
//! - **Wrap-around (periodic) boundary options are not implemented.** They exist
//!   in the reference for interferometry; MRI volumes are not periodic.
//!
//! # An upstream quirk, deliberately preserved
//!
//! The reference has an `extend_mask` step that reads as though it should exclude
//! mask-adjacent voxels from the reliability calculation — it tests whether all 26
//! neighbours are unmasked. It does not: `extended_mask` is `calloc`'d to
//! `NOMASK`, and the function only ever *assigns* `NOMASK`, never `MASK`, so the
//! buffer is uniformly `NOMASK` and the guard that reads it is always true. The
//! mask therefore plays no part in reliability at all; it only decides which edges
//! exist.
//!
//! This module reproduces the effective behaviour, not the apparent intent, and
//! that is a deliberate choice rather than bug-compatibility for its own sake.
//! Implementing the intent — stand-in reliabilities for mask-adjacent voxels —
//! measurably unwraps *worse* near the ROI boundary on noisy data, because it
//! throws away a graded quality signal and replaces it with a shuffle. A boundary
//! voxel whose neighbours lie in noise already earns a poor reliability on its own
//! and sorts late for the right reason.
//!
//! The absolute 2π offset of each connected component is arbitrary — it is fixed
//! only up to the choice of which voxel ends up as the component's representative.
//! Set [`BestPathParams::correct_global`] for a defined choice.
//!
//! # Verification
//!
//! Checked against `skimage.restoration.unwrap_phase` (scikit-image 0.26.0) on
//! 48×48×32 synthetic fields — smooth, noisy, and masked to an irregular ROI with
//! holes. Every voxel matched bit-for-bit, across several of scikit-image's own rng
//! seeds. The two diverge only once the added noise approaches ±π, where the wrap
//! counts are genuinely ambiguous: there they disagree on 0.07% of voxels, which is
//! the same margin by which scikit-image disagrees with *itself* between rng seeds
//! (0.06%).
//!
//! It was this comparison that turned up the `extend_mask` quirk above. Following
//! the apparent intent cost an order of magnitude in accuracy near the ROI boundary
//! at high noise — 0.51% of wrap counts wrong against ground truth, versus 0.06%
//! for the reference's actual behaviour, which this module now reproduces exactly.

use crate::grid::Grid;
use crate::utils::union_find::UnionFind;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

const PI: f64 = std::f64::consts::PI;
const TWO_PI: f64 = 2.0 * PI;

/// Parameters for [`unwrap_bestpath`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
pub struct BestPathParams {
    /// Subtract the median wrap count from the result so the unwrapped phase
    /// stays centred near the input range (default: false).
    ///
    /// Without it, each connected component carries an arbitrary integer multiple
    /// of 2π. Mirrors [`RomeoParams::correct_global`](super::RomeoParams::correct_global),
    /// which defaults to false for the same reason: it is a convention, not a
    /// correction, and multi-echo callers usually impose their own.
    pub correct_global: bool,
    /// Seed for the pseudo-random reliabilities given to voxels on the volume
    /// border (default: 0x05EEDB4D).
    ///
    /// Those voxels have no complete 26-neighbourhood and so no meaningful second
    /// difference. The reference assigns them `rand()`, which puts them far below
    /// every real reliability and shuffles them among themselves so the merge
    /// order along the volume faces carries no raster bias. Reproduced here with a
    /// fixed default seed, so output is deterministic.
    pub seed: u64,
}

impl Default for BestPathParams {
    fn default() -> Self {
        Self {
            correct_global: false,
            seed: 0x05EE_DB4D,
        }
    }
}

/// The 13 direction vectors of the 26-neighbourhood, one per antipodal pair.
///
/// Which representative of a pair is used does not matter: the second difference
/// `wrap(p[i-d] - p[i]) - wrap(p[i] - p[i+d])` is invariant under `d -> -d`,
/// because `wrap` is odd.
const DIRECTIONS: [(i32, i32, i32); 13] = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (-1, 1, 0),
    (1, 0, 1),
    (-1, 0, 1),
    (0, 1, 1),
    (0, -1, 1),
    (1, 1, 1),
    (-1, 1, 1),
    (1, -1, 1),
    (-1, -1, 1),
];

/// One edge of the voxel graph, awaiting its turn in the sorted merge.
struct Edge {
    /// Summed reliability of the two endpoints; lower is better.
    reliability: f64,
    a: u32,
    b: u32,
    /// Wraps implied across the edge, as `find_wrap(phase[a], phase[b])`.
    wraps: i8,
}

/// Wrap a value already known to lie within ±3π into `[-π, π]`.
///
/// Matches the reference's `wrap()`, which is a single conditional rather than a
/// general modulo. Every argument here is a difference of two wrapped phases, so
/// its magnitude is at most 2π and one correction always suffices.
#[inline]
fn wrap_pi(v: f64) -> f64 {
    if v > PI {
        v - TWO_PI
    } else if v < -PI {
        v + TWO_PI
    } else {
        v
    }
}

/// Number of 2π that separates two wrapped phases, as the reference's `find_wrap`.
///
/// The returned `w` satisfies `phase_b + 2π·(-w) ≈ phase_a` in the sense that the
/// residual difference lies in `[-π, π]`.
#[inline]
fn find_wrap(phase_a: f64, phase_b: f64) -> i8 {
    let d = phase_a - phase_b;
    if d > PI {
        -1
    } else if d < -PI {
        1
    } else {
        0
    }
}

/// SplitMix64, used to reproduce the reference's `rand()` initialisation
/// deterministically and independently of iteration order.
#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A pseudo-random stand-in reliability, uniform over `[0, 2^31)`.
///
/// The range matches C's `RAND_MAX`, which is what the reference draws from. Real
/// reliabilities are bounded by `13·(2π)² ≈ 513`, so these effectively always sort
/// last — exactly as in the reference, including its vanishingly rare exceptions.
#[inline]
fn stand_in_reliability(seed: u64, index: usize) -> f64 {
    (splitmix64(seed ^ (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 33) as f64
}

/// Unwrap a 3D phase volume with the best-path (3D-SRNCP) algorithm.
///
/// # Arguments
/// * `phase` — wrapped phase in radians, `nx * ny * nz`, expected in `[-π, π]`
/// * `mask` — binary mask, non-zero to process
/// * `params` — see [`BestPathParams`]
/// * `grid` — volume dimensions (voxel sizes are unused: the reliability is
///   defined on the voxel lattice, not in physical space)
///
/// # Returns
/// Unwrapped phase, same length as `phase`. Voxels outside the mask are returned
/// unchanged.
///
/// Each connected in-mask component is unwrapped independently and carries its own
/// arbitrary multiple of 2π; there is no information linking components across a
/// gap in the mask.
pub fn unwrap_bestpath(
    phase: &[f64],
    mask: &[u8],
    params: &BestPathParams,
    grid: &Grid,
) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let n_total = nx * ny * nz;
    assert_eq!(phase.len(), n_total, "phase length must match grid dimensions");
    assert_eq!(mask.len(), n_total, "mask length must match grid dimensions");
    assert!(
        n_total <= u32::MAX as usize,
        "best-path unwrapping indexes voxels by u32 (volume has {n_total} voxels)"
    );

    let mut unwrapped = phase.to_vec();
    if n_total == 0 {
        return unwrapped;
    }

    let reliability = compute_reliability(phase, params.seed, nx, ny, nz);
    let mut edges = build_edges(phase, mask, &reliability, nx, ny, nz);
    sort_edges(&mut edges);

    // Walk the edges best-first, merging the groups they join.
    let mut uf = UnionFind::new(n_total);
    for edge in &edges {
        uf.union_with_delta(edge.a as usize, edge.b as usize, -(edge.wraps as i32));
    }

    for i in 0..n_total {
        if mask[i] != 0 {
            let (_, wraps) = uf.find(i);
            unwrapped[i] = phase[i] + TWO_PI * wraps as f64;
        }
    }

    if params.correct_global {
        correct_global_offset(&mut unwrapped, mask);
    }

    unwrapped
}

/// Per-voxel reliability: the summed squared second difference of the wrapped
/// phase over the 13 directions of the 26-neighbourhood. Lower is better.
///
/// Every voxel with a complete neighbourhood — that is, every voxel off the volume
/// border — gets a real value, whether or not it or its neighbours are in the mask.
/// Border voxels keep the pseudo-random stand-in, which sorts them last. See the
/// module docs on `extend_mask` for why the mask is not consulted here.
fn compute_reliability(
    phase: &[f64],
    seed: u64,
    nx: usize,
    ny: usize,
    nz: usize,
) -> Vec<f64> {
    let n_total = nx * ny * nz;
    let mut reliability: Vec<f64> = (0..n_total).map(|i| stand_in_reliability(seed, i)).collect();

    if nx < 3 || ny < 3 || nz < 3 {
        // No voxel has a complete 26-neighbourhood.
        return reliability;
    }

    let nxy = nx * ny;
    let strides: [isize; 13] = DIRECTIONS.map(|(dx, dy, dz)| {
        dx as isize + dy as isize * nx as isize + dz as isize * nxy as isize
    });

    crate::maybe_par_chunks_mut!(reliability, nxy)
        .enumerate()
        .for_each(|(k, plane)| {
            if k == 0 || k == nz - 1 {
                return;
            }
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    let idx = k * nxy + j * nx + i;
                    let p = phase[idx];
                    let mut d2 = 0.0;
                    for &s in &strides {
                        let back = (idx as isize - s) as usize;
                        let fwd = (idx as isize + s) as usize;
                        let d = wrap_pi(phase[back] - p) - wrap_pi(p - phase[fwd]);
                        d2 += d * d;
                    }
                    plane[j * nx + i] = d2;
                }
            }
        });

    reliability
}

/// Build the 6-connected edge list, skipping any pair with a masked endpoint.
///
/// Emitted in x, then y, then z order, matching the reference's
/// `horizontalEDGEs`/`verticalEDGEs`/`normalEDGEs` — the order decides how equal
/// reliabilities are broken in [`sort_edges`].
fn build_edges(
    phase: &[f64],
    mask: &[u8],
    reliability: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
) -> Vec<Edge> {
    let nxy = nx * ny;
    let in_mask = mask.iter().filter(|&&m| m != 0).count();
    let mut edges: Vec<Edge> = Vec::with_capacity(3 * in_mask);

    let push = |edges: &mut Vec<Edge>, a: usize, b: usize| {
        if mask[a] != 0 && mask[b] != 0 {
            edges.push(Edge {
                reliability: reliability[a] + reliability[b],
                a: a as u32,
                b: b as u32,
                wraps: find_wrap(phase[a], phase[b]),
            });
        }
    };

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx - 1 {
                let idx = k * nxy + j * nx + i;
                push(&mut edges, idx, idx + 1);
            }
        }
    }
    for k in 0..nz {
        for j in 0..ny - 1 {
            for i in 0..nx {
                let idx = k * nxy + j * nx + i;
                push(&mut edges, idx, idx + nx);
            }
        }
    }
    for k in 0..nz - 1 {
        for j in 0..ny {
            for i in 0..nx {
                let idx = k * nxy + j * nx + i;
                push(&mut edges, idx, idx + nxy);
            }
        }
    }

    edges
}

/// Sort edges most-reliable (lowest value) first.
///
/// Ties break on construction order, so the result does not depend on the sort's
/// internal choices. Real ties are common in synthetic data and essentially absent
/// in measured phase.
fn sort_edges(edges: &mut [Edge]) {
    let key = |e: &Edge| (e.reliability, e.a, e.b);
    #[cfg(feature = "parallel")]
    edges.par_sort_unstable_by(|x, y| {
        key(x).partial_cmp(&key(y)).unwrap_or(std::cmp::Ordering::Equal)
    });
    #[cfg(not(feature = "parallel"))]
    edges.sort_unstable_by(|x, y| {
        key(x).partial_cmp(&key(y)).unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Subtract the median wrap count over the mask, so the result stays centred near
/// the input range. Mirrors ROMEO's `correct_global_offset`.
fn correct_global_offset(phase: &mut [f64], mask: &[u8]) {
    let mut wraps: Vec<f64> = phase
        .iter()
        .enumerate()
        .filter(|(i, v)| mask[*i] != 0 && v.is_finite())
        .map(|(_, &v)| (v / TWO_PI).round())
        .collect();

    if wraps.is_empty() {
        return;
    }

    wraps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = wraps[wraps.len() / 2];
    if median.abs() > 0.1 {
        let correction = TWO_PI * median;
        for (i, v) in phase.iter_mut().enumerate() {
            if mask[i] != 0 {
                *v -= correction;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(nx: usize, ny: usize, nz: usize) -> Grid {
        Grid::new(nx, ny, nz, 1.0, 1.0, 1.0)
    }

    fn wrap_to_pi(v: f64) -> f64 {
        v - TWO_PI * (v / TWO_PI).round()
    }

    /// Build a wrapped volume from a continuous field, plus the field itself.
    fn wrap_field<F: Fn(f64, f64, f64) -> f64>(
        nx: usize,
        ny: usize,
        nz: usize,
        f: F,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut truth = vec![0.0; nx * ny * nz];
        let mut wrapped = vec![0.0; nx * ny * nz];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = k * nx * ny + j * nx + i;
                    let v = f(i as f64, j as f64, k as f64);
                    truth[idx] = v;
                    wrapped[idx] = wrap_to_pi(v);
                }
            }
        }
        (truth, wrapped)
    }

    /// Assert that `got` equals `truth` up to one global 2π offset, over the mask.
    ///
    /// The offset is read off the *median* residual, not a single voxel, so the
    /// check still works when the input carries noise.
    fn assert_unwrapped_to_constant(got: &[f64], truth: &[f64], mask: &[u8], tol: f64) {
        let mut residuals: Vec<f64> = (0..got.len())
            .filter(|&i| mask[i] != 0)
            .map(|i| got[i] - truth[i])
            .collect();
        assert!(!residuals.is_empty(), "empty mask");
        residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = residuals[residuals.len() / 2];
        let offset = TWO_PI * (median / TWO_PI).round();
        assert!(
            (median - offset).abs() < tol,
            "median residual {median} is not within {tol} of a multiple of 2pi"
        );
        let worst = residuals
            .iter()
            .map(|d| (d - offset).abs())
            .fold(0.0f64, f64::max);
        assert!(worst < tol, "max deviation {worst} exceeds {tol}");
    }

    #[test]
    fn recovers_a_steep_linear_ramp() {
        let (nx, ny, nz) = (16, 16, 8);
        // ~0.9 rad per voxel in x: many wraps, still well under the π/voxel limit.
        let (truth, wrapped) = wrap_field(nx, ny, nz, |i, j, k| 0.9 * i + 0.3 * j + 0.2 * k);
        let mask = vec![1u8; nx * ny * nz];

        let out = unwrap_bestpath(&wrapped, &mask, &BestPathParams::default(), &grid(nx, ny, nz));
        assert_unwrapped_to_constant(&out, &truth, &mask, 1e-9);
    }

    #[test]
    fn recovers_a_quadratic_field() {
        let (nx, ny, nz) = (20, 20, 12);
        let (truth, wrapped) = wrap_field(nx, ny, nz, |i, j, k| {
            let (x, y, z) = (i - 10.0, j - 10.0, k - 6.0);
            0.05 * (x * x + y * y) + 0.1 * z * z
        });
        let mask = vec![1u8; nx * ny * nz];

        let out = unwrap_bestpath(&wrapped, &mask, &BestPathParams::default(), &grid(nx, ny, nz));
        assert_unwrapped_to_constant(&out, &truth, &mask, 1e-9);
    }

    #[test]
    fn unwrapped_result_is_already_smooth() {
        // The real property of interest: no 2pi jumps between neighbours.
        let (nx, ny, nz) = (16, 16, 16);
        let (_, wrapped) = wrap_field(nx, ny, nz, |i, j, k| 0.8 * i - 0.5 * j + 0.4 * k);
        let mask = vec![1u8; nx * ny * nz];

        let out = unwrap_bestpath(&wrapped, &mask, &BestPathParams::default(), &grid(nx, ny, nz));
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx - 1 {
                    let idx = k * nx * ny + j * nx + i;
                    let jump = (out[idx + 1] - out[idx]).abs();
                    assert!(jump < PI, "2pi jump of {jump} at ({i},{j},{k})");
                }
            }
        }
    }

    #[test]
    fn masked_voxels_are_left_untouched() {
        let (nx, ny, nz) = (12, 12, 12);
        // Steep enough that the 6-voxel cube below spans several wraps; a gentler
        // ramp can fit entirely inside one branch and unwrap to a no-op.
        let (truth, wrapped) = wrap_field(nx, ny, nz, |i, j, _| 0.9 * i + 0.9 * j);
        // Central cube only.
        let mut mask = vec![0u8; nx * ny * nz];
        for k in 3..9 {
            for j in 3..9 {
                for i in 3..9 {
                    mask[k * nx * ny + j * nx + i] = 1;
                }
            }
        }

        let out = unwrap_bestpath(&wrapped, &mask, &BestPathParams::default(), &grid(nx, ny, nz));
        for i in 0..nx * ny * nz {
            if mask[i] == 0 {
                assert_eq!(out[i], wrapped[i], "masked voxel {i} was modified");
            }
        }
        // And the in-mask part really was unwrapped, not just copied through.
        assert_unwrapped_to_constant(&out, &truth, &mask, 1e-9);
        let changed = (0..nx * ny * nz)
            .filter(|&i| mask[i] != 0 && (out[i] - wrapped[i]).abs() > 1e-9)
            .count();
        assert!(changed > 0, "nothing inside the mask was unwrapped");
    }

    #[test]
    fn disconnected_components_are_each_internally_consistent() {
        let (nx, ny, nz) = (24, 8, 8);
        let (truth, wrapped) = wrap_field(nx, ny, nz, |i, _, _| 0.9 * i);
        // Two slabs separated by a gap: no edge can link them.
        let mut mask = vec![0u8; nx * ny * nz];
        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 0..nx {
                    if !(8..16).contains(&i) {
                        mask[k * nx * ny + j * nx + i] = 1;
                    }
                }
            }
        }

        let out = unwrap_bestpath(&wrapped, &mask, &BestPathParams::default(), &grid(nx, ny, nz));

        // Each slab matches the truth up to its own constant; the constants need
        // not agree, and there is no information that could make them agree.
        for (lo, hi) in [(0usize, 8usize), (16, nx)] {
            let mut sub_mask = vec![0u8; nx * ny * nz];
            for k in 0..nz {
                for j in 0..ny {
                    for i in lo..hi {
                        let idx = k * nx * ny + j * nx + i;
                        sub_mask[idx] = mask[idx];
                    }
                }
            }
            assert_unwrapped_to_constant(&out, &truth, &sub_mask, 1e-9);
        }
    }

    #[test]
    fn is_deterministic_across_runs() {
        let (nx, ny, nz) = (14, 14, 10);
        let (_, wrapped) = wrap_field(nx, ny, nz, |i, j, k| 0.6 * i + 0.5 * j + 0.7 * k);
        let mut mask = vec![1u8; nx * ny * nz];
        // An irregular hole, to exercise the stand-in reliabilities.
        for k in 4..7 {
            for j in 4..7 {
                mask[k * nx * ny + j * nx + 5] = 0;
            }
        }

        let p = BestPathParams::default();
        let a = unwrap_bestpath(&wrapped, &mask, &p, &grid(nx, ny, nz));
        let b = unwrap_bestpath(&wrapped, &mask, &p, &grid(nx, ny, nz));
        assert_eq!(a, b);
    }

    #[test]
    fn survives_noise_in_the_phase() {
        let (nx, ny, nz) = (20, 20, 12);
        let (truth, mut wrapped) = wrap_field(nx, ny, nz, |i, j, k| 0.5 * i + 0.3 * j + 0.2 * k);
        // Deterministic low-amplitude noise, well below the wrap threshold.
        for (i, v) in wrapped.iter_mut().enumerate() {
            let r = (splitmix64(i as u64) >> 11) as f64 / (1u64 << 53) as f64;
            *v = wrap_to_pi(*v + 0.25 * (r - 0.5));
        }
        let mask = vec![1u8; nx * ny * nz];

        let out = unwrap_bestpath(&wrapped, &mask, &BestPathParams::default(), &grid(nx, ny, nz));
        // Noise survives into the output, so compare against the noisy truth
        // tolerance rather than demanding exactness.
        assert_unwrapped_to_constant(&out, &truth, &mask, 0.2);
    }

    #[test]
    fn correct_global_centres_the_result() {
        let (nx, ny, nz) = (12, 12, 12);
        // A field offset by many wraps: without correction the output sits far
        // from zero, with it the median wrap count is brought back to zero.
        let (_, wrapped) = wrap_field(nx, ny, nz, |i, _, _| 0.8 * i + 100.0 * TWO_PI);
        let mask = vec![1u8; nx * ny * nz];
        let g = grid(nx, ny, nz);

        let corrected = unwrap_bestpath(
            &wrapped,
            &mask,
            &BestPathParams { correct_global: true, ..Default::default() },
            &g,
        );
        let mut wraps: Vec<f64> = corrected.iter().map(|v| (v / TWO_PI).round()).collect();
        wraps.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(wraps[wraps.len() / 2], 0.0);
    }

    #[test]
    fn an_empty_mask_returns_the_input() {
        let (nx, ny, nz) = (6, 6, 6);
        let (_, wrapped) = wrap_field(nx, ny, nz, |i, _, _| 0.9 * i);
        let mask = vec![0u8; nx * ny * nz];
        let out = unwrap_bestpath(&wrapped, &mask, &BestPathParams::default(), &grid(nx, ny, nz));
        assert_eq!(out, wrapped);
    }

    #[test]
    fn thin_volumes_do_not_panic() {
        // Fewer than 3 voxels on an axis means no voxel has a full neighbourhood,
        // so every reliability is a stand-in. Unwrapping must still work.
        for dims in [(8, 8, 1), (8, 1, 8), (2, 2, 2), (1, 1, 1)] {
            let (nx, ny, nz) = dims;
            let (_, wrapped) = wrap_field(nx, ny, nz, |i, j, k| 0.9 * i + 0.9 * j + 0.9 * k);
            let mask = vec![1u8; nx * ny * nz];
            let out =
                unwrap_bestpath(&wrapped, &mask, &BestPathParams::default(), &grid(nx, ny, nz));
            assert_eq!(out.len(), nx * ny * nz);
            assert!(out.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn reliability_is_lower_where_phase_is_smooth() {
        let (nx, ny, nz) = (12, 12, 12);
        let (_, mut wrapped) = wrap_field(nx, ny, nz, |i, _, _| 0.2 * i);
        // Spike one interior voxel: its own reliability, and its neighbours',
        // should worsen relative to the smooth background.
        let spike = 6 * nx * ny + 6 * nx + 6;
        wrapped[spike] = wrap_to_pi(wrapped[spike] + 2.5);

        let rel = compute_reliability(&wrapped, 0, nx, ny, nz);
        let smooth = rel[4 * nx * ny + 4 * nx + 4];
        assert!(
            rel[spike] > smooth,
            "spiked voxel reliability {} should exceed smooth {}",
            rel[spike],
            smooth
        );
    }

    #[test]
    fn border_voxels_get_stand_in_reliabilities() {
        let (nx, ny, nz) = (8, 8, 8);
        let (_, wrapped) = wrap_field(nx, ny, nz, |i, _, _| 0.2 * i);
        let rel = compute_reliability(&wrapped, 7, nx, ny, nz);

        // Real reliabilities are bounded by 13*(2pi)^2; stand-ins are drawn from
        // [0, 2^31) and so are astronomically larger in practice.
        let max_real = 13.0 * TWO_PI * TWO_PI;
        assert!(rel[0] > max_real, "corner voxel should hold a stand-in value");
        assert!(rel[4 * nx * ny + 4 * nx + 4] < max_real, "interior voxel should be real");
    }

    #[test]
    fn find_wrap_matches_its_contract() {
        assert_eq!(find_wrap(0.0, 0.0), 0);
        assert_eq!(find_wrap(3.0, -3.0), -1);
        assert_eq!(find_wrap(-3.0, 3.0), 1);
        assert_eq!(find_wrap(1.0, -1.0), 0);
        // Applying the implied offset must land the difference back in [-pi, pi].
        for &(a, b) in &[(3.0, -3.0), (-3.0, 3.0), (0.5, -0.5), (PI, -PI)] {
            let w = find_wrap(a, b) as f64;
            let residual = a - (b + TWO_PI * -w);
            assert!(residual.abs() <= PI + 1e-12, "residual {residual} out of range");
        }
    }

    #[test]
    fn second_difference_is_invariant_to_direction_sign() {
        // Justifies picking one representative per antipodal pair in DIRECTIONS.
        let (a, c, b) = (0.7, -2.9, 2.6);
        let forward = wrap_pi(a - c) - wrap_pi(c - b);
        let reverse = wrap_pi(b - c) - wrap_pi(c - a);
        assert!((forward - reverse).abs() < 1e-12);
    }
}

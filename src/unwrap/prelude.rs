//! PRELUDE phase unwrapping — best-pair-first region merging.
//!
//! After M. Jenkinson, "A Fast, Automated, N-Dimensional Phase Unwrapping
//! Algorithm", FMRIB Technical Report TR01MJ1 (2001),
//! <https://www.fmrib.ox.ac.uk/datasets/techrep/tr01mj1/tr01mj1.pdf>, condensed as
//! Jenkinson M., *Magn Reson Med* 2003;49(1):193–197,
//! <https://doi.org/10.1002/mrm.10354>.
//!
//! PRELUDE is the reference unwrapper that the QSM and field-mapping literature
//! benchmarks against, which is the reason it is here: it makes this crate's
//! unwrapper comparisons readable against the published ones.
//!
//! The literature generally reports it as slower and less accurate than
//! [`unwrap_romeo`](super::unwrap_romeo) — Dymerska et al., *Magn Reson Med*
//! 2021;85(4):2294–2308, <https://doi.org/10.1002/mrm.28563>. **That comparison is
//! against FSL's implementation, and it does not transfer to this one.** Measured on
//! the `bids/` phantom (164×205×205, 1.31M in-mask voxels, release build), per echo:
//!
//! | | first echo | last echo |
//! |---|---|---|
//! | PRELUDE | 0.19 s | 0.34 s |
//! | ROMEO | 0.34 s | 0.29 s |
//! | best path | 0.32 s | 0.33 s |
//! | Laplacian | 0.46 s | 0.55 s |
//!
//! and the local-field correlations after an identical V-SHARP are 0.615 (PRELUDE),
//! 0.608 (Laplacian), 0.602 (best path), 0.592 (ROMEO) — a spread narrower than any
//! of them is from ground truth, which is the agreement the comparison is for rather
//! than a ranking. So treat this module as a baseline because of what it is, not
//! because it is slow; on this data it is not.
//!
//! # Provenance
//!
//! This is a **clean-room implementation written from the technical report alone**.
//! FSL's `prelude.cc` was not read, fetched, or consulted. That is a licensing
//! requirement, not a stylistic one: FSL is licensed for non-commercial use only and
//! the licence reaches "the Software or any derivative of it", so transliterating it
//! would encumber this MIT crate. The report specifies the whole algorithm, down to
//! the data-structure choices, so nothing had to be reverse-engineered. Equation
//! numbers in the comments below refer to it.
//!
//! # The algorithm
//!
//! The unwrapped phase is the wrapped phase plus a whole number of turns,
//! `θ_j = φ_j + 2π·m_j` (eq 1). Solving for a per-voxel `m_j` is hopeless, so the
//! volume is first cut into regions that are each assumed wrap-free, and every region
//! gets a single offset.
//!
//! 1. **Initial regions** (§2.4). Split `[-π, π)` into `num_phase_partitions` equal
//!    bins. For each bin, mask the in-mask voxels whose wrapped phase falls in it and
//!    label the 6-connected components; every component across every bin becomes a
//!    region with its own id. Two voxels in one region therefore differ in wrapped
//!    phase by less than one bin width, so a region straddling a wrap is unlikely —
//!    and the narrower the bin, the less likely still.
//!
//! 2. **Cost** (eq 2–4). Across the interface between regions A and B the cost is
//!    `Σ (φ_Aj − φ_Bk + 2π·M_AB)²`, summed over interfacing voxel pairs, with
//!    `M_AB = M_A − M_B`. Totalled over all interfaces that is the function to
//!    minimise. Differentiating (eq 5–6) gives the continuous minimum
//!    `M_AB = −P_AB / (2π·N_AB)` for `N_AB` interfacing pairs and `P_AB` their summed
//!    phase difference. But `M_AB` has to be an integer, which makes the exact problem
//!    integer programming with `2^N` neighbouring solutions — hence the greedy scheme
//!    below.
//!
//! 3. **Best-pair-first merging** (§2.3). Treat each interface as the constraint
//!    `M_AB = round(K_AB)`, `K_AB = −P_AB/(2π·N_AB)` (eq 7). Getting one of these
//!    wrong by ±1 costs `ΔC_AB = 8π²·N_AB·(½ − |K_AB − L_AB|)` (eq 8–9). Merge the
//!    pair that maximises it (eq 10) — in the report's words, *"choose the pair of
//!    regions where getting the offset 'wrong' is the most disastrous"*. A wide
//!    interface whose mean phase step sits close to a whole number of turns is both
//!    the most confident call available and the most expensive one to get wrong, so
//!    it is settled first and the ambiguous interfaces inherit what it decided.
//!    Merging sums the statistics over the merged region's remaining neighbours
//!    (eq 11–12). Repeat until no interfaces are left.
//!
//! # Deviations from the report
//!
//! - **The merge statistics are kept in the current frame.** Equations 11–12 read
//!   `N_QW = N_AW + N_BW`, `P_QW = P_AW + P_BW`. The counts add directly; the summed
//!   differences do not, until both sides are expressed in one frame. Merging at
//!   `L_AB` only fixes the two sets *relative* to each other, so one of them has to
//!   move — and its `P` to every other region shifts by `±2π·L_AB·N` before the sum.
//!   Which side moves is not free to choose here: the union-find keeps the larger
//!   set's representative, so the absorbed set is the one that moves, and the sign
//!   follows from that rather than always being B's. The report says to merge "using
//!   the 'optimal' offset `L_AB`" and leaves the bookkeeping implicit; `merge_pair`
//!   writes it out. Getting the direction wrong is silent — it biases every later
//!   merge by one turn, and only shows up on volumes where the two sets differ in
//!   size enough for the union-find to pick the other root.
//! - **A lazily-invalidated binary heap replaces the O(C) scan.** The report searches
//!   its `map` of interfaces linearly for the maximum `ΔC`, at most `R` times, which
//!   it costs as a small penalty next to the cheap insert and delete. That holds at
//!   its worked example's scale (1920 regions); it does not at the hundreds of
//!   thousands a high-resolution 3D volume can produce, where `O(R·C)` dominates
//!   everything else. The merge *order* is identical — the heap returns the same
//!   maximum — only the way it is found differs.
//! - **Regions are not restricted to be planar.** §4.1 notes that FSL does this on
//!   large volumes, to cut the chance of two distant areas at the same phase being
//!   labelled one region. Not implemented: it is a mitigation for a failure mode we
//!   have not observed, and it would change results on every volume.
//! - **Masked voxels keep their original wrapped phase** on output, matching
//!   [`unwrap_bestpath`](super::unwrap_bestpath) and [`unwrap_romeo`](super::unwrap_romeo).
//!
//! Each connected set of regions is unwrapped independently and carries its own
//! arbitrary multiple of 2π — nothing links regions across a gap in the mask. Set
//! [`PreludeParams::correct_global`] for a defined choice of that constant.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap, HashMap};

use crate::grid::Grid;
use crate::utils::connected::label_components;
use crate::utils::union_find::UnionFind;

use super::bestpath::correct_global_offset;

const PI: f64 = std::f64::consts::PI;
const TWO_PI: f64 = 2.0 * PI;

/// Sentinel in the region map for a voxel that belongs to no region — outside the
/// mask, or carrying a non-finite phase.
const NO_REGION: u32 = u32::MAX;

/// Parameters for [`unwrap_prelude`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
pub struct PreludeParams {
    /// Number of equal bins the phase range `[-π, π)` is cut into when forming the
    /// initial regions (default: 6, the report's example and FSL's default).
    ///
    /// More partitions means narrower bins, so a connected component is less likely
    /// to contain a wrap — at the price of more, smaller regions and a longer merge.
    /// Fewer means larger regions that are more likely to straddle a wrap, which the
    /// merge cannot undo: a region is committed to a single offset by construction.
    /// Must be at least 1; at 1 every in-mask component is one region and nothing is
    /// unwrapped at all.
    pub num_phase_partitions: usize,
    /// Subtract the median wrap count from the result so the unwrapped phase stays
    /// centred near the input range (default: false).
    ///
    /// Without it each connected set of regions carries an arbitrary integer multiple
    /// of 2π. Mirrors [`BestPathParams::correct_global`](super::BestPathParams::correct_global)
    /// and [`RomeoParams::correct_global`](super::RomeoParams::correct_global), which
    /// default to false for the same reason: it is a convention, not a correction, and
    /// multi-echo callers usually impose their own.
    pub correct_global: bool,
}

impl Default for PreludeParams {
    fn default() -> Self {
        Self {
            num_phase_partitions: 6,
            correct_global: false,
        }
    }
}

/// Running statistics for one interface between two regions.
///
/// `p` is directed: for the entry stored under `adjacency[a][b]` it is the sum taken
/// as *a minus b*, so `adjacency[b][a]` holds its negation.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Interface {
    /// `N_AB`: the number of interfacing voxel pairs.
    n: u32,
    /// `P_AB = Σ (ψ_Aj − ψ_Bk)` over those pairs, where ψ is the phase in the
    /// *current* working frame — the wrapped phase plus whatever turns previous
    /// merges have already committed to.
    p: f64,
}

impl Interface {
    /// `K_AB = −P_AB / (2π·N_AB)`, the continuous minimum of eq 6.
    #[inline]
    fn k(self) -> f64 {
        -self.p / (TWO_PI * self.n as f64)
    }

    /// `L_AB = round(K_AB)`: the integer offset the merge would commit to (eq 7).
    #[inline]
    fn offset(self) -> f64 {
        self.k().round()
    }

    /// `ΔC_AB = 8π²·N_AB·(½ − |K_AB − L_AB|)`, the smallest cost penalty for getting
    /// this interface's offset wrong by one turn (eq 9).
    ///
    /// Non-negative, since rounding bounds `|K − L|` by ½. It is largest for a wide
    /// interface whose mean phase step lands near a whole number of turns, and falls
    /// to zero when the step is exactly half a turn and the rounding is a coin toss.
    #[inline]
    fn cost(self) -> f64 {
        let k = self.k();
        8.0 * PI * PI * self.n as f64 * (0.5 - (k - k.round()).abs())
    }
}

/// One entry in the merge queue: an interface, with the `ΔC` it had when queued.
///
/// The cost is carried along so a stale entry can be recognised on the way out and
/// requeued at its true value, rather than the queue having to be rebuilt on each
/// merge.
#[derive(Clone, Copy, PartialEq)]
struct Candidate {
    cost: f64,
    a: u32,
    b: u32,
}

impl Eq for Candidate {}

impl Ord for Candidate {
    /// Orders by cost, then by region id so ties break the same way on every run.
    ///
    /// `BinaryHeap` is a max-heap, so the id comparisons are reversed: of two
    /// interfaces with equal `ΔC`, the one with the lower ids is merged first.
    fn cmp(&self, other: &Self) -> Ordering {
        self.cost
            .total_cmp(&other.cost)
            .then_with(|| other.a.cmp(&self.a))
            .then_with(|| other.b.cmp(&self.b))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Unwrap a 3D phase volume with PRELUDE's region-merging algorithm.
///
/// # Arguments
/// * `phase` — wrapped phase in radians, `nx * ny * nz`, expected in `[-π, π]`
/// * `mask` — binary mask, non-zero to process
/// * `grid` — volume dimensions (voxel sizes are unused: regions and interfaces are
///   defined on the voxel lattice, not in physical space)
/// * `params` — see [`PreludeParams`]
///
/// # Returns
/// Unwrapped phase, same length as `phase`. Voxels outside the mask are returned
/// unchanged, as are in-mask voxels whose phase is not finite.
///
/// Each connected set of regions is unwrapped independently and carries its own
/// arbitrary multiple of 2π; there is no information linking them across a gap in
/// the mask.
///
/// # Panics
/// If `phase` or `mask` does not match the grid, if `params.num_phase_partitions` is
/// zero, or if the volume has more than `u32::MAX` voxels.
pub fn unwrap_prelude(
    phase: &[f64],
    mask: &[u8],
    grid: &Grid,
    params: &PreludeParams,
) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let n_total = nx * ny * nz;
    assert_eq!(phase.len(), n_total, "phase length must match grid dimensions");
    assert_eq!(mask.len(), n_total, "mask length must match grid dimensions");
    assert!(
        params.num_phase_partitions > 0,
        "num_phase_partitions must be at least 1"
    );
    assert!(
        n_total < u32::MAX as usize,
        "PRELUDE indexes regions by u32 (volume has {n_total} voxels)"
    );

    let mut unwrapped = phase.to_vec();
    if n_total == 0 {
        return unwrapped;
    }

    let (region_of, n_regions) =
        label_regions(phase, mask, grid.dims, params.num_phase_partitions);
    if n_regions == 0 {
        return unwrapped;
    }

    let mut adjacency = build_interfaces(phase, &region_of, n_regions, grid.dims);
    let mut uf = UnionFind::new(n_regions);
    merge_regions(&mut adjacency, &mut uf);

    for i in 0..n_total {
        if region_of[i] != NO_REGION {
            let (_, turns) = uf.find(region_of[i] as usize);
            unwrapped[i] = phase[i] + TWO_PI * turns as f64;
        }
    }

    if params.correct_global {
        correct_global_offset(&mut unwrapped, mask);
    }

    unwrapped
}

/// Which phase partition a voxel falls in, or `None` if it has no usable phase.
///
/// Clamped rather than asserted: the caller is documented to supply phase in
/// `[-π, π]`, and exactly `+π` would otherwise fall off the end of the last bin.
/// Anything that strayed further out joins the nearest bin, which keeps the voxel in
/// play — any region it lands in still merges on its measured phase differences.
#[inline]
fn phase_partition(v: f64, width: f64, partitions: usize) -> Option<usize> {
    if !v.is_finite() {
        return None;
    }
    let bin = ((v + PI) / width).floor().max(0.0);
    // `as usize` saturates, so a wildly out-of-range phase lands in the last bin.
    Some((bin as usize).min(partitions - 1))
}

/// Partition the phase range, label the connected components of each partition, and
/// hand every component across every partition a unique region id (§2.4).
///
/// Returns the per-voxel region map, [`NO_REGION`] where there is none, and the
/// number of regions created.
fn label_regions(
    phase: &[f64],
    mask: &[u8],
    dims: (usize, usize, usize),
    partitions: usize,
) -> (Vec<u32>, usize) {
    let n_total = phase.len();
    let width = TWO_PI / partitions as f64;

    let mut region_of = vec![NO_REGION; n_total];
    let mut partition_mask = vec![0u8; n_total];
    let mut next_id: u32 = 0;

    for bin in 0..partitions {
        for i in 0..n_total {
            let inside = mask[i] != 0
                && phase_partition(phase[i], width, partitions) == Some(bin);
            partition_mask[i] = u8::from(inside);
        }

        let (labels, sizes) = label_components(&partition_mask, dims);
        for i in 0..n_total {
            if labels[i] != 0 {
                region_of[i] = next_id + labels[i] - 1;
            }
        }
        // `sizes[0]` is the background slot, so the component count is one less.
        next_id += (sizes.len() - 1) as u32;
    }

    (region_of, next_id as usize)
}

/// Accumulate `N_AB` and `P_AB` for every interface, as a symmetric adjacency list.
///
/// Only the three forward neighbours are visited, so each interfacing voxel pair is
/// counted exactly once — which is what `N_AB` in eq 6 is defined to be.
fn build_interfaces(
    phase: &[f64],
    region_of: &[u32],
    n_regions: usize,
    dims: (usize, usize, usize),
) -> Vec<BTreeMap<u32, Interface>> {
    let (nx, ny, nz) = dims;
    let nxy = nx * ny;

    // Keyed on the ordered pair packed into a u64, so the sums below accumulate in
    // raster order and the result does not depend on hash iteration order.
    let mut pairs: HashMap<u64, Interface> = HashMap::new();

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let i = z * nxy + y * nx + x;
                let ra = region_of[i];
                if ra == NO_REGION {
                    continue;
                }
                let accumulate = |j: usize, pairs: &mut HashMap<u64, Interface>| {
                    let rb = region_of[j];
                    if rb == NO_REGION || rb == ra {
                        return;
                    }
                    // Store the sum directed from the lower id to the higher.
                    let (lo, hi, d) = if ra < rb {
                        (ra, rb, phase[i] - phase[j])
                    } else {
                        (rb, ra, phase[j] - phase[i])
                    };
                    let entry = pairs
                        .entry((lo as u64) << 32 | hi as u64)
                        .or_insert(Interface { n: 0, p: 0.0 });
                    entry.n += 1;
                    entry.p += d;
                };
                if x + 1 < nx {
                    accumulate(i + 1, &mut pairs);
                }
                if y + 1 < ny {
                    accumulate(i + nx, &mut pairs);
                }
                if z + 1 < nz {
                    accumulate(i + nxy, &mut pairs);
                }
            }
        }
    }

    let mut adjacency = vec![BTreeMap::new(); n_regions];
    for (key, iface) in pairs {
        let lo = (key >> 32) as u32;
        let hi = key as u32;
        adjacency[lo as usize].insert(hi, iface);
        adjacency[hi as usize].insert(
            lo,
            Interface {
                n: iface.n,
                p: -iface.p,
            },
        );
    }
    adjacency
}

/// Run the best-pair-first merge to exhaustion (§2.3).
///
/// On return the union-find holds, for every region, the number of turns separating
/// it from its set's representative.
fn merge_regions(adjacency: &mut [BTreeMap<u32, Interface>], uf: &mut UnionFind) {
    let mut heap: BinaryHeap<Candidate> = BinaryHeap::new();
    for (a, neighbours) in adjacency.iter().enumerate() {
        for (&b, iface) in neighbours {
            if (a as u32) < b {
                heap.push(Candidate {
                    cost: iface.cost(),
                    a: a as u32,
                    b,
                });
            }
        }
    }

    while let Some(candidate) = heap.pop() {
        let (a, b) = (candidate.a as usize, candidate.b as usize);
        // An endpoint that is no longer its set's representative has been absorbed,
        // and the merge that absorbed it queued a replacement for this interface.
        if uf.find(a).0 != a || uf.find(b).0 != b {
            continue;
        }
        let Some(&iface) = adjacency[a].get(&candidate.b) else {
            continue;
        };
        let cost = iface.cost();
        // An interface carrying no usable cost tells us nothing about the offset, and
        // guessing one would be worse than leaving the two regions apart. `label_regions`
        // keeps non-finite phases out of every region, so this does not arise from
        // ordinary input; it is here so that a future change upstream degrades into
        // "unmerged" rather than into something silently wrong.
        if !cost.is_finite() {
            continue;
        }
        // Both operands are finite here — the guard above saw to `cost`, and a queued
        // NaN would have been dropped by that same guard when its entry was popped — so
        // `==` is exact, and the loop terminates. Removing that guard reintroduces the
        // NaN-never-equals-itself case and with it an infinite requeue, which is what
        // `an_interface_with_no_usable_cost_is_left_unmerged` exists to pin.
        if cost != candidate.cost {
            // The interface changed after this entry was queued. Requeue it at its true
            // cost; the entry that superseded this one is already in the heap. The
            // recomputation is bit-identical to the one that queued it, so this settles
            // in one round rather than oscillating.
            heap.push(Candidate { cost, ..candidate });
            continue;
        }
        merge_pair(adjacency, uf, candidate.a, candidate.b, iface, &mut heap);
    }
}

/// Merge regions `a` and `b` at the offset their interface implies, fold their
/// adjacency together, and queue the interfaces that changed.
fn merge_pair(
    adjacency: &mut [BTreeMap<u32, Interface>],
    uf: &mut UnionFind,
    a: u32,
    b: u32,
    iface: Interface,
    heap: &mut BinaryHeap<Candidate>,
) {
    let turns = iface.offset();
    debug_assert!(
        turns.abs() < i32::MAX as f64,
        "interface offset {turns} is out of range for the union-find's i32 potential"
    );

    // Committing to M_AB = L_AB means theta_A - theta_B = psi_A - psi_B + 2*pi*L_AB:
    // the two sets end up L_AB turns apart, with B the lower.
    let joined = uf.union_with_delta(a as usize, b as usize, -(turns as i32));
    debug_assert!(joined, "merge_pair called on two regions already in one set");
    let root = uf.find(a as usize).0 as u32;
    let absorbed = if root == a { b } else { a };

    // The interface just consumed is gone.
    adjacency[a as usize].remove(&b);
    adjacency[b as usize].remove(&a);

    // That constraint fixes only the *difference* between the two sets, so one of them
    // has to be re-expressed in the other's frame, and the union-find has already
    // chosen which: it keeps the larger set's representative, and `find` then reports
    // the absorbed set's potentials against it. So the absorbed set is exactly the one
    // whose phases moved — by L_AB turns down if that was B, up if it was A — and its
    // interfaces to everything else move with it. This is what equation 12 leaves
    // implicit when it says to merge "using the 'optimal' offset"; getting the
    // direction wrong here silently biases every later merge by 2*pi*L_AB.
    let shift = if absorbed == b { -TWO_PI * turns } else { TWO_PI * turns };

    // Fold the absorbed region's interfaces into the surviving one (eq 11-12).
    let moved = std::mem::take(&mut adjacency[absorbed as usize]);
    let mut touched: Vec<u32> = Vec::with_capacity(moved.len());
    for (w, from_absorbed) in moved {
        // Every merge drops the absorbed region's mirror entries, so adjacency only ever
        // holds live representatives. That is what lets `merge_regions` treat a missing
        // entry as "already merged" rather than as a lost interface, so it is worth
        // pinning: skipping the removal below leaks dead ids back in, and nothing else
        // would notice.
        debug_assert_eq!(
            uf.find(w as usize).0,
            w as usize,
            "adjacency holds region {w}, which is not its set's representative"
        );
        adjacency[w as usize].remove(&absorbed);
        let merged = {
            let entry = adjacency[root as usize]
                .entry(w)
                .or_insert(Interface { n: 0, p: 0.0 });
            entry.n += from_absorbed.n;
            entry.p += from_absorbed.p + shift * from_absorbed.n as f64;
            *entry
        };
        adjacency[w as usize].insert(
            root,
            Interface {
                n: merged.n,
                p: -merged.p,
            },
        );
        touched.push(w);
    }

    // Only the folded interfaces changed; the survivor's own are already in its frame,
    // and their queued candidates stay valid.
    for w in touched {
        if let Some(entry) = adjacency[root as usize].get(&w) {
            heap.push(Candidate {
                cost: entry.cost(),
                a: root,
                b: w,
            });
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
    /// The offset is read off the *median* residual, not a single voxel, so the check
    /// still works when the input carries noise.
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

    /// A deterministic uniform draw in `[0, 1)`, so the noisy tests do not pull in a
    /// dependency and do not change answer between runs.
    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64) / ((1u64 << 53) as f64)
    }

    // ---------------------------------------------------------------- recovery

    #[test]
    fn recovers_a_steep_linear_ramp() {
        let (nx, ny, nz) = (16, 16, 8);
        // ~0.9 rad per voxel in x: many wraps, still well under the pi/voxel limit.
        let (truth, wrapped) = wrap_field(nx, ny, nz, |i, j, k| 0.9 * i + 0.3 * j + 0.2 * k);
        let mask = vec![1u8; nx * ny * nz];

        let out = unwrap_prelude(&wrapped, &mask, &grid(nx, ny, nz), &PreludeParams::default());
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

        let out = unwrap_prelude(&wrapped, &mask, &grid(nx, ny, nz), &PreludeParams::default());
        assert_unwrapped_to_constant(&out, &truth, &mask, 1e-9);
    }

    #[test]
    fn a_single_wrap_between_two_regions_is_closed() {
        // Two voxels 6 rad apart in wrapped phase: one turn, and the whole algorithm
        // reduces to a single interface with N = 1 and L = 1.
        let phase = vec![3.0, -3.0];
        let mask = vec![1u8, 1];
        let out = unwrap_prelude(&phase, &mask, &grid(2, 1, 1), &PreludeParams::default());

        // Which of the two keeps its phase is the arbitrary constant; what the merge
        // decides is the step between them.
        let step = out[1] - out[0];
        assert!(
            (step - (TWO_PI - 6.0)).abs() < 1e-12,
            "expected the 6 rad gap to close to {}, got {step}",
            TWO_PI - 6.0
        );
        for i in 0..2 {
            let turns = (out[i] - phase[i]) / TWO_PI;
            assert!(
                (turns - turns.round()).abs() < 1e-12,
                "voxel {i} moved by {turns} turns, which is not a whole number"
            );
        }
    }

    #[test]
    fn unwrapped_result_is_already_smooth() {
        // The real property of interest: no 2pi jumps between neighbours.
        let (nx, ny, nz) = (16, 16, 16);
        let (_, wrapped) = wrap_field(nx, ny, nz, |i, j, k| 0.8 * i - 0.5 * j + 0.4 * k);
        let mask = vec![1u8; nx * ny * nz];

        let out = unwrap_prelude(&wrapped, &mask, &grid(nx, ny, nz), &PreludeParams::default());

        let nxy = nx * ny;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = k * nxy + j * nx + i;
                    for (step, limit, axis) in [(1, i + 1 < nx, 'x'), (nx, j + 1 < ny, 'y'), (nxy, k + 1 < nz, 'z')] {
                        if limit {
                            let d = (out[idx + step] - out[idx]).abs();
                            assert!(d < PI, "{axis} step of {d} at ({i},{j},{k}) exceeds pi");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn survives_noise_in_the_phase() {
        let (nx, ny, nz) = (24, 24, 12);
        let (truth, mut wrapped) = wrap_field(nx, ny, nz, |i, j, k| 0.5 * i + 0.2 * j + 0.15 * k);
        let mut state = 0xC0FFEEu64;
        let mut noisy_truth = truth.clone();
        for idx in 0..wrapped.len() {
            let noise = 0.3 * (lcg(&mut state) - 0.5);
            noisy_truth[idx] += noise;
            wrapped[idx] = wrap_to_pi(noisy_truth[idx]);
        }
        let mask = vec![1u8; nx * ny * nz];

        let out = unwrap_prelude(&wrapped, &mask, &grid(nx, ny, nz), &PreludeParams::default());
        // Every voxel must land on the right turn; the noise itself survives untouched.
        assert_unwrapped_to_constant(&out, &noisy_truth, &mask, 1e-9);
    }

    #[test]
    fn agrees_with_best_path_on_a_smooth_field() {
        let (nx, ny, nz) = (20, 20, 10);
        let (_, wrapped) = wrap_field(nx, ny, nz, |i, j, k| {
            0.6 * i - 0.35 * j + 0.5 * k + 0.02 * i * j
        });
        let mask = vec![1u8; nx * ny * nz];
        let g = grid(nx, ny, nz);

        let a = unwrap_prelude(&wrapped, &mask, &g, &PreludeParams::default());
        let b = super::super::unwrap_bestpath(&wrapped, &mask, &Default::default(), &g);

        // Both fix the phase only up to one global turn over the (single) connected
        // component, so compare after removing that constant.
        let offset = TWO_PI * ((a[0] - b[0]) / TWO_PI).round();
        let worst = (0..a.len())
            .map(|i| (a[i] - b[i] - offset).abs())
            .fold(0.0f64, f64::max);
        assert!(worst < 1e-9, "PRELUDE and best path differ by up to {worst} rad");
    }

    // ------------------------------------------------------------ mask handling

    #[test]
    fn masked_voxels_are_left_untouched() {
        let (nx, ny, nz) = (12, 12, 6);
        let (truth, wrapped) = wrap_field(nx, ny, nz, |i, j, k| 0.7 * i + 0.4 * j + 0.3 * k);
        let nxy = nx * ny;
        let mut mask = vec![0u8; nx * ny * nz];
        for k in 1..nz - 1 {
            for j in 2..ny - 2 {
                for i in 2..nx - 2 {
                    mask[k * nxy + j * nx + i] = 1;
                }
            }
        }

        let out = unwrap_prelude(&wrapped, &mask, &grid(nx, ny, nz), &PreludeParams::default());
        for i in 0..out.len() {
            if mask[i] == 0 {
                assert_eq!(out[i], wrapped[i], "voxel {i} is outside the mask");
            }
        }
        assert_unwrapped_to_constant(&out, &truth, &mask, 1e-9);
    }

    #[test]
    fn disconnected_components_are_each_internally_consistent() {
        // Two slabs with a gap: nothing links them, so each picks its own turn, but
        // each must be smooth on its own.
        let (nx, ny, nz) = (10, 10, 10);
        let (truth, wrapped) = wrap_field(nx, ny, nz, |i, j, k| 0.9 * i + 0.5 * j + 0.6 * k);
        let nxy = nx * ny;
        let mut mask = vec![0u8; nx * ny * nz];
        for k in 0..nz {
            if k == 4 || k == 5 {
                continue; // the gap
            }
            for idx in k * nxy..(k + 1) * nxy {
                mask[idx] = 1;
            }
        }

        let out = unwrap_prelude(&wrapped, &mask, &grid(nx, ny, nz), &PreludeParams::default());

        for (lo, hi) in [(0, 4), (6, 10)] {
            let mut part = vec![0u8; nx * ny * nz];
            part[lo * nxy..hi * nxy].fill(1);
            assert_unwrapped_to_constant(&out, &truth, &part, 1e-9);
        }
    }

    #[test]
    fn an_empty_mask_returns_the_input() {
        let (nx, ny, nz) = (6, 6, 6);
        let (_, wrapped) = wrap_field(nx, ny, nz, |i, j, k| 0.8 * i + j + k);
        let mask = vec![0u8; nx * ny * nz];
        let out = unwrap_prelude(&wrapped, &mask, &grid(nx, ny, nz), &PreludeParams::default());
        assert_eq!(out, wrapped);
    }

    #[test]
    fn non_finite_phase_voxels_are_left_alone() {
        let (nx, ny, nz) = (8, 8, 4);
        let (truth, mut wrapped) = wrap_field(nx, ny, nz, |i, j, k| 0.9 * i + 0.4 * j + 0.3 * k);
        let mut mask = vec![1u8; nx * ny * nz];
        for &idx in &[5usize, 40, 100] {
            wrapped[idx] = f64::NAN;
        }

        let out = unwrap_prelude(&wrapped, &mask, &grid(nx, ny, nz), &PreludeParams::default());
        for &idx in &[5usize, 40, 100] {
            assert!(out[idx].is_nan(), "voxel {idx} should have been left as NaN");
            mask[idx] = 0;
        }
        // Everything else still unwraps, around the holes the NaNs punch in the mask.
        assert_unwrapped_to_constant(&out, &truth, &mask, 1e-9);
    }

    #[test]
    fn thin_volumes_do_not_panic() {
        for dims in [(1, 1, 1), (8, 1, 1), (1, 8, 1), (1, 1, 8), (4, 4, 1)] {
            let (nx, ny, nz) = dims;
            let (_, wrapped) = wrap_field(nx, ny, nz, |i, j, k| 1.4 * i + 1.1 * j + 0.9 * k);
            let mask = vec![1u8; nx * ny * nz];
            let out = unwrap_prelude(&wrapped, &mask, &grid(nx, ny, nz), &PreludeParams::default());
            assert_eq!(out.len(), nx * ny * nz, "dims {dims:?}");
            assert!(out.iter().all(|v| v.is_finite()), "dims {dims:?}");
        }
    }

    // ---------------------------------------------------------------- behaviour

    #[test]
    fn is_deterministic_across_runs() {
        let (nx, ny, nz) = (18, 18, 9);
        let (_, mut wrapped) = wrap_field(nx, ny, nz, |i, j, k| 0.7 * i + 0.45 * j + 0.3 * k);
        let mut state = 0x5EEDu64;
        for v in wrapped.iter_mut() {
            *v = wrap_to_pi(*v + 0.6 * (lcg(&mut state) - 0.5));
        }
        let mask = vec![1u8; nx * ny * nz];
        let g = grid(nx, ny, nz);

        let first = unwrap_prelude(&wrapped, &mask, &g, &PreludeParams::default());
        for _ in 0..4 {
            assert_eq!(
                unwrap_prelude(&wrapped, &mask, &g, &PreludeParams::default()),
                first,
                "repeat runs disagree"
            );
        }
    }

    #[test]
    fn correct_global_centres_the_result() {
        // A ramp spanning tens of radians, so the unwrapped field runs several turns
        // clear of zero whichever region ends up carrying the arbitrary constant.
        let (nx, ny, nz) = (40, 8, 4);
        let (_, wrapped) = wrap_field(nx, ny, nz, |i, j, k| 1.2 * i + 0.4 * j + 0.3 * k);
        let mask = vec![1u8; nx * ny * nz];
        let g = grid(nx, ny, nz);

        let median_turns = |v: &[f64]| {
            let mut turns: Vec<f64> = v.iter().map(|x| (x / TWO_PI).round()).collect();
            turns.sort_by(|a, b| a.partial_cmp(b).unwrap());
            turns[turns.len() / 2]
        };

        let plain = unwrap_prelude(&wrapped, &mask, &g, &PreludeParams::default());
        assert_ne!(
            median_turns(&plain), 0.0,
            "the uncorrected result was already centred, so the test proves nothing"
        );

        let centred = unwrap_prelude(
            &wrapped,
            &mask,
            &g,
            &PreludeParams { correct_global: true, ..PreludeParams::default() },
        );
        assert_eq!(median_turns(&centred), 0.0, "corrected result is not centred");

        // The two differ by that constant and nothing else.
        let shift = plain[0] - centred[0];
        assert!(shift.abs() > 1e-9);
        for i in 0..plain.len() {
            assert!((plain[i] - centred[i] - shift).abs() < 1e-9);
        }
    }

    #[test]
    fn one_partition_leaves_connected_phase_alone() {
        // The documented degenerate case: a single bin makes every connected piece of
        // the mask one region, there are no interfaces, and nothing is unwrapped.
        let (nx, ny, nz) = (10, 10, 4);
        let (_, wrapped) = wrap_field(nx, ny, nz, |i, j, k| 0.9 * i + 0.5 * j + 0.4 * k);
        let mask = vec![1u8; nx * ny * nz];
        let out = unwrap_prelude(
            &wrapped,
            &mask,
            &grid(nx, ny, nz),
            &PreludeParams { num_phase_partitions: 1, ..PreludeParams::default() },
        );
        assert_eq!(out, wrapped);
    }

    #[test]
    #[should_panic(expected = "num_phase_partitions must be at least 1")]
    fn zero_partitions_is_rejected() {
        unwrap_prelude(
            &[0.0; 8],
            &[1u8; 8],
            &grid(2, 2, 2),
            &PreludeParams { num_phase_partitions: 0, ..PreludeParams::default() },
        );
    }

    #[test]
    #[should_panic(expected = "phase length must match grid dimensions")]
    fn mismatched_phase_length_is_rejected() {
        unwrap_prelude(&[0.0; 7], &[1u8; 8], &grid(2, 2, 2), &PreludeParams::default());
    }

    // --------------------------------------------------------- initial regions

    #[test]
    fn every_region_stays_inside_one_phase_partition() {
        let (nx, ny, nz) = (16, 16, 8);
        let (_, wrapped) = wrap_field(nx, ny, nz, |i, j, k| 0.7 * i + 0.3 * j + 0.9 * k);
        let mask = vec![1u8; nx * ny * nz];

        for partitions in [2usize, 3, 6, 12] {
            let (region_of, n_regions) =
                label_regions(&wrapped, &mask, (nx, ny, nz), partitions);
            assert!(n_regions > 0);
            let width = TWO_PI / partitions as f64;
            let mut lo = vec![f64::INFINITY; n_regions];
            let mut hi = vec![f64::NEG_INFINITY; n_regions];
            for i in 0..wrapped.len() {
                assert_ne!(region_of[i], NO_REGION, "voxel {i} was left unlabelled");
                let r = region_of[i] as usize;
                lo[r] = lo[r].min(wrapped[i]);
                hi[r] = hi[r].max(wrapped[i]);
            }
            for r in 0..n_regions {
                assert!(
                    hi[r] - lo[r] < width,
                    "region {r} spans {} rad, wider than the {width} rad partition",
                    hi[r] - lo[r]
                );
            }
        }
    }

    #[test]
    fn narrower_partitions_make_more_regions() {
        let (nx, ny, nz) = (20, 20, 10);
        let (_, wrapped) = wrap_field(nx, ny, nz, |i, j, k| 0.55 * i + 0.3 * j + 0.2 * k);
        let mask = vec![1u8; nx * ny * nz];

        let counts: Vec<usize> = [2usize, 4, 6, 12, 24]
            .iter()
            .map(|&p| label_regions(&wrapped, &mask, (nx, ny, nz), p).1)
            .collect();
        for w in counts.windows(2) {
            assert!(w[1] > w[0], "region counts are not increasing: {counts:?}");
        }
    }

    #[test]
    fn the_phase_endpoints_land_in_the_end_partitions() {
        let width = TWO_PI / 6.0;
        assert_eq!(phase_partition(-PI, width, 6), Some(0));
        assert_eq!(phase_partition(-PI + 1e-12, width, 6), Some(0));
        assert_eq!(phase_partition(0.0, width, 6), Some(3));
        assert_eq!(phase_partition(PI - 1e-12, width, 6), Some(5));
        assert_eq!(phase_partition(PI, width, 6), Some(5), "+pi must not fall off the end");
        // Out of range is clamped rather than dropped, and NaN is dropped.
        assert_eq!(phase_partition(-10.0, width, 6), Some(0));
        assert_eq!(phase_partition(10.0, width, 6), Some(5));
        assert_eq!(phase_partition(f64::NAN, width, 6), None);
        assert_eq!(phase_partition(f64::INFINITY, width, 6), None);
    }

    /// The report's §2.4 at the scale of its own worked example (§4.1): a
    /// 320x320 = 102400-voxel image with 31566 voxels in the mask gave 1920 regions
    /// from 6 partitions, 1102 of them single voxels and only 163 larger than 27
    /// voxels. Our data is synthetic, so the counts cannot match exactly; what the
    /// bounds here catch is a partitioning that is wrong by an order of magnitude.
    #[test]
    fn region_counts_are_the_order_the_report_reports() {
        let (nx, ny, nz) = (320, 320, 1);
        let (cx, cy) = (159.5f64, 159.5f64);
        // Radius chosen so the disc holds ~31566 voxels, as the report's mask does.
        let radius = (31566.0 / PI).sqrt();

        let mut state = 0xBEEFu64;
        let mut wrapped = vec![0.0; nx * ny];
        let mut mask = vec![0u8; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let idx = j * nx + i;
                let (x, y) = (i as f64 - cx, j as f64 - cy);
                if x * x + y * y > radius * radius {
                    continue;
                }
                mask[idx] = 1;
                // A field with several wraps across the mask, plus noise at the level
                // that produces the report's crop of single-voxel regions.
                let field = 0.09 * x + 0.004 * (x * x + y * y) / 10.0;
                wrapped[idx] = wrap_to_pi(field + 1.2 * (lcg(&mut state) - 0.5));
            }
        }
        let in_mask = mask.iter().filter(|&&m| m != 0).count();
        assert!(
            (30000..33000).contains(&in_mask),
            "test mask has {in_mask} voxels, not the report's ~31566"
        );

        let (region_of, n_regions) = label_regions(&wrapped, &mask, (nx, ny, nz), 6);
        let mut sizes = vec![0usize; n_regions];
        for &r in &region_of {
            if r != NO_REGION {
                sizes[r as usize] += 1;
            }
        }
        let singletons = sizes.iter().filter(|&&s| s == 1).count();
        let large = sizes.iter().filter(|&&s| s > 27).count();
        println!(
            "regions={n_regions} singletons={singletons} larger_than_27={large} \
             in_mask={in_mask}"
        );

        assert!(
            (500..6000).contains(&n_regions),
            "{n_regions} regions is not the order of the report's 1920"
        );
        assert!(
            singletons * 4 > n_regions && singletons < n_regions,
            "{singletons} single-voxel regions out of {n_regions} does not look like the \
             report's 1102/1920"
        );
        assert!(
            large < n_regions / 4,
            "{large} regions larger than 27 voxels, out of {n_regions}; the report had 163/1920"
        );

        // And the whole thing still unwraps at that scale.
        let out = unwrap_prelude(&wrapped, &mask, &grid(nx, ny, nz), &PreludeParams::default());
        for j in 0..ny {
            for i in 0..nx - 1 {
                let idx = j * nx + i;
                if mask[idx] != 0 && mask[idx + 1] != 0 {
                    let step = (out[idx + 1] - out[idx]).abs();
                    assert!(step < TWO_PI, "x step of {step} rad at ({i},{j})");
                }
            }
        }
    }

    #[test]
    fn an_interface_with_no_usable_cost_is_left_unmerged() {
        // `unwrap_prelude` cannot produce this — `label_regions` drops non-finite phases
        // before a region is formed — so drive `merge_regions` directly. What is being
        // pinned is that the loop *terminates*: with `==` in place of `total_cmp`, a NaN
        // cost compares unequal to itself and the candidate is requeued forever. A
        // regression here hangs rather than fails, so a timeout on this test is the
        // symptom to look for.
        let mut adjacency = vec![BTreeMap::new(); 3];
        for (a, b, iface) in [
            (0u32, 1u32, Interface { n: 4, p: f64::NAN }),
            (1, 2, Interface { n: 6, p: 0.0 }),
        ] {
            adjacency[a as usize].insert(b, iface);
            adjacency[b as usize].insert(a, Interface { n: iface.n, p: -iface.p });
        }

        let mut uf = UnionFind::new(3);
        merge_regions(&mut adjacency, &mut uf);

        // The usable interface merged; the one carrying NaN did not.
        assert_eq!(uf.find(1).0, uf.find(2).0, "regions 1 and 2 should have merged");
        assert_ne!(uf.find(0).0, uf.find(1).0, "region 0 had no usable interface");
        assert_eq!(uf.find(0).1, 0, "an unmerged region keeps its own frame");
    }

    // ------------------------------------------------- the heap against a linear scan

    /// Selection exactly as the report describes it: scan every interface for the
    /// largest `ΔC` (eq 10), merge that pair, repeat. `O(C)` per iteration.
    ///
    /// Shares [`merge_pair`] with the real implementation, so the only thing that
    /// differs is how the maximum is found — which is the one deviation the module
    /// docs claim is free.
    fn merge_regions_by_linear_scan(
        adjacency: &mut [BTreeMap<u32, Interface>],
        uf: &mut UnionFind,
    ) {
        let mut sink: BinaryHeap<Candidate> = BinaryHeap::new();
        loop {
            let mut best: Option<(u32, u32, Interface)> = None;
            for a in 0..adjacency.len() {
                if uf.find(a).0 != a {
                    continue;
                }
                for (&b, &iface) in &adjacency[a] {
                    if (a as u32) < b
                        && best.is_none_or(|(_, _, chosen)| iface.cost() > chosen.cost())
                    {
                        best = Some((a as u32, b, iface));
                    }
                }
            }
            let Some((a, b, iface)) = best else { break };
            merge_pair(adjacency, uf, a, b, iface, &mut sink);
            sink.clear();
        }
    }

    /// Random region graphs with deliberately inconsistent interfaces.
    ///
    /// `p` ranges far wider than any real volume would produce, which is the point: it
    /// makes merges that *lower* a neighbouring interface's `ΔC`, so a candidate queued
    /// earlier at a higher cost surfaces before the one that replaced it. That is the
    /// only situation the heap's staleness check exists for, and on real data it does
    /// not arise — 2582 merges over a noisy 26×26×14 volume triggered it zero times, so
    /// a test built from volumes cannot reach it.
    fn random_region_graph(
        state: &mut u64,
        n_regions: usize,
        edge_chance: f64,
    ) -> Vec<BTreeMap<u32, Interface>> {
        let mut adjacency = vec![BTreeMap::new(); n_regions];
        for a in 0..n_regions {
            for b in a + 1..n_regions {
                if lcg(state) > edge_chance {
                    continue;
                }
                let iface = Interface {
                    n: 1 + (lcg(state) * 60.0) as u32,
                    p: (lcg(state) - 0.5) * 400.0,
                };
                adjacency[a].insert(b as u32, iface);
                adjacency[b].insert(
                    a as u32,
                    Interface { n: iface.n, p: -iface.p },
                );
            }
        }
        adjacency
    }

    #[test]
    fn the_heap_picks_the_same_pair_as_a_linear_scan() {
        let mut state = 0x9A5Du64;
        let mut compared = 0;
        for n_regions in [2usize, 3, 5, 8, 13, 21, 34] {
            for &edge_chance in &[0.15, 0.4, 0.9] {
                for _ in 0..12 {
                    let graph = random_region_graph(&mut state, n_regions, edge_chance);

                    let mut by_heap = graph.clone();
                    let mut uf_heap = UnionFind::new(n_regions);
                    merge_regions(&mut by_heap, &mut uf_heap);

                    let mut by_scan = graph.clone();
                    let mut uf_scan = UnionFind::new(n_regions);
                    merge_regions_by_linear_scan(&mut by_scan, &mut uf_scan);

                    // Both must have merged the same regions together. Which member
                    // ends up as the set's representative is arbitrary — union-by-size
                    // decides it, and a different merge order gives a different tree —
                    // so compare the partition itself, not the roots.
                    for r in 0..n_regions {
                        for q in 0..n_regions {
                            assert_eq!(
                                uf_heap.find(r).0 == uf_heap.find(q).0,
                                uf_scan.find(r).0 == uf_scan.find(q).0,
                                "regions {r} and {q} of {n_regions} were merged by one \
                                 method and not the other"
                            );
                        }
                    }
                    // And committed the same turns. Each set carries an arbitrary
                    // constant, so measure every region against the lowest-numbered
                    // member of its own set.
                    let anchor = |uf: &mut UnionFind, r: usize| {
                        let root = uf.find(r).0;
                        (0..n_regions).find(|&q| uf.find(q).0 == root).expect("r is its own")
                    };
                    for r in 0..n_regions {
                        let (a_h, a_s) = (anchor(&mut uf_heap, r), anchor(&mut uf_scan, r));
                        let rel_h = uf_heap.find(r).1 - uf_heap.find(a_h).1;
                        let rel_s = uf_scan.find(r).1 - uf_scan.find(a_s).1;
                        assert_eq!(
                            rel_h, rel_s,
                            "region {r} of {n_regions} sits {rel_h} turns above its set's \
                             first member under the heap, but {rel_s} under the scan"
                        );
                    }
                    compared += 1;
                }
            }
        }
        assert_eq!(compared, 7 * 3 * 12);
    }

    // ------------------------------------------------ against a literal transcription

    /// The report's §2.3 written out with no incremental bookkeeping at all: every
    /// interface statistic is recomputed from the voxels after each merge, in the
    /// phases as they currently stand.
    ///
    /// Slow — `O(R)` passes over the volume — but it is the equations and nothing
    /// else, which makes it the yardstick for [`merge_regions`]'s incremental
    /// updates. Returns the unwrapped phase.
    fn merge_from_scratch_each_step(
        phase: &[f64],
        mask: &[u8],
        dims: (usize, usize, usize),
        partitions: usize,
    ) -> Vec<f64> {
        let (nx, ny, nz) = dims;
        let nxy = nx * ny;
        let (region_of, n_regions) = label_regions(phase, mask, dims, partitions);

        // Turns committed so far per region, and which merged set it belongs to.
        let mut turns = vec![0i64; n_regions];
        let mut set_of: Vec<usize> = (0..n_regions).collect();
        let psi = |turns: &[i64], i: usize| {
            phase[i] + TWO_PI * turns[region_of[i] as usize] as f64
        };

        loop {
            let mut stats: BTreeMap<(usize, usize), Interface> = BTreeMap::new();
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let i = z * nxy + y * nx + x;
                        if region_of[i] == NO_REGION {
                            continue;
                        }
                        let sa = set_of[region_of[i] as usize];
                        for (j, inside) in
                            [(i + 1, x + 1 < nx), (i + nx, y + 1 < ny), (i + nxy, z + 1 < nz)]
                        {
                            if !inside || region_of[j] == NO_REGION {
                                continue;
                            }
                            let sb = set_of[region_of[j] as usize];
                            if sa == sb {
                                continue;
                            }
                            let (lo, hi, d) = if sa < sb {
                                (sa, sb, psi(&turns, i) - psi(&turns, j))
                            } else {
                                (sb, sa, psi(&turns, j) - psi(&turns, i))
                            };
                            let e = stats.entry((lo, hi)).or_insert(Interface { n: 0, p: 0.0 });
                            e.n += 1;
                            e.p += d;
                        }
                    }
                }
            }

            // eq 10: the pair where getting the offset wrong is most disastrous.
            let Some((&(a, b), &iface)) = stats
                .iter()
                .max_by(|x, y| x.1.cost().total_cmp(&y.1.cost()))
            else {
                break;
            };
            let offset = iface.offset() as i64;
            for r in 0..n_regions {
                if set_of[r] == b {
                    turns[r] -= offset;
                    set_of[r] = a;
                }
            }
        }

        (0..phase.len())
            .map(|i| {
                if region_of[i] == NO_REGION {
                    phase[i]
                } else {
                    psi(&turns, i)
                }
            })
            .collect()
    }

    /// Assert two unwrapped fields are the same solution — equal up to the arbitrary
    /// constant each connected piece of the mask carries.
    ///
    /// Compares neighbour differences, which pin a component's field down completely
    /// and are the one thing the constant cannot hide.
    fn assert_same_solution(a: &[f64], b: &[f64], mask: &[u8], dims: (usize, usize, usize)) {
        let (nx, ny, nz) = dims;
        let nxy = nx * ny;
        let mut compared = 0usize;
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let i = z * nxy + y * nx + x;
                    if mask[i] == 0 {
                        continue;
                    }
                    for (j, inside) in
                        [(i + 1, x + 1 < nx), (i + nx, y + 1 < ny), (i + nxy, z + 1 < nz)]
                    {
                        if !inside || mask[j] == 0 {
                            continue;
                        }
                        compared += 1;
                        let d = (a[j] - a[i]) - (b[j] - b[i]);
                        assert!(
                            d.abs() < 1e-9,
                            "the two disagree by {d} rad across ({x},{y},{z})"
                        );
                    }
                }
            }
        }
        assert!(compared > 0, "nothing was compared");
    }

    #[test]
    fn the_incremental_merge_matches_a_literal_transcription() {
        let (nx, ny, nz) = (12, 12, 10);
        let nxy = nx * ny;
        let dims = (nx, ny, nz);
        let g = grid(nx, ny, nz);

        let (_, smooth) = wrap_field(nx, ny, nz, |i, j, k| 0.9 * i + 0.5 * j + 0.6 * k);
        let mut noisy = smooth.clone();
        let mut state = 0x15EEDu64;
        for v in noisy.iter_mut() {
            *v = wrap_to_pi(*v + 0.8 * (lcg(&mut state) - 0.5));
        }

        let full = vec![1u8; nx * ny * nz];
        // A gap across the volume, so there are two independent sets of regions. It
        // was this case that first exposed the frame-shift direction in `merge_pair`.
        let mut split = vec![1u8; nx * ny * nz];
        split[4 * nxy..6 * nxy].fill(0);
        // An irregular blob, so regions have ragged interfaces of very uneven size.
        let mut blob = vec![0u8; nx * ny * nz];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let (x, y, z) = (i as f64 - 5.5, j as f64 - 5.5, k as f64 - 4.5);
                    if x * x + y * y + 1.8 * z * z < 26.0 {
                        blob[k * nxy + j * nx + i] = 1;
                    }
                }
            }
        }

        for (field_name, phase) in [("smooth", &smooth), ("noisy", &noisy)] {
            for (mask_name, mask) in [("full", &full), ("split", &split), ("blob", &blob)] {
                for partitions in [2usize, 4, 6, 9] {
                    let params = PreludeParams {
                        num_phase_partitions: partitions,
                        correct_global: false,
                    };
                    let fast = unwrap_prelude(phase, mask, &g, &params);
                    let slow = merge_from_scratch_each_step(phase, mask, dims, partitions);
                    println!("{field_name}/{mask_name}/{partitions} partitions");
                    assert_same_solution(&fast, &slow, mask, dims);
                }
            }
        }
    }

    // ------------------------------------------------------- interfaces and cost

    #[test]
    fn interface_sums_match_their_definition() {
        // 3x1x1, regions [0, 1, 0]: region 0 interfaces region 1 on two voxel pairs.
        let phase = vec![1.0, -0.5, 2.0];
        let region_of = vec![0u32, 1, 0];
        let adjacency = build_interfaces(&phase, &region_of, 2, (3, 1, 1));

        // P_01 = (phase[0] - phase[1]) + (phase[2] - phase[1]) = 1.5 + 2.5 = 4.0
        assert_eq!(adjacency[0][&1], Interface { n: 2, p: 4.0 });
        assert_eq!(adjacency[1][&0], Interface { n: 2, p: -4.0 });
    }

    #[test]
    fn voxels_in_the_same_region_do_not_interface() {
        let phase = vec![0.0, 0.1, 0.2, 0.3];
        let region_of = vec![0u32, 0, 0, 0];
        let adjacency = build_interfaces(&phase, &region_of, 1, (4, 1, 1));
        assert!(adjacency[0].is_empty());
    }

    #[test]
    fn unlabelled_voxels_do_not_interface() {
        let phase = vec![0.0, 1.0, 2.0];
        let region_of = vec![0u32, NO_REGION, 1];
        let adjacency = build_interfaces(&phase, &region_of, 2, (3, 1, 1));
        assert!(adjacency[0].is_empty(), "a gap must not join the regions either side");
        assert!(adjacency[1].is_empty());
    }

    #[test]
    fn each_interfacing_pair_is_counted_once() {
        // A 2x2x2 checkerboard of two regions: every one of the 12 lattice edges
        // joins different regions, and each must be counted exactly once.
        let region_of: Vec<u32> = (0..8u32).map(|i| i.count_ones() % 2).collect();
        let phase = vec![0.0; 8];
        let adjacency = build_interfaces(&phase, &region_of, 2, (2, 2, 2));
        assert_eq!(adjacency[0][&1].n, 12);
    }

    #[test]
    fn cost_follows_equation_nine() {
        let eight_pi_sq = 8.0 * PI * PI;
        // K = 0 exactly: rounding is unambiguous, so getting it wrong is most costly.
        let certain = Interface { n: 4, p: 0.0 };
        assert_eq!(certain.offset(), 0.0);
        assert!((certain.cost() - eight_pi_sq * 4.0 * 0.5).abs() < 1e-9);

        // K = 1/4 of a turn off: the penalty drops linearly.
        let quarter = Interface { n: 4, p: -TWO_PI * 4.0 * 0.25 };
        assert!((quarter.k() - 0.25).abs() < 1e-12);
        assert_eq!(quarter.offset(), 0.0);
        assert!((quarter.cost() - eight_pi_sq * 4.0 * 0.25).abs() < 1e-9);

        // K = half a turn: a coin toss, and worth nothing to decide.
        let ambiguous = Interface { n: 4, p: -TWO_PI * 4.0 * 0.5 };
        assert!(ambiguous.cost().abs() < 1e-9);

        // The count scales the penalty: a wider interface at the same K costs more.
        let wide = Interface { n: 40, p: 0.0 };
        assert!((wide.cost() - 10.0 * certain.cost()).abs() < 1e-9);

        // And the offset rounds, including away from zero at the half-turn.
        assert_eq!(Interface { n: 1, p: -TWO_PI * 1.6 }.offset(), 2.0);
        assert_eq!(Interface { n: 1, p: TWO_PI * 2.4 }.offset(), -2.0);
    }

    #[test]
    fn cost_is_never_negative() {
        let mut state = 0xDEADu64;
        for _ in 0..2000 {
            let iface = Interface {
                n: 1 + (lcg(&mut state) * 50.0) as u32,
                p: (lcg(&mut state) - 0.5) * 400.0,
            };
            assert!(iface.cost() >= 0.0, "{iface:?} gave a negative cost");
            assert!((iface.k() - iface.offset()).abs() <= 0.5 + 1e-12);
        }
    }

    #[test]
    fn candidates_order_by_cost_then_by_id() {
        let mut heap: BinaryHeap<Candidate> = BinaryHeap::new();
        heap.push(Candidate { cost: 1.0, a: 0, b: 1 });
        heap.push(Candidate { cost: 5.0, a: 7, b: 9 });
        heap.push(Candidate { cost: 5.0, a: 2, b: 3 });
        heap.push(Candidate { cost: 5.0, a: 2, b: 1 });
        let order: Vec<(u32, u32)> =
            std::iter::from_fn(|| heap.pop()).map(|c| (c.a, c.b)).collect();
        assert_eq!(order, vec![(2, 1), (2, 3), (7, 9), (0, 1)]);
    }
}

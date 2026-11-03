//! MEDI's region-growing phase unwrapping: `unwrapPhase` / `mexUnwrap.c` (Tian Liu), a
//! magnitude-guided, quality-ordered flood fill.
//!
//! Liu T, Wisnieff C, Lou M, Chen W, Spincemaille P, Wang Y. *Nonlinear formulation of the
//! magnetic field to source relationship for robust quantitative susceptibility mapping.*
//! Magn Reson Med 2013;69(2):467–476, <https://doi.org/10.1002/mrm.24272>. The MEDI toolbox's
//! pipeline script unwraps `Fit_ppm_complex`'s field with it (`README.m:80-95`):
//!
//! ```text
//! iMag = sqrt(sum(abs(iField).^2,4));
//! [iFreq_raw, N_std] = Fit_ppm_complex(iField);
//! iFreq = unwrapPhase(iMag, iFreq_raw, matrix_size);
//! ```
//!
//! # Reference code
//!
//! The original is the MEDI toolbox 2024.11.26's `functions/unwrapPhase.m` and the C it calls,
//! `functions/mexUnwrap.c` ("written by Tian Liu", mex interface by Dong Zhou, last update
//! 7.11.13). This module is a line-by-line port of that C, and reproduces it **bit for bit**:
//! `tests/medi_unwrap_cornell_parity.rs` checks every voxel against the verified QSM-CI
//! `fit-ppm-complex-cornell` container (`ghcr.io/astewartau/qsm-ci/matlab-medi@sha256:7efa4bd6…`,
//! `mexUnwrap` compiled from that `mexUnwrap.c` with R2023b's `mex` and identical to licensed
//! MATLAB; QSMxT/QSM-CI#300), on phantoms built to exercise heap ties, the noise level, a region
//! cut off from the seed and even and odd matrix sizes. From the echoes, the field-mapping stage
//! with [`B0EstimationMethod::ComplexFit`](crate::pipeline::B0EstimationMethod::ComplexFit) and
//! this unwrapper gives that container's total field to 8.9e-16 ppm (the units' rounding). The
//! fixture is sensitive to the heap: making `heap_pop` take the left child on a tie moves 541 of
//! its 2970 voxels, and making `heap_add` sift up on a tie 1017. Every decision below is the C's:
//!
//! - **Precision: single.** `mexUnwrap` copies the magnitude and the phase from MATLAB's doubles
//!   into `float` arrays (`mexFunction`), and every comparison and update is done on those floats.
//!   The output is the float phase written back as double, so every output value, including the
//!   voxels it never touches, is the input rounded to `f32`. This is kept (not double precision,
//!   unlike other ports here) because the growth order is decided by comparing `f32` magnitudes:
//!   two magnitudes that differ in double but round to the same float are a tie in the original,
//!   and the tie decides which neighbour a voxel is unwrapped against.
//! - **Quality map: the magnitude, nothing else.** The heap is ordered by the magnitude value
//!   alone (`heap_add` / `heap_pop` compare `.mag`). MEDI passes `iMag`, the root-sum-of-squares
//!   of the echo magnitudes; the caller passes whatever magnitude it wants used (see
//!   [`UnwrappingAlgorithm::MediRegionGrowing`](crate::pipeline::UnwrappingAlgorithm::MediRegionGrowing)
//!   for what the pipeline passes).
//! - **Seed: the middle of the field of view**, voxel `(nx/2, ny/2, nz/2)` in integer division
//!   (`mexFunction`: `dim[2]/2*dim[1]*dim[0] + dim[1]/2*dim[0] + dim[0]/2`, "choose the middle of
//!   the FOV as the seeding point"). Not the brightest voxel: the brightest voxel only sets the
//!   noise level. The seed is unwrapped (left as it is) whatever its magnitude, even below the
//!   noise level; if none of its neighbours is above the noise level, nothing else is unwrapped.
//! - **Noise level: `0.005 ×` the largest magnitude**, in single precision (`noise_ratio = 5e-3`,
//!   a `float`; `noise_level = MRimage[max_ind].mag * noise_ratio`). The largest magnitude is
//!   found with a strict `>` from voxel 0, so the first of equal maxima is taken (it only matters
//!   through its value). Only voxels with magnitude strictly above the noise level are ever
//!   planned; the rest keep their wrapped phase. [`MEDI_NOISE_RATIO`]. MEDI's `unwrapPhase` has no
//!   way to change it, and neither does this function.
//! - **No mask.** `unwrapPhase(iMag, iFreq_raw, matrix_size)` takes none: the growth runs over the
//!   whole array, limited only by the noise level, so the phase outside the brain can decide the
//!   path into it. Voxels that the growth never reaches (below the noise level, or above it but
//!   cut off from the seed by voxels below it) are returned as they came in (rounded to `f32`).
//! - **Neighbours: the 6-neighbourhood, in the order `x−1, x+1, y−1, y+1, z−1, z+1`**
//!   (`get_neighbors`), with x fastest in memory (MATLAB's column-major order; QSM.rs's layout).
//!   Neighbours outside the array are skipped.
//! - **Growth.** A binary max-heap of voxel indices keyed on magnitude (`heap_create`, `heap_add`,
//!   `heap_pop`). Each step takes the top voxel, unwraps it against the voxel it was planned from
//!   (its `pre`; the seed has none), pops it, and plans each neighbour that is unplanned and above
//!   the noise level, in neighbour order: its `pre` becomes the popped voxel, it is pushed, and it
//!   is marked planned. So a voxel is unwrapped against the first unwrapped neighbour that found
//!   it, not the best one, and `pre` is fixed when the voxel is planned, not when it is popped.
//! - **Heap order and ties, exactly.** `heap_add` sifts up while the parent's magnitude is
//!   strictly less than the child's (equal magnitudes stay below their parent). `heap_pop` moves
//!   the last element to the root, clears the vacated slot, and sifts down while the parent is
//!   strictly less than the chosen child, choosing the left child when its magnitude is strictly
//!   greater than the right's and the right child otherwise (so the right one on a tie), and the
//!   left one when the right is past the end. Ties therefore resolve by heap position, not by
//!   voxel index or insertion time; reproducing them needs this exact heap, so no standard
//!   priority queue is used.
//! - **The unwrap step.** `phase = phase − round((phase − phase[pre]) / (2π)) · 2π` with the C's
//!   types: the difference of the two floats is a float; it is divided by the double `2*PI`
//!   (`PI` = 3.1415926535897931, the same double as [`std::f64::consts::PI`]); the quotient is
//!   passed to the file's own `static float round(float)`, i.e. converted to float, then
//!   `floor(num + 0.5)` in double (round half up, not half away from zero), returned as float;
//!   that float times the int 2 is a float, times `PI` a double, subtracted from the float phase
//!   in double, and the result stored as float.
//!
//! The C also prints `"%d voxels are unwrapped"`; that is not reproduced.
//!
//! # Pipeline rules
//!
//! [`crate::pipeline::rules::RULES`] has `complex-fit.unwrap-region-growing` (a `Reference`
//! rule: MEDI's pipeline unwraps `Fit_ppm_complex`'s field with this method), which fires only
//! when the complex fit is chosen; ROMEO stays the default unwrapper.
//!
//! # Use
//!
//! `mexUnwrap.c` assumes the input is wrapped; in MEDI it is `Fit_ppm_complex`'s field per echo
//! spacing. The pipeline offers it as
//! [`UnwrappingAlgorithm::MediRegionGrowing`](crate::pipeline::UnwrappingAlgorithm::MediRegionGrowing),
//! on every field-mapping path; with [`B0EstimationMethod::ComplexFit`](crate::pipeline::B0EstimationMethod::ComplexFit)
//! it gives MEDI's field map (QSM-CI's `fit-ppm-complex-cornell` pair).

use crate::grid::Grid;

/// `mexUnwrap.c`'s `noise_ratio`: voxels with magnitude at or below this fraction of the
/// largest magnitude are never unwrapped. A `float` in the C, as here.
pub const MEDI_NOISE_RATIO: f32 = 5e-3;

/// `mexUnwrap.c`'s `#define PI 3.1415926535897931`: that literal is this same double.
const PI: f64 = std::f64::consts::PI;

/// Unwrap `phase` (radians) by MEDI's region growing, ordered by `magnitude`, as MEDI's
/// `unwrapPhase(magnitude, phase, matrix_size)` does. See the [module docs](self) for every
/// detail of the original that is reproduced.
///
/// Both arrays are column-major over `grid.dims` (x fastest). There is no mask: the growth runs
/// over the whole array, above the noise level ([`MEDI_NOISE_RATIO`] of the largest magnitude).
/// The result is computed in single precision, as in the original, so every value (also those
/// the growth does not reach, which keep their wrapped phase) is an `f32` widened to `f64`.
///
/// # Panics
///
/// If `phase` or `magnitude` does not have `nx·ny·nz` elements, or the array is empty.
pub fn unwrap_medi_region_growing(phase: &[f64], magnitude: &[f64], grid: &Grid) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let n = nx * ny * nz;
    assert!(n > 0, "unwrap_medi_region_growing: empty array");
    assert_eq!(phase.len(), n, "unwrap_medi_region_growing: phase has the wrong length");
    assert_eq!(magnitude.len(), n, "unwrap_medi_region_growing: magnitude has the wrong length");

    // mexFunction: `mag[i] = p[i]` and `phase[i] = p[i]`, double to float.
    let mag: Vec<f32> = magnitude.iter().map(|&m| m as f32).collect();
    let mut ph: Vec<f32> = phase.iter().map(|&p| p as f32).collect();

    // assemble_data: `if (mag[ind] > mag[*max_ind]) *max_ind = ind;` from max_ind = 0.
    let mut max_ind = 0usize;
    for i in 0..n {
        if mag[i] > mag[max_ind] {
            max_ind = i;
        }
    }
    let noise = mag[max_ind] * MEDI_NOISE_RATIO;

    // "choose the middle of the FOV as the seeding point"
    let seed = nz / 2 * ny * nx + ny / 2 * nx + nx / 2;

    let mut planned = vec![false; n];
    // `pre`: the voxel each one was planned from; None for the seed (the C's -1).
    let mut pre: Vec<Option<usize>> = vec![None; n];
    let mut heap = Heap { idx: vec![0usize; n], len: 0 };

    // unwrap(): heap_create(seed); heap_length++; pre = -1; planned = 1.
    heap.idx[0] = seed;
    heap.len = 1;
    planned[seed] = true;

    let mut nei = [None; 6];
    loop {
        let top = heap.idx[0];
        if let Some(p) = pre[top] {
            ph[top] = unwrap_against(ph[top], ph[p]);
        }
        neighbours(&mut nei, top, nx, ny, nz);
        let curr = heap.pop(&mag);
        debug_assert_eq!(curr, top);
        for &nb in nei.iter().flatten() {
            if !planned[nb] && mag[nb] > noise {
                pre[nb] = Some(curr);
                heap.add(&mag, nb);
                planned[nb] = true;
            }
        }
        if heap.len == 0 {
            break;
        }
    }

    ph.into_iter().map(f64::from).collect()
}

/// `phase - round((phase - reference)/(2*PI))*2*PI` with `mexUnwrap.c`'s types (module docs,
/// *The unwrap step*).
#[inline]
fn unwrap_against(phase: f32, reference: f32) -> f32 {
    let diff: f32 = phase - reference;
    let turns = round_c(((diff as f64) / (2.0 * PI)) as f32);
    ((phase as f64) - ((turns * 2.0) as f64) * PI) as f32
}

/// The file's `static float round(float num) { return floor(num+0.5); }`: `num + 0.5` and the
/// floor in double, the result converted to float.
#[inline]
fn round_c(num: f32) -> f32 {
    ((num as f64) + 0.5).floor() as f32
}

/// `get_neighbors`: x−1, x+1, y−1, y+1, z−1, z+1, `None` past the array's edge (the C's −1).
#[inline]
fn neighbours(nei: &mut [Option<usize>; 6], ind: usize, nx: usize, ny: usize, nz: usize) {
    let z = ind / (nx * ny);
    let y = (ind - z * nx * ny) / nx;
    let x = ind - y * nx - z * nx * ny;
    let at = |x: usize, y: usize, z: usize| x + y * nx + z * nx * ny;
    nei[0] = (x > 0).then(|| at(x - 1, y, z));
    nei[1] = (x + 1 < nx).then(|| at(x + 1, y, z));
    nei[2] = (y > 0).then(|| at(x, y - 1, z));
    nei[3] = (y + 1 < ny).then(|| at(x, y + 1, z));
    nei[4] = (z > 0).then(|| at(x, y, z - 1));
    nei[5] = (z + 1 < nz).then(|| at(x, y, z + 1));
}

/// `mexUnwrap.c`'s binary max-heap of voxel indices, keyed on the magnitude. Kept as the C writes
/// it (module docs, *Heap order and ties*): its tie behaviour decides the growth order.
struct Heap {
    /// `heap`: `calloc(numel, sizeof(int))`, so one slot per voxel, zero-initialised.
    idx: Vec<usize>,
    /// `heap_length`
    len: usize,
}

impl Heap {
    /// `heap_add`: append, then sift up while the parent's magnitude is strictly less.
    fn add(&mut self, mag: &[f32], index: usize) {
        self.idx[self.len] = index;
        let mut ind = self.len;
        while ind > 0 {
            let parent = (ind - 1) / 2;
            if mag[self.idx[parent]] < mag[self.idx[ind]] {
                self.idx.swap(parent, ind);
                ind = parent;
            } else {
                break;
            }
        }
        self.len += 1;
    }

    /// `heap_pop`: take the root, move the last element there (and zero its old slot), then sift
    /// down: the left child if its magnitude is strictly greater than the right's, else the
    /// right, else the left when the right is past the end; swap while the parent's magnitude is
    /// strictly less than that child's.
    fn pop(&mut self, mag: &[f32]) -> usize {
        self.len -= 1;
        let len = self.len;
        let ret = self.idx[0];
        self.idx[0] = self.idx[len];
        self.idx[len] = 0;
        let mut ind = 0usize;
        // `while ((ind*2+1 <= *length - 1) && goon)`: with `len` 0 the C's `*length - 1` is -1 in
        // int arithmetic, so the loop does not run.
        while len > 0 && ind * 2 + 1 < len {
            let left = ind * 2 + 1;
            let right = ind * 2 + 2;
            // The C compares heap[right] even when right == len (the slot just zeroed, still in
            // the array); the choice is then overridden to `left`, so only in-range rights matter.
            let mut child = if mag[self.idx[left]] > mag[self.idx[right]] {
                left
            } else {
                right
            };
            if child > len - 1 {
                child = left;
            }
            if mag[self.idx[ind]] < mag[self.idx[child]] {
                self.idx.swap(ind, child);
                ind = child;
            } else {
                break;
            }
        }
        ret
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    fn wrap(x: f64) -> f64 {
        x.sin().atan2(x.cos())
    }

    /// A smooth field that wraps several times comes back unwrapped, up to the seed's own turn
    /// (the seed keeps its wrapped value), to f32 rounding.
    #[test]
    fn unwraps_a_smooth_field() {
        let (nx, ny, nz) = (20, 17, 9);
        let g = Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);
        let n = nx * ny * nz;
        let truth: Vec<f64> = (0..n)
            .map(|i| {
                let (x, y, z) = ((i % nx) as f64, (i / nx % ny) as f64, (i / (nx * ny)) as f64);
                0.9 * x - 0.6 * y + 0.4 * z + 0.02 * x * y
            })
            .collect();
        let wrapped: Vec<f64> = truth.iter().map(|&t| wrap(t)).collect();
        let mag: Vec<f64> = (0..n).map(|i| 1.0 + (i % 7) as f64).collect();
        let u = unwrap_medi_region_growing(&wrapped, &mag, &g);
        let seed = nz / 2 * ny * nx + ny / 2 * nx + nx / 2;
        let offset = truth[seed] - u[seed];
        assert!((offset / TAU - (offset / TAU).round()).abs() < 1e-6);
        for i in 0..n {
            assert!((u[i] + offset - truth[i]).abs() < 1e-4, "voxel {i}: {} vs {}", u[i] + offset, truth[i]);
        }
    }

    /// Voxels at or below 0.5 % of the largest magnitude are never unwrapped, nor is a bright
    /// region they cut off from the seed; both keep their wrapped phase (as f32).
    #[test]
    fn noise_and_unreachable_voxels_keep_their_phase() {
        let (nx, ny, nz) = (12, 5, 1);
        let g = Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);
        let n = nx * ny * nz;
        let truth: Vec<f64> = (0..n).map(|i| 1.1 * (i % nx) as f64).collect();
        let wrapped: Vec<f64> = truth.iter().map(|&t| wrap(t)).collect();
        // Columns 0..=6 bright (seed at x = 6), column 7 at exactly the noise level, 8.. bright.
        let mag: Vec<f64> = (0..n).map(|i| if i % nx == 7 { 0.005 } else { 1.0 }).collect();
        let u = unwrap_medi_region_growing(&wrapped, &mag, &g);
        for i in 0..n {
            let x = i % nx;
            if x >= 7 {
                assert_eq!(u[i], wrapped[i] as f32 as f64, "x = {x} was touched");
            }
        }
        assert!((u[6] - u[0] - (truth[6] - truth[0])).abs() < 1e-4);
    }

    /// The seed is the middle of the field of view, even when it is below the noise level: then
    /// nothing else is unwrapped.
    #[test]
    fn a_dark_seed_unwraps_nothing() {
        let (nx, ny, nz) = (5, 5, 5);
        let g = Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);
        let n = nx * ny * nz;
        let wrapped: Vec<f64> = (0..n).map(|i| wrap(0.9 * i as f64)).collect();
        let mut mag = vec![1.0; n];
        let seed = 2 * 25 + 2 * 5 + 2;
        for d in [1usize, 5, 25] {
            mag[seed - d] = 0.0;
            mag[seed + d] = 0.0;
        }
        let u = unwrap_medi_region_growing(&wrapped, &mag, &g);
        for i in 0..n {
            assert_eq!(u[i], wrapped[i] as f32 as f64);
        }
    }

    /// `round` is the C's `floor(x + 0.5)`: halves round up, also below zero.
    #[test]
    fn round_is_floor_of_x_plus_a_half() {
        assert_eq!(round_c(0.5), 1.0);
        assert_eq!(round_c(-0.5), 0.0);
        assert_eq!(round_c(-1.5), -1.0);
        assert_eq!(round_c(2.4999), 2.0);
    }
}

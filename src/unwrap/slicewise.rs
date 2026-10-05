//! Slice-wise (2D) phase unwrapping for 2D multi-slice acquisitions.
//!
//! In a 2D multi-slice acquisition each slice is excited separately and carries its own
//! constant receive phase offset; interleaved acquisition adds slice-to-slice jumps on top.
//! Every unwrapper in [`crate::unwrap`] reads across slice boundaries, and all of them are
//! corrupted by that:
//!
//! - [`unwrap_romeo`](crate::unwrap::unwrap_romeo) and
//!   [`unwrap_bestpath`](crate::unwrap::unwrap_bestpath) grow regions in 3D, so one bad slice
//!   boundary carries a wrong offset across everything downstream of it.
//! - [`laplacian_unwrap`](crate::unwrap::laplacian_unwrap) differentiates along z straight
//!   across the jump. It degrades more gracefully — it is a global FFT solve, so the error
//!   stays diffuse rather than propagating — but it is still wrong.
//!
//! On clean data the region growers look like they get away with it, and it is worth being
//! precise about why they do not. Their quality weighting puts the incoherent z edges last,
//! so a slice tends to be entered once and to come out a whole number of wraps from the
//! truth — the error a multi-echo fit is supposed to absorb. The catch is *which* whole
//! number: the jump the grower rounds is `offset_step + field_step * TE`, so it depends on
//! the echo. The per-slice wrap then differs between echoes, the offset stops being
//! echo-independent, and the fit that was going to cancel it is corrupted instead.
//! [`correct_multi_echo_wraps`](crate::unwrap::correct_multi_echo_wraps) cannot repair it,
//! because it corrects the whole volume at once and the error is per slice. Measured on this
//! module's test phantom, neither 3D path survives both offset patterns on its own — the
//! template path misreads random offsets, the individual path misreads interleaved ones.
//!
//! [`unwrap_slicewise`] runs the chosen unwrapper on each slice independently, so z is never
//! differentiated across. Each slice is handed to the same code with a grid of
//! `(d0, d1, 1)`; the ±z neighbours then fall outside the volume and are skipped by the
//! existing bounds checks, so the unwrappers themselves are untouched.
//!
//! # What this costs
//!
//! Unwrapping a slice on its own fixes it only up to a multiple of 2π: nothing in the slice
//! says which multiple. That is inherent to the problem, not to this implementation — the
//! information that ties slices together is exactly the information a 2D multi-slice
//! acquisition does not record.
//!
//! Multi-echo data recovers it. The slice offset is echo-independent, so in
//! `phi(TE) = phi0 + gamma*dB*TE` it lands entirely in the intercept, and a fit across echoes
//! returns the field from the slope with `phi0` discarded — slice offset and leftover 2π
//! alike. QSMxT exposes that as `--b0-estimation linear-fit`.
//!
//! # Where inter-echo consistency belongs
//!
//! That rescue only works if the leftover 2π is the *same* for every echo of a slice. Unwrap
//! each (slice, echo) independently and it is not: each one picks its own multiple, the
//! intercept stops being echo-independent, and the fit is corrupted rather than saved.
//!
//! So [`unwrap_slicewise_multi_echo`] repairs it here, in the unwrapper, rather than leaving
//! it to the B0 fit. The reason is not preference — it is that the repair needs the *wrapped
//! input* phase, and the unwrapper is the only stage that holds both that and its own output.
//! Given both, the echo-to-echo evolution is `wrap(p_e - p_{e-1})` with no unwrapping and no
//! knowledge of the offset, which pins the relative 2π exactly (see
//! [`enforce_inter_echo_consistency`]). A B0 fit sees only unwrapped phase and would
//! have to infer the jumps from the fit residual, which is both weaker and specific to one
//! estimator — QSMxT's other B0 estimators would silently keep the corruption.
//!
//! The division of labour that leaves is:
//!
//! - this module guarantees the leftover `2*pi*k` is **the same for every echo of a slice**;
//! - the B0 fit cancels it, together with the physical slice offset, because both are now
//!   echo-independent.
//!
//! # Which half does which job
//!
//! Measured on the qsm-forward 2D phantom at 3 mm slices with interleaved offsets, the two
//! halves fix different things, and it is worth being exact about which, because the obvious
//! reading — that slice-wise unwrapping is what rescues the field map — is wrong.
//!
//! **The consistency pass fixes the fitted field, from any starting point.** Correlation of a
//! linear-fit B0 estimate against the field the signal was generated from: 3D ROMEO alone
//! reaches 0.20, and the *same* 3D output put through [`enforce_inter_echo_consistency`]
//! reaches 0.983. So does the raw wrapped phase with no spatial unwrapping at all. A fitted
//! slope is blind to a constant per voxel, so whatever wrap state a voxel starts in lands in
//! the intercept and the fit discards it; all the slope needs is that consecutive echoes differ
//! by their measured evolution, which is exactly what the pass enforces. **If a linear-fit B0
//! map is all you want, you do not need this module** — you need the pass, which is public and
//! works on any unwrapper's output.
//!
//! **Slice-wise unwrapping fixes the phase itself.** The clean way to see this is on the *first*
//! echo, because [`enforce_inter_echo_consistency`] loops `for e in 1..n_echoes` and so provably
//! cannot touch it — any difference there belongs to the spatial unwrapping and nothing else, with
//! no null needed to rule the pass out. Counting in-plane neighbour pairs that jump more than π in
//! echo 1 of that session: **1432** in the wrapped input, **1432** after the pass (identical, as
//! the structure requires), **0** after slice-wise unwrapping.
//!
//! By the last echo both mechanisms are in play and the gap narrows but holds: 1433 wrapped, 2142
//! with the pass alone, 796 with slice-wise. The pass alone is *worse* there than not unwrapping,
//! because it propagates echo 1's wrap state faithfully into every later echo.
//!
//! Anything reading the unwrapped phase rather than its TE-slope — a single-echo field map, a
//! non-linear B0 estimator, phase fed straight to background removal — needs the spatial
//! unwrapping, and on 2D multi-slice data that means per slice.
//!
//! On the matched session with no offsets slice-wise is still ahead, 0.983 against 3D's 0.936:
//! 3 mm slices alone produce through-slice phase steps large enough to mislead a region grower,
//! with no receive offset involved. Slice-wise
//! [`UnwrapMethod::Laplacian`] reaches only 0.81 on the fit — its Poisson solution is not a
//! whole number of wraps from the truth to begin with, so the pass cannot re-seat it cleanly.
//! Prefer ROMEO or best path.
//!
//! Single-echo data has no second measurement and so no way to close the gap. [`unwrap_slicewise`]
//! returns each slice unwrapped and internally consistent, and the between-slice offsets
//! survive into the field map. That is a property of the acquisition; background field removal
//! will absorb a smoothly varying part of it, but not jumps.
//!
//! # Slice gaps
//!
//! This module does not care whether the slices are contiguous. Unwrapping is per slice and
//! never relates one slice to another spatially, so it is valid on gapped data — unlike the
//! dipole kernel, which is not (see [`crate::bgremove`]).

use std::f64::consts::PI;

use crate::Grid;
use super::laplacian::{laplacian_unwrap, wrap};
use super::{unwrap_bestpath, unwrap_romeo, BestPathParams, RomeoParams, UnwrapMethod};
use crate::grid::SliceLayout;

const TWO_PI: f64 = 2.0 * PI;

/// Parameters for slice-wise unwrapping.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct SliceWiseParams {
    /// Axis the slices are stacked along: 0 = x, 1 = y, 2 = z (default: 2).
    ///
    /// For an axial 2D acquisition stored in the usual orientation this is 2. A sagittal or
    /// coronal 2D acquisition stacks along another axis, and setting this is cheaper and
    /// safer than transposing the volume.
    pub slice_axis: usize,
    /// Parameters for [`UnwrapMethod::Romeo`]; ignored by the other methods.
    pub romeo: RomeoParams,
    /// Parameters for [`UnwrapMethod::BestPath`]; ignored by the other methods.
    pub bestpath: BestPathParams,
    /// Make each slice's leftover `2*pi*k` the same across echoes, so a fit across echoes
    /// cancels it (default: true). Multi-echo only; see the module docs.
    pub enforce_inter_echo_consistency: bool,
}

impl Default for SliceWiseParams {
    fn default() -> Self {
        Self {
            slice_axis: 2,
            romeo: RomeoParams::default(),
            bestpath: BestPathParams::default(),
            enforce_inter_echo_consistency: true,
        }
    }
}

/// Label the connected components of a slice mask, 4-connected.
///
/// 4-connectivity, not 8, because that is what the region-growing unwrappers can actually
/// traverse: their neighbourhood is the six face neighbours, which in plane is four.
/// Returns labels (0 = outside the mask) and the number of components.
fn label_components(mask: &[u8], d0: usize, d1: usize) -> (Vec<u32>, u32) {
    let mut labels = vec![0u32; mask.len()];
    let mut next = 0u32;
    let mut stack: Vec<usize> = Vec::new();
    for seed in 0..mask.len() {
        if mask[seed] == 0 || labels[seed] != 0 {
            continue;
        }
        next += 1;
        labels[seed] = next;
        stack.push(seed);
        while let Some(p) = stack.pop() {
            let (a, b) = (p % d0, p / d0);
            let mut visit = |q: usize, stack: &mut Vec<usize>| {
                if mask[q] != 0 && labels[q] == 0 {
                    labels[q] = next;
                    stack.push(q);
                }
            };
            if a + 1 < d0 { visit(p + 1, &mut stack); }
            if a > 0 { visit(p - 1, &mut stack); }
            if b + 1 < d1 { visit(p + d0, &mut stack); }
            if b > 0 { visit(p - d0, &mut stack); }
        }
    }
    (labels, next)
}

/// Re-anchor a Laplacian solution so it differs from the wrapped phase by whole 2π.
///
/// [`laplacian_unwrap`] solves a Poisson equation and zeroes the DC mode, so its output is
/// only defined up to an arbitrary real constant. In 3D that constant is global and harmless
/// — a receive phase offset, which the background removal or a multi-echo fit absorbs. Per
/// slice it is neither: an arbitrary real offset on each slice is worse than the 2π ambiguity
/// slice-wise unwrapping is entitled to, and it would make the inter-echo correction
/// meaningless, since that correction can only move a slice by whole multiples of 2π.
///
/// Subtracting the circular mean of `u - p` over each component restores the entitlement:
/// afterwards `u - p` is a whole multiple of 2π, and the only freedom left is which one.
/// The region-growing unwrappers already satisfy that by construction, so they skip this.
fn reanchor_to_wrapped(u: &mut [f64], p: &[f64], labels: &[u32], n_components: u32) {
    for label in 1..=n_components {
        let (mut sin_sum, mut cos_sum) = (0.0, 0.0);
        for i in 0..u.len() {
            if labels[i] == label {
                let d = u[i] - p[i];
                sin_sum += d.sin();
                cos_sum += d.cos();
            }
        }
        let offset = sin_sum.atan2(cos_sum);
        for i in 0..u.len() {
            if labels[i] == label {
                u[i] -= offset;
            }
        }
    }
}

/// Unwrap each slice independently, in plane.
///
/// `method` selects the unwrapper, which runs unchanged on a `(d0, d1, 1)` grid per slice so
/// the slice axis is never differentiated across. Masked-out voxels keep whatever the chosen
/// unwrapper leaves them: zero for [`UnwrapMethod::Laplacian`], the wrapped input for the
/// region-growing methods.
///
/// The result is self-consistent within each slice and determined only up to a multiple of 2π
/// between slices. For multi-echo data use [`unwrap_slicewise_multi_echo`], which makes that
/// multiple echo-independent so a fit across echoes cancels it. See the module docs.
///
/// # Arguments
/// * `phase` - Wrapped phase (nx * ny * nz)
/// * `mag` - Magnitude for the weighting of [`UnwrapMethod::Romeo`]; may be empty
/// * `mask` - Binary mask (nx * ny * nz), 1 = inside ROI
/// * `method` - Which unwrapper to run on each slice
/// * `params` - Slice axis and the per-method parameters
/// * `grid` - Volume grid
///
/// # Panics
/// If `params.slice_axis` is not 0, 1 or 2, or if `phase`/`mask` do not match `grid`.
pub fn unwrap_slicewise(
    phase: &[f64],
    mag: &[f64],
    mask: &[u8],
    method: UnwrapMethod,
    params: &SliceWiseParams,
    grid: &Grid,
) -> Vec<f64> {
    let n_total = grid.n_total();
    assert_eq!(phase.len(), n_total, "phase length must match grid dimensions");
    assert_eq!(mask.len(), n_total, "mask length must match grid dimensions");

    let layout = SliceLayout::new(grid, params.slice_axis);
    let n = layout.n_in_slice();
    let mut out = vec![0.0; n_total];
    let (mut p, mut m, mut k) = (vec![0.0; n], vec![0.0; n], vec![0u8; n]);

    for s in 0..layout.n_slices {
        layout.gather(phase, s, &mut p);
        layout.gather(mask, s, &mut k);
        let mag_slice: &[f64] = if mag.is_empty() {
            &[]
        } else {
            layout.gather(mag, s, &mut m);
            &m
        };

        let u = match method {
            UnwrapMethod::Romeo => {
                // ROMEO grows outward from one seed, so a slice whose mask has fallen into
                // several pieces - the temporal lobes low in the head, an eye, anything the
                // brain mask separates in plane - would leave every piece but the seed's
                // still wrapped. Run it once per piece. Whole slices usually have one.
                let (labels, n_components) = label_components(&k, layout.d0, layout.d1);
                if n_components <= 1 {
                    unwrap_romeo(&p, mag_slice, None, 0.0, 0.0, &k, &params.romeo, &layout.grid)
                } else {
                    let mut u = p.clone();
                    let mut sub = vec![0u8; n];
                    for label in 1..=n_components {
                        for (c, &l) in sub.iter_mut().zip(labels.iter()) {
                            *c = u8::from(l == label);
                        }
                        let part = unwrap_romeo(
                            &p, mag_slice, None, 0.0, 0.0, &sub, &params.romeo, &layout.grid,
                        );
                        for (idx, &l) in labels.iter().enumerate() {
                            if l == label {
                                u[idx] = part[idx];
                            }
                        }
                    }
                    u
                }
            }
            // Best path sorts every edge globally and merges, and no edge crosses between
            // components, so the pieces are already independent.
            UnwrapMethod::BestPath => unwrap_bestpath(&p, &k, &params.bestpath, &layout.grid),
            UnwrapMethod::Laplacian => {
                let mut u = laplacian_unwrap(&p, &k, &layout.grid);
                let (labels, n_components) = label_components(&k, layout.d0, layout.d1);
                reanchor_to_wrapped(&mut u, &p, &labels, n_components);
                // the solve wrote zeros outside the mask; keep them zero after re-anchoring
                for (v, &inside) in u.iter_mut().zip(k.iter()) {
                    if inside == 0 {
                        *v = 0.0;
                    }
                }
                u
            }
        };
        layout.scatter(&mut out, s, &u);
    }
    out
}

/// Unwrap each echo slice-wise, then make each slice's leftover 2π the same across echoes.
///
/// Each (slice, echo) is unwrapped independently by [`unwrap_slicewise`] and so picks its own
/// multiple of 2π. Left alone, that breaks the one thing that rescues 2D multi-slice data: the
/// slice offset is echo-independent and cancels in a fit across echoes, but only while the
/// leftover 2π is echo-independent too. [`enforce_inter_echo_consistency`] makes it so,
/// unless `params.enforce_inter_echo_consistency` is false.
///
/// `tes` is used only for the phase-gradient-coherence weighting of [`UnwrapMethod::Romeo`],
/// which compares each echo against its neighbour. The consistency correction itself does not
/// use it, and so does not care whether the echoes are evenly spaced.
///
/// # Arguments
/// * `phases` - Wrapped phase per echo, each (nx * ny * nz)
/// * `mags` - Magnitude per echo; may be empty
/// * `tes` - Echo times in seconds, one per echo
/// * `mask` - Binary mask (nx * ny * nz), 1 = inside ROI
/// * `method` - Which unwrapper to run on each slice
/// * `params` - Slice axis, per-method parameters, and the consistency switch
/// * `grid` - Volume grid
///
/// # Panics
/// If `phases` is empty or `tes` has a different length.
pub fn unwrap_slicewise_multi_echo<P: AsRef<[f64]>, M: AsRef<[f64]>>(
    phases: &[P],
    mags: &[M],
    tes: &[f64],
    mask: &[u8],
    method: UnwrapMethod,
    params: &SliceWiseParams,
    grid: &Grid,
) -> Vec<Vec<f64>> {
    let n_echoes = phases.len();
    assert!(n_echoes > 0, "phases must have at least one echo");
    assert_eq!(n_echoes, tes.len(), "phases and tes must have same length");

    let mut result: Vec<Vec<f64>> = (0..n_echoes)
        .map(|e| {
            let mag = if mags.is_empty() { &[] as &[f64] } else { mags[e].as_ref() };
            unwrap_slicewise(phases[e].as_ref(), mag, mask, method, params, grid)
        })
        .collect();

    if params.enforce_inter_echo_consistency && n_echoes > 1 {
        let wrapped: Vec<&[f64]> = phases.iter().map(|p| p.as_ref()).collect();
        enforce_inter_echo_consistency(&mut result, &wrapped, mask);
    }
    result
}

/// Re-seat every echo on the first, so the echoes of a voxel differ by their measured
/// phase evolution and nothing else.
///
/// Slice-wise unwrapping leaves echo `e` as `phi + 2*pi*k`, and the field is recoverable from
/// a fit across echoes only if `k` does not depend on `e`. This makes it so. It does not, and
/// cannot, say what `k` is — the leftover wrap stays in the intercept, along with the physical
/// slice offset, and the fit discards both.
///
/// The relative wrap is pinned by the wrapped input, without unwrapping anything and without
/// knowing the slice offset. Writing `p` for the wrapped phase and `u` for the unwrapped,
///
/// ```text
/// u_e - u_{e-1} - wrap(p_e - p_{e-1}) = 2*pi * (an integer)
/// ```
///
/// because `wrap(p_e - p_{e-1})` *is* the echo-to-echo evolution, wrapped. Rounding that to
/// the nearest multiple of 2π and subtracting it leaves `u_e - u_{e-1}` equal to the measured
/// step exactly. Formulating it on the wrapped difference rather than against a TE-scaled
/// reference is what lets it survive an echo-independent offset, which the TE-scaled form of
/// [`correct_multi_echo_wraps`](crate::unwrap::correct_multi_echo_wraps) would read as field.
///
/// # This is per voxel, deliberately
///
/// An earlier version of this did it per connected component of each slice, taking a median,
/// on the reasoning that slice-wise unwrapping offsets a whole component at a time. That
/// reasoning is wrong in practice: a region grower that cannot route through the slice axis
/// has to cross every in-plane fringe head-on, and where one is ambiguous it leaves a *part*
/// of the component offset by 2π. Measured on the qsm-forward 2D phantom, the wrap count
/// varies within a single slice of a single echo (2-3 distinct values), so a single median per
/// component has nothing it can do. Correcting per voxel repairs those plateaus as well, and
/// took the fitted field from r = -0.44 to r = 0.98 against ground truth during development.
/// The per-slice variant was not kept, so that figure records the decision rather than a test.
///
/// # This is the half that fixes a linear-fit B0 map
///
/// Public, and deliberately not tied to slice-wise unwrapping: it repairs any unwrapper's
/// output, including none at all. On the phantom's interleaved-offset session it takes 3D ROMEO
/// from 0.20 to 0.983 and the raw wrapped phase from -0.01 to 0.983. What it does *not* do is
/// make the phase spatially continuous — it propagates the first echo's wrap state faithfully
/// into the rest, which on that session leaves more in-plane 2π jumps than not unwrapping at
/// all. See the module docs for the division of labour.
///
/// # What it assumes
///
/// That the true phase evolution between consecutive echoes stays inside ±π, which is the
/// usual design constraint on echo spacing and the same assumption
/// [`correct_multi_echo_wraps`](crate::unwrap::correct_multi_echo_wraps) makes. A voxel whose
/// evolution exceeds it is not recoverable from this data by any means that only sees the
/// wrapped steps, and will carry a slope error of `2*pi / delta_TE`.
///
/// # Arguments
/// * `unwrapped` - Unwrapped phase per echo, modified in place
/// * `wrapped` - The wrapped input phase per echo
/// * `mask` - Binary mask (nx * ny * nz), 1 = inside ROI; voxels outside are left alone
pub fn enforce_inter_echo_consistency(
    unwrapped: &mut [Vec<f64>],
    wrapped: &[&[f64]],
    mask: &[u8],
) {
    let n_echoes = unwrapped.len();
    assert_eq!(n_echoes, wrapped.len(), "unwrapped and wrapped must have same length");
    if n_echoes < 2 {
        return;
    }
    for e in 1..n_echoes {
        let (before, after) = unwrapped.split_at_mut(e);
        let previous = &before[e - 1];
        for (i, value) in after[0].iter_mut().enumerate() {
            if mask[i] == 0 {
                continue;
            }
            let step = wrap(wrapped[e][i] - wrapped[e - 1][i]);
            let excess = (*value - previous[i] - step) / TWO_PI;
            if excess.is_finite() {
                *value -= TWO_PI * excess.round();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // What gates these tests, recorded so the sweep can be reconstructed rather than
    // re-derived. The script that ran it was scratch; the list is the part worth keeping.
    //
    // Each of these was applied and confirmed to fail at least one assertion below: the slice
    // index dropping its slice offset; the slice grid keeping the volume's depth instead of 1;
    // in-slice axes transposed; components merged under 8-connectivity; the Laplacian skipping
    // its re-anchoring; ROMEO run once per slice instead of once per connected component;
    // `find_seed_point` keeping a centre of mass that lies outside the mask; and five on the
    // consistency pass - switched off, chained against echo 0 rather than the previous echo,
    // referenced against a TE-scaled value instead of the wrapped step, truncating instead of
    // rounding, and sign-flipped.
    //
    // Two nulls matter more than any of those.
    // `the_consistency_pass_is_what_fixes_the_fit_even_without_slice_wise_unwrapping` is the
    // null for every fit-correlation claim here: the pass alone reaches 0.983 from raw wrapped
    // phase, so no fit-based assertion in this module may be read as evidence about the spatial
    // unwrapping. The echo-0 identity in
    // `slice_wise_unwrapping_is_what_makes_the_phase_spatially_continuous` is the opposite and
    // stronger case: confound-free by construction, because the pass only ever writes echoes
    // 1.., so nothing needs ruling out experimentally.

    const NX: usize = 24;
    const NY: usize = 24;
    const NZ: usize = 8;
    const TES: [f64; 4] = [4e-3, 12e-3, 20e-3, 28e-3];

    fn grid() -> Grid {
        Grid::new(NX, NY, NZ, 1.0, 1.0, 3.0)
    }

    /// Field in rad/s. Smooth and well below one radian per voxel in plane, but large enough
    /// that `b * TE` wraps a couple of times by the last echo.
    fn field() -> Vec<f64> {
        let mut b = vec![0.0; NX * NY * NZ];
        for k in 0..NZ {
            for j in 0..NY {
                for i in 0..NX {
                    // the in-plane pattern shifts with z, so the slice-to-slice difference
                    // varies across the slice as a real field does - without that the
                    // difference is constant and 3D unwrapping absorbs the offsets cleanly
                    let fx = (2.0 * PI * i as f64 / NX as f64 + 0.6 * k as f64).cos();
                    let fy = (2.0 * PI * j as f64 / NY as f64).cos();
                    b[i + j * NX + k * NX * NY] =
                        100.0 * (fx + fy) * (1.0 + 0.3 * k as f64 / NZ as f64);
                }
            }
        }
        b
    }

    /// Per-slice receive phase offsets, as a 2D multi-slice acquisition produces them.
    fn slice_offsets(interleaved: bool) -> Vec<f64> {
        if interleaved {
            (0..NZ).map(|s| if s % 2 == 0 { -1.5 } else { 1.5 }).collect()
        } else {
            // arbitrary but fixed, spread over the full circle
            (0..NZ).map(|s| -PI + TWO_PI * ((s * 7 + 3) % NZ) as f64 / NZ as f64).collect()
        }
    }

    /// Smooth echo-independent receive phase, spanning many radians as a real coil's does
    /// (~17 rad on the qsm-forward head phantom). Without one the model's phase is only a
    /// couple of radians, every plausible choice of reference in the consistency pass rounds
    /// to the same integer, and the tests cannot tell a correct reference from a wrong one.
    fn receive_phase() -> Vec<f64> {
        let mut r = vec![0.0; NX * NY * NZ];
        for k in 0..NZ {
            for j in 0..NY {
                for i in 0..NX {
                    let fx = (PI * i as f64 / NX as f64).cos();
                    let fy = (PI * j as f64 / NY as f64).cos();
                    r[i + j * NX + k * NX * NY] = 7.5 * fx + 4.0 * fy;
                }
            }
        }
        r
    }

    /// True (unwrapped) phase of echo `e`, built from the model rather than from any
    /// unwrapper: `phi = receive_phase + offset(slice) + b * TE`.
    fn true_phase(b: &[f64], offsets: &[f64], te: f64) -> Vec<f64> {
        let receive = receive_phase();
        let mut phi = vec![0.0; NX * NY * NZ];
        for (idx, v) in phi.iter_mut().enumerate() {
            *v = receive[idx] + offsets[idx / (NX * NY)] + b[idx] * te;
        }
        phi
    }

    #[test]
    fn the_model_phase_is_demanding_enough_to_be_worth_testing_against() {
        // Guards the tests below: the receive phase has to span several wraps (so the
        // consistency pass has a real reference to get right) while staying gentle enough in
        // plane to be unwrappable, and the echo-to-echo step has to stay inside +/-pi.
        let b = field();
        let truth = true_phase(&b, &slice_offsets(false), TES[3]);
        let span = truth.iter().cloned().fold(f64::MIN, f64::max)
            - truth.iter().cloned().fold(f64::MAX, f64::min);
        assert!(span > 6.0 * PI, "phase spans only {span} rad, under three wraps");

        let mut worst_inplane: f64 = 0.0;
        for k in 0..NZ {
            for j in 0..NY {
                for i in 0..NX - 1 {
                    let a = i + j * NX + k * NX * NY;
                    worst_inplane = worst_inplane.max((truth[a + 1] - truth[a]).abs());
                }
            }
        }
        assert!(worst_inplane < PI, "in-plane step reaches {worst_inplane} rad, not unwrappable");

        let step = true_phase(&b, &slice_offsets(false), TES[1]);
        let first = true_phase(&b, &slice_offsets(false), TES[0]);
        let worst_step = step.iter().zip(&first).map(|(a, c)| (a - c).abs()).fold(0.0_f64, f64::max);
        assert!(worst_step < PI, "echo-to-echo step reaches {worst_step} rad");
    }

    fn wrap_all(phi: &[f64]) -> Vec<f64> {
        phi.iter().map(|&v| wrap(v)).collect()
    }

    fn full_mask() -> Vec<u8> {
        vec![1u8; NX * NY * NZ]
    }

    /// Largest |in-plane gradient| error against the truth. A per-slice constant - the part a
    /// 2D multi-slice acquisition genuinely cannot determine - cancels out of this, so it
    /// measures only whether the structure inside each slice survived.
    fn max_inplane_gradient_error(u: &[f64], truth: &[f64]) -> f64 {
        let mut worst: f64 = 0.0;
        for k in 0..NZ {
            for j in 0..NY {
                for i in 0..NX - 1 {
                    let a = i + j * NX + k * NX * NY;
                    worst = worst.max(((u[a + 1] - u[a]) - (truth[a + 1] - truth[a])).abs());
                }
            }
            for j in 0..NY - 1 {
                for i in 0..NX {
                    let a = i + j * NX + k * NX * NY;
                    worst = worst.max(((u[a + NX] - u[a]) - (truth[a + NX] - truth[a])).abs());
                }
            }
        }
        worst
    }

    /// Per-voxel least-squares slope of phase against TE - the `linear-fit` B0 estimate.
    fn fit_slope(unwrapped: &[Vec<f64>], tes: &[f64]) -> Vec<f64> {
        let n = tes.len() as f64;
        let te_mean = tes.iter().sum::<f64>() / n;
        let sxx: f64 = tes.iter().map(|t| (t - te_mean).powi(2)).sum();
        (0..unwrapped[0].len())
            .map(|i| {
                let y_mean = unwrapped.iter().map(|u| u[i]).sum::<f64>() / n;
                let sxy: f64 = tes
                    .iter()
                    .zip(unwrapped)
                    .map(|(t, u)| (t - te_mean) * (u[i] - y_mean))
                    .sum();
                sxy / sxx
            })
            .collect()
    }

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f64, f64::max)
    }

    #[test]
    fn label_components_separates_blobs_and_is_four_connected() {
        // two 2x2 blobs with a clear gap
        let (d0, d1) = (8usize, 4usize);
        let mut mask = vec![0u8; d0 * d1];
        for (a, b) in [(0, 0), (1, 0), (0, 1), (1, 1), (5, 2), (6, 2), (5, 3), (6, 3)] {
            mask[a + b * d0] = 1;
        }
        let (labels, n) = label_components(&mask, d0, d1);
        assert_eq!(n, 2);
        assert_eq!(labels[0], labels[1 + d0]);
        assert_ne!(labels[0], labels[5 + 2 * d0]);
        assert_eq!(labels.iter().filter(|&&l| l == 1).count(), 4);
        assert_eq!(labels.iter().filter(|&&l| l == 0).count(), d0 * d1 - 8);

        // voxels touching only at a corner are two components under 4-connectivity
        let mut diag = vec![0u8; d0 * d1];
        diag[1 + d0] = 1;
        diag[2 + 2 * d0] = 1;
        assert_eq!(label_components(&diag, d0, d1).1, 2);

        // and are one component once the corner is bridged
        diag[2 + d0] = 1;
        assert_eq!(label_components(&diag, d0, d1).1, 1);
    }

    // ---- the headline behaviour -------------------------------------------------

    fn methods() -> [(UnwrapMethod, &'static str); 3] {
        [
            (UnwrapMethod::Romeo, "ROMEO"),
            (UnwrapMethod::BestPath, "best path"),
            (UnwrapMethod::Laplacian, "Laplacian"),
        ]
    }

    #[test]
    fn slicewise_recovers_in_plane_structure_through_slice_offsets() {
        let b = field();
        let offsets = slice_offsets(false);
        let mask = full_mask();
        let params = SliceWiseParams::default();

        for (method, name) in methods() {
            let truth = true_phase(&b, &offsets, TES[3]);
            let wrapped = wrap_all(&truth);
            // the data really does wrap, or the test proves nothing about unwrapping
            assert!(max_abs_diff(&wrapped, &truth) > 2.0, "{name}: input does not wrap");

            let u = unwrap_slicewise(&wrapped, &[], &mask, method, &params, &grid());
            let err = max_inplane_gradient_error(&u, &truth);
            assert!(err < 0.05, "{name}: in-plane structure lost, max gradient error {err}");
        }
    }

    #[test]
    fn slicewise_leaves_each_slice_a_whole_number_of_wraps_from_the_truth() {
        let b = field();
        let offsets = slice_offsets(false);
        let mask = full_mask();
        let truth = true_phase(&b, &offsets, TES[3]);
        let wrapped = wrap_all(&truth);

        // the region-growing methods land exactly on the truth plus 2*pi*k; the Laplacian
        // solves a Poisson equation and only approaches it, hence the two tolerances
        for (method, name, tol) in [
            (UnwrapMethod::Romeo, "ROMEO", 1e-9),
            (UnwrapMethod::BestPath, "best path", 1e-9),
            (UnwrapMethod::Laplacian, "Laplacian", 0.1),
        ] {
            let u = unwrap_slicewise(
                &wrapped, &[], &mask, method, &SliceWiseParams::default(), &grid(),
            );
            for s in 0..NZ {
                let base = s * NX * NY;
                let k = ((u[base] - truth[base]) / TWO_PI).round();
                for p in 0..NX * NY {
                    let residual = u[base + p] - truth[base + p] - TWO_PI * k;
                    assert!(
                        residual.abs() < tol,
                        "{name}: slice {s} is not a constant {k} wraps from the truth \
                         (residual {residual} at {p})"
                    );
                }
            }
        }
    }

    /// Counts in-plane neighbour pairs that jump more than pi. A fitted slope cannot see this —
    /// it is blind to a constant per voxel — so it is the measure on which slice-wise unwrapping
    /// and the consistency pass come apart.
    fn inplane_jumps(u: &[f64]) -> usize {
        let mut n = 0usize;
        for k in 0..NZ {
            for j in 0..NY {
                for i in 0..NX - 1 {
                    let a = i + j * NX + k * NX * NY;
                    if (u[a + 1] - u[a]).abs() > PI {
                        n += 1;
                    }
                }
            }
        }
        n
    }

    #[test]
    fn the_consistency_pass_is_what_fixes_the_fit_even_without_slice_wise_unwrapping() {
        // The correction this module's docs used to credit to slice-wise unwrapping. A slope is
        // blind to a constant per voxel, so the fit needs only that consecutive echoes differ by
        // their measured evolution - which the pass enforces from any starting point, including
        // none at all. Pinned because the module docs now say so, and because it tells a caller
        // who only wants a linear-fit B0 map that they do not need this module.
        let Inputs { field: b, wrapped, .. } = multi_echo_inputs(true);
        let mask = full_mask();
        let refs: Vec<&[f64]> = wrapped.iter().map(|p| p.as_slice()).collect();

        // raw wrapped phase, no spatial unwrapping whatsoever
        let mut raw = wrapped.clone();
        let before = max_abs_diff(&fit_slope(&raw, &TES), &b);
        assert!(before > 50.0, "the raw wrapped fit should be badly wrong: {before}");
        enforce_inter_echo_consistency(&mut raw, &refs, &mask);
        let after = max_abs_diff(&fit_slope(&raw, &TES), &b);
        assert!(after < 1e-6, "the pass alone did not fix the fit: {after}");

        // and the same for 3D ROMEO's output, which fails unaided
        use crate::unwrap::unwrap_romeo_multi_echo;
        let params = RomeoParams { individual: true, ..RomeoParams::default() };
        let mut u =
            unwrap_romeo_multi_echo(&wrapped, &[] as &[Vec<f64>], &TES, &mask, &params, &grid());
        enforce_inter_echo_consistency(&mut u, &refs, &mask);
        let err = max_abs_diff(&fit_slope(&u, &TES), &b);
        assert!(err < 1e-6, "the pass did not fix 3D ROMEO's output either: {err}");
    }

    #[test]
    fn slice_wise_unwrapping_is_what_makes_the_phase_spatially_continuous() {
        // The job the pass cannot do, and the reason this module exists for any consumer that
        // reads the phase rather than its TE-slope. The pass propagates echo 0's wrap state
        // faithfully into every later echo, so on its own it leaves the discontinuities there.
        let Inputs { wrapped, .. } = multi_echo_inputs(true);
        let mask = full_mask();
        let refs: Vec<&[f64]> = wrapped.iter().map(|p| p.as_slice()).collect();

        let mut raw = wrapped.clone();
        enforce_inter_echo_consistency(&mut raw, &refs, &mask);
        let slicewise = unwrap_slicewise_multi_echo(
            &wrapped, &[] as &[Vec<f64>], &TES, &mask, UnwrapMethod::Romeo,
            &SliceWiseParams::default(), &grid(),
        );

        // Echo 0 is the clean measurement, and it is clean *by construction* rather than by
        // comparison: `enforce_inter_echo_consistency` loops `for e in 1..n_echoes` and writes
        // only `unwrapped[e]`, so it can never touch the first echo. Any difference here is the
        // spatial unwrapping's and nothing else's - no null needed to rule the pass out, because
        // the structure already does.
        let pass_only_first = inplane_jumps(&raw[0]);
        let slicewise_first = inplane_jumps(&slicewise[0]);
        assert_eq!(
            pass_only_first,
            inplane_jumps(&wrapped[0]),
            "the pass changed echo 0, which it cannot do if it only writes echoes 1.. - the \
             structural argument this assertion rests on is broken"
        );
        assert_eq!(slicewise_first, 0, "slice-wise left {slicewise_first} jumps in echo 0");
        assert!(
            pass_only_first > 0,
            "the wrapped first echo has no in-plane jumps, so this phantom cannot tell the two \
             apart and the module docs' claim is untested here"
        );

        // The last echo is where both mechanisms are in play, so this one does need the null:
        // the pass propagates echo 0's wrap state forward, leaving more jumps than slice-wise.
        let last = TES.len() - 1;
        let (pass_only, with_slicewise) =
            (inplane_jumps(&raw[last]), inplane_jumps(&slicewise[last]));
        assert_eq!(with_slicewise, 0, "slice-wise left {with_slicewise} in-plane 2pi jumps");
        assert!(
            pass_only > 0,
            "the consistency pass alone left no in-plane jumps either, so this phantom cannot \
             tell the two apart and the claim in the module docs is untested here"
        );
    }

    #[test]
    fn no_3d_configuration_survives_both_offset_patterns_unaided() {
        // What slice-wise unwrapping is for. On clean data the 3D region growers are not
        // obviously wrong - the quality weighting defers the bad z edges, so each slice tends
        // to be entered once and picks up a whole number of wraps, which looks harmless.
        //
        // It is not harmless, because which whole number depends on the echo: the jump the
        // grower rounds is `offset_step + field_step * TE`, and TE is in it. The per-slice
        // wrap then differs between echoes, the slice offset stops being echo-independent,
        // and the fit across echoes that was supposed to cancel it is corrupted instead.
        // `correct_multi_echo_wraps` cannot repair that - it applies one correction to the
        // whole volume, and the error is per slice.
        //
        // Neither 3D path survives both offset patterns, and which one each fails on is not
        // something a caller can know in advance.
        use crate::unwrap::unwrap_romeo_multi_echo;
        let b = field();
        let g = grid();
        let mask = full_mask();

        for individual in [false, true] {
            let mut worst: f64 = 0.0;
            for interleaved in [true, false] {
                let offsets = slice_offsets(interleaved);
                let wrapped: Vec<Vec<f64>> = TES
                    .iter()
                    .map(|&te| wrap_all(&true_phase(&b, &offsets, te)))
                    .collect();
                let params = RomeoParams { individual, ..RomeoParams::default() };
                let u = unwrap_romeo_multi_echo(
                    &wrapped, &[] as &[Vec<f64>], &TES, &mask, &params, &g,
                );
                worst = worst.max(max_abs_diff(&fit_slope(&u, &TES), &b));
            }
            assert!(
                worst > 100.0,
                "3D ROMEO (individual={individual}) handled both offset patterns unaided (worst \
                 fit error {worst} rad/s) - then there was nothing to fix. Note this is 3D \
                 *without* the consistency pass; with it 3D is fine too, which is what \
                 the_consistency_pass_is_what_fixes_the_fit_even_without_slice_wise_unwrapping \
                 pins."
            );
        }

        // the slice-wise path gets both, which is the point
        for interleaved in [true, false] {
            let offsets = slice_offsets(interleaved);
            let wrapped: Vec<Vec<f64>> = TES
                .iter()
                .map(|&te| wrap_all(&true_phase(&b, &offsets, te)))
                .collect();
            let u = unwrap_slicewise_multi_echo(
                &wrapped, &[] as &[Vec<f64>], &TES, &mask, UnwrapMethod::Romeo,
                &SliceWiseParams::default(), &g,
            );
            let err = max_abs_diff(&fit_slope(&u, &TES), &b);
            assert!(err < 1e-6, "slice-wise missed interleaved={interleaved}: {err} rad/s");
        }
    }

    // ---- inter-echo consistency -------------------------------------------------

    /// Field, true phase per echo and wrapped phase per echo for one offset pattern.
    struct Inputs {
        field: Vec<f64>,
        truth: Vec<Vec<f64>>,
        wrapped: Vec<Vec<f64>>,
    }

    fn multi_echo_inputs(interleaved: bool) -> Inputs {
        let field = field();
        let offsets = slice_offsets(interleaved);
        let truth: Vec<Vec<f64>> = TES.iter().map(|&te| true_phase(&field, &offsets, te)).collect();
        let wrapped: Vec<Vec<f64>> = truth.iter().map(|t| wrap_all(t)).collect();
        Inputs { field, truth, wrapped }
    }

    #[test]
    fn consistency_makes_the_leftover_wrap_the_same_in_every_echo() {
        let Inputs { truth, wrapped, .. } = multi_echo_inputs(false);
        let mask = full_mask();

        for (method, name, tol) in [
            (UnwrapMethod::Romeo, "ROMEO", 1e-9),
            (UnwrapMethod::BestPath, "best path", 1e-9),
            (UnwrapMethod::Laplacian, "Laplacian", 0.1),
        ] {
            for enforce in [false, true] {
                let params = SliceWiseParams {
                    enforce_inter_echo_consistency: enforce,
                    ..SliceWiseParams::default()
                };
                let u = unwrap_slicewise_multi_echo(
                    &wrapped, &[] as &[Vec<f64>], &TES, &mask, method, &params, &grid(),
                );
                // leftover wraps per (slice, echo), measured against the model's own phase
                let wraps: Vec<Vec<f64>> = (0..NZ)
                    .map(|s| {
                        let base = s * NX * NY;
                        (0..TES.len())
                            .map(|e| ((u[e][base] - truth[e][base]) / TWO_PI).round())
                            .collect()
                    })
                    .collect();
                let consistent = wraps.iter().all(|w| w.iter().all(|&v| v == w[0]));
                if enforce {
                    assert!(consistent, "{name}: per-slice wraps still differ: {wraps:?}");
                    // and the result is still the truth plus that one shared multiple
                    for (e, ue) in u.iter().enumerate() {
                        for (s, w) in wraps.iter().enumerate() {
                            let base = s * NX * NY;
                            let shift = TWO_PI * w[0];
                            for p in 0..NX * NY {
                                let r = ue[base + p] - truth[e][base + p] - shift;
                                assert!(r.abs() < tol, "{name}: echo {e} slice {s} residual {r}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_linear_fit_across_echoes_then_recovers_the_field() {
        let Inputs { field: b, wrapped, .. } = multi_echo_inputs(true);
        let mask = full_mask();

        for (method, name, tol) in [
            (UnwrapMethod::Romeo, "ROMEO", 1e-6),
            (UnwrapMethod::BestPath, "best path", 1e-6),
            (UnwrapMethod::Laplacian, "Laplacian", 5.0),
        ] {
            let params = SliceWiseParams::default();
            let u = unwrap_slicewise_multi_echo(
                &wrapped, &[] as &[Vec<f64>], &TES, &mask, method, &params, &grid(),
            );
            let slope = fit_slope(&u, &TES);
            let err = max_abs_diff(&slope, &b);
            assert!(err < tol, "{name}: fitted field is off by {err} rad/s");
        }
    }

    #[test]
    fn without_consistency_the_fit_is_corrupted() {
        // The companion to the test above: the correction is load-bearing, not decorative.
        let Inputs { field: b, wrapped, .. } = multi_echo_inputs(true);
        let mask = full_mask();
        let params = SliceWiseParams {
            enforce_inter_echo_consistency: false,
            ..SliceWiseParams::default()
        };
        let u = unwrap_slicewise_multi_echo(
            &wrapped, &[] as &[Vec<f64>], &TES, &mask, UnwrapMethod::Romeo, &params, &grid(),
        );
        let err = max_abs_diff(&fit_slope(&u, &TES), &b);
        assert!(err > 50.0, "fit came out clean without the correction (error {err} rad/s)");
    }

    #[test]
    fn consistency_survives_a_minority_of_voxels_whose_echo_step_wraps() {
        // The wrap count per voxel is `m + relative_wrap`, and only `relative_wrap` is wanted.
        // `m` is zero wherever the echo-to-echo evolution stays inside +/-pi, which is the
        // usual case but not a guarantee: a strong local source can push a patch of a slice
        // past it. The median tolerates that as long as the patch is a minority; a mean would
        // be dragged off the integer and shift the slice by something that is not 2*pi.
        let mut b = field();
        let (cx, cy) = (NX as f64 / 2.0, NY as f64 / 2.0);
        let mut over = 0usize;
        for k in 0..NZ {
            for j in 0..NY {
                for i in 0..NX {
                    let r2 = (i as f64 - cx).powi(2) + (j as f64 - cy).powi(2);
                    let idx = i + j * NX + k * NX * NY;
                    b[idx] += 900.0 * (-r2 / 72.0).exp();
                    // the step between the closest echoes is what has to stay unwrappable
                    if (b[idx] * (TES[1] - TES[0])).abs() > PI {
                        over += 1;
                    }
                }
            }
        }
        let fraction = over as f64 / b.len() as f64;
        assert!(
            (0.05..0.45).contains(&fraction),
            "the bump has to make a minority of voxels wrap between echoes, not none and not              most: {fraction:.3}"
        );

        let offsets = slice_offsets(false);
        let truth: Vec<Vec<f64>> = TES.iter().map(|&te| true_phase(&b, &offsets, te)).collect();
        let wrapped: Vec<Vec<f64>> = truth.iter().map(|t| wrap_all(t)).collect();
        let u = unwrap_slicewise_multi_echo(
            &wrapped, &[] as &[Vec<f64>], &TES, &full_mask(), UnwrapMethod::Romeo,
            &SliceWiseParams::default(), &grid(),
        );
        for s in 0..NZ {
            let base = s * NX * NY;
            let k0 = ((u[0][base] - truth[0][base]) / TWO_PI).round();
            for (e, ue) in u.iter().enumerate() {
                let k = (ue[base] - truth[e][base]) / TWO_PI;
                assert!(
                    (k - k0).abs() < 1e-9,
                    "slice {s} echo {e} is {k} wraps from the truth, echo 0 is {k0} - a                      fractional or unequal offset means the correction was not an integer                      number of wraps"
                );
            }
        }
    }

    #[test]
    fn consistency_corrects_each_component_of_a_split_slice_separately() {
        // Two blobs in one slice can leave the unwrapper with different wrap counts, and a
        // single median over the whole slice would fix one and break the other.
        let Inputs { truth, wrapped, .. } = multi_echo_inputs(false);
        let mut mask = vec![0u8; NX * NY * NZ];
        for k in 0..NZ {
            for j in 0..NY {
                for i in 0..NX {
                    // left and right blobs, separated by a two-voxel gap
                    let inside = (2..10).contains(&i) || (14..22).contains(&i);
                    if inside && (2..22).contains(&j) {
                        mask[i + j * NX + k * NX * NY] = 1;
                    }
                }
            }
        }
        let u = unwrap_slicewise_multi_echo(
            &wrapped, &[] as &[Vec<f64>], &TES, &mask, UnwrapMethod::Romeo,
            &SliceWiseParams::default(), &grid(),
        );
        for s in 0..NZ {
            for (label, i0) in [("left", 2usize), ("right", 14usize)] {
                let base = i0 + 2 * NX + s * NX * NY;
                let k0 = ((u[0][base] - truth[0][base]) / TWO_PI).round();
                for e in 0..TES.len() {
                    let k = ((u[e][base] - truth[e][base]) / TWO_PI).round();
                    assert_eq!(k, k0, "slice {s} {label} blob: echo {e} has {k} wraps, echo 0 has {k0}");
                }
            }
        }
    }

    // ---- other axes and edge cases ----------------------------------------------

    #[test]
    fn unwrapping_along_another_axis_matches_the_transposed_problem() {
        // Build the same problem with the slices stacked along y, and check it comes out the
        // same as the z-stacked one read through the transpose.
        let b = field();
        let offsets = slice_offsets(false);
        let truth_z = true_phase(&b, &offsets, TES[3]);
        let g_z = grid();

        // transpose (x, y, z) -> (x, z, y), so the slice axis becomes 1
        let g_y = Grid::new(NX, NZ, NY, 1.0, 3.0, 1.0);
        let mut truth_y = vec![0.0; NX * NY * NZ];
        for k in 0..NZ {
            for j in 0..NY {
                for i in 0..NX {
                    truth_y[i + k * NX + j * NX * NZ] = truth_z[i + j * NX + k * NX * NY];
                }
            }
        }

        let u_z = unwrap_slicewise(
            &wrap_all(&truth_z), &[], &full_mask(), UnwrapMethod::Romeo,
            &SliceWiseParams::default(), &g_z,
        );
        let u_y = unwrap_slicewise(
            &wrap_all(&truth_y), &[], &full_mask(), UnwrapMethod::Romeo,
            &SliceWiseParams { slice_axis: 1, ..SliceWiseParams::default() }, &g_y,
        );
        for k in 0..NZ {
            for j in 0..NY {
                for i in 0..NX {
                    let a = u_z[i + j * NX + k * NX * NY];
                    let c = u_y[i + k * NX + j * NX * NZ];
                    assert!((a - c).abs() < 1e-9, "mismatch at ({i},{j},{k}): {a} vs {c}");
                }
            }
        }
    }

    #[test]
    fn single_echo_input_is_returned_unchanged_by_the_consistency_pass() {
        let Inputs { truth, .. } = multi_echo_inputs(false);
        let wrapped = vec![wrap_all(&truth[0])];
        let mask = full_mask();
        let u = unwrap_slicewise_multi_echo(
            &wrapped, &[] as &[Vec<f64>], &TES[..1], &mask, UnwrapMethod::Romeo,
            &SliceWiseParams::default(), &grid(),
        );
        assert_eq!(u.len(), 1);
        let direct = unwrap_slicewise(
            &wrapped[0], &[], &mask, UnwrapMethod::Romeo, &SliceWiseParams::default(), &grid(),
        );
        assert_eq!(u[0], direct);
    }

    #[test]
    fn magnitude_weighting_is_accepted_and_empty_magnitude_is_allowed() {
        let Inputs { truth, .. } = multi_echo_inputs(false);
        let wrapped = wrap_all(&truth[0]);
        let mask = full_mask();
        let mag = vec![1.0; NX * NY * NZ];
        let with_mag = unwrap_slicewise(
            &wrapped, &mag, &mask, UnwrapMethod::Romeo, &SliceWiseParams::default(), &grid(),
        );
        let without = unwrap_slicewise(
            &wrapped, &[], &mask, UnwrapMethod::Romeo, &SliceWiseParams::default(), &grid(),
        );
        assert!(max_inplane_gradient_error(&with_mag, &truth[0]) < 0.05);
        assert!(max_inplane_gradient_error(&without, &truth[0]) < 0.05);
    }

    #[test]
    fn an_empty_slice_is_skipped_rather_than_breaking_the_correction() {
        let Inputs { truth, wrapped, .. } = multi_echo_inputs(false);
        let mut mask = full_mask();
        for p in 0..NX * NY {
            mask[p + 3 * NX * NY] = 0; // slice 3 has no signal at all
        }
        let u = unwrap_slicewise_multi_echo(
            &wrapped, &[] as &[Vec<f64>], &TES, &mask, UnwrapMethod::Romeo,
            &SliceWiseParams::default(), &grid(),
        );
        // the remaining slices are still echo-consistent
        for s in [0usize, 1, 2, 4, 5, 6, 7] {
            let base = s * NX * NY;
            let k0 = ((u[0][base] - truth[0][base]) / TWO_PI).round();
            for e in 0..TES.len() {
                assert_eq!(((u[e][base] - truth[e][base]) / TWO_PI).round(), k0, "slice {s}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "phases must have at least one echo")]
    fn multi_echo_rejects_no_echoes() {
        unwrap_slicewise_multi_echo(
            &[] as &[Vec<f64>], &[] as &[Vec<f64>], &[], &full_mask(),
            UnwrapMethod::Romeo, &SliceWiseParams::default(), &grid(),
        );
    }

    #[test]
    #[should_panic(expected = "phase length must match grid dimensions")]
    fn rejects_a_phase_volume_that_does_not_match_the_grid() {
        unwrap_slicewise(
            &[0.0; 10], &[], &full_mask(), UnwrapMethod::Romeo,
            &SliceWiseParams::default(), &grid(),
        );
    }

}

//! Rigid-body (6-DOF) volume registration: *estimating* the transform between two volumes.
//!
//! [`crate::geometry`] can already **apply** an arbitrary transform — [`crate::geometry::GridMap`]
//! maps one voxel grid onto another given two affines, and [`crate::geometry::resample_onto`],
//! [`crate::geometry::resample_mask_onto`] and [`crate::geometry::resample_complex_onto`] move
//! continuous data, labels and wrapped phase across it. What was missing is *finding* the
//! transform in the first place, which is what this module does.
//!
//! # Why QSM needs it
//!
//! Multi-orientation QSM (COSMOS, [`crate::inversion::cosmos`]) and susceptibility tensor
//! imaging ([`crate::inversion::sti`]) both take N local field maps **on one common grid** plus
//! **one B0 direction per orientation in that grid's voxel frame**. Registration produces both,
//! and the second is not a by-product:
//!
//! > The object rotates and B0 does not. So the rigid rotation that aligns orientation *t* to
//! > the reference *is* where B0 pointed during orientation *t*, relative to the common grid.
//!
//! That matters most for the dataset shape where the affine is useless: identically prescribed
//! slabs, where every orientation has the *same* affine and the anatomy moved in voxel space
//! instead. [`crate::geometry::b0_direction_from_affine`] returns the same direction for all of
//! them, and N identical directions quietly collapse COSMOS to an unregularized
//! single-orientation inversion. Registration is the only thing that can recover the real answer
//! there — see [`RigidTransform::b0_direction_in_fixed`], which is where that convention lives
//! so no caller has to re-derive it.
//!
//! # Register on magnitude, resample phase through the complex domain
//!
//! Wrapped phase cannot be interpolated directly (halfway between `+3.0` and `-3.0` rad a linear
//! interpolator returns `0.0`; see the [`crate::geometry`] module docs), so it must not be used
//! as the registration input either. Register magnitude against magnitude, then move phase with
//! [`RigidTransform::resample_complex`], which goes through `mag·e^{iφ}`.
//!
//! # Metric: normalised cross-correlation, not mutual information
//!
//! Every use case here is mono-modal — GRE magnitude against GRE magnitude of the same subject,
//! same sequence. NCC is invariant to an affine intensity change (gain and offset), which is
//! what actually differs between two orientations of one acquisition, and it needs no joint
//! histogram, no binning choice and no parzen window. Mutual information buys nothing on
//! same-contrast data and is noisier at the small sample counts a coarse pyramid level uses.
//! Sum-of-squared-differences would be cheaper still but is not gain-invariant, so a receive-gain
//! difference between runs would bias it.
//!
//! The one thing NCC does *not* absorb is spatially varying shading: the receive field is fixed
//! to the coil, so when the head rotates the bias field rotates with the coil rather than with the
//! anatomy. That is a multiplicative *field*, not a scalar gain, and no global metric (MI
//! included) is invariant to it. Two mitigations, both available here: pass a brain `mask` so the
//! metric ignores everything outside the head, and/or hand in magnitude that has already been
//! bias-corrected ([`crate::homogeneity`]).
//!
//! # Optimiser, pyramid and cost
//!
//! Powell's direction-set method with a bracketed Brent line search, over a coarse-to-fine
//! Gaussian pyramid. Powell needs no derivatives of the metric — convenient, because the metric's
//! derivative runs through a trilinear interpolator — and it is what the established intensity
//! registration tools use for 6-DOF problems. No linear-algebra crate is involved; the whole
//! module is `f64` arrays, consistent with the rest of qsm-core.
//!
//! Work per metric evaluation is **bounded independently of volume size** by
//! [`RigidParams::max_samples`]: each level picks the smallest integer voxel stride whose sample
//! count fits the cap. Coarse levels therefore see smoothed, decimated data (a smooth cost
//! landscape, which is the point of the pyramid), while the finest level strides the *unsmoothed*
//! full-resolution volume (full spatial detail at the same cost, which is what gives the final
//! precision). Peak extra memory is the pyramid above the finest level — about 1/7 of one input
//! volume — plus the sample list, so a 32-bit WASM host with a 4 GB heap is not at risk.

use crate::geometry::{
    b0_direction_from_affine, resample_complex_onto, resample_mask_onto, resample_onto,
    trilinear_sample, voxel_sizes_from_affine, world_to_voxel,
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Fixed samples per parallel chunk. Fixed size, so the reduction combines in index order and
/// the metric is bit-identical regardless of thread count.
const SAMPLE_CHUNK: usize = 8192;

/// Registration parameters.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct RigidParams {
    /// Pyramid levels, finest included. `3` means the volume is registered at 1/4, then 1/2,
    /// then full resolution. Levels that would shrink any axis below 8 voxels are dropped.
    pub levels: usize,
    /// Upper bound on fixed voxels sampled per metric evaluation, per level.
    ///
    /// This, not the volume size, sets the cost of a registration: a 256³ and a 700³ volume take
    /// the same time. 6 parameters are vastly over-determined by 10⁵ samples, so lowering this
    /// buys speed long before it costs accuracy.
    pub max_samples: usize,
    /// Maximum Powell cycles (one pass over all six directions) per level.
    pub max_iterations: usize,
    /// Total metric evaluations across all levels, as a hard stop. Reached only on a cost
    /// surface that will not converge; the best parameters so far are returned either way.
    pub max_evaluations: usize,
    /// Relative cost improvement per Powell cycle below which a level is considered converged.
    pub tolerance: f64,
    /// Initial line-search step for the three rotations, at the finest level, in degrees.
    /// Coarser levels scale it up with the level's decimation factor.
    pub rotation_step_deg: f64,
    /// Initial line-search step for the three translations, at the finest level, in mm.
    /// Coarser levels scale it up with the level's decimation factor.
    pub translation_step_mm: f64,
    /// Smallest fraction of sampled fixed voxels that must still land inside the moving volume.
    ///
    /// NCC over a shrinking overlap is gameable — slide the volumes apart until only one
    /// well-correlated patch is left and the correlation goes up. Candidates below this floor are
    /// costed worse than any real match, with a gradient that pushes back into the valid region.
    /// Keep it low enough for a legitimately partial overlap (a thin slab against a whole head).
    pub min_overlap: f64,
    /// Starting parameters, in the convention of [`RigidTransform::params`]. `None` starts from
    /// the identity, i.e. it trusts the two affines to place the volumes in a common world —
    /// correct both for separately prescribed slabs (where the affines differ and are truthful)
    /// and for identically prescribed ones (where they agree and only the anatomy moved).
    ///
    /// Seed it when a previous answer is a good prior and the pyramid's coarse search is wasted
    /// work: registering a time series to one reference, for instance, where each volume starts
    /// from where the last one landed. The parameters are defined about the *fixed* volume's
    /// world centre, so they carry between pairs that share a fixed volume without conversion.
    pub initial: Option<[f64; 6]>,
}

impl Default for RigidParams {
    fn default() -> Self {
        Self {
            levels: 3,
            max_samples: 200_000,
            max_iterations: 20,
            max_evaluations: 5000,
            tolerance: 1e-5,
            rotation_step_deg: 3.0,
            translation_step_mm: 3.0,
            min_overlap: 0.25,
            initial: None,
        }
    }
}

/// A recovered rigid transform, together with everything needed to use it.
///
/// # Which way it points
///
/// [`Self::matrix`] maps **fixed world → moving world**: it answers "where in the moving
/// acquisition is the anatomy that the fixed volume has here", which is the direction resampling
/// needs (for each output voxel, find where to read). The inverse direction is one
/// [`Self::inverse`] away, and it is worth being pedantic about this: applying the rotation the
/// wrong way round produces a plausible-looking, wrong susceptibility map rather than an error.
#[derive(Clone, Debug)]
pub struct RigidTransform {
    /// Row-major 4×4 rigid transform, fixed world → moving world. Pure rotation plus
    /// translation; no scaling or shear.
    pub matrix: [f64; 16],
    /// The six parameters `matrix` was built from: `[rx, ry, rz, tx, ty, tz]`, rotations in
    /// **radians** composed as `Rz(rz)·Ry(ry)·Rx(rx)` about the fixed volume's world centre,
    /// translations in **mm** of world space.
    pub params: [f64; 6],
    /// Dimensions and voxel→world affine of the fixed volume — the common grid.
    pub fixed_dims: (usize, usize, usize),
    pub fixed_affine: [f64; 16],
    /// Dimensions and *original* voxel→world affine of the moving volume, as handed in.
    pub moving_dims: (usize, usize, usize),
    pub moving_affine: [f64; 16],
    /// Normalised cross-correlation at the solution, on `[-1, 1]`. Same-subject magnitude
    /// volumes that genuinely align land above ~0.9; below ~0.5 something is wrong.
    pub ncc: f64,
    /// Fraction of sampled fixed voxels that found a moving sample at the solution. Well under
    /// 1.0 means the volumes only partially overlap, which may be correct (a slab inside a
    /// head) or may mean the search wandered.
    pub overlap: f64,
    /// Metric evaluations used, summed over pyramid levels. Diagnostic.
    pub evaluations: usize,
}

impl RigidTransform {
    /// The identity transform between two grids — the "already registered" answer, for callers
    /// that skip estimation but still want one uniform type to resample and read directions
    /// through.
    pub fn identity(
        fixed_dims: (usize, usize, usize),
        fixed_affine: &[f64; 16],
        moving_dims: (usize, usize, usize),
        moving_affine: &[f64; 16],
    ) -> Self {
        Self {
            matrix: IDENTITY4,
            params: [0.0; 6],
            fixed_dims,
            fixed_affine: *fixed_affine,
            moving_dims,
            moving_affine: *moving_affine,
            ncc: f64::NAN,
            overlap: f64::NAN,
            evaluations: 0,
        }
    }

    /// The three rotation angles in degrees, in the order [`Self::params`] stores them.
    pub fn rotation_degrees(&self) -> [f64; 3] {
        [
            self.params[0].to_degrees(),
            self.params[1].to_degrees(),
            self.params[2].to_degrees(),
        ]
    }

    /// The translation in mm of world space.
    pub fn translation_mm(&self) -> [f64; 3] {
        [self.params[3], self.params[4], self.params[5]]
    }

    /// Total rotation angle in degrees, regardless of axis — the single number to eyeball when
    /// deciding whether a registration did anything.
    pub fn rotation_magnitude_deg(&self) -> f64 {
        let r = rotation_of(&self.matrix);
        let trace = r[0][0] + r[1][1] + r[2][2];
        (0.5 * (trace - 1.0)).clamp(-1.0, 1.0).acos().to_degrees()
    }

    /// The moving volume's voxel→world affine after alignment: re-tag the moving volume with
    /// this and its world coordinates agree with the fixed volume's anatomy, with no
    /// interpolation at all.
    ///
    /// **Do not then read B0 off it.** This affine describes the volume in the *reference
    /// anatomy's* world, where "world +z" is the reference orientation's B0 and not this
    /// orientation's, so [`b0_direction_from_affine`] applied to it is meaningless. Use
    /// [`Self::b0_direction_in_fixed`].
    pub fn aligned_affine(&self) -> [f64; 16] {
        mat4_mul(&rigid_inverse(&self.matrix), &self.moving_affine)
    }

    /// The moving acquisition's B0 direction expressed in the **fixed grid's** voxel frame —
    /// exactly the `bdirs` entry [`crate::inversion::cosmos`] and [`crate::inversion::sti`] want
    /// for this orientation.
    ///
    /// `declared_moving_b0` is a sidecar-declared B0 direction in the *moving* volume's own voxel
    /// frame, in the convention [`b0_direction_from_affine`] returns (the world B0 resolved onto
    /// the volume's normalised voxel axes). Pass `None` to read it off the moving affine, which is
    /// right whenever that affine is the scanner's own.
    ///
    /// The chain is: lift the direction into the moving volume's world, rotate it into the fixed
    /// volume's world with the inverse of [`Self::matrix`] (a free vector carries no translation),
    /// then resolve it onto the fixed grid's normalised voxel axes. With no rotation it reduces to
    /// `b0_direction_from_affine(fixed_affine)`, as it must.
    pub fn b0_direction_in_fixed(
        &self,
        declared_moving_b0: Option<(f64, f64, f64)>,
    ) -> (f64, f64, f64) {
        let d = declared_moving_b0.unwrap_or_else(|| b0_direction_from_affine(&self.moving_affine));
        let d = [d.0, d.1, d.2];
        // Moving voxel frame -> moving world.
        let in_moving_world = mat3_vec(&direction_cosines(&self.moving_affine), &d);
        // Moving world -> fixed world. R⁻¹ = Rᵀ for a rotation.
        let in_fixed_world = mat3_transpose_vec(&rotation_of(&self.matrix), &in_moving_world);
        // Fixed world -> fixed voxel frame.
        let v = mat3_transpose_vec(&direction_cosines(&self.fixed_affine), &in_fixed_world);
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if n < 1e-12 {
            (0.0, 0.0, 1.0)
        } else {
            (v[0] / n, v[1] / n, v[2] / n)
        }
    }

    /// Resample continuous moving data (magnitude, an unwrapped field map, χ) onto the fixed
    /// grid. **Not for wrapped phase** — use [`Self::resample_complex`].
    pub fn resample(&self, moving: &[f64]) -> Option<Vec<f64>> {
        resample_onto(
            moving,
            self.moving_dims,
            &self.aligned_affine(),
            self.fixed_dims,
            &self.fixed_affine,
        )
    }

    /// Resample a moving binary mask or label volume onto the fixed grid (nearest neighbour).
    pub fn resample_mask(&self, mask: &[u8]) -> Option<Vec<u8>> {
        resample_mask_onto(
            mask,
            self.moving_dims,
            &self.aligned_affine(),
            self.fixed_dims,
            &self.fixed_affine,
        )
    }

    /// Resample magnitude and wrapped phase onto the fixed grid together, interpolating
    /// `mag·e^{iφ}` so the wraps survive. Returns `(magnitude, phase)`.
    pub fn resample_complex(
        &self,
        magnitude: &[f64],
        phase: &[f64],
    ) -> Option<(Vec<f64>, Vec<f64>)> {
        resample_complex_onto(
            magnitude,
            phase,
            self.moving_dims,
            &self.aligned_affine(),
            self.fixed_dims,
            &self.fixed_affine,
        )
    }

    /// The same alignment with the roles swapped: the transform taking the old moving volume as
    /// fixed and vice versa. Lets a chain of pairwise registrations be re-referenced without
    /// re-estimating anything — what intra-run motion correction needs when the volume it
    /// registered against is not the one it wants to output on.
    pub fn inverse(&self) -> Self {
        let matrix = rigid_inverse(&self.matrix);
        let center = grid_center_world(self.moving_dims, &self.moving_affine);
        Self {
            params: params_from_matrix(&matrix, &center),
            matrix,
            fixed_dims: self.moving_dims,
            fixed_affine: self.moving_affine,
            moving_dims: self.fixed_dims,
            moving_affine: self.fixed_affine,
            ncc: self.ncc,
            overlap: self.overlap,
            evaluations: self.evaluations,
        }
    }
}

/// Estimate the rigid transform that aligns `moving` to `fixed`.
///
/// Both volumes are column-major with their own dimensions and row-major voxel→world affine;
/// they need not share a grid, voxel size or orientation. `mask` is optional and lives on the
/// **fixed** grid: it restricts which fixed voxels the metric looks at, and a brain mask is worth
/// passing — it keeps air, neck and the receive-field falloff out of the correlation.
///
/// Returns `None` if either affine is singular, if a volume's data length does not match its
/// dimensions, or if the metric could never be evaluated (empty mask, no overlap anywhere).
///
/// # Example
///
/// ```no_run
/// use qsm_core::registration::{register_rigid, RigidParams};
/// # fn f(fixed: &[f64], moving: &[f64], dims: (usize, usize, usize),
/// #      a_fixed: &[f64; 16], a_moving: &[f64; 16], brain: &[u8]) -> Option<()> {
/// let t = register_rigid(
///     fixed, dims, a_fixed,
///     moving, dims, a_moving,
///     Some(brain), &RigidParams::default(),
/// )?;
/// // The moving orientation's field map, on the fixed grid, with the B0 direction to match.
/// let field_on_common_grid = t.resample(moving)?;
/// let bdir = t.b0_direction_in_fixed(None);
/// # let _ = (field_on_common_grid, bdir); Some(()) }
/// ```
#[allow(clippy::too_many_arguments)] // Two volumes is six arguments before anything optional.
pub fn register_rigid(
    fixed: &[f64],
    fixed_dims: (usize, usize, usize),
    fixed_affine: &[f64; 16],
    moving: &[f64],
    moving_dims: (usize, usize, usize),
    moving_affine: &[f64; 16],
    mask: Option<&[u8]>,
    params: &RigidParams,
) -> Option<RigidTransform> {
    if fixed.len() != fixed_dims.0 * fixed_dims.1 * fixed_dims.2
        || moving.len() != moving_dims.0 * moving_dims.1 * moving_dims.2
    {
        return None;
    }
    if let Some(m) = mask {
        if m.len() != fixed.len() {
            return None;
        }
    }
    // Both affines must be invertible: the moving one to sample it, the fixed one because the
    // rotation centre and the B0 transport are defined through it.
    world_to_voxel(fixed_affine)?;
    world_to_voxel(moving_affine)?;

    // The rotation centre is the fixed volume's world centre, held fixed across pyramid levels
    // so one parameter vector means the same thing everywhere. Rotating about the centre rather
    // than the world origin is also what keeps the rotation and translation parameters from
    // being strongly coupled, which is the difference between Powell converging in a few cycles
    // and crawling along a diagonal valley.
    let center = grid_center_world(fixed_dims, fixed_affine);

    // Coarse-to-fine. Level 0 is the input itself (borrowed, not copied); each further level
    // halves every axis. Stop before any axis would drop below MIN_AXIS, where there is no
    // longer enough structure for a 6-parameter fit.
    const MIN_AXIS: usize = 8;
    let too_small = |d: (usize, usize, usize)| {
        d.0 < 2 * MIN_AXIS || d.1 < 2 * MIN_AXIS || d.2 < 2 * MIN_AXIS
    };
    let mut fixed_levels = vec![Volume::borrowed(fixed, fixed_dims, *fixed_affine)];
    let mut moving_levels = vec![Volume::borrowed(moving, moving_dims, *moving_affine)];
    // Level 0 reads the caller's mask directly; coarser levels own a decimated copy.
    let mut mask_levels: Vec<Option<Vec<u8>>> = vec![None];
    for level in 1..params.levels.max(1) {
        let prev = level - 1;
        if too_small(fixed_levels[prev].dims) || too_small(moving_levels[prev].dims) {
            break;
        }
        let prev_dims = fixed_levels[prev].dims;
        let next_mask = match (prev, mask) {
            (0, Some(m)) => Some(decimate_mask(m, prev_dims)),
            (0, None) => None,
            (_, _) => mask_levels[prev].as_ref().map(|m| decimate_mask(m, prev_dims)),
        };
        let next_fixed = fixed_levels[prev].halved();
        let next_moving = moving_levels[prev].halved();
        fixed_levels.push(next_fixed);
        moving_levels.push(next_moving);
        mask_levels.push(next_mask);
    }

    // The sample set per level does not depend on the parameters, so build it once. Bounded by
    // `max_samples` per level, which is what keeps the whole search independent of volume size.
    let samples: Vec<Vec<Sample>> = (0..fixed_levels.len())
        .map(|level| {
            let m = if level == 0 { mask } else { mask_levels[level].as_deref() };
            sample_set(&fixed_levels[level], m, params.max_samples)
        })
        .collect();

    let mut p = params.initial.unwrap_or([0.0; 6]);
    let mut evaluations = 0usize;
    let mut best: Option<(f64, f64)> = None;

    for level in (0..fixed_levels.len()).rev() {
        if samples[level].is_empty() {
            continue;
        }
        let factor = (1usize << level) as f64;
        let mv = &moving_levels[level];
        let mut cost = CostFn {
            samples: &samples[level],
            fixed_affine: fixed_levels[level].affine,
            moving: mv.data(),
            moving_dims: mv.dims,
            inv_moving: to_mat4(&world_to_voxel(&mv.affine)?),
            center,
            min_overlap: params.min_overlap,
            evaluations: 0,
            budget: params.max_evaluations.saturating_sub(evaluations),
        };
        // Coarse levels get proportionally bigger first steps: one voxel there is `factor`
        // voxels here, and the whole point of starting coarse is to move further per step.
        let rot = (params.rotation_step_deg * factor).to_radians();
        let trans = params.translation_step_mm * factor;
        let scale = [rot, rot, rot, trans, trans, trans];
        p = powell(&mut cost, p, scale, params.max_iterations, params.tolerance);
        evaluations += cost.evaluations;
        best = Some(cost.score(&p));
        if evaluations >= params.max_evaluations {
            break;
        }
    }

    let (ncc, overlap) = best?;
    Some(RigidTransform {
        matrix: matrix_from_params(&p, &center),
        params: p,
        fixed_dims,
        fixed_affine: *fixed_affine,
        moving_dims,
        moving_affine: *moving_affine,
        ncc,
        overlap,
        evaluations,
    })
}

// ------------------------------------------------------------------ pyramid

/// One pyramid level: either a borrowed view of the caller's data (level 0) or a decimated copy.
struct Volume<'a> {
    owned: Option<Vec<f64>>,
    view: &'a [f64],
    dims: (usize, usize, usize),
    affine: [f64; 16],
}

impl<'a> Volume<'a> {
    fn borrowed(data: &'a [f64], dims: (usize, usize, usize), affine: [f64; 16]) -> Self {
        Self { owned: None, view: data, dims, affine }
    }

    fn data(&self) -> &[f64] {
        self.owned.as_deref().unwrap_or(self.view)
    }

    /// Halve every axis: a 5-tap binomial blur per axis, then keep the even samples.
    ///
    /// The blur is the anti-aliasing the decimation needs; the binomial `[1 4 6 4 1]/16` kernel is
    /// the usual pyramid choice and needs no sigma to tune (the crate's box-filter Gaussian
    /// collapses to a no-op at the σ≈1 this calls for). Keeping the *even* samples means coarse
    /// voxel `i` sits exactly on fine voxel `2i`, so the level's affine is the fine affine with
    /// its three direction columns doubled and its translation untouched — exact, no half-voxel
    /// offset to get wrong.
    fn halved(&self) -> Volume<'static> {
        let (nx, ny, nz) = self.dims;
        let src = self.data();
        let blurred = binomial_blur3(src, self.dims);
        let (cx, cy, cz) = (nx.div_ceil(2), ny.div_ceil(2), nz.div_ceil(2));
        let mut out = Vec::with_capacity(cx * cy * cz);
        for k in 0..cz {
            for j in 0..cy {
                for i in 0..cx {
                    out.push(blurred[2 * i + 2 * j * nx + 2 * k * nx * ny]);
                }
            }
        }
        let mut affine = self.affine;
        for row in 0..3 {
            for col in 0..3 {
                affine[4 * row + col] *= 2.0;
            }
        }
        Volume { owned: Some(out), view: &[], dims: (cx, cy, cz), affine }
    }
}

/// Separable `[1 4 6 4 1]/16` blur with edge replication.
fn binomial_blur3(data: &[f64], dims: (usize, usize, usize)) -> Vec<f64> {
    const W: [f64; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
    let (nx, ny, nz) = dims;
    let mut a = data.to_vec();
    let mut b = vec![0.0f64; a.len()];
    // One pass per axis. `n` is its length and `stride` the step along it; `base` is the start
    // of one line, with `o` enumerating the other two axes.
    for axis in 0..3 {
        let (n, stride) = match axis {
            0 => (nx, 1usize),
            1 => (ny, nx),
            _ => (nz, nx * ny),
        };
        if n < 2 {
            continue;
        }
        let outer = a.len() / n;
        for o in 0..outer {
            let base = match axis {
                0 => o * nx,
                1 => (o % nx) + (o / nx) * nx * ny,
                _ => o,
            };
            for t in 0..n {
                let mut acc = 0.0;
                for (w, dt) in W.iter().zip(-2i64..=2) {
                    let u = (t as i64 + dt).clamp(0, n as i64 - 1) as usize;
                    acc += w * a[base + u * stride];
                }
                b[base + t * stride] = acc;
            }
        }
        std::mem::swap(&mut a, &mut b);
    }
    a
}

/// Decimate a mask the same way [`Volume::halved`] decimates data: keep the even samples, with
/// no blur, so the coarse mask stays binary and sits on the coarse sample positions exactly.
fn decimate_mask(mask: &[u8], dims: (usize, usize, usize)) -> Vec<u8> {
    let (nx, ny, nz) = dims;
    let (cx, cy, cz) = (nx.div_ceil(2), ny.div_ceil(2), nz.div_ceil(2));
    let mut out = Vec::with_capacity(cx * cy * cz);
    for k in 0..cz {
        for j in 0..cy {
            for i in 0..cx {
                out.push(mask[2 * i + 2 * j * nx + 2 * k * nx * ny]);
            }
        }
    }
    out
}

// ------------------------------------------------------------------ metric

/// One fixed-grid sample the metric is evaluated on: its voxel coordinate and its intensity.
///
/// `f32` throughout: the coordinates are small exact integers and the intensities only feed a
/// correlation, so this halves the memory traffic of the inner loop for no accuracy that matters.
/// The accumulation itself is `f64`.
#[derive(Clone, Copy)]
struct Sample {
    pos: [f32; 3],
    value: f32,
}

/// Pick the fixed voxels the metric looks at: the smallest integer stride whose masked sample
/// count fits `max_samples`.
///
/// Striding rather than blurring is deliberate at the finest level — it keeps full spatial detail
/// at a bounded cost. Coarse levels are already blurred and decimated, which is where the smooth
/// cost landscape comes from.
fn sample_set(fixed: &Volume, mask: Option<&[u8]>, max_samples: usize) -> Vec<Sample> {
    let (nx, ny, nz) = fixed.dims;
    let data = fixed.data();
    let eligible = |idx: usize| match mask {
        Some(m) => m[idx] != 0,
        None => true,
    };

    let count_at = |s: usize| {
        let mut n = 0usize;
        for k in (0..nz).step_by(s) {
            for j in (0..ny).step_by(s) {
                for i in (0..nx).step_by(s) {
                    if eligible(i + j * nx + k * nx * ny) {
                        n += 1;
                    }
                }
            }
        }
        n
    };

    let cap = max_samples.max(64);
    let mut stride = 1usize;
    while stride < nx.max(ny).max(nz) && count_at(stride) > cap {
        stride += 1;
    }

    let mut out = Vec::new();
    for k in (0..nz).step_by(stride) {
        for j in (0..ny).step_by(stride) {
            for i in (0..nx).step_by(stride) {
                let idx = i + j * nx + k * nx * ny;
                if eligible(idx) {
                    out.push(Sample {
                        pos: [i as f32, j as f32, k as f32],
                        value: data[idx] as f32,
                    });
                }
            }
        }
    }
    out
}

/// Running sums for one chunk of samples, so the reduction is order-independent.
#[derive(Clone, Copy, Default)]
struct Acc {
    n: usize,
    sf: f64,
    sm: f64,
    sff: f64,
    smm: f64,
    sfm: f64,
}

/// The cost being minimised: `1 - NCC` over the overlapping, masked samples.
struct CostFn<'a> {
    samples: &'a [Sample],
    fixed_affine: [f64; 16],
    moving: &'a [f64],
    moving_dims: (usize, usize, usize),
    /// World→voxel of the moving volume, as a 4×4 with a `[0,0,0,1]` last row.
    inv_moving: [f64; 16],
    center: [f64; 3],
    min_overlap: f64,
    evaluations: usize,
    budget: usize,
}

/// Costs assigned where NCC is not a meaningful number. Both are above the worst real cost
/// (`1 - (-1) = 2`), so a degenerate candidate can never win, and the overlap case keeps a
/// gradient that pushes the search back into the valid region.
const COST_DEGENERATE: f64 = 4.0;
const COST_NO_OVERLAP: f64 = 3.0;

impl CostFn<'_> {
    /// `(ncc, overlap)` at `p`. The sampling matrix folds the whole chain — fixed voxel →
    /// fixed world → moving world → moving voxel — into one 4×4, so the inner loop is three
    /// dot products and a trilinear fetch.
    fn score(&self, p: &[f64; 6]) -> (f64, f64) {
        let b = mat4_mul(
            &self.inv_moving,
            &mat4_mul(&matrix_from_params(p, &self.center), &self.fixed_affine),
        );
        let (nx, ny, nz) = self.moving_dims;
        let moving = self.moving;

        let partials: Vec<Acc> = maybe_par_chunks!(self.samples, SAMPLE_CHUNK)
            .map(|chunk| {
                let mut a = Acc::default();
                for s in chunk {
                    let (x, y, z) = (s.pos[0] as f64, s.pos[1] as f64, s.pos[2] as f64);
                    let ox = b[0] * x + b[1] * y + b[2] * z + b[3];
                    let oy = b[4] * x + b[5] * y + b[6] * z + b[7];
                    let oz = b[8] * x + b[9] * y + b[10] * z + b[11];
                    // Same acceptance window as GridMap::source_coord, so the metric and the
                    // final resampling agree on what "inside" means.
                    if ox < -0.5
                        || ox > nx as f64 - 0.5
                        || oy < -0.5
                        || oy > ny as f64 - 0.5
                        || oz < -0.5
                        || oz > nz as f64 - 0.5
                    {
                        continue;
                    }
                    let f = s.value as f64;
                    let m = trilinear_sample(moving, nx, ny, nz, ox, oy, oz);
                    a.n += 1;
                    a.sf += f;
                    a.sm += m;
                    a.sff += f * f;
                    a.smm += m * m;
                    a.sfm += f * m;
                }
                a
            })
            .collect();

        let mut t = Acc::default();
        for a in &partials {
            t.n += a.n;
            t.sf += a.sf;
            t.sm += a.sm;
            t.sff += a.sff;
            t.smm += a.smm;
            t.sfm += a.sfm;
        }

        let overlap = t.n as f64 / self.samples.len().max(1) as f64;
        if t.n < 2 {
            return (f64::NAN, overlap);
        }
        let n = t.n as f64;
        // Means over the *overlap*, not over the whole volume: a candidate that shifts which
        // voxels overlap changes both means, and using global means instead would leave a
        // spurious bias term in the correlation.
        let vf = t.sff - t.sf * t.sf / n;
        let vm = t.smm - t.sm * t.sm / n;
        if vf <= 0.0 || vm <= 0.0 {
            return (f64::NAN, overlap);
        }
        let cov = t.sfm - t.sf * t.sm / n;
        (cov / (vf * vm).sqrt(), overlap)
    }

    fn cost(&mut self, p: &[f64; 6]) -> f64 {
        if self.evaluations >= self.budget {
            // Out of budget: report a flat surface so the line searches stop moving and Powell
            // falls out on its own tolerance test, leaving the best-so-far parameters in place.
            return COST_DEGENERATE;
        }
        self.evaluations += 1;
        let (ncc, overlap) = self.score(p);
        if overlap < self.min_overlap {
            return COST_NO_OVERLAP + (self.min_overlap - overlap);
        }
        if !ncc.is_finite() {
            return COST_DEGENERATE;
        }
        1.0 - ncc
    }
}

// ------------------------------------------------------------------ optimiser

/// Powell's direction-set method.
///
/// Minimises along each of six directions in turn, then — this is the part that distinguishes
/// Powell from plain coordinate descent — folds the net displacement of the cycle in as a new
/// direction, so a valley running diagonally through parameter space gets followed instead of
/// zigzagged along. The initial directions are the coordinate axes scaled by `scale`, which is
/// how rotations (radians) and translations (mm) are put on comparable footing.
fn powell(
    cost: &mut CostFn,
    start: [f64; 6],
    scale: [f64; 6],
    max_cycles: usize,
    tol: f64,
) -> [f64; 6] {
    const N: usize = 6;
    let mut dirs = [[0.0f64; N]; N];
    for (i, d) in dirs.iter_mut().enumerate() {
        d[i] = scale[i];
    }
    let mut p = start;
    let mut fp = cost.cost(&p);

    for _ in 0..max_cycles {
        let p_start = p;
        let f_start = fp;
        let mut biggest_drop = 0.0f64;
        let mut biggest = 0usize;

        for (i, dir) in dirs.iter().enumerate() {
            let before = fp;
            let (np, nf) = line_minimise(cost, &p, dir);
            p = np;
            fp = nf;
            if before - fp > biggest_drop {
                biggest_drop = before - fp;
                biggest = i;
            }
        }

        if 2.0 * (f_start - fp) <= tol * (f_start.abs() + fp.abs()) + 1e-12 {
            break;
        }

        // Try the net direction of this cycle, extrapolated one cycle further.
        let mut net = [0.0f64; N];
        let mut extrap = p;
        for i in 0..N {
            net[i] = p[i] - p_start[i];
            extrap[i] = p[i] + net[i];
        }
        let f_extrap = cost.cost(&extrap);
        if f_extrap < f_start {
            // Numerical Recipes' test: only adopt the net direction when the decrease along it
            // is not already accounted for by the single best coordinate direction, which keeps
            // the direction set from collapsing towards linear dependence.
            let t = 2.0 * (f_start - 2.0 * fp + f_extrap) * (f_start - fp - biggest_drop).powi(2)
                - biggest_drop * (f_start - f_extrap).powi(2);
            if t < 0.0 {
                let (np, nf) = line_minimise(cost, &p, &net);
                p = np;
                fp = nf;
                dirs[biggest] = dirs[N - 1];
                dirs[N - 1] = net;
            }
        }
    }
    p
}

/// Minimise `cost` along `p + x·dir`: bracket the minimum, then Brent.
fn line_minimise(cost: &mut CostFn, p: &[f64; 6], dir: &[f64; 6]) -> ([f64; 6], f64) {
    let at = |x: f64| {
        let mut q = *p;
        for i in 0..6 {
            q[i] += x * dir[i];
        }
        q
    };
    let f = |x: f64, c: &mut CostFn| c.cost(&at(x));

    // Bracket. `dir` carries the step scale, so x = 1 is one nominal step.
    const GOLD: f64 = 1.618_033_988_749_895;
    const MAX_EXPANSIONS: usize = 24;
    let (mut ax, mut bx) = (0.0f64, 1.0f64);
    let f_at_zero = f(ax, cost);
    let f_at_one = f(bx, cost);
    // Point the bracket downhill: `b` must be the better of the two endpoints.
    let mut fb = if f_at_one > f_at_zero {
        std::mem::swap(&mut ax, &mut bx);
        f_at_zero
    } else {
        f_at_one
    };
    let mut cx = bx + GOLD * (bx - ax);
    let mut fc = f(cx, cost);
    let mut expansions = 0;
    while fc < fb && expansions < MAX_EXPANSIONS {
        ax = bx;
        bx = cx;
        fb = fc;
        cx = bx + GOLD * (bx - ax);
        fc = f(cx, cost);
        expansions += 1;
    }

    // Brent on the bracket [min(ax,cx), max(ax,cx)] with bx inside it.
    const ITERS: usize = 40;
    // `x` is measured in nominal steps (the direction vector carries the physical scale), so a
    // 1e-3-step floor is 0.003 degrees / 0.003 mm at the finest level — far below the precision
    // the interpolated metric can actually resolve, and it stops Brent grinding near x = 0.
    const ZEPS: f64 = 1e-3;
    const TOL: f64 = 1e-3;
    let (mut a, mut b) = (ax.min(cx), ax.max(cx));
    let (mut x, mut w, mut v) = (bx, bx, bx);
    let (mut fx, mut fw, mut fv) = (fb, fb, fb);
    let mut d = 0.0f64;
    let mut e = 0.0f64;
    for _ in 0..ITERS {
        let xm = 0.5 * (a + b);
        let tol1 = TOL * x.abs() + ZEPS;
        if (x - xm).abs() <= 2.0 * tol1 - 0.5 * (b - a) {
            break;
        }
        let mut use_golden = true;
        if e.abs() > tol1 {
            // Parabola through (x, fx), (w, fw), (v, fv).
            let r = (x - w) * (fx - fv);
            let q = (x - v) * (fx - fw);
            let mut pnum = (x - v) * q - (x - w) * r;
            let mut qden = 2.0 * (q - r);
            if qden > 0.0 {
                pnum = -pnum;
            } else {
                qden = -qden;
            }
            let etemp = e;
            e = d;
            if qden.abs() > 1e-300
                && pnum.abs() < (0.5 * qden * etemp).abs()
                && pnum > qden * (a - x)
                && pnum < qden * (b - x)
            {
                d = pnum / qden;
                let u = x + d;
                if (u - a) < 2.0 * tol1 || (b - u) < 2.0 * tol1 {
                    d = if xm - x >= 0.0 { tol1 } else { -tol1 };
                }
                use_golden = false;
            }
        }
        if use_golden {
            e = if x >= xm { a - x } else { b - x };
            d = 0.381_966_011_250_105 * e;
        }
        let u = if d.abs() >= tol1 {
            x + d
        } else {
            x + if d >= 0.0 { tol1 } else { -tol1 }
        };
        let fu = f(u, cost);
        if fu <= fx {
            if u >= x {
                a = x;
            } else {
                b = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
    }
    (at(x), fx)
}

// ------------------------------------------------------------------ small matrix helpers
//
// Row-major 4×4 with an implicit `[0,0,0,1]` last row, matching the affine layout the rest of
// the crate uses. Hand-written because the crate carries no linear-algebra dependency (see
// `utils::denoise`'s Jacobi eigensolver for the same decision).

const IDENTITY4: [f64; 16] = [
    1.0, 0.0, 0.0, 0.0, //
    0.0, 1.0, 0.0, 0.0, //
    0.0, 0.0, 1.0, 0.0, //
    0.0, 0.0, 0.0, 1.0,
];

fn mat4_mul(a: &[f64; 16], b: &[f64; 16]) -> [f64; 16] {
    let mut out = [0.0f64; 16];
    for i in 0..4 {
        for j in 0..4 {
            out[4 * i + j] = (0..4).map(|k| a[4 * i + k] * b[4 * k + j]).sum();
        }
    }
    out
}

fn to_mat4(m: &[[f64; 4]; 3]) -> [f64; 16] {
    let mut out = IDENTITY4;
    for i in 0..3 {
        out[4 * i..4 * i + 4].copy_from_slice(&m[i]);
    }
    out
}

/// The 3×3 block of a 4×4.
fn rotation_of(m: &[f64; 16]) -> [[f64; 3]; 3] {
    [
        [m[0], m[1], m[2]],
        [m[4], m[5], m[6]],
        [m[8], m[9], m[10]],
    ]
}

/// Inverse of a rigid 4×4: transpose the rotation, and carry the translation back through it.
fn rigid_inverse(m: &[f64; 16]) -> [f64; 16] {
    let r = rotation_of(m);
    let t = [m[3], m[7], m[11]];
    let mut out = IDENTITY4;
    for i in 0..3 {
        for j in 0..3 {
            out[4 * i + j] = r[j][i];
        }
        out[4 * i + 3] = -(r[0][i] * t[0] + r[1][i] * t[1] + r[2][i] * t[2]);
    }
    out
}

fn mat3_vec(m: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn mat3_transpose_vec(m: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}

/// The direction-cosine matrix of an affine: its 3×3 with the voxel sizes divided out, so what
/// is left is the pure rotation taking voxel axes to world axes.
///
/// Factoring the voxel sizes out first is what makes everything downstream correct for
/// anisotropic voxels — the same trap [`b0_direction_from_affine`] documents.
fn direction_cosines(affine: &[f64; 16]) -> [[f64; 3]; 3] {
    let (sx, sy, sz) = voxel_sizes_from_affine(affine);
    let s = [sx.max(1e-12), sy.max(1e-12), sz.max(1e-12)];
    let mut out = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = affine[4 * i + j] / s[j];
        }
    }
    out
}

/// World coordinate of a grid's centre voxel.
fn grid_center_world(dims: (usize, usize, usize), affine: &[f64; 16]) -> [f64; 3] {
    let c = [
        (dims.0 as f64 - 1.0) * 0.5,
        (dims.1 as f64 - 1.0) * 0.5,
        (dims.2 as f64 - 1.0) * 0.5,
    ];
    [
        affine[0] * c[0] + affine[1] * c[1] + affine[2] * c[2] + affine[3],
        affine[4] * c[0] + affine[5] * c[1] + affine[6] * c[2] + affine[7],
        affine[8] * c[0] + affine[9] * c[1] + affine[10] * c[2] + affine[11],
    ]
}

/// `Rz(rz)·Ry(ry)·Rx(rx)`, angles in radians.
fn euler_rotation(rx: f64, ry: f64, rz: f64) -> [[f64; 3]; 3] {
    let (sa, ca) = rx.sin_cos();
    let (sb, cb) = ry.sin_cos();
    let (sc, cc) = rz.sin_cos();
    [
        [cc * cb, cc * sb * sa - sc * ca, cc * sb * ca + sc * sa],
        [sc * cb, sc * sb * sa + cc * ca, sc * sb * ca - cc * sa],
        [-sb, cb * sa, cb * ca],
    ]
}

/// The rigid 4×4 for a parameter vector: `x_moving = center + R·(x_fixed - center) + t`.
fn matrix_from_params(p: &[f64; 6], center: &[f64; 3]) -> [f64; 16] {
    let r = euler_rotation(p[0], p[1], p[2]);
    let rc = mat3_vec(&r, center);
    let mut out = IDENTITY4;
    for i in 0..3 {
        out[4 * i..4 * i + 3].copy_from_slice(&r[i]);
        out[4 * i + 3] = center[i] - rc[i] + p[3 + i];
    }
    out
}

/// Recover a parameter vector from a rigid 4×4, about `center`. The inverse of
/// [`matrix_from_params`], used by [`RigidTransform::inverse`].
fn params_from_matrix(m: &[f64; 16], center: &[f64; 3]) -> [f64; 6] {
    let r = rotation_of(m);
    // Rz·Ry·Rx has -sin(ry) in position [2][0]; the other two angles come off the entries that
    // carry a single sine each.
    let ry = (-r[2][0]).clamp(-1.0, 1.0).asin();
    let (rx, rz) = if r[2][0].abs() < 1.0 - 1e-9 {
        (r[2][1].atan2(r[2][2]), r[1][0].atan2(r[0][0]))
    } else {
        // Gimbal lock: rx and rz are not separable, so put the whole in-plane rotation in rz.
        (0.0, (-r[0][1]).atan2(r[1][1]))
    };
    let rr = euler_rotation(rx, ry, rz);
    let rc = mat3_vec(&rr, center);
    [
        rx,
        ry,
        rz,
        m[3] - (center[0] - rc[0]),
        m[7] - (center[1] - rc[1]),
        m[11] - (center[2] - rc[2]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::resample_onto;

    fn identity_affine(vs: (f64, f64, f64)) -> [f64; 16] {
        [
            vs.0, 0.0, 0.0, 0.0, //
            0.0, vs.1, 0.0, 0.0, //
            0.0, 0.0, vs.2, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ]
    }

    /// A rigid 4×4 built from first principles: translate the rotation centre to the origin,
    /// apply three *literal* axis-rotation matrices, translate back, then offset.
    ///
    /// Deliberately not [`matrix_from_params`]. If the tests' ground truth came out of the same
    /// function the implementation parameterises with, a transposed rotation in it would cancel
    /// on both sides and every recovery assertion below would become unfailable — which is
    /// precisely what happened to the first draft of this file. The only shared machinery is
    /// [`mat4_mul`], pinned against hand-computed numbers in `mat4_mul_multiplies_row_major`.
    fn truth_transform(center: &[f64; 3], deg: [f64; 3], mm: [f64; 3]) -> [f64; 16] {
        let shift = |t: [f64; 3]| {
            [
                1.0, 0.0, 0.0, t[0], //
                0.0, 1.0, 0.0, t[1], //
                0.0, 0.0, 1.0, t[2], //
                0.0, 0.0, 0.0, 1.0,
            ]
        };
        let (sa, ca) = deg[0].to_radians().sin_cos();
        let (sb, cb) = deg[1].to_radians().sin_cos();
        let (sc, cc) = deg[2].to_radians().sin_cos();
        let rx = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, ca, -sa, 0.0, //
            0.0, sa, ca, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let ry = [
            cb, 0.0, sb, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            -sb, 0.0, cb, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let rz = [
            cc, -sc, 0.0, 0.0, //
            sc, cc, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let c = *center;
        let to_origin = shift([-c[0], -c[1], -c[2]]);
        let back = shift([c[0] + mm[0], c[1] + mm[1], c[2] + mm[2]]);
        mat4_mul(&back, &mat4_mul(&rz, &mat4_mul(&ry, &mat4_mul(&rx, &to_origin))))
    }

    /// A deterministic lumpy phantom: three ellipsoidal blobs of different brightness plus a
    /// shell, which gives the metric real structure in all three axes. A smooth blob alone is
    /// rotationally ambiguous and would not pin a rotation down.
    fn phantom(dims: (usize, usize, usize)) -> Vec<f64> {
        let (nx, ny, nz) = dims;
        let mut v = vec![0.0f64; nx * ny * nz];
        let blobs = [
            // (cx, cy, cz, rx, ry, rz, value) in fractions of the volume
            (0.50, 0.50, 0.50, 0.34, 0.26, 0.30, 1.0),
            (0.36, 0.44, 0.52, 0.10, 0.09, 0.11, 2.2),
            (0.63, 0.57, 0.44, 0.08, 0.13, 0.07, 1.7),
            (0.50, 0.62, 0.64, 0.07, 0.06, 0.09, 2.8),
        ];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let (x, y, z) = (
                        i as f64 / nx as f64,
                        j as f64 / ny as f64,
                        k as f64 / nz as f64,
                    );
                    let mut acc = 0.0;
                    for &(cx, cy, cz, rx, ry, rz, val) in &blobs {
                        let d = ((x - cx) / rx).powi(2) + ((y - cy) / ry).powi(2) + ((z - cz) / rz).powi(2);
                        if d <= 1.0 {
                            // Smooth falloff, so trilinear interpolation has a gradient to work
                            // with rather than a staircase.
                            acc += val * (1.0 - d).sqrt();
                        }
                    }
                    v[i + j * nx + k * nx * ny] = acc;
                }
            }
        }
        v
    }

    /// Build the moving volume that a known world transform `w` (fixed world → moving world)
    /// would have produced from `fixed`.
    ///
    /// Sampling `fixed` through `w·fixed_affine` is exactly the forward model
    /// [`RigidTransform::resample`] inverts, so a registration that recovers `w` is the only way
    /// this round-trips — which is what makes the test able to catch an inverted or transposed
    /// transform rather than just a blurry one.
    fn warp(
        fixed: &[f64],
        fixed_dims: (usize, usize, usize),
        fixed_affine: &[f64; 16],
        moving_dims: (usize, usize, usize),
        moving_affine: &[f64; 16],
        w: &[f64; 16],
    ) -> Vec<f64> {
        resample_onto(
            fixed,
            fixed_dims,
            &mat4_mul(w, fixed_affine),
            moving_dims,
            moving_affine,
        )
        .unwrap()
    }

    /// The angle between two rotations: the rotation angle of `Aᵀ·B`, whose trace is
    /// `Σ_ij A[j][i]·B[j][i]`.
    fn rotation_error_deg(a: &[f64; 16], b: &[f64; 16]) -> f64 {
        let (ra, rb) = (rotation_of(a), rotation_of(b));
        let trace: f64 = (0..3)
            .map(|i| (0..3).map(|j| ra[j][i] * rb[j][i]).sum::<f64>())
            .sum();
        (0.5 * (trace - 1.0)).clamp(-1.0, 1.0).acos().to_degrees()
    }

    fn translation_error_mm(a: &[f64; 16], b: &[f64; 16]) -> f64 {
        [(a[3] - b[3]), (a[7] - b[7]), (a[11] - b[11])]
            .iter()
            .fold(0.0f64, |m, v| m.max(v.abs()))
    }

    // ---------------------------------------------------------- algebra

    /// Row-major multiplication, against a product worked out by hand.
    #[test]
    fn mat4_mul_multiplies_row_major() {
        let a = [
            1.0, 2.0, 0.0, 3.0, //
            0.0, 1.0, 4.0, 0.0, //
            5.0, 0.0, 1.0, 1.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let b = [
            0.0, 1.0, 2.0, 1.0, //
            3.0, 0.0, 1.0, 0.0, //
            1.0, 1.0, 0.0, 2.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        // Row 0 of A times each column of B: [0+6+0, 1+0+0, 2+2+0, 1+0+0+3]
        let want = [
            6.0, 1.0, 4.0, 4.0, //
            7.0, 4.0, 1.0, 8.0, //
            1.0, 6.0, 10.0, 8.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let got = mat4_mul(&a, &b);
        for i in 0..16 {
            assert!((got[i] - want[i]).abs() < 1e-12, "element {i}: {} vs {}", got[i], want[i]);
        }
    }

    /// The implementation's parameterisation and the tests' independent builder must describe the
    /// same transform. This is the test that catches a transposed or mis-ordered rotation; the
    /// recovery tests below then inherit a ground truth that is not self-fulfilling.
    #[test]
    fn the_parameterisation_matches_an_independent_construction() {
        let center = [12.5, -3.0, 7.25];
        for (deg, mm) in [
            ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ([11.0, -7.0, 23.0], [2.0, -4.0, 6.0]),
            ([-30.0, 15.0, -5.0], [-1.0, 0.5, 3.0]),
        ] {
            let want = truth_transform(&center, deg, mm);
            let p = [
                deg[0].to_radians(),
                deg[1].to_radians(),
                deg[2].to_radians(),
                mm[0],
                mm[1],
                mm[2],
            ];
            let got = matrix_from_params(&p, &center);
            for i in 0..16 {
                assert!(
                    (got[i] - want[i]).abs() < 1e-9,
                    "deg {deg:?} mm {mm:?} element {i}: {} vs {}",
                    got[i], want[i]
                );
            }
        }
    }

    #[test]
    fn euler_round_trips_through_the_matrix() {
        let center = [3.0, -7.0, 11.0];
        for p in [
            [0.0; 6],
            [0.1, -0.2, 0.3, 4.0, -5.0, 6.0],
            [-0.4, 0.05, -0.6, -1.0, 2.0, -3.0],
        ] {
            let m = matrix_from_params(&p, &center);
            let back = params_from_matrix(&m, &center);
            for i in 0..6 {
                assert!((p[i] - back[i]).abs() < 1e-9, "param {i}: {p:?} vs {back:?}");
            }
        }
    }

    #[test]
    fn rigid_inverse_is_an_inverse() {
        let m = matrix_from_params(&[0.3, -0.2, 0.5, 7.0, -2.0, 4.0], &[1.0, 2.0, 3.0]);
        let p = mat4_mul(&m, &rigid_inverse(&m));
        for (i, (&got, &want)) in p.iter().zip(IDENTITY4.iter()).enumerate() {
            assert!((got - want).abs() < 1e-12, "element {i}: {got} vs {want}");
        }
    }

    /// Rotating about the volume centre must leave the centre where it is — the property that
    /// decouples the rotation and translation parameters.
    #[test]
    fn rotation_is_about_the_fixed_volume_centre() {
        let affine = identity_affine((1.0, 1.0, 2.0));
        let dims = (32, 40, 16);
        let c = grid_center_world(dims, &affine);
        let m = matrix_from_params(&[0.4, -0.3, 0.2, 0.0, 0.0, 0.0], &c);
        let moved = [
            m[0] * c[0] + m[1] * c[1] + m[2] * c[2] + m[3],
            m[4] * c[0] + m[5] * c[1] + m[6] * c[2] + m[7],
            m[8] * c[0] + m[9] * c[1] + m[10] * c[2] + m[11],
        ];
        for i in 0..3 {
            assert!((moved[i] - c[i]).abs() < 1e-9, "axis {i}: {moved:?} vs {c:?}");
        }
    }

    /// The pyramid's affine bookkeeping: coarse voxel `i` must land on the same world point as
    /// fine voxel `2i`, or every level would be searching a shifted problem.
    #[test]
    fn halved_level_keeps_world_coordinates() {
        let dims = (24, 24, 24);
        let affine = [
            0.8, 0.0, 0.0, -11.0, //
            0.0, 0.9, 0.0, 5.0, //
            0.0, 0.0, 3.0, 2.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let data = phantom(dims);
        let coarse = Volume::borrowed(&data, dims, affine).halved();
        assert_eq!(coarse.dims, (12, 12, 12));
        for (i, j, k) in [(0usize, 0usize, 0usize), (3, 5, 7), (11, 11, 11)] {
            let fine = [
                affine[0] * (2 * i) as f64 + affine[3],
                affine[5] * (2 * j) as f64 + affine[7],
                affine[10] * (2 * k) as f64 + affine[11],
            ];
            let a = coarse.affine;
            let got = [
                a[0] * i as f64 + a[3],
                a[5] * j as f64 + a[7],
                a[10] * k as f64 + a[11],
            ];
            for d in 0..3 {
                assert!((fine[d] - got[d]).abs() < 1e-12, "{fine:?} vs {got:?}");
            }
        }
    }

    // ---------------------------------------------------------- recovery

    /// The test that catches an inverted or transposed transform: warp a phantom by a known
    /// rigid transform, register, and require the recovered transform back.
    #[test]
    fn recovers_a_known_rotation_and_translation() {
        let dims = (48, 48, 48);
        let affine = identity_affine((1.0, 1.0, 1.0));
        let fixed = phantom(dims);
        let truth = truth_transform(&grid_center_world(dims, &affine), [9.0, -6.0, 12.0], [2.5, -1.5, 3.0]);
        let moving = warp(&fixed, dims, &affine, dims, &affine, &truth);

        let t = register_rigid(&fixed, dims, &affine, &moving, dims, &affine, None, &RigidParams::default())
            .expect("registration should run");

        let rot_err = rotation_error_deg(&t.matrix, &truth);
        let trans_err = translation_error_mm(&t.matrix, &truth);
        println!(
            "recovered rot err {rot_err:.3}deg, trans err {trans_err:.3}mm, ncc {:.4}, {} evals",
            t.ncc, t.evaluations
        );
        assert!(rot_err < 1.0, "rotation error {rot_err} deg");
        assert!(trans_err < 1.0, "translation error {trans_err} mm");
        assert!(t.ncc > 0.99, "ncc {}", t.ncc);
    }

    /// The assertion above has to be able to fail. Feeding the *unwarped* volume in as `moving`
    /// means the true answer is the identity, so measuring it against the same `truth` must blow
    /// through both thresholds — if it did not, the thresholds would be vacuous.
    #[test]
    fn the_recovery_thresholds_can_fail() {
        let dims = (48, 48, 48);
        let affine = identity_affine((1.0, 1.0, 1.0));
        let fixed = phantom(dims);
        let truth = truth_transform(&grid_center_world(dims, &affine), [9.0, -6.0, 12.0], [2.5, -1.5, 3.0]);
        let t = register_rigid(&fixed, dims, &affine, &fixed, dims, &affine, None, &RigidParams::default())
            .expect("registration should run");
        assert!(
            rotation_error_deg(&t.matrix, &truth) > 1.0,
            "a 16-degree rotation must not pass the 1-degree threshold"
        );
        assert!(
            translation_error_mm(&t.matrix, &truth) > 1.0,
            "a 3mm offset must not pass the 1mm threshold"
        );
    }

    /// Registering a volume against itself must return (close to) the identity, and must not
    /// drift away from it. The cheapest sanity check there is, and it fails loudly if the cost
    /// surface or the parameterisation has a sign error.
    #[test]
    fn self_registration_is_the_identity() {
        let dims = (40, 40, 40);
        let affine = identity_affine((1.0, 1.0, 1.0));
        let fixed = phantom(dims);
        let t = register_rigid(&fixed, dims, &affine, &fixed, dims, &affine, None, &RigidParams::default())
            .unwrap();
        assert!(t.rotation_magnitude_deg() < 0.3, "rotation {}", t.rotation_magnitude_deg());
        assert!(
            translation_error_mm(&t.matrix, &IDENTITY4) < 0.3,
            "translation {:?}",
            t.translation_mm()
        );
        assert!(t.ncc > 0.9999, "ncc {}", t.ncc);
    }

    /// Anisotropic voxels are where resampling code historically goes wrong (see the
    /// `b0_direction_from_affine` note about QSMxT 8.2.2): a rotation in *world* mm is not a
    /// rotation in voxel indices when the voxels are not cubic.
    #[test]
    fn recovers_a_rotation_on_anisotropic_voxels() {
        let dims = (48, 48, 24);
        let affine = identity_affine((0.8, 0.8, 2.0));
        let fixed = phantom(dims);
        let truth = truth_transform(&grid_center_world(dims, &affine), [5.0, -4.0, 8.0], [1.6, -2.4, 1.0]);
        let moving = warp(&fixed, dims, &affine, dims, &affine, &truth);
        let t = register_rigid(&fixed, dims, &affine, &moving, dims, &affine, None, &RigidParams::default())
            .unwrap();
        let rot_err = rotation_error_deg(&t.matrix, &truth);
        let trans_err = translation_error_mm(&t.matrix, &truth);
        println!("anisotropic: rot err {rot_err:.3}deg, trans err {trans_err:.3}mm");
        assert!(rot_err < 1.5, "rotation error {rot_err} deg");
        assert!(trans_err < 1.5, "translation error {trans_err} mm");
    }

    /// Noise must not move the answer much — NCC is a correlation, so zero-mean noise dilutes
    /// its value without moving where its maximum sits.
    ///
    /// The noise is scaled to the volume's own standard deviation rather than its peak, because
    /// the peak is one small bright blob: "30% of the contrast actually present" is the honest
    /// way to state the level, and it is applied to *both* volumes independently, which is twice
    /// the perturbation of noising only one.
    #[test]
    fn recovery_survives_noise() {
        let dims = (48, 48, 48);
        let affine = identity_affine((1.0, 1.0, 1.0));
        let clean = phantom(dims);
        let truth = truth_transform(&grid_center_world(dims, &affine), [7.0, 5.0, -9.0], [-2.0, 1.0, 2.0]);
        let mut moving = warp(&clean, dims, &affine, dims, &affine, &truth);
        let mut fixed = clean.clone();

        let n = clean.len() as f64;
        let mean = clean.iter().sum::<f64>() / n;
        let std = (clean.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
        let sigma = 0.3 * std;

        // SplitMix64 -> Box-Muller. The crate avoids an RNG dependency and the test only needs
        // repeatability.
        let mut state = 0xC0FFEEu64;
        let mut uniform = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
        };
        for v in fixed.iter_mut().chain(moving.iter_mut()) {
            let (u1, u2) = (uniform().max(1e-12), uniform());
            *v += sigma * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        }

        let t = register_rigid(&fixed, dims, &affine, &moving, dims, &affine, None, &RigidParams::default())
            .unwrap();
        let rot_err = rotation_error_deg(&t.matrix, &truth);
        let trans_err = translation_error_mm(&t.matrix, &truth);
        println!("noisy (sigma {sigma:.3}): rot err {rot_err:.3}deg, trans err {trans_err:.3}mm, ncc {:.3}", t.ncc);
        assert!(rot_err < 2.0, "rotation error {rot_err} deg");
        assert!(trans_err < 2.0, "translation error {trans_err} mm");
    }

    /// Different grids, not just different contents: the moving volume is on a coarser,
    /// differently-placed grid, which is the separately-prescribed-slab case.
    #[test]
    fn registers_across_different_grids() {
        let fixed_dims = (48, 48, 48);
        let fixed_affine = identity_affine((1.0, 1.0, 1.0));
        let fixed = phantom(fixed_dims);

        let moving_dims = (40, 40, 30);
        let moving_affine = [
            1.2, 0.0, 0.0, 3.0, //
            0.0, 1.2, 0.0, -2.0, //
            0.0, 0.0, 1.5, 1.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let truth = truth_transform(&grid_center_world(fixed_dims, &fixed_affine), [6.0, -5.0, 7.0], [1.0, -1.0, 2.0]);
        let moving = warp(&fixed, fixed_dims, &fixed_affine, moving_dims, &moving_affine, &truth);

        let t = register_rigid(
            &fixed, fixed_dims, &fixed_affine,
            &moving, moving_dims, &moving_affine,
            None, &RigidParams::default(),
        )
        .unwrap();
        let rot_err = rotation_error_deg(&t.matrix, &truth);
        println!("cross-grid: rot err {rot_err:.3}deg, overlap {:.2}, ncc {:.3}", t.overlap, t.ncc);
        assert!(rot_err < 2.0, "rotation error {rot_err} deg");
        assert!(translation_error_mm(&t.matrix, &truth) < 2.0);
    }

    /// Resampling through the recovered transform must put the moving volume back on the fixed
    /// grid. This is the half of the result COSMOS consumes directly, and it is a different
    /// claim from "the matrix is close to the truth" — it also pins the direction the transform
    /// is applied in.
    #[test]
    fn resampling_through_the_transform_restores_the_fixed_volume() {
        let dims = (48, 48, 48);
        let affine = identity_affine((1.0, 1.0, 1.0));
        let fixed = phantom(dims);
        let truth = truth_transform(&grid_center_world(dims, &affine), [10.0, 0.0, -8.0], [2.0, 2.0, -2.0]);
        let moving = warp(&fixed, dims, &affine, dims, &affine, &truth);
        let t = register_rigid(&fixed, dims, &affine, &moving, dims, &affine, None, &RigidParams::default())
            .unwrap();

        let back = t.resample(&moving).unwrap();
        // Compare where the phantom actually has signal; the rotated corners are empty in one
        // volume and not the other, which is a property of the field of view and not of the
        // registration.
        let (mut sf, mut sb, mut sff, mut sbb, mut sfb, mut n) = (0.0, 0.0, 0.0, 0.0, 0.0, 0usize);
        for (&a, &b) in fixed.iter().zip(back.iter()) {
            if a <= 0.0 && b <= 0.0 {
                continue;
            }
            sf += a;
            sb += b;
            sff += a * a;
            sbb += b * b;
            sfb += a * b;
            n += 1;
        }
        let n = n as f64;
        let corr = (sfb - sf * sb / n)
            / ((sff - sf * sf / n).sqrt() * (sbb - sb * sb / n).sqrt());
        println!("round-trip correlation {corr:.5}");
        assert!(corr > 0.98, "round-trip correlation {corr}");

        // And the wrong direction really is wrong: resampling with the *inverse* alignment
        // leaves the volume doubly rotated, so this must not also pass.
        let flipped = RigidTransform { matrix: rigid_inverse(&t.matrix), ..t.clone() };
        let bad = flipped.resample(&moving).unwrap();
        let (mut sf2, mut sb2, mut sff2, mut sbb2, mut sfb2, mut n2) = (0.0, 0.0, 0.0, 0.0, 0.0, 0usize);
        for (&a, &b) in fixed.iter().zip(bad.iter()) {
            if a <= 0.0 && b <= 0.0 {
                continue;
            }
            sf2 += a;
            sb2 += b;
            sff2 += a * a;
            sbb2 += b * b;
            sfb2 += a * b;
            n2 += 1;
        }
        let n2 = n2 as f64;
        let bad_corr = (sfb2 - sf2 * sb2 / n2)
            / ((sff2 - sf2 * sf2 / n2).sqrt() * (sbb2 - sb2 * sb2 / n2).sqrt());
        println!("wrong-direction correlation {bad_corr:.5}");
        assert!(bad_corr < corr - 0.01, "the inverted transform scored {bad_corr}, the correct one {corr}");
    }

    // ---------------------------------------------------------- B0 transport

    /// With no rotation, the B0 direction in the fixed frame is just the fixed affine's own —
    /// the degenerate case the general formula has to reduce to.
    #[test]
    fn b0_without_rotation_matches_the_fixed_affine() {
        let dims = (16, 16, 16);
        let oblique = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 0.9397, -0.3420, 0.0, //
            0.0, 0.3420, 0.9397, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let t = RigidTransform::identity(dims, &oblique, dims, &oblique);
        let got = t.b0_direction_in_fixed(None);
        let want = b0_direction_from_affine(&oblique);
        assert!((got.0 - want.0).abs() < 1e-12);
        assert!((got.1 - want.1).abs() < 1e-12);
        assert!((got.2 - want.2).abs() < 1e-12);
    }

    /// The case the whole feature exists for: identical affines, the head rotated in voxel
    /// space. The affine says `(0,0,1)` for both orientations, and the registration is what
    /// recovers the real direction.
    ///
    /// A 20-degree rotation of the object about world x tilts B0, *in the common frame*, by 20
    /// degrees the other way. The sign is the whole point: get it backwards and COSMOS gets a
    /// kernel rotated 40 degrees from the truth while every diagnostic still looks plausible.
    #[test]
    fn registration_recovers_b0_where_the_affine_cannot() {
        let dims = (48, 48, 48);
        let affine = identity_affine((1.0, 1.0, 1.0));
        let fixed = phantom(dims);
        let angle_deg = 20.0f64;
        let angle = angle_deg.to_radians();
        let truth = truth_transform(&grid_center_world(dims, &affine), [angle_deg, 0.0, 0.0], [0.0, 0.0, 0.0]);
        let moving = warp(&fixed, dims, &affine, dims, &affine, &truth);

        // Both volumes carry the same affine, so this is what a pipeline reading headers sees.
        assert_eq!(b0_direction_from_affine(&affine), (0.0, 0.0, 1.0));

        let t = register_rigid(&fixed, dims, &affine, &moving, dims, &affine, None, &RigidParams::default())
            .unwrap();
        let b = t.b0_direction_in_fixed(None);

        // Rx(angle) maps fixed world to moving world, so B0 (moving world +z) sits at
        // Rx(angle)ᵀ·(0,0,1) = (0, sin angle, cos angle) in the common frame.
        let want = (0.0, angle.sin(), angle.cos());
        println!("b0 recovered {b:?}, expected {want:?}");
        for (got, exp) in [(b.0, want.0), (b.1, want.1), (b.2, want.2)] {
            assert!((got - exp).abs() < 0.03, "{b:?} vs {want:?}");
        }
        // And the sign is not a coin flip: the y component must be positive, not negative.
        assert!(b.1 > 0.1, "B0 tilted the wrong way: {b:?}");

        // The tilt is large enough that a multi-orientation check would accept the pair, which
        // is exactly the error this converts into a supported path.
        let spread = b.2.abs().clamp(0.0, 1.0).acos().to_degrees();
        assert!(spread > 15.0, "spread against the reference is only {spread} deg");
    }

    /// A declared (sidecar) direction has to be carried through the same rotation. Passing the
    /// affine-derived direction explicitly must give the same answer as passing `None`.
    #[test]
    fn declared_b0_goes_through_the_same_transport() {
        let dims = (16, 16, 16);
        let fixed_affine = identity_affine((1.0, 1.0, 1.0));
        let moving_affine = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 0.866, -0.5, 0.0, //
            0.0, 0.5, 0.866, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let mut t = RigidTransform::identity(dims, &fixed_affine, dims, &moving_affine);
        t.matrix = matrix_from_params(&[0.2, -0.1, 0.3, 0.0, 0.0, 0.0], &grid_center_world(dims, &fixed_affine));

        let implicit = t.b0_direction_in_fixed(None);
        let explicit = t.b0_direction_in_fixed(Some(b0_direction_from_affine(&moving_affine)));
        assert!((implicit.0 - explicit.0).abs() < 1e-12);
        assert!((implicit.1 - explicit.1).abs() < 1e-12);
        assert!((implicit.2 - explicit.2).abs() < 1e-12);

        // A different declared direction must give a different answer, or the argument is
        // being ignored.
        let other = t.b0_direction_in_fixed(Some((1.0, 0.0, 0.0)));
        assert!(
            (other.0 - implicit.0).abs() + (other.1 - implicit.1).abs() + (other.2 - implicit.2).abs() > 0.1,
            "declared direction had no effect"
        );
    }

    /// `aligned_affine` must place the moving volume where resampling says it is: sampling the
    /// retagged volume on the fixed grid is the same operation `resample` performs.
    #[test]
    fn aligned_affine_agrees_with_resampling() {
        let dims = (32, 32, 32);
        let affine = identity_affine((1.0, 1.0, 1.0));
        let moving = phantom(dims);
        let mut t = RigidTransform::identity(dims, &affine, dims, &affine);
        t.matrix = matrix_from_params(&[0.15, -0.1, 0.25, 1.0, -2.0, 0.5], &grid_center_world(dims, &affine));

        let via_method = t.resample(&moving).unwrap();
        let via_affine =
            resample_onto(&moving, dims, &t.aligned_affine(), dims, &affine).unwrap();
        for (a, b) in via_method.iter().zip(via_affine.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn inverse_swaps_the_roles() {
        let fixed_dims = (32, 32, 32);
        let moving_dims = (24, 24, 20);
        let fixed_affine = identity_affine((1.0, 1.0, 1.0));
        let moving_affine = identity_affine((1.5, 1.5, 2.0));
        let mut t = RigidTransform::identity(fixed_dims, &fixed_affine, moving_dims, &moving_affine);
        t.matrix = matrix_from_params(&[0.2, 0.1, -0.3, 3.0, -1.0, 2.0], &grid_center_world(fixed_dims, &fixed_affine));

        let inv = t.inverse();
        assert_eq!(inv.fixed_dims, moving_dims);
        assert_eq!(inv.moving_dims, fixed_dims);
        // Its stored params must regenerate its own matrix.
        let rebuilt = matrix_from_params(&inv.params, &grid_center_world(moving_dims, &moving_affine));
        for (i, (&got, &want)) in rebuilt.iter().zip(inv.matrix.iter()).enumerate() {
            assert!((got - want).abs() < 1e-9, "element {i}: {got} vs {want}");
        }
        // And inverting twice must come back.
        let round = inv.inverse();
        for (i, (&got, &want)) in round.matrix.iter().zip(t.matrix.iter()).enumerate() {
            assert!((got - want).abs() < 1e-9, "element {i}: {got} vs {want}");
        }
    }

    // ---------------------------------------------------------- guards

    #[test]
    fn rejects_mismatched_lengths_and_singular_affines() {
        let dims = (8, 8, 8);
        let affine = identity_affine((1.0, 1.0, 1.0));
        let v = vec![0.0f64; 8 * 8 * 8];
        let short = vec![0.0f64; 10];
        let p = RigidParams::default();
        assert!(register_rigid(&short, dims, &affine, &v, dims, &affine, None, &p).is_none());
        assert!(register_rigid(&v, dims, &affine, &short, dims, &affine, None, &p).is_none());
        assert!(register_rigid(&v, dims, &affine, &v, dims, &affine, Some(&[1u8; 4]), &p).is_none());
        let singular = [0.0f64; 16];
        assert!(register_rigid(&v, dims, &singular, &v, dims, &affine, None, &p).is_none());
        assert!(register_rigid(&v, dims, &affine, &v, dims, &singular, None, &p).is_none());
    }

    /// A mask must actually restrict the metric.
    ///
    /// The fixed volume gets bright, structured content *outside* the mask that the moving volume
    /// does not have — which is the real situation, since the receive field is fixed to the coil
    /// and the scalp and neck differ between orientations. Masked, the registration should land;
    /// unmasked, it should do measurably worse. Scaling that content to the phantom's own peak
    /// rather than swamping it keeps the comparison about the mask and not about arithmetic.
    #[test]
    fn the_mask_restricts_the_metric() {
        let dims = (48, 48, 48);
        let affine = identity_affine((1.0, 1.0, 1.0));
        let fixed = phantom(dims);
        let truth = truth_transform(&grid_center_world(dims, &affine), [8.0, 0.0, 6.0], [1.0, -1.0, 0.0]);
        let moving = warp(&fixed, dims, &affine, dims, &affine, &truth);

        let (nx, ny, nz) = dims;
        let peak = fixed.iter().cloned().fold(0.0f64, f64::max);
        let mut mask = vec![0u8; nx * ny * nz];
        let mut spoiled = fixed.clone();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = i + j * nx + k * nx * ny;
                    // An ellipsoid a little larger than the phantom's body: the "brain mask".
                    let (x, y, z) = (i as f64 / nx as f64, j as f64 / ny as f64, k as f64 / nz as f64);
                    let d = ((x - 0.5) / 0.38).powi(2) + ((y - 0.5) / 0.30).powi(2) + ((z - 0.5) / 0.34).powi(2);
                    let inside = d <= 1.0;
                    mask[idx] = inside as u8;
                    if !inside {
                        spoiled[idx] = 0.6 * peak
                            * (1.0 + (0.7 * i as f64).sin() * (0.5 * j as f64).cos() * (0.3 * k as f64).sin());
                    }
                }
            }
        }
        let p = RigidParams::default();
        let masked = register_rigid(&spoiled, dims, &affine, &moving, dims, &affine, Some(&mask), &p).unwrap();
        let unmasked = register_rigid(&spoiled, dims, &affine, &moving, dims, &affine, None, &p).unwrap();
        let masked_err = rotation_error_deg(&masked.matrix, &truth);
        let unmasked_err = rotation_error_deg(&unmasked.matrix, &truth);
        println!("masked err {masked_err:.3}deg (ncc {:.3}), unmasked err {unmasked_err:.3}deg (ncc {:.3})",
                 masked.ncc, unmasked.ncc);
        assert!(masked_err < 1.5, "masked rotation error {masked_err} deg");
        assert!(
            unmasked_err > 2.0 * masked_err,
            "the mask made no real difference: masked {masked_err}, unmasked {unmasked_err}"
        );
    }

    /// `max_samples` is the knob that bounds cost. Lowering it must reduce the work done
    /// without destroying the answer — the claim the module docs make.
    #[test]
    fn max_samples_bounds_the_work() {
        let dims = (48, 48, 48);
        let affine = identity_affine((1.0, 1.0, 1.0));
        let fixed = phantom(dims);
        let truth = truth_transform(&grid_center_world(dims, &affine), [7.0, 0.0, -5.0], [1.0, 0.0, -1.0]);
        let moving = warp(&fixed, dims, &affine, dims, &affine, &truth);

        let cheap = RigidParams { max_samples: 3000, ..Default::default() };
        let t = register_rigid(&fixed, dims, &affine, &moving, dims, &affine, None, &cheap).unwrap();
        let err = rotation_error_deg(&t.matrix, &truth);
        println!("3000-sample registration: rot err {err:.3}deg");
        assert!(err < 2.0, "rotation error {err} deg with a 3000-sample cap");

        // The cap is honoured: a level must never build more samples than asked for.
        let v = Volume::borrowed(&fixed, dims, affine);
        assert!(sample_set(&v, None, 3000).len() <= 3000);
        assert!(sample_set(&v, None, 200_000).len() <= 200_000);
    }

    #[test]
    fn evaluation_budget_is_respected() {
        let dims = (32, 32, 32);
        let affine = identity_affine((1.0, 1.0, 1.0));
        let fixed = phantom(dims);
        let moving = warp(
            &fixed,
            dims,
            &affine,
            dims,
            &affine,
            &truth_transform(&grid_center_world(dims, &affine), [6.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        );
        let p = RigidParams { max_evaluations: 40, ..Default::default() };
        let t = register_rigid(&fixed, dims, &affine, &moving, dims, &affine, None, &p).unwrap();
        assert!(t.evaluations <= 60, "budget 40 overrun to {}", t.evaluations);
    }
}

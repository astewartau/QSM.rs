//! Retrospective motion correction for GRE/QSM: co-register a series of volumes onto one of
//! them, in the complex domain, and re-derive B0 for each.
//!
//! This is the image-domain half of motion correction, and it is the half a library that
//! consumes reconstructed NIfTI can actually do. The best QSM motion correction in the
//! literature is *prospective* — spherical or dual-echo spiral navigators driving a
//! conjugate-phase reconstruction — and it needs raw k-space plus a modified sequence. Nothing
//! here can reach that; the boundary is recorded so it is not mistaken for a gap.
//!
//! What this module does is what the field does in image space. The most recent published
//! example registers every echo of a UTE acquisition to the first TE as a fixed reference with a
//! rigid algorithm, and reports that it significantly reduced streaking from substantial
//! inter-scan motion — with the honest caveat that a residual boundary effect remains from
//! motion it could not correct. Expect the same shape of result here: a large improvement, not a
//! clean recovery.
//!
//! # The two things that are easy to get wrong
//!
//! **Phase must move through the complex domain.** Wrapped phase cannot be interpolated: halfway
//! between `+3.0` and `−3.0` rad a linear interpolator returns `0.0` when the answer is near
//! `±π`, so every wrap in the volume becomes a band of wrong values. Registration is therefore
//! estimated on **magnitude** and applied to the magnitude/phase pair together through
//! [`crate::geometry::resample_complex_onto`], which interpolates `mag·e^{iφ}`.
//! [`MotionSeries::apply_complex`] is the only phase-moving entry point here, and
//! [`MotionSeries::apply`] is documented as refusing to be the one.
//!
//! **B0 must be re-derived per volume.** When the head rotates, B0 does not: the field direction
//! *relative to the head* changes, so the dipole kernel orientation changes with it. A series
//! that moved is a series with one B0 direction per volume, and reusing a single one produces a
//! plausible-looking wrong susceptibility map rather than an error. This is the same physics that
//! makes COSMOS ([`crate::inversion::cosmos`]) work, and the transport lives in
//! [`crate::registration::RigidTransform::b0_direction_in_fixed`]; [`VolumeMotion::bdir`] is just
//! that value, carried alongside the volume it belongs to so a caller cannot lose track of which
//! direction goes with which field map.
//!
//! # Two ways to finish
//!
//! Having put the series on one grid there are two things to do with it, and they are not
//! equally safe:
//!
//! - **Pass it downstream per volume** ([`correct_motion`]), each field map with its own `bdir`.
//!   Always correct. With enough orientations and enough spread this *is* COSMOS.
//! - **Average it** ([`MotionSeries::average_complex`]). Correct for magnitude, and the right
//!   thing for repeats that barely moved, but it averages phase accumulated under *different*
//!   dipole-kernel orientations, so it is an approximation that degrades with the rotation it is
//!   averaging over. [`MotionSeries::rotation_spread_deg`] is the number to look at before
//!   trusting it.
//!
//! # What is deliberately not here
//!
//! Detecting a corrupted volume and *downweighting or dropping* it — a per-echo fit residual as a
//! corruption detector, feeding a weighted echo combination — is a separate piece of work that
//! belongs with the weighting machinery in [`crate::utils::multi_echo`], not here.
//! [`VolumeMotion::suspect`] reports when an *estimate* looks untrustworthy, which is a
//! registration diagnostic; acting on it is not this module's job.
//!
//! # Example
//!
//! ```no_run
//! use qsm_core::motion::{correct_motion, MotionParams, MotionReference};
//! # fn f(mags: &[Vec<f64>], phases: &[Vec<f64>], dims: (usize, usize, usize),
//! #      affine: &[f64; 16], brain: &[u8]) -> Option<()> {
//! let params = MotionParams {
//!     reference: MotionReference::First, // the first TE, as the UTE work does
//!     ..Default::default()
//! };
//! let corrected = correct_motion(mags, phases, dims, affine, Some(brain), &params)?;
//! println!(
//!     "max tissue displacement {:.2} mm over {:.2}° of rotation",
//!     corrected.motion.max_displacement_mm(),
//!     corrected.motion.rotation_spread_deg(),
//! );
//! // Each volume on the reference grid, each with the B0 direction it was acquired under.
//! for (phase, bdir) in corrected.phases.iter().zip(corrected.bdirs.iter()) {
//!     let _ = (phase, bdir);
//! }
//! # Some(()) }
//! ```

use crate::geometry::b0_direction_from_affine;
use crate::registration::{register_rigid, RigidParams, RigidTransform};

/// Which volume of the series the rest are aligned onto.
///
/// The reference defines the output grid and is itself left untouched, so it is the one volume
/// that never passes through an interpolator. Prefer the volume with the best SNR and the least
/// motion; for a multi-echo series that is conventionally the first echo, which is also what the
/// published UTE procedure uses.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MotionReference {
    /// The first volume — the first echo, or the first repeat.
    #[default]
    First,
    /// The last volume.
    Last,
    /// A specific index. Out of range makes [`estimate_motion`] return `None`.
    Index(usize),
}

impl MotionReference {
    /// Resolve to an index within a series of `n` volumes, or `None` if it does not name one.
    pub fn resolve(self, n: usize) -> Option<usize> {
        match self {
            MotionReference::First if n > 0 => Some(0),
            MotionReference::Last if n > 0 => Some(n - 1),
            MotionReference::Index(i) if i < n => Some(i),
            _ => None,
        }
    }
}

/// Motion-correction parameters.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct MotionParams {
    /// Which volume the others are aligned onto.
    pub reference: MotionReference,
    /// Parameters for each pairwise rigid registration.
    ///
    /// [`RigidParams::initial`] is **overwritten** when [`Self::chain`] is set, which is the
    /// default; set `chain: false` to have it respected as a fixed seed for every volume.
    pub rigid: RigidParams,
    /// Seed each volume's search from the previous volume's solution rather than from the
    /// identity.
    ///
    /// A series is acquired in order, so volume *t* usually sits close to volume *t−1* — the
    /// strong prior [`RigidParams::initial`] exists for. The walk starts at the reference and
    /// moves outward in both directions, so the seed is always the nearest already-solved
    /// neighbour. All volumes register against the same fixed volume, so the parameters are
    /// defined about one rotation centre and carry between pairs without conversion.
    ///
    /// The reference's own two neighbours are excepted: their nearest solved neighbour *is* the
    /// reference, whose solution is the identity by construction and therefore not a prior at
    /// all, so they run the full pyramid rather than trade capture range for nothing.
    ///
    /// Worth turning off if the series order is not acquisition order, or if one volume is so
    /// corrupted that its solution would be a bad seed for its neighbour rather than a good one.
    ///
    /// On its own a seed buys almost nothing — see [`Self::chain_levels`], which is what makes it
    /// pay.
    pub chain: bool,
    /// Pyramid levels to use for a volume that has a real chained seed, replacing
    /// [`RigidParams::levels`]. Ignored for the reference's two neighbours, which have no real
    /// seed — see [`Self::chain`].
    ///
    /// A seed alone is nearly free of benefit, which is worth stating because it is not what the
    /// hint on [`RigidParams::initial`] suggests. Measured on the 164x205x205 phantom, a
    /// four-volume drifting series registered against one reference:
    ///
    /// | | metric evaluations | worst rotation error |
    /// |---|---|---|
    /// | cold start, 5 levels | 4372 | 0.012° |
    /// | chained seed, 5 levels | 4310 | 0.012° |
    /// | chained seed, 3 levels | 2331 | 0.012° |
    /// | chained seed, 1 level | 938 | 0.013° |
    ///
    /// The coarse levels throw the seed away: they smooth and decimate until the cost surface's
    /// own minimum is somewhere else, walk to it, and arrive at the finer levels having learned
    /// nothing from the prior. A seed only pays once the coarse search it is meant to *replace*
    /// is removed.
    ///
    /// Evaluations are the honest unit here, and they are not all priced the same: a level below
    /// the third costs a few thousand samples against the 200k cap
    /// ([`RigidParams::max_samples`]), so dropping those levels removes evaluations that were
    /// nearly free. Expect the wall-clock saving to be smaller than the ratio above — the figures
    /// are deliberately not quoted in seconds, because the machine they were measured on was not
    /// quiet enough for that to mean anything.
    ///
    /// `3` is the default because the pyramid's job is capture range, and the capture range a
    /// seeded registration needs is only the motion *between neighbouring volumes* — not the
    /// whole head rotation. Three levels was measured at 40° in [`RigidParams::levels`], which
    /// covers any inter-volume motion that is not an outright jump, at half the cost of the full
    /// pyramid. Drop it to `1` for a series known to drift smoothly; raise it to
    /// [`RigidParams::levels`] to pay the full price for the full capture range, which is what
    /// `chain: false` does anyway.
    pub chain_levels: usize,
    /// B0 direction in each volume's **own voxel frame**, in the convention
    /// [`b0_direction_from_affine`] returns, when a sidecar declares it. `None` reads it off each
    /// volume's affine, which is right whenever that affine is the scanner's own.
    ///
    /// One value for the series, not one per volume: a series is one prescription, so every
    /// volume was acquired with B0 in the same place relative to its own grid. What differs
    /// between them is where the *head* was, which is what the registration recovers.
    pub declared_b0: Option<(f64, f64, f64)>,
    /// NCC below which an estimate is marked [`VolumeMotion::suspect`].
    ///
    /// A floor for "something went wrong", not a quality target. Two *repeats* that genuinely
    /// align land above ~0.99, but echoes do not, and the default is set for echoes because they
    /// are the common case: on the four-echo 7 T phantom, echoes at TE 12, 20 and 28 ms
    /// registered to a 4 ms reference scored NCC 0.84, 0.68 and 0.55 — while recovering a known
    /// 2.5-7.5 degree rotation to 0.04 degrees every time. See [`estimate_motion`] for why.
    /// `0.3` sits below that and still well clear of the ~0.2 a genuinely failed search returns;
    /// `0.5` would have flagged that last echo, whose transform was right to four decimal
    /// places. Raise it to ~0.9 for a series of repeats, where a low value really would mean
    /// something was wrong.
    pub min_ncc: f64,
}

impl Default for MotionParams {
    fn default() -> Self {
        Self {
            reference: MotionReference::First,
            rigid: RigidParams::default(),
            chain: true,
            chain_levels: 3,
            declared_b0: None,
            min_ncc: 0.3,
        }
    }
}

/// One volume of a series, borrowed: the magnitude to register on and the grid it lives on.
///
/// Volumes of one series normally share a grid — [`uniform_series`] builds that case in one
/// call — but separately prescribed runs need not, so the grid is carried per volume.
#[derive(Clone, Copy, Debug)]
pub struct SeriesVolume<'a> {
    /// Magnitude. Registration is estimated on this and never on phase.
    pub magnitude: &'a [f64],
    /// Volume dimensions, column-major.
    pub dims: (usize, usize, usize),
    /// Row-major voxel→world affine.
    pub affine: [f64; 16],
}

/// Borrow a series of volumes that all share one grid — the echoes of one acquisition, or
/// repeats of one prescription.
pub fn uniform_series<'a, T: AsRef<[f64]>>(
    magnitudes: &'a [T],
    dims: (usize, usize, usize),
    affine: &[f64; 16],
) -> Vec<SeriesVolume<'a>> {
    magnitudes
        .iter()
        .map(|m| SeriesVolume {
            magnitude: m.as_ref(),
            dims,
            affine: *affine,
        })
        .collect()
}

/// Where one volume of a series was, and which way B0 pointed while it was there.
#[derive(Clone, Debug)]
pub struct VolumeMotion {
    /// Index into the input series.
    pub index: usize,
    /// The recovered alignment onto the reference volume. Identity for the reference itself.
    pub transform: RigidTransform,
    /// This volume's B0 direction in the **reference grid's** voxel frame — the direction its
    /// field map was acquired under, and the one a dipole kernel built on the reference grid
    /// has to use for it.
    ///
    /// For the reference volume this is just its own direction. For a volume whose head had
    /// rotated it is that direction carried through the recovered rotation, which is the whole
    /// point: the object rotated and B0 did not.
    pub bdir: (f64, f64, f64),
    /// Total rotation angle in degrees, regardless of axis.
    pub rotation_deg: f64,
    /// Translation magnitude in mm of world space.
    pub translation_mm: f64,
    /// Largest distance any tissue inside the evaluation domain actually moved, in mm.
    ///
    /// Rotation and translation are not comparable to each other or to a voxel size, and a small
    /// rotation about a distant centre moves tissue a long way. This is the single number that
    /// answers "did this matter": compare it against the voxel size. The domain is the mask when
    /// one was passed to [`estimate_motion`] and the reference grid's corners otherwise; the
    /// maximum of a rigid displacement field over a convex region is attained at an extreme
    /// point, so the corner case is exact for the box rather than a sample of it.
    pub max_displacement_mm: f64,
    /// Normalised cross-correlation at the solution. `1.0` for the reference volume.
    pub ncc: f64,
    /// Fraction of sampled reference voxels that found a sample in this volume. `1.0` for the
    /// reference volume.
    pub overlap: f64,
    /// The **estimate** is not to be trusted: NCC below [`MotionParams::min_ncc`], or overlap
    /// below [`RigidParams::min_overlap`].
    ///
    /// Not a motion flag — a volume can move a long way and be registered perfectly, and that is
    /// the normal case. This says the registration itself may have failed, so `transform`,
    /// `bdir` and everything derived from them are questionable. Deciding what to do about a
    /// volume whose *data* is corrupted — dropping it, downweighting it in an echo
    /// combination — is a separate problem that belongs with [`crate::utils::multi_echo`]'s
    /// weighting, and this module deliberately does not do it.
    pub suspect: bool,
}

/// Where every volume of a series was, relative to the reference.
///
/// The output grid is the reference volume's grid throughout: `apply*` resamples onto it, and
/// [`VolumeMotion::bdir`] is expressed in it.
#[derive(Clone, Debug)]
pub struct MotionSeries {
    /// Index of the reference volume within the series.
    pub reference: usize,
    /// One entry per input volume, in input order. The reference's own entry is present and is
    /// the identity, so indices line up with the caller's data and nothing has to be skipped.
    pub volumes: Vec<VolumeMotion>,
}

/// A series of volumes put onto the reference grid, with the B0 direction each was acquired
/// under. The "pass downstream" half of Tier A.
#[derive(Clone, Debug)]
pub struct CorrectedSeries {
    /// Per-volume magnitude on the reference grid.
    pub magnitudes: Vec<Vec<f64>>,
    /// Per-volume wrapped phase on the reference grid, moved through the complex domain so the
    /// wraps survived.
    pub phases: Vec<Vec<f64>>,
    /// Per-volume B0 direction in the reference grid's voxel frame — [`VolumeMotion::bdir`],
    /// flattened for handing straight to a multi-orientation inversion.
    pub bdirs: Vec<(f64, f64, f64)>,
    /// Reference voxels every volume of the series reaches. Resampling can only fill the part of
    /// the reference grid a rotated volume still covers, so the rest is not data; masking to
    /// this intersection is what keeps an edge of zeros out of a field fit.
    pub coverage: Vec<u8>,
    /// The motion that was estimated and applied.
    pub motion: MotionSeries,
}

/// A complex average of a motion-corrected series.
///
/// Averaging phase across volumes acquired at **different head orientations** averages phase
/// accumulated under different dipole-kernel orientations, so this is an approximation whose
/// error grows with [`MotionSeries::rotation_spread_deg`]. It is exactly right for repeats that
/// did not rotate, a good approximation for a degree or two, and the wrong tool once the spread
/// is large enough to matter — at which point the spread is itself the signal, and
/// [`CorrectedSeries`] with a per-volume `bdir` is the honest path.
#[derive(Clone, Debug)]
pub struct MotionAverage {
    /// Complex-mean magnitude on the reference grid.
    pub magnitude: Vec<f64>,
    /// Complex-mean wrapped phase on the reference grid.
    pub phase: Vec<f64>,
    /// How many volumes of the series actually reached each reference voxel. `0` means no data,
    /// not a zero measurement.
    pub contributions: Vec<u32>,
    /// Rotation spread the average was taken over, in degrees — [`MotionSeries::rotation_spread_deg`],
    /// repeated here because it is the number that decides whether the average is trustworthy.
    pub rotation_spread_deg: f64,
}

// ---------------------------------------------------------------------- estimation

/// Estimate where every volume of a series was, relative to a reference volume.
///
/// `mask` is optional and lives on the **reference** volume's grid. Passing a brain mask is worth
/// it twice over: it keeps air, neck and the receive-field falloff out of the correlation, and it
/// is the domain [`VolumeMotion::max_displacement_mm`] is measured over.
///
/// Returns `None` if the series is empty, if [`MotionParams::reference`] does not name a volume,
/// if any volume's data length disagrees with its dimensions, if the mask length disagrees with
/// the reference volume, or if any pairwise registration could not be evaluated at all. A
/// per-volume failure fails the whole call rather than quietly substituting the identity, because
/// an identity transform for a volume that moved is indistinguishable downstream from a volume
/// that did not move.
///
/// # Echoes are harder than repeats
///
/// Repeats and runs are the easy case: same contrast, and NCC absorbs the gain difference
/// between them. Echoes are not — T2\* decay changes contrast *spatially*, far more in
/// iron-rich grey matter and vein than in white matter, so a late echo is not a scaled copy of
/// the first one. NCC still finds the alignment, because the anatomy's edges stay where they
/// are, but its *value* falls: on the four-echo 7 T phantom, echoes at TE 12, 20 and 28 ms
/// registered to a 4 ms reference scored 0.84, 0.68 and 0.55 while recovering a known 2.5-7.5
/// degree rotation to 0.04 degrees. A low NCC on an echo series is therefore the contrast and
/// not a failure; [`MotionParams::min_ncc`] is defaulted for that case and should be *raised*
/// for repeats. The same caveat applies, for the same reason, to receive-field shading that
/// rotates with the coil rather than the head — see [`crate::registration`] on what NCC does
/// and does not absorb.
pub fn estimate_motion(
    series: &[SeriesVolume<'_>],
    mask: Option<&[u8]>,
    params: &MotionParams,
) -> Option<MotionSeries> {
    let n = series.len();
    let reference = params.reference.resolve(n)?;

    for v in series {
        if v.magnitude.len() != v.dims.0 * v.dims.1 * v.dims.2 {
            return None;
        }
    }
    let fixed = series[reference];
    if let Some(m) = mask {
        if m.len() != fixed.magnitude.len() {
            return None;
        }
    }

    // The reference's own entry: the identity between its grid and itself, with the metrics it
    // trivially achieves rather than the NaNs `RigidTransform::identity` leaves for a caller who
    // never measured anything. A perfect self-correlation is not a claim about the data, so
    // nothing downstream can be misled by it; a NaN here would silently poison every `max` and
    // comparison over the series.
    let identity = {
        let mut t = RigidTransform::identity(fixed.dims, &fixed.affine, fixed.dims, &fixed.affine);
        t.ncc = 1.0;
        t.overlap = 1.0;
        t
    };

    let mut solved: Vec<Option<RigidTransform>> = vec![None; n];
    solved[reference] = Some(identity);

    // Walk outward from the reference in both directions, so a chained seed is always the
    // nearest already-solved neighbour rather than whichever volume happened to come first.
    let order: Vec<usize> = (reference + 1..n).chain((0..reference).rev()).collect();
    for t in order {
        // A chained seed replaces the coarse search rather than supplementing it, so it comes
        // with its own (shallower) pyramid — but only where the seed is worth something. The
        // reference's own two neighbours are seeded from the reference, whose solution is the
        // identity by construction and so says nothing about where they are; shortening their
        // pyramid would throw away capture range in exchange for a prior that does not exist.
        // They pay full price, which in a series of any length is two registrations.
        let neighbour = if t > reference { t - 1 } else { t + 1 };
        let seed = if params.chain && neighbour != reference {
            solved[neighbour].as_ref().map(|p| p.params)
        } else {
            None
        };
        let rigid = match seed {
            Some(initial) => RigidParams {
                initial: Some(initial),
                levels: params.chain_levels.max(1),
                ..params.rigid.clone()
            },
            None => params.rigid.clone(),
        };
        let moving = series[t];
        let tf = register_rigid(
            fixed.magnitude,
            fixed.dims,
            &fixed.affine,
            moving.magnitude,
            moving.dims,
            &moving.affine,
            mask,
            &rigid,
        )?;
        solved[t] = Some(tf);
    }

    let volumes = solved
        .into_iter()
        .enumerate()
        .map(|(index, tf)| {
            let transform = tf.expect("every index is either the reference or in the walk");
            let bdir = transform.b0_direction_in_fixed(params.declared_b0);
            let [tx, ty, tz] = transform.translation_mm();
            VolumeMotion {
                index,
                rotation_deg: transform.rotation_magnitude_deg(),
                translation_mm: (tx * tx + ty * ty + tz * tz).sqrt(),
                max_displacement_mm: max_displacement_mm(&transform, mask),
                ncc: transform.ncc,
                overlap: transform.overlap,
                suspect: transform.ncc < params.min_ncc
                    || transform.overlap < params.rigid.min_overlap,
                bdir,
                transform,
            }
        })
        .collect();

    Some(MotionSeries { reference, volumes })
}

/// Largest `|M·x − x|` over the evaluation domain, in mm.
///
/// `M` maps fixed world → moving world, and both are the *same scanner's* world — the scanner
/// does not move — so for the bit of tissue the reference has at world `x`, `M·x` is where that
/// same tissue was during the moving acquisition. The difference is the distance it travelled.
fn max_displacement_mm(t: &RigidTransform, mask: Option<&[u8]>) -> f64 {
    let a = &t.fixed_affine;
    let m = &t.matrix;
    let (nx, ny, nz) = t.fixed_dims;
    let displacement = |i: f64, j: f64, k: f64| {
        let w = [
            a[0] * i + a[1] * j + a[2] * k + a[3],
            a[4] * i + a[5] * j + a[6] * k + a[7],
            a[8] * i + a[9] * j + a[10] * k + a[11],
        ];
        let d: [f64; 3] = std::array::from_fn(|r| {
            m[4 * r] * w[0] + m[4 * r + 1] * w[1] + m[4 * r + 2] * w[2] + m[4 * r + 3] - w[r]
        });
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    };
    match mask {
        Some(mask) => {
            let mut worst = 0.0f64;
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        if mask[i + j * nx + k * nx * ny] != 0 {
                            worst = worst.max(displacement(i as f64, j as f64, k as f64));
                        }
                    }
                }
            }
            worst
        }
        // The maximum of a rigid displacement field over a convex region is attained at an
        // extreme point, so the eight corners are exact for the grid's bounding box.
        None => {
            let mut worst = 0.0f64;
            for &k in &[0.0, nz as f64 - 1.0] {
                for &j in &[0.0, ny as f64 - 1.0] {
                    for &i in &[0.0, nx as f64 - 1.0] {
                        worst = worst.max(displacement(i, j, k));
                    }
                }
            }
            worst
        }
    }
}

// ---------------------------------------------------------------------- application

impl MotionSeries {
    /// Number of volumes in the series.
    pub fn len(&self) -> usize {
        self.volumes.len()
    }

    /// Whether the series is empty. Never true for a [`MotionSeries`] [`estimate_motion`]
    /// returned, which rejects an empty series.
    pub fn is_empty(&self) -> bool {
        self.volumes.is_empty()
    }

    /// Dimensions of the output grid — the reference volume's.
    pub fn dims(&self) -> (usize, usize, usize) {
        self.volumes[self.reference].transform.fixed_dims
    }

    /// Voxel→world affine of the output grid — the reference volume's.
    pub fn affine(&self) -> [f64; 16] {
        self.volumes[self.reference].transform.fixed_affine
    }

    /// Every volume's B0 direction in the reference grid's voxel frame, in input order — ready to
    /// hand to a multi-orientation inversion alongside the resampled field maps.
    pub fn bdirs(&self) -> Vec<(f64, f64, f64)> {
        self.volumes.iter().map(|v| v.bdir).collect()
    }

    /// The largest tissue displacement anywhere in the series, in mm — the one number to compare
    /// against the voxel size when deciding whether the motion mattered. See
    /// [`VolumeMotion::max_displacement_mm`].
    pub fn max_displacement_mm(&self) -> f64 {
        self.volumes
            .iter()
            .fold(0.0f64, |m, v| m.max(v.max_displacement_mm))
    }

    /// The largest angle between any two volumes' recovered B0 directions, in degrees.
    ///
    /// This, not the rotation relative to the reference, is what decides whether the series can
    /// be averaged: it is the spread of dipole-kernel orientations the average would be taken
    /// over. Zero for a series that only translated, however far it translated.
    pub fn rotation_spread_deg(&self) -> f64 {
        let mut worst = 0.0f64;
        for (i, a) in self.volumes.iter().enumerate() {
            for b in &self.volumes[i + 1..] {
                let dot = (a.bdir.0 * b.bdir.0 + a.bdir.1 * b.bdir.1 + a.bdir.2 * b.bdir.2)
                    .clamp(-1.0, 1.0);
                worst = worst.max(dot.acos().to_degrees());
            }
        }
        worst
    }

    /// The volumes whose *estimate* looks untrustworthy. See [`VolumeMotion::suspect`].
    pub fn suspect(&self) -> impl Iterator<Item = &VolumeMotion> {
        self.volumes.iter().filter(|v| v.suspect)
    }

    /// Resample volume `t`'s continuous data (magnitude, an unwrapped field map, χ) onto the
    /// reference grid.
    ///
    /// **Not for wrapped phase.** Interpolating wrapped phase directly turns every wrap into a
    /// band of wrong values; use [`Self::apply_complex`], which is why it exists.
    pub fn apply(&self, t: usize, data: &[f64]) -> Option<Vec<f64>> {
        self.volumes.get(t)?.transform.resample(data)
    }

    /// Resample volume `t`'s magnitude and wrapped phase onto the reference grid together,
    /// interpolating `mag·e^{iφ}` so the wraps survive. Returns `(magnitude, phase)`.
    pub fn apply_complex(
        &self,
        t: usize,
        magnitude: &[f64],
        phase: &[f64],
    ) -> Option<(Vec<f64>, Vec<f64>)> {
        self.volumes.get(t)?.transform.resample_complex(magnitude, phase)
    }

    /// Resample volume `t`'s binary mask or label volume onto the reference grid (nearest
    /// neighbour).
    pub fn apply_mask(&self, t: usize, mask: &[u8]) -> Option<Vec<u8>> {
        self.volumes.get(t)?.transform.resample_mask(mask)
    }

    /// Which reference voxels volume `t` reaches at all.
    ///
    /// A rotated volume cannot fill the whole reference grid, and resampling leaves zeros where
    /// it does not reach. Those zeros are "no data", not a zero measurement, and this is what
    /// tells them apart.
    pub fn coverage(&self, t: usize) -> Option<Vec<u8>> {
        let v = self.volumes.get(t)?;
        let (mx, my, mz) = v.transform.moving_dims;
        v.transform.resample_mask(&vec![1u8; mx * my * mz])
    }

    /// Reference voxels that **every** volume of the series reaches.
    pub fn common_coverage(&self) -> Option<Vec<u8>> {
        let (nx, ny, nz) = self.dims();
        let mut out = vec![1u8; nx * ny * nz];
        for t in 0..self.len() {
            let c = self.coverage(t)?;
            for (o, &v) in out.iter_mut().zip(c.iter()) {
                *o &= v;
            }
        }
        Some(out)
    }

    /// Complex-average the series onto the reference grid: resample every volume's
    /// magnitude/phase pair through its recovered transform, then take the complex mean over the
    /// volumes that reached each voxel.
    ///
    /// The plain complex mean, not a magnitude-weighted one: for repeats with the same noise
    /// level that is the maximum-likelihood combination, and weighting by magnitude would bias
    /// the result toward whichever volume happened to be brightest.
    ///
    /// Read [`MotionAverage`] before using this on a series that rotated.
    ///
    /// Returns `None` if the number of magnitude or phase volumes disagrees with the series, or
    /// if any volume's length disagrees with its own grid.
    pub fn average_complex<T: AsRef<[f64]>>(
        &self,
        magnitudes: &[T],
        phases: &[T],
    ) -> Option<MotionAverage> {
        if magnitudes.len() != self.len() || phases.len() != self.len() {
            return None;
        }
        let (nx, ny, nz) = self.dims();
        let n_out = nx * ny * nz;
        let mut re = vec![0.0f64; n_out];
        let mut im = vec![0.0f64; n_out];
        let mut contributions = vec![0u32; n_out];

        for t in 0..self.len() {
            let (mag, pha) = (magnitudes[t].as_ref(), phases[t].as_ref());
            let v = &self.volumes[t];
            let (mx, my, mz) = v.transform.moving_dims;
            if mag.len() != mx * my * mz || pha.len() != mx * my * mz {
                return None;
            }
            let (m, p) = self.apply_complex(t, mag, pha)?;
            let cover = self.coverage(t)?;
            for i in 0..n_out {
                if cover[i] != 0 {
                    re[i] += m[i] * p[i].cos();
                    im[i] += m[i] * p[i].sin();
                    contributions[i] += 1;
                }
            }
        }

        let mut magnitude = vec![0.0f64; n_out];
        let mut phase = vec![0.0f64; n_out];
        for i in 0..n_out {
            if contributions[i] > 0 {
                let inv = 1.0 / contributions[i] as f64;
                let (r, m) = (re[i] * inv, im[i] * inv);
                magnitude[i] = r.hypot(m);
                phase[i] = m.atan2(r);
            }
        }

        Some(MotionAverage {
            magnitude,
            phase,
            contributions,
            rotation_spread_deg: self.rotation_spread_deg(),
        })
    }
}

/// Estimate the motion across a complex series and put every volume onto the reference grid.
///
/// The whole of Tier A in one call, for a series that shares a grid: register on magnitude,
/// resample the magnitude/phase pair through the complex domain, and carry each volume's B0
/// direction along with it.
///
/// `mask` is optional and lives on the reference volume's grid; see [`estimate_motion`] for what
/// it buys and for the `None` cases.
pub fn correct_motion<T: AsRef<[f64]>>(
    magnitudes: &[T],
    phases: &[T],
    dims: (usize, usize, usize),
    affine: &[f64; 16],
    mask: Option<&[u8]>,
    params: &MotionParams,
) -> Option<CorrectedSeries> {
    if magnitudes.len() != phases.len() {
        return None;
    }
    let series = uniform_series(magnitudes, dims, affine);
    let motion = estimate_motion(&series, mask, params)?;

    let mut out_mag = Vec::with_capacity(motion.len());
    let mut out_pha = Vec::with_capacity(motion.len());
    for t in 0..motion.len() {
        let (m, p) = motion.apply_complex(t, magnitudes[t].as_ref(), phases[t].as_ref())?;
        out_mag.push(m);
        out_pha.push(p);
    }

    Some(CorrectedSeries {
        magnitudes: out_mag,
        phases: out_pha,
        bdirs: motion.bdirs(),
        coverage: motion.common_coverage()?,
        motion,
    })
}

/// B0 direction a volume's own affine implies, in its own voxel frame — re-exported shape of
/// [`b0_direction_from_affine`] so a caller assembling [`MotionParams::declared_b0`] by hand can
/// see the convention it has to match.
pub fn implied_b0(affine: &[f64; 16]) -> (f64, f64, f64) {
    b0_direction_from_affine(affine)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::resample_complex_onto;

    fn identity_affine(vs: (f64, f64, f64)) -> [f64; 16] {
        [
            vs.0, 0.0, 0.0, 0.0, //
            0.0, vs.1, 0.0, 0.0, //
            0.0, 0.0, vs.2, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ]
    }

    /// Row-major 4x4 product, written out here rather than borrowed from `registration`'s private
    /// helper so the ground truth below shares no machinery with the code under test.
    fn mul4(a: &[f64; 16], b: &[f64; 16]) -> [f64; 16] {
        let mut out = [0.0f64; 16];
        for i in 0..4 {
            for j in 0..4 {
                out[4 * i + j] = (0..4).map(|k| a[4 * i + k] * b[4 * k + j]).sum();
            }
        }
        out
    }

    /// `mul4` is the one piece of algebra the ground truth shares with nothing, so it is pinned
    /// against a product worked out by hand before anything is built on it.
    #[test]
    fn mul4_multiplies_row_major() {
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
        let want = [
            6.0, 1.0, 4.0, 4.0, //
            7.0, 4.0, 1.0, 8.0, //
            1.0, 6.0, 10.0, 8.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let got = mul4(&a, &b);
        for i in 0..16 {
            assert!((got[i] - want[i]).abs() < 1e-12, "element {i}: {} vs {}", got[i], want[i]);
        }
    }

    /// A rigid 4x4 rotating `deg` about `axis` through the grid's world centre, plus `shift`.
    ///
    /// Rodrigues, written out. The ground truth must not come from the same parameterisation the
    /// implementation uses, or a transposed rotation would cancel on both sides of every
    /// comparison and the recovery assertions would be unfailable — which is exactly what
    /// happened to the first draft of `registration`'s tests.
    fn truth(
        axis: [f64; 3],
        deg: f64,
        shift: [f64; 3],
        dims: (usize, usize, usize),
        affine: &[f64; 16],
    ) -> [f64; 16] {
        let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        let k = [axis[0] / n, axis[1] / n, axis[2] / n];
        let (s, c) = deg.to_radians().sin_cos();
        let kx = [
            [0.0, -k[2], k[1]],
            [k[2], 0.0, -k[0]],
            [-k[1], k[0], 0.0],
        ];
        let mut r = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let kk: f64 = (0..3).map(|m| kx[i][m] * kx[m][j]).sum();
                r[i][j] = (i == j) as u8 as f64 + s * kx[i][j] + (1.0 - c) * kk;
            }
        }
        let ctr = [
            (dims.0 as f64 - 1.0) * 0.5 * affine[0],
            (dims.1 as f64 - 1.0) * 0.5 * affine[5],
            (dims.2 as f64 - 1.0) * 0.5 * affine[10],
        ];
        let mut m = [0.0f64; 16];
        for i in 0..3 {
            m[4 * i..4 * i + 3].copy_from_slice(&r[i]);
            m[4 * i + 3] =
                ctr[i] - (r[i][0] * ctr[0] + r[i][1] * ctr[1] + r[i][2] * ctr[2]) + shift[i];
        }
        m[15] = 1.0;
        m
    }

    /// Rotate a free vector by the transpose of the 3x3 of `m`.
    fn rotate_t(m: &[f64; 16], v: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|i| m[i] * v[0] + m[4 + i] * v[1] + m[8 + i] * v[2])
    }

    fn angle_deg(a: (f64, f64, f64), b: [f64; 3]) -> f64 {
        let nb = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
        let dot = (a.0 * b[0] + a.1 * b[1] + a.2 * b[2]) / nb;
        dot.clamp(-1.0, 1.0).acos().to_degrees()
    }

    /// Rotation angle of `Aᵀ·B`, in degrees.
    fn rotation_error_deg(a: &[f64; 16], b: &[f64; 16]) -> f64 {
        let trace: f64 = (0..3)
            .map(|i| (0..3).map(|j| a[4 * j + i] * b[4 * j + i]).sum::<f64>())
            .sum();
        (0.5 * (trace - 1.0)).clamp(-1.0, 1.0).acos().to_degrees()
    }

    fn translation_error_mm(a: &[f64; 16], b: &[f64; 16]) -> f64 {
        [a[3] - b[3], a[7] - b[7], a[11] - b[11]]
            .iter()
            .fold(0.0f64, |m, d| m.max(d.abs()))
    }

    /// A deterministic lumpy phantom: three ellipsoidal blobs of different brightness inside a
    /// larger one, which gives the metric real structure in all three axes. The same phantom
    /// `registration`'s own tests use, so the accuracy these tests can expect is the accuracy
    /// that module measured on it (sub-degree, not the 0.01° it reaches on real data).
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
                        let d = ((x - cx) / rx).powi(2)
                            + ((y - cy) / ry).powi(2)
                            + ((z - cz) / rz).powi(2);
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

    /// An ellipsoid enclosing the phantom's envelope with a margin — a stand-in brain mask, used
    /// where a *domain* is the thing under test.
    ///
    /// Deliberately **not** passed to the registrations that check recovery, which run unmasked.
    /// This phantom's exterior is exactly zero, so the background is information rather than the
    /// noise, neck and bias falloff a mask exists to exclude on real data: masking it away leaves
    /// a centred ellipsoid that is near-symmetric under a half turn, and a shift of five voxels
    /// or more then converges to an exact 180° flip at NCC 0.78 at every pyramid depth. The same
    /// transforms recover to 0.003 mm on the 164x205x205 phantom *with* its real brain mask, so
    /// this is the toy's symmetry and not the masked path — which
    /// `displacement_is_the_distance_tissue_actually_moved` exercises, on a rotation, where it is
    /// well inside the toy's capture range.
    fn phantom_mask(dims: (usize, usize, usize)) -> Vec<u8> {
        let (nx, ny, nz) = dims;
        let mut v = vec![0u8; nx * ny * nz];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let (x, y, z) = (
                        i as f64 / nx as f64,
                        j as f64 / ny as f64,
                        k as f64 / nz as f64,
                    );
                    let d = ((x - 0.50) / 0.37).powi(2)
                        + ((y - 0.50) / 0.32).powi(2)
                        + ((z - 0.52) / 0.34).powi(2);
                    v[i + j * nx + k * nx * ny] = (d <= 1.0) as u8;
                }
            }
        }
        v
    }

    /// A wrapped phase field with real wraps in it: a linear ramp steep enough to wrap several
    /// times across the volume, wrapped into (-pi, pi].
    fn wrapped_ramp(dims: (usize, usize, usize)) -> Vec<f64> {
        let (nx, ny, nz) = dims;
        let tau = 2.0 * std::f64::consts::PI;
        let mut v = vec![0.0f64; nx * ny * nz];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let raw = 0.9 * i as f64 + 0.35 * j as f64 + 0.2 * k as f64;
                    v[i + j * nx + k * nx * ny] = raw - tau * ((raw / tau) + 0.5).floor();
                }
            }
        }
        v
    }

    /// Build the moving volume a known world transform `w` (fixed world → moving world) would
    /// have produced, for a complex pair.
    ///
    /// Sampling through `w·affine` is exactly the forward model `RigidTransform::resample`
    /// inverts, so a registration that recovers `w` is the only way this round-trips — which is
    /// what makes these tests able to catch an inverted or transposed transform rather than just
    /// a blurry one.
    fn warp_complex(
        mag: &[f64],
        pha: &[f64],
        dims: (usize, usize, usize),
        affine: &[f64; 16],
        w: &[f64; 16],
    ) -> (Vec<f64>, Vec<f64>) {
        resample_complex_onto(mag, pha, dims, &mul4(w, affine), dims, affine).unwrap()
    }

    /// Absolute phase differences inside `domain`, each wrapped into (-pi, pi] first so a wrap
    /// does not count as a 2-pi error.
    fn phase_errors(a: &[f64], b: &[f64], domain: &[u8]) -> Vec<f64> {
        let tau = 2.0 * std::f64::consts::PI;
        (0..a.len())
            .filter(|&i| domain[i] != 0)
            .map(|i| {
                let d = a[i] - b[i];
                (d - tau * ((d / tau) + 0.5).floor()).abs()
            })
            .collect()
    }

    fn mean_abs_phase_error(a: &[f64], b: &[f64], domain: &[u8]) -> f64 {
        let e = phase_errors(a, b, domain);
        if e.is_empty() {
            f64::INFINITY
        } else {
            e.iter().sum::<f64>() / e.len() as f64
        }
    }

    /// Fraction of `domain` whose phase is wrong by more than a radian — the shape of damage a
    /// mishandled wrap does, as opposed to the small smooth error interpolation always costs.
    fn fraction_badly_wrong(a: &[f64], b: &[f64], domain: &[u8]) -> f64 {
        let e = phase_errors(a, b, domain);
        if e.is_empty() {
            return 1.0;
        }
        e.iter().filter(|&&v| v > 1.0).count() as f64 / e.len() as f64
    }

    // --------------------------------------------------------- guards

    #[test]
    fn reference_resolves_or_refuses() {
        assert_eq!(MotionReference::First.resolve(4), Some(0));
        assert_eq!(MotionReference::Last.resolve(4), Some(3));
        assert_eq!(MotionReference::Index(2).resolve(4), Some(2));
        assert_eq!(MotionReference::Index(4).resolve(4), None);
        assert_eq!(MotionReference::First.resolve(0), None);
        assert_eq!(MotionReference::Last.resolve(0), None);
        assert_eq!(MotionReference::Index(0).resolve(0), None);
    }

    #[test]
    fn empty_and_mismatched_series_are_refused() {
        let dims = (24, 24, 24);
        let affine = identity_affine((1.0, 1.0, 1.0));
        let p = MotionParams::default();
        assert!(estimate_motion(&[], None, &p).is_none(), "empty series");

        let mag = phantom(dims);
        let series = uniform_series(std::slice::from_ref(&mag), dims, &affine);
        assert!(
            estimate_motion(&series, Some(&[1u8; 10]), &p).is_none(),
            "mask length must match the reference volume"
        );

        let truncated = [SeriesVolume {
            magnitude: &mag[..mag.len() - 1],
            dims,
            affine,
        }];
        assert!(
            estimate_motion(&truncated, None, &p).is_none(),
            "a volume whose length disagrees with its own dims"
        );

        let out_of_range = MotionParams {
            reference: MotionReference::Index(3),
            ..MotionParams::default()
        };
        assert!(
            estimate_motion(&series, None, &out_of_range).is_none(),
            "reference index past the end"
        );

        // And the single-volume series it rejected those variants of does work.
        assert!(estimate_motion(&series, None, &p).is_some());
    }

    // --------------------------------------------------------- recovery and B0

    /// The headline claim: a series where volumes moved by known rigid transforms has those
    /// transforms recovered, and every volume's B0 direction comes back as the known rotation of
    /// the declared direction.
    #[test]
    fn a_known_transform_is_recovered_and_b0_follows_it() {
        let dims = (48, 48, 40);
        let affine = identity_affine((1.0, 1.0, 1.2));
        let mag = phantom(dims);
        let pha = wrapped_ramp(dims);

        // Three volumes: the reference, a mostly-rotated one, a mostly-translated one.
        let moves = [
            truth([0.2, -0.3, 1.0], 9.0, [0.8, -0.5, 0.4], dims, &affine),
            truth([1.0, 0.4, 0.1], 4.0, [-2.2, 1.6, 0.9], dims, &affine),
        ];
        let mut mags = vec![mag.clone()];
        let mut phas = vec![pha.clone()];
        for w in &moves {
            let (m, p) = warp_complex(&mag, &pha, dims, &affine, w);
            mags.push(m);
            phas.push(p);
        }

        let series = uniform_series(&mags, dims, &affine);
        let motion =
            estimate_motion(&series, None, &MotionParams::default()).expect("estimation runs");

        assert_eq!(motion.reference, 0);
        assert_eq!(motion.len(), 3);
        assert_eq!(motion.dims(), dims);
        assert_eq!(motion.volumes[0].rotation_deg, 0.0, "the reference is the identity");
        assert_eq!(motion.volumes[0].translation_mm, 0.0);
        assert_eq!(motion.volumes[0].max_displacement_mm, 0.0);
        assert_eq!(motion.volumes[0].ncc, 1.0);
        assert!(!motion.volumes[0].suspect);

        // B0 is read off the affine here, which is axial, so the reference's direction is voxel
        // +z and each moved volume's is that direction carried back through its own rotation.
        let declared = [0.0, 0.0, 1.0];
        assert!(
            angle_deg(motion.volumes[0].bdir, declared) < 1e-9,
            "reference bdir {:?} should be the affine's own direction",
            motion.volumes[0].bdir
        );

        for (t, w) in moves.iter().enumerate() {
            let v = &motion.volumes[t + 1];
            let rot_err = rotation_error_deg(w, &v.transform.matrix);
            let trans_err = translation_error_mm(w, &v.transform.matrix);
            println!(
                "volume {}: rotation error {rot_err:.4} deg, translation error {trans_err:.4} mm, \
                 NCC {:.4}, displacement {:.3} mm",
                t + 1,
                v.ncc,
                v.max_displacement_mm
            );
            // Measured ~0.4 deg / ~0.2 mm on this phantom, which is as well as `registration`
            // does on it (its own tests assert 1.0-2.0 deg); on real data the same code reaches
            // 0.01 deg. Thresholds set with roughly 2x headroom over the measurement.
            assert!(rot_err < 1.0, "volume {}: rotation off by {rot_err} deg", t + 1);
            assert!(trans_err < 0.5, "volume {}: translation off by {trans_err} mm", t + 1);
            assert!(v.ncc > 0.95, "volume {}: NCC only {}", t + 1, v.ncc);
            assert!(!v.suspect, "volume {} should not be suspect", t + 1);

            // The object rotated and B0 did not: in the reference frame, this volume's B0 sits
            // where the inverse of its rotation puts the declared direction.
            let want = rotate_t(w, declared);
            let err = angle_deg(v.bdir, want);
            println!(
                "volume {}: bdir {:?} vs expected [{:.4} {:.4} {:.4}], {err:.4} deg",
                t + 1,
                v.bdir,
                want[0],
                want[1],
                want[2]
            );
            assert!(err < 1.0, "volume {}: bdir off by {err} deg", t + 1);
            // And it genuinely moved off +z, or the assertion above would pass for free for an
            // implementation that returned the reference direction every time.
            assert!(
                angle_deg(v.bdir, declared) > 1.0,
                "volume {}: bdir {:?} did not move away from the declared direction, so the \
                 agreement above proves nothing",
                t + 1,
                v.bdir
            );
        }
    }

    /// The thresholds above have to be able to fail. Handing the *unmoved* volume in as the
    /// moving one makes the true answer the identity, so scoring it against the same `moves`
    /// must blow through every one of them.
    #[test]
    fn the_recovery_thresholds_can_fail() {
        let dims = (48, 48, 40);
        let affine = identity_affine((1.0, 1.0, 1.2));
        let mag = phantom(dims);
        let w = truth([0.2, -0.3, 1.0], 9.0, [0.8, -0.5, 0.4], dims, &affine);

        let mags = vec![mag.clone(), mag];
        let series = uniform_series(&mags, dims, &affine);
        let motion = estimate_motion(&series, None, &MotionParams::default()).unwrap();
        let v = &motion.volumes[1];
        assert!(
            rotation_error_deg(&w, &v.transform.matrix) > 1.0,
            "a 9-degree rotation must not pass the 1-degree threshold"
        );
        assert!(
            translation_error_mm(&w, &v.transform.matrix) > 0.5,
            "a 0.8mm offset must not pass the 0.5mm threshold"
        );
        assert!(
            angle_deg(v.bdir, rotate_t(&w, [0.0, 0.0, 1.0])) > 1.0,
            "an untransported bdir must not pass the 1-degree threshold"
        );
    }

    /// B0 transport has to honour a declared direction that is *not* voxel `+z`, or the test
    /// above would also pass for an implementation that ignored `declared_b0` and read the affine
    /// every time.
    #[test]
    fn a_declared_b0_is_transported_rather_than_ignored() {
        let dims = (40, 40, 36);
        let affine = identity_affine((1.0, 1.0, 1.0));
        let mag = phantom(dims);
        let w = truth([0.1, 1.0, 0.25], 8.0, [1.0, -0.7, 0.5], dims, &affine);
        let (moved, _) = warp_complex(&mag, &vec![0.0; mag.len()], dims, &affine, &w);

        // An oblique declared direction, normalised.
        let d = {
            let v = [0.30f64, -0.20, 0.9327];
            let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            [v[0] / n, v[1] / n, v[2] / n]
        };
        let params = MotionParams {
            declared_b0: Some((d[0], d[1], d[2])),
            ..MotionParams::default()
        };
        let mags = vec![mag, moved];
        let series = uniform_series(&mags, dims, &affine);
        let motion = estimate_motion(&series, None, &params).unwrap();

        assert!(
            angle_deg(motion.volumes[0].bdir, d) < 1e-9,
            "the reference must report the declared direction unchanged, got {:?}",
            motion.volumes[0].bdir
        );
        let want = rotate_t(&w, d);
        let err = angle_deg(motion.volumes[1].bdir, want);
        // Reading the affine instead of the declaration would have transported +z, which is a
        // different vector; check the two are far enough apart for this test to discriminate.
        let affine_answer = rotate_t(&w, [0.0, 0.0, 1.0]);
        println!(
            "declared-b0 transport error {err:.4} deg; the affine-derived answer would be \
             {:.2} deg away",
            angle_deg(motion.volumes[1].bdir, affine_answer)
        );
        assert!(err < 1.0, "transported declared bdir off by {err} deg");
        assert!(
            angle_deg(motion.volumes[1].bdir, affine_answer) > 5.0,
            "the declared and affine-derived answers are too close for this test to discriminate"
        );
    }

    // --------------------------------------------------------- the complex domain

    /// Why `apply_complex` exists, measured rather than asserted in a comment.
    ///
    /// A wrapped phase field is moved through both paths. Through the complex domain the wraps
    /// survive; interpolating the wrapped values directly averages across every wrap and turns it
    /// into a band of wrong values.
    ///
    /// The shift is deliberately a *half* voxel in each axis, so every output voxel is a genuine
    /// average of its neighbours. A whole-voxel shift makes trilinear interpolation degenerate —
    /// weights of exactly 1 and 0, no averaging — and then there is no interpolation for the
    /// direct path to get wrong, so it scores only 3x worse and the test stops measuring
    /// anything.
    ///
    /// The headline number is **how much of the volume is badly wrong**, not the mean error.
    /// Mean absolute error flatters the direct path badly: this ramp wraps about every seven
    /// voxels, so roughly a seventh of the volume is ruined and six sevenths is fine, and the
    /// mean lands at only ~5x the complex path's. The complex path's own error is not second
    /// order either, because magnitude varies across the interpolation and a magnitude-weighted
    /// circular mean is biased first-order by that — reducing the ramp slope scales both errors
    /// together and does not separate them. What does separate them is that one path's error is
    /// a small smooth bias everywhere and the other's is a radian or more in bands.
    #[test]
    fn wrapped_phase_survives_the_complex_path_and_not_the_direct_one() {
        let dims = (40, 40, 36);
        let affine = identity_affine((1.0, 1.0, 1.0));
        let mag = phantom(dims);
        let pha = wrapped_ramp(dims);
        assert!(
            pha.iter().any(|&v| v > 2.8) && pha.iter().any(|&v| v < -2.8),
            "the test field has to actually wrap, or neither path can fail"
        );

        let w = truth([0.0, 0.0, 1.0], 0.0, [2.5, 1.5, 0.5], dims, &affine);
        let (moved_mag, moved_pha) = warp_complex(&mag, &pha, dims, &affine, &w);

        let mags = vec![mag.clone(), moved_mag.clone()];
        let series = uniform_series(&mags, dims, &affine);
        let motion = estimate_motion(&series, None, &MotionParams::default()).unwrap();
        assert!(
            translation_error_mm(&w, &motion.volumes[1].transform.matrix) < 0.5,
            "the shift has to be recovered before the two paths can be compared fairly"
        );

        let (_, complex_path) = motion.apply_complex(1, &moved_mag, &moved_pha).unwrap();
        let direct_path = motion.apply(1, &moved_pha).unwrap();

        // Score only where the moved volume reached and the phantom has signal, so the
        // shifted-in edge and the empty background count against neither path.
        let cover = motion.coverage(1).unwrap();
        let domain: Vec<u8> = mag
            .iter()
            .zip(cover.iter())
            .map(|(&v, &c)| ((v > 0.2) && c != 0) as u8)
            .collect();
        assert!(
            domain.iter().filter(|&&v| v != 0).count() > 1000,
            "not enough overlap to score"
        );

        let e_complex = mean_abs_phase_error(&complex_path, &pha, &domain);
        let e_direct = mean_abs_phase_error(&direct_path, &pha, &domain);
        let bad_complex = fraction_badly_wrong(&complex_path, &pha, &domain);
        let bad_direct = fraction_badly_wrong(&direct_path, &pha, &domain);
        println!(
            "complex domain: mean {e_complex:.4} rad, {:.2}% of voxels wrong by >1 rad\n\
             direct interp.: mean {e_direct:.4} rad, {:.2}% of voxels wrong by >1 rad",
            100.0 * bad_complex,
            100.0 * bad_direct
        );
        assert!(
            bad_complex < 0.005,
            "the complex path should leave essentially nothing badly wrong, got {:.3}%",
            100.0 * bad_complex
        );
        assert!(
            bad_direct > 0.05,
            "direct interpolation should ruin a measurable share of the volume, got {:.3}%; if it \
             does not, this test is not measuring wrap handling",
            100.0 * bad_direct
        );
        assert!(
            bad_direct > 20.0 * bad_complex.max(1e-4),
            "direct {:.3}% vs complex {:.3}% is not a clear separation",
            100.0 * bad_direct,
            100.0 * bad_complex
        );
        assert!(e_complex < 0.15, "complex-domain mean phase error {e_complex} rad");
    }

    // --------------------------------------------------------- averaging

    /// Averaging four repeats, three of which moved, recovers the reference's own complex signal,
    /// and reports how many repeats reached each voxel.
    #[test]
    fn complex_averaging_recovers_the_reference_signal() {
        let dims = (40, 40, 36);
        let affine = identity_affine((1.0, 1.0, 1.0));
        let mag = phantom(dims);
        let pha = wrapped_ramp(dims);

        let moves = [
            truth([0.0, 0.0, 1.0], 0.0, [1.5, 0.0, 0.0], dims, &affine),
            truth([0.0, 0.0, 1.0], 0.0, [0.0, -1.5, 1.0], dims, &affine),
            truth([0.1, 0.2, 1.0], 2.0, [1.0, 1.0, 0.0], dims, &affine),
        ];
        let mut mags = vec![mag.clone()];
        let mut phas = vec![pha.clone()];
        for w in &moves {
            let (m, p) = warp_complex(&mag, &pha, dims, &affine, w);
            mags.push(m);
            phas.push(p);
        }

        let series = uniform_series(&mags, dims, &affine);
        let motion = estimate_motion(&series, None, &MotionParams::default()).unwrap();
        let avg = motion.average_complex(&mags, &phas).unwrap();

        let full = avg.contributions.iter().filter(|&&c| c == 4).count();
        println!(
            "contributions: {full} of {} voxels see all 4 repeats",
            avg.contributions.len()
        );
        assert!(full > avg.contributions.len() / 2, "most voxels should see every repeat");

        // Score where the phantom has signal and every repeat contributed.
        let domain: Vec<u8> = mag
            .iter()
            .zip(avg.contributions.iter())
            .map(|(&v, &c)| ((v > 0.2) && c == 4) as u8)
            .collect();
        let e_pha = mean_abs_phase_error(&avg.phase, &pha, &domain);
        let (mut e_mag, mut n) = (0.0f64, 0usize);
        for i in 0..mag.len() {
            if domain[i] != 0 {
                e_mag += (avg.magnitude[i] - mag[i]).abs();
                n += 1;
            }
        }
        e_mag /= n as f64;
        println!("average vs reference: mean |phase| {e_pha:.4} rad, mean |magnitude| {e_mag:.4}");
        assert!(e_pha < 0.1, "averaged phase error {e_pha} rad");
        assert!(e_mag < 0.1, "averaged magnitude error {e_mag}");

        // Three of the four repeats only translated and the fourth rotated 2 degrees, so the
        // spread of dipole-kernel orientations the average was taken over is small — which is the
        // condition under which averaging phase is legitimate at all.
        println!("rotation spread {:.3} deg", avg.rotation_spread_deg);
        assert!(
            avg.rotation_spread_deg < 3.0,
            "rotation spread {} deg is larger than this series was built with",
            avg.rotation_spread_deg
        );

        assert!(motion.average_complex(&mags[..2], &phas).is_none(), "length mismatch");
    }

    /// `rotation_spread_deg` is the angle between the extreme orientations, not the largest
    /// rotation from the reference — and a series that only translated has none of it, however
    /// far it translated.
    #[test]
    fn rotation_spread_is_between_the_extremes_not_from_the_reference() {
        let dims = (40, 40, 36);
        let affine = identity_affine((1.0, 1.0, 1.0));
        let mag = phantom(dims);
        let zero = vec![0.0f64; mag.len()];

        // Opposite rotations about one axis: each is 6 degrees from the reference, so the spread
        // across the series is 12.
        let moves = [
            truth([1.0, 0.0, 0.0], 6.0, [0.0; 3], dims, &affine),
            truth([1.0, 0.0, 0.0], -6.0, [0.0; 3], dims, &affine),
        ];
        let mut mags = vec![mag.clone()];
        for w in &moves {
            mags.push(warp_complex(&mag, &zero, dims, &affine, w).0);
        }
        let series = uniform_series(&mags, dims, &affine);
        let motion = estimate_motion(&series, None, &MotionParams::default()).unwrap();
        let spread = motion.rotation_spread_deg();
        println!(
            "rotations from the reference: {:.3}, {:.3}; spread {spread:.3}",
            motion.volumes[1].rotation_deg, motion.volumes[2].rotation_deg
        );
        assert!(
            (spread - 12.0).abs() < 1.0,
            "spread {spread} should be the 12 degrees between the extremes, not the 6 from the \
             reference"
        );

        // Pure translation: the head never rotated, so every bdir is the same and the spread is
        // zero no matter how large the displacement.
        let t = truth([0.0, 0.0, 1.0], 0.0, [2.0, -2.0, 1.5], dims, &affine);
        let pair = vec![mag.clone(), warp_complex(&mag, &zero, dims, &affine, &t).0];
        let series = uniform_series(&pair, dims, &affine);
        let motion = estimate_motion(&series, None, &MotionParams::default()).unwrap();
        println!(
            "pure translation: {:.3} mm displacement, spread {:.4} deg",
            motion.max_displacement_mm(),
            motion.rotation_spread_deg()
        );
        assert!(motion.max_displacement_mm() > 2.5, "the volume did move");
        assert!(
            motion.rotation_spread_deg() < 0.3,
            "a pure translation must not produce a rotation spread, got {}",
            motion.rotation_spread_deg()
        );
    }

    // --------------------------------------------------------- displacement

    /// `max_displacement_mm` is a distance in mm, checked against values worked out
    /// independently: `|t|` for a pure translation, and the chord `2·r·sin(θ/2)` at the farthest
    /// in-domain voxel for a pure rotation about the grid centre.
    ///
    /// Both domains are exercised: the translation runs with no mask (the eight grid corners) and
    /// the rotation with one (every mask voxel).
    #[test]
    fn displacement_is_the_distance_tissue_actually_moved() {
        let dims = (40, 40, 40);
        let vs = (1.0, 1.0, 1.0);
        let affine = identity_affine(vs);
        let mag = phantom(dims);
        let zero = vec![0.0f64; mag.len()];

        // Pure translation: every voxel moves by the same vector, so the maximum is its length
        // whatever the domain.
        let shift = [2.0, -2.5, 0.0];
        let t = truth([0.0, 0.0, 1.0], 0.0, shift, dims, &affine);
        let pair = vec![mag.clone(), warp_complex(&mag, &zero, dims, &affine, &t).0];
        let series = uniform_series(&pair, dims, &affine);
        let m = estimate_motion(&series, None, &MotionParams::default()).unwrap();
        let want = (shift[0] * shift[0] + shift[1] * shift[1]).sqrt();
        println!(
            "pure translation: displacement {:.4} mm, |t| = {want:.4} mm",
            m.volumes[1].max_displacement_mm
        );
        assert!(
            (m.volumes[1].max_displacement_mm - want).abs() < 0.3,
            "displacement {} should be |t| = {want}",
            m.volumes[1].max_displacement_mm
        );

        // Pure rotation about +z through the grid's world centre, with a mask as the domain. The
        // farthest mask voxel from the axis sets the answer, so find that radius from the mask
        // directly and then the chord.
        let deg = 8.0;
        let mask = phantom_mask(dims);
        let rot = truth([0.0, 0.0, 1.0], deg, [0.0; 3], dims, &affine);
        let pair = vec![mag.clone(), warp_complex(&mag, &zero, dims, &affine, &rot).0];
        let series = uniform_series(&pair, dims, &affine);
        let m = estimate_motion(&series, Some(&mask), &MotionParams::default()).unwrap();
        assert!(
            rotation_error_deg(&rot, &m.volumes[1].transform.matrix) < 1.0,
            "the rotation has to be recovered before its displacement means anything"
        );

        let (nx, ny, nz) = dims;
        let ctr = [(nx as f64 - 1.0) * 0.5 * vs.0, (ny as f64 - 1.0) * 0.5 * vs.1];
        let mut r_max = 0.0f64;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    if mask[i + j * nx + k * nx * ny] != 0 {
                        let (dx, dy) = (i as f64 * vs.0 - ctr[0], j as f64 * vs.1 - ctr[1]);
                        r_max = r_max.max((dx * dx + dy * dy).sqrt());
                    }
                }
            }
        }
        let want = 2.0 * r_max * (deg.to_radians() / 2.0).sin();
        println!(
            "pure rotation: displacement {:.4} mm, chord at r={r_max:.2} mm is {want:.4} mm",
            m.volumes[1].max_displacement_mm
        );
        assert!(
            (m.volumes[1].max_displacement_mm - want).abs() < 0.4,
            "displacement {} should be the chord {want} at the farthest mask voxel",
            m.volumes[1].max_displacement_mm
        );
        // And it is not the rotation angle, or the translation, in disguise.
        assert!(
            m.volumes[1].max_displacement_mm > 2.0,
            "an 8-degree rotation of this volume moves tissue millimetres, got {}",
            m.volumes[1].max_displacement_mm
        );
        assert!(
            m.volumes[1].translation_mm < 1.0,
            "this transform has no world translation to speak of, got {}",
            m.volumes[1].translation_mm
        );
    }

    // --------------------------------------------------------- chaining and bookkeeping

    /// A chained seed with the shallower pyramid it licenses costs fewer metric evaluations than
    /// a cold start, without losing accuracy. See `MotionParams::chain_levels` for why the seed
    /// alone does not.
    #[test]
    fn chaining_costs_less_than_starting_cold_each_time() {
        let dims = (40, 40, 36);
        let affine = identity_affine((1.0, 1.0, 1.0));
        let mag = phantom(dims);
        let zero = vec![0.0f64; mag.len()];

        // A steady drift, which is what chaining is for.
        let moves: Vec<[f64; 16]> = (1..5)
            .map(|s| {
                truth(
                    [0.2, 0.1, 1.0],
                    2.0 * s as f64,
                    [0.4 * s as f64, -0.3 * s as f64, 0.2 * s as f64],
                    dims,
                    &affine,
                )
            })
            .collect();
        let mut mags = vec![mag.clone()];
        for w in &moves {
            mags.push(warp_complex(&mag, &zero, dims, &affine, w).0);
        }
        let series = uniform_series(&mags, dims, &affine);

        // `chain_levels: 1` rather than the default 3, because this volume is small enough that
        // the pyramid only reaches two levels anyway (`registration` stops before any axis drops
        // below 16), so the default would be clamped to the full depth and there would be
        // nothing to measure. The real-data numbers for the default are in `chain_levels`' docs.
        let chained = estimate_motion(
            &series,
            None,
            &MotionParams {
                chain: true,
                chain_levels: 1,
                ..MotionParams::default()
            },
        )
        .unwrap();
        let cold = estimate_motion(
            &series,
            None,
            &MotionParams {
                chain: false,
                ..MotionParams::default()
            },
        )
        .unwrap();

        let evals =
            |m: &MotionSeries| -> usize { m.volumes.iter().map(|v| v.transform.evaluations).sum() };
        println!("evaluations: chained {}, cold {}", evals(&chained), evals(&cold));
        assert!(
            evals(&chained) < evals(&cold),
            "chaining should cost fewer evaluations: {} vs {}",
            evals(&chained),
            evals(&cold)
        );

        // And it is not cheaper by being wrong: both agree with the independent ground truth.
        for (t, w) in moves.iter().enumerate() {
            for (label, m) in [("chained", &chained), ("cold", &cold)] {
                let err = rotation_error_deg(w, &m.volumes[t + 1].transform.matrix);
                println!("{label} volume {}: rotation error {err:.4} deg", t + 1);
                assert!(err < 1.0, "{label} volume {}: rotation off by {err} deg", t + 1);
            }
        }
    }

    /// A mid-series reference: indices still line up with the caller's data, the reference's own
    /// entry is the identity, and volumes on both sides of it are solved.
    #[test]
    fn a_mid_series_reference_solves_both_directions() {
        let dims = (40, 40, 36);
        let affine = identity_affine((1.0, 1.0, 1.0));
        let mag = phantom(dims);
        let zero = vec![0.0f64; mag.len()];

        let moves = [
            truth([0.0, 1.0, 0.2], -5.0, [-1.0, 0.5, 0.0], dims, &affine),
            truth([0.0, 1.0, 0.2], 5.0, [1.0, -0.5, 0.0], dims, &affine),
        ];
        // Reference in the middle: [moved, reference, moved].
        let mags = vec![
            warp_complex(&mag, &zero, dims, &affine, &moves[0]).0,
            mag.clone(),
            warp_complex(&mag, &zero, dims, &affine, &moves[1]).0,
        ];
        let series = uniform_series(&mags, dims, &affine);
        let motion = estimate_motion(
            &series,
            None,
            &MotionParams {
                reference: MotionReference::Index(1),
                ..MotionParams::default()
            },
        )
        .unwrap();

        assert_eq!(motion.reference, 1);
        assert_eq!(motion.volumes[1].rotation_deg, 0.0, "the reference is the identity");
        for (t, w) in [(0usize, &moves[0]), (2usize, &moves[1])] {
            let err = rotation_error_deg(w, &motion.volumes[t].transform.matrix);
            println!("volume {t}: rotation error {err:.4} deg, NCC {:.4}", motion.volumes[t].ncc);
            assert!(err < 1.0, "volume {t}: rotation off by {err} deg");
        }
        assert!(
            motion.volumes[0].rotation_deg > 4.0 && motion.volumes[2].rotation_deg > 4.0,
            "both flanking volumes should have been found to have moved"
        );
    }

    /// `correct_motion` is the composition of the pieces, and its coverage is the intersection of
    /// what every volume reaches — strictly smaller than the grid once anything has moved off the
    /// edge.
    #[test]
    fn correct_motion_composes_and_reports_coverage() {
        let dims = (40, 40, 36);
        let affine = identity_affine((1.0, 1.0, 1.0));
        let mag = phantom(dims);
        let pha = wrapped_ramp(dims);

        let w = truth([0.1, 0.0, 1.0], 5.0, [2.0, -1.5, 1.0], dims, &affine);
        let (m1, p1) = warp_complex(&mag, &pha, dims, &affine, &w);
        let mags = vec![mag.clone(), m1];
        let phas = vec![pha.clone(), p1];

        let out = correct_motion(&mags, &phas, dims, &affine, None, &MotionParams::default())
            .expect("correction runs");
        assert_eq!(out.magnitudes.len(), 2);
        assert_eq!(out.phases.len(), 2);
        assert_eq!(out.bdirs.len(), 2);
        assert_eq!(out.coverage.len(), mag.len());
        assert_eq!(out.bdirs[0], out.motion.volumes[0].bdir);

        // The reference is resampled through the identity, which is exact rather than merely
        // close: it is the one volume that never passes through an interpolator.
        for (i, (&got, &want)) in out.magnitudes[0].iter().zip(mag.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-9,
                "the reference must come back untouched, differs at {i}"
            );
        }

        // Rotating and shifting takes a slab off the edge of the grid, so the intersection
        // cannot be the whole volume.
        let covered = out.coverage.iter().filter(|&&c| c != 0).count();
        println!("coverage {covered} of {} reference voxels", out.coverage.len());
        assert!(covered < out.coverage.len(), "a moved volume cannot cover the whole grid");
        assert!(covered > out.coverage.len() / 2, "and it should still cover most of it");

        // The corrected moved volume agrees with the reference where both have data.
        let domain: Vec<u8> = mag
            .iter()
            .zip(out.coverage.iter())
            .map(|(&v, &c)| ((v > 0.2) && c != 0) as u8)
            .collect();
        let err = mean_abs_phase_error(&out.phases[1], &pha, &domain);
        println!("corrected phase error {err:.4} rad");
        assert!(err < 0.2, "corrected phase error {err} rad");

        assert!(
            correct_motion(&mags, &phas[..1], dims, &affine, None, &MotionParams::default())
                .is_none(),
            "magnitude and phase counts must agree"
        );
    }

    /// `implied_b0` is the convention `declared_b0` has to match, including on the anisotropic
    /// voxels where reading it off the affine naively goes wrong.
    #[test]
    fn implied_b0_matches_the_affine_convention() {
        let d = implied_b0(&identity_affine((0.8, 0.8, 3.0)));
        assert!(
            d.0.abs() < 1e-12 && d.1.abs() < 1e-12 && (d.2 - 1.0).abs() < 1e-12,
            "axial acquisition should give voxel +z, got {d:?}"
        );
    }
}

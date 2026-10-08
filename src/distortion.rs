//! EPI susceptibility distortion: **applying** a known B0 field map, not estimating one.
//!
//! An echo-planar readout traverses k-space along the phase-encode (PE) axis so slowly that the
//! effective bandwidth per pixel along PE is tens of Hz. An off-resonance of a few hundred Hz —
//! routine near the sinuses and the petrous bone, which is exactly where susceptibility is
//! interesting — therefore displaces signal by several voxels along PE. Nothing displaces it
//! along the readout or slice axes to any comparable degree, so the whole correction is a
//! **one-dimensional, per-voxel resample along a single voxel axis**.
//!
//! # Scope: this module applies a field map, it does not estimate one
//!
//! BIDS formalises four B0-fieldmap cases (phase-difference, two phase maps, direct field
//! mapping, and reversed-PE "pepolar"), and the existing ecosystem — FSL `topup` and `fugue`,
//! SDCFlows, DR-BUDDI — already *estimates* the field well from any of them. What is missing,
//! and what a QSM library is uniquely placed to own, is the application step done correctly for
//! **complex** data. So the input here is a field map that already exists, and the same entry
//! point serves all four BIDS cases: whatever produced the field, hand it in.
//!
//! Estimating the field from a reversed-PE pair (what `topup` does) is a substantial nonlinear
//! optimisation and is deliberately **not** here.
//!
//! # The displacement relation
//!
//! ```text
//! shift [voxels along PE] = ΔB0 [Hz] × TotalReadoutTime [s] × polarity
//! ```
//!
//! This is FSL's convention and BIDS's. It looks like it ought to need the PE matrix size, and
//! the underlying physics does: a voxel off-resonant by `Δf` moves by `Δf × ESP_eff × N_PE`,
//! where `ESP_eff` is the effective echo spacing. BIDS defines `TotalReadoutTime` as
//! `EffectiveEchoSpacing × (ReconMatrixPE − 1)`, which differs from `ESP × N_PE` by a factor
//! `N/(N−1)`. The spec resolves the discrepancy by *definition* rather than by algebra —
//! `TotalReadoutTime` is
//!
//! > the readout duration […] that would have generated data with the given level of distortion.
//! > It is NOT the actual, physical duration of the readout train.
//!
//! — so `shift = Δf × TRT` is exact by construction, and no matrix size is needed. See
//! [`DisplacementField::from_field_hz`].
//!
//! Both candidate scales really do appear in real sidecars, and they really do differ. In
//! `bids-examples/2d_mb_pcasl/sub-1/fmap/sub-1_dir-AP_epi.json`, `EffectiveEchoSpacing × (N−1)`
//! reproduces the stated `TotalReadoutTime` of 0.0484496 s exactly, while
//! `EffectiveEchoSpacing × N` reproduces `1 / BandwidthPerPixelPhaseEncode` exactly, at
//! 0.0490196 s. That is a genuine 1.18% gap, not a rounding artefact. The second is the physical
//! scale; the first is what BIDS, FSL `topup`/`applytopup` and SDCFlows all use, so it is what
//! this uses. It is not an off-by-one.
//!
//! ## Sign
//!
//! With `PhaseEncodingDirection = "j"`, positive off-resonance displaces signal toward
//! *increasing* `j`; `"j-"` reverses it. If your fieldmap uses the opposite off-resonance sign
//! convention, negate the field — the symptom of getting it backwards is unmistakable, because
//! the output is distorted about twice as badly as the input rather than less.
//!
//! # Three things that are easy to get wrong
//!
//! **Phase must move in the complex domain.** Halfway between `+3.0` and `−3.0` rad a linear
//! interpolator returns `0.0` when the answer is near `±π`. [`unwarp_complex`] interpolates
//! `mag·e^{iφ}`, which is the same discipline [`crate::geometry`] applies. Never route wrapped
//! phase through [`unwarp_volume`].
//!
//! With an unwrapper in the pipeline there are exactly two correct orders, and mixing them is the
//! live mistake, because "unwarp, then unwrap" reads as the obvious one:
//!
//! 1. **Unwarp the complex signal, then unwrap** — [`unwarp_complex`] on magnitude and wrapped
//!    phase together, and the unwrapper sees undistorted input.
//! 2. **Unwrap first, then unwarp the unwrapped phase** — [`unwarp_volume`], which is safe here
//!    precisely because unwrapped phase is continuous.
//!
//! What is wrong is applying a resample to a *wrapped* phase image on its own, in either order.
//!
//! **Pile-up is unrecoverable.** Where the distortion makes the PE mapping non-monotonic, signal
//! from several true locations has already been summed into one voxel by the scanner, and no
//! unwarping separates it again. [`unwarp_complex`] returns [`PileUp`] rather than emitting a
//! plausible-looking wrong answer; see that type for why it refuses instead of warning.
//!
//! **Intensity needs the Jacobian.** Unwarping stretches and compresses voxels, so restoring
//! geometry without restoring amplitude leaves the magnitude wrong by the local stretch factor.
//! On by default here, unlike the functional-imaging convention — see
//! [`UnwarpParams::jacobian_modulation`] for why QSM cannot afford to skip it.
//!
//! # The output is on the acquisition's own uniform grid
//!
//! Worth stating, because the Jacobian is precisely the claim that the *true* object was sampled
//! non-uniformly along PE. Unwarping resamples back onto the regular voxel grid the data arrived
//! on: same dimensions, same spacing, same affine, and there is no output-grid parameter to get
//! wrong. Every FFT-based stage downstream depends on that. [`crate::kernels`]'s spherical-mean
//! kernels and the dipole kernel are built in millimetres from a [`crate::Grid`] and transformed,
//! so they assume uniform spacing; a correction that left the data non-uniformly sampled would
//! quietly invalidate all of them rather than fail.
//!
//! # Where it goes in a pipeline
//!
//! ```text
//! unwarp  →  co-register / motion-correct  →  field mapping  →  background removal  →  inversion
//! ```
//!
//! Unwarp is first so that the field map QSM ultimately inverts is computed from *undistorted*
//! phase. It must also precede registration, for a reason worth stating precisely.
//!
//! The displacement is a vector field, `s(x)·ê_PE`. It is tempting to say the whole thing is
//! fixed in the scanner frame, but that is too strong: the off-resonance `s` is mostly generated
//! by the head's own susceptibility, so to first order it travels with the head. What is
//! unarguably scanner-fixed is **`ê_PE`, the axis the displacement runs along**. Rotate the head
//! and that axis points somewhere else in head coordinates, so the same anatomy is smeared in a
//! different direction — a difference between the two volumes that no rigid transform can
//! absorb, because it is not rigid.
//!
//! How much depends entirely on the regime, and the two differ by an order of magnitude:
//!
//! - **Inter-echo or inter-run motion**, a few degrees. A 10-voxel frontal displacement under a
//!   3° rotation moves `10·sin 3° ≈ 0.5` voxels perpendicular to itself, 0.9 at 5°. Enough to
//!   break the rigid model, small enough that the residual is a fraction of a voxel.
//! - **Multi-orientation acquisition** (COSMOS, STI), ±25° by design. The same arithmetic gives
//!   `10·sin 25° ≈ 4.2` voxels. Not a correction-grade residual — a different image.
//!
//! And in the second regime a mechanism the first can ignore comes into play: the induced field
//! `s` is generated by the head's susceptibility *through a B0-oriented dipole kernel*, so
//! rotating the head relative to B0 changes the field's **pattern**, not only the direction it
//! is smeared along. That orientation dependence is precisely what makes COSMOS work. So for
//! multi-orientation data, unwarping each orientation before co-registration is not a refinement
//! — it is the difference between fitting a rigid transform and fitting one to images that
//! genuinely differ non-rigidly.
//!
//! So fitting [`crate::registration`]'s 6-DOF model to not-yet-unwarped volumes converges to
//! something, and that something is wrong. Unwarp each volume in its own acquired geometry first
//! and what remains between them really is rigid.
//!
//! Hand the shrunk [`UnwarpedComplex::mask`] to whatever registers next, not the original: a
//! registration metric evaluated over voxels that have no data is being asked to match noise.
//!
//! ## What this ordering still does not fix
//!
//! A field map is measured at **one head position**. Applying it to a volume acquired after the
//! head moved applies an estimate that is stale in the head frame — the anatomy generating the
//! field has moved relative to the grid the field is sampled on. Resolving that properly means
//! iterating unwarping and motion correction against each other, which this does not do. The
//! error is small while the motion is small, and a motion-correction stage that reports the
//! largest distance tissue actually moved (in mm, comparable against the voxel size) is what
//! tells you whether it is. Recorded here rather than left for someone to re-file as a gap.
//!
//! # Using the pipeline's own field map
//!
//! A QSM pipeline produces a ΔB0 map by construction, which makes it tempting to skip the
//! fieldmap acquisition entirely. The catch is that such a field is sampled in **distorted**
//! space, so `shift(x) = k·Δf(x)` is no longer an equation but a fixed point, and it degrades
//! worst exactly where distortion is worst. SDCFlows treats even registration-based
//! fieldmap-less estimation as a last resort, skipped whenever any real fieldmap exists, and the
//! same posture applies here. It is available — [`DisplacementField::from_self_measured_hz`]
//! solves the fixed point and reports whether it converged — but it is a separately named
//! constructor precisely so it can never be the silent path.

use crate::geometry::trilinear_sample;

/// Proton gyromagnetic ratio in Hz/T, matching [`crate::pipeline::hz_to_ppm`].
const GAMMA_HZ_PER_T: f64 = 42.576e6;

// ------------------------------------------------------------------ phase encoding

/// A BIDS `PhaseEncodingDirection`: which voxel axis the phase encode runs along, and which way.
///
/// The BIDS letters `i`, `j`, `k` name the first, second and third axis *of the NIfTI array*, so
/// they are array axes, not anatomical ones — `j` on an axial acquisition and `j` on a sagittal
/// one point in different anatomical directions, and neither this type nor the correction cares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhaseEncoding {
    /// Voxel axis the phase encode runs along: 0 = `i`, 1 = `j`, 2 = `k`.
    pub axis: usize,
    /// `true` for `i`/`j`/`k` (index increasing), `false` for the `-` spellings.
    pub positive: bool,
}

impl PhaseEncoding {
    /// Parse a BIDS `PhaseEncodingDirection` string: `i`, `i-`, `j`, `j-`, `k` or `k-`.
    ///
    /// Surrounding whitespace is tolerated; nothing else is. In particular the DICOM
    /// `InPlanePhaseEncodingDirection` values `ROW` and `COL` are **not** accepted, because they
    /// do not carry polarity and BIDS explicitly distinguishes them.
    pub fn parse(s: &str) -> Result<Self, PhaseEncodingParseError> {
        let t = s.trim();
        let (letter, positive) = match t.strip_suffix('-') {
            Some(rest) => (rest, false),
            None => (t, true),
        };
        let axis = match letter {
            "i" => 0,
            "j" => 1,
            "k" => 2,
            _ => return Err(PhaseEncodingParseError { got: s.to_string() }),
        };
        // "i+", "-j", "j--" and the rest never reach here: the match above sees the whole
        // string minus at most one trailing '-', and only the three bare letters pass it.
        Ok(Self { axis, positive })
    }

    /// The BIDS spelling this came from.
    pub fn to_bids(self) -> &'static str {
        match (self.axis, self.positive) {
            (0, true) => "i",
            (0, false) => "i-",
            (1, true) => "j",
            (1, false) => "j-",
            (2, true) => "k",
            _ => "k-",
        }
    }

    /// `+1.0` for a positive-polarity encode, `-1.0` otherwise.
    pub fn sign(self) -> f64 {
        if self.positive { 1.0 } else { -1.0 }
    }
}

/// `PhaseEncodingDirection` was not one of the six BIDS spellings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhaseEncodingParseError {
    /// The string that was rejected.
    pub got: String,
}

impl std::fmt::Display for PhaseEncodingParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PhaseEncodingDirection must be one of i, i-, j, j-, k, k- (got {:?})",
            self.got
        )
    }
}

impl std::error::Error for PhaseEncodingParseError {}

/// A BIDS sidecar field that could not be used. See [`DisplacementField::from_bids_sidecar`].
#[derive(Clone, Debug, PartialEq)]
pub enum SidecarError {
    /// `PhaseEncodingDirection` was not one of the six BIDS spellings.
    PhaseEncodingDirection(PhaseEncodingParseError),
    /// `TotalReadoutTime` was not a finite, strictly positive number of seconds.
    TotalReadoutTime {
        /// The value that was rejected.
        got: f64,
    },
}

impl From<PhaseEncodingParseError> for SidecarError {
    fn from(e: PhaseEncodingParseError) -> Self {
        Self::PhaseEncodingDirection(e)
    }
}

impl std::fmt::Display for SidecarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PhaseEncodingDirection(e) => write!(f, "{e}"),
            Self::TotalReadoutTime { got } => write!(
                f,
                "TotalReadoutTime must be a finite, strictly positive number of seconds (got {got})"
            ),
        }
    }
}

impl std::error::Error for SidecarError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::PhaseEncodingDirection(e) => Some(e),
            Self::TotalReadoutTime { .. } => None,
        }
    }
}

// ------------------------------------------------------------------ pile-up

/// The distortion folded signal from several locations into one voxel, so it cannot be undone.
///
/// Unwarping is a resample of a map `x' = x + s(x)`. That is invertible only while the map is
/// monotonic along PE, i.e. while its derivative `J = 1 + ds/dx` stays positive. Where `J ≤ 0`
/// the scanner has already summed signal from two or more true locations into a single acquired
/// voxel — the information is gone from the data, not merely hard to extract, and no choice of
/// interpolation recovers it.
///
/// # Panic or `Result`: whose mistake is it
///
/// The crate now splits these deliberately, and this type is on one side of it. A `panic!`
/// reports a **caller** error — echoes of mismatched length, an axis index out of range — where
/// the audience is a developer holding a backtrace and the fix is in their code. A `Result`
/// reports that the **acquisition** cannot yield a correct answer however well it is called,
/// where the audience is whoever scanned it. `PileUp` and `SliceGapError` are the second kind;
/// the dimension guards added in QSM.rs#135 are the first.
///
/// The borderline, named because a reviewer will ask: a value that arrives from a **sidecar**
/// rather than from code is the acquisition's, even when it is structurally wrong rather than
/// physically impossible. A `TotalReadoutTime` of `NaN` is a broken file or a mis-parse, and a
/// host that hit one should be told its data is unusable, not handed a panic — see
/// [`DisplacementField::from_bids_sidecar`], which is the entry point for values that came out
/// of JSON. [`DisplacementField::from_field_hz`] keeps the `assert!`, because by then the caller
/// is asserting it has already validated them.
///
/// This is returned as an error rather than reported as a warning for the same reason the 2D
/// multi-slice work refuses a gapped acquisition (`Grid::require_contiguous_slices`, QSM.rs#75):
/// resampling through a folded region produces an image that is wrong while still looking
/// entirely plausible. A stretched-looking but *smooth* frontal lobe invites the reader to trust
/// it.
///
/// What to do about it: the acquisition needs less distortion, not better post-processing —
/// a shorter effective readout (higher parallel-imaging factor, smaller PE matrix, increased
/// bandwidth), or a reversed-PE pair so a blip-up/blip-down method can combine the two halves
/// that each survived. If the folding is confined to tissue the study does not care about,
/// restricting the mask handed to [`unwarp_complex`] is legitimate and is the supported escape.
#[derive(Clone, Debug, PartialEq)]
pub struct PileUp {
    /// How many voxels inside the mask had `J ≤ threshold`.
    pub n_voxels: usize,
    /// The most negative (or smallest) Jacobian found inside the mask.
    pub min_jacobian: f64,
    /// Voxel coordinate where `min_jacobian` occurred.
    pub worst_voxel: (usize, usize, usize),
    /// The threshold that was applied — [`UnwarpParams::min_jacobian`].
    pub threshold: f64,
    /// Total voxels considered (the mask population).
    pub n_considered: usize,
}

impl std::fmt::Display for PileUp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (x, y, z) = self.worst_voxel;
        write!(
            f,
            "EPI signal pile-up: {} of {} masked voxels have phase-encode Jacobian <= {} \
             (minimum {:.4} at voxel ({}, {}, {})). The phase-encode mapping is not monotonic \
             there, so signal from several locations was summed into one voxel during \
             acquisition and no unwarping can separate it. Reduce the effective readout time, \
             acquire a reversed-phase-encode pair, or restrict the mask to tissue that is not \
             folded.",
            self.n_voxels, self.n_considered, self.threshold, self.min_jacobian, x, y, z
        )
    }
}

impl std::error::Error for PileUp {}

/// What the pile-up check actually looked at — the difference between "nothing folded" and
/// "nothing was checked", which `Ok` alone cannot express.
///
/// Returned on success so the distinction is machine-checkable rather than a documented caveat.
/// A caller that wants the check to mean something should assert
/// [`voxels_examined`](Self::voxels_examined) is the population it expected; a failed brain
/// extraction shows up here as zero while every other signal reads clean. The field name
/// matches `EchoQuality::voxels_examined` in the multi-echo work, which has the same hazard.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JacobianCheck {
    /// Voxels the mask selected, and so the number the check actually inspected.
    pub voxels_examined: usize,
    /// Smallest Jacobian among them, or `None` when there were none to examine.
    pub min_jacobian: Option<f64>,
}

// ------------------------------------------------------------------ displacement field

/// A per-voxel displacement along one voxel axis, in voxels — the whole geometry of an EPI
/// distortion.
///
/// Positive values displace toward increasing index along [`PhaseEncoding::axis`]; the polarity
/// of the encode is already folded in by the constructors, so `shift` is the signed displacement
/// in array terms and nothing downstream has to consult the polarity again.
///
/// **The displacement is defined in undistorted space.** `shift[x]` is where the signal that
/// *belongs* at `x` ended up, so unwarping is the gather `true(x) = distorted(x + shift[x])`.
/// Every constructor but [`from_self_measured_hz`](Self::from_self_measured_hz) assumes the
/// field map it is given was measured on an undistorted acquisition, which is true of a
/// GRE phase-difference fieldmap, a direct (Case 3) fieldmap, and `topup`'s output — but not of
/// a field map computed from the distorted EPI itself.
#[derive(Clone, Debug)]
pub struct DisplacementField {
    /// Signed displacement in voxels, column-major, one per voxel.
    pub shift: Vec<f64>,
    /// Volume dimensions the displacement is defined on.
    pub dims: (usize, usize, usize),
    /// Which axis it displaces along, and the encode polarity it came from.
    pub pe: PhaseEncoding,
}

/// How the [`from_self_measured_hz`](DisplacementField::from_self_measured_hz) fixed point went.
#[derive(Clone, Copy, Debug)]
pub struct SelfFieldConvergence {
    /// Iterations actually run.
    pub iterations: usize,
    /// Largest change in any voxel's displacement on the final iteration, in voxels.
    pub max_change_voxels: f64,
    /// Whether `max_change_voxels` fell below the requested tolerance before the iteration cap.
    pub converged: bool,
}

impl DisplacementField {
    /// Build from a field map in **Hz** and a BIDS `TotalReadoutTime` in **seconds**.
    ///
    /// `shift = field_hz × total_readout_time × polarity`, which is exact by BIDS's definition of
    /// `TotalReadoutTime` (see the module docs). The field must be sampled in undistorted space.
    ///
    /// # Panics
    /// If `field_hz.len()` does not match `dims`, or `total_readout_time` is not finite.
    pub fn from_field_hz(
        field_hz: &[f64],
        dims: (usize, usize, usize),
        pe: PhaseEncoding,
        total_readout_time: f64,
    ) -> Self {
        assert_eq!(
            field_hz.len(),
            dims.0 * dims.1 * dims.2,
            "field map length does not match dimensions"
        );
        assert!(
            total_readout_time.is_finite(),
            "TotalReadoutTime must be finite (got {total_readout_time})"
        );
        let k = total_readout_time * pe.sign();
        Self { shift: field_hz.iter().map(|&f| f * k).collect(), dims, pe }
    }

    /// Build from the two BIDS sidecar fields directly, validating both the same way.
    ///
    /// This exists because [`from_field_hz`](Self::from_field_hz) `assert!`s on a non-finite
    /// `TotalReadoutTime` while [`PhaseEncoding::parse`] returns a `Result` — two fields of the
    /// same JSON file handled by opposite conventions, which was simply an inconsistency. Values
    /// that came out of a sidecar belong on the `Result` side (see [`PileUp`] for the rule), so
    /// route them through here and keep `from_field_hz` for values the caller has already
    /// checked.
    ///
    /// `total_readout_time` must be finite and strictly positive: a readout takes time, and a
    /// zero or negative one is a broken sidecar rather than an undistorted acquisition. If you
    /// genuinely want no displacement, say so with
    /// [`from_voxel_shift`](Self::from_voxel_shift).
    ///
    /// # Panics
    /// If `field_hz.len()` does not match `dims` — that one really is the caller's.
    pub fn from_bids_sidecar(
        field_hz: &[f64],
        dims: (usize, usize, usize),
        phase_encoding_direction: &str,
        total_readout_time: f64,
    ) -> Result<Self, SidecarError> {
        let pe = PhaseEncoding::parse(phase_encoding_direction)?;
        if !total_readout_time.is_finite() || total_readout_time <= 0.0 {
            return Err(SidecarError::TotalReadoutTime { got: total_readout_time });
        }
        Ok(Self::from_field_hz(field_hz, dims, pe, total_readout_time))
    }

    /// Build from a field map in **rad/s** — the `Units` a BIDS Case-3 direct fieldmap most often
    /// declares.
    pub fn from_field_rad_per_s(
        field_rad_per_s: &[f64],
        dims: (usize, usize, usize),
        pe: PhaseEncoding,
        total_readout_time: f64,
    ) -> Self {
        let hz: Vec<f64> =
            field_rad_per_s.iter().map(|&v| v / (2.0 * std::f64::consts::PI)).collect();
        Self::from_field_hz(&hz, dims, pe, total_readout_time)
    }

    /// Build from a field map in **ppm** — the unit every other stage of this crate works in.
    ///
    /// `b0_tesla` is the scanner field strength, needed to turn a relative shift into Hz.
    pub fn from_field_ppm(
        field_ppm: &[f64],
        dims: (usize, usize, usize),
        pe: PhaseEncoding,
        total_readout_time: f64,
        b0_tesla: f64,
    ) -> Self {
        let scale = GAMMA_HZ_PER_T * b0_tesla * 1e-6;
        let hz: Vec<f64> = field_ppm.iter().map(|&v| v * scale).collect();
        Self::from_field_hz(&hz, dims, pe, total_readout_time)
    }

    /// Wrap an already-computed voxel-displacement map — `fugue --saveshift`, `topup`'s
    /// `--dfout`, or anything else that has already done the Hz × time arithmetic.
    ///
    /// The polarity is **not** applied: a shift map from an external tool is assumed to be signed
    /// already. `pe.positive` is still carried so [`PhaseEncoding::to_bids`] reports the
    /// acquisition faithfully.
    ///
    /// # Panics
    /// If `shift.len()` does not match `dims`.
    pub fn from_voxel_shift(
        shift: Vec<f64>,
        dims: (usize, usize, usize),
        pe: PhaseEncoding,
    ) -> Self {
        assert_eq!(
            shift.len(),
            dims.0 * dims.1 * dims.2,
            "shift map length does not match dimensions"
        );
        Self { shift, dims, pe }
    }

    /// Build from a field map that was measured **on the distorted EPI itself** — the pipeline's
    /// own ΔB0 rather than a separately acquired fieldmap.
    ///
    /// Prefer a real fieldmap. This exists so that a user who has none is not pushed toward the
    /// silently wrong thing, which is passing a self-measured field to
    /// [`from_field_hz`](Self::from_field_hz) as though it were measured in undistorted space.
    ///
    /// # Why it is a fixed point and not a formula
    ///
    /// The displacement is defined in undistorted space, but a field measured from the EPI is
    /// sampled in distorted space: the value recorded at array position `x'` is the off-resonance
    /// of the tissue that *landed* there, which belongs at `x = x' − s(x)`. So instead of
    /// `s(x) = k·Δf(x)` we have
    ///
    /// ```text
    /// s(x) = k · measured(x + s(x))
    /// ```
    ///
    /// which this iterates from `s₀ = k·measured(x)`. It contracts while `|ds/dx| < 1`, slightly
    /// stronger than the no-pile-up condition `ds/dx > −1` — a region stretched more than
    /// two-fold will not converge, and is reported rather than hidden. Accuracy degrades exactly
    /// where distortion is largest, which is where the correction matters most; that is the
    /// approximation, and it is why this is not the default path.
    ///
    /// # Panics
    /// If `measured_hz.len()` does not match `dims`, or `max_iterations` is zero.
    pub fn from_self_measured_hz(
        measured_hz: &[f64],
        dims: (usize, usize, usize),
        pe: PhaseEncoding,
        total_readout_time: f64,
        max_iterations: usize,
        tolerance_voxels: f64,
    ) -> (Self, SelfFieldConvergence) {
        assert!(max_iterations > 0, "max_iterations must be at least 1");
        let mut field = Self::from_field_hz(measured_hz, dims, pe, total_readout_time);
        let seed = field.shift.clone();
        let (nx, ny, nz) = dims;
        let axis = pe.axis;
        let n_pe = [nx, ny, nz][axis];

        let mut convergence =
            SelfFieldConvergence { iterations: 0, max_change_voxels: f64::INFINITY, converged: false };

        for iter in 1..=max_iterations {
            let mut next = vec![0.0f64; seed.len()];
            let mut max_change = 0.0f64;
            for (idx, slot) in next.iter_mut().enumerate() {
                let (i, j, k) = unravel(idx, nx, ny);
                let mut c = [i as f64, j as f64, k as f64];
                // Clamp rather than invalidate: the measured field, unlike image data, is a
                // smooth quantity whose edge value is a usable extrapolation, and refusing here
                // would strand the whole fixed point on a boundary voxel.
                c[axis] = (c[axis] + field.shift[idx]).clamp(0.0, n_pe as f64 - 1.0);
                let v = trilinear_sample(&seed, nx, ny, nz, c[0], c[1], c[2]);
                max_change = max_change.max((v - field.shift[idx]).abs());
                *slot = v;
            }
            field.shift = next;
            convergence =
                SelfFieldConvergence { iterations: iter, max_change_voxels: max_change, converged: max_change <= tolerance_voxels };
            if convergence.converged {
                break;
            }
        }
        (field, convergence)
    }

    /// Jacobian of the forward warp along PE: `J = 1 + ds/dx`, dimensionless, one per voxel.
    ///
    /// `s` is in voxels and `x` is in voxels, so no voxel size enters. Central differences in the
    /// interior, one-sided at the two PE faces. `J < 1` means the scanner compressed that
    /// neighbourhood (and the acquired magnitude there is correspondingly too bright); `J > 1`
    /// means it stretched it; `J ≤ 0` is [`PileUp`].
    pub fn jacobian(&self) -> Vec<f64> {
        let (nx, ny, nz) = self.dims;
        let axis = self.pe.axis;
        let n_pe = [nx, ny, nz][axis];
        let stride = [1usize, nx, nx * ny][axis];
        let mut out = vec![1.0f64; self.shift.len()];
        if n_pe < 2 {
            // A single sample along PE carries no derivative; J = 1 is the only defensible
            // answer, and such a volume cannot be distorted along that axis anyway.
            return out;
        }
        for (idx, slot) in out.iter_mut().enumerate() {
            let (i, j, k) = unravel(idx, nx, ny);
            let p = [i, j, k][axis];
            let base = idx - p * stride;
            let d = if p == 0 {
                self.shift[base + stride] - self.shift[base]
            } else if p == n_pe - 1 {
                self.shift[base + p * stride] - self.shift[base + (p - 1) * stride]
            } else {
                0.5 * (self.shift[base + (p + 1) * stride] - self.shift[base + (p - 1) * stride])
            };
            *slot = 1.0 + d;
        }
        out
    }

    /// Largest absolute displacement anywhere, in voxels — a one-number summary of how distorted
    /// the acquisition is. Over a voxel or two is worth telling the user about.
    pub fn max_shift_voxels(&self) -> f64 {
        self.shift.iter().fold(0.0f64, |m, v| m.max(v.abs()))
    }

    /// Check the field is invertible everywhere inside `mask` (or everywhere, if `None`).
    ///
    /// Restricting to a mask is not a convenience: outside the brain a fieldmap is unconstrained
    /// — extrapolated, noise-dominated, or literally zero-filled — and will routinely fold there
    /// on perfectly good data. Checking the whole volume would refuse almost every real scan.
    ///
    /// # Limitation: this is exactly as good as the mask, and cannot say so
    ///
    /// The check can only speak about voxels the mask selects, and it **cannot distinguish
    /// "nothing folded" from "nothing was checked"** from the `Ok` alone. An empty mask — a
    /// failed brain extraction, most likely — passes a field that folds everywhere, and a mask
    /// that simply does not reach the worst distortion passes for the same reason.
    ///
    /// So the success value is not `()`. [`JacobianCheck::voxels_examined`] reports what was
    /// actually inspected, which makes the two cases distinguishable *by a caller* rather than
    /// only by someone who happened to read this paragraph. The failure here is that the obvious
    /// reading of `Ok` is wrong, and documentation fights an obvious reading badly; a number you
    /// can assert on does not. Check it is the population you expected.
    ///
    /// Then read [`UnwarpedComplex::mask`] on the way out rather than treating `Ok` as a clean
    /// bill of health: `unwarp_complex` resamples the whole volume regardless, so a run with an
    /// empty mask returns a fully populated magnitude that was pulled straight through a fold.
    ///
    /// Not refused here, and the reason is the layer rather than the ambiguity: an empty mask is
    /// a property of the mask, not of unwarping, and unwarping is optional — it runs only when a
    /// fieldmap exists — so a gate here would cover a subset of runs while background removal and
    /// inversion are equally dead with one. QSM.rs#144 puts it at the masking stage, where it
    /// gates everything downstream. A primitive that is *handed* a mask reports what it examined
    /// and leaves the judgement upstream.
    ///
    /// Note also that proceeding is not the unsafe option it looks like. On the phantom at a
    /// readout long enough to fold the brain, unwarping anyway still scores 0.944 against 0.535
    /// for leaving the data distorted: the fold corrupts a few hundred voxels while the
    /// correction fixes the geometry of all of them, and the folded signal was destroyed during
    /// acquisition either way. Skipping a dependent stage as a safety measure is the wrong trade.
    pub fn check_invertible(
        &self,
        mask: Option<&[u8]>,
        min_jacobian: f64,
    ) -> Result<JacobianCheck, PileUp> {
        let jac = self.jacobian();
        self.check_jacobian(&jac, mask, min_jacobian)
    }

    fn check_jacobian(
        &self,
        jac: &[f64],
        mask: Option<&[u8]>,
        min_jacobian: f64,
    ) -> Result<JacobianCheck, PileUp> {
        let (nx, ny, _) = self.dims;
        let mut n_voxels = 0usize;
        let mut n_considered = 0usize;
        let mut worst = f64::INFINITY;
        let mut worst_idx = 0usize;
        for (idx, &j) in jac.iter().enumerate() {
            if mask.is_some_and(|m| m[idx] == 0) {
                continue;
            }
            n_considered += 1;
            if j < worst {
                worst = j;
                worst_idx = idx;
            }
            if j <= min_jacobian {
                n_voxels += 1;
            }
        }
        if n_voxels == 0 {
            return Ok(JacobianCheck {
                voxels_examined: n_considered,
                min_jacobian: (n_considered > 0).then_some(worst),
            });
        }
        Err(PileUp {
            n_voxels,
            min_jacobian: worst,
            worst_voxel: unravel(worst_idx, nx, ny),
            threshold: min_jacobian,
            n_considered,
        })
    }
}

// ------------------------------------------------------------------ parameters and results

/// Options for the unwarp functions.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug)]
pub struct UnwarpParams {
    /// Multiply the resampled magnitude by the Jacobian `1 + ds/dx`, restoring spin density.
    ///
    /// **Default `true`**. The usual argument for leaving it off comes from functional
    /// imaging, where a per-voxel constant gain is invisible to a time-series GLM, so the
    /// modulation only adds noise for nothing. QSM has no such luxury:
    ///
    /// - masking thresholds (Otsu, BET) are absolute, and an uncorrected pile-up region stays
    ///   artificially bright enough to survive them as a spurious blob;
    /// - magnitude is the SNR weight in multi-echo B0 fitting
    ///   ([`crate::fieldmap::calculate_b0_weighted`]), so an unmodulated magnitude silently
    ///   over-weights compressed tissue;
    /// - the magnitude is often reported as an output in its own right, where it should be
    ///   comparable with a non-EPI acquisition of the same subject.
    ///
    /// Phase is never modulated — the Jacobian is a positive real scale, so it cannot move the
    /// QSM measurement itself. Turn this off for quantitative maps where the value is not a
    /// density (an R2\* map resampled through [`unwarp_volume`], say).
    pub jacobian_modulation: bool,
    /// Refuse with [`PileUp`] when the Jacobian falls to or below this anywhere inside the mask.
    ///
    /// `0.0` (the default) refuses only genuine folding. Raising it to, say, `0.1` also refuses
    /// ten-fold compression, which is formally invertible but amplifies noise by the same factor
    /// — reasonable for a study that would rather drop a subject than publish that voxel.
    /// Negative values are rejected, since they would permit folding.
    pub min_jacobian: f64,
}

impl Default for UnwarpParams {
    fn default() -> Self {
        Self { jacobian_modulation: true, min_jacobian: 0.0 }
    }
}

/// Unwarped complex data and the bookkeeping that goes with it.
pub struct UnwarpedComplex {
    /// Magnitude on the undistorted grid, Jacobian-modulated if
    /// [`UnwarpParams::jacobian_modulation`] was set.
    pub magnitude: Vec<f64>,
    /// Wrapped phase in radians on `(−π, π]`, moved through the complex domain.
    pub phase: Vec<f64>,
    /// The input mask restricted to voxels that pulled from inside the field of view.
    ///
    /// Only the mask is restricted — `magnitude` and `phase` are unwarped everywhere, because a
    /// caller that wants the scalp or the neck back is entitled to it and blanking would be
    /// unrecoverable. The mask is the record of where the result means anything.
    ///
    /// A displacement of `s` voxels means the outermost `|s|` voxels at one PE face had nothing
    /// to be pulled from — that tissue was never acquired. Those voxels are zero in `magnitude`
    /// and `phase`, and zero here. The zeros are *absence of data*, not a zero measurement, and
    /// handing the original mask downstream is how a rim of nonsense gets into a field map.
    /// Pass **this** mask on.
    pub mask: Vec<u8>,
    /// The Jacobian that was computed, whether or not it was applied. Useful for reporting even
    /// on a run that did not modulate.
    pub jacobian: Vec<f64>,
    /// What the pile-up check examined. **Read it**: a run whose mask selected nothing returns
    /// `Ok` with a fully populated magnitude pulled straight through a fold, and
    /// [`JacobianCheck::voxels_examined`] of zero is what distinguishes that from a sound run.
    pub check: JacobianCheck,
}

// ------------------------------------------------------------------ unwarping

/// Unwarp magnitude and wrapped phase together, through the complex domain.
///
/// This is the entry point. `mask` restricts the [`PileUp`] check (see
/// [`DisplacementField::check_invertible`] for why that restriction is necessary rather than
/// convenient) and is intersected into [`UnwarpedComplex::mask`]; pass `None` to check the whole
/// volume, which is appropriate only for simulated data.
///
/// # Errors
/// [`PileUp`] if the phase-encode mapping folds anywhere inside the mask.
///
/// # Panics
/// If `magnitude`, `phase`, `mask` and the displacement field disagree in length.
pub fn unwarp_complex(
    magnitude: &[f64],
    phase: &[f64],
    displacement: &DisplacementField,
    mask: Option<&[u8]>,
    params: &UnwarpParams,
) -> Result<UnwarpedComplex, PileUp> {
    let (nx, ny, nz) = displacement.dims;
    let n = nx * ny * nz;
    assert_eq!(magnitude.len(), n, "magnitude length does not match the displacement field");
    assert_eq!(phase.len(), n, "phase length does not match the displacement field");
    if let Some(m) = mask {
        assert_eq!(m.len(), n, "mask length does not match the displacement field");
    }
    assert!(
        params.min_jacobian >= 0.0,
        "min_jacobian must be >= 0 (got {}); a negative threshold would permit folding",
        params.min_jacobian
    );

    let jacobian = displacement.jacobian();
    let check = displacement.check_jacobian(&jacobian, mask, params.min_jacobian)?;

    let real: Vec<f64> = (0..n).map(|i| magnitude[i] * phase[i].cos()).collect();
    let imag: Vec<f64> = (0..n).map(|i| magnitude[i] * phase[i].sin()).collect();

    let mut out_mag = vec![0.0f64; n];
    let mut out_phase = vec![0.0f64; n];
    let mut out_mask = vec![0u8; n];

    for idx in 0..n {
        let Some(c) = source_coord(idx, displacement) else { continue };
        let re = trilinear_sample(&real, nx, ny, nz, c[0], c[1], c[2]);
        let im = trilinear_sample(&imag, nx, ny, nz, c[0], c[1], c[2]);
        let gain = if params.jacobian_modulation { jacobian[idx] } else { 1.0 };
        out_mag[idx] = re.hypot(im) * gain;
        out_phase[idx] = im.atan2(re);
        out_mask[idx] = if mask.is_some_and(|m| m[idx] == 0) { 0 } else { 1 };
    }

    Ok(UnwarpedComplex { magnitude: out_mag, phase: out_phase, mask: out_mask, jacobian, check })
}

/// Unwarp a single continuous real volume — a magnitude-only EPI, an already-unwrapped field, a
/// relaxation map.
///
/// **Not for wrapped phase**: use [`unwarp_complex`], which interpolates `mag·e^{iφ}`. Routing
/// wrapped phase through here puts a band of wrong values at every wrap.
///
/// Set [`UnwarpParams::jacobian_modulation`] to `false` for a quantitative map whose value is not
/// a signal density: R2\* in Hz is a property of the tissue, and scaling it by the local stretch
/// would be wrong.
///
/// Voxels whose source lies outside the field of view read zero. There is no second return value
/// recording which those were — use [`unwarp_complex`], whose [`UnwarpedComplex::mask`] carries
/// it, or recompute it from the displacement.
///
/// # Errors
/// [`PileUp`], as [`unwarp_complex`].
pub fn unwarp_volume(
    data: &[f64],
    displacement: &DisplacementField,
    mask: Option<&[u8]>,
    params: &UnwarpParams,
) -> Result<Vec<f64>, PileUp> {
    let (nx, ny, nz) = displacement.dims;
    let n = nx * ny * nz;
    assert_eq!(data.len(), n, "data length does not match the displacement field");
    assert!(params.min_jacobian >= 0.0, "min_jacobian must be >= 0");

    let jacobian = displacement.jacobian();
    displacement.check_jacobian(&jacobian, mask, params.min_jacobian)?;

    let mut out = vec![0.0f64; n];
    for (idx, slot) in out.iter_mut().enumerate() {
        let Some(c) = source_coord(idx, displacement) else { continue };
        let v = trilinear_sample(data, nx, ny, nz, c[0], c[1], c[2]);
        *slot = if params.jacobian_modulation { v * jacobian[idx] } else { v };
    }
    Ok(out)
}

/// Unwarp a binary mask, nearest-neighbour, with no Jacobian (a label is not a density).
///
/// For a mask that was drawn on the distorted EPI. The result is additionally zeroed where the
/// source would have come from outside the field of view, for the reason given on
/// [`UnwarpedComplex::mask`].
///
/// # Errors
/// [`PileUp`], as [`unwarp_complex`]. The check runs against the mask being moved.
pub fn unwarp_mask(
    mask: &[u8],
    displacement: &DisplacementField,
    min_jacobian: f64,
) -> Result<Vec<u8>, PileUp> {
    let (nx, ny, nz) = displacement.dims;
    let n = nx * ny * nz;
    assert_eq!(mask.len(), n, "mask length does not match the displacement field");
    assert!(min_jacobian >= 0.0, "min_jacobian must be >= 0");
    displacement.check_invertible(Some(mask), min_jacobian)?;

    let mut out = vec![0u8; n];
    for (idx, slot) in out.iter_mut().enumerate() {
        let Some(c) = source_coord(idx, displacement) else { continue };
        let xi = (c[0].round() as isize).clamp(0, nx as isize - 1) as usize;
        let yi = (c[1].round() as isize).clamp(0, ny as isize - 1) as usize;
        let zi = (c[2].round() as isize).clamp(0, nz as isize - 1) as usize;
        *slot = mask[xi + yi * nx + zi * nx * ny];
    }
    Ok(out)
}

/// An unwarped echo series, the shape [`crate::pipeline`] stages pass between each other.
pub struct UnwarpedSeries {
    /// Per-echo magnitude on the undistorted grid.
    pub magnitudes: Vec<Vec<f64>>,
    /// Per-echo wrapped phase in radians on `(-pi, pi]`.
    pub phases: Vec<Vec<f64>>,
    /// The mask every echo still has data for — see [`UnwarpedComplex::mask`].
    pub mask: Vec<u8>,
}

/// Unwarp a whole echo series that shares one readout scheme.
///
/// Every echo of a multi-echo EPI is read out with the same phase-encode blips and the same
/// effective echo spacing, so they share one displacement field; only the complex data differs.
/// This is the shape [`crate::pipeline`] stages consume — per-echo magnitude and wrapped phase —
/// and the returned mask is already the intersection over every echo.
///
/// # Errors
/// [`PileUp`], checked once for the shared field.
///
/// # Panics
/// If the two series differ in length, or any volume has the wrong size.
pub fn unwarp_series(
    magnitudes: &[Vec<f64>],
    phases: &[Vec<f64>],
    displacement: &DisplacementField,
    mask: Option<&[u8]>,
    params: &UnwarpParams,
) -> Result<UnwarpedSeries, PileUp> {
    assert_eq!(magnitudes.len(), phases.len(), "magnitude and phase series differ in length");
    assert!(!magnitudes.is_empty(), "echo series is empty");
    let n = displacement.dims.0 * displacement.dims.1 * displacement.dims.2;
    let mut out_mag = Vec::with_capacity(magnitudes.len());
    let mut out_phase = Vec::with_capacity(phases.len());
    let mut out_mask = vec![1u8; n];
    for (m, p) in magnitudes.iter().zip(phases) {
        let r = unwarp_complex(m, p, displacement, mask, params)?;
        for (slot, &v) in out_mask.iter_mut().zip(&r.mask) {
            *slot &= v;
        }
        out_mag.push(r.magnitude);
        out_phase.push(r.phase);
    }
    Ok(UnwarpedSeries { magnitudes: out_mag, phases: out_phase, mask: out_mask })
}

// ------------------------------------------------------------------ forward simulation

/// Forward-warp undistorted complex data: simulate what the scanner would have acquired.
///
/// The inverse direction, for simulation and for validating [`unwarp_complex`] against something
/// that is not itself. Implemented as a **scatter**, not as a gather with a negated displacement:
/// each true voxel's complex signal is splatted into the two distorted voxels straddling
/// `x + s(x)` with linear weights. That is what the physics does — signal is conserved and simply
/// lands somewhere else — and it reproduces the two consequences a gather would have to be told
/// about separately. Compression brightens the result because several contributions land in one
/// bin, and folding happens of its own accord, irreversibly, exactly as it does on the scanner.
///
/// Signal pushed past either PE face is dropped: it left the field of view.
///
/// # Panics
/// If `magnitude` and `phase` disagree with the displacement field in length.
pub fn distort_complex(
    magnitude: &[f64],
    phase: &[f64],
    displacement: &DisplacementField,
) -> (Vec<f64>, Vec<f64>) {
    let (nx, ny, nz) = displacement.dims;
    let n = nx * ny * nz;
    assert_eq!(magnitude.len(), n, "magnitude length does not match the displacement field");
    assert_eq!(phase.len(), n, "phase length does not match the displacement field");
    let axis = displacement.pe.axis;
    let n_pe = [nx, ny, nz][axis];
    let stride = [1usize, nx, nx * ny][axis];

    let mut re = vec![0.0f64; n];
    let mut im = vec![0.0f64; n];
    for idx in 0..n {
        let (i, j, k) = unravel(idx, nx, ny);
        let p = [i, j, k][axis];
        let base = idx - p * stride;
        let target = p as f64 + displacement.shift[idx];
        let lo = target.floor();
        let w_hi = target - lo;
        let (sr, si) = (magnitude[idx] * phase[idx].cos(), magnitude[idx] * phase[idx].sin());
        for (offset, w) in [(0i64, 1.0 - w_hi), (1, w_hi)] {
            if w == 0.0 {
                continue;
            }
            let q = lo as i64 + offset;
            if q < 0 || q >= n_pe as i64 {
                continue;
            }
            let t = base + q as usize * stride;
            re[t] += sr * w;
            im[t] += si * w;
        }
    }
    (
        (0..n).map(|i| re[i].hypot(im[i])).collect(),
        (0..n).map(|i| im[i].atan2(re[i])).collect(),
    )
}

// ------------------------------------------------------------------ helpers

/// Column-major linear index → `(i, j, k)`.
#[inline]
fn unravel(idx: usize, nx: usize, ny: usize) -> (usize, usize, usize) {
    (idx % nx, (idx / nx) % ny, idx / (nx * ny))
}

/// Where destination voxel `idx` pulls from, or `None` if that is outside the field of view.
///
/// Only the PE component is fractional, so the trilinear sample this feeds degenerates to a
/// linear interpolation along that one axis — which is the point of reusing
/// [`trilinear_sample`] rather than writing a second interpolator: identical edge semantics to
/// every other resample in the crate, at the cost of six multiplications by zero.
#[inline]
fn source_coord(idx: usize, displacement: &DisplacementField) -> Option<[f64; 3]> {
    let (nx, ny, nz) = displacement.dims;
    let axis = displacement.pe.axis;
    let n_pe = [nx, ny, nz][axis];
    let (i, j, k) = unravel(idx, nx, ny);
    let mut c = [i as f64, j as f64, k as f64];
    c[axis] += displacement.shift[idx];
    // Strictly inside, so nothing is ever extrapolated from a clamped edge value: the outermost
    // voxels at one PE face genuinely have no source and must read as absent, not as a smear of
    // their neighbour.
    if !(c[axis] >= 0.0 && c[axis] <= n_pe as f64 - 1.0) {
        return None;
    }
    Some(c)
}

// ============================================================================
// Tests
//
// Expected values here are derived from the physics, not from the functions under test: a
// hand-built distorted array for the integer-shift case, closed-form linear algebra for the
// fractional and Jacobian cases, and the chord-versus-arc geometry of linear interpolation for
// the wrapped-phase case. `distort_complex` is used only in the final round trip, and it is a
// scatter where the unwarp is a gather, so an error in one does not cancel in the other.
//
// ---------------------------------------------------------------------------
// Mutation axes this suite is known to catch
//
// Recorded because the list is the durable half of a mutation sweep — the harness is ten
// minutes of boilerplate, knowing which perturbations are worth applying is not. All of these
// were run and all fail at least one assertion here; a change that makes any of them pass has
// removed coverage, whatever the test count says.
//
// Two guards for **any scripted edit of this file**, not only a rebuilt sweep: without them a
// script reports passes it has not earned, and a pass is what you were hoping for, so nobody
// checks it. The instrument you check your instruments with has no instrument of its own.
//
// Deliberately phrased for all scripted edits rather than for sweeps, because the narrower
// wording failed within three commits of being written: #141's author hit the duplicate-needle
// hazard in a one-off insertion script, having documented it for harnesses two commits earlier.
//
//   * Require each needle to match the file **exactly once** before patching. Substring
//     replacement over sites that differ only in indentation silently patches one site twice
//     and calls it two, which reports the second as caught when nothing was mutated.
//   * Require the patched text to **differ** from the original. A replacement that equals its
//     needle changes nothing and reports as a surviving mutation.
//
// (The first hazard is #141's, found there and checked here; this module's needles are all
// unique and both guards are in place.)
//
//   1.  displacement sign flipped
//   2.  displacement scaled by 1.01            <- the tightest; only the exact linear-profile
//                                                 assertion sees a 1% error
//   3.  Jacobian modulation skipped
//   4.  Jacobian computed as 1 - ds/dx
//   5.  pile-up check always returns Ok
//   6.  phase interpolated directly instead of through mag*e^{i*phi}
//   7.  displacement always applied along axis 0
//   8.  out-of-FOV voxels clamped instead of marked absent
//   9.  central difference replaced by a forward difference
//   10. self-measured fixed point never iterates
//   11. rad/s treated as Hz
//   12. self-measured fixed point's coordinate clamp removed   <- survived uncaught until a
//                                                                 test was written for it; see
//                                                                 the note below
//   13. from_bids_sidecar accepts any TotalReadoutTime
//   14. from_bids_sidecar rejects only non-finite, not zero or negative
//   15. JacobianCheck::voxels_examined reports the whole volume instead of the mask population
//
// Plus two *nulls*, which are the subtle half — see `tests/epi_unwarp.rs` for the measured
// numbers. A null that disables everything (return the input unchanged) proves only that the
// suite is connected to something. The informative null disables the mechanism under test and
// leaves its neighbours running: zeroing the displacement while Jacobian modulation and
// masking keep working.
//
// Axis 12 is the one worth reading about. It survived the first sweep because nothing tested it,
// and then survived the *first test written for it*, which asserted that the result stayed
// inside the seed's range — true for the even half of a two-cycle the missing clamp induces.
// An uncaught mutation is not automatically dead code: here the clamp is load-bearing
// (`trilinear_sample` clamps indices but not weights, so an off-grid coordinate extrapolates
// rather than reading the edge), and the finding was a missing test. But the first test for it
// was itself dead, which is the same trap twice in one afternoon.
//
// A question worth asking of every test here, which caught eight unrelated defects across this
// module and two sibling branches: **what does this test make true that the code does not
// require, and is the assertion riding on it?** Three distinct mechanisms, worth separating
// because they are found by looking in different places:
//
//   * **The input.** A linear ramp, an all-ones mask, a linear profile, a four-echo boundary —
//     a probe whose own symmetry collapses the distinction being drawn. Where the specialness
//     is deliberate and load-bearing it is named in that test's doc comment.
//   * **The tooling.** A substring needle matching two sites, twice. See the guards above.
//   * **The scoring region.** Not what the test feeds in but what it measures over. A region
//     can be sound today and degrade along an axis the test invites you to push — mine holds
//     204,396 voxels at a 20 ms readout and 12 at 4 ms — so both regions in
//     `tests/epi_unwarp.rs` now assert a floor, derived from the sampling error of the
//     statistic rather than backed off from the measured value.
//
// One prohibition, because it already caught a worthless assertion here: a **linear** probe
// cannot distinguish a central difference from a one-sided one, since both return the slope of
// a line. No ramp-based assertion in this module may be read as evidence about a stencil. The
// quadratic in `the_jacobian_uses_a_central_difference_in_the_interior` is what pins it.
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic textured volume — a wrong shift must change the answer, which a smooth
    /// ramp would not guarantee.
    fn textured(n: usize) -> Vec<f64> {
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                1.0 + (s >> 40) as f64 / 16_777_216.0
            })
            .collect()
    }

    const J_POS: PhaseEncoding = PhaseEncoding { axis: 1, positive: true };

    // ---------------------------------------------------------------- metadata

    #[test]
    fn every_bids_phase_encoding_spelling_parses_and_round_trips() {
        for (s, axis, positive) in [
            ("i", 0, true), ("i-", 0, false),
            ("j", 1, true), ("j-", 1, false),
            ("k", 2, true), ("k-", 2, false),
        ] {
            let pe = PhaseEncoding::parse(s).unwrap();
            assert_eq!((pe.axis, pe.positive), (axis, positive), "parsing {s:?}");
            assert_eq!(pe.to_bids(), s);
            assert_eq!(pe.sign(), if positive { 1.0 } else { -1.0 });
        }
        assert_eq!(PhaseEncoding::parse("  j-  ").unwrap(), PhaseEncoding { axis: 1, positive: false });
        // DICOM's InPlanePhaseEncodingDirection carries no polarity, so it must not be accepted.
        for bad in ["", "x", "ROW", "COL", "i+", "j--", "-j", "ij", "1"] {
            assert!(PhaseEncoding::parse(bad).is_err(), "{bad:?} must be rejected");
        }
    }

    /// shift[voxels] = field[Hz] x TotalReadoutTime[s], signed by polarity. Hand arithmetic.
    #[test]
    fn the_displacement_is_field_hz_times_total_readout_time_in_voxels() {
        let dims = (2, 2, 2);
        let field = vec![40.0; 8];
        let pos = DisplacementField::from_field_hz(&field, dims, J_POS, 0.05);
        // 40 Hz x 0.05 s = 2.0 voxels
        assert!(pos.shift.iter().all(|&s| (s - 2.0).abs() < 1e-15), "got {:?}", pos.shift);

        let neg = PhaseEncoding::parse("j-").unwrap();
        let rev = DisplacementField::from_field_hz(&field, dims, neg, 0.05);
        assert!(rev.shift.iter().all(|&s| (s + 2.0).abs() < 1e-15), "got {:?}", rev.shift);

        // A different readout time scales it linearly: 40 Hz x 0.0132 s = 0.528 voxels.
        let short = DisplacementField::from_field_hz(&field, dims, J_POS, 0.0132);
        assert!((short.shift[0] - 0.528).abs() < 1e-15, "got {}", short.shift[0]);
    }

    /// Both sidecar fields are validated the same way, which was the point of adding this
    /// entry point: `PhaseEncodingDirection` already returned a `Result` while
    /// `TotalReadoutTime` panicked, and they come out of the same JSON file.
    #[test]
    fn the_sidecar_entry_point_refuses_both_fields_rather_than_panicking() {
        let dims = (2, 4, 2);
        let n = 16;
        let field = vec![40.0; n];

        let ok = DisplacementField::from_bids_sidecar(&field, dims, "j-", 0.05).unwrap();
        assert_eq!(ok.pe.to_bids(), "j-");
        assert!((ok.shift[0] + 2.0).abs() < 1e-15, "40 Hz x 0.05 s reversed = -2 voxels");
        // Identical to the pre-validated route, so this is a gate and not a second code path.
        let direct = DisplacementField::from_field_hz(
            &field, dims, PhaseEncoding::parse("j-").unwrap(), 0.05);
        assert_eq!(ok.shift, direct.shift);

        match DisplacementField::from_bids_sidecar(&field, dims, "COL", 0.05) {
            Err(SidecarError::PhaseEncodingDirection(e)) => assert_eq!(e.got, "COL"),
            other => panic!("expected a PhaseEncodingDirection error, got {other:?}"),
        }

        // Finite-but-unusable and non-finite alike: a broken sidecar, not a caller bug.
        for bad in [0.0, -0.01, f64::NAN, f64::INFINITY] {
            match DisplacementField::from_bids_sidecar(&field, dims, "j", bad) {
                Err(SidecarError::TotalReadoutTime { got }) => {
                    assert!(got.is_nan() == bad.is_nan() && (got.is_nan() || got == bad));
                }
                other => panic!("TotalReadoutTime {bad} must be refused, got {other:?}"),
            }
        }
        assert!(SidecarError::TotalReadoutTime { got: 0.0 }
            .to_string()
            .contains("strictly positive"));
    }

    #[test]
    fn hz_rad_per_s_ppm_and_voxel_shift_constructors_all_agree() {
        let dims = (2, 2, 2);
        let trt = 0.04;
        let hz = 37.5_f64;
        let b0 = 3.0;
        let expected = hz * trt; // 1.5 voxels

        let a = DisplacementField::from_field_hz(&[hz; 8], dims, J_POS, trt);
        let b = DisplacementField::from_field_rad_per_s(
            &[hz * 2.0 * std::f64::consts::PI; 8], dims, J_POS, trt);
        let ppm = hz / (42.576e6 * b0) * 1e6;
        let c = DisplacementField::from_field_ppm(&[ppm; 8], dims, J_POS, trt, b0);
        let d = DisplacementField::from_voxel_shift(vec![expected; 8], dims, J_POS);

        for (name, f) in [("hz", &a), ("rad/s", &b), ("ppm", &c), ("shift", &d)] {
            assert!((f.shift[0] - expected).abs() < 1e-12, "{name}: {} != {expected}", f.shift[0]);
        }
        assert!((a.max_shift_voxels() - 1.5).abs() < 1e-12);
    }

    // ---------------------------------------------------------------- geometry

    /// A whole-number shift makes the answer a pure index translation, so the expected volume is
    /// built by index arithmetic alone. This is the test that pins the *sign* and the *axis*.
    #[test]
    fn a_whole_voxel_shift_translates_the_volume_by_exactly_that_many_voxels() {
        let dims = (5, 9, 4);
        let (nx, ny, nz) = dims;
        let n = nx * ny * nz;
        let truth = textured(n);
        let phase = vec![0.3; n];
        let trt = 0.05;
        let shift_voxels = 2usize;
        let field = vec![shift_voxels as f64 / trt; n]; // 40 Hz -> exactly 2 voxels

        for spelling in ["i", "i-", "j", "j-", "k", "k-"] {
            let pe = PhaseEncoding::parse(spelling).unwrap();
            let dir = if pe.positive { 1isize } else { -1 };
            let n_pe = [nx, ny, nz][pe.axis];
            let stride = [1usize, nx, nx * ny][pe.axis];
            let delta = dir * shift_voxels as isize;

            // Convention under test: positive off-resonance displaces signal toward *increasing*
            // index for "i"/"j"/"k". So the scanner recorded, at position p, whatever truly
            // belongs at p - delta.
            let mut distorted = vec![0.0f64; n];
            for (idx, slot) in distorted.iter_mut().enumerate() {
                let (i, j, k) = unravel(idx, nx, ny);
                let p = [i, j, k][pe.axis] as isize;
                let src = p - delta;
                if src >= 0 && src < n_pe as isize {
                    let base = idx - p as usize * stride;
                    *slot = truth[base + src as usize * stride];
                }
            }

            let disp = DisplacementField::from_field_hz(&field, dims, pe, trt);
            let out = unwarp_complex(&distorted, &phase, &disp, None, &UnwarpParams::default())
                .expect("a constant field cannot fold");

            // J = 1 for a constant shift, so modulation is a no-op and this isolates geometry.
            assert!(out.jacobian.iter().all(|&j| (j - 1.0).abs() < 1e-15));

            for (idx, &want) in truth.iter().enumerate() {
                let (i, j, k) = unravel(idx, nx, ny);
                let p = [i, j, k][pe.axis] as isize;
                let pulls_from = p + delta;
                if pulls_from >= 0 && pulls_from < n_pe as isize {
                    assert_eq!(out.mask[idx], 1, "{spelling}: voxel {idx} should have data");
                    assert!(
                        (out.magnitude[idx] - want).abs() < 1e-13,
                        "{spelling}: voxel {idx} got {} want {want}",
                        out.magnitude[idx]
                    );
                } else {
                    assert_eq!(out.mask[idx], 0, "{spelling}: voxel {idx} is outside the FOV");
                    assert_eq!(out.magnitude[idx], 0.0);
                }
            }
        }
    }

    /// A fractional shift of a function that is linear along PE is recovered exactly, because
    /// linear interpolation reproduces a linear function. Expected values are the closed form.
    ///
    /// What is special about this input: linearity is what makes the answer *exact*, and the
    /// assertion rides on it. So this pins the **shift magnitude** to 1e-12 — a 1% error moves
    /// it by 0.0375 — and says nothing about the interpolation kernel, since cubic or any other
    /// linear-reproducing scheme would pass identically. Real data is not linear along PE; the
    /// round trip on the phantom is what covers the kernel.
    #[test]
    fn a_fractional_shift_recovers_a_linear_profile_to_machine_precision() {
        let dims = (3, 20, 2);
        let (nx, ny, nz) = dims;
        let n = nx * ny * nz;
        let trt = 0.05;
        let s = 1.25_f64; // 25 Hz x 0.05 s
        let field = vec![s / trt; n];
        let truth_at = |y: f64| 10.0 + 3.0 * y;

        // What the scanner acquired: the true object sampled s voxels away.
        let distorted: Vec<f64> = (0..n)
            .map(|idx| { let (_, j, _) = unravel(idx, nx, ny); truth_at(j as f64 - s) })
            .collect();
        let phase = vec![-0.7; n];

        let disp = DisplacementField::from_field_hz(&field, dims, J_POS, trt);
        let out = unwarp_complex(&distorted, &phase, &disp, None, &UnwarpParams::default()).unwrap();

        let mut checked = 0;
        for idx in 0..n {
            let (_, j, _) = unravel(idx, nx, ny);
            if j as f64 + s > (ny - 1) as f64 { continue; }
            assert!(
                (out.magnitude[idx] - truth_at(j as f64)).abs() < 1e-12,
                "voxel {idx} (j={j}) got {} want {}", out.magnitude[idx], truth_at(j as f64)
            );
            assert!((out.phase[idx] + 0.7).abs() < 1e-12);
            checked += 1;
        }
        assert_eq!(checked, nx * nz * (ny - 2), "expected 18 of 20 PE positions to be in the FOV");
        let _ = nz;
    }

    #[test]
    fn a_single_sample_along_pe_has_no_derivative_to_take() {
        // A 2D-in-plane acquisition with one sample along the chosen axis cannot be distorted
        // along it, and a central difference has nothing to difference. J = 1 is the only
        // defensible answer, and the alternative is an out-of-bounds index.
        let dims = (4, 1, 3);
        let n = 12;
        let disp = DisplacementField::from_voxel_shift(
            (0..n).map(|i| i as f64 * 0.3).collect(), dims, J_POS);
        let jac = disp.jacobian();
        assert_eq!(jac.len(), n);
        assert!(jac.iter().all(|&j| j == 1.0), "got {jac:?}");
        disp.check_invertible(None, 0.0).expect("J = 1 cannot fold");
    }

    #[test]
    fn the_parse_error_names_what_it_rejected_and_what_it_wanted() {
        let err = PhaseEncoding::parse("COL").unwrap_err();
        assert_eq!(err.got, "COL");
        let msg = err.to_string();
        assert!(msg.contains("\"COL\""), "{msg}");
        assert!(msg.contains("i, i-, j, j-, k, k-"), "{msg}");
    }

    #[test]
    fn a_zero_field_is_the_identity() {
        let dims = (4, 5, 3);
        let n = 60;
        let mag = textured(n);
        let phase: Vec<f64> = (0..n).map(|i| (i as f64 * 1.1).sin() * 3.0).collect();
        let disp = DisplacementField::from_field_hz(&vec![0.0; n], dims, J_POS, 0.05);
        let out = unwarp_complex(&mag, &phase, &disp, None, &UnwarpParams::default()).unwrap();
        for i in 0..n {
            assert!((out.magnitude[i] - mag[i]).abs() < 1e-13, "voxel {i}");
            assert!((out.phase[i] - phase[i]).abs() < 1e-13, "voxel {i}");
            assert_eq!(out.mask[i], 1);
        }
    }

    // ---------------------------------------------------------------- wrapped phase

    /// Wrapped phase survives only because the resample happens in the complex domain. The
    /// achieved error is compared against the chord-versus-arc error of linear interpolation on
    /// the unit circle — a closed form in (fraction, phase step) that knows nothing about this
    /// module — and a control that interpolates the wrapped phase values directly is required to
    /// fail by two orders of magnitude more.
    #[test]
    fn wrapped_phase_survives_because_it_moves_through_the_complex_domain() {
        let dims = (2, 24, 2);
        let (nx, ny, _) = dims;
        let n = 2 * 24 * 2;
        let trt = 0.05;
        let s = 1.25_f64;
        let step = 0.9_f64; // rad per voxel along PE: wraps every ~7 voxels
        let wrap = |p: f64| p.sin().atan2(p.cos());

        // True continuous signal T(y) = exp(i*step*y); the scanner sampled T(j - s).
        let dist_phase: Vec<f64> = (0..n)
            .map(|idx| { let (_, j, _) = unravel(idx, nx, ny); wrap(step * (j as f64 - s)) })
            .collect();
        let dist_mag = vec![1.0f64; n];

        let disp = DisplacementField::from_field_hz(&vec![s / trt; n], dims, J_POS, trt);
        let out = unwarp_complex(&dist_mag, &dist_phase, &disp, None, &UnwarpParams::default()).unwrap();

        // Chord geometry: interpolating between exp(i*a) and exp(i*(a+step)) at fraction f lands
        // on the chord, whose argument advances by atan2(f sin step, (1-f) + f cos step) instead
        // of f*step, and whose modulus is |(1-f) + f exp(i step)| instead of 1.
        let f = s.fract();
        let chord_arg = (f * step.sin()).atan2((1.0 - f) + f * step.cos());
        let predicted_phase_err = (f * step - chord_arg).abs();
        let predicted_mag = ((1.0 - f) + f * step.cos()).hypot(f * step.sin());
        assert!(
            (0.011..0.012).contains(&predicted_phase_err),
            "sanity: the chord error for f=0.25, step=0.9 rad is ~0.0115 rad, got {predicted_phase_err}"
        );

        let mut worst_complex = 0.0f64;
        let mut worst_direct = 0.0f64;
        for idx in 0..n {
            let (_, j, _) = unravel(idx, nx, ny);
            if j as f64 + s > (ny - 1) as f64 { continue; }
            let want = wrap(step * j as f64);
            worst_complex = worst_complex.max(wrap(out.phase[idx] - want).abs());
            assert!((out.magnitude[idx] - predicted_mag).abs() < 1e-12);

            // Control: the same gather applied to the wrapped phase values themselves.
            let lo = (j as f64 + s).floor() as usize;
            let frac = (j as f64 + s).fract();
            let base = idx - j * nx;
            let naive = dist_phase[base + lo * nx] * (1.0 - frac)
                + dist_phase[base + (lo + 1) * nx] * frac;
            worst_direct = worst_direct.max(wrap(naive - want).abs());
        }
        assert!(
            worst_complex <= predicted_phase_err + 1e-12,
            "complex-domain error {worst_complex} exceeds the chord bound {predicted_phase_err}"
        );
        // The control's error is analytic too. Away from a wrap, linear interpolation of the
        // phase values *is* exact, because the phase is linear in j there. At a wrap the two
        // samples straddle a 2*pi step, so the interpolant misses by exactly f*2*pi -- here
        // pi/2, which is 137x the chord error the complex-domain route pays.
        let predicted_direct_err = 2.0 * std::f64::consts::PI * f;
        assert!(
            (worst_direct - predicted_direct_err).abs() < 1e-12,
            "the direct-phase control should miss by exactly f*2pi = {predicted_direct_err}, got {worst_direct}"
        );
        assert!(worst_direct > 100.0 * worst_complex);
    }

    // ---------------------------------------------------------------- Jacobian

    /// For a field that is linear along PE the Jacobian is the constant 1 + k*b, at the faces as
    /// well as in the interior, because one-sided and central differences agree on a line.
    ///
    /// That agreement is exactly why this test says nothing about *which* stencil is used --
    /// a linear probe is symmetric enough to collapse the distinction, and a mutation sweep
    /// confirmed it passes with the central difference replaced by a forward one. It pins the
    /// sign and the scale of `ds/dx`; [`the_jacobian_uses_a_central_difference_in_the_interior`]
    /// pins the stencil. Do not merge the two back together.
    #[test]
    fn the_jacobian_of_a_linear_field_is_the_hand_computed_constant() {
        let dims = (3, 12, 2);
        let (nx, ny, _) = dims;
        let n = 3 * 12 * 2;
        let trt = 0.05;
        for (a, b, expect) in [(0.0, 8.0, 1.4), (40.0, -5.0, 0.75), (10.0, 0.0, 1.0)] {
            // field(j) = a + b*j Hz; shift = trt*(a + b*j); dshift/dj = trt*b
            let field: Vec<f64> = (0..n)
                .map(|idx| { let (_, j, _) = unravel(idx, nx, ny); a + b * j as f64 })
                .collect();
            let disp = DisplacementField::from_field_hz(&field, dims, J_POS, trt);
            let jac = disp.jacobian();
            assert!((1.0 + trt * b - expect).abs() < 1e-12, "test arithmetic: {}", 1.0 + trt * b);
            for (idx, &j) in jac.iter().enumerate() {
                assert!((j - expect).abs() < 1e-12, "b={b} voxel {idx} got {j} want {expect}");
            }
        }
    }

    /// A quadratic separates the stencils where a ramp cannot. For `s(j) = c*j²` the central
    /// difference is `c*((p+1)² - (p-1)²)/2 = 2cp`, which is the analytic derivative and so exact;
    /// a forward difference returns `c*(2p+1)`, off by exactly `c` at every interior voxel. With
    /// `c = 0.01` that is 10^10 times the tolerance here.
    #[test]
    fn the_jacobian_uses_a_central_difference_in_the_interior() {
        let dims = (3, 14, 2);
        let (nx, ny, _) = dims;
        let n = 3 * 14 * 2;
        let c = 0.01_f64;
        let shift: Vec<f64> = (0..n)
            .map(|idx| { let (_, j, _) = unravel(idx, nx, ny); c * (j as f64).powi(2) })
            .collect();
        let jac = DisplacementField::from_voxel_shift(shift, dims, J_POS).jacobian();
        for (idx, &got) in jac.iter().enumerate() {
            let (_, j, _) = unravel(idx, nx, ny);
            let p = j as f64;
            let want = if j == 0 {
                1.0 + c // one-sided at the low face: s(1) - s(0) = c
            } else if j == ny - 1 {
                1.0 + c * (2.0 * p - 1.0) // one-sided at the high face: s(p) - s(p-1)
            } else {
                1.0 + 2.0 * c * p // central, exact for a quadratic
            };
            assert!((got - want).abs() < 1e-12, "voxel {idx} (j={j}) got {got} want {want}");
        }
    }

    /// Signal conservation: where the readout compressed the object by J, the acquired magnitude
    /// is 1/J too bright, and only the Jacobian modulation puts it back. The distorted array is
    /// built from `true/J`, which is conservation of signal, not anything this module computes.
    ///
    /// What is special about this input: the true magnitude is *constant*, which makes the
    /// resample exact and leaves the modulation as the only thing the assertion can see. That is
    /// the point — but it also means this cannot detect a correction that modulated correctly
    /// while blurring, since a constant survives any interpolation. The phantom round trip is
    /// where blur would show.
    #[test]
    fn jacobian_modulation_restores_the_amplitude_that_compression_inflated() {
        let dims = (2, 16, 2);
        let (nx, ny, _) = dims;
        let n = 2 * 16 * 2;
        let trt = 0.05;
        // shift(j) = 2.0 - 0.25*j  =>  J = 0.75, and the gather reads j+shift in [2.0, 13.25]
        let (ka, kb) = (2.0_f64, -0.25_f64);
        let jac_expected = 1.0 + kb;
        let field: Vec<f64> = (0..n)
            .map(|idx| { let (_, j, _) = unravel(idx, nx, ny); (ka + kb * j as f64) / trt })
            .collect();
        let disp = DisplacementField::from_field_hz(&field, dims, J_POS, trt);

        let true_mag = 5.0_f64;
        let distorted = vec![true_mag / jac_expected; n]; // signal conservation
        let phase = vec![0.0; n];

        let on = unwarp_complex(&distorted, &phase, &disp, None,
            &UnwarpParams { jacobian_modulation: true, min_jacobian: 0.0 }).unwrap();
        let off = unwarp_complex(&distorted, &phase, &disp, None,
            &UnwarpParams { jacobian_modulation: false, min_jacobian: 0.0 }).unwrap();

        for idx in 0..n {
            assert_eq!(on.mask[idx], 1, "voxel {idx} should be inside the FOV");
            assert!((on.magnitude[idx] - true_mag).abs() < 1e-12,
                "modulated voxel {idx} got {} want {true_mag}", on.magnitude[idx]);
            assert!((off.magnitude[idx] - true_mag / jac_expected).abs() < 1e-12,
                "unmodulated voxel {idx} got {} want {}", off.magnitude[idx], true_mag / jac_expected);
        }
        // The two differ by exactly the Jacobian: the modulation is doing real work here.
        assert!((on.magnitude[0] / off.magnitude[0] - jac_expected).abs() < 1e-12);
    }

    #[test]
    fn a_quantitative_map_can_opt_out_of_modulation() {
        let dims = (2, 10, 2);
        let n = 40;
        let field: Vec<f64> = (0..n)
            .map(|idx| { let (_, j, _) = unravel(idx, 2, 10); (1.0 - 0.1 * j as f64) / 0.05 })
            .collect();
        let disp = DisplacementField::from_field_hz(&field, dims, J_POS, 0.05);
        let data = vec![7.0; n];
        let plain = unwarp_volume(&data, &disp, None,
            &UnwarpParams { jacobian_modulation: false, min_jacobian: 0.0 }).unwrap();
        // J = 0.9 everywhere, so the modulated version is 0.9x and the unmodulated one is not.
        let modulated = unwarp_volume(&data, &disp, None, &UnwarpParams::default()).unwrap();
        for i in 0..n {
            if plain[i] == 0.0 { continue; }
            assert!((plain[i] - 7.0).abs() < 1e-12, "voxel {i}: {}", plain[i]);
            assert!((modulated[i] - 6.3).abs() < 1e-12, "voxel {i}: {}", modulated[i]);
        }
    }

    // ---------------------------------------------------------------- pile-up

    /// A localised fold: the voxel count, the worst Jacobian and the worst voxel are all worked
    /// out by hand from the central-difference stencil.
    #[test]
    fn a_folded_region_is_refused_with_the_numbers_that_explain_it() {
        let dims = (4, 10, 3);
        let (nx, ny, nz) = dims;
        let n = nx * ny * nz;
        // Per PE line: s = [0,0,0,0, 3, -3, 0,0,0,0]
        //   j=4: (s[5]-s[3])/2 = -1.5 -> J = -0.5     j=5: (s[6]-s[4])/2 = -1.5 -> J = -0.5
        //   j=3: (s[4]-s[2])/2 = +1.5 -> J = +2.5     j=6: (s[7]-s[5])/2 = +1.5 -> J = +2.5
        let profile = [0.0, 0.0, 0.0, 0.0, 3.0, -3.0, 0.0, 0.0, 0.0, 0.0];
        let shift: Vec<f64> = (0..n)
            .map(|idx| { let (_, j, _) = unravel(idx, nx, ny); profile[j] })
            .collect();
        let disp = DisplacementField::from_voxel_shift(shift, dims, J_POS);

        let err = disp.check_invertible(None, 0.0).expect_err("J = -0.5 must be refused");
        assert_eq!(err.n_voxels, 2 * nx * nz, "two folded PE positions on each of {} lines", nx * nz);
        assert_eq!(err.n_considered, n);
        assert!((err.min_jacobian + 0.5).abs() < 1e-12, "got {}", err.min_jacobian);
        // Column-major scan order reaches j=4, i=0, k=0 first among the folded voxels.
        assert_eq!(err.worst_voxel, (0, 4, 0));
        assert_eq!(err.threshold, 0.0);
        assert!(err.to_string().contains("pile-up"));

        // unwarp_complex must refuse too, not just the standalone check.
        let mag = vec![1.0; n];
        let ph = vec![0.0; n];
        assert!(unwarp_complex(&mag, &ph, &disp, None, &UnwarpParams::default()).is_err());
        assert!(unwarp_volume(&mag, &disp, None, &UnwarpParams::default()).is_err());
        assert!(unwarp_mask(&vec![1u8; n], &disp, 0.0).is_err());
    }

    /// A mask that selects nothing makes the check vacuous, and it reports `Ok` exactly as a
    /// genuinely clean scan does. Pinned so the behaviour is known rather than accidental; the
    /// limitation is documented on [`DisplacementField::check_invertible`].
    #[test]
    fn an_empty_mask_passes_a_field_that_folds_everywhere() {
        let dims = (4, 10, 3);
        let (nx, ny, nz) = dims;
        let n = nx * ny * nz;
        let profile = [0.0, 0.0, 0.0, 0.0, 3.0, -3.0, 0.0, 0.0, 0.0, 0.0];
        let shift: Vec<f64> = (0..n)
            .map(|idx| { let (_, j, _) = unravel(idx, nx, ny); profile[j] })
            .collect();
        let disp = DisplacementField::from_voxel_shift(shift, dims, J_POS);
        // Unmasked, this field is refused outright.
        assert!(disp.check_invertible(None, 0.0).is_err());

        let empty = vec![0u8; n];
        let check = disp
            .check_invertible(Some(&empty), 0.0)
            .expect("an empty mask selects no voxel to object to");
        // The machine-checkable part: `Ok` alone cannot say this, the report can.
        assert_eq!(check.voxels_examined, 0);
        assert_eq!(check.min_jacobian, None);

        // And the full pipeline returns Ok with data everywhere — pulled through the fold — so
        // the returned mask is the only thing distinguishing this from a sound run.
        let out = unwarp_complex(&vec![1.0; n], &vec![0.0; n], &disp, Some(&empty),
            &UnwarpParams::default()).expect("vacuously accepted");
        assert_eq!(out.check.voxels_examined, 0, "the entry point must report it too");
        assert!(out.mask.iter().all(|&m| m == 0), "every voxel must be marked as having no data");
        assert!(
            out.magnitude.iter().any(|&v| v > 0.0),
            "magnitude is still populated, which is why the mask has to be read"
        );
    }

    /// The counterpart: a real mask must report a real population and a real minimum, or
    /// `voxels_examined` is a field nobody could trust.
    #[test]
    fn a_populated_mask_reports_what_it_examined() {
        let dims = (2, 8, 2);
        let n = 32;
        let trt = 0.05;
        // shift(j) = 3.0 - 0.9*j  =>  J = 0.1 everywhere: invertible but heavily compressed.
        let field: Vec<f64> = (0..n)
            .map(|idx| { let (_, j, _) = unravel(idx, 2, 8); (3.0 - 0.9 * j as f64) / trt })
            .collect();
        let disp = DisplacementField::from_field_hz(&field, dims, J_POS, trt);

        let all = disp.check_invertible(None, 0.0).unwrap();
        assert_eq!(all.voxels_examined, n);
        assert!((all.min_jacobian.unwrap() - 0.1).abs() < 1e-12);

        // Half the volume, counted by hand: j < 4 on a 2 x 8 x 2 grid is 2 * 4 * 2 = 16.
        let half: Vec<u8> = (0..n)
            .map(|idx| { let (_, j, _) = unravel(idx, 2, 8); u8::from(j < 4) })
            .collect();
        let part = disp.check_invertible(Some(&half), 0.0).unwrap();
        assert_eq!(part.voxels_examined, 16);
        assert!((part.min_jacobian.unwrap() - 0.1).abs() < 1e-12);
    }

    /// The check is confined to the mask, because outside the brain a fieldmap is unconstrained
    /// and folds routinely on perfectly good data.
    #[test]
    fn folding_outside_the_mask_does_not_refuse_the_scan() {
        let dims = (4, 10, 3);
        let (nx, ny, nz) = dims;
        let n = nx * ny * nz;
        let profile = [0.0, 0.0, 0.0, 0.0, 3.0, -3.0, 0.0, 0.0, 0.0, 0.0];
        let shift: Vec<f64> = (0..n)
            .map(|idx| { let (_, j, _) = unravel(idx, nx, ny); profile[j] })
            .collect();
        let disp = DisplacementField::from_voxel_shift(shift, dims, J_POS);

        // Mask out only the two folded PE positions.
        let mask: Vec<u8> = (0..n)
            .map(|idx| { let (_, j, _) = unravel(idx, nx, ny); u8::from(j != 4 && j != 5) })
            .collect();
        disp.check_invertible(Some(&mask), 0.0).expect("the fold is outside the mask");

        // Masking out only one of them is not enough — the other still folds.
        let half: Vec<u8> = (0..n)
            .map(|idx| { let (_, j, _) = unravel(idx, nx, ny); u8::from(j != 4) })
            .collect();
        let err = disp.check_invertible(Some(&half), 0.0).expect_err("j=5 still folds");
        assert_eq!(err.n_voxels, nx * nz);
        assert_eq!(err.worst_voxel, (0, 5, 0));
    }

    /// `min_jacobian` lets a study refuse severe compression that is formally invertible.
    #[test]
    fn the_jacobian_threshold_separates_merely_compressed_from_folded() {
        let dims = (2, 8, 2);
        let n = 32;
        let trt = 0.05;
        // shift(j) = 3.0 - 0.9*j  =>  J = 0.1: compressed ten-fold, but not folded.
        let field: Vec<f64> = (0..n)
            .map(|idx| { let (_, j, _) = unravel(idx, 2, 8); (3.0 - 0.9 * j as f64) / trt })
            .collect();
        let disp = DisplacementField::from_field_hz(&field, dims, J_POS, trt);
        assert!((disp.jacobian()[0] - 0.1).abs() < 1e-12);

        disp.check_invertible(None, 0.0).expect("J = 0.1 > 0 is invertible");
        let err = disp.check_invertible(None, 0.2).expect_err("J = 0.1 is below a 0.2 floor");
        assert_eq!(err.n_voxels, n);
        assert!((err.min_jacobian - 0.1).abs() < 1e-12);
        assert_eq!(err.threshold, 0.2);
    }

    #[test]
    #[should_panic(expected = "min_jacobian must be >= 0")]
    fn a_negative_jacobian_floor_is_rejected_because_it_would_permit_folding() {
        let dims = (2, 2, 2);
        let disp = DisplacementField::from_field_hz(&[0.0; 8], dims, J_POS, 0.05);
        let _ = unwarp_complex(&[1.0; 8], &[0.0; 8], &disp, None,
            &UnwarpParams { jacobian_modulation: true, min_jacobian: -0.5 });
    }

    // ---------------------------------------------------------------- series and masks

    #[test]
    fn a_series_shares_one_field_and_returns_the_intersection_of_the_masks() {
        let dims = (2, 8, 2);
        let (nx, ny, _) = dims;
        let n = 32;
        let trt = 0.05;
        let disp = DisplacementField::from_field_hz(&vec![2.0 / trt; n], dims, J_POS, trt);
        let mags = vec![textured(n), textured(n).iter().map(|v| v * 0.5).collect()];
        let phases = vec![vec![0.1; n], vec![-2.9; n]];
        let s = unwarp_series(&mags, &phases, &disp, None, &UnwarpParams::default()).unwrap();
        let (om, mask) = (&s.magnitudes, &s.mask);
        assert_eq!(s.magnitudes.len(), 2);
        assert_eq!(s.phases.len(), 2);
        // Every echo loses the same last two PE positions, so the intersection is that mask.
        for (idx, &m) in mask.iter().enumerate() {
            let (_, j, _) = unravel(idx, nx, ny);
            assert_eq!(m, u8::from(j + 2 < ny), "voxel {idx}");
        }
        // The second echo is exactly half the first, which a shared field must preserve.
        for (a, b) in om[0].iter().zip(&om[1]) {
            assert!((b - 0.5 * a).abs() < 1e-13);
        }
    }

    #[test]
    fn a_mask_moves_by_nearest_neighbour_and_loses_what_left_the_fov() {
        let dims = (2, 8, 2);
        let (nx, ny, _) = dims;
        let n = 32;
        let trt = 0.05;
        let disp = DisplacementField::from_field_hz(&vec![2.0 / trt; n], dims, J_POS, trt);
        // A mask occupying PE positions 2..=5 in the distorted data.
        let mask: Vec<u8> = (0..n)
            .map(|idx| { let (_, j, _) = unravel(idx, nx, ny); u8::from((2..=5).contains(&j)) })
            .collect();
        let out = unwarp_mask(&mask, &disp, 0.0).unwrap();
        for idx in 0..n {
            let (_, j, _) = unravel(idx, nx, ny);
            // out[j] = mask[j+2] when j+2 is in range, else 0.
            let want = if j + 2 < ny { mask[idx + 2 * nx] } else { 0 };
            assert_eq!(out[idx], want, "voxel {idx} (j={j})");
        }
    }

    // ---------------------------------------------------------------- self-measured field

    /// A field measured on the distorted EPI is recovered by the fixed point. The "measured"
    /// array is built from the closed-form inverse of the forward map, so nothing in this
    /// module produced the input.
    ///
    /// What is special about this input: a linear field has a *spatially uniform* contraction
    /// rate (0.25 here), so convergence is uniform and the iteration count is meaningful. A real
    /// field contracts at different rates in different places and converges at the pace of its
    /// worst region, which is the one the correction matters most in. This pins correctness of
    /// the iteration, not its cost on real data.
    #[test]
    fn the_self_measured_fixed_point_recovers_a_field_sampled_in_distorted_space() {
        let dims = (2, 16, 2);
        let (nx, ny, _) = dims;
        let n = 2 * 16 * 2;
        let trt = 0.05;
        // True field a + b*j Hz, chosen so shift(j) = 2.0 - 0.25*j (contraction factor 0.25).
        let (ka, kb) = (2.0_f64, -0.25_f64);
        let (a, b) = (ka / trt, kb / trt);
        // Forward map y' = ka + (1+kb)*y, so the value recorded at y' belongs at
        // y = (y' - ka)/(1+kb).
        let measured: Vec<f64> = (0..n)
            .map(|idx| {
                let (_, j, _) = unravel(idx, nx, ny);
                let y = (j as f64 - ka) / (1.0 + kb);
                a + b * y
            })
            .collect();

        let (disp, conv) =
            DisplacementField::from_self_measured_hz(&measured, dims, J_POS, trt, 60, 1e-13);
        assert!(conv.converged, "should contract at 0.25 per iteration: {conv:?}");
        assert!(conv.iterations < 60);
        for idx in 0..n {
            let (_, j, _) = unravel(idx, nx, ny);
            let want = ka + kb * j as f64;
            assert!((disp.shift[idx] - want).abs() < 1e-10,
                "voxel {idx} (j={j}) got {} want {want}", disp.shift[idx]);
        }

        // Naively treating the measured field as undistorted gets a visibly different answer, so
        // the fixed point is not cosmetic.
        let naive = DisplacementField::from_field_hz(&measured, dims, J_POS, trt);
        let worst = (0..n)
            .map(|i| (naive.shift[i] - disp.shift[i]).abs())
            .fold(0.0f64, f64::max);
        assert!(worst > 0.5, "the naive and corrected fields differ by only {worst} voxels");
    }

    /// A field big enough to sample off the grid still has a well-defined fixed point, and the
    /// clamp is what makes that true.
    ///
    /// [`trilinear_sample`] clamps the two *indices* it reads but not the interpolation
    /// *weights*: a source coordinate of -30 gives `y0 = 0`, `y1 = 1` and `fy = -30`, so it
    /// extrapolates with weights of 31 and -30 instead of returning the edge value. Here that
    /// turns the iteration into a permanent two-cycle — the extrapolation lands on 0, sampling
    /// at 0 returns the seed, sampling at the seed extrapolates back to 0 — which never
    /// converges and whose reported answer depends on the parity of the iteration cap.
    ///
    /// Clamping the coordinate first collapses it to a genuine contraction: every voxel samples
    /// the low edge, and the second iteration is already stationary at -30.
    ///
    /// Written after the clamp survived a mutation uncaught, and then after the *first* attempt
    /// at this test also passed without it — asserting that the result stayed inside the seed's
    /// range, which the even half of the two-cycle happens to satisfy. Convergence is the
    /// property that actually separates them.
    #[test]
    fn a_field_that_samples_off_the_grid_still_has_a_fixed_point() {
        let dims = (2, 12, 2);
        let (nx, ny, _) = dims;
        let n = 2 * 12 * 2;
        let trt = 0.05;
        // shift(j) = -(30 + j) voxels on a 12-voxel axis: every source coordinate is far below 0.
        let measured: Vec<f64> = (0..n)
            .map(|idx| { let (_, j, _) = unravel(idx, nx, ny); -(30.0 + j as f64) / trt })
            .collect();
        let (solved, conv) =
            DisplacementField::from_self_measured_hz(&measured, dims, J_POS, trt, 10, 1e-12);

        assert!(conv.converged, "an off-grid field must still reach a fixed point: {conv:?}");
        assert_eq!(conv.iterations, 2, "it should be stationary after the second pass");
        for (idx, &s) in solved.shift.iter().enumerate() {
            assert!(s.is_finite(), "voxel {idx} is not finite: {s}");
            // Every voxel samples the low edge, whose seed value is -(30 + 0).
            assert!((s + 30.0).abs() < 1e-12, "voxel {idx} got {s}, want -30");
        }
    }

    #[test]
    fn the_self_measured_fixed_point_reports_when_it_has_not_converged() {
        let dims = (2, 16, 2);
        let n = 64;
        let measured: Vec<f64> = (0..n)
            .map(|idx| { let (_, j, _) = unravel(idx, 2, 16); (2.0 - 0.25 * j as f64) / 0.05 })
            .collect();
        let (_, conv) =
            DisplacementField::from_self_measured_hz(&measured, dims, J_POS, 0.05, 1, 1e-13);
        assert_eq!(conv.iterations, 1);
        assert!(!conv.converged);
        assert!(conv.max_change_voxels.is_finite() && conv.max_change_voxels > 0.0);
    }

    // ---------------------------------------------------------------- round trip

    /// Forward-warp a textured phantom with a smooth field, then unwarp it. The forward model is
    /// a scatter and the correction is a gather, so this is not a tautology; and the test
    /// requires the correction to remove most of the error it started with, so a no-op would
    /// fail it.
    #[test]
    fn a_scatter_forward_warp_round_trips_through_the_gather_correction() {
        let dims = (12, 32, 6);
        let (nx, ny, nz) = dims;
        let n = nx * ny * nz;
        let trt = 0.05;

        // A band-limited object: a smooth Gaussian envelope times a low-frequency texture, with
        // no hard edge anywhere. Splatting and then gathering is a *blur* as well as a shift --
        // a scatter spreads a sample over two bins and a gather point-samples them back, which
        // on a sharp edge costs far more than any geometric error. Keeping the object smooth
        // leaves the geometry as the thing this test is actually measuring.
        let truth_mag: Vec<f64> = (0..n)
            .map(|idx| {
                let (i, j, k) = unravel(idx, nx, ny);
                let (x, y, z) = (i as f64 / nx as f64, j as f64 / ny as f64, k as f64 / nz as f64);
                let r2 = ((x - 0.5) / 0.42).powi(2) + ((y - 0.5) / 0.42).powi(2)
                    + ((z - 0.5) / 0.45).powi(2);
                let envelope = (-3.0 * r2).exp();
                envelope * (1.0 + 0.3 * (std::f64::consts::TAU * x).sin()
                    * (std::f64::consts::TAU * y).cos())
            })
            .collect();
        let truth_phase: Vec<f64> = (0..n)
            .map(|idx| {
                let (i, j, _) = unravel(idx, nx, ny);
                let p = 0.25 * i as f64 - 0.18 * j as f64;
                p.sin().atan2(p.cos())
            })
            .collect();

        // Smooth field, peak ~1.9 voxels of shift, gentle enough not to fold.
        let field: Vec<f64> = (0..n)
            .map(|idx| {
                let (_, j, _) = unravel(idx, nx, ny);
                let y = (j as f64 - 0.5 * ny as f64) / (0.35 * ny as f64);
                38.0 * (-y * y).exp()
            })
            .collect();
        let disp = DisplacementField::from_field_hz(&field, dims, J_POS, trt);
        assert!(disp.max_shift_voxels() > 1.5, "the simulated distortion must be worth correcting");
        disp.check_invertible(None, 0.0).expect("the simulated field must not fold");

        let (dist_mag, dist_phase) = distort_complex(&truth_mag, &truth_phase, &disp);
        let out = unwarp_complex(&dist_mag, &dist_phase, &disp, None, &UnwarpParams::default())
            .unwrap();

        // Error measured over voxels the object actually occupies and that kept their data.
        let rms = |pred: &[f64]| {
            let (mut num, mut den, mut cnt) = (0.0, 0.0, 0usize);
            let peak = truth_mag.iter().fold(0.0f64, |m, &v| m.max(v));
            for i in 0..n {
                if truth_mag[i] < 0.05 * peak || out.mask[i] == 0 { continue; }
                num += (pred[i] - truth_mag[i]).powi(2);
                den += truth_mag[i].powi(2);
                cnt += 1;
            }
            assert!(cnt > 500, "only {cnt} voxels compared");
            (num / den).sqrt()
        };
        let before = rms(&dist_mag);
        let after = rms(&out.magnitude);
        // Measured: 0.269 distorted, 0.0202 corrected -- a 13-fold reduction. The residual is
        // the blur of splatting and then gathering, not a geometric error.
        assert!(before > 0.20, "the forward warp must actually distort (NRMSE {before})");
        assert!(after < 0.025, "corrected NRMSE {after} (uncorrected {before})");
        assert!(after < before / 10.0, "correction {after} vs distortion {before}");
    }
}

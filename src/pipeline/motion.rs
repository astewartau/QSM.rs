//! Motion-correction stage: co-register a multi-echo series onto one echo before field mapping.
//!
//! Runs before [`super::run_field_mapping`] and hands it the same shape of data it already takes,
//! so a pipeline that wants motion correction inserts this and changes nothing else. The
//! published procedure is exactly this: register every echo to the first TE as a fixed reference
//! with a rigid algorithm, then carry on.
//!
//! The library half lives in [`crate::motion`]; this is the stage wrapper that speaks
//! [`ScanMetadata`] and [`PipelineError`].
//!
//! # What it can and cannot fix
//!
//! It puts every echo's anatomy back in one place, which is what lets a cross-echo fit mean
//! anything at all. What it cannot undo is that the echoes were *acquired* with B0 in different
//! places relative to the head, so their dipole kernels genuinely differ:
//! [`MotionCorrectionResult::bdirs`] reports one direction per echo and
//! [`MotionCorrectionResult::rotation_spread_deg`] the angle between the extremes. A field map
//! fitted across them is a weighted average over that spread — the "residual boundary effect"
//! the UTE work reports after correcting substantial motion, and the reason this stage reports
//! the spread rather than quietly collapsing it. Below a degree or so it is negligible; a series
//! with a large spread is really a multi-orientation acquisition, and
//! [`crate::inversion::cosmos`] is the honest way to use it.

use super::config::{PipelineError, ScanMetadata};
use crate::motion::{estimate_motion, uniform_series, MotionParams, MotionSeries};

/// Configuration for the motion-correction stage.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug, Default)]
pub struct MotionCorrectionConfig {
    /// Run the stage at all. `false` passes the echoes through untouched, with identity motion
    /// reported, so a pipeline can keep one code path whether or not correction is wanted.
    pub enabled: bool,
    /// Everything about *how* to register. [`MotionParams::declared_b0`] is overridden by
    /// [`ScanMetadata::b0_direction`], which is where a pipeline's B0 direction actually comes
    /// from.
    pub motion: MotionParams,
}

/// A motion-corrected echo series, ready for [`super::run_field_mapping`].
pub struct MotionCorrectionResult {
    /// Per-echo wrapped phase on the reference echo's grid, moved through the complex domain so
    /// the wraps survived.
    pub phases: Vec<Vec<f64>>,
    /// Per-echo magnitude on the reference echo's grid.
    pub magnitudes: Vec<Vec<f64>>,
    /// The input mask restricted to voxels every echo still reaches after being moved.
    ///
    /// A rotated echo cannot fill the whole reference grid, and a resampled volume is zero where
    /// it does not reach. Those zeros are absence of data, not a zero measurement, and feeding
    /// them to an unwrapper as if they were brain is how a rim of nonsense gets into a field
    /// map. Pass **this** mask downstream, not the one handed in.
    pub mask: Vec<u8>,
    /// Each echo's B0 direction in the reference echo's voxel frame. See the module docs on why
    /// these differ and what that costs.
    pub bdirs: Vec<(f64, f64, f64)>,
    /// The motion that was estimated, for reporting and for the `suspect` flags.
    pub motion: MotionSeries,
}

impl MotionCorrectionResult {
    /// Angle between the extreme B0 directions in the series, in degrees — how far from a single
    /// dipole-kernel orientation this series actually is.
    pub fn rotation_spread_deg(&self) -> f64 {
        self.motion.rotation_spread_deg()
    }

    /// Largest distance any brain tissue moved during the series, in mm. Compare it against the
    /// voxel size: below one voxel the correction had nothing to do.
    pub fn max_displacement_mm(&self) -> f64 {
        self.motion.max_displacement_mm()
    }
}

/// Co-register a multi-echo series onto one of its echoes.
///
/// `phases` and `magnitudes` are per-echo and must be the same length; magnitude is required
/// here rather than optional, because registration is estimated on magnitude and there is
/// nothing to estimate it from without it. `mask` is the brain mask on the series' grid.
///
/// # Errors
///
/// [`PipelineError::InvalidInput`] for an empty series, a phase/magnitude count mismatch, or a
/// reference index that does not name an echo; [`PipelineError::DimensionMismatch`] if any volume
/// disagrees with `metadata.dims`; [`PipelineError::AlgorithmError`] if a registration could not
/// be evaluated at all (an empty mask, or volumes with no overlap).
pub fn run_motion_correction(
    phases: &[&[f64]],
    magnitudes: &[&[f64]],
    mask: &[u8],
    metadata: &ScanMetadata,
    config: &MotionCorrectionConfig,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<MotionCorrectionResult, PipelineError> {
    let (nx, ny, nz) = metadata.dims;
    let n_voxels = nx * ny * nz;
    let n_echoes = phases.len();

    if n_echoes == 0 {
        return Err(PipelineError::InvalidInput("no phase echoes provided".into()));
    }
    if magnitudes.len() != n_echoes {
        return Err(PipelineError::InvalidInput(format!(
            "{n_echoes} phase echoes but {} magnitude echoes",
            magnitudes.len()
        )));
    }
    for v in phases.iter().chain(magnitudes.iter()) {
        if v.len() != n_voxels {
            return Err(PipelineError::DimensionMismatch {
                expected: n_voxels,
                got: v.len(),
            });
        }
    }
    if mask.len() != n_voxels {
        return Err(PipelineError::DimensionMismatch {
            expected: n_voxels,
            got: mask.len(),
        });
    }
    let reference = config
        .motion
        .reference
        .resolve(n_echoes)
        .ok_or_else(|| {
            PipelineError::InvalidInput(format!(
                "reference {:?} does not name one of {n_echoes} echoes",
                config.motion.reference
            ))
        })?;

    // Every echo of one acquisition shares a grid, so one voxel->world affine serves the series.
    // A diagonal affine built from the voxel sizes is the right one: `ScanMetadata` carries the
    // B0 direction in *voxel* coordinates already, and passing it as the declared direction is
    // what makes the recovered per-echo directions come back in the same voxel frame the rest of
    // the pipeline's dipole kernels are built in. Deriving them from a fabricated affine instead
    // would silently assume an axial acquisition.
    let (vsx, vsy, vsz) = metadata.voxel_size;
    let affine = [
        vsx, 0.0, 0.0, 0.0, //
        0.0, vsy, 0.0, 0.0, //
        0.0, 0.0, vsz, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    ];
    let params = MotionParams {
        declared_b0: Some(metadata.b0_direction),
        ..config.motion.clone()
    };

    if !config.enabled {
        progress(1, 1);
        let series = uniform_series(magnitudes, metadata.dims, &affine);
        let motion = estimate_motion(
            &series[reference..reference + 1],
            Some(mask),
            &MotionParams {
                reference: crate::motion::MotionReference::First,
                ..params.clone()
            },
        )
        .ok_or_else(|| PipelineError::AlgorithmError("motion estimation failed".into()))?;
        // One identity entry, repeated: no registration ran, so every echo is where it was and
        // the reference echo's own direction is the only one there is.
        let bdir = motion.volumes[0].bdir;
        return Ok(MotionCorrectionResult {
            phases: phases.iter().map(|p| p.to_vec()).collect(),
            magnitudes: magnitudes.iter().map(|m| m.to_vec()).collect(),
            mask: mask.to_vec(),
            bdirs: vec![bdir; n_echoes],
            motion: MotionSeries {
                reference: 0,
                volumes: (0..n_echoes)
                    .map(|index| {
                        let mut v = motion.volumes[0].clone();
                        v.index = index;
                        v
                    })
                    .collect(),
            },
        });
    }

    progress(0, n_echoes + 1);
    let series = uniform_series(magnitudes, metadata.dims, &affine);
    let motion = estimate_motion(&series, Some(mask), &params)
        .ok_or_else(|| PipelineError::AlgorithmError("rigid registration failed".into()))?;
    progress(1, n_echoes + 1);

    let mut out_phases = Vec::with_capacity(n_echoes);
    let mut out_mags = Vec::with_capacity(n_echoes);
    let mut coverage = vec![1u8; n_voxels];
    for e in 0..n_echoes {
        let (m, p) = motion
            .apply_complex(e, magnitudes[e], phases[e])
            .ok_or_else(|| PipelineError::AlgorithmError(format!("resampling echo {e} failed")))?;
        let c = motion
            .coverage(e)
            .ok_or_else(|| PipelineError::AlgorithmError(format!("coverage for echo {e} failed")))?;
        for (a, &b) in coverage.iter_mut().zip(c.iter()) {
            *a &= b;
        }
        out_mags.push(m);
        out_phases.push(p);
        progress(e + 2, n_echoes + 1);
    }

    let out_mask: Vec<u8> = mask
        .iter()
        .zip(coverage.iter())
        .map(|(&m, &c)| m & c)
        .collect();

    Ok(MotionCorrectionResult {
        phases: out_phases,
        magnitudes: out_mags,
        bdirs: motion.bdirs(),
        mask: out_mask,
        motion,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motion::MotionReference;

    fn metadata(dims: (usize, usize, usize)) -> ScanMetadata {
        ScanMetadata {
            dims,
            voxel_size: (1.0, 1.0, 1.0),
            echo_times: vec![0.004, 0.012, 0.020, 0.028],
            field_strength: 3.0,
            b0_direction: (0.0, 0.0, 1.0),
        }
    }

    /// Two offset blobs, enough structure for a registration and cheap enough for a unit test.
    fn blobs(dims: (usize, usize, usize)) -> Vec<f64> {
        let (nx, ny, nz) = dims;
        let mut v = vec![0.0f64; nx * ny * nz];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let (x, y, z) = (
                        i as f64 / nx as f64,
                        j as f64 / ny as f64,
                        k as f64 / nz as f64,
                    );
                    let mut acc = 0.0;
                    for &(cx, cy, cz, r, val) in &[
                        (0.50, 0.50, 0.50, 0.30, 1.0),
                        (0.38, 0.44, 0.54, 0.10, 2.2),
                        (0.62, 0.58, 0.44, 0.09, 1.6),
                    ] {
                        let d = ((x - cx) / r).powi(2) + ((y - cy) / r).powi(2) + ((z - cz) / r).powi(2);
                        if d <= 1.0 {
                            acc += val * (1.0 - d).sqrt();
                        }
                    }
                    v[i + j * nx + k * nx * ny] = acc;
                }
            }
        }
        v
    }

    #[test]
    fn disabled_passes_the_series_through_untouched() {
        let dims = (20, 20, 20);
        let meta = metadata(dims);
        let mag = blobs(dims);
        let pha: Vec<f64> = (0..mag.len()).map(|i| (i as f64 * 0.37).sin() * 3.0).collect();
        let mask = vec![1u8; mag.len()];
        let mags: Vec<&[f64]> = vec![&mag; 4];
        let phas: Vec<&[f64]> = vec![&pha; 4];

        let out = run_motion_correction(
            &phas,
            &mags,
            &mask,
            &meta,
            &MotionCorrectionConfig::default(),
            &mut |_, _| {},
        )
        .unwrap();
        assert_eq!(out.phases.len(), 4);
        assert_eq!(out.magnitudes.len(), 4);
        assert_eq!(out.bdirs, vec![(0.0, 0.0, 1.0); 4]);
        assert_eq!(out.mask, mask, "a disabled stage must not shrink the mask");
        for e in 0..4 {
            assert_eq!(out.phases[e], pha, "echo {e} phase must be untouched");
            assert_eq!(out.magnitudes[e], mag, "echo {e} magnitude must be untouched");
            assert_eq!(out.motion.volumes[e].index, e);
            assert_eq!(out.motion.volumes[e].rotation_deg, 0.0);
        }
        assert_eq!(out.rotation_spread_deg(), 0.0);
        assert_eq!(out.max_displacement_mm(), 0.0);
    }

    #[test]
    fn bad_inputs_are_rejected() {
        let dims = (16, 16, 16);
        let meta = metadata(dims);
        let mag = blobs(dims);
        let mask = vec![1u8; mag.len()];
        let cfg = MotionCorrectionConfig {
            enabled: true,
            ..Default::default()
        };

        assert!(matches!(
            run_motion_correction(&[], &[], &mask, &meta, &cfg, &mut |_, _| {}),
            Err(PipelineError::InvalidInput(_))
        ));
        assert!(
            matches!(
                run_motion_correction(
                    &[mag.as_slice()], &[mag.as_slice(), mag.as_slice()],
                    &mask, &meta, &cfg, &mut |_, _| {},
                ),
                Err(PipelineError::InvalidInput(_))
            ),
            "phase and magnitude counts must agree"
        );
        assert!(
            matches!(
                run_motion_correction(
                    &[&mag[..mag.len() - 1]], &[&mag[..mag.len() - 1]],
                    &mask, &meta, &cfg, &mut |_, _| {},
                ),
                Err(PipelineError::DimensionMismatch { .. })
            ),
            "a volume that disagrees with metadata.dims"
        );
        assert!(
            matches!(
                run_motion_correction(
                    &[mag.as_slice()], &[mag.as_slice()],
                    &mask[..10], &meta, &cfg, &mut |_, _| {},
                ),
                Err(PipelineError::DimensionMismatch { .. })
            ),
            "a mask that disagrees with metadata.dims"
        );
        let out_of_range = MotionCorrectionConfig {
            enabled: true,
            motion: MotionParams {
                reference: MotionReference::Index(2),
                ..Default::default()
            },
        };
        assert!(
            matches!(
                run_motion_correction(
                    &[mag.as_slice()], &[mag.as_slice()],
                    &mask, &meta, &out_of_range, &mut |_, _| {},
                ),
                Err(PipelineError::InvalidInput(_))
            ),
            "a reference index past the end of the series"
        );
    }

    /// An enabled stage moves the echoes onto the reference's grid, shrinks the mask to what
    /// every echo still reaches, and reports one B0 direction per echo.
    #[test]
    fn an_enabled_stage_registers_and_shrinks_the_mask() {
        use crate::geometry::resample_complex_onto;

        let dims = (32, 32, 32);
        let meta = metadata(dims);
        let affine = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let mag = blobs(dims);
        let pha: Vec<f64> = (0..mag.len()).map(|i| (i as f64 * 0.37).sin() * 3.0).collect();
        let mask = vec![1u8; mag.len()];

        // Echo 1 is the reference; echoes 2-4 are shifted progressively along x, which takes a
        // slab off the edge of the grid and so must shrink the mask.
        let mut mags = vec![mag.clone()];
        let mut phas = vec![pha.clone()];
        for e in 1..4 {
            let shift = e as f64;
            let w = [
                1.0, 0.0, 0.0, shift, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 1.0,
            ];
            let (m, p) =
                resample_complex_onto(&mag, &pha, dims, &w, dims, &affine).unwrap();
            mags.push(m);
            phas.push(p);
        }
        let mag_refs: Vec<&[f64]> = mags.iter().map(|m| m.as_slice()).collect();
        let pha_refs: Vec<&[f64]> = phas.iter().map(|p| p.as_slice()).collect();

        let mut steps = Vec::new();
        let out = run_motion_correction(
            &pha_refs,
            &mag_refs,
            &mask,
            &meta,
            &MotionCorrectionConfig {
                enabled: true,
                ..Default::default()
            },
            &mut |a, b| steps.push((a, b)),
        )
        .unwrap();

        assert_eq!(out.bdirs.len(), 4);
        assert_eq!(steps.last(), Some(&(5, 5)), "progress should reach its total");
        let kept = out.mask.iter().filter(|&&m| m != 0).count();
        println!("mask kept {kept} of {}", mask.len());
        assert!(kept < mask.len(), "shifted echoes cannot cover the whole grid");
        assert!(kept > mask.len() / 2);

        // Pure translations, so every echo's B0 direction is still the declared one.
        for (e, b) in out.bdirs.iter().enumerate() {
            let err = (1.0f64 - b.2.abs()).abs();
            assert!(err < 1e-3, "echo {e}: bdir {b:?} should still be +z after a translation");
        }
        assert!(out.rotation_spread_deg() < 0.5, "spread {}", out.rotation_spread_deg());
        assert!(
            out.max_displacement_mm() > 2.0,
            "the series moved 3 mm, got {}",
            out.max_displacement_mm()
        );

        // And the corrected echoes agree with the reference, which is the point.
        for e in 1..4 {
            let worst = out.magnitudes[e]
                .iter()
                .zip(mag.iter())
                .zip(out.mask.iter())
                .filter(|(_, &m)| m != 0)
                .fold(0.0f64, |w, ((&got, &want), _)| w.max((got - want).abs()));
            println!("echo {e}: worst magnitude difference from the reference {worst:.4}");
            assert!(worst < 0.1, "echo {e} did not come back into register: {worst}");
        }
    }
}

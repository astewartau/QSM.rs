//! Per-echo QSM: a susceptibility map reconstructed from each echo on its own.
//!
//! The usual pipeline combines the echoes into one field ([`super::run_field_mapping`]) and
//! inverts that. A per-echo reconstruction ([`QsmPipelineConfig::qsm_per_echo`](super::config::QsmPipelineConfig::qsm_per_echo)) instead runs
//! every stage once per echo:
//!
//! 1. **Field**, [`run_field_mapping_per_echo`]: echo `e`'s phase unwrapped and divided by its own
//!    echo time. The echo combination, phase-offset removal and bipolar correction combine or
//!    compare echoes, so they do not run.
//! 2. **Background removal, inversion and referencing**: the host calls
//!    [`super::run_bg_removal`], [`super::run_dipole_inversion`] and [`super::apply_reference`]
//!    on that echo's field with [`ScanMetadata::echo`](super::config::ScanMetadata::echo)`(e)`, so a method that converts at an
//!    echo time (STAR-QSM's `te`, MEDI's, …) uses that echo's.
//! 3. **Outputs**: the per-echo χ maps, and their voxel-wise mean across echoes
//!    ([`mean_across_echoes`]) as the single QSM. [`voxel_major`] lays the per-echo maps out as
//!    [`SeparationInputs::qsm_per_echo`](super::SeparationInputs::qsm_per_echo) takes them.
//!
//! # Why, and the reference
//!
//! DECOMPOSE's signal model synthesises each echo's phase from that echo's χ(t): "Note that QSM
//! of each echo, χ(t), is used instead of the average QSM across all echoes" (Chen et al. 2021,
//! NeuroImage 242:118477, §2.2), and "it is necessary to use TE-dependent QSM as the input"
//! (§4.1). Their pipeline (§3.6, and `DECOMPOSE_analysis.m`) is the one above: Laplacian
//! unwrapping of each echo's raw phase, V-SHARP, and STAR-QSM "to compute the susceptibility
//! maps for each echo". The pipeline rule `decompose.qsm-per-echo` recommends this mode when
//! DECOMPOSE is chosen.
//!
//! # The mean across echoes
//!
//! [`mean_across_echoes`] is the plain voxel-wise arithmetic mean with equal weights: the
//! "average QSM across all echoes" Chen et al. contrast χ(t) with (§2.2), which names no
//! weighting. Referencing commutes with it: the mean of maps each referenced to their mean over
//! one mask is referenced to its mean over that mask.

pub use super::field_mapping::run_field_mapping_per_echo;

/// The voxel-wise mean of `maps` (one map per echo, each `n_voxels` long), with equal weights.
/// See the [module docs](self#the-mean-across-echoes). Empty when `maps` is; panics if the maps'
/// lengths differ.
pub fn mean_across_echoes<M: AsRef<[f64]>>(maps: &[M]) -> Vec<f64> {
    let Some(first) = maps.first() else { return Vec::new() };
    let n = first.as_ref().len();
    let mut out = vec![0.0; n];
    for m in maps {
        let m = m.as_ref();
        assert_eq!(m.len(), n, "per-echo maps differ in length");
        for (o, &v) in out.iter_mut().zip(m) {
            *o += v;
        }
    }
    let k = maps.len() as f64;
    out.iter_mut().for_each(|o| *o /= k);
    out
}

/// `maps` (one per echo, each `n_voxels` long) interleaved voxel-major, `(n_voxels, n_echoes)`
/// with the echo fastest, as [`SeparationInputs::qsm_per_echo`](super::SeparationInputs::qsm_per_echo)
/// and the multi-echo magnitude take them. Panics if the maps' lengths differ.
pub fn voxel_major<M: AsRef<[f64]>>(maps: &[M]) -> Vec<f64> {
    let ne = maps.len();
    let n = maps.first().map_or(0, |m| m.as_ref().len());
    let mut out = vec![0.0; n * ne];
    for (e, m) in maps.iter().enumerate() {
        let m = m.as_ref();
        assert_eq!(m.len(), n, "per-echo maps differ in length");
        for (v, &x) in m.iter().enumerate() {
            out[v * ne + e] = x;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::config::ScanMetadata;
    use crate::pipeline::config::{FieldMappingConfig, UnwrappingAlgorithm};
    use crate::pipeline::run_field_mapping;

    #[test]
    fn mean_and_layout() {
        let maps = [vec![1.0, 2.0, 3.0], vec![3.0, 4.0, 5.0]];
        assert_eq!(mean_across_echoes(&maps), vec![2.0, 3.0, 4.0]);
        assert_eq!(voxel_major(&maps), vec![1.0, 3.0, 2.0, 4.0, 3.0, 5.0]);
        assert!(mean_across_echoes::<Vec<f64>>(&[]).is_empty());
    }

    #[test]
    fn echo_metadata_keeps_one_echo() {
        let m = ScanMetadata {
            dims: (2, 2, 2), voxel_size: (1.0, 1.0, 1.0), echo_times: vec![0.004, 0.008, 0.012],
            field_strength: 3.0, b0_direction: (0.0, 0.0, 1.0), slice_geometry: None, affine: None,
        };
        let e = m.echo(1);
        assert_eq!(e.echo_times, vec![0.008]);
        assert_eq!((e.dims, e.field_strength), (m.dims, m.field_strength));
    }

    /// Each echo's field is that echo run alone through [`run_field_mapping`], whatever the echo
    /// combination is set to (here a robust fit, which one echo could not run).
    #[test]
    fn each_echo_is_a_single_echo_field() {
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        let tes = [0.005, 0.010, 0.015];
        let metadata = ScanMetadata {
            dims: (nx, ny, nz), voxel_size: (1.0, 1.0, 1.0), echo_times: tes.to_vec(),
            field_strength: 3.0, b0_direction: (0.0, 0.0, 1.0), slice_geometry: None, affine: None,
        };
        let freq: Vec<f64> = (0..n).map(|i| 20.0 * ((i % nx) as f64 - 3.5)).collect(); // rad/s
        let phases: Vec<Vec<f64>> = tes.iter().map(|&t| freq.iter().map(|f| f * t).collect()).collect();
        let refs: Vec<&[f64]> = phases.iter().map(|p| p.as_slice()).collect();
        let mask = vec![1u8; n];
        for alg in [UnwrappingAlgorithm::Romeo, UnwrappingAlgorithm::Laplacian] {
            let single = FieldMappingConfig { unwrapping_algorithm: alg, ..Default::default() };
            let config = FieldMappingConfig {
                b0_estimation: crate::pipeline::config::B0EstimationMethod::RobustFit(Default::default()),
                phase_offset_removal: true,
                ..single.clone()
            };
            let per_echo = run_field_mapping_per_echo(&refs, None, &mask, &metadata, &config, &mut |_, _| {}).unwrap();
            assert_eq!(per_echo.len(), tes.len());
            for (e, r) in per_echo.iter().enumerate() {
                let alone = run_field_mapping(&refs[e..=e], None, &mask, &metadata.echo(e), &single, &mut |_, _| {}).unwrap();
                assert_eq!(r.b0_field_ppm, alone.b0_field_ppm, "{alg:?} echo {e}");
            }
        }
    }
}

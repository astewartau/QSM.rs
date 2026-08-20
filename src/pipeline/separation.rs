//! Susceptibility source-separation stage.
//!
//! Config-driven dispatcher over the χ-separation methods (χ+ / χ−). Unlike the
//! main phase→field→BFR→inversion pipeline, separation is a downstream stage that
//! needs extra inputs (a conventional QSM plus a relaxometry map / multi-echo
//! magnitude), so it is invoked explicitly with a [`SeparationInputs`] bundle
//! rather than folded into the top-level runner.
//!
//! Units follow the library convention: local field & QSM in ppm, R2'/R2* in Hz,
//! echo times in seconds. Output is `(chi_pos ≥ 0, chi_neg ≤ 0, chi_total)` in ppm.

use super::config::*;
use crate::separation::{
    ChiSepIlsqrParams, ChiSepParams, DecomposeParams, HcChisepParams, R2starQsmParams,
};

/// Proton gyromagnetic ratio in Hz/T (for the central frequency `cf = γ·B0`).
const GAMMA_HZ_PER_T: f64 = 42.576e6;

/// Inputs for [`run_separation`]. Provide whatever the chosen algorithm needs;
/// the dispatcher errors if a required input is missing.
#[derive(Clone, Copy)]
pub struct SeparationInputs<'a> {
    /// Local (tissue) field in ppm — for `ChiSepIlsqr` / `ChiSepMedi`.
    pub local_field_ppm: &'a [f64],
    /// Conventional QSM χ_total in ppm — the QSM-init for `ChiSepIlsqr` and the
    /// input for `R2starQsm` / `WaveSep` / `Decompose` / `HcChisep`.
    pub qsm: &'a [f64],
    /// Binary brain mask (`n_voxels`, 1 = inside).
    pub mask: &'a [u8],
    /// R2' map in Hz (`ChiSepIlsqr` / `ChiSepMedi` / `WaveSep` / `HcChisep`).
    pub r2prime: Option<&'a [f64]>,
    /// R2* map in Hz (`R2starQsm`); if absent, fit from `magnitude_multi`.
    pub r2star: Option<&'a [f64]>,
    /// Root-sum-of-squares magnitude (`ChiSepIlsqr` / `ChiSepMedi`).
    pub magnitude_rss: Option<&'a [f64]>,
    /// Multi-echo magnitude, voxel-major `(n_voxels, n_echoes)`
    /// (`R2starQsm` / `Decompose` / `HcChisep`).
    pub magnitude_multi: Option<&'a [f64]>,
    /// Multi-echo spin-echo magnitude, voxel-major `(n_voxels, n_se)` (`HcChisep`, optional).
    pub se_magnitude_multi: Option<&'a [f64]>,
}

/// Result of [`run_separation`]: paramagnetic / diamagnetic / total maps in ppm.
#[derive(Clone, Debug)]
pub struct SeparationResult {
    /// χ+ ≥ 0 (paramagnetic).
    pub chi_pos: Vec<f64>,
    /// χ− ≤ 0 (diamagnetic, signed).
    pub chi_neg: Vec<f64>,
    /// χ_total = χ+ + χ−.
    pub chi_total: Vec<f64>,
}

/// Run the configured χ-separation algorithm.
///
/// # Arguments
/// * `inputs` — See [`SeparationInputs`]; provide what the algorithm requires.
/// * `metadata` — Scan metadata (grid, B0 direction, field strength, echo times).
/// * `config` — See [`SeparationConfig`].
/// * `progress` — Progress callback `(current_iter, max_iter)`.
pub fn run_separation(
    inputs: SeparationInputs,
    metadata: &ScanMetadata,
    config: &SeparationConfig,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<SeparationResult, PipelineError> {
    let grid = metadata.grid();
    let n = grid.n_total();
    let bdir = metadata.b0_direction;
    let cf = GAMMA_HZ_PER_T * metadata.field_strength;
    let b0 = metadata.field_strength;

    if inputs.qsm.len() != n || inputs.mask.len() != n {
        return Err(PipelineError::DimensionMismatch { expected: n, got: inputs.qsm.len() });
    }
    let mask = inputs.mask;

    let (chi_pos, chi_neg, chi_total) = match config.algorithm {
        SeparationAlgorithm::ChiSepIlsqr => {
            let r2prime = need(inputs.r2prime, "r2prime")?;
            let magnitude = need(inputs.magnitude_rss, "magnitude_rss")?;
            let params = ChiSepIlsqrParams { cf, ..config.chi_sep_ilsqr.clone() };
            crate::separation::chi_sep_ilsqr(
                inputs.local_field_ppm, r2prime, magnitude, inputs.qsm, mask,
                &grid, bdir, &params, |i, k| progress(i, k),
            )
        }
        SeparationAlgorithm::ChiSepMedi => {
            let r2prime = need(inputs.r2prime, "r2prime")?;
            let magnitude = need(inputs.magnitude_rss, "magnitude_rss")?;
            let params = ChiSepParams { cf, ..config.chi_sep_medi.clone() };
            crate::separation::chi_sep_medi(
                inputs.local_field_ppm, r2prime, magnitude, mask,
                &grid, bdir, &params, |i, k| progress(i, k),
            )
        }
        SeparationAlgorithm::R2starQsm => {
            let params = R2starQsmParams { b0, ..config.r2star_qsm.clone() };
            if let Some(r2star) = inputs.r2star {
                crate::separation::r2star_qsm(inputs.qsm, r2star, mask, &params)
            } else {
                let magnitude = need(inputs.magnitude_multi, "r2star or magnitude_multi")?;
                crate::separation::r2star_qsm_from_magnitude(
                    magnitude, &metadata.echo_times, inputs.qsm, mask, &params,
                )
            }
        }
        SeparationAlgorithm::WaveSep => {
            let r2prime = need(inputs.r2prime, "r2prime")?;
            crate::separation::wavesep(
                inputs.qsm, r2prime, mask, &grid, &config.wavesep, |i, k| progress(i, k),
            )
        }
        SeparationAlgorithm::Decompose => {
            let magnitude = need(inputs.magnitude_multi, "magnitude_multi")?;
            let params = DecomposeParams { b0, ..config.decompose.clone() };
            crate::separation::decompose(
                inputs.qsm, magnitude, &metadata.echo_times, mask, &params, |i, k| progress(i, k),
            )
        }
        SeparationAlgorithm::HcChisep => {
            let r2prime = need(inputs.r2prime, "r2prime")?;
            let magnitude = need(inputs.magnitude_multi, "magnitude_multi")?;
            let params = HcChisepParams { b0, ..config.hc_chisep.clone() };
            crate::separation::hc_chisep(
                inputs.qsm, r2prime, magnitude, &metadata.echo_times,
                inputs.se_magnitude_multi, mask, &grid, &params, |i, k| progress(i, k),
            )
        }
        SeparationAlgorithm::SusepNet => {
            let r2prime = need(inputs.r2prime, "r2prime")?;
            run_susep_net(inputs.local_field_ppm, inputs.qsm, r2prime, mask, &grid)?
        }
    };

    Ok(SeparationResult { chi_pos, chi_neg, chi_total })
}

/// Source the SUSEP-Net weights and run inference (requires the `onnx` feature).
#[cfg(feature = "onnx")]
fn run_susep_net(
    local_field_ppm: &[f64],
    qsm: &[f64],
    r2prime: &[f64],
    mask: &[u8],
    grid: &crate::Grid,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), PipelineError> {
    let spec = crate::models::find_model("susep-net")
        .ok_or_else(|| PipelineError::InvalidConfig("susep-net not in model registry".into()))?;
    let bytes = crate::models::primary_weight_bytes(spec).map_err(PipelineError::InvalidConfig)?;
    crate::separation::susep_net(
        local_field_ppm, qsm, r2prime, mask, grid, &bytes,
        &crate::separation::SusepNetNorm::default(),
    )
    .map_err(|e| PipelineError::AlgorithmError(e.to_string()))
}

#[cfg(not(feature = "onnx"))]
fn run_susep_net(
    _local_field_ppm: &[f64],
    _qsm: &[f64],
    _r2prime: &[f64],
    _mask: &[u8],
    _grid: &crate::Grid,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), PipelineError> {
    Err(PipelineError::InvalidConfig(
        "SUSEP-Net requires building qsm-core with the 'onnx' feature".into(),
    ))
}

/// Require an optional input, or return an `InvalidInput` error naming it.
fn need<'a>(opt: Option<&'a [f64]>, what: &str) -> Result<&'a [f64], PipelineError> {
    opt.ok_or_else(|| PipelineError::InvalidInput(format!("{what} required for this algorithm")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(nx: usize, ny: usize, nz: usize) -> ScanMetadata {
        ScanMetadata {
            dims: (nx, ny, nz),
            voxel_size: (1.0, 1.0, 1.0),
            echo_times: vec![0.004, 0.012, 0.020, 0.028],
            field_strength: 3.0,
            b0_direction: (0.0, 0.0, 1.0),
        }
    }

    fn base_inputs<'a>(qsm: &'a [f64], mask: &'a [u8]) -> SeparationInputs<'a> {
        SeparationInputs {
            local_field_ppm: qsm,
            qsm,
            mask,
            r2prime: None,
            r2star: None,
            magnitude_rss: None,
            magnitude_multi: None,
            se_magnitude_multi: None,
        }
    }

    #[test]
    fn r2star_qsm_dispatch_closed_form() {
        let (nx, ny, nz) = (6, 6, 6);
        let n = nx * ny * nz;
        let qsm = vec![0.02; n];
        let r2star = vec![10.0; n];
        let mask = vec![1u8; n];
        let m = meta(nx, ny, nz);
        let mut inputs = base_inputs(&qsm, &mask);
        inputs.r2star = Some(&r2star);
        let cfg = SeparationConfig { algorithm: SeparationAlgorithm::R2starQsm, ..Default::default() };
        let r = run_separation(inputs, &m, &cfg, &mut |_, _| {}).unwrap();
        assert_eq!(r.chi_pos.len(), n);
        for i in 0..n {
            assert!(r.chi_pos[i] >= 0.0 && r.chi_neg[i] <= 0.0);
            assert!((r.chi_total[i] - (r.chi_pos[i] + r.chi_neg[i])).abs() < 1e-9);
        }
    }

    #[test]
    fn wavesep_dispatch_runs() {
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        let qsm = vec![0.01; n];
        let r2prime = vec![2.0; n];
        let mask = vec![1u8; n];
        let m = meta(nx, ny, nz);
        let mut inputs = base_inputs(&qsm, &mask);
        inputs.r2prime = Some(&r2prime);
        let cfg = SeparationConfig { algorithm: SeparationAlgorithm::WaveSep, ..Default::default() };
        let r = run_separation(inputs, &m, &cfg, &mut |_, _| {}).unwrap();
        assert_eq!(r.chi_pos.len(), n);
    }

    #[test]
    fn chi_sep_ilsqr_and_medi_dispatch() {
        let (nx, ny, nz) = (6, 6, 6);
        let n = nx * ny * nz;
        let field = vec![0.01; n];
        let r2prime = vec![2.0; n];
        let mag = vec![1.0; n];
        let mask = vec![1u8; n];
        let m = meta(nx, ny, nz);
        for alg in [SeparationAlgorithm::ChiSepIlsqr, SeparationAlgorithm::ChiSepMedi] {
            let mut inputs = base_inputs(&field, &mask);
            inputs.r2prime = Some(&r2prime);
            inputs.magnitude_rss = Some(&mag);
            let cfg = SeparationConfig { algorithm: alg, ..Default::default() };
            let r = run_separation(inputs, &m, &cfg, &mut |_, _| {}).unwrap();
            assert_eq!(r.chi_pos.len(), n);
        }
    }

    #[test]
    fn decompose_and_hc_dispatch() {
        let (nx, ny, nz) = (6, 6, 6);
        let n = nx * ny * nz;
        let ne = 4;
        let qsm = vec![0.01; n];
        let r2prime = vec![2.0; n];
        let mag_multi = vec![1.0; n * ne];
        let mask = vec![1u8; n];
        let m = meta(nx, ny, nz);

        let mut d_inputs = base_inputs(&qsm, &mask);
        d_inputs.magnitude_multi = Some(&mag_multi);
        let d_cfg = SeparationConfig { algorithm: SeparationAlgorithm::Decompose, ..Default::default() };
        let rd = run_separation(d_inputs, &m, &d_cfg, &mut |_, _| {}).unwrap();
        assert_eq!(rd.chi_pos.len(), n);

        let mut h_inputs = base_inputs(&qsm, &mask);
        h_inputs.r2prime = Some(&r2prime);
        h_inputs.magnitude_multi = Some(&mag_multi);
        let h_cfg = SeparationConfig { algorithm: SeparationAlgorithm::HcChisep, ..Default::default() };
        let rh = run_separation(h_inputs, &m, &h_cfg, &mut |_, _| {}).unwrap();
        assert_eq!(rh.chi_pos.len(), n);
    }

    #[test]
    fn r2star_from_magnitude_path() {
        let (nx, ny, nz) = (6, 6, 6);
        let n = nx * ny * nz;
        let ne = 4;
        let qsm = vec![0.02; n];
        let mag_multi = vec![1.0; n * ne];
        let mask = vec![1u8; n];
        let m = meta(nx, ny, nz);
        // No r2star provided → fit from magnitude_multi.
        let mut inputs = base_inputs(&qsm, &mask);
        inputs.magnitude_multi = Some(&mag_multi);
        let cfg = SeparationConfig { algorithm: SeparationAlgorithm::R2starQsm, ..Default::default() };
        let r = run_separation(inputs, &m, &cfg, &mut |_, _| {}).unwrap();
        assert_eq!(r.chi_pos.len(), n);
    }

    #[test]
    fn missing_input_errors() {
        let (nx, ny, nz) = (6, 6, 6);
        let n = nx * ny * nz;
        let qsm = vec![0.02; n];
        let mask = vec![1u8; n];
        let m = meta(nx, ny, nz);
        // R2starQsm with neither r2star nor magnitude_multi → InvalidInput.
        let inputs = base_inputs(&qsm, &mask);
        let cfg = SeparationConfig { algorithm: SeparationAlgorithm::R2starQsm, ..Default::default() };
        let err = run_separation(inputs, &m, &cfg, &mut |_, _| {}).unwrap_err();
        assert!(matches!(err, PipelineError::InvalidInput(_)), "got {err:?}");
    }
}

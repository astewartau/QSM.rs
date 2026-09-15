//! Background field removal stage
//!
//! Dispatcher that calls the appropriate background removal algorithm
//! based on configuration, including algorithm-specific parameter defaults
//! (V-SHARP radii computation, LBV max_iter, iSMV radius).

use super::config::*;

/// Run background field removal on a total field map.
///
/// # Arguments
/// * `field_ppm` - Total field map in ppm
/// * `mask` - Binary brain mask
/// * `metadata` - Scan metadata
/// * `config` - Background removal configuration
/// * `progress` - Progress callback (current_iter, max_iter)
///
/// # Returns
/// `BgRemovalResult` with local field in ppm and eroded mask
pub fn run_bg_removal(
    field_ppm: &[f64],
    mask: &[u8],
    metadata: &ScanMetadata,
    config: &BgRemovalConfig,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<BgRemovalResult, PipelineError> {
    let grid = metadata.grid();
    let n_voxels = grid.n_total();

    if field_ppm.len() != n_voxels {
        return Err(PipelineError::DimensionMismatch {
            expected: n_voxels,
            got: field_ppm.len(),
        });
    }

    // mSMV needs the scan's field strength and echo time for its ppm↔radian cap.
    let msmv_scan = |base: crate::bgremove::MsmvParams| crate::bgremove::MsmvParams {
        b0: metadata.field_strength,
        te: metadata.echo_times.first().copied().unwrap_or(base.te),
        ..base
    };

    let (mut local_field, eroded_mask) = match config.algorithm {
        BgRemovalAlgorithm::Vsharp => {
            crate::bgremove::vsharp(
                field_ppm, mask, &grid, &config.vsharp, progress,
            )
        }
        BgRemovalAlgorithm::Pdf => {
            let local = crate::bgremove::pdf(
                field_ppm, mask, &grid,
                metadata.b0_direction, &config.pdf, progress,
            );
            (local, mask.to_vec())
        }
        BgRemovalAlgorithm::Lbv => {
            crate::bgremove::lbv(
                field_ppm, mask, &grid, &config.lbv, progress,
            )
        }
        BgRemovalAlgorithm::Ismv => {
            crate::bgremove::ismv(
                field_ppm, mask, &grid, &config.ismv, progress,
            )
        }
        BgRemovalAlgorithm::Sharp => {
            crate::bgremove::sharp(
                field_ppm, mask, &grid, &config.sharp,
            )
        }
        BgRemovalAlgorithm::Resharp => {
            crate::bgremove::resharp(
                field_ppm, mask, &grid,
                &config.resharp,
                progress,
            )
        }
        BgRemovalAlgorithm::Harperella => {
            crate::bgremove::harperella(
                field_ppm, mask, &grid,
                &config.harperella,
                progress,
            )
        }
        BgRemovalAlgorithm::Iharperella => {
            crate::bgremove::iharperella(
                field_ppm, mask, &grid,
                &config.harperella,
                progress,
            )
        }
        BgRemovalAlgorithm::Bfrnet => {
            let local = run_bfrnet(field_ppm, mask, &grid)?;
            (local, mask.to_vec())
        }
    };

    // Optional mSMV boundary-shadow refinement of the primary BFR's local field.
    // mSMV is a refinement (not a standalone primary remover), so this post-step
    // is the only way it's wired into the pipeline (Roberts 2024).
    if config.msmv_refine {
        let params = msmv_scan(crate::bgremove::MsmvParams {
            prefilter: false,
            ..config.msmv.clone()
        });
        let (refined, _) = crate::bgremove::msmv(&local_field, &eroded_mask, &grid, &params, |_, _| {});
        local_field = refined;
    }

    Ok(BgRemovalResult {
        local_field_ppm: local_field,
        eroded_mask,
    })
}

/// Source the BFRnet weights and run inference. Requires the `onnx` feature;
/// weights come from the model registry (local `$QSM_MODEL_DIR`/cache, or the
/// `download` feature).
#[cfg(feature = "onnx")]
fn run_bfrnet(
    field_ppm: &[f64],
    mask: &[u8],
    grid: &crate::Grid,
) -> Result<Vec<f64>, PipelineError> {
    let spec = crate::models::find_model("bfrnet")
        .ok_or_else(|| PipelineError::InvalidConfig("bfrnet not in model registry".into()))?;
    let bytes = crate::models::primary_weight_bytes(spec)
        .map_err(PipelineError::InvalidConfig)?;
    crate::bgremove::bfrnet(field_ppm, mask, grid, &bytes)
        .map_err(|e| PipelineError::AlgorithmError(e.to_string()))
}

#[cfg(not(feature = "onnx"))]
fn run_bfrnet(
    _field_ppm: &[f64],
    _mask: &[u8],
    _grid: &crate::Grid,
) -> Result<Vec<f64>, PipelineError> {
    Err(PipelineError::InvalidConfig(
        "BFRnet requires building qsm-core with the 'onnx' feature".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bg_removal_dispatches_vsharp() {
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        let field = vec![0.1; n];
        let mask = vec![1u8; n];
        let meta = ScanMetadata {
            dims: (nx, ny, nz),
            voxel_size: (1.0, 1.0, 1.0),
            echo_times: vec![0.005],
            field_strength: 3.0,
            b0_direction: (0.0, 0.0, 1.0),
        };

        let result = run_bg_removal(
            &field, &mask, &meta, &BgRemovalConfig::default(),
            &mut |_, _| {},
        );
        assert!(result.is_ok());
        let r = result.unwrap();
        assert_eq!(r.local_field_ppm.len(), n);
        assert_eq!(r.eroded_mask.len(), n);
    }

    #[test]
    fn test_bg_removal_dispatches_pdf() {
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        let field = vec![0.1; n];
        let mask = vec![1u8; n];
        let meta = ScanMetadata {
            dims: (nx, ny, nz), voxel_size: (1.0, 1.0, 1.0),
            echo_times: vec![0.005], field_strength: 3.0, b0_direction: (0.0, 0.0, 1.0),
        };
        let config = BgRemovalConfig { algorithm: BgRemovalAlgorithm::Pdf, ..Default::default() };
        let r = run_bg_removal(&field, &mask, &meta, &config, &mut |_, _| {}).unwrap();
        assert_eq!(r.local_field_ppm.len(), n);
    }

    /// PDF is the one background remover that builds a dipole kernel, so the
    /// pipeline must hand it `metadata.b0_direction` rather than assuming `+z`.
    /// Hardcoding `(0.0, 0.0, 1.0)` makes both calls below identical.
    #[test]
    fn test_bg_removal_pdf_uses_metadata_b0_direction() {
        let (nx, ny, nz) = (12, 12, 12);
        let n = nx * ny * nz;

        // Interior mask with real background voxels around it, so PDF has
        // background sources to project onto, plus a spatially varying field.
        let mut field = vec![0.0f64; n];
        let mut mask = vec![0u8; n];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = i + j * nx + k * nx * ny;
                    let (x, y, z) = (i as f64 - 5.5, j as f64 - 5.5, k as f64 - 5.5);
                    if x * x + y * y + z * z < 9.0 {
                        mask[idx] = 1;
                    }
                    field[idx] = 0.01 * (0.3 * x + 0.5 * y - 0.7 * z + 0.02 * x * y * z);
                }
            }
        }

        let config = BgRemovalConfig { algorithm: BgRemovalAlgorithm::Pdf, ..Default::default() };
        let run = |bdir: (f64, f64, f64)| {
            let meta = ScanMetadata {
                dims: (nx, ny, nz), voxel_size: (1.0, 1.0, 1.0),
                echo_times: vec![0.005], field_strength: 3.0, b0_direction: bdir,
            };
            run_bg_removal(&field, &mask, &meta, &config, &mut |_, _| {}).unwrap().local_field_ppm
        };

        let axial = run((0.0, 0.0, 1.0));
        // ~30 degrees off +z in the y-z plane.
        let theta = std::f64::consts::FRAC_PI_6;
        let oblique = run((0.0, theta.sin(), theta.cos()));

        assert_eq!(axial.len(), n);
        assert_eq!(oblique.len(), n);

        let max_diff = axial
            .iter()
            .zip(&oblique)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_diff > 1e-9,
            "oblique B0 gave the same local field as axial (max diff {max_diff:e}); \
             b0_direction is not reaching the PDF dipole kernel",
        );
    }

    #[test]
    fn test_bg_removal_dispatches_lbv() {
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        let field = vec![0.1; n];
        let mask = vec![1u8; n];
        let meta = ScanMetadata {
            dims: (nx, ny, nz), voxel_size: (1.0, 1.0, 1.0),
            echo_times: vec![0.005], field_strength: 3.0, b0_direction: (0.0, 0.0, 1.0),
        };
        let config = BgRemovalConfig { algorithm: BgRemovalAlgorithm::Lbv, ..Default::default() };
        let r = run_bg_removal(&field, &mask, &meta, &config, &mut |_, _| {}).unwrap();
        assert_eq!(r.local_field_ppm.len(), n);
    }

    #[test]
    fn test_bg_removal_dispatches_sharp() {
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        let field = vec![0.1; n];
        let mask = vec![1u8; n];
        let meta = ScanMetadata {
            dims: (nx, ny, nz), voxel_size: (1.0, 1.0, 1.0),
            echo_times: vec![0.005], field_strength: 3.0, b0_direction: (0.0, 0.0, 1.0),
        };
        let config = BgRemovalConfig { algorithm: BgRemovalAlgorithm::Sharp, ..Default::default() };
        let r = run_bg_removal(&field, &mask, &meta, &config, &mut |_, _| {}).unwrap();
        assert_eq!(r.local_field_ppm.len(), n);
    }

    #[test]
    fn test_bg_removal_dispatches_ismv() {
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        let field = vec![0.1; n];
        let mask = vec![1u8; n];
        let meta = ScanMetadata {
            dims: (nx, ny, nz), voxel_size: (1.0, 1.0, 1.0),
            echo_times: vec![0.005], field_strength: 3.0, b0_direction: (0.0, 0.0, 1.0),
        };
        let config = BgRemovalConfig { algorithm: BgRemovalAlgorithm::Ismv, ..Default::default() };
        let r = run_bg_removal(&field, &mask, &meta, &config, &mut |_, _| {}).unwrap();
        assert_eq!(r.local_field_ppm.len(), n);
    }

    #[test]
    fn test_bg_removal_msmv_refine_post_step() {
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        let field = vec![0.1; n];
        let mask = vec![1u8; n];
        let meta = ScanMetadata {
            dims: (nx, ny, nz), voxel_size: (1.0, 1.0, 1.0),
            echo_times: vec![0.005], field_strength: 3.0, b0_direction: (0.0, 0.0, 1.0),
        };
        // Primary V-SHARP + mSMV boundary-shadow refinement.
        let config = BgRemovalConfig {
            algorithm: BgRemovalAlgorithm::Vsharp,
            msmv_refine: true,
            ..Default::default()
        };
        let r = run_bg_removal(&field, &mask, &meta, &config, &mut |_, _| {}).unwrap();
        assert_eq!(r.local_field_ppm.len(), n);
    }

    #[test]
    fn test_bg_removal_dispatches_resharp() {
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        let field = vec![0.1; n];
        let mask = vec![1u8; n];
        let meta = ScanMetadata {
            dims: (nx, ny, nz), voxel_size: (1.0, 1.0, 1.0),
            echo_times: vec![0.005], field_strength: 3.0, b0_direction: (0.0, 0.0, 1.0),
        };
        let config = BgRemovalConfig { algorithm: BgRemovalAlgorithm::Resharp, ..Default::default() };
        let r = run_bg_removal(&field, &mask, &meta, &config, &mut |_, _| {}).unwrap();
        assert_eq!(r.local_field_ppm.len(), n);
    }

    #[test]
    fn test_bg_removal_validates_dims() {
        let meta = ScanMetadata {
            dims: (4, 4, 4),
            voxel_size: (1.0, 1.0, 1.0),
            echo_times: vec![0.005],
            field_strength: 3.0,
            b0_direction: (0.0, 0.0, 1.0),
        };
        let field = vec![0.0; 32]; // wrong size, should be 64
        let mask = vec![1u8; 64];

        let result = run_bg_removal(
            &field, &mask, &meta, &BgRemovalConfig::default(),
            &mut |_, _| {},
        );
        assert!(result.is_err());
    }
}

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
    let (vsx, vsy, vsz) = metadata.voxel_size;
    let n_voxels = grid.n_total();

    if field_ppm.len() != n_voxels {
        return Err(PipelineError::DimensionMismatch {
            expected: n_voxels,
            got: field_ppm.len(),
        });
    }

    let (local_field, eroded_mask) = match config.algorithm {
        BgRemovalAlgorithm::Vsharp => {
            let min_vox = vsx.min(vsy).min(vsz);
            let max_vox = vsx.max(vsy).max(vsz);
            let mut radii = Vec::new();
            let mut r = config.vsharp.max_radius_factor * min_vox;
            let step = config.vsharp.min_radius_factor * max_vox;
            while r >= step {
                radii.push(r);
                r -= step;
            }
            if radii.is_empty() {
                radii.push(config.vsharp.max_radius_factor * min_vox);
            }

            crate::bgremove::vsharp(
                field_ppm, mask, &grid,
                &radii, config.vsharp.threshold,
                progress,
            )
        }
        BgRemovalAlgorithm::Pdf => {
            let max_iter = (n_voxels as f64).sqrt() as usize;
            let local = crate::bgremove::pdf(
                field_ppm, mask, &grid,
                (0.0, 0.0, 1.0), config.pdf.tol, max_iter,
                progress,
            );
            (local, mask.to_vec())
        }
        BgRemovalAlgorithm::Lbv => {
            let (nx, ny, nz) = grid.dims;
            let max_iter = (3 * nx.max(ny).max(nz)).min(500);
            crate::bgremove::lbv(
                field_ppm, mask, &grid,
                config.lbv.tol, max_iter,
                progress,
            )
        }
        BgRemovalAlgorithm::Ismv => {
            let radius = config.ismv.radius_factor * vsx.max(vsy).max(vsz);
            crate::bgremove::ismv(
                field_ppm, mask, &grid,
                radius, config.ismv.tol, config.ismv.max_iter,
                progress,
            )
        }
        BgRemovalAlgorithm::Sharp => {
            let radius = config.sharp.radius_factor * vsx.min(vsy).min(vsz);
            crate::bgremove::sharp(
                field_ppm, mask, &grid,
                config.sharp.threshold, radius,
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
    };

    Ok(BgRemovalResult {
        local_field_ppm: local_field,
        eroded_mask,
    })
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

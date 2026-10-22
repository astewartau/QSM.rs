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
/// * `magnitude` - The magnitude of every echo, stacked with the echo slowest. PDF requires it
///   (its data weight is the `Fit_ppm_complex` noise map); the other methods ignore it.
/// * `progress` - Progress callback (current_iter, max_iter)
///
/// # Returns
/// `BgRemovalResult` with local field in ppm and eroded mask
pub fn run_bg_removal(
    field_ppm: &[f64],
    mask: &[u8],
    metadata: &ScanMetadata,
    config: &BgRemovalConfig,
    magnitude: Option<&[f64]>,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<BgRemovalResult, PipelineError> {
    // Every background removal here convolves with a kernel defined on the voxel grid, so a
    // volume with gaps between its slices is refused rather than approximated. See
    // ScanMetadata::require_contiguous_slices for why this is unconditional.
    metadata.require_contiguous_slices()?;
    let grid = metadata.grid();
    let n_voxels = grid.n_total();

    if field_ppm.len() != n_voxels {
        return Err(PipelineError::DimensionMismatch {
            expected: n_voxels,
            got: field_ppm.len(),
        });
    }

    let (local_field, eroded_mask) = match config.algorithm {
        BgRemovalAlgorithm::Vsharp => {
            if let crate::bgremove::VsharpGrid::Isotropic { spacing_mm } = config.vsharp.working_grid(&grid) {
                if !(spacing_mm.is_finite() && spacing_mm > 0.0) {
                    return Err(PipelineError::InvalidConfig(format!(
                        "V-SHARP isotropic working grid spacing must be finite and > 0 mm, got {spacing_mm}"
                    )));
                }
            }
            crate::bgremove::vsharp(
                field_ppm, mask, &grid, &config.vsharp, progress,
            )
            .map_err(|e| PipelineError::InvalidConfig(e.0))?
        }
        BgRemovalAlgorithm::Pdf => {
            // PDF weights the fit by 1/N_std, the noise of the field, which MEDI's pipeline takes
            // from Fit_ppm_complex (radians over the echo spacing, from the echo magnitudes). PDF
            // takes ppm like the others, and N_std must be in the field's unit (it sets the CG
            // tolerance), so it is converted with the factor that converts the field. That is
            // the echo spacing by the noise map's unit, not a choice of echo time: the noise is
            // that of the phase accrued over one echo spacing, and PDF itself is linear in the
            // field, so it has no radian conversion of its own to choose.
            let te = &metadata.echo_times;
            let mag = super::inversion::echo_magnitudes("PDF", magnitude, te, n_voxels)?;
            let delta_te = if te.len() == 1 { te[0] } else { te[1] - te[0] };
            let rad_per_ppm = 2.0 * std::f64::consts::PI * crate::constants::GAMMA_BAR_HZ_PER_T * metadata.field_strength * delta_te * 1e-6;
            let n_std: Vec<f64> = crate::utils::fit_ppm_complex_noise(mag.data, mag.n_echoes)
                .into_iter()
                .map(|v| v / rad_per_ppm)
                .collect();
            let local = crate::bgremove::pdf(
                field_ppm, &n_std, mask, &grid,
                metadata.b0_direction, &config.pdf, progress,
            )?;
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
        // HARPERELLA / iHARPERELLA are integrated unwrap + BFR methods, but this stage has the
        // unwrapped total field in ppm, so they take it as such (ordinary Laplacian). Wrapped
        // phase input is `HarperellaInput::WrappedPhase`, called directly.
        BgRemovalAlgorithm::Harperella => {
            crate::bgremove::harperella(
                crate::bgremove::HarperellaInput::FieldPpm(field_ppm), mask, &grid,
                &config.harperella,
                progress,
            )?
        }
        BgRemovalAlgorithm::Iharperella => {
            crate::bgremove::iharperella(
                crate::bgremove::HarperellaInput::FieldPpm(field_ppm), mask, &grid,
                &config.iharperella,
                progress,
            )?
        }
        BgRemovalAlgorithm::Bfrnet => {
            let local = run_bfrnet(field_ppm, mask, &grid, config.tile, progress)?;
            (local, mask.to_vec())
        }
    };

    Ok(BgRemovalResult {
        local_field_ppm: local_field,
        eroded_mask,
    })
}

/// Source the BFRnet weights and run inference, whole-volume or tiled (`tile`). Requires the
/// `onnx` feature; weights resolve through [`crate::models`] (bytes supplied with
/// [`crate::models::provide_weights`], then `$QSM_MODEL_DIR`/cache/download on native).
#[cfg(feature = "onnx")]
fn run_bfrnet(
    field_ppm: &[f64],
    mask: &[u8],
    grid: &crate::Grid,
    tile: Option<(usize, usize)>,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<Vec<f64>, PipelineError> {
    let spec = crate::models::find_model("bfrnet")
        .ok_or_else(|| PipelineError::InvalidConfig("bfrnet not in model registry".into()))?;
    let bytes = crate::models::primary_weight_bytes(spec)
        .map_err(PipelineError::InvalidConfig)?;
    match tile {
        Some((core, halo)) => crate::bgremove::bfrnet_tiled(
            field_ppm, mask, grid, &bytes, &crate::inversion::TileConfig { core, halo }, progress,
        ),
        None => crate::bgremove::bfrnet(field_ppm, mask, grid, &bytes),
    }
    .map_err(|e| PipelineError::AlgorithmError(e.to_string()))
}

#[cfg(not(feature = "onnx"))]
fn run_bfrnet(
    _field_ppm: &[f64],
    _mask: &[u8],
    _grid: &crate::Grid,
    _tile: Option<(usize, usize)>,
    _progress: &mut dyn FnMut(usize, usize),
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
            slice_geometry: None,
            affine: None,
        };

        let result = run_bg_removal(
            &field, &mask, &meta, &BgRemovalConfig::default(), None,
            &mut |_, _| {},
        );
        assert!(result.is_ok());
        let r = result.unwrap();
        assert_eq!(r.local_field_ppm.len(), n);
        assert_eq!(r.eroded_mask.len(), n);
    }

    #[test]
    fn vsharp_rejects_a_bad_working_grid_spacing() {
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        let meta = ScanMetadata {
            dims: (nx, ny, nz),
            voxel_size: (1.0, 1.0, 1.0),
            echo_times: vec![0.005],
            field_strength: 3.0,
            b0_direction: (0.0, 0.0, 1.0),
            slice_geometry: None,
            affine: None,
        };
        for spacing_mm in [0.0, -1.0, f64::NAN] {
            let mut config = BgRemovalConfig::default();
            config.vsharp.grid = crate::bgremove::VsharpGrid::Isotropic { spacing_mm };
            let r = run_bg_removal(&vec![0.1; n], &vec![1u8; n], &meta, &config, None, &mut |_, _| {});
            assert!(matches!(r, Err(PipelineError::InvalidConfig(_))), "spacing {spacing_mm}");
        }
    }

    /// The coverage test in `config.rs` reads the stage sources; this one actually calls the
    /// stage, so the wiring is proven end to end rather than by inspection.
    #[test]
    fn a_gapped_acquisition_is_refused_before_any_work() {
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        let field = vec![0.1; n];
        let mask = vec![1u8; n];
        let mut meta = ScanMetadata {
            dims: (nx, ny, nz),
            voxel_size: (1.0, 1.0, 3.0),
            echo_times: vec![0.005],
            field_strength: 3.0,
            b0_direction: (0.0, 0.0, 1.0),
            slice_geometry: None,
            affine: None,
        };

        // undeclared: nothing can be concluded, so it runs
        assert!(run_bg_removal(&field, &mask, &meta, &BgRemovalConfig::default(), None, &mut |_, _| {})
            .is_ok());

        // declared contiguous: runs
        meta.slice_geometry =
            Some(crate::grid::SliceGeometry { thickness: 3.0, axis: 2 });
        assert!(run_bg_removal(&field, &mask, &meta, &BgRemovalConfig::default(), None, &mut |_, _| {})
            .is_ok());

        // declared gapped: refused, for every algorithm the enum offers
        meta.slice_geometry =
            Some(crate::grid::SliceGeometry { thickness: 2.0, axis: 2 });
        for algorithm in [
            BgRemovalAlgorithm::Vsharp, BgRemovalAlgorithm::Pdf, BgRemovalAlgorithm::Lbv,
            BgRemovalAlgorithm::Ismv, BgRemovalAlgorithm::Sharp, BgRemovalAlgorithm::Resharp,
            BgRemovalAlgorithm::Harperella, BgRemovalAlgorithm::Iharperella,
        ] {
            let config = BgRemovalConfig { algorithm, ..BgRemovalConfig::default() };
            match run_bg_removal(&field, &mask, &meta, &config, None, &mut |_, _| {}) {
                Err(PipelineError::InvalidInput(msg)) => {
                    assert!(msg.contains("1.0000 mm gap"), "{algorithm:?}: {msg}");
                }
                Err(other) => panic!("{algorithm:?} gave {other:?}, not InvalidInput"),
                Ok(_) => panic!("{algorithm:?} accepted a gapped volume"),
            }
        }
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
            slice_geometry: None,
            affine: None,
        };
        let config = BgRemovalConfig { algorithm: BgRemovalAlgorithm::Pdf, ..Default::default() };
        let mag = vec![1.0; n];
        let r = run_bg_removal(&field, &mask, &meta, &config, Some(&mag), &mut |_, _| {}).unwrap();
        assert_eq!(r.local_field_ppm.len(), n);
    }

    /// PDF weights the data by the Fit_ppm_complex noise map, so it needs every echo's
    /// magnitude and says so rather than running unweighted.
    #[test]
    fn pdf_requires_the_echo_magnitudes() {
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        let meta = ScanMetadata {
            dims: (nx, ny, nz), voxel_size: (1.0, 1.0, 1.0),
            echo_times: vec![0.005, 0.01], field_strength: 3.0, b0_direction: (0.0, 0.0, 1.0),
            slice_geometry: None,
            affine: None,
        };
        let config = BgRemovalConfig { algorithm: BgRemovalAlgorithm::Pdf, ..Default::default() };
        let (field, mask) = (vec![0.1; n], vec![1u8; n]);
        for mag in [None, Some(vec![1.0; n])] {
            match run_bg_removal(&field, &mask, &meta, &config, mag.as_deref(), &mut |_, _| {}) {
                Err(PipelineError::InvalidInput(msg)) => {
                    assert!(msg.contains("PDF needs the magnitude of every echo (2 volumes"), "{msg}");
                }
                Err(other) => panic!("{other:?}"),
                Ok(_) => panic!("PDF ran without the echo magnitudes"),
            }
        }
        assert!(run_bg_removal(&field, &mask, &meta, &config, Some(&vec![1.0; 2 * n]), &mut |_, _| {}).is_ok());
    }

    /// The noise map reaches PDF in the field's unit: the pipeline gives the same local field as
    /// `pdf` called with `Fit_ppm_complex`'s N_std divided by the ppm-to-radian factor.
    #[test]
    fn pdf_gets_the_noise_map_in_ppm() {
        let (nx, ny, nz) = (12, 12, 12);
        let n = nx * ny * nz;
        let mut field = vec![0.0f64; n];
        let mut mask = vec![0u8; n];
        let mut mag = vec![0.0f64; 3 * n];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = i + j * nx + k * nx * ny;
                    let (x, y, z) = (i as f64 - 5.5, j as f64 - 5.5, k as f64 - 5.5);
                    mask[idx] = u8::from(x * x + y * y + z * z < 16.0);
                    field[idx] = 0.01 * (0.3 * x - 0.7 * z + 0.02 * x * y * z);
                    for e in 0..3 {
                        mag[idx + e * n] = (1.0 + 0.1 * x.abs()) * (1.0 - 0.2 * e as f64);
                    }
                }
            }
        }
        let meta = ScanMetadata {
            dims: (nx, ny, nz), voxel_size: (1.0, 1.0, 1.0),
            echo_times: vec![0.004, 0.008, 0.012], field_strength: 3.0, b0_direction: (0.0, 0.0, 1.0),
            slice_geometry: None,
            affine: None,
        };
        let config = BgRemovalConfig { algorithm: BgRemovalAlgorithm::Pdf, ..Default::default() };
        let got = run_bg_removal(&field, &mask, &meta, &config, Some(&mag), &mut |_, _| {}).unwrap();
        let rad_per_ppm = 2.0 * std::f64::consts::PI * crate::constants::GAMMA_BAR_HZ_PER_T * 3.0 * 0.004 * 1e-6;
        let n_std: Vec<f64> = crate::utils::fit_ppm_complex_noise(&mag, 3).iter().map(|v| v / rad_per_ppm).collect();
        let want = crate::bgremove::pdf(&field, &n_std, &mask, &meta.grid(), (0.0, 0.0, 1.0), &config.pdf, |_, _| {}).unwrap();
        assert_eq!(got.local_field_ppm, want);
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
                slice_geometry: None,
                affine: None,
            };
            // high SNR, so PDF's noise-based tolerance lets CG run
            let mag = vec![1e4; n];
            run_bg_removal(&field, &mask, &meta, &config, Some(&mag), &mut |_, _| {}).unwrap().local_field_ppm
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
            slice_geometry: None,
            affine: None,
        };
        let config = BgRemovalConfig { algorithm: BgRemovalAlgorithm::Lbv, ..Default::default() };
        let r = run_bg_removal(&field, &mask, &meta, &config, None, &mut |_, _| {}).unwrap();
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
            slice_geometry: None,
            affine: None,
        };
        let config = BgRemovalConfig { algorithm: BgRemovalAlgorithm::Sharp, ..Default::default() };
        let r = run_bg_removal(&field, &mask, &meta, &config, None, &mut |_, _| {}).unwrap();
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
            slice_geometry: None,
            affine: None,
        };
        let config = BgRemovalConfig { algorithm: BgRemovalAlgorithm::Ismv, ..Default::default() };
        let r = run_bg_removal(&field, &mask, &meta, &config, None, &mut |_, _| {}).unwrap();
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
            slice_geometry: None,
            affine: None,
        };
        let config = BgRemovalConfig { algorithm: BgRemovalAlgorithm::Resharp, ..Default::default() };
        let r = run_bg_removal(&field, &mask, &meta, &config, None, &mut |_, _| {}).unwrap();
        assert_eq!(r.local_field_ppm.len(), n);
    }

    /// iHARPERELLA reads its own params, not HARPERELLA's (#169).
    #[test]
    fn iharperella_uses_its_own_params() {
        let (nx, ny, nz) = (16, 16, 16);
        let n = nx * ny * nz;
        let meta = ScanMetadata {
            dims: (nx, ny, nz), voxel_size: (1.0, 1.0, 1.0),
            echo_times: vec![0.005], field_strength: 3.0, b0_direction: (0.0, 0.0, 1.0),
            slice_geometry: None,
            affine: None,
        };
        // a blob inside a 10-voxel cube: big enough to leave an interior after the ~3-voxel erosion
        let field: Vec<f64> = (0..n)
            .map(|i| {
                let (x, y, z) = ((i % nx) as f64 - 7.5, ((i / nx) % ny) as f64 - 7.5, (i / (nx * ny)) as f64 - 7.5);
                (-(x * x + y * y + z * z) / 10.0).exp() + 0.01 * x
            })
            .collect();
        let mask: Vec<u8> = (0..n)
            .map(|i| {
                let (x, y, z) = (i % nx, (i / nx) % ny, i / (nx * ny));
                u8::from((3..13).contains(&x) && (3..13).contains(&y) && (3..13).contains(&z))
            })
            .collect();
        let run = |config: &BgRemovalConfig| {
            run_bg_removal(&field, &mask, &meta, config, None, &mut |_, _| {}).unwrap().local_field_ppm
        };
        let base = BgRemovalConfig { algorithm: BgRemovalAlgorithm::Iharperella, ..Default::default() };
        let reference = run(&base);
        assert!(reference.iter().any(|v| v.abs() > 1e-6), "nothing came through");
        // HARPERELLA's settings must not reach iHARPERELLA
        let mut other = base.clone();
        other.harperella = crate::bgremove::HarperellaParams { radius: 2.0, max_iter: 1, tol: 0.5, pad: [3, 3, 3] };
        assert_eq!(run(&other), reference);
        // its own settings must
        let mut own = base.clone();
        own.iharperella.max_iter = 1;
        assert_ne!(run(&own), reference);
    }

    #[test]
    fn test_bg_removal_validates_dims() {
        let meta = ScanMetadata {
            dims: (4, 4, 4),
            voxel_size: (1.0, 1.0, 1.0),
            echo_times: vec![0.005],
            field_strength: 3.0,
            b0_direction: (0.0, 0.0, 1.0),
            slice_geometry: None,
            affine: None,
        };
        let field = vec![0.0; 32]; // wrong size, should be 64
        let mask = vec![1u8; 64];

        let result = run_bg_removal(
            &field, &mask, &meta, &BgRemovalConfig::default(), None,
            &mut |_, _| {},
        );
        assert!(result.is_err());
    }
}

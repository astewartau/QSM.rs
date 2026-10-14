//! xQSM through the pipeline dispatcher with weights supplied from memory
//! (`models::provide_weights`), the way a browser host runs it: no `$QSM_MODEL_DIR`, no cache,
//! no download.
//!
//! The weights are not vendored. The test reads `xqsm.onnx` from `$XQSM_ONNX`,
//! `$QSM_MODEL_DIR` or the qsm-rs cache, and skips (passing, with a note) when none has it.
//! This file holds a single test because it repoints the process-wide model directories.
#![cfg(feature = "onnx")]

use std::path::PathBuf;

use qsm_core::models;
use qsm_core::pipeline::config::{InversionAlgorithm, InversionConfig, ScanMetadata};

fn local_xqsm_bytes() -> Option<Vec<u8>> {
    if let Ok(p) = std::env::var("XQSM_ONNX") {
        return Some(std::fs::read(p).expect("read $XQSM_ONNX"));
    }
    let file = models::find_weight_file("xqsm.onnx").expect("xqsm.onnx in the registry");
    models::resolve_local(file).map(|p| std::fs::read(p).expect("read xqsm.onnx"))
}

#[test]
fn xqsm_dispatcher_runs_on_provided_weights() {
    let Some(bytes) = local_xqsm_bytes() else {
        eprintln!("skipping: xqsm.onnx not found ($XQSM_ONNX, $QSM_MODEL_DIR or the qsm-rs cache)");
        return;
    };

    // Nothing on disk: an empty model directory and cache, as in a browser.
    let empty: PathBuf =
        std::env::temp_dir().join(format!("qsm-xqsm-provider-{}", std::process::id()));
    std::fs::create_dir_all(&empty).unwrap();
    std::env::set_var("QSM_MODEL_DIR", &empty);
    std::env::set_var("QSM_MODEL_CACHE", &empty);

    // A 32³ phantom: a sphere with a smooth field inside a brain-like mask.
    let n = 32;
    let metadata = ScanMetadata {
        dims: (n, n, n),
        voxel_size: (1.0, 1.0, 1.0),
        echo_times: vec![0.02],
        field_strength: 3.0,
        b0_direction: (0.0, 0.0, 1.0),
        slice_geometry: None,
    };
    let c = (n as f64 - 1.0) / 2.0;
    let mut field = vec![0.0f64; n * n * n];
    let mut mask = vec![0u8; n * n * n];
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let (x, y, z) = (i as f64 - c, j as f64 - c, k as f64 - c);
                let r2 = x * x + y * y + z * z;
                let idx = i + n * (j + n * k);
                if r2 < 13.0 * 13.0 {
                    mask[idx] = 1;
                    // Dipole-like pattern around a small inclusion at the centre.
                    let r2 = r2.max(4.0);
                    field[idx] = 0.05 * (3.0 * z * z / r2 - 1.0) * (16.0 / r2).min(1.0);
                }
            }
        }
    }
    let config = InversionConfig {
        algorithm: InversionAlgorithm::Xqsm,
        ..Default::default()
    };

    // Without the provided bytes there is nothing to load (and, without `download`, no fetch).
    #[cfg(not(feature = "download"))]
    {
        let err = qsm_core::pipeline::inversion::run_dipole_inversion(
            &field,
            &mask,
            &metadata,
            &config,
            None,
            &mut |_, _| {},
        )
        .unwrap_err();
        assert!(err.to_string().contains("xqsm.onnx"), "{err}");
    }

    // The reference result, straight from the algorithm with the same bytes.
    let expected =
        qsm_core::inversion::xqsm(&field, &mask, &metadata.grid(), &bytes).expect("xqsm");

    models::provide_weights("xqsm.onnx", bytes).expect("registry hash matches");
    let chi = qsm_core::pipeline::inversion::run_dipole_inversion(
        &field,
        &mask,
        &metadata,
        &config,
        None,
        &mut |_, _| {},
    )
    .expect("xQSM through the dispatcher with provided weights");

    assert_eq!(chi.len(), n * n * n);
    assert!(chi.iter().all(|v| v.is_finite()));
    assert!(
        chi.iter().any(|&v| v != 0.0),
        "xQSM returned an all-zero map"
    );
    assert_eq!(
        chi, expected,
        "dispatcher result differs from a direct xqsm call on the same bytes"
    );

    assert!(models::forget_weights("xqsm.onnx"));
    let _ = std::fs::remove_dir_all(&empty);
}

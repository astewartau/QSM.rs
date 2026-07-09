//! Low-level pipeline: wire the algorithm building blocks together yourself.
//!
//! This mirrors how a downstream consumer (e.g. a WASM wrapper) drives the
//! individual algorithm functions, one stage at a time. Each algorithm takes a
//! [`Grid`], a `*Params` struct, and (for iterative methods) a progress closure.
//!
//! Run with: `cargo run --release --example pipeline_lowlevel`
//!
//! In real use, load `phase`/`magnitude` from NIfTI with [`qsm_core::io`].

use qsm_core::{Grid, bet, unwrap, bgremove, inversion};
use qsm_core::bet::BetParams;
use qsm_core::bgremove::VsharpParams;
use qsm_core::inversion::TvParams;

fn main() {
    let (nx, ny, nz) = (32, 32, 32);
    let grid = Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);
    let n = nx * ny * nz;
    let bdir = (0.0, 0.0, 1.0); // B0 along z

    // Synthetic inputs: a bright sphere (magnitude) with a smooth wrapped phase.
    let (mut magnitude, mut phase) = (vec![0.0f64; n], vec![0.0f64; n]);
    let center = [nx as f64 / 2.0, ny as f64 / 2.0, nz as f64 / 2.0];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let idx = i + j * nx + k * nx * ny;
                let r2 = (i as f64 - center[0]).powi(2)
                    + (j as f64 - center[1]).powi(2)
                    + (k as f64 - center[2]).powi(2);
                if r2 < 100.0 {
                    magnitude[idx] = 100.0;
                    phase[idx] = 0.5 * (i as f64 * 0.3).sin();
                }
            }
        }
    }

    // 1. Brain extraction (params tuned down for this tiny synthetic volume;
    //    use BetParams::default() on real data)
    let bet_params = BetParams { iterations: 100, subdivisions: 2, ..BetParams::default() };
    let mask = bet::run_bet(&magnitude, &grid, &bet_params, |_, _| {});

    // 2. Phase unwrapping (Laplacian: fast, closed-form, no progress callback)
    let unwrapped = unwrap::laplacian_unwrap(&phase, &mask, &grid);

    // 3. Background field removal (V-SHARP)
    let (local_field, eroded_mask) =
        bgremove::vsharp(&unwrapped, &mask, &grid, &VsharpParams::default(), |_, _| {});

    // 4. Dipole inversion (Total Variation via ADMM)
    let chi = inversion::tv_admm(
        &local_field, &eroded_mask, &grid, bdir, &TvParams::default(), |_, _| {},
    );

    let mean = chi.iter().sum::<f64>() / n as f64;
    println!("Reconstructed susceptibility map: {} voxels, mean = {:.6} ppm", chi.len(), mean);
}

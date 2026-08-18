//! Maximum Spherical Mean Value (mSMV) background field removal
//!
//! mSMV removes the residual harmonic background field near the brain boundary —
//! the dominant source of QSM shadow artifacts — using the maximum-value
//! corollary of Green's theorem, and does so **without eroding the brain mask**.
//!
//! ## Two ways to use it (the `prefilter` flag)
//! In the original work mSMV is a **refinement applied after a primary background
//! removal**, and that is its intended use:
//!
//! ```ignore
//! let (local, m) = bgremove::vsharp(&total, &mask, &grid, &VsharpParams::default(), |_,_| {});
//! let (refined, _) = bgremove::msmv(&local, &m, &grid, &MsmvParams::refine(), |_,_| {});
//! ```
//!
//! It composes with any harmonic BFR that leaves a boundary shadow —
//! [`sharp`](super::sharp), [`resharp`](super::resharp), [`vsharp`](super::vsharp),
//! [`pdf`](super::pdf), [`lbv`](super::lbv). (It is redundant after
//! [`ismv`](super::ismv), which is already iterative SMV.) In this mode set
//! [`MsmvParams::prefilter`] `= false` (via [`MsmvParams::refine`]) so mSMV does
//! **only** the boundary correction on the already-local field.
//!
//! With `prefilter = true` (the default, matching the QSM-CI submission) mSMV is a
//! self-contained total→local BFR: it first does its own SMV primary removal, then
//! the boundary correction. Run standalone it is only as strong as one non-
//! deconvolved SMV pass, so prefer the refinement mode above for real pipelines.
//!
//! The two internal steps:
//!   1. **SMV primary removal** (only if `prefilter`): `RDF_s = mask·(RDF − SMV(RDF))`
//!      (radius `radius` mm). When `prefilter` is false the input field is taken as
//!      the local field directly.
//!   2. **Boundary shadow correction**: iteratively detect boundary voxels whose
//!      residual field exceeds an adaptive threshold `t` and strip it with a
//!      minimum-radius SMV.
//!
//! ## Units
//! mSMV's shadow-detection threshold is defined in **radians** and capped at
//! `0.01·B0/3` rad. The library carries fields in **ppm**, so this port converts
//! ppm→rad with `rad = ppm · 2π · γ · B0 · TE` (γ in MHz/T), runs the filter,
//! then converts back. `TE` only sets the operating point of the radian
//! threshold; the filter is otherwise a (thresholded) linear high-pass.
//!
//! The optional vessel-protection step (which needs an R2* map) is omitted, per
//! the upstream behaviour when R2* is unavailable.
//!
//! Reference:
//! Roberts, A.G., et al. (2024). "Maximum spherical mean value (mSMV) filtering
//! for whole-brain quantitative susceptibility mapping." Magnetic Resonance in
//! Medicine, 91(4):1586-1597. https://doi.org/10.1002/mrm.29963
//!
//! Reference implementation: https://github.com/agr78/mSMV (`msmv.m`, MEDI SMV helpers)

use crate::fft::{apply_real_kernel, fft_real_kernel};
use crate::kernels::smv::smv_kernel;
use crate::Grid;

/// Proton gyromagnetic ratio in MHz/T (matches the QSM-CI `recon.m` constant).
const GYRO_MHZ_T: f64 = 42.5774;

/// Parameters for [`msmv`].
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct MsmvParams {
    /// SMV prefilter kernel radius in mm (paper default 5).
    pub radius: f64,
    /// Maximum number of residual-field removal iterations (paper default 5).
    pub maxk: usize,
    /// Main field strength in Tesla — sets the radian shadow-threshold cap `0.01·B0/3`.
    pub b0: f64,
    /// Echo time in seconds for the ppm↔radian conversion (MEDI-representative default).
    pub te: f64,
    /// Whether to run mSMV's own SMV primary removal before the boundary
    /// correction. `true` (default) = self-contained total→local BFR; `false` =
    /// refine an already-local field from a primary BFR (see [`MsmvParams::refine`]).
    pub prefilter: bool,
}

impl Default for MsmvParams {
    fn default() -> Self {
        Self { radius: 5.0, maxk: 5, b0: 3.0, te: 0.008, prefilter: true }
    }
}

impl MsmvParams {
    /// Parameters for using mSMV as a **boundary-shadow refinement** after a
    /// primary background-field removal (SHARP / RESHARP / V-SHARP / PDF / LBV):
    /// `prefilter = false`, all other fields at their defaults.
    pub fn refine() -> Self {
        Self { prefilter: false, ..Self::default() }
    }
}

/// mSMV background field removal (SMV primary removal + boundary shadow correction).
///
/// # Arguments
/// * `field` — Field in **ppm** (`nx·ny·nz`, column-major): the unwrapped total
///   field when `params.prefilter` is `true`, or an already-local field (from a
///   primary BFR) when `false`.
/// * `mask` — Binary brain mask (`nx·ny·nz`, 1 = inside).
/// * `grid` — Volume dimensions and voxel sizes.
/// * `params` — See [`MsmvParams`].
/// * `progress` — Progress callback `(iteration, max_iterations)` over the
///   boundary-correction loop.
///
/// # Returns
/// `(local_field_ppm, mask)` — the filtered local field in ppm restricted to
/// `mask`, and the mask unchanged (mSMV preserves the brain edge; it does not
/// erode). Same `(field, eroded_mask)` return shape as the other BFR methods.
pub fn msmv(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    params: &MsmvParams,
    mut progress: impl FnMut(usize, usize),
) -> (Vec<f64>, Vec<u8>) {
    let (nx, ny, nz) = grid.dims;
    let n = nx * ny * nz;
    assert_eq!(field.len(), n, "field length must match grid");
    assert_eq!(mask.len(), n, "mask length must match grid");

    // ppm -> radians: rad = ppm · 2π · γ(MHz/T) · B0 · TE  (the 1e6/1e-6 cancel).
    let ppm2rad = std::f64::consts::TAU * GYRO_MHZ_T * params.b0 * params.te;
    let rdf: Vec<f64> = field.iter().map(|&v| v * ppm2rad).collect();

    let maskf: Vec<f64> = mask.iter().map(|&m| m as f64).collect();

    // Prefilter SMV kernel (radius mm) and its real-valued FFT.
    let sphere_k = smv_kernel(grid, params.radius);
    let sphere_fft = fft_real_kernel(&sphere_k, nx, ny, nz);

    // Boundary shell Me = mask − erode(mask): voxels whose full sphere neighbourhood
    // is not entirely inside the mask (SMV(mask) ≤ 0.999).
    let smv_mask = apply_real_kernel(&maskf, &sphere_fft, nx, ny, nz);
    let me: Vec<f64> = (0..n)
        .map(|i| {
            let mne = if smv_mask[i] > 0.999 { 1.0 } else { 0.0 };
            maskf[i] - mne
        })
        .collect();

    // Step 1 — SMV primary removal: RDF_s = mask·(RDF − SMV(RDF)). Skipped in
    // refinement mode (`prefilter = false`), where the input is already a local
    // field from a primary BFR and only the boundary correction is wanted.
    let mut rdf_s: Vec<f64> = if params.prefilter {
        let smv_rdf = apply_real_kernel(&rdf, &sphere_fft, nx, ny, nz);
        (0..n).map(|i| maskf[i] * (rdf[i] - smv_rdf[i])).collect()
    } else {
        (0..n).map(|i| maskf[i] * rdf[i]).collect()
    };
    let rdf_s0 = rdf_s.clone();

    // Shadow-detection threshold t (radians). Upstream `kernel_lim.m` grows an
    // SMV kernel from ~0 until the max masked SMV-residual reaches the cap
    // `0.01·B0/3` and returns that value — i.e. t converges to the cap from
    // below. With the binary (non-rendered) SMV kernel used library-wide the
    // sub-voxel sweep is a no-op between voxel-radius thresholds and can never
    // exceed the cap, so we use the limit value directly.
    let t = 0.01 * params.b0 / 3.0;

    // Minimum-radius SMV kernel for the residual-removal step.
    let min_vox = grid.vsx().min(grid.vsy()).min(grid.vsz());
    let r2_radius = min_vox / 2.0 + 0.05;
    let small_k = smv_kernel(grid, r2_radius);
    let small_fft = fft_real_kernel(&small_k, nx, ny, nz);

    // Step 2 — iterative boundary shadow correction.
    // Initial boundary background estimate (|Me·RDF_s0| > t) sets the loop guard.
    let mut mb_count = (0..n).filter(|&i| (me[i] * rdf_s0[i]).abs() > t).count();
    let mask_count = mask.iter().filter(|&&m| m != 0).count().max(1);

    let mut k = 1usize;
    while (mb_count as f64) / (mask_count as f64) > 1e-6 {
        progress(k, params.maxk);
        // Boundary voxels whose residual field still exceeds the threshold.
        let mb: Vec<f64> = (0..n)
            .map(|i| if (me[i] * rdf_s[i]).abs() > t { 1.0 } else { 0.0 })
            .collect();
        mb_count = mb.iter().filter(|&&v| v > 0.0).count();

        // Strip the detected residual field with the minimum-radius SMV.
        let masked: Vec<f64> = (0..n).map(|i| mb[i] * rdf_s[i]).collect();
        let smv_masked = apply_real_kernel(&masked, &small_fft, nx, ny, nz);
        for i in 0..n {
            rdf_s[i] = maskf[i] * (rdf_s[i] - smv_masked[i]);
        }

        k += 1;
        if k > params.maxk.saturating_sub(1).max(1) {
            break;
        }
    }

    // radians -> ppm, restricted to the mask.
    let local_field: Vec<f64> = (0..n)
        .map(|i| if mask[i] != 0 { rdf_s[i] / ppm2rad } else { 0.0 })
        .collect();

    (local_field, mask.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(n: usize) -> Grid {
        Grid::new(n, n, n, 1.0, 1.0, 1.0)
    }

    /// A harmonic field (satisfying Laplace's equation) is annihilated by SMV, so
    /// mSMV should drive a purely-harmonic interior field toward zero.
    #[test]
    fn removes_harmonic_field() {
        let nn = 48;
        let g = grid(nn);
        let n = nn * nn * nn;
        // Harmonic background: f = x^2 - y^2 (∇²f = 0). Use centered coordinates.
        let mut field = vec![0.0f64; n];
        let mut mask = vec![0u8; n];
        let c = nn as f64 / 2.0;
        let r_mask = 16.0;
        for k in 0..nn {
            for j in 0..nn {
                for i in 0..nn {
                    let x = i as f64 - c;
                    let y = j as f64 - c;
                    let z = k as f64 - c;
                    let idx = i + j * nn + k * nn * nn;
                    // scale down to a ppm-like magnitude
                    field[idx] = 1e-3 * (x * x - y * y);
                    if (x * x + y * y + z * z).sqrt() < r_mask {
                        mask[idx] = 1;
                    }
                }
            }
        }
        let params = MsmvParams { radius: 4.0, maxk: 5, b0: 3.0, te: 0.008, prefilter: true };
        let (local, out_mask) = msmv(&field, &mask, &g, &params, |_, _| {});

        // Mask is preserved (mSMV does not erode).
        assert_eq!(out_mask, mask);

        // Interior (eroded) energy of the harmonic field should be strongly reduced.
        let inside: Vec<usize> = (0..n).filter(|&i| mask[i] == 1).collect();
        let in_rms = |a: &[f64]| {
            (inside.iter().map(|&i| a[i] * a[i]).sum::<f64>() / inside.len() as f64).sqrt()
        };
        let before = in_rms(&field);
        let after = in_rms(&local);
        assert!(
            after < 0.5 * before,
            "harmonic field should be reduced: before={before:.3e} after={after:.3e}"
        );
    }

    #[test]
    fn preserves_shapes_and_mask() {
        let g = grid(16);
        let n = 16 * 16 * 16;
        let field: Vec<f64> = (0..n).map(|i| ((i % 7) as f64 - 3.0) * 1e-3).collect();
        let mut mask = vec![0u8; n];
        for (i, m) in mask.iter_mut().enumerate() {
            // a solid central-ish block
            let x = i % 16;
            let y = (i / 16) % 16;
            let z = i / (16 * 16);
            if (4..12).contains(&x) && (4..12).contains(&y) && (4..12).contains(&z) {
                *m = 1;
            }
        }
        let (local, out_mask) = msmv(&field, &mask, &g, &MsmvParams::default(), |_, _| {});
        assert_eq!(local.len(), n);
        assert_eq!(out_mask, mask);
        // Output is zero outside the mask.
        assert!((0..n).all(|i| mask[i] != 0 || local[i] == 0.0));
    }

    /// In refinement mode (`prefilter = false`) mSMV does NOT run its own SMV
    /// removal: on an already-clean local field (all voxels below the shadow
    /// threshold) it returns the masked input unchanged — whereas the standalone
    /// mode SMV-filters it and changes the interior.
    #[test]
    fn refine_mode_skips_primary_smv() {
        let nn = 24;
        let g = grid(nn);
        let n = nn * nn * nn;
        // Small, smooth "already-local" field (well below the radian threshold).
        let mut field = vec![0.0f64; n];
        let mut mask = vec![0u8; n];
        let c = nn as f64 / 2.0;
        for k in 0..nn {
            for j in 0..nn {
                for i in 0..nn {
                    let idx = i + j * nn + k * nn * nn;
                    let (x, y, z) = (i as f64 - c, j as f64 - c, k as f64 - c);
                    field[idx] = 1e-4 * ((x * 0.3).sin() + (y * 0.2).cos());
                    if (x * x + y * y + z * z).sqrt() < 8.0 {
                        mask[idx] = 1;
                    }
                }
            }
        }
        let p = MsmvParams { b0: 3.0, te: 0.008, ..MsmvParams::default() };

        // Refinement mode: no primary SMV, field below threshold → unchanged in mask.
        let (refined, _) = msmv(&field, &mask, &g, &MsmvParams { prefilter: false, ..p.clone() }, |_, _| {});
        for i in 0..n {
            let expect = if mask[i] != 0 { field[i] } else { 0.0 };
            assert!((refined[i] - expect).abs() < 1e-12, "refine mode changed voxel {i}");
        }

        // Standalone mode: primary SMV removal actually alters the interior.
        let (standalone, _) = msmv(&field, &mask, &g, &MsmvParams { prefilter: true, ..p }, |_, _| {});
        let changed: f64 = (0..n)
            .filter(|&i| mask[i] != 0)
            .map(|i| (standalone[i] - field[i]).abs())
            .fold(0.0, f64::max);
        assert!(changed > 1e-9, "standalone mode should SMV-filter the field");
    }
}

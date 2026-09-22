//! COSMOS: calculation of susceptibility through multiple orientation sampling
//!
//! Single-orientation dipole inversion is ill-posed because the dipole kernel vanishes on a
//! pair of cones at the magic angle, so those spatial frequencies carry no information and
//! have to be regularized back in. COSMOS removes the ill-posedness at acquisition time
//! instead: the cones sit at a fixed angle to B0, so re-imaging the same object at a
//! different orientation moves them somewhere else. Combining N orientations, every
//! frequency is sampled well by at least one of them, and the inversion becomes a
//! well-conditioned least-squares problem needing no regularization at all.
//!
//! Minimizing the sum of squared residuals over orientations,
//!
//! ```text
//!     chi = argmin_x  sum_t || F^-1 D_t F x  -  f_t ||^2
//! ```
//!
//! diagonalizes in k-space (every operator involved is a k-space multiplication), giving a
//! closed form with no iteration:
//!
//! ```text
//!     chi = Re . IFFT(  sum_t D_t . FFT(f_t)  /  ( sum_t |D_t|^2 + lambda )  )
//! ```
//!
//! [`cosmos`] is that expression. [`cosmos_weighted`] is the original formulation, which
//! weights each orientation's residual by its signal magnitude; spatial weights do not
//! commute with the k-space operators, so that variant loses the closed form and solves the
//! normal equations with conjugate gradient instead.
//!
//! # Orientations
//!
//! Each orientation contributes its own B0 direction expressed *in the voxel frame of the
//! common (registered) grid* — the object is resampled into one shared frame, so what varies
//! between orientations is where B0 points relative to it. Derive each one with
//! [`crate::geometry::b0_direction_from_affine`] from that acquisition's affine, which
//! factors the voxel sizes out before taking the direction and so stays correct for
//! anisotropic voxels.
//!
//! # How many orientations, and what limits accuracy
//!
//! k-space coverage saturates fast. Measured on a 164x205x205 1 mm grid, the number of
//! k-points where *every* orientation's kernel is negligible goes 16952 (one orientation,
//! the magic-angle cone) to 16 (two orientations 60 degrees apart, where the two cones still
//! intersect along curves) to 0 (three). Even the real in-vivo orientations of the reference
//! dataset, which span only 0 to 25 degrees from B0 because that is as far as a head turns
//! inside a coil, leave nothing uncovered once there are twelve of them — a 17-degree
//! rotation already shifts the cone clear of where it started.
//!
//! So beyond about three orientations, extra scans buy noise averaging rather than coverage.
//! That still matters a great deal: with field noise at 5% of the field RMS, reconstruction
//! error on the same phantom runs roughly 205 / 0.09 / 0.03 (relative NRMSE) for one, two and
//! four orientations. The single-orientation case is catastrophic precisely because the cone
//! is uncovered and there is no redundancy to average against — which is the whole argument
//! for acquiring more than one.
//!
//! What is *not* fixed by adding orientations:
//!
//! - **Anisotropy.** COSMOS models susceptibility as a scalar. White matter is measurably
//!   anisotropic, and that mismatch does not shrink with more orientations; it is the
//!   motivation for susceptibility tensor imaging.
//! - **Registration.** Misalignment between orientations propagates straight into the result.
//! - **Rotations about a single axis.** These leave residual streaking that L2 regularization
//!   suppresses (Schweser & Zivadinov 2024) — one reason `lambda` is exposed.
//!
//! COSMOS is widely treated as a susceptibility ground truth; those three are what that
//! status actually rests on, not the orientation count.
//!
//! # Units and gauge
//!
//! Local (tissue) field in ppm in, susceptibility in ppm out, matching the rest of the crate.
//! The dipole kernel is zero at DC, so the recovered chi has zero mean by construction;
//! referencing to a tissue region is a separate pipeline stage ([`crate::pipeline::apply_reference`]).
//!
//! # References
//!
//! Liu T, Spincemaille P, de Rochefort L, Kressler B, Wang Y. "Calculation of susceptibility
//! through multiple orientation sampling (COSMOS): a method for conditioning the inverse
//! problem from measured magnetic field map to susceptibility source image in MRI."
//! Magnetic Resonance in Medicine. 2009;61(1):196-204. <https://doi.org/10.1002/mrm.21828>
//!
//! Bilgic B, Fan AP, Polimeni JR, et al. "Fast quantitative susceptibility mapping with
//! L1-regularization and automatic parameter selection" / "Rapid multi-orientation
//! quantitative susceptibility mapping." NeuroImage. 2015;111:622-630.
//! <https://doi.org/10.1016/j.neuroimage.2015.02.036>
//!
//! Schweser F, Zivadinov R. "Chaos and COSMOS - considerations on QSM methods with multiple
//! and single orientations and effects from local anisotropy." Magnetic Resonance Imaging.
//! 2024;109:135-146. <https://doi.org/10.1016/j.mri.2024.04.001>

use crate::fft::{apply_real_kernel, fft3d, ifft3d};
use crate::kernels::dipole::dipole_kernel;
use crate::solvers::cg_solve;
use crate::utils::apply_mask_zero;
use crate::Grid;
use num_complex::Complex64;

/// COSMOS parameters.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct CosmosParams {
    /// L2 regularization weight added to the k-space denominator.
    ///
    /// Zero — the default — is the plain least-squares solution, and is the right choice
    /// whenever three or more orientations cover k-space, which they generally do (see the
    /// module docs). Raise it (1e-3 to 1e-2 is the useful range) when there are only one or
    /// two orientations, so the magic-angle cone is still partly uncovered and noise there is
    /// amplified, or when the rotations share a single axis and streaking survives. With
    /// twelve well-covering orientations it does not help and a large value visibly damps the
    /// result.
    pub lambda: f64,
    /// Conjugate-gradient tolerance. [`cosmos_weighted`] only.
    pub tol: f64,
    /// Maximum conjugate-gradient iterations. [`cosmos_weighted`] only.
    pub max_iter: usize,
}

impl Default for CosmosParams {
    fn default() -> Self {
        Self { lambda: 0.0, tol: 1e-6, max_iter: 100 }
    }
}

/// Validate the shared preconditions of both entry points and return `(n_orientations, n_voxels)`.
fn check_inputs(n_fields: usize, n_bdirs: usize, lens: impl Iterator<Item = usize>, n: usize) -> usize {
    assert!(n_fields > 0, "cosmos: at least one orientation is required");
    assert_eq!(
        n_fields, n_bdirs,
        "cosmos: got {} field maps but {} B0 directions",
        n_fields, n_bdirs
    );
    for (t, len) in lens.enumerate() {
        assert_eq!(
            len, n,
            "cosmos: orientation {} has {} voxels, expected {}",
            t, len, n
        );
    }
    n_fields
}

/// COSMOS dipole inversion, closed form.
///
/// # Arguments
/// * `local_fields` - One local (tissue) field map per orientation, ppm, all resampled onto
///   the same grid.
/// * `bdirs` - B0 direction for each orientation, in the voxel frame of that shared grid.
/// * `mask` - Binary mask (1 = inside ROI).
/// * `grid` - The shared volume grid.
/// * `params` - See [`CosmosParams`]; only `lambda` applies here.
///
/// # Returns
/// Susceptibility map in ppm, zero outside `mask`.
///
/// # Panics
/// If no orientations are given, if the field and direction counts differ, or if any field
/// map does not match `grid`.
pub fn cosmos<F: AsRef<[f64]>>(
    local_fields: &[F],
    bdirs: &[(f64, f64, f64)],
    mask: &[u8],
    grid: &Grid,
    params: &CosmosParams,
) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let n = grid.n_total();
    let n_dir = check_inputs(
        local_fields.len(),
        bdirs.len(),
        local_fields.iter().map(|f| f.as_ref().len()),
        n,
    );

    // Accumulate numerator sum_t D_t . FFT(f_t) and denominator sum_t |D_t|^2 together, so
    // only one kernel is ever held at a time.
    let mut num = vec![Complex64::new(0.0, 0.0); n];
    let mut den = vec![0.0f64; n];

    for t in 0..n_dir {
        let d = dipole_kernel(grid, bdirs[t]);

        let mut f: Vec<Complex64> = local_fields[t]
            .as_ref()
            .iter()
            .map(|&v| Complex64::new(v, 0.0))
            .collect();
        fft3d(&mut f, nx, ny, nz);

        for i in 0..n {
            // The dipole kernel is real, so D* = D.
            num[i] += f[i] * d[i];
            den[i] += d[i] * d[i];
        }
    }

    // Divide, guarding the null space. The kernel is zero at DC for every orientation, so
    // the denominator vanishes there no matter how many orientations are supplied; dividing
    // through anyway lands a huge constant pedestal on the result.
    for i in 0..n {
        let dd = den[i] + params.lambda;
        num[i] = if dd > 1e-20 { num[i] / dd } else { Complex64::new(0.0, 0.0) };
    }

    ifft3d(&mut num, nx, ny, nz);

    let mut chi: Vec<f64> = num.iter().map(|c| c.re).collect();
    apply_mask_zero(&mut chi, mask);
    chi
}

/// COSMOS dipole inversion with per-orientation spatial weighting.
///
/// This is the original formulation: each orientation's residual is weighted by its signal
/// magnitude, so voxels where that orientation had poor SNR (or was corrupted, or fell
/// outside its own coverage) contribute less. Weighting is a spatial-domain operation and
/// the dipole convolution is a k-space one, so the two no longer diagonalize together and
/// the closed form of [`cosmos`] is unavailable. Instead the normal equations
///
/// ```text
///     ( sum_t D_t W_t^2 D_t  +  lambda I ) chi  =  sum_t D_t W_t^2 f_t
/// ```
///
/// are solved with conjugate gradient. The operator is symmetric positive semi-definite
/// (`D_t` is a real, even k-space multiplier, hence self-adjoint), so CG applies directly.
/// Each iteration costs two FFTs per orientation, making this substantially more expensive
/// than [`cosmos`] — prefer the closed form unless the weighting earns its cost.
///
/// `weights` are multiplied by `mask` internally, so they need not be pre-masked. Typically
/// they are the per-orientation magnitude images, normalized to a comparable scale across
/// orientations (their *relative* size between orientations is what matters; a global factor
/// only rescales `lambda`).
///
/// # Panics
/// As [`cosmos`], and additionally if the weight count or any weight map size is wrong.
pub fn cosmos_weighted<F: AsRef<[f64]>, W: AsRef<[f64]>>(
    local_fields: &[F],
    weights: &[W],
    bdirs: &[(f64, f64, f64)],
    mask: &[u8],
    grid: &Grid,
    params: &CosmosParams,
) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let n = grid.n_total();
    let n_dir = check_inputs(
        local_fields.len(),
        bdirs.len(),
        local_fields.iter().map(|f| f.as_ref().len()),
        n,
    );
    assert_eq!(
        weights.len(),
        n_dir,
        "cosmos_weighted: got {} weight maps but {} orientations",
        weights.len(),
        n_dir
    );
    for (t, w) in weights.iter().enumerate() {
        assert_eq!(
            w.as_ref().len(),
            n,
            "cosmos_weighted: weight map {} has {} voxels, expected {}",
            t,
            w.as_ref().len(),
            n
        );
    }

    let kernels: Vec<Vec<f64>> = bdirs.iter().map(|&b| dipole_kernel(grid, b)).collect();

    // Fold the mask in once and square, so the inner loops multiply by W^2 directly.
    let w2: Vec<Vec<f64>> = weights
        .iter()
        .map(|w| {
            w.as_ref()
                .iter()
                .zip(mask.iter())
                .map(|(&wi, &m)| if m == 0 { 0.0 } else { wi * wi })
                .collect()
        })
        .collect();

    // b = sum_t D_t (W_t^2 f_t)
    let mut b = vec![0.0f64; n];
    for t in 0..n_dir {
        let f = local_fields[t].as_ref();
        let wf: Vec<f64> = (0..n).map(|i| w2[t][i] * f[i]).collect();
        let term = apply_real_kernel(&wf, &kernels[t], nx, ny, nz);
        for i in 0..n {
            b[i] += term[i];
        }
    }

    // A x = sum_t D_t (W_t^2 (D_t x)) + lambda x
    let lambda = params.lambda;
    let a_op = |x: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0f64; n];
        for t in 0..n_dir {
            let dx = apply_real_kernel(x, &kernels[t], nx, ny, nz);
            let wdx: Vec<f64> = (0..n).map(|i| w2[t][i] * dx[i]).collect();
            let term = apply_real_kernel(&wdx, &kernels[t], nx, ny, nz);
            for i in 0..n {
                out[i] += term[i];
            }
        }
        if lambda != 0.0 {
            for i in 0..n {
                out[i] += lambda * x[i];
            }
        }
        out
    };

    let x0 = vec![0.0f64; n];
    let mut chi = cg_solve(a_op, &b, &x0, params.tol, params.max_iter);
    apply_mask_zero(&mut chi, mask);
    chi
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A spread of B0 directions wide enough to condition the inversion: straight along z,
    /// then 60 degrees off it in each of two perpendicular planes.
    fn spread_bdirs() -> Vec<(f64, f64, f64)> {
        let s = (60.0f64).to_radians().sin();
        let c = (60.0f64).to_radians().cos();
        vec![(0.0, 0.0, 1.0), (s, 0.0, c), (0.0, s, c)]
    }

    /// Forward dipole model: the field this chi would produce with B0 along `bdir`.
    fn simulate_field(chi: &[f64], grid: &Grid, bdir: (f64, f64, f64)) -> Vec<f64> {
        let (nx, ny, nz) = grid.dims;
        apply_real_kernel(chi, &dipole_kernel(grid, bdir), nx, ny, nz)
    }

    /// A smooth, zero-mean test susceptibility distribution. Zero-mean because the dipole
    /// kernel annihilates DC, so no method can recover a constant offset.
    fn test_chi(grid: &Grid) -> Vec<f64> {
        let (nx, ny, nz) = grid.dims;
        let mut chi = vec![0.0f64; grid.n_total()];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let (x, y, z) = (i as f64, j as f64, k as f64);
                    chi[i + j * nx + k * nx * ny] =
                        (0.7 * x).sin() * (0.5 * y).cos() + 0.4 * (0.3 * z).sin();
                }
            }
        }
        let mean = chi.iter().sum::<f64>() / chi.len() as f64;
        for v in chi.iter_mut() {
            *v -= mean;
        }
        chi
    }

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
    }

    /// The test that matters: forward-simulate one chi at several orientations, invert, and
    /// get the chi back. With noiseless data and a well-spread set this is exact to
    /// round-off, which is what makes COSMOS worth acquiring.
    #[test]
    fn recovers_chi_from_simulated_orientations() {
        let grid = Grid::new(16, 16, 16, 1.0, 1.0, 1.0);
        let mask = vec![1u8; grid.n_total()];
        let chi_true = test_chi(&grid);
        let bdirs = spread_bdirs();

        let fields: Vec<Vec<f64>> = bdirs.iter().map(|&b| simulate_field(&chi_true, &grid, b)).collect();
        let chi = cosmos(&fields, &bdirs, &mask, &grid, &CosmosParams::default());

        let err = max_abs_diff(&chi, &chi_true);
        let scale = chi_true.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(err / scale < 1e-9, "relative error {} too large", err / scale);
    }

    /// Anisotropic voxels exercise the k-space scaling in the kernel; recovery must hold.
    #[test]
    fn recovers_chi_with_anisotropic_voxels() {
        let grid = Grid::new(16, 16, 12, 0.8, 0.8, 2.0);
        let mask = vec![1u8; grid.n_total()];
        let chi_true = test_chi(&grid);
        let bdirs = spread_bdirs();

        let fields: Vec<Vec<f64>> = bdirs.iter().map(|&b| simulate_field(&chi_true, &grid, b)).collect();
        let chi = cosmos(&fields, &bdirs, &mask, &grid, &CosmosParams::default());

        let scale = chi_true.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(max_abs_diff(&chi, &chi_true) / scale < 1e-9);
    }

    /// A single orientation reduces COSMOS to plain k-space division, which is ill-posed at
    /// the magic angle. It must still return finite numbers rather than dividing by zero.
    #[test]
    fn single_orientation_stays_finite() {
        let grid = Grid::new(12, 12, 12, 1.0, 1.0, 1.0);
        let mask = vec![1u8; grid.n_total()];
        let chi_true = test_chi(&grid);
        let bdirs = vec![(0.0, 0.0, 1.0)];
        let fields = vec![simulate_field(&chi_true, &grid, bdirs[0])];

        let chi = cosmos(&fields, &bdirs, &mask, &grid, &CosmosParams::default());
        assert!(chi.iter().all(|v| v.is_finite()), "single orientation produced non-finite chi");
    }

    /// The kernel is zero at DC for every orientation, so the denominator vanishes there no
    /// matter how many are supplied. Without the guard this leaves a constant pedestal.
    #[test]
    fn no_dc_pedestal() {
        let grid = Grid::new(16, 16, 16, 1.0, 1.0, 1.0);
        let mask = vec![1u8; grid.n_total()];
        let mut chi_true = test_chi(&grid);
        // Add a large constant the inversion cannot possibly recover.
        for v in chi_true.iter_mut() {
            *v += 5.0;
        }
        let bdirs = spread_bdirs();
        let fields: Vec<Vec<f64>> = bdirs.iter().map(|&b| simulate_field(&chi_true, &grid, b)).collect();

        let chi = cosmos(&fields, &bdirs, &mask, &grid, &CosmosParams::default());
        let mean = chi.iter().sum::<f64>() / chi.len() as f64;
        assert!(chi.iter().all(|v| v.is_finite()), "DC guard failed, result not finite");
        assert!(mean.abs() < 1e-9, "expected zero-mean output, got mean {}", mean);
    }

    #[test]
    fn zero_field_gives_zero_chi() {
        let grid = Grid::new(8, 8, 8, 1.0, 1.0, 1.0);
        let n = grid.n_total();
        let mask = vec![1u8; n];
        let bdirs = spread_bdirs();
        let fields: Vec<Vec<f64>> = (0..3).map(|_| vec![0.0; n]).collect();

        let chi = cosmos(&fields, &bdirs, &mask, &grid, &CosmosParams::default());
        assert!(chi.iter().all(|v| v.abs() < 1e-12));
    }

    #[test]
    fn masked_voxels_are_zero() {
        let grid = Grid::new(8, 8, 8, 1.0, 1.0, 1.0);
        let n = grid.n_total();
        let mut mask = vec![1u8; n];
        mask[0] = 0;
        mask[5] = 0;
        let bdirs = spread_bdirs();
        let chi_true = test_chi(&grid);
        let fields: Vec<Vec<f64>> = bdirs.iter().map(|&b| simulate_field(&chi_true, &grid, b)).collect();

        let chi = cosmos(&fields, &bdirs, &mask, &grid, &CosmosParams::default());
        assert_eq!(chi[0], 0.0);
        assert_eq!(chi[5], 0.0);
    }

    /// lambda damps the solution; it must not be silently ignored.
    #[test]
    fn lambda_shrinks_the_solution() {
        let grid = Grid::new(12, 12, 12, 1.0, 1.0, 1.0);
        let mask = vec![1u8; grid.n_total()];
        let chi_true = test_chi(&grid);
        let bdirs = spread_bdirs();
        let fields: Vec<Vec<f64>> = bdirs.iter().map(|&b| simulate_field(&chi_true, &grid, b)).collect();

        let energy = |c: &[f64]| c.iter().map(|v| v * v).sum::<f64>();
        let plain = cosmos(&fields, &bdirs, &mask, &grid, &CosmosParams::default());
        let reg = cosmos(
            &fields,
            &bdirs,
            &mask,
            &grid,
            &CosmosParams { lambda: 0.1, ..Default::default() },
        );

        assert!(energy(&reg) < energy(&plain), "lambda did not damp the solution");
    }

    /// With uniform weights the weighted normal equations have the same solution as the
    /// closed form, so CG must converge onto it. This cross-checks the two code paths
    /// against each other.
    #[test]
    fn weighted_with_uniform_weights_matches_closed_form() {
        let grid = Grid::new(12, 12, 12, 1.0, 1.0, 1.0);
        let n = grid.n_total();
        let mask = vec![1u8; n];
        let chi_true = test_chi(&grid);
        let bdirs = spread_bdirs();
        let fields: Vec<Vec<f64>> = bdirs.iter().map(|&b| simulate_field(&chi_true, &grid, b)).collect();
        let weights: Vec<Vec<f64>> = (0..bdirs.len()).map(|_| vec![1.0; n]).collect();

        let params = CosmosParams { tol: 1e-12, max_iter: 400, ..Default::default() };
        let closed = cosmos(&fields, &bdirs, &mask, &grid, &params);
        let weighted = cosmos_weighted(&fields, &weights, &bdirs, &mask, &grid, &params);

        let scale = closed.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let rel = max_abs_diff(&closed, &weighted) / scale;
        assert!(rel < 1e-6, "weighted and closed-form solutions differ by {}", rel);
    }

    /// Down-weighting an orientation should pull the solution toward what the remaining
    /// orientations alone say, not toward the corrupted one.
    #[test]
    fn weighting_suppresses_a_corrupted_orientation() {
        let grid = Grid::new(12, 12, 12, 1.0, 1.0, 1.0);
        let n = grid.n_total();
        let mask = vec![1u8; n];
        let chi_true = test_chi(&grid);
        let bdirs = spread_bdirs();
        let mut fields: Vec<Vec<f64>> =
            bdirs.iter().map(|&b| simulate_field(&chi_true, &grid, b)).collect();

        // Corrupt the last orientation.
        for (i, v) in fields[2].iter_mut().enumerate() {
            *v += 0.5 * ((i as f64) * 0.37).sin();
        }

        let params = CosmosParams { tol: 1e-12, max_iter: 400, ..Default::default() };
        let equal: Vec<Vec<f64>> = (0..3).map(|_| vec![1.0; n]).collect();
        let mut downweighted = equal.clone();
        downweighted[2] = vec![1e-3; n];

        let err_equal = max_abs_diff(
            &cosmos_weighted(&fields, &equal, &bdirs, &mask, &grid, &params),
            &chi_true,
        );
        let err_down = max_abs_diff(
            &cosmos_weighted(&fields, &downweighted, &bdirs, &mask, &grid, &params),
            &chi_true,
        );

        assert!(
            err_down < err_equal,
            "down-weighting the corrupted orientation made things worse ({} vs {})",
            err_down,
            err_equal
        );
    }

    #[test]
    #[should_panic(expected = "but 2 B0 directions")]
    fn rejects_mismatched_direction_count() {
        let grid = Grid::new(8, 8, 8, 1.0, 1.0, 1.0);
        let n = grid.n_total();
        let fields: Vec<Vec<f64>> = (0..3).map(|_| vec![0.0; n]).collect();
        let bdirs = vec![(0.0, 0.0, 1.0), (1.0, 0.0, 0.0)];
        cosmos(&fields, &bdirs, &vec![1u8; n], &grid, &CosmosParams::default());
    }

    #[test]
    #[should_panic(expected = "at least one orientation")]
    fn rejects_empty_input() {
        let grid = Grid::new(8, 8, 8, 1.0, 1.0, 1.0);
        let fields: Vec<Vec<f64>> = Vec::new();
        cosmos(&fields, &[], &vec![1u8; grid.n_total()], &grid, &CosmosParams::default());
    }

    #[test]
    #[should_panic(expected = "orientation 1 has")]
    fn rejects_wrong_sized_field() {
        let grid = Grid::new(8, 8, 8, 1.0, 1.0, 1.0);
        let n = grid.n_total();
        let fields = vec![vec![0.0; n], vec![0.0; n - 1]];
        let bdirs = vec![(0.0, 0.0, 1.0), (1.0, 0.0, 0.0)];
        cosmos(&fields, &bdirs, &vec![1u8; n], &grid, &CosmosParams::default());
    }
}

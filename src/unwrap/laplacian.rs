//! Laplacian-based phase unwrapping
//!
//! The Laplacian of the wrapped phase equals the Laplacian of the true phase wherever
//! neighbouring samples differ by less than π, so the true phase can be recovered by
//! solving a Poisson equation. Path-independent and fast, unlike region-growing methods.
//!
//! This module provides **two algorithms that are not interchangeable**, because the
//! boundary condition used to solve the Poisson equation decides whether the harmonic
//! (background) component of the field survives:
//!
//! | Function | Boundary condition | Category |
//! |---|---|---|
//! | [`laplacian_unwrap_neumann`] | Neumann, on the array | Phase unwrapping |
//! | [`laplacian_unwrap`] | Dirichlet, on the ROI | Phase unwrapping **+ background field removal** |
//!
//! [`laplacian_unwrap`] zeroes ∇²φ outside the mask, which discards every field source
//! outside the ROI. Background fields are harmonic inside the ROI, and ∇²(harmonic) = 0
//! carries no information about them, so they cannot be recovered afterwards — the
//! function returns a partially background-removed field, not a total field. That is the
//! same combination HARPERELLA and iHARPERELLA perform, and is why it is categorised
//! with them rather than with ROMEO.
//!
//! Pair [`laplacian_unwrap`] with a separate background-removal stage only deliberately:
//! doing so removes background twice, by an amount that is not controlled.
//!
//! **Prefer unwrapping and background removal as two steps.** On the project's test data,
//! the background removal [`laplacian_unwrap`] performs implicitly reaches r = 0.52 against
//! the ground-truth local field, where [`crate::bgremove::lbv`] on the same field — a full
//! Laplacian boundary value solve — reaches r = 0.91 and V-SHARP 0.88. The Dirichlet
//! condition here is approximated by zeroing ∇² outside the mask and solving with a
//! periodic FFT, rather than the ROI solve the LBV reference describes, and it shows.
//! [`UnwrapMethod::Laplacian`](super::UnwrapMethod::Laplacian) therefore selects
//! [`laplacian_unwrap_neumann`]; this function is kept for callers that specifically want
//! the combination.
//!
//! # References
//!
//! Laplacian unwrapping:
//! Schofield, M.A., Zhu, Y. (2003). "Fast phase unwrapping algorithm for
//! interferometric applications." Optics Letters, 28(14):1194-1196.
//! <https://doi.org/10.1364/OL.28.001194>
//!
//! The background-removal half of [`laplacian_unwrap`] (solving the Laplacian as a
//! boundary value problem on the ROI):
//! Zhou, D., Liu, T., Spincemaille, P., Wang, Y. (2014). "Background field removal by
//! solving the Laplacian boundary value problem." NMR in Biomedicine, 27(3):312-319.
//! <https://doi.org/10.1002/nbm.3064>
//!
//! Reference implementation: <https://github.com/kamesy/QSM.jl> — its `unwrap_laplacian`
//! exposes the same split through its `solver` keyword (`:dct`/`:fft` impose the boundary
//! condition on the array and unwrap only; `:mgpcg` imposes it on the ROI and also removes
//! the harmonic background).

use std::f64::consts::PI;
use num_complex::Complex64;
use crate::fft::{fft3d, ifft3d};
use crate::Grid;

/// Wrap angle to [-π, π]
#[inline]
pub(crate) fn wrap(x: f64) -> f64 {
    let mut y = x % (2.0 * PI);
    if y > PI {
        y -= 2.0 * PI;
    } else if y < -PI {
        y += 2.0 * PI;
    }
    y
}

/// Compute wrapped Laplacian of phase with periodic boundary conditions
///
/// Uses second-order central finite differences on wrapped phase differences.
pub(crate) fn wrapped_laplacian_periodic(
    phase: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> Vec<f64> {
    let n_total = nx * ny * nz;
    let mut d2u = vec![0.0; n_total];

    let dx2 = 1.0 / (vsx * vsx);
    let dy2 = 1.0 / (vsy * vsy);
    let dz2 = 1.0 / (vsz * vsz);

    for k in 0..nz {
        let km1 = if k == 0 { nz - 1 } else { k - 1 };
        let kp1 = if k + 1 >= nz { 0 } else { k + 1 };

        for j in 0..ny {
            let jm1 = if j == 0 { ny - 1 } else { j - 1 };
            let jp1 = if j + 1 >= ny { 0 } else { j + 1 };

            for i in 0..nx {
                let im1 = if i == 0 { nx - 1 } else { i - 1 };
                let ip1 = if i + 1 >= nx { 0 } else { i + 1 };

                let idx = i + j * nx + k * nx * ny;
                let u_ijk = phase[idx];

                let idx_im1 = im1 + j * nx + k * nx * ny;
                let idx_ip1 = ip1 + j * nx + k * nx * ny;
                let idx_jm1 = i + jm1 * nx + k * nx * ny;
                let idx_jp1 = i + jp1 * nx + k * nx * ny;
                let idx_km1 = i + j * nx + km1 * nx * ny;
                let idx_kp1 = i + j * nx + kp1 * nx * ny;

                let lap_x = (wrap(phase[idx_ip1] - u_ijk) - wrap(u_ijk - phase[idx_im1])) * dx2;
                let lap_y = (wrap(phase[idx_jp1] - u_ijk) - wrap(u_ijk - phase[idx_jm1])) * dy2;
                let lap_z = (wrap(phase[idx_kp1] - u_ijk) - wrap(u_ijk - phase[idx_km1])) * dz2;

                d2u[idx] = lap_x + lap_y + lap_z;
            }
        }
    }

    d2u
}

/// Solve Poisson equation using FFT (periodic boundary conditions)
pub(crate) fn solve_poisson_fft(
    f: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> Vec<f64> {
    let mut f_complex: Vec<Complex64> = f.iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();
    fft3d(&mut f_complex, nx, ny, nz);

    let idx2 = 1.0 / (vsx * vsx);
    let idy2 = 1.0 / (vsy * vsy);
    let idz2 = 1.0 / (vsz * vsz);

    for k in 0..nz {
        let fk = if k <= nz / 2 { k as f64 / nz as f64 } else { (k as f64 - nz as f64) / nz as f64 };
        let lam_z = 2.0 * ((2.0 * PI * fk).cos() - 1.0) * idz2;

        for j in 0..ny {
            let fj = if j <= ny / 2 { j as f64 / ny as f64 } else { (j as f64 - ny as f64) / ny as f64 };
            let lam_y = 2.0 * ((2.0 * PI * fj).cos() - 1.0) * idy2;

            for i in 0..nx {
                let fi = if i <= nx / 2 { i as f64 / nx as f64 } else { (i as f64 - nx as f64) / nx as f64 };
                let lam_x = 2.0 * ((2.0 * PI * fi).cos() - 1.0) * idx2;

                let lam = lam_x + lam_y + lam_z;
                let idx = i + j * nx + k * nx * ny;

                if lam.abs() > 1e-20 {
                    f_complex[idx] /= lam;
                } else {
                    f_complex[idx] = Complex64::new(0.0, 0.0);
                }
            }
        }
    }

    ifft3d(&mut f_complex, nx, ny, nz);
    f_complex.iter().map(|c| c.re).collect()
}

/// Laplacian phase unwrapping **combined with background field removal**.
///
/// Solves the Poisson equation with the Laplacian zeroed outside `mask`, which imposes a
/// homogeneous Dirichlet condition on the ROI. Sources outside the ROI are discarded, so
/// the harmonic (background) component of the field is removed along with the wraps.
///
/// **The result is not a total field.** It is unwrapped *and* partially background-removed,
/// by an amount that depends on the mask and the field geometry. Following this with a
/// separate background-removal stage (V-SHARP, PDF, …) removes background twice.
///
/// Use [`laplacian_unwrap_neumann`] to unwrap without removing background.
///
/// Because ∇²(harmonic) = 0, the discarded component leaves no trace in the input to the
/// Poisson solve and cannot be restored afterwards.
///
/// # Arguments
/// * `phase` - Wrapped phase (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz), 1 = inside ROI
/// * `grid` - Volume grid (dimensions and voxel sizes)
///
/// # Returns
/// Unwrapped, partially background-removed phase, zero outside `mask`.
///
/// # References
/// Schofield & Zhu (2003) for the unwrapping; Zhou et al. (2014) for the
/// boundary-value formulation of the background removal. See the module docs.
pub fn laplacian_unwrap(
    phase: &[f64],
    mask: &[u8],
    grid: &Grid,
) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let (vsx, vsy, vsz) = grid.voxel_size;
    let n_total = nx * ny * nz;

    let d2u = wrapped_laplacian_periodic(phase, nx, ny, nz, vsx, vsy, vsz);

    let d2u_masked: Vec<f64> = d2u.iter()
        .enumerate()
        .map(|(i, &val)| if mask[i] != 0 { val } else { 0.0 })
        .collect();

    let unwrapped = solve_poisson_fft(&d2u_masked, nx, ny, nz, vsx, vsy, vsz);

    let mut result = vec![0.0; n_total];
    for i in 0..n_total {
        if mask[i] != 0 {
            result[i] = unwrapped[i];
        }
    }

    result
}

/// Laplacian phase unwrapping, **without** background field removal.
///
/// Solves the Poisson equation over the whole array under a Neumann (zero normal
/// derivative) boundary condition, realised by even-extending the phase to twice the size
/// along each axis before the FFT solve. Nothing is masked out, so field sources anywhere
/// in the FOV are retained and the harmonic (background) component survives: the result is
/// an unwrapped **total** field, suitable for a subsequent background-removal stage.
///
/// Use [`laplacian_unwrap`] if you want unwrapping and background removal together.
///
/// Because the whole array participates, this is sensitive to phase quality *outside* the
/// ROI in a way [`laplacian_unwrap`] is not. Where the phase outside the object is noise,
/// or wraps faster than one radian per voxel, prefer ROMEO
/// ([`super::romeo::unwrap_romeo`]) or the masked variant.
///
/// # Memory
/// The even extension allocates an 8x volume (2x per axis) for the FFT. On a 32-bit
/// target, or a large FOV, that can dominate — [`laplacian_unwrap`] works in place.
///
/// # Arguments
/// * `phase` - Wrapped phase (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz); applied to the *output* only
/// * `grid` - Volume grid (dimensions and voxel sizes)
///
/// # Returns
/// Unwrapped phase, zero outside `mask`.
///
/// # References
/// Schofield, M.A., Zhu, Y. (2003). "Fast phase unwrapping algorithm for interferometric
/// applications." Optics Letters, 28(14):1194-1196.
/// <https://doi.org/10.1364/OL.28.001194>
pub fn laplacian_unwrap_neumann(
    phase: &[f64],
    mask: &[u8],
    grid: &Grid,
) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let (vsx, vsy, vsz) = grid.voxel_size;
    let n_total = nx * ny * nz;

    // Even extension: continuous across the seam, so the periodic FFT solve realises a
    // Neumann condition on the original array instead of wrapping a discontinuity.
    let (px, py, pz) = (2 * nx, 2 * ny, 2 * nz);
    let mut ext = vec![0.0f64; px * py * pz];
    for k in 0..pz {
        let sk = if k < nz { k } else { 2 * nz - 1 - k };
        for j in 0..py {
            let sj = if j < ny { j } else { 2 * ny - 1 - j };
            for i in 0..px {
                let si = if i < nx { i } else { 2 * nx - 1 - i };
                ext[i + j * px + k * px * py] = phase[si + sj * nx + sk * nx * ny];
            }
        }
    }

    // Laplacian is taken on the extended array, so the wrapped differences never straddle
    // the original boundary — doing it before the extension reintroduces the seam.
    let d2u = wrapped_laplacian_periodic(&ext, px, py, pz, vsx, vsy, vsz);
    let solved = solve_poisson_fft(&d2u, px, py, pz, vsx, vsy, vsz);

    let mut result = vec![0.0; n_total];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let dst = i + j * nx + k * nx * ny;
                if mask[dst] != 0 {
                    result[dst] = solved[i + j * px + k * px * py];
                }
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(n: usize) -> Grid {
        Grid::new(n, n, n, 1.0, 1.0, 1.0)
    }

    #[test]
    fn test_wrap() {
        assert!((wrap(0.0) - 0.0).abs() < 1e-10);
        assert!((wrap(PI) - PI).abs() < 1e-10);
        assert!((wrap(-PI) - (-PI)).abs() < 1e-10);
        assert!((wrap(2.0 * PI) - 0.0).abs() < 1e-10);
        assert!((wrap(3.0 * PI) - PI).abs() < 1e-10);
        assert!((wrap(-3.0 * PI) - (-PI)).abs() < 1e-10);
    }

    #[test]
    fn test_laplacian_unwrap_constant() {
        let n = 8;
        let phase = vec![1.0; n * n * n];
        let mask = vec![1u8; n * n * n];

        let unwrapped = laplacian_unwrap(&phase, &mask, &grid(n));

        let mean: f64 = unwrapped.iter().sum::<f64>() / (n * n * n) as f64;
        for &val in unwrapped.iter() {
            assert!((val - mean).abs() < 1e-6, "Constant phase should unwrap to constant");
        }
    }

    #[test]
    fn test_laplacian_unwrap_smooth() {
        let n = 16;
        let mut phase = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];

        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let idx = i + j * n + k * n * n;
                    phase[idx] = 0.5 * (2.0 * PI * i as f64 / n as f64).sin();
                }
            }
        }

        let unwrapped = laplacian_unwrap(&phase, &mask, &grid(n));

        // A periodic input under a periodic solve round-trips up to the DC term, so this
        // can be checked properly. The previous tolerance here was 1.0 against a signal of
        // amplitude 0.5, which passed for an all-zero output.
        let (r, slope) = corr_slope(&unwrapped, &phase, &mask);
        assert!(r > 0.99, "smooth periodic phase should round-trip, got r = {r}");
        assert!((slope - 1.0).abs() < 0.05, "expected unit slope, got {slope}");
        assert!(unwrapped.iter().all(|v| v.is_finite()), "output must be finite");
    }

    /// Pearson r and the slope of `want` regressed on `obs`, inside `mask`.
    fn corr_slope(obs: &[f64], want: &[f64], mask: &[u8]) -> (f64, f64) {
        let (mut sa, mut sb, mut n) = (0.0, 0.0, 0usize);
        for i in 0..obs.len() {
            if mask[i] != 0 { sa += obs[i]; sb += want[i]; n += 1; }
        }
        let nf = n as f64;
        let (ma, mb) = (sa / nf, sb / nf);
        let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
        for i in 0..obs.len() {
            if mask[i] == 0 { continue; }
            let (da, db) = (obs[i] - ma, want[i] - mb);
            cov += da * db;
            va += da * da;
            vb += db * db;
        }
        (cov / (va.sqrt() * vb.sqrt()), cov / va)
    }

    /// A field split into a harmonic part (a linear ramp, ∇² = 0) and a non-harmonic part
    /// (a Gaussian blob), wrapped hard enough that unwrapping is doing real work.
    /// Returns (wrapped, truth, ramp, blob, mask).
    fn ramp_plus_blob(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<u8>) {
        let c = n as f64 / 2.0;
        let total = n * n * n;
        let (mut truth, mut ramp, mut blob) = (vec![0.0; total], vec![0.0; total], vec![0.0; total]);
        let mut mask = vec![0u8; total];
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let idx = i + j * n + k * n * n;
                    let (x, y, z) = (i as f64 - c, j as f64 - c, k as f64 - c);
                    let r = (x * x + y * y + z * z).sqrt();
                    ramp[idx] = 0.35 * x + 0.20 * y + 0.15 * z;
                    blob[idx] = 12.0 * (-(r * r) / (2.0 * 8.0f64.powi(2))).exp();
                    truth[idx] = ramp[idx] + blob[idx];
                    if r < 22.0 { mask[idx] = 1; }
                }
            }
        }
        let wrapped: Vec<f64> = truth.iter().map(|&v| wrap(v)).collect();
        (wrapped, truth, ramp, blob, mask)
    }

    #[test]
    fn neumann_variant_recovers_the_whole_field_including_background() {
        let n = 64;
        let (wrapped, truth, _, _, mask) = ramp_plus_blob(n);

        let out = laplacian_unwrap_neumann(&wrapped, &mask, &grid(n));
        let (r, slope) = corr_slope(&out, &truth, &mask);

        assert!(r > 0.99, "should recover the total field, got r = {r}");
        assert!((slope - 1.0).abs() < 0.05, "expected unit slope, got {slope}");
    }

    #[test]
    fn masked_variant_removes_the_harmonic_component() {
        // Pins the documented contract: laplacian_unwrap is unwrapping + background
        // removal. It reproduces the non-harmonic field faithfully and returns the
        // harmonic (background) component as zero, because ∇²(harmonic) = 0 leaves no
        // trace in the input to the Poisson solve.
        //
        // If the harmonic assertion starts failing, the limitation was lifted — update
        // the module docs and the README categorisation along with it.
        let n = 64;
        let (wrapped, _, ramp, blob, mask) = ramp_plus_blob(n);

        let out = laplacian_unwrap(&wrapped, &mask, &grid(n));

        let (r_blob, slope_blob) = corr_slope(&out, &blob, &mask);
        assert!(r_blob > 0.99, "non-harmonic part should survive, got r = {r_blob}");
        assert!((slope_blob - 1.0).abs() < 0.05, "expected unit slope, got {slope_blob}");

        let (r_ramp, _) = corr_slope(&out, &ramp, &mask);
        assert!(r_ramp.abs() < 0.05, "harmonic part should be discarded, got r = {r_ramp}");
    }

    #[test]
    fn the_two_variants_disagree_by_the_background_field() {
        // The difference between them is the harmonic component, which is what makes
        // them different algorithms rather than two settings of one.
        let n = 64;
        let (wrapped, _, ramp, _, mask) = ramp_plus_blob(n);

        let pure = laplacian_unwrap_neumann(&wrapped, &mask, &grid(n));
        let combined = laplacian_unwrap(&wrapped, &mask, &grid(n));
        let diff: Vec<f64> = pure.iter().zip(&combined).map(|(a, b)| a - b).collect();

        let (r, _) = corr_slope(&diff, &ramp, &mask);
        assert!(r > 0.95, "their difference should be the harmonic field, got r = {r}");
    }

    #[test]
    fn test_laplacian_unwrap_finite() {
        let n = 8;
        let phase: Vec<f64> = (0..n*n*n).map(|i| wrap((i as f64) * 0.1)).collect();
        let mask = vec![1u8; n * n * n];

        let unwrapped = laplacian_unwrap(&phase, &mask, &grid(n));

        for (i, &val) in unwrapped.iter().enumerate() {
            assert!(val.is_finite(), "Unwrapped phase should be finite at index {}", i);
        }
    }
}

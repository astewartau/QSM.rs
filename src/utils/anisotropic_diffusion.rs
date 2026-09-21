//! Gradient anisotropic diffusion (Perona–Malik) smoothing.
//!
//! Edge-preserving smoothing: the diffusion coefficient falls off exponentially
//! with the local gradient magnitude, so homogeneous regions are blurred while
//! edges are left alone.
//!
//! This follows ITK's `GradientAnisotropicDiffusionImageFilter`, which sets the
//! conductance scale adaptively from the image itself each iteration:
//!
//! ```text
//! K      = -2 · c² · ⟨|∇u|²⟩
//! C(∇u)  = exp(|∇u|² / K)
//! u ⟵ u + Δt · Σ_i [ D⁺_i u · C⁺_i − D⁻_i u · C⁻_i ]
//! ```
//!
//! where `⟨|∇u|²⟩` is the mean squared central-difference gradient magnitude over
//! the volume. Because `K` is re-derived from the data on every sweep, the
//! conductance parameter `c` is a dimensionless multiple of the image's own
//! gradient scale rather than an absolute intensity threshold.
//!
//! Like ITK, voxel spacing is ignored (`UseImageSpacing` off) — the stencil works
//! in voxel units — and out-of-bounds neighbours use a zero-flux Neumann
//! condition (index clamping).
//!
//! Reference:
//! Perona, P., Malik, J. (1990). "Scale-space and edge detection using anisotropic
//! diffusion." IEEE TPAMI, 12(7):629-639. https://doi.org/10.1109/34.56205

use crate::Grid;

/// Parameters for [`gradient_anisotropic_diffusion`].
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug)]
pub struct AnisotropicDiffusionParams {
    /// Number of diffusion sweeps.
    pub iterations: usize,
    /// Time step per sweep. Must satisfy `Δt < 1/2^D` (0.0625 in 3D) for stability.
    pub time_step: f64,
    /// Conductance, as a multiple of the image's own RMS gradient. Larger values
    /// smooth across stronger edges.
    pub conductance: f64,
}

impl Default for AnisotropicDiffusionParams {
    fn default() -> Self {
        // ITK's own defaults.
        Self { iterations: 2, time_step: 0.125, conductance: 1.0 }
    }
}

/// Clamped (zero-flux Neumann) neighbour offset along one axis.
#[inline]
fn clamp_step(pos: usize, delta: isize, n: usize) -> usize {
    let p = pos as isize + delta;
    if p < 0 {
        0
    } else if p as usize >= n {
        n - 1
    } else {
        p as usize
    }
}

/// Mean squared central-difference gradient magnitude over the volume.
///
/// This is ITK's `CalculateAverageGradientMagnitudeSquared`, and sets the
/// conductance scale `K` for the sweep that follows.
fn average_gradient_magnitude_squared(u: &[f64], nx: usize, ny: usize, nz: usize) -> f64 {
    let n_total = nx * ny * nz;
    if n_total == 0 {
        return 0.0;
    }

    let mut total = 0.0;
    for k in 0..nz {
        let kp = clamp_step(k, 1, nz);
        let km = clamp_step(k, -1, nz);
        for j in 0..ny {
            let jp = clamp_step(j, 1, ny);
            let jm = clamp_step(j, -1, ny);
            for i in 0..nx {
                let ip = clamp_step(i, 1, nx);
                let im = clamp_step(i, -1, nx);

                let dx = 0.5 * (u[ip + j * nx + k * nx * ny] - u[im + j * nx + k * nx * ny]);
                let dy = 0.5 * (u[i + jp * nx + k * nx * ny] - u[i + jm * nx + k * nx * ny]);
                let dz = 0.5 * (u[i + j * nx + kp * nx * ny] - u[i + j * nx + km * nx * ny]);

                total += dx * dx + dy * dy + dz * dz;
            }
        }
    }

    total / n_total as f64
}

/// Apply gradient anisotropic diffusion to a 3D volume.
///
/// # Arguments
/// * `data` - Input volume (`nx * ny * nz`, Fortran order)
/// * `grid` - Volume grid (only the dimensions are used; spacing is ignored, as in ITK)
/// * `params` - Iteration count, time step and conductance
///
/// # Returns
/// The smoothed volume. A perfectly flat input is returned unchanged.
pub fn gradient_anisotropic_diffusion(
    data: &[f64],
    grid: &Grid,
    params: &AnisotropicDiffusionParams,
) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let n_total = nx * ny * nz;
    assert_eq!(data.len(), n_total, "data length must match grid dimensions");

    let mut u = data.to_vec();
    if n_total == 0 || params.iterations == 0 {
        return u;
    }

    let mut next = vec![0.0; n_total];

    for _ in 0..params.iterations {
        let avg = average_gradient_magnitude_squared(&u, nx, ny, nz);
        let k_scale = -2.0 * params.conductance * params.conductance * avg;

        // A flat volume has nothing to diffuse, and dividing by K would be 0/0.
        if k_scale == 0.0 {
            return u;
        }

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let center = i + j * nx + k * nx * ny;
                    let mut delta = 0.0;

                    // One pass per axis: half-derivatives along the axis, plus the
                    // in-plane central derivatives that complete |∇u|² on each face.
                    for axis in 0..3 {
                        let (ip, jp, kp, im, jm, km) = match axis {
                            0 => (
                                clamp_step(i, 1, nx), j, k,
                                clamp_step(i, -1, nx), j, k,
                            ),
                            1 => (
                                i, clamp_step(j, 1, ny), k,
                                i, clamp_step(j, -1, ny), k,
                            ),
                            _ => (
                                i, j, clamp_step(k, 1, nz),
                                i, j, clamp_step(k, -1, nz),
                            ),
                        };

                        let fwd = ip + jp * nx + kp * nx * ny;
                        let bwd = im + jm * nx + km * nx * ny;

                        let dx_forward = u[fwd] - u[center];
                        let dx_backward = u[center] - u[bwd];

                        // Transverse gradient energy on the forward and backward faces,
                        // averaged between the centre voxel and the neighbour.
                        let mut accum_f = 0.0;
                        let mut accum_b = 0.0;
                        for other in 0..3 {
                            if other == axis {
                                continue;
                            }
                            let dims = (nx, ny, nz);
                            let dx_here = central(&u, (i, j, k), other, dims);
                            let dx_aug = central(&u, (ip, jp, kp), other, dims);
                            let dx_dim = central(&u, (im, jm, km), other, dims);

                            let sf = dx_here + dx_aug;
                            let sb = dx_here + dx_dim;
                            accum_f += 0.25 * sf * sf;
                            accum_b += 0.25 * sb * sb;
                        }

                        let c_f = ((dx_forward * dx_forward + accum_f) / k_scale).exp();
                        let c_b = ((dx_backward * dx_backward + accum_b) / k_scale).exp();

                        delta += dx_forward * c_f - dx_backward * c_b;
                    }

                    next[center] = u[center] + params.time_step * delta;
                }
            }
        }

        std::mem::swap(&mut u, &mut next);
    }

    u
}

/// Central difference along `axis` at `(i, j, k)`, with clamped boundaries.
#[inline]
fn central(
    u: &[f64],
    pos: (usize, usize, usize),
    axis: usize,
    dims: (usize, usize, usize),
) -> f64 {
    let (i, j, k) = pos;
    let (nx, ny, nz) = dims;
    let (plus, minus) = match axis {
        0 => (
            clamp_step(i, 1, nx) + j * nx + k * nx * ny,
            clamp_step(i, -1, nx) + j * nx + k * nx * ny,
        ),
        1 => (
            i + clamp_step(j, 1, ny) * nx + k * nx * ny,
            i + clamp_step(j, -1, ny) * nx + k * nx * ny,
        ),
        _ => (
            i + j * nx + clamp_step(k, 1, nz) * nx * ny,
            i + j * nx + clamp_step(k, -1, nz) * nx * ny,
        ),
    };
    0.5 * (u[plus] - u[minus])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(n: usize) -> Grid {
        Grid::new(n, n, n, 1.0, 1.0, 1.0)
    }

    #[test]
    fn constant_volume_is_unchanged() {
        let g = grid(6);
        let data = vec![3.0; g.n_total()];
        let out = gradient_anisotropic_diffusion(&data, &g, &AnisotropicDiffusionParams::default());
        for v in out {
            assert!((v - 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn zero_iterations_is_identity() {
        let g = grid(5);
        let data: Vec<f64> = (0..g.n_total()).map(|i| i as f64).collect();
        let params = AnisotropicDiffusionParams { iterations: 0, ..Default::default() };
        let out = gradient_anisotropic_diffusion(&data, &g, &params);
        assert_eq!(out, data);
    }

    #[test]
    fn noise_is_reduced_more_than_a_step_edge() {
        let n = 16;
        let g = grid(n);
        let n_total = g.n_total();

        // A step edge in x, plus deterministic small-amplitude ripple.
        let mut data = vec![0.0; n_total];
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let idx = i + j * n + k * n * n;
                    let step = if i < n / 2 { 0.0 } else { 10.0 };
                    let ripple = 0.1 * (((i * 7 + j * 13 + k * 29) % 5) as f64 - 2.0);
                    data[idx] = step + ripple;
                }
            }
        }

        let params = AnisotropicDiffusionParams {
            iterations: 5,
            time_step: 0.05,
            conductance: 1.0,
        };
        let out = gradient_anisotropic_diffusion(&data, &g, &params);

        // The edge survives: the jump across the middle stays near 10.
        let mid = n / 2;
        let left = out[(mid - 1) + mid * n + mid * n * n];
        let right = out[mid + mid * n + mid * n * n];
        assert!(
            (right - left) > 8.0,
            "step edge was smoothed away: {left} -> {right}"
        );

        // The ripple inside a flat region is damped.
        let ripple_before: f64 = (2..mid - 2)
            .map(|i| (data[i + mid * n + mid * n * n] - 0.0).abs())
            .sum();
        let ripple_after: f64 = (2..mid - 2)
            .map(|i| (out[i + mid * n + mid * n * n] - 0.0).abs())
            .sum();
        assert!(
            ripple_after < ripple_before,
            "ripple not damped: {ripple_before} -> {ripple_after}"
        );
    }
}

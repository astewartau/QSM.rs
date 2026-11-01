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
#[cfg(feature = "parallel")]
use rayon::prelude::*;

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
    /// ITK 3.20's arithmetic, for parity with code that runs its executable (qsm_heidi's
    /// HEIDI): a `float` image, and the mean squared gradient summed in `float` with the
    /// volume's edges and corners counted more than once
    /// ([`gradient_anisotropic_diffusion_itk`]). Off by default: the float sum underestimates
    /// the mean (by about 1 % on a whole-brain volume), and so `K`.
    pub reference_itk: bool,
}

impl Default for AnisotropicDiffusionParams {
    fn default() -> Self {
        // ITK's own defaults.
        Self { iterations: 2, time_step: 0.125, conductance: 1.0, reference_itk: false }
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
/// This is ITK's `CalculateAverageGradientMagnitudeSquared`, and sets the conductance scale `K`
/// for the sweep that follows. It is the true mean, in double precision: ITK 3.20 sums in
/// `float` and counts the voxels on the volume's edges and corners two or three times
/// ([`gradient_anisotropic_diffusion_itk`] does both).
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

/// Every voxel in the order ITK 3.20's `CalculateAverageGradientMagnitudeSquared` visits them:
/// the interior, then the faces x−, x+, y−, y+, z−, z+, each a full slab (edges and corners
/// repeat), in raster order.
fn for_each_itk_face_voxel((nx, ny, nz): (usize, usize, usize), visit: &mut dyn FnMut(usize, usize, usize)) {
    let mut region = |x: std::ops::Range<usize>, y: std::ops::Range<usize>, z: std::ops::Range<usize>| {
        for k in z {
            for j in y.clone() {
                for i in x.clone() {
                    visit(i, j, k);
                }
            }
        }
    };
    let inner = |n: usize| if n > 2 { 1..n - 1 } else { 0..0 };
    region(inner(nx), inner(ny), inner(nz));
    region(0..1, 0..ny, 0..nz);
    region(nx - 1..nx, 0..ny, 0..nz);
    region(0..nx, 0..1, 0..nz);
    region(0..nx, ny - 1..ny, 0..nz);
    region(0..nx, 0..ny, 0..1);
    region(0..nx, 0..ny, nz - 1..nz);
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
    if params.reference_itk {
        return gradient_anisotropic_diffusion_itk(data, grid, params);
    }
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

/// ITK 3.20's `GradientAnisotropicDiffusionImageFilter` on a `float` image, as the executable
/// qsm_heidi ships runs it: [`gradient_anisotropic_diffusion`]'s scheme, with ITK's arithmetic.
///
/// - The image is `float` (f32) throughout; each update is computed in double from f32
///   differences (`NumericTraits<float>::RealType`), stored as f32, and added as
///   `o += float(update · Δt)` (`DenseFiniteDifferenceImageFilter::ThreadedApplyUpdate`).
/// - `⟨|∇u|²⟩` is accumulated in f32 (`ScalarAnisotropicDiffusionFunction::
///   CalculateAverageGradientMagnitudeSquared`, `PixelType accumulator`), in ITK's order;
///   `K = float(⟨|∇u|²⟩ · c² · −2)`.
///
/// Voxel spacing is ignored and out-of-bounds neighbours are clamped, as in
/// [`gradient_anisotropic_diffusion`].
pub fn gradient_anisotropic_diffusion_itk(
    data: &[f64],
    grid: &Grid,
    params: &AnisotropicDiffusionParams,
) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let n_total = nx * ny * nz;
    assert_eq!(data.len(), n_total, "data length must match grid dimensions");
    let mut u: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    if n_total == 0 || params.iterations == 0 {
        return u.iter().map(|&v| v as f64).collect();
    }
    let idx = |i: usize, j: usize, k: usize| i + nx * (j + ny * k);
    let mut update = vec![0.0f32; n_total];
    for _ in 0..params.iterations {
        let avg = itk_average_gradient_magnitude_squared(&u, (nx, ny, nz));
        let k_scale = (avg * params.conductance * params.conductance * -2.0) as f32 as f64;
        let at = |u: &[f32], i: usize, j: usize, k: usize, d: [isize; 3]| -> f32 {
            u[idx(clamp_step(i, d[0], nx), clamp_step(j, d[1], ny), clamp_step(k, d[2], nz))]
        };
        crate::maybe_par_chunks_mut!(update, nx * ny).enumerate().for_each(|(k, slab)| {
            let u = &u;
            for j in 0..ny {
                for i in 0..nx {
                    let unit = |a: usize| -> [isize; 3] { let mut d = [0; 3]; d[a] = 1; d };
                    let c = u[idx(i, j, k)];
                    let mut dx = [0.0f64; 3];
                    for (a, dxa) in dx.iter_mut().enumerate() {
                        let e = unit(a);
                        let m = [-e[0], -e[1], -e[2]];
                        *dxa = ((at(u, i, j, k, e) - at(u, i, j, k, m)) / 2.0f32) as f64;
                    }
                    let mut delta = 0.0f64;
                    for a in 0..3 {
                        let e = unit(a);
                        let m = [-e[0], -e[1], -e[2]];
                        let fwd = (at(u, i, j, k, e) - c) as f64;
                        let bwd = (c - at(u, i, j, k, m)) as f64;
                        let (mut acc, mut acc_d) = (0.0f64, 0.0f64);
                        for b in 0..3 {
                            if b == a {
                                continue;
                            }
                            let f = unit(b);
                            let add = |x: [isize; 3], y: [isize; 3]| [x[0] + y[0], x[1] + y[1], x[2] + y[2]];
                            let neg = [-f[0], -f[1], -f[2]];
                            let aug = ((at(u, i, j, k, add(e, f)) - at(u, i, j, k, add(e, neg))) / 2.0f32) as f64;
                            let dim = ((at(u, i, j, k, add(m, f)) - at(u, i, j, k, add(m, neg))) / 2.0f32) as f64;
                            acc += 0.25 * (dx[b] + aug) * (dx[b] + aug);
                            acc_d += 0.25 * (dx[b] + dim) * (dx[b] + dim);
                        }
                        let (cx, cxd) = if k_scale == 0.0 {
                            (0.0, 0.0)
                        } else {
                            (((fwd * fwd + acc) / k_scale).exp(), ((bwd * bwd + acc_d) / k_scale).exp())
                        };
                        delta += fwd * cx - bwd * cxd;
                    }
                    slab[i + nx * j] = delta as f32;
                }
            }
        });
        for (o, &d) in u.iter_mut().zip(&update) {
            *o += (d as f64 * params.time_step) as f32;
        }
    }
    u.iter().map(|&v| v as f64).collect()
}

/// [`average_gradient_magnitude_squared`] in ITK's float arithmetic: f32 sums in ITK's order,
/// f32 division.
fn itk_average_gradient_magnitude_squared(u: &[f32], (nx, ny, nz): (usize, usize, usize)) -> f64 {
    let idx = |i: usize, j: usize, k: usize| i + nx * (j + ny * k);
    let (mut acc, mut count) = (0.0f32, 0.0f32);
    let mut visit = |i: usize, j: usize, k: usize| {
        count += 1.0;
        let pairs = [
            (idx(clamp_step(i, 1, nx), j, k), idx(clamp_step(i, -1, nx), j, k)),
            (idx(i, clamp_step(j, 1, ny), k), idx(i, clamp_step(j, -1, ny), k)),
            (idx(i, j, clamp_step(k, 1, nz)), idx(i, j, clamp_step(k, -1, nz))),
        ];
        for (p, m) in pairs {
            // `val / -2.0f` in float, through a double, back to float; squared and summed in float.
            let v = ((u[p] - u[m]) / -2.0f32) as f64 as f32;
            acc += v * v;
        }
    };
    for_each_itk_face_voxel((nx, ny, nz), &mut visit);
    (acc / count) as f64
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

    /// ITK's float arithmetic gives an f32 result close to the double one: its mean squared
    /// gradient differs slightly, and nothing else.
    #[test]
    fn reference_itk_is_float_and_close() {
        let g = grid(12);
        let data: Vec<f64> = (0..g.n_total()).map(|i| ((i as f64) * 0.37).sin() + (i % 7) as f64 * 0.1).collect();
        let p = AnisotropicDiffusionParams { iterations: 3, time_step: 0.05, conductance: 1.0, reference_itk: false };
        let a = gradient_anisotropic_diffusion(&data, &g, &p);
        let b = gradient_anisotropic_diffusion(&data, &g, &AnisotropicDiffusionParams { reference_itk: true, ..p });
        assert!(b.iter().all(|&v| v == v as f32 as f64), "an f32 image");
        assert_ne!(a, b);
        let max = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
        assert!(max < 1e-2, "{max}");
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
            reference_itk: false,
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

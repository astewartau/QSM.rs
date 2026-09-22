//! STI: susceptibility tensor imaging
//!
//! COSMOS combines orientations to make a *scalar* susceptibility well-posed. STI keeps the
//! orientations but changes the model: susceptibility becomes a symmetric rank-2 tensor, so
//! tissue whose magnetic response depends on direction — white matter, whose myelin sheaths
//! are diamagnetic along the fibre — is described rather than averaged away.
//!
//! Each orientation measures one projection of the tensor onto its own B0 direction:
//!
//! ```text
//!     delta_t(k) = (1/3) H_t^T X(k) H_t  -  (H_t . k) (k^T X(k) H_t) / |k|^2
//! ```
//!
//! With `X` symmetric there are six unknowns per voxel — `[X11, X12, X13, X22, X23, X33]` —
//! so at least six orientations are needed, and in practice ten or more are acquired for
//! conditioning. Setting `X = chi I` collapses the expression to `chi (1/3 - (H.k)^2/|k|^2)`,
//! the ordinary dipole kernel, which is the sense in which STI contains scalar QSM.
//!
//! # Why this is a per-voxel solve rather than a global one
//!
//! The reference implementations take two routes that look very different. STI Suite solves a
//! small least-squares problem at each k-point; Bilgic's toolbox builds a six-channel operator
//! and runs LSQR over the whole volume. They solve the same system: the operator couples the
//! six tensor channels to the N orientations **at each k-point independently**, with no
//! spatial coupling anywhere, so the global system is block-diagonal in k and separates into
//! `N_k` independent six-unknown problems. The per-k solve is therefore the exact answer, and
//! the global LSQR an early-stopped approximation of it.
//!
//! This module takes the direct route: form the 6x6 normal equations at each k-point and
//! solve by Cholesky. Exact, embarrassingly parallel, and with no iteration count to tune.
//! `lambda` adds a ridge term for the ill-conditioned case.
//!
//! # Units and gauge
//!
//! Local (tissue) field in ppm in, tensor components in ppm out. As with the scalar dipole
//! kernel there is no DC response, so the k = 0 term is null and every component is recovered
//! zero-mean.
//!
//! # A difference from the scalar dipole kernel
//!
//! [`crate::kernels::dipole::dipole_kernel`] keeps the Nyquist planes of an even-length axis;
//! this module drops them (see `is_nyquist_bin` below). The asymmetry is deliberate. Scalar
//! inversion divides by the kernel pointwise, so whatever convention the kernel picks at an
//! ambiguous bin cancels exactly between forward and inverse. STI instead solves a six-way
//! coupled system at each bin, where the same ambiguity leaves the components inseparable, so
//! the bins have to be excluded rather than carried through. An isotropic tensor therefore
//! reproduces the scalar dipole kernel exactly on an odd grid, and everywhere except those
//! planes on an even one.
//!
//! # Memory
//!
//! Every orientation's k-space is needed simultaneously, so this holds `N_orientations`
//! complex volumes at once — for twelve orientations of a 164x205x205 volume, about 1.3 GB.
//!
//! # References
//!
//! Liu C. "Susceptibility tensor imaging." Magnetic Resonance in Medicine. 2010;63(6):1471-7.
//! <https://doi.org/10.1002/mrm.22482>
//!
//! Li W, Liu C, Duong TQ, van Zijl PCM, Li X. "Susceptibility tensor imaging (STI) of the
//! brain." NMR in Biomedicine. 2017;30(4):e3540. <https://doi.org/10.1002/nbm.3540>
//!
//! Reference implementations: `STI_Parfor.m` (STI Suite v3.0, Chunlei Liu) and `apply_STI.m`
//! (COSMOS_STI_Toolbox, Berkin Bilgic). Ported from the published equations; the code was used
//! only to resolve component ordering and sign conventions, which the two agree on.

use crate::fft::{fft3d, fftfreq, ifft3d};
use crate::utils::denoise::jacobi_eigh;
use crate::Grid;
use num_complex::Complex64;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Number of independent components of a symmetric 3x3 tensor.
pub const N_TENSOR: usize = 6;

/// STI parameters.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct StiParams {
    /// Ridge term added to the diagonal of the 6x6 normal equations at each k-point.
    ///
    /// Zero is the plain least-squares solution and is appropriate when there are enough
    /// well-spread orientations to determine all six components. Raise it when orientations
    /// are few, clustered, or near-coplanar, where some k-points leave the six unknowns
    /// under-determined and the unregularized solve amplifies noise.
    pub lambda: f64,
}

impl Default for StiParams {
    fn default() -> Self {
        Self { lambda: 0.0 }
    }
}

/// A symmetric susceptibility tensor field, in ppm.
///
/// Components are stored in the order `[X11, X12, X13, X22, X23, X33]`, each a full volume
/// laid out like every other volume in the crate (`i + j*nx + k*nx*ny`).
#[derive(Clone, Debug)]
pub struct SusceptibilityTensor {
    pub components: [Vec<f64>; N_TENSOR],
}

impl SusceptibilityTensor {
    /// The full symmetric 3x3 matrix at voxel `i`, row-major.
    pub fn matrix_at(&self, i: usize) -> [f64; 9] {
        let [c11, c12, c13, c22, c23, c33] = [
            self.components[0][i],
            self.components[1][i],
            self.components[2][i],
            self.components[3][i],
            self.components[4][i],
            self.components[5][i],
        ];
        [c11, c12, c13, c12, c22, c23, c13, c23, c33]
    }
}

/// Rotationally invariant maps derived from the tensor's eigen-decomposition.
#[derive(Clone, Debug)]
pub struct TensorMaps {
    /// Eigenvalues per voxel in ASCENDING order: `[lambda_min, lambda_mid, lambda_max]`.
    pub eigenvalues: [Vec<f64>; 3],
    /// Mean magnetic susceptibility: the mean of the three eigenvalues. This is the
    /// orientation-independent counterpart of a scalar QSM map.
    pub mms: Vec<f64>,
    /// Magnetic susceptibility anisotropy: `lambda_max - (lambda_min + lambda_mid)/2`.
    pub msa: Vec<f64>,
    /// Principal eigenvector, the eigenvector of the LARGEST eigenvalue, as three component
    /// volumes `[x, y, z]` in voxel coordinates.
    ///
    /// Paired with the `msa` convention above, so the two describe the same axis. Note that
    /// which end of the spectrum is called "principal" differs between STI papers; this
    /// module is internally consistent and follows the reference implementations, but compare
    /// carefully before matching numbers against another toolbox.
    pub pev: [Vec<f64>; 3],
}

/// Build the row of the STI system for one orientation at one k-point.
///
/// This is the forward model written out against `[X11, X12, X13, X22, X23, X33]`. Both
/// reference implementations produce exactly this row.
#[inline]
fn sti_row(h: (f64, f64, f64), k: (f64, f64, f64), k2: f64) -> [f64; N_TENSOR] {
    let (h1, h2, h3) = h;
    let (kx, ky, kz) = k;
    // (H . k) / |k|^2 — the longitudinal projection that turns the tensor contraction into
    // the familiar dipole falloff.
    let hk = (h1 * kx + h2 * ky + h3 * kz) / k2;
    [
        h1 * h1 / 3.0 - h1 * kx * hk,
        2.0 * h1 * h2 / 3.0 - (h1 * ky + h2 * kx) * hk,
        2.0 * h1 * h3 / 3.0 - (h1 * kz + h3 * kx) * hk,
        h2 * h2 / 3.0 - h2 * ky * hk,
        2.0 * h2 * h3 / 3.0 - (h2 * kz + h3 * ky) * hk,
        h3 * h3 / 3.0 - h3 * kz * hk,
    ]
}

/// Whether a bin sits on a Nyquist plane, which only exists on even-length axes.
///
/// These bins are excluded from the reconstruction, for the same reason DC is: the model does
/// not determine them.
///
/// At a Nyquist bin `+f` and `-f` alias onto the same sample, so `fftfreq` must pick one (it
/// reports the negative) and the stored frequency no longer flips sign under the Hermitian
/// pairing `k -> -k`. That matters because the STI row is *not* even in `k` component-wise —
/// `(H.k)/|k|^2` is linear in `k`, so flipping one axis while leaving the others alone changes
/// the row. Away from Nyquist this never bites, since the Hermitian partner of a bin negates
/// every axis and the row is even under a full negation. At a Nyquist bin the partner flips
/// only the other axes, evenness fails, and the forward projection stops being Hermitian —
/// after which the imaginary part discarded by the inverse FFT is real signal.
///
/// Restoring evenness by averaging the row over both signs does make the projection Hermitian,
/// but the terms linear in that axis then cancel, leaving the six tensor components no longer
/// separable at those bins — the system becomes rank-deficient rather than merely awkward. The
/// information genuinely is not there, so these bins are zeroed rather than guessed at. They
/// are the outermost sampled frequencies, where a tissue field map carries almost no energy;
/// on a 164x205x205 volume exactly one axis is even, so this is one plane in 164.
#[inline]
fn is_nyquist_bin(i: usize, j: usize, k: usize, dims: (usize, usize, usize)) -> bool {
    let (nx, ny, nz) = dims;
    (nx % 2 == 0 && i == nx / 2)
        || (ny % 2 == 0 && j == ny / 2)
        || (nz % 2 == 0 && k == nz / 2)
}

/// Cholesky factorization of a 6x6 symmetric positive-definite matrix, in place.
/// Returns `false` if the matrix is not positive definite.
fn cholesky6(a: &mut [f64; 36]) -> bool {
    for i in 0..N_TENSOR {
        for j in 0..=i {
            let mut sum = a[i * N_TENSOR + j];
            for k in 0..j {
                sum -= a[i * N_TENSOR + k] * a[j * N_TENSOR + k];
            }
            if i == j {
                if sum <= 0.0 {
                    return false;
                }
                a[i * N_TENSOR + i] = sum.sqrt();
            } else {
                a[i * N_TENSOR + j] = sum / a[j * N_TENSOR + j];
            }
        }
    }
    true
}

/// Solve `L L^T x = b` in place given the lower factor from [`cholesky6`].
fn cholesky6_solve(l: &[f64; 36], b: &mut [f64; N_TENSOR]) {
    for i in 0..N_TENSOR {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i * N_TENSOR + k] * b[k];
        }
        b[i] = sum / l[i * N_TENSOR + i];
    }
    for i in (0..N_TENSOR).rev() {
        let mut sum = b[i];
        for k in (i + 1)..N_TENSOR {
            sum -= l[k * N_TENSOR + i] * b[k];
        }
        b[i] = sum / l[i * N_TENSOR + i];
    }
}

/// Reconstruct the susceptibility tensor from multi-orientation local field maps.
///
/// # Arguments
/// * `local_fields` - One local (tissue) field map per orientation, ppm, all on the same grid.
/// * `bdirs` - B0 direction for each orientation, in the voxel frame of that shared grid.
///   Derive with [`crate::geometry::b0_direction_from_affine`].
/// * `mask` - Binary mask (1 = inside ROI); applied to every output component.
/// * `grid` - The shared volume grid.
/// * `params` - See [`StiParams`].
///
/// # Panics
/// If fewer than [`N_TENSOR`] orientations are given (the tensor is then under-determined at
/// every k-point), if field and direction counts differ, or if any field does not match `grid`.
pub fn sti<F: AsRef<[f64]>>(
    local_fields: &[F],
    bdirs: &[(f64, f64, f64)],
    mask: &[u8],
    grid: &Grid,
    params: &StiParams,
) -> SusceptibilityTensor {
    let (nx, ny, nz) = grid.dims;
    let n = grid.n_total();
    let n_dir = local_fields.len();

    assert!(
        n_dir >= N_TENSOR,
        "sti: needs at least {} orientations to determine {} tensor components, got {}",
        N_TENSOR, N_TENSOR, n_dir
    );
    assert_eq!(
        n_dir, bdirs.len(),
        "sti: got {} field maps but {} B0 directions", n_dir, bdirs.len()
    );
    for (t, f) in local_fields.iter().enumerate() {
        assert_eq!(
            f.as_ref().len(), n,
            "sti: orientation {} has {} voxels, expected {}", t, f.as_ref().len(), n
        );
    }

    // Normalize the directions once; the forward model assumes unit vectors.
    let h: Vec<(f64, f64, f64)> = bdirs
        .iter()
        .map(|&(x, y, z)| {
            let m = (x * x + y * y + z * z).sqrt();
            assert!(m > 0.0, "sti: B0 direction must be non-zero");
            (x / m, y / m, z / m)
        })
        .collect();

    // k-space of every orientation, all needed at once.
    let spectra: Vec<Vec<Complex64>> = local_fields
        .iter()
        .map(|f| {
            let mut c: Vec<Complex64> =
                f.as_ref().iter().map(|&v| Complex64::new(v, 0.0)).collect();
            fft3d(&mut c, nx, ny, nz);
            c
        })
        .collect();

    let (kx, ky, kz) = (
        fftfreq(nx, grid.voxel_size.0),
        fftfreq(ny, grid.voxel_size.1),
        fftfreq(nz, grid.voxel_size.2),
    );

    // Solve each k-point independently into an interleaved 6-per-voxel buffer.
    let mut out = vec![Complex64::new(0.0, 0.0); n * N_TENSOR];
    let lambda = params.lambda;

    let solve_chunk = |(idx, slot): (usize, &mut [Complex64])| {
        let i = idx % nx;
        let j = (idx / nx) % ny;
        let k = idx / (nx * ny);
        let kvec = (kx[i], ky[j], kz[k]);
        let k2 = kvec.0 * kvec.0 + kvec.1 * kvec.1 + kvec.2 * kvec.2;

        // No DC response (as for the scalar dipole kernel), and no determinate solution on a
        // Nyquist plane — see [`is_nyquist_bin`].
        if k2 <= 1e-20 || is_nyquist_bin(i, j, k, grid.dims) {
            for s in slot.iter_mut() {
                *s = Complex64::new(0.0, 0.0);
            }
            return;
        }

        // Normal equations: A^T A and A^T b, accumulated over orientations.
        let mut ata = [0.0f64; 36];
        let mut atb = [Complex64::new(0.0, 0.0); N_TENSOR];
        for (t, &hv) in h.iter().enumerate() {
            let row = sti_row(hv, kvec, k2);
            let b = spectra[t][idx];
            for a in 0..N_TENSOR {
                atb[a] += b * row[a];
                for c in 0..=a {
                    ata[a * N_TENSOR + c] += row[a] * row[c];
                }
            }
        }
        for a in 0..N_TENSOR {
            ata[a * N_TENSOR + a] += lambda;
        }

        if !cholesky6(&mut ata) {
            // Under-determined at this k-point and no ridge to rescue it: contribute nothing
            // rather than amplifying noise through a singular system.
            for s in slot.iter_mut() {
                *s = Complex64::new(0.0, 0.0);
            }
            return;
        }

        // The system matrix is real, so the real and imaginary halves share one factorization.
        let mut re = [0.0f64; N_TENSOR];
        let mut im = [0.0f64; N_TENSOR];
        for a in 0..N_TENSOR {
            re[a] = atb[a].re;
            im[a] = atb[a].im;
        }
        cholesky6_solve(&ata, &mut re);
        cholesky6_solve(&ata, &mut im);
        for a in 0..N_TENSOR {
            slot[a] = Complex64::new(re[a], im[a]);
        }
    };

    #[cfg(feature = "parallel")]
    out.par_chunks_mut(N_TENSOR).enumerate().for_each(solve_chunk);
    #[cfg(not(feature = "parallel"))]
    out.chunks_mut(N_TENSOR).enumerate().for_each(solve_chunk);

    // De-interleave, transform back, mask.
    let mut components: [Vec<f64>; N_TENSOR] = Default::default();
    for (a, comp) in components.iter_mut().enumerate() {
        let mut channel: Vec<Complex64> = (0..n).map(|i| out[i * N_TENSOR + a]).collect();
        ifft3d(&mut channel, nx, ny, nz);
        *comp = channel
            .iter()
            .zip(mask.iter())
            .map(|(c, &m)| if m == 0 { 0.0 } else { c.re })
            .collect();
    }

    SusceptibilityTensor { components }
}

/// Forward STI model: the local field each orientation would measure from `tensor`.
///
/// The inverse of [`sti`], useful for simulation and for checking a reconstruction against
/// the data it came from.
pub fn sti_forward(
    tensor: &SusceptibilityTensor,
    bdirs: &[(f64, f64, f64)],
    grid: &Grid,
) -> Vec<Vec<f64>> {
    let (nx, ny, nz) = grid.dims;
    let n = grid.n_total();

    let spectra: Vec<Vec<Complex64>> = tensor
        .components
        .iter()
        .map(|c| {
            let mut z: Vec<Complex64> = c.iter().map(|&v| Complex64::new(v, 0.0)).collect();
            fft3d(&mut z, nx, ny, nz);
            z
        })
        .collect();

    let (kxs, kys, kzs) = (
        fftfreq(nx, grid.voxel_size.0),
        fftfreq(ny, grid.voxel_size.1),
        fftfreq(nz, grid.voxel_size.2),
    );

    bdirs
        .iter()
        .map(|&(x, y, z)| {
            let m = (x * x + y * y + z * z).sqrt();
            let hv = (x / m, y / m, z / m);
            let mut field = vec![Complex64::new(0.0, 0.0); n];
            for idx in 0..n {
                let i = idx % nx;
                let j = (idx / nx) % ny;
                let k = idx / (nx * ny);
                let kvec = (kxs[i], kys[j], kzs[k]);
                let k2 = kvec.0 * kvec.0 + kvec.1 * kvec.1 + kvec.2 * kvec.2;
                if k2 <= 1e-20 || is_nyquist_bin(i, j, k, grid.dims) {
                    continue;
                }
                let row = sti_row(hv, kvec, k2);
                let mut acc = Complex64::new(0.0, 0.0);
                for a in 0..N_TENSOR {
                    acc += spectra[a][idx] * row[a];
                }
                field[idx] = acc;
            }
            ifft3d(&mut field, nx, ny, nz);
            field.iter().map(|c| c.re).collect()
        })
        .collect()
}

/// Eigen-decompose the tensor and derive MMS, MSA and the principal eigenvector.
///
/// Only voxels inside `mask` are decomposed; everything else is left zero.
pub fn tensor_maps(tensor: &SusceptibilityTensor, mask: &[u8]) -> TensorMaps {
    let n = tensor.components[0].len();
    let mut eigenvalues: [Vec<f64>; 3] = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
    let mut pev: [Vec<f64>; 3] = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
    let mut mms = vec![0.0; n];
    let mut msa = vec![0.0; n];

    for i in 0..n {
        if mask[i] == 0 {
            continue;
        }
        // Ascending eigenvalues; eigenvectors column-wise, so column 2 is the largest.
        let (d, v) = jacobi_eigh(&tensor.matrix_at(i), 3);
        for a in 0..3 {
            eigenvalues[a][i] = d[a];
            pev[a][i] = v[a * 3 + 2];
        }
        mms[i] = (d[0] + d[1] + d[2]) / 3.0;
        msa[i] = d[2] - (d[0] + d[1]) / 2.0;
    }

    TensorMaps { eigenvalues, mms, msa, pev }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::dipole::dipole_kernel;
    use crate::fft::apply_real_kernel;

    /// `n` B0 directions spread over the hemisphere on a Fibonacci spiral, varying both polar
    /// angle and azimuth so the six tensor components are actually separable.
    fn hemisphere_dirs(n: usize) -> Vec<(f64, f64, f64)> {
        let golden = std::f64::consts::PI * (3.0 - 5.0f64.sqrt());
        (0..n)
            .map(|t| {
                let z = 1.0 - (t as f64) / (n as f64);
                let r = (1.0 - z * z).max(0.0).sqrt();
                let az = golden * t as f64;
                (r * az.cos(), r * az.sin(), z)
            })
            .collect()
    }

    /// A smooth zero-mean volume, scaled — DC is unrecoverable so the truth must have none.
    fn smooth_volume(grid: &Grid, scale: f64, phase: f64) -> Vec<f64> {
        let (nx, ny, nz) = grid.dims;
        let mut v = vec![0.0f64; grid.n_total()];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let (x, y, z) = (i as f64, j as f64, k as f64);
                    v[i + j * nx + k * nx * ny] =
                        scale * ((0.6 * x + phase).sin() * (0.4 * y).cos() + 0.5 * (0.35 * z + phase).sin());
                }
            }
        }
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        for x in v.iter_mut() {
            *x -= mean;
        }
        v
    }

    fn random_tensor(grid: &Grid) -> SusceptibilityTensor {
        SusceptibilityTensor {
            components: [
                smooth_volume(grid, 0.10, 0.0),
                smooth_volume(grid, 0.03, 1.1),
                smooth_volume(grid, 0.02, 2.2),
                smooth_volume(grid, 0.09, 3.3),
                smooth_volume(grid, 0.025, 4.4),
                smooth_volume(grid, 0.11, 5.5),
            ],
        }
    }

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
    }

    /// The forward model's keystone check: an isotropic tensor `chi I` must reproduce the
    /// ordinary scalar dipole kernel exactly. This validates `sti_row` against entirely
    /// independent code (`dipole_kernel`), so a sign or ordering slip cannot hide.
    ///
    /// Run on an odd grid, where neither operator excludes anything. On an even grid the two
    /// deliberately differ on the Nyquist planes, which `sti` drops and `dipole_kernel` keeps —
    /// see [`is_nyquist_bin`] and the note in the module docs.
    #[test]
    fn isotropic_tensor_reduces_to_the_dipole_kernel() {
        let grid = Grid::new(13, 13, 13, 1.0, 1.0, 1.0);
        let chi = smooth_volume(&grid, 0.1, 0.0);
        let zero = vec![0.0; grid.n_total()];
        let iso = SusceptibilityTensor {
            components: [chi.clone(), zero.clone(), zero.clone(), chi.clone(), zero, chi.clone()],
        };

        for bdir in [(0.0, 0.0, 1.0), (0.6, 0.0, 0.8), (0.3, -0.5, 0.81)] {
            let via_sti = &sti_forward(&iso, &[bdir], &grid)[0];
            let (nx, ny, nz) = grid.dims;
            let via_dipole = apply_real_kernel(&chi, &dipole_kernel(&grid, bdir), nx, ny, nz);
            let err = max_abs_diff(via_sti, &via_dipole);
            let scale = via_dipole.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            assert!(
                err / scale < 1e-12,
                "isotropic STI forward disagrees with the dipole kernel for {:?}: rel {}",
                bdir, err / scale
            );
        }
    }

    /// Forward-project a known tensor at enough orientations, invert, recover it.
    #[test]
    fn recovers_tensor_from_simulated_orientations() {
        // Odd dimensions, so there is no Nyquist plane and every sampled frequency is
        // determinate — this isolates the solver from the exclusion in `is_nyquist_bin`.
        let grid = Grid::new(13, 13, 13, 1.0, 1.0, 1.0);
        let mask = vec![1u8; grid.n_total()];
        let truth = random_tensor(&grid);
        let dirs = hemisphere_dirs(16);

        let fields = sti_forward(&truth, &dirs, &grid);
        let rec = sti(&fields, &dirs, &mask, &grid, &StiParams::default());

        for a in 0..N_TENSOR {
            let scale = truth.components[a].iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let rel = max_abs_diff(&rec.components[a], &truth.components[a]) / scale;
            assert!(rel < 1e-12, "component {} not recovered: rel err {}", a, rel);
        }
    }

    #[test]
    fn recovers_tensor_with_anisotropic_voxels() {
        let grid = Grid::new(13, 13, 11, 0.9, 0.9, 2.0);
        let mask = vec![1u8; grid.n_total()];
        let truth = random_tensor(&grid);
        let dirs = hemisphere_dirs(16);

        let fields = sti_forward(&truth, &dirs, &grid);
        let rec = sti(&fields, &dirs, &mask, &grid, &StiParams::default());

        for a in 0..N_TENSOR {
            let scale = truth.components[a].iter().fold(0.0f64, |m, v| m.max(v.abs()));
            assert!(max_abs_diff(&rec.components[a], &truth.components[a]) / scale < 1e-12);
        }
    }

    /// An isotropic tensor must come back with zero anisotropy and MMS equal to the scalar.
    #[test]
    fn isotropic_tensor_has_no_anisotropy() {
        let grid = Grid::new(10, 10, 10, 1.0, 1.0, 1.0);
        let n = grid.n_total();
        let mask = vec![1u8; n];
        let chi = smooth_volume(&grid, 0.1, 0.0);
        let zero = vec![0.0; n];
        let iso = SusceptibilityTensor {
            components: [chi.clone(), zero.clone(), zero.clone(), chi.clone(), zero, chi.clone()],
        };

        let maps = tensor_maps(&iso, &mask);
        let scale = chi.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(max_abs_diff(&maps.mms, &chi) / scale < 1e-12, "MMS should equal the scalar");
        assert!(
            maps.msa.iter().all(|v| v.abs() / scale < 1e-12),
            "an isotropic tensor must have zero MSA"
        );
    }

    /// A tensor built with a known principal axis must report that axis, and an MSA equal to
    /// the anisotropy it was built with.
    #[test]
    fn recovers_known_anisotropy_and_principal_axis() {
        let grid = Grid::new(8, 8, 8, 1.0, 1.0, 1.0);
        let n = grid.n_total();
        let mask = vec![1u8; n];

        // Diagonal tensor, largest along y: eigenvalues (0.02, 0.05, 0.09) with the maximum
        // on the y axis, so MSA = 0.09 - (0.02 + 0.05)/2 = 0.055.
        let t = SusceptibilityTensor {
            components: [
                vec![0.02; n], vec![0.0; n], vec![0.0; n],
                vec![0.09; n], vec![0.0; n], vec![0.05; n],
            ],
        };
        let maps = tensor_maps(&t, &mask);

        for i in 0..n {
            assert!((maps.mms[i] - (0.02 + 0.09 + 0.05) / 3.0).abs() < 1e-12);
            assert!((maps.msa[i] - 0.055).abs() < 1e-12, "MSA {} != 0.055", maps.msa[i]);
            // Principal axis is +/- y.
            assert!(maps.pev[0][i].abs() < 1e-9 && maps.pev[2][i].abs() < 1e-9);
            assert!((maps.pev[1][i].abs() - 1.0).abs() < 1e-9);
        }
    }

    /// Eigenvalues must come back ascending, which the MSA and PEV conventions rely on.
    #[test]
    fn eigenvalues_are_ascending() {
        let grid = Grid::new(6, 6, 6, 1.0, 1.0, 1.0);
        let mask = vec![1u8; grid.n_total()];
        let maps = tensor_maps(&random_tensor(&grid), &mask);
        for i in 0..grid.n_total() {
            assert!(maps.eigenvalues[0][i] <= maps.eigenvalues[1][i]);
            assert!(maps.eigenvalues[1][i] <= maps.eigenvalues[2][i]);
        }
    }

    #[test]
    fn masked_voxels_are_zero() {
        let grid = Grid::new(8, 8, 8, 1.0, 1.0, 1.0);
        let n = grid.n_total();
        let mut mask = vec![1u8; n];
        mask[0] = 0;
        mask[3] = 0;
        let truth = random_tensor(&grid);
        let dirs = hemisphere_dirs(12);
        let fields = sti_forward(&truth, &dirs, &grid);

        let rec = sti(&fields, &dirs, &mask, &grid, &StiParams::default());
        for a in 0..N_TENSOR {
            assert_eq!(rec.components[a][0], 0.0);
            assert_eq!(rec.components[a][3], 0.0);
        }
    }

    /// The ridge term must actually damp, and must not be silently ignored.
    #[test]
    fn lambda_shrinks_the_solution() {
        let grid = Grid::new(8, 8, 8, 1.0, 1.0, 1.0);
        let mask = vec![1u8; grid.n_total()];
        let truth = random_tensor(&grid);
        let dirs = hemisphere_dirs(8);
        let fields = sti_forward(&truth, &dirs, &grid);

        let energy = |t: &SusceptibilityTensor| -> f64 {
            t.components.iter().flat_map(|c| c.iter()).map(|v| v * v).sum()
        };
        let plain = sti(&fields, &dirs, &mask, &grid, &StiParams::default());
        let reg = sti(&fields, &dirs, &mask, &grid, &StiParams { lambda: 0.05 });
        assert!(energy(&reg) < energy(&plain), "lambda did not damp the solution");
    }

    #[test]
    fn cholesky_solves_a_known_system() {
        // A = L L^T for a simple lower-triangular L, so the answer is known independently.
        let mut a = [0.0f64; 36];
        for i in 0..6 {
            for j in 0..6 {
                a[i * 6 + j] = if i == j { (i as f64) + 2.0 } else { 0.5 };
            }
        }
        let x_true = [1.0, -2.0, 3.0, 0.5, -1.5, 2.5];
        let mut b = [0.0f64; 6];
        for i in 0..6 {
            for j in 0..6 {
                b[i] += a[i * 6 + j] * x_true[j];
            }
        }
        let mut l = a;
        assert!(cholesky6(&mut l), "matrix should be positive definite");
        cholesky6_solve(&l, &mut b);
        for i in 0..6 {
            assert!((b[i] - x_true[i]).abs() < 1e-10, "x[{}] = {} != {}", i, b[i], x_true[i]);
        }
    }

    #[test]
    fn matrix_at_is_symmetric() {
        let grid = Grid::new(4, 4, 4, 1.0, 1.0, 1.0);
        let t = random_tensor(&grid);
        let m = t.matrix_at(7);
        assert_eq!(m[1], m[3]);
        assert_eq!(m[2], m[6]);
        assert_eq!(m[5], m[7]);
    }

    /// On an even-length axis the outermost sampled frequency is not determined by the model,
    /// so it must come back empty rather than filled with a guess. See `is_nyquist_bin`.
    #[test]
    fn nyquist_planes_are_excluded_on_even_grids() {
        let grid = Grid::new(12, 12, 12, 1.0, 1.0, 1.0);
        let (nx, ny, nz) = grid.dims;
        let n = grid.n_total();
        let mask = vec![1u8; n];
        let dirs = hemisphere_dirs(16);
        let fields = sti_forward(&random_tensor(&grid), &dirs, &grid);
        let rec = sti(&fields, &dirs, &mask, &grid, &StiParams::default());

        for comp in rec.components.iter() {
            let mut spec: Vec<Complex64> =
                comp.iter().map(|&v| Complex64::new(v, 0.0)).collect();
            crate::fft::fft3d(&mut spec, nx, ny, nz);
            let scale = spec.iter().map(|c| c.norm()).fold(0.0, f64::max);
            for (idx, c) in spec.iter().enumerate() {
                let (i, j, k) = (idx % nx, (idx / nx) % ny, idx / (nx * ny));
                if is_nyquist_bin(i, j, k, grid.dims) {
                    assert!(
                        c.norm() / scale < 1e-12,
                        "Nyquist bin ({},{},{}) should be empty, got {:.3e}",
                        i, j, k, c.norm()
                    );
                }
            }
        }
    }

    /// Odd-length axes have no Nyquist bin, so nothing is excluded there.
    #[test]
    fn odd_grids_have_no_excluded_bins() {
        let dims = (13, 13, 13);
        for idx in 0..(13 * 13 * 13) {
            let (i, j, k) = (idx % 13, (idx / 13) % 13, idx / 169);
            assert!(!is_nyquist_bin(i, j, k, dims));
        }
        // A mixed grid excludes only the plane belonging to its even axis.
        assert!(is_nyquist_bin(6, 0, 0, (12, 13, 13)));
        assert!(!is_nyquist_bin(0, 6, 0, (12, 13, 13)));
    }

    #[test]
    #[should_panic(expected = "at least 6 orientations")]
    fn rejects_too_few_orientations() {
        let grid = Grid::new(6, 6, 6, 1.0, 1.0, 1.0);
        let n = grid.n_total();
        let fields: Vec<Vec<f64>> = (0..5).map(|_| vec![0.0; n]).collect();
        let dirs = hemisphere_dirs(5);
        sti(&fields, &dirs, &vec![1u8; n], &grid, &StiParams::default());
    }

    #[test]
    #[should_panic(expected = "but 7 B0 directions")]
    fn rejects_mismatched_direction_count() {
        let grid = Grid::new(6, 6, 6, 1.0, 1.0, 1.0);
        let n = grid.n_total();
        let fields: Vec<Vec<f64>> = (0..8).map(|_| vec![0.0; n]).collect();
        sti(&fields, &hemisphere_dirs(7), &vec![1u8; n], &grid, &StiParams::default());
    }

    #[test]
    #[should_panic(expected = "orientation 2 has")]
    fn rejects_wrong_sized_field() {
        let grid = Grid::new(6, 6, 6, 1.0, 1.0, 1.0);
        let n = grid.n_total();
        let mut fields: Vec<Vec<f64>> = (0..8).map(|_| vec![0.0; n]).collect();
        fields[2] = vec![0.0; n - 1];
        sti(&fields, &hemisphere_dirs(8), &vec![1u8; n], &grid, &StiParams::default());
    }
}

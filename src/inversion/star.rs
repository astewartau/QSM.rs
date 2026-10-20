//! STAR-QSM: streaking artifact reduction for QSM
//!
//! [`star_qsm`] implements **STI Suite 3.0's `QSM_star`** (Wei et al. 2015), as called with
//! `QSM_star(phase, mask, 'TE', TE, 'B0', B0, 'H', H, 'padsize', [p p p], 'voxelsize', vs)`
//! with `p` = [`StarParams::pad`] (STI's default, [`DEFAULT_PAD`] = 12 voxels).
//! STI Suite ships as protected P-code, so the algorithm was recovered by black-box probing
//! (call tracing through logging wrappers around the functions it calls, capturing the arrays
//! it hands them, then a MATLAB re-implementation that reproduces `QSM_star` to within 4e-16
//! relative on phantoms and an in-vivo field):
//!
//! 1. Zero-pad the phase `φ` (radians at the echo time) and the mask `M` by `pad` voxels on
//!    both sides of every axis. STI's dipole kernel errors on odd extents, so `QSM_star` does
//!    too; here an odd padded extent grows by one more voxel at the high end instead.
//!    Everything below runs on that grid.
//! 2. Dipole kernel `D = 1/3 − (k·H)²/|k|²` (`D = 0` at `k = 0`) with `k` in cycles/mm
//!    (computed as STI does, to the bit: see `sti_kernel`), and
//!    the k-space mask `K = |D| ≥ 0.065`. STI builds both on a centred k-grid and transforms
//!    with centred FFTs (`fftshift ∘ fftn ∘ fftshift`, unitary); on an even grid every operator
//!    below is the same as the plain FFT convolution used here.
//! 3. **Initial guess**: `z = M·F⁻¹(D⁺ F(Mφ))` with `D⁺ = 1/D` on `K` and `±1/0.065` off it
//!    (the sign of `D`, `+` where `D = 0`); then twice `z ← M·F⁻¹(K·F z)`; then
//!    `x₀ = z / max|z|`. All of it stays complex.
//! 4. **Solve** `min ½‖A x − Mφ‖² + τ‖x‖₁` (`A x = F⁻¹(D F(M x))`, `τ = 10⁻⁶`, complex `x`)
//!    from `x₀` with SpaRSA 2.0 (Wright, Nowak & Figueiredo) at the options STI passes:
//!    `'AT'` = `A` itself (not its adjoint `M F⁻¹ D F`), stopping criterion 0 (stop once fewer
//!    than 1 % of the components change between zero and non-zero, after at least 5 and at
//!    most 1000 iterations), and SpaRSA's defaults otherwise: Barzilai–Borwein step
//!    `α = ‖A Δx‖²/‖Δx‖²` clamped to [10⁻³⁰, 10³⁰], `α = 1` at the start, no monotone or
//!    safeguard test (every step is accepted), no continuation, no debiasing, and SpaRSA's
//!    complex soft threshold `x ↦ max(|x| − τ/α, 0)/(max(|x| − τ/α, 0) + τ/α) · x`. STI also
//!    passes a TV operator as `'PHI'`; SpaRSA ignores a `'PHI'` given without `'PSI'`.
//!    If `τ ≥ max|A φ|` SpaRSA returns zero.
//! 5. `χ = 1.1 · Re(x) / (2π·γ·B0·TE)` with γ = 42.575 MHz/T (STI's `ScalingFactor`), cropped
//!    back to the input grid and masked. The factor 1.1 is a constant in STI's code.
//!
//! Iterations: 13 on the 48×56×40 phantoms below, 4–5 on in-vivo brains.
//!
//! # Units
//!
//! Like the other inversions, [`star_qsm`] takes the field in ppm and returns χ in ppm, but
//! unlike them the result depends on the field's scale: `x₀` is normalised to a peak of one
//! (in radians), `τ` is fixed and the solve stops after a handful of iterations, so the same
//! field in ppm gives a different χ at another `TE` or `B0`. [`StarParams::te`] and
//! [`StarParams::b0`] are therefore part of the method: the field is converted to the phase
//! `φ = 2π·42.575·B0·TE · field` it would have at that echo, which is what STI is given. With
//! that conversion `star_qsm(field)` is `QSM_star(φ)` exactly (to rounding), 1.1 included.
//!
//! # Agreement with STI Suite
//!
//! On six 48×56×40 phantoms (3 T and 7 T, TE 5–20 ms, axial and oblique B0, voxels
//! 0.8×0.9×1.5 mm, `pad` 6, 8 and 12) and an in-vivo 7 T local field (164×206×206), χ agrees
//! with STI Suite's output to the last few bits (relative L2 difference ≤ 1e-14), with the same
//! iteration counts. `tests/star_sti_parity.rs` checks three of the phantoms.
//!
//! # Performance
//!
//! Each SpaRSA iteration costs four 3-D FFTs on the padded grid (`A` twice); the forward
//! transforms skip the zero rows outside the mask's bounding box and multiply by `D` in their
//! last pass. The initial guess and setup cost ten more. Memory is three complex and one real
//! array on the padded grid.
//!
//! Reference:
//! Wei, H., Dibb, R., Zhou, Y., Sun, Y., Xu, J., Wang, N., Liu, C. (2015). "Streaking artifact
//! reduction for quantitative susceptibility mapping of sources with large dynamic range."
//! NMR in Biomedicine, 28(10):1294-1303. <https://doi.org/10.1002/nbm.3383>
//!
//! Wright, S.J., Nowak, R.D., Figueiredo, M.A.T. (2009). "Sparse reconstruction by separable
//! approximation." IEEE Trans. Signal Process. 57(7):2479-2493.
//! <https://doi.org/10.1109/TSP.2009.2016892>

use num_complex::Complex64;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use crate::fft::{Box3, Fft3dWorkspace, FftOpts};
use crate::Grid;

/// Parameters for [`star_qsm`]. The defaults are STI Suite's (`QSM_star` without options).
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct StarParams {
    /// Zero padding per side in voxels, on every axis: STI's `'padsize'` (default 12).
    pub pad: usize,
    /// Echo time in seconds the field is converted to phase at (default 0.04, STI's 40 ms).
    /// See the module docs, *Units*; the pipeline sets it from the first echo time.
    pub te: f64,
    /// Field strength in tesla (default 3). The pipeline sets it from the scan metadata.
    pub b0: f64,
}

impl Default for StarParams {
    fn default() -> Self {
        Self { pad: DEFAULT_PAD, te: 0.04, b0: 3.0 }
    }
}

impl StarParams {
    /// Checks the parameters: `te` and `b0` must be finite and > 0.
    pub fn validate(&self) -> Result<(), String> {
        if !(self.te.is_finite() && self.te > 0.0) {
            return Err(format!("STAR-QSM echo time must be finite and > 0 s, got {}", self.te));
        }
        if !(self.b0.is_finite() && self.b0 > 0.0) {
            return Err(format!("STAR-QSM field strength must be finite and > 0 T, got {}", self.b0));
        }
        Ok(())
    }

    /// Radians of phase per ppm of field at this TE and B0, with STI's γ = 42.575 MHz/T.
    pub fn rad_per_ppm(&self) -> f64 {
        2.0 * std::f64::consts::PI * STI_GAMMA_MHZ_PER_T * self.b0 * self.te
    }
}

/// STI's default zero padding of `QSM_star`, in voxels per side.
pub const DEFAULT_PAD: usize = 12;

/// γ/2π in MHz/T as STI Suite's `ScalingFactor` uses it.
const STI_GAMMA_MHZ_PER_T: f64 = 42.575;
/// The constant STI multiplies the solution by.
const STI_CHI_FACTOR: f64 = 1.1;
/// SpaRSA's L1 weight `τ` as STI passes it.
const TAU: f64 = 1e-6;
/// `|D|` below which the initial guess clamps `1/D`, and outside which `K` keeps k-space.
const KSPACE_THRESHOLD: f64 = 0.065;
/// SpaRSA's `MaxiterA` as STI passes it.
const MAX_ITER: usize = 1000;
/// SpaRSA's default `MiniterA`.
const MIN_ITER: usize = 5;
/// SpaRSA's default `ToleranceA` (stopping criterion 0: fraction of components changing).
const TOL_A: f64 = 0.01;
/// SpaRSA's default `AlphaMin` / `AlphaMax`.
const ALPHA_MIN: f64 = 1e-30;
const ALPHA_MAX: f64 = 1e30;

/// Chunk length of the reductions: fixed, so every sum is independent of the thread count.
const CHUNK: usize = 1 << 16;

/// STAR-QSM as in STI Suite 3.0 (`QSM_star`).
///
/// See the module docs for the algorithm. `field` is the local field in ppm, `bdir` the B0
/// direction in voxel axes (normalised here). `progress` is called after every SpaRSA
/// iteration with `(iteration, 1000)`.
///
/// # Returns
/// χ in ppm on the input grid, zero outside the mask.
///
/// # Panics
/// If `params` fails [`StarParams::validate`] (the pipeline checks this first and returns
/// `InvalidConfig`), or `field` / `mask` do not match `grid`.
pub fn star_qsm(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &StarParams,
    progress: impl FnMut(usize, usize),
) -> Vec<f64> {
    if let Err(e) = params.validate() {
        panic!("{e}");
    }
    let (nx, ny, nz) = grid.dims;
    let n_in = nx * ny * nz;
    assert_eq!(field.len(), n_in, "field does not match the grid");
    assert_eq!(mask.len(), n_in, "mask does not match the grid");
    let mut chi = vec![0.0; n_in];
    let p = params.pad;
    let n = padded_dims([nx, ny, nz], p);
    let nt = n[0] * n[1] * n[2];
    let src_dst = |i: usize, j: usize, k: usize| (i + j * nx + k * nx * ny, (i + p) + (j + p) * n[0] + (k + p) * n[0] * n[1]);

    // 1. padded masked phase, mask, and the mask's bounding box on the padded grid
    let rad = params.rad_per_ppm();
    let mut y = vec![0.0; nt];
    let mut inside = vec![false; nt];
    let (mut lo, mut hi) = ([usize::MAX; 3], [0usize; 3]);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let (s, d) = src_dst(i, j, k);
                if mask[s] != 0 {
                    y[d] = field[s] * rad;
                    inside[d] = true;
                    for (a, c) in [i + p, j + p, k + p].into_iter().enumerate() {
                        lo[a] = lo[a].min(c);
                        hi[a] = hi[a].max(c);
                    }
                }
            }
        }
    }
    if lo[0] == usize::MAX {
        return chi;
    }
    let support: Box3 = [0, 1, 2].map(|a| lo[a]..hi[a] + 1);

    // 2. dipole kernel on the padded grid
    let vs = grid.voxel_size;
    let dk = sti_kernel(n, [vs.0, vs.1, vs.2], bdir);
    let mut ws = Fft3dWorkspace::new(n[0], n[1], n[2]);

    // 3. initial guess
    let Some(mut x) = initial_guess(&y, &inside, &dk, &support, &mut ws) else {
        return chi;
    };

    // 4. SpaRSA with A = F⁻¹ D F M (also as Aᵀ)
    let mut apply_a = |v: &mut [Complex64]| {
        ws.fft3d_with(v, &FftOpts { support: Some(&support), post: Some(&dk), ..FftOpts::default() });
        ws.ifft3d(v);
    };
    sparsa(&mut x, &y, &inside, TAU, MAX_ITER, &mut apply_a, progress);

    // 5. scale, crop, mask
    let scale = STI_CHI_FACTOR * (1.0 / rad);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let (s, d) = src_dst(i, j, k);
                if mask[s] != 0 {
                    chi[s] = scale * x[d].re;
                }
            }
        }
    }
    chi
}

/// STI's dipole kernel (`calcD2Matrix`) in unshifted order: `D = 1/3 − (k·H)²/|k|²`, 0 at
/// `k = 0`, with `k_d = f/(n_d·vs_d)` for the signed frequency index `f` (−n/2 at Nyquist) and
/// `H` normalised.
///
/// It is [`dipole_kernel`](crate::kernels::dipole::dipole_kernel)'s formula, but evaluated the
/// way STI does, bit for bit: `k` by division rather than by multiplying with `1/(n·vs)`. The
/// two differ in the last bit of `k` on anisotropic grids, which flips the sign of `D` at
/// k-space points exactly on the magic-angle cone, and the initial guess's `±1/0.065` with it
/// (a 3e-6 relative change in χ on the 0.8×0.9×1.5 mm phantom).
fn sti_kernel(n: [usize; 3], vs: [f64; 3], bdir: (f64, f64, f64)) -> Vec<f64> {
    let hn = (bdir.0 * bdir.0 + bdir.1 * bdir.1 + bdir.2 * bdir.2).sqrt();
    let h = [bdir.0 / hn, bdir.1 / hn, bdir.2 / hn];
    let k: Vec<Vec<f64>> = (0..3)
        .map(|a| {
            (0..n[a])
                .map(|i| {
                    let f = if i < n[a] / 2 { i as f64 } else { i as f64 - n[a] as f64 };
                    f / (n[a] as f64 * vs[a])
                })
                .collect()
        })
        .collect();
    let mut d = vec![0.0; n[0] * n[1] * n[2]];
    crate::maybe_par_chunks_mut!(d, n[0] * n[1]).enumerate().for_each(|(kz, plane)| {
        let z = k[2][kz];
        for (l, v) in plane.iter_mut().enumerate() {
            let (x, y) = (k[0][l % n[0]], k[1][l / n[0]]);
            let k2 = x * x + y * y + z * z;
            if k2 != 0.0 {
                let kh = x * h[0] + y * h[1] + z * h[2];
                *v = 1.0 / 3.0 - kh * kh / k2;
            }
        }
    });
    d
}

/// Padded grid: `pad` voxels on both sides, plus one at the high end where that is odd.
fn padded_dims(dims: [usize; 3], pad: usize) -> [usize; 3] {
    dims.map(|c| c + 2 * pad + (c % 2))
}

/// Step 3: `x₀ = z / max|z|` with `z = M F⁻¹(K F(M F⁻¹(K F(M F⁻¹(D⁺ F y))))))`. `y` is zero
/// outside `inside`, which lies in `support`. `None` if `z` is zero.
fn initial_guess(
    y: &[f64],
    inside: &[bool],
    dk: &[f64],
    support: &Box3,
    ws: &mut Fft3dWorkspace,
) -> Option<Vec<Complex64>> {
    let mut z: Vec<Complex64> = crate::maybe_par_iter!(y).map(|&v| Complex64::new(v, 0.0)).collect();
    let opts = FftOpts { support: Some(support), ..FftOpts::default() };
    ws.fft3d_with(&mut z, &opts);
    crate::maybe_par_iter_mut!(z).zip(crate::maybe_par_iter!(dk)).for_each(|(v, &d)| *v *= dinv(d));
    ws.ifft3d(&mut z);
    mask_inplace(&mut z, inside);
    for _ in 0..2 {
        ws.fft3d_with(&mut z, &opts);
        crate::maybe_par_iter_mut!(z).zip(crate::maybe_par_iter!(dk)).for_each(|(v, &d)| {
            if d.abs() < KSPACE_THRESHOLD {
                *v = Complex64::new(0.0, 0.0);
            }
        });
        ws.ifft3d(&mut z);
        mask_inplace(&mut z, inside);
    }
    let peak = z.iter().map(|v| v.norm()).fold(0.0, f64::max);
    if peak == 0.0 {
        return None;
    }
    crate::maybe_par_iter_mut!(z).for_each(|v| *v /= peak);
    Some(z)
}

/// The initial guess's inverse kernel: `1/D` where `|D| ≥ 0.065`, else `±1/0.065` by the sign
/// of `D` (`+` at `D = 0`).
fn dinv(d: f64) -> f64 {
    if d.abs() >= KSPACE_THRESHOLD {
        1.0 / d
    } else if d < 0.0 {
        -1.0 / KSPACE_THRESHOLD
    } else {
        1.0 / KSPACE_THRESHOLD
    }
}

fn mask_inplace(z: &mut [Complex64], inside: &[bool]) {
    crate::maybe_par_iter_mut!(z).zip(crate::maybe_par_iter!(inside)).for_each(|(v, &m)| {
        if !m {
            *v = Complex64::new(0.0, 0.0);
        }
    });
}

/// SpaRSA's complex soft threshold (`soft.m`): `max(|v| − t, 0)/(max(|v| − t, 0) + t) · v`.
#[inline]
fn soft(v: Complex64, t: f64) -> Complex64 {
    if t == 0.0 {
        return v;
    }
    let s = (v.norm() - t).max(0.0);
    v * (s / (s + t))
}

#[inline]
fn nsq(z: Complex64) -> f64 {
    z.re * z.re + z.im * z.im
}

/// SpaRSA 2.0 at STI's options (module docs, step 4) for `A x = op(M x)`, `Aᵀ` = `A`, with
/// `y` zero outside `inside`. `x` holds the initial guess on entry and the solution on return.
/// `apply_core` applies `op` in place. Returns the number of iterations (0 if `τ ≥ max|A y|`,
/// when `x` is set to zero as SpaRSA does).
fn sparsa(
    x: &mut [Complex64],
    y: &[f64],
    inside: &[bool],
    tau: f64,
    max_iter: usize,
    apply_core: &mut dyn FnMut(&mut [Complex64]),
    mut progress: impl FnMut(usize, usize),
) -> usize {
    let nt = x.len();
    let zero = Complex64::new(0.0, 0.0);
    // resid and a work buffer (Aᵀ resid, then the masked step, then A·step)
    let mut r: Vec<Complex64> = crate::maybe_par_iter!(y).map(|&v| Complex64::new(v, 0.0)).collect();
    let mut g = vec![zero; nt];

    // SpaRSA returns zero when τ ≥ max|Aᵀy| (y is already masked)
    apply_core(&mut r);
    let max_tau = r.iter().map(|v| v.norm()).fold(0.0, f64::max);
    if tau >= max_tau {
        x.fill(zero);
        return 0;
    }

    // resid = A x − y; then g = M·resid, the input of Aᵀ resid
    crate::maybe_par_iter_mut!(r).zip(crate::maybe_par_iter!(x)).zip(crate::maybe_par_iter!(inside))
        .for_each(|((rv, &xv), &m)| *rv = if m { xv } else { zero });
    apply_core(&mut r);
    crate::maybe_par_iter_mut!(r).zip(crate::maybe_par_iter_mut!(g)).zip(crate::maybe_par_iter!(y)).zip(crate::maybe_par_iter!(inside))
        .for_each(|(((rv, gv), &yv), &m)| {
            *rv -= yv;
            *gv = if m { *rv } else { zero };
        });

    let mut nz: Vec<bool> = crate::maybe_par_iter!(x).map(|v| *v != zero).collect();
    let mut alpha = 1.0;
    let mut iter = 1usize;
    let mut keep_going = true;
    while keep_going {
        apply_core(&mut g); // gradq = Aᵀ resid

        // x ← soft(x − gradq/α, τ/α); g ← M·Δx; ‖Δx‖² and the active-set change
        let (inv_alpha, t) = (1.0 / alpha, tau / alpha);
        let parts: Vec<(f64, usize, usize)> = crate::maybe_par_chunks_mut!(x, CHUNK)
            .zip(crate::maybe_par_chunks_mut!(g, CHUNK))
            .zip(crate::maybe_par_chunks_mut!(nz, CHUNK))
            .zip(crate::maybe_par_chunks!(inside, CHUNK))
            .map(|(((xc, gc), nc), mc)| {
                let (mut dd, mut changed, mut nnz) = (0.0, 0usize, 0usize);
                for (((xv, gv), nv), &m) in xc.iter_mut().zip(gc.iter_mut()).zip(nc.iter_mut()).zip(mc) {
                    let xn = soft(*xv - *gv * inv_alpha, t);
                    let dx = xn - *xv;
                    *xv = xn;
                    dd += nsq(dx);
                    *gv = if m { dx } else { zero };
                    let now = xn != zero;
                    changed += (now != *nv) as usize;
                    nnz += now as usize;
                    *nv = now;
                }
                (dd, changed, nnz)
            })
            .collect();
        let dd: f64 = parts.iter().map(|p| p.0).sum();
        let changed: usize = parts.iter().map(|p| p.1).sum();
        let nnz: usize = parts.iter().map(|p| p.2).sum();

        apply_core(&mut g); // A Δx

        // resid += A Δx, ‖A Δx‖², then g = M·resid for the next Aᵀ resid
        let dgd: f64 = crate::maybe_par_chunks_mut!(r, CHUNK)
            .zip(crate::maybe_par_chunks_mut!(g, CHUNK))
            .zip(crate::maybe_par_chunks!(inside, CHUNK))
            .map(|((rc, gc), mc)| {
                let mut s = 0.0;
                for ((rv, gv), &m) in rc.iter_mut().zip(gc.iter_mut()).zip(mc) {
                    s += nsq(*gv);
                    *rv += *gv;
                    *gv = if m { *rv } else { zero };
                }
                s
            })
            .collect::<Vec<f64>>()
            .iter()
            .sum();

        alpha = (dgd / (f64::MIN_POSITIVE + dd)).max(ALPHA_MIN).min(ALPHA_MAX);
        iter += 1;
        if nnz >= 1 {
            keep_going = changed as f64 / nnz as f64 > TOL_A;
        }
        if iter <= MIN_ITER {
            keep_going = true;
        } else if iter > max_iter {
            keep_going = false;
        }
        progress(iter - 1, max_iter);
    }
    iter - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::dipole::dipole_kernel;
    use std::f64::consts::PI;

    /// fftshift index map on one axis (even n: its own inverse).
    fn shift(i: usize, n: usize) -> usize {
        (i + n / 2) % n
    }

    /// STI's kernel: `D = 1/3 − (k·H)²/|k|²` on the centred grid `k_d = (−n/2 … n/2−1)/(n·vs)`.
    fn centred_kernel(n: [usize; 3], vs: [f64; 3], h: [f64; 3]) -> Vec<f64> {
        let mut d = vec![0.0; n[0] * n[1] * n[2]];
        for k in 0..n[2] {
            for j in 0..n[1] {
                for i in 0..n[0] {
                    let kv = [i, j, k].iter().enumerate()
                        .map(|(a, &c)| (c as f64 - (n[a] / 2) as f64) / (n[a] as f64 * vs[a]))
                        .collect::<Vec<_>>();
                    let k2: f64 = kv.iter().map(|v| v * v).sum();
                    let kh: f64 = kv.iter().zip(h).map(|(a, b)| a * b).sum();
                    d[i + j * n[0] + k * n[0] * n[1]] = if k2 == 0.0 { 0.0 } else { 1.0 / 3.0 - kh * kh / k2 };
                }
            }
        }
        d
    }

    /// STI's centred unitary transform, `fftshift(fftn(fftshift(x)))/sqrt(N)` (or the inverse),
    /// as a direct sum.
    fn fftnc_naive(x: &[Complex64], n: [usize; 3], inverse: bool) -> Vec<Complex64> {
        let nt = x.len();
        let sgn = if inverse { 1.0 } else { -1.0 };
        let mut out = vec![Complex64::new(0.0, 0.0); nt];
        for (o, ov) in out.iter_mut().enumerate() {
            let oc = [o % n[0], (o / n[0]) % n[1], o / (n[0] * n[1])];
            // output index o of the shifted spectrum is frequency index shift(oc)
            let f = [0, 1, 2].map(|a| shift(oc[a], n[a]));
            let mut s = Complex64::new(0.0, 0.0);
            for (l, &xv) in x.iter().enumerate() {
                let lc = [l % n[0], (l / n[0]) % n[1], l / (n[0] * n[1])];
                let t = [0, 1, 2].map(|a| shift(lc[a], n[a]));
                let ph: f64 = (0..3).map(|a| (f[a] * t[a]) as f64 / n[a] as f64).sum();
                s += xv * Complex64::from_polar(1.0, sgn * 2.0 * PI * ph);
            }
            *ov = s / (nt as f64).sqrt();
        }
        out
    }

    #[test]
    fn kernel_is_stis_centred_kernel_shifted() {
        // even extents, anisotropic voxels, oblique H: the Nyquist rows (k = −n/2) carry the
        // sign of the cross terms, so they must sit at −1/(2·vs) as on STI's grid
        let (n, vs) = ([6, 8, 4], [0.8, 0.9, 1.5]);
        let h = [0.3, -0.4, 0.5f64.sqrt() * 1.2];
        let hn = (h[0] * h[0] + h[1] * h[1] + h[2] * h[2]).sqrt();
        let c = centred_kernel(n, vs, h.map(|v| v / hn));
        let d = sti_kernel(n, vs, (h[0], h[1], h[2]));
        let g = dipole_kernel(&Grid::new(n[0], n[1], n[2], vs[0], vs[1], vs[2]), (h[0], h[1], h[2]));
        for k in 0..n[2] {
            for j in 0..n[1] {
                for i in 0..n[0] {
                    let l = i + j * n[0] + k * n[0] * n[1];
                    let b = c[shift(i, n[0]) + shift(j, n[1]) * n[0] + shift(k, n[2]) * n[0] * n[1]];
                    assert_eq!(d[l], b, "({i},{j},{k})");
                    assert!((g[l] - b).abs() < 1e-15, "the library kernel is the same up to rounding");
                }
            }
        }
    }

    #[test]
    fn padding_is_symmetric_and_even() {
        assert_eq!(padded_dims([48, 56, 40], 12), [72, 80, 64]);
        assert_eq!(padded_dims([47, 56, 39], 6), [60, 68, 52]);
        assert_eq!(padded_dims([1, 2, 3], 0), [2, 2, 4]);
    }

    #[test]
    fn inverse_kernel_and_soft_threshold() {
        assert_eq!(dinv(0.5), 2.0);
        assert_eq!(dinv(-0.065), -1.0 / 0.065);
        assert_eq!(dinv(0.01), 1.0 / 0.065);
        assert_eq!(dinv(-0.01), -1.0 / 0.065);
        assert_eq!(dinv(0.0), 1.0 / 0.065);
        let v = Complex64::new(3.0, 4.0);
        assert_eq!(soft(v, 0.0), v);
        assert_eq!(soft(v, 5.0), Complex64::new(0.0, 0.0));
        assert_eq!(soft(v, 7.0), Complex64::new(0.0, 0.0));
        let s = soft(v, 1.0); // |v| = 5 → 4, same phase
        assert!((s - Complex64::new(2.4, 3.2)).norm() < 1e-15);
    }

    /// The initial guess as the spec writes it (centred transforms by direct sums, STI's
    /// centred kernel) against [`initial_guess`] (plain FFTs on the unshifted grid).
    #[test]
    fn initial_guess_matches_stis_centred_formulation() {
        let (n, vs) = ([4, 6, 4], [1.0, 0.9, 1.2]);
        let h = [0.0, 0.6, 0.8];
        let nt = n[0] * n[1] * n[2];
        let idx = |i: usize, j: usize, k: usize| i + j * n[0] + k * n[0] * n[1];
        let mut inside = vec![false; nt];
        let mut y = vec![0.0; nt];
        for k in 1..3 {
            for j in 1..5 {
                for i in 1..3 {
                    inside[idx(i, j, k)] = true;
                    y[idx(i, j, k)] = ((i * 7 + j * 3 + k * 5) % 11) as f64 * 0.1 - 0.4;
                }
            }
        }
        // spec: z = ifftnc(fftnc(y) .* Dinv) .* M, twice z = ifftnc(K .* fftnc(z)) .* M
        let c = centred_kernel(n, vs, h);
        let yc: Vec<Complex64> = y.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        let mut z: Vec<Complex64> = fftnc_naive(&yc, n, false).iter().zip(&c).map(|(v, &d)| v * dinv(d)).collect();
        z = fftnc_naive(&z, n, true);
        mask_inplace(&mut z, &inside);
        for _ in 0..2 {
            let f: Vec<Complex64> = fftnc_naive(&z, n, false).iter().zip(&c)
                .map(|(v, &d)| if d.abs() >= KSPACE_THRESHOLD { *v } else { Complex64::new(0.0, 0.0) }).collect();
            z = fftnc_naive(&f, n, true);
            mask_inplace(&mut z, &inside);
        }
        let peak = z.iter().map(|v| v.norm()).fold(0.0, f64::max);
        let want: Vec<Complex64> = z.iter().map(|v| v / peak).collect();

        let dk = sti_kernel(n, vs, (h[0], h[1], h[2]));
        let support: Box3 = [1..3, 1..5, 1..3];
        let mut ws = Fft3dWorkspace::new(n[0], n[1], n[2]);
        let got = initial_guess(&y, &inside, &dk, &support, &mut ws).unwrap();
        let peak_got = got.iter().map(|v| v.norm()).fold(0.0, f64::max);
        assert!((peak_got - 1.0).abs() < 1e-15);
        assert!(got.iter().any(|v| v.im.abs() > 1e-6), "the initial guess stays complex");
        for l in 0..nt {
            assert!((got[l] - want[l]).norm() < 1e-12, "{l}: {} vs {}", got[l], want[l]);
            if !inside[l] {
                assert_eq!(got[l], Complex64::new(0.0, 0.0));
            }
        }
        // a zero field has no initial guess
        assert!(initial_guess(&vec![0.0; nt], &inside, &dk, &support, &mut ws).is_none());
    }

    /// SpaRSA.m transcribed line by line (unfused, one array per MATLAB variable) at STI's
    /// options, for `A v = op(M v)` and `Aᵀ = A`.
    fn sparsa_reference(
        y: &[f64],
        inside: &[bool],
        op: &dyn Fn(&[Complex64]) -> Vec<Complex64>,
        tau: f64,
        max_iter: usize,
        x0: &[Complex64],
    ) -> (Vec<Complex64>, usize) {
        let zero = Complex64::new(0.0, 0.0);
        let a = |v: &[Complex64]| op(&v.iter().zip(inside).map(|(z, &m)| if m { *z } else { zero }).collect::<Vec<_>>());
        let yc: Vec<Complex64> = y.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        let max_tau = a(&yc).iter().map(|v| v.norm()).fold(0.0, f64::max);
        if tau >= max_tau {
            return (vec![zero; y.len()], 0);
        }
        let mut x = x0.to_vec();
        let mut nz_x: Vec<bool> = x.iter().map(|v| *v != zero).collect();
        let mut resid: Vec<Complex64> = a(&x).iter().zip(&yc).map(|(p, q)| p - q).collect();
        let (mut alpha, mut iter, mut keep_going) = (1.0f64, 1usize, true);
        while keep_going {
            let gradq = a(&resid);
            let prev_x = x.clone();
            let prev_resid = resid.clone();
            x = prev_x.iter().zip(&gradq).map(|(p, g)| soft(p - g * (1.0 / alpha), tau / alpha)).collect();
            let dx: Vec<Complex64> = x.iter().zip(&prev_x).map(|(p, q)| p - q).collect();
            let adx = a(&dx);
            resid = prev_resid.iter().zip(&adx).map(|(p, q)| p + q).collect();
            let dd: f64 = dx.iter().map(|v| nsq(*v)).sum();
            let dgd: f64 = adx.iter().map(|v| nsq(*v)).sum();
            alpha = ALPHA_MAX.min(ALPHA_MIN.max(dgd / (f64::MIN_POSITIVE + dd)));
            iter += 1;
            let nz_x_prev = std::mem::replace(&mut nz_x, x.iter().map(|v| v.norm() != 0.0).collect());
            let num_nz_x = nz_x.iter().filter(|&&b| b).count();
            let num_changes_active = nz_x.iter().zip(&nz_x_prev).filter(|(p, q)| p != q).count();
            if num_nz_x >= 1 {
                keep_going = num_changes_active as f64 / num_nz_x as f64 > TOL_A;
            }
            if iter <= MIN_ITER {
                keep_going = true;
            } else if iter > max_iter {
                keep_going = false;
            }
        }
        (x, iter - 1)
    }

    /// Circular convolution along the vector (a non-diagonal, non-symmetric `op`).
    fn conv(v: &[Complex64]) -> Vec<Complex64> {
        let h = [1.0, 0.45, -0.3, 0.1];
        let n = v.len();
        (0..n).map(|i| h.iter().enumerate().map(|(k, &c)| v[(i + n - k) % n] * c).sum()).collect()
    }

    fn small_problem(n: usize) -> (Vec<f64>, Vec<bool>, Vec<Complex64>) {
        let y: Vec<f64> = (0..n).map(|i| if i % 5 == 3 { 0.0 } else { ((i * 37 + 11) % 17) as f64 / 17.0 - 0.45 }).collect();
        let inside: Vec<bool> = (0..n).map(|i| i % 7 != 6).collect();
        let x0: Vec<Complex64> = (0..n).map(|i| if inside[i] { Complex64::new(((i * 13) % 7) as f64 * 0.1 - 0.3, ((i * 5) % 3) as f64 * 0.05) } else { Complex64::new(0.0, 0.0) }).collect();
        (y, inside, x0)
    }

    /// The fused solver takes the same steps as the transcription, including its stopping
    /// rule (active-set change ≤ 1 % after at least 5 iterations) and the cap.
    #[test]
    fn sparsa_matches_a_transcription_of_sparsa_m() {
        let mut lens = Vec::new();
        for (n, tau, cap) in [(40, 1e-3, 1000), (40, 3e-2, 1000), (64, 1e-6, 1000), (64, 2e-2, 1000), (64, 2e-2, 6)] {
            let (y, inside, x0) = small_problem(n);
            let (want, want_it) = sparsa_reference(&y, &inside, &conv, tau, cap, &x0);
            let mut x = x0.clone();
            let mut seen = Vec::new();
            let it = sparsa(&mut x, &y, &inside, tau, cap, &mut |v: &mut [Complex64]| { let c = conv(v); v.copy_from_slice(&c); }, |i, m| seen.push((i, m)));
            assert_eq!(it, want_it, "n {n} tau {tau}");
            assert_eq!(seen, (1..=it).map(|i| (i, cap)).collect::<Vec<_>>());
            for l in 0..n {
                assert!((x[l] - want[l]).norm() <= 1e-14 * (1.0 + want[l].norm()), "n {n} tau {tau} [{l}]: {} vs {}", x[l], want[l]);
            }
            lens.push(it);
        }
        assert!(lens.iter().all(|&i| i >= MIN_ITER));
        assert_eq!(lens[4], 6, "the cap ends the run");
        assert!(lens[3] > 6, "the capped problem runs longer uncapped: {lens:?}");
    }

    /// SpaRSA applies `MiniterA` before `MaxiterA`, so a cap below 5 still runs 5 iterations.
    #[test]
    fn sparsa_minimum_overrides_a_lower_cap() {
        let y = vec![0.5, 0.0, 1.0];
        let inside = vec![true; 3];
        let mut x = vec![Complex64::new(0.3, 0.0); 3];
        assert_eq!(sparsa(&mut x, &y, &inside, 1e-3, 3, &mut |_| {}, |_, _| {}), MIN_ITER);
    }

    #[test]
    fn sparsa_returns_zero_when_tau_dominates() {
        let y = vec![1e-7, 0.0];
        let inside = vec![true, true];
        let mut x = vec![Complex64::new(1.0, 0.0); 2];
        assert_eq!(sparsa(&mut x, &y, &inside, 1e-6, 1000, &mut |_| {}, |_, _| {}), 0);
        assert!(x.iter().all(|v| *v == Complex64::new(0.0, 0.0)));
    }

    #[test]
    fn empty_mask_zero_field_and_validation() {
        let grid = Grid::new(4, 4, 4, 1.0, 1.0, 1.0);
        let p = StarParams::default();
        assert_eq!(star_qsm(&[1.0; 64], &[0; 64], &grid, (0.0, 0.0, 1.0), &p, |_, _| {}), vec![0.0; 64]);
        assert_eq!(star_qsm(&[0.0; 64], &[1; 64], &grid, (0.0, 0.0, 1.0), &p, |_, _| {}), vec![0.0; 64]);
        assert!(p.validate().is_ok());
        for bad in [StarParams { te: 0.0, ..StarParams::default() }, StarParams { b0: f64::NAN, ..StarParams::default() }] {
            assert!(bad.validate().is_err());
        }
        assert!((StarParams { te: 0.02, b0: 3.0, pad: 0 }.rad_per_ppm() - 2.0 * PI * 42.575 * 0.06).abs() < 1e-12);
    }

    /// A sphere's field on an odd grid: the padded grid grows to even, and χ comes out with
    /// the sphere's sign and magnitude, zero outside the mask.
    #[test]
    fn recovers_a_sphere_on_an_odd_grid() {
        let (nx, ny, nz) = (15, 16, 13);
        let grid = Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);
        let nt = nx * ny * nz;
        let mut chi0 = vec![0.0; nt];
        let mut mask = vec![0u8; nt];
        let c = [7.0, 7.5, 6.0];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let r2 = (i as f64 - c[0]).powi(2) + (j as f64 - c[1]).powi(2) + (k as f64 - c[2]).powi(2);
                    let l = i + j * nx + k * nx * ny;
                    if r2 < 2.5f64.powi(2) { chi0[l] = 0.1; }
                    if r2 < 5.5f64.powi(2) { mask[l] = 1; }
                }
            }
        }
        // field of chi0 (circular convolution on an even grid big enough not to wrap)
        let (mx, my, mz) = (32, 32, 32);
        let big = Grid::new(mx, my, mz, 1.0, 1.0, 1.0);
        let dk = dipole_kernel(&big, (0.0, 0.0, 1.0));
        let mut buf = vec![Complex64::new(0.0, 0.0); mx * my * mz];
        for k in 0..nz { for j in 0..ny { for i in 0..nx { buf[i + j * mx + k * mx * my].re = chi0[i + j * nx + k * nx * ny]; } } }
        let mut ws = Fft3dWorkspace::new(mx, my, mz);
        ws.fft3d(&mut buf);
        buf.iter_mut().zip(&dk).for_each(|(v, &d)| *v *= d);
        ws.ifft3d(&mut buf);
        let field: Vec<f64> = (0..nt).map(|l| {
            let (i, j, k) = (l % nx, (l / nx) % ny, l / (nx * ny));
            if mask[l] != 0 { buf[i + j * mx + k * mx * my].re } else { 0.0 }
        }).collect();

        let mut iters = 0;
        let chi = star_qsm(&field, &mask, &grid, (0.0, 0.0, 1.0), &StarParams { pad: 6, ..StarParams::default() }, |i, _| iters = i);
        assert!(iters >= MIN_ITER);
        let centre = chi[7 + 7 * nx + 6 * nx * ny];
        assert!(centre > 0.05 && centre < 0.2, "centre χ {centre}");
        for l in 0..nt {
            if mask[l] == 0 { assert_eq!(chi[l], 0.0); }
        }
    }
}

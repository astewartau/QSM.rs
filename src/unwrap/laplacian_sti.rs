//! Laplacian phase unwrapping as STI Suite 3.0 implements it, and the per-echo
//! "unwrap, then weighted echo average" field map built on it.
//!
//! STI Suite's `MRPhaseUnwrap` (Li, Liu et al.) is the unwrapper behind several large
//! QSM pipelines, used once per echo before combining. It is a different discretisation
//! from [`super::laplacian::laplacian_unwrap`], so the two do not give the same field:
//!
//! | | [`laplacian_unwrap_sti`] | [`laplacian_unwrap`](super::laplacian_unwrap) |
//! |---|---|---|
//! | ∇² of the true phase | spectral, `cos φ·∇²(sin φ) − sin φ·∇²(cos φ)` | finite differences of wrapped neighbour differences |
//! | ∇² kernel | continuous, `|k|²` on the FFT grid | discrete, `2(cos(πk/N) − 1)/h²` |
//! | boundary | periodic, after zero-padding (default 12 voxels per side) | Neumann on the array, no padding |
//! | DC / offset | mean over the **padded** box set to 0 | mean over the array set to 0 |
//! | output | whole volume, unmasked | zero outside the mask |
//!
//! The two agree on the non-harmonic part of a well-sampled field and differ mostly in
//! low-order harmonic terms (which background removal takes out anyway), at the mask edge
//! where the phase jumps to zero, and wherever the phase changes by more than about one
//! radian per voxel (the spectral identity needs `sin φ`, `cos φ` to be resolved by the grid).
//!
//! On the simulated head used by the integration tests (7 T, 1 mm, four echoes, V-SHARP
//! afterwards) the local fields are within ±0.02 in r of each other, this one marginally
//! closer to the truth (NRMSE 79.5 % against 81.3 %). The *total* field is a different
//! matter: zero-padding puts a step wherever the phase at the array edge is not zero, which
//! adds a harmonic error that the Neumann solve does not have (a wrapped ramp reaching
//! ±11 rad at the faces of a 64³ array comes back at r = 0.76, against r > 0.99). That is why
//! this is offered next to [`laplacian_unwrap`](super::laplacian_unwrap) rather than
//! replacing it: use this one to reproduce STI Suite-based results, the Neumann solve when the
//! total field itself is wanted.
//!
//! # Algorithm (recovered by black-box probing of STI Suite 3.0's `MRPhaseUnwrap.p`)
//!
//! 1. Zero-pad the phase by `pad` voxels on both sides of each axis. If a padded dimension is
//!    odd, add one more zero plane at its end (STI does this for the third axis and fails on
//!    odd in-plane sizes; here every axis is treated the same way).
//! 2. `K = |k|²`, `k_d = m / (M_d · h_d)` for the FFT frequency index `m ∈ [−M_d/2, M_d/2)`
//!    (cycles/mm; no 4π² factor), `M_d` the padded size and `h_d` the voxel size.
//! 3. `L = cos φ · F⁻¹[K·F(sin φ)] − sin φ · F⁻¹[K·F(cos φ)]` — that is `−∇²φ / 4π²`.
//! 4. `φ_u = F⁻¹[F(L) / K]`, with the DC term set to zero.
//! 5. Crop back to the input size.
//!
//! Nothing is masked and nothing is rescaled. Callers that want STI's behaviour on brain data
//! zero the phase outside the brain first (`mask · φ`), as the multi-echo helper does when
//! given a mask.
//!
//! Matches `MRPhaseUnwrap.p` to max |Δ| < 1e-13 rad on synthetic volumes at several voxel sizes
//! and paddings, and on a 256×288×48 UK Biobank SWI echo (pad 64).
//!
//! # References
//! Schofield, M.A., Zhu, Y. (2003). "Fast phase unwrapping algorithm for interferometric
//! applications." Optics Letters, 28(14):1194-1196. <https://doi.org/10.1364/OL.28.001194>
//!
//! Li, W., Wu, B., Liu, C. (2011). "Quantitative susceptibility mapping of human brain reflects
//! spatial variation in tissue composition." NeuroImage, 55(4):1645-1656.
//! <https://doi.org/10.1016/j.neuroimage.2010.11.088>

use num_complex::Complex64;

use crate::fft::{fftfreq, Fft3dWorkspace};
use crate::Grid;

/// Parameters for [`laplacian_unwrap_sti`].
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct LaplacianStiParams {
    /// Zero-padding added to both sides of each axis, in voxels (STI's `padsize`).
    /// STI's default is `[12, 12, 12]`; UK Biobank calls it with `[64, 64, 64]`.
    pub pad: [usize; 3],
}

impl Default for LaplacianStiParams {
    fn default() -> Self {
        Self { pad: [12, 12, 12] }
    }
}

/// Squared spatial frequency `|k|²` (cycles/mm)² along one axis of the padded grid.
fn k2_axis(n: usize, h: f64) -> Vec<f64> {
    fftfreq(n, h).into_iter().map(|f| f * f).collect()
}

/// Laplacian phase unwrapping, STI Suite 3.0 `MRPhaseUnwrap` formulation.
///
/// See the [module docs](self) for the algorithm and how it differs from
/// [`super::laplacian_unwrap`].
///
/// # Arguments
/// * `phase` - Wrapped phase in radians (nx * ny * nz, x fastest). Values outside the object
///   are used as they are; STI's callers zero them first.
/// * `grid` - Dimensions and voxel sizes (mm)
/// * `params` - Padding
///
/// # Returns
/// Unwrapped phase over the whole volume (not masked).
pub fn laplacian_unwrap_sti(phase: &[f64], grid: &Grid, params: &LaplacianStiParams) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let (vsx, vsy, vsz) = grid.voxel_size;
    assert_eq!(phase.len(), nx * ny * nz, "phase length does not match grid");
    let [px, py, pz] = params.pad;
    let even = |n: usize| n + (n % 2);
    let (mx, my, mz) = (even(nx + 2 * px), even(ny + 2 * py), even(nz + 2 * pz));
    let mxy = mx * my;
    let n_pad = mxy * mz;

    let (kx, ky, kz) = (k2_axis(mx, vsx), k2_axis(my, vsy), k2_axis(mz, vsz));
    let src = |i: usize, j: usize, k: usize| -> Option<usize> {
        let (i, j, k) = (i.checked_sub(px)?, j.checked_sub(py)?, k.checked_sub(pz)?);
        (i < nx && j < ny && k < nz).then(|| i + j * nx + k * nx * ny)
    };
    // Padded phase, as (sin φ, cos φ). Outside the input, φ = 0.
    let sincos = |i: usize, j: usize, k: usize| -> (f64, f64) {
        match src(i, j, k) {
            Some(s) => phase[s].sin_cos(),
            None => (0.0, 1.0),
        }
    };

    let mut ws = Fft3dWorkspace::new(mx, my, mz);
    let mut buf = vec![Complex64::new(0.0, 0.0); n_pad];

    // K is real and even in k, so F⁻¹[K·F(x)] is real for real x: transform sin φ + i·cos φ
    // once and read F⁻¹[K·F(sin φ)] and F⁻¹[K·F(cos φ)] off the real and imaginary parts.
    for k in 0..mz {
        for j in 0..my {
            for i in 0..mx {
                let (s, c) = sincos(i, j, k);
                buf[i + j * mx + k * mxy] = Complex64::new(s, c);
            }
        }
    }
    ws.fft3d(&mut buf);
    apply_k2(&mut buf, &kx, &ky, &kz, false);
    ws.ifft3d(&mut buf);

    // L = cos φ · F⁻¹[K F sin φ] − sin φ · F⁻¹[K F cos φ]  (= −∇²φ / 4π²)
    for k in 0..mz {
        for j in 0..my {
            for i in 0..mx {
                let idx = i + j * mx + k * mxy;
                let (s, c) = sincos(i, j, k);
                let v = buf[idx];
                buf[idx] = Complex64::new(c * v.re - s * v.im, 0.0);
            }
        }
    }

    // Poisson inverse: φ = F⁻¹[F(L) / K], DC → 0.
    ws.fft3d(&mut buf);
    apply_k2(&mut buf, &kx, &ky, &kz, true);
    ws.ifft3d(&mut buf);

    let mut out = vec![0.0; nx * ny * nz];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                out[i + j * nx + k * nx * ny] = buf[(i + px) + (j + py) * mx + (k + pz) * mxy].re;
            }
        }
    }
    out
}

/// Multiply (or, with `invert`, divide) a spectrum by `K = kx² + ky² + kz²`; DC → 0 on divide.
fn apply_k2(buf: &mut [Complex64], kx: &[f64], ky: &[f64], kz: &[f64], invert: bool) {
    let (mx, my) = (kx.len(), ky.len());
    let mxy = mx * my;
    let plane = |(k, slab): (usize, &mut [Complex64])| {
        for j in 0..my {
            let kyz = ky[j] + kz[k];
            let row = &mut slab[j * mx..(j + 1) * mx];
            for (v, &kxi) in row.iter_mut().zip(kx) {
                let kk = kxi + kyz;
                if invert {
                    *v = if kk == 0.0 { Complex64::new(0.0, 0.0) } else { *v / kk };
                } else {
                    *v *= kk;
                }
            }
        }
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        buf.par_chunks_mut(mxy).enumerate().for_each(plane);
    }
    #[cfg(not(feature = "parallel"))]
    buf.chunks_mut(mxy).enumerate().for_each(plane);
}

/// How echoes are weighted when averaging unwrapped phase in [`laplacian_unwrap_sti_multi_echo`].
///
/// Weights are global (one per echo). Echo times and `t2star` must be in the same unit.
#[derive(Clone, Debug, PartialEq)]
pub enum EchoWeighting {
    /// `w = TE · exp(−TE / T2*)`: the phase-SNR-optimal weight for a single-exponential decay
    /// with the given T2\*. UK Biobank uses T2\* = 40 ms.
    T2Star { t2star: f64 },
    /// `w = TE`.
    EchoTime,
    /// `w = 1`.
    Uniform,
    /// Explicit per-echo weights.
    Custom(Vec<f64>),
}

impl EchoWeighting {
    /// UK Biobank's weighting, `TE · exp(−TE / 40 ms)`, with echo times in seconds.
    pub fn t2star_40ms() -> Self {
        EchoWeighting::T2Star { t2star: 0.040 }
    }

    /// Per-echo weights for the given echo times.
    pub fn weights(&self, tes: &[f64]) -> Vec<f64> {
        match self {
            EchoWeighting::T2Star { t2star } => tes.iter().map(|&te| te * (-te / t2star).exp()).collect(),
            EchoWeighting::EchoTime => tes.to_vec(),
            EchoWeighting::Uniform => vec![1.0; tes.len()],
            EchoWeighting::Custom(w) => {
                assert_eq!(w.len(), tes.len(), "one custom weight per echo");
                w.clone()
            }
        }
    }
}

/// Result of [`laplacian_unwrap_sti_multi_echo`].
#[derive(Clone, Debug)]
pub struct EchoAverage {
    /// `Σ wᵢ φᵢ / Σ wᵢ`: weighted mean of the unwrapped echo phases (radians), whole volume.
    pub phase: Vec<f64>,
    /// `Σ wᵢ TEᵢ / Σ wᵢ`: the echo time that `phase` corresponds to (unit of the input TEs).
    pub te_eff: f64,
    /// The per-echo weights used.
    pub weights: Vec<f64>,
}

impl EchoAverage {
    /// Field in rad per unit of TE (rad/s for TEs in seconds): `phase / te_eff`.
    pub fn field_rad(&self) -> Vec<f64> {
        self.phase.iter().map(|&p| p / self.te_eff).collect()
    }

    /// Field in Hz, for TEs in seconds: `phase / (2π · te_eff)`.
    pub fn field_hz(&self) -> Vec<f64> {
        let s = 1.0 / (std::f64::consts::TAU * self.te_eff);
        self.phase.iter().map(|&p| p * s).collect()
    }
}

/// Multi-echo field map the way STI Suite-based pipelines (e.g. UK Biobank) build it:
/// each echo unwrapped on its own with [`laplacian_unwrap_sti`], then a weighted mean of the
/// unwrapped phases with a matching weighted echo time.
///
/// `phase_avg = Σ wᵢ φᵢ / Σ wᵢ`, `te_eff = Σ wᵢ TEᵢ / Σ wᵢ`, field = `phase_avg / te_eff`
/// (equivalently, a mean of `φᵢ / TEᵢ` weighted by `wᵢ·TEᵢ`). With
/// [`EchoWeighting::t2star_40ms`] and `pad = [64, 64, 64]` this is UK Biobank's field map.
///
/// The phases should be free of a per-echo phase offset (e.g. coil-combined with MCPC-3D-S);
/// any offset left in passes through scaled by `1 / te_eff`.
///
/// # Arguments
/// * `phases` - Wrapped phase per echo (radians)
/// * `tes` - Echo times (any unit; the field is per that unit)
/// * `mask` - If given, each echo is multiplied by it **before** unwrapping (as UK Biobank
///   does: `MRPhaseUnwrap(mask .* phase)`); the output is still not masked
/// * `grid`, `params` - As for [`laplacian_unwrap_sti`]
/// * `weighting` - Echo weights
pub fn laplacian_unwrap_sti_multi_echo<P: AsRef<[f64]>>(
    phases: &[P],
    tes: &[f64],
    mask: Option<&[u8]>,
    grid: &Grid,
    params: &LaplacianStiParams,
    weighting: &EchoWeighting,
) -> EchoAverage {
    assert!(!phases.is_empty(), "no echoes");
    assert_eq!(phases.len(), tes.len(), "one echo time per echo");
    let weights = weighting.weights(tes);
    let wsum: f64 = weights.iter().sum();
    assert!(wsum.abs() > 0.0 && wsum.is_finite(), "echo weights sum to {wsum}");
    let te_eff = weights.iter().zip(tes).map(|(w, t)| w * t).sum::<f64>() / wsum;

    let n = grid.n_total();
    let mut phase = vec![0.0; n];
    for (p, &w) in phases.iter().zip(&weights) {
        let p = p.as_ref();
        assert_eq!(p.len(), n, "phase length does not match grid");
        let u = match mask {
            Some(m) => {
                assert_eq!(m.len(), n, "mask length does not match grid");
                let masked: Vec<f64> = p.iter().zip(m).map(|(&v, &b)| if b != 0 { v } else { 0.0 }).collect();
                laplacian_unwrap_sti(&masked, grid, params)
            }
            None => laplacian_unwrap_sti(p, grid, params),
        };
        for (acc, v) in phase.iter_mut().zip(u) {
            *acc += w * v;
        }
    }
    for v in phase.iter_mut() {
        *v /= wsum;
    }
    EchoAverage { phase, te_eff, weights }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unwrap::laplacian::wrap;
    use std::f64::consts::PI;

    /// The 8×8×6 test volume used for the STI golden values (0-based indices here,
    /// 1-based `i+1` etc. in the MATLAB that produced them).
    fn golden_input() -> (Vec<f64>, Grid) {
        let (nx, ny, nz) = (8, 8, 6);
        let mut p = vec![0.0; nx * ny * nz];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let (a, b, c) = ((i + 1) as f64, (j + 1) as f64, (k + 1) as f64);
                    let t = 2.1 * (0.7 * a).sin() + 1.3 * (0.4 * b * c).cos() + 0.9 * a - 0.6 * c + 0.05 * a * b;
                    p[i + j * nx + k * nx * ny] = t.sin().atan2(t.cos());
                }
            }
        }
        (p, Grid::new(nx, ny, nz, 1.0, 1.0, 2.0))
    }

    #[test]
    fn matches_sti_golden_values() {
        // MRPhaseUnwrap(phi, 'voxelsize', [1 1 2], 'padsize', [2 3 1]) in STI Suite 3.0
        // (R2023b, the values in STI_GOLDEN).
        let (p, grid) = golden_input();
        let u = laplacian_unwrap_sti(&p, &grid, &LaplacianStiParams { pad: [2, 3, 1] });
        let sum: f64 = u.iter().sum();
        let sumsq: f64 = u.iter().map(|v| v * v).sum();
        assert!((sum - STI_GOLDEN.0).abs() < 1e-9, "sum {sum} vs STI {}", STI_GOLDEN.0);
        assert!((sumsq - STI_GOLDEN.1).abs() < 1e-9, "sum sq {sumsq} vs STI {}", STI_GOLDEN.1);
        for &(idx, want) in STI_GOLDEN.2 {
            assert!((u[idx] - want).abs() < 1e-12, "voxel {idx}: {} vs STI {want}", u[idx]);
        }
    }

    /// (sum, sum of squares, [(0-based linear index, value)]) of STI's output on golden_input().
    #[allow(clippy::type_complexity, clippy::excessive_precision)]
    const STI_GOLDEN: (f64, f64, &[(usize, f64)]) = (
        -47.110605048249184,
        278.05516319999214,
        &[
            (0, 0.035944170373255008), (1, 0.54603271528608766), (8, -0.12289609617968449),
            (64, 0.17121148831819896), (99, -0.58598953742943161), (199, 0.35334342671042662),
            (332, -0.18693470845945748), (383, 1.1196905106581743),
        ],
    );

    #[test]
    fn invariant_to_two_pi_jumps() {
        // Only sin φ and cos φ enter, so adding 2π anywhere changes nothing.
        let (p, grid) = golden_input();
        let jumped: Vec<f64> = p.iter().enumerate()
            .map(|(i, &v)| v + 2.0 * PI * ((i * 7919) % 5) as f64 - 4.0 * PI)
            .collect();
        let params = LaplacianStiParams::default();
        let a = laplacian_unwrap_sti(&p, &grid, &params);
        let b = laplacian_unwrap_sti(&jumped, &grid, &params);
        let d = a.iter().zip(&b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
        assert!(d < 1e-12, "max |diff| {d}");
    }

    /// Smooth blob, wrapped many times, inside a box of zeros.
    fn wrapped_blob(n: (usize, usize, usize), vs: (f64, f64, f64)) -> (Vec<f64>, Vec<f64>, Grid) {
        let grid = Grid::new(n.0, n.1, n.2, vs.0, vs.1, vs.2);
        let c = (n.0 as f64 / 2.0 * vs.0, n.1 as f64 / 2.0 * vs.1, n.2 as f64 / 2.0 * vs.2);
        let mut truth = vec![0.0; grid.n_total()];
        for k in 0..n.2 {
            for j in 0..n.1 {
                for i in 0..n.0 {
                    let (x, y, z) = (i as f64 * vs.0 - c.0, j as f64 * vs.1 - c.1, k as f64 * vs.2 - c.2);
                    truth[i + j * n.0 + k * n.0 * n.1] = 20.0 * (-(x * x + y * y + z * z) / (2.0 * 9.0f64.powi(2))).exp();
                }
            }
        }
        let wrapped = truth.iter().map(|&v| wrap(v)).collect();
        (wrapped, truth, grid)
    }

    fn max_dev_after_mean(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let off = a.iter().zip(b).map(|(x, y)| x - y).sum::<f64>() / n;
        a.iter().zip(b).fold(0.0f64, |m, (x, y)| m.max((x - y - off).abs()))
    }

    #[test]
    fn unwraps_a_smooth_field() {
        // Peak 20 rad (≈ 3 wraps), decays to ~0 at the edge: the periodic, zero-padded solve
        // should return it up to a constant. Steepest change ≈ 1.35 rad/mm, so at most ~1.6 rad
        // per voxel here; the sin/cos identity degrades well before π per voxel (at 2 mm,
        // 2.7 rad/voxel, the error is ~0.2 rad).
        for vs in [(1.0, 1.0, 1.0), (0.8, 0.8, 1.2), (1.2, 0.9, 1.0)] {
            let (wrapped, truth, grid) = wrapped_blob((64, 64, 48), vs);
            let u = laplacian_unwrap_sti(&wrapped, &grid, &LaplacianStiParams::default());
            let d = max_dev_after_mean(&u, &truth);
            assert!(d < 0.05, "voxel size {vs:?}: max deviation {d} rad");
        }
    }

    #[test]
    fn odd_dimensions_keep_the_input_size() {
        let (wrapped, truth, grid) = wrapped_blob((61, 63, 47), (1.0, 1.0, 1.0));
        for pad in [[0, 0, 0], [3, 4, 5]] {
            let u = laplacian_unwrap_sti(&wrapped, &grid, &LaplacianStiParams { pad });
            assert_eq!(u.len(), grid.n_total());
            assert!(max_dev_after_mean(&u, &truth) < 0.05);
        }
    }

    #[test]
    fn dc_of_the_padded_box_is_zero() {
        // The Poisson solve zeroes the mean over the padded box, not over the input.
        let (wrapped, _, grid) = wrapped_blob((32, 32, 24), (1.0, 1.0, 1.0));
        let u = laplacian_unwrap_sti(&wrapped, &grid, &LaplacianStiParams { pad: [0, 0, 0] });
        let mean = u.iter().sum::<f64>() / u.len() as f64;
        assert!(mean.abs() < 1e-12, "mean {mean}");
    }

    #[test]
    fn zero_padding_costs_total_field_fidelity() {
        // A harmonic ramp that is far from zero at the array faces: the zero padding puts a
        // step there, and the periodic solve turns it into a harmonic error inside. The Neumann
        // solve has no such step. This is the documented reason the STI formulation is not the
        // default; if it starts passing the other way, update the module docs.
        let n = 48;
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let c = n as f64 / 2.0;
        let mut truth = vec![0.0; n * n * n];
        for k in 0..n { for j in 0..n { for i in 0..n {
            let (x, y, z) = (i as f64 - c, j as f64 - c, k as f64 - c);
            truth[i + j * n + k * n * n] = 0.35 * x + 0.2 * y + 0.15 * z
                + 12.0 * (-(x * x + y * y + z * z) / 128.0).exp();
        }}}
        let wrapped: Vec<f64> = truth.iter().map(|&v| wrap(v)).collect();
        let mask = vec![1u8; truth.len()];
        let r = |u: &[f64]| {
            let m = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            let (mu, mt) = (m(u), m(&truth));
            let (mut cov, mut vu, mut vt) = (0.0, 0.0, 0.0);
            for (a, b) in u.iter().zip(&truth) { cov += (a - mu) * (b - mt); vu += (a - mu).powi(2); vt += (b - mt).powi(2); }
            cov / (vu * vt).sqrt()
        };
        let sti = laplacian_unwrap_sti(&wrapped, &grid, &LaplacianStiParams::default());
        let neumann = crate::unwrap::laplacian_unwrap(&wrapped, &mask, &grid);
        assert!(r(&neumann) > 0.99, "Neumann r = {}", r(&neumann));
        assert!(r(&sti) < 0.9, "STI r = {}", r(&sti));
    }

    #[test]
    fn t2star_weights_and_effective_te() {
        let tes = [9.42e-3, 19.7e-3];
        let w = EchoWeighting::t2star_40ms().weights(&tes);
        assert!((w[0] - 9.42e-3 * (-9.42f64 / 40.0).exp()).abs() < 1e-15);
        assert!((w[1] - 19.7e-3 * (-19.7f64 / 40.0).exp()).abs() < 1e-15);
        // te_eff for UK Biobank's echoes: 15.772350 ms (from the MATLAB run)
        let te_eff = (w[0] * tes[0] + w[1] * tes[1]) / (w[0] + w[1]);
        assert!((te_eff - 15.772350e-3).abs() < 1e-9, "{te_eff}");
        assert_eq!(EchoWeighting::Uniform.weights(&tes), vec![1.0, 1.0]);
        assert_eq!(EchoWeighting::EchoTime.weights(&tes), tes.to_vec());
    }

    #[test]
    fn multi_echo_average_recovers_the_frequency() {
        // Phase linear in TE (no offset): every weighting gives the same field.
        let (_, truth, grid) = wrapped_blob((64, 64, 48), (1.0, 1.0, 1.0));
        let tes = [0.004, 0.009, 0.014];
        // truth is the phase at the last echo, TE = 14 ms
        let phases: Vec<Vec<f64>> = tes.iter()
            .map(|&te| truth.iter().map(|&v| wrap(v * te / 0.014)).collect())
            .collect();
        let params = LaplacianStiParams::default();
        let reference: Vec<f64> = truth.iter().map(|&v| v / 0.014).collect();
        for wt in [EchoWeighting::t2star_40ms(), EchoWeighting::Uniform, EchoWeighting::EchoTime,
                   EchoWeighting::Custom(vec![1.0, 2.0, 3.0])] {
            let avg = laplacian_unwrap_sti_multi_echo(&phases, &tes, None, &grid, &params, &wt);
            let f = avg.field_rad();
            let d = max_dev_after_mean(&f, &reference) * 0.014;
            assert!(d < 0.05, "{wt:?}: max deviation {d} rad at 14 ms");
            let hz = avg.field_hz();
            assert!((hz[100] * std::f64::consts::TAU - f[100]).abs() < 1e-9);
        }
    }

    #[test]
    fn multi_echo_mask_is_applied_to_the_input() {
        let (wrapped, _, grid) = wrapped_blob((32, 32, 24), (1.0, 1.0, 1.0));
        let mask: Vec<u8> = (0..grid.n_total()).map(|i| (i % 3 != 0) as u8).collect();
        let masked: Vec<f64> = wrapped.iter().zip(&mask).map(|(&v, &m)| v * m as f64).collect();
        let params = LaplacianStiParams { pad: [4, 4, 4] };
        let a = laplacian_unwrap_sti_multi_echo(&[&wrapped], &[1.0], Some(&mask), &grid, &params, &EchoWeighting::Uniform);
        let b = laplacian_unwrap_sti(&masked, &grid, &params);
        assert_eq!(a.te_eff, 1.0);
        let d = a.phase.iter().zip(&b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
        assert!(d < 1e-14, "{d}");
    }
}

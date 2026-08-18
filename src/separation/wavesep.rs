//! WaveSep: wavelet-based susceptibility source separation.
//!
//! WaveSep splits net susceptibility (χ_total / QSM) into paramagnetic (χ+, iron)
//! and diamagnetic (χ−, myelin·calcium) sources using an R2' map to break the
//! para/dia degeneracy. It solves two voxel-wise data-fidelity terms under a
//! wavelet-domain L1 (soft-thresholding) sparsity prior by proximal gradient
//! (ISTA):
//!
//! ```text
//!   χ+ + χ−  ≈ χ_total          (net susceptibility)
//!   χ+ − χ−  ≈ R2' / Dr         (static-dephasing R2', single relaxivity kernel)
//! ```
//!
//! with the sign convention χ+ ≥ 0, χ− ≤ 0. `Dr` is the static-dephasing
//! relaxivity (Hz/ppm); the qsm-forward phantom's single kernel is 137. WaveSep's
//! QSM path uses **no** B0 direction (unlike an STI path), so single-orientation
//! data needs no reorientation.
//!
//! Each ISTA iteration is: a gradient step on the two quadratic fidelities, a
//! wavelet-L1 proximal step (forward db4 periodic transform → soft-threshold all
//! coefficients by `alpha·lambda` → inverse), then a sign projection
//! (χ+ = max(χ+,0), χ− = min(χ−,0)) restricted to the mask. It stops when the
//! relative change falls below `tol`.
//!
//! The volume is zero-padded so every axis is a multiple of `2^L` (L = the
//! periodic max decomposition level, [`dwt_max_level`]), matching PyWavelets'
//! `periodization` round-trip, then cropped back.
//!
//! Reference:
//! Fang, Z., Shin, H.-G., van Zijl, P., Li, X., Sulam, J. (2023). "WaveSep: A
//! flexible wavelet-based approach for source separation in susceptibility
//! imaging." Machine Learning in Clinical Neuroimaging (MLCN), MICCAI 2023,
//! Springer LNCS. https://doi.org/10.1007/978-3-031-44858-4_6
//!
//! Reference implementation: https://github.com/ZhenghanFang/WaveSep

use crate::utils::wavelet::{dwt_max_level, WaveletPlan};
use crate::Grid;

/// Parameters for [`wavesep`].
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct WaveSepParams {
    /// Paramagnetic static-dephasing relaxivity Dr+ in Hz/ppm (phantom kernel: 137).
    pub dr_pos: f64,
    /// Diamagnetic static-dephasing relaxivity Dr− in Hz/ppm (WaveSep assumes = Dr+).
    pub dr_neg: f64,
    /// Proximal-gradient step / data-consistency weight (repo default 0.2).
    pub alpha: f64,
    /// Wavelet-domain L1 (soft-thresholding) sparsity weight (repo default 0.02).
    pub lambda: f64,
    /// Daubechies order for the sparsity transform (repo default 4 = db4).
    pub wavelet_order: usize,
    /// Maximum proximal-gradient iterations (repo default 100, early-stops).
    pub max_iter: usize,
    /// Relative-change early-stop tolerance (repo default 1e-3).
    pub tol: f64,
}

impl Default for WaveSepParams {
    fn default() -> Self {
        Self {
            dr_pos: 137.0,
            dr_neg: 137.0,
            alpha: 0.2,
            lambda: 0.02,
            wavelet_order: 4,
            max_iter: 100,
            tol: 1e-3,
        }
    }
}

/// WaveSep source separation from a QSM and an R2' map.
///
/// # Arguments
/// * `chi_total` — Conventional QSM χ_total in **ppm** (`nx·ny·nz`, column-major).
/// * `r2prime` — R2' map in **Hz** (`nx·ny·nz`).
/// * `mask` — Binary brain mask (`nx·ny·nz`, 1 = inside).
/// * `grid` — Volume dimensions and voxel sizes.
/// * `params` — See [`WaveSepParams`].
/// * `progress` — Progress callback `(iteration, max_iterations)`.
///
/// # Returns
/// `(chi_pos, chi_neg, chi_total)` in ppm, restricted to `mask` — matching the
/// [`chi_sep_ilsqr`](super::chi_sep_ilsqr)/[`chi_sep_medi`](super::chi_sep_medi)
/// convention: `chi_pos` ≥ 0 (paramagnetic), `chi_neg` ≤ 0 (diamagnetic, signed),
/// and `chi_total = chi_pos + chi_neg`.
pub fn wavesep(
    chi_total: &[f64],
    r2prime: &[f64],
    mask: &[u8],
    grid: &Grid,
    params: &WaveSepParams,
    mut progress: impl FnMut(usize, usize),
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (nx, ny, nz) = grid.dims;
    let n = nx * ny * nz;
    assert_eq!(chi_total.len(), n, "chi_total length must match grid");
    assert_eq!(r2prime.len(), n, "r2prime length must match grid");
    assert_eq!(mask.len(), n, "mask length must match grid");

    // --- padding so each axis is a multiple of 2^L (periodic wavelet round-trip) ---
    let dec_len = 2 * params.wavelet_order;
    let (pdims, level) = pad_spec((nx, ny, nz), dec_len);
    let (pnx, pny, pnz) = pdims;
    let pn = pnx * pny * pnz;

    // Masked, padded inputs (zeros appended at the high end of each axis).
    let maskf: Vec<f64> = mask.iter().map(|&m| m as f64).collect();
    let mask_p = pad3d(&maskf, (nx, ny, nz), pdims);
    let qsm_p = {
        let mut q = pad3d(chi_total, (nx, ny, nz), pdims);
        for i in 0..pn {
            q[i] *= mask_p[i];
        }
        q
    };
    let r2p_p = {
        let mut r = pad3d(r2prime, (nx, ny, nz), pdims);
        for i in 0..pn {
            r[i] *= mask_p[i];
        }
        r
    };

    let plan = WaveletPlan::new(params.wavelet_order, pdims, level);
    let th = params.alpha * params.lambda;
    let dr_ratio = params.dr_neg / params.dr_pos;

    // ISTA state: xp (χ+) and xn (χ−); `prev` holds the last projected iterate
    // so the early stop matches WaveSep's ‖x − x_old‖ / ‖x‖.
    let mut xp = vec![0.0_f64; pn];
    let mut xn = vec![0.0_f64; pn];
    let mut prev = vec![0.0_f64; 2 * pn];

    for it in 0..params.max_iter {
        progress(it + 1, params.max_iter);
        // --- gradient step on the two quadratic fidelities ---
        // QSM fidelity: gq = χ+ + χ− − χ_total (same gradient for both components).
        // R2' fidelity: r = χ+ − (Dr−/Dr+)·χ− − R2'/Dr+ ; grads [r, −(Dr−/Dr+)·r].
        for i in 0..pn {
            let gq = xp[i] + xn[i] - qsm_p[i];
            let r = xp[i] - dr_ratio * xn[i] - r2p_p[i] / params.dr_pos;
            xp[i] -= params.alpha * (gq + r);
            xn[i] -= params.alpha * (gq - dr_ratio * r);
        }

        // --- wavelet-L1 proximal step (soft-threshold all coefficients) ---
        prox_wavelet_l1(&mut xp, &plan, th);
        prox_wavelet_l1(&mut xn, &plan, th);

        // --- sign projection + mask; accumulate ‖Δx‖ and ‖x‖ for the early stop ---
        let mut diff = 0.0_f64;
        let mut norm = 0.0_f64;
        for i in 0..pn {
            let m = mask_p[i];
            xp[i] = if xp[i] < 0.0 { 0.0 } else { xp[i] } * m;
            xn[i] = if xn[i] > 0.0 { 0.0 } else { xn[i] } * m;
            let dp = xp[i] - prev[i];
            let dn = xn[i] - prev[pn + i];
            diff += dp * dp + dn * dn;
            norm += xp[i] * xp[i] + xn[i] * xn[i];
            prev[i] = xp[i];
            prev[pn + i] = xn[i];
        }
        if norm > 0.0 && (diff.sqrt() / norm.sqrt()) < params.tol {
            break;
        }
    }

    // Crop back and emit χ+ (≥ 0), χ− (≤ 0, signed) and χ_total = χ+ + χ−.
    let xp_c = unpad3d(&xp, pdims, (nx, ny, nz));
    let xn_c = unpad3d(&xn, pdims, (nx, ny, nz));
    let mut chi_pos = vec![0.0_f64; n];
    let mut chi_neg = vec![0.0_f64; n];
    let mut chi_out = vec![0.0_f64; n];
    for i in 0..n {
        if mask[i] != 0 {
            chi_pos[i] = xp_c[i];
            chi_neg[i] = xn_c[i];
            chi_out[i] = xp_c[i] + xn_c[i];
        }
    }
    (chi_pos, chi_neg, chi_out)
}

/// Wavelet-L1 proximal operator for an orthonormal transform: soft-threshold all
/// coefficients (approximation + details) by `th`, in place.
fn prox_wavelet_l1(x: &mut [f64], plan: &WaveletPlan, th: f64) {
    let mut coef = plan.forward(x);
    for c in coef.iter_mut() {
        *c = soft_threshold(*c, th);
    }
    let rec = plan.inverse(&coef);
    x.copy_from_slice(&rec);
}

#[inline]
fn soft_threshold(z: f64, th: f64) -> f64 {
    if z > th {
        z - th
    } else if z < -th {
        z + th
    } else {
        0.0
    }
}

/// Padded dimensions and decomposition level, matching WaveSep's `pad_spec`:
/// grow each axis to a common multiple `P` (starting at 16) until every padded
/// dim is divisible by `2^L`, where `L = dwt_max_level(min_padded, dec_len)`.
fn pad_spec(dims: (usize, usize, usize), dec_len: usize) -> ((usize, usize, usize), usize) {
    let ds = [dims.0, dims.1, dims.2];
    let mut p = 16usize;
    loop {
        let padded = [
            ds[0].div_ceil(p) * p,
            ds[1].div_ceil(p) * p,
            ds[2].div_ceil(p) * p,
        ];
        let min_padded = *padded.iter().min().unwrap();
        let level = dwt_max_level(min_padded, dec_len);
        let m = 1usize << level;
        if padded.iter().all(|&d| d % m == 0) {
            return ((padded[0], padded[1], padded[2]), level);
        }
        p *= 2;
    }
}

/// Zero-pad a column-major 3D array from `from` to `to` (extra voxels appended at
/// the high end of each axis), matching `numpy.pad(a, [(0, p-d), ...])`.
fn pad3d(a: &[f64], from: (usize, usize, usize), to: (usize, usize, usize)) -> Vec<f64> {
    let (fx, fy, fz) = from;
    let (tx, ty, tz) = to;
    let mut out = vec![0.0_f64; tx * ty * tz];
    for k in 0..fz {
        for j in 0..fy {
            let src = (k * fy + j) * fx;
            let dst = (k * ty + j) * tx;
            out[dst..dst + fx].copy_from_slice(&a[src..src + fx]);
        }
    }
    out
}

/// Crop a column-major 3D array from `from` back to `to` (inverse of [`pad3d`]).
fn unpad3d(a: &[f64], from: (usize, usize, usize), to: (usize, usize, usize)) -> Vec<f64> {
    let (fx, fy, _fz) = from;
    let (tx, ty, tz) = to;
    let mut out = vec![0.0_f64; tx * ty * tz];
    for k in 0..tz {
        for j in 0..ty {
            let src = (k * fy + j) * fx;
            let dst = (k * ty + j) * tx;
            out[dst..dst + tx].copy_from_slice(&a[src..src + tx]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(nx: usize, ny: usize, nz: usize) -> Grid {
        Grid::new(nx, ny, nz, 1.0, 1.0, 1.0)
    }

    #[test]
    fn pad_spec_divisible_and_covers() {
        for dims in [(64, 64, 40), (100, 100, 60), (48, 56, 72)] {
            let (pd, level) = pad_spec(dims, 8);
            let m = 1usize << level;
            assert!(pd.0 % m == 0 && pd.1 % m == 0 && pd.2 % m == 0, "divisible {dims:?}");
            assert!(pd.0 >= dims.0 && pd.1 >= dims.1 && pd.2 >= dims.2, "covers {dims:?}");
        }
    }

    #[test]
    fn pad_unpad_roundtrip() {
        let from = (5, 6, 7);
        let to = (8, 8, 8);
        let n = 5 * 6 * 7;
        let a: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let p = pad3d(&a, from, to);
        let back = unpad3d(&p, to, from);
        assert_eq!(a, back);
    }

    /// On a phantom obeying the WaveSep model exactly (single kernel Dr, no
    /// regularisation needed), the solver should recover the sources well.
    #[test]
    fn recovers_model_sources() {
        let (nx, ny, nz) = (16, 16, 16);
        let g = grid(nx, ny, nz);
        let n = nx * ny * nz;
        let dr = 137.0;

        // Smooth, sign-correct source fields inside a central mask.
        let mut chi_pos = vec![0.0_f64; n];
        let mut chi_neg = vec![0.0_f64; n]; // ≤ 0
        let mut mask = vec![0u8; n];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = i + j * nx + k * nx * ny;
                    let inside = (3..13).contains(&i) && (3..13).contains(&j) && (3..13).contains(&k);
                    if inside {
                        mask[idx] = 1;
                        let fx = i as f64 / nx as f64;
                        let fy = j as f64 / ny as f64;
                        chi_pos[idx] = 0.10 * (1.0 + (fx * 6.0).sin()).max(0.0);
                        chi_neg[idx] = -0.08 * (1.0 + (fy * 5.0).cos()).max(0.0);
                    }
                }
            }
        }
        // Forward model: χ_total = χ+ + χ−, R2' = Dr·(χ+ + |χ−|).
        let chi_total: Vec<f64> = (0..n).map(|i| chi_pos[i] + chi_neg[i]).collect();
        let r2p: Vec<f64> = (0..n).map(|i| dr * (chi_pos[i] - chi_neg[i])).collect();

        let params = WaveSepParams {
            dr_pos: dr,
            dr_neg: dr,
            alpha: 0.2,
            lambda: 0.002, // light regularisation for a clean phantom
            wavelet_order: 4,
            max_iter: 300,
            tol: 1e-4,
        };
        let (para, neg, total) = wavesep(&chi_total, &r2p, &mask, &g, &params, |_, _| {});

        // Correlate recovered vs. truth inside the mask.
        let corr = |a: &[f64], b: &[f64]| {
            let idx: Vec<usize> = (0..n).filter(|&i| mask[i] == 1).collect();
            let m = idx.len() as f64;
            let ma = idx.iter().map(|&i| a[i]).sum::<f64>() / m;
            let mb = idx.iter().map(|&i| b[i]).sum::<f64>() / m;
            let mut sab = 0.0;
            let mut saa = 0.0;
            let mut sbb = 0.0;
            for &i in &idx {
                let da = a[i] - ma;
                let db = b[i] - mb;
                sab += da * db;
                saa += da * da;
                sbb += db * db;
            }
            sab / (saa.sqrt() * sbb.sqrt() + 1e-20)
        };
        // χ− is returned signed (≤ 0); correlate against the signed truth.
        let cp = corr(&para, &chi_pos);
        let cn = corr(&neg, &chi_neg);
        assert!(cp > 0.9, "χ+ correlation too low: {cp:.3}");
        assert!(cn > 0.9, "χ− correlation too low: {cn:.3}");
        // χ_total ≈ χ+ + χ− and χ− ≤ 0 inside the mask.
        for i in 0..n {
            if mask[i] == 1 {
                assert!(neg[i] <= 1e-9, "χ− must be ≤ 0");
                assert!((total[i] - (para[i] + neg[i])).abs() < 1e-9, "χ_total = χ+ + χ−");
            }
        }
    }
}

//! Marchenko–Pastur PCA (MP-PCA) denoising for multi-volume data.
//!
//! Denoises a stack of co-registered volumes (e.g. multi-echo magnitude) by
//! exploiting redundancy across the volume dimension: within a small spatial
//! patch the noise-free signal is low-rank, so PCA components whose eigenvalues
//! fall inside the Marchenko–Pastur (random-matrix) noise bulk are discarded.
//! Unlike Gaussian smoothing this preserves spatial edges — it removes noise
//! along the volume dimension, not across space.
//!
//! This is a port of DIPY's `mppca` (`localpca.genpca`, eigenvalue path), which implements
//! Veraart et al. (2016), "Denoising of diffusion MRI using random matrix theory", NeuroImage
//! 142:394-406, with Manjón et al. (2013)'s overlapping-patch aggregation: every patch's estimate
//! is given to all its voxels, weighted by `1 / (1 + n − ncomps)`, and each voxel is the weighted
//! mean of the estimates covering it. Voxels outside the mask are 0, as in DIPY.
//!
//! Applying this to multi-echo magnitude before R2*/R2 fitting reduces the
//! variance of the fitted relaxation rate without blurring structure.
//!
//! **Agreement with DIPY 1.12.1** (verified QSM-CI `py-originals` image, QSMxT/QSM-CI#302;
//! patch radius 2; measured 2026-10-31, astewartau/QSM.rs#327): NRMSE 2.6e-6 % on a 64×64×32
//! crop of a travelling head's five echoes (`prisma-bridge-run1`; DIPY changes the input by
//! 2.4 %), and 2.6e-6 % (the fixture's float32 rounding) on the six-echo 14×14×10 phantom of
//! `tests/mppca_dipy_parity.rs`. What is left is summation order and the eigensolver (cyclic
//! Jacobi here, LAPACK `syevr` in SciPy). Before this, each voxel took only its own patch's
//! estimate (a race-free shortcut), which was 0.97 % from DIPY on the full travelling head and
//! 1.3 % on the phantom.
//!
//! **Which implementation.** DIPY is the reference QSM-CI runs. Veraart's own MATLAB,
//! `MPdenoising.m` (NYU-DiffusionMRI/mppca_denoise), is not the same algorithm in its details:
//! - it does not centre the patch by default;
//! - its noise criterion compares two σ² estimates per rank, with a `(max(M, N) − r)/N` scaling;
//! - it assigns the centre voxel (`'full'` sampling) or whole blocks (`'fast'`, its default for
//!   denoising) rather than averaging overlapping patches;
//! - its licence is non-commercial.
//!
//! Whether to follow it instead is an open question (astewartau/QSM.rs#327).

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Cyclic Jacobi eigen-decomposition of a symmetric `n×n` matrix `a`
/// (row-major). Returns `(eigenvalues, eigenvectors)` with eigenvalues in
/// ASCENDING order and eigenvectors stored column-wise (`v[row*n + col]` is
/// component `row` of eigenvector `col`).
pub(crate) fn jacobi_eigh(a_in: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = a_in.to_vec();
    // v = identity (eigenvectors accumulate here)
    let mut v = vec![0.0_f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    // Cyclic Jacobi sweeps. Converges quadratically; ~10 sweeps suffice. Use a
    // RELATIVE off-diagonal threshold (an absolute one may never trip in f64 and
    // would waste all sweeps on every voxel).
    let diag_scale: f64 = (0..n).map(|i| a[i * n + i] * a[i * n + i]).sum::<f64>().max(1e-300);
    for _sweep in 0..20 {
        // Off-diagonal magnitude.
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[p * n + q] * a[p * n + q];
            }
        }
        if off <= 1e-24 * diag_scale {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // Rotate rows/cols p,q of A.
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                // Accumulate rotation into V.
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    // Eigenvalues on the diagonal; sort ascending and reorder eigenvectors.
    let mut eig: Vec<(f64, usize)> = (0..n).map(|i| (a[i * n + i], i)).collect();
    eig.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
    let mut d = vec![0.0_f64; n];
    let mut w = vec![0.0_f64; n * n];
    for (new_col, &(val, old_col)) in eig.iter().enumerate() {
        d[new_col] = val;
        for row in 0..n {
            w[row * n + new_col] = v[row * n + old_col];
        }
    }
    (d, w)
}

/// Marchenko–Pastur noise classifier (Veraart 2016; DIPY `_pca_classifier`).
///
/// `d` are covariance eigenvalues in ascending order; `n_samples` is the number
/// of voxels in the patch. Returns the estimated noise variance.
fn mp_noise_variance(d: &[f64], n_samples: usize) -> f64 {
    // Correct for the rank lost to mean subtraction.
    let start = if d.len() > n_samples.saturating_sub(1) {
        d.len() - (n_samples - 1)
    } else {
        0
    };
    let l = &d[start..];
    let m = l.len();
    if m == 0 {
        return 0.0;
    }
    let mean = |k: usize| -> f64 {
        if k == 0 {
            0.0
        } else {
            l[..k].iter().sum::<f64>() / k as f64
        }
    };
    let mut var = l.iter().sum::<f64>() / m as f64;
    let mut c = m - 1;
    let mut r = l[c] - l[0] - 4.0 * ((c as f64 + 1.0) / n_samples as f64).sqrt() * var;
    while r > 0.0 && c > 0 {
        var = mean(c);
        c -= 1;
        r = l[c] - l[0] - 4.0 * ((c as f64 + 1.0) / n_samples as f64).sqrt() * var;
    }
    var
}

/// Denoise multi-volume data with MP-PCA, as DIPY's `mppca` (`genpca` with `sigma=None`,
/// `pca_method='eig'`).
///
/// For every voxel at least `patch_radius` from the volume border and inside the mask, the
/// `(2r+1)³` patch around it is centred on its per-volume mean, its covariance eigen-decomposed,
/// the noise variance estimated from the Marchenko–Pastur bulk (DIPY's
/// `_pca_classifier`), and the components with eigenvalue below `τ = (1 + √(n/m))² σ²` dropped.
/// Every voxel of the patch gets that patch's estimate, weighted by `1 / (1 + n − ncomps)` with
/// `ncomps` the dropped count (Manjón et al. 2013, eq. 3), and the output is the weighted mean of
/// the estimates of all patches covering a voxel.
///
/// # Arguments
/// * `data` - Interleaved `[voxel0_vol0, voxel0_vol1, ..., voxel1_vol0, ...]`
///   (row-major `(n_voxels, n_vols)`).
/// * `dims` - Volume dimensions `(nx, ny, nz)`.
/// * `n_vols` - Number of volumes (e.g. echoes).
/// * `patch_radius` - Half-width of the cubic patch (radius 2 → 5×5×5).
/// * `mask` - Optional binary mask: patches are centred only on voxels inside it, and voxels
///   outside it are 0 in the output, as in DIPY. `None` is the whole volume.
///
/// # Returns
/// Denoised data in the same layout, with negative values set to 0 as DIPY's `genpca` does
/// (`clip(min=0)`): the input is magnitude, where a negative value has no physical meaning.
/// Voxels inside the mask that no patch covers (possible only at the volume border, next to the
/// mask's edge) keep their input values, where DIPY's `0/0` leaves NaN.
pub fn mppca_denoise(
    data: &[f64],
    dims: (usize, usize, usize),
    n_vols: usize,
    patch_radius: usize,
    mask: Option<&[u8]>,
) -> Vec<f64> {
    let (nx, ny, nz) = dims;
    let n_voxels = nx * ny * nz;
    assert_eq!(data.len(), n_voxels * n_vols, "data length must be n_voxels * n_vols");
    let n = n_vols;
    let r = patch_radius;
    let side = 2 * r + 1;
    let m = side * side * side; // patch voxel count
    let plane = nx * ny;
    let in_mask = |v: usize| mask.is_none_or(|mk| mk[v] != 0);
    let tau_factor = 1.0 + (n as f64 / m as f64).sqrt();

    // Σ θ and Σ θ·X̂ over the patches covering each voxel (DIPY's `theta`, `thetax`).
    let mut theta = vec![0.0_f64; n_voxels];
    let mut thetax = vec![0.0_f64; n_voxels * n];

    // The estimate of the patch centred at (cx, cy, cz), added into a slab of `theta`/`thetax`
    // that starts at plane `z0`.
    let add_patch = |cx: usize, cy: usize, cz: usize, z0: usize, th: &mut [f64], thx: &mut [f64]| {
        // Gather the patch X (m×n) and its voxel indices.
        let mut x = vec![0.0_f64; m * n];
        let mut vox_of = vec![0usize; m];
        let mut rr = 0usize;
        for dz in 0..side {
            let z = cz + dz - r;
            for dy in 0..side {
                let y = cy + dy - r;
                for dx in 0..side {
                    let xx = cx + dx - r;
                    let vox = xx + y * nx + z * plane;
                    x[rr * n..rr * n + n].copy_from_slice(&data[vox * n..vox * n + n]);
                    vox_of[rr] = vox;
                    rr += 1;
                }
            }
        }
        // Column means; centre X.
        let mut mean = vec![0.0_f64; n];
        for row in 0..m {
            for t in 0..n {
                mean[t] += x[row * n + t];
            }
        }
        for t in 0..n {
            mean[t] /= m as f64;
        }
        for row in 0..m {
            for t in 0..n {
                x[row * n + t] -= mean[t];
            }
        }
        // Covariance C = XᵀX / m  (n×n).
        let mut cov = vec![0.0_f64; n * n];
        for row in 0..m {
            for a in 0..n {
                let xa = x[row * n + a];
                if xa == 0.0 {
                    continue;
                }
                for b in a..n {
                    cov[a * n + b] += xa * x[row * n + b];
                }
            }
        }
        for a in 0..n {
            for b in a..n {
                let v = cov[a * n + b] / m as f64;
                cov[a * n + b] = v;
                cov[b * n + a] = v;
            }
        }
        let (d, w) = jacobi_eigh(&cov, n);
        let var = mp_noise_variance(&d, m);
        let tau = tau_factor * tau_factor * var;
        // Dropped (noise) components: eigenvalue below tau.
        let kept: Vec<usize> = (0..n).filter(|&k| d[k] >= tau).collect();
        let ncomps = n - kept.len();
        let this_theta = 1.0 / (1.0 + n as f64 - ncomps as f64);
        // X̂ = X W Wᵀ + M over the kept components, for every row of the patch.
        let mut est = vec![0.0_f64; n];
        for row in 0..m {
            est.copy_from_slice(&mean);
            let xr = &x[row * n..row * n + n];
            for &k in &kept {
                let mut proj = 0.0;
                for a in 0..n {
                    proj += xr[a] * w[a * n + k];
                }
                for t in 0..n {
                    est[t] += proj * w[t * n + k];
                }
            }
            let local = vox_of[row] - z0 * plane;
            th[local] += this_theta;
            for t in 0..n {
                thx[local * n + t] += est[t] * this_theta;
            }
        }
    };

    if nx >= side && ny >= side && nz >= side {
        // Patches centred on planes cz ≡ g (mod 2r+1) cover disjoint slabs of 2r+1 planes, so
        // each group's planes run in parallel; within a plane, centres go y then x, as DIPY's loop.
        for g in 0..side {
            let z_first = g; // slab of the centre plane cz = r + g starts at plane g
            if r + g >= nz - r {
                continue;
            }
            let slab = side * plane;
            let th_slabs = theta[z_first * plane..].chunks_exact_mut(slab);
            let thx_slabs = thetax[z_first * plane * n..].chunks_exact_mut(slab * n);
            let work = |(t, (th, thx)): (usize, (&mut [f64], &mut [f64]))| {
                let z0 = z_first + t * side;
                let cz = z0 + r;
                if cz >= nz - r {
                    return;
                }
                for cy in r..ny - r {
                    for cx in r..nx - r {
                        if in_mask(cx + cy * nx + cz * plane) {
                            add_patch(cx, cy, cz, z0, th, thx);
                        }
                    }
                }
            };
            #[cfg(feature = "parallel")]
            {
                let slabs: Vec<_> = th_slabs.zip(thx_slabs).enumerate().collect();
                slabs.into_par_iter().for_each(work);
            }
            #[cfg(not(feature = "parallel"))]
            {
                th_slabs.zip(thx_slabs).enumerate().for_each(work);
            }
        }
    }

    let mut out = vec![0.0_f64; n_voxels * n];
    for v in 0..n_voxels {
        if !in_mask(v) {
            continue; // 0, as DIPY's `denoised_arr[mask == 0] = 0`
        }
        if theta[v] == 0.0 {
            out[v * n..v * n + n].copy_from_slice(&data[v * n..v * n + n]);
            continue;
        }
        for t in 0..n {
            // Non-negativity (magnitude data), as DIPY's `clip(min=0)`.
            out[v * n + t] = (thetax[v * n + t] / theta[v]).max(0.0);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jacobi_diagonal() {
        // Diagonal matrix → eigenvalues are the diagonal, ascending.
        let a = vec![3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0];
        let (d, _) = jacobi_eigh(&a, 3);
        assert!((d[0] - 1.0).abs() < 1e-10);
        assert!((d[1] - 2.0).abs() < 1e-10);
        assert!((d[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_jacobi_known_symmetric() {
        // [[2,1],[1,2]] → eigenvalues 1 and 3.
        let a = vec![2.0, 1.0, 1.0, 2.0];
        let (d, w) = jacobi_eigh(&a, 2);
        assert!((d[0] - 1.0).abs() < 1e-9);
        assert!((d[1] - 3.0).abs() < 1e-9);
        // Reconstruct A = W diag(d) Wᵀ.
        let mut recon = [0.0_f64; 4];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    recon[i * 2 + j] += w[i * 2 + k] * d[k] * w[j * 2 + k];
                }
            }
        }
        for (r, a) in recon.iter().zip(a.iter()) {
            assert!((r - a).abs() < 1e-9);
        }
    }

    /// Pure zero-mean noise denoises to values around 0; none of them may be negative.
    #[test]
    fn test_mppca_output_is_never_negative() {
        let (nx, ny, nz, n) = (9usize, 9usize, 9usize, 6usize);
        let mut seed = 7u64;
        let data: Vec<f64> = (0..nx * ny * nz * n)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (seed >> 33) as f64 / (1u64 << 31) as f64 - 0.5
            })
            .collect();
        let den = mppca_denoise(&data, (nx, ny, nz), n, 2, None);
        let r = 2usize;
        let mut zeros = 0;
        for z in r..nz - r {
            for y in r..ny - r {
                for x in r..nx - r {
                    let v = x + y * nx + z * nx * ny;
                    for t in 0..n {
                        assert!(den[v * n + t] >= 0.0, "negative denoised magnitude");
                        zeros += (den[v * n + t] == 0.0) as usize;
                    }
                }
            }
        }
        assert!(zeros > 0, "zero-mean noise should have been clamped somewhere");
    }

    #[test]
    fn test_mppca_reduces_noise_preserves_signal() {
        // Build a low-rank signal volume (2 spatial regions, smooth decay) plus
        // noise, and check MP-PCA reduces the error to ground truth.
        let (nx, ny, nz, n) = (12usize, 12usize, 12usize, 16usize);
        let nvox = nx * ny * nz;
        let mut clean = vec![0.0_f64; nvox * n];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let v = x + y * nx + z * nx * ny;
                    // Two regions with different decay rates.
                    let r2 = if x < nx / 2 { 20.0 } else { 40.0 };
                    let s0 = 100.0;
                    for t in 0..n {
                        let te = (t + 1) as f64 * 0.005;
                        clean[v * n + t] = s0 * (-r2 * te).exp();
                    }
                }
            }
        }
        // Deterministic pseudo-noise.
        let mut noisy = clean.clone();
        let mut seed = 12345u64;
        let mut rand = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // Zero-mean noise in [-1, 1): average two uniforms for a rough Gaussian.
            let u = |s: u64| (s >> 33) as f64 / (1u64 << 31) as f64; // [0,1)
            let s2 = seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
            (u(seed) - 0.5) + (u(s2) - 0.5) // mean 0, in (-1,1)
        };
        let sigma = 3.0;
        for v in noisy.iter_mut() {
            *v += sigma * rand();
        }
        let den = mppca_denoise(&noisy, (nx, ny, nz), n, 2, None);
        // Measure error only over the interior (voxels that are actually denoised;
        // border voxels within the patch radius are copied through unchanged).
        let r = 2usize;
        let interior_err = |a: &[f64]| -> f64 {
            let mut s = 0.0;
            for z in r..nz - r {
                for y in r..ny - r {
                    for x in r..nx - r {
                        let v = x + y * nx + z * nx * ny;
                        for t in 0..n {
                            let d = a[v * n + t] - clean[v * n + t];
                            s += d * d;
                        }
                    }
                }
            }
            s.sqrt()
        };
        let e_noisy = interior_err(&noisy);
        let e_den = interior_err(&den);
        assert!(
            e_den < 0.4 * e_noisy,
            "MP-PCA should cut interior error substantially: noisy {:.1} -> denoised {:.1}",
            e_noisy,
            e_den
        );
    }
}

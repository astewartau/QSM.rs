//! k-space Tukey windowing of the complex multi-echo data, as the SNU χ-separation toolbox does
//! before everything else.
//!
//! # Source
//!
//! [`tukey_window_complex`] is `tukey_windowing` of the **χ-separation toolbox v1.1.3** (SNU-LIST,
//! Jongho Lee's lab; `functions/supportfunc/tukey_windowing.p`, sha256 `e4703efe…e85153`), as
//! `Chisep_script.m` calls it (l. 69-73) on the complex multi-echo data before the mask, the R2*
//! fit and every phase step:
//!
//! ```text
//! tukey_fac = 0.4; % Recommendation: Siemens, GE: 0.4, Philips: 0
//! img_tukey = tukey_windowing(mag_multi_echo.* exp(1i*phs_multi_echo),tukey_fac);
//! mag_multi_echo = abs(img_tukey);
//! phs_multi_echo = angle(img_tukey);
//! ```
//!
//! The toolbox ships as protected P-code; with the authors' permission the function was recovered
//! by black-box probing in licensed MATLAB R2023b (QSMxT/QSMxT#481, 2026-11-03): the profiler
//! (which functions it calls), a logging shadow of `tukeywin` (each call's length and ratio), and
//! the ratio `fftn(Y) ./ fftn(X)` of output to input on random complex volumes of even, odd and
//! singleton sizes, 1-3 echoes and factors 0, 0.25, 0.4 and 1. Findings:
//!
//! - **k-space, per echo, all three spatial axes.** It calls the toolbox's `fft3c`, multiplies by a
//!   window and calls `ifft3c`. `fft3c(x)` is `ifftshift(fft_1(fft_2(fft_3(fftshift(x)))))` and
//!   `ifft3c` the same with `ifft` (unnormalised `fft`, `1/N` in `ifft`; MATLAB's `fftshift` and
//!   `ifftshift` without a dimension argument). The ratio is real, the same for every echo (the
//!   echo axis is never mixed) and separable: `W(u, v, w) = w_x(u)·w_y(v)·w_z(w)`.
//! - **One MATLAB `tukeywin` per axis**, of that axis's length and the given factor ([`tukeywin`]):
//!   `tukeywin(nx, r)`, `tukeywin(ny, r)`, then `tukeywin(nz, r)` once for every (x, y) line.
//!   Factor 0 gives all ones (the identity, to rounding); factor ≥ 1 gives `hann(n)`.
//! - **The window sits one bin off centre.** Applied to the uncentred spectrum, axis `n`'s factor
//!   at FFT index `u` (frequency `u`, or `u − n` above `n/2`) is `tukeywin(n, r)[(u + ⌊n/2⌋ − 1)
//!   mod n]` (0-based), i.e. `ifftshift(circshift(tukeywin(n, r), 1))` in MATLAB. The window's
//!   centre therefore lies at +½ frequency bin for an even `n` and +1 bin for an odd `n`, not at
//!   DC (it would be `ifftshift(tukeywin(n, r))` for a window centred on DC at odd `n`). Real input
//!   so gives complex output. Why is not visible from outside the P-code (an even 16³ cube is
//!   shifted too, so it is not only `fft3c`'s odd-size `fftshift`/`ifftshift` asymmetry). It is
//!   kept here for parity: a centred window would be a different filter, and whether centring it
//!   is an unequivocal improvement has not been measured. Open question.
//!
//! With these, MATLAB's `ifftn(fftn(X(:,:,:,e)) .* W)` reproduces the toolbox's output to
//! 4.5·10⁻¹⁶ relative on all 16 probe cases. Inputs of class `single` come back `single`;
//! QSM.rs works in `f64`.
//!
//! # Agreement with the toolbox
//!
//! `tests/tukey_window_snu_parity.rs` checks [`tukey_window_complex`] against the toolbox's own
//! output (`tests/data/tukey_window_snu_small.bin`, made by `tests/data/tukey_window_snu_small.m`)
//! on four volumes: 10×9×7 with three echoes at 0.4, 16×12×8 at 0.25, 7×8×1 with two echoes at 1
//! (Hann) and 6×5×4 at 0. Measured 2026-11-03: max |Δ| 8.4·10⁻¹⁶ (rounding; outputs up to 1.1).
//!
//! # Vendor defaults
//!
//! The script's recommendation is 0.4 for Siemens and GE and 0 (no windowing) for Philips. The
//! script applies the window before all three of its χ-separation methods (χ-sepnet, χ-sep-MEDI
//! and χ-sep-iLSQR), so it is published as the rules `chi-sep.tukey-siemens-ge` and
//! `chi-sep.tukey-philips` (with χ-sep-MEDI or χ-sep-iLSQR) and `chi-sepnet.tukey-siemens-ge` and
//! `chi-sepnet.tukey-philips` (with χ-sepnet's toolbox route; its native route is not the
//! toolbox's pipeline) ([`crate::pipeline::rules`], on
//! [`QsmPipelineConfig::tukey_window`](crate::pipeline::QsmPipelineConfig::tukey_window) and the
//! scan's [`Manufacturer`](crate::pipeline::rules::Manufacturer)). They are
//! [`Kind::Reference`](crate::pipeline::rules::Kind::Reference) rules, the authors' setting: the
//! comment says "Recommendation" but gives no reason why these values are better, so a host
//! shows them and does not preselect them (decided 2026-11-04; QSM.rs#424 first had them as
//! recommendations for χ-sepnet only). Hosts apply it themselves, to every echo's magnitude
//! and phase on the acquired grid, before anything else reads them
//! ([`tukey_window_mag_phase`]). The default is no windowing (factor 0).

use num_complex::Complex64;

use crate::fft::Fft3dWorkspace;

/// MATLAB's `tukeywin(n, r)` (Signal Processing Toolbox, R2023b): a tapered cosine window of `n`
/// points with taper ratio `r`.
///
/// `n = 1` gives `[1]`; `r ≤ 0` all ones (a rectangle); `r ≥ 1` the symmetric `hann(n)`,
/// `0.5·(1 − cos(2πk/(n − 1)))`. Otherwise, with `t = linspace(0, 1, n)`, `p = r/2`,
/// `tl = ⌊p(n − 1)⌋ + 1` and `th = n − tl + 1` (1-based): `(1 + cos(π/p·(t − p)))/2` on the first
/// `tl` points, `(1 + cos(π/p·(t − 1 + p)))/2` from point `th` on, and 1 between.
pub fn tukeywin(n: usize, r: f64) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    if r <= 0.0 {
        return vec![1.0; n];
    }
    let m = (n - 1) as f64;
    if r >= 1.0 {
        return (0..n).map(|k| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * k as f64 / m).cos()).collect();
    }
    // MATLAB's linspace(0, 1, n): k·(1/(n−1)), with the last point exactly 1.
    let step = 1.0 / m;
    let t = |k: usize| if k == n - 1 { 1.0 } else { k as f64 * step };
    let per = r / 2.0;
    let tl = (per * m).floor() as usize + 1;
    let th = n - tl + 1;
    let pi = std::f64::consts::PI;
    (0..n)
        .map(|k| {
            if k < tl {
                (1.0 + (pi / per * (t(k) - per)).cos()) / 2.0
            } else if k + 1 >= th {
                (1.0 + (pi / per * (t(k) - 1.0 + per)).cos()) / 2.0
            } else {
                1.0
            }
        })
        .collect()
}

/// Axis `n`'s factors of the toolbox's window, in uncentred FFT order: index `u` takes
/// `tukeywin(n, r)[(u + ⌊n/2⌋ − 1) mod n]` (see the module docs: the toolbox's window sits one
/// bin off centre).
pub fn toolbox_axis_window(n: usize, r: f64) -> Vec<f64> {
    let w = tukeywin(n, r);
    (0..n).map(|u| w[(u + n / 2 + n - 1) % n]).collect()
}

/// The toolbox's `tukey_windowing` on one echo, in place: `data` is the complex image
/// (column-major, x fastest) on a `dims` grid, `factor` the taper ratio (0.4 for Siemens and GE,
/// 0 for Philips, by the toolbox's recommendation). Factor 0 (or less) leaves `data` unchanged;
/// the toolbox's FFT round trip would change it only by rounding.
///
/// Every echo is windowed alike and on its own, so a multi-echo volume is one call per echo.
pub fn tukey_window_complex(data: &mut [Complex64], dims: (usize, usize, usize), factor: f64) {
    let (nx, ny, nz) = dims;
    assert_eq!(data.len(), nx * ny * nz, "tukey_window_complex: data is not {nx}x{ny}x{nz}");
    if factor <= 0.0 {
        return;
    }
    let (wx, wy, wz) = (toolbox_axis_window(nx, factor), toolbox_axis_window(ny, factor), toolbox_axis_window(nz, factor));
    let mut ws = Fft3dWorkspace::new(nx, ny, nz);
    ws.fft3d(data);
    for k in 0..nz {
        for j in 0..ny {
            let wjk = wy[j] * wz[k];
            let row = &mut data[(k * ny + j) * nx..(k * ny + j + 1) * nx];
            for (v, &w) in row.iter_mut().zip(&wx) {
                *v *= w * wjk;
            }
        }
    }
    ws.ifft3d(data);
}

/// [`tukey_window_complex`] on one echo given as magnitude and phase (radians), in place: the
/// complex image `m·e^{iφ}` is windowed and split back into `|z|` and `arg z` (MATLAB's `abs` and
/// `angle`), as `Chisep_script.m` does. Factor 0 leaves both unchanged.
pub fn tukey_window_mag_phase(magnitude: &mut [f64], phase: &mut [f64], dims: (usize, usize, usize), factor: f64) {
    assert_eq!(magnitude.len(), phase.len(), "tukey_window_mag_phase: magnitude and phase differ in size");
    if factor <= 0.0 {
        return;
    }
    let mut z: Vec<Complex64> = magnitude.iter().zip(phase.iter()).map(|(&m, &p)| Complex64::from_polar(m, p)).collect();
    tukey_window_complex(&mut z, dims, factor);
    for ((m, p), v) in magnitude.iter_mut().zip(phase.iter_mut()).zip(&z) {
        *m = v.norm();
        *p = v.im.atan2(v.re);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MATLAB R2023b's `tukeywin` values (printed there).
    #[test]
    fn tukeywin_matches_matlab() {
        let close = |a: &[f64], b: &[f64]| a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 5e-5);
        assert!(close(&tukeywin(16, 0.4), &[0.0, 0.25, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.75, 0.25, 0.0]));
        assert!(close(&tukeywin(16, 0.25), &[0.0, 0.5523, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5523, 0.0]));
        assert!(close(&tukeywin(9, 0.4), &[0.0, 0.6913, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6913, 0.0]));
        assert!(close(&tukeywin(9, 1.0), &[0.0, 0.1464, 0.5, 0.8536, 1.0, 0.8536, 0.5, 0.1464, 0.0]));
        assert!(close(&tukeywin(8, 0.4), &[0.0, 0.8117, 1.0, 1.0, 1.0, 1.0, 0.8117, 0.0]));
        assert_eq!(tukeywin(5, 0.0), vec![1.0; 5]);
        assert_eq!(tukeywin(1, 0.4), vec![1.0]);
    }

    /// The window sits one bin off DC: at +½ bin for even n, +1 for odd n.
    #[test]
    fn the_toolbox_window_is_off_centre() {
        let w = toolbox_axis_window(9, 1.0);
        let peak = (0..9).max_by(|&a, &b| w[a].total_cmp(&w[b])).unwrap();
        assert_eq!(peak, 1);
        let w = toolbox_axis_window(16, 1.0);
        assert_eq!(w[0], w[1]);
        assert!(w[0] > w[15] && w[1] > w[2]);
    }

    #[test]
    fn factor_zero_is_the_identity() {
        let mut m = vec![1.0, 2.0, 3.0, 4.0];
        let mut p = vec![0.1, -0.2, 0.3, 3.0];
        let (m0, p0) = (m.clone(), p.clone());
        tukey_window_mag_phase(&mut m, &mut p, (2, 2, 1), 0.0);
        assert_eq!((m, p), (m0, p0));
    }

    /// The magnitude/phase form is the complex one through `|z|` and `arg z`.
    #[test]
    fn mag_phase_goes_through_the_complex_image() {
        let dims = (6, 5, 3);
        let n = 90;
        let mut m: Vec<f64> = (0..n).map(|i| 1.0 + (i % 7) as f64).collect();
        let mut p: Vec<f64> = (0..n).map(|i| ((i * 13) % 17) as f64 / 17.0 * 6.0 - 3.0).collect();
        let mut z: Vec<Complex64> = m.iter().zip(&p).map(|(&a, &b)| Complex64::from_polar(a, b)).collect();
        tukey_window_complex(&mut z, dims, 0.4);
        tukey_window_mag_phase(&mut m, &mut p, dims, 0.4);
        for i in 0..n {
            assert!((m[i] - z[i].norm()).abs() < 1e-14 && (p[i] - z[i].arg()).abs() < 1e-14);
        }
    }
}

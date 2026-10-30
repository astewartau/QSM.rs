//! N4 bias field correction
//!
//! [`n4_bias_correction`] implements **ANTs' `N4BiasFieldCorrection`** (ANTs 2.4.3, built on
//! ITK 5.3.0's `itk::N4BiasFieldCorrectionImageFilter`), the N4ITK algorithm of Tustison et
//! al. (2010). [`N4Params::default`] is what the ANTs command line does when it is given only
//! an input and an output, `N4BiasFieldCorrection -i in.nii -o out.nii`; the other fields map
//! to its options (see [`N4Params`]). The steps follow `N4BiasFieldCorrection.cxx`,
//! `itkN4BiasFieldCorrectionImageFilter.hxx`, `itkBSplineScatteredDataPointSetToImageFilter.hxx`,
//! `itkBSplineControlPointImageFilter.hxx` and `itkShrinkImageFilter.hxx`:
//!
//! 1. **Mask.** Without a mask the whole image is the mask (ANTs fills one with ones; it does
//!    *not* threshold). With a mask, voxels where it is non-zero take part.
//! 2. **Shrink** by `s` = [`N4Params::shrink_factor`] with ITK's `ShrinkImageFilter`, which
//!    subsamples (no averaging): `m = max(1, ⌊n/s⌋)` samples per axis, at input indices
//!    `o + s·i` where `o` keeps the physical centres of both grids together,
//!    `o = round((n−1)/2 − s·(m−1)/2)`, halves rounded up.
//! 3. **Log**: `log v` for voxels in the mask with `v > 0`; every other value is left as it
//!    is (a zero stays 0, the log of an intensity of 1).
//! 4. **Iterate.** For each fitting level (one per entry of [`N4Params::max_iterations`]), up
//!    to that many times while the convergence value exceeds the threshold:
//!    - *Sharpen* the current log image `u` (inside the mask): build a histogram of `u`
//!      (200 bins by default) between its minimum and maximum with linear (triangular Parzen)
//!      binning, zero-pad it to `2^(⌈log₂ bins⌉+1)` bins, Wiener-deconvolve it by a Gaussian
//!      (FWHM 0.15 in log intensity, noise 0.01 by default), clip negatives to zero, compute
//!      the conditional mean
//!      `E[u|v]` by convolving `bin centre · U` and `U` with the Gaussian and dividing, and map
//!      each voxel through `E` by linear interpolation.
//!    - The residual `u − E[u|v]` is fitted by a B-spline (cubic by default) with the scattered-data
//!      approximation of Lee, Wolberg & Shin (1997) (a single level, ITK's
//!      `BSplineScatteredDataPointSetToImageFilter`), each voxel a point at its index scaled
//!      onto the parametric domain `[0, spans)` (the last voxel moved in by `10⁻³` of a voxel).
//!      The control points are added to the running log bias field lattice, which is
//!      evaluated on the shrunk grid.
//!    - Convergence value: coefficient of variation (`σ/μ`) of `exp(old − new log field)` over
//!      the mask, computed with ITK's single-precision running mean and variance.
//!    - `u` = log input − log bias field.
//!
//!    Between levels (not after the last) the lattice is refined to twice as many spans, the
//!    same spline exactly. The default starts from one span per axis (4 control points for a
//!    cubic) and ends with 8 (11 control points).
//! 5. **Output.** The lattice is evaluated on the full-resolution grid, its first voxel on the
//!    start and its last on the end of the parametric domain (ITK's
//!    `BSplineControlPointImageFilter` maps the grid onto the domain this way regardless of the
//!    grid's physical position, so the full grid's field is the shrunk grid's stretched by
//!    `(n−1)/(s·(m−1))`). The bias field is `exp` of it and the corrected image is
//!    `input / bias`. With a mask, voxels outside it keep the input value and, unless
//!    [`N4Params::rescale_intensities`] is off, the corrected voxels where the mask is 1 are
//!    rescaled linearly so their minimum and maximum inside the mask match the input's, as
//!    the ANTs command line does (`-r 1` is its default, and applies only with a mask).
//!
//! Everything but the convergence value is computed in double precision; ITK computes in single
//! precision (its histogram FFTs are double).
//!
//! # Agreement with ANTs
//!
//! Compared with ANTs 2.4.3 (`N4BiasFieldCorrection -i AvgEcho.nii -o [out,bias]`, 4 threads) on
//! three in-vivo 3 T mean-over-echo magnitudes (QSMART's vessel-detection input):
//!
//! | Volume | Voxel (mm) | Corrected image: max relative difference (relative RMS) | ANTs 1 thread vs 4 threads |
//! |---|---|---|---|
//! | 178×182×148 | 1×1×1 | 8.4e-4 (6.7e-4) | 9.4e-4 (7.6e-4) |
//! | 178×180×64 | 1×1×2 | 1.4e-4 (1.1e-4) | 3.6e-5 (3.0e-5) |
//! | 168×224×150 | 0.94×0.94×1 | 2.5e-6 (1.8e-7) | 1.7e-5 (1.0e-5) |
//!
//! The bias field differs by the same. After one iteration the bias field agrees to single
//! precision (1.2e-7), and the convergence values of all 200 iterations agree to ~1e-4
//! relative (so a convergence threshold stops each level after the same iterations as ANTs).
//! What remains is rounding amplified by the later, finer-mesh levels, whose iterations do not
//! contract: ANTs' own output moves as much with its thread count (its B-spline fit sums the
//! points in per-thread blocks), and rounding the input to single precision, as ANTs does when
//! it reads a double-precision NIfTI, moves this implementation's output by 6.7e-4 and
//! 1.7e-4 on the first two volumes. Spline orders 1 and 2 and the masked, rescaled output agree
//! as closely (≤ 5e-5 and 6e-6 on the first volume). `tests/n4_ants_parity.rs` checks a
//! 30×32×26 synthetic volume against ANTs, with and without a mask: against the verified QSM-CI
//! `ants-n4` image (ANTs 2.4.3, 4 threads; QSMxT/QSM-CI#302; `tests/data/n4_ants_small.py`
//! regenerates it in Actions) the bias field agrees to 1.5e-5 and the masked, rescaled output to
//! 6.2e-5 relative (measured 2026-10-30).
//!
//! # Shrink offset
//!
//! ITK finds the shrink offset `o` above by mapping a physical point back to an index and
//! rounding, so where `(n−1)/2 − s·(m−1)/2` is a half-integer, ITK's rounding of a value that
//! is exactly `k + ½` in exact arithmetic depends on the image's origin, spacing and direction
//! cosines in floating point and can come out at `k` instead of `k + 1`. Here it is always
//! `k + 1` (what ITK computes for an axis-aligned image whose origin and spacing are exact in
//! binary, as NIfTI images with integer or dyadic geometry are). [`Grid`] carries no origin or
//! direction, so other geometries cannot be reproduced; the difference is a one-voxel shift of
//! the samples the bias field is fitted to.
//!
//! # Reference pipeline
//!
//! In QSM, QSMART runs `N4BiasFieldCorrection -i mag1_sos.nii -o mag1_sos_n4.nii` (ANTs defaults,
//! no mask) on the first echo's coil-RSS magnitude before `bet2 -f 0.2`
//! (`QSMART_toolbox_v1.0/brainmask.m:7-14`), and on the echo-averaged magnitude for vessel
//! detection (the comparison above). Nothing in N4 assumes a field strength or orientation.
//!
//! References:
//! Tustison, N.J., Avants, B.B., Cook, P.A., Zheng, Y., Egan, A., Yushkevich, P.A., Gee, J.C.
//! (2010). "N4ITK: Improved N3 Bias Correction." IEEE Trans. Med. Imaging 29(6):1310-1320.
//! <https://doi.org/10.1109/TMI.2010.2046908>
//!
//! Sled, J.G., Zijdenbos, A.P., Evans, A.C. (1998). "A nonparametric method for automatic
//! correction of intensity nonuniformity in MRI data." IEEE Trans. Med. Imaging 17(1):87-97.
//!
//! Lee, S., Wolberg, G., Shin, S.Y. (1997). "Scattered data interpolation with multilevel
//! B-splines." IEEE Trans. Vis. Comput. Graph. 3(3):228-244.
//!
//! Reference implementation: <https://github.com/ANTsX/ANTs> (`Examples/N4BiasFieldCorrection.cxx`)
//! and <https://github.com/InsightSoftwareConsortium/ITK> (`Modules/Filtering/BiasCorrection`).

use crate::error::InputError;
use crate::Grid;
use num_complex::Complex64;
use rustfft::FftPlanner;

/// Parameters of [`n4_bias_correction`]. The defaults are those of ANTs'
/// `N4BiasFieldCorrection` command line when only `-i` and `-o` are given.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct N4Params {
    /// Subsampling factor applied before fitting (`-s`, default 4).
    pub shrink_factor: usize,
    /// Maximum iterations at each fitting level; the number of entries is the number of levels
    /// (`-c [50x50x50x50, ...]`, default four levels of 50).
    pub max_iterations: Vec<usize>,
    /// Stop a level once the coefficient of variation of the change in the bias field falls to
    /// this value (`-c [..., 0.0]`, default 0: every level runs all its iterations).
    pub convergence_threshold: f64,
    /// B-spline mesh elements (spans) per axis at the first level (`-b [1x1x1, ...]`, default
    /// one span over the whole image); each later level doubles them.
    pub mesh: [usize; 3],
    /// B-spline order (`-b [..., 3]`, default cubic). 1, 2 or 3.
    pub spline_order: usize,
    /// FWHM of the Gaussian that models the bias field's blurring of the log-intensity
    /// histogram (`-t [0.15, ...]`).
    pub bias_field_fwhm: f64,
    /// Wiener filter noise of the histogram deconvolution (`-t [..., 0.01, ...]`).
    pub wiener_noise: f64,
    /// Histogram bins (`-t [..., 200]`).
    pub histogram_bins: usize,
    /// With a mask, rescale the corrected voxels to the input's intensity range inside the
    /// mask (`-r`, default on). Ignored without a mask, as in ANTs.
    pub rescale_intensities: bool,
}

impl Default for N4Params {
    fn default() -> Self {
        Self {
            shrink_factor: 4,
            max_iterations: vec![50; 4],
            convergence_threshold: 0.0,
            mesh: [1, 1, 1],
            spline_order: 3,
            bias_field_fwhm: 0.15,
            wiener_noise: 0.01,
            histogram_bins: 200,
            rescale_intensities: true,
        }
    }
}

impl N4Params {
    /// Checks the parameters [`n4_bias_correction`] cannot run with, without an image: at least
    /// one fitting level, the first with at least one iteration (ITK refines the lattice the
    /// previous level fitted, so the first level must fit one), a shrink factor and mesh sizes of at least 1, a spline order of 1, 2 or
    /// 3, at least two histogram bins, a finite FWHM > 0, a finite Wiener noise ≥ 0 and a finite
    /// convergence threshold. The errors name the parameter and its value. A host can call this
    /// (and [`Self::check_grid`] once it knows the grid) to refuse a run before any work starts;
    /// [`n4_bias_correction`] runs the same checks.
    pub fn validate(&self) -> Result<(), InputError> {
        if self.max_iterations.first().is_none_or(|&n| n == 0) {
            return Err(InputError::param(
                "N4 max_iterations",
                format!("{:?}", self.max_iterations),
                "at least one fitting level, the first with at least one iteration",
            ));
        }
        if self.shrink_factor == 0 {
            return Err(InputError::param("N4 shrink_factor", 0, ">= 1"));
        }
        if self.mesh.contains(&0) {
            return Err(InputError::param("N4 mesh", format!("{:?}", self.mesh), ">= 1 along each axis"));
        }
        if !(1..=3).contains(&self.spline_order) {
            return Err(InputError::param("N4 spline_order", self.spline_order, "1, 2 or 3"));
        }
        if self.histogram_bins < 2 {
            return Err(InputError::param("N4 histogram_bins", self.histogram_bins, ">= 2"));
        }
        if !(self.bias_field_fwhm.is_finite() && self.bias_field_fwhm > 0.0) {
            return Err(InputError::param("N4 bias_field_fwhm", self.bias_field_fwhm, "finite and > 0"));
        }
        if !(self.wiener_noise.is_finite() && self.wiener_noise >= 0.0) {
            return Err(InputError::param("N4 wiener_noise", self.wiener_noise, "finite and >= 0"));
        }
        if !self.convergence_threshold.is_finite() {
            return Err(InputError::param("N4 convergence_threshold", self.convergence_threshold, "finite"));
        }
        Ok(())
    }

    /// Checks that a grid of `dims` voxels is large enough for these parameters: N4 fits the
    /// bias field on the image shrunk by [`Self::shrink_factor`] (ITK's `ShrinkImageFilter`,
    /// `n / shrink_factor` voxels per axis), and its B-spline fit needs at least two voxels along
    /// each axis of the shrunk grid, i.e. `2 · shrink_factor` voxels per axis. The error names
    /// the axis, its size and the shrink factor. Assumes `shrink_factor >= 1` (see
    /// [`Self::validate`]).
    pub fn check_grid(&self, dims: (usize, usize, usize)) -> Result<(), InputError> {
        let s = self.shrink_factor.max(1);
        for (axis, n) in [dims.0, dims.1, dims.2].into_iter().enumerate() {
            if n / s < 2 {
                return Err(InputError(format!(
                    "N4 needs at least {} voxels along each axis at shrink_factor = {s} (two after \
                     shrinking); axis {axis} of the {}x{}x{} grid has {n}",
                    2 * s, dims.0, dims.1, dims.2
                )));
            }
        }
        Ok(())
    }
}

/// Output of [`n4_bias_correction`].
#[derive(Clone, Debug)]
pub struct N4Output {
    /// Bias-corrected image.
    pub corrected: Vec<f64>,
    /// Estimated (multiplicative) bias field; `corrected = input / bias_field`, before the
    /// masking and rescaling described in [`n4_bias_correction`].
    pub bias_field: Vec<f64>,
}

/// N4 bias field correction, as ANTs' `N4BiasFieldCorrection` (see the module documentation).
///
/// # Arguments
/// * `image` - Input image (nx * ny * nz, Fortran order). N4 log-transforms it, so it should be
///   positive where it matters; zeros are allowed (ANTs leaves them at 0 in the log domain).
/// * `mask` - Optional mask (`-x`): non-zero voxels are used for fitting; outside it the output
///   is the input. `None` uses the whole image, as ANTs does.
/// * `grid` - Grid dimensions and voxel size.
/// * `params` - See [`N4Params`]; the default reproduces the ANTs command line's defaults.
/// * `progress_callback` - Called with (iterations done, maximum iterations) after each
///   iteration.
///
/// # Errors
/// [`InputError`] if the parameters fail [`N4Params::validate`], the grid fails
/// [`N4Params::check_grid`] (fewer than two voxels along an axis after shrinking), the image or
/// mask does not match the grid, or the mask selects no voxel of the shrunk grid.
pub fn n4_bias_correction(
    image: &[f64],
    mask: Option<&[u8]>,
    grid: &Grid,
    params: &N4Params,
    progress_callback: impl Fn(usize, usize),
) -> Result<N4Output, InputError> {
    params.validate()?;
    let (nx, ny, nz) = grid.dims;
    let n_total = nx * ny * nz;
    let check_volume = |name: &str, len: usize| {
        if len == n_total {
            Ok(())
        } else {
            Err(InputError(format!(
                "N4 {name} has {len} voxels but the grid is {nx}x{ny}x{nz} ({n_total} voxels)"
            )))
        }
    };
    check_volume("image", image.len())?;
    if let Some(m) = mask {
        check_volume("mask", m.len())?;
    }
    params.check_grid(grid.dims)?;
    let order = params.spline_order;

    let dims = [grid.dims.0, grid.dims.1, grid.dims.2];

    // --- Shrink (ITK ShrinkImageFilter: subsampling, centres aligned) ---
    let s = params.shrink_factor;
    let mut sdims = [0usize; 3];
    let mut offs = [0usize; 3];
    for d in 0..3 {
        // At least 2, by `check_grid` above.
        let m = dims[d] / s;
        sdims[d] = m;
        // o = (n-1)/2 - s(m-1)/2, rounded half up; in halves: k = (n-1) - s(m-1).
        let k = (dims[d] as i64 - 1) - (s as i64) * (m as i64 - 1);
        offs[d] = ((k + 1).div_euclid(2)).max(0) as usize;
    }
    let ns = sdims[0] * sdims[1] * sdims[2];
    let mut log_input = vec![0.0f64; ns];
    let mut included = vec![false; ns];
    for k in 0..sdims[2] {
        for j in 0..sdims[1] {
            for i in 0..sdims[0] {
                let src = (offs[0] + s * i)
                    + (offs[1] + s * j) * dims[0]
                    + (offs[2] + s * k) * dims[0] * dims[1];
                let dst = i + j * sdims[0] + k * sdims[0] * sdims[1];
                log_input[dst] = image[src];
                included[dst] = mask.is_none_or(|m| m[src] != 0);
            }
        }
    }
    let inc_idx: Vec<usize> = (0..ns).filter(|&i| included[i]).collect();
    if inc_idx.is_empty() {
        return Err(InputError(format!(
            "N4's mask selects no voxel of the image shrunk by shrink_factor = {s} \
             ({}x{}x{} voxels)",
            sdims[0], sdims[1], sdims[2]
        )));
    }

    // --- Log transform (positive voxels inside the mask only) ---
    for &i in &inc_idx {
        if log_input[i] > 0.0 {
            log_input[i] = log_input[i].ln();
        }
    }

    // --- Iterate ---
    let sharpener = Sharpener::new(params);
    let mut log_uncorrected = log_input.clone();
    let mut log_bias = vec![0.0f64; ns];
    let mut sharpened = vec![0.0f64; ns];
    let mut residual = vec![0.0f64; inc_idx.len()];
    let mut ncp = [params.mesh[0] + order, params.mesh[1] + order, params.mesh[2] + order];
    let mut lattice: Option<Vec<f64>> = None;
    let n_levels = params.max_iterations.len();
    let total_iterations: usize = params.max_iterations.iter().sum();
    let mut done = 0usize;
    for (level, &max_iter) in params.max_iterations.iter().enumerate() {
        let tables = axis_tables(sdims, ncp, order);
        let threshold = params.convergence_threshold as f32;
        let mut convergence = f32::MAX;
        let mut elapsed = 0usize;
        while elapsed < max_iter && convergence > threshold {
            elapsed += 1;
            sharpener.sharpen(&log_uncorrected, &inc_idx, &mut sharpened);
            for (r, &i) in residual.iter_mut().zip(&inc_idx) {
                *r = log_uncorrected[i] - sharpened[i];
            }
            let phi = fit_lattice(&residual, &inc_idx, sdims, ncp, &tables);
            let lat = match lattice.take() {
                None => phi,
                Some(mut l) => {
                    for (a, b) in l.iter_mut().zip(&phi) {
                        *a += b;
                    }
                    l
                }
            };
            let new_log_bias = evaluate_lattice(&lat, sdims, ncp, &tables);
            lattice = Some(lat);
            convergence = convergence_measure(&log_bias, &new_log_bias, &inc_idx);
            log_bias = new_log_bias;
            for i in 0..ns {
                log_uncorrected[i] = log_input[i] - log_bias[i];
            }
            done += 1;
            progress_callback(done, total_iterations);
        }
        if level + 1 < n_levels {
            if let Some(l) = lattice.take() {
                let (refined, new_ncp) = refine_lattice(&l, ncp, order);
                lattice = Some(refined);
                ncp = new_ncp;
            } else {
                unreachable!("validate() requires an iteration at the first level");
            }
        }
    }
    let lattice = lattice.expect("n4_bias_correction: no iteration was run");

    // --- Full-resolution bias field and corrected image ---
    let full_tables = axis_tables(dims, ncp, order);
    let log_field = evaluate_lattice(&lattice, dims, ncp, &full_tables);
    let bias_field: Vec<f64> = log_field.iter().map(|v| v.exp()).collect();
    let mut corrected: Vec<f64> = image.iter().zip(&bias_field).map(|(v, b)| v / b).collect();

    if let Some(m) = mask {
        for i in 0..n_total {
            if m[i] == 0 {
                corrected[i] = image[i];
            }
        }
        if params.rescale_intensities {
            let range = |v: &[f64]| {
                let mut lo = f64::INFINITY;
                let mut hi = f64::NEG_INFINITY;
                for i in 0..n_total {
                    if m[i] != 0 {
                        lo = lo.min(v[i]);
                        hi = hi.max(v[i]);
                    }
                }
                (lo, hi)
            };
            let (min_o, max_o) = range(image);
            let (min_b, max_b) = range(&corrected);
            let slope = (max_o - min_o) / (max_b - min_b);
            for i in 0..n_total {
                if m[i] == 1 {
                    corrected[i] = max_o - slope * (max_b - corrected[i]);
                }
            }
        }
    }

    Ok(N4Output { corrected, bias_field })
}

//=============================================================================
// Histogram sharpening (N4BiasFieldCorrectionImageFilter::SharpenImage)
//=============================================================================

struct Sharpener {
    bins: usize,
    padded: usize,
    offset: usize,
    fwhm: f64,
    noise: f64,
    fft: std::sync::Arc<dyn rustfft::Fft<f64>>,
    ifft: std::sync::Arc<dyn rustfft::Fft<f64>>,
}

impl Sharpener {
    fn new(params: &N4Params) -> Self {
        let bins = params.histogram_bins;
        // ITK: exponent = ceil(log(float bins) / log(2.0)) + 1, with log of a float.
        let exponent = (((bins as f32).ln() as f64) / 2f64.ln()).ceil() + 1.0;
        let padded = (2f32.powf(exponent as f32) + 0.5) as usize;
        let offset = (0.5 * (padded - bins) as f64) as usize;
        let mut planner = FftPlanner::new();
        Self {
            bins,
            padded,
            offset,
            fwhm: params.bias_field_fwhm,
            noise: params.wiener_noise,
            fft: planner.plan_fft_forward(padded),
            ifft: planner.plan_fft_inverse(padded),
        }
    }

    fn sharpen(&self, image: &[f64], inc_idx: &[usize], out: &mut [f64]) {
        let (bins, np, off) = (self.bins, self.padded, self.offset);

        // Bin range. ITK's loop updates the minimum only when a value is not a new maximum.
        let mut bin_max = -f64::MAX;
        let mut bin_min = f64::MAX;
        for &i in inc_idx {
            let v = image[i];
            if v > bin_max {
                bin_max = v;
            } else if v < bin_min {
                bin_min = v;
            }
        }
        let slope = (bin_max - bin_min) / (bins - 1) as f64;

        // Histogram with linear (triangular Parzen) binning.
        let mut h = vec![0.0f64; bins];
        for &i in inc_idx {
            let cidx = (image[i] - bin_min) / slope;
            let fl = cidx.floor();
            if fl < 0.0 {
                continue; // ITK's unsigned index wraps: the value is dropped
            }
            let idx = fl as usize;
            let frac = cidx - fl;
            if frac == 0.0 {
                if idx < bins {
                    h[idx] += 1.0;
                }
            } else if idx < bins - 1 {
                h[idx] += 1.0 - frac;
                h[idx + 1] += frac;
            }
        }

        let mut vf = vec![Complex64::new(0.0, 0.0); np];
        for n in 0..bins {
            vf[n + off] = Complex64::new(h[n], 0.0);
        }
        self.fft.process(&mut vf);

        // Gaussian of the given FWHM (in bins), sampled symmetrically for the circular FFT.
        let scaled_fwhm = self.fwhm / slope;
        let exp_factor = 4.0 * 2f64.ln() / (scaled_fwhm * scaled_fwhm);
        let scale_factor = 2.0 * (2f64.ln() / std::f64::consts::PI).sqrt() / scaled_fwhm;
        let mut ff = vec![Complex64::new(0.0, 0.0); np];
        ff[0] = Complex64::new(scale_factor, 0.0);
        let half = np / 2;
        for n in 1..=half {
            let g = Complex64::new(scale_factor * (-((n * n) as f64) * exp_factor).exp(), 0.0);
            ff[n] = g;
            ff[np - n] = g;
        }
        self.fft.process(&mut ff);

        // Wiener deconvolution; ITK applies the real part of the filter.
        let mut u = vf;
        for n in 0..np {
            let c = ff[n].conj();
            let g = c / (c * ff[n] + self.noise);
            u[n] *= g.re;
        }
        self.ifft.process(&mut u);
        for x in u.iter_mut() {
            *x = Complex64::new(x.re.max(0.0), 0.0);
        }

        // E[u|v] = (G * (x U)) / (G * U).
        let mut num: Vec<Complex64> = (0..np)
            .map(|n| Complex64::new((bin_min + (n as f64 - off as f64) * slope) * u[n].re, 0.0))
            .collect();
        self.fft.process(&mut num);
        for n in 0..np {
            num[n] *= ff[n];
        }
        self.ifft.process(&mut num);
        let mut den = u;
        self.fft.process(&mut den);
        for n in 0..np {
            den[n] *= ff[n];
        }
        self.ifft.process(&mut den);
        let e: Vec<f64> = (off..off + bins)
            .map(|n| if den[n].re != 0.0 { num[n].re / den[n].re } else { 0.0 })
            .collect();

        // Map through E by linear interpolation.
        out.iter_mut().for_each(|x| *x = 0.0);
        for &i in inc_idx {
            let cidx = (image[i] - bin_min) / slope;
            let fl = cidx.floor();
            out[i] = if fl >= 0.0 && (fl as usize) < bins - 1 {
                let idx = fl as usize;
                e[idx] + (e[idx + 1] - e[idx]) * (cidx - fl)
            } else {
                e[bins - 1]
            };
        }
    }
}

//=============================================================================
// B-splines (ITK BSplineScatteredDataPointSetToImageFilter / BSplineControlPointImageFilter)
//=============================================================================

/// ITK's `BSplineKernelFunction` of order 1-3 (centred B-spline).
#[inline]
fn bspline_kernel(order: usize, u: f64) -> f64 {
    let a = u.abs();
    match order {
        1 => if a < 1.0 { 1.0 - a } else { 0.0 },
        2 => {
            if a < 0.5 {
                0.75 - a * a
            } else if a < 1.5 {
                (9.0 - 12.0 * a + 4.0 * a * a) * 0.125
            } else {
                0.0
            }
        }
        3 => {
            if a < 1.0 {
                (4.0 - 6.0 * a * a + 3.0 * a * a * a) / 6.0
            } else if a < 2.0 {
                (8.0 - 12.0 * a + 6.0 * a * a - a * a * a) / 6.0
            } else {
                0.0
            }
        }
        _ => unreachable!(),
    }
}

/// For each axis, the first control point and the `order + 1` weights that every grid index
/// along it touches. Grid index `i` of `m` sits at parametric `u = spans·i/(m−1)`, moved to
/// `spans − ε` when within `ε = 10⁻³·spans/(m−1)` of the end (ITK's `BSplineEpsilon`).
struct AxisTables {
    order: usize,
    cell: [Vec<usize>; 3],
    w: [Vec<f64>; 3], // m * (order + 1)
}

fn axis_tables(dims: [usize; 3], ncp: [usize; 3], order: usize) -> AxisTables {
    let mut cell: [Vec<usize>; 3] = Default::default();
    let mut w: [Vec<f64>; 3] = Default::default();
    for d in 0..3 {
        let m = dims[d];
        let spans = (ncp[d] - order) as f64;
        let eps = spans / (m - 1) as f64 * 1e-3;
        let mut c = Vec::with_capacity(m);
        let mut ws = Vec::with_capacity(m * (order + 1));
        for i in 0..m {
            let mut u = spans * i as f64 / (m - 1) as f64;
            if (u - spans).abs() <= eps {
                u = spans - eps;
            }
            if u < 0.0 && u.abs() <= eps {
                u = 0.0;
            }
            let fl = u.floor();
            c.push(fl as usize);
            for k in 0..=order {
                ws.push(bspline_kernel(order, u - fl - k as f64 + 0.5 * (order as f64 - 1.0)));
            }
        }
        cell[d] = c;
        w[d] = ws;
    }
    AxisTables { order, cell, w }
}

/// One B-spline approximation level (Lee et al. 1997) of `values` at the voxels `inc_idx` of a
/// grid of `dims`, on a lattice of `ncp` control points (all point weights 1).
fn fit_lattice(
    values: &[f64],
    inc_idx: &[usize],
    dims: [usize; 3],
    ncp: [usize; 3],
    t: &AxisTables,
) -> Vec<f64> {
    let o1 = t.order + 1;
    let nl = ncp[0] * ncp[1] * ncp[2];
    let mut delta = vec![0.0f64; nl];
    let mut omega = vec![0.0f64; nl];
    let plane = dims[0] * dims[1];
    for (&v, &idx) in values.iter().zip(inc_idx) {
        let (i, j, k) = (idx % dims[0], (idx / dims[0]) % dims[1], idx / plane);
        let wx = &t.w[0][i * o1..(i + 1) * o1];
        let wy = &t.w[1][j * o1..(j + 1) * o1];
        let wz = &t.w[2][k * o1..(k + 1) * o1];
        let sq = |w: &[f64]| w.iter().map(|x| x * x).sum::<f64>();
        let w2_sum = sq(wx) * sq(wy) * sq(wz);
        let (ci, cj, ck) = (t.cell[0][i], t.cell[1][j], t.cell[2][k]);
        for (c, &bz) in wz.iter().enumerate() {
            for (b, &by) in wy.iter().enumerate() {
                let byz = by * bz;
                let base = ci + (cj + b) * ncp[0] + (ck + c) * ncp[0] * ncp[1];
                for (a, &bx) in wx.iter().enumerate() {
                    let tw = bx * byz;
                    let t2 = tw * tw;
                    omega[base + a] += t2;
                    delta[base + a] += v * t2 * tw / w2_sum;
                }
            }
        }
    }
    // ITK: phi = delta / omega where omega is not (almost) zero, non-finite results set to 0.
    let tiny = 0.1 * f32::EPSILON as f64;
    delta
        .iter()
        .zip(&omega)
        .map(|(&d, &o)| {
            if o.abs() > tiny {
                let p = d / o;
                if p.is_finite() { p } else { 0.0 }
            } else {
                0.0
            }
        })
        .collect()
}

/// Evaluate a control point lattice on a grid of `dims` (separably, z then y then x).
fn evaluate_lattice(lattice: &[f64], dims: [usize; 3], ncp: [usize; 3], t: &AxisTables) -> Vec<f64> {
    let o1 = t.order + 1;
    let (nx, ny, nz) = (dims[0], dims[1], dims[2]);
    let (cx, cy) = (ncp[0], ncp[1]);
    let mut out = vec![0.0f64; nx * ny * nz];
    let mut plane = vec![0.0f64; cx * cy];
    let mut line = vec![0.0f64; cx];
    for k in 0..nz {
        let wz = &t.w[2][k * o1..(k + 1) * o1];
        let ck = t.cell[2][k];
        plane.iter_mut().for_each(|x| *x = 0.0);
        for (c, &bz) in wz.iter().enumerate() {
            let src = &lattice[(ck + c) * cx * cy..(ck + c + 1) * cx * cy];
            for (p, s) in plane.iter_mut().zip(src) {
                *p += bz * s;
            }
        }
        for j in 0..ny {
            let wy = &t.w[1][j * o1..(j + 1) * o1];
            let cj = t.cell[1][j];
            line.iter_mut().for_each(|x| *x = 0.0);
            for (b, &by) in wy.iter().enumerate() {
                let src = &plane[(cj + b) * cx..(cj + b + 1) * cx];
                for (l, s) in line.iter_mut().zip(src) {
                    *l += by * s;
                }
            }
            let row = &mut out[(j + k * ny) * nx..(j + k * ny + 1) * nx];
            for (i, r) in row.iter_mut().enumerate() {
                let wx = &t.w[0][i * o1..(i + 1) * o1];
                let ci = t.cell[0][i];
                *r = wx.iter().zip(&line[ci..ci + o1]).map(|(a, b)| a * b).sum();
            }
        }
    }
    out
}

/// Coefficient of variation (`σ/μ`) of `exp(a − b)` over the mask, in ITK's arithmetic: the
/// fields, `exp` and the running mean and variance are single precision, and the double
/// literals in ITK's update formulas make parts of each update double. It decides when a level
/// stops, so it follows ITK's rounding: in double precision the value comes out about 12 %
/// lower on in-vivo data (the single-precision running mean drifts).
fn convergence_measure(a: &[f64], b: &[f64], inc_idx: &[usize]) -> f32 {
    let (mut mu, mut sigma, mut n) = (0f32, 0f32, 0f32);
    for &i in inc_idx {
        let p = (a[i] as f32 - b[i] as f32).exp();
        n = (n as f64 + 1.0) as f32;
        if n > 1.0 {
            let d = p - mu;
            sigma = (sigma as f64 + (d * d) as f64 * (n as f64 - 1.0) / n as f64) as f32;
        }
        mu = (mu as f64 * (1.0 - 1.0 / n as f64) + (p / n) as f64) as f32;
    }
    let sigma = (sigma as f64 / (n as f64 - 1.0)).sqrt() as f32;
    sigma / mu
}

/// Twice as many spans per axis, the same spline: `2c − order` control points with the
/// B-spline two-scale relation `c'_k = 2^{−p} Σ_j C(p+1, k − 2j + p) c_j`
/// (ITK's `RefineControlPointLattice`; terms past the coarse lattice are dropped as there).
fn refine_lattice(lattice: &[f64], ncp: [usize; 3], order: usize) -> (Vec<f64>, [usize; 3]) {
    let binom = |n: usize, k: usize| -> f64 {
        (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64)
    };
    let scale = 0.5f64.powi(order as i32);
    let coeff = |off: usize, j: usize| -> f64 {
        let m = off as i64 - 2 * j as i64 + order as i64;
        if m < 0 || m > order as i64 + 1 { 0.0 } else { scale * binom(order + 1, m as usize) }
    };
    let mut cur = lattice.to_vec();
    let mut cdims = ncp;
    for axis in 0..3 {
        let n_old = cdims[axis];
        let n_new = 2 * n_old - order;
        let mut ndims = cdims;
        ndims[axis] = n_new;
        let mut next = vec![0.0f64; ndims[0] * ndims[1] * ndims[2]];
        let stride_old = [1, cdims[0], cdims[0] * cdims[1]];
        let stride_new = [1, ndims[0], ndims[0] * ndims[1]];
        for z in 0..ndims[2] {
            for y in 0..ndims[1] {
                for x in 0..ndims[0] {
                    let pos = [x, y, z];
                    let kk = pos[axis];
                    let (half, off) = (kk / 2, kk % 2);
                    let mut sum = 0.0;
                    for j in 0..=order {
                        if half + j >= n_old {
                            continue;
                        }
                        let mut p = pos;
                        p[axis] = half + j;
                        sum += coeff(off, j)
                            * cur[p[0] * stride_old[0] + p[1] * stride_old[1] + p[2] * stride_old[2]];
                    }
                    next[x * stride_new[0] + y * stride_new[1] + z * stride_new[2]] = sum;
                }
            }
        }
        cur = next;
        cdims = ndims;
    }
    (cur, cdims)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refinement_reproduces_the_spline() {
        for order in 1..=3 {
            let ncp = [order + 2, order + 1, order + 3];
            let n = ncp[0] * ncp[1] * ncp[2];
            let lat: Vec<f64> = (0..n).map(|i| ((i * 37 % 11) as f64 - 5.0) * 0.1).collect();
            let (fine, fncp) = refine_lattice(&lat, ncp, order);
            let dims = [17, 9, 13];
            let a = evaluate_lattice(&lat, dims, ncp, &axis_tables(dims, ncp, order));
            let b = evaluate_lattice(&fine, dims, fncp, &axis_tables(dims, fncp, order));
            let d = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
            assert!(d < 1e-12, "order {order}: refined spline differs by {d}");
        }
    }

    #[test]
    fn fit_interpolates_a_single_point() {
        // Lee et al.'s approximation is exact at a lone point (phi_c = w_c v / Σ w²), up to the
        // control points whose ω = w_c² ITK treats as zero (below 1.2e-8).
        let ncp = [4, 5, 4];
        let dims = [12, 10, 9];
        let t = axis_tables(dims, ncp, 3);
        for &idx in &[0usize, 7 + 3 * 12 + 4 * 120, 12 * 10 * 9 - 1] {
            let fit = fit_lattice(&[2.5], &[idx], dims, ncp, &t);
            let back = evaluate_lattice(&fit, dims, ncp, &t);
            assert!((back[idx] - 2.5).abs() < 1e-5, "voxel {idx}: {}", back[idx]);
        }
    }

    fn phantom(n: [usize; 3]) -> (Vec<f64>, Vec<f64>, Grid) {
        // Two tissue classes in a sphere, times a smooth multiplicative bias.
        let grid = Grid::new(n[0], n[1], n[2], 1.0, 1.0, 1.0);
        let mut img = vec![0.0; n[0] * n[1] * n[2]];
        let mut bias = vec![1.0; img.len()];
        for k in 0..n[2] {
            for j in 0..n[1] {
                for i in 0..n[0] {
                    let p = [i, j, k].map(|v| v as f64);
                    let c = n.map(|v| (v as f64 - 1.0) / 2.0);
                    let r = ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2)).sqrt();
                    let idx = i + j * n[0] + k * n[0] * n[1];
                    let b = (0.3 * (p[0] - c[0]) / n[0] as f64 + 0.2 * (p[2] - c[2]) / n[2] as f64).exp();
                    bias[idx] = b;
                    if r < 0.45 * n[0] as f64 {
                        let tissue = if (i / 3 + j / 3 + k / 3) % 2 == 0 { 100.0 } else { 160.0 };
                        img[idx] = tissue * b;
                    }
                }
            }
        }
        (img, bias, grid)
    }

    #[test]
    fn n4_reduces_a_smooth_bias() {
        let (img, _bias, grid) = phantom([24, 24, 24]);
        let params = N4Params { max_iterations: vec![20, 20], ..N4Params::default() };
        let out = n4_bias_correction(&img, None, &grid, &params, |_, _| {}).unwrap();
        // Within-class coefficient of variation drops.
        let cv = |v: &[f64], lo: f64, hi: f64| {
            let sel: Vec<f64> = img.iter().zip(v).filter(|(a, _)| **a > lo && **a < hi).map(|(_, b)| *b).collect();
            let m = sel.iter().sum::<f64>() / sel.len() as f64;
            (sel.iter().map(|x| (x - m).powi(2)).sum::<f64>() / sel.len() as f64).sqrt() / m
        };
        // Select the lower class by its uncorrected range (100·[0.83, 1.2]).
        let before = cv(&img, 70.0, 125.0);
        let after = cv(&out.corrected, 70.0, 125.0);
        assert!(after < 0.7 * before, "cv before {before}, after {after}");
        for (c, (i, b)) in out.corrected.iter().zip(img.iter().zip(&out.bias_field)) {
            assert!((c - i / b).abs() <= 1e-12 * i.abs().max(1.0));
            assert!(*b > 0.0);
        }
    }

    #[test]
    fn mask_keeps_outside_and_rescales_inside() {
        let (img, _bias, grid) = phantom([20, 20, 20]);
        let mask: Vec<u8> = img.iter().map(|&v| (v > 0.0) as u8).collect();
        let params = N4Params { max_iterations: vec![5], ..N4Params::default() };
        let out = n4_bias_correction(&img, Some(&mask), &grid, &params, |_, _| {}).unwrap();
        let (mut lo, mut hi, mut clo, mut chi) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
        for i in 0..img.len() {
            if mask[i] == 0 {
                assert_eq!(out.corrected[i], img[i]);
            } else {
                lo = lo.min(img[i]);
                hi = hi.max(img[i]);
                clo = clo.min(out.corrected[i]);
                chi = chi.max(out.corrected[i]);
            }
        }
        assert!((lo - clo).abs() < 1e-9 * hi && (hi - chi).abs() < 1e-9 * hi);
    }

    #[test]
    fn threshold_stops_each_level() {
        let (img, _bias, grid) = phantom([16, 16, 16]);
        let params = N4Params { convergence_threshold: 1.0, ..N4Params::default() };
        let last = std::cell::Cell::new(0);
        n4_bias_correction(&img, None, &grid, &params, |i, _| last.set(i)).unwrap();
        assert_eq!(last.get(), params.max_iterations.len());
    }

    #[test]
    fn refuses_unusable_input_with_an_error() {
        let (img, _bias, grid) = phantom([16, 16, 16]);
        let err = |p: N4Params| n4_bias_correction(&img, None, &grid, &p, |_, _| {}).unwrap_err().to_string();
        let d = N4Params::default;
        assert_eq!(err(N4Params { spline_order: 4, ..d() }), "N4 spline_order = 4 is out of range: must be 1, 2 or 3");
        assert_eq!(err(N4Params { max_iterations: vec![], ..d() }), "N4 max_iterations = [] is out of range: must be at least one fitting level, the first with at least one iteration");
        assert!(err(N4Params { max_iterations: vec![0, 50], ..d() }).starts_with("N4 max_iterations = [0, 50]"));
        assert!(n4_bias_correction(&img, None, &grid, &N4Params { max_iterations: vec![1, 0], ..d() }, |_, _| {}).is_ok());
        assert_eq!(err(N4Params { mesh: [1, 0, 1], ..d() }), "N4 mesh = [1, 0, 1] is out of range: must be >= 1 along each axis");
        assert_eq!(err(N4Params { histogram_bins: 1, ..d() }), "N4 histogram_bins = 1 is out of range: must be >= 2");
        assert_eq!(err(N4Params { shrink_factor: 0, ..d() }), "N4 shrink_factor = 0 is out of range: must be >= 1");
        assert!(err(N4Params { bias_field_fwhm: 0.0, ..d() }).starts_with("N4 bias_field_fwhm = 0"));
        assert!(err(N4Params { wiener_noise: f64::NAN, ..d() }).starts_with("N4 wiener_noise = NaN"));
        assert!(err(N4Params { convergence_threshold: f64::INFINITY, ..d() }).starts_with("N4 convergence_threshold = inf"));
        // The grid check names the axis, its size and the shrink factor: a 6-slice slab at 4.
        let slab = Grid::new(16, 16, 6, 1.0, 1.0, 1.0);
        let e = n4_bias_correction(&vec![1.0; 16 * 16 * 6], None, &slab, &d(), |_, _| {}).unwrap_err().to_string();
        assert!(e.contains("axis 2") && e.contains("has 6") && e.contains("shrink_factor = 4"), "{e}");
        assert_eq!(d().check_grid((16, 16, 6)).unwrap_err().to_string(), e);
        assert!(d().check_grid((8, 8, 8)).is_ok() && d().check_grid((8, 8, 7)).is_err());
        // Volumes that do not match the grid, and a mask that selects nothing.
        let e = n4_bias_correction(&img[1..], None, &grid, &d(), |_, _| {}).unwrap_err().to_string();
        assert!(e.starts_with("N4 image has 4095 voxels but the grid is 16x16x16"), "{e}");
        let e = n4_bias_correction(&img, Some(&[1u8; 10]), &grid, &d(), |_, _| {}).unwrap_err().to_string();
        assert!(e.starts_with("N4 mask has 10 voxels"), "{e}");
        let e = n4_bias_correction(&img, Some(&vec![0u8; img.len()]), &grid, &d(), |_, _| {}).unwrap_err().to_string();
        assert!(e.contains("mask selects no voxel"), "{e}");
        assert!(d().validate().is_ok());
    }

    #[test]
    fn shrink_offset_matches_itk() {
        // (n-1)/2 - s(m-1)/2 rounded half up: n=178, s=4 -> m=44, 2.5 -> 3; n=148 -> 1.5 -> 2.
        for (n, s, want) in [(178i64, 4i64, 3i64), (148, 4, 2), (32, 4, 2), (33, 4, 2), (10, 3, 2)] {
            let m = n / s;
            let k = (n - 1) - s * (m - 1);
            assert_eq!((k + 1).div_euclid(2), want, "n {n} s {s}");
        }
    }
}

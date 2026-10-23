//! V-SHARP background field removal
//!
//! Variable kernel SHARP uses multiple SMV kernel radii
//! to preserve more brain tissue at edges while still
//! removing background fields.
//!
//! Each voxel takes the SMV high-pass `φ − S_r∗φ` of the largest radius `r`
//! whose ball fits inside the mask. By default that high-passed field is returned
//! as is, which is what the authors' code, STI Suite's `V_SHARP`, does. With
//! [`VsharpParams::deconvolve`] on it is then deconvolved by the largest kernel,
//! `1/(1 − Ŝ)` with TSVD at `threshold` (as QSM.jl does). The deconvolution restores low
//! spatial frequencies; on real data those are dominated by residual background and
//! field-map error, so leaving it off gives a more reproducible local field at the cost
//! of attenuating genuinely low-frequency tissue contrast.
//!
//! ## Defaults
//!
//! The defaults are STI Suite 3.0's `V_SHARP` (Liu lab, the authors of the V-SHARP papers),
//! which is P-code and was probed black-box (ukb-qsmxt `docs/08_vsharp_sti.md`): crop to the
//! mask's bounding box + 12 voxels, linear resampling to a **1 mm** grid, balls `|x| ≤ r + 1/6`
//! (centre excluded) for r = 1…`smvsize` mm, the largest fitting ball's background per voxel,
//! **no deconvolution**, the background interpolated back; `smvsize` defaults to 12 mm. Here
//! that is `deconvolve: false`, an [`Isotropic`](VsharpGrid::Isotropic) 1 mm grid (same 12-voxel
//! crop margin, [`CROP_MARGIN`]), `max_radius` 12 and STI's radius schedule, 12 → 2 mm in 1 mm
//! steps ([`VsharpParams::radius_step`]). What remains different from STI: this module's ball
//! (voxel centres within r, centre included, where STI's is `|x| ≤ r + 1/6` without the
//! centre) and STI's mask back-interpolation. The papers and the code disagree on the deconvolution:
//! Li et al. 2011 (NeuroImage 55:1645, *Image Analysis* and *Results*) and Wu et al. 2012
//! (*Background phase removal*: "a deconvolution is carried out to restore the local phase",
//! with the kernel of the largest diameter used) both deconvolve the SMV-filtered phase,
//! without giving the kernel's truncation; STI Suite's `V_SHARP` does not (its impulse
//! response is exactly `δ − S₁₂`). The code is followed. On 150 UK Biobank subjects × 2
//! visits (ukb-qsmxt `docs/08_vsharp_sti.md`, PR #160) deconvolving lowered deep-grey-matter
//! test-retest ICC (0.922 → 0.915) and removed the detectable longitudinal change (SRM
//! putamen + caudate 0.161 → −0.019); on the noise-free CI phantom it is closer to the true
//! local field, since there the low frequencies are tissue. Measured with the pipeline's ROMEO
//! field and RTS against the previous defaults (deconvolution, native grid): qsm-forward
//! phantom χ XSIM 0.246 → 0.244, NRMSE 85.8 → 86.9 %, r 0.527 → 0.502; in-vivo 3T Philips scan,
//! agreement with the UK Biobank pipeline's χ (STI Suite V-SHARP + iLSQR) XSIM 0.626 → 0.690,
//! NRMSE 78.5 → 73.4 %. On these 1 mm isotropic inputs the 1 mm grid only adds the crop.
//!
//! ## Working grid
//!
//! The SMV balls live on a voxel grid. On strongly anisotropic voxels (e.g. 0.8 × 0.8 × 3 mm)
//! the native grid makes them crude in the coarse direction: a 1–2 mm ball is a single-slice
//! disk. [`VsharpParams::grid`] can instead estimate the background on an isotropic working
//! grid, at the finest input voxel size ([`VsharpGrid::IsotropicFinest`]) or at a fixed spacing
//! ([`VsharpGrid::Isotropic`]):
//!
//! 1. crop the field and mask to the mask's bounding box with a zero margin of
//!    [`CROP_MARGIN`] native voxels per side (this also fixes where the working grid sits
//!    relative to the anatomy, so it lands the same way on repeat scans);
//! 2. resample both trilinearly onto the isotropic grid (mask: interpolated value > 0.5), axes
//!    whose voxel size already equals the spacing are left as they are;
//! 3. run V-SHARP there (same balls, radii, edge test and deconvolution as on the native grid);
//! 4. interpolate the **background** (field − local field on the working grid) back to the native
//!    grid trilinearly, and return local = native field − background, on the voxels where the
//!    back-interpolated working-grid support is > 0.5.
//!
//! The local field thus keeps native resolution; only the smooth background estimate is made on
//! the working grid. The default is [`VsharpGrid::Isotropic`] at 1 mm (STI Suite's grid; see *Defaults*). [`VsharpGrid::IsotropicFinest`]
//! makes the balls spherical without coarsening any axis, at the cost of a working grid that
//! is larger by the anisotropy ratio. A fixed [`VsharpGrid::Isotropic`] spacing gives the same
//! working grid whatever the acquisition resolution, but coarsens axes finer than it,
//! including the mask edge the background is estimated from.
//!
//! Reference:
//! Wu, B., Li, W., Guidon, A., Liu, C. (2012).
//! "Whole brain susceptibility mapping using compressed sensing."
//! Magnetic Resonance in Medicine, 67(1):137-147. https://doi.org/10.1002/mrm.23000
//!
//! Implementation follows QSM.jl (https://github.com/kamesy/QSM.jl); defaults follow STI Suite
//! 3.0's `V_SHARP` (see *Defaults*).

use num_complex::Complex64;
use crate::error::InputError;
use crate::Grid;
use crate::fft::{fft3d, ifft3d};
use crate::kernels::smv::{smv_kernel, erode_mask_smv};

/// Zero margin (native voxels per side) around the mask's bounding box when V-SHARP runs on an
/// isotropic working grid. See the [module docs](self).
pub const CROP_MARGIN: usize = 12;

/// Smallest radius in effect in STI Suite's `V_SHARP`, in mm (it drops voxels only r = 1 fits):
/// the default of [`VsharpParams::min_radius`].
pub const STI_MIN_RADIUS_MM: f64 = 2.0;

/// Step between STI Suite `V_SHARP`'s radii, in mm: the default of [`VsharpParams::radius_step`].
pub const STI_RADIUS_STEP_MM: f64 = 1.0;

/// Working-grid spacing of STI Suite's `V_SHARP`, in mm: the default of [`VsharpParams::grid`].
pub const STI_GRID_MM: f64 = 1.0;

/// Grid V-SHARP estimates the background field on. See the [module docs](self#working-grid).
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum VsharpGrid {
    /// The input voxel grid (no resampling). The enum's `Default`; [`VsharpParams`] defaults
    /// to [`Isotropic`](Self::Isotropic) at [`STI_GRID_MM`].
    #[default]
    Native,
    /// An isotropic grid at the smallest input voxel dimension (e.g. 0.3 mm for
    /// 0.3 × 0.3 × 1 mm): no axis is coarsened. Isotropic input runs on the native grid.
    IsotropicFinest,
    /// An isotropic grid of `spacing_mm`, anchored at the mask's bounding box. The background
    /// is estimated there and interpolated back; the local field keeps native resolution.
    Isotropic {
        /// Working-grid voxel size in mm; must be finite and > 0.
        spacing_mm: f64,
    },
}

impl VsharpGrid {
    /// Voxels count as isotropic for [`IsotropicFinest`](Self::IsotropicFinest) when the
    /// largest voxel dimension exceeds the smallest by at most this fraction (1 %), so header
    /// rounding does not trigger a resample.
    pub const ANISOTROPY_TOLERANCE: f64 = 0.01;

    /// This choice resolved for `voxel_size`: [`IsotropicFinest`](Self::IsotropicFinest)
    /// becomes [`Isotropic`](Self::Isotropic) at the smallest voxel dimension, or
    /// [`Native`](Self::Native) when the voxels are isotropic. The others are returned as is.
    pub fn resolve(self, voxel_size: (f64, f64, f64)) -> Self {
        match self {
            VsharpGrid::IsotropicFinest => {
                let (a, b, c) = voxel_size;
                let (lo, hi) = (a.min(b).min(c), a.max(b).max(c));
                if hi > lo * (1.0 + Self::ANISOTROPY_TOLERANCE) {
                    VsharpGrid::Isotropic { spacing_mm: lo }
                } else {
                    VsharpGrid::Native
                }
            }
            g => g,
        }
    }
}

/// V-SHARP algorithm parameters
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct VsharpParams {
    /// Deconvolution (TSVD) threshold: k-space frequencies where |1 - S| < threshold are dropped
    /// rather than divided through. Higher = more regularisation, more robust to a noisy input field
    /// (default: 0.2). Only used when [`deconvolve`](Self::deconvolve) is true.
    pub threshold: f64,
    /// Maximum (starting) SMV kernel radius in mm (default: 12.0, STI Suite's `V_SHARP`
    /// `smvsize` default, also what its `QSM_GUI` and the χ-separation toolbox pass)
    pub max_radius: f64,
    /// Smallest SMV kernel radius in mm (default: 2.0, STI Suite's schedule; see
    /// [`radius_step`](Self::radius_step)). Voxels where no ball of at least this radius fits
    /// are dropped from the output mask.
    pub min_radius: f64,
    /// Step between successive radii in mm (default: 1.0). The radii are `max_radius`,
    /// `max_radius − radius_step`, … down to `min_radius`, largest first; each voxel takes the
    /// largest that fits.
    ///
    /// The default schedule, 12, 11, …, 2 mm, is STI Suite's `V_SHARP`: it computes balls for
    /// r = 1 … `smvsize` mm in 1 mm steps (ascending, larger radii overwriting, so each voxel
    /// ends with the largest ball that fits) and then drops the voxels where only r = 1 fits,
    /// so the radii in effect are 2 … 12 mm (black-box probe, ukb-qsmxt
    /// `docs/08_vsharp_sti.md` step 4). Before this parameter existed `min_radius` was also
    /// the step, which cannot express that schedule; the previous default, 12 → 1 mm in 1 mm
    /// steps, is `min_radius: 1.0, radius_step: 1.0`. STI's ball is `|x| ≤ r + 1/6` without its
    /// centre voxel; this module's is the voxel centres within `r`, centre included, so the
    /// kernels themselves still differ slightly from STI's.
    pub radius_step: f64,
    /// Deconvolve the SMV high-passed field by the largest kernel, `1/(1 − Ŝ)` with TSVD at
    /// [`threshold`](Self::threshold). Default false: return the high-passed field
    /// `φ − S_r∗φ` itself, each voxel with the largest radius whose ball fits in the mask, as
    /// the authors' code (STI Suite's `V_SHARP`) does. `true` is the papers' description
    /// (Li 2011, Wu 2012) as QSM.jl implements it. The deconvolution restores low spatial
    /// frequencies that `1 − Ŝ` attenuated; on real data those are dominated by residual
    /// background and field-map error, so `false` is the more reproducible choice there, while
    /// `true` recovers more low-frequency tissue contrast on a clean field (see the module
    /// docs, *Defaults*).
    pub deconvolve: bool,
    /// Grid the background is estimated on (default: [`VsharpGrid::Isotropic`] at 1 mm, as
    /// STI Suite's `V_SHARP` resamples to 1 mm before filtering). See
    /// [`VsharpParams::working_grid`] for the choice resolved on a given input.
    /// [`VsharpGrid::Native`] avoids the interpolation and does not coarsen sub-millimetre
    /// acquisitions' background estimate (on 1 mm isotropic data the two agree up to the crop).
    pub grid: VsharpGrid,
}

impl VsharpParams {
    /// Checks the parameters V-SHARP cannot run with: `min_radius` and `radius_step` must be
    /// finite and > 0 — at a step of 0 the radius list never ends — and an
    /// [`Isotropic`](VsharpGrid::Isotropic) working grid needs a spacing that is finite and > 0.
    pub fn validate(&self) -> Result<(), InputError> {
        if !(self.min_radius.is_finite() && self.min_radius > 0.0) {
            return Err(InputError::param("V-SHARP min_radius", self.min_radius, "finite and > 0 mm"));
        }
        if !(self.radius_step.is_finite() && self.radius_step > 0.0) {
            return Err(InputError::param("V-SHARP radius_step", self.radius_step, "finite and > 0 mm"));
        }
        if let VsharpGrid::Isotropic { spacing_mm } = self.grid {
            if !(spacing_mm.is_finite() && spacing_mm > 0.0) {
                return Err(InputError::param(
                    "V-SHARP isotropic working grid spacing_mm", spacing_mm, "finite and > 0 mm",
                ));
            }
        }
        Ok(())
    }

    /// The working grid V-SHARP will use on `grid`: [`Self::grid`] resolved by
    /// [`VsharpGrid::resolve`]; never [`VsharpGrid::IsotropicFinest`].
    pub fn working_grid(&self, grid: &Grid) -> VsharpGrid {
        self.grid.resolve(grid.voxel_size)
    }
}

impl Default for VsharpParams {
    fn default() -> Self {
        Self {
            // 0.2 (was 0.05): the deconvolution amplifies structured error in a noisy estimated field,
            // and real pipelines always feed V-SHARP an estimated (not ground-truth) field. A stronger
            // regularisation is robust to that; 0.05 under-regularises and only wins on a clean field
            // (which BFR is never run on in practice). See the PR for the benchmark.
            threshold: 0.2,
            max_radius: 12.0,
            min_radius: STI_MIN_RADIUS_MM,
            radius_step: STI_RADIUS_STEP_MM,
            deconvolve: false,
            grid: VsharpGrid::Isotropic { spacing_mm: STI_GRID_MM },
        }
    }
}

/// V-SHARP background field removal
///
/// Uses multiple SMV kernel radii, starting from largest and decreasing.
/// At each voxel, uses the largest radius whose ball fits inside the mask,
/// then (if `params.deconvolve`) deconvolves by the largest kernel.
///
/// # Arguments
/// * `field` - Unwrapped total field (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz), 1 = inside ROI
/// * `grid` - Volume dimensions and voxel sizes
/// * `params` - V-SHARP parameters (min/max radius in mm, deconvolution on/off and its
///   threshold, working grid)
/// * `progress` - Progress callback (radius_index, total_radii)
///
/// # Returns
/// (local_field, eroded_mask)
///
/// # Errors
/// If [`VsharpParams::validate`] rejects `params`.
pub fn vsharp(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    params: &VsharpParams,
    progress: impl FnMut(usize, usize),
) -> Result<(Vec<f64>, Vec<u8>), InputError> {
    params.validate()?;
    let radii = vsharp_radii(params);
    let deconvolve = params.deconvolve.then_some(params.threshold);
    Ok(match params.working_grid(grid) {
        VsharpGrid::Native => vsharp_with_radii(field, mask, grid, &radii, deconvolve, progress),
        VsharpGrid::Isotropic { spacing_mm } => {
            vsharp_isotropic(field, mask, grid, spacing_mm, &radii, deconvolve, progress)
        }
        VsharpGrid::IsotropicFinest => unreachable!("working_grid resolves IsotropicFinest"),
    })
}

/// V-SHARP with the background estimated on an isotropic grid of spacing `h` mm anchored at the
/// mask's bounding box (see the [module docs](self#working-grid)).
fn vsharp_isotropic(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    h: f64,
    radii: &[f64],
    deconvolve: Option<f64>,
    progress: impl FnMut(usize, usize),
) -> (Vec<f64>, Vec<u8>) {
    let (nx, ny, nz) = grid.dims;
    let vs = [grid.voxel_size.0, grid.voxel_size.1, grid.voxel_size.2];
    let mut local_out = vec![0.0; nx * ny * nz];
    let mut mask_out = vec![0u8; nx * ny * nz];
    let Some((lo, hi)) = bounding_box(mask, grid.dims) else { return (local_out, mask_out) };

    // 1. crop to the bounding box with a zero margin
    let pad = CROP_MARGIN;
    let np = [0, 1, 2].map(|d| hi[d] - lo[d] + 1 + 2 * pad);
    let mut p = vec![0.0; np[0] * np[1] * np[2]];
    let mut m = vec![0.0; p.len()];
    for k in lo[2]..=hi[2] {
        for j in lo[1]..=hi[1] {
            for i in lo[0]..=hi[0] {
                let src = i + j * nx + k * nx * ny;
                let dst = (i - lo[0] + pad) + (j - lo[1] + pad) * np[0] + (k - lo[2] + pad) * np[0] * np[1];
                p[dst] = field[src];
                m[dst] = if mask[src] != 0 { 1.0 } else { 0.0 };
            }
        }
    }

    // 2. resample onto the working grid W: sample q (1-based) of axis d sits at 1-based native
    // index q·h/vs[d]; an axis already at spacing h stays on its native lattice. W is zero-filled
    // to even dimensions.
    let ni = [0, 1, 2].map(|d| if vs[d] == h { np[d] } else { (np[d] as f64 * vs[d] / h).ceil() as usize });
    let fwd = [0, 1, 2].map(|d| {
        (1..=ni[d]).map(|q| if vs[d] == h { q as f64 } else { q as f64 * h / vs[d] }).collect::<Vec<f64>>()
    });
    let pi = trilinear(&p, np, &fwd);
    let mi = trilinear(&m, np, &fwd);
    let ne = ni.map(|n| n + n % 2);
    let mut pe = vec![0.0; ne[0] * ne[1] * ne[2]];
    let mut me = vec![0u8; pe.len()];
    for k in 0..ni[2] {
        for j in 0..ni[1] {
            for i in 0..ni[0] {
                let s = i + j * ni[0] + k * ni[0] * ni[1];
                let d = i + j * ne[0] + k * ne[0] * ne[1];
                pe[d] = pi[s];
                me[d] = u8::from(mi[s] > 0.5);
            }
        }
    }
    drop((pi, mi));

    // 3. V-SHARP on W
    let w_grid = Grid::new(ne[0], ne[1], ne[2], h, h, h);
    let (local_w, support_w) = vsharp_with_radii(&pe, &me, &w_grid, radii, deconvolve, progress);

    // 4. background (field − local) and support back to the native grid
    let mut bg_c = vec![0.0; ni[0] * ni[1] * ni[2]];
    let mut nm_c = vec![0.0; bg_c.len()];
    for k in 0..ni[2] {
        for j in 0..ni[1] {
            for i in 0..ni[0] {
                let d = i + j * ni[0] + k * ni[0] * ni[1];
                let s = i + j * ne[0] + k * ne[0] * ne[1];
                bg_c[d] = pe[s] - local_w[s];
                nm_c[d] = f64::from(support_w[s]);
            }
        }
    }
    let back = [0, 1, 2].map(|d| {
        (1..=np[d]).map(|a| if vs[d] == h { a as f64 } else { a as f64 * vs[d] / h }).collect::<Vec<f64>>()
    });
    let bg_n = trilinear(&bg_c, ni, &back);
    let nm_n = trilinear(&nm_c, ni, &back);
    for k in lo[2]..=hi[2] {
        for j in lo[1]..=hi[1] {
            for i in lo[0]..=hi[0] {
                let s = (i - lo[0] + pad) + (j - lo[1] + pad) * np[0] + (k - lo[2] + pad) * np[0] * np[1];
                if nm_n[s] > 0.5 {
                    let d = i + j * nx + k * nx * ny;
                    local_out[d] = p[s] - bg_n[s];
                    mask_out[d] = 1;
                }
            }
        }
    }
    (local_out, mask_out)
}

/// Inclusive bounding box of the nonzero mask voxels, or `None` for an empty mask.
fn bounding_box(mask: &[u8], dims: (usize, usize, usize)) -> Option<([usize; 3], [usize; 3])> {
    let (nx, ny, nz) = dims;
    let mut lo = [usize::MAX; 3];
    let mut hi = [0usize; 3];
    let mut any = false;
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                if mask[i + j * nx + k * nx * ny] != 0 {
                    any = true;
                    for (d, v) in [i, j, k].into_iter().enumerate() {
                        lo[d] = lo[d].min(v);
                        hi[d] = hi[d].max(v);
                    }
                }
            }
        }
    }
    any.then_some((lo, hi))
}

/// Separable trilinear interpolation (MATLAB `interp3(..., 'linear')`): `coords[d]` are 1-based
/// fractional source indices along axis `d`, one output sample per entry, output in the same
/// Fortran order. Samples outside `[1, n]` on any axis are 0.
fn trilinear(src: &[f64], n: [usize; 3], coords: &[Vec<f64>; 3]) -> Vec<f64> {
    let w = |d: usize| -> Vec<Option<(usize, f64)>> {
        coords[d]
            .iter()
            .map(|&u| {
                if !(1.0..=n[d] as f64).contains(&u) {
                    return None;
                }
                let t = u - 1.0;
                let i0 = (t.floor() as usize).min(n[d].saturating_sub(2));
                Some((i0, t - i0 as f64))
            })
            .collect()
    };
    let (wx, wy, wz) = (w(0), w(1), w(2));
    let (ox, oy) = (wx.len(), wy.len());
    let at = |i: usize, j: usize, k: usize| -> f64 {
        if i < n[0] && j < n[1] && k < n[2] { src[i + j * n[0] + k * n[0] * n[1]] } else { 0.0 }
    };
    let mut out = vec![0.0; ox * oy * wz.len()];
    for (c, wzk) in wz.iter().enumerate() {
        let Some((k0, fz)) = *wzk else { continue };
        for (b, wyj) in wy.iter().enumerate() {
            let Some((j0, fy)) = *wyj else { continue };
            for (a, wxi) in wx.iter().enumerate() {
                let Some((i0, fx)) = *wxi else { continue };
                let mut v = 0.0;
                for (dk, gz) in [(0, 1.0 - fz), (1, fz)] {
                    if gz == 0.0 { continue; }
                    for (dj, gy) in [(0, 1.0 - fy), (1, fy)] {
                        if gy == 0.0 { continue; }
                        for (di, gx) in [(0, 1.0 - fx), (1, fx)] {
                            if gx == 0.0 { continue; }
                            v += gx * gy * gz * at(i0 + di, j0 + dj, k0 + dk);
                        }
                    }
                }
                out[a + b * ox + c * ox * oy] = v;
            }
        }
    }
    out
}

/// Build the descending list of SMV kernel radii (mm) from the params.
///
/// Starts at `max_radius` (mm) and steps down by `radius_step` (mm) while the radius is at
/// least `min_radius` (with a tolerance of 1e-9 mm for the floating-point steps).
fn vsharp_radii(params: &VsharpParams) -> Vec<f64> {
    let mut radii = Vec::new();
    let step = params.radius_step;
    let mut k = 0usize;
    loop {
        let r = params.max_radius - k as f64 * step;
        if r < params.min_radius - 1e-9 {
            break;
        }
        radii.push(r);
        k += 1;
    }
    if radii.is_empty() {
        radii.push(params.max_radius);
    }
    radii
}

/// V-SHARP with an explicit list of SMV kernel radii (mm).
///
/// Internal entry point; the public [`vsharp`] wrapper derives the radii
/// from [`VsharpParams`]. `deconvolve` is the TSVD threshold of the final
/// deconvolution, or `None` to return the SMV high-passed field undeconvolved.
pub(crate) fn vsharp_with_radii(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    radii: &[f64],
    deconvolve: Option<f64>,
    mut progress: impl FnMut(usize, usize),
) -> (Vec<f64>, Vec<u8>) {
    let (nx, ny, nz) = grid.dims;

    if radii.is_empty() {
        return (vec![0.0; nx * ny * nz], mask.to_vec());
    }

    // If only one radius and deconvolving, this is regular SHARP
    if let (1, Some(threshold)) = (radii.len(), deconvolve) {
        progress(1, 1);
        return crate::bgremove::sharp::sharp_core(
            field, mask, grid, threshold, radii[0]
        );
    }

    let n_total = nx * ny * nz;
    let n_radii = radii.len();

    // Sort radii from largest to smallest
    let mut sorted_radii = radii.to_vec();
    sorted_radii.sort_by(|a, b| b.partial_cmp(a).unwrap());

    // FFT of field
    let mut field_complex: Vec<Complex64> = field.iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();
    fft3d(&mut field_complex, nx, ny, nz);
    let field_fft = field_complex.clone();

    // Track which voxels have been processed and final mask
    let mut processed = vec![false; n_total];
    let mut local_field = vec![0.0; n_total];
    let mut final_mask = vec![0u8; n_total];

    let delta = 1.0 - 1e-7_f64.sqrt();
    let mut inverse_kernel: Option<Vec<f64>> = None;

    for (idx, &radius) in sorted_radii.iter().enumerate() {
        // Report progress
        progress(idx + 1, n_radii);

        // Generate SMV kernel
        let s_kernel = smv_kernel(grid, radius);

        // FFT of SMV kernel
        let mut s_complex: Vec<Complex64> = s_kernel.iter()
            .map(|&x| Complex64::new(x, 0.0))
            .collect();
        fft3d(&mut s_complex, nx, ny, nz);
        let s_fft: Vec<f64> = s_complex.iter().map(|c| c.re).collect();

        // Store inverse of first (largest) kernel
        if let (0, Some(threshold)) = (idx, deconvolve) {
            inverse_kernel = Some(s_fft.iter().map(|&s| {
                let one_minus_s = 1.0 - s;
                if one_minus_s.abs() < threshold {
                    0.0
                } else {
                    1.0 / one_minus_s
                }
            }).collect());
        }

        // Erode mask for this radius
        let eroded = erode_mask_smv(mask, &s_fft, grid, delta);
        let current_mask: Vec<bool> = eroded.iter().map(|&m| m == 1).collect();

        // Apply high-pass filter
        let mut filtered = field_fft.clone();
        for i in 0..n_total {
            filtered[i] *= 1.0 - s_fft[i];
        }

        ifft3d(&mut filtered, nx, ny, nz);

        for i in 0..n_total {
            if current_mask[i] && !processed[i] {
                local_field[i] = filtered[i].re;
                processed[i] = true;
                final_mask[i] = 1;
            }
        }
    }

    // Deconvolution (only built when `deconvolve` is set)
    if let Some(inv_kernel) = inverse_kernel {
        let mut local_complex: Vec<Complex64> = local_field.iter()
            .map(|&x| Complex64::new(x, 0.0))
            .collect();

        fft3d(&mut local_complex, nx, ny, nz);

        for i in 0..n_total {
            local_complex[i] *= inv_kernel[i];
        }

        ifft3d(&mut local_complex, nx, ny, nz);

        for i in 0..n_total {
            local_field[i] = if final_mask[i] == 1 { local_complex[i].re } else { 0.0 };
        }
    }

    (local_field, final_mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vsharp_zero_field() {
        let n = 8;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        let radii = vec![4.0, 3.0, 2.0];
        let (local, _) = vsharp_with_radii(&field, &mask, &grid, &radii, Some(0.05), |_, _| {});

        for &val in local.iter() {
            assert!(val.abs() < 1e-10);
        }
    }

    #[test]
    fn test_vsharp_preserves_more_than_sharp() {
        let n = 16;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        // V-SHARP with multiple radii
        let radii = vec![5.0, 4.0, 3.0, 2.0];
        let (_, vsharp_mask) = vsharp_with_radii(&field, &mask, &grid, &radii, Some(0.05), |_, _| {});

        // SHARP with single large radius
        let (_, sharp_mask) = crate::bgremove::sharp::sharp_core(
            &field, &mask, &grid, 0.05, 5.0
        );

        let vsharp_count: usize = vsharp_mask.iter().map(|&m| m as usize).sum();
        let sharp_count: usize = sharp_mask.iter().map(|&m| m as usize).sum();

        // V-SHARP should preserve at least as many voxels as SHARP
        assert!(vsharp_count >= sharp_count,
            "V-SHARP {} should preserve at least as many as SHARP {}",
            vsharp_count, sharp_count);
    }

    #[test]
    fn test_vsharp_nonuniform_voxels() {
        let n = 8;
        let field: Vec<f64> = (0..n*n*n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 0.5, 1.0, 2.0);

        // Anisotropic voxel sizes
        let radii = vec![4.0, 3.0, 2.0];
        let (local, final_mask) = vsharp_with_radii(
            &field, &mask, &grid, &radii, Some(0.05), |_, _| {}
        );

        // All values should be finite
        for (i, &val) in local.iter().enumerate() {
            assert!(val.is_finite(), "V-SHARP nonuniform voxels: finite at index {}", i);
        }

        // Final mask should have some voxels
        let mask_count: usize = final_mask.iter().map(|&m| m as usize).sum();
        assert!(mask_count > 0, "V-SHARP nonuniform: final mask should have some voxels");
    }

    #[test]
    fn test_vsharp_single_radius() {
        let n = 8;
        let field: Vec<f64> = (0..n*n*n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        // Single radius should delegate to SHARP
        let radii = vec![3.0];
        let (local, final_mask) = vsharp_with_radii(
            &field, &mask, &grid, &radii, Some(0.05), |_, _| {}
        );

        // All values should be finite
        for (i, &val) in local.iter().enumerate() {
            assert!(val.is_finite(), "V-SHARP single radius: finite at index {}", i);
        }

        // Result should match SHARP with same radius
        let (sharp_local, sharp_mask) = crate::bgremove::sharp::sharp_core(
            &field, &mask, &grid, 0.05, 3.0
        );

        for i in 0..n*n*n {
            assert!(
                (local[i] - sharp_local[i]).abs() < 1e-10,
                "Single-radius V-SHARP should match SHARP at index {}", i
            );
        }

        assert_eq!(final_mask, sharp_mask, "Single-radius V-SHARP mask should match SHARP mask");
    }

    #[test]
    fn test_vsharp_empty_radii() {
        // Empty radii should return zeros and the original mask
        let n = 8;
        let field: Vec<f64> = (0..n * n * n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        let (local, returned_mask) = vsharp_with_radii(
            &field, &mask, &grid, &[], Some(0.05), |_, _| {}
        );

        for &val in &local {
            assert_eq!(val, 0.0, "Empty radii should return zero local field");
        }
        assert_eq!(returned_mask, mask, "Empty radii should return original mask");
    }

    #[test]
    fn test_vsharp_larger_volume() {
        // 16x16x16 volume with spherical mask
        let n = 16;
        let mut field = vec![0.0; n * n * n];
        // Linear background field in z
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    field[x + y * n + z * n * n] = (z as f64) * 0.1;
                }
            }
        }

        // Spherical mask
        let mut mask = vec![0u8; n * n * n];
        let center = n / 2;
        let radius = n / 3;
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let dx = (x as i32) - (center as i32);
                    let dy = (y as i32) - (center as i32);
                    let dz = (z as i32) - (center as i32);
                    if dx * dx + dy * dy + dz * dz <= (radius * radius) as i32 {
                        mask[x + y * n + z * n * n] = 1;
                    }
                }
            }
        }

        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let radii = vec![6.0, 4.0, 3.0, 2.0];
        let (local, final_mask) = vsharp_with_radii(
            &field, &mask, &grid, &radii, Some(0.05), |_, _| {}
        );

        assert_eq!(local.len(), n * n * n);
        for &val in &local {
            assert!(val.is_finite(), "V-SHARP larger volume values should be finite");
        }

        let mask_count: usize = final_mask.iter().map(|&m| m as usize).sum();
        assert!(mask_count > 0, "V-SHARP larger volume should have voxels in final mask");

        // Voxels outside the final mask should be zero
        for i in 0..n * n * n {
            if final_mask[i] == 0 {
                assert_eq!(local[i], 0.0, "Outside final mask should be zero");
            }
        }
    }

    #[test]
    fn test_vsharp_with_progress() {
        let n = 8;
        let field: Vec<f64> = (0..n * n * n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        let radii = vec![4.0, 3.0, 2.0];
        let mut progress_calls = Vec::new();
        let (local, _) = vsharp_with_radii(
            &field, &mask, &grid, &radii, Some(0.05),
            |idx, total| { progress_calls.push((idx, total)); }
        );

        assert_eq!(local.len(), n * n * n);
        assert!(!progress_calls.is_empty(), "Progress should be called");
        for &val in &local {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_vsharp_with_progress_single_radius() {
        let n = 8;
        let field: Vec<f64> = (0..n * n * n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        let radii = vec![3.0];
        let mut progress_calls = Vec::new();
        let (local, _) = vsharp_with_radii(
            &field, &mask, &grid, &radii, Some(0.05),
            |idx, total| { progress_calls.push((idx, total)); }
        );

        assert_eq!(local.len(), n * n * n);
        assert!(!progress_calls.is_empty(), "Progress should be called for single radius");
        for &val in &local {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_vsharp_with_progress_empty_radii() {
        let n = 8;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        let mut progress_calls = Vec::new();
        let (local, returned_mask) = vsharp_with_radii(
            &field, &mask, &grid, &[], Some(0.05),
            |idx, total| { progress_calls.push((idx, total)); }
        );

        for &val in &local {
            assert_eq!(val, 0.0);
        }
        assert_eq!(returned_mask, mask);
    }

    #[test]
    fn test_vsharp_unsorted_radii() {
        // Radii given in arbitrary order - should be sorted internally
        let n = 8;
        let field: Vec<f64> = (0..n * n * n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        let radii_sorted = vec![4.0, 3.0, 2.0];
        let radii_unsorted = vec![2.0, 4.0, 3.0];

        let (local_sorted, mask_sorted) = vsharp_with_radii(
            &field, &mask, &grid, &radii_sorted, Some(0.05), |_, _| {}
        );
        let (local_unsorted, mask_unsorted) = vsharp_with_radii(
            &field, &mask, &grid, &radii_unsorted, Some(0.05), |_, _| {}
        );

        // Results should be the same regardless of input order
        assert_eq!(mask_sorted, mask_unsorted, "Sorted and unsorted radii should give same mask");
        for i in 0..n * n * n {
            assert!(
                (local_sorted[i] - local_unsorted[i]).abs() < 1e-10,
                "Results should match at index {}",
                i
            );
        }
    }

    /// Spherical mask of radius `rad` voxels centred in an `n`³ volume, and a smooth
    /// non-harmonic field over it.
    fn sphere_case(n: usize, rad: f64) -> (Vec<f64>, Vec<u8>) {
        let c = n as f64 / 2.0;
        let mut field = vec![0.0; n * n * n];
        let mut mask = vec![0u8; n * n * n];
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let (dx, dy, dz) = (x as f64 - c, y as f64 - c, z as f64 - c);
                    let i = x + y * n + z * n * n;
                    field[i] = 0.01 * dz + 0.002 * (dx * dx - dy * dy) + (0.3 * dx).sin() * 0.05;
                    mask[i] = (dx * dx + dy * dy + dz * dz <= rad * rad) as u8;
                }
            }
        }
        (field, mask)
    }

    #[test]
    fn test_vsharp_defaults_are_sti_suites() {
        // STI Suite's V_SHARP: no deconvolution, 1 mm working grid, smvsize 12.
        let p = VsharpParams::default();
        assert!(!p.deconvolve);
        assert_eq!(p.grid, VsharpGrid::Isotropic { spacing_mm: STI_GRID_MM });
        assert_eq!(p.max_radius, 12.0);
        assert_eq!((p.min_radius, p.radius_step), (2.0, 1.0));
    }

    #[test]
    fn test_vsharp_no_deconvolve_is_smv_highpass() {
        // Without deconvolution each voxel is φ − S_r∗φ for the largest radius r whose
        // ball fits the mask; check against that computed directly, radius by radius.
        let n = 20;
        let (field, mask) = sphere_case(n, 8.0);
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let radii = [4.0, 3.0, 2.0];
        let (local, out_mask) = vsharp_with_radii(&field, &mask, &grid, &radii, None, |_, _| {});

        let delta = 1.0 - 1e-7_f64.sqrt();
        let mut expected = vec![0.0; n * n * n];
        let mut expected_mask = vec![0u8; n * n * n];
        for &r in &radii {
            let mut s: Vec<Complex64> = smv_kernel(&grid, r).iter().map(|&x| Complex64::new(x, 0.0)).collect();
            fft3d(&mut s, n, n, n);
            let s_fft: Vec<f64> = s.iter().map(|c| c.re).collect();
            let fits = erode_mask_smv(&mask, &s_fft, &grid, delta);
            let mut sf: Vec<Complex64> = field.iter().map(|&x| Complex64::new(x, 0.0)).collect();
            fft3d(&mut sf, n, n, n);
            for i in 0..sf.len() { sf[i] *= s_fft[i]; }
            ifft3d(&mut sf, n, n, n);
            for i in 0..n * n * n {
                if fits[i] == 1 && expected_mask[i] == 0 {
                    expected[i] = field[i] - sf[i].re;
                    expected_mask[i] = 1;
                }
            }
        }
        assert_eq!(out_mask, expected_mask);
        assert!(out_mask.contains(&1));
        for i in 0..n * n * n {
            assert!((local[i] - expected[i]).abs() < 1e-12, "voxel {i}: {} vs {}", local[i], expected[i]);
        }
    }

    #[test]
    fn test_vsharp_deconvolve_flag() {
        let n = 20;
        let (field, mask) = sphere_case(n, 8.0);
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let base = VsharpParams {
            max_radius: 4.0, min_radius: 1.0, deconvolve: true, grid: VsharpGrid::Native, ..Default::default()
        };
        let (dec, dec_mask) = vsharp(&field, &mask, &grid, &base, |_, _| {}).unwrap();
        let off = VsharpParams { deconvolve: false, ..base.clone() };
        let (hp, hp_mask) = vsharp(&field, &mask, &grid, &off, |_, _| {}).unwrap();
        // Same voxels either way; the deconvolution changes the values.
        assert_eq!(dec_mask, hp_mask);
        assert!(dec.iter().zip(&hp).any(|(a, b)| (a - b).abs() > 1e-6));
        // The threshold only applies to the deconvolution.
        let (hp2, _) = vsharp(&field, &mask, &grid, &VsharpParams { threshold: 0.9, ..off }, |_, _| {}).unwrap();
        assert_eq!(hp, hp2);
        for i in 0..n * n * n {
            if hp_mask[i] == 0 { assert_eq!(hp[i], 0.0); }
        }
    }

    #[test]
    fn test_vsharp_no_deconvolve_single_radius() {
        // One radius without deconvolution is the plain SMV high-pass, not SHARP.
        let n = 20;
        let (field, mask) = sphere_case(n, 8.0);
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let (hp, hp_mask) = vsharp_with_radii(&field, &mask, &grid, &[3.0], None, |_, _| {});
        let (sh, sh_mask) = crate::bgremove::sharp::sharp_core(&field, &mask, &grid, 0.05, 3.0);
        assert_eq!(hp_mask, sh_mask);
        assert!(hp.iter().zip(&sh).any(|(a, b)| (a - b).abs() > 1e-6));
    }

    #[test]
    fn test_grid_resolve() {
        let finest = VsharpGrid::IsotropicFinest;
        assert_eq!(finest.resolve((1.0, 1.0, 1.0)), VsharpGrid::Native);
        assert_eq!(finest.resolve((0.5, 0.5, 0.5)), VsharpGrid::Native);
        // header rounding is not anisotropy
        assert_eq!(finest.resolve((1.0, 1.0, 1.005)), VsharpGrid::Native);
        assert_eq!(finest.resolve((0.8, 0.8, 3.0)), VsharpGrid::Isotropic { spacing_mm: 0.8 });
        assert_eq!(finest.resolve((3.0, 0.8, 0.8)), VsharpGrid::Isotropic { spacing_mm: 0.8 });
        assert_eq!(finest.resolve((0.3, 0.3, 0.5)), VsharpGrid::Isotropic { spacing_mm: 0.3 });
        let iso1 = VsharpGrid::Isotropic { spacing_mm: 1.0 };
        assert_eq!(iso1.resolve((0.3, 0.3, 0.5)), iso1);
        assert_eq!(VsharpGrid::Native.resolve((0.8, 0.8, 3.0)), VsharpGrid::Native);
        // the default is STI's 1 mm grid, whatever the voxel size
        assert_eq!(VsharpParams::default().working_grid(&Grid::new(4, 4, 4, 0.8, 0.8, 3.0)), iso1);
        assert_eq!(VsharpParams::default().working_grid(&Grid::new(4, 4, 4, 1.0, 1.0, 1.0)), iso1);
    }

    /// Ellipsoidal mask on a `dims` grid of voxel size `vs` (semi-axes in mm) and a smooth
    /// non-harmonic field with a background-like linear and quadratic part.
    fn ellipsoid_case(dims: (usize, usize, usize), vs: (f64, f64, f64), semi: [f64; 3]) -> (Vec<f64>, Vec<u8>) {
        let (nx, ny, nz) = dims;
        let c = [nx as f64 / 2.0, ny as f64 / 2.0, nz as f64 / 2.0];
        let mut field = vec![0.0; nx * ny * nz];
        let mut mask = vec![0u8; nx * ny * nz];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let (dx, dy, dz) = ((x as f64 - c[0]) * vs.0, (y as f64 - c[1]) * vs.1, (z as f64 - c[2]) * vs.2);
                    let i = x + y * nx + z * nx * ny;
                    field[i] = 0.01 * dz + 0.002 * (dx * dx - dy * dy) + 0.05 * (0.3 * dx).sin() * (0.2 * dz).cos();
                    let e = (dx / semi[0]).powi(2) + (dy / semi[1]).powi(2) + (dz / semi[2]).powi(2);
                    mask[i] = (e <= 1.0) as u8;
                }
            }
        }
        (field, mask)
    }

    #[test]
    fn test_native_grid_is_unchanged_v_sharp() {
        // `grid: Native` and `IsotropicFinest` on isotropic voxels are the plain native-grid
        // V-SHARP.
        let n = 20;
        let (field, mask) = sphere_case(n, 8.0);
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        for deconvolve in [true, false] {
            let base = VsharpParams { max_radius: 4.0, deconvolve, grid: VsharpGrid::Native, ..Default::default() };
            let radii = vsharp_radii(&base);
            let expected = vsharp_with_radii(&field, &mask, &grid, &radii, deconvolve.then_some(base.threshold), |_, _| {});
            assert_eq!(vsharp(&field, &mask, &grid, &base, |_, _| {}).unwrap(), expected);
            let finest = VsharpParams { grid: VsharpGrid::IsotropicFinest, ..base.clone() };
            assert_eq!(vsharp(&field, &mask, &grid, &finest, |_, _| {}).unwrap(), expected);
        }
        // anisotropic voxels, native grid
        let ag = Grid::new(24, 24, 10, 0.8, 0.8, 3.0);
        let (af, am) = ellipsoid_case(ag.dims, ag.voxel_size, [8.0, 8.0, 12.0]);
        let p = VsharpParams { max_radius: 4.0, deconvolve: true, grid: VsharpGrid::Native, ..Default::default() };
        let expected = vsharp_with_radii(&af, &am, &ag, &vsharp_radii(&p), Some(p.threshold), |_, _| {});
        assert_eq!(vsharp(&af, &am, &ag, &p, |_, _| {}).unwrap(), expected);
    }

    #[test]
    fn test_isotropic_working_grids() {
        let grid = Grid::new(30, 30, 12, 0.8, 0.8, 3.0);
        let (field, mask) = ellipsoid_case(grid.dims, grid.voxel_size, [10.0, 10.0, 14.0]);
        let native = VsharpParams { max_radius: 4.0, deconvolve: false, grid: VsharpGrid::Native, ..Default::default() };
        let (ln, mn) = vsharp(&field, &mask, &grid, &native, |_, _| {}).unwrap();
        let finest = VsharpParams { grid: VsharpGrid::IsotropicFinest, ..native.clone() };
        assert_eq!(finest.working_grid(&grid), VsharpGrid::Isotropic { spacing_mm: 0.8 });
        let at_finest = VsharpParams { grid: VsharpGrid::Isotropic { spacing_mm: 0.8 }, ..native.clone() };
        let iso1 = VsharpParams { grid: VsharpGrid::Isotropic { spacing_mm: 1.0 }, ..native.clone() };
        let (lf, mf) = vsharp(&field, &mask, &grid, &finest, |_, _| {}).unwrap();
        assert_eq!((lf.clone(), mf.clone()), vsharp(&field, &mask, &grid, &at_finest, |_, _| {}).unwrap());
        let (l1, m1) = vsharp(&field, &mask, &grid, &iso1, |_, _| {}).unwrap();
        for (l, m) in [(&lf, &mf), (&l1, &m1)] {
            assert!(m.contains(&1));
            assert!(*m != mn || l.iter().zip(&ln).any(|(a, b)| (a - b).abs() > 1e-6));
            for i in 0..l.len() {
                if m[i] == 0 { assert_eq!(l[i], 0.0); } else { assert!(mask[i] == 1 && l[i].is_finite()); }
            }
        }
    }

    #[test]
    fn test_isotropic_grid_empty_mask_and_bad_spacing() {
        let grid = Grid::new(8, 8, 4, 0.8, 0.8, 3.0);
        let iso1 = VsharpParams { grid: VsharpGrid::Isotropic { spacing_mm: 1.0 }, ..Default::default() };
        let (l, m) = vsharp(&vec![1.0; 256], &[0u8; 256], &grid, &iso1, |_, _| {}).unwrap();
        assert!(l.iter().all(|&v| v == 0.0) && m.iter().all(|&v| v == 0));
        let bad = VsharpParams { grid: VsharpGrid::Isotropic { spacing_mm: 0.0 }, ..Default::default() };
        let err = vsharp(&vec![1.0; 256], &[1u8; 256], &grid, &bad, |_, _| {}).unwrap_err();
        assert!(err.0.contains("spacing_mm = 0") && err.0.contains("> 0 mm"), "{err}");
    }

    /// #199: `min_radius` used to be the step between radii too, so at 0 (or below, or NaN) the
    /// radius list never ended and V-SHARP hung. It is an error now, as is such a `radius_step`.
    #[test]
    fn min_radius_that_is_not_positive_is_an_error() {
        let grid = Grid::new(4, 4, 4, 1.0, 1.0, 1.0);
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let p = VsharpParams { min_radius: bad, ..Default::default() };
            let err = vsharp(&[0.0; 64], &[1u8; 64], &grid, &p, |_, _| {}).unwrap_err();
            assert!(err.0.contains("V-SHARP min_radius") && err.0.contains("> 0 mm"), "{err}");
        }
    }

    #[test]
    fn radius_step_that_is_not_positive_is_an_error() {
        let grid = Grid::new(4, 4, 4, 1.0, 1.0, 1.0);
        for bad in [0.0, -1.0, f64::NAN] {
            let p = VsharpParams { radius_step: bad, ..Default::default() };
            let err = vsharp(&[0.0; 64], &[1u8; 64], &grid, &p, |_, _| {}).unwrap_err();
            assert!(err.0.contains("V-SHARP radius_step"), "{err}");
        }
    }

    #[test]
    fn radius_schedules() {
        // STI Suite's V_SHARP: 12 → 2 mm in 1 mm steps (r = 1 is dropped)
        let sti: Vec<f64> = (2..=12).rev().map(|r| r as f64).collect();
        assert_eq!(vsharp_radii(&VsharpParams::default()), sti);
        // the previous default, 12 → 1 mm
        let old = VsharpParams { min_radius: 1.0, radius_step: 1.0, ..Default::default() };
        assert_eq!(vsharp_radii(&old), (1..=12).rev().map(|r| r as f64).collect::<Vec<_>>());
        // a step independent of the smallest radius
        let p = VsharpParams { max_radius: 10.0, min_radius: 3.0, radius_step: 2.5, ..Default::default() };
        assert_eq!(vsharp_radii(&p), vec![10.0, 7.5, 5.0]);
        let p = VsharpParams { max_radius: 1.0, min_radius: 2.0, ..Default::default() };
        assert_eq!(vsharp_radii(&p), vec![1.0]);
    }

    /// The QSM.rs V-SHARP ablation prototype's isotropic-grid path (`exp/vsharp-ablation`,
    /// `vsharp.rs::ablation::run`, spec `d?g1b0r0p1e0`: 1 mm grid, main ball, main radii,
    /// crop to the mask bbox + 12 voxels, main edge test), kept here verbatim as the reference
    /// that was scored on UK Biobank (ukb-qsmxt docs/18).
    fn prototype_iso(field: &[f64], mask: &[u8], grid: &Grid, p: &VsharpParams) -> (Vec<f64>, Vec<u8>) {
        fn kernel_fft(n: [usize; 3], vs: [f64; 3], r: f64) -> (Vec<Complex64>, f64) {
            let mut k = vec![Complex64::new(0.0, 0.0); n[0] * n[1] * n[2]];
            let lim = r * r;
            let h = [0, 1, 2].map(|d| ((r / vs[d]).floor() as i64).min(n[d] as i64 / 2));
            let wrap = |d: i64, nd: usize| d.rem_euclid(nd as i64) as usize;
            let mut count = 0.0;
            for dz in -h[2]..=h[2] { for dy in -h[1]..=h[1] { for dx in -h[0]..=h[0] {
                let (x, y, z) = (dx as f64 * vs[0], dy as f64 * vs[1], dz as f64 * vs[2]);
                if x * x + y * y + z * z > lim { continue; }
                let idx = wrap(dx, n[0]) + wrap(dy, n[1]) * n[0] + wrap(dz, n[2]) * n[0] * n[1];
                if k[idx].re == 0.0 { k[idx] = Complex64::new(1.0, 0.0); count += 1.0; }
            }}}
            fft3d(&mut k, n[0], n[1], n[2]);
            (k, count)
        }
        let to_complex = |v: &[f64]| -> Vec<Complex64> { v.iter().map(|&x| Complex64::new(x, 0.0)).collect() };
        let (nx, ny, nz) = grid.dims;
        let vs = [grid.voxel_size.0, grid.voxel_size.1, grid.voxel_size.2];
        let mut local_out = vec![0.0; nx * ny * nz];
        let mut mask_out = vec![0u8; nx * ny * nz];
        let Some((lo, hi)) = bounding_box(mask, grid.dims) else { return (local_out, mask_out) };
        let pad = 12;
        let np = [0, 1, 2].map(|d| hi[d] - lo[d] + 1 + 2 * pad);
        let mut pp = vec![0.0; np[0] * np[1] * np[2]];
        let mut m = vec![0.0; pp.len()];
        for k in lo[2]..=hi[2] { for j in lo[1]..=hi[1] { for i in lo[0]..=hi[0] {
            let src = i + j * nx + k * nx * ny;
            let dst = (i - lo[0] + pad) + (j - lo[1] + pad) * np[0] + (k - lo[2] + pad) * np[0] * np[1];
            pp[dst] = field[src];
            m[dst] = if mask[src] != 0 { 1.0 } else { 0.0 };
        }}}
        let h = 1.0;
        let ni = [0, 1, 2].map(|d| if vs[d] == h { np[d] } else { (np[d] as f64 * vs[d] / h).ceil() as usize });
        let fwd = [0, 1, 2].map(|d| (1..=ni[d]).map(|q| if vs[d] == h { q as f64 } else { q as f64 * h / vs[d] }).collect::<Vec<f64>>());
        let pi = trilinear(&pp, np, &fwd);
        let mi = trilinear(&m, np, &fwd);
        let ne = ni.map(|n| n + n % 2);
        let mut pe = vec![0.0; ne[0] * ne[1] * ne[2]];
        let mut me = vec![0.0; pe.len()];
        for k in 0..ni[2] { for j in 0..ni[1] { for i in 0..ni[0] {
            let s = i + j * ni[0] + k * ni[0] * ni[1];
            let d = i + j * ne[0] + k * ne[0] * ne[1];
            pe[d] = pi[s];
            me[d] = if mi[s] > 0.5 { 1.0 } else { 0.0 };
        }}}
        let nw = ne[0] * ne[1] * ne[2];
        let mut radii = vsharp_radii(p);
        radii.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let mut p_fft = to_complex(&pe);
        fft3d(&mut p_fft, ne[0], ne[1], ne[2]);
        let mut m_fft = to_complex(&me);
        fft3d(&mut m_fft, ne[0], ne[1], ne[2]);
        let mut bg = pe.clone();
        let mut nm = vec![0.0; nw];
        let mut buf = vec![Complex64::new(0.0, 0.0); nw];
        let mut inv: Option<Vec<f64>> = None;
        let delta = 1.0 - 1e-7_f64.sqrt();
        for &r in &radii {
            let (mut k_fft, count) = kernel_fft(ne, [h; 3], r);
            if count == 0.0 { continue; }
            for i in 0..nw { buf[i] = m_fft[i] * k_fft[i]; }
            ifft3d(&mut buf, ne[0], ne[1], ne[2]);
            let fits: Vec<bool> = (0..nw).map(|i| me[i] > 0.0 && buf[i].re / count > delta).collect();
            for v in k_fft.iter_mut() { *v /= count; }
            if inv.is_none() {
                inv = Some(k_fft.iter().map(|s| { let oms = 1.0 - s.re; if oms.abs() < p.threshold { 0.0 } else { 1.0 / oms } }).collect());
            }
            for i in 0..nw { buf[i] = p_fft[i] * k_fft[i]; }
            ifft3d(&mut buf, ne[0], ne[1], ne[2]);
            for i in 0..nw { if fits[i] && nm[i] == 0.0 { bg[i] = buf[i].re; nm[i] = 1.0; } }
        }
        let mut l: Vec<Complex64> = (0..nw).map(|i| Complex64::new((pe[i] - bg[i]) * nm[i], 0.0)).collect();
        if p.deconvolve {
            if let Some(inv) = inv {
                fft3d(&mut l, ne[0], ne[1], ne[2]);
                for i in 0..nw { l[i] *= inv[i]; }
                ifft3d(&mut l, ne[0], ne[1], ne[2]);
                for i in 0..nw { l[i] = Complex64::new(l[i].re * nm[i], 0.0); }
            }
        }
        let mut bg_c = vec![0.0; ni[0] * ni[1] * ni[2]];
        let mut nm_c = vec![0.0; bg_c.len()];
        for k in 0..ni[2] { for j in 0..ni[1] { for i in 0..ni[0] {
            let d = i + j * ni[0] + k * ni[0] * ni[1];
            let s = i + j * ne[0] + k * ne[0] * ne[1];
            bg_c[d] = if p.deconvolve { pe[s] - l[s].re } else { bg[s] };
            nm_c[d] = nm[s];
        }}}
        let back = [0, 1, 2].map(|d| (1..=np[d]).map(|a| if vs[d] == h { a as f64 } else { a as f64 * vs[d] / h }).collect::<Vec<f64>>());
        let bg_n = trilinear(&bg_c, ni, &back);
        let nm_n = trilinear(&nm_c, ni, &back);
        for k in lo[2]..=hi[2] { for j in lo[1]..=hi[1] { for i in lo[0]..=hi[0] {
            let s = (i - lo[0] + pad) + (j - lo[1] + pad) * np[0] + (k - lo[2] + pad) * np[0] * np[1];
            if nm_n[s] > 0.5 {
                let d = i + j * nx + k * nx * ny;
                local_out[d] = pp[s] - bg_n[s];
                mask_out[d] = 1;
            }
        }}}
        (local_out, mask_out)
    }

    #[test]
    fn test_isotropic_grid_reproduces_the_prototype() {
        // UK Biobank-like voxels (0.8 × 0.8 × 3 mm); the mask touches neither the FOV edge nor
        // itself across the wrap.
        let grid = Grid::new(40, 36, 14, 0.8, 0.8, 3.0);
        let (field, mask) = ellipsoid_case(grid.dims, grid.voxel_size, [13.0, 11.0, 17.0]);
        for deconvolve in [false, true] {
            let p = VsharpParams {
                max_radius: 6.0,
                deconvolve,
                grid: VsharpGrid::Isotropic { spacing_mm: 1.0 },
                ..Default::default()
            };
            let (want, want_mask) = prototype_iso(&field, &mask, &grid, &p);
            let (got, got_mask) = vsharp(&field, &mask, &grid, &p, |_, _| {}).unwrap();
            assert_eq!(got_mask, want_mask, "support (deconvolve {deconvolve})");
            assert!(got_mask.contains(&1));
            let scale = want.iter().map(|v| v.abs()).fold(0.0, f64::max);
            let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(worst <= 1e-12 * scale.max(1.0), "deconvolve {deconvolve}: max |Δ| {worst} (scale {scale})");
        }
    }
}

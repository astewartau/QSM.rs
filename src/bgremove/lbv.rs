//! Laplacian Boundary Value (LBV) background field removal
//!
//! LBV removes the background field by solving the Laplace equation inside the mask: the local
//! field `fL` satisfies `∇²fL = ∇²fT` on the mask's interior and is 0 on its boundary layer, so
//! the background `fT − fL` is harmonic there and matches the total field `fT` on the boundary.
//!
//! # Port of `LBV.m` and `mexMGv6`
//! This module is a port of the authors' code: `functions/_LBV/LBV.m` and its C++ solver
//! `mexMGv6.cpp` (Dong Zhou, 2013) in the MEDI toolbox, release 2024.11.26. The solver is a full
//! multigrid (FMG) with Gauss–Seidel relaxation, ported step for step, including its quirks:
//!
//! - **Grids.** Level 0 is the input grid; level `d+1` has `(1 + n_d) / 2` voxels along each axis
//!   and samples the level-0 grid at `2^(d+1)` × the index (`MG::initializeMem`). Its mask
//!   samples the (peeled) level-0 mask the same way (`MG::initializeROI`), and every level's
//!   *interior* is its mask minus the voxels on the array edge and those with a 6-neighbour
//!   outside the mask (`MG::createInterior`). Only interior voxels are ever updated.
//! - **Depth.** `depth` is the coarsest level. Automatic (`depth` −1 in `LBV.m`, `None` here):
//!   `⌊log2 m⌋ − 3` for the smallest dimension `m`, at least 0 (`MG::determineDepth`). mexMGv6
//!   holds at most 10 levels, so `depth` ≤ 8.
//! - **Right-hand side.** Each level's source `ρ_d` is the 7-point Laplacian of the total field
//!   sampled onto that level, with the *finest* voxel size on every level
//!   (`MG::initializeRho`; the coarse grids are not rescaled by `2^d`).
//! - **Relaxation** (`MG::relax`): lexicographic Gauss–Seidel in place,
//!   `v ← (Σ_axes h⁻²(v₊ + v₋) − r) / (2 Σ h⁻²)`. On the coarsest level it ignores the sweep count
//!   and repeats until the summed absolute update over the interior is at most `tol`; the
//!   check runs before the first sweep with the update set to 1, so `tol` ≥ 1 skips the
//!   coarsest solve. `tol` is absolute, in the field's units: LBV.m works on MEDI's `iFreq`,
//!   radians accumulated over the echo spacing, so [`lbv`] converts the field from ppm to
//!   those units first ([`LbvParams::te`]) and back at the end.
//! - **FMG** (`MG::FMG`): solve the coarsest level, then on each finer level prolong the coarser
//!   solution (`MG::C2F`, no correction) and run one V-cycle on `L v = ρ_d`.
//! - **V-cycle** (`MG::V_cycle`): `N1` relaxations; the residual `f − L v` on the interior,
//!   injected onto the next level (`MG::updateRes`, `MG::F2C`: the even-indexed samples, no
//!   full weighting); zero the coarse solution; recurse; add the prolonged coarse correction
//!   scaled by `4^−(d+1)` for coarse level `d+1` (the absolute level, `MG::C2F`'s `prefac`);
//!   `N2` relaxations. On the coarser levels of a V-cycle the right-hand side is the residual
//!   buffer itself, which `updateRes` overwrites with the new residual before the recursion,
//!   so their post-relaxation runs on that new residual. Kept as in the original.
//! - **Prolongation** (`MG::C2F`): every coarse mask voxel is copied to its fine position, and
//!   the fine voxels between it and its +x/+y/+z neighbours (edges, face diagonals, the body
//!   diagonal) get the mean of the coarse voxels around them when those are all in the coarse
//!   mask. These writes reach the fine mask's boundary layer and a few voxels just outside the
//!   mask, which the relaxation never resets, so the returned field is non-zero there too
//!   (in silico 1723 boundary voxels, 1.1 % of the norm inside the mask; in vivo 1192, and 64
//!   outside the mask, 2.3 %). `LBV.m` returns them and MEDI's pipeline passes that field to
//!   `MEDI_L1` with the full mask, so they are kept; the returned mask is LBV.m's `interior`.
//! - **Finish** (`MG::postProcessing`): `N3` relaxations on level 0 (to convergence if
//!   `depth` is 0).
//! - **Peel** (`MG::preProcessing`): `peel` more layers are removed from the mask before the
//!   levels are built, so the interior is the mask minus `peel + 1` layers.
//!
//! The authors' defaults are `LbvParams::default()`: `tol` 0.01, automatic depth, `peel` 0 and
//! `N1`/`N2`/`N3` = 30/100/100 (`LBV.m:29-46`). MEDI's pipeline script has LBV as a commented
//! alternative to PDF (`README.m:129`) with `tol` 0.005; Ashley's decision (2026-10-30) was to
//! keep the function's own 0.01.
//!
//! ## Precision
//! mexMGv6 computes in single precision (`float`), with the prolongation's averages and the
//! coarse-grid correction evaluated in double and rounded (`MG::C2F`). [`LbvPrecision::Single`]
//! reproduces that arithmetic operation for operation. The default, [`LbvPrecision::Double`],
//! runs the same algorithm in double precision, which does no harm and avoids single precision's
//! rounding in the relaxation (QSM.rs keeps double where the reference uses single).
//!
//! ## Parity
//! [`LbvPrecision::Single`] against the compiled mexMGv6 (MEDI 2024.11.26's `mexMGv6.cpp` built
//! with R2023b's `mex`, as QSM-CI's verified `lbv-cornell` container runs it), on the same
//! `iFreq`: bit-identical in the local field and the interior, at LBV.m's defaults, on QSM-CI's
//! in-silico head (164×205×205, 7 T) and an EMTP travelling-head scan (184×256×128, 3 T), and on
//! `tests/data/lbv_cornell_small.bin` (`tests/lbv_cornell_parity.rs`; made with the container by
//! `tests/data/lbv_cornell_small.py`), which also covers `peel`, explicit depths, another `tol`
//! and other relaxation counts. The default, double precision, against the same mex: NRMSE
//! 0.00077 % in silico and 0.0011 % in vivo (max |Δ| 5.9e-6 and 4.4e-6 rad).
//!
//! Not ported: with `peel` > 0 mexMGv6 also writes the peeled mask to `mask_p<peel>.bin` in the
//! working directory (and crashes MATLAB where it cannot write there).
//!
//! ## The previous solver
//! Until astewartau/QSM.rs#406 this module solved the boundary value problem by successive
//! over-relaxation (ω 1.5, to a relative tolerance, at most `min(3·max dim, 500)` sweeps). That is
//! a different solver, and it stopped elsewhere: against mexMGv6 it measured 7.72 % NRMSE in
//! silico and 24.0 % on KKI (QSMxT/QSM-CI parity run 41384710780). It was not kept as an option,
//! because it was not better: against the exact solution of the discrete problem (conjugate
//! gradients to 1e-12), the multigrid at LBV.m's defaults is off by 2.4 % in silico and 8.5 % in
//! vivo (NRMSE over the interior), the SOR solver by 7.0 % and 16.7 %; against the in-silico
//! head's true local field (interior, mean removed) all three are at LBV's own limit (exact
//! solution 43.5 %, multigrid 43.4 %, SOR 42.4 %).
//!
//! Reference:
//! Zhou, D., Liu, T., Spincemaille, P., Wang, Y. (2014).
//! "Background field removal by solving the Laplacian boundary value problem."
//! NMR in Biomedicine, 27(3):312-319. https://doi.org/10.1002/nbm.3064
//!
//! Reference implementation: MEDI toolbox 2024.11.26, `functions/_LBV/LBV.m` and `mexMGv6.cpp`
//! (http://pre.weill.cornell.edu/mri/pages/qsm.html).

use crate::error::{check_len, InputError};
use crate::utils::echo_time::EchoTime;
use crate::Grid;

/// Arithmetic of the multigrid solver.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LbvPrecision {
    /// Double precision throughout (the default).
    #[default]
    Double,
    /// mexMGv6's arithmetic: single precision, with the prolongation's averages and corrections
    /// in double rounded to single. Reproduces the compiled mex bit for bit.
    Single,
}

/// LBV parameters (`LBV.m`'s defaults)
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct LbvParams {
    /// Stopping tolerance of the coarsest-grid relaxation (`tol`, default 0.01): the
    /// relaxation repeats while the summed absolute update over the coarsest interior exceeds
    /// it. Absolute, in MEDI's units: radians accumulated over the echo time [`te`](Self::te)
    /// picks. Must be finite and ≥ 0.
    pub tol: f64,
    /// Coarsest multigrid level (`depth`). `None` (LBV.m's −1, the default): `⌊log2 m⌋ − 3`
    /// for the smallest dimension `m`, at least 0. At most 8.
    pub depth: Option<usize>,
    /// Boundary layers removed beyond the one boundary layer (`peel`, default 0)
    pub peel: usize,
    /// Relaxations on each level before the recursive call (`N1`, default 30)
    pub n1: usize,
    /// Relaxations on each level after the recursive call (`N2`, default 100)
    pub n2: usize,
    /// Relaxations on the finest level after the full multigrid cycle (`N3`, default 100)
    pub n3: usize,
    /// Arithmetic (default [`LbvPrecision::Double`]; `Single` is mexMGv6's own)
    pub precision: LbvPrecision,
    /// Field strength in Tesla (scan parameter, for the radian conversion)
    pub b0: f64,
    /// Echo times in seconds (scan parameter, for the radian conversion)
    pub echo_times: Vec<f64>,
    /// Which echo time converts the field from ppm to radians. Default
    /// [`EchoTime::Spacing`], MEDI's `delta_TE` (`TE(2) − TE(1)`, or `TE` for one echo): LBV.m
    /// takes `iFreq` from `Fit_ppm_complex`, the phase accrued over the echo spacing, and its
    /// `tol` is absolute in those units.
    pub te: EchoTime,
}

impl Default for LbvParams {
    fn default() -> Self {
        Self {
            tol: 0.01,
            depth: None,
            peel: 0,
            n1: 30,
            n2: 100,
            n3: 100,
            precision: LbvPrecision::Double,
            b0: 3.0,
            echo_times: vec![0.005],
            te: EchoTime::Spacing,
        }
    }
}

impl LbvParams {
    /// Check the parameters (not the scan's echo times, which [`lbv`] resolves).
    pub fn validate(&self) -> Result<(), InputError> {
        if !(self.tol.is_finite() && self.tol >= 0.0) {
            return Err(InputError::param("LBV tol", self.tol, "finite and ≥ 0"));
        }
        if let Some(d) = self.depth {
            if d > MAX_DEPTH {
                return Err(InputError::param("LBV depth", d, "≤ 8 (mexMGv6 keeps 10 grids)"));
            }
        }
        if !(self.b0.is_finite() && self.b0 > 0.0) {
            return Err(InputError::param("LBV b0", self.b0, "finite and > 0 T"));
        }
        Ok(())
    }
}

/// mexMGv6's arrays hold 10 levels, and `initializeMem` writes the size of level `depth + 1`.
const MAX_DEPTH: usize = 8;

/// Guard against a coarsest-level relaxation that never meets `tol` (e.g. `tol` 0 in single
/// precision, where the update can cycle at the last bit). mexMGv6 has no limit; at its default
/// `tol` the coarsest level converges in far fewer sweeps.
const MAX_COARSEST_SWEEPS: usize = 100_000;

/// LBV background field removal (`LBV.m`)
///
/// # Arguments
/// * `field` - Total field in ppm (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz), non-zero = brain
/// * `grid` - Volume dimensions and voxel sizes
/// * `params` - LBV parameters, with the scan's field strength and echo times
/// * `progress` - Progress callback (step, steps): one step per multigrid level, and the finish
///
/// # Returns
/// `(local_field, interior)`: the local field in ppm as LBV.m returns it (defined on the
/// interior; see the module docs for the voxels around it), and LBV.m's `interior` (the mask
/// minus `peel + 1` layers).
pub fn lbv(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    params: &LbvParams,
    progress: impl FnMut(usize, usize),
) -> Result<(Vec<f64>, Vec<u8>), InputError> {
    params.validate()?;
    let n = grid.n_total();
    check_len("LBV field", field.len(), n)?;
    check_len("LBV mask", mask.len(), n)?;
    let delta_te = params
        .te
        .resolve(&params.echo_times)
        .map_err(|e| InputError::new(format!("LBV: {e}")))?;
    let cf = crate::constants::GAMMA_BAR_HZ_PER_T * params.b0;
    let pi = std::f64::consts::PI;

    // ppm -> radians over delta_TE, in the order QSM-CI's lbv-cornell writes it
    // (field * 1e-6 * 2*pi * CF * delta_TE), so both hand mexMGv6 the same numbers.
    let ifreq: Vec<f64> = field.iter().map(|&f| f * 1e-6 * 2.0 * pi * cf * delta_te).collect();
    let (fl, interior) = lbv_multigrid(&ifreq, mask, grid, params, progress);
    let back = 2.0 * pi * cf * delta_te;
    let local = fl.iter().map(|&v| v / back * 1e6).collect();
    Ok((local, interior))
}

/// mexMGv6 on a field already in the units `tol` is given in (MEDI's `iFreq`), as
/// `[fL, interior] = LBV(iFreq, Mask, matrix_size, voxel_size, tol, depth, peel, N1, N2, N3)`.
/// `params.b0`, `echo_times` and `te` are not used; the other parameters are not validated
/// (see [`LbvParams::validate`]).
pub fn lbv_multigrid(
    ifreq: &[f64],
    mask: &[u8],
    grid: &Grid,
    params: &LbvParams,
    progress: impl FnMut(usize, usize),
) -> (Vec<f64>, Vec<u8>) {
    match params.precision {
        LbvPrecision::Double => Mg::<f64>::run(ifreq, mask, grid, params, progress),
        LbvPrecision::Single => Mg::<f32>::run(ifreq, mask, grid, params, progress),
    }
}

/// The arithmetic mexMGv6's steps need, in `f32` (its own) or `f64`.
trait Real:
    Copy
    + PartialOrd
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
{
    const ZERO: Self;
    const ONE: Self;
    const TWO: Self;
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
    fn abs(self) -> Self;
}

impl Real for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    fn from_f64(x: f64) -> Self { x as f32 }
    fn to_f64(self) -> f64 { self as f64 }
    fn abs(self) -> Self { f32::abs(self) }
}

impl Real for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    fn from_f64(x: f64) -> Self { x }
    fn to_f64(self) -> f64 { self }
    fn abs(self) -> Self { f64::abs(self) }
}

/// Which buffer is the right-hand side of a relaxation or V-cycle on a level.
#[derive(Clone, Copy)]
enum Rhs {
    /// `ρ_d`, the Laplacian of the sampled total field (`rho[d]`)
    Rho,
    /// the residual injected from the finer level (`res[d]`)
    Res,
}

/// Level `d` of the hierarchy (`MG`'s per-depth arrays).
struct Level<T> {
    nx: usize,
    ny: usize,
    nz: usize,
    mask: Vec<bool>,
    /// interior voxel indices in increasing order (the `interior[d][i] == true` loops)
    interior: Vec<usize>,
    rho: Vec<T>,
    v: Vec<T>,
    res: Vec<T>,
}

impl<T> Level<T> {
    fn nxy(&self) -> usize {
        self.nx * self.ny
    }
}

struct Mg<T> {
    levels: Vec<Level<T>>,
    depth: usize,
    /// `aux3`: 1/dx², 1/dy², 1/dz² of the finest grid, used on every level
    aux3: [T; 3],
    /// `aux2`: 1 / (2 Σ aux3)
    aux2: T,
    tol: T,
    n1: usize,
    n2: usize,
}

/// `MG::createInterior`: the mask minus voxels on the array edge or with a 6-neighbour outside.
fn create_interior(mask: &[bool], nx: usize, ny: usize, nz: usize) -> Vec<bool> {
    let nxy = nx * ny;
    let mut interior = mask.to_vec();
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let i = x + y * nx + z * nxy;
                if !mask[i] {
                    continue;
                }
                let on_edge = x == 0 || y == 0 || z == 0 || x == nx - 1 || y == ny - 1 || z == nz - 1;
                if on_edge
                    || !mask[i - 1]
                    || !mask[i + 1]
                    || !mask[i - nx]
                    || !mask[i + nx]
                    || !mask[i - nxy]
                    || !mask[i + nxy]
                {
                    interior[i] = false;
                }
            }
        }
    }
    interior
}

/// `MG::determineDepth`: `⌊log2⌋` of the smallest dimension, minus 3, at least 0.
fn auto_depth(nx: usize, ny: usize, nz: usize) -> usize {
    let mut m = nx.min(ny).min(nz);
    let mut scale = 0usize;
    while m > 1 {
        m >>= 1;
        scale += 1;
    }
    scale.saturating_sub(3)
}

/// One value `MG::C2F` writes at fine index `t`: the mean `sum · w` of the coarse values around
/// it (w = 1/2, 1/4, 1/8), formed in double as the C++ does (`/2.0` is a double literal). As a
/// correction it is scaled by `prefac` and added to the fine value in double, then rounded.
#[inline]
fn c2f_put<T: Real>(out: &mut [T], t: usize, sum: T, w: f64, cor: bool, prefac: f64) {
    out[t] = if cor {
        T::from_f64(out[t].to_f64() + sum.to_f64() * w * prefac)
    } else {
        T::from_f64(sum.to_f64() * w)
    };
}

impl<T: Real> Mg<T> {
    fn run(
        ifreq: &[f64],
        mask_in: &[u8],
        grid: &Grid,
        params: &LbvParams,
        mut progress: impl FnMut(usize, usize),
    ) -> (Vec<f64>, Vec<u8>) {
        let (nx0, ny0, nz0) = grid.dims;
        let depth = params.depth.unwrap_or_else(|| auto_depth(nx0, ny0, nz0));
        let steps = depth + 2;

        // initializeConstants: aux3[i] = 1.0 / v / v in double from the voxel size in T,
        // stored in T; aux2 sums them in T, then 1/2.0/aux2 in double.
        let vs = [grid.voxel_size.0, grid.voxel_size.1, grid.voxel_size.2];
        let mut aux3 = [T::ZERO; 3];
        let mut aux2 = T::ZERO;
        for i in 0..3 {
            let v = T::from_f64(vs[i]).to_f64();
            aux3[i] = T::from_f64(1.0 / v / v);
            aux2 = aux2 + aux3[i];
        }
        let aux2 = T::from_f64(1.0 / 2.0 / aux2.to_f64());

        // preProcessing: the level-0 interior, then `peel` more layers off the mask.
        let mut mask0: Vec<bool> = mask_in.iter().map(|&m| m != 0).collect();
        let mut interior0 = create_interior(&mask0, nx0, ny0, nz0);
        for _ in 0..params.peel {
            mask0.clone_from(&interior0);
            interior0 = create_interior(&mask0, nx0, ny0, nz0);
        }

        // initializeMem + initializeROI + initializeRho, level by level.
        let fine_index = |x: usize, y: usize, z: usize, s: usize| x * s + y * s * nx0 + z * s * nx0 * ny0;
        let mut levels: Vec<Level<T>> = Vec::with_capacity(depth + 1);
        let (mut nx, mut ny, mut nz) = (nx0, ny0, nz0);
        for d in 0..=depth {
            let n = nx * ny * nz;
            let nxy = nx * ny;
            let s = 1usize << d;
            let (mask, interior) = if d == 0 {
                (mask0.clone(), interior0.clone())
            } else {
                // the coarse mask samples the peeled level-0 mask
                let mut m = vec![false; n];
                for z in 0..nz {
                    for y in 0..ny {
                        for x in 0..nx {
                            m[x + y * nx + z * nxy] = mask0[fine_index(x, y, z, s)];
                        }
                    }
                }
                let it = create_interior(&m, nx, ny, nz);
                (m, it)
            };
            let interior: Vec<usize> = (0..n).filter(|&i| interior[i]).collect();

            // ρ_d: the Laplacian of the total field sampled onto this level, with the finest
            // voxel size, in initializeRho's order of operations.
            let mut ti = vec![T::ZERO; n];
            let mut tj = vec![T::ZERO; n];
            let mut tk = vec![T::ZERO; n];
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let j = x + y * nx + z * nxy;
                        let t = T::from_f64(ifreq[fine_index(x, y, z, s)]);
                        ti[j] = t * aux3[0];
                        tj[j] = t * aux3[1];
                        tk[j] = t * aux3[2];
                    }
                }
            }
            let mut rho = vec![T::ZERO; n];
            for &j in &interior {
                rho[j] = tk[j + nxy] + tk[j - nxy] + tj[j + nx] + tj[j - nx] + ti[j + 1] + ti[j - 1]
                    - T::TWO * (ti[j] + tj[j] + tk[j]);
            }

            levels.push(Level { nx, ny, nz, mask, interior, rho, v: vec![T::ZERO; n], res: vec![T::ZERO; n] });
            // (1 + n) / 2
            nx = nx.div_ceil(2);
            ny = ny.div_ceil(2);
            nz = nz.div_ceil(2);
        }

        let mut mg = Mg { levels, depth, aux3, aux2, tol: T::from_f64(params.tol), n1: params.n1, n2: params.n2 };
        mg.fmg(0, &mut progress, steps);
        // postProcessing: N3 more relaxations on the finest level.
        mg.relax(0, Rhs::Rho, params.n3);
        progress(steps, steps);

        let l0 = &mg.levels[0];
        let fl = l0.v.iter().map(|v| v.to_f64()).collect();
        let mut interior = vec![0u8; l0.v.len()];
        for &i in &l0.interior {
            interior[i] = 1;
        }
        (fl, interior)
    }

    /// `MG::FMG`
    fn fmg(&mut self, d: usize, progress: &mut impl FnMut(usize, usize), steps: usize) {
        if d == self.depth {
            self.relax(d, Rhs::Rho, 0);
        } else {
            self.fmg(d + 1, progress, steps);
            self.c2f(d + 1, false);
            self.v_cycle(d, Rhs::Rho);
        }
        progress(self.depth - d + 1, steps);
    }

    /// `MG::V_cycle` on level `d` with right-hand side `f`
    fn v_cycle(&mut self, d: usize, f: Rhs) {
        self.relax(d, f, self.n1);
        if d != self.depth {
            self.update_res(d, f);
            self.f2c(d);
            self.levels[d + 1].v.fill(T::ZERO);
            self.v_cycle(d + 1, Rhs::Res);
            self.c2f(d + 1, true);
        }
        self.relax(d, f, self.n2);
    }

    /// `MG::relax`: lexicographic Gauss–Seidel on the interior, `n` sweeps, or on the coarsest
    /// level until the summed absolute update is at most `tol`.
    fn relax(&mut self, d: usize, f: Rhs, n: usize) {
        let aux3 = self.aux3;
        let aux2 = self.aux2;
        let coarsest = d == self.depth;
        let tol = self.tol;
        let lv = &mut self.levels[d];
        let (nx, nxy) = (lv.nx, lv.nx * lv.ny);
        let rr = match f {
            Rhs::Rho => &lv.rho,
            Rhs::Res => &lv.res,
        };
        let v = &mut lv.v;
        let interior = &lv.interior;
        let sweep = |v: &mut [T]| {
            for &i in interior {
                let x = aux3[0] * (v[i + 1] + v[i - 1]) + aux3[1] * (v[i + nx] + v[i - nx])
                    + aux3[2] * (v[i + nxy] + v[i - nxy])
                    - rr[i];
                v[i] = x * aux2;
            }
        };
        if coarsest {
            let mut old: Vec<T> = vec![T::ZERO; v.len()];
            let mut err = T::ONE;
            let mut sweeps = 0;
            while err > tol && sweeps < MAX_COARSEST_SWEEPS {
                old.copy_from_slice(v);
                sweep(v);
                // computeDiff(old, v): Σ |old − v| over the interior, in order
                err = T::ZERO;
                for &i in interior {
                    err = err + (old[i] - v[i]).abs();
                }
                sweeps += 1;
            }
            return;
        }
        for _ in 0..n {
            sweep(v);
        }
    }

    /// `MG::updateRes`: `res_d = f − L v` on the interior. With `f` = `res_d` this overwrites the
    /// right-hand side, as in the original.
    fn update_res(&mut self, d: usize, f: Rhs) {
        let aux3 = self.aux3;
        let lv = &mut self.levels[d];
        let (nx, nxy) = (lv.nx, lv.nx * lv.ny);
        for &i in &lv.interior {
            let u = &lv.v;
            let lap = (u[i + nxy] + u[i - nxy] - T::TWO * u[i]) * aux3[2]
                + (u[i + nx] + u[i - nx] - T::TWO * u[i]) * aux3[1]
                + (u[i + 1] + u[i - 1] - T::TWO * u[i]) * aux3[0];
            let fi = match f {
                Rhs::Rho => lv.rho[i],
                Rhs::Res => lv.res[i],
            };
            lv.res[i] = fi - lap;
        }
    }

    /// `MG::F2C`: inject the residual of level `d` onto level `d + 1` (the even-indexed voxels).
    fn f2c(&mut self, d: usize) {
        let (fine, coarse) = self.levels.split_at_mut(d + 1);
        let (fine, coarse) = (&fine[d], &mut coarse[0]);
        let (cnx, cnxy) = (coarse.nx, coarse.nxy());
        let (fnx, fnxy) = (fine.nx, fine.nxy());
        for z in (0..fine.nz).step_by(2) {
            for y in (0..fine.ny).step_by(2) {
                for x in (0..fine.nx).step_by(2) {
                    coarse.res[x / 2 + (y / 2) * cnx + (z / 2) * cnxy] = fine.res[x + y * fnx + z * fnxy];
                }
            }
        }
    }

    /// `MG::C2F`: prolong level `d` onto level `d − 1`, replacing (`cor` false) or adding the
    /// correction scaled by `prefac` = `4^−d` (`cor` true).
    fn c2f(&mut self, d: usize, cor: bool) {
        let (fine, coarse) = self.levels.split_at_mut(d);
        let (fine, coarse) = (&mut fine[d - 1], &coarse[0]);
        let s = (1u64 << d) as f64;
        let prefac = T::from_f64(1.0 / s / s);
        let pd = prefac.to_f64();
        let (nx, ny, nz, nxy) = (coarse.nx, coarse.ny, coarse.nz, coarse.nxy());
        let (fnx, fnxy) = (fine.nx, fine.nxy());
        let m = &coarse.mask;
        let c = &coarse.v;
        let out = &mut fine.v;
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let idx = i + j * nx + k * nxy;
                    if !m[idx] {
                        continue;
                    }
                    let f = 2 * (i + j * fnx + k * fnxy);
                    let (xl, yl, zl) = (i != nx - 1, j != ny - 1, k != nz - 1);
                    // the coincident voxel, in T (`vv[d-1][idx_] += vv[d][idx] * prefac`)
                    out[f] = if cor { out[f] + c[idx] * prefac } else { c[idx] };
                    if zl && m[idx + nxy] {
                        c2f_put(out, f + fnxy, c[idx] + c[idx + nxy], 0.5, cor, pd);
                    }
                    if yl && m[idx + nx] {
                        c2f_put(out, f + fnx, c[idx] + c[idx + nx], 0.5, cor, pd);
                    }
                    if xl && m[idx + 1] {
                        c2f_put(out, f + 1, c[idx] + c[idx + 1], 0.5, cor, pd);
                    }
                    if xl && yl && m[idx + 1] && m[idx + nx] && m[idx + nx + 1] {
                        let sum = c[idx] + c[idx + 1] + c[idx + 1 + nx] + c[idx + nx];
                        c2f_put(out, f + 1 + fnx, sum, 0.25, cor, pd);
                    }
                    if xl && zl && m[idx + 1] && m[idx + nxy] && m[idx + nxy + 1] {
                        let sum = c[idx] + c[idx + 1] + c[idx + 1 + nxy] + c[idx + nxy];
                        c2f_put(out, f + 1 + fnxy, sum, 0.25, cor, pd);
                    }
                    if yl && zl && m[idx + nx] && m[idx + nxy] && m[idx + nx + nxy] {
                        let sum = c[idx] + c[idx + nx] + c[idx + nx + nxy] + c[idx + nxy];
                        c2f_put(out, f + fnx + fnxy, sum, 0.25, cor, pd);
                    }
                    if xl && yl && zl
                        && m[idx + 1]
                        && m[idx + 1 + nx]
                        && m[idx + 1 + nxy]
                        && m[idx + 1 + nxy + nx]
                        && m[idx + nx]
                        && m[idx + nxy]
                        && m[idx + nx + nxy]
                    {
                        let sum = c[idx + 1] + c[idx + nxy] + c[idx] + c[idx + nx] + c[idx + nx + 1]
                            + c[idx + nx + nxy] + c[idx + nx + nxy + 1] + c[idx + nxy + 1];
                        c2f_put(out, f + 1 + fnx + fnxy, sum, 0.125, cor, pd);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ball of radius `r` voxels in an `n`³ grid.
    fn ball(n: usize, r: f64) -> Vec<u8> {
        let c = (n as f64 - 1.0) / 2.0;
        let mut m = vec![0u8; n * n * n];
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let d2 = (x as f64 - c).powi(2) + (y as f64 - c).powi(2) + (z as f64 - c).powi(2);
                    m[x + y * n + z * n * n] = (d2 <= r * r) as u8;
                }
            }
        }
        m
    }

    fn run(field: &[f64], mask: &[u8], grid: &Grid, p: &LbvParams) -> (Vec<f64>, Vec<u8>) {
        lbv(field, mask, grid, p, |_, _| {}).unwrap()
    }

    #[test]
    fn auto_depth_is_log2_of_the_smallest_dimension_minus_3() {
        assert_eq!(auto_depth(164, 205, 205), 4);
        assert_eq!(auto_depth(184, 256, 128), 4);
        assert_eq!(auto_depth(19, 23, 17), 1);
        assert_eq!(auto_depth(15, 64, 64), 0);
        assert_eq!(auto_depth(1, 1, 1), 0);
    }

    #[test]
    fn interior_is_the_mask_less_its_boundary_and_peel_layers() {
        let n = 16;
        let mask = ball(n, 6.0);
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let m: Vec<bool> = mask.iter().map(|&v| v != 0).collect();
        let once = create_interior(&m, n, n, n);
        let twice = create_interior(&once, n, n, n);
        let (_, it0) = run(&vec![0.0; n * n * n], &mask, &grid, &LbvParams::default());
        let (_, it1) = run(&vec![0.0; n * n * n], &mask, &grid, &LbvParams { peel: 1, ..Default::default() });
        assert!(it0.iter().zip(&once).all(|(&a, &b)| (a != 0) == b));
        assert!(it1.iter().zip(&twice).all(|(&a, &b)| (a != 0) == b));
        assert!(twice.iter().filter(|&&b| b).count() < once.iter().filter(|&&b| b).count());
    }

    #[test]
    fn zero_field_gives_zero() {
        let n = 16;
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let (local, _) = run(&vec![0.0; n * n * n], &ball(n, 6.0), &grid, &LbvParams::default());
        assert!(local.iter().all(|&v| v == 0.0));
    }

    /// A harmonic field is all background: LBV leaves (nearly) nothing on the interior, in
    /// both precisions, with anisotropic voxels. A local source inside is kept.
    #[test]
    fn removes_a_harmonic_field_and_keeps_a_local_one() {
        let n = 24;
        let grid = Grid::new(n, n, n, 0.8, 1.0, 1.3);
        let mask = ball(n, 9.0);
        let at = |i: usize| ((i % n) as f64 * 0.8, ((i / n) % n) as f64, (i / (n * n)) as f64 * 1.3);
        // x² − z² (harmonic) plus a linear term, in ppm
        let bg: Vec<f64> = (0..n * n * n)
            .map(|i| {
                let (x, _, z) = at(i);
                0.002 * (x * x - z * z) + 0.01 * z
            })
            .collect();
        let bump: Vec<f64> = (0..n * n * n)
            .map(|i| {
                let (x, y, z) = at(i);
                let d2 = (x - 9.6).powi(2) + (y - 12.0).powi(2) + (z - 15.0).powi(2);
                0.05 / (1.0 + d2 / 4.0)
            })
            .collect();
        for precision in [LbvPrecision::Double, LbvPrecision::Single] {
            let p = LbvParams { precision, b0: 3.0, echo_times: vec![0.004, 0.008], ..Default::default() };
            let (l_bg, it) = run(&bg, &mask, &grid, &p);
            let total: Vec<f64> = bg.iter().zip(&bump).map(|(a, b)| a + b).collect();
            let (l_tot, _) = run(&total, &mask, &grid, &p);
            let inside = |v: &[f64]| (0..v.len()).filter(|&i| it[i] != 0).map(|i| v[i].abs()).fold(0.0, f64::max);
            let bg_max = (0..bg.len()).filter(|&i| it[i] != 0).map(|i| bg[i].abs()).fold(0.0, f64::max);
            assert!(inside(&l_bg) < 0.01 * bg_max, "{precision:?}: harmonic residue {} of {bg_max}", inside(&l_bg));
            assert!(inside(&l_tot) > 0.02, "{precision:?}: the local bump is lost ({})", inside(&l_tot));
        }
    }

    /// Single and double precision run the same algorithm: they differ by rounding only.
    #[test]
    fn single_and_double_agree() {
        let n = 20;
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let mask = ball(n, 8.0);
        let field: Vec<f64> = (0..n * n * n).map(|i| ((i * 7919) % 101) as f64 * 1e-3 + (i % n) as f64 * 0.01).collect();
        let (d, _) = run(&field, &mask, &grid, &LbvParams::default());
        let (s, _) = run(&field, &mask, &grid, &LbvParams { precision: LbvPrecision::Single, ..Default::default() });
        let num: f64 = d.iter().zip(&s).map(|(a, b)| (a - b).powi(2)).sum();
        let den: f64 = d.iter().map(|a| a * a).sum();
        assert!(den > 0.0 && (num / den).sqrt() < 1e-4, "relative difference {}", (num / den).sqrt());
    }

    /// `tol` ≥ 1 skips the coarsest solve (its update starts at 1), as in mexMGv6; the finer
    /// levels still relax.
    #[test]
    fn tol_at_least_one_skips_the_coarsest_solve() {
        let n = 32;
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let mask = ball(n, 12.0);
        let field: Vec<f64> = (0..n * n * n).map(|i| ((i % n) as f64 - 16.0).powi(2) * 1e-3).collect();
        let p = |tol| LbvParams { tol, depth: Some(2), n1: 0, n2: 0, n3: 0, ..Default::default() };
        let (skipped, _) = run(&field, &mask, &grid, &p(1.0));
        assert!(skipped.iter().all(|&v| v == 0.0), "no relaxation anywhere, so nothing moves");
        let (solved, _) = run(&field, &mask, &grid, &p(0.01));
        assert!(solved.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn rejects_bad_parameters_and_inputs() {
        let n = 8;
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let f = vec![0.0; n * n * n];
        let m = vec![1u8; n * n * n];
        let bad = |p: LbvParams| lbv(&f, &m, &grid, &p, |_, _| {}).is_err();
        assert!(bad(LbvParams { tol: -1.0, ..Default::default() }));
        assert!(bad(LbvParams { tol: f64::NAN, ..Default::default() }));
        assert!(bad(LbvParams { depth: Some(9), ..Default::default() }));
        assert!(bad(LbvParams { b0: 0.0, ..Default::default() }));
        assert!(bad(LbvParams { echo_times: vec![], ..Default::default() }));
        assert!(lbv(&f[1..], &m, &grid, &LbvParams::default(), |_, _| {}).is_err());
    }

    #[test]
    fn reports_progress_to_the_end() {
        let n = 16;
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let mut calls = Vec::new();
        lbv(&vec![0.1; n * n * n], &ball(n, 6.0), &grid, &LbvParams::default(), |a, b| calls.push((a, b))).unwrap();
        assert_eq!(calls.last(), Some(&(3, 3)), "depth 1: the coarsest, level 0, the finish ({calls:?})");
    }
}

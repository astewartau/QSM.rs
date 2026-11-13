//! Multigrid-preconditioned conjugate gradients (MGPCG) for the Poisson equation on a mask
//!
//! A port of QSM.jl's `solve_poisson_mgpcg` (kamesy/QSM.jl fde4b92,
//! `src/utils/poisson_solver/{poisson_solver.jl,mgpcg.jl,multigrid/*.jl}`), with the
//! smoother schedule QSM.jl's `unwrap_laplacian(…; solver = :mgpcg)` passes it (the
//! `__DEFAULT_LBV_*` schedule, `src/bgremove/lbv.jl:2-19`). It solves `∇²u = f` on the voxels of
//! a mask under a homogeneous Dirichlet condition: voxels outside the mask are boundary points
//! fixed at 0, voxels inside are unknowns. Used by
//! [`LaplacianSolver::Mgpcg`](crate::unwrap::LaplacianSolver::Mgpcg).
//!
//! Ported step for step:
//!
//! - **Operator** (`Poisson2`): the 7-point Laplacian with `h⁻²` per axis, applied at the mask
//!   voxels that are not on the array edge (`R8`, the ranges `2:n-1`); the diagonal is
//!   `D = −2 Σ h⁻²`. The CG vector updates run over every mask voxel (`RI`), the operator only
//!   over `R8`, as in QSM.jl (the two differ only for mask voxels on the array edge, whose
//!   right-hand side the callers leave at 0).
//! - **Grids** (`restrict`): a coarse grid has `(n + 1) / 2` voxels per axis; the coarse voxel
//!   `c` (0-based, `1 ≤ c ≤ n_c − 2`) is in the coarse mask when all 8 fine voxels
//!   `2c..2c+1` are; its spacing is `h · (n − 1)/(n_c − 1)`. `depth` grids in all, the finest
//!   included.
//! - **Boundary layer** (`boundary_mask`): the mask voxels next to (7-point stencil) a voxel
//!   that is in the shell between the ranges `2:n-1` and `3:n-2-iseven(n)` (1-based) or in a
//!   2×2×2 block of the coarse-grid pattern that is not entirely in the mask. Smoothed extra.
//! - **Smoother**: red-black Gauss–Seidel, `x ← (b − Σ h⁻²(x₊ + x₋)) / D`. A forward sweep
//!   updates the voxels with odd 0-based `i + j + k` then the even ones; a backward sweep
//!   the reverse. Each colour depends only on the other, so the order within a colour does
//!   not matter (the result is deterministic, unlike QSM.jl's threaded reductions below).
//! - **V-cycle** (`cycle!`), on level `l` (0-based) of `depth`: presmooth with `2^l` forward
//!   sweeps on the mask then `2^(l+1)` forward sweeps on the boundary layer; residual
//!   `r = b − A x` on the mask; full-weighting restriction (27-point, weights 8/4/2/1 over
//!   64) to the coarse mask; coarse solution zeroed; recurse, or on the coarsest level 512
//!   forward then 512 backward sweeps; trilinear prolongation added to `x` on the mask;
//!   postsmooth with `2^(l+1)` backward sweeps on the boundary layer then `2^l` backward
//!   sweeps on the mask. One V-cycle per CG iteration.
//! - **CG** (`_mgpcg!`): preconditioned CG from `x₀ = 0`, stopping when `‖r‖ ≤ max(rtol ‖r₀‖,
//!   atol)` or after `maxit` iterations.
//! - **Crop** (`solve_poisson_mgpcg!`): the problem is solved on the mask's bounding box grown
//!   by one voxel (clamped to the array) when that removes more than 10 voxels along some
//!   axis; the grids are then built on the cropped box (which changes them, so this is kept).
//!
//! The defaults [`MgpcgParams::for_unwrap`] are QSM.jl's `unwrap_laplacian` ones
//! (`src/unwrap/laplacian.jl:65-85`): depth `max(2, 1 + ⌊log2(min n) − 2⌋)`
//! (`__DEFAULT_MGPCG_DEPTH`) of the **cropped** grid, `atol = rtol = √eps(Float64)` (1.49e-8),
//! `maxit` = the largest dimension of the **uncropped** mask. (`unwrap_laplacian` computes a
//! depth from the uncropped mask too, but only to size its smoother lists; `mgpcg!` is not
//! given it, so it takes its default from the cropped operator. The smoother at a level
//! depends only on the level, so the list's length changes nothing.)
//!
//! Differences from QSM.jl: double precision whatever the input (QSM.jl computes in the
//! input's precision; its callers pass Float64 or Float32), sums accumulated sequentially (QSM.jl
//! splits them over threads, so its own result depends on the thread count at the rounding
//! level), and `a·b + c` evaluated without fused multiply-add (QSM.jl's `muladd` may fuse). All
//! three change results only at rounding level.
//!
//! Reference implementation: <https://github.com/kamesy/QSM.jl>
//! (`src/utils/poisson_solver/`).

/// MGPCG parameters (QSM.jl's names).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MgpcgParams {
    /// Number of grids, the finest included (`maxlevel`); at least 2. `None`: QSM.jl's
    /// default, `max(2, 1 + ⌊log2(min n) − 2⌋)` for the (cropped) grid the problem is solved on.
    pub depth: Option<usize>,
    /// Absolute stopping tolerance on the residual norm (`atol`).
    pub atol: f64,
    /// Relative stopping tolerance on the residual norm (`rtol`).
    pub rtol: f64,
    /// Maximum number of CG iterations (`maxit`).
    pub maxit: usize,
}

impl MgpcgParams {
    /// QSM.jl's `unwrap_laplacian` settings for a mask of size `dims`: the default depth (of
    /// the cropped grid), `atol = rtol = √eps(Float64)`, `maxit = max dims`.
    pub fn for_unwrap(dims: (usize, usize, usize)) -> Self {
        let (nx, ny, nz) = dims;
        let tol = f64::EPSILON.sqrt();
        MgpcgParams { depth: None, atol: tol, rtol: tol, maxit: nx.max(ny).max(nz) }
    }
}

/// QSM.jl's `__DEFAULT_MGPCG_DEPTH`: `max(2, 1 + ⌊log2(min n) − 2⌋)`.
pub fn default_depth(n: [usize; 3]) -> usize {
    let m = n[0].min(n[1]).min(n[2]).max(1) as f64;
    (1 + (m.log2() - 2.0).floor() as i64).max(2) as usize
}

/// One grid of the hierarchy: QSM.jl's `Poisson2`.
struct Level {
    n: [usize; 3],
    dx: [f64; 3],
    idx2: [f64; 3],
    d: f64,
    id: f64,
    interior: Vec<bool>,
    boundary: Vec<bool>,
    /// Every interior voxel (`RI`), in memory order.
    ri: Vec<usize>,
}

impl Level {
    fn new(interior: Vec<bool>, n: [usize; 3], dx: [f64; 3]) -> Self {
        let idx2 = [1.0 / (dx[0] * dx[0]), 1.0 / (dx[1] * dx[1]), 1.0 / (dx[2] * dx[2])];
        let d = -2.0 * (idx2[0] + idx2[1] + idx2[2]);
        let boundary = boundary_mask(&interior, n);
        let ri = interior.iter().enumerate().filter(|(_, &m)| m).map(|(i, _)| i).collect();
        Level { n, dx, idx2, d, id: 1.0 / d, interior, boundary, ri }
    }

    /// Calls `f(idx, sx, sy, sz)` for every interior voxel in the ranges `1..=n-2` (`R8`).
    #[inline(always)]
    fn for_r8(&self, mask: &[bool], mut f: impl FnMut(usize, usize, usize, usize)) {
        let [nx, ny, nz] = self.n;
        if nx < 3 || ny < 3 || nz < 3 {
            return;
        }
        let (sy, sz) = (nx, nx * ny);
        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                let row = j * sy + k * sz;
                for i in 1..nx - 1 {
                    let idx = row + i;
                    if mask[idx] {
                        f(idx, 1, sy, sz);
                    }
                }
            }
        }
    }

    /// `out = A x` on `R8` (`mul!`); returns `Σ x·(A x)` there (`_compute_q!`'s `pAp`).
    fn apply(&self, out: &mut [f64], x: &[f64]) -> f64 {
        let [idx2, idy2, idz2] = self.idx2;
        let d = self.d;
        let mut dot = 0.0;
        self.for_r8(&self.interior, |c, sx, sy, sz| {
            let dz = x[c - sz] + x[c + sz];
            let dy = x[c - sy] + x[c + sy];
            let dxs = x[c - sx] + x[c + sx];
            let mut v = d * x[c];
            v += idz2 * dz;
            v += idy2 * dy;
            v += idx2 * dxs;
            out[c] = v;
            dot += x[c] * v;
        });
        dot
    }

    /// `r = b − A x` on `R8` (`residual!`).
    fn residual(&self, r: &mut [f64], b: &[f64], x: &[f64]) {
        let [idx2, idy2, idz2] = self.idx2;
        let d = self.d;
        self.for_r8(&self.interior, |c, sx, sy, sz| {
            let dz = x[c - sz] + x[c + sz];
            let dy = x[c - sy] + x[c + sy];
            let dxs = x[c - sx] + x[c + sx];
            let mut v = -d * x[c] + b[c];
            v += -idz2 * dz;
            v += -idy2 * dy;
            v += -idx2 * dxs;
            r[c] = v;
        });
    }

    /// Red-black Gauss–Seidel sweeps (`rbgs!`) over `R8` ∩ (`interior` or `boundary`), `iter`
    /// times. `forward`: QSM.jl's colours 1:2 (odd 0-based `i+j+k`, then even); backward 2:3.
    fn rbgs(&self, x: &mut [f64], b: &[f64], on_boundary: bool, forward: bool, iter: usize) {
        let mask = if on_boundary { &self.boundary } else { &self.interior };
        let id = self.id;
        let (cx, cy, cz) = (-id * self.idx2[0], -id * self.idx2[1], -id * self.idx2[2]);
        let [nx, ny, nz] = self.n;
        if nx < 3 || ny < 3 || nz < 3 {
            return;
        }
        let (sy, sz) = (nx, nx * ny);
        // QSM.jl colour s updates odd 0-based i+j+k for s = 1, 3 and even for s = 2.
        let colours: [usize; 2] = if forward { [1, 0] } else { [0, 1] };
        for _ in 0..iter {
            for &parity in &colours {
                for k in 1..nz - 1 {
                    for j in 1..ny - 1 {
                        let row = j * sy + k * sz;
                        // first i ≥ 1 with (i + j + k) % 2 == parity
                        let i0 = if (1 + j + k) % 2 == parity { 1 } else { 2 };
                        let mut i = i0;
                        while i < nx - 1 {
                            let c = row + i;
                            if mask[c] {
                                let dz = x[c - sz] + x[c + sz];
                                let dy = x[c - sy] + x[c + sy];
                                let dxs = x[c - 1] + x[c + 1];
                                let mut v = id * b[c];
                                v += cz * dz;
                                v += cy * dy;
                                v += cx * dxs;
                                x[c] = v;
                            }
                            i += 2;
                        }
                    }
                }
            }
        }
    }
}

#[inline]
fn restrict_len(n: usize) -> usize {
    (n + 1) >> 1
}

/// Coarse mask (`restrict(interior)`): coarse voxel `c` (`1..=n_c-2`) is in it when the 8
/// fine voxels `2c..2c+1` all are.
fn restrict_mask(m: &[bool], n: [usize; 3]) -> (Vec<bool>, [usize; 3]) {
    let nc = [restrict_len(n[0]), restrict_len(n[1]), restrict_len(n[2])];
    let mut mc = vec![false; nc[0] * nc[1] * nc[2]];
    let (sy, sz) = (n[0], n[0] * n[1]);
    for kc in 1..nc[2].saturating_sub(1) {
        for jc in 1..nc[1].saturating_sub(1) {
            for ic in 1..nc[0].saturating_sub(1) {
                let f = 2 * ic + 2 * jc * sy + 2 * kc * sz;
                mc[ic + jc * nc[0] + kc * nc[0] * nc[1]] = m[f]
                    && m[f + 1]
                    && m[f + sy]
                    && m[f + 1 + sy]
                    && m[f + sz]
                    && m[f + 1 + sz]
                    && m[f + sy + sz]
                    && m[f + 1 + sy + sz];
            }
        }
    }
    (mc, nc)
}

/// The boundary layer smoothed extra (`boundary_mask!`); see the module docs.
fn boundary_mask(m: &[bool], n: [usize; 3]) -> Vec<bool> {
    let [nx, ny, nz] = n;
    let (sy, sz) = (nx, nx * ny);
    let mut b = vec![false; m.len()];
    let mut mb = vec![false; m.len()];
    if nx < 3 || ny < 3 || nz < 3 {
        return mb;
    }
    // The shell between `2:n-1` and `3:n-2-iseven(n)` (1-based), i.e. 0-based `1..=n-2` minus
    // `2..=n-3-iseven(n)` (an empty inner range leaves the whole outer range).
    let inner = |c: usize, n: usize| -> bool {
        let hi = n as isize - 3 - n.is_multiple_of(2) as isize;
        (c as isize) >= 2 && (c as isize) <= hi
    };
    for k in 1..nz - 1 {
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                if !(inner(i, nx) && inner(j, ny) && inner(k, nz)) {
                    let c = i + j * sy + k * sz;
                    b[c] = m[c];
                }
            }
        }
    }
    // The 2×2×2 blocks of the coarse pattern that are not entirely in the mask.
    let nc = [restrict_len(nx), restrict_len(ny), restrict_len(nz)];
    for kc in 1..nc[2].saturating_sub(1) {
        for jc in 1..nc[1].saturating_sub(1) {
            for ic in 1..nc[0].saturating_sub(1) {
                let f = 2 * ic + 2 * jc * sy + 2 * kc * sz;
                let block = [f, f + 1, f + sy, f + 1 + sy, f + sz, f + 1 + sz, f + sy + sz, f + 1 + sy + sz];
                if block.iter().any(|&c| !m[c]) {
                    for c in block {
                        b[c] = m[c];
                    }
                }
            }
        }
    }
    for k in 1..nz - 1 {
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let c = i + j * sy + k * sz;
                if m[c] {
                    mb[c] = b[c - sz] || b[c - sy] || b[c - 1] || b[c] || b[c + 1] || b[c + sy] || b[c + sz];
                }
            }
        }
    }
    mb
}

/// Full-weighting restriction of `x` (fine) onto the coarse interior (`restrict!`).
fn restrict_residual(xc: &mut [f64], coarse: &Level, x: &[f64], n: [usize; 3]) {
    let (a0, a1, a2, a3) = (8.0 / 64.0, 4.0 / 64.0, 2.0 / 64.0, 1.0 / 64.0);
    let (sy, sz) = (n[0] as isize, (n[0] * n[1]) as isize);
    let nc = coarse.n;
    if nc[0] < 3 || nc[1] < 3 || nc[2] < 3 {
        return;
    }
    for kc in 1..nc[2] - 1 {
        for jc in 1..nc[1] - 1 {
            for ic in 1..nc[0] - 1 {
                let cidx = ic + jc * nc[0] + kc * nc[0] * nc[1];
                if !coarse.interior[cidx] {
                    continue;
                }
                let f = (2 * ic + 2 * jc * n[0] + 2 * kc * n[0] * n[1]) as isize;
                let v = |di: isize, dj: isize, dk: isize| x[(f + di + dj * sy + dk * sz) as usize];
                // x{a}{b}{c}: offsets a−1, b−1, c−1 along i, j, k
                let x000 = v(-1, -1, -1);
                let x100 = v(0, -1, -1);
                let x200 = v(1, -1, -1);
                let x010 = v(-1, 0, -1);
                let x110 = v(0, 0, -1);
                let x210 = v(1, 0, -1);
                let x020 = v(-1, 1, -1);
                let x120 = v(0, 1, -1);
                let x220 = v(1, 1, -1);
                let x001 = v(-1, -1, 0);
                let x101 = v(0, -1, 0);
                let x201 = v(1, -1, 0);
                let x011 = v(-1, 0, 0);
                let x111 = v(0, 0, 0);
                let x211 = v(1, 0, 0);
                let x021 = v(-1, 1, 0);
                let x121 = v(0, 1, 0);
                let x221 = v(1, 1, 0);
                let x002 = v(-1, -1, 1);
                let x102 = v(0, -1, 1);
                let x202 = v(1, -1, 1);
                let x012 = v(-1, 0, 1);
                let x112 = v(0, 0, 1);
                let x212 = v(1, 0, 1);
                let x022 = v(-1, 1, 1);
                let x122 = v(0, 1, 1);
                let x222 = v(1, 1, 1);

                let s3 = x000 + x200 + x020 + x220 + x002 + x202 + x022 + x222;
                let s2 = x100 + x010 + x210 + x120 + x001 + x201 + x021 + x221 + x102 + x012 + x212 + x122;
                let s1 = x110 + x101 + x011 + x211 + x121 + x112;

                let mut c = a0 * x111;
                c += a3 * s3;
                c += a2 * s2;
                c += a1 * s1;
                xc[cidx] = c;
            }
        }
    }
}

/// Adds the trilinear prolongation of `xc` to `x` on the fine interior (`correct_prolong!`).
fn correct_prolong(x: &mut [f64], fine: &Level, xc: &[f64], nc: [usize; 3]) {
    let (a1, a2, a3) = (4.0 / 8.0, 2.0 / 8.0, 1.0 / 8.0);
    let n = fine.n;
    let (sy, sz) = (n[0], n[0] * n[1]);
    let (csy, csz) = (nc[0], nc[0] * nc[1]);
    let m = &fine.interior;
    for kc in 0..nc[2].saturating_sub(1) {
        for jc in 0..nc[1].saturating_sub(1) {
            for ic in 0..nc[0].saturating_sub(1) {
                let cc = ic + jc * csy + kc * csz;
                let x000 = xc[cc];
                let x100 = xc[cc + 1];
                let x010 = xc[cc + csy];
                let x110 = xc[cc + 1 + csy];
                let x001 = xc[cc + csz];
                let x101 = xc[cc + 1 + csz];
                let x011 = xc[cc + csy + csz];
                let x111 = xc[cc + 1 + csy + csz];

                let x1 = x000 + x100;
                let x2 = x000 + x010;
                let x4 = x010 + x110;
                let x5 = x001 + x101;
                let x6 = x001 + x011;

                let f = 2 * ic + 2 * jc * sy + 2 * kc * sz;
                if m[f] {
                    x[f] += x000;
                }
                let c = f + 1;
                if m[c] {
                    x[c] += a1 * x1;
                }
                let c = f + sy;
                if m[c] {
                    x[c] += a1 * x2;
                }
                let c = f + 1 + sy;
                if m[c] {
                    x[c] += a2 * (x1 + x4);
                }
                let c = f + sz;
                if m[c] {
                    x[c] += a1 * (x000 + x001);
                }
                let c = f + 1 + sz;
                if m[c] {
                    x[c] += a2 * (x1 + x5);
                }
                let c = f + sy + sz;
                if m[c] {
                    x[c] += a2 * (x2 + x6);
                }
                let c = f + 1 + sy + sz;
                if m[c] {
                    x[c] += a3 * (x1 + x4 + x5 + x011 + x111);
                }
            }
        }
    }
}

/// The grid hierarchy and its work arrays (`Multigrid`, `MultigridWorkspace`).
struct Multigrid {
    levels: Vec<Level>,
    x: Vec<Vec<f64>>,
    b: Vec<Vec<f64>>,
    r: Vec<Vec<f64>>,
}

impl Multigrid {
    fn new(interior: Vec<bool>, n: [usize; 3], dx: [f64; 3], depth: usize) -> Self {
        assert!(depth >= 2, "MGPCG needs at least 2 grids");
        let mut levels = vec![Level::new(interior, n, dx)];
        for _ in 1..depth {
            let fine = levels.last().unwrap();
            let (mc, nc) = restrict_mask(&fine.interior, fine.n);
            let ratio = |d: usize| (fine.n[d] as f64 - 1.0) / (nc[d] as f64 - 1.0);
            let dxc = [ratio(0) * fine.dx[0], ratio(1) * fine.dx[1], ratio(2) * fine.dx[2]];
            levels.push(Level::new(mc, nc, dxc));
        }
        let zeros = |l: &Level| vec![0.0; l.n[0] * l.n[1] * l.n[2]];
        let x = levels.iter().map(zeros).collect();
        let b = levels.iter().map(zeros).collect();
        let r = levels.iter().map(zeros).collect();
        Multigrid { levels, x, b, r }
    }

    /// One V-cycle from level `l` (`cycle!`), solving `A x[l] = b[l]` approximately.
    fn cycle(&mut self, l: usize) {
        let depth = self.levels.len();
        let pre = 1usize << l; // QSM.jl level l+1: 2^(l+1-1)
        let bnd = 1usize << (l + 1);
        {
            let lvl = &self.levels[l];
            lvl.rbgs(&mut self.x[l], &self.b[l], false, true, pre);
            lvl.rbgs(&mut self.x[l], &self.b[l], true, true, bnd);
            lvl.residual(&mut self.r[l], &self.b[l], &self.x[l]);
            restrict_residual(&mut self.b[l + 1], &self.levels[l + 1], &self.r[l], lvl.n);
        }
        {
            let coarse = &self.levels[l + 1];
            for &i in &coarse.ri {
                self.x[l + 1][i] = 0.0;
            }
        }
        if l + 2 == depth {
            let coarse = &self.levels[l + 1];
            coarse.rbgs(&mut self.x[l + 1], &self.b[l + 1], false, true, 512);
            coarse.rbgs(&mut self.x[l + 1], &self.b[l + 1], false, false, 512);
        } else {
            self.cycle(l + 1);
        }
        let (xf, xc) = self.x.split_at_mut(l + 1);
        let lvl = &self.levels[l];
        correct_prolong(&mut xf[l], lvl, &xc[0], self.levels[l + 1].n);
        lvl.rbgs(&mut xf[l], &self.b[l], true, false, bnd);
        lvl.rbgs(&mut xf[l], &self.b[l], false, false, pre);
    }
}

/// Solves `∇²u = f` on `mask` with `u = 0` outside it (homogeneous Dirichlet), by MGPCG
/// (QSM.jl's `solve_poisson_mgpcg`, from `u₀ = 0`). Returns `u`, 0 outside `mask`, and the
/// number of CG iterations run. `f` should be 0 outside `mask` (QSM.jl's callers make it so).
pub fn solve_poisson_mgpcg(
    f: &[f64],
    mask: &[u8],
    dims: (usize, usize, usize),
    voxel_size: (f64, f64, f64),
    params: &MgpcgParams,
) -> (Vec<f64>, usize) {
    let (nx, ny, nz) = dims;
    let n_total = nx * ny * nz;
    assert_eq!(f.len(), n_total, "f length does not match dims");
    assert_eq!(mask.len(), n_total, "mask length does not match dims");
    let mut u = vec![0.0; n_total];

    // Crop to the mask's bounding box grown by one voxel, if that saves more than 10 voxels
    // along some axis (`solve_poisson_mgpcg!`).
    let n = [nx, ny, nz];
    let (mut lo, mut hi) = ([usize::MAX; 3], [0usize; 3]);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                if mask[i + j * nx + k * nx * ny] != 0 {
                    for (d, c) in [i, j, k].into_iter().enumerate() {
                        lo[d] = lo[d].min(c);
                        hi[d] = hi[d].max(c);
                    }
                }
            }
        }
    }
    if lo[0] == usize::MAX {
        return (u, 0);
    }
    for d in 0..3 {
        lo[d] = lo[d].saturating_sub(1);
        hi[d] = (hi[d] + 1).min(n[d] - 1);
    }
    let szc = [hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1];
    let crop = (0..3).map(|d| n[d] - szc[d]).max().unwrap() > 10;
    let (o, nc) = if crop { (lo, szc) } else { ([0; 3], n) };
    let nct = nc[0] * nc[1] * nc[2];
    let full = |i: usize, j: usize, k: usize| (o[0] + i) + (o[1] + j) * nx + (o[2] + k) * nx * ny;
    let mut interior = vec![false; nct];
    let mut rhs = vec![0.0; nct];
    for k in 0..nc[2] {
        for j in 0..nc[1] {
            for i in 0..nc[0] {
                let c = i + j * nc[0] + k * nc[0] * nc[1];
                let g = full(i, j, k);
                interior[c] = mask[g] != 0;
                rhs[c] = f[g];
            }
        }
    }

    let (vsx, vsy, vsz) = voxel_size;
    let depth = params.depth.unwrap_or_else(|| default_depth(nc));
    let mut mg = Multigrid::new(interior, nc, [vsx, vsy, vsz], depth);
    let (x, iters) = mgpcg(&mut mg, rhs, params);

    for k in 0..nc[2] {
        for j in 0..nc[1] {
            for i in 0..nc[0] {
                u[full(i, j, k)] = x[i + j * nc[0] + k * nc[0] * nc[1]];
            }
        }
    }
    (u, iters)
}

/// Preconditioned CG with one V-cycle as the preconditioner (`_mgpcg!`), from `x₀ = 0`.
fn mgpcg(mg: &mut Multigrid, b: Vec<f64>, params: &MgpcgParams) -> (Vec<f64>, usize) {
    let len = b.len();
    let mut x = vec![0.0; len];
    let mut p = vec![0.0; len];
    // q = M.workspace.x[1] (the V-cycle's solution), r = M.workspace.b[1] (its right-hand side).
    mg.b[0] = b;
    mg.x[0].iter_mut().for_each(|v| *v = 0.0);

    // q = A x; r -= q; res = ‖r‖ (over the whole array, as `_init_r!`)
    {
        let (q, r) = (&mut mg.x[0], &mut mg.b[0]);
        mg.levels[0].apply(q, &x);
        let mut s = 0.0;
        for (ri, qi) in r.iter_mut().zip(q.iter()) {
            *ri -= qi;
            s += *ri * *ri;
        }
        let res = s.sqrt();
        let tol = (params.rtol * res).max(params.atol);
        if res <= tol {
            return (x, 0);
        }
        let mut rho = 1.0;
        let mut it = 0;
        for i in 1..=params.maxit {
            it = i;
            // q = M \ r
            for &c in &mg.levels[0].ri {
                mg.x[0][c] = 0.0;
            }
            mg.cycle(0);

            let ri = &mg.levels[0].ri;
            let (q, r) = (&mut mg.x[0], &mut mg.b[0]);
            let rho_prev = rho;
            rho = ri.iter().map(|&c| q[c] * r[c]).sum::<f64>();
            let beta = rho / rho_prev;
            for &c in ri {
                p[c] *= beta;
                p[c] += q[c];
            }
            let pap = mg.levels[0].apply(q, &p);
            let alpha = rho / pap;
            let mut s = 0.0;
            for &c in ri {
                x[c] += alpha * p[c];
                r[c] += -alpha * q[c];
                s += r[c] * r[c];
            }
            if s.sqrt() <= tol {
                break;
            }
        }
        (x, it)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A smooth function vanishing outside a ball: the solver's `u` reproduces it from its
    /// discrete Laplacian (the discrete problem has a unique solution).
    #[test]
    fn recovers_a_function_from_its_laplacian() {
        let (nx, ny, nz) = (24, 22, 20);
        let vs = (1.0, 0.9, 1.2);
        let n = nx * ny * nz;
        let mut mask = vec![0u8; n];
        let mut u = vec![0.0; n];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let (x, y, z) = (i as f64 - 11.5, j as f64 - 10.5, k as f64 - 9.5);
                    let c = i + j * nx + k * nx * ny;
                    if x * x + y * y + z * z < 64.0 {
                        mask[c] = 1;
                        u[c] = (0.3 * x).sin() + 0.1 * y * z;
                    }
                }
            }
        }
        // f = ∇²u at the mask voxels, with u = 0 outside
        let interior: Vec<bool> = mask.iter().map(|&m| m != 0).collect();
        let lvl = Level::new(interior, [nx, ny, nz], [vs.0, vs.1, vs.2]);
        let mut f = vec![0.0; n];
        lvl.apply(&mut f, &u);
        let params = MgpcgParams { atol: 0.0, rtol: 1e-12, ..MgpcgParams::for_unwrap((nx, ny, nz)) };
        let (got, iters) = solve_poisson_mgpcg(&f, &mask, (nx, ny, nz), vs, &params);
        let err = got.iter().zip(&u).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(err < 1e-9, "max error {err:e} after {iters} iterations");
        assert!(iters < 30, "{iters} iterations");
    }

    #[test]
    fn depth_follows_qsmjl() {
        assert_eq!(default_depth([16, 20, 30]), 3);
        assert_eq!(default_depth([4, 4, 4]), 2);
        assert_eq!(default_depth([22, 18, 12]), 2);
        assert_eq!(default_depth([256, 256, 160]), 6);
        assert_eq!(MgpcgParams::for_unwrap((256, 256, 160)).maxit, 256);
    }
}

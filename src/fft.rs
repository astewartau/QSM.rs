//! FFT wrapper for 3D transforms using rustfft
//!
//! Provides 3D FFT/IFFT operations compatible with NumPy's FFT conventions.
//! Uses Fortran (column-major) order indexing to match NIfTI convention.

use num_complex::{Complex, Complex32, Complex64};
use rustfft::{Fft, FftPlanner, FftDirection};
use std::f64::consts::PI;
use std::sync::Arc;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Wrapper to send a raw mutable pointer across threads.
/// Stores as usize to avoid auto-trait issues with raw pointers.
/// Safety: caller must guarantee non-overlapping access patterns.
#[cfg(feature = "parallel")]
#[derive(Clone, Copy)]
struct SendPtr<T> {
    ptr: usize,
    len: usize,
    _t: std::marker::PhantomData<T>,
}
#[cfg(feature = "parallel")]
unsafe impl<T> Send for SendPtr<T> {}
#[cfg(feature = "parallel")]
unsafe impl<T> Sync for SendPtr<T> {}

#[cfg(feature = "parallel")]
#[allow(clippy::mut_from_ref)]
impl<T> SendPtr<T> {
    fn new(data: &mut [Complex<T>]) -> Self {
        Self { ptr: data.as_mut_ptr() as usize, len: data.len(), _t: std::marker::PhantomData }
    }
    // SAFETY: Caller must guarantee non-overlapping access patterns across threads.
    // Each (k, i) or (j, i) pair accesses a unique strided column of the 3D array.
    unsafe fn as_slice(&self) -> &mut [Complex<T>] {
        std::slice::from_raw_parts_mut(self.ptr as *mut Complex<T>, self.len)
    }
}

/// Half-open index ranges per axis: a box of a 3-D grid.
pub type Box3 = [std::ops::Range<usize>; 3];

/// Produces the x line of a transform's input that starts at linear index `l` (the slice has
/// length nx), instead of reading it from the data buffer.
pub type RowLoader<'a, T = f64> = &'a (dyn Fn(usize, &mut [Complex<T>]) + Sync);

/// Pruning and fused elementwise steps for [`Fft3dWorkspace::fft3d_with`] and
/// [`Fft3dWorkspace::ifft3d_with`]. The default is the plain transform.
#[derive(Clone, Copy)]
pub struct FftOpts<'a, T = f64> {
    /// The input is treated as zero outside this box (whatever the buffer holds there).
    pub support: Option<&'a Box3>,
    /// Only this box of the output is computed; the buffer is unspecified (finite or not)
    /// elsewhere.
    pub needed: Option<&'a Box3>,
    /// Input lines come from this instead of the buffer (fused into the first pass).
    pub load: Option<RowLoader<'a, T>>,
    /// The output is multiplied elementwise by this real array (fused into the last pass).
    pub post: Option<&'a [T]>,
}

impl<T> Default for FftOpts<'_, T> {
    fn default() -> Self {
        Self { support: None, needed: None, load: None, post: None }
    }
}

/// FFT workspace that caches plans and scratch buffers for reuse
pub struct Fft3dWorkspace {
    nx: usize,
    ny: usize,
    nz: usize,
    n_total: usize,
    // Forward FFT plans
    fft_x: Arc<dyn Fft<f64>>,
    fft_y: Arc<dyn Fft<f64>>,
    fft_z: Arc<dyn Fft<f64>>,
    // Inverse FFT plans
    ifft_x: Arc<dyn Fft<f64>>,
    ifft_y: Arc<dyn Fft<f64>>,
    ifft_z: Arc<dyn Fft<f64>>,
    // Scratch buffers
    scratch_x: Vec<Complex64>,
    scratch_y: Vec<Complex64>,
    scratch_z: Vec<Complex64>,
    buffer_y: Vec<Complex64>,
    buffer_z: Vec<Complex64>,
}

impl Fft3dWorkspace {
    /// Create a new FFT workspace for the given dimensions
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        let mut planner = FftPlanner::new();

        let fft_x = planner.plan_fft(nx, FftDirection::Forward);
        let fft_y = planner.plan_fft(ny, FftDirection::Forward);
        let fft_z = planner.plan_fft(nz, FftDirection::Forward);

        let ifft_x = planner.plan_fft(nx, FftDirection::Inverse);
        let ifft_y = planner.plan_fft(ny, FftDirection::Inverse);
        let ifft_z = planner.plan_fft(nz, FftDirection::Inverse);

        let scratch_x = vec![Complex64::new(0.0, 0.0); fft_x.get_inplace_scratch_len().max(ifft_x.get_inplace_scratch_len())];
        let scratch_y = vec![Complex64::new(0.0, 0.0); fft_y.get_inplace_scratch_len().max(ifft_y.get_inplace_scratch_len())];
        let scratch_z = vec![Complex64::new(0.0, 0.0); fft_z.get_inplace_scratch_len().max(ifft_z.get_inplace_scratch_len())];

        Self {
            nx, ny, nz,
            n_total: nx * ny * nz,
            fft_x, fft_y, fft_z,
            ifft_x, ifft_y, ifft_z,
            scratch_x, scratch_y, scratch_z,
            buffer_y: vec![Complex64::new(0.0, 0.0); ny],
            buffer_z: vec![Complex64::new(0.0, 0.0); nz],
        }
    }

    /// In-place forward 3D FFT.
    ///
    /// With the `parallel` feature the transform is parallelised across the
    /// outer axis with rayon (reusing per-thread scratch); otherwise it runs
    /// sequentially reusing the workspace's pre-allocated scratch. Both paths
    /// are numerically identical.
    pub fn fft3d(&mut self, data: &mut [Complex64]) {
        #[cfg(feature = "parallel")]
        { self.fft3d_par(data); }
        #[cfg(not(feature = "parallel"))]
        { self.fft3d_seq(data); }
    }

    /// Sequential forward 3D FFT (reuses pre-allocated scratch).
    #[allow(dead_code)]
    fn fft3d_seq(&mut self, data: &mut [Complex64]) {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);

        // Transform along x-axis (sequential — workspace reuses pre-allocated scratch)
        for k in 0..nz {
            for j in 0..ny {
                let start = idx3d(0, j, k, nx, ny);
                self.fft_x.process_with_scratch(&mut data[start..start + nx], &mut self.scratch_x);
            }
        }

        // Transform along y-axis
        for k in 0..nz {
            for i in 0..nx {
                for j in 0..ny {
                    self.buffer_y[j] = data[idx3d(i, j, k, nx, ny)];
                }
                self.fft_y.process_with_scratch(&mut self.buffer_y, &mut self.scratch_y);
                for j in 0..ny {
                    data[idx3d(i, j, k, nx, ny)] = self.buffer_y[j];
                }
            }
        }

        // Transform along z-axis
        for j in 0..ny {
            for i in 0..nx {
                for k in 0..nz {
                    self.buffer_z[k] = data[idx3d(i, j, k, nx, ny)];
                }
                self.fft_z.process_with_scratch(&mut self.buffer_z, &mut self.scratch_z);
                for k in 0..nz {
                    data[idx3d(i, j, k, nx, ny)] = self.buffer_z[k];
                }
            }
        }
    }

    /// Parallel forward 3D FFT (rayon; one task per outer slab, per-thread scratch).
    #[cfg(feature = "parallel")]
    fn fft3d_par(&mut self, data: &mut [Complex64]) {
        let full = [0..self.nx, 0..self.ny, 0..self.nz];
        let pass = Pass { scale: None, post: None, load: None, support: full.clone(), needed: full };
        transform_3d_par(data, self.nx, self.ny, self.nz, &self.fft_x, &self.fft_y, &self.fft_z, &pass);
    }

    /// Forward 3D FFT with pruning and fused elementwise steps (see [`FftOpts`]):
    /// `data ← post ⊙ F(1_support · src)`, computed on `needed`, where `src` is `data` or the
    /// rows `opts.load` produces. Every 1-D transform it does is the same rustfft call on the
    /// same values as [`Self::fft3d`]'s, so the computed entries equal the plain transform of the
    /// masked input followed by the multiplication, bitwise (up to the sign of zeros).
    pub fn fft3d_with(&mut self, data: &mut [Complex64], opts: &FftOpts) {
        self.transform_with(data, false, opts);
    }

    /// Inverse 3D FFT (with normalization) with pruning and fused elementwise steps; see
    /// [`Self::fft3d_with`]. `post` multiplies after the 1/N.
    pub fn ifft3d_with(&mut self, data: &mut [Complex64], opts: &FftOpts) {
        self.transform_with(data, true, opts);
    }

    fn transform_with(&mut self, data: &mut [Complex64], inverse: bool, opts: &FftOpts) {
        let full: Box3 = [0..self.nx, 0..self.ny, 0..self.nz];
        let support = opts.support.unwrap_or(&full);
        let needed = opts.needed.unwrap_or(&full);
        #[cfg(feature = "parallel")]
        {
            let (px, py, pz) = if inverse {
                (&self.ifft_x, &self.ifft_y, &self.ifft_z)
            } else {
                (&self.fft_x, &self.fft_y, &self.fft_z)
            };
            let pass = Pass {
                scale: inverse.then_some(self.n_total as f64),
                post: opts.post,
                load: opts.load,
                support: support.clone(),
                needed: needed.clone(),
            };
            transform_3d_par(data, self.nx, self.ny, self.nz, px, py, pz, &pass);
        }
        #[cfg(not(feature = "parallel"))]
        {
            seq_prepare(data, self.nx, self.ny, opts.load, (support != &full).then_some(support));
            if inverse { self.ifft3d_seq(data) } else { self.fft3d_seq(data) }
            if let Some(post) = opts.post {
                data.iter_mut().zip(post).for_each(|(z, &p)| *z *= p);
            }
            let _ = needed;
        }
    }

    /// In-place inverse 3D FFT (with normalization).
    ///
    /// Parallel under the `parallel` feature, sequential otherwise; both are
    /// numerically identical.
    pub fn ifft3d(&mut self, data: &mut [Complex64]) {
        #[cfg(feature = "parallel")]
        { self.ifft3d_par(data); }
        #[cfg(not(feature = "parallel"))]
        { self.ifft3d_seq(data); }
    }

    /// Sequential inverse 3D FFT (reuses pre-allocated scratch).
    #[allow(dead_code)]
    fn ifft3d_seq(&mut self, data: &mut [Complex64]) {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let n_total = self.n_total as f64;

        // Transform along x-axis (sequential — workspace reuses pre-allocated scratch)
        for k in 0..nz {
            for j in 0..ny {
                let start = idx3d(0, j, k, nx, ny);
                self.ifft_x.process_with_scratch(&mut data[start..start + nx], &mut self.scratch_x);
            }
        }

        // Transform along y-axis
        for k in 0..nz {
            for i in 0..nx {
                for j in 0..ny { self.buffer_y[j] = data[idx3d(i, j, k, nx, ny)]; }
                self.ifft_y.process_with_scratch(&mut self.buffer_y, &mut self.scratch_y);
                for j in 0..ny { data[idx3d(i, j, k, nx, ny)] = self.buffer_y[j]; }
            }
        }

        // Transform along z-axis
        for j in 0..ny {
            for i in 0..nx {
                for k in 0..nz { self.buffer_z[k] = data[idx3d(i, j, k, nx, ny)]; }
                self.ifft_z.process_with_scratch(&mut self.buffer_z, &mut self.scratch_z);
                for k in 0..nz { data[idx3d(i, j, k, nx, ny)] = self.buffer_z[k]; }
            }
        }

        // Normalize
        for val in data.iter_mut() { *val /= n_total; }
    }

    /// Parallel inverse 3D FFT (rayon; one task per outer slab, per-thread scratch).
    #[cfg(feature = "parallel")]
    fn ifft3d_par(&mut self, data: &mut [Complex64]) {
        let full = [0..self.nx, 0..self.ny, 0..self.nz];
        let pass = Pass { scale: Some(self.n_total as f64), post: None, load: None, support: full.clone(), needed: full };
        transform_3d_par(data, self.nx, self.ny, self.nz, &self.ifft_x, &self.ifft_y, &self.ifft_z, &pass);
    }

    /// Apply dipole convolution in-place: out = real(ifft(D * fft(x)))
    /// Uses the provided complex buffer for the transform
    #[inline]
    pub fn apply_dipole_inplace(&mut self, x: &[f64], d_kernel: &[f64], out: &mut [f64], complex_buf: &mut [Complex64]) {
        // Copy real to complex buffer
        for (c, &r) in complex_buf.iter_mut().zip(x.iter()) {
            *c = Complex64::new(r, 0.0);
        }

        self.fft3d(complex_buf);

        // Multiply by kernel
        for (c, &d) in complex_buf.iter_mut().zip(d_kernel.iter()) {
            *c *= d;
        }

        self.ifft3d(complex_buf);

        // Extract real part
        for (o, c) in out.iter_mut().zip(complex_buf.iter()) {
            *o = c.re;
        }
    }
}

/// Index into a 3D array stored in Fortran order (column-major)
/// index = x + y*nx + z*nx*ny
#[inline(always)]
pub fn idx3d(i: usize, j: usize, k: usize, nx: usize, ny: usize) -> usize {
    i + j * nx + k * nx * ny
}

// ============================================================================
// F32 (Single Precision) FFT Workspace
// ============================================================================

/// FFT workspace using f32 for better WASM performance
/// Single precision halves memory bandwidth and is faster on most hardware
pub struct Fft3dWorkspaceF32 {
    nx: usize,
    ny: usize,
    nz: usize,
    n_total: usize,
    // Forward FFT plans
    fft_x: Arc<dyn Fft<f32>>,
    fft_y: Arc<dyn Fft<f32>>,
    fft_z: Arc<dyn Fft<f32>>,
    // Inverse FFT plans
    ifft_x: Arc<dyn Fft<f32>>,
    ifft_y: Arc<dyn Fft<f32>>,
    ifft_z: Arc<dyn Fft<f32>>,
    // Scratch buffers
    scratch_x: Vec<Complex32>,
    scratch_y: Vec<Complex32>,
    scratch_z: Vec<Complex32>,
    buffer_y: Vec<Complex32>,
    buffer_z: Vec<Complex32>,
}

impl Fft3dWorkspaceF32 {
    /// Create a new f32 FFT workspace for the given dimensions
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        let mut planner = FftPlanner::<f32>::new();

        let fft_x = planner.plan_fft(nx, FftDirection::Forward);
        let fft_y = planner.plan_fft(ny, FftDirection::Forward);
        let fft_z = planner.plan_fft(nz, FftDirection::Forward);

        let ifft_x = planner.plan_fft(nx, FftDirection::Inverse);
        let ifft_y = planner.plan_fft(ny, FftDirection::Inverse);
        let ifft_z = planner.plan_fft(nz, FftDirection::Inverse);

        let scratch_x = vec![Complex32::new(0.0, 0.0); fft_x.get_inplace_scratch_len().max(ifft_x.get_inplace_scratch_len())];
        let scratch_y = vec![Complex32::new(0.0, 0.0); fft_y.get_inplace_scratch_len().max(ifft_y.get_inplace_scratch_len())];
        let scratch_z = vec![Complex32::new(0.0, 0.0); fft_z.get_inplace_scratch_len().max(ifft_z.get_inplace_scratch_len())];

        Self {
            nx, ny, nz,
            n_total: nx * ny * nz,
            fft_x, fft_y, fft_z,
            ifft_x, ifft_y, ifft_z,
            scratch_x, scratch_y, scratch_z,
            buffer_y: vec![Complex32::new(0.0, 0.0); ny],
            buffer_z: vec![Complex32::new(0.0, 0.0); nz],
        }
    }

    /// In-place forward 3D FFT
    #[inline]
    pub fn fft3d(&mut self, data: &mut [Complex32]) {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);

        // Transform along x-axis
        for k in 0..nz {
            for j in 0..ny {
                let start = idx3d(0, j, k, nx, ny);
                self.fft_x.process_with_scratch(&mut data[start..start + nx], &mut self.scratch_x);
            }
        }

        // Transform along y-axis
        for k in 0..nz {
            for i in 0..nx {
                for j in 0..ny {
                    self.buffer_y[j] = data[idx3d(i, j, k, nx, ny)];
                }
                self.fft_y.process_with_scratch(&mut self.buffer_y, &mut self.scratch_y);
                for j in 0..ny {
                    data[idx3d(i, j, k, nx, ny)] = self.buffer_y[j];
                }
            }
        }

        // Transform along z-axis
        for j in 0..ny {
            for i in 0..nx {
                for k in 0..nz {
                    self.buffer_z[k] = data[idx3d(i, j, k, nx, ny)];
                }
                self.fft_z.process_with_scratch(&mut self.buffer_z, &mut self.scratch_z);
                for k in 0..nz {
                    data[idx3d(i, j, k, nx, ny)] = self.buffer_z[k];
                }
            }
        }
    }

    /// In-place inverse 3D FFT (with normalization)
    #[inline]
    pub fn ifft3d(&mut self, data: &mut [Complex32]) {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let n_total = self.n_total as f32;

        // Transform along x-axis
        for k in 0..nz {
            for j in 0..ny {
                let start = idx3d(0, j, k, nx, ny);
                self.ifft_x.process_with_scratch(&mut data[start..start + nx], &mut self.scratch_x);
            }
        }

        // Transform along y-axis
        for k in 0..nz {
            for i in 0..nx {
                for j in 0..ny {
                    self.buffer_y[j] = data[idx3d(i, j, k, nx, ny)];
                }
                self.ifft_y.process_with_scratch(&mut self.buffer_y, &mut self.scratch_y);
                for j in 0..ny {
                    data[idx3d(i, j, k, nx, ny)] = self.buffer_y[j];
                }
            }
        }

        // Transform along z-axis
        for j in 0..ny {
            for i in 0..nx {
                for k in 0..nz {
                    self.buffer_z[k] = data[idx3d(i, j, k, nx, ny)];
                }
                self.ifft_z.process_with_scratch(&mut self.buffer_z, &mut self.scratch_z);
                for k in 0..nz {
                    data[idx3d(i, j, k, nx, ny)] = self.buffer_z[k];
                }
            }
        }

        // Normalize
        for val in data.iter_mut() {
            *val /= n_total;
        }
    }

    /// Single-precision [`Fft3dWorkspace::fft3d_with`].
    pub fn fft3d_with(&mut self, data: &mut [Complex32], opts: &FftOpts<f32>) {
        self.transform_with(data, false, opts);
    }

    /// Single-precision [`Fft3dWorkspace::ifft3d_with`].
    pub fn ifft3d_with(&mut self, data: &mut [Complex32], opts: &FftOpts<f32>) {
        self.transform_with(data, true, opts);
    }

    fn transform_with(&mut self, data: &mut [Complex32], inverse: bool, opts: &FftOpts<f32>) {
        let full: Box3 = [0..self.nx, 0..self.ny, 0..self.nz];
        #[cfg(feature = "parallel")]
        {
            let (px, py, pz) = if inverse {
                (&self.ifft_x, &self.ifft_y, &self.ifft_z)
            } else {
                (&self.fft_x, &self.fft_y, &self.fft_z)
            };
            let pass = Pass {
                scale: inverse.then_some(self.n_total as f32),
                post: opts.post,
                load: opts.load,
                support: opts.support.unwrap_or(&full).clone(),
                needed: opts.needed.unwrap_or(&full).clone(),
            };
            transform_3d_par(data, self.nx, self.ny, self.nz, px, py, pz, &pass);
        }
        #[cfg(not(feature = "parallel"))]
        {
            seq_prepare(data, self.nx, self.ny, opts.load, opts.support.filter(|b| **b != full));
            if inverse { self.ifft3d(data) } else { self.fft3d(data) }
            if let Some(post) = opts.post {
                data.iter_mut().zip(post).for_each(|(z, &p)| *z *= p);
            }
        }
    }

    /// Apply dipole convolution in-place: out = real(ifft(D * fft(x)))
    #[inline]
    pub fn apply_dipole_inplace(&mut self, x: &[f32], d_kernel: &[f32], out: &mut [f32], complex_buf: &mut [Complex32]) {
        // Copy real to complex buffer
        for (c, &r) in complex_buf.iter_mut().zip(x.iter()) {
            *c = Complex32::new(r, 0.0);
        }

        self.fft3d(complex_buf);

        // Multiply by kernel
        for (c, &d) in complex_buf.iter_mut().zip(d_kernel.iter()) {
            *c *= d;
        }

        self.ifft3d(complex_buf);

        // Extract real part
        for (o, c) in out.iter_mut().zip(complex_buf.iter()) {
            *o = c.re;
        }
    }
}

/// 3D FFT (in-place, complex-to-complex)
///
/// Transforms data in Fortran order with shape (nx, ny, nz).
/// Matches numpy.fft.fftn behavior.
pub fn fft3d(data: &mut [Complex64], nx: usize, ny: usize, nz: usize) {
    Fft3dWorkspace::new(nx, ny, nz).fft3d(data);
}

/// Sequential builds' version of the first-pass work of [`transform_3d_par`]: load the rows
/// and zero the input outside the support.
#[cfg(not(feature = "parallel"))]
fn seq_prepare<T: rustfft::FftNum>(data: &mut [Complex<T>], nx: usize, ny: usize, load: Option<RowLoader<T>>, support: Option<&Box3>) {
    if let Some(load) = load {
        for (r, row) in data.chunks_mut(nx).enumerate() {
            load(r * nx, row);
        }
    }
    if let Some(s) = support {
        for (l, z) in data.iter_mut().enumerate() {
            let (i, j, k) = (l % nx, (l / nx) % ny, l / (nx * ny));
            if !(s[0].contains(&i) && s[1].contains(&j) && s[2].contains(&k)) {
                *z = Complex::new(T::zero(), T::zero());
            }
        }
    }
}

/// Columns gathered per strided (y/z) transform in [`transform_3d_par`]: adjacent x
/// positions, so each gather/scatter touches `COL_BLOCK` contiguous elements (whole cache
/// lines) instead of one, and rustfft runs the block as one batch of `COL_BLOCK` transforms.
#[cfg(feature = "parallel")]
const COL_BLOCK: usize = 16;

/// What a [`transform_3d_par`] call computes besides the three passes of 1-D transforms.
#[cfg(feature = "parallel")]
struct Pass<'a, T> {
    /// divide by this in the last pass's write-back (the inverse transform's N)
    scale: Option<T>,
    /// then multiply by this real array
    post: Option<&'a [T]>,
    /// x lines come from this instead of the buffer
    load: Option<RowLoader<'a, T>>,
    /// the input is zero outside this box
    support: Box3,
    /// only this box of the output is needed
    needed: Box3,
}

/// Shared parallel 3-axis transform (rayon) used by the workspace methods.
///
/// Applies the given per-axis FFT plans in place (x, then y, then z). Each 1-D transform is
/// the same rustfft call on the same data as a column-by-column loop, so the result does not
/// depend on the blocking or the thread count (bitwise).
/// - x: runs of contiguous rows, one rustfft batch per task;
/// - y and z: one task per (plane, block of [`COL_BLOCK`] adjacent x columns), gathered into
///   a per-thread buffer, transformed as one batch and scattered back.
///
/// Pruning: with input support S and needed output box R, the x pass transforms the rows with
/// (j, k) in S (zeroing their entries with i outside S), the y pass the columns with i in R and
/// k in S (reading zeros for j outside S, writing back j in R), the z pass the columns with
/// (i, j) in R (reading zeros for k outside S, writing back k in R). Lines left out are zero on
/// input or not needed on output. `scale` and `post` are applied in the z pass's write-back.
#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn transform_3d_par<T: rustfft::FftNum>(
    data: &mut [Complex<T>],
    nx: usize, ny: usize, nz: usize,
    plan_x: &Arc<dyn Fft<T>>,
    plan_y: &Arc<dyn Fft<T>>,
    plan_z: &Arc<dyn Fft<T>>,
    pass: &Pass<T>,
) {
    let nxy = nx * ny;
    let zero = Complex::new(T::zero(), T::zero());
    let (s, r) = (&pass.support, &pass.needed);

    // x-axis: contiguous rows of length nx, ~64 KiB of rows per task.
    {
        let slen = plan_x.get_inplace_scratch_len();
        let s0_full = s[0].len() == nx;
        let prep = |l: usize, row: &mut [Complex<T>]| {
            if let Some(load) = pass.load {
                load(l, row);
            }
            if !s0_full {
                row[..s[0].start].fill(zero);
                row[s[0].end..].fill(zero);
            }
        };
        let prep_any = pass.load.is_some() || !s0_full;
        if s[1].len() == ny && s[2].len() == nz {
            let rows = (4096 / nx.max(1)).clamp(1, ny * nz);
            data.par_chunks_mut(nx * rows).enumerate().for_each_init(
                || vec![zero; slen],
                |scratch, (c, run)| {
                    if prep_any {
                        for (q, row) in run.chunks_mut(nx).enumerate() {
                            prep((c * rows + q) * nx, row);
                        }
                    }
                    plan_x.process_with_scratch(run, scratch)
                },
            );
        } else if !s[1].is_empty() {
            let (j0, j1) = (s[1].start, s[1].end);
            data.par_chunks_mut(nxy).enumerate().filter(|(k, _)| s[2].contains(k)).for_each_init(
                || vec![zero; slen],
                |scratch, (k, plane)| {
                    let run = &mut plane[j0 * nx..j1 * nx];
                    if prep_any {
                        for (q, row) in run.chunks_mut(nx).enumerate() {
                            prep(k * nxy + (j0 + q) * nx, row);
                        }
                    }
                    plan_x.process_with_scratch(run, scratch)
                },
            );
        }
    }

    // y-axis: task (k, block) transforms columns i0..i0+w (i in R) of z-slab k (k in S).
    {
        let (ir, kr) = (&r[0], &s[2]);
        let nb = ir.len().div_ceil(COL_BLOCK);
        let slen = plan_y.get_inplace_scratch_len();
        let data_send = SendPtr::new(data);
        (0..kr.len() * nb).into_par_iter().for_each_init(
            || (vec![zero; COL_BLOCK * ny], vec![zero; slen]),
            |(buffer, scratch), t| {
                let (k, i0) = (kr.start + t / nb, ir.start + (t % nb) * COL_BLOCK);
                let w = COL_BLOCK.min(ir.end - i0);
                // SAFETY: distinct (k, block) → disjoint sets of (i, k) columns.
                let slice = unsafe { data_send.as_slice() };
                let buf = &mut buffer[..w * ny];
                for j in 0..ny {
                    if s[1].contains(&j) {
                        let src = &slice[i0 + j * nx + k * nxy..][..w];
                        for (b, &v) in src.iter().enumerate() { buf[b * ny + j] = v; }
                    } else {
                        for b in 0..w { buf[b * ny + j] = zero; }
                    }
                }
                plan_y.process_with_scratch(buf, scratch);
                for j in r[1].clone() {
                    let dst = &mut slice[i0 + j * nx + k * nxy..][..w];
                    for (b, d) in dst.iter_mut().enumerate() { *d = buf[b * ny + j]; }
                }
            },
        );
    }

    // z-axis: task (j, block) transforms columns i0..i0+w of xz-plane j ((i, j) in R).
    {
        let (ir, jr) = (&r[0], &r[1]);
        let nb = ir.len().div_ceil(COL_BLOCK);
        let slen = plan_z.get_inplace_scratch_len();
        let data_send = SendPtr::new(data);
        (0..jr.len() * nb).into_par_iter().for_each_init(
            || (vec![zero; COL_BLOCK * nz], vec![zero; slen]),
            |(buffer, scratch), t| {
                let (j, i0) = (jr.start + t / nb, ir.start + (t % nb) * COL_BLOCK);
                let w = COL_BLOCK.min(ir.end - i0);
                // SAFETY: distinct (j, block) → disjoint sets of (i, j) columns.
                let slice = unsafe { data_send.as_slice() };
                let buf = &mut buffer[..w * nz];
                for k in 0..nz {
                    if s[2].contains(&k) {
                        let src = &slice[i0 + j * nx + k * nxy..][..w];
                        for (b, &v) in src.iter().enumerate() { buf[b * nz + k] = v; }
                    } else {
                        for b in 0..w { buf[b * nz + k] = zero; }
                    }
                }
                plan_z.process_with_scratch(buf, scratch);
                for k in r[2].clone() {
                    let at = i0 + j * nx + k * nxy;
                    let dst = &mut slice[at..][..w];
                    match (pass.scale, pass.post.map(|p| &p[at..][..w])) {
                        (Some(sc), Some(p)) => for (b, d) in dst.iter_mut().enumerate() { *d = buf[b * nz + k] / sc * p[b]; },
                        (Some(sc), None) => for (b, d) in dst.iter_mut().enumerate() { *d = buf[b * nz + k] / sc; },
                        (None, Some(p)) => for (b, d) in dst.iter_mut().enumerate() { *d = buf[b * nz + k] * p[b]; },
                        (None, None) => for (b, d) in dst.iter_mut().enumerate() { *d = buf[b * nz + k]; },
                    }
                }
            },
        );
    }
}

/// 3D IFFT (in-place, complex-to-complex)
///
/// Transforms data in Fortran order with shape (nx, ny, nz).
/// Matches numpy.fft.ifftn behavior (includes 1/N normalization).
pub fn ifft3d(data: &mut [Complex64], nx: usize, ny: usize, nz: usize) {
    Fft3dWorkspace::new(nx, ny, nz).ifft3d(data);
}

/// 3D FFT of real data (real-to-complex)
///
/// Returns complex array. Output shape is (nx, ny, nz) for simplicity
/// (not the half-spectrum like numpy's rfft).
pub fn fft3d_real(data: &[f64], nx: usize, ny: usize, nz: usize) -> Vec<Complex64> {
    let mut complex_data: Vec<Complex64> = data.iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();
    fft3d(&mut complex_data, nx, ny, nz);
    complex_data
}

/// 3D IFFT returning real part (complex-to-real)
///
/// Takes complex array, returns real array (imaginary parts discarded).
pub fn ifft3d_real(data: &[Complex64], nx: usize, ny: usize, nz: usize) -> Vec<f64> {
    let mut complex_data = data.to_vec();
    ifft3d(&mut complex_data, nx, ny, nz);
    complex_data.iter().map(|c| c.re).collect()
}

/// FFT a real-valued kernel, returning only real parts.
///
/// For symmetric kernels (dipole, SMV, Laplacian) the FFT is purely real,
/// so only the real component is returned.
pub fn fft_real_kernel(kernel: &[f64], nx: usize, ny: usize, nz: usize) -> Vec<f64> {
    let mut complex: Vec<Complex64> = kernel.iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();
    fft3d(&mut complex, nx, ny, nz);
    complex.iter().map(|c| c.re).collect()
}

/// Apply a real-valued k-space kernel to a real signal.
///
/// Computes Re(IFFT(FFT(x) .* kernel_fft)) where kernel_fft contains
/// pre-computed FFT real parts of a symmetric kernel.
pub fn apply_real_kernel(
    x: &[f64],
    kernel_fft: &[f64],
    nx: usize, ny: usize, nz: usize,
) -> Vec<f64> {
    let mut complex: Vec<Complex64> = x.iter()
        .map(|&v| Complex64::new(v, 0.0))
        .collect();
    fft3d(&mut complex, nx, ny, nz);
    for i in 0..complex.len() {
        complex[i] *= kernel_fft[i];
    }
    ifft3d(&mut complex, nx, ny, nz);
    complex.iter().map(|c| c.re).collect()
}

/// Generate FFT frequency values for a given dimension
/// Matches numpy.fft.fftfreq(n, d)
pub fn fftfreq(n: usize, d: f64) -> Vec<f64> {
    let mut freq = vec![0.0; n];
    let val = 1.0 / (n as f64 * d);

    if n.is_multiple_of(2) {
        // Even: [0, 1, ..., n/2-1, -n/2, ..., -1]
        for i in 0..n / 2 {
            freq[i] = (i as f64) * val;
        }
        for i in n / 2..n {
            freq[i] = ((i as i64) - (n as i64)) as f64 * val;
        }
    } else {
        // Odd: [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1]
        for i in 0..=(n - 1) / 2 {
            freq[i] = (i as f64) * val;
        }
        for i in n.div_ceil(2)..n {
            freq[i] = ((i as i64) - (n as i64)) as f64 * val;
        }
    }
    freq
}

/// Generate FFT frequency values (f32 version for WASM performance)
/// Matches numpy.fft.fftfreq(n, d)
pub fn fftfreq_f32(n: usize, d: f32) -> Vec<f32> {
    let mut freq = vec![0.0f32; n];
    let val = 1.0f32 / (n as f32 * d);

    if n.is_multiple_of(2) {
        // Even: [0, 1, ..., n/2-1, -n/2, ..., -1]
        for i in 0..n / 2 {
            freq[i] = (i as f32) * val;
        }
        for i in n / 2..n {
            freq[i] = ((i as i64) - (n as i64)) as f32 * val;
        }
    } else {
        // Odd: [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1]
        for i in 0..=(n - 1) / 2 {
            freq[i] = (i as f32) * val;
        }
        for i in n.div_ceil(2)..n {
            freq[i] = ((i as i64) - (n as i64)) as f32 * val;
        }
    }
    freq
}

/// 3D FFT shift: swap quadrants so zero-frequency is at center
///
/// Returns a new array with the zero-frequency component shifted to the center.
/// Matches numpy.fft.fftshift behavior for 3D data in Fortran order.
pub fn fftshift(data: &[f64], nx: usize, ny: usize, nz: usize) -> Vec<f64> {
    let n_total = nx * ny * nz;
    let mut out = vec![0.0; n_total];

    let hx = nx / 2;
    let hy = ny / 2;
    let hz = nz / 2;

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let si = (i + hx) % nx;
                let sj = (j + hy) % ny;
                let sk = (k + hz) % nz;
                out[idx3d(si, sj, sk, nx, ny)] = data[idx3d(i, j, k, nx, ny)];
            }
        }
    }

    out
}

/// 3D inverse FFT shift: undo fftshift
///
/// Returns a new array with the zero-frequency component shifted back to the corner.
/// Matches numpy.fft.ifftshift behavior for 3D data in Fortran order.
pub fn ifftshift(data: &[f64], nx: usize, ny: usize, nz: usize) -> Vec<f64> {
    let n_total = nx * ny * nz;
    let mut out = vec![0.0; n_total];

    let hx = nx.div_ceil(2);
    let hy = ny.div_ceil(2);
    let hz = nz.div_ceil(2);

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let si = (i + hx) % nx;
                let sj = (j + hy) % ny;
                let sk = (k + hz) % nz;
                out[idx3d(si, sj, sk, nx, ny)] = data[idx3d(i, j, k, nx, ny)];
            }
        }
    }

    out
}

/// 3D FFT shift in-place: swap quadrants so zero-frequency is at center
///
/// Modifies the input array in place. Only works correctly for even-sized dimensions.
pub fn fftshift_inplace(data: &mut [f64], nx: usize, ny: usize, nz: usize) {
    let hx = nx / 2;
    let hy = ny / 2;
    let hz = nz / 2;

    // For even dimensions, fftshift is its own inverse and can be done
    // by swapping pairs of elements
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let si = (i + hx) % nx;
                let sj = (j + hy) % ny;
                let sk = (k + hz) % nz;

                let idx_src = idx3d(i, j, k, nx, ny);
                let idx_dst = idx3d(si, sj, sk, nx, ny);

                // Only swap once (when src < dst)
                if idx_src < idx_dst {
                    data.swap(idx_src, idx_dst);
                }
            }
        }
    }
}

/// Wrap angle to [-π, π]
#[inline]
pub fn wrap_angle(angle: f64) -> f64 {
    let mut a = angle % (2.0 * PI);
    if a > PI {
        a -= 2.0 * PI;
    } else if a < -PI {
        a += 2.0 * PI;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_ifft_roundtrip() {
        let nx = 4;
        let ny = 4;
        let nz = 4;

        // Create test data
        let original: Vec<f64> = (0..nx * ny * nz).map(|i| i as f64).collect();

        // FFT then IFFT
        let mut data: Vec<Complex64> = original.iter()
            .map(|&x| Complex64::new(x, 0.0))
            .collect();

        fft3d(&mut data, nx, ny, nz);
        ifft3d(&mut data, nx, ny, nz);

        // Check roundtrip
        for (i, (&orig, result)) in original.iter().zip(data.iter()).enumerate() {
            assert!(
                (result.re - orig).abs() < 1e-10,
                "Mismatch at index {}: expected {}, got {}",
                i, orig, result.re
            );
            assert!(
                result.im.abs() < 1e-10,
                "Imaginary part not zero at index {}: {}",
                i, result.im
            );
        }
    }

    #[test]
    fn test_fftfreq() {
        // Test even n=4
        let freq = fftfreq(4, 1.0);
        assert!((freq[0] - 0.0).abs() < 1e-10);
        assert!((freq[1] - 0.25).abs() < 1e-10);
        assert!((freq[2] - (-0.5)).abs() < 1e-10);
        assert!((freq[3] - (-0.25)).abs() < 1e-10);

        // Test odd n=5
        let freq = fftfreq(5, 1.0);
        assert!((freq[0] - 0.0).abs() < 1e-10);
        assert!((freq[1] - 0.2).abs() < 1e-10);
        assert!((freq[2] - 0.4).abs() < 1e-10);
        assert!((freq[3] - (-0.4)).abs() < 1e-10);
        assert!((freq[4] - (-0.2)).abs() < 1e-10);
    }

    #[test]
    fn test_fft_f32_roundtrip() {
        let nx = 4;
        let ny = 4;
        let nz = 4;

        let original: Vec<f32> = (0..nx * ny * nz).map(|i| i as f32).collect();

        let mut data: Vec<Complex32> = original.iter()
            .map(|&x| Complex32::new(x, 0.0))
            .collect();

        let mut ws = Fft3dWorkspaceF32::new(nx, ny, nz);
        ws.fft3d(&mut data);
        ws.ifft3d(&mut data);

        for (i, (&orig, result)) in original.iter().zip(data.iter()).enumerate() {
            assert!(
                (result.re - orig).abs() < 1e-4,
                "f32 roundtrip mismatch at index {}: expected {}, got {}",
                i, orig, result.re
            );
            assert!(
                result.im.abs() < 1e-4,
                "f32 imaginary part not zero at index {}: {}",
                i, result.im
            );
        }
    }

    #[test]
    fn test_fftshift_even() {
        // 4x4x4 array with sequential values
        let nx = 4;
        let ny = 4;
        let nz = 4;
        let n = nx * ny * nz;

        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let shifted = fftshift(&data, nx, ny, nz);

        // After fftshift, element at (0,0,0) should move to (2,2,2)
        // Original index for (0,0,0) = 0
        // Shifted position (2,2,2) -> index = 2 + 2*4 + 2*4*4 = 2 + 8 + 32 = 42
        assert!(
            (shifted[idx3d(2, 2, 2, nx, ny)] - data[idx3d(0, 0, 0, nx, ny)]).abs() < 1e-12,
            "fftshift: element at (0,0,0) should move to (2,2,2)"
        );

        // Element at (1,1,1) should move to (3,3,3)
        assert!(
            (shifted[idx3d(3, 3, 3, nx, ny)] - data[idx3d(1, 1, 1, nx, ny)]).abs() < 1e-12,
            "fftshift: element at (1,1,1) should move to (3,3,3)"
        );

        // Size should be preserved
        assert_eq!(shifted.len(), n);
    }

    #[test]
    fn test_ifftshift_roundtrip() {
        let nx = 4;
        let ny = 4;
        let nz = 4;
        let n = nx * ny * nz;

        let data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1).collect();

        // fftshift then ifftshift should be identity (for even dimensions)
        let shifted = fftshift(&data, nx, ny, nz);
        let unshifted = ifftshift(&shifted, nx, ny, nz);

        for i in 0..n {
            assert!(
                (unshifted[i] - data[i]).abs() < 1e-12,
                "ifftshift(fftshift(x)) != x at index {}: expected {}, got {}",
                i, data[i], unshifted[i]
            );
        }
    }

    #[test]
    fn test_fftshift_inplace() {
        let nx = 4;
        let ny = 4;
        let nz = 4;
        let n = nx * ny * nz;

        let original: Vec<f64> = (0..n).map(|i| i as f64).collect();

        // Compare in-place version with out-of-place version
        let shifted_copy = fftshift(&original, nx, ny, nz);

        let mut data = original.clone();
        fftshift_inplace(&mut data, nx, ny, nz);

        for i in 0..n {
            assert!(
                (data[i] - shifted_copy[i]).abs() < 1e-12,
                "fftshift_inplace mismatch at index {}: expected {}, got {}",
                i, shifted_copy[i], data[i]
            );
        }
    }

    /// Verify parallel workspace FFT matches sequential.
    #[cfg(feature = "parallel")]
    #[test]
    fn test_fft3d_workspace_parallel_matches_sequential() {
        let n = 16;
        let input: Vec<Complex64> = (0..n*n*n)
            .map(|i| Complex64::new((i as f64 * 0.3).sin(), (i as f64 * 0.7).cos()))
            .collect();

        // Sequential (1 thread)
        let pool_1 = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let result_seq = pool_1.install(|| {
            let mut ws = super::Fft3dWorkspace::new(n, n, n);
            let mut data = input.clone();
            ws.fft3d(&mut data);
            data
        });

        // Parallel (default threads)
        let result_par = {
            let mut ws = super::Fft3dWorkspace::new(n, n, n);
            let mut data = input.clone();
            ws.fft3d(&mut data);
            data
        };

        for (i, (s, p)) in result_seq.iter().zip(result_par.iter()).enumerate() {
            assert!(
                (s - p).norm() < 1e-10,
                "FFT mismatch at {}: seq={} par={}", i, s, p
            );
        }
    }

    /// The blocked parallel transforms (batched columns, 1/N folded into the last pass) are
    /// bitwise identical to the plain column-by-column sequential ones, on sizes that leave a
    /// partial column block and a single-row x pass.
    #[cfg(feature = "parallel")]
    #[test]
    fn test_workspace_parallel_bitwise_equals_sequential() {
        for &(nx, ny, nz) in &[(44, 50, 12), (33, 7, 5), (1, 9, 6), (20, 1, 3), (352, 4, 6)] {
            let input: Vec<Complex64> = (0..nx * ny * nz)
                .map(|i| Complex64::new((i as f64 * 0.37).sin(), (i as f64 * 0.71).cos()))
                .collect();
            let mut ws = super::Fft3dWorkspace::new(nx, ny, nz);
            let (mut a, mut b) = (input.clone(), input.clone());
            ws.fft3d_seq(&mut a);
            ws.fft3d_par(&mut b);
            assert_eq!(a, b, "fft {:?}", (nx, ny, nz));
            ws.ifft3d_seq(&mut a);
            ws.ifft3d_par(&mut b);
            assert_eq!(a, b, "ifft {:?}", (nx, ny, nz));
        }
    }

    /// `fft3d_with`/`ifft3d_with` equal the plain transforms with the masking, loading and
    /// multiplication done as separate passes: forward on input with garbage outside the support
    /// box, inverse inside the needed box; compared bitwise (`==` treats ±0 as equal).
    #[test]
    fn test_pruned_fused_transforms_match_full() {
        use super::{Box3, FftOpts};
        let (nx, ny, nz) = (40, 18, 12);
        let bx: Box3 = [5..37, 3..11, 2..9];
        let inbox = |l: usize| bx[0].contains(&(l % nx)) && bx[1].contains(&((l / nx) % ny)) && bx[2].contains(&(l / (nx * ny)));
        let n = nx * ny * nz;
        let input: Vec<Complex64> = (0..n).map(|l| Complex64::new((l as f64 * 0.37).sin(), (l as f64 * 0.71).cos())).collect();
        let post: Vec<f64> = (0..n).map(|l| 1.0 + (l as f64 * 0.13).sin()).collect();
        let pre: Vec<f64> = (0..n).map(|l| (l as f64 * 0.29).cos()).collect();
        let mut ws = super::Fft3dWorkspace::new(nx, ny, nz);

        // forward: post ⊙ F(1_box · input)
        let mut a: Vec<Complex64> = (0..n).map(|l| if inbox(l) { input[l] } else { Complex64::new(0.0, 0.0) }).collect();
        ws.fft3d(&mut a);
        a.iter_mut().zip(&post).for_each(|(z, &p)| *z *= p);
        let mut b = input.clone();
        ws.fft3d_with(&mut b, &FftOpts { support: Some(&bx), post: Some(&post), ..Default::default() });
        assert_eq!(a, b);

        // inverse of pre ⊙ input, loaded row by row, times post, needed on the box only
        let mut a: Vec<Complex64> = input.iter().zip(&pre).map(|(&z, &p)| z * p).collect();
        ws.ifft3d(&mut a);
        a.iter_mut().zip(&post).for_each(|(z, &p)| *z *= p);
        let load = |l: usize, row: &mut [Complex64]| {
            for (q, o) in row.iter_mut().enumerate() { *o = input[l + q] * pre[l + q]; }
        };
        let mut b = vec![Complex64::new(f64::NAN, 0.0); n];
        ws.ifft3d_with(&mut b, &FftOpts { needed: Some(&bx), load: Some(&load), post: Some(&post), ..Default::default() });
        for l in (0..n).filter(|&l| inbox(l)) {
            assert_eq!(a[l], b[l], "at {l}");
        }
    }

    /// Verify parallel workspace IFFT matches sequential.
    #[cfg(feature = "parallel")]
    #[test]
    fn test_ifft3d_workspace_parallel_matches_sequential() {
        let n = 16;
        let input: Vec<Complex64> = (0..n*n*n)
            .map(|i| Complex64::new((i as f64 * 0.3).sin(), (i as f64 * 0.7).cos()))
            .collect();

        let pool_1 = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let result_seq = pool_1.install(|| {
            let mut ws = super::Fft3dWorkspace::new(n, n, n);
            let mut data = input.clone();
            ws.ifft3d(&mut data);
            data
        });

        let result_par = {
            let mut ws = super::Fft3dWorkspace::new(n, n, n);
            let mut data = input.clone();
            ws.ifft3d(&mut data);
            data
        };

        for (i, (s, p)) in result_seq.iter().zip(result_par.iter()).enumerate() {
            assert!(
                (s - p).norm() < 1e-10,
                "IFFT mismatch at {}: seq={} par={}", i, s, p
            );
        }
    }

    /// Verify FFT roundtrip (FFT then IFFT = identity) with parallel.
    #[cfg(feature = "parallel")]
    #[test]
    fn test_fft_ifft_roundtrip_parallel() {
        let n = 16;
        let original: Vec<Complex64> = (0..n*n*n)
            .map(|i| Complex64::new((i as f64 * 0.3).sin(), 0.0))
            .collect();

        let mut ws = super::Fft3dWorkspace::new(n, n, n);
        let mut data = original.clone();
        ws.fft3d(&mut data);
        ws.ifft3d(&mut data);

        for (i, (orig, round)) in original.iter().zip(data.iter()).enumerate() {
            assert!(
                (orig - round).norm() < 1e-10,
                "Roundtrip mismatch at {}: orig={} round={}", i, orig, round
            );
        }
    }
}

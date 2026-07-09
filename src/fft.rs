//! FFT wrapper for 3D transforms using rustfft
//!
//! Provides 3D FFT/IFFT operations compatible with NumPy's FFT conventions.
//! Uses Fortran (column-major) order indexing to match NIfTI convention.

use num_complex::{Complex32, Complex64};
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
struct SendPtr {
    ptr: usize,
    len: usize,
}
#[cfg(feature = "parallel")]
unsafe impl Send for SendPtr {}
#[cfg(feature = "parallel")]
unsafe impl Sync for SendPtr {}

#[cfg(feature = "parallel")]
#[allow(clippy::mut_from_ref)]
impl SendPtr {
    fn new(data: &mut [Complex64]) -> Self {
        Self { ptr: data.as_mut_ptr() as usize, len: data.len() }
    }
    // SAFETY: Caller must guarantee non-overlapping access patterns across threads.
    // Each (k, i) or (j, i) pair accesses a unique strided column of the 3D array.
    unsafe fn as_slice(&self) -> &mut [Complex64] {
        std::slice::from_raw_parts_mut(self.ptr as *mut Complex64, self.len)
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
        transform_3d_par(
            data, self.nx, self.ny, self.nz,
            &self.fft_x, &self.fft_y, &self.fft_z,
        );
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
        transform_3d_par(
            data, self.nx, self.ny, self.nz,
            &self.ifft_x, &self.ifft_y, &self.ifft_z,
        );
        let n_total = self.n_total as f64;
        data.par_iter_mut().for_each(|val| { *val /= n_total; });
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

/// Shared parallel 3-axis transform (rayon) used by the workspace methods.
///
/// Applies the given per-axis FFT plans in place, with no normalization.
/// Parallelised across the outer axis: one task per slab, reusing a single
/// per-thread buffer+scratch across all columns in that slab (avoids the
/// per-column allocation of a naive `par_iter` over every column).
#[cfg(feature = "parallel")]
fn transform_3d_par(
    data: &mut [Complex64],
    nx: usize, ny: usize, nz: usize,
    plan_x: &Arc<dyn Fft<f64>>,
    plan_y: &Arc<dyn Fft<f64>>,
    plan_z: &Arc<dyn Fft<f64>>,
) {
    let nxy = nx * ny;

    // x-axis: contiguous rows of length nx.
    {
        let plan_x = plan_x.clone();
        let slen = plan_x.get_inplace_scratch_len();
        data.par_chunks_mut(nx).for_each_init(
            || vec![Complex64::new(0.0, 0.0); slen],
            |scratch, row| plan_x.process_with_scratch(row, scratch),
        );
    }

    // y-axis: one task per z-slab (fixed k), reusing buffer+scratch across its nx columns.
    {
        let plan_y = plan_y.clone();
        let slen = plan_y.get_inplace_scratch_len();
        let data_send = SendPtr::new(data);
        (0..nz).into_par_iter().for_each_init(
            || (vec![Complex64::new(0.0, 0.0); ny], vec![Complex64::new(0.0, 0.0); slen]),
            |(buffer, scratch), k| {
                // SAFETY: distinct k → disjoint z-slabs; distinct i → disjoint columns.
                let slice = unsafe { data_send.as_slice() };
                for i in 0..nx {
                    for j in 0..ny { buffer[j] = slice[i + j * nx + k * nxy]; }
                    plan_y.process_with_scratch(buffer, scratch);
                    for j in 0..ny { slice[i + j * nx + k * nxy] = buffer[j]; }
                }
            },
        );
    }

    // z-axis: one task per fixed-j plane, reusing buffer across its nx columns.
    {
        let plan_z = plan_z.clone();
        let slen = plan_z.get_inplace_scratch_len();
        let data_send = SendPtr::new(data);
        (0..ny).into_par_iter().for_each_init(
            || (vec![Complex64::new(0.0, 0.0); nz], vec![Complex64::new(0.0, 0.0); slen]),
            |(buffer, scratch), j| {
                // SAFETY: distinct j → disjoint (i,k) planes; distinct i → disjoint columns.
                let slice = unsafe { data_send.as_slice() };
                for i in 0..nx {
                    for k in 0..nz { buffer[k] = slice[i + j * nx + k * nxy]; }
                    plan_z.process_with_scratch(buffer, scratch);
                    for k in 0..nz { slice[i + j * nx + k * nxy] = buffer[k]; }
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

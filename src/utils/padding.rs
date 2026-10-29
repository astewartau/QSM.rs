//! Padding utilities for FFT
//!
//! Functions to pad arrays to sizes efficient for FFT.

/// Find next size that is efficient for FFT
///
/// FFT is most efficient when the size factors into small primes (2, 3, 5).
/// This finds the smallest n >= size that satisfies this.
pub fn next_fast_fft_size(size: usize) -> usize {
    let mut n = size;
    loop {
        let mut m = n;
        // Factor out 2, 3, 5
        while m.is_multiple_of(2) { m /= 2; }
        while m.is_multiple_of(3) { m /= 3; }
        while m.is_multiple_of(5) { m /= 5; }
        if m == 1 {
            return n;
        }
        n += 1;
    }
}

/// Next size `>= n` whose only prime factors are 2, 3, 5 and 7: QSM.jl's `nextfastfft`
/// (`nextprod((2, 3, 5, 7), n)`, `src/utils/utils.jl:15-17`, fde4b92).
pub fn qsmjl_next_fast_fft(n: usize) -> usize {
    let mut s = n.max(1);
    loop {
        let mut m = s;
        for p in [2, 3, 5, 7] {
            while m.is_multiple_of(p) {
                m /= p;
            }
        }
        if m == 1 {
            return s;
        }
        s += 1;
    }
}

/// QSM.jl's `fastfftsize(sz, ksz; rfft = true)` (`src/utils/utils.jl:41-72`, fde4b92), the array
/// size QSM.jl's `padfastfft` pads to before its FFTs. Axis `d` with `ksz[d] < 0` keeps its
/// size; the others grow to `qsmjl_next_fast_fft(sz[d] + max(ksz[d], 1) − 1)`, and the first
/// such axis (the real-to-complex FFT's halved axis) is then moved to the next even fast size
/// within three tries, or else to the fast size + 1.
///
/// QSM.jl uses this size for every FFT-based method, so a port that must reproduce QSM.jl's
/// periodic boundary (a deconvolution, an inverse filter) has to use the same size, even with
/// complex FFTs.
pub fn qsmjl_fast_fft_size(sz: [usize; 3], ksz: [i64; 3]) -> [usize; 3] {
    let Some(i1) = ksz.iter().position(|&k| k > -1) else { return sz };
    let mut out = sz;
    for d in 0..3 {
        if ksz[d] < 0 {
            continue;
        }
        let s0 = qsmjl_next_fast_fft(sz[d] + ksz[d].max(1) as usize - 1);
        out[d] = if d == i1 {
            let mut s = s0;
            for _ in 0..3 {
                if s.is_multiple_of(2) {
                    break;
                }
                s = qsmjl_next_fast_fft(s + 1);
            }
            if s.is_multiple_of(2) { s } else { s0 + 1 }
        } else {
            s0
        };
    }
    out
}

/// QSM.jl's default `pad` for its FFT inversions (`tkd`, `tsvd`, `tikh`, `tv`, `rts`, `nltv`):
/// `(0, 0, 0)`, padding each axis only up to QSM.jl's fast FFT size.
pub const QSMJL_PAD: [i64; 3] = [0; 3];

/// Runs `inv` on `field` and `mask` zero-padded the way QSM.jl's FFT inversions pad them (their
/// `pad` keyword: `padfastfft(f, pad; rfft = true)`, `src/inversion/direct.jl:182-184`,
/// `rts.jl:109-111`, `tv.jl`, fde4b92) and crops the result back. An axis with `pad < 0` is not
/// padded; otherwise it grows to [`qsmjl_fast_fft_size`]. QSM.jl centres the data in the
/// padding; here it sits at the origin, which is the same for these inversions (all periodic:
/// FFTs and periodic finite differences, the mask applied voxel-wise).
///
/// The padding changes the result, because the inverse filters act on the periodic extension
/// of the array: QSM.jl's defaults (`pad = (0, 0, 0)`) pad a 184-voxel axis to 192, for
/// example. A port that must reproduce QSM.jl at its defaults has to pad the same way.
pub fn with_qsmjl_padding(
    field: &[f64],
    mask: &[u8],
    grid: &crate::Grid,
    pad: [i64; 3],
    inv: impl FnOnce(&[f64], &[u8], &crate::Grid) -> Vec<f64>,
) -> Vec<f64> {
    let dims = [grid.dims.0, grid.dims.1, grid.dims.2];
    let p = qsmjl_fast_fft_size(dims, pad);
    if p == dims {
        return inv(field, mask, grid);
    }
    let pd = (p[0], p[1], p[2]);
    let pg = crate::Grid::new(p[0], p[1], p[2], grid.vsx(), grid.vsy(), grid.vsz());
    let out = inv(&pad3d(field, grid.dims, pd), &pad3d(mask, grid.dims, pd), &pg);
    unpad3d(&out, pd, grid.dims)
}

/// Pad a 3D array to fast FFT size
///
/// # Arguments
/// * `data` - Input array (nx * ny * nz)
/// * `grid` - Volume grid (dimensions and voxel sizes)
/// * `min_pad` - Minimum padding on each side (negative means no padding)
///
/// # Returns
/// (padded_data, new_nx, new_ny, new_nz)
pub fn pad_to_fast_fft(
    data: &[f64],
    grid: &crate::Grid,
    min_pad: (i32, i32, i32),
) -> (Vec<f64>, usize, usize, usize) {
    let (nx, ny, nz) = grid.dims;
    // Calculate new sizes
    let new_nx = if min_pad.0 >= 0 {
        next_fast_fft_size(nx + 2 * min_pad.0 as usize)
    } else {
        nx
    };
    let new_ny = if min_pad.1 >= 0 {
        next_fast_fft_size(ny + 2 * min_pad.1 as usize)
    } else {
        ny
    };
    let new_nz = if min_pad.2 >= 0 {
        next_fast_fft_size(nz + 2 * min_pad.2 as usize)
    } else {
        nz
    };

    // Create padded array (zero-filled)
    let new_total = new_nx * new_ny * new_nz;
    let mut padded = vec![0.0; new_total];

    // Copy original data (Fortran order: index = i + j*nx + k*nx*ny)
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let old_idx = i + j * nx + k * nx * ny;
                let new_idx = i + j * new_nx + k * new_nx * new_ny;
                padded[new_idx] = data[old_idx];
            }
        }
    }

    (padded, new_nx, new_ny, new_nz)
}

/// Pad a 3D array to `new_dims`, corner-placed (data at the origin) and
/// default-filled. Corner placement is equivalent to centring for circular
/// (FFT) convolution: the wrap gap between data end and data start is
/// `new_n - n` either way.
pub fn pad3d<T: Copy + Default>(
    data: &[T],
    dims: (usize, usize, usize),
    new_dims: (usize, usize, usize),
) -> Vec<T> {
    let (nx, ny, nz) = dims;
    let (px, py, pz) = new_dims;
    assert!(px >= nx && py >= ny && pz >= nz, "padded dims must not shrink");
    let mut out = vec![T::default(); px * py * pz];
    for k in 0..nz {
        for j in 0..ny {
            let src = k * nx * ny + j * nx;
            let dst = k * px * py + j * px;
            out[dst..dst + nx].copy_from_slice(&data[src..src + nx]);
        }
    }
    out
}

/// Extract the corner-placed `dims` region from an array of `padded_dims`
/// (inverse of [`pad3d`]).
pub fn unpad3d<T: Copy + Default>(
    padded: &[T],
    padded_dims: (usize, usize, usize),
    dims: (usize, usize, usize),
) -> Vec<T> {
    let (nx, ny, nz) = dims;
    let (px, py, _pz) = padded_dims;
    let mut out = vec![T::default(); nx * ny * nz];
    for k in 0..nz {
        for j in 0..ny {
            let src = k * px * py + j * px;
            let dst = k * nx * ny + j * nx;
            out[dst..dst + nx].copy_from_slice(&padded[src..src + nx]);
        }
    }
    out
}

/// Extract original-sized region from padded array
pub fn unpad(
    padded: &[f64],
    padded_nx: usize, padded_ny: usize, _padded_nz: usize,
    orig_grid: &crate::Grid,
) -> Vec<f64> {
    let (orig_nx, orig_ny, orig_nz) = orig_grid.dims;
    let orig_total = orig_nx * orig_ny * orig_nz;
    let mut data = vec![0.0; orig_total];

    // Fortran order: index = i + j*nx + k*nx*ny
    for k in 0..orig_nz {
        for j in 0..orig_ny {
            for i in 0..orig_nx {
                let padded_idx = i + j * padded_nx + k * padded_nx * padded_ny;
                let orig_idx = i + j * orig_nx + k * orig_nx * orig_ny;
                data[orig_idx] = padded[padded_idx];
            }
        }
    }

    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Grid;

    fn grid(nx: usize, ny: usize, nz: usize) -> Grid {
        Grid::new(nx, ny, nz, 1.0, 1.0, 1.0)
    }

    #[test]
    fn test_fast_fft_sizes() {
        // These should already be fast sizes
        assert_eq!(next_fast_fft_size(2), 2);
        assert_eq!(next_fast_fft_size(4), 4);
        assert_eq!(next_fast_fft_size(8), 8);
        assert_eq!(next_fast_fft_size(16), 16);
        assert_eq!(next_fast_fft_size(64), 64);

        // 7 is not a fast size, should round up to 8
        assert_eq!(next_fast_fft_size(7), 8);

        // 17 should round up to 18 (2 * 9 = 2 * 3^2)
        assert_eq!(next_fast_fft_size(17), 18);
    }

    #[test]
    fn test_pad_unpad_roundtrip() {
        let nx = 5;
        let ny = 6;
        let nz = 7;
        let g = grid(nx, ny, nz);
        let data: Vec<f64> = (0..nx*ny*nz).map(|i| i as f64).collect();

        let (padded, pnx, pny, pnz) = pad_to_fast_fft(&data, &g, (2, 2, 2));

        // Padded size should be >= original + 2*padding
        assert!(pnx >= nx + 4);
        assert!(pny >= ny + 4);
        assert!(pnz >= nz + 4);

        let recovered = unpad(&padded, pnx, pny, pnz, &g);

        // Should match original
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            assert_eq!(*orig, *rec);
        }
    }

    #[test]
    fn qsmjl_fast_fft_size_matches_qsm_jl() {
        // QSM.jl fde4b92 `QSM.fastfftsize(sz, ksz; rfft=true)`, run in the QSM-CI julia-qsm image
        let cases: [([usize; 3], [i64; 3], [usize; 3]); 9] = [
            ([11, 11, 13], [0; 3], [12, 12, 14]),
            ([176, 240, 150], [0; 3], [180, 240, 150]),
            ([255, 97, 49], [0; 3], [256, 98, 49]),
            ([15, 15, 15], [0; 3], [16, 15, 15]),
            ([141, 170, 141], [18; 3], [160, 189, 160]),
            ([33, 29, 25], [6; 3], [40, 35, 30]),
            ([33, 29, 25], [-1, 6, 6], [33, 36, 30]),
            ([37, 41, 23], [4, -1, 2], [40, 41, 24]),
            ([20, 20, 20], [-1; 3], [20, 20, 20]),
        ];
        for (sz, k, want) in cases {
            assert_eq!(qsmjl_fast_fft_size(sz, k), want, "fastfftsize({sz:?}, {k:?})");
        }
    }
}

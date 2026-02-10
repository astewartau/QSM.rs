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
        while m % 2 == 0 { m /= 2; }
        while m % 3 == 0 { m /= 3; }
        while m % 5 == 0 { m /= 5; }
        if m == 1 {
            return n;
        }
        n += 1;
    }
}

/// Pad a 3D array to fast FFT size
///
/// # Arguments
/// * `data` - Input array (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Original dimensions
/// * `min_pad` - Minimum padding on each side (negative means no padding)
///
/// # Returns
/// (padded_data, new_nx, new_ny, new_nz)
pub fn pad_to_fast_fft(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    min_pad: (i32, i32, i32),
) -> (Vec<f64>, usize, usize, usize) {
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

/// Extract original-sized region from padded array
pub fn unpad(
    padded: &[f64],
    padded_nx: usize, padded_ny: usize, _padded_nz: usize,
    orig_nx: usize, orig_ny: usize, orig_nz: usize,
) -> Vec<f64> {
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
        let data: Vec<f64> = (0..nx*ny*nz).map(|i| i as f64).collect();

        let (padded, pnx, pny, pnz) = pad_to_fast_fft(&data, nx, ny, nz, (2, 2, 2));

        // Padded size should be >= original + 2*padding
        assert!(pnx >= nx + 4);
        assert!(pny >= ny + 4);
        assert!(pnz >= nz + 4);

        let recovered = unpad(&padded, pnx, pny, pnz, nx, ny, nz);

        // Should match original
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            assert_eq!(*orig, *rec);
        }
    }
}

//! Lightweight 3D volume grid descriptor.
//!
//! Contains only the geometric information needed by algorithm kernels:
//! dimensions and voxel sizes. This eliminates the need to pass 6 separate
//! parameters (nx, ny, nz, vsx, vsy, vsz) to every function.

/// A 3D volume grid with dimensions and voxel sizes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grid {
    /// Volume dimensions (nx, ny, nz)
    pub dims: (usize, usize, usize),
    /// Voxel sizes in mm (vsx, vsy, vsz)
    pub voxel_size: (f64, f64, f64),
}

impl Grid {
    /// Create a new Grid from dimensions and voxel sizes.
    #[inline]
    pub fn new(nx: usize, ny: usize, nz: usize, vsx: f64, vsy: f64, vsz: f64) -> Self {
        Self {
            dims: (nx, ny, nz),
            voxel_size: (vsx, vsy, vsz),
        }
    }

    #[inline]
    pub fn nx(&self) -> usize { self.dims.0 }
    #[inline]
    pub fn ny(&self) -> usize { self.dims.1 }
    #[inline]
    pub fn nz(&self) -> usize { self.dims.2 }
    #[inline]
    pub fn vsx(&self) -> f64 { self.voxel_size.0 }
    #[inline]
    pub fn vsy(&self) -> f64 { self.voxel_size.1 }
    #[inline]
    pub fn vsz(&self) -> f64 { self.voxel_size.2 }

    /// Total number of voxels.
    #[inline]
    pub fn n_total(&self) -> usize {
        self.dims.0 * self.dims.1 * self.dims.2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_basic() {
        let g = Grid::new(64, 64, 32, 1.0, 1.0, 2.0);
        assert_eq!(g.nx(), 64);
        assert_eq!(g.ny(), 64);
        assert_eq!(g.nz(), 32);
        assert_eq!(g.n_total(), 64 * 64 * 32);
        assert_eq!(g.vsx(), 1.0);
        assert_eq!(g.vsz(), 2.0);
    }

    #[test]
    fn test_grid_copy() {
        let g = Grid::new(10, 20, 30, 0.5, 0.5, 1.0);
        let g2 = g;
        assert_eq!(g, g2);
    }
}

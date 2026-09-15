//! Scan geometry derived from the NIfTI affine, and resampling to an axial grid.
//!
//! Two things QSM needs from the affine and nothing else provides:
//!
//! - **Which way B0 points in voxel space.** The dipole kernel is built in the voxel grid, so an
//!   oblique acquisition must either be told the true B0 direction ([`b0_direction_from_affine`])
//!   or be resampled so that the grid is cardinal-aligned and B0 is `(0, 0, 1)` by construction
//!   ([`resample_complex_to_axial`]). Getting this wrong rotates the kernel and suppresses
//!   susceptibility contrast in exactly the iron-rich structures QSM is usually measuring.
//! - **How oblique the acquisition is** ([`obliquity_from_affine`]), so a pipeline can decide
//!   whether resampling is worth the interpolation.
//!
//! ## Phase must be resampled in the complex domain
//!
//! Wrapped phase cannot be interpolated directly: halfway between `+3.0` and `-3.0` rad, a linear
//! interpolator returns `0.0`, when the correct answer is near `±π`. Every wrap in the volume
//! becomes a band of wrong values. [`resample_complex_to_axial`] takes magnitude and phase
//! together, interpolates `mag·e^{iφ}` as real and imaginary parts, and recovers magnitude and
//! phase afterwards, which is well defined across wraps. Use [`resample_to_axial`] only for
//! quantities that are already continuous (magnitude, an unwrapped field map, χ).

/// Voxel sizes (mm) from a row-major 4×4 affine: the norms of its three columns.
pub fn voxel_sizes_from_affine(affine: &[f64; 16]) -> (f64, f64, f64) {
    let col = |j: usize| {
        (affine[j] * affine[j] + affine[4 + j] * affine[4 + j] + affine[8 + j] * affine[8 + j])
            .sqrt()
    };
    (col(0), col(1), col(2))
}

/// Direction of the scanner's B0 field (world `+z`) expressed in voxel coordinates, normalised.
///
/// The voxel→world matrix is `A = R·S`, with `R` a rotation and `S = diag(voxel sizes)`. Dividing
/// each column of `A` by its norm recovers `R`, whose inverse is its transpose, so the direction
/// is simply the third row of the normalised matrix. An axial acquisition returns `(0, 0, 1)`.
///
/// Factoring the voxel sizes out first is what makes this correct for anisotropic voxels. Using
/// `A⁻¹·(0,0,1)` instead — inverting the matrix *with* the voxel scaling still in it — silently
/// skews the direction: on a 0.8 × 0.8 × 3 mm acquisition tilted 23° it returns a direction 58°
/// from `z`, and the resulting dipole kernel destroys most of the susceptibility contrast. That
/// was a real regression in QSMxT 8.2.2; [`tests::b0_direction_anisotropic_oblique`] pins it.
pub fn b0_direction_from_affine(affine: &[f64; 16]) -> (f64, f64, f64) {
    let (sx, sy, sz) = voxel_sizes_from_affine(affine);
    if sx < 1e-10 || sy < 1e-10 || sz < 1e-10 {
        return (0.0, 0.0, 1.0);
    }
    // Third row of R = A · S⁻¹, i.e. world-z expressed in voxel axes.
    let (bx, by, bz) = (affine[8] / sx, affine[9] / sy, affine[10] / sz);
    let norm = (bx * bx + by * by + bz * bz).sqrt();
    if norm < 1e-10 {
        return (0.0, 0.0, 1.0);
    }
    (bx / norm, by / norm, bz / norm)
}

/// Angle (degrees) between B0 and the voxel `+z` axis — the tilt that matters for the dipole
/// kernel. Zero for an axial acquisition; equals the scanner's slice tilt for a simple oblique.
pub fn b0_angle_from_affine(affine: &[f64; 16]) -> f64 {
    let (_, _, bz) = b0_direction_from_affine(affine);
    bz.clamp(-1.0, 1.0).abs().acos().to_degrees()
}

/// Per-axis obliquity (degrees), matching `nibabel.affines.obliquity`: for each voxel axis, the
/// angle between it and the closest world axis. `[0, 0, 0]` for a cardinal-aligned acquisition.
pub fn obliquity_axes_from_affine(affine: &[f64; 16]) -> [f64; 3] {
    let sizes = voxel_sizes_from_affine(affine);
    let sizes = [sizes.0, sizes.1, sizes.2];
    let mut out = [0.0f64; 3];
    for (i, o) in out.iter_mut().enumerate() {
        // Row i of A · S⁻¹; its largest component is the cosine to the nearest world axis.
        let best = (0..3)
            .map(|j| {
                if sizes[j] < 1e-10 {
                    0.0
                } else {
                    (affine[4 * i + j] / sizes[j]).abs()
                }
            })
            .fold(0.0f64, f64::max);
        *o = best.clamp(0.0, 1.0).acos().to_degrees();
    }
    out
}

/// Scalar obliquity (degrees): the Euclidean norm of [`obliquity_axes_from_affine`].
///
/// This is the quantity QSMxT 8.x thresholded on (`nibabel` obliquity → degrees → `np.linalg.norm`),
/// so thresholds carry over unchanged. Note it is a combined measure, not a tilt: a single 23°
/// oblique acquisition scores ≈32° here because two voxel axes each move. For the physical tilt of
/// B0 relative to the slice stack, use [`b0_angle_from_affine`].
pub fn obliquity_from_affine(affine: &[f64; 16]) -> f64 {
    let a = obliquity_axes_from_affine(affine);
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

/// Mapping from one voxel grid to another, each described by its own affine.
///
/// Used for the return trip: a run reconstructed on a resampled grid still has to write its
/// outputs where the caller's other data lives. That matters more than it sounds — a FLIRT
/// matrix, for instance, is defined in a coordinate space derived from the image's dimensions
/// and voxel sizes, so handing it a volume on a different grid produces a wrong registration
/// rather than an error.
pub struct GridMap {
    pub dst_dims: (usize, usize, usize),
    src_dims: (usize, usize, usize),
    /// Voxel→world of the destination.
    dst_affine: [f64; 16],
    /// World→voxel of the source.
    inv_src: [[f64; 4]; 3],
}

/// Invert the 3×3 of an affine and fold in its translation, giving world→voxel as a 3×4.
fn world_to_voxel(affine: &[f64; 16]) -> Option<[[f64; 4]; 3]> {
    let r = [
        [affine[0], affine[1], affine[2]],
        [affine[4], affine[5], affine[6]],
        [affine[8], affine[9], affine[10]],
    ];
    let t = [affine[3], affine[7], affine[11]];
    let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
    if det.abs() < 1e-12 {
        return None;
    }
    let id = 1.0 / det;
    let inv = [
        [
            (r[1][1] * r[2][2] - r[1][2] * r[2][1]) * id,
            (r[0][2] * r[2][1] - r[0][1] * r[2][2]) * id,
            (r[0][1] * r[1][2] - r[0][2] * r[1][1]) * id,
        ],
        [
            (r[1][2] * r[2][0] - r[1][0] * r[2][2]) * id,
            (r[0][0] * r[2][2] - r[0][2] * r[2][0]) * id,
            (r[0][2] * r[1][0] - r[0][0] * r[1][2]) * id,
        ],
        [
            (r[1][0] * r[2][1] - r[1][1] * r[2][0]) * id,
            (r[0][1] * r[2][0] - r[0][0] * r[2][1]) * id,
            (r[0][0] * r[1][1] - r[0][1] * r[1][0]) * id,
        ],
    ];
    let mut out = [[0.0f64; 4]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        row[..3].copy_from_slice(&inv[i]);
        row[3] = -(inv[i][0] * t[0] + inv[i][1] * t[1] + inv[i][2] * t[2]);
    }
    Some(out)
}

impl GridMap {
    /// Map from `src` onto `dst`. Both affines are row-major voxel→world.
    pub fn new(
        src_dims: (usize, usize, usize),
        src_affine: &[f64; 16],
        dst_dims: (usize, usize, usize),
        dst_affine: &[f64; 16],
    ) -> Option<Self> {
        Some(Self {
            dst_dims,
            src_dims,
            dst_affine: *dst_affine,
            inv_src: world_to_voxel(src_affine)?,
        })
    }

    /// Source-voxel coordinate for a destination voxel, or `None` if it falls outside the source.
    fn source_coord(&self, i: usize, j: usize, k: usize) -> Option<(f64, f64, f64)> {
        let (i, j, k) = (i as f64, j as f64, k as f64);
        let a = &self.dst_affine;
        let w = [
            a[0] * i + a[1] * j + a[2] * k + a[3],
            a[4] * i + a[5] * j + a[6] * k + a[7],
            a[8] * i + a[9] * j + a[10] * k + a[11],
        ];
        let m = &self.inv_src;
        let o = (
            m[0][0] * w[0] + m[0][1] * w[1] + m[0][2] * w[2] + m[0][3],
            m[1][0] * w[0] + m[1][1] * w[1] + m[1][2] * w[2] + m[1][3],
            m[2][0] * w[0] + m[2][1] * w[1] + m[2][2] * w[2] + m[2][3],
        );
        let (nx, ny, nz) = self.src_dims;
        let inside = o.0 >= -0.5 && o.0 <= nx as f64 - 0.5
            && o.1 >= -0.5 && o.1 <= ny as f64 - 0.5
            && o.2 >= -0.5 && o.2 <= nz as f64 - 0.5;
        if inside { Some(o) } else { None }
    }

    /// Trilinearly resample continuous data onto the destination grid; 0 outside the source.
    pub fn sample(&self, data: &[f64]) -> Vec<f64> {
        let (nx, ny, nz) = self.src_dims;
        let (dx, dy, dz) = self.dst_dims;
        let mut out = vec![0.0f64; dx * dy * dz];
        for k in 0..dz {
            for j in 0..dy {
                for i in 0..dx {
                    if let Some((ox, oy, oz)) = self.source_coord(i, j, k) {
                        out[i + j * dx + k * dx * dy] = trilinear_sample(data, nx, ny, nz, ox, oy, oz);
                    }
                }
            }
        }
        out
    }

    /// Nearest-neighbour, for labels and binary masks.
    pub fn sample_nearest(&self, data: &[u8]) -> Vec<u8> {
        let (nx, ny, nz) = self.src_dims;
        let (dx, dy, dz) = self.dst_dims;
        let mut out = vec![0u8; dx * dy * dz];
        for k in 0..dz {
            for j in 0..dy {
                for i in 0..dx {
                    if let Some((ox, oy, oz)) = self.source_coord(i, j, k) {
                        let xi = (ox.round() as isize).clamp(0, nx as isize - 1) as usize;
                        let yi = (oy.round() as isize).clamp(0, ny as isize - 1) as usize;
                        let zi = (oz.round() as isize).clamp(0, nz as isize - 1) as usize;
                        out[i + j * dx + k * dx * dy] = data[xi + yi * nx + zi * nx * ny];
                    }
                }
            }
        }
        out
    }
}

/// Resample continuous data (magnitude, an unwrapped field, χ) from one grid onto another.
/// Returns `None` if the source affine is singular.
pub fn resample_onto(
    data: &[f64],
    src_dims: (usize, usize, usize),
    src_affine: &[f64; 16],
    dst_dims: (usize, usize, usize),
    dst_affine: &[f64; 16],
) -> Option<Vec<f64>> {
    Some(GridMap::new(src_dims, src_affine, dst_dims, dst_affine)?.sample(data))
}

/// Resample a binary mask from one grid onto another (nearest neighbour).
pub fn resample_mask_onto(
    mask: &[u8],
    src_dims: (usize, usize, usize),
    src_affine: &[f64; 16],
    dst_dims: (usize, usize, usize),
    dst_affine: &[f64; 16],
) -> Option<Vec<u8>> {
    Some(GridMap::new(src_dims, src_affine, dst_dims, dst_affine)?.sample_nearest(mask))
}

/// Resample magnitude and wrapped phase from one grid onto another, through the complex domain
/// so the wraps survive. Returns `(magnitude, phase)`.
pub fn resample_complex_onto(
    magnitude: &[f64],
    phase: &[f64],
    src_dims: (usize, usize, usize),
    src_affine: &[f64; 16],
    dst_dims: (usize, usize, usize),
    dst_affine: &[f64; 16],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let n = src_dims.0 * src_dims.1 * src_dims.2;
    assert_eq!(magnitude.len(), n, "magnitude length does not match source dimensions");
    assert_eq!(phase.len(), n, "phase length does not match source dimensions");
    let map = GridMap::new(src_dims, src_affine, dst_dims, dst_affine)?;
    let real: Vec<f64> = (0..n).map(|i| magnitude[i] * phase[i].cos()).collect();
    let imag: Vec<f64> = (0..n).map(|i| magnitude[i] * phase[i].sin()).collect();
    let (re, im) = (map.sample(&real), map.sample(&imag));
    Some((
        (0..re.len()).map(|i| re[i].hypot(im[i])).collect(),
        (0..re.len()).map(|i| im[i].atan2(re[i])).collect(),
    ))
}

/// The cardinal-aligned grid an oblique volume resamples onto, plus the mapping back to the
/// original voxel space. Build once with [`axial_grid_for`] and reuse for every volume that
/// shares the geometry (magnitude, phase, mask), so they land on identical grids.
pub struct AxialGrid {
    pub dims: (usize, usize, usize),
    pub voxel_size: (f64, f64, f64),
    /// Diagonal voxel→world affine of the new grid (row-major 4×4).
    pub affine: [f64; 16],
    /// Source dimensions this grid was built for.
    src_dims: (usize, usize, usize),
    world_min: [f64; 3],
    /// Inverse of the source 3×3, for world→source-voxel.
    inv_r: [[f64; 3]; 3],
    t: [f64; 3],
}

/// Build the axial grid covering the same world-space extent as an oblique volume, keeping its
/// voxel sizes. The new affine is diagonal, so [`b0_direction_from_affine`] on it returns
/// `(0, 0, 1)`.
pub fn axial_grid_for(
    nx: usize,
    ny: usize,
    nz: usize,
    affine: &[f64; 16],
) -> AxialGrid {
    let r = [
        [affine[0], affine[1], affine[2]],
        [affine[4], affine[5], affine[6]],
        [affine[8], affine[9], affine[10]],
    ];
    let t = [affine[3], affine[7], affine[11]];
    let (vsx, vsy, vsz) = voxel_sizes_from_affine(affine);

    // World bounding box of the source volume's eight corners.
    let mut world_min = [f64::INFINITY; 3];
    let mut world_max = [f64::NEG_INFINITY; 3];
    for &(vi, vj, vk) in &[
        (0.0, 0.0, 0.0),
        (nx as f64 - 1.0, 0.0, 0.0),
        (0.0, ny as f64 - 1.0, 0.0),
        (0.0, 0.0, nz as f64 - 1.0),
        (nx as f64 - 1.0, ny as f64 - 1.0, 0.0),
        (nx as f64 - 1.0, 0.0, nz as f64 - 1.0),
        (0.0, ny as f64 - 1.0, nz as f64 - 1.0),
        (nx as f64 - 1.0, ny as f64 - 1.0, nz as f64 - 1.0),
    ] {
        for d in 0..3 {
            let w = r[d][0] * vi + r[d][1] * vj + r[d][2] * vk + t[d];
            world_min[d] = world_min[d].min(w);
            world_max[d] = world_max[d].max(w);
        }
    }

    let dims = (
        ((world_max[0] - world_min[0]) / vsx).ceil() as usize + 1,
        ((world_max[1] - world_min[1]) / vsy).ceil() as usize + 1,
        ((world_max[2] - world_min[2]) / vsz).ceil() as usize + 1,
    );

    let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
    let inv_det = if det.abs() < 1e-12 { 0.0 } else { 1.0 / det };
    let inv_r = [
        [
            (r[1][1] * r[2][2] - r[1][2] * r[2][1]) * inv_det,
            (r[0][2] * r[2][1] - r[0][1] * r[2][2]) * inv_det,
            (r[0][1] * r[1][2] - r[0][2] * r[1][1]) * inv_det,
        ],
        [
            (r[1][2] * r[2][0] - r[1][0] * r[2][2]) * inv_det,
            (r[0][0] * r[2][2] - r[0][2] * r[2][0]) * inv_det,
            (r[0][2] * r[1][0] - r[0][0] * r[1][2]) * inv_det,
        ],
        [
            (r[1][0] * r[2][1] - r[1][1] * r[2][0]) * inv_det,
            (r[0][1] * r[2][0] - r[0][0] * r[2][1]) * inv_det,
            (r[0][0] * r[1][1] - r[0][1] * r[1][0]) * inv_det,
        ],
    ];

    AxialGrid {
        dims,
        voxel_size: (vsx, vsy, vsz),
        affine: [
            vsx, 0.0, 0.0, world_min[0],
            0.0, vsy, 0.0, world_min[1],
            0.0, 0.0, vsz, world_min[2],
            0.0, 0.0, 0.0, 1.0,
        ],
        src_dims: (nx, ny, nz),
        world_min,
        inv_r,
        t,
    }
}

impl AxialGrid {
    /// Source-voxel coordinate of a target voxel, or `None` if it falls outside the source.
    fn source_coord(&self, ni: usize, nj: usize, nk: usize) -> Option<(f64, f64, f64)> {
        let (nx, ny, nz) = self.src_dims;
        let dx = self.world_min[0] + ni as f64 * self.voxel_size.0 - self.t[0];
        let dy = self.world_min[1] + nj as f64 * self.voxel_size.1 - self.t[1];
        let dz = self.world_min[2] + nk as f64 * self.voxel_size.2 - self.t[2];
        let ox = self.inv_r[0][0] * dx + self.inv_r[0][1] * dy + self.inv_r[0][2] * dz;
        let oy = self.inv_r[1][0] * dx + self.inv_r[1][1] * dy + self.inv_r[1][2] * dz;
        let oz = self.inv_r[2][0] * dx + self.inv_r[2][1] * dy + self.inv_r[2][2] * dz;
        let inside = ox >= -0.5
            && ox <= nx as f64 - 0.5
            && oy >= -0.5
            && oy <= ny as f64 - 0.5
            && oz >= -0.5
            && oz <= nz as f64 - 0.5;
        if inside { Some((ox, oy, oz)) } else { None }
    }

    /// Trilinearly resample one continuous volume onto this grid. Voxels outside the source are 0.
    pub fn sample(&self, data: &[f64]) -> Vec<f64> {
        let (nx, ny, nz) = self.src_dims;
        let (dx, dy, dz) = self.dims;
        let mut out = vec![0.0f64; dx * dy * dz];
        for nk in 0..dz {
            for nj in 0..dy {
                for ni in 0..dx {
                    if let Some((ox, oy, oz)) = self.source_coord(ni, nj, nk) {
                        out[ni + nj * dx + nk * dx * dy] =
                            trilinear_sample(data, nx, ny, nz, ox, oy, oz);
                    }
                }
            }
        }
        out
    }

    /// Nearest-neighbour resample, for labels and binary masks.
    pub fn sample_nearest(&self, data: &[u8]) -> Vec<u8> {
        let (nx, ny, nz) = self.src_dims;
        let (dx, dy, dz) = self.dims;
        let mut out = vec![0u8; dx * dy * dz];
        for nk in 0..dz {
            for nj in 0..dy {
                for ni in 0..dx {
                    if let Some((ox, oy, oz)) = self.source_coord(ni, nj, nk) {
                        let xi = (ox.round() as isize).clamp(0, nx as isize - 1) as usize;
                        let yi = (oy.round() as isize).clamp(0, ny as isize - 1) as usize;
                        let zi = (oz.round() as isize).clamp(0, nz as isize - 1) as usize;
                        out[ni + nj * dx + nk * dx * dy] = data[xi + yi * nx + zi * nx * ny];
                    }
                }
            }
        }
        out
    }
}

/// Trilinear interpolation at a fractional voxel coordinate, clamped at the edges.
fn trilinear_sample(data: &[f64], nx: usize, ny: usize, nz: usize, x: f64, y: f64, z: f64) -> f64 {
    let x0 = (x.floor() as isize).clamp(0, nx as isize - 1) as usize;
    let y0 = (y.floor() as isize).clamp(0, ny as isize - 1) as usize;
    let z0 = (z.floor() as isize).clamp(0, nz as isize - 1) as usize;
    let x1 = (x0 + 1).min(nx - 1);
    let y1 = (y0 + 1).min(ny - 1);
    let z1 = (z0 + 1).min(nz - 1);
    let (fx, fy, fz) = (x - x0 as f64, y - y0 as f64, z - z0 as f64);
    let idx = |a: usize, b: usize, c: usize| a + b * nx + c * nx * ny;

    data[idx(x0, y0, z0)] * (1.0 - fx) * (1.0 - fy) * (1.0 - fz)
        + data[idx(x1, y0, z0)] * fx * (1.0 - fy) * (1.0 - fz)
        + data[idx(x0, y1, z0)] * (1.0 - fx) * fy * (1.0 - fz)
        + data[idx(x1, y1, z0)] * fx * fy * (1.0 - fz)
        + data[idx(x0, y0, z1)] * (1.0 - fx) * (1.0 - fy) * fz
        + data[idx(x1, y0, z1)] * fx * (1.0 - fy) * fz
        + data[idx(x0, y1, z1)] * (1.0 - fx) * fy * fz
        + data[idx(x1, y1, z1)] * fx * fy * fz
}

/// A volume resampled onto a cardinal-aligned grid.
pub struct ResampledVolume {
    pub data: Vec<f64>,
    pub dims: (usize, usize, usize),
    pub voxel_size: (f64, f64, f64),
    pub affine: [f64; 16],
}

/// Magnitude and phase resampled together onto a cardinal-aligned grid.
pub struct ResampledComplex {
    pub magnitude: Vec<f64>,
    /// Wrapped phase in radians, on `(-π, π]`.
    pub phase: Vec<f64>,
    pub dims: (usize, usize, usize),
    pub voxel_size: (f64, f64, f64),
    pub affine: [f64; 16],
}

/// Options for [`resample_complex_to_axial`].
#[derive(Debug, Clone, Copy)]
pub struct AxialResampleParams {
    /// Replace exactly-zero phase with uniform noise on `(-π, π]` when at least this fraction of
    /// the output is zero. Resampling an oblique volume leaves empty corners, and a large block of
    /// identical zeros is not something a region-growing unwrapper handles gracefully. `None`
    /// disables the fill. Default `Some(0.1)`, matching QSMxT 8.x.
    pub noise_fill_fraction: Option<f64>,
}

impl Default for AxialResampleParams {
    fn default() -> Self {
        Self { noise_fill_fraction: Some(0.1) }
    }
}

/// Resample a continuous volume (magnitude, unwrapped field, χ) to a cardinal-aligned grid.
///
/// **Not for wrapped phase** — see the module docs and [`resample_complex_to_axial`].
pub fn resample_to_axial(
    data: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    affine: &[f64; 16],
) -> ResampledVolume {
    let grid = axial_grid_for(nx, ny, nz, affine);
    ResampledVolume {
        data: grid.sample(data),
        dims: grid.dims,
        voxel_size: grid.voxel_size,
        affine: grid.affine,
    }
}

/// Resample a binary mask to a cardinal-aligned grid with nearest-neighbour interpolation.
pub fn resample_mask_to_axial(
    mask: &[u8],
    nx: usize,
    ny: usize,
    nz: usize,
    affine: &[f64; 16],
) -> Vec<u8> {
    axial_grid_for(nx, ny, nz, affine).sample_nearest(mask)
}

/// Resample magnitude and wrapped phase to a cardinal-aligned grid, interpolating in the complex
/// domain so wraps survive.
///
/// `mag·cos φ` and `mag·sin φ` are interpolated separately and recombined, which is continuous
/// across the `±π` boundary. The returned affine is diagonal, so the dipole kernel can be built
/// with B0 = `(0, 0, 1)`.
///
/// # Panics
/// If `magnitude` and `phase` differ in length or do not match `nx · ny · nz`.
pub fn resample_complex_to_axial(
    magnitude: &[f64],
    phase: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    affine: &[f64; 16],
    params: &AxialResampleParams,
) -> ResampledComplex {
    let n = nx * ny * nz;
    assert_eq!(magnitude.len(), n, "magnitude length does not match dimensions");
    assert_eq!(phase.len(), n, "phase length does not match dimensions");

    let real: Vec<f64> = (0..n).map(|i| magnitude[i] * phase[i].cos()).collect();
    let imag: Vec<f64> = (0..n).map(|i| magnitude[i] * phase[i].sin()).collect();

    let grid = axial_grid_for(nx, ny, nz, affine);
    let real_r = grid.sample(&real);
    let imag_r = grid.sample(&imag);

    let out_n = real_r.len();
    let mut mag_out = Vec::with_capacity(out_n);
    let mut pha_out = Vec::with_capacity(out_n);
    for i in 0..out_n {
        mag_out.push(real_r[i].hypot(imag_r[i]));
        pha_out.push(imag_r[i].atan2(real_r[i]));
    }

    if let Some(frac) = params.noise_fill_fraction {
        let zeros = pha_out.iter().filter(|p| **p == 0.0).count();
        if out_n > 0 && (zeros as f64 / out_n as f64) >= frac {
            // Seeded from the data so a rerun on the same input gives the same volume.
            let mut rng = SplitMix64::new(seed_from_slice(&pha_out));
            for p in pha_out.iter_mut() {
                if *p == 0.0 {
                    *p = rng.uniform_signed_pi();
                }
            }
        }
    }

    ResampledComplex {
        magnitude: mag_out,
        phase: pha_out,
        dims: grid.dims,
        voxel_size: grid.voxel_size,
        affine: grid.affine,
    }
}

/// Deterministic seed from volume contents (FNV-1a over the bit patterns).
fn seed_from_slice(data: &[f64]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in data {
        for b in v.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h | 1
}

/// SplitMix64 — small, dependency-free, and adequate for filling empty corners with noise.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    /// Uniform on `(-π, π]`.
    fn uniform_signed_pi(&mut self) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64; // [0, 1)
        (u - 0.5) * 2.0 * std::f64::consts::PI
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Row-major affine from a 3×3 and a translation.
    fn affine_from(r: [[f64; 3]; 3], t: [f64; 3]) -> [f64; 16] {
        [
            r[0][0], r[0][1], r[0][2], t[0],
            r[1][0], r[1][1], r[1][2], t[1],
            r[2][0], r[2][1], r[2][2], t[2],
            0.0, 0.0, 0.0, 1.0,
        ]
    }

    fn identity_affine(vs: (f64, f64, f64)) -> [f64; 16] {
        affine_from([[vs.0, 0.0, 0.0], [0.0, vs.1, 0.0], [0.0, 0.0, vs.2]], [0.0, 0.0, 0.0])
    }

    /// A real UK Biobank SWI affine: 0.8 × 0.8 × 3 mm, header "Tra>Cor(-22.9)>Sag(-2.1)".
    fn ukb_swi_affine() -> [f64; 16] {
        affine_from(
            [
                [0.7976, -0.0273, -0.1075],
                [0.0141, 0.7358, -1.1647],
                [0.0370, 0.3092, 2.7626],
            ],
            [-101.6, -87.4, -60.2],
        )
    }

    #[test]
    fn voxel_sizes_are_column_norms() {
        let (sx, sy, sz) = voxel_sizes_from_affine(&ukb_swi_affine());
        assert!((sx - 0.799).abs() < 0.002, "{sx}");
        assert!((sy - 0.799).abs() < 0.002, "{sy}");
        assert!((sz - 3.0).abs() < 0.002, "{sz}");
    }

    #[test]
    fn b0_direction_axial_is_z() {
        for vs in [(1.0, 1.0, 1.0), (0.8, 0.8, 3.0), (2.0, 0.5, 1.0)] {
            let (bx, by, bz) = b0_direction_from_affine(&identity_affine(vs));
            assert!(bx.abs() < 1e-9 && by.abs() < 1e-9, "{vs:?} -> {bx} {by}");
            assert!((bz - 1.0).abs() < 1e-9, "{vs:?} -> {bz}");
            assert!(b0_angle_from_affine(&identity_affine(vs)) < 1e-6);
        }
    }

    /// The regression that matters: with anisotropic voxels, factoring the voxel scaling out of
    /// the affine is not optional. Inverting the scaled matrix instead (`A⁻¹·(0,0,1)`) is the
    /// QSMxT 8.2.2 bug, and on this acquisition it is 35° away from the right answer.
    #[test]
    fn b0_direction_anisotropic_oblique() {
        let a = ukb_swi_affine();
        let (bx, by, bz) = b0_direction_from_affine(&a);
        // Reference value from nibabel: pure_rotation.T @ [0, 0, 1].
        assert!((bx - 0.0463).abs() < 1e-3, "bx={bx}");
        assert!((by - 0.3871).abs() < 1e-3, "by={by}");
        assert!((bz - 0.9209).abs() < 1e-3, "bz={bz}");

        // The tilt matches the acquisition's stated 22.9° obliquity.
        let angle = b0_angle_from_affine(&a);
        assert!((angle - 22.9).abs() < 0.5, "angle={angle}");

        // And is nowhere near what inverting the scaled matrix would give (≈58° off axis).
        let naive = {
            let r = [[a[0], a[1], a[2]], [a[4], a[5], a[6]], [a[8], a[9], a[10]]];
            let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
                - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
                + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
            let v = [
                (r[0][1] * r[1][2] - r[0][2] * r[1][1]) / det,
                (r[0][2] * r[1][0] - r[0][0] * r[1][2]) / det,
                (r[0][0] * r[1][1] - r[0][1] * r[1][0]) / det,
            ];
            let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            [v[0] / n, v[1] / n, v[2] / n]
        };
        let dot = (bx * naive[0] + by * naive[1] + bz * naive[2]).abs().clamp(0.0, 1.0);
        let between = dot.acos().to_degrees();
        assert!(between > 30.0, "the two differ by {between}°, expected >30");
    }

    /// With isotropic voxels the scaling cancels, so the bug above is invisible — which is why it
    /// survived: every isotropic test passes either way.
    #[test]
    fn b0_direction_isotropic_oblique_agrees_with_naive() {
        let (c, s) = (30.0f64.to_radians().cos(), 30.0f64.to_radians().sin());
        let a = affine_from([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], [0.0; 3]);
        let (bx, by, bz) = b0_direction_from_affine(&a);
        assert!(bx.abs() < 1e-9, "{bx}");
        assert!((by - s).abs() < 1e-9, "{by} vs {s}");
        assert!((bz - c).abs() < 1e-9, "{bz} vs {c}");
        assert!((b0_angle_from_affine(&a) - 30.0).abs() < 1e-6);
    }

    #[test]
    fn obliquity_matches_nibabel() {
        // nibabel.affines.obliquity on the same affine, in degrees.
        let axes = obliquity_axes_from_affine(&ukb_swi_affine());
        for (got, want) in axes.iter().zip([2.838, 22.869, 22.947]) {
            assert!((got - want).abs() < 0.05, "{axes:?}");
        }
        let norm = obliquity_from_affine(&ukb_swi_affine());
        assert!((norm - 32.521).abs() < 0.05, "{norm}");
        // Cardinal acquisitions score zero whatever the voxel sizes.
        assert!(obliquity_from_affine(&identity_affine((0.8, 0.8, 3.0))) < 1e-9);
    }

    #[test]
    fn axial_grid_of_an_axial_volume_is_the_same_grid() {
        let a = identity_affine((1.0, 1.0, 2.0));
        let g = axial_grid_for(4, 5, 6, &a);
        assert_eq!(g.dims, (4, 5, 6));
        let (vx, vy, vz) = g.voxel_size;
        assert!((vx - 1.0).abs() < 1e-9 && (vy - 1.0).abs() < 1e-9 && (vz - 2.0).abs() < 1e-9);
        // Resampling is then an identity.
        let data: Vec<f64> = (0..4 * 5 * 6).map(|i| i as f64).collect();
        let out = resample_to_axial(&data, 4, 5, 6, &a);
        for (i, (got, want)) in out.data.iter().zip(data.iter()).enumerate() {
            assert!((got - want).abs() < 1e-6, "voxel {i}: {got} vs {want}");
        }
    }

    #[test]
    fn resampled_affine_is_cardinal_and_b0_is_z() {
        let g = axial_grid_for(32, 32, 12, &ukb_swi_affine());
        for (i, v) in g.affine.iter().enumerate() {
            let diagonal = matches!(i, 0 | 5 | 10 | 15);
            let translation = matches!(i, 3 | 7 | 11);
            if !diagonal && !translation {
                assert!(v.abs() < 1e-12, "affine[{i}] = {v} should be 0");
            }
        }
        assert!(b0_angle_from_affine(&g.affine) < 1e-9);
        assert!(obliquity_from_affine(&g.affine) < 1e-9);
        // The oblique volume needs a larger cardinal box to fit inside.
        assert!(g.dims.0 >= 32 && g.dims.1 >= 32 && g.dims.2 >= 12, "{:?}", g.dims);
    }

    /// Interpolating wrapped phase directly is wrong at every wrap; going through the complex
    /// representation is not. Build a ramp that wraps many times and compare both.
    #[test]
    fn complex_resampling_survives_wraps_where_scalar_does_not() {
        let (nx, ny, nz) = (24, 24, 8);
        let n = nx * ny * nz;
        // Oblique enough to force real interpolation, isotropic so the true phase is easy to state.
        let (c, s) = (20.0f64.to_radians().cos(), 20.0f64.to_radians().sin());
        let a = affine_from([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], [0.0; 3]);

        // Smooth field, wrapping ~6 times across x.
        let true_phase = |x: f64, y: f64, _z: f64| 0.8 * x + 0.3 * y;
        let mut mag = vec![0.0; n];
        let mut pha = vec![0.0; n];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = i + j * nx + k * nx * ny;
                    mag[idx] = 100.0;
                    pha[idx] = wrap(true_phase(i as f64, j as f64, k as f64));
                }
            }
        }
        // Count real wrap discontinuities along x (adjacent voxels jumping by more than π).
        let wraps = (0..nz)
            .flat_map(|k| (0..ny).flat_map(move |j| (1..nx).map(move |i| (i, j, k))))
            .filter(|&(i, j, k)| {
                let idx = i + j * nx + k * nx * ny;
                (pha[idx] - pha[idx - 1]).abs() > PI
            })
            .count();
        assert!(wraps > 500, "test data should contain plenty of wraps, found {wraps}");

        let cpx = resample_complex_to_axial(&mag, &pha, nx, ny, nz, &a, &AxialResampleParams { noise_fill_fraction: None });
        let scalar = resample_to_axial(&pha, nx, ny, nz, &a);
        let grid = axial_grid_for(nx, ny, nz, &a);

        // Compare both against the analytic phase at each interior sample point.
        let (mut cpx_err, mut scalar_err, mut count) = (0.0f64, 0.0f64, 0usize);
        let (dx, dy, dz) = grid.dims;
        for k in 0..dz {
            for j in 0..dy {
                for i in 0..dx {
                    let Some((ox, oy, oz)) = grid.source_coord(i, j, k) else { continue };
                    // Interior only: edge voxels mix in the zero background.
                    if ox < 1.0 || ox > nx as f64 - 2.0 || oy < 1.0 || oy > ny as f64 - 2.0 || oz < 1.0 || oz > nz as f64 - 2.0 {
                        continue;
                    }
                    let idx = i + j * dx + k * dx * dy;
                    let want = wrap(true_phase(ox, oy, oz));
                    cpx_err += wrap(cpx.phase[idx] - want).abs();
                    scalar_err += wrap(scalar.data[idx] - want).abs();
                    count += 1;
                }
            }
        }
        assert!(count > 500, "too few interior samples: {count}");
        let (cpx_err, scalar_err) = (cpx_err / count as f64, scalar_err / count as f64);
        assert!(cpx_err < 0.05, "complex resampling error {cpx_err} rad is too high");
        assert!(
            scalar_err > 10.0 * cpx_err,
            "scalar {scalar_err} should be far worse than complex {cpx_err}"
        );
        // Magnitude comes through nearly intact. It dips slightly rather than exactly matching,
        // because averaging two phasors that differ by Δφ scales the result by cos(Δφ/2) — here
        // the ramp is 0.8 rad/voxel, so a midpoint sample can lose up to 1 − cos(0.4) ≈ 8%. That
        // is inherent to complex interpolation, not a defect, and it is the price of keeping the
        // wraps; an unwrapped field would not pay it.
        for (idx, m) in cpx.magnitude.iter().enumerate() {
            if grid_interior(&grid, idx) {
                assert!(*m > 90.0 && *m < 100.5, "magnitude {m} at {idx} outside the expected dip");
            }
        }
    }

    fn grid_interior(grid: &AxialGrid, idx: usize) -> bool {
        let (dx, dy, _) = grid.dims;
        let i = idx % dx;
        let j = (idx / dx) % dy;
        let k = idx / (dx * dy);
        grid.source_coord(i, j, k)
            .map(|(ox, oy, oz)| {
                let (nx, ny, nz) = grid.src_dims;
                ox > 1.0 && ox < nx as f64 - 2.0 && oy > 1.0 && oy < ny as f64 - 2.0 && oz > 1.0 && oz < nz as f64 - 2.0
            })
            .unwrap_or(false)
    }

    fn wrap(p: f64) -> f64 {
        let mut v = (p + PI) % (2.0 * PI);
        if v < 0.0 {
            v += 2.0 * PI;
        }
        v - PI
    }

    // --- GridMap: the return trip ---

    #[test]
    fn round_trip_to_axial_and_back_recovers_the_original() {
        let (nx, ny, nz) = (20, 22, 10);
        let n = nx * ny * nz;
        let a = ukb_swi_affine();
        // A smooth field, so interpolation error is the only thing being measured.
        let data: Vec<f64> = (0..n)
            .map(|i| {
                let (x, y, z) = (i % nx, (i / nx) % ny, i / (nx * ny));
                (x as f64 * 0.1).sin() + (y as f64 * 0.07).cos() + z as f64 * 0.02
            })
            .collect();

        let grid = axial_grid_for(nx, ny, nz, &a);
        let there = resample_to_axial(&data, nx, ny, nz, &a);
        let back = resample_onto(&there.data, grid.dims, &grid.affine, (nx, ny, nz), &a).unwrap();

        // Interior voxels come back close; edges lose data to the empty corners, as expected of
        // two interpolations.
        let (mut err, mut count) = (0.0f64, 0usize);
        for z in 2..nz - 2 {
            for y in 2..ny - 2 {
                for x in 2..nx - 2 {
                    let i = x + y * nx + z * nx * ny;
                    err += (back[i] - data[i]).abs();
                    count += 1;
                }
            }
        }
        let mean = err / count as f64;
        assert!(mean < 0.05, "round trip lost {mean} per voxel on a smooth field");
    }

    #[test]
    fn round_trip_puts_the_data_back_on_the_original_grid() {
        let (nx, ny, nz) = (16, 16, 8);
        let a = ukb_swi_affine();
        let grid = axial_grid_for(nx, ny, nz, &a);
        assert_ne!(grid.dims, (nx, ny, nz), "the axial grid should differ, else this proves nothing");
        let there = vec![1.0f64; grid.dims.0 * grid.dims.1 * grid.dims.2];
        let back = resample_onto(&there, grid.dims, &grid.affine, (nx, ny, nz), &a).unwrap();
        assert_eq!(back.len(), nx * ny * nz, "output must be on the acquired grid");
        // The acquired volume sits inside the axial box, so everything is covered.
        assert!(back.iter().filter(|v| **v > 0.5).count() > (nx * ny * nz) * 9 / 10);
    }

    #[test]
    fn identity_map_is_a_no_op() {
        let (nx, ny, nz) = (6, 5, 4);
        let a = identity_affine((1.0, 2.0, 3.0));
        let data: Vec<f64> = (0..nx * ny * nz).map(|i| i as f64).collect();
        let out = resample_onto(&data, (nx, ny, nz), &a, (nx, ny, nz), &a).unwrap();
        for (got, want) in out.iter().zip(data.iter()) {
            assert!((got - want).abs() < 1e-9, "{got} vs {want}");
        }
    }

    #[test]
    fn singular_affine_is_reported_not_panicked() {
        let zero = [0.0f64; 16];
        let a = identity_affine((1.0, 1.0, 1.0));
        assert!(resample_onto(&[0.0; 8], (2, 2, 2), &zero, (2, 2, 2), &a).is_none());
        assert!(GridMap::new((2, 2, 2), &zero, (2, 2, 2), &a).is_none());
    }

    #[test]
    fn mask_round_trip_stays_binary() {
        let (nx, ny, nz) = (16, 16, 8);
        let a = ukb_swi_affine();
        let mut mask = vec![0u8; nx * ny * nz];
        for z in 2..nz - 2 {
            for y in 4..ny - 4 {
                for x in 4..nx - 4 {
                    mask[x + y * nx + z * nx * ny] = 1;
                }
            }
        }
        let grid = axial_grid_for(nx, ny, nz, &a);
        let there = resample_mask_to_axial(&mask, nx, ny, nz, &a);
        let back = resample_mask_onto(&there, grid.dims, &grid.affine, (nx, ny, nz), &a).unwrap();
        assert!(back.iter().all(|v| *v == 0 || *v == 1));
        assert!(back.iter().filter(|v| **v == 1).count() > 100, "mask should survive the trip");
    }

    /// Phase has to come back through the complex domain too, for the same reason it went out
    /// that way.
    #[test]
    fn complex_round_trip_preserves_wraps() {
        let (nx, ny, nz) = (20, 20, 8);
        let n = nx * ny * nz;
        let a = ukb_swi_affine();
        let mag = vec![100.0f64; n];
        let pha: Vec<f64> = (0..n)
            .map(|i| {
                let (x, y) = (i % nx, (i / nx) % ny);
                wrap(0.6 * x as f64 + 0.2 * y as f64)
            })
            .collect();

        let grid = axial_grid_for(nx, ny, nz, &a);
        let out = resample_complex_to_axial(&mag, &pha, nx, ny, nz, &a, &AxialResampleParams { noise_fill_fraction: None });
        let (_, back) = resample_complex_onto(&out.magnitude, &out.phase, grid.dims, &grid.affine, (nx, ny, nz), &a).unwrap();

        let (mut err, mut count) = (0.0f64, 0usize);
        for z in 2..nz - 2 {
            for y in 3..ny - 3 {
                for x in 3..nx - 3 {
                    let i = x + y * nx + z * nx * ny;
                    err += wrap(back[i] - pha[i]).abs();
                    count += 1;
                }
            }
        }
        let mean = err / count as f64;
        assert!(mean < 0.25, "complex round trip drifted {mean} rad per voxel");
    }

    #[test]
    fn noise_fill_is_deterministic_and_bounded() {
        let (nx, ny, nz) = (16, 16, 6);
        let n = nx * ny * nz;
        let (c, s) = (25.0f64.to_radians().cos(), 25.0f64.to_radians().sin());
        let a = affine_from([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], [0.0; 3]);
        let mag = vec![50.0; n];
        let pha: Vec<f64> = (0..n).map(|i| wrap(0.4 * (i % nx) as f64)).collect();

        let p = AxialResampleParams::default();
        let a1 = resample_complex_to_axial(&mag, &pha, nx, ny, nz, &a, &p);
        let a2 = resample_complex_to_axial(&mag, &pha, nx, ny, nz, &a, &p);
        assert_eq!(a1.phase, a2.phase, "same input must give the same volume");
        assert!(a1.phase.iter().all(|v| v.is_finite() && v.abs() <= PI + 1e-9));

        // Without the fill, the empty corners stay at exactly zero.
        let off = resample_complex_to_axial(&mag, &pha, nx, ny, nz, &a, &AxialResampleParams { noise_fill_fraction: None });
        assert!(off.phase.iter().any(|v| *v == 0.0));
        assert!(off.phase.iter().filter(|v| **v == 0.0).count() > a1.phase.iter().filter(|v| **v == 0.0).count());
    }

    #[test]
    fn mask_resampling_stays_binary() {
        let (nx, ny, nz) = (12, 12, 6);
        let (c, s) = (15.0f64.to_radians().cos(), 15.0f64.to_radians().sin());
        let a = affine_from([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], [0.0; 3]);
        let mut mask = vec![0u8; nx * ny * nz];
        for k in 1..nz - 1 {
            for j in 3..ny - 3 {
                for i in 3..nx - 3 {
                    mask[i + j * nx + k * nx * ny] = 1;
                }
            }
        }
        let out = resample_mask_to_axial(&mask, nx, ny, nz, &a);
        assert!(out.iter().all(|v| *v == 0 || *v == 1), "mask must stay binary");
        assert!(out.iter().filter(|v| **v == 1).count() > 100, "mask should survive");
    }
}

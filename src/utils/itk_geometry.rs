//! Image geometry as ITK holds it, for reproducing ITK filters whose output depends on it.
//!
//! [`ItkGeometry`] is an image's origin, spacing and direction cosines in ITK's LPS convention,
//! as ITK 5.3's `NiftiImageIO` (the ITK in ANTs 2.4.3) derives them from a NIfTI-1 header
//! ([`ItkGeometry::from_nifti_header`]), or as QSMART's `make_nii`/`save_nii` writes them for a
//! voxel size ([`ItkGeometry::make_nii`]).
//!
//! The one use so far is the shrink offset of ITK's `ShrinkImageFilter`, which N4 shrinks its
//! input with ([`ItkGeometry::shrink_offsets`], used by
//! [`n4_bias_correction`](crate::utils::n4_bias_correction)). ITK finds that
//! offset by mapping the first shrunk voxel's physical point back to an input index and rounding.
//! Along an even axis the exact value is a half-integer `k + ½`, so the rounding goes whichever
//! way the floating-point arithmetic leaves it. A scanner header whose direction cosines carry a
//! 4.9e-12 off-diagonal term (in-vivo Prisma, QSMxT/QSM-CI#312) moves one axis's sample grid by a
//! voxel, which changes N4's output by 3 % (NRMSE). Every step of ITK's arithmetic is
//! reproduced here in the same order, in double precision, including the 3×3 inverse ITK takes
//! with VNL's `vnl_svd` (LINPACK `dsvdc`, ported line for line here, and bit-identical to VNL's
//! compiled C on the 52 matrices of `tests/data/vnl_inverse_3x3.txt`).
//!
//! Sources (ITK v5.3.0): `Modules/IO/NIFTI/src/itkNiftiImageIO.cxx`
//! (`SetImageIOOrientationFromNIfTI`, `Normalize`, `ReadImageInformation`),
//! `Modules/ThirdParty/NIFTI/src/nifti/niftilib/nifti1_io.c` (`nifti_convert_nhdr2nim`,
//! `nifti_quatern_to_mat44`), `Modules/Core/Common/include/itkImageBase.{h,hxx}`
//! (`ComputeIndexToPhysicalPointMatrices`, `TransformPhysicalPointToIndex`,
//! `TransformContinuousIndexToPhysicalPoint`), `Modules/Core/Common/include/itkMatrix.h`
//! (`GetInverse`: `vnl_matrix_inverse`), `Modules/Filtering/ImageGrid/include/itkShrinkImageFilter.hxx`,
//! `Modules/ThirdParty/VNL/src/vxl/core/vnl/algo/vnl_svd.hxx` and
//! `Modules/ThirdParty/VNL/src/vxl/v3p/netlib/linpack/dsvdc.c` with its BLAS
//! (`dnrm2`, `drotg`, `drot`, `ddot`, `daxpy`, `dscal`, `dswap`).

use crate::error::InputError;

/// An image's geometry as ITK holds it: physical point of index `i` is
/// `origin + direction · diag(spacing) · i`, in LPS millimetres.
#[derive(Clone, Debug, PartialEq)]
pub struct ItkGeometry {
    /// Physical point of the first voxel (ITK's `GetOrigin`).
    pub origin: [f64; 3],
    /// Voxel spacing (ITK's `GetSpacing`).
    pub spacing: [f64; 3],
    /// Direction cosines, `direction[row][col]`: column `j` is the direction of index axis `j`
    /// (ITK's `GetDirection`).
    pub direction: [[f64; 3]; 3],
}

const NIFTI_XFORM_UNKNOWN: i16 = 0;
const NIFTI_XFORM_SCANNER_ANAT: i16 = 1;

/// A NIfTI-1 `mat44` (single precision, as `nifti1_io` keeps `qto_xyz`/`sto_xyz`).
type Mat44 = [[f32; 4]; 4];

impl ItkGeometry {
    /// The geometry ITK gives an image with the identity direction, zero origin and this spacing.
    pub fn axis_aligned(spacing: [f64; 3]) -> Self {
        Self { origin: [0.0; 3], spacing, direction: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] }
    }

    /// The geometry ITK reads from a NIfTI file written by Jimmy Shen's `make_nii(img, vox)` and
    /// `save_nii` (the NIfTI toolbox QSMART uses for every file it hands to ANTs): `pixdim` is
    /// `vox` in single precision, `qform_code` 0 and `sform_code` 1 with `srow` = `diag(vox)` and
    /// offsets `(1 − 0)·vox` (`save_nii_hdr.m:15-28`, origin `[0 0 0]`). In ITK's LPS convention
    /// that is origin `(−vx, −vy, vz)` and direction `diag(−1, −1, 1)`.
    pub fn make_nii(voxel_size: (f64, f64, f64)) -> Self {
        let v = [voxel_size.0 as f32, voxel_size.1 as f32, voxel_size.2 as f32];
        let mut sto: Mat44 = [[0.0; 4]; 4];
        for i in 0..3 {
            sto[i][i] = v[i];
            sto[i][3] = v[i];
        }
        sto[3][3] = 1.0;
        Self::from_mat44(&sto, [v[0] as f64, v[1] as f64, v[2] as f64])
    }

    /// The geometry ITK 5.3's `NiftiImageIO` reads from a NIfTI-1 header (ANTs 2.4.3), for a
    /// 3-D image in millimetres.
    ///
    /// As ITK does: with neither a qform nor an sform, origin 0 and the identity direction.
    /// Otherwise the sform is used if the qform's and sform's matrices agree (to 1e-5 in the
    /// rotation-scale part and 1e-7 in the translation), or if the sform is orthonormal (to
    /// 1e-4) and either there is no qform or the sform's code is `NIFTI_XFORM_SCANNER_ANAT`;
    /// else the qform (`nifti_quatern_to_mat44`). The origin is the chosen matrix's translation
    /// with x and y negated (RAS to LPS), each direction is its column with x and y negated,
    /// normalised as ITK's `Normalize` does, and the spacing is `pixdim[1..4]` (0 or non-finite
    /// read as 1, as `nifti1_io` does).
    ///
    /// The case ITK resolves with an SVD of single-precision matrices (both forms present and
    /// different, the sform orthonormal but not `SCANNER_ANAT`) is reproduced with the
    /// double-precision SVD here; its tolerance (1e-4) is far above the difference.
    ///
    /// # Errors
    /// If the header has no usable orientation: ITK refuses a qform-less header whose sform is
    /// not orthonormal, and so does this.
    pub fn from_nifti_header(h: &nifti::NiftiHeader) -> Result<Self, InputError> {
        // nifti_convert_nhdr2nim: pixdim[1..=dim[0]] that are 0 or not finite become 1.
        let mut pixdim = h.pixdim;
        let nd = (h.dim[0] as usize).min(7);
        for p in pixdim.iter_mut().take(nd + 1).skip(1) {
            if *p == 0.0 || !p.is_finite() {
                *p = 1.0;
            }
        }
        let (dx, dy, dz) = (pixdim[1], pixdim[2], pixdim[3]);
        let spacing = [dx as f64, dy as f64, dz as f64];
        let fixed = |x: f32| if x.is_finite() { x } else { 0.0 };
        let mut qform_code = h.qform_code;
        let qto: Mat44 = if qform_code <= 0 {
            qform_code = NIFTI_XFORM_UNKNOWN;
            let mut m = [[0.0f32; 4]; 4];
            m[0][0] = dx;
            m[1][1] = dy;
            m[2][2] = dz;
            m[3][3] = 1.0;
            m
        } else {
            let qfac = if pixdim[0] < 0.0 { -1.0 } else { 1.0 };
            quatern_to_mat44(
                fixed(h.quatern_b), fixed(h.quatern_c), fixed(h.quatern_d),
                fixed(h.quatern_x), fixed(h.quatern_y), fixed(h.quatern_z),
                dx, dy, dz, qfac,
            )
        };
        let sform_code = if h.sform_code <= 0 { NIFTI_XFORM_UNKNOWN } else { h.sform_code };
        let mut sto: Mat44 = [[0.0; 4]; 4];
        if sform_code != NIFTI_XFORM_UNKNOWN {
            sto[0] = h.srow_x;
            sto[1] = h.srow_y;
            sto[2] = h.srow_z;
            sto[3] = [0.0, 0.0, 0.0, 1.0];
        }
        if qform_code == NIFTI_XFORM_UNKNOWN && sform_code == NIFTI_XFORM_UNKNOWN {
            return Ok(Self::axis_aligned(spacing));
        }
        let similar = {
            let rot = (0..3).all(|i| (0..3).all(|j| (sto[i][j] - qto[i][j]).abs() <= 1e-5));
            let tr: f32 = (0..4).map(|i| (sto[i][3] - qto[i][3]).abs()).sum();
            let bottom: f32 = (0..4).map(|j| (sto[3][j] - qto[3][j]).abs()).sum();
            rot && tr <= 1e-7 && bottom <= 1e-7
        };
        let mut prefer_sform = similar;
        if (!prefer_sform || sform_code != NIFTI_XFORM_UNKNOWN) && sform_orthonormal(&sto) {
            // An orthonormal sform wins if there is no qform or it is SCANNER_ANAT; with both
            // forms present otherwise, only if they describe the same geometry.
            if (qform_code == NIFTI_XFORM_UNKNOWN && sform_code != NIFTI_XFORM_UNKNOWN)
                || sform_code == NIFTI_XFORM_SCANNER_ANAT
            {
                prefer_sform = true;
            } else if qform_code != NIFTI_XFORM_UNKNOWN && sform_code != NIFTI_XFORM_UNKNOWN {
                prefer_sform = forms_very_similar(&sto, &qto);
            }
        }
        let mat = if prefer_sform {
            sto
        } else if qform_code != NIFTI_XFORM_UNKNOWN {
            qto
        } else {
            return Err(InputError::new(
                "ITK only supports orthonormal direction cosines: the header has no qform and its \
                 sform is not orthonormal",
            ));
        };
        Ok(Self::from_mat44(&mat, spacing))
    }

    /// Origin and direction from a chosen `mat44`, as `SetImageIOOrientationFromNIfTI` ends.
    fn from_mat44(m: &Mat44, spacing: [f64; 3]) -> Self {
        let origin = [-(m[0][3] as f64), -(m[1][3] as f64), m[2][3] as f64];
        let mut direction = [[0.0f64; 3]; 3];
        for col in 0..3 {
            let mut v = [m[0][col] as f64, m[1][col] as f64, m[2][col] as f64];
            v[0] *= -1.0;
            v[1] *= -1.0;
            // ITK's Normalize: sum of squares in order, sqrt, divide.
            let mut sum = 0.0f64;
            for x in v {
                sum += x * x;
            }
            if sum != 0.0 {
                let s = sum.sqrt();
                for x in v.iter_mut() {
                    *x /= s;
                }
            }
            for row in 0..3 {
                direction[row][col] = v[row];
            }
        }
        Self { origin, spacing, direction }
    }

    /// The input index of the first sample ITK's `ShrinkImageFilter` takes along each axis when
    /// it shrinks an image of `dims` voxels with this geometry by `shrink_factor` (the `offsetIndex`
    /// of `DynamicThreadedGenerateData`), computed as ITK computes it:
    ///
    /// 1. The output has `m = max(1, ⌊n/s⌋)` voxels of spacing `s·spacing`, the input's direction,
    ///    and an origin chosen so the centres coincide: `origin + (c_in − c_out)`, where `c_in` and
    ///    `c_out` are the physical points of the continuous centre indices `(n−1)/2` and `(m−1)/2`,
    ///    the latter on a grid with the input's origin (`GenerateOutputInformation`).
    /// 2. The output origin is mapped back to an input index with the inverse of
    ///    `direction · diag(spacing)` (VNL's SVD inverse) and rounded half up
    ///    (`Math::RoundHalfIntegerUp`), and the offset is that index, at least 0.
    ///
    /// For an axis-aligned geometry whose origin and spacing are exact in binary the result is
    /// `round((n−1)/2 − s·(m−1)/2)` with halves rounded up, which is what
    /// [`ItkGeometry::axis_aligned`] gives.
    ///
    /// # Errors
    /// If `shrink_factor` is 0 or the direction · spacing matrix is singular.
    pub fn shrink_offsets(&self, dims: (usize, usize, usize), shrink_factor: usize) -> Result<[usize; 3], InputError> {
        if shrink_factor == 0 {
            return Err(InputError::param("shrink_factor", 0, ">= 1"));
        }
        let n = [dims.0, dims.1, dims.2];
        let s = shrink_factor as f64;
        // ComputeIndexToPhysicalPointMatrices: Direction * diag(spacing), and its VNL inverse.
        let m_in = index_to_physical(&self.direction, &self.spacing);
        let p_in = vnl_inverse(&m_in)
            .ok_or_else(|| InputError::new("ITK geometry: direction · spacing is singular"))?;
        let out_spacing = [self.spacing[0] * s, self.spacing[1] * s, self.spacing[2] * s];
        let m_out = index_to_physical(&self.direction, &out_spacing);
        let mut c_in_idx = [0.0f64; 3];
        let mut c_out_idx = [0.0f64; 3];
        for d in 0..3 {
            let m = ((n[d] as f64 / s).floor() as usize).max(1);
            c_in_idx[d] = 0.0 + (n[d] as f64 - 1.0) / 2.0;
            c_out_idx[d] = 0.0 + (m as f64 - 1.0) / 2.0;
        }
        // TransformContinuousIndexToPhysicalPoint: sum over columns, then + origin.
        let to_point = |mat: &[[f64; 3]; 3], idx: &[f64; 3]| {
            let mut p = [0.0f64; 3];
            for r in 0..3 {
                let mut sum = 0.0f64;
                for c in 0..3 {
                    sum += mat[r][c] * idx[c];
                }
                p[r] = sum + self.origin[r];
            }
            p
        };
        let c_in = to_point(&m_in, &c_in_idx);
        let c_out = to_point(&m_out, &c_out_idx);
        let mut out_origin = [0.0f64; 3];
        for d in 0..3 {
            out_origin[d] = self.origin[d] + (c_in[d] - c_out[d]);
        }
        // TransformIndexToPhysicalPoint of output index 0: origin, then += M[i][j]·0.
        let mut point = out_origin;
        for (i, p) in point.iter_mut().enumerate() {
            for j in 0..3 {
                *p += m_out[i][j] * 0.0;
            }
        }
        // TransformPhysicalPointToIndex: sum of P[i][j]·(point[j] − origin[j]), rounded half up.
        let mut offs = [0usize; 3];
        for i in 0..3 {
            let mut sum = 0.0f64;
            for j in 0..3 {
                sum += p_in[i][j] * (point[j] - self.origin[j]);
            }
            let index = round_half_integer_up(sum);
            // offsetIndex = inputIndex − outputIndex·s (output index 0), at least 0.
            offs[i] = index.max(0) as usize;
        }
        Ok(offs)
    }
}

/// `itk::Math::RoundHalfIntegerUp` on x86-64: round-half-to-even of `2x + 0.5`, halved by an
/// arithmetic shift.
fn round_half_integer_up(x: f64) -> i64 {
    let y = 2.0 * x + 0.5;
    (y.round_ties_even() as i64) >> 1
}

/// `Direction * scale` (itk::Matrix product: accumulate over k in order).
fn index_to_physical(dir: &[[f64; 3]; 3], spacing: &[f64; 3]) -> [[f64; 3]; 3] {
    let mut scale = [[0.0f64; 3]; 3];
    for i in 0..3 {
        scale[i][i] = spacing[i];
    }
    matmul(dir, &scale)
}

/// vnl_matrix_fixed / vnl_matrix product: `accum = a[i][0]·b[0][j]`, then `+= a[i][k]·b[k][j]`.
fn matmul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut r = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut acc = a[i][0] * b[0][j];
            for k in 1..3 {
                acc += a[i][k] * b[k][j];
            }
            r[i][j] = acc;
        }
    }
    r
}

/// `itk::Matrix::GetInverse`: `vnl_matrix_inverse` = `vnl_svd(M).inverse()` =
/// `V · W⁻¹ · Uᵀ` (`pinverse`), with singular values ≤ 0 dropped (`zero_out_absolute(0)`).
/// `None` if the determinant is 0 (ITK throws).
fn vnl_inverse(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    if det3(m) == 0.0 {
        return None;
    }
    let (u, w, v) = svd3(m)?;
    let mut winv = [[0.0f64; 3]; 3];
    for k in 0..3 {
        winv[k][k] = if w[k].abs() <= 0.0 { 0.0 } else { 1.0 / w[k] };
    }
    let mut ut = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            ut[i][j] = u[j][i];
        }
    }
    Some(matmul(&matmul(&v, &winv), &ut))
}

/// `vnl_determinant` of a 3×3 (`vnl_det`): the rule of Sarrus as VNL writes it.
fn det3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * m[1][1] * m[2][2] - m[0][0] * m[2][1] * m[1][2] - m[1][0] * m[0][1] * m[2][2]
        + m[1][0] * m[2][1] * m[0][2]
        + m[2][0] * m[0][1] * m[1][2]
        - m[2][0] * m[1][1] * m[0][2]
}

/// `nifti_quatern_to_mat44` (nifti1_io.c). Its `1.0l`/`2.0l` literals make some steps long
/// double on x86-64; here they are double, which differs only when an intermediate needs more
/// than 53 bits and its double rounding lands differently (the qform is used only when the
/// sform is absent or disagrees).
#[allow(clippy::too_many_arguments)]
fn quatern_to_mat44(qb: f32, qc: f32, qd: f32, qx: f32, qy: f32, qz: f32, dx: f32, dy: f32, dz: f32, qfac: f32) -> Mat44 {
    let mut r = [[0.0f32; 4]; 4];
    r[3][3] = 1.0;
    let (mut b, mut c, mut d) = (qb as f64, qc as f64, qd as f64);
    let mut a = 1.0 - (b * b + c * c + d * d);
    if a < 1.0e-7 {
        a = 1.0 / (b * b + c * c + d * d).sqrt();
        b *= a;
        c *= a;
        d *= a;
        a = 0.0;
    } else {
        a = a.sqrt();
    }
    let xd = if dx > 0.0 { dx as f64 } else { 1.0 };
    let yd = if dy > 0.0 { dy as f64 } else { 1.0 };
    let mut zd = if dz > 0.0 { dz as f64 } else { 1.0 };
    if qfac < 0.0 {
        zd = -zd;
    }
    r[0][0] = ((a * a + b * b - c * c - d * d) * xd) as f32;
    r[0][1] = (2.0 * (b * c - a * d) * yd) as f32;
    r[0][2] = (2.0 * (b * d + a * c) * zd) as f32;
    r[1][0] = (2.0 * (b * c + a * d) * xd) as f32;
    r[1][1] = ((a * a + c * c - b * b - d * d) * yd) as f32;
    r[1][2] = (2.0 * (c * d - a * b) * zd) as f32;
    r[2][0] = (2.0 * (b * d - a * c) * xd) as f32;
    r[2][1] = (2.0 * (c * d + a * b) * yd) as f32;
    r[2][2] = ((a * a + d * d - c * c - b * b) * zd) as f32;
    r[0][3] = qx;
    r[1][3] = qy;
    r[2][3] = qz;
    r
}

/// ITK's `sform_decomposable_without_skew`: the sform is affine (bottom row 0 0 0 1 to float
/// epsilon) and its columns, normalised, are orthonormal to 1e-4.
fn sform_orthonormal(sto: &Mat44) -> bool {
    // IsAffine: bottom row (0, 0, 0, 1) to float epsilon, and invertible (VNL's well_condition,
    // the smallest over the largest singular value, above double epsilon).
    let bottom = (sto[3][3] as f64 - 1.0).abs() + (0..3).map(|i| (sto[3][i] as f64).abs()).sum::<f64>();
    if bottom > f32::EPSILON as f64 {
        return false;
    }
    let mut top = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            top[i][j] = sto[i][j] as f64;
        }
    }
    match svd3(&top) {
        Some((_, w, _)) if w[0] > 0.0 && w[2] / w[0] > f64::EPSILON => {}
        _ => return false,
    }
    let mut r = [[0.0f32; 3]; 3];
    for col in 0..3 {
        let norm = (0..3).map(|i| sto[i][col] * sto[i][col]).sum::<f32>().sqrt();
        for i in 0..3 {
            r[i][col] = if norm != 0.0 { sto[i][col] / norm } else { sto[i][col] };
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            let v: f32 = (0..3).map(|k| r[i][k] * r[j][k]).sum();
            let want = if i == j { 1.0 } else { 0.0 };
            if (v - want).abs() > 1.0e-4 {
                return false;
            }
        }
    }
    true
}

/// ITK's `sform_and_qform_are_very_similar`: same rotation (`U_s · U_q⁻¹ ≈ I`), singular values,
/// offsets and bottom rows, each to 1e-4.
fn forms_very_similar(sto: &Mat44, qto: &Mat44) -> bool {
    let to3 = |m: &Mat44| {
        let mut r = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = m[i][j] as f64;
            }
        }
        r
    };
    let (Some((us, ws, _)), Some((uq, wq, _))) = (svd3(&to3(sto)), svd3(&to3(qto))) else {
        return false;
    };
    // U_q is orthogonal, so its inverse is its transpose.
    let mut cand = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            cand[i][j] = (0..3).map(|k| us[i][k] * uq[j][k]).sum();
        }
    }
    let ident = (0..3).all(|i| (0..3).all(|j| (cand[i][j] - if i == j { 1.0 } else { 0.0 }).abs() <= 1e-4));
    let spacing = (0..3).all(|k| (ws[k] - wq[k]).abs() <= 1e-4);
    let offsets = (0..3).all(|i| ((sto[i][3] - qto[i][3]) as f64).abs() <= 1e-4);
    let persp = (0..4).all(|j| ((sto[3][j] - qto[3][j]) as f64).abs() <= 1e-4);
    ident && spacing && offsets && persp
}

/// `(U, W, V)` of a 3×3 SVD.
pub(crate) type Svd3 = ([[f64; 3]; 3], [f64; 3], [[f64; 3]; 3]);

/// VNL's `vnl_svd<double>` of a 3×3 matrix: LINPACK `dsvdc` (v3p netlib f2c translation) with
/// `job = 21`, returning `(U, W, V)` with `M = U·diag(W)·Vᵀ`, `W` the absolute singular values
/// in decreasing order. `None` if `dsvdc` reports non-convergence (`info != 0`).
///
/// A line-for-line port: 1-based indices, the same loop orders and the BLAS level-1 routines as
/// netlib's reference loops (for `n ≤ 3` none of their unrolled paths is taken).
#[allow(clippy::many_single_char_names)]
pub(crate) fn svd3(mat: &[[f64; 3]; 3]) -> Option<Svd3> {
    const N: usize = 3;
    const P: usize = 3;
    // Column-major, 1-based: x[i + j*ld] with i, j in 1..=3 (index 0 of each column unused).
    let ld = N + 1;
    let idx = |i: usize, j: usize| i + j * ld;
    let mut x = vec![0.0f64; ld * (P + 1)];
    for i in 1..=N {
        for j in 1..=P {
            x[idx(i, j)] = mat[i - 1][j - 1];
        }
    }
    let mut s = [0.0f64; 5];
    let mut e = [0.0f64; 5];
    let mut u = vec![0.0f64; ld * (P + 1)];
    let mut v = vec![0.0f64; ld * (P + 1)];
    let mut work = [0.0f64; 5];

    fn dnrm2(xs: &[f64]) -> f64 {
        let n = xs.len();
        if n < 1 {
            0.0
        } else if n == 1 {
            xs[0].abs()
        } else {
            let mut scale = 0.0f64;
            let mut ssq = 1.0f64;
            for &xi in xs {
                if xi != 0.0 {
                    let absxi = xi.abs();
                    if scale < absxi {
                        let d = scale / absxi;
                        ssq = ssq * (d * d) + 1.0;
                        scale = absxi;
                    } else {
                        let d = absxi / scale;
                        ssq += d * d;
                    }
                }
            }
            scale * ssq.sqrt()
        }
    }
    fn d_sign(a: f64, b: f64) -> f64 {
        let x = if a >= 0.0 { a } else { -a };
        if b >= 0.0 { x } else { -x }
    }
    fn drotg(da: &mut f64, db: &mut f64) -> (f64, f64) {
        let (c, s, r, z);
        let roe = if da.abs() > db.abs() { *da } else { *db };
        let scale = da.abs() + db.abs();
        if scale == 0.0 {
            c = 1.0;
            s = 0.0;
            r = 0.0;
            z = 0.0;
        } else {
            let d1 = *da / scale;
            let d2 = *db / scale;
            let rr = scale * (d1 * d1 + d2 * d2).sqrt();
            r = d_sign(1.0, roe) * rr;
            c = *da / r;
            s = *db / r;
            let mut zz = 1.0;
            if da.abs() > db.abs() {
                zz = s;
            }
            if db.abs() >= da.abs() && c != 0.0 {
                zz = 1.0 / c;
            }
            z = zz;
        }
        *da = r;
        *db = z;
        (c, s)
    }
    // Column helpers on a column-major 1-based buffer.
    let col = |buf: &Vec<f64>, j: usize, from: usize, len: usize| -> Vec<f64> {
        (0..len).map(|t| buf[(from + t) + j * ld]).collect()
    };
    fn ddot(a: &[f64], b: &[f64]) -> f64 {
        let mut t = 0.0f64;
        for k in 0..a.len() {
            t += a[k] * b[k];
        }
        t
    }
    fn drot(buf: &mut [f64], ld: usize, n: usize, jx: usize, jy: usize, c: f64, s: f64) {
        for i in 1..=n {
            let dx = buf[i + jx * ld];
            let dy = buf[i + jy * ld];
            let t = c * dx + s * dy;
            buf[i + jy * ld] = c * dy - s * dx;
            buf[i + jx * ld] = t;
        }
    }
    fn dswap(buf: &mut [f64], ld: usize, n: usize, jx: usize, jy: usize) {
        for i in 1..=n {
            buf.swap(i + jx * ld, i + jy * ld);
        }
    }

    let n = N;
    let p = P;
    let maxit = 1000;
    let wantu = true; // job / 10 % 10 = 2
    let wantv = true; // job % 10 = 1
    let ncu = n.min(p); // jobu > 1
    let nct = (n - 1).min(p);
    let nrt = (p as i64 - 2).min(n as i64).max(0) as usize;
    let lu = nct.max(nrt);

    // Reduce X to bidiagonal form.
    for l in 1..=lu {
        let lp1 = l + 1;
        if l <= nct {
            let len = n - l + 1;
            s[l] = dnrm2(&col(&x, l, l, len));
            if s[l] != 0.0 {
                if x[idx(l, l)] != 0.0 {
                    s[l] = d_sign(s[l], x[idx(l, l)]);
                }
                let f = 1.0 / s[l];
                for t in 0..len {
                    x[idx(l + t, l)] *= f;
                }
                x[idx(l, l)] += 1.0;
            }
            s[l] = -s[l];
        }
        if p >= lp1 {
            for j in lp1..=p {
                if l <= nct && s[l] != 0.0 {
                    let len = n - l + 1;
                    let t = -ddot(&col(&x, l, l, len), &col(&x, j, l, len)) / x[idx(l, l)];
                    if t != 0.0 {
                        for q in 0..len {
                            x[idx(l + q, j)] += t * x[idx(l + q, l)];
                        }
                    }
                }
                e[j] = x[idx(l, j)];
            }
        }
        if wantu && l <= nct {
            for i in l..=n {
                u[idx(i, l)] = x[idx(i, l)];
            }
        }
        if l <= nrt {
            let len = p - l;
            e[l] = dnrm2(&e[lp1..lp1 + len]);
            if e[l] != 0.0 {
                if e[lp1] != 0.0 {
                    e[l] = d_sign(e[l], e[lp1]);
                }
                let f = 1.0 / e[l];
                for t in 0..len {
                    e[lp1 + t] *= f;
                }
                e[lp1] += 1.0;
            }
            e[l] = -e[l];
            if lp1 <= n && e[l] != 0.0 {
                for i in lp1..=n {
                    work[i] = 0.0;
                }
                for j in lp1..=p {
                    let da = e[j];
                    if da != 0.0 {
                        for q in 0..(n - l) {
                            work[lp1 + q] += da * x[idx(lp1 + q, j)];
                        }
                    }
                }
                for j in lp1..=p {
                    let da = -e[j] / e[lp1];
                    if da != 0.0 {
                        for q in 0..(n - l) {
                            x[idx(lp1 + q, j)] += da * work[lp1 + q];
                        }
                    }
                }
            }
            if wantv {
                for i in lp1..=p {
                    v[idx(i, l)] = e[i];
                }
            }
        }
    }

    // Set up the final bidiagonal matrix of order m.
    let mut m = p.min(n + 1);
    let nctp1 = nct + 1;
    let nrtp1 = nrt + 1;
    if nct < p {
        s[nctp1] = x[idx(nctp1, nctp1)];
    }
    if n < m {
        s[m] = 0.0;
    }
    if nrtp1 < m {
        e[nrtp1] = x[idx(nrtp1, m)];
    }
    e[m] = 0.0;

    // Generate U.
    if wantu {
        if ncu >= nctp1 {
            for j in nctp1..=ncu {
                for i in 1..=n {
                    u[idx(i, j)] = 0.0;
                }
                u[idx(j, j)] = 1.0;
            }
        }
        for ll in 1..=nct {
            let l = nct - ll + 1;
            if s[l] != 0.0 {
                let lp1 = l + 1;
                if ncu >= lp1 {
                    for j in lp1..=ncu {
                        let len = n - l + 1;
                        let t = -ddot(&col(&u, l, l, len), &col(&u, j, l, len)) / u[idx(l, l)];
                        if t != 0.0 {
                            for q in 0..len {
                                u[idx(l + q, j)] += t * u[idx(l + q, l)];
                            }
                        }
                    }
                }
                for t in 0..(n - l + 1) {
                    u[idx(l + t, l)] *= -1.0;
                }
                u[idx(l, l)] += 1.0;
                for i in 1..l {
                    u[idx(i, l)] = 0.0;
                }
            } else {
                for i in 1..=n {
                    u[idx(i, l)] = 0.0;
                }
                u[idx(l, l)] = 1.0;
            }
        }
    }

    // Generate V.
    if wantv {
        for ll in 1..=p {
            let l = p - ll + 1;
            let lp1 = l + 1;
            if l <= nrt && e[l] != 0.0 {
                for j in lp1..=p {
                    let len = p - l;
                    let t = -ddot(&col(&v, l, lp1, len), &col(&v, j, lp1, len)) / v[idx(lp1, l)];
                    if t != 0.0 {
                        for q in 0..len {
                            v[idx(lp1 + q, j)] += t * v[idx(lp1 + q, l)];
                        }
                    }
                }
            }
            for i in 1..=p {
                v[idx(i, l)] = 0.0;
            }
            v[idx(l, l)] = 1.0;
        }
    }

    // Main iteration loop for the singular values.
    let mm = m;
    let mut iter = 0;
    loop {
        if m == 0 {
            break;
        }
        if iter >= maxit {
            return None;
        }
        // Inspect for negligible elements in the s and e arrays.
        let mut l = 0usize;
        for ll in 1..=m {
            l = m - ll;
            if l == 0 {
                break;
            }
            let test = s[l].abs() + s[l + 1].abs();
            let ztest = test + e[l].abs();
            if ztest == test {
                e[l] = 0.0;
                break;
            }
        }
        let kase;
        if l == m - 1 {
            kase = 4;
        } else {
            let lp1 = l + 1;
            let mp1 = m + 1;
            let mut ls = 0usize;
            for lls in lp1..=mp1 {
                ls = m - lls + lp1;
                if ls == l {
                    break;
                }
                let mut test = 0.0f64;
                if ls != m {
                    test += e[ls].abs();
                }
                if ls != l + 1 {
                    test += e[ls - 1].abs();
                }
                let ztest = test + s[ls].abs();
                if ztest == test {
                    s[ls] = 0.0;
                    break;
                }
            }
            if ls == l {
                kase = 3;
            } else if ls == m {
                kase = 1;
            } else {
                kase = 2;
                l = ls;
            }
        }
        l += 1;
        match kase {
            1 => {
                // Deflate negligible s(m).
                let mm1 = m - 1;
                let mut f = e[m - 1];
                e[m - 1] = 0.0;
                for kk in l..=mm1 {
                    let k = mm1 - kk + l;
                    let mut t1 = s[k];
                    let (cs, sn) = drotg(&mut t1, &mut f);
                    s[k] = t1;
                    if k != l {
                        f = -sn * e[k - 1];
                        e[k - 1] *= cs;
                    }
                    if wantv {
                        drot(&mut v, ld, p, k, m, cs, sn);
                    }
                }
            }
            2 => {
                // Split at negligible s(l).
                let mut f = e[l - 1];
                e[l - 1] = 0.0;
                for k in l..=m {
                    let mut t1 = s[k];
                    let (cs, sn) = drotg(&mut t1, &mut f);
                    s[k] = t1;
                    f = -sn * e[k];
                    e[k] *= cs;
                    if wantu {
                        drot(&mut u, ld, n, k, l - 1, cs, sn);
                    }
                }
            }
            3 => {
                // Perform one QR step: calculate the shift.
                // f2c's max chain (`max(a, b)` = `a >= b ? a : b`).
                let fmax = |a: f64, b: f64| if a >= b { a } else { b };
                let scale = fmax(fmax(fmax(fmax(s[m].abs(), s[m - 1].abs()), e[m - 1].abs()), s[l].abs()), e[l].abs());
                let sm = s[m] / scale;
                let smm1 = s[m - 1] / scale;
                let emm1 = e[m - 1] / scale;
                let sl = s[l] / scale;
                let el = e[l] / scale;
                let b = ((smm1 + sm) * (smm1 - sm) + emm1 * emm1) / 2.0;
                let c = (sm * emm1) * (sm * emm1);
                let mut shift = 0.0f64;
                if !(b == 0.0 && c == 0.0) {
                    shift = (b * b + c).sqrt();
                    if b < 0.0 {
                        shift = -shift;
                    }
                    shift = c / (b + shift);
                }
                let mut f = (sl + sm) * (sl - sm) + shift;
                let mut g = sl * el;
                // Chase zeros.
                let mm1 = m - 1;
                for k in l..=mm1 {
                    let (cs, sn) = drotg(&mut f, &mut g);
                    if k != l {
                        e[k - 1] = f;
                    }
                    f = cs * s[k] + sn * e[k];
                    e[k] = cs * e[k] - sn * s[k];
                    g = sn * s[k + 1];
                    s[k + 1] *= cs;
                    if wantv {
                        drot(&mut v, ld, p, k, k + 1, cs, sn);
                    }
                    let (cs, sn) = drotg(&mut f, &mut g);
                    s[k] = f;
                    f = cs * e[k] + sn * s[k + 1];
                    s[k + 1] = -sn * e[k] + cs * s[k + 1];
                    g = sn * e[k + 1];
                    e[k + 1] *= cs;
                    if wantu && k < n {
                        drot(&mut u, ld, n, k, k + 1, cs, sn);
                    }
                }
                e[m - 1] = f;
                iter += 1;
            }
            _ => {
                // Convergence: make the singular value positive.
                if s[l] < 0.0 {
                    s[l] = -s[l];
                    if wantv {
                        for i in 1..=p {
                            v[idx(i, l)] *= -1.0;
                        }
                    }
                }
                // Order the singular values.
                while l != mm && s[l] < s[l + 1] {
                    s.swap(l, l + 1);
                    if wantv && l < p {
                        dswap(&mut v, ld, p, l, l + 1);
                    }
                    if wantu && l < n {
                        dswap(&mut u, ld, n, l, l + 1);
                    }
                    l += 1;
                }
                iter = 0;
                m -= 1;
            }
        }
    }
    let mut uu = [[0.0f64; 3]; 3];
    let mut vv = [[0.0f64; 3]; 3];
    let mut w = [0.0f64; 3];
    for i in 0..3 {
        for j in 0..3 {
            uu[i][j] = u[idx(i + 1, j + 1)];
            vv[i][j] = v[idx(i + 1, j + 1)];
        }
        w[i] = s[i + 1].abs();
    }
    Some((uu, w, vv))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recompose(u: &[[f64; 3]; 3], w: &[f64; 3], v: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
        let mut r = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = (0..3).map(|k| u[i][k] * w[k] * v[j][k]).sum();
            }
        }
        r
    }

    #[test]
    fn svd3_decomposes() {
        let ms = [
            [[1.0, 4.897e-12, 0.0], [4.897e-12, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.9, -0.3, 0.1], [0.2, 1.1, -0.4], [0.05, 0.3, 2.0]],
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        ];
        for m in ms {
            let (u, w, v) = svd3(&m).unwrap();
            let r = recompose(&u, &w, &v);
            for i in 0..3 {
                for j in 0..3 {
                    assert!((r[i][j] - m[i][j]).abs() < 1e-14, "{m:?}");
                }
            }
            assert!(w[0] >= w[1] && w[1] >= w[2]);
        }
    }

    #[test]
    fn vnl_inverse_bit_identical_to_vnl() {
        // tests/data/vnl_inverse_3x3.sh: ITK v5.3.0's v3p dsvdc + vnl_svd::pinverse, built with gcc.
        let txt = include_str!("../../tests/data/vnl_inverse_3x3.txt");
        let hex = |s: &str| -> Vec<f64> {
            s.split_whitespace().map(|h| f64::from_bits(u64::from_str_radix(h, 16).unwrap())).collect()
        };
        let mut n = 0;
        for line in txt.lines().filter(|l| !l.starts_with('#')) {
            let (m, want) = line.split_once('|').unwrap();
            let (m, want) = (hex(m), hex(want));
            let m3 = [[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]];
            let got = vnl_inverse(&m3).unwrap();
            for i in 0..3 {
                for j in 0..3 {
                    assert_eq!(got[i][j].to_bits(), want[3 * i + j].to_bits(), "line {n}: [{i}][{j}] {} vs {}", got[i][j], want[3 * i + j]);
                }
            }
            n += 1;
        }
        assert_eq!(n, 52);
    }

    /// The echo-1 magnitude header of QSM-CI's in-vivo `prisma-bridge-run1` (QSMxT/QSM-CI#312):
    /// sform (code 2) only, with a 4.9e-12 off-diagonal term.
    pub(crate) fn prisma_header() -> nifti::NiftiHeader {
        let f = f32::from_bits;
        nifti::NiftiHeader {
            dim: [4, 184, 256, 128, 5, 1, 1, 1],
            pixdim: [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            qform_code: 0,
            sform_code: 2,
            srow_x: [f(0xbf800000), f(0xacac4c41), 0.0, f(0x42b602eb)],
            srow_y: [f(0xacac4c41), f(0x3f800000), f(0x80000000), f(0xc2e31894)],
            srow_z: [0.0, 0.0, f(0x3f800000), f(0xc21eab36)],
            ..Default::default()
        }
    }

    #[test]
    fn shrink_offsets_of_a_real_header() {
        // ANTs 2.4.3 on this header shrinks from (2, 1, 2): its N4 output matches this port's at
        // those offsets to 0.0054 % and differs by 1.1–3.0 % at every other combination
        // (2026-10-31). An axis-aligned header of the same size gives (2, 2, 2).
        let g = ItkGeometry::from_nifti_header(&prisma_header()).unwrap();
        assert_eq!(g.origin, [-91.00569915771484, 113.54800415039062, -39.667198181152344]);
        assert_eq!(g.shrink_offsets((184, 256, 128), 4).unwrap(), [2, 1, 2]);
        assert_eq!(ItkGeometry::axis_aligned([1.0; 3]).shrink_offsets((184, 256, 128), 4).unwrap(), [2, 2, 2]);
        eprintln!("prisma header at 30x32x26: {:?}", g.shrink_offsets((30, 32, 26), 4).unwrap());
        eprintln!("prisma header at 40x44x36: {:?}", g.shrink_offsets((40, 44, 36), 4).unwrap());
    }

    #[test]
    fn shrink_offsets_axis_aligned_round_half_up() {
        // Exact geometry: (n-1)/2 - s(m-1)/2 rounded half up.
        let g = ItkGeometry::make_nii((1.0, 1.0, 1.0));
        assert_eq!(g.shrink_offsets((164, 205, 206), 4).unwrap(), [2, 2, 3]);
        assert_eq!(g.shrink_offsets((178, 148, 64), 4).unwrap(), [3, 2, 2]);
    }
}

//! NIfTI file I/O for WASM
//!
//! Provides functions to load and save NIfTI files from/to byte arrays,
//! suitable for use in WebAssembly where filesystem access is not available.

use std::io::Cursor;
use nifti::{NiftiObject, InMemNiftiObject, NiftiHeader};
use nifti::volume::ndarray::IntoNdArray;
use flate2::read::GzDecoder;
use ndarray::Array;

/// NIfTI data loaded from bytes
pub struct NiftiData {
    /// Volume data as f64
    pub data: Vec<f64>,
    /// Dimensions (nx, ny, nz) - only 3D supported for now
    pub dims: (usize, usize, usize),
    /// Voxel sizes in mm
    pub voxel_size: (f64, f64, f64),
    /// Affine transformation matrix (4x4, row-major)
    pub affine: [f64; 16],
    /// Data scaling slope
    pub scl_slope: f64,
    /// Data scaling intercept
    pub scl_inter: f64,
}

/// Check if bytes are gzip compressed
fn is_gzip(bytes: &[u8]) -> bool {
    bytes.len() >= 2 && bytes[0] == 0x1f && bytes[1] == 0x8b
}

/// Get header info for diagnostics
fn get_header_info(bytes: &[u8]) -> String {
    if bytes.len() < 348 {
        return format!("File too small ({} bytes, need at least 348)", bytes.len());
    }

    // NIfTI-1 header size should be at offset 0, stored as i32
    let sizeof_hdr = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);

    // Magic bytes at offset 344 for NIfTI-1
    let magic = if bytes.len() >= 348 {
        String::from_utf8_lossy(&bytes[344..348]).to_string()
    } else {
        "N/A".to_string()
    };

    // Data type at offset 70 (dim[0..8] at 40, then datatype at 70)
    let datatype = if bytes.len() >= 72 {
        i16::from_le_bytes([bytes[70], bytes[71]])
    } else {
        -1
    };

    format!("sizeof_hdr={}, magic='{}', datatype={}", sizeof_hdr, magic, datatype)
}

/// Load a NIfTI file from bytes
///
/// Supports both .nii and .nii.gz files (gzip is auto-detected)
pub fn load_nifti(bytes: &[u8]) -> Result<NiftiData, String> {
    let obj: InMemNiftiObject = if is_gzip(bytes) {
        let cursor = Cursor::new(bytes);
        let decoder = GzDecoder::new(cursor);
        InMemNiftiObject::from_reader(decoder)
            .map_err(|e| {
                // Try to get header info from decompressed data
                let mut decoder2 = GzDecoder::new(Cursor::new(bytes));
                let mut decompressed = Vec::new();
                let info = if std::io::Read::read_to_end(&mut decoder2, &mut decompressed).is_ok() {
                    get_header_info(&decompressed)
                } else {
                    "Could not decompress".to_string()
                };
                format!("Failed to read gzipped NIfTI: {} ({})", e, info)
            })?
    } else {
        let info = get_header_info(bytes);
        let cursor = Cursor::new(bytes);
        InMemNiftiObject::from_reader(cursor)
            .map_err(|e| format!("Failed to read NIfTI: {} ({})", e, info))?
    };

    let header = obj.header();

    // Get dimensions (only support 3D for now)
    let dim = header.dim;
    let ndim = dim[0] as usize;
    if ndim < 3 {
        return Err(format!("Expected at least 3D volume, got {}D", ndim));
    }

    // Get voxel sizes
    let pixdim = header.pixdim;
    let vsx = pixdim[1] as f64;
    let vsy = pixdim[2] as f64;
    let vsz = pixdim[3] as f64;

    // Get scaling
    let scl_slope = if header.scl_slope == 0.0 { 1.0 } else { header.scl_slope as f64 };
    let scl_inter = header.scl_inter as f64;

    // Get affine matrix
    let affine = get_affine(header);

    // Convert volume to ndarray
    let volume = obj.into_volume();
    let array: Array<f64, _> = volume.into_ndarray()
        .map_err(|e| format!("Failed to convert to ndarray: {}", e))?;

    // Get the actual shape from the ndarray
    let shape = array.shape();

    // Verify shape is at least 3D
    if shape.len() < 3 {
        return Err(format!("Expected at least 3D array, got {}D", shape.len()));
    }

    // Use the actual array shape for dimensions (nifti-rs may reorder)
    let (dim0, dim1, dim2) = (shape[0], shape[1], shape[2]);
    let expected_size = dim0 * dim1 * dim2;

    // Extract data in Fortran order (x varies fastest) to match NIfTI convention
    // index = x + y*nx + z*nx*ny
    let mut data = Vec::with_capacity(expected_size);

    // Handle potentially 4D arrays (take first volume)
    if shape.len() == 3 {
        for k in 0..dim2 {
            for j in 0..dim1 {
                for i in 0..dim0 {
                    data.push(array[[i, j, k]]);
                }
            }
        }
    } else if shape.len() >= 4 {
        // 4D array - take first timepoint
        for k in 0..dim2 {
            for j in 0..dim1 {
                for i in 0..dim0 {
                    data.push(array[[i, j, k, 0]]);
                }
            }
        }
    }

    // Return dimensions matching the actual array shape order
    // This ensures data indexing is consistent with reported dimensions
    Ok(NiftiData {
        data,
        dims: (dim0, dim1, dim2),
        voxel_size: (vsx, vsy, vsz),
        affine,
        scl_slope,
        scl_inter,
    })
}

/// Load a 4D NIfTI file from bytes (for multi-echo data)
pub fn load_nifti_4d(bytes: &[u8]) -> Result<(Vec<f64>, (usize, usize, usize, usize), (f64, f64, f64), [f64; 16]), String> {
    let obj: InMemNiftiObject = if is_gzip(bytes) {
        let cursor = Cursor::new(bytes);
        let decoder = GzDecoder::new(cursor);
        InMemNiftiObject::from_reader(decoder)
            .map_err(|e| format!("Failed to read gzipped NIfTI: {}", e))?
    } else {
        let cursor = Cursor::new(bytes);
        InMemNiftiObject::from_reader(cursor)
            .map_err(|e| format!("Failed to read NIfTI: {}", e))?
    };

    let header = obj.header();
    let dim = header.dim;
    let _ndim = dim[0] as usize;

    let pixdim = header.pixdim;
    let vsx = pixdim[1] as f64;
    let vsy = pixdim[2] as f64;
    let vsz = pixdim[3] as f64;

    let affine = get_affine(header);

    // Convert volume to ndarray
    let volume = obj.into_volume();
    let array: Array<f64, _> = volume.into_ndarray()
        .map_err(|e| format!("Failed to convert to ndarray: {}", e))?;

    let shape = array.shape();

    // Use actual array shape for dimensions
    let (dim0, dim1, dim2) = (shape[0], shape[1], shape[2]);
    let dim3 = if shape.len() >= 4 { shape[3] } else { 1 };

    // Extract data in Fortran order (x varies fastest) to match NIfTI convention
    // For 4D: index = x + y*nx + z*nx*ny + t*nx*ny*nz
    let mut data = Vec::with_capacity(dim0 * dim1 * dim2 * dim3);

    if shape.len() == 3 {
        // 3D array
        for k in 0..dim2 {
            for j in 0..dim1 {
                for i in 0..dim0 {
                    data.push(array[[i, j, k]]);
                }
            }
        }
    } else if shape.len() >= 4 {
        // 4D array - each volume in Fortran order
        for t in 0..dim3 {
            for k in 0..dim2 {
                for j in 0..dim1 {
                    for i in 0..dim0 {
                        data.push(array[[i, j, k, t]]);
                    }
                }
            }
        }
    }

    // Return dimensions matching actual array shape
    Ok((data, (dim0, dim1, dim2, dim3), (vsx, vsy, vsz), affine))
}

/// Get affine transformation matrix from header
fn get_affine(header: &NiftiHeader) -> [f64; 16] {
    // The precedence is the spec's, and nibabel's and nifticlib's: sform, then qform, then the
    // bare voxel scaling. Files this crate writes carry both and they agree by construction (see
    // [`qform_for`]), but plenty of converters write only a qform, and falling straight through to
    // the voxel scaling there discards the orientation — a sagittal acquisition would come back
    // looking axial.
    if header.sform_code > 0 {
        let s = &header.srow_x;
        let t = &header.srow_y;
        let u = &header.srow_z;
        [
            s[0] as f64, s[1] as f64, s[2] as f64, s[3] as f64,
            t[0] as f64, t[1] as f64, t[2] as f64, t[3] as f64,
            u[0] as f64, u[1] as f64, u[2] as f64, u[3] as f64,
            0.0, 0.0, 0.0, 1.0,
        ]
    } else if header.qform_code > 0 {
        let q = QForm {
            quatern: [header.quatern_b as f64, header.quatern_c as f64, header.quatern_d as f64],
            offset: [header.quatern_x as f64, header.quatern_y as f64, header.quatern_z as f64],
            // pixdim[0] is +-1; the spec says to read anything that is not -1 as +1.
            qfac: if header.pixdim[0] < 0.0 { -1.0 } else { 1.0 },
        };
        let pixdim = (
            header.pixdim[1] as f64,
            header.pixdim[2] as f64,
            header.pixdim[3] as f64,
        );
        affine_from_quatern(&q, pixdim)
    } else {
        // Fall back to identity with voxel scaling
        let vsx = header.pixdim[1] as f64;
        let vsy = header.pixdim[2] as f64;
        let vsz = header.pixdim[3] as f64;
        [
            vsx, 0.0, 0.0, 0.0,
            0.0, vsy, 0.0, 0.0,
            0.0, 0.0, vsz, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]
    }
}

/// NIfTI-1's quaternion ("qform") description of a voxel→world transform — method 2 in the spec.
///
/// A reader rebuilds the matrix from the quaternion, the voxel sizes in `pixdim[1..=3]`, and
/// `qfac`. The quaternion's scalar part is not stored: it is implied by the other three
/// components, which is why only `b`, `c`, `d` appear here.
#[derive(Debug, Clone, Copy, PartialEq)]
struct QForm {
    /// `quatern_b`, `quatern_c`, `quatern_d`.
    quatern: [f64; 3],
    /// `qoffset_x`, `qoffset_y`, `qoffset_z` — the affine's translation, verbatim.
    offset: [f64; 3],
    /// `pixdim[0]`: `-1` for a left-handed (negative determinant) transform, `+1` otherwise.
    ///
    /// A quaternion only ever describes a rotation, so a mirrored acquisition cannot be expressed
    /// by one. This is how the spec carries the reflection separately: the reader negates the
    /// third column after rotating. `0` is not a legal value, and the spec says to read it as
    /// `+1`.
    qfac: f64,
}

/// Decompose a row-major voxel→world affine into its quaternion form, following the NIfTI-1
/// spec's `nifti_mat44_to_quatern`.
///
/// Returns `None` for an affine with a zero-length column, which describes no orientation at all.
///
/// The spec's reference implementation runs a polar decomposition at this point, to pick the
/// nearest orthogonal matrix when the caller's columns are not perpendicular. This deliberately
/// does not. A sheared affine has no quaternion, and approximating one is exactly how a qform
/// comes to disagree with its sform — the thing [`qform_for`] exists to prevent. Shear is caught
/// there instead, by checking the reconstruction, and the file then goes out with no qform rather
/// than a wrong one. For an orthogonal affine — every affine a scanner produces, and every affine
/// this crate builds — the polar step is the identity, so this agrees with the reference to
/// round-off.
fn quatern_from_affine(affine: &[f64; 16]) -> Option<QForm> {
    let mut r = [
        [affine[0], affine[1], affine[2]],
        [affine[4], affine[5], affine[6]],
        [affine[8], affine[9], affine[10]],
    ];
    let offset = [affine[3], affine[7], affine[11]];

    // The column norms are the voxel sizes; dividing them out leaves the rotation. They are
    // carried by `pixdim`, not by the quaternion.
    for c in 0..3 {
        let n = (r[0][c] * r[0][c] + r[1][c] * r[1][c] + r[2][c] * r[2][c]).sqrt();
        if n < 1e-12 {
            return None;
        }
        for row in r.iter_mut() {
            row[c] /= n;
        }
    }

    let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
    let qfac = if det < 0.0 { -1.0 } else { 1.0 };
    if qfac < 0.0 {
        // Flip the third column so what is left is a proper rotation; `qfac` tells the reader to
        // put the reflection back.
        for row in r.iter_mut() {
            row[2] = -row[2];
        }
    }

    // Trace formula, branching on whichever component is largest so the division stays well
    // conditioned. `a` is the scalar part — computed only to normalise the sign convention
    // (the spec keeps `a >= 0`), then discarded, since readers recover it from b, c, d.
    let trace = r[0][0] + r[1][1] + r[2][2] + 1.0;
    let (b, c, d) = if trace > 0.5 {
        let a = 0.5 * trace.sqrt();
        (
            0.25 * (r[2][1] - r[1][2]) / a,
            0.25 * (r[0][2] - r[2][0]) / a,
            0.25 * (r[1][0] - r[0][1]) / a,
        )
    } else {
        // 4b², 4c², 4d². For a rotation these sum to `4 - trace` ≥ 3.5, so at least one exceeds 1.
        let xd = 1.0 + r[0][0] - (r[1][1] + r[2][2]);
        let yd = 1.0 + r[1][1] - (r[0][0] + r[2][2]);
        let zd = 1.0 + r[2][2] - (r[0][0] + r[1][1]);
        let (a, b, c, d) = if xd > 1.0 {
            let b = 0.5 * xd.sqrt();
            (0.25 * (r[2][1] - r[1][2]) / b, b, 0.25 * (r[0][1] + r[1][0]) / b,
             0.25 * (r[0][2] + r[2][0]) / b)
        } else if yd > 1.0 {
            let c = 0.5 * yd.sqrt();
            (0.25 * (r[0][2] - r[2][0]) / c, 0.25 * (r[0][1] + r[1][0]) / c, c,
             0.25 * (r[1][2] + r[2][1]) / c)
        } else {
            let d = 0.5 * zd.sqrt();
            (0.25 * (r[1][0] - r[0][1]) / d, 0.25 * (r[0][2] + r[2][0]) / d,
             0.25 * (r[1][2] + r[2][1]) / d, d)
        };
        if a < 0.0 { (-b, -c, -d) } else { (b, c, d) }
    };

    Some(QForm { quatern: [b, c, d], offset, qfac })
}

/// Rebuild the voxel→world affine a reader derives from a qform — the spec's
/// `nifti_quatern_to_mat44`. `pixdim` is `pixdim[1..=3]` as it appears in the header.
fn affine_from_quatern(q: &QForm, pixdim: (f64, f64, f64)) -> [f64; 16] {
    let [mut b, mut c, mut d] = q.quatern;
    let mut a = 1.0 - (b * b + c * c + d * d);
    if a < 1e-7 {
        // A half-turn, where the scalar part vanishes. Renormalise the vector part rather than
        // taking the square root of a small negative number.
        let n = (b * b + c * c + d * d).sqrt();
        b /= n;
        c /= n;
        d /= n;
        a = 0.0;
    } else {
        a = a.sqrt();
    }

    let xd = if pixdim.0 > 0.0 { pixdim.0 } else { 1.0 };
    let yd = if pixdim.1 > 0.0 { pixdim.1 } else { 1.0 };
    let zd = if pixdim.2 > 0.0 { pixdim.2 } else { 1.0 };
    let zd = if q.qfac < 0.0 { -zd } else { zd };

    [
        (a * a + b * b - c * c - d * d) * xd, 2.0 * (b * c - a * d) * yd, 2.0 * (b * d + a * c) * zd, q.offset[0],
        2.0 * (b * c + a * d) * xd, (a * a + c * c - b * b - d * d) * yd, 2.0 * (c * d - a * b) * zd, q.offset[1],
        2.0 * (b * d - a * c) * xd, 2.0 * (c * d + a * b) * yd, (a * a + d * d - c * c - b * b) * zd, q.offset[2],
        0.0, 0.0, 0.0, 1.0,
    ]
}

/// The qform to write alongside an sform, or `None` when no quaternion reproduces the affine.
///
/// Every file this crate writes carries an sform, and for a long time that was all it carried:
/// `qform_code` and the quaternion fields were left at zero. That is spec-legal — a reader is
/// meant to fall back to method 1, the bare `pixdim` scaling — but it is not what readers
/// actually show. `fslhd` and friends print a `qto_xyz` derived from the all-zero quaternion
/// regardless, so every output looked as though it had been silently reoriented to a canonical
/// axial frame with no translation, disagreeing with the sform beside it (QSMxT#240). Writing a
/// real qform is what those tools want, so that is what we do.
///
/// `pixdim` has to be the voxel size *as written to the header*, because method 2 scales the
/// quaternion's rotation by it. An affine whose column norms disagree with `pixdim` therefore
/// cannot be expressed as a qform even when its rotation is exact, and that is one of two ways
/// the attempt can fail; shear is the other. Both are caught the same way — rebuild the affine a
/// reader would derive and compare — so a qform is only ever advertised when it is the sform.
fn qform_for(affine: &[f64; 16], pixdim: (f64, f64, f64)) -> Option<QForm> {
    let q = quatern_from_affine(affine)?;
    let rebuilt = affine_from_quatern(&q, pixdim);

    // Tolerance scales with the voxel size, the unit the 3×3 is in. Comfortably above the ~1e-7
    // relative noise an affine picks up from being stored as f32 in the file it was read from,
    // and far below any shear worth preserving. The translation is copied into the qoffset
    // verbatim, so it cannot disagree and is not checked.
    let scale = (0..3)
        .map(|c| (rebuilt[c] * rebuilt[c] + rebuilt[4 + c] * rebuilt[4 + c] + rebuilt[8 + c] * rebuilt[8 + c]).sqrt())
        .fold(0.0f64, f64::max);
    let tol = 1e-4 * scale.max(1e-6);
    for row in 0..3 {
        for col in 0..3 {
            if (rebuilt[4 * row + col] - affine[4 * row + col]).abs() > tol {
                return None;
            }
        }
    }
    Some(q)
}

/// Save data as NIfTI bytes
///
/// Writes an uncompressed .nii file
pub fn save_nifti(
    data: &[f64],
    dims: (usize, usize, usize),
    voxel_size: (f64, f64, f64),
    affine: &[f64; 16],
) -> Result<Vec<u8>, String> {
    use std::io::Write;

    let (nx, ny, nz) = dims;
    let (vsx, vsy, vsz) = voxel_size;

    // The header states a matrix size and every reader trusts it, so writing a buffer that does
    // not match produces a file that opens fine and then fails — or shows nothing — the moment
    // something reads the voxels (QSMxT#211). A wrong length is always a caller bug; the only
    // question is whether it surfaces here or as a damaged file someone opens later.
    let expected = nx * ny * nz;
    if data.len() != expected {
        return Err(format!(
            "cannot write a {}x{}x{} NIfTI ({} voxels) from {} values",
            nx, ny, nz, expected, data.len(),
        ));
    }

    // Create NIfTI-1 header (348 bytes)
    let mut header = [0u8; 348];

    // sizeof_hdr = 348
    header[0..4].copy_from_slice(&348i32.to_le_bytes());

    // dim[0..7]
    let dim: [i16; 8] = [3, nx as i16, ny as i16, nz as i16, 1, 1, 1, 1];
    for (i, &d) in dim.iter().enumerate() {
        let offset = 40 + i * 2;
        header[offset..offset + 2].copy_from_slice(&d.to_le_bytes());
    }

    // datatype = 16 (FLOAT32)
    header[70..72].copy_from_slice(&16i16.to_le_bytes());

    // bitpix = 32
    header[72..74].copy_from_slice(&32i16.to_le_bytes());

    // The quaternion has to be settled before `pixdim` is written, because `pixdim[0]` carries
    // its handedness: 0, which a zeroed buffer would leave there, is not a legal value.
    let qform = qform_for(affine, voxel_size);

    // pixdim[0..7]. pixdim[0] is qfac — meaningless without a qform, where +1 is the spec's
    // reading of anything that is not -1.
    let qfac = qform.map_or(1.0, |q| q.qfac);
    let pixdim: [f32; 8] = [qfac as f32, vsx as f32, vsy as f32, vsz as f32, 1.0, 1.0, 1.0, 1.0];
    for (i, &p) in pixdim.iter().enumerate() {
        let offset = 76 + i * 4;
        header[offset..offset + 4].copy_from_slice(&p.to_le_bytes());
    }

    // vox_offset = 352 (header + 4 bytes extension)
    header[108..112].copy_from_slice(&352.0f32.to_le_bytes());

    // scl_slope = 1.0
    header[112..116].copy_from_slice(&1.0f32.to_le_bytes());

    // scl_inter = 0.0
    header[116..120].copy_from_slice(&0.0f32.to_le_bytes());

    // qform_code / sform_code = 1 (scanner anat). Both describe the same transform, so they get
    // the same code; a qform we could not make agree with the sform is left absent (code 0),
    // which sends readers to the sform rather than to a transform that contradicts it.
    if let Some(q) = qform {
        header[252..254].copy_from_slice(&1i16.to_le_bytes());
        for (i, &v) in q.quatern.iter().chain(q.offset.iter()).enumerate() {
            let offset = 256 + i * 4;
            header[offset..offset + 4].copy_from_slice(&(v as f32).to_le_bytes());
        }
    }
    header[254..256].copy_from_slice(&1i16.to_le_bytes());

    // srow_x, srow_y, srow_z
    for i in 0..4 {
        let offset = 280 + i * 4;
        header[offset..offset + 4].copy_from_slice(&(affine[i] as f32).to_le_bytes());
    }
    for i in 0..4 {
        let offset = 296 + i * 4;
        header[offset..offset + 4].copy_from_slice(&(affine[4 + i] as f32).to_le_bytes());
    }
    for i in 0..4 {
        let offset = 312 + i * 4;
        header[offset..offset + 4].copy_from_slice(&(affine[8 + i] as f32).to_le_bytes());
    }

    // magic = "n+1\0" for NIfTI-1 single file
    header[344..348].copy_from_slice(b"n+1\0");

    // Build output buffer
    let mut buffer = Vec::with_capacity(352 + data.len() * 4);

    // Write header
    buffer.write_all(&header).map_err(|e| format!("Write header failed: {}", e))?;

    // Write extension (4 bytes, all zeros = no extension)
    buffer.write_all(&[0u8; 4]).map_err(|e| format!("Write extension failed: {}", e))?;

    // Write data as float32
    for &val in data {
        buffer.write_all(&(val as f32).to_le_bytes())
            .map_err(|e| format!("Write data failed: {}", e))?;
    }

    Ok(buffer)
}

/// Save data as gzipped NIfTI bytes (.nii.gz)
pub fn save_nifti_gz(
    data: &[f64],
    dims: (usize, usize, usize),
    voxel_size: (f64, f64, f64),
    affine: &[f64; 16],
) -> Result<Vec<u8>, String> {
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;

    // First create uncompressed NIfTI
    let uncompressed = save_nifti(data, dims, voxel_size, affine)?;

    // Compress with gzip
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&uncompressed)
        .map_err(|e| format!("Gzip compression failed: {}", e))?;

    encoder.finish()
        .map_err(|e| format!("Gzip finish failed: {}", e))
}

/// Read only the dimensions from a NIfTI file header without loading volume data.
///
/// Returns (nx, ny, nz). Much faster and uses negligible memory compared to
/// `read_nifti_file` since it only reads and parses the 348-byte header.
pub fn read_nifti_dims(path: &std::path::Path) -> Result<(usize, usize, usize), String> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)
        .map_err(|e| format!("Failed to open '{}': {}", path.display(), e))?;

    // Read enough bytes to detect gzip and parse header
    let mut header_bytes = [0u8; 348];

    let path_str = path.to_string_lossy();
    if path_str.ends_with(".gz") {
        // Decompress just the header
        let mut decoder = GzDecoder::new(file);
        decoder.read_exact(&mut header_bytes)
            .map_err(|e| format!("Failed to read gzipped NIfTI header '{}': {}", path.display(), e))?;
    } else {
        file.read_exact(&mut header_bytes)
            .map_err(|e| format!("Failed to read NIfTI header '{}': {}", path.display(), e))?;
    }

    // Determine endianness from sizeof_hdr (bytes 0-3, should be 348)
    let sizeof_hdr_le = i32::from_le_bytes([header_bytes[0], header_bytes[1], header_bytes[2], header_bytes[3]]);
    let is_le = sizeof_hdr_le == 348;

    if !is_le {
        let sizeof_hdr_be = i32::from_be_bytes([header_bytes[0], header_bytes[1], header_bytes[2], header_bytes[3]]);
        if sizeof_hdr_be != 348 {
            return Err(format!(
                "Invalid NIfTI header in '{}': sizeof_hdr={} (LE) / {} (BE), expected 348",
                path.display(), sizeof_hdr_le, sizeof_hdr_be
            ));
        }
    }

    // dim[0..7] at offset 40, each i16
    let read_i16 = |offset: usize| -> i16 {
        if is_le {
            i16::from_le_bytes([header_bytes[offset], header_bytes[offset + 1]])
        } else {
            i16::from_be_bytes([header_bytes[offset], header_bytes[offset + 1]])
        }
    };

    let ndim = read_i16(40);
    if ndim < 3 {
        return Err(format!("Expected at least 3D volume in '{}', got {}D", path.display(), ndim));
    }

    let nx = read_i16(42) as usize;
    let ny = read_i16(44) as usize;
    let nz = read_i16(46) as usize;

    Ok((nx, ny, nz))
}

/// Read a NIfTI file from a filesystem path
///
/// Supports both .nii and .nii.gz files.
pub fn read_nifti_file(path: &std::path::Path) -> Result<NiftiData, String> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("Failed to read file '{}': {}", path.display(), e))?;
    load_nifti(&bytes)
}

/// Save NIfTI data to a file
///
/// If the path ends with .nii.gz, the file is gzip compressed.
/// Otherwise it is saved as uncompressed .nii.
pub fn save_nifti_to_file(
    path: &std::path::Path,
    data: &[f64],
    dims: (usize, usize, usize),
    voxel_size: (f64, f64, f64),
    affine: &[f64; 16],
) -> Result<(), String> {
    let path_str = path.to_string_lossy();
    let bytes = if path_str.ends_with(".nii.gz") {
        save_nifti_gz(data, dims, voxel_size, affine)
    } else {
        save_nifti(data, dims, voxel_size, affine)
    }
    .map_err(|e| format!("{}: {}", path.display(), e))?;

    std::fs::write(path, &bytes)
        .map_err(|e| format!("Failed to write file '{}': {}", path.display(), e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_affine_identity() {
        let mut header = NiftiHeader::default();
        header.pixdim[1] = 1.0;
        header.pixdim[2] = 2.0;
        header.pixdim[3] = 3.0;
        header.sform_code = 0;

        let affine = get_affine(&header);
        assert_eq!(affine[0], 1.0);
        assert_eq!(affine[5], 2.0);
        assert_eq!(affine[10], 3.0);
    }

    #[test]
    fn test_gzip_detection() {
        assert!(is_gzip(&[0x1f, 0x8b, 0x00]));
        assert!(!is_gzip(&[0x00, 0x00, 0x00]));
        assert!(!is_gzip(&[0x1f])); // Too short
    }

    /// A buffer that does not match the dimensions must never reach a file: the header would
    /// promise voxels the file does not hold, and readers only discover that on `get_fdata()`
    /// (QSMxT#211).
    #[test]
    fn test_save_nifti_rejects_a_mismatched_payload() {
        let dims = (2, 2, 2);
        let voxel_size = (1.0, 1.0, 1.0);
        let affine = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];

        let short = save_nifti(&vec![0.0; 6], dims, voxel_size, &affine).unwrap_err();
        assert!(short.contains("2x2x2"), "{}", short);
        assert!(short.contains("8 voxels"), "{}", short);
        assert!(short.contains("6 values"), "{}", short);

        assert!(save_nifti(&vec![0.0; 9], dims, voxel_size, &affine).is_err());
        // Both the gzip and the file writers route through it.
        assert!(save_nifti_gz(&vec![0.0; 6], dims, voxel_size, &affine).is_err());
    }

    #[test]
    fn test_save_nifti_to_file_names_the_file_it_refused() {
        let dir = std::env::temp_dir().join("qsm-core-mismatch-test");
        std::fs::create_dir_all(&dir).unwrap();
        let affine = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        for name in ["bad.nii", "bad.nii.gz"] {
            let path = dir.join(name);
            let err = save_nifti_to_file(&path, &vec![0.0; 6], (2, 2, 2), (1.0, 1.0, 1.0), &affine)
                .unwrap_err();
            assert!(err.contains(name), "{}", err);
            assert!(!path.exists(), "nothing may be written for a mismatch");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_save_nifti_header() {
        let data = vec![0.0; 8]; // 2x2x2
        let dims = (2, 2, 2);
        let voxel_size = (1.0, 1.0, 1.0);
        let affine = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];

        let bytes = save_nifti(&data, dims, voxel_size, &affine).unwrap();

        // Check header size + extension + data
        assert_eq!(bytes.len(), 352 + 8 * 4); // 348 header + 4 ext + 8 floats

        // Check magic
        assert_eq!(&bytes[344..348], b"n+1\0");

        // Check sizeof_hdr
        let sizeof_hdr = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(sizeof_hdr, 348);
    }

    #[test]
    fn test_save_and_read_nifti_roundtrip() {
        let dims = (4, 4, 4);
        let n = dims.0 * dims.1 * dims.2;
        let voxel_size = (1.0, 2.0, 3.0);
        let affine = [
            1.0, 0.0, 0.0, 10.0,
            0.0, 2.0, 0.0, 20.0,
            0.0, 0.0, 3.0, 30.0,
            0.0, 0.0, 0.0, 1.0,
        ];

        // Create test data with known values
        let data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5 + 1.0).collect();

        // Save to temp file
        let tmp_dir = std::env::temp_dir();
        let tmp_path = tmp_dir.join("test_nifti_roundtrip.nii");

        save_nifti_to_file(&tmp_path, &data, dims, voxel_size, &affine).unwrap();

        // Read back
        let loaded = read_nifti_file(&tmp_path).unwrap();

        // Verify dimensions
        assert_eq!(loaded.dims, dims, "Dimensions should match");

        // Verify voxel sizes
        assert!((loaded.voxel_size.0 - voxel_size.0).abs() < 1e-5, "Voxel size X mismatch");
        assert!((loaded.voxel_size.1 - voxel_size.1).abs() < 1e-5, "Voxel size Y mismatch");
        assert!((loaded.voxel_size.2 - voxel_size.2).abs() < 1e-5, "Voxel size Z mismatch");

        // Verify data values (saved as f32, so some precision loss expected)
        assert_eq!(loaded.data.len(), n, "Data length should match");
        for i in 0..n {
            assert!(
                (loaded.data[i] - data[i]).abs() < 0.01,
                "Data mismatch at index {}: expected {}, got {}",
                i, data[i], loaded.data[i]
            );
        }

        // Cleanup
        std::fs::remove_file(&tmp_path).ok();
    }

    #[test]
    fn test_save_and_read_nifti_f32() {
        let dims = (4, 4, 4);
        let n = dims.0 * dims.1 * dims.2;
        let voxel_size = (1.5, 1.5, 1.5);
        let affine = [
            1.5, 0.0, 0.0, 0.0,
            0.0, 1.5, 0.0, 0.0,
            0.0, 0.0, 1.5, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];

        // Create small f32-precision data
        let data: Vec<f64> = (0..n).map(|i| (i as f32 * 0.1) as f64).collect();

        let tmp_dir = std::env::temp_dir();
        let tmp_path = tmp_dir.join("test_nifti_f32.nii");

        save_nifti_to_file(&tmp_path, &data, dims, voxel_size, &affine).unwrap();
        let loaded = read_nifti_file(&tmp_path).unwrap();

        assert_eq!(loaded.dims, dims);
        assert_eq!(loaded.data.len(), n);

        // f32 precision: data is saved as f32, so roundtrip should be exact for f32 values
        for i in 0..n {
            assert!(
                (loaded.data[i] - data[i]).abs() < 1e-5,
                "f32 data mismatch at index {}: expected {}, got {}",
                i, data[i], loaded.data[i]
            );
        }

        std::fs::remove_file(&tmp_path).ok();
    }

    #[test]
    fn test_save_nifti_gzip() {
        let dims = (4, 4, 4);
        let n = dims.0 * dims.1 * dims.2;
        let voxel_size = (1.0, 1.0, 1.0);
        let affine = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];

        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();

        let tmp_dir = std::env::temp_dir();
        let tmp_path = tmp_dir.join("test_nifti_gz.nii.gz");

        save_nifti_to_file(&tmp_path, &data, dims, voxel_size, &affine).unwrap();

        // Verify the file is actually gzip compressed
        let bytes = std::fs::read(&tmp_path).unwrap();
        assert!(is_gzip(&bytes), "File should be gzip compressed");

        // Read it back
        let loaded = read_nifti_file(&tmp_path).unwrap();
        assert_eq!(loaded.dims, dims);
        assert_eq!(loaded.data.len(), n);

        for i in 0..n {
            assert!(
                (loaded.data[i] - data[i]).abs() < 0.01,
                "Gzip roundtrip mismatch at index {}: expected {}, got {}",
                i, data[i], loaded.data[i]
            );
        }

        std::fs::remove_file(&tmp_path).ok();
    }

    #[test]
    fn test_load_nifti_invalid_bytes() {
        // Invalid bytes should return an error
        let result = load_nifti(&[0u8; 10]);
        assert!(result.is_err(), "Loading invalid bytes should error");
    }

    #[test]
    fn test_load_nifti_invalid_gzip() {
        // Bytes that look like gzip but are corrupt
        let result = load_nifti(&[0x1f, 0x8b, 0x00, 0x00, 0x00]);
        assert!(result.is_err(), "Loading invalid gzip should error");
    }

    #[test]
    fn test_get_header_info_small_file() {
        let info = get_header_info(&[0u8; 10]);
        assert!(info.contains("too small"), "Should report file too small");
    }

    #[test]
    fn test_get_header_info_normal() {
        // Create a 348-byte mock header
        let mut bytes = vec![0u8; 348];
        // sizeof_hdr at offset 0
        bytes[0..4].copy_from_slice(&348i32.to_le_bytes());
        // magic at offset 344
        bytes[344..348].copy_from_slice(b"n+1\0");
        // datatype at offset 70
        bytes[70..72].copy_from_slice(&16i16.to_le_bytes());

        let info = get_header_info(&bytes);
        assert!(info.contains("sizeof_hdr=348"), "Should contain sizeof_hdr");
        assert!(info.contains("datatype=16"), "Should contain datatype");
    }

    #[test]
    fn test_affine_sform() {
        // Test with sform_code > 0
        let mut header = NiftiHeader::default();
        header.sform_code = 1;
        header.srow_x = [1.0, 0.0, 0.0, 10.0];
        header.srow_y = [0.0, 2.0, 0.0, 20.0];
        header.srow_z = [0.0, 0.0, 3.0, 30.0];

        let affine = get_affine(&header);
        assert_eq!(affine[0], 1.0);
        assert_eq!(affine[3], 10.0);
        assert_eq!(affine[5], 2.0);
        assert_eq!(affine[7], 20.0);
        assert_eq!(affine[10], 3.0);
        assert_eq!(affine[11], 30.0);
        assert_eq!(affine[15], 1.0);
    }

    /// Niklas's own affine from QSMxT/QSMxT#240, as `fslhd` printed its `sto_xyz`. An oblique
    /// sagittal acquisition — and, as it happens, a left-handed one, so it only round-trips if
    /// `qfac` is handled.
    const ISSUE_240_AFFINE: [f64; 16] = [
        0.0, 0.0, 0.600000, -88.492340,
        0.520448, 0.300481, 0.0, -151.538284,
        0.300481, -0.520448, 0.0, 53.045219,
        0.0, 0.0, 0.0, 1.0,
    ];
    const ISSUE_240_VOXEL: (f64, f64, f64) = (0.600962, 0.600962, 0.6);

    /// The qform a reader finds in written header bytes: `(qform_code, qto_xyz)`, with the
    /// transform rebuilt the way the spec says to, from the stored quaternion, `pixdim` and
    /// `qfac`. Deliberately goes through the bytes rather than through [`qform_for`], so a header
    /// written to the wrong offsets fails the test instead of passing it.
    fn read_qform(bytes: &[u8]) -> (i16, [f64; 16]) {
        let f32_at = |o: usize| f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()) as f64;
        let code = i16::from_le_bytes([bytes[252], bytes[253]]);
        let q = QForm {
            quatern: [f32_at(256), f32_at(260), f32_at(264)],
            offset: [f32_at(268), f32_at(272), f32_at(276)],
            qfac: if f32_at(76) < 0.0 { -1.0 } else { 1.0 },
        };
        (code, affine_from_quatern(&q, (f32_at(80), f32_at(84), f32_at(88))))
    }

    /// The sform a reader finds in written header bytes.
    fn read_sform(bytes: &[u8]) -> (i16, [f64; 16]) {
        let f32_at = |o: usize| f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()) as f64;
        let mut a = [0.0f64; 16];
        for row in 0..3 {
            for col in 0..4 {
                a[4 * row + col] = f32_at(280 + 16 * row + 4 * col);
            }
        }
        a[15] = 1.0;
        (i16::from_le_bytes([bytes[254], bytes[255]]), a)
    }

    fn max_abs_diff(a: &[f64; 16], b: &[f64; 16]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
    }

    /// QSMxT/QSMxT#240: the writer set `sform_code` and the `srow_*` rows and left `qform_code`
    /// and the quaternion at zero, so `fslhd` showed every output with a `qto_xyz` of
    /// `diag(pixdim)` and no translation — a canonical axial frame that contradicted the sform
    /// beside it. Both transforms now describe the same geometry.
    #[test]
    fn qform_agrees_with_sform_on_an_oblique_affine() {
        let bytes = save_nifti(&vec![0.0; 24], (2, 3, 4), ISSUE_240_VOXEL, &ISSUE_240_AFFINE).unwrap();
        let (qcode, qto) = read_qform(&bytes);
        let (scode, sto) = read_sform(&bytes);
        assert_eq!((qcode, scode), (1, 1), "both transforms should be advertised");
        // f32 storage of a ~150 mm translation is the limiting term.
        assert!(max_abs_diff(&qto, &sto) < 1e-4, "qto_xyz {qto:?} vs sto_xyz {sto:?}");
        assert!(max_abs_diff(&sto, &ISSUE_240_AFFINE) < 1e-4, "sform drifted: {sto:?}");
    }

    /// A rotation matrix with a negative determinant is a reflection, which no quaternion
    /// describes; `qfac = -1` is how NIfTI carries it. The affine in #240 is one, so getting this
    /// wrong would emit a mirrored transform — left and right swapped — rather than a visibly
    /// broken one.
    #[test]
    fn a_left_handed_affine_is_written_with_qfac_minus_one() {
        for (label, voxel, affine) in [
            ("issue 240", ISSUE_240_VOXEL, ISSUE_240_AFFINE),
            ("LAS", (0.8, 0.8, 3.0), [
                -0.8, 0.0, 0.0, 60.0,
                0.0, 0.8, 0.0, -70.0,
                0.0, 0.0, 3.0, -25.0,
                0.0, 0.0, 0.0, 1.0,
            ]),
        ] {
            let bytes = save_nifti(&vec![0.0; 24], (2, 3, 4), voxel, &affine).unwrap();
            let qfac = f32::from_le_bytes(bytes[76..80].try_into().unwrap());
            assert_eq!(qfac, -1.0, "{label}: pixdim[0] should flag the reflection");
            let (code, qto) = read_qform(&bytes);
            assert_eq!(code, 1, "{label}");
            assert!(max_abs_diff(&qto, &affine) < 1e-4, "{label}: {qto:?}");

            // And the flag is load-bearing: read as right-handed, the same quaternion mirrors the
            // third column. Without it the file would look fine and be wrong.
            let f32_at = |o: usize| f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()) as f64;
            let ignored = QForm {
                quatern: [f32_at(256), f32_at(260), f32_at(264)],
                offset: [f32_at(268), f32_at(272), f32_at(276)],
                qfac: 1.0,
            };
            let wrong = affine_from_quatern(&ignored, (f32_at(80), f32_at(84), f32_at(88)));
            assert!(max_abs_diff(&wrong, &affine) > 0.1, "{label}: qfac changed nothing");
        }
    }

    /// `pixdim[0] = 0` is not a legal qfac, and a zeroed header buffer leaves it there. The
    /// writer always put 1.0, which was legal but only correct by luck.
    #[test]
    fn pixdim_zero_is_never_written_as_qfac() {
        for affine in [
            ISSUE_240_AFFINE,
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            // Sheared, so no qform is written and qfac falls back to +1.
            [1.0, 0.3, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ] {
            let vs = voxel_sizes(&affine);
            let bytes = save_nifti(&vec![0.0; 24], (2, 3, 4), vs, &affine).unwrap();
            let qfac = f32::from_le_bytes(bytes[76..80].try_into().unwrap());
            assert!(qfac == 1.0 || qfac == -1.0, "pixdim[0] = {qfac}");
        }
    }

    fn voxel_sizes(affine: &[f64; 16]) -> (f64, f64, f64) {
        let n = |c: usize| {
            (affine[c] * affine[c] + affine[4 + c] * affine[4 + c] + affine[8 + c] * affine[8 + c])
                .sqrt()
        };
        (n(0), n(1), n(2))
    }

    /// A qform cannot express shear, and a qform that silently disagrees with the sform is the
    /// bug this whole thing is about. So an affine no quaternion reproduces is written with
    /// `qform_code = 0`, which is spec-legal and sends readers to the sform.
    #[test]
    fn an_affine_no_quaternion_describes_gets_no_qform() {
        let cases: [(&str, (f64, f64, f64), [f64; 16]); 3] = [
            ("sheared", (1.0, 1.0, 1.0), [
                1.0, 0.3, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ]),
            // pixdim is what scales the qform's rotation, so a voxel size that contradicts the
            // affine's column norms makes the two transforms differ even with an exact rotation.
            ("pixdim disagrees with the affine", (1.0, 1.0, 1.0), [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 2.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ]),
            ("singular", (1.0, 1.0, 1.0), [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ]),
        ];
        for (label, voxel, affine) in cases {
            let bytes = save_nifti(&vec![0.0; 24], (2, 3, 4), voxel, &affine).unwrap();
            assert_eq!(read_qform(&bytes).0, 0, "{label}: should advertise no qform");
            assert_eq!(read_sform(&bytes).0, 1, "{label}: the sform still carries the geometry");
            // The quaternion fields stay zero, so nothing is left for a reader to misinterpret.
            assert!(bytes[256..280].iter().all(|&b| b == 0), "{label}");
        }
    }

    /// Both transforms survive a write/read cycle, and `get_affine` returns the same geometry
    /// whichever one a reader is left with.
    #[test]
    fn qform_and_sform_round_trip_to_the_same_affine() {
        let affines: [(&str, (f64, f64, f64), [f64; 16]); 4] = [
            ("axial anisotropic", (1.0, 1.0, 2.0), [
                1.0, 0.0, 0.0, -10.0,
                0.0, 1.0, 0.0, -20.0,
                0.0, 0.0, 2.0, -30.0,
                0.0, 0.0, 0.0, 1.0,
            ]),
            ("oblique (issue 240)", ISSUE_240_VOXEL, ISSUE_240_AFFINE),
            ("left-handed", (0.8, 0.8, 3.0), [
                -0.8, 0.0, 0.0, 60.0,
                0.0, 0.8, 0.0, -70.0,
                0.0, 0.0, 3.0, -25.0,
                0.0, 0.0, 0.0, 1.0,
            ]),
            // Sagittal: a cardinal permutation, where the quaternion is a quarter turn.
            ("sagittal anisotropic", (0.8, 0.8, 3.0), [
                0.0, 0.0, 3.0, -90.0,
                0.8, 0.0, 0.0, -110.0,
                0.0, 0.8, 0.0, -70.0,
                0.0, 0.0, 0.0, 1.0,
            ]),
        ];
        for (label, voxel, affine) in affines {
            let bytes = save_nifti(&vec![0.0; 24], (2, 3, 4), voxel, &affine).unwrap();
            let (qcode, qto) = read_qform(&bytes);
            let (scode, sto) = read_sform(&bytes);
            assert_eq!((qcode, scode), (1, 1), "{label}");
            assert!(max_abs_diff(&qto, &sto) < 1e-4, "{label}: qto {qto:?} sto {sto:?}");

            // And through the loader, which is what the rest of the crate sees.
            let loaded = load_nifti(&bytes).unwrap();
            assert!(max_abs_diff(&loaded.affine, &affine) < 1e-4, "{label}: {:?}", loaded.affine);
        }
    }

    /// Plenty of converters write a qform and no sform. Falling straight through to the bare
    /// `pixdim` scaling there throws the orientation away: a sagittal acquisition comes back
    /// looking axial, which is the same failure #240 reported, with the roles reversed.
    #[test]
    fn get_affine_reads_the_qform_when_there_is_no_sform() {
        // Sagittal 0.8 x 0.8 x 3 mm: voxel axis 0 runs A-P, axis 1 S-I, axis 2 L-R.
        let affine = [
            0.0, 0.0, 3.0, -90.0,
            0.8, 0.0, 0.0, -110.0,
            0.0, 0.8, 0.0, 70.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let voxel = (0.8, 0.8, 3.0);
        let q = qform_for(&affine, voxel).expect("a cardinal permutation has a quaternion");

        let mut header = NiftiHeader::default();
        header.sform_code = 0;
        header.qform_code = 1;
        header.quatern_b = q.quatern[0] as f32;
        header.quatern_c = q.quatern[1] as f32;
        header.quatern_d = q.quatern[2] as f32;
        header.quatern_x = q.offset[0] as f32;
        header.quatern_y = q.offset[1] as f32;
        header.quatern_z = q.offset[2] as f32;
        header.pixdim = [q.qfac as f32, 0.8, 0.8, 3.0, 1.0, 1.0, 1.0, 1.0];

        let got = get_affine(&header);
        assert!(max_abs_diff(&got, &affine) < 1e-4, "{got:?}");
        // The old behaviour, for contrast: diag(pixdim), which puts the 3 mm slice pitch on L-R
        // and calls a sagittal stack axial.
        header.qform_code = 0;
        let fallback = get_affine(&header);
        assert!((fallback[0] - 0.8).abs() < 1e-6, "fallback should be the bare voxel scaling");
        assert!(max_abs_diff(&fallback, &affine) > 1.0);
    }

    #[test]
    fn test_save_nifti_header_details() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2x2x2
        let dims = (2, 2, 2);
        let voxel_size = (1.5, 2.5, 3.5);
        let affine = [
            1.5, 0.0, 0.0, 5.0,
            0.0, 2.5, 0.0, 10.0,
            0.0, 0.0, 3.5, 15.0,
            0.0, 0.0, 0.0, 1.0,
        ];

        let bytes = save_nifti(&data, dims, voxel_size, &affine).unwrap();

        // Verify datatype = 16 (FLOAT32)
        let datatype = i16::from_le_bytes([bytes[70], bytes[71]]);
        assert_eq!(datatype, 16);

        // Verify bitpix = 32
        let bitpix = i16::from_le_bytes([bytes[72], bytes[73]]);
        assert_eq!(bitpix, 32);

        // Verify dim[0] = 3
        let ndim = i16::from_le_bytes([bytes[40], bytes[41]]);
        assert_eq!(ndim, 3);

        // Verify dim[1] = 2
        let nx = i16::from_le_bytes([bytes[42], bytes[43]]);
        assert_eq!(nx, 2);

        // Verify vox_offset = 352
        let vox_offset = f32::from_le_bytes([bytes[108], bytes[109], bytes[110], bytes[111]]);
        assert_eq!(vox_offset, 352.0);

        // Verify scl_slope = 1.0
        let scl_slope = f32::from_le_bytes([bytes[112], bytes[113], bytes[114], bytes[115]]);
        assert_eq!(scl_slope, 1.0);

        // Verify sform_code = 1
        let sform_code = i16::from_le_bytes([bytes[254], bytes[255]]);
        assert_eq!(sform_code, 1);

        // Verify pixdim[1] matches voxel_size
        let pixdim1 = f32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]);
        assert!((pixdim1 - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_save_nifti_data_values() {
        let data = vec![1.0f64, 2.0, -3.0, 4.5, 0.0, 100.0, -0.5, 999.0]; // 2x2x2
        let dims = (2, 2, 2);
        let voxel_size = (1.0, 1.0, 1.0);
        let affine = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];

        let bytes = save_nifti(&data, dims, voxel_size, &affine).unwrap();

        // Data starts at offset 352
        for i in 0..8 {
            let offset = 352 + i * 4;
            let val = f32::from_le_bytes([
                bytes[offset], bytes[offset + 1],
                bytes[offset + 2], bytes[offset + 3],
            ]);
            assert!(
                (val as f64 - data[i]).abs() < 0.01,
                "Data value {} mismatch: saved {}, expected {}",
                i, val, data[i]
            );
        }
    }

    #[test]
    fn test_save_nifti_gz_bytes() {
        let data = vec![0.0; 8]; // 2x2x2
        let dims = (2, 2, 2);
        let voxel_size = (1.0, 1.0, 1.0);
        let affine = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];

        let bytes = save_nifti_gz(&data, dims, voxel_size, &affine).unwrap();
        assert!(is_gzip(&bytes), "save_nifti_gz should produce gzip bytes");

        // Should be able to load it back
        let loaded = load_nifti(&bytes).unwrap();
        assert_eq!(loaded.dims, dims);
    }

    #[test]
    fn test_read_nonexistent_file() {
        let result = read_nifti_file(std::path::Path::new("/tmp/nonexistent_file_12345.nii"));
        assert!(result.is_err(), "Reading nonexistent file should error");
        match result {
            Err(err) => {
                assert!(err.contains("Failed to read file"), "Error should mention file reading: {}", err);
            }
            Ok(_) => panic!("Should have returned an error"),
        }
    }

    #[test]
    fn test_save_nifti_large_volume() {
        // Test with 8x8x8 volume (512 elements)
        let dims = (8, 8, 8);
        let n = dims.0 * dims.1 * dims.2;
        let voxel_size = (0.5, 0.5, 0.5);
        let affine = [
            0.5, 0.0, 0.0, -2.0,
            0.0, 0.5, 0.0, -2.0,
            0.0, 0.0, 0.5, -2.0,
            0.0, 0.0, 0.0, 1.0,
        ];

        let data: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();

        let bytes = save_nifti(&data, dims, voxel_size, &affine).unwrap();
        assert_eq!(bytes.len(), 352 + n * 4);

        // Load it back
        let loaded = load_nifti(&bytes).unwrap();
        assert_eq!(loaded.dims, dims);
        assert_eq!(loaded.data.len(), n);

        for i in 0..n {
            assert!(
                (loaded.data[i] - data[i]).abs() < 0.01,
                "Roundtrip mismatch at {}: expected {}, got {}",
                i, data[i], loaded.data[i]
            );
        }
    }

    #[test]
    fn test_nifti_roundtrip_affine() {
        // Verify affine is preserved through save/load
        let dims = (4, 4, 4);
        let n = dims.0 * dims.1 * dims.2;
        let voxel_size = (1.0, 2.0, 3.0);
        let affine = [
            1.0, 0.1, 0.2, 10.0,
            0.3, 2.0, 0.4, 20.0,
            0.5, 0.6, 3.0, 30.0,
            0.0, 0.0, 0.0, 1.0,
        ];

        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();

        let tmp_dir = std::env::temp_dir();
        let tmp_path = tmp_dir.join("test_nifti_affine_rt.nii");

        save_nifti_to_file(&tmp_path, &data, dims, voxel_size, &affine).unwrap();
        let loaded = read_nifti_file(&tmp_path).unwrap();

        // Affine values are stored as f32, so expect f32-level precision
        for i in 0..16 {
            assert!(
                (loaded.affine[i] - affine[i]).abs() < 0.01,
                "Affine[{}] mismatch: expected {}, got {}",
                i, affine[i], loaded.affine[i]
            );
        }

        std::fs::remove_file(&tmp_path).ok();
    }

    #[test]
    fn test_nifti_scl_slope_intercept() {
        // Test that scl_slope and scl_inter are reported correctly
        let dims = (4, 4, 4);
        let n = dims.0 * dims.1 * dims.2;
        let voxel_size = (1.0, 1.0, 1.0);
        let affine = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];

        let data = vec![1.0; n];
        let bytes = save_nifti(&data, dims, voxel_size, &affine).unwrap();
        let loaded = load_nifti(&bytes).unwrap();

        // Our save sets scl_slope = 1.0 and scl_inter = 0.0
        assert!((loaded.scl_slope - 1.0).abs() < 1e-5);
        assert!((loaded.scl_inter - 0.0).abs() < 1e-5);
    }
}

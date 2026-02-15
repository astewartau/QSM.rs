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

    let nx = dim[1] as usize;
    let ny = dim[2] as usize;
    let nz = dim[3] as usize;

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
    let ndim = dim[0] as usize;

    let nx = dim[1] as usize;
    let ny = dim[2] as usize;
    let nz = dim[3] as usize;
    let nt = if ndim >= 4 { dim[4] as usize } else { 1 };

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
    // Prefer sform if available (sform_code > 0)
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

    // pixdim[0..7]
    let pixdim: [f32; 8] = [1.0, vsx as f32, vsy as f32, vsz as f32, 1.0, 1.0, 1.0, 1.0];
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

    // sform_code = 1 (scanner anat)
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
        save_nifti_gz(data, dims, voxel_size, affine)?
    } else {
        save_nifti(data, dims, voxel_size, affine)?
    };

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

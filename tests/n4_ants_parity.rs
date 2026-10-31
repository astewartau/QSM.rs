//! `utils::n4_bias_correction` against ANTs 2.4.3's `N4BiasFieldCorrection` on a small synthetic
//! case.
//!
//! `tests/data/n4_ants_small.bin` is made by `tests/data/n4_ants_small.py` with the verified QSM-CI
//! `ants-n4` image (the official ANTs 2.4.3 release binary, ITK 4 threads; QSMxT/QSM-CI#302), so it
//! can be regenerated in Actions (`.github/workflows/regenerate-fixtures.yml`). Little-endian f32,
//! column-major, 30x32x26 grid, voxel 1x1x1.5 mm, identity orientation; it holds:
//! - the input: three intensity classes (60, 100, 160) in a sphere, Gaussian noise (σ = 3) inside
//!   it, times the bias `exp(0.35x − 0.25y + 0.2z²)` (x, y, z in [−1, 1] across the grid), zero
//!   outside;
//! - the bias field from `N4BiasFieldCorrection -i in.nii -o [n4.nii,bias.nii]` (all defaults);
//! - the corrected image from `N4BiasFieldCorrection -i in.nii -x mask.nii -o n4m.nii` with the
//!   mask `input > 30` (so with ANTs' default rescaling to the input's range in the mask);
//! - the bias field of the same input written with an oblique scanner header (`oblique.nii`, the
//!   sform of QSM-CI's in-vivo prisma-bridge-run1, 1 mm), all defaults. On that header ITK's
//!   shrink starts one voxel earlier along y than on an axis-aligned one (astewartau/QSM.rs#327).
//!
//! ITK runs N4 in single precision and multithreaded, and the rounding is amplified over the
//! iterations: ANTs' own results here change by up to 2e-5 (bias field) and 3e-4 (masked,
//! rescaled output) relative between thread counts (the fixture used to be made with another
//! ANTs build and thread count, and moved by 1.1e-5 and 7.7e-5 when remade with the container).
//! The tolerances allow for that.

use qsm_core::utils::{n4_bias_correction, ItkGeometry, N4Params};
use qsm_core::Grid;

const DIMS: (usize, usize, usize) = (30, 32, 26);

/// What ITK reads from the generator's `nib.Nifti1Image(img, diag(1, 1, 1.5))`: sform (code 2)
/// diag(1, 1, 1.5) at the origin, qform code 0, i.e. LPS direction diag(−1, −1, 1).
fn geometry() -> ItkGeometry {
    ItkGeometry { origin: [0.0; 3], spacing: [1.0, 1.0, 1.5], direction: [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]] }
}

/// `oblique.nii`'s header, the float32 bit patterns the generator writes.
fn oblique_header() -> nifti::NiftiHeader {
    let f = f32::from_bits;
    nifti::NiftiHeader {
        dim: [3, DIMS.0 as u16, DIMS.1 as u16, DIMS.2 as u16, 1, 1, 1, 1],
        pixdim: [1.0; 8],
        qform_code: 0,
        sform_code: 2,
        srow_x: [f(0xbf800000), f(0xacac4c41), 0.0, f(0x42b602eb)],
        srow_y: [f(0xacac4c41), f(0x3f800000), f(0x80000000), f(0xc2e31894)],
        srow_z: [0.0, 0.0, f(0x3f800000), f(0xc21eab36)],
        ..Default::default()
    }
}

fn load_oblique() -> Vec<f64> {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/n4_ants_small.bin")).expect("fixture");
    let n = DIMS.0 * DIMS.1 * DIMS.2;
    (0..n).map(|i| f32::from_le_bytes(raw[4 * (3 * n + i)..4 * (3 * n + i) + 4].try_into().unwrap()) as f64).collect()
}

fn load() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/n4_ants_small.bin")).expect("fixture");
    let n = DIMS.0 * DIMS.1 * DIMS.2;
    assert_eq!(raw.len(), 4 * 4 * n);
    let f = |a: usize| -> Vec<f64> {
        (0..n).map(|i| f32::from_le_bytes(raw[4 * (a * n + i)..4 * (a * n + i) + 4].try_into().unwrap()) as f64).collect()
    };
    (f(0), f(1), f(2))
}

fn max_rel(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).filter(|(_, y)| **y != 0.0).map(|(x, y)| ((x - y) / y).abs()).fold(0.0, f64::max)
}

#[test]
fn n4_matches_ants_defaults() {
    let (input, ants_bias, _) = load();
    let grid = Grid::new(DIMS.0, DIMS.1, DIMS.2, 1.0, 1.0, 1.5);
    let out = n4_bias_correction(&input, None, &grid, &geometry(), &N4Params::default(), |_, _| {}).unwrap();
    let d = max_rel(&out.bias_field, &ants_bias);
    println!("bias field vs ANTs: max relative difference {d:.2e}");
    assert!(d < 2e-4, "bias field differs from ANTs by {d}");
}

#[test]
fn n4_matches_ants_with_mask() {
    let (input, _, ants_masked) = load();
    let grid = Grid::new(DIMS.0, DIMS.1, DIMS.2, 1.0, 1.0, 1.5);
    let mask: Vec<u8> = input.iter().map(|&v| (v > 30.0) as u8).collect();
    let out = n4_bias_correction(&input, Some(&mask), &grid, &geometry(), &N4Params::default(), |_, _| {}).unwrap();
    let d = max_rel(&out.corrected, &ants_masked);
    println!("masked, rescaled output vs ANTs: max relative difference {d:.2e}");
    assert!(d < 1e-3, "corrected image differs from ANTs by {d}");
}

#[test]
fn n4_matches_ants_on_an_oblique_header() {
    // ITK reads the header, maps the shrunk grid's first voxel back to an input index and rounds
    // a half-integer the way its arithmetic leaves it: here (3, 1, 3), where the axis-aligned
    // header of the other cases gives (3, 2, 3).
    let (input, _, _) = load();
    let ants_bias = load_oblique();
    let grid = Grid::new(DIMS.0, DIMS.1, DIMS.2, 1.0, 1.0, 1.0);
    let geo = ItkGeometry::from_nifti_header(&oblique_header()).unwrap();
    assert_eq!(geo.shrink_offsets(grid.dims, 4).unwrap(), [3, 1, 3]);
    let out = n4_bias_correction(&input, None, &grid, &geo, &N4Params::default(), |_, _| {}).unwrap();
    let d = max_rel(&out.bias_field, &ants_bias);
    println!("oblique header: bias field vs ANTs: max relative difference {d:.2e}");
    assert!(d < 2e-4, "bias field differs from ANTs by {d}");
    // Ignoring the header (axis-aligned geometry) shifts the samples by a voxel along y.
    let aligned = ItkGeometry::axis_aligned([1.0; 3]);
    assert_eq!(aligned.shrink_offsets(grid.dims, 4).unwrap(), [3, 2, 3]);
    let off = n4_bias_correction(&input, None, &grid, &aligned, &N4Params::default(), |_, _| {}).unwrap();
    let d_off = max_rel(&off.bias_field, &ants_bias);
    println!("oblique header read as axis-aligned: max relative difference {d_off:.2e}");
    assert!(d_off > 10.0 * d, "the header should matter: {d_off} vs {d}");
}

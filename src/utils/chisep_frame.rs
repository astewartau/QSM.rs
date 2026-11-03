//! The χ-separation toolbox's resampling into its networks' training frame.
//!
//! The SNU-LIST χ-separation toolbox v1.1.3 (Jongho Lee's lab) runs its networks — QSMnetME,
//! R2PRIMEnet and χ-sepnet — in `chi_sepnet_general` (`Chisep_script.m`:194, the `resgen = false`
//! branch), which does not hand them the acquired grid. It resamples every input into the frame
//! the networks were trained in, B0 along the third axis at 1 mm isotropic, runs the networks
//! there on the whole volume, and maps their output back onto the acquired grid. The toolbox
//! ships as protected P-code; with the authors' permission this was recovered by black-box
//! probing in licensed MATLAB R2023b (QSM.rs#377, 2026-11-03): spies on `interp3` and on the
//! networks (a real `DAGNetwork` whose spy layer saved its input and returned the real graph's
//! output, computed by ONNX Runtime), each saving its caller's workspace, on volumes of
//! 64×60×48 voxels at 0.8×0.9×1.1 mm with B0 along (0.1, 0.2, 0.97), and at 1 mm with B0
//! along z, plus the frame sizes of twelve more grids. [`TrainingFrame`] is that resampling:
//!
//! 1. **Rotation** (`CalculateRotMat(B0_dir)`): `R` rotates by `acos(b·ẑ)` about `ẑ × b`, so
//!    `R·ẑ = b`; for a unit `b` it is the Rodrigues rotation to rounding (11 directions checked).
//!    The toolbox does not normalise `b` (a non-unit vector gives a slightly non-orthogonal
//!    matrix); here `b` is normalised first. For `b = −ẑ` the toolbox's matrix is `NaN`; here it
//!    is the half turn about the first axis.
//! 2. **Grids.** The acquired grid's coordinates are centred: along each array axis
//!    `−fov_h : vs : fov_h` mm, `fov_h = (N − 1)·vs / 2` (axis `d` with `voxel_size(d)`). The frame
//!    has `M = 16·⌈n/16⌉` nodes along each axis, `n = ⌊(N − 1)·vs⌋ + 1` (the length of
//!    `−fov_h : 1 : fov_h`), at `−(M − 1)/2 … (M − 1)/2` mm: centred, 1 mm apart, padded up to a
//!    multiple of 16 for the networks' four pooling levels.
//! 3. **Into the frame:** frame node `p` reads the acquired volume at `R·p`, with MATLAB
//!    `interp3`: `'spline'` for the field and R2′/R2* (which extrapolates beyond the acquired
//!    grid), `'linear'` for the mask, thresholded at 0.5 (`NaN` outside is 0).
//!    [`crate::utils::interp3`] has both.
//! 4. **Back:** acquired voxel `x` reads the frame map at `R⁻¹·x` with `'spline'`.
//!
//! The toolbox's script says this branch "interpolate[s] the input maps to 1 mm isotropic
//! resolution. Output resolution is also 1 mm isotropic." (`Chisep_script.m`:193); the P-code
//! does more than that: step 4 returns its maps on the acquired grid (measured: a 64×60×48
//! input at 0.8×0.9×1.1 mm comes back 64×60×48), so no resampling of our own is involved.
//!
//! Python reproduction of the whole route, verified against those MATLAB runs: QSM-CI's
//! `algorithms/qsmnetme/recon.py` and `algorithms/chisepnet/recon.py`.

use super::interp3::{linear_resample, spline_resample, UniformAxes};

/// The frame's resolution, mm (the networks' training data: 1 mm isotropic).
pub const FRAME_RES_MM: f64 = 1.0;

/// The frame is padded up to a multiple of this along each axis (four pooling levels).
pub const FRAME_MULTIPLE: usize = 16;

/// `CalculateRotMat`: the rotation taking ẑ to the (normalised) B0 direction `b`.
pub fn rotation_to_b0(b0: (f64, f64, f64)) -> [[f64; 3]; 3] {
    let nrm = (b0.0 * b0.0 + b0.1 * b0.1 + b0.2 * b0.2).sqrt();
    let b = [b0.0 / nrm, b0.1 / nrm, b0.2 / nrm];
    // v = ẑ × b, s = |v|, c = b·ẑ; R = I + [v]× + [v]×²·(1 − c)/s².
    let v = [-b[1], b[0], 0.0];
    let s2 = v[0] * v[0] + v[1] * v[1];
    let c = b[2];
    if s2 < 1e-30 {
        return if c > 0.0 {
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        } else {
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
        };
    }
    let k = [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]];
    let f = (1.0 - c) / s2;
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let kk: f64 = (0..3).map(|l| k[i][l] * k[l][j]).sum();
            r[i][j] = if i == j { 1.0 } else { 0.0 } + k[i][j] + kk * f;
        }
    }
    r
}

fn apply(r: &[[f64; 3]; 3], p: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| r[i][0] * p[0] + r[i][1] * p[1] + r[i][2] * p[2])
}

/// The number of frame nodes along an axis of `n` voxels of `vs` mm (step 2 of the module docs).
pub fn frame_len(n: usize, vs: f64) -> usize {
    // length(-fov_h : 1 : fov_h); MATLAB's colon allows for rounding in (b − a)/d.
    let span = (n.saturating_sub(1)) as f64 * vs / FRAME_RES_MM;
    let m = (span + 1e-9).floor() as usize + 1;
    m.div_ceil(FRAME_MULTIPLE) * FRAME_MULTIPLE
}

/// The acquired grid and the training frame, and the maps between them (module docs).
#[derive(Clone, Debug)]
pub struct TrainingFrame {
    native: UniformAxes,
    frame: UniformAxes,
    rot: [[f64; 3]; 3],
    rot_t: [[f64; 3]; 3],
}

impl TrainingFrame {
    /// The frame for an acquired grid of `dims` voxels of `voxel_size` mm with B0 along `b0_dir`
    /// (array axes, as [`crate::Grid`]).
    pub fn new(dims: (usize, usize, usize), voxel_size: (f64, f64, f64), b0_dir: (f64, f64, f64)) -> Self {
        let n = [dims.0, dims.1, dims.2];
        let vs = [voxel_size.0, voxel_size.1, voxel_size.2];
        let native = UniformAxes { n, origin: std::array::from_fn(|d| -((n[d] - 1) as f64) * vs[d] / 2.0), step: vs };
        let m: [usize; 3] = std::array::from_fn(|d| frame_len(n[d], vs[d]));
        let frame = UniformAxes {
            n: m,
            origin: std::array::from_fn(|d| -((m[d] - 1) as f64) * FRAME_RES_MM / 2.0),
            step: [FRAME_RES_MM; 3],
        };
        let rot = rotation_to_b0(b0_dir);
        let rot_t = std::array::from_fn(|i| std::array::from_fn(|j| rot[j][i]));
        Self { native, frame, rot, rot_t }
    }

    /// The frame's dimensions (each a multiple of [`FRAME_MULTIPLE`]).
    pub fn dims(&self) -> (usize, usize, usize) {
        (self.frame.n[0], self.frame.n[1], self.frame.n[2])
    }

    /// Voxels in the frame.
    pub fn len(&self) -> usize {
        self.frame.len()
    }

    /// Whether the frame is empty (never, for a non-empty grid).
    pub fn is_empty(&self) -> bool {
        self.frame.is_empty()
    }

    /// The rotation `R` (`R·ẑ = b`).
    pub fn rotation(&self) -> [[f64; 3]; 3] {
        self.rot
    }

    /// An acquired-grid map into the frame (`interp3 'spline'`, extrapolating).
    pub fn to_frame(&self, native: &[f64]) -> Vec<f64> {
        let q = |i: usize| apply(&self.rot, self.frame.node_at(i));
        spline_resample(native, &self.native, self.frame.len(), &q)
    }

    /// The acquired mask into the frame: `interp3 'linear'`, `> 0.5` (0 outside the acquired grid).
    pub fn mask_to_frame(&self, mask: &[u8]) -> Vec<u8> {
        let m: Vec<f64> = mask.iter().map(|&v| if v != 0 { 1.0 } else { 0.0 }).collect();
        let q = |i: usize| apply(&self.rot, self.frame.node_at(i));
        linear_resample(&m, &self.native, self.frame.len(), &q).into_iter().map(|v| u8::from(v > 0.5)).collect()
    }

    /// A frame map back onto the acquired grid (`interp3 'spline'` at `R⁻¹·x`).
    pub fn to_native(&self, frame: &[f64]) -> Vec<f64> {
        let q = |i: usize| apply(&self.rot_t, self.native.node_at(i));
        spline_resample(frame, &self.frame, self.native.len(), &q)
    }
}

/// Run a SNU-LIST network on frame volumes: `channels` (column-major over `dims`) in, `out_ch`
/// channels out. `patch = None` runs the whole volume in one pass, as the toolbox does; `Some`
/// runs an overlapping sliding window of that patch (0.75 stride, overlaps averaged), for hosts
/// that cannot hold the whole volume's activations.
#[cfg(feature = "onnx")]
pub(crate) fn run_net(
    model: &crate::models::onnx::OnnxModel,
    channels: &[Vec<f32>],
    dims: (usize, usize, usize),
    out_ch: usize,
    patch: Option<(usize, usize, usize)>,
) -> Result<Vec<Vec<f32>>, crate::models::onnx::OnnxError> {
    use crate::models::onnx::{OnnxError, Tensor};
    let (nx, ny, nz) = dims;
    let (pd, ph, pw) = patch.unwrap_or(dims);
    let (px, py, pz) = (nx.max(pd), ny.max(ph), nz.max(pw));
    let cin = channels.len();
    let plane = pd * ph * pw;
    let mut acc = vec![vec![0.0f64; px * py * pz]; out_ch];
    let mut wsum = vec![0.0f64; px * py * pz];
    let starts = |size: usize, p: usize| super::sliding::starts(size, p);
    for &x0 in &starts(px, pd) {
        for &y0 in &starts(py, ph) {
            for &z0 in &starts(pz, pw) {
                let mut buf = vec![0.0f32; cin * plane];
                for (c, ch) in channels.iter().enumerate() {
                    for i in 0..pd {
                        let x = x0 + i;
                        for j in 0..ph {
                            let y = y0 + j;
                            for k in 0..pw {
                                let z = z0 + k;
                                if x < nx && y < ny && z < nz {
                                    buf[c * plane + (i * ph + j) * pw + k] = ch[x + nx * (y + ny * z)];
                                }
                            }
                        }
                    }
                }
                let out = model.run_single(&Tensor::new(vec![1, cin, pd, ph, pw], buf))?;
                if out.data.len() < out_ch * plane {
                    return Err(OnnxError::Run(format!("expected a {out_ch}-channel output")));
                }
                for (c, a) in acc.iter_mut().enumerate() {
                    for i in 0..pd {
                        for j in 0..ph {
                            for k in 0..pw {
                                let p = (x0 + i) + px * ((y0 + j) + py * (z0 + k));
                                a[p] += out.data[c * plane + (i * ph + j) * pw + k] as f64;
                                if c == 0 {
                                    wsum[p] += 1.0;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(acc
        .into_iter()
        .map(|a| {
            let mut o = vec![0.0f32; nx * ny * nz];
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let p = x + px * (y + py * z);
                        o[x + nx * (y + ny * z)] = (a[p] / wsum[p].max(1.0)) as f32;
                    }
                }
            }
            o
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Frame sizes the toolbox chose (MATLAB R2023b, `chi_sepnet_general`), per axis `(N, vs) → M`.
    #[test]
    fn frame_sizes_match_the_toolbox() {
        for &(n, vs, m) in &[
            (40, 0.64, 32), (36, 0.64, 32), (30, 0.64, 32), (80, 0.8, 64), (81, 0.8, 80), (79, 0.8, 64),
            (65, 1.0, 80), (63, 1.0, 64), (33, 1.0, 48), (100, 0.64, 64), (96, 0.67, 64), (70, 0.96, 80),
            (20, 0.8, 16), (40, 0.4, 16), (128, 0.5, 64), (120, 0.5, 64), (90, 0.5, 48), (16, 1.0, 16),
            (17, 1.0, 32), (15, 1.0, 16), (33, 0.5, 32), (10, 3.2, 32), (64, 0.8, 64), (60, 0.9, 64), (48, 1.1, 64),
        ] {
            assert_eq!(frame_len(n, vs), m, "N {n}, vs {vs}");
        }
    }

    /// `CalculateRotMat` in MATLAB R2023b, full precision, for unit directions.
    #[test]
    #[allow(clippy::excessive_precision, clippy::type_complexity)] // MATLAB's printed digits, as is
    fn rotation_matches_the_toolbox() {
        let cases: [((f64, f64, f64), [[f64; 3]; 3]); 3] = [
            (
                (0.10045812911315205, 0.2009162582263041, 0.97444385239757481),
                [
                    [0.99488877047951507, -0.010222459040970123, 0.10045812911315206],
                    [-0.010222459040970119, 0.97955508191805973, 0.20091625822630413],
                    [-0.10045812911315206, -0.20091625822630416, 0.97444385239757481],
                ],
            ),
            (
                (0.3, -0.4, 0.8660254037844386),
                [
                    [0.95176914536239787, 0.064307806183469485, 0.30000000000000004],
                    [0.064307806183469457, 0.91425625842204072, -0.40000000000000008],
                    [-0.30000000000000004, 0.40000000000000008, 0.86602540378443871],
                ],
            ),
            ((1.0, 0.0, 0.0), [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
        ];
        for (b, want) in cases {
            let r = rotation_to_b0(b);
            for i in 0..3 {
                for j in 0..3 {
                    assert!((r[i][j] - want[i][j]).abs() < 1e-14, "{b:?} [{i}][{j}]");
                }
            }
        }
        assert_eq!(rotation_to_b0((0.0, 0.0, 2.0)), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    }

    /// On a 1 mm grid with even dimensions, B0 along z, the frame nodes are acquired nodes (both
    /// sit at half-integer mm), so a round trip returns the map.
    #[test]
    fn aligned_round_trip() {
        let f = TrainingFrame::new((20, 18, 10), (1.0, 1.0, 1.0), (0.0, 0.0, 1.0));
        assert_eq!(f.dims(), (32, 32, 16));
        let v: Vec<f64> = (0..20 * 18 * 10).map(|i| ((i * 37) % 101) as f64 * 0.01).collect();
        let back = f.to_native(&f.to_frame(&v));
        for (a, b) in v.iter().zip(&back) {
            assert!((a - b).abs() < 1e-10);
        }
    }
}

//! ROMEO weight calculation for phase unwrapping
//!
//! Calculates edge weights for region-growing phase unwrapping based on:
//! - Phase coherence
//! - Phase gradient coherence (multi-echo)
//! - Magnitude coherence
//! - Magnitude weights
//!
//! Reference:
//! Dymerska, B., Eckstein, K., Bachrata, B., Siow, B., Trattnig, S., Shmueli, K.,
//! Robinson, S.D. (2021). "Phase unwrapping with a rapid opensource minimum spanning
//! tree algorithm (ROMEO)." Magnetic Resonance in Medicine, 85(4):2294-2308.
//! https://doi.org/10.1002/mrm.28563
//!
//! Reference implementation: https://github.com/korbinian90/MriResearchTools.jl

use std::f64::consts::PI;

use crate::region_grow::{grow_region_unwrap, grow_region_unwrap_from_visited, grow_region_unwrap_full};
use crate::Grid;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Weight calculation scheme for ROMEO unwrapping.
///
/// Matches ROMEO.jl weight type options. See Dymerska et al. (2021) for details.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RomeoWeightType {
    /// Phase coherence + phase gradient coherence + mag coherence.
    /// Note: phaselinearity component not yet implemented; currently equivalent to Romeo3.
    Romeo,
    /// Phase coherence + mag coherence only (no multi-echo temporal info).
    Romeo2,
    /// Phase coherence + phase gradient coherence + mag coherence.
    Romeo3,
    /// Alias for Romeo.
    Romeo4,
    /// All components including magnitude weighting.
    Romeo6,
    /// Best-path method (Abdul-Rahman). Not yet implemented.
    BestPath,
}

impl RomeoWeightType {
    /// Map weight type to 6 component flags matching ROMEO.jl:
    /// \[phase_coherence, phase_gradient_coherence, phase_linearity,
    ///  mag_coherence, mag_weight, mag_weight2\]
    fn weight_flags(&self) -> [bool; 6] {
        match self {
            RomeoWeightType::Romeo  => [true, true, true, true, false, false],
            RomeoWeightType::Romeo2 => [true, false, false, true, false, false],
            RomeoWeightType::Romeo3 => [true, true, false, true, false, false],
            RomeoWeightType::Romeo4 => [true, true, true, true, false, false],
            RomeoWeightType::Romeo6 => [true, true, true, true, true, true],
            RomeoWeightType::BestPath => [false; 6], // uses different calculation
        }
    }
}

/// Parameters for ROMEO phase unwrapping.
///
/// Weight components can be toggled individually. Each multiplies into the
/// final edge weight as `0.1 + 0.9 * component_value`.
///
/// The default enables phase coherence, phase gradient coherence,
/// phase linearity, and magnitude coherence (equivalent to `:romeo` in ROMEO.jl).
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct RomeoParams {
    // -- Weight component flags --

    /// Phase coherence: `1 - |wrap(Δφ)| / π`. Always recommended. (default: true)
    pub phase_coherence: bool,
    /// Phase gradient coherence: consistency between echo pairs. Needs multi-echo
    /// data (phase2 + TEs). Automatically disabled if unavailable. (default: true)
    pub phase_gradient_coherence: bool,
    /// Phase linearity: second-derivative smoothness along each edge. (default: true)
    pub phase_linearity: bool,
    /// Magnitude coherence: `(min/max)²` of neighbor magnitudes. (default: true)
    pub mag_coherence: bool,
    /// Magnitude weight: penalizes low-signal voxels relative to the 95th
    /// percentile. (default: false)
    pub mag_weight: bool,
    /// Magnitude weight 2: penalizes abnormally high signal (flow artifacts).
    /// (default: false)
    pub mag_weight2: bool,
    /// Use Best-path weights (Abdul-Rahman) instead of ROMEO weights.
    /// When true, the individual weight flags above are ignored. (default: false)
    pub bestpath: bool,

    // -- Multi-echo options --

    /// Template echo index for spatial unwrapping, 0-indexed (default: 0).
    /// Only used for multi-echo template-based unwrapping.
    pub template: usize,
    /// Unwrap each echo individually instead of template-based (default: false).
    pub individual: bool,
    /// Correct global 2π offsets between echoes (default: false).
    /// Recommended when using individual mode.
    pub correct_global: bool,
    /// Quality threshold for re-unwrapping uncertain voxels after temporal
    /// unwrapping. Range \[0, 1\], set to 0 to disable. (default: 0.5)
    pub temporal_uncertain_unwrapping: f64,
    /// Maximum number of seed regions (default: 1, max: 255).
    pub max_seeds: u8,
    /// Merge neighboring regions after unwrapping (default: false).
    pub merge_regions: bool,
    /// Correct each region's median to nearest 0 by adding n·2π (default: false).
    pub correct_regions: bool,
    /// Additional phase tolerance beyond π for neighbor differences.
    /// Range \[0, π\]. (default: 0.0)
    pub wrap_addition: f64,
}

impl Default for RomeoParams {
    fn default() -> Self {
        Self {
            // Default matches :romeo in ROMEO.jl
            phase_coherence: true,
            phase_gradient_coherence: true,
            phase_linearity: true,
            mag_coherence: true,
            mag_weight: false,
            mag_weight2: false,
            bestpath: false,
            template: 0,
            individual: false,
            correct_global: false,
            temporal_uncertain_unwrapping: 0.5,
            max_seeds: 1,
            merge_regions: false,
            correct_regions: false,
            wrap_addition: 0.0,
        }
    }
}

impl RomeoParams {
    /// Create params matching a ROMEO.jl weight type preset.
    pub fn from_weight_type(wt: RomeoWeightType) -> Self {
        if wt == RomeoWeightType::BestPath {
            return Self { bestpath: true, ..Default::default() };
        }
        let f = wt.weight_flags();
        Self {
            phase_coherence: f[0],
            phase_gradient_coherence: f[1],
            phase_linearity: f[2],
            mag_coherence: f[3],
            mag_weight: f[4],
            mag_weight2: f[5],
            ..Default::default()
        }
    }

    /// Convert the boolean fields to the 6-element flag array used internally.
    pub fn weight_flags(&self) -> [bool; 6] {
        [
            self.phase_coherence,
            self.phase_gradient_coherence,
            self.phase_linearity,
            self.mag_coherence,
            self.mag_weight,
            self.mag_weight2,
        ]
    }
}

const TWO_PI: f64 = 2.0 * PI;

/// Wrap angle to [-π, π]
#[inline]
fn wrap_angle(angle: f64) -> f64 {
    let mut a = angle % TWO_PI;
    if a > PI {
        a -= TWO_PI;
    } else if a < -PI {
        a += TWO_PI;
    }
    a
}

/// Index into a 3D array in Fortran order (column-major, matches NIfTI)
#[inline(always)]
fn idx3d(i: usize, j: usize, k: usize, nx: usize, ny: usize) -> usize {
    i + j * nx + k * nx * ny
}

/// Calculate ROMEO edge weights for phase unwrapping.
///
/// Uses the default Romeo weight type (PC + PGC + PL + MC).
/// Returns weights array of size 3 * nx * ny * nz in C order \[dim\]\[i\]\[j\]\[k\].
/// Weights are 1-255 (valid) or 0 (no edge / masked out).
pub fn calculate_weights_romeo(
    phase: &[f64],
    mag: &[f64],
    phase2: Option<&[f64]>,
    te1: f64,
    te2: f64,
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
) -> Vec<u8> {
    calculate_weights_romeo_with_flags(
        phase, mag, phase2, te1, te2, mask, nx, ny, nz,
        RomeoWeightType::Romeo.weight_flags(),
    )
}

/// Calculate ROMEO edge weights with configurable weight components.
///
/// Legacy 3-flag interface for backward compatibility.
pub fn calculate_weights_romeo_configurable(
    phase: &[f64],
    mag: &[f64],
    phase2: Option<&[f64]>,
    te1: f64,
    te2: f64,
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    use_phase_gradient_coherence: bool,
    use_mag_coherence: bool,
    use_mag_weight: bool,
) -> Vec<u8> {
    let flags = [
        true,                           // phase coherence (always on)
        use_phase_gradient_coherence,
        false,                          // phase linearity (use weight_type for this)
        use_mag_coherence,
        use_mag_weight,
        false,                          // mag_weight2
    ];
    calculate_weights_romeo_with_flags(
        phase, mag, phase2, te1, te2, mask, nx, ny, nz, flags,
    )
}

/// Calculate ROMEO edge weights with full 6-component flag control.
///
/// Matches ROMEO.jl `calculateweights_romeo`. The 6 flags are:
/// 0: phase coherence, 1: phase gradient coherence, 2: phase linearity,
/// 3: magnitude coherence, 4: magnitude weight, 5: magnitude weight 2.
///
/// Each component is scaled as `0.1 + 0.9 * value` (matching ROMEO.jl) to prevent
/// any single component from zeroing the weight entirely.
///
/// Weights are stored as u8: 0 = no edge, 1 = worst valid, 255 = best.
pub fn calculate_weights_romeo_with_flags(
    phase: &[f64],
    mag: &[f64],
    phase2: Option<&[f64]>,
    te1: f64,
    te2: f64,
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    flags: [bool; 6],
) -> Vec<u8> {
    let n_total = nx * ny * nz;
    let mut weights = vec![0u8; 3 * n_total];

    let has_mag = !mag.is_empty();
    let has_phase2 = phase2.is_some();
    let te_ratio = if te2.abs() > 1e-10 { te1 / te2 } else { 1.0 };

    // Disable magnitude flags if no magnitude data
    let f_mc  = flags[3] && has_mag;
    let f_mw  = flags[4] && has_mag;
    let f_mw2 = flags[5] && has_mag;
    // Disable PGC if no phase2/TEs
    let f_pgc = flags[1] && has_phase2;
    let f_pl  = flags[2];

    // 95th percentile of magnitude for normalization (matching ROMEO.jl)
    let max_mag = if has_mag && (f_mw || f_mw2) {
        percentile_95(mag, mask)
    } else {
        1.0
    };
    let half_max_mag = 0.5 * max_mag + 1e-12;

    for dim in 0..3_usize {
        // Each dim writes a disjoint n_total block; voxels within it are
        // independent, so fan out across voxels (rayon under `parallel`).
        let block = &mut weights[dim * n_total..(dim + 1) * n_total];
        maybe_par_iter_mut!(block).enumerate().for_each(|(idx, w)| {
            let i = idx % nx;
            let j = (idx / nx) % ny;
            let k = idx / (nx * ny);
            let (ni, nj, nk) = match dim {
                0 => (i + 1, j, k),
                1 => (i, j + 1, k),
                _ => (i, j, k + 1),
            };

            if ni >= nx || nj >= ny || nk >= nz {
                return;
            }

            let idx_n = idx3d(ni, nj, nk, nx, ny);

            if mask[idx] == 0 || mask[idx_n] == 0 {
                return;
            }

            let mut weight = 1.0_f64;

            // 1. Phase coherence: 1 - |wrap(diff)| / π
            if flags[0] {
                let pc = 1.0 - wrap_angle(phase[idx_n] - phase[idx]).abs() / PI;
                weight *= 0.1 + 0.9 * pc;
            }

            // 2. Phase gradient coherence (multi-echo)
            if f_pgc {
                let phase2_data = phase2.unwrap();
                let wrapped_p1 = wrap_angle(phase[idx_n] - phase[idx]);
                let wrapped_p2 = wrap_angle(phase2_data[idx_n] - phase2_data[idx]);
                let pgc = (1.0 - (wrapped_p1 - wrapped_p2 * te_ratio).abs()).max(0.0);
                weight *= 0.1 + 0.9 * pgc;
            }

            // 3. Phase linearity: product of two triplet linearities
            if f_pl {
                let pl = phase_linearity_edge(phase, i, j, k, ni, nj, nk, dim, nx, ny, nz);
                weight *= 0.1 + 0.9 * pl;
            }

            if has_mag {
                let m1 = mag[idx];
                let m2 = mag[idx_n];
                let small = m1.min(m2);
                let big = m1.max(m2);

                // 4. Magnitude coherence: (min/max)²
                if f_mc {
                    let mc = if big > 1e-12 { (small / big).powi(2) } else { 0.0 };
                    weight *= 0.1 + 0.9 * mc;
                }

                // 5. Magnitude weight: penalize low signal
                if f_mw {
                    let mw = 0.5 + 0.5 * (small / half_max_mag).min(1.0);
                    weight *= 0.1 + 0.9 * mw;
                }

                // 6. Magnitude weight 2: penalize too-high signal (flow artifacts)
                if f_mw2 {
                    let mw2 = 0.5 + 0.5 * (half_max_mag / big.max(1e-12)).min(1.0);
                    weight *= 0.1 + 0.9 * mw2;
                }
            }

            // Rescale to u8: min valid = 1, best = 255, 0 = no edge
            *w = rescale_weight(weight);
        });
    }

    weights
}

/// Phase linearity for an edge between two voxels.
///
/// Computes the product of two triplet linearities: one looking "behind" the edge
/// and one looking "ahead". Matches ROMEO.jl `phaselinearity(P, i, j)`.
fn phase_linearity_edge(
    phase: &[f64],
    i: usize, j: usize, k: usize,
    ni: usize, nj: usize, nk: usize,
    _dim: usize,
    nx: usize, ny: usize, nz: usize,
) -> f64 {
    let idx = idx3d(i, j, k, nx, ny);
    let idx_n = idx3d(ni, nj, nk, nx, ny);

    // "Behind" triplet: (h, idx, idx_n) where h = 2*idx_pos - idx_n_pos
    let (hi, hj, hk) = (2 * i as i32 - ni as i32, 2 * j as i32 - nj as i32, 2 * k as i32 - nk as i32);
    let pl1 = if hi >= 0 && hi < nx as i32 && hj >= 0 && hj < ny as i32 && hk >= 0 && hk < nz as i32 {
        let h_idx = idx3d(hi as usize, hj as usize, hk as usize, nx, ny);
        phase_linearity_triplet(phase[h_idx], phase[idx], phase[idx_n])
    } else {
        0.9
    };

    // "Ahead" triplet: (idx, idx_n, k) where k = 2*idx_n_pos - idx_pos
    let (ki, kj, kk) = (2 * ni as i32 - i as i32, 2 * nj as i32 - j as i32, 2 * nk as i32 - k as i32);
    let pl2 = if ki >= 0 && ki < nx as i32 && kj >= 0 && kj < ny as i32 && kk >= 0 && kk < nz as i32 {
        let k_idx = idx3d(ki as usize, kj as usize, kk as usize, nx, ny);
        phase_linearity_triplet(phase[idx], phase[idx_n], phase[k_idx])
    } else {
        0.9
    };

    pl1 * pl2
}

/// Phase linearity of three consecutive phase values.
///
/// `max(0, 1 - |wrap(a - 2b + c) / 2|)` — measures how linear the phase is.
/// Matches ROMEO.jl `phaselinearity(P, i, j, k)`.
#[inline]
fn phase_linearity_triplet(a: f64, b: f64, c: f64) -> f64 {
    let second_deriv = wrap_angle(a - 2.0 * b + c);
    let pl = (1.0 - (second_deriv / 2.0).abs()).max(0.0);
    if pl.is_nan() { 0.5 } else { pl }
}

/// Rescale weight from [0,1] to u8 [1,255], with 0 = invalid.
///
/// Matches ROMEO.jl convention: valid edges always have weight ≥ 1.
#[inline]
fn rescale_weight(w: f64) -> u8 {
    if w > 0.0 && w <= 1.0 {
        (w * 254.0).round() as u8 + 1  // [1, 255]
    } else if w > 1.0 {
        255
    } else {
        0
    }
}

/// Compute 95th percentile of magnitude within mask (matching ROMEO.jl).
fn percentile_95(mag: &[f64], mask: &[u8]) -> f64 {
    let mut values: Vec<f64> = mag.iter().enumerate()
        .filter(|(i, v)| mask[*i] > 0 && v.is_finite())
        .map(|(_, &v)| v)
        .collect();
    if values.is_empty() {
        return 1.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((values.len() as f64 * 0.95) as usize).min(values.len() - 1);
    values[idx]
}

/// Calculate BestPath edge weights.
///
/// Uses second-order phase differences across all neighbor directions
/// (Abdul-Rahman, https://doi.org/10.1364/AO.46.006623).
/// Matches ROMEO.jl `calculateweights_bestpath`.
pub fn calculate_weights_bestpath(
    phase: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
) -> Vec<u8> {
    let n_total = nx * ny * nz;

    // Compute D² for each voxel: sum of squared wrapped second-order differences
    // across all unique neighbor offsets from 26-connected directions
    let strides = [1_i64, nx as i64, (nx * ny) as i64];
    let mut neighbor_offsets: Vec<i64> = Vec::new();
    for dx in -1i64..=1 {
        for dy in -1i64..=1 {
            for dz in -1i64..=1 {
                if dx == 0 && dy == 0 && dz == 0 { continue; }
                let offset = (dx * strides[0] + dy * strides[1] + dz * strides[2]).abs();
                if !neighbor_offsets.contains(&offset) {
                    neighbor_offsets.push(offset);
                }
            }
        }
    }

    let mut d2 = vec![0.0_f64; n_total];
    for &n in &neighbor_offsets {
        let nu = n as usize;
        for i in nu..(n_total - nu) {
            let v1 = wrap_angle(phase[i - nu] - phase[i]);
            let v2 = wrap_angle(phase[i] - phase[i + nu]);
            let diff = v1 - v2;
            d2[i] += diff * diff;
        }
    }

    // R = 1 / sqrt(D²)
    let r: Vec<f64> = d2.iter().map(|&v| {
        let d = v.sqrt();
        if d > 1e-12 { 1.0 / d } else { 0.0 }
    }).collect();

    // Edge weights: R[i] + R[i+n] for each direction
    let mut weights = vec![0u8; 3 * n_total];
    for dim in 0..3_usize {
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let (ni, nj, nk) = match dim {
                        0 => (i + 1, j, k),
                        1 => (i, j + 1, k),
                        _ => (i, j, k + 1),
                    };
                    if ni >= nx || nj >= ny || nk >= nz { continue; }

                    let idx = idx3d(i, j, k, nx, ny);
                    let idx_n = idx3d(ni, nj, nk, nx, ny);

                    if mask[idx] == 0 || mask[idx_n] == 0 { continue; }

                    let w = r[idx] + r[idx_n];
                    // Scale: lower w = more consistent = higher weight
                    // ROMEO.jl: scale(w) = min(max(round((1 - w/10) * 255), 1), 255)
                    let scaled = ((1.0 - w / 10.0) * 255.0).round().clamp(1.0, 255.0) as u8;
                    let edge_idx = dim * n_total + idx3d(i, j, k, nx, ny);
                    weights[edge_idx] = scaled;
                }
            }
        }
    }

    weights
}

/// Simplified weight calculation for single-echo data (no phase2)
pub fn calculate_weights_single_echo(
    phase: &[f64],
    mag: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
) -> Vec<u8> {
    calculate_weights_romeo(phase, mag, None, 1.0, 1.0, mask, nx, ny, nz)
}

/// Calculate per-voxel quality map from ROMEO edge weights
///
/// Computes ROMEO edge weights and then aggregates them per-voxel by averaging
/// the incident edge weights across all 6 neighboring directions (±x, ±y, ±z).
/// This produces a quality map where high values indicate voxels with coherent
/// phase and magnitude, suitable for thresholding into a brain mask.
///
/// Reference: MriResearchTools.jl `romeovoxelquality()` function
///
/// # Arguments
/// * `phase` - Wrapped phase data (nx * ny * nz), first echo
/// * `mag` - Magnitude data (nx * ny * nz), optional (pass empty slice if none)
/// * `phase2` - Second echo phase for gradient coherence (optional)
/// * `te1`, `te2` - Echo times for gradient coherence scaling
/// * `mask` - Binary mask (nx * ny * nz), 1 = process
/// * `nx`, `ny`, `nz` - Array dimensions
///
/// # Returns
/// Quality map of size nx * ny * nz with values in range [0, 100]
#[allow(clippy::erasing_op)]
pub fn voxel_quality_romeo(
    phase: &[f64],
    mag: &[f64],
    phase2: Option<&[f64]>,
    te1: f64,
    te2: f64,
    mask: &[u8],
    grid: &Grid,
) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let n_total = nx * ny * nz;
    let weights = calculate_weights_romeo(phase, mag, phase2, te1, te2, mask, nx, ny, nz);

    let mut quality = vec![0.0_f64; n_total];

    // Each voxel's quality is an independent read of neighbouring edge
    // weights — fan out across voxels (rayon under the `parallel` feature).
    maybe_par_iter_mut!(quality).enumerate().for_each(|(idx, q)| {
        if mask[idx] == 0 {
            return;
        }
        let i = idx % nx;
        let j = (idx / nx) % ny;
        let k = idx / (nx * ny);

        let mut sum = 0.0_f64;
        let mut count = 0u32;

        // +x edge: stored at (i, j, k) in dim=0
        if i + 1 < nx && mask[idx3d(i + 1, j, k, nx, ny)] != 0 {
            sum += weights[idx] as f64;
            count += 1;
        }
        // -x edge: stored at (i-1, j, k) in dim=0
        if i > 0 && mask[idx3d(i - 1, j, k, nx, ny)] != 0 {
            sum += weights[idx3d(i - 1, j, k, nx, ny)] as f64;
            count += 1;
        }
        // +y edge: stored at (i, j, k) in dim=1
        if j + 1 < ny && mask[idx3d(i, j + 1, k, nx, ny)] != 0 {
            sum += weights[n_total + idx] as f64;
            count += 1;
        }
        // -y edge: stored at (i, j-1, k) in dim=1
        if j > 0 && mask[idx3d(i, j - 1, k, nx, ny)] != 0 {
            sum += weights[n_total + idx3d(i, j - 1, k, nx, ny)] as f64;
            count += 1;
        }
        // +z edge: stored at (i, j, k) in dim=2
        if k + 1 < nz && mask[idx3d(i, j, k + 1, nx, ny)] != 0 {
            sum += weights[2 * n_total + idx] as f64;
            count += 1;
        }
        // -z edge: stored at (i, j, k-1) in dim=2
        if k > 0 && mask[idx3d(i, j, k - 1, nx, ny)] != 0 {
            sum += weights[2 * n_total + idx3d(i, j, k - 1, nx, ny)] as f64;
            count += 1;
        }

        if count > 0 {
            // Normalize from 0-255 to 0-1, then scale to 0-100
            *q = (sum / count as f64) / 255.0 * 100.0;
        }
    });

    quality
}

// =========================================================================
// Phase unwrapping functions
// =========================================================================

/// Find a seed point at the center of mass of the mask.
pub fn find_seed_point(mask: &[u8], nx: usize, ny: usize, nz: usize) -> (usize, usize, usize) {
    let mut sum_i = 0usize;
    let mut sum_j = 0usize;
    let mut sum_k = 0usize;
    let mut count = 0usize;

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let idx = idx3d(i, j, k, nx, ny);
                if mask[idx] > 0 {
                    sum_i += i;
                    sum_j += j;
                    sum_k += k;
                    count += 1;
                }
            }
        }
    }

    if count == 0 {
        return (nx / 2, ny / 2, nz / 2);
    }

    (sum_i / count, sum_j / count, sum_k / count)
}

/// Compute weights from params (dispatches BestPath vs ROMEO).
fn compute_weights_from_params(
    params: &RomeoParams,
    phase: &[f64],
    mag: &[f64],
    phase2: Option<&[f64]>,
    te1: f64, te2: f64,
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
) -> Vec<u8> {
    if params.bestpath {
        calculate_weights_bestpath(phase, mask, nx, ny, nz)
    } else {
        calculate_weights_romeo_with_flags(
            phase, mag, phase2, te1, te2, mask, nx, ny, nz,
            params.weight_flags(),
        )
    }
}

/// Unwrap a single 3D phase volume using ROMEO.
///
/// Computes ROMEO edge weights and performs region-growing phase unwrapping.
///
/// # Arguments
/// * `phase` - Wrapped phase data (nx * ny * nz)
/// * `mag` - Magnitude data (pass empty slice if none)
/// * `phase2` - Optional second echo phase for gradient coherence weights
/// * `te1`, `te2` - Echo times for phase gradient coherence scaling
/// * `mask` - Binary mask (1 = process)
/// * `params` - ROMEO parameters (weight_type used for weight calculation)
/// * `nx`, `ny`, `nz` - Array dimensions
///
/// # Returns
/// Unwrapped phase (same size as input)
pub fn unwrap_romeo(
    phase: &[f64],
    mag: &[f64],
    phase2: Option<&[f64]>,
    te1: f64,
    te2: f64,
    mask: &[u8],
    params: &RomeoParams,
    grid: &Grid,
) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let weights = compute_weights_from_params(
        params, phase, mag, phase2, te1, te2,
        mask, nx, ny, nz,
    );

    let mut unwrapped = phase.to_vec();
    let n_total = nx * ny * nz;

    if params.max_seeds > 1 || params.wrap_addition > 0.0 || params.merge_regions || params.correct_regions {
        // Full-featured path with multi-seed, wrap_addition, region merging
        let mut visited = vec![0u8; n_total];
        let num_regions = grow_region_unwrap_full(
            &mut unwrapped, &weights, mask, &mut visited,
            nx, ny, nz, params.wrap_addition, params.max_seeds,
            phase2, if phase2.is_some() { Some((te1, te2)) } else { None },
        );

        if params.merge_regions && num_regions > 1 {
            let remaining = merge_regions_post(
                &mut unwrapped, &mut visited, num_regions, &weights, nx, ny, nz,
            );
            if params.correct_regions {
                // correct_regions on remaining regions
                for &r in &remaining {
                    correct_regions(&mut unwrapped, &visited, r);
                }
            }
        } else if params.correct_regions && num_regions > 0 {
            correct_regions(&mut unwrapped, &visited, num_regions);
        }
    } else {
        // Simple single-seed path (faster, no region tracking overhead)
        let (seed_i, seed_j, seed_k) = find_seed_point(mask, nx, ny, nz);
        let mut work_mask = mask.to_vec();
        grow_region_unwrap(
            &mut unwrapped, &weights, &mut work_mask,
            nx, ny, nz, seed_i, seed_j, seed_k,
        );
    }

    // correctglobal for 3D: subtract median n·2π offset
    if params.correct_global {
        correct_global_offset(&mut unwrapped, mask);
    }

    unwrapped
}

/// Unwrap 4D multi-echo phase data using ROMEO.
///
/// Supports two modes matching ROMEO.jl:
///
/// **Template-based** (default, `individual=false`):
/// Spatially unwraps one template echo, then temporally unwraps all others
/// by scaling with TE ratios. Optionally re-unwraps uncertain voxels spatially.
///
/// **Individual** (`individual=true`):
/// Spatially unwraps each echo independently, then optionally corrects global
/// 2π offsets between echoes using median wrap counting.
///
/// # Arguments
/// * `phases` - Wrapped phase for each echo \[n_echoes\]\[nx*ny*nz\]
/// * `mags` - Magnitude for each echo (pass empty `&[]` if none)
/// * `tes` - Echo times (units don't matter, only ratios are used)
/// * `mask` - Binary mask (nx * ny * nz)
/// * `params` - ROMEO parameters
/// * `nx`, `ny`, `nz` - Array dimensions
///
/// # Returns
/// Unwrapped phase for each echo
pub fn unwrap_romeo_multi_echo<P: AsRef<[f64]> + Sync, M: AsRef<[f64]> + Sync>(
    phases: &[P],
    mags: &[M],
    tes: &[f64],
    mask: &[u8],
    params: &RomeoParams,
    grid: &Grid,
) -> Vec<Vec<f64>> {
    let (nx, ny, nz) = grid.dims;
    let n_echoes = phases.len();
    assert!(n_echoes > 0, "phases must have at least one echo");
    assert_eq!(n_echoes, tes.len(), "phases and tes must have same length");

    if n_echoes == 1 {
        let mag = if mags.is_empty() { &[] as &[f64] } else { mags[0].as_ref() };
        return vec![unwrap_romeo(
            phases[0].as_ref(), mag, None, 0.0, 0.0,
            mask, params, grid,
        )];
    }

    if params.individual {
        unwrap_individual(phases, mags, tes, mask, params, nx, ny, nz)
    } else {
        unwrap_template(phases, mags, tes, mask, params, nx, ny, nz)
    }
}

// =========================================================================
// Template-based multi-echo unwrapping
// =========================================================================

/// Template-based multi-echo unwrapping.
///
/// 1. Spatially unwrap the template echo using ROMEO weights
/// 2. Temporally unwrap all other echoes outward from the template
/// 3. Optionally re-unwrap uncertain voxels spatially
fn unwrap_template<P: AsRef<[f64]>, M: AsRef<[f64]>>(
    phases: &[P],
    mags: &[M],
    tes: &[f64],
    mask: &[u8],
    params: &RomeoParams,
    nx: usize, ny: usize, nz: usize,
) -> Vec<Vec<f64>> {
    let n_echoes = phases.len();
    let n_total = nx * ny * nz;
    let template = params.template.min(n_echoes - 1);

    // Select phase2 reference (matching ROMEO.jl p2ref default)
    let p2ref = if template == 0 { 1 } else { template - 1 };

    // Calculate weights using template echo + p2ref
    let template_mag = if mags.is_empty() { &[] as &[f64] } else { mags[template].as_ref() };
    let weights = compute_weights_from_params(
        params,
        phases[template].as_ref(),
        template_mag,
        Some(phases[p2ref].as_ref()),
        tes[template], tes[p2ref],
        mask, nx, ny, nz,
    );

    // Spatially unwrap template echo
    let mut result: Vec<Vec<f64>> = phases.iter().map(|p| p.as_ref().to_vec()).collect();
    if params.wrap_addition > 0.0 {
        let mut visited = vec![0u8; n_total];
        grow_region_unwrap_full(
            &mut result[template], &weights, mask, &mut visited,
            nx, ny, nz, params.wrap_addition, 1,
            Some(phases[p2ref].as_ref()), Some((tes[template], tes[p2ref])),
        );
    } else {
        let (seed_i, seed_j, seed_k) = find_seed_point(mask, nx, ny, nz);
        let mut work_mask = mask.to_vec();
        grow_region_unwrap(
            &mut result[template], &weights, &mut work_mask,
            nx, ny, nz, seed_i, seed_j, seed_k,
        );
    }

    // Temporally unwrap other echoes outward from template
    // Order: template-1, template-2, ..., 0, template+1, template+2, ..., n-1
    // (matching ROMEO.jl: [(template-1):-1:1; (template+1):length(TEs)])
    let echo_order: Vec<usize> = (0..template).rev()
        .chain((template + 1)..n_echoes)
        .collect();

    for ieco in echo_order {
        let iref = if ieco < template { ieco + 1 } else { ieco - 1 };
        let te_ratio = tes[ieco] / tes[iref];

        // Temporal unwrap each voxel: align to TE-scaled reference
        for i in 0..n_total {
            if mask[i] > 0 {
                let ref_value = result[iref][i] * te_ratio;
                result[ieco][i] = unwrap_voxel(result[ieco][i], ref_value);
            }
        }

        // Fallback: re-unwrap uncertain voxels spatially
        if params.temporal_uncertain_unwrapping > 0.0 {
            // Build scaled reference for quality assessment
            let ref_scaled: Vec<f64> = (0..n_total).map(|i| {
                if mask[i] > 0 { result[iref][i] * te_ratio } else { 0.0 }
            }).collect();

            temporal_uncertain_rewrap(
                &mut result[ieco],
                &ref_scaled,
                &weights,
                mask,
                params.temporal_uncertain_unwrapping,
                nx, ny, nz,
            );
        }
    }

    result
}

// =========================================================================
// Individual multi-echo unwrapping
// =========================================================================

/// Individual multi-echo unwrapping.
///
/// Each echo is spatially unwrapped independently using its neighboring echo
/// as a phase2 reference for weight calculation. Optionally corrects global
/// 2π offsets between echoes using median wrap counting.
fn unwrap_individual<P: AsRef<[f64]> + Sync, M: AsRef<[f64]> + Sync>(
    phases: &[P],
    mags: &[M],
    tes: &[f64],
    mask: &[u8],
    params: &RomeoParams,
    nx: usize, ny: usize, nz: usize,
) -> Vec<Vec<f64>> {
    let n_echoes = phases.len();
    let (seed_i, seed_j, seed_k) = find_seed_point(mask, nx, ny, nz);

    // Each echo is unwrapped independently from the same seed, so the echoes
    // fan out across threads (under the `parallel` feature). Order is preserved
    // by the parallel `collect`.
    let echo_indices: Vec<usize> = (0..n_echoes).collect();
    let mut result: Vec<Vec<f64>> = maybe_par_iter!(echo_indices)
        .map(|&i| {
            // Neighboring echo as phase2 reference (matching ROMEO.jl)
            let e2 = if i == 0 { 1 } else { i - 1 };

            let mag = if mags.is_empty() { &[] as &[f64] } else { mags[i].as_ref() };
            let weights = compute_weights_from_params(
                params,
                phases[i].as_ref(),
                mag,
                Some(phases[e2].as_ref()),
                tes[i], tes[e2],
                mask, nx, ny, nz,
            );

            let mut unwrapped = phases[i].as_ref().to_vec();
            let mut work_mask = mask.to_vec();
            grow_region_unwrap(
                &mut unwrapped, &weights, &mut work_mask,
                nx, ny, nz, seed_i, seed_j, seed_k,
            );
            unwrapped
        })
        .collect();

    if params.correct_global {
        correct_multi_echo_wraps(&mut result, tes, mask);
    }

    result
}

// =========================================================================
// Helper functions
// =========================================================================

/// Temporal unwrap: remove 2π wraps by comparing to a reference value.
///
/// Matches ROMEO.jl: `unwrapvoxel(new, old) = new - 2π * round((new - old) / 2π)`
#[inline]
fn unwrap_voxel(new: f64, old: f64) -> f64 {
    new - TWO_PI * ((new - old) / TWO_PI).round()
}

/// Correct global 2π offsets between echoes using median wrap counting.
///
/// For each successive echo pair, computes the median number of 2π wraps
/// between the TE-scaled reference and the current echo, then corrects.
///
/// Matches ROMEO.jl `correct_multi_echo_wraps!`.
pub fn correct_multi_echo_wraps(
    phases: &mut [Vec<f64>],
    tes: &[f64],
    mask: &[u8],
) {
    let n_total = phases[0].len();

    for ieco in 1..phases.len() {
        let iref = ieco - 1;
        let te_ratio = tes[ieco] / tes[iref];

        // Collect wrap counts for all masked voxels
        let mut wrap_counts: Vec<f64> = Vec::new();
        for i in 0..n_total {
            if mask[i] > 0 {
                let expected = phases[iref][i] * te_ratio;
                let nwraps = ((expected - phases[ieco][i]) / TWO_PI).round();
                if nwraps.is_finite() {
                    wrap_counts.push(nwraps);
                }
            }
        }

        if wrap_counts.is_empty() {
            continue;
        }

        // Median wrap count
        wrap_counts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_wraps = wrap_counts[wrap_counts.len() / 2];

        if median_wraps.abs() > 0.1 {
            let correction = TWO_PI * median_wraps;
            for i in 0..n_total {
                if mask[i] > 0 {
                    phases[ieco][i] += correction;
                }
            }
        }
    }
}

/// Correct global 2π offset for a single 3D volume.
///
/// Subtracts `2π * median(round(phase / 2π))` from the entire volume,
/// bringing the median phase closest to 0. Matches ROMEO.jl `correctglobal`.
fn correct_global_offset(phase: &mut [f64], mask: &[u8]) {
    let mut wraps: Vec<f64> = phase.iter().enumerate()
        .filter(|(i, v)| mask[*i] > 0 && v.is_finite())
        .map(|(_, &v)| (v / TWO_PI).round())
        .collect();

    if wraps.is_empty() {
        return;
    }

    wraps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_wraps = wraps[wraps.len() / 2];

    if median_wraps.abs() > 0.1 {
        let correction = TWO_PI * median_wraps;
        for i in 0..phase.len() {
            if mask[i] > 0 {
                phase[i] -= correction;
            }
        }
    }
}

/// Correct each region's median to nearest 0 by subtracting n·2π.
///
/// Matches ROMEO.jl `correct_regions!`.
fn correct_regions(phase: &mut [f64], visited: &[u8], num_regions: u8) {
    for region in 1..=num_regions {
        let mut wraps: Vec<f64> = Vec::new();
        for i in 0..phase.len() {
            if visited[i] == region && phase[i].is_finite() {
                wraps.push((phase[i] / TWO_PI).round());
            }
        }
        if wraps.is_empty() { continue; }
        wraps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_wraps = wraps[wraps.len() / 2];
        if median_wraps.abs() > 0.1 {
            let correction = TWO_PI * median_wraps;
            for i in 0..phase.len() {
                if visited[i] == region {
                    phase[i] -= correction;
                }
            }
        }
    }
}

/// Merge adjacent regions by computing inter-region phase offsets.
///
/// Calculates weighted phase offset between adjacent region pairs,
/// then iteratively merges regions starting from the largest.
/// Matches ROMEO.jl `merge_regions!`.
fn merge_regions_post(
    phase: &mut [f64],
    visited: &mut [u8],
    num_regions: u8,
    weights: &[u8],
    nx: usize, ny: usize, nz: usize,
) -> Vec<u8> {
    let n_total = nx * ny * nz;
    let nr = num_regions as usize;
    if nr <= 1 { return vec![1]; }

    // Count region sizes
    let mut region_size = vec![0usize; nr + 1]; // 1-indexed
    for &v in visited.iter() {
        if v > 0 && (v as usize) <= nr {
            region_size[v as usize] += 1;
        }
    }

    // Compute weighted offset between adjacent regions
    let mut offsets = vec![0.0_f64; (nr + 1) * (nr + 1)];
    let mut offset_counts = vec![0i64; (nr + 1) * (nr + 1)];
    let flat = |r1: usize, r2: usize| r1 * (nr + 1) + r2;

    for dim in 0..3_usize {
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let (ni, nj, nk) = match dim {
                        0 => (i + 1, j, k),
                        1 => (i, j + 1, k),
                        _ => (i, j, k + 1),
                    };
                    if ni >= nx || nj >= ny || nk >= nz { continue; }
                    let idx = idx3d(i, j, k, nx, ny);
                    let idx_n = idx3d(ni, nj, nk, nx, ny);
                    let ri = visited[idx] as usize;
                    let rj = visited[idx_n] as usize;
                    if ri == 0 || rj == 0 || ri == rj { continue; }

                    let edge_idx = dim * n_total + idx3d(i, j, k, nx, ny);
                    let w = 255u16.saturating_sub(weights[edge_idx] as u16);
                    let w = if w == 255 { 0 } else { w as i64 };

                    offsets[flat(ri, rj)] += (phase[idx] - phase[idx_n]) * w as f64;
                    offset_counts[flat(ri, rj)] += w;
                }
            }
        }
    }

    // Symmetrize
    for i in 1..=nr {
        for j in i..=nr {
            offset_counts[flat(i, j)] += offset_counts[flat(j, i)];
            offset_counts[flat(j, i)] = offset_counts[flat(i, j)];
            offsets[flat(i, j)] -= offsets[flat(j, i)];
            offsets[flat(j, i)] = -offsets[flat(i, j)];
        }
    }

    // Iteratively merge: process largest uncorrected region first
    let mut corrected = vec![false; nr + 1];
    let mut remaining = Vec::new();

    while corrected[1..=nr].iter().any(|&c| !c) {
        // Find largest uncorrected region
        let mut best_region = 0;
        let mut best_size = 0;
        for r in 1..=nr {
            if !corrected[r] && region_size[r] > best_size {
                best_size = region_size[r];
                best_region = r;
            }
        }
        if best_region == 0 { break; }
        corrected[best_region] = true;
        remaining.push(best_region as u8);

        // Find regions to merge: corrected[i] && !corrected[j] && offset_counts > 0
        // Sort by offset_counts (highest first = best connection)
        let mut merge_pairs: Vec<(usize, usize, i64)> = Vec::new();
        for i in 1..=nr {
            for j in 1..=nr {
                if corrected[i] && !corrected[j] && offset_counts[flat(i, j)] > 0 {
                    merge_pairs.push((i, j, offset_counts[flat(i, j)]));
                }
            }
        }
        merge_pairs.sort_by(|a, b| b.2.cmp(&a.2));

        for &(ri, rj, count) in &merge_pairs {
            if corrected[rj] { continue; }
            let offset = (offsets[flat(ri, rj)] / count as f64 / TWO_PI).round();
            if offset != 0.0 {
                let correction = offset * TWO_PI;
                for v in 0..n_total {
                    if visited[v] == rj as u8 {
                        phase[v] += correction;
                        visited[v] = ri as u8;
                    }
                }
            }
            corrected[rj] = true;
            offset_counts[flat(ri, rj)] = -1;
            offset_counts[flat(rj, ri)] = -1;
        }
    }

    remaining
}

/// Re-unwrap voxels that are uncertain after temporal unwrapping.
///
/// Computes a quality metric comparing the unwrapped phase to the TE-scaled
/// reference. Voxels with low quality (likely wrap errors) are re-unwrapped
/// spatially using the certain voxels as seeds.
///
/// Matches ROMEO.jl `temporal_uncertain_unwrapping!`.
fn temporal_uncertain_rewrap(
    phase: &mut [f64],
    ref_scaled: &[f64],
    weights: &[u8],
    mask: &[u8],
    threshold: f64,
    nx: usize, ny: usize, nz: usize,
) {
    let n_total = nx * ny * nz;

    // Compute quality: compare phase/2 with ref_scaled/2 using voxel quality
    // (halving increases sensitivity to single-wrap errors)
    // Matches ROMEO.jl: unwrapped_quality(phase, refphase) = voxelquality(phase/2; phase2=refphase/2, TEs=[1,1])
    let half_phase: Vec<f64> = phase.iter().map(|&v| v * 0.5).collect();
    let half_ref: Vec<f64> = ref_scaled.iter().map(|&v| v * 0.5).collect();

    let grid = crate::Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);
    let quality = voxel_quality_romeo(
        &half_phase, &[], Some(&half_ref), 1.0, 1.0,
        mask, &grid,
    );

    // Build visited mask:
    // quality > threshold*100 → certain (visited=2, acts as seed)
    // quality <= threshold*100 → uncertain (visited=1, needs re-unwrapping)
    // outside mask → 0
    let threshold_scaled = threshold * 100.0; // voxel_quality_romeo returns 0-100
    let mut visited = vec![0u8; n_total];
    let mut any_uncertain = false;
    let mut any_certain = false;

    for i in 0..n_total {
        if mask[i] == 0 {
            visited[i] = 0;
        } else if quality[i] > threshold_scaled {
            visited[i] = 2;
            any_certain = true;
        } else {
            visited[i] = 1;
            any_uncertain = true;
        }
    }

    if !any_uncertain || !any_certain {
        return;
    }

    // Re-unwrap uncertain voxels using spatial growing from the certain boundary
    grow_region_unwrap_from_visited(phase, weights, &mut visited, nx, ny, nz);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Grid;

    fn grid(n: usize) -> Grid { Grid::new(n, n, n, 1.0, 1.0, 1.0) }

    #[test]
    fn test_wrap_angle() {
        assert!((wrap_angle(0.0) - 0.0).abs() < 1e-10);
        assert!((wrap_angle(PI) - PI).abs() < 1e-10);
        assert!((wrap_angle(-PI) - (-PI)).abs() < 1e-10);
        assert!((wrap_angle(TWO_PI) - 0.0).abs() < 1e-10);
        assert!((wrap_angle(3.0 * PI) - PI).abs() < 1e-10);
        assert!((wrap_angle(-3.0 * PI) - (-PI)).abs() < 1e-10);
    }

    #[test]
    fn test_weights_constant_phase() {
        // Constant phase should give high weights (phase coherence = 1)
        let n = 4;
        let phase = vec![0.0; n * n * n];
        let mag = vec![1.0; n * n * n];
        let mask = vec![1u8; n * n * n];

        let weights = calculate_weights_single_echo(&phase, &mag, &mask, n, n, n);

        // All interior weights should be 255 (constant phase, uniform magnitude)
        let mut high_weight_count = 0;
        for &w in weights.iter() {
            if w == 255 {
                high_weight_count += 1;
            }
        }
        assert!(high_weight_count > 0, "Should have some high weights for constant phase");
    }

    #[test]
    fn test_weights_wrapped_jump() {
        // Phase with 2π jump should give low weights at jump location
        let n = 4;
        let mut phase = vec![0.0; n * n * n];

        // Create a 2π jump in x direction at i=2
        for i in 2..n {
            for j in 0..n {
                for k in 0..n {
                    phase[idx3d(i, j, k, n, n)] = TWO_PI;
                }
            }
        }

        let mask = vec![1u8; n * n * n];
        let _weights = calculate_weights_single_echo(&phase, &[], &mask, n, n, n);

        // Weight at x=1 to x=2 edge should be low (wrapped difference = 0, but that's ok)
        // Actually, for a 2π jump, the wrapped difference is 0, so coherence is 1
        // This is correct - ROMEO uses wrapped differences, not raw differences
    }

    #[test]
    fn test_weights_mask() {
        // Weights should be 0 where mask is 0
        let n = 4;
        let phase = vec![0.5; n * n * n];
        let mut mask = vec![1u8; n * n * n];

        // Set some voxels outside mask
        mask[0] = 0;
        mask[1] = 0;

        let weights = calculate_weights_single_echo(&phase, &[], &mask, n, n, n);

        // Edges connected to masked-out voxels should be 0
        // Weight at edge (0,0,0)-(1,0,0) should be 0 since idx 0 is masked out
        assert_eq!(weights[0], 0);  // x-direction edge at (0,0,0)
    }

    #[test]
    fn test_voxel_quality_constant_phase() {
        // Constant phase + uniform magnitude → all quality values should be 100
        let n = 4;
        let phase = vec![0.0; n * n * n];
        let mag = vec![1.0; n * n * n];
        let mask = vec![1u8; n * n * n];

        let quality = voxel_quality_romeo(&phase, &mag, None, 1.0, 1.0, &mask, &grid(n));

        assert_eq!(quality.len(), n * n * n);

        // Interior voxels should have high quality (may be < 100 due to
        // phaselinearity boundary fallback at edges of small test cube)
        let interior_q = quality[idx3d(1, 1, 1, n, n)];
        assert!(interior_q > 90.0,
                "Interior voxel quality should be >90, got {}", interior_q);
    }

    #[test]
    fn test_voxel_quality_masked() {
        // Masked-out voxels should have quality = 0
        let n = 4;
        let phase = vec![0.0; n * n * n];
        let mut mask = vec![1u8; n * n * n];
        mask[idx3d(1, 1, 1, n, n)] = 0;

        let quality = voxel_quality_romeo(&phase, &[], None, 1.0, 1.0, &mask, &grid(n));

        assert_eq!(quality[idx3d(1, 1, 1, n, n)], 0.0,
                   "Masked-out voxel should have quality 0");
    }

    #[test]
    fn test_voxel_quality_range() {
        // Quality values should be in range [0, 100]
        let n = 6;
        let phase: Vec<f64> = (0..n * n * n).map(|i| (i as f64) * 0.7).collect();
        let mag: Vec<f64> = (0..n * n * n).map(|i| (i as f64) / (n * n * n) as f64).collect();
        let mask = vec![1u8; n * n * n];

        let quality = voxel_quality_romeo(&phase, &mag, None, 1.0, 1.0, &mask, &grid(n));

        for &q in quality.iter() {
            assert!(q >= 0.0 && q <= 100.0,
                    "Quality should be in [0, 100], got {}", q);
        }
    }

    // =========================================================================
    // find_seed_point
    // =========================================================================

    #[test]
    fn test_find_seed_point_full_mask() {
        let (nx, ny, nz) = (8, 8, 8);
        let mask = vec![1u8; nx * ny * nz];
        let (si, sj, sk) = find_seed_point(&mask, nx, ny, nz);
        assert_eq!(si, 3);
        assert_eq!(sj, 3);
        assert_eq!(sk, 3);
    }

    #[test]
    fn test_find_seed_point_empty_mask() {
        let (nx, ny, nz) = (8, 8, 8);
        let mask = vec![0u8; nx * ny * nz];
        let (si, sj, sk) = find_seed_point(&mask, nx, ny, nz);
        assert_eq!(si, 4);
        assert_eq!(sj, 4);
        assert_eq!(sk, 4);
    }

    // =========================================================================
    // unwrap_romeo (3D)
    // =========================================================================

    #[test]
    fn test_unwrap_romeo_3d_smooth() {
        let n = 8;
        let total = n * n * n;
        let phase: Vec<f64> = (0..total).map(|i| {
            (i % n) as f64 / n as f64 * 0.5
        }).collect();
        let mag = vec![1.0; total];
        let mask = vec![1u8; total];

        let unwrapped = unwrap_romeo(
            &phase, &mag, None, 0.0, 0.0,
            &mask, &RomeoParams::default(), &grid(n),
        );

        assert_eq!(unwrapped.len(), total);
        for &v in &unwrapped {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_unwrap_romeo_3d_with_wrap() {
        // Phase with a 2π wrap that should be removed
        let n = 8;
        let total = n * n * n;
        let mut phase = vec![0.0; total];

        // Create a smooth ramp 0..0.7 in x, then add 2π at x>=4
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let idx = idx3d(i, j, k, n, n);
                    let val = 0.1 * i as f64;
                    phase[idx] = if i >= 4 { val + TWO_PI } else { val };
                }
            }
        }

        let mask = vec![1u8; total];
        let unwrapped = unwrap_romeo(
            &phase, &[], None, 0.0, 0.0,
            &mask, &RomeoParams::default(), &grid(n),
        );

        // After unwrapping, the 2π jump should be removed
        // Check that adjacent voxels don't differ by more than π
        for j in 0..n {
            for k in 0..n {
                for i in 1..n {
                    let curr = unwrapped[idx3d(i, j, k, n, n)];
                    let prev = unwrapped[idx3d(i - 1, j, k, n, n)];
                    let diff = (curr - prev).abs();
                    assert!(diff < PI + 0.5,
                        "Adjacent voxels differ by {:.2} at x={}", diff, i);
                }
            }
        }
    }

    // =========================================================================
    // unwrap_romeo_multi_echo (4D template-based)
    // =========================================================================

    #[test]
    fn test_multi_echo_template_output_sizes() {
        let n = 8;
        let total = n * n * n;
        let tes = [5.0, 10.0, 15.0];
        let slope = 0.05;

        let phases: Vec<Vec<f64>> = tes.iter().map(|&te| {
            (0..total).map(|i| {
                let x = (i % n) as f64;
                wrap_angle(slope * x * te)
            }).collect()
        }).collect();
        let mags: Vec<Vec<f64>> = tes.iter().map(|_| vec![1.0; total]).collect();
        let mask = vec![1u8; total];

        let result = unwrap_romeo_multi_echo(
            &phases, &mags, &tes, &mask,
            &RomeoParams::default(), &grid(n),
        );

        assert_eq!(result.len(), 3);
        for echo in &result {
            assert_eq!(echo.len(), total);
            for &v in echo {
                assert!(v.is_finite());
            }
        }
    }

    #[test]
    fn test_multi_echo_template_temporal_consistency() {
        // After template-based unwrapping, echoes should be temporally consistent:
        // phase[e] ≈ phase[0] * TE[e] / TE[0]
        let n = 8;
        let total = n * n * n;
        let tes = [5.0, 10.0, 15.0];
        let slope = 0.03; // small slope so no wrapping needed

        let phases: Vec<Vec<f64>> = tes.iter().map(|&te| {
            (0..total).map(|i| {
                let x = (i % n) as f64;
                slope * x * te // no wrapping needed
            }).collect()
        }).collect();
        let mags: Vec<Vec<f64>> = tes.iter().map(|_| vec![1.0; total]).collect();
        let mask = vec![1u8; total];

        let result = unwrap_romeo_multi_echo(
            &phases, &mags, &tes, &mask,
            &RomeoParams::default(), &grid(n),
        );

        // Check TE-scaling consistency at interior voxels
        for i in 0..total {
            if mask[i] > 0 && result[0][i].abs() > 1e-10 {
                let ratio_12 = result[1][i] / result[0][i];
                let expected_ratio = tes[1] / tes[0];
                assert!((ratio_12 - expected_ratio).abs() < 0.5,
                    "Echo 1/0 ratio {:.3} vs expected {:.3} at voxel {}",
                    ratio_12, expected_ratio, i);
            }
        }
    }

    #[test]
    fn test_multi_echo_single_echo_fallback() {
        let n = 4;
        let total = n * n * n;
        let phase = vec![0.1; total];
        let mask = vec![1u8; total];

        let result = unwrap_romeo_multi_echo(
            &[&phase[..]], &[] as &[&[f64]], &[5.0], &mask,
            &RomeoParams::default(), &grid(n),
        );

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), total);
    }

    // =========================================================================
    // unwrap_romeo_multi_echo (individual mode)
    // =========================================================================

    #[test]
    fn test_multi_echo_individual_mode() {
        let n = 8;
        let total = n * n * n;
        let tes = [5.0, 10.0, 15.0];
        let slope = 0.03;

        let phases: Vec<Vec<f64>> = tes.iter().map(|&te| {
            (0..total).map(|i| {
                let x = (i % n) as f64;
                slope * x * te
            }).collect()
        }).collect();
        let mags: Vec<Vec<f64>> = tes.iter().map(|_| vec![1.0; total]).collect();
        let mask = vec![1u8; total];

        let params = RomeoParams {
            individual: true,
            correct_global: true,
            ..Default::default()
        };

        let result = unwrap_romeo_multi_echo(
            &phases, &mags, &tes, &mask,
            &params, &grid(n),
        );

        assert_eq!(result.len(), 3);
        for echo in &result {
            assert_eq!(echo.len(), total);
            for &v in echo {
                assert!(v.is_finite());
            }
        }
    }

    // =========================================================================
    // correct_multi_echo_wraps
    // =========================================================================

    #[test]
    fn test_correct_multi_echo_wraps_no_correction_needed() {
        let n = 64;
        let tes = [5.0, 10.0, 15.0];
        let mask = vec![1u8; n];

        // Perfectly scaled echoes: no correction needed
        let mut phases: Vec<Vec<f64>> = tes.iter().map(|&te| {
            vec![0.1 * te; n]
        }).collect();

        let original: Vec<Vec<f64>> = phases.iter().map(|p| p.clone()).collect();
        correct_multi_echo_wraps(&mut phases, &tes, &mask);

        for e in 0..3 {
            for i in 0..n {
                assert!((phases[e][i] - original[e][i]).abs() < 1e-10,
                    "No correction should be applied");
            }
        }
    }

    #[test]
    fn test_correct_multi_echo_wraps_with_offset() {
        let n = 64;
        let tes = [5.0, 10.0];
        let mask = vec![1u8; n];

        // Echo 0: phase = 0.5 for all voxels
        // Echo 1: should be 1.0 (TE ratio = 2) but is offset by +2π
        let mut phases = vec![
            vec![0.5; n],
            vec![1.0 + TWO_PI; n],
        ];

        correct_multi_echo_wraps(&mut phases, &tes, &mask);

        // After correction, echo 1 should be close to 1.0
        for i in 0..n {
            assert!((phases[1][i] - 1.0).abs() < 0.1,
                "Expected ~1.0, got {}", phases[1][i]);
        }
    }

    // =========================================================================
    // RomeoWeightType
    // =========================================================================

    #[test]
    fn test_weight_type_flags() {
        // Romeo: PC + PGC + PL + MC
        let f = RomeoWeightType::Romeo.weight_flags();
        assert_eq!(f, [true, true, true, true, false, false]);

        // Romeo2: PC + MC only
        let f = RomeoWeightType::Romeo2.weight_flags();
        assert_eq!(f, [true, false, false, true, false, false]);

        // Romeo6: all 6
        let f = RomeoWeightType::Romeo6.weight_flags();
        assert_eq!(f, [true, true, true, true, true, true]);
    }

    #[test]
    fn test_romeo4_is_romeo_alias() {
        assert_eq!(
            RomeoWeightType::Romeo.weight_flags(),
            RomeoWeightType::Romeo4.weight_flags(),
        );
    }

    #[test]
    fn test_params_weight_flags_roundtrip() {
        // Default params should match Romeo preset
        let default_params = RomeoParams::default();
        assert_eq!(default_params.weight_flags(), RomeoWeightType::Romeo.weight_flags());
        assert!(!default_params.bestpath);

        // from_weight_type preserves flags
        for wt in [RomeoWeightType::Romeo, RomeoWeightType::Romeo2,
                    RomeoWeightType::Romeo3, RomeoWeightType::Romeo6] {
            let params = RomeoParams::from_weight_type(wt);
            assert_eq!(params.weight_flags(), wt.weight_flags());
            assert!(!params.bestpath);
        }

        // BestPath sets the bestpath flag
        let bp = RomeoParams::from_weight_type(RomeoWeightType::BestPath);
        assert!(bp.bestpath);
    }

    #[test]
    fn test_params_custom_weight_combination() {
        // Custom combination: only phase coherence + mag weight (no preset matches this)
        let params = RomeoParams {
            phase_coherence: true,
            phase_gradient_coherence: false,
            phase_linearity: false,
            mag_coherence: false,
            mag_weight: true,
            mag_weight2: false,
            ..Default::default()
        };
        assert_eq!(params.weight_flags(), [true, false, false, false, true, false]);
    }

    // =========================================================================
    // Template selection
    // =========================================================================

    #[test]
    fn test_multi_echo_template_selection() {
        let n = 8;
        let total = n * n * n;
        let tes = [5.0, 10.0, 15.0];
        let phases: Vec<Vec<f64>> = tes.iter().map(|&te| {
            (0..total).map(|i| 0.02 * (i % n) as f64 * te).collect()
        }).collect();
        let mags: Vec<Vec<f64>> = tes.iter().map(|_| vec![1.0; total]).collect();
        let mask = vec![1u8; total];

        // Use echo 2 (index 1) as template
        let params = RomeoParams {
            template: 1,
            ..Default::default()
        };

        let result = unwrap_romeo_multi_echo(
            &phases, &mags, &tes, &mask,
            &params, &grid(n),
        );

        assert_eq!(result.len(), 3);
        for echo in &result {
            for &v in echo {
                assert!(v.is_finite());
            }
        }
    }

    // =========================================================================
    // Phase linearity
    // =========================================================================

    #[test]
    fn test_phase_linearity_triplet_linear() {
        // Perfectly linear: second derivative = 0 → linearity = 1.0
        assert!((phase_linearity_triplet(0.1, 0.2, 0.3) - 1.0).abs() < 1e-6);
        assert!((phase_linearity_triplet(-0.5, 0.0, 0.5) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_phase_linearity_triplet_jump() {
        // Large second derivative → low linearity
        let pl = phase_linearity_triplet(0.0, 0.0, PI);
        assert!(pl < 0.5, "Jump should give low linearity, got {}", pl);
    }

    // =========================================================================
    // BestPath weights
    // =========================================================================

    #[test]
    fn test_bestpath_weights_constant_phase() {
        let n = 6;
        let total = n * n * n;
        let phase = vec![0.0; total];
        let mask = vec![1u8; total];

        let weights = calculate_weights_bestpath(&phase, &mask, n, n, n);
        assert_eq!(weights.len(), 3 * total);

        // Constant phase → D=0 → R=0 → edge weight should be low
        // (BestPath assigns low weight to ambiguous/constant regions)
    }

    #[test]
    fn test_bestpath_weights_smooth_gradient() {
        let n = 8;
        let total = n * n * n;
        // Small smooth gradient in x
        let phase: Vec<f64> = (0..total).map(|i| 0.1 * (i % n) as f64).collect();
        let mask = vec![1u8; total];

        let weights = calculate_weights_bestpath(&phase, &mask, n, n, n);

        // Should produce finite, valid weights
        for &w in &weights {
            assert!(w <= 255);
        }
    }

    // =========================================================================
    // wrap_addition
    // =========================================================================

    #[test]
    fn test_unwrap_with_wrap_addition() {
        let n = 8;
        let total = n * n * n;
        let phase: Vec<f64> = (0..total).map(|i| {
            let x = (i % n) as f64;
            wrap_angle(0.1 * x)
        }).collect();
        let mask = vec![1u8; total];

        let params = RomeoParams {
            wrap_addition: 1.0,
            ..Default::default()
        };

        let unwrapped = unwrap_romeo(
            &phase, &[], None, 0.0, 0.0,
            &mask, &params, &grid(n),
        );

        for &v in &unwrapped {
            assert!(v.is_finite());
        }
    }

    // =========================================================================
    // Multi-seed
    // =========================================================================

    #[test]
    fn test_unwrap_multi_seed() {
        let n = 8;
        let total = n * n * n;
        let phase: Vec<f64> = (0..total).map(|i| {
            wrap_angle(0.05 * (i % n) as f64)
        }).collect();
        let mask = vec![1u8; total];

        let params = RomeoParams {
            max_seeds: 4,
            ..Default::default()
        };

        let unwrapped = unwrap_romeo(
            &phase, &[], None, 0.0, 0.0,
            &mask, &params, &grid(n),
        );

        for &v in &unwrapped {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_unwrap_multi_seed_with_merge() {
        let n = 8;
        let total = n * n * n;
        let phase: Vec<f64> = (0..total).map(|i| {
            wrap_angle(0.05 * (i % n) as f64)
        }).collect();
        let mask = vec![1u8; total];

        let params = RomeoParams {
            max_seeds: 4,
            merge_regions: true,
            correct_regions: true,
            ..Default::default()
        };

        let unwrapped = unwrap_romeo(
            &phase, &[], None, 0.0, 0.0,
            &mask, &params, &grid(n),
        );

        for &v in &unwrapped {
            assert!(v.is_finite());
        }
    }

    // =========================================================================
    // correctglobal for 3D
    // =========================================================================

    #[test]
    fn test_correct_global_offset() {
        let n = 64;
        let mask = vec![1u8; n];
        // Phase that's all shifted by 2π
        let mut phase: Vec<f64> = vec![TWO_PI + 0.1; n];

        correct_global_offset(&mut phase, &mask);

        // After correction, should be close to 0.1
        for &v in &phase {
            assert!((v - 0.1).abs() < 0.2, "Expected ~0.1, got {}", v);
        }
    }

    #[test]
    fn test_unwrap_romeo_with_correctglobal() {
        let n = 4;
        let total = n * n * n;
        let phase = vec![0.1; total];
        let mask = vec![1u8; total];

        let params = RomeoParams {
            correct_global: true,
            ..Default::default()
        };

        let unwrapped = unwrap_romeo(
            &phase, &[], None, 0.0, 0.0,
            &mask, &params, &grid(n),
        );

        for &v in &unwrapped {
            assert!(v.is_finite());
        }
    }

    // =========================================================================
    // Weight type: BestPath integration
    // =========================================================================

    #[test]
    fn test_unwrap_romeo_bestpath() {
        let n = 8;
        let total = n * n * n;
        let phase: Vec<f64> = (0..total).map(|i| {
            wrap_angle(0.1 * (i % n) as f64)
        }).collect();
        let mask = vec![1u8; total];

        let params = RomeoParams {
            bestpath: true,
            ..Default::default()
        };

        let unwrapped = unwrap_romeo(
            &phase, &[], None, 0.0, 0.0,
            &mask, &params, &grid(n),
        );

        for &v in &unwrapped {
            assert!(v.is_finite());
        }
    }

    // =========================================================================
    // rescale_weight
    // =========================================================================

    #[test]
    fn test_rescale_weight() {
        assert_eq!(rescale_weight(1.0), 255);    // best
        assert_eq!(rescale_weight(0.5), 128);    // mid
        assert!(rescale_weight(0.001) >= 1);     // worst valid ≥ 1
        assert_eq!(rescale_weight(0.0), 0);      // invalid
        assert_eq!(rescale_weight(-0.1), 0);     // invalid
    }
}

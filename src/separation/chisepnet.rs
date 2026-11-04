//! χ-sepnet deep-learning χ-separation (`onnx` feature).
//!
//! Two entry points. [`chisepnet_toolbox`] is the authors' route, the χ-separation toolbox
//! v1.1.3's `chi_sepnet_general` (see *The authors' route* below), and the pipeline's default
//! ([`ChiSepNetRoute::Toolbox`]). [`chisepnet`] is the network alone on the acquired grid with the
//! caller's QSM ([`ChiSepNetRoute::Native`]), described first.
//!
//! χ-sepnet (SNU-LIST) is a 3D U-Net that maps a 3-channel patch — [QSM (χ_total,
//! ppm), local field (ppm), R2′/Dr] each z-scored by training statistics — to
//! paramagnetic (χ+) and diamagnetic (χ−) source magnitudes (z-scored). The network
//! is a fixed **192×192×128** patch; we run it as an overlapping sliding window over
//! the whole volume and average the overlaps, then de-normalize.
//!
//! Recipe (mirrors the SNU-LIST / QSM-CI `recon.py`): z-score each channel, zero
//! outside the mask, end-pad each dim up to the patch size, tile with a 0.75 stride
//! plus a final flush patch, average overlaps, de-normalize, crop, and mask. `Dr`
//! (114 Hz/ppm, the network's COSMOS-referenced relaxivity) scales R2′ into the
//! network's ppm-equivalent input channel. Negative de-normalised magnitudes are set to 0, as the
//! authors' `test.py` does ("zero-truncation"). We return χ− as a signed (≤ 0) value to
//! match the crate's separation convention `(chi_pos ≥ 0, chi_neg ≤ 0, chi_total)`.
//!
//! **Array orientation.** The network takes no B0 direction, so whatever orientation the caller
//! hands it is the one it reconstructs in. Kim et al. (Hum Brain Mapp 2025,
//! doi:10.1002/hbm.70136) say what that should be, and the two halves are not equally strict.
//! The training scans were acquired so that "the imaging slab was axial so that the z-axis was
//! oriented along the B0 field", and the patches were then augmented by rotating "only … within
//! the plane perpendicular to B0 (degree: −90° to +90°) to avoid complication from B0 orientation
//! dependent R2* in white matter". So the slice axis carries a real requirement, while the
//! in-plane order is deliberately covered by augmentation across the full quarter turn.
//!
//! The complete order is (A-P, L-R, I-S). The in-plane half of that is not in the paper: it comes
//! from the brain mask shipped in `Data/test_file.mat` — a dummy file, but the mask is real, and
//! axis 1 is the mirror-symmetry axis in all six COSMOS orientations (Dice 0.87-0.97 against
//! 0.78-0.88) while axis 0 is the longest — corroborated by the authors' own axis naming in
//! `train_data_patch.py`. The left/right *sign* stays undetermined, and the numbers below say it
//! does not matter. Nothing in the reference code enforces any of this: neither the toolbox's
//! `Chisep_script.m`, nor the authors' `test.py`, nor the QSM-CI `recon.py` ever permutes or flips
//! an axis, and neither does this function — it is handed a [`Grid`] (dims and voxel sizes), not
//! an affine, so the caller owns the orientation.
//!
//! That turns out to cost very little. Measured on the QSM Reconstruction Challenge 2.0 head
//! phantom (QSM-CI's χ-separation dataset, which builds χ+/χ− on that head model): correlation
//! against the ground-truth source maps, one 192³ pass, the volume relabelled and the result
//! relabelled back.
//!
//! | array orientation | χ+ r | χ− r |
//! |---|---|---|
//! | RAS — what QSMxT feeds today | 0.9480 | 0.8072 |
//! | all 8 in-plane relabellings (slice axis still along B0) | 0.9477–0.9483 | 0.8053–0.8078 |
//! | slice axis moved to A-P (coronal-like) | 0.9471 | 0.7914 |
//! | slice axis moved to L-R (sagittal-like) | 0.9485 | 0.7855 |
//!
//! which is what the ±90° in-plane augmentation predicts: the in-plane order is immaterial, and
//! even putting the slice axis across B0 costs only about 0.02 of χ− correlation and nothing of
//! χ+, less than the training recipe alone would lead you to expect. Two of the
//! three input channels determine χ± algebraically and carry no orientation at all
//! (χ+ + χ− = χ_total, χ+ − χ− ≈ R2′/Dr), which is most of why. The outputs are not *identical*
//! across orientations — two in-plane orientations of the same in-vivo volume differ by ~12%
//! NRMSE, against a ~3% floor from a pure 16-voxel translation — but neither is nearer the truth,
//! so there is no orientation to prefer. Running on an oblique acquisition is a different
//! question, and still wants [`resample_to_axial`](crate::geometry::resample_to_axial): a
//! relabelling can only fix axes that are already cardinal.
//!
//! **Patch size.** The published graph declares that patch as a fixed input shape, but it is
//! fully convolutional (Conv/Relu/MaxPool/ConvTranspose/Concat only), so the hosted
//! `chi-sepnet.onnx` has its spatial axes re-declared as dynamic — bit-identical at the
//! authors' patch, and able to run smaller ones. [`ChiSepNetParams::patch`] defaults to the
//! authors' 192×192×128; a 32-bit host (WASM) cannot afford it — one 64-channel activation at
//! that size is 1.2 GB — and uses [`WASM_PATCH`] instead. Patch dimensions must be multiples
//! of 16 (four pooling levels) or the skip-connection concatenations misalign.
//!
//! Weights are not bundled; the caller passes the exported `chi-sepnet.onnx` bytes.
//!
//! # Parity
//!
//! Measured against QSM-CI's `chisepnet` entry (the SNU-LIST toolbox's `240904_xsepnet.onnx` and
//! its normalisation `.mat`, run by QSM-CI's `recon.py`) on a 36×32×28 in-vivo box of
//! `prisma-bridge-run1`, at the authors' 192×192×128 patch (QSM.rs#330):
//!
//! - The hosted `chi-sepnet.onnx` (spatial axes re-declared dynamic) gives exactly the toolbox
//!   graph's outputs on the same patch: max |Δ| 0.
//! - The normalisation constants above are the `.mat`'s values.
//! - The only difference is zero-truncation. This function sets negative de-normalised
//!   magnitudes to 0, as the authors' `test.py` does. QSM-CI's `recon.py` keeps them: on this box
//!   2 487 of 19 996 χ+ voxels and 5 χ− voxels are negative. Against the entry as it runs, that is
//!   1.73 % NRMSE for χ+ and 0.045 % for χ−. Against the entry's output truncated the same way it
//!   is 0.0000 % (max |Δ| 1.4e-8 ppm). QSMxT/QSM-CI#333 adds the truncation to the entry.
//! - The SNU MATLAB toolbox truncates too. Its `chi_sepnet_general.p` → `chi_sepnet.p` (v1.1.3)
//!   was run as a black box on Bunya, MATLAB R2023b, 2026-10-30. The ONNX import was replaced by a
//!   one-layer network with constant z-scored outputs. χ+ = −2 and χ− = −3 came back as 0 and 0.
//!   χ+ = +2 came back as 2·0.03526 + 0.02591 = 0.0964. χ− = +0.5 came back as 0.0407, with
//!   χ_total = χ+ − |χ−|. So negative de-normalised magnitudes are set to 0, and positive ones pass
//!   through unchanged.
//! - The same run shows that the toolbox's χ-sepnet path works differently from this function.
//!   It takes the local field in Hz and R2′ (not a QSM). It makes the QSM channel itself with its
//!   own `240904_QSMnet.onnx`, after rotating to B0 (`CalculateRotMat`) and resampling to 1 mm
//!   (`interp3`): [`chisepnet_toolbox`], *The authors' route* below.
//!
//! # Reference pipeline
//!
//! Kim et al. 2025 (arXiv:2409.14030, §2.1.1 and §2.2):
//!
//! - **Training data:** 3 T (Siemens Tim Trio), 1 × 1 × 1 mm, 6 echoes, TE 7.70–32.85 ms
//!   (ΔTE ≈ 5 ms), six head orientations, with 64³ training patches.
//! - **Field chain:** magnitude by sum of squares, phase by MCPC-3D. Mask by FSL BET on the
//!   combined magnitude. Each echo unwrapped with ROMEO, background removal by V-SHARP with "a
//!   spherical mean value size of 20" per echo, then weighted echo averaging (Wu 2012) into the
//!   local field. The toolbox driver instead runs V-SHARP `smvsize = 12` after echo averaging
//!   (`Chisep_script.m:113-139`); the two disagree (open question).
//! - **QSM channel:** QSMnetME, a QSMnet trained on multi-echo data, applied to the local field
//!   (§2.2.2): [`crate::inversion::qsmnetme`], which [`chisepnet_toolbox`] runs.
//! - **R2' channel:** R2* by ARLO; R2 by StimFit dictionary matching of a 2D multi-echo SE with
//!   stimulated-echo correction (QSM.rs: `r2_epg`); R2' = R2* − R2 with negatives set to 0. With
//!   GRE only, R2PRIMEnet ([`crate::relaxometry`]) predicts R2'.
//! - **Resolution:** trained at 1 mm isotropic. The toolbox either interpolates the inputs to
//!   1 mm and returns 1 mm maps, or runs a resolution-generalisation pipeline for QSMnetME
//!   (`Chisep_script.m:188-195`, `chi_sepnet_general` vs `..._wResolGen`). The README says the
//!   networks handle resolutions above 0.6 mm. [`chisepnet_toolbox`] runs the first (the
//!   script's `resgen = false`; despite the script's comment it returns its maps on the acquired
//!   grid); [`chisepnet`] runs on the native grid with neither. The resolution-generalisation
//!   branch (`chi_sepnet_general_new_wResolGen`) is not ported.
//! - **Field strength:** 3 T only. At 7 T the authors keep this 3 T network and feed it the 7 T
//!   local field, a QSMnetME QSM and a *3 T-equivalent* R2' from R2PRIMEnet7T (Kim J. et al.,
//!   arXiv:2410.12239, §2.2.1). They name the choice of D_r at 7 T as unresolved (§4). Feeding a
//!   7 T measured R2' here is off-distribution.
//! - **Orientation:** "For non-axial input data, the orientation was rotated such that the B0
//!   orientation is along the z-direction" (§2.1.2), as covered above.
//!
//! # Agreement with the original
//!
//! `tests/models_onnx/separation.rs` runs it on a 36×32×28 box of in-vivo data (1 mm, 2.89 T,
//! across the brain's edge) against its original's verified QSM-CI entry, SNU-LIST's
//! `240904_xsepnet.onnx` in QSM-CI's sliding window (`chisepnet`) (QSMxT/QSM-CI#316; the fixture is
//! regenerated in Actions by `tests/data/dl_separation_qsmci_small.py`, from the entry with the
//! authors' zero-truncation, QSMxT/QSM-CI#333): at the authors' 192×192×128 patch, **χ+ NRMSE
//! 5.8e-5 %**, χ− 3.4e-5 % (max |Δ| 2.4e-7 ppm; measured 2026-11-02), float32 rounding. The 1.7 % /
//! 0.045 % first measured was the entry's missing truncation (above; astewartau/QSM.rs#330). The
//! patch matters for parity: a volume smaller than the patch is zero-padded up to it, and the
//! network's activations in that padding differ from the convolutions' own zero padding, so a
//! smaller [`ChiSepNetParams::patch`] changes the result near the volume's far edges.
//!
//! # The authors' route
//!
//! [`chisepnet_toolbox`] is `chi_sepnet_general` of the toolbox v1.1.3, which `Chisep_script.m`
//! calls with `resgen = false` (l. 194). The P-code was probed as a black box in licensed MATLAB
//! R2023b (QSM.rs#377, 2026-11-03; method in [`crate::utils::chisep_frame`]): the inputs go into
//! the training frame (B0 along the third axis, 1 mm, centred, padded to multiples of 16), QSMnetME
//! makes the QSM channel there ([`crate::inversion::qsmnetme`]), R2′ is the measured map or, for
//! R2*, R2PRIMEnet's output, χ-sepnet runs once over the whole frame, and χ± come back onto the
//! acquired grid. The steps and what was measured for each are listed on [`chisepnet_toolbox`].
//! Findings that settle earlier open points:
//!
//! - **Output grid.** The toolbox returns its maps on the acquired grid (`interp3 'spline'` at
//!   `R⁻¹·x`), not at 1 mm as the script's comment says, so no resampling of our own is involved.
//! - **R2* is not clipped** before R2PRIMEnet, and R2PRIMEnet's output reaches χ-sepnet unclipped
//!   (the toolbox clips its de-normalised copy, which it does not use).
//! - **Whole volume, no tiling**: the toolbox replaces each network's input layer to fit the frame.
//!   [`ChiSepNetParams::frame_patch`] tiles instead, for 32-bit hosts (not the authors').
//! - **B0 direction**: the toolbox does not normalise it; a non-unit vector gives it a slightly
//!   non-orthogonal rotation. QSM.rs normalises first (identical for unit vectors). For B0 = −ẑ the
//!   toolbox's rotation is `NaN`; QSM.rs takes the half turn about the first axis.
//! - **7 T**: the authors feed χ-sepnet a 3 T-equivalent R2′ from R2PRIMEnet7T (toolbox v1.2's
//!   `R2PNET_7T.onnx`, only through SNU's form, not in v1.1.3 and not available to us). So the
//!   R2* route refuses fields outside 3 T ± [`B0_TOLERANCE_T`](crate::relaxometry::B0_TOLERANCE_T),
//!   unless [`ChiSepNetParams::ignore_field_mismatch`]; a measured R2′ is taken at any field.
//! - **Not part of the route**: `Chisep_script.m` Tukey-windows the complex multi-echo data in
//!   k-space before everything (`tukey_fac` 0.4 for Siemens and GE, 0 for Philips, l. 69-73), so its
//!   mask, R2* fit and phase see windowed data. That happens upstream of the field and R2* maps
//!   this function takes, so it is the caller's: [`crate::utils::tukey_window`] ports it, and the
//!   rules `chi-sepnet.tukey-siemens-ge` and `chi-sepnet.tukey-philips` suggest the authors' value
//!   by manufacturer on this route, as the authors' setting (`Kind::Reference`: shown, never
//!   preselected).
//!
//! **Agreement with the toolbox.** On the probe volumes (64×60×48 at 0.8×0.9×1.1 mm with B0 along
//! (0.1, 0.2, 0.97)/|·| and R2*, and at 1.1×0.7×0.9 mm with B0 along (−0.15, 0.05, 0.95)/|·| and R2′;
//! MATLAB R2023b with the networks computed by ONNX Runtime), [`chisepnet_toolbox`] matches the
//! toolbox's χ+ to 5.0e-5 % / 5.2e-5 % NRMSE and χ− to 5.0e-5 % / 5.1e-5 % (max |Δ| 1.8e-7 ppm):
//! float32 rounding, tract against ONNX Runtime. QSM-CI's Python reproduction
//! (`algorithms/chisepnet/chisep_route.py`) matches the toolbox to 1.4e-5 to 2.1e-5 %.
//! `tests/models_onnx/snu_route.rs` checks [`chisepnet_toolbox`] (R2′ and R2*) and
//! [`crate::inversion::qsmnetme`] against that reproduction on the in-vivo box, as acquired and
//! with a tilted, anisotropic geometry (`tests/data/dl_snu_route_small.py`).
//!
//! # Pipeline rules
//!
//! The facts above that a configuration can depart from are published as rules in
//! [`crate::pipeline::rules::RULES`] (each with its strength, basis, reason and source): `chi-sep.mask-bet`, `chi-sep.unwrap-romeo`, `chi-sep.echo-combination`, `chi-sep.echo-weights`, `chi-sep.bg-vsharp`, `chi-sepnet.min-resolution`, `chi-sepnet.tukey-siemens-ge`, `chi-sepnet.tukey-philips`.

// The parameters are plain Rust, so configs can carry them; only inference needs `onnx`.
#![cfg_attr(not(feature = "onnx"), allow(dead_code))]

#[cfg(feature = "onnx")]
use crate::grid::Grid;
#[cfg(feature = "onnx")]
use crate::models::onnx::{OnnxError, OnnxModel, Tensor};
#[cfg(feature = "onnx")]
use crate::utils::sliding::starts;

/// The authors' patch: `(D, H, W)` = `(x, y, z)`.
pub const AUTHORS_PATCH: (usize, usize, usize) = (192, 192, 128);

/// A patch that fits a 32-bit (WASM) heap. Peak memory, measured natively (single-threaded) on a
/// 184×256×128 volume: 1.7 GB at this patch, 4.5 GB at [`AUTHORS_PATCH`]. On the QSM-CI
/// χ-separation phantom it agrees with the authors' patch to r ≥ 0.996 on both maps on
/// [`ChiSepNetRoute::Native`]; as [`ChiSepNetParams::frame_patch`] on the default
/// [`ChiSepNetRoute::Toolbox`] (QSMnetME and χ-sepnet both windowed) it agrees with the
/// whole-frame passes to r = 0.990 (χ+) / 0.983 (χ−) (`tests/chisep_xsepnet.rs`).
pub const WASM_PATCH: (usize, usize, usize) = (128, 128, 64);

/// Four pooling levels, so every patch dimension must be a multiple of this. Public (and built
/// without `onnx`) so a host can check a patch, or state the rule, without restating the number.
pub const POOL_MULTIPLE: usize = 16;

/// Which route χ-sepnet's inputs take ([`ChiSepNetParams::route`]).
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ChiSepNetRoute {
    /// The authors' route, `chi_sepnet_general` of the χ-separation toolbox v1.1.3
    /// ([`chisepnet_toolbox`]): the inputs resampled into the training frame (B0 along the third
    /// axis, 1 mm), the QSM channel made by QSMnetME there, R2′ from R2PRIMEnet when only R2* is
    /// given, the whole volume in one pass, and the maps resampled back. The default.
    #[default]
    Toolbox,
    /// The network alone on the acquired grid ([`chisepnet`]): the caller's QSM and R2′, a
    /// sliding window of [`ChiSepNetParams::patch`], no resampling. QSM-CI's first χ-sepnet entry
    /// ran it this way (with the ground-truth χ as the QSM).
    Native,
}

impl ChiSepNetRoute {
    /// Stable id: `"toolbox"` or `"native"` (the pipeline rules' `separation.chi_sepnet.route`).
    pub fn id(self) -> &'static str {
        match self {
            ChiSepNetRoute::Toolbox => "toolbox",
            ChiSepNetRoute::Native => "native",
        }
    }
}

/// Inference parameters for [`chisepnet`] and [`chisepnet_toolbox`].
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChiSepNetParams {
    /// The input route; see [`ChiSepNetRoute`]. Defaults to the authors' ([`ChiSepNetRoute::Toolbox`]).
    pub route: ChiSepNetRoute,
    /// [`ChiSepNetRoute::Native`]'s sliding-window patch `(D, H, W)`; each dimension a multiple of
    /// 16. Defaults to [`AUTHORS_PATCH`]; WASM hosts pass [`WASM_PATCH`].
    pub patch: (usize, usize, usize),
    /// [`ChiSepNetRoute::Toolbox`]: `None` (default) runs each network on the whole training-frame
    /// volume in one pass, as the toolbox does; `Some(patch)` runs a sliding window of that patch
    /// instead, for hosts that cannot hold the whole volume's activations (not the authors').
    pub frame_patch: Option<(usize, usize, usize)>,
    /// [`ChiSepNetRoute::Toolbox`] given R2*: run R2PRIMEnet even when the field strength is
    /// outside [`B0_TOLERANCE_T`](crate::relaxometry::B0_TOLERANCE_T) of its 3 T training (see
    /// [`R2PrimeNetParams::ignore_field_mismatch`](crate::relaxometry::R2PrimeNetParams)). Off
    /// by default.
    pub ignore_field_mismatch: bool,
}

impl Default for ChiSepNetParams {
    fn default() -> Self {
        Self { route: ChiSepNetRoute::Toolbox, patch: AUTHORS_PATCH, frame_patch: None, ignore_field_mismatch: false }
    }
}

impl ChiSepNetParams {
    /// Settings for a 32-bit (WASM) heap: the defaults with [`WASM_PATCH`] for both routes.
    pub fn low_memory() -> Self {
        Self { patch: WASM_PATCH, frame_patch: Some(WASM_PATCH), ..Self::default() }
    }

    /// Check the patches: every dimension a positive multiple of 16 (the network's four pooling
    /// levels). [`chisepnet`] panics on what this refuses.
    pub fn validate(&self) -> Result<(), String> {
        let check = |name: &str, (d, h, w): (usize, usize, usize)| -> Result<(), String> {
            for (axis, v) in [("D", d), ("H", h), ("W", w)] {
                if v == 0 || v % POOL_MULTIPLE != 0 {
                    return Err(format!(
                        "χ-sepnet {name} {axis} = {v} must be a positive multiple of {POOL_MULTIPLE} \
                         (four pooling levels)"
                    ));
                }
            }
            Ok(())
        };
        check("patch", self.patch)?;
        if let Some(p) = self.frame_patch {
            check("frame patch", p)?;
        }
        Ok(())
    }
}

/// Training z-score constants for χ-sepnet (`xsepnet_train_patch_norm_factor…mat`):
/// each field is `(mean, std)`. Inputs are normalized `(x-mean)/std`; outputs
/// de-normalized `y*std + mean`. `dr` scales R2′ (Hz) into the ppm-equivalent
/// channel `(r2prime/dr - mean)/std`.
#[derive(Clone, Copy, Debug)]
pub struct ChiSepNetNorm {
    pub qsm: (f64, f64),
    pub field: (f64, f64),
    pub r2prime: (f64, f64),
    pub chi_pos: (f64, f64),
    pub chi_neg: (f64, f64),
    pub dr: f64,
}

impl Default for ChiSepNetNorm {
    /// Constants from `xsepnet_train_patch_norm_factor_inplane_largedegree_romeo_arlo.mat`.
    fn default() -> Self {
        Self {
            qsm: (-0.0013402086915448308, 0.031504031270742416),
            field: (-0.0026045702397823334, 0.010849659331142902),
            r2prime: (0.05141879618167877, 0.06977531313896179),
            chi_pos: (0.025912819430232048, 0.03526417911052704),
            chi_neg: (0.026700690388679504, 0.027936099097132683),
            dr: 114.0,
        }
    }
}

/// Run χ-sepnet χ-separation.
///
/// * `local_field_ppm`, `qsm` (χ_total, ppm), `r2prime` (Hz) — column-major `(nx,ny,nz)`.
/// * `mask` — binary brain mask (same layout).
/// * `model_onnx` — bytes of the exported `chi-sepnet.onnx` (192×192×128, 3→2 chan).
/// * `norm` — training normalization constants.
/// * `params` — patch size; see [`ChiSepNetParams`].
///
/// Returns `(chi_pos ≥ 0, chi_neg ≤ 0, chi_total = chi_pos + chi_neg)` in ppm,
/// masked, in the same layout.
///
/// # Panics
/// If a patch dimension is not a multiple of 16 (the network's four pooling levels).
#[cfg(feature = "onnx")]
#[allow(clippy::too_many_arguments)]
pub fn chisepnet(
    local_field_ppm: &[f64],
    qsm: &[f64],
    r2prime: &[f64],
    mask: &[u8],
    grid: &Grid,
    model_onnx: &[u8],
    norm: &ChiSepNetNorm,
    params: &ChiSepNetParams,
) -> Result<super::ChiSepMaps, OnnxError> {
    let (nx, ny, nz) = grid.dims;
    if let Err(e) = params.validate() {
        panic!("{e}");
    }
    let (pd, ph, pw) = params.patch;
    let n = nx * ny * nz;
    for (name, v) in [("field", local_field_ppm), ("qsm", qsm), ("r2prime", r2prime)] {
        assert_eq!(v.len(), n, "{name} length must match grid");
    }
    assert_eq!(mask.len(), n, "mask length must match grid");

    // End-pad each dim up to at least the patch size (col-major padded volume).
    let (px, py, pz) = (nx.max(pd), ny.max(ph), nz.max(pw));
    let mut ch = [vec![0.0f32; px * py * pz], vec![0.0f32; px * py * pz], vec![0.0f32; px * py * pz]];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let i = x + nx * (y + ny * z);
                if mask[i] == 0 {
                    continue;
                }
                let p = x + px * (y + py * z);
                ch[0][p] = ((qsm[i] - norm.qsm.0) / norm.qsm.1) as f32;
                ch[1][p] = ((local_field_ppm[i] - norm.field.0) / norm.field.1) as f32;
                ch[2][p] = ((r2prime[i] / norm.dr - norm.r2prime.0) / norm.r2prime.1) as f32;
            }
        }
    }

    let model = OnnxModel::load(model_onnx)?;
    let plane = pd * ph * pw;
    let mut acc0 = vec![0.0f64; px * py * pz];
    let mut acc1 = vec![0.0f64; px * py * pz];
    let mut wsum = vec![0.0f64; px * py * pz];
    for &x0 in &starts(px, pd) {
        for &y0 in &starts(py, ph) {
            for &z0 in &starts(pz, pw) {
                // Build the [1,3,pd,ph,pw] NCDHW patch (D=x, H=y, W=z).
                let mut buf = vec![0.0f32; 3 * plane];
                for i in 0..pd {
                    for j in 0..ph {
                        for k in 0..pw {
                            let p = (x0 + i) + px * ((y0 + j) + py * (z0 + k));
                            let o = (i * ph + j) * pw + k;
                            buf[o] = ch[0][p];
                            buf[plane + o] = ch[1][p];
                            buf[2 * plane + o] = ch[2][p];
                        }
                    }
                }
                let out = model.run_single(&Tensor::new(vec![1, 3, pd, ph, pw], buf))?;
                if out.data.len() < 2 * plane {
                    return Err(OnnxError::Run("expected 2-channel output".into()));
                }
                for i in 0..pd {
                    for j in 0..ph {
                        for k in 0..pw {
                            let p = (x0 + i) + px * ((y0 + j) + py * (z0 + k));
                            let o = (i * ph + j) * pw + k;
                            acc0[p] += out.data[o] as f64;
                            acc1[p] += out.data[plane + o] as f64;
                            wsum[p] += 1.0;
                        }
                    }
                }
            }
        }
    }

    // Average overlaps, de-normalize, crop to (nx,ny,nz), mask. χ− → signed (≤ 0).
    let mut chi_pos = vec![0.0f64; n];
    let mut chi_neg = vec![0.0f64; n];
    let mut chi_total = vec![0.0f64; n];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let i = x + nx * (y + ny * z);
                if mask[i] == 0 {
                    continue;
                }
                let p = x + px * (y + py * z);
                let w = wsum[p].max(1.0);
                let pos = (acc0[p] / w) * norm.chi_pos.1 + norm.chi_pos.0;
                let neg = (acc1[p] / w) * norm.chi_neg.1 + norm.chi_neg.0;
                // test.py: pred_x_pos[pred_x_pos < 0] = 0; pred_x_neg[pred_x_neg < 0] = 0
                (chi_pos[i], chi_neg[i], chi_total[i]) = super::sources_from_magnitudes(pos, neg);
            }
        }
    }
    Ok((chi_pos, chi_neg, chi_total))
}

/// χ-sepnet's R2 input for [`chisepnet_toolbox`].
#[derive(Clone, Copy, Debug)]
pub enum ChiSepNetR2<'a> {
    /// A measured R2′ (Hz): χ-sepnet-R2′.
    R2Prime(&'a [f64]),
    /// R2* (Hz): χ-sepnet-R2*, with R2′ predicted by R2PRIMEnet inside the route.
    R2Star(&'a [f64]),
}

/// The weights [`chisepnet_toolbox`] runs: χ-sepnet, QSMnetME, and R2PRIMEnet for
/// [`ChiSepNetR2::R2Star`].
#[derive(Clone, Copy, Debug)]
pub struct ChiSepNetToolboxModels<'a> {
    pub chisepnet: &'a [u8],
    pub qsmnetme: &'a [u8],
    pub r2primenet: Option<&'a [u8]>,
}

/// χ-sepnet by the authors' route: the χ-separation toolbox v1.1.3's `chi_sepnet_general`
/// (`Chisep_script.m`:177-194), recovered by black-box probing in licensed MATLAB R2023b
/// (QSM.rs#377; method in [`crate::utils::chisep_frame`]). Unlike [`chisepnet`] it takes no QSM:
///
/// 1. the local field, the R2 map and the mask go into the training frame
///    ([`TrainingFrame`](crate::utils::chisep_frame::TrainingFrame): B0 along the third axis,
///    1 mm, `interp3 'spline'` / `'linear'` > 0.5);
/// 2. the QSM channel is QSMnetME's ([`crate::inversion::qsmnetme`], whose field channel is
///    χ-sepnet's field channel too);
/// 3. the R2′ channel is `(R2′/Dr − mean)/std` for a measured R2′, or, given R2*, R2PRIMEnet's
///    raw z-scored output for `(R2*/Dr − r2star_mean)/r2star_std`. Neither is clipped: the
///    toolbox clips its de-normalised R2′ (`max(·, 0)`) but feeds χ-sepnet the unclipped channel,
///    and does not clip negative R2* before R2PRIMEnet (measured: the channels matched the
///    unclipped forms to float32 rounding, 2 040 and 4 691 negative frame voxels). These are
///    intermediates the route never returns, so QSM.rs's rule against negative R2′ maps
///    (applied by [`r2primenet`](crate::relaxometry::r2primenet)) does not reach them;
/// 4. every channel is 0 outside the frame mask; χ-sepnet runs on the whole frame;
///    `χ± = max(std·y + mean, 0)` inside the frame mask;
/// 5. each map is resampled back onto the acquired grid (`interp3 'spline'`), set to 0 where
///    negative (the spline overshoots) and outside the mask. `χ_total = χ+ − |χ−|`.
///
/// `local_field_ppm` is in ppm (the toolbox takes Hz and divides by `CF·1e-6`; the same).
/// `b0_tesla` is only checked for [`ChiSepNetR2::R2Star`]: R2PRIMEnet is trained at 3 T and the
/// v1.1.3 toolbox has no 7 T network (v1.2's `R2PNET_7T.onnx` is not available to us), so other
/// field strengths are refused unless [`ChiSepNetParams::ignore_field_mismatch`] is set.
///
/// Returns `(χ+ ≥ 0, χ− ≤ 0, χ_total)` in ppm on the acquired grid.
#[cfg(feature = "onnx")]
#[allow(clippy::too_many_arguments)]
pub fn chisepnet_toolbox(
    local_field_ppm: &[f64],
    r2: ChiSepNetR2,
    mask: &[u8],
    grid: &Grid,
    b0_dir: (f64, f64, f64),
    b0_tesla: f64,
    models: &ChiSepNetToolboxModels,
    norm: &ChiSepNetNorm,
    params: &ChiSepNetParams,
) -> Result<super::ChiSepMaps, OnnxError> {
    use crate::relaxometry::{R2PrimeNetNorm, B0_TOLERANCE_T, TRAINED_B0_T};
    use crate::utils::chisep_frame::{run_net, TrainingFrame};
    let n = grid.n_total();
    assert_eq!(local_field_ppm.len(), n, "field length must match grid");
    assert_eq!(mask.len(), n, "mask length must match grid");
    params.validate().map_err(OnnxError::Domain)?;
    let r2_map = match r2 {
        ChiSepNetR2::R2Prime(v) | ChiSepNetR2::R2Star(v) => v,
    };
    assert_eq!(r2_map.len(), n, "R2 map length must match grid");
    if matches!(r2, ChiSepNetR2::R2Star(_)) {
        if !params.ignore_field_mismatch && (b0_tesla - TRAINED_B0_T).abs() > B0_TOLERANCE_T {
            return Err(OnnxError::Domain(format!(
                "χ-sepnet-R2* predicts R2' with R2PRIMEnet, whose weights are trained at {TRAINED_B0_T} T, \
                 and this R2* map is from {b0_tesla} T: outside the training distribution. Give a \
                 measured R2' (χ-sepnet-R2'), or set ChiSepNetParams::ignore_field_mismatch."
            )));
        }
        if models.r2primenet.is_none() {
            return Err(OnnxError::Domain("χ-sepnet-R2* needs the R2PRIMEnet weights".into()));
        }
    }

    let frame = TrainingFrame::new(grid.dims, grid.voxel_size, b0_dir);
    let dims = frame.dims();
    let fmask = frame.mask_to_frame(mask);
    let field = frame.to_frame(local_field_ppm);
    let r2f = frame.to_frame(r2_map);

    // QSM channel: QSMnetME (its z-scored field input is χ-sepnet's field channel).
    let qnorm = crate::inversion::QsmnetmeNorm { field: norm.field, chi: norm.qsm };
    let qsmnetme = OnnxModel::load(models.qsmnetme)?;
    let (qsm, field_ch) = crate::inversion::qsmnetme::qsmnetme_in_frame(&field, &fmask, dims, &qsmnetme, &qnorm, params.frame_patch)?;
    drop(qsmnetme);
    let masked = |v: f64, m: u8| if m != 0 { v as f32 } else { 0.0 };
    let qsm_ch: Vec<f32> = qsm.iter().zip(&fmask).map(|(&q, &m)| masked((q - norm.qsm.0) / norm.qsm.1, m)).collect();
    drop(qsm);

    // R2′ channel.
    let r2p_ch: Vec<f32> = match r2 {
        ChiSepNetR2::R2Prime(_) => r2f
            .iter()
            .zip(&fmask)
            .map(|(&v, &m)| masked((v / norm.dr - norm.r2prime.0) / norm.r2prime.1, m))
            .collect(),
        ChiSepNetR2::R2Star(_) => {
            let rn = R2PrimeNetNorm::default();
            let x: Vec<f32> = r2f.iter().zip(&fmask).map(|(&v, &m)| masked((v / norm.dr - rn.r2star.0) / rn.r2star.1, m)).collect();
            let net = OnnxModel::load(models.r2primenet.expect("checked above"))?;
            let y = run_net(&net, std::slice::from_ref(&x), dims, 1, params.frame_patch)?;
            y[0].iter().zip(&fmask).map(|(&v, &m)| if m != 0 { v } else { 0.0 }).collect()
        }
    };
    drop(r2f);
    drop(field);

    let net = OnnxModel::load(models.chisepnet)?;
    let y = run_net(&net, &[qsm_ch, field_ch, r2p_ch], dims, 2, params.frame_patch)?;
    drop(net);
    let source = |c: usize, (mean, std): (f64, f64)| -> Vec<f64> {
        y[c].iter().zip(&fmask).map(|(&v, &m)| if m != 0 { (v as f64 * std + mean).max(0.0) } else { 0.0 }).collect()
    };
    let pos_f = source(0, norm.chi_pos);
    let neg_f = source(1, norm.chi_neg);
    let pos = frame.to_native(&pos_f);
    let neg = frame.to_native(&neg_f);
    let mut chi_pos = vec![0.0; n];
    let mut chi_neg = vec![0.0; n];
    let mut chi_total = vec![0.0; n];
    for i in 0..n {
        if mask[i] != 0 {
            (chi_pos[i], chi_neg[i], chi_total[i]) = super::sources_from_magnitudes(pos[i], neg[i]);
        }
    }
    Ok((chi_pos, chi_neg, chi_total))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both shipped patches are legal; anything off the 16-grid, or zero, is refused.
    #[test]
    fn patch_validation() {
        assert_eq!(ChiSepNetParams::default().patch, AUTHORS_PATCH);
        assert_eq!(ChiSepNetParams::low_memory().patch, WASM_PATCH);
        assert!(ChiSepNetParams::default().validate().is_ok());
        assert!(ChiSepNetParams::low_memory().validate().is_ok());
        for bad in [(0, 128, 64), (128, 120, 64), (128, 128, 8)] {
            let e = ChiSepNetParams { patch: bad, ..Default::default() }.validate().unwrap_err();
            assert!(e.contains("multiple of 16"), "{e}");
        }
    }
}

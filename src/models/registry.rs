//! The static table of known deep-learning QSM models.
//!
//! Weights are hosted externally and fetched on use (see [`super`]); nothing is
//! vendored here. Entries marked [`WeightStatus::Pending`] are recognized
//! targets whose ONNX weights are not yet converted and hosted — their `url`
//! and `sha256` are filled in once the exported `.onnx` is uploaded (OSF /
//! Hugging Face). Until then a host can still run them via bring-your-own-weights
//! (`$QSM_MODEL_DIR`).
//!
//! Editing checklist when a model goes from Pending → Available:
//! 1. export/convert to ONNX, upload the file,
//! 2. set `url`, `sha256` (lowercase hex), `bytes`,
//! 3. flip `status` to [`WeightStatus::Available`],
//! 4. confirm `inputs`/`outputs`/`size_divisor` match the exported graph.

use super::{Framework, ModelSpec, ModelStage, WeightFile, WeightStatus};

/// All models known to QSM-Core, in a stable order.
pub fn all_models() -> &'static [ModelSpec] {
    MODELS
}

/// Look up a model by its [`ModelSpec::id`] (case-sensitive), e.g. `"qsmnet"`.
pub fn find_model(id: &str) -> Option<&'static ModelSpec> {
    MODELS.iter().find(|m| m.id == id)
}

// A single ONNX weight file whose location is not yet known (Pending model).
const fn pending_onnx(name: &'static str) -> WeightFile {
    WeightFile { name, url: "", sha256: "", bytes: 0 }
}

const MODELS: &[ModelSpec] = &[
    // ---- Background field removal ------------------------------------------
    ModelSpec {
        id: "bfrnet",
        name: "BFRnet",
        stage: ModelStage::BackgroundRemoval,
        status: WeightStatus::Available,
        origin: Framework::Matlab,
        description: "Dual-frequency octave-convolution U-Net for background field \
                      removal (total field → local field). Fully convolutional.",
        paper: "Kames et al. / Sun group; https://github.com/sunhongfu/BFRnet",
        source: "https://github.com/sunhongfu/BFRnet",
        license: "Author-permitted (Sun group); redistribution per project agreement",
        // Hosted on OSF project erv6n (https://osf.io/erv6n). Verified: anonymous
        // download + SHA-256 match.
        files: &[WeightFile {
            name: "bfrnet.onnx",
            url: "https://osf.io/download/6a8546e927e06d15b781a605/",
            sha256: "6f693f0a02c94550179c4b5188ce652fc4bd8198ff57aaddd102fba67fe873d7",
            bytes: 79_600_612,
        }],
        inputs: &["field"],
        outputs: &["local_field"],
        size_divisor: 8,
    },
    // ---- Dipole inversion --------------------------------------------------
    ModelSpec {
        id: "xqsm",
        name: "xQSM",
        stage: ModelStage::DipoleInversion,
        status: WeightStatus::Available,
        origin: Framework::PyTorch,
        description: "Octave-convolution U-Net with a learned residual for dipole \
                      inversion (local field → susceptibility).",
        paper: "Gao et al., NMR Biomed 2021; doi:10.1002/nbm.4461",
        source: "https://github.com/sunhongfu/xQSM",
        license: "Author-permitted (Sun group)",
        // Exported from xQSM_invivo.pth (v1.0-demo) with scripts/onnx-export/export_xqsm.py;
        // torch↔onnxruntime parity max|Δ| ≈ 6e-5. Hosted on OSF project erv6n.
        files: &[WeightFile {
            name: "xqsm.onnx",
            url: "https://osf.io/download/6a8546cd27e06d15b781a601/",
            sha256: "81854ec2ca85bba25c2f9aae05efdcac299d0efffb03fe6a0e6797019e46b487",
            bytes: 20_901_826,
        }],
        inputs: &["field"],
        outputs: &["chi"],
        size_divisor: 8,
    },
    ModelSpec {
        id: "qsmnet",
        name: "QSMnet",
        stage: ModelStage::DipoleInversion,
        status: WeightStatus::Available,
        origin: Framework::TensorFlow,
        description: "3D U-Net dipole inversion trained on COSMOS. Expects 1 mm \
                      isotropic input; z-scored with training mean/std.",
        paper: "Yoon et al., NeuroImage 2018; doi:10.1016/j.neuroimage.2018.06.030",
        source: "https://github.com/SNU-LIST/QSMnet",
        license: "SNU-LIST academic; author-permitted (per project agreement)",
        // Clean PyTorch re-export of the TF1.14 checkpoint (tract-friendly NCDHW);
        // see scripts/onnx-export/export_qsmnet.py. Hosted on OSF project erv6n.
        files: &[WeightFile {
            name: "qsmnet.onnx",
            url: "https://osf.io/download/6a854e3fa325bd72433d5cb8/",
            sha256: "8fd1d79b7a9258a262a2faab9e8469757ae7735f3da6212e431acfb207c21b54",
            bytes: 397_801_854,
        }],
        inputs: &["field"],
        outputs: &["chi"],
        size_divisor: 16,
    },
    ModelSpec {
        id: "qsmnet-plus",
        name: "QSMnet+",
        stage: ModelStage::DipoleInversion,
        status: WeightStatus::Available,
        origin: Framework::TensorFlow,
        description: "QSMnet retrained with susceptibility-scaling augmentation for \
                      a wider, more linear χ range.",
        paper: "Jung et al., NeuroImage 2020; doi:10.1016/j.neuroimage.2020.116579",
        source: "https://github.com/SNU-LIST/QSMnet",
        license: "SNU-LIST academic; author-permitted (per project agreement)",
        // Clean PyTorch re-export of the TF1.14 QSMnet+_64 checkpoint (same U-Net
        // as QSMnet, different weights + norm). Hosted on OSF project erv6n.
        files: &[WeightFile {
            name: "qsmnet-plus.onnx",
            url: "https://osf.io/download/6a85500bd7ccc815476a888c/",
            sha256: "0ed31f9a1b66f75fee4a96022b6bc3838b91bc5b819714dade5ca87cd331683d",
            bytes: 397_801_854,
        }],
        inputs: &["field"],
        outputs: &["chi"],
        size_divisor: 16,
    },
    ModelSpec {
        id: "qsmgan",
        name: "QSMGAN",
        stage: ModelStage::DipoleInversion,
        status: WeightStatus::Pending,
        origin: Framework::PyTorch,
        description: "3D U-Net generator (WGAN-GP refined) for dipole inversion. \
                      Only the generator is used at inference.",
        paper: "Chen et al., NeuroImage 2020; doi:10.1016/j.neuroimage.2019.116389",
        source: "https://github.com/mmorri10/QSMGAN-LupoLab",
        license: "MIT (fork)",
        files: &[pending_onnx("qsmgan.onnx")],
        inputs: &["field"],
        outputs: &["chi"],
        size_divisor: 8,
    },
    ModelSpec {
        id: "lpcnn",
        name: "LPCNN",
        stage: ModelStage::DipoleInversion,
        status: WeightStatus::Pending,
        origin: Framework::PyTorch,
        description: "Learned proximal CNN, 3 unrolled iterations. The k-space \
                      dipole data-consistency step runs in Rust (rustfft); the \
                      learned proximal CNN runs via ONNX.",
        paper: "Lai et al., MICCAI 2020; doi:10.1007/978-3-030-59713-9_28",
        source: "https://github.com/Sulam-Group/LPCNN",
        license: "Author-permitted",
        files: &[pending_onnx("lpcnn-prox.onnx")],
        inputs: &["x", "aht_b"],
        outputs: &["chi"],
        size_divisor: 8,
    },
    ModelSpec {
        id: "ir2qsm",
        name: "IR2QSM",
        stage: ModelStage::DipoleInversion,
        status: WeightStatus::Pending,
        origin: Framework::PyTorch,
        description: "Iterated U-Net with reverse concatenations, 4 fixed unrolled \
                      iterations. Inference-time AddNoise is removed for export.",
        paper: "Feng et al., 2024; arXiv:2406.12300",
        source: "https://github.com/sunhongfu/deepMRI",
        license: "Author-permitted (Sun group)",
        files: &[pending_onnx("ir2qsm.onnx")],
        inputs: &["field"],
        outputs: &["chi"],
        size_divisor: 8,
    },
    // ---- Single-step (phase/total field → χ) -------------------------------
    ModelSpec {
        id: "autoqsm",
        name: "AutoQSM",
        stage: ModelStage::SingleStep,
        status: WeightStatus::Available,
        origin: Framework::TensorFlow,
        description: "V-Net single-step reconstruction (total field → susceptibility) \
                      with no separate brain extraction. Fixed 64³→32³ patch net; \
                      the Rust glue does the sliding-window tiling + blend.",
        paper: "Wei et al., NeuroImage 2019; doi:10.1016/j.neuroimage.2019.116064",
        source: "https://github.com/AMRI-Lab/AutoQSM",
        license: "Author-permitted (per project agreement)",
        // Clean PyTorch re-export of the Keras V-Net (tract-friendly NCDHW);
        // see scripts/onnx-export/export_autoqsm.py. Hosted on OSF project erv6n.
        files: &[WeightFile {
            name: "autoqsm.onnx",
            url: "https://osf.io/download/6a855583bc038122f06a886f/",
            sha256: "c2f23c6ae735fd1677e8623da22cb6f25b4a932ae50ba1ef08a75439e240977e",
            bytes: 5_609_662,
        }],
        // Fixed-size patch net; `size_divisor` is not a whole-volume constraint here
        // (tiling handles padding), left at 1.
        inputs: &["field"],
        outputs: &["chi"],
        size_divisor: 1,
    },
    ModelSpec {
        id: "iqsm",
        name: "iQSM",
        stage: ModelStage::SingleStep,
        status: WeightStatus::Available,
        origin: Framework::PyTorch,
        description: "LoT-Unet single-step reconstruction from wrapped phase to \
                      susceptibility. The learnable-Laplacian front-end is fused into \
                      the exported graph; inputs are phase, mask, TE (s), B0 (T).",
        paper: "Gao et al., NeuroImage 2022; doi:10.1016/j.neuroimage.2022.119410",
        source: "https://github.com/sunhongfu/iQSM",
        license: "Author-permitted (Sun group)",
        files: &[WeightFile {
            name: "iqsm.onnx",
            url: "https://osf.io/download/6a8642664b84b0c1461c0a22/",
            sha256: "8538a33892a877812e6cd0e22927a5040827c126e9347073d3dd0934bf226b09",
            bytes: 17_233_763,
        }],
        inputs: &["phase", "mask", "te", "b0", "border"],
        outputs: &["chi"],
        size_divisor: 8,
    },
    ModelSpec {
        id: "iqsm-plus",
        name: "iQSM+",
        stage: ModelStage::SingleStep,
        status: WeightStatus::Available,
        origin: Framework::PyTorch,
        description: "iQSM with orientation-adaptive latent feature editing (OA-LFE); \
                      the B0 direction is a genuine network input. Inputs: phase, \
                      mask, TE, B0, z_prjs (B0 dir), border.",
        paper: "Gao et al., Med Image Anal 2024; doi:10.1016/j.media.2024.103160",
        source: "https://github.com/sunhongfu/iQSM_Plus",
        license: "Author-permitted (Sun group)",
        files: &[WeightFile {
            name: "iqsm-plus.onnx",
            url: "https://osf.io/download/6a864b736621f1b55b6c579c/",
            sha256: "f93a4c83804759a4a0d24603f7754efb997ddd2ba5e3a0346ed6493725cafb1d",
            bytes: 17_740_526,
        }],
        inputs: &["phase", "mask", "te", "b0", "z_prjs", "border"],
        outputs: &["chi"],
        size_divisor: 16,
    },
    ModelSpec {
        id: "nextqsm",
        name: "NeXtQSM",
        stage: ModelStage::SingleStep,
        status: WeightStatus::Available,
        origin: Framework::TensorFlow,
        description: "U-Net background removal followed by a 6-step variational-network \
                      dipole inversion (total field → susceptibility). Reimplemented as \
                      a Rust hybrid: two exported U-Nets (`nextqsm-bf` = BFR forward, \
                      `nextqsm-vjp` = the regularizer gradient ∇ₓ mean|VarNet(x)| written \
                      as a forward graph) plus the FFT data-consistency gradient and unroll \
                      in Rust. Order the files BFR-first, VJP-second.",
        paper: "Cognolato et al., NeuroImage 2023; doi:10.1016/j.neuroimage.2022.119729",
        source: "https://github.com/wayne1123/NeXtQSM",
        license: "MIT",
        files: &[
            WeightFile {
                name: "nextqsm-bf.onnx",
                url: "https://osf.io/download/6a8677348d4da2a8c67e7de7/",
                sha256: "d0f0b0391153f8e5e07ef85d46fbe492afc8394fab008350540a6868d0b4a3ba",
                bytes: 113_211_474,
            },
            WeightFile {
                name: "nextqsm-vjp.onnx",
                url: "https://osf.io/download/6a867742f509e68e077e7d89/",
                sha256: "50fe388663c05b1d3996139975270e500e13feb025f1b792576dbc0da814a4bc",
                bytes: 42_437_407,
            },
        ],
        inputs: &["field", "mask", "b_vec"],
        outputs: &["chi"],
        size_divisor: 64,
    },
    ModelSpec {
        id: "modl-qsm",
        name: "MoDL-QSM",
        stage: ModelStage::DipoleInversion,
        status: WeightStatus::Pending,
        origin: Framework::TensorFlow,
        description: "Model-based deep learning: fixed-depth unroll of a QSM \
                      forward-model gradient descent with a learned CNN prior.",
        paper: "Feng et al., NeuroImage 2021; doi:10.1016/j.neuroimage.2021.117876",
        source: "https://github.com/AMRI-Lab/MoDL-QSM",
        license: "Author-permitted (per project agreement)",
        files: &[pending_onnx("modl-qsm.onnx")],
        inputs: &["field"],
        outputs: &["chi"],
        size_divisor: 16,
    },
    // ---- χ-separation (χ+ / χ−) --------------------------------------------
    ModelSpec {
        id: "susep-net",
        name: "SUSEP-Net",
        stage: ModelStage::ChiSeparation,
        status: WeightStatus::Available,
        origin: Framework::PyTorch,
        description: "Dual-branch 3D U-Net source separation from z-scored \
                      [QSM, R2', local field] → [χ+, χ−]. Normalization constants \
                      are baked into the Rust glue (SusepNetNorm).",
        paper: "Li, Gao, Sun et al., arXiv:2506.13293 (2025)",
        source: "https://github.com/YangGaoUQ/SUSEP-Net",
        license: "Academic use; author-permitted",
        files: &[WeightFile {
            name: "susep-net.onnx",
            url: "https://osf.io/download/6a8552a5329b2090036a8920/",
            sha256: "9bc85aa28451fda2c661de4b8fae8604ced7a54f028f94ed33629f6111a48b03",
            bytes: 205_743_584,
        }],
        inputs: &["qsm", "r2prime", "lfs"],
        outputs: &["chi_pos", "chi_neg"],
        size_divisor: 8,
    },
    ModelSpec {
        id: "chi-sepnet",
        name: "χ-sepnet",
        stage: ModelStage::ChiSeparation,
        status: WeightStatus::Pending,
        origin: Framework::Onnx,
        description: "SNU-LIST χ-separation network. Already ONNX. Weights are \
                      gated (academic Google form) and cannot be redistributed — \
                      supply via $QSM_MODEL_DIR (bring-your-own-weights).",
        paper: "Kim et al. / SNU-LIST chi-separation toolbox",
        source: "https://github.com/SNU-LIST/chi-separation",
        license: "SNU-LIST academic, gated — NOT redistributable",
        files: &[pending_onnx("chi-sepnet.onnx"), pending_onnx("chi-sepnet-norm.json")],
        inputs: &["local_field", "qsm", "r2prime"],
        outputs: &["chi_pos", "chi_neg"],
        size_divisor: 8,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_unique_and_findable() {
        let mut seen = std::collections::HashSet::new();
        for m in MODELS {
            assert!(seen.insert(m.id), "duplicate model id: {}", m.id);
            assert!(find_model(m.id).is_some(), "find_model failed for {}", m.id);
            assert!(!m.name.is_empty(), "{} has empty name", m.id);
            assert!(!m.files.is_empty(), "{} has no weight files", m.id);
        }
        assert!(find_model("does-not-exist").is_none());
    }

    #[test]
    fn pending_models_are_not_available() {
        // A Pending model is not runnable via download: it must lack a URL on at
        // least one file (a known SHA-256 ahead of hosting is fine, e.g. BFRnet).
        for m in MODELS.iter().filter(|m| m.status == WeightStatus::Pending) {
            assert!(!m.is_available(), "{} is Pending but reports available", m.id);
            assert!(
                m.files.iter().any(|f| f.url.is_empty()),
                "{} is Pending but every file has a url — flip status to Available",
                m.id
            );
            // Any precomputed hash must still be a valid 64-char hex digest.
            for f in m.files.iter().filter(|f| !f.sha256.is_empty()) {
                assert_eq!(f.sha256.len(), 64, "{}/{} sha256 must be 64 hex chars", m.id, f.name);
                assert!(
                    f.sha256.bytes().all(|b| b.is_ascii_hexdigit()),
                    "{}/{} sha256 not hex",
                    m.id,
                    f.name
                );
            }
        }
    }

    #[test]
    fn available_models_are_fully_specified() {
        // Invariant: anything marked Available must have url + hex sha256 per file.
        for m in MODELS.iter().filter(|m| m.status == WeightStatus::Available) {
            for f in m.files {
                assert!(!f.url.is_empty(), "{}/{} available but no url", m.id, f.name);
                assert_eq!(f.sha256.len(), 64, "{}/{} sha256 must be 64 hex chars", m.id, f.name);
                assert!(
                    f.sha256.bytes().all(|b| b.is_ascii_hexdigit()),
                    "{}/{} sha256 not hex",
                    m.id,
                    f.name
                );
            }
            assert!(m.is_available());
        }
    }
}

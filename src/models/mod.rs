//! Deep-learning model registry and weight management.
//!
//! Several QSM stages have deep-learning implementations (BFRnet, QSMnet,
//! xQSM, χ-sepnet, …). QSM-Core runs them via **ONNX** rather than bundling a
//! Python/PyTorch/TensorFlow runtime. The trained weights are *not* vendored in
//! this crate — they are fetched on first use and cached on disk.
//!
//! This module is the single source of truth for *which* models exist and
//! *where/how* to obtain their weights. It is split into three layers:
//!
//! - **Registry (always compiled).** [`ModelSpec`] / [`WeightFile`] describe each
//!   model and its weight files (URL, SHA-256, size, license). [`all_models`] /
//!   [`find_model`] query the table. This layer has no heavy dependencies and is
//!   safe to expose through the WASM bindings so a JavaScript host (e.g. qsmbly)
//!   can discover download URLs and fetch weights itself.
//! - **Download & cache (`download` feature).** [`download::ensure_model`] fetches
//!   any missing weight files over HTTP, verifies their SHA-256, and stores them
//!   in a local cache. This is the path a native host (e.g. QSMxT) uses to
//!   "download on use". WASM hosts skip this layer.
//! - **Inference (`onnx` feature).** Runs an ONNX graph with the pure-Rust
//!   [`tract`](https://docs.rs/tract-onnx) engine. The entry points take the
//!   model as a **byte buffer** ([`onnx::OnnxModel::load`]) rather than a path,
//!   so the *same* inference code runs natively and in WASM. A native host feeds
//!   bytes from the download cache; a WASM host (e.g. qsmbly) fetches the weights
//!   in JavaScript and passes the bytes back into WASM to run.
//!
//! The registry stays runtime-agnostic: `tract` is the default portable engine,
//! but a host is free to read a model's URLs from the registry and run it any
//! other way.
//!
//! ### Weight resolution order (native)
//!
//! 1. `$QSM_MODEL_DIR/<file>` — an explicit directory of local weight files
//!    (bring-your-own-weights; also how gated models like χ-sepnet are supplied).
//! 2. The on-disk cache ([`cache_dir`]), if the file is present and its SHA-256
//!    matches.
//! 3. Download from [`WeightFile::url`] into the cache (`download` feature).

use std::path::PathBuf;

mod registry;
pub use registry::{all_models, find_model};

#[cfg(feature = "download")]
pub mod download;

#[cfg(feature = "onnx")]
pub mod onnx;

/// The reconstruction stage a model implements.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelStage {
    /// Wrapped/unwrapped phase → local tissue field (joint unwrap + background removal).
    PhaseToField,
    /// Total field → local tissue field (background field removal).
    BackgroundRemoval,
    /// Local tissue field → susceptibility (dipole inversion).
    DipoleInversion,
    /// Total field or phase → susceptibility in one network (single-step).
    SingleStep,
    /// Inputs → paramagnetic/diamagnetic susceptibility (χ+ / χ−).
    ChiSeparation,
}

/// The framework the weights were originally trained in (before ONNX export).
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Framework {
    /// Authored/exported directly as ONNX.
    Onnx,
    /// PyTorch `.pth`/`.pt` → exported with `torch.onnx.export`.
    PyTorch,
    /// TensorFlow/Keras → exported with `tf2onnx`.
    TensorFlow,
    /// MATLAB Deep Learning Toolbox → exported with `exportONNXNetwork`.
    Matlab,
}

/// Whether a model's weights are hosted and ready to run.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightStatus {
    /// Weights are converted, hosted at [`WeightFile::url`], and ready to fetch.
    Available,
    /// A recognized target whose ONNX weights are not yet converted/hosted.
    /// [`WeightFile::url`]/[`WeightFile::sha256`] may be empty. Listed so hosts
    /// can surface the roadmap and accept bring-your-own-weights via
    /// `$QSM_MODEL_DIR`.
    Pending,
}

/// One weight file a model needs at inference time (an exported `.onnx`, plus
/// any auxiliary file such as normalization statistics).
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug)]
pub struct WeightFile {
    /// Filename used as the local cache key. Also the name looked up under
    /// `$QSM_MODEL_DIR`.
    pub name: &'static str,
    /// Direct download URL (OSF/Hugging Face/…). Empty while [`WeightStatus::Pending`].
    pub url: &'static str,
    /// Lowercase hex SHA-256 of the file, for integrity and cache validation.
    /// Empty while [`WeightStatus::Pending`].
    pub sha256: &'static str,
    /// File size in bytes (0 if not yet known).
    pub bytes: u64,
}

/// A deep-learning QSM model and everything needed to obtain and run it.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug)]
pub struct ModelSpec {
    /// Stable identifier, e.g. `"bfrnet"`, `"qsmnet"`. Used by [`find_model`] and
    /// as the env-var suffix `$QSM_MODEL_DIR`.
    pub id: &'static str,
    /// Human-readable name, e.g. `"BFRnet"`.
    pub name: &'static str,
    /// Reconstruction stage this model implements.
    pub stage: ModelStage,
    /// Whether the weights are hosted and ready.
    pub status: WeightStatus,
    /// Framework the weights came from (before ONNX export).
    pub origin: Framework,
    /// One-line description of the method.
    pub description: &'static str,
    /// Reference (citation / DOI / arXiv).
    pub paper: &'static str,
    /// Upstream code repository.
    pub source: &'static str,
    /// License / redistribution note for the weights.
    pub license: &'static str,
    /// ONNX weight file(s) required at inference time.
    pub files: &'static [WeightFile],
    /// ONNX graph input tensor name(s), in the order the runner feeds them.
    pub inputs: &'static [&'static str],
    /// ONNX graph output tensor name(s).
    pub outputs: &'static [&'static str],
    /// Spatial dimensions must be zero-padded to a multiple of this before
    /// inference (`0`/`1` = no constraint). Fully-convolutional nets with
    /// pooling need e.g. 8 or 16.
    pub size_divisor: u32,
}

impl ModelSpec {
    /// `true` if every weight file has a non-empty URL and hash (i.e. hosted).
    pub fn is_available(&self) -> bool {
        self.status == WeightStatus::Available
            && self.files.iter().all(|f| !f.url.is_empty() && !f.sha256.is_empty())
    }
}

/// Root directory for cached weight files.
///
/// Resolved from, in order: `$QSM_MODEL_CACHE`, `$XDG_CACHE_HOME/qsm-rs/models`,
/// `$HOME/.cache/qsm-rs/models`, else `./.qsm-rs-models`.
pub fn cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("QSM_MODEL_CACHE") {
        return PathBuf::from(dir);
    }
    if let Ok(dir) = std::env::var("XDG_CACHE_HOME") {
        return PathBuf::from(dir).join("qsm-rs").join("models");
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".cache").join("qsm-rs").join("models");
    }
    PathBuf::from(".qsm-rs-models")
}

/// The path a weight file would occupy in the cache (whether or not it exists).
pub fn cache_path(file: &WeightFile) -> PathBuf {
    cache_dir().join(file.name)
}

/// Find an already-present copy of `file` without downloading, honoring the
/// `$QSM_MODEL_DIR` override first, then the cache. Returns `None` if absent.
///
/// This is std-only and works even without the `download` feature, so a host
/// can supply bring-your-own-weights for pending or gated models.
pub fn resolve_local(file: &WeightFile) -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("QSM_MODEL_DIR") {
        let p = PathBuf::from(dir).join(file.name);
        if p.is_file() {
            return Some(p);
        }
    }
    let p = cache_path(file);
    if p.is_file() {
        Some(p)
    } else {
        None
    }
}

/// Read the bytes of a model's primary (first) weight file for native inference.
///
/// Resolution: a locally-supplied file (`$QSM_MODEL_DIR` / cache) is used first;
/// otherwise, with the `download` feature, it is fetched and cached. Returns a
/// human-readable error if the file is neither local nor downloadable.
///
/// WASM hosts do not call this — they obtain the bytes in JavaScript (using the
/// registry's URLs) and pass them straight to [`onnx::OnnxModel::load`].
pub fn primary_weight_bytes(spec: &ModelSpec) -> Result<Vec<u8>, String> {
    let file = spec
        .files
        .first()
        .ok_or_else(|| format!("model '{}' has no weight files", spec.id))?;
    weight_file_bytes(spec.id, file)
}

/// Read every weight file of a model, in registry order, for native inference.
///
/// Multi-file models (e.g. NeXtQSM's BFR + VJP U-Nets) need all pieces; the order
/// matches [`ModelSpec::files`], which the per-model glue relies on. Same
/// local-first/download resolution as [`primary_weight_bytes`].
pub fn all_weight_bytes(spec: &ModelSpec) -> Result<Vec<Vec<u8>>, String> {
    if spec.files.is_empty() {
        return Err(format!("model '{}' has no weight files", spec.id));
    }
    spec.files.iter().map(|f| weight_file_bytes(spec.id, f)).collect()
}

/// Convenience: resolve a model **by id** and read its primary weight file. Combines
/// [`find_model`] + [`primary_weight_bytes`] so callers (pipeline runners, external
/// hosts) don't repeat the lookup+fetch boilerplate. Errors if the id is unknown.
pub fn primary_weight(id: &str) -> Result<Vec<u8>, String> {
    let spec = find_model(id).ok_or_else(|| format!("'{id}' not in model registry"))?;
    primary_weight_bytes(spec)
}

/// Convenience: resolve a model **by id** and read all its weight files, in registry
/// order (for multi-file models like NeXtQSM). See [`all_weight_bytes`].
pub fn weights(id: &str) -> Result<Vec<Vec<u8>>, String> {
    let spec = find_model(id).ok_or_else(|| format!("'{id}' not in model registry"))?;
    all_weight_bytes(spec)
}

/// Resolve one weight file to its bytes (local override / cache, then download).
fn weight_file_bytes(id: &str, file: &WeightFile) -> Result<Vec<u8>, String> {
    if let Some(path) = resolve_local(file) {
        return std::fs::read(&path).map_err(|e| format!("reading {}: {e}", path.display()));
    }

    #[cfg(feature = "download")]
    {
        download::ensure_file(id, file)
            .map_err(|e| e.to_string())
            .and_then(|p| std::fs::read(&p).map_err(|e| format!("reading {}: {e}", p.display())))
    }
    #[cfg(not(feature = "download"))]
    {
        Err(format!(
            "weights for '{id}' ('{}') not found locally. Set $QSM_MODEL_DIR to a directory \
             containing it, or build with the 'download' feature to fetch it.",
            file.name
        ))
    }
}

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
//!   in JavaScript and supplies them with [`provide_weights`], after which the same
//!   pipeline dispatchers run as natively.
//!
//! The registry stays runtime-agnostic: `tract` is the default portable engine,
//! but a host is free to read a model's URLs from the registry and run it any
//! other way.
//!
//! ### Weight resolution order
//!
//! Every pipeline dispatcher that runs a network ([`crate::pipeline`]: inversion, BFRnet,
//! iQFM, iQSM, HD-BET/RS2-Net masking, χ-separation, …) and every host that loads a model
//! through [`primary_weight`] / [`weights`] resolves each weight file in this order:
//!
//! 1. Bytes the host supplied with [`provide_weights`] (checked against the registry's
//!    SHA-256 when they were supplied). This is the only source on
//!    `wasm32-unknown-unknown`, where there is no file system or network: a browser host
//!    fetches the files itself and hands them over, and the same dispatchers run.
//! 2. `$QSM_MODEL_DIR/<file>` — an explicit directory of local weight files
//!    (bring-your-own-weights; also how gated models like χ-sepnet are supplied). Native only.
//! 3. The on-disk cache ([`cache_dir`]). Native only.
//! 4. Download from [`WeightFile::url`] into the cache (`download` feature). Native only.
//!
//! On `wasm32-unknown-unknown` steps 2–4 are not compiled; a file that was not provided is an
//! error naming the file and [`provide_weights`].
//!
//! ### Memory
//!
//! Every model in the registry carries [`ModelSpec::memory`], measured with tract, so a host can
//! estimate a run's memory before it starts:
//!
//! - [`ModelSpec::resident_bytes`]: the parsed model, held for as long as it is loaded.
//! - [`ModelSpec::inference_bytes`]: one inference at any input shape, above the resident model
//!   (native): a fixed part plus bytes per voxel of the network's input.
//! - [`ModelSpec::peak_bytes`]: loading the model and running it once, for a native process or a
//!   `wasm32` heap ([`Heap`]).
//! - [`crate::bet::hd_bet_memory_bytes`]: the whole HD-BET step on a scan, from its field of view.
//!   HD-BET resamples to 1 mm, so its network input follows the head's size in mm, not the
//!   acquired matrix ([`crate::bet::hd_bet_network_input`]).
//!
//! The bytes of the weight files themselves ([`WeightFile::bytes`]) are not included, nor are a
//! wrapper's own buffers or further tiles in flight (for those, see
//! [`ModelSpec::tile_bytes_per_voxel`]).
//!
//! **How they were measured.** Natively, `examples/inference_memory.rs` runs each network
//! single-threaded under a counting global allocator, at 1–8 input shapes per network from 32³ to
//! 192×256×128 (to 256³ for MindGrab and 176×256×256 for HD-BET; the fixed-patch networks at
//! their patch). Per shape it takes the larger of a run that plans inside
//! ([`onnx::OnnxModel::run`]) and a plan built once and then run ([`onnx::OnnxModel::plan_for`]),
//! since wrappers use both. The figures are the least-squares line through those peaks, with the
//! intercept raised to cover every one. From 64³ up they sit 0–7 % above the measured peaks
//! (0–1.3 % from 96³ up); at 32³ up to 28 % above (at most 25 MB), where the fixed part
//! dominates. MindGrab is fitted from 128³ up: it always runs at 256³, and smaller inputs fall
//! well below the line. Threads (the `parallel` feature) add 0.1–0.6 % at 8–16 threads. Peak
//! RSS adds the program, the weight file's bytes and the allocator's slack to these.
//!
//! In WASM, the `browser_memory::net_*` cases in `tests/wasm_pipeline.rs` run the same inferences
//! under Node (`wasm32-unknown-unknown`, SIMD128, single-threaded) at one or two shapes per
//! network that fit the 4 GiB heap, and read the size of the linear memory, which only grows. The
//! WASM figures are a line through those points. The WASM heap grows by 0.8–1.6 times the native
//! peak of the same run (median 1.2). The ratio varies with the network and the shape, because
//! the allocator cannot always reuse a freed tensor's block for the next, larger one. Two runs of
//! the same graph (QSMnet and QSMnet+) differed by 7 %, so budget a margin on top.
//!
//! `src/models/memory_calibration.rs` holds every measured peak, and its tests check that each
//! figure covers its measurements and sits within 10 % (or 32 MB) of them. Re-measure after a
//! tract upgrade or a new export. QSMnet+ shares QSMnet's figures and iQFM shares iQSM's, since
//! they are the same graphs. SynthSeg 2.0's weights are not hosted, so its figures are SynthSeg
//! 1.0's plus its 33rd output channel.

use std::collections::BTreeMap;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

mod registry;
pub use registry::{all_models, find_model};

#[cfg(test)]
mod memory_calibration;

#[cfg(all(
    feature = "download",
    not(all(target_arch = "wasm32", target_os = "unknown"))
))]
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
    /// R2* → R2′ (reversible relaxation), for GRE-only χ-separation.
    R2PrimeGeneration,
    /// Magnitude image → binary brain mask (brain extraction).
    BrainExtraction,
    /// Magnitude image → anatomical label map (whole-brain segmentation).
    Segmentation,
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
    /// Heap one tile of this network holds while it runs, in bytes per voxel of its padded
    /// patch, for the networks QSM-Core runs tiled (`None` for the rest).
    ///
    /// Covers the tile's activations and everything else its wrapper allocates per tile (the
    /// input and output patches, and for the unrolled networks their FFT buffers), as measured
    /// in WASM, fragmentation included: the slope of the heap's high-water against patch voxels,
    /// one tile in flight (the `browser_memory::tile_cost_*` cases in `tests/wasm_pipeline.rs`;
    /// `examples/tile_memory_footprint.rs` measures the same natively by counting allocations).
    /// What a run holds once whatever the number of tiles (the parsed model and its plan, the
    /// full-volume buffers) is not in it, except for the networks whose wrapper loads the network
    /// inside every patch (LPCNN, MoDL-QSM, NeXtQSM): each of their tiles in flight holds its own
    /// parsed model, so their figure is a whole 64³ tile's heap per voxel, which over-counts larger
    /// patches slightly.
    ///
    /// `inversion::tile_bytes` turns it into a tile's bytes for a config, and it is what
    /// `inversion::tile_concurrency` budgets with on WASM, so a host estimating a tiled run's
    /// memory should use this same figure.
    pub tile_bytes_per_voxel: Option<u64>,
    /// Heap the network takes in tract: the parsed model, and one inference as a function of its
    /// input shape. See [`ModelSpec::inference_bytes`] and the [module docs](self#memory).
    pub memory: NetworkMemory,
}

/// Measured heap of one network in tract (see the [module docs](self#memory) for how).
///
/// Bytes of heap allocated by tract and QSM-Core's ONNX layer. The bytes of the weight file itself
/// ([`WeightFile::bytes`], held by whoever supplied them) are not included.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NetworkMemory {
    /// Heap the parsed model ([`onnx::OnnxModel::load`]) holds for as long as it is loaded; for
    /// NeXtQSM, both its networks.
    pub resident_bytes: u64,
    /// High-water of parsing the model, from nothing: about twice the resident size, as tract
    /// holds the decoded protobuf while it builds the graph.
    pub load_peak_bytes: u64,
    /// Native: the fixed part of one inference's peak above the resident model.
    pub inference_fixed_bytes: u64,
    /// Native: the part of one inference's peak above the resident model that grows with the
    /// input, per voxel of the network's input (activations, the input and output tensors, the
    /// plan's per-shape tables and tract's scratch).
    pub inference_bytes_per_voxel: u64,
    /// WASM: the fixed part of the heap's growth over loading the model and running it once.
    pub wasm_fixed_bytes: u64,
    /// WASM: the part of that growth per voxel of the network's input.
    pub wasm_bytes_per_voxel: u64,
}

/// Which heap a memory figure is for.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Heap {
    /// A native process, measured as live allocations (the resident set adds the program and the
    /// allocator's slack).
    Native,
    /// A `wasm32` linear memory, measured as its size, which only grows: fragmentation included.
    Wasm32,
}

/// What one tile of a tiled network costs, for budgeting how many fit in memory at once.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileCost {
    /// The network's size divisor: a tile's patch is `core + 2·halo` rounded up to a multiple of
    /// it ([`ModelSpec::size_divisor`]).
    pub divisor: usize,
    /// Heap per patch voxel ([`ModelSpec::tile_bytes_per_voxel`]).
    pub bytes_per_voxel: u64,
}

impl ModelSpec {
    /// The cost of one tile of this network, for the networks QSM-Core runs tiled. The tiled
    /// wrappers budget with it, and a host estimating a tiled run's memory should use it too
    /// (`inversion::tile_bytes`, `inversion::wasm_tiles_in_flight`).
    pub fn tile_cost(&self) -> Option<TileCost> {
        self.tile_bytes_per_voxel
            .map(|bytes_per_voxel| TileCost { divisor: self.size_divisor.max(1) as usize, bytes_per_voxel })
    }

    /// Native peak heap of one inference of this network at input spatial shape `shape`, above
    /// the resident model ([`Self::resident_bytes`]), in bytes:
    /// [`NetworkMemory::inference_fixed_bytes`] + voxels · [`NetworkMemory::inference_bytes_per_voxel`].
    ///
    /// `shape` is the spatial shape of the tensor the network itself sees: a tile's or window's
    /// patch, or for a whole-volume run the volume after the wrapper's padding (to
    /// [`Self::size_divisor`]; for HD-BET to multiples of 16×32×32 in its `(z, y, x)` order, see
    /// [`crate::bet::hd_bet_network_input`]). Only its number of voxels matters. Pass the spatial
    /// shape even for networks with several input channels (χ-separation, MoDL-QSM) or inputs
    /// (iQSM): those are counted. Fixed-shape networks (RS2-Net, AutoQSM, QSMGAN) always see their
    /// patch; pass that.
    ///
    /// It covers the plan tract builds for the shape (its copy of the weights and per-shape
    /// tables), the input and output tensors, the activations and tract's scratch, both for a plan
    /// built per call ([`onnx::OnnxModel::run`]) and for one built once and reused
    /// ([`onnx::OnnxModel::plan_for`]). A wrapper's own buffers (the padded volume, overlap
    /// accumulators, test-time flips, pre- and post-processing) come on top, as do further tiles
    /// in flight. Measured single-threaded: with the `parallel` feature, 8–16 threads added
    /// 1–13 MB (0.1–0.6 %) for HD-BET, SynthSeg and xQSM.
    pub fn inference_bytes(&self, shape: [usize; 3]) -> u64 {
        let m = &self.memory;
        m.inference_fixed_bytes + voxels(shape) * m.inference_bytes_per_voxel
    }

    /// Heap the parsed model holds while loaded, in bytes ([`NetworkMemory::resident_bytes`]).
    pub fn resident_bytes(&self) -> u64 {
        self.memory.resident_bytes
    }

    /// Peak heap of loading this network and running it once at input spatial shape `shape`
    /// (as for [`Self::inference_bytes`]), in bytes: what one inference needs besides the
    /// caller's own buffers and the weight file's bytes.
    ///
    /// - [`Heap::Native`]: the larger of parsing the model ([`NetworkMemory::load_peak_bytes`])
    ///   and the resident model plus [`Self::inference_bytes`].
    /// - [`Heap::Wasm32`]: the growth of a `wasm32` linear memory over the same,
    ///   [`NetworkMemory::wasm_fixed_bytes`] + voxels · [`NetworkMemory::wasm_bytes_per_voxel`]
    ///   (at least the load peak). Between 0.8 and 1.6 times the native figure, depending on the
    ///   network and the shape: the WASM allocator cannot reuse every freed block for the next,
    ///   larger tensor.
    pub fn peak_bytes(&self, shape: [usize; 3], heap: Heap) -> u64 {
        let m = &self.memory;
        let after_load = match heap {
            Heap::Native => m.resident_bytes + self.inference_bytes(shape),
            Heap::Wasm32 => m.wasm_fixed_bytes + voxels(shape) * m.wasm_bytes_per_voxel,
        };
        m.load_peak_bytes.max(after_load)
    }

    /// `true` if every weight file has a non-empty URL and hash (i.e. hosted).
    pub fn is_available(&self) -> bool {
        self.status == WeightStatus::Available
            && self
                .files
                .iter()
                .all(|f| !f.url.is_empty() && !f.sha256.is_empty())
    }
}

fn voxels(shape: [usize; 3]) -> u64 {
    shape.iter().map(|&d| d as u64).product()
}

/// The bytes of one weight file, shared rather than copied.
///
/// Returned by [`primary_weight`] / [`weights`] and friends. Bytes supplied with
/// [`provide_weights`] are handed out as further references to the same buffer, so a run does
/// not hold a second copy of a large model while the provider keeps it. Dereferences to
/// `[u8]`, so `&weights` passes wherever `&[u8]` is expected.
#[derive(Clone)]
pub struct WeightBytes(Arc<Vec<u8>>);

impl WeightBytes {
    fn new(bytes: Vec<u8>) -> Self {
        Self(Arc::new(bytes))
    }
}

impl From<Vec<u8>> for WeightBytes {
    fn from(bytes: Vec<u8>) -> Self {
        Self::new(bytes)
    }
}

impl std::ops::Deref for WeightBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.0
    }
}

impl AsRef<[u8]> for WeightBytes {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl std::fmt::Debug for WeightBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WeightBytes({} bytes)", self.0.len())
    }
}

/// Why [`provide_weights`] refused a buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProvideWeightsError {
    /// No model in the registry has a weight file of this name.
    UnknownFile(String),
    /// The bytes do not hash to the registry's SHA-256 for this file (a wrong, truncated or
    /// corrupted download).
    ChecksumMismatch {
        file: String,
        expected: String,
        got: String,
    },
}

impl std::fmt::Display for ProvideWeightsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownFile(name) => write!(
                f,
                "'{name}' is not a weight file of any model in the registry"
            ),
            Self::ChecksumMismatch {
                file,
                expected,
                got,
            } => write!(
                f,
                "weights supplied for '{file}' do not match the registry: SHA-256 is {got}, \
                 expected {expected} (wrong, truncated or corrupted file)"
            ),
        }
    }
}

impl std::error::Error for ProvideWeightsError {}

/// Weight files supplied from memory, keyed by the registry's file name.
static PROVIDED: Mutex<BTreeMap<&'static str, WeightBytes>> = Mutex::new(BTreeMap::new());

fn provided_map() -> std::sync::MutexGuard<'static, BTreeMap<&'static str, WeightBytes>> {
    // Poisoning only means another thread panicked while holding the lock; the map itself is
    // always in a consistent state (single inserts/removes), so carry on with it.
    PROVIDED.lock().unwrap_or_else(|e| e.into_inner())
}

/// Look a weight file up by name across every model in the registry.
pub fn find_weight_file(file_name: &str) -> Option<&'static WeightFile> {
    all_models()
        .iter()
        .flat_map(|m| m.files.iter())
        .find(|f| f.name == file_name)
}

/// Lowercase hex SHA-256 of a byte slice.
pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write as _;
        let _ = write!(out, "{b:02x}");
    }
    out
}

/// Supply a weight file from memory, so every dispatcher that needs it uses these bytes before
/// looking in `$QSM_MODEL_DIR`, the cache or the network.
///
/// `file_name` is the registry's [`WeightFile::name`] (e.g. `"xqsm.onnx"`; for a multi-file
/// model such as NeXtQSM, supply each of its [`ModelSpec::files`]). The bytes are checked
/// against the registry's SHA-256 and refused on a mismatch; a file whose model is
/// [`WeightStatus::Pending`] has no hash yet and is taken as given, as `$QSM_MODEL_DIR` is.
///
/// This is how a host without a file system (a browser, `wasm32-unknown-unknown`) runs the
/// deep-learning stages: fetch each file the run needs, call this, then call the pipeline
/// functions as usual. The buffer is moved in, not copied, and stays until
/// [`forget_weights`] (or until replaced by another call for the same file); runs borrow it
/// without copying. Supplying the same file again replaces the earlier buffer.
pub fn provide_weights(file_name: &str, bytes: Vec<u8>) -> Result<(), ProvideWeightsError> {
    let file = find_weight_file(file_name)
        .ok_or_else(|| ProvideWeightsError::UnknownFile(file_name.to_string()))?;
    if !file.sha256.is_empty() {
        let got = sha256_hex(&bytes);
        if got != file.sha256 {
            return Err(ProvideWeightsError::ChecksumMismatch {
                file: file.name.to_string(),
                expected: file.sha256.to_string(),
                got,
            });
        }
    }
    provided_map().insert(file.name, WeightBytes::new(bytes));
    Ok(())
}

/// Drop a buffer supplied with [`provide_weights`], releasing its memory once no running stage
/// still holds it. Returns `true` if one was held.
pub fn forget_weights(file_name: &str) -> bool {
    provided_map().remove(file_name).is_some()
}

/// Drop every buffer supplied with [`provide_weights`].
pub fn forget_all_weights() {
    provided_map().clear();
}

/// The bytes supplied for `file_name` with [`provide_weights`], if any.
pub fn provided_weights(file_name: &str) -> Option<WeightBytes> {
    provided_map().get(file_name).cloned()
}

/// `true` if `file` can be resolved without a download: it was supplied with
/// [`provide_weights`], or (native only) it is in `$QSM_MODEL_DIR` or the cache.
pub fn weight_file_present(file: &WeightFile) -> bool {
    if provided_map().contains_key(file.name) {
        return true;
    }
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    {
        resolve_local(file).is_some()
    }
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    {
        false
    }
}

/// Root directory for cached weight files (native only).
///
/// Resolved from, in order: `$QSM_MODEL_CACHE`, `$XDG_CACHE_HOME/qsm-rs/models`,
/// `$HOME/.cache/qsm-rs/models`, else `./.qsm-rs-models`.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub fn cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("QSM_MODEL_CACHE") {
        return PathBuf::from(dir);
    }
    if let Ok(dir) = std::env::var("XDG_CACHE_HOME") {
        return PathBuf::from(dir).join("qsm-rs").join("models");
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home)
            .join(".cache")
            .join("qsm-rs")
            .join("models");
    }
    PathBuf::from(".qsm-rs-models")
}

/// The path a weight file would occupy in the cache (whether or not it exists). Native only.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub fn cache_path(file: &WeightFile) -> PathBuf {
    cache_dir().join(file.name)
}

/// Find an already-present copy of `file` on disk without downloading, honoring the
/// `$QSM_MODEL_DIR` override first, then the cache. Returns `None` if absent. Native only;
/// does not see bytes supplied with [`provide_weights`] (see [`weight_file_present`]).
///
/// This is std-only and works even without the `download` feature, so a host
/// can supply bring-your-own-weights for pending or gated models.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
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

/// The bytes of a model's primary (first) weight file.
///
/// Resolved as described in the [module docs](self): supplied bytes ([`provide_weights`])
/// first, then (native) `$QSM_MODEL_DIR`, the cache, and a download with the `download`
/// feature. Returns a human-readable error if the file is in none of them.
pub fn primary_weight_bytes(spec: &ModelSpec) -> Result<WeightBytes, String> {
    let file = spec
        .files
        .first()
        .ok_or_else(|| format!("model '{}' has no weight files", spec.id))?;
    weight_file_bytes(spec.id, file)
}

/// The bytes of every weight file of a model, in registry order.
///
/// Multi-file models (e.g. NeXtQSM's BFR + VJP U-Nets) need all pieces; the order
/// matches [`ModelSpec::files`], which the per-model glue relies on. Same
/// resolution as [`primary_weight_bytes`], per file.
pub fn all_weight_bytes(spec: &ModelSpec) -> Result<Vec<WeightBytes>, String> {
    if spec.files.is_empty() {
        return Err(format!("model '{}' has no weight files", spec.id));
    }
    spec.files
        .iter()
        .map(|f| weight_file_bytes(spec.id, f))
        .collect()
}

/// Convenience: resolve a model **by id** and read its primary weight file. Combines
/// [`find_model`] + [`primary_weight_bytes`] so callers (pipeline runners, external
/// hosts) don't repeat the lookup+fetch boilerplate. Errors if the id is unknown.
pub fn primary_weight(id: &str) -> Result<WeightBytes, String> {
    let spec = find_model(id).ok_or_else(|| format!("'{id}' not in model registry"))?;
    primary_weight_bytes(spec)
}

/// Convenience: resolve a model **by id** and read all its weight files, in registry
/// order (for multi-file models like NeXtQSM). See [`all_weight_bytes`].
pub fn weights(id: &str) -> Result<Vec<WeightBytes>, String> {
    let spec = find_model(id).ok_or_else(|| format!("'{id}' not in model registry"))?;
    all_weight_bytes(spec)
}

/// Pre-fetch (download + cache) all of a model's weight files, reporting per-file
/// progress via `on_progress(file_name, downloaded_bytes, total_bytes)`.
///
/// Intended for hosts (e.g. QSMxT) that want to show a download bar before running
/// inference: call this first, then the normal `primary_weight`/`weights` path finds the
/// files and does no further network I/O. Files already present (supplied with
/// [`provide_weights`], in `$QSM_MODEL_DIR` or the cache) are skipped silently. Unknown `id`
/// (a non-model / classical algorithm) is a no-op — returns `Ok(())` so callers can invoke it
/// unconditionally. On `wasm32-unknown-unknown` a file that was not supplied is an error.
pub fn prefetch_with_progress(
    id: &str,
    #[allow(unused_variables)] on_progress: &mut dyn FnMut(&str, u64, u64),
) -> Result<(), String> {
    let Some(spec) = find_model(id) else {
        return Ok(());
    };
    for file in spec.files {
        if weight_file_present(file) {
            continue;
        }
        #[cfg(all(
            feature = "download",
            not(all(target_arch = "wasm32", target_os = "unknown"))
        ))]
        {
            download::ensure_file_with_progress(id, file, &mut |done, total| {
                on_progress(file.name, done, total)
            })
            .map_err(|e| e.to_string())?;
        }
        #[cfg(not(all(
            feature = "download",
            not(all(target_arch = "wasm32", target_os = "unknown"))
        )))]
        {
            return Err(missing_weights_message(id, file, HOST_FS));
        }
    }
    Ok(())
}

/// Whether this build can reach a file system and network for weights (`$QSM_MODEL_DIR`, the
/// cache, a download). `false` on `wasm32-unknown-unknown`, where [`provide_weights`] is the
/// only source.
#[cfg_attr(
    all(
        feature = "download",
        not(all(target_arch = "wasm32", target_os = "unknown"))
    ),
    allow(dead_code)
)]
const HOST_FS: bool = !cfg!(all(target_arch = "wasm32", target_os = "unknown"));

/// The error for a weight file that is nowhere this build can look.
#[cfg_attr(
    all(
        feature = "download",
        not(all(target_arch = "wasm32", target_os = "unknown"))
    ),
    allow(dead_code)
)]
fn missing_weights_message(id: &str, file: &WeightFile, host_fs: bool) -> String {
    if !host_fs {
        format!(
            "weights for '{id}' ('{}') were not provided. This build has no file system or \
             network: fetch the file (registry URL {}) and pass its bytes to \
             qsm_core::models::provide_weights(\"{}\", bytes) before running.",
            file.name,
            if file.url.is_empty() {
                "<not hosted>"
            } else {
                file.url
            },
            file.name
        )
    } else {
        format!(
            "weights for '{id}' ('{}') not found. Supply them with \
             qsm_core::models::provide_weights, set $QSM_MODEL_DIR to a directory containing \
             the file, or build with the 'download' feature to fetch it.",
            file.name
        )
    }
}

/// Resolve one weight file to its bytes: supplied bytes, then (native) local override / cache,
/// then download.
fn weight_file_bytes(id: &str, file: &WeightFile) -> Result<WeightBytes, String> {
    if let Some(bytes) = provided_weights(file.name) {
        return Ok(bytes);
    }
    host_weight_bytes(id, file)
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
fn host_weight_bytes(id: &str, file: &WeightFile) -> Result<WeightBytes, String> {
    Err(missing_weights_message(id, file, HOST_FS))
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
fn host_weight_bytes(id: &str, file: &WeightFile) -> Result<WeightBytes, String> {
    let read = |p: &std::path::Path| {
        std::fs::read(p)
            .map(WeightBytes::new)
            .map_err(|e| format!("reading {}: {e}", p.display()))
    };
    if let Some(path) = resolve_local(file) {
        return read(&path);
    }

    #[cfg(feature = "download")]
    {
        download::ensure_file(id, file)
            .map_err(|e| e.to_string())
            .and_then(|p| read(&p))
    }
    #[cfg(not(feature = "download"))]
    {
        Err(missing_weights_message(id, file, HOST_FS))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The provider and `$QSM_MODEL_DIR` are process-wide; tests that touch them take this.
    static LOCK: Mutex<()> = Mutex::new(());

    fn lock() -> std::sync::MutexGuard<'static, ()> {
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn sha256_matches_known_vector() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn weight_file_names_are_unique() {
        // The provider (like `$QSM_MODEL_DIR` and the cache) is keyed by file name.
        let mut names: Vec<_> = all_models()
            .iter()
            .flat_map(|m| m.files.iter().map(|f| f.name))
            .collect();
        let n = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(
            names.len(),
            n,
            "duplicate weight file names in the registry"
        );
    }

    #[test]
    fn unknown_file_is_refused() {
        let _g = lock();
        assert_eq!(
            provide_weights("no-such-model.onnx", vec![1, 2, 3]),
            Err(ProvideWeightsError::UnknownFile(
                "no-such-model.onnx".into()
            ))
        );
    }

    #[test]
    fn checksum_mismatch_is_refused_and_not_stored() {
        let _g = lock();
        forget_weights("xqsm.onnx");
        let err = provide_weights("xqsm.onnx", b"not the xqsm weights".to_vec()).unwrap_err();
        match &err {
            ProvideWeightsError::ChecksumMismatch {
                file,
                expected,
                got,
            } => {
                assert_eq!(file, "xqsm.onnx");
                assert_eq!(expected, find_weight_file("xqsm.onnx").unwrap().sha256);
                assert_eq!(got, &sha256_hex(b"not the xqsm weights"));
            }
            other => panic!("expected a checksum mismatch, got {other:?}"),
        }
        assert!(err.to_string().contains("xqsm.onnx"));
        assert!(provided_weights("xqsm.onnx").is_none());
    }

    #[test]
    fn provided_bytes_take_precedence_over_model_dir() {
        let _g = lock();
        let dir = std::env::temp_dir().join(format!("qsm-provider-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // A pending model has no registry hash, so any bytes are accepted (as from $QSM_MODEL_DIR).
        let name = "synthseg-2.0.onnx";
        std::fs::write(dir.join(name), b"from the model dir").unwrap();
        let old = std::env::var_os("QSM_MODEL_DIR");
        std::env::set_var("QSM_MODEL_DIR", &dir);

        assert_eq!(
            &*primary_weight("synthseg-2.0").unwrap(),
            b"from the model dir"
        );
        assert!(weight_file_present(find_weight_file(name).unwrap()));

        let supplied = b"supplied in memory".to_vec();
        let ptr = supplied.as_ptr();
        provide_weights(name, supplied).unwrap();
        let got = primary_weight("synthseg-2.0").unwrap();
        assert_eq!(&*got, b"supplied in memory");
        // Handed out by reference: the run sees the host's buffer, not a copy.
        assert_eq!(got.as_ptr(), ptr);
        assert_eq!(weights("synthseg-2.0").unwrap()[0].as_ptr(), ptr);

        assert!(forget_weights(name));
        assert!(!forget_weights(name));
        assert_eq!(
            &*primary_weight("synthseg-2.0").unwrap(),
            b"from the model dir"
        );

        match old {
            Some(v) => std::env::set_var("QSM_MODEL_DIR", v),
            None => std::env::remove_var("QSM_MODEL_DIR"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn prefetch_skips_provided_files() {
        let _g = lock();
        provide_weights("synthseg-2.0.onnx", b"x".to_vec()).unwrap();
        // Pending (no URL): without the provided bytes this would have to fail.
        prefetch_with_progress("synthseg-2.0", &mut |_, _, _| panic!("nothing to fetch")).unwrap();
        forget_weights("synthseg-2.0.onnx");
    }

    #[test]
    fn missing_weights_without_a_file_system_names_the_provider() {
        // The message `wasm32-unknown-unknown` builds return when a file was not provided.
        let spec = find_model("xqsm").unwrap();
        let msg = missing_weights_message(spec.id, &spec.files[0], false);
        assert!(msg.contains("xqsm.onnx"), "{msg}");
        assert!(msg.contains("provide_weights"), "{msg}");
        assert!(msg.contains(spec.files[0].url), "{msg}");
        assert!(!msg.contains("QSM_MODEL_DIR"), "{msg}");
    }
}

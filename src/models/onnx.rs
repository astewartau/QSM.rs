//! ONNX inference via the pure-Rust [`tract`](https://docs.rs/tract-onnx) engine
//! (`onnx` feature).
//!
//! The model is loaded from a **byte buffer**, never a path, so the identical
//! code path runs natively (bytes from the [`super::download`] cache) and in WASM
//! (bytes fetched by JavaScript and handed back in). All tensors are `f32`
//! (NCDHW for the volumetric nets); callers convert to/from the crate's `f64`
//! volumes and handle model-specific normalization and padding.
//!
//! With the `parallel` feature on a native target, each inference spreads tract's matrix kernels
//! over a shared thread pool (calls from rayon workers — the tiled drivers — and WASM builds stay
//! single-threaded).
//!
//! ```no_run
//! # #[cfg(feature = "onnx")] {
//! use qsm_core::models::onnx::{OnnxModel, Tensor};
//! # fn go(model_bytes: &[u8], field: Vec<f32>, d: usize, h: usize, w: usize) -> Result<(), Box<dyn std::error::Error>> {
//! let model = OnnxModel::load(model_bytes)?;
//! let out = model.run_single(&Tensor::new(vec![1, 1, d, h, w], field))?;
//! # let _ = out; Ok(()) }
//! # }
//! ```

use tract_onnx::prelude::*;

/// A dense `f32` tensor: row-major `data` interpreted with `shape`.
#[derive(Clone, Debug)]
pub struct Tensor {
    /// Dimensions, e.g. `[1, 1, D, H, W]` for a single-channel volume.
    pub shape: Vec<usize>,
    /// Row-major values, length = product of `shape`.
    pub data: Vec<f32>,
}

impl Tensor {
    /// Construct a tensor, panicking if `data.len()` disagrees with `shape`.
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(n, data.len(), "tensor shape {shape:?} does not match data len {}", data.len());
        Self { shape, data }
    }
}

/// Error from loading or running an ONNX model.
#[derive(Debug)]
pub enum OnnxError {
    /// Failed to parse/optimize the ONNX graph.
    Load(String),
    /// Input shape or dtype could not be reconciled with the graph.
    Shape(String),
    /// Failure during the forward pass.
    Run(String),
}

impl std::fmt::Display for OnnxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Load(m) => write!(f, "onnx load error: {m}"),
            Self::Shape(m) => write!(f, "onnx shape error: {m}"),
            Self::Run(m) => write!(f, "onnx run error: {m}"),
        }
    }
}

impl std::error::Error for OnnxError {}

/// Run one inference, spreading tract's matrix kernels over a shared thread pool when the
/// `parallel` feature is on in a native build. Calls made from inside a rayon worker — the tiled
/// drivers, which already run one tile per thread — stay single-threaded to avoid
/// oversubscription. WASM always stays on the calling thread: tract's executor builds its own
/// rayon pool, which cannot spawn threads there (wasm threads come from wasm-bindgen-rayon).
fn run_threaded<R>(f: impl FnOnce() -> R) -> R {
    #[cfg(all(feature = "parallel", not(target_family = "wasm")))]
    {
        use std::sync::OnceLock;
        use tract_linalg::multithread::{multithread_tract_scope, Executor};
        static EXECUTOR: OnceLock<Option<Executor>> = OnceLock::new();
        if rayon::current_thread_index().is_none() {
            let exec = EXECUTOR.get_or_init(|| {
                let n = rayon::current_num_threads();
                (n > 1).then(|| Executor::multithread_with_name(n, "qsm-tract"))
            });
            if let Some(exec) = exec {
                return multithread_tract_scope(exec.clone(), f);
            }
        }
    }
    f()
}

/// A parsed ONNX model, ready to run at any spatial size.
///
/// The graph is kept in tract's shape-inferring form and specialized to the
/// concrete input shapes on each [`run`](OnnxModel::run) call, so one instance
/// serves volumes of different dimensions (fully-convolutional nets).
pub struct OnnxModel {
    model: InferenceModel,
}

impl OnnxModel {
    /// Parse an ONNX model from its serialized bytes.
    pub fn load(bytes: &[u8]) -> Result<Self, OnnxError> {
        let model = tract_onnx::onnx()
            .model_for_read(&mut std::io::Cursor::new(bytes))
            .map_err(|e| OnnxError::Load(e.to_string()))?;
        Ok(Self { model })
    }

    /// Run the model with `inputs` bound to graph inputs in order; returns every
    /// graph output as an `f32` [`Tensor`].
    pub fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, OnnxError> {
        let mut model = self.model.clone();
        for (i, inp) in inputs.iter().enumerate() {
            model
                .set_input_fact(i, f32::fact(inp.shape.as_slice()).into())
                .map_err(|e| OnnxError::Shape(e.to_string()))?;
        }
        // Prefer the optimized plan; if an optimization pass rejects the graph
        // (e.g. `PushSliceUp` on the backprop-as-forward graphs used by NeXtQSM),
        // fall back to the un-optimized typed plan, which still runs correctly.
        let plan = match model.clone().into_optimized().and_then(|m| m.into_runnable()) {
            Ok(p) => p,
            Err(_) => model
                .into_typed()
                .and_then(|m| m.into_runnable())
                .map_err(|e| OnnxError::Load(e.to_string()))?,
        };

        let mut feeds: TVec<TValue> = tvec!();
        for inp in inputs {
            let t = tract_onnx::prelude::Tensor::from_shape(&inp.shape, &inp.data)
                .map_err(|e| OnnxError::Shape(e.to_string()))?;
            feeds.push(t.into());
        }

        let result = run_threaded(|| plan.run(feeds)).map_err(|e| OnnxError::Run(e.to_string()))?;

        result
            .iter()
            .map(|t| {
                let view = t
                    .to_plain_array_view::<f32>()
                    .map_err(|e| OnnxError::Run(e.to_string()))?;
                Ok(Tensor {
                    shape: view.shape().to_vec(),
                    data: view.iter().copied().collect(),
                })
            })
            .collect()
    }

    /// Convenience for single-input / single-output nets.
    pub fn run_single(&self, input: &Tensor) -> Result<Tensor, OnnxError> {
        let mut out = self.run(std::slice::from_ref(input))?;
        if out.is_empty() {
            return Err(OnnxError::Run("model produced no outputs".into()));
        }
        Ok(out.swap_remove(0))
    }

    /// Compile a **reusable** execution plan specialized to fixed input shapes.
    ///
    /// [`run`](Self::run) re-clones and re-optimizes the whole graph on *every* call — fine for
    /// one-shot whole-volume inference, but wasteful when running many equal-shaped tensors
    /// (e.g. every patch of a tiled inversion). Build one [`OnnxPlan`] for the patch shape and
    /// reuse it: tract's optimizer (`into_optimized` — constant-folding conv weights, operator
    /// fusion, plan building) then runs exactly once instead of per patch.
    pub fn plan_for(&self, input_shapes: &[&[usize]]) -> Result<OnnxPlan, OnnxError> {
        let mut model = self.model.clone();
        for (i, shape) in input_shapes.iter().enumerate() {
            model
                .set_input_fact(i, f32::fact(*shape).into())
                .map_err(|e| OnnxError::Shape(e.to_string()))?;
        }
        // Same optimize-or-fallback strategy as `run`, but paid once here, not per call.
        let plan = match model.clone().into_optimized().and_then(|m| m.into_runnable()) {
            Ok(p) => p,
            Err(_) => model
                .into_typed()
                .and_then(|m| m.into_runnable())
                .map_err(|e| OnnxError::Load(e.to_string()))?,
        };
        Ok(OnnxPlan { plan })
    }
}

/// A compiled execution plan for fixed input shapes, built by [`OnnxModel::plan_for`]. Running
/// it skips graph optimization, so reusing one plan across many equal-shaped inputs (tiled
/// inference) amortizes the optimizer to a single up-front cost.
pub struct OnnxPlan {
    plan: std::sync::Arc<TypedRunnableModel>,
}

impl OnnxPlan {
    /// Run the plan with `inputs` bound to graph inputs in order (shapes must match the ones
    /// this plan was compiled for); returns every graph output as an `f32` [`Tensor`].
    pub fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, OnnxError> {
        let mut feeds: TVec<TValue> = tvec!();
        for inp in inputs {
            let t = tract_onnx::prelude::Tensor::from_shape(&inp.shape, &inp.data)
                .map_err(|e| OnnxError::Shape(e.to_string()))?;
            feeds.push(t.into());
        }
        let result = run_threaded(|| self.plan.run(feeds)).map_err(|e| OnnxError::Run(e.to_string()))?;
        result
            .iter()
            .map(|t| {
                let view = t
                    .to_plain_array_view::<f32>()
                    .map_err(|e| OnnxError::Run(e.to_string()))?;
                Ok(Tensor {
                    shape: view.shape().to_vec(),
                    data: view.iter().copied().collect(),
                })
            })
            .collect()
    }

    /// Convenience for single-input / single-output nets.
    pub fn run_single(&self, input: &Tensor) -> Result<Tensor, OnnxError> {
        let mut out = self.run(std::slice::from_ref(input))?;
        if out.is_empty() {
            return Err(OnnxError::Run("model produced no outputs".into()));
        }
        Ok(out.swap_remove(0))
    }
}

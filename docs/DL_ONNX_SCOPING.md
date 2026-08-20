# Scoping: native deep-learning QSM in QSM.rs via ONNX

Compiled 2026-08-19. Goal: run the deep-learning QSM methods from QSM-CI natively in Rust (no
Python/PyTorch/TF/MATLAB at inference time), by exporting each network to ONNX once, hosting the
weights somewhere stable (OSF / HF), and running them with an ONNX runtime bound into `qsm-core`.

**Verdict: feasible, but only for a well-defined subset.** Roughly half the DL methods are static
feed-forward networks that export cleanly to ONNX and run as a single forward pass — those are in
scope. The other half are *not* plain networks: they do per-subject optimisation (deep image prior /
implicit neural rep) or are locked to legacy TensorFlow 1.x graphs. Those are out of scope for an
"export-and-run" ONNX path.

---

## 0. Implementation status

The infrastructure (`src/models/`: registry + `download` + `onnx`/`tract`) and the offline export
tooling (`qsmci/scripts/onnx-export/`) are in place. Models are wired one at a time; each is
parity-checked against the authors' Python ONNX-Runtime output on `data/sim/dev`.

| Model | Stage | Wired | Parity (Rust tract vs Python ORT) | Weights |
|-------|-------|-------|-----------------------------------|---------|
| **BFRnet** | background removal | ✅ `bgremove::bfrnet` + `BgRemovalAlgorithm::Bfrnet` | corr 1.000000, max\|Δ\| 1.4e-7 ppm | ✅ OSF `Available` |
| **xQSM** | dipole inversion | ✅ `inversion::xqsm` + `InversionAlgorithm::Xqsm` | corr 1.000000, max\|Δ\| 3.2e-7 ppm | ✅ OSF `Available` |
| **QSMnet** | dipole inversion | ✅ `inversion::qsmnet` + `InversionAlgorithm::Qsmnet` | corr 1.000000, max\|Δ\| 1.7e-6 ppm | ✅ OSF `Available` |
| **QSMnet+** | dipole inversion | ✅ `InversionAlgorithm::QsmnetPlus` (`QsmnetNorm::qsmnet_plus`) | corr 1.000000 (clean rebuild) | ✅ OSF `Available` |
| **SUSEP-Net** | chi-separation | ✅ `separation::susep_net` + `SeparationAlgorithm::SusepNet` | tract-verified (3-in/2-out) | ✅ OSF `Available` |
| **AutoQSM** | single-step (total field→χ) | ✅ `inversion::autoqsm` + `InversionAlgorithm::Autoqsm` | vs Keras patch-stitch | ✅ OSF `Available` |
| **iQSM** | single-step (phase→χ) | ✅ `inversion::{iqsm,iqsm_multi_echo}` | vs original torch inference | ✅ OSF `Available` |
| **iQSM+** | single-step (phase→χ, orientation-adaptive) | ✅ `inversion::{iqsm_plus,iqsm_plus_multi_echo}` | vs original torch inference | ✅ OSF `Available` |

iQSM+ adds OA-LFE (B0 direction as a `z_prjs` input) and a brain-bbox-crop preprocessing. Two more
export workarounds: the OA-LFE builds an orientation-conditioned conv kernel per channel — torch.onnx
can't trace the dynamic kernel shape, so it's rewritten as `conv3d(x.sum(channels), K)` with a static
`(1,1,·)` kernel (identical since the same kernel hits every channel, batch=1); and the same LoT
`border`-input trick as iQSM. The bbox crop + paste-back lives in the Rust glue.

iQSM is the first **phase-input** net (4 inputs: phase, mask, scalar TE/B0). Two export gotchas
handled: the LoT layer's in-place boundary-zeroing (`out[:,:,[0,h-1]...]=0`) traced as *constant*
ScatterND indices → rewritten as shape-relative crop-1 + zero-pad-1 (identical result, dynamic);
and the sphere mask erosion + phase-sign flip + centre-pad + magnitude·TE² multi-echo combination
live in the Rust glue. Exposed as standalone functions (end-to-end phase→χ, like TGV), not an
`InversionAlgorithm` arm.

AutoQSM is the first **patch-tiled** net: a fixed 64³→32³ V-Net, with the sliding-window
tiling + 8-voxel linear blend (`util.data_predict`/`patch_process`) ported into the Rust glue.
Same clean-rebuild recipe (Keras V-Net → PyTorch → clean ONNX), no BatchNorm.

All hosted on OSF project **erv6n** (`https://osf.io/download/<id>/`), registry `status:
Available` with verified SHA-256; the native `download` path is end-to-end tested
(`tests/models_onnx.rs::osf_download_and_verify`). xQSM came from `export_xqsm.py` (torch↔ORT
max\|Δ\| ≈ 6e-5).

**Legacy-TF learning — tract can't run `tf2onnx` output.** QSMnet converts fine via `tf2onnx` and
runs in onnxruntime, but `tract` fails shape-analysis on the NHWC↔NCHW Reshape/Transpose and dynamic
deconv-shape ops `tf2onnx` emits (tried dynamic, static/onnxsim, `--inputs-as-nchw`). Rather than add
`ort` (a ~15–40 MB native-only C++ lib that kills the wasm story), the fix is to **rebuild the net in
PyTorch, port the TF checkpoint weights, and `torch.onnx.export`** → a clean NCDHW graph tract runs
(exactly like xQSM). `scripts/onnx-export/export_qsmnet.py` does this for QSMnet's plain U-Net
(validated torch-vs-TF max\|Δ\| = 1.4e-5). The same recipe extends to QSMnet+ (same arch) and, with
more effort, AutoQSM / MoDL-QSM / NeXtQSM. **Net effect: everything stays tract-only, wasm-capable,
and adds no heavy dependency.** Weights are dumped from the QSM-CI TF1.14 image
(`qsm-ci/qsmnet:v1`) and ported in a normal PyTorch env.

---

## 1. The three buckets

Of the 18 DL methods QSM-CI carries, they split cleanly:

### Bucket A — already ONNX (drop straight into a Rust runtime)

| Method | Stage | Weights | Size | Source | License gate |
|--------|-------|---------|------|--------|--------------|
| **BFRnet** | background removal | `BFRnet.onnx` | 76 MB | already exported from MATLAB `.mat`; committed in `algorithms/bfrnet/` | author permission (Sun) |
| **χ-sepnet** | chi-separation | `240904_xsepnet.onnx` + norm `.mat` | ~gated | SNU-LIST Google form | **gated, not redistributable** |

These are the fastest wins — no export step, we just need a Rust ONNX runtime. BFRnet is fully
unblocked (fully-convolutional, dynamic spatial dims, "186 std layers, no custom classes", matches
MATLAB `predict` to 1e-8). χ-sepnet is technically ready but its weights are behind an academic form
with no license, so **we can run it locally but cannot re-host the weights** (see §5).

### Bucket B — static feed-forward PyTorch (need a one-time `torch.onnx.export`)

All pure single-forward-pass nets. Export is mechanical: load `.pth` into the model class, run
`torch.onnx.export`, verify parity, done.

| Method | Stage | Weights | Size | Source | Notes |
|--------|-------|---------|------|--------|-------|
| **xQSM** | dipole | `xQSM_invivo.pth` + `Unet_invivo.pth` | ~34 MB | GitHub release v1.0-demo | octave-conv U-Net + learned residual |
| **iQSM** | phase→χ (single-step) | `iQSM_40_v2.pth` + `LoTLayer_chi_40_v2.pth` | ~32 MB | HF `sunhongfu/iQSM` | LoT-Unet; per-echo, mag·TE² weighting done outside net |
| **iQFM** | phase→localfield | `iQFM_40_v2.pth` + `LoTLayer_lfs_40_v2.pth` | ~32 MB | HF `sunhongfu/iQSM` | same net as iQSM, field head |
| **iQSM+** | phase→χ | `iQSM_plus.pth` + `LoTLayer_chi.pth` | ~40 MB | HF `sunhongfu/iQSM_Plus` | adds orientation-adaptive (OA-LFE) blocks; B0 dir is a real input |
| **SUSEP-Net** | chi-separation | `SUSEPNet.pth` + `all_mean_std.mat` | 197 MB | authors' Google Drive | dual-branch U-Net; z-scored [QSM,R2′,localfield] → [χ+,χ−] |
| **QSMGAN** | dipole | `WGAN_i64o48/net_best.pt` | ~30 MB | fork `mmorri10/QSMGAN-LupoLab` (**MIT**) | only the U-Net generator; legacy torch 1.1 checkpoint, load then re-save |

The LoT "Laplacian layer" checkpoints (iQSM/iQFM/iQSM+) are fixed convolutional front-ends — they
export as part of the graph or as a second small model. Preprocessing (per-echo split, magnitude·TE²
weighting, mask) stays in Rust; the net is a clean tensor-in/tensor-out.

### Bucket C — not an export-and-run target

| Method | Why it's out |
|--------|--------------|
| **DIP-UP** | test-time deep-image-prior loop (backprop per input) — ONNX is inference-only |
| **INR-QSM** | untrained SIREN, optimised per subject — no weights to export |
| **MoDIP** | untrained deep image prior, optimised per subject — no weights to export |
| **QSMnet**, **QSMnet+** | TensorFlow 1.14 checkpoints; tf2onnx path is brittle (contrib ops) |
| **AutoQSM** | TensorFlow 1.15 / Keras 2.2.5 legacy |
| **MoDL-QSM** | TF 1.15, unrolled with custom data-consistency Lambda layers |
| **NeXtQSM** | TF variational network + custom unrolled solver |

Two sub-notes:
- **DIP-UP / INR-QSM / MoDIP** are *architecturally* incompatible with ONNX — they solve an
  optimisation problem at inference. If we ever want these, the right move is to reimplement the
  optimisation loop natively in Rust against our existing FFT/dipole machinery (they're essentially
  "classical iterative solver with a CNN/MLP regulariser"), **not** ONNX. Sizeable effort each.
- **NeXtQSM** joins this group (assessed 2026-08-20). Its BFR is a plain U-Net, but its dipole
  inversion is a 6-step variational optimisation whose update is `x ← x − ∇ₓ(λ·E_D + E_R)` with
  `E_R = mean(|UNet(x)|)`, computed by **autodiff through the U-Net** each step. The crux —
  exporting `∇ₓ mean(|UNet(x)|)` to ONNX — was de-risked and **fails on every path**
  (`torch.func.grad`+legacy, `torch.func.grad`+dynamo, `autograd.grad`+legacy all error). Supporting
  it would need the U-Net's backward pass hand-written in Rust (a mini-autodiff) or a Rust autodiff
  runtime (candle/burn) — abandoning the tract/wasm/tiny-binary story. Deferred to native reimpl.
- **The TF-1.x nets (QSMnet/QSMnet+/AutoQSM/MoDL/NeXtQSM)** are *theoretically* convertible via
  `tf2onnx`, but TF1 graph-def conversion is where op-coverage breaks in practice. QSMnet is the
  highest-value of these (widely cited dipole net) — worth a **time-boxed spike** but not a
  commitment.

### Middle case — fixed-unroll nets (Bucket B-ish, with caveats)

| Method | Stage | Weights | Caveat |
|--------|-------|---------|--------|
| **IR2QSM** | dipole | `model_IR2Unet.pth` 40 MB (gdown) | 4× fixed unroll = static graph; **strip the inference-time `AddNoise()`** (no ONNX RNG) before export |
| **LPCNN** | dipole | `lpcnn_test_Bmodel.pkl` ~5 MB | 3× fixed unroll, but each step has an **FFT-based dipole data-consistency term** — either keep FFT in the graph (opset-17 `DFT`, patchy runtime support) or split the net and do the FFT step in Rust between sub-model calls |

LPCNN is the interesting one: its data-consistency step *is* our dipole convolution. Cleanest
implementation is a hybrid — run the learned proximal CNN via ONNX, do the k-space dipole step in
Rust with `rustfft` (which we already have), loop 3×. That reuses `kernels/dipole.rs` and avoids
ONNX FFT entirely.

---

## 2. Rust runtime choice — decided: `tract`

The deciding requirement is that **WASM must run inference inside QSM.rs on weights handed in from
JavaScript** (qsmbly fetches the `.onnx` bytes in JS, passes them back into the WASM module to run).
That rules out `ort` (Rust bindings to Microsoft's ONNX Runtime C++) as the portable path — it needs
a native shared library and doesn't compile cleanly to wasm. It points to **`tract`** (pure-Rust
ONNX engine, Sonos): one codebase compiles for native *and* wasm, and its API is byte-buffer based.

So the inference entry point takes **model bytes**, not a file path
([`models::onnx::OnnxModel::load(&[u8])`]). A native host feeds bytes from the download cache; a
wasm host feeds bytes from a JS `fetch`. `ort` stays a possible *native-only, faster* backend to add
behind a second feature later if op coverage or speed demands it, but it is not required.

**Verified (2026-08-19):** `tract` loads and runs the real 76 MB BFRnet ONNX (a MATLAB-exported
octave-convolution 3D U-Net) end-to-end, correct output shape, finite values — so the "op coverage"
worry that would have favoured `ort` did not materialise for at least the octave-conv U-Net family.
Re-verify per model at export time.

Toolchain note: `tract`'s dep tree pulls `kstring`, whose recent patch releases require rustc ≥1.96;
pinned to `kstring 2.0.0` in `Cargo.lock` for the local 1.95 toolchain. Harmless on newer rustc.

### WASM reality check

Perf is still the open question, not capability. Full-volume 3D U-Nets (76–197 MB weights, billions
of MACs) on a single-threaded wasm CPU backend may be slow (tens of seconds to minutes). That's a
UX/tiling problem for qsmbly to manage (patch-based inference, progress, worker threads), not a
blocker for the QSM.rs side. The `onnx` feature is off in default builds; hosts opt in.

---

## 3. Integration into QSM.rs

The dispatcher architecture already has the slots. For each stage, add an enum variant + a dispatch
arm that calls a thin ONNX wrapper:

```toml
# Cargo.toml
[dependencies]
ort = { version = "2", optional = true }

[features]
onnx = ["dep:ort"]
```

```rust
// src/inversion/onnx.rs  (analogous: src/bgremove/onnx.rs, src/separation/onnx.rs)
#[cfg(feature = "onnx")]
pub fn onnx_dipole(local_field_ppm: &[f64], mask: &[u8], grid: &Grid,
                   bdir: (f64,f64,f64), params: &OnnxParams) -> Vec<f64> { ... }
```

- New enum variants: `InversionAlgorithm::{Xqsm, Iqsm, QsmGan, ...}`,
  `BgRemovalAlgorithm::Bfrnet`, `SeparationAlgorithm::{SusepNet, ChiSepNet}`.
- Wrapper responsibilities: f64↔f32 cast (nets are f32), pad-to-multiple-of-8/16, per-model
  normalisation (z-score / scale constants baked from the `.mat` stats), NCDHW layout, run session,
  crop, re-mask. All the pre/post steps already exist in `utils/` (`padding`, `gradient`, `mask`,
  `multi_echo`).
- A small shared `utils/onnx.rs` for session creation + a weight-resolution helper (see §4).
- The nets are f32 and column-major differences matter: verify axis order against a known
  input/output pair at wire-up (the Python `recon.py` for each is the reference).

Effort per Bucket-A/B model once the harness exists: ~0.5–1 day each (mostly parity testing),
front-loaded by ~2–3 days building the shared ONNX harness + weight fetcher + one reference test.

---

## 4. Weight extraction, conversion, and hosting

**Extraction** — weights are obtainable now for the in-scope set:
- committed in QSM-CI repo: BFRnet.onnx (A), QSMGAN, LPCNN;
- public downloads: xQSM (GH release), iQSM/iQFM/iQSM+ (HF Hub), SUSEP-Net (GDrive), IR2QSM (GDrive);
- We can also lift any of these out of the already-built QSM-CI Docker images if a source link rots.

**Conversion (Bucket B)** — one-time offline step per model, scripted:
```python
model = ModelClass(); model.load_state_dict(torch.load(pth, map_location='cpu')); model.eval()
torch.onnx.export(model, example, "model.onnx", opset_version=17,
                  dynamic_axes={'in': {2:'D',3:'H',4:'W'}, 'out': {2:'D',3:'H',4:'W'}})
# verify: max|onnxruntime(x) - torch(x)| < 1e-4 on a real volume
```
Keep these scripts + parity checks in QSM-CI (it already has the Python env and reference
`recon.py`), and treat the emitted `.onnx` as the artifact.

**Hosting** — put the exported `.onnx` files on **OSF** (stable, citable, direct-download URLs;
handles 50–200 MB fine) and/or mirror on **Hugging Face Hub** (iQSM already lives there). QSM.rs then
fetches on demand and caches under a local dir (e.g. `$XDG_CACHE_HOME/qsm-rs/models/`), pinned by
SHA-256 so tests are reproducible. Do **not** commit weights into the QSM.rs git repo. Follow the
existing pattern (env-var-pointed model path for tests, like `AMPPE_BIDS`).

---

## 5. Licensing — the real gate (not the tech)

The engineering is straightforward; redistribution rights are the actual blocker. Before re-hosting
any weight file on OSF:

- **MIT / clearly free:** QSMGAN (fork is MIT), NeXtQSM (MIT, but out of scope for other reasons).
  Safe to re-host.
- **"Author permission" / no LICENSE:** BFRnet, iQSM/iQFM/iQSM+, xQSM, SUSEP-Net, IR2QSM, LPCNN.
  Default = all rights reserved. We can run them locally, but **must get written OK from each author
  before mirroring weights on OSF.** Many share authors (Hongfu Sun / deepMRI group covers
  xQSM/iQSM/iQFM/iQSM+/IR2QSM/BFRnet) — likely **one email** clears most of the list.
- **Gated, do not re-host:** χ-sepnet and QSMnet/QSMnet+ (SNU-LIST academic license, Google-form
  distribution, no LICENSE). We can support *running* a user-supplied model file, but cannot
  bundle/mirror the weights.

Recommendation: where redistribution isn't cleared, support a "bring-your-own-weights" path (point
QSM.rs at a local `.onnx`) so the code works without us shipping the file.

---

## 6. Recommended plan

1. **Spike (½ day):** wire `ort` behind an `onnx` feature, load `BFRnet.onnx`, run it on one BIDS
   volume, correlate against QSM-CI's Python BFRnet output. This de-risks the whole runtime path with
   zero conversion work.
2. **Harness (2–3 days):** shared `utils/onnx.rs` (session + weight cache/fetch + SHA pin), one
   dispatcher wiring (`BgRemovalAlgorithm::Bfrnet`), one `--ignored` integration test on `bids/`.
3. **First dipole net — xQSM (1 day):** public weights, clean feed-forward, no gnarly preprocessing;
   proves the PyTorch→ONNX export + parity workflow end-to-end.
4. **Batch Bucket B:** iQSM/iQFM/iQSM+ (share a net + author), SUSEP-Net (chi-sep), QSMGAN
   (MIT — safe re-host, good first OSF upload).
5. **Middle cases:** LPCNN as a hybrid (ONNX proximal + Rust FFT dipole step), then IR2QSM
   (AddNoise-stripped).
6. **In parallel, non-blocking:** email authors (Sun group first) for re-hosting permission; upload
   cleared weights to OSF.
7. **Optional later spike:** `tf2onnx` on QSMnet (highest-value TF-1 net) — time-boxed, abandon if
   op coverage breaks.

**In scope now:** BFRnet, xQSM, iQSM, iQFM, iQSM+, SUSEP-Net, QSMGAN (7 solid), + LPCNN/IR2QSM
(2 with caveats), + χ-sepnet as bring-your-own-weights. ~10 methods spanning BFR, phase→field,
dipole, and chi-separation.

**Out of scope:** DIP-UP, INR-QSM, MoDIP (per-subject optimisation → native reimplementation, not
ONNX), QSMnet/QSMnet+/AutoQSM/MoDL-QSM/NeXtQSM (legacy TF, brittle conversion).
</content>
</invoke>

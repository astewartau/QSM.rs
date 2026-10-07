# Scoping: DIP-UP as a deep-learning phase unwrapper (issue #123)

Compiled 2026-10-07 against DIP-UP @ `bae8a90` and the QSM.rs 7 T simulation phantom.

**Verdict: export works, accuracy does not. Not shipped.** Both pretrained networks export to ONNX
cleanly and `tract` reproduces PyTorch exactly (corr 1.00000000, zero wrap-class disagreement), so
the engineering path is proven and cheap. But carried through the full chain on this repo's own
phantom, the pretrained network **without** the test-time Deep Image Prior loop yields a
susceptibility map correlating **0.023** (PhaseNet3D) / **0.018** (PHU-NET3D) with ground truth,
against **0.44–0.50** for the three classical unwrappers already in the crate, which run 40–300×
faster and need no 15 GB forward pass. Measuring the DIP loop shows it does **not** account for that
gap. So no `UnwrapMethod` variant was added, no registry entry, and no weights were uploaded.

This supersedes the one-line DIP-UP row in [`DL_ONNX_SCOPING.md`](DL_ONNX_SCOPING.md) (Bucket C),
which called the method architecturally incompatible with ONNX. That is right about the *method* and
wrong about its *pretrained half*: the CNN exports fine. The blocker is accuracy, not architecture.

Everything below is reproducible with `qsmci/scripts/onnx-export/reference/ref_dipup.py` plus
`dipup_baseline.rs`, `dipup_downstream.rs`, `dipup_tract_parity.rs` and `dipup_figure.py` beside it
(the `.rs` files copy into `examples/`). Conversion details, hashes and the
upstream-divergence notes live in that directory's
[`PROVENANCE.md`](../../qsmci/scripts/onnx-export/PROVENANCE.md). None of this is a CI gate (#140).

---

## 1. What DIP-UP is

Zhu, Gao, Xiong, Jiang, Liu, Sun — *Information* 2025, doi:10.3390/info16070592;
<https://github.com/sunhongfu/DIP-UP>.

A pretrained 3D U-Net classifies every voxel of a **single-echo** wrapped phase into one of **9
wrap-count classes**; unwrapping is then `phase + 2π·n`, with `n = class − shift_base` and
`shift_base = 5`, so `n ∈ [−5, +3]`. Two variants ship as checkpoints:

| variant | input channels | width | params |
|---|---|---|---|
| **PHU-NET3D** (paper default) | 2 — wrapped phase + its Laplacian | 64 | 68.72 M |
| **PhaseNet3D** | 1 — wrapped phase | 48 | 38.66 M |

A test-time **Deep Image Prior** loop then fine-tunes the network on each input under a masked
total-variation loss and a Laplacian-consistency loss. Being single-echo, multi-echo use means
running it per echo and doing the usual per-voxel linear fit of unwrapped phase over TE.

Predicting an integer label field is genuinely attractive here, and was the reason to look: the
output is checkable against classical unwrappers voxel-by-voxel, rather than being a map you can
only eyeball.

### Two premises from the issue, now resolved

- **Licence.** Redistribution permission from Hongfu Sun exists, so the convention would have been
  `license: ""` as for `chi-sepnet`. **Not exercised** — nothing was uploaded. The upstream repo still
  has no LICENSE file, and the Zenodo deposit (22091288) is marked *"Other (Not Open)"*.
- **Upstream is public**, but **incomplete**: both `Unet_{1,2}Chan_9Class.py` do
  `from Unet_blocks import *` and **`Unet_blocks.py` is not in the repo**, so neither the authors'
  `Demo_DIP_*.py` nor their `inference.py` runs from a clean clone. The block layout was
  reconstructed and pinned by `load_state_dict(strict=True)` — all 156 keys and every shape match
  both checkpoints, so the layout has no remaining freedom. Checkpoints come from the authors'
  Dropbox (linked from the README); the repo carries no weights.

---

## 2. The DIP loop cannot be ported, and that is settled

The loop is RMSprop on the **network weights** at inference time. `tract` does inference only.
`lpcnn`, `modl_qsm` and `nextqsm` set a precedent for reimplementing a non-CNN half in Rust, but each
unrolls a *forward physics model*; this is a training loop, and backprop in tract is not attempted.

Worth recording: the repo's own packaged entry point (`run.py` → `inference.py`, `config.yaml`)
defaults to `checkpoint: null  # (null = random init)`, and its docstring states *"the network is
jointly trained on the input at inference time (no general checkpoint)"*. In **that** path the loop
is the entire method and the pretrained weights are optional. So "the CNN without the loop" is not a
degraded DIP-UP — it is the prior feed-forward baseline the DIP paper exists to improve, and calling
it "DIP-UP" would overstate it. Anything user-facing would have to say *"the PHU-NET3D/PhaseNet3D
network from DIP-UP"*.

---

## 3. Export and tract parity — the part that works

`export_dipup.py`, opset 17, `dynamo=False`, dynamic spatial axes, 9-channel logits out (decoding
left to the caller). No graph rewrites needed.

| check | PhaseNet3D | PHU-NET3D |
|---|---|---|
| torch ↔ onnxruntime (2 sizes) | rel 1.28e-6, argmax exact | rel 1.13e-6, argmax exact |
| **torch ↔ tract** | corr 1.00000000, rel 1.57e-6, argmax 0/32768 | corr 1.00000000, rel 1.45e-6, argmax 0/32768 |

Parity is scored **relative**: these are unnormalized logits and the variants differ ~5× in scale
(|logit| ~108 for PHU-NET3D, ~19 for PhaseNet3D), so a flat 1e-4 absolute tolerance flags PHU-NET3D
(max|Δ| 1.6e-4) while passing PhaseNet3D at the *same* relative error. The hard gate is **argmax
exactness** — logit drift that never flips a class cannot change a wrap count. Both gates were
confirmed to fail when tightened.

One upstream behaviour had to be changed to get a deterministic graph at all: `forward` calls
`F.dropout(x, 0.2)` with no `training=` argument, so **dropout stays active after `.eval()`**. Two
runs on one input disagree on the predicted wrap count at **30%** of voxels (PHU-NET3D) / **22%**
(PhaseNet3D). The export threads `training=self.training`.

---

## 4. Accuracy on the phantom

Phantom: 164×205×205, 1 mm, **7 T**, TE = 4/12/20/28 ms. The networks were trained on brain phase at
TE = 10 ms (simulation) / 5.8 ms (in vivo), so later echoes are out of distribution — QSM-CI's own
`algorithms/dip-up/README.md` flags this domain shift in advance.

### 4.1 The reference is sound

Accuracy is scored against ROMEO's wrap count. That reference was validated without ground truth:
ROMEO and best-path are both **congruent** (`(unwrapped − wrapped)/2π` is integer to ~3e-7) and agree
with **each other** on **99.995% / 99.985% / 99.896% / 99.545%** of in-mask voxels across the four
echoes — two independent algorithms. (Laplacian unwrapping is *not* congruent — residual 0.5 — so it
cannot serve as a wrap-count reference; its 46–70% agreement reflects that, not an error.)

### 4.2 Wrap-count accuracy, and the baseline that matters

The 9 classes are adequate: 100% / 99.97% / 99.78% / 99.08% of in-mask voxels need an `n` inside
`[−5, +3]`. Representability is not the problem.

Read this section with §4.5 in mind: wrap-count agreement is a *diagnostic*, not a quality score.
An unwrapper can disagree with ROMEO everywhere and still be excellent, provided the disagreement is
a smooth harmonic field — Laplacian unwrapping does exactly that. What the numbers below are good
for is characterising *how* the CNNs differ, not deciding whether they are acceptable.

With that caveat: the number that matters is not raw accuracy — the reference is mostly zero, so
"predict no wraps anywhere" already scores well, and any accuracy figure has to be read against it. Full volume, best
global integer shift granted (it comes out 0 everywhere). `n≠0` columns are accuracy restricted to
the voxels that actually need unwrapping:

| echo | needs n≠0 | trivial "no wraps" | PHU-NET3D | on n≠0 | PhaseNet3D (pre-masked) | on n≠0 |
|---|---|---|---|---|---|---|
| 1 (4 ms) | 4.25% | 95.75% | 98.66% | 90.81% | **99.40%** | 95.07% |
| 2 (12 ms) | 20.36% | 79.64% | 90.55% | 66.99% | **88.71%** | 62.46% |
| 3 (20 ms) | 29.45% | 70.55% | 78.05% | 51.29% | **76.14%** | 46.11% |
| 4 (28 ms) | 36.02% | 63.98% | 62.93% ⟵ *below trivial* | 35.56% | **67.03%** | 39.31% |

Both degrade monotonically with TE. PHU-NET3D — the paper's default — ends up **worse than
predicting no wraps at all** by echo 4. PhaseNet3D with pre-masked input is the better configuration
and stays above trivial at every echo, but its margin at echo 4 is 3 pp, and accuracy on the voxels
that need unwrapping has fallen to 39%. Against a reference two independent classical methods agree
on to 99.5%, a 39% hit rate means ~2π errors scattered through a third of the brain.

### 4.3 Conventions the public repo does not pin down

Each decided by measurement, so the numbers above are not an artefact of a bad guess:

- **Input must be brain-masked.** The authors' demo derives its mask as `image != 0`, implying
  pre-masked input; our phantom phase is whole-head. Pre-masking is worth up to ~9 pp to PhaseNet3D
  and flips it above the trivial baseline at echo 4.
- **Sign convention is correct as-is.** Flipping the phase sign collapses wrapped-voxel accuracy
  to under 10% (1.4–10% across the configurations tried).
- **PHU-NET3D's Laplacian channel is not recoverable, and barely earns its place.** Training consumed
  precomputed `*_wph_10ms_Lap.nii`; the repo ships `dker.mat` (27-point isotropic Laplacian) but no
  script that builds them. The 27-point kernel, QSM-CI's 7-point `torch.roll` stencil and the sin/cos
  identity all land within ~2 pp of each other, and **ablating the channel to zeros costs only
  ~1.5–6 pp**. Note QSM-CI's wrapper uses a kernel that is not the one the repo ships.

### 4.4 Total field after the echo fit — a weak metric, shown for completeness

Per-echo unwrap → per-voxel linear fit over TE → ppm, scored against the phantom ground-truth field
map. Identical echo-fit maths for every row, so this is a like-for-like comparison.

| method | corr | NRMSE |
|---|---|---|
| classical: **ROMEO** (`qsm-core`) | **0.68106** | 0.7324 |
| classical: **best path** (`qsm-core`) | 0.67819 | 0.7351 |
| classical: **Laplacian** (`qsm-core`) | 0.54112 | 0.8417 |
| pretrained **PhaseNet3D**, no DIP | 0.23862 | 1.0301 |
| pretrained **PHU-NET3D**, no DIP | 0.10774 | 1.1812 |
| *no unwrapping at all* | 0.02833 | 1.0439 |

Both networks land far below all three classical unwrappers. But **this table should not be used to
decide anything**, for two reasons. The 0.68 ceiling is low because this simple echo fit leaves the
phantom's receive offset and shim field in, so the metric is dominated by something unwrapping is
not responsible for. More fundamentally, unwrapping has no unique correct answer: two unwrappings
differing by a *harmonic* field produce the **same** local field once background-field removal has
run, so penalising that difference here is meaningless. §4.5 does the comparison properly.

### 4.5 The comparison that actually decides it — carry each unwrapper through to χ

Same per-echo unwrapped phase → linear fit over TE → **V-SHARP** → **RTS**, scored against ground
truth at each stage (local field and χ on V-SHARP's eroded mask, which is what a real pipeline
carries forward):

| unwrapper | total field | local field (V-SHARP) | **χ (RTS)** |
|---|---|---|---|
| | corr / NRMSE | corr / NRMSE | corr / NRMSE |
| ROMEO | 0.6811 / 0.7324 | 0.4824 / 1.0141 | 0.4448 / 0.9945 |
| best path | 0.6782 / 0.7351 | 0.4608 / 1.0537 | 0.4293 / 1.0201 |
| **Laplacian** | 0.5411 / 0.8417 | **0.5190** / 0.9229 | **0.4980** / 0.9019 |
| PhaseNet3D | 0.2386 / 1.0301 | 0.0309 / 6.2230 | **0.0231** / 5.7363 |
| PHU-NET3D | 0.1077 / 1.1812 | 0.0340 / 9.4512 | **0.0178** / 8.8575 |

**The Laplacian row is the proof that §4.2 and §4.4 are the wrong yardsticks.** Laplacian unwrapping
disagrees with ROMEO's wrap count at **53.6%** of voxels and comes *last* on total field (0.541) —
yet it comes **first** on χ (0.498, ahead of ROMEO's 0.445). Its disagreement with ROMEO is a smooth,
large-scale, harmonic field, so V-SHARP deletes it and nothing reaches the susceptibility map. A
wrap-count or total-field score cannot distinguish that from a real error, and would have ranked
Laplacian last.

**The CNNs fail the opposite way, and background-field removal makes it worse rather than better.**
Their χ correlation collapses to 0.02 and NRMSE rises to 6–9×. Their wrap errors are *scattered*:
measured as the fraction of in-mask voxels where the wrap error changes between neighbours, they sit
at 20–26% against a 33–36% error rate — i.e. mostly small, speckled clusters rather than a few large
regions. Each is a 2π discontinuity, which is high-spatial-frequency and non-harmonic, so SMV
filtering cannot remove it and the dipole inversion amplifies it. Visually the unwrapped phase is
blocky and the residual fringe pattern is still present, and the resulting χ maps are blown out.

So the conclusion survives the correct test — but it rests on χ, not on phase correlation, and the
reason is the *character* of the error (scattered 2π steps) rather than its magnitude.

### 4.6 Cost

| | ROMEO / best path / Laplacian | pretrained CNN (whole volume, CPU) |
|---|---|---|
| time, 164×205×205, per echo | 0.6–1.5 s | 41–57 s (PhaseNet3D), 134–234 s (PHU-NET3D) |
| peak RSS | not measured; these are in-place f64 passes | **15.2 GB** |
| weights | none | 155 MB / 275 MB |

---

## 5. What the DIP loop actually adds

Measured directly: 64³ brain-centred patch, echo 4, 100 iterations (QSM-CI's default), reference
`vlr` schedule (lr 1e-6, ×0.9 every 10 iters), losses/optimiser/decoding following
`dip_up_infer.py` verbatim. Trivial baseline on this patch is 84.64%.

| | exact, iter 0 → 100 | on n≠0 voxels, iter 0 → 100 |
|---|---|---|
| **PHU-NET3D** | 69.82% → **86.31%** | 54.01% → **36.13%** |
| **PhaseNet3D** | 63.36% → 63.48% | 56.67% → **63.08%** |

PHU-NET3D's aggregate gain is an artefact: the loop drives the prediction **toward zero wraps**,
which flatters a mostly-zero reference while its accuracy on the voxels that actually need
unwrapping falls 18 pp. That is what the losses ask for — masked TV penalises spatial variation of
the unwrapped phase, and a constant count field minimises it. PhaseNet3D does genuinely improve on
the n≠0 voxels (+6.4 pp), but its aggregate stays ~21 pp *below* the trivial baseline.

Either way the movement is single-to-double-digit percentage points in wrap-count agreement, against
a gap in χ correlation from 0.02 to 0.44. **The missing loop does not explain the shortfall.** Caveats, stated plainly:
this is 100 iterations rather than the repo's 2000, on a patch rather than the whole volume, on
out-of-distribution 7 T data. The learning-rate schedule is self-limiting (by iteration 100 the rate
is already down to 3.5e-7 and both curves are flattening), so more iterations would not change the
picture qualitatively — but this is a bounded measurement, not the published method reproduced.

---

## 6. Two integration problems, for whenever this is revisited

Worth recording because they are not obvious and they outlast this verdict.

**The enum the issue names is the wrong one.** There are two:

- `UnwrapMethod` (`src/unwrap/mod.rs`) — consumers are `utils/multi_echo.rs:387` and `:707`, which
  both unwrap a **Hermitian-inner-product phase *difference*** between two echoes for receive-offset
  estimation, plus `unwrap/slicewise.rs`, which unwraps **one 2D slice** at a time. A single-echo 3D
  wrap-count CNN trained at a fixed TE fits none of those: a HIP difference has entirely different
  wrap statistics, and a 2D slice cannot feed a net needing ≥16 voxels in all three axes.
- `UnwrappingAlgorithm` (`src/pipeline/config.rs`, `Romeo | Laplacian`) — consumed by
  `pipeline/field_mapping.rs`, which is where **each echo's own phase** is unwrapped. *That* is the
  only place this model belongs.

**Tiling a wrap-count field is not like tiling a field inversion.** WASM has a 4 GB heap and the
whole-volume forward pass peaks at 15.2 GB, so a browser path would need tiling at ≤96³ for
PHU-NET3D. But a misclassified voxel is a **2π step**, not a small smooth error, so tile seams would
show up as hard phase discontinuities rather than the mild boundary blur that `WASM_PATCH` tolerates
in `chisepnet.rs` / `susep_net.rs`. Read #89 (erratic threading in the tiled path) before going
there.

---

## 7. If this is picked up again

The export tooling is committed and the parity is proven, so re-entry is cheap. What would change
the verdict:

1. **In-distribution data.** The failure is concentrated at long TE and high field. On 3 T data near
   TE 10 ms the network may well be competitive — echo 1 here (TE 4 ms, 90.8% on wrapped voxels) is
   the encouraging end of the trend. Scoring it on 3 T in-vivo data before writing it off entirely
   would be fair to the method.
2. **A retrained or wider-class model.** 9 classes spanning `[−5, +3]` is tight for 7 T multi-echo.
3. **Reimplementing DIP natively.** The honest route to the published method is a Rust optimisation
   loop around our FFT/dipole machinery, as `DL_ONNX_SCOPING.md` already suggests for INR-QSM and
   MoDIP — not ONNX. Sizeable, and it needs autodiff through a CNN.

Until then #123 should stay open with this as its answer, and nothing user-facing should claim
QSM.rs has a deep-learning unwrapper.

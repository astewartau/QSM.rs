# STATUS — EPG R2/R2′ relaxometry for chi-separation

Session summary of work done in QSM.rs (with supporting changes in `qsm-forward`
and `qsmci`). Goal: give `chi_sep_medi` a usable **R2′ = R2\* − R2** input —
R2\* from multi-echo GRE, R2 from multi-echo spin-echo (MESE) — validated on a
simulated phantom. Algorithms land in QSM.rs first, then get wired into QSMxT.

---

## 1. Code added to QSM.rs

All standalone, opt-in library functions. `cargo test` green, clippy clean.

### `qsm_core::relaxometry` — EPG-based R2/R2′  (`src/utils/epg.rs`)
- `epg_cpmg_echoes(t2, t1, b1, esp, n_echoes)` — CPMG multi-echo spin-echo forward
  model (Weigel 2015). At `b1 = 1` it reduces to `exp(-TE/T2)` to <1e-6.
- `r2_epg(magnitude, mask, echo_times, grid, &R2EpgParams) -> (r2_map, b1_map)` —
  per-voxel R2 from MESE via EPG dictionary matching over a (T2×B1) grid. Hz output.
- `r2prime(r2star, r2, mask)` — R2\* − R2, clamped ≥0.

### `qsm_core::denoise` — MP-PCA  (`src/utils/denoise.rs`)
- `mppca_denoise(data, dims, n_vols, patch_radius, mask)` — Marchenko-Pastur PCA
  denoising (Veraart 2016), faithful DIPY port with a hand-written cyclic Jacobi
  eigensolver (no linalg dependency). Removes random noise along the volume axis,
  edge-preserving.

### `qsm_core::unring` — Gibbs unringing  (`src/utils/gibbs.rs`)
- `gibbs_unring(data, dims, n_vols)` — Kellner-2016 subvoxel-shift unringing,
  generalised from DIPY's 2D-slice to **full 3D** (cosine weight per axis
  `∏_{j≠i}(1+cos k_j)/Σ`, reduces to DIPY's 2D form). Validated: matches DIPY 1D
  output to <5e-3.
- `gibbs_unring_volume(vol, dims)` — single 3D volume.
- `gibbs_unring_masked(data, dims, n_vols, mask, margin)` — bounding-box fast path.

### Fix: `qsm_core::r2star::r2star_arlo`  (`src/utils/r2star.rs`)
- The old code was **not ARLO** — a biased ratio-mean whose `alpha ≤ 1` filter kept
  only downward noise fluctuations, inflating R2 on noisy/many-echo data. Replaced
  with the genuine Pei-2015 integral estimator. Improved phantom R2\* corr 0.20 →
  0.62. All 7 existing r2star tests still pass; also improves the QSMxT pipeline's
  R2\*/T2\* maps.

### Tests  (`tests/r2prime_integration.rs`, all `#[ignore]`)
Component diagnostics, end-to-end R2′ vs ground truth, chi-separation with
derived-vs-GT R2′, unring/denoise/bbox benchmarks, SNR-sweep + MP-PCA-low-SNR.
Read echo times from `params.json`; select phantom via `QSMCI_CHISEP` env.
**Run with `cargo test --release`** — `chi_sep_medi`/EPG in debug take hours.

---

## 2. Supporting changes (phantom infrastructure)

### `qsm-forward` (working tree)
- `generate_se_signal_epg` + `_epg_*` helpers and `generate_bids(se_refocus_b1=…)`
  to simulate **imperfect-refocusing** MESE (fast (T2,T1,B1) dictionary + interp).
  Python EPG matches the Rust version to machine precision.

### `qsmci/scripts/gen_chisep_epg.py`
- Generator for the EPG chi-sep phantom. Added `--peak-snr` (e.g. `inf` for
  noiseless) and `--voxel` (e.g. `0.64` = source res → NO k-space crop).
- Phantoms produced under `qsmci/data/sim/`:
  `chisep-epg` (SNR 250), `chisep-epg-noiseless` (SNR ∞), `chisep-epg-nocrop`
  (native 0.64 mm, noiseless).

---

## 3. Validation results

**EPG works** — recovers true R2 (bias −0.4 Hz, corr 0.82 on the SNR-250 phantom;
recovers B1). Unit test shows it beats a mono-exp fit under imperfect refocusing.

**Chi-separation end-to-end** (`chi_sep_medi`, derived vs ground-truth R2′, scored
vs GT χ-para/χ-dia):

| R2′ source | corr para (iron) | corr dia (myelin) |
|---|---|---|
| ground truth | 0.83 | 0.73 |
| derived | 0.48 | 0.25 |

Diamagnetic/myelin depends most on R2′ → it's the fragile component.

**R2\* improvement:** NLLS is worse than the fixed ARLO; **MP-PCA was inert**
(0.616→0.616) but **Gibbs unringing helped** (0.616→0.664) — see lesson 6.

**The decisive experiments** (why R2′ looked bad, resolved):

| condition | R2\* | R2 | R2′ |
|---|---|---|---|
| SNR 250, vs analytic-map GT | 0.617 | 0.820 | **0.379** |
| SNR ∞ (noiseless), vs analytic-map GT | 0.617 | 0.820 | 0.379 |
| native no-crop, noiseless, vs analytic-map GT | 1.000 | 0.998 | **0.995** |
| SNR 250 **cropped**, vs effective-truth GT | 0.998 | 0.997 | **0.995** |

**SNR sweep** (R2′ corr vs the fair effective-truth reference, crop present):
∞ → 1.00, 250 → 0.995, 100 → 0.967, 50 → 0.866, 20 → 0.457, 10 → 0.182.
So R2′ is genuinely good at realistic SNR (50–100); it only collapses below ~20–30.

**EPG bias–variance** (R2 corr vs SNR): EPG beats mono-exp at good SNR (bias
correction) but is worse at low SNR (SNR 20: EPG 0.52 vs mono-exp 0.55) — the B1
search overfits noise. Fixing B1 recovers most of the low-SNR gap; the B1 *value*
affects bias, not correlation.

**MP-PCA at low SNR** (`bench_denoise_lowsnr`, raw → denoised, corr vs
effective-truth): its benefit scales with noise — inert at SNR 250, then:

| SNR | R2\* raw→dn | R2 raw→dn | R2′ raw→dn |
|---|---|---|---|
| 100 | 0.987→0.990 | 0.989→0.991 | 0.968→0.975 |
| 50 | 0.950→0.971 | 0.943→0.957 | 0.867→0.902 |
| 20 | 0.743→0.879 | 0.517→0.594 | 0.458→0.548 |

So MP-PCA does nothing against systematic error (high SNR) but recovers
substantial R2\*/R2/R2′ quality against random noise (low SNR: R2\* +0.14, R2′
+0.09 at SNR 20) — the right tool for the noise artifact, useful exactly where
real acquisitions live.

---

## 4. Lessons learned

1. **Validate the validator first.** The apparent "R2′ is hard" ceiling (corr 0.38)
   was **not** physics, noise, or the estimators — it was a phantom artifact: the
   ground truth was built by *image-space downsampling* the analytic maps while the
   signal was built by *k-space truncation*. Two different PSFs. When matched (no
   crop, or a fair effective-truth reference), R2′ recovers at **0.995**.

2. **Consistency beats fidelity for a reference.** It's not the crop that hurt — a
   *single, consistent* crop is fine. The damage was the *mismatch* between how GT
   and signal were downsampled. A phantom can't simultaneously have a clean
   parameter-map ground truth AND realistic sub-voxel partial-volume signal
   (mono-exp fit of a mixed voxel ≠ any downsampled rate map).

3. **R2′ = R2\* − R2 has a subtraction penalty.** R2′'s dynamic range (~3 Hz) is
   ~2.5× smaller than its parents' (~7–8 Hz), while the difference inherits both
   parents' errors. So R2′ degrades faster than R2\*/R2 with noise. The real-world
   fix is **acquisition** (joint GESSE/GESFIDE, shared PSF/noise → cancels in the
   subtraction), not better post-processing. This is why χ-sepnet-R2\* exists.

4. **EPG is a bias–variance tradeoff.** Its extra B1 degree of freedom corrects the
   imperfect-refocusing bias at good SNR but overfits noise at low SNR. A known/
   fixed B1 (from a separate B1 map) would give the bias correction *and* the
   robustness — a clean future enhancement to `r2_epg`.

5. **`r2star_arlo` was never ARLO.** A plausible-looking ratio-mean with a noise
   selection bias. The real integral ARLO tripled the R2\* correlation. Worth
   re-reading "known-good" code before building on it.

6. **Right tool for the right artifact.** MP-PCA (removes random noise) did nothing
   on the high-SNR phantom because the residual error was *systematic* (Gibbs/
   partial-volume), not noise. Gibbs unringing (removes truncation ringing) *did*
   help. An earlier "Gaussian denoise +0.07" was exposed as spatial blurring toward
   a smooth GT, not real denoising.

7. **Measure, don't assume.** Several confident hypotheses were wrong until tested:
   the "bug" was in `r2star_arlo` not EPG; noise contributed ~0 at SNR 250 (not the
   ~half a quadrature budget assumed); the Gaussian "improvement" was cheating.
   Every conclusion here is backed by an experiment, and a few overturned my priors.

8. **Operational.** Run numeric integration tests with `--release` (debug is 10–30×
   slower — a `chi_sep_medi` run went from hours to minutes). Native-res phantoms
   (26 M voxels) OOM the pack step and Rust double-buffering; do large-volume
   confirmations per-echo in Python. `pkill -f` on a substring that also matches the
   launching command kills the wrong process.

---

## 5. Open items / TODO

- ~~**MP-PCA low-SNR benchmark**~~ — done, results in §3.
- ~~**`r2_epg` optional B1-map input**~~ — done: `r2_epg(..., b1_map: Option<&[f64]>)`
  snaps each voxel's B1 to the nearest `b1_grid` entry and restricts the search to
  that column (T2-only fit). Unit-tested (recovers R2, snapping, wrong-B1 bias).
- ~~**Gibbs unring perf**~~ — done, ~2.5× (128³: 1.35 s → 0.54 s), not via the
  approximate-upsampling idea (FLOP analysis: a 50×-zero-padded IFFT costs about the
  same as the 90 small IFFTs it replaces). Instead: (1) both shift directions share
  one complex IFFT via Hermitian packing (`IFFT(c·(ph + i·conj(ph))) = img₊ + i·img₋`,
  even-n Nyquist bin symmetrised — max |diff| ~4e-3 vs old, r > 0.9999999);
  (2) `tv_min_line` de-modded (diff array + windowed sums, no `%` in the interior).
- **QSMxT wiring**: MESE BIDS discovery (suffix `MESE`, echo required) + MESE→GRE
  reslice (QSMxT has only header-based oblique→axial reslice; extend to
  `resample_onto_grid`) + R2′/chi-sep pipeline stages, with unring/MP-PCA as opt-in
  preprocessing. No BIDS suffix exists for R2′/para/dia maps (needs `.bidsignore`).
- **Consider R2\*-direct chi-sep** (χ-sepnet-R2\*) to sidestep the R2′ subtraction —
  note it's a deep-learning method, so a port implies ONNX/weights infrastructure.

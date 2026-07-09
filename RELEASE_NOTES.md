# QSM.rs v0.14.0

*Changes since v0.13.0. Minor bump due to a breaking `run_qsmart` API change (0.x SemVer).*

## Highlights
QSMART is now correct and configurable: a structural bug that made `run_qsmart` produce near-useless maps is fixed, and the inner dipole inversion is now selectable.

## 🐛 Fixed
- **`run_qsmart` produced structurally-wrong susceptibility maps.** Several bugs caused the reconstruction to decorrelate from ground truth (corr ≈ 0): wrong stage masks (stage 1 excluded vessels; both stages used the full mask instead of whole-ROI / tissue-only), the wrong `removed_voxels` passed to `adjust_offset` (`vasc_mask` instead of `mask − vasc`), an incorrect `ppm` rescale in the offset step, and the output not being zeroed outside the mask. `run_qsmart` now reproduces the validated reference reconstruction to floating-point precision.

## ✨ Added
- **Swappable QSMART inner dipole inversion.** `QsmartParams` gained an `inversion: InversionAlgorithm` field (default `Ilsqr`); QSMART can now use any standard inversion (TKD, TSVD, Tikhonov, TV, RTS, NLTV, MEDI, iLSQR) for both stages. iLSQR `tol`/`max_iter` stay pinned to QSMART's own fields, so the default path is byte-for-byte unchanged.

## ⚠️ Changed (breaking)
- **`run_qsmart` signature:** now takes `&InversionConfig` instead of `&QsmartParams` (it reads `inversion_config.qsmart` internally and dispatches the inner inversion). `Qsmart`/`Tgv` are rejected as inner inversions.
- **`QsmartParams`** gained the `inversion` field (additive; constructors using `..Default::default()` are unaffected).

## 🔧 Internal / CI
- Integration tests now exercise `run_qsmart` (the production path) via a shared helper, instead of a manual reimplementation (−240/+46 lines).
- Added `test_pipeline_qsmart_tikhonov`; wired both QSMART tests into the change-detection matrix and the results table; added `--exact` to avoid substring-matching collisions.
- QSMART integration tests run and report metrics/figures but are **non-gating** (`continue-on-error`): QSMART's inherent ~0.39 correlation / ~0.14 XSIM on the synthetic test data is a known accuracy limitation, tracked separately (not a regression — the reference reconstruction scores identically).
- Corrected the QSMART figure label (`render_slices.py`).

## Upgrade notes (downstream)
Callers of `run_qsmart` (qsmxt, qsmbly) must update the call from `&QsmartParams` to `&InversionConfig` when bumping to this tag.

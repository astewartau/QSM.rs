#!/usr/bin/env python3
"""Render center slices from binary files produced by QSM-core integration tests.

Each .bin file contains center axial, coronal, and sagittal slices of a 3D
result volume plus corresponding mask slices.

Two kinds of output:

* **Stage montages** (`stage_*.png`) — one figure per pipeline stage, laying every
  method in that stage out as centre-axial panels on a shared window and colorbar,
  with the ground truth first. This is what the PR comment shows: comparing ~20
  dipole methods is only practical side by side, and it replaces the wall of ~50
  individual figures the comment used to carry.
* **Individual 3-panel figures** — axial/coronal/sagittal for anything NOT covered by
  a montage (BET, SWI, R2*/T2*, before/after pairs, and so on), where there is nothing
  to compare against and the extra views are the point.

Usage:
    python render_slices.py <input_dir> <output_dir>
"""

import math
import struct
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Mapping from binary file slug to display name
NAMES = {
    "bgremove_sharp": "SHARP",
    "bgremove_resharp": "RESHARP",
    "bgremove_vsharp": "V-SHARP",
    "bgremove_pdf": "PDF",
    "bgremove_ismv": "iSMV",
    "bgremove_lbv": "LBV",
    "inversion_tkd": "TKD",
    "inversion_tsvd": "TSVD",
    "inversion_tikhonov": "Tikhonov",
    "inversion_tv": "TV-ADMM",
    "inversion_rts": "RTS",
    "inversion_medi": "MEDI",
    "inversion_nltv": "NLTV",
    "inversion_ndi": "NDI",
    "inversion_fansi": "FANSI (nlTV)",
    "inversion_fansi_tgv": "FANSI (nlTGV)",
    "inversion_l1qsm": "L1-QSM",
    "inversion_whqsm": "WH-QSM",
    "inversion_hdqsm": "HD-QSM",
    "inversion_amp_pe": "AMP-PE",
    "bet": "BET",
    "combined_tgv": "TGV (Combined)",
    "bias_correction": "Bias Correction",
    "pipeline_romeo_b0": "ROMEO + B0",
    "pipeline_harperella": "HARPERELLA",
    "pipeline_iharperella": "iHARPERELLA",
    "pipeline_tgv": "TGV",
    "pipeline_qsmart": "QSMART",
    "pipeline_qsmart_tikhonov": "QSMART (Tikhonov)",
    "swi": "CLEAR-SWI",
    "swi_mip": "CLEAR-SWI mIP",
    "r2star": "R2* (Hz)",
    "t2star": "T2* (s)",
    # Relaxometry demo (self-contained synthetic phantom)
    "relax_r2_truth": "R2 truth (Hz)",
    "relax_r2_epg": "R2 — EPG (Hz)",
    "relax_r2_monoexp": "R2 — mono-exp (Hz)",
    "relax_r2prime_truth": "R2' truth (Hz)",
    "relax_r2prime_derived": "R2' derived = R2*-R2 (Hz)",
    "relax_denoise_after": "R2* — noisy vs MP-PCA denoised (Hz)",
    "relax_unring_after": "R2* map — rung vs Gibbs-unrung (Hz)",
    # Field-based chi-separation (chi_sep_ilsqr)
    "chisep_para": "chi_sep_ilsqr χ+",
    "chisep_dia": "chi_sep_ilsqr χ− (magnitude)",
    # Relaxometry-based chi-separation (R2*-QSM, WaveSep, DECOMPOSE, HC-ChiSep)
    "relaxchisep_r2starqsm_para": "R2*-QSM χ+",
    "relaxchisep_wavesep_para": "WaveSep χ+",
    "relaxchisep_r2starqsm_dia": "R2*-QSM χ− (magnitude)",
    "relaxchisep_wavesep_dia": "WaveSep χ− (magnitude)",
    "relaxchisep_decompose_para": "DECOMPOSE χ+",
    "relaxchisep_decompose_dia": "DECOMPOSE χ− (magnitude)",
    "relaxchisep_hcchisep_para": "HC-ChiSep χ+",
    "relaxchisep_hcchisep_dia": "HC-ChiSep χ− (magnitude)",
    "ground_truth_local_field": "Ground truth",
    "ground_truth_chi": "Ground truth",
    "chisep_para_truth": "Ground truth",
    "chisep_dia_truth": "Ground truth",
    "inversion_ilsqr": "iLSQR",
    "combined_tfi": "TFI (Combined)",
    "dl_qsmgan": "QSMGAN",
    "dl_lpcnn": "LPCNN",
    "dl_ir2qsm": "iR2QSM",
    "dl_modl_qsm": "MoDL-QSM",
    "dl_nextqsm": "NeXtQSM",
    "dl_iqfm": "iQFM",
    "dl_hdbet": "HD-BET",
    "chisep_xsepnet_para": "xSepNet χ+",
    "chisep_xsepnet_dia": "xSepNet χ− (magnitude)",
}

# Panel labels inside a stage montage. The montage already says which stage and, for
# χ-separation, which source a row is — so the per-panel label only needs the method.
MONTAGE_LABELS = {
    "chisep_para": "chi-sep iLSQR", "chisep_dia": "chi-sep iLSQR",
    "chisep_xsepnet_para": "xSepNet", "chisep_xsepnet_dia": "xSepNet",
    "relaxchisep_r2starqsm_para": "R2*-QSM", "relaxchisep_r2starqsm_dia": "R2*-QSM",
    "relaxchisep_wavesep_para": "WaveSep", "relaxchisep_wavesep_dia": "WaveSep",
    "relaxchisep_decompose_para": "DECOMPOSE", "relaxchisep_decompose_dia": "DECOMPOSE",
    "relaxchisep_hcchisep_para": "HC-ChiSep", "relaxchisep_hcchisep_dia": "HC-ChiSep",
    # Relaxometry: the row label already names the tool and units.
    "relax_r2_truth": "Truth", "relax_r2_epg": "EPG", "relax_r2_monoexp": "Mono-exp",
    "relax_r2prime_truth": "Truth", "relax_r2prime_derived": "Derived (R2* − R2)",
    "relax_denoise_before": "Noisy", "relax_denoise_after": "Denoised",
    "relax_unring_before": "Rung", "relax_unring_after": "Unrung",
    "swi": "CLEAR-SWI", "swi_mip": "CLEAR-SWI mIP", "r2star": "R2* (Hz)", "t2star": "T2* (s)",
}

# Fixed display windows (ppm)
WINDOWS = {
    "bgremove_sharp": (-0.025, 0.025),
    "bgremove_resharp": (-0.025, 0.025),
    "bgremove_vsharp": (-0.025, 0.025),
    "bgremove_pdf": (-0.025, 0.025),
    "bgremove_ismv": (-0.025, 0.025),
    "bgremove_lbv": (-0.025, 0.025),
    "inversion_tkd": (-0.1, 0.1),
    "inversion_tsvd": (-0.1, 0.1),
    "inversion_tikhonov": (-0.1, 0.1),
    "inversion_tv": (-0.1, 0.1),
    "inversion_rts": (-0.1, 0.1),
    "inversion_medi": (-0.1, 0.1),
    "inversion_nltv": (-0.1, 0.1),
    "inversion_ndi": (-0.1, 0.1),
    "inversion_fansi": (-0.1, 0.1),
    "inversion_fansi_tgv": (-0.1, 0.1),
    "inversion_l1qsm": (-0.1, 0.1),
    "inversion_whqsm": (-0.1, 0.1),
    "inversion_hdqsm": (-0.1, 0.1),
    "inversion_amp_pe": (-0.1, 0.1),
    "bet": (0, 1),
    "combined_tgv": (-0.1, 0.1),
    "bias_correction": (-0.1, 0.1),  # fallback; before/after rendering uses auto-range
    "pipeline_romeo_b0": (-0.05, 0.05),  # total field in ppm (wider range than local field)
    "pipeline_harperella": (-0.025, 0.025),
    "pipeline_iharperella": (-0.025, 0.025),
    "pipeline_tgv": (-0.1, 0.1),
    "pipeline_qsmart": (-0.1, 0.1),
    "pipeline_qsmart_tikhonov": (-0.1, 0.1),
    "swi": None,       # auto-range (magnitude-weighted)
    "swi_mip": None,   # auto-range (magnitude-weighted)
    "r2star": (0, 100),    # Hz
    "t2star": (0, 0.08),   # seconds (0-80 ms)
    "relax_r2_truth": (0, 45),
    "relax_r2_epg": (0, 45),
    "relax_r2_monoexp": (0, 45),
    "relax_r2prime_truth": (0, 16),
    "relax_r2prime_derived": (0, 16),
    # Before/after pairs share ONE fixed window. They used to auto-range independently, which
    # rescaled each panel and visually cancelled the change the figure exists to show.
    "relax_denoise_before": (0, 60),
    "relax_denoise_after": (0, 60),
    "relax_unring_before": (0, 60),
    "relax_unring_after": (0, 60),
    # chi-separation source maps (ppm; chi- rendered as positive magnitude)
    "chisep_para_truth": (0, 0.1),
    "chisep_para": (0, 0.1),
    "chisep_dia_truth": (0, 0.1),
    "chisep_dia": (0, 0.1),
    # relaxometry-based chi-separation source maps (same ppm window as above)
    "relaxchisep_r2starqsm_para": (0, 0.1),
    "relaxchisep_wavesep_para": (0, 0.1),
    "relaxchisep_r2starqsm_dia": (0, 0.1),
    "relaxchisep_wavesep_dia": (0, 0.1),
    "relaxchisep_decompose_para": (0, 0.1),
    "relaxchisep_decompose_dia": (0, 0.1),
    "relaxchisep_hcchisep_para": (0, 0.1),
    "relaxchisep_hcchisep_dia": (0, 0.1),
}


# Stage montages. Each entry: (output stem, title, colorbar unit, display window, rows).
# A row is a list of slugs; missing slugs are dropped so a partial CI run (only some test
# categories triggered) still produces a sensible montage of what actually ran.
#
# The dipole montage deliberately includes the stage-SPANNING methods (combined BFR+dipole,
# and full pipeline). They consume different inputs, so the sub-row labels say so rather than
# implying a like-for-like comparison with the plain dipole methods.
MONTAGES = [
    ("stage_bfr", "Local field", "ppm", (-0.025, 0.025), [
        ["ground_truth_local_field", "bgremove_sharp", "bgremove_resharp", "bgremove_vsharp",
         "bgremove_pdf", "bgremove_ismv"],
        ["bgremove_lbv", "pipeline_harperella", "pipeline_iharperella", "dl_iqfm"],
    ], ["background field removal", "spans / deep learning"]),
    ("stage_dipole", "Susceptibility maps", "ppm", (-0.1, 0.1), [
        ["ground_truth_chi", "inversion_tkd", "inversion_tsvd", "inversion_tikhonov",
         "inversion_tv", "inversion_rts"],
        ["inversion_ilsqr", "inversion_medi", "inversion_nltv", "inversion_ndi",
         "inversion_fansi", "inversion_fansi_tgv"],
        ["inversion_l1qsm", "inversion_whqsm", "inversion_hdqsm", "inversion_amp_pe"],
        ["dl_qsmgan", "dl_lpcnn", "dl_ir2qsm", "dl_modl_qsm", "dl_nextqsm"],
        ["combined_tgv", "combined_tfi", "pipeline_tgv", "pipeline_qsmart",
         "pipeline_qsmart_tikhonov"],
    ], ["dipole inversion", "dipole inversion", "dipole inversion",
        "deep learning", "spans (BFR+dipole / full pipeline)"]),
    ("stage_chisep", "χ-separation", "ppm", (0, 0.1), [
        ["chisep_para_truth", "chisep_para", "chisep_xsepnet_para",
         "relaxchisep_r2starqsm_para", "relaxchisep_wavesep_para",
         "relaxchisep_decompose_para", "relaxchisep_hcchisep_para"],
        ["chisep_dia_truth", "chisep_dia", "chisep_xsepnet_dia",
         "relaxchisep_r2starqsm_dia", "relaxchisep_wavesep_dia",
         "relaxchisep_decompose_dia", "relaxchisep_hcchisep_dia"],
    ], ["χ+ (para)", "χ− (dia, magnitude)"]),
]

# Supplementary utility outputs. Unlike the stage montages these are unrelated quantities
# (a.u. / Hz / s), so `None` for the window means: take each panel's window from WINDOWS and
# give it its own colorbar rather than forcing a shared scale that would be meaningless.
# Relaxometry toolkit demo (its own workflow). Per-panel windows because R2 and R2' live on
# different scales; the before/after tools get a difference panel, which is the only place
# their effect is actually visible on this phantom.
RELAXOMETRY = ("stage_relaxometry", "Relaxometry toolkit", None, None, [
    ["relax_r2_truth", "relax_r2_epg", "relax_r2_monoexp"],
    ["relax_r2prime_truth", "relax_r2prime_derived"],
    ["relax_denoise_before", "relax_denoise_after",
     "diff:relax_denoise_before-vs-relax_denoise_after"],
    ["relax_unring_before", "relax_unring_after",
     "diff:relax_unring_before-vs-relax_unring_after"],
], ["R2 (Hz)", "R2' (Hz)", "MP-PCA denoise (R2*, Hz)", "Gibbs unring (R2*, Hz)"])

SUPPLEMENTARY = ("stage_supplementary", "Supplementary outputs", None, None, [
    ["swi", "swi_mip", "r2star", "t2star"],
], None)
MONTAGES.append(SUPPLEMENTARY)
MONTAGES.append(RELAXOMETRY)

# Every slug a montage covers; these get no individual 3-panel figure.
MONTAGED = {slug for _, _, _, _, rows, _ in MONTAGES for row in rows for slug in row}


def load_axial(path):
    """Centre axial slice, masked.

    The ground truth is stored unmasked — air reaches ~9 ppm and saturates the display
    window, which would make the reference panel unreadable next to the masked method
    panels. The mask is already in the file, so apply it and let NaN render as the
    figure background.
    """
    with open(path, "rb") as f:
        nx, ny, nz = struct.unpack("<QQQ", f.read(24))
        axial = np.frombuffer(f.read(nx * ny * 8), dtype="<f8").reshape(ny, nx)
        f.read(nx * nz * 8)      # coronal result
        f.read(ny * nz * 8)      # sagittal result
        amask = np.frombuffer(f.read(nx * ny), dtype=np.uint8).reshape(ny, nx)
    return np.where(amask > 0, axial, np.nan)


def _cell_available(input_dir, slug):
    """A plain slug needs its own .bin; a `diff:` cell needs both of its operands."""
    if slug.startswith("diff:"):
        a, b = slug[len("diff:"):].split("-vs-")
        return (input_dir / f"{a}.bin").exists() and (input_dir / f"{b}.bin").exists()
    return (input_dir / f"{slug}.bin").exists()


def _panel(input_dir, slug, gray, diverging):
    """Resolve a montage cell to (data, cmap, window-or-None, label).

    `diff:a-b` is a pseudo-slug rendering the difference of two volumes on a symmetric
    diverging scale. Before/after pairs are close to indistinguishable side by side — the
    whole effect of denoising or unringing lives in what was removed, so show that directly
    rather than asking a reader to spot it between two nearly identical panels.
    """
    if slug.startswith("diff:"):
        a, b = slug[len("diff:"):].split("-vs-")
        data = load_axial(input_dir / f"{a}.bin") - load_axial(input_dir / f"{b}.bin")
        finite = np.abs(data[np.isfinite(data)])
        lim = float(np.percentile(finite, 99.5)) if finite.size else 1.0
        lim = lim if lim > 0 else 1.0
        return data, diverging, (-lim, lim), "Difference (removed)"
    data = load_axial(input_dir / f"{slug}.bin")
    return data, gray, None, MONTAGE_LABELS.get(slug, NAMES.get(slug, slug))


def load_axial_with_mask(path):
    """Centre axial slice plus its stored mask slice, both unmodified."""
    with open(path, "rb") as f:
        nx, ny, nz = struct.unpack("<QQQ", f.read(24))
        axial = np.frombuffer(f.read(nx * ny * 8), dtype="<f8").reshape(ny, nx)
        f.read(nx * nz * 8)
        f.read(ny * nz * 8)
        amask = np.frombuffer(f.read(nx * ny), dtype=np.uint8).reshape(ny, nx)
    return axial, amask > 0


def render_bet_overlay(input_dir, output_path):
    """Brain extraction: every method's mask boundary drawn with the ground truth on the magnitude.

    A bare binary mask says nothing about whether the boundary is in the right place. Drawing
    the predicted and ground-truth outlines together on the magnitude shows exactly where they
    agree and where they part, which is the only thing worth looking at here.

    `bet.bin` / `dl_hdbet.bin` hold each method's mask (result = predicted, mask slice = ground
    truth); `bet_magnitude.bin` (or `dl_hdbet_magnitude.bin`) supplies the underlying image.
    Whichever methods ran are overlaid on one figure.
    """
    methods = [(label, input_dir / f"{slug}.bin", color)
               for slug, label, color in (("bet", "BET", "#2fa8ff"), ("dl_hdbet", "HD-BET", "#e4572e"))
               if (input_dir / f"{slug}.bin").exists()]
    mag_file = next((f for f in (input_dir / "bet_magnitude.bin", input_dir / "dl_hdbet_magnitude.bin")
                     if f.exists()), None)
    if not methods or mag_file is None:
        return False
    magnitude, truth = load_axial_with_mask(mag_file)

    fig, ax = plt.subplots(figsize=(5.2, 5.6))
    finite = magnitude[np.isfinite(magnitude)]
    vmax = float(np.percentile(finite, 99)) if finite.size else 1.0
    ax.imshow(magnitude, cmap="gray", vmin=0, vmax=vmax, origin="lower")
    ax.contour(truth.astype(float), levels=[0.5], colors="#ffd400", linewidths=1.6)
    handles = [Line2D([], [], color="#ffd400", lw=1.8, label="Ground truth")]
    dices = []
    for label, path, color in methods:
        predicted = load_axial_with_mask(path)[0] > 0.5
        ax.contour(predicted.astype(float), levels=[0.5], colors=color, linewidths=1.2)
        inter = np.logical_and(predicted, truth).sum()
        dices.append(f"{label} {2.0 * inter / max(predicted.sum() + truth.sum(), 1):.3f}")
        handles.append(Line2D([], [], color=color, lw=1.4, label=label))
    ax.axis("off")
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.85)
    ax.set_title(f"Brain extraction — centre axial slice\n"
                 f"(this slice: Dice {', '.join(dices)})", fontsize=12, fontweight="bold")
    fig.savefig(output_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Rendered {output_path} (mask overlay)")
    return True


def render_montage(input_dir, stem, title, unit, window, rows, row_labels, output_path):
    """One figure per stage: centre-axial panels on a shared window and colorbar."""
    kept, kept_labels = [], []
    for i, row in enumerate(rows):
        present = [s for s in row if _cell_available(input_dir, s)]
        if present:
            kept.append(present)
            kept_labels.append(row_labels[i] if row_labels else None)
    if not kept:
        return False

    ncol = max(len(r) for r in kept)
    per_panel = window is None      # unrelated quantities: own window + own colorbar each
    # Reserve a fixed band for the suptitle. Without it a one-row montage has no headroom and
    # the figure title lands on top of the panel titles.
    title_band = 0.75
    fig_h = 2.45 * len(kept) + title_band
    fig, axes = plt.subplots(len(kept), ncol,
                             figsize=((2.55 if per_panel else 2.0) * ncol, fig_h),
                             squeeze=False)
    gray = plt.get_cmap("gray").copy()
    gray.set_bad("#cccccc")
    diverging = plt.get_cmap("RdBu_r").copy()
    diverging.set_bad("#cccccc")
    im = None
    for ri, row in enumerate(kept):
        for ci in range(ncol):
            ax = axes[ri][ci]
            ax.axis("off")
            if ci >= len(row):
                continue
            slug = row[ci]
            data, cmap, panel_window, label = _panel(input_dir, slug, gray, diverging)
            if panel_window is None:
                panel_window = window if not per_panel else WINDOWS.get(slug)
            if panel_window is None:                    # still nothing: auto-range
                finite = data[np.isfinite(data)]
                panel_window = ((float(finite.min()), float(finite.max()))
                                if finite.size else (0.0, 1.0))
            im = ax.imshow(data, cmap=cmap, vmin=panel_window[0], vmax=panel_window[1],
                           origin="lower")
            is_truth = "truth" in slug
            ax.set_title(label, fontsize=9,
                         fontweight="bold" if is_truth else "normal",
                         color="#0a7d3a" if is_truth else "black")
            if per_panel or slug.startswith("diff:"):
                fig.colorbar(im, ax=ax, shrink=0.72, aspect=14, pad=0.02)
        # Only label a row when the group changes, so a group spanning several rows reads
        # as one block instead of repeating its name down the side.
        if kept_labels[ri] and (ri == 0 or kept_labels[ri] != kept_labels[ri - 1]):
            axes[ri][0].text(-0.09, 0.5, kept_labels[ri], transform=axes[ri][0].transAxes,
                             rotation=90, va="center", ha="center", fontsize=9.5,
                             fontweight="bold", color="#444444")
    fig.suptitle(f"{title} — centre axial slice", fontsize=13, fontweight="bold",
                 y=1.0 - 0.22 / fig_h)
    fig.subplots_adjust(top=1.0 - title_band / fig_h)
    if not per_panel:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.62, aspect=34, pad=0.015, label=unit)
    fig.savefig(output_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    n = sum(len(r) for r in kept)
    print(f"  Rendered {output_path} ({n} panels)")
    return True


def load_slices(path):
    """Load center slices from a binary file.

    Binary format (little-endian):
        nx: u64, ny: u64, nz: u64
        axial result:    f64 * (nx * ny)
        coronal result:  f64 * (nx * nz)
        sagittal result: f64 * (ny * nz)
        axial mask:      u8  * (nx * ny)
        coronal mask:    u8  * (nx * nz)
        sagittal mask:   u8  * (ny * nz)
    """
    with open(path, "rb") as f:
        nx, ny, nz = struct.unpack("<QQQ", f.read(24))

        axial = np.frombuffer(f.read(nx * ny * 8), dtype="<f8").reshape(ny, nx)
        coronal = np.frombuffer(f.read(nx * nz * 8), dtype="<f8").reshape(nz, nx)
        sagittal = np.frombuffer(f.read(ny * nz * 8), dtype="<f8").reshape(nz, ny)

        # Skip mask slices (not needed with fixed windows)
        f.read(nx * ny + nx * nz + ny * nz)

    return {
        "axial": axial,
        "coronal": coronal,
        "sagittal": sagittal,
    }


def render_figure(slices, name, slug, output_path):
    """Render a 3-panel figure (axial, coronal, sagittal) and save as PNG."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    window = WINDOWS.get(slug, (-0.1, 0.1))
    if window is not None:
        vmin, vmax = window
    else:
        # Auto-range from data (e.g. magnitude images)
        all_vals = np.concatenate([slices[k].ravel() for k in ("axial", "coronal", "sagittal")])
        finite = all_vals[np.isfinite(all_vals)]
        vmin, vmax = (float(finite.min()), float(finite.max())) if len(finite) > 0 else (0, 1)

    for ax, (label, key) in zip(
        axes, [("Axial", "axial"), ("Coronal", "coronal"), ("Sagittal", "sagittal")]
    ):
        im = ax.imshow(slices[key], cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(label, fontsize=11)
        ax.axis("off")

    fig.suptitle(name, fontsize=14, fontweight="bold", y=1.0)
    cb_label = "Mask" if slug == "bet" else ("Hz" if slug.startswith("relax_") else "ppm")
    fig.colorbar(im, ax=axes, shrink=0.85, aspect=30, pad=0.02, label=cb_label)
    fig.savefig(output_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Rendered {output_path}")


def render_before_after(before_slices, after_slices, name, output_path):
    """Render a 2-panel before/after axial figure with independent ranges."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    def auto_range(data):
        finite = data[np.isfinite(data)]
        return (float(finite.min()), float(finite.max())) if len(finite) > 0 else (0, 1)

    bmin, bmax = auto_range(before_slices["axial"].ravel())
    amin, amax = auto_range(after_slices["axial"].ravel())

    axes[0].imshow(before_slices["axial"], cmap="gray", vmin=bmin, vmax=bmax, origin="lower")
    axes[0].set_title("Before", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(after_slices["axial"], cmap="gray", vmin=amin, vmax=amax, origin="lower")
    axes[1].set_title("After", fontsize=11)
    axes[1].axis("off")

    fig.suptitle(name, fontsize=14, fontweight="bold", y=1.0)
    fig.savefig(output_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Rendered {output_path}")


def main():
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("slices")
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    bin_files = sorted(input_dir.glob("*.bin"))
    if not bin_files:
        print(f"No .bin files found in {input_dir}")
        sys.exit(0)

    # Handle before/after pairs (e.g. bias_correction_before + bias_correction)
    BEFORE_AFTER = {
        "bias_correction": "bias_correction_before",
        "relax_denoise_after": "relax_denoise_before",
        "relax_unring_after": "relax_unring_before",
    }

    print(f"Rendering figures from {len(bin_files)} files...")

    # Stage montages first — these are what the PR comment embeds.
    for stem, title, unit, window, rows, row_labels in MONTAGES:
        render_montage(input_dir, stem, title, unit, window, rows, row_labels,
                       output_dir / f"{stem}.png")

    # Slugs covered by a montage get no individual figure: the montage replaces them,
    # which is the whole point of consolidating. Everything else (BET, SWI, R2*/T2*,
    # before/after pairs) still gets its 3-panel axial/coronal/sagittal figure, since
    # there is nothing to compare it against and the extra views carry the information.
    render_bet_overlay(input_dir, output_dir / "bet.png")

    rendered = set(MONTAGED) | {"bet", "bet_magnitude", "dl_hdbet", "dl_hdbet_magnitude"}
    for slug, before_slug in BEFORE_AFTER.items():
        after_file = input_dir / f"{slug}.bin"
        before_file = input_dir / f"{before_slug}.bin"
        if after_file.exists() and before_file.exists():
            name = NAMES.get(slug, slug)
            try:
                before_slices = load_slices(before_file)
                after_slices = load_slices(after_file)
                render_before_after(before_slices, after_slices, name, output_dir / f"{slug}.png")
                rendered.add(slug)
                rendered.add(before_slug)
            except Exception as e:
                print(f"  WARNING: Failed to render {slug} before/after: {e}")

    for bin_file in bin_files:
        slug = bin_file.stem
        if slug in rendered:
            continue
        name = NAMES.get(slug, slug)
        try:
            slices = load_slices(bin_file)
            render_figure(slices, name, slug, output_dir / f"{slug}.png")
        except Exception as e:
            print(f"  WARNING: Failed to render {slug}: {e}")


if __name__ == "__main__":
    main()

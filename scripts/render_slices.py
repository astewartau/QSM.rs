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
    # before/after pairs use auto-range
    "relax_denoise_after": None,
    "relax_unring_after": None,
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
    ("stage_bfr", "Background field removal", "ppm", (-0.025, 0.025), [
        ["ground_truth_local_field", "bgremove_sharp", "bgremove_resharp", "bgremove_vsharp",
         "bgremove_pdf"],
        ["bgremove_ismv", "bgremove_lbv", "pipeline_harperella", "pipeline_iharperella"],
    ], None),
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


def render_montage(input_dir, stem, title, unit, window, rows, row_labels, output_path):
    """One figure per stage: centre-axial panels on a shared window and colorbar."""
    kept, kept_labels = [], []
    for i, row in enumerate(rows):
        present = [s for s in row if (input_dir / f"{s}.bin").exists()]
        if present:
            kept.append(present)
            kept_labels.append(row_labels[i] if row_labels else None)
    if not kept:
        return False

    ncol = max(len(r) for r in kept)
    fig, axes = plt.subplots(len(kept), ncol, figsize=(2.0 * ncol, 2.45 * len(kept)),
                             squeeze=False)
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("#cccccc")
    im = None
    for ri, row in enumerate(kept):
        for ci in range(ncol):
            ax = axes[ri][ci]
            ax.axis("off")
            if ci >= len(row):
                continue
            slug = row[ci]
            im = ax.imshow(load_axial(input_dir / f"{slug}.bin"), cmap=cmap,
                           vmin=window[0], vmax=window[1], origin="lower")
            is_truth = "truth" in slug
            label = MONTAGE_LABELS.get(slug, NAMES.get(slug, slug))
            ax.set_title(label, fontsize=9,
                         fontweight="bold" if is_truth else "normal",
                         color="#0a7d3a" if is_truth else "black")
        # Only label a row when the group changes, so a group spanning several rows reads
        # as one block instead of repeating its name down the side.
        if kept_labels[ri] and (ri == 0 or kept_labels[ri] != kept_labels[ri - 1]):
            axes[ri][0].text(-0.09, 0.5, kept_labels[ri], transform=axes[ri][0].transAxes,
                             rotation=90, va="center", ha="center", fontsize=9.5,
                             fontweight="bold", color="#444444")
    fig.suptitle(f"{title} — centre axial slice", fontsize=13, fontweight="bold")
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
    rendered = set(MONTAGED)
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

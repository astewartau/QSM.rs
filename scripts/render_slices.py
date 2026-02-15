#!/usr/bin/env python3
"""Render center slices from binary files produced by QSM-core integration tests.

Each .bin file contains center axial, coronal, and sagittal slices of a 3D
result volume plus corresponding mask slices. This script reads them and
produces a 3-panel matplotlib figure per algorithm.

Usage:
    python render_slices.py <input_dir> <output_dir>
"""

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
    "bet": "BET",
    "combined_tgv": "TGV (Combined)",
    "bias_correction": "Bias Correction",
    "pipeline_tgv": "TGV",
    "pipeline_qsmart": "QSMART",
}

# Fixed display windows (ppm)
WINDOWS = {
    "bgremove_sharp": (-0.025, 0.025),
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
    "bet": (0, 1),
    "combined_tgv": (-0.1, 0.1),
    "bias_correction": (-0.1, 0.1),  # fallback; before/after rendering uses auto-range
    "pipeline_tgv": (-0.1, 0.1),
    "pipeline_qsmart": (-0.1, 0.1),
}


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
    cb_label = "Mask" if slug == "bet" else "ppm"
    fig.colorbar(im, ax=axes, shrink=0.85, aspect=30, pad=0.02, label=cb_label)
    fig.savefig(output_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Rendered {output_path}")


def render_before_after(before_slices, after_slices, name, output_path):
    """Render a 2-panel before/after axial figure."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Shared auto-range across both panels
    all_vals = np.concatenate([
        before_slices["axial"].ravel(),
        after_slices["axial"].ravel(),
    ])
    finite = all_vals[np.isfinite(all_vals)]
    vmin, vmax = (float(finite.min()), float(finite.max())) if len(finite) > 0 else (0, 1)

    axes[0].imshow(before_slices["axial"], cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    axes[0].set_title("Before", fontsize=11)
    axes[0].axis("off")

    im = axes[1].imshow(after_slices["axial"], cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    axes[1].set_title("After", fontsize=11)
    axes[1].axis("off")

    fig.suptitle(name, fontsize=14, fontweight="bold", y=1.0)
    fig.colorbar(im, ax=axes, shrink=0.85, aspect=30, pad=0.02, label="Intensity")
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
    }

    print(f"Rendering figures from {len(bin_files)} files...")
    rendered = set()
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

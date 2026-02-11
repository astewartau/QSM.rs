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

        mask_ax = np.frombuffer(f.read(nx * ny), dtype=np.uint8).reshape(ny, nx)
        mask_cor = np.frombuffer(f.read(nx * nz), dtype=np.uint8).reshape(nz, nx)
        mask_sag = np.frombuffer(f.read(ny * nz), dtype=np.uint8).reshape(nz, ny)

    return {
        "axial": np.where(mask_ax > 0, axial, np.nan),
        "coronal": np.where(mask_cor > 0, coronal, np.nan),
        "sagittal": np.where(mask_sag > 0, sagittal, np.nan),
    }


def render_figure(slices, name, output_path):
    """Render a 3-panel figure (axial, coronal, sagittal) and save as PNG."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    # Color limits from the 1st and 99th percentiles
    all_vals = np.concatenate([
        slices["axial"][np.isfinite(slices["axial"])],
        slices["coronal"][np.isfinite(slices["coronal"])],
        slices["sagittal"][np.isfinite(slices["sagittal"])],
    ])
    vmin = np.percentile(all_vals, 1)
    vmax = np.percentile(all_vals, 99)

    cmap = plt.cm.gray.copy()
    cmap.set_bad("black")

    for ax, (label, key) in zip(
        axes, [("Axial", "axial"), ("Coronal", "coronal"), ("Sagittal", "sagittal")]
    ):
        im = ax.imshow(slices[key], cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(label, fontsize=11)
        ax.axis("off")

    fig.suptitle(name, fontsize=14, fontweight="bold", y=1.0)
    fig.colorbar(im, ax=axes, shrink=0.85, aspect=30, pad=0.02, label="ppm")
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

    print(f"Rendering {len(bin_files)} figures...")
    for bin_file in bin_files:
        slug = bin_file.stem
        name = NAMES.get(slug, slug)
        try:
            slices = load_slices(bin_file)
            render_figure(slices, name, output_dir / f"{slug}.png")
        except Exception as e:
            print(f"  WARNING: Failed to render {slug}: {e}")


if __name__ == "__main__":
    main()

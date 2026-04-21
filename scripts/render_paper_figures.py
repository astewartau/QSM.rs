#!/usr/bin/env python3
"""Generate publication-quality composite figures for the QSMbly paper.

Reads center-slice .bin files from QSM.rs integration tests and produces
multi-panel EPS/PNG figures suitable for MRM submission.

Usage:
    python render_paper_figures.py [input_dir] [output_dir]
"""

import struct
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_slices(path):
    """Load center slices from a binary file."""
    with open(path, "rb") as f:
        nx, ny, nz = struct.unpack("<QQQ", f.read(24))
        axial = np.frombuffer(f.read(nx * ny * 8), dtype="<f8").reshape(ny, nx)
        coronal = np.frombuffer(f.read(nx * nz * 8), dtype="<f8").reshape(nz, nx)
        sagittal = np.frombuffer(f.read(ny * nz * 8), dtype="<f8").reshape(nz, ny)
        # mask slices
        mask_ax = np.frombuffer(f.read(nx * ny), dtype=np.uint8).reshape(ny, nx)
        mask_cor = np.frombuffer(f.read(nx * nz), dtype=np.uint8).reshape(nz, nx)
        mask_sag = np.frombuffer(f.read(ny * nz), dtype=np.uint8).reshape(nz, ny)
    return {
        "axial": axial, "coronal": coronal, "sagittal": sagittal,
        "mask_axial": mask_ax, "mask_coronal": mask_cor, "mask_sagittal": mask_sag,
    }


def render_bgremove_comparison(input_dir, output_dir):
    """Figure: Background field removal — ground truth + all methods in one row."""
    # Ground truth first, then methods
    gt_slug = "ground_truth_local_field"
    methods = [
        ("bgremove_sharp", "SHARP"),
        ("bgremove_vsharp", "V-SHARP"),
        ("bgremove_pdf", "PDF"),
        ("bgremove_lbv", "LBV"),
        ("bgremove_ismv", "iSMV"),
    ]

    panels = []
    gt_path = input_dir / f"{gt_slug}.bin"
    if gt_path.exists():
        panels.append((gt_slug, "Ground Truth"))
    for s, n in methods:
        if (input_dir / f"{s}.bin").exists():
            panels.append((s, n))

    if len(panels) < 2:
        print("  Skipping BFR figure — insufficient data")
        return

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(2.8 * n, 3.5))
    if n == 1:
        axes = [axes]

    vmin, vmax = -0.025, 0.025
    for ax, (slug, name) in zip(axes, panels):
        slices = load_slices(input_dir / f"{slug}.bin")
        im = ax.imshow(slices["axial"], cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.axis("off")

    fig.colorbar(im, ax=list(axes), shrink=0.85, aspect=30, pad=0.03, label="ppm")
    for fmt in ("eps", "png"):
        fig.savefig(output_dir / f"fig_bgremove.{fmt}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Rendered fig_bgremove (axial, {n} panels)")


def render_inversion_comparison(input_dir, output_dir):
    """Figure: Dipole inversion — ground truth + all methods + TGV + QSMART in 2 rows."""
    gt_slug = "ground_truth_chi"
    methods = [
        ("inversion_tkd", "TKD"),
        ("inversion_tsvd", "TSVD"),
        ("inversion_tikhonov", "Tikhonov"),
        ("inversion_tv", "TV-ADMM"),
        ("inversion_nltv", "NLTV"),
        ("inversion_rts", "RTS"),
        ("inversion_medi", "MEDI"),
        ("combined_tgv", "TGV"),
        ("pipeline_qsmart", "QSMART"),
    ]

    panels = []
    gt_path = input_dir / f"{gt_slug}.bin"
    if gt_path.exists():
        panels.append((gt_slug, "Ground Truth"))
    for s, n in methods:
        if (input_dir / f"{s}.bin").exists():
            panels.append((s, n))

    if len(panels) < 2:
        print("  Skipping inversion figure — insufficient data")
        return

    # Layout: 2 rows, ncols = ceil(n/2)
    ncols = (len(panels) + 1) // 2
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 6.5))

    vmin, vmax = -0.1, 0.1
    im = None
    for i, (slug, name) in enumerate(panels):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        slices = load_slices(input_dir / f"{slug}.bin")
        im = ax.imshow(slices["axial"], cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.axis("off")

    # Hide unused axes
    for i in range(len(panels), nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].axis("off")

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, aspect=30, pad=0.03, label="ppm")
    for fmt in ("eps", "png"):
        fig.savefig(output_dir / f"fig_inversion.{fmt}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Rendered fig_inversion (2 rows, {len(panels)} panels)")


def render_swi_figure(input_dir, output_dir):
    """Figure: SWI and mIP side by side — single-column sized."""
    swi_path = input_dir / "swi.bin"
    mip_path = input_dir / "swi_mip.bin"
    if not swi_path.exists():
        print("  Skipping SWI figure — no data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(5, 3))

    swi = load_slices(swi_path)
    data = swi["axial"]
    finite = data[np.isfinite(data)]
    vmin_s, vmax_s = float(finite.min()), float(finite.max())
    axes[0].imshow(data, cmap="gray", vmin=vmin_s, vmax=vmax_s, origin="lower")
    axes[0].set_title("CLEAR-SWI", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    if mip_path.exists():
        mip = load_slices(mip_path)
        data_m = mip["axial"]
        finite_m = data_m[np.isfinite(data_m)]
        vmin_m, vmax_m = float(finite_m.min()), float(finite_m.max())
        axes[1].imshow(data_m, cmap="gray", vmin=vmin_m, vmax=vmax_m, origin="lower")
        axes[1].set_title("CLEAR-SWI mIP", fontsize=10, fontweight="bold")
        axes[1].axis("off")
    else:
        axes[1].axis("off")

    fig.tight_layout()
    for fmt in ("eps", "png"):
        fig.savefig(output_dir / f"fig_swi.{fmt}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Rendered fig_swi (single-column, SWI + mIP)")


def main():
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("slices")
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("../paper")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating paper figures from {input_dir} -> {output_dir}")
    render_bgremove_comparison(input_dir, output_dir)
    render_inversion_comparison(input_dir, output_dir)
    render_swi_figure(input_dir, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()

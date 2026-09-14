#!/usr/bin/env python3
"""Figure 2 --- what the method does, stage by stage, and why each stage exists.

Single-column vertical flow. Colour separates the two things the method is doing:
estimating microstructure (blue) and separating sources (orange).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C_EST, C_SEP = "#0072B2", "#D55E00"
INK, MUTED, GRID, PANEL = "#1a1a1a", "#5c5c5c", "#d0d0d0", "#f4f4f1"

STAGES = [
    (1, C_SEP, r"Calibrate $D_r^+$ from the data",
     "Among iron-dominated voxels the smallest $R_2'/\\chi_{\\mathrm{total}}$ is the\n"
     "iron-only ratio, because any myelin inflates it."),
    (2, C_SEP, "Solve the conventional two-source system",
     "The textbook closed form, evaluated everywhere. This is\n"
     "the answer outside white matter, and the fallback inside it."),
    (3, C_EST, "Find myelinated white matter, from the signal",
     "Fit the pool model and a plain exponential; whichever wins\n"
     "says what the voxel is. The wobble is below single-voxel\n"
     "noise, so this is done on lightly smoothed data. Output is a\n"
     "soft weight \u2014 no segmentation, atlas or user input."),
    (4, C_EST, r"Fit fibre angle $\theta$ and myelin water fraction",
     "Search a precomputed library of decay curves, one per\n"
     "$(\\theta,\\,\\mathrm{MWF})$ pair. Voxels are grouped by their $R_2'$, so each\n"
     "group is scored with a single matrix product."),
    (5, C_SEP, "Separate into iron and myelin",
     "White matter has no iron dephasing term to invert, so route it\n"
     "through myelin content: $|\\chi^-| = K_\\chi\\,\\mathrm{MWF}$, then $\\chi^+$ from\n"
     "$\\chi_{\\mathrm{total}}$. Blend with stage 2 by the soft weight."),
]

H = [11.5, 11.5, 18.0, 15.0, 15.0]          # per-stage box heights
GAPY, TOP = 5.4, 122.0

fig = plt.figure(figsize=(3.42, 5.35))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 143); ax.axis("off")

def box(x, y, w, h, fc, ec, r=1.4, lw=0.8, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))

def arrow(x0, y0, x1, y1, color=MUTED, lw=0.9):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6,
                                 lw=lw, color=color, zorder=1, shrinkA=0, shrinkB=0))

X, W = 9.0, 84.0

# ---- inputs ---------------------------------------------------------------
ax.text(X, 140.5, "INPUTS", fontsize=5.6, color=MUTED, weight="bold", va="top")
chips = [r"QSM $\chi_{\mathrm{total}}$", r"$R_2'$", "multi-echo GRE", "mask", "spin echo (opt.)"]
cw = [17.5, 9.0, 24.0, 10.5, 23.0]
cx = X
for name, w in zip(chips, cw):
    box(cx, 128.5, w, 6.0, PANEL, GRID, r=1.0, lw=0.6)
    ax.text(cx + w / 2, 131.5, name, fontsize=5.4, color=INK, ha="center", va="center", zorder=3)
    cx += w + 1.6
arrow(X + W / 2, 128.2, X + W / 2, TOP + 0.4)

# ---- stages ---------------------------------------------------------------
y = TOP
tops = []
for (num, c, title, why), h in zip(STAGES, H):
    y -= h
    tops.append(y + h)
    box(X, y, W, h, "white", c, lw=0.9)
    box(X, y + h - 6.4, W, 6.4, c, c, lw=0.9)
    ax.add_patch(plt.Rectangle((X, y + h - 6.4), W, 3.2, fc=c, ec="none", zorder=2))
    ax.text(X + 2.6, y + h - 3.2, str(num), fontsize=6.6, color="white", weight="bold",
            ha="center", va="center", zorder=4)
    ax.text(X + 6.0, y + h - 3.2, title, fontsize=6.0, color="white", weight="bold",
            ha="left", va="center", zorder=4)
    ax.text(X + 3.0, y + (h - 6.4) / 2, why, fontsize=5.3, color=INK, ha="left",
            va="center", linespacing=1.55, zorder=4)
    if y > 20:
        arrow(X + W / 2, y - 0.3, X + W / 2, y - GAPY + 0.6)
    y -= GAPY

# ---- the anchor, as the link from R2' into stage 4 -------------------------
y4 = tops[3] - H[3] / 2
ax.plot([32.6, X + W + 0.8], [126.9, 126.9], lw=1.0, color=C_EST, zorder=1,
        solid_capstyle="round")
ax.add_patch(FancyArrowPatch((X + W + 0.8, 126.9), (X + W + 0.8, y4),
                             connectionstyle="arc3,rad=-0.10", arrowstyle="-|>",
                             mutation_scale=6, lw=1.0, color=C_EST, zorder=1))
ax.text(98.6, (126.9 + y4) / 2, "the anchor", fontsize=5.6, color=C_EST, weight="bold",
        rotation=-90, ha="center", va="center")

# ---- outputs --------------------------------------------------------------
ybot = y + GAPY
arrow(X + W / 2, ybot - 0.3, X + W / 2, 19.4)
ax.text(X, 6.6, "OUTPUTS", fontsize=5.6, color=MUTED, weight="bold", va="bottom")
outs = [(r"$\chi^+$ iron", C_SEP, 16.0), (r"$\chi^-$ myelin", C_SEP, 20.0),
        (r"$\theta$", C_EST, 8.0), ("MWF", C_EST, 13.0)]
cx = X
for name, c, w in outs:
    box(cx, 12.4, w, 6.0, PANEL, c, r=1.0, lw=0.7)
    ax.text(cx + w / 2, 15.4, name, fontsize=5.6, color=INK, ha="center", va="center", zorder=3)
    cx += w + 1.8

# ---- legend ---------------------------------------------------------------
ax.add_patch(plt.Rectangle((X + 62.0, 5.2), 3.0, 2.6, fc=C_EST, ec="none"))
ax.text(X + 66.4, 6.5, "estimator", fontsize=5.2, color=INK, va="center")
ax.add_patch(plt.Rectangle((X + 62.0, 1.4), 3.0, 2.6, fc=C_SEP, ec="none"))
ax.text(X + 66.4, 2.7, "separation", fontsize=5.2, color=INK, va="center")

out = __file__.rsplit("/scripts/", 1)[0] + "/figures/fig2_pipeline"
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300)
print("wrote", out + ".pdf")

#!/usr/bin/env python3
"""Figure 1 --- the concept figure for the HC-ChiSep manuscript.

Everything plotted here is computed from the same hollow-cylinder model the method
uses (Wharton & Bowtell, PNAS 2012), with the constants of Table 1, so the figure is
the model rather than a cartoon of it. Panel (a) alone is drawn.

    python3 scripts/fig1_concept.py      ->  figures/fig1_concept.pdf (+ .png)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, FancyArrowPatch, Arc

# --- model constants (identical to src/separation/hc_chisep.rs) --------------
GAMMA_BAR = 42.577e6      # Hz/T
CHI_I, CHI_A, E_EXCH = -0.06e-6, -0.10e-6, 0.02e-6
G_RATIO = 0.7
T2_M, T2_A, T2_E = 10e-3, 64e-3, 48e-3
F_AXON = 0.55
MWF_REF = 0.12
B0 = 7.0
TES = np.arange(3, 46, 6) * 1e-3      # 8 echoes, 3-45 ms
SNR = 100.0

# --- colours: Okabe-Ito, validated colourblind-safe -------------------------
C0, C45, C90 = "#0072B2", "#009E73", "#D55E00"
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#d4d4d4"


def compartment_freqs(theta_rad, b0=B0):
    """(myelin, axonal) frequency offsets in Hz; extra-axonal is 0 by construction."""
    s2 = np.sin(theta_rad) ** 2
    w0 = GAMMA_BAR * b0
    ln_term = 0.75 * CHI_A * np.log(1.0 / G_RATIO) * s2
    f_my = w0 * (CHI_I * (2.0 / 3.0 - s2) / 2.0
                 + CHI_A * (1.0 / 12.0 - 5.0 / 12.0 * s2) + ln_term + E_EXCH)
    f_ax = w0 * ln_term
    return f_my, f_ax


def wm_signal_mag(te, theta_rad, mwf=MWF_REF, b0=B0):
    """|S(TE)| of the three-pool hollow-cylinder model (no mesoscopic term)."""
    f_m = mwf
    rest = 1.0 - mwf
    pools = [(f_m, T2_M), (rest * F_AXON, T2_A), (rest * (1 - F_AXON), T2_E)]
    dfm, dfa = compartment_freqs(theta_rad, b0)
    dfs = [dfm, dfa, 0.0]
    z = sum(f * np.exp(-te / t2) * np.exp(2j * np.pi * df * te)
            for (f, t2), df in zip(pools, dfs))
    return np.abs(z)


def monoexp_residual(theta_deg):
    """RMS residual of the best log-linear mono-exponential fit to the normalised decay."""
    th = np.deg2rad(theta_deg)
    s = wm_signal_mag(TES, th)
    s = s / s[0]
    A = np.vstack([np.ones_like(TES), TES]).T
    coef, *_ = np.linalg.lstsq(A, np.log(s), rcond=None)
    return float(np.sqrt(np.mean((s - np.exp(A @ coef)) ** 2)))


# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(7.1, 2.75))
gs = fig.add_gridspec(1, 4, width_ratios=[1.02, 1.0, 1.34, 0.90],
                      wspace=0.52, left=0.030, right=0.988, bottom=0.175, top=0.865)

# --- (a) geometry -----------------------------------------------------------
ax = fig.add_subplot(gs[0])
ax.set_aspect("equal"); ax.axis("off")
ax.set_xlim(-1.55, 1.55); ax.set_ylim(-1.35, 1.45)

R_OUT, R_IN = 0.80, 0.80 * G_RATIO
ax.add_patch(Circle((0, 0), 1.42, fc="#f2f2ef", ec="none", zorder=0))
ax.add_patch(Wedge((0, 0), R_OUT, 0, 360, width=R_OUT - R_IN,
                   fc=C90, ec="none", alpha=0.85, zorder=2))
ax.add_patch(Circle((0, 0), R_IN, fc=C0, ec="none", alpha=0.80, zorder=3))

ax.annotate("myelin water\n$T_2$ 10 ms", xy=(0.62, 0.62), xytext=(1.02, 1.10),
            fontsize=6.0, color=INK, ha="center", va="bottom",
            arrowprops=dict(arrowstyle="-", lw=0.6, color=MUTED))
ax.text(0, 0, "axonal\n$T_2$ 64 ms", fontsize=6.0, color="white",
        ha="center", va="center", zorder=4)
ax.annotate("extra-axonal\n$T_2$ 48 ms", xy=(-1.02, -0.72), xytext=(-1.20, -1.28),
            fontsize=6.0, color=INK, ha="center", va="bottom",
            arrowprops=dict(arrowstyle="-", lw=0.6, color=MUTED))

# B0 and the fibre axis, with theta between them
ax.add_patch(FancyArrowPatch((-1.42, -1.15), (-1.42, 1.28), lw=1.1, color=INK,
                             arrowstyle="-|>", mutation_scale=7, zorder=5))
ax.text(-1.42, 1.34, "$B_0$", fontsize=7.0, color=INK, ha="center", va="bottom")
ax.plot([-1.42 + 0.62 * np.sin(np.deg2rad(38)) * -1, -1.42], [0.62 * np.cos(np.deg2rad(38)), 0],
        lw=1.1, color=MUTED, ls=(0, (3, 2)), zorder=5)
ax.add_patch(FancyArrowPatch((-1.42, 0), (-1.42 + 0.98 * np.sin(np.deg2rad(38)),
                                          0.98 * np.cos(np.deg2rad(38))),
                             lw=1.1, color=MUTED, arrowstyle="-|>", mutation_scale=7, zorder=5))
ax.add_patch(Arc((-1.42, 0), 0.72, 0.72, theta1=52, theta2=90, lw=0.7, color=MUTED, zorder=5))
ax.text(-1.16, 0.44, r"$\theta$", fontsize=7.5, color=INK, ha="center", va="center")
ax.set_title("(a)  myelinated axon", fontsize=7.2, color=INK, loc="left", pad=6)

# --- (b) compartment frequency offsets vs theta -----------------------------
ax = fig.add_subplot(gs[1])
th = np.linspace(0, np.pi / 2, 200)
fm, fa = compartment_freqs(th)
ax.plot(np.degrees(th), fm, lw=1.6, color=C90, label="myelin")
ax.plot(np.degrees(th), fa, lw=1.6, color=C0, label="axonal")
ax.plot(np.degrees(th), np.zeros_like(th), lw=1.6, color=MUTED, ls=(0, (4, 2)),
        label="extra-axonal")
ax.text(46, fm[-1] * 0.97, "myelin", fontsize=6.0, color=C90, ha="center", va="top")
ax.text(46, fa[-1] * 0.72, "axonal", fontsize=6.0, color=C0, ha="center", va="bottom")
ax.text(46, 0.9, "extra-axonal", fontsize=6.0, color=MUTED, ha="center", va="bottom")
ax.set_ylim(fa[-1] * 1.30, fm[-1] * 1.30)
ax.set_xlabel(r"fibre angle $\theta$  (deg)", fontsize=6.8)
ax.set_ylabel("frequency offset  (Hz)", fontsize=6.8)
ax.set_xticks([0, 30, 60, 90])
ax.set_title("(b)  pool frequencies", fontsize=7.2, color=INK, loc="left", pad=6)

# --- (c) the resulting magnitude decay --------------------------------------
ax = fig.add_subplot(gs[2])
te_fine = np.linspace(TES[0], TES[-1], 400)
for deg, c in ((0, C0), (45, C45), (90, C90)):
    s = wm_signal_mag(te_fine, np.deg2rad(deg))
    s = s / wm_signal_mag(TES[0], np.deg2rad(deg))
    ax.plot(te_fine * 1e3, s, lw=1.6, color=c, zorder=3)
    sd = wm_signal_mag(TES, np.deg2rad(deg))
    ax.plot(TES * 1e3, sd / sd[0], "o", ms=3.0, color=c, mec="white", mew=0.5, zorder=4)
    ax.text(TES[-1] * 1e3 + 1.4, s[-1], rf"$\theta={deg}^\circ$", fontsize=6.0,
            color=c, va="center", ha="left")
# mono-exponential reference through the 90 deg endpoints
s90 = wm_signal_mag(TES, np.deg2rad(90)); s90 = s90 / s90[0]
k = -np.log(s90[-1]) / (TES[-1] - TES[0])
ax.plot(te_fine * 1e3, np.exp(-k * (te_fine - TES[0])), lw=1.0, color=INK,
        ls=(0, (2, 2)), zorder=2)
_smin = min(float((wm_signal_mag(te_fine, np.deg2rad(d))
                   / wm_signal_mag(TES[0], np.deg2rad(d))).min()) for d in (0, 45, 90))
ax.annotate("mono-exponential\nreference", xy=(21, np.exp(-k * (21e-3 - TES[0]))),
            xytext=(30.5, 0.93), fontsize=5.9, color=INK, ha="left", va="top",
            arrowprops=dict(arrowstyle="-", lw=0.6, color=MUTED,
                            shrinkA=1, shrinkB=2))
ax.set_xlabel("echo time  (ms)", fontsize=6.8)
ax.set_ylabel("$|S|$  (first echo = 1)", fontsize=6.8)
ax.set_xlim(0, 56); ax.set_ylim(_smin * 0.94, 1.045)
ax.set_title("(c)  magnitude decay", fontsize=7.2, color=INK, loc="left", pad=6)

# inset: departure from mono-exponential vs the per-voxel noise floor
axi = ax.inset_axes([0.085, 0.085, 0.40, 0.355])
degs = np.arange(0, 91, 2.0)
res = np.array([monoexp_residual(d) for d in degs])
axi.axhspan(0, 1.0 / SNR, color=MUTED, alpha=0.20, lw=0)
axi.plot(degs, res * 100, lw=1.2, color=INK)
axi.axhline(100.0 / SNR, lw=0.7, color=MUTED, ls=(0, (2, 2)))
axi.text(4, 100.0 / SNR + 0.28, "noise", fontsize=5.0, color=MUTED, va="bottom")
axi.set_xticks([0, 45, 90]); axi.set_xlim(0, 90)
axi.set_ylim(0, max(res * 100) * 1.25)
axi.tick_params(labelsize=5.0, length=1.8, pad=1)
axi.set_xlabel(r"$\theta$", fontsize=5.4, labelpad=0)
axi.set_ylabel("resid. (%)", fontsize=5.4, labelpad=1)

# --- (d) the same theta in the relaxivity -----------------------------------
ax = fig.add_subplot(gs[3])
drneg = 0.5 * 42.58 * 2 * np.pi * B0 * np.sin(th) ** 2
ax.fill_between(np.degrees(th), 0, drneg, color=C90, alpha=0.16, lw=0)
ax.plot(np.degrees(th), drneg, lw=1.6, color=C90)
ax.set_xlabel(r"fibre angle $\theta$  (deg)", fontsize=6.8)
ax.set_ylabel(r"$D_r^-(\theta)$  (Hz/ppm)", fontsize=6.8)
ax.set_xticks([0, 30, 60, 90])
ax.set_title("(d)  relaxivity", fontsize=7.2, color=INK, loc="left", pad=6)

for a in fig.axes:
    a.tick_params(labelsize=6.0, length=2.2, width=0.6, colors=MUTED, pad=1.6)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        a.spines[sp].set_linewidth(0.6); a.spines[sp].set_color(GRID)
    for lbl in a.get_xticklabels() + a.get_yticklabels():
        lbl.set_color(INK)
    a.xaxis.label.set_color(INK); a.yaxis.label.set_color(INK)

out = __file__.rsplit("/scripts/", 1)[0] + "/figures/fig1_concept"
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300)
print("wrote", out + ".pdf")
print(f"  mono-exp residual RMS: theta=0 {monoexp_residual(0)*100:.2f}%,"
      f" 45 {monoexp_residual(45)*100:.2f}%, 90 {monoexp_residual(90)*100:.2f}%"
      f"   (noise floor 1/SNR = {100/SNR:.2f}%)")
fm90, fa90 = compartment_freqs(np.pi / 2)
print(f"  offsets at 90 deg: myelin {fm90:+.2f} Hz, axonal {fa90:+.2f} Hz")

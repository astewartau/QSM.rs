# Figure briefs

Figures 1 and 2 are generated — `make figures` (or `python3 scripts/fig*.py`) rebuilds them from
the model itself, so the artwork cannot drift from the constants in Table 1. Figures 3–6 are still
placeholder boxes whose text is the brief; the captions in `hc-chisep.tex` are authoritative.

| Fig | Content | Source | Status |
|---|---|---|---|
| 1 | Concept: axon geometry → θ-dependent pool frequencies → non-mono-exponential decay (with the noise-floor inset) → Dr−(θ) | `scripts/fig1_concept.py` | **done** |
| 2 | Pipeline: five stages, what each is for, inputs/outputs, and the anchor | `scripts/fig2_pipeline.py` | **done** |
| 3 | SSE landscape, unanchored vs anchored | `qsmci/reports/2026-08-chisep/nonoracle/sse_landscape.png` (left panel only) | **anchored panel must be generated** |
| 4 | θ error and interference residual vs field | `qsmci/reports/2026-08-chisep/b0_study/{theta_mae.png,beat_noise.png}` (+ `summary.npz`, `run_study.py`) | replot for print |
| 5 | θ / MWF / weight maps and scatters | `nonoracle/{maps.png,theta_scatter.png}` for the blind arm; anchored arm must be re-rendered | partly exists |
| 6 | Separation maps vs comparators | was `/tmp/hc-chisep-work/contact_sheet.png` — **scratch, regenerate** | to regenerate |
| 7 | In vivo | — | pending data |

Colours are the Okabe-Ito trio `#0072B2 / #009E73 / #D55E00`, checked for colourblind separation
and print contrast; series are also direct-labelled, so identity never rests on colour alone. Reuse
these three in Figures 3–6 so the set reads as one system.

Everything a figure depends on must be reproducible from a committed script; the scratch outputs
under `/tmp/hc-chisep-work/` will not survive and must be regenerated into
`qsmci/reports/2026-08-chisep/hc-chisep/`.

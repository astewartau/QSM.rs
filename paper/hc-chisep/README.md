# HC-ChiSep manuscript

- `PLAN.md` — the writing plan: claim set, section-by-section argument, evidence inventory
  (every number traced to the file it came from), and the list of experiments still outstanding.
- `hc-chisep.tex` — the draft. `hc-chisep.bib` — the bibliography.
- `figures/` — artwork; see `figures/README.md` for the spec and status of each panel.
- `scripts/` — figure generators. `make figures` rebuilds Figures 1 and 2 from the model itself,
  so the artwork cannot drift from the constants in Table 1.

## Build

```bash
make          # -> hc-chisep.pdf
make final    # same, with draft markers hidden
make clean
```

Draft markers: `\TODO{...}` (red, things to write or verify), `\NUM{...}` (red, a number that
must be filled or checked), `\NOTE{...}` (blue, an editorial decision to make). All three vanish
when `\DRAFTtrue` is switched to `\DRAFTfalse`, which is what `make final` does temporarily.

## Vendored template

`MRM.cls`, `MRM-AMA.bst`, `NJDnatbib.sty`, `widetext.sty`, `ama.bst`, `Orcidlogo.eps`,
`empty.eps` are the Wiley MRM LaTeX template, copied from `~/repos/qsm/qsmbly/paper/`.

Two local deviations, both deliberate:

1. `MRM.cls` line ~3040 wraps `\usepackage{algorithm, algorithmicx, algpseudocode}` in an
   `\IfFileExists` guard. Those packages are absent from some TeX Live installations (including
   this machine's) and the class does not use them.
2. The author line does not use `\orcid{T}`, because the ORCID logo is an EPS and pulling it in
   requires `--shell-escape`. Restore it when submitting.

## State of the draft

Prose is complete for Introduction, Theory, Methods (except in vivo), Results (except in vivo),
Discussion and Conclusion. What is deliberately unwritten:

- the in vivo Methods and Results subsections, and the in vivo figure;
- the comparator rows of Table 4, pending a re-score of the remaining QSM-CI entries on this
  build of the phantom;
- author list, funding, acknowledgments, data-availability statement;
- several references flagged `TODO`/`VERIFY` in `hc-chisep.bib`.

Figures 1 (concept) and 2 (pipeline) are finished and generated from `scripts/`. Figures 3–6 are
placeholder boxes containing the specification of the panel; their captions are written and are
meant to be read as the brief for the artwork.

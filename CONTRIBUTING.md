# Contributing to QSM.rs

QSM.rs (`qsm-core`) is the Rust library that holds every QSM algorithm used by QSMxT and qsmbly:
from brain extraction to dipole inversion, source separation and deep-learning inference. It
runs natively and in WebAssembly.

## Where code belongs

| Repository | Owns |
|---|---|
| **QSM.rs** | Anything that inspects or transforms image data; algorithm defaults, deviations and the pipeline rules. |
| [QSMxT](https://github.com/QSMxT/QSMxT) | Wiring: config, plan-time checks a user sees, how steps link, CLI/TUI, BIDS, provenance, methods text. Simple numerics (ROI statistics, unit conversions). No algorithms. |
| [qsmbly](https://github.com/astewartau/qsmbly) | The browser UI. Runs QSM.rs through QSMxT's `qsmxt-engine`. |

If QSMxT or qsmbly needs behaviour that QSM.rs lacks or gets wrong, fix it here and they move
their pin. No downstream workarounds. Every issue names its owner (`owner: QSM.rs` or
`owner: QSMxT`); if both need changes, file it here and note the QSMxT follow-up.

## Principles

1. **Ports are the original algorithm.** A port reproduces the authors' paper *and* their code.
   Which reference to follow:
   - the authors' published code, in any language;
   - otherwise one designated implementation per family (e.g. QSM.jl for SHARP, iSMV, TKD, TSVD,
     Tikhonov, TV, RTS, NLTV), with the alternatives' values noted in the module docs.

   Third-party reimplementations are not the original. If information is missing, ask the
   authors rather than invent a substitute.
2. **Algorithm vs tuning.** A different solver, model or pipeline structure is a different
   algorithm. A different tolerance, iteration count or threshold is tuning. The algorithm must
   match the original, and with matched parameters so must the output.
3. **Deviations.** An unequivocal improvement (e.g. padding odd axes so a kernel does not
   diverge) becomes the default, and the original behaviour stays available as an option for
   parity. Where the original has a bug or a weaker choice, the better-reasoned behaviour is the
   default. A trade-off is an option, not the default. Performance work must not change what
   the algorithm is.
4. **Take the reference's inputs.** If the reference uses magnitude, per-echo data, noise maps or
   scan parameters, require them; do not make them optional with a weaker fallback.
5. **Defaults are the reference's** (what users of the reference actually run), unless a
   documented measurement justifies another value.
6. **No physically meaningless negatives.** R2\*, T2\*, R2, T2, R2′, S0 and similar maps are
   clamped at 0, even where the reference returns noise negatives.
7. **Every algorithm decision is in the code.** The default and why, its source in the
   reference, each deviation, the measurement behind it and the options kept go in that
   method's doc comments, not only in a PR body or issue. QSMxT and qsmbly do not repeat these
   details; they point here.
8. **Get the API right; no backwards compatibility.** Break callers when the right API needs it.
   No deprecated aliases or compatibility wrappers. Output meant to stay the same is still
   checked (bit-identical or a parity test).
9. **Same bits on every target.** Native and WASM give the same output.
10. **Judge by XSIM first, in context.** A marginal XSIM gain with a much worse NRMSE can be
    worse overall; report both.

## Code conventions

- One module per stage (`bet`, `unwrap`, `bgremove`, `inversion`, `separation`, `swi`, `utils`,
  `models`, `pipeline`), one file per method. A method a host can choose gets a variant in its
  stage's algorithm enum with a kebab-case `id()`.
- Each method has a `*Params` struct whose `Default` holds the reference values. Every field's
  doc gives its unit, default and the default's source (reference file and line).
- Units: seconds, tesla, ppm at the public boundary, mm unless named `*_voxels`. Put the unit in
  the name where it could be misread (`pad_mm`, `threshold_ppm`). Constants come from
  `crate::constants`; a port whose reference hard-codes another value keeps it locally, with a
  comment.
- Bad input returns `InputError` (`InputError::param`, `error::check_len`, a `validate()` on the
  params), never a panic.
- Algorithm paths use no `Instant`, threads or file system (they panic on wasm32). Values that
  feed a discrete decision use `utils::portable_math`. Parallel reductions are deterministic.
- Deep-learning weights are never vendored: each model is a `ModelSpec` in
  `src/models/registry.rs` with URL and SHA-256.
- Dependencies are pure Rust and build for wasm32; licences go in `THIRD_PARTY_NOTICES.md`.
- The code is not rustfmt-clean. Do not reformat code you are not changing. Clippy with
  `-D warnings` is a CI gate.

## Comments and documentation

A port's module docs say, in this order: what is implemented and from which reference (toolbox,
version or commit, and the call); the algorithm, and where the paper and code differ; units;
agreement with the reference (what was compared, the numbers, which test); deviations and why;
`# Reference pipeline` (what the authors run around the method, with sources);
`Reference:` (the papers, with DOI); `# Pipeline rules` (rule ids).

- Cite the paper by section or equation and the code by file and line, with the version.
- Measured claims give numbers and the data. Cite only public sources and data.
- Inline comments say *why*, or which reference line the code follows, not what it does.
- The README is general usage and the algorithm tables only. No changelog or release-notes
  files: change summaries go in PR bodies, release notes on the GitHub release.

## Pipeline rules

`src/pipeline/rules.rs` publishes what each method requires or recommends, for hosts to show.
Its module docs are the specification. In short:

- **No source, no rule.** A rule states what the reference pipeline does (with file and line or
  paper section) or what QSM.rs measured (recorded in the method's docs). Never a preference.
- **Strengths:** `Required` (cannot run correctly otherwise), `Recommended` (a stated or
  measured reason), `Discouraged` (known to fail), all preselected by hosts; `Suggested` (a
  reason that holds on some data only, or an author's reason the measurements do not clearly
  confirm) and `Reference` (the authors' setting, no reason given), never preselected.
- The first condition is the choice that triggers the rule. Every rule also appears in the
  method's docs, and gets a test case.
- Rules never change settings themselves, and hosts never change a setting silently.

## Tests

QSM.rs is where algorithm correctness is tested; QSMxT and qsmbly test only their wiring.

- Unit tests sit in the file they test, named as sentences
  (`param_error_names_parameter_value_and_range`). Every `InputError` path gets a test.
- Reference parity tests (`tests/<method>_<reference>_parity.rs`) compare a port with the
  authors' own output; the test documents the fixture and the exact reference call.
- Integration tests run each method on the QSM-CI phantom (`#[ignore]`, `test_<stage>_<method>`)
  and must be registered in `.github/workflows/integration-tests.yml` (test list and path map)
  and the figure scripts. `tests/wasm_pipeline.rs` gets a case for each new method.
- Never delete or weaken an assertion to make a change pass; explain a changed reference value.
- Locally, build, run clippy and run only the new or changed tests (quick ones). Push early and
  let CI run the full suite, the `#[ignore]`d integration tests and any repeats.

## Pull requests

- Branch from current `origin/main` (`feat/`, `fix/`, `perf/`, `docs/`, `ci/`, `chore/`), one
  worktree and target directory per branch. Never commit to `main`.
- **Title**: a plain sentence led by the method or area, e.g. "STAR-QSM: default to the STI
  Suite GUI's padding".
- **Body**: what changed and its effect at default settings; breaking API changes; measured
  impact (XSIM first, and on what data); decisions and reasons (also in the code); paper vs
  code; tests run; downstream follow-ups for QSMxT/qsmbly; `Fixes #n`.
- Open as a draft. A PR is ready only when CI is green. Once it is marked ready, put further
  changes in a new PR unless the reviewer agrees.
- Stacked PRs: retarget children to `main` before the parent is merged.
- The maintainer merges, by squash.

## Releases

`Cargo.toml` stays at `0.0.0`; `release.yml` sets the version from the tag. Releases are made
only with the maintainer's go, and follow [`docs/RELEASING.md`](docs/RELEASING.md): CI green,
parity checked for every pair whose computation changed, notes drafted from the merged PR
bodies. Prefer one complete release over a quick follow-up or patch release.

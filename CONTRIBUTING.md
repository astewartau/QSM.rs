# Contributing to QSM.rs

QSM.rs (`qsm-core`) is a Rust library of QSM algorithms: brain extraction, phase unwrapping,
field mapping, background field removal, dipole inversion, source separation, SWI and the
utilities around them. It runs natively and in WebAssembly. This document is the standard for
code, comments, tests and process in this repository. Read it before opening a PR.

## Related repositories

QSM.rs is the bottom of a three-layer chain:

| Repository | Role | Belongs there |
|---|---|---|
| **QSM.rs** (this one) | The algorithms | Anything that inspects or transforms image data (unwrapping, echo combination, field mapping, background removal, inversion, masking, resampling, separation, deep-learning inference). Algorithm defaults, port details, deviations from the reference, and the pipeline rules. |
| [QSMxT](https://github.com/QSMxT/QSMxT) (`qsmxt-engine`, CLI, TUI) | Wires QSM.rs into a BIDS pipeline | Config, parameter checks a user should see before a run, how steps link together, CLI/TUI, BIDS input/output, provenance, methods text. Simple numerics such as ROI statistics and unit conversions. No algorithms of its own. |
| [qsmbly](https://github.com/astewartau/qsmbly) | The same pipeline in the browser | UI only. It runs QSM.rs (WASM) through `qsmxt-engine`, so it gets QSMxT's behaviour by sharing code, not by copying it. |

If QSMxT or qsmbly needs behaviour that QSM.rs lacks or gets wrong, fix it here and let them
move their pin. Do not add a downstream workaround.

## Principles

1. **Ports are the original algorithm.** A ported method reproduces the authors' paper *and*
   their code where it exists. No extensions, no substitute inputs, no invented parameters.
   Third-party reimplementations are not the original: where one differs from the paper, get the
   authors' code. If information is missing (an unreleased solver, an unstated normalisation),
   stop and ask rather than invent a substitute. *Why:* a benchmark or a pipeline that cites a
   method must be running that method.
2. **Take the inputs the reference takes.** If the reference uses magnitude, per-echo data, a
   noise map or scan parameters, require them. Do not make them optional with a weaker fallback.
   *Why:* we must always be able to reproduce (or beat) the reference's result.
3. **Defaults are the reference values.** A method's `Default` is what its authors' pipeline
   runs. Where a toolbox has two references (a function default and a GUI/CLI), use what users of
   the reference actually run, and say which. *Why:* "default" should mean "as published".
4. **Deviations only when they are unequivocal improvements.** An improvement with no downside
   (e.g. padding odd axes so a kernel does not diverge) goes in by default and is documented. A
   trade-off, or a benefit that was not measured, keeps the reference behaviour as the default and
   offers the change as an option.
5. **No physically meaningless negatives.** R2\*, T2\*, R2, T2, R2′, S0 and similar maps are
   clamped at 0 even where the reference returns negatives from noise. Document the deviation.
6. **Every algorithm decision is in the code.** The chosen default and why, its source in the
   reference, each deviation, the measurement that motivated it, and the options kept for the
   alternative go in that method's doc comments. A PR body or an issue thread is not enough.
   *Why:* the code is the only record that survives; QSMxT and qsmbly do not repeat these
   details, they point here.
7. **Get the API right; no backwards compatibility.** Break callers when the right API needs
   it. No deprecated aliases, wrapper functions or extra arguments kept only for old callers.
   Numerical output that is meant to stay the same must still be checked (bit-identical or a
   parity test). *Why:* QSMxT and qsmbly pin a revision and catch up.
8. **The same bits on every target.** Native and WASM must give the same output from the same
   input. *Why:* `qsmxt run` and the browser must agree.
9. **Judge reconstructions by XSIM first, in context.** XSIM leads, but a marginal XSIM gain
   with a much worse NRMSE (or correlation, DGM linearity) can be worse overall. Report all of
   them.

## Code conventions

### Layout

- One module per stage: `bet`, `unwrap`, `bgremove`, `inversion`, `separation`, `segment`,
  `swi`, `utils` (shared numerics and smaller methods), `kernels` and `solvers` (shared maths),
  `models` (deep-learning registry, weights, ONNX), `pipeline` (the high-level API).
- One file per method (`src/inversion/star.rs`). A stage's `mod.rs` holds declarations and
  re-exports, and little else.
- The **low-level API** is a plain function per algorithm, taking the data as slices, a `&Grid`,
  a `&*Params` struct and, for iterative methods, a progress callback `FnMut(usize, usize)`.
- The **high-level API** is `pipeline`: `ScanMetadata`, `QsmPipelineConfig` and its stage
  configs, and the `run_*` dispatchers. A new method that a host should be able to choose gets a
  variant in the stage's algorithm enum (`InversionAlgorithm`, `BgRemovalAlgorithm`, ...), with a
  kebab-case `id()`, and a case in the dispatcher.
- Volumes are flat `Vec<f64>`/`&[f64]` in Fortran order (x fastest), masks are `u8`.

### Naming and parameters

- Each method has a `*Params` struct (`StarParams`, `VsharpParams`) that implements `Default`
  with the reference values and derives `Serialize` under the `introspection` feature
  (`#[cfg_attr(feature = "introspection", derive(serde::Serialize))]`).
- Every parameter field has a doc comment that gives its unit, its default and where the default
  comes from (the reference's file and line, or its GUI), e.g. "Maximum SMV kernel radius in mm
  (default: 12.0, STI Suite's `V_SHARP` `smvsize` default)".
- Put the unit in the name where the value could be read in another unit:
  `phase_offset_sigma_mm`, `pad_mm`, `threshold_ppm`, `rotation_deg`, `*_bytes`.
- Units across the crate: echo times in seconds, field strength in tesla, fields and χ in ppm
  at the public boundary (a method that works in radians converts inside, at the echo time the
  reference uses), lengths in mm unless named `*_voxels` or documented as voxels.
- Physical constants come from `crate::constants` (γ̄ = 42.576 MHz/T). A port whose reference
  hard-codes a different constant keeps that value locally and says so in a comment, so it
  matches the reference.

### Errors

- An algorithm returns `Result<_, InputError>` for input it cannot process (too few echoes,
  wrong buffer lengths, out-of-range parameters) instead of panicking. The message says what was
  wrong and with what values, so a host can show it to a user unchanged.
- Out-of-range parameters use `InputError::param(name, value, allowed)`, which reads
  `"{name} = {value} is out of range: must be {allowed}"`. `name` says whose parameter it is
  (`"V-SHARP min_radius"`), `allowed` gives the range with its unit (`"finite and > 0 mm"`).
  Buffer lengths use `error::check_len`.
- Parameter checks that a host may want to run before any work live on the params struct as
  `fn validate(&self) -> Result<(), InputError>` (plus grid checks such as `check_grid` where
  needed). The algorithm calls the same function, so host and algorithm check the same thing.
  Publish limits a host needs (e.g. patch multiples) as public constants.
- The pipeline stages return `PipelineError`; an `InputError` becomes
  `PipelineError::InvalidInput`.
- These checks are the last line of defence. The user-facing check before a run is QSMxT's.

### Feature flags

| Feature | Use |
|---|---|
| `parallel` | Rayon. Write loops with the `maybe_par_*!` macros (`src/par.rs`) so the code builds with and without it. Parallel reductions must be deterministic: fixed chunks, combined in index order. |
| `simd` | `wide`-based inner loops (`utils::simd_ops`), with a scalar path. |
| `onnx` | Deep-learning methods through `tract`, native and WASM. Gate every DL module and its re-exports with `#[cfg(feature = "onnx")]`. |
| `download` | Native HTTP fetch and SHA-256-checked cache of weights. WASM hosts fetch weights in JavaScript instead, so nothing else may depend on it. |
| `introspection` | `Serialize` on the `*Params` structs, for downstream coverage tests only. |

The default feature set is empty and must build. Examples or tests that need a feature declare
`required-features` in `Cargo.toml`.

### WASM

- Code on any algorithm path must not use `std::time::Instant`, `SystemTime`, threads or the
  file system; they compile for `wasm32-unknown-unknown` and panic at run time.
- The WASM heap ends at 4 GiB. Methods whose peak memory can exceed it publish memory figures
  (see `models` and `NetworkMemory`) and a lower-memory mode, and the limit is a `Required` rule.
- Where a value feeds a discrete decision (a rounding, a threshold, a sign test), compute it with
  `utils::portable_math` (libm), not `f64::exp`/`sin`/`powf`, which differ in the last bit
  between targets.
- `.cargo/config.toml` enables `+simd128` for wasm32 builds in this repository; `models::onnx`
  refuses to compile on wasm32 without it.

### Deep-learning models

- Weights are never vendored. Each model has a `ModelSpec` in `src/models/registry.rs` with its
  paper, source repository, weight URL (Hugging Face), lowercase SHA-256, size, inputs, outputs
  and memory figures. Follow the editing checklist at the top of that file.
- ONNX export scripts live in `scripts/onnx-export/`. Record the export's parity against the
  original framework (max |Δ|) in the registry comment.
- Pre- and post-processing are ports of the authors' inference code, like any other port.

### Formatting and lints

- The codebase is **not rustfmt-clean** and there is no `rustfmt.toml`. Do not run `cargo fmt`
  over files you are not otherwise changing, and do not reformat untouched code in a file you
  edit: it buries the real diff. Match the surrounding style. Lines wrap at about 100 columns,
  comments included.
- Clippy is a gate: `cargo clippy --all-targets --features "parallel onnx download" -- -D
  warnings`. The two crate-wide allows (`needless_range_loop`, `too_many_arguments`) are in
  `Cargo.toml` with their reasons. Fix any other lint rather than allow it; a new `#[allow]`
  carries a comment saying why.

### Dependencies

- Pure Rust only, and it must build for `wasm32-unknown-unknown`. Optional dependencies hang off
  a feature.
- Say in `Cargo.toml` why each dependency is there (the existing entries do).
- No third-party source is vendored. If a module was written with reference to code under
  another licence, reproduce that licence in `THIRD_PARTY_NOTICES.md` and name it in the
  module's docs.

## Comments and documentation

Doc comments are where QSM.rs records what each method is and why it behaves as it does. They
are written for someone checking the port against the paper and the reference code.

### Module docs (`//!`)

The first line names the method and says what it is, without a full stop:
`//! STAR-QSM: streaking artifact reduction for QSM`. Then, for an algorithm port, in this order
(leave out what does not apply):

1. **What is implemented, and from which reference.** Name the toolbox, version or commit, and
   the function as the reference calls it, e.g. "STI Suite 3.0's `QSM_star`, as called with
   `QSM_star(phase, mask, 'TE', TE, ...)`". If the reference code is closed (protected P-code),
   say how it was recovered (black-box probing) and to what tolerance it was verified.
2. **The algorithm**, as numbered steps or equations, close enough to the reference that a
   reader can follow it line by line. Say where the paper and the code differ and which one is
   followed (normally the code), with a heading such as `# Where the paper and the code differ`.
3. **`# Units`** where the method's units are not the crate's (radians inside, a field-scale
   dependence, an echo time it picks).
4. **Agreement with the reference**: what was compared, on what data, and the result (relative
   L2, max |Δ|, iteration counts), and which test checks it.
5. **Deviations**: each one, why it is an unequivocal improvement (or why it is an option), and
   the measurement behind it.
6. **`# Performance`** where cost or memory matters to a host.
7. **`# Reference pipeline`**: what the authors' pipeline runs around this method (masking,
   unwrapping, echo combination, background removal, their parameters), each with its source.
   This is the evidence for the method's pipeline rules.
8. **`Reference:`** the paper(s), as `Authors (year). "Title." Journal, vol(issue):pages.
   <https://doi.org/...>`.
9. **`# Pipeline rules`**: the ids of the rules in `pipeline::rules::RULES` that come from the
   facts above.

### Citing sources

- Cite the paper by section or equation (`§2.3`, `Eq. 12`, `Methods`) and the authors' code by
  repository-relative file and line: `README.m:144`, `parse_QSM_input.m:10-20`,
  `functions/bet2/bet2.cpp:136`, `run.py:499-506`. Name the toolbox version or commit the line
  numbers refer to.
- A measured claim gives the numbers and the data it was measured on, e.g. "XSIM 0.704 → 0.714,
  NRMSE 70.4 → 69.4 % on an in-vivo 3 T scan". Cite only public sources and public data.
- A quirk of the reference that the port keeps (a constant, an off-by-one, an unusual default)
  gets a comment at the line that keeps it, saying it is the reference's and where it is.

### Item docs (`///`)

- Public functions say what they compute, what they take (units, layout, which mask) and what
  they return, and link the module docs for the method. Link items with intra-doc links
  (``[`StarParams::pad`]``).
- Params fields: unit, default, source of the default (see *Naming and parameters*).
- Keep them factual and short. The long story belongs in the module docs.

### Inline comments (`//`)

- Explain *why*, or *which reference line* the code follows; not what the next line does.
- Mark where the code follows the reference's formula or loop structure, so a reader can check
  it, e.g. `// reg_CSF.m: opHop(x) = lambda_CSF * LT(LT(x)), ...`.
- Workflow files and scripts follow the same rule: a comment says why a step exists and what
  goes wrong without it.

### What does not go in docs

- The README covers general usage (install, feature flags, quick start) and the algorithm
  tables only. Experiments, comparisons and method-specific notes go in module docs or the PR
  body.
- No changelog or release-notes files. User-visible changes are summarised in the PR body;
  release notes are written on the GitHub release.

## Pipeline rules (`src/pipeline/rules.rs`)

`pipeline::rules` publishes, as data, what each method requires, recommends and discourages, so
QSMxT and qsmbly can show it. Its module docs are the specification; the essentials:

- **A rule reads** "when every condition in `when` holds, `setting` should pass `expect`, with
  `default` as the value to apply", at a `strength`, on a `basis`, with a one-line `reason` and a
  `source`.
- **No source, no rule.** A rule states what the method's reference pipeline does (authors' code,
  file and line, or paper section) or what QSM.rs measured (with the measurement recorded in that
  method's docs). It never encodes a preference. If the reference is ambiguous, there is no rule;
  record the open question in the method's docs.
- **Strength and basis** (`Required > Recommended > Discouraged > Reference`):

  | Strength | Meaning | Basis allowed |
  |---|---|---|
  | `Required` | The method cannot run correctly otherwise (e.g. it does not fit the WASM heap) | `Measured`, `Stated`, `Implementation` |
  | `Recommended` | The authors explicitly recommend it, or QSM.rs measured a benefit | `Stated`, `Measured` |
  | `Discouraged` | A combination known to fail, or one in which QSM.rs ignores a setting | `Stated`, `Measured`, `Implementation` |
  | `Reference` | The reference pipeline does it, without saying it is needed or better | `ReferenceChoice` |

  Use the constructors: `reference(...)`, `recommend(...)`, `measured_better(...)`,
  `discourage(...)`, or a full `Rule { ... }` for `Required` rules and ranges.
- **Scoped to the trigger.** The first condition is the choice that triggers the rule (usually
  the method being selected). "MEDI recommends PDF" fires only when MEDI is chosen.
- **Ids** are `<method>.<what>` in kebab case (`star.te-sum`, `hd-bet.wasm-heap`). Settings are
  dotted paths into QSM.rs's own config types and must be listed in `SETTINGS`; choices are named
  by their `id()`. Values carry their `Unit`; a rule never converts mm to voxels itself.
- **Sources** look like `"MEDI toolbox 2024.11.26 README.m:80,118; functions/bet2/bet2.cpp:136"`
  or `"Yaghmaie et al. 2021 NeuroImage 231:117701 (QSMART) §3.1"`, or name the QSM.rs docs that
  hold the measurement.
- **Every rule is also in the method's docs**: its fact under `# Reference pipeline`, its id
  under `# Pipeline rules`.
- **Tests** in the same file must keep passing: add a violating case for the new rule to
  `violations()`; the default config must still raise nothing; strengths must match their basis.
- Rules never change settings themselves. A host shows the finding and lets the user apply the
  rule's `default`; no setting changes silently because another was chosen.

## Tests

QSM.rs is where algorithm correctness is tested. QSMxT and qsmbly only test their wiring, so a
method's numerics are covered here or nowhere.

- **Unit tests** sit in the file they test, in `#[cfg(test)] mod tests`. Names are sentences in
  snake case that state the property: `param_error_names_parameter_value_and_range`,
  `default_config_raises_nothing`. Use small synthetic volumes. Every guard (each `InputError`
  path) gets a test. Unit-test coverage is gated at 90 %.
- **Reference parity tests** (`tests/<method>_<reference>_parity.rs`, e.g. `star_sti_parity.rs`,
  `n4_ants_parity.rs`) compare a port with the authors' own output to float tolerance. The
  fixture in `tests/data/<method>_<reference>_small.bin` holds only the reference's output; the
  inputs are rebuilt in the test exactly as the script that made the fixture built them. The
  test's module docs give the fixture's byte layout and the exact reference call that produced
  it. They are not `#[ignore]`d and run with the unit tests.
- **Integration tests** (`tests/integration_tests.rs`, `tests/dl_models.rs`, `tests/chisep_*.rs`)
  run each method on the QSM-CI phantom (`bids/`, fetched by `scripts/download-test-data.sh`).
  They are `#[ignore]` and named `test_<stage>_<method>` (`test_inversion_star`,
  `test_bgremove_vsharp`, `test_dl_xqsm`). Each one prints its metrics with
  `print_ci_metrics` (NRMSE, challenge NRMSE, correlation, XSIM, time), saves centre slices with
  `common::save_center_slices(result, mask, dims, "<stage>_<method>")`, and asserts bounds that
  catch divergence, with a comment saying what the reference itself scores where that explains
  the bound. A test that cannot find its data calls `common::skip_for_missing_data`, which fails
  in CI (`QSM_REQUIRE_TEST_DATA`).
- **WASM**: `tests/wasm_pipeline.rs` runs every pipeline stage and every classical algorithm on a
  16³ phantom, and every network on its smallest input, under Node. Add a case for a new method.
- Do not delete or weaken an assertion to make a change pass. If a reference value changes,
  say why in the test and in the PR.

### Adding a method to the integration suite (and its CI figures)

1. Add the `#[ignore]` test as above.
2. In `.github/workflows/integration-tests.yml`: add its name to the right `all_*` list (or to
   `not_in_ci` with a reason) and map its source file to it in the path map. CI fails if a
   test is in no list, and an unmapped `src/` file runs the whole suite.
3. In `scripts/render_slices.py`: a display name in `NAMES`, a colour window, and a cell in the
   stage's montage.
4. In `scripts/integration_report.sh`: a row in the stage's metrics table.
5. Optionally a time in `SECONDS` in `scripts/ci_shards.py` (a missing one is packed at the
   default).

On a PR, CI runs the tests the changed files map to and posts one comment with the metrics
tables, the stage montages and the change in each metric against the newest `main` run. Figures
and metrics records live on the `ci-figures` branch; every push to `main` runs the full suite
and records the baseline. The `run-all-tests` label runs the whole suite on a PR. Deep-learning
and QSMART results are informational and do not gate the merge.

## Processes

### Issues

- Every issue states its owner, QSM.rs or QSMxT (the table under *Related repositories*), and
  is filed in the owner's repository. If both need changes, file it here and note the QSMxT
  follow-up.
- Labels: a type (`bug`, `enhancement`, `documentation`, `question`), an effort
  (`effort: small`, `effort: medium`, `effort: large`), an owner (`owner: QSM.rs`,
  `owner: QSMxT`), and `outdated?` or `needs info` where they apply.

### Branches and worktrees

- Never commit to `main`. Branch from an up-to-date `origin/main` with a prefix that says what
  the work is: `feat/`, `fix/`, `perf/`, `ci/`, `docs/`, `chore/`.
- Work in a separate `git worktree` per branch, with its own `CARGO_TARGET_DIR`. Do not share a
  target directory between worktrees of this repository: cargo identifies path crates relative to
  the workspace, so one worktree can link another's build.

### Pull requests

- **Title**: a plain sentence saying what changed, led by the method or area:
  "STAR-QSM: default to STI Suite GUI's padding [8, 8, 20]", "N4 returns InputError; ...",
  "CI: ...". Append the issue number when it closes one: "... (#274)".
- **Body**, as applies:
  - *What changed*, per item, with the user-visible effect at default settings.
  - *Breaking API changes*: every changed signature, field or error message.
  - *Measured impact*: metrics before and after (XSIM first), and on what data.
  - *Decisions*: each choice made, with its reason. These must also be in the code comments.
  - *Paper vs code*: for ports, where they differ and which was followed.
  - *Tests*: what was run, locally and in CI.
  - *Downstream follow-ups*: what QSMxT or qsmbly must change to use it.
  - `Fixes #n` / `Closes #n`.
- Open as a draft until CI is green and the body is complete.
- **Stacked PRs**: before a PR is merged, retarget every PR stacked on it to `main` and rebase.
  A PR merged into a branch that has already been squash-merged never reaches `main`.
- Merging is the maintainer's call. PRs are **squash-merged**; the commit title is the PR title
  with its number.

### CI, and running it locally

Workflows (`.github/workflows/`):

| Workflow | When | What |
|---|---|---|
| `integration-tests.yml` | PRs to `main`, pushes to `main` | Maps changed files to tests; unit tests, check, clippy, coverage and the WASM builds (`unit-tests.yml`); builds the test binaries once per feature set (`test-binaries.yml`) and runs them in shards (`scripts/ci_shards.py`); posts the results comment. `tests-passed` is the gate. |
| `relaxometry-demo.yml` | PRs touching its paths | The relaxometry demo and its own comment. |
| `combinatorial-tests.yml` | Manual | Every background removal × inversion pair. |
| `warm-caches.yml` | Pushes to `main`, weekly | Saves the caches PRs restore. |
| `ci-figures-cleanup.yml` | Weekly | Prunes old figures from `ci-figures`. |
| `release.yml`, `docs.yml` | A release is published | Packages the crate; builds and deploys the website and API docs. |

The same checks locally:

```
cargo check --all-targets --features "parallel onnx download"
cargo clippy --all-targets --features "parallel onnx download" -- -D warnings
cargo test --lib --features parallel
cargo test --release --target wasm32-unknown-unknown --test wasm_pipeline --no-run
# one integration test, with the phantom in bids/ (scripts/download-test-data.sh):
cargo test --release --features parallel --test integration_tests -- --ignored --exact --nocapture test_inversion_star
```

Keep CI fast. Unit tests use the smallest data and cheapest settings that still check the
property; a slow one needs a reason in a comment. Integration tests run in parallel shards, so
a new one should not become the longest shard without a reason.

### Releases

- `Cargo.toml` stays at `version = "0.0.0"`. Publishing a GitHub release with tag `vX.Y.Z`
  sets the version from the tag, packages the crate and attaches it to the release, and deploys
  the docs. Do not bump the version in a PR.
- Release notes are drafted from the merged PRs' bodies and posted on the GitHub release.
- A release is cut only when the maintainer approves it.

### Docs

- The website (`web/`) and the API docs (`cargo doc`) deploy when a release is published.
- The README is usage and algorithm tables only. When adding a method, add its row (name,
  one-line description with the function name, full reference with DOI) and, if new, its
  reference implementation to the *Reference Implementations* table.

## Before opening a PR

- [ ] Branch off current `origin/main`, in its own worktree.
- [ ] For a port: paper section and authors' code file/line cited; every decision, deviation
      and quirk recorded in the module docs; `# Reference pipeline` and `# Pipeline rules`
      sections and the matching rules in `RULES`.
- [ ] Defaults are the reference's; inputs are the reference's (required, not optional).
- [ ] Bad input returns `InputError` (via `InputError::param` for ranges), with a test.
- [ ] Builds with no features, and with `parallel onnx download`; clippy clean; no new
      `Instant`/threads/fs on an algorithm path.
- [ ] Unit tests, and for a new method an integration test registered in CI (list, path map,
      figures, report row) and a `wasm_pipeline.rs` case.
- [ ] Only the lines you changed are reformatted.
- [ ] README algorithm table updated if a method was added; no changelog file.
- [ ] PR title and body as above, with breaking changes and downstream follow-ups listed.

# Instructions for coding agents

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing anything and follow it. The rules most
often broken:

- Ports follow the original authors' paper **and** code, with the reference's inputs (required,
  not optional) and defaults. No invented extensions or substitute inputs.
- Change an algorithm only for an unequivocal improvement, and keep the original behaviour as an
  option. Trade-offs are options, not defaults.
- Record every algorithm decision (default and why, reference file and line, deviations,
  measurements) in the method's doc comments, not only in the PR body.
- A new method a host can choose gets a catalogue entry (`src/catalogue/`) with checked
  citations; a rule needs a source and a test case.
- No backwards-compatibility shims; get the API right.
- Push early: locally run only a build, clippy and the new or changed tests (about 2 minutes at
  most); CI runs the full suite, the ignored integration tests and repeats.
- Do not reformat code you did not change. Never push to `main`; the maintainer merges.

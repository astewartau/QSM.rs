# Releasing QSM.rs

A release is a GitHub release of a commit on `main`, tagged `vX.Y.Z`. `Cargo.toml` stays at
`0.0.0`: `.github/workflows/release.yml` sets the version from the tag and uploads the packaged
crate to the release. Releases are made only with the maintainer's go.

Checklist, for the release candidate (the `main` commit to be tagged):

1. **CI passes** on that commit (tests, and the parity fixture tests in `tests/`).
2. **The QSM-CI parity release gate passes** against that commit: QSMxT/QSM-CI's `parity.yml` on
   its development track, with `qsmrs_ref` set to the candidate's SHA
   (QSM-CI `docs/parity.md`, "Before a QSM.rs release"):

   ```bash
   gh workflow run parity.yml -R QSMxT/QSM-CI \
     -f algorithm=enabled -f dataset=configured -f qsmrs_ref=<release candidate SHA>
   gh run watch -R QSMxT/QSM-CI --exit-status "$(gh run list -R QSMxT/QSM-CI -w parity -L 1 \
     --json databaseId -q '.[0].databaseId')"
   ```

   The run summary must say "Release gate: passed": every enabled pair passed on every configured
   dataset against exactly this commit. A failing pair blocks the release until the port is fixed,
   or until the change is shown to be intended and the pair is re-baselined in a QSM-CI PR. If the
   candidate changes an API that QSMxT `main` does not build against yet, pass the QSMxT branch
   that moves to it (`-f qsmxt_ref=<branch>`).
3. **Release notes** are drafted from the merged PRs' bodies since the last release (there is no
   changelog file).
4. **Publish** the GitHub release `vX.Y.Z` at that commit, and check that `release.yml` attached
   `qsm-core-X.Y.Z.crate`.

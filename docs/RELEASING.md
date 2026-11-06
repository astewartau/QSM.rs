# Releasing QSM.rs

A release is a GitHub release of a commit on `main`, tagged `vX.Y.Z`. `Cargo.toml` stays at
`0.0.0`: `.github/workflows/release.yml` sets the version from the tag and uploads the packaged
crate to the release. Releases are made only with the maintainer's go.

Checklist, for the release candidate (the `main` commit to be tagged):

1. **CI passes** on that commit (tests, and the parity fixture tests in `tests/`).
2. **Parity holds for every pair whose code changed** since its last recorded check. QSM-CI's
   status says which checks are stale (QSM-CI `docs/parity.md`, "Before a QSM.rs release"):

   ```bash
   python scripts/parity.py status        # in a QSM-CI checkout
   ```

   A stale flag means a file on the pair's path changed, not that its result can have: read what
   changed (the diff since the checked QSM.rs and QSMxT commits) and run only the pairs whose
   computation it touches, against the candidate:

   ```bash
   gh workflow run parity.yml -R QSMxT/QSM-CI \
     -f algorithm=<pairs> -f dataset=configured -f qsmrs_ref=<release candidate SHA>
   ```

   Record which flagged pairs were left out and why (in the release PR or notes' discussion). A
   failing pair blocks the release until the port is fixed, or until the change is shown to be
   intended and the pair is re-baselined in a QSM-CI PR. If the candidate changes an API that
   QSMxT `main` does not build against yet, pass the QSMxT branch that moves to it
   (`-f qsmxt_ref=<branch>`). The released track checks every enabled pair against the next QSMxT
   release by itself.
3. **Release notes** are drafted from the merged PRs' bodies since the last release (there is no
   changelog file).
4. **Publish** the GitHub release `vX.Y.Z` at that commit, and check that `release.yml` attached
   `qsm-core-X.Y.Z.crate`.

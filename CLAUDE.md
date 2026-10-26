# Instructions for coding agents

- Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing anything, and follow it. It is the standard for this repository.
- Never commit or push to `main`. Work on a branch in its own `git worktree`, with its own `CARGO_TARGET_DIR` (never shared between worktrees).
- Open PRs as drafts. Merging and releases are the maintainer's decision.
- Record every algorithm decision (default and why, reference file and line, deviations, measurements) in the method's doc comments, not only in the PR body.
- Do not run `cargo fmt` over code you did not change.

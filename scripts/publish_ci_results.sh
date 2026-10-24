#!/usr/bin/env bash
# Publish a run's figures and metrics record to the `ci-figures` branch.
#
# usage: publish_ci_results.sh
# env:   FIGURES_DIR     PNGs to publish (may be empty or missing)
#        FIGURES_DEST    the run's own directory on the branch:
#                        figures/pr-<N>/<workflow>/<run_id>-<attempt> or
#                        figures/main/<workflow>/<run_id>-<attempt>
#        PRUNE_OLDER     optional: the parent of FIGURES_DEST, whose run directories older than
#                        this run are deleted (a PR keeps only its latest run's figures; main
#                        keeps every run and leaves them to the weekly cleanup)
#        METRICS_FILE    optional: the run's metrics record (scripts/ci_metrics.py record)
#        METRICS_DEST    its path on the branch, under metrics/; metrics are never pruned
#        GITHUB_REPOSITORY, GITHUB_OUTPUT
#   Run from a checkout whose `origin` can push: the checkout action's token, with
#   `contents: write`. A fork PR's token cannot, so the workflows skip this step for forks.
# Writes `base_url=<raw URL of FIGURES_DEST>` to GITHUB_OUTPUT once the figures are published.
#
# Images are linked by branch path, not commit SHA. The weekly cleanup (ci-figures-cleanup.yml)
# squashes the branch to one commit, after which every earlier commit is unreachable: a
# SHA-pinned link would break at the first cleanup however recent its figures. A branch-path
# link works until its directory is deleted (a newer run of the same PR, the PR closing, or the
# 30-day limit). A run directory is written once and never changed.
#
# Built with plumbing on a temporary index: only the branch tip's trees are fetched (no blobs,
# no checkout), the run's files are added, older run directories of the same PR removed, and the
# commit pushed as a fast-forward. Each run touches only its own paths, so if another run pushed
# first the whole commit is simply rebuilt on the new tip. Only older runs are pruned, so a run
# that finishes late cannot delete a newer run's figures. Never fails the step: unpublished
# figures are named in the comment instead, and the artifact still has them.
set -uo pipefail
branch=ci-figures
repo="${GITHUB_REPOSITORY:?}"
dest="${FIGURES_DEST:?}"
prune="${PRUNE_OLDER:-}"
own="$(basename "$dest")"    # <run_id>-<attempt>

shopt -s nullglob
files=()
[ -n "${FIGURES_DIR:-}" ] && files=("$FIGURES_DIR"/*.png)
if [ "${#files[@]}" -eq 0 ] && [ -z "${METRICS_FILE:-}" ]; then
  echo "Nothing to publish."
  exit 0
fi

# Whether run directory $1 (<run_id>-<attempt>) is older than this run's.
older() {
  local a_run="${1%-*}" a_att="${1##*-}" b_run="${own%-*}" b_att="${own##*-}"
  [[ "$a_run" =~ ^[0-9]+$ && "$a_att" =~ ^[0-9]+$ ]] || return 1
  [ "$a_run" -lt "$b_run" ] || { [ "$a_run" -eq "$b_run" ] && [ "$a_att" -lt "$b_att" ]; }
}

export GIT_AUTHOR_NAME="github-actions[bot]" GIT_COMMITTER_NAME="github-actions[bot]"
export GIT_AUTHOR_EMAIL="41898282+github-actions[bot]@users.noreply.github.com"
export GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
index="$(mktemp)"
export GIT_INDEX_FILE="$index"

for attempt in 1 2 3 4 5 6 7 8; do
  rm -f "$index"
  parent=""
  if git fetch -q --no-tags --depth 1 --filter=blob:none origin \
       "+refs/heads/${branch}:refs/remotes/origin/${branch}" 2>/dev/null; then
    parent=$(git rev-parse "refs/remotes/origin/${branch}")
    git read-tree "$parent"
  else
    git read-tree --empty   # the branch does not exist yet
  fi
  run_files=("${files[@]}")
  if [ -n "$prune" ] && [ -n "$parent" ]; then
    for d in $(git ls-tree --name-only "$parent" "${prune}/"); do
      [ "$(basename "$d")" = "$own" ] && continue
      if older "$(basename "$d")"; then
        git ls-files -z -- "$d" | xargs -0 -r git update-index --force-remove --
        echo "Removing the older run's figures ${d}."
      else
        # A newer run of this PR got here first: its figures stand, and so does its comment.
        echo "A newer run's figures are already published (${d}); not publishing these."
        run_files=()
      fi
    done
  fi
  for f in "${run_files[@]}"; do
    blob=$(git hash-object -w "$f")
    git update-index --add --cacheinfo "100644,${blob},${dest}/$(basename "$f")"
  done
  if [ "${#run_files[@]}" -gt 0 ]; then
    # The cleanup reads this for the directory's age: the squashed branch has no history to ask.
    stamp=$(date -u +%Y-%m-%dT%H:%M:%SZ | git hash-object -w --stdin)
    git update-index --add --cacheinfo "100644,${stamp},${dest}/.created"
  fi
  if [ -n "${METRICS_FILE:-}" ] && [ -f "$METRICS_FILE" ]; then
    blob=$(git hash-object -w "$METRICS_FILE")
    git update-index --add --cacheinfo "100644,${blob},${METRICS_DEST:?}"
  fi
  tree=$(git write-tree)
  msg="Results of ${dest#figures/}"
  if [ -n "$parent" ]; then
    commit=$(git commit-tree "$tree" -p "$parent" -m "$msg")
  else
    commit=$(git commit-tree "$tree" -m "$msg")
  fi
  if git push -q origin "${commit}:refs/heads/${branch}"; then
    echo "Published ${#run_files[@]} figure(s) to ${branch}:${dest}${METRICS_DEST:+ and ${METRICS_DEST}} (attempt $attempt)."
    if [ "${#run_files[@]}" -gt 0 ]; then
      echo "base_url=https://raw.githubusercontent.com/${repo}/${branch}/${dest}" >> "${GITHUB_OUTPUT:-/dev/null}"
    fi
    exit 0
  fi
  wait=$((attempt * 5 + RANDOM % 10))
  echo "Push to ${branch} rejected (attempt $attempt); rebuilding on the new tip in ${wait}s..."
  sleep "$wait"
done
echo "Could not publish to ${branch}; the comment will name the figures instead (non-fatal)."
exit 0

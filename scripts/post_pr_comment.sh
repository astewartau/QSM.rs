#!/usr/bin/env bash
# Post a markdown report as a PR comment, or update the one an earlier run of the same workflow
# posted.
#
# usage: post_pr_comment.sh <report.md> <key>
#   <key> names the comment (e.g. `integration-tests`). It goes into a hidden marker, and a later
#   run finds the comment by that marker and edits it, so a PR carries one results comment per
#   workflow rather than one per push. A second hidden line records the run; a run older than
#   the one already in the comment (one that finished late) leaves the comment alone.
# env:   GH_TOKEN (GITHUB_TOKEN), GITHUB_REPOSITORY, GITHUB_RUN_ID, GITHUB_RUN_ATTEMPT,
#        PR_NUMBER, RUN_URL, HEAD_SHA,
#        FIGURES_BASE_URL (where scripts/publish_ci_results.sh published the figures; empty if
#        it did not, e.g. on a fork PR, whose token cannot push),
#        FIGURES_URL (the figures artifact's download link; empty if none was uploaded),
#        FIGURES_RETENTION_DAYS (how long both are kept)
#
# The report shows the figures inline when they were published and names them otherwise; the
# footer links the artifact either way. Never fails the step: the comment is informational and
# the run's pass/fail gate is elsewhere.
set -uo pipefail
report="$1"
key="$2"

if [ -z "${PR_NUMBER:-}" ]; then
  echo "No PR number; nothing posted."
  exit 0
fi
repo="${GITHUB_REPOSITORY:?}"
marker="<!-- qsmrs-ci:${key} -->"
run="${GITHUB_RUN_ID:-0}-${GITHUB_RUN_ATTEMPT:-1}"
days="${FIGURES_RETENTION_DAYS:-30}"

body="$(mktemp)"
{
  echo "$marker"
  echo "<!-- run: ${run} -->"
  cat "$report"
  echo ""
  echo "---"
  if [ -n "${FIGURES_BASE_URL:-}" ] && [ -n "${FIGURES_URL:-}" ]; then
    echo "**Figures** (kept ${days} days): shown above, and as [a zip artifact](${FIGURES_URL}). After ${days} days, or once a newer run of this PR replaces them, the images in this version of the comment stop loading."
  elif [ -n "${FIGURES_URL:-}" ]; then
    echo "**Figures** (kept ${days} days): [download the \`figures\` artifact](${FIGURES_URL}), a zip of the PNG montages named above."
  else
    echo "**Figures:** none were rendered for this run."
  fi
  echo ""
  echo "**Run:** ${RUN_URL:-} (commit \`${HEAD_SHA:-unknown}\`)"
} > "$body"

# Whether run $1 (<run_id>-<attempt>) is newer than this one.
newer() {
  local a_run="${1%-*}" a_att="${1##*-}" b_run="${run%-*}" b_att="${run##*-}"
  [[ "$a_run" =~ ^[0-9]+$ && "$a_att" =~ ^[0-9]+$ ]] || return 1
  [ "$a_run" -gt "$b_run" ] || { [ "$a_run" -eq "$b_run" ] && [ "$a_att" -gt "$b_att" ]; }
}

for attempt in 1 2 3; do
  # The newest earlier comment from this workflow carrying the marker, if any.
  existing=$(gh api "repos/${repo}/issues/${PR_NUMBER}/comments" --paginate \
    --jq ".[] | select(.user.login == \"github-actions[bot]\" and (.body | startswith(\"${marker}\"))) | .id" \
    | tail -n 1) || existing=""
  if [ -n "$existing" ]; then
    theirs=$(gh api "repos/${repo}/issues/comments/${existing}" --jq .body \
      | sed -n 's/^<!-- run: \([0-9]*-[0-9]*\) -->$/\1/p' | head -n 1)
    if newer "$theirs"; then
      echo "PR comment ${existing} already shows a newer run (${theirs}); left as it is."
      exit 0
    fi
    if gh api -X PATCH "repos/${repo}/issues/comments/${existing}" -F "body=@${body}" --jq .html_url; then
      echo "PR comment ${existing} updated (attempt $attempt)."
      exit 0
    fi
  elif gh pr comment "$PR_NUMBER" -R "$repo" --body-file "$body"; then
    echo "PR comment posted (attempt $attempt)."
    exit 0
  fi
  sleep $((attempt * 20))
done
echo "Could not post the PR comment (non-fatal)."
exit 0

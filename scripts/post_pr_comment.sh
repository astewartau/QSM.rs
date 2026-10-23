#!/usr/bin/env bash
# Post a markdown report as a PR comment, or update the one an earlier run of the same workflow
# posted.
#
# usage: post_pr_comment.sh <report.md> <key>
#   <key> names the comment (e.g. `integration-tests`). It goes into a hidden marker, and a later
#   run finds the comment by that marker and edits it, so a PR carries one results comment per
#   workflow rather than one per push.
# env:   GH_TOKEN (GITHUB_TOKEN), GITHUB_REPOSITORY, PR_NUMBER, RUN_URL, HEAD_SHA,
#        FIGURES_URL (the figures artifact's download link; empty if none was uploaded),
#        FIGURES_RETENTION_DAYS (how long the artifact is kept)
#
# The figures are not in the comment: a workflow artifact cannot be embedded as an image, so the
# comment links to it instead. Never fails the step: the comment is informational and the run's
# pass/fail gate is elsewhere.
set -uo pipefail
report="$1"
key="$2"

if [ -z "${PR_NUMBER:-}" ]; then
  echo "No PR number; nothing posted."
  exit 0
fi
repo="${GITHUB_REPOSITORY:?}"
marker="<!-- qsmrs-ci:${key} -->"

body="$(mktemp)"
{
  echo "$marker"
  cat "$report"
  echo ""
  echo "---"
  if [ -n "${FIGURES_URL:-}" ]; then
    echo "**Figures:** [download the \`figures\` artifact](${FIGURES_URL}) (a zip of the PNG montages named above; kept for ${FIGURES_RETENTION_DAYS:-30} days)."
  else
    echo "**Figures:** none were rendered for this run."
  fi
  echo ""
  echo "**Run:** ${RUN_URL:-} (commit \`${HEAD_SHA:-unknown}\`)"
} > "$body"

for attempt in 1 2 3; do
  # The newest earlier comment from this workflow carrying the marker, if any.
  existing=$(gh api "repos/${repo}/issues/${PR_NUMBER}/comments" --paginate \
    --jq ".[] | select(.user.login == \"github-actions[bot]\" and (.body | startswith(\"${marker}\"))) | .id" \
    | tail -n 1) || existing=""
  if [ -n "$existing" ]; then
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

#!/usr/bin/env bash
# Post a markdown report as a PR comment.
#
# usage: post_pr_comment.sh <report.md>
# env:   REPO_TOKEN / GH_TOKEN (GITHUB_TOKEN), PR_NUMBER, RUN_URL
#
# First through CML, which uploads the report's local images (figures/*.png) to its asset host
# and posts the comment, retried with backoff (the GitHub API is rate limited across concurrent
# runs). If CML keeps failing (for example its asset host is down), the report goes out through
# the GitHub CLI without the images, each replaced by a pointer to the run's `figures` artifact,
# so an image-host outage costs the pictures, not the whole comment. Never fails the step:
# the comment is informational and the run's pass/fail gate is elsewhere.
set -uo pipefail
report="$1"

if command -v cml >/dev/null 2>&1; then
  for attempt in 1 2 3 4 5; do
    if cml comment create "$report"; then
      echo "PR comment posted through CML (attempt $attempt)."
      exit 0
    fi
    if [ "$attempt" -lt 5 ]; then
      wait=$((attempt * 30))
      echo "cml comment create failed (attempt $attempt); retrying in ${wait}s..."
      sleep "$wait"
    fi
  done
  echo "cml comment create still failing after 5 attempts; posting without figures."
else
  echo "CML is not available; posting without figures."
fi

if [ -z "${PR_NUMBER:-}" ]; then
  echo "No PR number; nothing posted."
  exit 0
fi
export GH_TOKEN="${GH_TOKEN:-${REPO_TOKEN:-}}"
fallback="$(mktemp)"
note="*Figures not shown: the image host was unavailable. They are in the \`figures\` artifact of [this run](${RUN_URL:-})."
note="${note}*"
# Each local image line becomes the note (the first one only; the rest are dropped).
awk -v note="$note" '
  /^!\[[^]]*\]\(figures\/[^)]*\)[[:space:]]*$/ { if (!shown) { print note; shown = 1 } next }
  { print }
' "$report" > "$fallback"
for attempt in 1 2 3; do
  if gh pr comment "$PR_NUMBER" --body-file "$fallback"; then
    echo "PR comment posted through the GitHub CLI, without figures (attempt $attempt)."
    exit 0
  fi
  sleep $((attempt * 20))
done
echo "Could not post the PR comment (non-fatal)."
exit 0

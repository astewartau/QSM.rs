#!/usr/bin/env bash
# Prune the `ci-figures` branch and squash it to one commit (ci-figures-cleanup.yml, weekly).
#
# usage: ci_figures_cleanup.sh
# env:   GH_TOKEN, GITHUB_REPOSITORY; run from a checkout whose `origin` can push
#        CLOSED_DAYS (default 7)  a closed PR's figures go this many days after it closed
#        MAX_DAYS (default 30)    any run's figures go this many days after they were published
#        NOW (optional, epoch seconds; for testing)
#
# Figures live in figures/<pr-N|main>/<workflow>/<run_id>-<attempt>/, each run directory with a
# `.created` timestamp (scripts/publish_ci_results.sh). Metrics records (metrics/) are never
# removed. The new commit has no parent, so the images of every deleted run stop being stored;
# links to the runs that remain keep working, since comments link by branch path. A run that
# publishes while this runs moves the branch, the lease fails, and the cleanup starts over on the
# new tip.
set -euo pipefail
branch=ci-figures
repo="${GITHUB_REPOSITORY:?}"
closed_days="${CLOSED_DAYS:-7}"
max_days="${MAX_DAYS:-30}"
now="${NOW:-$(date -u +%s)}"
index="$(mktemp)"
export GIT_INDEX_FILE="$index"
export GIT_AUTHOR_NAME="github-actions[bot]" GIT_COMMITTER_NAME="github-actions[bot]"
export GIT_AUTHOR_EMAIL="41898282+github-actions[bot]@users.noreply.github.com"
export GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
declare -A pr_closed   # PR number -> close time (epoch), or "open"

closed_at() {
  local n="$1"
  if [ -z "${pr_closed[$n]:-}" ]; then
    local t
    t=$(gh api "repos/${repo}/pulls/${n}" --jq '.closed_at // ""') || t=""
    if [ -n "$t" ]; then pr_closed[$n]=$(date -u -d "$t" +%s); else pr_closed[$n]=open; fi
  fi
  echo "${pr_closed[$n]}"
}

for attempt in 1 2 3 4 5; do
  rm -f "$index"   # git reads an empty file as a corrupt index; read-tree below creates it
  if ! git fetch -q --no-tags --depth 1 origin "+refs/heads/${branch}:refs/remotes/origin/${branch}"; then
    echo "No ${branch} branch; nothing to clean."
    exit 0
  fi
  tip=$(git rev-parse "refs/remotes/origin/${branch}")
  git read-tree "$tip"
  removed=0 kept=0
  # Run directories: figures/<scope>/<workflow>/<run>/.created
  while IFS= read -r created; do
    dir="${created%/.created}"
    scope=$(cut -d/ -f2 <<< "$dir")
    born=$(date -u -d "$(git cat-file blob "${tip}:${created}")" +%s 2>/dev/null) || born=0
    why=""
    if [ $(( now - born )) -ge $(( max_days * 86400 )) ]; then
      why="older than ${max_days} days"
    elif [[ "$scope" =~ ^pr-([0-9]+)$ ]]; then
      c=$(closed_at "${BASH_REMATCH[1]}")
      if [ "$c" != open ] && [ $(( now - c )) -ge $(( closed_days * 86400 )) ]; then
        why="PR closed more than ${closed_days} days ago"
      fi
    fi
    if [ -n "$why" ]; then
      git ls-files -z -- "$dir" | xargs -0 -r git update-index --force-remove --
      echo "Removing ${dir} (${why})."
      removed=$((removed + 1))
    else
      kept=$((kept + 1))
    fi
  done < <(git ls-tree -r --name-only "$tip" -- figures/ | grep '/\.created$' || true)

  tree=$(git write-tree)
  commit=$(git commit-tree "$tree" -m "ci-figures: ${kept} run(s) of figures and every metrics record (squashed $(date -u +%Y-%m-%d))")
  if git push -q --force-with-lease="refs/heads/${branch}:${tip}" origin "${commit}:refs/heads/${branch}"; then
    echo "Removed ${removed} run(s) of figures, kept ${kept}; ${branch} squashed to ${commit:0:7} (attempt ${attempt})."
    exit 0
  fi
  echo "${branch} moved while cleaning (attempt ${attempt}); starting over on the new tip."
  sleep $((attempt * 10))
done
echo "Could not update ${branch}."
exit 1

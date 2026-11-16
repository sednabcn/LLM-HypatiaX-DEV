#!/usr/bin/env bash
# Commit only generated paths; preserve them across concurrent branch updates.
set -euo pipefail

if [[ $# -eq 0 ]]; then echo "::error::No output paths supplied."; exit 2; fi
: "${GITHUB_REF_NAME:?GITHUB_REF_NAME must be set}"
: "${COMMIT_MESSAGE:?COMMIT_MESSAGE must be set}"
RESULTS_ROOT="$(realpath -m -- hypatiax/data/results)"
for path in "$@"; do
  [[ -f "$path" ]] || { echo "::error::Missing generated file: $path"; exit 1; }
  resolved="$(realpath -m -- "$path")"
  case "$resolved" in
    "$RESULTS_ROOT"/*) ;;
    *) echo "::error::Refusing to commit a path outside hypatiax/data/results: $path"; exit 1 ;;
  esac
done
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
stash_before="$(git rev-parse -q --verify refs/stash 2>/dev/null || true)"
git stash push --include-untracked -m "ci-analysis-${GITHUB_RUN_ID:-manual}"
stash_after="$(git rev-parse -q --verify refs/stash 2>/dev/null || true)"
git fetch origin "$GITHUB_REF_NAME"
git rebase "origin/$GITHUB_REF_NAME"
if [[ -n "$stash_after" && "$stash_after" != "$stash_before" ]]; then
  git stash pop --index
fi
git add -f -- "$@"
if git diff --cached --quiet; then echo "Generated outputs unchanged — nothing to commit."; exit 0; fi
git commit -m "$COMMIT_MESSAGE"
git push origin "HEAD:${GITHUB_REF_NAME}"
echo "Committed generated analysis outputs."

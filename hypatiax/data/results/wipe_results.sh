#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  wipe_results.sh — hard-reset all experiment outputs to a clean state
#
#  Removes every result file (*.json, *.csv) from:
#    - canonical tree paths  (comparison_results/, extrapolation/, …)
#    - staging area          (hypatiax/data/results/)
#    - reports               (hypatiax/data/reports/)
#    - figures / tables      (figures/, tables/)
#
#  Pass --commit to automatically git-add and push the wipe.
#  Pass --dry-run to preview what would be deleted without touching anything.
#
#  Usage:
#    bash wipe_results.sh [--dry-run] [--commit]
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

DRY_RUN=false
COMMIT=false

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    --commit)  COMMIT=true  ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

# ── Directories to wipe ──────────────────────────────────────────────────────
# Canonical result tree (matches tree_results.txt)
CANONICAL_DIRS=(
  "comparison_results"
  "extrapolation"
  "hybrid_llm_nn"
  "hybrid_pysr"
  "llm_guided"
  "standalone_llm_nn"
  "figures"
  "tables"
)

# Staging + reports under hypatiax/
HYPATIAX_DIRS=(
  "hypatiax/data/results"
  "hypatiax/data/reports"
)

# ── Helpers ──────────────────────────────────────────────────────────────────
deleted=0

delete_files() {
  local pattern="$1"
  # Use find to locate matching files; skip if nothing found
  local files
  files=$(find . -path "./.git" -prune -o -type f -name "$pattern" -print 2>/dev/null || true)
  while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    if $DRY_RUN; then
      echo "  [dry-run] would delete: $f"
    else
      rm -f "$f"
      echo "  deleted: $f"
    fi
    (( deleted++ )) || true
  done <<< "$files"
}

wipe_dir() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    echo "  (skip — not found): $dir"
    return
  fi
  echo "Wiping $dir ..."
  pushd "$dir" > /dev/null
    delete_files "*.json"
    delete_files "*.csv"
  popd > /dev/null
}

# ── Main ─────────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════"
echo " HypatiaX results wipe$( $DRY_RUN && echo ' [DRY RUN]' || echo '')"
echo "══════════════════════════════════════════════════════"

for d in "${CANONICAL_DIRS[@]}";  do wipe_dir "$d"; done
for d in "${HYPATIAX_DIRS[@]}";   do wipe_dir "$d"; done

echo ""
echo "══════════════════════════════════════════════════════"
if $DRY_RUN; then
  echo " DRY RUN complete — $deleted file(s) would be deleted."
  echo " Re-run without --dry-run to actually wipe."
else
  echo " Wipe complete — $deleted file(s) deleted."
fi
echo "══════════════════════════════════════════════════════"

# ── Optional git commit ───────────────────────────────────────────────────────
if $COMMIT && ! $DRY_RUN; then
  echo ""
  echo "Staging deleted files with git ..."
  git config user.name  "github-actions[bot]"  2>/dev/null || true
  git config user.email "github-actions[bot]@users.noreply.github.com" 2>/dev/null || true

  # Stage all deletions (git rm --cached handles already-deleted files)
  for d in "${CANONICAL_DIRS[@]}" "${HYPATIAX_DIRS[@]}"; do
    if [[ -d "$d" ]]; then
      git rm -rf --cached --ignore-unmatch "$d"/*.json "$d"/**/*.json \
                                           "$d"/*.csv  "$d"/**/*.csv \
        2>/dev/null || true
    fi
  done

  if git diff --cached --quiet; then
    echo "Nothing staged — either already clean or no tracked result files existed."
  else
    git commit -m "chore: wipe all experiment results — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    git push origin master
    echo "Pushed wipe commit to master."
  fi
fi

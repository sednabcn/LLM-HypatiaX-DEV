#!/usr/bin/env bash
# cleanup_figures_prefix.sh
#
# Deletes every file whose name starts with "figures__" or "Figures__"
# (single-prefix AND double-prefix variants) from the entire repo tree.
# Also removes REPO_AUDIT / PROD__REPO_AUDIT PDFs that leaked into figures/ dirs.
#
# DRY-RUN by default — pass --delete to actually remove files.
#
# Usage:
#   bash cleanup_figures_prefix.sh            # preview what would be deleted
#   bash cleanup_figures_prefix.sh --delete   # delete for real

set -euo pipefail

REPO_ROOT="${1:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
DRY_RUN=true
[[ "${1:-}" == "--delete" || "${2:-}" == "--delete" ]] && DRY_RUN=false

echo "=== figures__ prefix cleanup ==="
echo "Repo root : ${REPO_ROOT}"
echo "Mode      : $( $DRY_RUN && echo 'DRY RUN (pass --delete to apply)' || echo 'DELETE' )"
echo ""

TOTAL=0

# Pattern 1: files named figures__* or Figures__* (catches single + double prefix)
while IFS= read -r -d '' f; do
    echo "  REMOVE  $f"
    $DRY_RUN || rm -f "$f"
    (( TOTAL++ )) || true
done < <(find "${REPO_ROOT}" \
    \( -name "figures__*" -o -name "Figures__*" \) \
    -type f -print0)

# Pattern 2: REPO_AUDIT / PROD__REPO_AUDIT pdfs that leaked into figures/ dirs
while IFS= read -r -d '' f; do
    echo "  REMOVE  $f"
    $DRY_RUN || rm -f "$f"
    (( TOTAL++ )) || true
done < <(find "${REPO_ROOT}" \
    -path "*/figures/REPO_AUDIT*" -type f -print0)

while IFS= read -r -d '' f; do
    echo "  REMOVE  $f"
    $DRY_RUN || rm -f "$f"
    (( TOTAL++ )) || true
done < <(find "${REPO_ROOT}" \
    -path "*/figures/PROD__REPO_AUDIT*" -type f -print0)

echo ""
echo "=== Total files $( $DRY_RUN && echo 'flagged' || echo 'deleted' ): ${TOTAL} ==="

if $DRY_RUN; then
    echo ""
    echo "Re-run with --delete to apply:"
    echo "  bash cleanup_figures_prefix.sh --delete"
fi

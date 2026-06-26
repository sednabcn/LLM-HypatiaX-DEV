#!/usr/bin/env bash
# cleanup_figures_prefix.sh
#
# Finds every file whose name starts with "figures__" or "Figures__"
# (single-prefix AND double-prefix variants) from the entire repo tree.
# Also finds REPO_AUDIT / PROD__REPO_AUDIT PDFs that leaked into figures/ dirs.
#
# By default this just PREVIEWS (dry run) what would happen.
# Use --delete to actually delete the matched files.
# Use --move   to actually move the matched files into a backup folder
#              (default: <repo_root>/hypatiax/data/results/figures_back).
# Add --dry-run alongside --delete/--move to preview that specific
# action without making any changes.
#
# Usage:
#   bash cleanup_figures_prefix.sh                       # preview (delete-style) — default
#   bash cleanup_figures_prefix.sh --delete              # delete for real
#   bash cleanup_figures_prefix.sh --move                # move for real
#   bash cleanup_figures_prefix.sh --move --dry-run      # preview move destinations only
#   bash cleanup_figures_prefix.sh --move --dest=/some/other/dir
#   bash cleanup_figures_prefix.sh /path/to/repo --move  # explicit repo root + move

set -euo pipefail

REPO_ROOT=""
ACTION="delete"        # delete | move
ACTION_EXPLICIT=false  # true once --delete or --move is actually passed
APPLY=""               # set by --dry-run; otherwise derived below
DEST_DIR_OVERRIDE=""

for arg in "$@"; do
    case "$arg" in
        --delete)
            ACTION="delete"
            ACTION_EXPLICIT=true
            ;;
        --move)
            ACTION="move"
            ACTION_EXPLICIT=true
            ;;
        --dry-run)
            APPLY=false
            ;;
        --dest=*)
            DEST_DIR_OVERRIDE="${arg#--dest=}"
            ;;
        --*)
            echo "Unknown option: $arg" >&2
            exit 1
            ;;
        *)
            REPO_ROOT="$arg"
            ;;
    esac
done

# No flags at all (or only a repo path) -> safe preview, same as the
# original script's behavior. Only --delete/--move actually applies changes,
# unless --dry-run is also given, in which case it always previews.
if [[ -z "$APPLY" ]]; then
    if $ACTION_EXPLICIT; then
        APPLY=true
    else
        APPLY=false
    fi
fi

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
DEST_DIR="${DEST_DIR_OVERRIDE:-${REPO_ROOT}/hypatiax/data/results/figures_back}"

echo "=== figures__ prefix cleanup ==="
echo "Repo root : ${REPO_ROOT}"
echo "Action    : ${ACTION}"
if [[ "$ACTION" == "move" ]]; then
    echo "Dest dir  : ${DEST_DIR}"
fi
echo "Mode      : $( $APPLY && echo 'APPLY (changes will be made)' || echo 'DRY RUN (preview only)' )"
echo ""

if [[ "$ACTION" == "move" && "$APPLY" == true ]]; then
    mkdir -p "$DEST_DIR"
fi

TOTAL=0
declare -A USED_DESTS

process_file() {
    local f="$1"
    local base dest ext stem n

    if [[ "$ACTION" == "delete" ]]; then
        if $APPLY; then
            echo "  REMOVE  $f"
            rm -f "$f"
        else
            echo "  [DRY-RUN] would REMOVE  $f"
        fi
    else
        base="$(basename -- "$f")"
        dest="${DEST_DIR}/${base}"
        # Check both what's already on disk AND what we've already planned
        # to put there this run, so dry-run previews match real --move output.
        if [[ -e "$dest" || -n "${USED_DESTS[$dest]:-}" ]]; then
            if [[ "$base" == *.* ]]; then
                stem="${base%.*}"
                ext=".${base##*.}"
            else
                stem="$base"
                ext=""
            fi
            n=1
            while [[ -e "${DEST_DIR}/${stem}__${n}${ext}" || -n "${USED_DESTS[${DEST_DIR}/${stem}__${n}${ext}]:-}" ]]; do
                (( n++ ))
            done
            dest="${DEST_DIR}/${stem}__${n}${ext}"
        fi
        USED_DESTS["$dest"]=1

        if $APPLY; then
            echo "  MOVE    $f -> $dest"
            mv -- "$f" "$dest"
        else
            echo "  [DRY-RUN] would MOVE    $f -> $dest"
        fi
    fi

    (( TOTAL++ )) || true
}

# Pattern 1: files named figures__* or Figures__* (catches single + double prefix)
while IFS= read -r -d '' f; do
    process_file "$f"
done < <(find "${REPO_ROOT}" \
    -not -path "${DEST_DIR}/*" \
    \( -name "figures__*" -o -name "Figures__*" \) \
    -type f -print0)

# Pattern 2: REPO_AUDIT / PROD__REPO_AUDIT pdfs that leaked into figures/ dirs
while IFS= read -r -d '' f; do
    process_file "$f"
done < <(find "${REPO_ROOT}" \
    -not -path "${DEST_DIR}/*" \
    -path "*/figures/REPO_AUDIT*" -type f -print0)

while IFS= read -r -d '' f; do
    process_file "$f"
done < <(find "${REPO_ROOT}" \
    -not -path "${DEST_DIR}/*" \
    -path "*/figures/PROD__REPO_AUDIT*" -type f -print0)

echo ""
ACTION_LABEL="deleted"
[[ "$ACTION" == "move" ]] && ACTION_LABEL="moved"
echo "=== Total files $( $APPLY && echo "$ACTION_LABEL" || echo 'flagged' ): ${TOTAL} ==="

if ! $APPLY; then
    echo ""
    echo "Re-run with one of the following to apply:"
    echo "  bash cleanup_figures_prefix.sh --delete"
    echo "  bash cleanup_figures_prefix.sh --move"
fi

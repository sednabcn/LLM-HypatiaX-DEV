#!/usr/bin/env bash
# =============================================================================
# purge_actions_cache.sh
# Manually delete ALL GitHub Actions caches for the HypatiaX exp2 Feynman
# workflow (key prefix: feynman-sh-checkpoint-*).
#
# Usage:
#   ./purge_actions_cache.sh                        # delete feynman-sh-checkpoint-* only
#   ./purge_actions_cache.sh --all                  # delete every cache in the repo
#   ./purge_actions_cache.sh --dry-run              # list what would be deleted, no action
#   ./purge_actions_cache.sh --all --dry-run
#
# Requirements:
#   gh CLI  (https://cli.github.com) — authenticated via `gh auth login`
#   jq
#
# The script pages through the GitHub cache API (100 per page), filters by
# key prefix, and deletes each entry individually.  Deletion is irreversible.
# =============================================================================

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
KEY_PREFIX="feynman-sh-checkpoint-"
DELETE_ALL=false
DRY_RUN=false

# ── Argument parsing ──────────────────────────────────────────────────────────
for arg in "$@"; do
  case "$arg" in
    --all)     DELETE_ALL=true ;;
    --dry-run) DRY_RUN=true    ;;
    --help|-h)
      sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg"
      echo "Usage: $0 [--all] [--dry-run]"
      exit 1
      ;;
  esac
done

# ── Dependency checks ─────────────────────────────────────────────────────────
for cmd in gh jq; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "ERROR: '$cmd' is required but not found."
    echo "  gh  → https://cli.github.com"
    echo "  jq  → https://stedolan.github.io/jq/"
    exit 1
  fi
done

# ── Repo detection ────────────────────────────────────────────────────────────
# Use GITHUB_REPOSITORY env (set in CI) or infer from `gh repo view`.
if [[ -n "${GITHUB_REPOSITORY:-}" ]]; then
  REPO="$GITHUB_REPOSITORY"
else
  REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || true)
  if [[ -z "$REPO" ]]; then
    echo "ERROR: Could not detect repository."
    echo "  Run from inside the repo clone, or set GITHUB_REPOSITORY=owner/repo"
    exit 1
  fi
fi

echo "Repository : $REPO"
if $DELETE_ALL; then
  echo "Scope      : ALL caches"
else
  echo "Scope      : caches with key prefix '$KEY_PREFIX'"
fi
$DRY_RUN && echo "Mode       : DRY RUN — nothing will be deleted"
echo ""

# ── Fetch all cache entries (paginated) ───────────────────────────────────────
PAGE=1
PER_PAGE=100
ALL_ENTRIES="[]"

while true; do
  RESPONSE=$(gh api \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "/repos/${REPO}/actions/caches?per_page=${PER_PAGE}&page=${PAGE}" \
    2>/dev/null)

  COUNT=$(echo "$RESPONSE" | jq '.actions_caches | length')
  if [[ "$COUNT" -eq 0 ]]; then
    break
  fi

  ALL_ENTRIES=$(echo "$ALL_ENTRIES" "$RESPONSE" | jq -s '.[0] + .[1].actions_caches')
  (( PAGE++ ))
done

TOTAL=$(echo "$ALL_ENTRIES" | jq 'length')
echo "Total caches found in repo: $TOTAL"

# ── Filter ────────────────────────────────────────────────────────────────────
if $DELETE_ALL; then
  TARGET_ENTRIES="$ALL_ENTRIES"
else
  TARGET_ENTRIES=$(echo "$ALL_ENTRIES" | \
    jq --arg prefix "$KEY_PREFIX" '[.[] | select(.key | startswith($prefix))]')
fi

TARGET_COUNT=$(echo "$TARGET_ENTRIES" | jq 'length')

if [[ "$TARGET_COUNT" -eq 0 ]]; then
  echo "No matching caches found. Nothing to do."
  exit 0
fi

echo "Caches to delete: $TARGET_COUNT"
echo ""

# ── List targets ──────────────────────────────────────────────────────────────
echo "$TARGET_ENTRIES" | jq -r '.[] | "  [\(.id)]  \(.key)  (\(.size_in_bytes) bytes)  ref=\(.ref)  created=\(.created_at)"'
echo ""

# ── Confirm (interactive only) ────────────────────────────────────────────────
if ! $DRY_RUN; then
  if [[ -t 0 ]]; then
    # stdin is a terminal — ask for confirmation
    read -r -p "Delete $TARGET_COUNT cache(s)? [y/N] " CONFIRM
    if [[ "${CONFIRM,,}" != "y" ]]; then
      echo "Aborted."
      exit 0
    fi
  else
    # Non-interactive (piped / CI) — proceed without prompt
    echo "Non-interactive mode — proceeding with deletion."
  fi
fi

# ── Delete ────────────────────────────────────────────────────────────────────
DELETED=0
FAILED=0

while IFS= read -r CACHE_ID; do
  CACHE_KEY=$(echo "$TARGET_ENTRIES" | jq -r --argjson id "$CACHE_ID" '.[] | select(.id == $id) | .key')

  if $DRY_RUN; then
    echo "  [dry-run] would delete id=$CACHE_ID  key=$CACHE_KEY"
    (( DELETED++ ))
    continue
  fi

  HTTP_STATUS=$(gh api \
    --method DELETE \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "/repos/${REPO}/actions/caches/${CACHE_ID}" \
    -w "%{http_code}" -o /dev/null 2>/dev/null || echo "000")

  if [[ "$HTTP_STATUS" == "204" || "$HTTP_STATUS" == "200" ]]; then
    echo "  ✓ deleted  id=$CACHE_ID  key=$CACHE_KEY"
    (( DELETED++ ))
  else
    echo "  ✗ FAILED   id=$CACHE_ID  key=$CACHE_KEY  (HTTP $HTTP_STATUS)"
    (( FAILED++ ))
  fi

done < <(echo "$TARGET_ENTRIES" | jq -r '.[].id')

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
if $DRY_RUN; then
  echo "Dry run complete. Would have deleted: $DELETED cache(s)."
else
  echo "Done. Deleted: $DELETED   Failed: $FAILED"
  if [[ "$FAILED" -gt 0 ]]; then
    echo "Re-run the script to retry failed deletions."
    exit 1
  fi
fi

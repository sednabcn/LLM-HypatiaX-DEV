#!/usr/bin/env bash
#
# run_solve_rate_all.sh — run compute_solve_rate.py over every leaf results
# directory under a root, one directory at a time, and print a final
# per-directory summary table.
#
# WHY A LOOP INSTEAD OF ONE BIG NUMBER
# -------------------------------------
# compute_solve_rate.py is intentionally scoped to one directory at a time
# so you can see which files actually contributed in each folder. This
# wrapper preserves that: it does NOT sum every directory into one grand
# total by default, because:
#   - different directories are different experiments (ablation, extrapolation,
#     noise-sweep, sample-complexity, pca variants, ...) and silently adding
#     their pass/fail counts together would blend incomparable conditions
#     into a single misleading "solve rate."
#   - some directories (e.g. */_saved/) are verbatim duplicates of files
#     that already exist one level up (confirmed for exp2_extrap/_saved and
#     exp2_pca_4060/_saved — identical filenames to their parent dir), so
#     including both would double-count the same rows.
#
# What it does:
#   1. Finds every directory under <root> that directly contains at least
#      one *.json file.
#   2. Skips directories named "_saved" by default (duplicate archives —
#      see above). Use --include-saved to keep them.
#   3. Runs compute_solve_rate.py on each, with your chosen --threshold /
#      --method-filter / --exclude / --source, saving full output per dir.
#   4. Prints a compact per-directory summary table at the end (directory,
#      pass/total, rate) — clearly labeled per-directory, never pooled
#      across directories, so you can see at a glance where a real gap is,
#      and can decide for yourself whether combining any specific subset
#      makes scientific sense.
#
# USAGE
# -----
#   ./run_solve_rate_all.sh <root_dir> [compute_solve_rate.py args...]
#
#   # Matches Gate C's own documented method-filter criteria:
#   ./run_solve_rate_all.sh hypatiax/data/results \
#       --threshold 0.999999 \
#       --method-filter hypatiax,hybridv50,hybrid50,hybridsymbolic,hybriddefi,hypatia,hybrid,ours,proposed
#
#   # Include _saved/ duplicate-archive directories too:
#   ./run_solve_rate_all.sh hypatiax/data/results --include-saved --threshold 0.999999
#
# All arguments after <root_dir> (other than --include-saved, which this
# wrapper consumes) are passed straight through to compute_solve_rate.py.

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <root_dir> [compute_solve_rate.py args...]" >&2
  exit 1
fi

ROOT="$1"; shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPUTE_SCRIPT="${SCRIPT_DIR}/compute_solve_rate.py"
if [ ! -f "$COMPUTE_SCRIPT" ]; then
  # fall back to same-directory-as-invocation, or PATH lookup
  if [ -f "./compute_solve_rate.py" ]; then
    COMPUTE_SCRIPT="./compute_solve_rate.py"
  else
    echo "ERROR: could not find compute_solve_rate.py next to this script or in ." >&2
    exit 1
  fi
fi

INCLUDE_SAVED=0
PASSTHROUGH_ARGS=()
for arg in "$@"; do
  if [ "$arg" == "--include-saved" ]; then
    INCLUDE_SAVED=1
  else
    PASSTHROUGH_ARGS+=("$arg")
  fi
done

if [ ! -d "$ROOT" ]; then
  echo "ERROR: root directory not found: $ROOT" >&2
  exit 1
fi

OUT_DIR="$(mktemp -d)"
echo "=== Scanning $ROOT for leaf directories containing .json files ==="

# Find directories that directly contain at least one *.json file.
mapfile -t DIRS < <(find "$ROOT" -type f -name '*.json' -exec dirname {} \; | sort -u)

if [ "$INCLUDE_SAVED" -eq 0 ]; then
  FILTERED=()
  for d in "${DIRS[@]}"; do
    base="$(basename "$d")"
    if [ "$base" == "_saved" ]; then
      echo "  (skipping $d — _saved/ archive dir, duplicates its parent; use --include-saved to keep)"
      continue
    fi
    FILTERED+=("$d")
  done
  DIRS=("${FILTERED[@]}")
fi

echo "Found ${#DIRS[@]} director$([ "${#DIRS[@]}" -eq 1 ] && echo y || echo ies) to process."
echo

declare -a SUMMARY_LINES=()

i=0
for d in "${DIRS[@]}"; do
  i=$((i+1))
  safe_name="$(echo "$d" | tr '/ ' '__')"
  out_file="${OUT_DIR}/${safe_name}.out"

  echo "############################################################"
  echo "### [$i/${#DIRS[@]}] $d"
  echo "############################################################"

  if python3 "$COMPUTE_SCRIPT" "$d" "${PASSTHROUGH_ARGS[@]}" > "$out_file" 2>&1; then
    cat "$out_file"
  else
    echo "  !! compute_solve_rate.py failed on this directory (see below)"
    cat "$out_file"
  fi
  echo

  # Pull out the pooled/total line (if any) for the end-of-run summary.
  # We deliberately keep this per-directory — it is NOT summed further.
  summary_line="$(grep -E '^\s*(POOLED|TOTAL):' "$out_file" | tail -1 || true)"
  if [ -z "$summary_line" ]; then
    # single-method dirs print "  method  p/t  rate=..." with no POOLED/TOTAL
    # line when there's exactly one method row and total_all is falsy; fall
    # back to showing "no data" so the dir isn't silently dropped from the table.
    summary_line="  (no matching rows)"
  fi
  SUMMARY_LINES+=("$d :: ${summary_line#  }")
done

echo "============================================================"
echo "=== Per-directory summary (NOT pooled across directories) ==="
echo "============================================================"
for line in "${SUMMARY_LINES[@]}"; do
  echo "$line"
done
echo
echo "Full per-directory output saved under: $OUT_DIR"
echo "(These are NOT combined into one grand total on purpose — see the"
echo " header comment in this script for why. If you want a number for a"
echo " specific, deliberately-chosen subset of directories, run"
echo " compute_solve_rate.py by hand on just those, or point it at a single"
echo " already-consolidated flat file if one genuinely covers that subset.)"

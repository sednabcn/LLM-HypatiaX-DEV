#!/usr/bin/env bash
# fix_model_strings.sh
#
# Replaces stale "claude-sonnet-4-5" references with "claude-sonnet-4-6"
# in the live hypatiax/ + hypatiax-orchestrator/ trees, then verifies
# no stale references remain and prints the diff for review.
#
# Usage: run from the repo root (LLM-HypatiaX-DEV/)
#   bash fix_model_strings.sh

set -euo pipefail

FILES=(
  "hypatiax/experiments/benchmarks/exp1_ablation.py"
  "hypatiax/experiments/benchmarks/hypatia.py"
  "hypatiax-orchestrator/repro_master.py"
  "hypatiax-orchestrator/run_all_checkpoint.py"
)

# Files needing the older claude-sonnet-4-20250514 -> claude-sonnet-4-6 fix
# (comments / doc strings, not the "-4-5" pattern above).
LEGACY_FILES=(
  "hypatiax/reproducibility/HypatiaX_Step_By_Step_Guide.html"
  "hypatiax/run_all.sh"
)

echo "========================================================================"
echo "STEP 1: Pre-flight — confirm all target files exist"
echo "========================================================================"
missing=0
for f in "${FILES[@]}" "${LEGACY_FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "  MISSING: $f"
    missing=1
  else
    echo "  found:   $f"
  fi
done
if [[ "$missing" -eq 1 ]]; then
  echo
  echo "ERROR: one or more target files not found. Run this from the repo root."
  exit 1
fi

echo
echo "========================================================================"
echo "STEP 2: Show occurrences BEFORE the fix"
echo "========================================================================"
grep -n "claude-sonnet-4-5\b" "${FILES[@]}" || echo "  (none found in FILES — already clean?)"
grep -n "claude-sonnet-4-20250514" "${LEGACY_FILES[@]}" || echo "  (none found in LEGACY_FILES — already clean?)"

echo
echo "========================================================================"
echo "STEP 3: Apply sed replacement (claude-sonnet-4-5 -> claude-sonnet-4-6)"
echo "========================================================================"
for f in "${FILES[@]}"; do
  sed -i.bak 's/"claude-sonnet-4-5"/"claude-sonnet-4-6"/g' "$f"
  echo "  patched: $f"
done

echo
echo "========================================================================"
echo "STEP 3b: Apply sed replacement (claude-sonnet-4-20250514 -> claude-sonnet-4-6)"
echo "         in legacy comment/doc files"
echo "========================================================================"
for f in "${LEGACY_FILES[@]}"; do
  sed -i.bak 's/claude-sonnet-4-20250514/claude-sonnet-4-6/g' "$f"
  echo "  patched: $f"
done

echo
echo "========================================================================"
echo "STEP 4: Verify — no stale claude-sonnet-4-5 or claude-sonnet-4-20250514"
echo "        remain in live code/docs under hypatiax/ or hypatiax-orchestrator/"
echo "        (excludes .bak backups and historical data/results run logs,"
echo "         which are records of past runs and should not be edited)"
echo "========================================================================"
stale=$(grep -rn "claude-sonnet-4-20250514\|claude-sonnet-4-5\b" hypatiax/ hypatiax-orchestrator/ 2>/dev/null \
  | grep -v '\.bak:' \
  | grep -v '/data/results/' \
  || true)
if [[ -n "$stale" ]]; then
  echo "$stale"
  echo
  echo "WARNING: stale references still found above — review manually."
else
  echo "  CLEAN — no stale model strings remain in live hypatiax/ or hypatiax-orchestrator/ files"
fi

echo
echo "========================================================================"
echo "STEP 5: Confirm replacement count per file matches expectation"
echo "========================================================================"
declare -A EXPECTED=(
  ["hypatiax/experiments/benchmarks/exp1_ablation.py"]=1
  ["hypatiax/experiments/benchmarks/hypatia.py"]=1
  ["hypatiax-orchestrator/repro_master.py"]=3
  ["hypatiax-orchestrator/run_all_checkpoint.py"]=1
)
all_match=1
for f in "${FILES[@]}"; do
  count=$(grep -c '"claude-sonnet-4-6"' "$f" || true)
  expected="${EXPECTED[$f]}"
  if [[ "$count" -ge "$expected" ]]; then
    echo "  OK   $f — found $count occurrence(s) of claude-sonnet-4-6 (expected >= $expected)"
  else
    echo "  FAIL $f — found $count occurrence(s), expected >= $expected"
    all_match=0
  fi
done

declare -A LEGACY_EXPECTED=(
  ["hypatiax/reproducibility/HypatiaX_Step_By_Step_Guide.html"]=1
  ["hypatiax/run_all.sh"]=2
)
for f in "${LEGACY_FILES[@]}"; do
  count=$(grep -c 'claude-sonnet-4-6' "$f" || true)
  expected="${LEGACY_EXPECTED[$f]}"
  if [[ "$count" -ge "$expected" ]]; then
    echo "  OK   $f — found $count occurrence(s) of claude-sonnet-4-6 (expected >= $expected)"
  else
    echo "  FAIL $f — found $count occurrence(s), expected >= $expected"
    all_match=0
  fi
done

echo
echo "========================================================================"
echo "STEP 6: git diff for review"
echo "========================================================================"
git diff "${FILES[@]}" "${LEGACY_FILES[@]}" || echo "  (not a git repo, or git not available — skipping diff)"

echo
echo "========================================================================"
echo "STEP 7: Clean up .bak files (sed -i.bak backups), if all checks passed"
echo "========================================================================"
if [[ "$all_match" -eq 1 ]]; then
  for f in "${FILES[@]}" "${LEGACY_FILES[@]}"; do
    rm -f "${f}.bak"
  done
  echo "  Removed .bak backup files."
  echo
  echo "DONE — all six files patched and verified clean."
else
  echo "  Leaving .bak backup files in place since one or more checks failed."
  echo "  Review the output above before committing."
fi

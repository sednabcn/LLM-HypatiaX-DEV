#!/usr/bin/env bash
# Local replica of the paper_check CI job (.github/workflows/ci_postprocess.yml).
# Writes ONLY to a scratch dir; committed tables/paper files are never modified.
# Run from the repo root:  bash paper_check_local.sh
set -uo pipefail

[ -f config/experiments.yml ] || { echo "Run this from the repo root (config/experiments.yml not found)."; exit 2; }

SCRATCH="${SCRATCH:-/tmp/paper_check_local}"
rm -rf "$SCRATCH"; mkdir -p "$SCRATCH/tables"
LOG="$SCRATCH/generate_tables.log"; : > "$LOG"

ROOT="$(python3 -c "import yaml;print(yaml.safe_load(open('config/experiments.yml'))['results_root'])")"
echo "results_root = $ROOT"
echo "scratch      = $SCRATCH"

echo "== 1/3 regenerate all tables =="
python3 scripts/generate_tables.py \
  --results-dir "$ROOT" --output-dir "$SCRATCH/tables" 2>&1 | tee -a "$LOG"
python3 scripts/generate_nguyen12_fiveseed.py \
  --results-dir "$ROOT" --output-dir "$SCRATCH/tables" 2>&1 | tee -a "$LOG"
echo "Generated $(ls "$SCRATCH/tables"/*.tex 2>/dev/null | wc -l) table file(s)."

echo; echo "== skipped tables (reason) =="
grep -E "SKIP" "$LOG" || echo "(none)"

echo; echo "== 2/3 check paper tables (strict NOSOURCE) =="
EXC=""
[ -f paper/paper_table_exceptions.json ] && EXC="--exceptions paper/paper_table_exceptions.json"
python3 scripts/check_paper_tables.py \
  --tables "$SCRATCH/tables" \
  --docs paper/*.tex \
  $EXC \
  --strict-nosource \
  --report "$SCRATCH/paper_table_report.md" \
  --emit-template "$SCRATCH/todo_exceptions.json"
RC=$?
echo "checker exit code: $RC (non-zero = CI would fail)"

echo; echo "== 3/3 rows to send back =="
grep -n -i -E "randomsplit|time_noise|winrate" "$SCRATCH/paper_table_report.md" \
  || echo "(no matching rows; see full report below)"

echo; echo "Full report:  $SCRATCH/paper_table_report.md"
echo "Exceptions template: $SCRATCH/todo_exceptions.json"
echo "Generator log: $LOG"
exit $RC

NAME="${1:?usage: bash run_check.sh before|after}"
ROOT="$(python3 -c "import yaml;print(yaml.safe_load(open('config/experiments.yml'))['results_root'])")"
OUT="/tmp/tables_$NAME"
rm -rf "$OUT" && mkdir -p "$OUT"
python3 scripts/generate_tables.py --results-dir "$ROOT" --output-dir "$OUT" > "/tmp/gen_$NAME.log" 2>&1
python3 scripts/generate_nguyen12_fiveseed.py --results-dir "$ROOT" --output-dir "$OUT" >> "/tmp/gen_$NAME.log" 2>&1
python3 scripts/check_paper_tables.py --tables "$OUT" --docs paper/*.tex \
  --exceptions paper/paper_table_exceptions.json --strict-nosource \
  --report "/tmp/report_$NAME.md" > /dev/null
echo "check exit code: $?"
grep -E "^\| tab:" "/tmp/report_$NAME.md" | cut -d'|' -f2,4,5 | sort > "/tmp/local_$NAME.txt"
echo "saved /tmp/local_$NAME.txt ($(wc -l < /tmp/local_$NAME.txt) rows)"
if [ "$NAME" = "before" ] && [ -f dup_check.py ]; then
  echo "--- duplicate-record check ---"
  python3 dup_check.py "$ROOT"
fi

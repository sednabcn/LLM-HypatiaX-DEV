#!/usr/bin/env python3
"""
Runs the EXACT aggregation logic from run_all.sh (the [FIX-C3] block,
lines ~1416-1489) against the real exp2_pca_4060/ shard directory, to
settle definitively whether this code can even produce a nonzero n_total
against the current shard schema -- without touching/overwriting the real
exp2_pca_4060_summary.json (writes to a sibling file instead).
"""
import glob, json, pathlib, sys

PCA_DIR   = pathlib.Path("hypatiax/data/results/comparison_results/feynman-tests/exp2_pca_4060")
SUMMARY   = PCA_DIR / "exp2_pca_4060_summary_TEST_DRYRUN.json"  # do NOT touch the real file
THRESHOLD = 0.999999
PREFERRED = {'hypatiax','hybridv50','hybrid50','hybridsymbolic',
             'hybriddefi','hypatia','hybrid','ours','proposed'}

def _r2(row):
    for k in ('r2','r2_test','r2_train','best_r2','R2'):
        v = row.get(k)
        if v is not None:
            try:
                f = float(v)
                if f <= 1.01:
                    return f
            except (TypeError, ValueError):
                pass
    return None

def _rows(data):
    if isinstance(data, dict):
        for key in ('results','equation_results','data','rows'):
            v = data.get(key)
            if v is not None:
                yield from _rows(v)
                return
        yield data
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item

n_pass = n_total = 0
source_files = []
files_seen = 0
rows_yielded_total = 0
rows_with_r2 = 0

for fp in sorted(PCA_DIR.glob('*.json')) if PCA_DIR.exists() else []:
    if any(x in fp.name for x in ('checkpoint','disclosure','summary','baseline')):
        continue
    try:
        data = json.loads(fp.read_text())
    except Exception:
        continue
    files_seen += 1
    source_files.append(fp.name)
    for row in _rows(data):
        rows_yielded_total += 1
        raw    = row.get('method') or row.get('model') or ''
        method = str(raw).lower().replace('-','').replace('_','').replace(' ','')
        if method and not any(p in method for p in PREFERRED):
            continue
        r2 = _r2(row)
        if r2 is None:
            continue
        rows_with_r2 += 1
        n_total += 1
        if r2 >= THRESHOLD:
            n_pass += 1

print(f"Files matched (non-checkpoint/disclosure/summary/baseline): {files_seen}")
print(f"Total 'rows' yielded by _rows() across all files: {rows_yielded_total}")
print(f"  (if this equals {files_seen}, _rows() is yielding one row PER FILE -- confirms schema mismatch)")
print(f"Rows with a usable r2 value: {rows_with_r2}")
print(f"n_pass: {n_pass}")
print(f"n_total: {n_total}")
print(f"solve_rate: {(n_pass/n_total) if n_total else None}")

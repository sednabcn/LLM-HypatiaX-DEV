#!/usr/bin/env python3
"""
Same idea as test_fixc3_aggregation.py, but pointed at the LEGACY
exp2/ (random_80_20) directory that fixc3_baseline.json was built from,
to check whether it's affected by the same stale-ENV-FAIL issue that
exp2_pca_4060 had. Read-only -- writes nothing.
"""
import glob, json, pathlib

LEG_DIR   = pathlib.Path("hypatiax/data/results/comparison_results/feynman-tests/exp2")
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

def run(threshold):
    n_pass = n_total = 0
    hds_env_fail = 0
    hds_seen = 0
    files_seen = 0
    for fp in sorted(LEG_DIR.glob('*.json')) if LEG_DIR.exists() else []:
        if any(x in fp.name for x in ('checkpoint','disclosure','summary','baseline')):
            continue
        try:
            data = json.loads(fp.read_text())
        except Exception:
            continue
        files_seen += 1
        for row in _rows(data):
            raw = row.get('method') or row.get('model') or ''
            method_lc = str(raw).lower()
            if 'hybriddiscoverysystem' in method_lc.replace(' ', '') or 'v50_2' in method_lc:
                hds_seen += 1
                if row.get('error') and ('not available' in str(row.get('error')).lower()):
                    hds_env_fail += 1
            method = method_lc.replace('-','').replace('_','').replace(' ','')
            if method and not any(p in method for p in PREFERRED):
                continue
            r2 = _r2(row)
            if r2 is None:
                continue
            n_total += 1
            if r2 >= threshold:
                n_pass += 1
    return files_seen, n_pass, n_total, hds_seen, hds_env_fail

if not pathlib.Path("hypatiax/data/results/comparison_results/feynman-tests/exp2").exists():
    print("LEGACY exp2/ directory not found -- shards may have been removed/archived.")
    print("If so, fixc3_baseline.json cannot be re-verified and should be treated as")
    print("frozen historical record (or regenerated from whatever legacy source remains).")
else:
    for t in (0.999999, 0.99):
        files_seen, n_pass, n_total, hds_seen, hds_env_fail = run(t)
        print(f"THRESHOLD={t}: files={files_seen}  n_pass={n_pass}  n_total={n_total}  "
              f"rate={(n_pass/n_total if n_total else None)}")
        print(f"  HDS v50_2 rows seen={hds_seen}  ENV-FAIL/'not available'={hds_env_fail}")

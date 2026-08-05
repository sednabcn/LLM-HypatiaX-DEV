#!/usr/bin/env python3
"""
Per-method pass/total breakdown across all protocol_core_noiseless_pca_*.json
shards in exp2_pca_4060/, using R2>=0.9999 as the strict threshold (matching
check_run_health.py Check 10) so we can see exactly which method(s) drove the
106/180 -> 142/180 jump.

Run from repo root:
    python3 aggregate_pca_shards.py
"""
import json
import glob
import sys
from collections import Counter, defaultdict

SHARD_DIR = "hypatiax/data/results/comparison_results/feynman-tests/exp2_pca_4060"
THRESHOLD = 0.9999

def main():
    files = sorted(glob.glob(f"{SHARD_DIR}/protocol_core_noiseless_pca_*.json"))
    if not files:
        print(f"No shard files found in {SHARD_DIR}")
        sys.exit(1)

    pass_by_method = Counter()
    total_by_method = Counter()
    seen_test_keys = set()
    duplicate_tests = 0

    for fpath in files:
        d = json.load(open(fpath))
        for test in d.get("tests", []):
            # de-dup in case a test appears in more than one shard file
            key = (test.get("domain"), test.get("description"))
            for method, res in test.get("results", {}).items():
                # count every occurrence per (file, test, method) — but flag
                # if the same (domain, description, method) combo appears
                # in more than one shard, since that would inflate totals
                full_key = (key, method)
                if full_key in seen_test_keys:
                    duplicate_tests += 1
                seen_test_keys.add(full_key)

                total_by_method[method] += 1
                success = res.get("success")
                r2 = res.get("r2")
                if success and r2 is not None and r2 >= THRESHOLD:
                    pass_by_method[method] += 1

    print(f"Shards read: {len(files)}")
    print(f"Duplicate (domain, description, method) combos across shards: {duplicate_tests}")
    print()
    print(f"{'Method':45} {'Pass':>6} {'Total':>6} {'Rate':>8}")
    print("-" * 68)
    grand_pass, grand_total = 0, 0
    for method in total_by_method:
        p, t = pass_by_method[method], total_by_method[method]
        grand_pass += p
        grand_total += t
        print(f"{method:45} {p:>6} {t:>6} {p/t:>8.3f}")
    print("-" * 68)
    print(f"{'TOTAL (pooled, all methods)':45} {grand_pass:>6} {grand_total:>6} {grand_pass/grand_total:>8.3f}")

if __name__ == "__main__":
    main()

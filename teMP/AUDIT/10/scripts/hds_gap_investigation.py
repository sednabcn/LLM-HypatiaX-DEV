#!/usr/bin/env python3
"""
Item: HDS delta investigation (still open)
-------------------------------------------
Table 4 (supp_benchmark_report.tex) reports HDS = 27/30 (90.0%) on the
Feynman-30 noiseless six-method comparison.

Today's exp2_feynman reproduction (2026-08-07, all 11 domains, 30/30 tests,
zero harness errors) instead shows HDS = 22/30 @0.9999 (21/30 @0.999999) —
a 5-point gap unexplained so far.

This script pulls every HDS ("HybridDiscoverySystem v50_2 (tools)") result
from today's 11 protocol_core_noiseless_20260807_*.json files, flags the
failing cases against the paper's stated threshold (R^2 >= 0.9999), and
classifies each failure as:
  - TIMEOUT-LIKE   : wall time close to the configured PySR/method timeout
                     (>= ~85% of 1100s, or >= ~85% of 900s outer budget)
  - NEAR-MISS      : R^2 below threshold but close (>= 0.99)
  - GENUINE-MISS   : R^2 well below threshold (< 0.99) without timeout signal
  - ERROR          : harness reported a hard error/exception

Usage: python3 hds_gap_investigation.py [glob]
"""
import json
import glob
import sys

METHOD = "HybridDiscoverySystem v50_2 (tools)"
THRESHOLD = 0.9999
PYSR_TIMEOUT = 1100
METHOD_TIMEOUT = 900
TIMEOUT_FRACTION = 0.85  # flag as timeout-like if time >= 85% of a known budget


def classify(r2, elapsed, error):
    if error:
        return "ERROR"
    if r2 is None:
        return "ERROR"
    if r2 >= THRESHOLD:
        return "PASS"
    # timeout-like: wall time sitting near either the inner PySR
    # timeout (1100s) or the outer per-method timeout (900s)
    near_pysr_to = elapsed >= PYSR_TIMEOUT * TIMEOUT_FRACTION
    near_method_to = elapsed >= METHOD_TIMEOUT * TIMEOUT_FRACTION
    if near_pysr_to or near_method_to:
        return "TIMEOUT-LIKE"
    if r2 >= 0.99:
        return "NEAR-MISS"
    return "GENUINE-MISS"


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else "protocol_core_noiseless_20260807_*.json"
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files matched: {pattern}")
        return

    rows = []
    n_total = 0
    n_pass = 0

    for f in files:
        d = json.load(open(f))
        for t in d.get("tests", []):
            n_total += 1
            r = t["results"].get(METHOD)
            if r is None:
                continue
            r2 = r.get("r2")
            elapsed = r.get("time", 0.0) or 0.0
            error = r.get("error")
            success = r.get("success")
            label = classify(r2, elapsed, error)
            if label == "PASS":
                n_pass += 1
                continue
            rows.append({
                "file": f,
                "domain": t["domain"],
                "description": t["description"],
                "r2": r2,
                "time_s": round(elapsed, 1),
                "success": success,
                "error": error,
                "metadata": r.get("metadata", {}),
                "label": label,
            })

    print(f"HDS ({METHOD})")
    print(f"Pass @ R2>={THRESHOLD}: {n_pass}/{n_total}\n")

    if not rows:
        print("No failing cases found.")
        return

    print(f"{'Label':<14} {'Domain':<24} {'R2':>12} {'Time(s)':>8}  Description")
    print("-" * 100)
    for row in sorted(rows, key=lambda x: (x["label"], -x["time_s"])):
        r2_str = f"{row['r2']:.6f}" if row["r2"] is not None else "None"
        print(f"{row['label']:<14} {row['domain']:<24} {r2_str:>12} {row['time_s']:>8}  {row['description']}")

    print("\n--- Detail ---")
    for row in sorted(rows, key=lambda x: (x["label"], -x["time_s"])):
        print(f"\n[{row['label']}] {row['domain']} — {row['description']}")
        print(f"  file: {row['file']}")
        print(f"  r2={row['r2']}  time={row['time_s']}s  success={row['success']}  error={row['error']}")
        if row["metadata"]:
            print(f"  metadata: {row['metadata']}")

    # summary counts by label
    from collections import Counter
    counts = Counter(r["label"] for r in rows)
    print("\n--- Summary ---")
    for label, n in counts.most_common():
        print(f"  {label}: {n}")


if __name__ == "__main__":
    main()

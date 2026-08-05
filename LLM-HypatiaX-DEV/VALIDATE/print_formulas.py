#!/usr/bin/env python3
"""Print the actual formula string for each null equation -- no re-run needed."""
import argparse, glob, json, os

ap = argparse.ArgumentParser()
ap.add_argument("directory")
ap.add_argument("--method", required=True)
ap.add_argument("--pattern", default="protocol_core_*.json")
args = ap.parse_args()

CANDIDATES = ("y", "result", "output", "pred", "f")

for fp in sorted(glob.glob(os.path.join(args.directory, args.pattern))):
    with open(fp) as f:
        data = json.load(f)
    records = data.get("tests", data if isinstance(data, list) else [data])
    for rec in records:
        far_map = rec.get("extrap_r2_far") or {}
        if args.method not in far_map or far_map[args.method] is not None:
            continue  # only show the nulls
        block = (rec.get("results") or {}).get(args.method, {})
        formula = (block.get("formula") or "").strip()
        eq = rec.get("description", rec.get("equation_name", ""))
        # cheap check: does the formula assign to one of the 5 whitelisted names?
        has_whitelisted_lhs = any(
            formula.replace(" ", "").startswith(c + "=") for c in CANDIDATES
        )
        flag = "OK (uses whitelisted name)" if has_whitelisted_lhs else "SUSPECT (non-whitelisted LHS)"
        print(f"[{rec.get('domain','')}] {eq}")
        print(f"    formula: {formula!r}")
        print(f"    -> {flag}\n")

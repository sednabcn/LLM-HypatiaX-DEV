#!/usr/bin/env python3
"""Recompute tab:winrate (M3 vs M4 head-to-head) from shards, under several tie rules.
Records are paired by (shard file, test description). Run once per sweep:
  python compute_winrate.py --dir <noise-sweep dir>
  python compute_winrate.py --dir <sample-complexity dir> --pattern 'protocol_core_*.json'
Reports every rule so you can see which, if any, reproduces the claimed 13.3/20.7/66.0 split."""
import argparse, collections
from hypatiax_common import load, inspect

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True); ap.add_argument("--pattern", default="protocol_core_*.json")
ap.add_argument("--inspect", action="store_true")
a = ap.parse_args()
if a.inspect: inspect(a.dir, a.pattern); raise SystemExit
R = load(a.dir, a.pattern)
m = collections.defaultdict(dict)
for r in R:
    if r["which"]: m[(r["file"], r["test"])][r["which"]] = r["r2"]
pairs = [(v["M3"], v["M4"]) for v in m.values() if "M3" in v and "M4" in v]
n = len(pairs)
print(f"pairs: {n}   (unpaired keys: {sum(1 for v in m.values() if len(v)<2)})")
print("pairing: by (shard file, test description)")
def report(label, win3, win4):
    t = n - win3 - win4
    print(f"{label:34s} M3>{win3:4d} ({100*win3/n:5.1f}%)  M4>{win4:4d} ({100*win4/n:5.1f}%)  tie {t:4d} ({100*t/n:5.1f}%)")
print("\nclaimed (noise): M3 20/150 13.3%, M4 31/150 20.7%, tie 99/150 66.0%\n")
report("exact comparison", sum(x > y for x, y in pairs), sum(y > x for x, y in pairs))
for tol in (1e-12, 1e-9, 1e-6, 1e-4, 1e-3):
    report(f"|diff| <= {tol:g} is tie", sum(x - y > tol for x, y in pairs), sum(y - x > tol for x, y in pairs))
t = sum(x > 0.9999 and y > 0.9999 for x, y in pairs)
print(f"\nboth R2>0.9999 (paper's tie rule): {t}/{n} ({100*t/n:.1f}%)")
w3 = sum(x > y for x, y in pairs if not (x > 0.9999 and y > 0.9999))
w4 = sum(y > x for x, y in pairs if not (x > 0.9999 and y > 0.9999))
report("strict wins outside >0.9999 tie", w3, w4)

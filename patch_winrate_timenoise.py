#!/usr/bin/env python3
"""Patch scripts/generate_tables.py so tab:winrate / tab:time_noise match the paper's formatting.
  1. gen_suppb_time_noise: M3/M4 averages printed with 2 decimals (paper prints 7.55, 6.92).
  2. gen_suppb_winrate:    R^2 -> \\Rsq in the three row labels (paper macro; removes the stray '2').
Usage: python patch_winrate_timenoise.py scripts/generate_tables.py [--apply]
Each anchor must occur exactly once; otherwise nothing is written."""
import sys, shutil
path = sys.argv[1]; apply = "--apply" in sys.argv
s = open(path, encoding="utf-8").read()
EDITS = [
 ('t3_str = f"{t3:.1f}" if isinstance(t3, float) else "---"',
  't3_str = f"{t3:.2f}" if isinstance(t3, float) else "---"'),
 ('t4_str = f"{t4:.1f}" if isinstance(t4, float) else "---"',
  't4_str = f"{t4:.2f}" if isinstance(t4, float) else "---"'),
 ('f"M3 strictly higher $R^2$ & ', 'f"M3 strictly higher $\\\\Rsq$ & '),
 ('f"M4 strictly higher $R^2$ & ', 'f"M4 strictly higher $\\\\Rsq$ & '),
 ('f"Tied ($R^2 > 0.9999$)    & ', 'f"Tied ($\\\\Rsq > 0.9999$)    & '),
]
done = [new in s and old not in s for old, new in EDITS]
if all(done):
    print("already patched; nothing to do"); sys.exit(0)
bad = [old for old, new in EDITS if s.count(old) != 1]
if bad:
    print("ABORT, anchor not found exactly once:"); [print("  ", b) for b in bad]; sys.exit(1)
for old, new in EDITS: s = s.replace(old, new)
print(f"{len(EDITS)} anchors found, patch is applicable.")
if apply:
    shutil.copy(path, path + ".bak2")
    open(path, "w", encoding="utf-8").write(s)
    print(f"patched {path} (backup {path}.bak2)")
else:
    print("dry run: re-run with --apply to write it.")

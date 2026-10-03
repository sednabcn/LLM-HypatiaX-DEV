#!/usr/bin/env python3
"""Make the 0.99 fixed-threshold table carry the label the paper uses (tab:noise_sensitivity_fixed).
Other thresholds keep their tagged label, so labels cannot collide.
Usage: python patch_fixed_label.py scripts/generate_tables.py [--apply]"""
import sys, shutil
path = sys.argv[1]; apply = "--apply" in sys.argv
s = open(path, encoding="utf-8").read()
old = 'f"tab:noise_sensitivity_fixed_{tag}",'
new = '("tab:noise_sensitivity_fixed" if tag == "0p99" else f"tab:noise_sensitivity_fixed_{tag}"),'
if new in s and old not in s:
    print("already patched; nothing to do"); sys.exit(0)
if s.count(old) != 1:
    print(f"ABORT: anchor found {s.count(old)} times (need exactly 1)"); sys.exit(1)
s = s.replace(old, new)
print("1 anchor found, patch is applicable.")
if apply:
    shutil.copy(path, path + ".bak3"); open(path, "w", encoding="utf-8").write(s)
    print(f"patched {path} (backup {path}.bak3)")
else:
    print("dry run: re-run with --apply to write it.")

#!/usr/bin/env python3
"""Tidy paper/paper_table_exceptions.json (flat label -> reason dict).
Removes entries that are stale or point at labels in no document; optionally adds new ones.
Usage: python update_exceptions.py paper/paper_table_exceptions.json [--apply] [--add-feynman30]"""
import json, sys, shutil
p = sys.argv[1]; apply = "--apply" in sys.argv
d = json.load(open(p, encoding="utf-8"))
REMOVE = ["tab:figlist_complete","tab:figlist_main","tab:figlist_supplementary",
          "tab:feynman30-legacy","tab:arch","tab:winrate","tab:time_noise"]
for k in REMOVE:
    print(("remove " if k in d else "absent ") + k); d.pop(k, None)
if "--add-feynman30" in sys.argv:
    d["tab:feynman30-methods"] = ("Pass counts for the random-split and PCA-reported columns recomputed by "
        "scripts/check_feynman30_counts.py from the per-equation rows of tab:randomsplit and tab:pcasplit; "
        "the PCA held-out column (SE 17/30, HD 13/30) comes from extrap_r2_far and has no generator "
        "(generate_tables.py feynman.tex is skipped: no parsable equation rows).")
    print("add    tab:feynman30-methods")
if apply:
    shutil.copy(p, p + ".bak")
    json.dump(d, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False); open(p, "a").write("\n")
    print("written; backup", p + ".bak")
else:
    print("dry run: add --apply to write")

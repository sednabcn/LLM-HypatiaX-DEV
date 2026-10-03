#!/usr/bin/env python3
"""Cross-check tab:feynman30-methods against the paper's own per-equation tables.
Counts bold (pass) cells per method column in tab:randomsplit and tab:pcasplit.
Usage: python check_feynman30_counts.py paper/jmlr_paper_main_cleaned.tex"""
import re, sys
tex = open(sys.argv[1], encoding="utf-8").read()
METHODS = ["PL","NN","EH","LN","SE","HD"]
# values printed in tab:feynman30-methods (random, pca-reported)
EXPECT = {"randomsplit": dict(PL=29,NN=0,EH=23,LN=30,SE=27,HD=21),
          "pcasplit":    dict(PL=29,NN=3,EH=23,LN=30,SE=23,HD=18)}
def table(label):
    m = re.search(r"\\label\{tab:%s\}" % label, tex)
    if not m: return None
    end = tex.find(r"\end{longtable}", m.end())
    return tex[m.end():end]
ok = True
for label, exp in EXPECT.items():
    body = table(label)
    if body is None: print(f"{label}: table not found"); ok = False; continue
    rows = []
    for ln in body.splitlines():
        if ln.rstrip().endswith(r"\\") and ln.count("&") == 7:
            rows.append([c.strip().rstrip("\\").strip() for c in ln.split("&")])
    got = {m: sum(r[2+i].startswith(r"\textbf") for r in rows) for i, m in enumerate(METHODS)}
    print(f"{label}: {len(rows)} rows")
    for m in METHODS:
        flag = "ok" if got[m] == exp[m] else "MISMATCH"
        if flag != "ok": ok = False
        print(f"   {m}: counted {got[m]}/30, tab:feynman30-methods says {exp[m]}/30  {flag}")
print("ALL CONSISTENT" if ok else "INCONSISTENT: see above")
sys.exit(0 if ok else 1)

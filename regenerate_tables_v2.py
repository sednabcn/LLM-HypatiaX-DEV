#!/usr/bin/env python3
"""tab:arch and tab:noise_sensitivity, with optional fixed solve threshold.
  python regenerate_tables_v2.py --dir <noise-sweep dir> --out tables_out [--fixed-threshold 0.99]
  python regenerate_tables_v2.py --dir <dir> --inspect
Per-run thresholds are used unless --fixed-threshold is given (repeat it for several)."""
import argparse, os, collections
from hypatiax_common import load, inspect

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True); ap.add_argument("--out", default="tables_out")
ap.add_argument("--pattern", default="protocol_core_*.json")
ap.add_argument("--fixed-threshold", type=float, action="append", default=[])
ap.add_argument("--inspect", action="store_true")
a = ap.parse_args()
if a.inspect: inspect(a.dir, a.pattern); raise SystemExit
R = load(a.dir, a.pattern); os.makedirs(a.out, exist_ok=True)
levels = sorted({r["noise"] for r in R if r["noise"] is not None})
print("noise levels present:", levels)
for lv in (0.02, 0.05, 0.10, 0.20):
    if lv not in levels: print(f"  note: level {lv} not present")

# ---- tab:arch
arch = collections.defaultdict(collections.Counter)
by_noise = collections.defaultdict(collections.Counter)
for r in R:
    if r["which"] and r["decision"]:
        arch[r["which"]][r["decision"]] += 1
        by_noise[(r["which"], r["noise"])][r["decision"]] += 1
print("\n== tab:arch ==")
for s in ("M3", "M4"):
    n = sum(arch[s].values())
    for d, c in arch[s].most_common(): print(f"{s} -> {d}: {c}/{n} ({100*c/n:.1f}%)")
with open(f"{a.out}/tab_arch_by_noise.tex", "w") as f:
    f.write("\\begin{tabular}{l r r r r}\n\\toprule\n$\\sigma/\\mathrm{std}(y)$ & M3 ensemble & M3 NN & M4 LLM & M4 ensemble\\\\\n\\midrule\n")
    for lv in levels:
        m3, m4 = by_noise[("M3", lv)], by_noise[("M4", lv)]
        f.write(f"{lv:g} & {m3['ensemble']} & {m3['nn']} & {m4['llm']} & {m4['ensemble']}\\\\\n")
    f.write(f"\\midrule\nTotal & {arch['M3']['ensemble']} & {arch['M3']['nn']} & {arch['M4']['llm']} & {arch['M4']['ensemble']}\\\\\n\\bottomrule\n\\end{{tabular}}\n")

# ---- tab:noise_sensitivity
def table(thr_fn, label, fname, note):
    cnt = collections.defaultdict(lambda: [0, 0])
    for r in R:
        t = thr_fn(r)
        if t is None: continue
        c = cnt[(r["method"], r["noise"])]; c[1] += 1; c[0] += r["r2"] >= t
    print(f"\n== {label} ==\n{note}")
    rows = []
    for s in sorted({r["method"] for r in R}):
        cells = [cnt[(s, lv)] for lv in levels]
        if not any(c[1] for c in cells): continue
        p, n = sum(c[0] for c in cells), sum(c[1] for c in cells)
        txt = [f"{c[0]}/{c[1]}" for c in cells] + [f"{p}/{n} ({100*p/n:.1f}\\%)"]
        print(f"{s:40s}", " | ".join(txt)); rows.append((s, txt))
    with open(f"{a.out}/{fname}", "w") as f:
        f.write("\\begin{tabular}{@{}l" + "c" * len(levels) + "r@{}}\n\\toprule\n\\textbf{Method} & "
                + " & ".join(f"\\textbf{{{lv:g}}}" for lv in levels) + " & \\textbf{Pooled}\\\\\n\\midrule\n")
        for s, txt in rows: f.write(s.replace("_", "\\_") + " & " + " & ".join(txt) + "\\\\\n")
        f.write("\\bottomrule\n\\end{{tabular}}\n")

table(lambda r: r["thr"], "per-run threshold", "tab_noise_sensitivity.tex",
      "NOTE: thresholds differ per run; not comparable across levels.")
for t in a.fixed_threshold:
    table(lambda r, t=t: t, f"fixed R2>={t}", f"tab_noise_sensitivity_fixed_{t}.tex",
          "Comparable across noise levels and valid to pool.")
print("\nwrote", a.out)

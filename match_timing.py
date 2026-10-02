#!/usr/bin/env python3
"""match_timing.py -- which DeFi seed files reproduce the paper's timing table (tab:timing_full)?

For every hypatiax_defi_benchmark*_results_seed*.json it recomputes the numbers the generator uses
(Pure LLM, Neural MLP, Hybrid mean/median, LLM-routed count) and compares them with the paper's row
for that seed and variant.  Read-only.

Run from the repo root:
    python3 match_timing.py                  (searches the DEV repo only)
    python3 match_timing.py --also DIR       (also reads DIR, e.g. to test a candidate before copying it in; repeatable)
    python3 match_timing.py --brief --also DIR   (only matches + summary)
"""
import argparse, json, os, re
from pathlib import Path

# paper rows, copied from tab:timing_full in jmlr_paper_main.tex: (llm mean/med, nn mean/med, hybrid mean/med, routed)
PAPER = {
 ("v3c", "42"):   ((9.69, 8.85), (0.33, 0.33), (1.99, 1.52), 55),
 ("v3c", "99"):   ((9.40, 8.74), (0.36, 0.36), (2.17, 1.57), 54),
 ("v3c", "123"):  ((9.46, 8.42), (0.36, 0.36), (1.93, 1.46), 55),
 ("v3c", "777"):  ((9.68, 8.92), (0.25, 0.25), (1.94, 1.48), 54),
 ("v3c", "2024"): ((9.43, 8.69), (0.25, 0.25), (1.92, 1.45), 55),
 ("PCA", "42"):   ((9.85, 8.95), (0.33, 0.33), (1.63, 1.34), 68),
 ("PCA", "99"):   ((9.53, 8.77), (0.33, 0.33), (1.67, 1.37), 69),
 ("PCA", "123"):  ((9.81, 8.87), (0.33, 0.33), (1.76, 1.36), 69),
 ("PCA", "777"):  ((9.44, 8.77), (0.33, 0.33), (1.72, 1.38), 68),
 ("PCA", "2024"): ((9.25, 8.34), (0.34, 0.33), (1.62, 1.35), 68),
}
SKIP = {".git", "node_modules", "__pycache__", "venv", ".venv"}

def mm(xs):
    xs = sorted(xs); n = len(xs)
    return sum(xs) / n, (xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2)

def stats(recs):
    nn, llm, hy, rt = [], [], [], 0
    for r in recs:
        cr = r.get("results", {}) or {}
        a = (cr.get("neural_network", {}) or {}).get("time_s")
        b = (cr.get("pure_llm", {}) or {}).get("time_s")
        h = cr.get("hybrid", {}) or {}
        c = h.get("time_s")
        if isinstance(a, (int, float)): nn.append(a)
        if isinstance(b, (int, float)): llm.append(b)
        if isinstance(c, (int, float)):
            hy.append(c); rt += h.get("decision") == "llm"
    if not (nn and llm and hy): return None
    return mm(llm), mm(nn), mm(hy), rt

def close(a, b): return all(abs(round(x, 2) - y) < 0.0051 for x, y in zip(a, b))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--also", action="append", default=[]); ap.add_argument("--brief", action="store_true", help="print only matching files and the final summary"); a = ap.parse_args()
    repo = Path.cwd().resolve(); roots = [repo] + [Path(x) for x in a.also]
    seen = set(); rows = []
    for r in roots:
        for dp, dn, fn in os.walk(r):
            dn[:] = [d for d in dn if d not in SKIP]
            for f in fn:
                if re.fullmatch(r"hypatiax_defi_benchmark.*_results_seed\d+\.json", f):
                    p = Path(dp) / f
                    if p.resolve() in seen: continue
                    seen.add(p.resolve()); rows.append(p)
    print(f"{len(rows)} seed files found\n")
    ok_by_key = {}
    for p in sorted(rows, key=lambda p: (("_pca_" in p.name), p.name, str(p))):
        try: d = json.loads(p.read_text())
        except Exception: print(f"  unreadable: {p}"); continue
        seed = re.search(r"seed(\d+)", p.name).group(1); var = "PCA" if "_pca_" in p.name else "v3c"
        where = str(p.relative_to(repo)) if repo in p.resolve().parents else str(p)
        if not (isinstance(d, list) and d and isinstance(d[0], dict) and "results" in d[0]):
            print(f"  {where}\n     not a per-task list"); continue
        st = stats(d); ref = PAPER.get((var, seed))
        if st is None: print(f"  {where}\n     no timing fields"); continue
        llm, nn, hy, rt = st
        tag = "no paper row" if ref is None else ("MATCHES paper row" if (close(llm, ref[0]) and close(nn, ref[1]) and close(hy, ref[2]) and rt == ref[3]) else "differs from paper row")
        if tag.startswith("MATCH"): ok_by_key[(var, seed)] = where
        if a.brief and not tag.startswith("MATCH"): continue
        print(f"  {where}\n     {var} seed{seed}  n={len(d)}  LLM {llm[0]:.2f}/{llm[1]:.2f}  NN {nn[0]:.2f}/{nn[1]:.2f}  hybrid {hy[0]:.2f}/{hy[1]:.2f}  routed {rt}   -> {tag}")
        if ref and tag.startswith("differs") and not a.brief:
            print(f"     paper:  LLM {ref[0][0]:.2f}/{ref[0][1]:.2f}  NN {ref[1][0]:.2f}/{ref[1][1]:.2f}  hybrid {ref[2][0]:.2f}/{ref[2][1]:.2f}  routed {ref[3]}")
    print("\nPaper rows reproduced by some file:", len(ok_by_key), "of", len(PAPER))
    for k in PAPER:
        print(f"  {k[0]} seed{k[1]}: {ok_by_key.get(k, 'NO FILE REPRODUCES IT')}")

if __name__ == "__main__":
    main()

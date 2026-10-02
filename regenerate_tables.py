#!/usr/bin/env python3
"""Regenerate tab:arch (routing) and tab:noise_sensitivity (solve counts)
directly from protocol_core_*.json files.

Rules (stated so the paper can quote them):
  * noise level  = file["protocol"]["noise_level"]  (not the filename)
  * solved       = r2 >= file["protocol"]["threshold"]  (each run's own threshold)
  * routed to X  = record["metadata"]["decision"] == X
  * the `success` flag is NOT used (M3/M4 hardcode success=True)

Usage:
  python regenerate_tables.py --dir hypatiax/data/results/comparison_results/feynman-tests/noise-sweep --out tables_out
"""
import argparse, collections, glob, json, os

SYS = {"M3": "EnhancedHybridSystemDeFi", "M4": "HybridSystemLLMNN"}


def load(d):
    rows, seen = [], set()
    for f in sorted(glob.glob(os.path.join(d, "protocol_core_*.json"))):
        j = json.load(open(f))
        p = j["protocol"]
        for t in j["tests"]:
            for method, rec in t.get("results", {}).items():
                key = (p["noise_level"], t["description"], method)
                if key in seen:
                    raise SystemExit(f"duplicate record {key} in {f}")
                seen.add(key)
                r2 = rec.get("r2")
                md = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
                rows.append(dict(
                    noise=p["noise_level"], thr=p["threshold"], method=method,
                    r2=r2 if isinstance(r2, (int, float)) else float("-inf"),
                    decision=md.get("decision"), file=os.path.basename(f)))
    return rows


def pct(a, b):
    return f"{100 * a / b:.1f}\\%" if b else "--"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="tables_out")
    a = ap.parse_args()
    rows = load(a.dir)
    os.makedirs(a.out, exist_ok=True)
    noises = sorted({r["noise"] for r in rows})
    methods = sorted({r["method"] for r in rows})

    # ---- tab:arch: routing counts for M3 and M4 ----
    arch = ["\\begin{tabular}{llrrr}", "\\toprule",
            "System & Routed to & Count & Total & Share \\\\", "\\midrule"]
    print("== tab:arch (routing) ==")
    for s, pre in SYS.items():
        recs = [r for r in rows if r["method"].startswith(pre)]
        c = collections.Counter(r["decision"] for r in recs)
        for dec, n in sorted(c.items(), key=lambda x: -x[1]):
            print(f"{s} routed to {dec}: {n}/{len(recs)} ({100*n/len(recs):.1f}%)")
            arch.append(f"{s} & {dec} & {n} & {len(recs)} & {pct(n, len(recs))} \\\\")
        for nz in noises:
            sub = [r for r in recs if r["noise"] == nz]
            cc = collections.Counter(r["decision"] for r in sub)
            print(f"   noise {nz}: {dict(cc)} of {len(sub)}")
    arch += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(a.out, "tab_arch.tex"), "w").write("\n".join(arch) + "\n")

    # ---- tab:noise_sensitivity: solved at each file's own threshold ----
    print("\n== tab:noise_sensitivity (r2 >= file threshold) ==")
    hdr = "Method | " + " | ".join(f"{n} (thr {next(r['thr'] for r in rows if r['noise']==n)})" for n in noises) + " | pooled"
    print(hdr)
    tex = ["\\begin{tabular}{l" + "r" * (len(noises) + 1) + "}", "\\toprule",
           "Method & " + " & ".join(str(n) for n in noises) + " & Pooled \\\\", "\\midrule"]
    for m in methods:
        cells, tot_s, tot_n = [], 0, 0
        for nz in noises:
            sub = [r for r in rows if r["method"] == m and r["noise"] == nz]
            s = sum(r["r2"] >= r["thr"] for r in sub)
            cells.append((s, len(sub)))
            tot_s += s
            tot_n += len(sub)
        print(f"{m} | " + " | ".join(f"{s}/{n}" for s, n in cells) + f" | {tot_s}/{tot_n} ({100*tot_s/tot_n:.1f}%)")
        tex.append(m.replace("_", "\\_") + " & " + " & ".join(f"{s}/{n}" for s, n in cells)
                   + f" & {tot_s}/{tot_n} ({pct(tot_s, tot_n)}) \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(a.out, "tab_noise_sensitivity.tex"), "w").write("\n".join(tex) + "\n")

    print("\nNOTE: thresholds differ per run, so solve counts are not comparable across noise levels.")
    print(f"Noise levels present: {noises}. Levels 0.02, 0.05, 0.10, 0.20 are NOT in these files.")
    print(f"Wrote LaTeX to {a.out}/")


if __name__ == "__main__":
    main()

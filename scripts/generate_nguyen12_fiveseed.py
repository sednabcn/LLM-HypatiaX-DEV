#!/usr/bin/env python3
"""
generate_nguyen12_fiveseed.py -- writes nguyen12_fiveseed.tex with two labelled tables so
check_paper_tables.py can verify the paper's five-seed Nguyen-12 numbers:

    tab:nguyen12_fiveseed   recoveries per seed (out of 12) + total (out of 60), H and P
    tab:nguyen12_byeq       recoveries per equation (out of 5 seeds), H and P

Kept out of generate_tables.py on purpose: it is independent of that file's gen_nguyen12(),
so it cannot conflict with whichever version of that function you keep.

Sources (under --results-dir, normally hypatiax/data/results):
    extrapolation/exp3_nguyen12_seed42.json           raw, Shape H (seed 42)
    extrapolation/multi_seed/_merged.json             dict {"N10__seed123": {...}} or a list
        else extrapolation/multi_seed/exp3b_results.json
        else extrapolation/multi_seed/exp3_nguyen12_seed*_nshards*.json (raw, Shape H)

A run counts as a recovery iff R^2 >= 0.9999. -inf, NaN, missing and non-numeric values
(e.g. the string "FAILED") are FAILURES. If any of the five seeds lacks all 12 equations for
a system, nothing is written (the checker then reports NOSOURCE) rather than a partial table.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path

THRESH = 0.9999
SYSTEMS = ("hypatiax", "pysr")
SEEDS = (42, 99, 123, 777, 2024)
N_EQ = 12


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _eq_key(meta: dict):
    for k in ("nguyen_id", "equation_name", "name"):
        v = meta.get(k)
        if v:
            m = re.search(r"(\d+)\s*$", str(v))
            if m:
                return int(m.group(1))
    return None


def _from_raw(d: dict, default_seed=None):
    """Shape H: {"config":{"seed":..}, "results":{"hypatiax":[{metadata, evaluation:{r2}}], "pysr":[...]}}"""
    out = []
    if not (isinstance(d, dict) and isinstance(d.get("results"), dict)):
        return out
    seed = (d.get("config") or {}).get("seed", default_seed)
    for system, recs in d["results"].items():
        if system not in SYSTEMS or not isinstance(recs, list):
            continue
        for it in recs:
            meta = it.get("metadata") if isinstance(it, dict) else None
            if not isinstance(meta, dict):
                continue
            eq = _eq_key(meta)
            if eq is not None:
                out.append((seed, eq, system, (it.get("evaluation") or {}).get("r2")))
    return out


def _from_merged(m):
    rows = m if isinstance(m, list) else [v for k, v in m.items() if not str(k).startswith("_")] \
        if isinstance(m, dict) else []
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        eq = _eq_key({"nguyen_id": r.get("nguyen_id")})
        for system, rec in (r.get("systems") or {}).items():
            if system in SYSTEMS and isinstance(rec, dict) and eq is not None:
                out.append((r.get("seed"), eq, system, rec.get("r2_raw")))
    return out


def collect(results: Path):
    ext = results / "extrapolation"
    ms = ext / "multi_seed"
    rows, used = [], []

    f42 = ext / "exp3_nguyen12_seed42.json"
    if not f42.exists():
        cands = sorted(ext.glob("exp3_nguyen12_seed42*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        f42 = cands[0] if cands else None
    if f42 and f42.exists():
        d = _load(f42)
        if d is not None:
            rows += [r for r in _from_raw(d, 42) if r[0] == 42]
            used.append(f42.name)

    merged = None
    for name in ("_merged.json", "exp3b_results.json"):
        if (ms / name).exists():
            merged = _load(ms / name)
            if merged:
                rows += [r for r in _from_merged(merged) if r[0] != 42]
                used.append(name)
                break
    if not merged:
        for p in sorted(ms.glob("exp3_nguyen12_seed*_nshards*.json")):
            d = _load(p)
            if d is not None:
                rows += [r for r in _from_raw(d) if r[0] != 42]
                used.append(p.name)
    return rows, used


def build(rows):
    cell = {}                        # (seed, eq, system) -> bool (last write wins; same value expected)
    for seed, eq, system, r2 in rows:
        v = _num(r2)
        cell[(seed, eq, system)] = bool(v is not None and math.isfinite(v) and v >= THRESH)
    seeds_present = sorted({k[0] for k in cell})
    return cell, seeds_present


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    a = ap.parse_args()

    rows, used = collect(a.results_dir)
    cell, seeds = build(rows)
    missing = [(s, sy) for s in SEEDS for sy in SYSTEMS
               if sum(1 for e in range(1, N_EQ + 1) if (s, e, sy) in cell) != N_EQ]
    if missing:
        print(f"  SKIP nguyen12_fiveseed.tex: incomplete (seed, system) pairs: {missing[:6]} "
              f"(seeds found: {seeds}; sources: {used})")
        return 0

    per_seed = {(s, sy): sum(cell[(s, e, sy)] for e in range(1, N_EQ + 1)) for s in SEEDS for sy in SYSTEMS}
    tot = {sy: sum(per_seed[(s, sy)] for s in SEEDS) for sy in SYSTEMS}
    n = N_EQ * len(SEEDS)
    per_eq = {(e, sy): sum(cell[(s, e, sy)] for s in SEEDS) for e in range(1, N_EQ + 1) for sy in SYSTEMS}

    L = [f"% Auto-generated by generate_nguyen12_fiveseed.py on {datetime.now():%Y-%m-%d %H:%M}",
         f"% Sources: {', '.join(used)}",
         f"% Recovery = R^2 >= {THRESH}; -inf, NaN, missing and non-numeric count as failures.",
         r"\begin{table}[h]", r"\centering",
         r"\caption{Nguyen-12, recoveries per seed at $\Rsq\ge0.9999$ (out of 12 equations). "
         r"$H$ = HypatiaX, $P$ = PySR-only; $-\infty$, NaN and missing runs count as failures. "
         r"All runs use \texttt{SPARSE\_SEED=off}.}",
         r"\label{tab:nguyen12_fiveseed}", r"\begin{tabular}{lrr}", r"\toprule",
         r"Seed & HypatiaX ($H$) & PySR-only ($P$) \\", r"\midrule"]
    for s in SEEDS:
        L.append(f"{s} & {per_seed[(s, 'hypatiax')]}/{N_EQ} & {per_seed[(s, 'pysr')]}/{N_EQ} \\\\")
    L += [r"\midrule",
          f"Total & {tot['hypatiax']}/{n} ({tot['hypatiax']/n*100:.1f}\\%) & "
          f"{tot['pysr']}/{n} ({tot['pysr']/n*100:.1f}\\%) \\\\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
          r"\begin{table}[h]", r"\centering",
          r"\caption{Nguyen-12, recoveries per equation across the five seeds (42, 99, 123, 777, 2024) "
          r"at $\Rsq\ge0.9999$. $H$ = HypatiaX, $P$ = PySR-only.}",
          r"\label{tab:nguyen12_byeq}", r"\begin{tabular}{lrr}", r"\toprule",
          r"Eq. & HypatiaX ($H$) & PySR-only ($P$) \\", r"\midrule"]
    for e in range(1, N_EQ + 1):
        L.append(f"N-{e} & {per_eq[(e, 'hypatiax')]}/{len(SEEDS)} & {per_eq[(e, 'pysr')]}/{len(SEEDS)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir / "nguyen12_fiveseed.tex").write_text("\n".join(L))
    print(f"  ✅ nguyen12_fiveseed.tex  H {tot['hypatiax']}/{n}  P {tot['pysr']}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

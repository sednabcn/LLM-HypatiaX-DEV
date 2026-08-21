#!/usr/bin/env python3
"""
build_five_system_rows.py
--------------------------
Run this from the repo root (LLM-HypatiaX-REPRO). Reads real exp2 output
and prints (a) diagnostic counts per method so we can sanity-check before
wiring anything into generate_tables.py, and (b) a candidate five_system
JSON block in the exact shape _rows_from_data() in generate_tables.py
expects, ready to drop into a file under one of the _FIVE_SYSTEM_CANDIDATES
search paths (or to hand-verify against FIVE_SYSTEM_PAPER_ROWS).

This does NOT modify generate_tables.py. It's read-only reconnaissance.
"""
import json
import glob
import math
import statistics
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(".")
MULTI_DIR = REPO_ROOT / "hypatiax/data/results/comparison_results/feynman-tests/exp2_multi"
EXTRAP_DIR = REPO_ROOT / "hypatiax/data/results/comparison_results/feynman-tests/exp2_extrap"

# real method key -> paper row name (from the confirmed mapping)
METHOD_TO_ROW = {
    "PureLLM Baseline (core)":              "Pure LLM",
    "ImprovedNN (core)":                    "Neural Network",
    "SymbolicEngineWithLLM (tools)":         "System 2 Symbolic",
    "HybridSystemLLMNN all-domains (core)":  "System 3 LLM+Fallback",
    "HybridDiscoverySystem v50_2 (tools)":   "Hybrid v50_2",
    # EnhancedHybridSystemDeFi (core) intentionally excluded -- DeFi-scoped,
    # not part of the 10-domain exp2 five(six)-system comparison.
}

DESIGN_FOCUS = {
    "Pure LLM": "Recognition",
    "Neural Network": "Baseline",
    "System 2 Symbolic": "Validation",
    "System 3 LLM+Fallback": "Robustness",
    "Hybrid v50_2": "Extrapolation",
}

def finite(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def load_noiseless_train_r2():
    """Aggregate in-distribution r2 per method from protocol_core_noiseless_*.json shards.
    Dedup by (domain, description) -- later timestamp wins, matching the
    checkpoint/resume append pattern used elsewhere in this pipeline."""
    files = sorted(glob.glob(str(MULTI_DIR / "protocol_core_noiseless_*.json")))
    print(f"[train] found {len(files)} protocol_core_noiseless_*.json shard(s)")
    latest = {}  # (domain, description) -> record
    for f in files:
        try:
            d = json.loads(Path(f).read_text())
        except Exception as e:
            print(f"  [skip] {f}: {e}")
            continue
        ts = d.get("timestamp", f)
        for rec in d.get("tests", []):
            key = (rec.get("domain"), rec.get("description"))
            latest[key] = rec  # last file wins; files sorted by name (timestamp-prefixed)

    r2_by_method = defaultdict(list)
    for rec in latest.values():
        for mname, mres in rec.get("results", {}).items():
            if not mres.get("success"):
                continue
            r2 = finite(mres.get("r2"))
            if r2 is not None:
                r2_by_method[mname].append(r2)

    print(f"[train] {len(latest)} unique (domain, test) records after dedup")
    return r2_by_method


def load_extrap():
    """Load benchmark_results_extrap.json (canonical, pre-deduped flat file)."""
    path = EXTRAP_DIR / "benchmark_results_extrap.json"
    if not path.exists():
        print(f"[extrap] MISSING: {path}")
        return {}
    d = json.loads(path.read_text())
    print(f"[extrap] {len(d)} flat records in {path.name}")

    by_method = defaultdict(list)
    for rec in d:
        by_method[rec["method"]].append(rec)
    return by_method


def robust_stats(values):
    """values: list of floats (already filtered to finite or None-excluded upstream
    for the 'attempted' count, but we also want to see raw non-finite counts)."""
    finite_vals = [v for v in values if v is not None]
    return {
        "n": len(finite_vals),
        "median": statistics.median(finite_vals) if finite_vals else None,
        "mean": statistics.mean(finite_vals) if finite_vals else None,
        "std": statistics.stdev(finite_vals) if len(finite_vals) >= 2 else None,
    }


def main():
    train_r2 = load_noiseless_train_r2()
    extrap = load_extrap()

    print("\n" + "=" * 100)
    print(f"{'method':45s} {'train_n':>8} {'train_r2_mean':>14} {'train_r2_std':>13} "
          f"{'extrap_n':>9} {'extrap_n_nonfinite':>19} {'err_median%':>12} {'err_mean%':>12}")
    print("-" * 100)

    candidate_rows = []
    for mname, row_name in METHOD_TO_ROW.items():
        tr = train_r2.get(mname, [])
        tr_stats = robust_stats(tr)

        ex_records = extrap.get(mname, [])
        # Records where extrap was even attempted (far region existed):
        # extrap_r2_far present in the record dict (even if -inf/None value).
        attempted = [r for r in ex_records if "extrap_r2_far" in r]
        # FIX: distinguish "never computed" (None) from "computed but non-finite"
        # (e.g. -inf) from "computed and finite". Previous version only caught
        # the middle case, so NN/System-3 (which are always None, per
        # compute_extrap_r2_far()'s NN-tag/"N/A" skip) showed extrap_n=30
        # instead of 0 -- misleading, since None means "never evaluated" not
        # "evaluated with a real number".
        raw_err = [r.get("extrap_error_pct") for r in attempted]
        n_never_computed = sum(1 for v in raw_err if v is None)
        n_nonfinite = sum(1 for v in raw_err if v is not None and finite(v) is None)
        err_pcts_finite = [finite(v) for v in raw_err]
        err_stats = robust_stats(err_pcts_finite)

        # Robust mean via IQR-based clipping (matches _robust_stats convention
        # used elsewhere in this repo for extrapolation-error aggregates,
        # since raw means are dominated by catastrophic outliers -- see
        # Pure LLM's 4.6e9% raw mean below).
        finite_vals = sorted(v for v in err_pcts_finite if v is not None)
        clipped_mean = None
        if finite_vals:
            q1 = finite_vals[len(finite_vals) // 4]
            q3 = finite_vals[(3 * len(finite_vals)) // 4]
            iqr = q3 - q1
            hi = q3 + 1.5 * iqr
            clipped = [v for v in finite_vals if v <= hi] if iqr > 0 else finite_vals
            if clipped:
                clipped_mean = sum(clipped) / len(clipped)

        print(f"{mname:45s} {tr_stats['n']:8d} "
              f"{('%.4f' % tr_stats['mean']) if tr_stats['mean'] is not None else '---':>14} "
              f"{('%.4f' % tr_stats['std']) if tr_stats['std'] is not None else '---':>13} "
              f"{len(attempted):9d} "
              f"{f'{n_never_computed}none/{n_nonfinite}inf':>19} "
              f"{('%.1f' % err_stats['median']) if err_stats['median'] is not None else '---':>12} "
              f"{('%.1f' % clipped_mean) if clipped_mean is not None else '---':>12}")

        candidate_rows.append({
            "name": row_name,
            "n": err_stats["n"],  # n = extrapolation records with a finite error pct
            "extrap_median_pct": (f"{err_stats['median']:.1f}"
                                   if err_stats["median"] is not None else None),
            "extrap_mean_pct": (f"{err_stats['mean']:.1f}"
                                 if err_stats["mean"] is not None else None),
            "train_r2_mean": (f"{tr_stats['mean']:.3f}"
                               if tr_stats["mean"] is not None else None),
            "std": (f"{err_stats['std']:.1f}" if err_stats["std"] is not None else None),
            "design_focus": DESIGN_FOCUS[row_name],
        })

    print("=" * 100)
    print("\nCandidate five_system JSON block (matches _rows_from_data() schema):\n")
    print(json.dumps({"five_system": candidate_rows}, indent=2))

    print("\nNOTE: 'n' above = count of records with a finite extrap_error_pct for that "
          "method (i.e. extrapolation was both attempted AND numerically evaluable). "
          "Per compute_extrap_r2_far()'s own docstring, methods returning NN architecture "
          "tags or 'N/A' formulas get null extrap values -- so a low/zero n for Neural "
          "Network and/or System 3 is EXPECTED real behavior, not a bug in this script. "
          "Compare these n values against FIVE_SYSTEM_PAPER_ROWS before treating this as "
          "a drop-in replacement -- if paper rows show real n>0 extrap data for methods "
          "this script shows n=0 for, that's a substantive discrepancy to resolve, not "
          "something to silently overwrite.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
merge_extrap_into_benchmark.py
-------------------------------
Merges extrapolation R² values (extrap_r2_far, extrap_r2_near) into the
flat benchmark_results.json produced by run_comparative_suite_benchmark_v2.py,
producing ablation_paired.json in the schema that run_analysis.py analyse_ablation()
expects:

    [
      {
        "equation_name":  str,
        "equation_id":    str,
        "domain":         str,
        "hypatia":   { "train_r2": float, "extrap_r2_near": float, "extrap_r2_far": float },
        "pysr_only": { "extrap_r2_far": float, "extrap_r2_near": float },
        "complexity": { "hypatia": int|null, "pysr_only": int|null },
        "scale_log":  float|null,
      },
      ...
    ]

Called by ci_runner.yml "Run extrapolation evaluation (exp2_feynman only)" step.

Usage:
    python3 merge_extrap_into_benchmark.py \\
        --benchmark-dir  <dir containing benchmark_results.json> \\
        --extrap-dir     <dir containing extrap_results_*.json>  \\
        --output         <path to write ablation_paired.json>
"""

import argparse
import json
import math
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Method name normalisation
# ---------------------------------------------------------------------------
# Map from the verbose method strings in benchmark_results.json to canonical
# short names used in the ablation schema.  Any method containing "PySR" or
# "Symbolic" (without LLM/Hybrid qualifier) is treated as pysr_only.
# The HypatiaX method is whichever non-PySR method achieves the best r2 on
# a per-equation basis (or the first method not classified as pysr_only).

def _classify_method(method_str: str) -> str:
    """Return 'hypatia' | 'pysr_only' | 'other'."""
    m = method_str.lower()
    # Pure symbolic / PySR-only (no LLM/hybrid component)
    if "pysr" in m or ("symbolic" in m and "llm" not in m and "hybrid" not in m):
        return "pysr_only"
    # Any LLM, hybrid, NN, or HypatiaX variant
    if any(k in m for k in ("llm", "hybrid", "neural", "nn", "hypatia",
                             "improved", "enhanced", "discovery")):
        return "hypatia"
    return "other"


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def _load_benchmark(bench_dir: Path) -> list[dict]:
    """Load benchmark_results.json from bench_dir (Shape C — flat list)."""
    path = bench_dir / "benchmark_results.json"
    if not path.exists():
        print(f"::error::benchmark_results.json not found in {bench_dir}", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        print(f"::error::benchmark_results.json is not a list (got {type(data).__name__})",
              file=sys.stderr)
        sys.exit(1)
    print(f"  Loaded {len(data)} benchmark records from {path}")
    return data


def _load_extrap_results(extrap_dir: Path) -> dict[str, dict]:
    """
    Load extrap_results_*.json files from extrap_dir.

    Expected shape (one file per domain):
        {
          "domain": "feynman_biology",
          "equations": {
            "Michaelis-Menten enzyme kinetics": {
              "hypatia":   { "extrap_r2_far": 0.91, "extrap_r2_near": 0.99 },
              "pysr_only": { "extrap_r2_far": 0.61, "extrap_r2_near": 0.87 },
            },
            ...
          }
        }

    Returns: { equation_name: { "hypatia": {...}, "pysr_only": {...} } }
    """
    results: dict[str, dict] = {}
    files = sorted(extrap_dir.glob("extrap_results_*.json"))
    if not files:
        print(f"  No extrap_results_*.json found in {extrap_dir} — "
              f"extrap_r2_far will be None for all equations.", file=sys.stderr)
        return results

    for fp in files:
        try:
            with open(fp) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ::warning::Could not read {fp.name}: {e}", file=sys.stderr)
            continue
        equations = data.get("equations", {})
        for eq_name, eq_data in equations.items():
            if isinstance(eq_data, dict):
                results[eq_name] = eq_data
        print(f"  Loaded {len(equations)} extrap equations from {fp.name}")

    print(f"  Total extrap equations loaded: {len(results)}")
    return results


# ---------------------------------------------------------------------------
# Main merge logic
# ---------------------------------------------------------------------------

def merge(benchmark_records: list[dict],
          extrap_results: dict[str, dict]) -> list[dict]:
    """
    Group benchmark_records by (test, domain), classify methods, and
    produce one paired record per equation in the ablation schema.
    """
    # Group by equation name
    from collections import defaultdict
    by_eq: dict[str, dict] = defaultdict(lambda: {
        "equation_name": None,
        "domain": None,
        "hypatia_r2": [],
        "pysr_r2": [],
        "hypatia_success": [],
        "pysr_success": [],
    })

    for rec in benchmark_records:
        eq = rec.get("test", rec.get("equation_name", rec.get("equation_id", "?")))
        mtype = _classify_method(rec.get("method", ""))
        domain = rec.get("domain", "?")

        g = by_eq[eq]
        g["equation_name"] = eq
        g["domain"] = domain

        r2 = rec.get("r2")
        success = rec.get("success", False)

        if mtype == "hypatia":
            if r2 is not None:
                g["hypatia_r2"].append(float(r2))
            g["hypatia_success"].append(success)
        elif mtype == "pysr_only":
            if r2 is not None:
                g["pysr_r2"].append(float(r2))
            g["pysr_success"].append(success)

    paired: list[dict] = []
    n_with_extrap = 0
    n_missing_extrap = 0

    for eq_name, g in by_eq.items():
        # Best train R² for hypatia (max across methods classified as hypatia)
        h_train_r2 = max(g["hypatia_r2"]) if g["hypatia_r2"] else None

        # Extrap values from the extrap step
        extrap = extrap_results.get(eq_name, {})
        h_extrap = extrap.get("hypatia", {}) or {}
        p_extrap = extrap.get("pysr_only", {}) or {}

        h_far  = h_extrap.get("extrap_r2_far")
        h_near = h_extrap.get("extrap_r2_near")
        p_far  = p_extrap.get("extrap_r2_far")
        p_near = p_extrap.get("extrap_r2_near")

        if h_far is not None or p_far is not None:
            n_with_extrap += 1
        else:
            n_missing_extrap += 1

        paired.append({
            "equation_name": eq_name,
            "equation_id":   eq_name,
            "domain":        g["domain"],
            "hypatia": {
                "train_r2":       h_train_r2,
                "extrap_r2_near": h_near,
                "extrap_r2_far":  h_far,
                "success":        any(g["hypatia_success"]),
            },
            "pysr_only": {
                "train_r2":       max(g["pysr_r2"]) if g["pysr_r2"] else None,
                "extrap_r2_near": p_near,
                "extrap_r2_far":  p_far,
                "success":        any(g["pysr_success"]),
            },
        })

    print(f"  Paired records: {len(paired)}")
    print(f"  With extrap_r2_far: {n_with_extrap}")
    print(f"  Missing extrap_r2_far: {n_missing_extrap}"
          + (" ← run the extrap step to populate these" if n_missing_extrap else ""))
    return paired


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge extrap_r2_far into benchmark_results → ablation_paired.json"
    )
    ap.add_argument("--benchmark-dir", required=True,
                    help="Directory containing benchmark_results.json")
    ap.add_argument("--extrap-dir", required=True,
                    help="Directory containing extrap_results_*.json files")
    ap.add_argument("--output", required=True,
                    help="Output path for ablation_paired.json")
    args = ap.parse_args()

    bench_dir  = Path(args.benchmark_dir)
    extrap_dir = Path(args.extrap_dir)
    out_path   = Path(args.output)

    print(f"merge_extrap_into_benchmark.py")
    print(f"  benchmark-dir : {bench_dir}")
    print(f"  extrap-dir    : {extrap_dir}")
    print(f"  output        : {out_path}")

    benchmark_records = _load_benchmark(bench_dir)
    extrap_results    = _load_extrap_results(extrap_dir)
    paired            = merge(benchmark_records, extrap_results)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(paired, f, indent=2)

    print(f"  Written {len(paired)} paired records → {out_path}")

    # Warn if no extrap data at all — analysis will fail with TOO_FEW_MW_PAIRS
    n_with_far = sum(
        1 for r in paired
        if r.get("hypatia", {}).get("extrap_r2_far") is not None
    )
    if n_with_far == 0:
        print("::warning::ablation_paired.json has 0 equations with hypatia.extrap_r2_far. "
              "run_analysis.py will emit TOO_FEW_MW_PAIRS. "
              "Ensure run_all.sh --step exp2_feynman_extrap ran successfully.", file=sys.stderr)
    elif n_with_far < 3:
        print(f"::warning::Only {n_with_far} equation(s) have extrap_r2_far. "
              f"Mann-Whitney test needs ≥ 3 pairs.", file=sys.stderr)
    else:
        print(f"  OK: {n_with_far} equations have extrap_r2_far — "
              f"sufficient for Mann-Whitney test.")


if __name__ == "__main__":
    main()

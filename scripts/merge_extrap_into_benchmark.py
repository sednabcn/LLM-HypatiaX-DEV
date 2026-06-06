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

PRIMARY data source (preferred):
    benchmark_results_extrap.json  — written by run_comparative_suite_benchmark_v2.py
    when --extrap is active.  Contains both train r2 AND extrap_r2_far in one file,
    keyed by (test, method).  Pass its directory via --extrap-benchmark-dir.

LEGACY fallback (backward compat):
    extrap_results_*.json  — one file per domain in the schema consumed by the
    old merge script.  Used when benchmark_results_extrap.json is absent.
    Pass its directory via --extrap-dir (unchanged from the original script).

Usage:
    # Preferred — single source of truth from v2.2+ runner:
    python3 merge_extrap_into_benchmark.py \\
        --benchmark-dir       <dir containing benchmark_results.json> \\
        --extrap-benchmark-dir <dir containing benchmark_results_extrap.json> \\
        --output              <path to write ablation_paired.json>

    # Legacy fallback:
    python3 merge_extrap_into_benchmark.py \\
        --benchmark-dir  <dir containing benchmark_results.json> \\
        --extrap-dir     <dir containing extrap_results_*.json>  \\
        --output         <path to write ablation_paired.json>

    # Both supplied — benchmark_results_extrap.json wins per-equation,
    # extrap_results_*.json fills any gaps:
    python3 merge_extrap_into_benchmark.py \\
        --benchmark-dir        <dir> \\
        --extrap-benchmark-dir <dir> \\
        --extrap-dir           <dir> \\
        --output               <path>
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Method name normalisation
# ---------------------------------------------------------------------------

def _classify_method(method_str: str) -> str:
    """Return 'hypatia' | 'pysr_only' | 'other'.

    Role mapping against the actual method names produced by
    run_comparative_suite_benchmark_v2.py:

      pysr_only  — HybridDiscoverySystem v50_2 (tools)
                   Pure PySR run; SymbolicEngineWithLLM is called with
                   llm_mode="none" inside the v50_2 subprocess path, so no
                   LLM guidance is applied.  This is the PySR-baseline arm.

      hypatia    — Everything else that uses LLM, hybrid, or neural components:
                   SymbolicEngineWithLLM (tools), PureLLM Baseline,
                   ImprovedNN, EnhancedHybridSystemDeFi,
                   HybridSystemLLMNN all-domains, etc.

    The old classifier required "pysr" or bare "symbolic" (without "llm"/"hybrid")
    in the name — conditions never satisfied by any current method — so pysr_only
    was always empty.  We now key on "v50_2" as the unambiguous PySR-baseline
    marker, with "pysr" retained for forward-compat with any future method names.
    """
    m = method_str.lower()

    # PySR-only baseline: v50_2 runs PySR with llm_mode="none" (no LLM loop).
    # "pysr" kept for forward-compat with future method names.
    if "v50_2" in m or "pysr" in m:
        return "pysr_only"

    # Hypatia: any method with LLM, hybrid, neural, or discovery components.
    if any(k in m for k in ("llm", "hybrid", "neural", "nn", "hypatia",
                             "improved", "enhanced", "discovery", "symbolic")):
        return "hypatia"

    return "other"


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def _load_benchmark(bench_dir: Path) -> list:
    """Load benchmark_results.json (flat list, train r2 / metadata)."""
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


def _load_extrap_benchmark(extrap_bench_dir: Path) -> dict:
    """
    Load benchmark_results_extrap.json written by run_comparative_suite_benchmark_v2.py
    when --extrap is active.

    Returns a dict keyed by equation name:
        {
          eq_name: {
            "hypatia":   { "extrap_r2_far": float|None, "extrap_rmse_far": float|None,
                           "train_r2": float|None, "success": bool },
            "pysr_only": { ... },
          }
        }

    When multiple methods map to the same role (e.g. several hypatia methods),
    the one with the highest extrap_r2_far is kept (same tie-break as train_r2).
    """
    path = extrap_bench_dir / "benchmark_results_extrap.json"
    if not path.exists():
        print(f"  benchmark_results_extrap.json not found in {extrap_bench_dir} "
              f"— will fall back to extrap_results_*.json if supplied.", file=sys.stderr)
        return {}

    with open(path) as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        print(f"  ::warning::benchmark_results_extrap.json is not a list — skipping.",
              file=sys.stderr)
        return {}

    # Group by equation, then by role, keeping best extrap_r2_far per role.
    results: dict = defaultdict(lambda: {"hypatia": None, "pysr_only": None})

    def _better(a: dict | None, b: dict) -> dict:
        """Return whichever candidate has the higher finite extrap_r2_far.
        Non-finite values (-inf, nan) are treated the same as None so that a
        method which failed extrapolation never blocks a finite result."""
        import math as _math

        def _fin(v):
            return v if (v is not None and isinstance(v, float) and _math.isfinite(v)) else None

        if a is None:
            return b
        a_far = _fin(a.get("extrap_r2_far"))
        b_far = _fin(b.get("extrap_r2_far"))
        if a_far is None and b_far is None:
            # Fall back to train r2
            return a if (a.get("train_r2") or 0) >= (b.get("train_r2") or 0) else b
        if a_far is None:
            return b
        if b_far is None:
            return a
        return a if a_far >= b_far else b

    for row in rows:
        eq   = row.get("test", row.get("equation_name", "?"))
        role = _classify_method(row.get("method", ""))
        if role == "other":
            continue

        candidate = {
            "extrap_r2_far":     row.get("extrap_r2_far"),
            "extrap_rmse_far":   row.get("extrap_rmse_far"),
            # extrap_r2_near is not computed by the benchmark runner — leave None
            "extrap_r2_near":    row.get("extrap_r2_near"),
            "train_r2":          row.get("r2"),
            "success":           row.get("success", False),
            "extrap_train_frac": row.get("extrap_train_frac"),
            "extrap_n_train":    row.get("extrap_n_train"),
            "extrap_n_test":     row.get("extrap_n_test"),
        }
        results[eq][role] = _better(results[eq][role], candidate)

    print(f"  Loaded extrap data for {len(results)} equations "
          f"from {path}  ({len(rows)} rows)")
    return dict(results)


def _load_extrap_results_legacy(extrap_dir: Path) -> dict:
    """
    Legacy loader: reads extrap_results_*.json files (one per domain).

    Expected shape:
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
    results: dict = {}
    files = sorted(extrap_dir.glob("extrap_results_*.json"))
    if not files:
        print(f"  No extrap_results_*.json found in {extrap_dir}.", file=sys.stderr)
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

    print(f"  Total extrap equations loaded (legacy): {len(results)}")
    return results


# ---------------------------------------------------------------------------
# Main merge logic
# ---------------------------------------------------------------------------

def merge(
    benchmark_records: list,
    extrap_new: dict,          # from benchmark_results_extrap.json  (preferred)
    extrap_legacy: dict,       # from extrap_results_*.json           (fallback)
) -> list:
    """
    Group benchmark_records by equation name, classify methods, and produce
    one paired record per equation in the ablation schema.

    extrap_new takes priority per-equation; extrap_legacy fills any gaps.
    """
    # ── Group train metrics by equation ─────────────────────────────────────
    by_eq: dict = defaultdict(lambda: {
        "equation_name":  None,
        "domain":         None,
        "hypatia_r2":     [],
        "pysr_r2":        [],
        "hypatia_success":[],
        "pysr_success":   [],
    })

    for rec in benchmark_records:
        eq     = rec.get("test", rec.get("equation_name", rec.get("equation_id", "?")))
        mtype  = _classify_method(rec.get("method", ""))
        domain = rec.get("domain", "?")

        g = by_eq[eq]
        g["equation_name"] = eq
        g["domain"]        = domain

        r2      = rec.get("r2")
        success = rec.get("success", False)

        if mtype == "hypatia":
            if r2 is not None:
                g["hypatia_r2"].append(float(r2))
            g["hypatia_success"].append(success)
        elif mtype == "pysr_only":
            if r2 is not None:
                g["pysr_r2"].append(float(r2))
            g["pysr_success"].append(success)

    # ── Build paired records ──────────────────────────────────────────────
    paired: list = []
    n_with_extrap   = 0
    n_missing_extrap = 0

    for eq_name, g in by_eq.items():
        h_train_r2 = max(g["hypatia_r2"]) if g["hypatia_r2"] else None

        # --- Resolve extrap values: new file first, legacy as fallback ------
        # New source carries train_r2 too; we keep benchmark_results.json's
        # train_r2 (max across all hypatia methods) as the canonical value.
        new_eq    = extrap_new.get(eq_name, {})
        legacy_eq = extrap_legacy.get(eq_name, {})

        def _pick(new_role: dict | None, legacy_role: dict | None, key: str):
            """Return value from new_role if finite, else try legacy_role.
            Non-finite floats (-inf, nan) are treated as absent so they never
            shadow a finite value from the legacy source."""
            import math as _math

            def _fin(v):
                if v is None:
                    return None
                if isinstance(v, float) and not _math.isfinite(v):
                    return None
                return v

            v = _fin((new_role or {}).get(key))
            if v is not None:
                return v
            return _fin((legacy_role or {}).get(key))

        h_new    = new_eq.get("hypatia")    or {}
        p_new    = new_eq.get("pysr_only")  or {}
        h_leg    = (legacy_eq.get("hypatia")   or {}) if isinstance(legacy_eq, dict) else {}
        p_leg    = (legacy_eq.get("pysr_only") or {}) if isinstance(legacy_eq, dict) else {}

        h_far    = _pick(h_new, h_leg, "extrap_r2_far")
        h_near   = _pick(h_new, h_leg, "extrap_r2_near")
        p_far    = _pick(p_new, p_leg, "extrap_r2_far")
        p_near   = _pick(p_new, p_leg, "extrap_r2_near")

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

    print(f"  Paired records   : {len(paired)}")
    print(f"  With extrap_r2_far    : {n_with_extrap}")
    print(f"  Missing extrap_r2_far : {n_missing_extrap}"
          + (" ← run with --extrap to populate" if n_missing_extrap else ""))
    return paired


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge extrap_r2_far into benchmark_results → ablation_paired.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Preferred (v2.2+ runner):
  python3 merge_extrap_into_benchmark.py \\
      --benchmark-dir        results/ \\
      --extrap-benchmark-dir results/ \\
      --output               ablation_paired.json

  # Legacy fallback:
  python3 merge_extrap_into_benchmark.py \\
      --benchmark-dir  results/ \\
      --extrap-dir     results/extrap/ \\
      --output         ablation_paired.json
        """,
    )
    ap.add_argument(
        "--benchmark-dir", required=True,
        help="Directory containing benchmark_results.json (train r2 / method metadata)",
    )
    ap.add_argument(
        "--extrap-benchmark-dir", default=None, dest="extrap_benchmark_dir",
        help=(
            "Directory containing benchmark_results_extrap.json written by the "
            "v2.2+ runner when --extrap is active.  Primary extrap source. "
            "Falls back to --extrap-dir when absent."
        ),
    )
    ap.add_argument(
        "--extrap-dir", default=None, dest="extrap_dir",
        help=(
            "Directory containing extrap_results_*.json files (legacy format). "
            "Used when benchmark_results_extrap.json is absent or as gap-filler."
        ),
    )
    ap.add_argument(
        "--output", required=True,
        help="Output path for ablation_paired.json",
    )
    args = ap.parse_args()

    if args.extrap_benchmark_dir is None and args.extrap_dir is None:
        ap.error("Supply at least one of --extrap-benchmark-dir or --extrap-dir")

    bench_dir = Path(args.benchmark_dir)
    out_path  = Path(args.output)

    print("merge_extrap_into_benchmark.py")
    print(f"  benchmark-dir        : {bench_dir}")
    if args.extrap_benchmark_dir:
        print(f"  extrap-benchmark-dir : {args.extrap_benchmark_dir}")
    if args.extrap_dir:
        print(f"  extrap-dir (legacy)  : {args.extrap_dir}")
    print(f"  output               : {out_path}")

    benchmark_records = _load_benchmark(bench_dir)

    extrap_new: dict = {}
    if args.extrap_benchmark_dir:
        extrap_new = _load_extrap_benchmark(Path(args.extrap_benchmark_dir))

    extrap_legacy: dict = {}
    if args.extrap_dir:
        extrap_legacy = _load_extrap_results_legacy(Path(args.extrap_dir))

    paired = merge(benchmark_records, extrap_new, extrap_legacy)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(paired, f, indent=2)
    print(f"  Written {len(paired)} paired records → {out_path}")

    # Warn if Mann-Whitney will fail
    n_with_far = sum(
        1 for r in paired
        if r.get("hypatia", {}).get("extrap_r2_far") is not None
    )
    if n_with_far == 0:
        print(
            "::warning::ablation_paired.json has 0 equations with hypatia.extrap_r2_far. "
            "run_analysis.py will emit TOO_FEW_MW_PAIRS. "
            "Ensure run_all.sh --step exp2_feynman_extrap ran successfully and "
            "benchmark_results_extrap.json was produced.",
            file=sys.stderr,
        )
    elif n_with_far < 3:
        print(
            f"::warning::Only {n_with_far} equation(s) have extrap_r2_far. "
            f"Mann-Whitney test needs ≥ 3 pairs.",
            file=sys.stderr,
        )
    else:
        print(f"  OK: {n_with_far} equations have extrap_r2_far — "
              f"sufficient for Mann-Whitney test.")


if __name__ == "__main__":
    main()

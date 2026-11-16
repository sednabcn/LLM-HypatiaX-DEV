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
        "hypatia":   { "train_r2": float, "extrap_r2_near": float|null, "extrap_r2_far": float|null,
                       "train_success": bool, "extrap_success": bool },
        "pysr_only": { "train_r2": float, "extrap_r2_near": float|null, "extrap_r2_far": float|null,
                       "train_success": bool, "extrap_success": bool },
        "complexity": { "hypatia": int|null, "pysr_only": int|null },
        "scale_log":  float|null,
      },
      ...
    ]

NOTE on extrap_r2_near: the current benchmark runner (run_comparative_suite_
benchmark_v2.py) does not compute a near-domain extrapolation split -- it only
writes extrap_r2_far. This field will therefore be null for every record until
that runner is extended to produce it. It is carried through here (not
dropped) so the schema is forward-compatible and so run_analysis.py's
instability-index table can distinguish "not measured" (null) from a real 0.0.

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

    # Both supplied — benchmark_results_extrap.json provides method-level
    # pairs. Legacy extrap_results_*.json is retained separately for audit
    # and is never used to fill a named method's missing score:
    python3 merge_extrap_into_benchmark.py \\
        --benchmark-dir        <dir> \\
        --extrap-benchmark-dir <dir> \\
        --extrap-dir           <dir> \\
        --output               <path>
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from merge_shards import EXPECTED_SWEEP_METHODS as EXPECTED_METHOD_NAMES

# Must match EXTRAP_SUCCESS_THRESHOLD in run_analysis.py (paper §10.7: "9/30
# successes"). Duplicated here (not imported) because this script runs in a
# separate CI step/venv from run_analysis.py. If you change one, change both.
EXTRAP_SUCCESS_THRESHOLD = 0.99
EXPECTED_EQUATION_COUNT = 30


def validate_method_coverage(benchmark_records: list, extrap_by_equation: dict) -> None:
    """Require the full 30-equation by six-method Feynman comparison matrix."""
    benchmark_index: dict = defaultdict(set)
    for row in benchmark_records:
        equation = row.get("test", row.get("equation_name", row.get("equation_id")))
        method = row.get("method")
        if equation and method:
            missing_fields = {"r2", "success"} - set(row)
            if missing_fields or not isinstance(row.get("success"), bool):
                raise ValueError(
                    "benchmark_results.json has incomplete scoring fields for "
                    f"equation={equation!r}, method={method!r}: "
                    f"missing={sorted(missing_fields)}"
                )
            benchmark_index[equation].add(method)

    benchmark_equations = set(benchmark_index)
    extrap_equations = set(extrap_by_equation)
    if len(benchmark_equations) != EXPECTED_EQUATION_COUNT:
        raise ValueError(
            f"benchmark_results.json has {len(benchmark_equations)}/"
            f"{EXPECTED_EQUATION_COUNT} equations"
        )
    if benchmark_equations != extrap_equations:
        raise ValueError(
            "benchmark/extrap equation sets differ; "
            f"missing_extrap={sorted(benchmark_equations - extrap_equations)}, "
            f"missing_benchmark={sorted(extrap_equations - benchmark_equations)}"
        )

    expected = set(EXPECTED_METHOD_NAMES)
    for equation in sorted(benchmark_equations):
        benchmark_methods = benchmark_index[equation]
        extrap_methods = set(extrap_by_equation[equation])
        if benchmark_methods != expected or extrap_methods != expected:
            raise ValueError(
                f"method coverage incomplete for {equation!r}; "
                f"benchmark_missing={sorted(expected - benchmark_methods)}, "
                f"benchmark_unexpected={sorted(benchmark_methods - expected)}, "
                f"extrap_missing={sorted(expected - extrap_methods)}, "
                f"extrap_unexpected={sorted(extrap_methods - expected)}"
            )


# ---------------------------------------------------------------------------
# Method name normalisation
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
    """Load extrapolation rows keyed by (equation, exact method name).

    No best-of-method selection occurs here. Every method remains a separate
    result and duplicate (equation, method) rows must agree exactly.
    """
    path = extrap_bench_dir / "benchmark_results_extrap.json"
    if not path.exists():
        print(
            f"  benchmark_results_extrap.json not found in {extrap_bench_dir} "
            "— legacy role-only records will be retained separately, not paired to methods.",
            file=sys.stderr,
        )
        return {}

    with open(path, encoding="utf-8") as stream:
        rows = json.load(stream)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON list")

    results: dict = defaultdict(dict)
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: row {row_number} is not an object")
        equation = row.get("test", row.get("equation_name", row.get("equation_id")))
        method = row.get("method")
        if not equation or not method:
            raise ValueError(
                f"{path}: row {row_number} lacks an equation identity or method"
            )
        missing_fields = {"r2", "extrap_r2_far", "success"} - set(row)
        if missing_fields or not isinstance(row.get("success"), bool):
            raise ValueError(
                f"{path}: row {row_number} has incomplete scoring fields; "
                f"missing={sorted(missing_fields)}, success_type={type(row.get('success')).__name__}"
            )
        candidate = {
            "method": method,
            "extrap_r2_far": row.get("extrap_r2_far"),
            "extrap_rmse_far": row.get("extrap_rmse_far"),
            "extrap_r2_near": row.get("extrap_r2_near"),
            "extrap_rmse_near": row.get("extrap_rmse_near"),
            "train_r2": row.get("r2"),
            "train_rmse": row.get("rmse"),
            "success": row.get("success", False),
            "extrap_train_frac": row.get("extrap_train_frac"),
            "extrap_n_train": row.get("extrap_n_train"),
            "extrap_n_test": row.get("extrap_n_test"),
            "formula": row.get("formula"),
            "seed": row.get("seed", row.get("benchmark_seed")),
            "protocol": row.get("protocol", row.get("protocol_id")),
            "config_hash": row.get("config_hash"),
            "split_hash": row.get("split_hash"),
        }
        previous = results[equation].get(method)
        if previous is not None:
            if json.dumps(previous, sort_keys=True, allow_nan=True) != json.dumps(
                candidate, sort_keys=True, allow_nan=True
            ):
                raise ValueError(
                    f"{path}: conflicting duplicate for equation={equation!r}, "
                    f"method={method!r}; refusing retrospective selection"
                )
            continue
        results[equation][method] = candidate

    print(
        f"  Loaded {sum(len(v) for v in results.values())} method/equation "
        f"rows across {len(results)} equations from {path}"
    )
    return {equation: dict(methods) for equation, methods in results.items()}

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
            with open(fp, encoding="utf-8") as stream:
                data = json.load(stream)
        except Exception as e:
            raise ValueError(f"Cannot read legacy extrapolation input {fp}: {e}") from e
        equations = data.get("equations", {})
        if not isinstance(equations, dict):
            raise ValueError(f"{fp} has no equations object")
        for eq_name, eq_data in equations.items():
            if not isinstance(eq_data, dict):
                continue
            previous = results.get(eq_name)
            if previous is not None and json.dumps(
                previous, sort_keys=True, allow_nan=True
            ) != json.dumps(eq_data, sort_keys=True, allow_nan=True):
                raise ValueError(
                    f"Conflicting legacy extrapolation rows for equation={eq_name!r}; "
                    f"at least one source is {fp.name}"
                )
            results[eq_name] = eq_data
        print(f"  Loaded {len(equations)} extrap equations from {fp.name}")

    print(f"  Total extrap equations loaded (legacy): {len(results)}")
    return results


# ---------------------------------------------------------------------------
# Main merge logic
# ---------------------------------------------------------------------------

PRIMARY_HYPATIA_METHOD = "SymbolicEngineWithLLM (tools)"
PRIMARY_PYSR_METHOD = "HybridDiscoverySystem v50_2 (tools)"


def _finite(value):
    import math
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)


def _method_result(benchmark_row: dict | None, extrap_row: dict | None) -> dict:
    """Pair train and extrapolation metrics from the same named method only."""
    benchmark_row = benchmark_row or {}
    extrap_row = extrap_row or {}
    far = extrap_row.get("extrap_r2_far")
    has_extrap_row = bool(extrap_row)
    train_r2 = extrap_row.get("train_r2") if has_extrap_row else benchmark_row.get("r2")
    train_rmse = extrap_row.get("train_rmse") if has_extrap_row else benchmark_row.get("rmse")
    train_success = extrap_row.get("success", False) if has_extrap_row else benchmark_row.get("success", False)
    return {
        "method": benchmark_row.get("method") or extrap_row.get("method"),
        "train_r2": train_r2,
        "train_rmse": train_rmse,
        "train_success": bool(train_success),
        "benchmark_reference_train_r2": benchmark_row.get("r2"),
        "benchmark_reference_success": benchmark_row.get("success"),
        "extrap_r2_near": extrap_row.get("extrap_r2_near"),
        "extrap_rmse_near": extrap_row.get("extrap_rmse_near"),
        "extrap_r2_far": far,
        "extrap_rmse_far": extrap_row.get("extrap_rmse_far"),
        "extrap_success": bool(_finite(far) and far >= EXTRAP_SUCCESS_THRESHOLD),
        "formula": benchmark_row.get("formula") or extrap_row.get("formula"),
        "extrap_train_frac": extrap_row.get("extrap_train_frac"),
        "extrap_n_train": extrap_row.get("extrap_n_train"),
        "extrap_n_test": extrap_row.get("extrap_n_test"),
        "seed": extrap_row.get("seed", benchmark_row.get("seed")),
        "protocol": extrap_row.get("protocol", benchmark_row.get("protocol")),
        "config_hash": extrap_row.get("config_hash", benchmark_row.get("config_hash")),
        "split_hash": extrap_row.get("split_hash", benchmark_row.get("split_hash")),
        "train_source": (
            "benchmark_results_extrap.json" if has_extrap_row
            else "benchmark_results.json" if benchmark_row else None
        ),
        "extrap_source": "benchmark_results_extrap.json" if has_extrap_row else None,
    }


def merge(
    benchmark_records: list,
    extrap_new: dict,
    extrap_legacy: dict,
) -> list:
    """Pair every method by exact equation and method identity.

    The compatibility hypatia/pysr_only fields are populated only from the
    explicitly named primary methods below. All other methods remain available
    under method_results; no score-based winner is selected after evaluation.
    Legacy role-only extrap data is retained separately and is not mixed with
    a train score from a different method.
    """
    benchmark_by_equation: dict = defaultdict(dict)
    domains: dict = {}

    for row_number, record in enumerate(benchmark_records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"benchmark_results.json row {row_number} is not an object")
        equation = record.get(
            "test", record.get("equation_name", record.get("equation_id"))
        )
        method = record.get("method")
        if not equation or not method:
            raise ValueError(
                f"benchmark_results.json row {row_number} lacks equation or method"
            )
        previous = benchmark_by_equation[equation].get(method)
        if previous is not None:
            if json.dumps(previous, sort_keys=True, allow_nan=True) != json.dumps(
                record, sort_keys=True, allow_nan=True
            ):
                raise ValueError(
                    f"benchmark_results.json has conflicting duplicate for "
                    f"equation={equation!r}, method={method!r}"
                )
            continue
        benchmark_by_equation[equation][method] = record
        domains[equation] = record.get("domain")

    all_equations = sorted(set(benchmark_by_equation) | set(extrap_new))
    paired = []

    for equation in all_equations:
        bench_methods = benchmark_by_equation.get(equation, {})
        extrap_methods = extrap_new.get(equation, {})
        method_results = {}
        for method in sorted(set(bench_methods) | set(extrap_methods)):
            method_results[method] = _method_result(
                bench_methods.get(method), extrap_methods.get(method)
            )

        # Keep legacy role-only results separate and explicitly labelled.
        legacy = extrap_legacy.get(equation, {})
        legacy_roles = {}
        if isinstance(legacy, dict):
            for role in ("hypatia", "pysr_only"):
                value = legacy.get(role)
                if isinstance(value, dict):
                    legacy_roles[role] = {
                        "method": None,
                        "train_r2": None,
                        "train_success": None,
                        "extrap_r2_near": value.get("extrap_r2_near"),
                        "extrap_r2_far": value.get("extrap_r2_far"),
                        "extrap_success": bool(
                            _finite(value.get("extrap_r2_far"))
                            and value.get("extrap_r2_far") >= EXTRAP_SUCCESS_THRESHOLD
                        ),
                        "source": "legacy_role_only_no_method_identity",
                    }

        def primary(method_name):
            # Do not use legacy role-only metrics here: their method identity is
            # unknown, so pairing them with a named method would reintroduce
            # cross-method comparisons.
            return dict(method_results.get(method_name) or _method_result(None, None))

        hypatia = primary(PRIMARY_HYPATIA_METHOD)
        pysr_only = primary(PRIMARY_PYSR_METHOD)
        paired.append({
            "name": equation,
            "equation_name": equation,
            "equation_id": equation,
            "domain": domains.get(equation),
            "primary_methods": {
                "hypatia": PRIMARY_HYPATIA_METHOD,
                "pysr_only": PRIMARY_PYSR_METHOD,
            },
            "hypatia": hypatia,
            "pysr_only": pysr_only,
            "method_results": method_results,
            "legacy_role_results": legacy_roles,
            "method_count": len(method_results),
            "methods_with_far_extrapolation": sorted(
                method for method, result in method_results.items()
                if _finite(result.get("extrap_r2_far"))
            ),
        })

    n_with_far = sum(
        1 for record in paired
        if _finite(record.get("hypatia", {}).get("extrap_r2_far"))
    )
    n_with_primary_pair = sum(
        1 for record in paired
        if _finite(record.get("hypatia", {}).get("extrap_r2_far"))
        and _finite(record.get("pysr_only", {}).get("extrap_r2_far"))
    )
    print(f"  Paired equation records : {len(paired)}")
    print(f"  Primary Hypatia far-R²  : {n_with_far}")
    print(f"  Primary method pairs    : {n_with_primary_pair}")
    print(
        "  All per-method train/extrap values remain separately recorded; "
        "no best-of-method aggregation was performed."
    )
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
        "--benchmark-dir", required=True, dest="benchmark_dir",
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

    if not extrap_new:
        raise ValueError(
            "No method-level benchmark_results_extrap.json was found. "
            "Legacy role-only results cannot support a method-consistent paired comparison."
        )
    validate_method_coverage(benchmark_records, extrap_new)

    paired = merge(benchmark_records, extrap_new, extrap_legacy)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(paired, f, indent=2)
    print(f"  Written {len(paired)} paired records → {out_path}")

    # -----------------------------------------------------------------------
    # Provenance manifest — this is the "run log" for ablation_paired.json.
    # ablation_paired.json is a *derived* artifact (built here from
    # benchmark_results.json + benchmark_results_extrap.json), not a direct
    # pipeline checkpoint, so it has no checkpoint of its own. This manifest
    # is that checkpoint: which inputs (with content hashes) produced this
    # exact output, when, and what the two ambiguous fields mean.
    # Written as <output>.manifest.json alongside ablation_paired.json and
    # committed together with it by ci_analysis.yml.
    # -----------------------------------------------------------------------
    def _sha256(p: Path) -> str | None:
        if not p.exists():
            return None
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    n_with_far = sum(
        1 for r in paired
        if _finite(r.get("hypatia", {}).get("extrap_r2_far"))
    )
    n_primary_pairs = sum(
        1 for r in paired
        if _finite(r.get("hypatia", {}).get("extrap_r2_far"))
        and _finite(r.get("pysr_only", {}).get("extrap_r2_far"))
    )
    n_method_results_with_far = sum(
        1
        for record in paired
        for result in record.get("method_results", {}).values()
        if _finite(result.get("extrap_r2_far"))
    )
    method_names = sorted({
        method
        for record in paired
        for method in record.get("method_results", {})
    })

    bench_path        = bench_dir / "benchmark_results.json"
    extrap_bench_path = (Path(args.extrap_benchmark_dir) / "benchmark_results_extrap.json"
                          if args.extrap_benchmark_dir else None)

    legacy_files = (
        sorted(Path(args.extrap_dir).glob("extrap_results_*.json"))
        if args.extrap_dir else []
    )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "merge_extrap_into_benchmark.py",
        "output_path": str(out_path),
        "output_sha256": _sha256(out_path),
        "record_count": len(paired),
        "n_primary_hypatia_with_far_extrapolation": n_with_far,
        "n_primary_method_pairs": n_primary_pairs,
        "n_method_results_with_far_extrapolation": n_method_results_with_far,
        "primary_methods": {
            "hypatia": PRIMARY_HYPATIA_METHOD,
            "pysr_only": PRIMARY_PYSR_METHOD,
        },
        "method_names_preserved": method_names,
        "pairing_policy": (
            "exact equation + exact method name; when extrapolation exists, train "
            "and extrapolation metrics come from the same benchmark_results_extrap.json "
            "row; benchmark_results.json is retained as reference metadata only; "
            "no role-wise maximum or cross-method train/extrap pairing"
        ),
        "inputs": {
            "benchmark_results.json": {
                "path": str(bench_path),
                "sha256": _sha256(bench_path),
            },
            "benchmark_results_extrap.json": {
                "path": str(extrap_bench_path) if extrap_bench_path else None,
                "sha256": _sha256(extrap_bench_path) if extrap_bench_path else None,
            },
            "extrap_results_legacy": [
                {"path": str(fp), "sha256": _sha256(fp)}
                for fp in legacy_files
            ],
        },
        "field_definitions": {
            "train_success": (
                "Whether the underlying fit completed/converged, inherited from the "
                "upstream benchmark's per-run 'success' flag. NOT a measure of "
                "extrapolation quality -- almost always true even when "
                "extrap_r2_far is very poor."
            ),
            "extrap_success": (
                f"extrap_r2_far is finite and >= {EXTRAP_SUCCESS_THRESHOLD} "
                "(mirrors run_analysis.py EXTRAP_SUCCESS_THRESHOLD / Tier-3 "
                "'success-subset' criterion, paper §10.7 '9/30 successes'). "
                "Keep this constant in sync with run_analysis.py if it changes."
            ),
            "extrap_r2_near": (
                "Copied from the same equation/method row in "
                "benchmark_results_extrap.json. Null means the upstream runner "
                "did not emit a near-domain score for that method."
            ),
        },
    }
    manifest_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Written provenance manifest → {manifest_path}")

    # Warn if Mann-Whitney will fail
    if n_primary_pairs == 0:
        print(
            "::warning::ablation_paired.json has 0 complete primary-method pairs. "
            "run_analysis.py will emit TOO_FEW_MW_PAIRS. "
            "Ensure run_all.sh --step exp2_feynman_extrap ran successfully and "
            "benchmark_results_extrap.json was produced.",
            file=sys.stderr,
        )
    elif n_primary_pairs < 3:
        print(
            f"::warning::Only {n_with_far} equation(s) have extrap_r2_far. "
            f"Mann-Whitney test needs ≥ 3 pairs.",
            file=sys.stderr,
        )
    else:
        print(f"  OK: {n_primary_pairs} equations have both primary methods' far-R² — "
              f"sufficient for Mann-Whitney test.")


if __name__ == "__main__":
    main()

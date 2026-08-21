#!/usr/bin/env python3
"""
investigate_arrhenius.py — Pull the *actual* HypatiaX Hybrid output for the
Arrhenius task out of one or more raw per-task result JSON files, so you can
sanity-check the real formula/predictions instead of trusting a summary R^2
number that disagrees between sources.

WHY THIS EXISTS
----------------
Two sources disagree sharply on Arrhenius's Far-regime R^2 for HypatiaX
Hybrid: a table in the paper says ~0.90 (success), a regenerated figure says
-inf (crash). A summary R^2 number can't settle that on its own — the only
way to know which is real is to look at what HypatiaX actually predicted
and, if raw points are available, recompute R^2 by hand.

WHAT THIS SCRIPT DOES
----------------------
1. Loads one or more raw result JSON files (the per-task files your harness
   writes — the same kind of file generate_table1.py consumes).
2. Recursively searches each file for anything that looks like the
   Arrhenius task (case-insensitive name match), without assuming an exact
   schema, since the schema of your rerun's output isn't known in advance.
3. For every match, prints:
     - which arm(s) it found (pure_llm / neural_network / hybrid, or
       whatever keys are present)
     - the predicted formula/expression string, if present under any of a
       few common key names (formula, expression, equation, best_formula, ...)
     - the reported R^2 / time / decision fields, if present
     - if raw (y_true, y_pred) arrays are present under any of a few common
       key names, recomputes R^2 directly with the standard formula, so you
       can compare the recomputed value against whatever summary number is
       reported elsewhere.
4. If two or more files are given, prints each file's Arrhenius block one
   after another so you can visually diff the run that built the paper's
   table against your rerun.

USAGE
-----
    python3 investigate_arrhenius.py path/to/original_results.json path/to/rerun_results.json

    # or with globs:
    python3 investigate_arrhenius.py "./results/*seed*.json"

This script does not guess or fabricate anything: if a file has no
Arrhenius-like entry, or no formula field under any of the checked names,
it says so explicitly rather than inventing a value.
"""

from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

TASK_NAME_NEEDLE = "arrhenius"

FORMULA_KEYS = [
    "formula", "expression", "equation", "best_formula", "predicted_formula",
    "sympy_expr", "expr_str", "final_formula", "formula_str",
]
R2_KEYS = ["r2", "R2", "r_squared", "R_squared", "score"]
TIME_KEYS = ["time_s", "time", "runtime_s"]
DECISION_KEYS = ["decision"]
YTRUE_KEYS = ["y_true", "y_test", "targets", "ground_truth"]
YPRED_KEYS = ["y_pred", "predictions", "preds", "y_hat"]


def load_files(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if not matches and Path(pat).exists():
            matches = [pat]
        if not matches:
            print(f"  [warn] no files matched: {pat}", file=sys.stderr)
        files.extend(Path(m) for m in matches)
    return files


def first_present(d: dict, keys: list[str]):
    for k in keys:
        if k in d and d[k] is not None:
            return k, d[k]
    return None, None


def recompute_r2(y_true, y_pred) -> float | None:
    try:
        yt = [float(v) for v in y_true]
        yp = [float(v) for v in y_pred]
    except (TypeError, ValueError):
        return None
    if len(yt) != len(yp) or len(yt) == 0:
        return None
    mean_yt = sum(yt) / len(yt)
    ss_tot = sum((v - mean_yt) ** 2 for v in yt)
    ss_res = sum((t - p) ** 2 for t, p in zip(yt, yp))
    if ss_tot == 0:
        return None
    return 1.0 - ss_res / ss_tot


def find_arrhenius_nodes(obj, path="root") -> list[tuple[str, dict]]:
    """Recursively walk the JSON looking for dict nodes whose task name
    matches 'arrhenius', returning (path, node) pairs. Schema-agnostic by
    design since we don't know your rerun's exact structure."""
    found = []
    if isinstance(obj, dict):
        name = obj.get("name") or obj.get("task") or obj.get("task_name") or obj.get("id")
        if isinstance(name, str) and TASK_NAME_NEEDLE in name.lower():
            found.append((path, obj))
        for k, v in obj.items():
            found.extend(find_arrhenius_nodes(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found.extend(find_arrhenius_nodes(v, f"{path}[{i}]"))
    return found


def describe_arm(arm_name: str, arm_dict: dict, indent: str = "    ") -> None:
    if not isinstance(arm_dict, dict):
        print(f"{indent}{arm_name}: (not a dict — {type(arm_dict).__name__})")
        return
    fk, fv = first_present(arm_dict, FORMULA_KEYS)
    rk, rv = first_present(arm_dict, R2_KEYS)
    tk, tv = first_present(arm_dict, TIME_KEYS)
    dk, dv = first_present(arm_dict, DECISION_KEYS)

    print(f"{indent}{arm_name}:")
    print(f"{indent}  formula   : {fv if fv is not None else '(no formula field found — checked ' + ', '.join(FORMULA_KEYS) + ')'}")
    print(f"{indent}  reported R^2 : {rv if rv is not None else '(none found)'}")
    if dv is not None:
        print(f"{indent}  decision  : {dv}")
    if tv is not None:
        print(f"{indent}  time_s    : {tv}")

    ytk, ytv = first_present(arm_dict, YTRUE_KEYS)
    ypk, ypv = first_present(arm_dict, YPRED_KEYS)
    if ytv is not None and ypv is not None:
        recomputed = recompute_r2(ytv, ypv)
        print(f"{indent}  RAW POINTS FOUND ({ytk} / {ypk}, n={len(ytv)}) — recomputed R^2 = "
              f"{recomputed if recomputed is not None else 'could not compute (bad/empty data)'}")
        if recomputed is not None and rv is not None:
            try:
                if not math.isclose(recomputed, float(rv), rel_tol=1e-3, abs_tol=1e-3):
                    print(f"{indent}  [!] recomputed R^2 does NOT match reported R^2 "
                          f"({recomputed:.4g} vs {rv}) — the summary number may be stale or mis-scored.")
            except (TypeError, ValueError):
                pass
    else:
        print(f"{indent}  (no raw y_true/y_pred arrays found under this arm — cannot independently recompute R^2)")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    files = load_files(sys.argv[1:])
    if not files:
        print("[error] no input files found.", file=sys.stderr)
        sys.exit(2)

    for f in files:
        print("=" * 78)
        print(f"FILE: {f}")
        print("=" * 78)
        try:
            with open(f) as fh:
                data = json.load(fh)
        except Exception as e:
            print(f"  [error] could not parse as JSON: {e}")
            continue

        nodes = find_arrhenius_nodes(data)
        if not nodes:
            print(f"  No task matching '{TASK_NAME_NEEDLE}' found in this file "
                  f"(searched all nested dict 'name'/'task'/'task_name'/'id' fields).")
            continue

        for path, node in nodes:
            print(f"\n  Found Arrhenius-like node at: {path}")
            results = node.get("results", node)  # some schemas nest arms under "results"
            if isinstance(results, dict) and any(
                k in results for k in ("pure_llm", "neural_network", "hybrid")
            ):
                for arm in ("pure_llm", "neural_network", "hybrid"):
                    if arm in results:
                        describe_arm(arm, results[arm])
            else:
                # Fall back: just dump top-level formula/R2 fields on the node itself.
                describe_arm("(task-level, no per-arm 'results' key found)", node)
        print()


if __name__ == "__main__":
    main()

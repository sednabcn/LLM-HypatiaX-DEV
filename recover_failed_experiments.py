#!/usr/bin/env python3
"""
recover_failed_experiments.py
==============================
Detect and recover failed experiment cases from ANY HypatiaX result tree.

Replaces the original single-pipeline version with a fully generalised tool
that works across all experiments (exp1, exp1b, exp2, exp2_feynman, exp3,
exp3b, suppA, suppB, suppB_sc, hybrid_all_domains, instability, extrap)
and with any checkpoint format (pipeline-level OR equation-level).

Usage
-----
    python3 recover_failed_experiments.py --scan
    python3 recover_failed_experiments.py --clean
    python3 recover_failed_experiments.py --recover
    python3 recover_failed_experiments.py --dry-run

    # Scope to one experiment
    python3 recover_failed_experiments.py --recover --exp suppB_sc

    # Use a different results tree or runner
    python3 recover_failed_experiments.py --results-dir /data/results \\
        --runner "bash run_all.sh --step {step}"

Options
-------
    --results-dir PATH      Path to results root (default: auto-detect)
    --checkpoint PATH       Path to a single checkpoint JSON (default: auto-detect)
    --checkpoints-dir DIR   Scan all *_checkpoint.json files in DIR
    --exp FRAGMENT          Filter by experiment name fragment
    --output FILE           Output manifest/script name (default: recover_failed.sh)
    --min-r2 FLOAT          Minimum acceptable R² (default: 0.0)
    --max-rmse FLOAT        Maximum acceptable RMSE (default: inf)
    --runner TEMPLATE       Shell command template to re-run a step.
                            Use {step} as placeholder.
                            Default: "python3 run_all_checkpoint.py --resume --only {step}"
    --verbose / -v          Print per-equation pass/fail detail
    --dry-run               Preview only, no modifications

Exit codes
----------
    0   Clean (or nothing to do)
    1   Failures found / write error
    2   Path not found
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# ── numpy is optional — used only for isfinite; we fall back to math ──────────
try:
    import numpy as np
    _isfinite = np.isfinite
except ImportError:
    def _isfinite(x):  # type: ignore[misc]
        try:
            return math.isfinite(float(x))
        except (TypeError, ValueError):
            return False

# =============================================================================
# Configuration — step catalogue
# =============================================================================
# Complete mapping of ALL current _STEP_ORDER entries.
# Keys must match run_all.sh _STEP_ORDER exactly.
# Each value is a list of lowercase filename/path fragments that identify
# result files belonging to that step.  Add new steps here when run_all.sh grows.

STEP_SEARCH_PATTERNS: dict[str, list[str]] = {
    "exp1":              ["exp1_ablation", "defi_74", "core15",
                          "hypatiax_defi_benchmark_v3",
                          "exp1_rf01", "ablation_"],
    "exp1b":             ["portfolio_variance", "portfolio", "defi_v3_"],
    "exp2_feynman":      ["exp2/exp2_results", "feynman-tests/exp2"],
    "exp2":              ["exp2_run", "all30", "five_system",
                          "all_systems_merged"],
    "exp3":              ["nguyen12", "nguyen", "exp3_nguyen"],
    "exp3b":             ["exp3b", "nguyen12_seed"],
    "suppA":             ["hybrid_routing", "consolidated_hybrid",
                          "hybrid_system", "hybrid_llm_nn/defi"],
    "suppB":             ["noise_sweep", "noise-sweep/noise_sweep",
                          "suppb_noise"],
    "suppB_sc":          ["sample_complexity", "sample-complexity",
                          "suppb_sc"],
    "hybrid_all_domains":["hybrid_llm_nn/all_domains",
                          "hybrid_llm_nn_all_domains",
                          "all_domains_hybrid"],
    "instability":       ["instability_analysis", "instability_extrapolation",
                          "run_instability", "k=30"],
    "extrap":            ["extrapolation_comparative",
                          "all_domains_extrap_v4",
                          "comparison_results/extrapolation"],
    "tables":            ["tables/", ".tex"],
    "figures":           ["figures/fig_", "generate_figures"],
}

# Fields that carry R² values in any HypatiaX result JSON
R2_FIELDS = [
    "r2", "r2_score", "train_r2",
    "extrap_r2_near", "extrap_r2_medium", "extrap_r2_far",
    "hyp_extrap_r2", "nn_extrap_r2",
]

# Fields that carry RMSE values
RMSE_FIELDS = [
    "rmse", "train_rmse",
    "extrap_rmse_near", "extrap_rmse_medium", "extrap_rmse_far",
]

# String patterns in expression / formula fields that signal failure
FAILURE_STRINGS = {
    "DISCOVERY_FAILED", "NO_VALID_EQUATIONS", "TIMED_OUT", "ERROR",
    "N/A", "inf", "NaN", "nan", "null", "None",
}

# Default runner template (overridable via --runner)
DEFAULT_RUNNER = "python3 run_all_checkpoint.py --resume --only {step}"


# =============================================================================
# Path auto-detection
# =============================================================================

def find_results_dir(hint: Optional[str] = None) -> Path:
    if hint:
        p = Path(hint)
        if p.exists():
            return p.resolve()
        raise FileNotFoundError(f"Results dir not found: {hint}")

    env = os.environ.get("RESULTS_DIR")
    if env:
        p = Path(env)
        if p.exists():
            return p.resolve()

    candidates = [
        Path("hypatiax/data/results"),
        Path("../hypatiax/data/results"),
        Path("papers/2025-JMLR/hypatiax/data/results"),
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()

    # Walk upward looking for a results dir that contains comparison_results/
    for p in Path.cwd().rglob("results"):
        if p.is_dir() and (p / "comparison_results").exists():
            return p

    raise FileNotFoundError(
        "Could not locate results directory. "
        "Pass --results-dir or set RESULTS_DIR env-var."
    )


def find_checkpoints(
    checkpoint_hint: Optional[str] = None,
    checkpoints_dir: Optional[str] = None,
    exp_filter: Optional[str] = None,
) -> list[Path]:
    """
    Return a list of checkpoint paths to operate on.
    Priority: --checkpoint > --checkpoints-dir > auto-detect.
    """
    if checkpoint_hint:
        p = Path(checkpoint_hint)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return [p.resolve()]

    search_root: Optional[Path] = None
    if checkpoints_dir:
        search_root = Path(checkpoints_dir)
    else:
        env = os.environ.get("CHECKPOINT_DIR")
        if env:
            search_root = Path(env)
        else:
            for candidate in [Path("logs"), Path("../logs")]:
                if candidate.is_dir():
                    search_root = candidate
                    break

    if search_root and search_root.is_dir():
        paths = sorted(search_root.glob("**/*_checkpoint.json"))
        if exp_filter:
            paths = [p for p in paths if exp_filter.lower() in p.name.lower()]
        if paths:
            return [p.resolve() for p in paths]

    # Absolute fallback
    fallback = Path("logs/pipeline_checkpoint.json")
    if fallback.exists():
        return [fallback.resolve()]

    return []


# =============================================================================
# Failure detection
# =============================================================================

def _is_failed_value(
    value: Any,
    min_r2: float = 0.0,
    max_rmse: float = float("inf"),
    is_r2_field: bool = False,
    is_rmse_field: bool = False,
) -> tuple[bool, str]:
    """Return (failed, reason) for a single scalar value."""
    if value is None:
        return True, "None value"

    if isinstance(value, bool):
        return False, ""   # booleans are not metrics

    if isinstance(value, (int, float)):
        if not _isfinite(value):
            return True, f"non-finite ({value})"
        if is_r2_field and value < min_r2:
            return True, f"r2={value:.4f} < {min_r2}"
        if is_rmse_field and value > max_rmse:
            return True, f"rmse={value:.4f} > {max_rmse}"
        return False, ""

    if isinstance(value, str):
        low = value.lower().strip()
        if not low:
            return True, "empty string"
        for pat in FAILURE_STRINGS:
            if pat.lower() in low:
                return True, f"string='{pat}'"
        return False, ""

    return False, ""


def scan_result_file(
    filepath: Path,
    min_r2: float = 0.0,
    max_rmse: float = float("inf"),
    verbose: bool = False,
) -> tuple[list[dict], dict]:
    """
    Recursively scan one result JSON for failed entries.
    Returns (failed_entries, stats).
    """
    failed_entries: list[dict] = []
    stats: dict = {"total": 0, "failed": 0, "file": str(filepath)}

    try:
        with open(filepath, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        stats["error"] = str(exc)
        return [], stats

    def _walk(obj: Any, path: str = "") -> None:
        if isinstance(obj, dict):
            has_r2   = any(k in obj for k in R2_FIELDS)
            has_rmse = any(k in obj for k in RMSE_FIELDS)
            has_expr = any(k in obj for k in
                           ["expression", "best_expression", "formula",
                            "final_formula"])

            if has_r2 or has_rmse or has_expr:
                stats["total"] += 1
                eq_name = (obj.get("equation_name") or obj.get("name")
                           or obj.get("equation") or path)
                cond    = (obj.get("condition") or obj.get("llm_mode")
                           or obj.get("mode") or "—")
                reasons: list[str] = []

                for field in R2_FIELDS:
                    if field in obj:
                        bad, why = _is_failed_value(
                            obj[field], min_r2=min_r2,
                            is_r2_field=True)
                        if bad:
                            reasons.append(f"{field}: {why}")

                for field in RMSE_FIELDS:
                    if field in obj:
                        bad, why = _is_failed_value(
                            obj[field], max_rmse=max_rmse,
                            is_rmse_field=True)
                        if bad:
                            reasons.append(f"{field}: {why}")

                for ef in ["expression", "best_expression",
                           "formula", "final_formula"]:
                    if isinstance(obj.get(ef), str):
                        expr = obj[ef]
                        bad, why = _is_failed_value(expr)
                        if bad:
                            reasons.append(f"{ef}: {why}")

                if obj.get("timed_out") or obj.get("timeout"):
                    reasons.append("timeout=True")
                if obj.get("success") is False:
                    reasons.append("success=False")
                if obj.get("error") and str(obj["error"]).strip():
                    reasons.append(f"error={str(obj['error'])[:80]}")

                if reasons:
                    stats["failed"] += 1
                    failed_entries.append({
                        "file":        str(filepath),
                        "path":        path,
                        "equation":    eq_name,
                        "condition":   cond,
                        "r2":          obj.get("r2") or obj.get("train_r2"),
                        "r2_far":      obj.get("extrap_r2_far"),
                        "expression":  (
                            obj.get("expression")
                            or obj.get("best_expression")
                            or obj.get("formula", "")
                        )[:100],
                        "reasons":     reasons,
                    })
                    if verbose:
                        print(f"  ✗ {eq_name} [{cond}] — "
                              f"{', '.join(reasons[:2])}")
                elif verbose:
                    print(f"  ✓ {eq_name} [{cond}]")

            # Recurse
            for k, v in obj.items():
                _walk(v, f"{path}.{k}" if path else k)

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                _walk(item, f"{path}[{i}]")

    _walk(data)
    return failed_entries, stats


def scan_all_results(
    results_dir: Path,
    min_r2: float = 0.0,
    max_rmse: float = float("inf"),
    verbose: bool = False,
    exp_filter: Optional[str] = None,
) -> tuple[list[dict], dict]:
    """Scan all result JSON files under results_dir."""
    all_failed: list[dict] = []
    totals = {"files_scanned": 0, "total_entries": 0, "total_failed": 0}

    json_files = [
        f for f in results_dir.rglob("*.json")
        if "checkpoint" not in f.name and not f.name.startswith(".")
    ]

    if exp_filter:
        json_files = [
            f for f in json_files
            if exp_filter.lower() in str(f).lower()
        ]

    print(f"Scanning {len(json_files)} JSON file(s) in {results_dir}"
          + (f" [filter: '{exp_filter}']" if exp_filter else "") + " …")

    for jf in json_files:
        failed, stats = scan_result_file(jf, min_r2, max_rmse, verbose)
        all_failed.extend(failed)
        totals["files_scanned"]  += 1
        totals["total_entries"]  += stats["total"]
        totals["total_failed"]   += stats["failed"]
        if verbose and failed:
            print(f"  → {jf.name}: {len(failed)}/{stats['total']} failed")

    return all_failed, totals


# =============================================================================
# Step identification
# =============================================================================

def identify_step(entry: dict) -> str:
    """
    Map a failed entry back to its run_all.sh step ID.
    Returns 'unknown' (with a warning) when no pattern matches,
    instead of silently defaulting to 'exp1'.
    """
    filepath = Path(entry["file"])
    file_str = str(filepath).lower()
    equation = entry.get("equation", "").lower()

    for step, patterns in STEP_SEARCH_PATTERNS.items():
        for pat in patterns:
            if pat.lower() in file_str or pat.lower() in equation:
                return step

    print(f"  ⚠  Could not identify step for: {filepath.name} "
          f"(equation={entry.get('equation', '?')}) — tagged 'unknown'",
          file=sys.stderr)
    return "unknown"


# =============================================================================
# Checkpoint cleaner  (handles BOTH pipeline-level and equation-level formats)
# =============================================================================

def clean_checkpoint(
    checkpoint_path: Path,
    failed_entries: list[dict],
    dry_run: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Remove failed entries from a checkpoint file so they will be re-run.

    Handles two checkpoint formats:
      • Pipeline-level:  { "exp1": "pass", "exp2": "fail", … }
      • Equation-level:  { "results": { "eq_name": { "result": {…} } },
                           "completed": ["eq_name", …] }
    """
    if not checkpoint_path.exists():
        print(f"  ⚠  Checkpoint not found: {checkpoint_path} — skipping",
              file=sys.stderr)
        return {"removed": 0, "backup": None}

    try:
        with open(checkpoint_path) as fh:
            ckpt = json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"  ❌  JSON error in {checkpoint_path}: {exc}", file=sys.stderr)
        return {"removed": 0, "backup": None}

    removed_count = 0
    removed_steps: dict[str, list[str]] = defaultdict(list)

    # ── Detect format ─────────────────────────────────────────────────────────
    is_equation_level = "results" in ckpt and isinstance(ckpt["results"], dict)

    if is_equation_level:
        # Equation-level: remove individual equation entries
        results   = ckpt["results"]
        completed = ckpt.get("completed", [])

        failed_eq_names = {
            e.get("equation") for e in failed_entries
            if e.get("equation")
        }

        for eq_name in list(results.keys()):
            entry   = results[eq_name]
            res     = entry.get("result") if isinstance(entry, dict) else None
            is_bad  = (
                eq_name in failed_eq_names
                or res is None
                or (isinstance(res, dict) and not res.get("success"))
            )
            if is_bad:
                if dry_run:
                    print(f"  [DRY RUN] Would remove equation: {eq_name}")
                else:
                    del results[eq_name]
                    removed_count += 1
                    step = identify_step({"file": str(checkpoint_path),
                                          "equation": eq_name})
                    removed_steps[step].append(eq_name)
                    if verbose:
                        print(f"  ✗ removed {eq_name} from {checkpoint_path.name}")

        if not dry_run:
            ckpt["results"]   = results
            ckpt["completed"] = [k for k in completed if k in results]

    else:
        # Pipeline-level: mark failed steps as 'fail'
        failed_by_step: dict[str, list[dict]] = defaultdict(list)
        for entry in failed_entries:
            failed_by_step[identify_step(entry)].append(entry)

        for step, entries in failed_by_step.items():
            if step not in ckpt:
                continue
            if ckpt[step] == "pass":
                if dry_run:
                    print(f"  [DRY RUN] Would mark step '{step}' as fail")
                else:
                    ckpt[step] = "fail"
                    removed_count += 1
                    removed_steps[step].append("step_marked_fail")

            # Handle nested {step: {eq_name: result}} structure
            for entry in entries:
                eq_name = entry.get("equation")
                if eq_name and isinstance(ckpt.get(step), dict):
                    if eq_name in ckpt[step]:
                        if dry_run:
                            print(f"  [DRY RUN] Would remove {step}/{eq_name}")
                        else:
                            del ckpt[step][eq_name]
                            removed_count += 1
                            removed_steps[step].append(eq_name)

    if not dry_run and removed_count > 0:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = checkpoint_path.with_suffix(f".backup_{ts}.json")
        shutil.copy(checkpoint_path, backup)
        print(f"  📁 Backup: {backup}")

        tmp = checkpoint_path.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            json.dump(ckpt, fh, indent=2, default=str)
        os.replace(tmp, checkpoint_path)
        print(f"  ✓ Cleaned: {removed_count} entries removed from "
              f"{checkpoint_path.name}")

        return {"removed": removed_count, "removed_steps": dict(removed_steps),
                "backup": str(backup)}

    if removed_count == 0 and not dry_run:
        print(f"  ℹ  {checkpoint_path.name}: nothing to clean")

    return {"removed": removed_count,
            "removed_steps": dict(removed_steps), "backup": None}


# =============================================================================
# Recovery script generator
# =============================================================================

def generate_recover_script(
    failed_entries: list[dict],
    output_path: Path,
    checkpoint_path: Path,
    runner_template: str = DEFAULT_RUNNER,
) -> Path:
    """
    Generate a shell script + JSON manifest to re-run failed experiments.
    The runner command is fully configurable via runner_template.
    """
    step_entries: dict[str, list[dict]] = defaultdict(list)
    for entry in failed_entries:
        step_entries[identify_step(entry)].append(entry)

    lines = [
        "#!/bin/bash",
        "# Auto-generated recovery script",
        f"# Generated : {datetime.now().isoformat()}",
        f"# Failed    : {len(failed_entries)} entries across "
        f"{len(step_entries)} step(s)",
        f"# Runner    : {runner_template}",
        "#",
        "set -euo pipefail",
        "",
        'echo "=== Re-running failed experiments ==="',
        "",
    ]

    for step, entries in sorted(step_entries.items()):
        cmd = runner_template.format(step=step)
        lines += [
            "",
            f'echo "  --- Step: {step} ({len(entries)} failed) ---"',
            cmd,
        ]

    lines += [
        "",
        'echo "=== Recovery complete ==="',
        "",
        "# Optional: verify results",
        "# python3 recover_failed_experiments.py --scan",
    ]

    # JSON manifest
    manifest: dict = {
        "timestamp":          datetime.now().isoformat(),
        "total_failed":       len(failed_entries),
        "runner_template":    runner_template,
        "checkpoint_cleaned": str(checkpoint_path),
        "recovery_command":   f"bash {output_path.name}",
        "by_step": {
            step: [
                {
                    "equation":  e.get("equation"),
                    "condition": e.get("condition"),
                    "file":      e.get("file"),
                    "reasons":   e.get("reasons"),
                }
                for e in entries
            ]
            for step, entries in step_entries.items()
        },
        "failed_entries": failed_entries,
    }

    manifest_path = output_path.with_suffix(".json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    with open(output_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    output_path.chmod(0o755)

    print(f"  ✓ Script  : {output_path}")
    print(f"  ✓ Manifest: {manifest_path}")
    return output_path


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detect and recover failed HypatiaX experiment cases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--scan",    action="store_true",
                   help="Scan results for failures")
    p.add_argument("--clean",   action="store_true",
                   help="Clean checkpoint(s) of failed entries")
    p.add_argument("--recover", action="store_true",
                   help="Full pipeline: scan + clean + generate recover script")
    p.add_argument("--dry-run", action="store_true",
                   help="Preview only, no modifications")

    p.add_argument("--results-dir",     default=None, metavar="PATH",
                   help="Path to results root directory")
    p.add_argument("--checkpoint",      default=None, metavar="PATH",
                   help="Path to a single checkpoint JSON")
    p.add_argument("--checkpoints-dir", default=None, metavar="DIR",
                   help="Directory to scan for *_checkpoint.json files")
    p.add_argument("--exp",             default=None, metavar="FRAGMENT",
                   help="Filter results/checkpoints by experiment name fragment "
                        "(e.g. suppB_sc, feynman, extrap)")
    p.add_argument("--output",          default="recover_failed.sh",
                   metavar="FILE",
                   help="Output shell script name (default: recover_failed.sh)")
    p.add_argument("--min-r2",          type=float, default=0.0,
                   metavar="FLOAT",
                   help="Minimum acceptable R² (default: 0.0)")
    p.add_argument("--max-rmse",        type=float, default=float("inf"),
                   metavar="FLOAT",
                   help="Maximum acceptable RMSE (default: inf)")
    p.add_argument("--runner",          default=DEFAULT_RUNNER,
                   metavar="TEMPLATE",
                   help=f"Runner command template. Use {{step}} as placeholder. "
                        f"Default: '{DEFAULT_RUNNER}'")
    p.add_argument("--verbose", "-v",   action="store_true",
                   help="Print per-equation pass/fail detail")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    # Default action: full recovery
    if not (args.scan or args.clean or args.recover):
        args.recover = True

    print("=" * 68)
    print("  FAILED EXPERIMENT RECOVERY TOOL")
    print("=" * 68)
    if args.exp:
        print(f"  Filter     : '{args.exp}'")
    print(f"  Min R²     : {args.min_r2}")
    print(f"  Max RMSE   : {args.max_rmse}")
    print(f"  Runner     : {args.runner}")
    print(f"  Dry run    : {args.dry_run}")
    print("=" * 68)

    failed_entries: list[dict] = []

    # ── Step 1: Scan ──────────────────────────────────────────────────────────
    if args.scan or args.recover:
        try:
            results_dir = find_results_dir(args.results_dir)
        except FileNotFoundError as exc:
            print(f"❌  {exc}", file=sys.stderr)
            return 2

        print(f"\n  Results dir: {results_dir}")
        print("\n🔍 Scanning for failures …")
        failed_entries, scan_stats = scan_all_results(
            results_dir,
            min_r2=args.min_r2,
            max_rmse=args.max_rmse,
            verbose=args.verbose,
            exp_filter=args.exp,
        )

        n = scan_stats["total_entries"]
        f = scan_stats["total_failed"]
        pct = f / max(1, n) * 100
        print(f"\n📊 Scan results:")
        print(f"   Files scanned  : {scan_stats['files_scanned']}")
        print(f"   Total entries  : {n}")
        print(f"   Failed entries : {f}  ({pct:.1f}%)")

        if failed_entries:
            reasons_count: dict[str, int] = defaultdict(int)
            for e in failed_entries:
                for r in e["reasons"]:
                    reasons_count[r.split(":")[0].strip()] += 1
            print("\n❌ Failure breakdown (top 10):")
            for reason, count in sorted(
                reasons_count.items(), key=lambda x: -x[1]
            )[:10]:
                print(f"   {reason:<45} {count:>4}")

            if args.verbose:
                print("\n📋 Failed entries (first 20):")
                for e in failed_entries[:20]:
                    print(f"   {Path(e['file']).name}: "
                          f"{e.get('equation', '?')} — "
                          f"{', '.join(e['reasons'][:2])}")
                if len(failed_entries) > 20:
                    print(f"   … and {len(failed_entries) - 20} more")
        else:
            print("\n✅ No failed experiments found!")
            if not args.recover:
                return 0

    # ── Step 2: Clean checkpoints ─────────────────────────────────────────────
    if (args.clean or args.recover) and failed_entries:
        try:
            checkpoints = find_checkpoints(
                args.checkpoint, args.checkpoints_dir, args.exp
            )
        except FileNotFoundError as exc:
            print(f"❌  {exc}", file=sys.stderr)
            return 2

        if not checkpoints:
            print("\n⚠  No checkpoint files found — skipping clean step.",
                  file=sys.stderr)
        else:
            print(f"\n🧹 Cleaning {len(checkpoints)} checkpoint(s) …")
            total_removed = 0
            for cp in checkpoints:
                print(f"\n  [{cp.name}]")
                result = clean_checkpoint(
                    cp, failed_entries,
                    dry_run=args.dry_run, verbose=args.verbose,
                )
                total_removed += result["removed"]
            print(f"\n  Total removed: {total_removed} entries")

    # ── Step 3: Generate recovery script ──────────────────────────────────────
    if args.recover and failed_entries and not args.dry_run:
        print("\n📝 Generating recovery script …")
        cp_path = (
            Path(args.checkpoint) if args.checkpoint
            else Path("logs/pipeline_checkpoint.json")
        )
        script = generate_recover_script(
            failed_entries,
            Path(args.output),
            cp_path,
            runner_template=args.runner,
        )
        print(f"\n🚀 To recover:")
        print(f"   bash {script}")
        steps = sorted({identify_step(e) for e in failed_entries})
        print("\n   Or manually, per step:")
        for step in steps:
            print(f"   {args.runner.format(step=step)}")

    elif args.recover and not failed_entries:
        print("\n✅ No failures to recover!")

    print("\n" + "=" * 68)
    print("  DONE")
    print("=" * 68 + "\n")
    return 1 if failed_entries else 0


if __name__ == "__main__":
    sys.exit(main())

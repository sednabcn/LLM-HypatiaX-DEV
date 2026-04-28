#!/usr/bin/env python3
"""
recover_failed_experiments.py — Detect and recover failed experiment cases

Usage:
    python3 recover_failed_experiments.py --scan          # Scan results for failures
    python3 recover_failed_experiments.py --clean         # Clean checkpoint of failed entries
    python3 recover_failed_experiments.py --recover       # Full: scan + clean + generate recover script
    python3 recover_failed_experiments.py --dry-run       # Preview only, no modifications

Options:
    --results-dir PATH     # Path to hypatiax/data/results (default: auto-detect)
    --checkpoint PATH      # Path to logs/pipeline_checkpoint.json (default: auto-detect)
    --output FILE          # Output manifest for re-run (default: failed_experiments.json)
    --min-r2 FLOAT         # Minimum acceptable R² (default: 0.0)
    --max-rmse FLOAT       # Maximum acceptable RMSE (default: inf)
    --verbose              # Print detailed failure information
"""

import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ============================================================================
# Configuration
# ============================================================================

FAILURE_PATTERNS = [
    "DISCOVERY_FAILED",
    "NO_VALID_EQUATIONS",
    "TIMED_OUT",
    "ERROR",
    "N/A",
    "inf",
    "NaN",
    "nan",
]

# Experiment to step ID mapping (for run_all_checkpoint.py --only)
EXPERIMENT_STEP_MAP = {
    "exp1_ablation": "exp1",
    "exp1_ablation_results": "exp1",
    "defi_74": "exp1",
    "portfolio_variance": "exp1b",
    "feynman": "exp2",
    "nguyen12": "exp3",
    "noise_sweep": "suppB",
    "hybrid_routing": "suppA",
    "instability": "instability",
    "extrapolation_comparative": "extrap",
}

# Inverse: step ID -> search patterns
STEP_SEARCH_PATTERNS = {
    "exp1": ["exp1_ablation", "defi_74", "core15"],
    "exp1b": ["portfolio_variance", "seed"],
    "exp2": ["feynman", "I.", "II.", "III."],
    "exp3": ["nguyen12", "Nguyen"],
    "suppB": ["noise_sweep", "noise"],
    "suppA": ["hybrid_routing"],
    "instability": ["instability", "k=30"],
    "extrap": ["extrapolation_comparative", "extrap"],
}


# ============================================================================
# Result Scanner
# ============================================================================

def find_results_dir() -> Path:
    """Auto-detect results directory."""
    candidates = [
        Path("hypatiax/data/results"),
        Path("../hypatiax/data/results"),
        Path("papers/2025-JMLR/hypatiax/data/results"),
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    # Fallback: search from current directory
    for p in Path.cwd().rglob("results"):
        if p.is_dir() and (p / "comparison_results").exists():
            return p
    raise FileNotFoundError("Could not locate results directory")


def find_checkpoint() -> Path:
    """Auto-detect checkpoint file."""
    candidates = [
        Path("logs/pipeline_checkpoint.json"),
        Path("pipeline_checkpoint.json"),
        Path("../logs/pipeline_checkpoint.json"),
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    # Return default path (may not exist yet)
    return Path("logs/pipeline_checkpoint.json")


def is_failed(value: Any, min_r2: float = 0.0, max_rmse: float = float("inf")) -> Tuple[bool, str]:
    """
    Check if a result value indicates failure.
    
    Returns:
        (is_failed, reason)
    """
    if value is None:
        return True, "None value"
    
    if isinstance(value, (int, float)):
        if not np.isfinite(value):
            return True, f"Non-finite: {value}"
        # Check R² bounds (if value likely represents R²)
        if -1.5 <= value <= 1.5:  # R² typically in [-1, 1]
            if value < min_r2:
                return True, f"R²={value:.4f} < {min_r2}"
        if value > max_rmse:
            return True, f"RMSE={value:.4f} > {max_rmse}"
        return False, ""
    
    if isinstance(value, str):
        value_low = value.lower()
        for pattern in FAILURE_PATTERNS:
            if pattern.lower() in value_low:
                return True, f"String pattern: {pattern}"
        return False, ""
    
    return False, ""


def scan_result_file(
    filepath: Path,
    min_r2: float = 0.0,
    max_rmse: float = float("inf"),
    verbose: bool = False
) -> Tuple[List[Dict], Dict]:
    """
    Scan a single result JSON file for failures.
    
    Returns:
        (failed_entries, summary_stats)
    """
    failed_entries = []
    stats = {"total": 0, "failed": 0, "file": str(filepath)}
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        stats["error"] = str(e)
        return [], stats
    
    def _scan_dict(obj, path: str = "", experiment: str = "", domain: str = ""):
        nonlocal stats
        
        if isinstance(obj, dict):
            # Check if this dict contains result fields
            has_r2 = any(k in obj for k in ["r2", "r2_score", "train_r2", "extrap_r2_far"])
            has_rmse = any(k in obj for k in ["rmse", "train_rmse", "extrap_rmse_far"])
            has_expr = "expression" in obj or "best_expression" in obj or "formula" in obj
            
            if has_r2 or has_rmse or has_expr:
                stats["total"] += 1
                
                # Extract identifiers
                eq_name = obj.get("equation_name") or obj.get("name") or obj.get("equation") or path
                cond = obj.get("condition") or obj.get("llm_mode") or "unknown"
                
                # Check R² fields
                r2_fields = ["r2", "r2_score", "train_r2", "extrap_r2_near", "extrap_r2_medium", "extrap_r2_far"]
                rmse_fields = ["rmse", "train_rmse", "extrap_rmse_near", "extrap_rmse_medium", "extrap_rmse_far"]
                
                failed_reasons = []
                
                for field in r2_fields:
                    if field in obj:
                        is_bad, reason = is_failed(obj[field], min_r2=min_r2)
                        if is_bad:
                            failed_reasons.append(f"{field}={obj[field]} ({reason})")
                
                for field in rmse_fields:
                    if field in obj:
                        is_bad, reason = is_failed(obj[field], max_rmse=max_rmse)
                        if is_bad:
                            failed_reasons.append(f"{field}={obj[field]} ({reason})")
                
                # Check expression string
                for expr_field in ["expression", "best_expression", "formula", "final_formula"]:
                    if expr_field in obj and isinstance(obj[expr_field], str):
                        expr = obj[expr_field]
                        if expr in ["DISCOVERY_FAILED", "NO_VALID_EQUATIONS", "TIMED_OUT", "ERROR"]:
                            failed_reasons.append(f"{expr_field}={expr}")
                        elif "inf" in expr or "nan" in expr:
                            failed_reasons.append(f"{expr_field} contains inf/nan")
                
                # Check timeout flag
                if obj.get("timed_out") or obj.get("timeout"):
                    failed_reasons.append("timeout=True")
                
                if obj.get("success") is False:
                    failed_reasons.append("success=False")
                
                if obj.get("excluded_from_timing"):
                    failed_reasons.append("excluded_from_timing=True")
                
                # Check error field
                if obj.get("error") and str(obj["error"]).strip():
                    failed_reasons.append(f"error={str(obj['error'])[:100]}")
                
                if failed_reasons:
                    stats["failed"] += 1
                    failed_entries.append({
                        "file": str(filepath),
                        "path": path,
                        "equation": eq_name,
                        "condition": cond,
                        "r2": obj.get("r2") or obj.get("train_r2"),
                        "r2_far": obj.get("extrap_r2_far"),
                        "expression": obj.get("expression") or obj.get("best_expression") or obj.get("formula", "")[:100],
                        "reasons": failed_reasons,
                        "original_data": obj,
                    })
                    if verbose:
                        print(f"  ✗ FAIL: {eq_name} [{cond}] - {', '.join(failed_reasons[:2])}")
                elif verbose:
                    print(f"  ✓ PASS: {eq_name} [{cond}]")
            
            # Recurse
            for k, v in obj.items():
                new_path = f"{path}.{k}" if path else k
                new_experiment = experiment or k
                _scan_dict(v, new_path, new_experiment, domain)
        
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                _scan_dict(item, f"{path}[{i}]", experiment, domain)
    
    _scan_dict(data)
    
    return failed_entries, stats


def scan_all_results(
    results_dir: Path,
    min_r2: float = 0.0,
    max_rmse: float = float("inf"),
    verbose: bool = False
) -> Tuple[List[Dict], Dict]:
    """Scan all JSON files in results directory."""
    all_failed = []
    total_stats = {"files_scanned": 0, "total_entries": 0, "total_failed": 0}
    
    json_files = list(results_dir.rglob("*.json"))
    print(f"Scanning {len(json_files)} JSON files in {results_dir}...")
    
    for jf in json_files:
        # Skip checkpoint files and lock files
        if "checkpoint" in jf.name or jf.name.startswith("."):
            continue
        
        failed, stats = scan_result_file(jf, min_r2, max_rmse, verbose)
        all_failed.extend(failed)
        total_stats["files_scanned"] += 1
        total_stats["total_entries"] += stats["total"]
        total_stats["total_failed"] += stats["failed"]
        
        if verbose and failed:
            print(f"  → {jf.name}: {len(failed)}/{stats['total']} failed")
    
    return all_failed, total_stats


# ============================================================================
# Checkpoint Cleaner
# ============================================================================

def identify_step_from_failed_entry(entry: Dict) -> Optional[str]:
    """
    Determine which experiment step a failed entry belongs to.
    """
    filepath = Path(entry["file"])
    file_stem = filepath.stem.lower()
    file_parent = str(filepath.parent).lower()
    
    equation = entry.get("equation", "").lower()
    entry.get("condition", "").lower()
    
    # Check by filename
    for step, patterns in STEP_SEARCH_PATTERNS.items():
        for pattern in patterns:
            if pattern in file_stem or pattern in file_parent:
                return step
            if pattern in equation:
                return step
    
    # Default fallback: heuristic
    if "feynman" in file_stem or "i." in equation or "ii." in equation:
        return "exp2"
    if "nguyen" in file_stem:
        return "exp3"
    if "noise" in file_stem:
        return "suppB"
    if "instability" in file_stem or "k=30" in file_stem:
        return "instability"
    if "portfolio" in file_stem:
        return "exp1b"
    if "routing" in file_stem:
        return "suppA"
    if "extrap" in file_stem:
        return "extrap"
    
    return "exp1"  # Default to exp1


def clean_checkpoint(
    checkpoint_path: Path,
    failed_entries: List[Dict],
    dry_run: bool = False,
    verbose: bool = False
) -> Dict:
    """
    Remove failed entries from checkpoint so they can be re-run.
    
    Returns:
        Stats dict with removed count and backup path
    """
    if not checkpoint_path.exists():
        print(f"⚠ Checkpoint not found: {checkpoint_path}")
        return {"removed": 0, "backup": None}
    
    with open(checkpoint_path, "r") as f:
        checkpoint = json.load(f)
    
    checkpoint.copy()
    removed_count = 0
    removed_steps = defaultdict(list)
    
    # Group failed entries by step
    failed_by_step = defaultdict(list)
    for entry in failed_entries:
        step = identify_step_from_failed_entry(entry)
        failed_by_step[step].append(entry)
    
    # For each failed step, mark checkpoint entries as incomplete
    for step, entries in failed_by_step.items():
        if step not in checkpoint:
            continue
        
        # If the checkpoint has this step as 'pass', change to 'fail'
        if checkpoint.get(step) == "pass":
            if dry_run:
                print(f"  [DRY RUN] Would mark {step} as fail (was pass)")
            else:
                checkpoint[step] = "fail"
                removed_count += 1
                removed_steps[step].append("step_marked_fail")
        
        # Also handle nested experiment results if present
        for entry in entries:
            eq_name = entry.get("equation")
            if eq_name and isinstance(checkpoint.get(step), dict):
                # Some checkpoints have nested structure: {step: {eq_name: result}}
                if eq_name in checkpoint.get(step, {}):
                    if dry_run:
                        print(f"  [DRY RUN] Would remove {step}/{eq_name}")
                    else:
                        del checkpoint[step][eq_name]
                        removed_count += 1
                        removed_steps[step].append(eq_name)
    
    if not dry_run and removed_count > 0:
        # Create backup
        backup_path = checkpoint_path.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        shutil.copy(checkpoint_path, backup_path)
        print(f"📁 Backup saved: {backup_path}")
        
        # Write cleaned checkpoint
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)
        print(f"✓ Checkpoint cleaned: {removed_count} entries removed/modified")
    else:
        print(f"  No checkpoint modifications (dry_run={dry_run}, removed={removed_count})")
    
    return {
        "removed": removed_count,
        "removed_steps": dict(removed_steps),
        "backup": str(backup_path) if not dry_run and removed_count > 0 else None,
    }


# ============================================================================
# Re-run Manifest Generator
# ============================================================================

def generate_recover_script(
    failed_entries: List[Dict],
    output_path: Path,
    checkpoint_path: Path,
) -> Path:
    """
    Generate a shell script to re-run failed experiments.
    """
    # Group by step
    step_entries = defaultdict(list)
    for entry in failed_entries:
        step = identify_step_from_failed_entry(entry)
        step_entries[step].append(entry)
    
    lines = [
        "#!/bin/bash",
        "# Auto-generated recover script",
        f"# Generated: {datetime.now().isoformat()}",
        f"# Failed entries: {len(failed_entries)}",
        "#",
        "",
        "set -e",
        "",
        "echo \"=== Re-running failed experiments ===\"",
        "",
    ]
    
    # Add commands for each step
    for step, entries in step_entries.items():
        lines.append("")
        lines.append("echo \"")
        lines.append(f"  === Step: {step} ===")
        lines.append(f"  Failed equations: {len(entries)}")
        lines.append("\"")
        lines.append(f"python3 run_all_checkpoint.py --resume --only {step}")
        lines.append("")
    
    lines.extend([
        "",
        "echo \"=== Recovery complete ===\"",
        "",
        "# Verify results",
        "python3 run_all_checkpoint.py --verify-only",
    ])
    
    # Also generate JSON manifest
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "total_failed": len(failed_entries),
        "by_step": {
            step: [
                {
                    "equation": e.get("equation"),
                    "condition": e.get("condition"),
                    "file": e.get("file"),
                    "reasons": e.get("reasons"),
                }
                for e in entries
            ]
            for step, entries in step_entries.items()
        },
        "failed_entries": failed_entries,
        "checkpoint_cleaned": str(checkpoint_path),
        "recovery_command": "bash recover_failed.sh",
    }
    
    manifest_path = output_path.with_suffix(".json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    # Write shell script
    script_path = output_path
    with open(script_path, "w") as f:
        f.write("\n".join(lines))
    
    # Make executable
    script_path.chmod(0o755)
    
    print(f"✓ Recovery script: {script_path}")
    print(f"✓ Manifest: {manifest_path}")
    
    return script_path


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Detect and recover failed experiment cases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--scan", action="store_true", help="Scan results for failures")
    parser.add_argument("--clean", action="store_true", help="Clean checkpoint of failed entries")
    parser.add_argument("--recover", action="store_true", help="Full: scan + clean + generate recover script")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, no modifications")
    parser.add_argument("--results-dir", type=str, help="Path to results directory")
    parser.add_argument("--checkpoint", type=str, help="Path to pipeline_checkpoint.json")
    parser.add_argument("--output", type=str, default="recover_failed.sh", help="Output script name")
    parser.add_argument("--min-r2", type=float, default=0.0, help="Minimum acceptable R²")
    parser.add_argument("--max-rmse", type=float, default=float("inf"), help="Maximum acceptable RMSE")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # If no action specified, run full recovery
    if not (args.scan or args.clean or args.recover):
        args.recover = True
    
    # Auto-detect paths
    try:
        results_dir = Path(args.results_dir) if args.results_dir else find_results_dir()
        checkpoint_path = Path(args.checkpoint) if args.checkpoint else find_checkpoint()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print("=" * 68)
    print("FAILED EXPERIMENT RECOVERY TOOL")
    print("=" * 68)
    print(f"Results dir: {results_dir}")
    print(f"Checkpoint:  {checkpoint_path}")
    print(f"Min R²:      {args.min_r2}")
    print(f"Max RMSE:    {args.max_rmse}")
    print(f"Dry run:     {args.dry_run}")
    print("=" * 68)
    
    # Step 1: Scan
    if args.scan or args.recover:
        print("\n🔍 Scanning results for failures...")
        failed_entries, stats = scan_all_results(
            results_dir, args.min_r2, args.max_rmse, args.verbose
        )
        
        print("\n📊 Scan results:")
        print(f"   Files scanned : {stats['files_scanned']}")
        print(f"   Total entries : {stats['total_entries']}")
        print(f"   Failed entries: {stats['total_failed']} ({stats['total_failed']/max(1,stats['total_entries'])*100:.1f}%)")
        
        if failed_entries:
            print("\n❌ Failed experiments by type:")
            reasons_count = defaultdict(int)
            for e in failed_entries:
                for r in e["reasons"]:
                    # Extract short reason
                    short = r.split("(")[0].strip()
                    reasons_count[short] += 1
            
            for reason, count in sorted(reasons_count.items(), key=lambda x: -x[1])[:10]:
                print(f"   {reason}: {count}")
            
            if args.verbose:
                print("\n📋 Failed entries detail:")
                for e in failed_entries[:20]:
                    print(f"   {e['file']}: {e.get('equation', '?')} - {', '.join(e['reasons'][:2])}")
                if len(failed_entries) > 20:
                    print(f"   ... and {len(failed_entries) - 20} more")
        else:
            print("\n✅ No failed experiments found!")
            if not args.recover:
                sys.exit(0)
    
    # Step 2: Clean checkpoint
    if (args.clean or args.recover) and failed_entries:
        print("\n🧹 Cleaning checkpoint...")
        clean_result = clean_checkpoint(
            checkpoint_path, failed_entries, dry_run=args.dry_run, verbose=args.verbose
        )
        print(f"   Removed: {clean_result['removed']} entries")
        if clean_result.get("backup"):
            print(f"   Backup: {clean_result['backup']}")
    
    # Step 3: Generate recover script
    if args.recover and failed_entries and not args.dry_run:
        print("\n📝 Generating recovery script...")
        output_path = Path(args.output)
        script_path = generate_recover_script(
            failed_entries, output_path, checkpoint_path
        )
        
        print("\n🚀 To recover failed experiments:")
        print(f"   bash {script_path}")
        print("\n   Or manually:")
        for step in set(identify_step_from_failed_entry(e) for e in failed_entries):
            print(f"   python3 run_all_checkpoint.py --resume --only {step}")
    
    elif args.recover and not failed_entries:
        print("\n✅ No failures to recover!")
    
    print("\n" + "=" * 68)
    print("RECOVERY PREP COMPLETE")
    print("=" * 68)


if __name__ == "__main__":
    import numpy as np  # noqa: E402
    main()
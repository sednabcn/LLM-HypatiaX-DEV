#!/usr/bin/env python3
"""
assemble_experiment_results.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HypatiaX  ·  Compose partial shard JSONs → final per-experiment result JSON

Source of truth: run_all_checkpoint.py (v8.1)
  - output paths / globs come from each Step's result_glob + post_move
  - merge key comes from what each experiment script actually writes:
      exp1/exp1b         : task_id  (DeFi task string)
      exp2_feynman/exp2  : equation name  (e.g. "I.6.2a")
      exp3/exp3b         : equation name  (e.g. "N1", "N2", ...)
      suppA              : domain key
      suppB              : equation + noise_level composite key
      suppB_sc           : equation + n_samples composite key
      hybrid_all_domains : domain name
      instability        : case_id
      extrap             : equation + extrap_multiplier key

Usage
-----
    # assemble one experiment (workers already ran, shards are in RESULTS_DIR)
    python3 assemble_experiment_results.py --exp exp2_feynman

    # assemble all experiments
    python3 assemble_experiment_results.py --all

    # point at a custom results dir
    python3 assemble_experiment_results.py --all --results-dir /path/to/results

    # also accept downloaded CI shard artifacts (flat dirs per shard)
    python3 assemble_experiment_results.py --exp exp2_feynman \\
        --shard-dirs ./downloaded_artifacts/results-exp2_feynman-shard0 \\
                     ./downloaded_artifacts/results-exp2_feynman-shard1 \\
                     ./downloaded_artifacts/results-exp2_feynman-shard2 \\
                     ./downloaded_artifacts/results-exp2_feynman-shard3

Output (written next to the partial files, inside RESULTS_DIR/<subdir>/)
    <exp>_assembled.json   ← all task records merged, keyed by task_id
    <exp>_assembled.csv    ← flat CSV view of the same records
"""

import argparse
import csv
import glob
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Canonical paths (mirrors run_all_checkpoint.py) ──────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "hypatiax" / "data" / "results"

# ── Per-experiment config ─────────────────────────────────────────────────────
# Each entry mirrors:
#   result_subdir  → EXP_RESULT_SUBDIR dict in run_all_checkpoint.py
#   shard_glob     → patterns the workers write (result_glob + post_move targets)
#   merge_key      → the JSON field used as the unique task identifier
#   partial_prefix → filename prefix the workers produce per task
#
EXP_CONFIG = {
    # ── Phase 1: Core experiments ──────────────────────────────────────────────

    "exp1": dict(
        label="Exp 1 · Core DeFi extrapolation benchmark",
        result_subdir="comparison_results/noise-noiseless/noiseless",
        # hypatiax_defi_benchmark_v3c.py writes:
        #   hypatiax_defi_benchmark_v3*results*.json  (whole-shard summaries)
        #   protocol_core_noiseless_*.json            (per-task files, legacy name)
        shard_globs=[
            "hypatiax_defi_benchmark_v3*results*.json",
            "protocol_core_noiseless_*.json",
        ],
        merge_key="task_id",
        fallback_merge_key="equation_id",
        array_key="results",       # if the file is a wrapper with a list inside
    ),

    "exp1b": dict(
        label="Exp 1b · DeFi seed sweep + portfolio variance",
        result_subdir="comparison_results/noise-noiseless/15",
        shard_globs=[
            "hypatiax_defi_benchmark_v3*results*.json",
            "comparison_FIXED_*.json",
            "*portfolio*variance*.json",
            "defi_v3_*.json",
        ],
        merge_key="task_id",
        fallback_merge_key="equation_id",
        array_key="results",
    ),

    "exp2_feynman": dict(
        label="Exp 2 · Feynman-30 SR benchmark (§10.7)",
        result_subdir="comparison_results/feynman-tests/exp2",
        # run_comparative_suite_benchmark_v2.py writes per-equation JSON files:
        #   exp2_feynman_checkpoint_shard*.json   (shard checkpoints)
        #   <EQ_NAME>.json                        (per-equation result, e.g. I_6_2a.json)
        # The isolated run_exp2_feynman() in run_all_checkpoint.py writes:
        #   <eq_name.replace('.','_')>.json       (e.g. I_6_2a.json)
        #   exp2_results.json                     (whole-run consolidated)
        shard_globs=[
            "exp2_feynman_checkpoint_shard*.json",
            "exp2_results.json",
            "I_*.json",
            "II_*.json",
            "III_*.json",
            "exp2_feynman_merged*.json",
        ],
        merge_key="equation",
        fallback_merge_key="name",
        array_key="results",
    ),

    "exp2": dict(
        label="Exp 2 · Combined five-system comparison — all methods (§10.7 combined)",
        result_subdir="comparison_results/feynman-tests/exp2_multi",
        shard_globs=[
            "exp2_checkpoint_shard*.json",
            "exp2_merged*.json",
            "exp2_stats.json",
        ],
        merge_key="equation",
        fallback_merge_key="task_id",
        array_key="results",
    ),

    "exp3": dict(
        label="Exp 3 · Nguyen-12 SEED=42 (§10.8 primary)",
        result_subdir="extrapolation",
        # exp3_nguyen12_hybrid50v_02.py --seed 42 writes:
        #   *nguyen*seed42*.json   (post_move target, src = RESULTS_DIR root)
        #   full_run_*.json
        #   report_hybrid_*.json
        #   hybrid_defi_*.json
        shard_globs=[
            "*nguyen*seed42*.json",
            "*nguyen12*42*.json",
            "full_run_*.json",
            "report_hybrid_*.json",
            "hybrid_defi_*.json",
        ],
        merge_key="equation",
        fallback_merge_key="task_id",
        array_key="results",
    ),

    "exp3b": dict(
        label="Exp 3b · Nguyen-12 seeds 99/123/777/2024 (§10.8 stability)",
        result_subdir="extrapolation/multi_seed",
        shard_globs=[
            "*nguyen*.json",
            "full_run_*.json",
            "report_hybrid_*.json",
            "hybrid_defi_*.json",
        ],
        merge_key="equation",
        fallback_merge_key="task_id",
        array_key="results",
    ),

    # ── Phase 2: Supplementary benchmarks ─────────────────────────────────────

    "suppA": dict(
        label="Supp A · Hybrid-PySR DeFi benchmark",
        result_subdir="hybrid_pysr/defi",
        shard_globs=[
            "consolidated_hybrid_*.json",
            "hybrid_system*.json",
            "hybrid_llm_nn_all_domains_*.json",
            "ablation_exp1_*.json",
        ],
        merge_key="domain",
        fallback_merge_key="task_id",
        array_key="results",
    ),

    "suppB": dict(
        label="Supp B · Noise sweep σ ∈ {0, 0.5, 1, 5, 10}% × 30 equations",
        result_subdir="comparison_results/feynman-tests/noise-sweep",
        # run_noise_sweep_benchmark.py writes:
        #   noise_sweep_*.json   (per noise-level or per equation)
        shard_globs=[
            "noise_sweep_*.json",
            "suppB_*.json",
        ],
        merge_key="task_id",           # composite: equation + noise_level
        fallback_merge_key="equation",
        array_key="results",
    ),

    "suppB_sc": dict(
        label="Supp B-SC · Sample-complexity sweep n ∈ {50…1000} × 30 equations",
        result_subdir="comparison_results/feynman-tests/sample-complexity",
        shard_globs=[
            "sample_complexity_*.json",
        ],
        merge_key="task_id",           # composite: equation + n_samples
        fallback_merge_key="equation",
        array_key="results",
    ),

    "hybrid_all_domains": dict(
        label="Hybrid · LLM+NN all-domains one-shot run (§10.9)",
        result_subdir="hybrid_llm_nn/all_domains",
        shard_globs=[
            "hybrid_llm_nn_all_domains_*.json",
        ],
        merge_key="domain",
        fallback_merge_key="task_id",
        array_key="results",
    ),

    "instability": dict(
        label="Instability · Index analysis + regime figures (§10.9)",
        result_subdir="figures",
        shard_globs=[
            "instability_analysis.csv",   # treated as-is (not merged as JSON)
            "instability_extrapolation.csv",
        ],
        merge_key="case_id",
        fallback_merge_key="equation",
        array_key=None,                   # CSV-only experiment — handled separately
    ),

    "extrap": dict(
        label="Extrap · OOD extrapolation comparative suite (Tab 9 OOD)",
        result_subdir="comparison_results/extrapolation",
        shard_globs=[
            "all_domains_extrap_v4_*.json",
            "standalone_llm_nn_*.json",
            "standalone_real_methods_*.json",
        ],
        merge_key="equation",
        fallback_merge_key="task_id",
        array_key="results",
    ),
}

ALL_EXPERIMENTS = list(EXP_CONFIG.keys())


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_shard_files(search_dirs: list[Path], globs: list[str]) -> list[Path]:
    """
    Collect files matching any of `globs` inside each directory in `search_dirs`.
    Skips the assembled output files themselves to avoid double-counting.
    """
    found = []
    seen  = set()
    for d in search_dirs:
        if not d.exists():
            continue
        for pattern in globs:
            for match in sorted(d.glob(pattern)):
                if match.is_file() and match not in seen:
                    # Skip files we wrote ourselves
                    if "_assembled" not in match.name:
                        seen.add(match)
                        found.append(match)
    return found


def _extract_records(filepath: Path, cfg: dict) -> list[dict]:
    """
    Load one shard / partial JSON file and return a flat list of task records.

    Handles three shapes:
      A.  {"results": [ {task_id: ..., ...}, ... ]}   ← array under a key
      B.  [ {task_id: ..., ...}, ... ]                ← top-level array
      C.  {task_id: { ... }, task_id2: { ... }}        ← dict keyed by task
      D.  { ... single record ... }                    ← one task record
    """
    try:
        raw = json.loads(filepath.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        print(f"    ⚠  JSON parse error in {filepath.name}: {e}", file=sys.stderr)
        return []

    # Skip stub checkpoints from FIX-G6
    if isinstance(raw, dict) and raw.get("_meta", {}).get("stub"):
        return []

    # Skip worker checkpoint files (they track task IDs, not result payloads)
    if isinstance(raw, dict) and "completed" in raw and "run_id_map" in raw:
        return []

    array_key = cfg.get("array_key")
    merge_key  = cfg["merge_key"]
    fb_key     = cfg.get("fallback_merge_key", "")

    # Shape A: wrapper dict with a list under array_key
    if array_key and isinstance(raw, dict) and isinstance(raw.get(array_key), list):
        records = raw[array_key]
        return [r for r in records if isinstance(r, dict)]

    # Shape B: top-level list
    if isinstance(raw, list):
        return [r for r in raw if isinstance(r, dict)]

    # Shape C: dict of { task_id -> record_dict }
    if isinstance(raw, dict):
        # Heuristic: if every value is a dict, treat as keyed-by-id map
        values = [v for v in raw.values() if not isinstance(v, str)]
        if values and all(isinstance(v, dict) for v in values):
            records = []
            for k, v in raw.items():
                if k.startswith("_"):   # skip _meta, _stats etc.
                    continue
                rec = dict(v)
                # ensure the merge key is set
                if not rec.get(merge_key):
                    if fb_key and rec.get(fb_key):
                        rec[merge_key] = rec[fb_key]
                    else:
                        rec[merge_key] = k
                records.append(rec)
            return records

        # Shape D: single record — wrap it
        if raw.get(merge_key) or raw.get(fb_key):
            rec = dict(raw)
            if not rec.get(merge_key) and fb_key:
                rec[merge_key] = rec.get(fb_key, filepath.stem)
            return [rec]

    return []


def _infer_task_id(record: dict, cfg: dict) -> str:
    """Return a stable string identifier for a task record."""
    mk = cfg["merge_key"]
    fb = cfg.get("fallback_merge_key", "")
    for key in (mk, fb, "task_id", "equation_id", "equation", "name", "domain", "case_id"):
        val = record.get(key)
        if val and str(val) not in ("?", ""):
            return str(val)
    return "unknown"


def _records_to_csv(records: list[dict], path: Path) -> None:
    """Write a flat CSV from a list of dicts (heterogeneous fields OK)."""
    if not records:
        return
    all_keys: list[str] = []
    seen_keys: set[str] = set()
    for r in records:
        for k in r:
            if k not in seen_keys:
                all_keys.append(k)
                seen_keys.add(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


# ─────────────────────────────────────────────────────────────────────────────
#  Per-experiment assembler
# ─────────────────────────────────────────────────────────────────────────────

def assemble_experiment(
    exp_id: str,
    results_dir: Path,
    extra_shard_dirs: list[Path] | None = None,
    verbose: bool = True,
) -> dict:
    """
    Collect all partial shard JSON files for `exp_id`, merge by task_id,
    write <exp>_assembled.json and <exp>_assembled.csv into the result subdir.

    Returns a summary dict.
    """
    cfg = EXP_CONFIG[exp_id]
    subdir    = results_dir / cfg["result_subdir"]
    out_json  = subdir / f"{exp_id}_assembled.json"
    out_csv   = subdir / f"{exp_id}_assembled.csv"

    if verbose:
        sep = "─" * 68
        print(f"\n{sep}")
        print(f"  [{exp_id}]  {cfg['label']}")
        print(f"  subdir : {subdir.relative_to(results_dir)}")
        print(sep)

    # ── Gather search dirs ─────────────────────────────────────────────────
    search_dirs: list[Path] = [subdir]
    if extra_shard_dirs:
        search_dirs.extend(extra_shard_dirs)
    # Also search one level deeper (per-equation sub-dirs, e.g. noise-sweep/)
    for d in list(search_dirs):
        if d.exists():
            for sub in d.iterdir():
                if sub.is_dir():
                    search_dirs.append(sub)

    shard_files = _find_shard_files(search_dirs, cfg["shard_globs"])

    if verbose:
        print(f"  Partial files found: {len(shard_files)}")
        for f in shard_files:
            try:
                rel = f.relative_to(results_dir)
            except ValueError:
                rel = f
            print(f"    · {rel}")

    if not shard_files:
        msg = f"  ⚠  No partial result files found for [{exp_id}] in {subdir}"
        print(msg)
        return {"exp_id": exp_id, "status": "no_files", "n_tasks": 0}

    # ── instability is CSV-only ────────────────────────────────────────────
    if cfg.get("array_key") is None and exp_id == "instability":
        import shutil
        csvs = [f for f in shard_files if f.suffix == ".csv"]
        if csvs:
            assembled_csv = subdir / f"{exp_id}_assembled.csv"
            # Concatenate all CSVs, deduplicate by header
            all_rows: list[dict] = []
            seen_ids: set[str] = set()
            header_written = False
            with open(assembled_csv, "w", newline="", encoding="utf-8") as fout:
                for csv_path in csvs:
                    try:
                        with open(csv_path, newline="", encoding="utf-8") as fin:
                            reader = csv.DictReader(fin)
                            if not header_written and reader.fieldnames:
                                writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
                                writer.writeheader()
                                header_written = True
                            for row in reader:
                                uid = row.get("case_id") or row.get("equation") or str(row)
                                if uid not in seen_ids:
                                    seen_ids.add(uid)
                                    all_rows.append(row)
                                    if header_written:
                                        writer.writerow(row)
                    except Exception as e:
                        print(f"    ⚠  CSV error {csv_path.name}: {e}", file=sys.stderr)
            if verbose:
                print(f"\n  ✅  {len(all_rows)} rows assembled → {assembled_csv.name}")
            return {
                "exp_id": exp_id, "status": "ok",
                "n_tasks": len(all_rows),
                "out_csv": str(assembled_csv),
            }
        print(f"  ⚠  No CSV files found for instability experiment.")
        return {"exp_id": exp_id, "status": "no_files", "n_tasks": 0}

    # ── Extract records from all partial files ─────────────────────────────
    # merged: task_id → record dict  (later file wins on conflict)
    merged: dict[str, dict] = {}
    n_total_records = 0

    for fpath in shard_files:
        records = _extract_records(fpath, cfg)
        n_total_records += len(records)
        for rec in records:
            tid = _infer_task_id(rec, cfg)
            if tid == "unknown":
                # Last resort: use filename stem + index
                tid = f"{fpath.stem}_{len(merged)}"
            # Ensure task_id is always set on the record
            if not rec.get(cfg["merge_key"]):
                rec[cfg["merge_key"]] = tid
            existing = merged.get(tid)
            if existing is None:
                merged[tid] = rec
            else:
                # Merge: update existing with any non-null new fields
                # but preserve fields already present (earlier = canonical)
                for k, v in rec.items():
                    if k not in existing or existing[k] in (None, "", "?"):
                        existing[k] = v

    if verbose:
        print(f"  Records extracted  : {n_total_records} (across all shards)")
        print(f"  Unique task IDs    : {len(merged)}")

    if not merged:
        print(f"  ⚠  No task records could be extracted for [{exp_id}].")
        return {"exp_id": exp_id, "status": "empty", "n_tasks": 0}

    # ── Compute quick stats ────────────────────────────────────────────────
    all_records = list(merged.values())
    n_solved    = sum(1 for r in all_records if r.get("status") == "ok")
    r2_values   = [float(r["r2"]) for r in all_records
                   if r.get("r2") is not None
                   and str(r.get("r2")) not in ("nan", "NaN", "")]
    r2_mean     = (sum(r2_values) / len(r2_values)) if r2_values else None
    r2_ge_99    = sum(1 for v in r2_values if v >= 0.99)
    r2_ge_9999  = sum(1 for v in r2_values if v >= 0.9999)

    stats = {
        "n_tasks":       len(merged),
        "n_solved":      n_solved,
        "solve_rate":    round(n_solved / len(merged), 4) if merged else 0.0,
        "r2_mean":       round(r2_mean, 4) if r2_mean is not None else None,
        "r2_ge_0.99":    r2_ge_99,
        "r2_ge_0.9999":  r2_ge_9999,
        "n_shard_files": len(shard_files),
    }

    # ── Write assembled JSON ───────────────────────────────────────────────
    subdir.mkdir(parents=True, exist_ok=True)
    assembled = {
        "_meta": {
            "exp_id":       exp_id,
            "assembled_at": datetime.now(timezone.utc).isoformat(),
            "result_subdir": cfg["result_subdir"],
            "shard_files":  [str(f) for f in shard_files],
            "merge_key":    cfg["merge_key"],
        },
        "stats":   stats,
        "results": merged,   # keyed by task_id
    }
    out_json.write_text(json.dumps(assembled, indent=2, ensure_ascii=False))

    # ── Write assembled CSV ────────────────────────────────────────────────
    _records_to_csv(all_records, out_csv)

    if verbose:
        print(f"\n  Stats:")
        print(f"    n_tasks     : {stats['n_tasks']}")
        if n_solved:
            print(f"    n_solved    : {n_solved}  ({stats['solve_rate']*100:.1f}%)")
        if r2_mean is not None:
            print(f"    R² mean     : {r2_mean:.4f}")
            print(f"    R² ≥ 0.99   : {r2_ge_99}")
            print(f"    R² ≥ 0.9999 : {r2_ge_9999}  (strict threshold §10.8)")
        print(f"\n  ✅  Written:")
        print(f"    {out_json.name}")
        print(f"    {out_csv.name}")

    return {
        "exp_id":   exp_id,
        "status":   "ok",
        "n_tasks":  len(merged),
        "stats":    stats,
        "out_json": str(out_json),
        "out_csv":  str(out_csv),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compose partial shard JSONs → final per-experiment assembled JSON.\n"
            "Derived from run_all_checkpoint.py v8.1 Step definitions."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--exp",
        metavar="EXP_ID",
        choices=ALL_EXPERIMENTS,
        help=f"Single experiment to assemble. Choices: {', '.join(ALL_EXPERIMENTS)}",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Assemble all experiments.",
    )
    parser.add_argument(
        "--results-dir",
        metavar="PATH",
        default=str(RESULTS_DIR),
        help=f"Root results directory (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--shard-dirs",
        nargs="+",
        metavar="DIR",
        default=[],
        help=(
            "Extra directories to search for shard files "
            "(e.g. downloaded CI artifact folders). "
            "Used in addition to the canonical result_subdir."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-file progress output.",
    )
    args = parser.parse_args()

    results_dir  = Path(args.results_dir).expanduser().resolve()
    shard_dirs   = [Path(d).expanduser().resolve() for d in args.shard_dirs]
    verbose      = not args.quiet

    experiments  = ALL_EXPERIMENTS if args.all else [args.exp]

    print("═" * 68)
    print("  HypatiaX · Shard Assembler")
    print(f"  results_dir : {results_dir}")
    if shard_dirs:
        print(f"  shard_dirs  : {[str(d) for d in shard_dirs]}")
    print("═" * 68)

    summaries = []
    for exp_id in experiments:
        result = assemble_experiment(
            exp_id,
            results_dir=results_dir,
            extra_shard_dirs=shard_dirs if shard_dirs else None,
            verbose=verbose,
        )
        summaries.append(result)

    # ── Final summary table ────────────────────────────────────────────────
    print("\n" + "═" * 68)
    print("  Assembly summary")
    print("═" * 68)
    print(f"  {'Experiment':<22}  {'Status':<10}  {'Tasks':>6}  Output file")
    print("  " + "─" * 62)
    all_ok = True
    for s in summaries:
        status  = s["status"]
        n       = s.get("n_tasks", 0)
        out     = Path(s.get("out_json", s.get("out_csv", "—"))).name if s.get("out_json") or s.get("out_csv") else "—"
        icon    = "✅" if status == "ok" else "⚠ "
        if status != "ok":
            all_ok = False
        print(f"  {icon} {s['exp_id']:<20}  {status:<10}  {n:>6}  {out}")

    print()
    if all_ok:
        print("  ✅  All experiments assembled successfully.")
    else:
        failed = [s["exp_id"] for s in summaries if s["status"] != "ok"]
        print(f"  ⚠  {len(failed)} experiment(s) had no files or errors: {failed}")
        print("     Run the corresponding CI workers first, then re-run this script.")
    print()


if __name__ == "__main__":
    main()

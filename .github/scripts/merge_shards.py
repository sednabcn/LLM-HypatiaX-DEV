#!/usr/bin/env python3
"""
.github/scripts/merge_shards.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HypatiaX  ·  Consolidate per-shard partial JSONs → final experiment result

Called by ci_experiment.yml consolidate job (Job 3):

    python .github/scripts/merge_shards.py \\
        --experiment    "${EXP}" \\
        --input-root    downloaded_artifacts \\
        --output-dir    "${OUT_BASE}/${RESULT_SUBDIR}" \\
        --result-subdir "${RESULT_SUBDIR}"

Source of truth: run_all_checkpoint.py v8.1
  · EXP_CONFIG shard_globs   ← Step.result_glob + Step.post_move destinations
  · merge_key per experiment  ← what each benchmark script writes as task ID
  · result_subdir             ← EXP_RESULT_SUBDIR dict

Writes four canonical output files into --output-dir:
    _merged.json        all task records merged by task_id
    _merged.csv         flat CSV view
    _stats.json         pre-aggregated counts + R² summaries
    _checkpoint.json    provenance / run metadata (consumed by ci_analysis.yml)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
#  Per-experiment configuration
#  Mirrors run_all_checkpoint.py: Step.result_glob, Step.post_move, and
#  EXP_RESULT_SUBDIR; plus the merge_key each benchmark script uses.
# ─────────────────────────────────────────────────────────────────────────────
EXP_CONFIG: dict[str, dict] = {

    # ── exp1: Core DeFi extrapolation (noiseless) ─────────────────────────────
    # CI shard artifact contents (confirmed from CI log):
    #   hypatiax_defi_benchmark_v3_results.json   ← primary result file
    #   protocol_core_noiseless_<timestamp>.json  ← protocol-wrapper
    #   _report.md                                ← skipped (not JSON)
    "exp1": dict(
        result_subdir="comparison_results/noise-noiseless/noiseless",
        shard_globs=[
            "hypatiax_defi_benchmark_v3_results.json",   # exact name from CI
            "hypatiax_defi_benchmark_v3*results*.json",  # any variant
            "hypatiax_defi_benchmark_v3*.json",          # any v3 file
            "protocol_core_noiseless_*.json",
            "defi_v3_*.json",
        ],
        merge_key="task_id",
        fallback_keys=["equation_id", "equation", "name"],
        array_key="results",
    ),

    # ── exp1b: DeFi seed sweep + portfolio variance (noise=15) ───────────────
    # CI shard artifact contents (confirmed from CI log):
    #   comparison_FIXED_<timestamp>.json  ← primary result file
    #   _report.md                         ← skipped (not JSON)
    "exp1b": dict(
        result_subdir="comparison_results/noise-noiseless/15",
        shard_globs=[
            "comparison_FIXED_*.json",           # exact pattern from CI
            "hypatiax_defi_benchmark_v3*.json",  # fallback if script name changes
            "*portfolio*variance*.json",
            "defi_v3_*.json",
        ],
        merge_key="task_id",
        fallback_keys=["equation_id", "equation", "name"],
        array_key="results",
    ),

    # ── exp2_feynman: comparative Feynman suite, LLM+NN only ─────────────────
    # Workers write one JSON per domain via:
    #   run_comparative_suite_benchmark_v2.py --benchmark feynman --domain <D>
    #   --output-dir <RESULT_SUBDIR>
    # File names:  exp2_feynman_checkpoint_shard*.json  (checkpoint)
    #              exp2_feynman_merged.*                 (if script merges internally)
    #              exp2_feynman_stats.json
    #              exp2_all*_checkpoint.json             (in comparison_results/ root)
    # The run_all_checkpoint.py isolated runner also writes per-equation:
    #   I_6_2a.json, II_11_27.json, ... (equation name with dots→underscores)
    #   exp2_results.json  (whole-run consolidated)
    "exp2_feynman": dict(
        result_subdir="comparison_results/feynman-tests/exp2",
        shard_globs=[
            "exp2_feynman_checkpoint_shard*.json",
            "exp2_feynman_merged*.json",
            "exp2_feynman_stats.json",
            "exp2_results.json",
            "exp2_all*_checkpoint.json",  # written to comparison_results/ root by worker
            "I_*.json",
            "II_*.json",
            "III_*.json",
        ],
        merge_key="equation",
        fallback_keys=["name", "task_id", "equation_id"],
        array_key="results",
    ),

    # ── exp2: Combined five-system comparison — all methods ───────────────────
    # Workers write:
    #   exp2_checkpoint_shard*.json
    #   exp2_merged.json / exp2_merged.csv
    #   exp2_stats.json
    "exp2": dict(
        result_subdir="comparison_results/feynman-tests/exp2_multi",
        shard_globs=[
            "exp2_checkpoint_shard*.json",
            "exp2_merged*.json",
            "exp2_stats.json",
        ],
        merge_key="equation",
        fallback_keys=["task_id", "name", "equation_id"],
        array_key="results",
    ),

    # ── exp3: Nguyen-12 SEED=42 ───────────────────────────────────────────────
    # exp3_nguyen12_hybrid50v_02.py --seed 42 writes to RESULTS_DIR root,
    # then post_move copies *nguyen*seed42*.json → extrapolation/
    "exp3": dict(
        result_subdir="extrapolation",
        shard_globs=[
            "*nguyen*seed42*.json",
            "*nguyen12*42*.json",
            "full_run_*.json",
            "report_hybrid_*.json",
            "hybrid_defi_*.json",
        ],
        merge_key="equation",
        fallback_keys=["task_id", "name", "equation_id"],
        array_key="results",
    ),

    # ── exp3b: Nguyen-12 seeds 99/123/777/2024 ────────────────────────────────
    # post_move: *nguyen*.json → extrapolation/multi_seed/
    "exp3b": dict(
        result_subdir="extrapolation/multi_seed",
        shard_globs=[
            "*nguyen*.json",
            "full_run_*.json",
            "report_hybrid_*.json",
            "hybrid_defi_*.json",
        ],
        merge_key="equation",
        fallback_keys=["task_id", "name", "equation_id"],
        array_key="results",
    ),

    # ── suppA: Hybrid-PySR DeFi benchmark ────────────────────────────────────
    # run_hybrid_system_benchmark.py + post_move:
    #   consolidated_hybrid_*.json → hybrid_pysr/defi/
    #   hybrid_system*.json        → hybrid_pysr/defi/
    "suppA": dict(
        result_subdir="hybrid_pysr/defi",
        shard_globs=[
            "consolidated_hybrid_*.json",
            "hybrid_system*.json",
            "hybrid_llm_nn_all_domains_*.json",
            "ablation_exp1_*.json",
        ],
        merge_key="domain",
        fallback_keys=["task_id", "equation", "name"],
        array_key="results",
    ),

    # ── suppB: Noise sweep σ ∈ {0, 0.5, 1, 5, 10}% × 30 equations ───────────
    # run_noise_sweep_benchmark.py writes noise_sweep_*.json
    # task_id format: "noise{σ}__{feynman_id}"  e.g. "noise5.0__I.6.20"
    "suppB": dict(
        result_subdir="comparison_results/feynman-tests/noise-sweep",
        shard_globs=[
            "noise_sweep_*.json",
            "suppB_*.json",
        ],
        merge_key="task_id",
        fallback_keys=["equation", "name", "equation_id"],
        array_key="results",
    ),

    # ── suppB_sc: Sample-complexity sweep n ∈ {50…1000} × 30 equations ───────
    # run_sample_complexity_benchmark.py writes sample_complexity_*.json
    # post_move (checkpoint): moves from feynman-tests/ root → sample-complexity/
    #   recursive=True, exclude="sample-complexity"
    # task_id format: "sc_n{n}__{feynman_id}"  e.g. "sc_n200__I.6.20"
    "suppB_sc": dict(
        result_subdir="comparison_results/feynman-tests/sample-complexity",
        shard_globs=[
            "sample_complexity_*.json",
            "sample_complexity_*.csv",   # post_move may also land CSV shards here
        ],
        merge_key="task_id",
        fallback_keys=["equation", "name", "equation_id"],
        array_key="results",
    ),

    # ── hybrid_all_domains: LLM+NN all-domains one-shot ──────────────────────
    # hybrid_system_llm_nn_all_domains.py --domains <subset>
    # writes hybrid_llm_nn_all_domains_*.json per domain
    "hybrid_all_domains": dict(
        result_subdir="hybrid_llm_nn/all_domains",
        shard_globs=[
            "hybrid_llm_nn_all_domains_*.json",
        ],
        merge_key="domain",
        fallback_keys=["task_id", "equation", "name"],
        array_key="results",
    ),

    # ── exp1_ablation: paired pysr_only vs hypatia ablation (manual-only) ────
    # NOT dispatched by ci_experiment.yml or ci_schedule_all.yml.
    # Kept here to support manual standalone runs alongside exp2_feynman.
    # Uses the same ablation schema (hypatia/pysr_only keys, extrap_r2_far);
    # routes to analyse_ablation() in run_analysis.py.
    # If promoted to CI, entries are also needed in ci_experiment.yml and
    # ci_analysis.yml (result_subdir mapping + dispatch menu).
    "exp1_ablation": dict(
        result_subdir="comparison_results/feynman-tests/exp1_ablation",
        shard_globs=[
            "exp1_ablation_checkpoint_shard*.json",
            "exp1_ablation_merged*.json",
            "exp1_ablation_results.json",
            "exp1_ablation_stats.json",
        ],
        merge_key="equation",
        fallback_keys=["equation_name", "equation_id", "task_id", "name"],
        array_key="results",
    ),

    # ── extrap: OOD extrapolation comparative suite ───────────────────────────
    # run_comparative_suite_benchmark_v2.py --extrap writes:
    #   all_domains_extrap_v4_*.json  → comparison_results/extrapolation/
    #   standalone_llm_nn_*.json      → standalone_llm_nn/
    #   standalone_real_methods_*.json
    "extrap": dict(
        result_subdir="comparison_results/extrapolation",
        shard_globs=[
            "all_domains_extrap_v4_*.json",
            "standalone_llm_nn_*.json",
            "standalone_real_methods_*.json",
        ],
        merge_key="equation",
        fallback_keys=["task_id", "name", "equation_id"],
        array_key="results",
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_shard_files(root: Path, globs: list[str]) -> list[Path]:
    """
    Collect all JSON files under `root` (recursively) matching any glob.
    Skips:
      · _report.md and any non-.json files (they are never result data)
      · our own output files (_merged.json etc.)
      · worker checkpoint/stub files
    Each unique Path is returned once regardless of how many globs match it.
    """
    found: list[Path] = []
    seen:  set[Path]  = set()

    # Always skip these by name — they appear in every shard artifact dir
    _SKIP_NAMES = frozenset({
        "_report.md", "_merged.json", "_merged.csv",
        "_stats.json", "_checkpoint.json",
    })

    for pattern in globs:
        for match in sorted(root.rglob(pattern)):
            if not match.is_file():
                continue
            if match in seen:
                continue
            if match.name in _SKIP_NAMES:
                continue
            if match.suffix.lower() != ".json":
                continue
            if "_assembled" in match.name:
                continue
            seen.add(match)
            found.append(match)

    return found


def _is_stub(raw: object) -> bool:
    """True for {"_meta": {"stub": true}} written by FIX-G6."""
    return (
        isinstance(raw, dict)
        and isinstance(raw.get("_meta"), dict)
        and raw["_meta"].get("stub") is True
    )


def _is_worker_checkpoint(raw: object) -> bool:
    """True for checkpoint_worker_shard*.json files (task tracking, not results)."""
    return isinstance(raw, dict) and "completed" in raw and "run_id_map" in raw


def _collect_run_id_map(shard_files: list[Path]) -> dict[str, str]:
    """
    Scan all shard files and collect the merged run_id_map from any worker
    checkpoint files (which are otherwise skipped by _extract_records).

    run_id_map is written by the worker step 'Set consolidate outputs' as:
        {"task_id": "stable_run_id", ...}
    and lets _enrich_equation_id patch task_id from the stable run identifier
    so records are consistently keyed even when a shard re-runs under a new
    GitHub run_id.
    """
    merged_map: dict[str, str] = {}
    for fpath in shard_files:
        try:
            raw = json.loads(fpath.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        if _is_worker_checkpoint(raw):
            run_id_map = raw.get("run_id_map", {})
            if isinstance(run_id_map, dict):
                merged_map.update(run_id_map)
    return merged_map


def _extract_records(filepath: Path, cfg: dict) -> list[dict]:
    """
    Load one partial result file and return a flat list of task record dicts.

    Handles the four shapes the HypatiaX benchmark scripts produce:

    A.  {"results": [ {...}, ... ]}          wrapper dict with list under array_key
    B.  [ {...}, ... ]                       top-level list
    C.  {"task_id": {...}, "task_id2": {...}} dict keyed by task identifiers
    D.  { single record }                    one task record as a bare dict
    """
    try:
        raw = json.loads(filepath.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        print(f"  ⚠  JSON error in {filepath.name}: {exc}", file=sys.stderr)
        return []

    if _is_stub(raw) or _is_worker_checkpoint(raw):
        return []

    array_key    = cfg.get("array_key")
    merge_key    = cfg["merge_key"]
    fallback_keys = cfg.get("fallback_keys", [])

    # Shape A
    if array_key and isinstance(raw, dict) and isinstance(raw.get(array_key), list):
        return [r for r in raw[array_key] if isinstance(r, dict)]

    # Shape B
    if isinstance(raw, list):
        return [r for r in raw if isinstance(r, dict)]

    if isinstance(raw, dict):
        values = [v for k, v in raw.items() if not k.startswith("_")]

        # Shape C — every non-meta value is a dict  → dict-of-tasks
        if values and all(isinstance(v, dict) for v in values):
            records: list[dict] = []
            for k, v in raw.items():
                if k.startswith("_"):
                    continue
                rec = dict(v)
                # Ensure merge_key is populated
                if not rec.get(merge_key):
                    for fb in fallback_keys:
                        if rec.get(fb):
                            rec[merge_key] = str(rec[fb])
                            break
                    else:
                        rec[merge_key] = k
                records.append(rec)
            return records

        # Shape D — single task record
        has_id = raw.get(merge_key) or any(raw.get(k) for k in fallback_keys)
        if has_id:
            rec = dict(raw)
            if not rec.get(merge_key):
                for fb in fallback_keys:
                    if rec.get(fb):
                        rec[merge_key] = str(rec[fb])
                        break
            return [rec]

    return []


def _task_id(record: dict, cfg: dict) -> str:
    """Return the stable unique string for a task record."""
    for key in [cfg["merge_key"]] + cfg.get("fallback_keys", []) + ["task_id", "name"]:
        val = record.get(key)
        if val and str(val) not in ("", "?"):
            return str(val)
    return "unknown"


def _compute_stats(merged: dict[str, dict]) -> dict:
    records = list(merged.values())
    n       = len(records)
    n_ok    = sum(1 for r in records if r.get("status") == "ok")
    r2_vals = []
    for r in records:
        raw = r.get("r2") or r.get("r2_score")
        if raw is None:
            continue
        try:
            v = float(raw)
            if not (v != v):  # NaN check
                r2_vals.append(v)
        except (TypeError, ValueError):
            pass

    return {
        "n_tasks":         n,
        "n_solved":        n_ok,
        "solve_rate":      round(n_ok / n, 4) if n else 0.0,
        "r2_mean":         round(sum(r2_vals) / len(r2_vals), 4) if r2_vals else None,
        "r2_median":       round(sorted(r2_vals)[len(r2_vals) // 2], 4) if r2_vals else None,
        "r2_ge_0_99":      sum(1 for v in r2_vals if v >= 0.99),
        "r2_ge_0_9999":    sum(1 for v in r2_vals if v >= 0.9999),
        "n_with_r2":       len(r2_vals),
    }


def _write_csv(records: list[dict], path: Path) -> None:
    if not records:
        return
    all_keys: list[str] = []
    seen: set[str] = set()
    for r in records:
        for k in r:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


# ─────────────────────────────────────────────────────────────────────────────
#  instability: CSV-only path
# ─────────────────────────────────────────────────────────────────────────────

def _merge_instability_csvs(shard_files: list[Path], out_dir: Path) -> dict:
    """
    Concatenate instability_analysis.csv shards, deduplicate by case_id.
    Writes _merged.csv, _stats.json, _checkpoint.json (no _merged.json).
    """
    all_rows: list[dict] = []
    seen_ids: set[str] = set()
    fieldnames: list[str] = []

    for csv_path in shard_files:
        if csv_path.suffix.lower() != ".csv":
            continue
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if not fieldnames and reader.fieldnames:
                    fieldnames = list(reader.fieldnames)
                for row in reader:
                    uid = row.get("case_id") or row.get("equation") or str(row)
                    if uid not in seen_ids:
                        seen_ids.add(uid)
                        all_rows.append(row)
        except Exception as exc:
            print(f"  ⚠  CSV error {csv_path.name}: {exc}", file=sys.stderr)

    if fieldnames and all_rows:
        with open(out_dir / "_merged.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)

    stats = {"n_tasks": len(all_rows), "n_shard_files": len(shard_files)}
    (out_dir / "_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


# ─────────────────────────────────────────────────────────────────────────────
#  Core merge
# ─────────────────────────────────────────────────────────────────────────────

def merge_experiment(
    exp_id: str,
    input_root: Path,
    output_dir: Path,
    result_subdir: str,
    run_id: str = "",
    verbose: bool = True,
) -> int:
    """
    Merge all per-shard partial JSONs for `exp_id` found under `input_root`
    into the four canonical output files in `output_dir`.

    Returns 0 on success, 1 on error.
    """
    cfg = EXP_CONFIG.get(exp_id)
    if cfg is None:
        print(f"ERROR: unknown experiment '{exp_id}'", file=sys.stderr)
        print(f"  Known: {', '.join(sorted(EXP_CONFIG))}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'═'*68}")
        print(f"  merge_shards · [{exp_id}]")
        print(f"  input_root  : {input_root}")
        print(f"  output_dir  : {output_dir}")
        print(f"  subdir      : {result_subdir}")
        print(f"{'═'*68}")

    # ── 1. Find all partial shard files ────────────────────────────────────
    shard_files = _find_shard_files(input_root, cfg["shard_globs"])

    if verbose:
        print(f"\n  Shard files found: {len(shard_files)}")
        for f in shard_files:
            try:
                rel = f.relative_to(input_root)
            except ValueError:
                rel = f
            print(f"    · {rel}")

    if not shard_files:
        print(f"  ERROR: no partial result files found under {input_root}", file=sys.stderr)
        print(f"         globs tried: {cfg['shard_globs']}", file=sys.stderr)
        _write_stub_checkpoint(output_dir, exp_id, result_subdir, run_id,
                               error="no_shard_files")
        return 1

    # ── 2. instability is CSV-only ──────────────────────────────────────────
    if cfg.get("array_key") is None:
        stats = _merge_instability_csvs(shard_files, output_dir)
        _write_checkpoint(output_dir, exp_id, result_subdir, run_id,
                          shard_files, stats, n_merged=stats["n_tasks"])
        if verbose:
            print(f"\n  ✅  instability: {stats['n_tasks']} rows assembled")
        return 0

    # ── 3. Extract + merge records ──────────────────────────────────────────
    merged: dict[str, dict] = {}
    total_raw = 0

    for fpath in shard_files:
        records = _extract_records(fpath, cfg)
        total_raw += len(records)
        for rec in records:
            tid = _task_id(rec, cfg)
            if tid == "unknown":
                tid = f"{fpath.stem}_{len(merged)}"
            # Set the merge key on the record if absent
            if not rec.get(cfg["merge_key"]):
                rec[cfg["merge_key"]] = tid
            existing = merged.get(tid)
            if existing is None:
                merged[tid] = rec
            else:
                # First-write-wins for each field; fill blanks from later shards
                for k, v in rec.items():
                    if k not in existing or existing[k] in (None, "", "?"):
                        existing[k] = v

    if verbose:
        print(f"\n  Raw records extracted : {total_raw}")
        print(f"  Unique task IDs       : {len(merged)}")

    if not merged:
        print("  ERROR: no task records could be extracted.", file=sys.stderr)
        _write_stub_checkpoint(output_dir, exp_id, result_subdir, run_id,
                               error="no_records_extracted")
        return 1

    # ── 4. Enrich equation_id (and task_id via run_id_map) ─────────────────
    run_id_map = _collect_run_id_map(shard_files)
    _enrich_equation_id(merged, run_id_map=run_id_map)

    # ── 5. Compute stats ────────────────────────────────────────────────────
    stats = _compute_stats(merged)
    if verbose:
        print(f"\n  Stats:")
        print(f"    n_tasks     : {stats['n_tasks']}")
        print(f"    n_solved    : {stats['n_solved']}  "
              f"({stats['solve_rate']*100:.1f}%)")
        if stats["r2_mean"] is not None:
            print(f"    R² mean     : {stats['r2_mean']:.4f}")
            print(f"    R² ≥ 0.9999 : {stats['r2_ge_0_9999']}  "
                  f"(strict §10.8 threshold)")

    # ── 6. Write _merged.json ───────────────────────────────────────────────
    merged_json_path = output_dir / "_merged.json"
    merged_json_path.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ── 7. Write _merged.csv ────────────────────────────────────────────────
    merged_csv_path = output_dir / "_merged.csv"
    _write_csv(list(merged.values()), merged_csv_path)

    # ── 8. Write _stats.json ────────────────────────────────────────────────
    stats_path = output_dir / "_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # ── 9. Write _checkpoint.json ───────────────────────────────────────────
    _write_checkpoint(output_dir, exp_id, result_subdir, run_id,
                      shard_files, stats, n_merged=len(merged))

    if verbose:
        print(f"\n  ✅  Written to {output_dir}:")
        for name in ("_merged.json", "_merged.csv", "_stats.json", "_checkpoint.json"):
            size = (output_dir / name).stat().st_size
            print(f"    {name:<22}  {size:>8,} bytes")

    return 0


def _enrich_equation_id(merged: dict[str, dict], run_id_map: dict[str, str] | None = None) -> None:
    """
    Patch every record so equation_id is set, and backfill task_id from
    run_id_map when present.

    Priority (mirrors ci_experiment.yml consolidate 'Enrich _merged.json' step):
      1. Already set and not "?"
      2. equation_id / eq_id / equation inside any per-method sub-record
      3. Parse 'description' field before first separator (: — - |)
      4. Top-level dict key (= task_id)

    task_id patching (mirrors the Enrich step's run_id_map cross-reference):
      · If run_id_map is provided and the top-level key is present in it,
        set task_id to the stable run identifier from the map so records
        remain consistently keyed across re-runs with different GitHub run_ids.
    """
    if run_id_map is None:
        run_id_map = {}
    for top_key, record in merged.items():
        if not isinstance(record, dict):
            continue
        if record.get("equation_id") and record["equation_id"] != "?":
            continue

        eq_id = None

        # Priority 2
        for v in record.values():
            if isinstance(v, dict):
                candidate = (v.get("equation_id") or v.get("eq_id")
                             or v.get("equation"))
                if candidate and candidate != "?":
                    eq_id = str(candidate)
                    break

        # Priority 3
        if not eq_id:
            desc = record.get("description", "")
            if desc:
                for sep in (":", "—", " - ", "|"):
                    if sep in desc:
                        eq_id = desc.split(sep)[0].strip()
                        break
                if not eq_id:
                    eq_id = desc.strip()

        # Priority 4
        if not eq_id:
            eq_id = top_key if top_key not in ("", "?") else None

        if eq_id:
            record["equation_id"] = eq_id
            if not record.get("task_id") or record["task_id"] == "?":
                # Prefer the stable run_id_map identifier over the raw top-level key.
                record["task_id"] = run_id_map.get(top_key, top_key)


def _write_checkpoint(
    out_dir: Path,
    exp_id: str,
    result_subdir: str,
    run_id: str,
    shard_files: list[Path],
    stats: dict,
    n_merged: int,
) -> None:
    """
    Write _checkpoint.json consumed by ci_analysis.yml via workflow_run event.
    Fields match what ci_experiment.yml's 'Set consolidate outputs' step emits.
    """
    checkpoint = {
        "exp_id":         exp_id,
        "result_subdir":  result_subdir,
        "run_id":         run_id,
        "merged_at":      datetime.now(timezone.utc).isoformat(),
        "n_merged":       n_merged,
        "stats":          stats,
        "shard_files":    [str(f) for f in shard_files],
        "n_shard_files":  len(shard_files),
    }
    (out_dir / "_checkpoint.json").write_text(
        json.dumps(checkpoint, indent=2), encoding="utf-8"
    )


def _write_stub_checkpoint(
    out_dir: Path,
    exp_id: str,
    result_subdir: str,
    run_id: str,
    error: str,
) -> None:
    """
    Write a minimal _checkpoint.json even on failure (mirrors FIX-G6 stub pattern)
    so actions/cache/save never fails on a missing path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "_meta":         {"stub": True},
        "exp_id":        exp_id,
        "result_subdir": result_subdir,
        "run_id":        run_id,
        "merged_at":     datetime.now(timezone.utc).isoformat(),
        "n_merged":      0,
        "error":         error,
    }
    (out_dir / "_checkpoint.json").write_text(
        json.dumps(checkpoint, indent=2), encoding="utf-8"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CLI  — matches the exact invocation in ci_experiment.yml consolidate job:
#
#    python .github/scripts/merge_shards.py \
#        --experiment    "${EXP}" \
#        --input-root    downloaded_artifacts \
#        --output-dir    "${OUT_BASE}/${RESULT_SUBDIR}" \
#        --result-subdir "${RESULT_SUBDIR}"
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge HypatiaX per-shard partial JSONs → 4 canonical output files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--experiment", "-e",
        required=True,
        metavar="EXP_ID",
        choices=sorted(EXP_CONFIG),
        help=f"Experiment ID. One of: {', '.join(sorted(EXP_CONFIG))}",
    )
    parser.add_argument(
        "--input-root", "-i",
        required=True,
        metavar="DIR",
        help=(
            "Root directory that contains the downloaded shard artifact folders. "
            "Searched recursively for files matching each experiment's shard globs."
        ),
    )
    parser.add_argument(
        "--output-dir", "-o",
        required=True,
        metavar="DIR",
        help=(
            "Directory where _merged.json, _merged.csv, _stats.json and "
            "_checkpoint.json are written. Created if absent."
        ),
    )
    parser.add_argument(
        "--result-subdir",
        required=True,
        metavar="SUBDIR",
        help=(
            "Relative result subdir (e.g. 'comparison_results/feynman-tests/exp2'). "
            "Embedded in _checkpoint.json for ci_analysis.yml."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=os.environ.get("GITHUB_RUN_ID", ""),
        metavar="ID",
        help="GitHub Actions run_id (written to _checkpoint.json). "
             "Defaults to $GITHUB_RUN_ID.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output.",
    )

    args = parser.parse_args()

    rc = merge_experiment(
        exp_id        = args.experiment,
        input_root    = Path(args.input_root).expanduser().resolve(),
        output_dir    = Path(args.output_dir).expanduser().resolve(),
        result_subdir = args.result_subdir,
        run_id        = args.run_id,
        verbose       = not args.quiet,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()

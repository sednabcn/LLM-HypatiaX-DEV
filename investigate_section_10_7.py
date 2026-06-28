#!/usr/bin/env python3
"""
investigate_section_10_7.py
============================

Purpose
-------
Resolve the §10.7 "9/30 vs 106/180" denominator-mismatch question by:

  1. Loading all `protocol_core_noiseless_pca_*.json` files and filtering
     to the HypatiaX method only (EnhancedHybridSystemDeFi), to compute the
     real single-method success count out of 30 equations under the
     PCA 40/60 split — the number that should replace "9/30" in §10.7.

  2. Cross-checking that count against exp2_pca_4060_summary.json's
     n_pass / n_total fields, to determine whether 106/180 actually is
     (or isn't) the all-6-methods pooled count, or whether it's something
     else entirely.

  3. Diffing every duplicate copy of fixc3_baseline.json and
     split_protocol_disclosure.json found in the repo tree, since multiple
     copies of the same filename living in different directories is a
     known failure mode in this repo (see _merged.json vs
     hypatiax_defi_benchmark_pca_results.json disagreeing on 3/74 cases
     earlier in this investigation).

Usage
-----
    python3 investigate_section_10_7.py [SEARCH_DIR]

If SEARCH_DIR is omitted, defaults to /mnt/user-data/uploads.
The script recursively searches SEARCH_DIR for the relevant filenames,
so it doesn't matter whether you upload them flat or preserve the
original directory structure from tree_r.txt.

Output
------
Prints a structured report to stdout AND writes it to
investigate_section_10_7_report.json next to this script for archiving.
"""

import json
import math
import sys
import hashlib
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config — adjust here if naming conventions differ from what's been seen
# so far in this investigation.
# ---------------------------------------------------------------------------

HYPATIAX_METHOD_CANDIDATES = {
    "HybridDiscoverySystem": "HybridDiscoverySystem v50_2 (tools)",
    "EnhancedHybridSystemDeFi": "EnhancedHybridSystemDeFi (core)",
}

ALL_SIX_METHODS_RAW = [
    "PureLLM Baseline (core)",
    "ImprovedNN (core)",
    "EnhancedHybridSystemDeFi (core)",
    "HybridSystemLLMNN all-domains (core)",
    "SymbolicEngineWithLLM (tools)",
    "HybridDiscoverySystem v50_2 (tools)",
]

PROTOCOL_GLOB = "protocol_core_noiseless_pca_*.json"
SUMMARY_NAME = "exp2_pca_4060_summary.json"
DUP_WATCH_NAMES = ["fixc3_baseline.json", "split_protocol_disclosure.json"]

# ---------------------------------------------------------------------------
# Exact relative paths, as reconstructed from tree_r.txt.
# These are the canonical locations the script will check FIRST (and report
# explicitly as found/missing), before falling back to a recursive filename
# search anywhere else under SEARCH_DIR. This matters because several of
# these filenames exist in MORE THAN ONE directory in the real repo, with
# possibly divergent content (see DUP_WATCH_NAMES groups below) — checking
# the exact expected path first means the report tells you specifically
# whether the canonical copy was uploaded, vs. some other stray copy.
# ---------------------------------------------------------------------------

EXPECTED_PROTOCOL_PATHS = [
    "comparison_results/feynman-tests/exp2/protocol_core_noiseless_pca_20260604_131820.json",
    "comparison_results/feynman-tests/exp2/protocol_core_noiseless_pca_20260604_132102.json",
    "comparison_results/feynman-tests/exp2/protocol_core_noiseless_pca_20260604_132207.json",
    "comparison_results/feynman-tests/exp2/protocol_core_noiseless_pca_20260604_132722.json",
    "comparison_results/feynman-tests/exp2/protocol_core_noiseless_pca_20260604_133021.json",
    "comparison_results/feynman-tests/exp2/protocol_core_noiseless_pca_20260604_133137.json",
    "comparison_results/feynman-tests/exp2/protocol_core_noiseless_pca_20260604_133648.json",
    "comparison_results/feynman-tests/exp2/protocol_core_noiseless_pca_20260604_134004.json",
    "comparison_results/feynman-tests/exp2/protocol_core_noiseless_pca_20260604_134112.json",
    "comparison_results/feynman-tests/exp2/protocol_core_noiseless_pca_20260604_134723.json",
    "comparison_results/feynman-tests/exp2/protocol_core_noiseless_pca_20260604_135159.json",
]

EXPECTED_SUMMARY_PATH = "comparison_results/feynman-tests/exp2_pca_4060/exp2_pca_4060_summary.json"

# Every known on-disk location of each duplicate-prone filename, per tree_r.txt.
EXPECTED_DUPLICATE_PATHS = {
    "fixc3_baseline.json": [
        "ablation/exp1_ablation/fixc3_baseline.json",
        "comparison_results/feynman-tests/noise-sweep/fixc3_baseline.json",
        "comparison_results/feynman-tests/sample-complexity/fixc3_baseline.json",
        "figures/fixc3_baseline.json",
        "fixc3_baseline.json",
    ],
    "split_protocol_disclosure.json": [
        "comparison_results/feynman-tests/exp2_pca_4060/split_protocol_disclosure.json",
        "comparison_results/noise-noiseless/15_pca/split_protocol_disclosure.json",
        "comparison_results/noise-noiseless/noiseless/defi_pca/split_protocol_disclosure.json",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def safe_load_json(path: Path):
    """Load JSON, tolerating NaN/Infinity literals like the rest of this
    investigation's files have used (Python's json module accepts these
    by default via allow_nan, but some files may have non-standard tokens)."""
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  [WARN] Failed to parse {path}: {e}")
        return None


def resolve_canonical_or_fallback(search_dir: Path, relative_path: str, filename: str):
    """
    Look for a file at its exact expected relative path under search_dir
    first (preserving the directory structure from tree_r.txt, e.g. if the
    user uploaded a zip / preserved folder tree). If not found there, fall
    back to a recursive search by filename anywhere under search_dir (e.g.
    if the user just flat-uploaded the files via chat attachment, which
    flattens directory structure).

    Returns (path_or_None, matched_canonical: bool)
    """
    canonical = search_dir / relative_path
    if canonical.is_file():
        return canonical, True

    # Fallback: flat upload, search by filename anywhere
    matches = sorted(search_dir.rglob(filename))
    if matches:
        return matches[0], False
    return None, False


def report_path_fidelity(search_dir: Path):
    """
    Explicitly report, for every canonical path we expect, whether it was
    found at its exact expected location, found elsewhere (flat upload),
    or missing entirely. Returns the resolved file objects needed by the
    rest of the script.
    """
    print("=" * 78)
    print("STEP 0 — Resolving expected files against canonical paths (tree_r.txt)")
    print("=" * 78)

    resolved_protocol_files = []
    for rel in EXPECTED_PROTOCOL_PATHS:
        filename = rel.rsplit("/", 1)[-1]
        path, exact = resolve_canonical_or_fallback(search_dir, rel, filename)
        if path is None:
            print(f"  [MISSING]      {rel}")
        elif exact:
            print(f"  [OK canonical] {rel}")
        else:
            print(f"  [OK flat]      {filename}  (found at {path}, not at canonical path)")
        if path is not None:
            resolved_protocol_files.append(path)

    summary_filename = EXPECTED_SUMMARY_PATH.rsplit("/", 1)[-1]
    summary_path, summary_exact = resolve_canonical_or_fallback(
        search_dir, EXPECTED_SUMMARY_PATH, summary_filename
    )
    if summary_path is None:
        print(f"  [MISSING]      {EXPECTED_SUMMARY_PATH}")
    elif summary_exact:
        print(f"  [OK canonical] {EXPECTED_SUMMARY_PATH}")
    else:
        print(f"  [OK flat]      {summary_filename}  (found at {summary_path}, not at canonical path)")

    print()
    return resolved_protocol_files, summary_path


def report_duplicate_path_fidelity(search_dir: Path):
    """
    For each duplicate-prone filename, check every known canonical location
    from tree_r.txt explicitly, in addition to any other copies found via
    recursive search (in case there are even more copies than tree_r.txt
    revealed, e.g. if the tree dump was itself incomplete/stale).
    """
    dup_groups = {}
    for filename, canonical_rels in EXPECTED_DUPLICATE_PATHS.items():
        found_paths = []
        print(f"  Canonical locations for {filename}:")
        for rel in canonical_rels:
            p = search_dir / rel
            if p.is_file():
                print(f"    [OK canonical] {rel}")
                found_paths.append(p)
            else:
                print(f"    [MISSING]      {rel}")

        # Also catch any copies NOT in the canonical list (flat uploads, or
        # locations tree_r.txt didn't capture)
        all_matches = sorted(search_dir.rglob(filename))
        extra = [p for p in all_matches if p not in found_paths]
        for p in extra:
            print(f"    [EXTRA/FLAT]   {p}  (not in canonical list)")
            found_paths.append(p)

        dup_groups[filename] = found_paths
    return dup_groups



    """Load JSON, tolerating NaN/Infinity literals like the rest of this
    investigation's files have used (Python's json module accepts these
    by default via allow_nan, but some files may have non-standard tokens)."""
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  [WARN] Failed to parse {path}: {e}")
        return None


def find_records(obj):
    """
    Real schema confirmed from protocol_core_noiseless_pca_20260604_131820.json:
        {
          "timestamp": ..., "script": ..., "protocol": {...},
          "total_tests": int,
          "methods": [ ... 6 full method name strings ... ],
          "tests": [
            {
              "description": "<equation name>",
              "domain": "...",
              "results": {
                 "<full method name>": {"method":..., "success": bool, "r2":..., "error":...},
                 ...
              }
            },
            ...
          ]
        }
    Each file is a SHARD covering only a few equations (e.g. 3 here), not all 30.
    """
    if isinstance(obj, dict) and isinstance(obj.get("tests"), list):
        return obj["tests"]
    # fallbacks for other shapes seen elsewhere in this investigation
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("results", "records", "data", "rows"):
            if key in obj and isinstance(obj[key], list):
                return obj[key]
        if all(isinstance(v, dict) for v in obj.values()):
            return list(obj.values())
    return []


def extract_method_records(record):
    """
    For the real schema, each test record's 'results' dict IS already
    method_full_name -> result_dict. Keep full names (with " (core)"/" (tools)"
    suffixes) as keys so HypatiaX candidate matching can be done explicitly
    against HYPATIAX_METHOD_CANDIDATES rather than guessing.
    """
    if not isinstance(record, dict):
        return {}

    if "results" in record and isinstance(record["results"], dict):
        return record["results"]

    method = record.get("method") or record.get("method_name") or record.get("system")
    if method:
        return {method: record}

    if any(k in ALL_SIX_METHODS_RAW for k in record.keys()):
        return {k: v for k, v in record.items() if k in ALL_SIX_METHODS_RAW}

    return {}


def get_equation_id(record):
    return (
        record.get("description")
        or record.get("equation_id")
        or record.get("name")
        or record.get("equation")
        or record.get("id")
    )


def is_success(method_result):
    if not isinstance(method_result, dict):
        return False
    val = method_result.get("success")
    if isinstance(val, bool):
        return val
    r2 = method_result.get("r2", method_result.get("train_r2"))
    if isinstance(r2, (int, float)) and not (isinstance(r2, float) and math.isnan(r2)):
        return r2 > 0.0
    return False


def is_env_failure(method_result):
    """Distinguish 'not available' / import-probe failures from genuine
    accuracy failures, since these mean completely different things for
    the §10.7 conclusion (see hybrid_system_v50_2.py + exp1_ablation_run.log
    investigation earlier — same root cause as the env-probe bug)."""
    if not isinstance(method_result, dict):
        return False
    err = method_result.get("error")
    if not err:
        return False
    err_lower = str(err).lower()
    return "not available" in err_lower or "import" in err_lower or "probe" in err_lower


# ---------------------------------------------------------------------------
# Main investigation steps
# ---------------------------------------------------------------------------

def step1_recount_hypatiax_only(protocol_files):
    print("=" * 78)
    print("STEP 1 — Recount across protocol_core_noiseless_pca_*.json (real schema)")
    print("=" * 78)

    if not protocol_files:
        print("  [MISSING] No protocol_core_noiseless_pca_*.json files found.")
        return None

    print(f"  Found {len(protocol_files)} file(s):")
    for f in protocol_files:
        print(f"    - {f}")

    # Track BOTH HypatiaX candidates separately, plus all 6 methods pooled.
    candidate_eq_results = {name: {} for name in HYPATIAX_METHOD_CANDIDATES}
    candidate_env_failures = {name: 0 for name in HYPATIAX_METHOD_CANDIDATES}
    method_pool_counts = defaultdict(lambda: defaultdict(int))
    total_rows_seen = 0
    seen_equations_per_file = {}

    for f in protocol_files:
        obj = safe_load_json(f)
        if obj is None:
            continue
        records = find_records(obj)
        n_in_file = len(records)
        seen_equations_per_file[f.name] = n_in_file
        print(f"\n  --- {f.name}: {n_in_file} test(s) in this shard ---")

        for rec in records:
            eq_id = get_equation_id(rec)
            method_map = extract_method_records(rec)
            if not method_map:
                continue
            print(f"      {eq_id!r}")

            for method_name, mres in method_map.items():
                total_rows_seen += 1
                succ = is_success(mres)
                envfail = is_env_failure(mres)
                method_pool_counts[method_name]["success" if succ else "fail"] += 1
                tag = "OK" if succ else ("ENV-FAIL" if envfail else "FAIL")
                print(f"          {method_name:45s} {tag:9s} r2={mres.get('r2')}")

                for cand_short, cand_full in HYPATIAX_METHOD_CANDIDATES.items():
                    if method_name == cand_full:
                        if eq_id in candidate_eq_results[cand_short]:
                            print(f"          [WARN] duplicate equation_id {eq_id!r} "
                                  f"for candidate {cand_short} — overwriting")
                        candidate_eq_results[cand_short][eq_id] = succ
                        if envfail:
                            candidate_env_failures[cand_short] += 1

    print(f"\n  Total (equation, method) rows seen: {total_rows_seen}")
    print("\n  Per-method pooled success/fail counts (sanity check vs N_equations x 6):")
    for method, counts in sorted(method_pool_counts.items()):
        s, fa = counts.get("success", 0), counts.get("fail", 0)
        print(f"    {method:45s} success={s:3d}  fail={fa:3d}  total={s+fa:3d}")

    print("\n  --- HypatiaX candidate comparison ---")
    summary_per_candidate = {}
    for cand_short, results in candidate_eq_results.items():
        n_pass = sum(1 for v in results.values() if v)
        n_total = len(results)
        envfails = candidate_env_failures[cand_short]
        rate = (n_pass / n_total) if n_total else None
        print(f"  Candidate: {cand_short} ({HYPATIAX_METHOD_CANDIDATES[cand_short]})")
        print(f"    -> {n_pass}/{n_total} = {rate}" if n_total else "    -> no rows found")
        if envfails:
            print(f"    [ALERT] {envfails}/{n_total} of these are ENVIRONMENT FAILURES "
                  f"('not available'/import errors), not genuine accuracy failures.")
            print(f"    -> These rows cannot be counted as a real performance result "
                  f"until the underlying env/import issue is fixed (see "
                  f"hybrid_system_v50_2.py probe investigation).")
        summary_per_candidate[cand_short] = {
            "n_pass": n_pass, "n_total": n_total, "rate": rate,
            "env_failures": envfails,
        }

    if not protocol_files:
        return None

    return {
        "method_pool_counts": {k: dict(v) for k, v in method_pool_counts.items()},
        "candidates": summary_per_candidate,
        "candidate_equation_results": candidate_eq_results,
        "shard_sizes": seen_equations_per_file,
        # backward-compat top-level fields point at whichever candidate has
        # the most total rows (best guess until confirmed by the user)
        "n_pass": max(summary_per_candidate.values(), key=lambda v: v["n_total"])["n_pass"]
        if summary_per_candidate else 0,
        "n_total": max(summary_per_candidate.values(), key=lambda v: v["n_total"])["n_total"]
        if summary_per_candidate else 0,
    }


def step2_crosscheck_summary(summary_file, recount_result):
    print("\n" + "=" * 78)
    print("STEP 2 — Cross-check exp2_pca_4060_summary.json")
    print("=" * 78)

    if summary_file is None:
        print("  [MISSING] exp2_pca_4060_summary.json not found.")
        return None

    obj = safe_load_json(summary_file)
    if obj is None:
        return None

    print(f"  File: {summary_file}")
    print(f"  Top-level keys: {list(obj.keys()) if isinstance(obj, dict) else type(obj)}")

    n_pass = obj.get("n_pass") if isinstance(obj, dict) else None
    n_total = obj.get("n_total") if isinstance(obj, dict) else None
    print(f"  n_pass={n_pass}  n_total={n_total}")

    if n_pass is not None and n_total:
        rate = n_pass / n_total
        print(f"  Reported rate: {n_pass}/{n_total} = {rate:.4f}")

        # Compare against the pooled-6-method total from step 1, if available
        if recount_result:
            pooled_total = sum(
                c.get("success", 0) + c.get("fail", 0)
                for c in recount_result["method_pool_counts"].values()
            )
            pooled_pass = sum(
                c.get("success", 0) for c in recount_result["method_pool_counts"].values()
            )
            print(f"\n  Pooled-across-all-methods count from protocol files: "
                  f"{pooled_pass}/{pooled_total}")
            if pooled_total == n_total and pooled_pass == n_pass:
                print("  [MATCH] exp2_pca_4060_summary.json's n_pass/n_total IS the "
                      "pooled-all-methods count from these protocol files.")
                print("  -> 106/180 (or whatever n_pass/n_total is) is NOT HypatiaX-only.")
            else:
                print("  [NO MATCH] exp2_pca_4060_summary.json's figures do NOT match the "
                      "pooled count from these 11 protocol files.")
                print("  -> Source of n_pass/n_total is still unconfirmed; "
                      "may come from yet another file, or these files are incomplete.")

            hx_pass = recount_result["n_pass"]
            hx_total = recount_result["n_total"]
            print(f"\n  HypatiaX-only count from these files: {hx_pass}/{hx_total}")
            print(f"  *** This is the figure that should replace '9/30' in §10.7, "
                  f"NOT {n_pass}/{n_total}. ***")

    return obj


def step3_diff_duplicates(dup_groups):
    print("\n" + "=" * 78)
    print("STEP 3 — Diff duplicate fixc3_baseline.json / split_protocol_disclosure.json copies")
    print("=" * 78)

    for name, paths in dup_groups.items():
        print(f"\n  {name}: {len(paths)} copy/copies found")
        if len(paths) < 2:
            for p in paths:
                print(f"    (only one copy) {p}")
            continue

        hashes = {}
        for p in paths:
            h = sha256_of(p)
            hashes.setdefault(h, []).append(p)

        if len(hashes) == 1:
            print(f"    [OK] All {len(paths)} copies are byte-identical.")
        else:
            print(f"    [DIVERGENT] {len(hashes)} distinct content variant(s) found:")
            for h, plist in hashes.items():
                print(f"      hash {h[:12]}...:")
                for p in plist:
                    print(f"        - {p}")
                # show a quick content fingerprint for the divergent groups
                obj = safe_load_json(plist[0])
                if isinstance(obj, dict):
                    interesting = {
                        k: obj[k] for k in
                        ("n_pass", "n_total", "solve_rate", "random_split_used",
                         "split_function", "test_size", "paper_claim")
                        if k in obj
                    }
                    if interesting:
                        print(f"        content: {interesting}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    search_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/mnt/user-data/uploads")
    if not search_dir.exists():
        print(f"[ERROR] Search directory does not exist: {search_dir}")
        sys.exit(1)

    print(f"Searching under: {search_dir}\n")
    print(f"(Checking exact canonical paths from tree_r.txt first, "
          f"falling back to flat filename search.)\n")

    protocol_files, summary_file = report_path_fidelity(search_dir)

    print("=" * 78)
    print("Duplicate-prone files — canonical path check")
    print("=" * 78)
    dup_groups = report_duplicate_path_fidelity(search_dir)
    print()

    recount_result = step1_recount_hypatiax_only(protocol_files)
    summary_obj = step2_crosscheck_summary(summary_file, recount_result)
    step3_diff_duplicates(dup_groups)

    # Write machine-readable report alongside this script
    report = {
        "protocol_files_found": [str(p) for p in protocol_files],
        "summary_file_found": str(summary_file) if summary_file else None,
        "recount_result": {
            k: v for k, v in (recount_result or {}).items() if k != "candidate_equation_results"
        } if recount_result else None,
        "candidate_equation_results": recount_result["candidate_equation_results"] if recount_result else None,
        "exp2_pca_4060_summary_contents": summary_obj,
        "duplicate_files_found": {
            name: [str(p) for p in paths] for name, paths in dup_groups.items()
        },
    }
    out_path = Path(__file__).parent / "investigate_section_10_7_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n\nFull machine-readable report written to: {out_path}")


if __name__ == "__main__":
    main()

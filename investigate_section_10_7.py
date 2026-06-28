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

HYPATIAX_METHOD_NAMES = {
    "EnhancedHybridSystemDeFi",  # confirmed HypatiaX column in the 6-method pool
    "HybridDiscoverySystem",     # fallback alias, in case naming differs by file
}

ALL_SIX_METHODS = {
    "PureLLM",
    "ImprovedNN",
    "EnhancedHybridSystemDeFi",
    "HybridSystemLLMNN",
    "SymbolicEngineWithLLM",
    "HybridDiscoverySystem",
}

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
    Normalize the many shapes seen so far in this investigation:
      - a flat list of records
      - a dict keyed by index/equation name -> record
      - a dict with a top-level 'results' / 'records' / 'data' wrapper
    Returns a list of dict-like records.
    """
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("results", "records", "data", "rows"):
            if key in obj and isinstance(obj[key], list):
                return obj[key]
        # dict keyed by id -> record
        if all(isinstance(v, dict) for v in obj.values()):
            return list(obj.values())
    return []


def extract_method_records(record):
    """
    Given one top-level record (one equation's results, possibly nested
    under a 'results' sub-dict keyed by method name, as seen in
    hypatiax_defi_benchmark_pca_results.json / _merged.json), return a
    dict of method_name -> method_result_dict.

    Also handles the flatter '30 equations x 6 methods, all flattened
    into one list' shape described in the original transcript, where each
    row IS one (equation, method) pair with its own 'method' field.
    """
    if not isinstance(record, dict):
        return {}

    # Shape A: {'results': {method_name: {...}, ...}, ...}
    if "results" in record and isinstance(record["results"], dict):
        return record["results"]

    # Shape B: flat row with an explicit 'method' field
    method = record.get("method") or record.get("method_name") or record.get("system")
    if method:
        return {method: record}

    # Shape C: top-level dict IS already method_name -> result
    if any(k in ALL_SIX_METHODS for k in record.keys()):
        return {k: v for k, v in record.items() if k in ALL_SIX_METHODS}

    return {}


def is_success(method_result):
    if not isinstance(method_result, dict):
        return False
    val = method_result.get("success")
    if isinstance(val, bool):
        return val
    # fallback heuristics seen elsewhere in this investigation
    r2 = method_result.get("r2", method_result.get("train_r2"))
    if isinstance(r2, (int, float)) and not (isinstance(r2, float) and math.isnan(r2)):
        return r2 > 0.0
    return False


# ---------------------------------------------------------------------------
# Main investigation steps
# ---------------------------------------------------------------------------

def step1_recount_hypatiax_only(protocol_files):
    print("=" * 78)
    print("STEP 1 — HypatiaX-only recount across protocol_core_noiseless_pca_*.json")
    print("=" * 78)

    if not protocol_files:
        print("  [MISSING] No protocol_core_noiseless_pca_*.json files found.")
        print("  -> Cannot compute the corrected 'N/30' figure without these.")
        return None

    print(f"  Found {len(protocol_files)} file(s):")
    for f in protocol_files:
        print(f"    - {f}")

    equation_results = {}  # equation_id -> bool success (HypatiaX only), last-write-wins
    method_pool_counts = defaultdict(lambda: defaultdict(int))  # method -> {success,fail}
    total_rows_seen = 0
    duplicate_equation_ids = defaultdict(int)

    for f in protocol_files:
        obj = safe_load_json(f)
        if obj is None:
            continue
        records = find_records(obj)
        print(f"\n  --- {f.name}: {len(records)} top-level record(s) ---")

        for rec in records:
            eq_id = (
                rec.get("equation_id")
                or rec.get("name")
                or rec.get("equation")
                or rec.get("id")
            )
            method_map = extract_method_records(rec)
            if not method_map:
                continue

            for method_name, mres in method_map.items():
                total_rows_seen += 1
                succ = is_success(mres)
                method_pool_counts[method_name]["success" if succ else "fail"] += 1

                if method_name in HYPATIAX_METHOD_NAMES and eq_id is not None:
                    if eq_id in equation_results:
                        duplicate_equation_ids[eq_id] += 1
                    equation_results[eq_id] = succ

    print(f"\n  Total (equation, method) rows seen across all files: {total_rows_seen}")
    print("\n  Per-method pooled success/fail counts (sanity check vs 30x6=180):")
    for method, counts in sorted(method_pool_counts.items()):
        s, fa = counts.get("success", 0), counts.get("fail", 0)
        print(f"    {method:30s} success={s:3d}  fail={fa:3d}  total={s+fa:3d}")

    if duplicate_equation_ids:
        print("\n  [WARN] Equation IDs appearing in HypatiaX rows more than once "
              "(possible duplicate shards / re-runs — last value won):")
        for eq, n in duplicate_equation_ids.items():
            print(f"    {eq!r}: seen {n + 1} times")

    n_pass = sum(1 for v in equation_results.values() if v)
    n_total = len(equation_results)
    print(f"\n  >>> HypatiaX-only result: {n_pass}/{n_total} "
          f"({n_pass / n_total:.4f})" if n_total else "  >>> No HypatiaX rows found.")
    if n_total != 30:
        print(f"  [WARN] Expected 30 distinct equations, found {n_total}. "
              f"Check for missing files or naming mismatches.")

    return {
        "n_pass": n_pass,
        "n_total": n_total,
        "rate": (n_pass / n_total) if n_total else None,
        "method_pool_counts": {k: dict(v) for k, v in method_pool_counts.items()},
        "equation_results": equation_results,
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
            k: v for k, v in (recount_result or {}).items() if k != "equation_results"
        } if recount_result else None,
        "equation_results": recount_result["equation_results"] if recount_result else None,
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

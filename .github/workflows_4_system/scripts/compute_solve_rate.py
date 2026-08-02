#!/usr/bin/env python3
"""
compute_solve_rate.py — recompute a real solve-rate number from HypatiaX
benchmark result JSON, without trusting the repo's own `success` flags
or Gate C's row-parser (which silently returns 0 rows for one of the
two known output schemas).

WHY THIS EXISTS
----------------
Two things in this repo make it easy to get a wrong number by accident:

1. Several methods (ImprovedNN, EnhancedHybridSystemDeFi, ...) hardcode
   `success=True` whenever the method runs without raising an exception —
   they never check r2 against the threshold. Confirmed directly: an
   ImprovedNN row with r2=0.9999966 (below the 0.999999 threshold) was
   still recorded success=True. Trusting `success` inflates every rate.

2. Result files come in at least two incompatible shapes:
     (a) {"tests": [ {"description":..., "results": {method: {...}}} ]}
         — written per-domain by run_comparative_suite_benchmark_v2.py
     (b) [ {"test":..., "method":..., "r2":...}, ... ]
         — a flat consolidated file (e.g. benchmark_results.json)
   Gate C's own baseline-lock script in run_all.sh only understands a
   generic dict-with-key-in-('results','equation_results','data','rows')
   layout, which matches NEITHER (a) nor a flat list-of-dicts cleanly
   for (a) — it silently yields 0 usable rows for shape (a) and quietly
   locks n_total=0. This script explicitly handles both shapes so you
   can tell whether n_total=0 means "no data" or "parser didn't understand
   the format."

USAGE
-----
  # Fetch files first, e.g.:
  #   git clone --filter=blob:none --no-checkout <repo_url> repo
  #   cd repo && git checkout <sha_or_branch> -- <results_dir>
  #
  python3 compute_solve_rate.py <results_dir> \\
      [--threshold 0.999999] \\
      [--method-filter hybrid,hypatia,proposed] \\
      [--exclude checkpoint,disclosure,baseline]

  # Example matching Gate C's own (documented) criteria:
  python3 compute_solve_rate.py exp2/ \\
      --threshold 0.999999 \\
      --method-filter hypatiax,hybridv50,hybrid50,hybridsymbolic,hybriddefi,hypatia,hybrid,ours,proposed

OUTPUT
------
Per-file row counts (so you can see which files actually contributed),
per-method pass/total, and a pooled total — printed separately, never
silently merged, so a filter that catches more than one system is
visible instead of hidden inside one number.
"""

import argparse
import glob
import json
import pathlib
import sys
from collections import defaultdict


def normalize(name: str) -> str:
    return str(name or "").lower().replace("-", "").replace("_", "").replace(" ", "")


def extract_r2(row: dict) -> float | None:
    """Same key-fallback order Gate C uses: r2, r2_test, r2_train, best_r2, R2."""
    for k in ("r2", "r2_test", "test_r2", "r2_train", "train_r2", "best_r2", "R2"):
        v = row.get(k)
        if v is not None:
            try:
                f = float(v)
                if f <= 1.01:  # sanity guard against non-r2 numeric fields
                    return f
            except (TypeError, ValueError):
                pass
    return None


def rows_from_shape_a(data: dict):
    """{"tests": [ {"results": {method_name: {...row...}}} ]}"""
    for test in data.get("tests", []):
        if not isinstance(test, dict):
            continue
        results = test.get("results", {})
        if not isinstance(results, dict):
            continue
        for method_name, row in results.items():
            if isinstance(row, dict):
                yield method_name, row, test.get("description", "")


def rows_from_shape_b(data: list):
    """[ {"method": ..., "r2": ..., "test": ...}, ... ]"""
    for item in data:
        if isinstance(item, dict) and "method" in item:
            yield item.get("method", ""), item, item.get("test", item.get("description", ""))


def rows_from_shape_c(data: list):
    """
    [ {"equation_id":..., "results": {method_name: {...row...}}}, ... ]
    Bare list of test-cases (no "tests" wrapper), each holding a per-method
    results dict. Seen in per-seed defi benchmark files, e.g.
    hypatiax_defi_benchmark_v3_results_seed*.json. Distinct from shape_b:
    items here have "results" (dict), not a top-level "method" key.
    """
    for item in data:
        if not isinstance(item, dict):
            continue
        results = item.get("results", {})
        if not isinstance(results, dict):
            continue
        for method_name, row in results.items():
            if isinstance(row, dict):
                yield method_name, row, item.get("equation_id", item.get("description", ""))


def rows_from_generic_gatec_shape(data):
    """
    Reproduces Gate C's own _rows()/row.get('method') logic exactly, for
    comparison. Included so you can see when Gate C's parser would have
    silently returned nothing for a file that shape_a/shape_b DO parse.
    """
    def _rows(d):
        if isinstance(d, dict):
            for key in ("results", "equation_results", "data", "rows"):
                v = d.get(key)
                if v is not None:
                    yield from _rows(v)
                    return
            yield d
        elif isinstance(d, list):
            for item in d:
                if isinstance(item, dict):
                    yield item

    for row in _rows(data):
        method = row.get("method") or row.get("model") or ""
        yield method, row, row.get("description", "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", help="Directory containing result JSON files")
    ap.add_argument("--threshold", type=float, default=0.999999)
    ap.add_argument("--method-filter", default="", help="Comma-separated substrings; empty = no filter (all methods)")
    ap.add_argument("--exclude", default="checkpoint,disclosure,baseline",
                     help="Comma-separated substrings to exclude from filenames")
    ap.add_argument("--exclude-pca", action="store_true", default=False,
                     help="Skip files with '_pca' in the name (default: OFF — "
                          "protocol_core_*_pca_*.json files are real per-domain "
                          "experiment results, not just disclosure metadata; the "
                          "actual disclosure file is already caught by --exclude's "
                          "default 'disclosure' term)")
    ap.add_argument("--show-gatec-comparison", action="store_true",
                     help="Also show what Gate C's own generic parser would extract, for contrast")
    ap.add_argument("--source", choices=["auto", "shards", "flat"], default="auto",
                     help="auto (default): if any flat list-shaped file is found (e.g. a consolidated "
                          "benchmark_results.json), use ONLY flat file(s) and skip per-domain shard files, "
                          "since a consolidated file typically duplicates the shard files' rows. "
                          "'shards' forces per-domain-file parsing only; 'flat' forces flat-file-only.")
    args = ap.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    exclude_terms = [t for t in args.exclude.split(",") if t]
    method_filter = [t.strip().lower() for t in args.method_filter.split(",") if t.strip()]

    per_method = defaultdict(lambda: [0, 0])   # method -> [pass, total]
    files_used = []
    files_skipped = []

    # ── decide source mode up front (auto: prefer flat if any exist) ──
    candidate_files = []
    for fp in sorted(results_dir.glob("*.json")):
        if any(x in fp.name for x in exclude_terms):
            continue
        if args.exclude_pca and "_pca" in fp.name:
            continue
        try:
            data = json.loads(fp.read_text())
        except Exception:
            continue
        if isinstance(data, list):
            if data and isinstance(data[0], dict) and "method" in data[0]:
                shape = "flat"
            elif data and isinstance(data[0], dict) and "results" in data[0]:
                shape = "seedlist"
            else:
                shape = "other"
        elif isinstance(data, dict) and "tests" in data:
            shape = "shard"
        else:
            shape = "other"
        candidate_files.append((fp, data, shape))

    if args.source == "auto":
        has_flat = any(s == "flat" for _, _, s in candidate_files)
        mode = "flat" if has_flat else "shard"
        if has_flat:
            print(f"[auto] Found flat consolidated file(s) — using ONLY those, skipping per-domain "
                  f"shard files to avoid double-counting the same rows. Use --source shards to override.\n")
    elif args.source == "shards":
        mode = "shard"
    else:
        mode = "flat"

    # ── explicit warning when more than one flat file will be pooled ──
    # Multiple flat-shaped files in the same directory are NOT necessarily
    # the same experiment saved twice — one can be a consolidated result
    # for THIS directory's condition, and another can be a same-shaped file
    # left over from (or copied from) a DIFFERENT experiment entirely (e.g.
    # an extrapolation-robustness sweep sitting inside a PCA-split results
    # directory). auto/flat mode has no way to tell these apart — it pools
    # every flat file's rows together. Per-file row counts in "Files
    # considered" below make this visible if you look for it, but a
    # non-blocking diagnostic buried in a long CI log is easy to miss, so
    # flag it loudly here instead of only implicitly.
    if mode == "flat":
        flat_files = [fp for fp, _, s in candidate_files if s == "flat"]
        if len(flat_files) > 1:
            print("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"  !!! WARNING: {len(flat_files)} flat-shaped files found in this directory —")
            print("  !!! ALL of them will be pooled into the numbers below. This script cannot tell")
            print("  !!! whether they're genuinely the same experiment's rows or a same-shaped file")
            print("  !!! from a DIFFERENT experiment condition that doesn't belong here. Verify each")
            print("  !!! file's contents (e.g. check for extrapolation-only fields like extrap_r2_far,")
            print("  !!! or a 'test'/'description' naming pattern that doesn't match this directory's")
            print("  !!! purpose) before trusting the pooled figure:")
            for f in flat_files:
                print(f"  !!!   - {f.name}")
            print("  !!! If any of these don't belong, rerun with --exclude to drop them by filename.")
            print("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

    for fp, data, shape in candidate_files:
        if mode == "shard" and shape not in ("shard", "seedlist"):
            files_skipped.append((fp.name, f"skipped ({shape} file, mode=shard)"))
            continue
        if mode == "flat" and shape not in ("flat", "seedlist"):
            files_skipped.append((fp.name, f"skipped ({shape} file, mode=flat)"))
            continue
        if shape == "other":
            files_skipped.append((fp.name, "unrecognized JSON shape"))
            continue

        if shape == "shard":
            row_iter = rows_from_shape_a(data)
        elif shape == "flat":
            row_iter = rows_from_shape_b(data)
        elif shape == "seedlist":
            row_iter = rows_from_shape_c(data)
        else:
            row_iter = []  # unrecognized shape

        n_rows_here = 0
        for method_name, row, _desc in row_iter:
            norm = normalize(method_name)
            if method_filter and not any(f in norm for f in method_filter):
                continue
            r2 = extract_r2(row)
            if r2 is None:
                continue
            per_method[method_name][1] += 1
            n_rows_here += 1
            if r2 >= args.threshold:
                per_method[method_name][0] += 1

        files_used.append((fp.name, n_rows_here))

    print(f"=== Files considered from {results_dir} ===")
    for name, n in files_used:
        print(f"  {name:<55} {n} matching row(s)")
    if files_skipped:
        print(f"\n=== Files skipped ===")
        for name, reason in files_skipped:
            print(f"  {name:<55} {reason}")

    print(f"\n=== Per-method pass/total (threshold r2 >= {args.threshold}) ===")
    total_pass = total_all = 0
    for m, (p, t) in sorted(per_method.items()):
        rate = p / t if t else float("nan")
        print(f"  {m:<45} {p}/{t}   rate={rate:.3f}")
        total_pass += p
        total_all += t

    if len(per_method) > 1:
        print(f"\n  POOLED across {len(per_method)} method(s) matching filter: "
              f"{total_pass}/{total_all} = {total_pass/total_all:.3f}" if total_all else "  POOLED: no data")
        print("  ^ NOTE: pooling multiple distinctly-named methods into one number can hide")
        print("    a filter that's too broad. Verify each method above is really the same system")
        print("    before treating the pooled figure as meaningful.")
    elif total_all:
        print(f"\n  TOTAL: {total_pass}/{total_all} = {total_pass/total_all:.3f}")

    if args.show_gatec_comparison:
        print(f"\n=== For contrast: Gate C's own generic row parser on the same files ===")
        gc_total = 0
        for fp in sorted(results_dir.glob("*.json")):
            if any(x in fp.name for x in exclude_terms) or (args.exclude_pca and "_pca" in fp.name):
                continue
            try:
                data = json.loads(fp.read_text())
            except Exception:
                continue
            n = sum(1 for _ in rows_from_generic_gatec_shape(data))
            gc_total += n
            print(f"  {fp.name:<55} {n} row(s) extracted by Gate C's parser")
        print(f"  TOTAL rows Gate C's parser would see: {gc_total}")


if __name__ == "__main__":
    main()

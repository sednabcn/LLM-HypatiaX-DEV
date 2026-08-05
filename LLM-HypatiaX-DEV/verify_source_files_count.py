#!/usr/bin/env python3
"""
verify_source_files_count.py
-----------------------------
Item 10.5 verification: does fixc3_baseline.json's `source_files` list
(currently 5 entries, all dated 2026-07-14) reflect the TRUE contents of
comparison_results/feynman-tests/exp2/, or does it coincidentally match
what the old [:5] truncation bug would have produced regardless of how
many real files exist?

WHAT THIS CHECKS
-----------------
1. Globs comparison_results/feynman-tests/exp2/ for candidate legacy
   result files -- by default with the SAME unsafe rule the doc says Gate
   C currently uses ("pools any *.json regardless of protocol"), i.e.
   *.json minus a small, obvious exclude-list of non-result artifacts
   (_merged.json, _report.md-adjacent JSON, ablation_paired.json, etc.).
   *** This filter is a best-effort reconstruction from the doc's prose,
   *** NOT read from your actual gate_c_local.py. If you share that file's
   *** real `legacy_jsons` glob/filter, tell me and I'll swap this out for
   *** the exact logic instead of this approximation.
2. Compares the count and the actual filenames against
   fixc3_baseline.json's source_files list.
3. Flags three possible outcomes:
     MATCH_CONFIRMED    - true file count == 5 AND the exact filenames
                           match the manifest -> 5 is real, truncation
                           fix is validated.
     COINCIDENCE_SUSPECT - true file count > 5 (i.e. there really are
                           more files, but the manifest still only lists
                           5) -> the old [:5]-style truncation bug (or an
                           equivalent) is still live; "5" is NOT trustworthy.
     UNDER_COUNT         - true file count < 5 -> manifest lists files that
                           aren't on disk, or dating/glob mismatch; also
                           not trustworthy, different problem.

USAGE
-----
    python3 verify_source_files_count.py \
        --result-dir  /path/to/comparison_results/feynman-tests/exp2 \
        --manifest    /path/to/fixc3_baseline.json

    # Or, if you don't have local access and just want to check a
    # directory listing you already captured (e.g. `ls -la` output saved
    # to a text file, one filename per line):
    python3 verify_source_files_count.py \
        --listing-file dir_listing.txt \
        --manifest     fixc3_baseline.json
"""

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Best-effort reconstruction of "any *.json regardless of protocol" minus the
# obviously-non-result artifacts that would never plausibly be a legacy
# per-shard result file. THIS IS A GUESS -- replace with the real filter
# from gate_c_local.py / run_all.sh as soon as it's available.
# ---------------------------------------------------------------------------
KNOWN_NON_RESULT_FILES = {
    "_merged.json",
    "_analysis.json",
    "_checkpoint.json",
    "_merged.csv",
    "_stats.json",
    "ablation_paired.json",
    "shard_manifest.txt",       # not json but harmless to list here
    "fixc3_baseline.json",
    "benchmark_results.json",          # aggregate/flat file, not a per-shard result
    "benchmark_results_extrap.json",   # aggregate/flat file, not a per-shard result
}


def _is_candidate_result_file(name: str) -> bool:
    if not name.endswith(".json"):
        return False
    if name in KNOWN_NON_RESULT_FILES:
        return False
    if name.endswith(".manifest.json"):
        return False
    return True


def load_directory_files(result_dir: Path) -> list[str]:
    if not result_dir.exists():
        print(f"::error:: result-dir does not exist: {result_dir}", file=sys.stderr)
        sys.exit(1)
    all_json = sorted(p.name for p in result_dir.glob("*.json"))
    candidates = sorted(n for n in all_json if _is_candidate_result_file(n))
    return all_json, candidates


def load_listing_file(listing_path: Path) -> list[str]:
    lines = [l.strip() for l in listing_path.read_text().splitlines() if l.strip()]
    # Accept either bare filenames or `ls -la`-style lines; take the last
    # whitespace-separated token on each line as the filename.
    names = [l.split()[-1] for l in lines]
    all_json = sorted(n for n in names if n.endswith(".json"))
    candidates = sorted(n for n in all_json if _is_candidate_result_file(n))
    return all_json, candidates


def load_manifest_source_files(manifest_path: Path) -> list[str]:
    with open(manifest_path) as f:
        manifest = json.load(f)
    # Try a few plausible key names/shapes defensively, since we haven't
    # seen the real fixc3_baseline.json schema.
    for key in ("source_files", "sourceFiles", "legacy_jsons", "files"):
        if key in manifest:
            raw = manifest[key]
            break
    else:
        print(f"::error:: could not find a source_files-like key in {manifest_path}. "
              f"Top-level keys present: {sorted(manifest.keys())}", file=sys.stderr)
        sys.exit(1)

    # Entries might be bare filenames or full paths; normalise to basename.
    names = []
    for entry in raw:
        if isinstance(entry, str):
            names.append(Path(entry).name)
        elif isinstance(entry, dict):
            # e.g. {"path": ..., "sha256": ...} shape, matching the
            # manifest style used elsewhere in this pipeline
            p = entry.get("path") or entry.get("file") or entry.get("name")
            if p:
                names.append(Path(p).name)
    return sorted(names)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result-dir", type=Path, default=None,
                     help="Path to comparison_results/feynman-tests/exp2/")
    ap.add_argument("--listing-file", type=Path, default=None,
                     help="Path to a saved directory listing (one filename per line, "
                          "or `ls -la` output) — use this if you don't have local "
                          "filesystem access to result-dir.")
    ap.add_argument("--manifest", type=Path, required=True,
                     help="Path to fixc3_baseline.json")
    args = ap.parse_args()

    if not args.result_dir and not args.listing_file:
        ap.error("Supply either --result-dir or --listing-file")

    if args.result_dir:
        all_json, candidates = load_directory_files(args.result_dir)
        source_label = str(args.result_dir)
    else:
        all_json, candidates = load_listing_file(args.listing_file)
        source_label = str(args.listing_file)

    manifest_files = load_manifest_source_files(args.manifest)

    print(f"Directory/listing source : {source_label}")
    print(f"Manifest                 : {args.manifest}")
    print()
    print(f"All *.json in dir/listing        : {len(all_json)}")
    for n in all_json:
        flag = "" if n in candidates else "  (excluded: known non-result file)"
        print(f"    {n}{flag}")
    print()
    print(f"Candidate legacy result files     : {len(candidates)}")
    print(f"Manifest source_files entries     : {len(manifest_files)}")
    print()

    candidates_set = set(candidates)
    manifest_set = set(manifest_files)

    only_on_disk = sorted(candidates_set - manifest_set)
    only_in_manifest = sorted(manifest_set - candidates_set)

    if only_on_disk:
        print(f"Files on disk but NOT in manifest ({len(only_on_disk)}):")
        for n in only_on_disk:
            print(f"    {n}")
        print()
    if only_in_manifest:
        print(f"Files in manifest but NOT found on disk ({len(only_in_manifest)}):")
        for n in only_in_manifest:
            print(f"    {n}")
        print()

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    if len(candidates) == len(manifest_files) and not only_on_disk and not only_in_manifest:
        verdict = "MATCH_CONFIRMED"
        detail = (f"Directory genuinely contains {len(candidates)} candidate result "
                  f"file(s), exactly matching the manifest. 5 is real — truncation "
                  f"fix is validated. Safe to mark 10.5 live-confirmed.")
    elif len(candidates) > len(manifest_files):
        verdict = "COINCIDENCE_SUSPECT"
        detail = (f"Directory actually contains {len(candidates)} candidate result "
                  f"file(s) — MORE than the {len(manifest_files)} the manifest lists. "
                  f"The manifest is silently dropping real files. This is exactly the "
                  f"failure mode item 10.5 warned about: '5' is NOT trustworthy, a "
                  f"truncation-style bug is still live somewhere in the selection path.")
    elif len(candidates) < len(manifest_files):
        verdict = "UNDER_COUNT"
        detail = (f"Manifest lists {len(manifest_files)} file(s) but only "
                  f"{len(candidates)} were found on disk/listing. Either files were "
                  f"deleted/moved after the lock, or the manifest references files "
                  f"under a different path/naming convention than this check assumes. "
                  f"Not trustworthy as-is — needs manual review, different failure "
                  f"mode than truncation.")
    else:
        verdict = "COUNT_MATCHES_BUT_NAMES_DIFFER"
        detail = (f"Counts match ({len(candidates)} == {len(manifest_files)}) but the "
                  f"actual filenames differ (see lists above). Could still be a "
                  f"truncation bug that happens to drop one file and pick up a "
                  f"different stray one — do not treat count-equality alone as proof.")

    print("=" * 70)
    print(f"VERDICT: {verdict}")
    print(detail)
    print("=" * 70)

    sys.exit(0 if verdict == "MATCH_CONFIRMED" else 1)


if __name__ == "__main__":
    main()

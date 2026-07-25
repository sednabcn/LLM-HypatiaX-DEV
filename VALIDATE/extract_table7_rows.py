#!/usr/bin/env python3
"""
extract_table7_rows.py (v2)

Extracts per-equation train/extrapolation r2 values from HypatiaX benchmark
JSON files, for building paper tables (e.g. Table 7).

Two changes from v1, both driven by real bugs found in this project's data:

1. NO MORE SILENT FALLBACK BETWEEN FIELDS.
   v1 fell back from `extrap_r2_far` to a generic `r2` field when the former
   was missing. That generic `r2` field is the TRAINING/interpolation fit,
   not an extrapolation metric -- the fallback silently produced a plausible-
   looking but wrong "extrapolation R2" for methods (e.g. ImprovedNN) whose
   extrap_r2_far is null on every record. v2 keeps train_r2 and extrap_r2_far
   as always-separate columns. A missing extrap_r2_far is reported as
   genuinely missing (blank in the CSV, counted explicitly in the summary),
   never silently filled from train_r2 or any other field.

2. SHA256 MANIFEST VERIFICATION.
   Several "flat" consolidated files in this project have turned out to be
   stale or mismatched copies sharing a filename with a different, genuine
   run (confirmed by hash mismatch against a *.manifest.json's recorded
   sha256 for that exact filename). v2 looks for a manifest file (either
   passed explicitly via --manifest or auto-detected as *.manifest.json in
   the target directory) and, for every input file it reads whose filename
   appears in the manifest's `inputs` block, verifies the file's sha256
   against the manifest's recorded value before trusting it. A mismatch is
   a loud, blocking warning, not a footnote -- the file is still read (so
   you can inspect it) but every row extracted from it is tagged
   "UNVERIFIED SOURCE" in the output.

3. FLAT-FILE CAUTION BY DEFAULT.
   Files whose name matches typical "pre-merged" naming
   (benchmark_results*.json, *_summary.json, *paired*.json) but are NOT
   shard files (protocol_core_*.json) are flagged as flat/consolidated and
   trigger a warning recommending the raw per-domain shard files instead,
   unless the file hash-verifies against a manifest or --allow-flat is
   passed explicitly.

Supports two JSON record schemas, auto-detected per file:
  - "flat" schema: a list of records, each with test/domain/method/r2/
    extrap_r2_far (as in benchmark_results_extrap.json, or a
    protocol_core_*.json shard).
  - "paired" schema: a list of records, each with equation_name/domain and
    one sub-dict per system (e.g. "hypatia", "pysr_only"), each sub-dict
    carrying train_r2/extrap_r2_far/extrap_success (as in
    ablation_paired.json).

Usage:
    # flat schema, single method, from raw shards (recommended)
    python extract_table7_rows.py DIR \
        --schema flat --method "HybridDiscoverySystem v50_2 (tools)" \
        --pattern "protocol_core_extrap_20260722_*.json" \
        --threshold 0.999999 --out rows.csv

    # paired schema, both systems, with manifest verification
    python extract_table7_rows.py DIR \
        --schema paired --systems hypatia,pysr_only \
        --pattern "ablation_paired.json" \
        --manifest DIR/ablation_paired.json.manifest.json \
        --threshold 0.999999 --out rows.csv
"""

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import sys


FLAT_HINTS = ("benchmark_results", "_summary", "paired")
SHARD_HINT = "protocol_core"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path):
    """Return {basename: expected_sha256} from a *.manifest.json's 'inputs' block."""
    with open(path) as f:
        m = json.load(f)
    out = {}
    for fname, info in (m.get("inputs") or {}).items():
        if isinstance(info, dict) and "sha256" in info:
            out[os.path.basename(fname)] = info["sha256"]
    return out


def find_manifest(directory):
    candidates = glob.glob(os.path.join(directory, "*.manifest.json"))
    return candidates[0] if candidates else None


def is_flat_like(filename):
    base = os.path.basename(filename)
    if SHARD_HINT in base:
        return False
    return any(hint in base for hint in FLAT_HINTS)


def is_finite_number(v):
    return isinstance(v, (int, float)) and not (isinstance(v, float) and math.isinf(v))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory")
    ap.add_argument("--schema", choices=["flat", "paired", "auto"], default="auto")
    ap.add_argument("--method", help="Method name for --schema flat")
    ap.add_argument("--systems", help="Comma-separated system keys for --schema paired, "
                                        "e.g. hypatia,pysr_only")
    ap.add_argument("--pattern", default="*.json")
    ap.add_argument("--threshold", type=float, default=0.999999)
    ap.add_argument("--manifest", default=None,
                     help="Path to a *.manifest.json for hash verification. "
                          "If omitted, auto-searched for in --directory.")
    ap.add_argument("--allow-flat", action="store_true",
                     help="Proceed with flat/consolidated files even without "
                          "manifest verification (NOT recommended -- prefer "
                          "raw shard files instead).")
    ap.add_argument("--out", default="rows.csv")
    args = ap.parse_args()

    manifest_path = args.manifest or find_manifest(args.directory)
    manifest = {}
    if manifest_path:
        manifest = load_manifest(manifest_path)
        print(f"[manifest] loaded {len(manifest)} known-hash input(s) from {manifest_path}")
    else:
        print("[manifest] none found -- no hash verification will be performed for this run")

    files = sorted(glob.glob(os.path.join(args.directory, args.pattern)))
    if not files:
        print(f"No files matched {args.pattern!r} in {args.directory!r}", file=sys.stderr)
        sys.exit(1)

    rows = []
    seen_ids = set()

    for fp in files:
        base = os.path.basename(fp)
        actual_hash = sha256_of(fp)
        verified = None  # None = no manifest entry; True/False = matched/mismatched

        if base in manifest:
            verified = (actual_hash == manifest[base])
            if verified:
                print(f"[hash OK]      {base}  matches manifest sha256")
            else:
                print(f"[HASH MISMATCH] {base}")
                print(f"    manifest expects: {manifest[base]}")
                print(f"    file actually is: {actual_hash}")
                print(f"    -> this file does NOT match what the manifest says it merged from.")
                print(f"       Rows from it will be tagged UNVERIFIED SOURCE in the output.")

        flat_like = is_flat_like(base)
        if flat_like and verified is not True and not args.allow_flat:
            print(f"[SKIPPED] {base} looks like a pre-merged/consolidated file "
                  f"(matched hint in {FLAT_HINTS}) and is not hash-verified against a "
                  f"manifest. Prefer the raw protocol_core_*.json shard files instead, "
                  f"or pass --allow-flat to force reading it anyway.")
            continue

        source_tag = "verified" if verified is True else (
            "UNVERIFIED SOURCE" if verified is False else
            ("no-manifest-entry" if manifest else "no-manifest-available")
        )

        with open(fp) as f:
            data = json.load(f)

        # normalize to a list of records
        if isinstance(data, dict) and "tests" in data:
            records = data["tests"]
        elif isinstance(data, list):
            records = data
        else:
            records = [data]

        schema = args.schema
        if schema == "auto":
            schema = "paired" if records and (
                "hypatia" in records[0] or
                any(isinstance(records[0].get(k), dict) and "train_r2" in (records[0].get(k) or {})
                    for k in records[0])
            ) else "flat"

        for rec in records:
            if schema == "flat":
                if args.method and rec.get("method") != args.method:
                    continue
                rid = (rec.get("test") or rec.get("description"), rec.get("method"))
                if rid in seen_ids:
                    continue
                seen_ids.add(rid)
                rows.append({
                    "domain": rec.get("domain", ""),
                    "equation": rec.get("test") or rec.get("description", ""),
                    "system": rec.get("method", ""),
                    "train_r2": rec.get("r2"),  # NOTE: this is the interpolation/train fit,
                                                  # never used as a stand-in for extrap_r2_far
                    "extrap_r2_far": rec.get("extrap_r2_far"),
                    "source_file": base,
                    "source_status": source_tag,
                })
            else:  # paired
                systems = (args.systems or "hypatia,pysr_only").split(",")
                equation = rec.get("equation_name") or rec.get("equation_id", "")
                domain = rec.get("domain", "")
                for sysname in systems:
                    sysname = sysname.strip()
                    block = rec.get(sysname)
                    if block is None:
                        continue
                    rid = (equation, sysname)
                    if rid in seen_ids:
                        continue
                    seen_ids.add(rid)
                    rows.append({
                        "domain": domain,
                        "equation": equation,
                        "system": sysname,
                        "train_r2": block.get("train_r2"),
                        "extrap_r2_far": block.get("extrap_r2_far"),
                        "source_file": base,
                        "source_status": source_tag,
                    })

    rows.sort(key=lambda r: (r["system"], r["domain"], r["equation"]))

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "system", "domain", "equation", "train_r2", "extrap_r2_far",
            "source_file", "source_status"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {args.out}")

    # per-system honest summary: never treat null as pass or fail silently --
    # print the null count explicitly alongside the pass count.
    by_system = {}
    for r in rows:
        by_system.setdefault(r["system"], []).append(r)

    print("\n=== Per-system summary (genuine extrap_r2_far only, no fallback) ===")
    for sysname, sysrows in by_system.items():
        total = len(sysrows)
        nulls = sum(1 for r in sysrows if r["extrap_r2_far"] is None)
        neg_inf = sum(1 for r in sysrows if isinstance(r["extrap_r2_far"], float)
                      and math.isinf(r["extrap_r2_far"]) and r["extrap_r2_far"] < 0)
        finite = [r["extrap_r2_far"] for r in sysrows
                  if is_finite_number(r["extrap_r2_far"])]
        passes = sum(1 for v in finite if v >= args.threshold)
        unverified = sum(1 for r in sysrows if r["source_status"] == "UNVERIFIED SOURCE")
        print(f"  {sysname}: total={total}  null={nulls}  -inf={neg_inf}  "
              f"finite={len(finite)}  pass(>= {args.threshold})={passes}"
              f"{'  ** ' + str(unverified) + ' row(s) UNVERIFIED SOURCE **' if unverified else ''}")

    print("\nNote: 'pass' counts only genuinely non-null, finite extrap_r2_far values. "
          "Null values are NOT counted as failures or successes -- they mean the metric "
          "was never computed for that record, and are reported separately above so "
          "nothing is silently absorbed into either bucket.")


if __name__ == "__main__":
    main()

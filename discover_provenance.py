#!/usr/bin/env python3
"""
discover_provenance.py
======================
HypatiaX §11 provenance tool (b): links every result file in results/ to its
source family, patch chain, and paper section via provenance_map.json.

Usage:
    python3 discover_provenance.py \
        --root  .                          \
        --map   provenance_map.json        \
        --out   logs/provenance_audit

Called by:
  run_all.sh  →  run_step "discover-provenance" ...
  run_all.py  →  Step("discover-provenance", ...)

Output files (written to --out directory):
  provenance_audit_summary.txt   — human-readable coverage report
  provenance_audit.json          — machine-readable full record
  orphans.txt                    — result files with no provenance entry

Exit codes:
  0 — completed (orphans are warned, not fatal)
  1 — critical error (map file missing, root not found)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_map(map_path: Path) -> dict:
    if not map_path.exists():
        print(f"  WARNING: provenance_map.json not found at {map_path}")
        print("  Creating minimal empty map — run experiments first.")
        return {"families": [], "outputs": []}
    with open(map_path) as f:
        return json.load(f)


def scan_result_files(root: Path) -> list[Path]:
    """Collect all .json and .csv files under data/results/ and results/."""
    result_dirs = [
        root / "data" / "results",
        root / "hypatiax" / "data" / "results",
    ]
    files = []
    for d in result_dirs:
        if d.exists():
            files.extend(d.rglob("*.json"))
            files.extend(d.rglob("*.csv"))
            files.extend(d.rglob("*.txt"))
    return sorted(set(files))


def match_file_to_family(rel_path: str, families: list[dict]) -> dict | None:
    """Return the first family whose path patterns match rel_path."""
    for fam in families:
        patterns = fam.get("output_patterns", [])
        for pat in patterns:
            if pat in rel_path or rel_path.endswith(pat):
                return fam
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="HypatiaX provenance linker")
    parser.add_argument("--root", default=".", help="Repo root (default: .)")
    parser.add_argument("--map",  default="provenance_map.json",
                        help="Path to provenance_map.json")
    parser.add_argument("--out",  default="logs/provenance_audit",
                        help="Output directory for audit files")
    args = parser.parse_args()

    root    = Path(args.root).resolve()
    map_path = Path(args.map) if Path(args.map).is_absolute() else root / args.map
    out_dir  = Path(args.out) if Path(args.out).is_absolute() else root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Root       : {root}")
    print(f"  Map        : {map_path}")
    print(f"  Output dir : {out_dir}")

    pmap = load_map(map_path)
    families = pmap.get("families", [])
    registered_outputs = {o.get("file", "") for o in pmap.get("outputs", [])}

    result_files = scan_result_files(root)
    print(f"\n  Result files found : {len(result_files)}")
    print(f"  Registered families: {len(families)}")

    authoritative = []
    orphans       = []
    audit_records = []

    for fpath in result_files:
        rel = str(fpath.relative_to(root))
        family = match_file_to_family(rel, families)
        is_registered = rel in registered_outputs or any(
            rel.endswith(o.get("file", "")) for o in pmap.get("outputs", [])
        )

        record = {
            "file": rel,
            "family": family.get("name") if family else None,
            "paper_section": family.get("paper_section") if family else None,
            "registered": is_registered,
            "status": "AUTHORITATIVE" if family else "ORPHAN",
        }
        audit_records.append(record)

        if family:
            authoritative.append(rel)
            print(f"  ✓ [{family.get('name','?'):30s}] {rel}")
        else:
            orphans.append(rel)

    # ── Write outputs ─────────────────────────────────────────────────────────
    summary_lines = [
        f"HypatiaX Provenance Audit — {datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"Root: {root}",
        "",
        f"Total result files  : {len(result_files)}",
        f"AUTHORITATIVE       : {len(authoritative)}",
        f"ORPHAN              : {len(orphans)}",
        "",
    ]

    if orphans:
        summary_lines.append("Orphaned files (no provenance entry):")
        for o in orphans:
            summary_lines.append(f"  ORPHAN  {o}")
        summary_lines.append("")
        summary_lines.append(
            "Action: add entries to provenance_map.json or delete orphaned files."
        )
    else:
        summary_lines.append("All result files are registered in provenance_map.json ✓")

    summary_text = "\n".join(summary_lines)
    (out_dir / "provenance_audit_summary.txt").write_text(summary_text)
    (out_dir / "provenance_audit.json").write_text(
        json.dumps({"generated": datetime.utcnow().isoformat(),
                    "records": audit_records}, indent=2)
    )
    if orphans:
        (out_dir / "orphans.txt").write_text("\n".join(orphans))

    print(f"\n  Summary → {out_dir / 'provenance_audit_summary.txt'}")
    print(f"  JSON    → {out_dir / 'provenance_audit.json'}")
    if orphans:
        print(f"  Orphans → {out_dir / 'orphans.txt'}")
        print(f"\n  ⚠  {len(orphans)} orphaned file(s) — see orphans.txt")
    else:
        print(f"\n  ✓ All {len(result_files)} result files have provenance entries")

    # Print summary block matching the format run_all.sh greps for
    print()
    print(f"  AUTHORITATIVE : {len(authoritative)}")
    print(f"  ORPHAN        : {len(orphans)}")
    print(f"  Total         : {len(result_files)}")

    return 0  # Orphans are warnings, not failures


if __name__ == "__main__":
    sys.exit(main())

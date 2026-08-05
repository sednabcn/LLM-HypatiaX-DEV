#!/usr/bin/env python3
"""
check_hypatiax_protocols.py
===========================
Preflight check: verify that all required hypatiax/protocols/ input-data modules
are present before any experiments run.

These files are imported by root protocols/ scripts (e.g.
experiment_protocol_ablation_exp1.py imports from hypatiax.protocols.experiment_protocol_defi).
If any are missing the pipeline will fail mid-run with an import error rather than
a clear diagnostic. This script surfaces the problem before any compute is spent.

Called by:
  run_all.sh  →  run_step "check-hypatiax-protocols" ...
  run_all.py  →  Step("check-hypatiax-protocols", ...)

Exit codes:
  0 — all required modules present
  1 — one or more modules missing
"""

import sys
from pathlib import Path

# ── Required modules ──────────────────────────────────────────────────────────
REQUIRED = [
    "experiment_protocol_defi.py",
    "experiment_protocol_defi_20.py",
    "experiment_protocol_nguyen12.py",
    "experiment_protocol_all_18_a.py",
    "experiment_protocol_all_20.py",
    "experiment_protocol_all_30.py",
    "experiment_protocol_benchmark.py",
    "experiment_protocol_benchmark_v2.py",
    "experiment_protocol_comparative.py",
]


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]  # scripts/patches/ → repo root
    proto_dir = repo_root / "hypatiax" / "protocols"

    print(f"  Checking hypatiax/protocols/ at: {proto_dir}")

    if not proto_dir.exists():
        print(f"  ✗ Directory not found: {proto_dir}")
        print("  Ensure the repository is fully cloned (hypatiax/ must be present).")
        return 1

    missing = []
    for fname in REQUIRED:
        path = proto_dir / fname
        if path.exists():
            print(f"  ✓  {fname}")
        else:
            print(f"  ✗  {fname}  ← MISSING")
            missing.append(fname)

    print()
    if missing:
        print(f"  ERROR: {len(missing)} of {len(REQUIRED)} input-data modules missing "
              f"from hypatiax/protocols/")
        print("  Copy the files into hypatiax/protocols/ and re-run.")
        return 1

    print(f"  hypatiax/protocols/ — all {len(REQUIRED)} input-data modules present ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())

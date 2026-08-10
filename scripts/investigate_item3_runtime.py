#!/usr/bin/env python3
"""
Investigation script template for patch_runner.py.

CONTRACT (required for every script referenced by a manifest item's
`investigation_script` field):
  - Read raw logs/JSON/data. NEVER read or infer from the .tex files.
  - Print exactly one JSON-encoded value to stdout (a number, string, or
    small dict/list if new_str_template needs multiple fields). Put any
    diagnostics/progress output on stderr instead.
  - Exit 0 only if the value is trustworthy. Any nonzero exit means
    patch_runner.py reports this item as SKIPPED rather than patching with
    a placeholder or guess.

This one is a stub for plan-action item #3 (EHSDeFi runtime: 20.2s vs
841.4s). Replace the body with real trace-back logic, e.g.:

    import json, glob

    # Find every raw run that could have produced tab:overall's EHSDeFi row
    candidates = glob.glob(
        "hypatiax/data/results/**/protocol_core_noiseless_*.json",
        recursive=True,
    )
    # ... load each, find the one whose config matches tab:overall's stated
    # conditions, extract EHSDeFi's sigma=0 runtime, decide whether 20.2s or
    # 841.4s (or neither) is correct, print the result:
    print(json.dumps(correct_value))

Until real logic is written here, this stub deliberately exits nonzero so
the runner reports the item as SKIPPED instead of silently patching with
a placeholder.
"""
import sys

sys.stderr.write(
    "investigate_item3_runtime.py: not yet implemented — wire this up to "
    "the raw timing logs (or your own investigation results) before running "
    "the patch.\n"
)
sys.exit(2)

#!/usr/bin/env python3
"""
seed_audit_registry.py — one-time (or re-runnable/idempotent) import of a
PLAN-ACTION-style seed file into audit_registry.json.

Usage:
  python3 scripts/patches/seed_audit_registry.py \
      --seed audit/seed/plan_action_seed.json \
      --registry audit_registry.json

Safe to re-run: only adds issue_ids not already present. Never touches an
existing entry, regardless of its source — if you need to update a seeded
entry after re-investigation, edit audit_registry.json directly (source
stays "seed" so Tier 1 will still never touch it).
"""
from __future__ import annotations

import argparse
import json

import audit_registry_lib as reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True)
    ap.add_argument("--registry", required=True)
    args = ap.parse_args()

    with open(args.seed) as f:
        seed = json.load(f)

    data = reg.load(args.registry)
    existing_ids = {i["issue_id"] for i in data["items"]}

    added = 0
    for item in seed["items"]:
        if item["issue_id"] in existing_ids:
            print(f"skip (already present): {item['issue_id']}")
            continue
        data["items"].append(item)
        added += 1
        print(f"added: {item['issue_id']}  [{item['status']}]")

    reg.save(args.registry, data)
    print(f"\nAdded {added} of {len(seed['items'])} seed items to {args.registry}.")

    blocked = reg.blocked_labels(data)
    if blocked:
        print(f"\n{len(blocked)} label(s) now blocked from Tier-1 auto-fix:")
        for label, entry in blocked.items():
            print(f"  {label!r} -> blocked by {entry['issue_id']}")


if __name__ == "__main__":
    main()

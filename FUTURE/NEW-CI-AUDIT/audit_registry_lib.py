#!/usr/bin/env python3
"""
audit_registry_lib.py — shared logic for reading/writing audit_registry.json.

Schema v2 (list-of-items, not the v1 per-claim-hash dict this replaces):

{
  "schema_version": 2,
  "items": [
    {
      "issue_id": "14_ground_truth_leakage_defi_hybrid",
      "title": "...",
      "document_position": {"file": "...", "sections": [...], ...},
      "index_category": "OPEN blocked-pending" | "TEXT" | "STALE" | "OPEN" | "Already Resolved (doc only)" | ...,
      "audit_status": "free text narrative",
      "type": "text_fix" | "code_investigation" | "blocked_pending" | "stale_doc_sync",
      "fix": "free text",
      "status": "open" | "resolved" | "false_positive" | "blocked_pending",
      "blocks": ["Table 9", "tab:hybrid_all", "Figure 9", ...],   # table/figure
                                                                    # labels that
                                                                    # Tier 1 must
                                                                    # refuse to
                                                                    # auto-fix
                                                                    # while this
                                                                    # entry is open
      "linked_report": "audit/incident_reports/leak_report.tex" | null,
      "source": "seed" | "tier1_auto" | "tier2_upload"
    },
    ...
  ]
}

OWNERSHIP RULE (why this file merges instead of overwrites):
  - source == "tier1_auto"  -> owned by llm_audit_inspector.py. Freely
    updated/resolved/added by each Tier-1 run based on current findings.
  - source == "seed" | "tier2_upload" -> NEVER touched by Tier 1. These come
    from PLAN-ACTION-style seeding or from human+Claude investigation
    sessions uploaded via audit/observations/*.json (type: "block" or
    type: "correction"). Tier 1 only READS them, to build the blocklist and
    to avoid duplicating an issue_id a human has already filed.
"""
from __future__ import annotations

import json
import os

SCHEMA_VERSION = 2


def load(path: str) -> dict:
    if not os.path.exists(path):
        return {"schema_version": SCHEMA_VERSION, "items": []}
    with open(path) as f:
        data = json.load(f)
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path} is schema_version={data.get('schema_version')}, "
            f"expected {SCHEMA_VERSION}. Migrate before running."
        )
    data.setdefault("items", [])
    return data


def save(path: str, data: dict) -> None:
    data["schema_version"] = SCHEMA_VERSION
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def index_by_id(data: dict) -> dict[str, dict]:
    return {item["issue_id"]: item for item in data["items"]}


def blocked_labels(data: dict) -> dict[str, dict]:
    """Returns {label: entry} for every table/figure label currently
    blocked by an open/blocked_pending entry, whatever its source. A label
    like 'Table 9' or 'tab:hybrid_all' or 'Figure 9' is treated as an exact
    string match against whatever gets extracted from the tex (see
    llm_audit_inspector.py's label detection — best-effort, not a full tex
    parser)."""
    out = {}
    for item in data["items"]:
        if item.get("status") not in ("open", "blocked_pending"):
            continue
        for label in item.get("blocks", []):
            out[label] = item
    return out


def upsert_tier1(data: dict, entry: dict) -> None:
    """Insert or update a tier1_auto-owned entry. Never touches seed/
    tier2_upload entries even if issue_id collides — that's a bug in the
    caller, so raise loudly instead of silently clobbering a human entry."""
    existing = next((i for i in data["items"] if i["issue_id"] == entry["issue_id"]), None)
    if existing is not None:
        if existing.get("source") != "tier1_auto":
            raise ValueError(
                f"Refusing to overwrite non-tier1_auto entry '{entry['issue_id']}' "
                f"(source={existing.get('source')}) from Tier 1."
            )
        existing.update(entry)
    else:
        data["items"].append(entry)


def resolve_stale_tier1(data: dict, still_open_ids: set[str]) -> int:
    """Marks any tier1_auto entry not present in this run's fresh findings
    as resolved (the underlying claim is no longer mismatched). Returns
    count resolved. Never touches seed/tier2_upload entries."""
    resolved = 0
    for item in data["items"]:
        if item.get("source") != "tier1_auto":
            continue
        if item["status"] == "open" and item["issue_id"] not in still_open_ids:
            item["status"] = "resolved"
            resolved += 1
    return resolved


def merge_uploaded_observations(data: dict, structured_observations: list[dict]) -> int:
    """Folds audit/observations/*.json entries of type 'block' into the
    registry as tier2_upload entries (type 'correction' entries are handled
    separately by llm_audit_inspector.py's existing structured_override
    path, unchanged from before). Idempotent on issue_id."""
    added = 0
    for obs in structured_observations:
        if obs.get("type") != "block":
            continue
        issue_id = obs.get("issue_id")
        if not issue_id:
            continue
        if any(i["issue_id"] == issue_id for i in data["items"]):
            continue  # already present — human owns updates to their own entry
        data["items"].append({
            "issue_id": issue_id,
            "title": obs.get("title", issue_id),
            "document_position": obs.get("document_position", {}),
            "index_category": obs.get("index_category", "OPEN blocked-pending"),
            "audit_status": obs.get("audit_status", ""),
            "type": obs.get("type_detail", "blocked_pending"),
            "fix": obs.get("fix", ""),
            "status": obs.get("status", "blocked_pending"),
            "blocks": obs.get("blocks", []),
            "linked_report": obs.get("linked_report"),
            "source": "tier2_upload",
        })
        added += 1
    return added

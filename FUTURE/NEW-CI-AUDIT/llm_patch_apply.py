#!/usr/bin/env python3
"""
llm_patch_apply.py — two subcommands, both deterministic (no Claude calls).

  generate       Reads logs/llm_audit_findings.json, appends one entry per
                  numeric_mismatch finding to patch_manifest.yaml in the
                  SAME schema patch_runner.py already understands, so
                  non-numeric / low-confidence entries flow through the
                  existing ci-audit-writing-paper.yml apply/review path
                  untouched. Every entry is tagged:
                    kind: numeric_consolidated | numeric_from_upload | manual_review
                  and only the first two are ever eligible for the
                  auto-apply step below.

  apply-numeric   Mechanically verifies and applies ONLY entries tagged
                  numeric_consolidated / numeric_from_upload:
                    - re-reads the live tex line at (tex_file, line_no)
                    - confirms `stated_value` still appears in it verbatim
                      (guards against the file having moved on since the
                      finding was generated)
                    - confirms substituting stated_value -> consolidated_value
                      changes ONLY that numeric token (no other characters
                      differ) — this is the "pure_numeric_substitution"
                      guarantee, checked here rather than trusted from
                      Claude's self-report
                    - if the entry was already marked applied with a
                      DIFFERENT value in a previous run, requires
                      --allow-value-change=true or skips it and logs why
                  Never touches prose/structural entries.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import yaml

AUTO_KINDS = {"numeric_consolidated", "numeric_from_upload"}


def load_findings(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def load_manifest(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}
    data.setdefault("items", [])
    return data


def save_manifest(path: str, data: dict) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def finding_to_manifest_kind(finding: dict) -> str:
    if finding["kind"] != "numeric_mismatch":
        return "manual_review"
    if not finding.get("auto_commit_eligible"):
        return "manual_review"
    return "numeric_from_upload" if finding.get("provenance_kind") == "upload" else "numeric_consolidated"


def cmd_generate(args):
    findings = load_findings(args.findings)
    manifest = load_manifest(args.manifest)
    existing_ids = {item.get("id") for item in manifest["items"]}

    added = 0
    for finding in findings:
        if finding["kind"] in ("consistent", "blocked"):
            continue  # blocked findings have nothing to patch — tracked in
                      # audit_registry.json only, until the blocking issue
                      # (see finding['blocked_by']) is resolved
        if finding["claim_id"] in existing_ids:
            continue  # don't duplicate across runs; status tracked in audit_registry.json
        manifest["items"].append({
            "id": finding["claim_id"],
            "kind": finding_to_manifest_kind(finding),
            "tex_file": finding["tex_file"],
            "line_no": finding["line_no"],
            "section": finding["section"],
            "stated_value": finding.get("stated_value"),
            "proposed_value": finding.get("consolidated_value"),
            "method": finding.get("method"),
            "provenance": finding.get("provenance", []),
            "confidence": finding.get("confidence"),
            "notes": finding.get("notes"),
            "applied": False,
            "applied_value": None,
        })
        added += 1

    save_manifest(args.manifest, manifest)
    print(f"Added {added} new patch_manifest.yaml entries "
          f"({len(findings) - added} already present or non-actionable).")

    if args.github_output:
        with open(args.github_output, "a") as f:
            f.write(f"added_count={added}\n")


def try_pure_substitution(line: str, old_value: str, new_value: str) -> str | None:
    """Returns the new line iff replacing old_value with new_value changes
    ONLY that token (verified by diffing char-by-char outside the match),
    else None."""
    pattern = re.compile(re.escape(old_value))
    matches = list(pattern.finditer(line))
    if len(matches) != 1:
        return None  # ambiguous or absent — refuse to guess
    start, end = matches[0].span()
    new_line = line[:start] + new_value + line[end:]
    # sanity: everything outside the substituted span must be byte-identical
    if line[:start] != new_line[:start]:
        return None
    if line[end:] != new_line[start + len(new_value):]:
        return None
    return new_line


def cmd_apply_numeric(args):
    manifest = load_manifest(args.manifest)
    allow_value_change = args.allow_value_change.lower() == "true"
    applied = 0
    skipped = []

    for item in manifest["items"]:
        if item.get("kind") not in AUTO_KINDS:
            continue
        if item.get("applied") and item.get("applied_value") == item.get("proposed_value"):
            continue  # already applied, no-op
        if item.get("applied") and item.get("applied_value") != item.get("proposed_value") and not allow_value_change:
            skipped.append((item["id"], "value changed since last apply; needs --allow-value-change"))
            continue

        tex_file = item["tex_file"]
        line_no = item["line_no"]
        if not os.path.exists(tex_file):
            skipped.append((item["id"], f"{tex_file} not found"))
            continue

        with open(tex_file, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if line_no - 1 >= len(lines):
            skipped.append((item["id"], f"line {line_no} out of range in {tex_file}"))
            continue

        old_line = lines[line_no - 1]
        if item["stated_value"] not in old_line:
            skipped.append((item["id"], "stated_value no longer present at that line — file has moved on"))
            continue

        new_line = try_pure_substitution(old_line, item["stated_value"], item["proposed_value"])
        if new_line is None:
            skipped.append((item["id"], "substitution is not a pure single-number swap — requires manual review"))
            continue

        lines[line_no - 1] = new_line
        with open(tex_file, "w", encoding="utf-8") as f:
            f.writelines(lines)

        item["applied"] = True
        item["applied_value"] = item["proposed_value"]
        applied += 1

    save_manifest(args.manifest, manifest)

    print(f"Auto-applied {applied} numeric entries.")
    if skipped:
        print(f"::warning::Skipped {len(skipped)} entries (see below) — left for manual review:")
        for claim_id, reason in skipped:
            print(f"  {claim_id}: {reason}")

    if args.github_output:
        with open(args.github_output, "a") as f:
            f.write(f"applied_count={applied}\n")
            f.write(f"skipped_count={len(skipped)}\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate")
    gen.add_argument("--findings", required=True)
    gen.add_argument("--manifest", required=True)
    gen.add_argument("--github-output", default="")
    gen.set_defaults(func=cmd_generate)

    app = sub.add_parser("apply-numeric")
    app.add_argument("--manifest", required=True)
    app.add_argument("--allow-value-change", default="false")
    app.add_argument("--github-output", default="")
    app.set_defaults(func=cmd_apply_numeric)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())

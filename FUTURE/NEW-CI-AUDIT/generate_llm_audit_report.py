#!/usr/bin/env python3
"""
generate_llm_audit_report.py — writes the "open/flagging points" report in
all three forms agreed:
  1. audit/llm_audit_report.md          — cumulative, overwritten each run,
                                            grouped by status across the
                                            whole audit_registry.json
  2. logs/llm_audit_report_<run_id>.md  — this run only, timestamped,
                                            artifact-only (never committed)
  3. <dir>/<texfile>_llm_audit.md       — one per *_patched.tex, committed
                                            next to the file it describes
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from collections import defaultdict


def load(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt_finding(f: dict) -> str:
    lines = [f"- **{f['tex_file']}:{f['line_no']}** (`{f['section']}`) — {f['kind'].upper()}"]
    if f["kind"] == "numeric_mismatch":
        lines.append(f"  - paper states: `{f['stated_value']}` -> consolidated: `{f.get('consolidated_value')}`")
        lines.append(f"  - method: {f.get('method') or 'n/a'}")
        lines.append(f"  - confidence: {f.get('confidence') or 'n/a'} | "
                      f"auto-commit eligible: {f.get('auto_commit_eligible')}")
        if f.get("provenance"):
            lines.append(f"  - sources: {', '.join(f['provenance'])}")
    elif f["kind"] == "missing":
        lines.append(f"  - stated value `{f.get('stated_value')}` has no resolvable results")
    elif f["kind"] == "blocked":
        lines.append(f"  - **skipped, no Claude call made** — blocked pending `{f.get('blocked_by')}`")
    if f.get("notes"):
        lines.append(f"  - note: {f['notes']}")
    return "\n".join(lines)


def build_body(findings: list[dict], registry: dict | None, run_id: str | None) -> str:
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    by_kind = defaultdict(list)
    for f in findings:
        by_kind[f["kind"]].append(f)

    lines = ["# Claude paper audit — open / flagging points", ""]
    if run_id:
        lines.append(f"Run: `{run_id}` — generated {ts}")
    else:
        lines.append(f"Generated {ts}")
    lines.append("")
    lines.append(f"| Kind | Count |")
    lines.append(f"|---|---|")
    for kind in ("numeric_mismatch", "missing", "blocked"):
        lines.append(f"| {kind} | {len(by_kind.get(kind, []))} |")
    lines.append("")

    if by_kind.get("blocked"):
        lines.append("## Blocked (skipped — pending Tier-2 investigation)")
        lines.append("")
        lines.append("No Claude call was made for these — consolidating against "
                      "currently-invalid data would produce a confident-looking but "
                      "contaminated number. Resolves automatically once the blocking "
                      "issue's status changes in `audit_registry.json`.")
        lines.append("")
        for f in by_kind["blocked"]:
            lines.append(fmt_finding(f))
            lines.append("")

    if by_kind.get("numeric_mismatch"):
        lines.append("## Numeric mismatches")
        lines.append("")
        for f in by_kind["numeric_mismatch"]:
            lines.append(fmt_finding(f))
            lines.append("")

    if by_kind.get("missing"):
        lines.append("## Missing / unresolvable claims")
        lines.append("")
        for f in by_kind["missing"]:
            lines.append(fmt_finding(f))
            lines.append("")

    if not findings:
        lines.append("No open findings.")
        lines.append("")

    if registry is not None:
        open_count = sum(1 for e in registry.get("entries", {}).values() if e["status"] == "open")
        lines.append(f"---\n\n`audit_registry.json`: {open_count} open of {len(registry.get('entries', {}))} total entries.")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--findings", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--cumulative-out", required=True)
    ap.add_argument("--timestamped-out", required=True)
    ap.add_argument("--per-tex-suffix", required=True)
    args = ap.parse_args()

    findings = load(args.findings) or []
    registry = load(args.registry)
    run_id = os.environ.get("RUN_ID") or os.environ.get("GITHUB_RUN_ID")

    cumulative = build_body(findings, registry, run_id=None)
    os.makedirs(os.path.dirname(args.cumulative_out) or ".", exist_ok=True)
    with open(args.cumulative_out, "w") as f:
        f.write(cumulative)

    timestamped = build_body(findings, registry, run_id=run_id)
    os.makedirs(os.path.dirname(args.timestamped_out) or ".", exist_ok=True)
    with open(args.timestamped_out, "w") as f:
        f.write(timestamped)

    by_tex = defaultdict(list)
    for f in findings:
        by_tex[f["tex_file"]].append(f)
    for tex_file, tex_findings in by_tex.items():
        out_path = os.path.splitext(tex_file)[0] + args.per_tex_suffix
        with open(out_path, "w") as f:
            f.write(build_body(tex_findings, registry=None, run_id=None))

    print(f"Wrote cumulative -> {args.cumulative_out}")
    print(f"Wrote timestamped -> {args.timestamped_out}")
    print(f"Wrote {len(by_tex)} per-tex report(s).")


if __name__ == "__main__":
    main()

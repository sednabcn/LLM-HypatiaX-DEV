#!/usr/bin/env python3
"""
llm_audit_inspector.py — Claude-powered audit of *_patched.tex against
hypatiax/data/results/, parallel to (but independent from) hypatia_inspector.py.

Pipeline per *_patched.tex file:
  1. Extract candidate claims: numeric statements + their surrounding
     sentence, tagged with a stable claim_id (file + line + a hash of the
     sentence, so a claim keeps its identity across whitespace-only diffs).
  2. Resolve each claim to the section of the paper it belongs to (via
     \\section/\\subsection tracking while walking the file) and to the
     hypatiax/data/results/ subdirectory(ies) that section's experiments
     live in.

     ASSUMPTION — adjust `resolve_section_to_results()` to match your real
     paper_targets.json / config/experiments.yml schema. As given here it
     expects paper_targets.json to contain entries like:
       { "section": "10.7", "experiment_ids": ["exp2_extrap", "exp2_pca_4060"] }
     and experiment_ids to map onto RESULTS_ROOT/**/<experiment_id>/ dirs.

  3. If a claim has no resolvable results -> finding type MISSING.
  4. If a claim resolves to results, gather every matching result file
     (JSON summaries, _analysis.json, tables/*.tex — NOT raw shard/log
     files) and call consolidate_results.consolidate() to get Claude to
     produce a single authoritative value + method + provenance for that
     section.
  5. Compare consolidated value to the value currently stated in the tex.
     Equal (within stated precision) -> no finding. Different -> finding
     type NUMERIC_MISMATCH, tagged with enough provenance for the patch
     step to do a mechanical, non-LLM-verified substitution later.
  6. Uploaded observation docs (audit/observations/*.md, *.json) are also
     loaded and passed to Claude as additional context for every claim in
     the section they reference; a *.json observation with an explicit
     claim_id + corrected_value can independently produce a
     NUMERIC_MISMATCH finding of provenance kind "upload".

Writes:
  logs/llm_audit_findings.json   — this workflow's own findings file
  audit_registry.json            — open/resolved/false_positive ledger,
                                    independent of issue_registry.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional

import audit_registry_lib as reg
from consolidate_results import consolidate, ConsolidationResult

NUMBER_RE = re.compile(
    r"(?<![\w.])(\d+\.\d+|\d+)"          # the number itself
    r"(%|[x\u00d7]|\s?s(?:ec)?)?"        # optional unit: %, x/×, s/sec
    r"(?![\w.])"
)
SECTION_RE = re.compile(r"\\(sub)*section\*?\{([^}]*)\}")

# Best-effort table/figure label detection — NOT a full tex parser. Catches
# the common patterns: \label{tab:...}/\label{fig:...} near a table/figure
# environment, and plain "Table N" / "Figure N" prose references. Extend if
# your papers use other conventions. This only needs to be conservative
# enough that a real blocked table doesn't slip through — false positives
# (skipping a claim that wasn't actually blocked) are cheap; false
# negatives (auto-fixing a blocked claim) are not.
LABEL_RE = re.compile(r"\\label\{((?:tab|fig):[^}]+)\}")
PROSE_LABEL_RE = re.compile(r"\b(Table|Figure)\s+(\d+)\b")


@dataclass
class Claim:
    claim_id: str
    tex_file: str
    line_no: int
    section: str
    sentence: str
    stated_value: str
    labels: list = field(default_factory=list)  # e.g. ["tab:hybrid_all", "Table 9"]


@dataclass
class Finding:
    claim_id: str
    tex_file: str
    line_no: int
    section: str
    kind: str  # "numeric_mismatch" | "missing" | "blocked" | "consistent"
    stated_value: Optional[str] = None
    consolidated_value: Optional[str] = None
    method: Optional[str] = None
    provenance: list = field(default_factory=list)  # source files / observation doc
    provenance_kind: str = "consolidated"  # "consolidated" | "upload"
    confidence: Optional[str] = None
    notes: Optional[str] = None
    auto_commit_eligible: bool = False
    blocked_by: Optional[str] = None  # issue_id of the blocking registry entry


def stable_claim_id(tex_file: str, line_no: int, sentence: str) -> str:
    h = hashlib.sha256(f"{tex_file}:{sentence.strip()}".encode()).hexdigest()[:12]
    return f"{os.path.basename(tex_file)}:L{line_no}:{h}"


def extract_claims(tex_path: str) -> list[Claim]:
    claims = []
    current_section = "preamble"
    # Best-effort "which table/figure am I inside" tracker: a \label{tab:...}
    # or \label{fig:...} anywhere in the last ~40 lines is treated as the
    # active label context for numbers that follow, since captions/labels
    # commonly appear after the tabular body or before a figure's numbers.
    active_labels: list[str] = []
    LABEL_CONTEXT_LINES = 40
    label_expiry = 0

    with open(tex_path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    for i, line in enumerate(lines, start=1):
        sec_match = SECTION_RE.search(line)
        if sec_match:
            current_section = sec_match.group(2).strip()
            continue

        lbl_match = LABEL_RE.search(line)
        if lbl_match:
            active_labels = [lbl_match.group(1)]
            label_expiry = i + LABEL_CONTEXT_LINES

        prose_labels = [f"{kind} {num}" for kind, num in PROSE_LABEL_RE.findall(line)]

        for m in NUMBER_RE.finditer(line):
            sentence = line.strip()
            labels = list(active_labels) if i <= label_expiry else []
            labels.extend(prose_labels)
            stated_value = m.group(1) + (m.group(2) or "")
            claims.append(Claim(
                claim_id=stable_claim_id(tex_path, i, sentence),
                tex_file=tex_path,
                line_no=i,
                section=current_section,
                sentence=sentence,
                stated_value=stated_value,
                labels=list(dict.fromkeys(labels)),  # dedupe, preserve order
            ))
    return claims


def load_paper_targets(path: str = "scripts/patches/paper_targets.json") -> dict:
    if not os.path.exists(path):
        print(f"::warning::{path} not found — section->results resolution will be empty.")
        return {}
    with open(path) as f:
        return json.load(f)


def resolve_section_to_results(section: str, results_root: str, paper_targets: dict) -> list[str]:
    """
    ASSUMPTION: see module docstring. Adjust to your real paper_targets.json
    schema. Falls back to a loose substring match against directory names
    under results_root if no explicit mapping is found, so the pipeline
    degrades gracefully rather than silently finding nothing.
    """
    exp_ids: list[str] = []
    for entry in paper_targets.get("sections", []):
        if entry.get("section", "").strip().lower() == section.strip().lower():
            exp_ids.extend(entry.get("experiment_ids", []))

    matched_dirs = []
    for dirpath, dirnames, _ in os.walk(results_root):
        base = os.path.basename(dirpath)
        if exp_ids and base in exp_ids:
            matched_dirs.append(dirpath)
        elif not exp_ids and section.strip().lower() in base.lower():
            matched_dirs.append(dirpath)
    return matched_dirs


def gather_result_files(result_dirs: list[str]) -> list[str]:
    """Only condensed summary files — never raw shard/log/checkpoint noise."""
    wanted_suffixes = ("_analysis.json", "_report.md", "_merged.json", "_summary.json")
    files = []
    for d in result_dirs:
        for dirpath, _, filenames in os.walk(d):
            if os.sep + "tables" + os.sep in dirpath + os.sep or dirpath.endswith("tables"):
                for fn in filenames:
                    if fn.endswith(".tex"):
                        files.append(os.path.join(dirpath, fn))
                continue
            for fn in filenames:
                if fn.startswith("_") and fn.endswith(("analysis.json", "report.md", "merged.json")):
                    files.append(os.path.join(dirpath, fn))
                elif fn.endswith(wanted_suffixes):
                    files.append(os.path.join(dirpath, fn))
    return sorted(set(files))


def load_observations(observations_dir: str) -> tuple[list[str], list[dict]]:
    """Returns (free_text_notes, structured_corrections)."""
    notes = []
    structured = []
    if not os.path.isdir(observations_dir):
        return notes, structured
    for path in sorted(glob.glob(os.path.join(observations_dir, "*"))):
        if path.endswith(".md") or path.endswith(".txt"):
            with open(path, encoding="utf-8", errors="replace") as f:
                notes.append(f"# {os.path.basename(path)}\n{f.read()}")
        elif path.endswith(".json"):
            with open(path) as f:
                data = json.load(f)
                if isinstance(data, list):
                    structured.extend(data)
                else:
                    structured.append(data)
    return notes, structured


def apply_ad_hoc_observation(notes: list[str], observation_text: str, observation_file: str) -> None:
    if observation_text:
        notes.append(f"# ad-hoc (workflow_dispatch)\n{observation_text}")
    if observation_file and os.path.exists(observation_file):
        with open(observation_file, encoding="utf-8", errors="replace") as f:
            notes.append(f"# ad-hoc file: {observation_file}\n{f.read()}")


def structured_override(claim: Claim, structured: list[dict]) -> Optional[dict]:
    for corr in structured:
        if corr.get("claim_id") == claim.claim_id or (
            corr.get("section", "").strip().lower() == claim.section.strip().lower()
            and corr.get("current_text", "") and corr["current_text"] in claim.sentence
        ):
            return corr
    return None


def inspect_all(tex_glob: str, results_root: str, observations_dir: str,
                 observation_text: str, observation_file: str,
                 registry_data: dict) -> tuple[list[Finding], set[str]]:
    """Returns (findings, still_open_claim_ids). registry_data is the
    ALREADY-MERGED registry (seed + tier2_upload + prior tier1_auto
    entries) — this function only reads it for the blocklist and never
    mutates seed/tier2_upload entries itself."""
    paper_targets = load_paper_targets()
    notes, structured = load_observations(observations_dir)
    apply_ad_hoc_observation(notes, observation_text, observation_file)
    blocked = reg.blocked_labels(registry_data)

    findings: list[Finding] = []
    still_open: set[str] = set()
    tex_files = [p for p in glob.glob(tex_glob, recursive=True) if p.endswith("_patched.tex")]
    if not tex_files:
        print(f"::warning::No *_patched.tex files matched glob '{tex_glob}'.")

    for tex_path in tex_files:
        for claim in extract_claims(tex_path):
            blocking_entry = next((blocked[lbl] for lbl in claim.labels if lbl in blocked), None)
            if blocking_entry is not None:
                # Never call Claude for a claim inside a blocked table/figure —
                # consolidating against currently-invalid data would produce a
                # confident-looking but contaminated number.
                findings.append(Finding(
                    claim_id=claim.claim_id, tex_file=claim.tex_file, line_no=claim.line_no,
                    section=claim.section, kind="blocked",
                    stated_value=claim.stated_value,
                    blocked_by=blocking_entry["issue_id"],
                    notes=f"Skipped — blocked pending {blocking_entry['issue_id']}: "
                          f"{blocking_entry.get('fix', '')}",
                ))
                continue

            override = structured_override(claim, structured)
            if override and override.get("corrected_value"):
                findings.append(Finding(
                    claim_id=claim.claim_id, tex_file=claim.tex_file, line_no=claim.line_no,
                    section=claim.section, kind="numeric_mismatch",
                    stated_value=claim.stated_value,
                    consolidated_value=str(override["corrected_value"]),
                    method="uploaded correction",
                    provenance=[override.get("source", "audit/observations upload")],
                    provenance_kind="upload",
                    confidence="high",
                    notes=override.get("note"),
                    auto_commit_eligible=True,
                ))
                continue

            result_dirs = resolve_section_to_results(claim.section, results_root, paper_targets)
            if not result_dirs:
                findings.append(Finding(
                    claim_id=claim.claim_id, tex_file=claim.tex_file, line_no=claim.line_no,
                    section=claim.section, kind="missing",
                    stated_value=claim.stated_value,
                    notes="No hypatiax/data/results subdirectory resolves to this section.",
                ))
                continue

            result_files = gather_result_files(result_dirs)
            if not result_files:
                findings.append(Finding(
                    claim_id=claim.claim_id, tex_file=claim.tex_file, line_no=claim.line_no,
                    section=claim.section, kind="missing",
                    stated_value=claim.stated_value,
                    notes=f"Result dirs resolved ({result_dirs}) but contained no summary files.",
                ))
                continue

            consolidation: ConsolidationResult = consolidate(
                section=claim.section,
                sentence=claim.sentence,
                stated_value=claim.stated_value,
                result_files=result_files,
                extra_notes=notes,
            )

            if consolidation.value is None:
                findings.append(Finding(
                    claim_id=claim.claim_id, tex_file=claim.tex_file, line_no=claim.line_no,
                    section=claim.section, kind="missing",
                    stated_value=claim.stated_value,
                    notes=consolidation.notes,
                ))
                continue

            if consolidation.value.strip() == claim.stated_value.strip():
                continue  # consistent — no finding emitted

            findings.append(Finding(
                claim_id=claim.claim_id, tex_file=claim.tex_file, line_no=claim.line_no,
                section=claim.section, kind="numeric_mismatch",
                stated_value=claim.stated_value,
                consolidated_value=consolidation.value,
                method=consolidation.method,
                provenance=consolidation.source_files,
                provenance_kind="consolidated",
                confidence=consolidation.confidence,
                notes=consolidation.notes,
                auto_commit_eligible=(consolidation.confidence == "high"
                                       and consolidation.pure_numeric_substitution),
            ))

    still_open = {f.claim_id for f in findings if f.kind != "consistent"}
    return findings, still_open


def finding_to_tier1_entry(f: Finding) -> dict:
    """Converts a Finding into a tier1_auto registry item. Kept distinct
    from seed/tier2_upload entries by issue_id namespace (claim_id, which
    always contains the tex filename + line + hash) and by source."""
    title = {
        "numeric_mismatch": f"Numeric drift: {f.tex_file}:{f.line_no}",
        "missing": f"No resolvable results: {f.tex_file}:{f.line_no}",
        "blocked": f"Blocked (see {f.blocked_by}): {f.tex_file}:{f.line_no}",
    }.get(f.kind, f"{f.kind}: {f.tex_file}:{f.line_no}")

    return {
        "issue_id": f.claim_id,
        "title": title,
        "document_position": {"file": f.tex_file, "line": f.line_no, "sections": [f.section]},
        "index_category": "OPEN (mechanical)" if f.kind == "numeric_mismatch" else "OPEN",
        "audit_status": f.notes or "",
        "type": "code_investigation" if f.kind == "numeric_mismatch" else "text_fix",
        "fix": f"stated={f.stated_value} -> consolidated={f.consolidated_value}" if f.consolidated_value else "",
        "status": "open",
        "blocks": [],
        "linked_report": None,
        "source": "tier1_auto",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--observations-dir", required=True)
    ap.add_argument("--tex-glob", required=True)
    ap.add_argument("--findings-out", required=True)
    ap.add_argument("--registry-out", required=True)
    ap.add_argument("--observation-text", default="")
    ap.add_argument("--observation-file", default="")
    ap.add_argument("--github-output", default="")
    args = ap.parse_args()

    # Load the EXISTING registry (seed + tier2_upload + prior tier1_auto
    # entries all persist here across runs — this step never starts fresh).
    registry_data = reg.load(args.registry_out)

    # Fold in any new *.json observations of type "block" as tier2_upload
    # entries before inspection runs, so a block filed in the same push
    # that updates hypatiax/data/results takes effect immediately.
    _, structured = load_observations(args.observations_dir)
    added_blocks = reg.merge_uploaded_observations(registry_data, structured)
    if added_blocks:
        print(f"Merged {added_blocks} new block entry(ies) from audit/observations/*.json")

    findings, still_open_ids = inspect_all(
        tex_glob=args.tex_glob,
        results_root=args.results_root,
        observations_dir=args.observations_dir,
        observation_text=args.observation_text,
        observation_file=args.observation_file,
        registry_data=registry_data,
    )

    for f in findings:
        reg.upsert_tier1(registry_data, finding_to_tier1_entry(f))
    resolved_count = reg.resolve_stale_tier1(registry_data, still_open_ids)

    os.makedirs(os.path.dirname(args.findings_out) or ".", exist_ok=True)
    with open(args.findings_out, "w") as f:
        json.dump([asdict(x) for x in findings], f, indent=2)
    reg.save(args.registry_out, registry_data)

    open_count = sum(1 for i in registry_data["items"] if i["status"] in ("open", "blocked_pending"))
    blocked_count = sum(1 for f in findings if f.kind == "blocked")
    print(f"Findings: {len(findings)} | Open in registry: {open_count} | "
          f"Blocked from auto-fix: {blocked_count} | Auto-resolved stale: {resolved_count}")

    if args.github_output:
        with open(args.github_output, "a") as f:
            f.write(f"findings_count={len(findings)}\n")
            f.write(f"open_count={open_count}\n")
            f.write(f"blocked_count={blocked_count}\n")


if __name__ == "__main__":
    sys.exit(main())

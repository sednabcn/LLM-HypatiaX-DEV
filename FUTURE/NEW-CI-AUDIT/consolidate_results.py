#!/usr/bin/env python3
"""
consolidate_results.py — the ONLY module in this pipeline that calls the
Claude API. Given a paper claim (section + sentence + stated numeric value)
and the set of hypatiax/data/results/ summary files that section maps to,
asks Claude to consolidate them into a single authoritative value.

Design notes:
  - Only condensed summary files are sent (see gather_result_files in
    llm_audit_inspector.py) — never raw shard/log/checkpoint files — to
    keep context bounded and cost predictable regardless of how many reruns
    an experiment has accumulated.
  - The prompt forces STRICT JSON output so this stays a deterministic,
    parseable step in an automated pipeline, not a chat response.
  - `pure_numeric_substitution` is Claude's own self-report of whether the
    fix is a single-number swap vs. something that also needs prose
    rewording. The CALLER (llm_patch_apply.py) does NOT trust this flag for
    auto-commit eligibility on its own — it re-verifies mechanically with a
    regex diff of old vs. proposed line before ever writing to a
    *_patched.tex file. This flag only affects whether the finding is
    *offered* for the numeric fast path at all.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import anthropic

MODEL = "claude-sonnet-4-6"
MAX_FILE_CHARS = 4000  # per source file, to bound prompt size
MAX_FILES = 12

SYSTEM_PROMPT = """You are consolidating experimental results for a scientific \
paper audit. You will be given:
  - the paper section name
  - the sentence in the paper stating a numeric claim
  - the value currently stated in the paper
  - the content of one or more result summary files (JSON or generated .tex \
tables) that are the ground truth for that section

Your job: determine the single authoritative value that should be cited for \
this claim, based ONLY on the provided result files. If files disagree (e.g. \
multiple reruns), state which one is authoritative and why (e.g. "most \
recent run", "matches the methodology described in the sentence", \
"aggregate/mean across N seeds as the paper's own methodology section \
specifies"). Do not use outside knowledge of the paper's topic — reason only \
from what's given.

Respond with STRICT JSON ONLY, no markdown fences, matching exactly:
{
  "value": "<the authoritative value as it should appear in the tex, or null \
if the provided files do not actually support this claim>",
  "method": "<one sentence: how you derived/chose this value>",
  "confidence": "high" | "medium" | "low",
  "pure_numeric_substitution": true | false,
  "source_files": ["<filenames you relied on>"],
  "notes": "<anything the human reviewer should know, or null>"
}

confidence must be "high" only if the source files unambiguously support a \
single value with no material disagreement between them. Use "low" if the \
files are contradictory, incomplete, or you are inferring across a \
methodology gap.
"""


@dataclass
class ConsolidationResult:
    value: str | None
    method: str | None = None
    confidence: str | None = None
    pure_numeric_substitution: bool = False
    source_files: list = field(default_factory=list)
    notes: str | None = None


def _read_truncated(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError as exc:
        return f"[could not read {path}: {exc}]"
    if len(content) > MAX_FILE_CHARS:
        content = content[:MAX_FILE_CHARS] + f"\n...[truncated, {len(content)} chars total]"
    return content


def _build_user_prompt(section: str, sentence: str, stated_value: str,
                        result_files: list[str], extra_notes: list[str]) -> str:
    parts = [
        f"## Paper section\n{section}",
        f"## Sentence containing the claim\n{sentence}",
        f"## Value currently stated in the paper\n{stated_value}",
        "## Result summary files",
    ]
    for path in result_files[:MAX_FILES]:
        parts.append(f"### {path}\n```\n{_read_truncated(path)}\n```")
    if len(result_files) > MAX_FILES:
        parts.append(f"(+{len(result_files) - MAX_FILES} more result files not shown, truncated for size)")
    if extra_notes:
        parts.append("## Uploaded observation/audit notes (context only, may not apply to this claim)")
        parts.extend(extra_notes[:5])
    return "\n\n".join(parts)


def consolidate(section: str, sentence: str, stated_value: str,
                 result_files: list[str], extra_notes: list[str] | None = None) -> ConsolidationResult:
    extra_notes = extra_notes or []
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    user_prompt = _build_user_prompt(section, sentence, stated_value, result_files, extra_notes)

    resp = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw_text = "".join(block.text for block in resp.content if block.type == "text")
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        return ConsolidationResult(
            value=None, confidence="low",
            notes=f"Claude response was not valid JSON: {raw_text[:300]}",
        )

    return ConsolidationResult(
        value=data.get("value"),
        method=data.get("method"),
        confidence=data.get("confidence"),
        pure_numeric_substitution=bool(data.get("pure_numeric_substitution")),
        source_files=data.get("source_files", []),
        notes=data.get("notes"),
    )

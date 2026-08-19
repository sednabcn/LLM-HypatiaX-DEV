# audit/observations/

Human-supplied observation / audit notes consumed by `ci_llm_paper_audit.yml`.
Both delivery mechanisms are supported, and both formats:

## A. Committed files (picked up automatically on push)

- **`*.md` / `*.txt`** — free-text notes. Passed to Claude as extra context
  for every claim in the section they're relevant to. Cannot, on their own,
  trigger a numeric auto-commit (too unstructured to mechanically verify) —
  they inform the consolidation call and any resulting suggestion is written
  to `patch_manifest.yaml` as `manual_review`.

- **`*.json`** — structured corrections, eligible for the
  `numeric_from_upload` auto-commit fast path when they include an explicit
  value and source:

  ```json
  [
    {
      "claim_id": "jmlr_paper_main_patched.tex:L482:9f3a2c1b0e77",
      "section": "10.7",
      "current_text": "achieves 91.7% accuracy",
      "corrected_value": "92.1%",
      "source": "rerun 2026-08-14, protocol_core_noiseless_20260814.json",
      "note": "Supersedes prior run after fixing seed leakage (FIX-N3)."
    }
  ]
  ```

  `claim_id` is optional — if omitted, matching falls back to
  `section` + a substring match of `current_text` against the live tex
  sentence. Prefer supplying `claim_id` (visible in
  `logs/llm_audit_findings.json` / `audit/llm_audit_report.md` from a prior
  run) for an exact match.

## B. Ad-hoc, single-run (workflow_dispatch, no commit required)

Run `ci_llm_paper_audit.yml` manually with either:
- `observation_text` — paste free text directly
- `observation_file` — path to a file already in the repo (e.g. one you
  push in the same branch first, or an existing file under this directory)

Ad-hoc inputs are treated the same as a `.md` note: additional context only,
not independently auto-commit eligible.

## C. Filing a block (Tier-2 investigation output)

When a human+Claude investigation session (outside CI — see the leak-report
pattern) finds something that invalidates a table/figure until a rerun or
further fix happens, file it as a `type: "block"` entry in a `*.json` file
here. This is the ONLY way a table/figure gets added to Tier 1's blocklist —
Tier 1 never creates or resolves these itself.

```json
[
  {
    "type": "block",
    "issue_id": "14_ground_truth_leakage_defi_hybrid",
    "title": "Ground-truth answer leakage into the hybrid arm's LLM prompt",
    "document_position": {
      "file": "jmlr_paper_main.tex",
      "sections": ["Hybrid Decision-Attribution Bug: Quantification"]
    },
    "index_category": "OPEN blocked-pending",
    "audit_status": "Confirmed from source across 3 scripts. Full 5-seed rerun required.",
    "type_detail": "blocked_pending",
    "fix": "Full rerun across all 5 seeds; regenerate every blocked table/figure from rerun output only.",
    "status": "blocked_pending",
    "blocks": ["Table 9", "Table 10", "tab:hybrid_all", "Figure 9"],
    "linked_report": "audit/incident_reports/leak_report.tex"
  }
]
```

`blocks` entries must match how `llm_audit_inspector.py` detects labels:
either a `\label{tab:...}`/`\label{fig:...}` value, or a plain `"Table N"` /
`"Figure N"` prose reference (see `LABEL_RE` / `PROSE_LABEL_RE` in that
script — extend those patterns if your papers use other label conventions).

To unblock, edit the entry's `status` in `audit_registry.json` directly
(e.g. to `"resolved"`) once the rerun/fix is committed — Tier 1 will only
ever read this file, so removing the block is a manual, reviewed action.

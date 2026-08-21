# Correction Memo: Runtime Table (`tab:timing_full`) — Second, Independent Defect Found and Resolved

**For:** Paper authors / PI record
**From:** Pipeline audit (follow-on to Issue 17, "runtime.tex / timing_detail.tex
Do Not Reproduce From Their Own Cited Source")
**Status:** Closed — table regenerated and independently verified against raw data.

## Summary

A prior audit pass correctly retracted a data-completeness concern (the
worry that seed-suffixed result files for seeds other than 42 were
truncated to 5 tasks) and published a "mechanically regenerated"
`tab:timing_full`, claiming the hybrid system was $2.5\times$–$9.3\times$
slower than the neural baseline, with three PCA-directed seeds (123, 777,
2024) reported as routing *zero* tasks to the LLM.

**That published table was itself wrong**, independent of the
data-completeness question. It was not caught at the time because the
retraction of the truncation concern was correct on its own terms (the
raw files genuinely are complete), which made the table built on top of
them look trustworthy by association. The table itself was never
actually checked against the files it claimed to regenerate from.

## What was found

Re-running `generate_table1.py` — the paper's own designated,
no-manual-editing regeneration script — directly against all ten raw
result files (v3c and PCA-directed splits, seeds 42/99/123/777/2024, 74
tasks each) gives numbers that disagree with the published table on
**every seed, in both splits**, in a way too systematic to be noise:

- **PCA seeds 99, 123, 777, 2024**: published as routing 0–26 of 74 tasks
  to the LLM, with Pure LLM mean times collapsed to 0.19–2.97s. The raw
  files show 68–69 of 74 tasks routed to the LLM on every one of these
  seeds, with Pure LLM mean times of 9.25–9.81s — indistinguishable from
  seed 42 and from every v3c seed. The "zero-routing" pattern in the
  published table is not a property of the data; it matches the exact
  fingerprint of the truncated-file defect that a previous pass had
  already investigated and ruled out for these same files.
- **v3c split, all seeds**: published LLM-routed counts (68–69/74) are
  consistently ~14–15 tasks higher than what the raw `decision` field in
  each file actually shows (54–55/74), on every seed with no exceptions.

Data integrity was re-verified directly (not assumed): every one of the
ten files has exactly 74 unique `equation_id`s, a correctly-matching
internal `seed` field, and Pure LLM per-task times in a sane 4–29s range
with no truncation, duplication, or corruption.

## Root cause

Not determined with certainty. Two explanations are consistent with the
evidence and cannot be distinguished from the artifacts available:

1. The retraction paragraph was written correctly (files are complete),
   but `tab:timing_full` was never actually re-run against the
   fixed/verified files — i.e. the table in circulation predated the fix
   it claimed to reflect.
2. A second, separate bug in whatever process produced the published
   table caused it to treat non-seed-42 PCA runs as if truncated, even
   though the underlying files were not.

Distinguishing these would require the commit/run history of whatever
produced the previously-published table, which was not available for
this pass. It does not change the fix: `generate_table1.py`, run fresh
against the verified files, is unambiguous and reproducible.

## What changed

`Table~\ref{tab:timing_full}` and `Table~\ref{tab:timing_llm_routed_full}`
in `jmlr_paper_main_patched.tex` (\S\ref{sec:timing}) have been replaced
with the freshly-verified regeneration. All narrative and headline
mentions of the retracted "$2.5\times$–$9.3\times$ slower" figure
elsewhere in the paper (author's-note correction box, abstract-adjacent
contribution list, conclusion) have been updated to the corrected
**$4.1\times$–$7.7\times$ slower** range. Mentions of the retracted
figure that appear *inside* the historical narrative explaining what the
earlier, wrong claim was have been left as-is — they are accurate
descriptions of a past error, not live results.

No change was needed in `supp_benchmark_report_patched.tex`: it does not
contain a parallel runtime/timing section referencing these figures.

## Why this is being treated as closed rather than another open decision

Unlike the Five-System Table 1 problem, there is no interpretive choice
here about what the table should measure or which of several legitimate
data sources to use — the intended computation (`generate_table1.py`
against the ten raw per-task result files) was never in dispute, and the
raw files themselves check out cleanly. The prior published numbers had
no competing claim to correctness; they simply didn't match the data
they cited. Regenerating mechanically from verified inputs, with the
paper's own already-designated tool, resolves it without requiring a
new judgment call from a human decision-maker.

## Open item

The root-cause question above (why the previously published table
diverged from its own cited source, given the files check out) is not
resolved and is not blocking — flagged here for anyone with access to
the generation history who wants to close that separately.

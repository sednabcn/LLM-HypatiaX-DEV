# Decision Memo: What Should Table 1 (Five-System Comparison) Actually Measure?

**For:** Paper authors / PI sign-off
**From:** Pipeline audit (following "Investigation Report: Table 1 — Data
Source Cannot Be Reproduced," Aug 1 2026)
**Status:** Blocking — `generate_tables.py` cannot be safely patched until
this is answered.

## Why this memo exists

The investigation report established, with high confidence, that
`FIVE_SYSTEM_PAPER_ROWS` — the hardcoded constant `gen_five_system()` falls
back to — **cannot be reproduced by any data source currently in the repo**,
and that patching the fallback to point at either of the two candidate
pipelines examined (exp2 Feynman suite, or the DeFi 73-case harness) would
silently swap the published numbers for a different, differently-computed
set without resolving *why* they disagree. That's not a decision engineers
should make unilaterally — it's a scope question about what the paper's
claim is supposed to be. This memo lays out the concrete options so that
decision can be made quickly and on the record.

## The core question

**What is Table 1 supposed to demonstrate, and over what problem set?**

Three options, with what's actually available for each:

### Option A — Feynman-only (physics equations, 6-method exp2 suite)

Real, currently-running data exists: `exp2_five` / `exp2_extrap`, 30
equations across 10 physics/science domains, 6 methods.

Constraints if you choose this:
- **Neural Network can never have extrapolation data from this pipeline** —
  `compute_extrap_r2_far()` structurally skips any method that doesn't return
  a re-evaluable symbolic formula string, and the NN returns an architecture
  tag, not a formula. This isn't a missing-run problem; it's a property of
  the code. If Table 1 needs an NN extrapolation number, this pipeline can't
  supply one under the current method design.
- Two of five paper rows (`System 3 LLM+Fallback`, and NN as above) would
  have **zero** extrapolation samples.
- `Hybrid v50_2`'s real extrapolation error here is ≈125.7% (median), not
  the published 0.0% — i.e., adopting this source changes Hybrid's headline
  number by 2–3 orders of magnitude, in the unfavorable direction.
- Still open: whether `train_r2_mean` should be full-dataset R² (from the
  noiseless exp2 run) or training-split-only R² (from the `--extrap` run) —
  the two exp2 output files disagree with each other on this, unresolved.

### Option B — DeFi-only (finance formulas, 3-method harness)

Real data exists: `extrapolation_73cases_enhanced.json`, 73 DeFi/finance
cases (67 after excluding 6 intractable ones), 3 methods
(`pure_llm`/`neural_network`/`hybrid`).

Constraints if you choose this:
- Only 3 of the paper's 5 systems exist in this harness at all (no
  `System 2 Symbolic` or `System 3 LLM+Fallback` — those would need to be
  added as new runs, not recovered from existing data).
- The *sample sizes* don't match either: every method here returns
  **n=67** non-null scores, not the published n=13 (NN) / n=14 (Hybrid).
  Whatever produced 13/14 either used a different filter than anything in
  the current source, or is a different run entirely (see "Open question"
  below) — adopting this pipeline doesn't hand you the published numbers
  for free; you'd still need to find or recreate whatever narrowed 67 → 13/14.
- Method names here (`pure_llm`, `neural_network`, `hybrid`) are a coarser,
  DeFi-specific set — not a direct match to the paper's five named systems
  (e.g. no distinct System 2/System 3).

### Option C — Deliberate cross-domain comparison, kept as one table

This is closest to what the current fallback constant implicitly claims to
be (a single table with all five named systems and Feynman-sounding + DeFi
row semantics mixed). Given the findings above, **this option is not
currently supportable from any live data** — no single pipeline produces
comparable numbers for all five systems on the same problem set, and
splicing two incompatible metric definitions (RMSE-ratio-based
`extrap_error_pct` from candidate A vs. direct `test_r2`/`extrapolation_gap`
from candidate B) into one table is exactly the practice the investigation
report flags as already having produced the current, unreproducible
`FIVE_SYSTEM_PAPER_ROWS`.

### Option D — Split into two tables (report's suggestion #2)

If the intent really is "show both a physics-equation comparison and a
DeFi-equation comparison," the report suggests making that explicit rather
than presenting one table that looks single-source. Concretely:
- **Table 1a** — Feynman/exp2, 6 methods, with the NN-extrapolation
  limitation stated in the caption (not silently left as `---`).
- **Table 1b** — DeFi/73-case harness, whichever 3–5 methods actually exist
  there, with `n` reported honestly per method.
- Drops the "five systems, one number each" narrative the current table
  implies, in favor of two narrower, fully-reproducible claims.

## What is *not* an option

Patching `load_five_system_data()`/`rows_from_data()` to point at exp2 or
exp2_extrap as a drop-in replacement for the fallback, without doing one of
A–D above first. This was explicitly the report's warning: it would
misrepresent the table as "regenerated from real data" while still
measuring a different metric, on different problems, for at least the
Hybrid v50_2 row, and simply cannot populate two of the five rows at all.

## Open questions that need answering regardless of which option is chosen

1. **Where did the published n=13 (NN) / n=14 (Hybrid v50_2) / 0.0% (Hybrid
   error) numbers actually come from?** No script in the current repo, run
   today, reproduces them. Candidates not yet checked (per the report):
   - an older git-history version of `run_hybrid_system_benchmark.py` with a
     different `INTRACTABLE_CASES` set or clip/threshold rule,
   - a manually curated subset,
   - a run whose output file was deleted or never committed.
   `git log -p` on `run_hybrid_system_benchmark.py` and `generate_tables.py`
   (specifically the commit that introduced `FIVE_SYSTEM_PAPER_ROWS`) is the
   suggested next step if this is worth pursuing before falling back to a
   fresh re-run.
2. **Is `train_r2_mean` meant to be full-dataset R² or training-split-only
   R²?** Affects Option A regardless of everything else.

## Recommended immediate action

Given the magnitude of what's at stake (the table's headline claim — Hybrid
extrapolation error 0.0% — is off from the closest real measurement by
2–3 orders of magnitude, in the wrong direction), we'd recommend:

1. Answer the "where did 13/14/0.0% come from" question first (§ above,
   bullet 1) — it's a bounded git-archaeology task and may resolve
   everything else for free if a real, complete source turns up.
2. If that search comes back empty, default to **Option D (split tables)**
   using the real exp2 and DeFi-harness data, with every limitation (NN's
   structural inability to report extrapolation under exp2, DeFi harness's
   3-method-only coverage, `train_r2_mean` definition) stated in the
   captions rather than silently absorbed into a single spliced row.
3. Do not merge any assembled "five-system" section (including the one
   produced earlier in this pipeline-repair pass, from `five_system.tex` /
   `five_system_exp2five.tex` and siblings) into the paper as a *replacement*
   for Table 1 until (1)–(2) are resolved. It's useful supporting material
   for this decision, not a publication-ready substitute.

---
*Sign-off needed from:* _______________________  *Date:* _______________
*Decision (A/B/C/D):* _______________

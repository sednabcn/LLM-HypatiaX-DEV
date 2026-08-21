# Fix/Writing Job Schedule — Time-Optimized, Hierarchical Order

*Source issues: `report-wrong.tex` (13 confirmed discrepancies across `jmlr_paper_main.tex`, `supp_benchmark_report.tex`, `supp_routing_improvements.tex`).*

**Ordering principle:** start the slowest, most blocking jobs first and let them run in the background; fill the wait time with the fastest, dependency-free fixes; save open-ended investigation for after the mechanical backlog is cleared; close out once the background jobs return; finish with one consistency pass.

---

## Phase 0 — Launch long-running jobs first (start immediately)

Nothing else depends on *starting* these, only on their *output*. Queue them now so they run while everything else happens.

| # | Job | Why it's Phase 0 |
|---|---|---|
| 9 | Full corrected decision-attribution re-run | Mean R² fix is explicitly blocked on this; longest job in the set |
| 4 | Re-run CI computation against the disclosed SD/n | Same harness as #9 — cheap to queue alongside it |
| 3 | Trace/re-run to locate the source of the 20.2s EHSDeFi figure | Needs a re-run against raw timing logs |

**Action:** kick off all three before doing anything else.

---

## Phase 1 — Pure text fixes, zero dependencies (work these while Phase 0 runs)

All in `jmlr_paper_main.tex`; batch into a single editing pass.

| # | Fix |
|---|---|
| 5 | Attach seed/split to every "68 of 74" citation |
| 6 | Define "v3c" and "PCA-directed" as two explicit, named split variants |
| 7 | Correct the Remark 9 tier-count arithmetic (+2 Medium/+1 Hard, not "+3 Hard") |
| 13 | Rename the near-duplicate liquidation-price task labels for clarity |
| 1 (partial) | Hand re-tally the Nguyen-12 Success row against its own printed per-equation cells |

**Action:** clear this whole list in one sitting — no waiting, no decisions required.

---

## Phase 2 — Mechanical fixes, source already known

Root cause identified for each; execution only.

| # | Fix |
|---|---|
| 11 | Overwrite 3 stale `claude-sonnet-4-20250514`/temp entries → confirmed `claude-sonnet-5`/0.25 |
| 12 | One-line fix in `generate_table1.py` (full-precision, not display-rounded division) + regen Table 5/6 |
| 8 | Regenerate Figure 3 from the same `_merged.json` already used for Table 8 |

**Action:** do these right after Phase 1, while still in "editing/execution mode."

---

## Phase 3 — Requires investigation before a fix can be written

Open-ended; do these once the mechanical backlog is clear so investigation isn't competing with quick wins.

| # | Investigation needed |
|---|---|
| 2 | Identify which run produced the abstract's 27/30 M4 figure; confirm same run as Table 4 |
| 10a | Reconcile 900s vs. 1100s config bleed between two write-ups |
| 10b | Check whether 300s timeout enforcement is actually broken in the harness code |

---

## Phase 4 — Close out once Phase 0 jobs return

| # | Action |
|---|---|
| 9 | Fold corrected mean R² into `tab:main_results` and its caption |
| 4 | Fold re-verified CI into Table 1 |
| 3 | Fold traced runtime source into `tab:overall` |

---

## Phase 5 — Final consistency pass (last, touches everything above)

- Re-check #1's per-equation values against raw JSON (not just the hand recount), now that other tables were regenerated the same way — for methodological consistency.
- One full read-through of both papers for any cross-reference now stale because of the edits above (e.g., anything still citing the old 90.5% / 0.8721 / 3.40× figures elsewhere in prose).

---

## Why this ordering saves time

- **Phase 0 concurrency:** ~3 background jobs run while ~45 minutes of Phase 1 + Phase 2 hands-on editing happens — by Phase 4 the slow jobs have likely already returned, so there's no idle waiting.
- **Fast-before-slow within each phase:** zero-dependency text fixes (Phase 1) come before mechanical-but-manual fixes (Phase 2), which come before open-ended investigation (Phase 3) — cheapest wins banked first.
- **Single consistency pass at the end** instead of after every phase — avoids re-reading the same sections multiple times as numbers keep changing underneath.

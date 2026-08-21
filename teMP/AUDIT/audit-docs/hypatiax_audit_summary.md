# HypatiaX Paper Trail — Consolidated Audit Summary

Synthesizing: *Supplementary Material A* (routing improvements PDF), *Consolidated
Reproducibility & Audit Supplement* (`supplementary_consolidated_audit`), and the
*Unconfirmed, Withdrawn, and Stale Claims* quarantine document. These three are
mutually consistent as a system: the quarantine doc is the ledger, the consolidated
audit is the "what's been fixed since" update, and Supp. A is one of the documents
being audited.

---

## 1. The single most important fact across all three documents

**The paper's headline result is reversed, and this is now fully acknowledged in
all three documents, not just disclosed once and forgotten.**

- Original claim: Hybrid beats Pure LLM, 90.5% vs. baseline (a "+28.3pp" story).
- Root cause found: a "decision-attribution bug" — `hybrid.success = True` was set
  whenever *any* numeric r² could be located, without checking whether the
  sub-method named by `hybrid.decision` actually succeeded. This inflated 22/74
  (seed 42) near-perfect "successes" that the pipeline's own records show had
  failed.
- Corrected figure: **45/74 = 60.8%**, *below* Pure LLM's 62.2% on the same run.
- Status across documents: consistent. The consolidated audit's tracker item A
  confirms this is now the story stated in all 7 places in the main manuscript
  where 89.2%/90.5% used to conflict (a separate, now-resolved internal
  inconsistency). The quarantine doc treats the DeFi 1.73×/89.2%/62.2% figures as
  **"not independently audited by this review"** — i.e., that review scope was
  Feynman/Core-15, not DeFi — so the DeFi correction's provenance rests on
  Supp. A / the main paper's own "Hybrid Decision-Attribution Bug" section, not
  on independent re-derivation by the quarantine reviewers.
- **Caveat not to lose**: the quarantine doc's DeFi section says the *mechanism*
  (train_r2-based undercounting) originally hypothesized was never confirmed —
  what's confirmed instead is the *opposite*-direction overcounting bug, and a
  drafted fix for it **has not been executed against any live run**. So even the
  corrected 60.8%/62.2% comparison should be read as "root cause identified and
  demonstrated on 5/5 seeds for one task," not as "recomputed end-to-end and
  reconfirmed."

**Action if citing this paper**: cite 60.8% vs 62.2%, not 90.5%. Do not treat the
underlying fix as validated by a full corrected re-run — that re-run is still
described as pending in the newest document.

---

## 2. Runtime / speedup — fully resolved, in the *opposite* direction

All three documents now agree: **no speedup claim survives.**

- Withdrawn: 1.73× (mean) / 1.64× (median) hybrid-faster-than-NN claim, and a
  73% runtime-reduction claim.
- What actually happened, per full 5-seed × 74-task × 2-split reconstruction
  (`generate_table1.py`, no hand-edited cells):
  - Aggressive (v3c) split: hybrid **6.3×–9.3× slower** than NN, on *every single
    seed*, no exceptions.
  - PCA-directed split: **2.5×–6.1× slower**, and even on seeds routing **zero**
    tasks to the LLM, hybrid is still ~2.5× slower than NN — meaning there's a
    fixed hybrid-wrapper overhead independent of LLM calls.
  - The old NN timing (3.0s mean) is 8–9× larger than what raw files show
    (0.34–0.42s) — the audit's read is that the old table came from a different,
    likely older/heavier run, not a computation error.
- Status: **closed/superseded**, tracker item B. Do not resurrect any draft that
  still asserts a positive speedup or treats this as based on a single seed —
  the document explicitly warns against exactly that regression.

---

## 3. Core-15 ablation ("Table 6") — severe fabrication-pattern, resolved

This is the most damning single item in the whole trail, and all three documents
independently converge on it:

| Case | Published | Real (`_merged.json`) |
|---|---|---|
| Arrhenius (far) | −12.5553 (H), "identical collapse" to PySR | **+0.9028** (a success) |
| Michaelis-Menten (far) | P=−83,900, H=−635 ("132× reduction," abstract headline) | P=**+0.647** (success); H is **null** (never run) |
| Rate Law / Logistic Growth (H) | ≈0.9999–1.0000 | **null**, all three ranges |
| Gravitational Force (H) | finite, moderate-negative | **−∞** (genuine crash) |
| Gravitational Force (P) | finite, moderate-negative | **≈0.997–0.9998** (near-perfect) |
| Constant Product (H, med/far) | tracks P ≈0.9996 | **−5.4×10³⁴ / −1.9×10⁴⁶** (collapse) |
| Portfolio Std Dev (P, far) | −118.4482 | **1.0000** (success) |

Notable: errors run in **both directions** — some published numbers make PySR
look artificially worse, others make Hybrid look artificially better — so this
isn't simply "thumb on the scale for the headline method," and the paper's
domain-selectivity narrative ("Physics/Risk favor HypatiaX, Chemistry/AMM favor
PySR-only") was built on these unverified numbers and has been withdrawn along
with any Mann-Whitney stats computed from the table.

- Downstream retractions confirmed: abstract's "132× reduction" headline; the
  Arrhenius/Portfolio "scale-incompatibility" failure-mode narrative; Fig. 9's
  caption, which independently repeated the same fabricated −83,899
  Michaelis-Menten figure as a **third, separately-caught occurrence**; Fig. 18's
  caption, found to describe a different figure's structure entirely.
- Status: **closed** — table rebuilt directly from `_merged.json`, no
  transcription step, cross-checked cell-by-cell in the consolidated audit.

---

## 4. Feynman-30 benchmark — now has a resolved provenance trail, but read the caveats

- Final confirmed figures: **12/30 (40.0%)** random-split, **13/30 (43.3%)**
  PCA-split. These supersede an earlier 9/30 (withdrawn — traced to a file with
  `n_pass: 0, n_total: 0, solve_rate: null`, duplicated byte-for-byte across 15+
  unrelated directories — clearly an unpopulated template, not a real result)
  and an earlier 18/30 PCA figure (superseded once a 5th defect, missing
  `asin_of_sin`, was found and patched).
- **Run-date confusion, now disclosed rather than hidden**: at least four
  distinct PCA-split run dates exist in the repo's history (June 4 — discarded,
  0/30 environment failure; June 29 — file never located; June 30 — feeds an
  unverified, still-untrusted 142/180 pooling artifact; **July 22/23 — the only
  canonical pair**, confirmed directly by the corresponding author). This
  matters because a reader who independently opens the June-30 file would see
  142/180 and could wrongly conclude the paper is internally inconsistent — it
  isn't, but only because of this disclosure.
- **The 142/180 figure itself is separately withdrawn**: confirmed to be an exact
  arithmetic doubling of one 90-record run, not two independent pools.
- **One defect remains permanently unresolvable, not merely unfixed**:
  placeholder tokens (e.g. `x0`, `x1`) baked into 4 random-split stored formulas
  reference variables that were never real inputs — no evaluator patch can
  recover this; root cause (an upstream serialization step) is not yet located.
- **Gaps to flag if citing per-equation detail**: the 13 already-scored
  (not rescored) PCA-split equations' individual names/R² values were never
  captured this round — only the aggregate 10-pass/3-fail split is available.
  Don't assume you can reconstruct which 10 passed.
- The main paper's old per-equation Feynman table used a **completely different
  equation set** (Gaussian, Coulomb Force, relativistic momentum, Doppler shift)
  than the corrected 12/30 and 13/30 runs — it's been relabeled as legacy and two
  new tables generated from the canonical data replace it.

---

## 5. Nguyen-12 pipeline — model identity resolved, discrepancy narrowed but not closed

- Resolved: the benchmark's actual LLM call path is `hypatia.py`, using
  **`claude-sonnet-5` at temperature 0.25** — not `claude-sonnet-4-6` at 0.3 as
  the more commonly referenced `LLMConfig` class would suggest. That confusion
  had a clean explanation: a script sets an `LLM_MODEL` env var that **nothing
  ever reads** — a "looks wired in but isn't" pattern found twice in this
  codebase (also true of a `repro.yaml` field).
- The 91.7%-vs-100% seed-123 discrepancy: **ruled out** as an explanation — seed
  variance and reduced sample size (cross-seed and within-seed checks both
  confirm this). **Narrowed** to a ~4-week configuration-migration window
  (Apr 23–May 22, 2026), plausibly coinciding with a GCP/Kaggle→GitHub Actions
  infrastructure migration — but this is flagged as worth checking, not
  confirmed as the cause.
- Remaining unverified link: `run_all.sh` — the one file that would confirm
  which script CI actually invokes.
- Doesn't change any number currently printed in the paper (which already states
  33.3%/91.7% in prose), but future re-benchmarking should report a distribution
  over repeated seed=123 runs, not a single point figure, given confirmed
  server-side unseeded temperature sampling.

---

## 6. Process/credibility items — do not cite until closed

| Item | Status |
|---|---|
| July 9 CI dashboard ("0 open issues") | **Open.** Two notebooks directly inspected: one never executed (0/10 cells), one stopped mid-run before its own status cell could declare fixes resolved. `build_report.py`'s dashboard logic is independent of whether notebooks ran at all — a stale/cached registry could assert "0 open issues" with nothing having run. Code fix (hard-fail on missing/incomplete notebooks) is applied and verified; **the actual re-run has not happened**. Don't cite this dashboard's conclusions until it has. |
| Baseline-lock 74/90 vs. 81/90 | **Resolved, live-confirmed.** Threshold bug (0.9999 vs. intended 0.999999) inflated legacy baseline; fresh run shows exactly 74/90, reproduced independently a second time via a separately-discovered Gate C schema bug fix. Cite 74/90. |
| Gate C `_iter_rows` schema bug | **Confirmed and fixed** — a `"tests"` container key wasn't recognized, silently collapsing every real result file to `n_total=0`. Verified against synthetic data and a live 12-file real directory, independently reproducing the same 74/90. Porting the fix to two other files still in progress. |
| Manifest truncation `[:N]` bug | Fixed in code; **not yet fully reconfirmed live** — the "5 source files" count is consistent with the fix but hasn't been checked against an actual directory listing. |
| Second, unreconciled `exp2_run.log` | **Still open** — a materially different, non-Feynman-domain benchmark run shares the same filename convention; its relationship to the Feynman results is unresolved. |
| PCA `random_state` documentation | **Not investigated** — tier C, nothing drafted, just flagged. |

---

## 7. Smaller but real issues, fixed

- Routing-cascade figure vs. architecture figure: **System 1/2/3 numbering
  conflicts between two figures in the same manuscript** — flagged, genuinely
  unresolved (not just a caption slip; both figures' own internal legends
  disagree with each other).
- Seed-sweep caption overgeneralized ("collapses on all seeds," "substantially
  higher") when the real chart shows a much narrower, less dramatic range —
  fixed.
- Bibliography: five citations existed only as local `\bibitem`s, not in
  `references.bib` (added, not fabricated); one citation pointed at the wrong
  AI-Feynman paper version (2020 original vs. 2.0 extension) for a claim
  specifically about the 2.0 figure — both fixed.
- Hardware-variant numbers (Celeron 14/30, Colab 8/30): already correctly
  caveated in the main text as belonging to the withdrawn 9/30 run; current CI
  is architecturally incapable of reproducing genuine hardware-variant numbers
  (uniform `ubuntu-latest` runners) — a capability gap, not a correctness issue,
  but worth knowing if someone asks for updated hardware numbers.

---

## 8. What this means for citing the paper today

**Safe to cite as current:**
- Corrected 60.8% (Hybrid) vs. 62.2% (Pure LLM) on the DeFi benchmark.
- Hybrid is 2.5×–9.3× *slower* than NN, not faster, on every seed/split.
- Rebuilt Core-15 ablation table (from `_merged.json`, cross-checked).
- Feynman: 12/30 random-split, 13/30 PCA-split.
- 74/90 legacy baseline-lock figure.
- `claude-sonnet-5` @ temp 0.25 as the actual Nguyen-12 model.

**Do not cite:**
- 90.5% / "+28.3pp Hybrid advantage" (retracted).
- 1.73×/1.64× speedup, or the 73% runtime-reduction claim (retracted, reversed).
- Any Table 6 number from the withdrawn draft, or aggregate stats/Mann-Whitney
  results derived from it.
- The July 9 CI dashboard's "0 open issues" (not yet re-run).
- 142/180 PCA figure, 9/30 Feynman figure, 81/90 baseline (all withdrawn).
- 18/30 PCA-split figure (superseded by 13/30).

**Cite with an explicit caveat:**
- The DeFi decision-attribution bug's *fix* — root cause confirmed, direction
  confirmed on 5/5 seeds for one task, but the correction has not been executed
  against a full live run; treat the 60.8%/62.2% comparison as "diagnosed and
  directionally validated," not "recomputed end-to-end."
- Nguyen-12 91.7%-vs-100% gap — narrowed to a migration window, not fully
  explained.
- Planck blackbody's split-divergence (near-miss on random split, catastrophic
  on PCA split) — the numbers are solid and fully disclosed, but *why* it
  diverges this sharply remains an open mechanistic question.

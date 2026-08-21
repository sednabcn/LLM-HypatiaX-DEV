# Cross-Reference Audit — Main Paper + Both Supplements
Compiled 2026-07-31. Method: every `\label{}` and every ref-like command
(`\ref`, `\S\ref`, `\eqref`, `\pageref`, `\autoref`, `\cref`/`\Cref`) was
extracted from each document, then each usage was checked against the
labels defined in *that same document* (since none of the three files use
`xr`/`\externaldocument`, each is compiled standalone — a `\ref` to a
label that only exists in a different file renders as `??` in the actual
PDF, not a hypothetical). Confirmed by regex extraction **and** by an
actual two-pass `pdflatex` compile of all three files.

## 1. `jmlr_paper_main_3007217.tex` (main paper)
- Labels defined: 192
- Ref-like commands used: 105
- **Dangling: 0**
- Compile-confirmed: after 3 passes, no `Reference ... undefined` warnings
  remain (only sandbox artifacts unrelated to cross-refs: `proof`
  environment undefined because my local stub `jmlr2e.sty` doesn't load
  `amsthm`, and missing `.bib` citations because no bibliography file was
  provided).

## 2. `supp_benchmark_report.tex`
- Labels defined: 60
- Ref-like commands used: 30 (uses `cleveref`; `\cref`/`\Cref` included in
  the check)
- **Dangling: 0**
- Compile-confirmed: after 2 passes, no `Reference ... undefined`
  warnings remain (only sandbox artifacts: `keywords` and `example`
  environments undefined, again from the incomplete local stub, not a
  real document defect).
- Side note, not a cross-ref issue: this file's preamble has
  `\usepackage[expansion=false]{microtype}` — the same setting
  `cross-ref-jul30.txt` flagged as the cause of 61 of 62 overfull-hbox
  warnings elsewhere. Flagging here since it's now directly visible in
  this file, but not fixed as part of this pass (out of scope for a
  cross-ref audit; see that note for the fix and the one remaining
  non-scalable-font complication).

## 3. `supp_routing_improvements.tex`
- Labels defined: 23
- Ref-like commands used: 3
- **Dangling found: 2 — both now fixed**

| Line | Command | Target | Problem | Fix applied |
|---|---|---|---|---|
| 118 | `\ref` | `tab:main_results` | Label only exists in `jmlr_paper_main_3007217.tex`; this file compiles standalone, so it rendered as `Table~??` | Replaced with plain text: "its primary results table, in `jmlr_paper_main.tex`, §‘The Hybrid Decision-Attribution Bug’" |
| 437 | `\ref` | `tab:main_results` | Same issue, different caption | Same fix: replaced with plain text pointer to the same section |

Both fixes verified by recompiling the corrected file twice: zero
`Reference ... undefined` warnings remain (the only residual warning is
the same sandbox-only `proof`-environment artifact seen in the main
paper, unrelated to cross-references).

This matches what `cross-ref-jul30.txt` reported as already done — but
the version of `supp_routing_improvements.tex` actually uploaded still
had both instances unfixed. They are fixed now, in the copy delivered
alongside this list.

## Summary

| Document | Dangling refs found | Fixed |
|---|---|---|
| `jmlr_paper_main_3007217.tex` | 0 | — |
| `supp_benchmark_report.tex` | 0 | — |
| `supp_routing_improvements.tex` | 2 | 2 |

**Total across all three documents: 2 dangling cross-references, both in
`supp_routing_improvements.tex`, both now resolved.**

## Not covered by this pass
Per the earlier scope note: the figures-inventory merge and the
remaining non-scalable-font `pdfTeX` error from `cross-ref-jul30.txt`
still require the real `jmlr2e.sty`, the actual figure files, and the two
figure inventories, none of which have been provided in this session.

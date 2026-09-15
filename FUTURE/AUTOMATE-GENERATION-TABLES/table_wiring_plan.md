# Table-wiring plan for `generate_tables.py` → paper `.tex`

Updated after adding generators for the previously-uncovered tables. Of the
~76 `\label{tab:...}` entries across the three documents, coverage is now:

## Newly added this pass (34 generators)

Classified by how the source was determined — see the block comment above
`gen_timing_full()` in `generate_tables.py` for the full rationale.

**(A) JSON-backed** — real generator, `skip_table()`s if the named source
file isn't present, never fabricates a number:

| Table | Label | Source (as named in the paper's own correction notes) |
|---|---|---|
| `timing_full.tex` | `tab:timing_full` | 10 raw per-seed `hypatiax_defi_benchmark_{v3c,pca}_seed*_results*.json` |
| `timing_llm_routed_full.tex` | `tab:timing_llm_routed_full` | same, LLM-routed subset |
| `hybrid_bug_breakdown.tex` | `tab:hybrid-bug-breakdown` | `hypatiax_defi_benchmark_v3_results_seed42.json` |
| `randomsplit.tex` | `tab:randomsplit` | July 22 random-80/20 per-test JSON |
| `pcasplit.tex` | `tab:pcasplit` | July 23 PCA-directed per-test JSON |
| `hardcoded.tex` | `tab:hardcoded` | `protocol_core_noiseless_*.json`, `is_hardcoded` flag |
| `sota.tex` | `tab:sota` | same, this work's own pass rates only (lit. rows stay static) |
| `nrmse.tex` | `tab:nrmse` | same, `rmse`/`r2` fields |
| `wilcoxon.tex` | `tab:wilcoxon` | same, pairwise Wilcoxon signed-rank (stdlib normal approx) |
| `arch.tex` | `tab:arch` | same, `decision`/`strategy` field tally (**unconfirmed schema** — see comment) |
| `interpolation_stats.tex` | `tab:interpolation_stats` | same |
| `domain_success_detailed.tex` | `tab:domain_success_detailed` | same |
| `llm_detailed.tex` | `tab:llm_detailed` | `hypatiax_defi_benchmark_v3*results*.json` |
| `defi_detailed.tex` | `tab:defi_detailed` | same |
| `noise_sensitivity.tex` | `tab:noise_sensitivity` | `noise_sweep_*.json` |
| `validation_stats.tex` | `tab:validation_stats` | `validation_log*.json` (**path guessed** — no other source named) |
| `fix5_cases.tex` | `tab:fix5_cases` | `fix5_cases*.json` (**path guessed**) |
| `timing_breakdown.tex` | `tab:timing_breakdown` | `*timing_breakdown*.json` (**path guessed**) |
| `scalability.tex` | `tab:scalability` | `scalability_*.json` (**path guessed**) |

**(B) Static/stable** — transcribed from the paper's own already-reviewed
text (same precedent as the pre-existing `version_history.tex`), or read
from a `config/<name>.json` first if one exists:

`provenance.tex`, `methods.tex`, `sweeps.tex`, `software_env.tex`,
`runtime_reproducibility.tex`, `hyperparameters.tex`. `hyperparameters_
complete.tex`, `formulas_detailed.tex`, and `jmlr.tex` try `config/*.json`
first and `skip_table()` (rather than guess) if it isn't there.

**Mechanical (always safe)**: `supplementary_files.tex` and `figlist.tex`
just list what's actually on disk in the output/figures directories.

**(C) Intentionally not automated** — flagged with `skip_table()` and a
comment explaining why, no generator logic written:

- `additional_stats.tex` — no named source anywhere in the paper text.
- `changes.tex` — a hand-authored code changelog, not a measurement.
- `equation_prevalence.tex` — an *estimate* of training-data memorization, not a measurement.
- `conceptual_complexity.tex` — qualitative comparison, no JSON source.
- `feynman30_legacy.tex` — the withdrawn run's source file is confirmed absent from the released artifacts (paper's own text).

**Excluded on purpose, no generator written at all**: `tab:baseline`,
`tab:projected`, `tab:cost_accuracy_tradeoff` in
`supp_routing_improvements_pached_CLEANED.tex` — all three still carry live
`[VALUE REDACTED --- pending re-verification]` or disputed/provisional
markers in the paper text right now. Auto-wiring these would silently
replace carefully-flagged, pending-verification content with fresh,
unreviewed numbers — exactly what this plan already warned against for
`tab:main_results`/`tab:llm_ablation` below. Resolve the redaction first,
by hand, before adding a generator.

## From the previous pass (unchanged)

### Safe to `\input{}` now

| generated_file | label |
|---|---|
| `five_system.tex` | `tab:five_systems_full` |
| `portfolio_sweep.tex` | `tab:portfolio_seed_sweep` |
| `instability.tex` | `tab:instability` |
| `version_history.tex` | *(none expected)* |
| `timing_detail.tex` | *(none expected)* |
| `suppb_noiseless.tex` | `tab:overall` |
| `suppb_r2_noise.tex` | `tab:r2_noise` |
| `suppb_rr_noise.tex` | `tab:rr_noise` |
| `suppb_time_noise.tex` | `tab:time_noise` |
| `suppb_sc_metrics.tex` | `tab:sc_metrics` |
| `suppb_winrate.tex` | `tab:winrate` (paper text itself calls this "unverified pending exact methodology" — input with that caveat still attached) |

### Blocked — do not wire yet

- `defi_tiers.tex` (`tab:difficulty`) — parser still always `skip_table()`s.
- `runtime.tex` (Table 4) — superseded by `timing_full.tex`/`timing_llm_routed_full.tex`, added this pass; wire those instead and drop `runtime.tex`'s claim on `tab:timing`.

### Needs a human decision before wiring

- `defi_main.tex` (`tab:main_results`) and `ablation.tex` (`tab:llm_ablation`) — paper's hand-authored versions carry active `[VALUE REDACTED]` / open reconciliation notes. **Note:** the paper's `tab:llm_ablation` has since been rebuilt with a nan/-inf-aware, no-aggregate-stats format (see the current table in `jmlr_paper_main_patched_CLEANED.tex`); `gen_ablation()` was *not* updated to match this pass — still uses its older always-compute-a-mean-row logic. Needs a follow-up pass before wiring.
- `nguyen12.tex` (`tab:nguyen12`) — never run against real result JSON.

## Not tracked anywhere — separate cleanup

`all30_domain_summary.tex`, `multi_domain_rank_table.tex`,
`hybrid_all_domains_summary.tex`, `domain_rank_table.tex`,
`extrap_ood_table.tex`, `sc_summary.tex`, `sc_by_sample.tex`, and both
`five_system_exp2five*` tables are generated but appear in neither the
paper nor a manifest. Add to the manifest as a deliberate appendix
addition, or remove their generators.

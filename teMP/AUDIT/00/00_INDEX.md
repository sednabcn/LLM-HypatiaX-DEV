# Wrong-Section Report — Index of 13 Extracted Issues

Source: `Wrong-Section Report`, compiled 2026-07-31, covering `jmlr_paper_main.tex`, `supp_benchmark_report.tex`, `supp_routing_improvements.tex`.

Each issue below has been split into its own file so it can be checked against the `.tex` sources by hand without editing them.

| # | Issue | Fix Category | Primary File(s) | File |
|---|-------|---------------|------------------|------|
| 1 | Nguyen-12 success-count arithmetic (10/12, 7-actual, 4/12) | Stale + Open | jmlr_paper_main | `01_nguyen12_success_count_arithmetic.md` |
| 2 | Supp Table 4 (30/30) vs. Abstract (27/30), M4 | Stale | supp_benchmark_report | `02_hypersymloop_30_vs_27.md` |
| 3 | EHSDeFi runtime 20.2 s vs. 841.4 s | Open | supp_benchmark_report | `03_ehsdefi_runtime_20_vs_841.md` |
| 4 | CI/SD statistical incompatibility | Open | jmlr_paper_main / supp_benchmark_report | `04_ci_sd_statistical_incompatibility.md` |
| 5 | 68 of 74 without seed/split | Text | jmlr_paper_main | `05_68_of_74_no_seed_split.md` |
| 6 | v3c vs. PCA-directed undefined | Text | jmlr_paper_main | `06_v3c_vs_pca_directed_undefined.md` |
| 7 | Remark: tier arithmetic (+3 Hard) | Text | jmlr_paper_main | `07_difficulty_tier_counts_remark.md` |
| 8 | Fig. 3 caption −78.04 untraceable + tier clash | Open (source ID'd) | jmlr_paper_main | `08_figure3_caption_defi_amm.md` |
| 9 | Uncorrected Mean R² despite masked failure | Open (pending re-run) | jmlr_paper_main | `09_uncorrected_mean_r2_masked_failure.md` |
| 10 | Timeout contradiction (1100 vs. 900 s; 300 s limit / 27,574 s hang) | Open + Stale | supp_benchmark_report | `10_timeout_contradiction.md` |
| 11 | Model/temperature mismatch | Already Resolved (doc only) | supp_benchmark_report / supp_routing_improvements | `11_model_temperature_mismatch.md` |
| 12 | Rounding slippage 3.36× vs. 3.40× | Open (mechanical) | jmlr_paper_main | `12_rounding_slippage_336_vs_340.md` |
| 13 | Near-duplicate task names | Text | jmlr_paper_main | `13_near_duplicate_task_names.md` |

## Category legend
- **TEXT** — pure writing/citation fix, underlying data is fine
- **STALE** — reconciliation only (e.g. mismatched run outputs, config bleed)
- **OPEN** — needs live code/data investigation
- **ALREADY RESOLVED** — root cause found, only stale doc copies remain

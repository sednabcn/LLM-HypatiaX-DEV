# Issue 4 — CI / SD Statistical Incompatibility

**Category:** [OPEN — needs code/data investigation] — needs live code/data investigation

**Location:** `jmlr_paper_main.tex`, `subsec:fivesystem-comparative-analysis-feynman-` ("Five-System Comparative Analysis (Feynman Core-15)"), `tab:five_systems_full` and `thm:five_system_hierarchy`, l. 1213–1259 vs. `supp_benchmark_report.tex`, Appendix G "Statistical Test Details", `app:statistical_tests`, l. 1423–1471.

## What is wrong
The main paper's Result box states:

> neural networks exhibit mean error 1231% (95% CI: [1087%, 1456%], n = 13)

That gives a CI half-width of ≈ 185 percentage points around a mean of 1231%. Appendix G, describing the same neural-network distribution, states:

> Neural Network: Mean = 1231%, Median = 86.7%, SD = 1842%, Range = [12.3, 5847%] (CORRECTED)

A standard 95% CI on the mean with n = 13 and SD = 1842% has a half-width of roughly t₀.₉₇₅,₁₂ · 1842/√13 ≈ 2.18 × 511 ≈ 1114 percentage points — almost six times narrower a CI is reported in the main text than the disclosed SD supports.

## Fix required
Re-run whatever CI computation produced Table 1's figure and check it against Appendix G's SD; the CI was very likely computed with the wrong SD, wrong n, or an undisclosed non-standard method.

# Audit report (read-only)

## Item 1: benchmark_results.json mismatch
  ./hypatiax/data/results/comparison_results/feynman-tests/exp2/benchmark_results.json: 30 tests, 180 method-rows, 6 methods, 11 domains
    method: EnhancedHybridSystemDeFi (core)
    method: HybridDiscoverySystem v50_2 (tools)
    method: HybridSystemLLMNN all-domains (core)
    method: ImprovedNN (core)
    method: PureLLM Baseline (core)
    method: SymbolicEngineWithLLM (tools)
  recomputed from this file (pass = r2 >= 0.999999):
    EnhancedHybridSystemDeFi (core): pass 23/30, success-flag 30/30, missing/NaN r2 0, mean r2 0.999998
    HybridDiscoverySystem v50_2 (tools): pass 21/30, success-flag 30/30, missing/NaN r2 0, mean r2 0.999473
    HybridSystemLLMNN all-domains (core): pass 30/30, success-flag 30/30, missing/NaN r2 0, mean r2 1.000000
    ImprovedNN (core): pass 0/30, success-flag 30/30, missing/NaN r2 0, mean r2 0.999323
    PureLLM Baseline (core): pass 29/30, success-flag 29/30, missing/NaN r2 0, mean r2 0.966667
    SymbolicEngineWithLLM (tools): pass 27/30, success-flag 30/30, missing/NaN r2 0, mean r2 0.999868
  compare these with the 29/30 and 27/30 claims listed below; any method whose count differs needs its claim fixed.
  per-domain runs in ./hypatiax/data/results/comparison_results/feynman-tests/exp2: 11 files
    protocol_core_noiseless_20260812_053010.json: 3 tests, 6 methods
    protocol_core_noiseless_20260812_053456.json: 2 tests, 6 methods
    protocol_core_noiseless_20260812_053842.json: 1 tests, 6 methods
    protocol_core_noiseless_20260812_054855.json: 5 tests, 6 methods
    protocol_core_noiseless_20260812_055602.json: 2 tests, 6 methods
    protocol_core_noiseless_20260812_055758.json: 1 tests, 6 methods
    protocol_core_noiseless_20260812_060702.json: 4 tests, 6 methods
    protocol_core_noiseless_20260812_061223.json: 2 tests, 6 methods
    protocol_core_noiseless_20260812_061400.json: 1 tests, 6 methods
    protocol_core_noiseless_20260812_062629.json: 5 tests, 6 methods
    protocol_core_noiseless_20260812_063432.json: 4 tests, 6 methods
  union of methods across per-domain runs: 6
  -> six-method data exists in the per-domain runs; a merge can regenerate benchmark_results.json.
  exp2_run.log: 0 line(s) suggesting a method was skipped/unavailable
  claims in supplement that depend on the six-method file: 47
    line 7: six-method
    line 131: six-method
    line 154: 27/30
    line 164: 29/30
    line 165: 27/30
    line 226: 29/30
    line 226: 29/30
    line 228: 27/30
    line 233: 27/30
    line 263: six-method
    line 312: Six-Method
    line 316: six-method
    line 465: Six-Method
    line 469: Six-method
    line 482: 29/30
    line 505: 29/30
    line 514: 29/30
    line 516: 27/30
    line 516: 29/30
    line 832: 29/30
    line 1034: 29/30
    line 1039: 29/30
    line 1177: six-method
    line 1188: 27/30
    line 1312: 29/30
    line 1315: 29/30
    line 1349: six-method
    line 1371: 29/30
    line 1390: 29/30
    line 1597: 29/30
    line 1599: 29/30
    line 2337: 29/30
    line 2337: 29/30
    line 2337: 27/30
    line 2337: 29/30
    line 2338: 29/30
    line 2338: 29/30
    line 2339: 29/30
    line 2339: 29/30
    line 2360: 29/30
  If the six-method file can't be produced, rewrite each of these to cite only the M3/M4 file.
  attribution check: each 'N/30' is compared with the last method macro (\PureLLM, \SEL, \HDS, \EHD) before it:
    line 164: \PureLLM then 30/30, but this file gives 29/30 - check (may be a different table/threshold)
    line 165: \SEL then 28/30, but this file gives 27/30 - check (may be a different table/threshold)
    line 165: \HDS then 27/30, but this file gives 21/30 - check (may be a different table/threshold)
    line 328: \PureLLM then 11/30, but this file gives 29/30 - check (may be a different table/threshold)
    line 471: \PureLLM then 11/30, but this file gives 29/30 - check (may be a different table/threshold)
    line 510: \SEL then 28/30, but this file gives 27/30 - check (may be a different table/threshold)
    line 514: \INN then 29/30, but this file gives 0/30 - check (may be a different table/threshold)
    line 516: \SEL then 23/30, but this file gives 27/30 - check (may be a different table/threshold)
    line 516: \PureLLM then 30/30, but this file gives 29/30 - check (may be a different table/threshold)
    line 832: \EHD then 29/30, but this file gives 23/30 - check (may be a different table/threshold)
    line 1034: \PureLLM then 18/30, but this file gives 29/30 - check (may be a different table/threshold)
    line 1034: \PureLLM then 18/30, but this file gives 29/30 - check (may be a different table/threshold)
    line 1155: \PureLLM then 11/30, but this file gives 29/30 - check (may be a different table/threshold)
    line 1371: \PureLLM then 30/30, but this file gives 29/30 - check (may be a different table/threshold)
    line 1390: \PureLLM then 30/30, but this file gives 29/30 - check (may be a different table/threshold)
    line 1390: \SEL then 28/30, but this file gives 27/30 - check (may be a different table/threshold)
    line 2334: \EHD then 30/30, but this file gives 23/30 - check (may be a different table/threshold)
    line 2334: \EHD then 30/30, but this file gives 23/30 - check (may be a different table/threshold)
    line 2334: \EHD then 30/30, but this file gives 23/30 - check (may be a different table/threshold)
    line 2334: \EHD then 30/30, but this file gives 23/30 - check (may be a different table/threshold)
    line 2335: \EHD then 21/30, but this file gives 23/30 - check (may be a different table/threshold)
    line 2335: \EHD then 25/30, but this file gives 23/30 - check (may be a different table/threshold)
    line 2335: \EHD then 26/30, but this file gives 23/30 - check (may be a different table/threshold)
    line 2335: \EHD then 26/30, but this file gives 23/30 - check (may be a different table/threshold)
    line 2337: \HSL then 0/30, but this file gives 30/30 - check (may be a different table/threshold)
    line 2337: \HSL then 29/30, but this file gives 30/30 - check (may be a different table/threshold)
    line 2337: \HSL then 29/30, but this file gives 30/30 - check (may be a different table/threshold)
    line 2337: \HSL then 27/30, but this file gives 30/30 - check (may be a different table/threshold)
    line 2337: \HSL then 29/30, but this file gives 30/30 - check (may be a different table/threshold)
    line 2357: \EHD then 30/30, but this file gives 23/30 - check (may be a different table/threshold)
    line 2357: \EHD then 30/30, but this file gives 23/30 - check (may be a different table/threshold)
    line 2357: \EHD then 30/30, but this file gives 23/30 - check (may be a different table/threshold)
    line 2357: \EHD then 30/30, but this file gives 23/30 - check (may be a different table/threshold)
    line 2357: \EHD then 30/30, but this file gives 23/30 - check (may be a different table/threshold)
    line 2358: \EHD then 30/30, but this file gives 23/30 - check (may be a different table/threshold)
    line 2358: \EHD then 24/30, but this file gives 23/30 - check (may be a different table/threshold)
    line 2358: \EHD then 26/30, but this file gives 23/30 - check (may be a different table/threshold)
    line 2358: \EHD then 24/30, but this file gives 23/30 - check (may be a different table/threshold)
    line 2358: \EHD then 26/30, but this file gives 23/30 - check (may be a different table/threshold)
    line 2360: \HSL then 29/30, but this file gives 30/30 - check (may be a different table/threshold)
    line 2360: \HSL then 27/30, but this file gives 30/30 - check (may be a different table/threshold)
    line 2360: \HSL then 28/30, but this file gives 30/30 - check (may be a different table/threshold)
    line 2360: \HSL then 29/30, but this file gives 30/30 - check (may be a different table/threshold)
    line 2360: \HSL then 29/30, but this file gives 30/30 - check (may be a different table/threshold)
  44 claim(s) to check
    main paper line 272: random-split
    main paper line 275: random split
    main paper line 275: random split
    main paper line 2293: Random split
    main paper line 2319: random split
    main paper line 2324: random split
    main paper line 2332: random split
    main paper line 2338: random split
    main paper line 2339: random split
    main paper line 2351: random-split
    main paper line 2356: random-split
    main paper line 2364: random split
    main paper line 2899: random split
    main paper line 2899: random split

## Item 2: Item 11 and remaining unverified content
  Item 11 in main paper (./paper/jmlr_paper_main.tex): not found
  Item 11 in routing report (./paper/supp_routing_improvements.tex): not found
  Item 11 in post-submission report (./paper/post_submission_report.tex): line 792
    Item~11 Finding~E.]} (3) \texttt{tab:formulas\_detailed}: ground-truth constants are hand-typed;
    they need extracting from the benchmark code into \texttt{config/ground\_truth\_formulas.json}. (4) Not touched, same provenance
    problem: ``Across 40 tests, 6 equations suffered catastrophic extrapolation failure'' in the main paper's LLM-extrapolation
    paragraph, and the ``(40 tests)'' label on the Pur
  Kruskal-Wallis: 2 hit(s) at lines [1874, 1875]
  power analysis: 3 hit(s) at lines [2284, 2287, 2288]
  timing: 22 hit(s) at lines [340, 388, 502, 515, 790, 964, 1083, 1266, 1266, 1391, 1392, 1396, 1426, 1436, 1440] ...
  scalability: 0 hit(s) at lines []
  manually entered: 1 hit(s) at lines [2258]
  reproduction commands: 3 hit(s) at lines [1336, 1355, 1376]
  path not found under '.': noise_sweep_.._nshards01.json (line 734)
  path not found under '.': noise_sweep_20260316_192711.json (line 1273)
  path not found under '.': sample_complexity_20260316_193447.json (line 1274)
  path not found under '.': protocol_core_noiseless_20260304_154510.json (line 1366)
  path not found under '.': run_protocol_benchmark_core.py (line 1376)
  path not found under '.': verify_proc_box_fix.py (line 1424)
  path not found under '.': run_comparative_suite_benchmark_v2_FIXED.py (line 1429)
  path not found under '.': hardware_info.py (line 1474)
  path not found under '.': noise_sweep_20260315_091018.json (line 1585)
  path not found under '.': sample_complexity_20260315_124310.json (line 1586)
  path not found under '.': S3_validation_logs.json (line 2142)

## Item 3: noise-level 'off by 10x' correction vs shard files
    stored noise_levels = [0.001]; per_noise keys = ['0.0010']
  noise_sweep_20260812_155519_nshards03.json: noise_level=[0.001] threshold_used=[0.995]
    stored noise_levels = [0.01]; per_noise keys = ['0.0100']
  noise_sweep_20260812_155520_nshards05.json: noise_level=[0.01] threshold_used=[0.99]
    stored noise_levels = [0.0]; per_noise keys = ['0.0000']
  noise_sweep_20260812_155532_nshards01.json: noise_level=[0.0] threshold_used=[0.999999]
    stored noise_levels = [0.0005]; per_noise keys = ['0.0005']
  noise_sweep_20260812_155541_nshards02.json: noise_level=[0.0005] threshold_used=[0.995]
    stored noise_levels = [0.005]; per_noise keys = ['0.0050']
  noise_sweep_20260812_155544_nshards04.json: noise_level=[0.005] threshold_used=[0.995]
  observed levels as fractions->percent: [0.0, 0.05, 0.1, 0.5, 1.0]
  -> consistent with the corrected labels [0.0, 0.05, 0.1, 0.5, 1.0] if noise_level is a FRACTION.
  threshold per level:
    0.0: [0.999999]
    0.0005: [0.995]
    0.001: [0.995]
    0.005: [0.995]
    0.01: [0.99]
  Multiple thresholds in use [0.99, 0.995, 0.999999] - add a sentence in Appendix H explaining why the two noise tables differ.

## Item 4: bracketed working notes
  7 block(s) found
    line 1875 [Removed] 325 chars
      before: y has zero for both (see the corresponding tables in Section~\ref{sec:five_systems} of the main paper).} \subsection{Interpolation Performance Comparison} \paragraph{Kruskal-Wallis H Test:} \textbf{
      block : [Removed: a Kruskal-Wallis statistic ($H = 42.3$, $p < 0.001$) and the ``significant differences across methods'' interpretation previously stated here. No data or script behind them was located, and the per-system sample sizes in the table below are unequal and unverified. Withdrawn pending recomputation from source data.]
      after : } \paragraph{System Comparisons:} \begin{table}[H] \centering \caption{Interpolation Performance Statistics by System. \textbf{[Unverified: no source data located for these summary statistics; not
    line 1885 [Unverified] 493 chars
      before: ual and unverified. Withdrawn pending recomputation from source data.]} \paragraph{System Comparisons:} \begin{table}[H] \centering \caption{Interpolation Performance Statistics by System. \textbf{
      block : [Unverified: no source data located for these summary statistics; not compared against result files (see the Post-Submission Report). Cross-check against the main paper's regenerated Table~\texttt{tab:five\_systems\_performance} (Core-15): System~3 has $n=15$ there, not $30$; Pure PySR is absent there; and the Hybrid v50\_2 and Neural Network means ($0.931$, $0.940$) differ from the regenerated train-$R^2$ means ($0.997$, $0.982$). These rows appear to predate the regeneration.]
      after : }} \label{tab:interpolation_stats} \begin{tabular}{lccccc} \toprule \textbf{System} & \textbf{n} & \textbf{Mean $R^2$} & \textbf{Median $R^2$} & \textbf{SD} & \textbf{Range} \\ \midrule System 3 (LLM+
    line 1913 [Unverified] 163 chars
      before: ative of typical performance. \subsection{Domain-Specific Success Rates} Success rate defined as R$^{2}$ $> 0.95$: \begin{table}[H] \centering \caption{Success Rate by Domain and Method. \textbf{
      block : [Unverified: not regenerated from result files; the percentages (multiples of 33\%) imply about three tasks per domain. Only the column averages were checked.]
      after : }} \label{tab:domain_success_detailed} \begin{tabular}{lccccc} \toprule \textbf{Method} & \textbf{Physics} & \textbf{Chemistry} & \textbf{Biology} & \textbf{DeFi} & \textbf{Economics} \\ \midrule Hybr
    line 1943 [Unverified] 145 chars
      before: section{Validation Layer Effectiveness} Error detection breakdown across four-layer validation framework: \begin{table}[H] \centering \caption{Validation Layer Error Detection Statistics. \textbf{
      block : [Unverified: error counts not regenerated from result files; only the percentages and cumulative column were checked for internal consistency.]
      after : }} \label{tab:validation_stats} \begin{tabular}{@{}lcccp{3cm}@{}} \toprule \textbf{Layer} & \textbf{Errors} & \textbf{\%} & \textbf{Cum.} & \textbf{Example} \\ \midrule 1. Dimensional Analysis & 18 &
    line 2258 [Unverified] 145 chars
      before: vailable at: \url{https://sednabcn.github.io/ai-llm-blog/tutorials/hypatiax/} \subsection{Additional Statistical Tests} \begin{table}[H] \centering \caption{Additional Statistical Tests. \textbf{
      block : [Unverified, manually entered: no script or data located that reproduces these test statistics or $p$-values; do not cite as computed results.]
      after : }} \label{tab:additional_stats} {\small\begin{tabular}{@{}lccl@{}} \toprule \textbf{Comparison} & \textbf{Test} & \textbf{Result} & \textbf{Interpretation} \\ \midrule Hybrid v50\_2 vs Pure PySR & Man
    line 2287 [Unverified] 47 chars
      before: nd no post-hoc power figure for this comparison has been recomputed against verified data. Removed pending recomputation.]} \item Domain comparisons (n=3 per domain): Power = 0.65 for d > 0.8 \textbf{
      block : [Unverified: no calculation or source located.]
      after : } \item Method comparisons (n=15 per method): Power = 0.95 for d > 0.6 \textbf{[Unverified: no calculation or source located; ``n=15 per method'' also disagrees with the sample sizes $15$/$15$/$18$/$3
    line 2288 [Unverified] 159 chars
      before: tem Domain comparisons (n=3 per domain): Power = 0.65 for d > 0.8 \textbf{[Unverified: no calculation or source located.]} \item Method comparisons (n=15 per method): Power = 0.95 for d > 0.6 \textbf{
      block : [Unverified: no calculation or source located; ``n=15 per method'' also disagrees with the sample sizes $15$/$15$/$18$/$30$ in \Cref{tab:interpolation_stats}.]
      after : } \end{itemize} Recommendation: Future work should increase per-domain sample sizes to n $\geq$ 10 for robust inference on secondary effects. % ======================================================
  (--apply --brackets comment hides only the note; 'drop-host' also hides the table/item/paragraph it sits in; both reversible)

## Item 5: main paper correction notes and unsupported phrase
  supplement: 47 correction-style note(s) (review by hand; stripping them automatically is unsafe)
    line 133: [Correction: previously stated as $\{0, 0.5, 1, 5, 10\}\%$, off by $10\times$ pe
    line 146: [Correction: previously stated as ``100\% recovery at all noise levels tested,''
    line 153: [Correction history, now CLOSED (five passes): this abstract's stray 90.0\% (27/
    line 178: [Correction: previously ``$1.5$--$76\times$ faster depending on noise level''; t
    line 187: [Corrected: was 96.7\%/80.0\%, then 80.0\%/66.7\%, figures presented as a Feynma
    line 264: [Note:     this evaluation uses a custom 30-equation benchmark, not a subset of 
    line 359: [Correction: previously $\{0,0.5,1,5,10\}\%$, off by $10\times$ per level; see}}
    line 383: [Correction: an earlier revision described the flag as ``based on cross-seed con
    line 472: [Correction: regenerated directly from   \texttt{protocol\_core\_noiseless\_2026
    line 543: [Corrected: was 96.7\%/$+17.4$pp, then 80.0\%/$+0.7$pp; see tablenote on \Cref{t
    line 546: [Corrected: was previously 90.0\%/$+10.7$pp, then 80.0\%/$+0.7$pp, then 66.7\%/$
    line 572: [Correction: \texttt{protocol\_core\_noiseless\_20260812\_203510.json} does not 
    line 615: [Correction: recomputed from the eleven per-domain noiseless files (the source o
    line 664: [Correction: regenerated from source, see tablenote.]}} \label{tab:r2_noise} \re
    line 684: [Correction, source-data audit: this paragraph and all three tables in this sect
    line 720: [Correction: regenerated from source, see tablenote on \Cref{tab:r2_noise}.]}} \
    line 737: [Correction, Phase~3 audit: this row}}\\ \multicolumn{5}{l}{\footnotesize\textbf
    line 745: [Correction: regenerated from source, see tablenote on \Cref{tab:r2_noise}. \tex
    line 772: [Correction: this paragraph previously claimed an 841\,s average M3 noiseless ti
    line 803: [Correction: previously captioned $\sigma=5\%$ with values that did not match an
    line 869: [Corrected: the earlier noise-sweep figures (M3 13.3\%, M4 20.7\%, tie 66.0\%) d
    line 965: [Correction: previously ``$\sim$15\,s/eq vs.\ M3's 20--841\,s;   $1.5$--$76\time
    line 1036: [Correction: this paragraph previously reported a 100\% pass rate (30/30) and a 
    line 1076: [Correction: this item     previously read ``The 841\,s/eq cost is an offline ex
    line 1145: [Corrected: was ``STRONG/STRONG, M3 100\% all $\sigma$'']} M3 76.7\% at $\sigma{
    line 1151: [Corrected: was ``COND., M3: frame 841\,s/eq as offline search'']} Both $<8$\,s/
    line 1171: [Correction: previously stated     as $0.5$--$10\%$; see tablenote on \Cref{tab:
    line 1182: [Correction, Phase~3 audit: this item previously attributed     \HDS's 90\% figu
    line 1202: [Correction: previously $1.5$--$76\times$; see     \Cref{tab:time_noise}]}; the 
    line 1208: [Corrected: previously reported as \EHD{} 96.7\%/$+17.4$pp and     \HDS{} 90.0\%
    line 1316: [Correction: this box previously stated 100\% (30/30) and 100\% (19/19), and bef
    line 1363: [Correction: this block previously pointed to \texttt{run\_comparative\_suite\_b
    line 1405: [Correction: this item was previously marked closed on the basis that a root cau
    line 1444: [note: originally ``\S\ref{sec:findings:item10b}''; that label does not exist in
    line 1494: [corrected: previously listed                    \texttt{claude-sonnet-4-2025051
    line 1578: [Correction: previously listed as $\{0, 0.5, 1, 5, 10\}\%$;     see tablenote on
    line 1600: [Correction: a prior revision claimed 100\% (19/19) for the non-hardcoded equati
    line 1703: [corrected: was   claude-sonnet-4-20250514]} \\ & Temperature & 0.25 \textbf{[co
    line 1705: [corrected: was 0.0 (deterministic)]} \\ & Max tokens & 1024 \\ & Top-p & 1.0 \\
    line 1722: [corrected: was 0.3]} \\ & Max complexity & 30 \\ & Validation threshold ($R^2$)
    line 1739: [Correction, Item 5 (post-submission report): the hand-typed 40-case table % tha
    line 1753: [Correction, Item 5 (post-submission report): the hand-typed 23-row table % that
    line 1799: [Correction, Issue 4 (CI/SD statistical incompatibility): this appendix % previo
    line 1933: [Correction: the Spearman rho = 0.89, p < 0.001 previously stated here is withdr
    line 2140: [Correction, Issue 4:]} this entry previously described the file as containing a
    line 2232: [Corrected: the six experiment notebooks previously listed here (\texttt{01\_dat
    line 2286: [Correction, F-1: this line previously reported a post-hoc power figure using re
    UNSUPPORTED PHRASE line 2013: ...'th constant error $< 10^{-5}$, achieving $R^2 = 1.0000$ and near-zero extrapolation error (see the discovered-expression examples'
  main paper: 7 correction-style note(s) (review by hand; stripping them automatically is unsafe)
    line 517: [Correction: a Spearman correlation (rho = 0.89, p < 0.001) between LLM success 
    line 541: [Correction: this paragraph previously claimed the LLM recovers the functional f
    line 1279: [Correction (2026-09-13, later pass): this description does not     match \textt
    line 1354: [Correction, Issue 4 (CI/SD statistical incompatibility): the previous version o
    line 1408: [Correction 2026-09-28: the previous version listed $n=11/15/1/13$ and put   the
    line 2484: [Correction (item N-3): the ``Neural MLP ($N$)'' column below has been removed. 
    line 2653: [Revised 2026-09-28: this paragraph previously reported solve counts (12/12 per 
    UNSUPPORTED PHRASE line 1138: ...'all five stages. An earlier draft of this paragraph claimed\nnear-zero extrapolation error on the Core~15 benchmark for this varia'

## Item 6: threshold entries and stale print line in the sweep script
  threshold map '_sigma_thresholds' at line 462: 0.0: args_thresholds.get(0.0, 0.999999), # noiseless — near-perfect 0.005: args_thresholds.get(0.005, 0.995), # σ=0.5% 0.01: args_thresholds.get(0.01, 0.990), # σ=1% 0.05: args_thresholds.get(0.05, 0.950), # σ=5% 0.10: args_thresholds.get(0.10, 0.900), # σ=10%
  entry for 0.02 present: NO
  entry for 0.2 present: NO
  entry for 0.20 present: NO
  run_all.sh line 283: NOISE_LEVELS="${NOISE_LEVELS:-0.0,0.05,0.1,0.5,1.0}"
  the sweep script reads singular NOISE_LEVEL (percent) or --noise-levels (fractions); plural NOISE_LEVELS is ignored. Launch one task per level or pass --noise-levels.

## Item 7: are the source files tracked in git (what CI sees)?
  checked 18 file(s); 0 not tracked. CI regenerates tables from tracked files only, so untracked sources explain 'no result file in the repository' and 'incomplete sweep' rows.

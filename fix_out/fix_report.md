# fix_paper_issues report

## Issue 1 - split variant behind the accuracy results
  seed-42 copy: hypatiax/data/results/comparison_results/noise-noiseless/15/hypatiax_defi_benchmark_v3_results_seed42.json  md5=97f1a16c958e9bf366559ecc47f5b4c1
  seed-42 copy: hypatiax/data/results/comparison_results/noise-noiseless/noiseless/defi/hypatiax_defi_benchmark_v3_results_seed42.json  md5=cbf503d0b3985b4c1bccf7ed18e0316b
  seed-42 copy: hypatiax/data/results/defi/hypatiax_defi_benchmark_v3_results_seed42.json  md5=cbf503d0b3985b4c1bccf7ed18e0316b

  Loader validation on hypatiax_defi_benchmark_v3_results_seed42.json (md5 cbf503d0..., paper cites cbf503d0...):
  PASS: all four rows of tab:main_results are reproduced from the file.
  skipped: _checkpoint_shard0.json: unreadable (cannot locate per-task records; structure is {"completed": ["str", "... 10 items)
  skipped: exp1_pca_summary.json: unreadable (cannot locate per-task records; structure is {"fixc3_step": "str", "description")
  skipped: fixc3_baseline.json: unreadable (cannot locate per-task records; structure is {"fixc3_gate": "str", "description")
  skipped: _checkpoint_shard0.json: unreadable (cannot locate per-task records; structure is {"completed": ["str", "... 10 items)
  skipped: _checkpoint_shard1.json: unreadable (cannot locate per-task records; structure is {"completed": ["str", "... 10 items)
  skipped: _checkpoint_shard2.json: unreadable (cannot locate per-task records; structure is {"completed": ["str", "... 10 items)
  skipped: _checkpoint_shard3.json: unreadable (cannot locate per-task records; structure is {"completed": ["str", "... 10 items)
  skipped: _checkpoint_shard4.json: unreadable (cannot locate per-task records; structure is {"completed": ["str", "... 10 items)
  skipped: fixc3_baseline.json: unreadable (cannot locate per-task records; structure is {"fixc3_gate": "str", "description")
  skipped: fixc3_baseline__shard0_rescued_1790894334.json: unreadable (cannot locate per-task records; structure is {"fixc3_gate": "str", "description")
  skipped: fixc3_baseline__shard2_rescued_1790894314.json: unreadable (cannot locate per-task records; structure is {"fixc3_gate": "str", "description")
  skipped: fixc3_baseline__shard3_rescued_1790894333.json: unreadable (cannot locate per-task records; structure is {"fixc3_gate": "str", "description")
  skipped: fixc3_baseline__shard4_rescued_1790894333.json: unreadable (cannot locate per-task records; structure is {"fixc3_gate": "str", "description")
  skipped: split_protocol_disclosure.json: unreadable (cannot locate per-task records; structure is {"fixc3": "bool", "split_protocol":)
  PCA seed 42: hypatiax/data/results/comparison_results/noise-noiseless/noiseless/defi_pca/hypatiax_defi_benchmark_pca_results_seed42.json
  PCA seed 99: hypatiax/data/results/comparison_results/noise-noiseless/noiseless/defi_pca/multi-seeds/hypatiax_defi_benchmark_pca_results_seed99.json
  PCA seed 123: hypatiax/data/results/comparison_results/noise-noiseless/noiseless/defi_pca/multi-seeds/hypatiax_defi_benchmark_pca_results_seed123.json
  PCA seed 777: hypatiax/data/results/comparison_results/noise-noiseless/noiseless/defi_pca/multi-seeds/hypatiax_defi_benchmark_pca_results_seed777.json
  PCA seed 2024: hypatiax/data/results/comparison_results/noise-noiseless/noiseless/defi_pca/multi-seeds/hypatiax_defi_benchmark_pca_results_seed2024.json

  Accuracy, near-perfect rate (R2>0.99, %):  v3-s42 | s42 s99 s123 s777 s2024 | pooled
    Pure LLM                  59.5 |  66.2  64.9  64.9  68.9  64.9 |  65.9
    Neural MLP                 0.0 |   2.7   0.0   1.4   0.0   1.4 |   1.1
    HypatiaX (uncorrected)    90.5 |  89.2  89.2  90.5  90.5  89.2 |  89.7
    HypatiaX (corrected)      59.5 |  66.2  64.9  63.5  67.6  64.9 |  65.4

  Evidence on the split variant of the seed-42 file:
   * fields naming a split/variant inside the records: none
   * per-task test_r2 identical, v3-seed42 vs PCA-seed42 (pure_llm / NN / hybrid): 72% / 1% / 92%
   * control, PCA-seed42 vs PCA-seed99 (different seeds)  : 89% / 14% / 93%
     Reading: near-100% agreement on the NN arm (deterministic given a seed and a partition) means the same train/test partition; low agreement means a different one. This is evidence only.
   => Not determinable from the files. Caveat is rewritten to say exactly that and to point at the new PCA table.
      If you know from the code which variant produced the seed-42 file, re-run with --split-variant pca|v3c.

## Issue 2 - Mann-Whitney regime label
Paper-internal evidence for the far regime:
  * the appendix medians (2415.2 % Hybrid, 13283.9 % NN) are the same numbers as the main paper's
    tab:five_systems_full, whose caption defines the metric as 100*extrap_rmse_far/train_rmse;
  * n=12 / n=15 equal the finite-value counts that table reports before its Tukey-fence
    trimming ('Hybrid keeps 10 of 12, Neural Network 12 of 15');
  * U=74.0 with n=12/15 gives one-tailed p=0.2247 (normal approximation, continuity
    correction), matching the appendix's 2.25e-01, so those three numbers are mutually consistent.
  * the 2x multiplier is the medium regime (supplement S2 lists 1.2x, 2x, 5x); the far regime is 5x.

  Data check: hypatiax/data/results/five_systems/exp1_five/exp1_five_results.json
  regime | n_hyb n_nn | median_hyb  median_nn |     U   p(one-tailed, hyb<nn)
  near   |    12   15 |       55.4      297.8 |   56.0   0.0511
  medium |    11   15 |      225.9     2888.4 |   46.0   0.0309
  far    |    12   15 |     2415.2    13283.9 |   74.0   0.2247
  Appendix reports: n=12/15, medians 2415.2 / 13283.9, U=74.0, p=0.225.  Regime reproducing U and n: ['far']

## Issue 3 - tables labelled Unverified

### tab:portfolio_seed_sweep
  arithmetic in the table: mean P -10.686, mean H -6.261, strict wins 4/5, win column consistent: True
  several candidates, using the first (override with --sweep-json): hypatiax/data/results/comparison_results/noise-noiseless/15/portfolio_variance_seed_sweep.json, hypatiax/data/results/portfolio_variance_audit/portfolio_variance_seed_sweep.json
  hypatiax/data/results/comparison_results/noise-noiseless/15/portfolio_variance_seed_sweep.json: could not extract per-seed values (seed 42: ambiguous far-R2 leaves; PySR=[] H=[]). Edit extract_sweep() or pass a different file.
  Source files not located/readable: tag narrowed to what was checked.

### tab:arch
  92.7% = [139]/150 and 74.7% = [112]/150 -> both are shares of the 150 noise-sweep comparisons (neither is a whole count out of the 180 sample-complexity comparisons).
  skipped duplicate: _checkpoint_shard3.json (identical to _checkpoint_shard0.json)
  skipped duplicate: _checkpoint_shard4.json (identical to _checkpoint_shard0.json)
  skipped duplicate: _checkpoint_shard1.json (identical to _checkpoint_shard0.json)
  skipped duplicate: suppB_results.json (identical to _merged.json)
  skipped duplicate: _checkpoint_shard2.json (identical to _checkpoint_shard0.json)
  skipped duplicate: fixc3_baseline_shard3.json (identical to fixc3_baseline_shard2.json)
  skipped duplicate: fixc3_baseline_shard0.json (identical to fixc3_baseline_shard2.json)
  skipped duplicate: _merged.json (identical to exp1b_results.json)
  skipped duplicate: fixc3_baseline_shard1.json (identical to fixc3_baseline_shard2.json)
  skipped duplicate: fixc3_baseline_shard4.json (identical to fixc3_baseline_shard2.json)
  skipped duplicate: fixc3_baseline.json (identical to fixc3_baseline_shard2.json)
  skipped duplicate: fixc3_baseline.json (identical to fixc3_baseline.json)
  skipped duplicate: fixc3_baseline__shard4_rescued_1790894333.json (identical to fixc3_baseline.json)
  skipped duplicate: fixc3_baseline__shard3_rescued_1790894333.json (identical to fixc3_baseline.json)
  skipped duplicate: fixc3_baseline__shard2_rescued_1790894314.json (identical to fixc3_baseline.json)
  skipped duplicate: fixc3_baseline__shard0_rescued_1790894334.json (identical to fixc3_baseline.json)
  skipped duplicate: fixc3_baseline.json (identical to fixc3_baseline.json)
  skipped duplicate: fixc3_baseline__shard1_rescued_1790686645.json (identical to fixc3_baseline.json)
  skipped duplicate: fixc3_baseline__shard0_rescued_1790686663.json (identical to fixc3_baseline.json)
  skipped duplicate: fixc3_baseline__shard2_rescued_1790686648.json (identical to fixc3_baseline.json)
  65 unique noise shard file(s) after dedupe; M3 records: 150, M4 records: 150 (paper: 150 each)
  M3 cross-tab (decision, nn_applied, success): ('ensemble', False, True)=29; ('ensemble', True, True)=111; ('nn', False, True)=2; ('nn', True, True)=8
  M4 cross-tab (decision, nn_applied, success): ('ensemble', True, True)=27; ('llm', True, True)=123
  Candidate definitions (paper: M3 [139]/150, M4 [112]/150):
    M3  decision == ensemble                 140/150  93.3%
    M3  ensemble and nn_applied is False     29/150  19.3%
    M3  ensemble and success                 140/150  93.3%
    M4  decision == llm                      123/150  82.0%
    M4  llm and nn_applied is False          0/150  0.0%
    M4  llm and success                      123/150  82.0%
  NOTE: a definition that reproduces a number is a hypothesis about what was counted, not a verification.
        The tag is NOT removed automatically: confirm the definition in the code that generated the table.
  No single definition reproduces both counts (or n != 150) - tag kept.

### tab:noise_sensitivity
  arithmetic: all 7 CIs are exact Clopper-Pearson intervals of the stated counts: True; Fisher exact (sigma<=0.10: 327/375 vs >0.10: 52/150) p=4.28e-32
  scanning --repo for a result file with all seven noise levels (this can take a while)...
  No file with all seven noise levels found.

## Edits
  [applied] main: insert tab:pca_accuracy before the observations paragraph
  [applied] main: conclusion said no 74-task PCA run is available
  [applied] main: appendix said the only PCA seed files hold 5 records
  [applied] main: disambiguate 'those five files' after the previous edit
  [applied x3] main: abstract / contributions / conclusion split-variant caveat
  [applied] supp: split-variant caveat
  [applied] supp: Mann-Whitney regime label (medium 2x -> far 5x)
  [applied] main: cross-reference to the Mann-Whitney test (medium -> far)
  [applied] main: tab:portfolio_seed_sweep tag narrowed
  [applied] supp: tab:arch tag narrowed
  [applied] supp: tab:noise_sensitivity tag narrowed

Dry run: nothing written except the report, changes.diff and generated/ (fix_out). Add --apply for patched copies.

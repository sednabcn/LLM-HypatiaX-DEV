# Issue 10 — Timeout Contradiction: 1100 s vs. 900 s; 300 s Limit but a 27,574 s Hang

**Category:** [OPEN — needs code/data investigation] (enforcement bug, second half) / [STALE — reconciliation only] (config bleed, first half)

**Location:** `supp_benchmark_report.tex`, Sweep hyperparameters (`tab:sweeps`, l. 294–311) vs. Reproducibility Statement, Phase 3 reproduction command, l. 988–1004.

## What is wrong
`tab:sweeps`' hyperparameter note states:

> Hyperparameters: NN seeds = 5, random seed 42, method timeout = 900 s, PySR timeout = 1100 s, threshold (noisy) = 0.995, threshold (noiseless) = 0.999999. All 30 equations completed with zero timeouts across all 11 conditions.

The Phase 3 reproduction command block, describing (per its own caption) the run underlying `tab:overall`, instead shows:

```
python run_comparative_suite_benchmark_v2.py \
  noiseless threshold 0.9999 \
  nn-seeds 3 samples 200 \
  method-timeout 900 \
  pysr-timeout 900
```

i.e. 900 s for both timeouts, not 1100 s for PySR as `tab:sweeps` states.

Immediately below, the same reproducibility note directly contradicts the "zero timeouts" claim:

> Note on timeout upgrade: method-timeout was set to 300 s for tests 1–18 and upgraded to 900 s for tests 19–30. Arrhenius (test 4) ran under the 300 s limit and hung for 27574 s before the Julia process was killed... Nernst (test 6) also timed out under the 300 s default; counted as failure.

A stated 300 s limit allowing a process to run for 27,574 s (over 90× the configured limit) before manual kill indicates the timeout was not actually enforced for that run.

## Fix required
- **First half (1100 s vs. 900 s):** reads as two different phases' configs bleeding into one write-up — reconcile which value is correct for `tab:sweeps` and state it consistently.
- **Second half (300 s limit, 27,574 s hang):** the timeout enforcement itself needs checking in the harness code (or was silently disabled for that run) — this is a code-level question, not just a documentation fix.

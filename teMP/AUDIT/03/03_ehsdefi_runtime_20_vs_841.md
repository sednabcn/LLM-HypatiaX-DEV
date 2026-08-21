# Issue 3 — EHSDeFi (M3) Runtime: 20.2 s vs. 841.4 s

**Category:** [OPEN — needs code/data investigation] — needs live code/data investigation

**Location:** `supp_benchmark_report.tex`, `tab:overall` l. 413 vs. `tab:time_noise` ("Sweep Results: Noise Robustness") l. 571–584.

## What is wrong
`tab:overall` reports EHSDeFi's (M3) average runtime as:

> EHSDeFi (M3) Core — 29/30 (96.7%) — 0.999993 — 20.2 s

`tab:time_noise`, covering the same method at σ = 0% (the same noiseless condition), reports:

> 0% — 841.4 — 11.1 — 75.8×

The abstract's own claim that "EHSDeFi offers exact symbolic interpretability; HyperSymLoop is 1.576× faster depending on noise level" corroborates the 841.4 s figure (75.8× vs. M4's 11.1 s), not the 20.2 s figure.

## Fix required
Trace which script/run produced 20.2 s. If it reflects different hardware/config, that must be stated explicitly; if it is simply wrong, it must be regenerated from the same raw timing logs that produced `tab:time_noise`.

Ruperto Pedro Bonet Chaple
Independent Researcher
London, United Kingdom
ruperto.bonet@modelphysmat.com

July 14, 2026

The Editor
Journal of Machine Learning Research

Dear Editor,

I am pleased to submit my manuscript, "HypatiaX: A Hybrid Symbolic-Neural Framework for Extrapolation-Reliable Analytical Discovery," for consideration for publication in the Journal of Machine Learning Research. Two supplementary materials accompany the submission: Supplementary A (Routing Improvements for the Hybrid LLM–Neural-Network System) and Supplementary B (Noise Robustness, Sample Complexity, and Convergence on the Feynman Benchmark).

**Summary of contributions.** The paper addresses a central weakness in automated formula discovery: systems that interpolate well but extrapolate poorly are of limited use in high-stakes domains. I show that large language models and neural networks fail in complementary ways — LLMs exhibit bimodal, occasionally catastrophic performance tied to how well a formula is represented in pretraining data, while neural networks degrade systematically under distribution shift regardless of training accuracy. HypatiaX is a five-stage routing and ensembling architecture (trust gating, transcendental complexity detection, neural degradation probing, out-of-distribution detection, and uncertainty-weighted ensembling) that exploits this complementarity using training-data diagnostics alone.

On a benchmark of 74 DeFi mathematical tasks under an aggressive PCA-directed 40%/60% extrapolation split, HypatiaX achieves an 89.2% near-perfect success rate (R² > 0.99) versus 62.2% for a pure LLM baseline and 5.4% for a neural baseline, while eliminating all six catastrophic failures (R² < −10) observed in the LLM-only approach. Gains scale with task difficulty (+12.5 pp easy, +31.1 pp medium, +38.1 pp hard), and the architecture generalizes to the standard Nguyen-12 suite (11/12 vs. 10/12 for symbolic regression alone) and to 30 Feynman physics equations, indicating that the reliability benefit is not specific to the DeFi domain.

**A note on transparency.** In preparing this submission I identified and corrected a runtime claim from an earlier draft, which asserted an unsupported 73% runtime reduction (3.7× speedup) over the neural baseline. The verified analysis, reported in full in Section 10.4, is that HypatiaX delivers a 1.73× speedup specifically on the 68 of 74 tasks (92%) where the LLM formula is trusted; when the six harder fallback cases requiring full neural retraining are included, the aggregate mean runtime is higher than the neural baseline. I report this discrepancy explicitly, along with several other corrections made during a v3.0 revision of the benchmark pipeline (data-leakage in ensembling, extrapolation-split misconfiguration, inconsistent formula evaluation, and an overly permissive trust threshold), because I believe honest disclosure of measurement corrections is essential to reproducible machine learning research. I have also reported several null and mixed results as such — for example, the domain-selective (rather than universal) advantage on the Core-15 ablation, and the single seed on which HypatiaX underperforms PySR-only in the Portfolio Variance robustness study — rather than omitting them.

**Why JMLR.** This work sits at the intersection of symbolic regression, LLM reasoning, and reliability under distribution shift, and I believe JMLR's readership is well positioned to evaluate both the empirical benchmark contribution (74 DeFi tasks with a rigorous extrapolation protocol) and the architectural contribution (the five-stage routing cascade, with a proof sketch of its convergence property in Proposition 11). The two supplementary materials provide full implementation detail for the routing fixes and an independent noise/sample-complexity study, supporting reproducibility.

I confirm that this manuscript is original, is not under review elsewhere, and that all authors (myself, as sole author) have approved its submission. I have no conflicts of interest to declare. Code, data-generation protocols, and result JSONs are made available for reproducibility, as described in the manuscript's reproducibility statement.

Thank you for your consideration. I welcome any questions from the editorial team or reviewers.

Sincerely,

Ruperto Pedro Bonet Chaple
Independent Researcher, London, United Kingdom
ruperto.bonet@modelphysmat.com

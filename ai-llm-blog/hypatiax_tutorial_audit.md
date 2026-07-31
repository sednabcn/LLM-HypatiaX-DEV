# HypatiaX Tutorial Series — Cross-Reference & New Features Audit

**Scope:** index.md, setup.md, experiments.md, analysis.md, extensions.md  
**Audited:** June 2026

---

## 1. Internal Link Consistency

| Source file | Link text | Target permalink | Status |
|---|---|---|---|
| index.md | Tutorial 1 | `/tutorials/hypatiax/setup/` | ✅ matches setup.md |
| index.md | Tutorial 2 | `/tutorials/hypatiax/experiments/` | ✅ matches experiments.md |
| index.md | Tutorial 3 | `/tutorials/hypatiax/analysis/` | ✅ matches analysis.md |
| index.md | Tutorial 4 | `/tutorials/hypatiax/extensions/` | ✅ matches extensions.md |
| setup.md → Next | Tutorial 2 | `/tutorials/hypatiax/experiments/` | ✅ |
| experiments.md → Prev | Tutorial 1 | `/tutorials/hypatiax/setup/` | ✅ |
| experiments.md → Next | Tutorial 3 | `/tutorials/hypatiax/analysis/` | ✅ |
| analysis.md → Prev | Tutorial 2 | `/tutorials/hypatiax/experiments/` | ✅ |
| analysis.md → Next | Tutorial 4 | `/tutorials/hypatiax/extensions/` | ✅ |
| extensions.md → Prev | Tutorial 3 | `/tutorials/hypatiax/analysis/` | ✅ |
| extensions.md (Tutorial 4 back-refs) | Tutorials 1 & 3 | correct permalinks | ✅ |

**All navigation links are internally consistent.**

---

## 2. Key Numerical Claims — Cross-Reference Check

### 2a. Success rate (131 equations)

| File | Claim | Match? |
|---|---|---|
| index.md | 95.8% success on 131 equations | ✅ |
| setup.md | "95.8% success rate on 131 scientific equations" | ✅ |
| experiments.md | `--output ... --v2` → "success rate 95.8%" | ✅ |
| analysis.md | `claims["Hybrid extrapolation: median error < 10⁻¹²"]` | ✅ |
| extensions.md | Does not repeat global rate | ✅ (not required) |

### 2b. Extrapolation error

| File | Claim |
|---|---|
| index.md | Median extrap error < 10⁻¹² |
| setup.md | Median relative error 2.34e-13; NN 12.47 (1,247%) |
| experiments.md | Hybrid v40 median < 10⁻¹², NN mean 1,231% / median 86.7% |
| analysis.md | Code reproduces < 10⁻¹² for Hybrid, > 1,200% for NN |

> ⚠️ **Minor inconsistency:** index.md says NN "median error 1,231%" in the "Why HypatiaX?" bullet, but experiments.md says "mean 1,231%, median 86.7%". The setup.md introductory example quotes 12.47 (1,247%), which is the single-equation Ohm's Law result — not the global benchmark figure. These are consistent with the underlying data (Ohm's Law is a simple equation; the 1,231% is the cross-benchmark mean) but could confuse readers comparing numbers across pages.  
> **Recommendation:** Add a brief note in setup.md that the Ohm's Law NN error (1,247%) is an illustrative single-equation example; the benchmark mean is 1,231%.

### 2c. Mean discovery time

| File | Claim |
|---|---|
| index.md | 390 seconds |
| setup.md | 390 seconds |
| experiments.md | (not repeated; implied by benchmark runs) |
| analysis.md | (not repeated) |

✅ Consistent.

### 2d. DeFi extrapolation benchmark

| File | Hybrid R²>0.99 | Pure LLM R²>0.99 | Denominator |
|---|---|---|---|
| index.md | Not stated numerically | Not stated | — |
| experiments.md | 72.7% (48/66) | 69.7% (46/66) | n=66 fixed ✅ |
| analysis.md | 72.7% | 69.7% | n=66 fixed ✅ |

✅ Consistent. Both correctly use honest n=66 denominator and flag that per-method denominators (83.6%, 77.4%) are incomparable.

### 2e. Feynman SR benchmark

| File | Hybrid DeFi | AI Feynman baseline | Gain |
|---|---|---|---|
| index.md | 96.7% (+17.4 pp over AI Feynman 2.0) | implied 79.3% | ✅ |
| experiments.md | 96.7% (29/30) | 79.3% | +17.4 pp ✅ |
| analysis.md | 96.7% recovery, table shows AI Feynman 79.3% | 79.3% | ✅ |

✅ Consistent.

### 2f. LLM speedup

| File | Claim |
|---|---|
| index.md | 73% speedup |
| setup.md | 73% speedup |
| experiments.md | Not mentioned |
| analysis.md | Not mentioned |
| extensions.md | Not mentioned |

✅ Consistent where cited.

### 2g. Mann-Whitney U statistic

| File | U | p-value |
|---|---|---|
| index.md | U=0, p<10⁻⁶ | ✅ |
| setup.md | U=0, p<10⁻⁶ | ✅ |
| experiments.md | U=0, p=1.11×10⁻⁶ | ✅ (more precise) |
| analysis.md | U=0, p=1.11×10⁻⁶ | ✅ |

✅ Consistent.

---

## 3. Domain Labels — Inconsistency Found

### ❌ Problem: "Economics" domain listed in setup.md, absent from paper

**setup.md** (Project Structure comment and Core 15 description) does not mention an Economics domain explicitly, but **index.md** "Core Concepts" bullet lists *"Health factor calculations in DeFi / Liquidation thresholds"* under what appears to be the DeFi domain correctly.

However, **experiments.md** contains an explicit correction note:

> *"Note: Earlier documentation listed an 'Economics' domain. The JMLR paper does not have a separate Economics campaign. The five domains are as listed above."*

The five correct domains are:
- Physics, Chemistry, Biology, DeFi AMM, DeFi Risk

**setup.md does not contain this correction notice.** Readers of Tutorial 1 only may encounter old documentation elsewhere and be confused.

**Recommendation:** Add a note to setup.md (Project Structure section) listing the four benchmark domains explicitly: Physics, Biology/Chemistry, DeFi AMM, DeFi Risk.

---

## 4. v2 Bug Fix Notice Coverage

The v2 R² evaluation bug fix is documented in:

| File | v2 notice present? |
|---|---|
| index.md | ❌ No mention |
| setup.md | ❌ No mention |
| experiments.md | ✅ Prominent callout box at top + `--v2` flag on all commands |
| analysis.md | ✅ Prominent callout box at top + `--v2` flag on all commands |
| extensions.md | ❌ No mention |

> ⚠️ **Gap:** index.md serves as the entry point for the entire series. Users who start there get no indication that a bug fix exists. setup.md is Tutorial 1 — readers who complete Tutorial 1 and stop (or share a link directly to Tutorial 1) will not see the v2 notice.  
> **Recommendation:** Add a short v2 note to index.md (perhaps in the "Key Results" section) and a brief mention in setup.md (Environment Setup is fine; users won't run benchmarks there, but they should know --v2 exists before Tutorial 2).

---

## 5. Figure Number Cross-Reference

The paper contains 13 figures. The tutorials reference them as follows:

| Tutorial ref | Paper label | Description | Consistent? |
|---|---|---|---|
| analysis.md "Figure 3" | `fig:arrhenius_extrapolation` | Arrhenius failure | ⚠️ Tutorial calls it Figure 3 but saves as `figure1_arrhenius_extrapolation.pdf` — numbering mismatch |
| analysis.md "Figure 2" | `fig:domain_comparison` | Domain success rates | ⚠️ Tutorial calls it Figure 2 but saves as `figure2_domain_comparison.pdf` — internally consistent but out of paper order |
| analysis.md "Figure 6" | `fig:validation_breakdown` | Validation cascade | ⚠️ Tutorial says Figure 6, saves as `figure3_validation_breakdown.pdf` |
| analysis.md "Figure 7" | `fig:method_comparison` | Three-system comparison | ⚠️ Tutorial says Figure 7, saves as `figure5_method_comparison.pdf` |
| analysis.md "Figure 9" | `fig:five_systems_comparison` | Five-system comparison | ✅ Consistent label |
| analysis.md "Figures 11–13" | DeFi figures | New in v2 | ✅ Consistent |

> ⚠️ **Systematic mismatch:** The tutorial's Figure labels (2, 3, 6, 7) do not match the output filenames (figure1_, figure2_, figure3_, figure5_). The note in analysis.md says "Figure 1 is an inline TikZ diagram; Figures 4–5 are architecture diagrams" which explains the gap, but the output filenames do not reflect paper figure numbers.  
> **Recommendation:** Either (a) rename output files to match paper figure numbers (`figure3_arrhenius.pdf`, `figure6_validation.pdf`, etc.) or (b) add a mapping table in analysis.md cross-referencing tutorial output filename → paper figure number.

---

## 6. Citation Inconsistency

Two different BibTeX entries appear across tutorials:

**Version A** (index.md, experiments.md, analysis.md):
```bibtex
@article{bonetchaple2026hypatiax,
  title={Why Extrapolation Breaks Naïve Analytical Discovery},
  author={Bonet Chaple, Ruperto Pedro},
  journal={Journal of Machine Learning Research},
  year={2026},
  volume={27},
  pages={1--47}
}
```

**Version B** (setup.md):
```bibtex
@article{hypatiax2026,
  title={HypatiaX: A Hybrid Symbolic-Neural Framework for Extrapolation-Reliable Analytical Discovery},
  author={...},  ← author field missing!
  journal={Journal of Machine Learning Research},
  year={2026}
}
```

**Version C** (extensions.md):
```bibtex
@article{hypatiax2026,
  title={LLMs as Interfaces to Symbolic Discovery: Perfect Extrapolation via Hybrid Architectures},
  journal={Journal of Machine Learning Research},
  year={2026}
}  ← author field missing!
```

> ❌ **Three different titles, two different citation keys, missing author fields in two entries.**  
> **Recommendation:** Standardise to Version A (bonetchaple2026hypatiax) throughout. Update setup.md and extensions.md to use the full entry including author field.

---

## 7. New Features in Tutorial 4 (extensions.md) — Coverage Check

Tutorial 4 introduces the following features not mentioned in earlier tutorials:

| Feature | Mentioned in index.md? | Described in earlier tutorials? |
|---|---|---|
| Scikit-learn pipeline integration | ✅ (listed in Tutorial 4 description) | ❌ not pre-explained |
| REST API / Flask deployment | ✅ (listed) | ❌ |
| Docker containerization | ✅ (listed) | ❌ |
| Multi-equation / ODE discovery | ✅ (predator-prey listed) | ❌ |
| Custom PySR operators | ✅ (listed) | ❌ |
| DomainValidator subclassing | ❌ not listed in index.md | ❌ |
| `complexity_penalty` parameter | ❌ not mentioned in index.md | ❌ (only `populations`, `niterations` in setup.md) |

> ℹ️ Most omissions from index.md are intentional (advanced content deferred to T4). Two gaps worth filling:
> - **DomainValidator** class is an important extensibility point; worth a one-line mention in index.md Tutorial 4 description.
> - **`complexity_penalty`** parameter is introduced in extensions.md without any setup.md/experiments.md precedent. Add it to the Configuration Options table in setup.md.

---

## 8. Routing Fixes — Coverage Check

The six routing improvements (fixes 0–5) are documented only in experiments.md. They are:

- Referenced in analysis.md indirectly (DeFi 72.7% result depends on them)
- Not mentioned in extensions.md Example 2 (CO₂ sequestration), which uses `enable_routing_fixes=True` without explanation

> **Recommendation:** Add a one-line comment in extensions.md where `enable_routing_fixes=True` appears, pointing readers to experiments.md §Campaign B for details.

---

## 9. Missing Cross-References

| Missing link | Where needed | What to add |
|---|---|---|
| Routing fixes table | extensions.md `enable_routing_fixes=True` | Link to experiments.md §Campaign B |
| v2 bugfix explanation | index.md and setup.md | Short callout pointing to experiments.md v2 note |
| Figure-to-filename mapping | analysis.md | Table mapping paper figure numbers to output filenames |
| DomainValidator | index.md Tutorial 4 description | One-line mention |
| `complexity_penalty` | setup.md Configuration Options | Add to parameter table |
| Snell's law near-miss note | analysis.md Feynman section | experiments.md mentions it; analysis.md omits it |

---

## 10. Summary: Issues by Severity

### 🔴 High Priority (correctness / reproducibility)

1. **Citation inconsistency** — three different titles/keys across tutorials; two entries missing author. Standardise to `bonetchaple2026hypatiax` with full fields.
2. **v2 notice missing from index.md and setup.md** — first-contact pages don't warn users about the evaluation bug fix and `--v2` flag requirement.

### 🟡 Medium Priority (clarity / consistency)

3. **Figure number vs filename mismatch** in analysis.md — tutorial text says "Figure 3" but file is saved as `figure1_*.pdf`. Add a mapping table or rename files.
4. **NN error figure confusion** — 12.47 (Ohm's Law demo in setup.md) vs 1,231% (benchmark mean in experiments/index) vs 1,247% (index.md bullets). Each is correct in context but needs a clarifying note in setup.md.
5. **Economics domain removal not noted in setup.md** — experiments.md has the correction; setup.md does not.

### 🟢 Low Priority (enhancements)

6. Add `DomainValidator` to index.md Tutorial 4 listing.
7. Add `complexity_penalty` to setup.md configuration table.
8. Add back-reference in extensions.md `enable_routing_fixes=True` → experiments.md.
9. Add Snell's law near-miss note to analysis.md Feynman section (already in experiments.md).

---

## 11. New Features Not Yet Covered Anywhere

The following items appear in the codebase references or paper sections but are not covered in any tutorial:

| Feature | Paper section | Tutorial coverage |
|---|---|---|
| `hybrid_all_domains/` protocol | Implied by directory listing in setup.md | ❌ Not explained |
| `hybrid_defi_llm_nn/` protocol | Directory listing in setup.md | ❌ Not explained |
| `hybrid_llm_guide_validation/` protocol | Directory listing in setup.md | ❌ Not explained |
| Appendix `app:statistical_tests` (rank-biserial) | analysis.md mentions it | ❌ Not demonstrated in code |
| Phase 2 (noisy) vs Phase 3 (noiseless) Feynman distinction | experiments.md | analysis.md Feynman section only shows Phase 3 results — Phase 2 not reproduced |
| `--no-llm-cache` flag | experiments.md | ❌ Not explained (what does it do?) |
| `systems_2_3_detailed.csv` format | analysis.md (Figure 6 code) | ❌ Not described; users won't know how to generate it |

> **Recommendation:** Consider a short "Generated files reference" appendix or FAQ in analysis.md listing what each output CSV/JSON contains and which experiment command produces it.

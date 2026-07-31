---
title: "HypatiaX Tutorial Series"
date: 2026-02-16
permalink: /tutorials/hypatiax/
layout: single
classes:
  - inner-page
  - header-image-readability
author_profile: true
header:
  overlay_image: /assets/images/tutorials/hypatiax-banner.png
  overlay_filter: 0.5
  caption: "LLM-Guided Symbolic Discovery"
sidebar:
  nav: "tutorials"
toc: true
toc_label: "Tutorial Series"
toc_icon: "flask"
---

# HypatiaX: LLM-Guided Symbolic Discovery

Welcome to the complete HypatiaX tutorial series! Learn how to use hybrid LLM + symbolic regression to discover scientific equations from data with near-perfect extrapolation.

---

## 🎯 What is HypatiaX?

HypatiaX is a groundbreaking framework that combines:
- **Large Language Models (LLMs)** for intelligent initialization
- **Symbolic Regression** for mathematically rigorous discovery
- **Multi-layer Validation** for ensuring correctness

**Key Results from JMLR Paper:**
- ✅ **95.8% success rate** on 131 scientific equations
- ✅ **Median extrapolation error < 10⁻¹²** (floating-point precision limit)
- ✅ **Complete statistical separation** from neural networks (Mann-Whitney U=0, p<10⁻⁶)
- ✅ **New state-of-the-art on Feynman SR Benchmark**: 96.7% exact recovery at R²>0.9999 (Hybrid DeFi, +17.4 pp over AI Feynman 2.0)
- ✅ **Mean discovery time: 390 seconds** per equation

---

## 📚 Tutorial Series

### [Tutorial 1: Environment Setup and First Discovery](/tutorials/hypatiax/setup/)
**Time:** 15 minutes | **Difficulty:** Beginner

Learn to install HypatiaX and discover your first equation (Ohm's Law). Includes:
- Installation and verification
- Your first formula discovery
- Extrapolation validation
- Understanding the core advantage: symbolic vs neural

**What you'll discover:**
```python
# HypatiaX: Median error 2.34e-13 (near floating-point precision!)
# Neural Network: Median error 12.47 (1,247% error!)
```

---

### [Tutorial 2: Running Benchmark Experiments](/tutorials/hypatiax/experiments/)
**Time:** 45 minutes (active) + 3–8 hours (compute) | **Difficulty:** Intermediate

Reproduce the three benchmark evaluations from the JMLR paper:

| Benchmark | Equations | Primary metric | Section |
|---|---|---|---|
| **Core 15** | 15 across 4 domains | Extrapolation error (%) | §6.4 |
| **DeFi Extrapolation** | 73 test cases (66 standard) | R²>0.99 at fixed n=66 | §6.5 |
| **Feynman SR** | 30-equation subset | Recovery rate at R²>0.9999 | §5.8 |

**Key experiments:**
- Run individual benchmark campaigns
- Compare discovery systems (Pure Symbolic, Hybrid, Pure LLM)
- Parallel execution for faster results
- Checkpoint/resume functionality

---

### [Tutorial 3: Statistical Analysis and Publication Figures](/tutorials/hypatiax/analysis/)
**Time:** 45 minutes | **Difficulty:** Intermediate

Generate all 13 publication-quality figures and reproduce statistical analyses:
- **Figure 3:** Arrhenius equation extrapolation failure
- **Figure 2:** Success rate comparison across domains
- **Figure 6:** Validation cascade breakdown
- **Figure 9:** Five-system unified comparison (13 figures total in paper)
- **Figures 11–13:** DeFi extrapolation benchmark (new in v2)

**Statistical validation:**
```python
Mann-Whitney U test: U=0, p<10⁻⁶
Cohen's d: 0.95 (pooled; see paper note on degenerate distributions)
95% CI for neural error: [1,087%, 1,456%]
```

---

### [Tutorial 4: Custom Applications and Extensions](/tutorials/hypatiax/extensions/)
**Time:** 45 minutes | **Difficulty:** Advanced

Apply HypatiaX to your own scientific problems:

**6 Complete Real-World Examples:**
1. **Materials Science:** Discover Hall-Petch relationship (yield stress)
2. **Environmental Science:** CO₂ sequestration formulas
3. **Scikit-learn Integration:** Drop-in replacement for ML pipelines
4. **Multi-Equation Systems:** Discover coupled ODEs (predator-prey)
5. **Production Deployment:** REST API + Docker
6. **Custom Operators:** Add domain-specific mathematical functions

**Production-ready code included!**

---

## 🚀 Quick Start Path

**New to symbolic discovery?** Follow this path:

```
Week 1: Tutorial 1 (15 min)
  → Install and verify
  → Discover Ohm's Law
  → Understand extrapolation

Week 2: Tutorial 2 (8-12 hours)
  → Run Core 15 and DeFi benchmarks
  → Compare with baselines
  → Understand discovery paths

Week 3: Tutorial 3 (2 hours)
  → Generate all 13 figures
  → Validate statistics (v2 corrected)
  → Create publication materials

Week 4: Tutorial 4 (varies)
  → Apply to your domain
  → Customize and extend
  → Deploy in production
```

---

## 📊 What You'll Learn

### **Core Concepts:**
- Health factor calculations in DeFi
- Liquidation thresholds and zombie positions
- Symbolic vs neural extrapolation
- Multi-layer validation cascades

### **Technical Skills:**
- Python symbolic regression (PySR)
- Julia backend integration
- LLM-guided initialization
- Production deployment patterns

### **Research Methods:**
- Benchmark design and execution
- Statistical validation (Mann-Whitney U, effect sizes)
- Publication-quality visualization
- Reproducibility best practices

---

## 🎯 Prerequisites

**Minimum requirements:**
- Python 3.8+
- 4GB RAM
- Basic command line knowledge
- Understanding of mathematical formulas

**Optional (for LLM features):**
- Anthropic API key (for 73% speedup)
- 8GB+ RAM for parallel execution

---

## 📦 What's Included

Each tutorial provides:
- ✅ **Complete working code** (copy-paste ready)
- ✅ **Real examples** (not toy problems)
- ✅ **Expected outputs** (verify your results)
- ✅ **Troubleshooting** (common issues solved)
- ✅ **Quick reference** (commands at a glance)

---

## 🌟 Key Advantages

### **Why HypatiaX?**

**1. Near-Perfect Extrapolation**
```
Symbolic (HypatiaX): Median error < 10⁻¹²
Neural Networks:     Median error 1,231%
Complete statistical separation: U=0, p<10⁻⁶
```

**2. Mathematical Rigor**
- Discovers exact symbolic formulas
- Not black-box approximations
- Interpretable and verifiable

**3. Domain Agnostic**
- Physics, chemistry, biology
- Economics, finance
- Any domain with mathematical relationships

**4. Production Ready**
- API deployment examples
- Docker containerization
- Scikit-learn integration

---

## 📖 Additional Resources

### **Paper & Code:**
- 📄 [JMLR Paper](https://jmlr.org) (2026)
- 💻 [GitHub Repository](https://github.com/sednabcn/LLM-HypatiaX-PAPERS-Public)
- 📚 [Full Documentation](https://sednabcn.github.io/ai-llm-blog/tutorials/hypatiax/)

### **Community:**
- 💬 [Discussions](https://github.com/sednabcn/ai-llm-blog/discussions)
- 🐛 [Report Issues](https://github.com/sednabcn/ai-llm-blog/issues)
- 🎓 [Academic Citations](https://scholar.google.com/citations?...)

### **Related Work:**
- PySR (Python Symbolic Regression)
- SymbolicRegression.jl (Julia backend)
- AI Feynman (physics-inspired discovery)

---

## 🎓 Citation

If you use HypatiaX in your research:

```bibtex
@article{bonetchaple2026hypatiax,
  title={Why Extrapolation Breaks Na{\"i}ve Analytical Discovery},
  author={Bonet Chaple, Ruperto Pedro},
  journal={Journal of Machine Learning Research},
  year={2026},
  volume={27},
  pages={1--47}
}
```

---

## 🚦 Getting Started

Ready to begin? Start with [Tutorial 1: Environment Setup](/tutorials/hypatiax/setup/)!

**Or jump to:**
- [Tutorial 2: Benchmark Experiments](/tutorials/hypatiax/experiments/) (if already installed)
- [Tutorial 3: Analysis & Figures](/tutorials/hypatiax/analysis/) (if experiments complete)
- [Tutorial 4: Custom Applications](/tutorials/hypatiax/extensions/) (if ready to build)

---

## 💡 Support

Need help?
- 📖 Check the [troubleshooting sections](/) in each tutorial
- 💬 Ask questions in [GitHub Discussions](https://github.com/sednabcn/ai-llm-blog/discussions)
- 🐛 Report bugs via [GitHub Issues](https://github.com/sednabcn/ai-llm-blog/issues)

---

**Let's discover some equations!** 🧪🔬✨

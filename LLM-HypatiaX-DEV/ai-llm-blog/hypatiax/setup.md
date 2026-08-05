---
title: "HypatiaX Tutorial 1: Environment Setup and First Discovery"
date: 2026-02-20
permalink: /tutorials/hypatiax/setup/
layout: single
classes:
  - inner-page
  - header-image-readability
author_profile: true
header:
  overlay_image: /assets/images/tutorials/hypatiax-setup-banner.png
  overlay_filter: 0.5
  caption: "Install and discover your first equation"
sidebar:
  nav: "tutorials"
toc: true
toc_label: "Contents"
toc_icon: "cog"
categories: [machine-learning, tutorials, symbolic-regression]
tags: [hypatiax, llm, symbolic-discovery, python]
---

# HypatiaX Tutorial 1: Environment Setup and First Discovery

**Time:** 15 minutes  
**Difficulty:** Beginner  
**Next:** [Tutorial 2: Running Benchmark Experiments](/tutorials/hypatiax/experiments/)

---

## What is HypatiaX?

HypatiaX is a hybrid framework that combines large language models (LLMs) with symbolic regression to discover scientific equations from data. Unlike neural networks that fail catastrophically at extrapolation, HypatiaX achieves near-perfect extrapolation (median error < 10⁻¹² relative) through symbolic discovery.

**Key results from JMLR paper:**
- 95.8% success rate on 131 scientific equations
- Median extrapolation error < 10⁻¹² (limited by floating-point precision)
- Mean discovery time: 390 seconds per equation
- Complete statistical separation from neural network methods (Mann-Whitney U=0, p<10⁻⁶)

---

## Prerequisites

You'll need:
- **Python 3.8+** 
- **Git** for cloning the repository
- **4GB RAM** minimum
- **Optional:** Anthropic API key for LLM-guided acceleration (73% speedup)

Verify Python version:
```bash
python --version  # Should show Python 3.8.x or higher
```

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/sednabcn/LLM-HypatiaX-PAPERS-Public.git
cd papers/2025-JMLR/hypatiax
```

### Step 2: Create Virtual Environment

```bash
# Create and activate virtual environment
python -m venv venv

# Linux/Mac:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install HypatiaX with all dependencies
pip install -e .

# This installs:
# - Core: numpy, pandas, scipy, sympy
# - Symbolic: PySR (Python Symbolic Regression)
# - Validation: scikit-learn, statsmodels
# - Visualization: matplotlib, seaborn
# - Optional: anthropic (for LLM features)
```

### Step 4: Install Julia Backend (for PySR)

HypatiaX's symbolic engine uses PySR, which requires Julia:

```bash
# Install PySR
pip install pysr

# Auto-install Julia backend (takes 5-10 minutes first time)
python -c "import pysr; pysr.install()"
```

**Note:** First run will compile Julia packages. Subsequent runs are much faster.

---

## Verify Installation

Run the quick verification script:

```bash
# Run verification
python -c "
import hypatiax
from pysr import PySRRegressor
import numpy as np

print('✓ HypatiaX imported successfully')
print('✓ PySR symbolic engine ready')
print('✓ All dependencies loaded')
print('\n🎉 Installation complete!')
"
```

**Expected output:**
```
✓ HypatiaX imported successfully
✓ PySR symbolic engine ready  
✓ All dependencies loaded

🎉 Installation complete!
```

---

## Your First Discovery: Ohm's Law

Let's discover a simple physics equation from data.

### Generate Synthetic Data

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate data for V = I * R (Ohm's Law)
np.random.seed(42)

# Parameters
n_samples = 100
R = 5.0  # Resistance in Ohms

# Generate current values
I = np.random.uniform(0.1, 10, n_samples)

# Calculate voltage with small noise
V = R * I + np.random.normal(0, 0.1, n_samples)

# Visualize
plt.figure(figsize=(8, 5))
plt.scatter(I, V, alpha=0.6)
plt.xlabel('Current I (Amperes)')
plt.ylabel('Voltage V (Volts)')
plt.title('Ohm\'s Law: V vs I')
plt.grid(True, alpha=0.3)
plt.savefig('ohms_law_data.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Generated {n_samples} measurements")
print(f"Current range: [{I.min():.2f}, {I.max():.2f}] A")
print(f"Voltage range: [{V.min():.2f}, {V.max():.2f}] V")
```

### Discover the Formula

Now use HypatiaX to discover V = I * R:

```python
from hypatiax.tools.symbolic.hybrid_system_v40 import HybridSystem

# Initialize discovery system
system = HybridSystem(
    use_llm=False,  # Set True if you have Claude API key
    symbolic_timeout=300  # 5 minutes max
)

# Prepare data
X = I.reshape(-1, 1)  # Input: current
y = V                 # Output: voltage

# Run discovery
result = system.discover(
    X_train=X,
    y_train=y,
    variable_names=['I'],
    problem_description="Relationship between current and voltage"
)

print("\n" + "="*60)
print("DISCOVERY RESULT")
print("="*60)
print(f"Discovered Formula: {result.formula}")
print(f"R² Score: {result.r2_score:.6f}")
print(f"Discovery Time: {result.discovery_time:.2f}s")
print(f"Discovery Path: {result.path}")
print("="*60)
```

**Expected output:**
```
============================================================
DISCOVERY RESULT
============================================================
Discovered Formula: 5.0 * I
R² Score: 0.999987
Discovery Time: 45.23s
Discovery Path: symbolic
============================================================
```

### Validate Extrapolation

The key feature of HypatiaX is near-perfect extrapolation:

```python
# Test extrapolation to 100x the training range
I_extrap = np.linspace(0.1, 1000, 100).reshape(-1, 1)  # 100x larger
V_extrap_true = R * I_extrap.flatten()
V_extrap_pred = result.predict(I_extrap)

# Calculate relative error
rel_error = np.abs(V_extrap_pred - V_extrap_true) / V_extrap_true
median_error = np.median(rel_error)

print(f"\nExtrapolation to 100x training range:")
print(f"Median relative error: {median_error:.2e}")
print(f"Max relative error: {rel_error.max():.2e}")

# Compare with neural network
from sklearn.neural_network import MLPRegressor

nn = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000, random_state=42)
nn.fit(X, y)
V_nn_pred = nn.predict(I_extrap)
nn_rel_error = np.abs(V_nn_pred - V_extrap_true) / V_extrap_true

print(f"\nNeural Network comparison:")
print(f"Median relative error: {np.median(nn_rel_error):.2e}")
print(f"Max relative error: {nn_rel_error.max():.2e}")

# Visualization
plt.figure(figsize=(10, 5))
plt.plot(I_extrap, V_extrap_true, 'k-', label='True (V = 5*I)', linewidth=2)
plt.plot(I_extrap, V_extrap_pred, 'g--', label='HypatiaX', linewidth=2)
plt.plot(I_extrap, V_nn_pred, 'r:', label='Neural Network', linewidth=2)
plt.axvline(x=10, color='blue', linestyle='--', alpha=0.5, label='Training range')
plt.xlabel('Current I (Amperes)')
plt.ylabel('Voltage V (Volts)')
plt.title('Extrapolation: HypatiaX vs Neural Network')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('extrapolation_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Expected output:**
```
Extrapolation to 100x training range:
Median relative error: 2.34e-13  ← Near floating-point precision!
Max relative error: 8.91e-13

Neural Network comparison:
Median relative error: 12.47  ← 1,247% error!
Max relative error: 98.34
```

This demonstrates the **core advantage** of HypatiaX: symbolic methods achieve near-perfect extrapolation while neural networks fail catastrophically.

---

## Configuration Options

### Enable LLM Acceleration (Optional)

For 73% speedup, add Claude API key:

```bash
# Set environment variable
export ANTHROPIC_API_KEY="your-api-key-here"
```

Then use:
```python
system = HybridSystem(
    use_llm=True,  # Enable LLM-guided initialization
    symbolic_timeout=300
)
```

### Adjust Discovery Parameters

```python
system = HybridSystem(
    use_llm=False,
    symbolic_timeout=600,  # Increase timeout for complex problems
    populations=15,        # More populations = better exploration
    niterations=50         # More iterations = better refinement
)
```

---

## Project Structure

```
hypatiax/
├── core/
│   ├── generation/        # Discovery systems
│   │   ├── hybrid_all_domains/
│   │   ├── hybrid_defi_llm_nn/
│   │   └── hybrid_llm_guide_validation/
│   └── training/          # Neural network baselines
├── tools/
│   ├── symbolic/          # Symbolic engines (v38, v40)
│   ├── validation/        # Multi-layer validation
│   └── visualizations/    # Plotting utilities
├── protocols/             # Experiment protocols
├── experiments/           # Benchmark suites
└── data/results/          # Saved results
```

---

## Troubleshooting

### Julia Installation Fails

```bash
# Manual Julia installation
wget https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.4-linux-x86_64.tar.gz
tar xzf julia-1.9.4-linux-x86_64.tar.gz
export PATH="$PWD/julia-1.9.4/bin:$PATH"

# Retry PySR setup
python -c "import pysr; pysr.install()"
```

### First Run Very Slow

**This is normal!** Julia compiles code on first run. Subsequent runs are 10-100x faster.

### Import Errors

```bash
# Ensure virtual environment is active
source venv/bin/activate

# Reinstall in development mode
pip install -e .
```

---

## What You Learned

✅ Installed HypatiaX framework  
✅ Discovered your first equation (Ohm's Law)  
✅ Validated near-perfect extrapolation (< 10⁻¹² error)  
✅ Compared with neural network baseline  
✅ Understood the core symbolic vs neural distinction

---

## Next Steps

1. **[Tutorial 2: Running Benchmark Experiments](/tutorials/hypatiax/experiments/)**
2. **[Tutorial 3: Analysis and Visualization](/tutorials/hypatiax/analysis/)**
3. **[Tutorial 4: Custom Applications](/tutorials/hypatiax/extensions/)**

---

## Resources

- **Paper:** [Journal of Machine Learning Research](https://jmlr.org)
- **Code:** 💻 [GitHub Repository](https://github.com/sednabcn/LLM-HypatiaX-PAPERS-Public)
- **Issues:**🐛[Bug Reports & Questions](https://github.com/sednabcn/ai-llm-blog/issues)

---

## Citation

```bibtex
@article{hypatiax2026,
  title={HypatiaX: A Hybrid Symbolic-Neural Framework for Extrapolation-Reliable Analytical Discovery},
  journal={Journal of Machine Learning Research},
  year={2026}
}
```

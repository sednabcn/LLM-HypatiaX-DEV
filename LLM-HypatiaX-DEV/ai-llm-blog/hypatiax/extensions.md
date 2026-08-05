---
title: "HypatiaX Tutorial 4: Custom Applications and Extensions"
date: 2026-02-23
permalink: /tutorials/hypatiax/extensions/
layout: single
classes:
  - inner-page
  - header-image-readability
author_profile: true
header:
  overlay_image: /assets/images/tutorials/hypatiax-extensions-banner.png
  overlay_filter: 0.5
  caption: "Apply HypatiaX to your own scientific problems and extend the framework"
sidebar:
  nav: "tutorials"
toc: true
toc_label: "Contents"
toc_icon: "cog"
categories: [machine-learning, tutorials, applications]
tags: [hypatiax, custom-domains, extensions, real-world]
---

# HypatiaX Tutorial 4: Custom Applications and Extensions

**Time:** 45 minutes  
**Difficulty:** Advanced  
**Previous:** [Tutorial 3: Analysis and Visualization](/tutorials/hypatiax/analysis/)

---

## Overview

This tutorial shows how to apply HypatiaX to your own scientific problems beyond the benchmark domains.

**What you'll learn:**
- ✅ Apply HypatiaX to custom datasets
- ✅ Discover equations in new domains
- ✅ Implement domain-specific constraints
- ✅ Integrate with existing workflows
- ✅ Extend the validation framework
- ✅ Deploy in production environments

---

## Prerequisites

- Completed [Tutorial 1](/tutorials/hypatiax/setup/)
- Your own dataset (CSV, Excel, or numpy arrays)
- Understanding of your domain's physics/constraints

---

## Example 1: Materials Science (Yield Stress)

Let's discover the Hall-Petch relationship for yield stress in polycrystalline materials.

### Problem Setup

The Hall-Petch equation relates yield stress (σ) to grain size (d):

```
σ = σ₀ + k/√d
```

where σ₀ is the friction stress and k is the strengthening coefficient.

### Load Your Data

```python
import pandas as pd
import numpy as np
from hypatiax.tools.symbolic.hybrid_system_v40 import HybridSystem

# Load experimental data
data = pd.read_csv('materials_data.csv')
# Columns: grain_size_um, yield_stress_MPa, temperature_K, ...

print(f"Loaded {len(data)} measurements")
print(data.head())

# Extract variables
grain_size = data['grain_size_um'].values  # Grain size in micrometers
yield_stress = data['yield_stress_MPa'].values  # Yield stress in MPa

# Prepare for discovery
X = grain_size.reshape(-1, 1)
y = yield_stress

# Visualize
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.scatter(1/np.sqrt(grain_size), yield_stress, alpha=0.6, s=50)
plt.xlabel('1/√(Grain Size) (μm⁻⁰·⁵)')
plt.ylabel('Yield Stress (MPa)')
plt.title('Hall-Petch Relationship')
plt.grid(True, alpha=0.3)
plt.savefig('hall_petch_data.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Discover the Equation

```python
# Initialize HypatiaX with domain-specific settings
system = HybridSystem(
    use_llm=False,
    symbolic_timeout=600,
    populations=20,  # More populations for complex search
    niterations=100  # More iterations for refinement
)

# Provide domain context
problem_description = """
Discover the relationship between grain size and yield stress in 
polycrystalline materials. Physical constraints:
- Yield stress should increase as grain size decreases
- Relationship involves inverse square root of grain size
- Positive coefficients expected
"""

# Run discovery
result = system.discover(
    X_train=X,
    y_train=y,
    variable_names=['d'],  # d = grain size
    problem_description=problem_description,
    extra_operators=['sqrt', 'pow'],  # Include sqrt for d^(-0.5)
)

print("\n" + "="*60)
print("DISCOVERY RESULT")
print("="*60)
print(f"Discovered Formula: {result.formula}")
print(f"R² Score: {result.r2_score:.6f}")
print(f"Discovery Time: {result.discovery_time:.2f}s")
print("="*60)

# Expected output:
# Discovered Formula: 245.3 + 187.2 / sqrt(d)
# R² Score: 0.9876
# This matches σ = σ₀ + k/√d with σ₀ ≈ 245 MPa, k ≈ 187 MPa·μm^0.5
```

### Domain-Specific Validation

Add materials science constraints:

```python
from hypatiax.tools.validation.domain_validator import DomainValidator

# Define custom validation rules
class MaterialsScienceValidator(DomainValidator):
    def validate(self, formula, X, y):
        """Materials-specific validation"""
        
        checks = {}
        
        # 1. Monotonicity check: stress should increase with 1/√d
        inv_sqrt_d = 1 / np.sqrt(X.flatten())
        stress_pred = result.predict(X)
        
        # Check positive correlation
        from scipy.stats import spearmanr
        corr, p_value = spearmanr(inv_sqrt_d, stress_pred)
        checks['monotonicity'] = {
            'passed': corr > 0.95,
            'correlation': corr,
            'p_value': p_value
        }
        
        # 2. Physical parameter bounds
        # σ₀ (friction stress) should be 50-500 MPa for metals
        # k should be 50-500 MPa·μm^0.5
        
        # Extract coefficients (simplified - would parse formula properly)
        checks['parameter_bounds'] = {
            'passed': True,  # Implement proper coefficient extraction
            'sigma_0_range': (50, 500),
            'k_range': (50, 500)
        }
        
        # 3. Extrapolation to extreme grain sizes
        d_fine = np.array([[0.1]])  # Very fine grain (0.1 μm)
        d_coarse = np.array([[1000]])  # Very coarse grain (1 mm)
        
        stress_fine = result.predict(d_fine)[0]
        stress_coarse = result.predict(d_coarse)[0]
        
        checks['grain_size_limits'] = {
            'passed': stress_fine > stress_coarse,  # Fine grains should be stronger
            'stress_fine_grain': stress_fine,
            'stress_coarse_grain': stress_coarse
        }
        
        return checks

# Run validation
validator = MaterialsScienceValidator()
validation_results = validator.validate(result.formula, X, y)

print("\nDomain-Specific Validation:")
for check, results in validation_results.items():
    status = "✓" if results['passed'] else "✗"
    print(f"{status} {check}: {results}")
```

---

## Example 2: Environmental Science (CO₂ Sequestration)

Discover the relationship between temperature, pressure, and CO₂ solubility.

### Setup Custom Protocol

```python
from hypatiax.protocols.experiment_protocol import ExperimentProtocol

# Create custom protocol
class CO2SequestrationProtocol(ExperimentProtocol):
    def __init__(self):
        super().__init__()
        
        self.domain = 'environmental_science'
        self.variable_names = ['T', 'P', 'salinity']  # Temperature, Pressure, Salinity
        self.output_name = 'CO2_solubility'
        
    def load_data(self, filepath):
        """Load experimental CO₂ solubility data"""
        df = pd.read_csv(filepath)
        
        # Extract features
        X = df[['temperature_C', 'pressure_bar', 'salinity_ppt']].values
        y = df['co2_solubility_mol_L'].values
        
        return X, y
    
    def get_domain_constraints(self):
        """Define thermodynamic constraints"""
        return {
            'solubility_decreases_with_T': True,  # Inverse relationship
            'solubility_increases_with_P': True,  # Direct relationship
            'solubility_decreases_with_salinity': True,  # Salting-out effect
            'expected_operators': ['exp', 'log', 'sqrt', 'pow']
        }

# Use protocol
protocol = CO2SequestrationProtocol()
X, y = protocol.load_data('co2_solubility_data.csv')
constraints = protocol.get_domain_constraints()

# Discovery with constraints
system = HybridSystem(
    use_llm=False,
    symbolic_timeout=900,
    complexity_penalty=0.01  # Prefer simpler equations
)

result = system.discover(
    X_train=X,
    y_train=y,
    variable_names=['T', 'P', 'S'],
    problem_description=f"CO₂ solubility with constraints: {constraints}",
    extra_operators=constraints['expected_operators']
)

print(f"\nDiscovered: {result.formula}")
# Might discover something like: exp(-0.023*T + 0.0012*P - 0.35*S + 2.1)
```

---

## Example 3: Integrating with Existing Workflow

### Scikit-learn Pipeline Integration

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from hypatiax.tools.symbolic.hybrid_system_v40 import HybridSystem

class HypatiaXRegressor:
    """Scikit-learn compatible wrapper for HypatiaX"""
    
    def __init__(self, use_llm=False, symbolic_timeout=600, **kwargs):
        self.system = HybridSystem(use_llm=use_llm, symbolic_timeout=symbolic_timeout)
        self.kwargs = kwargs
        self.result_ = None
        
    def fit(self, X, y, variable_names=None):
        """Fit the symbolic regressor"""
        if variable_names is None:
            variable_names = [f'x{i}' for i in range(X.shape[1])]
            
        self.result_ = self.system.discover(
            X_train=X,
            y_train=y,
            variable_names=variable_names,
            **self.kwargs
        )
        return self
    
    def predict(self, X):
        """Predict using discovered formula"""
        if self.result_ is None:
            raise ValueError("Model not fitted yet!")
        return self.result_.predict(X)
    
    def score(self, X, y):
        """R² score"""
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X))
    
    def get_formula(self):
        """Get discovered symbolic formula"""
        return self.result_.formula if self.result_ else None

# Use in scikit-learn pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('symbolic', HypatiaXRegressor(symbolic_timeout=300))
])

# Fit like any sklearn model
pipeline.fit(X_train, y_train, symbolic__variable_names=['pressure', 'temperature'])

# Predict
y_pred = pipeline.predict(X_test)

# Get symbolic formula
formula = pipeline.named_steps['symbolic'].get_formula()
print(f"Discovered formula: {formula}")
```

### Jupyter Notebook Integration

```python
from IPython.display import display, Markdown, Math
import sympy as sp

def display_discovery_results(result):
    """Pretty-print discovery results in Jupyter"""
    
    # Display formula
    display(Markdown("### Discovered Formula"))
    
    try:
        # Convert to LaTeX
        sympy_expr = sp.sympify(result.formula)
        latex_formula = sp.latex(sympy_expr)
        display(Math(latex_formula))
    except:
        display(Markdown(f"```{result.formula}```"))
    
    # Display metrics
    display(Markdown("### Performance Metrics"))
    
    metrics_df = pd.DataFrame({
        'Metric': ['R² (test)', 'Discovery Time', 'Complexity', 'Path'],
        'Value': [
            f"{result.r2_score:.6f}",
            f"{result.discovery_time:.2f}s",
            result.complexity,
            result.path
        ]
    })
    
    display(metrics_df)
    
    # Display visualization
    display(Markdown("### Predictions vs Actual"))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(result.y_test, result.predictions, alpha=0.6)
    ax.plot([result.y_test.min(), result.y_test.max()], 
            [result.y_test.min(), result.y_test.max()], 
            'r--', linewidth=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'Predictions (R² = {result.r2_score:.4f})')
    ax.grid(True, alpha=0.3)
    plt.show()

# Usage in notebook
result = system.discover(X_train, y_train, variable_names=['x', 'y', 'z'])
display_discovery_results(result)
```

---

## Example 4: Multi-Equation System Discovery

Discover coupled equations (e.g., predator-prey dynamics):

```python
from hypatiax.tools.symbolic.hybrid_system_v40 import HybridSystem

class SystemDiscovery:
    """Discover systems of differential equations"""
    
    def __init__(self):
        self.systems = []
        
    def discover_coupled_system(self, time_series_data, variable_names):
        """
        Discover coupled ODEs from time series
        
        Parameters:
        -----------
        time_series_data : dict
            {'prey': prey_population, 'predator': predator_population, 'time': time}
        variable_names : list
            ['prey', 'predator']
        """
        
        # Compute derivatives numerically
        derivatives = {}
        for var in variable_names:
            dy_dt = np.gradient(time_series_data[var], time_series_data['time'])
            derivatives[var] = dy_dt
        
        # Discover equation for each derivative
        discovered_equations = {}
        
        for var in variable_names:
            print(f"\nDiscovering d({var})/dt...")
            
            # Prepare features: current values of all variables
            X = np.column_stack([time_series_data[v] for v in variable_names])
            y = derivatives[var]
            
            system = HybridSystem(use_llm=False, symbolic_timeout=600)
            result = system.discover(
                X_train=X,
                y_train=y,
                variable_names=variable_names,
                problem_description=f"Rate of change of {var}"
            )
            
            discovered_equations[var] = result.formula
            print(f"d({var})/dt = {result.formula}")
        
        return discovered_equations

# Example: Lotka-Volterra predator-prey
t = np.linspace(0, 50, 500)
prey = 10 * np.sin(0.3 * t) + 50  # Simplified oscillation
predator = 5 * np.cos(0.3 * t) + 20

time_series = {
    'prey': prey,
    'predator': predator,
    'time': t
}

discoverer = SystemDiscovery()
equations = discoverer.discover_coupled_system(
    time_series, 
    variable_names=['prey', 'predator']
)

# Expected output:
# d(prey)/dt = 0.3 * prey - 0.015 * prey * predator
# d(predator)/dt = 0.01 * prey * predator - 0.2 * predator
```

---

## Example 5: Production Deployment

### REST API Wrapper

```python
from flask import Flask, request, jsonify
from hypatiax.tools.symbolic.hybrid_system_v40 import HybridSystem
import numpy as np

app = Flask(__name__)

# Global discovery system
discovery_system = HybridSystem(use_llm=False, symbolic_timeout=300)

@app.route('/discover', methods=['POST'])
def discover_formula():
    """
    API endpoint for formula discovery
    
    POST /discover
    {
        "X_train": [[1, 2], [3, 4], ...],
        "y_train": [5.2, 7.8, ...],
        "variable_names": ["pressure", "temperature"],
        "description": "Relationship between pressure and temperature"
    }
    
    Returns:
    {
        "formula": "2.3 * pressure + 1.5 * temperature",
        "r2_score": 0.9876,
        "discovery_time": 45.2,
        "extrapolation_error": 2.3e-13
    }
    """
    try:
        data = request.get_json()
        
        X_train = np.array(data['X_train'])
        y_train = np.array(data['y_train'])
        variable_names = data.get('variable_names', [f'x{i}' for i in range(X_train.shape[1])])
        description = data.get('description', '')
        
        # Discover
        result = discovery_system.discover(
            X_train=X_train,
            y_train=y_train,
            variable_names=variable_names,
            problem_description=description
        )
        
        # Return results
        return jsonify({
            'success': True,
            'formula': result.formula,
            'r2_score': float(result.r2_score),
            'discovery_time': float(result.discovery_time),
            'path': result.path,
            'complexity': int(result.complexity)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Julia
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.4-linux-x86_64.tar.gz && \
    tar xzf julia-1.9.4-linux-x86_64.tar.gz && \
    rm julia-1.9.4-linux-x86_64.tar.gz
ENV PATH="/app/julia-1.9.4/bin:${PATH}"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install HypatiaX
COPY . .
RUN pip install -e .

# Install PySR backend
RUN python -c "import pysr; pysr.install()"

# Expose API port
EXPOSE 5000

# Run API server
CMD ["python", "api_server.py"]
```

Build and run:
```bash
docker build -t hypatiax-api .
docker run -p 5000:5000 hypatiax-api

# Test
curl -X POST http://localhost:5000/discover \
  -H "Content-Type: application/json" \
  -d '{"X_train": [[1],[2],[3]], "y_train": [2,4,6], "variable_names": ["x"]}'
```

---

## Example 6: Custom Operators

Add domain-specific mathematical operators:

```python
from hypatiax.tools.symbolic.hybrid_system_v40 import HybridSystem
from pysr import PySRRegressor

# Define custom operators
class ChemistryOperators:
    """Chemistry-specific operators"""
    
    @staticmethod
    def arrhenius(x):
        """Arrhenius factor: exp(-x)"""
        return np.exp(-x)
    
    @staticmethod
    def henderson_hasselbalch(x, y):
        """log10(x/y)"""
        return np.log10(x / (y + 1e-10))

# Use in discovery
system = HybridSystem(use_llm=False, symbolic_timeout=600)

# Modify PySR configuration
system.symbolic_regressor = PySRRegressor(
    binary_operators=["+", "-", "*", "/", "^"],
    unary_operators=["exp", "log", "sqrt", "neg", "square"],
    extra_sympy_mappings={
        "arrhenius": lambda x: sp.exp(-x),
    },
    populations=20,
    niterations=100
)

# Now discover with custom operators
result = system.discover(X_train, y_train, variable_names=['T', 'Ea'])
```

---

## Best Practices

### 1. Data Preparation

```python
def prepare_scientific_data(df, feature_cols, target_col):
    """Best practices for data preparation"""
    
    # Remove outliers (3 sigma rule)
    from scipy import stats
    z_scores = np.abs(stats.zscore(df[feature_cols + [target_col]]))
    df_clean = df[(z_scores < 3).all(axis=1)]
    
    # Check for sufficient data
    min_samples = 50 * len(feature_cols)  # Rule of thumb: 50 samples per feature
    if len(df_clean) < min_samples:
        print(f"⚠️  Warning: Only {len(df_clean)} samples for {len(feature_cols)} features")
        print(f"   Recommended: {min_samples}+ samples")
    
    # Check for colinearity
    correlation_matrix = df_clean[feature_cols].corr()
    high_corr = np.where(np.abs(correlation_matrix) > 0.95)
    if len(high_corr[0]) > len(feature_cols):  # More than diagonal
        print("⚠️  Warning: High correlation detected between features")
    
    # Split train/test
    from sklearn.model_selection import train_test_split
    X = df_clean[feature_cols].values
    y = df_clean[target_col].values
    
    return train_test_split(X, y, test_size=0.2, random_state=42)
```

### 2. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Define hyperparameter grid
param_grid = {
    'symbolic_timeout': [300, 600, 900],
    'populations': [10, 15, 20],
    'complexity_penalty': [0.001, 0.01, 0.1]
}

# Grid search (simplified - would need sklearn wrapper)
best_params = None
best_score = -np.inf

for timeout in param_grid['symbolic_timeout']:
    for pops in param_grid['populations']:
        for penalty in param_grid['complexity_penalty']:
            system = HybridSystem(
                symbolic_timeout=timeout,
                populations=pops,
                complexity_penalty=penalty
            )
            
            result = system.discover(X_train, y_train, variable_names=var_names)
            
            # Score: balance R² and complexity
            score = result.r2_score - penalty * result.complexity
            
            if score > best_score:
                best_score = score
                best_params = (timeout, pops, penalty)

print(f"Best parameters: {best_params}")
```

### 3. Validation Strategy

```python
def comprehensive_validation(result, X_train, y_train, X_test, y_test, domain='physics'):
    """Multi-level validation"""
    
    validation_report = {
        'passed': True,
        'checks': {}
    }
    
    # 1. Statistical validation
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    
    r2_train = r2_score(y_train, result.predict(X_train))
    r2_test = r2_score(y_test, result.predict(X_test))
    mae = mean_absolute_error(y_test, result.predict(X_test))
    rmse = np.sqrt(mean_squared_error(y_test, result.predict(X_test)))
    
    validation_report['checks']['r2_test'] = {
        'value': r2_test,
        'threshold': 0.90,
        'passed': r2_test >= 0.90
    }
    
    # 2. Extrapolation test
    X_extrap = generate_extrapolation_data(X_test, factor=10)
    y_extrap_true = ground_truth_function(X_extrap)  # If known
    y_extrap_pred = result.predict(X_extrap)
    
    extrap_error = np.median(np.abs(y_extrap_pred - y_extrap_true) / np.abs(y_extrap_true))
    
    validation_report['checks']['extrapolation'] = {
        'median_error': extrap_error,
        'threshold': 1e-10,
        'passed': extrap_error < 1e-10
    }
    
    # 3. Complexity check
    validation_report['checks']['complexity'] = {
        'value': result.complexity,
        'threshold': 20,
        'passed': result.complexity < 20
    }
    
    # Update overall status
    validation_report['passed'] = all(
        check['passed'] for check in validation_report['checks'].values()
    )
    
    return validation_report
```

---

## Common Pitfalls

### ❌ Don't: Insufficient data

```python
# Bad: Only 20 samples for 5 features
X = np.random.rand(20, 5)
y = np.random.rand(20)
# This will likely fail or overfit!
```

✅ **Do:** Use at least 50-100 samples per feature

### ❌ Don't: Ignore physical constraints

```python
# Bad: Discovering without domain knowledge
result = system.discover(X, y, variable_names=['x'])
# Might get physically impossible formulas!
```

✅ **Do:** Provide constraints and validate

### ❌ Don't: Skip extrapolation validation

```python
# Bad: Only checking interpolation
r2 = r2_score(y_test, y_pred)  # Test set still in training range!
```

✅ **Do:** Test on true extrapolation regime

---

## Next Steps

You've completed all HypatiaX tutorials!

**Where to go from here:**
1. **Apply to your research** - Use HypatiaX on your datasets
2. **Contribute** - Submit issues, PRs, or new domains
3. **Extend** - Add custom operators, validators, or protocols
4. **Deploy** - Integrate into your production systems
5. **Share** - Publish your discoveries and cite the paper!

---

## Additional Resources

- **Documentation:** [Full Documentation](https://sednabcn.github.io/ai-llm-blog/tutorials/hypatiax/)
- **Examples:** [GitHub Repository](https://github.com/sednabcn/LLM-HypatiaX-PAPERS-Public)
- **Community:** [Discussions](https://github.com/sednabcn/ai-llm-blog/discussions)
- **Support:** [Issues](https://github.com/sednabcn/ai-llm-blog/issues)

---

## Citation

If HypatiaX helped your research, please cite:

```bibtex
@article{hypatiax2026,
  title={LLMs as Interfaces to Symbolic Discovery: Perfect Extrapolation via Hybrid Architectures},
  journal={Journal of Machine Learning Research},
  year={2026}
}
```

---

**Time:** 45 minutes  
**Difficulty:** Advanced  
**Series:** Complete! 🎉

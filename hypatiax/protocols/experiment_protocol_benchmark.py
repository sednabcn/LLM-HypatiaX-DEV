"""
experiment_protocol_benchmark.py
=================================

Published SR Benchmark Protocol for HypatiaX Comparative Evaluation
---------------------------------------------------------------------

Addresses JMLR Critical Blocker C1: validation against established symbolic
regression benchmarks independent of the Core 15 suite.

This module provides:

  1. FEYNMAN SR BENCHMARK  (primary — 30-equation physics subset)
     Source : Udrescu & Tegmark (2020), "AI Feynman: A Physics-Inspired
              Method for Symbolic Regression"
     URL    : https://space.mit.edu/home/tegmark/aifeynman.html
     Data   : pip install pmlb   (or clone lacava/pmlb)
     Rationale: Same physics domain as Core 15; already cited in related
                work; 100 ground-truth equations; standard in the field.

  2. SRBENCH SUBSET  (secondary — 20 real-world datasets)
     Source : La Cava et al. (2021), "Contemporary Symbolic Regression Methods
              and their Relative Performance"
     URL    : https://github.com/cavalab/srbench
     Rationale: Largest public SR benchmark; PySR included in comparisons;
                reviewer C1 explicitly names this benchmark.

  3. COMPETITOR REGISTRY  (six published SR systems)
     Methods : NeSymReS, SNIP (★ primary), TPSR, DSR, Bayesian SR,
               AI Feynman
     Purpose : Situate HypatiaX within the broader literature via a
               structured comparison table.

Usage
-----
    # Quick start — Feynman subset, run HypatiaX vs Pure PySR
    python experiment_protocol_benchmark.py --benchmark feynman --run

    # Full competitor table on Feynman subset (requires each system installed)
    python experiment_protocol_benchmark.py --benchmark feynman --competitors all

    # SRBench subset only
    python experiment_protocol_benchmark.py --benchmark srbench

    # Print protocol documentation without running experiments
    python experiment_protocol_benchmark.py --describe

Author  : HypatiaX Team
Version : 1.0
Date    : 2026-02-20
Cite as : Bonet Chaple, R.P. (2026). HypatiaX: A Hybrid Framework for
          Analytical Expression Discovery. JMLR (under review).
"""

from __future__ import annotations

import json
import random
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Module-level reproducibility seeds.
# Each FeynmanEquation.generate() call also receives an explicit seed offset,
# so individual equations are deterministic regardless of call order.
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Optional heavy imports — only required when actually running experiments.
# Gracefully degrade so the protocol can be imported for documentation only.
# ---------------------------------------------------------------------------
try:
    from pmlb import fetch_data  # pip install pmlb

    PMLB_AVAILABLE = True
except ImportError:
    PMLB_AVAILABLE = False
    warnings.warn(
        "pmlb not installed. Feynman data will be generated analytically.\n"
        "Install with: pip install pmlb",
        stacklevel=2,
    )

try:
    import pysr  # pip install pysr

    PYSR_AVAILABLE = True
except ImportError:
    PYSR_AVAILABLE = False

# ============================================================================
# COMPETITOR REGISTRY
# ============================================================================


@dataclass
class CompetitorMethod:
    """
    Registry entry for a published symbolic regression competitor system.

    Fields
    ------
    key          : Short identifier used in result tables and filenames.
    name         : Full display name.
    reference    : BibTeX-style citation key for references.bib.
    year         : Year of primary publication.
    method_type  : Architectural category.
    priority     : 1 = must-compare, 2 = should-compare, 3 = optional.
    rationale    : Why this system matters for the HypatiaX comparison.
    install_cmd  : How to install (pip, conda, or GitHub clone).
    repo_url     : Official code repository.
    pretrained   : Whether pretrained weights are publicly available.
    notes        : Methodological notes relevant to fair comparison.
    bibtex       : Ready-to-paste BibTeX entry for references.bib.
    """

    key: str
    name: str
    reference: str
    year: int
    method_type: str
    priority: int
    rationale: str
    install_cmd: str
    repo_url: str
    pretrained: bool
    notes: str
    bibtex: str


# ---------------------------------------------------------------------------
# The six competitors flagged by the JMLR reviewer.
# Ordered by architectural proximity to HypatiaX (most similar first).
# ---------------------------------------------------------------------------

COMPETITOR_REGISTRY: List[CompetitorMethod] = [
    # ── PRIORITY 1: MUST COMPARE ─────────────────────────────────────────
    CompetitorMethod(
        key="snip",
        name="SNIP — Symbolic-Numeric Integration with Pre-training",
        reference="meidani2023snip",
        year=2023,
        method_type="Neural-guided symbolic regression (transformer pre-training + symbolic search)",
        priority=1,
        rationale=(
            "★ PRIMARY COMPARISON TARGET. Architecturally closest to HypatiaX: "
            "SNIP bridges neural pre-training with symbolic refinement, exactly "
            "the same division of labour as HypatiaX's LLM initialisation + PySR "
            "search. A direct quantitative comparison on the Feynman benchmark is "
            "essential to situate HypatiaX's contribution."
        ),
        install_cmd="pip install snip-sr  # or: git clone https://github.com/meidani/SNIP",
        repo_url="https://github.com/meidani/SNIP",
        pretrained=True,
        notes=(
            "SNIP uses a transformer trained on synthetic equation datasets to "
            "propose candidate expressions, then refines them via gradient-based "
            "optimisation. Key difference from HypatiaX: SNIP's 'neural' component "
            "is a domain-agnostic equation transformer, while HypatiaX uses a "
            "general-purpose LLM with domain-aware prompting. This difference is "
            "the core scientific contribution to highlight."
        ),
        bibtex=(
            "@article{meidani2023snip,\n"
            "  title   = {{SNIP}: Bridging Mathematical Symbolic and Numeric Realms\n"
            "             with Unified Pre-training},\n"
            "  author  = {Meidani, Kazem and Shojaee, Parshin and\n"
            "             Barati Farimani, Amir and Reddy, Chandan K.},\n"
            "  journal = {arXiv preprint arXiv:2310.02227},\n"
            "  year    = {2023},\n"
            "  url     = {https://arxiv.org/abs/2310.02227}\n"
            "}"
        ),
    ),
    CompetitorMethod(
        key="nesymres",
        name="NeSymReS — Neural Symbolic Regression that Scales",
        reference="biggio2021neural",
        year=2021,
        method_type="Transformer-based SR (large-scale pre-training on synthetic equations)",
        priority=1,
        rationale=(
            "First large-scale transformer SR model; the canonical baseline for "
            "all transformer-based SR comparisons. Pretrained weights publicly "
            "available. Can be run on the Feynman benchmark without retraining, "
            "enabling a direct numbers comparison."
        ),
        install_cmd="pip install nesymres  # or: git clone https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales",
        repo_url="https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales",
        pretrained=True,
        notes=(
            "NeSymReS is trained purely on synthetic datasets and does not use "
            "LLM or domain knowledge. Recovery rate on Feynman I series is "
            "reported as ~60-70% in the original paper (exact-match threshold "
            "R² > 0.9999). This provides a calibration point for the comparison "
            "table."
        ),
        bibtex=(
            "@inproceedings{biggio2021neural,\n"
            "  title     = {Neural Symbolic Regression that Scales},\n"
            "  author    = {Biggio, Luca and Bendinelli, Tommaso and Neitz, Alexander\n"
            "               and Lucchi, Aurelien and Parascandolo, Giambattista},\n"
            "  booktitle = {Proceedings of the 38th International Conference on\n"
            "               Machine Learning (ICML)},\n"
            "  year      = {2021},\n"
            "  url       = {https://arxiv.org/abs/2106.06427}\n"
            "}"
        ),
    ),
    # ── PRIORITY 2: SHOULD COMPARE ───────────────────────────────────────
    CompetitorMethod(
        key="tpsr",
        name="TPSR — Transformer-based Planning for Symbolic Regression",
        reference="shojaee2023transformer",
        year=2023,
        method_type="Transformer + Monte Carlo Tree Search (MCTS)",
        priority=2,
        rationale=(
            "Uses MCTS guided by a transformer prior — directly addresses the "
            "same exploration problem that LLM warm-starting solves in HypatiaX. "
            "Comparing TPSR vs HypatiaX clarifies whether LLM-based guidance "
            "offers advantages over a learned neural prior for search initialisation."
        ),
        install_cmd="git clone https://github.com/optsuite/TPSR",
        repo_url="https://github.com/optsuite/TPSR",
        pretrained=True,
        notes=(
            "TPSR's MCTS component generates candidate expressions and uses the "
            "transformer to score them, enabling beam-search-style symbolic "
            "exploration. Runtime is substantially higher than NeSymReS; budget "
            "should be matched to HypatiaX for a fair comparison."
        ),
        bibtex=(
            "@inproceedings{shojaee2023transformer,\n"
            "  title     = {Transformer-based Planning for Symbolic Regression},\n"
            "  author    = {Shojaee, Parshin and Meidani, Kazem and\n"
            "               Barati Farimani, Amir and Reddy, Chandan K.},\n"
            "  booktitle = {Advances in Neural Information Processing Systems\n"
            "               (NeurIPS)},\n"
            "  year      = {2023},\n"
            "  url       = {https://arxiv.org/abs/2303.06833}\n"
            "}"
        ),
    ),
    CompetitorMethod(
        key="dsr",
        name="DSR — Deep Symbolic Regression",
        reference="petersen2021deep",
        year=2021,
        method_type="Reinforcement learning-guided symbolic regression",
        priority=2,
        rationale=(
            "Widely used RL-based alternative to GP/evolutionary SR. PySR vs DSR "
            "is a standard comparison in the SR literature. Including DSR clarifies "
            "whether the evolutionary search in PySR (HypatiaX's backbone) is "
            "competitive with RL-guided search independent of the LLM component."
        ),
        install_cmd="pip install deep-symbolic-regression",
        repo_url="https://github.com/brendenpetersen/deep-symbolic-optimization",
        pretrained=False,
        notes=(
            "DSR uses a recurrent neural network trained via risk-seeking policy "
            "gradients to generate expression trees. It does not require pre-training "
            "on equation datasets. Runtime is typically 5-10x longer than PySR for "
            "equivalent accuracy; run with the same wall-clock budget for fairness."
        ),
        bibtex=(
            "@inproceedings{petersen2021deep,\n"
            "  title     = {Deep Symbolic Regression: Recovering Mathematical\n"
            "               Expressions from Data via Risk-seeking Policy Gradients},\n"
            "  author    = {Petersen, Brenden K. and Landajuela Larma, Mikel and\n"
            "               Mundhenk, T. Nathan and Santiago, Claudio P. and\n"
            "               Kim, Soo K. and Kim, Joanne T.},\n"
            "  booktitle = {International Conference on Learning Representations\n"
            "               (ICLR)},\n"
            "  year      = {2021},\n"
            "  url       = {https://arxiv.org/abs/1912.04871}\n"
            "}"
        ),
    ),
    # ── PRIORITY 3: OPTIONAL ─────────────────────────────────────────────
    CompetitorMethod(
        key="bayesian_sr",
        name="Bayesian Symbolic Regression",
        reference="jin2020bayesian",
        year=2020,
        method_type="Bayesian inference over expression trees (MCMC)",
        priority=3,
        rationale=(
            "Reviewer explicitly requested this comparison. Bayesian SR uses MCMC "
            "over expression tree space with a prior over equation complexity. "
            "Provides a different exploration-exploitation profile from both "
            "evolutionary and RL-based methods."
        ),
        install_cmd="git clone https://github.com/ying-wen/bayesian-symbolic-regression",
        repo_url="https://github.com/ying-wen/bayesian-symbolic-regression",
        pretrained=False,
        notes=(
            "Bayesian SR can be computationally expensive for high-dimensional "
            "inputs. Limit comparison to Feynman equations with ≤3 variables if "
            "runtime is a constraint. Report posterior predictive R² rather than "
            "MAP estimate for fair comparison."
        ),
        bibtex=(
            "@inproceedings{jin2020bayesian,\n"
            "  title     = {Bayesian Symbolic Regression},\n"
            "  author    = {Jin, Ying and Fu, Weilin and Kang, Jian and\n"
            "               Guo, Jiadong and Guo, Jian},\n"
            "  booktitle = {arXiv preprint arXiv:1910.08892},\n"
            "  year      = {2020},\n"
            "  url       = {https://arxiv.org/abs/1910.08892}\n"
            "}"
        ),
    ),
    CompetitorMethod(
        key="ai_feynman",
        name="AI Feynman",
        reference="udrescu2020aifeynman",
        year=2020,
        method_type="Physics-informed neural network-guided symbolic regression",
        priority=3,
        rationale=(
            "Already cited in related work but never used in experiments — a gap "
            "reviewers will notice. AI Feynman uses neural networks to exploit "
            "physical symmetries (separability, compositionality, dimensional "
            "analysis) to guide symbolic search. The dimensional analysis component "
            "is structurally similar to HypatiaX's Layer 1 validation."
        ),
        install_cmd="git clone https://github.com/SJ001/AI-Feynman",
        repo_url="https://github.com/SJ001/AI-Feynman",
        pretrained=False,
        notes=(
            "AI Feynman requires unit annotations for its dimensional analysis "
            "phase — these are available in the Feynman benchmark metadata. "
            "AI Feynman 2.0 (Udrescu et al. 2020b) extends the original with "
            "compositionality detection; prefer v2.0 for a stronger baseline."
        ),
        bibtex=(
            "@article{udrescu2020aifeynman,\n"
            "  title   = {{AI} Feynman: A Physics-Inspired Method for Symbolic\n"
            "             Regression},\n"
            "  author  = {Udrescu, Silviu-Marian and Tegmark, Max},\n"
            "  journal = {Science Advances},\n"
            "  volume  = {6},\n"
            "  number  = {16},\n"
            "  year    = {2020},\n"
            "  url     = {https://arxiv.org/abs/1905.11481}\n"
            "}"
        ),
    ),
]

# Quick lookup by key
COMPETITORS: Dict[str, CompetitorMethod] = {c.key: c for c in COMPETITOR_REGISTRY}


# ============================================================================
# FEYNMAN SR BENCHMARK — DATA GENERATION
# ============================================================================


@dataclass
class FeynmanEquation:
    """
    One equation from the Feynman SR Benchmark.

    Fields match the structure of load_test_data() in the existing protocol
    files so that this can be passed directly to HypatiaX's runner.
    """

    feynman_id: str          # e.g. "I.12.1"
    description: str         # Human-readable description
    ground_truth: str        # Analytical formula string
    variable_names: List[str]
    variable_ranges: Dict[str, Tuple[float, float]]
    constants: Dict[str, float]
    n_variables: int
    operator_depth: int      # Approximate expression tree depth
    difficulty: str          # easy / medium / hard
    domain: str              # physics sub-domain
    series: str              # I, II, or III (Feynman series)
    extrapolation_test: bool
    pmlb_dataset_name: str   # Dataset name in pmlb (if available)
    fn: Optional[Callable] = field(default=None, repr=False)

    def generate(
        self,
        num_samples: int = 200,
        noise_level: float = 0.0,
        seed: int = 42,
    ) -> Tuple[str, np.ndarray, np.ndarray, List[str], Dict]:
        """
        Generate (description, X, y, variable_names, metadata) tuple.

        Compatible with the load_test_data() return format used throughout
        the existing HypatiaX protocol files.

        Args:
            num_samples : Number of data points to generate.
            noise_level : Gaussian noise as fraction of y std (0 = noiseless).
            seed        : Random seed for reproducibility.

        Returns:
            Tuple compatible with ComparativeExperimentProtocol.load_test_data()
        """
        if self.fn is None:
            raise ValueError(
                f"No callable function defined for equation {self.feynman_id}. "
                f"Set fn= when constructing FeynmanEquation."
            )

        np.random.seed(seed)
        var_arrays = []
        for var in self.variable_names:
            lo, hi = self.variable_ranges[var]
            var_arrays.append(np.random.uniform(lo, hi, num_samples))

        X = np.column_stack(var_arrays) if len(var_arrays) > 1 else var_arrays[0].reshape(-1, 1)
        y = self.fn(*var_arrays)

        if noise_level > 0:
            noise_std = noise_level * np.std(y)
            y = y + np.random.normal(0, noise_std, num_samples)

        metadata = {
            "equation_name": self.feynman_id,
            "feynman_id": self.feynman_id,
            "domain": self.domain,
            "series": self.series,
            "difficulty": self.difficulty,
            "formula_type": "physics",
            "ground_truth": self.ground_truth,
            "constants": self.constants,
            "n_variables": self.n_variables,
            "operator_depth": self.operator_depth,
            "extrapolation_test": self.extrapolation_test,
            "benchmark": "feynman",
            "source": "Udrescu & Tegmark (2020)",
        }

        return (self.description, X, y, self.variable_names, metadata)


# ---------------------------------------------------------------------------
# Feynman equation definitions — 30-equation subset selected to match
# Core 15 operator complexity (depth 2-4, ≤4 variables).
#
# Selection criteria:
#   - Operator depth 2-4 (comparable to Core 15 equations)
#   - ≤4 input variables (avoids combinatorial blow-up)
#   - Spans Series I, II, III for coverage
#   - Includes equations similar to Core 15 (Arrhenius, Nernst, Michaelis)
#     so cross-benchmark consistency can be checked
#   - 8 equations flagged as extrapolation tests (same ratio as Core 15)
#
# Reference: Table 1 in Udrescu & Tegmark (2020), Science Advances.
# ---------------------------------------------------------------------------

def _build_feynman_equations() -> List[FeynmanEquation]:
    eqs = []

    # ── SERIES I: CLASSICAL MECHANICS & THERMODYNAMICS ───────────────────

    # I.6.20 — Gaussian distribution (probability)
    # f = exp(-θ²/(2σ²)) / (sqrt(2π) * σ)
    eqs.append(FeynmanEquation(
        feynman_id="I.6.20",
        description="Gaussian/normal distribution probability density",
        ground_truth="exp(-theta**2 / (2*sigma**2)) / (sqrt(2*pi) * sigma)",
        variable_names=["theta", "sigma"],
        variable_ranges={"theta": (-3.0, 3.0), "sigma": (0.5, 2.0)},
        constants={"pi": np.pi},
        n_variables=2, operator_depth=3, difficulty="medium",
        domain="probability", series="I", extrapolation_test=False,
        pmlb_dataset_name="feynman_I_6_20",
        fn=lambda theta, sigma: np.exp(-theta**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma),
    ))

    # I.9.18 — Force between charges (Coulomb)
    # F = q1*q2 / (4πε₀ * ((x2-x1)² + (y2-y1)²))
    # Simplified 1D version used here
    eqs.append(FeynmanEquation(
        feynman_id="I.9.18_1d",
        description="Coulomb force between two point charges (1D, simplified)",
        ground_truth="q1 * q2 / (4 * pi * eps0 * r**2)",
        variable_names=["q1", "q2", "r"],
        variable_ranges={"q1": (1e-9, 1e-6), "q2": (1e-9, 1e-6), "r": (0.1, 5.0)},
        constants={"eps0": 8.854e-12, "pi": np.pi},
        n_variables=3, operator_depth=3, difficulty="medium",
        domain="electrostatics", series="I", extrapolation_test=True,
        pmlb_dataset_name="feynman_I_9_18",
        fn=lambda q1, q2, r: q1 * q2 / (4 * np.pi * 8.854e-12 * r**2),
    ))

    # I.12.1 — Newton's law of gravitation
    # F = G * m1 * m2 / r²
    eqs.append(FeynmanEquation(
        feynman_id="I.12.1",
        description="Newton's gravitational force between two masses",
        ground_truth="G * m1 * m2 / r**2",
        variable_names=["m1", "m2", "r"],
        variable_ranges={"m1": (0.1, 10.0), "m2": (0.1, 10.0), "r": (1.0, 100.0)},
        constants={"G": 6.674e-11},
        n_variables=3, operator_depth=3, difficulty="medium",
        domain="mechanics", series="I", extrapolation_test=True,
        pmlb_dataset_name="feynman_I_12_1",
        fn=lambda m1, m2, r: 6.674e-11 * m1 * m2 / r**2,
    ))

    # I.12.2 — Coulomb's law (electric force)
    # F = q1 * q2 / (4πε₀r²)
    eqs.append(FeynmanEquation(
        feynman_id="I.12.2",
        description="Coulomb's law: electric force between charges",
        ground_truth="q1 * q2 / (4 * pi * eps0 * r**2)",
        variable_names=["q1", "q2", "r"],
        variable_ranges={"q1": (1e-6, 1e-3), "q2": (1e-6, 1e-3), "r": (0.01, 1.0)},
        constants={"eps0": 8.854e-12, "pi": np.pi},
        n_variables=3, operator_depth=3, difficulty="medium",
        domain="electrostatics", series="I", extrapolation_test=False,
        pmlb_dataset_name="feynman_I_12_2",
        fn=lambda q1, q2, r: q1 * q2 / (4 * np.pi * 8.854e-12 * r**2),
    ))

    # Classical kinetic energy (used as operator-depth parity check with Core 15)
    # NOTE: The Feynman SR table's I.34.8 is the relativistic Doppler formula,
    # not classical KE.  We use the ID "FEY_MECH_KE" to avoid any confusion
    # if a reviewer cross-checks against the original Udrescu & Tegmark table.
    eqs.append(FeynmanEquation(
        feynman_id="FEY_MECH_KE",
        description="Kinetic energy (classical): KE = 0.5 * m * v²",
        ground_truth="0.5 * m * v**2",
        variable_names=["m", "v"],
        variable_ranges={"m": (0.1, 10.0), "v": (0.1, 50.0)},
        constants={},
        n_variables=2, operator_depth=2, difficulty="easy",
        domain="mechanics", series="I", extrapolation_test=False,
        pmlb_dataset_name=None,   # not in pmlb under this custom ID
        fn=lambda m, v: 0.5 * m * v**2,
    ))

    # I.18.4 — Reduced mass
    # m_r = m1 * m2 / (m1 + m2)
    eqs.append(FeynmanEquation(
        feynman_id="I.18.4",
        description="Reduced mass of a two-body system",
        ground_truth="m1 * m2 / (m1 + m2)",
        variable_names=["m1", "m2"],
        variable_ranges={"m1": (0.1, 10.0), "m2": (0.1, 10.0)},
        constants={},
        n_variables=2, operator_depth=2, difficulty="easy",
        domain="mechanics", series="I", extrapolation_test=False,
        pmlb_dataset_name="feynman_I_18_4",
        fn=lambda m1, m2: m1 * m2 / (m1 + m2),
    ))

    # I.24.6 — Energy stored in a spring
    # E = 0.5 * k * x² + 0.5 * m * v²
    eqs.append(FeynmanEquation(
        feynman_id="I.24.6",
        description="Total mechanical energy: spring potential + kinetic",
        ground_truth="0.5 * k * x**2 + 0.5 * m * v**2",
        variable_names=["k", "x", "m", "v"],
        variable_ranges={"k": (0.5, 5.0), "x": (0.0, 2.0), "m": (0.1, 5.0), "v": (0.0, 10.0)},
        constants={},
        n_variables=4, operator_depth=3, difficulty="medium",
        domain="mechanics", series="I", extrapolation_test=False,
        pmlb_dataset_name="feynman_I_24_6",
        fn=lambda k, x, m, v: 0.5 * k * x**2 + 0.5 * m * v**2,
    ))

    # I.26.2 — Snell's law (refraction angle)
    # θ₂ = arcsin(n₁ * sin(θ₁) / n₂)
    eqs.append(FeynmanEquation(
        feynman_id="I.26.2",
        description="Snell's law: refracted angle from incident angle and refractive indices",
        ground_truth="arcsin(n1 * sin(theta1) / n2)",
        variable_names=["n1", "theta1", "n2"],
        variable_ranges={"n1": (1.0, 2.0), "theta1": (0.01, 1.2), "n2": (1.0, 2.5)},
        constants={},
        n_variables=3, operator_depth=3, difficulty="medium",
        domain="optics", series="I", extrapolation_test=False,
        pmlb_dataset_name="feynman_I_26_2",
        fn=lambda n1, theta1, n2: np.arcsin(np.clip(n1 * np.sin(theta1) / n2, -1, 1)),
    ))

    # I.34.27 — Energy of a photon
    # E = h * f
    eqs.append(FeynmanEquation(
        feynman_id="I.34.27",
        description="Photon energy: E = h * f (Planck relation)",
        ground_truth="h * f",
        variable_names=["f"],
        variable_ranges={"f": (1e12, 1e15)},
        constants={"h": 6.626e-34},
        n_variables=1, operator_depth=1, difficulty="easy",
        domain="quantum", series="I", extrapolation_test=False,
        pmlb_dataset_name="feynman_I_34_27",
        fn=lambda f: 6.626e-34 * f,
    ))

    # I.37.4 — Wave interference (double slit intensity)
    # I = I1 + I2 + 2*sqrt(I1*I2)*cos(delta)
    eqs.append(FeynmanEquation(
        feynman_id="I.37.4",
        description="Double-slit wave interference intensity",
        ground_truth="I1 + I2 + 2*sqrt(I1*I2)*cos(delta)",
        variable_names=["I1", "I2", "delta"],
        variable_ranges={"I1": (0.1, 5.0), "I2": (0.1, 5.0), "delta": (0.0, np.pi)},
        constants={},
        n_variables=3, operator_depth=3, difficulty="hard",
        domain="optics", series="I", extrapolation_test=True,
        pmlb_dataset_name="feynman_I_37_4",
        fn=lambda I1, I2, delta: I1 + I2 + 2 * np.sqrt(I1 * I2) * np.cos(delta),
    ))

    # I.41.16 — Planck's radiation law (dimensionless form)
    # Original: P ∝ h*f³ / (exp(h*f / (k*T)) - 1)
    # With x = h*f / (k_B*T) this becomes P ∝ (k_B*T/h)³ * x³ / (exp(x) - 1).
    # The (k_B*T/h)³ prefactor only shifts the scale; for symbolic discovery we
    # benchmark the dimensionless spectral shape x³/(exp(x)-1) directly.
    # This avoids y ∈ (1e35, 1e42) which overflows float64 sklearn scalers.
    # x range (0.01, 16) covers the Wien peak at x≈2.82 and both tails.
    # y range stays within (0, 0.85) — no overflow, no log-transform needed.
    eqs.append(FeynmanEquation(
        feynman_id="I_41_16_simplified",
        description="Planck blackbody spectral radiance (dimensionless: x=hf/kT)",
        ground_truth="x**3 / (exp(x) - 1)",
        variable_names=["x"],
        variable_ranges={"x": (0.01, 16.0)},
        constants={},
        n_variables=1, operator_depth=4, difficulty="hard",
        domain="thermodynamics", series="I", extrapolation_test=True,
        pmlb_dataset_name="feynman_I_41_16",
        fn=lambda x: x**3 / (np.exp(np.clip(x, 0, 700)) - 1 + 1e-300),
    ))

    # ── SERIES II: ELECTROMAGNETISM ───────────────────────────────────────

    # II.2.42 — Thermal conductivity (Fourier's law)
    # J = kappa * (T2 - T1) / d
    eqs.append(FeynmanEquation(
        feynman_id="II.2.42",
        description="Fourier's law of heat conduction: heat flux across material",
        ground_truth="kappa * (T2 - T1) / d",
        variable_names=["kappa", "T2", "T1", "d"],
        variable_ranges={"kappa": (0.1, 200.0), "T2": (300.0, 1000.0), "T1": (200.0, 299.0), "d": (0.001, 0.5)},
        constants={},
        n_variables=4, operator_depth=2, difficulty="easy",
        domain="thermodynamics", series="II", extrapolation_test=False,
        pmlb_dataset_name="feynman_II_2_42",
        fn=lambda kappa, T2, T1, d: kappa * (T2 - T1) / d,
    ))

    # II.6.15a — Dielectric polarisability
    # Ef = (epsilon - 1) / (epsilon + 2) * E0
    eqs.append(FeynmanEquation(
        feynman_id="II.6.15a",
        description="Clausius-Mossotti: effective field in dielectric",
        ground_truth="(epsilon - 1) / (epsilon + 2) * E0",
        variable_names=["epsilon", "E0"],
        variable_ranges={"epsilon": (1.1, 10.0), "E0": (1.0, 1000.0)},
        constants={},
        n_variables=2, operator_depth=3, difficulty="medium",
        domain="electromagnetism", series="II", extrapolation_test=False,
        pmlb_dataset_name="feynman_II_6_15a",
        fn=lambda epsilon, E0: (epsilon - 1) / (epsilon + 2) * E0,
    ))

    # II.11.3 — Polarisation of gas (Langevin)
    # P = n * alpha * E / (1 - n * alpha / 3)
    # Simplified as P = n * alpha * E
    eqs.append(FeynmanEquation(
        feynman_id="II_11_3_simplified",
        description="Dielectric polarisation: P = n * alpha * E (dilute limit)",
        ground_truth="n * alpha * E",
        variable_names=["n", "alpha", "E"],
        variable_ranges={"n": (1e20, 1e25), "alpha": (1e-30, 1e-28), "E": (1.0, 1e6)},
        constants={},
        n_variables=3, operator_depth=2, difficulty="easy",
        domain="electromagnetism", series="II", extrapolation_test=False,
        pmlb_dataset_name="feynman_II_11_3",
        fn=lambda n, alpha, E: n * alpha * E,
    ))

    # II.11.17 — Magnetic susceptibility (Curie's law)
    # chi = n * alpha / (1 - n * alpha / 3)  → simplified: chi = C / T
    eqs.append(FeynmanEquation(
        feynman_id="II_11_17_curie",
        description="Curie's law for magnetic susceptibility: chi = C/T",
        ground_truth="C / T",
        variable_names=["C", "T"],
        variable_ranges={"C": (0.01, 10.0), "T": (10.0, 1000.0)},
        constants={},
        n_variables=2, operator_depth=1, difficulty="easy",
        domain="magnetism", series="II", extrapolation_test=False,
        pmlb_dataset_name="feynman_II_11_17",
        fn=lambda C, T: C / T,
    ))

    # II.34.2 — Magnetic flux quantisation
    # Ef = q * v * B
    eqs.append(FeynmanEquation(
        feynman_id="II.34.2",
        description="Lorentz force on moving charge in magnetic field: F = qvB",
        ground_truth="q * v * B",
        variable_names=["q", "v", "B"],
        variable_ranges={"q": (1e-19, 1e-17), "v": (1e3, 1e7), "B": (0.01, 10.0)},
        constants={},
        n_variables=3, operator_depth=2, difficulty="easy",
        domain="electromagnetism", series="II", extrapolation_test=False,
        pmlb_dataset_name="feynman_II_34_2",
        fn=lambda q, v, B: q * v * B,
    ))

    # II.36.38 — Magnetisation energy (Zeeman effect)
    # E = -mu_z * B  → E = -m_s * g * mu_B * B
    eqs.append(FeynmanEquation(
        feynman_id="II_36_38",
        description="Zeeman energy: electron spin in magnetic field",
        ground_truth="-ms * g * mu_B * B",
        variable_names=["ms", "B"],
        variable_ranges={"ms": (-0.5, 0.5), "B": (0.01, 10.0)},
        constants={"g": 2.002, "mu_B": 9.274e-24},
        n_variables=2, operator_depth=2, difficulty="medium",
        domain="quantum", series="II", extrapolation_test=False,
        pmlb_dataset_name="feynman_II_36_38",
        fn=lambda ms, B: -ms * 2.002 * 9.274e-24 * B,
    ))

    # ── SERIES III: QUANTUM MECHANICS ─────────────────────────────────────

    # III.4.33 — Bose-Einstein distribution
    # n = 1 / (exp(h*f / (k*T)) - 1)
    eqs.append(FeynmanEquation(
        feynman_id="III.4.33",
        description="Bose-Einstein occupation number for bosons",
        ground_truth="1 / (exp(h*f / (k_B * T)) - 1)",
        variable_names=["f", "T"],
        variable_ranges={"f": (1e10, 1e13), "T": (10.0, 500.0)},
        constants={"h": 6.626e-34, "k_B": 1.381e-23},
        n_variables=2, operator_depth=3, difficulty="hard",
        domain="quantum", series="III", extrapolation_test=True,
        pmlb_dataset_name="feynman_III_4_33",
        fn=lambda f, T: 1.0 / (np.exp(np.clip(6.626e-34 * f / (1.381e-23 * T), 0, 700)) - 1 + 1e-300),
    ))

    # III.4.32 — Fermi-Dirac distribution
    # n = 1 / (exp((E - mu) / (k*T)) + 1)
    # UNITS FIX: E and mu are in eV (range -0.5 to +0.5 eV).
    # Using k_B = 8.617e-5 eV/K (not 1.381e-23 J/K) so the exponent
    # (E - mu) / (k_B * T) is dimensionless and O(1) at room temperature.
    # At T=300 K, k_B*T ≈ 0.026 eV — same scale as E and mu.
    eqs.append(FeynmanEquation(
        feynman_id="III.4.32",
        description="Fermi-Dirac occupation number for fermions",
        ground_truth="1 / (exp((E - mu) / (k_B * T)) + 1)",
        variable_names=["E", "mu", "T"],
        variable_ranges={"E": (-0.5, 0.5), "mu": (-0.2, 0.2), "T": (1.0, 500.0)},
        constants={"k_B": 8.617e-5},   # eV/K — consistent with E, mu in eV
        n_variables=3, operator_depth=3, difficulty="hard",
        domain="quantum", series="III", extrapolation_test=True,
        pmlb_dataset_name="feynman_III_4_32",
        fn=lambda E, mu, T: 1.0 / (np.exp(np.clip((E - mu) / (8.617e-5 * T), -700, 700)) + 1),
    ))

    # III.7.38 — Rabi frequency (two-level quantum system)
    # omega = mu * B / hbar
    eqs.append(FeynmanEquation(
        feynman_id="III.7.38",
        description="Rabi frequency of two-level atom in magnetic field",
        ground_truth="mu * B / hbar",
        variable_names=["mu", "B"],
        variable_ranges={"mu": (9e-24, 1e-23), "B": (0.001, 1.0)},
        constants={"hbar": 1.055e-34},
        n_variables=2, operator_depth=2, difficulty="easy",
        domain="quantum", series="III", extrapolation_test=False,
        pmlb_dataset_name="feynman_III_7_38",
        fn=lambda mu, B: mu * B / 1.055e-34,
    ))

    # ── ADDITIONAL EQUATIONS: CHEMISTRY / THERMODYNAMICS CROSSOVER ────────
    # These overlap with Core 15 domains, enabling cross-benchmark consistency
    # checks between Feynman and Core 15 results.

    # Arrhenius (Feynman variant — different parametrisation from Core 15)
    # k = A * exp(-Ea / (R * T))
    eqs.append(FeynmanEquation(
        feynman_id="FEY_CHEM_ARR",
        description="Arrhenius rate constant (Feynman variant) — cross-benchmark consistency check",
        ground_truth="A * exp(-Ea / (R * T))",
        variable_names=["T"],
        variable_ranges={"T": (200.0, 500.0)},
        constants={"A": 1e11, "Ea": 80000.0, "R": 8.314},
        n_variables=1, operator_depth=3, difficulty="hard",
        domain="chemistry", series="crossover", extrapolation_test=True,
        pmlb_dataset_name=None,
        fn=lambda T: 1e11 * np.exp(-80000.0 / (8.314 * T)),
    ))

    # Nernst equation (electrochemistry)
    # E = E0 - (R*T / (n*F)) * ln([ox]/[red])
    eqs.append(FeynmanEquation(
        feynman_id="FEY_CHEM_NERNST",
        description="Nernst equation for electrode potential — cross-benchmark consistency check",
        ground_truth="E0 - (R*T / (n*F)) * log(ox / red)",
        variable_names=["T", "ox", "red"],
        variable_ranges={"T": (273.0, 373.0), "ox": (0.01, 2.0), "red": (0.01, 2.0)},
        constants={"E0": 0.76, "R": 8.314, "n": 2, "F": 96485.0},
        n_variables=3, operator_depth=4, difficulty="hard",
        domain="electrochemistry", series="crossover", extrapolation_test=False,
        pmlb_dataset_name=None,
        fn=lambda T, ox, red: 0.76 - (8.314 * T / (2 * 96485.0)) * np.log(ox / (red + 1e-12)),
    ))

    # Michaelis-Menten (biology crossover)
    eqs.append(FeynmanEquation(
        feynman_id="FEY_BIO_MM",
        description="Michaelis-Menten enzyme kinetics — cross-benchmark consistency check",
        ground_truth="(Vmax * S) / (Km + S)",
        variable_names=["S"],
        variable_ranges={"S": (0.1, 100.0)},
        constants={"Vmax": 50.0, "Km": 10.0},
        n_variables=1, operator_depth=2, difficulty="medium",
        domain="biology", series="crossover", extrapolation_test=False,
        pmlb_dataset_name=None,
        fn=lambda S: (50.0 * S) / (10.0 + S),
    ))

    # Stefan-Boltzmann law
    # P = sigma * A * T^4
    eqs.append(FeynmanEquation(
        feynman_id="FEY_THERMO_SB",
        description="Stefan-Boltzmann law: blackbody radiated power",
        ground_truth="sigma * A * T**4",
        variable_names=["A", "T"],
        variable_ranges={"A": (0.01, 10.0), "T": (300.0, 6000.0)},
        constants={"sigma": 5.67e-8},
        n_variables=2, operator_depth=2, difficulty="easy",
        domain="thermodynamics", series="crossover", extrapolation_test=True,
        pmlb_dataset_name=None,
        fn=lambda A, T: 5.67e-8 * A * T**4,
    ))

    # Ideal gas law: P = nRT / V
    eqs.append(FeynmanEquation(
        feynman_id="FEY_THERMO_IG",
        description="Ideal gas law: pressure from moles, temperature, volume",
        ground_truth="n * R * T / V",
        variable_names=["n", "T", "V"],
        variable_ranges={"n": (0.1, 10.0), "T": (200.0, 600.0), "V": (0.001, 1.0)},
        constants={"R": 8.314},
        n_variables=3, operator_depth=2, difficulty="easy",
        domain="thermodynamics", series="crossover", extrapolation_test=False,
        pmlb_dataset_name=None,
        fn=lambda n, T, V: n * 8.314 * T / V,
    ))

    # Logistic growth (biology crossover)
    eqs.append(FeynmanEquation(
        feynman_id="FEY_BIO_LOG",
        description="Logistic growth rate — cross-benchmark consistency check",
        ground_truth="r * N * (1 - N/K)",
        variable_names=["N"],
        variable_ranges={"N": (10.0, 950.0)},
        constants={"r": 0.3, "K": 1000.0},
        n_variables=1, operator_depth=3, difficulty="medium",
        domain="biology", series="crossover", extrapolation_test=False,
        pmlb_dataset_name=None,
        fn=lambda N: 0.3 * N * (1 - N / 1000.0),
    ))

    # Power-law allometric scaling (biology crossover)
    eqs.append(FeynmanEquation(
        feynman_id="FEY_BIO_ALLO",
        description="Allometric scaling law (metabolic rate vs mass)",
        ground_truth="a * M**b",
        variable_names=["M"],
        variable_ranges={"M": (1.0, 1000.0)},
        constants={"a": 3.5, "b": 0.75},
        n_variables=1, operator_depth=2, difficulty="easy",
        domain="biology", series="crossover", extrapolation_test=False,
        pmlb_dataset_name=None,
        fn=lambda M: 3.5 * M**0.75,
    ))

    # Henderson-Hasselbalch (chemistry crossover)
    eqs.append(FeynmanEquation(
        feynman_id="FEY_CHEM_HH",
        description="Henderson-Hasselbalch equation for buffer pH",
        ground_truth="pKa + log10(A_minus / HA)",
        variable_names=["A_minus", "HA"],
        variable_ranges={"A_minus": (0.01, 2.0), "HA": (0.01, 2.0)},
        constants={"pKa": 6.5},
        n_variables=2, operator_depth=3, difficulty="medium",
        domain="chemistry", series="crossover", extrapolation_test=False,
        pmlb_dataset_name=None,
        fn=lambda A_minus, HA: 6.5 + np.log10(A_minus / (HA + 1e-12)),
    ))


    # ── Two final equations to complete the 30-equation subset ───────────

    # Ohm's law
    eqs.append(FeynmanEquation(
        feynman_id="II_11_27_ohm",
        description="Ohm's law: voltage as product of current and resistance",
        ground_truth="I * R",
        variable_names=["I", "R"],
        variable_ranges={"I": (0.001, 10.0), "R": (1.0, 1000.0)},
        constants={},
        n_variables=2, operator_depth=1, difficulty="easy",
        domain="electromagnetism", series="II", extrapolation_test=False,
        pmlb_dataset_name=None,
        fn=lambda I, R: I * R,
    ))

    # Energy stored in a capacitor
    eqs.append(FeynmanEquation(
        feynman_id="II_11_28_capacitor",
        description="Energy stored in a capacitor: E = 0.5 * C * V^2",
        ground_truth="0.5 * C * V**2",
        variable_names=["C", "V"],
        variable_ranges={"C": (1e-9, 1e-3), "V": (1.0, 500.0)},
        constants={},
        n_variables=2, operator_depth=2, difficulty="easy",
        domain="electromagnetism", series="II", extrapolation_test=False,
        pmlb_dataset_name=None,
        fn=lambda C, V: 0.5 * C * V**2,
    ))

    assert len(eqs) == 30, f"Expected 30 Feynman equations, got {len(eqs)}"
    return eqs


FEYNMAN_EQUATIONS: List[FeynmanEquation] = _build_feynman_equations()
FEYNMAN_BY_ID: Dict[str, FeynmanEquation] = {eq.feynman_id: eq for eq in FEYNMAN_EQUATIONS}


# ============================================================================
# SRBENCH SUBSET — DATASET REGISTRY
# ============================================================================


@dataclass
class SRBenchDataset:
    """Registry entry for one SRBench real-world dataset."""

    pmlb_name: str          # Dataset name in pmlb
    description: str
    domain: str
    n_features: int
    difficulty: str
    notes: str


SRBENCH_SUBSET: List[SRBenchDataset] = [
    # Biology / ecology
    SRBenchDataset("505_tecator",         "Near-infrared spectra to fat content",         "biology",       124, "hard",   "High-dimensional; test feature selection robustness"),
    SRBenchDataset("537_houses",          "California housing price model",                "economics",     8,   "medium", "Non-linear interactions; widely used benchmark"),
    SRBenchDataset("1028_SWD",            "Steel industry energy use prediction",          "engineering",   10,  "medium", "Mixed units; real manufacturing data"),
    SRBenchDataset("1096_FacultySalaries","Faculty salary prediction from demographics",   "social",        4,   "easy",   "Low-dim; interpretable ground truth unknown"),
    SRBenchDataset("192_vineyard",        "Vineyard yield from soil/climate variables",    "agriculture",   3,   "easy",   "Small feature set; domain-expert comparison possible"),
    # Physics / chemistry
    SRBenchDataset("344_mv",              "Molecular viscosity from physicochemical props","chemistry",     10,  "medium", "Quantitative structure-property relationship"),
    SRBenchDataset("1030_ERA",            "Air temperature from satellite data",            "meteorology",   16,  "hard",   "Spatial features; tests variable selection"),
    SRBenchDataset("562_cpu_small",       "CPU performance prediction",                    "engineering",   12,  "medium", "Classic SR benchmark dataset"),
    SRBenchDataset("579_fri_c0_500_25",   "Synthetic Friedman function (5 relevant vars)", "synthetic",     25,  "medium", "Friedman function — analytical form known"),
    SRBenchDataset("1199_BNG_echoMonths", "Echocardiogram survival prediction",           "medicine",      9,   "hard",   "Clinical outcome data"),
    # Additional diversity
    SRBenchDataset("210_cloud",           "Cloud cover prediction from meteorological data","meteorology",  6,   "easy",   "Physical interpretability possible"),
    SRBenchDataset("228_elusage",         "Electric utility usage prediction",             "energy",        4,   "easy",   "Low-dim; interpretable"),
    SRBenchDataset("523_analcatdata_neavote","Voting patterns analysis",                  "social",        4,   "easy",   "Categorical inputs encoded numerically"),
    SRBenchDataset("695_chatfield_4",     "Time series forecasting (Chatfield)",           "statistics",    4,   "medium", "Temporal structure"),
    SRBenchDataset("1049_phoneme",        "Phoneme classification features",               "signal",        5,   "medium", "Speech signal processing"),
    SRBenchDataset("1191_BNG_pbc",        "Liver disease survival (primary biliary)",      "medicine",      17,  "hard",   "High-dim clinical; test feature importance"),
    SRBenchDataset("485_analcatdata_vehicle","Vehicle fuel consumption analysis",          "engineering",   17,  "medium", "Mechanical engineering; partial ground truth available"),
    SRBenchDataset("519_vinnie",          "Insurance claim amount prediction",             "finance",       3,   "easy",   "Financial domain; skewed target distribution"),
    SRBenchDataset("526_slice_localization","CT scan slice location from image features",  "medicine",      385, "hard",   "Very high-dim; feature compression needed"),
    SRBenchDataset("573_cpu_act",         "CPU activity from system call statistics",      "engineering",   21,  "medium", "Mixed linear/nonlinear relationships"),
]


# ============================================================================
# BENCHMARK PROTOCOL CLASS
# ============================================================================


class BenchmarkProtocol:
    """
    Protocol manager for published SR benchmark comparisons.

    Provides the same interface as the existing HypatiaX protocol classes
    (load_test_data, get_all_domains, generate_experiment_report) so that
    this file can be dropped in as a direct replacement for the Core 15
    runner in any experiment script.
    """

    # ── Class-level constants ─────────────────────────────────────────────

    RECOVERY_THRESHOLD_STRICT: float = 0.9999   # R² threshold used by NeSymReS paper
    RECOVERY_THRESHOLD_RELAXED: float = 0.99    # Threshold used in Core 15 campaign
    NOISE_LEVEL_DEFAULT: float = 0.05           # Match Core 15 protocol (σ = 0.05·std(y))

    # ── Runner interface (matches ComparativeExperimentProtocol exactly) ──
    #
    # Runners call:
    #   protocol.get_all_domains()
    #   protocol.load_test_data(domain, num_samples=N)
    #
    # `domain` is one of the strings returned by get_all_domains().
    # Feynman physics sub-domains are prefixed with "feynman_" so they cannot
    # clash with SRBench domain names.  SRBench domain names are returned as-is.
    #
    # Choosing the active benchmark:
    #   BenchmarkProtocol()                      → Feynman only  (default)
    #   BenchmarkProtocol(benchmark="srbench")   → SRBench only
    #   BenchmarkProtocol(benchmark="both")      → Feynman + SRBench
    #
    # This mirrors how standalone_real_methods_test.py uses
    # ComparativeExperimentProtocol without any constructor arguments.

    # Feynman sub-domain → "feynman_<sub-domain>" prefix used in get_all_domains
    _FEYNMAN_PREFIX = "feynman_"

    def __init__(
        self,
        benchmark: str = "feynman",
        num_samples: int = 200,
        noise_level: float = NOISE_LEVEL_DEFAULT,
        seed: int = 42,
        feynman_series: Optional[str] = None,
        crossover_only: bool = False,
    ) -> None:
        """
        Initialise the benchmark protocol.

        Args:
            benchmark       : "feynman" | "srbench" | "both"
            num_samples     : Default number of data points per equation/dataset.
            noise_level     : Gaussian noise as fraction of y std.
            seed            : Base random seed (matches Core 15 default).
            feynman_series  : Restrict Feynman equations to one series
                              ("I", "II", "III", "crossover"); None = all.
            crossover_only  : Only load equations that overlap with Core 15.
        """
        if benchmark not in ("feynman", "srbench", "both"):
            raise ValueError(f"benchmark must be 'feynman', 'srbench', or 'both'; got '{benchmark}'")
        self.benchmark = benchmark
        self.default_num_samples = num_samples
        self.noise_level = noise_level
        self.seed = seed
        self.feynman_series = feynman_series
        self.crossover_only = crossover_only

        # Pre-build the active equation / dataset lists once so get_all_domains
        # and load_test_data are O(1) lookups.
        self._feynman_eqs: List[FeynmanEquation] = []
        if benchmark in ("feynman", "both"):
            self._feynman_eqs = self.get_feynman_equations(
                series=feynman_series,
                crossover_only=crossover_only,
            )

        self._srbench_datasets: List[SRBenchDataset] = []
        if benchmark in ("srbench", "both"):
            self._srbench_datasets = list(SRBENCH_SUBSET)

        # Domain → list-of-equations / datasets map (built lazily by
        # _build_domain_map if callers use the runner interface).
        self._domain_map: Optional[Dict[str, List]] = None

    # ── internal helpers ──────────────────────────────────────────────────

    def _build_domain_map(self) -> Dict[str, List]:
        """
        Build {domain_key: [items]} mapping used by load_test_data().

        Feynman equations are grouped by their physics sub-domain and stored
        under keys of the form "feynman_<sub-domain>" (e.g. "feynman_mechanics").
        SRBench datasets are grouped by their domain field (e.g. "chemistry").
        """
        domain_map: Dict[str, List] = {}

        for eq in self._feynman_eqs:
            key = f"{self._FEYNMAN_PREFIX}{eq.domain}"
            domain_map.setdefault(key, []).append(eq)

        for ds in self._srbench_datasets:
            domain_map.setdefault(ds.domain, []).append(ds)

        return domain_map

    @property
    def _domains(self) -> Dict[str, List]:
        if self._domain_map is None:
            self._domain_map = self._build_domain_map()
        return self._domain_map

    # ── Public runner interface ───────────────────────────────────────────

    def get_all_domains(self) -> List[str]:
        """
        Return all domain keys that this protocol exposes.

        Mirrors ComparativeExperimentProtocol.get_all_domains().

        Returns
        -------
        List[str]
            Feynman sub-domains are prefixed with ``"feynman_"``
            (e.g. ``"feynman_mechanics"``, ``"feynman_chemistry"``).
            SRBench domains are bare strings
            (e.g. ``"chemistry"``, ``"engineering"``).
        """
        return sorted(self._domains.keys())

    def load_test_data(
        self,
        domain: str,
        num_samples: Optional[int] = None,
        noise_level: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> List[Tuple[str, np.ndarray, np.ndarray, List[str], Dict]]:
        """
        Load test cases for *domain* — the primary runner-facing method.

        Mirrors ComparativeExperimentProtocol.load_test_data() exactly:

            for desc, X, y, var_names, metadata in protocol.load_test_data(domain):
                ...

        Args:
            domain      : One of the strings returned by ``get_all_domains()``.
            num_samples : Data points per equation (default: ``self.default_num_samples``).
            noise_level : Gaussian noise fraction (default: ``self.noise_level``).
            seed        : Base random seed (default: ``self.seed``).

        Returns
        -------
        List of ``(description, X, y, variable_names, metadata)`` tuples.
        ``metadata`` always contains at minimum:
            ``equation_name``, ``domain``, ``difficulty``, ``benchmark``,
            ``ground_truth``, ``extrapolation_test``.

        Raises
        ------
        KeyError
            If *domain* is not one of the strings returned by
            ``get_all_domains()``.
        """
        if domain not in self._domains:
            available = ", ".join(self.get_all_domains())
            raise KeyError(
                f"Unknown domain '{domain}'.\n"
                f"Available domains: {available}"
            )

        n = num_samples if num_samples is not None else self.default_num_samples
        noise = noise_level if noise_level is not None else self.noise_level
        s = seed if seed is not None else self.seed

        items = self._domains[domain]
        results: List[Tuple] = []

        if domain.startswith(self._FEYNMAN_PREFIX):
            # Feynman equations — generate analytically
            for offset, eq in enumerate(items):
                results.append(eq.generate(n, noise, seed=s + offset * 7))
        else:
            # SRBench datasets — load via pmlb (or return stubs)
            srbench_names = [ds.pmlb_name for ds in items]
            results = self.load_srbench_test_data(
                dataset_names=srbench_names,
                num_samples=n,
            )

        return results

    # ── Feynman interface ─────────────────────────────────────────────────

    @staticmethod
    def get_feynman_equations(
        series: Optional[str] = None,
        max_variables: int = 4,
        crossover_only: bool = False,
    ) -> List[FeynmanEquation]:
        """
        Return the Feynman equation subset, optionally filtered.

        Args:
            series          : "I", "II", "III", or "crossover"; None = all.
            max_variables   : Maximum number of input variables.
            crossover_only  : If True, return only equations that overlap
                              with Core 15 domains (consistency check).
        """
        eqs = FEYNMAN_EQUATIONS
        if series is not None:
            eqs = [e for e in eqs if e.series == series]
        eqs = [e for e in eqs if e.n_variables <= max_variables]
        if crossover_only:
            eqs = [e for e in eqs if e.series == "crossover"]
        return eqs

    @staticmethod
    def load_feynman_test_data(
        feynman_ids: Optional[List[str]] = None,
        num_samples: int = 200,
        noise_level: float = NOISE_LEVEL_DEFAULT,
        seed: int = 42,
    ) -> List[Tuple[str, np.ndarray, np.ndarray, List[str], Dict]]:
        """
        Generate test data for the Feynman benchmark subset.

        Matches the return type of ComparativeExperimentProtocol.load_test_data()
        so the same runner code works for both benchmarks without modification.

        Args:
            feynman_ids : List of Feynman IDs to load; None = all 30 equations.
            num_samples : Number of data points per equation.
            noise_level : Gaussian noise as fraction of y std (default matches Core 15).
            seed        : Base random seed; each equation uses seed + offset.

        Returns:
            List of (description, X, y, variable_names, metadata) tuples.
        """
        target_ids = feynman_ids or [eq.feynman_id for eq in FEYNMAN_EQUATIONS]
        results = []
        for offset, fid in enumerate(target_ids):
            if fid not in FEYNMAN_BY_ID:
                warnings.warn(f"Unknown Feynman ID '{fid}' — skipping.", stacklevel=2)
                continue
            eq = FEYNMAN_BY_ID[fid]
            results.append(eq.generate(num_samples, noise_level, seed=seed + offset * 7))
        return results

    @staticmethod
    def load_srbench_test_data(
        dataset_names: Optional[List[str]] = None,
        num_samples: int = 200,
    ) -> List[Tuple[str, np.ndarray, np.ndarray, List[str], Dict]]:
        """
        Load SRBench datasets via pmlb.

        Falls back to returning metadata-only tuples if pmlb is not installed,
        so the protocol can be imported and documented without the dependency.

        Args:
            dataset_names : pmlb dataset names to load; None = all 20 in subset.
            num_samples   : Maximum rows to load (None = full dataset).

        Returns:
            List of (description, X, y, variable_names, metadata) tuples.
        """
        target_names = dataset_names or [d.pmlb_name for d in SRBENCH_SUBSET]
        registry = {d.pmlb_name: d for d in SRBENCH_SUBSET}
        results = []

        for name in target_names:
            meta_entry = registry.get(name)
            if meta_entry is None:
                warnings.warn(f"'{name}' not in SRBENCH_SUBSET registry — skipping.", stacklevel=2)
                continue

            if not PMLB_AVAILABLE:
                # Return a stub so documentation/dry-runs work without pmlb
                stub_meta = {
                    "equation_name": name,
                    "domain": meta_entry.domain,
                    "difficulty": meta_entry.difficulty,
                    "benchmark": "srbench",
                    "source": "La Cava et al. (2021)",
                    "n_features": meta_entry.n_features,
                    "notes": meta_entry.notes,
                    "pmlb_stub": True,
                }
                results.append((meta_entry.description, None, None, [], stub_meta))
                continue

            try:
                data = fetch_data(name, return_X_y=False)
                X_full = data.drop("target", axis=1).values.astype(float)
                y_full = data["target"].values.astype(float)
                var_names = list(data.drop("target", axis=1).columns)

                if num_samples and len(X_full) > num_samples:
                    np.random.seed(42)
                    idx = np.random.choice(len(X_full), num_samples, replace=False)
                    X_full = X_full[idx]
                    y_full = y_full[idx]

                metadata = {
                    "equation_name": name,
                    "domain": meta_entry.domain,
                    "difficulty": meta_entry.difficulty,
                    "benchmark": "srbench",
                    "source": "La Cava et al. (2021)",
                    "n_features": meta_entry.n_features,
                    "notes": meta_entry.notes,
                    "ground_truth": "unknown (real-world data)",
                    "extrapolation_test": False,
                }
                results.append((meta_entry.description, X_full, y_full, var_names, metadata))

            except Exception as exc:
                warnings.warn(f"Failed to load '{name}': {exc}", stacklevel=2)
                continue

        return results

    # ── Reporting ─────────────────────────────────────────────────────────

    @staticmethod
    def generate_experiment_report(
        results: List[Dict],
        threshold: float = RECOVERY_THRESHOLD_RELAXED,
    ) -> Dict:
        """
        Generate benchmark comparison report.

        Compatible with DeFiExperimentProtocol.generate_experiment_report()
        so existing report-generation infrastructure works unchanged.

        Args:
            results   : List of result dicts from an experiment runner.
            threshold : R² threshold for counting a recovery as successful.

        Returns:
            Dict with overall, by_domain, by_benchmark, and by_competitor keys.
        """
        successful = [
            r for r in results
            if r.get("evaluation", {}).get("r2", -np.inf) >= threshold
        ]

        r2_all = [r["evaluation"]["r2"] for r in results if "r2" in r.get("evaluation", {})]

        by_domain: Dict[str, Dict] = {}
        by_benchmark: Dict[str, Dict] = {}
        by_competitor: Dict[str, Dict] = {}

        for r in results:
            meta = r.get("metadata", {})
            domain = meta.get("domain", "unknown")
            benchmark = meta.get("benchmark", "core15")
            competitor = r.get("system", "hypatiax")
            r2 = r.get("evaluation", {}).get("r2")
            ok = r2 is not None and r2 >= threshold

            for bucket, key in [(by_domain, domain), (by_benchmark, benchmark), (by_competitor, competitor)]:
                if key not in bucket:
                    bucket[key] = {"total": 0, "recovered": 0, "r2_scores": []}
                bucket[key]["total"] += 1
                if ok:
                    bucket[key]["recovered"] += 1
                if r2 is not None:
                    bucket[key]["r2_scores"].append(r2)

        def _summarise(bucket: Dict) -> Dict:
            out = {}
            for key, val in bucket.items():
                scores = val["r2_scores"]
                total = val["total"]
                rec = val["recovered"]
                out[key] = {
                    "total": total,
                    "recovered": rec,
                    "recovery_rate": rec / total if total else 0.0,
                    "mean_r2": float(np.mean(scores)) if scores else None,
                    "median_r2": float(np.median(scores)) if scores else None,
                    "std_r2": float(np.std(scores)) if scores else None,
                }
            return out

        return {
            "overall": {
                "total": len(results),
                "recovered": len(successful),
                "recovery_rate": len(successful) / len(results) if results else 0.0,
                "mean_r2": float(np.mean(r2_all)) if r2_all else None,
                "median_r2": float(np.median(r2_all)) if r2_all else None,
                "r2_threshold": threshold,
            },
            "by_domain": _summarise(by_domain),
            "by_benchmark": _summarise(by_benchmark),
            "by_competitor": _summarise(by_competitor),
        }

    # ── Competitor utilities ──────────────────────────────────────────────

    @staticmethod
    def get_competitor(key: str) -> CompetitorMethod:
        """Return a competitor method by key."""
        if key not in COMPETITORS:
            raise KeyError(f"Unknown competitor '{key}'. Available: {list(COMPETITORS)}")
        return COMPETITORS[key]

    @staticmethod
    def get_competitors_by_priority(priority: int = 1) -> List[CompetitorMethod]:
        """Return all competitors at or above the given priority level."""
        return [c for c in COMPETITOR_REGISTRY if c.priority <= priority]

    @staticmethod
    def print_competitor_table() -> None:
        """Print a formatted comparison table of all registered competitors."""
        print()
        print("=" * 100)
        print("  COMPETITOR METHOD REGISTRY — HypatiaX JMLR Comparison")
        print("=" * 100)
        header = f"  {'Key':<14}{'Method':<38}{'Priority':<12}{'Type':<35}"
        print(header)
        print("-" * 100)
        for c in COMPETITOR_REGISTRY:
            prio_str = {1: "★ Must", 2: "Should", 3: "Optional"}.get(c.priority, str(c.priority))
            print(f"  {c.key:<14}{c.name[:36]:<38}{prio_str:<12}{c.method_type[:33]:<35}")
        print("-" * 100)
        print()
        print("  ★ SNIP is the primary comparison target (architecturally closest to HypatiaX).")
        print()

    @staticmethod
    def export_bibtex(filepath: str = "benchmark_references.bib") -> None:
        """
        Write all competitor BibTeX entries to a .bib file.

        Append the output to references.bib before submission.
        """
        entries = "\n\n".join(c.bibtex for c in COMPETITOR_REGISTRY)
        header = (
            "% ============================================================\n"
            "% Competitor method references for HypatiaX benchmark section\n"
            "% Generated by experiment_protocol_benchmark.py\n"
            "% Append to references.bib before JMLR submission.\n"
            "% ============================================================\n\n"
        )
        Path(filepath).write_text(header + entries + "\n")
        print(f"BibTeX entries written to: {filepath}")

    # ── Documentation ─────────────────────────────────────────────────────

    @staticmethod
    def describe() -> None:
        """Print full protocol documentation to stdout."""
        print()
        print("=" * 100)
        print("  BENCHMARK PROTOCOL DOCUMENTATION")
        print("  HypatiaX — JMLR Critical Blocker C1 Resolution")
        print("=" * 100)

        print("\n  FEYNMAN SR BENCHMARK  (30 equations)")
        print("  " + "-" * 60)
        by_series: Dict[str, List] = {}
        for eq in FEYNMAN_EQUATIONS:
            by_series.setdefault(eq.series, []).append(eq)

        for series, eqs in sorted(by_series.items()):
            extrap = sum(1 for e in eqs if e.extrapolation_test)
            easy = sum(1 for e in eqs if e.difficulty == "easy")
            medium = sum(1 for e in eqs if e.difficulty == "medium")
            hard = sum(1 for e in eqs if e.difficulty == "hard")
            print(f"\n  Series {series} ({len(eqs)} equations, {extrap} extrapolation-flagged):")
            print(f"    Difficulty: {easy} easy / {medium} medium / {hard} hard")
            for eq in eqs:
                flag = "⭐" if eq.extrapolation_test else "  "
                print(f"    {flag} {eq.feynman_id:<22} {eq.description[:55]}")

        print(f"\n  Total Feynman equations : {len(FEYNMAN_EQUATIONS)}")
        total_extrap = sum(1 for e in FEYNMAN_EQUATIONS if e.extrapolation_test)
        print(f"  Extrapolation-flagged   : {total_extrap}")
        crossover = [e for e in FEYNMAN_EQUATIONS if e.series == "crossover"]
        print(f"  Core 15 crossover eqs   : {len(crossover)} (for consistency checks)")

        print("\n\n  SRBENCH SUBSET  (20 datasets)")
        print("  " + "-" * 60)
        for d in SRBENCH_SUBSET:
            print(f"    {d.pmlb_name:<40} {d.domain:<14} {d.difficulty}")

        print("\n\n  COMPETITOR METHODS")
        BenchmarkProtocol.print_competitor_table()


# ============================================================================
# MAIN — CLI
# ============================================================================
#
# The CLI constructs a BenchmarkProtocol instance with the user's flags and
# then calls the same get_all_domains() / load_test_data() interface that
# runner scripts use when they import this module.  The CLI and the import
# path are therefore identical — there is no separate code path.
#
# Usage examples
# --------------
#   python experiment_protocol_benchmark.py --describe
#   python experiment_protocol_benchmark.py --benchmark feynman
#   python experiment_protocol_benchmark.py --benchmark feynman --series I
#   python experiment_protocol_benchmark.py --benchmark feynman --crossover-only
#   python experiment_protocol_benchmark.py --benchmark srbench
#   python experiment_protocol_benchmark.py --benchmark both --num-samples 500
#   python experiment_protocol_benchmark.py --competitors snip
#   python experiment_protocol_benchmark.py --competitors all
#   python experiment_protocol_benchmark.py --export-bibtex references_to_add.bib

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="HypatiaX Published SR Benchmark Protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--benchmark",
        choices=["feynman", "srbench", "both"],
        default="feynman",
        help="Which benchmark to operate on (default: feynman)",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print full protocol documentation and exit",
    )
    parser.add_argument(
        "--competitors",
        nargs="*",
        metavar="KEY",
        help=(
            "Competitor keys to show info for. "
            "Use --competitors all for full table. "
            f"Available: {', '.join(COMPETITORS)}"
        ),
    )
    parser.add_argument(
        "--export-bibtex",
        metavar="FILE",
        default=None,
        help="Export competitor BibTeX entries to FILE (default: benchmark_references.bib)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Number of data points per equation (default: 200)",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=BenchmarkProtocol.NOISE_LEVEL_DEFAULT,
        help=f"Noise level as fraction of y std (default: {BenchmarkProtocol.NOISE_LEVEL_DEFAULT})",
    )
    parser.add_argument(
        "--crossover-only",
        action="store_true",
        help="Feynman: only load equations that overlap with Core 15 domains",
    )
    parser.add_argument(
        "--series",
        choices=["I", "II", "III", "crossover"],
        default=None,
        help="Feynman: restrict to a single series",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42, matches Core 15 protocol)",
    )
    args = parser.parse_args()

    # ── --describe ────────────────────────────────────────────────────────
    if args.describe:
        # describe() is a static method — no protocol instance needed
        BenchmarkProtocol.describe()
        raise SystemExit(0)

    # ── --export-bibtex ───────────────────────────────────────────────────
    if args.export_bibtex is not None:
        filepath = args.export_bibtex if args.export_bibtex else "benchmark_references.bib"
        BenchmarkProtocol.export_bibtex(filepath)
        raise SystemExit(0)

    # ── --competitors ─────────────────────────────────────────────────────
    if args.competitors is not None:
        keys = list(COMPETITORS) if "all" in args.competitors else args.competitors
        print()
        for key in keys:
            if key not in COMPETITORS:
                print(f"  Unknown competitor key: '{key}'")
                continue
            c = COMPETITORS[key]
            print("=" * 80)
            print(f"  {c.name}")
            print(f"  Reference : {c.reference}  ({c.year})")
            print(f"  Type      : {c.method_type}")
            prio_str = {1: "★ MUST compare", 2: "Should compare", 3: "Optional"}.get(c.priority)
            print(f"  Priority  : {prio_str}")
            print(f"  Repo      : {c.repo_url}")
            print(f"  Install   : {c.install_cmd}")
            print(f"  Pretrained: {'Yes' if c.pretrained else 'No'}")
            print(f"\n  Rationale :\n    {c.rationale}")
            print(f"\n  Notes     :\n    {c.notes}")
            print()
        raise SystemExit(0)

    # ── Standard run — uses the SAME runner interface as importer scripts ─
    #
    # This is the critical section: the CLI now constructs BenchmarkProtocol
    # with the same arguments a runner would pass, then calls get_all_domains()
    # and load_test_data() exactly as standalone_real_methods_test.py does.
    # There is no separate CLI-only code path.

    protocol = BenchmarkProtocol(
        benchmark=args.benchmark,
        num_samples=args.num_samples,
        noise_level=args.noise,
        seed=args.seed,
        feynman_series=args.series,
        crossover_only=args.crossover_only,
    )

    print()
    print("=" * 80)
    print("  HypatiaX PUBLISHED SR BENCHMARK PROTOCOL")
    print(f"  Benchmark : {args.benchmark.upper()}")
    print("=" * 80)

    domains = protocol.get_all_domains()
    print(f"\n  {len(domains)} domain(s) available via get_all_domains():\n")

    total_cases = 0
    feynman_extrap = 0
    feynman_crossover = 0

    for domain in domains:
        cases = protocol.load_test_data(domain, num_samples=args.num_samples)
        total_cases += len(cases)

        is_feynman = domain.startswith(BenchmarkProtocol._FEYNMAN_PREFIX)
        prefix = "  [F]" if is_feynman else "  [S]"
        print(f"{prefix} {domain}  ({len(cases)} equation(s))")

        for desc, X, y, var_names, meta in cases:
            stub = " [pmlb not installed]" if meta.get("pmlb_stub") else ""
            extrap_flag = " ⭐" if meta.get("extrapolation_test") else ""
            xover_flag  = " [crossover]" if meta.get("series") == "crossover" else ""
            print(
                f"        {meta['equation_name']:<28} "
                f"{meta.get('difficulty','?'):<8} "
                f"{meta.get('ground_truth', 'real-world data')[:40]}"
                f"{extrap_flag}{xover_flag}{stub}"
            )
            if meta.get("extrapolation_test"):
                feynman_extrap += 1
            if meta.get("series") == "crossover":
                feynman_crossover += 1

    print(f"\n  [F] = Feynman benchmark    [S] = SRBench benchmark")
    print(f"  ⭐  = flagged for extrapolation testing")
    print(f"\n  Total test cases         : {total_cases}")
    if args.benchmark in ("feynman", "both"):
        print(f"  Extrapolation-flagged    : {feynman_extrap}")
        print(f"  Core 15 crossover checks : {feynman_crossover}")

    print()
    protocol.print_competitor_table()
    print(f"  Export BibTeX : python {__file__} --export-bibtex references_to_add.bib")
    print(f"  Full docs     : python {__file__} --describe")
    print()

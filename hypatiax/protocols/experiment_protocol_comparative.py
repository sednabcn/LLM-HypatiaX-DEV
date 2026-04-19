"""
Experiment Protocol for Comparative Test Suite
===============================================

Provides clean test cases specifically designed for comparative testing
between LLM+PySR, Pure PySR, Pure LLM, NN, and LLM+NN Ensemble.

Key Features:
- Proper variable separation (varying vs constants)
- Realistic data ranges
- Clean ground truth formulas
- Compatible with comparative_test_suite.py

Domains:
- Scientific (chemistry, biology, physics)
- DeFi (amm, risk, liquidity)

Author: HypatiaX Team
Version: 1.0
Date: 2026-01-15
"""

import numpy as np
from typing import List, Tuple, Dict


class ComparativeExperimentProtocol:
    """Experiment protocol optimized for comparative testing"""

    @staticmethod
    def get_all_domains() -> List[str]:
        """Return list of all domains"""
        return [
            "chemistry",
            "biology",
            "physics",
            "defi_amm",
            "defi_risk",
        ]

    @staticmethod
    def get_scientific_domains() -> List[str]:
        """Return scientific domains only"""
        return ["chemistry", "biology", "physics"]

    @staticmethod
    def get_defi_domains() -> List[str]:
        """Return DeFi domains only"""
        return ["defi_amm", "defi_risk"]

    @staticmethod
    def load_test_data(
        domain: str, num_samples: int = 200
    ) -> List[Tuple[str, np.ndarray, np.ndarray, List[str], Dict]]:
        """
        Load test data for a domain

        Returns:
            List of (description, X, y, variable_names, metadata) tuples
        """
        np.random.seed(42)
        test_cases = []

        # ================================================================
        # SCIENTIFIC DOMAINS
        # ================================================================

        if domain == "chemistry":
            # Test 1: Arrhenius Equation - k = A*exp(-Ea/(R*T))
            # Only T varies, others are constants embedded in formula
            T = np.random.uniform(273, 373, num_samples)
            A = 1e11
            Ea = 80000
            R = 8.314

            X = T.reshape(-1, 1)
            y = A * np.exp(-Ea / (R * T))

            test_cases.append(
                (
                    "Arrhenius Equation: k = A*exp(-Ea/(R*T))",
                    X,
                    y,
                    ["T"],
                    {
                        "equation_name": "arrhenius",
                        "domain": "chemistry",
                        "difficulty": "hard",
                        "ground_truth": "1e11 * exp(-80000 / (8.314 * T))",
                        "constants": {"A": 1e11, "Ea": 80000, "R": 8.314},
                        "formula_type": "exponential",
                    },
                )
            )

            # Test 2: Henderson-Hasselbalch - pH = pKa + log10([A-]/[HA])
            pKa = 6.5
            A_minus = np.random.uniform(0.1, 2.0, num_samples)
            HA = np.random.uniform(0.1, 2.0, num_samples)

            X = np.column_stack([A_minus, HA])
            y = pKa + np.log10(A_minus / (HA + 1e-10))

            test_cases.append(
                (
                    "Henderson-Hasselbalch: pH = pKa + log10([A-]/[HA])",
                    X,
                    y,
                    ["A_minus", "HA"],
                    {
                        "equation_name": "henderson_hasselbalch",
                        "domain": "chemistry",
                        "difficulty": "medium",
                        "ground_truth": "6.5 + log10(A_minus / HA)",
                        "constants": {"pKa": 6.5},
                        "formula_type": "logarithmic",
                    },
                )
            )

            # Test 3: Rate Law - rate = k*[A]^m*[B]^n
            k = 0.5
            A_conc = np.random.uniform(0.1, 5.0, num_samples)
            B_conc = np.random.uniform(0.1, 5.0, num_samples)
            m = 2.0
            n = 1.0

            X = np.column_stack([A_conc, B_conc])
            y = k * (A_conc**m) * (B_conc**n)

            test_cases.append(
                (
                    "Rate Law: rate = k*[A]²*[B]",
                    X,
                    y,
                    ["A_conc", "B_conc"],
                    {
                        "equation_name": "rate_law",
                        "domain": "chemistry",
                        "difficulty": "medium",
                        "ground_truth": "0.5 * A_conc**2 * B_conc",
                        "constants": {"k": 0.5, "m": 2.0, "n": 1.0},
                        "formula_type": "power_law",
                    },
                )
            )

        elif domain == "biology":
            # Test 1: Allometric Scaling - Y = a*M^b
            a = 3.5
            M = np.random.uniform(1, 100, num_samples)
            b = 0.75

            X = M.reshape(-1, 1)
            y = a * (M**b)

            test_cases.append(
                (
                    "Allometric Scaling: Y = a*M^b (metabolic rate)",
                    X,
                    y,
                    ["M"],
                    {
                        "equation_name": "allometric_scaling",
                        "domain": "biology",
                        "difficulty": "easy",
                        "ground_truth": "3.5 * M**0.75",
                        "constants": {"a": 3.5, "b": 0.75},
                        "formula_type": "power_law",
                    },
                )
            )

            # Test 2: Michaelis-Menten - v = (Vmax*S)/(Km+S)
            Vmax = 50.0
            S = np.random.uniform(0.1, 50, num_samples)
            Km = 10.0

            X = S.reshape(-1, 1)
            y = (Vmax * S) / (Km + S)

            test_cases.append(
                (
                    "Michaelis-Menten: v = (Vmax*[S])/(Km+[S])",
                    X,
                    y,
                    ["S"],
                    {
                        "equation_name": "michaelis_menten",
                        "domain": "biology",
                        "difficulty": "medium",
                        "ground_truth": "(50 * S) / (10 + S)",
                        "constants": {"Vmax": 50.0, "Km": 10.0},
                        "formula_type": "rational",
                    },
                )
            )

            # Test 3: Logistic Growth - dN/dt = r*N*(1-N/K)
            r = 0.3
            N = np.random.uniform(10, 900, num_samples)
            K = 1000.0

            X = N.reshape(-1, 1)
            y = r * N * (1 - N / K)

            test_cases.append(
                (
                    "Logistic Growth: dN/dt = r*N*(1-N/K)",
                    X,
                    y,
                    ["N"],
                    {
                        "equation_name": "logistic_growth",
                        "domain": "biology",
                        "difficulty": "medium",
                        "ground_truth": "0.3 * N * (1 - N/1000)",
                        "constants": {"r": 0.3, "K": 1000.0},
                        "formula_type": "nonlinear",
                    },
                )
            )

        elif domain == "physics":
            # Test 1: Kinetic Energy - KE = 0.5*m*v²
            m = np.random.uniform(0.1, 10, num_samples)
            v = np.random.uniform(0.1, 50, num_samples)

            X = np.column_stack([m, v])
            y = 0.5 * m * (v**2)

            test_cases.append(
                (
                    "Kinetic Energy: KE = 0.5*m*v²",
                    X,
                    y,
                    ["m", "v"],
                    {
                        "equation_name": "kinetic_energy",
                        "domain": "physics",
                        "difficulty": "easy",
                        "ground_truth": "0.5 * m * v**2",
                        "constants": {},
                        "formula_type": "power_law",
                    },
                )
            )

            # Test 2: Gravitational Force - F = G*m1*m2/r²
            G = 6.674e-11
            m1 = np.random.uniform(1e20, 1e24, num_samples)
            m2 = np.random.uniform(1e20, 1e24, num_samples)
            r = np.random.uniform(1e6, 1e9, num_samples)

            X = np.column_stack([m1, m2, r])
            y = G * m1 * m2 / (r**2)

            test_cases.append(
                (
                    "Gravitational Force: F = G*m1*m2/r²",
                    X,
                    y,
                    ["m1", "m2", "r"],
                    {
                        "equation_name": "gravitational_force",
                        "domain": "physics",
                        "difficulty": "medium",
                        "ground_truth": "6.674e-11 * m1 * m2 / r**2",
                        "constants": {"G": 6.674e-11},
                        "formula_type": "inverse_square",
                    },
                )
            )

            # Test 3: Ideal Gas Law - P = nRT/V
            n = np.random.uniform(0.1, 10, num_samples)
            R = 8.314
            T = np.random.uniform(200, 400, num_samples)
            V = np.random.uniform(0.01, 1, num_samples)

            X = np.column_stack([n, T, V])
            y = n * R * T / V

            test_cases.append(
                (
                    "Ideal Gas Law: P = nRT/V",
                    X,
                    y,
                    ["n", "T", "V"],
                    {
                        "equation_name": "ideal_gas_law",
                        "domain": "physics",
                        "difficulty": "easy",
                        "ground_truth": "n * 8.314 * T / V",
                        "constants": {"R": 8.314},
                        "formula_type": "algebraic",
                    },
                )
            )

        # ================================================================
        # DEFI DOMAINS
        # ================================================================

        elif domain == "defi_amm":
            # Test 1: Impermanent Loss - IL = 2*sqrt(r)/(1+r) - 1
            price_ratio = np.random.uniform(0.5, 2.5, num_samples)

            X = price_ratio.reshape(-1, 1)
            y = 2 * np.sqrt(price_ratio) / (1 + price_ratio) - 1

            test_cases.append(
                (
                    "Impermanent Loss: IL = 2*sqrt(r)/(1+r) - 1",
                    X,
                    y,
                    ["price_ratio"],
                    {
                        "equation_name": "impermanent_loss",
                        "domain": "defi",
                        "difficulty": "hard",
                        "ground_truth": "2*sqrt(r)/(1+r) - 1",
                        "constants": {},
                        "formula_type": "rational_sqrt",
                    },
                )
            )

            # Test 2: Price Impact - impact = dx/(x+dx)
            reserve = np.random.uniform(10000, 100000, num_samples)
            swap = reserve * np.random.uniform(0.001, 0.1, num_samples)

            X = np.column_stack([reserve, swap])
            y = swap / (reserve + swap)

            test_cases.append(
                (
                    "Price Impact: impact = dx/(x+dx)",
                    X,
                    y,
                    ["reserve", "swap"],
                    {
                        "equation_name": "price_impact",
                        "domain": "defi",
                        "difficulty": "easy",
                        "ground_truth": "swap / (reserve + swap)",
                        "constants": {},
                        "formula_type": "rational",
                    },
                )
            )

            # Test 3: Constant Product - y = k/x
            x = np.random.uniform(100, 10000, num_samples)
            k = 1e6

            X = x.reshape(-1, 1)
            y = k / x

            test_cases.append(
                (
                    "Constant Product: y = k/x",
                    X,
                    y,
                    ["x"],
                    {
                        "equation_name": "constant_product",
                        "domain": "defi",
                        "difficulty": "easy",
                        "ground_truth": "1000000 / x",
                        "constants": {"k": 1e6},
                        "formula_type": "inverse",
                    },
                )
            )

        elif domain == "defi_risk":
            # Test 1: Value at Risk 95% - VaR = portfolio * vol * 1.645
            portfolio = np.random.uniform(10000, 1000000, num_samples)
            volatility = np.random.uniform(0.01, 0.05, num_samples)

            X = np.column_stack([portfolio, volatility])
            y = portfolio * volatility * 1.645

            test_cases.append(
                (
                    "Value at Risk 95%: VaR = P*σ*1.645",
                    X,
                    y,
                    ["portfolio", "volatility"],
                    {
                        "equation_name": "var_95",
                        "domain": "defi",
                        "difficulty": "easy",
                        "ground_truth": "portfolio * volatility * 1.645",
                        "constants": {"z_score": 1.645},
                        "formula_type": "multiplicative",
                    },
                )
            )

            # Test 2: Liquidation Price LONG - liq = entry*(1 - 1/(leverage*0.8))
            entry_price = np.random.uniform(30000, 50000, num_samples)
            leverage = np.random.uniform(2, 10, num_samples)

            X = np.column_stack([entry_price, leverage])
            y = entry_price * (1 - 1 / (leverage * 0.8))

            test_cases.append(
                (
                    "Liquidation Price LONG: liq = entry*(1 - 1/(L*0.8))",
                    X,
                    y,
                    ["entry_price", "leverage"],
                    {
                        "equation_name": "liquidation_long",
                        "domain": "defi",
                        "difficulty": "hard",
                        "ground_truth": "entry * (1 - 1/(leverage*0.8))",
                        "constants": {"maintenance_margin": 0.8},
                        "formula_type": "complex_rational",
                    },
                )
            )

            # Test 3: Portfolio VaR - sqrt(var1² + var2² + 2*ρ*var1*var2)
            var1 = np.random.uniform(5000, 50000, num_samples)
            var2 = np.random.uniform(3000, 30000, num_samples)
            rho = np.random.uniform(-0.5, 0.9, num_samples)

            X = np.column_stack([var1, var2, rho])
            y = np.sqrt(var1**2 + var2**2 + 2 * rho * var1 * var2)

            test_cases.append(
                (
                    "Portfolio VaR: sqrt(var1² + var2² + 2ρ*var1*var2)",
                    X,
                    y,
                    ["var1", "var2", "rho"],
                    {
                        "equation_name": "portfolio_var",
                        "domain": "defi",
                        "difficulty": "hard",
                        "ground_truth": "sqrt(var1**2 + var2**2 + 2*rho*var1*var2)",
                        "constants": {},
                        "formula_type": "quadratic_sqrt",
                    },
                )
            )

        return test_cases

    @staticmethod
    def get_domain_description(domain: str) -> str:
        """Get domain description"""
        descriptions = {
            "chemistry": "Chemistry - reaction kinetics, equilibria, thermodynamics",
            "biology": "Biology - enzyme kinetics, population dynamics, scaling laws",
            "physics": "Physics - mechanics, thermodynamics, gravitational laws",
            "defi_amm": "DeFi AMM - impermanent loss, price impact, constant product",
            "defi_risk": "DeFi Risk - VaR, liquidation, portfolio risk",
        }
        return descriptions.get(domain, "Unknown domain")


if __name__ == "__main__":
    protocol = ComparativeExperimentProtocol()

    print("=" * 80)
    print("COMPARATIVE EXPERIMENT PROTOCOL v1.0".center(80))
    print("=" * 80)

    print("\n📊 Scientific Domains:")
    for domain in protocol.get_scientific_domains():
        tests = protocol.load_test_data(domain, num_samples=10)
        print(f"\n  {domain.upper()}: {len(tests)} tests")
        print(f"  {protocol.get_domain_description(domain)}")
        for desc, X, y, vars, meta in tests:
            print(f"    • {meta['equation_name']}: {desc}")

    print("\n\n💰 DeFi Domains:")
    for domain in protocol.get_defi_domains():
        tests = protocol.load_test_data(domain, num_samples=10)
        print(f"\n  {domain.upper()}: {len(tests)} tests")
        print(f"  {protocol.get_domain_description(domain)}")
        for desc, X, y, vars, meta in tests:
            print(f"    • {meta['equation_name']}: {desc}")

    print("\n" + "=" * 80)
    total = sum(len(protocol.load_test_data(d, 10)) for d in protocol.get_all_domains())
    print(f"Total test cases: {total}")
    print("=" * 80)

"""
Experiment Protocol A: 18 Physics & Engineering Test Cases
============================================================
Focus: Classical physics, quantum mechanics, optics, fluid dynamics, thermodynamics, electromagnetism

Aligned with complete_hybrid_system_all_domains_.py (v3.6) test cases
"""

import numpy as np
from typing import List, Tuple, Dict
import json
import os


class ExperimentProtocolA:
    """18 test cases from mechanics, thermodynamics, EM, fluids, optics, quantum."""

    @staticmethod
    def get_all_domains() -> List[str]:
        """Return list of all experimental domains."""
        return [
            "mechanics",
            "thermodynamics",
            "electromagnetism",
            "fluid_dynamics",
            "optics",
            "quantum",
        ]

    @staticmethod
    def load_test_data(
        domain: str, num_samples: int = 300
    ) -> List[Tuple[str, np.ndarray, np.ndarray, List[str], Dict]]:
        """
        Load test data for Protocol A (18 physics/engineering cases).

        Returns:
            List of (description, X, y, variable_names, metadata) tuples
        """
        np.random.seed(42)
        test_cases = []

        if domain == "mechanics":
            # ========== MECHANICS (3 tests) ==========

            # 1. Kinetic Energy
            m = np.random.uniform(0.1, 10, num_samples)
            v = np.random.uniform(0.1, 50, num_samples)
            X = np.column_stack([m, v])
            y = 0.5 * m * v**2
            test_cases.append(
                (
                    "Kinetic Energy: KE = (1/2)*m*v²",
                    X,
                    y,
                    ["m", "v"],
                    {
                        "equation_name": "kinetic_energy",
                        "difficulty": "easy",
                        "formula_type": "power_law",
                        "ground_truth": "(1/2)*m*v**2",
                        "units": {"m": "kg", "v": "m/s", "KE": "J"},
                        "test_config": "simple_product",
                    },
                )
            )

            # 2. Gravitational Potential Energy
            m = np.random.uniform(0.1, 100, num_samples)
            g = np.random.uniform(9.7, 9.9, num_samples)
            h = np.random.uniform(0, 100, num_samples)
            X = np.column_stack([m, g, h])
            y = m * g * h
            test_cases.append(
                (
                    "Gravitational Potential Energy: PE = m*g*h",
                    X,
                    y,
                    ["m", "g", "h"],
                    {
                        "equation_name": "gravitational_potential_energy",
                        "difficulty": "easy",
                        "formula_type": "product",
                        "ground_truth": "m*g*h",
                        "units": {"m": "kg", "g": "m/s^2", "h": "m", "PE": "J"},
                    },
                )
            )

            # 3. Hooke's Law (Spring Force)
            k = np.random.uniform(1, 100, num_samples)
            x = np.random.uniform(-2, 2, num_samples)
            X = np.column_stack([k, x])
            y = k * np.abs(x)
            test_cases.append(
                (
                    "Hooke's Law: F = k*x",
                    X,
                    y,
                    ["k", "x"],
                    {
                        "equation_name": "hookes_law",
                        "difficulty": "easy",
                        "formula_type": "linear",
                        "ground_truth": "k*x",
                        "units": {"k": "N/m", "x": "m", "F": "N"},
                        "test_config": "simple_product",
                    },
                )
            )

        elif domain == "thermodynamics":
            # ========== THERMODYNAMICS (3 tests) ==========

            # 1. Ideal Gas Law
            n = np.random.uniform(0.1, 10, num_samples)
            R = np.full(num_samples, 8.314)
            T = np.random.uniform(200, 400, num_samples)
            V = np.random.uniform(0.01, 1, num_samples)
            X = np.column_stack([n, R, T, V])
            y = n * R * T / V
            test_cases.append(
                (
                    "Ideal Gas Law: PV = nRT => P = nRT/V",
                    X,
                    y,
                    ["n", "R", "T", "V"],
                    {
                        "equation_name": "ideal_gas_law",
                        "difficulty": "medium",
                        "formula_type": "algebraic",
                        "ground_truth": "n*R*T/V",
                        "units": {
                            "n": "mol",
                            "R": "J/(mol*K)",
                            "T": "K",
                            "V": "m^3",
                            "P": "Pa",
                        },
                    },
                )
            )

            # 2. Heat Capacity
            m = np.random.uniform(0.1, 10, num_samples)
            c = np.random.uniform(100, 5000, num_samples)
            dT = np.random.uniform(1, 100, num_samples)
            X = np.column_stack([m, c, dT])
            y = m * c * dT
            test_cases.append(
                (
                    "Heat Capacity: Q = m*c*ΔT",
                    X,
                    y,
                    ["m", "c", "dT"],
                    {
                        "equation_name": "heat_capacity",
                        "difficulty": "easy",
                        "formula_type": "product",
                        "ground_truth": "m*c*dT",
                        "units": {"m": "kg", "c": "J/(kg*K)", "dT": "K", "Q": "J"},
                    },
                )
            )

            # 3. Carnot Efficiency
            Tc = np.random.uniform(200, 300, num_samples)
            Th = np.random.uniform(400, 600, num_samples)
            X = np.column_stack([Tc, Th])
            y = 1 - Tc / Th
            test_cases.append(
                (
                    "Carnot Efficiency: η = 1 - Tc/Th",
                    X,
                    y,
                    ["Tc", "Th"],
                    {
                        "equation_name": "carnot_efficiency",
                        "difficulty": "easy",
                        "formula_type": "algebraic",
                        "ground_truth": "1 - Tc/Th",
                        "units": {"Tc": "K", "Th": "K", "eta": "dimensionless"},
                    },
                )
            )

        elif domain == "electromagnetism":
            # ========== ELECTROMAGNETISM (3 tests) ==========

            # 1. Coulomb's Law
            k = np.full(num_samples, 8.99e9)
            q1 = np.random.uniform(1e-9, 1e-6, num_samples)
            q2 = np.random.uniform(1e-9, 1e-6, num_samples)
            r = np.random.uniform(0.01, 1, num_samples)
            X = np.column_stack([k, q1, q2, r])
            y = k * q1 * q2 / r**2
            test_cases.append(
                (
                    "Coulomb's Law: F = k*q1*q2/r²",
                    X,
                    y,
                    ["k", "q1", "q2", "r"],
                    {
                        "equation_name": "coulomb_law",
                        "difficulty": "medium",
                        "formula_type": "power_law",
                        "ground_truth": "k*q1*q2/r**2",
                        "units": {
                            "k": "N*m^2/C^2",
                            "q1": "C",
                            "q2": "C",
                            "r": "m",
                            "F": "N",
                        },
                    },
                )
            )

            # 2. Ohm's Law
            I = np.random.uniform(0.1, 10, num_samples)
            R = np.random.uniform(1, 1000, num_samples)
            X = np.column_stack([I, R])
            y = I * R
            test_cases.append(
                (
                    "Ohm's Law: V = I*R",
                    X,
                    y,
                    ["I", "R"],
                    {
                        "equation_name": "ohms_law",
                        "difficulty": "easy",
                        "formula_type": "linear",
                        "ground_truth": "I*R",
                        "units": {"I": "A", "R": "Ω", "V": "V"},
                        "test_config": "simple_product",
                    },
                )
            )

            # 3. Magnetic Force (Lorentz)
            q = np.random.uniform(1e-9, 1e-6, num_samples)
            v = np.random.uniform(1, 100, num_samples)
            B = np.random.uniform(0.1, 10, num_samples)
            X = np.column_stack([q, v, B])
            y = q * v * B
            test_cases.append(
                (
                    "Lorentz Force: F = q*v*B",
                    X,
                    y,
                    ["q", "v", "B"],
                    {
                        "equation_name": "lorentz_force",
                        "difficulty": "easy",
                        "formula_type": "product",
                        "ground_truth": "q*v*B",
                        "units": {"q": "C", "v": "m/s", "B": "T", "F": "N"},
                    },
                )
            )

        elif domain == "fluid_dynamics":
            # ========== FLUID DYNAMICS (3 tests) ==========

            # 1. Bernoulli's Equation (Simplified)
            P = np.random.uniform(1e5, 2e5, num_samples)
            rho = np.random.uniform(800, 1200, num_samples)
            v = np.random.uniform(0.1, 10, num_samples)
            X = np.column_stack([P, rho, v])
            y = P + 0.5 * rho * v**2
            test_cases.append(
                (
                    "Bernoulli Equation: Total = P + (1/2)*ρ*v²",
                    X,
                    y,
                    ["P", "rho", "v"],
                    {
                        "equation_name": "bernoulli_equation",
                        "difficulty": "medium",
                        "formula_type": "algebraic",
                        "ground_truth": "P + 0.5*rho*v**2",
                        "units": {"P": "Pa", "rho": "kg/m^3", "v": "m/s"},
                        "test_config": "complex",
                    },
                )
            )

            # 2. Reynolds Number
            rho = np.random.uniform(800, 1200, num_samples)
            v = np.random.uniform(0.1, 10, num_samples)
            L = np.random.uniform(0.01, 1, num_samples)
            mu = np.random.uniform(0.001, 0.1, num_samples)
            X = np.column_stack([rho, v, L, mu])
            y = rho * v * L / mu
            test_cases.append(
                (
                    "Reynolds Number: Re = ρ*v*L/μ",
                    X,
                    y,
                    ["rho", "v", "L", "mu"],
                    {
                        "equation_name": "reynolds_number",
                        "difficulty": "easy",
                        "formula_type": "algebraic",
                        "ground_truth": "rho*v*L/mu",
                        "units": {"rho": "kg/m^3", "v": "m/s", "L": "m", "mu": "Pa*s"},
                    },
                )
            )

            # 3. Hagen-Poiseuille Flow
            dP = np.random.uniform(100, 10000, num_samples)
            r = np.random.uniform(0.001, 0.1, num_samples)
            mu = np.full(num_samples, 0.001)
            L = np.random.uniform(0.1, 10, num_samples)
            X = np.column_stack([dP, r, mu, L])
            y = (np.pi * r**4 * dP) / (8 * mu * L)
            test_cases.append(
                (
                    "Hagen-Poiseuille: Q = π*r⁴*ΔP/(8*μ*L)",
                    X,
                    y,
                    ["dP", "r", "mu", "L"],
                    {
                        "equation_name": "hagen_poiseuille",
                        "difficulty": "hard",
                        "formula_type": "power_law",
                        "ground_truth": "dP*r**4/(mu*L)",
                        "units": {
                            "dP": "Pa",
                            "r": "m",
                            "mu": "Pa*s",
                            "L": "m",
                            "Q": "m^3/s",
                        },
                    },
                )
            )

        elif domain == "optics":
            # ========== OPTICS (3 tests) ==========

            # 1. Thin Lens Equation
            do = np.random.uniform(0.1, 10, num_samples)
            di = np.random.uniform(0.1, 10, num_samples)
            X = np.column_stack([do, di])
            y = 1 / do + 1 / di
            test_cases.append(
                (
                    "Thin Lens: 1/f = 1/do + 1/di",
                    X,
                    y,
                    ["do", "di"],
                    {
                        "equation_name": "thin_lens_equation",
                        "difficulty": "easy",
                        "formula_type": "algebraic",
                        "ground_truth": "1/do + 1/di",
                        "units": {"do": "m", "di": "m", "f": "1/m"},
                    },
                )
            )

            # 2. Snell's Law
            n1 = np.random.uniform(1.0, 2.5, num_samples)
            sin_theta1 = np.random.uniform(0.1, 0.9, num_samples)
            X = np.column_stack([n1, sin_theta1])
            y = n1 * sin_theta1
            test_cases.append(
                (
                    "Snell's Law: n1*sin(θ1) = n2*sin(θ2)",
                    X,
                    y,
                    ["n1", "sin_theta1"],
                    {
                        "equation_name": "snells_law",
                        "difficulty": "easy",
                        "formula_type": "linear",
                        "ground_truth": "n1*sin_theta1",
                        "units": {"n1": "dimensionless", "sin_theta1": "dimensionless"},
                    },
                )
            )

            # 3. Single Slit Diffraction
            wavelength = np.random.uniform(400e-9, 700e-9, num_samples)
            a = np.random.uniform(1e-6, 1e-4, num_samples)
            X = np.column_stack([wavelength, a])
            y = wavelength / a
            test_cases.append(
                (
                    "Diffraction: sin(θ) = λ/a",
                    X,
                    y,
                    ["wavelength", "a"],
                    {
                        "equation_name": "single_slit_diffraction",
                        "difficulty": "easy",
                        "formula_type": "algebraic",
                        "ground_truth": "wavelength/a",
                        "units": {"wavelength": "m", "a": "m"},
                    },
                )
            )

        elif domain == "quantum":
            # ========== QUANTUM MECHANICS (3 tests) ==========

            # 1. Photon Energy
            h = np.full(num_samples, 6.626e-34)
            f = np.random.uniform(1e14, 1e15, num_samples)
            X = np.column_stack([h, f])
            y = h * f
            test_cases.append(
                (
                    "Photon Energy: E = h*f",
                    X,
                    y,
                    ["h", "f"],
                    {
                        "equation_name": "photon_energy",
                        "difficulty": "easy",
                        "formula_type": "linear",
                        "ground_truth": "h*f",
                        "units": {"h": "J*s", "f": "Hz", "E": "J"},
                    },
                )
            )

            # 2. de Broglie Wavelength
            h = np.full(num_samples, 6.626e-34)
            m = np.random.uniform(1e-30, 1e-27, num_samples)
            v = np.random.uniform(1e3, 1e6, num_samples)
            X = np.column_stack([h, m, v])
            y = h / (m * v)
            test_cases.append(
                (
                    "de Broglie Wavelength: λ = h/(m*v)",
                    X,
                    y,
                    ["h", "m", "v"],
                    {
                        "equation_name": "de_broglie_wavelength",
                        "difficulty": "easy",
                        "formula_type": "algebraic",
                        "ground_truth": "h/(m*v)",
                        "units": {"h": "J*s", "m": "kg", "v": "m/s", "lambda": "m"},
                    },
                )
            )

            # 3. Compton Scattering
            h = np.full(num_samples, 6.626e-34)
            me = np.full(num_samples, 9.109e-31)
            c = np.full(num_samples, 3e8)
            cos_theta = np.random.uniform(-1, 1, num_samples)
            X = np.column_stack([h, me, c, cos_theta])
            y = (h / (me * c)) * (1 - cos_theta)
            test_cases.append(
                (
                    "Compton Shift: Δλ = (h/(me*c))*(1-cos(θ))",
                    X,
                    y,
                    ["h", "me", "c", "cos_theta"],
                    {
                        "equation_name": "compton_shift",
                        "difficulty": "medium",
                        "formula_type": "algebraic",
                        "ground_truth": "(h/(me*c))*(1 - cos_theta)",
                        "units": {
                            "h": "J*s",
                            "me": "kg",
                            "c": "m/s",
                            "cos_theta": "dimensionless",
                        },
                    },
                )
            )

        return test_cases

    @staticmethod
    def get_domain_description(domain: str) -> str:
        """Get domain description."""
        descriptions = {
            "mechanics": "Classical Mechanics - kinematics, dynamics, energy",
            "thermodynamics": "Thermodynamics - heat, temperature, efficiency",
            "electromagnetism": "Electromagnetism - forces, circuits, fields",
            "fluid_dynamics": "Fluid Dynamics - flow, pressure, viscosity",
            "optics": "Optics - light, refraction, diffraction",
            "quantum": "Quantum Mechanics - photons, waves, particles",
        }
        return descriptions.get(domain, "Unknown domain")

    @staticmethod
    def save_protocol_documentation(filepath: str = "docs/experiment_protocol_a.json"):
        """Save protocol documentation."""
        protocol_doc = {
            "title": "Experiment Protocol A: 18 Physics & Engineering Cases",
            "version": "1.0",
            "date": "2026-01-03",
            "total_tests": 18,
            "domains": {},
        }

        for domain in ExperimentProtocolA.get_all_domains():
            test_cases = ExperimentProtocolA.load_test_data(domain, num_samples=10)
            protocol_doc["domains"][domain] = {
                "description": ExperimentProtocolA.get_domain_description(domain),
                "num_test_cases": len(test_cases),
                "test_cases": [
                    {
                        "description": desc,
                        "variables": vars,
                        "equation_name": meta.get("equation_name"),
                        "difficulty": meta["difficulty"],
                        "ground_truth": meta["ground_truth"],
                    }
                    for desc, _, _, vars, meta in test_cases
                ],
            }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(protocol_doc, f, indent=2)

        print(f"Protocol A documentation saved to: {filepath}")
        return protocol_doc


if __name__ == "__main__":
    protocol = ExperimentProtocolA()

    print("=" * 80)
    print("EXPERIMENT PROTOCOL A: 18 PHYSICS & ENGINEERING TEST CASES".center(80))
    print("=" * 80)

    for domain in protocol.get_all_domains():
        test_cases = protocol.load_test_data(domain, num_samples=10)
        print(f"\n{domain.upper()} ({len(test_cases)} tests):")
        for i, (desc, _, _, vars, meta) in enumerate(test_cases, 1):
            print(f"  {i}. {desc}")
            print(f"     Variables: {', '.join(vars)}")
            print(f"     Ground truth: {meta['ground_truth']}")

    total = sum(len(protocol.load_test_data(d)) for d in protocol.get_all_domains())
    print(f"\n{'=' * 80}")
    print(f"Total: {total} test cases")
    print(f"{'=' * 80}")

    protocol.save_protocol_documentation()

"""
Experiment Protocol ALL v4.0: 30 Complete Multi-Domain Test Cases - BEST OF BOTH
==================================================================================
FIXES in v4.0:
✅ Complete implementation from v2.0 (all domains fully coded)
✅ Quantum fixes from v2.1 (normalized units for better numerical properties)
✅ All comprehensive metadata and documentation from v2.0
✅ Enhanced structure hints for difficult equations
✅ Compatible with suite v4.3

Focus: All scientific domains with comprehensive coverage
- Physics/Engineering: 18 tests (mechanics, thermodynamics, EM, fluids, optics, quantum)
- Multi-Domain: 12 additional tests (chemistry, biology, mathematics, economics)

Total: 30 complete test cases

Author: HypatiaX Team
Version: 4.0 COMPLETE
Date: 2026-01-13
"""

import numpy as np
from typing import List, Tuple, Dict
import json
import os


class ExperimentProtocolAll:
    """Complete protocol with all 30 test cases - v2.2 BEST OF BOTH"""

    @staticmethod
    def get_all_domains() -> List[str]:
        """Return list of all experimental domains."""
        return [
            # Protocol A domains (Physics/Engineering)
            "mechanics",
            "thermodynamics",
            "electromagnetism",
            "fluid_dynamics",
            "optics",
            "quantum",
            # Protocol B domains (Multi-Domain)
            "chemistry",
            "biology",
            "mathematics",
            "economics",
        ]

    @staticmethod
    def load_test_data(
        domain: str, num_samples: int = 300
    ) -> List[Tuple[str, np.ndarray, np.ndarray, List[str], Dict]]:
        """
        Load test data for all 30 test cases.

        Returns:
            List of (description, X, y, variable_names, metadata) tuples
        """
        np.random.seed(42)
        test_cases = []

        # ====================================================================
        # PROTOCOL A: PHYSICS & ENGINEERING (18 tests)
        # ====================================================================

        if domain == "mechanics":
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
                        "ground_truth": "0.5 * m * v**2",
                        "units": {"m": "kg", "v": "m/s", "KE": "J"},
                        "variable_descriptions": {
                            "m": "Object mass",
                            "v": "Object velocity",
                        },
                        "variable_roles": {"m": "varying", "v": "varying"},
                        "structure_hints": {
                            "v": "quadratic",
                            "multiplicative_terms": True,
                        },
                        "use_enhanced_config": True,
                        "protocol": "A",
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
                        "ground_truth": "m * g * h",
                        "units": {"m": "kg", "g": "m/s^2", "h": "m", "PE": "J"},
                        "variable_descriptions": {
                            "m": "Object mass",
                            "g": "Gravitational acceleration",
                            "h": "Height above reference",
                        },
                        "variable_roles": {
                            "m": "varying",
                            "g": "constant",
                            "h": "varying",
                        },
                        "protocol": "A",
                    },
                )
            )

            # 3. Hooke's Law
            k = np.random.uniform(1, 100, num_samples)
            x = np.random.uniform(-2, 2, num_samples)
            X = np.column_stack([k, x])
            y = k * x
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
                        "ground_truth": "k * x",
                        "units": {"k": "N/m", "x": "m", "F": "N"},
                        "variable_descriptions": {
                            "k": "Spring stiffness constant",
                            "x": "Displacement from equilibrium (signed)",
                        },
                        "variable_roles": {"k": "varying", "x": "varying"},
                        "protocol": "A",
                    },
                )
            )

        elif domain == "thermodynamics":
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
                        "ground_truth": "n * R * T / V",
                        "units": {
                            "n": "mol",
                            "R": "J/(mol*K)",
                            "T": "K",
                            "V": "m^3",
                            "P": "Pa",
                        },
                        "variable_descriptions": {
                            "n": "Number of moles of gas",
                            "R": "Universal gas constant",
                            "T": "Absolute temperature",
                            "V": "Volume of gas container",
                        },
                        "variable_roles": {
                            "n": "varying",
                            "R": "constant",
                            "T": "varying",
                            "V": "varying",
                        },
                        "protocol": "A",
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
                        "ground_truth": "m * c * dT",
                        "units": {"m": "kg", "c": "J/(kg*K)", "dT": "K", "Q": "J"},
                        "variable_descriptions": {
                            "m": "Mass of substance",
                            "c": "Specific heat capacity",
                            "dT": "Temperature change",
                        },
                        "variable_roles": {
                            "m": "varying",
                            "c": "varying",
                            "dT": "varying",
                        },
                        "protocol": "A",
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
                        "ground_truth": "1 - Tc / Th",
                        "units": {"Tc": "K", "Th": "K", "eta": "dimensionless"},
                        "variable_descriptions": {
                            "Tc": "Cold reservoir temperature",
                            "Th": "Hot reservoir temperature",
                        },
                        "variable_roles": {"Tc": "varying", "Th": "varying"},
                        "protocol": "A",
                    },
                )
            )

        elif domain == "electromagnetism":
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
                        "ground_truth": "k * q1 * q2 / r**2",
                        "units": {
                            "k": "N*m^2/C^2",
                            "q1": "C",
                            "q2": "C",
                            "r": "m",
                            "F": "N",
                        },
                        "variable_descriptions": {
                            "k": "Coulomb's constant",
                            "q1": "First point charge",
                            "q2": "Second point charge",
                            "r": "Distance between charges",
                        },
                        "variable_roles": {
                            "k": "constant",
                            "q1": "varying",
                            "q2": "varying",
                            "r": "varying",
                        },
                        "structure_hints": {"r": "inverse_square"},
                        "protocol": "A",
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
                        "ground_truth": "I * R",
                        "units": {"I": "A", "R": "Ω", "V": "V"},
                        "variable_descriptions": {
                            "I": "Electric current through conductor",
                            "R": "Electrical resistance",
                        },
                        "variable_roles": {"I": "varying", "R": "varying"},
                        "protocol": "A",
                    },
                )
            )

            # 3. Lorentz Force
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
                        "ground_truth": "q * v * B",
                        "units": {"q": "C", "v": "m/s", "B": "T", "F": "N"},
                        "variable_descriptions": {
                            "q": "Electric charge",
                            "v": "Particle velocity",
                            "B": "Magnetic field strength",
                        },
                        "variable_roles": {
                            "q": "varying",
                            "v": "varying",
                            "B": "varying",
                        },
                        "protocol": "A",
                    },
                )
            )

        elif domain == "fluid_dynamics":
            # 1. Bernoulli's Equation
            P = np.random.uniform(1e5, 2e5, num_samples)
            rho = np.random.uniform(800, 1200, num_samples)
            v = np.random.uniform(0.1, 15.0, num_samples)
            g = np.random.uniform(9.6, 9.9, num_samples)
            h = np.random.uniform(0, 10, num_samples)
            X = np.column_stack([P, rho, v, g, h])
            y = P + 0.5 * rho * v**2 + rho * g * h
            test_cases.append(
                (
                    "Bernoulli's Equation: Total = P + (1/2)*ρ*v² + ρ*g*h",
                    X,
                    y,
                    ["P", "rho", "v", "g", "h"],
                    {
                        "equation_name": "bernoulli_equation",
                        "difficulty": "hard",
                        "formula_type": "additive_polynomial",
                        "ground_truth": "P + 0.5 * rho * v**2 + rho * g * h",
                        "units": {
                            "P": "Pa",
                            "rho": "kg/m^3",
                            "v": "m/s",
                            "g": "m/s^2",
                            "h": "m",
                            "E": "Pa",
                        },
                        "variable_descriptions": {
                            "P": "Static pressure in fluid",
                            "rho": "Fluid density (mass per unit volume)",
                            "v": "Flow velocity of fluid",
                            "g": "Gravitational acceleration",
                            "h": "Height above reference datum",
                        },
                        "variable_roles": {
                            "P": "varying",
                            "rho": "varying",
                            "v": "varying",
                            "g": "varying",
                            "h": "varying",
                        },
                        "structure_hints": {
                            "v": "quadratic",
                            "additive_terms": True,
                            "multiplicative_groups": ["rho*v**2", "rho*g*h"],
                            "has_constant_coefficient": True,
                            "term_count": 3,
                        },
                        "use_enhanced_config": True,
                        "protocol": "A",
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
                        "ground_truth": "rho * v * L / mu",
                        "units": {
                            "rho": "kg/m^3",
                            "v": "m/s",
                            "L": "m",
                            "mu": "Pa*s",
                            "Re": "dimensionless",
                        },
                        "variable_descriptions": {
                            "rho": "Fluid density",
                            "v": "Flow velocity",
                            "L": "Characteristic length scale",
                            "mu": "Dynamic viscosity",
                        },
                        "variable_roles": {
                            "rho": "varying",
                            "v": "varying",
                            "L": "varying",
                            "mu": "varying",
                        },
                        "protocol": "A",
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
                        "ground_truth": "(np.pi * r**4 * dP) / (8 * mu * L)",
                        "units": {
                            "dP": "Pa",
                            "r": "m",
                            "mu": "Pa*s",
                            "L": "m",
                            "Q": "m^3/s",
                        },
                        "variable_descriptions": {
                            "dP": "Pressure difference",
                            "r": "Pipe radius",
                            "mu": "Dynamic viscosity",
                            "L": "Pipe length",
                        },
                        "variable_roles": {
                            "dP": "varying",
                            "r": "varying",
                            "mu": "constant",
                            "L": "varying",
                        },
                        "structure_hints": {"r": "fourth_power"},
                        "protocol": "A",
                    },
                )
            )

        elif domain == "optics":
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
                        "variable_descriptions": {
                            "do": "Object distance from lens",
                            "di": "Image distance from lens",
                        },
                        "variable_roles": {"do": "varying", "di": "varying"},
                        "protocol": "A",
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
                        "ground_truth": "n1 * sin_theta1",
                        "units": {"n1": "dimensionless", "sin_theta1": "dimensionless"},
                        "variable_descriptions": {
                            "n1": "Refractive index of first medium",
                            "sin_theta1": "Sine of incident angle",
                        },
                        "variable_roles": {"n1": "varying", "sin_theta1": "varying"},
                        "protocol": "A",
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
                        "ground_truth": "wavelength / a",
                        "units": {
                            "wavelength": "m",
                            "a": "m",
                            "sin_theta": "dimensionless",
                        },
                        "variable_descriptions": {
                            "wavelength": "Wavelength of light",
                            "a": "Slit width",
                        },
                        "variable_roles": {"wavelength": "varying", "a": "varying"},
                        "protocol": "A",
                    },
                )
            )

        elif domain == "quantum":
            # QUANTUM TESTS - v2.1 FIXES APPLIED

            # 1. Photon Energy - FIXED: Use eV·s units
            h_ev = np.full(num_samples, 4.136e-15)  # Planck's constant in eV·s
            f = np.random.uniform(4e14, 7.5e14, num_samples)  # Visible light
            X = np.column_stack([h_ev, f])
            y = h_ev * f  # Energy in eV
            test_cases.append(
                (
                    "Photon Energy: E = h*f (visible light, eV units)",
                    X,
                    y,
                    ["h", "f"],
                    {
                        "equation_name": "photon_energy",
                        "difficulty": "easy",
                        "formula_type": "linear",
                        "ground_truth": "h * f",
                        "units": {"h": "eV*s", "f": "Hz", "E": "eV"},
                        "variable_descriptions": {
                            "h": "Planck's constant (electron-volt seconds)",
                            "f": "Photon frequency (visible spectrum)",
                        },
                        "variable_roles": {"h": "constant", "f": "varying"},
                        "quantum_fix_v22": "Use eV·s units for better numerical properties",
                        "protocol": "A",
                    },
                )
            )

            # 2. de Broglie Wavelength - FIXED: Normalized units
            h_norm = np.full(num_samples, 1.0)  # Normalized Planck's constant
            m_norm = np.full(num_samples, 1.0)  # Normalized electron mass
            v_km_s = np.random.uniform(100, 10000, num_samples)  # km/s
            X = np.column_stack([h_norm, m_norm, v_km_s])
            y = h_norm / (m_norm * v_km_s)
            test_cases.append(
                (
                    "de Broglie Wavelength: λ = h/(m*v) (normalized units)",
                    X,
                    y,
                    ["h", "m", "v"],
                    {
                        "equation_name": "de_broglie_wavelength",
                        "difficulty": "easy",
                        "formula_type": "algebraic",
                        "ground_truth": "h / (m * v)",
                        "units": {
                            "h": "normalized",
                            "m": "normalized",
                            "v": "km/s",
                            "lambda": "normalized",
                        },
                        "variable_descriptions": {
                            "h": "Planck's constant (normalized)",
                            "m": "Particle mass (normalized)",
                            "v": "Particle velocity (km/s range)",
                        },
                        "variable_roles": {
                            "h": "constant",
                            "m": "constant",
                            "v": "varying",
                        },
                        "quantum_fix_v22": "Normalized units, reasonable velocity range",
                        "protocol": "A",
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
                        "ground_truth": "(h / (me * c)) * (1 - cos_theta)",
                        "units": {
                            "h": "J*s",
                            "me": "kg",
                            "c": "m/s",
                            "cos_theta": "dimensionless",
                            "delta_lambda": "m",
                        },
                        "variable_descriptions": {
                            "h": "Planck's constant",
                            "me": "Electron mass",
                            "c": "Speed of light",
                            "cos_theta": "Cosine of scattering angle",
                        },
                        "variable_roles": {
                            "h": "constant",
                            "me": "constant",
                            "c": "constant",
                            "cos_theta": "varying",
                        },
                        "use_scaling": True,
                        "protocol": "A",
                    },
                )
            )

        # ====================================================================
        # PROTOCOL B: MULTI-DOMAIN (12 tests)
        # ====================================================================

        elif domain == "chemistry":
            # 1. Arrhenius Equation
            A = np.full(num_samples, 1e11)
            Ea = np.full(num_samples, 80000)
            R = np.full(num_samples, 8.314)
            T = np.random.uniform(273, 373, num_samples)
            X = np.column_stack([A, Ea, R, T])
            y = A * np.exp(-Ea / (R * T))
            test_cases.append(
                (
                    "Arrhenius Equation: k = A*exp(-Ea/(R*T))",
                    X,
                    y,
                    ["A", "Ea", "R", "T"],
                    {
                        "equation_name": "arrhenius_equation",
                        "difficulty": "hard",
                        "formula_type": "exponential",
                        "ground_truth": "A * np.exp(-Ea / (R * T))",
                        "units": {
                            "A": "1/s",
                            "Ea": "J/mol",
                            "R": "J/(mol*K)",
                            "T": "K",
                            "k": "1/s",
                        },
                        "variable_descriptions": {
                            "A": "Pre-exponential frequency factor",
                            "Ea": "Activation energy",
                            "R": "Universal gas constant",
                            "T": "Absolute temperature",
                        },
                        "variable_roles": {
                            "A": "constant",
                            "Ea": "constant",
                            "R": "constant",
                            "T": "varying",
                        },
                        "structure_hints": {"T": "inverse", "exponential_terms": True},
                        "use_enhanced_config": True,
                        "protocol": "B",
                    },
                )
            )

            # 2. Henderson-Hasselbalch
            pKa = np.full(num_samples, 6.5)
            A_minus = np.random.uniform(0.01, 1.0, num_samples)
            HA = np.random.uniform(0.01, 1.0, num_samples)
            X = np.column_stack([pKa, A_minus, HA])
            y = pKa + np.log10(A_minus / (HA + 1e-12))
            test_cases.append(
                (
                    "Henderson-Hasselbalch: pH = pKa + log10([A-]/[HA])",
                    X,
                    y,
                    ["pKa", "A_minus", "HA"],
                    {
                        "equation_name": "henderson_hasselbalch",
                        "difficulty": "medium",
                        "formula_type": "logarithmic",
                        "ground_truth": "pKa + np.log10(A_minus / HA)",
                        "units": {
                            "pKa": "dimensionless",
                            "A_minus": "mol/L",
                            "HA": "mol/L",
                            "pH": "dimensionless",
                        },
                        "variable_descriptions": {
                            "pKa": "Acid dissociation constant (negative log)",
                            "A_minus": "Conjugate base concentration",
                            "HA": "Weak acid concentration",
                        },
                        "variable_roles": {
                            "pKa": "constant",
                            "A_minus": "varying",
                            "HA": "varying",
                        },
                        "structure_hints": {
                            "additive_terms": True,
                            "logarithmic_ratio": True,
                        },
                        "protocol": "B",
                    },
                )
            )

            # 3. Nernst Equation - FIXED: Renamed Q to Qr to avoid PySR conflict
            E0 = np.random.uniform(0.1, 1.5, num_samples)
            R = np.full(num_samples, 8.314)
            T = np.random.uniform(273, 373, num_samples)
            n = np.random.randint(1, 3, num_samples).astype(float)
            F = np.full(num_samples, 96485)
            Qr = np.random.uniform(0.01, 100, num_samples)
            X = np.column_stack([E0, R, T, n, F, Qr])
            y = E0 - (R * T / (n * F)) * np.log(Qr)
            test_cases.append(
                (
                    "Nernst Equation: E = E0 - (RT/nF)*ln(Qr)",
                    X,
                    y,
                    ["E0", "R", "T", "n", "F", "Qr"],
                    {
                        "equation_name": "nernst_equation",
                        "difficulty": "hard",
                        "formula_type": "logarithmic",
                        "ground_truth": "E0 - (R * T / (n * F)) * np.log(Qr)",
                        "units": {
                            "E0": "V",
                            "R": "J/(mol*K)",
                            "T": "K",
                            "n": "dimensionless",
                            "F": "C/mol",
                            "Qr": "dimensionless",
                            "E": "V",
                        },
                        "variable_descriptions": {
                            "E0": "Standard electrode potential",
                            "R": "Universal gas constant",
                            "T": "Absolute temperature",
                            "n": "Number of electrons transferred",
                            "F": "Faraday constant",
                            "Qr": "Reaction quotient",
                        },
                        "variable_roles": {
                            "E0": "varying",
                            "R": "constant",
                            "T": "varying",
                            "n": "constant",
                            "F": "constant",
                            "Qr": "varying",
                        },
                        "structure_hints": {
                            "additive_terms": True,
                            "logarithmic_terms": True,
                        },
                        "pysr_fix_v22": "Renamed Q to Qr to avoid conflict with PySR built-in function",
                        "protocol": "B",
                    },
                )
            )

        elif domain == "biology":
            # 1. Michaelis-Menten
            Vmax = np.full(num_samples, 50.0)
            S = np.random.uniform(0.1, 50, num_samples)
            Km = np.full(num_samples, 10.0)
            X = np.column_stack([Vmax, S, Km])
            y = (Vmax * S) / (Km + S)
            test_cases.append(
                (
                    "Michaelis-Menten: v = (Vmax*[S])/(Km+[S])",
                    X,
                    y,
                    ["Vmax", "S", "Km"],
                    {
                        "equation_name": "michaelis_menten",
                        "difficulty": "medium",
                        "formula_type": "rational",
                        "ground_truth": "(Vmax * S) / (Km + S)",
                        "units": {
                            "Vmax": "mol/(L*s)",
                            "S": "mol/L",
                            "Km": "mol/L",
                            "v": "mol/(L*s)",
                        },
                        "variable_descriptions": {
                            "Vmax": "Maximum reaction velocity",
                            "S": "Substrate concentration",
                            "Km": "Michaelis constant (substrate affinity)",
                        },
                        "variable_roles": {
                            "Vmax": "constant",
                            "S": "varying",
                            "Km": "constant",
                        },
                        "structure_hints": {
                            "rational_form": True,
                            "saturation_curve": True,
                        },
                        "protocol": "B",
                    },
                )
            )

            # 2. Logistic Growth
            r = np.random.uniform(0.1, 0.5, num_samples)
            N = np.random.uniform(10, 900, num_samples)
            K = np.random.uniform(1000, 2000, num_samples)
            X = np.column_stack([r, N, K])
            y = r * N * (1 - N / K)
            test_cases.append(
                (
                    "Logistic Growth: dN/dt = r*N*(1-N/K)",
                    X,
                    y,
                    ["r", "N", "K"],
                    {
                        "equation_name": "logistic_growth",
                        "difficulty": "medium",
                        "formula_type": "nonlinear",
                        "ground_truth": "r * N * (1 - N / K)",
                        "units": {
                            "r": "1/s",
                            "N": "dimensionless",
                            "K": "dimensionless",
                            "dNdt": "1/s",
                        },
                        "variable_descriptions": {
                            "r": "Intrinsic growth rate",
                            "N": "Current population size",
                            "K": "Carrying capacity (maximum sustainable population)",
                        },
                        "variable_roles": {
                            "r": "constant",
                            "N": "varying",
                            "K": "constant",
                        },
                        "structure_hints": {
                            "multiplicative_terms": True,
                            "subtraction_in_factor": True,
                        },
                        "protocol": "B",
                    },
                )
            )

            # 3. Allometric Scaling
            a = np.full(num_samples, 3.5)
            M = np.random.uniform(0.1, 100, num_samples)
            b = np.full(num_samples, 0.75)
            X = np.column_stack([a, M, b])
            y = a * M**b
            test_cases.append(
                (
                    "Allometric Scaling: Y = a*M^b",
                    X,
                    y,
                    ["a", "M", "b"],
                    {
                        "equation_name": "allometric_scaling",
                        "difficulty": "easy",
                        "formula_type": "power_law",
                        "ground_truth": "a * M**b",
                        "units": {
                            "a": "W/kg^0.75",
                            "M": "kg",
                            "b": "dimensionless",
                            "Y": "W",
                        },
                        "variable_descriptions": {
                            "a": "Allometric coefficient",
                            "M": "Body mass",
                            "b": "Scaling exponent (typically 0.75)",
                        },
                        "variable_roles": {
                            "a": "constant",
                            "M": "varying",
                            "b": "constant",
                        },
                        "structure_hints": {"M": "power_law"},
                        "protocol": "B",
                    },
                )
            )

        elif domain == "mathematics":
            # 1. Pythagorean Theorem
            a = np.random.uniform(1, 10, num_samples)
            b = np.random.uniform(1, 10, num_samples)
            X = np.column_stack([a, b])
            y = np.sqrt(a**2 + b**2)
            test_cases.append(
                (
                    "Pythagorean Theorem: c = sqrt(a² + b²)",
                    X,
                    y,
                    ["a", "b"],
                    {
                        "equation_name": "pythagorean_theorem",
                        "difficulty": "easy",
                        "formula_type": "power_law",
                        "ground_truth": "np.sqrt(a**2 + b**2)",
                        "units": {"a": "m", "b": "m", "c": "m"},
                        "variable_descriptions": {
                            "a": "First perpendicular side of right triangle",
                            "b": "Second perpendicular side of right triangle",
                        },
                        "variable_roles": {"a": "varying", "b": "varying"},
                        "structure_hints": {
                            "sqrt_of_sum": True,
                            "quadratic_terms": True,
                        },
                        "use_enhanced_config": True,
                        "protocol": "B",
                    },
                )
            )

            # 2. Compound Interest
            P = np.random.uniform(1000, 10000, num_samples)
            r = np.random.uniform(0.01, 0.1, num_samples)
            n = np.random.choice([1, 4, 12], num_samples).astype(float)
            t = np.random.uniform(1, 20, num_samples)
            X = np.column_stack([P, r, n, t])
            y = P * (1 + r / n) ** (n * t)
            test_cases.append(
                (
                    "Compound Interest: A = P*(1+r/n)^(n*t)",
                    X,
                    y,
                    ["P", "r", "n", "t"],
                    {
                        "equation_name": "compound_interest",
                        "difficulty": "medium",
                        "formula_type": "exponential",
                        "ground_truth": "P * (1 + r/n)**(n*t)",
                        "units": {
                            "P": "USD",
                            "r": "1/year",
                            "n": "1/year",
                            "t": "year",
                            "A": "USD",
                        },
                        "variable_descriptions": {
                            "P": "Principal amount (initial investment)",
                            "r": "Annual interest rate (as decimal)",
                            "n": "Compounding frequency per year",
                            "t": "Time period in years",
                        },
                        "variable_roles": {
                            "P": "varying",
                            "r": "varying",
                            "n": "varying",
                            "t": "varying",
                        },
                        "structure_hints": {
                            "exponential_growth": True,
                            "compound_exponent": True,
                        },
                        "use_enhanced_config": True,
                        "protocol": "B",
                    },
                )
            )

            # 3. Quadratic Discriminant
            a = np.random.uniform(-5, 5, num_samples)
            a[np.abs(a) < 0.1] = 1.0
            b = np.random.uniform(-10, 10, num_samples)
            c = np.random.uniform(-5, 5, num_samples)
            X = np.column_stack([a, b, c])
            y = b**2 - 4 * a * c
            test_cases.append(
                (
                    "Quadratic Discriminant: Δ = b² - 4ac",
                    X,
                    y,
                    ["a", "b", "c"],
                    {
                        "equation_name": "quadratic_discriminant",
                        "difficulty": "easy",
                        "formula_type": "polynomial",
                        "ground_truth": "b**2 - 4*a*c",
                        "units": {
                            "a": "dimensionless",
                            "b": "dimensionless",
                            "c": "dimensionless",
                            "delta": "dimensionless",
                        },
                        "variable_descriptions": {
                            "a": "Quadratic coefficient (ax²)",
                            "b": "Linear coefficient (bx)",
                            "c": "Constant term",
                        },
                        "variable_roles": {
                            "a": "varying",
                            "b": "varying",
                            "c": "varying",
                        },
                        "structure_hints": {
                            "b": "quadratic",
                            "subtraction_terms": True,
                        },
                        "protocol": "B",
                    },
                )
            )

        elif domain == "economics":
            # 1. Price Elasticity
            Q = np.random.uniform(100, 1000, num_samples)
            delta_Q = np.random.uniform(-50, 50, num_samples)
            P = np.random.uniform(10, 100, num_samples)
            delta_P = np.random.uniform(-5, 5, num_samples)
            delta_P[np.abs(delta_P) < 0.1] = 0.1
            X = np.column_stack([Q, delta_Q, P, delta_P])
            y = (delta_Q / (Q + 1e-10)) / ((delta_P / (P + 1e-10)) + 1e-10)
            test_cases.append(
                (
                    "Price Elasticity: Ed = (ΔQ/Q)/(ΔP/P)",
                    X,
                    y,
                    ["Q", "delta_Q", "P", "delta_P"],
                    {
                        "equation_name": "elasticity_demand",
                        "difficulty": "medium",
                        "formula_type": "rational",
                        "ground_truth": "(delta_Q / Q) / (delta_P / P)",
                        "units": {
                            "Q": "dimensionless",
                            "delta_Q": "dimensionless",
                            "P": "dimensionless",
                            "delta_P": "dimensionless",
                            "Ed": "dimensionless",
                        },
                        "variable_descriptions": {
                            "Q": "Initial quantity demanded",
                            "delta_Q": "Change in quantity demanded",
                            "P": "Initial price",
                            "delta_P": "Change in price",
                        },
                        "variable_roles": {
                            "Q": "varying",
                            "delta_Q": "varying",
                            "P": "varying",
                            "delta_P": "varying",
                        },
                        "structure_hints": {
                            "double_ratio": True,
                            "division_terms": True,
                        },
                        "protocol": "B",
                    },
                )
            )

            # 2. Cobb-Douglas Production Function
            A = np.random.uniform(1, 5, num_samples)
            K = np.random.uniform(100, 1000, num_samples)
            L = np.random.uniform(10, 100, num_samples)
            alpha = np.full(num_samples, 0.3)
            beta = np.full(num_samples, 0.7)
            X = np.column_stack([A, K, L, alpha, beta])
            y = A * K**alpha * L**beta
            test_cases.append(
                (
                    "Cobb-Douglas: Y = A*K^α*L^β",
                    X,
                    y,
                    ["A", "K", "L", "alpha", "beta"],
                    {
                        "equation_name": "cobb_douglas",
                        "difficulty": "medium",
                        "formula_type": "power_law",
                        "ground_truth": "A * K**alpha * L**beta",
                        "units": {
                            "A": "dimensionless",
                            "K": "dimensionless",
                            "L": "dimensionless",
                            "alpha": "dimensionless",
                            "beta": "dimensionless",
                            "Y": "dimensionless",
                        },
                        "variable_descriptions": {
                            "A": "Total factor productivity",
                            "K": "Capital input",
                            "L": "Labor input",
                            "alpha": "Output elasticity of capital",
                            "beta": "Output elasticity of labor",
                        },
                        "variable_roles": {
                            "A": "varying",
                            "K": "varying",
                            "L": "varying",
                            "alpha": "constant",
                            "beta": "constant",
                        },
                        "structure_hints": {
                            "K": "power_law",
                            "L": "power_law",
                            "multiplicative_terms": True,
                        },
                        "protocol": "B",
                    },
                )
            )

            # 3. Break-Even Point
            FC = np.random.uniform(10000, 100000, num_samples)
            P = np.random.uniform(50, 200, num_samples)
            VC = np.random.uniform(20, 100, num_samples)
            X = np.column_stack([FC, P, VC])
            y = FC / (P - VC + 1e-10)
            test_cases.append(
                (
                    "Break-Even Point: BEP = FC/(P-VC)",
                    X,
                    y,
                    ["FC", "P", "VC"],
                    {
                        "equation_name": "break_even_point",
                        "difficulty": "easy",
                        "formula_type": "algebraic",
                        "ground_truth": "FC / (P - VC)",
                        "units": {"FC": "USD", "P": "USD", "VC": "USD", "BEP": "units"},
                        "variable_descriptions": {
                            "FC": "Fixed costs",
                            "P": "Price per unit",
                            "VC": "Variable cost per unit",
                        },
                        "variable_roles": {
                            "FC": "varying",
                            "P": "varying",
                            "VC": "varying",
                        },
                        "protocol": "B",
                    },
                )
            )

        return test_cases

    @staticmethod
    def get_domain_description(domain: str) -> str:
        """Get domain description."""
        descriptions = {
            # Protocol A
            "mechanics": "Classical Mechanics - kinematics, dynamics, energy",
            "thermodynamics": "Thermodynamics - heat, temperature, efficiency",
            "electromagnetism": "Electromagnetism - forces, circuits, fields",
            "fluid_dynamics": "Fluid Dynamics - flow, pressure, viscosity",
            "optics": "Optics - light, refraction, diffraction",
            "quantum": "Quantum Mechanics - photons, waves, particles (v2.2 FIXED)",
            # Protocol B
            "chemistry": "Chemistry - kinetics, equilibrium, electrochemistry",
            "biology": "Biology - enzyme kinetics, population dynamics, allometry",
            "mathematics": "Mathematics - geometry, finance, algebra",
            "economics": "Economics - elasticity, production functions, break-even",
        }
        return descriptions.get(domain, "Unknown domain")

    @staticmethod
    def get_protocol_statistics() -> Dict:
        """Get comprehensive protocol statistics."""
        return {
            "version": "2.2",
            "total_tests": 30,
            "improvements": {
                "from_v20": "Complete implementation with all metadata",
                "from_v21": "Quantum fixes with normalized/eV units",
            },
            "protocol_breakdown": {
                "A": {"tests": 18, "focus": "Physics & Engineering"},
                "B": {"tests": 12, "focus": "Multi-Domain Sciences"},
            },
            "domains": {
                "mechanics": 3,
                "thermodynamics": 3,
                "electromagnetism": 3,
                "fluid_dynamics": 3,
                "optics": 3,
                "quantum": 3,
                "chemistry": 3,
                "biology": 3,
                "mathematics": 3,
                "economics": 3,
            },
            "difficulty": {"easy": 15, "medium": 10, "hard": 5},
        }

    @staticmethod
    def save_protocol_documentation(
        filepath: str = "docs/experiment_protocol_all_30_v2.2.json",
    ):
        """Save complete protocol documentation."""
        protocol_doc = {
            "title": "Experiment Protocol ALL v2.2: Best of v2.0 + v2.1",
            "version": "2.2 COMPLETE",
            "date": "2026-01-13",
            "total_tests": 30,
            "improvements": {
                "v20": "Complete implementation, full metadata, structure hints",
                "v21": "Quantum tests with better numerical properties",
            },
            "protocols": {
                "A": {
                    "name": "Physics & Engineering",
                    "tests": 18,
                    "domains": [
                        "mechanics",
                        "thermodynamics",
                        "electromagnetism",
                        "fluid_dynamics",
                        "optics",
                        "quantum",
                    ],
                },
                "B": {
                    "name": "Multi-Domain Sciences",
                    "tests": 12,
                    "domains": ["chemistry", "biology", "mathematics", "economics"],
                },
            },
            "domains": {},
        }

        for domain in ExperimentProtocolAll.get_all_domains():
            test_cases = ExperimentProtocolAll.load_test_data(domain, num_samples=10)
            if test_cases:
                protocol_doc["domains"][domain] = {
                    "description": ExperimentProtocolAll.get_domain_description(domain),
                    "num_test_cases": len(test_cases),
                    "test_cases": [
                        {
                            "description": desc,
                            "variables": vars,
                            "equation_name": meta.get("equation_name"),
                            "difficulty": meta["difficulty"],
                            "ground_truth": meta["ground_truth"],
                            "protocol": meta["protocol"],
                            "variable_descriptions": meta.get(
                                "variable_descriptions", {}
                            ),
                            "use_enhanced_config": meta.get(
                                "use_enhanced_config", False
                            ),
                        }
                        for desc, _, _, vars, meta in test_cases
                    ],
                }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(protocol_doc, f, indent=2)

        print(f"✅ Protocol ALL v2.2 documentation saved to: {filepath}")
        return protocol_doc


if __name__ == "__main__":
    protocol = ExperimentProtocolAll()

    print("=" * 80)
    print("EXPERIMENT PROTOCOL ALL v2.2: COMPLETE (BEST OF v2.0 + v2.1)".center(80))
    print("=" * 80)
    print(f"Version: 2.2 | Date: 2026-01-13")
    print("=" * 80)

    total_count = 0
    difficulty_count = {"easy": 0, "medium": 0, "hard": 0}

    for domain in protocol.get_all_domains():
        test_cases = protocol.load_test_data(domain, num_samples=10)
        if test_cases:
            print(f"\n{domain.upper()} ({len(test_cases)} tests):")
            for i, (desc, _, _, vars, meta) in enumerate(test_cases, 1):
                protocol_label = meta.get("protocol", "?")
                difficulty = meta["difficulty"]
                difficulty_count[difficulty] += 1
                enhanced = "🚀" if meta.get("use_enhanced_config") else "  "
                quantum_fix = "⚛️" if "quantum_fix_v22" in meta else "  "
                print(f"  [{protocol_label}] {enhanced}{quantum_fix} {desc}")
                print(f"      Equation: {meta['equation_name']}")
                print(f"      Variables: {', '.join(vars)}")
                print(f"      Difficulty: {difficulty} | Type: {meta['formula_type']}")
            total_count += len(test_cases)

    print(f"\n{'=' * 80}")
    print(f"SUMMARY".center(80))
    print(f"{'=' * 80}")
    print(f"Total test cases: {total_count}")
    print(f"Protocol A (Physics/Engineering): 18 tests")
    print(f"Protocol B (Multi-Domain): 12 tests")
    print(f"\nImprovements in v2.2:")
    print(f"  ✅ Complete implementation from v2.0 (all domains fully coded)")
    print(f"  ✅ Quantum fixes from v2.1 (normalized/eV units)")
    print(f"  ✅ All metadata and structure hints preserved")
    print(f"\nDifficulty distribution:")
    for diff, count in difficulty_count.items():
        print(f"  - {diff.capitalize()}: {count} tests")
    print(f"\nDomains: {len(protocol.get_all_domains())}")
    print(f"{'=' * 80}")

    protocol.save_protocol_documentation()

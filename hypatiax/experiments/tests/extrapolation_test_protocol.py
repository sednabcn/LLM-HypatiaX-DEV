#!/usr/bin/env python3
"""
Extrapolation Test Protocol
============================

Systematic testing of extrapolation capabilities for symbolic discovery methods.
Generates training data in one range, tests on multiple extrapolation regimes.

Usage:
    python extrapolation_test_protocol.py --method hybrid --test arrhenius
    python extrapolation_test_protocol.py --method all --domain chemistry --plot
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class ExtrapolationRegime:
    """Define an extrapolation test regime"""

    name: str
    multiplier: float  # How far beyond training max
    description: str


# Standard extrapolation regimes
REGIMES = [
    ExtrapolationRegime("near", 1.2, "Near extrapolation (20% beyond training)"),
    ExtrapolationRegime("medium", 2.0, "Medium extrapolation (2x training range)"),
    ExtrapolationRegime("far", 5.0, "Far extrapolation (5x training range)"),
]


class ExtrapolationTestProtocol:
    """Protocol for systematic extrapolation testing"""

    def __init__(self, base_range: Tuple[float, float] = (0.1, 1.0)):
        """
        Initialize protocol

        Args:
            base_range: (min, max) for training data
        """
        self.x_min, self.x_max = base_range
        self.x_range = self.x_max - self.x_min

    def generate_training_data(
        self,
        ground_truth: Callable,
        n_samples: int = 200,
        noise_level: float = 0.05,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data within base range

        Args:
            ground_truth: Function to generate y = f(x)
            n_samples: Number of training samples
            noise_level: Gaussian noise std as fraction of y range
            seed: Random seed

        Returns:
            X_train, y_train
        """
        np.random.seed(seed)

        # Sample uniformly in training range
        X = np.random.uniform(self.x_min, self.x_max, n_samples)

        # Generate ground truth
        y_true = ground_truth(X)

        # Add noise
        noise_std = noise_level * (np.max(y_true) - np.min(y_true))
        y = y_true + np.random.normal(0, noise_std, n_samples)

        return X.reshape(-1, 1), y

    def generate_extrapolation_data(
        self,
        ground_truth: Callable,
        regime: ExtrapolationRegime,
        n_samples: int = 100,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate test data in extrapolation regime

        Args:
            ground_truth: Function to generate y = f(x)
            regime: Which extrapolation regime to test
            n_samples: Number of test samples
            seed: Random seed

        Returns:
            X_test, y_test
        """
        np.random.seed(seed + 1000)  # Different seed than training

        # Define extrapolation range
        extrap_min = self.x_max * regime.multiplier
        extrap_max = self.x_max * regime.multiplier * 1.5

        # Sample uniformly in extrapolation range
        X = np.random.uniform(extrap_min, extrap_max, n_samples)

        # Generate ground truth (no noise for fair comparison)
        y = ground_truth(X)

        return X.reshape(-1, 1), y

    def calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate RMSE, handling non-finite predictions"""
        if not np.all(np.isfinite(y_pred)):
            return float("inf")
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    def calculate_extrapolation_error(
        self, rmse_train: float, rmse_extrap: float
    ) -> float:
        """
        Calculate extrapolation error percentage

        Formula: (RMSE_extrap / RMSE_train) × 100%

        Returns:
            Extrapolation error percentage
        """
        if rmse_train == 0:
            return float("inf") if rmse_extrap > 0 else 0.0
        return (rmse_extrap / rmse_train) * 100.0

    def test_method_extrapolation(
        self,
        method_name: str,
        predict_fn: Callable,
        ground_truth: Callable,
        n_train: int = 200,
        n_test: int = 100,
        noise_level: float = 0.05,
        verbose: bool = True,
    ) -> Dict:
        """
        Test a method's extrapolation capability

        Args:
            method_name: Name of the method being tested
            predict_fn: Function that takes X and returns predictions
            ground_truth: True function for data generation
            n_train: Number of training samples
            n_test: Number of test samples per regime
            noise_level: Training noise level
            verbose: Print progress

        Returns:
            Dictionary with results for each regime
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"Testing {method_name}")
            print(f"{'='*80}")
            print(f"Training range: [{self.x_min:.2f}, {self.x_max:.2f}]")

        # Generate training data
        X_train, y_train = self.generate_training_data(
            ground_truth, n_train, noise_level
        )

        # Get training predictions
        y_pred_train = predict_fn(X_train)
        rmse_train = self.calculate_rmse(y_train, y_pred_train)

        if verbose:
            print(f"\n📊 Training Performance:")
            print(f"   RMSE: {rmse_train:.6f}")

        # Test each extrapolation regime
        results = {
            "method": method_name,
            "training": {
                "rmse": rmse_train,
                "range": [self.x_min, self.x_max],
                "n_samples": n_train,
            },
            "regimes": {},
        }

        for regime in REGIMES:
            # Generate extrapolation test data
            X_test, y_test = self.generate_extrapolation_data(
                ground_truth, regime, n_test
            )

            # Get predictions
            y_pred_test = predict_fn(X_test)
            rmse_test = self.calculate_rmse(y_test, y_pred_test)

            # Calculate extrapolation error
            extrap_error = self.calculate_extrapolation_error(rmse_train, rmse_test)

            results["regimes"][regime.name] = {
                "description": regime.description,
                "multiplier": regime.multiplier,
                "rmse": rmse_test,
                "extrapolation_error_pct": extrap_error,
                "range": [
                    self.x_max * regime.multiplier,
                    self.x_max * regime.multiplier * 1.5,
                ],
                "n_samples": n_test,
            }

            if verbose:
                print(f"\n🔬 {regime.description}:")
                print(
                    f"   Range: [{self.x_max * regime.multiplier:.2f}, "
                    f"{self.x_max * regime.multiplier * 1.5:.2f}]"
                )
                print(f"   RMSE: {rmse_test:.6f}")
                print(f"   Extrapolation Error: {extrap_error:.1f}%")

                # Categorize performance
                if extrap_error < 50:
                    status = "✅ EXCELLENT"
                elif extrap_error < 100:
                    status = "✓ GOOD"
                elif extrap_error < 200:
                    status = "⚠️  MODERATE"
                elif extrap_error < 500:
                    status = "❌ POOR"
                else:
                    status = "💥 CATASTROPHIC"
                print(f"   Status: {status}")

        return results

    def plot_extrapolation_results(
        self,
        results_list: List[Dict],
        ground_truth: Callable,
        save_path: Optional[Path] = None,
    ):
        """
        Plot extrapolation performance comparison

        Args:
            results_list: List of result dicts from test_method_extrapolation
            ground_truth: True function for plotting
            save_path: Where to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Training + Extrapolation ranges
        ax1 = axes[0, 0]
        x_full = np.linspace(self.x_min, self.x_max * 10, 1000)
        y_full = ground_truth(x_full)

        ax1.plot(x_full, y_full, "k--", label="Ground Truth", linewidth=2)
        ax1.axvspan(
            self.x_min, self.x_max, alpha=0.2, color="green", label="Training Range"
        )

        for regime in REGIMES:
            r_min = self.x_max * regime.multiplier
            r_max = self.x_max * regime.multiplier * 1.5
            ax1.axvspan(
                r_min, r_max, alpha=0.1, label=f"{regime.name.capitalize()} Extrap."
            )

        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_title("Extrapolation Test Regimes")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: RMSE by regime
        ax2 = axes[0, 1]
        regime_names = [r.name for r in REGIMES]
        x_pos = np.arange(len(regime_names))
        width = 0.8 / len(results_list)

        for i, results in enumerate(results_list):
            rmses = [results["regimes"][r]["rmse"] for r in regime_names]
            ax2.bar(x_pos + i * width, rmses, width, label=results["method"], alpha=0.8)

        ax2.set_xlabel("Extrapolation Regime")
        ax2.set_ylabel("RMSE")
        ax2.set_title("RMSE by Extrapolation Distance")
        ax2.set_xticks(x_pos + width * (len(results_list) - 1) / 2)
        ax2.set_xticklabels([r.capitalize() for r in regime_names])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.set_yscale("log")

        # Plot 3: Extrapolation Error %
        ax3 = axes[1, 0]
        for i, results in enumerate(results_list):
            errors = [
                results["regimes"][r]["extrapolation_error_pct"] for r in regime_names
            ]
            ax3.bar(
                x_pos + i * width, errors, width, label=results["method"], alpha=0.8
            )

        ax3.set_xlabel("Extrapolation Regime")
        ax3.set_ylabel("Extrapolation Error (%)")
        ax3.set_title("Extrapolation Error by Distance")
        ax3.set_xticks(x_pos + width * (len(results_list) - 1) / 2)
        ax3.set_xticklabels([r.capitalize() for r in regime_names])
        ax3.axhline(
            y=100, color="r", linestyle="--", alpha=0.5, label="100% (training error)"
        )
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis="y")
        ax3.set_yscale("log")

        # Plot 4: Performance degradation
        ax4 = axes[1, 1]
        multipliers = [r.multiplier for r in REGIMES]

        for results in results_list:
            errors = [
                results["regimes"][r]["extrapolation_error_pct"] for r in regime_names
            ]
            ax4.plot(
                multipliers, errors, marker="o", linewidth=2, label=results["method"]
            )

        ax4.set_xlabel("Distance from Training (×)")
        ax4.set_ylabel("Extrapolation Error (%)")
        ax4.set_title("Error Growth with Extrapolation Distance")
        ax4.axhline(y=100, color="r", linestyle="--", alpha=0.5)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale("log")
        ax4.set_xscale("log")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"\n💾 Plot saved to: {save_path}")
        else:
            plt.show()


# ============================================================================
# EXAMPLE TEST FUNCTIONS
# ============================================================================


def test_arrhenius_extrapolation():
    """Example: Test Arrhenius equation extrapolation"""

    # Ground truth: k = A * exp(-Ea / (R*T))
    A = 1e11
    Ea = 80000
    R = 8.314

    def arrhenius(T):
        return A * np.exp(-Ea / (R * T))

    # Initialize protocol
    protocol = ExtrapolationTestProtocol(base_range=(273, 373))  # 273-373K training

    # Example method 1: Pure LLM (simulated - assumes wrong functional form)
    def llm_predict(X):
        # Simulate LLM producing log instead of exp
        T = X.flatten()
        return A * np.log(T / 273 + 1)  # WRONG FUNCTIONAL FORM

    # Example method 2: Correct formula
    def hybrid_predict(X):
        T = X.flatten()
        return arrhenius(T)

    # Test both methods
    results_llm = protocol.test_method_extrapolation(
        "Pure LLM (Wrong Form)", llm_predict, arrhenius, verbose=True
    )

    results_hybrid = protocol.test_method_extrapolation(
        "Hybrid (Correct Form)", hybrid_predict, arrhenius, verbose=True
    )

    # Plot comparison
    protocol.plot_extrapolation_results(
        [results_llm, results_hybrid],
        arrhenius,
        save_path=Path("arrhenius_extrapolation.png"),
    )

    # Save results
    with open("arrhenius_extrapolation_results.json", "w") as f:
        json.dump({"llm": results_llm, "hybrid": results_hybrid}, f, indent=2)

    return results_llm, results_hybrid


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extrapolation Test Protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--test",
        type=str,
        default="arrhenius",
        help="Test case (arrhenius, hall_petch, etc.)",
    )
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--save", type=str, help="Save results to file")

    args = parser.parse_args()

    if args.test == "arrhenius":
        results_llm, results_hybrid = test_arrhenius_extrapolation()

        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(
            f"\nPure LLM Medium Extrapolation Error: "
            f"{results_llm['regimes']['medium']['extrapolation_error_pct']:.1f}%"
        )
        print(
            f"Hybrid Medium Extrapolation Error: "
            f"{results_hybrid['regimes']['medium']['extrapolation_error_pct']:.1f}%"
        )
        print(f"\n{'='*80}\n")

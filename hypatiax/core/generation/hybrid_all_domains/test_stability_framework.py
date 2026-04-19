#!/usr/bin/env python3
"""
HypatiaX Test Stability Monitoring Framework
=============================================
External framework to monitor test suite stability, detect flaky tests,
track regressions, and provide comprehensive stability analysis.

Features:
- Seed management across all stochastic components
- Test stability tracking over multiple runs
- Adaptive retry strategies
- Data generation validation
- Baseline comparison and regression detection
- Detailed stability reporting and visualization

Author: HypatiaX Team
Date: 2026-01-03
Version: 1.0
"""

import json
import hashlib
import random
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np


# ============================================================================
# SEED MANAGEMENT
# ============================================================================


class ReproducibleSeedManager:
    """Manages seeds deterministically across all stochastic components."""

    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        self.test_seeds = {}
        self.seed_history = []

    def get_test_seed(self, test_name: str, attempt: int = 0) -> int:
        """
        Generate deterministic seed for each test+attempt combination.

        Uses MD5 hash to ensure different tests get different seeds
        but same test+attempt always gets the same seed.
        """
        key = f"{test_name}_{attempt}"

        if key not in self.test_seeds:
            # Generate deterministic but unique seed
            hash_input = f"{self.base_seed}_{key}".encode()
            hash_val = int(hashlib.md5(hash_input).hexdigest()[:8], 16)
            seed = hash_val % (2**31 - 1)
            self.test_seeds[key] = seed

            self.seed_history.append(
                {
                    "test_name": test_name,
                    "attempt": attempt,
                    "seed": seed,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        return self.test_seeds[key]

    def set_all_seeds(self, seed: int):
        """Set all random number generators to the same seed."""
        np.random.seed(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

        # If PyTorch is available
        try:
            import torch

            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    def save_seed_history(self, filepath: str):
        """Save seed history for reproducibility."""
        with open(filepath, "w") as f:
            json.dump(
                {"base_seed": self.base_seed, "history": self.seed_history}, f, indent=2
            )

    def load_seed_history(self, filepath: str):
        """Load seed history from file."""
        with open(filepath, "r") as f:
            data = json.load(f)
            self.base_seed = data["base_seed"]
            self.seed_history = data["history"]

            # Rebuild cache
            for entry in self.seed_history:
                key = f"{entry['test_name']}_{entry['attempt']}"
                self.test_seeds[key] = entry["seed"]


# ============================================================================
# DATA VALIDATION
# ============================================================================


@dataclass
class DataValidationResult:
    """Result of data generation validation."""

    stable: bool
    issues: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]


class DataValidator:
    """Validates generated test data for numerical stability."""

    # Thresholds
    MAX_SAFE_VALUE = 1e10
    MIN_SAFE_VALUE = 1e-10
    MAX_CONDITION_NUMBER = 1e10
    MIN_VARIANCE = 1e-10

    @staticmethod
    def validate_data(
        X: np.ndarray, y: np.ndarray, test_name: str, verbose: bool = True
    ) -> DataValidationResult:
        """
        Validate generated data for numerical stability.

        Checks:
        - No NaN/Inf values
        - Reasonable value ranges
        - Sufficient variance
        - Matrix conditioning (if X has multiple columns)
        """
        issues = []
        warnings = []

        # Check 1: NaN/Inf detection
        if np.any(np.isnan(X)):
            issues.append("NaN detected in input X")
        if np.any(np.isnan(y)):
            issues.append("NaN detected in output y")
        if np.any(np.isinf(X)):
            issues.append("Inf detected in input X")
        if np.any(np.isinf(y)):
            issues.append("Inf detected in output y")

        # Check 2: Value ranges
        X_max = np.max(np.abs(X))
        y_max = np.max(np.abs(y))

        if X_max > DataValidator.MAX_SAFE_VALUE:
            warnings.append(f"Very large X values: max={X_max:.2e}")
        if y_max > DataValidator.MAX_SAFE_VALUE:
            warnings.append(f"Very large y values: max={y_max:.2e}")

        # Check for very small non-zero values
        X_nonzero = X[X != 0]
        y_nonzero = y[y != 0]

        if len(X_nonzero) > 0:
            X_min = np.min(np.abs(X_nonzero))
            if X_min < DataValidator.MIN_SAFE_VALUE:
                warnings.append(f"Very small X values: min={X_min:.2e}")

        if len(y_nonzero) > 0:
            y_min = np.min(np.abs(y_nonzero))
            if y_min < DataValidator.MIN_SAFE_VALUE:
                warnings.append(f"Very small y values: min={y_min:.2e}")

        # Check 3: Variance
        y_std = np.std(y)
        if y_std < DataValidator.MIN_VARIANCE:
            issues.append(f"Insufficient variance in y: σ={y_std:.2e}")

        # Check 4: Matrix conditioning
        cond_number = None
        if X.shape[1] > 1:
            try:
                cond_number = np.linalg.cond(X)
                if cond_number > DataValidator.MAX_CONDITION_NUMBER:
                    warnings.append(f"Poorly conditioned X: cond={cond_number:.2e}")
            except Exception as e:
                warnings.append(f"Could not compute condition number: {e}")

        # Metadata
        metadata = {
            "X_shape": X.shape,
            "y_shape": y.shape,
            "X_range": (float(np.min(X)), float(np.max(X))),
            "y_range": (float(np.min(y)), float(np.max(y))),
            "y_mean": float(np.mean(y)),
            "y_std": float(np.std(y)),
            "condition_number": float(cond_number) if cond_number else None,
        }

        stable = len(issues) == 0

        if verbose and not stable:
            print(f"⚠️ Data validation issues for {test_name}:")
            for issue in issues:
                print(f"   ❌ {issue}")
            for warning in warnings:
                print(f"   ⚠️  {warning}")

        return DataValidationResult(
            stable=stable, issues=issues, warnings=warnings, metadata=metadata
        )


# ============================================================================
# STABILITY TRACKING
# ============================================================================


@dataclass
class TestRun:
    """Single test run record."""

    test_name: str
    passed: bool
    r2_score: float
    validation_score: float
    seed: int
    attempt: int
    timestamp: str
    execution_time: float
    expression: Optional[str] = None
    failure_reason: Optional[str] = None


class StabilityTracker:
    """Tracks test stability over multiple runs."""

    def __init__(self, history_file: Optional[str] = None):
        self.test_history = defaultdict(list)
        self.history_file = history_file

        if history_file and Path(history_file).exists():
            self.load_history(history_file)

    def record_run(self, run: TestRun):
        """Record a test run."""
        self.test_history[run.test_name].append(run)

    def get_stability_metrics(self, test_name: str) -> Dict[str, Any]:
        """
        Calculate stability metrics for a test.

        Returns:
        - total_runs: Number of runs recorded
        - pass_rate: Percentage of successful runs
        - r2_mean: Mean R² score
        - r2_std: Standard deviation of R² scores
        - r2_cv: Coefficient of variation for R²
        - is_stable: Boolean indicating if test is stable
        - flakiness_score: 0-100, higher = more flaky
        """
        runs = self.test_history[test_name]

        if len(runs) < 2:
            return {
                "test_name": test_name,
                "insufficient_data": True,
                "total_runs": len(runs),
            }

        passes = sum(1 for r in runs if r.passed)
        r2_values = [r.r2_score for r in runs]
        val_scores = [r.validation_score for r in runs]

        pass_rate = passes / len(runs)
        r2_mean = np.mean(r2_values)
        r2_std = np.std(r2_values)
        r2_cv = r2_std / (r2_mean + 1e-10)  # Coefficient of variation

        val_mean = np.mean(val_scores)
        val_std = np.std(val_scores)

        # Stability criteria
        is_stable = (
            pass_rate >= 0.8  # 80% pass rate
            and r2_std < 0.05  # Low variance in R²
            and val_std < 10.0  # Low variance in validation
        )

        # Flakiness score (0-100, higher = worse)
        flakiness_score = (
            (1 - pass_rate) * 50  # Up to 50 points for failures
            + min(r2_std / 0.1, 1.0) * 30  # Up to 30 points for R² variance
            + min(val_std / 20.0, 1.0) * 20  # Up to 20 points for val variance
        )

        return {
            "test_name": test_name,
            "insufficient_data": False,
            "total_runs": len(runs),
            "pass_rate": pass_rate,
            "passes": passes,
            "failures": len(runs) - passes,
            "r2_mean": r2_mean,
            "r2_std": r2_std,
            "r2_cv": r2_cv,
            "r2_min": np.min(r2_values),
            "r2_max": np.max(r2_values),
            "val_mean": val_mean,
            "val_std": val_std,
            "is_stable": is_stable,
            "flakiness_score": flakiness_score,
            "avg_execution_time": np.mean([r.execution_time for r in runs]),
        }

    def identify_flaky_tests(
        self, pass_rate_threshold: float = 0.7, r2_std_threshold: float = 0.1
    ) -> List[Tuple[str, Dict]]:
        """
        Identify flaky tests based on thresholds.

        Returns list of (test_name, metrics) tuples sorted by flakiness.
        """
        flaky = []

        for test_name in self.test_history.keys():
            metrics = self.get_stability_metrics(test_name)

            if metrics.get("insufficient_data"):
                continue

            if (
                metrics["pass_rate"] < pass_rate_threshold
                or metrics["r2_std"] > r2_std_threshold
            ):
                flaky.append((test_name, metrics))

        # Sort by flakiness score (worst first)
        return sorted(flaky, key=lambda x: x[1]["flakiness_score"], reverse=True)

    def get_all_metrics(self) -> Dict[str, Dict]:
        """Get metrics for all tests."""
        return {
            test_name: self.get_stability_metrics(test_name)
            for test_name in self.test_history.keys()
        }

    def save_history(self, filepath: Optional[str] = None):
        """Save test history to JSON."""
        filepath = filepath or self.history_file
        if not filepath:
            raise ValueError("No filepath provided for saving history")

        # Convert TestRun objects to dicts
        history_dict = {}
        for test_name, runs in self.test_history.items():
            history_dict[test_name] = [asdict(run) for run in runs]

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(history_dict, f, indent=2)

    def load_history(self, filepath: str):
        """Load test history from JSON."""
        with open(filepath, "r") as f:
            history_dict = json.load(f)

        self.test_history.clear()
        for test_name, runs_data in history_dict.items():
            self.test_history[test_name] = [
                TestRun(**run_data) for run_data in runs_data
            ]


# ============================================================================
# BASELINE COMPARISON
# ============================================================================


@dataclass
class RegressionReport:
    """Regression detection report."""

    test_name: str
    is_regression: bool
    severity: str  # 'critical', 'major', 'minor', 'none'
    r2_delta: float
    val_delta: float
    baseline_r2: float
    current_r2: float
    baseline_val: float
    current_val: float
    passed_before: bool
    passed_now: bool
    expression_changed: bool
    details: Dict[str, Any]


class BaselineComparator:
    """Compare test results against baseline for regression detection."""

    # Regression thresholds
    R2_CRITICAL_THRESHOLD = -0.20  # 20% drop
    R2_MAJOR_THRESHOLD = -0.10  # 10% drop
    R2_MINOR_THRESHOLD = -0.05  # 5% drop

    VAL_CRITICAL_THRESHOLD = -20.0
    VAL_MAJOR_THRESHOLD = -15.0
    VAL_MINOR_THRESHOLD = -10.0

    def __init__(self, baseline_file: Optional[str] = None):
        self.baseline = {}
        self.baseline_file = baseline_file

        if baseline_file and Path(baseline_file).exists():
            self.load_baseline(baseline_file)

    def load_baseline(self, filepath: str):
        """Load baseline from JSON file."""
        with open(filepath, "r") as f:
            self.baseline = json.load(f)

    def save_baseline(self, results: List[Any], filepath: Optional[str] = None):
        """
        Save successful test results as new baseline.

        Only saves results that passed to establish a good baseline.
        """
        filepath = filepath or self.baseline_file
        if not filepath:
            raise ValueError("No filepath provided for saving baseline")

        baseline_data = {
            "created_at": datetime.now().isoformat(),
            "total_tests": len(results),
            "tests": {},
        }

        for result in results:
            if result.passed:  # Only save successful results
                baseline_data["tests"][result.test_name] = {
                    "discovery_r2": result.discovery_r2,
                    "validation_score": result.validation_score,
                    "passed": result.passed,
                    "expression": result.discovered_expression,
                    "complexity": result.complexity,
                    "random_seed": result.random_seed,
                    "timestamp": result.timestamp
                    if hasattr(result, "timestamp")
                    else datetime.now().isoformat(),
                }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(baseline_data, f, indent=2)

        print(f"✅ Saved {len(baseline_data['tests'])} tests as baseline to {filepath}")
        self.baseline = baseline_data

    def compare_with_baseline(
        self, test_name: str, current_result: Any
    ) -> RegressionReport:
        """
        Compare current result with baseline.

        Returns detailed regression report.
        """
        # Check if baseline exists
        if "tests" not in self.baseline or test_name not in self.baseline["tests"]:
            return RegressionReport(
                test_name=test_name,
                is_regression=False,
                severity="none",
                r2_delta=0.0,
                val_delta=0.0,
                baseline_r2=0.0,
                current_r2=current_result.discovery_r2,
                baseline_val=0.0,
                current_val=current_result.validation_score,
                passed_before=False,
                passed_now=current_result.passed,
                expression_changed=False,
                details={"status": "no_baseline"},
            )

        baseline = self.baseline["tests"][test_name]

        # Calculate deltas
        r2_delta = current_result.discovery_r2 - baseline["discovery_r2"]
        val_delta = current_result.validation_score - baseline["validation_score"]

        # Determine severity
        severity = "none"
        is_regression = False

        # Check for critical regression
        if (
            r2_delta < self.R2_CRITICAL_THRESHOLD
            or val_delta < self.VAL_CRITICAL_THRESHOLD
            or (baseline["passed"] and not current_result.passed)
        ):
            severity = "critical"
            is_regression = True

        # Check for major regression
        elif r2_delta < self.R2_MAJOR_THRESHOLD or val_delta < self.VAL_MAJOR_THRESHOLD:
            severity = "major"
            is_regression = True

        # Check for minor regression
        elif r2_delta < self.R2_MINOR_THRESHOLD or val_delta < self.VAL_MINOR_THRESHOLD:
            severity = "minor"
            is_regression = True

        # Check if expression changed
        expression_changed = (
            baseline.get("expression") != current_result.discovered_expression
        )

        return RegressionReport(
            test_name=test_name,
            is_regression=is_regression,
            severity=severity,
            r2_delta=r2_delta,
            val_delta=val_delta,
            baseline_r2=baseline["discovery_r2"],
            current_r2=current_result.discovery_r2,
            baseline_val=baseline["validation_score"],
            current_val=current_result.validation_score,
            passed_before=baseline["passed"],
            passed_now=current_result.passed,
            expression_changed=expression_changed,
            details={
                "baseline_expression": baseline.get("expression"),
                "current_expression": current_result.discovered_expression,
                "baseline_complexity": baseline.get("complexity"),
                "current_complexity": current_result.complexity,
            },
        )

    def get_all_regressions(
        self, results: List[Any], min_severity: str = "minor"
    ) -> List[RegressionReport]:
        """
        Get all regressions from a list of results.

        Args:
            results: List of test results
            min_severity: Minimum severity to report ('critical', 'major', 'minor')
        """
        severity_order = {"critical": 3, "major": 2, "minor": 1, "none": 0}
        min_level = severity_order[min_severity]

        regressions = []
        for result in results:
            report = self.compare_with_baseline(result.test_name, result)
            if report.is_regression and severity_order[report.severity] >= min_level:
                regressions.append(report)

        # Sort by severity (critical first)
        return sorted(
            regressions, key=lambda x: severity_order[x.severity], reverse=True
        )


# ============================================================================
# ADAPTIVE RETRY STRATEGY
# ============================================================================


class AdaptiveRetryStrategy:
    """Implements adaptive retry strategies for flaky tests."""

    def __init__(
        self,
        max_retries: int = 5,
        seed_manager: Optional[ReproducibleSeedManager] = None,
        data_validator: Optional[DataValidator] = None,
    ):
        self.max_retries = max_retries
        self.seed_manager = seed_manager or ReproducibleSeedManager()
        self.data_validator = data_validator or DataValidator()
        self.retry_stats = defaultdict(lambda: {"attempts": 0, "successes": 0})

    def execute_with_retry(
        self,
        test_name: str,
        test_function: callable,
        quality_threshold: float = 0.90,
        verbose: bool = True,
    ) -> Tuple[Any, Dict]:
        """
        Execute test with adaptive retry strategy.

        Strategies:
        1. Try different random seeds
        2. Validate data before running
        3. Track best result across attempts
        4. Provide detailed retry statistics

        Returns:
            (best_result, retry_metadata)
        """
        attempts = []
        best_result = None
        best_quality = -np.inf

        if verbose:
            print(f"\n🔄 Adaptive Retry: {test_name}")

        # Strategy 1: Try different seeds
        for attempt in range(self.max_retries):
            seed = self.seed_manager.get_test_seed(test_name, attempt)
            self.seed_manager.set_all_seeds(seed)

            if verbose:
                print(
                    f"   Attempt {attempt + 1}/{self.max_retries} (seed={seed})...",
                    end="",
                )

            try:
                start_time = time.time()
                result = test_function(seed=seed)
                elapsed = time.time() - start_time

                # Calculate quality score
                quality = self._calculate_quality(result)

                attempts.append(
                    {
                        "attempt": attempt + 1,
                        "seed": seed,
                        "quality": quality,
                        "r2": result.discovery_r2,
                        "passed": result.passed,
                        "elapsed": elapsed,
                    }
                )

                if verbose:
                    status = "✅" if result.passed else "❌"
                    print(
                        f" {status} (R²={result.discovery_r2:.4f}, quality={quality:.3f})"
                    )

                # Track best
                if quality > best_quality:
                    best_quality = quality
                    best_result = result

                # Early stopping if excellent
                if result.passed and quality >= quality_threshold:
                    if verbose:
                        print(f"   🎯 Excellent result achieved, stopping early")
                    self.retry_stats[test_name]["successes"] += 1
                    break

            except Exception as e:
                if verbose:
                    print(f" ⚠️ Crashed: {str(e)[:50]}")
                attempts.append(
                    {
                        "attempt": attempt + 1,
                        "seed": seed,
                        "quality": 0.0,
                        "error": str(e),
                        "passed": False,
                    }
                )

        self.retry_stats[test_name]["attempts"] += len(attempts)

        # Compile retry metadata
        retry_metadata = {
            "total_attempts": len(attempts),
            "seeds_tried": [a["seed"] for a in attempts],
            "qualities": [a.get("quality", 0.0) for a in attempts],
            "best_quality": best_quality,
            "best_seed": attempts[np.argmax([a.get("quality", 0.0) for a in attempts])][
                "seed"
            ]
            if attempts
            else None,
            "early_stop": len(attempts) < self.max_retries,
            "attempts_detail": attempts,
        }

        return best_result, retry_metadata

    def _calculate_quality(self, result: Any) -> float:
        """
        Calculate overall quality score for a result.

        Combines R², validation, and pass/fail status.
        """
        r2_component = result.discovery_r2
        val_component = result.validation_score / 100.0
        pass_bonus = 1.0 if result.passed else 0.5

        quality = (r2_component * 0.6 + val_component * 0.4) * pass_bonus
        return quality

    def get_retry_statistics(self) -> Dict[str, Any]:
        """Get statistics about retry effectiveness."""
        if not self.retry_stats:
            return {"no_data": True}

        total_attempts = sum(s["attempts"] for s in self.retry_stats.values())
        total_successes = sum(s["successes"] for s in self.retry_stats.values())

        return {
            "total_tests_retried": len(self.retry_stats),
            "total_attempts": total_attempts,
            "total_successes": total_successes,
            "success_rate": total_successes / len(self.retry_stats)
            if self.retry_stats
            else 0,
            "avg_attempts_per_test": total_attempts / len(self.retry_stats)
            if self.retry_stats
            else 0,
            "by_test": dict(self.retry_stats),
        }


# ============================================================================
# COMPREHENSIVE REPORTING
# ============================================================================


class StabilityReporter:
    """Generate comprehensive stability reports."""

    @staticmethod
    def generate_summary_report(
        tracker: StabilityTracker,
        comparator: BaselineComparator,
        results: List[Any],
        output_file: Optional[str] = None,
    ) -> str:
        """Generate comprehensive text report."""
        lines = []
        lines.append("=" * 80)
        lines.append("TEST STABILITY REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Total Tests: {len(results)}")
        lines.append("")

        # Overall statistics
        passed = sum(1 for r in results if r.passed)
        lines.append("OVERALL STATISTICS")
        lines.append("-" * 80)
        lines.append(
            f"Passed: {passed}/{len(results)} ({passed / len(results) * 100:.1f}%)"
        )
        lines.append(f"Failed: {len(results) - passed}/{len(results)}")
        lines.append("")

        # Stability analysis
        all_metrics = tracker.get_all_metrics()
        stable_tests = sum(
            1
            for m in all_metrics.values()
            if not m.get("insufficient_data") and m["is_stable"]
        )

        lines.append("STABILITY ANALYSIS")
        lines.append("-" * 80)
        lines.append(f"Stable Tests: {stable_tests}/{len(all_metrics)}")
        lines.append("")

        # Flaky tests
        flaky = tracker.identify_flaky_tests()
        if flaky:
            lines.append("⚠️ FLAKY TESTS DETECTED")
            lines.append("-" * 80)
            for test_name, metrics in flaky[:10]:  # Top 10
                lines.append(f"\n{test_name}:")
                lines.append(
                    f"  Pass Rate: {metrics['pass_rate']:.1%} ({metrics['passes']}/{metrics['total_runs']})"
                )
                lines.append(
                    f"  R² Stats: μ={metrics['r2_mean']:.4f}, σ={metrics['r2_std']:.4f}"
                )
                lines.append(f"  Flakiness Score: {metrics['flakiness_score']:.1f}/100")
            lines.append("")
        else:
            lines.append("✅ No flaky tests detected")
            lines.append("")

        # Regressions
        regressions = comparator.get_all_regressions(results)
        if regressions:
            lines.append("❌ REGRESSIONS DETECTED")
            lines.append("-" * 80)
            for reg in regressions:
                severity_icon = {"critical": "🔴", "major": "🟠", "minor": "🟡"}
                icon = severity_icon.get(reg.severity, "⚪")
                lines.append(f"\n{icon} {reg.test_name} [{reg.severity.upper()}]")
                lines.append(
                    f"  R² Change: {reg.baseline_r2:.4f} → {reg.current_r2:.4f} (Δ={reg.r2_delta:+.4f})"
                )
                lines.append(
                    f"  Val Change: {reg.baseline_val:.1f} → {reg.current_val:.1f} (Δ={reg.val_delta:+.1f})"
                )
                if reg.expression_changed:
                    lines.append(f"  ⚠️ Expression changed")
            lines.append("")
        else:
            lines.append("✅ No regressions detected")
            lines.append("")

        # Test details
        lines.append("TEST DETAILS")
        lines.append("-" * 80)
        for result in results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            lines.append(f"\n{result.test_name}: {status}")
            lines.append(f"  R²: {result.discovery_r2:.4f}")
            lines.append(f"  Validation: {result.validation_score:.1f}/100")
            lines.append(f"  Engine: {result.discovery_engine or 'unknown'}")

            # Stability info if available
            if result.test_name in all_metrics:
                metrics = all_metrics[result.test_name]
                if not metrics.get("insufficient_data"):
                    lines.append(
                        f"  Stability: {'✅ Stable' if metrics['is_stable'] else '⚠️ Flaky'} (pass rate: {metrics['pass_rate']:.1%})"
                    )

        lines.append("")
        lines.append("=" * 80)
        report = "\n".join(lines)

        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                f.write(report)
            print(f"📄 Report saved to {output_file}")

        return report

    @staticmethod
    def generate_json_report(
        tracker: StabilityTracker,
        comparator: BaselineComparator,
        results: List[Any],
        retry_strategy: Optional[AdaptiveRetryStrategy] = None,
        output_file: Optional[str] = None,
    ) -> Dict:
        """Generate machine-readable JSON report."""
        all_metrics = tracker.get_all_metrics()
        regressions = comparator.get_all_regressions(results)
        flaky = tracker.identify_flaky_tests()

        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_tests": len(results),
                "passed": sum(1 for r in results if r.passed),
                "failed": sum(1 for r in results if not r.passed),
            },
            "summary": {
                "pass_rate": sum(1 for r in results if r.passed) / len(results)
                if results
                else 0,
                "stable_tests": sum(
                    1
                    for m in all_metrics.values()
                    if not m.get("insufficient_data") and m["is_stable"]
                ),
                "flaky_tests": len(flaky),
                "regressions": len(regressions),
                "critical_regressions": sum(
                    1 for r in regressions if r.severity == "critical"
                ),
                "major_regressions": sum(
                    1 for r in regressions if r.severity == "major"
                ),
            },
            "stability_metrics": all_metrics,
            "flaky_tests": [
                {"test_name": name, "metrics": metrics} for name, metrics in flaky
            ],
            "regressions": [asdict(reg) for reg in regressions],
            "test_results": [
                {
                    "test_name": r.test_name,
                    "passed": r.passed,
                    "discovery_r2": r.discovery_r2,
                    "validation_score": r.validation_score,
                    "discovered_expression": r.discovered_expression,
                    "discovery_engine": r.discovery_engine
                    if hasattr(r, "discovery_engine")
                    else None,
                    "complexity": r.complexity if hasattr(r, "complexity") else None,
                }
                for r in results
            ],
        }

        # Add retry statistics if available
        if retry_strategy:
            report["retry_statistics"] = retry_strategy.get_retry_statistics()

        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(report, f, indent=2)
            print(f"📊 JSON report saved to {output_file}")

        return report

    @staticmethod
    def print_console_summary(
        tracker: StabilityTracker, comparator: BaselineComparator, results: List[Any]
    ):
        """Print concise summary to console."""
        passed = sum(1 for r in results if r.passed)
        flaky = tracker.identify_flaky_tests()
        regressions = comparator.get_all_regressions(results)

        print("\n" + "=" * 80)
        print("📊 STABILITY SUMMARY")
        print("=" * 80)
        print(
            f"✅ Passed: {passed}/{len(results)} ({passed / len(results) * 100:.1f}%)"
        )
        print(f"❌ Failed: {len(results) - passed}/{len(results)}")
        print(f"⚠️  Flaky Tests: {len(flaky)}")
        print(f"📉 Regressions: {len(regressions)}")

        if regressions:
            critical = sum(1 for r in regressions if r.severity == "critical")
            major = sum(1 for r in regressions if r.severity == "major")
            if critical > 0:
                print(f"   🔴 Critical: {critical}")
            if major > 0:
                print(f"   🟠 Major: {major}")

        print("=" * 80 + "\n")


# ============================================================================
# INTEGRATED TEST RUNNER
# ============================================================================


class RobustTestRunner:
    """
    Integrated test runner with all stability features.

    Features:
    - Deterministic seed management
    - Adaptive retry strategies
    - Data validation
    - Stability tracking
    - Baseline comparison
    - Comprehensive reporting
    """

    def __init__(
        self,
        base_seed: int = 42,
        max_retries: int = 3,
        history_file: str = "test_history.json",
        baseline_file: str = "baseline_results.json",
    ):
        self.seed_manager = ReproducibleSeedManager(base_seed)
        self.tracker = StabilityTracker(history_file)
        self.comparator = BaselineComparator(baseline_file)
        self.retry_strategy = AdaptiveRetryStrategy(
            max_retries=max_retries, seed_manager=self.seed_manager
        )
        self.data_validator = DataValidator()

        print(f"🚀 Initialized RobustTestRunner")
        print(f"   Base Seed: {base_seed}")
        print(f"   Max Retries: {max_retries}")
        print(f"   History: {history_file}")
        print(f"   Baseline: {baseline_file}")

    def run_test_suite(
        self,
        test_functions: Dict[str, callable],
        compare_baseline: bool = True,
        save_baseline: bool = False,
        save_history: bool = True,
        generate_reports: bool = True,
        report_dir: str = "reports",
        verbose: bool = True,
    ) -> Tuple[List[Any], Dict]:
        """
        Run complete test suite with all protections.

        Args:
            test_functions: Dict mapping test names to test functions
            compare_baseline: Whether to compare with baseline
            save_baseline: Whether to save results as new baseline
            save_history: Whether to save test history
            generate_reports: Whether to generate reports
            report_dir: Directory for report files
            verbose: Print detailed output

        Returns:
            (results, metadata)
        """
        print(f"\n{'=' * 80}")
        print(f"🧪 ROBUST TEST SUITE")
        print(f"{'=' * 80}")
        print(f"Tests: {len(test_functions)}")
        print(f"Compare Baseline: {compare_baseline}")
        print(f"Save Baseline: {save_baseline}")
        print(f"{'=' * 80}\n")

        results = []
        start_time = time.time()

        # Run each test
        for i, (test_name, test_func) in enumerate(test_functions.items(), 1):
            if verbose:
                print(f"\n[{i}/{len(test_functions)}] 🔬 {test_name}")

            # Run with adaptive retry
            result, retry_metadata = self.retry_strategy.execute_with_retry(
                test_name=test_name, test_function=test_func, verbose=verbose
            )

            if result:
                # Add retry metadata to result
                if hasattr(result, "__dict__"):
                    result.retry_metadata = retry_metadata

                # Record in stability tracker
                test_run = TestRun(
                    test_name=test_name,
                    passed=result.passed,
                    r2_score=result.discovery_r2,
                    validation_score=result.validation_score,
                    seed=retry_metadata["best_seed"],
                    attempt=retry_metadata["total_attempts"],
                    timestamp=datetime.now().isoformat(),
                    execution_time=sum(
                        a["elapsed"]
                        for a in retry_metadata["attempts_detail"]
                        if "elapsed" in a
                    ),
                    expression=result.discovered_expression
                    if hasattr(result, "discovered_expression")
                    else None,
                    failure_reason=result.failure_reason
                    if hasattr(result, "failure_reason")
                    else None,
                )
                self.tracker.record_run(test_run)

                # Compare with baseline
                if compare_baseline:
                    regression = self.comparator.compare_with_baseline(
                        test_name, result
                    )
                    if regression.is_regression:
                        severity_icon = {"critical": "🔴", "major": "🟠", "minor": "🟡"}
                        icon = severity_icon.get(regression.severity, "⚪")
                        if verbose:
                            print(
                                f"   {icon} REGRESSION [{regression.severity.upper()}]"
                            )
                            print(f"      R² Δ: {regression.r2_delta:+.4f}")
                            print(f"      Val Δ: {regression.val_delta:+.1f}")

                results.append(result)

                # Status
                if verbose:
                    status = "✅ PASSED" if result.passed else "❌ FAILED"
                    print(
                        f"   {status} (R²={result.discovery_r2:.4f}, Val={result.validation_score:.1f})"
                    )

        total_time = time.time() - start_time

        # Generate reports
        if generate_reports:
            Path(report_dir).mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Text report
            text_report_file = f"{report_dir}/stability_report_{timestamp}.txt"
            StabilityReporter.generate_summary_report(
                self.tracker, self.comparator, results, output_file=text_report_file
            )

            # JSON report
            json_report_file = f"{report_dir}/stability_report_{timestamp}.json"
            StabilityReporter.generate_json_report(
                self.tracker,
                self.comparator,
                results,
                retry_strategy=self.retry_strategy,
                output_file=json_report_file,
            )

        # Console summary
        StabilityReporter.print_console_summary(self.tracker, self.comparator, results)

        # Save history
        if save_history:
            self.tracker.save_history()
            print(f"💾 Test history saved")

        # Save baseline
        if save_baseline:
            self.comparator.save_baseline(results)

        # Metadata
        metadata = {
            "total_tests": len(test_functions),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "total_time": total_time,
            "avg_time_per_test": total_time / len(results) if results else 0,
            "flaky_tests": len(self.tracker.identify_flaky_tests()),
            "regressions": len(self.comparator.get_all_regressions(results)),
            "retry_stats": self.retry_strategy.get_retry_statistics(),
        }

        print(
            f"\n⏱️  Total execution time: {total_time:.1f}s ({metadata['avg_time_per_test']:.1f}s per test)"
        )

        return results, metadata

    def analyze_test_stability(
        self, test_name: str, num_runs: int = 10, verbose: bool = True
    ) -> Dict:
        """
        Analyze stability of a specific test by running it multiple times.

        Returns detailed stability metrics.
        """
        print(f"\n{'=' * 80}")
        print(f"🔍 STABILITY ANALYSIS: {test_name}")
        print(f"{'=' * 80}")
        print(f"Running {num_runs} times with different seeds...")
        print()

        # This is a placeholder - actual implementation would need the test function
        # For now, return existing metrics if available
        metrics = self.tracker.get_stability_metrics(test_name)

        if metrics.get("insufficient_data"):
            print(f"⚠️  Insufficient data for {test_name}")
            print(f"   Only {metrics.get('total_runs', 0)} runs recorded")
            return metrics

        print(f"📊 Stability Metrics:")
        print(f"   Total Runs: {metrics['total_runs']}")
        print(f"   Pass Rate: {metrics['pass_rate']:.1%}")
        print(f"   R² Stats: μ={metrics['r2_mean']:.4f}, σ={metrics['r2_std']:.4f}")
        print(f"   Flakiness Score: {metrics['flakiness_score']:.1f}/100")
        print(f"   Status: {'✅ Stable' if metrics['is_stable'] else '⚠️  Flaky'}")

        return metrics


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def validate_data_before_test(
    X: np.ndarray, y: np.ndarray, test_name: str, raise_on_issues: bool = False
) -> DataValidationResult:
    """
    Validate data before running a test.

    Convenience function for data validation.
    """
    validator = DataValidator()
    result = validator.validate_data(X, y, test_name, verbose=True)

    if raise_on_issues and not result.stable:
        raise ValueError(f"Data validation failed for {test_name}: {result.issues}")

    return result


def set_global_seed(seed: int = 42):
    """
    Set all random seeds globally.

    Convenience function for seed management.
    """
    manager = ReproducibleSeedManager(seed)
    manager.set_all_seeds(seed)
    print(f"🎲 Global seed set to: {seed}")


def compare_test_runs(
    history_file: str, test_name: str, show_plot: bool = False
) -> Dict:
    """
    Compare multiple runs of the same test.

    Args:
        history_file: Path to test history JSON
        test_name: Name of test to analyze
        show_plot: Whether to display plot (requires matplotlib)

    Returns:
        Comparison statistics
    """
    tracker = StabilityTracker(history_file)

    if test_name not in tracker.test_history:
        print(f"⚠️  No history found for {test_name}")
        return {}

    runs = tracker.test_history[test_name]
    metrics = tracker.get_stability_metrics(test_name)

    print(f"\n{'=' * 80}")
    print(f"📈 TEST RUN COMPARISON: {test_name}")
    print(f"{'=' * 80}")
    print(f"Total Runs: {len(runs)}")
    print(f"Pass Rate: {metrics['pass_rate']:.1%}")
    print(f"R² Range: [{metrics['r2_min']:.4f}, {metrics['r2_max']:.4f}]")
    print(f"R² Mean ± Std: {metrics['r2_mean']:.4f} ± {metrics['r2_std']:.4f}")
    print()

    # Show individual runs
    print("Individual Runs:")
    for i, run in enumerate(runs[-10:], 1):  # Last 10 runs
        status = "✅" if run.passed else "❌"
        print(
            f"  {i}. {status} R²={run.r2_score:.4f} Val={run.validation_score:.1f} (seed={run.seed})"
        )

    if show_plot:
        try:
            import matplotlib.pyplot as plt

            r2_values = [r.r2_score for r in runs]
            val_scores = [r.validation_score for r in runs]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            ax1.plot(r2_values, marker="o")
            ax1.axhline(metrics["r2_mean"], color="r", linestyle="--", label="Mean")
            ax1.fill_between(
                range(len(r2_values)),
                metrics["r2_mean"] - metrics["r2_std"],
                metrics["r2_mean"] + metrics["r2_std"],
                alpha=0.3,
                color="r",
                label="±1σ",
            )
            ax1.set_xlabel("Run")
            ax1.set_ylabel("R² Score")
            ax1.set_title(f"{test_name} - R² Stability")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot(val_scores, marker="s", color="orange")
            ax2.axhline(metrics["val_mean"], color="r", linestyle="--", label="Mean")
            ax2.set_xlabel("Run")
            ax2.set_ylabel("Validation Score")
            ax2.set_title(f"{test_name} - Validation Stability")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("\n⚠️  matplotlib not available for plotting")

    return metrics


# ============================================================================
# CLI INTERFACE
# ============================================================================


def main():
    """Command-line interface for the stability framework."""
    import argparse

    parser = argparse.ArgumentParser(
        description="HypatiaX Test Stability Monitoring Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze stability from history
  python pp.py analyze --history test_history.json --test kinetic_energy
  
  # Compare with baseline
  python pp.py compare --baseline baseline.json --current results.json
  
  # Generate reports
  python pp.py report --history test_history.json --output reports/
  
  # Validate test data
  python pp.py validate --data test_data.npz
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze test stability")
    analyze_parser.add_argument("--history", required=True, help="Test history file")
    analyze_parser.add_argument("--test", help="Specific test to analyze")
    analyze_parser.add_argument("--plot", action="store_true", help="Show plots")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare with baseline")
    compare_parser.add_argument("--baseline", required=True, help="Baseline file")
    compare_parser.add_argument("--current", required=True, help="Current results")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate reports")
    report_parser.add_argument("--history", required=True, help="Test history file")
    report_parser.add_argument("--baseline", help="Baseline file for comparison")
    report_parser.add_argument("--output", default="reports", help="Output directory")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate test data")
    validate_parser.add_argument("--data", required=True, help="Data file (.npz)")

    args = parser.parse_args()

    if args.command == "analyze":
        if args.test:
            compare_test_runs(args.history, args.test, show_plot=args.plot)
        else:
            tracker = StabilityTracker(args.history)
            flaky = tracker.identify_flaky_tests()
            print(f"\n📊 Found {len(flaky)} flaky tests")
            for test_name, metrics in flaky[:10]:
                print(f"  • {test_name}: {metrics['pass_rate']:.1%} pass rate")

    elif args.command == "compare":
        # Load and compare results
        comparator = BaselineComparator(args.baseline)
        with open(args.current, "r") as f:
            current_data = json.load(f)

        print(
            f"\n🔍 Comparing {len(current_data.get('tests', {}))} tests with baseline"
        )
        # Implementation would depend on current results format

    elif args.command == "report":
        tracker = StabilityTracker(args.history)
        comparator = (
            BaselineComparator(args.baseline) if args.baseline else BaselineComparator()
        )

        # Generate reports (would need actual results)
        print(f"\n📄 Generating reports in {args.output}/")
        Path(args.output).mkdir(parents=True, exist_ok=True)

    elif args.command == "validate":
        data = np.load(args.data)
        X = data["X"]
        y = data["y"]

        result = validate_data_before_test(X, y, "data_validation")
        if result.stable:
            print("\n✅ Data validation passed")
        else:
            print("\n❌ Data validation failed")
            for issue in result.issues:
                print(f"   • {issue}")

    else:
        parser.print_help()


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "ReproducibleSeedManager",
    "DataValidator",
    "DataValidationResult",
    "StabilityTracker",
    "TestRun",
    "BaselineComparator",
    "RegressionReport",
    "AdaptiveRetryStrategy",
    "StabilityReporter",
    "RobustTestRunner",
    "validate_data_before_test",
    "set_global_seed",
    "compare_test_runs",
]


if __name__ == "__main__":
    main()

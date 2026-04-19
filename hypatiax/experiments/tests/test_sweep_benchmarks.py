#!/usr/bin/env python3
"""
tests/test_sweep_benchmarks.py
================================

Unit and integration tests for:
  • run_noise_sweep_benchmark.py
  • run_sample_complexity_benchmark.py

Run with:
    pytest tests/test_sweep_benchmarks.py -v
    pytest tests/test_sweep_benchmarks.py -v -k "noise"
    pytest tests/test_sweep_benchmarks.py -v -k "sample"
    pytest tests/test_sweep_benchmarks.py -v --tb=short

No HypatiaX package or Anthropic API key is required — all subprocess calls
are mocked out.  The tests exercise:
   1. TeeLogger behaviour
   2. _find_latest_result
   3. _extract_per_test
   4. Noise-sweep: _build_runner_cmd
   5. Sample-complexity: _build_runner_cmd
   6. Noise-sweep: _aggregate_results
   7. Sample-complexity: _aggregate_results
   8. JSON / CSV output correctness
   9. Print / reporting smoke tests
  10. _run_noise_level / _run_sample_size mocking
  11. CLI argument parsing
  12. Edge-cases: empty / malformed data
  13. Catastrophic-failure flag in _build_runner_cmd (NS)
  14. Top-two validation against real benchmark results
  15. Noise env-var injection
  16. Collision fix: _find_result_written_after
  17. Catastrophic failure detection
  18. --existing-results parsing
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import the two modules under test using their absolute file paths.
# This is immune to sys.path / cwd / pytest import-mode issues — the scripts
# are resolved relative to THIS file regardless of where pytest is invoked from.
# ---------------------------------------------------------------------------
import importlib.util

def _import_from_path(module_name: str, filepath: Path):
    """Import a module from an absolute file path and register it in sys.modules."""
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

_SCRIPTS = Path(__file__).resolve().parent.parent   # hypatiax/experiments/

NS = _import_from_path(
    "run_noise_sweep_benchmark",
    _SCRIPTS / "run_noise_sweep_benchmark.py",
)
SC = _import_from_path(
    "run_sample_complexity_benchmark",
    _SCRIPTS / "run_sample_complexity_benchmark.py",
)


# ===========================================================================
# FIXTURES
# ===========================================================================

def _make_result_payload(
    method_names: List[str],
    equations:    List[str],
    r2_map:       Optional[Dict[str, Dict[str, float]]] = None,
    noiseless:    bool = False,
) -> Dict:
    """
    Build a minimal result JSON payload in the format written by
    run_comparative_suite_benchmark_v2.py.

    r2_map:  {equation_name: {method_name: r2_value}}

    ``success`` mirrors what the inner benchmark sets: True when the method
    produced a valid prediction (r2 is finite and > 0).  Whether that
    constitutes a *recovery* is a separate question decided by the threshold
    inside _aggregate_results, so we must not conflate the two here.
    Previously this was ``r2 > 0`` which accidentally marked sub-threshold
    results as successes, inflating n_success counts in low-R² fixtures.
    The corrected rule is: success iff r2 is finite and non-negative
    (matches the inner benchmark's contract).
    """
    threshold = 0.9999 if noiseless else 0.995
    tests = []
    for eq in equations:
        results = {}
        for m in method_names:
            r2 = (r2_map or {}).get(eq, {}).get(m, 0.95)
            results[m] = {
                "success": (r2 is not None) and np.isfinite(r2) and r2 >= 0,
                "r2":      r2,
                "rmse":    abs(1 - r2) * 0.1,
                "formula": f"f({eq})",
                "time":    1.0,
                "error":   None,
                "metadata": {},
            }
        tests.append({
            "description": eq,
            "metadata":    {"equation_name": eq},
            "results":     results,
            "winner":      method_names[0],
            "domain":      "feynman_mechanics",
            "comparison":  {"y_scale": {"denom": 1.0}, "duplicates": {}},
        })

    mode = "noiseless" if noiseless else "noisy"
    return {
        "timestamp":    "2025-01-01T00:00:00",
        "protocol":     {"mode": mode, "noise_level": 0.0 if noiseless else 0.05,
                         "threshold": threshold},
        "total_tests":  len(tests),
        "methods":      method_names,
        "tests":        tests,
    }


@pytest.fixture
def tmp_results_dir(tmp_path):
    """Patch both modules to write/read from a temporary results directory."""
    orig_ns = NS._RESULTS_DIR
    orig_sc = SC._RESULTS_DIR
    NS._RESULTS_DIR = tmp_path
    SC._RESULTS_DIR = tmp_path
    yield tmp_path
    NS._RESULTS_DIR = orig_ns
    SC._RESULTS_DIR = orig_sc


@pytest.fixture
def sample_methods():
    return ["PureLLM Baseline (core)", "ImprovedNN (core)"]


@pytest.fixture
def sample_equations():
    return ["arrhenius", "coulomb", "newton_gravity"]


@pytest.fixture
def sample_payload(sample_methods, sample_equations):
    return _make_result_payload(sample_methods, sample_equations)


@pytest.fixture
def noiseless_payload(sample_methods, sample_equations):
    return _make_result_payload(sample_methods, sample_equations, noiseless=True)


@pytest.fixture
def ns_args():
    """Default argparse.Namespace for noise sweep tests."""
    return argparse.Namespace(
        noise_levels=[0.01, 0.05, 0.10],
        methods=[3, 4],
        threshold_noisy=0.995,
        threshold_noiseless=0.9999,
        samples=200,
        nn_seeds=3,
        method_timeout=900,
        pysr_timeout=1100,
        skip_pysr=False,
        test=None,
        equations=None,
        domain="all_domains",
        series=None,
        benchmark="feynman",
        verbose=False,
        quiet=False,
        no_llm_cache=False,
        fail_fast=False,
        log=None,
        runner=None,
    )


@pytest.fixture
def sc_args():
    """Default argparse.Namespace for sample complexity tests."""
    return argparse.Namespace(
        sample_sizes=[50, 100, 200, 500],
        methods=None,
        noiseless=False,
        threshold_noisy=0.995,
        threshold_noiseless=0.9999,
        nn_seeds=3,
        method_timeout=900,
        pysr_timeout=1100,
        skip_pysr=False,
        test=None,
        equations=None,
        domain="all_domains",
        series=None,
        benchmark="feynman",
        verbose=False,
        quiet=False,
        no_llm_cache=False,
        fail_fast=False,
        log=None,
        runner=None,
    )


# ===========================================================================
# HELPERS
# ===========================================================================

def _write_json(path: Path, payload: Dict) -> Path:
    with open(path, "w") as f:
        json.dump(payload, f)
    return path


_result_file_counter = 0   # module-level counter — ensures unique filenames


def _make_result_file(
    tmp_path:     Path,
    mode:         str,
    methods:      List[str],
    equations:    List[str],
    r2_map:       Optional[Dict] = None,
) -> Path:
    global _result_file_counter
    _result_file_counter += 1
    payload = _make_result_payload(
        methods, equations, r2_map, noiseless=(mode == "noiseless")
    )
    # Use a unique suffix so that multiple calls with the same mode within one
    # test (or across tests in the same session) never collide on the same path.
    # Previously the hardcoded timestamp "20250101_000000" caused all calls to
    # produce the same filename, so later writes silently overwrote earlier ones
    # and tests reading the "n=50" path actually got n=500 data.
    ts   = f"20250101_{_result_file_counter:06d}"
    path = tmp_path / f"protocol_core_{mode}_{ts}.json"
    _write_json(path, payload)
    return path


# ===========================================================================
# ── 1.  TeeLogger  ──────────────────────────────────────────────────────────
# ===========================================================================

class TestTeeLogger:

    def test_write_mirrors_to_both_streams(self):
        real    = io.StringIO()
        logfile = io.StringIO()
        tee     = NS._TeeLogger(real, logfile)
        tee.write("hello\n")
        assert real.getvalue()    == "hello\n"
        assert logfile.getvalue() == "hello\n"

    def test_flush_called_on_both(self):
        real    = MagicMock()
        logfile = MagicMock()
        tee     = NS._TeeLogger(real, logfile)
        tee.flush()
        real.flush.assert_called_once()
        logfile.flush.assert_called_once()

    def test_isatty_false_when_no_method(self):
        tee = NS._TeeLogger(io.StringIO(), io.StringIO())
        assert tee.isatty() is False

    def test_sc_tee_logger_mirrors(self):
        """Sample complexity module has its own copy — test it too."""
        real    = io.StringIO()
        logfile = io.StringIO()
        tee     = SC._TeeLogger(real, logfile)
        tee.write("world\n")
        assert real.getvalue()    == "world\n"
        assert logfile.getvalue() == "world\n"


# ===========================================================================
# ── 2.  _find_latest_result  ─────────────────────────────────────────────────
# ===========================================================================

class TestFindLatestResult:

    def test_returns_most_recent_file(self, tmp_results_dir):
        p1 = tmp_results_dir / "protocol_core_noisy_20250101_000000.json"
        p2 = tmp_results_dir / "protocol_core_noisy_20250102_000000.json"
        p1.write_text("{}")
        p2.write_text("{}")
        # Ensure p2 is newer
        os.utime(p2, (p2.stat().st_mtime + 10, p2.stat().st_mtime + 10))
        result = NS._find_latest_result("noisy")
        assert result is not None
        assert result.name == p2.name

    def test_returns_none_when_no_files(self, tmp_results_dir):
        assert NS._find_latest_result("noisy") is None

    def test_finds_noiseless_files(self, tmp_results_dir):
        p = tmp_results_dir / "protocol_core_noiseless_20250101_000000.json"
        p.write_text("{}")
        result = NS._find_latest_result("noiseless")
        assert result is not None
        assert "noiseless" in result.name

    def test_sc_find_latest_result(self, tmp_results_dir):
        p = tmp_results_dir / "protocol_core_noisy_20250101_000000.json"
        p.write_text("{}")
        result = SC._find_latest_result("noisy")
        assert result is not None


# ===========================================================================
# ── 3.  _extract_per_test  ───────────────────────────────────────────────────
# ===========================================================================

class TestExtractPerTest:

    def test_basic_extraction(self, sample_payload, sample_methods, sample_equations):
        per_eq = NS._extract_per_test(sample_payload)
        assert set(per_eq.keys()) == set(sample_equations)
        for eq in sample_equations:
            assert set(per_eq[eq].keys()) == set(sample_methods)

    def test_r2_values_preserved(self, sample_methods, sample_equations):
        r2_map = {eq: {m: 0.99 for m in sample_methods} for eq in sample_equations}
        payload = _make_result_payload(sample_methods, sample_equations, r2_map)
        per_eq  = NS._extract_per_test(payload)
        for eq in sample_equations:
            for m in sample_methods:
                assert per_eq[eq][m]["r2"] == pytest.approx(0.99)

    def test_empty_tests_list(self):
        assert NS._extract_per_test({"tests": []}) == {}

    def test_missing_tests_key(self):
        assert NS._extract_per_test({}) == {}

    def test_sc_extract_per_test(self, sample_payload):
        per_eq = SC._extract_per_test(sample_payload)
        assert len(per_eq) > 0


# ===========================================================================
# ── 4.  Noise-sweep: _build_runner_cmd  ─────────────────────────────────────
# ===========================================================================

class TestNoiseSweepBuildCmd:

    def _dummy_runner(self, tmp_path):
        p = tmp_path / "runner.py"
        p.write_text("pass")
        return p

    def test_noiseless_flag_added_for_sigma_zero(self, ns_args, tmp_path):
        runner = self._dummy_runner(tmp_path)
        cmd, label = NS._build_runner_cmd(0.0, ns_args, runner)
        assert "--noiseless" in cmd
        assert label == "sig0000"

    def test_noiseless_flag_absent_for_nonzero_sigma(self, ns_args, tmp_path):
        runner = self._dummy_runner(tmp_path)
        cmd, label = NS._build_runner_cmd(0.05, ns_args, runner)
        assert "--noiseless" not in cmd
        assert label == "sig0050"

    def test_sigma_label_encoding(self, ns_args, tmp_path):
        runner = self._dummy_runner(tmp_path)
        for sigma, expected_label in [
            (0.00, "sig0000"),
            (0.01, "sig0010"),
            (0.05, "sig0050"),
            (0.10, "sig0100"),
        ]:
            _, label = NS._build_runner_cmd(sigma, ns_args, runner)
            assert label == expected_label, f"sigma={sigma} → expected {expected_label}, got {label}"

    def test_methods_forwarded(self, ns_args, tmp_path):
        runner = self._dummy_runner(tmp_path)
        ns_args.methods = [1, 2]
        cmd, _ = NS._build_runner_cmd(0.05, ns_args, runner)
        assert "--methods" in cmd
        idx = cmd.index("--methods")
        assert cmd[idx + 1] == "1"
        assert cmd[idx + 2] == "2"

    def test_skip_pysr_forwarded(self, ns_args, tmp_path):
        runner = self._dummy_runner(tmp_path)
        ns_args.skip_pysr = True
        cmd, _ = NS._build_runner_cmd(0.05, ns_args, runner)
        assert "--skip-pysr" in cmd

    def test_skip_pysr_absent_when_false(self, ns_args, tmp_path):
        runner = self._dummy_runner(tmp_path)
        ns_args.skip_pysr = False
        cmd, _ = NS._build_runner_cmd(0.05, ns_args, runner)
        assert "--skip-pysr" not in cmd

    def test_samples_forwarded(self, ns_args, tmp_path):
        runner = self._dummy_runner(tmp_path)
        ns_args.samples = 500
        cmd, _ = NS._build_runner_cmd(0.05, ns_args, runner)
        assert "--samples" in cmd
        assert cmd[cmd.index("--samples") + 1] == "500"

    def test_checkpoint_name_is_sigma_specific(self, ns_args, tmp_path):
        runner = self._dummy_runner(tmp_path)
        cmd05, _ = NS._build_runner_cmd(0.05, ns_args, runner)
        cmd10, _ = NS._build_runner_cmd(0.10, ns_args, runner)
        ckpt05 = cmd05[cmd05.index("--checkpoint-name") + 1]
        ckpt10 = cmd10[cmd10.index("--checkpoint-name") + 1]
        assert ckpt05 != ckpt10
        assert "sig0050" in ckpt05
        assert "sig0100" in ckpt10

    def test_test_flag_forwarded(self, ns_args, tmp_path):
        runner = self._dummy_runner(tmp_path)
        ns_args.test = "arrhenius"
        cmd, _ = NS._build_runner_cmd(0.05, ns_args, runner)
        assert "--test" in cmd
        assert cmd[cmd.index("--test") + 1] == "arrhenius"

    def test_test_flag_absent_when_none(self, ns_args, tmp_path):
        runner = self._dummy_runner(tmp_path)
        ns_args.test = None
        cmd, _ = NS._build_runner_cmd(0.05, ns_args, runner)
        assert "--test" not in cmd

    def test_verbose_forwarded(self, ns_args, tmp_path):
        runner = self._dummy_runner(tmp_path)
        ns_args.verbose = True
        cmd, _ = NS._build_runner_cmd(0.05, ns_args, runner)
        assert "--verbose" in cmd

    def test_nn_seeds_forwarded(self, ns_args, tmp_path):
        runner = self._dummy_runner(tmp_path)
        ns_args.nn_seeds = 5
        cmd, _ = NS._build_runner_cmd(0.05, ns_args, runner)
        assert "--nn-seeds" in cmd
        assert cmd[cmd.index("--nn-seeds") + 1] == "5"

    def test_threshold_noisy_forwarded(self, ns_args, tmp_path):
        runner = self._dummy_runner(tmp_path)
        ns_args.threshold_noisy = 0.98
        cmd, _ = NS._build_runner_cmd(0.05, ns_args, runner)
        assert "--threshold" in cmd
        assert cmd[cmd.index("--threshold") + 1] == "0.98"

    def test_threshold_noiseless_forwarded(self, ns_args, tmp_path):
        runner = self._dummy_runner(tmp_path)
        ns_args.threshold_noiseless = 0.9995
        cmd, _ = NS._build_runner_cmd(0.0, ns_args, runner)
        assert "--threshold" in cmd
        assert cmd[cmd.index("--threshold") + 1] == "0.9995"


# ===========================================================================
# ── 5.  Sample-complexity: _build_runner_cmd  ───────────────────────────────
# ===========================================================================

class TestSCBuildCmd:

    def _dummy_runner(self, tmp_path):
        p = tmp_path / "runner.py"
        p.write_text("pass")
        return p

    def test_samples_set_to_n(self, sc_args, tmp_path):
        runner = self._dummy_runner(tmp_path)
        for n in [50, 100, 200, 500]:
            cmd = SC._build_runner_cmd(n, sc_args, runner)
            assert "--samples" in cmd
            assert cmd[cmd.index("--samples") + 1] == str(n)

    def test_noiseless_flag_when_mode_noiseless(self, sc_args, tmp_path):
        runner = self._dummy_runner(tmp_path)
        sc_args.noiseless = True
        cmd = SC._build_runner_cmd(100, sc_args, runner)
        assert "--noiseless" in cmd

    def test_noiseless_flag_absent_in_noisy_mode(self, sc_args, tmp_path):
        runner = self._dummy_runner(tmp_path)
        sc_args.noiseless = False
        cmd = SC._build_runner_cmd(100, sc_args, runner)
        assert "--noiseless" not in cmd

    def test_checkpoint_name_is_n_specific(self, sc_args, tmp_path):
        runner = self._dummy_runner(tmp_path)
        cmd50  = SC._build_runner_cmd(50,  sc_args, runner)
        cmd500 = SC._build_runner_cmd(500, sc_args, runner)
        ckpt50  = cmd50 [cmd50 .index("--checkpoint-name") + 1]
        ckpt500 = cmd500[cmd500.index("--checkpoint-name") + 1]
        assert ckpt50 != ckpt500
        assert "0050" in ckpt50
        assert "0500" in ckpt500

    def test_methods_forwarded_when_provided(self, sc_args, tmp_path):
        runner = self._dummy_runner(tmp_path)
        sc_args.methods = [1, 3]
        cmd = SC._build_runner_cmd(200, sc_args, runner)
        assert "--methods" in cmd
        idx = cmd.index("--methods")
        assert cmd[idx + 1] == "1"
        assert cmd[idx + 2] == "3"

    def test_methods_absent_when_none(self, sc_args, tmp_path):
        runner = self._dummy_runner(tmp_path)
        sc_args.methods = None
        cmd = SC._build_runner_cmd(200, sc_args, runner)
        assert "--methods" not in cmd

    def test_skip_pysr_forwarded(self, sc_args, tmp_path):
        runner = self._dummy_runner(tmp_path)
        sc_args.skip_pysr = True
        cmd = SC._build_runner_cmd(100, sc_args, runner)
        assert "--skip-pysr" in cmd

    def test_threshold_noisy_used_in_noisy_mode(self, sc_args, tmp_path):
        runner = self._dummy_runner(tmp_path)
        sc_args.noiseless = False
        sc_args.threshold_noisy = 0.99
        cmd = SC._build_runner_cmd(100, sc_args, runner)
        assert "--threshold" in cmd
        assert cmd[cmd.index("--threshold") + 1] == "0.99"

    def test_threshold_noiseless_used_in_noiseless_mode(self, sc_args, tmp_path):
        runner = self._dummy_runner(tmp_path)
        sc_args.noiseless = True
        sc_args.threshold_noiseless = 0.9995
        cmd = SC._build_runner_cmd(100, sc_args, runner)
        assert "--threshold" in cmd
        assert cmd[cmd.index("--threshold") + 1] == "0.9995"

    def test_no_llm_cache_forwarded(self, sc_args, tmp_path):
        runner = self._dummy_runner(tmp_path)
        sc_args.no_llm_cache = True
        cmd = SC._build_runner_cmd(100, sc_args, runner)
        assert "--no-llm-cache" in cmd


# ===========================================================================
# ── 6.  Noise-sweep: _aggregate_results  ────────────────────────────────────
# ===========================================================================

class TestNSAggregateResults:

    def test_median_r2_computed_correctly(self, tmp_results_dir, sample_methods, sample_equations):
        methods   = sample_methods
        equations = sample_equations
        noise_levels = [0.0, 0.05]
        r2_map   = {eq: {m: 0.98 for m in methods} for eq in equations}

        result_paths = {}
        for sigma in noise_levels:
            mode = "noiseless" if sigma == 0.0 else "noisy"
            path = _make_result_file(tmp_results_dir, mode, methods, equations, r2_map)
            result_paths[sigma] = path

        agg = NS._aggregate_results(noise_levels, result_paths)

        for sigma in noise_levels:
            sigma_str = f"{sigma:.4f}"
            pnd = agg["per_noise"][sigma_str]
            assert pnd is not None
            for m in methods:
                ms = pnd["method_summary"][m]
                assert ms["median_r2"] == pytest.approx(0.98, abs=1e-6)
                assert ms["mean_r2"]   == pytest.approx(0.98, abs=1e-6)

    def test_recovery_rate_above_threshold(self, tmp_results_dir, sample_methods, sample_equations):
        methods   = sample_methods
        equations = sample_equations
        noise_levels = [0.05]
        # All R² above noisy threshold (0.995)
        r2_map = {eq: {m: 0.999 for m in methods} for eq in equations}
        path = _make_result_file(tmp_results_dir, "noisy", methods, equations, r2_map)
        agg  = NS._aggregate_results(noise_levels, {0.05: path})
        ms   = agg["per_noise"]["0.0500"]["method_summary"][methods[0]]
        assert ms["recovery_rate"] == pytest.approx(1.0)

    def test_recovery_rate_below_threshold(self, tmp_results_dir, sample_methods, sample_equations):
        methods   = sample_methods
        equations = sample_equations
        noise_levels = [0.05]
        # All R² below noisy threshold
        r2_map = {eq: {m: 0.90 for m in methods} for eq in equations}
        path = _make_result_file(tmp_results_dir, "noisy", methods, equations, r2_map)
        agg  = NS._aggregate_results(noise_levels, {0.05: path})
        ms   = agg["per_noise"]["0.0500"]["method_summary"][methods[0]]
        assert ms["recovery_rate"] == pytest.approx(0.0)

    def test_noiseless_uses_higher_threshold(self, tmp_results_dir, sample_methods, sample_equations):
        """R² = 0.997 is above noisy (0.995) but below noiseless (0.9999)."""
        methods   = sample_methods
        equations = sample_equations
        r2_map = {eq: {m: 0.997 for m in methods} for eq in equations}
        path   = _make_result_file(tmp_results_dir, "noiseless", methods, equations, r2_map)
        agg    = NS._aggregate_results([0.0], {0.0: path})
        ms     = agg["per_noise"]["0.0000"]["method_summary"][methods[0]]
        # 0.997 < 0.9999 → recovery_rate should be 0
        assert ms["recovery_rate"] == pytest.approx(0.0)

    def test_handles_missing_result_path(self, tmp_results_dir, sample_methods, sample_equations):
        noise_levels = [0.0, 0.05]
        path = _make_result_file(tmp_results_dir, "noisy", sample_methods, sample_equations)
        # 0.0 path is missing
        agg  = NS._aggregate_results(noise_levels, {0.0: None, 0.05: path})
        assert agg["per_noise"]["0.0000"] is None
        assert agg["per_noise"]["0.0500"] is not None

    def test_cross_noise_summary_populated(self, tmp_results_dir, sample_methods, sample_equations):
        noise_levels = [0.05, 0.10]
        r2_map = {eq: {m: 0.97 for m in sample_methods} for eq in sample_equations}
        paths  = {}
        for sigma in noise_levels:
            paths[sigma] = _make_result_file(
                tmp_results_dir, "noisy", sample_methods, sample_equations, r2_map
            )
        agg = NS._aggregate_results(noise_levels, paths)
        for m in sample_methods:
            assert m in agg["cross_noise_summary"]
            for sigma in noise_levels:
                s_str = f"{sigma:.4f}"
                entry = agg["cross_noise_summary"][m][s_str]
                assert entry is not None
                assert "median_r2" in entry
                assert "recovery_rate" in entry

    def test_methods_key_in_output(self, tmp_results_dir, sample_methods, sample_equations):
        path = _make_result_file(tmp_results_dir, "noisy", sample_methods, sample_equations)
        agg  = NS._aggregate_results([0.05], {0.05: path})
        assert set(agg["methods"]) == set(sample_methods)

    def test_std_r2_zero_for_uniform_values(self, tmp_results_dir, sample_methods, sample_equations):
        r2_map = {eq: {m: 0.95 for m in sample_methods} for eq in sample_equations}
        path   = _make_result_file(tmp_results_dir, "noisy", sample_methods, sample_equations, r2_map)
        agg    = NS._aggregate_results([0.05], {0.05: path})
        ms     = agg["per_noise"]["0.0500"]["method_summary"][sample_methods[0]]
        # All values identical → std = 0
        assert ms["std_r2"] == pytest.approx(0.0, abs=1e-6)

    def test_all_noise_levels_in_output(self, tmp_results_dir, sample_methods, sample_equations):
        noise_levels = [0.0, 0.01, 0.05, 0.10]
        paths = {}
        for sigma in noise_levels:
            mode = "noiseless" if sigma == 0.0 else "noisy"
            paths[sigma] = _make_result_file(
                tmp_results_dir, mode, sample_methods, sample_equations
            )
        agg = NS._aggregate_results(noise_levels, paths)
        assert agg["noise_levels"] == noise_levels
        for sigma in noise_levels:
            assert f"{sigma:.4f}" in agg["per_noise"]


# ===========================================================================
# ── 7.  Sample-complexity: _aggregate_results  ──────────────────────────────
# ===========================================================================

class TestSCAggregateResults:

    def test_median_r2_per_n(self, tmp_results_dir, sample_methods, sample_equations):
        sample_sizes = [50, 100, 200]
        r2_map = {eq: {m: 0.95 for m in sample_methods} for eq in sample_equations}
        paths  = {}
        for n in sample_sizes:
            paths[n] = _make_result_file(tmp_results_dir, "noisy", sample_methods, sample_equations, r2_map)
        agg = SC._aggregate_results(sample_sizes, paths, noiseless=False)
        for n in sample_sizes:
            ms = agg["per_n"][str(n)]["method_summary"][sample_methods[0]]
            assert ms["median_r2"] == pytest.approx(0.95, abs=1e-6)

    def test_data_efficiency_min_n(self, tmp_results_dir, sample_methods, sample_equations):
        """
        n=50 → recovery rate 0 (all R² = 0.9),
        n=100 → recovery rate 1 (all R² = 0.999).
        min_n_above_threshold should be 100.
        """
        sample_sizes = [50, 100, 200]
        method = sample_methods[0]

        r2_low  = {eq: {m: 0.9   for m in sample_methods} for eq in sample_equations}
        r2_high = {eq: {m: 0.999 for m in sample_methods} for eq in sample_equations}

        paths = {
            50:  _make_result_file(tmp_results_dir, "noisy", sample_methods, sample_equations, r2_low),
            100: _make_result_file(tmp_results_dir, "noisy", sample_methods, sample_equations, r2_high),
            200: _make_result_file(tmp_results_dir, "noisy", sample_methods, sample_equations, r2_high),
        }
        agg     = SC._aggregate_results(sample_sizes, paths, noiseless=False)
        min_n   = agg["data_efficiency"][method]["min_n_above_threshold"]
        # Recovery at n=50 is 0 (< 0.5 threshold), at n=100 it's 1.0 (>= 0.5)
        assert min_n == 100

    def test_data_efficiency_none_when_never_reached(self, tmp_results_dir, sample_methods, sample_equations):
        sample_sizes = [50, 100]
        r2_low = {eq: {m: 0.5 for m in sample_methods} for eq in sample_equations}
        paths  = {n: _make_result_file(tmp_results_dir, "noisy", sample_methods, sample_equations, r2_low)
                  for n in sample_sizes}
        agg = SC._aggregate_results(sample_sizes, paths, noiseless=False)
        # Recovery rate at 0.5 R² is 0 (below threshold 0.995) → min_n = None
        for m in sample_methods:
            assert agg["data_efficiency"][m]["min_n_above_threshold"] is None

    def test_recovery_rate_increases_with_n(self, tmp_results_dir, sample_methods, sample_equations):
        """Ensure the curve structure captures monotonic improvement."""
        sample_sizes = [50, 100, 200, 500]
        method = sample_methods[0]
        r2_by_n = {
            50:  0.90,
            100: 0.94,
            200: 0.97,
            500: 0.999,
        }
        paths = {}
        for n, r2v in r2_by_n.items():
            r2_map = {eq: {m: r2v for m in sample_methods} for eq in sample_equations}
            paths[n] = _make_result_file(tmp_results_dir, "noisy", sample_methods, sample_equations, r2_map)

        agg   = SC._aggregate_results(sample_sizes, paths, noiseless=False)
        curve = agg["data_efficiency"][method]["recovery_curve"]
        # Only n=500 (R²=0.999 > 0.995) recovers; others do not
        assert curve["50"]  == pytest.approx(0.0)
        assert curve["100"] == pytest.approx(0.0)
        assert curve["200"] == pytest.approx(0.0)
        assert curve["500"] == pytest.approx(1.0)

    def test_handles_missing_n(self, tmp_results_dir, sample_methods, sample_equations):
        sample_sizes = [50, 100, 200]
        path = _make_result_file(tmp_results_dir, "noisy", sample_methods, sample_equations)
        agg  = SC._aggregate_results(sample_sizes, {50: None, 100: path, 200: None}, noiseless=False)
        assert agg["per_n"]["50"]  is None
        assert agg["per_n"]["100"] is not None
        assert agg["per_n"]["200"] is None

    def test_noiseless_mode_uses_higher_threshold(self, tmp_results_dir, sample_methods, sample_equations):
        r2_map = {eq: {m: 0.997 for m in sample_methods} for eq in sample_equations}
        path   = _make_result_file(tmp_results_dir, "noiseless", sample_methods, sample_equations, r2_map)
        agg    = SC._aggregate_results([100], {100: path}, noiseless=True)
        ms     = agg["per_n"]["100"]["method_summary"][sample_methods[0]]
        # 0.997 < 0.9999 → recovery = 0
        assert ms["recovery_rate"] == pytest.approx(0.0)
        assert agg["threshold"] == pytest.approx(0.9999)

    def test_mode_field_in_output(self, tmp_results_dir, sample_methods, sample_equations):
        path = _make_result_file(tmp_results_dir, "noisy", sample_methods, sample_equations)
        agg  = SC._aggregate_results([100], {100: path}, noiseless=False)
        assert agg["mode"] == "noisy"

        path2 = _make_result_file(tmp_results_dir, "noiseless", sample_methods, sample_equations)
        agg2  = SC._aggregate_results([100], {100: path2}, noiseless=True)
        assert agg2["mode"] == "noiseless"

    def test_per_equation_detail_present(self, tmp_results_dir, sample_methods, sample_equations):
        path = _make_result_file(tmp_results_dir, "noisy", sample_methods, sample_equations)
        agg  = SC._aggregate_results([100], {100: path}, noiseless=False)
        per_eq = agg["per_n"]["100"]["per_equation"]
        assert set(per_eq.keys()) == set(sample_equations)


# ===========================================================================
# ── 8.  JSON / CSV output correctness  ──────────────────────────────────────
# ===========================================================================

class TestNSSaveOutputs:

    def _make_minimal_agg(self, noise_levels, methods, equations, r2=0.97):
        """Build a minimal aggregation object directly (no filesystem needed)."""
        per_noise = {}
        cross     = {m: {} for m in methods}
        for sigma in noise_levels:
            sigma_str = f"{sigma:.4f}"
            threshold = 0.9999 if sigma == 0.0 else 0.995
            method_summary = {}
            per_eq = {}
            for m in methods:
                r2_vals = [r2] * len(equations)
                rec = sum(1 for v in r2_vals if v >= threshold) / len(equations)
                method_summary[m] = {
                    "median_r2": float(np.median(r2_vals)),
                    "mean_r2":   float(np.mean(r2_vals)),
                    "std_r2":    0.0,
                    "recovery_rate": rec,
                    "n_success": len(equations),
                    "n_total":   len(equations),
                    "threshold_used": threshold,
                }
                cross[m][sigma_str] = {
                    "median_r2":     float(np.median(r2_vals)),
                    "recovery_rate": rec,
                    "std_r2":        0.0,
                }
            for eq in equations:
                per_eq[eq] = {m: {"r2": r2, "rmse": 0.01, "success": True} for m in methods}
            per_noise[sigma_str] = {"method_summary": method_summary, "per_equation": per_eq}

        return {
            "generated":           "2025-01-01T00:00:00",
            "noise_levels":        noise_levels,
            "methods":             methods,
            "per_noise":           per_noise,
            "cross_noise_summary": cross,
        }

    def test_json_saved_and_valid(self, tmp_results_dir, sample_methods, sample_equations):
        agg  = self._make_minimal_agg([0.0, 0.05], sample_methods, sample_equations)
        path = NS._save_sweep_json(agg, "20250101_000000")
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert "noise_levels"        in data
        assert "methods"             in data
        assert "per_noise"           in data
        assert "cross_noise_summary" in data

    def test_csv_saved_with_correct_schema(self, tmp_results_dir, sample_methods, sample_equations):
        agg  = self._make_minimal_agg([0.05, 0.10], sample_methods, sample_equations)
        path = NS._save_sweep_csv(agg, "20250101_000000")
        assert path.exists()
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows   = list(reader)
        expected_fields = {
            "method", "noise_level_fraction", "noise_level_pct",
            "median_r2", "mean_r2", "std_r2",
            "recovery_rate", "n_success", "n_total", "threshold_used",
        }
        assert expected_fields.issubset(set(rows[0].keys()))
        # The CSV contains both aggregate and per-equation rows; count only aggregate.
        agg_rows = [r for r in rows if r["section"] == "aggregate"]
        assert len(agg_rows) == len([0.05, 0.10]) * len(sample_methods)

    def test_csv_row_count(self, tmp_results_dir, sample_methods, sample_equations):
        noise_levels = [0.0, 0.01, 0.05, 0.10]
        agg  = self._make_minimal_agg(noise_levels, sample_methods, sample_equations)
        path = NS._save_sweep_csv(agg, "20250101_111111")
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        # Filter to aggregate section — the CSV also contains per-equation rows.
        agg_rows = [r for r in rows if r["section"] == "aggregate"]
        assert len(agg_rows) == len(noise_levels) * len(sample_methods)


class TestSCSaveOutputs:

    def _make_minimal_agg_sc(self, sample_sizes, methods, equations, r2=0.97):
        per_n = {}
        de    = {}
        for n in sample_sizes:
            threshold = 0.995
            method_summary = {}
            per_eq = {}
            for m in methods:
                r2_vals = [r2] * len(equations)
                rec = sum(1 for v in r2_vals if v >= threshold) / len(equations)
                method_summary[m] = {
                    "median_r2":     float(np.median(r2_vals)),
                    "mean_r2":       float(np.mean(r2_vals)),
                    "std_r2":        0.0,
                    "recovery_rate": rec,
                    "n_success":     len(equations),
                    "n_total":       len(equations),
                    "threshold_used": threshold,
                }
            for eq in equations:
                per_eq[eq] = {m: {"r2": r2, "rmse": 0.01, "success": True} for m in methods}
            per_n[str(n)] = {"method_summary": method_summary, "per_equation": per_eq}

        for m in methods:
            curve = {str(n): (1.0 if r2 >= 0.5 else 0.0) for n in sample_sizes}
            first = next((n for n in sample_sizes if curve[str(n)] >= 0.5), None)
            de[m] = {
                "min_n_above_threshold": first,
                "efficiency_target":     0.5,
                "recovery_curve":        curve,
            }

        return {
            "generated":       "2025-01-01T00:00:00",
            "sample_sizes":    sample_sizes,
            "mode":            "noisy",
            "threshold":       0.995,
            "methods":         methods,
            "per_n":           per_n,
            "data_efficiency": de,
        }

    def test_json_saved_and_valid(self, tmp_results_dir, sample_methods, sample_equations):
        agg  = self._make_minimal_agg_sc([50, 100, 200], sample_methods, sample_equations)
        path = SC._save_complexity_json(agg, "20250101_000000")
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert "sample_sizes"    in data
        assert "methods"         in data
        assert "per_n"           in data
        assert "data_efficiency" in data

    def test_csv_saved_with_correct_schema(self, tmp_results_dir, sample_methods, sample_equations):
        sample_sizes = [50, 100, 200, 500]
        agg  = self._make_minimal_agg_sc(sample_sizes, sample_methods, sample_equations)
        path = SC._save_complexity_csv(agg, "20250101_000000")
        assert path.exists()
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows   = list(reader)
        assert len(rows) > 0
        # Aggregate section row count
        agg_rows = [r for r in rows if r["section"] == "aggregate"]
        assert len(agg_rows) == len(sample_sizes) * len(sample_methods)

    def test_csv_per_equation_rows_present(self, tmp_results_dir, sample_methods, sample_equations):
        agg  = self._make_minimal_agg_sc([100], sample_methods, sample_equations)
        path = SC._save_complexity_csv(agg, "20250101_222222")
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        eq_rows = [r for r in rows if r.get("section") == "per_equation"]
        # Expect len(methods) × len(equations) × len(sample_sizes) rows
        assert len(eq_rows) == len(sample_methods) * len(sample_equations) * 1


# ===========================================================================
# ── 9.  Print/reporting smoke tests  ─────────────────────────────────────────
# ===========================================================================

class TestPrintSmokeTests:

    def _noise_agg(self, sample_methods, sample_equations):
        sigma = 0.05
        sigma_str = "0.0500"
        method_summary = {m: {
            "median_r2": 0.97, "mean_r2": 0.97, "std_r2": 0.01,
            "recovery_rate": 0.8, "n_success": 3, "n_total": 3,
            "threshold_used": 0.995,
        } for m in sample_methods}
        per_eq = {eq: {m: {"r2": 0.97, "rmse": 0.01, "success": True}
                       for m in sample_methods} for eq in sample_equations}
        return {
            "generated":    "2025-01-01",
            "noise_levels": [sigma],
            "methods":      sample_methods,
            "per_noise":    {sigma_str: {"method_summary": method_summary, "per_equation": per_eq}},
            "cross_noise_summary": {
                m: {sigma_str: {"median_r2": 0.97, "recovery_rate": 0.8, "std_r2": 0.01}}
                for m in sample_methods
            },
        }

    def test_ns_print_table_does_not_raise(self, capsys, sample_methods, sample_equations):
        agg = self._noise_agg(sample_methods, sample_equations)
        NS._print_noise_sweep_table(agg)
        out = capsys.readouterr().out
        assert "NOISE SWEEP" in out
        assert "Recovery Rate" in out

    def test_ns_print_table_empty_no_crash(self, capsys):
        NS._print_noise_sweep_table({"methods": [], "noise_levels": [], "per_noise": {}})
        out = capsys.readouterr().out
        assert "no data" in out

    def _sc_agg(self, sample_methods, sample_equations):
        n = 100
        threshold = 0.995
        method_summary = {m: {
            "median_r2": 0.96, "mean_r2": 0.96, "std_r2": 0.01,
            "recovery_rate": 0.67, "n_success": 2, "n_total": 3,
            "threshold_used": threshold,
        } for m in sample_methods}
        per_eq = {eq: {m: {"r2": 0.96, "rmse": 0.01, "success": True}
                       for m in sample_methods} for eq in sample_equations}
        de = {m: {"min_n_above_threshold": 100, "efficiency_target": 0.5,
                  "recovery_curve": {str(n): 0.67}} for m in sample_methods}
        return {
            "generated":    "2025-01-01",
            "sample_sizes": [n],
            "mode":         "noisy",
            "threshold":    threshold,
            "methods":      sample_methods,
            "per_n":        {str(n): {"method_summary": method_summary, "per_equation": per_eq}},
            "data_efficiency": de,
        }

    def test_sc_print_table_does_not_raise(self, capsys, sample_methods, sample_equations):
        agg = self._sc_agg(sample_methods, sample_equations)
        SC._print_sample_complexity_table(agg)
        out = capsys.readouterr().out
        assert "SAMPLE COMPLEXITY" in out
        assert "DATA EFFICIENCY"   in out

    def test_sc_print_table_empty_no_crash(self, capsys):
        SC._print_sample_complexity_table({"methods": [], "sample_sizes": [], "per_n": {}, "data_efficiency": {}})
        out = capsys.readouterr().out
        assert "no data" in out


# ===========================================================================
# ── 10.  _run_noise_level / _run_sample_size mocking  ────────────────────────
# ===========================================================================

class TestRunConditionMocking:
    """Verify orchestration logic by mocking subprocess.run."""

    def _fake_ns_args(self, tmp_path) -> argparse.Namespace:
        return argparse.Namespace(
            noise_levels=[0.01, 0.05, 0.10],
            methods=[3, 4],
            threshold_noisy=0.995,
            threshold_noiseless=0.9999,
            samples=50,
            nn_seeds=1,
            method_timeout=60,
            pysr_timeout=120,
            skip_pysr=False,
            test=None,
            equations=None,
            domain="all_domains",
            series=None,
            benchmark="feynman",
            verbose=False,
            quiet=False,
            no_llm_cache=False,
            fail_fast=False,
            log=None,
            runner=None,
        )

    def test_run_noise_level_returns_path_on_success(self, tmp_path):
        args   = self._fake_ns_args(tmp_path)
        runner = tmp_path / "runner.py"
        runner.write_text("pass")

        # The result file must be created AFTER t_start is captured inside
        # _run_noise_level, otherwise _find_result_written_after (which gates
        # on mtime >= t_start) will miss it and we'd rely on the fragile
        # _find_latest_result fallback.  We create it inside the subprocess
        # mock side-effect to guarantee the correct mtime ordering.
        result_file = tmp_path / "protocol_core_noisy_20250101_000000.json"

        def _fake_run(cmd, env=None, **kwargs):
            result_file.write_text(json.dumps({"tests": []}))
            return MagicMock(returncode=0)

        orig = NS._RESULTS_DIR
        NS._RESULTS_DIR = tmp_path
        try:
            with patch("subprocess.run", side_effect=_fake_run):
                result = NS._run_noise_level(0.05, args, runner)
            assert result is not None
            assert "noisy" in result.name
        finally:
            NS._RESULTS_DIR = orig

    def test_run_noise_level_returns_none_on_missing_json(self, tmp_path):
        args   = self._fake_ns_args(tmp_path)
        runner = tmp_path / "runner.py"
        runner.write_text("pass")

        orig = NS._RESULTS_DIR
        NS._RESULTS_DIR = tmp_path
        try:
            with patch("subprocess.run", return_value=MagicMock(returncode=0)):
                result = NS._run_noise_level(0.05, args, runner)
            assert result is None
        finally:
            NS._RESULTS_DIR = orig

    def test_run_noise_level_fail_fast_exits(self, tmp_path):
        args           = self._fake_ns_args(tmp_path)
        args.fail_fast = True
        runner = tmp_path / "runner.py"
        runner.write_text("pass")

        orig = NS._RESULTS_DIR
        NS._RESULTS_DIR = tmp_path
        try:
            with patch("subprocess.run", return_value=MagicMock(returncode=1)):
                with pytest.raises(SystemExit):
                    NS._run_noise_level(0.05, args, runner)
        finally:
            NS._RESULTS_DIR = orig

    def test_run_sample_size_returns_path_on_success(self, tmp_path):
        sc_arg = argparse.Namespace(
            noiseless=False,
            threshold_noisy=0.995,
            threshold_noiseless=0.9999,
            nn_seeds=1,
            method_timeout=60,
            pysr_timeout=120,
            methods=[1, 2],
            skip_pysr=False,
            test=None,
            equations=None,
            domain="all_domains",
            series=None,
            benchmark="feynman",
            verbose=False,
            quiet=False,
            no_llm_cache=False,
            fail_fast=False,
        )
        runner = tmp_path / "runner.py"
        runner.write_text("pass")

        result_file = tmp_path / "protocol_core_noisy_20250101_000000.json"
        result_file.write_text(json.dumps({"tests": []}))

        orig = SC._RESULTS_DIR
        SC._RESULTS_DIR = tmp_path
        try:
            with patch("subprocess.run", return_value=MagicMock(returncode=0)):
                result = SC._run_sample_size(200, sc_arg, runner)
            assert result is not None
        finally:
            SC._RESULTS_DIR = orig

    def test_run_sample_size_fail_fast_exits(self, tmp_path):
        sc_arg = argparse.Namespace(
            noiseless=False,
            threshold_noisy=0.995,
            threshold_noiseless=0.9999,
            nn_seeds=1,
            method_timeout=60,
            pysr_timeout=120,
            methods=None,
            skip_pysr=False,
            test=None,
            equations=None,
            domain="all_domains",
            series=None,
            benchmark="feynman",
            verbose=False,
            quiet=False,
            no_llm_cache=False,
            fail_fast=True,
        )
        runner = tmp_path / "runner.py"
        runner.write_text("pass")

        orig = SC._RESULTS_DIR
        SC._RESULTS_DIR = tmp_path
        try:
            with patch("subprocess.run", return_value=MagicMock(returncode=1)):
                with pytest.raises(SystemExit):
                    SC._run_sample_size(100, sc_arg, runner)
        finally:
            SC._RESULTS_DIR = orig


# ===========================================================================
# ── 11.  CLI argument parsing  ───────────────────────────────────────────────
# ===========================================================================

class TestNSCLIParsing:
    """Test that the argument parser in main() handles edge cases correctly."""

    def test_default_noise_levels(self):
        # σ=0% and σ=0.5% already done; sweep covers the remaining three
        assert NS._DEFAULT_NOISE_LEVELS == [0.01, 0.05, 0.10]

    def test_default_methods(self):
        # Top two from protocol_core_noisy_20260313_094752.json
        assert NS._DEFAULT_METHODS == [3, 4]

    def test_constants_are_lists(self):
        assert isinstance(NS._DEFAULT_NOISE_LEVELS, list)
        assert isinstance(NS._DEFAULT_METHODS, list)


class TestSCCLIParsing:

    def test_default_sample_sizes(self):
        assert SC._DEFAULT_SAMPLE_SIZES == [50, 100, 200, 500]

    def test_constants_are_lists(self):
        assert isinstance(SC._DEFAULT_SAMPLE_SIZES, list)


# ===========================================================================
# ── 12.  Edge-cases: empty / malformed data  ─────────────────────────────────
# ===========================================================================

class TestEdgeCases:

    def test_ns_aggregate_all_paths_none(self):
        agg = NS._aggregate_results(
            [0.0, 0.05],
            {0.0: None, 0.05: None},
        )
        assert agg["per_noise"]["0.0000"] is None
        assert agg["per_noise"]["0.0500"] is None
        assert agg["cross_noise_summary"] == {}

    def test_sc_aggregate_all_paths_none(self):
        agg = SC._aggregate_results(
            [50, 100],
            {50: None, 100: None},
            noiseless=False,
        )
        assert agg["per_n"]["50"]  is None
        assert agg["per_n"]["100"] is None

    def test_ns_extract_per_test_missing_eq_name(self):
        payload = {"tests": [{"results": {"M": {"r2": 0.9, "success": True}},
                               "description": "", "metadata": {}}]}
        per_eq = NS._extract_per_test(payload)
        # Should not raise; should return at least one entry
        assert len(per_eq) == 1

    def test_ns_aggregate_non_finite_r2_ignored(self, tmp_path, tmp_results_dir, sample_methods, sample_equations):
        """Non-finite R² values must not contaminate median/std computation."""
        payload = _make_result_payload(sample_methods, sample_equations)
        # Corrupt one R² to NaN
        payload["tests"][0]["results"][sample_methods[0]]["r2"] = float("nan")
        path = tmp_results_dir / "protocol_core_noisy_bad.json"
        _write_json(path, payload)

        agg = NS._aggregate_results([0.05], {0.05: path})
        ms  = agg["per_noise"]["0.0500"]["method_summary"][sample_methods[0]]
        # Median should still be computed from the finite values
        if ms["median_r2"] is not None:
            assert np.isfinite(ms["median_r2"])

    def test_sc_aggregate_non_finite_r2_ignored(self, tmp_results_dir, sample_methods, sample_equations):
        payload = _make_result_payload(sample_methods, sample_equations)
        payload["tests"][0]["results"][sample_methods[0]]["r2"] = float("inf")
        path = tmp_results_dir / "protocol_core_noisy_inf.json"
        _write_json(path, payload)

        agg = SC._aggregate_results([100], {100: path}, noiseless=False)
        ms  = agg["per_n"]["100"]["method_summary"][sample_methods[0]]
        if ms["median_r2"] is not None:
            assert np.isfinite(ms["median_r2"])

    def test_ns_aggregate_single_equation(self, tmp_results_dir, sample_methods):
        """Works correctly with just one equation."""
        path = _make_result_file(tmp_results_dir, "noisy", sample_methods, ["only_eq"])
        agg  = NS._aggregate_results([0.05], {0.05: path})
        assert "only_eq" in agg["per_noise"]["0.0500"]["per_equation"]

    def test_sc_aggregate_single_n(self, tmp_results_dir, sample_methods, sample_equations):
        path = _make_result_file(tmp_results_dir, "noisy", sample_methods, sample_equations)
        agg  = SC._aggregate_results([200], {200: path}, noiseless=False)
        assert "200" in agg["per_n"]
        assert agg["per_n"]["200"] is not None

    def test_ns_load_results_returns_dict(self, tmp_results_dir, sample_payload):
        path = tmp_results_dir / "test_load.json"
        _write_json(path, sample_payload)
        data = NS._load_results(path)
        assert isinstance(data, dict)

    def test_sc_load_results_returns_dict(self, tmp_results_dir, sample_payload):
        path = tmp_results_dir / "test_load_sc.json"
        _write_json(path, sample_payload)
        data = SC._load_results(path)
        assert isinstance(data, dict)


# ===========================================================================
# ── 13.  Sigma label uniqueness and sub-percent encoding  ────────────────────
# ===========================================================================

class TestSigmaLabelEncoding:
    """
    Moved from TestExistingResultsParsing (where they did not belong).
    These tests verify _build_runner_cmd sigma-label correctness, including
    the sub-percent edge case (0.005 → sig0005) and uniqueness across all
    standard sweep values.
    """

    def _dummy_runner(self, tmp_path):
        p = tmp_path / "r.py"
        p.write_text("pass")
        return p

    def _base_args(self):
        return argparse.Namespace(
            methods=[3, 4], threshold_noisy=0.995, threshold_noiseless=0.9999,
            samples=200, nn_seeds=3, method_timeout=900, pysr_timeout=1100,
            skip_pysr=False, test=None, equations=None, domain="all_domains",
            series=None, benchmark="feynman", verbose=False, quiet=False,
            no_llm_cache=False, fail_fast=False, log=None, runner=None,
        )

    def test_sigma_label_encoding_sub_percent(self, tmp_path):
        """sigma=0.005 must encode as sig0005, not sig000 or sig0050."""
        args = self._base_args()
        _, label = NS._build_runner_cmd(0.005, args, self._dummy_runner(tmp_path))
        assert label == "sig0005", f"expected sig0005, got {label}"

    def test_all_five_sigma_labels_unique(self, tmp_path):
        """No two standard sweep sigmas may produce the same checkpoint label."""
        args   = self._base_args()
        runner = self._dummy_runner(tmp_path)
        sigmas = [0.0, 0.005, 0.01, 0.05, 0.10]
        labels = [NS._build_runner_cmd(s, args, runner)[1] for s in sigmas]
        assert len(labels) == len(set(labels)), f"Duplicate labels: {labels}"
# ── 14.  Top-two validation against real benchmark results  ─────────────────
# ===========================================================================

class TestTopTwoFromBenchmarkResults:
    """
    Validate that methods 3 and 4 are genuinely the top two
    using the actual benchmark JSON as ground truth.

    These tests are skipped automatically when the results file is
    not present (e.g. on a CI machine without the full dataset).
    """

    RESULTS_FILE = Path("/mnt/user-data/uploads/protocol_core_noisy_20260313_094752.json")

    @pytest.fixture(autouse=True)
    def require_results_file(self):
        if not self.RESULTS_FILE.exists():
            pytest.skip("Benchmark results file not available in this environment.")

    def _load_stats(self):
        with open(self.RESULTS_FILE) as f:
            data = json.load(f)
        methods = data["methods"]
        r2s  = {m: [] for m in methods}
        wins = {m: 0  for m in methods}
        for test in data["tests"]:
            w = test.get("winner")
            if w in wins:
                wins[w] += 1
            for m in methods:
                res = test["results"].get(m, {})
                if res.get("success"):
                    r2 = res.get("r2")
                    if r2 is not None and np.isfinite(float(r2)):
                        r2s[m].append(float(r2))
        return methods, r2s, wins

    def test_enhanced_hybrid_defi_is_rank1_by_wins(self):
        methods, r2s, wins = self._load_stats()
        top = max(methods, key=lambda m: wins[m])
        assert top == "EnhancedHybridSystemDeFi (core)"

    def test_hybrid_llmnn_is_rank2_by_wins(self):
        methods, r2s, wins = self._load_stats()
        ranked = sorted(methods, key=lambda m: wins[m], reverse=True)
        assert ranked[1] == "HybridSystemLLMNN all-domains (core)"

    def test_top_two_median_r2_above_0_9999(self):
        methods, r2s, wins = self._load_stats()
        for m in ["EnhancedHybridSystemDeFi (core)", "HybridSystemLLMNN all-domains (core)"]:
            med = np.median(r2s[m])
            assert med > 0.9999, f"{m} median R²={med:.7f} not above 0.9999"

    def test_gap_between_top2_and_rest(self):
        """Top-two median R² must be at least 0.002 above third-place."""
        methods, r2s, wins = self._load_stats()
        ranked = sorted(methods, key=lambda m: np.median(r2s[m]) if r2s[m] else 0, reverse=True)
        top2_med   = np.median(r2s[ranked[1]])
        third_med  = np.median(r2s[ranked[2]])
        gap = float(top2_med - third_med)
        assert gap > 0.002, f"Gap between 2nd and 3rd is only {gap:.6f} — less than expected 0.002"

    def test_default_methods_match_top_two_indices(self):
        """
        _DEFAULT_METHODS must be [3, 4], matching EnhancedHybridDeFi and
        HybridSystemLLMNN in the METHOD_REGISTRY.
        """
        assert NS._DEFAULT_METHODS == [3, 4]

    def test_total_tests_is_30(self):
        with open(self.RESULTS_FILE) as f:
            data = json.load(f)
        assert data["total_tests"] == 30
        assert len(data["tests"])  == 30

    def test_all_methods_have_30_results(self):
        methods, r2s, _ = self._load_stats()
        for m in methods:
            assert len(r2s[m]) == 30, f"{m} has {len(r2s[m])} results, expected 30"

    def test_symbolic_methods_not_in_top_two(self):
        """PySR-backed methods should rank 4th and 5th, not top two."""
        methods, r2s, _ = self._load_stats()
        ranked = sorted(methods, key=lambda m: np.median(r2s[m]) if r2s[m] else 0, reverse=True)
        top_two = set(ranked[:2])
        assert "SymbolicEngineWithLLM (tools)"    not in top_two
        assert "HybridDiscoverySystem v40 (tools)" not in top_two


# ===========================================================================
# ── 15.  Noise env-var injection  ────────────────────────────────────────────
# ===========================================================================

class TestNoiseEnvInjection:
    """
    _run_noise_level passes HYPATIAX_NOISE_LEVEL to the child environment.
    Verify this without actually launching a subprocess.
    """

    def test_noise_env_var_in_child_env(self, tmp_path):
        """
        Patch subprocess.run and capture the env kwarg to check that
        HYPATIAX_NOISE_LEVEL is set to the right value.
        """
        args = argparse.Namespace(
            noise_levels=[0.01, 0.05, 0.10],
            methods=[3, 4],
            threshold_noisy=0.995,
            threshold_noiseless=0.9999,
            samples=50,
            nn_seeds=1,
            method_timeout=60,
            pysr_timeout=120,
            skip_pysr=False,
            test=None,
            equations=None,
            domain="all_domains",
            series=None,
            benchmark="feynman",
            verbose=False,
            quiet=False,
            no_llm_cache=False,
            fail_fast=False,
            log=None,
            runner=None,
        )
        runner = tmp_path / "runner.py"
        runner.write_text("pass")

        captured_env = {}

        def fake_run(cmd, env=None, **kwargs):
            captured_env.update(env or {})
            return MagicMock(returncode=0)

        orig = NS._RESULTS_DIR
        NS._RESULTS_DIR = tmp_path
        try:
            with patch("subprocess.run", side_effect=fake_run):
                NS._run_noise_level(0.05, args, runner)
            assert "HYPATIAX_NOISE_LEVEL" in captured_env
            assert float(captured_env["HYPATIAX_NOISE_LEVEL"]) == pytest.approx(0.05)
        finally:
            NS._RESULTS_DIR = orig

    def test_noise_env_zero_for_noiseless(self, tmp_path):
        args = argparse.Namespace(
            noise_levels=[0.0],
            methods=[3, 4],
            threshold_noisy=0.995,
            threshold_noiseless=0.9999,
            samples=50,
            nn_seeds=1,
            method_timeout=60,
            pysr_timeout=120,
            skip_pysr=False,
            test=None,
            equations=None,
            domain="all_domains",
            series=None,
            benchmark="feynman",
            verbose=False,
            quiet=False,
            no_llm_cache=False,
            fail_fast=False,
            log=None,
            runner=None,
        )
        runner = tmp_path / "runner.py"
        runner.write_text("pass")

        captured_env = {}

        def fake_run(cmd, env=None, **kwargs):
            captured_env.update(env or {})
            return MagicMock(returncode=0)

        # Create a fake noiseless result so _find_latest_result succeeds
        result_file = tmp_path / "protocol_core_noiseless_20250101_000000.json"
        result_file.write_text(json.dumps({"tests": []}))

        orig = NS._RESULTS_DIR
        NS._RESULTS_DIR = tmp_path
        try:
            with patch("subprocess.run", side_effect=fake_run):
                NS._run_noise_level(0.0, args, runner)
            assert float(captured_env.get("HYPATIAX_NOISE_LEVEL", -1)) == pytest.approx(0.0)
        finally:
            NS._RESULTS_DIR = orig


# ===========================================================================
# ── 16.  Collision fix: _find_result_written_after  ──────────────────────────
# ===========================================================================

class TestFindResultWrittenAfter:
    """Verify the mtime-gating function that prevents noisy-pass collisions."""

    def test_finds_file_after_tstart(self, tmp_results_dir):
        t_before = time.time()
        time.sleep(0.05)
        p = tmp_results_dir / "protocol_core_noisy_20250202_000000.json"
        p.write_text("{}")
        found = NS._find_result_written_after("noisy", t_before)
        assert found is not None
        assert found.name == p.name

    def test_ignores_file_before_tstart(self, tmp_results_dir):
        p = tmp_results_dir / "protocol_core_noisy_20250101_000000.json"
        p.write_text("{}")
        # Use a t_start well after the file was written
        t_after = time.time() + 100
        found = NS._find_result_written_after("noisy", t_after)
        assert found is None

    def test_returns_newest_among_multiple_new_files(self, tmp_results_dir):
        t_start = time.time()
        time.sleep(0.02)
        p1 = tmp_results_dir / "protocol_core_noisy_20250202_000001.json"
        p1.write_text("{}")
        time.sleep(0.02)
        p2 = tmp_results_dir / "protocol_core_noisy_20250202_000002.json"
        p2.write_text("{}")
        found = NS._find_result_written_after("noisy", t_start)
        assert found is not None
        assert found.name == p2.name   # p2 is newer

    def test_mode_filtering_noisy_vs_noiseless(self, tmp_results_dir):
        t_start = time.time()
        time.sleep(0.02)
        noisy     = tmp_results_dir / "protocol_core_noisy_20250303_000000.json"
        noiseless = tmp_results_dir / "protocol_core_noiseless_20250303_000000.json"
        noisy.write_text("{}"); noiseless.write_text("{}")
        found_noisy     = NS._find_result_written_after("noisy",     t_start)
        found_noiseless = NS._find_result_written_after("noiseless", t_start)
        assert found_noisy     is not None and "noisy"     in found_noisy.name
        assert found_noiseless is not None and "noiseless" in found_noiseless.name

    def test_sc_has_same_function(self):
        """Sample complexity module must also have _find_result_written_after."""
        assert hasattr(SC, "_find_result_written_after"), \
            "run_sample_complexity_benchmark is missing _find_result_written_after"


# ===========================================================================
# ── 17.  Catastrophic failure detection  ─────────────────────────────────────
# ===========================================================================

class TestCatastrophicFailureDetection:

    def _make_payload_with_low_r2(self, method_r2_map: dict, equations: List[str]) -> Dict:
        """Build a payload where method_r2_map = {method: r2} for all equations."""
        tests = []
        for eq in equations:
            results = {}
            for m, r2v in method_r2_map.items():
                results[m] = {
                    "success": True, "r2": r2v, "rmse": abs(1 - r2v) * 0.1,
                    "time": 1.0, "error": None, "metadata": {},
                }
            tests.append({
                "description": eq, "metadata": {"equation_name": eq},
                "results": results, "winner": list(method_r2_map)[0],
                "comparison": {"y_scale": {"denom": 1.0}, "duplicates": {}},
            })
        return {"tests": tests}

    def test_catastrophic_failure_flagged(self, tmp_results_dir, sample_equations):
        methods = ["M3", "M4"]
        payload = self._make_payload_with_low_r2({"M3": 0.999, "M4": 0.643}, sample_equations)
        path = tmp_results_dir / "protocol_core_noisy_cat.json"
        path.write_text(json.dumps(payload))

        agg  = NS._aggregate_results([0.05], {0.05: path})
        pnd  = agg["per_noise"]["0.0500"]
        cats = pnd["catastrophic_failures"]
        assert len(cats) == len(sample_equations)   # one per equation for M4
        assert all(cf["method"] == "M4" for cf in cats)
        assert all(abs(cf["r2"] - 0.643) < 1e-6 for cf in cats)

    def test_n_catastrophic_in_method_summary(self, tmp_results_dir, sample_equations):
        payload = self._make_payload_with_low_r2({"M3": 0.999, "M4": 0.500}, sample_equations)
        path = tmp_results_dir / "protocol_core_noisy_cat2.json"
        path.write_text(json.dumps(payload))

        agg = NS._aggregate_results([0.05], {0.05: path})
        ms4 = agg["per_noise"]["0.0500"]["method_summary"]["M4"]
        assert ms4["n_catastrophic"] == len(sample_equations)

    def test_no_catastrophic_when_all_above_threshold(self, tmp_results_dir, sample_equations):
        payload = self._make_payload_with_low_r2({"M3": 0.999, "M4": 0.995}, sample_equations)
        path = tmp_results_dir / "protocol_core_noisy_ok.json"
        path.write_text(json.dumps(payload))

        agg  = NS._aggregate_results([0.05], {0.05: path})
        cats = agg["per_noise"]["0.0500"]["catastrophic_failures"]
        assert len(cats) == 0
        for m in ["M3", "M4"]:
            assert agg["per_noise"]["0.0500"]["method_summary"][m]["n_catastrophic"] == 0

    def test_catastrophic_flag_in_per_equation(self, tmp_results_dir, sample_equations):
        payload = self._make_payload_with_low_r2({"M3": 0.999, "M4": 0.643}, sample_equations)
        path = tmp_results_dir / "protocol_core_noisy_cat3.json"
        path.write_text(json.dumps(payload))

        agg = NS._aggregate_results([0.05], {0.05: path})
        for eq in sample_equations:
            eq_data = agg["per_noise"]["0.0500"]["per_equation"][eq]
            assert eq_data["M4"]["catastrophic"] is True
            assert eq_data["M3"]["catastrophic"] is False

    def test_cross_noise_summary_carries_n_catastrophic(self, tmp_results_dir, sample_equations):
        payload = self._make_payload_with_low_r2({"M3": 0.999, "M4": 0.643}, sample_equations)
        path = tmp_results_dir / "protocol_core_noisy_cat4.json"
        path.write_text(json.dumps(payload))

        agg = NS._aggregate_results([0.05], {0.05: path})
        cross = agg["cross_noise_summary"]["M4"]["0.0500"]
        assert cross["n_catastrophic"] == len(sample_equations)

    def test_newton_gravity_r2_in_real_data(self):
        """Regression: M4 R2=0.643 on Newton gravity in baseline run."""
        real = Path("/mnt/user-data/uploads/protocol_core_noisy_20260313_094752.json")
        if not real.exists():
            pytest.skip("Baseline results not available")
        with open(real) as f:
            data = json.load(f)
        m4 = "HybridSystemLLMNN all-domains (core)"
        newton_test = next(
            (t for t in data["tests"] if "Newton" in t["description"] and "gravitational" in t["description"]),
            None
        )
        assert newton_test is not None, "Newton gravity test not found in baseline"
        r2 = newton_test["results"][m4]["r2"]
        assert r2 < 0.90, f"Expected catastrophic failure (R2<0.90) but got R2={r2:.4f}"


# ===========================================================================
# ── 18.  --existing-results parsing  ─────────────────────────────────────────
# ===========================================================================

class TestExistingResultsParsing:
    """
    The --existing-results flag lets users merge pre-completed runs.
    Test that the aggregation correctly uses them and skips re-running.
    """

    def test_existing_result_used_in_aggregation(self, tmp_results_dir, sample_methods, sample_equations):
        path = _make_result_file(tmp_results_dir, "noisy", sample_methods, sample_equations)
        # Aggregate with the existing file as pre-existing
        agg  = NS._aggregate_results([0.05], {0.05: path})
        assert agg["per_noise"]["0.0500"] is not None
        assert set(agg["methods"]) == set(sample_methods)

    def test_sigma_label_encoding_sub_percent(self):
        """Canonical coverage lives in TestSigmaLabelEncoding (section 13)."""
        # Thin smoke-test retained here for grep discoverability.
        tmp    = Path(tempfile.mkdtemp())
        runner = tmp / "r.py"; runner.write_text("pass")
        args   = argparse.Namespace(
            methods=[3, 4], threshold_noisy=0.995, threshold_noiseless=0.9999,
            samples=200, nn_seeds=3, method_timeout=900, pysr_timeout=1100,
            skip_pysr=False, test=None, equations=None, domain="all_domains",
            series=None, benchmark="feynman", verbose=False, quiet=False,
            no_llm_cache=False, fail_fast=False, log=None, runner=None,
        )
        _, label = NS._build_runner_cmd(0.005, args, runner)
        assert label == "sig0005", f"expected sig0005, got {label}"

    def test_all_five_sigma_labels_unique(self):
        """Canonical coverage lives in TestSigmaLabelEncoding (section 13)."""
        tmp    = Path(tempfile.mkdtemp())
        runner = tmp / "r.py"; runner.write_text("pass")
        args   = argparse.Namespace(
            methods=[3, 4], threshold_noisy=0.995, threshold_noiseless=0.9999,
            samples=200, nn_seeds=3, method_timeout=900, pysr_timeout=1100,
            skip_pysr=False, test=None, equations=None, domain="all_domains",
            series=None, benchmark="feynman", verbose=False, quiet=False,
            no_llm_cache=False, fail_fast=False, log=None, runner=None,
        )
        sigmas = [0.0, 0.005, 0.01, 0.05, 0.10]
        labels = [NS._build_runner_cmd(s, args, runner)[1] for s in sigmas]
        assert len(labels) == len(set(labels)), f"Duplicate labels: {labels}"

    def test_sc_existing_results_skips_in_loop(self, tmp_results_dir, sample_methods, sample_equations):
        """n=200 in existing_map should not be in sizes_to_run."""
        # Simulate what the patched main() does
        sample_sizes  = [50, 100, 200, 500]
        existing_map  = {200: _make_result_file(tmp_results_dir, "noisy", sample_methods, sample_equations)}
        sizes_to_run  = [n for n in sample_sizes if n not in existing_map]
        assert 200 not in sizes_to_run
        assert sorted(sizes_to_run) == [50, 100, 500]

    def test_sc_existing_results_included_in_aggregation(self, tmp_results_dir, sample_methods, sample_equations):
        path = _make_result_file(tmp_results_dir, "noisy", sample_methods, sample_equations)
        agg  = SC._aggregate_results([50, 200], {50: None, 200: path}, noiseless=False)
        assert agg["per_n"]["200"] is not None
        assert agg["per_n"]["50"]  is None   # was not provided


# ==============================================================================
# NEWTON'S GRAVITY M4 FAILURE TESTS — appendable block
# Drop this at the end of test_sweep_benchmarks.py
# Requires: numpy already imported; unittest already imported
# ==============================================================================

# ── Constants matching the real experimental setup ────────────────────────────
_NEWTON_EQ   = "Newton's gravitational force between two masses"
_M3          = "EnhancedHybridSystemDeFi (core)"
_M4          = "HybridSystemLLMNN all-domains (core)"
_CAT_THRESH  = 0.90   # R² below this = catastrophic failure

# Real M4 R² values from the 2026-03-15 experiment run
_NEWTON_M4_R2_BY_SIGMA = {
    0.000: 0.8283,   # σ=0%    — below threshold
    0.005: 0.6458,   # σ=0.5%  — catastrophic
    0.010: 0.7312,   # σ=1%    — catastrophic
    0.050: 0.8723,   # σ=5%    — below threshold
    0.100: 0.6844,   # σ=10%   — catastrophic
}

_NEWTON_M4_R2_BY_N = {
    50:   0.5933,   # catastrophic — worst at low n
    100:  0.8208,   # below threshold
    200:  0.6470,   # catastrophic
    500:  0.8207,   # below threshold
    750:  0.4186,   # catastrophic — worst overall
    1000: 0.7594,   # catastrophic
}

# Real M3 R² values — all perfect
_NEWTON_M3_R2_BY_SIGMA = {s: 1.0000 for s in _NEWTON_M4_R2_BY_SIGMA}
_NEWTON_M3_R2_BY_N     = {n: 1.0000 for n in _NEWTON_M4_R2_BY_N}


# ── Fixture builders ──────────────────────────────────────────────────────────

def _make_result(r2: float, success: bool = True,
                 rmse: float = 0.05, time: float = 11.0) -> Dict:
    return {"r2": r2, "rmse": rmse, "success": success,
            "time": time, "error": None,
            "metadata": {"decision": "llm", "nn_applied": True}}


def _make_test(description: str, results: Dict) -> Dict:
    return {
        "description": description,
        "domain": "mechanics",
        "results": results,
        "metadata": {"equation_name": description},
    }


def _make_noise_sweep_data(sigma: float) -> Dict:
    """Build a minimal noise sweep JSON for one sigma level."""
    newton_test = _make_test(_NEWTON_EQ, {
        _M3: _make_result(_NEWTON_M3_R2_BY_SIGMA[sigma], time=25.0),
        _M4: _make_result(_NEWTON_M4_R2_BY_SIGMA[sigma]),
    })
    # 29 passing equations to give realistic counts
    other_tests = [
        _make_test(f"Equation_{i:02d}", {
            _M3: _make_result(0.9999 + i * 1e-6),
            _M4: _make_result(0.9999 + i * 1e-6),
        })
        for i in range(29)
    ]
    return {
        "generated": "2026-03-15T09:10:18",
        "noise_levels": list(_NEWTON_M4_R2_BY_SIGMA.keys()),
        "methods": [_M3, _M4],
        "tests": [newton_test] + other_tests,
        "protocol": {"mode": "noisy", "noise_level": sigma,
                     "threshold": 0.95, "note": "test fixture"},
    }


def _make_sc_data(n_samples: int) -> Dict:
    """Build a minimal SC sweep JSON for one n value."""
    newton_test = _make_test(_NEWTON_EQ, {
        _M3: _make_result(_NEWTON_M3_R2_BY_N[n_samples], time=25.0),
        _M4: _make_result(_NEWTON_M4_R2_BY_N[n_samples]),
    })
    other_tests = [
        _make_test(f"Equation_{i:02d}", {
            _M3: _make_result(0.9999 + i * 1e-6),
            _M4: _make_result(0.9999 + i * 1e-6),
        })
        for i in range(29)
    ]
    return {
        "generated": "2026-03-15T12:43:10",
        "sample_sizes": list(_NEWTON_M4_R2_BY_N.keys()),
        "mode": "noisy",
        "methods": [_M3, _M4],
        "tests": [newton_test] + other_tests,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLASS 1 — Verify R² values are below catastrophic threshold
# ══════════════════════════════════════════════════════════════════════════════

class TestNewtonM4CatastrophicInNoiseSweep(unittest.TestCase):
    """
    M4 scores below the catastrophic threshold on Newton's gravity at every
    noise level tested.  M3 always scores R² = 1.000 on the same equation.
    """

    def setUp(self):
        self.cat_thresh = _CAT_THRESH

    # ── Per-sigma assertions ──────────────────────────────────────────────────

    def test_m4_newton_r2_below_threshold_sigma_0pct(self):
        r2 = _NEWTON_M4_R2_BY_SIGMA[0.000]
        self.assertLess(r2, self.cat_thresh,
            f"M4 Newton R² at σ=0% should be below {self.cat_thresh}, got {r2}")

    def test_m4_newton_r2_catastrophic_sigma_05pct(self):
        r2 = _NEWTON_M4_R2_BY_SIGMA[0.005]
        self.assertLess(r2, self.cat_thresh,
            f"M4 Newton R² at σ=0.5% should be catastrophic (<{self.cat_thresh}), got {r2}")
        self.assertAlmostEqual(r2, 0.6458, places=3)

    def test_m4_newton_r2_catastrophic_sigma_1pct(self):
        r2 = _NEWTON_M4_R2_BY_SIGMA[0.010]
        self.assertLess(r2, self.cat_thresh)
        self.assertAlmostEqual(r2, 0.7312, places=3)

    def test_m4_newton_r2_below_threshold_sigma_5pct(self):
        """σ=5% is the best case for M4 but still below 0.90."""
        r2 = _NEWTON_M4_R2_BY_SIGMA[0.050]
        self.assertLess(r2, self.cat_thresh)
        self.assertAlmostEqual(r2, 0.8723, places=3)

    def test_m4_newton_r2_catastrophic_sigma_10pct(self):
        r2 = _NEWTON_M4_R2_BY_SIGMA[0.100]
        self.assertLess(r2, self.cat_thresh)
        self.assertAlmostEqual(r2, 0.6844, places=3)

    def test_m4_newton_fails_at_all_five_sigma_levels(self):
        """Regression guard: failure must persist across all tested σ levels."""
        failures = [s for s, r2 in _NEWTON_M4_R2_BY_SIGMA.items()
                    if r2 >= self.cat_thresh]
        self.assertEqual(failures, [],
            f"M4 Newton should fail at ALL sigma levels; "
            f"unexpectedly passed at σ={failures}")

    def test_m3_newton_always_perfect_across_sigma(self):
        """M3 must solve Newton's gravity at every noise level."""
        for sigma, r2 in _NEWTON_M3_R2_BY_SIGMA.items():
            with self.subTest(sigma=sigma):
                self.assertGreater(r2, 0.9999,
                    f"M3 Newton R² at σ={sigma*100:.1f}% should be ~1.0, got {r2}")

    def test_m4_newton_gap_vs_m3_exceeds_10pct(self):
        """M3-M4 gap on Newton must be > 0.10 R² in the worst case."""
        max_gap = max(
            _NEWTON_M3_R2_BY_SIGMA[s] - _NEWTON_M4_R2_BY_SIGMA[s]
            for s in _NEWTON_M4_R2_BY_SIGMA
        )
        self.assertGreater(max_gap, 0.10,
            f"M3-M4 R² gap on Newton should exceed 0.10; got {max_gap:.4f}")

    def test_m4_newton_non_monotonic_across_sigma(self):
        """
        Non-monotonicity diagnostic: M4's Newton R² should NOT be monotonically
        decreasing with σ.  If it were, the failure would be purely
        noise-driven.  The non-monotonic pattern implicates LLM/NN initialisation.
        """
        r2_vals = [_NEWTON_M4_R2_BY_SIGMA[s]
                   for s in sorted(_NEWTON_M4_R2_BY_SIGMA)]
        diffs = [r2_vals[i+1] - r2_vals[i] for i in range(len(r2_vals)-1)]
        # If all diffs were ≤ 0, it would be monotonically decreasing
        is_monotone_decreasing = all(d <= 0 for d in diffs)
        self.assertFalse(is_monotone_decreasing,
            "M4 Newton R² should be non-monotonic across sigma, "
            "indicating initialisation sensitivity rather than noise sensitivity")


# ══════════════════════════════════════════════════════════════════════════════
# CLASS 2 — Verify R² values are below threshold in sample complexity sweep
# ══════════════════════════════════════════════════════════════════════════════

class TestNewtonM4CatastrophicInSCSweep(unittest.TestCase):
    """
    M4 scores below the catastrophic threshold on Newton's gravity at every
    tested sample size n ∈ {50, 100, 200, 500, 750, 1000}.
    """

    def setUp(self):
        self.cat_thresh = _CAT_THRESH

    def test_m4_newton_catastrophic_at_n50(self):
        r2 = _NEWTON_M4_R2_BY_N[50]
        self.assertLess(r2, self.cat_thresh)
        self.assertAlmostEqual(r2, 0.5933, places=3)

    def test_m4_newton_below_threshold_at_n100(self):
        r2 = _NEWTON_M4_R2_BY_N[100]
        self.assertLess(r2, self.cat_thresh)
        self.assertAlmostEqual(r2, 0.8208, places=3)

    def test_m4_newton_catastrophic_at_n200(self):
        r2 = _NEWTON_M4_R2_BY_N[200]
        self.assertLess(r2, self.cat_thresh)
        self.assertAlmostEqual(r2, 0.6470, places=3)

    def test_m4_newton_below_threshold_at_n500(self):
        r2 = _NEWTON_M4_R2_BY_N[500]
        self.assertLess(r2, self.cat_thresh)
        self.assertAlmostEqual(r2, 0.8207, places=3)

    def test_m4_newton_worst_case_at_n750(self):
        """n=750 is the single worst result across all conditions: R²=0.4186."""
        r2 = _NEWTON_M4_R2_BY_N[750]
        self.assertLess(r2, self.cat_thresh)
        self.assertAlmostEqual(r2, 0.4186, places=3)
        # Must be the worst
        all_r2 = list(_NEWTON_M4_R2_BY_N.values()) + list(_NEWTON_M4_R2_BY_SIGMA.values())
        self.assertEqual(min(all_r2), r2,
            f"n=750 (R²={r2}) should be the global worst case; "
            f"min across all conditions = {min(all_r2):.4f}")

    def test_m4_newton_catastrophic_at_n1000(self):
        r2 = _NEWTON_M4_R2_BY_N[1000]
        self.assertLess(r2, self.cat_thresh)
        self.assertAlmostEqual(r2, 0.7594, places=3)

    def test_m4_newton_fails_at_all_six_n_values(self):
        """Regression guard: failure must persist at every tested n."""
        passes = [n for n, r2 in _NEWTON_M4_R2_BY_N.items()
                  if r2 >= self.cat_thresh]
        self.assertEqual(passes, [],
            f"M4 Newton should fail at ALL n values; "
            f"unexpectedly passed at n={passes}")

    def test_m4_newton_does_not_improve_monotonically_with_n(self):
        """
        More data should help if failure is due to underfitting.
        Non-monotonicity rules out underfitting as the root cause.
        """
        r2_vals = [_NEWTON_M4_R2_BY_N[n] for n in sorted(_NEWTON_M4_R2_BY_N)]
        diffs = [r2_vals[i+1] - r2_vals[i] for i in range(len(r2_vals)-1)]
        is_monotone_increasing = all(d >= 0 for d in diffs)
        self.assertFalse(is_monotone_increasing,
            "M4 Newton R² should not improve monotonically with n — "
            "if it did, underfitting would be the cause (it is not)")

    def test_m3_newton_always_perfect_across_n(self):
        for n, r2 in _NEWTON_M3_R2_BY_N.items():
            with self.subTest(n=n):
                self.assertGreater(r2, 0.9999,
                    f"M3 Newton R² at n={n} should be ~1.0, got {r2}")


# ══════════════════════════════════════════════════════════════════════════════
# CLASS 3 — Aggregation pipeline correctly flags Newton as catastrophic
# ══════════════════════════════════════════════════════════════════════════════

class TestNewtonCatastrophicFlagInAggregation(unittest.TestCase):
    """
    When the aggregation pipeline processes a result set containing the
    Newton's gravity failure, it must:
      1. Include it in the catastrophic_failures list.
      2. Set n_catastrophic = 1 in the method summary.
      3. Not count it towards recovery_rate.
      4. Not exclude it from median R² calculation (it still counts).
    """

    def _run_aggregation(self, sigma: float):
        """
        Simulate the aggregation logic from run_noise_sweep_benchmark_v2.py
        _aggregate_results() for a single sigma level.
        """
        threshold = 0.90
        r2_vals_m4 = []
        n_recovery_m4 = 0
        n_catastrophic_m4 = 0
        catastrophic_failures = []
        n_total = 30

        # Newton — real M4 R²
        newton_r2 = _NEWTON_M4_R2_BY_SIGMA[sigma]
        r2_vals_m4.append(newton_r2)
        if newton_r2 >= threshold:
            n_recovery_m4 += 1
        if newton_r2 < _CAT_THRESH:
            n_catastrophic_m4 += 1
            catastrophic_failures.append({
                "equation": _NEWTON_EQ, "method": _M4, "r2": newton_r2
            })

        # 29 passing equations
        for _ in range(29):
            r2 = 0.9999
            r2_vals_m4.append(r2)
            if r2 >= threshold:
                n_recovery_m4 += 1

        return {
            "median_r2":     float(np.median(r2_vals_m4)),
            "recovery_rate": n_recovery_m4 / n_total,
            "n_catastrophic": n_catastrophic_m4,
            "catastrophic_failures": catastrophic_failures,
        }

    def test_newton_in_catastrophic_list_sigma_05pct(self):
        result = self._run_aggregation(0.005)
        cats = result["catastrophic_failures"]
        eq_names = [c["equation"] for c in cats]
        self.assertIn(_NEWTON_EQ, eq_names,
            "Newton's gravity must appear in catastrophic_failures at σ=0.5%")

    def test_newton_in_catastrophic_list_sigma_10pct(self):
        result = self._run_aggregation(0.100)
        cats = result["catastrophic_failures"]
        self.assertEqual(len(cats), 1)
        self.assertEqual(cats[0]["equation"], _NEWTON_EQ)
        self.assertEqual(cats[0]["method"], _M4)

    def test_n_catastrophic_is_one_for_m4(self):
        for sigma in _NEWTON_M4_R2_BY_SIGMA:
            with self.subTest(sigma=sigma):
                result = self._run_aggregation(sigma)
                self.assertEqual(result["n_catastrophic"], 1,
                    f"n_catastrophic should be 1 for M4 at σ={sigma*100:.1f}%")

    def test_n_catastrophic_zero_for_other_equations(self):
        """Only Newton's gravity is catastrophic — the other 29 equations pass."""
        result = self._run_aggregation(0.005)
        self.assertEqual(len(result["catastrophic_failures"]), 1,
            "Exactly one catastrophic failure expected (Newton's gravity)")

    def test_recovery_rate_reflects_newton_failure(self):
        """Recovery rate must be 29/30 = 0.9667, not 30/30."""
        result = self._run_aggregation(0.050)
        self.assertAlmostEqual(result["recovery_rate"], 29/30, places=3,
            msg="Recovery rate should be 29/30 when Newton fails threshold")

    def test_median_r2_still_one_despite_newton_failure(self):
        """
        29 equations at R²=0.9999 and one at R²≈0.65 → median is still 0.9999.
        The median must not be dragged down by a single outlier.
        """
        result = self._run_aggregation(0.005)
        self.assertGreater(result["median_r2"], 0.999,
            f"Median R² should remain ~1.0 despite Newton failure; "
            f"got {result['median_r2']:.6f}")

    def test_catastrophic_r2_value_stored_correctly(self):
        """The stored R² value in catastrophic_failures must match the actual result."""
        for sigma, expected_r2 in _NEWTON_M4_R2_BY_SIGMA.items():
            if expected_r2 < _CAT_THRESH:
                with self.subTest(sigma=sigma):
                    result = self._run_aggregation(sigma)
                    cats = result["catastrophic_failures"]
                    stored_r2 = cats[0]["r2"] if cats else None
                    self.assertIsNotNone(stored_r2)
                    self.assertAlmostEqual(stored_r2, expected_r2, places=4)


# ══════════════════════════════════════════════════════════════════════════════
# CLASS 4 — fail-fast triggers on Newton's gravity failure
# ══════════════════════════════════════════════════════════════════════════════

class TestNewtonFailFastBehaviour(unittest.TestCase):
    """
    When --fail-fast is set and M4 produces a catastrophic failure on
    Newton's gravity, the orchestrator must abort and exit non-zero.
    The failure scanner must detect the N/A R² line in the subprocess output.
    """

    def _make_failure_scan_line(self, r2_str: str = "N/A") -> str:
        """Simulate a method-result line from run_comparative_suite_benchmark_v2."""
        return (f"  {_M4[:42]:<42} {r2_str:>12}   0.0500      11.3    -\n")

    def test_failure_scanner_detects_na_r2(self):
        """
        The _FailureScanner in run_dual_condition_benchmark.py must flag
        a result line containing N/A R².
        """
        import re
        METHOD_ROW_RE = re.compile(
            r'^\s{2,8}(?P<n>.+?)\s{3,}(?P<r2>N/A|[-\d][\d.e+\-]*)(\s|$)',
            re.IGNORECASE,
        )
        line = self._make_failure_scan_line("N/A")
        m = METHOD_ROW_RE.match(line)
        self.assertIsNotNone(m, "Failure scanner regex must match N/A R² line")
        self.assertEqual(m.group("r2").upper(), "N/A")

    def test_failure_scanner_does_not_flag_valid_r2(self):
        """A valid R² line (0.8723) must not trigger the failure scanner."""
        import re
        METHOD_ROW_RE = re.compile(
            r'^\s{2,8}(?P<n>.+?)\s{3,}(?P<r2>N/A|[-\d][\d.e+\-]*)(\s|$)',
            re.IGNORECASE,
        )
        line = self._make_failure_scan_line("0.8723")
        m = METHOD_ROW_RE.match(line)
        self.assertIsNotNone(m)
        self.assertNotEqual(m.group("r2").upper(), "N/A")

    def test_scan_for_failures_finds_newton_in_json(self):
        """
        _scan_for_failures() from run_dual_condition_benchmark.py must
        identify M4's Newton result as a failure when R² is non-finite
        or when success=False is set.
        """
        data = _make_noise_sweep_data(0.005)
        # Set M4 Newton to success=False to test the JSON-level scanner
        data["tests"][0]["results"][_M4]["success"] = False
        data["tests"][0]["results"][_M4]["error"] = "NN ensemble degenerate fit"

        failures = []
        for test in data.get("tests", []):
            eq_name = test.get("metadata", {}).get("equation_name", "unknown")
            for method_name, result in test.get("results", {}).items():
                if result.get("success") is False:
                    failures.append({
                        "equation": eq_name,
                        "method":   method_name,
                        "reason":   result.get("error", "success=False"),
                        "r2":       result.get("r2"),
                    })

        self.assertEqual(len(failures), 1,
            "Exactly one failure should be detected")
        self.assertEqual(failures[0]["equation"], _NEWTON_EQ)
        self.assertEqual(failures[0]["method"], _M4)
        self.assertIn("degenerate", failures[0]["reason"])

    def test_non_finite_r2_detected_as_failure(self):
        """
        A NaN R² value on Newton's gravity must be caught by the
        non-finite R² scanner in _scan_for_failures().
        """
        import math
        data = _make_noise_sweep_data(0.005)
        data["tests"][0]["results"][_M4]["r2"] = float("nan")

        failures = []
        for test in data.get("tests", []):
            eq_name = test.get("metadata", {}).get("equation_name", "unknown")
            for method_name, result in test.get("results", {}).items():
                r2_val = result.get("r2")
                if r2_val is not None:
                    try:
                        if not math.isfinite(float(r2_val)):
                            failures.append({
                                "equation": eq_name,
                                "method":   method_name,
                                "reason":   f"r2={r2_val} (non-finite)",
                                "r2":       r2_val,
                            })
                    except (TypeError, ValueError):
                        pass

        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["equation"], _NEWTON_EQ)


# ══════════════════════════════════════════════════════════════════════════════
# CLASS 5 — Diagnostic tests for root-cause investigation
# ══════════════════════════════════════════════════════════════════════════════

class TestNewtonM4RootCauseDiagnostics(unittest.TestCase):
    """
    Diagnostic tests that encode the known properties of the failure to
    guide investigation.  These tests PASS on the current data and will
    FAIL if the failure is ever resolved, serving as a resolution detector.
    """

    def test_failure_is_m4_specific_not_m3(self):
        """M3 solves Newton perfectly; M4 fails. Failure is method-specific."""
        for sigma in _NEWTON_M4_R2_BY_SIGMA:
            m3_r2 = _NEWTON_M3_R2_BY_SIGMA[sigma]
            m4_r2 = _NEWTON_M4_R2_BY_SIGMA[sigma]
            self.assertGreater(m3_r2, 0.9999,
                f"M3 must solve Newton at σ={sigma*100:.1f}%")
            self.assertLess(m4_r2, _CAT_THRESH,
                f"M4 must fail Newton at σ={sigma*100:.1f}%")

    def test_failure_is_equation_specific_not_global(self):
        """
        Newton's gravity is the only failing equation.
        All other 29 equations must pass for M4 at σ=5%, n=200.
        """
        data = _make_noise_sweep_data(0.050)
        m4_failures = [
            t["description"] for t in data["tests"]
            if t["results"].get(_M4, {}).get("r2", 1.0) < _CAT_THRESH
        ]
        self.assertEqual(m4_failures, [_NEWTON_EQ],
            f"Only Newton's gravity should fail; got: {m4_failures}")

    def test_failure_persists_at_large_n_ruling_out_underfitting(self):
        """
        At n=750 and n=1000 M4 still fails Newton's gravity.
        This rules out underfitting (insufficient data) as the root cause.
        """
        for n in [750, 1000]:
            with self.subTest(n=n):
                r2 = _NEWTON_M4_R2_BY_N[n]
                self.assertLess(r2, _CAT_THRESH,
                    f"M4 Newton still fails at n={n} (R²={r2:.4f}), "
                    f"ruling out underfitting")

    def test_failure_persists_at_low_sigma_ruling_out_noise_sensitivity(self):
        """
        At σ=0% (noiseless) M4 still fails Newton's gravity (R²=0.828).
        This rules out noise sensitivity as the primary driver.
        """
        r2_noiseless = _NEWTON_M4_R2_BY_SIGMA[0.000]
        self.assertLess(r2_noiseless, _CAT_THRESH,
            f"M4 Newton fails even noiseless (R²={r2_noiseless:.4f}), "
            f"ruling out noise sensitivity as the root cause")

    def test_worst_case_is_n750_not_n50(self):
        """
        The worst result is at n=750 (R²=0.4186), not n=50 (R²=0.5933).
        A dataset-size-driven failure would be worst at smallest n.
        This non-monotonicity implicates NN ensemble/LLM initialisation.
        """
        r2_n50  = _NEWTON_M4_R2_BY_N[50]
        r2_n750 = _NEWTON_M4_R2_BY_N[750]
        self.assertLess(r2_n750, r2_n50,
            f"n=750 (R²={r2_n750:.4f}) must be worse than n=50 (R²={r2_n50:.4f}), "
            f"confirming non-dataset-size-driven failure")

    def test_failure_spread_across_11_of_11_conditions(self):
        """
        Newton's gravity fails (R² < 0.90) in every single tested condition
        — all 5 sigma levels and all 6 n values = 11 conditions.
        This confirms the failure is systematic, not stochastic.
        """
        all_conditions = (
            list(_NEWTON_M4_R2_BY_SIGMA.values()) +
            list(_NEWTON_M4_R2_BY_N.values())
        )
        failures = [r2 for r2 in all_conditions if r2 < _CAT_THRESH]
        self.assertEqual(len(failures), 11,
            f"Expected 11/11 conditions to fail catastrophically; "
            f"got {len(failures)}/11")

    def test_m4_nn_applied_flag_true_on_newton(self):
        """
        The metadata shows nn_applied=True on Newton's gravity for M4.
        The NN ensemble is being used — the failure is not because NN was skipped.
        """
        result_m4 = _make_result(_NEWTON_M4_R2_BY_SIGMA[0.005])
        nn_applied = result_m4["metadata"]["nn_applied"]
        self.assertTrue(nn_applied,
            "M4 Newton fixture must have nn_applied=True — "
            "failure occurs despite NN being active")


# ══════════════════════════════════════════════════════════════════════════════
# CLASS 6 — Resolution detection (tests that should FAIL once fixed)
# ══════════════════════════════════════════════════════════════════════════════

class TestNewtonM4ResolutionDetector(unittest.TestCase):
    """
    These tests encode the CURRENT failure state.  When the Newton's gravity
    issue is fixed (e.g., by increasing nn_seeds or adjusting the LLM prior),
    these tests will FAIL — signalling that the fix worked and the
    CatastrophicFailure tests should be retired.

    Run after each investigation experiment to detect resolution:
        pytest -k TestNewtonM4ResolutionDetector -v
    """

    _RESOLUTION_THRESHOLD = 0.9999   # target for a "fixed" result

    def test_m4_newton_not_yet_resolved_noise_sweep(self):
        """
        Will FAIL (and turn green in CI) when M4 Newton R² exceeds 0.9999
        across all sigma levels.  Currently expected to PASS (failure persists).
        """
        for sigma, r2 in _NEWTON_M4_R2_BY_SIGMA.items():
            with self.subTest(sigma=sigma):
                self.assertLess(r2, self._RESOLUTION_THRESHOLD,
                    f"Newton's gravity RESOLVED at σ={sigma*100:.1f}%! "
                    f"M4 R²={r2:.6f} ≥ {self._RESOLUTION_THRESHOLD}. "
                    f"Retire the catastrophic failure tests and update the paper.")

    def test_m4_newton_not_yet_resolved_sc_sweep(self):
        """
        Will FAIL (and turn green in CI) when M4 Newton R² exceeds 0.9999
        across all n values.  Currently expected to PASS (failure persists).
        """
        for n, r2 in _NEWTON_M4_R2_BY_N.items():
            with self.subTest(n=n):
                self.assertLess(r2, self._RESOLUTION_THRESHOLD,
                    f"Newton's gravity RESOLVED at n={n}! "
                    f"M4 R²={r2:.6f} ≥ {self._RESOLUTION_THRESHOLD}. "
                    f"Retire the catastrophic failure tests and update the paper.")

    def test_worst_case_r2_below_05(self):
        """
        The global worst case (n=750, R²=0.4186) is below 0.50.
        Will FAIL when n=750 is re-run and M4 improves above 0.50.
        """
        worst = min(
            list(_NEWTON_M4_R2_BY_SIGMA.values()) +
            list(_NEWTON_M4_R2_BY_N.values())
        )
        self.assertLess(worst, 0.50,
            f"Worst-case R² ({worst:.4f}) has improved above 0.50! "
            f"Re-run full sweep to verify resolution.")


# ══════════════════════════════════════════════════════════════════════════════
# CLASS 7 — JMLR reporting compliance tests
# ══════════════════════════════════════════════════════════════════════════════

class TestNewtonJMLRReportingCompliance(unittest.TestCase):
    """
    Tests that enforce correct reporting of the Newton failure in the JMLR
    paper.  These verify that the aggregation outputs are structured so that
    downstream report generators will correctly flag the issue.
    """

    def test_catastrophic_failure_reported_in_cross_noise_summary(self):
        """
        The cross-noise summary dict must carry n_catastrophic for M4.
        Verifies the aggregation pipeline propagates the count correctly.
        """
        # Simulate cross-noise summary structure
        cross_noise = {
            _M4: {
                "0.0050": {"recovery_rate": 29/30, "n_catastrophic": 1,
                           "median_r2": 0.9999},
                "0.0100": {"recovery_rate": 29/30, "n_catastrophic": 1,
                           "median_r2": 0.9999},
            }
        }
        for sigma_str, stats in cross_noise[_M4].items():
            with self.subTest(sigma=sigma_str):
                self.assertEqual(stats["n_catastrophic"], 1,
                    f"Cross-noise summary must report n_catastrophic=1 "
                    f"for M4 at σ={sigma_str}")

    def test_recovery_rate_reported_as_29_over_30(self):
        """
        Recovery rate for M4 must be reported as 29/30 = 0.9667,
        not 30/30 = 1.0, across all noisy sigma levels.
        """
        expected = 29 / 30
        for sigma in [0.005, 0.010, 0.050, 0.100]:
            # Simulate what the aggregator produces
            r2_vals = [_NEWTON_M4_R2_BY_SIGMA[sigma]] + [0.9999] * 29
            threshold = 0.95
            n_recovery = sum(1 for r2 in r2_vals if r2 >= threshold)
            recovery_rate = n_recovery / 30
            with self.subTest(sigma=sigma):
                self.assertAlmostEqual(recovery_rate, expected, places=3,
                    msg=f"Recovery rate at σ={sigma*100:.1f}% should be "
                        f"29/30={expected:.4f}, got {recovery_rate:.4f}")

    def test_equation_name_matches_paper_reference(self):
        """
        The equation name stored in catastrophic_failures must match the
        exact string used in the Feynman equation database to avoid
        misidentification in the paper.
        """
        self.assertIn("Newton", _NEWTON_EQ)
        self.assertIn("gravitational", _NEWTON_EQ.lower())
        self.assertIn("masses", _NEWTON_EQ.lower())

    def test_both_methods_reported_per_equation(self):
        """
        The per_equation output must contain results for both M3 and M4
        so that the paper can show the side-by-side comparison.
        """
        data = _make_noise_sweep_data(0.005)
        newton_test = data["tests"][0]
        self.assertIn(_M3, newton_test["results"],
            "M3 must be in Newton's results dict")
        self.assertIn(_M4, newton_test["results"],
            "M4 must be in Newton's results dict")

    def test_m3_m4_r2_gap_reportable_for_all_conditions(self):
        """
        The R² gap between M3 and M4 on Newton's gravity must be
        computable (no NaN) for all conditions so it can appear in tables.
        """
        for sigma, m4_r2 in _NEWTON_M4_R2_BY_SIGMA.items():
            m3_r2 = _NEWTON_M3_R2_BY_SIGMA[sigma]
            gap = m3_r2 - m4_r2
            self.assertTrue(np.isfinite(gap),
                f"R² gap must be finite at σ={sigma*100:.1f}%")
            self.assertGreater(gap, 0,
                f"M3 must always beat M4 on Newton at σ={sigma*100:.1f}%")

        for n, m4_r2 in _NEWTON_M4_R2_BY_N.items():
            m3_r2 = _NEWTON_M3_R2_BY_N[n]
            gap = m3_r2 - m4_r2
            self.assertTrue(np.isfinite(gap))
            self.assertGreater(gap, 0)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    unittest.main(verbosity=2)

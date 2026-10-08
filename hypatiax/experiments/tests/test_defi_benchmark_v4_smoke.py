"""
Smoke tests for hypatiax_defi_benchmark_v4.py / hypatiax_defi_benchmark_v4_pca.py.

Goal: catch "the script is broken" fast (import errors, CLI/API regressions,
crashed end-to-end run, malformed output JSON) — NOT re-validate benchmark
science. Runs in well under a minute, offline, no Anthropic spend.

What this deliberately does NOT mock:
- DeFiExperimentProtocol / build_extrap_split (real data generation — pure
  numpy/math, no network, cheap)
- The neural-network arm (real, tiny MLP on 200 samples — this is exactly
  the code path the ONE_EQUATION=1 built-in smoke mode exists to exercise)

What it DOES mock (both are network calls that cost money / require
credentials and would make this test flaky and non-hermetic):
- The Anthropic client used by the hybrid arm's inline `_generate_llm_formula`
- `PureLLMBaseline` (hypatiax.core.base_pure_llm.baseline_pure_llm_defi_discovery),
  used by the "pure_llm" arm

Run with:  pytest tests/test_defi_benchmark_v4_smoke.py -v

Assumed repo layout (matches _ROOT = Path(__file__).resolve().parent * 4
inside the benchmark scripts themselves):

    <repo_root>/
      hypatiax/
        experiments/benchmarks/hypatiax_defi_benchmark_v4.py
        experiments/benchmarks/hypatiax_defi_benchmark_v4_pca.py
        protocols/experiment_protocol_defi.py
        core/base_pure_llm/baseline_pure_llm_defi_discovery.py
      tests/
        test_defi_benchmark_v4_smoke.py   <- this file

If your checkout differs, set HYPATIAX_REPO_ROOT to the repo root before
running pytest and the module finder below will still work.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

# ─────────────────────────────────────────────────────────────────────────
# Module discovery / import
# ─────────────────────────────────────────────────────────────────────────

MODULE_FILENAMES = {
    "v4": "hypatiax_defi_benchmark_v4.py",
    "v4_pca": "hypatiax_defi_benchmark_v4_pca.py",
}

# A case that appears exactly once in the 74-case catalogue (so the
# `--cases` substring filter can't accidentally pick up a sibling case,
# the way "Constant product formula" would also match
# "Constant product formula (multivariate)"), is "easy"/"linear", and has
# no LLM-formula edge cases (norm_cdf, sqrt, etc.) — keeps the NN + hybrid
# arms fast and well-behaved.
SMOKE_CASE_NAME = "Value at Risk at 95%"

# A minimal, well-behaved formula the fake LLM returns for every prompt.
# Matches SMOKE_CASE_NAME's declared formula_type ("linear") closely enough
# to fit cleanly, and deliberately uses only the exact 2 variable names the
# risk_var domain's VaR case is generated with (see below — if this ever
# mismatches, _execute_formula just returns None and the LLM arm reports a
# clean failure rather than crashing, so the test stays safe either way).
FAKE_LLM_FORMULA_SRC = (
    "def formula(portfolio_value, z_score):\n"
    "    return portfolio_value * z_score\n"
)


def _find_repo_root() -> Path:
    env = os.environ.get("HYPATIAX_REPO_ROOT")
    if env:
        return Path(env).resolve()
    # tests/<this file> -> repo root is the parent of tests/
    return Path(__file__).resolve().parents[1]


def _find_module_path(key: str) -> Path:
    filename = MODULE_FILENAMES[key]
    repo_root = _find_repo_root()
    candidate = repo_root / "hypatiax" / "experiments" / "benchmarks" / filename
    if candidate.exists():
        return candidate
    # Fall back to a search in case the layout has drifted.
    hits = list(repo_root.rglob(filename))
    if hits:
        return hits[0]
    pytest.skip(
        f"Could not locate {filename} under {repo_root} "
        f"(set HYPATIAX_REPO_ROOT if your checkout layout differs)."
    )


def _load_module(key: str):
    path = _find_module_path(key)
    mod_name = f"_smoke_{key}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)  # raises on any import-time error
    return module


@pytest.fixture(scope="module", params=["v4", "v4_pca"])
def defi_module(request):
    """Import hypatiax_defi_benchmark_v4[.py|_pca.py] fresh, per variant."""
    return _load_module(request.param)


# ─────────────────────────────────────────────────────────────────────────
# Network / credential stubs
# ─────────────────────────────────────────────────────────────────────────

class _FakeAnthropicMessages:
    def create(self, model, max_tokens, messages):
        text = FAKE_LLM_FORMULA_SRC
        content_block = types.SimpleNamespace(text=text)
        return types.SimpleNamespace(content=[content_block])


class _FakeAnthropicClient:
    def __init__(self, api_key=None):
        self.messages = _FakeAnthropicMessages()


@pytest.fixture
def stub_anthropic(monkeypatch):
    """
    _generate_llm_formula() does `from anthropic import Anthropic` *inside*
    the function body, so patching sys.modules["anthropic"] before it's
    called is what takes effect — regardless of whether the real `anthropic`
    package is installed.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-smoke-test-not-real")
    fake_anthropic_module = types.ModuleType("anthropic")
    fake_anthropic_module.Anthropic = _FakeAnthropicClient
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic_module)
    yield


class _FakePureLLMBaseline:
    """
    Stub for hypatiax.core.base_pure_llm.baseline_pure_llm_defi_discovery.PureLLMBaseline,
    matching the interface hypatiax_defi_benchmark_v4[.py] actually calls:
      - generate_formula(description, domain, var_names, metadata) -> dict
        with at least "python_code" (or "formula_code") and "error"
      - test_formula_accuracy(llm_res, X, y, var_names, verbose=False) -> dict
        with at least "r2" and "success"
    """
    def __init__(self, model=None):
        self.model = model
        self._cache = {}

    def generate_formula(self, description, domain, var_names, metadata):
        return {
            "python_code": FAKE_LLM_FORMULA_SRC,
            "formula_code": FAKE_LLM_FORMULA_SRC,
            "raw_response": FAKE_LLM_FORMULA_SRC,
            "model": self.model,
            "error": None,
        }

    def test_formula_accuracy(self, llm_res, X, y, var_names, verbose=False):
        import numpy as np
        try:
            local: dict = {}
            exec(llm_res["python_code"], {"np": np}, local)
            fn = next(v for v in local.values() if callable(v))
            args = [X[:, i] for i in range(X.shape[1])]
            pred = np.asarray(fn(*args), dtype=float).flatten()
            err = y - pred
            ss_r = float(np.sum(err ** 2))
            ss_t = float(np.sum((y - y.mean()) ** 2))
            r2 = 1 - ss_r / ss_t if ss_t > 1e-10 else 0.0
            return {"r2": r2, "success": True}
        except Exception as e:
            return {"r2": float("nan"), "success": False, "error": str(e)}


@pytest.fixture
def stub_pure_llm_baseline(monkeypatch):
    """
    PureLLMBaseline is imported lazily, inline, from
    hypatiax.core.base_pure_llm.baseline_pure_llm_defi_discovery — patch the
    class on that real module so the lazy `from ... import PureLLMBaseline`
    picks up the stub. Requires the real module to exist (it's part of the
    repo, not part of these two files) but never touches the network.
    """
    mod_path = "hypatiax.core.base_pure_llm.baseline_pure_llm_defi_discovery"
    try:
        real_module = importlib.import_module(mod_path)
    except ImportError:
        pytest.skip(
            f"{mod_path} not importable — full hypatiax package checkout "
            f"required for the pure_llm arm's end-to-end smoke test."
        )
    monkeypatch.setattr(real_module, "PureLLMBaseline", _FakePureLLMBaseline)
    yield


@pytest.fixture
def fast_nn(defi_module, monkeypatch):
    """Defense in depth: cap NN wall-clock time low so a smoke test never
    hangs even if a future regression reintroduces a runaway-convergence
    case (this is exactly what _NN_MAX_TIME_S was added to prevent)."""
    monkeypatch.setattr(defi_module, "_NN_MAX_TIME_S", 20)


# ─────────────────────────────────────────────────────────────────────────
# Import-level smoke tests
# ─────────────────────────────────────────────────────────────────────────

def test_module_imports_cleanly(defi_module):
    assert hasattr(defi_module, "run_benchmark")
    assert hasattr(defi_module, "report_only")


def test_case_catalogue_has_74_tractable_cases(defi_module):
    cases = defi_module._get_test_cases()
    assert len(cases) == 74
    names = [c["name"] for c in cases]
    assert len(names) == len(set(names)), "duplicate case name in catalogue"
    for c in cases:
        assert {"name", "domain", "difficulty", "formula_type", "num_samples", "config"} <= c.keys()


def test_smoke_case_name_is_unique_substring(defi_module):
    """Guards the test fixture itself: SMOKE_CASE_NAME must not be a
    substring of any other case name, or --cases-style filtering (used in
    the end-to-end test below) would silently run more than one case."""
    names = [c["name"].lower() for c in defi_module._get_test_cases()]
    matches = [n for n in names if SMOKE_CASE_NAME.lower() in n]
    assert matches == [SMOKE_CASE_NAME.lower()]


# ─────────────────────────────────────────────────────────────────────────
# Pure-function smoke tests (fast, no I/O)
# ─────────────────────────────────────────────────────────────────────────

def test_compute_moneyness_log_and_ratio(defi_module):
    import numpy as np
    S, K = np.array([110.0]), np.array([100.0])
    assert defi_module.compute_moneyness(S, K, mode="log")[0] == pytest.approx(np.log(1.1))
    assert defi_module.compute_moneyness(S, K, mode="ratio")[0] == pytest.approx(1.1)
    with pytest.raises(ValueError):
        defi_module.compute_moneyness(S, K, mode="bogus")


def test_compute_metrics_perfect_fit_is_r2_one(defi_module):
    import numpy as np
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = defi_module._compute_metrics(y, y.copy())
    assert m["r2"] == pytest.approx(1.0)
    assert m["mae"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(0.0)


def test_execute_formula_happy_path(defi_module):
    import numpy as np
    X = np.column_stack([np.linspace(1, 10, 20), np.linspace(2, 5, 20)])
    preds = defi_module._execute_formula("def formula(a, b):\n    return a + b\n", X)
    assert preds is not None
    assert np.allclose(preds, X[:, 0] + X[:, 1])


def test_execute_formula_rejects_constant_output(defi_module):
    import numpy as np
    X = np.column_stack([np.linspace(1, 10, 20), np.linspace(2, 5, 20)])
    preds = defi_module._execute_formula("def formula(a, b):\n    return 0*a + 1.0\n", X)
    assert preds is None  # degenerate constant formula must be rejected


def test_execute_formula_rejects_syntax_error(defi_module):
    import numpy as np
    X = np.column_stack([np.linspace(1, 10, 20), np.linspace(2, 5, 20)])
    assert defi_module._execute_formula("def formula(a, b:\n    return a\n", X) is None


@pytest.mark.parametrize("bad_code", [
    "def formula(x):\n    return 1/0\n",
    "def formula(x):\n    return np.inf * x\n",
    "def formula(x):\n    return x**1000\n",
    "def formula(x):\n    return nan + x\n",
])
def test_pathological_formula_detection(defi_module, bad_code):
    assert defi_module._formula_has_pathological_behavior(bad_code) is True


def test_pathological_detection_no_false_positive_on_nan_substring_identifier(defi_module):
    # regression guard for the whole-word-match bugfix documented in the
    # script itself ("nominal", "channel", etc. must NOT match bare "nan")
    code = "def formula(nominal, channel):\n    return nominal + channel\n"
    assert defi_module._formula_has_pathological_behavior(code) is False


# ─────────────────────────────────────────────────────────────────────────
# Checkpoint dedup smoke test (regression guard for the (equation_id, seed)
# dedup fix noted in the project's own audit history)
# ─────────────────────────────────────────────────────────────────────────

def test_checkpoint_dedup_by_equation_id_and_seed(defi_module, tmp_path, monkeypatch):
    defi_module._configure_output_dir(str(tmp_path))
    duplicated = [
        {"equation_id": "Value at Risk at 95%", "seed": 42, "results": {"pass": 1}},
        {"equation_id": "Value at Risk at 95%", "seed": 42, "results": {"pass": 2}},  # newer dup, same key
        {"equation_id": "Value at Risk at 95%", "seed": 99, "results": {"pass": 1}},  # distinct seed
        {"equation_id": "Value at Risk at 99%", "seed": 42, "results": {"pass": 1}},  # distinct case
    ]
    defi_module.CHECKPOINT_FILE.write_text(json.dumps(duplicated))
    data, n = defi_module._load_checkpoint()
    assert n == 3, "same (equation_id, seed) pair must collapse to one record"
    vAr95_seed42 = [d for d in data if d["equation_id"] == "Value at Risk at 95%" and d["seed"] == 42]
    assert len(vAr95_seed42) == 1
    assert vAr95_seed42[0]["results"]["pass"] == 2, "must keep the LAST occurrence, not the first"


# ─────────────────────────────────────────────────────────────────────────
# GT-leak regression guard
# ─────────────────────────────────────────────────────────────────────────

def test_llm_prompt_never_leaks_ground_truth(defi_module, stub_anthropic):
    """Regression guard for the ground-truth-leak bug: the hybrid arm's LLM
    prompt must never contain the answer it's supposed to be deriving."""
    metadata = {"ground_truth": "SECRET_ANSWER_MUST_NOT_LEAK", "constants": {}}
    captured = {}

    class _CapturingMessages:
        def create(self, model, max_tokens, messages):
            captured["prompt"] = messages[0]["content"]
            return types.SimpleNamespace(
                content=[types.SimpleNamespace(text=FAKE_LLM_FORMULA_SRC)]
            )

    class _CapturingClient:
        def __init__(self, api_key=None):
            self.messages = _CapturingMessages()

    fake_mod = types.ModuleType("anthropic")
    fake_mod.Anthropic = _CapturingClient
    sys.modules["anthropic"] = fake_mod

    defi_module._generate_llm_formula(
        "Value at risk at the 95% confidence level",
        "risk_var",
        ["portfolio_value", "z_score"],
        metadata,
    )
    assert "prompt" in captured, "LLM was never called — test fixture broken"
    assert "SECRET_ANSWER_MUST_NOT_LEAK" not in captured["prompt"]
    assert "ground_truth" not in captured["prompt"].lower()


# ─────────────────────────────────────────────────────────────────────────
# End-to-end smoke test: one case, both arms, real NN, real data generation,
# stubbed LLMs — this is the main event.
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.timeout(120) if False else (lambda f: f)  # no-op if pytest-timeout absent
def test_run_benchmark_end_to_end_single_case(
    defi_module, tmp_path, stub_anthropic, stub_pure_llm_baseline, fast_nn
):
    defi_module._configure_output_dir(str(tmp_path))

    results = defi_module.run_benchmark(
        resume=False,
        verify_fix5=False,
        verbose=False,
        cases=[SMOKE_CASE_NAME],
    )

    assert len(results) == 1, f"expected exactly 1 case run, got {len(results)}"
    record = results[0]

    # ── record-level schema ──────────────────────────────────────────
    assert record["equation_id"] == SMOKE_CASE_NAME
    for key in ("seed", "difficulty", "formula_type", "extrapolation_intractable", "results"):
        assert key in record

    arms = record["results"]
    for arm in ("pure_llm", "neural_network", "hybrid"):
        assert arm in arms, f"missing {arm} arm in results"
        assert "test_r2" in arms[arm]
        assert "train_r2" in arms[arm]

    # ── parity / regression guard: extrapolation_unmitigated must be
    #    present on the hybrid arm on both the success path AND the
    #    exception-fallback path (this is the exact parity gap the audit
    #    found and fixed between v4.py and v4_pca.py) ──────────────────
    assert "extrapolation_unmitigated" in arms["hybrid"]
    # llm_train_r2 must be emitted on the success path too, otherwise the
    # exception-path parity test below compares against an incomplete baseline.
    assert "llm_train_r2" in arms["hybrid"]
    # llm_model / model_used are the model-identity audit fields; both benchmark
    # variants (v4 and v4_pca) must emit them so mixed-model runs stay auditable.
    assert "llm_model" in arms["hybrid"]
    assert "model_used" in arms["hybrid"]

    # ── files actually got written ───────────────────────────────────
    assert defi_module.FINAL_OUTPUT.exists()
    on_disk = json.loads(defi_module.FINAL_OUTPUT.read_text())
    assert len(on_disk) == 1
    assert on_disk[0]["equation_id"] == SMOKE_CASE_NAME

    # ── checkpoint is cleaned up after a clean, non-resumed run ───────
    assert not defi_module.CHECKPOINT_FILE.exists()

    # ── report_only() must run against what was just written ─────────
    defi_module.report_only()  # must not raise


def test_run_benchmark_hybrid_exception_fallback_has_same_keys_as_success(
    defi_module, tmp_path, stub_anthropic, stub_pure_llm_baseline, fast_nn, monkeypatch
):
    """Force the hybrid arm's except-branch and assert its dict has the same
    key set as the success-path dict — the exact parity check the audit
    history says was fixed (pca had extrapolation_unmitigated + 5 other
    keys the v4.py except-branch was missing; now both should agree).
    v4.py and v4_pca.py are additionally kept identical on llm_train_r2,
    llm_model and model_used."""
    defi_module._configure_output_dir(str(tmp_path))

    def _boom(*args, **kwargs):
        raise RuntimeError("forced failure for exception-path parity test")

    monkeypatch.setattr(defi_module, "_v4_hybrid_predict_and_eval", _boom)

    results = defi_module.run_benchmark(cases=[SMOKE_CASE_NAME])
    assert len(results) == 1
    hybrid = results[0]["results"]["hybrid"]

    expected_keys = {
        "train_r2", "test_r2", "success", "error",
        "llm_trustworthy", "selected_candidate", "llm_train_r2",
        "validation_r2", "validation_n", "timed_out",
        "llm_model", "model_used",
        "extrapolation_unmitigated",
    }
    assert expected_keys <= hybrid.keys()
    assert hybrid["extrapolation_unmitigated"] is None
    assert hybrid["success"] is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

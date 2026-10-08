"""
Selection-logic tests for hypatiax_defi_benchmark_v4.py's hybrid arm.

Unlike test_defi_benchmark_v4_smoke.py (which only checks that the script runs
end-to-end), these tests assert that _select_v4_candidate and
_v4_hybrid_predict_and_eval actually ROUTE to the correct candidate given known
inputs. No LLM calls, no network, and (except for the opt-in aggregate test)
no dependency on a real protocol checkout.

Layer 1 -- deterministic logic tests (default, CI-safe, ~10 s):

  Trustworthy gate  (train_r2 > 0.5 AND no pathological code pattern)
    * test_gate_rejects_low_train_r2_formula
    * test_gate_rejects_pathological_pattern_even_when_fit_is_perfect
    * test_untrustworthy_formula_loses_to_fallback
    * test_untrustworthy_formula_on_extrapolative_split_is_flagged

  np./numpy. bypass of the NaN-safe math wrappers  (Fix 19)
    * test_np_prefixed_math_matches_bare_wrapper (parametrized: log/-x, log/0, sqrt/-x)
    * test_np_log_on_black_scholes_style_ratio_is_guarded
    * test_np_namespace_still_exposes_ordinary_numpy_functions

  Selector (_select_v4_candidate)
    * test_selector_tie_break_prefers_llm_when_formula_is_exact
    * test_selector_extrapolative_without_llm_picks_simplest_nn_and_flags
    * test_selector_in_domain_without_llm_is_not_flagged

  Extrapolation guard vs. blend  (Fix 18)
    * test_extrapolative_blend_cannot_collapse_to_bare_nn
    * test_in_domain_blend_grid_is_unrestricted

  Ranked fallback walk after full-fit failure  (Fix 15 / Fix 16)
    * test_ranked_walk_lands_on_llm_when_it_is_next_best
    * test_ranked_walk_is_rank_based_not_hardwired_to_llm
    * test_ranked_walk_skips_llm_when_untrustworthy
    * test_ranked_walk_last_resort_is_nn_fallback
    * test_ranked_fallback_end_to_end_with_real_selector   (integration)
    * test_no_llm_formula_still_yields_usable_nn

  The walk tests stub the selector's *ranking* (validation_candidates) so the
  order is fixed by construction rather than by NN training noise; only the
  final fits are real.

Layer 2 -- statistical check on real runs (opt-in):

  * test_aggregate_hybrid_does_not_regress_on_catalogue
    Answers a different question: does validation-R2 still predict test-R2
    well enough, on real LLM formulas across the catalogue, for the selector to
    be worth having? Needs a real hypatiax checkout, an API key and a
    checked-in baseline, so it only runs when HYPATIAX_RUN_SLOW=1.

Run:
    pytest test_defi_benchmark_v4_selection.py -v                 # layer 1
    HYPATIAX_RUN_SLOW=1 pytest test_defi_benchmark_v4_selection.py -v   # + layer 2
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────────────────
# Module discovery / import (mirrors test_defi_benchmark_v4_smoke.py)
# ─────────────────────────────────────────────────────────────────────────

MODULE_FILENAME = "hypatiax_defi_benchmark_v4.py"


def _find_repo_root() -> Path:
    env = os.environ.get("HYPATIAX_REPO_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[1]


def _find_module_path() -> Path:
    repo_root = _find_repo_root()
    candidate = repo_root / "hypatiax" / "experiments" / "benchmarks" / MODULE_FILENAME
    if candidate.exists():
        return candidate
    # This file may be sitting next to the module itself.
    here = Path(__file__).resolve().parent / MODULE_FILENAME
    if here.exists():
        return here
    hits = list(repo_root.rglob(MODULE_FILENAME))
    if hits:
        return hits[0]
    pytest.skip(
        f"Could not locate {MODULE_FILENAME} under {repo_root} "
        f"(set HYPATIAX_REPO_ROOT if your checkout layout differs)."
    )


def _load_module():
    path = _find_module_path()
    mod_name = "_selection_test_v4"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)  # raises on any import-time error
    return module


@pytest.fixture(scope="module")
def defi_module():
    """Import hypatiax_defi_benchmark_v4.py once per test module."""
    return _load_module()


@pytest.fixture
def fast_nn(defi_module, monkeypatch):
    """Keep NN fits fast: these tests need the ranking/routing, not a
    well-converged network."""
    monkeypatch.setattr(defi_module, "_NN_MAX_TIME_S", 20)
    monkeypatch.setattr(defi_module, "_V4_MAX_TIME_S", 5)


# ─────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────

VAR_NAMES = ["portfolio_value", "z_score"]
CONFIG = {"split_var_idx": 0}

# True relationship is y = 3*x1 + 2*x2 in every fixture below.
EXACT_FORMULA = (
    "def formula(portfolio_value, z_score):\n"
    "    return 3.0*portfolio_value + 2.0*z_score\n"
)
# Near-zero slope: executes fine (not constant, so _execute_formula accepts it)
# but is essentially uncorrelated with y -> deeply negative train R2.
BAD_FORMULA = (
    "def formula(portfolio_value, z_score):\n"
    "    return 0.0001*portfolio_value - 0.0001*z_score\n"
)
# Decent-but-imperfect (right linear structure, wrong coefficients -- weights
# swapped relative to the truth): train R2 ~0.82, comfortably clears the 0.5
# gate with margin (checked across 20 seeds: min 0.78 / max 0.86) while
# remaining clearly beatable by a correction model.
DECENT_FORMULA = (
    "def formula(portfolio_value, z_score):\n"
    "    return 2.0*portfolio_value + 3.0*z_score\n"
)
# Fits the data perfectly, yet contains a "1/0" token in dead code. The gate's
# pathological-pattern check is a substring match, so this must be rejected
# on the pattern alone.
PATHOLOGICAL_BUT_PERFECT_FORMULA = (
    "def formula(portfolio_value, z_score):\n"
    "    _unused = 0 if True else 1/0\n"
    "    return 3.0*portfolio_value + 2.0*z_score\n"
)


def _in_domain_case(seed: int = 0):
    """Random 160/40 split of y = 3*x1 + 2*x2.

    Train and test are drawn from the SAME domain, so `_v4_extrapolates` is
    False and the extrapolation guard in `_select_v4_candidate` stays out of
    the way. A NN fit on this data generalises well (test R2 ~ 0.999), which
    is what lets the tests assert an absolute quality floor instead of only
    "less terrible than the alternative".
    """
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(1, 10, 200)
    x2 = rng.uniform(2, 5, 200)
    y = 3.0 * x1 + 2.0 * x2
    X = np.column_stack([x1, x2])
    idx = rng.permutation(200)
    tr, te = idx[:160], idx[160:]
    return X[tr], y[tr], X[te], y[te]


def _extrapolative_case():
    """Sorted split: the last 40 points lie ABOVE the training range of x1,
    so `_v4_extrapolates` is True."""
    x1 = np.linspace(1, 10, 200)
    x2 = np.linspace(2, 5, 200)
    y = 3.0 * x1 + 2.0 * x2
    X = np.column_stack([x1, x2])
    return X[:160], y[:160], X[160:], y[160:]


def _run_hybrid(mod, case, code, seed=42):
    X_tr, y_tr, X_te, y_te = case
    return mod._v4_hybrid_predict_and_eval(
        "desc", "risk_var", X_tr, y_tr, X_te, y_te,
        VAR_NAMES, {"constants": {}}, CONFIG,
        seed=seed, pure_llm_code=code, pure_llm_model="test-model",
    )


def _blind_r2(mod, code, X_te, y_te):
    """R2 you'd get by trusting `code` blindly -- computed independently of
    the hybrid path, as the ground-truth comparator."""
    pred = mod._execute_formula(code, X_te, constants={})
    return mod._compute_metrics(y_te, pred)["r2"]


def _fake_selection(selected, ranked):
    """Build a `_select_v4_candidate` return value with a FIXED ranking.

    `ranked` is a list of (candidate_name, hidden, val_r2); the dict key is
    "<prefix>:<hidden>" exactly as the real selector builds it. Because the
    ranking is set by hand, the fallback-walk tests don't depend on which
    candidate NN training noise would have favoured.
    """
    vc, vr2 = {}, {}
    for cand, hidden, r2 in ranked:
        prefix = "residual" if cand == "residual_nn" else cand
        key = f"{prefix}:{hidden}" if hidden else prefix
        vc[key] = {"candidate": cand, "hidden": hidden,
                   "alpha": 0.0 if cand == "nn" else 1.0, "r2": r2}
        vr2[key] = r2
    return {
        "selected": selected, "hidden": [64, 32], "blend_alpha": 1.0,
        "validation_r2": vr2, "validation_candidates": vc,
        "validation_n": 40, "extrapolation_unmitigated": False,
    }


def _record_fit_calls(mod, monkeypatch, fail=()):
    """Wrap _fit_candidate_full: record every candidate name it is asked to
    fit, and return None (simulated full-fit failure) for names in `fail`."""
    real = mod._fit_candidate_full
    calls = []

    def wrapper(candidate, *a, **kw):
        calls.append(candidate)
        if candidate in fail:
            return None
        return real(candidate, *a, **kw)

    monkeypatch.setattr(mod, "_fit_candidate_full", wrapper)
    return calls


# ─────────────────────────────────────────────────────────────────────────
# Trustworthy gate
# ─────────────────────────────────────────────────────────────────────────

def test_gate_rejects_low_train_r2_formula(defi_module, fast_nn):
    """Half 1 of the gate: train_r2 must exceed 0.5."""
    case = _in_domain_case()
    pred = defi_module._execute_formula(BAD_FORMULA, case[0], constants={})
    assert defi_module._compute_metrics(case[1], pred)["r2"] < 0.5
    assert defi_module._formula_has_pathological_behavior(BAD_FORMULA) is False

    result = _run_hybrid(defi_module, case, BAD_FORMULA)
    assert result["llm_trustworthy"] is False
    assert result["llm_train_r2"] < 0.5


def test_gate_rejects_pathological_pattern_even_when_fit_is_perfect(defi_module, fast_nn):
    """Half 2 of the gate: a formula with train_r2 == 1.0 must still be
    rejected if it contains a pathological pattern (here a "1/0" token)."""
    case = _in_domain_case()
    assert defi_module._formula_has_pathological_behavior(PATHOLOGICAL_BUT_PERFECT_FORMULA)

    result = _run_hybrid(defi_module, case, PATHOLOGICAL_BUT_PERFECT_FORMULA)

    # It really does fit perfectly -- so only the pattern check can be what
    # rejected it.
    assert result["llm_train_r2"] > 0.999
    assert result["llm_trustworthy"] is False
    assert result["selected_candidate"] != "llm"


def test_untrustworthy_formula_loses_to_fallback(defi_module, fast_nn):
    """A gated-out formula must not be used, and the fallback must beat blind
    trust by a wide margin AND be genuinely good in absolute terms."""
    case = _in_domain_case()
    result = _run_hybrid(defi_module, case, BAD_FORMULA)

    assert result["llm_trustworthy"] is False
    assert result["selected_candidate"] == "nn"

    blind = _blind_r2(defi_module, BAD_FORMULA, case[2], case[3])
    assert result["test_r2"] > 0.9, (
        f"fallback NN should generalise on in-domain data, got {result['test_r2']}"
    )
    assert result["test_r2"] > blind + 0.5, (
        f"fallback (test_r2={result['test_r2']}) did not clearly beat "
        f"blind trust in the bad formula (blind_r2={blind})"
    )


def test_untrustworthy_formula_on_extrapolative_split_is_flagged(defi_module, fast_nn):
    """Bad formula + test points outside the training range: there is no
    LLM-anchored candidate for the extrapolation guard to prefer, so the run
    must be flagged `extrapolation_unmitigated` rather than passing off an
    in-domain-validated NN as a safe choice."""
    case = _extrapolative_case()
    assert defi_module._v4_extrapolates(case[0], case[2]) is True

    result = _run_hybrid(defi_module, case, BAD_FORMULA)

    assert result["llm_trustworthy"] is False
    assert result["selected_candidate"] in ("nn", "nn_fallback")
    assert result["extrapolation_unmitigated"] is True


# ─────────────────────────────────────────────────────────────────────────
# np./numpy. bypass of the NaN-safe math wrappers (Fix 19)
#
# The formula-generation prompt tells the model to "Use numpy (imported as
# np)", so generated code overwhelmingly calls np.log(...)/np.sqrt(...)
# rather than the bare-name wrappers ("log", "sqrt", ...) _EXEC_GLOBALS also
# provides. Before Fix 19, "np" was bound to the real numpy module, so
# np.log(x) resolved via attribute lookup straight past the "log" wrapper of
# the same name sitting next to it in the same globals dict -- at domain
# boundaries (log/sqrt of <= 0: Black-Scholes d1/d2, moneyness, leverage
# ratios, ...) that raised "RuntimeWarning: invalid value encountered in
# log/sqrt" and returned -inf/nan uncontained, corrupting llm_train_r2 (or
# tipping it to nan outright) and forcing the untrustworthy formula off the
# LLM-anchored path entirely.
# ─────────────────────────────────────────────────────────────────────────

def _assert_no_runtime_warning(fn, *args, **kwargs):
    """Run fn(*args, **kwargs) with RuntimeWarning promoted to an error, so
    a stray 'invalid value encountered in log/sqrt/exp' fails the test
    instead of silently printing to stderr."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        return fn(*args, **kwargs)


@pytest.mark.parametrize(
    "fn_name, bad",
    [
        ("log", -2.0),    # log of a negative value
        ("log", 0.0),     # log of zero
        ("sqrt", -9.0),   # sqrt of a negative value
    ],
)
def test_np_prefixed_math_matches_bare_wrapper(defi_module, fn_name, bad):
    """np.<fn>(x) must be exactly as NaN-safe as the bare <fn>(x) wrapper --
    same output, and no RuntimeWarning escaping the guard -- regardless of
    which spelling the LLM-generated formula happens to use.

    The 19 in-domain points vary (1..19) rather than repeating a single
    value: a constant array would trip _execute_formula's *separate*
    degenerate-output guard (std < 1e-12 -> None) before this test ever
    reaches the log/sqrt guard it's meant to check. Only one point in
    twenty is out-of-domain (5%), staying under _execute_formula's 10%
    NaN/inf rejection threshold, so this exercises the median-fill path
    rather than the "too many bad points -> None" path.
    """
    X = np.array([[float(v)] for v in range(1, 20)] + [[bad]])
    code_bare = f"def formula(x):\n    return {fn_name}(x)\n"
    code_np = f"def formula(x):\n    return np.{fn_name}(x)\n"

    out_bare = _assert_no_runtime_warning(defi_module._execute_formula, code_bare, X)
    out_np = _assert_no_runtime_warning(defi_module._execute_formula, code_np, X)

    assert out_bare is not None, "bare wrapper should survive a 5% NaN rate"
    assert out_np is not None, (
        f"np.{fn_name}(x) was rejected outright -- if this is None while "
        f"the bare wrapper isn't, np. is still bypassing the guard"
    )
    assert np.all(np.isfinite(out_bare))
    assert np.all(np.isfinite(out_np))
    assert np.allclose(out_bare, out_np), (
        f"np.{fn_name}(x) must behave identically to the guarded {fn_name}(x) "
        f"wrapper -- a mismatch means np.{fn_name} is resolving to the real, "
        f"unguarded numpy function again"
    )


def test_np_log_on_black_scholes_style_ratio_is_guarded(defi_module):
    """A realistic case: a Black-Scholes-style formula that does
    `np.log(S / K)` directly, the way the LLM prompt's "use np" instruction
    actually produces, on a batch where S dips negative for one sample
    (e.g. a synthetic/adversarial point, or float noise at a domain edge).
    Must not warn and must not propagate -inf/nan into the output."""
    rng = np.random.default_rng(0)
    S = np.concatenate([rng.uniform(50, 150, 19), [-5.0]])
    K = np.full(20, 100.0)
    code = "def formula(S, K):\n    return np.log(S / K)\n"

    out = _assert_no_runtime_warning(
        defi_module._execute_formula, code, np.column_stack([S, K])
    )

    assert out is not None
    assert np.all(np.isfinite(out))


def test_np_namespace_still_exposes_ordinary_numpy_functions(defi_module):
    """The Fix 19 proxy must be a pass-through for everything it doesn't
    explicitly guard -- np.where/np.clip/np.array etc. must keep working
    exactly as the real module, or a huge range of otherwise-fine formulas
    would break."""
    code = (
        "def formula(x):\n"
        "    return np.where(x > 0, np.clip(x, 0, 10), np.array(0.0))\n"
    )
    X = np.array([[5.0], [-3.0], [20.0]])
    out = defi_module._execute_formula(code, X)
    assert out is not None
    assert np.allclose(out, [5.0, 0.0, 10.0])


# ─────────────────────────────────────────────────────────────────────────
# Selector unit tests (_select_v4_candidate)
# ─────────────────────────────────────────────────────────────────────────

def test_selector_tie_break_prefers_llm_when_formula_is_exact(defi_module, fast_nn):
    """An exact formula scores validation R2 = 1.0; blend also reaches 1.0.
    The deterministic tie-break (llm < residual < blend < nn) must pick llm."""
    X_tr, y_tr, _, _ = _in_domain_case()
    sel = defi_module._select_v4_candidate(
        X_tr, y_tr, EXACT_FORMULA, {}, CONFIG, seed=42, extrapolative=False)

    assert sel["validation_r2"]["llm"] == pytest.approx(1.0)
    # The tie is real: at least one non-llm candidate matches llm's score.
    assert any(v == pytest.approx(1.0) for k, v in sel["validation_r2"].items() if k != "llm")
    assert sel["selected"] == "llm"


def test_selector_extrapolative_without_llm_picks_simplest_nn_and_flags(defi_module, fast_nn):
    """Extrapolative test domain + no LLM candidate: pick the fewest-hidden-
    units architecture (not the best in-domain scorer) and flag it."""
    X_tr, y_tr, _, _ = _in_domain_case()
    sel = defi_module._select_v4_candidate(
        X_tr, y_tr, "", {}, CONFIG, seed=42, extrapolative=True)

    simplest = min(defi_module._V4_ARCHITECTURES, key=sum)
    assert sel["selected"] == "nn"
    assert list(sel["hidden"]) == list(simplest)
    assert sel["extrapolation_unmitigated"] is True


def test_selector_in_domain_without_llm_is_not_flagged(defi_module, fast_nn):
    """Negative control: same inputs but in-domain -> the guard must not fire."""
    X_tr, y_tr, _, _ = _in_domain_case()
    sel = defi_module._select_v4_candidate(
        X_tr, y_tr, "", {}, CONFIG, seed=42, extrapolative=False)

    assert sel["selected"] == "nn"
    assert sel["extrapolation_unmitigated"] is False


# ─────────────────────────────────────────────────────────────────────────
# Extrapolation guard vs. blend (Fix 18)
#
# blend(alpha=0) IS the bare NN. The blend grid contains 0.0, so blend's
# validation R2 is >= the NN's by construction and it wins ties -- which let a
# bare NN pass the extrapolation guard as "LLM-anchored". These tests make the
# NN's in-domain validation score PERFECT (so it is the validation winner by
# construction) and check that the guard still holds. The NN fitter is stubbed
# so the outcome doesn't depend on NN training noise.
# ─────────────────────────────────────────────────────────────────────────

def _truth(X):
    return 3.0 * X[:, 0] + 2.0 * X[:, 1]


def _install_oracle_nn(mod, monkeypatch):
    """NN stand-in that learns y perfectly, but is useless on any other
    target (e.g. a residual). => nn:* validation R2 == 1.0, residual == llm."""
    def fake(X_fit, y_fit, X_eval, hidden, seed, epochs=300, max_time_s=8):
        if np.allclose(y_fit, _truth(X_fit)):
            return _truth(X_eval), False
        return np.zeros(len(X_eval)), False
    monkeypatch.setattr(mod, "_fit_nn_predict", fake)


def test_extrapolative_blend_cannot_collapse_to_bare_nn(defi_module, monkeypatch):
    """Extrapolative + trusted-but-imperfect formula + NN that wins validation:
    the selector must NOT return a blend that is really the bare NN."""
    _install_oracle_nn(defi_module, monkeypatch)
    X_tr, y_tr, _, _ = _in_domain_case()
    floor = defi_module._V4_EXTRAP_MIN_BLEND_ALPHA

    sel = defi_module._select_v4_candidate(
        X_tr, y_tr, DECENT_FORMULA, {}, CONFIG, seed=42, extrapolative=True)

    # Premise: the NN really is the best in-domain scorer here.
    assert sel["validation_r2"]["nn:[64, 32]"] == pytest.approx(1.0)
    assert sel["selected"] != "nn"
    # Every blend on offer must keep real weight on the formula ...
    blends = [v for v in sel["validation_candidates"].values() if v["candidate"] == "blend"]
    assert blends and all(v["alpha"] >= floor - 1e-9 for v in blends), (
        f"extrapolative blend grid must not go below alpha={floor}: "
        f"{[v['alpha'] for v in blends]}")
    # ... so whatever won is genuinely formula-anchored.
    if sel["selected"] == "blend":
        assert sel["blend_alpha"] >= floor - 1e-9, (
            f"selected blend has alpha={sel['blend_alpha']} < {floor}: a bare NN "
            f"passed the extrapolation guard as 'LLM-anchored'")


def test_in_domain_blend_grid_is_unrestricted(defi_module, monkeypatch):
    """Negative control: the floor must apply ONLY under extrapolation. In-domain
    the perfect NN should still be free to win via blend(alpha=0)."""
    _install_oracle_nn(defi_module, monkeypatch)
    X_tr, y_tr, _, _ = _in_domain_case()

    sel = defi_module._select_v4_candidate(
        X_tr, y_tr, DECENT_FORMULA, {}, CONFIG, seed=42, extrapolative=False)

    assert sel["validation_r2"]["blend:[64, 32]"] == pytest.approx(1.0)
    assert sel["selected"] in ("blend", "nn")
    assert sel["blend_alpha"] == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────
# Ranked fallback walk after full-fit failure (Fix 15 / Fix 16)
#
# Fix 16 made the LLM "a normal ranked fallback rather than a special-case
# shortcut". These tests pin BOTH halves of that: the LLM is used when it is
# next best (Fix 15's intent), and it is NOT used when something else ranks
# above it (Fix 16's semantics).
# ─────────────────────────────────────────────────────────────────────────

def test_ranked_walk_lands_on_llm_when_it_is_next_best(defi_module, fast_nn, monkeypatch):
    case = _in_domain_case()
    monkeypatch.setattr(defi_module, "_select_v4_candidate", lambda *a, **k: _fake_selection(
        "residual_nn",
        [("residual_nn", [64, 32], 0.95), ("llm", None, 0.80), ("nn", [64, 32], 0.60)]))
    calls = _record_fit_calls(defi_module, monkeypatch, fail={"residual_nn"})

    result = _run_hybrid(defi_module, case, EXACT_FORMULA)

    assert result["llm_trustworthy"] is True
    assert result["selected_candidate"] == "llm_fallback"
    assert calls == ["residual_nn"], "walk should stop at llm without fitting any NN"
    assert result["extrapolation_unmitigated"] is True
    assert result["test_r2"] > 0.999


def test_ranked_walk_is_rank_based_not_hardwired_to_llm(defi_module, fast_nn, monkeypatch):
    """If a bare NN outranks the LLM on validation, the walk must take the NN
    -- proving the LLM is a ranked candidate, not a special case."""
    case = _in_domain_case()
    monkeypatch.setattr(defi_module, "_select_v4_candidate", lambda *a, **k: _fake_selection(
        "residual_nn",
        [("residual_nn", [64, 32], 0.95), ("nn", [64, 32], 0.80), ("llm", None, 0.60)]))
    calls = _record_fit_calls(defi_module, monkeypatch, fail={"residual_nn"})

    result = _run_hybrid(defi_module, case, EXACT_FORMULA)

    assert result["selected_candidate"] == "nn"          # not llm_fallback
    assert calls == ["residual_nn", "nn"]
    assert result["extrapolation_unmitigated"] is True


def test_ranked_walk_skips_llm_when_untrustworthy(defi_module, fast_nn, monkeypatch):
    """Even if the (stubbed) ranking lists `llm` high, an untrustworthy
    formula must never be returned as `llm_fallback`."""
    case = _in_domain_case()
    monkeypatch.setattr(defi_module, "_select_v4_candidate", lambda *a, **k: _fake_selection(
        "residual_nn",
        [("residual_nn", [64, 32], 0.95), ("llm", None, 0.90), ("nn", [64, 32], 0.50)]))
    # residual_nn fails on its own here: with the formula gated out there is
    # no llm_train/llm_test, so _fit_candidate_full returns None for it.
    calls = _record_fit_calls(defi_module, monkeypatch)

    result = _run_hybrid(defi_module, case, BAD_FORMULA)

    assert result["llm_trustworthy"] is False
    assert result["selected_candidate"] == "nn"
    assert calls == ["residual_nn", "nn"]


def test_ranked_walk_last_resort_is_nn_fallback(defi_module, fast_nn, monkeypatch):
    """No other ranked candidate exists -> the historical defensive NN path,
    reported as `nn_fallback`."""
    case = _in_domain_case()
    monkeypatch.setattr(defi_module, "_select_v4_candidate", lambda *a, **k: _fake_selection(
        "residual_nn", [("residual_nn", [64, 32], 0.95)]))
    _record_fit_calls(defi_module, monkeypatch, fail={"residual_nn"})

    result = _run_hybrid(defi_module, case, EXACT_FORMULA)

    assert result["selected_candidate"] == "nn_fallback"
    assert result["extrapolation_unmitigated"] is True
    assert np.isfinite(result["test_r2"])


def test_ranked_fallback_end_to_end_with_real_selector(defi_module, fast_nn, monkeypatch):
    """Integration check using the REAL selector (no stubbed ranking).

    The winner is whatever validation picks; we fail exactly that candidate's
    full fit and require the walk to land somewhere finite and different.
    Because the winner depends on NN training, a premise mismatch (validation
    picks llm outright) is skipped rather than failed -- the stubbed tests
    above are the authoritative guards.
    """
    case = _in_domain_case()
    baseline = _run_hybrid(defi_module, case, DECENT_FORMULA)
    winner = baseline["selected_candidate"]      # exact name, e.g. "residual_nn"
    print("BASELINE:", baseline)
    print("LLM TRUSTWORTHY:", baseline["llm_trustworthy"])
    print("SELECTED:", baseline["selected_candidate"])
    if winner == "llm":
        pytest.skip("validation picked llm outright; no full-fit failure to walk from")
    assert baseline["llm_trustworthy"] is True

    _record_fit_calls(defi_module, monkeypatch, fail={winner})
    result = _run_hybrid(defi_module, case, DECENT_FORMULA)

    assert result["selected_candidate"] != winner
    assert result["extrapolation_unmitigated"] is True
    assert np.isfinite(result["test_r2"])


def test_no_llm_formula_still_yields_usable_nn(defi_module, fast_nn):
    """Negative control: with no formula at all the hybrid must degrade to a
    working NN, not raise or return NaN (guards against a fix that makes the
    LLM mandatory)."""
    result = _run_hybrid(defi_module, _in_domain_case(), "")

    assert result["llm_trustworthy"] is False
    assert result["selected_candidate"] == "nn"
    assert result["extrapolation_unmitigated"] is False
    assert result["test_r2"] > 0.9


# ─────────────────────────────────────────────────────────────────────────
# Layer 2: opt-in aggregate statistical check (real runs)
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    os.environ.get("HYPATIAX_RUN_SLOW") != "1",
    reason="opt-in: set HYPATIAX_RUN_SLOW=1 (needs real protocol checkout, API key, baseline JSON)",
)
def test_aggregate_hybrid_does_not_regress_on_catalogue(defi_module):
    """Run the full catalogue and check, via a bootstrap CI rather than a
    fixed point threshold, that hybrid's mean test_r2 has not regressed
    against a checked-in baseline by more than `tolerance` (it fails only
    when the CI's UPPER bound is below baseline - tolerance).

    Per-case records (selected candidate, blend alpha, formula trust,
    llm_code) are written to $HYPATIAX_DIAG_OUT (default:
    <tmp>/hypatiax_v4_aggregate_diagnostics.json) and a grouped summary is
    printed and attached to the failure message, so a red run says WHICH
    cases collapsed and via which path.

    Note: this calls the hybrid WITHOUT `pure_llm_code`, so it makes its own
    LLM call per case (the pre-Fix-17 standalone path). That is deliberate for
    a strategy-level regression check, but it means results carry LLM
    non-determinism -- hence the CI and tolerance, and why this is not a
    commit gate. It checks regression vs. a baseline; it does not by itself
    show hybrid beats pure-LLM or pure-NN (that needs a paired comparison).

    Skips (rather than fails) if the protocol or baseline is unavailable.
    """
    try:
        protocol = defi_module.DeFiExperimentProtocol()
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"No real DeFiExperimentProtocol available: {e}")

    baseline_path = Path(__file__).resolve().parent / "v4_hybrid_baseline_r2.json"
    if not baseline_path.exists():
        pytest.skip(
            f"No checked-in baseline at {baseline_path}. Expected JSON: "
            f'{{"mean_test_r2": <float>, "tolerance": <float, default 0.03>}}'
        )

    baseline = json.loads(baseline_path.read_text())
    baseline_mean_r2 = float(baseline["mean_test_r2"])
    tolerance = float(baseline.get("tolerance", 0.03))

    records, n_unmatched = [], 0
    for tc in defi_module._get_test_cases():
        protocol_cases = protocol.load_test_data(tc["domain"], num_samples=tc["num_samples"])
        match = next(
            ((d, X, y, v, m) for d, X, y, v, m in protocol_cases
             if tc["name"].lower() in d.lower()),
            None,
        )
        if not match:
            n_unmatched += 1
            continue
        desc, X_full, y_full, var_names, metadata = match
        metadata = {**metadata, "constants": metadata.get("constants", {})}
        X_tr, y_tr, X_te, y_te = defi_module.build_extrap_split(X_full, y_full, tc["config"])
        rec = {"name": tc["name"], "domain": tc["domain"], "difficulty": tc.get("difficulty"),
               "extrapolates": bool(defi_module._v4_extrapolates(X_tr, X_te))}
        try:
            result = defi_module._v4_hybrid_predict_and_eval(
                desc, tc["domain"], X_tr, y_tr, X_te, y_te, var_names, metadata,
                tc["config"], seed=defi_module._NN_SEED,
            )
            rec.update(
                test_r2=float(result["test_r2"]),
                selected=result["selected_candidate"],
                blend_alpha=result["blend_alpha"],
                llm_trustworthy=result["llm_trustworthy"],
                llm_train_r2=result["llm_train_r2"],
                unmitigated=result["extrapolation_unmitigated"],
                validation_r2=result["validation_r2"],
                llm_code=result["llm_code"],   # lets you replay selection offline, no new LLM call
            )
        except Exception as e:  # keep the other cases' results: the run is ~18 min
            rec.update(test_r2=float("nan"), error=f"{type(e).__name__}: {e}")
        records.append(rec)
        print(f"  [{len(records):02d}] {tc['name'][:44]:<44} r2={rec['test_r2']:>10.4f}  "
              f"{rec.get('selected')!s:<12} trusted={rec.get('llm_trustworthy')}", flush=True)

    # Persist everything BEFORE asserting, so a red run is diagnosable without
    # paying for another 18 minutes of LLM calls.
    import tempfile
    diag_path = Path(os.environ.get(
        "HYPATIAX_DIAG_OUT",
        Path(tempfile.gettempdir()) / "hypatiax_v4_aggregate_diagnostics.json"))
    diag_path.write_text(json.dumps(records, indent=2, default=str))

    finite = [r for r in records if np.isfinite(r["test_r2"])]
    errors = [r for r in records if "error" in r]
    assert finite, "no case produced a finite test_r2 -- nothing to compare"
    scores = np.asarray([r["test_r2"] for r in finite])

    # Where does the shortfall come from? Group by (formula trusted?, what got selected).
    groups = {}
    for r in finite:
        groups.setdefault((r["llm_trustworthy"], r["selected"]), []).append(r["test_r2"])
    lines = [f"  trusted={k[0]!s:<5} selected={k[1]!s:<12} n={len(v):>3}  "
             f"mean={np.mean(v):>9.3f}  min={np.min(v):>10.3f}"
             for k, v in sorted(groups.items(), key=lambda kv: np.mean(kv[1]))]
    worst = sorted(finite, key=lambda r: r["test_r2"])[:8]
    worst_lines = [f"  {r['test_r2']:>10.3f}  {r['name'][:40]:<40} sel={r['selected']} "
                   f"trusted={r['llm_trustworthy']} llm_train_r2={r['llm_train_r2']:.3g} "
                   f"unmitigated={r['unmitigated']}" for r in worst]
    summary = (
        f"cases scored={len(finite)}/{len(records)} (unmatched in protocol={n_unmatched}, "
        f"errors={len(errors)}); mean={scores.mean():.4f} median={np.median(scores):.4f}\n"
        f"by (trusted, selected), worst first:\n" + "\n".join(lines) + "\n"
        f"worst cases:\n" + "\n".join(worst_lines) + "\n"
        f"full per-case records: {diag_path}"
    )
    print("\n" + summary)

    assert not errors, (
        f"{len(errors)} case(s) raised inside the hybrid: "
        f"{[(r['name'], r['error']) for r in errors[:5]]}\n{summary}"
    )

    rng = np.random.default_rng(0)
    boot_means = [rng.choice(scores, size=len(scores), replace=True).mean() for _ in range(2000)]
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    # Non-inferiority: fail only when the data show a REAL regression, i.e. even
    # the CI's UPPER bound is below baseline - tolerance. The previous form,
    # `ci_low > baseline - tolerance`, demanded that the LOWER bound clear the
    # baseline, which a system whose true mean equals the baseline passes only
    # ~10-15% of the time at n~74 -- a red test with no regression.
    assert ci_high >= baseline_mean_r2 - tolerance, (
        f"hybrid mean test_r2 regressed: 95% CI [{ci_low:.4f}, {ci_high:.4f}] "
        f"vs. baseline {baseline_mean_r2:.4f} (tolerance {tolerance})\n{summary}"
    )

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

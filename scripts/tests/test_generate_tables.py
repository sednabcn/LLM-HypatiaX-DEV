"""
scripts/tests/test_generate_tables.py

Regression tests for the parts of scripts/generate_tables.py that were
audited/changed in the Day 1 abstract-numbers work:

  - gen_defi_tiers()        (tab:difficulty — fixed to read the real Shape-3
                              schema instead of a dict-shape no generator
                              ever produces)
  - gen_hybrid_bug_breakdown() (tab:hybrid-bug-breakdown — the existing
                              fabricated-success / decision-attribution logic)
  - gen_abstract_macros()   (new — abstract's near-perfect rate + deltas,
                              computed from source, never hardcoded)

SCOPE — read this before assuming "all tests" means the whole file:
generate_tables.py has ~100 gen_*() functions covering every table in the
paper and its supplements. This suite does NOT exercise all of them; it
covers only the functions this thread's audit actually touched, plus the
cross-function consistency property (defi_tiers vs hybrid_bug_breakdown)
that was specifically claimed and needed a real regression test rather than
a one-off manual check. Treat this as "tests for the reviewed code," not
"tests for the whole pipeline." Extending coverage to the rest of the file
is a separate, much larger task.

Approach: invoke generate_tables.py as a real subprocess against a temp
--results-dir / --output-dir, exactly the way ci_postprocess.yml does. This
avoids relying on the module's import-time argparse/global state, and tests
the actual CLI entry point CI uses rather than internal functions directly.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "generate_tables.py"


# ── Fixtures: synthetic Shape-3 result sets ─────────────────────────────────

def _make_case(tier: str, llm_r2: float, nn_r2: float, hybrid_r2: float,
                decision: str) -> dict:
    return {
        "difficulty": tier,
        "results": {
            "pure_llm": {"test_r2": llm_r2},
            "neural_network": {"test_r2": nn_r2},
            "hybrid": {"test_r2": hybrid_r2, "decision": decision,
                       "success": hybrid_r2 > 0.99},
        },
    }


def _write_seed42(results_dir: Path, records: list[dict]) -> Path:
    defi_dir = results_dir / "defi"
    defi_dir.mkdir(parents=True, exist_ok=True)
    path = defi_dir / "hypatiax_defi_benchmark_v4_results_seed42.json"
    path.write_text(json.dumps(records))
    return path


def _tiers(n_easy: int, n_medium: int, n_hard: int) -> list[str]:
    return ["Easy"] * n_easy + ["Medium"] * n_medium + ["Hard"] * n_hard


@pytest.fixture
def genuine_improvement_dataset() -> list[dict]:
    """
    HypatiaX genuinely, independently outperforms both baselines on most
    cases, with a *few* honestly-fabricated cases mixed in (hybrid claims
    near-perfect via a routing decision whose named baseline does NOT
    confirm it) so the fabricated-detection path is exercised too, not just
    the trivial "everything is real" path.

    Deterministic hand-computed expectation (see test that uses this
    fixture): 74 cases, tiers 24/29/21.
    """
    tiers = _tiers(24, 29, 21)
    records = []
    for i, tier in enumerate(tiers):
        if i % 10 == 0:
            # A fabricated case: hybrid claims near-perfect via "llm" routing,
            # but pure_llm itself does not confirm > 0.99.
            records.append(_make_case(tier, llm_r2=0.4, nn_r2=0.1, hybrid_r2=0.995, decision="llm"))
        elif i % 3 == 0:
            # Genuine hybrid success, independently confirmed -- split across
            # llm- and nn-routed decisions (not just "llm") so the corrected
            # rate actually exceeds EACH individual baseline's own rate.
            # NOTE: if every genuine pass here were routed via "llm" only (as
            # a prior version of this fixture did), n_corrected_pass would be
            # identical to n_llm_pass by construction -- the correction
            # algebra guarantees a non-fabricated "llm"-routed pass is always
            # also a pure_llm pass, and vice versa for that subset -- so the
            # LLM delta could never be positive and the "must not be
            # withheld" assertion below would be unreachable. See DAY02
            # findings for how this was caught.
            if i % 2 == 0:
                records.append(_make_case(tier, llm_r2=0.995, nn_r2=0.1, hybrid_r2=0.995, decision="llm"))
            else:
                records.append(_make_case(tier, llm_r2=0.1, nn_r2=0.995, hybrid_r2=0.995, decision="nn"))
        else:
            # Hybrid does not claim near-perfect; low baselines.
            records.append(_make_case(tier, llm_r2=0.3, nn_r2=0.05, hybrid_r2=0.6, decision="ensemble"))
    return records


@pytest.fixture
def no_real_gain_dataset() -> list[dict]:
    """
    Every hybrid 'success' is fabricated (claims near-perfect via a routing
    decision whose named baseline does not confirm it), while the raw LLM
    baseline is strong. Regression fixture for the "never claim a false
    gain" behaviour: abstractDeltaLLM must be withheld here.
    """
    tiers = _tiers(24, 29, 21)
    records = []
    for i, tier in enumerate(tiers):
        llm_r2 = 0.995 if i % 5 == 0 else 0.5
        nn_r2 = 0.995 if i % 20 == 0 else 0.2
        decision = "llm" if i % 3 == 0 else ("nn" if i % 3 == 1 else "nn_fallback")
        records.append(_make_case(tier, llm_r2=llm_r2, nn_r2=nn_r2, hybrid_r2=0.995, decision=decision))
    return records


def _run_generate_tables(tmp_path: Path, records: list[dict] | None,
                          experiment: str = "exp1") -> tuple[subprocess.CompletedProcess, Path]:
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    if records is not None:
        _write_seed42(results_dir, records)
    else:
        results_dir.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--experiment", experiment,
         "--results-dir", str(results_dir),
         "--output-dir", str(output_dir)],
        capture_output=True, text=True, timeout=120,
    )
    return proc, output_dir


def _read(output_dir: Path, name: str) -> str | None:
    p = output_dir / name
    return p.read_text() if p.exists() else None


def _macro_value(tex: str, name: str) -> str | None:
    m = re.search(r"\\newcommand\{\\" + re.escape(name) + r"\}\{([^}]*)\}", tex)
    return m.group(1) if m else None


def _table_row_values(tex: str, label: str) -> tuple[str, str, str, str] | None:
    """Pulls (n, pure_llm_pct, hypatiax_pct, gain) out of a defi_tiers.tex row."""
    m = re.search(re.escape(label) + r" & (\d+) & ([\d.\-]+|---) & ([\d.\-]+|---) & ([\d.+\-]+|---)", tex)
    return m.groups() if m else None


def _bug_breakdown_row(tex: str, label: str) -> tuple[str, str, str, str] | None:
    """Pulls (n, reported, fabricated, corrected) out of hybrid_bug_breakdown.tex."""
    m = re.search(re.escape(label) + r" & (\d+) & (\d+) & (\d+) & (\d+)", tex)
    return m.groups() if m else None


# ── Cross-function consistency: defi_tiers vs hybrid_bug_breakdown ─────────

@pytest.mark.parametrize("dataset_fixture", ["genuine_improvement_dataset", "no_real_gain_dataset"])
def test_defi_tiers_matches_hybrid_bug_breakdown(tmp_path, request, dataset_fixture):
    """
    defi_tiers.tex's HypatiaX column and hybrid_bug_breakdown.tex's
    "Corrected successes" column must always agree per-tier and Overall,
    since corrected == reported_pass AND baseline_confirms == hyp_pass by
    construction (see the algebra in the prior conversation turn). This is
    the property that was previously only checked manually; it's now a real
    regression test so a future edit to either function that breaks the
    identity fails CI instead of silently drifting.
    """
    records = request.getfixturevalue(dataset_fixture)
    proc, output_dir = _run_generate_tables(tmp_path, records)
    assert proc.returncode == 0, proc.stderr

    tiers_tex = _read(output_dir, "defi_tiers.tex")
    bug_tex = _read(output_dir, "hybrid_bug_breakdown.tex")
    assert tiers_tex is not None, f"defi_tiers.tex was not written.\nstderr:\n{proc.stderr}"
    assert bug_tex is not None, f"hybrid_bug_breakdown.tex was not written.\nstderr:\n{proc.stderr}"

    for label in ("Easy", "Medium", "Hard", "Overall"):
        tier_row = _table_row_values(tiers_tex, label)
        bug_row = _bug_breakdown_row(bug_tex, label)
        assert tier_row is not None, f"Could not find '{label}' row in defi_tiers.tex"
        assert bug_row is not None, f"Could not find '{label}' row in hybrid_bug_breakdown.tex"

        n_tier, _llm_pct, hyp_pct_str, _gain = tier_row
        n_bug, reported, fabricated, corrected = bug_row

        assert n_tier == n_bug, f"{label}: tier n ({n_tier}) != bug-breakdown n ({n_bug})"

        # Recompute the HypatiaX % from the bug-breakdown's raw counts and
        # compare against defi_tiers' reported percentage — independent of
        # which function's internal arithmetic is "trusted".
        n = int(n_tier)
        expected_pct = round(100.0 * int(corrected) / n, 1) if n else 0.0
        assert hyp_pct_str != "---", f"{label}: defi_tiers HypatiaX cell was skipped ('---')"
        assert abs(float(hyp_pct_str) - expected_pct) < 0.05, (
            f"{label}: defi_tiers HypatiaX%={hyp_pct_str} does not match "
            f"corrected/n = {corrected}/{n} = {expected_pct}"
        )
        assert int(reported) - int(fabricated) == int(corrected), (
            f"{label}: hybrid_bug_breakdown's own arithmetic is inconsistent "
            f"({reported} - {fabricated} != {corrected})"
        )


# ── Numeric correctness against hand-computed expectations ─────────────────

def test_genuine_improvement_dataset_hand_computed_totals(tmp_path, genuine_improvement_dataset):
    """
    Independently hand-computes expected totals for genuine_improvement_dataset
    (not by re-running the same correction logic) and checks generate_tables.py
    reproduces them exactly. This is the check that actually guards against a
    shared bug in both generators, since the expectation here is computed by
    a separate loop in the test, not by calling into generate_tables.py.
    """
    records = genuine_improvement_dataset
    n_total = len(records)
    assert n_total == 74

    n_fabricated = 0
    n_corrected_pass = 0
    n_llm_pass = 0
    n_nn_pass = 0
    baseline_map = {"llm": "pure_llm", "nn": "neural_network", "nn_fallback": "neural_network"}
    for rec in records:
        cr = rec["results"]
        if cr["pure_llm"]["test_r2"] > 0.99:
            n_llm_pass += 1
        if cr["neural_network"]["test_r2"] > 0.99:
            n_nn_pass += 1
        hybrid = cr["hybrid"]
        if hybrid["test_r2"] > 0.99:
            baseline_key = baseline_map.get(hybrid["decision"])
            baseline_r2 = cr[baseline_key]["test_r2"] if baseline_key else hybrid["test_r2"]
            if baseline_r2 > 0.99:
                n_corrected_pass += 1
            else:
                n_fabricated += 1

    expected_hyp_rate = round(100.0 * n_corrected_pass / n_total, 1)
    expected_llm_rate = round(100.0 * n_llm_pass / n_total, 1)
    expected_nn_rate = round(100.0 * n_nn_pass / n_total, 1)

    proc, output_dir = _run_generate_tables(tmp_path, records)
    assert proc.returncode == 0, proc.stderr

    abstract_tex = _read(output_dir, "abstract_macros.tex")
    assert abstract_tex is not None, f"abstract_macros.tex was not written.\nstderr:\n{proc.stderr}"

    got_hyp = _macro_value(abstract_tex, "abstractNearPerfectRate")
    got_llm = _macro_value(abstract_tex, "abstractLLMBaseline")
    got_nn = _macro_value(abstract_tex, "abstractNNBaseline")
    assert got_hyp is not None and abs(float(got_hyp) - expected_hyp_rate) < 0.05
    assert got_llm is not None and abs(float(got_llm) - expected_llm_rate) < 0.05
    assert got_nn is not None and abs(float(got_nn) - expected_nn_rate) < 0.05

    # This fixture is constructed so HypatiaX genuinely beats both baselines;
    # both delta macros must therefore be present (not withheld).
    got_delta_llm = _macro_value(abstract_tex, "abstractDeltaLLM")
    got_delta_nn = _macro_value(abstract_tex, "abstractDeltaNN")
    assert got_delta_llm is not None, "abstractDeltaLLM was withheld despite a genuine improvement"
    assert got_delta_nn is not None, "abstractDeltaNN was withheld despite a genuine improvement"
    assert abs(float(got_delta_llm) - (expected_hyp_rate - expected_llm_rate)) < 0.05
    assert abs(float(got_delta_nn) - (expected_hyp_rate - expected_nn_rate)) < 0.05

    # Sanity: this fixture must actually exercise the fabrication path,
    # or the test isn't testing what it claims to.
    assert n_fabricated > 0


# ── Safety behaviour: never claim a false / reversed gain ──────────────────

def test_abstract_macros_withholds_delta_when_no_real_gain(tmp_path, no_real_gain_dataset):
    """
    Regression test for the exact behaviour demonstrated manually earlier in
    this conversation: when the corrected HypatiaX rate does not actually
    exceed a baseline, that baseline's delta macro must be absent from
    abstract_macros.tex (never written as a negative/zero 'gain'), and the
    reason must be surfaced in the run's skipped-items reporting.
    """
    proc, output_dir = _run_generate_tables(tmp_path, no_real_gain_dataset)
    assert proc.returncode == 0, proc.stderr

    abstract_tex = _read(output_dir, "abstract_macros.tex")
    assert abstract_tex is not None

    # abstractNearPerfectRate/LLMBaseline/NNBaseline are always written...
    assert _macro_value(abstract_tex, "abstractNearPerfectRate") is not None
    assert _macro_value(abstract_tex, "abstractLLMBaseline") is not None
    # ...but the LLM delta must be withheld given this fixture's construction
    # (LLM baseline beats the corrected hybrid rate here by design).
    assert _macro_value(abstract_tex, "abstractDeltaLLM") is None, (
        "abstractDeltaLLM was written even though the corrected HypatiaX "
        "rate is not an improvement over the LLM baseline in this fixture"
    )
    combined_output = proc.stdout + proc.stderr
    assert "abstractDeltaLLM withheld" in combined_output, (
        "the withheld-delta reason was not surfaced in stdout/stderr"
    )


# ── Never fabricate when source data is missing/malformed ──────────────────

def test_no_macros_written_when_source_json_missing(tmp_path):
    """No seed-42 JSON present at all -> abstract_macros.tex must not exist,
    and the run must report it as skipped rather than exiting silently."""
    proc, output_dir = _run_generate_tables(tmp_path, records=None)
    assert proc.returncode == 0, proc.stderr
    assert _read(output_dir, "abstract_macros.tex") is None
    combined_output = proc.stdout + proc.stderr
    assert "abstract_macros.tex" in combined_output
    assert "SKIPPED" in combined_output or "skip" in combined_output.lower()


def test_defi_tiers_skipped_on_legacy_dict_shape(tmp_path):
    """
    Regression guard for the exact bug this thread found and fixed: a
    dict-with-per-tier-subdicts JSON (the shape no generator in this
    codebase has ever produced on disk) must be skipped, not silently
    "parsed" into a table — and it must specifically NOT reappear as a
    written defi_tiers.tex with fabricated-looking numbers.
    """
    results_dir = tmp_path / "results"
    defi_dir = results_dir / "defi"
    defi_dir.mkdir(parents=True, exist_ok=True)
    legacy_shape = {
        "easy":    {"n": 24, "llm_r99": 0.5, "hypatiax_r99": 0.6},
        "medium":  {"n": 29, "llm_r99": 0.5, "hypatiax_r99": 0.6},
        "hard":    {"n": 21, "llm_r99": 0.5, "hypatiax_r99": 0.6},
        "overall": {"n": 74, "llm_r99": 0.5, "hypatiax_r99": 0.6},
    }
    (defi_dir / "hypatiax_defi_benchmark_v4_results_seed42.json").write_text(json.dumps(legacy_shape))

    output_dir = tmp_path / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--experiment", "exp1",
         "--results-dir", str(results_dir),
         "--output-dir", str(output_dir)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert _read(output_dir, "defi_tiers.tex") is None
    assert _read(output_dir, "abstract_macros.tex") is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

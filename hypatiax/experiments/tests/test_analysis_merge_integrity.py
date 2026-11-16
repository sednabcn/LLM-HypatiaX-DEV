"""Offline regression contracts for the post-run analysis pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / ".github" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import merge_extrap_into_benchmark as extrap_merge
import merge_shards
import result_subdir_map
import validate_analysis_input


def test_equation_identity_does_not_collapse_to_domain_alias():
    first = {"equation": "Annualised Portfolio tracking error", "domain": "DeFi Risk"}
    second = {"equation": "Portfolio VaR for two correlated", "domain": "DeFi Risk"}

    assert merge_shards.canonical_task_id(first) == "Annualised Portfolio tracking error"
    assert merge_shards.canonical_task_id(second) == "Portfolio VaR for two correlated"
    assert merge_shards.canonical_task_id(first) != merge_shards.canonical_task_id(second)
    assert merge_shards.canonical_task_id({"test_case": "Impermanent loss"}) == "Impermanent loss"
    assert merge_shards.canonical_task_id({"domain": "DeFi Risk"}) is None
    assert merge_shards.normalise_row({"domain": "DeFi Risk"}) is None


def test_normalise_row_keeps_seed_as_part_of_task_identity():
    row = merge_shards.normalise_row({
        "equation_id": "N1",
        "name": "Nguyen 1",
        "domain": "physics",
        "seed": 99,
        "hypatia": {"train_r2": 0.9},
    })

    assert row is not None
    assert row["task_id"] == "N1__seed99"
    assert row["domain"] is None


def test_protocol_wrapper_uses_protocol_normalizer_and_preserves_equations():
    payload = {
        "tests": [
            {
                "equation_id": "eq-A",
                "domain": "shared-domain",
                "results": {"pure_llm": {"r2": 0.81}, "neural_network": {"r2": 0.71}},
            },
            {
                "equation_id": "eq-B",
                "domain": "shared-domain",
                "results": {"pure_llm": {"r2": 0.82}, "neural_network": {"r2": 0.72}},
            },
        ]
    }

    rows = merge_shards.extract_rows(payload)

    assert [row["task_id"] for row in rows] == ["eq-A", "eq-B"]
    assert rows[0]["hypatia"]["extrap_r2"] == 0.81
    assert rows[0]["nn"]["extrap_r2"] == 0.71


def test_merge_rows_combines_complementary_values_and_source_provenance():
    merged = merge_shards.merge_rows([
        {
            "task_id": "eq-A",
            "name": "Equation A",
            "hypatia": {"train_r2": 0.9},
            "source_file": "shard0.json",
        },
        {
            "task_id": "eq-A",
            "name": "Equation A",
            "hypatia": {"extrap_r2": 0.8},
            "source_file": "shard1.json",
        },
    ])

    assert merged["eq-A"]["hypatia"] == {"train_r2": 0.9, "extrap_r2": 0.8}
    assert merged["eq-A"]["source_files"] == ["shard0.json", "shard1.json"]


def test_merge_rows_fails_on_conflicting_duplicate_metric():
    with pytest.raises(ValueError, match="conflicting duplicate task_id"):
        merge_shards.merge_rows([
            {"task_id": "eq-A", "hypatia": {"train_r2": 0.9}, "source_file": "a.json"},
            {"task_id": "eq-A", "hypatia": {"train_r2": 0.7}, "source_file": "b.json"},
        ])


def test_sweep_merge_fails_on_conflicting_duplicate_point(tmp_path):
    payload_a = {
        "noise_levels": [0.05],
        "methods": ["Method A"],
        "per_noise": {
            "0.05": {
                "method_summary": {"Method A": {"median_r2": 0.9}},
                "per_equation": {"eq-A": {"Method A": {"r2": 0.9}}},
            }
        },
    }
    payload_b = {
        "noise_levels": [0.05],
        "methods": ["Method A"],
        "per_noise": {
            "0.05": {
                "method_summary": {"Method A": {"median_r2": 0.8}},
                "per_equation": {"eq-A": {"Method A": {"r2": 0.8}}},
            }
        },
    }
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    first.write_text(json.dumps(payload_a))
    second.write_text(json.dumps(payload_b))

    with pytest.raises(RuntimeError, match="conflicting duplicate sweep point"):
        merge_shards.merge_sweep_files([first, second], "suppB")


def test_extrap_pairing_uses_exact_method_not_best_role_score():
    benchmark = [
        {"test": "eq-A", "domain": "biology", "method": "SymbolicEngineWithLLM (tools)", "r2": 0.91, "success": True},
        {"test": "eq-A", "domain": "biology", "method": "HybridDiscoverySystem v50_2 (tools)", "r2": 0.92, "success": True},
        {"test": "eq-A", "domain": "biology", "method": "EnhancedHybridSystemDeFi (core)", "r2": 0.99, "success": True},
    ]
    extrap = {
        "eq-A": {
            "SymbolicEngineWithLLM (tools)": {"method": "SymbolicEngineWithLLM (tools)", "train_r2": 0.89, "extrap_r2_far": 0.41},
            "HybridDiscoverySystem v50_2 (tools)": {"method": "HybridDiscoverySystem v50_2 (tools)", "train_r2": 0.93, "extrap_r2_far": 0.31},
            "EnhancedHybridSystemDeFi (core)": {"method": "EnhancedHybridSystemDeFi (core)", "train_r2": 0.98, "extrap_r2_far": 0.99},
        }
    }

    paired = extrap_merge.merge(benchmark, extrap, {})[0]

    assert paired["hypatia"]["method"] == extrap_merge.PRIMARY_HYPATIA_METHOD
    assert paired["hypatia"]["train_r2"] == 0.89
    assert paired["hypatia"]["benchmark_reference_train_r2"] == 0.91
    assert paired["hypatia"]["extrap_r2_far"] == 0.41
    assert paired["pysr_only"]["train_r2"] == 0.93
    assert paired["pysr_only"]["benchmark_reference_train_r2"] == 0.92
    assert paired["pysr_only"]["extrap_r2_far"] == 0.31
    assert paired["method_results"]["EnhancedHybridSystemDeFi (core)"]["extrap_r2_far"] == 0.99
    assert paired["method_count"] == 3


def test_legacy_role_fallback_does_not_borrow_another_methods_train_score():
    benchmark = [
        {"test": "eq-A", "domain": "biology", "method": "SymbolicEngineWithLLM (tools)", "r2": 0.91, "success": True},
    ]
    legacy = {"eq-A": {"hypatia": {"extrap_r2_far": 0.8}}}

    paired = extrap_merge.merge(benchmark, {}, legacy)[0]

    assert paired["hypatia"]["extrap_r2_far"] is None
    assert paired["hypatia"]["train_r2"] is None
    assert paired["legacy_role_results"]["hypatia"]["extrap_r2_far"] == 0.8
    assert paired["legacy_role_results"]["hypatia"]["source"] == "legacy_role_only_no_method_identity"


def test_result_subdirectory_rejects_traversal_and_keeps_five_system_paths():
    assert result_subdir_map.validate_result_subdir("five_systems/exp1_five") == "five_systems/exp1_five"
    with pytest.raises(ValueError, match="traversal"):
        result_subdir_map.validate_result_subdir("../outside")


def test_analysis_input_path_must_stay_inside_result_directory(tmp_path):
    result_root = tmp_path / "results"
    result_root.mkdir()
    valid = result_root / "input.json"
    valid.write_text("{}")
    outside = tmp_path / "outside.json"
    outside.write_text("{}")

    assert validate_analysis_input._resolve_input_path(str(valid), result_root, "INPUT_JSON") == valid.resolve()
    with pytest.raises(ValueError, match="escapes RESULT_DIR"):
        validate_analysis_input._resolve_input_path(str(outside), result_root, "INPUT_JSON")


def test_checkpoint_records_source_hashes_and_shard_ids(tmp_path):
    root = tmp_path / "shards"
    root.mkdir()
    shard = root / "result_shard3.json"
    shard.write_text('{"task": 1}')
    checkpoint = root / "_checkpoint.json"

    merge_shards.write_checkpoint(
        checkpoint,
        "exp1_ablation",
        "ablation/exp1_ablation",
        {"eq-A": {}},
        [shard],
        root,
    )
    data = json.loads(checkpoint.read_text())

    assert data["source_file_count"] == 1
    assert data["source_files"][0]["path"] == "result_shard3.json"
    assert len(data["source_files"][0]["sha256"]) == 64
    assert data["shard_ids_observed"] == [3]


def test_configured_shard_coverage_is_exact():
    files = [Path(f"exp1_ablation_results_shard{i}.json") for i in range(4)]
    assert merge_shards.validate_shard_coverage("exp1_ablation", files) == {0, 1, 2, 3}

    with pytest.raises(RuntimeError, match="incomplete shard coverage"):
        merge_shards.validate_shard_coverage(
            "exp1_ablation",
            [Path("exp1_ablation_results_shard0.json"), Path("exp1_ablation_results_shard1.json")],
        )


def test_result_subdirectory_rejects_symlink_escape(tmp_path):
    results_root = tmp_path / "results"
    outside = tmp_path / "outside"
    results_root.mkdir()
    outside.mkdir()
    (results_root / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="outside"):
        result_subdir_map.validate_result_subdir("escape", str(results_root))


def test_analysis_workflow_does_not_swallow_stash_failures_or_duplicate_routing():
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci_analysis.yml").read_text()
    locate = (SCRIPTS_DIR / "locate_analysis_input.sh").read_text()
    model_validation = (
        REPO_ROOT / ".github" / "workflows" / "ci_model_validation.yml"
    ).read_text()

    assert "git stash pop || true" not in workflow
    assert "SINGLE_SHARD_EXPERIMENTS" not in workflow
    assert "result_subdir_map.py" in workflow
    assert "commit_analysis_outputs.sh" in workflow
    assert "EXPERIMENT=\"$EXPERIMENT\" python3 -c" in locate
    assert "'$EXPERIMENT' in MERGE_REQUIRED_EXPERIMENTS" not in locate
    assert (
        "ANTHROPIC_API_KEY: ${{ github.event_name == 'schedule' "
        "&& secrets.ANTHROPIC_API_KEY || '' }}"
    ) in model_validation


def test_extrap_merge_rejects_conflicting_duplicate_method_rows():
    benchmark = [
        {"test": "eq-A", "method": "SymbolicEngineWithLLM (tools)", "r2": 0.91},
        {"test": "eq-A", "method": "SymbolicEngineWithLLM (tools)", "r2": 0.81},
    ]

    with pytest.raises(ValueError, match="conflicting duplicate"):
        extrap_merge.merge(benchmark, {}, {})


def _sweep_payload(methods):
    return {
        "noise_levels": [0.05],
        "methods": list(methods),
        "per_noise": {
            "0.0500": {
                "method_summary": {method: {"median_r2": 0.9} for method in methods},
                "per_equation": {
                    f"eq-{index:02d}": {
                        method: {"r2": 0.9, "success": True}
                        for method in methods
                    }
                    for index in range(30)
                },
            }
        },
    }


def test_sweep_merge_requires_all_six_methods_for_all_equations(tmp_path):
    payload = _sweep_payload(merge_shards.EXPECTED_SWEEP_METHODS)
    source = tmp_path / "noise_sweep_nshards01.json"
    source.write_text(json.dumps(payload))

    merged = merge_shards.merge_sweep_files([source], "suppB")
    assert len(merged["per_noise"]["0.0500"]["per_equation"]) == 30
    assert set(merged["methods"]) == set(merge_shards.EXPECTED_SWEEP_METHODS)


def test_sweep_merge_rejects_missing_method_in_one_equation(tmp_path):
    payload = _sweep_payload(merge_shards.EXPECTED_SWEEP_METHODS)
    first_equation = next(iter(payload["per_noise"]["0.0500"]["per_equation"]))
    payload["per_noise"]["0.0500"]["per_equation"][first_equation].pop(
        next(iter(merge_shards.EXPECTED_SWEEP_METHODS))
    )
    source = tmp_path / "noise_sweep_nshards01.json"
    source.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="incomplete method records"):
        merge_shards.merge_sweep_files([source], "suppB")


def test_sweep_merge_rejects_missing_equation(tmp_path):
    payload = _sweep_payload(merge_shards.EXPECTED_SWEEP_METHODS)
    payload["per_noise"]["0.0500"]["per_equation"].pop("eq-29")
    source = tmp_path / "noise_sweep_nshards01.json"
    source.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="29/30 equation records"):
        merge_shards.merge_sweep_files([source], "suppB")


def test_extrap_validation_requires_30_equations_and_six_methods():
    benchmark = []
    extrap = {}
    for equation_index in range(30):
        equation = f"Feynman equation {equation_index:02d}"
        extrap[equation] = {}
        for method in merge_shards.EXPECTED_SWEEP_METHODS:
            benchmark.append({"test": equation, "method": method, "r2": 0.9})
            extrap[equation][method] = {"method": method, "extrap_r2_far": 0.8}

    extrap_merge.validate_method_coverage(benchmark, extrap)


def test_extrap_validation_rejects_missing_method():
    benchmark = []
    extrap = {}
    for equation_index in range(30):
        equation = f"Feynman equation {equation_index:02d}"
        extrap[equation] = {}
        for method in merge_shards.EXPECTED_SWEEP_METHODS:
            benchmark.append({"test": equation, "method": method, "r2": 0.9})
            extrap[equation][method] = {"method": method, "extrap_r2_far": 0.8}
    extrap["Feynman equation 00"].pop(next(iter(merge_shards.EXPECTED_SWEEP_METHODS)))

    with pytest.raises(ValueError, match="method coverage incomplete"):
        extrap_merge.validate_method_coverage(benchmark, extrap)


def test_core15_task_coverage_requires_both_arms_and_all_15_equations():
    row = {
        "hypatia": {"train_r2": 0.9, "extrap_r2_far": 0.8, "success": True},
        "pysr_only": {"train_r2": 0.8, "extrap_r2_far": 0.7, "success": True},
    }
    merged = {f"eq-{index:02d}": row for index in range(15)}
    merge_shards.validate_task_coverage("exp1_ablation", merged)

    with pytest.raises(RuntimeError, match="15 required task records"):
        merge_shards.validate_task_coverage("exp1_ablation", dict(list(merged.items())[:-1]))


def test_nguyen12_task_coverage_requires_four_complete_seeds():
    merged = {}
    for seed in (42, 99, 123, 777):
        for equation_index in range(12):
            equation = f"N{equation_index + 1}"
            task_id = f"{equation}__seed{seed}"
            merged[task_id] = {
                "seed": seed,
                "nguyen_id": equation,
                "systems": {"hypatiax": {"r2_raw": 0.9}, "pysr": {"r2_raw": 0.8}},
            }

    merge_shards.validate_task_coverage("exp3b", merged)
    merged["N1__seed42"]["systems"].pop("pysr")
    with pytest.raises(RuntimeError, match="lacks both hypatiax and pysr"):
        merge_shards.validate_task_coverage("exp3b", merged)


def test_experiment_id_validation_rejects_path_fragments():
    import subprocess

    script = SCRIPTS_DIR / "result_subdir_map.py"
    result = subprocess.run(
        [sys.executable, str(script), "../outside"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "experiment must be a simple identifier" in result.stderr

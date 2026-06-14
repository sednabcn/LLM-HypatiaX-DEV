"""
gate_c3_verify.py  —  AST-level Gate A / B / C check for FIX-C3.

Gate A: pca_directed_split is imported in run_comparative_suite_benchmark_pca.py
Gate B: pca_directed_split is *called* (not just imported) inside ImprovedNN.run()
Gate C: train_test_split is NOT called as primary split inside ImprovedNN.run()
        (an aliased import used only in _tts_internal is acceptable)

Exit 0 = all gates pass.  Exit 1 = one or more gates fail.

Called by ci_paper_audit.yml:
    python scripts/patches/gate_c3_verify.py \\
        hypatiax/experiments/benchmarks/run_comparative_suite_benchmark_pca.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def _find_calls(tree: ast.AST, func_name: str) -> list[ast.Call]:
    """Return all Call nodes where the function name matches func_name."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == func_name)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == func_name)
        )
    ]


def _get_method_body(tree: ast.AST, class_name: str, method_name: str) -> ast.AST | None:
    """Extract the AST subtree for Class.method."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    return None


def check_gate(pca_runner: Path) -> dict[str, bool | str]:
    src = pca_runner.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(pca_runner))

    results: dict[str, bool | str] = {}

    # ── Gate A: pca_directed_split imported ──────────────────────────────────
    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.asname or alias.name)

    results["gate_a_import"] = "pca_directed_split" in imported_names
    results["gate_a_detail"] = (
        "pca_directed_split found in imports"
        if results["gate_a_import"]
        else "pca_directed_split NOT found in imports"
    )

    # ── Gate B: pca_directed_split called inside ImprovedNN.run() ────────────
    run_method = _get_method_body(tree, "ImprovedNN", "run")
    if run_method is None:
        results["gate_b_called_in_run"] = False
        results["gate_b_detail"] = "ImprovedNN.run() not found"
    else:
        pca_calls = _find_calls(run_method, "pca_directed_split")
        results["gate_b_called_in_run"] = len(pca_calls) > 0
        results["gate_b_detail"] = f"{len(pca_calls)} call(s) found"

    # ── Gate C: train_test_split NOT called inside ImprovedNN.run() ──────────
    if run_method is not None:
        tts_calls = _find_calls(run_method, "train_test_split")
        tts_internal_calls = _find_calls(run_method, "_tts_internal")
        direct_calls = len(tts_calls)
        results["gate_c_no_random_split"] = direct_calls == 0
        results["gate_c_detail"] = (
            f"{direct_calls} direct train_test_split call(s); "
            f"{len(tts_internal_calls)} _tts_internal call(s) [acceptable]"
        )
    else:
        results["gate_c_no_random_split"] = False
        results["gate_c_detail"] = "ImprovedNN.run() not found"

    return results


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        pca_runner = Path(
            "hypatiax/experiments/benchmarks/run_comparative_suite_benchmark_pca.py"
        )
    else:
        pca_runner = Path(argv[1])

    if not pca_runner.exists():
        print(f"ERROR: {pca_runner} not found", file=sys.stderr)
        return 1

    print(f"Checking: {pca_runner}")
    print("-" * 60)
    results = check_gate(pca_runner)

    all_pass = True
    for key, value in results.items():
        if key.endswith("_detail"):
            continue
        gate_id = key.upper().replace("_", " ")
        status = "PASS" if value else "FAIL"
        detail = results.get(f"{key}_detail", "")
        print(f"  [{status}]  {gate_id}" + (f"  ({detail})" if detail else ""))
        if not value:
            all_pass = False

    print("-" * 60)
    if all_pass:
        print("FIX-C3 Gates A/B/C: ALL PASSED")
        return 0
    else:
        print("FIX-C3 Gates A/B/C: ONE OR MORE FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))

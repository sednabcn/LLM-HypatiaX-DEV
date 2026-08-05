"""
gate_c3_verify.py  —  AST-level Gate A / B / C check for FIX-C3.

Gate A: pca_directed_split is imported in run_comparative_suite_benchmark_pca.py
Gate B: pca_directed_split is *called* (not just imported) inside ImprovedNN.run()
Gate C: train_test_split is NOT called as primary split inside ImprovedNN.run()
        (a train_test_split import aliased to the literal local name
        `_tts_internal` is treated as an acceptable internal helper and
        excluded; calls aliased to any OTHER name still count and fail
        this gate — this includes module-qualified calls and simple
        one-hop local variable reassignment, e.g. `splitter = train_test_split`)

Exit 0 = all gates pass.  Exit 1 = one or more gates fail.

Called by ci_paper_audit.yml:
    python scripts/patches/gate_c3_verify.py \\
        hypatiax/experiments/benchmarks/run_comparative_suite_benchmark_pca.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def _build_import_map(tree: ast.AST) -> dict[str, str]:
    """
    Map every local name introduced by import statements to its canonical
    (originally-imported) name.

      from x import train_test_split            -> {"train_test_split": "train_test_split"}
      from x import train_test_split as tts     -> {"tts": "train_test_split"}
      import sklearn.model_selection as sms     -> {"sms": "sklearn.model_selection"}

    Module-qualified attribute calls (sms.train_test_split) are resolved
    separately in _resolve_call_name by attribute name, so this map only
    needs to cover direct-name imports (ImportFrom) reliably.
    """
    import_map: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                local = alias.asname or alias.name
                import_map[local] = alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name
                import_map[local] = alias.name
    return import_map


def _build_reassignment_map(tree: ast.AST, import_map: dict[str, str]) -> dict[str, str]:
    """
    Catch simple one-hop local renames of the form:
        splitter = train_test_split
        splitter = some_module.train_test_split

    Maps the new local name -> canonical name, resolved through import_map.
    Does not attempt to track deeper indirection (dict/list storage, function
    returns, conditional rebinding, etc.) — those are out of scope.
    """
    reassign_map: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = node.value
        if isinstance(value, ast.Name):
            canonical = import_map.get(value.id, value.id)
            reassign_map[target.id] = canonical
        elif isinstance(value, ast.Attribute):
            # e.g. splitter = sms.train_test_split -> resolve by attribute name
            reassign_map[target.id] = value.attr
    return reassign_map


def _resolve_call_name(node: ast.Call, import_map: dict[str, str],
                        reassign_map: dict[str, str]) -> str | None:
    """
    Resolve what a Call node's callee actually refers to, in canonical-name
    terms, by following (in order): reassignment map -> import map -> literal
    token. Handles both `name(...)` and `module.name(...)` call shapes.
    """
    func = node.func
    if isinstance(func, ast.Name):
        token = func.id
        if token in reassign_map:
            return reassign_map[token]
        return import_map.get(token, token)
    elif isinstance(func, ast.Attribute):
        # module.func(...) or obj.func(...) — match on attribute name; this
        # already self-resolves regardless of what the module/object alias is.
        return func.attr
    return None


def _call_site_local_name(node: ast.Call) -> str | None:
    """
    Return the literal local token written at the call site (before any
    alias/reassignment resolution) — e.g. for `_tts_internal(...)` this is
    "_tts_internal", regardless of what it's actually imported from.
    Used only for the documented _tts_internal name-based exception.
    """
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    elif isinstance(func, ast.Attribute):
        return func.attr
    return None


def _find_calls(tree: ast.AST, func_name: str,
                 import_map: dict[str, str] | None = None,
                 reassign_map: dict[str, str] | None = None,
                 exempt_local_names: set[str] | None = None) -> list[ast.Call]:
    """
    Return all Call nodes that resolve to func_name.

    If import_map/reassign_map are provided, resolves aliased and renamed
    call targets back to their canonical name before matching, so that e.g.
    `from x import train_test_split as tts; tts(...)` or
    `splitter = train_test_split; splitter(...)` are still detected.
    Falls back to literal-token matching if no maps are given.

    exempt_local_names: call sites whose literal local token (the name as
    written at the call site, e.g. "_tts_internal") is in this set are
    excluded from the results even if they resolve to func_name. This backs
    the documented exception that a train_test_split import aliased to the
    literal name `_tts_internal` is an acceptable internal helper.
    """
    import_map = import_map or {}
    reassign_map = reassign_map or {}
    exempt_local_names = exempt_local_names or set()
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        resolved = _resolve_call_name(node, import_map, reassign_map)
        if resolved != func_name:
            continue
        if _call_site_local_name(node) in exempt_local_names:
            continue
        matches.append(node)
    return matches


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

    # ── Shared alias-resolution maps ──────────────────────────────────────────
    import_map = _build_import_map(tree)
    reassign_map = _build_reassignment_map(tree, import_map)

    # ── Gate A: pca_directed_split imported (any alias) ──────────────────────
    # Check canonical names (import targets), not just local bound names, so
    # `from hypatiax.splits import pca_directed_split as pds` still counts.
    canonical_imports = set(import_map.values())
    results["gate_a_import"] = "pca_directed_split" in canonical_imports
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
        pca_calls = _find_calls(run_method, "pca_directed_split", import_map, reassign_map)
        results["gate_b_called_in_run"] = len(pca_calls) > 0
        results["gate_b_detail"] = f"{len(pca_calls)} call(s) found"

    # ── Gate C: train_test_split NOT called as primary split in run() ────────
    # Exception (per module docstring): a train_test_split import aliased to
    # the literal local name `_tts_internal` is treated as an acceptable
    # internal helper and excluded from the count, even though it resolves
    # to the same canonical function. Any OTHER alias/rename still counts.
    TTS_EXEMPT_NAMES = {"_tts_internal"}
    if run_method is not None:
        tts_calls = _find_calls(
            run_method, "train_test_split", import_map, reassign_map,
            exempt_local_names=TTS_EXEMPT_NAMES,
        )
        tts_internal_calls = _find_calls(run_method, "train_test_split", import_map, reassign_map)
        tts_internal_calls = [
            c for c in tts_internal_calls
            if _call_site_local_name(c) in TTS_EXEMPT_NAMES
        ]
        direct_calls = len(tts_calls)
        results["gate_c_no_random_split"] = direct_calls == 0
        results["gate_c_detail"] = (
            f"{direct_calls} direct train_test_split call(s) (alias-resolved); "
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

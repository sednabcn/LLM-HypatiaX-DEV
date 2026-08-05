#!/usr/bin/env python3
"""
rescore_nulls.py

Regenerates the held-out test split deterministically for the Feynman
benchmark (no raw arrays were cached in the JSON) and re-scores every
equation using a PATCHED evaluator.

Supports TWO split protocols, auto-detected per-record from the JSON:

  EXTRAP records -- have extrap_train_frac / extrap_multiplier set.
    Split is regenerated with fixed_mod.build_extrap_split(train_frac=...,
    multiplier=...). Scored against extrap_r2_far / extrap_n_train /
    extrap_n_test.

  PCA records -- have pca_split_protocol set (e.g. "pca_40_60"), and
    extrap_train_frac / extrap_multiplier are None for these records.
    Split is regenerated with pca_directed_split(X, y, test_size=...)
    where test_size is parsed from pca_split_protocol. Scored against
    pca_test_r2 / pca_n_train / pca_n_test -- NOT extrap_r2_far, even
    though the two maps may coincidentally hold identical values.

Two-phase approach, in this order, on purpose, PER split type:

  PHASE 1 -- SELF-VERIFICATION (do this first, always)
    Regenerate data for the equations that ALREADY have a real,
    non-null recorded score in the original JSON, and confirm the patched
    evaluator reproduces the SAME formula-evaluation result at the
    SAME split sizes as what's already recorded. This is the only way
    to confirm noise_level/seed/num_samples/split-params are all correct
    BEFORE trusting phase 2's output for the previously-null equations --
    if phase 1 doesn't match, do not trust phase 2 until it does.

  PHASE 2 -- RESCORE THE NULLS
    Only run this once phase 1 confirms the regeneration parameters are
    correct. Regenerates the far/test split for each null equation and
    computes a real R^2 using the already-discovered formula string
    (no re-running PySR/LLM discovery -- the formula is reused as-is).

Usage (extrap run):
    python rescore_nulls.py \
        --shard-dir hypatiax/data/results/comparison_results/feynman-tests/exp2_extrap \
        --pattern "protocol_core_extrap_20260722_*.json" \
        --method "HybridDiscoverySystem v50_2 (tools)" \
        --protocol-file experiment_protocol_benchmark_v2.py \
        --fixed-evaluator run_comparative_suite_benchmark_v2_FIXED.py \
        --noise-level 0.0    # try 0.0 first; if phase 1 fails, try 0.05

Usage (PCA run):
    python rescore_nulls.py \
        --shard-dir hypatiax/data/results/comparison_results/feynman-tests/exp2_pca_4060 \
        --pattern "protocol_core_noiseless_pca_20260723_*.json" \
        --method "HybridDiscoverySystem v50_2 (tools)" \
        --protocol-file experiment_protocol_benchmark_v2.py \
        --fixed-evaluator run_comparative_suite_benchmark_pca_FIXED.py \
        --pca-split-utils pca_split_utils.py \
        --noise-level 0.0

Records are auto-detected and routed to the right protocol; --split-type
can force extrap-only or pca-only handling if a shard dir mixes both and
you want to isolate one.
"""
import argparse
import glob
import importlib.util
import json
import math
import os
import sys

import numpy as np


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_pca_split_protocol(protocol_str):
    """
    Parse strings like "pca_40_60" -> test_size=0.6 (the second number is
    the TEST fraction, matching pca_n_train=80/pca_n_test=120 seen in the
    July-23 shards, i.e. 40% train / 60% test).

    Returns None if the string doesn't match the expected pattern --
    callers should treat that as "can't verify this record automatically,
    skip it" rather than guessing.
    """
    if not protocol_str:
        return None
    parts = protocol_str.strip().split("_")
    nums = [p for p in parts if p.isdigit()]
    if len(nums) < 2:
        return None
    train_pct, test_pct = int(nums[-2]), int(nums[-1])
    if train_pct + test_pct != 100 or test_pct <= 0 or test_pct >= 100:
        return None
    return test_pct / 100.0


def find_equation_in_protocol(protocol, description, domain):
    """
    Search across all Feynman domains in the protocol for the (description,
    X, y, var_names, metadata) tuple matching this equation. We search by
    domain first (fast), then fall back to searching every domain if not
    found there (in case of a domain-naming mismatch between the JSON's
    'domain' field and the protocol's internal domain keys).
    """
    domains_to_try = [domain] + [d for d in protocol.get_all_domains() if d != domain]
    for dom in domains_to_try:
        try:
            data = protocol.load_test_data(dom)
        except KeyError:
            continue
        for desc, X, y, var_names, metadata in data:
            if desc.strip() == description.strip():
                return X, y, var_names, metadata, dom
    return None


def diag_eval_formula(fixed_mod, formula, X_far, var_names):
    """
    Same three strategies as the real _runner_eval_formula, but returns
    (y_pred, error_log) so EVAL FAIL cases show the real exception instead
    of a bare None.
    """
    import numpy as np
    errors = []
    code = formula.strip()
    code = fixed_mod.re.sub(r"^\s*\[[^\]]*\]\s*", "", code)
    code = code.replace("^", "**")

    safe_globals = None
    # Reuse the exact same safe_globals construction path by calling the
    # real function first -- if it succeeds, no need to diagnose further.
    y_pred = fixed_mod._runner_eval_formula(formula, X_far, var_names)
    if y_pred is not None:
        return y_pred, []

    # Real function returned None -- rebuild its three strategies here
    # with exceptions surfaced, using the SAME cleaned `code` it would use.
    import math as _math
    try:
        import scipy.special as _spsp
    except ImportError:
        _spsp = None
    safe_globals = {
        "__builtins__": {}, "np": np, "numpy": np, "math": _math,
        "pi": np.pi, "e": np.e, "inf": np.inf, "nan": np.nan,
        "exp": lambda x: np.exp(np.clip(x, -500.0, 500.0)),
        "log": np.log, "log10": np.log10, "log2": np.log2, "sqrt": np.sqrt,
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "arcsin": lambda x: np.arcsin(np.clip(x, -1.0, 1.0)),
        "arccos": lambda x: np.arccos(np.clip(x, -1.0, 1.0)),
        "arctan": np.arctan, "arctan2": np.arctan2,
        "abs": np.abs, "fabs": np.abs, "floor": np.floor, "ceil": np.ceil,
        "sign": np.sign, "power": np.power, "tanh": np.tanh,
        "sinh": np.sinh, "cosh": np.cosh,
        "safe_asin": lambda x: np.arcsin(np.clip(x, -1.0, 1.0)),
        "safe_acos": lambda x: np.arccos(np.clip(x, -1.0, 1.0)),
    }
    local_ns = {}
    for i, vn in enumerate(var_names):
        local_ns[vn] = X_far[:, i] if X_far.ndim == 2 else X_far

    try:
        eval(code, safe_globals, local_ns)
    except Exception as e:
        errors.append(f"Strategy1(eval): {type(e).__name__}: {e}")
    try:
        exec_ns = {**safe_globals, **local_ns}
        exec(code, exec_ns)
        found_candidate = any(
            c in exec_ns and isinstance(exec_ns[c], (np.ndarray, float, int))
            for c in ("y", "result", "output", "pred", "f")
        )
        if not found_candidate:
            errors.append(f"Strategy2(exec): ran fine, but no whitelisted "
                           f"output var found. Vars defined: "
                           f"{[k for k in exec_ns if not k.startswith('_') and k not in safe_globals]}")
    except Exception as e:
        errors.append(f"Strategy2(exec): {type(e).__name__}: {e}")

    errors.append(f"var_names available: {var_names}")
    errors.append(f"code after fix: {code[:150]}")
    return None, errors


def collect_aug_features(formula, var_names):
    """
    Faithful copy of _collect_aug_features from run_comparative_suite_benchmark_v2.py
    (lines 3456-3486). Finds every ratio_*/gm_* token in `formula` and resolves
    it to the pair of base variable names it's derived from.
    """
    import re as _re
    feats = []
    for tok in _re.findall(r"\b(ratio_\w+|gm_\w+)\b", formula):
        kind = "ratio" if tok.startswith("ratio_") else "gm"
        body = tok[len(kind) + 1:]
        found = False
        for sep_idx in range(1, len(body)):
            if body[sep_idx] == "_":
                a_nm = body[:sep_idx]
                b_nm = body[sep_idx + 1:]
                if a_nm in var_names and b_nm in var_names:
                    ai = var_names.index(a_nm)
                    bi = var_names.index(b_nm)
                    feats.append((tok, ai, bi, kind))
                    found = True
                    break
        if not found:
            pass  # unresolvable token -- formula will still fail, same as original
    return feats


def build_aug(X_base, feat_specs, existing_names):
    """
    Faithful copy of _build_aug (lines 3488-3520). Appends engineered
    ratio_*/gm_* columns to X_base.
    """
    if not feat_specs:
        return X_base, list(existing_names)
    extra_cols = []
    extra_names = []
    seen = set(existing_names)
    for feat_name, ai, bi, kind in feat_specs:
        if feat_name in seen:
            continue
        col_a = X_base[:, ai]
        col_b = X_base[:, bi]
        if kind == "ratio":
            col = col_a / (col_b + 1e-12)
        else:
            col = np.sqrt(np.abs(col_a * col_b) + 1e-12)
        extra_cols.append(col)
        extra_names.append(feat_name)
        seen.add(feat_name)
    if not extra_cols:
        return X_base, list(existing_names)
    return (
        np.hstack([X_base, np.column_stack(extra_cols)]),
        list(existing_names) + extra_names,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--pattern", default="protocol_core_extrap_*.json")
    ap.add_argument("--method", required=True)
    ap.add_argument("--protocol-file", required=True)
    ap.add_argument("--fixed-evaluator", required=True)
    ap.add_argument("--noise-level", type=float, default=0.0)
    ap.add_argument("--threshold", type=float, default=0.999999)
    ap.add_argument("--split-type", choices=["auto", "extrap", "pca"], default="auto",
                     help="Force handling of only extrap or only pca records. "
                          "Default 'auto' detects per-record from pca_split_protocol.")
    ap.add_argument("--pca-split-utils", default="pca_split_utils.py",
                     help="Path to a module providing pca_directed_split(X, y, "
                          "test_size, random_state). Only loaded if any PCA "
                          "records are encountered.")
    ap.add_argument("--pca-random-state", type=int, default=None,
                     help="random_state passed to pca_directed_split. PCA with "
                          "n_components=1 is deterministic under sklearn's full "
                          "SVD solver, so this usually shouldn't matter, but "
                          "Phase 1 will reveal it via SIZE/R2 mismatches if it does.")
    args = ap.parse_args()

    proto_mod = load_module(args.protocol_file, "protocol_mod")
    fixed_mod = load_module(args.fixed_evaluator, "fixed_mod")

    pca_directed_split = None  # loaded lazily below, only if PCA records show up

    # Adjust this if BenchmarkProtocol lives under a different name/module
    # path in your actual file -- this assumes the class shown in this
    # session's inspection (BenchmarkProtocol, benchmark="feynman").
    protocol = proto_mod.BenchmarkProtocol(
        benchmark="feynman", noise_level=args.noise_level
    )

    files = sorted(glob.glob(os.path.join(args.shard_dir, args.pattern)))
    if not files:
        print(f"No files matched {args.pattern!r} in {args.shard_dir!r}", file=sys.stderr)
        sys.exit(1)

    already_scored = []   # list of entry dicts, see below
    nulls = []
    n_pca_records = 0
    n_extrap_records = 0
    n_unrecognized_split = 0

    for fp in files:
        with open(fp) as f:
            data = json.load(f)
        records = data.get("tests", data if isinstance(data, list) else [data])
        for rec in records:
            block = (rec.get("results") or {}).get(args.method, {})
            if not block:
                continue
            formula = (block.get("formula") or "").strip()

            split_protocol = rec.get("pca_split_protocol")
            is_pca_record = split_protocol is not None

            if args.split_type == "extrap" and is_pca_record:
                continue
            if args.split_type == "pca" and not is_pca_record:
                continue

            if is_pca_record:
                score_map = rec.get("pca_test_r2") or {}
                if args.method not in score_map:
                    continue
                n_pca_records += 1
                entry = {
                    "desc": rec.get("description", ""),
                    "domain": rec.get("domain", ""),
                    "recorded_r2": score_map[args.method],
                    "formula": formula,
                    "is_pca": True,
                    "split_protocol": split_protocol,
                    "n_train": rec.get("pca_n_train"),
                    "n_test": rec.get("pca_n_test"),
                }
            else:
                far_map = rec.get("extrap_r2_far") or {}
                if args.method not in far_map:
                    continue
                n_extrap_records += 1
                entry = {
                    "desc": rec.get("description", ""),
                    "domain": rec.get("domain", ""),
                    "recorded_r2": far_map[args.method],
                    "formula": formula,
                    "is_pca": False,
                    "train_frac": rec.get("extrap_train_frac"),
                    "multiplier": rec.get("extrap_multiplier"),
                    "n_train": rec.get("extrap_n_train"),
                    "n_test": rec.get("extrap_n_test"),
                }

            if entry["recorded_r2"] is None:
                nulls.append(entry)
            else:
                already_scored.append(entry)

    if n_pca_records:
        print(f"Detected {n_pca_records} PCA-split record(s) and "
              f"{n_extrap_records} extrap-split record(s) for method "
              f"{args.method!r}.")
        try:
            pca_utils_mod = load_module(args.pca_split_utils, "pca_utils_mod")
            pca_directed_split = pca_utils_mod.pca_directed_split
        except (ImportError, FileNotFoundError, AttributeError) as e:
            print(f"ERROR: {n_pca_records} PCA record(s) found but could not load "
                  f"pca_directed_split from {args.pca_split_utils!r}: {e}", file=sys.stderr)
            print("Pass --pca-split-utils pointing at a module that defines "
                  "pca_directed_split(X, y, test_size, random_state), or use "
                  "--split-type extrap to skip PCA records entirely.", file=sys.stderr)
            sys.exit(1)

    # ---------------- PHASE 1: self-verification ----------------
    print("=" * 70)
    print("PHASE 1: self-verification against already-scored equations")
    print("=" * 70)
    phase1_ok = 0
    phase1_total = 0
    for entry in already_scored:
        desc = entry["desc"]
        domain = entry["domain"]
        recorded_r2 = entry["recorded_r2"]
        formula = entry["formula"]
        n_train = entry["n_train"]
        n_test = entry["n_test"]

        if not formula:
            continue

        if entry["is_pca"]:
            test_size = parse_pca_split_protocol(entry["split_protocol"])
            if test_size is None:
                n_unrecognized_split += 1
                print(f"[SKIP] '{desc}': could not parse pca_split_protocol "
                      f"{entry['split_protocol']!r}")
                continue
        else:
            if entry["train_frac"] is None or entry["multiplier"] is None:
                continue

        phase1_total += 1
        found = find_equation_in_protocol(protocol, desc, domain)
        if found is None:
            print(f"[MISS] Could not locate '{desc}' in protocol data at all.")
            continue
        X, y, var_names, metadata, found_domain = found

        if entry["is_pca"]:
            X_train, X_far, y_train, y_far = pca_directed_split(
                X, y, test_size=test_size, random_state=args.pca_random_state
            )
        else:
            X_train, y_train, X_far, y_far, split_meta = fixed_mod.build_extrap_split(
                X, y, description=desc, train_frac=entry["train_frac"],
                multiplier=entry["multiplier"]
            )
        size_ok = (len(X_train) == n_train) and (len(X_far) == n_test)
        if not size_ok:
            print(f"[SIZE MISMATCH] '{desc}' ({'pca' if entry['is_pca'] else 'extrap'}): "
                  f"got train={len(X_train)}/far={len(X_far)}, "
                  f"expected train={n_train}/far={n_test} -- "
                  f"{'random_state or test_size is probably wrong' if entry['is_pca'] else 'num_samples is probably wrong'}")
            continue
        if len(X_far) < 2:
            continue

        # Reconstruct any ratio_*/gm_* engineered features this formula
        # references, matching the real pipeline's FIX-N3a/N3b behaviour --
        # otherwise formulas using these will always NameError.
        aug_specs = collect_aug_features(formula, var_names)
        X_far_aug, aug_names = build_aug(X_far, aug_specs, var_names)

        y_far_pred = fixed_mod._runner_eval_formula(formula, X_far_aug, aug_names)
        if y_far_pred is None:
            _, err_log = diag_eval_formula(fixed_mod, formula, X_far_aug, aug_names)
            print(f"[EVAL FAIL] '{desc}':")
            for line in err_log:
                print(f"    {line}")
            continue
        ss_res = np.sum((y_far - y_far_pred) ** 2)
        ss_tot = np.sum((y_far - y_far.mean()) ** 2)
        recomputed_r2 = 1 - ss_res / ss_tot if ss_tot > 1e-300 else float("nan")
        recorded_is_catastrophic = (
            recorded_r2 is not None and
            (recorded_r2 == float("-inf")) or
            (isinstance(recorded_r2, (int, float)) and recorded_r2 < -1000)
        )
        recomputed_is_catastrophic = (
            not math.isnan(recomputed_r2) and recomputed_r2 < -1000
        )
        if recorded_is_catastrophic and recomputed_is_catastrophic:
            match = True  # both agree "catastrophic", exact magnitude doesn't need to match
        else:
            match = math.isclose(recomputed_r2, recorded_r2, rel_tol=1e-3, abs_tol=1e-6)
        if match:
            phase1_ok += 1
        else:
            print(f"[R2 MISMATCH] '{desc}': recomputed={recomputed_r2:.6f} vs "
                  f"recorded={recorded_r2}")

    print(f"\nPhase 1 result: {phase1_ok}/{phase1_total} already-scored equations "
          f"reproduce exactly with noise_level={args.noise_level}.")
    if n_unrecognized_split:
        print(f"({n_unrecognized_split} record(s) skipped -- unrecognized "
              f"pca_split_protocol format.)")
    if phase1_total == 0 or phase1_ok < phase1_total:
        print("\nDO NOT TRUST PHASE 2 YET. Try --noise-level 0.05 instead (or vice versa). "
              "If PCA records are involved and sizes mismatch, also try a different "
              "--pca-random-state. Or check that --protocol-file / class name / "
              "method match your actual repo.")
        print("Not running phase 2.")
        return

    # ---------------- PHASE 2: rescore the nulls ----------------
    print("\n" + "=" * 70)
    print("PHASE 2: rescoring the null equations")
    print("=" * 70)
    results = []
    for entry in nulls:
        desc = entry["desc"]
        domain = entry["domain"]
        formula = entry["formula"]
        if not formula:
            print(f"[SKIP] '{desc}': no formula stored -- not a re-evaluatable case")
            continue
        found = find_equation_in_protocol(protocol, desc, domain)
        if found is None:
            print(f"[MISS] Could not locate '{desc}' in protocol data.")
            continue
        X, y, var_names, metadata, found_domain = found

        if entry["is_pca"]:
            test_size = parse_pca_split_protocol(entry["split_protocol"])
            if test_size is None:
                print(f"[SKIP] '{desc}': could not parse pca_split_protocol "
                      f"{entry['split_protocol']!r}")
                continue
            X_train, X_far, y_train, y_far = pca_directed_split(
                X, y, test_size=test_size, random_state=args.pca_random_state
            )
        else:
            if entry["train_frac"] is None or entry["multiplier"] is None:
                print(f"[SKIP] '{desc}': missing extrap_train_frac/extrap_multiplier")
                continue
            X_train, y_train, X_far, y_far, split_meta = fixed_mod.build_extrap_split(
                X, y, description=desc, train_frac=entry["train_frac"],
                multiplier=entry["multiplier"]
            )
        if len(X_far) < 2:
            print(f"[SKIP] '{desc}': far/test region too small even after regeneration")
            continue

        aug_specs = collect_aug_features(formula, var_names)
        X_far_aug, aug_names = build_aug(X_far, aug_specs, var_names)

        y_far_pred = fixed_mod._runner_eval_formula(formula, X_far_aug, aug_names)
        if y_far_pred is None:
            _, err_log = diag_eval_formula(fixed_mod, formula, X_far_aug, aug_names)
            print(f"[{desc}] -> still None:")
            for line in err_log:
                print(f"    {line}")
            continue
        ss_res = np.sum((y_far - y_far_pred) ** 2)
        ss_tot = np.sum((y_far - y_far.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-300 else float("nan")
        status = "PASS" if (not math.isnan(r2) and r2 >= args.threshold) else \
                 ("NaN (domain issue)" if math.isnan(r2) or np.isnan(y_far_pred).any() else "fail")
        results.append((desc, r2, status))
        print(f"  {desc:55s} -> R2={r2 if not math.isnan(r2) else 'NaN':>10}  [{status}]")

    n_pass = sum(1 for _, _, s in results if s == "PASS")
    n_already_pass = sum(
        1 for e in already_scored
        if e["recorded_r2"] is not None and e["recorded_r2"] >= args.threshold
    )
    print(f"\nOf {len(nulls)} previously-null equations: {len(results)} re-evaluated, "
          f"{n_pass} now pass at R2 >= {args.threshold}.")
    print(f"New total ({n_already_pass} previously confirmed + {n_pass} recovered) "
          f"= {n_already_pass + n_pass}/{len(already_scored) + len(nulls)}")


if __name__ == "__main__":
    main()

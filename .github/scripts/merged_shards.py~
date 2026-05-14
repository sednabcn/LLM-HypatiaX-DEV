# Refactor Consolidation Logic into `.github/scripts/merged_shards.py`

import glob
import json
import os
from datetime import datetime, timezone

import numpy as np
from scipy import stats as scipy_stats

EXP        = os.environ["EXP"]
ART_DIR    = "downloaded_artifacts"
RESULT_SUB = os.environ["RESULT_SUBDIR"]
OUT_DIR    = os.path.join("hypatiax/data/results", RESULT_SUB)
os.makedirs(OUT_DIR, exist_ok=True)

ALL_PENDING = json.loads(os.environ.get("ALL_PENDING", "[]"))

_DEFI_IDS = frozenset({
    "amm",
    "risk_var",
    "liquidity",
    "expected_shortfall",
    "liquidation",
    "risk",
    "lending",
    "staking",
    "trading",
    "derivatives",
})

_META_KEYS = frozenset({
    "purelm_truncation_audit",
    "experiment",
    "generated_at",
    "source_run_id",
    "n_total",
    "n_merged",
    "n_successes",
    "success_rate",
    "hyp_extrap_mean",
    "hyp_extrap_median",
    "nn_extrap_mean",
    "nn_extrap_median",
    "config",
    "metadata",
    "run_info",
    "summary",
    "protocol",
})

_RESULT_KEYS = frozenset({
    "equation_id",
    "task_id",
    "id",
    "protocol",
    "hypatia",
    "nn",
    "r2",
    "rmse",
    "success",
    "best_expression",
    "domain",
    "name",
    "formula",
    "pure_llm",
    "neural_network",
})


def _normalise(item):

    if not isinstance(item, dict):
        return item

    result = dict(item)

    inner = result.get("results")

    if isinstance(inner, dict):

        inner = dict(inner)

        if "pure_llm" in inner and "hypatia" not in inner:
            inner["hypatia"] = inner.pop("pure_llm")

        if "neural_network" in inner and "nn" not in inner:
            inner["nn"] = inner.pop("neural_network")

        for sub in (inner.get("hypatia") or {}, inner.get("nn") or {}):
            if (
                isinstance(sub, dict)
                and "extrap_r2" not in sub
                and "test_r2" in sub
            ):
                sub["extrap_r2"] = sub["test_r2"]

        result["results"] = inner
        result.update(inner)

    else:

        if "pure_llm" in result and "hypatia" not in result:
            result["hypatia"] = result.pop("pure_llm")

        if "neural_network" in result and "nn" not in result:
            result["nn"] = result.pop("neural_network")

        for sub in (result.get("hypatia") or {}, result.get("nn") or {}):
            if (
                isinstance(sub, dict)
                and "extrap_r2" not in sub
                and "test_r2" in sub
            ):
                sub["extrap_r2"] = sub["test_r2"]

    return result


candidates = sorted(glob.glob(f"{ART_DIR}/**/*.json", recursive=True))

print(f"\nCandidate JSON files: {len(candidates)}")

for p in candidates:
    print(f"  {p}")

merged = {}

for path in candidates:

    try:

        with open(path) as f:
            data = json.load(f)

        print("\n" + "=" * 70)
        print(f"DEBUG FILE: {path}")
        print(f"TOP LEVEL TYPE: {type(data).__name__}")

        if isinstance(data, dict):
            print(f"TOP LEVEL KEYS: {list(data.keys())[:20]}")

        rows = {}

        # Strategy A
        if isinstance(data, list):

            print("Strategy A fired")

            for item in data:

                if not isinstance(item, dict):
                    continue

                key = (
                    item.get("equation_id")
                    or item.get("task_id")
                    or item.get("id")
                    or item.get("protocol")
                )

                if key is not None:
                    rows[key] = _normalise(item)

        # Strategy B
        elif (
            isinstance(data, dict)
            and "results" in data
            and isinstance(data["results"], list)
        ):

            print("Strategy B fired")

            rows = {
                (
                    r.get("equation_id")
                    or r.get("task_id")
                    or r.get("id")
                ): _normalise(r)
                for r in data["results"]
                if isinstance(r, dict)
            }

        # Strategy C
        elif isinstance(data, dict) and "task_id" in data:

            print("Strategy C fired")

            rows = {
                data["task_id"]: _normalise(data)
            }

        # Strategy D2
        elif (
            isinstance(data, dict)
            and any(k in _DEFI_IDS for k in data.keys())
        ):

            print("Strategy D2 fired")

            rows = {
                k: _normalise(v)
                for k, v in data.items()
                if (
                    k in _DEFI_IDS
                    and isinstance(v, dict)
                )
            }

        # Strategy D
        elif isinstance(data, dict):

            print("Strategy D fired")

            rows = {
                k: _normalise(v)
                for k, v in data.items()
                if (
                    isinstance(v, dict)
                    and k not in _META_KEYS
                    and any(rk in v for rk in _RESULT_KEYS)
                )
            }

        rows = {
            k: v
            for k, v in rows.items()
            if k is not None
        }

        overlap = set(merged).intersection(rows)

        if overlap:
            print(f"WARNING overlap keys: {sorted(overlap)}")

        merged.update(rows)

        if rows:
            print(f"+{len(rows)} rows from {os.path.basename(path)}")
            print(f"ROW KEYS: {sorted(rows.keys())}")
        else:
            print(f"(no task rows) {os.path.basename(path)}")

        print("=" * 70)

    except Exception as e:

        print(f"SKIP {path}: {e}")


print(f"\nTotal merged: {len(merged)} entries")

missing = sorted(set(ALL_PENDING) - set(merged))

if ALL_PENDING:

    pct = 100.0 * len(merged) / len(ALL_PENDING)

    print(
        f"Coverage: {len(merged)}/{len(ALL_PENDING)} "
        f"({pct:.1f}%)"
    )

    if missing:
        print(f"Missing ({len(missing)}): {missing}")


with open(os.path.join(OUT_DIR, f"{EXP}_merged.json"), "w") as f:
    json.dump(merged, f, indent=2)


hyp, nn = [], []

for r in merged.values():

    if not isinstance(r, dict):
        continue

    hr2 = (r.get("hypatia") or {}).get("extrap_r2")
    nr2 = (r.get("nn") or {}).get("extrap_r2")

    if hr2 is not None:
        hyp.append(float(hr2))

    if nr2 is not None:
        nn.append(float(nr2))


successes = [v for v in hyp if v > 0.99]

stats = {
    "experiment": EXP,
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "source_run_id": os.environ.get("SRC_RUN_ID", ""),
    "n_total": len(ALL_PENDING) if ALL_PENDING else len(merged),
    "n_merged": len(merged),
    "n_successes": len(successes),
    "success_rate": len(successes) / len(merged) if merged else None,
    "hyp_extrap_mean": float(np.mean(hyp)) if hyp else None,
    "hyp_extrap_median": float(np.median(hyp)) if hyp else None,
    "nn_extrap_mean": float(np.mean(nn)) if nn else None,
    "nn_extrap_median": float(np.median(nn)) if nn else None,
}

if len(hyp) >= 5 and len(nn) >= 5:
    u, p = scipy_stats.mannwhitneyu(hyp, nn, alternative="greater")
    stats.update({
        "mw_U": float(u),
        "mw_p": float(p),
        "mw_significant": bool(p < 0.05),
    })

with open(os.path.join(OUT_DIR, f"{EXP}_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)

print(f"\nOutputs written to: {OUT_DIR}")

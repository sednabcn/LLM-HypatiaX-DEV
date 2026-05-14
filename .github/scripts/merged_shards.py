import os
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from scipy import stats as scipy_stats


# =========================
# Config
# =========================

EXP = os.environ.get("EXP", "exp1")
RESULT_SUBDIR = os.environ["RESULT_SUBDIR"]
ALL_PENDING = json.loads(os.environ.get("ALL_PENDING", "[]"))

BASE_DIR = Path("hypatiax/data/results")
RESULT_DIR = BASE_DIR / RESULT_SUBDIR
OUT_DIR = RESULT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

MERGED_FILE = OUT_DIR / f"{EXP}_merged.json"


# =========================
# Constants
# =========================

_DEFI_IDS = {
    "amm", "risk_var", "liquidity", "expected_shortfall",
    "liquidation", "risk", "lending", "staking",
    "trading", "derivatives",
}

_META_KEYS = {
    "experiment", "generated_at", "source_run_id",
    "n_total", "n_merged", "n_successes",
    "success_rate", "config", "metadata",
}

_RESULT_KEYS = {
    "equation_id", "task_id", "id",
    "hypatia", "nn", "r2", "rmse",
    "success", "best_expression",
}


# =========================
# Normalization
# =========================

def _normalize(item: dict) -> dict:
    if not isinstance(item, dict):
        return item

    item = dict(item)

    inner = item.get("results")

    # unify nested structure
    if isinstance(inner, dict):
        inner = dict(inner)

        if "pure_llm" in inner and "hypatia" not in inner:
            inner["hypatia"] = inner.pop("pure_llm")

        if "neural_network" in inner and "nn" not in inner:
            inner["nn"] = inner.pop("neural_network")

        item.update(inner)
        item["results"] = inner

    return item


# =========================
# Loaders
# =========================

def load_merged_file(path: Path) -> dict:
    print(f"[INFO] Loading merged file: {path}")
    with open(path) as f:
        return json.load(f)


def load_shards(directory: Path) -> list[Path]:
    files = list(directory.rglob("*.json"))
    print(f"[INFO] Found {len(files)} JSON files in {directory}")
    return files


# =========================
# Parsing logic
# =========================

def extract_rows(data):
    rows = {}

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                key = item.get("equation_id") or item.get("task_id") or item.get("id")
                if key:
                    rows[key] = _normalize(item)

    elif isinstance(data, dict):

        if "results" in data and isinstance(data["results"], list):
            for r in data["results"]:
                if isinstance(r, dict):
                    key = r.get("equation_id") or r.get("task_id") or r.get("id")
                    if key:
                        rows[key] = _normalize(r)

        elif "task_id" in data:
            rows[data["task_id"]] = _normalize(data)

        else:
            # fallback structured dict
            for k, v in data.items():
                if isinstance(v, dict) and (
                    k in _DEFI_IDS or any(rk in v for rk in _RESULT_KEYS)
                ):
                    rows[k] = _normalize(v)

    return rows


# =========================
# Main pipeline
# =========================

def main():

    merged = {}

    print("\n==============================")
    print("HypatiaX Merge Pipeline")
    print("==============================")
    print(f"EXP: {EXP}")
    print(f"RESULT_DIR: {RESULT_DIR}")
    print(f"OUT_DIR: {OUT_DIR}")

    # =========================
    # MODE 1: Prefer merged file
    # =========================
    if MERGED_FILE.exists():

        print("\n[MODE] Using merged JSON (fast path)")

        data = load_merged_file(MERGED_FILE)
        merged = extract_rows(data)

    else:

        # =========================
        # MODE 2: legacy shard mode
        # =========================
        print("\n[MODE] Using shard scan (legacy mode)")

        files = load_shards(RESULT_DIR)

        if not files:
            raise FileNotFoundError(
                f"No JSON files found in {RESULT_DIR}"
            )

        for path in files:
            try:
                with open(path) as f:
                    data = json.load(f)

                rows = extract_rows(data)

                if rows:
                    print(f"[OK] {path.name}: {len(rows)} rows")

                merged.update(rows)

            except Exception as e:
                print(f"[SKIP] {path}: {e}")

    # =========================
    # Summary
    # =========================

    print("\n==============================")
    print(f"Total merged: {len(merged)}")
    print("==============================")

    if ALL_PENDING:
        missing = sorted(set(ALL_PENDING) - set(merged))
        coverage = len(merged) / len(ALL_PENDING) * 100

        print(f"Coverage: {len(merged)}/{len(ALL_PENDING)} ({coverage:.1f}%)")

        if missing:
            print(f"Missing: {missing}")

    # =========================
    # Save merged output
    # =========================

    merged_path = OUT_DIR / f"{EXP}_merged.json"
    with open(merged_path, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"[WRITE] {merged_path}")

    # =========================
    # Stats computation
    # =========================

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

    successes = [x for x in hyp if x > 0.99]

    stats = {
        "experiment": EXP,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_total": len(ALL_PENDING) if ALL_PENDING else len(merged),
        "n_merged": len(merged),
        "n_successes": len(successes),
        "success_rate": (len(successes) / len(merged)) if merged else None,
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

    stats_path = OUT_DIR / f"{EXP}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[WRITE] {stats_path}")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()

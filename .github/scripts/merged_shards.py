import os
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from scipy import stats as scipy_stats


# =========================================================
# CONFIG
# =========================================================

EXP = os.environ.get("EXP", "exp1")
RESULT_SUBDIR = os.environ["RESULT_SUBDIR"]
ALL_PENDING = json.loads(os.environ.get("ALL_PENDING", "[]"))

BASE_DIR = Path("hypatiax/data/results")
RESULT_DIR = BASE_DIR / RESULT_SUBDIR
RESULT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_MERGED_FILE = RESULT_DIR / f"{EXP}_merged.json"
OUTPUT_MERGED_FILE = RESULT_DIR / f"{EXP}_merged_clean.json"
OUTPUT_STATS_FILE = RESULT_DIR / f"{EXP}_stats.json"


# =========================================================
# DOMAIN KEYS
# =========================================================

_DEFI_IDS = {
    "amm", "risk_var", "liquidity", "expected_shortfall",
    "liquidation", "risk", "lending", "staking",
    "trading", "derivatives",
}


# =========================================================
# NORMALIZATION
# =========================================================

def normalize(item: dict) -> dict:
    """Standardize HypatiaX / NN schema formats."""
    if not isinstance(item, dict):
        return item

    item = dict(item)

    inner = item.get("results")

    if isinstance(inner, dict):
        inner = dict(inner)

        # unify naming
        if "pure_llm" in inner and "hypatia" not in inner:
            inner["hypatia"] = inner.pop("pure_llm")

        if "neural_network" in inner and "nn" not in inner:
            inner["nn"] = inner.pop("neural_network")

        item.update(inner)
        item["results"] = inner

    return item


# =========================================================
# EXTRACTION ENGINE (FIXED CORE)
# =========================================================

def extract_rows(data: dict | list) -> dict:
    """
    Supports 3 formats:
    1. {"amm": {...}, "risk_var": {...}}
    2. {"results": [...]}
    3. task_id / equation_id based systems
    """

    rows = {}

    # -------------------------
    # CASE 1: LIST FORMAT
    # -------------------------
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue

            key = (
                item.get("equation_id")
                or item.get("task_id")
                or item.get("id")
            )

            if key:
                rows[key] = normalize(item)

    # -------------------------
    # CASE 2: DICT FORMAT
    # -------------------------
    elif isinstance(data, dict):

        # CASE 2A: nested results list
        if isinstance(data.get("results"), list):
            for r in data["results"]:
                if isinstance(r, dict):
                    key = r.get("equation_id") or r.get("task_id") or r.get("id")
                    if key:
                        rows[key] = normalize(r)

        # CASE 2B: DIRECT DEFI FORMAT (YOUR CASE)
        elif any(k in _DEFI_IDS for k in data.keys()):
            for k in _DEFI_IDS:
                if k in data and isinstance(data[k], dict):
                    rows[k] = normalize(data[k])

        # CASE 2C: SINGLE TASK OBJECT
        elif "task_id" in data:
            rows[data["task_id"]] = normalize(data)

        # CASE 2D: GENERIC FALLBACK
        else:
            for k, v in data.items():
                if isinstance(v, dict):
                    rows[k] = normalize(v)

    return rows


# =========================================================
# LOADING
# =========================================================

def load_input():
    """
    Priority:
    1. merged file (preferred)
    2. raw JSON scan fallback
    """

    if INPUT_MERGED_FILE.exists():
        print(f"[INFO] Loading merged file: {INPUT_MERGED_FILE}")
        with open(INPUT_MERGED_FILE) as f:
            return json.load(f), "merged"

    print(f"[WARN] No merged file found, scanning directory fallback")

    files = list(RESULT_DIR.rglob("*.json"))

    if not files:
        raise FileNotFoundError(f"No JSON found in {RESULT_DIR}")

    print(f"[INFO] Found {len(files)} JSON files")
    combined = []

    for f in files:
        try:
            with open(f) as fh:
                combined.append(json.load(fh))
        except Exception as e:
            print(f"[SKIP] {f}: {e}")

    return combined, "shards"


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():

    print("\n==============================")
    print(" HypatiaX Clean Merge Engine ")
    print("==============================")

    print(f"EXP: {EXP}")
    print(f"RESULT_DIR: {RESULT_DIR}")

    data, mode = load_input()

    print(f"[MODE] {mode}")

    merged = {}

    # normalize input into rows
    extracted = extract_rows(data)

    merged.update(extracted)

    # =========================
    # VALIDATION (IMPORTANT)
    # =========================

    if not merged:
        print("\n==============================")
        print("❌ ERROR: EMPTY MERGE")
        print("==============================")
        print("Likely causes:")
        print(" - Wrong schema (amm/risk_var not detected)")
        print(" - Wrong input file")
        print(" - Unexpected JSON structure")
        raise RuntimeError("Merge produced 0 rows")

    # =========================
    # SAVE CLEAN OUTPUT
    # =========================

    with open(OUTPUT_MERGED_FILE, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"\n[WRITE] {OUTPUT_MERGED_FILE}")

    # =========================
    # COVERAGE
    # =========================

    if ALL_PENDING:
        missing = sorted(set(ALL_PENDING) - set(merged))
        coverage = len(merged) / len(ALL_PENDING)

        print("\n==============================")
        print(f"Coverage: {len(merged)}/{len(ALL_PENDING)} ({coverage:.1%})")

        if missing:
            print(f"Missing: {missing}")

    # =========================
    # STATS
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
        "mode": mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_total": len(ALL_PENDING) if ALL_PENDING else len(merged),
        "n_merged": len(merged),
        "n_successes": len(successes),
        "success_rate": (len(successes) / len(merged)) if merged else None,
        "hyp_mean": float(np.mean(hyp)) if hyp else None,
        "nn_mean": float(np.mean(nn)) if nn else None,
    }

    if len(hyp) >= 5 and len(nn) >= 5:
        u, p = scipy_stats.mannwhitneyu(hyp, nn, alternative="greater")
        stats.update({
            "mw_U": float(u),
            "mw_p": float(p),
            "mw_significant": bool(p < 0.05),
        })

    with open(OUTPUT_STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[WRITE] {OUTPUT_STATS_FILE}")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()


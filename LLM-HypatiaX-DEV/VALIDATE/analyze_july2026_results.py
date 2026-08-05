#!/usr/bin/env python3
"""
analyze_july2026_results.py
============================
Parses the actual July 15, 2026 result files (real schema, confirmed by
inspection) and produces:
  1. Six-method noiseless pass/fail counts at both the paper's original
     threshold (0.9999) and HLLMNN's strict threshold (0.999999).
  2. A merged noise-level summary across all 5 noise_sweep shard files.
  3. Confirmation of which specific equations flipped from fail->pass for
     HLLMNN relative to the March 2026 supplementary (Lorentz force, Photon
     energy, Zeeman energy).

Input files (as uploaded):
  protocol_core_noiseless_20260715_230428.json   - six-method noiseless
  noise_sweep_20260715_215520_nshards04.json     - noise=0.005
  noise_sweep_20260715_215527_nshards05.json     - noise=0.01
  noise_sweep_20260715_215529_nshards02.json     - noise=0.0005
  noise_sweep_20260715_215537_nshards01.json     - noise=0.0
  noise_sweep_20260715_215619_nshards03.json     - noise=0.001
  protocol_core_noisy_20260715_23*.json          - noisy runs, different
                                                    thresholds per level,
                                                    explicitly marked "NOT
                                                    directly comparable to
                                                    published noiseless
                                                    figures" in their own
                                                    protocol metadata.

IMPORTANT CAVEAT (do not silently ignore): the noise_sweep shards use
noise levels {0, 0.05%, 0.1%, 0.5%, 1%} of relative scale, NOT the paper's
{0, 0.5%, 1%, 5%, 10%}. Only the 0% (noiseless) point is directly comparable
to the March 2026 supplementary's noise sweep (Table 8/9/10). This script
does not attempt to paper over that mismatch.
"""
import json
import glob
from pathlib import Path

UPLOAD_DIR = Path("/mnt/user-data/uploads")

METHOD_NAMES = [
    "PureLLM Baseline (core)",
    "ImprovedNN (core)",
    "EnhancedHybridSystemDeFi (core)",
    "HybridSystemLLMNN all-domains (core)",
    "SymbolicEngineWithLLM (tools)",
    "HybridDiscoverySystem v50_2 (tools)",
]
SHORT_NAME = {
    "PureLLM Baseline (core)": "PureLLM",
    "ImprovedNN (core)": "ImpNN",
    "EnhancedHybridSystemDeFi (core)": "EHSDeFi",
    "HybridSystemLLMNN all-domains (core)": "HLLMNN",
    "SymbolicEngineWithLLM (tools)": "SymLLM",
    "HybridDiscoverySystem v50_2 (tools)": "HDSv50_2",
}

PREVIOUSLY_FAILING_HLLMNN = [
    "Lorentz force", "Photon energy", "Zeeman energy",
]


def analyze_noiseless(path):
    d = json.load(open(path))
    out = {"source_file": path.name, "timestamp": d["timestamp"],
           "script": d["script"], "n_tests": d["total_tests"], "methods": {}}
    for m in METHOD_NAMES:
        r2s = []
        for t in d["tests"]:
            r = t["results"].get(m)
            if r and r.get("r2") is not None:
                r2s.append((t["description"], r["r2"]))
        n = len(r2s)
        vals = [v for _, v in r2s]
        row = {"n": n, "mean_r2": sum(vals) / n if n else None}
        for thr in (0.9999, 0.999999):
            npass = sum(1 for v in vals if v >= thr)
            row[f"pass_at_{thr}"] = f"{npass}/{n}"
            row[f"pct_at_{thr}"] = round(100 * npass / n, 1) if n else None
        row["fails_at_0.999999"] = [desc for desc, v in r2s if v < 0.999999]
        out["methods"][SHORT_NAME[m]] = row
    return out


def analyze_noise_sweep_shards(paths):
    merged = {}
    for p in paths:
        d = json.load(open(p))
        noise_key = list(d["per_noise"].keys())[0]
        noise_val = float(noise_key)
        ms = d["per_noise"][noise_key]["method_summary"]
        merged[noise_val] = {
            SHORT_NAME[m]: {
                "recovery_rate": round(s["recovery_rate"] * 100, 1),
                "mean_r2": s["mean_r2"],
                "n_catastrophic": s["n_catastrophic"],
            }
            for m, s in ms.items()
        }
    return dict(sorted(merged.items()))


def main():
    noiseless_path = UPLOAD_DIR / "protocol_core_noiseless_20260715_230428.json"
    shard_paths = sorted(UPLOAD_DIR.glob("noise_sweep_20260715_*_nshards*.json"))

    print("=" * 70)
    print("SIX-METHOD NOISELESS RERUN:", noiseless_path.name)
    print("=" * 70)
    result = analyze_noiseless(noiseless_path)
    print(f"script version: {result['script']}")
    print(f"timestamp: {result['timestamp']}\n")
    print(f"{'Method':<10} {'@0.9999':<10} {'@0.999999':<10} {'mean R2':<12}")
    for short, row in result["methods"].items():
        print(f"{short:<10} {row['pass_at_0.9999']:<10} "
              f"{row['pass_at_0.999999']:<10} {row['mean_r2']:.6f}")

    print("\nHLLMNN fails at strict threshold (0.999999):",
          result["methods"]["HLLMNN"]["fails_at_0.999999"] or "NONE (30/30)")

    print("\nCross-check vs March 2026 paper's 3 named HLLMNN failures:")
    hllmnn_fails = result["methods"]["HLLMNN"]["fails_at_0.999999"]
    for name in PREVIOUSLY_FAILING_HLLMNN:
        still_failing = any(name.lower() in f.lower() for f in hllmnn_fails)
        status = "STILL FAILING" if still_failing else "NOW PASSES"
        print(f"  {name}: {status}")

    print("\n" + "=" * 70)
    print("NOISE SWEEP (5 shards merged) — HLLMNN recovery rate by noise level")
    print("=" * 70)
    print("CAVEAT: noise levels here are {0, 0.05%, 0.1%, 0.5%, 1%}, NOT the")
    print("paper's {0, 0.5%, 1%, 5%, 10%}. Only noise=0 is directly comparable")
    print("to the March 2026 supplementary's Table 8/9/10.\n")
    sweep = analyze_noise_sweep_shards(shard_paths)
    for noise_val, methods in sweep.items():
        print(f"noise={noise_val*100:.2f}%:")
        for short, s in methods.items():
            print(f"  {short:<10} recovery={s['recovery_rate']:>5.1f}%  "
                  f"mean_r2={s['mean_r2']:.6f}  catastrophic={s['n_catastrophic']}")

    out = {"noiseless": result, "noise_sweep": sweep}
    out_path = Path("/mnt/user-data/outputs/july2026_results_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

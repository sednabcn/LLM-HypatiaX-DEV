#!/usr/bin/env python3
"""
verify_abstract_numbers.py
===========================

An INDEPENDENT, from-scratch cross-check of the paper's abstract headline
numbers (near-perfect success rate + the two baseline deltas), and of
tab:difficulty / tab:hybrid-bug-breakdown's underlying counts.

Why this exists as a *separate* script instead of just calling
generate_tables.py's gen_abstract_macros() / gen_defi_tiers() /
gen_hybrid_bug_breakdown():

    Those three functions already compute these numbers correctly and are
    already wired into the postprocess pipeline (see main()'s dispatch in
    generate_tables.py and the "Generate tables" step in
    ci_postprocess.yml) and covered by test_generate_tables.py. Re-running
    THEM only proves internal self-consistency -- if there's a shared bug
    in that codebase's decision-attribution logic, this would reproduce it
    and still "match."

    This script re-implements the same arithmetic independently, reading
    only the raw JSON, so it can catch a bug that's shared across
    generate_tables.py's own functions. It is still not a substitute for a
    human spot-checking a handful of the 74 cases by hand -- see the
    --sample flag below, which is meant to make that spot-check easy.

WHAT IT DOES NOT DO:
    - It does not compare its output against 90.5 / 62.2 / 5.4 / 28.3 / 85.1
      (the abstract's old, untraceable figures) or against 59.5 (the
      File-A figure "already confirmed" by a prior audit pass). It prints
      raw counts and percentages only. You decide what they mean.
    - It does not write any .tex file and does not touch the paper.
    - It does not emit a pass/fail verdict.

USAGE
    python3 verify_abstract_numbers.py path/to/hypatiax_defi_benchmark_v4_results_seed42.json
    python3 verify_abstract_numbers.py FILE.json --sample 5      # print 5 random cases in full, for hand-checking
    python3 verify_abstract_numbers.py FILE.json --json          # machine-readable output (for CI)

EXPECTED SCHEMA ("Shape-3"): a flat JSON list of ~74 dicts, each shaped like
    {
      "difficulty": "Easy" | "Medium" | "Hard",   # (or "tier")
      "results": {
        "pure_llm":       {"test_r2": <float>},
        "neural_network":  {"test_r2": <float>},
        "hybrid":          {"test_r2": <float>, "decision": "llm"|"nn"|"nn_fallback"|"ensemble", ...}
      }
    }
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PASS_THRESHOLD = 0.99

# Routing decision -> which independently-computed sub-method result that
# decision actually names. "ensemble" has no separate baseline arm, so a
# case routed to "ensemble" is left on hybrid's own reported value (with a
# warning emitted -- see _check_case below), same as generate_tables.py.
DECISION_TO_BASELINE = {
    "llm": "pure_llm",
    "nn": "neural_network",
    "nn_fallback": "neural_network",
    "v4_llm": "pure_llm",
    "v4_nn": "neural_network",
    # v4_residual_nn is deliberately NOT mapped: it is unclear whether this
    # names the same thing as "neural_network" or a distinct residual-fitting
    # arm with no corresponding baseline field in the schema. Mapping it
    # incorrectly would either wrongly clear a fabricated pass or wrongly
    # flag a real one, so cases routed here fall through to the "no
    # independent baseline" branch below and stay flagged for manual review
    # until someone who knows the v4 routing logic confirms what it means.
}
KNOWN_DECISIONS = {"llm", "nn", "nn_fallback", "ensemble", "v4_llm", "v4_nn", "v4_residual_nn"}
KNOWN_TIERS = ("Easy", "Medium", "Hard")


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v  # excludes NaN/bool


def load_records(path: Path) -> list[dict]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise SystemExit(
            f"error: {path} is not a flat list at the top level "
            f"(got {type(raw).__name__}) -- this doesn't look like the Shape-3 schema."
        )
    if not raw:
        raise SystemExit(f"error: {path} parsed to an empty list.")
    return raw


def check_and_score(records: list[dict]) -> tuple[dict, list[str], list[dict]]:
    """
    Independently walks the raw records once. Returns:
      - counts: dict of all the raw counts this script computed
      - warnings: schema/data anomalies worth a human's attention
      - scored: per-case scoring detail, for --sample / manual spot-checks
    """
    warnings: list[str] = []
    scored: list[dict] = []

    tier_counts = {t: {"n": 0, "llm_pass": 0, "nn_pass": 0, "reported_pass": 0, "fabricated": 0, "corrected_pass": 0}
                    for t in KNOWN_TIERS}
    n_total = n_llm_pass = n_nn_pass = n_reported_pass = n_fabricated = n_corrected_pass = 0
    seen_case_ids = set()

    for i, rec in enumerate(records):
        if not isinstance(rec, dict) or "results" not in rec:
            warnings.append(f"record[{i}]: missing 'results' key -- skipped entirely, not counted in n_total.")
            continue

        case_id = rec.get("case_id") or rec.get("id") or rec.get("task_id")
        if case_id is not None:
            if case_id in seen_case_ids:
                warnings.append(f"record[{i}]: duplicate case id {case_id!r} -- check for double-counting.")
            seen_case_ids.add(case_id)

        tier_raw = rec.get("difficulty") or rec.get("tier")
        tier = tier_raw.strip().capitalize() if isinstance(tier_raw, str) else tier_raw
        if tier != tier_raw:
            warnings.append(f"record[{i}] (case_id={case_id!r}): tier {tier_raw!r} normalised to {tier!r}.")
        if tier not in KNOWN_TIERS:
            warnings.append(f"record[{i}] (case_id={case_id!r}): tier {tier!r} not in {KNOWN_TIERS} -- excluded from the tier breakdown (but counted in n_total below).")

        cr = rec.get("results", {}) or {}

        llm_r2 = (cr.get("pure_llm", {}) or {}).get("test_r2")
        nn_r2 = (cr.get("neural_network", {}) or {}).get("test_r2")
        hybrid = cr.get("hybrid", {}) or {}
        reported_r2 = hybrid.get("test_r2")
        decision = hybrid.get("decision", "")

        if not _is_num(llm_r2):
            warnings.append(f"record[{i}] (case_id={case_id!r}): pure_llm.test_r2 missing/non-numeric ({llm_r2!r}).")
        if not _is_num(nn_r2):
            warnings.append(f"record[{i}] (case_id={case_id!r}): neural_network.test_r2 missing/non-numeric ({nn_r2!r}).")
        if not _is_num(reported_r2):
            warnings.append(f"record[{i}] (case_id={case_id!r}): hybrid.test_r2 missing/non-numeric ({reported_r2!r}).")
        if decision and decision not in KNOWN_DECISIONS:
            warnings.append(f"record[{i}] (case_id={case_id!r}): unrecognised hybrid.decision {decision!r}.")

        llm_pass = _is_num(llm_r2) and llm_r2 > PASS_THRESHOLD
        nn_pass = _is_num(nn_r2) and nn_r2 > PASS_THRESHOLD
        reported_pass = _is_num(reported_r2) and reported_r2 > PASS_THRESHOLD

        fabricated = False
        corrected_pass = False
        baseline_r2 = None
        if reported_pass:
            baseline_key = DECISION_TO_BASELINE.get(decision)
            if baseline_key:
                baseline_r2 = (cr.get(baseline_key, {}) or {}).get("test_r2")
                corrected_pass = _is_num(baseline_r2) and baseline_r2 > PASS_THRESHOLD
            else:
                # "ensemble" or unrecognised/missing decision: no independent
                # baseline arm exists to cross-check against, so we cannot
                # independently confirm this one -- flag it rather than
                # silently trusting hybrid's self-report.
                warnings.append(
                    f"record[{i}] (case_id={case_id!r}): hybrid reports a pass (r2={reported_r2}) "
                    f"via decision={decision!r}, which has no independent baseline to cross-check -- "
                    f"counted as corrected_pass=True (same convention as generate_tables.py) but this "
                    f"is exactly the kind of case that deserves a manual look."
                )
                corrected_pass = True
            fabricated = not corrected_pass

        n_total += 1
        n_llm_pass += int(llm_pass)
        n_nn_pass += int(nn_pass)
        n_reported_pass += int(reported_pass)
        n_fabricated += int(fabricated)
        n_corrected_pass += int(corrected_pass)

        if tier in KNOWN_TIERS:
            tc = tier_counts[tier]
            tc["n"] += 1
            tc["llm_pass"] += int(llm_pass)
            tc["nn_pass"] += int(nn_pass)
            tc["reported_pass"] += int(reported_pass)
            tc["fabricated"] += int(fabricated)
            tc["corrected_pass"] += int(corrected_pass)

        scored.append({
            "index": i, "case_id": case_id, "tier": tier,
            "pure_llm_r2": llm_r2, "neural_network_r2": nn_r2,
            "hybrid_reported_r2": reported_r2, "decision": decision,
            "baseline_r2_used_for_check": baseline_r2,
            "llm_pass": llm_pass, "nn_pass": nn_pass,
            "reported_pass": reported_pass, "fabricated": fabricated,
            "corrected_pass": corrected_pass,
        })

    tier_sum_n = sum(tc["n"] for tc in tier_counts.values())
    if tier_sum_n != n_total:
        warnings.append(
            f"tier breakdown covers {tier_sum_n} of {n_total} total records -- "
            f"{n_total - tier_sum_n} record(s) had an unrecognised/missing tier and are "
            f"included in the Overall row but absent from Easy/Medium/Hard."
        )

    counts = {
        "n_total": n_total,
        "n_llm_pass": n_llm_pass,
        "n_nn_pass": n_nn_pass,
        "n_reported_pass": n_reported_pass,
        "n_fabricated": n_fabricated,
        "n_corrected_pass": n_corrected_pass,
        "tiers": tier_counts,
    }
    return counts, warnings, scored


def rate(n: int, d: int) -> float:
    return (100.0 * n / d) if d else float("nan")


def fmt_pct(v: float) -> str:
    return f"{v:.1f}%" if v == v else "---"


def render_report(counts: dict, src: Path) -> str:
    n = counts["n_total"]
    hyp_rate = rate(counts["n_corrected_pass"], n)
    llm_rate = rate(counts["n_llm_pass"], n)
    nn_rate = rate(counts["n_nn_pass"], n)
    reported_rate = rate(counts["n_reported_pass"], n)
    delta_llm = hyp_rate - llm_rate
    delta_nn = hyp_rate - nn_rate

    lines = []
    lines.append(f"Source file: {src}")
    lines.append(f"n_total (fixed denominator): {n}")
    lines.append("")
    lines.append("── Per-tier breakdown (independent re-derivation) ──────────────")
    lines.append(f"{'Tier':8} {'n':>4} {'LLM pass':>9} {'NN pass':>8} {'Reported':>9} {'Fabricated':>11} {'Corrected':>10}")
    for t in KNOWN_TIERS:
        tc = counts["tiers"][t]
        lines.append(f"{t:8} {tc['n']:>4} {tc['llm_pass']:>9} {tc['nn_pass']:>8} "
                      f"{tc['reported_pass']:>9} {tc['fabricated']:>11} {tc['corrected_pass']:>10}")
    lines.append(f"{'Overall':8} {n:>4} {counts['n_llm_pass']:>9} {counts['n_nn_pass']:>8} "
                  f"{counts['n_reported_pass']:>9} {counts['n_fabricated']:>11} {counts['n_corrected_pass']:>10}")
    lines.append("")
    lines.append("── Rates over fixed denominator (this is what the abstract's macros use) ──")
    lines.append(f"  Corrected HypatiaX >0.99 rate : {counts['n_corrected_pass']}/{n} = {fmt_pct(hyp_rate)}")
    lines.append(f"  Raw (uncorrected) hybrid rate : {counts['n_reported_pass']}/{n} = {fmt_pct(reported_rate)}  (includes fabricated passes -- NOT what the abstract should cite)")
    lines.append(f"  Pure LLM baseline rate        : {counts['n_llm_pass']}/{n} = {fmt_pct(llm_rate)}")
    lines.append(f"  Neural network baseline rate  : {counts['n_nn_pass']}/{n} = {fmt_pct(nn_rate)}")
    lines.append("")
    lines.append("── Deltas ────────────────────────────────────────────────────")
    if delta_llm > 0:
        lines.append(f"  vs. LLM baseline : +{delta_llm:.1f}pp")
    else:
        lines.append(f"  vs. LLM baseline : WITHHELD -- corrected rate ({fmt_pct(hyp_rate)}) is not an improvement over the LLM baseline ({fmt_pct(llm_rate)}); delta = {delta_llm:.1f}pp")
    if delta_nn > 0:
        lines.append(f"  vs. NN baseline  : +{delta_nn:.1f}pp")
    else:
        lines.append(f"  vs. NN baseline  : WITHHELD -- corrected rate ({fmt_pct(hyp_rate)}) is not an improvement over the NN baseline ({fmt_pct(nn_rate)}); delta = {delta_nn:.1f}pp")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json_path", type=Path, help="path to the seed-42 Shape-3 results JSON (File A)")
    ap.add_argument("--sample", type=int, default=0, metavar="K",
                     help="also print K randomly-chosen cases in full (raw r2 values, decision, "
                          "pass/fail at every stage) so a human can hand-check them against the "
                          "original model outputs")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed for --sample (default: unseeded)")
    ap.add_argument("--json", action="store_true", dest="json_out",
                     help="emit machine-readable JSON instead of the text report (for CI wiring)")
    args = ap.parse_args()

    if not args.json_path.exists():
        raise SystemExit(f"error: {args.json_path} does not exist.")

    records = load_records(args.json_path)
    counts, warnings, scored = check_and_score(records)

    if not any(counts["tiers"][t]["n"] for t in KNOWN_TIERS) and counts["n_total"] == 0:
        raise SystemExit("error: no usable records found -- see warnings above (none, because nothing parsed at all).")

    if args.json_out:
        out = {"source": str(args.json_path), "counts": counts, "warnings": warnings}
        print(json.dumps(out, indent=2))
        return

    print(render_report(counts, args.json_path))

    if warnings:
        print("")
        print(f"── {len(warnings)} warning(s) -- review before trusting these numbers ──────")
        for w in warnings:
            print(f"  ⚠ {w}")
    else:
        print("")
        print("(no schema/data warnings)")

    if args.sample:
        rng = random.Random(args.seed)
        k = min(args.sample, len(scored))
        print("")
        print(f"── {k} sampled case(s) for manual spot-checking ───────────────")
        for case in rng.sample(scored, k):
            print(json.dumps(case, indent=2))

    print("")
    print("This script does not judge whether these numbers are 'correct' -- it only")
    print("re-derives them independently from the raw file. Compare the rates above")
    print("against whatever the abstract/table currently says, by hand.")


if __name__ == "__main__":
    main()

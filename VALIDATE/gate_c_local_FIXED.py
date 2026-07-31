import glob, json, os, pathlib, sys, re

OUT_BASE = pathlib.Path(os.environ.get("OUT_BASE", "hypatiax/data/results"))

LEGACY_DIR    = OUT_BASE / "comparison_results/feynman-tests/exp2"
CORRECTED_DIR = OUT_BASE / "comparison_results/feynman-tests/exp2_pca_4060"
BASELINE_FILE = OUT_BASE / "fixc3_baseline.json"

failures = []

# ── Check 1: legacy and corrected output paths are DIFFERENT ─────
assert LEGACY_DIR != CORRECTED_DIR, (
    "Gate C internal error: legacy and corrected dirs are identical"
)
print(f"  [OK]  Legacy path   : {LEGACY_DIR.relative_to(OUT_BASE)}")
print(f"  [OK]  Corrected path: {CORRECTED_DIR.relative_to(OUT_BASE)}")

# ── Check 2: if legacy results exist, capture baseline ────────────
# FIX-GLOB-PROTOCOL-FILTER (verification report Revision 21, item 4):
# filename-based filtering was considered and rejected — both the
# Feynman-only protocol and an all_domains-protocol run share the
# identical protocol_core_noiseless_{ts}.json naming convention, so
# no prefix/suffix reliably distinguishes them. Content-based
# filtering is used instead, keyed on each file's own tests[].domain
# values against the eleven canonical Feynman domains (the same
# field this report's compute_solve_rate.py already uses for
# domain-to-timestamp mapping). This rejects any file whose domains
# fall outside that set — including an all_domains file sharing the
# same filename convention — so a future coincidental overlap in
# LEGACY_DIR drops the foreign file instead of silently pooling it.
FEYNMAN_DOMAINS = {
    "feynman_biology", "feynman_chemistry", "feynman_electrochemistry",
    "feynman_electromagnetism", "feynman_electrostatics", "feynman_magnetism",
    "feynman_mechanics", "feynman_optics", "feynman_probability",
    "feynman_quantum", "feynman_thermodynamics",
}

def _is_feynman_protocol(fp: pathlib.Path) -> bool:
    # FIX-PROTOCOL-FILTER-LISTROOT: some legacy files (e.g.
    # benchmark_results.json) are JSON arrays at the top level,
    # not {"tests": [...]} objects. json.loads() succeeds on
    # those, so the earlier bare except only caught malformed
    # JSON, not this well-formed-but-wrong-shape case, and
    # data.get(...) then raised AttributeError on the list.
    try:
        data = json.loads(fp.read_text())
        if not isinstance(data, dict):
            return False
        domains = {t.get("domain") for t in data.get("tests", []) if isinstance(t, dict)}
        return bool(domains) and domains <= FEYNMAN_DOMAINS
    except Exception:
        return False

legacy_jsons = [
    f for f in (sorted(LEGACY_DIR.glob("*.json")) if LEGACY_DIR.exists() else [])
    if "checkpoint" not in f.name
    and "disclosure" not in f.name
    and not f.name.startswith("_")   # exclude meta files (_analysis, _merged, _stats …)
    and _is_feynman_protocol(f)       # new: content-based protocol filter
]

if legacy_jsons:
    # Extract 9/30 solve rate from legacy results
    n_total = n_pass = 0
    # FIX-GATE-C-THRESHOLD: was 0.9999 — drifted from the pipeline's own
    # r²≥0.999999 noiseless threshold (FEYNMAN_NOISELESS_THRESHOLD in
    # run_all.sh / ci_runner_repro.yml). At 0.9999 this locked an inflated
    # legacy baseline of 81/90 (90.0%) instead of the correct 74/90 (82.2%).
    THRESHOLD = 0.999999
    PREFERRED = {"hypatiax","hybridv50","hybrid50","hybridsymbolic",
                 "hybriddefi","hypatia","hybrid","ours","proposed"}

    def _r2_from_row(row):
        for key in ("r2","r2_test","r2_train","best_r2","R2"):
            v = row.get(key)
            if v is not None:
                try:
                    f = float(v)
                    if f <= 1.01:
                        return f
                except (TypeError, ValueError):
                    pass
        return None

    def _iter_rows(data):
        if isinstance(data, dict):
            for key in ("results","equation_results","data","rows"):
                v = data.get(key)
                if v is not None:
                    yield from _iter_rows(v)
                    return   # FIX: return only when a container key was found
            # FIX-TESTS-SCHEMA: run_comparative_suite_benchmark_v2.py's
            # payload nests per-equation records under "tests", and each
            # record's own "results" field is a dict keyed by method
            # display name (not a list) whose values are already complete
            # rows (each has its own "method"/"r2" via MethodResult.to_dict()).
            # Neither shape was previously recognized here, so every real
            # Feynman result file collapsed to a single unscoreable row
            # (the whole payload) and this gate silently computed
            # n_total=0 on genuine, non-empty data. Verified against the
            # actual _save() schema in run_comparative_suite_benchmark_v2.py.
            if isinstance(data.get("tests"), list):
                for test in data["tests"]:
                    if isinstance(test, dict) and isinstance(test.get("results"), dict):
                        yield from test["results"].values()
                return
            yield data
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    yield item

    for fp in legacy_jsons:
        if "checkpoint" in fp.name or "disclosure" in fp.name:
            continue
        try:
            data = json.loads(fp.read_text())
        except Exception:
            continue
        for row in _iter_rows(data):
            raw = row.get("method") or row.get("model") or ""
            method = str(raw).lower().replace("-","").replace("_","").replace(" ","")
            if method and not any(p in method for p in PREFERRED):
                continue
            r2 = _r2_from_row(row)
            if r2 is None:
                continue
            n_total += 1
            if r2 >= THRESHOLD:
                n_pass += 1

    baseline = {
        "fixc3_gate":  "C",
        "description": "Pre-fix baseline — Feynman result (random 80/20 split)",
        "split_protocol": "random_80_20",
        "n_pass":  n_pass,
        "n_total": n_total,
        "solve_rate": (n_pass / n_total) if n_total > 0 else None,
        "paper_claim": "9/30 = 0.300",
        # FIX-MANIFEST-TRUNCATION (mirrors run_all.sh:1612-1621): was
        # legacy_jsons[:5]. Files are named protocol_core_{mode}_{ts}
        # {shard}.json and sorted() puts them in ascending timestamp
        # order, i.e. domain-completion order. With 11 Feynman domains,
        # [:5] kept only the earliest 5 and silently dropped every later
        # one from the manifest -- deterministically omitting whichever
        # domain ran last, even though that domain's rows were still
        # counted correctly in n_pass/n_total above. List every source
        # file so the manifest matches what was actually counted.
        "source_files": [fp.name for fp in legacy_jsons],
    }
    assert len(baseline["source_files"]) == len(legacy_jsons), (
        "Gate C internal error: source_files manifest does not match "
        "legacy_jsons — baseline-lock logic has re-drifted from "
        "run_all.sh; see FIX-MANIFEST-TRUNCATION"
    )

    # Write baseline file (create once, refuse to overwrite)
    if BASELINE_FILE.exists():
        existing = json.loads(BASELINE_FILE.read_text())
        existing_rate = existing.get("solve_rate")
        if existing_rate is None:
            # FIX: a null solve_rate means the baseline was written from
            # a run where no legacy JSONs were present (n_total=0). Rather
            # than permanently failing and requiring an operator to
            # manually delete fixc3_baseline.json, self-heal: if legacy
            # JSONs are present *now* (n_total > 0), the baseline was
            # simply written too early — recompute and overwrite it with
            # the now-valid data. Only fail if legacy results are still
            # absent, since there's nothing to lock in that case.
            if n_total > 0:
                BASELINE_FILE.write_text(json.dumps(baseline, indent=2))
                print(
                    "  [OK]  Baseline was invalid (solve_rate=null, written "
                    "before exp2_feynman results existed) — recomputed and "
                    f"rewritten → {n_pass}/{n_total} (rate={baseline['solve_rate']:.3f})"
                )
                # FIX-LOG-SOURCE-FILES (verification report Revision 21, item 5):
                # neither this branch nor the "already locked" branch below
                # previously echoed source_files, so no live CI log could ever
                # confirm the manifest-truncation fix by direct inspection —
                # only by the absence of an AssertionError. Print it explicitly.
                print(
                    f"  [OK]  source_files ({len(baseline['source_files'])}): "
                    f"{baseline['source_files']}"
                )
            else:
                failures.append(
                    "fixc3_baseline.json exists but solve_rate is null, and no "
                    "legacy JSONs were found in this checkout either — "
                    "confirm comparison_results/feynman-tests/exp2/ contains "
                    "result JSONs before this gate can lock a valid baseline."
                )
        elif n_total > 0:
            new_rate = n_pass / n_total
            if abs(new_rate - existing_rate) > 0.05:
                failures.append(
                    f"Baseline solve_rate changed from {existing_rate:.3f} "
                    f"to {new_rate:.3f} — possible overwrite of 9/30 baseline. "
                    "Delete fixc3_baseline.json manually to reset."
                )
            else:
                print(
                    f"  [OK]  Baseline already locked: "
                    f"{existing.get('n_pass')}/{existing.get('n_total')} "
                    f"(rate={existing_rate:.3f})"
                )
                # FIX-LOG-SOURCE-FILES (verification report Revision 21, item 5):
                # print the *committed* file's manifest here, since this branch
                # doesn't rewrite BASELINE_FILE — this is what a live run in the
                # "already locked" state actually has on disk, distinct from the
                # fresh `baseline` dict computed above from this checkout's files.
                _existing_sf = existing.get("source_files", [])
                print(f"  [OK]  source_files ({len(_existing_sf)}): {_existing_sf}")
        else:
            # n_total=0 in current scan but existing rate is non-null:
            # legacy dir may be empty on this runner (results not checked out).
            # Accept the existing baseline but warn.
            print(f"  [WARN] Baseline file exists (rate={existing_rate:.3f}) "
                  "but no legacy JSONs found in this checkout — "
                  "rate could not be re-verified; accepting existing baseline")
    else:
        BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_FILE.write_text(json.dumps(baseline, indent=2))
        print(
            f"  [OK]  Baseline locked → {BASELINE_FILE.name}  "
            f"({n_pass}/{n_total} equations passed)"
        )
        # FIX-LOG-SOURCE-FILES (verification report Revision 21, item 5):
        # see rationale on the "rewritten" branch above.
        print(
            f"  [OK]  source_files ({len(baseline['source_files'])}): "
            f"{baseline['source_files']}"
        )
else:
    print(
        f"  [INFO] No legacy Feynman results in {LEGACY_DIR} — "
        "baseline lock skipped (run exp2_feynman first)"
    )

# ── Check 3: corrected run (if complete) is in a DIFFERENT dir ───
corrected_jsons = (
    sorted(CORRECTED_DIR.glob("*.json")) if CORRECTED_DIR.exists() else []
)
corrected_jsons = [
    f for f in corrected_jsons
    if "checkpoint" not in f.name
    and "disclosure" not in f.name
    and not f.name.startswith("_")   # exclude meta files (_analysis, _merged, _stats …)
]

if corrected_jsons:
    # Confirm no file in CORRECTED_DIR is also in LEGACY_DIR (name clash)
    legacy_names    = {f.name for f in legacy_jsons}
    corrected_names = {f.name for f in corrected_jsons}
    overlap = legacy_names & corrected_names
    # Safety net: _analysis.json and similar meta files are written to both
    # dirs by design and must never trigger a false collision failure.
    overlap = {
        name for name in overlap
        if not name.startswith("_")
        and "checkpoint" not in name
        and "disclosure" not in name
    }
    if overlap:
        failures.append(
            f"Filename collision between legacy and corrected dirs: {overlap} — "
            "corrected results may have silently overwritten the 9/30 baseline"
        )
    else:
        print(
            f"  [OK]  Corrected dir has {len(corrected_jsons)} result file(s), "
            "no filename collision with legacy dir"
        )

    # Confirm baseline file still exists alongside the corrected results
    if not BASELINE_FILE.exists():
        failures.append(
            "fixc3_baseline.json not found but corrected results exist — "
            "the pre-fix 9/30 baseline was not locked before rerunning"
        )
    else:
        print(f"  [OK]  fixc3_baseline.json present alongside corrected results")
else:
    print(
        f"  [INFO] No corrected results in {CORRECTED_DIR} yet — "
        "overwrite-prevention check skipped"
    )

# ── Summary ────────────────────────────────────────────────────────
if failures:
    print(f"\n  Gate C FAILED — {len(failures)} issue(s):")
    for f in failures:
        print(f"    ✗  {f}")
    sys.exit(1)
else:
    print("\n  ✅  Gate C PASSED — baseline_lock_test")

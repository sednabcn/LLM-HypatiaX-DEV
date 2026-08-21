# Phase 3 Investigation Report

**Scope:** the three open items from `Phase03.txt` — item 2 (27/30 M4 provenance), item 10a (900s vs 1100s), item 10b (300s timeout enforcement).

---

## Item 2 — 27/30 M4 figure: RESOLVED

**Source data:** 10 shard files `protocol_core_noiseless_20260805_*.json`, 3 tests/shard, 30 unique tests total, spanning the 10-domain set (`biology, chemistry, economics, electromagnetism, fluid_dynamics, mathematics, mechanics, optics, quantum, thermodynamics`) — this is the `all_domains` protocol used by `exp2` / `exp1_five` in `run_all.sh`, six methods per test:

| # | Method | @0.999999 | @0.9999 | @0.999 |
|---|---|---|---|---|
| M1 | PureLLM Baseline (core) | 26/30 | 26/30 | 27/30 |
| M2 | ImprovedNN (core) | 0/30 | 19/30 | 24/30 |
| M3 | EnhancedHybridSystemDeFi (core) | 25/30 | 29/30 | 30/30 |
| M4 | HybridSystemLLMNN all-domains (core) | 29/30 | 30/30 | 30/30 |
| M5 | SymbolicEngineWithLLM (tools) | 24/30 | **27/30** | 27/30 |
| M6 | HybridDiscoverySystem v50_2 (tools) | 20/30 | 22/30 | 26/30 |

No method hits 27/30 under `METHOD_REGISTRY` numbering at any threshold — **except** `SymbolicEngineWithLLM (tools)` (registry index 5), which lands exactly on **27/30 at the 4‑decimal threshold (R² ≥ 0.9999)**, the same threshold basis the noiseless protocol note in the file cites as "directly comparable to published SR literature" (NeSymReS/AI Feynman/TPSR/DSR).

**Why that's "M4" in the abstract, not M5:** `exp1_five_system.py` / `provenance_map_exp1_five.json` (confirmed earlier this session) explicitly **excludes registry index 3** (`HybridDeFiMethod`, DeFi-scoped, not one of the five paper system rows) and renumbers the remaining five methods (`1,2,4,5,6`) sequentially for the Five-System table:

| Registry idx | Method | Five-System table position |
|---|---|---|
| 1 | PureLLM Baseline | M1 |
| 2 | ImprovedNN | M2 |
| 4 | HybridSystemLLMNN all-domains | M3 |
| **5** | **SymbolicEngineWithLLM** | **M4** |
| 6 | HybridDiscoverySystem v50_2 | M5 |

Registry index 5 → table position **M4**. So the abstract's "27/30, M4" is `SymbolicEngineWithLLM (tools)` scored at R² ≥ 0.9999, reported under the Five-System table's own M1–M5 numbering rather than the full six-method `METHOD_REGISTRY` numbering.

**Failing cases for this method (why it's 27, not 30):**
- `chemistry` — Arrhenius Equation (R²=0.087)
- `chemistry` — Henderson-Hasselbalch (R²=0.886)
- `fluid_dynamics` — Bernoulli's Equation (R²=0.981)

Worth flagging: **Arrhenius** is the same equation used as the reproduction case for Bug 10b's timeout fix, and chemistry is the domain with two of the three failures here. Nothing in this data links the two bugs causally (this is a fit-quality shortfall, not a hang), but chemistry-domain PySR fits are clearly a weak spot worth a closer look independently of 10b.

**Remaining verification (not done here, low-risk):** confirm against the actual paper source (`jmlr-hypatiax*.tex` / Table 4 caption) that (a) Table 4 is in fact sourced from this `all_domains` 30-test run rather than a different 30-count set, and (b) the table's own M-column header confirms the "exclude idx 3, renumber" convention assumed above. `audit_nb04` (Numerical Consistency & Abstract Claims) is the existing pipeline step that should catch this if the convention is wrong.

---

## Item 10a — 900s vs 1100s: RESOLVED (doc-only)

`run_all.sh` already resolves this: `FEYNMAN_TIMEOUT=1100` is explicitly the corrected paper value —

```
export FEYNMAN_TIMEOUT=1100   # FIX-G2: paper value 1100s (was 900)
```

— while `METHOD_TIMEOUT=900` is a distinct variable for a different budget (the outer per-method wall-clock allowance, not the PySR fit timeout). **1100s is correct**; the sweep write-up's "900s" reference is stale. No code change — just align both write-ups on 1100s and note that `METHOD_TIMEOUT` (900s) and `PYSR`/`FEYNMAN_TIMEOUT` (1100s) are intentionally different knobs, not a typo.

---

## Item 10b — 300s timeout enforcement: DIAGNOSIS UPDATED, FIX SCOPED, NOT YET IMPLEMENTED

**Timeline of the diagnosis in this investigation:**

1. **Original claim:** killing the wrapper process didn't kill the Julia process inside it → 27,574s hang. "Fix": run the subprocess in its own process group (`start_new_session=True`) and kill the whole group on timeout.
2. **Correction #1:** `symbolic_engine.py` shows PySR is invoked via `juliacall`, which embeds Julia **in-process** (loads `libjulia` as a shared library), not as a separate OS subprocess. `PYTHON_JULIACALL_HANDLE_SIGNALS` is a juliacall-specific in-process signal-handoff setting — it only makes sense if Julia is embedded. Grepping both `symbolic_engine.py` and `hybrid_system_v50_2.py` for calls to an actual `julia` executable: zero matches. So there's no separate Julia PID to orphan, and the process-group fix, while harmless, likely wasn't addressing the real mechanism.
3. **Correction #2 (revised real cause):** the outer layer (`ThreadPoolExecutor` + `future.result(timeout=300)`) governs the 300s tests-1–18 limit; the inner layer (`_run_pysr_in_subprocess`'s own `communicate(timeout=...)`) uses the larger 900/1100s budget. When the outer 300s fires, the harness abandons the background thread but never actually stops it — `proc` (the real subprocess handle) is a local variable three call layers below the outer timeout handler, with no reference threaded back up. That's the real gap; `_kill_thread` was a workaround for not having that reference, and it silently did nothing because it referenced an undefined function.
4. **New evidence this round (`symbolic_engine.py`, `wall_clock_flags.json`):** `symbolic_engine.py` already has a `PROC_TIMEOUT` env-var cap (comment: `# Fix: Step 4 from diagnosis`) that shrinks PySR's own `timeout_in_seconds` to stay under an outer wall-clock budget *before* PySR ever needs to be killed externally. And critically, `wall_clock_flags.json` from the `exp1_five` run shows PySR's own cooperative timeout working correctly on its own: the three DeFi cases that hit the ceiling (Impermanent Loss, Price Impact, Constant Product) stopped at **1101–1115s** — essentially exactly `pysr_timeout=1100` plus Julia startup overhead — not a runaway. `hybrid_system_v50_2.py` never touches `subprocess`/`ThreadPoolExecutor`/`PROC_TIMEOUT` at all.

**Conclusion:** the 27,574s hang is isolated to `run_comparative_suite_benchmark_v2_FIXED.py`'s own outer/inner timeout wiring — not a property of the underlying engine, which self-bounds correctly via PySR's cooperative `timeout_in_seconds` whether or not an outer wrapper is watching.

**What's still open:**
- The `_proc_box` threading fix (pass a mutable handle for the real subprocess `proc` up through `symbolic_engine.py` / `hybrid_system_v50_2.py` to the outer `ThreadPoolExecutor` timeout handler in the harness) is **designed but not implemented**.
- Verification should be a **wall-clock timing test**, not a Julia process-tree check (Julia can't be installed in this sandbox — `julialang-s3.julialang.org` is network-blocked, confirmed `host_not_allowed`, and there's no GitHub-hosted mirror). Run the harness with a short outer `--method-timeout` against a case with a long inner PySR budget and confirm the harness's own elapsed time tracks the outer timeout, not the inner one.

**Recommended next action:** implement the `_proc_box` fix, then re-run the Arrhenius-style short-timeout test purely to time the harness's outer-timeout responsiveness (no Julia process inspection needed).

---

## Summary

| Item | Status | Action needed |
|---|---|---|
| 2 (27/30 M4) | Resolved | Confirm against actual paper Table 4 source / `audit_nb04` |
| 10a (900 vs 1100) | Resolved | Align both write-ups on 1100s; no code change |
| 10b (300s enforcement) | Root cause re-confirmed and narrowed | Implement `_proc_box` fix in the CI harness; verify via wall-clock timing, not process-tree inspection |

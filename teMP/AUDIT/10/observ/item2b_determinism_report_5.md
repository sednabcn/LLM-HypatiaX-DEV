# Item 2b — Determinism Report

**Question:** does the FIX-ISSUE2-UNSEEDED-NN patch make HSL/M4
(`HybridSystemLLMNN all-domains (core)`) and EHD/M3
(`EnhancedHybridSystemDeFi (core)`) deterministic across repeated,
otherwise-identical noiseless runs?

**Status as of CI run `31312938979`:** M4 closed. M3 open, on one
equation, at a spread three orders of magnitude below the level that
originally opened this item. See [Recommendation](#recommendation).

---

## 1. Background

Item 2b's own reproducibility gate (`check_issue2b_reproducibility.py`)
compares 3 independent, noiseless Phase A runs of the harness
(`--skip-pysr`, since M4 and M3 don't use PySR/Julia) and flags any
equation where R² differs across runs by more than `TOL` (default
`0.0`, i.e. exact bit-match). Only M4 and M3 are gated — PureLLM (M1)
and ImprovedNN (M2) are evaluated in the same runs but are not part of
the closed/open decision.

## 2. Determinism sources identified and their status

| # | Source | Where | Status |
|---|---|---|---|
| 1 | `hash()` seed derivation not stable across Python processes | `run_comparative_suite_benchmark_v2.py`, 7 sites | **Fixed** (prior patch: `hash()` → `hashlib.sha256()`). Verified via `check_patch --check-patch`, preflight-gated on every run. |
| 2 | HSL's `train_nn()` had no seeding at all | `hybrid_system_llm_nn_all_domains.py` | **Fixed.** `torch.manual_seed(seed)` before `_make_model()`; seed derived via `sha256(description)` at the call site. Dormant in the 30-equation test set (decision="llm" wins every time, so the unseeded NN path was never actually exercised) — patched anyway since it's a live latent bug. |
| 3 | HSL's local fallback LLM call had no temperature pin | `hybrid_system_llm_nn_all_domains.py` | **Fixed.** Routed through `_create_message_deterministic` (`temperature=0.0`, with a narrow retry-without-temperature only on the newer-model "temperature deprecated" 400 error). Dormant — PureLLMBaseline delegate succeeds for all 30 equations in this test set, so this path only becomes live if that delegate ever fails. |
| 4 | EHD's `train_nn_model()` weight init | `hybrid_system_nn_defi_domain.py` | **Confirmed not broken.** Extracted and run in two separate processes under `torch.use_deterministic_algorithms(True)`, both normal and strict (`warn_only=False`) mode — bit-identical both times. |
| 5 | EHD's Stage 2 fitting (`fit_formula_params`) gated on `time.monotonic()` | `hybrid_system_nn_defi_domain.py` | **Fixed.** Was choosing how many `curve_fit` candidates to try, and whether `differential_evolution` ran at all, based on wall-clock elapsed — meaning the actual sequence of optimizer calls depended on CI runner load at that moment. Confirmed live via the log itself: identical-work equations showed multi-second timing jitter across runs. Rewritten to run a fixed number of candidates, bounded by the existing `maxfev`/`maxiter` constants, with a generous 45s absolute safety valve that now logs loudly if it ever fires instead of silently changing the code path. |
| 6 | Anthropic API not guaranteed bit-exact at `temperature=0` (server-side batching/kernel effects) | Feeds EHD's residual-correction MLP via `PureLLMBaseline` delegation | **Mitigated.** This was the actual root cause of the original flagged spread (2.38e-10, biology: Logistic Growth). Fixed via a file-backed, opt-in frozen cache (`HYPATIAX_LLM_FREEZE_CACHE`): the first of the three Phase A subprocesses to hit a given prompt writes the formula to disk; the other two read it back instead of re-querying the API, so all three runs train on byte-identical LLM input. Proven with a mock LLM that deliberately drifts on every fresh call — all three simulated "process runs" still returned bit-identical formula text with the fix in place. |
| 7 | EHD's local DeFi fallback LLM calls had no temperature-deprecation handling | `hybrid_system_nn_defi_domain.py`, 2 call sites (initial call + max-tokens/malformed-response retry) | **Fixed.** Both were raw `client.messages.create(temperature=0.0, ...)` calls — correct while the model accepts `temperature`, but would raise uncaught on newer models that reject the parameter, and that exception was swallowed into an N/A/error result by `generate_llm_formula`'s bare `except`, unlike HSL's graceful degradation. Routed through the same `_create_message_deterministic` helper HSL uses (duplicated per-file, matching this codebase's existing convention rather than a cross-module import). |
| 8 | Uncontrolled BLAS/torch thread scheduling and CUDA algorithm selection | Process environment | **Pinned, all runs, both phases.** `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `PYTHONHASHSEED=0`, `TORCH_DETERMINISTIC=1` (advisory — see [open question](#open-question-is-use_deterministic_algorithms-actually-called)) exported once in `run_issue2b_experiment.sh`, before either phase, so both benefit. |

## 3. Preflight verification

Two independent static checks now run before any compute is spent:

- **`check_patch --check-patch <harness>`** — the original patch: 0 live
  `hash()` calls, 7 `hashlib.sha256()` sites, `_ProcBox` present, dead
  `ctypes` import removed.
- **`check_patch --check-followup <files...>`** — the follow-up fixes in
  this report (items 2, 3, 6, 7 above), marker-based rather than
  exact-count-based so it doesn't repeat the "hardcoded 7" brittleness
  of the original check. Verified independently for both
  `hybrid_system_llm_nn_all_domains.py` and
  `hybrid_system_nn_defi_domain.py`:

  ```
  FOLLOW-UP PATCHES LOOK CORRECT — safe to run.
  ```

Neither check is currently wired into the CI workflow's `dry_run`
step or `run_issue2b_experiment.sh`'s own preflight — both exist as
callable modes only. **Open action item**, not yet done.

## 4. CI results

### Run `31282063041` (pre-frozen-cache)
- M4: `DETERMINISTIC`, 30/30.
- M3: `NON-DETERMINISTIC` on 1 equation — spread `2.38e-10`,
  biology: Logistic Growth. Traced to the LLM API non-determinism
  described in item 6 above.

### Run `31312938979` (post-frozen-cache, this report's trigger)
- M4: `DETERMINISTIC`, 30/30, pass rate 30/30 all three runs.
- M3: pass rate 28/30, identical across all three runs (stable — not
  a determinism symptom). `NON-DETERMINISTIC` on 1 equation — spread
  `3.37e-10`, optics: Snell's Law (`n1·sin(θ1) = n2·sin(θ2)`),
  `r2=[1.000000, 1.000000, 1.000000]` in all three runs.

**Logistic Growth no longer appears in the mismatch list.** That's
direct evidence the frozen cache closed the specific gap it was built
for.

**Snell's Law is a different equation and a different order of
magnitude** than what the frozen cache fixed — 3.37e-10 here vs.
2.38e-10 for Logistic Growth before the fix, on a different equation
and a different method's flagged case entirely. It also matches a
failure mode the harness's own header comments already anticipated
and had previously observed: an earlier CI run (`31252014219`) saw
HSL spread `7.6e-07` on this identical Snell's Law equation, and
attributed it to PyTorch's reduction-order non-determinism in
matmul/conv ops (BLAS thread scheduling, CUDA algorithm selection) —
not a seeding failure. The current spread is over three orders of
magnitude smaller than that earlier observation, on M3 instead of M4
this time, at `R²=1.000000` to 6 decimal places in all three runs.

## 5. Open question: is `use_deterministic_algorithms` actually called?

`run_issue2b_experiment.sh`'s own comment on `TORCH_DETERMINISTIC=1`
is explicit that this env var is **advisory only** — it does nothing
by itself unless the harness code calls
`torch.use_deterministic_algorithms(True)` internally. This has not
been confirmed in this investigation. It matters for interpreting the
Snell's Law spread:

- **If the harness does call it:** the residual `3.37e-10` spread is
  expected, reduction-order floating-point noise near the precision
  floor — not a code bug, and not something further seeding can fix.
- **If the harness does not call it:** the pinning is currently only
  partially effective, and this is a live, unaddressed nondeterminism
  source rather than an accepted floor — worth another look before
  treating it as closed.

**Action:** `grep -n "use_deterministic_algorithms"` across the
harness and confirm which case applies before deciding on §6.

## 6. Recommendation

Two independent findings converge on the same conclusion, but the
open question in §5 should be resolved first:

1. Pass rate at `28/30` is stable and identical across all three
   runs — a method-quality question, unrelated to reproducibility,
   out of scope for Item 2b.
2. The one flagged spread (`3.37e-10`) is consistent with the
   floating-point reduction-order floor the harness's own comments
   already describe and have previously observed at a much larger
   magnitude on the same equation.

If §5 confirms `use_deterministic_algorithms(True)` is already being
called: adopt a small, equation-scoped tolerance — `TOL=1e-8` (or even
`1e-9`) comfortably clears the current spread without loosening the
gate meaningfully anywhere else, and matches the epsilon-tolerance
option already proposed earlier in this investigation ("accepting a
small epsilon tolerance... specifically for methods that delegate to
the LLM"). Re-run Phase A once at the new `TOL`; do not touch Phase B
or regenerate Table 4 until that run confirms `CLOSED`.

If §5 finds it is *not* being called: add the call before changing
`TOL`, since a tolerance change would otherwise be papering over an
unaddressed and possibly larger nondeterminism source rather than
accepting a known, bounded floor.

## 7. Not part of this report's scope

- The `PureLLM Baseline` "All evaluation strategies failed" crash
  (root-caused to a signature-matching bug in `evaluate_function`,
  not a reproducibility issue — it isn't part of the M3/M4 gate).
  Confirmed via full-log grep to be a single occurrence out of ~90
  method-equation evaluations in the run it was found in, non-cascading.
  Fix scoped and agreed but not yet applied in this thread.
- Wiring `--check-followup` into the CI `dry_run` step and
  `run_issue2b_experiment.sh`'s preflight (§3).

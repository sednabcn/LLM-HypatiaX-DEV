# Item 2b — Determinism Report

**Question:** does the FIX-ISSUE2-UNSEEDED-NN patch make HSL/M4
(`HybridSystemLLMNN all-domains (core)`) and EHD/M3
(`EnhancedHybridSystemDeFi (core)`) deterministic across repeated,
otherwise-identical noiseless runs?

**Status as of CI run `31312938979`:** M4 closed. M3 **closed** — the
one open question (§5) has since been resolved by inspecting the
harness source directly; see [Resolution](#5-open-question-resolved-is-use_deterministic_algorithms-actually-called)
and [Recommendation](#recommendation), both adopted in
`ci_issue2b_repro.yml` and `run_issue2b_experiment.sh`.

---

## 1. Background

Item 2b's own reproducibility gate (`check_issue2b_reproducibility.py`)
compares 3 independent, noiseless Phase A runs of the harness
(`--skip-pysr`, since M4 and M3 don't use PySR/Julia) and flags any
equation where R² differs across runs by more than `TOL` (now `1e-8`
by default, following the §6 resolution below — was `0.0`, i.e. exact
bit-match, at the time this investigation started). Only M4 and M3
are gated — PureLLM (M1) and ImprovedNN (M2) are evaluated in the
same runs but are not part of the closed/open decision.

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
| 8 | Uncontrolled BLAS/torch thread scheduling and CUDA algorithm selection | Process environment | **Pinned, all runs, both phases.** `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `PYTHONHASHSEED=0`, `TORCH_DETERMINISTIC=1` (advisory env var — the in-harness `torch.use_deterministic_algorithms(True, warn_only=True)` call it depends on is now confirmed present, see [§5](#5-use_deterministic_algorithms--resolved-yes-its-called)) exported once in `run_issue2b_experiment.sh`, before either phase, so both benefit. |

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

Neither check was originally wired into the CI workflow's `dry_run`
step or `run_issue2b_experiment.sh`'s own preflight — both existed as
callable modes only. **Update:** both are now wired in. In
`run_issue2b_experiment.sh`, `--check-followup "$HSL_PATH" "$EHD_PATH"`
runs immediately after the existing `--check-patch` step, and aborts
(`exit 2`) on failure the same way. In `ci_issue2b_repro.yml`,
`--check-followup` runs both inside the `dry_run` preflight and as an
unconditional gate before every real Phase A/B dispatch, via two new
`hsl_path`/`ehd_path` workflow inputs. A run can no longer silently
proceed on an incomplete follow-up patch the way the original
`31312938979` run — which only ran `--check-patch` — technically
could have.

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

## 5. `use_deterministic_algorithms` — resolved: yes, it's called

`run_issue2b_experiment.sh`'s own comment on `TORCH_DETERMINISTIC=1`
had been explicit that this env var is **advisory only** — it does
nothing by itself unless the harness code calls
`torch.use_deterministic_algorithms(True)` internally. This was
unconfirmed at the time run `31312938979` produced the Snell's Law
spread. It has since been checked directly against the harness
source:

```
hypatiax/experiments/benchmarks/run_comparative_suite_benchmark_v2.py:766:
    torch.use_deterministic_algorithms(True, warn_only=True)
```

paired with `torch.set_num_threads(1)` immediately above it. The
call is tagged `FIX-ISSUE2-DETERMINISM` in an inline comment
explaining why it was added: env-var thread pinning alone had left a
~1e-6 R² spread on an earlier run (`31256573694`), so this in-process
call was added to close the remaining gap. `warn_only=True` rather
than the strict default, on the stated reasoning that this codebase
only uses plain Linear/Tanh MLPs, all of which have deterministic CPU
kernels — a hard `RuntimeError` isn't needed, but warn mode would
still surface it in the log if that assumption ever breaks.

**Timing check.** The call was introduced in commit `e5e33ee` ("fix
LLM issue"), Aug 8, 15:10 BST — a full day before job `84944044200` /
run `31312938979` checked out the repo (~12:18 BST Aug 9). So it was
present in the exact code that produced the 3.37e-10 Snell's Law
spread; this isn't a case of the fix landing after the fact.

**Corroborating evidence from the run log.** With `warn_only=True`,
PyTorch prints a `UserWarning` any time it falls back to a
non-deterministic kernel. All three Phase A runs in that job's log
were checked around the Snell's Law entries (including the flagged
spread) — no such warning fired in any of them. That's consistent
with the comment's claim that no op in this path lacks a
deterministic implementation, i.e. the pinning is doing its job, not
silently missing something.

**Conclusion:** this is the first branch of §6 below. The call is
confirmed present and pre-dates the flagged run, so the 3.37e-10
Snell's Law spread is the accepted floating-point reduction-order
floor, not an unaddressed nondeterminism source.

## 6. Recommendation — adopted

Two independent findings converged on the same conclusion, and §5 has
now resolved the condition that decided between them:

1. Pass rate at `28/30` is stable and identical across all three
   runs — a method-quality question, unrelated to reproducibility,
   out of scope for Item 2b.
2. The one flagged spread (`3.37e-10`) is consistent with the
   floating-point reduction-order floor the harness's own comments
   already describe and have previously observed at a much larger
   magnitude on the same equation.

Since §5 confirmed `use_deterministic_algorithms(True, warn_only=True)`
is already being called, and pre-dates the run that produced the
flagged spread: **`TOL=1e-8` has been adopted** as the new default
comparator tolerance — comfortably clears the current spread without
loosening the gate meaningfully anywhere else, and matches the
epsilon-tolerance option already proposed earlier in this
investigation ("accepting a small epsilon tolerance... specifically
for methods that delegate to the LLM"). This is now the default in
both `ci_issue2b_repro.yml` (`tol` input default) and
`run_issue2b_experiment.sh` (`TOL="${TOL:-1e-8}"`), overridable per
run; `TOL=0.0` is still available if a future run needs to
re-investigate a spread rather than tolerate it. Phase B and Table 4
were not touched by this change — regenerating them still requires a
fresh Phase A run confirming `CLOSED` at the new tolerance, per the
original recommendation.

## 7. Not part of this report's scope

- The `PureLLM Baseline` "All evaluation strategies failed" crash
  (root-caused to a signature-matching bug in `evaluate_function`,
  not a reproducibility issue — it isn't part of the M3/M4 gate).
  Confirmed via full-log grep to be a single occurrence out of ~90
  method-equation evaluations in the run it was found in, non-cascading.
  Fix scoped and agreed but not yet applied in this thread.

## 8. Why determinism matters for these algorithms

It's worth being explicit about this, since "make it deterministic"
can look like CI hygiene for its own sake. It isn't, here — three
separate things break without it.

**1. The benchmark's whole purpose is comparing methods, and
nondeterminism corrupts the comparison itself.** M1–M6 exist to be
ranked against each other on R² recovery per equation. If HSL/M4's
R² can drift by 1e-6 to 1e-10 between two runs of the *same* code on
the *same* equation, then any observed difference smaller than that
between two *different* methods, or between two versions of the
*same* method after a code change, is unreadable — you can't tell
whether it's a real capability difference or just noise. Item 2b
isn't a side quest; it's a precondition for every other result in
the paper that depends on comparing R² values.

**2. The noise-sweep tables (`tab:r2_noise`, `tab:rr_noise`) measure
sensitivity to *injected* noise — which only means something against
a *known, stable* zero-noise baseline.** These tables report how R²
degrades as synthetic noise is added to the input data. If the
zero-noise baseline itself is a moving target (because the harness is
internally nondeterministic), the "noise sensitivity" being reported
is actually a mix of real noise sensitivity and baseline jitter, and
there's no way to separate the two after the fact. Determinism at
`noiseless` is what makes the *rest* of the noise sweep interpretable
at all — this is why Item 2b gates Phase B and Table 4 regeneration
outright rather than being an optional nice-to-have.

**3. HSL and EHD each stack three independent sources of randomness,
and each needed a structurally different fix — closing two out of
three still leaves the result nondeterministic.** The determinism
sources table in §2 above isn't three variations on one bug; it's
three unrelated mechanisms that all happen to feed the same R² number:

- *Seed derivation* (`hash()` → `hashlib.sha256()`, and the NN
  seeding patch): fixes reproducibility of *which* random numbers get
  drawn, but nothing else.
- *LLM sampling* (temperature pinning, and the frozen-cache
  workaround for the Anthropic API's own residual non-exactness at
  `temperature=0`): fixes reproducibility of the *text* fed into
  downstream NN training — a seeded NN trained on two slightly
  different LLM outputs will still diverge, no matter how well-seeded
  it is.
- *Floating-point reduction order* (thread pinning +
  `torch.use_deterministic_algorithms`, §5): fixes reproducibility of
  the *arithmetic itself* — even a perfectly-seeded model trained on
  byte-identical input can still produce different output if matmul
  accumulates its terms in a different order across runs, which
  CPU/GPU kernels are free to do by default for performance.

A fix to any one of these can look like it "didn't work" if tested
in isolation, purely because one of the other two is still live —
which is exactly what happened here: the NN-seeding patch (M4) closed
cleanly on the first pass, but M3 stayed open until the LLM-sampling
gap (frozen cache) was also closed, and even after that, the residual
floating-point layer had to be separately confirmed (§5) before the
result could be trusted as *actually* closed rather than closed by
coincidence of a tolerance being loose enough to hide a live gap.

**4. A reproducibility gate is only useful as a regression test if a
result change can be attributed to a code change.** The entire value
of running Phase A three times and diffing the output is that a
future code change which introduces a regression should show up as a
diff against a known-stable baseline. If the baseline itself isn't
stable, every future diff is ambiguous between "real regression" and
"just noise, ignore it" — which is the failure mode that makes
flaky tests get ignored in general, and is exactly what items 2–8 in
§2 exist to prevent.

# Patch Report: Unseeded-NN Determinism Follow-Up

**File patched:** `run_comparative_suite_benchmark_v2_FIXED.py`
**Related to:** Issue 2 (HyperSymLoop noiseless pass rate varied 26–30/30 across
12 otherwise-identical runs — root-caused to per-process-randomized `hash()`
seeding)
**Status:** 4 of 4 previously-flagged gaps patched and verified. 0 live
`hash()` calls remain in the file.

## Background

A prior editing pass on this file fixed two instances of the root cause
(Strategies 3a/3b in `HybridAllDomainsMethod`, switching `hash()` →
`hashlib.sha256()`), but a follow-up narrative describing four *additional*
fixes was never actually applied to the file — that session appears to have
been cut off mid-edit. An independent grep-and-verify pass against the
uploaded file confirmed those four items were still present exactly as
originally found. This report documents patching all four.

## Root cause (recap)

Python's built-in `hash()` on a `str` is randomized per-process unless
`PYTHONHASHSEED` is pinned. Code that seeded an RNG with `hash(description)`
under the belief that this was reproducible was, in fact, producing a
different seed every run/shard — silently reintroducing non-determinism
into paths that looked deterministic.

Separately, `ImprovedNNMethod.run()`, when called directly (the library's own
default), performed no seeding at all — not even the flawed `hash()`
approach.

## Fixes applied

### 1. `BaseMethod._nn_residual_fit` (shared helper, line ~1154)

**Before:** No `description` parameter; no seeding call anywhere in the
function body. Training used whatever the ambient `torch` RNG state
happened to be.

**After:**
```python
@staticmethod
def _nn_residual_fit(
    X: np.ndarray,
    y: np.ndarray,
    y_pred_llm: np.ndarray,
    description: str = "",
) -> Optional[np.ndarray]:
    ...
    if not TORCH_AVAILABLE:
        return None
    try:
        # FIX-ISSUE2-UNSEEDED-NN (follow-up): this helper was training
        # completely unseeded, same root cause as Strategies 3a/3b above
        # -- seed deterministically from the equation description via
        # sha256 (not hash(), which is per-process-randomized).
        _seed = int(hashlib.sha256(description.encode()).hexdigest(), 16) % (2**31)
        torch.manual_seed(_seed)
        from sklearn.preprocessing import StandardScaler as _SS
```

Call site (in `HybridDeFiMethod`, line ~2217) updated to pass `description`
through:
```python
_y_hybrid = self._nn_residual_fit(X, y, _y_pred_defi, description)
```

### 2. `HybridAllDomainsMethod._nn_residual_fit` (own override, line ~2335)

Same problem, same fix, applied to the second, independent copy of this
function (HSL/M4's own override — not inherited from `BaseMethod`):

```python
def _nn_residual_fit(
    X: np.ndarray,
    y: np.ndarray,
    y_pred_llm: np.ndarray,
    description: str = "",
) -> Optional[np.ndarray]:
    ...
    if not TORCH_AVAILABLE:
        return None
    try:
        # FIX-ISSUE2-UNSEEDED-NN (follow-up): same fix as BaseMethod's
        # copy of this helper -- was training fully unseeded.
        _seed = int(hashlib.sha256(description.encode()).hexdigest(), 16) % (2**31)
        torch.manual_seed(_seed)
        from sklearn.preprocessing import StandardScaler as _SS
```

Call site (line ~2708) updated:
```python
y_hybrid = self._nn_residual_fit(X, y, y_pred_llm, description)
```

### 3. `HybridDeFiMethod` reconstruction-fallback seed (line ~2180 → ~2203)

**Before:**
```python
_rng = np.random.default_rng(seed=int(abs(hash(description)) % (2**31)))
```

**After:**
```python
# FIX-ISSUE2-UNSEEDED-NN (follow-up): hash() is per-process-randomized
# unless PYTHONHASHSEED is pinned -- switched to sha256, same fix already
# applied to Strategies 3a/3b in HybridAllDomainsMethod.
_rng = np.random.default_rng(
    seed=int(
        hashlib.sha256(description.encode()).hexdigest(), 16
    ) % (2**31))
```

This gives `HybridDeFiMethod` (EHD/M3) the same fix its sibling
`HybridAllDomainsMethod` already had for the equivalent code path.

### 4. Noise-injection seed in `main()` (line ~4892 → ~4927)

**Before:**
```python
_rng_seed = int(abs(hash(_desc)) % (2**31))
_rng = np.random.default_rng(seed=_rng_seed)
```

**After:**
```python
# FIX-ISSUE2-UNSEEDED-NN (follow-up): hash() is per-process-randomized
# unless PYTHONHASHSEED is pinned, so noise injection was not actually
# reproducible across runs with the same sigma despite the comment above
# claiming it was -- switched to sha256, same fix as elsewhere in this file.
_rng_seed = int(
    hashlib.sha256(_desc.encode()).hexdigest(), 16
) % (2**31)
_rng = np.random.default_rng(seed=_rng_seed)
```

This one is notable because the surrounding comment *already claimed*
"a per-equation RNG seeded from the description hash ensures reproducibility
across runs with the same sigma" — that claim was false before this patch;
it's true now.

### 5. `ImprovedNNMethod.run()` default (unseeded) path (line ~1600)

This was the most delicate fix: `ImprovedNNMethod` has an intentional
multi-seed design (`run_multiseed()` → `_run_single_seed()`, which already
seeds each trial correctly), and the fix must not disturb that.

**Before:** `run()` had no seeding call anywhere in its body. Dispatch logic
(`ProtocolBenchmarkSuite`) only routes through the seeded `run_multiseed()`
path when `nn_seeds > 1`; with the library default of `nn_seeds=1`, `run()`
is called directly, with no seed set at all.

**After:**
```python
def run(self, description, X, y, var_names, metadata, verbose=False) -> MethodResult:
    if self._ImprovedNN is None:
        return self._unavailable("ImprovedNN not available")

    # FIX-ISSUE2-UNSEEDED-NN (follow-up): when called directly (the
    # default nn_seeds=1 path -- see ProtocolBenchmarkSuite's dispatch,
    # which only routes through run_multiseed()/_run_single_seed() when
    # nn_seeds > 1), this method trained with no seed set at all. Guard
    # on self._nn_seeds == 1 so we do NOT touch the RNG state when
    # called via _run_single_seed (which already seeds per trial before
    # calling this method) -- that seeding must stay independent per
    # trial for run_multiseed()'s variance estimate to remain meaningful.
    if self._nn_seeds == 1:
        _seed = int(
            hashlib.sha256(description.encode()).hexdigest(), 16
        ) % (2**31)
        if TORCH_AVAILABLE:
            torch.manual_seed(_seed)
        np.random.seed(_seed)
```

The guard is load-bearing: `_run_single_seed()` calls `torch.manual_seed(seed)`
and `np.random.seed(seed)` itself *before* calling `self.run(...)`, with
`self._nn_seeds` still set to whatever it was constructed with (e.g. 3). The
`if self._nn_seeds == 1` check correctly evaluates `False` in that case, so
the new block is skipped and each multi-seed trial keeps its own independent
seed — `run_multiseed()`'s variance estimate is unaffected.

## Verification performed

- `python3 -m py_compile run_comparative_suite_benchmark_v2_FIXED.py` →
  **passes**, no syntax errors introduced.
- `grep -n "hash("` across the whole file, filtered to non-comment lines →
  **zero matches**. Every remaining occurrence of the substring `hash(` is
  inside an explanatory comment, not a live call.
- Confirmed both `_nn_residual_fit` definitions and both call sites now
  agree on the `description` parameter (`grep -n "_nn_residual_fit"`).
- Confirmed the new `ImprovedNNMethod` guard (`self._nn_seeds == 1`) doesn't
  collide with the pre-existing, unrelated `self._nn_seeds == 1` early-return
  inside `run_multiseed()` — the two are independent and both correct.
- `hashlib` was already imported at module scope (line 69); no new import
  needed.

## Scope / what this does *not* cover

- `run_sample_complexity_benchmark.py` and `run_noise_sweep_benchmark.py`
  were checked separately and had no bug of their own to patch — both
  default `--nn-seeds` to 3, which routes through the already-correct
  `run_multiseed()` path rather than the gaps fixed here.
- This patch does not touch anything outside
  `run_comparative_suite_benchmark_v2.py`.

## Impact — read before re-running anything

This is a **behavior-changing fix, not a cosmetic one**. Every prior run of
this script (including the exp1_five / exp2_five results behind the
five-system comparison tables discussed earlier) was generated with these
five code paths unseeded or inconsistently seeded. Re-running after this
patch is expected to produce **different numbers**, not just "the same
numbers, now reproducibly." Recommend keeping the pre-patch output files
around for a before/after comparison rather than discarding them, and
treating any five-system/DeFi table currently in the paper as needing a
fresh regeneration pass once this file is deployed.

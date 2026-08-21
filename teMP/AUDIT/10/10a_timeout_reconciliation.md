# Item 10a — 900s vs 1100s reconciliation

**Status: Resolved (doc-only, no code change).**

`run_all.sh` (per the Phase 3 investigation) already sets the corrected value explicitly:

```
export FEYNMAN_TIMEOUT=1100   # FIX-G2: paper value 1100s (was 900)
```

`900s` and `1100s` are **two different knobs, not a typo**:

| Variable | Value | Meaning |
|---|---|---|
| `METHOD_TIMEOUT` | 900s | outer per-method wall-clock allowance |
| `FEYNMAN_TIMEOUT` / PySR `timeout_in_seconds` | 1100s | inner PySR fit timeout (the paper-comparable value) |

**Action for both write-ups:** wherever a sweep table or note cites "900s" as *the* PySR timeout, change it to 1100s and add a one-line clarification that 900s refers to the separate outer `METHOD_TIMEOUT`. No source file needs to change — `run_all.sh` is already correct; this is purely aligning prose in the write-ups to match it.

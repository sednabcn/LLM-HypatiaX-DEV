# LaTeX audit-fix patch tooling

Applies a fix manifest (`patch_manifest.yaml`) to your `.tex` files safely:
unique-match find/replace only, no line numbers, no guessed values for
items that need investigation, and a provenance ledger so a value can
change later without silently clobbering what's already there.

## Setup

    pip install pyyaml --break-system-packages
    # put patch_runner.py, patch_manifest.yaml, scripts/ next to your .tex files
    # (or edit the `file:` paths in patch_manifest.yaml to point at them)

## Usage

    # Dry run — reports what WOULD happen, changes nothing on disk
    python3 patch_runner.py --manifest patch_manifest.yaml

    # Apply everything that's ready
    python3 patch_runner.py --manifest patch_manifest.yaml --apply

    # Just one or a few items
    python3 patch_runner.py --manifest patch_manifest.yaml --apply --ids 5,7,13

    # Recompile check only, no patching
    python3 patch_runner.py --manifest patch_manifest.yaml --compile-only

    # A value that was already patched has changed (new investigation,
    # re-run, etc.) — the runner refuses this by default; opt in explicitly:
    python3 patch_runner.py --manifest patch_manifest.yaml --apply --ids 3 --allow-value-change

## Manifest item statuses

| status               | meaning                                                          | what the runner does |
|----------------------|-------------------------------------------------------------------|-----------------------|
| `ready`               | `old_str`/`new_str` known and confirmed unique                    | patches directly |
| `needs_investigation` | value must come from `investigation_script`                       | runs the script, caches result, patches from the returned value |
| `blocked`             | fix is gated on something not yet done (e.g. a pending re-run)    | only inserts an explicit caveat footnote — never a guessed number |
| `SUPERSEDED`          | a *later* audit already resolved this differently                 | skipped, reason printed, points at the real fix |
| `TODO`                | not yet filled in                                                  | skipped, reason printed |

## Rules that keep this safe to automate

1. **`old_str` must come from `grep`-ing the raw `.tex`, never copied from a
   rendered PDF.** Macros expand differently in the rendered document than
   they appear in source (e.g. `\HSL` vs. its expansion) — copying from a
   PDF report will silently fail to match, which is the *safe* failure
   mode, but grep the source directly to avoid the wasted round-trip.

2. **A match count of anything other than 1 is a hard refusal**, not a
   warning. Zero matches usually means the file already changed underneath
   you; more than one means your anchor isn't unique enough to trust
   unattended.

3. **Cross-check every item against your existing audit trail before
   marking it `ready`.** A newer audit pass may have already resolved the
   same issue differently — patch tooling has no way to know that on its
   own; only you (or a `cross_checked_against` note reviewed by a human)
   catches it. Use the `SUPERSEDED` status to record when this happens
   instead of silently dropping the item.

4. **Investigation scripts read raw data only, never the manuscript.** They
   print one JSON value (or exit non-zero) — see
   `scripts/investigate_item3_runtime.py` for the contract.

5. **Value changes require an explicit flag.** `applied_log.json` (created
   next to your files on first `--apply`) remembers what was last written
   and where it came from. If a later investigation produces a different
   value for an item that's already been patched, the runner reports
   `VALUE-CHANGED` and does nothing further until you pass
   `--allow-value-change`. Full before/after history is kept in
   `applied_log.json` per item, so nothing is ever silently overwritten.

6. **Recompile after every real apply.** The runner shells out to
   `pdflatex` (2 passes) on every touched file and reports new errors or
   undefined references, so a bad patch is caught immediately.

## Recommended workflow with git

Commit per item, not per batch, so every number change traces back to a
reason:

    python3 patch_runner.py --manifest patch_manifest.yaml --apply --ids 5
    git add -A && git commit -m "fix(item5): attach seed/split to '68 of 74' citation [plan-action #5]"

    python3 patch_runner.py --manifest patch_manifest.yaml --apply --ids 3 --allow-value-change
    git add -A && git commit -m "fix(item3): update EHSDeFi runtime 841.4s -> 312.9s per re-trace of raw logs [plan-action #3]"

## Files

- `patch_runner.py` — the runner (no manuscript-specific logic in here).
- `patch_manifest.yaml` — your fix list, seeded with the 13 plan-action
  items; several are `TODO` pending you confirming the exact source text.
- `scripts/investigate_item3_runtime.py` — template/contract for
  investigation scripts; copy this pattern for items #1, #4, #12.

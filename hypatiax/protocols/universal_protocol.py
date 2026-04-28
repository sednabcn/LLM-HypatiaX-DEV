import json
import os
import time

# ── hash_dict: try all known locations, inline fallback last ──────────────
# Priority:
#   1. hypatiax/reproducibility/hash_lock.py  (canonical after restructure)
#   2. reproducibility/hash_lock.py           (pre-restructure repo root)
#   3. hypatiax/shared/utilities.py           (alternative post-restructure)
#   4. shared/utilities.py                    (alternative pre-restructure)
#   5. Inline SHA-256 — self-contained fallback, always deterministic
try:
    from hypatiax.reproducibility.hash_lock import hash_config as hash_dict
except ImportError:
    try:
        from reproducibility.hash_lock import hash_config as hash_dict
    except ImportError:
        try:
            from hypatiax.shared.utilities import hash_dict
        except ImportError:
            try:
                from shared.utilities import hash_dict
            except ImportError:
                import hashlib as _hl
                import json as _json
                def hash_dict(d: dict) -> str:
                    return _hl.sha256(
                        _json.dumps(d, sort_keys=True).encode()
                    ).hexdigest()

# ── ensure_dir: same fallback chain (minus hash_lock which doesn't have it)
try:
    from hypatiax.shared.utilities import ensure_dir
except ImportError:
    try:
        from shared.utilities import ensure_dir
    except ImportError:
        import pathlib as _pl
        def ensure_dir(path) -> None:
            _pl.Path(path).mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────

RESULTS_DIR = os.environ.get(
    "HYPATIAX_RESULTS",
    "hypatiax/data/results"
)

def run_protocol(config, runner):
    ensure_dir(RESULTS_DIR)

    h = hash_dict(config)
    lock = os.path.join(RESULTS_DIR, f".lock_{h}")

    if os.path.exists(lock):
        print(f"Skipping {config['name']} (cached)")
        return {"status": "skipped", "success": True}

    start = time.time()

    try:
        # Propagate DEFI_V3C_NO_TIMEOUT_FLAGS into config so the runner can
        # suppress --pysr-timeout / --method-timeout args when building its
        # subprocess command.  Injecting here rather than in each runner keeps
        # the check in one place.
        if os.environ.get("DEFI_V3C_NO_TIMEOUT_FLAGS"):
            config = {**config, "no_timeout_flags": True}

        results = runner(config)

        # 🔥 Detect failure explicitly
        if isinstance(results, dict) and results.get("status") in (
            "failed", "missing_script", "no_script"
        ):
            print(f"  ✗ {config['name']} failed — lock NOT written, will retry next run")

            return {
                "status": "failed",
                "success": False,
                "reason": results.get("status", "unknown"),
                "name": config["name"],
            }

        elapsed = time.time() - start

        # ✅ Write lock ONLY on success
        with open(lock, "w") as f:
            f.write("locked")

        out = os.path.join(RESULTS_DIR, f"{config['name']}_{h[:8]}.json")

        with open(out, "w") as f:
            json.dump({
                "config": config,
                "results": results,
                "meta": {
                    "success": True,
                    "elapsed": elapsed
                }
            }, f, indent=2)

        return {
            "status": "success",
            "success": True,
            "elapsed": elapsed
        }

    except Exception as e:
        print(f"[ERROR] {config['name']} crashed: {e}")

        return {
            "status": "crashed",
            "success": False,
            "error": str(e),
            "name": config["name"],
        }


if __name__ == "__main__":
    pass

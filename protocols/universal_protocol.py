from shared.utilities import hash_dict, ensure_dir
import json, os, time

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
        results = runner(config)

        # 🔥 Detect failure explicitly
        if isinstance(results, dict) and results.get("status") in (
            "failed", "missing_script", "no_script"
        ):
            print(f"  ✗ {config['name']} failed — lock NOT written, will retry next run")

            # Return failure dict so run_all --continue-on-fail can handle it
            # gracefully instead of killing the entire pipeline process.
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

        # Return crash dict — caller (run_all) decides whether to abort or
        # continue based on --continue-on-fail flag.
        return {
            "status": "crashed",
            "success": False,
            "error": str(e),
            "name": config["name"],
        }

  
if __name__ == "__main__":
    pass

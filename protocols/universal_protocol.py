
from shared.utilities import hash_dict, ensure_dir
import json, os

def run_protocol(config, runner):
    ensure_dir("results")
    h = hash_dict(config)
    lock = f"results/.lock_{h}"

    if os.path.exists(lock):
        print(f"Skipping {config['name']} (cached)")
        return {"status": "skipped"}

    with open(lock, "w") as f:
        f.write("locked")

    results = runner(config)

    out = f"results/{config['name']}_{h[:8]}.json"
    with open(out, "w") as f:
        json.dump({"config": config, "results": results}, f, indent=2)

    return results

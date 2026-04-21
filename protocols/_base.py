import json, os, hashlib, time

RESULTS_DIR = os.environ.get("HYPATIAX_RESULTS", "hypatiax/data/results")

def hash_config(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()

def save_results(results, config):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    h = hash_config(config)
    path = os.path.join(RESULTS_DIR, f"{config['name']}_{h[:8]}.json")
    with open(path, "w") as f:
        json.dump({"config": config, "results": results}, f, indent=2)
    return path

def reproducibility_lock(config):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    h = hash_config(config)
    lock_file = os.path.join(RESULTS_DIR, f".lock_{h}")
    if os.path.exists(lock_file):
        print("Skipping (already run):", config["name"])
        return False
    with open(lock_file, "w") as f:
        f.write(str(time.time()))
    return True

if __name__ == "__main__":
    pass

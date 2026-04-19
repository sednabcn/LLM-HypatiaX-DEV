
import json, os, hashlib, time

def hash_config(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()

def save_results(results, config):
    os.makedirs("results", exist_ok=True)
    h = hash_config(config)
    path = f"results/{config['name']}_{h[:8]}.json"
    with open(path, "w") as f:
        json.dump({"config": config, "results": results}, f, indent=2)
    return path

def reproducibility_lock(config):
    h = hash_config(config)
    lock_file = f"results/.lock_{h}"
    if os.path.exists(lock_file):
        print("Skipping (already run):", config["name"])
        return False
    with open(lock_file, "w") as f:
        f.write(str(time.time()))
    return True

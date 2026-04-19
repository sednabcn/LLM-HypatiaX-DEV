
import json, hashlib, os

def hash_dict(d):
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

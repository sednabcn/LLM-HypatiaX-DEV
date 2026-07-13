
import json
import os

root = "hypatiax/data/results"

print("Scanning for duplicate keys...")

# simple sanity check
for path, _, files in os.walk(root):
    for f in files:
        if f.endswith(".json"):
            try:
                json.load(open(os.path.join(path,f)))
            except Exception:
                print("Corrupt:", f)

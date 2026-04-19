
import os, json

root = "hypatiax/data/results"

print("Scanning for duplicate keys...")

# simple sanity check
for path, _, files in os.walk(root):
    for f in files:
        if f.endswith(".json"):
            try:
                json.load(open(os.path.join(path,f)))
            except:
                print("Corrupt:", f)

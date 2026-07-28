#!/usr/bin/env python3
"""
.github/scripts/dedup_requirements.py

Removes duplicate package pins from requirements.txt.
When the same package appears more than once (e.g. GitPython==3.1.46
and gitpython==3.1.47), keeps the last occurrence and comments out
the earlier conflicting lines.
"""

import pathlib
import re
import sys

req_path = pathlib.Path("requirements.txt")
if not req_path.exists():
    print("requirements.txt not found — skipped")
    sys.exit(0)

lines = req_path.read_text().splitlines()
seen = {}       # pkg_name_lower -> (line_index, original_line)
output = []

for line in lines:
    stripped = line.strip()

    # Pass through blanks and comments unchanged
    if not stripped or stripped.startswith("#"):
        output.append(line)
        continue

    # Extract bare package name (strip version specifiers, extras, markers)
    name = re.split(r"[><=!;\[]", stripped)[0].strip().lower()

    if name in seen:
        prev_idx, prev_line = seen[name]
        print(f"  DEDUP: keeping  '{stripped}'")
        print(f"         dropping '{prev_line}' (line {prev_idx + 1})")
        output[prev_idx] = f"# (deduped — superseded by later pin) {prev_line}"

    seen[name] = (len(output), stripped)
    output.append(line)

req_path.write_text("\n".join(output) + "\n")
print(f"requirements.txt: {len(lines)} lines in, {len(output)} lines out")

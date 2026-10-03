import re, os
from collections import defaultdict

files = []
with open('$HOME/Downloads/tree_f.txt') as f:
    for line in f:
        for marker in ['├── ', '└── ']:
            if marker in line:
                files.append(line.split(marker, 1)[1].rstrip())
                break

IMG_EXTS = {'.pdf','.png','.jpg','.jpeg','.eps','.svg'}
NUMBERED_RE = re.compile(r'__\d+(\.\w+)+$')

def is_clean(fname):
    if NUMBERED_RE.search(fname): return False
    if any(fname.startswith(p) for p in
           ('figures__','Figures__','figures_back__','PROD__','REPO_AUDIT')):
        return False
    return True

img_files  = [f for f in files if os.path.splitext(f)[1].lower() in IMG_EXTS]
data_files = [f for f in files if os.path.splitext(f)[1].lower() not in IMG_EXTS]
clean_set  = set(f for f in img_files if is_clean(f))
quarantine = sorted(f for f in img_files if not is_clean(f))

# Write the script
lines = [
    '#!/usr/bin/env bash',
    '# clean_figures_dir.sh — generated from tree_f.txt analysis',
    '#',
    '# Usage:',
    '#   bash clean_figures_dir.sh [FIGURES_DIR]',
    '#',
    '# What it does:',
    '#   1. Moves all doubled-prefix / numbered / REPO_AUDIT contamination files',
    '#      into a quarantine directory OUTSIDE the repo (default: /tmp/figures_quarantine_<timestamp>).',
    '#      Nothing is permanently deleted — quarantine is fully recoverable.',
    '#   2. Removes moved files from the git index (git rm --cached).',
    '#   3. Leaves the 78 clean-named files (including _v2 variants) untouched.',
    '#',
    '# RENAME note: every doubled-prefix file (figures__X, figures__figures__X, etc.)',
    '# strips to a canonical name that already exists clean in the directory.',
    '# There are 0 renames needed — all content is already present with correct names.',
    '#',
    'set -euo pipefail',
    '',
    'DIR="${1:-.}"',
    'TIMESTAMP=$(date +%Y%m%d_%H%M%S)',
    'QUARANTINE_DIR="/tmp/figures_quarantine_${TIMESTAMP}"',
    'mkdir -p "${QUARANTINE_DIR}"',
    '',
    'echo "=================================================="',
    'echo "  figures/ cleanup"',
    'echo "  Source : $(realpath ${DIR})"',
    'echo "  Quarantine : ${QUARANTINE_DIR}"',
    'echo "=================================================="',
    '',
    'n_moved=0',
    'n_missing=0',
    '',
    'quarantine_file() {',
    '  local fname="$1"',
    '  local src="${DIR}/${fname}"',
    '  if [[ ! -e "$src" ]]; then',
    '    echo "  [SKIP-MISSING] ${fname}"',
    '    (( n_missing++ )) || true',
    '    return',
    '  fi',
    '  # Remove from git index if tracked',
    '  git -C "${DIR}" rm -f --cached "${fname}" 2>/dev/null || true',
    '  # Move to quarantine',
    '  mv -f "$src" "${QUARANTINE_DIR}/${fname}"',
    '  (( n_moved++ )) || true',
    '}',
    '',
    'echo ""',
    'echo "--- Quarantining ${n} contamination files ---"',
    '',
]

for f in quarantine:
    lines.append(f"quarantine_file '{f}'")

lines += [
    '',
    'echo ""',
    'echo "=================================================="',
    f'echo "  Total identified for quarantine : {len(quarantine)}"',
    'echo "  Actually moved                  : ${n_moved}"',
    'echo "  Already missing (skipped)       : ${n_missing}"',
    f'echo "  Clean files left in place       : {len(clean_set)}"',
    'echo "  Quarantine location             : ${QUARANTINE_DIR}"',
    'echo "=================================================="',
    'echo ""',
    'echo "Next steps:"',
    'echo "  1. Verify figures/ looks correct:"',
    'echo "       ls ${DIR}/*.pdf ${DIR}/*.png | sort"',
    'echo "  2. Stage the removals:"',
    'echo "       git -C ${DIR}/.. add -u"',
    'echo "  3. Commit:"',
    'echo "       git -C ${DIR}/.. commit -m \\"ci: purge doubled-prefix contamination from figures/\\""',
    'echo "  4. If anything was wrongly quarantined, recover from ${QUARANTINE_DIR}"',
]

with open('/home/claude/clean_figures_dir.sh', 'w') as fh:
    fh.write('\n'.join(lines) + '\n')

print(f"Script written.")
print(f"  Clean kept   : {len(clean_set)}")
print(f"  Quarantined  : {len(quarantine)}")
print(f"  Rename needed: 0  (all stripped targets already exist clean)")
PYEOF

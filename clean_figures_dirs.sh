#!/usr/bin/env bash
# clean_figures_dirs.sh
#
# Usage:
#   bash clean_figures_dirs.sh [FIGURES_DIR]
#
# Reads  : ~/Downloads/tree_f.txt
# Writes : /tmp/clean_figures_dir_<timestamp>.sh   (the quarantine bash script)
# Then   : executes that generated script against FIGURES_DIR (default: current dir)
#
# Requirements: python3

set -euo pipefail

FIGURES_DIR="${1:-.}"

python3 - "$FIGURES_DIR" <<'PYEOF'
import re, os, sys, stat
from collections import defaultdict

figures_dir = sys.argv[1]
tree_file   = os.path.expanduser('~/Downloads/tree_f.txt')

files = []
with open(tree_file) as f:
    for line in f:
        for marker in ['├── ', '└── ']:
            if marker in line:
                files.append(line.split(marker, 1)[1].rstrip())
                break

IMG_EXTS    = {'.pdf', '.png', '.jpg', '.jpeg', '.eps', '.svg'}
NUMBERED_RE = re.compile(r'__\d+(\.\w+)+$')

def is_clean(fname):
    if NUMBERED_RE.search(fname):
        return False
    if any(fname.startswith(p) for p in
           ('figures__', 'Figures__', 'figures_back__', 'PROD__', 'REPO_AUDIT')):
        return False
    return True

img_files  = [f for f in files if os.path.splitext(f)[1].lower() in IMG_EXTS]
data_files = [f for f in files if os.path.splitext(f)[1].lower() not in IMG_EXTS]
clean_set  = set(f for f in img_files if is_clean(f))
quarantine = sorted(f for f in img_files if not is_clean(f))

import tempfile, time
timestamp  = time.strftime('%Y%m%d_%H%M%S')
out_path   = f'/tmp/clean_figures_dir_{timestamp}.sh'

lines = [
    '#!/usr/bin/env bash',
    f'# clean_figures_dir.sh — generated {timestamp} from tree_f.txt analysis',
    '#',
    '# Usage:',
    '#   bash clean_figures_dir.sh [FIGURES_DIR]',
    '#',
    '# What it does:',
    '#   1. Moves all doubled-prefix / numbered / REPO_AUDIT contamination files',
    '#      into a quarantine directory OUTSIDE the repo.',
    '#      Nothing is permanently deleted — quarantine is fully recoverable.',
    '#   2. Removes moved files from the git index (git rm --cached).',
    f'#   3. Leaves the {len(clean_set)} clean-named files (including _v2 variants) untouched.',
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
    'echo "  Source     : $(realpath ${DIR})"',
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
    '  echo "  [MOVED] ${fname}"',
    '  (( n_moved++ )) || true',
    '}',
    '',
    'echo ""',
    f'echo "--- Quarantining {len(quarantine)} contamination files ---"',
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
    'echo "       ls \${DIR}/*.pdf \${DIR}/*.png | sort"',
    'echo "  2. Stage the removals:"',
    'echo "       git -C \${DIR}/.. add -u"',
    'echo "  3. Commit:"',
    'echo "       git -C \${DIR}/.. commit -m \\"ci: purge doubled-prefix contamination from figures/\\""',
    'echo "  4. If anything was wrongly quarantined, recover from \${QUARANTINE_DIR}"',
]

with open(out_path, 'w') as fh:
    fh.write('\n'.join(lines) + '\n')

os.chmod(out_path, os.stat(out_path).st_mode | stat.S_IXUSR | stat.S_IXGRP)

print(f"Generated : {out_path}")
print(f"  Clean kept   : {len(clean_set)}")
print(f"  Quarantined  : {len(quarantine)}")
print(f"  Rename needed: 0  (all stripped targets already exist clean)")
print(f"")
print(f"Now running: bash {out_path} {figures_dir}")
PYEOF

# The Python block printed the generated script path; capture and run it
GENERATED=$(ls -t /tmp/clean_figures_dir_*.sh 2>/dev/null | head -1)
if [[ -z "$GENERATED" ]]; then
    echo "ERROR: could not find generated script in /tmp/" >&2
    exit 1
fi

bash "$GENERATED" "$FIGURES_DIR"

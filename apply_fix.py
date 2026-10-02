import sys
p = sys.argv[1]
s = open(p).read()
if "_det_key" in s:
    sys.exit("already patched")
helper = '''

def _det_key(p: Path) -> tuple[str, str]:
    """Deterministic 'age' key for result files: basename, then full path.
    Replaces os.path.getmtime (which differs between a git checkout and a
    working tree). Shard names carry a YYYYMMDD_HHMMSS stamp, so name order is
    chronological order on every machine. Use reverse=True for newest-first."""
    return (p.name, str(p))
'''
anchor = "\n\n# Basename fragment of the PCA-variant output files"
assert s.count(anchor) == 1, "anchor not found"
s = s.replace(anchor, helper.rstrip("\n") + "\n" + anchor, 1)
reps = [
 ("            key=os.path.getmtime, reverse=True)\n        if candidates:", "            key=_det_key, reverse=True)\n        if candidates:"),
 ("    files = sorted({f.resolve(): f for f in files}.values(), key=os.path.getmtime)", "    files = sorted({f.resolve(): f for f in files}.values(), key=_det_key)"),
 ("candidates = sorted(_filtered_glob(sweep_dir, glob_pat), key=os.path.getmtime, reverse=True)", "candidates = sorted(_filtered_glob(sweep_dir, glob_pat), key=_det_key, reverse=True)"),
 ("candidates = sorted(_filtered_glob(alt_dir, glob_pat), key=os.path.getmtime, reverse=True)", "candidates = sorted(_filtered_glob(alt_dir, glob_pat), key=_det_key, reverse=True)"),
 ("d = max(fs, key=os.path.getmtime)", "d = max(fs, key=_det_key)"),
 ("key=lambda p: p.stat().st_mtime, reverse=True)]:", "key=_det_key, reverse=True)]:"),
]
for a, b in reps:
    assert s.count(a) == 1, "pattern not found exactly once: " + a
    s = s.replace(a, b)
open(p, "w").write(s)
print("patched", p)

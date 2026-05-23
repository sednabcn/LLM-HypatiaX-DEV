#!/usr/bin/env python3
"""
generate_figures.py — Invoke the HypatiaX visualisation pipeline.

Usage (called by ci_postprocess.yml):
  python figures/generate_figures.py \\
      --repo-root   hypatiax/data/results/<subdir> \\
      --figures-dir hypatiax/data/results/<subdir>/figures \\
      --source auto

All arguments are forwarded verbatim to plot_results.py so that
figures land in the directory ci_postprocess.yml stages and commits.

FIX 4: the original script ignored sys.argv entirely and called
plot_results.py with no arguments, so figures were always written to
plot_results.py's hardcoded default path rather than the per-experiment
${SUB}/figures/ directory.  The subsequent 'git add .../figures/' step
therefore staged nothing and no figures were ever committed.
"""

import subprocess
import sys


def main():
    cmd = [
        sys.executable,
        "hypatiax/tools/visualizations/plot_results.py",
    ] + sys.argv[1:]   # forward --repo-root, --figures-dir, --source (and any future flags)

    print("Generating figures...")
    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(
            f"::error::plot_results.py exited with code {result.returncode}",
            file=sys.stderr,
        )
        sys.exit(result.returncode)

    print("Figures generated successfully.")


if __name__ == "__main__":
    main()

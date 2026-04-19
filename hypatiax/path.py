from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

def ensure_project_root_on_path():
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

"""
HypatiaX package initialization.

Ensures project root is on sys.path so that all scripts
can be executed directly without import errors.
"""

__all__ = []

from .path import ensure_project_root_on_path
ensure_project_root_on_path()

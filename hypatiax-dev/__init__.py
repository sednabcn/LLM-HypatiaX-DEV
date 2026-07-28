"""
hypatiax/__init__.py
HypatiaX — Hybrid Symbolic-Neural Framework for Extrapolation-Reliable
Analytical Discovery.

Fixes applied (CI lint pass):
  I001  L10 — import block unsorted → sorted alphabetically
"""
# Standard-library imports first, then third-party, then local — all sorted.
__all__ = []
HypatiaX = None  # type: ignore  # noqa: F401

try:
    from hypatiax.version import __version__  # noqa: F401
except Exception:
    __version__ = "3.0.0"  # fallback until hypatiax/version.py is present

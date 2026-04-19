"""
setup.py — legacy shim for HypatiaX.

All project metadata lives in pyproject.toml.
This file exists solely so that:
  - `pip install -e .` works with older pip (<21.3)
  - tools that call `python setup.py --version` still function

Do NOT add metadata here; edit pyproject.toml instead.
"""
from setuptools import setup

if __name__ == "__main__":
    setup()

"""
HypatiaX Core Module
====================

Core functionality for the HypatiaX NER system.
"""

__version__ = "1.0.0"
__author__ = "HypatiaX Team"

__all__ = []

# Optional module
try:
    from . import preprocessing
    __all__.append("preprocessing")
except ImportError:
    preprocessing = None


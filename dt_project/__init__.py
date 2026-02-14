"""
dt_project - Compatibility shim package.

Re-exports from the refactored `backend/` package so that legacy test
imports continue to work without rewriting 33+ test files.
"""

from . import healthcare  # noqa: F401

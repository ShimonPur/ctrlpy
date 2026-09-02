"""Symbolic and pedagogical control systems analysis engine."""

from __future__ import annotations

try:
    import sympy  # noqa: F401
except ImportError:
    raise ImportError(
        "The 'ctrlpy.symbolic' submodule requires SymPy.\n"
        "Please install it with:\n"
        '    pip install "ctrlpy[symbolic]@git+https://github.com/ShimonPur/ctrlpy.git"\n'
        "or with uv:\n"
        "    uv add git+https://github.com/ShimonPur/ctrlpy.git --extra symbolic"
    ) from None

from ctrlpy.symbolic.root_locus import (
    RootLocusRules,
    RootLocusRulesResult,
    root_locus_rules,
)
from ctrlpy.symbolic.routh import RouthArray, RouthResult, routh_array, routh_table

__all__ = [
    "RootLocusRules",
    "RootLocusRulesResult",
    "RouthArray",
    "RouthResult",
    "root_locus_rules",
    "routh_array",
    "routh_table",
]

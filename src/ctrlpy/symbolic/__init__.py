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
from ctrlpy.symbolic.state_space import (
    CanonicalFormResult,
    ModeAnalysis,
    StateSpaceTutor,
    controllability_matrix,
    controllable_canonical_form,
    jordan_canonical_form,
    observability_matrix,
    observable_canonical_form,
    state_space_tutor,
)

__all__ = [
    "CanonicalFormResult",
    "ModeAnalysis",
    "RootLocusRules",
    "RootLocusRulesResult",
    "RouthArray",
    "RouthResult",
    "StateSpaceTutor",
    "controllability_matrix",
    "controllable_canonical_form",
    "jordan_canonical_form",
    "observability_matrix",
    "observable_canonical_form",
    "root_locus_rules",
    "routh_array",
    "routh_table",
    "state_space_tutor",
]

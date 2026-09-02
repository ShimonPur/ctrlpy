"""Pedagogical and symbolic tools for control systems analysis."""

try:
    import sympy  # noqa: F401
except ImportError:
    raise ImportError(
        "The 'ctrlpy.pedagogy' submodule requires SymPy.\n"
        "Please install it with:\n"
        '    pip install "ctrlpy[symbolic]@git+https://github.com/ShimonPur/ctrlpy.git"\n'
        "or with uv:\n"
        "    uv add git+https://github.com/ShimonPur/ctrlpy.git --extra symbolic"
    ) from None

from ctrlpy.pedagogy.root_locus_rules import RootLocusRulesResult, root_locus_rules
from ctrlpy.pedagogy.routh import RouthArray, RouthResult, routh_array, routh_table
from ctrlpy.pedagogy.state_space import (
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
from ctrlpy.pedagogy.steady_state import SteadyStateResult, steady_state_analysis

__all__ = [
    "CanonicalFormResult",
    "ModeAnalysis",
    "RootLocusRulesResult",
    "RouthArray",
    "RouthResult",
    "StateSpaceTutor",
    "SteadyStateResult",
    "controllability_matrix",
    "controllable_canonical_form",
    "jordan_canonical_form",
    "observability_matrix",
    "observable_canonical_form",
    "root_locus_rules",
    "routh_array",
    "routh_table",
    "state_space_tutor",
    "steady_state_analysis",
]

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
from ctrlpy.pedagogy.routh import RouthResult, routh_table
from ctrlpy.pedagogy.steady_state import SteadyStateResult, steady_state_analysis

__all__ = [
    "RootLocusRulesResult",
    "RouthResult",
    "SteadyStateResult",
    "root_locus_rules",
    "routh_table",
    "steady_state_analysis",
]

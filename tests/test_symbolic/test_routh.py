"""Comprehensive test suite for RouthArray and pedagogical Routh-Hurwitz tools."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

import ctrlpy as cp
from ctrlpy.symbolic.routh import (
    RouthArray,
    RouthResult,
    routh_array,
    routh_table,
)


class TestRouthArrayStandard:
    """Test suite for standard numeric polynomials and basic RouthArray operations."""

    def test_standard_stable_polynomial(self) -> None:
        """P(s) = s^3 + 2s^2 + 3s + 4."""
        ra = RouthArray([1, 2, 3, 4])
        assert isinstance(ra, RouthArray)
        assert ra.is_stable is True
        assert ra.num_rhp_poles == 0
        assert len(ra.table) == 4
        # s^3: [1, 3]
        # s^2: [2, 4]
        # s^1: [1, 0]
        # s^0: [4, 0]
        assert float(ra.table[0][0]) == 1.0
        assert float(ra.table[1][0]) == 2.0
        assert float(ra.table[2][0]) == 1.0
        assert float(ra.table[3][0]) == 4.0

    def test_standard_unstable_polynomial(self) -> None:
        """P(s) = s^3 + s^2 + 2s + 24 -> 2 RHP poles."""
        ra = routh_array([1, 1, 2, 24])
        assert ra.is_stable is False
        assert ra.num_rhp_poles == 2
        assert float(ra.table[2][0]) == -22.0

    def test_transfer_function_input(self) -> None:
        """Test RouthArray initialization from TransferFunction."""
        G = cp.tf([1], [1, 3, 3, 1])
        ra = routh_table(G)
        assert ra.is_stable is True
        assert ra.num_rhp_poles == 0

    def test_numpy_array_input(self) -> None:
        """Test RouthArray initialization with NumPy array."""
        arr = np.array([1.0, 4.0, 5.0, 2.0])
        ra = RouthArray(arr)
        assert ra.is_stable is True
        assert ra.num_rhp_poles == 0

    def test_sympy_expression_and_poly_input(self) -> None:
        """Test initialization with SymPy expressions and sp.Poly."""
        s = sp.Symbol("s")
        expr = s**4 + 2 * s**3 + 3 * s**2 + 4 * s + 5
        ra1 = RouthArray(expr)
        assert ra1.num_rhp_poles == 2
        assert ra1.is_stable is False

        poly = sp.Poly(expr, s)
        ra2 = RouthArray(poly)
        assert ra2.num_rhp_poles == 2
        assert ra2.is_stable is False

    def test_degree_zero_polynomial(self) -> None:
        """Test edge case of degree 0 constant polynomial."""
        ra = RouthArray([5])
        assert ra.is_stable is True
        assert ra.num_rhp_poles == 0
        assert len(ra.table) == 1
        assert ra.table[0][0] == 5

    def test_invalid_inputs(self) -> None:
        """Test error handling for invalid input types or empty sequences."""
        with pytest.raises(TypeError):
            RouthArray("invalid_string_not_expr")

        with pytest.raises(ValueError):
            RouthArray([])


class TestRouthArraySpecialCases:
    """Test edge cases: first-column zero substitution and rows of all zeros."""

    def test_special_case_1_epsilon_substitution(self) -> None:
        """P(s) = s^5 + 2s^4 + 2s^3 + 4s^2 + 11s + 10.

        Row s^3 has leading element 0, replaced by epsilon > 0 -> 2 sign changes.
        """
        ra = RouthArray([1, 2, 2, 4, 11, 10])
        assert ra.num_rhp_poles == 2
        assert ra.is_stable is False
        assert any("epsilon" in s.lower() or "special case 1" in s.lower() for s in ra.steps)

    def test_special_case_2_row_of_zeros_auxiliary(self) -> None:
        """P(s) = s^3 + 2s^2 + 4s + 8 = (s+2)(s^2+4).

        Roots at s = -2, ±2j (marginally stable / imaginary axis roots).
        """
        ra = RouthArray([1, 2, 4, 8])
        assert ra.is_stable is False
        assert ra.num_rhp_poles == 0
        assert len(ra.auxiliary_polynomials) >= 1
        assert any("all zeros" in s.lower() or "auxiliary" in s.lower() for s in ra.steps)

    def test_special_case_2_fourth_order(self) -> None:
        """P(s) = s^4 + 2s^3 + 11s^2 + 18s + 18 = (s^2 + 9)(s^2 + 2s + 2)."""
        ra = RouthArray([1, 2, 11, 18, 18])
        assert ra.is_stable is False
        assert len(ra.auxiliary_polynomials) >= 1


class TestRouthArrayParametricAndOutputs:
    """Test parametric stability interval solving and pedagogical outputs."""

    def test_parametric_k_range(self) -> None:
        """P(s) = s^3 + 3s^2 + 3s + 1 + K -> Stable for -1 < K < 8."""
        s, K = sp.symbols("s K")
        expr = s**3 + 3 * s**2 + 3 * s + 1 + K
        ra = RouthArray(expr, k_symbol="K")
        assert ra.k_range is not None
        assert "-1" in ra.k_range and "8" in ra.k_range

    def test_parametric_transfer_function(self) -> None:
        """G(s) = 1 / (s(s+1)(s+2)) = 1 / (s^3 + 3s^2 + 2s).

        Closed loop: s^3 + 3s^2 + 2s + K -> 0 < K < 6.
        """
        G = cp.tf([1], [1, 3, 2, 0])
        ra = RouthArray(G, k_symbol="K")
        assert ra.k_range is not None
        assert "0 < K < 6" in ra.k_range

    def test_explain_steps(self) -> None:
        """Verify explain_steps returns a detailed list of pedagogical derivation notes."""
        ra = RouthArray([1, 2, 3, 4])
        steps = ra.explain_steps()
        assert isinstance(steps, list)
        assert len(steps) >= 4
        # Contains characteristic equation, row initialization, row computation, and conclusion
        full_text = "\n".join(steps)
        assert "Characteristic Equation" in full_text
        assert "Row $s^3$ Initialization" in full_text
        assert "Stability Conclusion" in full_text

    def test_repr_latex_and_markdown(self) -> None:
        """Verify LaTeX and Markdown rich formatting."""
        ra = RouthArray([1, 2, 3, 4])
        latex_str = ra._repr_latex_()
        assert r"\begin{aligned}" in latex_str
        assert r"\begin{array}" in latex_str
        assert "Strictly Stable" in latex_str

        md_str = ra._repr_markdown_()
        assert md_str.startswith("$$")
        assert md_str.endswith("$$")

    def test_str_ascii_table(self) -> None:
        """Verify ASCII string output."""
        ra = RouthArray([1, 2, 3, 4])
        s_out = str(ra)
        assert "Routh-Hurwitz Stability Criterion" in s_out
        assert "s^3" in s_out
        assert "s^2" in s_out
        assert "Asymptotically Stable: True" in s_out

    def test_backward_compatibility_aliases(self) -> None:
        """Verify RouthResult alias and routh_table / routh_array function helpers."""
        assert RouthResult is RouthArray
        r1 = routh_table([1, 2, 1])
        r2 = routh_array([1, 2, 1])
        assert isinstance(r1, RouthArray)
        assert isinstance(r2, RouthArray)
        assert r1.is_stable is True
        assert r2.is_stable is True

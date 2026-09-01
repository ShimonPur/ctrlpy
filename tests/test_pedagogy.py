"""Unit tests for ctrlpy.pedagogy submodule."""

from __future__ import annotations

import math

import pytest
import sympy as sp

import ctrlpy as cp
from ctrlpy.pedagogy import (
    RootLocusRulesResult,
    RouthResult,
    SteadyStateResult,
    root_locus_rules,
    routh_table,
    steady_state_analysis,
)


class TestRouthHurwitz:
    """Test suite for Routh-Hurwitz stability criterion."""

    def test_standard_stable_polynomial(self) -> None:
        # P(s) = s^3 + 2s^2 + 3s + 4
        res = routh_table([1, 2, 3, 4])
        assert isinstance(res, RouthResult)
        assert res.is_stable is True
        assert res.num_rhp_poles == 0
        assert len(res.table) == 4
        # s^3: [1, 3]
        # s^2: [2, 4]
        # s^1: [1, 0]
        # s^0: [4, 0]
        assert float(res.table[0][0]) == 1.0
        assert float(res.table[1][0]) == 2.0
        assert float(res.table[2][0]) == 1.0
        assert float(res.table[3][0]) == 4.0

    def test_standard_unstable_polynomial(self) -> None:
        # P(s) = s^3 + s^2 + 2s + 24
        res = routh_table([1, 1, 2, 24])
        assert res.is_stable is False
        assert res.num_rhp_poles == 2
        # s^1 entry is (1*2 - 1*24)/1 = -22 -> 2 sign changes (+, +, -, +)
        assert float(res.table[2][0]) == -22.0

    def test_transfer_function_input(self) -> None:
        G = cp.tf([1], [1, 3, 3, 1])
        res = routh_table(G)
        assert res.is_stable is True
        assert res.num_rhp_poles == 0

    def test_symbolic_expression_input(self) -> None:
        s = sp.Symbol("s")
        expr = s**4 + 2 * s**3 + 3 * s**2 + 4 * s + 5
        res = routh_table(expr)
        assert isinstance(res, RouthResult)
        assert res.num_rhp_poles == 2
        assert res.is_stable is False

    def test_special_case_1_epsilon_substitution(self) -> None:
        # P(s) = s^5 + 2s^4 + 2s^3 + 4s^2 + 11s + 10
        # Row s^3 has first element 0, but second element is 6
        res = routh_table([1, 2, 2, 4, 11, 10])
        assert res.num_rhp_poles == 2
        assert res.is_stable is False
        assert any("epsilon" in s.lower() or "special case 1" in s.lower() for s in res.steps)

    def test_special_case_2_row_of_zeros_auxiliary(self) -> None:
        # P(s) = s^3 + 2s^2 + 4s + 8 = (s+2)(s^2+4)
        # Roots at -2, +2j, -2j -> marginally stable
        res = routh_table([1, 2, 4, 8])
        assert res.is_stable is False
        assert res.num_rhp_poles == 0
        assert len(res.auxiliary_polynomials) >= 1
        assert any("all zeros" in s.lower() or "auxiliary" in s.lower() for s in res.steps)

    def test_special_case_2_fourth_order(self) -> None:
        # P(s) = s^4 + 2s^3 + 11s^2 + 18s + 18
        # (s^2 + 9)(s^2 + 2s + 2)
        res = routh_table([1, 2, 11, 18, 18])
        assert res.is_stable is False
        assert len(res.auxiliary_polynomials) >= 1

    def test_parametric_k_range(self) -> None:
        s, K = sp.symbols("s K")
        # P(s) = s^3 + 3s^2 + 3s + 1 + K
        # Stable for -1 < K < 8
        expr = s**3 + 3 * s**2 + 3 * s + 1 + K
        res = routh_table(expr, k_symbol="K")
        assert res.k_range is not None
        assert "-1" in res.k_range and "8" in res.k_range

    def test_routh_latex_and_str(self) -> None:
        res = routh_table([1, 2, 3, 4])
        latex_output = res._repr_latex_()
        assert r"\begin{array}" in latex_output
        assert "Strictly Stable" in latex_output
        md_output = res._repr_markdown_()
        assert "$$" in md_output
        assert len(str(res)) > 0

        str_output = str(res)
        assert "Routh-Hurwitz" in str_output
        assert "Number of RHP Poles: 0" in str_output

    def test_degree_0_and_1(self) -> None:
        res0 = routh_table([5])
        assert res0.is_stable is True
        assert res0.num_rhp_poles == 0

        res1 = routh_table([2, 4])
        assert res1.is_stable is True
        assert res1.num_rhp_poles == 0


class TestRootLocusRules:
    """Test suite for analytical Root Locus rules."""

    def test_third_order_system_rules(self) -> None:
        # G(s) = 1 / (s(s+1)(s+2)) = 1 / (s^3 + 3s^2 + 2s)
        G = cp.tf([1], [1, 3, 2, 0])
        res = root_locus_rules(G)

        assert isinstance(res, RootLocusRulesResult)
        assert res.num_poles == 3
        assert res.num_zeros == 0
        assert res.num_branches == 3
        assert res.num_asymptotes == 3

        # Real axis segments: [0, -1] and [-2, -inf]
        assert len(res.real_axis_segments) == 2
        assert res.real_axis_segments[0][1] == 0.0
        assert res.real_axis_segments[0][0] == -1.0
        assert res.real_axis_segments[1][1] == -2.0
        assert math.isinf(res.real_axis_segments[1][0])

        # Centroid: (0 - 1 - 2) / 3 = -1.0
        assert res.centroid is not None
        assert math.isclose(res.centroid, -1.0, abs_tol=1e-5)

        # Asymptote angles: 60, 180, -60 (or 300)
        assert len(res.asymptote_angles_deg) == 3
        assert any(math.isclose(a, 60.0, abs_tol=1.0) for a in res.asymptote_angles_deg)
        assert any(math.isclose(a, 180.0, abs_tol=1.0) for a in res.asymptote_angles_deg)

        # Breakaway point near -0.4226 (on locus)
        assert len(res.breakaway_points) >= 1
        bp = res.breakaway_points[0]
        assert math.isclose(bp["s"], -0.4226, abs_tol=1e-2)
        assert bp["type"] == "breakaway"

        # Imaginary axis crossing: s = ±j*sqrt(2) ≈ ±j1.414 at K_crit ≈ 6.0
        assert len(res.imag_axis_crossings) >= 1
        cross = res.imag_axis_crossings[0]
        assert math.isclose(cross["omega"], math.sqrt(2.0), abs_tol=0.1)
        assert math.isclose(cross["k"], 6.0, abs_tol=0.2)

    def test_complex_poles_departure_angle(self) -> None:
        # G(s) = (s + 2) / (s^2 + 2s + 2)
        # Poles at -1 ± 1j, zero at -2
        G = cp.tf([1, 2], [1, 2, 2])
        res = root_locus_rules(G)

        assert res.num_poles == 2
        assert res.num_zeros == 1
        assert res.num_asymptotes == 1

        # Departure angle from -1 + 1j:
        # theta_dep = 180 - (angle to conjugate pole 90 deg) + (angle to zero at -2: 45 deg) = 135 deg
        assert len(res.departure_angles) >= 1
        dep_angles = list(res.departure_angles.values())
        assert any(math.isclose(abs(a), 135.0, abs_tol=2.0) for a in dep_angles)

    def test_root_locus_latex_and_str(self) -> None:
        G = cp.tf([1], [1, 3, 2, 0])
        res = root_locus_rules(G)
        latex_str = res._repr_latex_()
        assert r"\textbf{Rule 1" in latex_str
        assert r"\sigma_a" in latex_str
        assert "$$" in res._repr_markdown_()

        text_str = str(res)
        assert "Analytical Root Locus Rules" in text_str
        assert "sigma_a = -1.0000" in text_str


class TestSteadyState:
    """Test suite for steady-state error analysis and static constants."""

    def test_type_0_system(self) -> None:
        # G(s) = 10 / (s^2 + 3s + 2) -> Kp = 10/2 = 5
        G = cp.tf([10], [1, 3, 2])
        res = steady_state_analysis(G)

        assert isinstance(res, SteadyStateResult)
        assert res.system_type == 0
        assert math.isclose(res.kp, 5.0, abs_tol=1e-5)
        assert res.kv == 0.0
        assert res.ka == 0.0

        # ess(step) = 1 / (1 + 5) = 1/6 ≈ 0.1667
        assert math.isclose(res.ess_step, 1.0 / 6.0, abs_tol=1e-4)
        assert math.isinf(res.ess_ramp)
        assert math.isinf(res.ess_parabolic)
        assert res.is_closed_loop_stable is True

    def test_type_1_system(self) -> None:
        # G(s) = 10 / (s^2 + 2s) = 10 / (s(s+2)) -> Kv = 10/2 = 5
        G = cp.tf([10], [1, 2, 0])
        res = steady_state_analysis(G)

        assert res.system_type == 1
        assert math.isinf(res.kp)
        assert math.isclose(res.kv, 5.0, abs_tol=1e-5)
        assert res.ka == 0.0

        assert res.ess_step == 0.0
        assert math.isclose(res.ess_ramp, 0.2, abs_tol=1e-4)
        assert math.isinf(res.ess_parabolic)

    def test_type_2_system(self) -> None:
        # G(s) = 10 / (s^3 + 2s^2) = 10 / (s^2(s+2)) -> Ka = 10/2 = 5
        G = cp.tf([10], [1, 2, 0, 0])
        res = steady_state_analysis(G)

        assert res.system_type == 2
        assert math.isinf(res.kp)
        assert math.isinf(res.kv)
        assert math.isclose(res.ka, 5.0, abs_tol=1e-5)

        assert res.ess_step == 0.0
        assert res.ess_ramp == 0.0
        assert math.isclose(res.ess_parabolic, 0.2, abs_tol=1e-4)

    def test_steady_state_latex_and_str(self) -> None:
        G = cp.tf([10], [1, 2, 0])
        res = steady_state_analysis(G)

        latex_str = res._repr_latex_()
        assert r"\textbf{System Classification:}" in latex_str
        assert "Type 1" in latex_str
        assert "K_v = 5" in latex_str
        assert "$$" in res._repr_markdown_()

        text_str = str(res)
        assert "Steady-State Error Analysis" in text_str
        assert "Type 1" in text_str

    def test_unstable_closed_loop_warning(self) -> None:
        # G(s) = 10 / (s - 1) -> closed loop pole at s = -1 + 10? No: 1 + G = (s-1+10)/(s-1) = (s+9)/(s-1)
        # Open loop pole in RHP: s=1
        G_unstable = cp.tf(
            [-10], [1, -2]
        )  # Closed loop: s - 2 - 10 = s - 12 -> pole at +12 (unstable)
        res = steady_state_analysis(G_unstable)
        assert res.is_closed_loop_stable is False
        assert "Unstable" in res._repr_latex_()
        assert "WARNING" in str(res)


class TestPedagogyEdgeCases:
    """Edge cases, type handling, and error validations."""

    def test_routh_invalid_inputs(self) -> None:
        with pytest.raises(TypeError):
            routh_table("invalid_string_not_expr")

        with pytest.raises(ValueError):
            routh_table([])

    def test_routh_tf_with_k_symbol(self) -> None:
        G = cp.tf([1], [1, 3, 2, 0])
        res = routh_table(G, k_symbol="K")
        assert res.k_range is not None
        assert "0 < K < 6" in res.k_range

    def test_root_locus_no_asymptotes(self) -> None:
        # G(s) = (s + 1) / (s + 2) -> n = 1, m = 1 -> n - m = 0
        G = cp.tf([1, 1], [1, 2])
        res = root_locus_rules(G)
        assert res.num_asymptotes == 0
        assert res.centroid is None
        assert "No asymptotes" in res._repr_latex_()

    def test_root_locus_complex_zeros_arrival(self) -> None:
        # G(s) = (s^2 + 2s + 2) / (s(s+1)(s+2))
        G = cp.tf([1, 2, 2], [1, 3, 2, 0])
        res = root_locus_rules(G)
        assert len(res.arrival_angles) >= 1
        assert "Rule 6" in res._repr_latex_()

    def test_root_locus_invalid_type(self) -> None:
        with pytest.raises(TypeError):
            root_locus_rules(12345)

    def test_steady_state_invalid_type(self) -> None:
        with pytest.raises(TypeError):
            steady_state_analysis("not_a_system")

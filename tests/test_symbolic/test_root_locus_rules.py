"""Comprehensive test suite for RootLocusRules analytical derivations."""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

import ctrlpy as cp
from ctrlpy.symbolic.root_locus import (
    RootLocusRules,
    RootLocusRulesResult,
    root_locus_rules,
)


class TestRootLocusRulesBasics:
    """Test standard 3rd order open-loop system: G(s) = 1 / (s(s+1)(s+2))."""

    def test_third_order_system(self) -> None:
        G = cp.tf([1], [1, 3, 2, 0])
        rlr = RootLocusRules(G)

        assert rlr.num_poles == 3
        assert rlr.num_zeros == 0
        assert rlr.num_branches == 3
        assert rlr.num_asymptotes == 3

        # Centroid sigma_a = (0 - 1 - 2) / 3 = -1.0
        assert rlr.centroid is not None
        assert abs(rlr.centroid - (-1.0)) < 1e-4

        # Asymptote angles: 60, 180, -60 (or equivalent)
        angles = sorted(rlr.asymptote_angles_deg)
        assert any(abs(a - 60.0) < 1.0 for a in angles)
        assert any(abs(a - 180.0) < 1.0 or abs(a - (-180.0)) < 1.0 for a in angles)
        assert any(abs(a - (-60.0)) < 1.0 for a in angles)

        # Real axis segments: [-1, 0] and [-inf, -2]
        assert len(rlr.real_axis_segments) == 2
        seg1, seg2 = rlr.real_axis_segments
        assert (abs(seg1[0] - (-1.0)) < 1e-3 and abs(seg1[1] - 0.0) < 1e-3) or (
            abs(seg2[0] - (-1.0)) < 1e-3 and abs(seg2[1] - 0.0) < 1e-3
        )
        assert math.isinf(seg1[0]) or math.isinf(seg2[0])

        # Breakaway point: s = -0.4226, K = 0.3849
        assert len(rlr.breakaway_points) == 1
        bp = rlr.breakaway_points[0]
        assert abs(bp["s"] - (-0.4226)) < 1e-2
        assert abs(bp["k"] - 0.3849) < 1e-2
        assert bp["type"] == "breakaway"

        # Imaginary axis crossing: omega = sqrt(2) approx 1.414, K_crit = 6.0
        assert len(rlr.imag_axis_crossings) == 1
        cross = rlr.imag_axis_crossings[0]
        assert abs(cross["omega"] - 1.4142) < 0.05
        assert abs(cross["k"] - 6.0) < 0.1


class TestRootLocusRulesWithZeros:
    """Test systems with finite open-loop zeros."""

    def test_system_with_finite_zero(self) -> None:
        """G(s) = (s + 3) / (s(s + 1)(s + 2)(s + 4))."""
        G = cp.tf([1, 3], np.poly([0, -1, -2, -4]))
        rlr = root_locus_rules(G)

        assert rlr.num_poles == 4
        assert rlr.num_zeros == 1
        assert rlr.num_branches == 4
        assert rlr.num_asymptotes == 3

        # Centroid: (0 - 1 - 2 - 4 - (-3)) / 3 = -4 / 3 approx -1.3333
        assert rlr.centroid is not None
        assert abs(rlr.centroid - (-4.0 / 3.0)) < 1e-3

    def test_system_equal_poles_and_zeros(self) -> None:
        """G(s) = (s + 2) / (s + 1) -> n = m = 1 -> 0 asymptotes to infinity."""
        G = cp.tf([1, 2], [1, 1])
        rlr = RootLocusRules(G)
        assert rlr.num_asymptotes == 0
        assert rlr.centroid is None


class TestRootLocusComplexPolesAndZeros:
    """Test complex pole departure angles and complex zero arrival angles."""

    def test_departure_angles(self) -> None:
        """G(s) = 1 / (s(s^2 + 2s + 2)) -> poles at 0, -1 + j, -1 - j."""
        G = cp.tf([1], [1, 2, 2, 0])
        rlr = RootLocusRules(G)

        assert len(rlr.departure_angles) == 2
        for p, deg in rlr.departure_angles.items():
            if p.imag > 0:
                # Pole at -1 + j: angle to 0 is 135 deg, angle to -1-j is 90 deg -> dep = 180 - (135 + 90) = -45 deg
                assert abs(deg - (-45.0)) < 1.0 or abs(deg - 315.0) < 1.0

    def test_arrival_angles(self) -> None:
        """G(s) = (s^2 + 2s + 2) / (s(s + 1)(s + 2)) -> zeros at -1 ± j."""
        G = cp.tf([1, 2, 2], [1, 3, 2, 0])
        rlr = RootLocusRules(G)
        assert len(rlr.arrival_angles) == 2


class TestRootLocusConstructorVariations:
    """Test different ways of instantiating RootLocusRules."""

    def test_explicit_num_den(self) -> None:
        rlr = RootLocusRules(num=[1], den=[1, 3, 2, 0])
        assert rlr.num_poles == 3

    def test_explicit_poles_zeros(self) -> None:
        rlr = RootLocusRules(poles=[0, -1, -2], zeros=[])
        assert rlr.num_poles == 3
        assert rlr.num_asymptotes == 3

    def test_sympy_expression(self) -> None:
        s = sp.Symbol("s")
        expr = 1 / (s * (s + 1) * (s + 2))
        rlr = RootLocusRules(expr)
        assert rlr.num_poles == 3

    def test_tuple_input(self) -> None:
        rlr = RootLocusRules(([1], [1, 3, 2, 0]))
        assert rlr.num_poles == 3

    def test_invalid_input(self) -> None:
        with pytest.raises(TypeError):
            RootLocusRules("invalid_input_type")


class TestRootLocusPedagogicalOutputs:
    """Test explain_steps, _repr_latex_, _repr_markdown_, and __str__."""

    def test_explain_steps(self) -> None:
        G = cp.tf([1], [1, 3, 2, 0])
        rlr = RootLocusRules(G)
        steps = rlr.explain_steps()
        assert isinstance(steps, list)
        assert len(steps) >= 7

        full_text = "\n".join(steps)
        assert "Rule 1 (Branches & Terminations)" in full_text
        assert "Rule 2 (Real-Axis Segments)" in full_text
        assert "Rule 3 (Asymptotes" in full_text
        assert "Rule 4 (Breakaway / Break-in Points)" in full_text
        assert "Rule 7 (Imaginary Axis Crossings" in full_text

    def test_repr_latex_and_markdown(self) -> None:
        G = cp.tf([1], [1, 3, 2, 0])
        rlr = RootLocusRules(G)
        latex_str = rlr._repr_latex_()
        assert r"\begin{aligned}" in latex_str
        assert "Rule 1" in latex_str
        assert "Rule 2" in latex_str

        md_str = rlr._repr_markdown_()
        assert md_str.startswith("$$")
        assert md_str.endswith("$$")

    def test_str_output(self) -> None:
        G = cp.tf([1], [1, 3, 2, 0])
        rlr = RootLocusRules(G)
        s_out = str(rlr)
        assert "Analytical Root Locus Rules Summary" in s_out
        assert "Branches: 3" in s_out

    def test_backward_compatibility_alias(self) -> None:
        assert RootLocusRulesResult is RootLocusRules
        G = cp.tf([1], [1, 1])
        res = root_locus_rules(G)
        assert isinstance(res, RootLocusRules)

"""Comprehensive unit test suite for StateSpaceTutor and canonical transformations."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

import ctrlpy as cp
from ctrlpy.exceptions import DimensionMismatchError
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


class TestStateSpaceTutorStandard:
    """Test suite for standard 2nd and 3rd order observable/controllable state-space systems."""

    def test_second_order_system_canonical_forms(self) -> None:
        """P(s) = s^2 + 3s + 2, roots at -1, -2."""
        A = [[0, 1], [-2, -3]]
        B = [[0], [1]]
        C = [[1, 0]]
        D = [[0]]

        tutor = StateSpaceTutor(A, B, C, D)
        assert tutor.n_states == 2
        assert tutor.n_inputs == 1
        assert tutor.n_outputs == 1
        assert tutor.is_siso is True

        # Controllability
        assert tutor.is_controllable is True
        assert tutor.controllability_rank == 2
        C_expected = sp.Matrix([[0, 1], [1, -3]])
        assert tutor.controllability_matrix == C_expected

        # Observability
        assert tutor.is_observable is True
        assert tutor.observability_rank == 2
        O_expected = sp.Matrix([[1, 0], [0, 1]])
        assert tutor.observability_matrix == O_expected

        # Eigenvalues
        assert len(tutor.eigenvalues) == 2
        assert sp.Integer(-1) in tutor.eigenvalues
        assert sp.Integer(-2) in tutor.eigenvalues
        assert len(tutor.uncontrollable_modes) == 0
        assert len(tutor.unobservable_modes) == 0

        # Controllable Canonical Form (already in CCF)
        ccf = tutor.controllable_canonical_form()
        assert isinstance(ccf, CanonicalFormResult)
        assert ccf.is_valid is True
        assert ccf.T == sp.eye(2)
        assert ccf.A == sp.Matrix([[0, 1], [-2, -3]])
        assert ccf.B == sp.Matrix([[0], [1]])
        assert ccf.C == sp.Matrix([[1, 0]])

        # Observable Canonical Form
        ocf = tutor.observable_canonical_form()
        assert isinstance(ocf, CanonicalFormResult)
        assert ocf.is_valid is True
        assert ocf.A == sp.Matrix([[0, -2], [1, -3]])
        assert ocf.B == sp.Matrix([[1], [0]])
        assert ocf.C == sp.Matrix([[0, 1]])

        # Verify OCF Round-trip similarity transformation: A_o = T_o^-1 A T_o, etc.
        To = ocf.T
        To_inv = ocf.T_inv
        assert To is not None and To_inv is not None
        assert sp.simplify(To_inv * sp.Matrix(A) * To) == ocf.A
        assert sp.simplify(To_inv * sp.Matrix(B)) == ocf.B
        assert sp.simplify(sp.Matrix(C) * To) == ocf.C

        # Jordan Canonical Form
        jcf = tutor.jordan_canonical_form()
        assert isinstance(jcf, CanonicalFormResult)
        assert jcf.is_valid is True
        V = jcf.T
        V_inv = jcf.T_inv
        assert V is not None and V_inv is not None
        assert sp.simplify(V_inv * sp.Matrix(A) * V) == jcf.A
        assert sp.simplify(V_inv * sp.Matrix(B)) == jcf.B
        assert sp.simplify(sp.Matrix(C) * V) == jcf.C

        # Numerical conversion checks
        ss_num = tutor.to_ss()
        assert isinstance(ss_num, cp.StateSpace)
        assert ss_num.n_states == 2
        tf_num = tutor.to_tf()
        assert isinstance(tf_num, cp.TransferFunction)
        assert np.allclose(tf_num.den, [1, 3, 2])

    def test_third_order_system(self) -> None:
        """3rd order system with poles at -1, -2, -3: (s+1)(s+2)(s+3) = s^3 + 6s^2 + 11s + 6."""
        A = [[0, 1, 0], [0, 0, 1], [-6, -11, -6]]
        B = [[0], [0], [1]]
        C = [[1, 0, 0]]
        D = [[0]]

        tutor = state_space_tutor(A, B, C, D)
        assert tutor.n_states == 3
        assert tutor.is_controllable is True
        assert tutor.is_observable is True
        assert tutor.controllability_rank == 3
        assert tutor.observability_rank == 3

        # CCF
        ccf = controllable_canonical_form(A, B, C, D)
        assert ccf.is_valid is True
        assert ccf.A[2, 0] == -6
        assert ccf.A[2, 1] == -11
        assert ccf.A[2, 2] == -6

        # OCF
        ocf = observable_canonical_form(A, B, C, D)
        assert ocf.is_valid is True
        assert ocf.A[0, 2] == -6
        assert ocf.A[1, 2] == -11
        assert ocf.A[2, 2] == -6

        # Jordan form
        jcf = jordan_canonical_form(A, B, C, D)
        assert jcf.is_valid is True
        j_diag = [jcf.A[i, i] for i in range(3)]
        assert sp.Integer(-1) in j_diag
        assert sp.Integer(-2) in j_diag
        assert sp.Integer(-3) in j_diag


class TestStateSpaceTutorPoleZeroCancellations:
    """Test suite for systems with pole-zero cancellations (uncontrollable or unobservable modes)."""

    def test_uncontrollable_system_pole_cancellation(self) -> None:
        """System with uncontrollable mode at s = -2."""
        A = [[-1, 0], [0, -2]]
        B = [[1], [0]]
        C = [[1, 1]]
        D = [[0]]

        tutor = StateSpaceTutor(A, B, C, D)
        assert tutor.is_controllable is False
        assert tutor.controllability_rank == 1
        assert tutor.is_observable is True
        assert tutor.observability_rank == 2

        # Uncontrollable mode
        assert len(tutor.uncontrollable_modes) == 1
        assert sp.Integer(-2) in tutor.uncontrollable_modes
        assert len(tutor.unobservable_modes) == 0
        assert isinstance(tutor.modes[0], ModeAnalysis)

        # PBH rank test
        pbh_res = tutor.pbh_test()
        assert len(pbh_res) == 2
        for item in pbh_res:
            if item["eigenvalue"] == -2:
                assert item["is_controllable"] is False
                assert item["is_observable"] is True
                assert item["pbh_controllability_rank"] == 1
                assert item["kalman_type"] == "unctrl_obs"
            elif item["eigenvalue"] == -1:
                assert item["is_controllable"] is True
                assert item["is_observable"] is True
                assert item["kalman_type"] == "co"

        # CCF transformation fails (singular Tc)
        ccf = tutor.controllable_canonical_form()
        assert ccf.is_valid is False
        assert ccf.T is None

        # OCF transformation succeeds
        ocf = tutor.observable_canonical_form()
        assert ocf.is_valid is True
        assert ocf.T is not None

        # Transfer function has cancelled pole at s = -2
        s = sp.Symbol("s")
        assert tutor.transfer_function == 1 / (s + 1)
        tf_num = tutor.to_tf()
        assert len(tf_num.den) == 2  # s + 1 (1st order minimal realization)

    def test_unobservable_system_pole_cancellation(self) -> None:
        """System with unobservable mode at s = -5."""
        A = [[-2, 0], [0, -5]]
        B = [[1], [1]]
        C = [[1, 0]]
        D = [[0]]

        tutor = StateSpaceTutor(A, B, C, D)
        assert tutor.is_controllable is True
        assert tutor.controllability_rank == 2
        assert tutor.is_observable is False
        assert tutor.observability_rank == 1

        assert len(tutor.uncontrollable_modes) == 0
        assert len(tutor.unobservable_modes) == 1
        assert sp.Integer(-5) in tutor.unobservable_modes

        # CCF succeeds
        ccf = tutor.controllable_canonical_form()
        assert ccf.is_valid is True
        assert ccf.T is not None

        # OCF fails (singular To)
        ocf = tutor.observable_canonical_form()
        assert ocf.is_valid is False
        assert ocf.T is None

        # Transfer function: 1 / (s + 2)
        s = sp.Symbol("s")
        assert tutor.transfer_function == 1 / (s + 2)

    def test_both_uncontrollable_and_unobservable_modes(self) -> None:
        """3-state system: mode -1 is CO, mode -2 is C_unobs, mode -3 is unctrl_obs."""
        A = [[-1, 0, 0], [0, -2, 0], [0, 0, -3]]
        B = [[1], [1], [0]]
        C = [[1, 0, 1]]
        D = [[0]]

        tutor = StateSpaceTutor(A, B, C, D)
        assert tutor.is_controllable is False
        assert tutor.is_observable is False
        assert tutor.controllability_rank == 2
        assert tutor.observability_rank == 2

        assert sp.Integer(-3) in tutor.uncontrollable_modes
        assert sp.Integer(-2) in tutor.unobservable_modes


class TestStateSpaceTutorInputTypes:
    """Test suite for initializing StateSpaceTutor from different representations."""

    def test_from_ctrlpy_statespace(self) -> None:
        """Initialize from ctrlpy StateSpace instance."""
        ss_model = cp.ss([[0, 1], [-6, -5]], [[0], [1]], [[1, 1]], [[0]])
        tutor = StateSpaceTutor(ss_model)
        assert tutor.n_states == 2
        assert tutor.is_controllable is True
        assert tutor.is_observable is True

    def test_from_ctrlpy_transferfunction(self) -> None:
        """Initialize from ctrlpy TransferFunction instance."""
        tf_model = cp.tf([2, 1], [1, 5, 6])
        tutor = StateSpaceTutor(tf_model)
        assert tutor.n_states == 2
        assert tutor.is_controllable is True
        assert tutor.A == sp.Matrix([[0, 1], [-6, -5]])
        assert tutor.B == sp.Matrix([[0], [1]])
        assert tutor.C == sp.Matrix([[1, 2]])

    def test_from_polynomial_coefficients(self) -> None:
        """Initialize from numerator/denominator coefficient lists."""
        tutor = StateSpaceTutor(num=[1, 3], den=[1, 4, 3])
        assert tutor.n_states == 2
        assert tutor.A == sp.Matrix([[0, 1], [-3, -4]])
        assert tutor.B == sp.Matrix([[0], [1]])

    def test_from_numpy_arrays(self) -> None:
        """Initialize from NumPy arrays."""
        A = np.array([[-1.0, 1.0], [0.0, -2.0]])
        B = np.array([[0.0], [1.0]])
        C = np.array([[1.0, 0.0]])
        D = np.array([[0.0]])
        tutor = StateSpaceTutor(A, B, C, D)
        assert tutor.n_states == 2
        assert tutor.is_controllable is True
        assert tutor.is_observable is True

    def test_from_sympy_matrices(self) -> None:
        """Initialize from SymPy matrices."""
        A = sp.Matrix([[0, 1], [-4, -4]])
        B = sp.Matrix([[0], [1]])
        C = sp.Matrix([[1, 0]])
        D = sp.Matrix([[0]])
        tutor = StateSpaceTutor(A, B, C, D)
        assert tutor.n_states == 2
        assert tutor.is_controllable is True

    def test_discrete_time_system(self) -> None:
        """Discrete-time system with sampling period dt."""
        tutor = StateSpaceTutor([[0.5, 0.2], [0.0, 0.8]], [[0], [1]], [[1, 0]], [[0]], dt=0.05)
        assert tutor.dt == 0.05
        assert "z" in str(tutor.characteristic_polynomial)
        assert "x[k+1]" in tutor._repr_latex_()
        assert "z[k+1]" in tutor.controllable_canonical_form()._repr_latex_()


class TestStateSpaceTutorSpecialCases:
    """Test suite for edge cases, Jordan blocks, non-zero D, and standalone functions."""

    def test_repeated_eigenvalues_jordan_block(self) -> None:
        """Deficient matrix with Jordan block: (s+2)^2."""
        A = [[-2, 1], [0, -2]]
        B = [[0], [1]]
        C = [[1, 0]]
        D = [[0]]

        tutor = StateSpaceTutor(A, B, C, D)
        assert tutor.n_states == 2
        assert tutor.is_controllable is True
        assert tutor.is_observable is True

        jcf = tutor.jordan_canonical_form()
        assert jcf.is_valid is True
        assert jcf.A[0, 0] == -2
        assert jcf.A[1, 1] == -2

    def test_nonzero_direct_feedthrough_D(self) -> None:
        """Transfer function with non-zero D: G(s) = (2s + 5) / (s + 1) = 2 + 3/(s+1)."""
        tutor = StateSpaceTutor(num=[2, 5], den=[1, 1])
        assert tutor.n_states == 1
        assert tutor.D == sp.Matrix([[2]])
        assert tutor.C == sp.Matrix([[3]])
        assert tutor.A == sp.Matrix([[-1]])
        assert tutor.B == sp.Matrix([[1]])

    def test_first_order_system(self) -> None:
        """1-state system: dx/dt = -3x + 2u, y = x."""
        tutor = StateSpaceTutor([[-3]], [[2]], [[1]], [[0]])
        assert tutor.n_states == 1
        assert tutor.is_controllable is True
        assert tutor.is_observable is True
        assert tutor.controllability_matrix == sp.Matrix([[2]])
        assert tutor.observability_matrix == sp.Matrix([[1]])

        ccf = tutor.controllable_canonical_form()
        assert ccf.is_valid is True
        assert ccf.A == sp.Matrix([[-3]])
        assert ccf.B == sp.Matrix([[1]])
        assert ccf.C == sp.Matrix([[2]])

    def test_standalone_matrix_functions(self) -> None:
        """Test standalone controllability_matrix and observability_matrix functions."""
        A = [[0, 1], [-2, -3]]
        B = [[0], [1]]
        C = [[1, 0]]

        c_mat = controllability_matrix(A, B)
        assert c_mat == sp.Matrix([[0, 1], [1, -3]])

        o_mat = observability_matrix(A, C)
        assert o_mat == sp.Matrix([[1, 0], [0, 1]])

        ss_model = cp.ss(A, B, C, [[0]])
        assert controllability_matrix(ss_model) == c_mat
        assert observability_matrix(ss_model) == o_mat


class TestStateSpaceTutorPedagogyAndReprs:
    """Test suite for explain_steps, LaTeX formatting, Markdown, str, and repr."""

    def test_explain_steps_content(self) -> None:
        A = [[0, 1], [-2, -3]]
        B = [[0], [1]]
        C = [[1, 0]]
        D = [[0]]

        tutor = StateSpaceTutor(A, B, C, D)
        steps = tutor.explain_steps()
        assert len(steps) >= 7
        assert any("Step 1: System Definition" in s for s in steps)
        assert any("Step 2: Controllability Matrix" in s for s in steps)
        assert any("Step 3: Observability Matrix" in s for s in steps)
        assert any("Step 4: Popov-Belevitch-Hautus" in s for s in steps)
        assert any("Step 5: Controllable Canonical Form" in s for s in steps)
        assert any("Step 6: Observable Canonical Form" in s for s in steps)
        assert any("Step 7: Jordan / Diagonal Modal Form" in s for s in steps)

    def test_latex_and_markdown_reprs(self) -> None:
        A = [[0, 1], [-2, -3]]
        B = [[0], [1]]
        C = [[1, 0]]
        D = [[0]]

        tutor = StateSpaceTutor(A, B, C, D)
        latex_str = tutor._repr_latex_()
        assert r"\begin{aligned}" in latex_str
        assert r"\operatorname{rank}(\mathcal{C})" in latex_str
        assert r"\operatorname{rank}(\mathcal{O})" in latex_str

        md_str = tutor._repr_markdown_()
        assert "$$" in md_str

        # CanonicalFormResult representations
        ccf = tutor.controllable_canonical_form()
        ccf_latex = ccf._repr_latex_()
        assert "Controllable Canonical Form" in ccf_latex
        assert "$$" in ccf._repr_markdown_()

        # str and repr
        str_out = str(tutor)
        assert "Pedagogical State-Space Analysis" in str_out
        assert "Controllability Matrix" in str_out

        repr_out = repr(tutor)
        assert "StateSpaceTutor" in repr_out

        ccf_str = str(ccf)
        assert "Controllable Canonical Form" in ccf_str
        ccf_repr = repr(ccf)
        assert "CanonicalFormResult" in ccf_repr

    def test_invalid_dimensions_and_errors(self) -> None:
        with pytest.raises(DimensionMismatchError):
            StateSpaceTutor([[1, 2, 3], [4, 5, 6]], [[1], [1]], [[1, 1]], [[0]])

        with pytest.raises(DimensionMismatchError):
            StateSpaceTutor([[1, 0], [0, 1]], [[1], [1], [1]], [[1, 0]], [[0]])

        with pytest.raises(DimensionMismatchError):
            StateSpaceTutor([[1, 0], [0, 1]], [[1], [0]], [[1, 0, 0]], [[0]])

        with pytest.raises(DimensionMismatchError):
            StateSpaceTutor([[1, 0], [0, 1]], [[1], [0]], [[1, 0]], [[1, 2], [3, 4]])

        with pytest.raises(ValueError):
            StateSpaceTutor()

        with pytest.raises(ValueError):
            StateSpaceTutor([[1, 0], [0, 1]])

        with pytest.raises(ValueError):
            StateSpaceTutor([[1, 0], [0, 1]], B=[[1], [0]])

        with pytest.raises(ValueError):
            controllability_matrix([[1, 0], [0, 1]])

        with pytest.raises(ValueError):
            observability_matrix([[1, 0], [0, 1]])

    def test_symbolic_variables_and_to_ss_error(self) -> None:
        k = sp.Symbol("k")
        tutor = StateSpaceTutor([[0, 1], [-k, -2]], [[0], [1]], [[1, 0]], [[0]])
        assert tutor.n_states == 2
        with pytest.raises(ValueError, match="Cannot convert symbolic expression"):
            tutor.to_ss()
        with pytest.raises(ValueError, match="Cannot convert symbolic expression"):
            tutor.controllable_canonical_form().to_ss()

    def test_mimo_system_properties(self) -> None:
        A = [[-1, 0], [0, -2]]
        B = [[1, 0], [0, 1]]
        C = [[1, 0], [0, 1]]
        D = [[0, 0], [0, 0]]
        tutor = StateSpaceTutor(A, B, C, D)
        assert tutor.is_siso is False
        assert tutor.n_inputs == 2
        assert tutor.n_outputs == 2
        assert tutor.is_controllable is True
        assert tutor.is_observable is True

        ccf = tutor.controllable_canonical_form()
        assert ccf.is_valid is False
        assert "defined for SISO" in ccf.explanation

        ocf = tutor.observable_canonical_form()
        assert ocf.is_valid is False
        assert "defined for SISO" in ocf.explanation

        with pytest.raises(
            NotImplementedError, match="to_tf\\(\\) is only supported for SISO systems"
        ):
            tutor.to_tf()

    def test_scalar_gain_transfer_function(self) -> None:
        tutor = StateSpaceTutor(num=[5], den=[1])
        assert tutor.n_states == 1
        assert tutor.D == sp.Matrix([[5]])

    def test_complex_and_float_entries(self) -> None:
        A = [[0, 1.0], [-2.5, -3.5]]
        B = [0, 1]  # 1D list
        C = [1, 0]  # 1D list
        D = 0  # scalar
        tutor = StateSpaceTutor(A, B, C, D)
        assert tutor.n_states == 2
        assert tutor.is_controllable is True
        assert tutor.is_observable is True

    def test_auto_broadcasting_scalar_D(self) -> None:
        A = [[-1, 0], [0, -2]]
        B = [[1, 0], [0, 1]]
        C = [[1, 0], [0, 1]]
        D = 3  # scalar broadcast to (2, 2)
        tutor = StateSpaceTutor(A, B, C, D)
        assert tutor.D == sp.Matrix([[3, 0], [0, 3]])

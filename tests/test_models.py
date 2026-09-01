"""Comprehensive test suite for LTI models (TransferFunction and StateSpace)."""

from __future__ import annotations

import numpy as np
import pytest

import ctrlpy
from ctrlpy import (
    CtrlPyError,
    DimensionMismatchError,
    LinearTimeInvariant,
    StateSpace,
    TransferFunction,
    UnstableSystemError,
    UnstableSystemWarning,
    ss,
    tf,
)


class TestAliasesAndBase:
    """Test module-level exports, base class, and aliases."""

    def test_exports(self) -> None:
        """Verify all classes, aliases, and version are exported at root."""
        assert ctrlpy.__version__ == "0.1.0"
        assert ctrlpy.LinearTimeInvariant is LinearTimeInvariant
        assert ctrlpy.LTI is LinearTimeInvariant
        assert ctrlpy.TransferFunction is TransferFunction
        assert ctrlpy.tf is TransferFunction
        assert ctrlpy.StateSpace is StateSpace
        assert ctrlpy.ss is StateSpace
        assert ctrlpy.CtrlPyError is CtrlPyError
        assert ctrlpy.UnstableSystemError is UnstableSystemError
        assert ctrlpy.UnstableSystemWarning is UnstableSystemWarning
        assert ctrlpy.DimensionMismatchError is DimensionMismatchError

    def test_exception_hierarchy(self) -> None:
        """Verify exception inheritance structure."""
        assert issubclass(CtrlPyError, Exception)
        assert issubclass(UnstableSystemError, CtrlPyError)
        assert issubclass(UnstableSystemWarning, UserWarning)
        assert issubclass(DimensionMismatchError, CtrlPyError)
        assert issubclass(DimensionMismatchError, ValueError)

    def test_abstract_base_cannot_be_instantiated(self) -> None:
        """Verify LinearTimeInvariant cannot be directly instantiated."""
        with pytest.raises(TypeError):
            LinearTimeInvariant()  # type: ignore[abstract]

    def test_incomplete_subclass_raises(self) -> None:
        """Verify subclasses missing abstract methods cannot be instantiated."""

        class IncompleteLTI(LinearTimeInvariant):
            @property
            def inputs(self) -> int:
                return 1

        with pytest.raises(TypeError):
            IncompleteLTI()  # type: ignore[abstract]


class TestTransferFunction:
    """Tests for TransferFunction representation."""

    def test_initialization_various_types(self) -> None:
        """Test instantiation with lists, arrays, and scalars."""
        # List initialization
        sys1 = TransferFunction([1], [1, 2, 1])
        np.testing.assert_allclose(sys1.num, [1.0])
        np.testing.assert_allclose(sys1.den, [1.0, 2.0, 1.0])
        assert sys1.inputs == 1
        assert sys1.outputs == 1
        assert sys1.is_siso is True

        # NumPy array initialization
        sys2 = tf(np.array([2.0, 4.0]), np.array([1.0, 3.0]))
        np.testing.assert_allclose(sys2.num, [2.0, 4.0])
        np.testing.assert_allclose(sys2.den, [1.0, 3.0])

        # Scalar denominator
        sys3 = tf([1.0, 2.0], 2.0)
        np.testing.assert_allclose(sys3.num, [0.5, 1.0])
        np.testing.assert_allclose(sys3.den, [1.0])

        # Default denominator
        sys4 = tf([3.0, 6.0])
        np.testing.assert_allclose(sys4.num, [3.0, 6.0])
        np.testing.assert_allclose(sys4.den, [1.0])

    def test_normalization_and_leading_zeros(self) -> None:
        """Test normalization by leading denominator coefficient and trimming zeros."""
        # Normalization by leading denominator coefficient
        sys = tf([2.0, 4.0], [2.0, 6.0, 4.0])
        np.testing.assert_allclose(sys.num, [1.0, 2.0])
        np.testing.assert_allclose(sys.den, [1.0, 3.0, 2.0])

        # Trimming leading zeros in denominator
        sys_lead_den = tf([1.0], [0.0, 0.0, 1.0, 2.0])
        np.testing.assert_allclose(sys_lead_den.den, [1.0, 2.0])

        # Trimming leading zeros in numerator
        sys_lead_num = tf([0.0, 0.0, 2.0, 4.0], [1.0, 2.0])
        np.testing.assert_allclose(sys_lead_num.num, [2.0, 4.0])

        # Zero numerator
        sys_zero = tf([0.0, 0.0], [1.0, 1.0])
        np.testing.assert_allclose(sys_zero.num, [0.0])
        np.testing.assert_allclose(sys_zero.den, [1.0, 1.0])

    def test_invalid_initialization(self) -> None:
        """Test invalid numerator and denominator inputs."""
        with pytest.raises(ValueError, match="Numerator cannot be empty"):
            tf([], [1, 2])

        with pytest.raises(ValueError, match="Denominator cannot be empty"):
            tf([1], [])

        with pytest.raises(ValueError, match="Denominator cannot be identically zero"):
            tf([1], [0, 0, 0])

    def test_poles_and_zeros(self) -> None:
        """Test pole and zero calculation against analytical solutions."""
        # First order system: G(s) = (s + 3) / (s + 2) -> zero = -3, pole = -2
        sys1 = tf([1, 3], [1, 2])
        np.testing.assert_allclose(np.sort(sys1.poles()), [-2.0])
        np.testing.assert_allclose(np.sort(sys1.zeros()), [-3.0])

        # Second order system with real roots: G(s) = (s + 2) / (s^2 + 3s + 2) -> poles = -2, -1
        sys2 = tf([1, 2], [1, 3, 2])
        np.testing.assert_allclose(np.sort(sys2.poles().real), [-2.0, -1.0])
        np.testing.assert_allclose(np.sort(sys2.zeros().real), [-2.0])

        # Complex conjugate poles: s^2 + 2s + 5 = 0 -> s = -1 +- 2j
        sys3 = tf([1], [1, 2, 5])
        poles3 = np.sort_complex(sys3.poles())
        np.testing.assert_allclose(poles3, [-1.0 - 2.0j, -1.0 + 2.0j])
        assert len(sys3.zeros()) == 0

        # Complex conjugate zeros: s^2 + 4 = 0 -> s = +- 2j
        sys4 = tf([1, 0, 4], [1, 1])
        zeros4 = np.sort_complex(sys4.zeros())
        np.testing.assert_allclose(zeros4, [-2.0j, 2.0j])

        # Static gain: no poles, no zeros
        sys_gain = tf([5], [1])
        assert len(sys_gain.poles()) == 0
        assert len(sys_gain.zeros()) == 0

        # Zero numerator: no zeros
        sys_zero = tf([0], [1, 1])
        assert len(sys_zero.zeros()) == 0

    def test_to_ss_conversion(self) -> None:
        """Test conversion of TransferFunction to StateSpace."""
        # SISO transfer function: 1 / (s^2 + 2s + 1)
        sys_tf = tf([1], [1, 2, 1])
        sys_ss = sys_tf.to_ss()
        assert isinstance(sys_ss, StateSpace)
        assert sys_ss.n_states == 2
        assert sys_ss.inputs == 1
        assert sys_ss.outputs == 1
        np.testing.assert_allclose(np.sort_complex(sys_ss.poles()), np.sort_complex(sys_tf.poles()))

        # Improper transfer function raises ValueError
        sys_improper = tf([1, 2, 3], [1, 2])
        with pytest.raises(ValueError, match="Improper transfer function"):
            sys_improper.to_ss()

    def test_string_representations(self) -> None:
        """Test __repr__, __str__, and _repr_latex_."""
        sys = tf([1], [1, 2, 1])
        assert repr(sys) == "TransferFunction(num=[1.0], den=[1.0, 2.0, 1.0])"

        str_repr = str(sys)
        assert "1" in str_repr
        assert "s^2 + 2 s + 1" in str_repr
        assert "-" in str_repr

        # Check LaTeX representations for various polynomials
        assert sys._repr_latex_() == r"$$\frac{1}{s^{2} + 2 s + 1}$$"

        sys2 = tf([2, -3, 0], [1, 0, 4])
        assert sys2._repr_latex_() == r"$$\frac{2 s^{2} - 3 s}{s^{2} + 4}$$"

        sys3 = tf([-1, 0], [1])
        assert sys3._repr_latex_() == r"$$\frac{-s}{1}$$"

        sys4 = tf([0], [1, 1])
        assert sys4._repr_latex_() == r"$$\frac{0}{s + 1}$$"


class TestStateSpace:
    """Tests for StateSpace representation."""

    def test_initialization_and_validation(self) -> None:
        """Test initialization and shape compatibility checks."""
        # Valid SISO system
        A = [[-2.0, -1.0], [1.0, 0.0]]
        B = [[1.0], [0.0]]
        C = [[0.0, 1.0]]
        D = [[0.0]]
        sys = StateSpace(A, B, C, D)
        assert sys.n_states == 2
        assert sys.inputs == 1
        assert sys.outputs == 1
        assert sys.is_siso is True
        np.testing.assert_allclose(sys.A, A)
        np.testing.assert_allclose(sys.B, B)
        np.testing.assert_allclose(sys.C, C)
        np.testing.assert_allclose(sys.D, D)

        # 1D array convenience for B and C, scalar for D
        sys_1d = StateSpace(A, [1.0, 0.0], [0.0, 1.0], 0.0)
        assert sys_1d.B.shape == (2, 1)
        assert sys_1d.C.shape == (1, 2)
        assert sys_1d.D.shape == (1, 1)

        # Scalar 1-state system
        sys_scalar = ss(-3.0, 2.0, 1.0, 0.0)
        assert sys_scalar.n_states == 1
        assert sys_scalar.inputs == 1
        assert sys_scalar.outputs == 1

        # MIMO system
        sys_mimo = ss(
            np.eye(2),
            np.eye(2),
            np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            np.zeros((3, 2)),
        )
        assert sys_mimo.n_states == 2
        assert sys_mimo.inputs == 2
        assert sys_mimo.outputs == 3
        assert sys_mimo.is_siso is False

    def test_invalid_dimensions(self) -> None:
        """Test dimension mismatch errors."""
        # Non-square A
        with pytest.raises(ValueError, match="Matrix A must be a square 2D array"):
            ss([[1, 2, 3], [4, 5, 6]], [1, 2], [1, 2], 0)

        # B row mismatch
        with pytest.raises(ValueError, match="Matrix B must have 2 rows"):
            ss(np.eye(2), [[1], [2], [3]], [[1, 0]], [[0]])

        # C col mismatch
        with pytest.raises(ValueError, match="Matrix C must have 2 columns"):
            ss(np.eye(2), [[1], [0]], [[1, 0, 0]], [[0]])

        # D shape mismatch
        with pytest.raises(ValueError, match="Matrix D must have shape"):
            ss(np.eye(2), [[1], [0]], [[1, 0]], np.zeros((2, 2)))

    def test_poles_and_zeros(self) -> None:
        """Test pole and zero calculation for state-space systems."""
        # System: dot(x) = [[-2, -1], [1, 0]] x + [1, 0]^T u, y = [1, 1] x
        # Poles: roots of det(sI - A) = s(s+2) + 1 = s^2 + 2s + 1 -> -1, -1
        # Zeros: G(s) = C(sI - A)^-1 B = (s + 1) / (s^2 + 2s + 1) -> zero at -1
        A = [[-2.0, -1.0], [1.0, 0.0]]
        B = [[1.0], [0.0]]
        C = [[1.0, 1.0]]
        D = [[0.0]]
        sys = ss(A, B, C, D)

        poles = np.sort_complex(sys.poles())
        np.testing.assert_allclose(poles, [-1.0 + 0.0j, -1.0 + 0.0j], atol=1e-7)

        zeros = sys.zeros()
        np.testing.assert_allclose(zeros, [-1.0 + 0.0j], atol=1e-7)

    def test_mimo_zeros_not_implemented(self) -> None:
        """Verify MIMO zeros() raises NotImplementedError."""
        sys_mimo = ss(np.eye(2), np.eye(2), np.eye(2), np.zeros((2, 2)))
        with pytest.raises(NotImplementedError, match="zeros.*SISO"):
            sys_mimo.zeros()

    def test_to_tf_conversion(self) -> None:
        """Test conversion of StateSpace to TransferFunction."""
        A = [[-2.0, -1.0], [1.0, 0.0]]
        B = [[1.0], [0.0]]
        C = [[0.0, 1.0]]
        D = [[0.0]]
        sys_ss = ss(A, B, C, D)
        sys_tf = sys_ss.to_tf()

        assert isinstance(sys_tf, TransferFunction)
        np.testing.assert_allclose(sys_tf.num, [1.0], atol=1e-10)
        np.testing.assert_allclose(sys_tf.den, [1.0, 2.0, 1.0], atol=1e-10)

        # Out-of-bounds channel indices
        with pytest.raises(IndexError):
            sys_ss.to_tf(input_index=2)
        with pytest.raises(IndexError):
            sys_ss.to_tf(output_index=2)

    def test_string_representations(self) -> None:
        """Test __repr__, __str__, and _repr_latex_."""
        A = [[-2.0, -1.0], [1.0, 0.0]]
        B = [[1.0], [0.0]]
        C = [[0.0, 1.0]]
        D = [[0.0]]
        sys = ss(A, B, C, D)

        assert "StateSpace(" in repr(sys)
        assert "A=" in repr(sys)

        str_repr = str(sys)
        assert "StateSpace:" in str_repr
        assert "A =" in str_repr
        assert "B =" in str_repr

        latex_repr = sys._repr_latex_()
        assert r"\begin{bmatrix}" in latex_repr
        assert r"\dot{x}" in latex_repr
        assert r"y &=" in latex_repr
        assert r"\\" in latex_repr
        assert "\n" not in latex_repr
        assert latex_repr.startswith(r"$$\begin{aligned}")
        assert latex_repr.endswith(r"\end{aligned}$$")


class TestRoundTripConversion:
    """Test round-trip conversions between TransferFunction and StateSpace."""

    @pytest.mark.parametrize(
        ("num", "den"),
        [
            ([1.0], [1.0, 2.0]),  # 1st order strictly proper
            ([1.0, 1.0], [1.0, 3.0, 2.0]),  # 2nd order strictly proper
            ([2.0, 3.0], [1.0, 1.0]),  # 1st order proper with feedthrough
            ([3.0, 2.0, 5.0], [1.0, 4.0, 6.0, 4.0]),  # 3rd order strictly proper
            (
                [1.0, 0.0, 2.0],
                [1.0, 2.0, 1.0],
            ),  # Proper with feedthrough and middle zero
        ],
    )
    def test_tf_to_ss_to_tf_roundtrip(self, num: list[float], den: list[float]) -> None:
        """Verify tf.to_ss().to_tf() preserves normalized coefficients."""
        tf_orig = tf(num, den)
        ss_model = tf_orig.to_ss()
        tf_recovered = ss_model.to_tf()

        assert tf_recovered.num == pytest.approx(tf_orig.num, abs=1e-7)
        assert tf_recovered.den == pytest.approx(tf_orig.den, abs=1e-7)

        # Poles and zeros should also match
        np.testing.assert_allclose(
            np.sort_complex(tf_recovered.poles()),
            np.sort_complex(tf_orig.poles()),
            atol=1e-7,
        )
        if len(tf_orig.zeros()) > 0:
            np.testing.assert_allclose(
                np.sort_complex(tf_recovered.zeros()),
                np.sort_complex(tf_orig.zeros()),
                atol=1e-7,
            )


class TestMixedArithmeticAndEdgeCases:
    """Test mixed arithmetic between TransferFunction, StateSpace, and scalars."""

    def test_tf_ss_mixed_arithmetic(self) -> None:
        """Test + - * / between TransferFunction and StateSpace."""
        g = tf([1], [1, 1])
        s = ss([[-2]], [[1]], [[1]], [[0]])

        # Addition
        add_tf_ss = g + s
        assert isinstance(add_tf_ss, StateSpace)
        add_ss_tf = s + g
        assert isinstance(add_ss_tf, StateSpace)

        # Subtraction
        sub_tf_ss = g - s
        assert isinstance(sub_tf_ss, StateSpace)
        sub_ss_tf = s - g
        assert isinstance(sub_ss_tf, StateSpace)

        # Multiplication
        mul_tf_ss = g * s
        assert isinstance(mul_tf_ss, StateSpace)
        mul_ss_tf = s * g
        assert isinstance(mul_ss_tf, StateSpace)

        # Scalar operations with StateSpace
        ss_plus_scalar = s + 2.0
        assert isinstance(ss_plus_scalar, StateSpace)
        scalar_plus_ss = 2.0 + s
        assert isinstance(scalar_plus_ss, StateSpace)
        ss_sub_scalar = s - 2.0
        assert isinstance(ss_sub_scalar, StateSpace)
        scalar_sub_ss = 2.0 - s
        assert isinstance(scalar_sub_ss, StateSpace)
        ss_mul_scalar = s * 3.0
        assert isinstance(ss_mul_scalar, StateSpace)
        scalar_mul_ss = 3.0 * s
        assert isinstance(scalar_mul_ss, StateSpace)
        ss_neg = -s
        assert isinstance(ss_neg, StateSpace)
        ss_pos = +s
        assert isinstance(ss_pos, StateSpace)

        # Scalar with TransferFunction
        tf_scalar_sub = 5.0 - g
        assert isinstance(tf_scalar_sub, TransferFunction)
        tf_pos = +g
        assert isinstance(tf_pos, TransferFunction)

        # Division
        div_tf_scalar = g / 2.0
        assert isinstance(div_tf_scalar, TransferFunction)
        div_scalar_tf = 2.0 / g
        assert isinstance(div_scalar_tf, TransferFunction)

    def test_repr_latex_and_markdown(self) -> None:
        """Test _repr_latex_ and _repr_markdown_ for TF and SS."""
        g = tf([1], [1, 2, 1])
        s = ss([[-2]], [[1]], [[1]], [[0]])

        assert "$$" in g._repr_latex_()
        assert g._repr_markdown_() == g._repr_latex_()
        assert "bmatrix" in s._repr_latex_()
        assert s._repr_markdown_() == s._repr_latex_()
        assert "TransferFunction" in repr(g)
        assert "StateSpace" in repr(s)

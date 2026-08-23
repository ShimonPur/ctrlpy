"""Test suite for system algebra, arithmetic operators, and interconnections."""

from __future__ import annotations

import numpy as np
import pytest

from ctrlpy import TransferFunction, feedback, parallel, series, ss, tf


class TestTransferFunctionArithmetic:
    """Tests for TransferFunction arithmetic operations and feedback."""

    def test_addition_parallel(self) -> None:
        """Test parallel interconnection (G1 + G2)."""
        g1 = tf([1], [1, 1])  # 1 / (s + 1)
        g2 = tf([2], [1, 2])  # 2 / (s + 2)
        # G1 + G2 = (1*(s+2) + 2*(s+1)) / ((s+1)*(s+2)) = (3s + 4) / (s^2 + 3s + 2)
        g_sum = g1 + g2

        np.testing.assert_allclose(g_sum.num, [3.0, 4.0])
        np.testing.assert_allclose(g_sum.den, [1.0, 3.0, 2.0])

    def test_addition_with_scalar(self) -> None:
        """Test addition with scalar numbers (G + k and k + G)."""
        g = tf([1], [1, 1])  # 1 / (s + 1)
        # G + 3 = 1/(s+1) + 3 = (3s + 4) / (s + 1)
        g_add = g + 3
        np.testing.assert_allclose(g_add.num, [3.0, 4.0])
        np.testing.assert_allclose(g_add.den, [1.0, 1.0])

        # 3 + G
        g_radd = 3 + g
        np.testing.assert_allclose(g_radd.num, [3.0, 4.0])
        np.testing.assert_allclose(g_radd.den, [1.0, 1.0])

    def test_subtraction_and_negation(self) -> None:
        """Test subtraction and unary negation."""
        g1 = tf([1], [1, 1])
        g2 = tf([2], [1, 2])

        # -G1 = -1 / (s + 1)
        g_neg = -g1
        np.testing.assert_allclose(g_neg.num, [-1.0])
        np.testing.assert_allclose(g_neg.den, [1.0, 1.0])

        # +G1
        g_pos = +g1
        np.testing.assert_allclose(g_pos.num, [1.0])
        np.testing.assert_allclose(g_pos.den, [1.0, 1.0])

        # G1 - G2 = (1*(s+2) - 2*(s+1)) / ((s+1)(s+2)) = (-s) / (s^2 + 3s + 2)
        g_sub = g1 - g2
        np.testing.assert_allclose(g_sub.num, [-1.0, 0.0])
        np.testing.assert_allclose(g_sub.den, [1.0, 3.0, 2.0])

        # G1 - 2 = (1 - 2(s+1)) / (s+1) = (-2s - 1) / (s + 1)
        g_sub_k = g1 - 2
        np.testing.assert_allclose(g_sub_k.num, [-2.0, -1.0])
        np.testing.assert_allclose(g_sub_k.den, [1.0, 1.0])

        # 2 - G1 = (2(s+1) - 1) / (s+1) = (2s + 1) / (s + 1)
        g_rsub_k = 2 - g1
        np.testing.assert_allclose(g_rsub_k.num, [2.0, 1.0])
        np.testing.assert_allclose(g_rsub_k.den, [1.0, 1.0])

    def test_multiplication_series(self) -> None:
        """Test series cascade multiplication (G1 * G2)."""
        g1 = tf([1], [1, 1])
        g2 = tf([2, 1], [1, 3])
        # G1 * G2 = (2s + 1) / (s^2 + 4s + 3)
        g_prod = g1 * g2

        np.testing.assert_allclose(g_prod.num, [2.0, 1.0])
        np.testing.assert_allclose(g_prod.den, [1.0, 4.0, 3.0])

    def test_multiplication_with_scalar(self) -> None:
        """Test scalar multiplication (G * k and k * G)."""
        g = tf([2, 1], [1, 2])

        g_mul1 = g * 3.0
        np.testing.assert_allclose(g_mul1.num, [6.0, 3.0])
        np.testing.assert_allclose(g_mul1.den, [1.0, 2.0])

        g_mul2 = 3.0 * g
        np.testing.assert_allclose(g_mul2.num, [6.0, 3.0])
        np.testing.assert_allclose(g_mul2.den, [1.0, 2.0])

    def test_division(self) -> None:
        """Test division between TFs and with scalars."""
        g1 = tf([1], [1, 1])
        g2 = tf([2], [1, 2])

        # G1 / G2 = (s + 2) / (2s + 2) = (0.5s + 1) / (s + 1)
        g_div = g1 / g2
        np.testing.assert_allclose(g_div.num, [0.5, 1.0])
        np.testing.assert_allclose(g_div.den, [1.0, 1.0])

        # G1 / 2 = 1 / (2s + 2) = 0.5 / (s + 1)
        g_div_k = g1 / 2.0
        np.testing.assert_allclose(g_div_k.num, [0.5])
        np.testing.assert_allclose(g_div_k.den, [1.0, 1.0])

        # 2 / G1 = 2(s + 1) / 1 = 2s + 2
        g_rdiv_k = 2.0 / g1
        np.testing.assert_allclose(g_rdiv_k.num, [2.0, 2.0])
        np.testing.assert_allclose(g_rdiv_k.den, [1.0])

        # Zero division error
        with pytest.raises(ZeroDivisionError):
            _ = g1 / 0.0

    def test_feedback_unity_and_non_unity(self) -> None:
        """Test closed-loop feedback calculation."""
        g = tf([1], [1, 1])

        # Unity negative feedback: G / (1 + G) = 1 / (s + 2)
        t_unity = g.feedback()
        np.testing.assert_allclose(t_unity.num, [1.0])
        np.testing.assert_allclose(t_unity.den, [1.0, 2.0])
        np.testing.assert_allclose(t_unity.poles(), [-2.0])

        # Unity positive feedback: G / (1 - G) = 1 / s
        t_pos = g.feedback(sign=1)
        np.testing.assert_allclose(t_pos.num, [1.0])
        np.testing.assert_allclose(t_pos.den, [1.0, 0.0])
        np.testing.assert_allclose(t_pos.poles(), [0.0])

        # Dynamic feedback H(s) = 1 / (s + 2):
        # G / (1 + GH) = (s + 2) / (s^2 + 3s + 3)
        h = tf([1], [1, 2])
        t_dyn = g.feedback(h, sign=-1)
        np.testing.assert_allclose(t_dyn.num, [1.0, 2.0])
        np.testing.assert_allclose(t_dyn.den, [1.0, 3.0, 3.0])
        np.testing.assert_allclose(t_dyn.zeros(), [-2.0])


class TestStateSpaceArithmetic:
    """Tests for StateSpace arithmetic operations and feedback."""

    def test_addition_parallel(self) -> None:
        """Test parallel interconnection of StateSpace models."""
        ss1 = ss(-1.0, 1.0, 1.0, 0.0)
        ss2 = ss(-2.0, 1.0, 2.0, 0.0)

        ss_sum = ss1 + ss2
        assert ss_sum.n_states == 2
        assert ss_sum.inputs == 1
        assert ss_sum.outputs == 1

        # Check equivalent transfer function
        tf_sum = ss_sum.to_tf()
        np.testing.assert_allclose(tf_sum.num, [3.0, 4.0], atol=1e-10)
        np.testing.assert_allclose(tf_sum.den, [1.0, 3.0, 2.0], atol=1e-10)

    def test_addition_scalar(self) -> None:
        """Test addition of scalar to StateSpace."""
        ss1 = ss(-1.0, 1.0, 1.0, 0.0)  # 1 / (s + 1)
        ss_add = ss1 + 3.0
        assert ss_add.n_states == 1
        np.testing.assert_allclose(ss_add.D, [[3.0]])

        tf_add = ss_add.to_tf()
        np.testing.assert_allclose(tf_add.num, [3.0, 4.0], atol=1e-10)
        np.testing.assert_allclose(tf_add.den, [1.0, 1.0], atol=1e-10)

        # radd
        ss_radd = 3.0 + ss1
        np.testing.assert_allclose(ss_radd.D, [[3.0]])

    def test_subtraction_and_negation(self) -> None:
        """Test subtraction and negation on StateSpace."""
        ss1 = ss(-1.0, 1.0, 1.0, 0.0)
        ss2 = ss(-2.0, 1.0, 2.0, 0.0)

        ss_neg = -ss1
        np.testing.assert_allclose(ss_neg.C, [[-1.0]])

        ss_sub = ss1 - ss2
        tf_sub = ss_sub.to_tf()
        np.testing.assert_allclose(tf_sub.num, [-1.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(tf_sub.den, [1.0, 3.0, 2.0], atol=1e-10)

    def test_multiplication_series(self) -> None:
        """Test series cascade multiplication in state space."""
        # G1 = 1 / (s + 1), G2 = (2s + 1) / (s + 3)
        tf1 = tf([1], [1, 1])
        tf2 = tf([2, 1], [1, 3])

        ss1 = tf1.to_ss()
        ss2 = tf2.to_ss()

        # Cascade: u -> ss2 -> ss1 -> y
        ss_prod = ss1 * ss2
        assert ss_prod.n_states == 2
        tf_prod = ss_prod.to_tf()
        np.testing.assert_allclose(tf_prod.num, [2.0, 1.0], atol=1e-10)
        np.testing.assert_allclose(tf_prod.den, [1.0, 4.0, 3.0], atol=1e-10)

    def test_multiplication_scalar(self) -> None:
        """Test scalar multiplication with StateSpace."""
        ss1 = ss(-1.0, 1.0, 1.0, 0.0)
        ss_mul = 5.0 * ss1
        np.testing.assert_allclose(ss_mul.C, [[5.0]])

        ss_mul2 = ss1 * 5.0
        np.testing.assert_allclose(ss_mul2.C, [[5.0]])

    def test_feedback(self) -> None:
        """Test closed-loop feedback in state-space."""
        # G(s) = 1 / (s^2 + 2s + 1)
        g_tf = tf([1], [1, 2, 1])
        g_ss = g_tf.to_ss()

        # Unity feedback: T(s) = 1 / (s^2 + 2s + 2)
        cl_ss = g_ss.feedback()
        tf_recovered = cl_ss.to_tf()
        np.testing.assert_allclose(tf_recovered.num, [1.0], atol=1e-10)
        np.testing.assert_allclose(tf_recovered.den, [1.0, 2.0, 2.0], atol=1e-10)

        # Dynamic feedback H(s) = 1 / (s + 3)
        h_tf = tf([1], [1, 3])
        h_ss = h_tf.to_ss()
        cl_dyn_ss = g_ss.feedback(h_ss, sign=-1)
        tf_dyn_rec = cl_dyn_ss.to_tf()

        # Analytical: (s + 3) / (s^3 + 5s^2 + 7s + 4)
        np.testing.assert_allclose(tf_dyn_rec.num, [1.0, 3.0], atol=1e-7)
        np.testing.assert_allclose(tf_dyn_rec.den, [1.0, 5.0, 7.0, 4.0], atol=1e-7)

    def test_singular_algebraic_loop_raises(self) -> None:
        """Test that algebraic loop singularity raises ValueError."""
        # System with D = 1 in unity positive feedback (I - sign*D) = 1 - 1*1 = 0
        sys_sing = ss([[1.0]], [[1.0]], [[1.0]], [[1.0]])
        with pytest.raises(ValueError, match="Well-posedness error"):
            sys_sing.feedback(1.0, sign=1)


class TestAlgebraHelpers:
    """Tests for public helper functions: series, parallel, feedback."""

    def test_series_helper(self) -> None:
        """Test series() function with multiple systems and scalars."""
        g1 = tf([1], [1, 1])
        g2 = tf([1], [1, 2])
        g3 = tf([2], [1, 3])

        # series(g1, g2, g3) = 2 / ((s+1)(s+2)(s+3)) = 2 / (s^3 + 6s^2 + 11s + 6)
        res = series(g1, g2, g3)
        assert isinstance(res, TransferFunction)
        np.testing.assert_allclose(res.num, [2.0])
        np.testing.assert_allclose(res.den, [1.0, 6.0, 11.0, 6.0])

        # with scalar gains
        res_scaled = series(2, g1, 3)
        assert isinstance(res_scaled, TransferFunction)
        np.testing.assert_allclose(res_scaled.num, [6.0])
        np.testing.assert_allclose(res_scaled.den, [1.0, 1.0])

        # empty raises
        with pytest.raises(ValueError):
            series()

    def test_parallel_helper(self) -> None:
        """Test parallel() function with multiple systems."""
        g1 = tf([1], [1, 1])
        g2 = tf([1], [1, 2])
        # parallel(g1, g2) = (2s + 3) / (s^2 + 3s + 2)
        res = parallel(g1, g2)
        assert isinstance(res, TransferFunction)
        np.testing.assert_allclose(res.num, [2.0, 3.0])
        np.testing.assert_allclose(res.den, [1.0, 3.0, 2.0])

        # with scalar
        res_k = parallel(g1, 2)
        assert isinstance(res_k, TransferFunction)
        np.testing.assert_allclose(res_k.num, [2.0, 3.0])
        np.testing.assert_allclose(res_k.den, [1.0, 1.0])

        # empty raises
        with pytest.raises(ValueError):
            parallel()

    def test_feedback_helper(self) -> None:
        """Test feedback() helper function."""
        g = tf([1], [1, 1])
        h = tf([1], [1, 2])

        cl = feedback(g, h, sign=-1)
        assert isinstance(cl, TransferFunction)
        np.testing.assert_allclose(cl.num, [1.0, 2.0])
        np.testing.assert_allclose(cl.den, [1.0, 3.0, 3.0])

        # scalar forward path
        cl_scalar = feedback(2, g)
        assert isinstance(cl_scalar, TransferFunction)
        # 2 / (1 + 2/(s+1)) = 2(s+1) / (s+3) = (2s + 2) / (s + 3)
        np.testing.assert_allclose(cl_scalar.num, [2.0, 2.0])
        np.testing.assert_allclose(cl_scalar.den, [1.0, 3.0])


class TestCrossRepresentationConsistency:
    """Verify algebraic consistency between TransferFunction and StateSpace representations."""

    def test_tf_ss_addition_consistency(self) -> None:
        """Verify (tf1 + tf2).to_ss() == tf1.to_ss() + tf2.to_ss()."""
        tf1 = tf([1, 2], [1, 3, 2])
        tf2 = tf([3], [1, 4])

        tf_sum = tf1 + tf2
        ss_sum = tf1.to_ss() + tf2.to_ss()

        tf_recovered = ss_sum.to_tf()
        assert tf_recovered.num == pytest.approx(tf_sum.num, abs=1e-7)
        assert tf_recovered.den == pytest.approx(tf_sum.den, abs=1e-7)

    def test_tf_ss_multiplication_consistency(self) -> None:
        """Verify (tf1 * tf2).to_ss() matches tf1.to_ss() * tf2.to_ss()."""
        tf1 = tf([2, 1], [1, 2])
        tf2 = tf([1, 3], [1, 4, 3])

        tf_prod = tf1 * tf2
        ss_prod = tf1.to_ss() * tf2.to_ss()

        tf_recovered = ss_prod.to_tf()
        assert tf_recovered.num == pytest.approx(tf_prod.num, abs=1e-7)
        assert tf_recovered.den == pytest.approx(tf_prod.den, abs=1e-7)

    def test_tf_ss_feedback_consistency(self) -> None:
        """Verify feedback in TF matches feedback in SS."""
        g_tf = tf([2, 1], [1, 2])
        h_tf = tf([1, 1], [2, 3])

        t_tf = g_tf.feedback(h_tf, sign=-1)
        t_ss = g_tf.to_ss().feedback(h_tf.to_ss(), sign=-1)

        t_recovered = t_ss.to_tf()
        assert t_recovered.num == pytest.approx(t_tf.num, abs=1e-7)
        assert t_recovered.den == pytest.approx(t_tf.den, abs=1e-7)

        # Poles and zeros must match
        np.testing.assert_allclose(
            np.sort_complex(t_recovered.poles()),
            np.sort_complex(t_tf.poles()),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            np.sort_complex(t_recovered.zeros()),
            np.sort_complex(t_tf.zeros()),
            atol=1e-7,
        )

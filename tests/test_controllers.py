"""Unit tests for PID controller design and Ziegler-Nichols tuning."""

from __future__ import annotations

import numpy as np
import pytest

from ctrlpy.algebra import feedback
from ctrlpy.controllers import PID, pd, pi, pid, tune_ziegler_nichols
from ctrlpy.exceptions import UnstableSystemError
from ctrlpy.models.state_space import StateSpace
from ctrlpy.models.transfer_function import TransferFunction, tf


class TestPIDControllers:
    """Test suite for PID, PI, and PD controller constructors and filter configurations."""

    def test_proportional_only(self) -> None:
        """Test pure proportional gain controller."""
        c = PID(Kp=3.5)
        assert isinstance(c, TransferFunction)
        assert np.allclose(c.num, [3.5])
        assert np.allclose(c.den, [1.0])
        assert len(c.poles()) == 0
        assert len(c.zeros()) == 0

    def test_proportional_integral(self) -> None:
        """Test PI controller C(s) = Kp + Ki/s = (Kp*s + Ki)/s."""
        c = pi(Kp=2.0, Ki=4.0)
        assert isinstance(c, TransferFunction)
        assert np.allclose(c.num, [2.0, 4.0])
        assert np.allclose(c.den, [1.0, 0.0])
        assert np.allclose(c.poles(), [0.0])
        assert np.allclose(c.zeros(), [-2.0])

    def test_proportional_derivative_ideal(self) -> None:
        """Test ideal PD controller C(s) = Kd*s + Kp."""
        c = pd(Kp=3.0, Kd=0.5)
        assert isinstance(c, TransferFunction)
        assert np.allclose(c.num, [0.5, 3.0])
        assert np.allclose(c.den, [1.0])
        assert len(c.poles()) == 0
        assert np.allclose(c.zeros(), [-6.0])

    def test_proportional_derivative_filtered_tf(self) -> None:
        """Test filtered PD controller C(s) = Kp + Kd*s / (Tf*s + 1)."""
        # Kp=2, Kd=0.5, Tf=0.05 -> num = [2*0.05 + 0.5, 2] = [0.6, 2], den = [0.05, 1]
        c = pd(Kp=2.0, Kd=0.5, Tf=0.05)
        assert isinstance(c, TransferFunction)
        # Normalized TransferFunction: den leads with 1.0 -> den = [1, 20], num = [12, 40]
        assert np.allclose(c.poles(), [-20.0])
        assert np.allclose(c.zeros(), [-2.0 / 0.6])

    def test_proportional_derivative_filtered_n(self) -> None:
        """Test filtered PD controller with filter coefficient N (Tf = Kd / (N * Kp))."""
        # Kp=2, Kd=0.5, N=10 -> Tf = 0.5 / (10 * 2) = 0.025
        c_n = pd(Kp=2.0, Kd=0.5, N=10)
        c_tf = pd(Kp=2.0, Kd=0.5, Tf=0.025)
        assert np.allclose(c_n.num, c_tf.num)
        assert np.allclose(c_n.den, c_tf.den)

    def test_full_pid_ideal(self) -> None:
        """Test ideal PID controller C(s) = (Kd*s^2 + Kp*s + Ki)/s."""
        c = pid(Kp=4.0, Ki=6.0, Kd=1.0)
        assert isinstance(c, TransferFunction)
        assert np.allclose(c.num, [1.0, 4.0, 6.0])
        assert np.allclose(c.den, [1.0, 0.0])
        assert np.allclose(c.poles(), [0.0])
        # Roots of s^2 + 4s + 6 = 0 -> -2 +- sqrt(2)j
        assert np.allclose(c.zeros(), [-2.0 + 1.41421356j, -2.0 - 1.41421356j])

    def test_full_pid_filtered_tf(self) -> None:
        """Test filtered PID controller with Tf parameter."""
        # Kp=2.0, Ki=3.0, Kd=0.5, Tf=0.1
        # num = [(2*0.1 + 0.5), (2 + 3*0.1), 3] = [0.7, 2.3, 3]
        # den = [0.1, 1.0, 0.0]
        c = PID(Kp=2.0, Ki=3.0, Kd=0.5, Tf=0.1)
        assert isinstance(c, TransferFunction)
        # Normalized denominator: s^2 + 10s
        assert np.allclose(c.den, [1.0, 10.0, 0.0])
        assert np.allclose(c.num, [7.0, 23.0, 30.0])

    def test_full_pid_filtered_n(self) -> None:
        """Test filtered PID controller with N parameter."""
        # Kp=5.0, Ki=2.0, Kd=1.0, N=20 -> Tf = 1.0 / (20 * 5) = 0.01
        c_n = PID(Kp=5.0, Ki=2.0, Kd=1.0, N=20)
        c_tf = PID(Kp=5.0, Ki=2.0, Kd=1.0, Tf=0.01)
        assert np.allclose(c_n.num, c_tf.num)
        assert np.allclose(c_n.den, c_tf.den)

    def test_invalid_parameters(self) -> None:
        """Test invalid Tf and N parameter validations."""
        with pytest.raises(ValueError, match="Tf must be non-negative"):
            PID(Kp=1.0, Tf=-0.1)

        with pytest.raises(ValueError, match="N must be positive"):
            PID(Kp=1.0, N=0.0)

        with pytest.raises(ValueError, match="N must be positive"):
            PID(Kp=1.0, N=-5.0)


class TestZieglerNicholsTuning:
    """Test suite for Ziegler-Nichols reaction curve tuning."""

    def test_tuning_third_order_plant(self) -> None:
        """Test Ziegler-Nichols step tuning on G(s) = 1 / ((s+1)(s+2)(s+5))."""
        # Plant: G(s) = 1 / (s^3 + 8s^2 + 17s + 10)
        plant = tf(1.0, np.polymul([1.0, 1.0], np.polymul([1.0, 2.0], [1.0, 5.0])))

        c_pid = tune_ziegler_nichols(plant, method="step", controller_type="pid")
        assert isinstance(c_pid, TransferFunction)

        # Ensure PID controller has degree 2 num and degree 1 den (or filtered)
        assert len(c_pid.num) == 3
        assert len(c_pid.den) == 2

        # Verify closed-loop system is stable
        cl = feedback(c_pid * plant)
        assert np.all(np.real(cl.poles()) < 0.0)

    def test_tuning_controller_types(self) -> None:
        """Test tuning with P, PI, and PID controller types."""
        plant = tf(1.0, [1.0, 3.0, 2.0])  # 1 / ((s+1)(s+2))

        c_p = tune_ziegler_nichols(plant, controller_type="p")
        assert len(c_p.num) == 1 and len(c_p.den) == 1

        c_pi = tune_ziegler_nichols(plant, controller_type="pi")
        assert len(c_pi.num) == 2 and len(c_pi.den) == 2

        c_pid = tune_ziegler_nichols(plant, controller_type="pid")
        assert len(c_pid.num) == 3 and len(c_pid.den) == 2

    def test_tuning_unstable_plant_error(self) -> None:
        """Test error when tuning an unstable plant."""
        unstable_plant = tf(1.0, [1.0, -1.0, 2.0])
        with pytest.raises(UnstableSystemError, match="open-loop strictly stable"):
            tune_ziegler_nichols(unstable_plant)

    def test_tuning_integrator_plant_error(self) -> None:
        """Test error when plant has an unconstrained integrator pole at s=0."""
        integrator_plant = tf(1.0, [1.0, 1.0, 0.0])
        with pytest.raises(UnstableSystemError, match="open-loop strictly stable"):
            tune_ziegler_nichols(integrator_plant)

    def test_tuning_invalid_options(self) -> None:
        """Test error for invalid tuning options."""
        plant = tf(1.0, [1.0, 1.0])

        with pytest.raises(ValueError, match="Unsupported tuning method"):
            tune_ziegler_nichols(plant, method="frequency")  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="Unsupported controller_type"):
            tune_ziegler_nichols(plant, controller_type="state_feedback")  # type: ignore[arg-type]

    def test_tuning_mimo_error(self) -> None:
        """Test error when non-SISO system is passed."""
        mimo = StateSpace(
            A=[[-1, 0], [0, -2]],
            B=[[1, 0], [0, 1]],
            C=[[1, 0]],
            D=[[0, 0]],
        )
        with pytest.raises(ValueError, match="requires a SISO system"):
            tune_ziegler_nichols(mimo)

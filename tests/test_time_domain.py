"""Comprehensive test suite for time-domain simulation and response functions."""

from __future__ import annotations

import numpy as np
import pytest

import ctrlpy
from ctrlpy import (
    TimeResponseData,
    UnstableSystemError,
    forced_response,
    impulse_response,
    ss,
    step_response,
    tf,
)


class TestExports:
    """Test module exports and API availability."""

    def test_exports_at_root(self) -> None:
        """Verify time-domain functions and classes are exported at root level."""
        assert ctrlpy.step_response is step_response
        assert ctrlpy.impulse_response is impulse_response
        assert ctrlpy.forced_response is forced_response
        assert ctrlpy.TimeResponseData is TimeResponseData


class TestTimeResponseData:
    """Test TimeResponseData dataclass, unpacking, and metrics calculation."""

    def test_dataclass_and_unpacking(self) -> None:
        """Verify initialization, array casting, and tuple unpacking."""
        t = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 0.5, 1.0])
        x = np.array([[0.0, 0.0], [0.5, 0.2], [1.0, 0.4]])

        res_tf = TimeResponseData(t, y)
        assert isinstance(res_tf.t, np.ndarray)
        assert isinstance(res_tf.y, np.ndarray)
        assert res_tf.x is None

        # Unpack 2 elements
        t_out, y_out = res_tf
        np.testing.assert_allclose(t_out, t)
        np.testing.assert_allclose(y_out, y)

        # Unpack 3 elements when x is present
        res_ss = TimeResponseData(t, y, x)
        t_ss, y_ss, x_ss = res_ss
        np.testing.assert_allclose(t_ss, t)
        np.testing.assert_allclose(y_ss, y)
        np.testing.assert_allclose(x_ss, x)

    def test_metrics_on_first_order_system(self) -> None:
        """Verify metrics on analytical 1st-order step response: y(t) = 1 - exp(-t/tau)."""
        tau = 2.0
        sys = tf([1], [tau, 1])
        t = np.linspace(0, 20.0, 2000)
        res = step_response(sys, T=t)

        # Steady-state value
        assert res.steady_state_value() == pytest.approx(1.0, rel=1e-3)

        # Rise time (10% to 90%): analytical is tau * ln(9) = 2.0 * 2.197224577 = 4.394449
        expected_tr = tau * np.log(9.0)
        assert res.rise_time() == pytest.approx(expected_tr, rel=1e-3)

        # Settling time (2%): analytical is -tau * ln(0.02) = 2.0 * 3.912023 = 7.824046
        expected_ts_2 = -tau * np.log(0.02)
        assert res.settling_time(tolerance=0.02) == pytest.approx(expected_ts_2, rel=1e-3)

        # Settling time (5%): analytical is -tau * ln(0.05) = 2.0 * 2.995732 = 5.991465
        expected_ts_5 = -tau * np.log(0.05)
        assert res.settling_time(tolerance=0.05) == pytest.approx(expected_ts_5, rel=1e-3)

        # Overshoot should be 0.0 for 1st order monotonic system
        assert res.overshoot() == 0.0

    def test_metrics_on_second_order_underdamped_system(self) -> None:
        """Verify metrics on standard 2nd-order underdamped system against analytical formulas."""
        wn = 3.0
        zeta = 0.4
        sys = tf([wn**2], [1, 2 * zeta * wn, wn**2])
        t = np.linspace(0, 10.0, 5000)
        res = step_response(sys, T=t)

        # Steady-state value
        assert res.steady_state_value() == pytest.approx(1.0, rel=1e-3)

        # Analytical percent overshoot: 100 * exp(-pi * zeta / sqrt(1 - zeta^2))
        wd = wn * np.sqrt(1.0 - zeta**2)
        expected_os = 100.0 * np.exp(-np.pi * zeta / np.sqrt(1.0 - zeta**2))
        assert res.overshoot() == pytest.approx(expected_os, rel=1e-3)

        # Analytical peak time: tp = pi / wd
        expected_tp = np.pi / wd
        assert res.peak_time() == pytest.approx(expected_tp, rel=1e-3)

        # Settling time (2%): exact boundary entry is approx 2.80s, within theoretical 4/(zeta*wn) bound of 3.33s
        ts = res.settling_time(tolerance=0.02)
        assert 2.5 < ts < 3.5

    def test_metrics_negative_step_response(self) -> None:
        """Verify metrics on a system with negative steady-state response."""
        sys = tf([-2], [1, 1])
        t = np.linspace(0, 10.0, 2000)
        res = step_response(sys, T=t)

        assert res.steady_state_value() == pytest.approx(-2.0, rel=1e-3)
        assert res.overshoot() == 0.0
        expected_tr = np.log(9.0)
        assert res.rise_time() == pytest.approx(expected_tr, rel=1e-3)

    def test_metrics_edge_cases(self) -> None:
        """Verify metrics for constant, flat, unsettled, or multi-channel outputs."""
        # Flat response at 0
        t = np.linspace(0, 5, 100)
        y_flat = np.zeros(100)
        res_flat = TimeResponseData(t, y_flat)
        assert res_flat.steady_state_value() == 0.0
        assert res_flat.overshoot() == 0.0
        assert res_flat.rise_time() == 0.0
        assert res_flat.settling_time() == 0.0

        # Unsettled / diverging response
        y_div = np.exp(t)
        res_div = TimeResponseData(t, y_div)
        assert np.isnan(res_div.settling_time())

        # Multi-channel output indexing
        y_multi = np.column_stack([np.ones(100), 2 * np.ones(100)])
        res_multi = TimeResponseData(t, y_multi)
        assert res_multi.steady_state_value(channel=0) == 1.0
        assert res_multi.steady_state_value(channel=1) == 2.0
        with pytest.raises(IndexError):
            res_multi.steady_state_value(channel=2)

    def test_unstable_system_raises_unstable_system_error(self) -> None:
        """Verify metric methods raise UnstableSystemError for unstable systems."""
        # Unstable TF: G(s) = 1 / (s - 1)
        sys_unstable = tf([1], [1, -1])
        res = step_response(sys_unstable, T=2.0)

        with pytest.raises(UnstableSystemError, match="unstable system"):
            res.steady_state_value()

        with pytest.raises(UnstableSystemError, match="unstable system"):
            res.settling_time()

        with pytest.raises(UnstableSystemError, match="unstable system"):
            res.overshoot()

        with pytest.raises(UnstableSystemError, match="unstable system"):
            res.rise_time()

        with pytest.raises(UnstableSystemError, match="unstable system"):
            res.peak_time()

        # Unstable StateSpace
        sys_ss_unstable = ss([[2.0]], [[1.0]], [[1.0]], [[0.0]])
        res_ss = step_response(sys_ss_unstable, T=2.0)
        with pytest.raises(UnstableSystemError):
            res_ss.steady_state_value()


class TestStepResponse:
    """Test step_response function and LinearTimeInvariant.step method."""

    def test_first_order_step_response_tau(self) -> None:
        """Verify G(s) = 1/(tau*s + 1) step response: y(tau) ~ 0.632 and y_ss = 1.0."""
        tau = 1.5
        sys = tf([1], [tau, 1])

        t = np.linspace(0, 10 * tau, 5000)
        res = step_response(sys, T=t)

        # Verify steady-state
        assert res.steady_state_value() == pytest.approx(1.0, rel=1e-4)

        # Verify y(tau) approx 1 - exp(-1) = 0.6321205588
        idx_tau = int(np.argmin(np.abs(res.t - tau)))
        expected_y_tau = 1.0 - np.exp(-1.0)
        assert res.y[idx_tau] == pytest.approx(expected_y_tau, rel=1e-3)

    def test_auto_time_vector_generation(self) -> None:
        """Verify automatic time horizon generation covers settling dynamics."""
        # Fast system
        sys_fast = tf([1], [0.05, 1])
        res_fast = step_response(sys_fast)
        assert res_fast.t[-1] < 1.0  # ~0.3s
        assert res_fast.steady_state_value() == pytest.approx(1.0, rel=1e-3)

        # Slow system
        sys_slow = tf([1], [50, 1])
        res_slow = step_response(sys_slow)
        assert res_slow.t[-1] >= 300.0  # 6 * 50 = 300s
        assert res_slow.steady_state_value() == pytest.approx(1.0, rel=1e-3)

        # Scalar duration
        res_dur = step_response(sys_fast, T=2.0)
        assert res_dur.t[-1] == pytest.approx(2.0)
        assert len(res_dur.t) == 1000

    def test_convenience_method_on_models(self) -> None:
        """Verify .step() method on TransferFunction and StateSpace."""
        sys_tf = tf([1], [1, 1])
        res1 = sys_tf.step(T=5.0)
        assert isinstance(res1, TimeResponseData)
        assert res1.t[-1] == pytest.approx(5.0)

        sys_ss = sys_tf.to_ss()
        res2 = sys_ss.step(T=5.0)
        assert isinstance(res2, TimeResponseData)
        assert res2.x is not None
        np.testing.assert_allclose(res1.y, res2.y, atol=1e-7)

    def test_state_space_initial_condition(self) -> None:
        """Verify step response of StateSpace with initial state X0."""
        # dot(x) = -x + u, y = x. Step response with x(0) = 2.
        # Analytical: x(t) = 1 + (x(0) - 1) * exp(-t) = 1 + exp(-t).
        sys = ss([[-1.0]], [[1.0]], [[1.0]], [[0.0]])
        t = np.linspace(0, 5.0, 100)
        res = step_response(sys, T=t, X0=[2.0])

        y_analytical = 1.0 + np.exp(-t)
        np.testing.assert_allclose(res.y, y_analytical, atol=1e-6)
        assert res.x is not None
        np.testing.assert_allclose(res.x[:, 0], y_analytical, atol=1e-6)

    def test_tf_raises_on_x0(self) -> None:
        """Verify passing X0 to TransferFunction raises ValueError."""
        sys = tf([1], [1, 1])
        with pytest.raises(ValueError, match="Initial state X0 cannot be specified"):
            step_response(sys, X0=[1.0])


class TestImpulseResponse:
    """Test impulse_response function and LinearTimeInvariant.impulse method."""

    def test_first_order_impulse_response(self) -> None:
        """Verify G(s) = 1/(s + 1) impulse response: y(t) = exp(-t)."""
        sys = tf([1], [1, 1])
        t = np.linspace(0, 5.0, 500)
        res = impulse_response(sys, T=t)

        y_analytical = np.exp(-t)
        np.testing.assert_allclose(res.y, y_analytical, atol=1e-6)
        assert res.x is None

    def test_second_order_impulse_response(self) -> None:
        """Verify second order impulse response matches analytical formula."""
        # G(s) = 1 / (s^2 + 1) -> y(t) = sin(t)
        sys = tf([1], [1, 0, 1])
        t = np.linspace(0, 10.0, 1000)
        res = impulse_response(sys, T=t)

        y_analytical = np.sin(t)
        np.testing.assert_allclose(res.y, y_analytical, atol=1e-3)

    def test_impulse_convenience_method(self) -> None:
        """Verify .impulse() method on LTI instances."""
        sys = tf([2], [1, 2])
        res = sys.impulse(T=4.0)
        assert isinstance(res, TimeResponseData)
        assert res.t[-1] == pytest.approx(4.0)


class TestTFvsSSEquivalence:
    """Test numerical equivalence of responses between TransferFunction and StateSpace."""

    @pytest.mark.parametrize(
        ("num", "den"),
        [
            ([1.0], [1.0, 1.0]),  # 1st order
            ([2.0, 5.0], [1.0, 3.0, 2.0]),  # 2nd order with zero
            ([1.0], [1.0, 2.0, 5.0]),  # 2nd order underdamped
            ([1.0, 3.0], [1.0, 4.0, 6.0, 4.0]),  # 3rd order
            ([2.0, 1.0], [1.0, 2.0]),  # Proper with feedthrough (D != 0)
        ],
    )
    def test_step_and_impulse_equivalence(self, num: list[float], den: list[float]) -> None:
        """Verify step and impulse response outputs are identical for TF and SS."""
        sys_tf = tf(num, den)
        sys_ss = sys_tf.to_ss()
        t = np.linspace(0, 6.0, 300)

        # Step response equivalence
        res_step_tf = step_response(sys_tf, T=t)
        res_step_ss = step_response(sys_ss, T=t)
        np.testing.assert_allclose(res_step_tf.y, res_step_ss.y, atol=1e-7)
        assert res_step_tf.x is None
        assert res_step_ss.x is not None
        assert res_step_ss.x.shape == (len(t), sys_ss.n_states)

        # Impulse response equivalence
        res_imp_tf = impulse_response(sys_tf, T=t)
        res_imp_ss = impulse_response(sys_ss, T=t)
        np.testing.assert_allclose(res_imp_tf.y, res_imp_ss.y, atol=1e-7)


class TestForcedResponse:
    """Test forced_response with various arbitrary input signals."""

    def test_ramp_input(self) -> None:
        """Verify response to unit ramp u(t) = t for G(s) = 1/(s + 1).

        Analytical solution with zero initial state:
        y(t) = t - 1 + exp(-t)
        """
        sys = tf([1], [1, 1])
        t = np.linspace(0, 8.0, 1000)
        u_ramp = t

        res = forced_response(sys, T=t, U=u_ramp)
        y_analytical = t - 1.0 + np.exp(-t)
        np.testing.assert_allclose(res.y, y_analytical, atol=1e-6)

    def test_sinusoidal_input(self) -> None:
        """Verify response to sinusoidal input u(t) = sin(t) for G(s) = 1/(s + 1).

        Analytical solution:
        y(t) = 0.5 * exp(-t) + 0.5 * sin(t) - 0.5 * cos(t)
        """
        sys = tf([1], [1, 1])
        t = np.linspace(0, 10.0, 2000)
        u_sin = np.sin(t)

        res = forced_response(sys, T=t, U=u_sin)
        y_analytical = 0.5 * np.exp(-t) + 0.5 * np.sin(t) - 0.5 * np.cos(t)
        np.testing.assert_allclose(res.y, y_analytical, atol=1e-4)

    def test_constant_input_matches_step_response(self) -> None:
        """Verify constant input U = 1 matches step_response."""
        sys = tf([2, 1], [1, 2, 2])
        t = np.linspace(0, 5.0, 200)

        res_step = step_response(sys, T=t)
        res_forced = forced_response(sys, T=t, U=1.0)
        np.testing.assert_allclose(res_step.y, res_forced.y, atol=1e-7)

    def test_mimo_forced_response(self) -> None:
        """Verify forced_response for a 2-input 2-output StateSpace system."""
        A = [[-1.0, 0.0], [0.0, -2.0]]
        B = [[1.0, 0.0], [0.0, 1.0]]
        C = [[1.0, 0.0], [0.0, 1.0]]
        D = [[0.0, 0.0], [0.0, 0.0]]
        sys = ss(A, B, C, D)

        t = np.linspace(0, 5.0, 100)
        # Input 1 is unit step, Input 2 is unit ramp
        U = np.column_stack([np.ones_like(t), t])

        res = forced_response(sys, T=t, U=U)
        assert res.y.shape == (100, 2)
        assert res.x is not None
        assert res.x.shape == (100, 2)

        # Output 1: 1 - exp(-t)
        np.testing.assert_allclose(res.y[:, 0], 1.0 - np.exp(-t), atol=1e-5)
        # Output 2: 0.5 * t - 0.25 + 0.25 * exp(-2t)
        y2_analytical = 0.5 * t - 0.25 + 0.25 * np.exp(-2.0 * t)
        np.testing.assert_allclose(res.y[:, 1], y2_analytical, atol=1e-5)


class TestValidationAndErrors:
    """Test validation errors for invalid arguments and mismatched dimensions."""

    def test_invalid_system_type(self) -> None:
        """Verify passing non-LTI system raises TypeError."""
        with pytest.raises(TypeError, match="LinearTimeInvariant"):
            step_response("not_a_system")  # type: ignore[arg-type]

    def test_invalid_time_vector(self) -> None:
        """Verify errors on negative duration, too short, or non-monotonic time vector."""
        sys = tf([1], [1, 1])
        with pytest.raises(ValueError, match="duration T must be positive"):
            step_response(sys, T=-5.0)

        with pytest.raises(ValueError, match="at least 2 points"):
            step_response(sys, T=[0.0])

        with pytest.raises(ValueError, match="must start at t >= 0"):
            step_response(sys, T=[-1.0, 0.0, 1.0])

        with pytest.raises(ValueError, match="strictly monotonically increasing"):
            step_response(sys, T=[0.0, 2.0, 1.0])

    def test_invalid_x0_dimensions(self) -> None:
        """Verify error when X0 length does not match StateSpace states."""
        sys = ss([[-1, 0], [0, -2]], [[1], [1]], [[1, 0]], [[0]])
        with pytest.raises(ValueError, match="Initial state X0 must have length 2"):
            step_response(sys, X0=[1.0, 2.0, 3.0])

    def test_invalid_input_index(self) -> None:
        """Verify IndexError when input_index is out of range."""
        sys = ss([[-1]], [[1]], [[1]], [[0]])
        with pytest.raises(IndexError):
            step_response(sys, input_index=2)

    def test_mismatched_u_length_in_forced_response(self) -> None:
        """Verify error when U length does not match time vector T."""
        sys = tf([1], [1, 1])
        with pytest.raises(ValueError, match="Input U length"):
            forced_response(sys, T=[0, 1, 2, 3], U=[1, 1])

    def test_additional_error_cases_and_coverage(self) -> None:
        """Verify additional error cases in impulse_response and forced_response."""
        sys_tf = tf([1], [1, 1])
        sys_ss = ss([[-1]], [[1]], [[1]], [[0]])

        # Non-LTI passed to impulse_response and forced_response
        with pytest.raises(TypeError, match="LinearTimeInvariant"):
            impulse_response("not_a_system")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="LinearTimeInvariant"):
            forced_response("not_a_system", T=[0, 1], U=[0, 1])  # type: ignore[arg-type]

        # X0 passed to TransferFunction
        with pytest.raises(ValueError, match="Initial state X0"):
            impulse_response(sys_tf, X0=[1.0])
        with pytest.raises(ValueError, match="Initial state X0"):
            forced_response(sys_tf, T=[0, 1], U=[1, 1], X0=[1.0])

        # StateSpace impulse response errors
        with pytest.raises(IndexError, match="input_index"):
            impulse_response(sys_ss, input_index=5)
        with pytest.raises(ValueError, match="Initial state X0"):
            impulse_response(sys_ss, X0=[1.0, 2.0, 3.0])

        # StateSpace forced response errors
        with pytest.raises(ValueError, match="Initial state X0"):
            forced_response(sys_ss, T=[0, 1], U=[1, 1], X0=[1.0, 2.0, 3.0])

        # Auto time vector on unstable, imaginary, and integrator poles
        sys_unstable = tf([1], [1, -2])
        res_unstable = step_response(sys_unstable)
        assert len(res_unstable.t) == 1000

        sys_imag = tf([1], [1, 0, 9])
        res_imag = step_response(sys_imag)
        assert len(res_imag.t) >= 1000

        sys_integrator = tf([1], [1, 0])
        res_int = step_response(sys_integrator)
        assert len(res_int.t) == 1000

        # Multi-input step and forced response
        A = [[-1.0, 0.0], [0.0, -2.0]]
        B = [[1.0, 0.0], [0.0, 1.0]]
        C = [[1.0, 0.0]]
        D = [[0.0, 0.0]]
        sys_miso = ss(A, B, C, D)
        res_miso_step = step_response(sys_miso, input_index=1, T=[0, 1, 2])
        assert res_miso_step.y.ndim == 1

        res_miso_imp = impulse_response(sys_miso, input_index=0, T=[0, 1, 2])
        assert res_miso_imp.y.ndim == 1

        # MIMO forced response with scalar U, 1D U matching inputs, and transposed 2D U
        t = np.linspace(0, 2, 50)
        res_scalar_u = forced_response(sys_miso, T=t, U=2.5)
        assert res_scalar_u.y.ndim == 1

        res_1d_u = forced_response(sys_miso, T=t, U=[1.0, 2.0])
        assert res_1d_u.y.ndim == 1

        with pytest.raises(ValueError, match="does not match"):
            forced_response(sys_miso, T=t, U=[1.0, 2.0, 3.0])

        U_transposed = np.ones((2, 50))
        res_transposed = forced_response(sys_miso, T=t, U=U_transposed)
        assert res_transposed.y.ndim == 1

        with pytest.raises(ValueError, match="is incompatible with T"):
            forced_response(sys_miso, T=t, U=np.ones((4, 50)))

        with pytest.raises(ValueError, match="must be 1D or 2D"):
            forced_response(sys_miso, T=t, U=np.ones((2, 50, 2)))

    def test_time_response_data_repr(self) -> None:
        """Test _repr_latex_ and _repr_markdown_ on TimeResponseData."""
        sys = tf([1], [1, 2, 1])
        res = step_response(sys, T=10.0)

        assert "$$" in res._repr_latex_()
        assert res._repr_markdown_() == res._repr_latex_()

        # Unstable system representation
        sys_unstable = tf([1], [1, -2])
        res_unstable = step_response(sys_unstable, T=2.0)
        assert "Unstable" in res_unstable._repr_latex_()
        assert "$$" in res_unstable._repr_markdown_()

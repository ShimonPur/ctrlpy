"""Unit tests for discrete-time LTI control systems and discretization methods."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import ctrlpy as cp
from ctrlpy.core.discrete import (
    DiscreteLTI,
    DiscreteTransferFunction,
    dtf,
    plot_pzmap_plotly,
)
from ctrlpy.exceptions import UnstableSystemError


class TestDiscreteTransferFunctionInit:
    """Tests for initialization, normalization, and properties of DiscreteTransferFunction."""

    def test_basic_initialization(self) -> None:
        """Test standard initialization in variable 'z'."""
        h = DiscreteTransferFunction([1.0], [1.0, -0.5], dt=0.1)
        assert isinstance(h, DiscreteLTI)
        assert isinstance(h, DiscreteTransferFunction)
        assert h.dt == 0.1
        assert h.Ts == 0.1
        assert h.is_discrete is True
        assert h.inputs == 1
        assert h.outputs == 1
        assert h.is_siso is True
        assert h.var == "z"
        np.testing.assert_allclose(h.num, [1.0])
        np.testing.assert_allclose(h.den, [1.0, -0.5])

    def test_alias_dtf(self) -> None:
        """Test convenient alias dtf."""
        h = dtf([0.2], [1.0, -0.8], dt=0.05)
        assert isinstance(h, DiscreteTransferFunction)
        assert h.dt == 0.05

    def test_z_inv_initialization(self) -> None:
        """Test initialization in variable 'z^-1'."""
        # H(z^-1) = (1 + 2 z^-1) / (1 - 0.5 z^-1) = (z + 2) / (z - 0.5)
        h = DiscreteTransferFunction([1.0, 2.0], [1.0, -0.5], dt=0.2, var="z^-1")
        assert h.var == "z^-1"
        np.testing.assert_allclose(h.num, [1.0, 2.0])
        np.testing.assert_allclose(h.den, [1.0, -0.5])
        np.testing.assert_allclose(h.poles(), [0.5])
        np.testing.assert_allclose(h.zeros(), [-2.0])

    def test_z_inv_different_degrees(self) -> None:
        """Test z^-1 representation with unequal degrees."""
        # H(z^-1) = 1 / (1 - 0.5 z^-1) = z / (z - 0.5)
        h = DiscreteTransferFunction([1.0], [1.0, -0.5], dt=0.1, var="z^-1")
        np.testing.assert_allclose(h.num, [1.0, 0.0])
        np.testing.assert_allclose(h.den, [1.0, -0.5])
        np.testing.assert_allclose(h.poles(), [0.5])
        np.testing.assert_allclose(h.zeros(), [0.0])

    def test_normalization_leading_den(self) -> None:
        """Test normalization when denominator leading coefficient != 1."""
        h = DiscreteTransferFunction([2.0, 4.0], [2.0, -1.0], dt=0.5)
        np.testing.assert_allclose(h.num, [1.0, 2.0])
        np.testing.assert_allclose(h.den, [1.0, -0.5])

    def test_leading_zeros_trimmed(self) -> None:
        """Test leading zeros are properly trimmed."""
        h = DiscreteTransferFunction([0.0, 0.0, 3.0], [0.0, 1.0, -0.4], dt=0.1)
        np.testing.assert_allclose(h.num, [3.0])
        np.testing.assert_allclose(h.den, [1.0, -0.4])

    def test_all_zero_numerator(self) -> None:
        """Test zero transfer function."""
        h = DiscreteTransferFunction([0.0], [1.0, -0.5], dt=0.1)
        np.testing.assert_allclose(h.num, [0.0])
        assert len(h.zeros()) == 0

    def test_invalid_init_exceptions(self) -> None:
        """Test ValueError exceptions for invalid arguments."""
        with pytest.raises(ValueError, match="strictly positive"):
            DiscreteTransferFunction([1.0], [1.0, -0.5], dt=0.0)

        with pytest.raises(ValueError, match="strictly positive"):
            DiscreteTransferFunction([1.0], [1.0, -0.5], dt=-0.1)

        with pytest.raises(ValueError, match="Numerator cannot be empty"):
            DiscreteTransferFunction([], [1.0, -0.5], dt=0.1)

        with pytest.raises(ValueError, match="Denominator cannot be empty"):
            DiscreteTransferFunction([1.0], [], dt=0.1)

        with pytest.raises(ValueError, match="identically zero"):
            DiscreteTransferFunction([1.0], [0.0, 0.0], dt=0.1)

        with pytest.raises(ValueError, match="Unsupported discrete variable symbol"):
            DiscreteTransferFunction([1.0], [1.0, -0.5], dt=0.1, var="s")

    def test_dcgain(self) -> None:
        """Test discrete DC gain H(1)."""
        # H(z) = 0.5 / (z - 0.5) -> H(1) = 0.5 / (1 - 0.5) = 1.0
        h1 = DiscreteTransferFunction([0.5], [1.0, -0.5], dt=0.1)
        assert np.isclose(h1.dcgain(), 1.0)

        # Integrator H(z) = 1 / (z - 1) -> H(1) = inf
        h2 = DiscreteTransferFunction([1.0], [1.0, -1.0], dt=0.1)
        assert np.isinf(h2.dcgain())


class TestUnitCircleStability:
    """Tests for unit-circle stability classification."""

    def test_strictly_stable_systems(self) -> None:
        """Poles strictly inside the unit circle |p| < 1."""
        h1 = DiscreteTransferFunction([1.0], [1.0, -0.5], dt=0.1)
        assert h1.is_stable() is True
        assert h1.is_marginally_stable() is False
        assert h1.stability() == "stable"

        # Complex conjugate stable poles: 0.4 +/- 0.5j (mag = sqrt(0.16 + 0.25) = 0.64 < 1)
        # (z - 0.4 - 0.5j)(z - 0.4 + 0.5j) = z^2 - 0.8 z + 0.41
        h2 = DiscreteTransferFunction([1.0], [1.0, -0.8, 0.41], dt=0.1)
        assert h2.is_stable() is True
        assert h2.stability() == "stable"

    def test_marginally_stable_systems(self) -> None:
        """Poles on the unit circle |p| = 1 with multiplicity 1."""
        # Single pole at z = 1 (integrator)
        h_int = DiscreteTransferFunction([1.0], [1.0, -1.0], dt=0.1)
        assert h_int.is_stable() is False
        assert h_int.is_marginally_stable() is True
        assert h_int.stability() == "marginally stable"

        # Poles at z = +/- j (pure oscillator: z^2 + 1 = 0)
        h_osc = DiscreteTransferFunction([1.0], [1.0, 0.0, 1.0], dt=0.1)
        assert h_osc.is_stable() is False
        assert h_osc.is_marginally_stable() is True
        assert h_osc.stability() == "marginally stable"

    def test_unstable_systems_outside_circle(self) -> None:
        """Poles outside unit circle |p| > 1."""
        h_unstable = DiscreteTransferFunction([1.0], [1.0, -1.2], dt=0.1)
        assert h_unstable.is_stable() is False
        assert h_unstable.is_marginally_stable() is False
        assert h_unstable.stability() == "unstable"

    def test_unstable_systems_repeated_unit_circle(self) -> None:
        """Multiple poles on unit circle cause unbounded growth (unstable)."""
        # Double pole at z = 1: (z - 1)^2 = z^2 - 2z + 1
        h_double_int = DiscreteTransferFunction([1.0], [1.0, -2.0, 1.0], dt=0.1)
        assert h_double_int.is_stable() is False
        assert h_double_int.is_marginally_stable() is False
        assert h_double_int.stability() == "unstable"

    def test_static_gain_stability(self) -> None:
        """Static gain transfer function with no poles."""
        h_gain = DiscreteTransferFunction([5.0], [1.0], dt=0.1)
        assert h_gain.is_stable() is True
        assert h_gain.stability() == "stable"


class TestDiscretizationMethods:
    """Numerical verification of continuous-to-discrete conversion methods."""

    def test_zoh_first_order_lag(self) -> None:
        """Verify ZOH discretization of G(s) = a / (s + a).

        Analytical: H(z) = (1 - e^(-a Ts)) / (z - e^(-a Ts))
        """
        a = 2.0
        Ts = 0.1
        G = cp.tf([a], [1.0, a])
        H = cp.c2d(G, dt=Ts, method="zoh")

        expected_pole = np.exp(-a * Ts)
        expected_num = 1.0 - expected_pole

        np.testing.assert_allclose(H.poles(), [expected_pole], rtol=1e-5)
        np.testing.assert_allclose(H.num, [expected_num], rtol=1e-5)
        np.testing.assert_allclose(H.den, [1.0, -expected_pole], rtol=1e-5)
        assert np.isclose(H.dcgain(), 1.0)

    def test_zoh_integrator(self) -> None:
        """Verify ZOH discretization of G(s) = 1/s.

        Analytical: H(z) = Ts / (z - 1)
        """
        Ts = 0.2
        G = cp.tf([1.0], [1.0, 0.0])
        H = cp.c2d(G, dt=Ts, method="zoh")

        np.testing.assert_allclose(H.num, [Ts], rtol=1e-5)
        np.testing.assert_allclose(H.den, [1.0, -1.0], rtol=1e-5)

    def test_tustin_first_order_lag(self) -> None:
        """Verify Tustin (Bilinear) discretization of G(s) = 1 / (s + 1).

        Substitution: s = (2/Ts) * (z - 1)/(z + 1)
        H(z) = (Ts (z + 1)) / ((2 + Ts) z + (Ts - 2))
        """
        Ts = 0.1
        G = cp.tf([1.0], [1.0, 1.0])
        H = cp.c2d(G, dt=Ts, method="tustin")

        expected_pole = (2.0 - Ts) / (2.0 + Ts)
        np.testing.assert_allclose(H.poles(), [expected_pole], rtol=1e-5)
        np.testing.assert_allclose(H.zeros(), [-1.0], rtol=1e-5)
        assert np.isclose(H.dcgain(), 1.0)

    def test_tustin_with_prewarping(self) -> None:
        """Verify Tustin with frequency pre-warping matches continuous response at w_warp."""
        wn = 5.0
        zeta = 0.3
        Ts = 0.05
        # Underdamped second-order: G(s) = wn^2 / (s^2 + 2*zeta*wn*s + wn^2)
        G = cp.tf([wn**2], [1.0, 2.0 * zeta * wn, wn**2])

        # Pre-warp at natural frequency wn
        H_prewarp = cp.c2d(G, dt=Ts, method="tustin", prewarp_frequency=wn)

        # Continuous response at s = j*wn
        s_target = 1j * wn
        G_cont_val = np.polyval(G.num, s_target) / np.polyval(G.den, s_target)

        # Discrete response at z = exp(j * wn * Ts)
        z_target = np.exp(1j * wn * Ts)
        H_disc_val = np.polyval(H_prewarp.num, z_target) / np.polyval(H_prewarp.den, z_target)

        # Must match exactly at pre-warped frequency
        np.testing.assert_allclose(np.abs(H_disc_val), np.abs(G_cont_val), rtol=1e-5)
        np.testing.assert_allclose(np.angle(H_disc_val), np.angle(G_cont_val), rtol=1e-5)

    def test_foh_discretization(self) -> None:
        """Verify First-Order Hold (FOH) discretization."""
        Ts = 0.1
        G = cp.tf([1.0], [1.0, 2.0])
        H = cp.c2d(G, dt=Ts, method="foh")
        assert isinstance(H, DiscreteTransferFunction)
        assert np.isclose(H.dcgain(), 0.5)

    def test_matched_pole_zero_method(self) -> None:
        """Verify Matched pole-zero mapping method."""
        p1, p2 = -1.0, -3.0
        z1 = -2.0
        Ts = 0.1
        # G(s) = (s + 2) / ((s + 1)(s + 3))
        G = cp.tf([1.0, -z1], [1.0, -(p1 + p2), p1 * p2])
        H = cp.c2d(G, dt=Ts, method="matched")

        # Poles should map directly to e^(p_i * Ts)
        expected_poles = np.sort([np.exp(p1 * Ts), np.exp(p2 * Ts)])
        actual_poles = np.sort(np.real(H.poles()))
        np.testing.assert_allclose(actual_poles, expected_poles, rtol=1e-5)

    def test_c2d_on_system_methods(self) -> None:
        """Test fluent OOP methods .c2d() and .to_discrete() on TransferFunction and StateSpace."""
        G = cp.tf([1.0], [1.0, 1.0])
        H1 = G.c2d(0.1, method="zoh")
        H2 = G.to_discrete(0.1, method="zoh")
        np.testing.assert_allclose(H1.num, H2.num)
        np.testing.assert_allclose(H1.den, H2.den)

        # StateSpace c2d
        sys_ss = G.to_ss()
        H_ss = sys_ss.c2d(0.1, method="zoh")
        np.testing.assert_allclose(H_ss.den, H1.den, rtol=1e-5)

    def test_c2d_invalid_inputs(self) -> None:
        """Test c2d raises errors for invalid parameters."""
        G = cp.tf([1.0], [1.0, 1.0])
        with pytest.raises(ValueError, match="strictly positive"):
            cp.c2d(G, dt=-0.1)

        with pytest.raises(ValueError, match="Unknown discretization method"):
            cp.c2d(G, dt=0.1, method="invalid_method")  # type: ignore

        with pytest.raises(ValueError, match="prewarp_frequency"):
            cp.c2d(G, dt=0.1, method="prewarping")

        with pytest.raises(ValueError, match="strictly positive"):
            cp.c2d(G, dt=0.1, method="tustin", prewarp_frequency=-5.0)

        with pytest.raises(ValueError, match="Nyquist frequency"):
            # dt = 0.1 -> Nyquist freq = pi / 0.1 = 31.4159 rad/s
            cp.c2d(G, dt=0.1, method="tustin", prewarp_frequency=40.0)

        with pytest.raises(TypeError, match="Expected continuous"):
            cp.c2d("invalid_sys", dt=0.1)  # type: ignore


class TestDiscreteTimeDomainSimulation:
    """Tests for discrete step, impulse, and forced difference equation simulations."""

    def test_step_response_analytical(self) -> None:
        """Verify step response against analytical difference equation solution.

        H(z) = (1 - a) / (z - a) = (1 - a) z^-1 / (1 - a z^-1)
        y[k] = a y[k-1] + (1 - a) u[k-1]
        For unit step u[k] = 1 (k >= 0):
        y[0] = 0
        y[1] = 1 - a
        y[2] = 1 - a^2
        y[k] = 1 - a^k
        """
        a = 0.5
        dt = 0.1
        H = DiscreteTransferFunction([1.0 - a], [1.0, -a], dt=dt)
        res = cp.discrete_step_response(H, n_steps=20)

        assert len(res.t) == 20
        assert np.isclose(res.t[0], 0.0)
        assert np.isclose(res.t[1], dt)

        k_vec = np.arange(20)
        expected_y = np.where(k_vec == 0, 0.0, 1.0 - a**k_vec)
        np.testing.assert_allclose(res.y, expected_y, atol=1e-12)

    def test_impulse_response_analytical(self) -> None:
        """Verify impulse response against analytical solution.

        H(z) = b / (z - a) -> y[k] = b * a^(k-1) for k >= 1, y[0] = 0
        """
        a = 0.6
        b = 0.4
        dt = 0.05
        H = DiscreteTransferFunction([b], [1.0, -a], dt=dt)
        res = cp.discrete_impulse_response(H, n_steps=15)

        k_vec = np.arange(15)
        expected_y = np.where(k_vec == 0, 0.0, b * (a ** (k_vec - 1)))
        np.testing.assert_allclose(res.y, expected_y, atol=1e-12)

    def test_forced_response(self) -> None:
        """Verify arbitrary input forced response."""
        H = DiscreteTransferFunction([0.2], [1.0, -0.8], dt=0.1)
        u = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        res = cp.discrete_forced_response(H, U=u)
        assert len(res.t) == 5
        assert len(res.y) == 5
        # y[0] = 0
        # y[1] = 0.8*0 + 0.2*1 = 0.2
        # y[2] = 0.8*0.2 + 0.2*2 = 0.16 + 0.4 = 0.56
        assert np.isclose(res.y[0], 0.0)
        assert np.isclose(res.y[1], 0.2)
        assert np.isclose(res.y[2], 0.56)

    def test_transient_metrics_extraction(self) -> None:
        """Verify transient response metrics on discrete step response."""
        H = DiscreteTransferFunction([0.2], [1.0, -0.8], dt=0.1)
        res = H.step(n_steps=100)

        yss = res.steady_state_value()
        assert np.isclose(yss, 1.0, atol=1e-4)

        tr = res.rise_time()
        assert tr > 0.0

        ts = res.settling_time()
        assert ts > 0.0

        os_val = res.overshoot()
        assert np.isclose(os_val, 0.0)  # Monotonic first-order has 0% overshoot

    def test_unstable_system_metrics_raise(self) -> None:
        """Unstable discrete system raises UnstableSystemError on transient metrics."""
        H_unstable = DiscreteTransferFunction([1.0], [1.0, -1.5], dt=0.1)
        res = H_unstable.step(n_steps=20)
        with pytest.raises(UnstableSystemError, match="outside the unit circle"):
            res.steady_state_value()

    def test_improper_transfer_function_simulation_error(self) -> None:
        """Improper discrete transfer function (num deg > den deg) raises ValueError."""
        H_improper = DiscreteTransferFunction([1.0, 2.0, 3.0], [1.0, -0.5], dt=0.1)
        with pytest.raises(ValueError, match="cannot be simulated forward"):
            H_improper.step()


class TestDiscreteFrequencyResponse:
    """Tests for discrete frequency response and Bode analysis."""

    def test_bode_data_range(self) -> None:
        """Verify frequency response evaluated up to Nyquist limit pi/Ts."""
        dt = 0.05
        H = DiscreteTransferFunction([0.1], [1.0, -0.9], dt=dt)
        bdata = H.bode(n_points=200)

        assert len(bdata.w) == 200
        assert np.isclose(bdata.w[-1], np.pi / dt, rtol=1e-4)
        assert len(bdata.mag) == 200
        assert len(bdata.phase) == 200
        assert len(bdata.mag_db) == 200

        # DC magnitude check (w -> 0)
        assert np.isclose(bdata.mag[0], H.dcgain(), rtol=1e-2)

    def test_freqresp_custom_omega(self) -> None:
        """Test evaluation at explicit user frequencies."""
        H = DiscreteTransferFunction([0.2], [1.0, -0.8], dt=0.1)
        omega = np.array([0.1, 1.0, 5.0, 10.0])
        w_out, resp = H.freqresp(omega=omega)

        np.testing.assert_allclose(w_out, omega)
        # Analytical at w: H(e^(j*w*dt)) = 0.2 / (e^(j*w*dt) - 0.8)
        expected_resp = 0.2 / (np.exp(1j * omega * 0.1) - 0.8)
        np.testing.assert_allclose(resp, expected_resp, rtol=1e-5)


class TestDiscreteAlgebra:
    """Tests for algebraic operations (+, -, *, /, feedback) on discrete systems."""

    def test_addition(self) -> None:
        """Test parallel addition."""
        h1 = DiscreteTransferFunction([1.0], [1.0, -0.5], dt=0.1)
        h2 = DiscreteTransferFunction([2.0], [1.0, -0.2], dt=0.1)
        h_sum = h1 + h2
        # (1*(z-0.2) + 2*(z-0.5)) / ((z-0.5)(z-0.2)) = (3z - 1.2) / (z^2 - 0.7z + 0.1)
        np.testing.assert_allclose(h_sum.num, [3.0, -1.2])
        np.testing.assert_allclose(h_sum.den, [1.0, -0.7, 0.1])

    def test_scalar_addition(self) -> None:
        """Test adding scalar gain."""
        h = DiscreteTransferFunction([1.0], [1.0, -0.5], dt=0.1)
        h_add = h + 2.0
        # 1 / (z-0.5) + 2 = (2z) / (z - 0.5)
        np.testing.assert_allclose(h_add.num, [2.0, 0.0])
        np.testing.assert_allclose(h_add.den, [1.0, -0.5])

    def test_multiplication(self) -> None:
        """Test series multiplication."""
        h1 = DiscreteTransferFunction([1.0], [1.0, -0.5], dt=0.1)
        h2 = DiscreteTransferFunction([2.0, 1.0], [1.0, -0.2], dt=0.1)
        h_prod = h1 * h2
        np.testing.assert_allclose(h_prod.num, [2.0, 1.0])
        np.testing.assert_allclose(h_prod.den, [1.0, -0.7, 0.1])

    def test_feedback(self) -> None:
        """Test closed-loop unity feedback."""
        h = DiscreteTransferFunction([0.5], [1.0, -0.5], dt=0.1)
        t_closed = h.feedback(1.0)
        # T(z) = 0.5 / ((z - 0.5) + 0.5) = 0.5 / z
        np.testing.assert_allclose(t_closed.num, [0.5])
        np.testing.assert_allclose(t_closed.den, [1.0, 0.0])
        np.testing.assert_allclose(t_closed.poles(), [0.0])

    def test_sampling_time_mismatch_raises(self) -> None:
        """Attempting to combine discrete systems with different dt raises ValueError."""
        h1 = DiscreteTransferFunction([1.0], [1.0, -0.5], dt=0.1)
        h2 = DiscreteTransferFunction([1.0], [1.0, -0.5], dt=0.2)
        with pytest.raises(ValueError, match="different sampling times"):
            _ = h1 + h2
        with pytest.raises(ValueError, match="different sampling times"):
            _ = h1 * h2
        with pytest.raises(ValueError, match="different sampling times"):
            _ = h1.feedback(h2)

    def test_combining_with_continuous_raises(self) -> None:
        """Attempting to combine discrete and continuous systems raises TypeError."""
        h_disc = DiscreteTransferFunction([1.0], [1.0, -0.5], dt=0.1)
        g_cont = cp.tf([1.0], [1.0, 1.0])
        with pytest.raises(TypeError, match="Discretize"):
            _ = h_disc + g_cont


class TestJupyterRenderingAndPlotting:
    """Tests for LaTeX string representations and pole-zero plotting."""

    def test_repr_latex(self) -> None:
        """Test _repr_latex_ includes H(z) and sampling time Ts."""
        h = DiscreteTransferFunction([0.05, 0.04], [1.0, -1.6, 0.64], dt=0.1)
        latex_str = h._repr_latex_()
        assert r"H(z)" in latex_str
        assert r"T_s" in latex_str
        assert "0.1" in latex_str

        # z^-1 LaTeX representation
        h_inv = DiscreteTransferFunction([1.0, 0.5], [1.0, -0.8], dt=0.05, var="z^-1")
        latex_inv = h_inv._repr_latex_()
        assert r"H(z^{-1})" in latex_inv

    def test_str_and_repr(self) -> None:
        """Test __str__ and __repr__ formatting."""
        h = DiscreteTransferFunction([1.0], [1.0, -0.5], dt=0.1)
        assert "DiscreteTransferFunction" in repr(h)
        assert "Ts = 0.1 s" in str(h)

    def test_plot_pzmap_matplotlib(self) -> None:
        """Test Matplotlib pole-zero map rendering."""
        h = DiscreteTransferFunction([1.0, -0.5], [1.0, -0.8, 0.64], dt=0.1)
        fig, ax = cp.plot_pzmap(h)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_iplot_pzmap_plotly(self) -> None:
        """Test Plotly interactive pole-zero map rendering."""
        h = DiscreteTransferFunction([1.0, -0.5], [1.0, -0.8, 0.64], dt=0.1)
        fig_plotly = cp.iplot_pzmap(h)
        assert isinstance(fig_plotly, go.Figure)
        fig_alias = plot_pzmap_plotly(h)
        assert isinstance(fig_alias, go.Figure)

    def test_discrete_plot_step_matplotlib_and_plotly(self) -> None:
        """Test .plot_step() on DiscreteTransferFunction."""
        h = DiscreteTransferFunction([0.2], [1.0, -0.8], dt=0.1)
        fig_mpl, _ = h.plot_step(backend="matplotlib")
        assert isinstance(fig_mpl, plt.Figure)
        plt.close(fig_mpl)

        # Plot with existing axis
        fig_custom, ax_custom = plt.subplots()
        _, ax_ret = h.plot_step(backend="matplotlib", ax=ax_custom)
        assert ax_ret is ax_custom
        plt.close(fig_custom)

        fig_plotly = h.plot_step(backend="plotly")
        assert isinstance(fig_plotly, go.Figure)

        with pytest.raises(ValueError, match="Unknown backend"):
            h.plot_step(backend="invalid")  # type: ignore

    def test_repr_markdown(self) -> None:
        """Test _repr_markdown_."""
        h = DiscreteTransferFunction([1.0], [1.0, -0.5], dt=0.1)
        assert h._repr_markdown_() == h._repr_latex_()

    def test_poly_formatting_edge_cases(self) -> None:
        """Test polynomial string and LaTeX formatting edge cases."""
        from ctrlpy.core.discrete import (
            _format_discrete_poly_latex,
            _format_discrete_poly_str,
        )

        assert _format_discrete_poly_str(np.array([0.0]), var="z") == "0"
        assert _format_discrete_poly_latex(np.array([0.0]), var="z") == "0"
        assert _format_discrete_poly_str(np.array([0.0]), var="z^-1") == "0"
        assert _format_discrete_poly_latex(np.array([0.0]), var="z^-1") == "0"

        # Negative and non-integer coefficients
        coeffs = np.array([-2.5, 0.0, 1.25])
        str_z = _format_discrete_poly_str(coeffs, var="z")
        assert "-2.5 z^2" in str_z
        assert "+ 1.25" in str_z

        tex_z = _format_discrete_poly_latex(coeffs, var="z")
        assert "-2.5 z^{2}" in tex_z

        str_inv = _format_discrete_poly_str(coeffs, var="z^-1")
        assert "-2.5" in str_inv
        assert "+ 1.25 z^-2" in str_inv

        tex_inv = _format_discrete_poly_latex(coeffs, var="z^-1")
        assert "z^{-2}" in tex_inv


class TestDiscreteAlgebraCoverage:
    """Additional coverage for arithmetic operators."""

    def test_subtraction(self) -> None:
        """Test subtraction operators."""
        h1 = DiscreteTransferFunction([1.0], [1.0, -0.5], dt=0.1)
        h2 = DiscreteTransferFunction([0.5], [1.0, -0.5], dt=0.1)
        diff = h1 - h2
        # (1*(z-0.5) - 0.5*(z-0.5)) / (z-0.5)^2 = (0.5z - 0.25) / (z^2 - z + 0.25)
        np.testing.assert_allclose(diff.num, [0.5, -0.25])
        np.testing.assert_allclose(diff.den, [1.0, -1.0, 0.25])

        # Scalar subtraction
        diff_scalar = h1 - 1.0
        # 1 / (z-0.5) - 1 = (1 - z + 0.5) / (z-0.5) = (-z + 1.5) / (z-0.5)
        np.testing.assert_allclose(diff_scalar.num, [-1.0, 1.5])

        # rsub
        rsub = 2.0 - h1
        # 2 - 1/(z-0.5) = (2z - 2) / (z-0.5)
        np.testing.assert_allclose(rsub.num, [2.0, -2.0])

        with pytest.raises(TypeError, match="Cannot subtract continuous"):
            _ = h1 - cp.tf([1.0], [1.0, 1.0])

        with pytest.raises(ValueError, match="different sampling times"):
            _ = h1 - DiscreteTransferFunction([1.0], [1.0, -0.5], dt=0.2)

    def test_multiplication_and_division(self) -> None:
        """Test multiplication and division operations."""
        h1 = DiscreteTransferFunction([1.0], [1.0, -0.5], dt=0.1)
        h2 = DiscreteTransferFunction([2.0], [1.0, -0.3], dt=0.1)

        # Unary pos and neg
        h_pos = +h1
        np.testing.assert_allclose(h_pos.num, h1.num)
        h_neg = -h1
        np.testing.assert_allclose(h_neg.num, -h1.num)

        # Scalar multiplication and rmul
        h_mul_scalar = h1 * 3.0
        np.testing.assert_allclose(h_mul_scalar.num, [3.0])
        h_rmul = 4.0 * h1
        np.testing.assert_allclose(h_rmul.num, [4.0])

        with pytest.raises(TypeError, match="Cannot multiply discrete"):
            _ = h1 * cp.tf([1.0], [1.0, 1.0])

        # Division
        h_div = h1 / h2
        # (1/(z-0.5)) / (2/(z-0.3)) = (z - 0.3) / (2z - 1.0) = (0.5z - 0.15) / (z - 0.5)
        np.testing.assert_allclose(h_div.poles(), [0.5])
        np.testing.assert_allclose(h_div.zeros(), [0.3])

        # Scalar division and rtruediv
        h_div_scalar = h1 / 2.0
        np.testing.assert_allclose(h_div_scalar.num, [0.5])

        h_rdiv = 2.0 / h1
        # 2 / (1 / (z-0.5)) = 2z - 1
        np.testing.assert_allclose(h_rdiv.num, [2.0, -1.0])

        with pytest.raises(ZeroDivisionError):
            _ = h1 / 0.0

        with pytest.raises(ValueError, match="different sampling times"):
            _ = h1 / DiscreteTransferFunction([1.0], [1.0, -0.5], dt=0.2)

    def test_feedback_errors(self) -> None:
        """Test feedback type errors."""
        h = DiscreteTransferFunction([1.0], [1.0, -0.5], dt=0.1)
        with pytest.raises(TypeError, match="Unsupported feedback block type"):
            h.feedback("invalid")  # type: ignore


class TestDiscreteSimulationAndMethodsCoverage:
    """Coverage for simulation options, frequency domain, and plotting dispatch."""

    def test_step_duration_and_array(self) -> None:
        """Test step simulation with float duration and array T."""
        h = DiscreteTransferFunction([0.2], [1.0, -0.8], dt=0.1)
        res_dur = h.step(T=2.0)
        assert np.isclose(res_dur.t[-1], 2.0)

        t_custom = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        res_arr = h.step(T=t_custom)
        np.testing.assert_allclose(res_arr.t, t_custom)

        # Automatic step simulation with no T and no n_steps
        res_auto = h.step()
        assert len(res_auto.t) >= 30

        with pytest.raises(ValueError, match="Duration T must be positive"):
            h.step(T=-1.0)

        with pytest.raises(ValueError, match="n_steps must be at least 2"):
            h.step(n_steps=1)

        with pytest.raises(ValueError, match="Time vector T must contain at least 2 points"):
            h.step(T=[0.0])

    def test_impulse_duration_and_array(self) -> None:
        """Test impulse simulation with duration, array, and automatic horizon."""
        h = DiscreteTransferFunction([0.5], [1.0, -0.5], dt=0.1)
        res_dur = h.impulse(T=1.0)
        assert np.isclose(res_dur.t[-1], 1.0)

        res_arr = h.impulse(T=[0.0, 0.1, 0.2, 0.3])
        assert len(res_arr.t) == 4

        res_auto = h.impulse()
        assert len(res_auto.t) >= 30

        # Unstable impulse auto horizon
        h_unstable = DiscreteTransferFunction([1.0], [1.0, -1.5], dt=0.1)
        res_unstable = h_unstable.impulse()
        assert len(res_unstable.t) == 40

        # Marginally stable auto horizon
        h_marg = DiscreteTransferFunction([1.0], [1.0, -1.0], dt=0.1)
        res_marg = h_marg.impulse()
        assert len(res_marg.t) == 50

        with pytest.raises(ValueError, match="Duration T must be positive"):
            h.impulse(T=0.0)

        with pytest.raises(ValueError, match="n_steps must be at least 2"):
            h.impulse(n_steps=0)

        with pytest.raises(ValueError, match="Time vector T must contain at least 2 points"):
            h.impulse(T=[0.0])

        h_improper = DiscreteTransferFunction([1.0, 2.0, 3.0], [1.0, -0.5], dt=0.1)
        with pytest.raises(ValueError, match="cannot be simulated"):
            h_improper.impulse()

    def test_forced_response_options(self) -> None:
        """Test forced response with scalar U, explicit T, n_steps, and error conditions."""
        h = DiscreteTransferFunction([0.2], [1.0, -0.8], dt=0.1)

        # Scalar U with n_steps
        res_scalar_n = h.forced_response(U=2.0, n_steps=50)
        assert len(res_scalar_n.t) == 50
        assert np.allclose(res_scalar_n.y[-1], 2.0 * h.dcgain(), atol=0.01)

        # Scalar U with explicit T
        res_scalar_t = h.forced_response(U=1.5, T=np.linspace(0.0, 1.0, 11))
        assert len(res_scalar_t.t) == 11

        # Scalar U with default
        res_scalar_def = h.forced_response(U=1.0)
        assert len(res_scalar_def.t) == 50

        # Mismatched lengths
        with pytest.raises(ValueError, match="does not match time vector"):
            h.forced_response(U=[1.0, 2.0], T=[0.0, 0.1, 0.2, 0.3])

        h_improper = DiscreteTransferFunction([1.0, 2.0, 3.0], [1.0, -0.5], dt=0.1)
        with pytest.raises(ValueError, match="Improper discrete transfer function"):
            h_improper.forced_response(U=[1.0, 2.0])

    def test_bode_errors_and_dispatch(self) -> None:
        """Test discrete_bode_data errors."""
        h = DiscreteTransferFunction([0.2], [1.0, -0.8], dt=0.1)
        with pytest.raises(ValueError, match="must contain at least 2 points"):
            h.bode(omega=[1.0])

        with pytest.raises(ValueError, match="strictly positive"):
            h.bode(omega=[-1.0, 2.0])

    def test_plot_pzmap_backend_and_options(self) -> None:
        """Test plot_pzmap with backend dispatch and options."""
        h = DiscreteTransferFunction([1.0, -0.5], [1.0, -0.8, 0.64], dt=0.1)

        fig_plotly = h.plot_pzmap(backend="plotly", title="Custom Plotly Map")
        assert isinstance(fig_plotly, go.Figure)

        fig_mpl, _ = h.plot_pzmap(backend="matplotlib", title="Custom Matplotlib Map")
        assert isinstance(fig_mpl, plt.Figure)
        plt.close(fig_mpl)

        with pytest.raises(ValueError, match="Unknown backend"):
            h.plot_pzmap(backend="seaborn")  # type: ignore

    def test_matched_c2d_with_origin_pole(self) -> None:
        """Test matched pole-zero conversion for continuous system with integrator pole at s=0."""
        # G(s) = 1 / s -> integrator
        G_int = cp.tf([1.0], [1.0, 0.0])
        H_int = cp.c2d(G_int, dt=0.1, method="matched")
        assert isinstance(H_int, DiscreteTransferFunction)
        np.testing.assert_allclose(H_int.poles(), [1.0], atol=1e-5)

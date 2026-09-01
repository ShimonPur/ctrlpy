"""Unit tests for frequency-domain analysis, stability margins, and plotting."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Use non-interactive backend for testing
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ctrlpy.freq_domain import (
    BodeData,
    NyquistData,
    RootLocusData,
    StabilityMargins,
    bode_data,
    margin,
    nyquist_data,
    root_locus_data,
)
from ctrlpy.models.state_space import StateSpace
from ctrlpy.models.transfer_function import tf
from ctrlpy.plotting import (
    plot_bode,
    plot_nyquist,
    plot_root_locus,
    plot_step,
)


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    """Ensure all matplotlib figures are closed after each test."""
    yield
    plt.close("all")


class TestBodeData:
    """Test suite for bode_data calculation."""

    def test_integrator_slope_and_phase(self) -> None:
        """Test Bode calculation of 1/s: slope must be -20 dB/dec and phase -90 deg."""
        sys = tf(1.0, [1.0, 0.0])  # G(s) = 1/s

        w = np.array([0.1, 1.0, 10.0, 100.0])
        bdata = bode_data(sys, omega=w)

        assert isinstance(bdata, BodeData)
        assert np.allclose(bdata.w, w)

        # Expected linear magnitudes: 10, 1, 0.1, 0.01
        assert np.allclose(bdata.mag, 1.0 / w)

        # Expected dB magnitudes: 20, 0, -20, -40 dB
        expected_db = np.array([20.0, 0.0, -20.0, -40.0])
        assert np.allclose(bdata.mag_db, expected_db, atol=1e-6)

        # Check slope: -20 dB per decade
        slopes = np.diff(bdata.mag_db) / np.diff(np.log10(bdata.w))
        assert np.allclose(slopes, -20.0, atol=1e-6)

        # Phase must be exactly -90 degrees
        assert np.allclose(bdata.phase, -90.0, atol=1e-6)
        assert np.allclose(bdata.phase_rad, -np.pi / 2.0, atol=1e-6)

    def test_first_order_system(self) -> None:
        """Test Bode response of first-order lag G(s) = 1 / (s + 1)."""
        sys = tf(1.0, [1.0, 1.0])

        w = np.array([0.01, 1.0, 100.0])
        bdata = bode_data(sys, omega=w)

        # At w = 1 rad/s (cutoff frequency):
        # |G(j1)| = 1 / sqrt(2) -> -3.0103 dB, phase = -45 deg
        assert np.isclose(bdata.mag[1], 1.0 / np.sqrt(2.0), atol=1e-5)
        assert np.isclose(bdata.mag_db[1], -3.0102999566, atol=1e-4)
        assert np.isclose(bdata.phase[1], -45.0, atol=1e-4)

        # At low frequency (w = 0.01): |G| ~ 1 (0 dB), phase ~ 0 deg
        assert np.isclose(bdata.mag_db[0], 0.0, atol=0.01)
        assert np.isclose(bdata.phase[0], 0.0, atol=1.0)

        # At high frequency (w = 100): phase ~ -90 deg
        assert np.isclose(bdata.phase[2], -90.0, atol=1.0)

    def test_auto_omega_generation(self) -> None:
        """Test automatic frequency vector generation when omega is None."""
        sys = tf([10.0], [1.0, 2.0, 10.0])
        bdata = bode_data(sys)

        assert len(bdata.w) == 1000
        assert bdata.w[0] > 0.0
        assert bdata.w[-1] > bdata.w[0]
        assert np.all(np.diff(bdata.w) > 0.0)

    def test_state_space_and_lti_method(self) -> None:
        """Test Bode calculation on StateSpace models and sys.bode() method."""
        sys_tf = tf([1.0], [1.0, 3.0, 2.0])
        sys_ss = sys_tf.to_ss()

        w = np.logspace(-1, 2, 50)
        data_tf = sys_tf.bode(omega=w)
        data_ss = sys_ss.bode(omega=w)

        assert np.allclose(data_tf.mag_db, data_ss.mag_db, atol=1e-5)
        assert np.allclose(data_tf.phase, data_ss.phase, atol=1e-5)

    def test_unpacking(self) -> None:
        """Test tuple unpacking of BodeData."""
        sys = tf(1.0, [1.0, 1.0])
        w, mag, phase = bode_data(sys, omega=[0.1, 1.0, 10.0])
        assert len(w) == 3
        assert len(mag) == 3
        assert len(phase) == 3

    def test_invalid_omega_validation(self) -> None:
        """Test error handling for invalid omega inputs."""
        sys = tf(1.0, [1.0, 1.0])

        # Less than 2 points
        with pytest.raises(ValueError, match="at least 2 points"):
            bode_data(sys, omega=[1.0])

        # Non-positive frequency
        with pytest.raises(ValueError, match="strictly positive"):
            bode_data(sys, omega=[0.0, 1.0, 2.0])

        # Non-monotonic
        with pytest.raises(ValueError, match="monotonically increasing"):
            bode_data(sys, omega=[2.0, 1.0, 3.0])

    def test_mimo_error(self) -> None:
        """Test error when non-SISO system is passed to bode_data."""
        mimo_ss = StateSpace(
            A=[[0, 1], [-2, -3]],
            B=[[1, 0], [0, 1]],
            C=[[1, 0]],
            D=[[0, 0]],
        )
        with pytest.raises(ValueError, match="requires a SISO system"):
            bode_data(mimo_ss)


class TestStabilityMargins:
    """Test suite for stability margin calculations."""

    def test_second_order_system_analytical_margins(self) -> None:
        """Test stability margins of standard second-order system against known analytical values.

        For open-loop G(s) = wn^2 / (s * (s + 2*zeta*wn)):
        wn = 10, zeta = 0.5 (wn^2 = 100, 2*zeta*wn = 10)
        Analytical Wcg = wn * sqrt(sqrt(4*zeta^4 + 1) - 2*zeta^2)
                       = 10 * sqrt(sqrt(1.25) - 0.5) = 7.86151377757 rad/s
        Analytical PM  = arctan(2*zeta*wn / Wcg) = 51.827292 deg
        Phase never reaches -180 deg for finite w -> Wcp = NaN, GM = inf dB
        """
        wn = 10.0
        zeta = 0.5
        sys = tf([wn**2], [1.0, 2.0 * zeta * wn, 0.0])

        sm = margin(sys)
        assert isinstance(sm, StabilityMargins)

        expected_wcg = wn * np.sqrt(np.sqrt(4.0 * zeta**4 + 1.0) - 2.0 * zeta**2)
        expected_pm = np.rad2deg(np.arctan((2.0 * zeta * wn) / expected_wcg))

        assert np.isclose(sm.wcg, expected_wcg, rtol=1e-4)
        assert np.isclose(sm.pm_deg, expected_pm, atol=0.05)
        assert np.isnan(sm.wcp)
        assert np.isinf(sm.gm_db)

        # Test alias properties and unpacking
        assert sm.gm == sm.gm_db
        assert sm.pm == sm.pm_deg
        assert sm.Wcg == sm.wcg
        assert np.isnan(sm.Wcp) and np.isnan(sm.wcp)

        gm_val, pm_val, wcg_val, wcp_val = sm
        assert np.isinf(gm_val)
        assert np.isclose(pm_val, expected_pm, atol=0.05)
        assert np.isclose(wcg_val, expected_wcg, rtol=1e-4)
        assert np.isnan(wcp_val)

    def test_third_order_system_analytical_margins(self) -> None:
        """Test stability margins of G(s) = K / (s * (s + 1) * (s + 2)).

        Analytical:
        Phase crossover: wcp = sqrt(2) ~= 1.41421356 rad/s
        |G(j*sqrt(2))| = K / 6 -> GM = 6 / K, GM_dB = 20*log10(6/K)
        For K = 1: GM_dB = 20*log10(6) ~= 15.563 dB
        """
        # K = 1
        sys1 = tf(1.0, [1.0, 3.0, 2.0, 0.0])
        sm1 = margin(sys1)

        assert np.isclose(sm1.wcp, np.sqrt(2.0), rtol=1e-4)
        assert np.isclose(sm1.gm_db, 20.0 * np.log10(6.0), atol=0.05)
        assert sm1.wcg < sm1.wcp
        assert sm1.pm_deg > 0.0  # Stable

        # K = 6 (Marginally stable, GM = 0 dB, PM = 0 deg at w = sqrt(2))
        sys6 = tf(6.0, [1.0, 3.0, 2.0, 0.0])
        sm6 = margin(sys6)

        assert np.isclose(sm6.wcp, np.sqrt(2.0), rtol=1e-4)
        assert np.isclose(sm6.wcg, np.sqrt(2.0), rtol=1e-4)
        assert np.isclose(sm6.gm_db, 0.0, atol=0.05)
        assert np.isclose(sm6.pm_deg, 0.0, atol=0.05)

        # K = 12 (Unstable, GM = -6.02 dB)
        sys12 = tf(12.0, [1.0, 3.0, 2.0, 0.0])
        sm12 = margin(sys12)

        assert np.isclose(sm12.wcp, np.sqrt(2.0), rtol=1e-4)
        assert np.isclose(sm12.gm_db, 20.0 * np.log10(0.5), atol=0.05)
        assert sm12.pm_deg < 0.0

    def test_integrator_margins(self) -> None:
        """Test stability margins of 1/s."""
        sys = tf(1.0, [1.0, 0.0])
        sm = sys.margin()

        assert np.isclose(sm.wcg, 1.0, rtol=1e-4)
        assert np.isclose(sm.pm_deg, 90.0, atol=0.05)
        assert np.isnan(sm.wcp)
        assert np.isinf(sm.gm_db)

        # String representation
        repr_str = repr(sm)
        assert "StabilityMargins" in repr_str
        assert "gm_db=inf" in str(sm)

    def test_margin_unsupported_type(self) -> None:
        """Verify TypeError when non-LTI is passed to margin, bode_data, nyquist_data."""
        with pytest.raises(TypeError):
            margin("not_a_system")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            bode_data("not_a_system")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            nyquist_data("not_a_system")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            root_locus_data("not_a_system")  # type: ignore[arg-type]


class TestNyquistData:
    """Test suite for nyquist_data calculation."""

    def test_first_order_semicircle(self) -> None:
        """Test Nyquist data for 1 / (s + 1)."""
        sys = tf(1.0, [1.0, 1.0])
        w = np.array([0.01, 1.0, 100.0])
        ndata = nyquist_data(sys, omega=w)

        assert isinstance(ndata, NyquistData)
        assert len(ndata.response) == 3

        # At w = 1, G(j1) = 1/(1+j) = 0.5 - 0.5j
        assert np.isclose(ndata.real[1], 0.5, atol=1e-4)
        assert np.isclose(ndata.imag[1], -0.5, atol=1e-4)

    def test_integrator_avoid_zero_division(self) -> None:
        """Test Nyquist data on pure integrator 1/s does not error or produce infs."""
        sys = tf(1.0, [1.0, 0.0])
        ndata = nyquist_data(sys)

        assert not np.any(np.isnan(ndata.real))
        assert not np.any(np.isnan(ndata.imag))
        assert not np.any(np.isinf(ndata.real))
        assert not np.any(np.isinf(ndata.imag))

    def test_integrator_indented_arc(self) -> None:
        """Test indented Nyquist arc for G(s) = (2s + 5) / (s * (s + 2))."""
        sys = tf([2.0, 5.0], [1.0, 2.0, 0.0])
        ndata = nyquist_data(sys)

        assert ndata.arc_s is not None
        assert ndata.arc_response is not None
        assert len(ndata.arc_s) == 100
        assert len(ndata.arc_response) == 100

        # Arc should sweep from theta = -pi/2 to +pi/2
        eps = float(ndata.w[0])
        assert np.isclose(ndata.arc_s[0], -1j * eps)
        assert np.isclose(ndata.arc_s[-1], 1j * eps)

        # arc_response[0] must match G(-j*eps) = conj(G(j*eps))
        assert np.isclose(ndata.arc_response[0], np.conj(ndata.response[0]), rtol=1e-3)
        # arc_response[-1] must match G(j*eps)
        assert np.isclose(ndata.arc_response[-1], ndata.response[0], rtol=1e-3)

    def test_unpacking(self) -> None:
        """Test tuple unpacking of NyquistData."""
        sys = tf(1.0, [1.0, 1.0])
        w, resp = nyquist_data(sys, omega=[0.1, 1.0, 10.0])
        assert len(w) == 3
        assert len(resp) == 3


class TestRootLocusData:
    """Test suite for root_locus_data calculation."""

    def test_open_loop_endpoints(self) -> None:
        """Test Root Locus endpoints (start at poles at k=0, approach zeros as k -> inf)."""
        # G(s) = (s + 2) / ((s + 1) * (s + 3))
        # 1 zero at -2, 2 poles at -1 and -3
        sys = tf([1.0, 2.0], [1.0, 4.0, 3.0])

        gains = np.concatenate(([0.0], np.geomspace(1e-4, 1e5, 500)))
        rldata = root_locus_data(sys, gains=gains)

        assert isinstance(rldata, RootLocusData)
        assert rldata.roots.shape == (501, 2)

        # Endpoints at k = 0: must match open-loop poles {-1, -3}
        roots_k0 = np.sort(rldata.roots[0].real)
        expected_poles = np.sort(sys.poles().real)
        assert np.allclose(roots_k0, expected_poles, atol=1e-4)

        # As k -> inf: one root approaches zero at -2, the other goes to -inf
        roots_kinf = rldata.roots[-1].real
        min_root = min(roots_kinf)
        max_root = max(roots_kinf)

        assert np.isclose(max_root, -2.0, atol=1e-3)  # approaches zero at -2
        assert min_root < -1e4  # goes along asymptote to -inf

    def test_complex_conjugate_trajectories(self) -> None:
        """Test Root Locus for G(s) = 1 / (s * (s + 2))."""
        # Poles at 0 and -2. Breakaway point at s = -1 for k >= 1
        sys = tf(1.0, [1.0, 2.0, 0.0])
        gains = [0.0, 1.0, 5.0, 10.0]
        rldata = root_locus_data(sys, gains=gains)

        # At k = 1: roots are double root at -1
        assert np.allclose(rldata.roots[1], [-1.0, -1.0], atol=1e-4)

        # At k = 5: s^2 + 2s + 5 = 0 -> s = -1 +- 2j
        r5 = rldata.roots[2]
        assert np.isclose(r5[0].real, -1.0, atol=1e-4)
        assert np.isclose(r5[1].real, -1.0, atol=1e-4)
        assert np.isclose(abs(r5[0].imag), 2.0, atol=1e-4)

    def test_invalid_gains_validation(self) -> None:
        """Test error handling for negative gains."""
        sys = tf(1.0, [1.0, 1.0])
        with pytest.raises(ValueError, match="must be non-negative"):
            root_locus_data(sys, gains=[-1.0, 0.0, 1.0])

        with pytest.raises(ValueError, match="cannot be empty"):
            root_locus_data(sys, gains=[])


class TestPlottingFunctions:
    """Test suite for Matplotlib visualization routines."""

    def test_plot_bode_default(self) -> None:
        """Test plot_bode with default parameters returns Figure and Axes."""
        sys = tf([10.0], [1.0, 3.0, 2.0, 0.0])
        fig, axes = plot_bode(sys, margins=True)

        assert isinstance(fig, Figure)
        assert isinstance(axes, tuple)
        assert len(axes) == 2
        assert isinstance(axes[0], Axes)
        assert isinstance(axes[1], Axes)

    def test_plot_bode_custom_axes(self) -> None:
        """Test plot_bode with existing axes and margins=False."""
        sys = tf(1.0, [1.0, 1.0])
        custom_fig, (ax1, ax2) = plt.subplots(2, 1)

        fig_out, (ax_mag, ax_phase) = plot_bode(
            sys,
            margins=False,
            ax=(ax1, ax2),
        )

        assert fig_out is custom_fig
        assert ax_mag is ax1
        assert ax_phase is ax2

    def test_plot_bode_invalid_axes(self) -> None:
        """Test plot_bode raises error if wrong number of axes provided."""
        sys = tf(1.0, [1.0, 1.0])
        _, ax_single = plt.subplots()

        with pytest.raises(ValueError, match="requires exactly 2 Axes"):
            plot_bode(sys, ax=[ax_single])

    def test_plot_nyquist(self) -> None:
        """Test plot_nyquist returns Figure and Axes."""
        sys = tf([1.0], [1.0, 2.0, 1.0])
        fig, ax = plot_nyquist(sys)

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_plot_nyquist_bounded_limits_for_integrator(self) -> None:
        """Test plot_nyquist auto-scales reasonably around (-1, 0) without infinite asymptotes."""
        sys = tf([2.0, 5.0], [1.0, 2.0, 0.0])
        _fig, ax = plot_nyquist(sys)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Both xlim and ylim should be finite and bounded reasonably near critical point (-1, 0)
        assert xlim[0] >= -15.0 and xlim[1] <= 15.0
        assert ylim[0] >= -15.0 and ylim[1] <= 15.0

        # Critical point (-1, 0) must be within limits
        assert xlim[0] < -1.0 < xlim[1]
        assert ylim[0] < 0.0 < ylim[1]

    def test_plot_nyquist_custom_ax(self) -> None:
        """Test plot_nyquist with custom ax."""
        sys = tf(1.0, [1.0, 1.0])
        custom_fig, custom_ax = plt.subplots()

        fig_out, ax_out = plot_nyquist(sys, ax=custom_ax)
        assert fig_out is custom_fig
        assert ax_out is custom_ax

    def test_plot_root_locus(self) -> None:
        """Test plot_root_locus returns Figure and Axes."""
        sys = tf([1.0, 2.0], [1.0, 4.0, 3.0])
        fig, ax = plot_root_locus(sys)

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_plot_step(self) -> None:
        """Test plot_step helper returns Figure and Axes."""
        sys = tf([2.0], [1.0, 2.0, 2.0])
        fig, ax = plot_step(sys, T=10.0)

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_dataclass_reprs(self) -> None:
        """Test _repr_latex_ and _repr_markdown_ on frequency-domain dataclasses."""
        sys = tf([2.0], [1.0, 2.0, 2.0])
        bdata = bode_data(sys)
        ndata = nyquist_data(sys)
        rldata = root_locus_data(sys)
        sm = margin(sys)

        assert "$$" in bdata._repr_latex_()
        assert bdata._repr_markdown_() == bdata._repr_latex_()

        assert "$$" in ndata._repr_latex_()
        assert ndata._repr_markdown_() == ndata._repr_latex_()

        assert "$$" in rldata._repr_latex_()
        assert rldata._repr_markdown_() == rldata._repr_latex_()

        assert "$$" in sm._repr_latex_()
        assert sm._repr_markdown_() == sm._repr_latex_()

"""Unit tests for Plotly interactive plotting and fluent LTI methods."""

from __future__ import annotations

import matplotlib
import plotly.graph_objects as go
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

matplotlib.use("Agg")

from ctrlpy.models.state_space import ss
from ctrlpy.models.transfer_function import tf
from ctrlpy.plotting_plotly import (
    iplot_bode,
    iplot_impulse,
    iplot_nyquist,
    iplot_root_locus,
    iplot_step,
)


class TestPlotlyFunctions:
    """Test suite for stand-alone Plotly plotting routines."""

    def test_iplot_bode_default(self) -> None:
        """Test iplot_bode generates valid Figure with subplots and margins."""
        sys = tf([10.0], [1.0, 3.0, 2.0, 0.0])
        fig = iplot_bode(sys, margins=True)

        assert isinstance(fig, go.Figure)
        trace_names = [trace.name for trace in fig.data if trace.name is not None]
        assert "Magnitude" in trace_names
        assert "Phase" in trace_names
        assert fig.layout.xaxis2.type == "log"

    def test_iplot_bode_no_margins(self) -> None:
        """Test iplot_bode with margins=False."""
        sys = tf(1.0, [1.0, 1.0])
        fig = iplot_bode(sys, margins=False)

        assert isinstance(fig, go.Figure)
        trace_names = [trace.name for trace in fig.data if trace.name is not None]
        assert "Magnitude" in trace_names
        assert "Phase" in trace_names
        assert "Gain Margin" not in trace_names
        assert "Phase Margin" not in trace_names

    def test_iplot_nyquist(self) -> None:
        """Test iplot_nyquist creates positive/negative traces, critical point and unit circle."""
        sys = tf([2.0, 5.0], [1.0, 2.0, 0.0])
        fig = iplot_nyquist(sys)

        assert isinstance(fig, go.Figure)
        trace_names = [trace.name for trace in fig.data if trace.name is not None]
        assert "G(jω), ω > 0" in trace_names
        assert "G(jω), ω < 0" in trace_names
        assert "Critical Point (-1, 0)" in trace_names
        assert "Unit Circle" in trace_names

    def test_iplot_root_locus(self) -> None:
        """Test iplot_root_locus plots branches and open-loop poles/zeros."""
        sys = tf([1.0, 2.0], [1.0, 4.0, 3.0])  # zero at -2, poles at -1, -3
        fig = iplot_root_locus(sys)

        assert isinstance(fig, go.Figure)
        trace_names = [trace.name for trace in fig.data if trace.name is not None]
        assert "Branch 1" in trace_names
        assert "Open-loop Poles" in trace_names
        assert "Open-loop Zeros" in trace_names

    def test_iplot_step(self) -> None:
        """Test iplot_step plots step response with performance markers."""
        sys = tf([25.0], [1.0, 4.0, 25.0])  # underdamped 2nd order
        fig = iplot_step(sys, T=5.0)

        assert isinstance(fig, go.Figure)
        trace_names = [trace.name for trace in fig.data if trace.name is not None]
        assert "Step Response" in trace_names
        assert "Peak Overshoot" in trace_names

    def test_iplot_impulse(self) -> None:
        """Test iplot_impulse plots impulse response."""
        sys = tf([1.0], [1.0, 1.0])
        fig = iplot_impulse(sys, T=3.0)

        assert isinstance(fig, go.Figure)
        trace_names = [trace.name for trace in fig.data if trace.name is not None]
        assert "Impulse Response" in trace_names


class TestFluentPlottingMethods:
    """Test suite for fluent plotting methods on LinearTimeInvariant models."""

    @pytest.fixture
    def sample_tf(self) -> tf:
        return tf([2.0, 5.0], [1.0, 3.0, 2.0])

    @pytest.fixture
    def sample_ss(self) -> ss:
        return ss([[-1, 1], [0, -2]], [[0], [1]], [[1, 0]], [[0]])

    def test_plot_step_matplotlib(self, sample_tf: tf) -> None:
        fig, ax = sample_tf.plot_step(backend="matplotlib", T=5.0)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_plot_step_plotly(self, sample_tf: tf) -> None:
        fig = sample_tf.plot_step(backend="plotly", T=5.0)
        assert isinstance(fig, go.Figure)

    def test_iplot_step_shortcut(self, sample_tf: tf) -> None:
        fig = sample_tf.iplot_step(T=5.0)
        assert isinstance(fig, go.Figure)

    def test_plot_impulse_matplotlib(self, sample_ss: ss) -> None:
        fig, ax = sample_ss.plot_impulse(backend="matplotlib", T=5.0)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_plot_impulse_plotly(self, sample_ss: ss) -> None:
        fig = sample_ss.plot_impulse(backend="plotly", T=5.0)
        assert isinstance(fig, go.Figure)

    def test_iplot_impulse_shortcut(self, sample_ss: ss) -> None:
        fig = sample_ss.iplot_impulse(T=5.0)
        assert isinstance(fig, go.Figure)

    def test_plot_bode_matplotlib(self, sample_tf: tf) -> None:
        fig, axes = sample_tf.plot_bode(backend="matplotlib")
        assert isinstance(fig, Figure)
        assert isinstance(axes, tuple)
        assert len(axes) == 2
        assert isinstance(axes[0], Axes)
        assert isinstance(axes[1], Axes)

    def test_plot_bode_plotly(self, sample_tf: tf) -> None:
        fig = sample_tf.plot_bode(backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_iplot_bode_shortcut(self, sample_tf: tf) -> None:
        fig = sample_tf.iplot_bode()
        assert isinstance(fig, go.Figure)

    def test_plot_nyquist_matplotlib(self, sample_tf: tf) -> None:
        fig, ax = sample_tf.plot_nyquist(backend="matplotlib")
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_plot_nyquist_plotly(self, sample_tf: tf) -> None:
        fig = sample_tf.plot_nyquist(backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_iplot_nyquist_shortcut(self, sample_tf: tf) -> None:
        fig = sample_tf.iplot_nyquist()
        assert isinstance(fig, go.Figure)

    def test_plot_root_locus_matplotlib(self, sample_tf: tf) -> None:
        fig, ax = sample_tf.plot_root_locus(backend="matplotlib")
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_plot_root_locus_plotly(self, sample_tf: tf) -> None:
        fig = sample_tf.plot_root_locus(backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_iplot_root_locus_shortcut(self, sample_tf: tf) -> None:
        fig = sample_tf.iplot_root_locus()
        assert isinstance(fig, go.Figure)

    def test_invalid_backend_error(self, sample_tf: tf) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            sample_tf.plot_step(backend="seaborn")  # type: ignore[arg-type]

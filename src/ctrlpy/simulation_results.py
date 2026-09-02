"""Simulation results data structures and performance metrics extraction."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ctrlpy.exceptions import UnstableSystemError


@dataclass
class TimeResponseData:
    """Container for time-domain simulation response data and performance metrics.

    Parameters
    ----------
    t : NDArray[np.float64]
        1D array containing simulation time points.
    y : NDArray[np.float64]
        1D or 2D array containing the system output trajectory.
    x : NDArray[np.float64] | None, optional
        Optional 2D array containing state trajectories (if simulated from StateSpace).
    sys : Any, optional
        Optional reference to the simulated LTI system.
    poles : NDArray[np.complex128] | None, optional
        Optional array of system poles used for stability validation.
    """

    t: NDArray[np.float64]
    y: NDArray[np.float64]
    x: NDArray[np.float64] | None = None
    sys: Any = None
    poles: NDArray[np.complex128] | None = None

    def __post_init__(self) -> None:
        """Validate and cast input arrays to float64."""
        self.t = np.asarray(self.t, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=np.float64)
        if self.x is not None:
            x_arr = np.asarray(self.x, dtype=np.float64)
            if x_arr.ndim == 1:
                x_arr = x_arr.reshape(-1, 1)
            self.x = x_arr

    def __iter__(self) -> Iterator[NDArray[np.float64]]:
        """Enable unpacking of (t, y) or (t, y, x).

        Yields
        ------
        NDArray[np.float64]
            Time array `t`, output array `y`, and optionally state array `x` if present.
        """
        yield self.t
        yield self.y
        if self.x is not None:
            yield self.x

    def _get_channel_1d(self, channel: int = 0) -> NDArray[np.float64]:
        """Extract a 1D output slice for a specified output channel.

        Parameters
        ----------
        channel : int, optional
            Output channel index, default is 0.

        Returns
        -------
        NDArray[np.float64]
            1D array of output values over time.

        Raises
        ------
        IndexError
            If channel index is out of bounds.
        ValueError
            If output array has unsupported dimensionality.
        """
        if self.y.ndim == 1:
            if channel != 0:
                raise IndexError(f"Channel index {channel} is invalid for 1D output.")
            return self.y
        if self.y.ndim == 2:
            if 0 <= channel < self.y.shape[1]:
                return self.y[:, channel]
            raise IndexError(
                f"Channel index {channel} out of range for output shape {self.y.shape}."
            )
        raise ValueError(f"Output array has unsupported dimension {self.y.ndim}.")

    def _check_stability(self) -> None:
        """Check if the system is unstable and raise UnstableSystemError."""
        poles = self.poles
        if poles is None and self.sys is not None:
            poles = self.sys.poles()

        if poles is not None and len(poles) > 0:
            if getattr(self.sys, "is_discrete", False):
                unstable = [p for p in poles if np.abs(p) > 1.0 + 1e-6]
                if len(unstable) > 0:
                    raise UnstableSystemError(
                        f"Cannot calculate transient response metrics for an unstable discrete system. "
                        f"System has pole(s) outside the unit circle (|p| > 1): {unstable}"
                    )
            else:
                unstable = [p for p in poles if np.real(p) > 1e-6]
                if len(unstable) > 0:
                    raise UnstableSystemError(
                        f"Cannot calculate transient response metrics for an unstable system. "
                        f"System has pole(s) in the Right-Half Plane (Re(p) > 0): {unstable}"
                    )

    def steady_state_value(self, channel: int = 0) -> float:
        r"""Compute the final steady-state response value $y_{ss}$.

        .. math::

            y_{ss} = \lim_{t \to \infty} y(t) \approx y(t_{\text{final}})

        Parameters
        ----------
        channel : int, optional
            Output channel index, default is 0.

        Returns
        -------
        float
            The final output value at the end of the simulation horizon.

        Raises
        ------
        UnstableSystemError
            If the system is unstable (has poles with $\mathrm{Re}(p) > 0$).
        """
        self._check_stability()
        y_vec = self._get_channel_1d(channel)
        return float(y_vec[-1])

    def rise_time(
        self,
        low: float = 0.1,
        high: float = 0.9,
        channel: int = 0,
    ) -> float:
        r"""Compute the rise time $t_r$ from low% to high% of the steady-state transition.

        By default, computes the 10% to 90% rise time with linear interpolation
        between discrete time points for high numerical accuracy:

        .. math::

            t_r = t_{90\%} - t_{10\%}

        Parameters
        ----------
        low : float, optional
            Lower threshold fraction of the transition (default is 0.1 for 10%).
        high : float, optional
            Upper threshold fraction of the transition (default is 0.9 for 90%).
        channel : int, optional
            Output channel index, default is 0.

        Returns
        -------
        float
            Rise time in seconds, or NaN if the response does not cross the thresholds.

        Raises
        ------
        UnstableSystemError
            If the system is unstable (has poles with $\mathrm{Re}(p) > 0$).
        """
        self._check_stability()
        y_vec = self._get_channel_1d(channel)
        t_vec = self.t
        y0 = float(y_vec[0])
        yss = float(y_vec[-1])
        dy = yss - y0

        if abs(dy) < 1e-12:
            return 0.0

        y_low = y0 + low * dy
        y_high = y0 + high * dy

        if dy > 0:
            idx_low = np.flatnonzero(y_vec >= y_low)
            idx_high = np.flatnonzero(y_vec >= y_high)
        else:
            idx_low = np.flatnonzero(y_vec <= y_low)
            idx_high = np.flatnonzero(y_vec <= y_high)

        if idx_low.size == 0 or idx_high.size == 0:
            return float("nan")

        i_l = int(idx_low[0])
        if i_l == 0:
            t_l = float(t_vec[0])
        else:
            y_prev, y_curr = float(y_vec[i_l - 1]), float(y_vec[i_l])
            t_prev, t_curr = float(t_vec[i_l - 1]), float(t_vec[i_l])
            t_l = t_prev + (t_curr - t_prev) * (y_low - y_prev) / (y_curr - y_prev)

        i_h = int(idx_high[0])
        if i_h == 0:
            t_h = float(t_vec[0])
        else:
            y_prev, y_curr = float(y_vec[i_h - 1]), float(y_vec[i_h])
            t_prev, t_curr = float(t_vec[i_h - 1]), float(t_vec[i_h])
            t_h = t_prev + (t_curr - t_prev) * (y_high - y_prev) / (y_curr - y_prev)

        return float(t_h - t_l)

    def settling_time(
        self,
        tolerance: float = 0.02,
        channel: int = 0,
    ) -> float:
        r"""Compute the settling time $t_s$ within a specified tolerance band around steady-state.

        Settling time is the time after which the response remains permanently within
        the error band:

        .. math::

            |y(t) - y_{ss}| \le \text{tolerance} \cdot |y_{ss} - y_0|, \quad \forall t \ge t_s

        The boundary crossing is linearly interpolated for sub-sample precision.

        Parameters
        ----------
        tolerance : float, optional
            Settling tolerance fraction, default is 0.02 (2% error band).
        channel : int, optional
            Output channel index, default is 0.

        Returns
        -------
        float
            Settling time in seconds, or NaN if the system has not settled by the
            end of the simulation.

        Raises
        ------
        UnstableSystemError
            If the system is unstable (has poles with $\mathrm{Re}(p) > 0$).
        """
        self._check_stability()
        y_vec = self._get_channel_1d(channel)
        t_vec = self.t
        y0 = float(y_vec[0])
        yss = float(y_vec[-1])
        dy = abs(yss - y0)
        delta = tolerance * dy if dy >= 1e-12 else tolerance * max(abs(yss), 1.0)

        # Check if the signal is actively sweeping or divergent at the end
        k_tail = max(2, int(0.02 * len(t_vec)))
        if np.max(np.abs(y_vec[-k_tail:] - yss)) > delta:
            return float("nan")

        outside = np.flatnonzero(np.abs(y_vec - yss) > delta)
        if outside.size == 0:
            return float(t_vec[0])

        last_idx = int(outside[-1])
        if last_idx == len(t_vec) - 1:
            return float("nan")

        k = last_idx
        y_k = float(y_vec[k])
        y_next = float(y_vec[k + 1])
        t_k = float(t_vec[k])
        t_next = float(t_vec[k + 1])
        target_y = yss + np.sign(y_k - yss) * delta

        if abs(y_next - y_k) > 1e-15:
            return float(t_k + (t_next - t_k) * (target_y - y_k) / (y_next - y_k))
        return float(t_next)

    def overshoot(self, channel: int = 0) -> float:
        r"""Compute the maximum percent overshoot $\%OS$ above steady-state.

        .. math::

            \%OS = \frac{y_{\text{peak}} - y_{ss}}{|y_{ss} - y_0|} \times 100\%

        Parameters
        ----------
        channel : int, optional
            Output channel index, default is 0.

        Returns
        -------
        float
            Percent overshoot (e.g. 16.3 for 16.3% overshoot). Returns 0.0 if no
            overshoot occurs.

        Raises
        ------
        UnstableSystemError
            If the system is unstable (has poles with $\mathrm{Re}(p) > 0$).
        """
        self._check_stability()
        y_vec = self._get_channel_1d(channel)
        y0 = float(y_vec[0])
        yss = float(y_vec[-1])
        dy = yss - y0

        if abs(dy) < 1e-12:
            return 0.0

        if dy > 0:
            y_peak = float(np.max(y_vec))
            if y_peak <= yss:
                return 0.0
            return float(((y_peak - yss) / dy) * 100.0)
        else:
            y_peak = float(np.min(y_vec))
            if y_peak >= yss:
                return 0.0
            return float(((yss - y_peak) / abs(dy)) * 100.0)

    def peak_time(self, channel: int = 0) -> float:
        r"""Compute the time $t_p$ at which the maximum peak response occurs.

        .. math::

            t_p = \arg\max_t |y(t) - y_0|

        Parameters
        ----------
        channel : int, optional
            Output channel index, default is 0.

        Returns
        -------
        float
            Peak time in seconds.

        Raises
        ------
        UnstableSystemError
            If the system is unstable (has poles with $\mathrm{Re}(p) > 0$).
        """
        self._check_stability()
        y_vec = self._get_channel_1d(channel)
        t_vec = self.t
        y0 = float(y_vec[0])
        yss = float(y_vec[-1])
        dy = yss - y0

        if dy >= 0:
            k = int(np.argmax(y_vec))
        else:
            k = int(np.argmin(y_vec))

        return float(t_vec[k])

    def _repr_latex_(self) -> str:
        """Return a LaTeX table of transient response performance metrics for Jupyter."""
        try:
            yss_val = self.steady_state_value()
            yss_str = f"{yss_val:.4f}"
            tr_val = self.rise_time()
            tr_str = f"{tr_val:.4f}\\text{{ s}}" if not np.isnan(tr_val) else r"\text{N/A}"
            ts_val = self.settling_time(tolerance=0.02)
            ts_str = f"{ts_val:.4f}\\text{{ s}}" if not np.isnan(ts_val) else r"\text{N/A}"
            os_val = self.overshoot()
            os_str = f"{os_val:.2f}\\%"
            tp_val = self.peak_time()
            tp_str = f"{tp_val:.4f}\\text{{ s}}"
        except UnstableSystemError:
            return r"$$\textbf{\textcolor{red}{Unstable System: Transient metrics not applicable (divergent response)}}$$"
        except (ValueError, IndexError, RuntimeError, TypeError):
            final_y = self.y[-1] if self.y.ndim == 1 else self.y[-1, 0]
            return (
                rf"$$\text{{TimeResponseData: {len(self.t)} points, final value: {final_y:.4f}}}$$"
            )

        lines = [
            r"$$\begin{array}{|l|c|}",
            r"\hline",
            r"\textbf{Transient Response Metric} & \textbf{Value} \\",
            r"\hline",
            rf"\text{{Steady-State Value }} (y_{{ss}}) & {yss_str} \\",
            rf"\text{{Rise Time }} (t_r, 10\%-90\%) & {tr_str} \\",
            rf"\text{{Settling Time }} (t_s, 2\%) & {ts_str} \\",
            rf"\text{{Percent Overshoot }} (\%OS) & {os_str} \\",
            rf"\text{{Peak Time }} (t_p) & {tp_str} \\",
            r"\hline",
            r"\end{array}$$",
        ]
        return "\n".join(lines)

    def _repr_markdown_(self) -> str:
        """Return a Markdown representation for Jupyter environments."""
        return self._repr_latex_()

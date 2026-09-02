"""Discrete-time LTI control systems engine, z-domain representations, and discretization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from ctrlpy.freq_domain import BodeData
from ctrlpy.models.base import LinearTimeInvariant
from ctrlpy.simulation_results import TimeResponseData

if TYPE_CHECKING:
    import plotly.graph_objects as go
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ctrlpy.models.state_space import StateSpace
    from ctrlpy.models.transfer_function import TransferFunction


def _format_discrete_poly_str(coeffs: NDArray[np.float64], var: str = "z") -> str:
    """Format discrete polynomial coefficients into a human-readable string.

    Parameters
    ----------
    coeffs : NDArray[np.float64]
        Polynomial coefficients.
    var : str, optional
        Variable symbol, default is 'z'.

    Returns
    -------
    str
        Formatted polynomial string.
    """
    n = len(coeffs) - 1
    if n < 0 or (len(coeffs) == 1 and np.isclose(coeffs[0], 0.0)):
        return "0"

    is_inv_z = var in ("z^-1", "z^{-1}")
    terms: list[tuple[str, str]] = []

    for i, c in enumerate(coeffs):
        if np.isclose(c, 0.0):
            continue
        power = i if is_inv_z else (n - i)
        abs_c = abs(c)
        if np.isclose(abs_c, round(abs_c)):
            abs_c_str = f"{round(abs_c)}"
        else:
            abs_c_str = f"{abs_c:g}"

        var_base = "z" if not is_inv_z else "z^-1"
        if power == 0:
            term_str = abs_c_str
        elif power == 1 and not is_inv_z:
            term_str = f"{var_base}" if abs_c_str == "1" else f"{abs_c_str} {var_base}"
        else:
            if is_inv_z:
                term_str = f"z^-{power}" if abs_c_str == "1" else f"{abs_c_str} z^-{power}"
            else:
                term_str = (
                    f"{var_base}^{power}" if abs_c_str == "1" else f"{abs_c_str} {var_base}^{power}"
                )

        sign = "-" if c < 0 else "+"
        terms.append((sign, term_str))

    if not terms:
        return "0"

    first_sign, first_term = terms[0]
    res = f"-{first_term}" if first_sign == "-" else first_term
    for sign, term in terms[1:]:
        res += f" {sign} {term}"
    return res


def _format_discrete_poly_latex(coeffs: NDArray[np.float64], var: str = "z") -> str:
    """Format discrete polynomial coefficients into a LaTeX string.

    Parameters
    ----------
    coeffs : NDArray[np.float64]
        Polynomial coefficients.
    var : str, optional
        Variable symbol, default is 'z'.

    Returns
    -------
    str
        Formatted LaTeX string for the polynomial.
    """
    n = len(coeffs) - 1
    if n < 0 or (len(coeffs) == 1 and np.isclose(coeffs[0], 0.0)):
        return "0"

    is_inv_z = var in ("z^-1", "z^{-1}")
    terms: list[tuple[str, str]] = []

    for i, c in enumerate(coeffs):
        if np.isclose(c, 0.0):
            continue
        power = i if is_inv_z else (n - i)
        abs_c = abs(c)
        if np.isclose(abs_c, round(abs_c)):
            abs_c_str = f"{round(abs_c)}"
        else:
            abs_c_str = f"{abs_c:g}"

        var_base = "z" if not is_inv_z else "z^{-1}"
        if power == 0:
            term_str = abs_c_str
        elif power == 1 and not is_inv_z:
            term_str = f"{var_base}" if abs_c_str == "1" else f"{abs_c_str} {var_base}"
        else:
            if is_inv_z:
                term_str = f"z^{{-{power}}}" if abs_c_str == "1" else f"{abs_c_str} z^{{-{power}}}"
            else:
                term_str = (
                    f"{var_base}^{{{power}}}"
                    if abs_c_str == "1"
                    else f"{abs_c_str} {var_base}^{{{power}}}"
                )

        sign = "-" if c < 0 else "+"
        terms.append((sign, term_str))

    if not terms:
        return "0"

    first_sign, first_term = terms[0]
    res = f"-{first_term}" if first_sign == "-" else first_term
    for sign, term in terms[1:]:
        res += f" {sign} {term}"
    return res


class DiscreteLTI(ABC):
    r"""Abstract base class representing a Discrete-Time Linear Time-Invariant (LTI) system.

    Discrete-time systems operate on sampled signals $x[k] = x(k T_s)$ at discrete
    sampling intervals $T_s > 0$.
    """

    @property
    @abstractmethod
    def dt(self) -> float:
        """Sampling period $T_s$ in seconds."""
        ...

    @property
    def Ts(self) -> float:
        """Alias for sampling period $T_s$ in seconds."""
        return self.dt

    @property
    def is_discrete(self) -> bool:
        """Whether the system is discrete-time (always True for DiscreteLTI)."""
        return True

    @property
    @abstractmethod
    def inputs(self) -> int:
        """Number of input channels."""
        ...

    @property
    @abstractmethod
    def outputs(self) -> int:
        """Number of output channels."""
        ...

    @property
    def is_siso(self) -> bool:
        """Whether the system is Single-Input Single-Output (SISO)."""
        return self.inputs == 1 and self.outputs == 1

    @abstractmethod
    def poles(self) -> NDArray[np.complex128]:
        """Compute the poles of the discrete system in the complex $z$-plane.

        Returns
        -------
        NDArray[np.complex128]
            1D array of discrete poles.
        """
        ...

    @abstractmethod
    def zeros(self) -> NDArray[np.complex128]:
        """Compute the zeros of the discrete system in the complex $z$-plane.

        Returns
        -------
        NDArray[np.complex128]
            1D array of discrete zeros.
        """
        ...

    def is_stable(self, tol: float = 1e-7) -> bool:
        r"""Assess asymptotic/BIBO stability of the discrete system.

        A discrete LTI system is strictly asymptotically stable if and only if
        all poles lie strictly inside the unit circle:

        .. math::

            |p_i| < 1, \quad \forall i = 1, \dots, n

        Parameters
        ----------
        tol : float, optional
            Numerical tolerance for unit circle boundary check, default is 1e-7.

        Returns
        -------
        bool
            True if all poles have $|p_i| < 1 - \text{tol}$, False otherwise.
        """
        p = self.poles()
        if len(p) == 0:
            return True
        return bool(np.all(np.abs(p) < 1.0 - tol))

    def is_marginally_stable(self, tol: float = 1e-7) -> bool:
        r"""Check if the system is marginally stable.

        A discrete LTI system is marginally stable if no poles lie outside the
        unit circle ($|p_i| \le 1 + \text{tol}$), at least one pole lies on the
        unit circle ($||p_i| - 1| \le \text{tol}$), and all unit-circle poles
        have multiplicity 1 (non-repeated).

        Parameters
        ----------
        tol : float, optional
            Numerical tolerance, default is 1e-7.

        Returns
        -------
        bool
            True if system is marginally stable, False otherwise.
        """
        p = self.poles()
        if len(p) == 0:
            return False

        mags = np.abs(p)
        if np.any(mags > 1.0 + tol):
            return False

        on_circle = p[np.abs(mags - 1.0) <= tol]
        if len(on_circle) == 0:
            return False

        # Check multiplicity of poles on the unit circle
        for i in range(len(on_circle)):
            for j in range(i + 1, len(on_circle)):
                if np.abs(on_circle[i] - on_circle[j]) <= 1e-4:
                    return False  # Repeated pole on unit circle causes unbounded response

        return True

    def stability(self, tol: float = 1e-7) -> Literal["stable", "marginally stable", "unstable"]:
        r"""Classify the stability status of the discrete system relative to the unit circle $|z|=1$.

        Parameters
        ----------
        tol : float, optional
            Numerical tolerance, default is 1e-7.

        Returns
        -------
        {"stable", "marginally stable", "unstable"}
            Stability classification string.
        """
        if self.is_stable(tol=tol):
            return "stable"
        if self.is_marginally_stable(tol=tol):
            return "marginally stable"
        return "unstable"

    def step(
        self,
        T: Sequence[float] | NDArray[np.floating[Any]] | float | None = None,
        n_steps: int | None = None,
    ) -> TimeResponseData:
        """Compute the step response of the discrete-time system iteratively.

        Parameters
        ----------
        T : Sequence[float] | NDArray[np.floating] | float | None, optional
            Simulation duration in seconds, or explicit discrete time points.
        n_steps : int | None, optional
            Number of discrete sample steps $N$ to simulate.

        Returns
        -------
        TimeResponseData
            Simulation response object.
        """
        return discrete_step_response(self, T=T, n_steps=n_steps)

    def impulse(
        self,
        T: Sequence[float] | NDArray[np.floating[Any]] | float | None = None,
        n_steps: int | None = None,
    ) -> TimeResponseData:
        """Compute the impulse response of the discrete-time system iteratively.

        Parameters
        ----------
        T : Sequence[float] | NDArray[np.floating] | float | None, optional
            Simulation duration in seconds, or explicit discrete time points.
        n_steps : int | None, optional
            Number of discrete sample steps $N$ to simulate.

        Returns
        -------
        TimeResponseData
            Simulation response object.
        """
        return discrete_impulse_response(self, T=T, n_steps=n_steps)

    def forced_response(
        self,
        U: Sequence[float] | NDArray[np.floating[Any]] | float,
        T: Sequence[float] | NDArray[np.floating[Any]] | None = None,
        n_steps: int | None = None,
    ) -> TimeResponseData:
        """Compute the simulation response to an arbitrary discrete input sequence $u[k]$.

        Parameters
        ----------
        U : Sequence[float] | NDArray[np.floating] | float
            Input sequence $u[k]$.
        T : Sequence[float] | NDArray[np.floating] | None, optional
            Simulation time points.
        n_steps : int | None, optional
            Number of sample steps.

        Returns
        -------
        TimeResponseData
            Simulation response object.
        """
        return discrete_forced_response(self, U=U, T=T, n_steps=n_steps)

    def bode(
        self,
        omega: Sequence[float] | NDArray[np.floating[Any]] | None = None,
        n_points: int = 500,
    ) -> BodeData:
        """Compute the discrete frequency response over frequencies up to the Nyquist limit.

        Parameters
        ----------
        omega : Sequence[float] | NDArray[np.floating] | None, optional
            Frequency vector in rad/s, $\\omega \\in (0, \\pi / T_s]$.
        n_points : int, optional
            Number of frequency points, default is 500.

        Returns
        -------
        BodeData
            Frequency response object.
        """
        return discrete_bode_data(self, omega=omega, n_points=n_points)

    def freqresp(
        self,
        omega: Sequence[float] | NDArray[np.floating[Any]] | None = None,
        n_points: int = 500,
    ) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
        """Evaluate discrete complex frequency response $H(e^{j \\omega T_s})$.

        Parameters
        ----------
        omega : Sequence[float] | NDArray[np.floating] | None, optional
            Frequency vector in rad/s.
        n_points : int, optional
            Number of frequency points when omega is None.

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.complex128]]
            Tuple of (omega, complex_response).
        """
        bdata = self.bode(omega=omega, n_points=n_points)
        return bdata.w, bdata.response

    def plot_pzmap(
        self,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        ax: Axes | None = None,
        title: str | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes] | go.Figure:
        """Plot the discrete pole-zero map in the complex $z$-plane with unit circle $|z|=1$.

        Parameters
        ----------
        backend : {"matplotlib", "plotly"}, optional
            Plotting backend, default is "matplotlib".
        ax : Axes | None, optional
            Existing Matplotlib Axes (for backend="matplotlib").
        title : str | None, optional
            Custom chart title.
        **kwargs : Any
            Additional options passed to plotting routines.

        Returns
        -------
        tuple[Figure, Axes] | go.Figure
            Matplotlib (fig, ax) tuple or Plotly Figure.
        """
        if backend == "plotly":
            return iplot_pzmap(self, title=title, **kwargs)
        elif backend == "matplotlib":
            return plot_pzmap(self, ax=ax, title=title, **kwargs)
        else:
            raise ValueError(f"Unknown backend '{backend}'. Supported: 'matplotlib', 'plotly'.")

    def iplot_pzmap(self, title: str | None = None, **kwargs: Any) -> go.Figure:
        """Plot interactive discrete pole-zero map in the complex $z$-plane using Plotly.

        Parameters
        ----------
        title : str | None, optional
            Custom chart title.
        **kwargs : Any
            Additional options.

        Returns
        -------
        go.Figure
            Interactive Plotly Figure.
        """
        return iplot_pzmap(self, title=title, **kwargs)

    def plot_step(
        self,
        T: Sequence[float] | NDArray[np.floating[Any]] | float | None = None,
        n_steps: int | None = None,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        ax: Axes | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes] | go.Figure:
        """Plot the discrete step response using Matplotlib or Plotly.

        Parameters
        ----------
        T : Sequence[float] | NDArray[np.floating] | float | None, optional
            Simulation duration or time vector.
        n_steps : int | None, optional
            Number of steps.
        backend : {"matplotlib", "plotly"}, optional
            Plotting backend, default is "matplotlib".
        ax : Axes | None, optional
            Matplotlib Axes.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        tuple[Figure, Axes] | go.Figure
            Matplotlib (fig, ax) or Plotly Figure.
        """
        res = self.step(T=T, n_steps=n_steps)
        if backend == "plotly":
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=res.t,
                    y=res.y,
                    mode="lines+markers",
                    name="Discrete Step Response",
                    line={"color": "#1f77b4", "shape": "hv"},
                    marker={"size": 5},
                    hovertemplate="<b>k</b>: %{customdata}<br><b>t</b>: %{x:.3g} s<br><b>y[k]</b>: %{y:.4f}<extra></extra>",
                    customdata=np.arange(len(res.t)),
                )
            )
            fig.update_layout(
                title=f"Discrete Step Response (Ts = {self.dt} s)",
                xaxis_title="Time (s)",
                yaxis_title="Amplitude",
                template="plotly_white",
            )
            return fig
        elif backend == "matplotlib":
            import matplotlib.pyplot as plt

            if ax is None:
                fig, ax_out = plt.subplots(figsize=(8, 5))
            else:
                ax_out = ax
                fig = cast("Figure", ax_out.figure)

            ax_out.step(res.t, res.y, "b-", where="post", linewidth=1.5, label="y[k]")
            ax_out.plot(res.t, res.y, "bo", markersize=4)
            ax_out.set_xlabel("Time (s)")
            ax_out.set_ylabel("Amplitude")
            ax_out.set_title(f"Discrete Step Response ($T_s = {self.dt}$ s)")
            ax_out.grid(True, linestyle="--", alpha=0.6)
            ax_out.legend(loc="best")
            return fig, ax_out
        else:
            raise ValueError(f"Unknown backend '{backend}'.")


class DiscreteTransferFunction(DiscreteLTI):
    r"""Discrete-time Linear Time-Invariant Transfer Function representation in the $z$-domain.

    Represents a rational polynomial transfer function $H(z)$ with sampling time $T_s > 0$:

    .. math::

        H(z) = \frac{N(z)}{D(z)} = \frac{b_m z^m + b_{m-1} z^{m-1} + \cdots + b_1 z + b_0}{a_n z^n + a_{n-1} z^{n-1} + \cdots + a_1 z + a_0}

    Or equivalently in terms of backward shift / delay operator $z^{-1}$:

    .. math::

        H(z^{-1}) = \frac{b_0 + b_1 z^{-1} + \cdots + b_m z^{-m}}{a_0 + a_1 z^{-1} + \cdots + a_n z^{-n}}

    Parameters
    ----------
    num : Sequence[float] | NDArray[np.floating] | float
        Numerator polynomial coefficients.
    den : Sequence[float] | NDArray[np.floating] | float, optional
        Denominator polynomial coefficients, defaults to `(1.0,)`.
    dt : float, optional
        Sampling period $T_s > 0$ in seconds, defaults to 1.0.
    var : {"z", "z^-1", "z^{-1}", "q"}, optional
        Polynomial variable symbol, default is 'z'.

    Raises
    ------
    ValueError
        If numerator/denominator is empty, denominator is zero, or $T_s \le 0$.
    """

    def __init__(
        self,
        num: Sequence[float] | NDArray[np.floating[Any]] | float,
        den: Sequence[float] | NDArray[np.floating[Any]] | float = (1.0,),
        dt: float = 1.0,
        var: str = "z",
    ) -> None:
        if dt <= 0.0:
            raise ValueError(f"Sampling period dt (Ts) must be strictly positive, got {dt}.")

        self._dt = float(dt)
        var_clean = var.strip()
        if var_clean.lower() not in ("z", "z^-1", "z^{-1}", "q"):
            raise ValueError(
                f"Unsupported discrete variable symbol '{var}'. Supported: 'z', 'z^-1', 'z^{{-1}}', 'q'."
            )
        self._var = var_clean

        num_arr = np.asarray(num, dtype=np.float64).ravel()
        den_arr = np.asarray(den, dtype=np.float64).ravel()

        if num_arr.size == 0:
            raise ValueError("Numerator cannot be empty.")
        if den_arr.size == 0:
            raise ValueError("Denominator cannot be empty.")

        # Convert z^-1 representation to standard z polynomial form
        if self._var in ("z^-1", "z^{-1}"):
            m = len(num_arr) - 1
            n = len(den_arr) - 1
            k = max(m, n)
            # Pad polynomials to degree k in z
            num_z = np.zeros(k + 1, dtype=np.float64)
            den_z = np.zeros(k + 1, dtype=np.float64)
            num_z[: len(num_arr)] = num_arr
            den_z[: len(den_arr)] = den_arr
            num_arr = num_z
            den_arr = den_z

        # Trim leading zeros from denominator
        den_nonzero = np.flatnonzero(den_arr)
        if den_nonzero.size == 0:
            raise ValueError("Denominator cannot be identically zero.")
        den_arr = den_arr[den_nonzero[0] :]

        # Trim leading zeros from numerator
        num_nonzero = np.flatnonzero(num_arr)
        if num_nonzero.size == 0:
            num_arr = np.array([0.0], dtype=np.float64)
        else:
            num_arr = num_arr[num_nonzero[0] :]

        # Normalize by leading coefficient of denominator
        lead = den_arr[0]
        self._num: NDArray[np.float64] = num_arr / lead
        self._den: NDArray[np.float64] = den_arr / lead

    @property
    def dt(self) -> float:
        """Sampling period $T_s$ in seconds."""
        return self._dt

    @property
    def num(self) -> NDArray[np.float64]:
        """Numerator polynomial coefficients in descending powers of $z$."""
        return self._num

    @property
    def den(self) -> NDArray[np.float64]:
        """Denominator polynomial coefficients in descending powers of $z$."""
        return self._den

    @property
    def var(self) -> str:
        """Discrete variable symbol ('z' or 'z^-1')."""
        return self._var

    @property
    def inputs(self) -> int:
        """Number of inputs (1 for SISO discrete transfer function)."""
        return 1

    @property
    def outputs(self) -> int:
        """Number of outputs (1 for SISO discrete transfer function)."""
        return 1

    def poles(self) -> NDArray[np.complex128]:
        """Compute the poles of the discrete transfer function in the complex $z$-plane.

        Returns
        -------
        NDArray[np.complex128]
            1D array of pole locations in the $z$-plane.
        """
        if len(self._den) <= 1:
            return np.array([], dtype=np.complex128)
        return np.roots(self._den).astype(np.complex128)

    def zeros(self) -> NDArray[np.complex128]:
        """Compute the zeros of the discrete transfer function in the complex $z$-plane.

        Returns
        -------
        NDArray[np.complex128]
            1D array of zero locations in the $z$-plane.
        """
        if len(self._num) <= 1 or np.all(np.isclose(self._num, 0.0)):
            return np.array([], dtype=np.complex128)
        return np.roots(self._num).astype(np.complex128)

    def dcgain(self) -> float:
        r"""Compute the discrete DC (steady-state) gain $H(z=1)$.

        .. math::

            K_{dc} = H(1) = \frac{\sum b_i}{\sum a_i}

        Returns
        -------
        float
            DC gain value, or Inf / NaN if the system has an integrator pole at $z=1$.
        """
        den_sum = float(np.sum(self._den))
        if np.isclose(den_sum, 0.0):
            return float("inf")
        return float(np.sum(self._num) / den_sum)

    def __add__(
        self,
        other: DiscreteTransferFunction | float | np.number[Any],
    ) -> DiscreteTransferFunction:
        """Parallel interconnection (self + other)."""
        if isinstance(other, (int, float, np.number)):
            k = float(other)
            num = np.polyadd(self._num, k * self._den)
            return DiscreteTransferFunction(num, self._den, dt=self._dt, var=self._var)

        if isinstance(other, DiscreteTransferFunction):
            if not np.isclose(self._dt, other._dt):
                raise ValueError(
                    f"Cannot add discrete transfer functions with different sampling times: "
                    f"{self._dt} s vs {other._dt} s."
                )
            num = np.polyadd(
                np.polymul(self._num, other._den),
                np.polymul(other._num, self._den),
            )
            den = np.polymul(self._den, other._den)
            return DiscreteTransferFunction(num, den, dt=self._dt, var=self._var)

        if isinstance(other, LinearTimeInvariant):
            raise TypeError(
                "Cannot combine DiscreteTransferFunction with continuous system. "
                "Discretize the continuous system first using c2d()."
            )

        return NotImplemented

    def __radd__(
        self,
        other: DiscreteTransferFunction | float | np.number[Any],
    ) -> DiscreteTransferFunction:
        """Parallel interconnection with left operand."""
        return self.__add__(other)

    def __neg__(self) -> DiscreteTransferFunction:
        """Negation (-self)."""
        return DiscreteTransferFunction(-self._num, self._den, dt=self._dt, var=self._var)

    def __pos__(self) -> DiscreteTransferFunction:
        """Positive (+self)."""
        return DiscreteTransferFunction(
            self._num.copy(), self._den.copy(), dt=self._dt, var=self._var
        )

    def __sub__(
        self,
        other: DiscreteTransferFunction | float | np.number[Any],
    ) -> DiscreteTransferFunction:
        """Subtraction (self - other)."""
        if isinstance(other, (int, float, np.number)):
            return self.__add__(-float(other))

        if isinstance(other, DiscreteTransferFunction):
            return self.__add__(-other)

        if isinstance(other, LinearTimeInvariant):
            raise TypeError("Cannot subtract continuous system from discrete system.")

        return NotImplemented

    def __rsub__(
        self,
        other: DiscreteTransferFunction | float | np.number[Any],
    ) -> DiscreteTransferFunction:
        """Subtraction with left operand (other - self)."""
        if isinstance(other, (int, float, np.number)):
            return DiscreteTransferFunction(float(other), 1.0, dt=self._dt, var=self._var) - self

        return NotImplemented

    def __mul__(
        self,
        other: DiscreteTransferFunction | float | np.number[Any],
    ) -> DiscreteTransferFunction:
        """Series (cascade) interconnection (self * other)."""
        if isinstance(other, (int, float, np.number)):
            return DiscreteTransferFunction(
                float(other) * self._num, self._den, dt=self._dt, var=self._var
            )

        if isinstance(other, DiscreteTransferFunction):
            if not np.isclose(self._dt, other._dt):
                raise ValueError(
                    f"Cannot cascade discrete transfer functions with different sampling times: "
                    f"{self._dt} s vs {other._dt} s."
                )
            num = np.polymul(self._num, other._num)
            den = np.polymul(self._den, other._den)
            return DiscreteTransferFunction(num, den, dt=self._dt, var=self._var)

        if isinstance(other, LinearTimeInvariant):
            raise TypeError("Cannot multiply discrete system with continuous system.")

        return NotImplemented

    def __rmul__(
        self,
        other: DiscreteTransferFunction | float | np.number[Any],
    ) -> DiscreteTransferFunction:
        """Series interconnection with left operand (other * self)."""
        return self.__mul__(other)

    def __truediv__(
        self,
        other: DiscreteTransferFunction | float | np.number[Any],
    ) -> DiscreteTransferFunction:
        """Division (self / other)."""
        if isinstance(other, (int, float, np.number)):
            k = float(other)
            if np.isclose(k, 0.0):
                raise ZeroDivisionError("Cannot divide DiscreteTransferFunction by zero.")
            return DiscreteTransferFunction(self._num, self._den * k, dt=self._dt, var=self._var)

        if isinstance(other, DiscreteTransferFunction):
            if not np.isclose(self._dt, other._dt):
                raise ValueError(
                    f"Cannot divide discrete transfer functions with different sampling times: "
                    f"{self._dt} s vs {other._dt} s."
                )
            num = np.polymul(self._num, other._den)
            den = np.polymul(self._den, other._num)
            return DiscreteTransferFunction(num, den, dt=self._dt, var=self._var)

        return NotImplemented

    def __rtruediv__(
        self,
        other: DiscreteTransferFunction | float | np.number[Any],
    ) -> DiscreteTransferFunction:
        """Division with left operand (other / self)."""
        if isinstance(other, (int, float, np.number)):
            return DiscreteTransferFunction(float(other), 1.0, dt=self._dt, var=self._var) / self

        return NotImplemented

    def feedback(
        self,
        other: DiscreteTransferFunction | float | np.number[Any] = 1.0,
        sign: float = -1.0,
    ) -> DiscreteTransferFunction:
        r"""Closed-loop feedback interconnection of discrete transfer functions.

        .. math::

            T(z) = \frac{G(z)}{1 - \text{sign} \cdot G(z) H(z)}

        Parameters
        ----------
        other : DiscreteTransferFunction | float | int, optional
            Feedback path transfer function $H(z)$, default is 1.0 (unity feedback).
        sign : float, optional
            Feedback sign, default is -1.0 for standard negative feedback.

        Returns
        -------
        DiscreteTransferFunction
            Closed-loop discrete transfer function.
        """
        if isinstance(other, (int, float, np.number)):
            num_h = np.array([float(other)], dtype=np.float64)
            den_h = np.array([1.0], dtype=np.float64)
        elif isinstance(other, DiscreteTransferFunction):
            if not np.isclose(self._dt, other._dt):
                raise ValueError(
                    f"Cannot feedback discrete transfer functions with different sampling times: "
                    f"{self._dt} s vs {other._dt} s."
                )
            num_h = other._num
            den_h = other._den
        else:
            raise TypeError(
                f"Unsupported feedback block type for DiscreteTransferFunction: {type(other).__name__}"
            )

        num = np.polymul(self._num, den_h)
        den = np.polyadd(
            np.polymul(self._den, den_h),
            -float(sign) * np.polymul(self._num, num_h),
        )
        return DiscreteTransferFunction(num, den, dt=self._dt, var=self._var)

    def __repr__(self) -> str:
        """String representation suitable for code reproduction."""
        return (
            f"DiscreteTransferFunction(num={self._num.tolist()}, "
            f"den={self._den.tolist()}, dt={self._dt}, var='{self._var}')"
        )

    def __str__(self) -> str:
        """Formatted ASCII fraction representation with sampling time."""
        num_str = _format_discrete_poly_str(self._num, var=self._var)
        den_str = _format_discrete_poly_str(self._den, var=self._var)
        width = max(len(num_str), len(den_str)) + 2
        dash_line = "-" * width
        return (
            f"DiscreteTransferFunction (Ts = {self._dt} s):\n"
            f"{num_str.center(width)}\n"
            f"{dash_line}\n"
            f"{den_str.center(width)}"
        )

    def _repr_latex_(self) -> str:
        """LaTeX representation for Jupyter environments displaying H(z) and Ts."""
        num_latex = _format_discrete_poly_latex(self._num, var=self._var)
        den_latex = _format_discrete_poly_latex(self._den, var=self._var)
        var_symbol = "z" if self._var not in ("z^-1", "z^{-1}") else "z^{-1}"
        return (
            r"$$\begin{aligned} "
            rf"H({var_symbol}) &= \frac{{{num_latex}}}{{{den_latex}}} \\ "
            rf"T_s &= {self._dt:g}\text{{ s}} "
            r"\end{aligned}$$"
        )

    def _repr_markdown_(self) -> str:
        """Markdown representation for Jupyter environments."""
        return self._repr_latex_()


# Convenient alias
dtf = DiscreteTransferFunction


def _matched_c2d(
    num: NDArray[np.float64],
    den: NDArray[np.float64],
    dt: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute matched pole-zero discretization for a SISO continuous transfer function."""
    poles = np.roots(den) if len(den) > 1 else np.array([], dtype=np.complex128)
    zeros = (
        np.roots(num)
        if (len(num) > 1 and not np.all(np.isclose(num, 0.0)))
        else np.array([], dtype=np.complex128)
    )

    # 1. Discrete poles
    d_poles = np.exp(poles * dt)
    if len(d_poles) > 0:
        den_d = np.real(np.poly(d_poles))
    else:
        den_d = np.array([1.0], dtype=np.float64)

    # 2. Discrete zeros
    d_zeros: list[complex | float] = list(np.exp(zeros * dt)) if len(zeros) > 0 else []

    # 3. Add (r - 1) zeros at z = -1 for relative degree r = n - m
    r = len(poles) - len(zeros)
    n_inf = max(0, r - 1)
    for _ in range(n_inf):
        d_zeros.append(-1.0)

    if len(d_zeros) > 0:
        num_d_unscaled = np.real(np.poly(d_zeros))
    else:
        num_d_unscaled = np.array([1.0], dtype=np.float64)

    # 4. Gain matching
    has_origin_pole = np.any(np.isclose(poles, 0.0, atol=1e-5))
    if not has_origin_pole:
        g0 = float(np.polyval(num, 0.0) / np.polyval(den, 0.0))
        h0_unscaled = float(np.polyval(num_d_unscaled, 1.0) / np.polyval(den_d, 1.0))
        if not np.isclose(h0_unscaled, 0.0):
            gain = g0 / h0_unscaled
        else:
            gain = 1.0
    else:
        w0 = min(1.0 / dt, 0.1)
        s0 = 1j * w0
        z0 = np.exp(1j * w0 * dt)
        g_val = np.polyval(num, s0) / np.polyval(den, s0)
        h_unscaled = np.polyval(num_d_unscaled, z0) / np.polyval(den_d, z0)
        gain = float(np.abs(g_val) / max(np.abs(h_unscaled), 1e-12))

    num_d = num_d_unscaled * gain
    return num_d, den_d


def c2d(
    sys: LinearTimeInvariant | TransferFunction | StateSpace,
    dt: float,
    method: Literal[
        "zoh",
        "zero_order_hold",
        "foh",
        "first_order_hold",
        "tustin",
        "bilinear",
        "prewarping",
        "tustin_prewarp",
        "bilinear_prewarp",
        "matched",
    ] = "zoh",
    prewarp_frequency: float | None = None,
    w_warp: float | None = None,
) -> DiscreteTransferFunction:
    r"""Discretize a continuous-time LTI system into a discrete-time TransferFunction $H(z)$.

    Supports standard continuous-to-discrete (C2D) transformation methods:
    1. **Zero-Order Hold (ZOH)**: Assumes piecewise constant input $u(t) = u[k]$ over each interval.
    2. **First-Order Hold (FOH)**: Assumes linear interpolation between sample points $u[k]$ and $u[k+1]$.
    3. **Tustin (Bilinear) Transform**: Trapezoidal numerical integration substitution
       $s = \frac{2}{T_s} \frac{z-1}{z+1}$.
    4. **Tustin with Frequency Pre-warping**: Bilinear substitution adjusted to match
       exact frequency response at a critical continuous frequency $\omega_{\text{warp}}$:
       $s = \frac{\omega_{\text{warp}}}{\tan(\omega_{\text{warp}} T_s / 2)} \frac{z-1}{z+1}$.
    5. **Matched Pole-Zero Method**: Direct exponential mapping of poles and zeros $z_i = e^{s_i T_s}$
       with added zeros at $z = -1$ for relative degree matching.

    Parameters
    ----------
    sys : LinearTimeInvariant | TransferFunction | StateSpace
        Continuous-time LTI system to discretize.
    dt : float
        Sampling period $T_s > 0$ in seconds.
    method : {"zoh", "foh", "tustin", "bilinear", "prewarping", "matched"}, optional
        Discretization method, default is "zoh".
    prewarp_frequency : float | None, optional
        Pre-warping frequency $\omega_{\text{warp}} > 0$ in rad/s (for Tustin pre-warping).
    w_warp : float | None, optional
        Alias for `prewarp_frequency`.

    Returns
    -------
    DiscreteTransferFunction
        Discretized transfer function $H(z)$ with sampling time $T_s = \text{dt}$.

    Raises
    ------
    ValueError
        If sampling time $T_s \le 0$, prewarp frequency is invalid, or unknown method.
    TypeError
        If `sys` is not a continuous LTI system.
    """
    if not isinstance(sys, LinearTimeInvariant):
        raise TypeError(
            f"Expected continuous LinearTimeInvariant instance, got {type(sys).__name__}."
        )

    if dt <= 0.0:
        raise ValueError(f"Sampling period dt (Ts) must be strictly positive, got {dt}.")

    from ctrlpy.models.state_space import StateSpace
    from ctrlpy.models.transfer_function import TransferFunction

    if isinstance(sys, StateSpace):
        if not sys.is_siso:
            raise NotImplementedError(
                "c2d currently supports Single-Input Single-Output (SISO) systems."
            )
        tf_sys = sys.to_tf()
    elif isinstance(sys, TransferFunction):
        tf_sys = sys
    else:
        raise TypeError(f"Unsupported continuous system type: {type(sys).__name__}")

    method_clean = method.lower().strip()

    # Pre-warping frequency handling
    warp_freq = prewarp_frequency if prewarp_frequency is not None else w_warp

    if warp_freq is not None or method_clean in (
        "prewarping",
        "tustin_prewarp",
        "bilinear_prewarp",
    ):
        if warp_freq is None:
            raise ValueError(
                "Tustin with pre-warping requires 'prewarp_frequency' (or 'w_warp') to be specified."
            )
        if warp_freq <= 0.0:
            raise ValueError(
                f"prewarp_frequency must be strictly positive (w > 0), got {warp_freq}."
            )
        w_nyquist = np.pi / dt
        if warp_freq >= w_nyquist:
            raise ValueError(
                f"prewarp_frequency ({warp_freq:.4g} rad/s) must be strictly below "
                f"the Nyquist frequency pi/dt ({w_nyquist:.4g} rad/s)."
            )

        dt_eff = float(2.0 * np.tan(warp_freq * dt / 2.0) / warp_freq)
        num_d, den_d, _ = signal.cont2discrete((tf_sys.num, tf_sys.den), dt_eff, method="bilinear")
        return DiscreteTransferFunction(num_d.ravel(), den_d.ravel(), dt=dt)

    if method_clean in ("zoh", "zero_order_hold"):
        num_d, den_d, _ = signal.cont2discrete((tf_sys.num, tf_sys.den), dt, method="zoh")
        return DiscreteTransferFunction(num_d.ravel(), den_d.ravel(), dt=dt)

    if method_clean in ("foh", "first_order_hold"):
        num_d, den_d, _ = signal.cont2discrete((tf_sys.num, tf_sys.den), dt, method="foh")
        return DiscreteTransferFunction(num_d.ravel(), den_d.ravel(), dt=dt)

    if method_clean in ("tustin", "bilinear"):
        num_d, den_d, _ = signal.cont2discrete((tf_sys.num, tf_sys.den), dt, method="bilinear")
        return DiscreteTransferFunction(num_d.ravel(), den_d.ravel(), dt=dt)

    if method_clean in ("matched", "matched_z", "matched_pole_zero"):
        num_d, den_d = _matched_c2d(tf_sys.num, tf_sys.den, dt)
        return DiscreteTransferFunction(num_d, den_d, dt=dt)

    raise ValueError(
        f"Unknown discretization method '{method}'. "
        f"Supported methods: 'zoh', 'foh', 'tustin', 'bilinear', 'prewarping', 'matched'."
    )


def discrete_step_response(
    sys: DiscreteLTI,
    T: Sequence[float] | NDArray[np.floating[Any]] | float | None = None,
    n_steps: int | None = None,
) -> TimeResponseData:
    r"""Compute the step response of a discrete-time LTI system iteratively.

    Simulates the difference equation for unit step excitation:

    .. math::

        u[k] = 1[k] = \begin{cases} 1, & k \ge 0 \\ 0, & k < 0 \end{cases}

    Parameters
    ----------
    sys : DiscreteLTI
        Discrete-time LTI system.
    T : Sequence[float] | NDArray[np.floating] | float | None, optional
        Simulation duration or explicit time vector.
    n_steps : int | None, optional
        Number of sample steps $N$.

    Returns
    -------
    TimeResponseData
        Simulation response object containing time points `t`, output trajectory `y`,
        and discrete system reference.
    """
    if not isinstance(sys, DiscreteLTI):
        raise TypeError(f"Expected DiscreteLTI instance, got {type(sys).__name__}.")

    if not isinstance(sys, DiscreteTransferFunction):
        raise TypeError(f"Unsupported discrete system type: {type(sys).__name__}")

    dt = sys.dt
    if n_steps is not None:
        if n_steps < 2:
            raise ValueError(f"n_steps must be at least 2, got {n_steps}.")
        t_arr = np.arange(n_steps, dtype=np.float64) * dt
    elif isinstance(T, (int, float, np.number)):
        duration = float(T)
        if duration <= 0.0:
            raise ValueError(f"Duration T must be positive, got {duration}.")
        n_pts = max(2, int(np.round(duration / dt)) + 1)
        t_arr = np.arange(n_pts, dtype=np.float64) * dt
    elif T is not None:
        t_arr = np.asarray(T, dtype=np.float64).ravel()
        if t_arr.size < 2:
            raise ValueError("Time vector T must contain at least 2 points.")
    else:
        # Automatic time horizon estimation from discrete poles
        poles = sys.poles()
        stable_mags = [float(np.abs(p)) for p in poles if np.abs(p) < 1.0 - 1e-6]
        unstable_mags = [float(np.abs(p)) for p in poles if np.abs(p) > 1.0 + 1e-6]

        if stable_mags:
            slowest_mag = max(stable_mags)
            if slowest_mag > 1e-4:
                # Equivalent continuous pole: sigma = ln(|p|) / dt
                sigma = abs(float(np.log(slowest_mag))) / dt
                tau = 1.0 / sigma
                t_final = max(7.0 * tau, 10.0 * dt)
            else:
                t_final = 10.0 * dt
            n_pts = max(30, min(2000, int(np.ceil(t_final / dt)) + 1))
        elif unstable_mags:
            n_pts = 40
        else:
            n_pts = 50
        t_arr = np.arange(n_pts, dtype=np.float64) * dt

    u = np.ones(len(t_arr), dtype=np.float64)

    # Pad numerator with leading zeros to match denominator degree
    deg_diff = len(sys.den) - len(sys.num)
    if deg_diff > 0:
        b_pad = np.pad(sys.num, (deg_diff, 0), mode="constant")
    elif deg_diff < 0:
        raise ValueError(
            f"Improper / non-causal DiscreteTransferFunction (num degree {len(sys.num) - 1} > "
            f"den degree {len(sys.den) - 1}) cannot be simulated forward in time."
        )
    else:
        b_pad = sys.num

    y = signal.lfilter(b_pad, sys.den, u)
    return TimeResponseData(t=t_arr, y=y, x=None, sys=sys, poles=sys.poles())


def discrete_impulse_response(
    sys: DiscreteLTI,
    T: Sequence[float] | NDArray[np.floating[Any]] | float | None = None,
    n_steps: int | None = None,
) -> TimeResponseData:
    r"""Compute the impulse response of a discrete-time LTI system iteratively.

    Simulates the difference equation for unit discrete impulse (Kronecker delta):

    .. math::

        u[k] = \delta[k] = \begin{cases} 1, & k = 0 \\ 0, & k \neq 0 \end{cases}

    Parameters
    ----------
    sys : DiscreteLTI
        Discrete-time LTI system.
    T : Sequence[float] | NDArray[np.floating] | float | None, optional
        Simulation duration or explicit time vector.
    n_steps : int | None, optional
        Number of sample steps $N$.

    Returns
    -------
    TimeResponseData
        Simulation response object.
    """
    if not isinstance(sys, DiscreteLTI):
        raise TypeError(f"Expected DiscreteLTI instance, got {type(sys).__name__}.")

    if not isinstance(sys, DiscreteTransferFunction):
        raise TypeError(f"Unsupported discrete system type: {type(sys).__name__}")

    dt = sys.dt
    if n_steps is not None:
        if n_steps < 2:
            raise ValueError(f"n_steps must be at least 2, got {n_steps}.")
        t_arr = np.arange(n_steps, dtype=np.float64) * dt
    elif isinstance(T, (int, float, np.number)):
        duration = float(T)
        if duration <= 0.0:
            raise ValueError(f"Duration T must be positive, got {duration}.")
        n_pts = max(2, int(np.round(duration / dt)) + 1)
        t_arr = np.arange(n_pts, dtype=np.float64) * dt
    elif T is not None:
        t_arr = np.asarray(T, dtype=np.float64).ravel()
        if t_arr.size < 2:
            raise ValueError("Time vector T must contain at least 2 points.")
    else:
        poles = sys.poles()
        stable_mags = [float(np.abs(p)) for p in poles if np.abs(p) < 1.0 - 1e-6]
        unstable_mags = [float(np.abs(p)) for p in poles if np.abs(p) > 1.0 + 1e-6]

        if stable_mags:
            slowest_mag = max(stable_mags)
            if slowest_mag > 1e-4:
                sigma = abs(float(np.log(slowest_mag))) / dt
                tau = 1.0 / sigma
                t_final = max(7.0 * tau, 10.0 * dt)
            else:
                t_final = 10.0 * dt
            n_pts = max(30, min(2000, int(np.ceil(t_final / dt)) + 1))
        elif unstable_mags:
            n_pts = 40
        else:
            n_pts = 50
        t_arr = np.arange(n_pts, dtype=np.float64) * dt

    u = np.zeros(len(t_arr), dtype=np.float64)
    u[0] = 1.0

    deg_diff = len(sys.den) - len(sys.num)
    if deg_diff > 0:
        b_pad = np.pad(sys.num, (deg_diff, 0), mode="constant")
    elif deg_diff < 0:
        raise ValueError(
            f"Improper / non-causal DiscreteTransferFunction (num degree {len(sys.num) - 1} > "
            f"den degree {len(sys.den) - 1}) cannot be simulated forward in time."
        )
    else:
        b_pad = sys.num

    y = signal.lfilter(b_pad, sys.den, u)
    return TimeResponseData(t=t_arr, y=y, x=None, sys=sys, poles=sys.poles())


def discrete_forced_response(
    sys: DiscreteLTI,
    U: Sequence[float] | NDArray[np.floating[Any]] | float,
    T: Sequence[float] | NDArray[np.floating[Any]] | None = None,
    n_steps: int | None = None,
) -> TimeResponseData:
    r"""Compute the simulation response of a discrete LTI system to arbitrary input sequence $u[k]$.

    Parameters
    ----------
    sys : DiscreteLTI
        Discrete-time LTI system.
    U : Sequence[float] | NDArray[np.floating] | float
        Input sequence $u[k]$.
    T : Sequence[float] | NDArray[np.floating] | None, optional
        Simulation time points.
    n_steps : int | None, optional
        Number of steps.

    Returns
    -------
    TimeResponseData
        Simulation response object.
    """
    if not isinstance(sys, DiscreteLTI):
        raise TypeError(f"Expected DiscreteLTI instance, got {type(sys).__name__}.")

    if not isinstance(sys, DiscreteTransferFunction):
        raise TypeError(f"Unsupported discrete system type: {type(sys).__name__}")

    dt = sys.dt
    if isinstance(U, (int, float, np.number)):
        if n_steps is not None:
            u_arr = np.full(n_steps, float(U), dtype=np.float64)
        elif T is not None:
            t_eval = np.asarray(T, dtype=np.float64).ravel()
            u_arr = np.full(len(t_eval), float(U), dtype=np.float64)
        else:
            u_arr = np.full(50, float(U), dtype=np.float64)
    else:
        u_arr = np.asarray(U, dtype=np.float64).ravel()

    n_pts = len(u_arr)
    if T is not None:
        t_arr = np.asarray(T, dtype=np.float64).ravel()
        if len(t_arr) != n_pts:
            raise ValueError(
                f"Input U length {n_pts} does not match time vector T length {len(t_arr)}."
            )
    else:
        t_arr = np.arange(n_pts, dtype=np.float64) * dt

    deg_diff = len(sys.den) - len(sys.num)
    if deg_diff > 0:
        b_pad = np.pad(sys.num, (deg_diff, 0), mode="constant")
    elif deg_diff < 0:
        raise ValueError("Improper discrete transfer function cannot be simulated.")
    else:
        b_pad = sys.num

    y = signal.lfilter(b_pad, sys.den, u_arr)
    return TimeResponseData(t=t_arr, y=y, x=None, sys=sys, poles=sys.poles())


def discrete_bode_data(
    sys: DiscreteLTI,
    omega: Sequence[float] | NDArray[np.floating[Any]] | None = None,
    n_points: int = 500,
) -> BodeData:
    r"""Compute the discrete frequency response over frequencies $\omega \in (0, \pi / T_s]$.

    Evaluates:

    .. math::

        H(e^{j \omega T_s}) = \frac{N(e^{j \omega T_s})}{D(e^{j \omega T_s})}

    Parameters
    ----------
    sys : DiscreteLTI
        Discrete-time LTI system.
    omega : Sequence[float] | NDArray[np.floating] | None, optional
        Frequencies in rad/s. If None, generated up to Nyquist limit $\omega_N = \pi / T_s$.
    n_points : int, optional
        Number of logarithmic frequency points, default is 500.

    Returns
    -------
    BodeData
        Bode data container.
    """
    if not isinstance(sys, DiscreteLTI):
        raise TypeError(f"Expected DiscreteLTI instance, got {type(sys).__name__}.")

    if not isinstance(sys, DiscreteTransferFunction):
        raise TypeError(f"Unsupported discrete system type: {type(sys).__name__}")

    dt = sys.dt
    w_nyquist = np.pi / dt

    if omega is not None:
        w_arr = np.asarray(omega, dtype=np.float64).ravel()
        if w_arr.size < 2:
            raise ValueError("Frequency vector omega must contain at least 2 points.")
        if np.any(w_arr <= 0.0):
            raise ValueError("Frequencies must be strictly positive (w > 0).")
    else:
        w_min = 1e-3
        w_max = w_nyquist
        w_arr = np.logspace(np.log10(w_min), np.log10(w_max), num=n_points, dtype=np.float64)

    # Evaluate at z = exp(j * omega * Ts)
    z_eval = np.exp(1j * w_arr * dt)
    num_eval = np.polyval(sys.num, z_eval)
    den_eval = np.polyval(sys.den, z_eval)
    resp = num_eval / den_eval

    mag = np.abs(resp).astype(np.float64)
    with np.errstate(divide="ignore"):
        mag_db = np.where(mag > 0.0, 20.0 * np.log10(mag), -np.inf)

    phase_rad = np.unwrap(np.angle(resp)).astype(np.float64)
    phase_deg = np.rad2deg(phase_rad)

    return BodeData(
        w=w_arr,
        mag=mag,
        phase=phase_deg,
        mag_db=mag_db,
        phase_rad=phase_rad,
        response=resp,
    )


def plot_pzmap(
    sys: DiscreteLTI,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    r"""Plot the discrete pole-zero map in the complex $z$-plane with unit circle $|z|=1$ (Matplotlib).

    Parameters
    ----------
    sys : DiscreteLTI
        Discrete-time LTI system.
    ax : Axes | None, optional
        Matplotlib Axes to plot into.
    title : str | None, optional
        Custom title.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib Figure and Axes.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax_out = plt.subplots(figsize=(6, 6))
    else:
        ax_out = ax
        fig = cast("Figure", ax_out.figure)

    # Draw Unit Circle: |z| = 1
    theta = np.linspace(0.0, 2.0 * np.pi, 300)
    ax_out.plot(
        np.cos(theta),
        np.sin(theta),
        "k--",
        linewidth=1.2,
        alpha=0.7,
        label=r"Unit Circle $|z| = 1$",
    )

    # Reference axes
    ax_out.axhline(0.0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    ax_out.axvline(0.0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)

    poles = sys.poles()
    zeros = sys.zeros()

    if len(poles) > 0:
        ax_out.plot(
            poles.real,
            poles.imag,
            "rx",
            markersize=9,
            markeredgewidth=2.0,
            label="Discrete Poles",
        )

    if len(zeros) > 0:
        ax_out.plot(
            zeros.real,
            zeros.imag,
            "go",
            markersize=8,
            markeredgewidth=2.0,
            fillstyle="none",
            label="Discrete Zeros",
        )

    ax_out.set_aspect("equal")
    ax_out.set_xlabel(r"$\mathrm{Re}(z)$")
    ax_out.set_ylabel(r"$\mathrm{Im}(z)$")

    status = sys.stability()
    header_title = (
        title
        or f"Discrete Pole-Zero Map ($z$-plane, $T_s = {sys.dt}$ s)\nStatus: {status.capitalize()}"
    )
    ax_out.set_title(header_title)
    ax_out.grid(True, linestyle="--", alpha=0.5)
    ax_out.legend(loc="upper right")

    # Determine bounding box
    all_re = np.concatenate([[-1.2, 1.2], poles.real, zeros.real])
    all_im = np.concatenate([[-1.2, 1.2], poles.imag, zeros.imag])
    max_bound = max(1.3, float(np.max(np.abs(np.concatenate([all_re, all_im])))) * 1.15)
    ax_out.set_xlim(-max_bound, max_bound)
    ax_out.set_ylim(-max_bound, max_bound)

    return fig, ax_out


def iplot_pzmap(
    sys: DiscreteLTI,
    title: str | None = None,
) -> go.Figure:
    r"""Plot interactive discrete pole-zero map in the complex $z$-plane using Plotly.

    Parameters
    ----------
    sys : DiscreteLTI
        Discrete-time LTI system.
    title : str | None, optional
        Custom title.

    Returns
    -------
    go.Figure
        Interactive Plotly Figure.
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    # Unit circle trace
    theta = np.linspace(0.0, 2.0 * np.pi, 300)
    fig.add_trace(
        go.Scatter(
            x=np.cos(theta),
            y=np.sin(theta),
            mode="lines",
            name="Unit Circle (|z|=1)",
            line={"color": "gray", "dash": "dash", "width": 1.5},
            hoverinfo="skip",
        )
    )

    # Reference lines
    fig.add_hline(y=0.0, line_dash="solid", line_color="lightgray", line_width=0.8)
    fig.add_vline(x=0.0, line_dash="solid", line_color="lightgray", line_width=0.8)

    poles = sys.poles()
    zeros = sys.zeros()
    dt = sys.dt

    if len(poles) > 0:
        p_re = poles.real
        p_im = poles.imag
        p_mag = np.abs(poles)
        p_angle_deg = np.rad2deg(np.angle(poles))
        p_freq = np.abs(np.angle(poles)) / dt

        customdata = np.column_stack([p_mag, p_angle_deg, p_freq])

        fig.add_trace(
            go.Scatter(
                x=p_re,
                y=p_im,
                mode="markers",
                name="Discrete Poles",
                marker={
                    "symbol": "x",
                    "size": 11,
                    "color": "red",
                    "line": {"width": 2.5},
                },
                customdata=customdata,
                hovertemplate=(
                    "<b>Pole</b>: %{x:.4f} + %{y:.4f}j<br>"
                    "<b>Magnitude |z|</b>: %{customdata[0]:.4f}<br>"
                    "<b>Angle ∠z</b>: %{customdata[1]:.2f}°<br>"
                    "<b>Freq (ω)</b>: %{customdata[2]:.4g} rad/s<extra></extra>"
                ),
            )
        )

    if len(zeros) > 0:
        z_re = zeros.real
        z_im = zeros.imag
        z_mag = np.abs(zeros)
        z_angle_deg = np.rad2deg(np.angle(zeros))
        z_freq = np.abs(np.angle(zeros)) / dt

        customdata_z = np.column_stack([z_mag, z_angle_deg, z_freq])

        fig.add_trace(
            go.Scatter(
                x=z_re,
                y=z_im,
                mode="markers",
                name="Discrete Zeros",
                marker={
                    "symbol": "circle-open",
                    "size": 10,
                    "color": "green",
                    "line": {"width": 2.0},
                },
                customdata=customdata_z,
                hovertemplate=(
                    "<b>Zero</b>: %{x:.4f} + %{y:.4f}j<br>"
                    "<b>Magnitude |z|</b>: %{customdata[0]:.4f}<br>"
                    "<b>Angle ∠z</b>: %{customdata[1]:.2f}°<br>"
                    "<b>Freq (ω)</b>: %{customdata[2]:.4g} rad/s<extra></extra>"
                ),
            )
        )

    status = sys.stability()
    chart_title = (
        title or f"Discrete Pole-Zero Map (z-plane, Ts = {sys.dt} s) — {status.capitalize()}"
    )

    all_re = np.concatenate([[-1.2, 1.2], poles.real, zeros.real])
    all_im = np.concatenate([[-1.2, 1.2], poles.imag, zeros.imag])
    max_bound = max(1.3, float(np.max(np.abs(np.concatenate([all_re, all_im])))) * 1.15)

    fig.update_layout(
        title=chart_title,
        xaxis_title="Real Axis (Re)",
        yaxis_title="Imaginary Axis (Im)",
        template="plotly_white",
        width=600,
        height=600,
        xaxis={"range": [-max_bound, max_bound]},
        yaxis={"range": [-max_bound, max_bound], "scaleanchor": "x", "scaleratio": 1},
    )

    return fig


# Alias
plot_pzmap_plotly = iplot_pzmap

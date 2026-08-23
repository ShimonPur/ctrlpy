"""Base classes for Linear Time-Invariant (LTI) systems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import plotly.graph_objects as go
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ctrlpy.freq_domain import (
        BodeData,
        NyquistData,
        RootLocusData,
        StabilityMargins,
    )
    from ctrlpy.simulation_results import TimeResponseData


class LinearTimeInvariant(ABC):
    """Abstract base class representing a Linear Time-Invariant (LTI) system.

    This class defines the common interface and properties for all LTI models,
    such as transfer functions and state-space models.
    """

    @property
    @abstractmethod
    def inputs(self) -> int:
        """Number of inputs to the system.

        Returns
        -------
        int
            The number of input channels.
        """
        ...

    @property
    @abstractmethod
    def outputs(self) -> int:
        """Number of outputs from the system.

        Returns
        -------
        int
            The number of output channels.
        """
        ...

    @property
    def is_siso(self) -> bool:
        """Whether the system is Single-Input Single-Output (SISO).

        Returns
        -------
        bool
            True if inputs == 1 and outputs == 1, False otherwise.
        """
        return self.inputs == 1 and self.outputs == 1

    @abstractmethod
    def poles(self) -> NDArray[np.complex128]:
        """Compute the poles of the system.

        Returns
        -------
        NDArray[np.complex128]
            1D array of system poles.
        """
        ...

    @abstractmethod
    def zeros(self) -> NDArray[np.complex128]:
        """Compute the zeros of the system.

        Returns
        -------
        NDArray[np.complex128]
            1D array of system zeros.
        """
        ...

    @abstractmethod
    def __add__(self, other: Any) -> Any:
        """Parallel interconnection."""
        ...

    @abstractmethod
    def __radd__(self, other: Any) -> Any:
        """Parallel interconnection with left operand."""
        ...

    @abstractmethod
    def __sub__(self, other: Any) -> Any:
        """Subtraction."""
        ...

    @abstractmethod
    def __rsub__(self, other: Any) -> Any:
        """Subtraction with left operand."""
        ...

    @abstractmethod
    def __mul__(self, other: Any) -> Any:
        """Series (cascade) interconnection."""
        ...

    @abstractmethod
    def __rmul__(self, other: Any) -> Any:
        """Series interconnection with left operand."""
        ...

    @abstractmethod
    def __neg__(self) -> Any:
        """Negation."""
        ...

    @abstractmethod
    def feedback(self, other: Any = 1, sign: float = -1) -> Any:
        """Closed-loop feedback interconnection."""
        ...

    def step(
        self,
        T: Sequence[float] | NDArray[np.floating] | float | None = None,
        X0: Sequence[float] | NDArray[np.floating] | None = None,
        input_index: int = 0,
    ) -> TimeResponseData:
        """Compute the step response of the system.

        Parameters
        ----------
        T : Sequence[float] | NDArray[np.floating] | float | None, optional
            Time vector or duration. If None, automatically determined.
        X0 : Sequence[float] | NDArray[np.floating] | None, optional
            Initial state vector (for StateSpace systems).
        input_index : int, optional
            Input channel index, default is 0.

        Returns
        -------
        TimeResponseData
            Simulation response object.
        """
        from ctrlpy.time_domain import step_response

        return step_response(self, T=T, X0=X0, input_index=input_index)

    def impulse(
        self,
        T: Sequence[float] | NDArray[np.floating] | float | None = None,
        X0: Sequence[float] | NDArray[np.floating] | None = None,
        input_index: int = 0,
    ) -> TimeResponseData:
        """Compute the impulse response of the system.

        Parameters
        ----------
        T : Sequence[float] | NDArray[np.floating] | float | None, optional
            Time vector or duration. If None, automatically determined.
        X0 : Sequence[float] | NDArray[np.floating] | None, optional
            Initial state vector (for StateSpace systems).
        input_index : int, optional
            Input channel index, default is 0.

        Returns
        -------
        TimeResponseData
            Simulation response object.
        """
        from ctrlpy.time_domain import impulse_response

        return impulse_response(self, T=T, X0=X0, input_index=input_index)

    def bode(
        self,
        omega: Sequence[float] | NDArray[np.floating] | None = None,
    ) -> BodeData:
        """Compute the frequency response (Bode) data for the system.

        Parameters
        ----------
        omega : Sequence[float] | NDArray[np.floating] | None, optional
            Frequency vector in rad/s.

        Returns
        -------
        BodeData
            Bode response object.
        """
        from ctrlpy.freq_domain import bode_data

        return bode_data(self, omega=omega)

    def nyquist(
        self,
        omega: Sequence[float] | NDArray[np.floating] | None = None,
    ) -> NyquistData:
        """Compute the Nyquist frequency response data for the system.

        Parameters
        ----------
        omega : Sequence[float] | NDArray[np.floating] | None, optional
            Frequency vector in rad/s.

        Returns
        -------
        NyquistData
            Nyquist response object.
        """
        from ctrlpy.freq_domain import nyquist_data

        return nyquist_data(self, omega=omega)

    def margin(self) -> StabilityMargins:
        """Compute the stability margins of the system.

        Returns
        -------
        StabilityMargins
            Dataclass with GM, PM, Wcg, and Wcp.
        """
        from ctrlpy.freq_domain import margin

        return margin(self)

    def root_locus(
        self,
        gains: Sequence[float] | NDArray[np.floating] | None = None,
    ) -> RootLocusData:
        """Compute closed-loop root locus trajectories for varying gains.

        Parameters
        ----------
        gains : Sequence[float] | NDArray[np.floating] | None, optional
            Gain values k >= 0.

        Returns
        -------
        RootLocusData
            Root locus trajectory data.
        """
        from ctrlpy.freq_domain import root_locus_data

        return root_locus_data(self, gains=gains)

    def plot_step(
        self,
        T: Sequence[float] | NDArray[np.floating[Any]] | float | None = None,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        ax: Axes | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes] | go.Figure:
        """Plot the step response using Matplotlib or Plotly.

        Parameters
        ----------
        T : Sequence[float] | NDArray[np.floating] | float | None, optional
            Time vector or simulation duration.
        backend : {"matplotlib", "plotly"}, optional
            Plotting backend, default is "matplotlib".
        ax : Axes | None, optional
            Existing Matplotlib Axes (for backend="matplotlib").
        **kwargs : Any
            Additional keyword arguments passed to the plotting routine.

        Returns
        -------
        tuple[Figure, Axes] | go.Figure
            Matplotlib (fig, ax) tuple or Plotly Figure.
        """
        if backend == "plotly":
            from ctrlpy.plotting_plotly import iplot_step

            return iplot_step(self, T=T, **kwargs)
        elif backend == "matplotlib":
            from ctrlpy.plotting import plot_step

            return plot_step(self, T=T, ax=ax, **kwargs)
        else:
            raise ValueError(f"Unknown backend '{backend}'. Supported: 'matplotlib', 'plotly'.")

    def iplot_step(
        self,
        T: Sequence[float] | NDArray[np.floating[Any]] | float | None = None,
        **kwargs: Any,
    ) -> go.Figure:
        """Plot interactive step response using Plotly.

        Parameters
        ----------
        T : Sequence[float] | NDArray[np.floating] | float | None, optional
            Time vector or simulation duration.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        go.Figure
            Interactive Plotly Figure.
        """
        from ctrlpy.plotting_plotly import iplot_step

        return iplot_step(self, T=T, **kwargs)

    def plot_impulse(
        self,
        T: Sequence[float] | NDArray[np.floating[Any]] | float | None = None,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        ax: Axes | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes] | go.Figure:
        """Plot the impulse response using Matplotlib or Plotly.

        Parameters
        ----------
        T : Sequence[float] | NDArray[np.floating] | float | None, optional
            Time vector or simulation duration.
        backend : {"matplotlib", "plotly"}, optional
            Plotting backend, default is "matplotlib".
        ax : Axes | None, optional
            Existing Matplotlib Axes (for backend="matplotlib").
        **kwargs : Any
            Additional keyword arguments passed to the plotting routine.

        Returns
        -------
        tuple[Figure, Axes] | go.Figure
            Matplotlib (fig, ax) tuple or Plotly Figure.
        """
        if backend == "plotly":
            from ctrlpy.plotting_plotly import iplot_impulse

            return iplot_impulse(self, T=T, **kwargs)
        elif backend == "matplotlib":
            from ctrlpy.plotting import plot_impulse

            return plot_impulse(self, T=T, ax=ax, **kwargs)
        else:
            raise ValueError(f"Unknown backend '{backend}'. Supported: 'matplotlib', 'plotly'.")

    def iplot_impulse(
        self,
        T: Sequence[float] | NDArray[np.floating[Any]] | float | None = None,
        **kwargs: Any,
    ) -> go.Figure:
        """Plot interactive impulse response using Plotly.

        Parameters
        ----------
        T : Sequence[float] | NDArray[np.floating] | float | None, optional
            Time vector or simulation duration.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        go.Figure
            Interactive Plotly Figure.
        """
        from ctrlpy.plotting_plotly import iplot_impulse

        return iplot_impulse(self, T=T, **kwargs)

    def plot_bode(
        self,
        omega: Sequence[float] | NDArray[np.floating[Any]] | None = None,
        margins: bool = True,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        ax: Sequence[Axes] | NDArray[Any] | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, tuple[Axes, Axes]] | go.Figure:
        """Plot Bode diagram using Matplotlib or Plotly.

        Parameters
        ----------
        omega : Sequence[float] | NDArray[np.floating] | None, optional
            Frequency vector in rad/s.
        margins : bool, optional
            Whether to calculate and annotate stability margins, default is True.
        backend : {"matplotlib", "plotly"}, optional
            Plotting backend, default is "matplotlib".
        ax : Sequence[Axes] | NDArray[Any] | None, optional
            Sequence of 2 Matplotlib Axes (for backend="matplotlib").
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        tuple[Figure, tuple[Axes, Axes]] | go.Figure
            Matplotlib (fig, (ax_mag, ax_phase)) or Plotly Figure.
        """
        if backend == "plotly":
            from ctrlpy.plotting_plotly import iplot_bode

            return iplot_bode(self, omega=omega, margins=margins, **kwargs)
        elif backend == "matplotlib":
            from ctrlpy.plotting import plot_bode

            return plot_bode(self, omega=omega, margins=margins, ax=ax, **kwargs)
        else:
            raise ValueError(f"Unknown backend '{backend}'. Supported: 'matplotlib', 'plotly'.")

    def iplot_bode(
        self,
        omega: Sequence[float] | NDArray[np.floating[Any]] | None = None,
        margins: bool = True,
        **kwargs: Any,
    ) -> go.Figure:
        """Plot interactive Bode diagram using Plotly.

        Parameters
        ----------
        omega : Sequence[float] | NDArray[np.floating] | None, optional
            Frequency vector in rad/s.
        margins : bool, optional
            Whether to calculate and annotate stability margins, default is True.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        go.Figure
            Interactive Plotly Figure.
        """
        from ctrlpy.plotting_plotly import iplot_bode

        return iplot_bode(self, omega=omega, margins=margins, **kwargs)

    def plot_nyquist(
        self,
        omega: Sequence[float] | NDArray[np.floating[Any]] | None = None,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        ax: Axes | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes] | go.Figure:
        """Plot Nyquist diagram using Matplotlib or Plotly.

        Parameters
        ----------
        omega : Sequence[float] | NDArray[np.floating] | None, optional
            Frequency vector in rad/s.
        backend : {"matplotlib", "plotly"}, optional
            Plotting backend, default is "matplotlib".
        ax : Axes | None, optional
            Matplotlib Axes (for backend="matplotlib").
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        tuple[Figure, Axes] | go.Figure
            Matplotlib (fig, ax) or Plotly Figure.
        """
        if backend == "plotly":
            from ctrlpy.plotting_plotly import iplot_nyquist

            return iplot_nyquist(self, omega=omega, **kwargs)
        elif backend == "matplotlib":
            from ctrlpy.plotting import plot_nyquist

            return plot_nyquist(self, omega=omega, ax=ax, **kwargs)
        else:
            raise ValueError(f"Unknown backend '{backend}'. Supported: 'matplotlib', 'plotly'.")

    def iplot_nyquist(
        self,
        omega: Sequence[float] | NDArray[np.floating[Any]] | None = None,
        **kwargs: Any,
    ) -> go.Figure:
        """Plot interactive Nyquist diagram using Plotly.

        Parameters
        ----------
        omega : Sequence[float] | NDArray[np.floating] | None, optional
            Frequency vector in rad/s.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        go.Figure
            Interactive Plotly Figure.
        """
        from ctrlpy.plotting_plotly import iplot_nyquist

        return iplot_nyquist(self, omega=omega, **kwargs)

    def plot_root_locus(
        self,
        gains: Sequence[float] | NDArray[np.floating[Any]] | None = None,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        ax: Axes | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes] | go.Figure:
        """Plot Root Locus diagram using Matplotlib or Plotly.

        Parameters
        ----------
        gains : Sequence[float] | NDArray[np.floating] | None, optional
            Gain values k >= 0.
        backend : {"matplotlib", "plotly"}, optional
            Plotting backend, default is "matplotlib".
        ax : Axes | None, optional
            Matplotlib Axes (for backend="matplotlib").
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        tuple[Figure, Axes] | go.Figure
            Matplotlib (fig, ax) or Plotly Figure.
        """
        if backend == "plotly":
            from ctrlpy.plotting_plotly import iplot_root_locus

            return iplot_root_locus(self, gains=gains, **kwargs)
        elif backend == "matplotlib":
            from ctrlpy.plotting import plot_root_locus

            return plot_root_locus(self, gains=gains, ax=ax, **kwargs)
        else:
            raise ValueError(f"Unknown backend '{backend}'. Supported: 'matplotlib', 'plotly'.")

    def iplot_root_locus(
        self,
        gains: Sequence[float] | NDArray[np.floating[Any]] | None = None,
        **kwargs: Any,
    ) -> go.Figure:
        """Plot interactive Root Locus diagram using Plotly.

        Parameters
        ----------
        gains : Sequence[float] | NDArray[np.floating] | None, optional
            Gain values k >= 0.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        go.Figure
            Interactive Plotly Figure.
        """
        from ctrlpy.plotting_plotly import iplot_root_locus

        return iplot_root_locus(self, gains=gains, **kwargs)


# Convenient alias
LTI = LinearTimeInvariant

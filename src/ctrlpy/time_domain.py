"""Time-domain simulation functions for Linear Time-Invariant (LTI) systems."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from ctrlpy.models.base import LinearTimeInvariant
from ctrlpy.models.state_space import StateSpace
from ctrlpy.models.transfer_function import TransferFunction
from ctrlpy.simulation_results import TimeResponseData


def _generate_time_vector(
    sys: LinearTimeInvariant,
    T: Sequence[float] | NDArray[np.floating] | float | None = None,
    n_points: int = 1000,
) -> NDArray[np.float64]:
    """Generate or validate the simulation time vector based on system dynamics.

    Parameters
    ----------
    sys : LinearTimeInvariant
        The LTI system being simulated.
    T : Sequence[float] | NDArray[np.floating] | float | None, optional
        Explicit time vector, simulation duration, or None for auto-computation.
    n_points : int, optional
        Default number of time points to generate when duration or auto is used.

    Returns
    -------
    NDArray[np.float64]
        1D strictly monotonically increasing array of time points starting at 0.

    Raises
    ------
    ValueError
        If time vector is invalid (too short, negative start, or non-monotonic).
    """
    if isinstance(T, (int, float, np.number)):
        t_stop = float(T)
        if t_stop <= 0.0:
            raise ValueError(f"Simulation duration T must be positive, got {t_stop}.")
        return np.linspace(0.0, t_stop, n_points, dtype=np.float64)

    if T is not None:
        t_arr = np.asarray(T, dtype=np.float64).ravel()
        if t_arr.size < 2:
            raise ValueError("Time vector T must contain at least 2 points.")
        if t_arr[0] < 0.0:
            raise ValueError(f"Time vector T must start at t >= 0, got {t_arr[0]}.")
        if np.any(np.diff(t_arr) <= 0.0):
            raise ValueError("Time vector T must be strictly monotonically increasing.")
        return t_arr

    # Automatic time vector computation based on dominant poles
    poles = sys.poles()
    stable_re = [abs(float(p.real)) for p in poles if p.real < -1e-6]
    unstable_re = [float(p.real) for p in poles if p.real > 1e-6]
    imag_freqs = [abs(float(p.imag)) for p in poles if abs(p.real) <= 1e-6 and abs(p.imag) > 1e-6]

    if stable_re:
        tau_slow = 1.0 / min(stable_re)
        t_final = 7.0 * tau_slow
        omega_max = max((abs(complex(p)) for p in poles), default=1.0)
        n_pts = max(1000, min(10000, int(t_final * max(omega_max, 1.0) * 20)))
    elif unstable_re:
        t_final = max(4.0 / max(unstable_re), 1.0)
        n_pts = 1000
    elif imag_freqs:
        w0 = min(imag_freqs)
        t_final = 5.0 * (2.0 * np.pi / w0)
        w_max = max(imag_freqs)
        n_pts = max(1000, int(t_final * w_max * 20))
    else:
        t_final = 10.0
        n_pts = 1000

    return np.linspace(0.0, t_final, n_pts, dtype=np.float64)


def step_response(
    sys: LinearTimeInvariant,
    T: Sequence[float] | NDArray[np.floating] | float | None = None,
    X0: Sequence[float] | NDArray[np.floating] | None = None,
    input_index: int = 0,
) -> TimeResponseData:
    """Compute the step response of a continuous-time LTI system.

    Parameters
    ----------
    sys : LinearTimeInvariant
        The LTI system (TransferFunction or StateSpace).
    T : Sequence[float] | NDArray[np.floating] | float | None, optional
        Time vector or duration. If None, automatically determined from dominant poles.
    X0 : Sequence[float] | NDArray[np.floating] | None, optional
        Initial state vector (only valid for StateSpace systems).
    input_index : int, optional
        Input channel index for multi-input systems, default is 0.

    Returns
    -------
    TimeResponseData
        Object containing time vector `t`, output trajectory `y`, and state trajectory `x`.

    Raises
    ------
    ValueError
        If initial condition X0 is passed to a TransferFunction model or if
        X0 dimensions do not match state-space model.
    IndexError
        If input_index is out of bounds.
    TypeError
        If sys is not a LinearTimeInvariant instance.
    """
    if not isinstance(sys, LinearTimeInvariant):
        raise TypeError(f"Expected LinearTimeInvariant instance, got {type(sys).__name__}.")

    t_arr = _generate_time_vector(sys, T)

    if isinstance(sys, TransferFunction):
        if X0 is not None:
            raise ValueError(
                "Initial state X0 cannot be specified for TransferFunction models. "
                "Convert to StateSpace first."
            )
        t_out, y_out = signal.step((sys.num, sys.den), T=t_arr)
        return TimeResponseData(t=t_out, y=y_out, x=None, sys=sys, poles=sys.poles())

    if isinstance(sys, StateSpace):
        if not (0 <= input_index < sys.inputs):
            raise IndexError(f"input_index {input_index} out of range for {sys.inputs} inputs.")
        x0_arr = (
            np.zeros(sys.n_states, dtype=np.float64)
            if X0 is None
            else np.asarray(X0, dtype=np.float64).ravel()
        )
        if x0_arr.size != sys.n_states:
            raise ValueError(
                f"Initial state X0 must have length {sys.n_states}, got {x0_arr.size}."
            )

        if sys.inputs == 1:
            u: NDArray[np.float64] = np.ones(len(t_arr), dtype=np.float64)
        else:
            u = np.zeros((len(t_arr), sys.inputs), dtype=np.float64)
            u[:, input_index] = 1.0

        t_out, y_out, x_out = signal.lsim(
            (sys.A, sys.B, sys.C, sys.D),
            u,
            t_arr,
            X0=x0_arr,
        )
        if sys.outputs == 1 and y_out.ndim == 2:
            y_out = y_out.ravel()
        if x_out.ndim == 1:
            x_out = x_out.reshape(-1, sys.n_states)

        return TimeResponseData(t=t_out, y=y_out, x=x_out, sys=sys, poles=sys.poles())

    raise TypeError(f"Unsupported system type: {type(sys).__name__}")


def impulse_response(
    sys: LinearTimeInvariant,
    T: Sequence[float] | NDArray[np.floating] | float | None = None,
    X0: Sequence[float] | NDArray[np.floating] | None = None,
    input_index: int = 0,
) -> TimeResponseData:
    """Compute the impulse response of a continuous-time LTI system.

    Parameters
    ----------
    sys : LinearTimeInvariant
        The LTI system (TransferFunction or StateSpace).
    T : Sequence[float] | NDArray[np.floating] | float | None, optional
        Time vector or duration. If None, automatically determined from dominant poles.
    X0 : Sequence[float] | NDArray[np.floating] | None, optional
        Initial state vector before the impulse (only valid for StateSpace systems).
    input_index : int, optional
        Input channel index for multi-input systems, default is 0.

    Returns
    -------
    TimeResponseData
        Object containing time vector `t`, output trajectory `y`, and state trajectory `x`.

    Raises
    ------
    ValueError
        If initial condition X0 is passed to a TransferFunction model or if
        X0 dimensions do not match state-space model.
    IndexError
        If input_index is out of bounds.
    TypeError
        If sys is not a LinearTimeInvariant instance.
    """
    if not isinstance(sys, LinearTimeInvariant):
        raise TypeError(f"Expected LinearTimeInvariant instance, got {type(sys).__name__}.")

    t_arr = _generate_time_vector(sys, T)

    if isinstance(sys, TransferFunction):
        if X0 is not None:
            raise ValueError(
                "Initial state X0 cannot be specified for TransferFunction models. "
                "Convert to StateSpace first."
            )
        t_out, y_out = signal.impulse((sys.num, sys.den), T=t_arr)
        return TimeResponseData(t=t_out, y=y_out, x=None, sys=sys, poles=sys.poles())

    if isinstance(sys, StateSpace):
        if not (0 <= input_index < sys.inputs):
            raise IndexError(f"input_index {input_index} out of range for {sys.inputs} inputs.")
        x0_arr = (
            np.zeros(sys.n_states, dtype=np.float64)
            if X0 is None
            else np.asarray(X0, dtype=np.float64).ravel()
        )
        if x0_arr.size != sys.n_states:
            raise ValueError(
                f"Initial state X0 must have length {sys.n_states}, got {x0_arr.size}."
            )

        b_col = sys.B[:, input_index]
        x_init = b_col + x0_arr
        b_zero = np.zeros_like(sys.B)

        t_out, y_out, x_out = signal.lsim(
            (sys.A, b_zero, sys.C, sys.D),
            0.0,
            t_arr,
            X0=x_init,
            interp=False,
        )
        if sys.outputs == 1 and y_out.ndim == 2:
            y_out = y_out.ravel()
        if x_out.ndim == 1:
            x_out = x_out.reshape(-1, sys.n_states)

        return TimeResponseData(t=t_out, y=y_out, x=x_out, sys=sys, poles=sys.poles())

    raise TypeError(f"Unsupported system type: {type(sys).__name__}")


def forced_response(
    sys: LinearTimeInvariant,
    T: Sequence[float] | NDArray[np.floating],
    U: Sequence[float] | NDArray[np.floating] | float | np.number[Any],
    X0: Sequence[float] | NDArray[np.floating] | None = None,
) -> TimeResponseData:
    """Compute the simulation response of an LTI system to arbitrary inputs U(t).

    Parameters
    ----------
    sys : LinearTimeInvariant
        The LTI system (TransferFunction or StateSpace).
    T : Sequence[float] | NDArray[np.floating]
        Simulation time points (1D array).
    U : Sequence[float] | NDArray[np.floating] | float | int
        Input signal over time. For SISO systems, a 1D array of length `len(T)` or a scalar.
        For MIMO systems, a 2D array of shape `(len(T), inputs)`.
    X0 : Sequence[float] | NDArray[np.floating] | None, optional
        Initial state vector (only valid for StateSpace systems).

    Returns
    -------
    TimeResponseData
        Object containing time vector `t`, output trajectory `y`, and state trajectory `x`.

    Raises
    ------
    ValueError
        If U or X0 dimensions are incompatible with T or sys.
    TypeError
        If sys is not a LinearTimeInvariant instance.
    """
    if not isinstance(sys, LinearTimeInvariant):
        raise TypeError(f"Expected LinearTimeInvariant instance, got {type(sys).__name__}.")

    t_arr = _generate_time_vector(sys, T)
    n_t = len(t_arr)

    # Process and validate input U
    if isinstance(U, (int, float, np.number)):
        if sys.inputs == 1:
            u_arr: NDArray[np.float64] = np.full(n_t, float(U), dtype=np.float64)
        else:
            u_arr = np.full((n_t, sys.inputs), float(U), dtype=np.float64)
    else:
        u_raw = np.asarray(U, dtype=np.float64)
        if sys.inputs == 1:
            u_arr = u_raw.ravel()
            if u_arr.size != n_t:
                raise ValueError(
                    f"Input U length {u_arr.size} must match time vector T length {n_t}."
                )
        else:
            if u_raw.ndim == 1:
                if u_raw.size == sys.inputs:
                    u_arr = np.tile(u_raw, (n_t, 1))
                else:
                    raise ValueError(
                        f"1D input U of size {u_raw.size} does not match {sys.inputs} inputs."
                    )
            elif u_raw.ndim == 2:
                if u_raw.shape == (n_t, sys.inputs):
                    u_arr = u_raw
                elif u_raw.shape == (sys.inputs, n_t) and n_t != sys.inputs:
                    u_arr = u_raw.T
                else:
                    raise ValueError(
                        f"Input U shape {u_raw.shape} is incompatible with T of "
                        f"length {n_t} and {sys.inputs} inputs."
                    )
            else:
                raise ValueError(f"Input U array must be 1D or 2D, got {u_raw.ndim}D.")

    if isinstance(sys, TransferFunction):
        if X0 is not None:
            raise ValueError(
                "Initial state X0 cannot be specified for TransferFunction models. "
                "Convert to StateSpace first."
            )
        t_out, y_out, _ = signal.lsim((sys.num, sys.den), u_arr, t_arr)
        return TimeResponseData(t=t_out, y=y_out, x=None, sys=sys, poles=sys.poles())

    if isinstance(sys, StateSpace):
        x0_arr = (
            np.zeros(sys.n_states, dtype=np.float64)
            if X0 is None
            else np.asarray(X0, dtype=np.float64).ravel()
        )
        if x0_arr.size != sys.n_states:
            raise ValueError(
                f"Initial state X0 must have length {sys.n_states}, got {x0_arr.size}."
            )

        t_out, y_out, x_out = signal.lsim(
            (sys.A, sys.B, sys.C, sys.D),
            u_arr,
            t_arr,
            X0=x0_arr,
        )
        if sys.outputs == 1 and y_out.ndim == 2:
            y_out = y_out.ravel()
        if x_out.ndim == 1:
            x_out = x_out.reshape(-1, sys.n_states)

        return TimeResponseData(t=t_out, y=y_out, x=x_out, sys=sys, poles=sys.poles())

    raise TypeError(f"Unsupported system type: {type(sys).__name__}")

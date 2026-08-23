"""System interconnection and block diagram algebra functions."""

from __future__ import annotations

from typing import Any

import numpy as np

from ctrlpy.models.base import LinearTimeInvariant
from ctrlpy.models.transfer_function import TransferFunction


def series(
    *systems: LinearTimeInvariant | float | np.number[Any],
) -> LinearTimeInvariant:
    """Connect two or more systems in series (cascade).

    The output of each system is connected to the input of the next system:
    u -> systems[0] -> systems[1] -> ... -> systems[n-1] -> y

    Parameters
    ----------
    *systems : LinearTimeInvariant | float | int | np.number
        Two or more systems (or scalar gains) to cascade in order.

    Returns
    -------
    LinearTimeInvariant
        The resulting series-connected system.

    Raises
    ------
    ValueError
        If no systems are provided.
    """
    if not systems:
        raise ValueError("At least one system must be provided for series connection.")

    first = systems[0]
    res: LinearTimeInvariant = (
        first if isinstance(first, LinearTimeInvariant) else TransferFunction(float(first), 1.0)
    )

    for sys in systems[1:]:
        sys_obj = sys if isinstance(sys, LinearTimeInvariant) else TransferFunction(float(sys), 1.0)
        # Cascade order: u -> res -> sys_obj -> y
        res = sys_obj * res

    return res


def parallel(
    *systems: LinearTimeInvariant | float | np.number[Any],
) -> LinearTimeInvariant:
    """Connect two or more systems in parallel.

    Inputs are connected together and outputs are summed:
    y = systems[0](u) + systems[1](u) + ... + systems[n-1](u)

    Parameters
    ----------
    *systems : LinearTimeInvariant | float | int | np.number
        Two or more systems (or scalar gains) to connect in parallel.

    Returns
    -------
    LinearTimeInvariant
        The resulting parallel-connected system.

    Raises
    ------
    ValueError
        If no systems are provided.
    """
    if not systems:
        raise ValueError("At least one system must be provided for parallel connection.")

    first = systems[0]
    res: LinearTimeInvariant = (
        first if isinstance(first, LinearTimeInvariant) else TransferFunction(float(first), 1.0)
    )

    for sys in systems[1:]:
        sys_obj = sys if isinstance(sys, LinearTimeInvariant) else TransferFunction(float(sys), 1.0)
        res = res + sys_obj

    return res


def feedback(
    sys1: LinearTimeInvariant | float | np.number[Any],
    sys2: LinearTimeInvariant | float | np.number[Any] = 1,
    sign: float = -1,
) -> LinearTimeInvariant:
    """Connect two systems in a closed-loop feedback interconnection.

    The closed-loop system relates reference r to output y:
    e = r + sign * sys2(y)
    y = sys1(e)

    Closed-loop transfer function: T = sys1 / (1 - sign * sys1 * sys2)

    Parameters
    ----------
    sys1 : LinearTimeInvariant | float | int | np.number
        Forward path system G.
    sys2 : LinearTimeInvariant | float | int | np.number, optional
        Feedback path system H, defaults to 1 (unity feedback).
    sign : int | float, optional
        Feedback sign. Defaults to -1 for standard negative feedback.

    Returns
    -------
    LinearTimeInvariant
        Closed-loop system.
    """
    sys1_obj: LinearTimeInvariant = (
        sys1 if isinstance(sys1, LinearTimeInvariant) else TransferFunction(float(sys1), 1.0)
    )
    if hasattr(sys1_obj, "feedback"):
        result: LinearTimeInvariant = sys1_obj.feedback(sys2, sign=sign)
        return result
    raise TypeError(f"System {type(sys1_obj).__name__} does not support feedback interconnection.")

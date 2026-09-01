"""System interconnection and block diagram algebra functions."""

from __future__ import annotations

from typing import Any

import numpy as np

from ctrlpy.models.base import LinearTimeInvariant
from ctrlpy.models.transfer_function import TransferFunction


def series(
    *systems: LinearTimeInvariant | float | np.number[Any],
) -> LinearTimeInvariant:
    r"""Connect two or more systems in series (cascade).

    The output of each system is connected to the input of the next system:

    .. math::

        u \longrightarrow G_1(s) \longrightarrow G_2(s) \longrightarrow \cdots \longrightarrow G_n(s) \longrightarrow y

    The equivalent transfer function is the product of individual blocks:

    .. math::

        G_{\text{total}}(s) = G_n(s) \cdots G_2(s) G_1(s)

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
    r"""Connect two or more systems in parallel.

    Inputs are connected together and outputs are summed:

    .. math::

        y(t) = \sum_{i=1}^n y_i(t)

    The equivalent transfer function is the sum of individual blocks:

    .. math::

        G_{\text{total}}(s) = G_1(s) + G_2(s) + \cdots + G_n(s)

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
    r"""Connect two systems in a closed-loop feedback interconnection.

    The closed-loop system relates reference $r(t)$ to output $y(t)$:

    .. math::

        e(t) = r(t) + \text{sign} \cdot (H y)(t), \quad y(t) = (G e)(t)

    The closed-loop transfer function is given by:

    .. math::

        T(s) = \frac{G(s)}{1 - \text{sign} \cdot G(s) H(s)}

    For standard negative feedback ($\text{sign} = -1$):

    .. math::

        T(s) = \frac{G(s)}{1 + G(s) H(s)}

    Parameters
    ----------
    sys1 : LinearTimeInvariant | float | int | np.number
        Forward path system $G(s)$.
    sys2 : LinearTimeInvariant | float | int | np.number, optional
        Feedback path system $H(s)$, defaults to 1 (unity feedback).
    sign : int | float, optional
        Feedback sign. Defaults to -1 for standard negative feedback.

    Returns
    -------
    LinearTimeInvariant
        Closed-loop system $T(s)$.
    """
    sys1_obj: LinearTimeInvariant = (
        sys1 if isinstance(sys1, LinearTimeInvariant) else TransferFunction(float(sys1), 1.0)
    )
    if hasattr(sys1_obj, "feedback"):
        result: LinearTimeInvariant = sys1_obj.feedback(sys2, sign=sign)
        return result
    raise TypeError(f"System {type(sys1_obj).__name__} does not support feedback interconnection.")

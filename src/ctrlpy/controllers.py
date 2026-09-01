"""PID controller design and tuning module for Linear Time-Invariant systems."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ctrlpy.exceptions import UnstableSystemError
from ctrlpy.models.base import LinearTimeInvariant
from ctrlpy.models.transfer_function import TransferFunction
from ctrlpy.time_domain import step_response


def PID(
    Kp: float = 1.0,
    Ki: float = 0.0,
    Kd: float = 0.0,
    Tf: float = 0.0,
    N: float | None = None,
) -> TransferFunction:
    """Construct a continuous-time PID controller as a TransferFunction.

    The controller transfer function in standard form with derivative filter is:

    .. math::

        C(s) = K_p + \\frac{K_i}{s} + \\frac{K_d s}{T_f s + 1}

    where :math:`T_f` is the derivative filter time constant. Alternatively,
    if the filter coefficient :math:`N` is specified, the filter time constant is
    computed as :math:`T_f = \\frac{K_d}{N K_p}` (for :math:`K_p > 0`).

    Parameters
    ----------
    Kp : float, optional
        Proportional gain, default is 1.0.
    Ki : float, optional
        Integral gain, default is 0.0.
    Kd : float, optional
        Derivative gain, default is 0.0.
    Tf : float, optional
        Derivative filter time constant (:math:`T_f \\ge 0`), default is 0.0 (no filter).
    N : float | None, optional
        Derivative filter coefficient (:math:`N > 0`). If specified, overrides `Tf`
        with :math:`T_f = K_d / (N K_p)` when :math:`K_d > 0` and :math:`K_p > 0`.

    Returns
    -------
    TransferFunction
        Continuous-time transfer function representing the PID controller.

    Raises
    ------
    ValueError
        If Tf is negative or N is non-positive.
    """
    kp = float(Kp)
    ki = float(Ki)
    kd = float(Kd)

    if N is not None:
        if N <= 0:
            raise ValueError(f"Filter coefficient N must be positive, got {N}.")
        if kd == 0.0:
            tf_val = 0.0
        elif kp != 0.0:
            tf_val = kd / (float(N) * abs(kp))
        else:
            tf_val = kd / float(N)
    else:
        tf_val = float(Tf)

    if tf_val < 0.0:
        raise ValueError(f"Derivative filter time constant Tf must be non-negative, got {tf_val}.")

    # Pure Proportional Controller
    if ki == 0.0 and kd == 0.0:
        return TransferFunction([kp], [1.0])

    # Proportional-Derivative (PD) Controller
    if ki == 0.0:
        if tf_val > 0.0:
            # C(s) = Kp + (Kd * s) / (Tf * s + 1) = ((Kp * Tf + Kd) * s + Kp) / (Tf * s + 1)
            num = [kp * tf_val + kd, kp]
            den = [tf_val, 1.0]
        else:
            # C(s) = Kd * s + Kp
            num = [kd, kp]
            den = [1.0]
        return TransferFunction(num, den)

    # Proportional-Integral (PI) Controller
    if kd == 0.0:
        # C(s) = Kp + Ki / s = (Kp * s + Ki) / s
        num = [kp, ki]
        den = [1.0, 0.0]
        return TransferFunction(num, den)

    # Full PID Controller
    if tf_val > 0.0:
        # C(s) = Kp + Ki/s + (Kd*s)/(Tf*s + 1)
        #      = ((Kp*Tf + Kd)*s^2 + (Kp + Ki*Tf)*s + Ki) / (Tf*s^2 + s)
        num = [kp * tf_val + kd, kp + ki * tf_val, ki]
        den = [tf_val, 1.0, 0.0]
    else:
        # C(s) = (Kd*s^2 + Kp*s + Ki) / s
        num = [kd, kp, ki]
        den = [1.0, 0.0]

    return TransferFunction(num, den)


def pid(
    Kp: float = 1.0,
    Ki: float = 0.0,
    Kd: float = 0.0,
    Tf: float = 0.0,
    N: float | None = None,
) -> TransferFunction:
    r"""Construct a continuous-time PID controller as a TransferFunction.

    .. math::

        C(s) = K_p + \frac{K_i}{s} + \frac{K_d s}{T_f s + 1}

    Parameters
    ----------
    Kp : float, optional
        Proportional gain $K_p$, default is 1.0.
    Ki : float, optional
        Integral gain $K_i$, default is 0.0.
    Kd : float, optional
        Derivative gain $K_d$, default is 0.0.
    Tf : float, optional
        Derivative filter time constant $T_f \ge 0$, default is 0.0.
    N : float | None, optional
        Derivative filter coefficient $N > 0$, default is None.

    Returns
    -------
    TransferFunction
        Continuous-time transfer function representing the PID controller.
    """
    return PID(Kp=Kp, Ki=Ki, Kd=Kd, Tf=Tf, N=N)


def pi(Kp: float = 1.0, Ki: float = 0.0) -> TransferFunction:
    r"""Construct a continuous-time Proportional-Integral (PI) controller.

    .. math::

        C(s) = K_p + \frac{K_i}{s} = \frac{K_p s + K_i}{s}

    Parameters
    ----------
    Kp : float, optional
        Proportional gain $K_p$, default is 1.0.
    Ki : float, optional
        Integral gain $K_i$, default is 0.0.

    Returns
    -------
    TransferFunction
        Continuous-time PI controller transfer function.
    """
    return PID(Kp=Kp, Ki=Ki, Kd=0.0, Tf=0.0, N=None)


def pd(
    Kp: float = 1.0,
    Kd: float = 0.0,
    Tf: float = 0.0,
    N: float | None = None,
) -> TransferFunction:
    r"""Construct a continuous-time Proportional-Derivative (PD) controller.

    .. math::

        C(s) = K_p + \frac{K_d s}{T_f s + 1}

    Parameters
    ----------
    Kp : float, optional
        Proportional gain $K_p$, default is 1.0.
    Kd : float, optional
        Derivative gain $K_d$, default is 0.0.
    Tf : float, optional
        Derivative filter time constant $T_f \ge 0$, default is 0.0.
    N : float | None, optional
        Derivative filter coefficient $N > 0$, default is None.

    Returns
    -------
    TransferFunction
        Continuous-time PD controller transfer function.
    """
    return PID(Kp=Kp, Ki=0.0, Kd=Kd, Tf=Tf, N=N)


# Aliases
pi_controller = pi
pd_controller = pd
pid_parallel = pid


def tune_ziegler_nichols(
    sys: LinearTimeInvariant,
    method: Literal["step"] = "step",
    controller_type: Literal["pid", "pi", "p"] = "pid",
) -> TransferFunction:
    """Tune a PID controller using the Ziegler-Nichols reaction curve (step response) heuristic.

    Determines apparent dead time :math:`L` and reaction rate (maximum slope) :math:`R`
    from the open-loop step response of a stable plant.

    Tuning rules:
    - **P**: :math:`K_p = \\frac{1}{R L}`
    - **PI**: :math:`K_p = \\frac{0.9}{R L}`, :math:`K_i = \\frac{K_p}{3.33 L}`
    - **PID**: :math:`K_p = \\frac{1.2}{R L}`, :math:`K_i = \\frac{K_p}{2 L}`, :math:`K_d = 0.5 L K_p`

    Parameters
    ----------
    sys : LinearTimeInvariant
        Open-loop SISO LTI system. Must be stable.
    method : {"step"}, optional
        Tuning method, default is "step" (open-loop reaction curve method).
    controller_type : {"pid", "pi", "p"}, optional
        Controller structure to generate, default is "pid".

    Returns
    -------
    TransferFunction
        Tuned controller transfer function.

    Raises
    ------
    ValueError
        If method or controller_type is unsupported, or if the system is not SISO.
    UnstableSystemError
        If the open-loop plant is unstable.
    """
    if method != "step":
        raise ValueError(f"Unsupported tuning method '{method}'. Supported methods: 'step'.")

    ctype = controller_type.lower()
    if ctype not in ("pid", "pi", "p"):
        raise ValueError(
            f"Unsupported controller_type '{controller_type}'. Supported types: 'pid', 'pi', 'p'."
        )

    if not sys.is_siso:
        raise ValueError(
            f"Ziegler-Nichols tuning requires a SISO system, got {sys.inputs} inputs and {sys.outputs} outputs."
        )

    # Check stability of open-loop system
    poles = sys.poles()
    if len(poles) > 0 and np.any(np.real(poles) >= -1e-7):
        raise UnstableSystemError(
            "Ziegler-Nichols step response tuning requires an open-loop strictly stable plant (all Re(poles) < 0)."
        )

    # Simulate open-loop step response with adequate time resolution
    res = step_response(sys)
    t = res.t
    y = res.y if res.y.ndim == 1 else res.y[:, 0]

    y0 = float(y[0])
    yss = float(y[-1])
    dy = yss - y0

    if abs(dy) < 1e-12:
        raise ValueError(
            "Plant step response output change is too small to determine reaction curve parameters."
        )

    # Find inflection point (maximum slope magnitude)
    dydt = np.gradient(y, t)
    if dy > 0:
        idx_inflection = int(np.argmax(dydt))
    else:
        idx_inflection = int(np.argmin(dydt))

    r_slope = float(abs(dydt[idx_inflection]))
    t_infl = float(t[idx_inflection])
    y_infl = float(y[idx_inflection])

    if r_slope < 1e-12:
        raise ValueError(
            "Maximum slope of step response reaction curve is zero; cannot tune Ziegler-Nichols parameters."
        )

    # Tangent line at inflection point: y(t) = y_infl + sign(dy)*r_slope * (t - t_infl)
    # Intersects baseline y0 at t = L:
    # y0 = y_infl + sign(dy)*r_slope * (L - t_infl)
    # L = t_infl - |y_infl - y0| / r_slope
    l_delay = t_infl - abs(y_infl - y0) / r_slope
    l_delay = max(l_delay, 1e-5)  # Safeguard against zero or slightly negative delay

    # Apply Ziegler-Nichols formulas
    rl = r_slope * l_delay
    if ctype == "p":
        kp = 1.0 / rl
        ki = 0.0
        kd = 0.0
    elif ctype == "pi":
        kp = 0.9 / rl
        ti = 3.3333333333333335 * l_delay
        ki = kp / ti
        kd = 0.0
    else:  # pid
        kp = 1.2 / rl
        ti = 2.0 * l_delay
        td = 0.5 * l_delay
        ki = kp / ti
        kd = kp * td

    return PID(Kp=kp, Ki=ki, Kd=kd)

"""Steady-state error analysis and static error constants."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ctrlpy.models.transfer_function import TransferFunction


@dataclass
class SteadyStateResult:
    """Container for steady-state error analysis and static constants.

    Attributes
    ----------
    system_type : int
        Number of open-loop integrators (poles at s = 0).
    kp : float
        Position error constant Kp = lim(s->0) G(s).
    kv : float
        Velocity error constant Kv = lim(s->0) s*G(s).
    ka : float
        Acceleration error constant Ka = lim(s->0) s^2*G(s).
    ess_step : float
        Steady-state tracking error for unit step input: 1 / (1 + Kp).
    ess_ramp : float
        Steady-state tracking error for unit ramp input: 1 / Kv.
    ess_parabolic : float
        Steady-state tracking error for unit parabolic input: 1 / Ka.
    is_closed_loop_stable : bool
        True if all unity feedback closed-loop poles lie strictly in the open LHP.
    closed_loop_poles : list[complex]
        The poles of the closed-loop transfer function T(s) = G(s) / (1 + G(s)).
    steps : list[str]
        Pedagogical step-by-step explanatory notes.
    """

    system_type: int
    kp: float
    kv: float
    ka: float
    ess_step: float
    ess_ramp: float
    ess_parabolic: float
    is_closed_loop_stable: bool
    closed_loop_poles: list[complex] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)

    def _repr_latex_(self) -> str:
        """Render steady-state error metrics and table as formatted LaTeX for Jupyter."""

        def _fmt(val: float) -> str:
            if math.isinf(val):
                return r"\infty"
            if abs(val) < 1e-12:
                return "0"
            return f"{val:.4g}"

        cl_status = (
            r"\text{Stable (Final Value Theorem valid)}"
            if self.is_closed_loop_stable
            else r"\textbf{\textcolor{red}{Unstable (FVT invalid - steady-state errors diverge!)}}"
        )

        latex_str = (
            r"\begin{aligned}"
            rf"\textbf{{System Classification:}} &\quad \text{{Type {self.system_type}}} \\"
            rf"\textbf{{Closed-Loop Stability:}} &\quad {cl_status} \\"
            r"\end{aligned}" + "\n\n"
            r"\begin{array}{|l|c|c|}"
            r"\hline"
            r"\textbf{Test Input } r(t) & \textbf{Static Error Constant} & \textbf{Steady-State Error } e_{ss} \\"
            r"\hline"
            rf"\text{{Unit Step: }} 1(t) & K_p = {_fmt(self.kp)} & e_{{ss,\text{{step}}}} = {_fmt(self.ess_step)} \\"
            rf"\text{{Unit Ramp: }} t \cdot 1(t) & K_v = {_fmt(self.kv)} & e_{{ss,\text{{ramp}}}} = {_fmt(self.ess_ramp)} \\"
            rf"\text{{Unit Parabola: }} \frac{{1}}{{2}}t^2 \cdot 1(t) & K_a = {_fmt(self.ka)} & e_{{ss,\text{{parabolic}}}} = {_fmt(self.ess_parabolic)} \\"
            r"\hline"
            r"\end{array}"
        )
        return latex_str

    def __str__(self) -> str:
        """Format a human-readable ASCII representation of steady-state analysis."""

        def _fmt(val: float) -> str:
            if math.isinf(val):
                return "inf"
            if abs(val) < 1e-12:
                return "0.0"
            return f"{val:.4f}"

        lines = ["=== Steady-State Error Analysis ==="]
        lines.append(f"System Type: Type {self.system_type}")
        lines.append(f"Closed-Loop Asymptotically Stable: {self.is_closed_loop_stable}")
        if not self.is_closed_loop_stable:
            lines.append(
                "  [WARNING: Closed-loop is unstable! Final Value Theorem does not apply.]"
            )
        lines.append("")
        lines.append(
            f"{'Input Type':<20} | {'Static Constant':<18} | {'Steady-State Error ess':<22}"
        )
        lines.append("-" * 66)
        lines.append(
            f"{'Unit Step 1(t)':<20} | {f'Kp = {_fmt(self.kp)}':<18} | {_fmt(self.ess_step):<22}"
        )
        lines.append(
            f"{'Unit Ramp t':<20} | {f'Kv = {_fmt(self.kv)}':<18} | {_fmt(self.ess_ramp):<22}"
        )
        lines.append(
            f"{'Unit Parabola 0.5*t^2':<20} | {f'Ka = {_fmt(self.ka)}':<18} | {_fmt(self.ess_parabolic):<22}"
        )
        lines.append("-" * 66)
        if self.steps:
            lines.append("\nAnalysis Notes:")
            for s in self.steps:
                lines.append(f"  * {s}")
        return "\n".join(lines)


def steady_state_analysis(sys: TransferFunction | Any) -> SteadyStateResult:
    """Perform steady-state tracking error analysis for canonical test inputs.

    Identifies system type (number of poles at s = 0), computes static error constants
    (Kp, Kv, Ka), evaluates steady-state errors (ess for step, ramp, parabola), and
    verifies closed-loop stability to validate the Final Value Theorem.

    Parameters
    ----------
    sys : TransferFunction
        The open-loop LTI transfer function G(s) (assuming unity negative feedback).

    Returns
    -------
    SteadyStateResult
        Dataclass containing the system type, static error constants, steady-state errors,
        and closed-loop stability verification.

    Examples
    --------
    >>> import ctrlpy as cp
    >>> from ctrlpy.pedagogy import steady_state_analysis
    >>> G = cp.tf([10], [1, 2, 0])  # Type 1 system
    >>> res = steady_state_analysis(G)
    >>> res.system_type
    1
    >>> res.kp
    inf
    >>> res.kv
    5.0
    >>> res.ess_step
    0.0
    >>> res.ess_ramp
    0.2
    """
    if not isinstance(sys, TransferFunction):
        if hasattr(sys, "to_tf"):
            sys = sys.to_tf()
        else:
            raise TypeError(
                "Expected a TransferFunction instance or an LTI object convertible to TransferFunction."
            )

    steps: list[str] = []

    # 1. Identify System Type (number of poles at the origin s = 0)
    den = list(sys.den)
    # Count trailing zeros in denominator
    type_n = 0
    for coeff in reversed(den):
        if abs(coeff) < 1e-12:
            type_n += 1
        else:
            break

    steps.append(
        f"Step 1: Open-loop denominator has {type_n} trailing zero(s) at s = 0 -> Classified as Type {type_n} system."
    )

    # 2. Compute Static Error Constants
    # G(s) = N(s) / (s^N * D_bar(s))
    # D_bar(0) is the last non-zero coefficient of den
    d_bar_0 = den[len(den) - 1 - type_n] if (len(den) - 1 - type_n) >= 0 else 1.0
    n_0 = sys.num[-1]

    k_bode = float(n_0 / d_bar_0)

    # Position Error Constant: Kp = lim_{s->0} G(s)
    if type_n == 0:
        kp = k_bode
    else:
        kp = float("inf")

    # Velocity Error Constant: Kv = lim_{s->0} s*G(s)
    if type_n == 0:
        kv = 0.0
    elif type_n == 1:
        kv = k_bode
    else:
        kv = float("inf")

    # Acceleration Error Constant: Ka = lim_{s->0} s^2*G(s)
    if type_n < 2:
        ka = 0.0
    elif type_n == 2:
        ka = k_bode
    else:
        ka = float("inf")

    steps.append(
        f"Step 2: Static error constants: Kp = {kp:g}, Kv = {kv:g}, Ka = {ka:g} (Bode gain K_0 = {k_bode:.4g})."
    )

    # 3. Compute Steady-State Errors ess
    # Step input
    if math.isinf(kp):
        ess_step = 0.0
    else:
        ess_step = 1.0 / (1.0 + kp)

    # Ramp input
    if kv == 0.0:
        ess_ramp = float("inf")
    elif math.isinf(kv):
        ess_ramp = 0.0
    else:
        ess_ramp = 1.0 / kv

    # Parabolic input
    if ka == 0.0:
        ess_parabolic = float("inf")
    elif math.isinf(ka):
        ess_parabolic = 0.0
    else:
        ess_parabolic = 1.0 / ka

    steps.append(
        f"Step 3: Steady-state errors: ess(step) = {ess_step:g}, ess(ramp) = {ess_ramp:g}, ess(parabola) = {ess_parabolic:g}."
    )

    # 4. Verify Closed-Loop Stability
    # Characteristic equation D_cl(s) = D(s) + N(s)
    pad_len = max(len(sys.den), len(sys.num))
    den_padded = np.pad(sys.den, (pad_len - len(sys.den), 0))
    num_padded = np.pad(sys.num, (pad_len - len(sys.num), 0))
    cl_den = den_padded + num_padded

    cl_roots = np.roots(cl_den)
    cl_poles = [complex(r) for r in cl_roots]
    is_cl_stable = all(p.real < -1e-7 for p in cl_poles)

    if is_cl_stable:
        steps.append(
            "Step 4: Unity feedback closed-loop system is asymptotically stable (all closed-loop poles in LHP)."
        )
    else:
        steps.append(
            "Step 4: WARNING: Closed-loop system has unstable or imaginary poles! Final Value Theorem calculations are invalid."
        )

    return SteadyStateResult(
        system_type=type_n,
        kp=kp,
        kv=kv,
        ka=ka,
        ess_step=ess_step,
        ess_ramp=ess_ramp,
        ess_parabolic=ess_parabolic,
        is_closed_loop_stable=is_cl_stable,
        closed_loop_poles=cl_poles,
        steps=steps,
    )

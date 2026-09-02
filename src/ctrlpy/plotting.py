"""Visualization and plotting routines for control systems analysis."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ctrlpy.freq_domain import bode_data, margin, nyquist_data, root_locus_data
from ctrlpy.models.base import LinearTimeInvariant
from ctrlpy.time_domain import impulse_response, step_response


def plot_bode(
    sys: LinearTimeInvariant,
    omega: Sequence[float] | NDArray[np.floating] | None = None,
    margins: bool = True,
    ax: Sequence[Axes] | NDArray[Any] | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot the Bode diagram (magnitude and phase) for a SISO LTI system.

    Parameters
    ----------
    sys : LinearTimeInvariant
        Linear Time-Invariant system.
    omega : Sequence[float] | NDArray[np.floating] | None, optional
        Frequencies in rad/s. If None, auto-generated based on system poles and zeros.
    margins : bool, optional
        Whether to calculate and annotate stability margins (gain and phase margins),
        by default True.
    ax : Sequence[Axes] | NDArray[Any] | None, optional
        Sequence of 2 Matplotlib Axes for magnitude and phase subplots.
        If None, a new Figure and subplots are created.

    Returns
    -------
    tuple[Figure, tuple[Axes, Axes]]
        The Matplotlib Figure and a tuple of (magnitude_axis, phase_axis).

    Raises
    ------
    ValueError
        If provided ax does not contain exactly 2 Axes.
    """
    if ax is None:
        fig, axes_arr = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        ax_mag, ax_phase = axes_arr[0], axes_arr[1]
    else:
        if len(ax) != 2:
            raise ValueError(
                f"plot_bode requires exactly 2 Axes (magnitude, phase), got {len(ax)}."
            )
        ax_mag, ax_phase = ax[0], ax[1]
        fig = cast(Figure, ax_mag.figure)

    bdata = bode_data(sys, omega=omega)

    # Plot Magnitude
    ax_mag.semilogx(bdata.w, bdata.mag_db, "b-", linewidth=1.5, label="Magnitude")
    ax_mag.set_ylabel("Magnitude (dB)")
    ax_mag.set_title("Bode Diagram")
    ax_mag.grid(True, which="both", linestyle="--", alpha=0.6)

    # Plot Phase
    ax_phase.semilogx(bdata.w, bdata.phase, "b-", linewidth=1.5, label="Phase")
    ax_phase.set_ylabel("Phase (deg)")
    ax_phase.set_xlabel("Frequency (rad/s)")
    ax_phase.grid(True, which="both", linestyle="--", alpha=0.6)

    if margins:
        sm = margin(sys)

        # Baseline reference lines
        ax_mag.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_phase.axhline(-180.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        # Gain Crossover Frequency (Wcg) and Phase Margin (PM)
        if not np.isnan(sm.wcg) and not np.isinf(sm.wcg):
            ax_mag.axvline(sm.wcg, color="r", linestyle=":", linewidth=1.2, alpha=0.8)
            ax_phase.axvline(sm.wcg, color="r", linestyle=":", linewidth=1.2, alpha=0.8)

            if not np.isinf(sm.pm_deg):
                phase_at_wcg = sm.pm_deg - 180.0
                ax_phase.plot(
                    [sm.wcg, sm.wcg],
                    [-180.0, phase_at_wcg],
                    "r-",
                    linewidth=2.0,
                )
                ax_phase.plot(sm.wcg, phase_at_wcg, "ro", markersize=4)
                ax_phase.text(
                    sm.wcg * 1.15,
                    (-180.0 + phase_at_wcg) / 2.0,
                    f"PM = {sm.pm_deg:.1f}°\n({sm.wcg:.2g} rad/s)",
                    color="r",
                    fontsize=8,
                    verticalalignment="center",
                )

        # Phase Crossover Frequency (Wcp) and Gain Margin (GM)
        if not np.isnan(sm.wcp) and not np.isinf(sm.wcp):
            ax_mag.axvline(sm.wcp, color="g", linestyle=":", linewidth=1.2, alpha=0.8)
            ax_phase.axvline(sm.wcp, color="g", linestyle=":", linewidth=1.2, alpha=0.8)

            if not np.isinf(sm.gm_db):
                mag_at_wcp = -sm.gm_db
                ax_mag.plot(
                    [sm.wcp, sm.wcp],
                    [0.0, mag_at_wcp],
                    "g-",
                    linewidth=2.0,
                )
                ax_mag.plot(sm.wcp, mag_at_wcp, "go", markersize=4)
                ax_mag.text(
                    sm.wcp * 1.15,
                    mag_at_wcp / 2.0,
                    f"GM = {sm.gm_db:.1f} dB\n({sm.wcp:.2g} rad/s)",
                    color="g",
                    fontsize=8,
                    verticalalignment="center",
                )

    return fig, (ax_mag, ax_phase)


def plot_nyquist(
    sys: LinearTimeInvariant,
    omega: Sequence[float] | NDArray[np.floating] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot the Nyquist diagram for a SISO LTI system.

    Plots the real and imaginary parts of the frequency response G(jw),
    indented circular arc around s=0 if poles at the origin exist,
    marks the critical point (-1, 0j), and overlays the unit circle.

    Parameters
    ----------
    sys : LinearTimeInvariant
        Linear Time-Invariant system.
    omega : Sequence[float] | NDArray[np.floating] | None, optional
        Frequencies in rad/s. If None, auto-generated based on system poles and zeros.
    ax : Axes | None, optional
        Matplotlib Axes to plot into. If None, a new Figure and Axes are created.

    Returns
    -------
    tuple[Figure, Axes]
        The Matplotlib Figure and Axes.
    """
    if ax is None:
        fig, ax_out = plt.subplots(figsize=(7, 6))
    else:
        ax_out = ax
        fig = cast(Figure, ax_out.figure)

    ndata = nyquist_data(sys, omega=omega)

    # Positive frequencies G(jw)
    ax_out.plot(
        ndata.real,
        ndata.imag,
        "b-",
        linewidth=1.5,
        label=r"$G(j\omega), \omega > 0$",
    )
    # Negative frequencies G(-jw)
    ax_out.plot(
        ndata.real,
        -ndata.imag,
        "b--",
        linewidth=1.2,
        alpha=0.7,
        label=r"$G(j\omega), \omega < 0$",
    )

    # Plot indented arc if present
    if ndata.arc_response is not None:
        ax_out.plot(
            np.real(ndata.arc_response),
            np.imag(ndata.arc_response),
            "b:",
            linewidth=1.2,
            alpha=0.7,
            label=r"$\omega \to 0\text{ arc } (s = \epsilon e^{j\theta})$",
        )

    # Critical point (-1, 0)
    ax_out.plot(
        -1.0,
        0.0,
        "r+",
        markersize=12,
        markeredgewidth=2.0,
        label="Critical Point (-1, 0)",
    )

    # Unit circle
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    ax_out.plot(
        np.cos(theta),
        np.sin(theta),
        "k:",
        linewidth=1.0,
        alpha=0.5,
        label="Unit Circle",
    )

    # Axes lines
    ax_out.axhline(0.0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    ax_out.axvline(0.0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)

    # Smart auto-scaling around the critical point (-1, 0j)
    all_resps = [ndata.response, np.conj(ndata.response)]
    if ndata.arc_response is not None:
        all_resps.append(ndata.arc_response)
    comb = np.concatenate(all_resps)
    finite_resp = comb[np.isfinite(comb)]

    if finite_resp.size > 0:
        mags = np.abs(finite_resp)
        min_mag = float(np.min(mags))
        r_thresh = max(3.0, min_mag * 3.0)
        interesting = finite_resp[mags <= r_thresh]
        if interesting.size == 0:
            interesting = finite_resp

        x_pts = np.concatenate([[-1.5, 1.5], np.real(interesting)])
        y_pts = np.concatenate([[-1.5, 1.5], np.imag(interesting)])

        x_min, x_max = float(np.min(x_pts)), float(np.max(x_pts))
        y_min, y_max = float(np.min(y_pts)), float(np.max(y_pts))

        # Clamp excessively huge ranges to reasonable bounding box around critical point
        x_min = max(x_min, -10.0)
        x_max = min(x_max, 10.0)
        y_min = max(y_min, -10.0)
        y_max = min(y_max, 10.0)

        pad_x = 0.1 * (x_max - x_min)
        pad_y = 0.1 * (y_max - y_min)
        ax_out.set_xlim(x_min - pad_x, x_max + pad_x)
        ax_out.set_ylim(y_min - pad_y, y_max + pad_y)

    ax_out.set_xlabel(r"$\mathrm{Re}(G(j\omega))$")
    ax_out.set_ylabel(r"$\mathrm{Im}(G(j\omega))$")
    ax_out.set_title("Nyquist Diagram")
    ax_out.grid(True, linestyle="--", alpha=0.6)
    ax_out.legend(loc="best")

    return fig, ax_out


def plot_root_locus(
    sys: LinearTimeInvariant,
    gains: Sequence[float] | NDArray[np.floating] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot the root locus diagram in the complex s-plane.

    Plots pole trajectories for varying feedback gains k >= 0 with 'x' for
    open-loop poles and 'o' for open-loop zeros.

    Parameters
    ----------
    sys : LinearTimeInvariant
        Open-loop SISO LTI system.
    gains : Sequence[float] | NDArray[np.floating] | None, optional
        Gain values k >= 0. If None, auto-generated.
    ax : Axes | None, optional
        Matplotlib Axes to plot into. If None, a new Figure and Axes are created.

    Returns
    -------
    tuple[Figure, Axes]
        The Matplotlib Figure and Axes.
    """
    if ax is None:
        fig, ax_out = plt.subplots(figsize=(8, 6))
    else:
        ax_out = ax
        fig = cast(Figure, ax_out.figure)

    rldata = root_locus_data(sys, gains=gains)

    # Plot pole trajectory branches
    for branch in range(rldata.roots.shape[1]):
        branch_roots = rldata.roots[:, branch]
        ax_out.plot(branch_roots.real, branch_roots.imag, "b-", linewidth=1.5)

    # Mark open-loop poles
    if rldata.poles.size > 0:
        ax_out.plot(
            rldata.poles.real,
            rldata.poles.imag,
            "rx",
            markersize=9,
            markeredgewidth=2.0,
            label="Open-loop Poles",
        )

    # Mark open-loop zeros
    if rldata.zeros.size > 0:
        ax_out.plot(
            rldata.zeros.real,
            rldata.zeros.imag,
            "go",
            markersize=8,
            markeredgewidth=2.0,
            fillstyle="none",
            label="Open-loop Zeros",
        )

    # Axes lines
    ax_out.axhline(0.0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    ax_out.axvline(0.0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)

    ax_out.set_xlabel(r"$\mathrm{Real}(\sigma)$")
    ax_out.set_ylabel(r"$\mathrm{Imag}(j\omega)$")
    ax_out.set_title("Root Locus")
    ax_out.grid(True, linestyle="--", alpha=0.6)

    if rldata.poles.size > 0 or rldata.zeros.size > 0:
        ax_out.legend(loc="best")

    return fig, ax_out


def plot_step(
    sys: LinearTimeInvariant,
    T: Sequence[float] | NDArray[np.floating] | float | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot the time-domain step response of an LTI system.

    Parameters
    ----------
    sys : LinearTimeInvariant
        Linear Time-Invariant system.
    T : Sequence[float] | NDArray[np.floating] | float | None, optional
        Time vector or simulation duration. If None, auto-generated.
    ax : Axes | None, optional
        Matplotlib Axes to plot into. If None, a new Figure and Axes are created.

    Returns
    -------
    tuple[Figure, Axes]
        The Matplotlib Figure and Axes.
    """
    if ax is None:
        fig, ax_out = plt.subplots(figsize=(8, 5))
    else:
        ax_out = ax
        fig = cast(Figure, ax_out.figure)

    res = step_response(sys, T=T)

    ax_out.plot(res.t, res.y, "b-", linewidth=1.5, label="Step Response")

    if res.y.ndim == 1:
        yss = res.steady_state_value()
        ax_out.axhline(
            yss,
            color="r",
            linestyle="--",
            linewidth=1.2,
            alpha=0.7,
            label=f"Steady State ({yss:.3g})",
        )

    ax_out.set_xlabel("Time (s)")
    ax_out.set_ylabel("Amplitude")
    ax_out.set_title("Step Response")
    ax_out.grid(True, linestyle="--", alpha=0.6)
    ax_out.legend(loc="best")

    return fig, ax_out


def plot_impulse(
    sys: LinearTimeInvariant,
    T: Sequence[float] | NDArray[np.floating] | float | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot the time-domain impulse response of an LTI system.

    Parameters
    ----------
    sys : LinearTimeInvariant
        Linear Time-Invariant system.
    T : Sequence[float] | NDArray[np.floating] | float | None, optional
        Time vector or simulation duration. If None, auto-generated.
    ax : Axes | None, optional
        Matplotlib Axes to plot into. If None, a new Figure and Axes are created.

    Returns
    -------
    tuple[Figure, Axes]
        The Matplotlib Figure and Axes.
    """
    if ax is None:
        fig, ax_out = plt.subplots(figsize=(8, 5))
    else:
        ax_out = ax
        fig = cast(Figure, ax_out.figure)

    res = impulse_response(sys, T=T)

    ax_out.plot(res.t, res.y, "b-", linewidth=1.5, label="Impulse Response")
    ax_out.axhline(0.0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)

    ax_out.set_xlabel("Time (s)")
    ax_out.set_ylabel("Amplitude")
    ax_out.set_title("Impulse Response")
    ax_out.grid(True, linestyle="--", alpha=0.6)
    ax_out.legend(loc="best")

    return fig, ax_out


from ctrlpy.core.discrete import plot_pzmap

__all__ = [
    "plot_bode",
    "plot_impulse",
    "plot_nyquist",
    "plot_pzmap",
    "plot_root_locus",
    "plot_step",
]

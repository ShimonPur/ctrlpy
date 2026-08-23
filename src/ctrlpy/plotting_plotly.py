"""Interactive visualization engine for control systems using Plotly."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots

from ctrlpy.exceptions import UnstableSystemError
from ctrlpy.freq_domain import bode_data, margin, nyquist_data, root_locus_data
from ctrlpy.time_domain import impulse_response, step_response

if TYPE_CHECKING:
    from ctrlpy.models.base import LinearTimeInvariant


def iplot_bode(
    sys: LinearTimeInvariant,
    omega: Sequence[float] | NDArray[np.floating[Any]] | None = None,
    margins: bool = True,
) -> go.Figure:
    """Generate an interactive Bode diagram using Plotly.

    Produces stacked subplots with magnitude in dB and phase in degrees versus
    logarithmic frequency, with stability margin annotations.

    Parameters
    ----------
    sys : LinearTimeInvariant
        Linear Time-Invariant system.
    omega : Sequence[float] | NDArray[np.floating] | None, optional
        Frequency vector in rad/s. If None, automatically determined.
    margins : bool, optional
        Whether to display Gain Margin (GM) and Phase Margin (PM) indicators,
        by default True.

    Returns
    -------
    go.Figure
        Interactive Plotly Figure containing the Bode diagram.
    """
    bdata = bode_data(sys, omega=omega)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Magnitude Response", "Phase Response"),
    )

    # Magnitude trace
    fig.add_trace(
        go.Scatter(
            x=bdata.w,
            y=bdata.mag_db,
            mode="lines",
            name="Magnitude",
            line={"color": "#1f77b4", "width": 2.0},
            hovertemplate="<b>Frequency</b>: %{x:.3g} rad/s<br><b>Magnitude</b>: %{y:.2f} dB<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Phase trace
    fig.add_trace(
        go.Scatter(
            x=bdata.w,
            y=bdata.phase,
            mode="lines",
            name="Phase",
            line={"color": "#1f77b4", "width": 2.0},
            hovertemplate="<b>Frequency</b>: %{x:.3g} rad/s<br><b>Phase</b>: %{y:.2f}°<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Reference lines
    fig.add_hline(
        y=0.0,
        line_dash="dash",
        line_color="gray",
        line_width=1.0,
        row=1,
        col=1,
    )
    fig.add_hline(
        y=-180.0,
        line_dash="dash",
        line_color="gray",
        line_width=1.0,
        row=2,
        col=1,
    )

    if margins:
        sm = margin(sys)

        # Gain Crossover Frequency (Wcg) -> Phase Margin
        if not np.isnan(sm.wcg) and not np.isinf(sm.wcg):
            wcg_val = float(sm.wcg)
            fig.add_vline(
                x=wcg_val,
                line_dash="dot",
                line_color="red",
                line_width=1.5,
                row=1,
                col=1,
            )
            fig.add_vline(
                x=wcg_val,
                line_dash="dot",
                line_color="red",
                line_width=1.5,
                row=2,
                col=1,
            )

            if not np.isinf(sm.pm_deg):
                phase_at_wcg = float(sm.pm_deg - 180.0)
                fig.add_trace(
                    go.Scatter(
                        x=[wcg_val, wcg_val],
                        y=[-180.0, phase_at_wcg],
                        mode="lines+markers",
                        name="Phase Margin",
                        line={"color": "red", "width": 2.5},
                        marker={"size": 6, "color": "red"},
                        hovertemplate=f"<b>PM</b>: {sm.pm_deg:.1f}° at {wcg_val:.2g} rad/s<extra></extra>",
                    ),
                    row=2,
                    col=1,
                )
                fig.add_annotation(
                    x=np.log10(wcg_val),
                    y=(-180.0 + phase_at_wcg) / 2.0,
                    text=f"PM = {sm.pm_deg:.1f}°<br>({wcg_val:.2g} rad/s)",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="red",
                    font={"color": "red", "size": 11},
                    xref="x2",
                    yref="y2",
                )

        # Phase Crossover Frequency (Wcp) -> Gain Margin
        if not np.isnan(sm.wcp) and not np.isinf(sm.wcp):
            wcp_val = float(sm.wcp)
            fig.add_vline(
                x=wcp_val,
                line_dash="dot",
                line_color="green",
                line_width=1.5,
                row=1,
                col=1,
            )
            fig.add_vline(
                x=wcp_val,
                line_dash="dot",
                line_color="green",
                line_width=1.5,
                row=2,
                col=1,
            )

            if not np.isinf(sm.gm_db):
                mag_at_wcp = float(-sm.gm_db)
                fig.add_trace(
                    go.Scatter(
                        x=[wcp_val, wcp_val],
                        y=[0.0, mag_at_wcp],
                        mode="lines+markers",
                        name="Gain Margin",
                        line={"color": "green", "width": 2.5},
                        marker={"size": 6, "color": "green"},
                        hovertemplate=f"<b>GM</b>: {sm.gm_db:.1f} dB at {wcp_val:.2g} rad/s<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )
                fig.add_annotation(
                    x=np.log10(wcp_val),
                    y=mag_at_wcp / 2.0,
                    text=f"GM = {sm.gm_db:.1f} dB<br>({wcp_val:.2g} rad/s)",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="green",
                    font={"color": "green", "size": 11},
                    xref="x",
                    yref="y",
                )

    fig.update_xaxes(type="log", row=1, col=1)
    fig.update_xaxes(type="log", title_text="Frequency (rad/s)", row=2, col=1)
    fig.update_yaxes(title_text="Magnitude (dB)", row=1, col=1)
    fig.update_yaxes(title_text="Phase (deg)", row=2, col=1)

    fig.update_layout(
        title="Bode Diagram",
        template="plotly_white",
        height=650,
        showlegend=False,
        hovermode="x unified",
    )

    return fig


def iplot_nyquist(
    sys: LinearTimeInvariant,
    omega: Sequence[float] | NDArray[np.floating[Any]] | None = None,
) -> go.Figure:
    """Generate an interactive Nyquist diagram using Plotly.

    Plots positive and negative frequency responses, indented origin arcs if present,
    the critical point (-1, 0), and the unit circle.

    Parameters
    ----------
    sys : LinearTimeInvariant
        Linear Time-Invariant system.
    omega : Sequence[float] | NDArray[np.floating] | None, optional
        Frequency vector in rad/s. If None, automatically determined.

    Returns
    -------
    go.Figure
        Interactive Plotly Figure containing the Nyquist diagram.
    """
    ndata = nyquist_data(sys, omega=omega)
    fig = go.Figure()

    # Positive frequency branch: G(jw), w > 0
    fig.add_trace(
        go.Scatter(
            x=ndata.real,
            y=ndata.imag,
            mode="lines",
            name="G(jω), ω > 0",
            line={"color": "#1f77b4", "width": 2.0},
            customdata=ndata.w,
            hovertemplate="<b>ω</b>: %{customdata:.3g} rad/s<br><b>Re</b>: %{x:.3f}<br><b>Im</b>: %{y:.3f}<extra>G(jω)</extra>",
        )
    )

    # Negative frequency branch: G(-jw), w < 0
    fig.add_trace(
        go.Scatter(
            x=ndata.real,
            y=-ndata.imag,
            mode="lines",
            name="G(jω), ω < 0",
            line={"color": "#1f77b4", "width": 1.5, "dash": "dash"},
            customdata=ndata.w,
            hovertemplate="<b>ω</b>: -%{customdata:.3g} rad/s<br><b>Re</b>: %{x:.3f}<br><b>Im</b>: %{y:.3f}<extra>G(-jω)</extra>",
        )
    )

    # Indented circular arc if present
    if ndata.arc_response is not None:
        fig.add_trace(
            go.Scatter(
                x=np.real(ndata.arc_response),
                y=np.imag(ndata.arc_response),
                mode="lines",
                name="ω → 0 Arc",
                line={"color": "#1f77b4", "width": 1.2, "dash": "dot"},
                hovertemplate="<b>Origin Arc</b><br><b>Re</b>: %{x:.3f}<br><b>Im</b>: %{y:.3f}<extra></extra>",
            )
        )

    # Critical Point (-1, 0)
    fig.add_trace(
        go.Scatter(
            x=[-1.0],
            y=[0.0],
            mode="markers",
            name="Critical Point (-1, 0)",
            marker={
                "symbol": "cross",
                "size": 12,
                "color": "red",
                "line": {"width": 2},
            },
            hovertemplate="<b>Critical Point</b>: (-1, 0)<extra></extra>",
        )
    )

    # Unit circle
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    fig.add_trace(
        go.Scatter(
            x=np.cos(theta),
            y=np.sin(theta),
            mode="lines",
            name="Unit Circle",
            line={"color": "gray", "width": 1.0, "dash": "dot"},
            hoverinfo="skip",
        )
    )

    # Axes reference lines
    fig.add_hline(y=0.0, line_dash="solid", line_color="lightgray", line_width=0.8)
    fig.add_vline(x=0.0, line_dash="solid", line_color="lightgray", line_width=0.8)

    # Compute reasonable axis bounds around critical point (-1, 0)
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

        x_min = max(x_min, -10.0)
        x_max = min(x_max, 10.0)
        y_min = max(y_min, -10.0)
        y_max = min(y_max, 10.0)

        pad_x = 0.1 * (x_max - x_min)
        pad_y = 0.1 * (y_max - y_min)
        fig.update_xaxes(range=[x_min - pad_x, x_max + pad_x])
        fig.update_yaxes(range=[y_min - pad_y, y_max + pad_y])

    fig.update_layout(
        title="Nyquist Diagram",
        xaxis_title="Real Axis (Re)",
        yaxis_title="Imaginary Axis (Im)",
        template="plotly_white",
        height=600,
        showlegend=True,
    )

    return fig


def iplot_root_locus(
    sys: LinearTimeInvariant,
    gains: Sequence[float] | NDArray[np.floating[Any]] | None = None,
) -> go.Figure:
    """Generate an interactive Root Locus diagram in the complex s-plane using Plotly.

    Plots pole trajectories for varying feedback gain k >= 0 with tooltips displaying
    gain, pole location, damping ratio (zeta), and natural frequency (wn).

    Parameters
    ----------
    sys : LinearTimeInvariant
        Open-loop SISO LTI system.
    gains : Sequence[float] | NDArray[np.floating] | None, optional
        Gain values k >= 0. If None, automatically determined.

    Returns
    -------
    go.Figure
        Interactive Plotly Figure containing the Root Locus diagram.
    """
    rldata = root_locus_data(sys, gains=gains)
    fig = go.Figure()

    # Axes reference lines
    fig.add_hline(y=0.0, line_dash="solid", line_color="lightgray", line_width=0.8)
    fig.add_vline(x=0.0, line_dash="solid", line_color="lightgray", line_width=0.8)

    num_branches = rldata.roots.shape[1]
    gains_arr = rldata.gains

    for branch in range(num_branches):
        branch_roots = rldata.roots[:, branch]
        sigma = branch_roots.real
        omega = branch_roots.imag
        wn = np.abs(branch_roots)
        zeta = np.where(wn > 1e-12, -sigma / wn, 0.0)

        customdata = np.column_stack([gains_arr, zeta, wn])

        fig.add_trace(
            go.Scatter(
                x=sigma,
                y=omega,
                mode="lines",
                name=f"Branch {branch + 1}",
                line={"color": "#1f77b4", "width": 2.0},
                customdata=customdata,
                hovertemplate=(
                    "<b>Gain (k)</b>: %{customdata[0]:.3g}<br>"
                    "<b>Pole</b>: %{x:.3f} + %{y:.3f}j<br>"
                    "<b>Damping (ζ)</b>: %{customdata[1]:.3f}<br>"
                    "<b>Nat. Freq (ωn)</b>: %{customdata[2]:.3g} rad/s<extra></extra>"
                ),
            )
        )

    # Open-loop poles
    if rldata.poles.size > 0:
        fig.add_trace(
            go.Scatter(
                x=rldata.poles.real,
                y=rldata.poles.imag,
                mode="markers",
                name="Open-loop Poles",
                marker={
                    "symbol": "x",
                    "size": 10,
                    "color": "red",
                    "line": {"width": 2},
                },
                hovertemplate="<b>Open-loop Pole</b>: %{x:.3f} + %{y:.3f}j<extra></extra>",
            )
        )

    # Open-loop zeros
    if rldata.zeros.size > 0:
        fig.add_trace(
            go.Scatter(
                x=rldata.zeros.real,
                y=rldata.zeros.imag,
                mode="markers",
                name="Open-loop Zeros",
                marker={
                    "symbol": "circle-open",
                    "size": 9,
                    "color": "green",
                    "line": {"width": 2},
                },
                hovertemplate="<b>Open-loop Zero</b>: %{x:.3f} + %{y:.3f}j<extra></extra>",
            )
        )

    fig.update_layout(
        title="Root Locus",
        xaxis_title="Real Axis (σ)",
        yaxis_title="Imaginary Axis (jω)",
        template="plotly_white",
        height=600,
        showlegend=(num_branches > 1 or rldata.poles.size > 0 or rldata.zeros.size > 0),
    )

    return fig


def iplot_step(
    sys: LinearTimeInvariant,
    T: Sequence[float] | NDArray[np.floating[Any]] | float | None = None,
) -> go.Figure:
    """Generate an interactive step response plot using Plotly.

    Displays the step response with interactive indicators for steady-state value,
    rise time, and peak overshoot.

    Parameters
    ----------
    sys : LinearTimeInvariant
        Linear Time-Invariant system.
    T : Sequence[float] | NDArray[np.floating] | float | None, optional
        Time vector or simulation duration. If None, automatically determined.

    Returns
    -------
    go.Figure
        Interactive Plotly Figure containing the step response.
    """
    res = step_response(sys, T=T)
    fig = go.Figure()

    y_1d = res.y if res.y.ndim == 1 else res.y[:, 0]

    # Step response curve
    fig.add_trace(
        go.Scatter(
            x=res.t,
            y=y_1d,
            mode="lines",
            name="Step Response",
            line={"color": "#1f77b4", "width": 2.0},
            hovertemplate="<b>Time</b>: %{x:.3f} s<br><b>Amplitude</b>: %{y:.3f}<extra></extra>",
        )
    )

    # Add performance annotations if system is stable
    try:
        yss = res.steady_state_value()
        fig.add_hline(
            y=yss,
            line_dash="dash",
            line_color="red",
            line_width=1.2,
            annotation_text=f"Steady State ({yss:.3g})",
            annotation_position="bottom right",
            annotation_font_color="red",
        )

        tr = res.rise_time()
        if not np.isnan(tr):
            fig.add_annotation(
                text=f"Rise Time (10-90%): {tr:.2f} s",
                xref="paper",
                yref="paper",
                x=0.02,
                y=0.95,
                showarrow=False,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="gray",
                borderwidth=1,
            )

        os_val = res.overshoot()
        tp_val = res.peak_time()
        if os_val > 0.0:
            y_peak = float(np.max(y_1d))
            fig.add_trace(
                go.Scatter(
                    x=[tp_val],
                    y=[y_peak],
                    mode="markers",
                    name="Peak Overshoot",
                    marker={"symbol": "star", "size": 10, "color": "darkorange"},
                    hovertemplate=f"<b>Peak</b>: {y_peak:.3f}<br><b>Overshoot</b>: {os_val:.1f}%<br><b>Peak Time</b>: {tp_val:.2f} s<extra></extra>",
                )
            )
    except UnstableSystemError:
        pass

    fig.update_layout(
        title="Step Response",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        template="plotly_white",
        height=500,
        showlegend=True,
    )

    return fig


def iplot_impulse(
    sys: LinearTimeInvariant,
    T: Sequence[float] | NDArray[np.floating[Any]] | float | None = None,
) -> go.Figure:
    """Generate an interactive impulse response plot using Plotly.

    Parameters
    ----------
    sys : LinearTimeInvariant
        Linear Time-Invariant system.
    T : Sequence[float] | NDArray[np.floating] | float | None, optional
        Time vector or simulation duration. If None, automatically determined.

    Returns
    -------
    go.Figure
        Interactive Plotly Figure containing the impulse response.
    """
    res = impulse_response(sys, T=T)
    fig = go.Figure()

    y_1d = res.y if res.y.ndim == 1 else res.y[:, 0]

    fig.add_trace(
        go.Scatter(
            x=res.t,
            y=y_1d,
            mode="lines",
            name="Impulse Response",
            line={"color": "#1f77b4", "width": 2.0},
            hovertemplate="<b>Time</b>: %{x:.3f} s<br><b>Amplitude</b>: %{y:.3f}<extra></extra>",
        )
    )

    fig.add_hline(y=0.0, line_dash="solid", line_color="lightgray", line_width=0.8)

    fig.update_layout(
        title="Impulse Response",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        template="plotly_white",
        height=500,
        showlegend=True,
    )

    return fig


# Convenient aliases
plot_step_plotly = iplot_step
plot_impulse_plotly = iplot_impulse
plot_bode_plotly = iplot_bode
plot_nyquist_plotly = iplot_nyquist
plot_root_locus_plotly = iplot_root_locus

__all__ = [
    "iplot_bode",
    "iplot_impulse",
    "iplot_nyquist",
    "iplot_root_locus",
    "iplot_step",
    "plot_bode_plotly",
    "plot_impulse_plotly",
    "plot_nyquist_plotly",
    "plot_root_locus_plotly",
    "plot_step_plotly",
]

"""Frequency-domain analysis and stability margin calculations for LTI systems."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment, root_scalar

from ctrlpy.models.base import LinearTimeInvariant
from ctrlpy.models.state_space import StateSpace
from ctrlpy.models.transfer_function import TransferFunction


@dataclass
class BodeData:
    """Frequency response data for Bode diagram analysis.

    Parameters
    ----------
    w : NDArray[np.float64]
        Frequencies in radians per second (rad/s).
    mag : NDArray[np.float64]
        Linear magnitude |G(jw)|.
    phase : NDArray[np.float64]
        Unwrapped phase in degrees.
    mag_db : NDArray[np.float64]
        Magnitude in decibels: 20 * log10(|G(jw)|).
    phase_rad : NDArray[np.float64]
        Unwrapped phase in radians.
    response : NDArray[np.complex128]
        Complex frequency response values G(jw).
    """

    w: NDArray[np.float64]
    mag: NDArray[np.float64]
    phase: NDArray[np.float64]
    mag_db: NDArray[np.float64]
    phase_rad: NDArray[np.float64]
    response: NDArray[np.complex128]

    def __iter__(self) -> Iterator[NDArray[np.float64]]:
        """Enable unpacking: w, mag, phase = bode_data(sys).

        Yields
        ------
        NDArray[np.float64]
            Frequency array `w`, linear magnitude array `mag`, and phase in degrees `phase`.
        """
        yield self.w
        yield self.mag
        yield self.phase


@dataclass
class NyquistData:
    """Frequency response data for Nyquist diagram analysis.

    Parameters
    ----------
    w : NDArray[np.float64]
        Frequencies in radians per second (rad/s).
    response : NDArray[np.complex128]
        Complex frequency response values G(jw).
    real : NDArray[np.float64]
        Real part Re(G(jw)).
    imag : NDArray[np.float64]
        Imaginary part Im(G(jw)).
    arc_s : NDArray[np.complex128] | None, optional
        Complex frequencies s along the indented arc around s=0.
    arc_response : NDArray[np.complex128] | None, optional
        Complex frequency response G(s) along the indented arc around s=0.
    """

    w: NDArray[np.float64]
    response: NDArray[np.complex128]
    real: NDArray[np.float64]
    imag: NDArray[np.float64]
    arc_s: NDArray[np.complex128] | None = None
    arc_response: NDArray[np.complex128] | None = None

    def __iter__(self) -> Iterator[Any]:
        """Enable unpacking: w, response = nyquist_data(sys).

        Yields
        ------
        NDArray[np.float64] | NDArray[np.complex128]
            Frequency array `w` and complex response array `response`.
        """
        yield self.w
        yield self.response


@dataclass
class RootLocusData:
    """Root locus pole trajectory data across varying gains.

    Parameters
    ----------
    gains : NDArray[np.float64]
        1D array of non-negative feedback gain values k >= 0.
    roots : NDArray[np.complex128]
        2D array of closed-loop pole locations of shape (n_gains, n_poles).
    poles : NDArray[np.complex128]
        Open-loop poles (corresponding to k = 0).
    zeros : NDArray[np.complex128]
        Open-loop zeros.
    """

    gains: NDArray[np.float64]
    roots: NDArray[np.complex128]
    poles: NDArray[np.complex128]
    zeros: NDArray[np.complex128]

    def __iter__(self) -> Iterator[Any]:
        """Enable unpacking: gains, roots = root_locus_data(sys).

        Yields
        ------
        NDArray[np.float64] | NDArray[np.complex128]
            Gains array and roots 2D array.
        """
        yield self.gains
        yield self.roots


@dataclass
class StabilityMargins:
    """Container for classical stability margins and crossover frequencies.

    Parameters
    ----------
    gm_db : float
        Gain margin in decibels (dB). Inf if no phase crossover occurs.
    pm_deg : float
        Phase margin in degrees. Inf if no gain crossover occurs.
    wcg : float
        Gain crossover frequency in rad/s (where |G(jw)| = 1 / 0 dB). NaN if no crossover.
    wcp : float
        Phase crossover frequency in rad/s (where phase crosses -180 deg). NaN if no crossover.
    """

    gm_db: float
    pm_deg: float
    wcg: float
    wcp: float

    @property
    def gm(self) -> float:
        """Gain margin in dB."""
        return self.gm_db

    @property
    def pm(self) -> float:
        """Phase margin in degrees."""
        return self.pm_deg

    @property
    def Wcg(self) -> float:
        """Gain crossover frequency in rad/s."""
        return self.wcg

    @property
    def Wcp(self) -> float:
        """Phase crossover frequency in rad/s."""
        return self.wcp

    def __iter__(self) -> Iterator[float]:
        """Enable unpacking: gm_db, pm_deg, wcg, wcp = margin(sys).

        Yields
        ------
        float
            gm_db, pm_deg, wcg, wcp.
        """
        yield self.gm_db
        yield self.pm_deg
        yield self.wcg
        yield self.wcp


def _generate_omega(
    sys: LinearTimeInvariant,
    omega: Sequence[float] | NDArray[np.floating] | None = None,
    n_points: int = 1000,
) -> NDArray[np.float64]:
    """Generate or validate a logarithmic frequency vector for frequency response analysis.

    Parameters
    ----------
    sys : LinearTimeInvariant
        The LTI system.
    omega : Sequence[float] | NDArray[np.floating] | None, optional
        Explicit frequencies in rad/s, or None for automatic calculation.
    n_points : int, optional
        Number of logarithmic frequency points to generate when omega is None.

    Returns
    -------
    NDArray[np.float64]
        1D strictly positive, monotonically increasing array of frequencies.

    Raises
    ------
    ValueError
        If provided omega is invalid (non-positive, too short, or non-monotonic),
        or if sys is not SISO.
    """
    if not isinstance(sys, LinearTimeInvariant):
        raise TypeError(f"Expected LinearTimeInvariant instance, got {type(sys).__name__}.")
    if not sys.is_siso:
        raise ValueError(
            f"Frequency-domain analysis requires a SISO system, got "
            f"({sys.inputs} inputs, {sys.outputs} outputs)."
        )

    if omega is not None:
        w_arr = np.asarray(omega, dtype=np.float64).ravel()
        if w_arr.size < 2:
            raise ValueError("Frequency vector omega must contain at least 2 points.")
        if np.any(w_arr <= 0.0):
            raise ValueError("Frequency vector omega must be strictly positive (w > 0).")
        if np.any(np.diff(w_arr) <= 0.0):
            raise ValueError("Frequency vector omega must be strictly monotonically increasing.")
        return w_arr

    poles = sys.poles()
    zeros = sys.zeros()

    crit_w = [float(abs(p)) for p in poles if abs(p) > 1e-6] + [
        float(abs(z)) for z in zeros if abs(z) > 1e-6
    ]

    if crit_w:
        w_min = min(crit_w)
        w_max = max(crit_w)
        dec_min = float(np.floor(np.log10(w_min))) - 1.0
        dec_max = float(np.ceil(np.log10(w_max))) + 1.0
        if dec_max - dec_min < 3.0:
            center = (dec_min + dec_max) / 2.0
            dec_min = center - 1.5
            dec_max = center + 1.5
    else:
        dec_min = -2.0
        dec_max = 2.0

    return np.logspace(dec_min, dec_max, num=n_points, dtype=np.float64)


def _evaluate_siso_freqresp(
    sys: LinearTimeInvariant,
    w: NDArray[np.float64],
) -> tuple[TransferFunction, NDArray[np.complex128]]:
    """Evaluate complex frequency response G(jw) for a SISO LTI system.

    Parameters
    ----------
    sys : LinearTimeInvariant
        Linear Time-Invariant SISO system.
    w : NDArray[np.float64]
        Frequencies in rad/s.

    Returns
    -------
    tuple[TransferFunction, NDArray[np.complex128]]
        TransferFunction representation and evaluated complex frequency response array.

    Raises
    ------
    TypeError
        If sys is not a LinearTimeInvariant instance.
    ValueError
        If sys is not SISO.
    """
    if not isinstance(sys, LinearTimeInvariant):
        raise TypeError(f"Expected LinearTimeInvariant instance, got {type(sys).__name__}.")
    if not sys.is_siso:
        raise ValueError(
            f"Frequency-domain analysis requires a SISO system, got "
            f"({sys.inputs} inputs, {sys.outputs} outputs)."
        )

    if isinstance(sys, TransferFunction):
        tf_sys = sys
    elif isinstance(sys, StateSpace):
        tf_sys = sys.to_tf()
    else:
        raise TypeError(f"Unsupported system type: {type(sys).__name__}")

    s = 1j * w
    num_eval = np.polyval(tf_sys.num, s)
    den_eval = np.polyval(tf_sys.den, s)

    # Prevent division by zero on imaginary axis poles via small RHP indentation
    zero_mask = np.isclose(den_eval, 0.0, atol=1e-12)
    if np.any(zero_mask):
        s_indented = 1e-6 + 1j * w[zero_mask]
        num_indented = np.polyval(tf_sys.num, s_indented)
        den_indented = np.polyval(tf_sys.den, s_indented)
        resp = np.empty_like(s, dtype=np.complex128)
        resp[~zero_mask] = num_eval[~zero_mask] / den_eval[~zero_mask]
        resp[zero_mask] = num_indented / den_indented
    else:
        resp = num_eval / den_eval

    return tf_sys, resp


def bode_data(
    sys: LinearTimeInvariant,
    omega: Sequence[float] | NDArray[np.floating] | None = None,
) -> BodeData:
    """Calculate frequency response data for Bode diagram analysis.

    Parameters
    ----------
    sys : LinearTimeInvariant
        Linear Time-Invariant system (TransferFunction or StateSpace).
    omega : Sequence[float] | NDArray[np.floating] | None, optional
        Frequencies in rad/s. If None, auto-generated based on poles and zeros.

    Returns
    -------
    BodeData
        Dataclass containing frequencies w, linear magnitude mag,
        unwrapped phase in degrees, magnitude in dB, phase in radians,
        and complex response.
    """
    w = _generate_omega(sys, omega)
    _, resp = _evaluate_siso_freqresp(sys, w)

    mag = np.abs(resp).astype(np.float64)
    with np.errstate(divide="ignore"):
        mag_db = np.where(mag > 0.0, 20.0 * np.log10(mag), -np.inf)

    phase_rad = np.unwrap(np.angle(resp)).astype(np.float64)
    phase_deg = np.rad2deg(phase_rad)

    return BodeData(
        w=w,
        mag=mag,
        phase=phase_deg,
        mag_db=mag_db,
        phase_rad=phase_rad,
        response=resp,
    )


def nyquist_data(
    sys: LinearTimeInvariant,
    omega: Sequence[float] | NDArray[np.floating] | None = None,
    n_arc_points: int = 100,
) -> NyquistData:
    """Calculate complex frequency response data for Nyquist diagram analysis.

    Parameters
    ----------
    sys : LinearTimeInvariant
        Linear Time-Invariant system (TransferFunction or StateSpace).
    omega : Sequence[float] | NDArray[np.floating] | None, optional
        Frequencies in rad/s. If None, auto-generated based on poles and zeros.
    n_arc_points : int, optional
        Number of points to evaluate along the indented D-contour around s=0
        when an origin pole is present, by default 100.

    Returns
    -------
    NyquistData
        Dataclass containing frequencies w, complex frequency response G(jw),
        real part Re(G(jw)), imaginary part Im(G(jw)), and indented arc data if applicable.
    """
    w = _generate_omega(sys, omega)
    tf_sys, resp = _evaluate_siso_freqresp(sys, w)

    poles = tf_sys.poles()
    has_origin_pole = np.any(np.isclose(poles, 0.0, atol=1e-5))

    arc_s: NDArray[np.complex128] | None = None
    arc_resp: NDArray[np.complex128] | None = None

    if has_origin_pole:
        eps = float(w[0])
        # Indented Nyquist D-contour around s=0: evaluate s = eps * exp(j * theta)
        # for theta in [-pi/2, +pi/2] connecting omega = 0^- (s = -j*eps) to omega = 0^+ (s = +j*eps).
        theta = np.linspace(-np.pi / 2.0, np.pi / 2.0, n_arc_points, dtype=np.float64)
        s_arc: NDArray[np.complex128] = (eps * np.exp(1j * theta)).astype(np.complex128)
        num_arc = np.polyval(tf_sys.num, s_arc)
        den_arc = np.polyval(tf_sys.den, s_arc)
        arc_s = s_arc
        arc_resp = np.asarray(num_arc / den_arc, dtype=np.complex128)

    return NyquistData(
        w=w,
        response=resp,
        real=np.real(resp).astype(np.float64),
        imag=np.imag(resp).astype(np.float64),
        arc_s=arc_s,
        arc_response=arc_resp,
    )


def root_locus_data(
    sys: LinearTimeInvariant,
    gains: Sequence[float] | NDArray[np.floating] | None = None,
) -> RootLocusData:
    """Calculate closed-loop root locus trajectories for varying gains k >= 0.

    Computes the roots of 1 + k * G(s) = 0 for varying feedback gains k >= 0.

    Parameters
    ----------
    sys : LinearTimeInvariant
        Open-loop SISO LTI system.
    gains : Sequence[float] | NDArray[np.floating] | None, optional
        1D array of non-negative gain values k. If None, automatically generated.

    Returns
    -------
    RootLocusData
        Dataclass containing gains, 2D roots array of shape (n_gains, n_poles),
        open-loop poles, and open-loop zeros.

    Raises
    ------
    ValueError
        If gains array is empty or contains negative values.
    """
    if not isinstance(sys, LinearTimeInvariant):
        raise TypeError(f"Expected LinearTimeInvariant instance, got {type(sys).__name__}.")
    if not sys.is_siso:
        raise ValueError(
            f"Root locus requires a SISO system, got ({sys.inputs} inputs, {sys.outputs} outputs)."
        )

    if isinstance(sys, TransferFunction):
        tf_sys = sys
    elif isinstance(sys, StateSpace):
        tf_sys = sys.to_tf()
    else:
        raise TypeError(f"Unsupported system type: {type(sys).__name__}")

    ol_poles = tf_sys.poles()
    ol_zeros = tf_sys.zeros()

    if gains is not None:
        gains_arr = np.asarray(gains, dtype=np.float64).ravel()
        if gains_arr.size == 0:
            raise ValueError("gains array cannot be empty.")
        if np.any(gains_arr < 0.0):
            raise ValueError("All gains in root locus must be non-negative (k >= 0).")
    else:
        gains_arr = np.concatenate(([0.0], np.geomspace(1e-4, 1e4, num=500, dtype=np.float64)))

    num = tf_sys.num
    den = tf_sys.den

    n_poles = len(den) - 1
    if n_poles <= 0:
        roots_arr = np.empty((len(gains_arr), 0), dtype=np.complex128)
        return RootLocusData(
            gains=gains_arr,
            roots=roots_arr,
            poles=ol_poles,
            zeros=ol_zeros,
        )

    deg_d = len(den)
    deg_n = len(num)
    max_deg = max(deg_d, deg_n)

    num_pad = np.pad(num, (max_deg - deg_n, 0), mode="constant")
    den_pad = np.pad(den, (max_deg - deg_d, 0), mode="constant")

    n_gains = len(gains_arr)
    roots_matrix = np.zeros((n_gains, n_poles), dtype=np.complex128)

    p0 = np.roots(den)
    idx0 = np.lexsort((p0.real, p0.imag))
    prev_roots = p0[idx0].astype(np.complex128)
    roots_matrix[0] = prev_roots

    for i in range(1, n_gains):
        k = gains_arr[i]
        poly_k = den_pad + k * num_pad
        curr_roots = np.roots(poly_k).astype(np.complex128)

        if curr_roots.size == n_poles:
            dist_matrix = np.abs(curr_roots[:, np.newaxis] - prev_roots[np.newaxis, :])
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            matched_roots = np.empty(n_poles, dtype=np.complex128)
            matched_roots[col_ind] = curr_roots[row_ind]
            roots_matrix[i] = matched_roots
            prev_roots = matched_roots
        else:
            padded = np.full(n_poles, np.nan + 1j * np.nan, dtype=np.complex128)
            padded[: curr_roots.size] = curr_roots
            roots_matrix[i] = padded

    return RootLocusData(
        gains=gains_arr,
        roots=roots_matrix,
        poles=ol_poles,
        zeros=ol_zeros,
    )


def margin(sys: LinearTimeInvariant) -> StabilityMargins:
    """Compute gain margin, phase margin, and crossover frequencies for a SISO system.

    Parameters
    ----------
    sys : LinearTimeInvariant
        Open-loop SISO LTI system.

    Returns
    -------
    StabilityMargins
        Dataclass containing:
        - gm_db : Gain Margin in dB (inf if phase never reaches -180 deg)
        - pm_deg : Phase Margin in degrees (inf if magnitude never reaches 0 dB)
        - wcg : Gain Crossover Frequency in rad/s (where |G(jw)| = 1)
        - wcp : Phase Crossover Frequency in rad/s (where phase crosses -180 deg)
    """
    if not isinstance(sys, LinearTimeInvariant):
        raise TypeError(f"Expected LinearTimeInvariant instance, got {type(sys).__name__}.")
    if not sys.is_siso:
        raise ValueError(
            f"Stability margin analysis requires a SISO system, got "
            f"({sys.inputs} inputs, {sys.outputs} outputs)."
        )

    if isinstance(sys, TransferFunction):
        tf_sys = sys
    elif isinstance(sys, StateSpace):
        tf_sys = sys.to_tf()
    else:
        raise TypeError(f"Unsupported system type: {type(sys).__name__}")

    # Generate a wide, dense grid for initial bracket detection
    w_grid = _generate_omega(sys, n_points=4000)
    min_log = min(np.log10(w_grid[0]), -4.0)
    max_log = max(np.log10(w_grid[-1]), 5.0)
    w_dense = np.logspace(min_log, max_log, num=5000, dtype=np.float64)

    _, resp_dense = _evaluate_siso_freqresp(tf_sys, w_dense)
    mag_dense = np.abs(resp_dense)
    phase_rad_dense = np.unwrap(np.angle(resp_dense))
    phase_deg_dense = np.rad2deg(phase_rad_dense)

    with np.errstate(divide="ignore"):
        mag_db_dense = np.where(mag_dense > 0.0, 20.0 * np.log10(mag_dense), -np.inf)

    # 1. Gain Crossover Frequency (Wcg) where |G(jw)| = 1 (0 dB)
    wcg_candidates: list[float] = []
    sign_mag = np.sign(mag_db_dense)

    def mag_objective(w: float) -> float:
        _, r = _evaluate_siso_freqresp(tf_sys, np.array([w], dtype=np.float64))
        m = float(np.abs(r[0]))
        return float(20.0 * np.log10(m)) if m > 0 else -1000.0

    for i in range(len(w_dense) - 1):
        if sign_mag[i] * sign_mag[i + 1] <= 0 and sign_mag[i] != sign_mag[i + 1]:
            try:
                sol = root_scalar(
                    mag_objective,
                    bracket=[w_dense[i], w_dense[i + 1]],
                    method="brentq",
                )
                if sol.converged:
                    wcg_candidates.append(float(sol.root))
            except (ValueError, RuntimeError):
                pass

    if wcg_candidates:
        wcg = float(wcg_candidates[0])
        _, r_wcg = _evaluate_siso_freqresp(tf_sys, np.array([wcg], dtype=np.float64))
        raw_phi_deg = float(np.rad2deg(np.angle(r_wcg[0])))
        pm_val = (180.0 + raw_phi_deg + 180.0) % 360.0 - 180.0
        pm_deg = float(pm_val)
    else:
        wcg = float("nan")
        pm_deg = float("inf")

    # 2. Phase Crossover Frequency (Wcp) where phase crosses -180 deg
    wcp_candidates: list[float] = []

    def phase_objective(w: float) -> float:
        _, r = _evaluate_siso_freqresp(tf_sys, np.array([w], dtype=np.float64))
        raw_phi = float(np.rad2deg(np.angle(r[0])))
        err = (raw_phi + 180.0 + 180.0) % 360.0 - 180.0
        return err

    phase_err_dense = (phase_deg_dense + 180.0 + 180.0) % 360.0 - 180.0
    sign_phase = np.sign(phase_err_dense)

    for i in range(len(w_dense) - 1):
        if (
            sign_phase[i] * sign_phase[i + 1] <= 0
            and sign_phase[i] != sign_phase[i + 1]
            and abs(phase_err_dense[i] - phase_err_dense[i + 1]) < 180.0
        ):
            try:
                sol = root_scalar(
                    phase_objective,
                    bracket=[w_dense[i], w_dense[i + 1]],
                    method="brentq",
                )
                if sol.converged:
                    wcp_candidates.append(float(sol.root))
            except (ValueError, RuntimeError):
                pass

    if wcp_candidates:
        wcp = float(wcp_candidates[0])
        _, r_wcp = _evaluate_siso_freqresp(tf_sys, np.array([wcp], dtype=np.float64))
        mag_wcp = float(np.abs(r_wcp[0]))
        if mag_wcp > 0.0:
            gm_db = float(-20.0 * np.log10(mag_wcp))
        else:
            gm_db = float("inf")
    else:
        wcp = float("nan")
        gm_db = float("inf")

    return StabilityMargins(
        gm_db=gm_db,
        pm_deg=pm_deg,
        wcg=wcg,
        wcp=wcp,
    )

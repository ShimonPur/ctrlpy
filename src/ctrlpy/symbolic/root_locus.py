"""Classroom analytical Root Locus rules and step-by-step derivations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import sympy as sp

from ctrlpy.models.transfer_function import TransferFunction


def _normalize_angle_deg(angle_deg: float) -> float:
    """Normalize angle in degrees to the interval (-180, 180]."""
    a = (angle_deg + 180.0) % 360.0 - 180.0
    if a == -180.0:
        return 180.0
    return a


class RootLocusRules:
    r"""Classroom pedagogical Root Locus rule derivation and analyzer.

    Performs complete analytical calculations for all classical Evans root locus rules:
    1. **Branches and Terminations:** Number of open-loop poles ($n$), zeros ($m$), branches ($n$),
       and branches terminating at infinity ($n - m$).
    2. **Real-Axis Segments:** Identifies segments on the real axis lying to the left of an odd
       number of real poles and zeros.
    3. **Asymptotes:** Calculates the real-axis centroid $\sigma_a = \frac{\sum p_i - \sum z_j}{n - m}$
       and asymptote angles $\theta_a = \frac{(2k+1)\cdot 180^\circ}{n - m}$.
    4. **Breakaway and Break-in Points:** Solves $\frac{dK}{ds} = 0 \implies D'(s)N(s) - D(s)N'(s) = 0$
       for real points on the locus with $K \ge 0$.
    5. **Departure and Arrival Angles:** Calculates departure angles from complex open-loop poles
       $\theta_d = 180^\circ - \sum \theta_p + \sum \theta_z$ and arrival angles at complex zeros
       $\theta_a = 180^\circ - \sum \theta_z + \sum \theta_p$.
    6. **Imaginary-Axis Crossings:** Solves for crossing frequencies $\omega$ and critical stability
       gains $K_{\text{crit}}$ where branches cross the $j\omega$-axis ($s = j\omega$).

    Parameters
    ----------
    sys_or_poly : TransferFunction | Any, optional
        Open-loop transfer function $G(s)H(s)$, LTI model, or symbolic expression.
    num : Sequence[float] | None, default=None
        Numerator coefficients if specified explicitly.
    den : Sequence[float] | None, default=None
        Denominator coefficients if specified explicitly.
    poles : Sequence[complex | float] | None, default=None
        Explicit list of open-loop poles.
    zeros : Sequence[complex | float] | None, default=None
        Explicit list of open-loop zeros.
    gain : float, default=1.0
        Open-loop scalar gain factor.

    Attributes
    ----------
    num_poles : int
        Number of open-loop poles ($n$).
    num_zeros : int
        Number of open-loop zeros ($m$).
    num_branches : int
        Number of Root Locus branches ($\max(n, m)$).
    num_asymptotes : int
        Number of branches approaching infinity ($\max(0, n - m)$).
    poles : list[complex]
        Open-loop pole locations.
    zeros : list[complex]
        Open-loop zero locations.
    real_axis_segments : list[tuple[float, float]]
        Intervals on the real axis belonging to the root locus.
    centroid : float | None
        Asymptote center of mass ($\sigma_a$) on the real axis.
    asymptote_angles_deg : list[float]
        Asymptote angles in degrees.
    breakaway_points : list[dict[str, Any]]
        Breakaway and break-in points with their critical gain $K$ and type.
    departure_angles : dict[complex, float]
        Angle of departure from complex open-loop poles in degrees.
    arrival_angles : dict[complex, float]
        Angle of arrival to complex open-loop zeros in degrees.
    imag_axis_crossings : list[dict[str, Any]]
        Crossing frequencies ($\omega$) and critical gains ($K_{\text{crit}}$) on the imaginary axis.
    steps : list[str]
        Pedagogical step-by-step summary for each rule.
    """

    def __init__(
        self,
        sys_or_poly: Any = None,
        *,
        num: Sequence[float] | None = None,
        den: Sequence[float] | None = None,
        poles: Sequence[complex | float] | None = None,
        zeros: Sequence[complex | float] | None = None,
        gain: float = 1.0,
    ) -> None:
        steps: list[str] = []
        detailed_derivations: list[str] = []

        # Parse inputs
        parsed_num: np.ndarray | None = None
        parsed_den: np.ndarray | None = None
        parsed_poles: list[complex] = []
        parsed_zeros: list[complex] = []

        if isinstance(sys_or_poly, TransferFunction):
            parsed_num = np.asarray(sys_or_poly.num, dtype=float)
            parsed_den = np.asarray(sys_or_poly.den, dtype=float)
            parsed_poles = [complex(p) for p in sys_or_poly.poles()]
            parsed_zeros = [complex(z) for z in sys_or_poly.zeros()]
        elif sys_or_poly is not None and hasattr(sys_or_poly, "to_tf"):
            tf_obj = sys_or_poly.to_tf()
            parsed_num = np.asarray(tf_obj.num, dtype=float)
            parsed_den = np.asarray(tf_obj.den, dtype=float)
            parsed_poles = [complex(p) for p in tf_obj.poles()]
            parsed_zeros = [complex(z) for z in tf_obj.zeros()]
        elif sys_or_poly is not None and isinstance(sys_or_poly, sp.Basic):
            # Parse symbolic expression
            s = sp.Symbol("s")
            # Remove any symbolic gain K if present in numerator
            expr = sys_or_poly
            numer, denom = sp.fraction(sp.together(expr))
            p_num = sp.Poly(numer, s)
            p_den = sp.Poly(denom, s)
            c_num = [float(c) for c in p_num.all_coeffs()]
            c_den = [float(c) for c in p_den.all_coeffs()]
            parsed_num = np.asarray(c_num, dtype=float)
            parsed_den = np.asarray(c_den, dtype=float)
            parsed_poles = [complex(r) for r in np.roots(parsed_den)] if len(parsed_den) > 1 else []
            parsed_zeros = [complex(r) for r in np.roots(parsed_num)] if len(parsed_num) > 1 else []
        elif num is not None and den is not None:
            parsed_num = np.asarray(num, dtype=float)
            parsed_den = np.asarray(den, dtype=float)
            parsed_poles = [complex(r) for r in np.roots(parsed_den)] if len(parsed_den) > 1 else []
            parsed_zeros = [complex(r) for r in np.roots(parsed_num)] if len(parsed_num) > 1 else []
        elif poles is not None:
            parsed_poles = [complex(p) for p in poles]
            parsed_zeros = [complex(z) for z in zeros] if zeros is not None else []
            # Form polynomials
            p_poly = np.poly(parsed_poles) if parsed_poles else np.array([1.0])
            z_poly = gain * np.poly(parsed_zeros) if parsed_zeros else np.array([float(gain)])
            parsed_num = np.asarray(z_poly, dtype=float)
            parsed_den = np.asarray(p_poly, dtype=float)
        elif isinstance(sys_or_poly, (list, tuple)) and len(sys_or_poly) == 2:
            parsed_num = np.asarray(sys_or_poly[0], dtype=float)
            parsed_den = np.asarray(sys_or_poly[1], dtype=float)
            parsed_poles = [complex(r) for r in np.roots(parsed_den)] if len(parsed_den) > 1 else []
            parsed_zeros = [complex(r) for r in np.roots(parsed_num)] if len(parsed_num) > 1 else []
        else:
            raise TypeError(
                "RootLocusRules expects a TransferFunction, LTI object, (num, den) pair, "
                "or explicit poles/zeros."
            )

        if parsed_num is None or parsed_den is None:
            raise ValueError("Could not construct open-loop numerator and denominator polynomials.")

        # Ensure monic or float representation
        n = len(parsed_poles)
        m = len(parsed_zeros)
        num_branches = max(n, m)
        num_asymptotes = max(0, n - m)

        # -------------------------------------------------------------
        # Rule 1: Number of Branches & Terminations
        # -------------------------------------------------------------
        rule1_summary = (
            f"Rule 1: Open-loop transfer function has n = {n} poles and m = {m} zeros. "
            f"The root locus has {num_branches} branches; {num_asymptotes} branches terminate at infinity."
        )
        steps.append(rule1_summary)

        poles_fmt = ", ".join(
            f"{p.real:.4g}" if abs(p.imag) < 1e-6 else f"{p.real:.4g} ± {abs(p.imag):.4g}j"
            for p in parsed_poles
        )
        zeros_fmt = (
            ", ".join(
                f"{z.real:.4g}" if abs(z.imag) < 1e-6 else f"{z.real:.4g} ± {abs(z.imag):.4g}j"
                for z in parsed_zeros
            )
            if parsed_zeros
            else r"\text{None}"
        )

        detailed_derivations.append(
            rf"**Rule 1 (Branches & Terminations):** Open-loop system has $n = {n}$ poles (${poles_fmt}$) "
            rf"and $m = {m}$ zeros (${zeros_fmt}$). The root locus consists of **{num_branches} branches** starting at the poles ($K=0$). "
            rf"**{m} branches** terminate at finite zeros, and **{num_asymptotes} branches** terminate at infinity ($\infty$) as $K \to \infty$."
        )

        # -------------------------------------------------------------
        # Rule 2: Real-Axis Segments
        # -------------------------------------------------------------
        real_pts: list[float] = []
        for p in parsed_poles:
            if abs(p.imag) < 1e-6:
                real_pts.append(float(p.real))
        for z in parsed_zeros:
            if abs(z.imag) < 1e-6:
                real_pts.append(float(z.real))

        real_pts.sort(reverse=True)
        real_axis_segments: list[tuple[float, float]] = []

        for idx in range(len(real_pts)):
            if (idx + 1) % 2 == 1:
                high = real_pts[idx]
                low = real_pts[idx + 1] if idx + 1 < len(real_pts) else -float("inf")
                real_axis_segments.append((low, high))

        if real_axis_segments:
            seg_strs = []
            for low, high in real_axis_segments:
                low_str = r"-\infty" if math.isinf(low) else f"{low:.4g}"
                high_str = r"\infty" if math.isinf(high) else f"{high:.4g}"
                seg_strs.append(rf"[{low_str}, {high_str}]")
            seg_tex = r" \cup ".join(seg_strs)
        else:
            seg_tex = r"\emptyset"

        rule2_summary = (
            f"Rule 2: Real-axis locus lies to the left of an odd number of real open-loop poles and zeros: "
            f"{real_axis_segments}."
        )
        steps.append(rule2_summary)
        detailed_derivations.append(
            rf"**Rule 2 (Real-Axis Segments):** A point on the real axis belongs to the root locus if and only if the total number of real open-loop poles and zeros to its right is odd: "
            rf"$$s \in {seg_tex}$$"
        )

        # -------------------------------------------------------------
        # Rule 3: Asymptotes (Centroid and Angles)
        # -------------------------------------------------------------
        centroid: float | None = None
        asymptote_angles: list[float] = []

        if num_asymptotes > 0:
            sum_poles = sum(p.real for p in parsed_poles)
            sum_zeros = sum(z.real for z in parsed_zeros)
            centroid = float((sum_poles - sum_zeros) / num_asymptotes)
            angles_formula_parts: list[str] = []
            for k in range(num_asymptotes):
                angle = ((2 * k + 1) * 180.0) / num_asymptotes
                norm_angle = round(_normalize_angle_deg(angle), 2)
                asymptote_angles.append(norm_angle)
                angles_formula_parts.append(
                    rf"\theta_{k} = \frac{{(2({k})+1) \cdot 180^\circ}}{{{num_asymptotes}}} = {norm_angle:g}^\circ"
                )

            rule3_summary = (
                f"Rule 3: Asymptote centroid sigma_a = ({sum_poles:.4g} - {sum_zeros:.4g}) / {num_asymptotes} = {centroid:.4g}. "
                f"Asymptote angles: {asymptote_angles}."
            )
            steps.append(rule3_summary)
            angles_joined = ", ".join(angles_formula_parts)
            detailed_derivations.append(
                rf"**Rule 3 (Asymptotes of Locus as $s \to \infty$):** "
                rf"The $n - m = {num_asymptotes}$ branches radiating toward infinity follow straight-line asymptotes intersecting the real axis at centroid $\sigma_a$:"
                rf"$$\sigma_a = \frac{{\sum_{{i=1}}^n \operatorname{{Re}}(p_i) - \sum_{{j=1}}^m \operatorname{{Re}}(z_j)}}{{n - m}} = \frac{{{sum_poles:.4g} - ({sum_zeros:.4g})}}{{{num_asymptotes}}} = {centroid:.4g}$$"
                rf"The asymptote angles with the positive real axis are:"
                rf"$$\theta_k = \frac{{(2k + 1) \cdot 180^\circ}}{{n - m}} \implies {angles_joined}$$"
            )
        else:
            steps.append("Rule 3: Number of asymptotes is 0 since n = m.")
            detailed_derivations.append(
                r"**Rule 3 (Asymptotes):** All branches terminate at finite open-loop zeros ($n = m \implies$ no asymptotes toward $\infty$)."
            )

        # -------------------------------------------------------------
        # Rule 4: Breakaway and Break-in Points via dK/ds = 0
        # -------------------------------------------------------------
        num_poly = np.poly1d(parsed_num)
        den_poly = np.poly1d(parsed_den)

        d_den = np.polyder(den_poly)
        d_num = np.polyder(num_poly)

        # Characteristic: D'(s)*N(s) - D(s)*N'(s) = 0
        p_ba = np.polysub(np.polymul(d_den, num_poly), np.polymul(den_poly, d_num))

        breakaway_points: list[dict[str, Any]] = []
        if len(p_ba) > 0 and not np.all(p_ba == 0):
            roots_ba = np.roots(p_ba)
            for r in roots_ba:
                if abs(r.imag) < 1e-5:
                    s_val = float(r.real)
                    num_val = float(np.polyval(parsed_num, s_val))
                    if abs(num_val) > 1e-12:
                        den_val = float(np.polyval(parsed_den, s_val))
                        k_val = -den_val / num_val
                        if k_val >= -1e-6:
                            k_val = max(0.0, k_val)
                            on_locus = any(
                                low - 1e-6 <= s_val <= high + 1e-6
                                for low, high in real_axis_segments
                            )
                            if on_locus:
                                delta = 1e-4
                                den_m = float(np.polyval(parsed_den, s_val - delta))
                                num_m = float(np.polyval(parsed_num, s_val - delta))
                                den_p = float(np.polyval(parsed_den, s_val + delta))
                                num_p = float(np.polyval(parsed_num, s_val + delta))

                                k_m = -den_m / num_m if abs(num_m) > 1e-12 else k_val
                                k_p = -den_p / num_p if abs(num_p) > 1e-12 else k_val

                                pt_type = (
                                    "breakaway"
                                    if (k_val >= k_m - 1e-6 and k_val >= k_p - 1e-6)
                                    else "break-in"
                                )

                                if not any(abs(bp["s"] - s_val) < 1e-4 for bp in breakaway_points):
                                    breakaway_points.append(
                                        {
                                            "s": round(s_val, 4),
                                            "k": round(k_val, 4),
                                            "type": pt_type,
                                        }
                                    )

        rule4_summary = (
            f"Rule 4: Solved dK/ds = 0 for breakaway/break-in candidates: {breakaway_points}."
        )
        steps.append(rule4_summary)

        if breakaway_points:
            bp_tex_list = [
                rf"s = {bp['s']:.4g} \text{{ ({bp['type']}, }} K = {bp['k']:.4g}\text{{)}}"
                for bp in breakaway_points
            ]
            detailed_derivations.append(
                rf"**Rule 4 (Breakaway / Break-in Points):** Solving $\frac{{dK}}{{ds}} = -\frac{{d}}{{ds}}\left[\frac{{D(s)}}{{N(s)}}\right] = 0 \implies D'(s)N(s) - D(s)N'(s) = 0$ yields valid real-axis locus points with $K \ge 0$:"
                rf"$$\text{{Breakaway/Break-in: }} {', '.join(bp_tex_list)}$$"
            )
        else:
            detailed_derivations.append(
                r"**Rule 4 (Breakaway / Break-in Points):** Solving $\frac{dK}{ds} = 0$ yields no valid points with $K \ge 0$ on the real-axis root locus segments."
            )

        # -------------------------------------------------------------
        # Rule 5: Departure Angles from Complex Poles
        # -------------------------------------------------------------
        departure_angles: dict[complex, float] = {}
        for idx, p in enumerate(parsed_poles):
            if abs(p.imag) > 1e-6:
                angle_p_sum = sum(
                    math.atan2(p.imag - other_p.imag, p.real - other_p.real)
                    for j, other_p in enumerate(parsed_poles)
                    if j != idx
                )
                angle_z_sum = sum(
                    math.atan2(p.imag - z.imag, p.real - z.real) for z in parsed_zeros
                )
                theta_dep_rad = math.pi - angle_p_sum + angle_z_sum
                theta_dep_deg = math.degrees(theta_dep_rad)
                departure_angles[p] = round(_normalize_angle_deg(theta_dep_deg), 2)

        if departure_angles:
            steps.append(
                f"Rule 5: Calculated departure angles for complex poles: {departure_angles}."
            )
            dep_tex_list = [
                rf"\theta_{{\text{{dep}}}}(p = {p.real:.3g} + {p.imag:.3g}j) = 180^\circ - \sum \theta_p + \sum \theta_z = {deg:.2f}^\circ"
                for p, deg in departure_angles.items()
                if p.imag > 0
            ]
            detailed_derivations.append(
                rf"**Rule 5 (Angles of Departure from Complex Poles):** "
                rf"By the phase condition $\angle G(s)H(s) = (2k+1)180^\circ$ at $s \approx p_i$:"
                rf"$$\theta_{{\text{{dep}}}} = 180^\circ - \sum \theta_{{p}} + \sum \theta_{{z}} \implies {', '.join(dep_tex_list)}$$"
            )
        else:
            detailed_derivations.append(
                r"**Rule 5 (Angles of Departure):** No complex open-loop poles exist ($\implies$ no departure angle calculations needed)."
            )

        # -------------------------------------------------------------
        # Rule 6: Arrival Angles to Complex Zeros
        # -------------------------------------------------------------
        arrival_angles: dict[complex, float] = {}
        for idx, z in enumerate(parsed_zeros):
            if abs(z.imag) > 1e-6:
                angle_z_sum = sum(
                    math.atan2(z.imag - other_z.imag, z.real - other_z.real)
                    for j, other_z in enumerate(parsed_zeros)
                    if j != idx
                )
                angle_p_sum = sum(
                    math.atan2(z.imag - p.imag, z.real - p.real) for p in parsed_poles
                )
                theta_arr_rad = math.pi - angle_z_sum + angle_p_sum
                theta_arr_deg = math.degrees(theta_arr_rad)
                arrival_angles[z] = round(_normalize_angle_deg(theta_arr_deg), 2)

        if arrival_angles:
            steps.append(f"Rule 6: Calculated arrival angles for complex zeros: {arrival_angles}.")
            arr_tex_list = [
                rf"\theta_{{\text{{arr}}}}(z = {z.real:.3g} + {z.imag:.3g}j) = 180^\circ - \sum \theta_z + \sum \theta_p = {deg:.2f}^\circ"
                for z, deg in arrival_angles.items()
                if z.imag > 0
            ]
            detailed_derivations.append(
                rf"**Rule 6 (Angles of Arrival at Complex Zeros):** "
                rf"By the phase condition $\angle G(s)H(s) = (2k+1)180^\circ$ at $s \approx z_j$:"
                rf"$$\theta_{{\text{{arr}}}} = 180^\circ - \sum \theta_{{z}} + \sum \theta_{{p}} \implies {', '.join(arr_tex_list)}$$"
            )
        else:
            detailed_derivations.append(
                r"**Rule 6 (Angles of Arrival):** No complex open-loop zeros exist ($\implies$ no arrival angle calculations needed)."
            )

        # -------------------------------------------------------------
        # Rule 7: Imaginary Axis Crossings (s = j*omega)
        # -------------------------------------------------------------
        imag_crossings: list[dict[str, Any]] = []
        w_scan = np.logspace(-3, 4, 3000)
        jw_scan = 1j * w_scan

        d_vals = np.polyval(parsed_den, jw_scan)
        n_vals = np.polyval(parsed_num, jw_scan)

        k_complex = -d_vals / n_vals
        k_imag = np.imag(k_complex)

        zero_crossings = np.where(np.diff(np.sign(k_imag)))[0]
        for idx_val in zero_crossings:
            idx_int = int(idx_val)
            w1, w2 = float(w_scan[idx_int]), float(w_scan[idx_int + 1])
            im1, im2 = float(k_imag[idx_int]), float(k_imag[idx_int + 1])
            if im2 != im1:
                w_root = float(w1 - im1 * (w2 - w1) / (im2 - im1))
                for _ in range(8):
                    d_val_c = complex(np.polyval(parsed_den, 1j * w_root))
                    n_val_c = complex(np.polyval(parsed_num, 1j * w_root))
                    if abs(n_val_c) < 1e-12:
                        break
                    k_val_c = -d_val_c / n_val_c
                    im_val = float(np.imag(k_val_c))
                    if abs(im_val) < 1e-9:
                        break
                    dw = 1e-6
                    k_val_dw = -complex(np.polyval(parsed_den, 1j * (w_root + dw))) / complex(
                        np.polyval(parsed_num, 1j * (w_root + dw))
                    )
                    d_im = float((np.imag(k_val_dw) - im_val) / dw)
                    if abs(d_im) < 1e-12:
                        break
                    w_root = float(w_root - im_val / d_im)

                d_cross = complex(np.polyval(parsed_den, 1j * w_root))
                n_cross = complex(np.polyval(parsed_num, 1j * w_root))
                k_at_cross = float(np.real(-d_cross / n_cross))

                if (
                    k_at_cross > 1e-6
                    and w_root > 1e-4
                    and not any(abs(c["omega"] - w_root) < 1e-3 for c in imag_crossings)
                ):
                    imag_crossings.append(
                        {
                            "omega": round(float(w_root), 4),
                            "k": round(k_at_cross, 4),
                            "s": complex(0, round(float(w_root), 4)),
                        }
                    )

        rule7_summary = f"Rule 7: Imaginary axis crossings found: {imag_crossings}."
        steps.append(rule7_summary)

        if imag_crossings:
            cross_tex_list = [
                rf"s = \pm j{c['omega']:.4g} \text{{ (}}\omega = {c['omega']:.4g}\text{{ rad/s) at }} K_{{\text{{crit}}}} = {c['k']:.4g}"
                for c in imag_crossings
            ]
            detailed_derivations.append(
                rf"**Rule 7 (Imaginary Axis Crossings & Marginal Stability):** "
                rf"Substituting $s = j\omega$ into the characteristic equation $D(j\omega) + K N(j\omega) = 0$ yields critical stability crossings with the $j\omega$-axis:"
                rf"$$\text{{Crossings: }} {', '.join(cross_tex_list)}$$"
            )
        else:
            detailed_derivations.append(
                r"**Rule 7 (Imaginary Axis Crossings):** The root locus branches do not intersect the imaginary $j\omega$-axis for any real $K > 0$."
            )

        self.num_poles = n
        self.num_zeros = m
        self.num_branches = num_branches
        self.num_asymptotes = num_asymptotes
        self.poles = parsed_poles
        self.zeros = parsed_zeros
        self.real_axis_segments = real_axis_segments
        self.centroid = centroid
        self.asymptote_angles_deg = asymptote_angles
        self.breakaway_points = breakaway_points
        self.departure_angles = departure_angles
        self.arrival_angles = arrival_angles
        self.imag_axis_crossings = imag_crossings
        self.steps = steps
        self._detailed_derivations = detailed_derivations

    def explain_steps(self) -> list[str]:
        """Return a structured step-by-step list of Markdown/LaTeX explanations for all Evans rules.

        Returns
        -------
        list[str]
            List of step-by-step markdown explanations detailing each rule derivation.
        """
        return list(self._detailed_derivations)

    def _repr_latex_(self) -> str:
        """Render the complete Root Locus derivations as formatted LaTeX for Jupyter."""
        lines = [r"\begin{aligned}"]
        lines.append(r"\textbf{Analytical Root Locus Derivation Summary:}" + r" \\")

        # Rule 1
        lines.append(
            rf"\textbf{{Rule 1 (Branches):}} &\quad n = {self.num_poles}\text{{ poles}}, "
            rf"m = {self.num_zeros}\text{{ zeros}} \implies {self.num_branches}\text{{ branches}}, "
            rf"{self.num_asymptotes}\text{{ asymptotes to }}\infty \\"
        )

        # Rule 2: Real axis
        if self.real_axis_segments:
            seg_strs = []
            for low, high in self.real_axis_segments:
                low_str = r"-\infty" if math.isinf(low) else f"{low:.3g}"
                high_str = r"\infty" if math.isinf(high) else f"{high:.3g}"
                seg_strs.append(rf"[{low_str}, {high_str}]")
            seg_tex = r" \cup ".join(seg_strs)
        else:
            seg_tex = r"\emptyset"
        lines.append(rf"\textbf{{Rule 2 (Real-Axis Segments):}} &\quad s \in {seg_tex} \\")

        # Rule 3: Asymptotes
        if self.num_asymptotes > 0 and self.centroid is not None:
            angles_str = ", ".join(rf"{deg:g}^\circ" for deg in self.asymptote_angles_deg)
            lines.append(
                rf"\textbf{{Rule 3 (Asymptotes):}} &\quad \sigma_a = {self.centroid:.4g}, \quad \theta_k \in \{{{angles_str}\}} \\"
            )
        else:
            lines.append(
                r"\textbf{Rule 3 (Asymptotes):} &\quad \text{No asymptotes to } \infty (n = m) \\"
            )

        # Rule 4: Breakaway points
        if self.breakaway_points:
            bp_strs = [
                rf"s = {bp['s']:.4g} \text{{ ({bp['type']}, }} K = {bp['k']:.4g}\text{{)}}"
                for bp in self.breakaway_points
            ]
            lines.append(rf"\textbf{{Rule 4 (Breakaway/Break-in):}} &\quad {', '.join(bp_strs)} \\")
        else:
            lines.append(
                r"\textbf{Rule 4 (Breakaway/Break-in):} &\quad \text{None on the root locus} \\"
            )

        # Rule 5: Departure angles
        if self.departure_angles:
            dep_strs = [
                rf"\theta_{{\text{{dep}}}}(p={p.real:.3g}{p.imag:+.3g}j) = {deg:.2f}^\circ"
                for p, deg in self.departure_angles.items()
                if p.imag > 0
            ]
            lines.append(rf"\textbf{{Rule 5 (Departure Angles):}} &\quad {', '.join(dep_strs)} \\")

        # Rule 6: Arrival angles
        if self.arrival_angles:
            arr_strs = [
                rf"\theta_{{\text{{arr}}}}(z={z.real:.3g}{z.imag:+.3g}j) = {deg:.2f}^\circ"
                for z, deg in self.arrival_angles.items()
                if z.imag > 0
            ]
            lines.append(rf"\textbf{{Rule 6 (Arrival Angles):}} &\quad {', '.join(arr_strs)} \\")

        # Rule 7: Imaginary axis crossings
        if self.imag_axis_crossings:
            cross_strs = [
                rf"s = \pm j{c['omega']:.4g} \text{{ at }} K_{{\text{{crit}}}} = {c['k']:.4g}"
                for c in self.imag_axis_crossings
            ]
            lines.append(
                rf"\textbf{{Rule 7 (j\omega Crossings):}} &\quad {', '.join(cross_strs)} \\"
            )
        else:
            lines.append(
                r"\textbf{Rule 7 (j\omega Crossings):} &\quad \text{No crossings with the imaginary axis} \\"
            )

        lines.append(r"\end{aligned}")
        return "\n".join(lines)

    def _repr_markdown_(self) -> str:
        """Return a Markdown representation for Jupyter environments."""
        return f"$${self._repr_latex_()}$$"

    def __str__(self) -> str:
        """Format a human-readable ASCII representation of Root Locus rules."""
        lines = ["=== Analytical Root Locus Rules Summary ==="]
        lines.append(
            f"Poles (n={self.num_poles}): {[complex(round(p.real, 4), round(p.imag, 4)) for p in self.poles]}"
        )
        lines.append(
            f"Zeros (m={self.num_zeros}): {[complex(round(z.real, 4), round(z.imag, 4)) for z in self.zeros]}"
        )
        lines.append(
            f"Branches: {self.num_branches}, Asymptotes to Infinity: {self.num_asymptotes}"
        )
        lines.append("")

        # Real axis
        seg_strs = [f"[{low:g}, {high:g}]" for low, high in self.real_axis_segments]
        lines.append(f"Real-Axis Locus Segments: {' U '.join(seg_strs) if seg_strs else 'None'}")

        # Asymptotes
        if self.centroid is not None:
            angles_str = ", ".join(f"{deg:g}°" for deg in self.asymptote_angles_deg)
            lines.append(f"Asymptote Centroid: sigma_a = {self.centroid:.4f}")
            lines.append(f"Asymptote Angles: {angles_str}")

        # Breakaway
        if self.breakaway_points:
            lines.append("Breakaway / Break-in Points:")
            for bp in self.breakaway_points:
                lines.append(f"  - s = {bp['s']:.4f} ({bp['type']}) at K = {bp['k']:.4f}")
        else:
            lines.append("Breakaway / Break-in Points: None")

        # Departure / Arrival
        if self.departure_angles:
            lines.append("Departure Angles from Complex Poles:")
            for p, deg in self.departure_angles.items():
                lines.append(f"  - Pole {p}: {deg:.2f}°")
        if self.arrival_angles:
            lines.append("Arrival Angles to Complex Zeros:")
            for z, deg in self.arrival_angles.items():
                lines.append(f"  - Zero {z}: {deg:.2f}°")

        # jw Crossings
        if self.imag_axis_crossings:
            lines.append("Imaginary Axis Crossings:")
            for c in self.imag_axis_crossings:
                lines.append(f"  - s = ±j{c['omega']:.4f} at K_crit = {c['k']:.4f}")
        else:
            lines.append("Imaginary Axis Crossings: None")

        if self.steps:
            lines.append("\nStep-by-Step Derivation:")
            for s_note in self.steps:
                lines.append(f"  * {s_note}")

        return "\n".join(lines)


# Backwards compatibility alias
RootLocusRulesResult = RootLocusRules


def root_locus_rules(
    sys_or_poly: Any = None,
    *,
    num: Sequence[float] | None = None,
    den: Sequence[float] | None = None,
    poles: Sequence[complex | float] | None = None,
    zeros: Sequence[complex | float] | None = None,
    gain: float = 1.0,
) -> RootLocusRules:
    """Derive formal classroom Root Locus analytical rules step-by-step.

    Parameters
    ----------
    sys_or_poly : TransferFunction | Any, optional
        Open-loop transfer function $G(s)H(s)$, LTI model, or symbolic expression.
    num : Sequence[float] | None, default=None
        Numerator polynomial coefficients.
    den : Sequence[float] | None, default=None
        Denominator polynomial coefficients.
    poles : Sequence[complex | float] | None, default=None
        Open-loop pole locations.
    zeros : Sequence[complex | float] | None, default=None
        Open-loop zero locations.
    gain : float, default=1.0
        Open-loop scalar gain factor.

    Returns
    -------
    RootLocusRules
        Analyzed rules object with metrics, derivation steps, and LaTeX formatting.
    """
    return RootLocusRules(
        sys_or_poly=sys_or_poly,
        num=num,
        den=den,
        poles=poles,
        zeros=zeros,
        gain=gain,
    )

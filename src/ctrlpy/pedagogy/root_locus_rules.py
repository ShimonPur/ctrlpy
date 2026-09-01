"""Classroom analytical Root Locus rules and derivations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ctrlpy.models.transfer_function import TransferFunction


@dataclass
class RootLocusRulesResult:
    """Container for analytical Root Locus rules and derivations.

    Attributes
    ----------
    num_poles : int
        Number of open-loop poles (n).
    num_zeros : int
        Number of open-loop zeros (m).
    num_branches : int
        Number of Root Locus branches.
    num_asymptotes : int
        Number of branches approaching infinity (n - m).
    poles : list[complex]
        Open-loop pole locations.
    zeros : list[complex]
        Open-loop zero locations.
    real_axis_segments : list[tuple[float, float]]
        Intervals on the real axis belonging to the root locus.
    centroid : float | None
        Asymptote center of mass (sigma_a) on the real axis.
    asymptote_angles_deg : list[float]
        Asymptote angles in degrees.
    breakaway_points : list[dict[str, Any]]
        Breakaway and break-in points with their critical gain K.
    departure_angles : dict[complex, float]
        Angle of departure from complex open-loop poles in degrees.
    arrival_angles : dict[complex, float]
        Angle of arrival to complex open-loop zeros in degrees.
    imag_axis_crossings : list[dict[str, Any]]
        Crossing frequencies (omega) and critical gains (K_crit) on the imaginary axis.
    steps : list[str]
        Pedagogical step-by-step summary for each rule.
    """

    num_poles: int
    num_zeros: int
    num_branches: int
    num_asymptotes: int
    poles: list[complex]
    zeros: list[complex]
    real_axis_segments: list[tuple[float, float]]
    centroid: float | None
    asymptote_angles_deg: list[float]
    breakaway_points: list[dict[str, Any]] = field(default_factory=list)
    departure_angles: dict[complex, float] = field(default_factory=dict)
    arrival_angles: dict[complex, float] = field(default_factory=dict)
    imag_axis_crossings: list[dict[str, Any]] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)

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
                rf"\theta_{{\text{{dep}}}}(p={p:.3g}) = {deg:.2f}^\circ"
                for p, deg in self.departure_angles.items()
            ]
            lines.append(rf"\textbf{{Rule 5 (Departure Angles):}} &\quad {', '.join(dep_strs)} \\")

        # Rule 6: Arrival angles
        if self.arrival_angles:
            arr_strs = [
                rf"\theta_{{\text{{arr}}}}(z={z:.3g}) = {deg:.2f}^\circ"
                for z, deg in self.arrival_angles.items()
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
            for s in self.steps:
                lines.append(f"  * {s}")

        return "\n".join(lines)


def _normalize_angle_deg(angle_deg: float) -> float:
    """Normalize angle in degrees to the interval (-180, 180]."""
    a = (angle_deg + 180.0) % 360.0 - 180.0
    if a == -180.0:
        return 180.0
    return a


def root_locus_rules(sys: TransferFunction | Any) -> RootLocusRulesResult:
    """Derive formal classroom Root Locus analytical rules step-by-step.

    Computes:
    1. Number of poles (n), zeros (m), branches (n), and asymptotes (n - m).
    2. Real-axis root locus segments.
    3. Asymptote angles and real-axis centroid (sigma_a).
    4. Breakaway and break-in points via dK/ds = 0.
    5. Departure angles from complex poles and arrival angles to complex zeros.
    6. Imaginary-axis crossing points (s = jw) and critical gain K_crit.

    Parameters
    ----------
    sys : TransferFunction
        The open-loop LTI transfer function G(s)H(s).

    Returns
    -------
    RootLocusRulesResult
        Dataclass containing the analytical rules, calculated values, and pedagogical steps.

    Examples
    --------
    >>> import ctrlpy as cp
    >>> from ctrlpy.pedagogy import root_locus_rules
    >>> G = cp.tf([1], [1, 3, 2, 0])
    >>> res = root_locus_rules(G)
    >>> res.num_asymptotes
    3
    >>> res.centroid
    -1.0
    """
    if not isinstance(sys, TransferFunction):
        if hasattr(sys, "to_tf"):
            sys = sys.to_tf()
        else:
            raise TypeError(
                "Expected a TransferFunction instance or an LTI object convertible to TransferFunction."
            )

    steps: list[str] = []

    # 1. Poles and Zeros
    raw_poles = sys.poles()
    raw_zeros = sys.zeros()
    poles = [complex(p) for p in raw_poles]
    zeros = [complex(z) for z in raw_zeros]

    n = len(poles)
    m = len(zeros)
    num_branches = max(n, m)
    num_asymptotes = max(0, n - m)

    steps.append(
        f"Rule 1: Open-loop has n = {n} poles and m = {m} zeros. "
        f"There are {num_branches} branches; {num_asymptotes} branches terminate at infinity."
    )

    # 2. Real-Axis Segments
    # Collect real poles and real zeros
    real_pts: list[float] = []
    for p in poles:
        if abs(p.imag) < 1e-6:
            real_pts.append(float(p.real))
    for z in zeros:
        if abs(z.imag) < 1e-6:
            real_pts.append(float(z.real))

    real_pts.sort(reverse=True)
    real_axis_segments: list[tuple[float, float]] = []

    # Check intervals between sorted real points
    for idx in range(len(real_pts)):
        # Points strictly to the right has count (idx + 1)
        if (idx + 1) % 2 == 1:
            high = real_pts[idx]
            low = real_pts[idx + 1] if idx + 1 < len(real_pts) else -float("inf")
            real_axis_segments.append((low, high))

    steps.append(
        f"Rule 2: Real-axis locus lies to the left of an odd number of real open-loop poles and zeros: "
        f"{real_axis_segments}."
    )

    # 3. Asymptotes (Centroid and Angles)
    centroid: float | None = None
    asymptote_angles: list[float] = []

    if num_asymptotes > 0:
        sum_poles = sum(p.real for p in poles)
        sum_zeros = sum(z.real for z in zeros)
        centroid = float((sum_poles - sum_zeros) / num_asymptotes)
        for k in range(num_asymptotes):
            angle = ((2 * k + 1) * 180.0) / num_asymptotes
            asymptote_angles.append(round(_normalize_angle_deg(angle), 2))
        steps.append(
            f"Rule 3: Asymptote centroid sigma_a = ({sum_poles:.4g} - {sum_zeros:.4g}) / {num_asymptotes} = {centroid:.4g}. "
            f"Asymptote angles: {asymptote_angles}."
        )
    else:
        steps.append("Rule 3: Number of asymptotes is 0 since n = m.")

    # 4. Breakaway and Break-in Points via dK/ds = 0
    # K(s) = - D(s) / N(s) => dK/ds = - (D'(s)N(s) - D(s)N'(s)) / N(s)^2 = 0
    num_poly = np.poly1d(sys.num)
    den_poly = np.poly1d(sys.den)

    d_den = np.polyder(den_poly)
    d_num = np.polyder(num_poly)

    # P_ba = D'(s)*N(s) - D(s)*N'(s)
    p_ba = np.polysub(np.polymul(d_den, num_poly), np.polymul(den_poly, d_num))

    breakaway_points: list[dict[str, Any]] = []
    if len(p_ba) > 0 and not np.all(p_ba == 0):
        roots_ba = np.roots(p_ba)
        for r in roots_ba:
            if abs(r.imag) < 1e-5:
                s_val = float(r.real)
                # Check gain K(s_val) = - D(s)/N(s)
                num_val = float(np.polyval(sys.num, s_val))
                if abs(num_val) > 1e-12:
                    den_val = float(np.polyval(sys.den, s_val))
                    k_val = -den_val / num_val
                    if k_val >= -1e-6:
                        k_val = max(0.0, k_val)
                        # Check if s_val is on a real axis segment
                        on_locus = any(
                            low - 1e-6 <= s_val <= high + 1e-6 for low, high in real_axis_segments
                        )
                        if on_locus:
                            # Evaluate second derivative / nearby curvature to distinguish breakaway from break-in
                            delta = 1e-4
                            den_m = float(np.polyval(sys.den, s_val - delta))
                            num_m = float(np.polyval(sys.num, s_val - delta))
                            den_p = float(np.polyval(sys.den, s_val + delta))
                            num_p = float(np.polyval(sys.num, s_val + delta))

                            k_m = -den_m / num_m if abs(num_m) > 1e-12 else k_val
                            k_p = -den_p / num_p if abs(num_p) > 1e-12 else k_val

                            # If local max of K(s) on real axis -> breakaway
                            pt_type = (
                                "breakaway"
                                if (k_val >= k_m - 1e-6 and k_val >= k_p - 1e-6)
                                else "break-in"
                            )

                            # Avoid duplicate points
                            if not any(abs(bp["s"] - s_val) < 1e-4 for bp in breakaway_points):
                                breakaway_points.append(
                                    {
                                        "s": round(s_val, 4),
                                        "k": round(k_val, 4),
                                        "type": pt_type,
                                    }
                                )

    steps.append(
        f"Rule 4: Solved dK/ds = 0 for breakaway/break-in candidates. Found on-locus points: {breakaway_points}."
    )

    # 5. Departure Angles from Complex Poles
    departure_angles: dict[complex, float] = {}
    for idx, p in enumerate(poles):
        if abs(p.imag) > 1e-6:
            # Sum angles from other poles
            angle_p_sum = sum(
                math.atan2(p.imag - other_p.imag, p.real - other_p.real)
                for j, other_p in enumerate(poles)
                if j != idx
            )
            # Sum angles from zeros
            angle_z_sum = sum(math.atan2(p.imag - z.imag, p.real - z.real) for z in zeros)
            # theta_dep = 180 - sum(pole_angles) + sum(zero_angles)
            theta_dep_rad = math.pi - angle_p_sum + angle_z_sum
            theta_dep_deg = math.degrees(theta_dep_rad)
            departure_angles[p] = round(_normalize_angle_deg(theta_dep_deg), 2)

    if departure_angles:
        steps.append(f"Rule 5: Calculated departure angles for complex poles: {departure_angles}.")

    # 6. Arrival Angles to Complex Zeros
    arrival_angles: dict[complex, float] = {}
    for idx, z in enumerate(zeros):
        if abs(z.imag) > 1e-6:
            # Sum angles from other zeros
            angle_z_sum = sum(
                math.atan2(z.imag - other_z.imag, z.real - other_z.real)
                for j, other_z in enumerate(zeros)
                if j != idx
            )
            # Sum angles from poles
            angle_p_sum = sum(math.atan2(z.imag - p.imag, z.real - p.real) for p in poles)
            # theta_arr = 180 - sum(zero_angles) + sum(pole_angles)
            theta_arr_rad = math.pi - angle_z_sum + angle_p_sum
            theta_arr_deg = math.degrees(theta_arr_rad)
            arrival_angles[z] = round(_normalize_angle_deg(theta_arr_deg), 2)

    if arrival_angles:
        steps.append(f"Rule 6: Calculated arrival angles for complex zeros: {arrival_angles}.")

    # 7. Imaginary Axis Crossings (s = j*omega)
    # D(jw) + K * N(jw) = 0 => Re(D(jw))*Im(N(jw)) - Im(D(jw))*Re(N(jw)) = 0
    # We test frequency grid / polynomial roots in w
    imag_crossings: list[dict[str, Any]] = []

    # Let's search for w > 0 where Re(D)/Re(N) == Im(D)/Im(N)
    # Form polynomial D_R(w)*N_I(w) - D_I(w)*N_R(w)
    # D(jw): powers of jw -> (jw)^k = j^k * w^k
    # We can evaluate symbolically or scan high-resolution roots
    w_scan = np.logspace(-3, 4, 2000)
    jw_scan = 1j * w_scan

    d_vals = np.polyval(sys.den, jw_scan)
    n_vals = np.polyval(sys.num, jw_scan)

    # Condition for real positive K: Im(-D(jw)/N(jw)) == 0 and Re(-D(jw)/N(jw)) > 0
    k_complex = -d_vals / n_vals
    k_imag = np.imag(k_complex)

    # Detect zero crossings of k_imag
    zero_crossings = np.where(np.diff(np.sign(k_imag)))[0]
    for idx_val in zero_crossings:
        idx = int(idx_val)
        w1, w2 = float(w_scan[idx]), float(w_scan[idx + 1])
        im1, im2 = float(k_imag[idx]), float(k_imag[idx + 1])
        if im2 != im1:
            w_root = float(w1 - im1 * (w2 - w1) / (im2 - im1))
            # Refine root using secant / Newton
            for _ in range(5):
                d_val_c = complex(np.polyval(sys.den, 1j * w_root))
                n_val_c = complex(np.polyval(sys.num, 1j * w_root))
                if abs(n_val_c) < 1e-12:
                    break
                k_val_c = -d_val_c / n_val_c
                im_val = float(np.imag(k_val_c))
                if abs(im_val) < 1e-8:
                    break
                # numerical derivative
                dw = 1e-6
                k_val_dw = -complex(np.polyval(sys.den, 1j * (w_root + dw))) / complex(
                    np.polyval(sys.num, 1j * (w_root + dw))
                )
                d_im = float((np.imag(k_val_dw) - im_val) / dw)
                if abs(d_im) < 1e-12:
                    break
                w_root = float(w_root - im_val / d_im)

            d_cross = complex(np.polyval(sys.den, 1j * w_root))
            n_cross = complex(np.polyval(sys.num, 1j * w_root))
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

    steps.append(f"Rule 7: Imaginary axis crossings found: {imag_crossings}.")

    return RootLocusRulesResult(
        num_poles=n,
        num_zeros=m,
        num_branches=num_branches,
        num_asymptotes=num_asymptotes,
        poles=poles,
        zeros=zeros,
        real_axis_segments=real_axis_segments,
        centroid=centroid,
        asymptote_angles_deg=asymptote_angles,
        breakaway_points=breakaway_points,
        departure_angles=departure_angles,
        arrival_angles=arrival_angles,
        imag_axis_crossings=imag_crossings,
        steps=steps,
    )

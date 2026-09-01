"""Routh-Hurwitz stability criterion and step-by-step array construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sympy as sp

from ctrlpy.models.transfer_function import TransferFunction


@dataclass
class RouthResult:
    """Container for Routh-Hurwitz stability analysis results.

    Attributes
    ----------
    table : list[list[Any]]
        The complete computed Routh array containing numeric or symbolic expressions.
    num_rhp_poles : int
        Number of open-loop / closed-loop poles in the open Right-Half Plane (RHP).
    is_stable : bool
        True if all poles lie strictly in the open Left-Half Plane (LHP).
    k_range : str | None
        Parametric stability range for symbolic gain K, if applicable.
    steps : list[str]
        Pedagogical step-by-step explanatory notes during array construction.
    row_labels : list[str]
        Labels for each row of the Routh array (e.g. ['s^3', 's^2', 's^1', 's^0']).
    auxiliary_polynomials : list[Any]
        Any auxiliary polynomials formed when encountering rows of all zeros.
    polynomial : Any
        The original characteristic polynomial analyzed.
    variable : sp.Symbol
        The symbolic variable used for polynomial powers.
    """

    table: list[list[Any]]
    num_rhp_poles: int
    is_stable: bool
    k_range: str | None = None
    steps: list[str] = field(default_factory=list)
    row_labels: list[str] = field(default_factory=list)
    auxiliary_polynomials: list[Any] = field(default_factory=list)
    polynomial: Any = None
    variable: Any = None

    def _repr_latex_(self) -> str:
        """Render the Routh table and stability results as formatted LaTeX for Jupyter."""
        max_cols = max((len(r) for r in self.table), default=1)
        col_spec = "c|" + "c" * max_cols
        rows_latex: list[str] = []

        for label, row in zip(self.row_labels, self.table, strict=False):
            # Format row label
            lbl_tex = f"{label}"
            if "^" in label:
                base, pwr = label.split("^", 1)
                lbl_tex = f"{base}^{{{pwr}}}"

            # Format row entries
            entries = [
                sp.latex(val) if hasattr(val, "as_expr") or isinstance(val, sp.Basic) else str(val)
                for val in row
            ]
            rows_latex.append(f"{lbl_tex} & " + " & ".join(entries) + r" \\")

        table_body = "\n".join(rows_latex)
        latex_str = (
            r"\begin{aligned}"
            r"\textbf{Routh-Hurwitz Array:}" + "\n"
            r"\begin{array}{" + col_spec + r"}" + "\n" + table_body + "\n"
            r"\end{array}" + "\n\n"
        )

        status_str = (
            r"\text{Strictly Stable (All LHP poles)}"
            if self.is_stable
            else r"\text{Unstable / Marginally Stable}"
        )
        latex_str += rf"\textbf{{Stability Status:}} &\quad {status_str} \\" + "\n"
        latex_str += rf"\textbf{{RHP Poles:}} &\quad {self.num_rhp_poles} \\" + "\n"

        if self.k_range:
            latex_str += rf"\textbf{{Stable Gain Range (K):}} &\quad {self.k_range} \\" + "\n"

        if self.auxiliary_polynomials:
            aux_str = ", ".join(sp.latex(p) for p in self.auxiliary_polynomials)
            latex_str += rf"\textbf{{Auxiliary Polynomials:}} &\quad {aux_str} \\" + "\n"

        latex_str += r"\end{aligned}"
        return latex_str

    def _repr_markdown_(self) -> str:
        """Return a Markdown representation for Jupyter environments."""
        return f"$${self._repr_latex_()}$$"

    def __str__(self) -> str:
        """Format a human-readable ASCII representation of the Routh table."""
        lines = ["=== Routh-Hurwitz Stability Criterion ==="]
        if self.polynomial is not None:
            lines.append(f"Polynomial: {self.polynomial} = 0")
        lines.append("")

        # Calculate max string length for columns
        col_strs: list[list[str]] = []
        for label, row in zip(self.row_labels, self.table, strict=False):
            row_str = [label] + [str(sp.simplify(v) if isinstance(v, sp.Basic) else v) for v in row]
            col_strs.append(row_str)

        max_cols = max(len(r) for r in col_strs)
        col_widths = [0] * max_cols
        for r in col_strs:
            for c_idx, val in enumerate(r):
                col_widths[c_idx] = max(col_widths[c_idx], len(val))

        for r in col_strs:
            label_part = r[0].rjust(col_widths[0])
            entries_part = " | " + "  ".join(r[i].rjust(col_widths[i]) for i in range(1, len(r)))
            lines.append(f"{label_part}{entries_part}")

        lines.append("-" * 42)
        lines.append(f"Number of RHP Poles: {self.num_rhp_poles}")
        lines.append(f"Asymptotically Stable: {self.is_stable}")
        if self.k_range:
            lines.append(f"Stability Range for K: {self.k_range}")
        if self.auxiliary_polynomials:
            lines.append(f"Auxiliary Polynomials: {self.auxiliary_polynomials}")
        if self.steps:
            lines.append("\nConstruction Notes:")
            for s in self.steps:
                lines.append(f"  - {s}")
        return "\n".join(lines)


def _is_zero(expr: Any) -> bool:
    """Check if an expression is algebraically equal to zero."""
    if expr == 0 or expr == 0.0:
        return True
    if isinstance(expr, sp.Basic):
        return bool(sp.simplify(expr) == 0)
    try:
        return bool(float(expr) == 0.0)
    except (TypeError, ValueError):
        return False


def _eval_sign_at_limit(expr: Any, eps: sp.Symbol) -> int:
    """Evaluate the sign of an expression in the limit as eps -> 0+."""
    if not isinstance(expr, sp.Basic) or eps not in expr.free_symbols:
        try:
            val = float(expr)
            if val > 1e-14:
                return 1
            if val < -1e-14:
                return -1
            return 0
        except (TypeError, ValueError):
            return 0

    lim = sp.limit(expr, eps, 0, dir="+")
    if lim == sp.oo:
        return 1
    if lim == -sp.oo:
        return -1
    try:
        val = float(lim)
        if val > 1e-14:
            return 1
        if val < -1e-14:
            return -1
    except (TypeError, ValueError):
        pass

    # If limit is 0, inspect the sign of the leading series term
    try:
        series = sp.series(expr, eps, 0, n=6)
        lead = series.as_leading_term(eps)
        coeff = lead.as_coeff_exponent(eps)[0]
        c_val = float(coeff)
        return 1 if c_val > 0 else (-1 if c_val < 0 else 0)
    except (TypeError, ValueError, AttributeError):
        return 1


def _solve_k_range(col1_entries: list[Any], k_sym: sp.Symbol) -> str | None:
    """Solve for the valid parametric range of K that keeps all column 1 entries positive."""
    conditions: list[sp.Expr] = []
    for entry in col1_entries:
        if isinstance(entry, sp.Basic) and k_sym in entry.free_symbols:
            conditions.append(entry)

    if not conditions:
        return None

    try:
        # Solve system of inequalities entry > 0 for all entries
        # Test across real line critical points (zeros and poles of conditions)
        crit_points: set[sp.Rational | sp.Float | sp.Integer] = set()
        for cond in conditions:
            numer, denom = sp.fraction(sp.together(cond))
            for z in sp.solve(numer, k_sym):
                if hasattr(z, "is_real") and z.is_real:
                    crit_points.add(z)
            for p in sp.solve(denom, k_sym):
                if hasattr(p, "is_real") and p.is_real:
                    crit_points.add(p)

        sorted_pts = sorted(crit_points, key=lambda x: float(x))
        # Build test intervals
        intervals: list[tuple[float, float]] = []
        bounds = [-sp.oo] + sorted_pts + [sp.oo]

        for i in range(len(bounds) - 1):
            low, high = bounds[i], bounds[i + 1]
            if low == -sp.oo and high == sp.oo:
                test_pt = sp.Integer(0)
            elif low == -sp.oo:
                test_pt = high - 1
            elif high == sp.oo:
                test_pt = low + 1
            else:
                test_pt = (low + high) / 2

            # Check if all conditions are strictly positive at test_pt
            all_pos = True
            for cond in conditions:
                val = cond.subs(k_sym, test_pt)
                if not (_is_zero(val) is False and val > 0):
                    all_pos = False
                    break
            if all_pos:
                intervals.append((float(low), float(high)))

        # Format intervals nicely
        if not intervals:
            return "No stable range for K"

        parts: list[str] = []
        for low, high in intervals:
            if low == -float("inf") and high == float("inf"):
                parts.append(f"All real {k_sym}")
            elif low == -float("inf"):
                parts.append(f"{k_sym} < {high:g}")
            elif high == float("inf"):
                parts.append(f"{k_sym} > {low:g}")
            else:
                parts.append(f"{low:g} < {k_sym} < {high:g}")
        return " or ".join(parts)
    except (TypeError, ValueError, AttributeError, sp.SympifyError):
        # Fallback to symbolic solve representation
        return None


def routh_table(
    poly_or_sys: Any,
    variable: str | sp.Symbol = "s",
    k_symbol: str | sp.Symbol | None = None,
) -> RouthResult:
    """Construct the Routh-Hurwitz array and analyze system stability.

    Handles textbook special cases:
    1. Zero in the first column: replaced with symbolic epsilon > 0.
    2. Row of all zeros: forms auxiliary polynomial A(s) and replaces row with dA/ds.
    3. Parametric gain K: evaluates column 1 inequalities to return stable gain bounds.

    Parameters
    ----------
    poly_or_sys : TransferFunction | list | np.ndarray | sp.Expr | sp.Poly
        The polynomial coefficients (highest degree first), characteristic expression,
        or a TransferFunction model.
    variable : str | sp.Symbol, default="s"
        The Laplace variable symbol.
    k_symbol : str | sp.Symbol | None, default=None
        Optional parametric gain symbol to solve for stability boundaries.

    Returns
    -------
    RouthResult
        Dataclass containing the Routh array, RHP pole count, stability flag,
        parametric K-range, and pedagogical steps.

    Examples
    --------
    >>> from ctrlpy.pedagogy import routh_table
    >>> res = routh_table([1, 2, 3, 4])
    >>> res.is_stable
    True
    >>> res.num_rhp_poles
    0
    """
    s = sp.Symbol(variable) if isinstance(variable, str) else variable
    k_sym = sp.Symbol(k_symbol) if isinstance(k_symbol, str) else k_symbol

    steps: list[str] = []
    auxiliary_polys: list[Any] = []

    # 1. Parse input into list of coefficients
    if isinstance(poly_or_sys, TransferFunction):
        # Denominator of transfer function or closed-loop characteristic equation
        if k_sym is not None:
            # Form closed loop characteristic poly: D(s) + K * N(s)
            num_poly = sum(
                c * s ** (len(poly_or_sys.num) - 1 - i) for i, c in enumerate(poly_or_sys.num)
            )
            den_poly = sum(
                c * s ** (len(poly_or_sys.den) - 1 - i) for i, c in enumerate(poly_or_sys.den)
            )
            char_expr = sp.expand(den_poly + k_sym * num_poly)
            p = sp.Poly(char_expr, s)
            coeffs = [sp.simplify(c) for c in p.all_coeffs()]
            poly_display = char_expr
        else:
            coeffs = [sp.sympify(c) for c in poly_or_sys.den]
            poly_display = sum(c * s ** (len(coeffs) - 1 - i) for i, c in enumerate(coeffs))
    elif isinstance(poly_or_sys, (list, tuple, np.ndarray)):
        coeffs = [sp.sympify(c) for c in poly_or_sys]
        poly_display = sum(c * s ** (len(coeffs) - 1 - i) for i, c in enumerate(coeffs))
    elif isinstance(poly_or_sys, sp.Poly):
        coeffs = [sp.simplify(c) for c in poly_or_sys.all_coeffs()]
        poly_display = poly_or_sys.as_expr()
    elif isinstance(poly_or_sys, sp.Basic):
        p = sp.Poly(poly_or_sys, s)
        coeffs = [sp.simplify(c) for c in p.all_coeffs()]
        poly_display = poly_or_sys
    else:
        raise TypeError(f"Unsupported polynomial or system type: {type(poly_or_sys)}")

    # Detect if k_sym is present in expressions if not explicitly provided
    if k_sym is None:
        free_syms = set()
        for c in coeffs:
            if isinstance(c, sp.Basic):
                free_syms.update(c.free_symbols)
        free_syms.discard(s)
        if len(free_syms) == 1:
            k_sym = next(iter(free_syms))

    n = len(coeffs) - 1
    if n < 0:
        raise ValueError("Empty polynomial provided.")

    if n == 0:
        return RouthResult(
            table=[[coeffs[0]]],
            num_rhp_poles=0,
            is_stable=True,
            steps=["Degree 0 polynomial (constant)."],
            row_labels=["s^0"],
            polynomial=poly_display,
            variable=s,
        )

    num_cols = (n + 2) // 2
    row_labels = [f"s^{n - i}" for i in range(n + 1)]
    table: list[list[Any]] = []

    # Row 0 (s^n) and Row 1 (s^(n-1))
    row0: list[Any] = [
        coeffs[2 * j] if 2 * j < len(coeffs) else sp.Integer(0) for j in range(num_cols)
    ]
    row1: list[Any] = [
        coeffs[2 * j + 1] if 2 * j + 1 < len(coeffs) else sp.Integer(0) for j in range(num_cols)
    ]
    table.append(row0)
    table.append(row1)

    steps.append(
        f"Initialized Row {row_labels[0]} and Row {row_labels[1]} from polynomial coefficients."
    )

    eps = sp.Symbol("epsilon", positive=True)

    # Compute rows 2 through n
    for i in range(2, n + 1):
        prev_row = table[i - 1]
        prev_prev_row = table[i - 2]
        pwr_label = row_labels[i]

        # Check Special Case 1: First element in previous row is 0, but row is not all zeros
        if _is_zero(prev_row[0]) and not all(_is_zero(x) for x in prev_row):
            table[i - 1][0] = eps
            prev_row[0] = eps
            steps.append(
                f"Special Case 1: First element in row {row_labels[i - 1]} is zero. "
                "Replaced with symbolic epsilon > 0."
            )

        # Check if previous row is all zeros (Special Case 2)
        if all(_is_zero(x) for x in prev_row):
            # Auxiliary polynomial from row i-2
            aux_pwr = n - (i - 2)
            aux_terms = []
            deriv_row: list[Any] = []
            for j in range(num_cols):
                term_pwr = aux_pwr - 2 * j
                c = prev_prev_row[j]
                if term_pwr >= 0 and not _is_zero(c):
                    aux_terms.append(c * s**term_pwr)
                deriv_c = c * term_pwr if term_pwr >= 0 else sp.Integer(0)
                deriv_row.append(sp.simplify(deriv_c))

            aux_poly = sum(aux_terms)
            auxiliary_polys.append(aux_poly)
            table[i - 1] = deriv_row
            prev_row = deriv_row
            steps.append(
                f"Special Case 2: Row {row_labels[i - 1]} was all zeros. "
                f"Formed auxiliary polynomial A(s) = {aux_poly} from row {row_labels[i - 2]} "
                "and replaced zero row with dA/ds coefficients."
            )

        # Compute row i entries: R[i, j] = (R[i-1, 0]*R[i-2, j+1] - R[i-2, 0]*R[i-1, j+1]) / R[i-1, 0]
        new_row: list[Any] = []
        a = prev_row[0]
        c = prev_prev_row[0]

        for j in range(num_cols):
            b = prev_prev_row[j + 1] if j + 1 < num_cols else sp.Integer(0)
            d = prev_row[j + 1] if j + 1 < num_cols else sp.Integer(0)
            if _is_zero(a):
                entry = sp.Integer(0)
            else:
                entry = sp.simplify((a * b - c * d) / a)
            new_row.append(entry)

        # If the newly computed row is all zeros and not at the end
        if all(_is_zero(x) for x in new_row) and i <= n:
            aux_pwr = n - (i - 1)
            aux_terms = []
            deriv_row = []
            for j in range(num_cols):
                term_pwr = aux_pwr - 2 * j
                coeff_val = prev_row[j]
                if term_pwr >= 0 and not _is_zero(coeff_val):
                    aux_terms.append(coeff_val * s**term_pwr)
                deriv_c = coeff_val * term_pwr if term_pwr >= 0 else sp.Integer(0)
                deriv_row.append(sp.simplify(deriv_c))

            aux_poly = sum(aux_terms)
            auxiliary_polys.append(aux_poly)
            steps.append(
                f"Special Case 2: Row {pwr_label} computed as all zeros. "
                f"Formed auxiliary polynomial A(s) = {aux_poly} from row {row_labels[i - 1]} "
                "and replaced with dA/ds coefficients."
            )
            new_row = deriv_row

        table.append(new_row)

    # 3. Analyze Column 1 for Stability & Sign Changes
    col1 = [row[0] for row in table]
    k_range = _solve_k_range(col1, k_sym) if k_sym is not None else None

    # Evaluate signs for numerical / epsilon stability
    signs: list[int] = []
    has_unresolved_param = False
    for entry in col1:
        if isinstance(entry, sp.Basic) and k_sym is not None and k_sym in entry.free_symbols:
            has_unresolved_param = True
            break
        sign = _eval_sign_at_limit(entry, eps)
        signs.append(sign)

    if has_unresolved_param:
        num_rhp = -1
        is_stable = False
        steps.append(f"Parametric analysis performed for symbolic gain {k_sym}.")
    else:
        # Count sign changes
        num_rhp = 0
        non_zero_signs = [sg for sg in signs if sg != 0]
        for idx in range(len(non_zero_signs) - 1):
            if non_zero_signs[idx] * non_zero_signs[idx + 1] < 0:
                num_rhp += 1

        is_stable = (
            (num_rhp == 0) and (len(auxiliary_polys) == 0) and (len(non_zero_signs) == len(signs))
        )
        steps.append(f"Column 1 signs: {signs} -> {num_rhp} sign changes detected.")

    return RouthResult(
        table=table,
        num_rhp_poles=num_rhp,
        is_stable=is_stable,
        k_range=k_range,
        steps=steps,
        row_labels=row_labels,
        auxiliary_polynomials=auxiliary_polys,
        polynomial=poly_display,
        variable=s,
    )

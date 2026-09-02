"""Analytical State-Space Canonical Forms and Controllability/Observability Tutor."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import sympy as sp

from ctrlpy.exceptions import DimensionMismatchError

if TYPE_CHECKING:
    from ctrlpy.models.state_space import StateSpace
    from ctrlpy.models.transfer_function import TransferFunction


def _to_sympy_matrix(val: Any) -> sp.Matrix:
    """Convert input array, list, scalar, or SymPy object to a simplified SymPy Matrix.

    Parameters
    ----------
    val : Any
        Input array-like or SymPy matrix.

    Returns
    -------
    sp.Matrix
        SymPy Matrix representation with simplified exact/rational entries.
    """
    if isinstance(val, sp.Matrix):
        return val.applyfunc(
            lambda x: (
                sp.nsimplify(x, rational=True)
                if hasattr(x, "is_Float") and x.is_Float
                else sp.simplify(x)
            )
        )

    if hasattr(val, "tolist"):
        val = val.tolist()

    if isinstance(val, (int, float, complex, sp.Basic)) and not isinstance(val, (list, tuple)):
        val = [[val]]

    mat = sp.Matrix(val)
    if mat.cols == 0 and mat.rows > 0:
        mat = mat.reshape(mat.rows, 1)

    # Simplify floating numbers to clean rationals/integers when appropriate
    def _clean_elem(elem: Any) -> sp.Expr:
        if isinstance(elem, (int, sp.Integer)):
            return sp.Integer(elem)
        if isinstance(elem, (float, np.floating)):
            if np.isclose(elem, round(float(elem)), atol=1e-12):
                return sp.Integer(round(float(elem)))
            return sp.nsimplify(elem, rational=True, tolerance=1e-10)
        if isinstance(elem, (complex, np.complexfloating)):
            r = elem.real
            i = elem.imag
            r_sp = (
                sp.Integer(round(r))
                if np.isclose(r, round(r), atol=1e-12)
                else sp.nsimplify(r, rational=True, tolerance=1e-10)
            )
            i_sp = (
                sp.Integer(round(i))
                if np.isclose(i, round(i), atol=1e-12)
                else sp.nsimplify(i, rational=True, tolerance=1e-10)
            )
            return r_sp + sp.I * i_sp
        if isinstance(elem, sp.Float):
            return sp.nsimplify(elem, rational=True, tolerance=1e-10)
        if isinstance(elem, sp.Basic):
            return sp.simplify(elem)
        return sp.sympify(elem)

    return mat.applyfunc(_clean_elem)


def _sp_matrix_to_latex(mat: sp.Matrix) -> str:
    """Render a SymPy Matrix into clean LaTeX bmatrix format.

    Parameters
    ----------
    mat : sp.Matrix
        SymPy matrix.

    Returns
    -------
    str
        LaTeX bmatrix string.
    """
    if mat.rows == 0 or mat.cols == 0:
        return r"\begin{bmatrix} \end{bmatrix}"
    rows_tex: list[str] = []
    for i in range(mat.rows):
        row_tex = " & ".join(sp.latex(mat[i, j]) for j in range(mat.cols))
        rows_tex.append(row_tex)
    return r"\begin{bmatrix} " + r" \\ ".join(rows_tex) + r" \end{bmatrix}"


@dataclass
class ModeAnalysis:
    r"""Modal controllability and observability analysis for an individual eigenvalue.

    Attributes
    ----------
    eigenvalue : sp.Expr
        The eigenvalue / pole $\lambda_i$.
    multiplicity : int
        Algebraic multiplicity of $\lambda_i$.
    is_controllable : bool
        True if $\lambda_i$ satisfies the PBH controllability rank test.
    is_observable : bool
        True if $\lambda_i$ satisfies the PBH observability rank test.
    pbh_c_rank : int
        Rank of the PBH controllability matrix $[\lambda_i I - A \quad B]$.
    pbh_o_rank : int
        Rank of the PBH observability matrix $\begin{bmatrix} \lambda_i I - A \\ C \end{bmatrix}$.
    kalman_type : str
        Kalman decomposition classification:
        - ``'co'``: Controllable and Observable
        - ``'c_unobs'``: Controllable and Unobservable
        - ``'unctrl_obs'``: Uncontrollable and Observable
        - ``'unctrl_unobs'``: Uncontrollable and Unobservable
    description : str
        Human-readable explanation of the mode's physical behavior and pole-zero cancellation status.
    """

    eigenvalue: sp.Expr
    multiplicity: int
    is_controllable: bool
    is_observable: bool
    pbh_c_rank: int
    pbh_o_rank: int
    kalman_type: str
    description: str


class CanonicalFormResult:
    r"""Analytical canonical form state-space system and transformation details.

    Represents transformed state-space representations:

    .. math::

        \dot{z} = A_{\text{can}} z + B_{\text{can}} u, \quad y = C_{\text{can}} z + D u

    under the state transformation $x = T z$ ($z = T^{-1} x$).

    Parameters
    ----------
    A : sp.Matrix
        Canonical state matrix $A_{\text{can}} = T^{-1} A T$.
    B : sp.Matrix
        Canonical input matrix $B_{\text{can}} = T^{-1} B$.
    C : sp.Matrix
        Canonical output matrix $C_{\text{can}} = C T$.
    D : sp.Matrix
        Direct feedthrough matrix $D$.
    T : sp.Matrix | None
        Similarity transformation matrix $T$ mapping canonical states $z$ to original states $x = T z$.
    T_inv : sp.Matrix | None
        Inverse transformation matrix $T^{-1}$ mapping original states to canonical states $z = T^{-1} x$.
    form_type : str
        Type of canonical form (``'controllable'``, ``'observable'``, ``'jordan'``).
    is_valid : bool
        Whether the canonical transformation is nonsingular and valid.
    explanation : str
        Analytical derivation and pedagogical notes for this canonical form.
    dt : float | None, default=None
        Sampling time for discrete systems (None for continuous-time).
    """

    def __init__(
        self,
        A: sp.Matrix,
        B: sp.Matrix,
        C: sp.Matrix,
        D: sp.Matrix,
        T: sp.Matrix | None,
        T_inv: sp.Matrix | None,
        form_type: str,
        is_valid: bool = True,
        explanation: str = "",
        dt: float | None = None,
    ) -> None:
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.T = T
        self.T_inv = T_inv
        self.form_type = form_type
        self.is_valid = is_valid
        self.explanation = explanation
        self.dt = dt

    def to_ss(self) -> StateSpace:
        """Convert the symbolic canonical form to a numerical StateSpace model.

        Returns
        -------
        StateSpace
            Numerical StateSpace instance.

        Raises
        ------
        ValueError
            If matrices contain unresolved symbolic variables.
        """
        from ctrlpy.models.state_space import StateSpace

        def _to_np(m: sp.Matrix) -> np.ndarray:
            arr = np.zeros((m.rows, m.cols), dtype=np.float64)
            for i in range(m.rows):
                for j in range(m.cols):
                    val = m[i, j]
                    if hasattr(val, "free_symbols") and val.free_symbols:
                        raise ValueError(
                            f"Cannot convert symbolic expression {val} to numeric float."
                        )
                    arr[i, j] = float(sp.re(val.evalf()))
            return arr

        return StateSpace(_to_np(self.A), _to_np(self.B), _to_np(self.C), _to_np(self.D))

    def _repr_latex_(self) -> str:
        r"""Render LaTeX representation for Jupyter environments."""
        name_map = {
            "controllable": "Controllable Canonical Form (Phase-Variable)",
            "observable": "Observable Canonical Form",
            "jordan": "Jordan / Diagonal Modal Canonical Form",
        }
        title = name_map.get(self.form_type, f"{self.form_type.title()} Canonical Form")
        a_tex = _sp_matrix_to_latex(self.A)
        b_tex = _sp_matrix_to_latex(self.B)
        c_tex = _sp_matrix_to_latex(self.C)
        d_tex = _sp_matrix_to_latex(self.D)

        t_tex = _sp_matrix_to_latex(self.T) if self.T is not None else r"\text{None (Singular)}"
        t_inv_tex = (
            _sp_matrix_to_latex(self.T_inv) if self.T_inv is not None else r"\text{None (Singular)}"
        )

        dt_sym = r"\dot{z}" if self.dt is None else "z[k+1]"
        state_sym = "z" if self.dt is None else "z[k]"
        u_sym = "u" if self.dt is None else "u[k]"
        y_sym = "y" if self.dt is None else "y[k]"

        lines = [
            r"\begin{aligned}",
            rf"\textbf{{{title}:}} \\",
            rf"{dt_sym} &= {a_tex} {state_sym} + {b_tex} {u_sym} \\",
            rf"{y_sym} &= {c_tex} {state_sym} + {d_tex} {u_sym} \\",
            rf"T &= {t_tex}, \quad T^{{-1}} = {t_inv_tex}",
            r"\end{aligned}",
        ]
        return "\n".join(lines)

    def _repr_markdown_(self) -> str:
        """Render Markdown for Jupyter environments."""
        return f"$${self._repr_latex_()}$$"

    def __str__(self) -> str:
        """ASCII representation of canonical form matrices."""
        return (
            f"=== {self.form_type.title()} Canonical Form ===\n"
            f"A =\n{self.A}\n\n"
            f"B =\n{self.B}\n\n"
            f"C =\n{self.C}\n\n"
            f"D =\n{self.D}\n\n"
            f"Transformation Matrix T (x = T z):\n{self.T}\n\n"
            f"Inverse Transformation T_inv (z = T^-1 x):\n{self.T_inv}"
        )

    def __repr__(self) -> str:
        return f"CanonicalFormResult(form_type={self.form_type!r}, is_valid={self.is_valid})"


class StateSpaceTutor:
    r"""Pedagogical state-space canonical forms and controllability/observability tutor.

    Provides rigorous analytical, step-by-step transformations and checks for LTI state-space systems:
    1. **Controllability Analysis**: Symbolically builds $\mathcal{C} = [B \quad AB \dots A^{n-1}B]$,
       computes determinant, rank, and nullspace.
    2. **Observability Analysis**: Symbolically builds $\mathcal{O} = [C^T \quad A^TC^T \dots (A^{n-1})^TC^T]^T$,
       computes determinant, rank, and unobservable subspaces.
    3. **Popov-Belevitch-Hautus (PBH) Mode Decomposition**: Evaluates eigenvalue-by-eigenvalue rank tests
       $\operatorname{rank}[\lambda_i I - A \quad B]$ and $\operatorname{rank}[\lambda_i I - A; C]$ to classify
       all modes into the 4 Kalman canonical subspaces ($\Sigma_{co}, \Sigma_{c\bar{o}}, \Sigma_{\bar{c}o}, \Sigma_{\bar{c}\bar{o}}$).
    4. **Controllable Canonical Form (CCF / Phase-Variable)**: Derives $(A_c, B_c, C_c, D)$ and transformation $T_c = \mathcal{C}\mathcal{W}$.
    5. **Observable Canonical Form (OCF)**: Derives $(A_o, B_o, C_o, D)$ and transformation $T_o = \mathcal{O}^{-1}\mathcal{C}_c$.
    6. **Jordan / Diagonal Modal Canonical Form**: Derives modal matrix $V$ (eigenvectors/generalized eigenvectors) and decoupled state equations.
    7. **Step-by-Step Pedagogical Derivations**: `.explain_steps()` returns an exhaustive mathematical trace.

    Parameters
    ----------
    sys_or_A : Any, optional
        StateSpace model, TransferFunction model, $A$ matrix, or polynomial numerator.
    B : Any, optional
        Input matrix $B \in \mathbb{R}^{n \times m}$.
    C : Any, optional
        Output matrix $C \in \mathbb{R}^{p \times n}$.
    D : Any, optional
        Direct feedthrough matrix $D \in \mathbb{R}^{p \times m}$.
    num : Sequence[Any] | None, default=None
        Transfer function numerator polynomial coefficients if initializing from transfer function.
    den : Sequence[Any] | None, default=None
        Transfer function denominator polynomial coefficients if initializing from transfer function.
    dt : float | None, default=None
        Sampling period for discrete-time systems.

    Attributes
    ----------
    A : sp.Matrix
        State transition matrix ($n \times n$).
    B : sp.Matrix
        Input matrix ($n \times m$).
    C : sp.Matrix
        Output matrix ($p \times n$).
    D : sp.Matrix
        Direct transmission matrix ($p \times m$).
    n_states : int
        Number of state variables $n$.
    n_inputs : int
        Number of inputs $m$.
    n_outputs : int
        Number of outputs $p$.
    dt : float | None
        Sampling period (None for continuous-time).
    controllability_matrix : sp.Matrix
        The Kalman controllability matrix $\mathcal{C}$.
    observability_matrix : sp.Matrix
        The Kalman observability matrix $\mathcal{O}$.
    controllability_rank : int
        Rank of $\mathcal{C}$.
    observability_rank : int
        Rank of $\mathcal{O}$.
    is_controllable : bool
        Whether the system is fully controllable ($\operatorname{rank}(\mathcal{C}) = n$).
    is_observable : bool
        Whether the system is fully observable ($\operatorname{rank}(\mathcal{O}) = n$).
    characteristic_polynomial : sp.Poly
        Symbolic characteristic polynomial $p(s) = \det(sI - A)$.
    eigenvalues : list[sp.Expr]
        List of all eigenvalues $\lambda_i$.
    modes : list[ModeAnalysis]
        Detailed PBH and Kalman modal classification for each eigenvalue.
    uncontrollable_modes : list[sp.Expr]
        List of eigenvalues that fail PBH controllability.
    unobservable_modes : list[sp.Expr]
        List of eigenvalues that fail PBH observability.
    transfer_function : sp.Expr
        Analytical transfer function $G(s) = C(sI - A)^{-1}B + D$.
    ccf : CanonicalFormResult
        Controllable Canonical Form transformation result.
    ocf : CanonicalFormResult
        Observable Canonical Form transformation result.
    jcf : CanonicalFormResult
        Jordan / Diagonal Canonical Form transformation result.
    steps : list[str]
        Pedagogical step-by-step mathematical explanations.
    """

    def __init__(
        self,
        sys_or_A: Any = None,
        B: Any = None,
        C: Any = None,
        D: Any = None,
        *,
        num: Sequence[Any] | None = None,
        den: Sequence[Any] | None = None,
        dt: float | None = None,
    ) -> None:
        self.dt = dt
        steps: list[str] = []

        # 1. Parse Inputs into SymPy Matrices
        A_sp, B_sp, C_sp, D_sp, inferred_dt = self._parse_inputs(sys_or_A, B, C, D, num, den)
        if self.dt is None:
            self.dt = inferred_dt

        self.A = A_sp
        self.B = B_sp
        self.C = C_sp
        self.D = D_sp

        n = self.A.rows
        m = self.B.cols
        p = self.C.rows

        self.n_states = n
        self.n_inputs = m
        self.n_outputs = p

        sym_var = sp.Symbol("s") if self.dt is None else sp.Symbol("z")
        var_str = "s" if self.dt is None else "z"

        # 2. Compute Characteristic Polynomial and Eigenvalues
        sI_minus_A = sym_var * sp.eye(n) - self.A
        char_poly_expr = sp.simplify(sI_minus_A.det())
        char_poly = sp.Poly(char_poly_expr, sym_var)
        self.char_poly_expr = char_poly_expr
        self.characteristic_polynomial = char_poly

        # Monic polynomial coefficients
        coeffs = char_poly.all_coeffs()
        lead = coeffs[0]
        monic_coeffs = [sp.simplify(c / lead) for c in coeffs]
        # p(s) = s^n + a_{n-1} s^{n-1} + ... + a_1 s + a_0
        # monic_coeffs has length n+1: [1, a_{n-1}, a_{n-2}, ..., a_0]
        a_coeffs = monic_coeffs[1:]  # [a_{n-1}, a_{n-2}, ..., a_0]
        # a_dict: a_dict[0] = a_0, a_dict[1] = a_1, ..., a_dict[n-1] = a_{n-1}
        a_dict: dict[int, sp.Expr] = {}
        for idx, coef in enumerate(reversed(a_coeffs)):
            a_dict[idx] = coef

        eigen_dict = self.A.eigenvals()
        eigenvalues: list[sp.Expr] = []
        eigenvalue_multiplicities: dict[sp.Expr, int] = {}
        for ev, mult in eigen_dict.items():
            eigenvalues.append(ev)
            eigenvalue_multiplicities[ev] = mult
        self.eigenvalues = eigenvalues
        self.eigenvalue_multiplicities = eigenvalue_multiplicities

        # 3. Compute Controllability Matrix
        ctrl_cols: list[sp.Matrix] = []
        intermediate_ab: list[sp.Matrix] = []
        for k in range(n):
            a_pow = self.A**k
            ab_k = a_pow * self.B
            intermediate_ab.append(ab_k)
            ctrl_cols.append(ab_k)

        self.controllability_matrix = sp.Matrix.hstack(*ctrl_cols) if ctrl_cols else sp.zeros(n, 0)
        self.controllability_rank = int(self.controllability_matrix.rank())
        self.is_controllable = bool(self.controllability_rank == n)

        # 4. Compute Observability Matrix
        obs_rows: list[sp.Matrix] = []
        intermediate_ca: list[sp.Matrix] = []
        for k in range(n):
            a_pow = self.A**k
            ca_k = self.C * a_pow
            intermediate_ca.append(ca_k)
            obs_rows.append(ca_k)

        self.observability_matrix = sp.Matrix.vstack(*obs_rows) if obs_rows else sp.zeros(0, n)
        self.observability_rank = int(self.observability_matrix.rank())
        self.is_observable = bool(self.observability_rank == n)

        # 5. Compute Analytical Transfer Function
        try:
            adj_matrix = sI_minus_A.adjugate()
            tf_matrix = sp.simplify(
                (self.C * adj_matrix * self.B + self.D * char_poly_expr) / char_poly_expr
            )
            if self.is_siso:
                self.transfer_function = sp.simplify(tf_matrix[0, 0])
                num_expr, den_expr = sp.fraction(self.transfer_function)
                self.tf_numerator = sp.Poly(num_expr, sym_var)
                self.tf_denominator = sp.Poly(den_expr, sym_var)
            else:
                self.transfer_function = tf_matrix
                self.tf_numerator = None
                self.tf_denominator = None
        except (TypeError, ValueError, AttributeError, sp.SympifyError, sp.MatrixError):
            self.transfer_function = sp.sympify(0)
            self.tf_numerator = None
            self.tf_denominator = None

        # 6. PBH Eigenvalue Mode Analysis
        modes: list[ModeAnalysis] = []
        uncontrollable_modes: list[sp.Expr] = []
        unobservable_modes: list[sp.Expr] = []

        for ev in eigenvalues:
            mult = eigenvalue_multiplicities[ev]
            # Controllability PBH: [ev * I - A, B]
            pbh_c_mat = sp.Matrix.hstack(ev * sp.eye(n) - self.A, self.B)
            c_rank = int(pbh_c_mat.rank())
            ev_ctrl = bool(c_rank == n)

            # Observability PBH: [ev * I - A; C]
            pbh_o_mat = sp.Matrix.vstack(ev * sp.eye(n) - self.A, self.C)
            o_rank = int(pbh_o_mat.rank())
            ev_obs = bool(o_rank == n)

            if not ev_ctrl:
                uncontrollable_modes.append(ev)
            if not ev_obs:
                unobservable_modes.append(ev)

            if ev_ctrl and ev_obs:
                k_type = "co"
                desc = f"Mode λ = {sp.latex(ev)} is both Controllable and Observable (present in transfer function)."
            elif ev_ctrl and not ev_obs:
                k_type = "c_unobs"
                desc = f"Mode λ = {sp.latex(ev)} is Controllable but Unobservable (actuated by input, hidden from output / pole-zero cancelled)."
            elif not ev_ctrl and ev_obs:
                k_type = "unctrl_obs"
                desc = f"Mode λ = {sp.latex(ev)} is Uncontrollable but Observable (visible in output, unaffected by input / pole-zero cancelled)."
            else:
                k_type = "unctrl_unobs"
                desc = f"Mode λ = {sp.latex(ev)} is Uncontrollable and Unobservable (completely decoupled internal dynamic)."

            modes.append(
                ModeAnalysis(
                    eigenvalue=ev,
                    multiplicity=mult,
                    is_controllable=ev_ctrl,
                    is_observable=ev_obs,
                    pbh_c_rank=c_rank,
                    pbh_o_rank=o_rank,
                    kalman_type=k_type,
                    description=desc,
                )
            )

        self.modes = modes
        self.uncontrollable_modes = uncontrollable_modes
        self.unobservable_modes = unobservable_modes

        # 7. Canonical Form Transformations (SISO Companion Forms & Jordan Modal Form)
        self.ccf = self._derive_ccf(a_dict, sym_var)
        self.ocf = self._derive_ocf(a_dict, sym_var)
        self.jcf = self._derive_jcf()

        # 8. Compile Step-by-Step Pedagogical Derivations
        self._build_pedagogical_steps(steps, var_str, a_dict, intermediate_ab, intermediate_ca)
        self.steps = steps

    @property
    def is_siso(self) -> bool:
        """Return True if system is Single-Input Single-Output."""
        return bool(self.n_inputs == 1 and self.n_outputs == 1)

    @staticmethod
    def _parse_inputs(
        sys_or_A: Any,
        B: Any,
        C: Any,
        D: Any,
        num: Sequence[Any] | None,
        den: Sequence[Any] | None,
    ) -> tuple[sp.Matrix, sp.Matrix, sp.Matrix, sp.Matrix, float | None]:
        """Parse various system representations into standard SymPy (A, B, C, D) matrices."""
        inferred_dt: float | None = None

        # Case 1: Transfer function numerator / denominator sequences provided
        if num is not None and den is not None:
            return StateSpaceTutor._from_tf_coeffs(num, den, dt=inferred_dt)

        # Case 2: TransferFunction model instance
        if hasattr(sys_or_A, "num") and hasattr(sys_or_A, "den"):
            inferred_dt = getattr(sys_or_A, "dt", None)
            return StateSpaceTutor._from_tf_coeffs(sys_or_A.num, sys_or_A.den, dt=inferred_dt)

        # Case 3: StateSpace model instance
        if (
            hasattr(sys_or_A, "A")
            and hasattr(sys_or_A, "B")
            and hasattr(sys_or_A, "C")
            and hasattr(sys_or_A, "D")
        ):
            inferred_dt = getattr(sys_or_A, "dt", None)
            return (
                _to_sympy_matrix(sys_or_A.A),
                _to_sympy_matrix(sys_or_A.B),
                _to_sympy_matrix(sys_or_A.C),
                _to_sympy_matrix(sys_or_A.D),
                inferred_dt,
            )

        # Case 4: Explicit matrices A, B, C, D
        if sys_or_A is not None:
            a_sp = _to_sympy_matrix(sys_or_A)
            if a_sp.rows != a_sp.cols:
                raise DimensionMismatchError(
                    f"Matrix A must be square, got shape ({a_sp.rows}, {a_sp.cols})."
                )
            n = a_sp.rows

            if B is None:
                raise ValueError("Matrix B must be provided when matrix A is given.")
            b_sp = _to_sympy_matrix(B)
            if b_sp.rows != n:
                if b_sp.cols == n and b_sp.rows == 1:
                    b_sp = b_sp.T
                elif b_sp.rows == 1 and n == 1:
                    pass
                else:
                    raise DimensionMismatchError(
                        f"Matrix B rows ({b_sp.rows}) must match matrix A dimension ({n})."
                    )

            if C is None:
                raise ValueError("Matrix C must be provided when matrix A is given.")
            c_sp = _to_sympy_matrix(C)
            if c_sp.cols != n:
                if c_sp.rows == n and c_sp.cols == 1:
                    c_sp = c_sp.T
                elif c_sp.cols == 1 and n == 1:
                    pass
                else:
                    raise DimensionMismatchError(
                        f"Matrix C columns ({c_sp.cols}) must match matrix A dimension ({n})."
                    )

            m = b_sp.cols
            p = c_sp.rows

            if D is None:
                d_sp = sp.zeros(p, m)
            else:
                d_sp = _to_sympy_matrix(D)
                if d_sp.rows != p or d_sp.cols != m:
                    if d_sp.rows == 1 and d_sp.cols == 1 and (p != 1 or m != 1):
                        d_sp = sp.Matrix(
                            [[d_sp[0, 0] if i == j else 0 for j in range(m)] for i in range(p)]
                        )
                    else:
                        raise DimensionMismatchError(
                            f"Matrix D shape ({d_sp.rows}, {d_sp.cols}) must match outputs {p} and inputs {m}."
                        )

            return a_sp, b_sp, c_sp, d_sp, inferred_dt

        raise ValueError(
            "Must provide either a StateSpace/TransferFunction model or explicit A, B, C, D matrices."
        )

    @staticmethod
    def _from_tf_coeffs(
        num: Sequence[Any],
        den: Sequence[Any],
        dt: float | None = None,
    ) -> tuple[sp.Matrix, sp.Matrix, sp.Matrix, sp.Matrix, float | None]:
        """Derive standard controllable companion state-space matrices from transfer function coefficients."""
        s = sp.Symbol("s")
        clean_num = [_to_sympy_matrix(x)[0, 0] for x in num]
        clean_den = [_to_sympy_matrix(x)[0, 0] for x in den]

        # Build polynomials
        num_poly = sp.Poly(clean_num, s)
        den_poly = sp.Poly(clean_den, s)

        den_coeffs = den_poly.all_coeffs()
        lead = den_coeffs[0]
        monic_den_coeffs = [sp.simplify(c / lead) for c in den_coeffs]
        monic_num_coeffs = [sp.simplify(c / lead) for c in num_poly.all_coeffs()]

        n = len(monic_den_coeffs) - 1
        if n <= 0:
            # Scalar gain
            d_val = monic_num_coeffs[0] if monic_num_coeffs else sp.Integer(0)
            return sp.Matrix([[0]]), sp.Matrix([[0]]), sp.Matrix([[0]]), sp.Matrix([[d_val]]), dt

        # Pad numerator to length n + 1
        num_padded = [sp.Integer(0)] * (n + 1 - len(monic_num_coeffs)) + monic_num_coeffs
        d_val = num_padded[0]

        # Strictly proper numerator: N'(s) = N(s) - D * Den(s)
        beta_coeffs = [
            sp.simplify(num_padded[i] - d_val * monic_den_coeffs[i]) for i in range(1, n + 1)
        ]
        # beta_coeffs: [beta_{n-1}, beta_{n-2}, ..., beta_0]

        # CCF Companion matrix A:
        # [ 0, 1, 0, ..., 0 ]
        # [ 0, 0, 1, ..., 0 ]
        # [ -a_0, -a_1, ..., -a_{n-1} ]
        a_den = list(reversed(monic_den_coeffs[1:]))  # [a_0, a_1, ..., a_{n-1}]
        a_rows: list[list[sp.Expr]] = []
        for i in range(n - 1):
            row = [sp.Integer(0)] * n
            row[i + 1] = sp.Integer(1)
            a_rows.append(row)
        a_rows.append([-a_den[i] for i in range(n)])

        A_sp = _to_sympy_matrix(sp.Matrix(a_rows))
        B_sp = sp.zeros(n, 1)
        B_sp[n - 1, 0] = sp.Integer(1)

        # C = [beta_0, beta_1, ..., beta_{n-1}]
        c_entries = list(reversed(beta_coeffs))  # [beta_0, beta_1, ..., beta_{n-1}]
        C_sp = _to_sympy_matrix(sp.Matrix([c_entries]))
        D_sp = _to_sympy_matrix(sp.Matrix([[d_val]]))

        return A_sp, B_sp, C_sp, D_sp, dt

    def _derive_ccf(self, a_dict: dict[int, sp.Expr], sym_var: sp.Symbol) -> CanonicalFormResult:
        """Derive Controllable Canonical Form (Phase-Variable Form) and transformation matrix."""
        n = self.n_states
        if not self.is_siso:
            return CanonicalFormResult(
                self.A,
                self.B,
                self.C,
                self.D,
                None,
                None,
                form_type="controllable",
                is_valid=False,
                explanation="Controllable Canonical Form (Phase-Variable) is defined for SISO systems.",
                dt=self.dt,
            )

        # Build companion matrices (Ac, Bc)
        # Ac has 1s on superdiagonal and bottom row [-a_0, -a_1, ..., -a_{n-1}]
        a_rows: list[list[sp.Expr]] = []
        for i in range(n - 1):
            row = [sp.Integer(0)] * n
            row[i + 1] = sp.Integer(1)
            a_rows.append(row)
        a_rows.append([-a_dict[i] for i in range(n)])
        Ac = sp.Matrix(a_rows)

        Bc = sp.zeros(n, 1)
        Bc[n - 1, 0] = sp.Integer(1)

        # Compute strictly proper transfer function numerator coefficients beta_0 ... beta_{n-1}
        # G(s) = C(sI - A)^-1 B + D
        tf_strictly_proper = sp.simplify(self.transfer_function - self.D[0, 0])
        # Multiply strictly proper transfer function by characteristic polynomial to obtain full uncancelled numerator
        num_uncancelled = sp.simplify(tf_strictly_proper * self.char_poly_expr)
        num_poly = sp.Poly(num_uncancelled, sym_var)
        num_coeffs = num_poly.all_coeffs()
        padded_num = [sp.Integer(0)] * (n - len(num_coeffs)) + num_coeffs
        # padded_num is [beta_{n-1}, beta_{n-2}, ..., beta_0]
        c_entries = list(reversed(padded_num))  # [beta_0, beta_1, ..., beta_{n-1}]
        Cc = sp.Matrix([c_entries])
        Dc = self.D

        if not self.is_controllable:
            return CanonicalFormResult(
                Ac,
                Bc,
                Cc,
                Dc,
                None,
                None,
                form_type="controllable",
                is_valid=False,
                explanation=(
                    f"System is uncontrollable (rank(C) = {self.controllability_rank} < {n}). "
                    "A nonsingular similarity transformation to Controllable Canonical Form does not exist."
                ),
                dt=self.dt,
            )

        # Compute controllability matrix of companion form: Cc_mat = [Bc, Ac*Bc, ..., Ac^(n-1)*Bc]
        cols_cc: list[sp.Matrix] = []
        for k in range(n):
            cols_cc.append((Ac**k) * Bc)
        Cc_mat = sp.Matrix.hstack(*cols_cc)

        # Transformation matrix: Tc = C * Cc_mat^-1 = C * W
        Cc_inv = Cc_mat.inv()
        Tc = sp.simplify(self.controllability_matrix * Cc_inv)
        Tc_inv = sp.simplify(Tc.inv())

        return CanonicalFormResult(
            Ac,
            Bc,
            Cc,
            Dc,
            Tc,
            Tc_inv,
            form_type="controllable",
            is_valid=True,
            explanation=(
                "Controllable Canonical Form (Phase-Variable form) derived via similarity transformation "
                r"$x = T_c z_c$, where $T_c = \mathcal{C} \mathcal{C}_c^{-1} = \mathcal{C} \mathcal{W}$."
            ),
            dt=self.dt,
        )

    def _derive_ocf(self, a_dict: dict[int, sp.Expr], sym_var: sp.Symbol) -> CanonicalFormResult:
        """Derive Observable Canonical Form and transformation matrix."""
        n = self.n_states
        if not self.is_siso:
            return CanonicalFormResult(
                self.A,
                self.B,
                self.C,
                self.D,
                None,
                None,
                form_type="observable",
                is_valid=False,
                explanation="Observable Canonical Form is defined for SISO systems.",
                dt=self.dt,
            )

        # Companion Ao = Ac^T
        # Rows:
        # [0, 0, ..., -a_0]
        # [1, 0, ..., -a_1]
        # [0, 1, ..., -a_2]
        # [0, 0, ..., -a_{n-1}]
        ao_rows: list[list[sp.Expr]] = []
        for i in range(n):
            row = [sp.Integer(0)] * n
            if i > 0:
                row[i - 1] = sp.Integer(1)
            row[n - 1] = -a_dict[i]
            ao_rows.append(row)
        Ao = sp.Matrix(ao_rows)

        # Strictly proper numerator coefficients
        tf_strictly_proper = sp.simplify(self.transfer_function - self.D[0, 0])
        num_uncancelled = sp.simplify(tf_strictly_proper * self.char_poly_expr)
        num_poly = sp.Poly(num_uncancelled, sym_var)
        num_coeffs = num_poly.all_coeffs()
        padded_num = [sp.Integer(0)] * (n - len(num_coeffs)) + num_coeffs
        # [beta_{n-1}, ..., beta_0] -> Bo = [beta_0, beta_1, ..., beta_{n-1}]^T
        b_entries = list(reversed(padded_num))
        Bo = sp.Matrix([[b] for b in b_entries])

        # Co = [0, 0, ..., 1]
        Co = sp.zeros(1, n)
        Co[0, n - 1] = sp.Integer(1)
        Do = self.D

        if not self.is_observable:
            return CanonicalFormResult(
                Ao,
                Bo,
                Co,
                Do,
                None,
                None,
                form_type="observable",
                is_valid=False,
                explanation=(
                    f"System is unobservable (rank(O) = {self.observability_rank} < {n}). "
                    "A nonsingular similarity transformation to Observable Canonical Form does not exist."
                ),
                dt=self.dt,
            )

        # Observability matrix of companion form: Oo_mat = [Co; Co*Ao; ...; Co*Ao^(n-1)]
        rows_oo: list[sp.Matrix] = []
        for k in range(n):
            rows_oo.append(Co * (Ao**k))
        Oo_mat = sp.Matrix.vstack(*rows_oo)

        # Transformation matrix: O * To = Oo_mat => To = O^-1 * Oo_mat
        O_inv = self.observability_matrix.inv()
        To = sp.simplify(O_inv * Oo_mat)
        To_inv = sp.simplify(To.inv())

        return CanonicalFormResult(
            Ao,
            Bo,
            Co,
            Do,
            To,
            To_inv,
            form_type="observable",
            is_valid=True,
            explanation=(
                "Observable Canonical Form derived via similarity transformation "
                r"$x = T_o z_o$, where $T_o = \mathcal{O}^{-1} \mathcal{O}_o$."
            ),
            dt=self.dt,
        )

    def _derive_jcf(self) -> CanonicalFormResult:
        """Derive Diagonal / Jordan Canonical Form and modal transformation matrix."""
        try:
            P, J = self.A.jordan_form()
            P = sp.simplify(P)
            J = sp.simplify(J)
            P_inv = sp.simplify(P.inv())

            Ad = J
            Bd = sp.simplify(P_inv * self.B)
            Cd = sp.simplify(self.C * P)
            Dd = self.D

            return CanonicalFormResult(
                Ad,
                Bd,
                Cd,
                Dd,
                P,
                P_inv,
                form_type="jordan",
                is_valid=True,
                explanation=(
                    "Jordan / Diagonal Canonical Form derived via modal transformation "
                    r"$x = V z_d$, where columns of $V$ are eigenvectors and generalized eigenvectors."
                ),
                dt=self.dt,
            )
        except (TypeError, ValueError, AttributeError, sp.SympifyError, sp.MatrixError) as err:
            return CanonicalFormResult(
                self.A,
                self.B,
                self.C,
                self.D,
                None,
                None,
                form_type="jordan",
                is_valid=False,
                explanation=f"Failed to compute Jordan canonical form: {err}",
                dt=self.dt,
            )

    def _build_pedagogical_steps(
        self,
        steps: list[str],
        var_str: str,
        a_dict: dict[int, sp.Expr],
        intermediate_ab: list[sp.Matrix],
        intermediate_ca: list[sp.Matrix],
    ) -> None:
        """Assemble structured LaTeX/Markdown explanations for all derivation stages."""
        n = self.n_states
        # 1. System Representation & Characteristic Polynomial
        poly_str = " + ".join(
            f"{sp.latex(a_dict[i])}{var_str}^{{{i}}}"
            if i > 1
            else (f"{sp.latex(a_dict[i])}{var_str}" if i == 1 else f"{sp.latex(a_dict[i])}")
            for i in reversed(range(n))
            if a_dict[i] != 0
        )
        if poly_str:
            poly_full = f"{var_str}^{{{n}}} + {poly_str}"
        else:
            poly_full = f"{var_str}^{{{n}}}"

        ev_items = [
            f"{sp.latex(ev)} \\text{{ (mult: {mult})}}"
            for ev, mult in self.eigenvalue_multiplicities.items()
        ]
        ev_str = ", ".join(ev_items)

        steps.append(
            f"### Step 1: System Definition & Characteristic Polynomial\n"
            f"The continuous/discrete state-space matrices are defined as:\n\n"
            f"$$A = {_sp_matrix_to_latex(self.A)}, \\quad B = {_sp_matrix_to_latex(self.B)}, "
            f"\\quad C = {_sp_matrix_to_latex(self.C)}, \\quad D = {_sp_matrix_to_latex(self.D)}$$\n\n"
            f"The characteristic polynomial is obtained by evaluating $p({var_str}) = \\det({var_str} I - A)$:\n\n"
            f"$$p({var_str}) = \\det\\left({var_str} I - A\\right) = {sp.latex(self.char_poly_expr)} = {poly_full} = 0$$\n\n"
            f"The open-loop eigenvalues (system poles) are:\n\n"
            f"$$\\lambda = \\left\\{{ {ev_str} \\right\\}}$$"
        )

        # 2. Controllability Derivation
        ab_terms = " \\quad ".join(
            rf"A^{k} B = {_sp_matrix_to_latex(mat)}" for k, mat in enumerate(intermediate_ab)
        )
        ctrl_det_str = (
            rf"\det(\mathcal{{C}}) = {sp.latex(self.controllability_matrix.det())}"
            if self.controllability_matrix.rows == self.controllability_matrix.cols
            else r"\text{N/A (Non-square matrix)}"
        )
        if self.is_controllable:
            ctrl_verdict = (
                r"The system is **fully controllable** ($\operatorname{rank}(\mathcal{C}) = n$). "
                r"All state trajectories can be steered arbitrarily in finite time."
            )
        else:
            ctrl_verdict = (
                rf"The system is **uncontrollable** ($\operatorname{{rank}}(\mathcal{{C}}) = {self.controllability_rank} < {n}$). "
                rf"A {n - self.controllability_rank}-dimensional uncontrollable subspace exists."
            )

        steps.append(
            f"### Step 2: Controllability Matrix & Kalman Rank Test\n"
            f"The Kalman controllability matrix $\\mathcal{{C}} \\in \\mathbb{{R}}^{{{n} \\times {self.controllability_matrix.cols}}}$ "
            f"is constructed from successive matrix powers $A^k B$:\n\n"
            f"$$\\mathcal{{C}} = \\begin{{bmatrix}} B & AB & A^2 B & \\dots & A^{{{n - 1}}} B \\end{{bmatrix}}$$\n\n"
            f"Evaluating individual block columns:\n\n"
            f"$${ab_terms}$$\n\n"
            f"Concatenating yields the full controllability matrix:\n\n"
            f"$$\\mathcal{{C}} = {_sp_matrix_to_latex(self.controllability_matrix)}$$\n\n"
            f"Determinant and rank evaluation:\n\n"
            f"$${ctrl_det_str}, \\quad \\operatorname{{rank}}(\\mathcal{{C}}) = {self.controllability_rank} \\quad (n = {n})$$\n\n"
            f"**Controllability Verdict:** {ctrl_verdict}"
        )

        # 3. Observability Derivation
        ca_terms = " \\quad ".join(
            rf"C A^{k} = {_sp_matrix_to_latex(mat)}" for k, mat in enumerate(intermediate_ca)
        )
        obs_det_str = (
            rf"\det(\mathcal{{O}}) = {sp.latex(self.observability_matrix.det())}"
            if self.observability_matrix.rows == self.observability_matrix.cols
            else r"\text{N/A (Non-square matrix)}"
        )

        if self.is_observable:
            obs_verdict = (
                r"The system is **fully observable** ($\operatorname{rank}(\mathcal{O}) = n$). "
                r"The initial state $x(0)$ can be uniquely reconstructed from output history $y(t)$."
            )
        else:
            obs_verdict = (
                rf"The system is **unobservable** ($\operatorname{{rank}}(\mathcal{{O}}) = {self.observability_rank} < {n}$). "
                rf"A {n - self.observability_rank}-dimensional unobservable subspace exists."
            )

        steps.append(
            f"### Step 3: Observability Matrix & Kalman Rank Test\n"
            f"The Kalman observability matrix $\\mathcal{{O}} \\in \\mathbb{{R}}^{{{self.observability_matrix.rows} \\times {n}}}$ "
            f"is constructed from successive output matrix products $C A^k$:\n\n"
            f"$$\\mathcal{{O}} = \\begin{{bmatrix}} C \\\\ CA \\\\ CA^2 \\\\ \\vdots \\\\ CA^{{{n - 1}}} \\end{{bmatrix}}$$\n\n"
            f"Evaluating individual block rows:\n\n"
            f"$${ca_terms}$$\n\n"
            f"Concatenating yields the full observability matrix:\n\n"
            f"$$\\mathcal{{O}} = {_sp_matrix_to_latex(self.observability_matrix)}$$\n\n"
            f"Determinant and rank evaluation:\n\n"
            f"$${obs_det_str}, \\quad \\operatorname{{rank}}(\\mathcal{{O}}) = {self.observability_rank} \\quad (n = {n})$$\n\n"
            f"**Observability Verdict:** {obs_verdict}"
        )

        # 4. PBH Eigenvalue Test & Mode Decomposition
        mode_rows: list[str] = []
        for m_obj in self.modes:
            mode_rows.append(
                f"- **Mode $\\lambda = {sp.latex(m_obj.eigenvalue)}$** (Algebraic Mult: {m_obj.multiplicity}):\n"
                f"  - Controllability PBH: $\\operatorname{{rank}}\\begin{{bmatrix}} \\lambda I - A & B \\end{{bmatrix}} = {m_obj.pbh_c_rank} / {n} "
                f"\\implies \\textbf{{{('Controllable' if m_obj.is_controllable else 'Uncontrollable')}}}$\n"
                f"  - Observability PBH: $\\operatorname{{rank}}\\begin{{bmatrix}} \\lambda I - A \\\\ C \\end{{bmatrix}} = {m_obj.pbh_o_rank} / {n} "
                f"\\implies \\textbf{{{('Observable' if m_obj.is_observable else 'Unobservable')}}}$\n"
                f"  - Kalman Subspace Classification: `{m_obj.kalman_type.upper()}` — {m_obj.description}"
            )
        mode_text = "\n".join(mode_rows)

        steps.append(
            f"### Step 4: Popov-Belevitch-Hautus (PBH) Mode Decomposition\n"
            f"The PBH rank tests check the modal controllability and observability of each eigenvalue $\\lambda_i$:\n\n"
            f"$$\\operatorname{{rank}}\\begin{{bmatrix}} \\lambda_i I - A & B \\end{{bmatrix}} = n \\iff \\text{{Mode }} \\lambda_i \\text{{ Controllable}}$$\n"
            f"$$\\operatorname{{rank}}\\begin{{bmatrix}} \\lambda_i I - A \\\\ C \\end{{bmatrix}} = n \\iff \\text{{Mode }} \\lambda_i \\text{{ Observable}}$$\n\n"
            f"{mode_text}"
        )

        # 5. Controllable Canonical Form
        if self.ccf.is_valid and self.ccf.T is not None:
            steps.append(
                f"### Step 5: Controllable Canonical Form (Phase-Variable)\n"
                f"The phase-variable Controllable Canonical Form $(A_c, B_c, C_c, D)$ is given by:\n\n"
                f"$$A_c = {_sp_matrix_to_latex(self.ccf.A)}, \\quad B_c = {_sp_matrix_to_latex(self.ccf.B)}, "
                f"\\quad C_c = {_sp_matrix_to_latex(self.ccf.C)}, \\quad D = {_sp_matrix_to_latex(self.ccf.D)}$$\n\n"
                f"Transformation matrix $T_c$ ($x = T_c z_c$) and inverse $T_c^{{-1}}$ ($z_c = T_c^{{-1}} x$):\n\n"
                f"$$T_c = \\mathcal{{C}} \\mathcal{{C}}_c^{{-1}} = {_sp_matrix_to_latex(self.ccf.T)}, \\quad "
                f"T_c^{{-1}} = {_sp_matrix_to_latex(self.ccf.T_inv)}$$\n\n"
                f"Verification: $T_c^{{-1}} A T_c = A_c$, $T_c^{{-1}} B = B_c$, $C T_c = C_c$."
            )
        else:
            steps.append(
                f"### Step 5: Controllable Canonical Form (Phase-Variable)\n{self.ccf.explanation}"
            )

        # 6. Observable Canonical Form
        if self.ocf.is_valid and self.ocf.T is not None:
            steps.append(
                f"### Step 6: Observable Canonical Form\n"
                f"The Observable Canonical Form $(A_o, B_o, C_o, D)$ is dual to the controllable form:\n\n"
                f"$$A_o = {_sp_matrix_to_latex(self.ocf.A)}, \\quad B_o = {_sp_matrix_to_latex(self.ocf.B)}, "
                f"\\quad C_o = {_sp_matrix_to_latex(self.ocf.C)}, \\quad D = {_sp_matrix_to_latex(self.ocf.D)}$$\n\n"
                f"Transformation matrix $T_o$ ($x = T_o z_o$) and inverse $T_o^{{-1}}$ ($z_o = T_o^{{-1}} x$):\n\n"
                f"$$T_o = \\mathcal{{O}}^{{-1}} \\mathcal{{O}}_o = {_sp_matrix_to_latex(self.ocf.T)}, \\quad "
                f"T_o^{{-1}} = {_sp_matrix_to_latex(self.ocf.T_inv)}$$\n\n"
                f"Verification: $T_o^{{-1}} A T_o = A_o$, $T_o^{{-1}} B = B_o$, $C T_o = C_o$."
            )
        else:
            steps.append(f"### Step 6: Observable Canonical Form\n{self.ocf.explanation}")

        # 7. Jordan Canonical Form
        if self.jcf.is_valid and self.jcf.T is not None:
            steps.append(
                f"### Step 7: Jordan / Diagonal Modal Form\n"
                f"The modal transformation matrix $V$ diagonalizes the state matrix $A$ into Jordan form:\n\n"
                f"$$A_d = {_sp_matrix_to_latex(self.jcf.A)}, \\quad B_d = {_sp_matrix_to_latex(self.jcf.B)}, "
                f"\\quad C_d = {_sp_matrix_to_latex(self.jcf.C)}, \\quad D = {_sp_matrix_to_latex(self.jcf.D)}$$\n\n"
                f"Modal eigenvector matrix $V$ ($x = V z_d$) and inverse $V^{{-1}}$ ($z_d = V^{{-1}} x$):\n\n"
                f"$$V = {_sp_matrix_to_latex(self.jcf.T)}, \\quad V^{{-1}} = {_sp_matrix_to_latex(self.jcf.T_inv)}$$\n\n"
                f"Decoupled modal equations: $\\dot{{z}}_{{d,i}} = \\lambda_i z_{{d,i}} + b_{{d,i}} u$, $y = \\sum c_{{d,i}} z_{{d,i}} + D u$."
            )
        else:
            steps.append(f"### Step 7: Jordan / Diagonal Modal Form\n{self.jcf.explanation}")

        # 8. Transfer Function & Cancellations
        if self.is_siso:
            steps.append(
                f"### Step 8: Analytical Transfer Function & Pole-Zero Cancellation\n"
                f"The input-output transfer function $G({var_str}) = C({var_str} I - A)^{{-1}} B + D$ is:\n\n"
                f"$$G({var_str}) = {sp.latex(self.transfer_function)}$$\n\n"
                f"- Characteristic polynomial order: $n = {n}$\n"
                f"- Transfer function minimal denominator order: $\\deg(D({var_str})) = {self.tf_denominator.degree() if self.tf_denominator is not None else n}$\n"
                f"- Uncontrollable poles: {[sp.latex(p) for p in self.uncontrollable_modes] or 'None'}\n"
                f"- Unobservable poles: {[sp.latex(p) for p in self.unobservable_modes] or 'None'}\n"
                f"- Poles cancelled in transfer function: {f'Yes ({n - (self.tf_denominator.degree() if self.tf_denominator is not None else n)} cancelled mode(s))' if len(self.uncontrollable_modes) > 0 or len(self.unobservable_modes) > 0 else 'None (Minimal Realization)'}"
            )

    def explain_steps(self) -> list[str]:
        """Return the step-by-step list of Markdown/LaTeX explanations.

        Returns
        -------
        list[str]
            List of formatted derivation steps.
        """
        return list(self.steps)

    def controllable_canonical_form(self) -> CanonicalFormResult:
        """Return the Controllable Canonical Form (Phase-Variable) result.

        Returns
        -------
        CanonicalFormResult
            CCF state matrices, transformation matrix, and derivation status.
        """
        return self.ccf

    def observable_canonical_form(self) -> CanonicalFormResult:
        """Return the Observable Canonical Form result.

        Returns
        -------
        CanonicalFormResult
            OCF state matrices, transformation matrix, and derivation status.
        """
        return self.ocf

    def jordan_canonical_form(self) -> CanonicalFormResult:
        """Return the Jordan / Diagonal Canonical Form result.

        Returns
        -------
        CanonicalFormResult
            Jordan/diagonal modal state matrices, modal matrix V, and derivation status.
        """
        return self.jcf

    def pbh_test(self) -> list[dict[str, Any]]:
        """Return summary of PBH controllability and observability test for each eigenvalue.

        Returns
        -------
        list[dict[str, Any]]
            List of dictionaries containing eigenvalue details, PBH ranks, and Kalman types.
        """
        return [
            {
                "eigenvalue": m.eigenvalue,
                "multiplicity": m.multiplicity,
                "is_controllable": m.is_controllable,
                "is_observable": m.is_observable,
                "pbh_controllability_rank": m.pbh_c_rank,
                "pbh_observability_rank": m.pbh_o_rank,
                "kalman_type": m.kalman_type,
                "description": m.description,
            }
            for m in self.modes
        ]

    def to_ss(self) -> StateSpace:
        """Convert system to numerical StateSpace model.

        Returns
        -------
        StateSpace
            Numerical StateSpace object.
        """
        from ctrlpy.models.state_space import StateSpace

        def _to_np(m: sp.Matrix) -> np.ndarray:
            arr = np.zeros((m.rows, m.cols), dtype=np.float64)
            for i in range(m.rows):
                for j in range(m.cols):
                    val = m[i, j]
                    if hasattr(val, "free_symbols") and val.free_symbols:
                        raise ValueError(
                            f"Cannot convert symbolic expression {val} to numeric float."
                        )
                    arr[i, j] = float(sp.re(val.evalf()))
            return arr

        return StateSpace(_to_np(self.A), _to_np(self.B), _to_np(self.C), _to_np(self.D))

    def to_tf(self) -> TransferFunction:
        """Convert system to numerical TransferFunction model.

        Returns
        -------
        TransferFunction
            Numerical TransferFunction object.

        Raises
        ------
        NotImplementedError
            If the system is MIMO.
        """
        from ctrlpy.models.transfer_function import TransferFunction

        if not self.is_siso:
            raise NotImplementedError("to_tf() is only supported for SISO systems.")

        if self.tf_numerator is None or self.tf_denominator is None:
            return self.to_ss().to_tf()

        num_coeffs = [float(sp.re(c.evalf())) for c in self.tf_numerator.all_coeffs()]
        den_coeffs = [float(sp.re(c.evalf())) for c in self.tf_denominator.all_coeffs()]
        return TransferFunction(num_coeffs, den_coeffs)

    def _repr_latex_(self) -> str:
        r"""Render a comprehensive LaTeX breakdown of the state-space system and canonical transformations."""
        dt_sym = r"\dot{x}" if self.dt is None else "x[k+1]"
        state_sym = "x" if self.dt is None else "x[k]"
        u_sym = "u" if self.dt is None else "u[k]"
        y_sym = "y" if self.dt is None else "y[k]"

        a_tex = _sp_matrix_to_latex(self.A)
        b_tex = _sp_matrix_to_latex(self.B)
        c_tex = _sp_matrix_to_latex(self.C)
        d_tex = _sp_matrix_to_latex(self.D)
        ctrl_tex = _sp_matrix_to_latex(self.controllability_matrix)
        obs_tex = _sp_matrix_to_latex(self.observability_matrix)

        ctrl_status = (
            r"\textbf{Controllable}" if self.is_controllable else r"\textbf{Uncontrollable}"
        )
        obs_status = r"\textbf{Observable}" if self.is_observable else r"\textbf{Unobservable}"

        ev_joined = ", ".join(sp.latex(ev) for ev in self.eigenvalues)
        lines = [
            r"\begin{aligned}",
            r"\textbf{State-Space Representation:} \\",
            rf"{dt_sym} &= {a_tex} {state_sym} + {b_tex} {u_sym} \\",
            rf"{y_sym} &= {c_tex} {state_sym} + {d_tex} {u_sym} \\[1em]",
            (
                r"\textbf{Controllability Matrix } (\mathcal{C}): \quad "
                rf"\operatorname{{rank}}(\mathcal{{C}}) = {self.controllability_rank} / {self.n_states} \implies {ctrl_status} \\"
            ),
            rf"\mathcal{{C}} &= {ctrl_tex} \\[1em]",
            (
                r"\textbf{Observability Matrix } (\mathcal{O}): \quad "
                rf"\operatorname{{rank}}(\mathcal{{O}}) = {self.observability_rank} / {self.n_states} \implies {obs_status} \\"
            ),
            rf"\mathcal{{O}} &= {obs_tex} \\[1em]",
            (
                r"\textbf{Characteristic Polynomial:} \quad "
                rf"p(s) = {sp.latex(self.char_poly_expr)} = 0 \\"
            ),
            (
                r"\textbf{Eigenvalues / Modes:} \quad "
                rf"\lambda = \left\{{ {ev_joined} \right\}}"
            ),
            r"\end{aligned}",
        ]
        return "\n".join(lines)

    def _repr_markdown_(self) -> str:
        """Render Markdown for Jupyter environments."""
        return f"$${self._repr_latex_()}$$"

    def __str__(self) -> str:
        """ASCII formatted breakdown of state-space system and canonical forms."""
        lines = [
            "=== Pedagogical State-Space Analysis ===",
            f"State Dimension: n={self.n_states}, Inputs: m={self.n_inputs}, Outputs: p={self.n_outputs}",
            f"A =\n{self.A}",
            f"B =\n{self.B}",
            f"C =\n{self.C}",
            f"D =\n{self.D}",
            "",
            f"Controllability Matrix C (Rank {self.controllability_rank}/{self.n_states}, Controllable={self.is_controllable}):",
            f"{self.controllability_matrix}",
            "",
            f"Observability Matrix O (Rank {self.observability_rank}/{self.n_states}, Observable={self.is_observable}):",
            f"{self.observability_matrix}",
            "",
            f"Characteristic Polynomial: {self.char_poly_expr} = 0",
            f"Eigenvalues: {self.eigenvalues}",
            f"Uncontrollable Modes: {self.uncontrollable_modes}",
            f"Unobservable Modes: {self.unobservable_modes}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"StateSpaceTutor(n_states={self.n_states}, is_controllable={self.is_controllable}, "
            f"is_observable={self.is_observable})"
        )


def state_space_tutor(
    sys_or_A: Any = None,
    B: Any = None,
    C: Any = None,
    D: Any = None,
    *,
    num: Sequence[Any] | None = None,
    den: Sequence[Any] | None = None,
    dt: float | None = None,
) -> StateSpaceTutor:
    r"""Create a pedagogical StateSpaceTutor for canonical transformations and controllability/observability analysis.

    Parameters
    ----------
    sys_or_A : Any, optional
        StateSpace model, TransferFunction model, or $A$ matrix.
    B : Any, optional
        Input matrix $B$.
    C : Any, optional
        Output matrix $C$.
    D : Any, optional
        Feedthrough matrix $D$.
    num : Sequence[Any] | None, default=None
        Numerator polynomial coefficients.
    den : Sequence[Any] | None, default=None
        Denominator polynomial coefficients.
    dt : float | None, default=None
        Discrete sampling period (None for continuous-time).

    Returns
    -------
    StateSpaceTutor
        The analyzed tutor instance.
    """
    return StateSpaceTutor(sys_or_A, B, C, D, num=num, den=den, dt=dt)


def controllable_canonical_form(
    sys_or_A: Any = None,
    B: Any = None,
    C: Any = None,
    D: Any = None,
    *,
    num: Sequence[Any] | None = None,
    den: Sequence[Any] | None = None,
    dt: float | None = None,
) -> CanonicalFormResult:
    """Compute the Controllable Canonical Form (Phase-Variable Form) of a system.

    Parameters
    ----------
    sys_or_A : Any, optional
        StateSpace model, TransferFunction model, or $A$ matrix.
    B : Any, optional
        Input matrix $B$.
    C : Any, optional
        Output matrix $C$.
    D : Any, optional
        Feedthrough matrix $D$.
    num : Sequence[Any] | None, default=None
        Numerator polynomial coefficients.
    den : Sequence[Any] | None, default=None
        Denominator polynomial coefficients.
    dt : float | None, default=None
        Sampling period.

    Returns
    -------
    CanonicalFormResult
        Controllable canonical form representation with transformation matrix $T_c$.
    """
    tutor = StateSpaceTutor(sys_or_A, B, C, D, num=num, den=den, dt=dt)
    return tutor.controllable_canonical_form()


def observable_canonical_form(
    sys_or_A: Any = None,
    B: Any = None,
    C: Any = None,
    D: Any = None,
    *,
    num: Sequence[Any] | None = None,
    den: Sequence[Any] | None = None,
    dt: float | None = None,
) -> CanonicalFormResult:
    """Compute the Observable Canonical Form of a system.

    Parameters
    ----------
    sys_or_A : Any, optional
        StateSpace model, TransferFunction model, or $A$ matrix.
    B : Any, optional
        Input matrix $B$.
    C : Any, optional
        Output matrix $C$.
    D : Any, optional
        Feedthrough matrix $D$.
    num : Sequence[Any] | None, default=None
        Numerator polynomial coefficients.
    den : Sequence[Any] | None, default=None
        Denominator polynomial coefficients.
    dt : float | None, default=None
        Sampling period.

    Returns
    -------
    CanonicalFormResult
        Observable canonical form representation with transformation matrix $T_o$.
    """
    tutor = StateSpaceTutor(sys_or_A, B, C, D, num=num, den=den, dt=dt)
    return tutor.observable_canonical_form()


def jordan_canonical_form(
    sys_or_A: Any = None,
    B: Any = None,
    C: Any = None,
    D: Any = None,
    *,
    num: Sequence[Any] | None = None,
    den: Sequence[Any] | None = None,
    dt: float | None = None,
) -> CanonicalFormResult:
    """Compute the Jordan / Diagonal Modal Form of a system.

    Parameters
    ----------
    sys_or_A : Any, optional
        StateSpace model, TransferFunction model, or $A$ matrix.
    B : Any, optional
        Input matrix $B$.
    C : Any, optional
        Output matrix $C$.
    D : Any, optional
        Feedthrough matrix $D$.
    num : Sequence[Any] | None, default=None
        Numerator polynomial coefficients.
    den : Sequence[Any] | None, default=None
        Denominator polynomial coefficients.
    dt : float | None, default=None
        Sampling period.

    Returns
    -------
    CanonicalFormResult
        Jordan / diagonal canonical form representation with modal matrix $V$.
    """
    tutor = StateSpaceTutor(sys_or_A, B, C, D, num=num, den=den, dt=dt)
    return tutor.jordan_canonical_form()


def controllability_matrix(
    sys_or_A: Any = None,
    B: Any = None,
) -> sp.Matrix:
    r"""Compute the symbolic Kalman controllability matrix $\mathcal{C} = [B \quad AB \dots A^{n-1}B]$.

    Parameters
    ----------
    sys_or_A : Any
        StateSpace model, or state transition matrix $A$.
    B : Any, optional
        Input matrix $B$.

    Returns
    -------
    sp.Matrix
        Controllability matrix $\mathcal{C}$.
    """
    if hasattr(sys_or_A, "A") and hasattr(sys_or_A, "B"):
        A_sp = _to_sympy_matrix(sys_or_A.A)
        B_sp = _to_sympy_matrix(sys_or_A.B)
    else:
        A_sp = _to_sympy_matrix(sys_or_A)
        if B is None:
            raise ValueError("Input matrix B must be provided when A is given.")
        B_sp = _to_sympy_matrix(B)

    n = A_sp.rows
    cols: list[sp.Matrix] = []
    for k in range(n):
        cols.append((A_sp**k) * B_sp)
    return sp.Matrix.hstack(*cols)


def observability_matrix(
    sys_or_A: Any = None,
    C: Any = None,
) -> sp.Matrix:
    r"""Compute the symbolic Kalman observability matrix $\mathcal{O} = [C^T \quad A^TC^T \dots (A^{n-1})^TC^T]^T$.

    Parameters
    ----------
    sys_or_A : Any
        StateSpace model, or state transition matrix $A$.
    C : Any, optional
        Output matrix $C$.

    Returns
    -------
    sp.Matrix
        Observability matrix $\mathcal{O}$.
    """
    if hasattr(sys_or_A, "A") and hasattr(sys_or_A, "C"):
        A_sp = _to_sympy_matrix(sys_or_A.A)
        C_sp = _to_sympy_matrix(sys_or_A.C)
    else:
        A_sp = _to_sympy_matrix(sys_or_A)
        if C is None:
            raise ValueError("Output matrix C must be provided when A is given.")
        C_sp = _to_sympy_matrix(C)

    n = A_sp.rows
    rows: list[sp.Matrix] = []
    for k in range(n):
        rows.append(C_sp * (A_sp**k))
    return sp.Matrix.vstack(*rows)

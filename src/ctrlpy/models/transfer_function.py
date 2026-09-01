"""Transfer Function representation for Linear Time-Invariant (LTI) systems."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from ctrlpy.models.base import LinearTimeInvariant

if TYPE_CHECKING:
    from ctrlpy.models.state_space import StateSpace


def _format_poly_str(coeffs: NDArray[np.float64], var: str = "s") -> str:
    """Format polynomial coefficients into a standard mathematical string.

    Parameters
    ----------
    coeffs : NDArray[np.float64]
        Polynomial coefficients in descending powers of `var`.
    var : str, optional
        Variable name, default is 's'.

    Returns
    -------
    str
        Formatted polynomial string.
    """
    n = len(coeffs) - 1
    if n < 0 or (len(coeffs) == 1 and np.isclose(coeffs[0], 0.0)):
        return "0"

    terms: list[tuple[str, str]] = []
    for i, c in enumerate(coeffs):
        if np.isclose(c, 0.0):
            continue
        power = n - i
        abs_c = abs(c)
        if np.isclose(abs_c, round(abs_c)):
            abs_c_str = f"{round(abs_c)}"
        else:
            abs_c_str = f"{abs_c:g}"

        if power == 0:
            term_str = abs_c_str
        elif power == 1:
            term_str = f"{var}" if abs_c_str == "1" else f"{abs_c_str} {var}"
        else:
            term_str = f"{var}^{power}" if abs_c_str == "1" else f"{abs_c_str} {var}^{power}"

        sign = "-" if c < 0 else "+"
        terms.append((sign, term_str))

    if not terms:
        return "0"

    first_sign, first_term = terms[0]
    res = f"-{first_term}" if first_sign == "-" else first_term
    for sign, term in terms[1:]:
        res += f" {sign} {term}"
    return res


def _format_poly_latex(coeffs: NDArray[np.float64], var: str = "s") -> str:
    """Format polynomial coefficients into a LaTeX string.

    Parameters
    ----------
    coeffs : NDArray[np.float64]
        Polynomial coefficients in descending powers of `var`.
    var : str, optional
        Variable name, default is 's'.

    Returns
    -------
    str
        Formatted LaTeX string for the polynomial.
    """
    n = len(coeffs) - 1
    if n < 0 or (len(coeffs) == 1 and np.isclose(coeffs[0], 0.0)):
        return "0"

    terms: list[tuple[str, str]] = []
    for i, c in enumerate(coeffs):
        if np.isclose(c, 0.0):
            continue
        power = n - i
        abs_c = abs(c)
        if np.isclose(abs_c, round(abs_c)):
            abs_c_str = f"{round(abs_c)}"
        else:
            abs_c_str = f"{abs_c:g}"

        if power == 0:
            term_str = abs_c_str
        elif power == 1:
            term_str = f"{var}" if abs_c_str == "1" else f"{abs_c_str} {var}"
        else:
            term_str = (
                f"{var}^{{{power}}}" if abs_c_str == "1" else f"{abs_c_str} {var}^{{{power}}}"
            )

        sign = "-" if c < 0 else "+"
        terms.append((sign, term_str))

    if not terms:
        return "0"

    first_sign, first_term = terms[0]
    res = f"-{first_term}" if first_sign == "-" else first_term
    for sign, term in terms[1:]:
        res += f" {sign} {term}"
    return res


class TransferFunction(LinearTimeInvariant):
    r"""Continuous-time Linear Time-Invariant Transfer Function representation.

    Represents a rational polynomial transfer function in Laplace variable $s$:

    .. math::

        G(s) = \frac{N(s)}{D(s)} = \frac{b_m s^m + b_{m-1} s^{m-1} + \cdots + b_1 s + b_0}{a_n s^n + a_{n-1} s^{n-1} + \cdots + a_1 s + a_0}

    Parameters
    ----------
    num : Sequence[float] | NDArray[np.floating] | float
        Numerator polynomial coefficients $N(s)$ in descending powers of $s$.
    den : Sequence[float] | NDArray[np.floating] | float, optional
        Denominator polynomial coefficients $D(s)$ in descending powers of $s$.
        Defaults to `(1.0,)`.

    Raises
    ------
    ValueError
        If numerator or denominator is empty, or denominator is identically zero.
    """

    def __init__(
        self,
        num: Sequence[float] | NDArray[np.floating] | float,
        den: Sequence[float] | NDArray[np.floating] | float = (1.0,),
    ) -> None:
        num_arr = np.asarray(num, dtype=np.float64).ravel()
        den_arr = np.asarray(den, dtype=np.float64).ravel()

        if num_arr.size == 0:
            raise ValueError("Numerator cannot be empty.")
        if den_arr.size == 0:
            raise ValueError("Denominator cannot be empty.")

        # Trim leading zeros from denominator
        den_nonzero = np.flatnonzero(den_arr)
        if den_nonzero.size == 0:
            raise ValueError("Denominator cannot be identically zero.")
        den_arr = den_arr[den_nonzero[0] :]

        # Trim leading zeros from numerator
        num_nonzero = np.flatnonzero(num_arr)
        if num_nonzero.size == 0:
            num_arr = np.array([0.0], dtype=np.float64)
        else:
            num_arr = num_arr[num_nonzero[0] :]

        # Normalize by leading coefficient of denominator
        lead = den_arr[0]
        self._num: NDArray[np.float64] = num_arr / lead
        self._den: NDArray[np.float64] = den_arr / lead

    @property
    def num(self) -> NDArray[np.float64]:
        """Numerator polynomial coefficients."""
        return self._num

    @property
    def den(self) -> NDArray[np.float64]:
        """Denominator polynomial coefficients."""
        return self._den

    @property
    def inputs(self) -> int:
        """Number of inputs (1 for SISO transfer function)."""
        return 1

    @property
    def outputs(self) -> int:
        """Number of outputs (1 for SISO transfer function)."""
        return 1

    def poles(self) -> NDArray[np.complex128]:
        """Compute the poles of the transfer function.

        Returns
        -------
        NDArray[np.complex128]
            Array of pole locations in the complex s-plane.
        """
        if len(self._den) <= 1:
            return np.array([], dtype=np.complex128)
        return np.roots(self._den).astype(np.complex128)

    def zeros(self) -> NDArray[np.complex128]:
        """Compute the zeros of the transfer function.

        Returns
        -------
        NDArray[np.complex128]
            Array of zero locations in the complex s-plane.
        """
        if len(self._num) <= 1 or np.all(np.isclose(self._num, 0.0)):
            return np.array([], dtype=np.complex128)
        return np.roots(self._num).astype(np.complex128)

    def to_ss(self) -> StateSpace:
        """Convert transfer function to StateSpace representation.

        Returns
        -------
        StateSpace
            Equivalent state-space model.

        Raises
        ------
        ValueError
            If the transfer function is improper (degree of num > degree of den).
        """
        from ctrlpy.models.state_space import StateSpace

        if len(self._num) > len(self._den):
            raise ValueError(
                f"Improper transfer function (num degree {len(self._num) - 1} > "
                f"den degree {len(self._den) - 1}) cannot be converted to standard state-space."
            )

        a, b, c, d = signal.tf2ss(self._num, self._den)
        return StateSpace(a, b, c, d)

    def __add__(self, other: TransferFunction | StateSpace | float | np.number[Any]) -> Any:
        """Parallel interconnection (self + other)."""
        from ctrlpy.models.state_space import StateSpace

        if isinstance(other, (int, float, np.number)):
            k = float(other)
            num = np.polyadd(self._num, k * self._den)
            return TransferFunction(num, self._den)

        if isinstance(other, TransferFunction):
            num = np.polyadd(
                np.polymul(self._num, other._den),
                np.polymul(other._num, self._den),
            )
            den = np.polymul(self._den, other._den)
            return TransferFunction(num, den)

        if isinstance(other, StateSpace):
            return self.to_ss() + other

        return NotImplemented

    def __radd__(self, other: TransferFunction | StateSpace | float | np.number[Any]) -> Any:
        """Parallel interconnection with left operand (other + self)."""
        return self.__add__(other)

    def __neg__(self) -> TransferFunction:
        """Negate the transfer function (-self)."""
        return TransferFunction(-self._num, self._den)

    def __pos__(self) -> TransferFunction:
        """Positive representation (+self)."""
        return TransferFunction(self._num.copy(), self._den.copy())

    def __sub__(self, other: TransferFunction | StateSpace | float | np.number[Any]) -> Any:
        """Subtract systems (self - other)."""
        from ctrlpy.models.state_space import StateSpace

        if isinstance(other, (int, float, np.number)):
            return self.__add__(-float(other))

        if isinstance(other, TransferFunction):
            return self.__add__(-other)

        if isinstance(other, StateSpace):
            return self.to_ss() - other

        return NotImplemented

    def __rsub__(self, other: TransferFunction | StateSpace | float | np.number[Any]) -> Any:
        """Subtract from other system (other - self)."""
        from ctrlpy.models.state_space import StateSpace

        if isinstance(other, (int, float, np.number)):
            return TransferFunction(float(other), 1.0) - self

        if isinstance(other, StateSpace):
            return other - self.to_ss()

        return NotImplemented

    def __mul__(self, other: TransferFunction | StateSpace | float | np.number[Any]) -> Any:
        """Series (cascade) interconnection (self * other)."""
        from ctrlpy.models.state_space import StateSpace

        if isinstance(other, (int, float, np.number)):
            return TransferFunction(float(other) * self._num, self._den)

        if isinstance(other, TransferFunction):
            num = np.polymul(self._num, other._num)
            den = np.polymul(self._den, other._den)
            return TransferFunction(num, den)

        if isinstance(other, StateSpace):
            return self.to_ss() * other

        return NotImplemented

    def __rmul__(self, other: TransferFunction | StateSpace | float | np.number[Any]) -> Any:
        """Series (cascade) interconnection with left operand (other * self)."""
        return self.__mul__(other)

    def __truediv__(self, other: TransferFunction | float | np.number[Any]) -> TransferFunction:
        """Divide transfer functions (self / other)."""
        if isinstance(other, (int, float, np.number)):
            k = float(other)
            if np.isclose(k, 0.0):
                raise ZeroDivisionError("Cannot divide TransferFunction by zero.")
            return TransferFunction(self._num, self._den * k)

        if isinstance(other, TransferFunction):
            num = np.polymul(self._num, other._den)
            den = np.polymul(self._den, other._num)
            return TransferFunction(num, den)

        return NotImplemented

    def __rtruediv__(self, other: TransferFunction | float | np.number[Any]) -> TransferFunction:
        """Divide other by self (other / self)."""
        if isinstance(other, (int, float, np.number)):
            return TransferFunction(float(other), 1.0) / self

        return NotImplemented

    def feedback(
        self,
        other: TransferFunction | StateSpace | float | np.number[Any] = 1,
        sign: float = -1,
    ) -> Any:
        """Compute the closed-loop feedback interconnection.

        T(s) = self / (1 - sign * self * other)

        Parameters
        ----------
        other : TransferFunction | StateSpace | float | int, optional
            Feedback path transfer function or gain H(s), defaults to 1 (unity feedback).
        sign : int | float, optional
            Sign of feedback summation. Defaults to -1 for standard negative feedback.

        Returns
        -------
        TransferFunction | StateSpace
            Closed-loop system representation.
        """
        from ctrlpy.models.state_space import StateSpace

        if isinstance(other, StateSpace):
            return self.to_ss().feedback(other, sign=sign)

        if isinstance(other, (int, float, np.number)):
            num_h = np.array([float(other)], dtype=np.float64)
            den_h = np.array([1.0], dtype=np.float64)
        elif isinstance(other, TransferFunction):
            num_h = other._num
            den_h = other._den
        else:
            raise TypeError(f"Unsupported feedback block type: {type(other).__name__}")

        num = np.polymul(self._num, den_h)
        den = np.polyadd(
            np.polymul(self._den, den_h),
            -float(sign) * np.polymul(self._num, num_h),
        )
        return TransferFunction(num, den)

    def __repr__(self) -> str:
        """Return a string representation suitable for reproduction."""
        return f"TransferFunction(num={self._num.tolist()}, den={self._den.tolist()})"

    def __str__(self) -> str:
        """Return a formatted string representing the rational function in 's'."""
        num_str = _format_poly_str(self._num)
        den_str = _format_poly_str(self._den)
        width = max(len(num_str), len(den_str)) + 2
        dash_line = "-" * width
        return f"{num_str.center(width)}\n{dash_line}\n{den_str.center(width)}"

    def _repr_latex_(self) -> str:
        """Return a LaTeX representation for Jupyter environments."""
        num_latex = _format_poly_latex(self._num)
        den_latex = _format_poly_latex(self._den)
        return f"$$\\frac{{{num_latex}}}{{{den_latex}}}$$"

    def _repr_markdown_(self) -> str:
        """Return a Markdown representation for Jupyter environments."""
        return self._repr_latex_()


# Convenient alias
tf = TransferFunction

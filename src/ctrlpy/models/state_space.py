"""State-Space representation for Linear Time-Invariant (LTI) systems."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import signal

from ctrlpy.exceptions import DimensionMismatchError
from ctrlpy.models.base import LinearTimeInvariant

if TYPE_CHECKING:
    from ctrlpy.models.transfer_function import TransferFunction


def _matrix_to_latex(mat: NDArray[np.float64]) -> str:
    """Format a 2D NumPy array into LaTeX bmatrix syntax.

    Parameters
    ----------
    mat : NDArray[np.float64]
        2D matrix to format.

    Returns
    -------
    str
        LaTeX string representation of the matrix.
    """
    lines: list[str] = []
    for row in mat:
        row_str = " & ".join(f"{round(x)}" if np.isclose(x, round(x)) else f"{x:g}" for x in row)
        lines.append(row_str)
    return r"\begin{bmatrix} " + r" \\ ".join(lines) + r" \end{bmatrix}"


class StateSpace(LinearTimeInvariant):
    """Continuous-time Linear Time-Invariant State-Space representation.

    Represents systems of the form:
        dx/dt = A x + B u
            y = C x + D u

    Parameters
    ----------
    A : ArrayLike
        State transition matrix (n x n).
    B : ArrayLike
        Input matrix (n x m).
    C : ArrayLike
        Output matrix (p x n).
    D : ArrayLike
        Direct feedthrough matrix (p x m).

    Raises
    ------
    ValueError
        If matrix dimensions are incompatible.
    """

    def __init__(
        self,
        A: ArrayLike,
        B: ArrayLike,
        C: ArrayLike,
        D: ArrayLike,
    ) -> None:
        self._A, self._B, self._C, self._D = self._validate_and_convert(A, B, C, D)

    @staticmethod
    def _validate_and_convert(
        A: ArrayLike,
        B: ArrayLike,
        C: ArrayLike,
        D: ArrayLike,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        a_arr = np.asarray(A, dtype=np.float64)
        b_arr = np.asarray(B, dtype=np.float64)
        c_arr = np.asarray(C, dtype=np.float64)
        d_arr = np.asarray(D, dtype=np.float64)

        if a_arr.ndim == 0 or (a_arr.ndim == 1 and a_arr.size == 1):
            a_arr = a_arr.reshape((1, 1))

        if a_arr.ndim != 2 or a_arr.shape[0] != a_arr.shape[1]:
            raise DimensionMismatchError(
                f"Matrix A must be a square 2D array, got shape {a_arr.shape}."
            )

        n = a_arr.shape[0]

        if b_arr.ndim == 0:
            if n == 1:
                b_arr = b_arr.reshape((1, 1))
            else:
                raise DimensionMismatchError(
                    f"Scalar B is incompatible with A of shape {a_arr.shape}."
                )
        elif b_arr.ndim == 1:
            if b_arr.shape[0] == n:
                b_arr = b_arr.reshape((n, 1))
            elif n == 1:
                b_arr = b_arr.reshape((1, -1))
            else:
                raise DimensionMismatchError(
                    f"1D array B of length {b_arr.shape[0]} is incompatible "
                    f"with A of shape {a_arr.shape}."
                )

        if b_arr.ndim != 2 or b_arr.shape[0] != n:
            raise DimensionMismatchError(
                f"Matrix B must have {n} rows matching A, got shape {b_arr.shape}."
            )

        m = b_arr.shape[1]

        if c_arr.ndim == 0:
            if n == 1:
                c_arr = c_arr.reshape((1, 1))
            else:
                raise DimensionMismatchError(
                    f"Scalar C is incompatible with A of shape {a_arr.shape}."
                )
        elif c_arr.ndim == 1:
            if c_arr.shape[0] == n:
                c_arr = c_arr.reshape((1, n))
            elif n == 1:
                c_arr = c_arr.reshape((-1, 1))
            else:
                raise DimensionMismatchError(
                    f"1D array C of length {c_arr.shape[0]} is incompatible "
                    f"with A of shape {a_arr.shape}."
                )

        if c_arr.ndim != 2 or c_arr.shape[1] != n:
            raise DimensionMismatchError(
                f"Matrix C must have {n} columns matching A, got shape {c_arr.shape}."
            )

        p = c_arr.shape[0]

        if d_arr.ndim == 0:
            d_arr = np.full((p, m), d_arr.item(), dtype=np.float64)
        elif d_arr.ndim == 1:
            if d_arr.size == p * m:
                d_arr = d_arr.reshape((p, m))
            elif d_arr.size == 1:
                d_arr = np.full((p, m), d_arr[0], dtype=np.float64)
            else:
                raise DimensionMismatchError(
                    f"1D array D of size {d_arr.size} is incompatible with "
                    f"expected shape ({p}, {m})."
                )

        if d_arr.ndim != 2 or d_arr.shape != (p, m):
            raise DimensionMismatchError(
                f"Matrix D must have shape ({p}, {m}), got shape {d_arr.shape}."
            )

        return a_arr, b_arr, c_arr, d_arr

    @property
    def A(self) -> NDArray[np.float64]:
        """State transition matrix (n x n)."""
        return self._A

    @property
    def B(self) -> NDArray[np.float64]:
        """Input matrix (n x m)."""
        return self._B

    @property
    def C(self) -> NDArray[np.float64]:
        """Output matrix (p x n)."""
        return self._C

    @property
    def D(self) -> NDArray[np.float64]:
        """Direct feedthrough matrix (p x m)."""
        return self._D

    @property
    def inputs(self) -> int:
        """Number of inputs (m)."""
        return int(self._B.shape[1])

    @property
    def outputs(self) -> int:
        """Number of outputs (p)."""
        return int(self._C.shape[0])

    @property
    def n_states(self) -> int:
        """Number of state variables (n)."""
        return int(self._A.shape[0])

    def poles(self) -> NDArray[np.complex128]:
        """Compute the poles (eigenvalues of A) of the state-space system.

        Returns
        -------
        NDArray[np.complex128]
            1D array of system poles.
        """
        return np.linalg.eigvals(self._A).astype(np.complex128)

    def zeros(self) -> NDArray[np.complex128]:
        """Compute the zeros of the state-space system.

        Returns
        -------
        NDArray[np.complex128]
            1D array of system zeros.

        Raises
        ------
        NotImplementedError
            If the system is not SISO.
        """
        if not self.is_siso:
            raise NotImplementedError("zeros() is currently supported for SISO systems.")
        return self.to_tf().zeros()

    def to_tf(self, input_index: int = 0, output_index: int = 0) -> TransferFunction:
        """Convert state-space model to TransferFunction representation.

        Parameters
        ----------
        input_index : int, optional
            Index of the input channel for multi-input systems, by default 0.
        output_index : int, optional
            Index of the output channel for multi-output systems, by default 0.

        Returns
        -------
        TransferFunction
            Equivalent transfer function representation for the specified input/output channel.

        Raises
        ------
        IndexError
            If input_index or output_index is out of range.
        """
        from ctrlpy.models.transfer_function import TransferFunction

        if not (0 <= input_index < self.inputs):
            raise IndexError(f"input_index {input_index} out of range for {self.inputs} inputs.")
        if not (0 <= output_index < self.outputs):
            raise IndexError(
                f"output_index {output_index} out of range for {self.outputs} outputs."
            )

        num, den = signal.ss2tf(self._A, self._B, self._C, self._D, input=input_index)
        num_channel = num[output_index]

        # Clean numerical precision artifacts
        max_coeff = max(
            float(np.max(np.abs(num_channel))),
            float(np.max(np.abs(den))),
            1.0,
        )
        tol = 1e-12 * max_coeff
        num_channel = np.where(np.isclose(num_channel, 0.0, atol=tol), 0.0, num_channel)
        den = np.where(np.isclose(den, 0.0, atol=tol), 0.0, den)

        return TransferFunction(num_channel, den)

    def __add__(
        self,
        other: StateSpace | TransferFunction | float | np.number[Any],
    ) -> StateSpace:
        """Parallel interconnection of StateSpace systems (self + other)."""
        from ctrlpy.models.transfer_function import TransferFunction

        if isinstance(other, (int, float, np.number)):
            k = float(other)
            if self.is_siso:
                d_add = np.array([[k]], dtype=np.float64)
            elif self.inputs == self.outputs:
                d_add = k * np.eye(self.inputs, dtype=np.float64)
            else:
                raise ValueError("Cannot add scalar to non-square MIMO StateSpace system.")
            return StateSpace(self._A.copy(), self._B.copy(), self._C.copy(), self._D + d_add)

        if isinstance(other, TransferFunction):
            return self.__add__(other.to_ss())

        if isinstance(other, StateSpace):
            if self.inputs != other.inputs or self.outputs != other.outputs:
                raise ValueError(
                    "Cannot add StateSpace systems with incompatible inputs/outputs: "
                    f"({self.inputs}, {self.outputs}) vs ({other.inputs}, {other.outputs})."
                )
            n1 = self.n_states
            n2 = other.n_states
            a_block = np.block(
                [
                    [self._A, np.zeros((n1, n2), dtype=np.float64)],
                    [np.zeros((n2, n1), dtype=np.float64), other._A],
                ]
            )
            b_block = np.vstack([self._B, other._B])
            c_block = np.hstack([self._C, other._C])
            d_block = self._D + other._D
            return StateSpace(a_block, b_block, c_block, d_block)

        return NotImplemented

    def __radd__(
        self,
        other: StateSpace | TransferFunction | float | np.number[Any],
    ) -> StateSpace:
        """Parallel interconnection with left operand (other + self)."""
        return self.__add__(other)

    def __neg__(self) -> StateSpace:
        """Negate the StateSpace system (-self)."""
        return StateSpace(self._A.copy(), self._B.copy(), -self._C, -self._D)

    def __pos__(self) -> StateSpace:
        """Positive representation (+self)."""
        return StateSpace(self._A.copy(), self._B.copy(), self._C.copy(), self._D.copy())

    def __sub__(
        self,
        other: StateSpace | TransferFunction | float | np.number[Any],
    ) -> StateSpace:
        """Subtract systems (self - other)."""
        if isinstance(other, (int, float, np.number)):
            return self.__add__(-float(other))

        if isinstance(other, (StateSpace, LinearTimeInvariant)):
            return self.__add__(-other)

        return NotImplemented

    def __rsub__(
        self,
        other: StateSpace | TransferFunction | float | np.number[Any],
    ) -> StateSpace:
        """Subtract StateSpace system from other (other - self)."""
        if isinstance(other, (int, float, np.number, StateSpace, LinearTimeInvariant)):
            return (-self).__add__(other)

        return NotImplemented

    def __mul__(
        self,
        other: StateSpace | TransferFunction | float | np.number[Any],
    ) -> StateSpace:
        """Series (cascade) interconnection: u -> other -> self -> y."""
        from ctrlpy.models.transfer_function import TransferFunction

        if isinstance(other, (int, float, np.number)):
            k = float(other)
            return StateSpace(self._A.copy(), self._B.copy(), k * self._C, k * self._D)

        if isinstance(other, TransferFunction):
            return self.__mul__(other.to_ss())

        if isinstance(other, StateSpace):
            if other.outputs != self.inputs:
                raise ValueError(
                    "Cannot cascade StateSpace systems: output dimension of "
                    f"first system ({other.outputs}) must match input dimension "
                    f"of second system ({self.inputs})."
                )
            a1, b1, c1, d1 = self._A, self._B, self._C, self._D
            a2, b2, c2, d2 = other._A, other._B, other._C, other._D
            n1 = self.n_states
            n2 = other.n_states

            a_block = np.block(
                [
                    [a1, b1 @ c2],
                    [np.zeros((n2, n1), dtype=np.float64), a2],
                ]
            )
            b_block = np.vstack([b1 @ d2, b2])
            c_block = np.hstack([c1, d1 @ c2])
            d_block = d1 @ d2
            return StateSpace(a_block, b_block, c_block, d_block)

        return NotImplemented

    def __rmul__(
        self,
        other: StateSpace | TransferFunction | float | np.number[Any],
    ) -> StateSpace:
        """Series (cascade) interconnection with left operand."""
        from ctrlpy.models.transfer_function import TransferFunction

        if isinstance(other, (int, float, np.number)):
            k = float(other)
            return StateSpace(self._A.copy(), self._B.copy(), k * self._C, k * self._D)

        if isinstance(other, TransferFunction):
            return other.to_ss() * self

        return NotImplemented

    def feedback(
        self,
        other: (StateSpace | TransferFunction | float | np.number[Any] | None) = None,
        sign: float = -1,
    ) -> StateSpace:
        """Compute the closed-loop feedback interconnection in state-space.

        Parameters
        ----------
        other : StateSpace | TransferFunction | float | int | None, optional
            Feedback path system H, defaults to None (unity feedback identity gain).
        sign : int | float, optional
            Sign of feedback summation. Defaults to -1 for standard negative feedback.

        Returns
        -------
        StateSpace
            Closed-loop state-space system.

        Raises
        ------
        ValueError
            If matrix dimensions are incompatible or if (I - sign*D1*D2) is singular.
        """
        from ctrlpy.models.transfer_function import TransferFunction

        if isinstance(other, TransferFunction):
            return self.feedback(other.to_ss(), sign=sign)

        s = float(sign)
        p = self.outputs
        m = self.inputs
        a1, b1, c1, d1 = self._A, self._B, self._C, self._D

        if other is None or isinstance(other, (int, float, np.number)):
            k = 1.0 if other is None else float(other)
            if self.is_siso:
                d2 = np.array([[k]], dtype=np.float64)
            else:
                d2 = k * np.eye(m, p, dtype=np.float64)

            i_p = np.eye(p, dtype=np.float64)
            i_m = np.eye(m, dtype=np.float64)
            m_p = i_p - s * (d1 @ d2)
            m_m = i_m - s * (d2 @ d1)

            try:
                e = np.linalg.inv(m_p)
                e_tilde = np.linalg.inv(m_m)
            except np.linalg.LinAlgError as err:
                raise ValueError("Well-posedness error: (I - sign*D1*D2) is singular.") from err

            a_cl = a1 + s * (b1 @ e_tilde @ d2 @ c1)
            b_cl = b1 @ e_tilde
            c_cl = e @ c1
            d_cl = e @ d1
            return StateSpace(a_cl, b_cl, c_cl, d_cl)

        if isinstance(other, StateSpace):
            if other.inputs != self.outputs or other.outputs != self.inputs:
                raise ValueError(
                    "Feedback connection dimension mismatch: forward system shape "
                    f"(outputs={self.outputs}, inputs={self.inputs}) vs feedback system "
                    f"shape (inputs={other.inputs}, outputs={other.outputs})."
                )

            a2, b2, c2, d2 = other._A, other._B, other._C, other._D
            i_p = np.eye(p, dtype=np.float64)
            i_m = np.eye(m, dtype=np.float64)
            m_p = i_p - s * (d1 @ d2)
            m_m = i_m - s * (d2 @ d1)

            try:
                e = np.linalg.inv(m_p)
                e_tilde = np.linalg.inv(m_m)
            except np.linalg.LinAlgError as err:
                raise ValueError("Well-posedness error: (I - sign*D1*D2) is singular.") from err

            a_cl = np.block(
                [
                    [a1 + s * (b1 @ e_tilde @ d2 @ c1), s * (b1 @ e_tilde @ c2)],
                    [b2 @ e @ c1, a2 + s * (b2 @ e @ d1 @ c2)],
                ]
            )
            b_cl = np.vstack([b1 @ e_tilde, b2 @ e @ d1])
            c_cl = np.hstack([e @ c1, s * (e @ d1 @ c2)])
            d_cl = e @ d1
            return StateSpace(a_cl, b_cl, c_cl, d_cl)

        raise TypeError(f"Unsupported feedback block type: {type(other).__name__}")

    def __repr__(self) -> str:
        """Return a string representation of the StateSpace object."""
        return (
            f"StateSpace(\n"
            f"  A={self._A.tolist()},\n"
            f"  B={self._B.tolist()},\n"
            f"  C={self._C.tolist()},\n"
            f"  D={self._D.tolist()}\n"
            f")"
        )

    def __str__(self) -> str:
        """Return a formatted string representing the state-space matrices."""
        return f"StateSpace:\nA =\n{self._A}\n\nB =\n{self._B}\n\nC =\n{self._C}\n\nD =\n{self._D}"

    def _repr_latex_(self) -> str:
        """Return a LaTeX representation for Jupyter environments."""
        a_tex = _matrix_to_latex(self._A)
        b_tex = _matrix_to_latex(self._B)
        c_tex = _matrix_to_latex(self._C)
        d_tex = _matrix_to_latex(self._D)
        return (
            r"$$\begin{aligned} "
            rf"\dot{{x}} &= {a_tex} x + {b_tex} u \\ "
            rf"y &= {c_tex} x + {d_tex} u "
            r"\end{aligned}$$"
        )


# Convenient alias
ss = StateSpace

"""Custom exception hierarchy and domain errors for ctrlpy."""

from __future__ import annotations


class CtrlPyError(Exception):
    """Base exception class for all ctrlpy domain errors."""


class UnstableSystemError(CtrlPyError):
    """Exception raised when an operation is invalid for an unstable system.

    For example, steady-state metrics, settling time, or overshoot calculations
    are not well-defined for systems with poles in the Right Half Plane (Re(p) > 0).
    """


class UnstableSystemWarning(UserWarning):
    """Warning emitted when an operation may be problematic for an unstable system."""


class DimensionMismatchError(CtrlPyError, ValueError):
    """Exception raised when system matrix or vector dimensions are incompatible."""

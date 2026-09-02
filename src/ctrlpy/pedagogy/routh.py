"""Routh-Hurwitz stability criterion and step-by-step array construction."""

from __future__ import annotations

from ctrlpy.symbolic.routh import (
    RouthArray,
    RouthResult,
    routh_array,
    routh_table,
)

__all__ = [
    "RouthArray",
    "RouthResult",
    "routh_array",
    "routh_table",
]

from __future__ import annotations

import numpy as np


def get_minimum_precise_numeric_dtype(value: int | float) -> np.dtype:
    """
    Get the minimum numeric dtype for a value without reducing precision.

    Integers will return the smallest integer type that can hold the value, while floats
    will return their current precision.
    """
    return (
        np.min_scalar_type(value)
        if np.issubdtype(type(value), np.integer)
        else np.dtype(type(value))
    )

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


def can_cast_value(value: float | int | np.number, to_dtype: np.dtype) -> bool:
    """
    Test whether a given numeric value can be safely cast to the target dtype.

    Compared to `np.can_cast` and `np.min_scalar_type`, this check is determined based
    on the value itself and is able to cast positive integers to compatible signed
    types.

    Examples
    --------
    >>> can_cast_value(-5, np.uint8)
    False
    >>> can_cast_value(255, np.uint8)
    True
    >>> can_cast_value(1.0, np.uint8)
    True

    Note that `can_cast_value` supports casting from positive integers to compatible
    signed types where Numpy does not:

    >>> import numpy as np
    >>> np.can_cast(np.min_scalar_type(999), np.int16)
    False
    >>> can_cast_value(999, np.int16)
    True
    """
    value_type = type(value)

    if np.issubdtype(to_dtype, np.floating):
        # Use Numpy precision rules when casting to float
        return np.can_cast(np.min_scalar_type(value), to_dtype)

    if np.issubdtype(to_dtype, np.integer):
        # If the value is a float whole number, continue to the integer casting rules
        if np.issubdtype(value_type, np.floating) and value % 1 == 0:
            value = int(value)
            value_type = int

        # When casting between integer types, check that the value is within the range
        if np.issubdtype(value_type, np.integer):
            info = np.iinfo(to_dtype)
            return value >= info.min and value <= info.max

    return False

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
    Test whether a given value can be safely cast to the target dtype.

    This is implemented with `np.can_cast(np.min_scalar_type(value), to_dtype)` except
    in the case of integer target types, where casting is more permissive to allow:

    - Casting from float whole numbers to integers (e.g. `1.0` to `np.int8`)
    - Casting from unsigned to signed integers where safe (e.g. `9999` to `np.int16`)

    Examples
    --------
    >>> can_cast_value(-5, np.uint8)
    False
    >>> can_cast_value(255, np.uint8)
    True
    >>> can_cast_value(True, np.uint8)
    True

    `can_cast_value` is more permissive than `np.can_cast` for some values:

    >>> import numpy as np
    >>> np.can_cast(np.min_scalar_type(999), np.int16)
    False
    >>> can_cast_value(999, np.int16)
    True
    >>> np.can_cast(np.min_scalar_type(1.0), np.int8)
    False
    >>> can_cast_value(1.0, np.int8)
    True
    """
    # Override Numpy with permissive rules for integer target types
    if np.issubdtype(to_dtype, np.integer):
        value_type = type(value)

        # If the value is an integer or whole-number float, check that it is in range
        if np.issubdtype(value_type, np.integer) or (
            np.issubdtype(value_type, np.floating) and value % 1 == 0
        ):
            info = np.iinfo(to_dtype)
            return value >= info.min and value <= info.max

    # Use Numpy casting rules for everything else
    return np.can_cast(np.min_scalar_type(value), to_dtype)

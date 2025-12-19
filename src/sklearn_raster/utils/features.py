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


def can_cast_nodata_value(
    value: float | int | bool | np.number, to_dtype: np.dtype
) -> bool:
    """
    Test whether a given NoData value can be safely cast to the target dtype.

    This generally follows Numpy safe casting rules with a few exceptions:

    - Casting from unsigned to signed integers is allowed where safe (e.g. `999` to
    `np.int16`) since this is logical expected behavior.
    - Casting from float whole numbers to integers is allowed (e.g. `1.0` to `np.int8`)
    to permit shared NoData values across features of different dtypes.
    - Casting from boolean to numeric is NOT allowed (e.g. `False` to `np.uint8`) as the
    implicit cast could result in unexpected values being masked.

    Examples
    --------
    >>> can_cast_nodata_value(-5, np.uint8)
    False
    >>> can_cast_nodata_value(255, np.uint8)
    True

    `can_cast_nodata_value` is more permissive than `np.can_cast` for some values:

    >>> import numpy as np
    >>> np.can_cast(np.min_scalar_type(999), np.int16)
    False
    >>> can_cast_nodata_value(999, np.int16)
    True
    >>> np.can_cast(np.min_scalar_type(1.0), np.int8)
    False
    >>> can_cast_nodata_value(1.0, np.int8)
    True

    `can_cast_nodata_value` is more restrictive than `np.can_cast` for boolean values:

    >>> np.can_cast(np.min_scalar_type(True), np.uint8)
    True
    >>> can_cast_nodata_value(True, np.uint8)
    False
    >>> can_cast_nodata_value(True, np.int16)
    False
    >>> can_cast_nodata_value(True, np.float32)
    False
    """
    value_type = type(value)

    # Allow casting integer or whole-number floats to integer types based on value range
    if np.issubdtype(to_dtype, np.integer) and (
        np.issubdtype(value_type, np.integer)
        or (np.issubdtype(value_type, np.floating) and value % 1 == 0)
    ):
        info = np.iinfo(to_dtype)
        return value >= info.min and value <= info.max

    # Disallow casting from boolean to numeric types
    if np.issubdtype(value_type, np.bool_) and not np.issubdtype(to_dtype, np.bool_):
        return False

    # Use Numpy casting rules for everything else
    return np.can_cast(np.min_scalar_type(value), to_dtype)

from __future__ import annotations

import re

import numpy as np
import pytest

from sklearn_raster.utils.decorators import (
    map_over_arguments,
    with_inputs_reshaped_to_ndim,
)
from sklearn_raster.utils.features import (
    can_cast_value,
    get_minimum_precise_numeric_dtype,
)


def test_minimum_precise_numeric_dtype():
    """Test that correct minimum precise numeric dtypes are returned."""
    # Integers should return the smallest dtype that can hold the value
    assert get_minimum_precise_numeric_dtype(1) == np.uint8
    assert get_minimum_precise_numeric_dtype(-1) == np.int8
    assert get_minimum_precise_numeric_dtype(256) == np.uint16

    # Floats should return their current precision
    assert get_minimum_precise_numeric_dtype(42.0) == np.float64
    assert get_minimum_precise_numeric_dtype(np.float32(np.nan)) == np.float32


def test_map_over_arguments():
    """Test that map_over_arguments decorator works as expected."""

    @map_over_arguments("a", "b")
    def func(a, b):
        return a + b

    assert func(1, 2) == 3
    assert func(1, [2, 3]) == (3, 4)
    assert func(a=[1, 2], b=[3, 4]) == (4, 6)

    with pytest.raises(ValueError, match="must be the same length or scalar"):
        func(a=[1, 2], b=[3, 4, 5])


def test_map_over_arguments_validation():
    """Test that map_over_arguments raises for unaccepted arguments."""
    with pytest.raises(ValueError, match=re.escape("cannot be mapped over: ['a']")):

        @map_over_arguments("a")
        def _(): ...


@pytest.mark.parametrize(
    ("ndim", "expected_shape"),
    [
        (1, (8,)),
        (2, (8, 1)),
        (3, (2, 4, 1)),
        (4, (1, 2, 4, 1)),
    ],
)
def test_with_inputs_reshaped_to_ndim(ndim: int, expected_shape: tuple[int, ...]):
    """
    Test that with_inputs_reshaped_to_ndim flattens, expands, and restores dimensions.
    """

    @with_inputs_reshaped_to_ndim(ndim)
    def assert_shape(x):
        assert x.shape == expected_shape
        return x

    shape_in = (2, 4, 1)
    array = np.zeros(shape_in)
    result = assert_shape(array)
    assert result.shape == shape_in


@pytest.mark.parametrize(
    ("value", "to_dtype", "can_cast"),
    [
        # Negative ints cannot cast to unsigned types
        (-1, np.uint8, False),
        (-5, np.uint8, False),
        (-32768, np.uint16, False),
        # Non-negative ints can cast to compatible signed and unsigned types
        (127, np.int8, True),
        (32767, np.int16, True),
        # Ints can cast at dtype boundaries
        (0, np.uint8, True),
        (255, np.uint8, True),
        (256, np.uint8, False),
        (0, np.int8, True),
        (65535, np.uint16, True),
        (65536, np.uint16, False),
        # Floats can cast to integer types if they are whole numbers and in range
        (0.0, np.uint8, True),
        (255.0, np.uint8, True),
        (256.0, np.uint8, False),
        (0.0, np.int8, True),
        (65535.0, np.uint16, True),
        (65536.0, np.uint16, False),
        # Non-whole number floats cannot cast to integer types
        (1.5, np.uint8, False),
        (-2.3, np.int8, False),
        # Floats can always cast to other float types within precision limits
        (1.0, np.float32, True),
        (1.5, np.float32, True),
        (1e40, np.float32, False),
        # Special float values can cast to float types
        (np.nan, np.float32, True),
        (np.inf, np.float32, True),
        (-np.inf, np.float32, True),
        # Special float values cannot cast to integer types
        (np.nan, np.int32, False),
        (np.inf, np.int32, False),
        (-np.inf, np.int32, False),
    ],
)
def test_can_cast_value(value: float | int, to_dtype: np.dtype, can_cast: bool):
    """
    Test that can_cast_value works as expected.
    """
    assert can_cast_value(value, to_dtype) == can_cast

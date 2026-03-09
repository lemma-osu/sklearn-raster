from __future__ import annotations

import numpy as np
import pytest

from sklearn_raster.utils.decorators import with_inputs_reshaped_to_ndim
from sklearn_raster.utils.features import (
    can_cast_nodata_value,
    get_minimum_precise_numeric_dtype,
)
from sklearn_raster.utils.ufunc import _UfuncResult


def test_minimum_precise_numeric_dtype():
    """Test that correct minimum precise numeric dtypes are returned."""
    # Integers should return the smallest dtype that can hold the value
    assert get_minimum_precise_numeric_dtype(1) == np.uint8
    assert get_minimum_precise_numeric_dtype(-1) == np.int8
    assert get_minimum_precise_numeric_dtype(256) == np.uint16

    # Floats should return their current precision
    assert get_minimum_precise_numeric_dtype(42.0) == np.float64
    assert get_minimum_precise_numeric_dtype(np.float32(np.nan)) == np.float32


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
        # Boolean values cannot cast to numeric types
        (True, np.uint8, False),
        (False, np.int8, False),
        (True, np.float32, False),
        (False, np.float32, False),
        # Boolean values can only cast to boolean types
        (True, np.bool_, True),
        (False, np.bool_, True),
    ],
)
def test_can_cast_nodata_value(value: float | int, to_dtype: np.dtype, can_cast: bool):
    """
    Test that can_cast_nodata_value works as expected.
    """
    assert can_cast_nodata_value(value, to_dtype) == can_cast


@pytest.mark.parametrize(("result", "unwrapped"), [(0, 0), ((0,), 0), ((0, 1), (0, 1))])
def test_ufuncresult_unwraps(result, unwrapped):
    assert _UfuncResult(result).unwrap() == unwrapped


@pytest.mark.parametrize(("result", "length"), [(0, 1), ((0,), 1), ((0, 1), 2)])
def test_ufuncresult_length(result, length):
    assert len(_UfuncResult(result)) == length


def test_ufuncresult_map():
    result = _UfuncResult((1, 2, 3))
    assert result.map(lambda x: x**2).unwrap() == (1, 4, 9)


def test_ufuncresult_zipmap_single_value():
    result = _UfuncResult(0)
    assert result.zip_map(lambda x, y: x + y, (5,)).unwrap() == 5


def test_ufuncresult_zipmap_tuple():
    result = _UfuncResult((0, 0))
    assert result.zip_map(lambda x, y: x + y, (5, 5)).unwrap() == (5, 5)


def test_ufuncresult_map_passes_kwargs():
    result = _UfuncResult(2)
    assert result.zip_map(lambda x, pow: x**pow, pow=8).unwrap() == 256

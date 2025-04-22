import re

import numpy as np
import pytest

from sklearn_raster.utils.features import get_minimum_precise_numeric_dtype
from sklearn_raster.utils.wrapper import map_over_arguments


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

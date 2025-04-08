import numpy as np

from sklearn_raster.utils.features import get_minimum_precise_numeric_dtype


def test_minimum_precise_numeric_dtype():
    """Test that correct minimum precise numeric dtypes are returned."""
    # Integers should return the smallest dtype that can hold the value
    assert get_minimum_precise_numeric_dtype(1) == np.uint8
    assert get_minimum_precise_numeric_dtype(-1) == np.int8
    assert get_minimum_precise_numeric_dtype(256) == np.uint16

    # Floats should return their current precision
    assert get_minimum_precise_numeric_dtype(42.0) == np.float64
    assert get_minimum_precise_numeric_dtype(np.float32(np.nan)) == np.float32

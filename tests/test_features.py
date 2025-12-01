"""Test the features module."""

from __future__ import annotations

import re

import numpy as np
import pytest
import threadpoolctl
import xarray as xr
from numpy.testing import assert_array_equal

from sklearn_raster.features import FeatureArray
from sklearn_raster.types import FeatureArrayType, MissingType
from sklearn_raster.utils.features import get_minimum_precise_numeric_dtype

from .feature_utils import (
    parametrize_feature_array_types,
    unwrap_features,
    wrap_features,
)


@parametrize_feature_array_types()
@pytest.mark.parametrize("skip_nodata", [True, False])
def test_input_array_not_mutated(
    feature_array_type: type[FeatureArrayType], skip_nodata: bool
):
    """Ensure that applying a ufunc to features doesn't mutate the original array."""
    a = np.array([[[0, 1]], [[1, np.nan]]])
    original_array = a.copy()

    array = wrap_features(a, type=feature_array_type)

    features = FeatureArray.from_feature_array(array, nodata_input=0)
    features.apply_ufunc_across_features(
        lambda x: x * 2.0,
        skip_nodata=skip_nodata,
        output_dims=[["variable"]],
        output_sizes={"variable": a.shape[0]},
        output_dtypes=[a.dtype],
    )

    assert_array_equal(a, original_array)


@parametrize_feature_array_types()
@pytest.mark.parametrize("skip_nodata", [True, False])
@pytest.mark.parametrize(
    "val_dtype", [(-1, np.dtype(np.uint8)), (np.nan, np.dtype(np.int16))]
)
def test_nodata_output_with_unsupported_dtype(
    val_dtype: tuple[int | float, np.dtype],
    feature_array_type: type[FeatureArrayType],
    skip_nodata: bool,
):
    """Test that an error is raised when nodata_output is the wrong dtype."""
    # Make sure there's a value to mask in the input array
    a = np.array([[[np.nan]]])
    array = wrap_features(a, type=feature_array_type)
    features = FeatureArray.from_feature_array(array, nodata_input=0)

    output_nodata, return_dtype = val_dtype
    nodata_output_type = get_minimum_precise_numeric_dtype(output_nodata)
    expected_msg = (
        f"({nodata_output_type}) does not fit in the array dtype ({return_dtype})"
    )
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        # Unwrap to force computation for lazy arrays
        unwrap_features(
            features.apply_ufunc_across_features(
                lambda x: np.ones_like(x).astype(return_dtype),
                nodata_output=output_nodata,
                skip_nodata=skip_nodata,
                output_dims=[["variable"]],
                output_sizes={"variable": a.shape[0]},
                output_dtypes=[a.dtype],
            )
        )


@parametrize_feature_array_types()
@pytest.mark.parametrize(
    "dtypes", [(np.float64, np.dtype(np.float32)), (np.float32, np.dtype(np.float16))]
)
def test_nodata_output_float_precision(
    feature_array_type: type[FeatureArrayType], dtypes: tuple[np.dtype, np.dtype]
):
    """Test that an error is raised when nodata_output is the wrong float precision."""
    # Make sure there's a value to mask in the input array
    a = np.array([[[np.nan]]])
    array = wrap_features(a, type=feature_array_type)
    features = FeatureArray.from_feature_array(array, nodata_input=0)

    chosen_dtype, return_dtype = dtypes
    expected_msg = "Consider casting `nodata_output` to a lower precision"
    with pytest.raises(ValueError, match=expected_msg):
        # Unwrap to force computation for lazy arrays
        unwrap_features(
            features.apply_ufunc_across_features(
                lambda x: np.ones_like(x).astype(return_dtype),
                nodata_output=chosen_dtype(np.nan),
                output_dims=[["variable"]],
                output_sizes={"variable": a.shape[0]},
                output_dtypes=[a.dtype],
            )
        )


@parametrize_feature_array_types()
@pytest.mark.parametrize("skip_nodata", [True, False])
@pytest.mark.parametrize(
    "val_dtypes", [(-1, np.uint8, np.int8), (np.nan, np.int16, np.float64)]
)
def test_nodata_output_with_allow_cast(
    val_dtypes: tuple[int | float, np.dtype, np.dtype],
    feature_array_type: type[FeatureArrayType],
    skip_nodata: bool,
):
    """Test that an unsupported nodata_output value correctly casts if allowed."""
    # Make sure there's a value to mask in the input array
    a = np.array([[[np.nan]]])
    array = wrap_features(a, type=feature_array_type)
    features = FeatureArray.from_feature_array(array, nodata_input=0)

    output_nodata, output_dtype, expected_dtype = val_dtypes
    # Unwrap to force computation for lazy arrays
    result = unwrap_features(
        features.apply_ufunc_across_features(
            lambda x: np.ones_like(x).astype(output_dtype),
            nodata_output=output_nodata,
            skip_nodata=skip_nodata,
            allow_cast=True,
            output_dims=[["variable"]],
            output_sizes={"variable": a.shape[0]},
            output_dtypes=[a.dtype],
        )
    )

    assert result.dtype == expected_dtype


@pytest.mark.parametrize("nodata_output", [np.nan, 42.0])
@parametrize_feature_array_types()
@pytest.mark.parametrize("skip_nodata", [True, False])
def test_nodata_output_set(
    nodata_output: int | float,
    feature_array_type: type[FeatureArrayType],
    skip_nodata: bool,
):
    """Test that NoData in the features are filled or not."""
    nodata_input = 0

    # Encoded NoData and NaN should both be replaced across features with the
    # nodata_output value.
    a = np.array([[[nodata_input, 1, np.nan]]])
    expected_output = np.array([[[nodata_output, 1, nodata_output]]])

    features = FeatureArray.from_feature_array(
        wrap_features(a, type=feature_array_type), nodata_input=nodata_input
    )
    result = features.apply_ufunc_across_features(
        lambda x: x,
        nodata_output=nodata_output,
        skip_nodata=skip_nodata,
        output_dims=[["variable"]],
        output_sizes={"variable": a.shape[0]},
        output_dtypes=[a.dtype],
    )

    assert_array_equal(unwrap_features(result), expected_output)


@pytest.mark.parametrize("n_features", [1, 2])
@parametrize_feature_array_types()
def test_shape_when_ufunc_squeezes_dimension(
    n_features: int, feature_array_type: type[FeatureArrayType]
):
    """Test the output shape when a ufunc squeezes the feature dimension."""
    nodata_input = 0
    nodata_output = -99

    # Insert at least one NoData value to trigger sample skipping
    a = np.full((n_features, 3, 3), 1)
    a[0, 0, 0] = nodata_input

    features = FeatureArray.from_feature_array(
        wrap_features(a, type=feature_array_type), nodata_input=nodata_input
    )
    result = features.apply_ufunc_across_features(
        # Squeeze out the feature dimension (like a single-output predict method would)
        lambda x: x.mean(axis=1, keepdims=False),
        nodata_output=nodata_output,
        skip_nodata=True,
        output_dims=[["variable"]],
        output_sizes={"variable": 1},
        output_dtypes=[a.dtype],
    )

    assert unwrap_features(result).shape == (1, 3, 3)


@parametrize_feature_array_types()
@pytest.mark.parametrize("skip_nodata", [True, False])
def test_warn_when_ufunc_returns_nodata(
    feature_array_type: type[FeatureArrayType], skip_nodata: bool
):
    """Test that a warning is raised when `nodata_output` is returned by the ufunc."""
    nodata_input = 0
    nodata_output = -32768

    # The input features need to contain NoData since the check only occurs when filling
    a = np.full((1, 1, 1), nodata_input, dtype=np.int16)
    features = FeatureArray.from_feature_array(
        wrap_features(a, type=feature_array_type), nodata_input=nodata_input
    )
    with pytest.warns(UserWarning, match=f"{nodata_output} was found in the array"):
        unwrap_features(
            features.apply_ufunc_across_features(
                lambda x: np.full_like(x, nodata_output),
                skip_nodata=skip_nodata,
                nodata_output=nodata_output,
                output_dims=[["variable"]],
                output_sizes={"variable": a.shape[0]},
                output_dtypes=[a.dtype],
            )
        )


@pytest.mark.parametrize("min_samples", [0, 1, 30])
@parametrize_feature_array_types()
def test_ensure_min_samples(
    min_samples: int, feature_array_type: type[FeatureArrayType]
):
    """Test that the correct number of minimum samples are passed."""
    a = np.full((1, 1, 50), np.nan, dtype=np.float64)

    def assert_array_size(x, n):
        assert x.size == n
        return x

    features = FeatureArray.from_feature_array(
        wrap_features(a, type=feature_array_type), nodata_input=0
    )
    result = features.apply_ufunc_across_features(
        lambda x: assert_array_size(x, min_samples),
        skip_nodata=True,
        ensure_min_samples=min_samples,
        output_dims=[["variable"]],
        output_sizes={"variable": a.shape[0]},
        output_dtypes=[a.dtype],
    )

    unwrap_features(result)


@parametrize_feature_array_types()
def test_ensure_too_many_samples(feature_array_type: type[FeatureArrayType]):
    """Test that an error is raised if ensure_min_samples is larger than the array."""
    a = np.full((1, 1, 10), np.nan, dtype=np.float64)

    features = FeatureArray.from_feature_array(
        wrap_features(a, type=feature_array_type), nodata_input=0
    )
    with pytest.raises(ValueError, match="Cannot ensure 50 samples with only 10"):
        unwrap_features(
            features.apply_ufunc_across_features(
                lambda x: x,
                skip_nodata=True,
                ensure_min_samples=50,
                output_dims=[["variable"]],
                output_sizes={"variable": a.shape[0]},
                output_dtypes=[a.dtype],
            )
        )


@parametrize_feature_array_types()
def test_ensure_min_samples_doesnt_overwrite(
    feature_array_type: type[FeatureArrayType],
):
    """
    Test that valid samples aren't overwritten by dummy samples when ensuring size.
    """
    nan_fill = -99.0
    valid_pixel = 1.0
    nodata_output = -32768

    # Create a fully masked array and set the middle pixel to be valid
    a = np.full((1, 3, 1), 0, dtype=np.float64)
    a[0, 1, 0] = valid_pixel

    def check_for_valid_sample(x: np.ndarray):
        # The valid pixel should be used as one of the three minimum samples
        assert_array_equal(x.squeeze(), [nan_fill, valid_pixel, nan_fill])
        return x

    features = FeatureArray.from_feature_array(
        wrap_features(a, type=feature_array_type), nodata_input=0
    )
    result = unwrap_features(
        features.apply_ufunc_across_features(
            check_for_valid_sample,
            skip_nodata=True,
            ensure_min_samples=3,
            nan_fill=nan_fill,
            nodata_output=nodata_output,
            output_dims=[["variable"]],
            output_sizes={"variable": a.shape[0]},
            output_dtypes=[a.dtype],
        )
    )

    # The ufunc returned pixels unchanged, so the valid pixel should be preserved while
    # the dummy pixels were replaced with the `nodata_output`
    assert_array_equal(result.squeeze(), [nodata_output, valid_pixel, nodata_output])


@pytest.mark.parametrize("num_valid", [0, 1, 3])
@pytest.mark.parametrize("nodata_input", [-32768, np.nan])
@parametrize_feature_array_types()
def test_nodata_is_skipped(
    num_valid: int,
    nodata_input: int | float,
    feature_array_type: type[FeatureArrayType],
):
    """Test that NoData values are skipped if the flag is set."""
    # Create a full NoData array and the expected number of valid values
    a = np.full((1, 1, 3), nodata_input, dtype=np.float64)
    for i in range(num_valid):
        a[0, 0, i] = 1

    def assert_array_size(x, n):
        assert x.size == n
        return x

    features = FeatureArray.from_feature_array(
        wrap_features(a, type=feature_array_type), nodata_input=nodata_input
    )
    result = features.apply_ufunc_across_features(
        lambda x: assert_array_size(x, num_valid),
        skip_nodata=True,
        ensure_min_samples=0,
        output_dims=[["variable"]],
        output_sizes={"variable": a.shape[0]},
        output_dtypes=[a.dtype],
    )

    unwrap_features(result)


@parametrize_feature_array_types()
@pytest.mark.parametrize("nan_fill", [None, 42.0])
def test_nan_filled(feature_array_type: type[FeatureArrayType], nan_fill: float | None):
    """Test that NaNs in the features are filled before passing to func."""
    a = np.array([[[1, np.nan]]])
    features = FeatureArray.from_feature_array(
        wrap_features(a, type=feature_array_type)
    )

    def nan_check(x):
        fill_val = nan_fill if nan_fill is not None else np.nan
        assert_array_equal(x.squeeze(), np.array([1, fill_val]))

        return x

    result = features.apply_ufunc_across_features(
        nan_check,
        nan_fill=nan_fill,
        skip_nodata=False,
        output_dims=[["variable"]],
        output_sizes={"variable": a.shape[0]},
        output_dtypes=[a.dtype],
    )

    unwrap_features(result)


@pytest.mark.parametrize("nodata_input", ["test", {}, False], ids=type)
def test_nodata_validates_type(nodata_input):
    """Test that invalid NoData types are recognized."""
    a = np.zeros((3, 2, 2))

    with pytest.raises(
        TypeError, match=f"Invalid type `{type(nodata_input).__name__}`"
    ):
        FeatureArray.from_feature_array(a, nodata_input=nodata_input)


def test_nodata_validates_length():
    """Test that invalid NoData lengths are recognized."""
    n_features = 3
    a = np.zeros((n_features, 2, 2))

    with pytest.raises(
        ValueError, match=f"Expected {n_features} NoData values but got 1"
    ):
        FeatureArray.from_feature_array(a, nodata_input=[-32768])


def test_nodata_single_value():
    """Test that a single NoData value is broadcast to all features."""
    n_features = 3
    nodata_val = -32768
    a = np.zeros((n_features, 2, 2))

    features = FeatureArray.from_feature_array(a, nodata_input=nodata_val)
    assert features.nodata_input.data.tolist() == [nodata_val] * n_features
    assert features.nodata_input.mask.tolist() == [False] * n_features


def test_nodata_multiple_values():
    """Test that multiple NoData values are correctly stored."""
    n_features = 3
    nodata_input = [-32768, 0, 255]
    a = np.zeros((n_features, 2, 2))

    features = FeatureArray.from_feature_array(a, nodata_input=nodata_input)
    assert features.nodata_input.data.tolist() == nodata_input
    assert features.nodata_input.mask.tolist() == [False] * n_features


def test_nodata_input_unsupported_dtype():
    """
    When input NoData values don't fit in the feature array, they should be skipped and
    a warning should be raised.
    """
    # Place a missing None value at the start of the list to ensure that unsupported
    # values are correctly indexed after the missing value is removed
    nodata_input = [None, -1, 0, 0.99, 42, np.nan]
    a = np.zeros((len(nodata_input), 2, 2), dtype=np.uint8)

    # Note that None should be missing, not unsupported, so it shouldn't appear in the
    # warning message.
    expected_msg = re.escape(
        "The selected or inferred NoData value(s) [-1, 0.99, nan] cannot be safely "
        "cast to the feature array dtype uint8, so they will be ignored."
    )
    with pytest.warns(UserWarning, match=expected_msg):
        fa = FeatureArray.from_feature_array(a, nodata_input=nodata_input)

    assert fa.nodata_input.data.tolist() == [0, 0, 0, 0, 42, 0]
    assert fa.nodata_input.mask.tolist() == [True, True, False, True, False, True]


@pytest.mark.parametrize("array_dtype", [np.uint8, np.float32])
def test_nodata_input_masks_missing_values(array_dtype: np.dtype):
    """Test that None is treated as a missing value in the NoData."""
    nodata_input = [None, 255, None, 0]
    a = np.ones((len(nodata_input), 2, 2), dtype=array_dtype)
    features = FeatureArray.from_feature_array(a, nodata_input=nodata_input)

    # nodata_input should match the array dtype
    assert features.nodata_input.dtype == a.dtype

    # Missing values (None) should be replaced with zeroes and masked
    assert features.nodata_input.data.tolist() == [0, 255, 0, 0]
    assert features.nodata_input.mask.tolist() == [True, False, True, False]


@pytest.mark.parametrize("nodata_input", [MissingType.MISSING, None, 255])
def test_nodata_dataarray_fillvalue(nodata_input):
    """Test that a _FillValue in a DataArray is broadcast if NoData is not provided."""
    n_features = 3
    fill_val = -99

    da = xr.DataArray(np.ones((n_features, 2, 2))).assign_attrs(
        {"_FillValue": fill_val}
    )
    features = FeatureArray.from_feature_array(da, nodata_input=nodata_input)

    # _FillValue should only be used if nodata_input is not provided (including None)
    if nodata_input == MissingType.MISSING:
        assert features.nodata_input.data.tolist() == [fill_val] * n_features
        assert features.nodata_input.mask.tolist() == [False] * n_features
    # None should be treated as all missing values
    elif nodata_input is None:
        assert features.nodata_input.data.tolist() == [0] * n_features
        assert features.nodata_input.mask.tolist() == [True] * n_features
    # Otherwise, nodata_input should be used
    else:
        assert features.nodata_input.data.tolist() == [nodata_input] * n_features
        assert features.nodata_input.mask.tolist() == [False] * n_features


@pytest.mark.parametrize("nodata_input", [MissingType.MISSING, None, -32768])
@pytest.mark.parametrize(
    "fill_vals",
    [[1, 2, 3], [None, 1, None], [None, None, None]],
    ids=["no_none", "some_none", "all_none"],
)
def test_nodata_dataset_infers_fillvalues(nodata_input, fill_vals):
    """Test that feature-wise _FillValues are correctly inferred."""
    n_features = len(fill_vals)
    das = [
        xr.DataArray(np.ones((n_features, 2, 2)), name=f"da{i}")
        for i in range(n_features)
    ]

    # Assign per-feature fill values
    for i, fill_val in enumerate(fill_vals):
        # A missing _FillValue should be inferred as None
        if fill_val is None:
            continue
        das[i] = das[i].assign_attrs({"_FillValue": fill_val})

    # A global _FillValue should be ignored in every case
    ds = xr.merge(das).assign_attrs({"_FillValue": -999})
    features = FeatureArray.from_feature_array(ds, nodata_input=nodata_input)

    # nodata_input is missing: use non-null _FillValue for each feature
    if nodata_input is MissingType.MISSING:
        assert features.nodata_input.data.tolist() == [val or 0 for val in fill_vals]
        assert features.nodata_input.mask.tolist() == [val is None for val in fill_vals]

    # nodata_input is None: ignore _FillValue and treat all as missing
    elif nodata_input is None:
        assert features.nodata_input.data.tolist() == [0] * n_features
        assert features.nodata_input.mask.tolist() == [True] * n_features

    # nodata_input is non-null: ignore _FillValue and use provided value
    else:
        assert features.nodata_input.data.tolist() == [nodata_input] * n_features
        assert features.nodata_input.mask.tolist() == [False] * n_features


@pytest.mark.parametrize("nodata_output", [np.nan, 0, -32768])
def test_nodata_output_set_in_dataarray_attrs(nodata_output: int | float):
    """Test that the output NoData value is stored as the _FillValue for a DataArray."""
    a = np.array([[[1, 2, 3]]])
    features = FeatureArray.from_feature_array(
        wrap_features(a, type=xr.DataArray), nodata_input=0
    )

    result = features.apply_ufunc_across_features(
        lambda x: x,
        nodata_output=nodata_output,
        output_dims=[["variable"]],
        output_sizes={"variable": a.shape[0]},
        output_dtypes=[a.dtype],
    )

    if np.isnan(nodata_output):
        # NaN should not be stored as a _FillValue
        assert "_FillValue" not in result.attrs
    else:
        assert result.attrs.get("_FillValue") == nodata_output


@pytest.mark.parametrize("nodata_output", [np.nan, 0, -32768])
def test_nodata_output_set_in_dataset_attrs(nodata_output: int | float):
    """Test that the output NoData value is stored as the _FillValue for a DataArray."""
    a = np.array([[[1, 2, 3]]])
    features = FeatureArray.from_feature_array(
        wrap_features(a, type=xr.Dataset), nodata_input=0
    )

    result = features.apply_ufunc_across_features(
        lambda x: x,
        nodata_output=nodata_output,
        output_dims=[["variable"]],
        output_sizes={"variable": a.shape[0]},
        output_dtypes=[a.dtype],
    )

    for var in result.data_vars:
        # NaN should not be stored as a _FillValue
        if np.isnan(nodata_output):
            assert "_FillValue" not in result[var].attrs
        else:
            assert result[var].attrs.get("_FillValue") == nodata_output


@pytest.mark.parametrize("thread_limit", [1, 2, 8])
def test_inner_thread_limit(thread_limit: int):
    """
    Test that the inner thread limit is set within applied ufuncs.

    Note that threadpool limits aren't constrained by hardware capabilities, so this
    test should work regardless of the number of available CPU cores.
    """
    features = FeatureArray.from_feature_array(np.zeros((1, 1)))
    original_thread_info = threadpoolctl.threadpool_info()

    def check_inner_threads(x):
        inner_thread_info = threadpoolctl.threadpool_info()
        for lib in inner_thread_info:
            assert lib["num_threads"] == thread_limit
        return x

    features.apply_ufunc_across_features(
        check_inner_threads,
        output_dims=[["variable"]],
        inner_thread_limit=thread_limit,
    )

    # Ensure that the threadpool limits don't leak outside the ufunc
    assert threadpoolctl.threadpool_info() == original_thread_info


@parametrize_feature_array_types()
def test_wrappers(feature_array_type):
    """Confirm that the test wrappers function as expected."""
    array = np.random.rand(3, 32, 16)

    wrapped = wrap_features(array, type=feature_array_type)
    assert isinstance(wrapped, feature_array_type)
    assert_array_equal(unwrap_features(wrapped), array)

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
import threadpoolctl
import xarray as xr
from numpy.testing import assert_array_equal

from sklearn_raster.features import FeatureArray
from sklearn_raster.types import FeatureArrayType
from sklearn_raster.ufunc import FeaturewiseUfunc
from sklearn_raster.utils.features import get_minimum_precise_numeric_dtype

from .feature_utils import (
    parametrize_feature_array_types,
    unwrap_features,
    wrap_features,
)


@parametrize_feature_array_types()
@pytest.mark.parametrize("n_inputs", [1, 2, 5])
def test_ufunc_with_multiple_inputs(
    feature_array_type: type[FeatureArrayType],
    n_inputs: int,
):
    """Ensure that ufuncs are called correctly with multiple inputs."""
    a = np.ones((1, 2, 2))
    input_arrays = [a.copy() for _ in range(n_inputs)]
    input_features = [
        FeatureArray.from_feature_array(wrap_features(arr, type=feature_array_type))
        for arr in input_arrays
    ]

    ufunc = FeaturewiseUfunc(
        lambda *args: sum(args),
        output_dims=[["variable"]],
        output_sizes={"variable": 1},
        output_dtypes=[input_arrays[0].dtype],
    )
    result = unwrap_features(ufunc(*input_features))
    expected_output = np.full((1, 2, 2), n_inputs)

    assert_array_equal(result, expected_output)


@parametrize_feature_array_types()
@pytest.mark.parametrize("skip_nodata", [True, False])
def test_ufunc_propagates_nodata_from_inputs(
    feature_array_type: type[FeatureArrayType], skip_nodata: bool
):
    """Ensure that NoData values in input arrays are propagated to the output."""
    d1 = np.asarray(
        [
            [
                [1, 0],
                [1, 1],
            ]
        ]
    )
    d2 = np.asarray(
        [
            [
                [1, 1],
                [1, 0],
            ]
        ]
    )
    d3 = np.asarray(
        [
            [
                [1, 1],
                [0, 0],
            ]
        ]
    )

    f1 = FeatureArray.from_feature_array(
        wrap_features(d1, type=feature_array_type), nodata_input=0
    )
    f2 = FeatureArray.from_feature_array(
        wrap_features(d2, type=feature_array_type), nodata_input=0
    )
    f3 = FeatureArray.from_feature_array(
        wrap_features(d3, type=feature_array_type), nodata_input=0
    )

    ufunc = FeaturewiseUfunc(
        lambda x, y, z: x + y + z,
        output_dims=[["variable"]],
        output_sizes={"variable": 1},
        output_dtypes=[d1.dtype],
    )
    result = unwrap_features(
        ufunc(f1, f2, f3, skip_nodata=skip_nodata, nodata_output=-99)
    )
    expected_output = np.asarray(
        [
            [
                [3, -99],
                [-99, -99],
            ]
        ]
    )

    assert_array_equal(result, expected_output)


def test_ufunc_raises_with_no_inputs():
    """Test that calling a ufunc without any arrays raises an error."""
    ufunc = FeaturewiseUfunc(
        lambda: None,
        output_dims=[["variable"]],
        output_sizes={"variable": 1},
        output_dtypes=[np.float64],
    )
    with pytest.raises(ValueError, match="requires at least one feature array input"):
        ufunc()


@pytest.mark.parametrize("dims", [["foo", "bar"], ["foo", "foo"]])
def test_ufunc_sets_explicit_output_coords(dims):
    """Test that output coordinates are set correctly when specified."""
    a = np.zeros((3, 2, 2))
    features = FeatureArray.from_feature_array(wrap_features(a, type=xr.DataArray))

    ufunc = FeaturewiseUfunc(
        lambda x: (x, x),
        output_dims=[[dims[0]], [dims[1]]],
        output_sizes={dims[0]: 3, dims[1]: 3},
        output_dtypes=[a.dtype, a.dtype],
        output_coords=[{dims[0]: ["a0", "a1", "a2"]}, {dims[1]: ["b0", "b1", "b2"]}],
    )
    r1, r2 = ufunc(features)

    assert r1.coords[dims[0]].values.tolist() == ["a0", "a1", "a2"]
    assert r2.coords[dims[1]].values.tolist() == ["b0", "b1", "b2"]


def test_ufunc_sets_implicit_output_coords():
    """Test that output coordinates are set correctly when not specified."""
    a = np.zeros((3, 2, 2))
    features = FeatureArray.from_feature_array(wrap_features(a, type=xr.DataArray))

    ufunc = FeaturewiseUfunc(
        lambda x: (x, x),
        output_dims=[["foo"], ["bar"]],
        output_sizes={"foo": 3, "bar": 3},
        output_dtypes=[a.dtype, a.dtype],
    )
    r1, r2 = ufunc(features)

    assert r1.coords["foo"].values.tolist() == [0, 1, 2]
    assert r2.coords["bar"].values.tolist() == [0, 1, 2]


@pytest.mark.parametrize(
    ("input_types", "expected_type"),
    [
        ((xr.Dataset, xr.DataArray), (xr.Dataset)),
        ((xr.Dataset, pd.DataFrame), (xr.Dataset)),
        ((xr.DataArray, pd.DataFrame), (xr.DataArray)),
    ],
    ids=[
        "dataset_over_dataarray",
        "dataset_over_dataframe",
        "dataarray_over_dataframe",
    ],
)
@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
def test_ufunc_return_type_priority(
    input_types: tuple[type, ...], expected_type: type, reverse: bool
):
    n_features = 3
    base = np.zeros((n_features, 10), dtype=np.float64)

    wrapped_features = [wrap_features(base, type=t) for t in input_types]
    features = [
        FeatureArray.from_feature_array(raw, nodata_input=None)
        for raw in wrapped_features
    ]
    if reverse:
        features = list(reversed(features))

    ufunc = FeaturewiseUfunc(
        lambda *x: x[0],
        output_dims=[["variable"]],
        output_sizes={"variable": n_features},
        output_coords=[{"variable": [f"b{i}" for i in range(n_features)]}],
        output_dtypes=[np.dtype(np.float64)],
    )

    assert isinstance(ufunc(*features), expected_type)


@parametrize_feature_array_types()
@pytest.mark.parametrize("skip_nodata", [True, False])
def test_ufunc_does_not_mutate_input(
    feature_array_type: type[FeatureArrayType], skip_nodata: bool
):
    """Ensure that applying a ufunc to features doesn't mutate the original array."""
    a = np.array([[[0, 1]], [[1, np.nan]]])
    original_array = a.copy()

    array = wrap_features(a, type=feature_array_type)

    features = FeatureArray.from_feature_array(array, nodata_input=0)
    FeaturewiseUfunc(
        lambda x: x * 2.0,
        output_dims=[["variable"]],
        output_sizes={"variable": a.shape[0]},
        output_dtypes=[a.dtype],
    )(features, skip_nodata=skip_nodata)

    assert_array_equal(a, original_array)


@parametrize_feature_array_types()
@pytest.mark.parametrize("skip_nodata", [True, False])
@pytest.mark.parametrize(
    "val_dtype", [(-1, np.dtype(np.uint8)), (np.nan, np.dtype(np.int16))]
)
def test_ufunc_raises_with_unsupported_nodata_output_dtype(
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
    ufunc = FeaturewiseUfunc(
        lambda x: np.ones_like(x).astype(return_dtype),
        output_dims=[["variable"]],
        output_sizes={"variable": a.shape[0]},
        output_dtypes=[a.dtype],
    )
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        # Unwrap to force computation for lazy arrays
        unwrap_features(
            ufunc(
                features,
                nodata_output=output_nodata,
                skip_nodata=skip_nodata,
            )
        )


@parametrize_feature_array_types()
@pytest.mark.parametrize(
    "dtypes", [(np.float64, np.dtype(np.float32)), (np.float32, np.dtype(np.float16))]
)
def test_ufunc_raises_with_unsupported_nodata_output_float_precision(
    feature_array_type: type[FeatureArrayType], dtypes: tuple[np.dtype, np.dtype]
):
    """Test that an error is raised when nodata_output is the wrong float precision."""
    # Make sure there's a value to mask in the input array
    a = np.array([[[np.nan]]])
    array = wrap_features(a, type=feature_array_type)
    features = FeatureArray.from_feature_array(array, nodata_input=0)

    chosen_dtype, return_dtype = dtypes
    expected_msg = "Consider casting `nodata_output` to a lower precision"
    ufunc = FeaturewiseUfunc(
        lambda x: np.ones_like(x).astype(return_dtype),
        output_dims=[["variable"]],
        output_sizes={"variable": a.shape[0]},
        output_dtypes=[a.dtype],
    )
    with pytest.raises(ValueError, match=expected_msg):
        # Unwrap to force computation for lazy arrays
        unwrap_features(
            ufunc(
                features,
                nodata_output=chosen_dtype(np.nan),
            )
        )


@parametrize_feature_array_types()
@pytest.mark.parametrize("skip_nodata", [True, False])
@pytest.mark.parametrize(
    "val_dtypes", [(-1, np.uint8, np.int8), (np.nan, np.int16, np.float64)]
)
def test_ufunc_allows_casting_of_unsupported_nodata_output(
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
    ufunc = FeaturewiseUfunc(
        lambda x: np.ones_like(x).astype(output_dtype),
        output_dims=[["variable"]],
        output_sizes={"variable": a.shape[0]},
        output_dtypes=[a.dtype],
    )
    result = unwrap_features(
        ufunc(
            features,
            nodata_output=output_nodata,
            skip_nodata=skip_nodata,
            allow_cast=True,
        )
    )

    assert result.dtype == expected_dtype


@parametrize_feature_array_types()
@pytest.mark.parametrize("input_dtype", [np.uint8, np.float32])
def test_ufunc_skips_nodata_output_validation_if_unmasked(
    feature_array_type: type[FeatureArrayType],
    input_dtype: np.dtype,
):
    """If the input doesn't contain NoData, we shouldn't test nodata_output's dtype."""
    # Make a feature array without any NoData
    a = np.ones((4, 4), dtype=input_dtype)
    array = wrap_features(a, type=feature_array_type)
    features = FeatureArray.from_feature_array(array, nodata_input=0)

    ufunc = FeaturewiseUfunc(
        lambda x: x,
        output_dims=[["variable"]],
        output_sizes={"variable": a.shape[0]},
        output_dtypes=[a.dtype],
    )
    # Unwrap to force computation for lazy arrays
    unwrap_features(
        ufunc(
            features,
            # Set a value that doesn't fit in the feature array type
            nodata_output=np.float64(np.nan),
            skip_nodata=True,
            allow_cast=False,
        )
    )


@pytest.mark.parametrize("nodata_output", [np.nan, 42.0])
@parametrize_feature_array_types()
@pytest.mark.parametrize("skip_nodata", [True, False])
def test_ufunc_sets_nodata_output(
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
    ufunc = FeaturewiseUfunc(
        lambda x: x,
        output_dims=[["variable"]],
        output_sizes={"variable": a.shape[0]},
        output_dtypes=[a.dtype],
    )
    result = ufunc(
        features,
        nodata_output=nodata_output,
        skip_nodata=skip_nodata,
    )

    assert_array_equal(unwrap_features(result), expected_output)


@pytest.mark.parametrize("n_features", [1, 2])
@parametrize_feature_array_types()
def test_ufunc_shape_when_squeezing_dimension(
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
    ufunc = FeaturewiseUfunc(
        lambda x: x.mean(axis=1, keepdims=False),
        # Squeeze out the feature dimension (like a single-output predict method would)
        output_dims=[["variable"]],
        output_sizes={"variable": 1},
        output_dtypes=[a.dtype],
    )
    result = ufunc(
        features,
        nodata_output=nodata_output,
        skip_nodata=True,
    )

    assert unwrap_features(result).shape == (1, 3, 3)


@parametrize_feature_array_types()
@pytest.mark.parametrize("skip_nodata", [True, False])
def test_ufunc_warns_when_returning_nodata(
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
    ufunc = FeaturewiseUfunc(
        lambda x: np.full_like(x, nodata_output),
        output_dims=[["variable"]],
        output_sizes={"variable": a.shape[0]},
        output_dtypes=[a.dtype],
    )
    with pytest.warns(UserWarning, match=f"{nodata_output} was found in the array"):
        unwrap_features(
            ufunc(
                features,
                skip_nodata=skip_nodata,
                nodata_output=nodata_output,
            )
        )


@pytest.mark.parametrize("min_samples", [0, 1, 30])
@parametrize_feature_array_types()
def test_ufunc_ensures_min_samples(
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
    ufunc = FeaturewiseUfunc(
        lambda x: assert_array_size(x, min_samples),
        output_dims=[["variable"]],
        output_sizes={"variable": a.shape[0]},
        output_dtypes=[a.dtype],
    )
    result = ufunc(
        features,
        skip_nodata=True,
        ensure_min_samples=min_samples,
    )

    unwrap_features(result)


@parametrize_feature_array_types()
def test_ufunc_raises_on_ensure_too_many_samples(
    feature_array_type: type[FeatureArrayType],
):
    """Test that an error is raised if ensure_min_samples is larger than the array."""
    a = np.full((1, 1, 10), np.nan, dtype=np.float64)

    features = FeatureArray.from_feature_array(
        wrap_features(a, type=feature_array_type), nodata_input=0
    )
    ufunc = FeaturewiseUfunc(
        lambda x: x,
        output_dims=[["variable"]],
        output_sizes={"variable": a.shape[0]},
        output_dtypes=[a.dtype],
    )
    with pytest.raises(ValueError, match="Cannot ensure 50 samples with only 10"):
        unwrap_features(
            ufunc(
                features,
                skip_nodata=True,
                ensure_min_samples=50,
            )
        )


@parametrize_feature_array_types()
def test_ufunc_ensure_min_samples_does_not_overwrite(
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
    ufunc = FeaturewiseUfunc(
        check_for_valid_sample,
        output_dims=[["variable"]],
        output_sizes={"variable": a.shape[0]},
        output_dtypes=[a.dtype],
    )
    result = unwrap_features(
        ufunc(
            features,
            skip_nodata=True,
            ensure_min_samples=3,
            nan_fill=nan_fill,
            nodata_output=nodata_output,
        )
    )

    # The ufunc returned pixels unchanged, so the valid pixel should be preserved while
    # the dummy pixels were replaced with the `nodata_output`
    assert_array_equal(result.squeeze(), [nodata_output, valid_pixel, nodata_output])


@pytest.mark.parametrize("num_valid", [0, 1, 3])
@pytest.mark.parametrize("nodata_input", [-32768, np.nan])
@parametrize_feature_array_types()
def test_ufunc_skips_nodata(
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
    ufunc = FeaturewiseUfunc(
        lambda x: assert_array_size(x, num_valid),
        output_dims=[["variable"]],
        output_sizes={"variable": a.shape[0]},
        output_dtypes=[a.dtype],
    )
    result = ufunc(
        features,
        skip_nodata=True,
        ensure_min_samples=0,
    )

    unwrap_features(result)


@parametrize_feature_array_types()
@pytest.mark.parametrize("nan_fill", [None, 42.0])
def test_ufunc_fills_nans(
    feature_array_type: type[FeatureArrayType], nan_fill: float | None
):
    """Test that NaNs in the features are filled before passing to func."""
    a = np.array([[[1, np.nan]]])
    features = FeatureArray.from_feature_array(
        wrap_features(a, type=feature_array_type)
    )

    def nan_check(x):
        fill_val = nan_fill if nan_fill is not None else np.nan
        assert_array_equal(x.squeeze(), np.array([1, fill_val]))

        return x

    ufunc = FeaturewiseUfunc(
        nan_check,
        output_dims=[["variable"]],
        output_sizes={"variable": a.shape[0]},
        output_dtypes=[a.dtype],
    )
    result = ufunc(
        features,
        nan_fill=nan_fill,
        skip_nodata=False,
    )

    unwrap_features(result)


@pytest.mark.parametrize("nodata_output", [np.nan, 0, -32768])
def test_ufunc_sets_dataarray_fillvalue(nodata_output: int | float):
    """Test that the output NoData value is stored as the _FillValue for a DataArray."""
    a = np.array([[[1, 2, 3]]])
    features = FeatureArray.from_feature_array(
        wrap_features(a, type=xr.DataArray), nodata_input=0
    )

    ufunc = FeaturewiseUfunc(
        lambda x: x,
        output_dims=[["variable"]],
        output_sizes={"variable": a.shape[0]},
        output_dtypes=[a.dtype],
    )
    result = ufunc(
        features,
        nodata_output=nodata_output,
    )

    if np.isnan(nodata_output):
        # NaN should not be stored as a _FillValue
        assert "_FillValue" not in result.attrs
    else:
        assert result.attrs.get("_FillValue") == nodata_output


@pytest.mark.parametrize("nodata_output", [np.nan, 0, -32768])
def test_ufunc_sets_dataset_fillvalue(nodata_output: int | float):
    """
    Test that the output NoData value is stored as the _FillValue for each variable in a
    Dataset.
    """
    a = np.array([[[1, 2, 3]]])
    features = FeatureArray.from_feature_array(
        wrap_features(a, type=xr.Dataset), nodata_input=0
    )

    ufunc = FeaturewiseUfunc(
        lambda x: x,
        output_dims=[["variable"]],
        output_sizes={"variable": a.shape[0]},
        output_dtypes=[a.dtype],
    )
    result = ufunc(
        features,
        nodata_output=nodata_output,
    )

    for var in result.data_vars:
        # NaN should not be stored as a _FillValue
        if np.isnan(nodata_output):
            assert "_FillValue" not in result[var].attrs
        else:
            assert result[var].attrs.get("_FillValue") == nodata_output


@parametrize_feature_array_types(feature_array_types=[xr.DataArray, xr.Dataset])
def test_ufunc_raises_on_missing_output_sizes(
    feature_array_type: type[FeatureArrayType],
):
    """Test that missing output sizes raise a helpful error."""
    a = wrap_features(np.array([[[1, 2]]]), type=feature_array_type)
    features = FeatureArray.from_feature_array(a)
    msg = (
        "dimension 'variable' in 'output_core_dims' needs corresponding (dim, size) in "
        "'output_sizes'"
    )
    ufunc = FeaturewiseUfunc(
        lambda x: x,
        output_dims=[["variable"]],
        output_sizes=None,
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        ufunc(features)


def test_ufunc_raises_on_missing_arrays():
    """Test that applying a ufunc without any arrays raises an error."""
    ufunc = FeaturewiseUfunc(
        lambda x: x,
        output_dims=[["variable"]],
    )
    with pytest.raises(ValueError, match="requires at least one feature array input"):
        ufunc()


def test_ufunc_raises_on_mismatched_feature_dim_names():
    """Test that applying a ufunc with mismatched feature dimension names raises."""
    a = wrap_features(np.array([[[1, 2]]]), type=xr.DataArray).rename(
        {"variable": "feat1"}
    )
    b = wrap_features(np.array([[[3, 4]]]), type=xr.DataArray).rename(
        {"variable": "feat2"}
    )

    features_a = FeatureArray.from_feature_array(a)
    features_b = FeatureArray.from_feature_array(b)

    ufunc = FeaturewiseUfunc(
        lambda x, y: x + y,
        output_dims=[["variable"]],
    )
    with pytest.raises(
        ValueError,
        match="All input feature arrays must share the same feature dimension name",
    ):
        ufunc(features_a, features_b)


def test_ufunc_raises_on_mismatched_shapes():
    """Test that applying a ufunc with mismatched number of samples raises."""
    a = np.zeros((3, 10))
    b = np.zeros((3, 11))

    features_a = FeatureArray.from_feature_array(a)
    features_b = FeatureArray.from_feature_array(b)

    ufunc = FeaturewiseUfunc(
        lambda x, y: x + y,
        output_dims=[["variable"]],
    )
    with pytest.raises(ValueError, match="All arrays must have the same shape"):
        ufunc(features_a, features_b)


def test_ufunc_broadcasts_feature_dimensions():
    """Test that ufuncs can broadcast the feature dimension."""
    a = np.zeros((1, 10))
    b = np.zeros((5, 10))

    features_a = FeatureArray.from_feature_array(a)
    features_b = FeatureArray.from_feature_array(b)

    ufunc = FeaturewiseUfunc(
        lambda x, y: x + y,
        output_dims=[["variable"]],
    )
    result = ufunc(features_a, features_b)
    assert result.shape == (5, 10)


@pytest.mark.parametrize("thread_limit", [1, 2, 8])
def test_ufunc_applied_with_inner_thread_limit(thread_limit: int):
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

    ufunc = FeaturewiseUfunc(
        check_inner_threads,
        output_dims=[["variable"]],
    )
    ufunc(
        features,
        inner_thread_limit=thread_limit,
    )

    # Ensure that the threadpool limits don't leak outside the ufunc
    assert threadpoolctl.threadpool_info() == original_thread_info

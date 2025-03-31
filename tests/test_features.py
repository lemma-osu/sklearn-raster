"""Test the features module."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

from sklearn_raster.features import FeatureArray
from sklearn_raster.types import FeatureType

from .feature_utils import (
    parametrize_feature_types,
    unwrap_features,
    wrap_features,
)


@parametrize_feature_types()
@pytest.mark.parametrize("skip_nodata", [True, False])
def test_input_array_not_mutated(feature_type: type[FeatureType], skip_nodata: bool):
    """Ensure that applying a ufunc to features doesn't mutate the original array."""
    a = np.array([[[0, 1]], [[1, np.nan]]])
    original_array = a.copy()

    array = wrap_features(a, type=feature_type)

    features = FeatureArray.from_features(array, nodata_input=0)
    features.apply_ufunc_across_features(
        lambda x: x * 2.0,
        skip_nodata=skip_nodata,
        output_dims=[["variable"]],
        output_sizes={"variable": a.shape[0]},
        output_dtypes=[a.dtype],
    )

    assert_array_equal(a, original_array)


@parametrize_feature_types()
@pytest.mark.parametrize("skip_nodata", [True, False])
@pytest.mark.parametrize("val_dtype", [(-1, np.uint8), (np.nan, np.int16)])
def test_nodata_output_with_unsupported_dtype(
    val_dtype: tuple[int | float, np.dtype],
    feature_type: type[FeatureType],
    skip_nodata: bool,
):
    """Test that an unsupported nodata_output value raises an error."""
    # Make sure there's a value to mask in the input array
    a = np.array([[[np.nan]]])
    array = wrap_features(a, type=feature_type)
    features = FeatureArray.from_features(array, nodata_input=0)

    output_nodata, output_dtype = val_dtype
    with pytest.raises(ValueError, match="does not fit in the array dtype"):
        # Unwrap to force computation for lazy arrays
        unwrap_features(
            features.apply_ufunc_across_features(
                lambda x: np.ones_like(x).astype(output_dtype),
                nodata_output=output_nodata,
                skip_nodata=skip_nodata,
                output_dims=[["variable"]],
                output_sizes={"variable": a.shape[0]},
                output_dtypes=[a.dtype],
            )
        )


@parametrize_feature_types()
@pytest.mark.parametrize("skip_nodata", [True, False])
@pytest.mark.parametrize(
    "val_dtypes", [(-1, np.uint8, np.int8), (np.nan, np.int16, np.float64)]
)
def test_nodata_output_with_allow_cast(
    val_dtypes: tuple[int | float, np.dtype, np.dtype],
    feature_type: type[FeatureType],
    skip_nodata: bool,
):
    """Test that an unsupported nodata_output value correctly casts if allowed."""
    # Make sure there's a value to mask in the input array
    a = np.array([[[np.nan]]])
    array = wrap_features(a, type=feature_type)
    features = FeatureArray.from_features(array, nodata_input=0)

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
@parametrize_feature_types()
@pytest.mark.parametrize("skip_nodata", [True, False])
def test_nodata_output_set(
    nodata_output: int | float, feature_type: type[FeatureType], skip_nodata: bool
):
    """Test that NoData in the features are filled or not."""
    nodata_input = 0

    # Encoded NoData and NaN should both be replaced across features with the
    # nodata_output value.
    a = np.array([[[nodata_input, 1, np.nan]]])
    expected_output = np.array([[[nodata_output, 1, nodata_output]]])

    features = FeatureArray.from_features(
        wrap_features(a, type=feature_type), nodata_input=nodata_input
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
@parametrize_feature_types()
def test_shape_when_ufunc_squeezes_dimension(
    n_features: int, feature_type: type[FeatureType]
):
    """Test the output shape when a ufunc squeezes the feature dimension."""
    nodata_input = 0
    nodata_output = -99

    # Insert at least one NoData value to trigger sample skipping
    a = np.full((n_features, 3, 3), 1)
    a[0, 0, 0] = nodata_input

    features = FeatureArray.from_features(
        wrap_features(a, type=feature_type), nodata_input=nodata_input
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


@parametrize_feature_types()
@pytest.mark.parametrize("skip_nodata", [True, False])
def test_warn_when_ufunc_returns_nodata(
    feature_type: type[FeatureType], skip_nodata: bool
):
    """Test that a warning is raised when `nodata_output` is returned by the ufunc."""
    nodata_input = 0
    nodata_output = -32768

    # The input features need to contain NoData since the check only occurs when filling
    a = np.full((1, 1, 1), nodata_input, dtype=np.int16)
    features = FeatureArray.from_features(
        wrap_features(a, type=feature_type), nodata_input=nodata_input
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
@parametrize_feature_types()
def test_ensure_min_samples(min_samples: int, feature_type: type[FeatureType]):
    """Test that the correct number of minimum samples are passed."""
    a = np.full((1, 1, 50), np.nan, dtype=np.float64)

    def assert_array_size(x, n):
        assert x.size == n
        return x

    features = FeatureArray.from_features(
        wrap_features(a, type=feature_type), nodata_input=0
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


@parametrize_feature_types()
def test_ensure_too_many_samples(feature_type: type[FeatureType]):
    """Test that an error is raised if ensure_min_samples is larger than the array."""
    a = np.full((1, 1, 10), np.nan, dtype=np.float64)

    features = FeatureArray.from_features(
        wrap_features(a, type=feature_type), nodata_input=0
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


@parametrize_feature_types()
def test_ensure_min_samples_doesnt_overwrite(feature_type: type[FeatureType]):
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

    features = FeatureArray.from_features(
        wrap_features(a, type=feature_type), nodata_input=0
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
@parametrize_feature_types()
def test_nodata_is_skipped(
    num_valid: int, nodata_input: int | float, feature_type: type[FeatureType]
):
    """Test that NoData values are skipped if the flag is set."""
    # Create a full NoData array and the expected number of valid values
    a = np.full((1, 1, 3), nodata_input, dtype=np.float64)
    for i in range(num_valid):
        a[0, 0, i] = 1

    def assert_array_size(x, n):
        assert x.size == n
        return x

    features = FeatureArray.from_features(
        wrap_features(a, type=feature_type), nodata_input=nodata_input
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


@parametrize_feature_types()
@pytest.mark.parametrize("nan_fill", [None, 42.0])
def test_nan_filled(feature_type: type[FeatureType], nan_fill: float | None):
    """Test that NaNs in the features are filled before passing to func."""
    a = np.array([[[1, np.nan]]])
    features = FeatureArray.from_features(wrap_features(a, type=feature_type))

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


def test_skip_nodata_mask_if_unneeded():
    """If features are not float and nodata isn't specified, there should be no mask."""
    a = np.ones((3, 2, 2), dtype=int)
    features = FeatureArray.from_features(a, nodata_input=None)

    assert features.nodata_input is None


@pytest.mark.parametrize("nodata_input", ["test", {}, False], ids=type)
def test_nodata_validates_type(nodata_input):
    """Test that invalid NoData types are recognized."""
    a = np.zeros((3, 2, 2))

    with pytest.raises(
        TypeError, match=f"Invalid type `{type(nodata_input).__name__}`"
    ):
        FeatureArray.from_features(a, nodata_input=nodata_input)


def test_nodata_validates_length():
    """Test that invalid NoData lengths are recognized."""
    n_features = 3
    a = np.zeros((n_features, 2, 2))

    with pytest.raises(
        ValueError, match=f"Expected {n_features} NoData values but got 1"
    ):
        FeatureArray.from_features(a, nodata_input=[-32768])


def test_nodata_single_value():
    """Test that a single NoData value is broadcast to all features."""
    n_features = 3
    nodata_val = -32768
    a = np.zeros((n_features, 2, 2))

    features = FeatureArray.from_features(a, nodata_input=nodata_val)
    assert features.nodata_input.tolist() == [nodata_val] * n_features


def test_nodata_multiple_values():
    """Test that multiple NoData values are correctly stored."""
    n_features = 3
    nodata_input = [-32768, 0, 255]
    a = np.zeros((n_features, 2, 2))

    features = FeatureArray.from_features(a, nodata_input=nodata_input)
    assert features.nodata_input.tolist() == nodata_input


@pytest.mark.parametrize("nodata_input", [None, -32768])
def test_nodata_dataarray_fillvalue(nodata_input):
    """Test that a _FillValue in a DataArray is broadcast if NoData is not provided."""
    n_features = 3
    fill_val = -99

    da = xr.DataArray(np.ones((n_features, 2, 2))).assign_attrs(
        {"_FillValue": fill_val}
    )
    features = FeatureArray.from_features(da, nodata_input=nodata_input)

    # _FillValue should be ignored if nodata_input is provided
    if nodata_input is not None:
        assert features.nodata_input.tolist() == [nodata_input] * n_features
    else:
        assert features.nodata_input.tolist() == [fill_val] * n_features


@pytest.mark.parametrize(
    "nodata_input", [None, -32768], ids=["without_nodata", "with_nodata"]
)
@pytest.mark.parametrize(
    "fill_vals", [[1, 2, 3], [None, 1, None]], ids=["no_nones", "some_nones"]
)
def test_nodata_dataset_some_fillvalues(nodata_input, fill_vals):
    """Test that feature-wise _FillValues are applied if some exist"""
    n_features = 3
    das = [xr.DataArray(np.ones((n_features, 2, 2))) for i in range(n_features)]

    # Assign per-feature fill values
    for i, fill_val in enumerate(fill_vals):
        das[i] = das[i].assign_attrs({"_FillValue": fill_val}).rename(i)

    ds = xr.merge(das)
    features = FeatureArray.from_features(ds, nodata_input=nodata_input)

    # _FillValue should be ignored if nodata_input is provided
    if nodata_input is not None:
        assert features.nodata_input.tolist() == [nodata_input] * n_features
    # Nodata vals should match the fill values, even if some are None
    else:
        assert features.nodata_input.tolist() == fill_vals


@pytest.mark.parametrize(
    "nodata_input", [None, -32768], ids=["without_nodata", "with_nodata"]
)
def test_nodata_dataset_global_fillvalue(nodata_input):
    """Test that a global _FillValue is broadcast if per-feature don't exist."""
    n_features = 3
    global_fill_val = 42
    das = [
        xr.DataArray(np.ones((n_features, 2, 2))).rename(i) for i in range(n_features)
    ]

    ds = xr.merge(das).assign_attrs({"_FillValue": global_fill_val})
    features = FeatureArray.from_features(ds, nodata_input=nodata_input)

    # _FillValue should be ignored if nodata_input is provided
    if nodata_input is not None:
        assert features.nodata_input.tolist() == [nodata_input] * n_features
    # The global fill value should be used when per-feature fill values are unavailable
    else:
        assert features.nodata_input.tolist() == [global_fill_val] * n_features


@pytest.mark.parametrize("nodata_output", [np.nan, 0, -32768])
def test_nodata_output_set_in_dataarray_attrs(nodata_output: int | float):
    """Test that the output NoData value is stored as the _FillValue for a DataArray."""
    a = np.array([[[1, 2, 3]]])
    features = FeatureArray.from_features(
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
    features = FeatureArray.from_features(
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


@parametrize_feature_types()
def test_wrappers(feature_type):
    """Confirm that the test wrappers function as expected."""
    array = np.random.rand(3, 32, 16)

    wrapped = wrap_features(array, type=feature_type)
    assert isinstance(wrapped, feature_type)
    assert_array_equal(unwrap_features(wrapped), array)

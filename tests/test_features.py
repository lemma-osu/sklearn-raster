from __future__ import annotations

import re

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

from sklearn_raster.features import FeatureArray, FeatureArrayType
from sklearn_raster.types import MissingType

from .feature_utils import (
    parametrize_feature_array_types,
    unwrap_features,
    wrap_features,
)


@pytest.mark.parametrize("nodata_input", ["test"], ids=type)
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


@pytest.mark.parametrize(
    ("nodata_val", "dtype"),
    [
        (-32768, np.int16),
        (False, np.bool_),
        (0, np.uint8),
    ],
)
def test_nodata_single_value(nodata_val, dtype):
    """Test that a single NoData value is broadcast to all features."""
    n_features = 3
    a = np.zeros((n_features, 2, 2), dtype=dtype)

    features = FeatureArray.from_feature_array(a, nodata_input=nodata_val)
    assert features.nodata_input.data.tolist() == [nodata_val] * n_features
    assert features.nodata_input.mask.tolist() == [False] * n_features
    assert features.nodata_input.dtype == a.dtype


def test_nodata_multiple_values():
    """Test that multiple NoData values are correctly stored."""
    n_features = 3
    nodata_input = [-32768, 0, 255]
    a = np.zeros((n_features, 2, 2))

    features = FeatureArray.from_feature_array(a, nodata_input=nodata_input)
    assert features.nodata_input.data.tolist() == nodata_input
    assert features.nodata_input.mask.tolist() == [False] * n_features
    assert features.nodata_input.dtype == a.dtype


def test_nodata_positive_value_cast_to_signed_dtype():
    """Test that a large positive NoData value can be cast to a signed dtype."""
    # https://github.com/lemma-osu/sklearn-raster/issues/96
    n_features = 1
    nodata_val = 999
    a = np.zeros((n_features, 2, 2), dtype=np.int16)

    features = FeatureArray.from_feature_array(a, nodata_input=nodata_val)
    assert features.nodata_input.data.tolist() == [nodata_val] * n_features
    assert features.nodata_input.mask.tolist() == [False] * n_features
    assert features.nodata_input.dtype == a.dtype


@parametrize_feature_array_types(feature_array_types=(xr.DataArray, xr.Dataset))
def test_nodata_name_dict(feature_array_type: type[FeatureArrayType]):
    """Test that NoData values can be assigned by name."""
    n_features = 3
    nodata_input = {"b0": -32768, "b2": 255}
    a = np.zeros((n_features, 2, 2))
    wrapped = wrap_features(a, type=feature_array_type)

    features = FeatureArray.from_feature_array(wrapped, nodata_input=nodata_input)
    # Feature b1 wasn't specified, so it should be missing
    assert features.nodata_input.data.tolist() == [-32768, 0, 255]
    assert features.nodata_input.mask.tolist() == [False, True, False]

    # Add a fill value to ensure that unspecified features are inferred, if possible
    if feature_array_type is xr.DataArray:
        wrapped.attrs.update({"_FillValue": 123})
    else:
        wrapped["b1"].attrs.update({"_FillValue": 123})

    features = FeatureArray.from_feature_array(wrapped, nodata_input=nodata_input)
    # Feature b1 wasn't specified, so it should be inferred from the _FillValue
    assert features.nodata_input.data.tolist() == [-32768, 123, 255]
    assert features.nodata_input.mask.tolist() == [False, False, False]


def test_nodata_index_dict():
    """Test that NoData values can be assigned by index."""
    n_features = 3
    nodata_input = {0: -32768, 2: 255}
    a = np.zeros((n_features, 2, 2))

    features = FeatureArray.from_feature_array(a, nodata_input=nodata_input)
    # Feature at index 1 wasn't specified, so it should be missing
    assert features.nodata_input.data.tolist() == [-32768, 0, 255]
    assert features.nodata_input.mask.tolist() == [False, True, False]


@parametrize_feature_array_types(feature_array_types=(xr.DataArray, xr.Dataset))
def test_nodata_dict_raises_with_invalid_name(
    feature_array_type: type[FeatureArrayType],
):
    """Test that an invalid feature name raises a helpful error."""
    n_features = 3
    nodata_input = {"foo": -32768}
    a = np.zeros((n_features, 2, 2))
    features = wrap_features(a, type=feature_array_type)

    with pytest.raises(KeyError) as e:
        FeatureArray.from_feature_array(features, nodata_input=nodata_input)
    assert "`foo` is not a valid feature name/index" in str(e.value)
    assert "Choose from ['b0', 'b1', 'b2']" in str(e.value)


def test_nodata_dict_raises_with_invalid_index():
    """Test that an invalid feature index raises a helpful error."""
    n_features = 3
    nodata_input = {99: -32768}
    a = np.zeros((n_features, 2, 2))

    with pytest.raises(KeyError) as e:
        FeatureArray.from_feature_array(a, nodata_input=nodata_input)
    assert "`99` is not a valid feature name/index" in str(e.value)
    assert "Choose from [0, 1, 2]" in str(e.value)


def test_nodata_input_unsupported_dtype():
    """When input NoData values don't fit in the feature array, an error is raised."""
    # Place a missing None value at the start of the list to ensure that unsupported
    # values are correctly indexed after the missing value is removed
    nodata_input = [None, -1, 0, 0.99, 42, np.nan]
    a = np.zeros((len(nodata_input), 2, 2), dtype=np.uint8)

    # Note that None should be missing, not unsupported, so it shouldn't appear in the
    # error message.
    expected_msg = re.escape(
        "The selected or inferred NoData value(s) [-1, 0.99, nan] cannot be safely "
        "cast to the feature array dtype uint8."
    )
    with pytest.raises(ValueError, match=expected_msg):
        FeatureArray.from_feature_array(a, nodata_input=nodata_input)


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
    assert features.nodata_input.dtype == a.dtype


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


def test_ndarray_feature_names():
    """Test that NDArray sets no feature names."""
    assert_array_equal(
        FeatureArray.from_feature_array(np.ones((3, 8, 8))).feature_names,
        np.array([]),
    )


@pytest.mark.parametrize("duplicate_names", [True, False])
def test_dataarray_feature_names(duplicate_names: bool):
    """Test that DataArray sets and validates feature names."""
    feature_names = ["A", "B", "A"] if duplicate_names else ["A", "B", "C"]

    da = xr.DataArray(
        np.ones((3, 8, 8)),
        dims=["feature", "y", "x"],
        coords={"feature": feature_names},
    )

    if duplicate_names:
        expected = re.escape("Found duplicated names ['A']")
        with pytest.raises(ValueError, match=expected):
            FeatureArray.from_feature_array(da)
    else:
        assert_array_equal(
            FeatureArray.from_feature_array(da).feature_names, feature_names
        )


def test_dataset_feature_names():
    """Test that Dataset sets feature names."""
    feature_names = ["A", "B", "C"]

    # Note that Dataset can't contain duplicate feature names, so we don't test for it
    ds = xr.DataArray(
        np.ones((3, 8, 8)),
        dims=["feature", "y", "x"],
        coords={"feature": feature_names},
    ).to_dataset(dim="feature")

    assert_array_equal(FeatureArray.from_feature_array(ds).feature_names, feature_names)


@pytest.mark.parametrize("duplicate_names", [True, False])
def test_dataframe_feature_names(duplicate_names: bool):
    """Test that DataFrame sets and validates feature names."""
    feature_names = ["A", "B", "A"] if duplicate_names else ["A", "B", "C"]

    df = xr.DataArray(
        np.ones((3, 8)), dims=["feature", "samples"], coords={"feature": feature_names}
    ).T.to_pandas()

    if duplicate_names:
        # Duplicate columns are detected by Xarray during FeatureArray instantiation,
        # so we never reach our validation and end up with a different error.
        expected = "cannot convert DataFrame with non-unique columns"
        with pytest.raises(ValueError, match=expected):
            FeatureArray.from_feature_array(df)
    else:
        assert_array_equal(
            FeatureArray.from_feature_array(df).feature_names, feature_names
        )


@parametrize_feature_array_types()
def test_wrappers(feature_array_type):
    """Confirm that the test wrappers function as expected."""
    array = np.random.rand(3, 32, 16)

    wrapped = wrap_features(array, type=feature_array_type)
    assert isinstance(wrapped, feature_array_type)
    assert_array_equal(unwrap_features(wrapped), array)

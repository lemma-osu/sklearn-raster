"""Test the image module."""

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

from sknnr_spatial.image import Image
from sknnr_spatial.types import ImageType

from .image_utils import (
    parametrize_image_types,
    unwrap_image,
    wrap_image,
)


@parametrize_image_types
def test_input_array_not_mutated(image_type: type[ImageType]):
    """Ensure that applying a ufunc to an image doesn't mutate the original array."""
    array = np.array([[[0, 1]], [[1, np.nan]]])
    original_array = array.copy()

    img = wrap_image(array, type=image_type)

    image = Image.from_image(img, nodata_vals=0)
    image.apply_ufunc_across_bands(
        lambda x: x * 2.0,
        output_dims=[["variable"]],
        output_sizes={"variable": array.shape[0]},
        output_dtypes=[array.dtype],
    )

    assert_array_equal(array, original_array)


@parametrize_image_types
@pytest.mark.parametrize("nan_fill", [None, 42.0])
def test_nans_filled(nan_fill, image_type: type[ImageType]):
    """Test that NaNs in the image are filled or not."""
    a = np.random.rand(3, 8, 8)
    a[1][0][0] = np.nan
    a[2][4][3] = np.nan

    if nan_fill is not None:
        expected_output = np.where(np.isnan(a), nan_fill, a)
    else:
        expected_output = a.copy()

    image = Image.from_image(wrap_image(a, type=image_type))

    output = image.apply_ufunc_across_bands(
        func=lambda x: x,
        nan_fill=nan_fill,
        # Masking NoData would broadcast NaNs across the band dimension since a missing
        # value in any band is considered missing in all bands of the output, so skip
        # that step.
        mask_nodata=False,
        output_dims=[["variable"]],
        output_sizes={"variable": a.shape[0]},
        output_dtypes=[a.dtype],
    )

    assert_array_equal(unwrap_image(output), expected_output)


def test_skip_nodata_if_unneeded():
    """If an image is not float and nodata isn't specified, there should be no mask."""
    a = np.ones((3, 2, 2), dtype=int)
    image = Image.from_image(a, nodata_vals=None)

    assert image.nodata_vals is None


@pytest.mark.parametrize("nodata", [99, np.nan])
def test_nodata_masked_in_all_bands(nodata):
    """If one band is NoData, those pixels should be masked in all bands."""
    # Build an array with one band filled with NoData
    a = np.ones((3, 8, 8))
    a[1, ...] = nodata

    # The output should be fully masked because missing values are broadcast to all
    # bands in the output.
    expected_output = np.full((3, 8, 8), np.nan)

    image = Image.from_image(a, nodata_vals=nodata)

    output = image.apply_ufunc_across_bands(
        func=lambda x: x,
        mask_nodata=True,
        output_dims=[["variable"]],
    )

    assert_array_equal(output, expected_output)


@pytest.mark.parametrize("nodata_vals", ["test", {}, False], ids=type)
def test_nodata_validates_type(nodata_vals):
    """Test that invalid NoData types are recognized."""
    a = np.zeros((3, 2, 2))

    with pytest.raises(TypeError, match=f"Invalid type `{type(nodata_vals).__name__}`"):
        Image.from_image(a, nodata_vals=nodata_vals)


def test_nodata_validates_length():
    """Test that invalid NoData lengths are recognized."""
    n_bands = 3
    a = np.zeros((n_bands, 2, 2))

    with pytest.raises(ValueError, match=f"Expected {n_bands} NoData values but got 1"):
        Image.from_image(a, nodata_vals=[-32768])


def test_nodata_single_value():
    """Test that a single NoData value is broadcast to all bands."""
    n_bands = 3
    nodata_val = -32768
    a = np.zeros((n_bands, 2, 2))

    image = Image.from_image(a, nodata_vals=nodata_val)
    assert image.nodata_vals.tolist() == [nodata_val] * n_bands


def test_nodata_multiple_values():
    """Test that multiple NoData values are correctly stored."""
    n_bands = 3
    nodata_vals = [-32768, 0, 255]
    a = np.zeros((n_bands, 2, 2))

    image = Image.from_image(a, nodata_vals=nodata_vals)
    assert image.nodata_vals.tolist() == nodata_vals


@pytest.mark.parametrize("nodata_vals", [None, -32768])
def test_nodata_dataarray_fillvalue(nodata_vals):
    """Test that a _FillValue in a DataArray is broadcast if NoData is not provided."""
    n_bands = 3
    fill_val = -99

    da = xr.DataArray(np.ones((n_bands, 2, 2))).assign_attrs({"_FillValue": fill_val})
    image = Image.from_image(da, nodata_vals=nodata_vals)

    # _FillValue should be ignored if nodata_vals is provided
    if nodata_vals is not None:
        assert image.nodata_vals.tolist() == [nodata_vals] * n_bands
    else:
        assert image.nodata_vals.tolist() == [fill_val] * n_bands


@pytest.mark.parametrize(
    "nodata_vals", [None, -32768], ids=["without_nodata", "with_nodata"]
)
@pytest.mark.parametrize(
    "fill_vals", [[1, 2, 3], [None, 1, None]], ids=["no_nones", "some_nones"]
)
def test_nodata_dataset_some_fillvalues(nodata_vals, fill_vals):
    """Test that band-wise _FillValues are applied if some exist"""
    n_bands = 3
    das = [xr.DataArray(np.ones((n_bands, 2, 2))) for i in range(n_bands)]

    # Assign per-band fill values
    for i, fill_val in enumerate(fill_vals):
        das[i] = das[i].assign_attrs({"_FillValue": fill_val}).rename(i)

    ds = xr.merge(das)
    image = Image.from_image(ds, nodata_vals=nodata_vals)

    # _FillValue should be ignored if nodata_vals is provided
    if nodata_vals is not None:
        assert image.nodata_vals.tolist() == [nodata_vals] * n_bands
    # Nodata vals should match the fill values, even if some are None
    else:
        assert image.nodata_vals.tolist() == fill_vals


@pytest.mark.parametrize(
    "nodata_vals", [None, -32768], ids=["without_nodata", "with_nodata"]
)
def test_nodata_dataset_global_fillvalue(nodata_vals):
    """Test that a global _FillValue is broadcast if per-band don't exist."""
    n_bands = 3
    global_fill_val = 42
    das = [xr.DataArray(np.ones((n_bands, 2, 2))).rename(i) for i in range(n_bands)]

    ds = xr.merge(das).assign_attrs({"_FillValue": global_fill_val})
    image = Image.from_image(ds, nodata_vals=nodata_vals)

    # _FillValue should be ignored if nodata_vals is provided
    if nodata_vals is not None:
        assert image.nodata_vals.tolist() == [nodata_vals] * n_bands
    # The global fill value should be used when per-band fill values are unavailable
    else:
        assert image.nodata_vals.tolist() == [global_fill_val] * n_bands


@parametrize_image_types
def test_wrappers(image_type):
    """Confirm that the test wrappers function as expected."""
    array = np.random.rand(3, 32, 16)

    wrapped = wrap_image(array, type=image_type)
    assert isinstance(wrapped, image_type)
    assert_array_equal(unwrap_image(wrapped), array)

"""Test the image module."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

from sklearn_raster.image import Image
from sklearn_raster.types import ImageType

from .image_utils import (
    parametrize_image_types,
    unwrap_image,
    wrap_image,
)


@parametrize_image_types
@pytest.mark.parametrize("skip_nodata", [True, False])
def test_input_array_not_mutated(image_type: type[ImageType], skip_nodata: bool):
    """Ensure that applying a ufunc to an image doesn't mutate the original array."""
    array = np.array([[[0, 1]], [[1, np.nan]]])
    original_array = array.copy()

    img = wrap_image(array, type=image_type)

    image = Image.from_image(img, nodata_input=0)
    image.apply_ufunc_across_bands(
        lambda x: x * 2.0,
        skip_nodata=skip_nodata,
        output_dims=[["variable"]],
        output_sizes={"variable": array.shape[0]},
        output_dtypes=[array.dtype],
    )

    assert_array_equal(array, original_array)


@parametrize_image_types
@pytest.mark.parametrize("skip_nodata", [True, False])
@pytest.mark.parametrize("val_dtype", [(-1, np.uint8), (np.nan, np.int16)])
def test_nodata_output_with_unsupported_dtype(
    val_dtype: tuple[int | float, np.dtype],
    image_type: type[ImageType],
    skip_nodata: bool,
):
    """Test that an unsupported nodata_output value raises an error."""
    # Make sure there's a value to mask in the input array
    a = np.array([[[np.nan]]])
    img = wrap_image(a, type=image_type)
    image = Image.from_image(img, nodata_input=0)

    output_nodata, output_dtype = val_dtype
    with pytest.raises(ValueError, match="does not fit in the array dtype"):
        # Unwrap to force computation for lazy arrays
        unwrap_image(
            image.apply_ufunc_across_bands(
                lambda x: np.ones_like(x).astype(output_dtype),
                nodata_output=output_nodata,
                skip_nodata=skip_nodata,
                output_dims=[["variable"]],
                output_sizes={"variable": a.shape[0]},
                output_dtypes=[a.dtype],
            )
        )


@pytest.mark.parametrize("nodata_output", [np.nan, 42.0])
@parametrize_image_types
@pytest.mark.parametrize("skip_nodata", [True, False])
def test_nodata_output_set(
    nodata_output: int | float, image_type: type[ImageType], skip_nodata: bool
):
    """Test that NoData in the image are filled or not."""
    nodata_input = 0

    # Encoded NoData and NaN should both be replaced across bands with the nodata_output
    # value.
    a = np.array([[[nodata_input, 1, np.nan]]])
    expected_output = np.array([[[nodata_output, 1, nodata_output]]])

    image = Image.from_image(wrap_image(a, type=image_type), nodata_input=0)
    result = image.apply_ufunc_across_bands(
        lambda x: x,
        nodata_output=nodata_output,
        skip_nodata=skip_nodata,
        output_dims=[["variable"]],
        output_sizes={"variable": a.shape[0]},
        output_dtypes=[a.dtype],
    )

    assert_array_equal(unwrap_image(result), expected_output)


@parametrize_image_types
@pytest.mark.parametrize("skip_nodata", [True, False])
@pytest.mark.parametrize("nan_fill", [None, 42.0])
def test_nan_filled(
    image_type: type[ImageType], nan_fill: float | None, skip_nodata: bool
):
    """Test that NaNs in the image are filled before passing to func."""
    a = np.array([[[1, np.nan]]])
    image = Image.from_image(wrap_image(a, type=image_type))

    def nan_check(x):
        nonlocal nan_fill
        if nan_fill is not None:
            assert not np.isnan(x).any()

        return x

    result = image.apply_ufunc_across_bands(
        nan_check,
        skip_nodata=skip_nodata,
        output_dims=[["variable"]],
        output_sizes={"variable": a.shape[0]},
        output_dtypes=[a.dtype],
    )
    # Unwrap to force computation for lazy arrays
    unwrap_image(result)


def test_skip_nodata_if_unneeded():
    """If an image is not float and nodata isn't specified, there should be no mask."""
    a = np.ones((3, 2, 2), dtype=int)
    image = Image.from_image(a, nodata_input=None)

    assert image.nodata_input is None


@pytest.mark.parametrize("nodata_input", ["test", {}, False], ids=type)
def test_nodata_validates_type(nodata_input):
    """Test that invalid NoData types are recognized."""
    a = np.zeros((3, 2, 2))

    with pytest.raises(
        TypeError, match=f"Invalid type `{type(nodata_input).__name__}`"
    ):
        Image.from_image(a, nodata_input=nodata_input)


def test_nodata_validates_length():
    """Test that invalid NoData lengths are recognized."""
    n_bands = 3
    a = np.zeros((n_bands, 2, 2))

    with pytest.raises(ValueError, match=f"Expected {n_bands} NoData values but got 1"):
        Image.from_image(a, nodata_input=[-32768])


def test_nodata_single_value():
    """Test that a single NoData value is broadcast to all bands."""
    n_bands = 3
    nodata_val = -32768
    a = np.zeros((n_bands, 2, 2))

    image = Image.from_image(a, nodata_input=nodata_val)
    assert image.nodata_input.tolist() == [nodata_val] * n_bands


def test_nodata_multiple_values():
    """Test that multiple NoData values are correctly stored."""
    n_bands = 3
    nodata_input = [-32768, 0, 255]
    a = np.zeros((n_bands, 2, 2))

    image = Image.from_image(a, nodata_input=nodata_input)
    assert image.nodata_input.tolist() == nodata_input


@pytest.mark.parametrize("nodata_input", [None, -32768])
def test_nodata_dataarray_fillvalue(nodata_input):
    """Test that a _FillValue in a DataArray is broadcast if NoData is not provided."""
    n_bands = 3
    fill_val = -99

    da = xr.DataArray(np.ones((n_bands, 2, 2))).assign_attrs({"_FillValue": fill_val})
    image = Image.from_image(da, nodata_input=nodata_input)

    # _FillValue should be ignored if nodata_input is provided
    if nodata_input is not None:
        assert image.nodata_input.tolist() == [nodata_input] * n_bands
    else:
        assert image.nodata_input.tolist() == [fill_val] * n_bands


@pytest.mark.parametrize(
    "nodata_input", [None, -32768], ids=["without_nodata", "with_nodata"]
)
@pytest.mark.parametrize(
    "fill_vals", [[1, 2, 3], [None, 1, None]], ids=["no_nones", "some_nones"]
)
def test_nodata_dataset_some_fillvalues(nodata_input, fill_vals):
    """Test that band-wise _FillValues are applied if some exist"""
    n_bands = 3
    das = [xr.DataArray(np.ones((n_bands, 2, 2))) for i in range(n_bands)]

    # Assign per-band fill values
    for i, fill_val in enumerate(fill_vals):
        das[i] = das[i].assign_attrs({"_FillValue": fill_val}).rename(i)

    ds = xr.merge(das)
    image = Image.from_image(ds, nodata_input=nodata_input)

    # _FillValue should be ignored if nodata_input is provided
    if nodata_input is not None:
        assert image.nodata_input.tolist() == [nodata_input] * n_bands
    # Nodata vals should match the fill values, even if some are None
    else:
        assert image.nodata_input.tolist() == fill_vals


@pytest.mark.parametrize(
    "nodata_input", [None, -32768], ids=["without_nodata", "with_nodata"]
)
def test_nodata_dataset_global_fillvalue(nodata_input):
    """Test that a global _FillValue is broadcast if per-band don't exist."""
    n_bands = 3
    global_fill_val = 42
    das = [xr.DataArray(np.ones((n_bands, 2, 2))).rename(i) for i in range(n_bands)]

    ds = xr.merge(das).assign_attrs({"_FillValue": global_fill_val})
    image = Image.from_image(ds, nodata_input=nodata_input)

    # _FillValue should be ignored if nodata_input is provided
    if nodata_input is not None:
        assert image.nodata_input.tolist() == [nodata_input] * n_bands
    # The global fill value should be used when per-band fill values are unavailable
    else:
        assert image.nodata_input.tolist() == [global_fill_val] * n_bands


@parametrize_image_types
def test_wrappers(image_type):
    """Confirm that the test wrappers function as expected."""
    array = np.random.rand(3, 32, 16)

    wrapped = wrap_image(array, type=image_type)
    assert isinstance(wrapped, image_type)
    assert_array_equal(unwrap_image(wrapped), array)

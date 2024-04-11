import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

from sknnr_spatial.image.dataarray import DataArrayPreprocessor
from sknnr_spatial.image.dataset import DatasetPreprocessor
from sknnr_spatial.image.ndarray import NDArrayPreprocessor

"""
TODO
- Figure out how to efficiently test the Dask preprocessor. We could easily swap Numpy 
arrays for Dask arrays, but xarray.DataArray is trickier
"""


def test_flat_nans_filled():
    """NaNs in the flat image should always be filled."""
    fill_value = 0.0

    a = np.ones((2, 2, 3))
    a[0, 0, 0] = np.nan

    # The expected flat array, with the NaN filled with the default fill value
    expected_array = np.ones((4, 3))
    expected_array[0, 0] = fill_value

    preproc = NDArrayPreprocessor(a)
    assert_array_equal(preproc.flat, expected_array)


def test_flatten():
    """Flattening an 3D array should return a 2D array."""
    a = np.ones((2, 2, 3))
    preproc = NDArrayPreprocessor(a)

    assert preproc.flat.shape == (4, 3)
    assert_array_equal(preproc.flat, np.ones((4, 3)))


@pytest.mark.parametrize("dtype", [float, int])
@pytest.mark.parametrize("nodata_vals", [None, -32768, [-32768, -32768, -32768]])
def test_flatten_is_reversible(dtype, nodata_vals):
    """Unflattening a flattened array should return the original array."""
    a = np.random.randint(0, 1e4, (2, 2, 3)).astype(dtype)
    preproc = NDArrayPreprocessor(a, nodata_vals=nodata_vals)

    assert_array_equal(preproc.unflatten(preproc.flat), a)


def test_unflatten_masks_nans():
    """NaNs in the original image should be persisted to an unflattened image."""
    # Make an array with one NaN value
    a = np.random.rand(2, 2, 1)
    a[0, 0, 0] = np.nan
    preproc = NDArrayPreprocessor(a, nodata_vals=None)

    # Build a new flat array with no NaNs, simulating a prediction output
    flat_array = np.ones_like(preproc.flat)

    # After unflattening, the nodata value from the original array should be masked
    expected_array = np.ones_like(a)
    expected_array[0, 0, 0] = np.nan

    assert_array_equal(preproc.unflatten(flat_array), expected_array)


def test_skip_nodata_mask_if_unneeded():
    """If an image is not float and nodata isn't specified, there should be no mask."""
    a = np.ones((2, 2, 3), dtype=int)
    preproc = NDArrayPreprocessor(a, nodata_vals=None)

    assert preproc.nodata_vals is None
    assert preproc.nodata_mask is None


def test_nodata_mask_one_band_masks_all():
    """If one band is nodata, those pixels should be masked."""
    nodata = 99

    # Build an array with nodata values for one of the 3 bands
    a = np.ones((2, 2, 3))
    a[..., 0] = nodata

    # The output should be fully masked
    expected_mask = np.full((4,), True)

    preproc = NDArrayPreprocessor(a, nodata_vals=99)
    assert_array_equal(preproc.nodata_mask, expected_mask)


@pytest.mark.parametrize("nodata_vals", ["test", {}, False], ids=type)
def test_nodata_validates_type(nodata_vals):
    """Test that invalid nodata types are recognized."""
    a = np.zeros((2, 2, 3))

    with pytest.raises(TypeError, match=f"Invalid type `{type(nodata_vals).__name__}`"):
        NDArrayPreprocessor(a, nodata_vals=nodata_vals)


def test_nodata_validates_length():
    """Test that invalid nodata lengths are recognized."""
    n_bands = 3
    a = np.zeros((2, 2, n_bands))

    with pytest.raises(ValueError, match=f"Expected {n_bands} nodata values but got 1"):
        NDArrayPreprocessor(a, nodata_vals=[-32768])


def test_nodata_single_value():
    """Test that a single nodata value is broadcast to all bands."""
    n_bands = 3
    nodata_val = -32768
    a = np.zeros((2, 2, n_bands))

    preproc = NDArrayPreprocessor(a, nodata_vals=nodata_val)
    assert preproc.nodata_vals.tolist() == [nodata_val] * n_bands


def test_nodata_multiple_values():
    """Test that multiple nodata values are correctly stored."""
    n_bands = 3
    nodata_vals = [-32768, 0, 255]
    a = np.zeros((2, 2, n_bands))

    preproc = NDArrayPreprocessor(a, nodata_vals=nodata_vals)
    assert preproc.nodata_vals.tolist() == nodata_vals


@pytest.mark.parametrize("nodata_vals", [None, -32768])
def test_nodata_dataarray_fillvalue(nodata_vals):
    """Test that a _FillValue in a DataArray is broadcast if nodata is not provided."""
    n_bands = 3
    fill_val = -99

    da = xr.DataArray(np.ones((n_bands, 2, 2))).assign_attrs({"_FillValue": fill_val})
    preproc = DataArrayPreprocessor(da, nodata_vals=nodata_vals)

    # _FillValue should be ignored if nodata_vals is provided
    if nodata_vals is not None:
        assert preproc.nodata_vals.tolist() == [nodata_vals] * n_bands
    else:
        assert preproc.nodata_vals.tolist() == [fill_val] * n_bands


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
    preproc = DatasetPreprocessor(ds, nodata_vals=nodata_vals)

    # _FillValue should be ignored if nodata_vals is provided
    if nodata_vals is not None:
        assert preproc.nodata_vals.tolist() == [nodata_vals] * n_bands
    # Nodata vals should match the fill values, even if some are None
    else:
        assert preproc.nodata_vals.tolist() == fill_vals


@pytest.mark.parametrize(
    "nodata_vals", [None, -32768], ids=["without_nodata", "with_nodata"]
)
def test_nodata_dataset_global_fillvalue(nodata_vals):
    """Test that a global _FillValue is broadcast if per-band don't exist."""
    n_bands = 3
    global_fill_val = 42
    das = [xr.DataArray(np.ones((n_bands, 2, 2))).rename(i) for i in range(n_bands)]

    ds = xr.merge(das).assign_attrs({"_FillValue": global_fill_val})
    preproc = DatasetPreprocessor(ds, nodata_vals=nodata_vals)

    # _FillValue should be ignored if nodata_vals is provided
    if nodata_vals is not None:
        assert preproc.nodata_vals.tolist() == [nodata_vals] * n_bands
    # The global fill value should be used when per-band fill values are unavailable
    else:
        assert preproc.nodata_vals.tolist() == [global_fill_val] * n_bands

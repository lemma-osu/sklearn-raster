"""
Image wrappers used for testing various image types with a common Numpy NDArray
interface.

An example usage is shown below, where an array is wrapped into an Xarray image,
modified, unwrapped back to a Numpy array, and compared to another array.

>>> from numpy.testing import assert_array_equal
>>> array = np.ones((8, 8, 3))
>>> wrapped = wrap(np.ones((8, 8, 3)), type=xr.Dataset)
>>> wrapped += 1
>>> assert_array_equal(unwrap(wrapped), array + 1)

The advantage of this system is that you can easily parameterize over multiple types by
changing the `type` parameter, without having to modify the test code.
"""

from dataclasses import dataclass
from functools import singledispatch

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray

from sknnr_spatial.image._base import ImagePreprocessor
from sknnr_spatial.image.dataarray import DataArrayPreprocessor
from sknnr_spatial.image.dataset import DatasetPreprocessor
from sknnr_spatial.image.ndarray import NDArrayPreprocessor
from sknnr_spatial.types import ImageType


@dataclass
class TestImageType:
    """A container for image types used in testing."""

    __test__ = False
    cls: type[ImageType]
    preprocessor: type[ImagePreprocessor]

    @property
    def name(self):
        return self.cls.__name__


parametrize_image_types = pytest.mark.parametrize(
    "image_type",
    [
        TestImageType(np.ndarray, NDArrayPreprocessor),
        TestImageType(xr.DataArray, DataArrayPreprocessor),
        TestImageType(xr.Dataset, DatasetPreprocessor),
    ],
    ids=lambda t: t.name,
)

parametrize_xarray_image_types = pytest.mark.parametrize(
    "image_type",
    [
        TestImageType(xr.DataArray, DataArrayPreprocessor),
        TestImageType(xr.Dataset, DatasetPreprocessor),
    ],
    ids=lambda t: t.name,
)


def wrap(image: NDArray, type: type[ImageType]) -> ImageType:
    """Wrap a Numpy NDArray image (y, x, bands) into the specified type."""
    if type is np.ndarray:
        return image

    if type is xr.DataArray:
        n_bands = image.shape[-1]
        band_names = [f"b{i}" for i in range(n_bands)]

        return (
            xr.DataArray(
                image,
                dims=["y", "x", "variable"],
                coords={"variable": band_names},
            )
            .chunk("auto")
            .transpose("variable", "y", "x")
        )

    if type is xr.Dataset:
        return wrap(image, xr.DataArray).to_dataset(dim="variable")

    raise ValueError(f"Unsupported image type: {type}")


@singledispatch
def unwrap(image: ImageType) -> NDArray:
    """Unwrap an image to a Numpy NDArray in the shape (y, x, band)."""
    raise NotImplementedError()


@unwrap.register(np.ndarray)
def _unwrap_ndarray(image: np.ndarray) -> NDArray:
    return image


@unwrap.register(xr.DataArray)
def _unwrap_dataarray(image: xr.DataArray) -> NDArray:
    band_dim_name = image.dims[0]

    return image.transpose("y", "x", band_dim_name).values


@unwrap.register(xr.Dataset)
def _unwrap_dataset(image: xr.Dataset) -> NDArray:
    return unwrap(image.to_dataarray())


@unwrap.register(da.Array)
def _unwrap_dask_array(image: da.Array) -> NDArray:
    # We don't support Dask array image inputs, but they're used internally for flat
    # arrays, which we need to be able to unwrap for testing.
    return image.compute()

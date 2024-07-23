from __future__ import annotations

from typing import Any, Generic

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.typing import NDArray

from sknnr_spatial.types import ImageType

# Parametrize over all supported image types
parametrize_image_types = pytest.mark.parametrize(
    "image_type",
    [
        np.ndarray,
        xr.DataArray,
        xr.Dataset,
    ],
    ids=lambda t: t.__name__,
)


class ModelData(Generic[ImageType]):
    """
    Data used to train and predict with image-compatible estimators for testing.

    Examples
    --------
    ModelData is designed to be instantiated with Numpy array data and unpacked to
    retrieve compatible images and training data.

    >>> X_image = np.random.random((16, 16, 5))
    >>> X = np.random.random((10, 5))
    >>> y = np.random.random((10, 3))
    >>> model_data = ModelData(X_image, X, y, image_type=xr.Dataset)
    >>> X_image, X, y = model_data
    >>> type(X_image)
    <class 'xarray.core.dataset.Dataset'>
    >>> type(y)
    <class 'pandas.core.frame.DataFrame'>

    Data is generated on-the-fly, so you can mutate the ModelData with `set` and then
    retrieve new data.

    >>> y.shape
    (10, 3)
    >>> _, _, y = model_data.set(single_output=True, squeeze=True)
    >>> y.shape
    (10,)
    """

    def __init__(
        self,
        X_image: NDArray,
        X: NDArray,
        y: NDArray,
        image_type: type[ImageType] = np.ndarray,
    ):
        self._image_type = image_type
        self._X_image = X_image
        self._X = X
        self._y = y
        self._single_output = False
        self._squeeze = False

    def __iter__(self):
        """Unpack the data into (X_image, X, y)."""
        return iter((self.X_image, self.X, self.y))

    @property
    def n_targets(self):
        return self._y.shape[-1]

    @property
    def n_rows(self):
        return self._X_image.shape[0]

    @property
    def n_cols(self):
        return self._X_image.shape[1]

    @property
    def n_features(self):
        return self._X_image.shape[-1]

    @property
    def X_image(self) -> ImageType:
        """Feature image."""
        return wrap_image(self._X_image, self._image_type)

    @property
    def X(self) -> NDArray | pd.DataFrame:
        """Feature data in array or dataframe format."""
        X = self._X.copy()

        if self._image_type in (xr.DataArray, xr.Dataset):
            band_names = [f"b{i}" for i in range(self.n_features)]
            X = pd.DataFrame(X, columns=band_names)

        return X

    @property
    def y(self) -> NDArray | pd.DataFrame:
        """Label data in array or dataframe format."""
        y = self._y.copy()

        n_targets = self.n_targets
        if self._single_output:
            y = y[:, :1]
            n_targets = 1

        if self._image_type in (xr.DataArray, xr.Dataset):
            target_names = [f"t{i}" for i in range(n_targets)]
            y = pd.DataFrame(y, columns=target_names)

        if self._squeeze:
            y = y.squeeze()

        return y

    def set(self, **kwargs):
        """
        Update the ModelData's attributes.

        Attributes should be referenced by their public names, e.g. `X_image`, but
        will actually update corresponding private attributes to avoid shadowing public
        properties.
        """
        for k, v in kwargs.items():
            if k.startswith("_"):
                msg = (
                    "Set attributes based on their public names, e.g. `X_image` "
                    "instead of `_X_image`."
                )
                raise ValueError(msg)

            # Attributes are all stored with a leading underscore to avoid shadowing
            # public properties.
            k = f"_{k}"

            if not hasattr(self, k):
                raise AttributeError(f"{k} is not a valid property of ModelData.")

            setattr(self, k, v)

        return self


def parametrize_model_data(
    label="model_data",
    X_image=None,
    X=None,
    y=None,
    image_types=(np.ndarray, xr.DataArray, xr.Dataset),
):
    """Parametrize over multiple image types with the same test data."""
    n_features = (
        X_image.shape[-1]
        if X_image is not None
        else X.shape[-1]
        if X is not None
        else 5
    )
    n_targets = y.shape[-1] if y is not None else 3
    n_rows = X.shape[0] if X is not None else y.shape[0] if y is not None else 10

    # Default test data
    if X_image is None:
        X_image = np.random.rand(8, 16, n_features)
    if X is None:
        X = np.random.rand(n_rows, n_features)
    if y is None:
        y = np.random.rand(n_rows, n_targets)

    model_data = [ModelData(X_image, X, y, cls) for cls in image_types]

    return pytest.mark.parametrize(
        label, model_data, ids=map(lambda x: x.__name__, image_types)
    )


def wrap_image(image: NDArray, type: type[ImageType]) -> ImageType:
    """
    Wrap a Numpy NDArray image (y, x, bands) into the specified type.

    Examples
    --------

    Wrap a Numpy array into a desired image type:

    >>> array = np.ones((8, 8, 3))
    >>> wrapped = wrap_image(array, type=xr.DataArray)
    >>> type(wrapped)
    <class 'xarray.core.dataarray.DataArray'>

    Combine with `unwrap_image` to allow testing functions that are compatible with any
    image type:

    >>> from numpy.testing import assert_array_equal
    >>> array = np.ones((8, 8, 3))
    >>> wrapped = wrap_image(array, type=xr.Dataset)
    >>> wrapped += 1
    >>> assert_array_equal(unwrap_image(wrapped), array + 1)
    """

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
        return wrap_image(image, xr.DataArray).to_dataset(dim="variable")

    raise ValueError(f"Unsupported image type: {type}")


def unwrap_image(image: Any) -> NDArray:
    """
    Unwrap an image to a Numpy NDArray in the shape (y, x, band).

    Examples
    --------
    Unwrap an xarray DataArray to a Numpy array:

    >>> from numpy.testing import assert_array_equal
    >>> array = np.ones((8, 8, 3))
    >>> wrapped = wrap_image(array, type=xr.Dataset)
    >>> unwrapped = unwrap_image(wrapped)
    >>> assert_array_equal(unwrapped, array)
    """
    if isinstance(image, np.ndarray):
        return image

    if isinstance(image, xr.DataArray):
        band_dim_name = image.dims[0]

        return image.transpose("y", "x", band_dim_name).values

    if isinstance(image, xr.Dataset):
        return unwrap_image(image.to_dataarray())

    raise ValueError(f"Unsupported image type: {type(image)}")

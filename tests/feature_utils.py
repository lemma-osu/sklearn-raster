from __future__ import annotations

from typing import Any, Generic, Literal

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.typing import NDArray

from sklearn_raster.types import FeatureArrayType

# Dimension names to use when building Xarray features, in order of increasing
# dimensionality, excluding the first "variable" dimension.
EXTRA_DIM_NAMES = ["x", "y", "z", "time"]


def parametrize_feature_array_types(
    label="feature_array_type",
    feature_array_types=(np.ndarray, xr.DataArray, xr.Dataset),
):
    """Parametrize over multiple feature types."""
    return pytest.mark.parametrize(label, feature_array_types, ids=lambda t: t.__name__)


class ModelData(Generic[FeatureArrayType]):
    """
    Data used to train and predict with raster-compatible estimators for testing.

    Examples
    --------
    ModelData is designed to be instantiated with Numpy array data and unpacked to
    retrieve compatible features and training data.

    >>> X_image = np.random.random((5, 16, 16))
    >>> X = np.random.random((10, 5))
    >>> y = np.random.random((10, 3))
    >>> model_data = ModelData(X_image, X, y, feature_array_type=xr.Dataset)
    >>> X_image, X, y = model_data
    >>> type(X_image)
    <class 'xarray.core.dataset.Dataset'>
    >>> type(y) # doctest: +SKIP
    <class 'pandas.DataFrame'>

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
        feature_array_type: type[FeatureArrayType] = np.ndarray,
    ):
        self._feature_array_type = feature_array_type
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
        return self._X_image.shape[1]

    @property
    def n_cols(self):
        return self._X_image.shape[2]

    @property
    def n_features(self):
        return self._X_image.shape[0]

    @property
    def X_image_shape(self):
        return self._X_image.shape

    @property
    def X_image(self) -> FeatureArrayType:
        """Feature image."""
        return wrap_features(self._X_image, self._feature_array_type)

    @property
    def X(self) -> NDArray | pd.DataFrame:
        """Feature data in array or dataframe format."""
        X = self._X.copy()

        if self._feature_array_type in (xr.DataArray, xr.Dataset):
            feature_names = [f"b{i}" for i in range(self.n_features)]
            X = pd.DataFrame(X, columns=feature_names)

        return X

    @property
    def y(self) -> NDArray | pd.DataFrame:
        """Label data in array or dataframe format."""
        y = self._y.copy()

        n_targets = self.n_targets
        if self._single_output:
            y = y[:, :1]
            n_targets = 1

        if self._feature_array_type in (xr.DataArray, xr.Dataset):
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
    feature_array_types=(np.ndarray, xr.DataArray, xr.Dataset),
    mode: Literal["regression", "classification"] = "regression",
    n_features: int = 5,
    n_targets: int = 3,
    n_rows: int = 10,
):
    """Parametrize over multiple feature types with the same test data."""
    n_features = (
        X_image.shape[0]
        if X_image is not None
        else X.shape[-1]
        if X is not None
        else n_features
    )
    n_targets = y.shape[-1] if y is not None else n_targets
    n_rows = X.shape[0] if X is not None else y.shape[0] if y is not None else n_rows

    # Default test data
    if X_image is None:
        X_image = np.random.rand(n_features, 8, 16)
    if X is None:
        X = np.random.rand(n_rows, n_features)
    if y is None:
        if mode == "classification":
            y = np.random.choice([0, 1], (n_rows, n_targets))
        else:
            y = np.random.rand(n_rows, n_targets)

    model_data = [ModelData(X_image, X, y, cls) for cls in feature_array_types]

    return pytest.mark.parametrize(
        label, model_data, ids=map(lambda x: x.__name__, feature_array_types)
    )


def wrap_features(features: NDArray, type: type[FeatureArrayType]) -> FeatureArrayType:
    """
    Wrap a Numpy NDArray with features in the first dimension into the specified type.

    Parameters
    ----------
    features : NDArray
        The array to wrap, with features in the first dimension and between 1 and 4
        additional dimensions, from (features, samples) up to (features, time, z, y, x).
    type : type
        The desired feature type, either np.ndarray, xr.DataArray, or xr.Dataset.

    Returns
    -------
    FeatureArrayType
        The features in the desired format.

    Examples
    --------

    Wrap a Numpy array into a desired feature type:

    >>> array = np.ones((3, 8, 8))
    >>> wrapped = wrap_features(array, type=xr.DataArray)
    >>> type(wrapped)
    <class 'xarray.core.dataarray.DataArray'>

    Combine with `unwrap_features` to allow testing functions that are compatible with
    any feature type:

    >>> from numpy.testing import assert_array_equal
    >>> array = np.ones((3, 8, 8))
    >>> wrapped = wrap_features(array, type=xr.Dataset)
    >>> wrapped += 1
    >>> assert_array_equal(unwrap_features(wrapped), array + 1)
    """

    if type is np.ndarray:
        return features

    if type is xr.DataArray:
        n_features = features.shape[0]
        feature_names = [f"b{i}" for i in range(n_features)]

        if features.ndim < 2 or features.ndim > 5:
            raise ValueError("Feature dimensionality must be between 2 and 5.")

        # Include other dimensions in reverse order, following typical NetCDF
        # conventions (e.g. time, z, y, x).
        dims = ["variable"] + EXTRA_DIM_NAMES[: features.ndim - 1][::-1]

        return xr.DataArray(
            features,
            dims=dims,
            coords={"variable": feature_names},
        ).chunk("auto")

    if type is xr.Dataset:
        return wrap_features(features, xr.DataArray).to_dataset(dim="variable")
    
    if type is pd.DataFrame:
        if features.ndim != 2:
            raise ValueError("DataFrame features must be 2D (features, samples).")
        da = wrap_features(features, xr.DataArray)
        return (
            da
            # Transpose from (target, samples) back to (samples, target)
            .T.to_pandas()
            # Preserve the input index name(s)
            .rename_axis([None], axis=0)
        )

    raise ValueError(f"Unsupported feature type: {type}")


def unwrap_features(features: Any) -> NDArray:
    """
    Unwrap features to a Numpy NDArray.

    Examples
    --------
    Unwrap an xarray DataArray to a Numpy array:

    >>> from numpy.testing import assert_array_equal
    >>> array = np.ones((3, 8, 8))
    >>> wrapped = wrap_features(array, type=xr.Dataset)
    >>> unwrapped = unwrap_features(wrapped)
    >>> assert_array_equal(unwrapped, array)
    """
    if isinstance(features, np.ndarray):
        return features

    if isinstance(features, xr.DataArray):
        return features.values

    if isinstance(features, xr.Dataset):
        return unwrap_features(features.to_dataarray())
    
    if isinstance(features, pd.DataFrame):
        return unwrap_features(features.to_xarray())

    raise ValueError(f"Unsupported feature type: {type(features)}")

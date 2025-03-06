from functools import wraps
from typing import Callable

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from typing_extensions import Any, Concatenate

from ..types import RT, ImageType, MaybeTuple, P
from .wrapper import GenericWrapper, map_function_over_tuples


def is_image_type(X: Any) -> bool:
    # Feature array images must have exactly 3 dimensions: (y, x, band) or (band, y, x)
    if isinstance(X, (np.ndarray, xr.DataArray)):
        return X.ndim == 3

    # Feature Dataset images must have exactly 2 dimensions: (x, y)
    if isinstance(X, xr.Dataset):
        return len(X.dims) == 2

    return False


def image_or_fallback(
    func: Callable[Concatenate[GenericWrapper, ImageType, P], RT],
) -> Callable[Concatenate[GenericWrapper, ImageType, P], RT]:
    """Decorator that calls the wrapped method for non-image X arrays."""

    @wraps(func)
    def wrapper(self: GenericWrapper, X_image: ImageType, *args, **kwargs):
        if not is_image_type(X_image):
            return getattr(self._wrapped, func.__name__)(X_image, *args, **kwargs)

        return func(self, X_image, *args, **kwargs)

    return wrapper


def image_to_samples(
    func: Callable[Concatenate[NDArray, P], MaybeTuple[NDArray]],
) -> Callable[Concatenate[NDArray, P], MaybeTuple[NDArray]]:
    """
    Decorator that flattens input images to samples and unflattens results to images.

    Parameters
    ----------
    func : Callable
        The decorated function that takes an array of shape (samples, features) and
        returns one or more arrays of the same shape.

    Returns
    -------
    Callable
        The decorated function that instead takes an image of shape (y, x, bands) and
        returns one or more images of the same shape.

    Notes
    -----
    The dimension order (y, x, band) matches the chunks passed by `xarray.apply_ufunc`,
    rather than the (band, y, x) order used by rasterio.
    """

    @wraps(func)
    def wrapper(image: NDArray, *args, **kwargs) -> MaybeTuple[NDArray]:
        result = func(image.reshape(-1, image.shape[-1]), *args, **kwargs)

        @map_function_over_tuples
        def unflatten(r: NDArray) -> NDArray:
            return r.reshape(*image.shape[:2], -1)

        return unflatten(result)

    return wrapper

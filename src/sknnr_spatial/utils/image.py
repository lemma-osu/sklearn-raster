from functools import wraps
from typing import Callable

import numpy as np
import xarray as xr
from typing_extensions import Any, Concatenate, ParamSpec, TypeVar

from ..image._base import ImagePreprocessor, ImageType, ImageWrapper
from ..image.dataarray import DataArrayWrapper
from ..image.dataset import DatasetWrapper
from ..image.ndarray import NDArrayWrapper
from .wrapper import GenericWrapper

RT = TypeVar("RT")
P = ParamSpec("P")


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


def get_image_wrapper(X_image: ImageType) -> type[ImageWrapper]:
    """Get an ImageWrapper subclass for a given image."""
    if isinstance(X_image, np.ndarray):
        return NDArrayWrapper
    if isinstance(X_image, xr.DataArray):
        return DataArrayWrapper
    if isinstance(X_image, xr.Dataset):
        return DatasetWrapper

    raise TypeError(f"Unsupported image type: {type(X_image).__name__}")


def get_image_preprocessor(X_image: ImageType) -> type[ImagePreprocessor]:
    """Get an ImagePreprocessor subclass for a given image."""
    return get_image_wrapper(X_image).preprocessor_cls

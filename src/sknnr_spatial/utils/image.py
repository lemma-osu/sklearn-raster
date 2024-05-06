import numpy as np
import xarray as xr

from ..image._base import ImageType, ImageWrapper
from ..image.dataarray import DataArrayWrapper
from ..image.dataset import DatasetWrapper
from ..image.ndarray import NDArrayWrapper


def is_image_type(X: ImageType) -> bool:
    # Feature array images must have exactly 3 dimensions: (y, x, band) or (band, y, x)
    if isinstance(X, (np.ndarray, xr.DataArray)):
        return X.ndim == 3

    # Feature Dataset images must have exactly 2 dimensions: (x, y)
    if isinstance(X, xr.Dataset):
        return len(X.dims) == 2

    return False


def get_image_wrapper(x_image: ImageType) -> type[ImageWrapper]:
    """Get an ImageWrapper subclass for a given image."""
    if isinstance(x_image, np.ndarray):
        return NDArrayWrapper
    if isinstance(x_image, xr.DataArray):
        return DataArrayWrapper
    if isinstance(x_image, xr.Dataset):
        return DatasetWrapper

    raise TypeError(f"Unsupported image type: {type(x_image).__name__}")

from functools import wraps
from typing import Callable

from numpy.typing import NDArray
from typing_extensions import Concatenate

from ..types import MaybeTuple, P
from .wrapper import map_function_over_tuples


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
            return r.reshape(*image.shape[:-1], -1)

        return unflatten(result)

    return wrapper

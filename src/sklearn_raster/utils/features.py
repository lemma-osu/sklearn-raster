from functools import wraps
from typing import Callable

from numpy.typing import NDArray
from typing_extensions import Concatenate

from ..types import MaybeTuple, P
from .wrapper import map_function_over_tuples


def reshape_to_samples(
    func: Callable[Concatenate[NDArray, P], MaybeTuple[NDArray]],
) -> Callable[Concatenate[NDArray, P], MaybeTuple[NDArray]]:
    """
    Decorator that reshapes to and from samples by flattening non-feature dimensions.

    Parameters
    ----------
    func : Callable
        The decorated function that takes an array of shape (samples, features) and
        returns one or more arrays of the same shape.

    Returns
    -------
    Callable
        The decorated function that instead takes an array of shape (..., features) and
        returns one or more arrays of the same shape.

    Notes
    -----
    This expects features in the last dimension, as passed by `xarray.apply_ufunc`,
    rather than in the first dimension as expected elsewhere in the package.
    """

    @wraps(func)
    def wrapper(array: NDArray, *args, **kwargs) -> MaybeTuple[NDArray]:
        result = func(array.reshape(-1, array.shape[-1]), *args, **kwargs)

        @map_function_over_tuples
        def unflatten(r: NDArray) -> NDArray:
            return r.reshape(*array.shape[:-1], -1)

        return unflatten(result)

    return wrapper

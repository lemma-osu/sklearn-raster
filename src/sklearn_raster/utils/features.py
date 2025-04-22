from __future__ import annotations

from functools import wraps
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Concatenate

from ..types import MaybeTuple, P
from .wrapper import map_over_arguments


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

        @map_over_arguments("r")
        def unflatten(r: NDArray) -> NDArray:
            return r.reshape(*array.shape[:-1], -1)

        return unflatten(result)

    return wrapper


def get_minimum_precise_numeric_dtype(value: int | float) -> np.dtype:
    """
    Get the minimum numeric dtype for a value without reducing precision.

    Integers will return the smallest integer type that can hold the value, while floats
    will return their current precision.
    """
    return (
        np.min_scalar_type(value)
        if np.issubdtype(type(value), np.integer)
        else np.dtype(type(value))
    )

from __future__ import annotations

from functools import wraps
from inspect import signature
from typing import TYPE_CHECKING, Callable

import threadpoolctl
from numpy.typing import NDArray
from sklearn.utils.validation import check_is_fitted
from typing_extensions import Concatenate

from ..types import RT, MaybeTuple, P

if TYPE_CHECKING:
    from ..estimator import FeatureArrayEstimator


# Global threadpool controller instance for limiting threads in decorated ufuncs.
# Access via _get_threadpool_controller() or indirectly via limit_inner_threads
# decorator. This implementation is adopted from scikit-learn, but we maintain our own
# controller for flexibility and to avoid accessing their private API.
_threadpool_controller = None


def requires_fitted(
    func: Callable[Concatenate[FeatureArrayEstimator, P], RT],
) -> Callable[Concatenate[FeatureArrayEstimator, P], RT]:
    """Decorator to check if an estimator is fitted before calling a method."""

    @wraps(func)
    def wrapper(self: FeatureArrayEstimator, *args, **kwargs):
        check_is_fitted(self)
        return func(self, *args, **kwargs)

    return wrapper


def requires_implementation(
    func: Callable[Concatenate[FeatureArrayEstimator, P], RT],
) -> Callable[Concatenate[FeatureArrayEstimator, P], RT]:
    """
    A decorator that raises if the wrapped instance doesn't implement the given method.
    """
    return requires_attributes(func.__name__)(func)


def requires_attributes(
    *attrs: str,
) -> Callable[
    [Callable[Concatenate[FeatureArrayEstimator, P], RT]],
    Callable[Concatenate[FeatureArrayEstimator, P], RT],
]:
    """
    A decorator that raises if the wrapped instance is missing required attributes.
    """

    def decorator(
        func: Callable[Concatenate[FeatureArrayEstimator, P], RT],
    ) -> Callable[Concatenate[FeatureArrayEstimator, P], RT]:
        @wraps(func)
        def wrapper(self: FeatureArrayEstimator, *args, **kwargs):
            for attr in attrs:
                if hasattr(self.wrapped_estimator, attr):
                    continue
                wrapped_class = self.wrapped_estimator.__class__.__name__
                if attr == func.__name__:
                    msg = f"`{wrapped_class}` does not implement `{func.__name__}`."
                else:
                    msg = (
                        f"`{wrapped_class}` is missing a required attribute `{attr}` "
                        f"needed to implement `{func.__name__}`."
                    )
                raise NotImplementedError(msg)

            return func(self, *args, **kwargs)

        return wrapper

    return decorator


def map_over_arguments(
    *map_args: str,
    mappable=(tuple, list),
    validate_args=True,
):
    """
    A decorator that allows a function to map over selected arguments.

    When the selected arguments are mappable, the function will be called once with
    each value and a tuple of results will be returned. Non-mapped arguments and scalar
    mapped arguments will be passed to each call.

    Parameters
    ----------
    map_args : str
        The names of the arguments to support mapping over.
    mappable : tuple[type], default (list, tuple)
        The types that will be mapped over when passed to a mapped argument.
    validate_args : bool, default True
        If True, the decorator will check that all mapped arguments are defined as
        parameters of the decorated function and raise a ValueError if not.

    Examples
    --------

    Providing an iterable to a mapped argument will return a tuple of results mapped
    over each value:

    >>> @map_over_arguments('b')
    ... def func(a, b):
    ...     return a + b
    >>> func(1, b=[2, 3])
    (3, 4)

    When multiple arguments are mapped, they will be mapped together:

    >>> @map_over_arguments('a', 'b')
    ... def func(a, b):
    ...     return a + b
    >>> func(a=[1, 2], b=[3, 4])
    (4, 6)

    Providing a mapped argument as a scalar will disable mapping over that argument:

    >>> @map_over_arguments('a', 'b')
    ... def func(a, b):
    ...     return a + b
    >>> func(a=1, b=[2, 3])
    (3, 4)
    >>> func(a=1, b=2)
    3
    """

    def arg_mapper(func: Callable[P, RT]) -> Callable[P, MaybeTuple[RT]]:
        if validate_args:
            accepted_args = signature(func).parameters
            invalid_args = [arg for arg in map_args if arg not in accepted_args]
            if invalid_args:
                msg = (
                    "The following arguments are not accepted by the decorated "
                    f"function and cannot be mapped over: {invalid_args}"
                )
                raise ValueError(msg)

        def wrapper(*args, **kwargs):
            # Bind the arguments as they will be called to allow mapping over positional
            # or keyword arguments.
            bound_args = signature(func).bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Collect the mapped arguments that have mappable values
            to_map = {
                arg: val
                for arg, val in bound_args.arguments.items()
                if arg in map_args and isinstance(val, mappable)
            }
            if not to_map:
                return func(*args, **kwargs)

            num_mapped_vals = [len(v) for v in to_map.values()]
            if any([val < max(num_mapped_vals) for val in num_mapped_vals]):
                raise ValueError(
                    "All mapped arguments must be the same length or scalar."
                )

            # Group the mapped arguments for each call
            map_groups = [
                {**{k: v[i] for k, v in to_map.items()}}
                for i in range(max(num_mapped_vals))
            ]

            # Return one result per group of mapped values
            results = []
            for map_group in map_groups:
                bound_args.arguments.update(map_group)
                results.append(func(*bound_args.args, **bound_args.kwargs))
            return tuple(results)

        return wrapper

    return arg_mapper


def _get_threadpool_controller() -> threadpoolctl.ThreadpoolController:
    """Return the global threadpool controller instance."""
    global _threadpool_controller

    if _threadpool_controller is None:
        _threadpool_controller = threadpoolctl.ThreadpoolController()

    return _threadpool_controller


def limit_inner_threads(
    limits: int | None = 1, user_api: str | None = None
) -> Callable:
    """
    A decorator that limits the number of threads used by the decorated function.

    This is useful to avoid oversubscription when Dask workers that share threads
    apply thread-parallelized libraries such as OpenBLAS via NumPy and SciPy. Unlike
    `threadpoolctl.ThreadpoolController.wrap`, the decorated function remains
    serializable for use with distributed Dask schedulers.

    Parameters
    ----------
    limits : int | None, default 1
        The maximum number of threads to allow the decorated function to use. If None,
        no limit is applied.
    user_api : str | None, default None
        The thread-parallelized library to limit. If None, all supported libraries
        (OpenBLAS, MKL, etc.) will be limited.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if limits is None:
                return func(*args, **kwargs)
            controller = _get_threadpool_controller()
            with controller.limit(limits=limits, user_api=user_api):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def with_inputs_reshaped_to_ndim(ndim: int | None):
    """
    A decorator that reshapes input arrays to a given dimensionality.

    This allows functions designed for a specific dimensionality to generalize for
    n-dimensional inputs by 1) flattening or padding the dimensions of input arrays, and
    2) restoring the original dimensionality in output arrays. Dimensions are modified
    from left to right.

    Notes
    -----
    The decorated function must:
    1. Accept one or more arrays as the only positional arguments. Keyword arguments are
       passed through unmodified.
    2. Return one or more arrays with the same dimensionality and shape as its inputs,
       except for the last dimension which may change.

    Parameters
    ----------
    ndim : int | None
        The dimensionality expected by the decorated function. When `ndim` is less than
        the array dimensionality, dimensions are flattened from the left. When `ndim` is
        greater than the array dimensionality, new dimensions are added to the left. For
        example, an array of shape `(10, 10, 1)` with `ndim=2` is flattened to
        `(100, 1)`, while an array of shape `(10, 1)` with `ndim=3` is expanded to
        `(1, 10, 1)`. When `ndim=None`, the function is returned unmodified.

    Returns
    -------
    Callable
        The decorated function which will receive reshaped arrays and return outputs
        with restored dimensionality.
    """
    # Skip the decorator and return the unmodified function
    if ndim is None:
        return lambda func: func

    def decorator(func):
        @wraps(func)
        def validate_and_reshape(*arrays: NDArray, **kwargs) -> MaybeTuple[NDArray]:
            # No-op if called without inputs
            if not arrays:
                return func(*arrays, **kwargs)

            # Input shapes must be consistent to determine the output shape
            shapes = tuple(set([a.shape for a in arrays]))
            if len(shapes) > 1:
                raise ValueError("All arrays must have the same shape.")

            shape_in = shapes[0]
            ndim_in = len(shape_in)
            # Input is already the correct dimensionality - no need for reshaping
            if ndim_in == ndim:
                return func(*arrays, **kwargs)

            @map_over_arguments("out")
            def restore_dimensions(out: NDArray) -> NDArray:
                # For the output shape to be solvable, only one dimension can change.
                # Since we flatten/expand from the left, we allow the rightmost final
                # dimension to change.
                shape_out = tuple([*shape_in[:-1], -1])
                return out.reshape(shape_out)

            reshaped = [_reshape_to_ndim(a, ndim) for a in arrays]
            result = func(*reshaped, **kwargs)
            return restore_dimensions(result)

        return validate_and_reshape

    return decorator


def _reshape_to_ndim(array: NDArray, ndim: int) -> NDArray:
    """
    Reshape an array to ndim, flattening or expanding dimensions from left to right.
    """
    if ndim < 1:
        raise ValueError("Cannot reshape to ndim < 1")

    shape_in = array.shape
    ndim_in = len(shape_in)

    # Flatten extra dimensions
    if ndim < ndim_in:
        keep_dims = shape_in[ndim_in - ndim + 1 :]
        new_dims: tuple[int, ...] = (-1,)

    # Expand new dimensions
    else:
        keep_dims = shape_in
        new_dims = tuple([1] * (ndim - ndim_in))

    shape_out = (*new_dims, *keep_dims)
    return array.reshape(shape_out)

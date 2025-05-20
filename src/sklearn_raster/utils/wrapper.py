from functools import wraps
from inspect import signature
from typing import Callable, Generic

from typing_extensions import Concatenate, TypeVar

from ..types import RT, AnyType, MaybeTuple, P


class AttrWrapper(Generic[AnyType]):
    """A transparent object wrapper that accesses a wrapped object's attributes."""

    _wrapped: AnyType

    def __init__(self, wrapped: AnyType):
        self._wrapped = wrapped

    def __getattr__(self, name: str):
        return getattr(self._wrapped, name)

    @property
    def __dict__(self):
        return self._wrapped.__dict__


GenericWrapper = TypeVar("GenericWrapper", bound=AttrWrapper)


def requires_implementation(
    func: Callable[Concatenate[GenericWrapper, P], RT],
) -> Callable[Concatenate[GenericWrapper, P], RT]:
    """
    A decorator that raises if the wrapped instance doesn't implement the given method.
    """
    return requires_attributes(func.__name__)(func)


def requires_attributes(
    *attrs: str,
) -> Callable[
    [Callable[Concatenate[GenericWrapper, P], RT]],
    Callable[Concatenate[GenericWrapper, P], RT],
]:
    """
    A decorator that raises if the wrapped instance is missing required attributes.
    """

    def decorator(
        func: Callable[Concatenate[GenericWrapper, P], RT],
    ) -> Callable[Concatenate[GenericWrapper, P], RT]:
        @wraps(func)
        def wrapper(self: GenericWrapper, *args, **kwargs):
            for attr in attrs:
                if hasattr(self._wrapped, attr):
                    continue
                wrapped_class = self._wrapped.__class__.__name__
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

from functools import wraps
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


def check_wrapper_implements(
    func: Callable[Concatenate[GenericWrapper, P], RT],
) -> Callable[Concatenate[GenericWrapper, P], RT]:
    """Decorator that raises if the wrapped instance doesn't implement the method."""

    @wraps(func)
    def wrapper(self: GenericWrapper, *args, **kwargs):
        if not hasattr(self._wrapped, func.__name__):
            wrapped_class = self._wrapped.__class__.__name__
            msg = f"{wrapped_class} does not implement {func.__name__}."
            raise NotImplementedError(msg)

        return func(self, *args, **kwargs)

    return wrapper


def map_over_arguments(*map_args: str, mappable=(tuple, list)):
    """
    A decorator that allows a function to map over selected arguments.

    When the selected arguments are mappable, the function will be called once with
    each value and a tuple of results will be returned. Non-mapped arguments and scalar
    mapped arguments will be passed to each call. To map over an argument, it must be
    provided by name.

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
        def wrapper(*args, **kwargs):
            # Collect the mapped arguments that have mappable values
            to_map = {
                name: kwargs.pop(name)
                for name in map_args
                if isinstance(kwargs.get(name), mappable)
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
            return tuple(
                [func(*args, **map_group, **kwargs) for map_group in map_groups]
            )

        return wrapper

    return arg_mapper

from functools import wraps
from typing import Callable, Generic

from typing_extensions import Concatenate, TypeVar

from ..types import RT, AnyType, MaybeTuple, P, Self, T


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


def map_function_over_tuples(
    func: Callable[Concatenate[T, P], T],
) -> Callable[Concatenate[MaybeTuple[T], P], MaybeTuple[T]]:
    """
    Decorate a function that accepts and returns type T to also accept and return tuples
    of type T by mapping the function over each element.

    The function will be mapped over the first positional argument.
    """

    def wrapper(arg: MaybeTuple[T], *args: P.args, **kwargs: P.kwargs) -> MaybeTuple[T]:
        if isinstance(arg, tuple):
            return tuple(func(arg, *args, **kwargs) for arg in arg)
        return func(arg, *args, **kwargs)

    return wrapper


def map_method_over_tuples(
    func: Callable[Concatenate[Self, T, P], T],
) -> Callable[Concatenate[Self, MaybeTuple[T], P], MaybeTuple[T]]:
    """
    Decorate a method that accepts and returns type T to also accept and return tuples
    of type T by mapping the method over each element.

    The method will be mapped over the second positional argument (i.e. excluding
    `self`).
    """

    def wrapper(
        self: Self, arg: MaybeTuple[T], *args: P.args, **kwargs: P.kwargs
    ) -> MaybeTuple[T]:
        if isinstance(arg, tuple):
            return tuple(func(self, arg, *args, **kwargs) for arg in arg)
        return func(self, arg, *args, **kwargs)

    return wrapper

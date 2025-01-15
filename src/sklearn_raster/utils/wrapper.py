from functools import wraps
from typing import Callable, Generic

from typing_extensions import Concatenate, ParamSpec, TypeVar

from ..types import AnyType

RT = TypeVar("RT")
P = ParamSpec("P")


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

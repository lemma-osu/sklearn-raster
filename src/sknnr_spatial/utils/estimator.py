from functools import wraps
from typing import Callable, Concatenate, Generic, ParamSpec, TypeVar

from sklearn.base import BaseEstimator
from sklearn.utils.validation import NotFittedError, check_is_fitted

from ..image._base import ImageType
from ..types import AnyType
from .image import is_image_type

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


# Callable that takes an AttrWrapper instance and other generic parameters and returns a
# generic type.
GenericWrappedCallable = Callable[Concatenate[AttrWrapper, P], RT]
# Callable that takes an AttrWrapper instance, an ImageType, and other generic
# parameters and returns a generic type.
GenericWrappedImageCallable = Callable[Concatenate[AttrWrapper, ImageType, P], RT]


def check_wrapper_implements(func: GenericWrappedCallable) -> GenericWrappedCallable:
    """Decorator that raises if the wrapped instance doesn't implement the method."""

    @wraps(func)
    def wrapper(self: AttrWrapper, *args, **kwargs):
        if not hasattr(self._wrapped, func.__name__):
            wrapped_class = self._wrapped.__class__.__name__
            msg = f"{wrapped_class} does not implement {func.__name__}."
            raise NotImplementedError(msg)

        return func(self, *args, **kwargs)

    return wrapper


def image_or_fallback(func: GenericWrappedImageCallable) -> GenericWrappedImageCallable:
    """Decorator that calls the wrapped method for non-image X arrays."""

    @wraps(func)
    def wrapper(self: AttrWrapper, X_image: ImageType, *args, **kwargs):
        if not is_image_type(X_image):
            return getattr(self._wrapped, func.__name__)(X_image, *args, **kwargs)

        return func(self, X_image, *args, **kwargs)

    return wrapper


def is_fitted(estimator: BaseEstimator) -> bool:
    """Return whether an estimator is fitted or not."""
    try:
        check_is_fitted(estimator)
        return True
    except NotFittedError:
        return False

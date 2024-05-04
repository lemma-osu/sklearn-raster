from functools import wraps
from typing import Callable, Generic

from sklearn.base import BaseEstimator
from sklearn.utils.validation import NotFittedError, check_is_fitted

from ..types import AnyType, ImageType
from .image import is_image_type


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


def check_wrapper_implements(func: Callable) -> Callable:
    """Decorator that raises if the wrapped instance doesn't implement the method."""

    @wraps(func)
    def wrapper(self: AttrWrapper, *args, **kwargs):
        if not hasattr(self._wrapped, func.__name__):
            wrapped_class = self._wrapped.__class__.__name__
            msg = f"{wrapped_class} does not implement {func.__name__}."
            raise NotImplementedError(msg)

        return func(self, *args, **kwargs)

    return wrapper


def check_is_x_image(func: Callable) -> Callable:
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

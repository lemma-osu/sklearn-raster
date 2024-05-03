from __future__ import annotations

from functools import wraps
from typing import Callable
from warnings import warn

from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import NotFittedError, check_is_fitted

from .image._base import is_image_type, kneighbors, predict


class _AttrWrapper:
    """A transparent object wrapper that accesses a wrapped object's attributes."""

    def __init__(self, wrapped):
        self._wrapped = wrapped

    def __getattr__(self, name: str):
        return getattr(self._wrapped, name)

    @property
    def __dict__(self):
        return self._wrapped.__dict__

    @staticmethod
    def check_for_wrapped_method(func: Callable) -> Callable:
        """Check that the method is implemented by the caller's wrapped instance."""

        @wraps(func)
        def wrapper(self: _AttrWrapper, *args, **kwargs):
            if not hasattr(self._wrapped, func.__name__):
                wrapped_class = self._wrapped.__class__.__name__
                msg = f"{wrapped_class} does not implement {func.__name__}."
                raise NotImplementedError(msg)

            return func(self, *args, **kwargs)

        return wrapper


class SpatialEstimator(_AttrWrapper):
    def __init__(self, wrapped: BaseEstimator):
        self._wrapped = self._reset_estimator(wrapped)

    @staticmethod
    def _reset_estimator(estimator: BaseEstimator) -> BaseEstimator:
        """Take an estimator and reset and warn if it was previously fitted."""
        try:
            check_is_fitted(estimator)
            warn(
                "Wrapping estimator that has already been fit. The estimator must be "
                "fit again after wrapping.",
                stacklevel=2,
            )

            return clone(estimator)
        except NotFittedError:
            return estimator

    @_AttrWrapper.check_for_wrapped_method
    def fit(self, X, y=None, **kwargs) -> SpatialEstimator:
        """Fit the wrapped estimator and return the wrapper."""
        self._wrapped = self._wrapped.fit(X, y, **kwargs)
        # TODO: Store all needed metadata, e.g. n targets, target names, etc.
        # TODO: Override the builtin feature name warning somehow
        return self

    @_AttrWrapper.check_for_wrapped_method
    def predict(self, X):
        # Allow predicting with non-image data using the wrapped estimator
        if not is_image_type(X):
            return self._wrapped.predict(X)

        return predict(X, estimator=self._wrapped)

    @_AttrWrapper.check_for_wrapped_method
    def kneighbors(self, X, return_distance=True, **kwargs):
        # Allow kneighbors with non-image data using the wrapped estimator
        if not is_image_type(X):
            return self._wrapped.kneighbors(
                X, return_distance=return_distance, **kwargs
            )

        return kneighbors(
            X, return_distance=return_distance, estimator=self._wrapped, **kwargs
        )


def wrap(estimator: BaseEstimator) -> SpatialEstimator:
    return SpatialEstimator(estimator)

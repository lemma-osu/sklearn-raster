from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Callable, Generic
from warnings import warn

from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import NotFittedError, check_is_fitted

from .image._base import is_image_type, kneighbors, predict
from .types import AnyType, EstimatorType

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


@dataclass
class FittedMetadata:
    """Metadata from a fitted estimator."""

    n_targets: int
    target_names: tuple[str, ...]


class _AttrWrapper(Generic[AnyType]):
    """A transparent object wrapper that accesses a wrapped object's attributes."""

    _wrapped: AnyType

    def __init__(self, wrapped: AnyType):
        self._wrapped = wrapped

    def __getattr__(self, name: str):
        return getattr(self._wrapped, name)

    @property
    def __dict__(self):
        return self._wrapped.__dict__

    @staticmethod
    def _check_for_wrapped_method(func: Callable) -> Callable:
        """Check that the method is implemented by the caller's wrapped instance."""

        @wraps(func)
        def wrapper(self: _AttrWrapper, *args, **kwargs):
            if not hasattr(self._wrapped, func.__name__):
                wrapped_class = self._wrapped.__class__.__name__
                msg = f"{wrapped_class} does not implement {func.__name__}."
                raise NotImplementedError(msg)

            return func(self, *args, **kwargs)

        return wrapper


class ImageEstimator(_AttrWrapper[EstimatorType]):
    """
    An sklearn-compatible estimator wrapper with overriden methods for image data.

    Parameters
    ----------
    wrapped : BaseEstimator
        An sklearn-compatible estimator to wrap with image methods. Fitted estimators
        will be reset when wrapped and must be re-fit after wrapping.
    """

    _wrapped: EstimatorType
    _wrapped_meta: FittedMetadata

    def __init__(self, wrapped: EstimatorType):
        super().__init__(self._reset_estimator(wrapped))

    @staticmethod
    def _reset_estimator(estimator: EstimatorType) -> EstimatorType:
        """Take an estimator and reset and warn if it was previously fitted."""
        if is_fitted(estimator):
            warn(
                "Wrapping estimator that has already been fit. The estimator must be "
                "fit again after wrapping.",
                stacklevel=2,
            )
            return clone(estimator)

        return estimator

    def _get_n_targets(self, y: np.ndarray | pd.DataFrame | pd.Series) -> int:
        """Get the number of targets used to fit the estimator."""
        if y.ndim == 1:
            return 1

        return y.shape[-1]

    def _get_target_names(
        self, y: np.ndarray | pd.DataFrame | pd.Series
    ) -> tuple[str, ...]:
        """Get the target names used to fit the estimator, if available."""
        # Dataframe
        if hasattr(y, "columns"):
            return tuple(y.columns)

        # Series
        if hasattr(y, "name"):
            return tuple([y.name])

        # Default to sequential identifiers
        return tuple([f"b{i}" for i in range(self._get_n_targets(y))])

    @_AttrWrapper._check_for_wrapped_method
    def fit(self, X, y=None, **kwargs) -> ImageEstimator[EstimatorType]:
        """Fit the wrapped estimator and return the wrapper."""
        # Squeeze extra y dimensions. This will convert from shape (n_samples, 1) which
        # causes inconsistent output shapes with different sklearn estimators, to
        # (n_samples,), which has a consistent output shape.
        y = y.squeeze()
        self._wrapped = self._wrapped.fit(X, y, **kwargs)

        self._wrapped_meta = FittedMetadata(
            n_targets=self._get_n_targets(y),
            target_names=self._get_target_names(y),
        )

        # TODO: Override the builtin feature name warning somehow
        return self

    @_AttrWrapper._check_for_wrapped_method
    def predict(self, X):
        # Allow predicting with non-image data using the wrapped estimator
        if not is_image_type(X):
            return self._wrapped.predict(X)

        return predict(X, estimator=self)

    @_AttrWrapper._check_for_wrapped_method
    def kneighbors(self, X, return_distance=True, **kwargs):
        # Allow kneighbors with non-image data using the wrapped estimator
        if not is_image_type(X):
            return self._wrapped.kneighbors(
                X, return_distance=return_distance, **kwargs
            )

        return kneighbors(X, return_distance=return_distance, estimator=self, **kwargs)


def wrap(estimator: EstimatorType) -> ImageEstimator[EstimatorType]:
    """
    Wrap an sklearn-compatible estimator with overriden methods for image data.

    Parameters
    ----------
    estimator : BaseEstimator
        An sklearn-compatible estimator to wrap with image methods. Fitted estimators
        will be reset when wrapped and must be re-fit after wrapping.

    Returns
    -------
    ImageEstimator
        An estimator with relevant methods overriden to work with image data, e.g.
        `predict` and `kneighbors`. Methods will continue to work with non-image data
        and non-overriden methods and attributes will be unchanged.

    Examples
    --------
    Instantiate an estimator, wrap it, then fit as usual:

    >>> from sklearn.neighbors import KNeighborsRegressor
    >>> from sknnr_spatial.datasets import load_swo_ecoplot
    >>> X_img, X, y = load_swo_ecoplot()
    >>> est = wrap(KNeighborsRegressor(n_neighbors=3)).fit(X, y)

    Use a wrapped estimator to predict from image data stored in Numpy or Xarray arrays:

    >>> pred = est.predict(X_img)
    >>> pred.shape
    (128, 128, 25)
    """
    return ImageEstimator(estimator)


def is_fitted(estimator: BaseEstimator) -> bool:
    """Return whether an estimator is fitted or not."""
    try:
        check_is_fitted(estimator)
        return True
    except NotFittedError:
        return False

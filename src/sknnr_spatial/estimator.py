from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from warnings import warn

from sklearn.base import clone

from .image._base import kneighbors, predict
from .types import EstimatorType
from .utils.estimator import (
    AttrWrapper,
    check_is_x_image,
    check_wrapper_implements,
    is_fitted,
)

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from .types import ImageType, NoDataType


@dataclass
class FittedMetadata:
    """Metadata from a fitted estimator."""

    n_targets: int
    target_names: tuple[str | int, ...]


class ImageEstimator(AttrWrapper[EstimatorType]):
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
    ) -> tuple[str | int, ...]:
        """Get the target names used to fit the estimator, if available."""
        # Dataframe
        if hasattr(y, "columns"):
            return tuple(y.columns)

        # Series
        if hasattr(y, "name"):
            return tuple([y.name])

        # Default to sequential identifiers
        return tuple(range(self._get_n_targets(y)))

    @check_wrapper_implements
    def fit(self, X, y=None, **kwargs) -> ImageEstimator[EstimatorType]:
        """
        Fit an estimator from a training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression). Single-output targets of shape (n_samples, 1) will be squeezed
            to shape (n_samples,) to allow consistent prediction across all estimators.

        Returns
        -------
        self : ImageEstimator
            The wrapper around the fitted estimator.
        """
        # Squeeze extra y dimensions. This will convert from shape (n_samples, 1) which
        # causes inconsistent output shapes with different sklearn estimators, to
        # (n_samples,), which has a consistent output shape.
        y = y.squeeze()
        self._wrapped = self._wrapped.fit(X, y, **kwargs)

        self._wrapped_meta = FittedMetadata(
            n_targets=self._get_n_targets(y),
            target_names=self._get_target_names(y),
        )

        return self

    @check_wrapper_implements
    @check_is_x_image
    def predict(
        self, X_image: ImageType, *, nodata_vals: NoDataType = None
    ) -> ImageType:
        """
        Predict target(s) for X_image.

        Notes
        -----
        If X_image is not an image, the estimator's unmodified predict method will be
        called instead.

        Parameters
        ----------
        X_image : Numpy or Xarray image with 3 dimensions (y, x, band)
            The input image. Features in the band dimension should correspond with the
            features used to fit the estimator.
        nodata_vals : float or sequence of floats, optional
            NoData values to mask in the output image. A single value will be broadcast
            to all bands while sequences of values will be assigned band-wise. If None,
            values will be inferred if possible based on image metadata.

        Returns
        -------
        y_image : Numpy or Xarray image with 3 dimensions (y, x, targets)
            The predicted values.
        """
        return predict(X_image, estimator=self, nodata_vals=nodata_vals)

    @check_wrapper_implements
    @check_is_x_image
    def kneighbors(
        self, X_image: ImageType, *, nodata_vals: NoDataType = None, **kneighbors_kwargs
    ) -> ImageType | tuple[ImageType, ImageType]:
        """
        Find the K-neighbors of each pixel in an image.

        Returns indices of and distances to the neighbors for each pixel.

        Notes
        -----
        If X_image is not an image, the estimator's unmodified kneighbors method will be
        called instead.

        Parameters
        ----------
        X_image : Numpy or Xarray image with 3 dimensions (y, x, band)
            The input image. Features in the band dimension should correspond with the
            features used to fit the estimator.
        nodata_vals : float or sequence of floats, optional
            NoData values to mask in the output image. A single value will be broadcast
            to all bands while sequences of values will be assigned band-wise. If None,
            values will be inferred if possible based on image metadata.
        **kneighbors_kwargs
            Additional arguments passed to the estimator's kneighbors method, e.g.
            `return_distance`.

        Returns
        -------
        neigh_dist : Numpy or Xarray image with 3 dimensions (y, x, neighbor)
            Array representing the lengths to points, only present if
            return_distance=True.
        neigh_ind : Numpy or Xarray image with 3 dimensions (y, x, neighbor)
            Indices of the nearest points in the population matrix.
        """
        return kneighbors(
            X_image, estimator=self, nodata_vals=nodata_vals, **kneighbors_kwargs
        )


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

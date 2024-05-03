from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ._base import ImagePreprocessor, kneighbors, predict

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from sklearn.base import BaseEstimator
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

    from ..estimator import ImageEstimator


class NDArrayPreprocessor(ImagePreprocessor):
    """Pre-processor for multi-band NumPy NDArrays."""

    _backend = np
    band_dim = -1

    def _flatten(self, image: NDArray) -> NDArray:
        """Flatten the array from (y, x, bands) to (pixels, bands)."""
        # Reshape typically returns a view rather than a copy. To avoid modifying the
        # original input array during masking and filling, make the flat array a copy.
        return image.reshape(-1, self.n_bands).copy()

    def unflatten(self, flat_image: NDArray, *, apply_mask=True) -> NDArray:
        if apply_mask:
            flat_image = self._fill_nodata(flat_image, np.nan)

        return flat_image.reshape(*self.image.shape[:2], -1)


@predict.register(np.ndarray)
def _predict_from_ndarray(
    X_image: NDArray, *, estimator: ImageEstimator[BaseEstimator], nodata_vals=None
) -> NDArray:
    """Predict attributes from an array of X_image."""
    check_is_fitted(estimator)
    preprocessor = NDArrayPreprocessor(X_image, nodata_vals=nodata_vals)

    # TODO: Deal with sklearn warning about missing feature names
    y_pred_flat = estimator._wrapped.predict(preprocessor.flat)

    return preprocessor.unflatten(y_pred_flat, apply_mask=True)


@kneighbors.register(np.ndarray)
def _kneighbors_from_ndarray(
    X_image: NDArray,
    *,
    estimator: ImageEstimator[KNeighborsRegressor | KNeighborsClassifier],
    nodata_vals=None,
    **kneighbors_kwargs,
) -> NDArray:
    check_is_fitted(estimator)
    preprocessor = NDArrayPreprocessor(X_image, nodata_vals=nodata_vals)
    return_distance = kneighbors_kwargs.pop("return_distance", True)

    result = estimator._wrapped.kneighbors(
        preprocessor.flat,
        return_distance=return_distance,
        **kneighbors_kwargs,
    )
    if return_distance:
        dist, nn = result

        dist = preprocessor.unflatten(dist)
        nn = preprocessor.unflatten(nn)

        return dist, nn

    return preprocessor.unflatten(result)

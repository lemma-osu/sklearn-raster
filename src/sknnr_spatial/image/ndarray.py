import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils.validation import check_is_fitted

from ._base import ImagePreprocessor, kneighbors, predict


class NDArrayPreprocessor(ImagePreprocessor):
    """
    Pre-processor multi-band NumPy NDArrays.
    """

    _backend = np
    band_dim = -1

    def _flatten(self, image: NDArray) -> NDArray:
        """Flatten the array from (y, x, bands) to (pixels, bands)."""
        return image.reshape(-1, self.n_bands)

    def unflatten(self, flat_image: NDArray, *, apply_mask=True) -> NDArray:
        if apply_mask:
            flat_image = self._fill_nodata(flat_image, np.nan)

        return flat_image.reshape(*self.image.shape[:2], -1)


@predict.register(np.ndarray)
def _predict_from_ndarray(
    X_image: NDArray, *, estimator: BaseEstimator, nodata_vals=None
) -> NDArray:
    """Predict attributes from an array of X_image."""
    check_is_fitted(estimator)
    preprocessor = NDArrayPreprocessor(X_image, nodata_vals=nodata_vals)

    # TODO: Deal with sklearn warning about missing feature names
    y_pred_flat = estimator.predict(preprocessor.flat)

    return preprocessor.unflatten(y_pred_flat, apply_mask=True)


@kneighbors.register(np.ndarray)
def _kneighbors_from_ndarray(
    X_image: NDArray,
    *,
    estimator: KNeighborsRegressor,
    nodata_vals=None,
    **kneighbors_kwargs,
) -> NDArray:
    check_is_fitted(estimator)
    preprocessor = NDArrayPreprocessor(X_image, nodata_vals=nodata_vals)
    return_distance = kneighbors_kwargs.pop("return_distance", True)

    result = estimator.kneighbors(
        preprocessor.flat,
        return_distance=True,
        **kneighbors_kwargs,
    )
    if return_distance:
        dist, nn = result

        dist = preprocessor.unflatten(dist)
        nn = preprocessor.unflatten(nn)

        return dist, nn

    return preprocessor.unflatten(result)

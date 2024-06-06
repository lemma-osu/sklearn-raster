from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ._base import ImagePreprocessor, ImageWrapper

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator
    from sklearn.neighbors._base import KNeighborsMixin

    from ..estimator import ImageEstimator


class NDArrayPreprocessor(ImagePreprocessor):
    """Pre-processor for multi-band NumPy NDArrays."""

    _backend = np
    band_dim = -1
    band_names = np.array([])

    def _flatten(self, image: NDArray) -> NDArray:
        """Flatten the array from (y, x, bands) to (pixels, bands)."""
        # Reshape typically returns a view rather than a copy. To avoid modifying the
        # original input array during masking and filling, make the flat array a copy.
        return image.reshape(-1, self.n_bands).copy()

    def unflatten(self, flat_image: NDArray, *, apply_mask=True) -> NDArray:
        if apply_mask:
            flat_image = self._fill_nodata(flat_image, np.nan)

        return flat_image.reshape(*self.image.shape[:2], -1)


class NDArrayWrapper(ImageWrapper[NDArray]):
    preprocessor_cls = NDArrayPreprocessor

    def predict(
        self,
        *,
        estimator: ImageEstimator[BaseEstimator],
    ) -> NDArray:
        """Predict attributes from an array of X_image."""
        y_pred_flat = estimator._wrapped.predict(self.preprocessor.flat)

        # Reshape from (n_samples,) to (n_samples, 1)
        if estimator._wrapped_meta.n_targets == 1:
            y_pred_flat = y_pred_flat.reshape(-1, 1)

        return self.preprocessor.unflatten(y_pred_flat, apply_mask=True)

    def kneighbors(
        self,
        *,
        estimator: ImageEstimator[KNeighborsMixin],
        n_neighbors: int | None = None,
        return_distance: bool = True,
        **kneighbors_kwargs,
    ) -> NDArray | tuple[NDArray, NDArray]:
        result = estimator._wrapped.kneighbors(
            self.preprocessor.flat,
            n_neighbors=n_neighbors,
            return_distance=return_distance,
            **kneighbors_kwargs,
        )
        if return_distance:
            dist, nn = result

            dist = self.preprocessor.unflatten(dist)
            nn = self.preprocessor.unflatten(nn)

            return dist, nn

        return self.preprocessor.unflatten(result)

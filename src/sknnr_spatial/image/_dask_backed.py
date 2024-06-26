from __future__ import annotations

from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
from sklearn.utils.validation import check_is_fitted

from ..types import DaskBackedType
from ..utils.estimator import suppress_feature_name_warnings
from ._base import ImageWrapper

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator
    from sklearn.neighbors._base import KNeighborsMixin

    from ..estimator import ImageEstimator
    from .dataarray import DataArrayPreprocessor
    from .dataset import DatasetPreprocessor

ESTIMATOR_OUTPUT_DTYPES: dict[str, np.dtype] = {
    "classifier": np.int32,
    "clusterer": np.int32,
    "regressor": np.float64,
}


class DaskBackedWrapper(ImageWrapper[DaskBackedType]):
    """A wrapper around a Dask-backed image that provides sklearn methods."""

    preprocessor_cls: type[DataArrayPreprocessor] | type[DatasetPreprocessor]
    preprocessor: DataArrayPreprocessor | DatasetPreprocessor

    def predict(
        self,
        *,
        estimator: ImageEstimator[BaseEstimator],
    ) -> DaskBackedType:
        """Generic predict wrapper for Dask-backed arrays."""
        check_is_fitted(estimator)
        meta = estimator._wrapped_meta

        if single_output := meta.n_targets == 1:
            signature = "(x)->()"
            output_sizes = {}
        else:
            signature = "(x)->(y)"
            output_sizes = {"y": meta.n_targets}

        # Any estimator with an undefined type should fall back to floating
        # point for safety.
        estimator_type = getattr(estimator, "_estimator_type", "")
        output_dtype = ESTIMATOR_OUTPUT_DTYPES.get(estimator_type, np.float64)

        y_pred = self._apply_gufunc(
            estimator._wrapped.predict,
            signature=signature,
            output_sizes=output_sizes,
            output_dtypes=[output_dtype],
        )

        # Reshape from (n_samples,) to (n_samples, 1)
        if single_output:
            y_pred = y_pred.reshape(-1, 1)

        return self.preprocessor.unflatten(y_pred, var_names=list(meta.target_names))

    def kneighbors(
        self,
        *,
        estimator: ImageEstimator[KNeighborsMixin],
        n_neighbors: int | None = None,
        return_distance: bool = True,
        **kneighbors_kwargs,
    ) -> DaskBackedType | tuple[DaskBackedType, DaskBackedType]:
        """Generic kneighbors wrapper for Dask-backed arrays."""
        k = n_neighbors if n_neighbors is not None else estimator.n_neighbors

        var_names = [f"k{i + 1}" for i in range(k)]

        # Set the expected gufunc output depending on whether distances will be included
        signature = "(x)->(k)" if not return_distance else "(x)->(k),(k)"
        output_dtypes: list[type] = [int] if not return_distance else [float, int]

        result = self._apply_gufunc(
            estimator._wrapped.kneighbors,
            signature=signature,
            output_sizes={"k": k},
            output_dtypes=output_dtypes,
            n_neighbors=n_neighbors,
            return_distance=return_distance,
            **kneighbors_kwargs,
        )

        if return_distance:
            dist, nn = result

            dist = self.preprocessor.unflatten(dist, var_names=var_names)
            nn = self.preprocessor.unflatten(nn, var_names=var_names)

            return dist, nn

        return self.preprocessor.unflatten(result, var_names=var_names)

    def _apply_gufunc(self, func, *, signature, output_sizes, output_dtypes, **kwargs):
        """Apply a gufunc to the image across bands."""
        # sklearn estimator methods like `predict` may warn about missing feature
        # names because this passes unnamed arrays. We can suppress those and let
        # the wrapper handle feature name checks.
        suppressed_func = suppress_feature_name_warnings(func)

        return da.apply_gufunc(
            suppressed_func,
            signature,
            self.preprocessor.flat,
            output_sizes=output_sizes,
            output_dtypes=output_dtypes,
            axis=self.preprocessor.flat_band_dim,
            allow_rechunk=True,
            **kwargs,
        )

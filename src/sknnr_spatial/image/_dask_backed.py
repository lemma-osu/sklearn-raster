from __future__ import annotations

from typing import TYPE_CHECKING

import dask.array as da
from sklearn.utils.validation import check_is_fitted

from ..types import DaskBackedType
from ._base import ImageWrapper

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

    from ..estimator import ImageEstimator
    from .dataarray import DataArrayPreprocessor
    from .dataset import DatasetPreprocessor


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

        y_pred = da.apply_gufunc(
            estimator._wrapped.predict,
            signature,
            self.preprocessor.flat,
            axis=self.preprocessor.flat_band_dim,
            output_dtypes=[float],
            output_sizes=output_sizes,
            allow_rechunk=True,
        )

        # Reshape from (n_samples,) to (n_samples, 1)
        if single_output:
            y_pred = y_pred.reshape(-1, 1)

        return self.preprocessor.unflatten(y_pred, var_names=list(meta.target_names))

    def kneighbors(
        self,
        *,
        estimator: ImageEstimator[KNeighborsRegressor | KNeighborsClassifier],
        **kneighbors_kwargs,
    ) -> DaskBackedType | tuple[DaskBackedType, DaskBackedType]:
        """Generic kneighbors wrapper for Dask-backed arrays."""
        check_is_fitted(estimator)
        return_distance = kneighbors_kwargs.pop("return_distance", True)

        k = estimator.n_neighbors
        var_names = [f"k{i + 1}" for i in range(k)]

        # Set the expected gufunc output depending on whether distances will be included
        signature = "(x)->(k)" if not return_distance else "(x)->(k),(k)"
        output_dtypes: list[type] = [int] if not return_distance else [float, int]

        result = da.apply_gufunc(
            estimator._wrapped.kneighbors,
            signature,
            self.preprocessor.flat,
            output_sizes={"k": k},
            output_dtypes=output_dtypes,
            axis=self.preprocessor.flat_band_dim,
            allow_rechunk=True,
            return_distance=return_distance,
            **kneighbors_kwargs,
        )

        if return_distance:
            dist, nn = result

            dist = self.preprocessor.unflatten(dist, var_names=var_names)
            nn = self.preprocessor.unflatten(nn, var_names=var_names)

            return dist, nn

        return self.preprocessor.unflatten(result, var_names=var_names)

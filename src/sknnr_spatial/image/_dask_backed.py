from __future__ import annotations

from typing import TYPE_CHECKING

import dask.array as da
from sklearn.utils.validation import check_is_fitted

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

    from ..estimator import ImageEstimator
    from ..types import DaskBackedType
    from .dataarray import DataArrayPreprocessor
    from .dataset import DatasetPreprocessor


class TargetInferenceError(Exception):
    """Raised when the number of targets cannot be inferred from an estimator."""


def predict_from_dask_backed_array(
    X_image: DaskBackedType,
    *,
    estimator: ImageEstimator[BaseEstimator],
    preprocessor_cls: type[DataArrayPreprocessor] | type[DatasetPreprocessor],
    nodata_vals=None,
) -> DaskBackedType:
    """Generic predict wrapper for Dask-backed arrays."""
    check_is_fitted(estimator)
    preprocessor = preprocessor_cls(X_image, nodata_vals=nodata_vals)
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
        preprocessor.flat,
        axis=preprocessor.flat_band_dim,
        output_dtypes=[float],
        output_sizes=output_sizes,
        allow_rechunk=True,
    )

    # Reshape from (n_samples,) to (n_samples, 1)
    if single_output:
        y_pred = y_pred.reshape(-1, 1)

    return preprocessor.unflatten(y_pred, var_names=list(meta.target_names))


def kneighbors_from_dask_backed_array(
    X_image: DaskBackedType,
    *,
    estimator: ImageEstimator[KNeighborsRegressor | KNeighborsClassifier],
    preprocessor_cls: type[DataArrayPreprocessor] | type[DatasetPreprocessor],
    nodata_vals=None,
    **kneighbors_kwargs,
) -> DaskBackedType:
    """Generic kneighbors wrapper for Dask-backed arrays."""
    check_is_fitted(estimator)
    preprocessor = preprocessor_cls(X_image, nodata_vals=nodata_vals)
    return_distance = kneighbors_kwargs.pop("return_distance", True)

    k = estimator.n_neighbors
    var_names = [f"k{i + 1}" for i in range(k)]

    # Set the expected gufunc output depending on whether distances will be included
    signature = "(x)->(k)" if not return_distance else "(x)->(k),(k)"
    output_dtypes: list[type] = [int] if not return_distance else [float, int]

    result = da.apply_gufunc(
        estimator._wrapped.kneighbors,
        signature,
        preprocessor.flat,
        output_sizes={"k": k},
        output_dtypes=output_dtypes,
        axis=preprocessor.flat_band_dim,
        allow_rechunk=True,
        return_distance=return_distance,
        **kneighbors_kwargs,
    )

    if return_distance:
        dist, nn = result

        dist = preprocessor.unflatten(dist, var_names=var_names)
        nn = preprocessor.unflatten(nn, var_names=var_names)

        return dist, nn

    return preprocessor.unflatten(result, var_names=var_names)

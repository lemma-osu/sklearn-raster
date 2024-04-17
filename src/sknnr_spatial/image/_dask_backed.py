from __future__ import annotations

from typing import TYPE_CHECKING

import dask.array as da
from sklearn.utils.validation import check_is_fitted

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray
    from sklearn.base import BaseEstimator
    from sklearn.neighbors import KNeighborsRegressor

    from ..types import DaskBackedType
    from .dataarray import DataArrayPreprocessor
    from .dataset import DatasetPreprocessor


class TargetInferenceError(Exception):
    """Raised when the number of targets cannot be inferred from an estimator."""


def predict_from_dask_backed_array(
    X_image: DaskBackedType,
    *,
    estimator: BaseEstimator,
    y=None,
    preprocessor_cls: type[DataArrayPreprocessor] | type[DatasetPreprocessor],
    nodata_vals=None,
) -> DaskBackedType:
    """Generic predict wrapper for Dask-backed arrays."""
    check_is_fitted(estimator)
    preprocessor = preprocessor_cls(X_image, nodata_vals=nodata_vals)

    try:
        n_targets = _infer_num_targets(y, estimator)
        target_names = _infer_target_names(y, n_targets)
    except TargetInferenceError:
        msg = (
            "The number of targets could not be inferred from the estimator. Pass a "
            "`y` array or dataframe used to fit the estimator."
        )
        raise ValueError(msg) from None

    y_pred = da.apply_gufunc(
        estimator.predict,
        "(x)->(y)",
        preprocessor.flat,
        axis=preprocessor.flat_band_dim,
        output_dtypes=[float],
        output_sizes={"y": n_targets},
        allow_rechunk=True,
    )

    return preprocessor.unflatten(y_pred, var_names=target_names)


def kneighbors_from_dask_backed_array(
    X_image: DaskBackedType,
    *,
    estimator: KNeighborsRegressor,
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
        estimator.kneighbors,
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


def _infer_target_names(y: NDArray | pd.DataFrame | None, n_targets: int) -> list[str]:
    # Create sequential numeric target names
    if y is None or not hasattr(y, "columns"):
        return [f"b{i}" for i in range(n_targets)]

    return y.columns


def _infer_num_targets(y, estimator: BaseEstimator) -> int:
    """
    scikit-learn doesn't have a consistent standard for storing the number of targets
    in a fitted estimator, but there are a number of common places we can look.
    """
    if y is not None:
        return y.shape[-1]

    tags = estimator._get_tags() if hasattr(estimator, "_get_tags") else {}

    # Single output estimators can only be trained with a single feature
    if tags.get("multioutput") is False:
        return 1

    # KNeighborsRegressor
    if hasattr(estimator, "_y"):
        return estimator._y.shape[-1]

    # RandomForestRegressor
    if hasattr(estimator, "n_outputs_"):
        return estimator.n_outputs_

    # LinearRegressor
    if hasattr(estimator, "intercept_"):
        return estimator.intercept_.shape[0]

    msg = (
        "The number of targets could not be inferred from estimator of type "
        f"`{estimator.__class__.name__}`."
    )
    raise TargetInferenceError(msg)

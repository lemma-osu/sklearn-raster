from __future__ import annotations

from typing import TYPE_CHECKING

import dask.array as da
import xarray as xr
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils.validation import check_is_fitted

if TYPE_CHECKING:
    from .dataarray import DataArrayPreprocessor
    from .dataset import DatasetPreprocessor


def _predict_from_dask_backed_array(
    X_image: xr.DataArray | xr.Dataset,
    *,
    estimator: BaseEstimator,
    y,
    preprocessor_cls: type[DataArrayPreprocessor] | type[DatasetPreprocessor],
    nodata_vals=None,
):
    """Generic predict wrapper for Dask-backed arrays."""
    check_is_fitted(estimator)
    preprocessor = preprocessor_cls(X_image, nodata_vals=nodata_vals)

    y_pred = da.apply_gufunc(
        estimator.predict,
        "(x)->(y)",
        preprocessor.flat,
        axis=preprocessor.flat_band_dim,
        output_dtypes=[float],
        output_sizes={"y": y.shape[-1]},
        allow_rechunk=True,
    )

    return preprocessor.unflatten(y_pred, var_names=y.columns)


def _kneighbors_from_dask_backed_array(
    X_image: xr.DataArray,
    *,
    estimator: KNeighborsRegressor,
    preprocessor_cls: type[DataArrayPreprocessor] | type[DatasetPreprocessor],
    nodata_vals=None,
    **kneighbors_kwargs,
) -> xr.DataArray:
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

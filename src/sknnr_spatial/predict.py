from __future__ import annotations

from functools import singledispatch

import dask.array as da
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils.validation import check_is_fitted

from sknnr_spatial.preprocessing import (
    DataArrayPreprocessor,
    DatasetPreprocessor,
    NDArrayPreprocessor,
)


@singledispatch
def predict(
    X_image: NDArray | xr.DataArray | xr.Dataset,
    *,
    estimator: BaseEstimator,
    nodata_vals=None,
) -> None:
    raise NotImplementedError


@singledispatch
def kneighbors(
    X_image: NDArray | xr.DataArray | xr.Dataset,
    *,
    estimator: KNeighborsRegressor,
    nodata_vals=None,
    **kneighbors_kwargs,
) -> None:
    raise NotImplementedError


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


@predict.register(xr.DataArray)
def _predict_from_dataarray(
    X_image: xr.DataArray, *, estimator: BaseEstimator, y, nodata_vals=None
) -> xr.DataArray:
    return _predict_from_dask_backed_array(
        X_image,
        estimator=estimator,
        y=y,
        nodata_vals=nodata_vals,
        preprocessor_cls=DataArrayPreprocessor,
    )


@predict.register(xr.Dataset)
def _predict_from_dataset(
    X_image: xr.Dataset, *, estimator: BaseEstimator, y, nodata_vals=None
) -> xr.Dataset:
    return _predict_from_dask_backed_array(
        X_image,
        estimator=estimator,
        y=y,
        nodata_vals=nodata_vals,
        preprocessor_cls=DatasetPreprocessor,
    )


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


@kneighbors.register(xr.Dataset)
def _kneighbors_from_dataset(
    X_image: xr.Dataset,
    *,
    estimator: KNeighborsRegressor,
    nodata_vals=None,
    **kneighbors_kwargs,
) -> xr.Dataset:
    return _kneighbors_from_dask_backed_array(
        X_image,
        estimator=estimator,
        nodata_vals=nodata_vals,
        preprocessor_cls=DatasetPreprocessor,
        **kneighbors_kwargs,
    )


@kneighbors.register(xr.DataArray)
def _kneighbors_from_dataarray(
    X_image: xr.Dataset,
    *,
    estimator: KNeighborsRegressor,
    nodata_vals=None,
    **kneighbors_kwargs,
) -> xr.Dataset:
    return _kneighbors_from_dask_backed_array(
        X_image,
        estimator=estimator,
        nodata_vals=nodata_vals,
        preprocessor_cls=DataArrayPreprocessor,
        **kneighbors_kwargs,
    )


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

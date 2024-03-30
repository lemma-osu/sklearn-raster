from __future__ import annotations

from functools import singledispatch

import dask.array as da
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils.validation import check_is_fitted

from sknnr_spatial.preprocessing import DataArrayPreprocessor, NDArrayPreprocessor

"""
TODO:
- Figure out if there's a way to get decent docstrings with single dispatch (unless I end up using a class wrapper)
- Probably add an xr.Dataset version that preserves names
"""


@singledispatch
def predict(
    X_image: NDArray | xr.DataArray, *, estimator: BaseEstimator, nodata_vals=None
) -> None:
    raise NotImplementedError


@singledispatch
def kneighbors(
    X_image: NDArray | xr.DataArray,
    *,
    estimator: KNeighborsRegressor,
    return_distance=True,
    return_dataframe_index=False,
    nodata_vals=None,
) -> None:
    raise NotImplementedError


@predict.register(np.ndarray)
def _predict_from_ndarray(
    X_image: NDArray, *, estimator: BaseEstimator, nodata_vals=None
) -> NDArray:
    """Predict attributes from an array of X_image."""
    check_is_fitted(estimator)
    preprocessor = NDArrayPreprocessor(X_image, nodata_vals=nodata_vals)

    # TODO: Deal with sklearn warning about missing feature names if it was fitted with names
    y_pred_flat = estimator.predict(preprocessor.flat)

    return preprocessor.unflatten(y_pred_flat, apply_mask=True)


@kneighbors.register(np.ndarray)
def _kneighbors_from_ndarray(
    X_image: NDArray,
    *,
    estimator: KNeighborsRegressor,
    return_distance=True,
    return_dataframe_index=False,
    nodata_vals=None,
) -> NDArray:
    check_is_fitted(estimator)
    preprocessor = NDArrayPreprocessor(X_image, nodata_vals=nodata_vals)

    result = estimator.kneighbors(
        preprocessor.flat,
        return_distance=return_distance,
        return_dataframe_index=return_dataframe_index,
    )
    if return_distance:
        dist, nn = result

        dist = preprocessor.unflatten(dist)
        nn = preprocessor.unflatten(nn)

        return dist, nn

    else:
        return preprocessor.unflatten(result)


@predict.register(xr.DataArray)
def _predict_from_dataarray(
    X_image: xr.DataArray, *, estimator: BaseEstimator, y, nodata_vals=None
) -> xr.DataArray:
    check_is_fitted(estimator)
    preprocessor = DataArrayPreprocessor(X_image, nodata_vals=nodata_vals)

    y_pred = da.apply_gufunc(
        estimator.predict,
        "(x)->(y)",
        preprocessor.flat,
        axis=preprocessor.flat_band_dim,
        output_dtypes=[float],
        output_sizes={"y": y.shape[-1]},
        allow_rechunk=True,
    )

    return preprocessor.unflatten(y_pred, var_names=y.columns, name="pred")


@kneighbors.register(xr.DataArray)
def _kneighbors_from_dataarray(
    X_image: xr.DataArray,
    *,
    estimator: BaseEstimator,
    nodata_vals=None,
    **kneighbors_kwargs
) -> xr.DataArray:
    check_is_fitted(estimator)
    preprocessor = DataArrayPreprocessor(X_image, nodata_vals=nodata_vals)
    return_distance = kneighbors_kwargs.pop("return_distance", True)

    signature = "(x)->(k)"
    dtypes = [int]

    # Modify the gufunc signature to include the distance output
    if return_distance:
        signature += ",(k)"
        dtypes = [float, *dtypes]

    k = estimator.n_neighbors
    var_names = [f"k{i + 1}" for i in range(k)]

    result = da.apply_gufunc(
        estimator.kneighbors,
        signature,
        preprocessor.flat,
        output_dtypes=dtypes,
        output_sizes={"k": k},
        axis=preprocessor.flat_band_dim,
        allow_rechunk=True,
        **kneighbors_kwargs,
    )

    if return_distance:
        dist, nn = result

        dist = preprocessor.unflatten(dist, var_names=var_names, name="dist")
        nn = preprocessor.unflatten(nn, var_names=var_names, name="nn")

        return dist, nn

    else:
        return preprocessor.unflatten(result, var_names=var_names, name="nn")

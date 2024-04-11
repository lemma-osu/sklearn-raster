from __future__ import annotations

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsRegressor

from ._base import kneighbors, predict
from ._dask_backed import (
    _kneighbors_from_dask_backed_array,
    _predict_from_dask_backed_array,
)
from .dataarray import DataArrayPreprocessor


class DatasetPreprocessor(DataArrayPreprocessor):
    """
    Pre-processor for multi-band xr.Datasets.

    Unlike a DataArray, a Dataset will retrieve variable names and nodata values from
    metadata, if possible.
    """

    def __init__(
        self,
        image: xr.Dataset,
        nodata_vals: float | tuple[float] | NDArray | None = None,
        nan_fill: float | None = 0.0,
    ):
        # The image itself will be stored as a DataArray, but keep the Dataset for
        # metadata like _FillValues.
        self.dataset = image
        super().__init__(image.to_dataarray(), nodata_vals, nan_fill)

    def _validate_nodata_vals(
        self, nodata_vals: float | tuple[float] | NDArray | None
    ) -> NDArray | None:
        """
        Get an array of NoData values in the shape (bands,) based on user input and
        Dataset metadata.
        """
        fill_vals = [
            self.dataset[var].attrs.get("_FillValue") for var in self.dataset.data_vars
        ]

        # Defer to provided nodata vals first. Next, try using per-variable fill values.
        # If at least one variable specifies a nodata value, use them all. Variables
        # that didn't specify a fill value will be assigned None.
        if nodata_vals is None and not all(v is None for v in fill_vals):
            return np.array(fill_vals)

        # Fall back to the DataArray logic for handling nodata
        return super()._validate_nodata_vals(nodata_vals)

    def unflatten(
        self,
        flat_image: xr.DataArray,
        *,
        apply_mask=True,
        var_names=None,
    ) -> xr.Dataset:
        return (
            super()
            .unflatten(
                flat_image,
                apply_mask=apply_mask,
                var_names=var_names,
            )
            .to_dataset(dim="variable")
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

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from ._dask_backed import DaskBackedWrapper
from .dataarray import DataArrayPreprocessor

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..types import NoDataType


class DatasetPreprocessor(DataArrayPreprocessor):
    """
    Pre-processor for multi-band xr.Datasets.

    Unlike a DataArray, a Dataset will retrieve variable names and NoData values from
    metadata, if possible.
    """

    def __init__(
        self,
        image: xr.Dataset,
        nodata_vals: NoDataType = None,
        nan_fill: float | None = 0.0,
    ):
        # The image itself will be stored as a DataArray, but keep the Dataset for
        # metadata like _FillValues.
        self.dataset = image
        super().__init__(image.to_dataarray(), nodata_vals, nan_fill)

    def _validate_nodata_vals(self, nodata_vals: NoDataType) -> NDArray | None:
        """
        Get an array of NoData values in the shape (bands,) based on user input and
        Dataset metadata.
        """
        fill_vals = [
            self.dataset[var].attrs.get("_FillValue") for var in self.dataset.data_vars
        ]

        # Defer to provided NoData vals first. Next, try using per-variable fill values.
        # If at least one variable specifies a NoData value, use them all. Variables
        # that didn't specify a fill value will be assigned None.
        if nodata_vals is None and not all(v is None for v in fill_vals):
            return np.array(fill_vals)

        # Fall back to the DataArray logic for handling NoData
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


class DatasetWrapper(DaskBackedWrapper[xr.Dataset]):
    """A wrapper around a Dataset that provides sklearn methods."""

    preprocessor_cls = DatasetPreprocessor

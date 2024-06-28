from __future__ import annotations

from typing import Callable, Generic

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from typing_extensions import Concatenate

from .types import ImageType, NoDataType, P


class _ImageChunk:
    """A chunk of an NDArray in shape (y, x, band)."""

    def __init__(self, array: NDArray, nodata_vals: list[float] | None = None):
        self.array = array
        self.nodata_vals = nodata_vals

    def _mask_nodata(self, flat_image: NDArray) -> NDArray:
        """
        Set NaNs in the flat image where NoData values are present.
        """
        # Skip allocating a mask if the image is float and NoData wasn't given
        if not (is_float := flat_image.dtype.kind == "f") and self.nodata_vals is None:
            return flat_image

        mask = np.zeros(flat_image.shape, dtype=bool)

        # If it's floating point, always mask NaNs
        if is_float:
            mask |= np.isnan(flat_image)

        # If NoData was specified, mask those values
        if self.nodata_vals is not None:
            mask |= flat_image == self.nodata_vals

        # Set the mask where any band contains NoData
        flat_image[mask.max(axis=-1)] = np.nan

        return flat_image

    def _preprocess(self, nan_fill: float = 0.0) -> NDArray:
        """Preprocess the chunk by flattening to (pixels, bands) and filling NaNs."""
        flat = self.array.reshape(-1, self.array.shape[-1])
        if nan_fill is not None:
            flat[np.isnan(flat)] = nan_fill

        return flat

    def _postprocess(self, array: NDArray, mask_nodata: bool = True) -> NDArray:
        """Postprocess the chunk by unflattening to (y, x, band) and masking NoData."""
        output_shape = [*self.array.shape[:2], -1]
        if mask_nodata:
            array = self._mask_nodata(array)

        return array.reshape(output_shape)

    def apply(
        self, func, returns_tuple=False, nan_fill=0.0, mask_nodata=True, **kwargs
    ) -> NDArray | tuple[NDArray]:
        """
        Apply a function to the flattened, processed chunk.

        The function should accept and return one or more NDArrays in shape
        (pixels, bands). The output will be reshaped back to the original chunk shape.
        """
        flat_chunk = self._preprocess(nan_fill=nan_fill)

        flat_result = func(flat_chunk, **kwargs)

        if returns_tuple:
            return tuple(
                self._postprocess(result, mask_nodata=mask_nodata)
                for result in flat_result
            )

        return self._postprocess(flat_result, mask_nodata=mask_nodata)


class Image(Generic[ImageType]):
    """A wrapper around a multi-band image"""

    def __init__(self, image: ImageType, nodata_vals: NoDataType = None):
        self.image = image

        # TODO: Parse NoData values
        self.nodata_vals = nodata_vals
        # TODO: Parse band dim name
        self.band_dim_name = "variable"

    def apply_ufunc_across_bands(
        self,
        func: Callable[Concatenate[NDArray, P], NDArray],
        *,
        output_dims: list[list[str]] | None = None,
        output_dtypes: list[np.dtype] | None = None,
        output_sizes: dict[str, int] | None = None,
        nan_fill: float = 0.0,
        mask_nodata: bool = True,
        **ufunc_kwargs,
    ) -> ImageType:
        """
        Apply a universal function to all bands of the image.

        If the image is backed by a Dask array, the computation will be parallelized
        across spatial chunks.
        """
        image = self.image

        # TODO: Decide on reasonable defaults
        output_dims = output_dims or [["output"]]
        n_outputs = len(output_dims)
        output_dtypes = output_dtypes or [np.float32] * n_outputs
        output_sizes = output_sizes or {"output": 1}

        def ufunc(x):
            return _ImageChunk(x, nodata_vals=self.nodata_vals).apply(
                func,
                returns_tuple=len(output_dims) > 1,
                nan_fill=nan_fill,
                mask_nodata=mask_nodata,
                **ufunc_kwargs,
            )

        if isinstance(image, np.ndarray):
            return ufunc(image)

        if isinstance(image, xr.Dataset):
            # TODO: Convert back to dataset after predicting
            image = image.to_dataarray()

        # TODO: Assign target dim names
        return xr.apply_ufunc(
            ufunc,
            image,
            dask="parallelized",
            input_core_dims=[[self.band_dim_name]],
            exclude_dims=set((self.band_dim_name,)),
            output_core_dims=output_dims,
            output_dtypes=output_dtypes,
            dask_gufunc_kwargs=dict(
                output_sizes=output_sizes,
                allow_rechunk=True,
            ),
        )

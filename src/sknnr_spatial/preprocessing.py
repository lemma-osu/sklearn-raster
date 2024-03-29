from abc import ABC, abstractmethod
from types import ModuleType
from typing import TypeVar

import dask.array as da
import numpy as np
import xarray as xr
from numpy.typing import NDArray

ImageType = TypeVar("ImageType", NDArray, xr.DataArray)

"""
TODO:

- Maybe add a convenience function for accessing the 2D mask via unflattening
"""


class _ImagePreprocessor(ABC):
    """The module used for array operations on the image type."""
    _backend: ModuleType
    """The dimension used for bands in the image shape."""
    _band_dim: int

    def __init__(
        self,
        image: ImageType,
        nodata_vals: float | tuple[float] | NDArray | None = None,
    ):
        """
        Parameters
        ----------
        image: ImageType
            An image to be processed.
        nodata_vals : float | tuple[float] | NDArray | None, default None
            Values representing NoData in the image. This can be represented by a
            single value or a sequence of values. If a single value is passed, it will
            be applied to all bands. If a sequence is passed, there must be one element
            for each band in the image. For float images, np.nan will always be treated
            as NoData.
        """
        self.image = image
        self.flat = self._flatten()

        self.nodata_vals = self._validate_nodata_vals(nodata_vals)
        self.nodata_mask = self._get_nodata_mask()

        # TODO: Mask NaNs in the flat image
        # self.flat = self._backend.nan_to_num(self.flat)

    @property
    def n_bands(self):
        return self.image.shape[self._band_dim]

    @abstractmethod
    def _flatten(self) -> ImageType:
        """
        Ravel the image's x, y dimensions while keeping the band dimension.
        """

    @abstractmethod
    def unflatten(self, flat_image: ImageType, apply_mask: bool = True) -> ImageType:
        """
        Reconstruct the x, y dimensions of a flattened image to the original shape.
        """

    def _get_nodata_mask(self) -> ImageType | None:
        """
        Get a mask of NoData values in the shape (pixels,) for the flat image.
        """
        # Skip allocating a mask if the image is float and nodata wasn't given
        if not (is_float := self.flat.dtype.kind == "f") and self.nodata_vals is None:
            return None

        mask = self._backend.zeros(self.flat.shape, dtype=bool)

        # If it's floating point, always mask NaNs
        if is_float:
            mask |= self._backend.isnan(self.flat)

        # If nodata was specified, mask those values
        if self.nodata_vals is not None:
            mask |= self.flat == self.nodata_vals

        # Set the mask where any band contains nodata
        return mask.max(axis=-1)

    def _validate_nodata_vals(
        self, nodata_vals: float | tuple[float] | NDArray | None
    ) -> NDArray | None:
        """
        Get an array of nodata values in the shape (bands,) based on user input.

        Scalars are broadcast to all bands while sequences are checked against the 
        number of bands and cast to ndarrays.
        """
        if nodata_vals is None:
            return None
        
        if isinstance(nodata_vals, (float, int)) and not isinstance(nodata_vals, bool):
            return np.full((self.n_bands,), nodata_vals)
        
        if not hasattr(nodata_vals, "__len__") or isinstance(nodata_vals, (str, dict)):
            raise TypeError(
                f"Invalid type `{type(nodata_vals).__name__}` for `nodata_vals`. "
                "Provide a single number to apply to all bands, a sequence of numbers, "
                "or None."
            )
        
        if len(nodata_vals) != self.n_bands:
            raise ValueError(
                f"Expected {self.n_bands} nodata values but got {len(nodata_vals)}. "
                f"The length of `nodata_vals` must match the number of bands."
            )

        return np.asarray(nodata_vals, dtype=float)

    def _fill_nodata(self, flat_image: ImageType, nodata_fill_value=0.0) -> ImageType:
        """Fill values in a flat image based on the pre-calculated mask."""
        if self.nodata_mask is None:
            return flat_image

        if isinstance(nodata_fill_value, float):
            flat_image = flat_image.astype(float)

        flat_image[self.nodata_mask, :] = nodata_fill_value

        return flat_image


class NDArrayPreprocessor(_ImagePreprocessor):
    """
    Pre-processor multi-band NumPy NDArrays.
    """

    _backend = np
    _band_dim = -1

    def _flatten(self) -> NDArray:
        """Flatten the array from (y, x, bands) to (pixels, bands)."""
        return self.image.reshape(-1, self.n_bands)

    def unflatten(self, flat_image: NDArray, apply_mask=True) -> NDArray:
        unflattened = flat_image.reshape(*self.image.shape[:2], -1)

        if apply_mask:
            unflattened = self._fill_nodata(unflattened, np.nan)

        return unflattened


class DataArrayPreprocessor(_ImagePreprocessor):
    """
    Pre-processor for multi-band xarray DataArrays.
    """
    _backend = da
    _band_dim = 0

    def _flatten(self) -> xr.DataArray:
        """Flatten the dataarray from (bands, y, x) to (pixels, bands)."""
        # Dask can only reshape one dimension at a time, so the transpose is necessary 
        # to get the correct dimension order
        return self.image.data.reshape(self.n_bands, -1).T

    def unflatten(
        self, 
        flat_image: xr.DataArray, 
        apply_mask=True, 
        var_names=None, 
        name=None,
    ) -> xr.DataArray:
        if apply_mask:
            flat_image = self._fill_nodata(flat_image, np.nan)

        n_outputs = flat_image.shape[-1]
        var_names = var_names if var_names is not None else range(n_outputs)

        # Replace the original variable coordinates and dimensions
        dims = {**self.image.sizes, "variable": n_outputs}
        coords = {**self.image.coords, "variable": var_names}
        shape = list(dims.values())

        unflattened = xr.DataArray(
            flat_image.T.reshape(shape),
            coords=coords,
            dims=dims,
            name=name,
        )

        return unflattened

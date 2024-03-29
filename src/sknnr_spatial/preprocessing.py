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

- Unfortunately, probably go back to masking the flat array so that we can a) coerce to the same dimension order, b) avoid annoying broadcasting, c) avoid the nan_to_num that seems to be running eagerly for some reason, d) avoid 2d indexing issue when masking that forces us to use da.where that converts to Array
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
        self.nodata_vals = self._validate_nodata_vals(nodata_vals)
        self.nodata_mask = self._get_nodata_mask()
        self.flat = self._flatten()
        self.flat = self._backend.nan_to_num(self.flat)

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
        Get a mask of NoData values in the shape (y, x).
        """
        # If the image isn't float and no nodata was given, skip allocating a mask
        if not (is_float := self.image.dtype.kind == "f") and self.nodata_vals is None:
            return None

        mask = self._backend.zeros(self.image.shape, dtype=bool)

        # If it's floating point, always mask NaNs
        if is_float:
            mask |= self._backend.isnan(self.image)

        # If nodata was specified, mask those values
        if self.nodata_vals is not None:
            mask |= self.image == self.nodata_vals

        # Set the mask where any band contains nodata
        return mask.max(axis=self._band_dim)

    def _validate_nodata_vals(
        self, nodata_vals: float | tuple[float] | NDArray | None
    ) -> NDArray | None:
        """
        Check and process user-provided nodata values to the expected format.
        """
        if nodata_vals is None:
            return None
        if isinstance(nodata_vals, (float, int)) and not isinstance(nodata_vals, bool):
            # Broadcast single values to all bands
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

        nodata_array = np.asarray(nodata_vals, dtype=float)

        # Dask arrays are pickier about broadcasting than numpy arrays, so we need to
        # create the y and x axes.
        # TODO: Figure out if this is totally necessary, or if there's a cleaner way to do it. When I was masking the flat array, it didn't seem necessary
        broadcast_shape = [1, 1, 1]
        broadcast_shape[self._band_dim] = -1
        return nodata_array.reshape(broadcast_shape)

    def _fill_nodata(self, image: ImageType, nodata_fill_value=0.0) -> ImageType:
        """Fill masked values in a flat image with a given value."""
        if self.nodata_mask is None:
            return image

        # TODO: Only cast to float if required by the fill value
        image = image.astype(float)
        # TODO: Dask doesn't support 2d indexing, so it looks like I need to use where
        # image[:, self.nodata_mask] = nodata_fill_value
        # TODO: This converts xr.DataArray to da.Array. Is that a problem?
        image = self._backend.where(self.nodata_mask, nodata_fill_value, self.image)

        return image


class NDArrayPreprocessor(_ImagePreprocessor):
    """
    Pre-process a multi-band NDArray for prediction with an sklearn estimator.
    """

    _backend = np
    _band_dim = -1

    def _flatten(self) -> NDArray:
        """Flatten the array from (x, y, bands) to (pixels, bands) and fill NaNs."""
        return self.image.reshape(-1, self.n_bands)

    def unflatten(self, flat_image: NDArray, apply_mask=True) -> NDArray:
        unflattened = flat_image.reshape(*self.image.shape[:2], -1)

        if apply_mask:
            unflattened = self._fill_nodata(unflattened, np.nan)

        return unflattened


class DataArrayPreprocessor(_ImagePreprocessor):
    _backend = da
    _band_dim = 0

    def _flatten(self) -> xr.DataArray:
        return self.image.data.reshape(self.n_bands, -1)

    # TODO: If possible, get it so that you can flatten and unflatten an image without modifying the chunks
    def unflatten(
        self, flat_image: xr.DataArray, n_outputs, apply_mask=True
    ) -> xr.DataArray:
        dims = dict(self.image.sizes)
        dims.update({"variable": n_outputs})

        shape = list(dims.values())

        coords = dict(self.image.coords)
        coords.update({"variable": range(n_outputs)})

        unflattened = xr.DataArray(
            flat_image.reshape(shape),
            coords=coords,
            dims=dims,
        )

        if apply_mask:
            unflattened = self._fill_nodata(unflattened, np.nan)

        return unflattened

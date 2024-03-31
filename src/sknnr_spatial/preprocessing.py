from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sized
from types import ModuleType
from typing import TypeVar

import dask.array as da
import numpy as np
import xarray as xr
from numpy.typing import NDArray

ImageType = TypeVar("ImageType", NDArray, xr.DataArray)


class _ImagePreprocessor(ABC):
    """
    A pre-processor for multi-channel image data in a machine learning workflow.

    This class handles flattening an image from 3D pixel space (e.g. y, x, channel)
    to 2D sample space (sample, channel) to allow prediction and other operations
    with scikit-learn estimators designed for sample data, and unflattening to
    convert scikit-learn outputs from sample space back to pixel space.

    Pre-processing can also fill NaN values in sample space to allow prediction with
    estimators that prohibit NaNs, and mask NoData values in sample space outputs
    during unflattening to preserve NoData masks from the input image.

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
    nan_fill : float | None, default 0.0
        If not none, NaN values in the flattened image will be filled with this
        value. This is important when passing flattened arrays into scikit-learn
        estimators that prohibit NaN values.

    Attributes
    ----------
    image : ImageType
        The original, unmodified image used to construct the preprocessor.
    flat : ImageType
        A flat representation of the image, with the x and y dimensions flattened to a
        sample dimension.
    n_bands : int
        The number of bands present in the image.
    band_dim : int
        The dimension used for bands or features in the unmodified image.
    flat_band_dim : int
        The dimension used for bands or features in the flattened image.
    """

    # The module used for array operations on the image type.
    _backend: ModuleType

    band_dim: int
    flat_band_dim = -1

    def __init__(
        self,
        image: ImageType,
        nodata_vals: float | tuple[float] | NDArray | None = None,
        nan_fill: float | None = 0.0,
    ):
        self.image = image
        self.flat = self._flatten(self.image)

        self.nodata_vals = self._validate_nodata_vals(nodata_vals)
        self.nodata_mask = self._get_nodata_mask(self.flat)

        if nan_fill is not None:
            self.flat[self._backend.isnan(self.flat)] = nan_fill

    @property
    def n_bands(self) -> int:
        """
        Return the number of bands in the image.
        """
        return self.image.shape[self.band_dim]

    @abstractmethod
    def _flatten(self, image: ImageType) -> ImageType:
        """
        Ravel the image's x, y dimensions while keeping the band dimension.
        """

    @abstractmethod
    def unflatten(self, flat_image: ImageType, apply_mask: bool = True) -> ImageType:
        """
        Reconstruct the x, y dimensions of a flattened image to the original shape.
        """

    def _get_nodata_mask(self, flat_image: ImageType) -> ImageType | None:
        """
        Get a mask of NoData values in the shape (pixels,) for the flat image.

        NoData values are represented by True in the output array.
        """
        # Skip allocating a mask if the image is float and nodata wasn't given
        if not (is_float := flat_image.dtype.kind == "f") and self.nodata_vals is None:
            return None

        mask = self._backend.zeros(flat_image.shape, dtype=bool)

        # If it's floating point, always mask NaNs
        if is_float:
            mask |= self._backend.isnan(flat_image)

        # If nodata was specified, mask those values
        if self.nodata_vals is not None:
            mask |= flat_image == self.nodata_vals

        # Set the mask where any band contains nodata
        return mask.max(axis=self.flat_band_dim)

    def _validate_nodata_vals(
        self, nodata_vals: float | tuple[float] | NDArray | None
    ) -> NDArray | None:
        """
        Get an array of NoData values in the shape (bands,) based on user input.

        Scalars are broadcast to all bands while sequences are checked against the
        number of bands and cast to ndarrays. There is no need to specify np.nan as a
        NoData value because it will be masked automatically for floating point images.
        """
        if nodata_vals is None:
            return None

        # If it's a numeric scalar, broadcast it to all bands
        if isinstance(nodata_vals, (float, int)) and not isinstance(nodata_vals, bool):
            return np.full((self.n_bands,), nodata_vals)

        # If it's not a scalar, it must be an interable
        if not isinstance(nodata_vals, Sized) or isinstance(nodata_vals, (str, dict)):
            raise TypeError(
                f"Invalid type `{type(nodata_vals).__name__}` for `nodata_vals`. "
                "Provide a single number to apply to all bands, a sequence of numbers, "
                "or None."
            )

        # If it's an iterable, it must contain one element per band
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
    band_dim = -1

    def _flatten(self, image: NDArray) -> NDArray:
        """Flatten the array from (y, x, bands) to (pixels, bands)."""
        return image.reshape(-1, self.n_bands)

    def unflatten(self, flat_image: NDArray, apply_mask=True) -> NDArray:
        if apply_mask:
            flat_image = self._fill_nodata(flat_image, np.nan)

        return flat_image.reshape(*self.image.shape[:2], -1)


class DataArrayPreprocessor(_ImagePreprocessor):
    __doc__ = _ImagePreprocessor.__doc__
    _backend = da
    band_dim = 0

    def _flatten(self, image: xr.DataArray) -> xr.DataArray:
        """Flatten the dataarray from (bands, y, x) to (pixels, bands)."""
        # Dask can't reshape multiple dimensions at once, so transpose to swap axes
        return image.data.reshape(self.n_bands, -1).T

    def unflatten(
        self,
        flat_image: xr.DataArray,
        apply_mask=True,
        var_names=None,
        name=None,
    ) -> xr.DataArray:
        if apply_mask:
            flat_image = self._fill_nodata(flat_image, np.nan)

        n_outputs = flat_image.shape[self.flat_band_dim]
        var_names = var_names if var_names is not None else range(n_outputs)

        # Replace the original variable coordinates and dimensions
        dims = {**self.image.sizes, "variable": n_outputs}
        coords = {**self.image.coords, "variable": var_names}
        shape = list(dims.values())

        return xr.DataArray(
            # Transpose the flat image from (pixels, bands) to (bands, pixels) prior
            # to reshaping to match the expected output.
            flat_image.T.reshape(shape),
            coords=coords,
            dims=dims,
            name=name,
        )

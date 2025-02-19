from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sized
from typing import Any, Callable, Generic

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from typing_extensions import Concatenate

from .types import ImageType, NoDataType, P
from .ufunc import UfuncArrayProcessor


class Image(Generic[ImageType], ABC):
    """A wrapper around a multi-band image"""

    band_dim_name: str | None = None
    band_dim: int = 0
    band_names: NDArray

    def __init__(self, image: ImageType, nodata_input: NoDataType = None):
        self.image = image
        self.n_bands = self.image.shape[self.band_dim]
        self.nodata_input = self._validate_nodata_input(nodata_input)

    def _validate_nodata_input(self, nodata_input: NoDataType) -> NDArray | None:
        """
        Get an array of NoData values in the shape (bands,) based on user input.

        Scalars are broadcast to all bands while sequences are checked against the
        number of bands and cast to ndarrays. There is no need to specify np.nan as a
        NoData value because it will be masked automatically for floating point images.
        """
        if nodata_input is None:
            return None

        # If it's a numeric scalar, broadcast it to all bands
        if isinstance(nodata_input, (float, int)) and not isinstance(
            nodata_input, bool
        ):
            return np.full((self.n_bands,), nodata_input)

        # If it's not a scalar, it must be an iterable
        if not isinstance(nodata_input, Sized) or isinstance(nodata_input, (str, dict)):
            raise TypeError(
                f"Invalid type `{type(nodata_input).__name__}` for `nodata_input`. "
                "Provide a single number to apply to all bands, a sequence of numbers, "
                "or None."
            )

        # If it's an iterable, it must contain one element per band
        if len(nodata_input) != self.n_bands:
            raise ValueError(
                f"Expected {self.n_bands} NoData values but got {len(nodata_input)}. "
                f"The length of `nodata_input` must match the number of bands."
            )

        return np.asarray(nodata_input)

    def apply_ufunc_across_bands(
        self,
        func: Callable[Concatenate[NDArray, P], NDArray],
        *,
        output_dims: list[list[str]],
        output_dtypes: list[np.dtype] | None = None,
        output_sizes: dict[str, int] | None = None,
        output_coords: dict[str, list[str | int]] | None = None,
        skip_nodata: bool = True,
        nodata_output: float | int = np.nan,
        nan_fill: float = 0.0,
        ensure_min_samples: int = 1,
        **ufunc_kwargs,
    ) -> ImageType | tuple[ImageType]:
        """Apply a universal function to all bands of the image."""
        n_outputs = len(output_dims)

        if output_sizes is not None:
            # Default to sequential coordinates for each output dimension
            output_coords = output_coords or {
                k: list(range(s)) for k, s in output_sizes.items()
            }

        def ufunc(x):
            x = self._preprocess_ufunc_input(x)
            return UfuncArrayProcessor(x, nodata_input=self.nodata_input).apply(
                func,
                skip_nodata=skip_nodata,
                nodata_output=nodata_output,
                nan_fill=nan_fill,
                ensure_min_samples=ensure_min_samples,
                **ufunc_kwargs,
            )

        result = xr.apply_ufunc(
            ufunc,
            self.image,
            dask="parallelized",
            input_core_dims=[[self.band_dim_name]],
            exclude_dims=set((self.band_dim_name,)),
            output_core_dims=output_dims,
            output_dtypes=output_dtypes,
            keep_attrs=True,
            dask_gufunc_kwargs=dict(
                output_sizes=output_sizes,
                allow_rechunk=True,
            ),
        )

        if n_outputs > 1:
            result = tuple(
                self._postprocess_ufunc_output(
                    x, output_coords=output_coords, nodata_output=nodata_output
                )
                for x in result
            )
        else:
            result = self._postprocess_ufunc_output(
                result, output_coords=output_coords, nodata_output=nodata_output
            )

        return result

    def _preprocess_ufunc_input(self, image: ImageType) -> ImageType:
        """
        Preprocess the input of an applied ufunc. No-op unless overridden by subclasses.
        """
        return image

    @abstractmethod
    def _postprocess_ufunc_output(
        self,
        result: ImageType,
        nodata_output: float | int,
        output_coords: dict[str, list[str | int]] | None = None,
    ) -> ImageType:
        """
        Postprocess the output of an applied ufunc.

        This method should be overridden by subclasses to handle any necessary
        transformations to the output data, e.g. transposing dimensions.
        """

    @staticmethod
    def from_image(image: Any, nodata_input: NoDataType = None) -> Image:
        """Create an Image object from a supported image type."""
        if isinstance(image, np.ndarray):
            return NDArrayImage(image, nodata_input=nodata_input)

        if isinstance(image, xr.DataArray):
            return DataArrayImage(image, nodata_input=nodata_input)

        if isinstance(image, xr.Dataset):
            return DatasetImage(image, nodata_input=nodata_input)

        raise TypeError(f"Unsupported image type `{type(image).__name__}`.")


class NDArrayImage(Image):
    """An image stored in a Numpy NDArray of shape (band, y, x)."""

    band_names = np.array([])

    def __init__(self, image: NDArray, nodata_input: NoDataType = None):
        super().__init__(image, nodata_input=nodata_input)

    def _preprocess_ufunc_input(self, image: NDArray) -> NDArray:
        """Preprocess the image by transposing to (y, x, band) for apply_ufunc."""
        # Copy to avoid mutating the original image
        return image.copy().transpose(1, 2, 0)

    def _postprocess_ufunc_output(
        self, result: NDArray, nodata_output: float | int, output_coords=None
    ) -> NDArray:
        """Postprocess the ufunc output by transposing back to (band, y, x)."""
        return result.transpose(2, 0, 1)


class DataArrayImage(Image):
    """An image stored in an xarray DataArray of shape (band, y, x)."""

    def __init__(self, image: xr.DataArray, nodata_input: NoDataType = None):
        super().__init__(image, nodata_input=nodata_input)
        self.band_dim_name = image.dims[self.band_dim]

    @property
    def band_names(self) -> NDArray:
        return self.image[self.band_dim_name].values

    def _validate_nodata_input(self, nodata_input: NoDataType) -> NDArray | None:
        """
        Get an array of NoData values in the shape (bands,) based on user input and
        DataArray metadata.
        """
        # Defer to user-provided NoData values over stored attributes
        if nodata_input is not None:
            return super()._validate_nodata_input(nodata_input)

        # If present, broadcast the _FillValue attribute to all bands
        fill_val = self.image.attrs.get("_FillValue")
        if fill_val is not None:
            return np.full((self.n_bands,), fill_val)

        return None

    def _postprocess_ufunc_output(
        self,
        result: xr.DataArray,
        nodata_output: float | int,
        output_coords: dict[str, list[str | int]] | None = None,
    ) -> xr.DataArray:
        """Process the ufunc output by assigning coordinates and transposing."""
        if output_coords is not None:
            result = result.assign_coords(output_coords)

        # Transpose from (y, x, band) to (band, y, x)
        result = result.transpose(result.dims[-1], ...)

        if not np.isnan(nodata_output):
            result.attrs["_FillValue"] = nodata_output
        else:
            # Remove the _FillValue copied from the input array
            result.attrs.pop("_FillValue", None)

        return result


class DatasetImage(DataArrayImage):
    """An image stored in an xarray Dataset of shape (y, x) with bands as variables."""

    def __init__(self, image: xr.Dataset, nodata_input: NoDataType = None):
        # The image itself will be stored as a DataArray, but keep the Dataset for
        # metadata like _FillValues.
        self.dataset = image
        super().__init__(image.to_dataarray(), nodata_input=nodata_input)

    @property
    def band_names(self) -> NDArray:
        return np.array(list(self.dataset.data_vars))

    def _validate_nodata_input(self, nodata_input: NoDataType) -> NDArray | None:
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
        if nodata_input is None and not all(v is None for v in fill_vals):
            return np.array(fill_vals)

        # Fall back to the DataArray logic for handling NoData
        return super()._validate_nodata_input(nodata_input)

    def _postprocess_ufunc_output(
        self,
        result: xr.DataArray,
        nodata_output: float | int,
        output_coords: dict[str, list[str | int]] | None = None,
    ) -> xr.Dataset:
        """Process the ufunc output converting from DataArray to Dataset."""
        result = super()._postprocess_ufunc_output(
            result, output_coords=output_coords, nodata_output=nodata_output
        )
        var_dim = result.dims[self.band_dim]
        ds = result.to_dataset(dim=var_dim)

        for var in ds.data_vars:
            if not np.isnan(nodata_output):
                ds[var].attrs["_FillValue"] = nodata_output

        return ds

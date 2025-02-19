from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sized
from typing import Any, Callable, Generic, cast

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from typing_extensions import Concatenate

from .types import ImageType, NoDataType, P


class _ImageChunk:
    """
    A chunk of an NDArray in shape (y, x, band).

    Note that this dimension order is different from the (band, y, x) order used by
    rasterio, rioxarray, and elsewhere in sklearn-raster. This is because `_ImageChunk`
    is called via `xr.apply_ufunc` which automatically swaps the core dimension to the
    last axis, resulting in arrays of (y, x, band).
    """

    band_dim = -1

    def __init__(self, array: NDArray, nodata_input: list[float] | None = None):
        self.array = array
        self.nodata_input = nodata_input
        self.flat_array = array.reshape(-1, array.shape[self.band_dim])

        # We can take some shortcuts if the input array type can't contain NaNs
        self._input_supports_nan = np.issubdtype(array.dtype, np.floating)
        self.nodata_mask = self._get_flat_nodata_mask()

        num_pixels = self.flat_array.shape[0]
        self._num_masked = 0 if self.nodata_mask is None else self.nodata_mask.sum()
        self._num_unmasked = num_pixels - self._num_masked

    def _get_flat_nodata_mask(self) -> NDArray | None:
        # Skip allocating a mask if the image is not float and NoData wasn't given
        if not self._input_supports_nan and self.nodata_input is None:
            return None

        mask = np.zeros(self.flat_array.shape, dtype=bool)

        # If it's floating point, always mask NaNs
        if self._input_supports_nan:
            mask |= np.isnan(self.flat_array)

        # If NoData was specified, mask those values
        if self.nodata_input is not None:
            mask |= self.flat_array == self.nodata_input

        # Return a mask where any band contains NoData
        return mask.max(axis=self.band_dim)

    def _validate_nodata_output(
        self, output: NDArray, nodata_output: float | int
    ) -> None:
        """Check that a given output array can support the NoData value."""
        if not np.can_cast(type(nodata_output), output.dtype):
            msg = (
                f"The selected `nodata_output` value {nodata_output} does not fit in "
                f"the array dtype {output.dtype}."
            )
            raise ValueError(msg)

    def _mask_nodata(self, flat_image: NDArray, nodata_output: float | int) -> NDArray:
        """
        Replace NoData values in the input array with `output_nodata`.
        """
        self._validate_nodata_output(flat_image, nodata_output)
        flat_image[self.nodata_mask] = nodata_output
        return flat_image

    # TODO: Try to refactor out the need for the mask_nodata parameter.
    def _postprocess(
        self,
        result: NDArray | tuple[NDArray, ...],
        nodata_output: float | int,
        mask_nodata: bool = True,
    ) -> NDArray | tuple[NDArray, ...]:
        """Postprocess results by unflattening to (y, x, band) and masking NoData."""
        if isinstance(result, tuple):
            return tuple(
                self._postprocess(
                    array, mask_nodata=mask_nodata, nodata_output=nodata_output
                )
                for array in result
            )

        output_shape = [*self.array.shape[:2], -1]
        if mask_nodata:
            result = self._mask_nodata(result, nodata_output=nodata_output)

        return result.reshape(output_shape)

    def apply(
        self,
        func,
        *,
        skip_nodata: bool = True,
        nodata_output: float | int = np.nan,
        nan_fill: float | int | None = None,
        ensure_min_samples: int = 1,
        **kwargs,
    ) -> NDArray | tuple[NDArray]:
        """
        Apply a function to the flattened chunk.

        The function should accept and return one or more NDArrays in shape
        (pixels, bands). The output will be reshaped back to the original chunk shape.

        Parameters
        ----------
        func : callable
            A function to apply to the flattened array. The function should accept one
            array of shape (pixels, bands) and return one or more arrays of the same
            shape.
        skip_nodata : bool, default True
            If True, NoData and NaN values will be removed before passing the array to
            `func`. This can speed up processing of partially masked arrays, but may be
            incompatible with functions that expect a consistent number of samples.
        nodata_output : float or int, default np.nan
            NoData pixels in the input will be replaced with this value in the output.
            If the value does not fit the array dtype returned by `func`, an error will
            be raised.
        nan_fill : float or int, optional
            If `skip_nodata=False`, any NaNs in the input array will be filled with this
            value prior to calling `func` to avoid errors from functions that do not
            support NaN inputs. If None, NaNs will not be filled.
        ensure_min_samples : int, default 1
            The minimum number of samples passed to `func` even if the array is fully
            masked and `skip_nodata=True`. This is necessary for functions that require
            non-empty array inputs, like some scikit-learn `predict` methods. No effect
            if the array contains enough valid pixels or if `skip_nodata=False`.
        **kwargs : dict
            Additional keyword arguments passed to `func`.
        """
        # No need to fill NaNs if they're skipped anyways or unsupported
        if skip_nodata is True or self._input_supports_nan is False or nan_fill is None:
            flat_array = self.flat_array
        else:
            flat_array = np.where(np.isnan(self.flat_array), nan_fill, self.flat_array)

        # Only skip NoData if there's something to skip
        if skip_nodata and self._num_masked > 0:
            flat_result = self._masked_apply(
                func,
                flat_array=flat_array,
                ensure_min_samples=ensure_min_samples,
                nodata_output=nodata_output,
                **kwargs,
            )
            # NoData is now pre-masked
            mask_nodata = False
        else:
            flat_result = func(flat_array, **kwargs)
            mask_nodata = self._num_masked > 0

        return self._postprocess(
            flat_result, mask_nodata=mask_nodata, nodata_output=nodata_output
        )

    def _masked_apply(
        self,
        func,
        *,
        flat_array: NDArray,
        nodata_output: float | int,
        ensure_min_samples: int,
        **kwargs,
    ) -> NDArray | tuple[NDArray, ...]:
        """
        Apply a function to all non-NoData values in a flat array.
        """
        if inserted_dummy_values := self._num_unmasked < ensure_min_samples:
            if ensure_min_samples > flat_array.shape[0]:
                raise ValueError(
                    f"Cannot ensure {ensure_min_samples} samples with only "
                    f"{flat_array.shape[0]} total pixels in the image chunk."
                )

            cast(NDArray, self.nodata_mask)[:ensure_min_samples] = False
            flat_array[:ensure_min_samples] = 0

        def insert_result(result: NDArray):
            """Insert the array result for valid pixels into the full-shaped array."""
            self._validate_nodata_output(result, nodata_output)

            # Build an output array pre-masked with the fill value and cast to the
            # output dtype. The shape will be (n, b) where n is the number of pixels
            # in the flat array and b is the number of bands in the func result.
            full_result = np.full(
                (flat_array.shape[0], result.shape[-1]),
                nodata_output,
                dtype=result.dtype,
            )
            full_result[~cast(NDArray, self.nodata_mask)] = result

            # Re-mask any pixels that were filled to ensure minimum samples
            if inserted_dummy_values:
                cast(NDArray, self.nodata_mask)[:ensure_min_samples] = True
                full_result[:ensure_min_samples] = nodata_output

            return full_result

        func_result = func(flat_array[~cast(NDArray, self.nodata_mask)], **kwargs)
        if isinstance(func_result, tuple):
            return tuple(insert_result(result) for result in func_result)

        return insert_result(func_result)


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
            return _ImageChunk(x, nodata_input=self.nodata_input).apply(
                func,
                skip_nodata=skip_nodata,
                nodata_output=nodata_output,
                nan_fill=nan_fill,
                ensure_min_samples=ensure_min_samples,
                **ufunc_kwargs,
            )

        result = xr.apply_ufunc(
            ufunc,
            self._preprocess_ufunc_input(self.image),
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
            else:
                # Remove the _FillValue copied from the input array
                ds[var].attrs.pop("_FillValue", None)

        return ds

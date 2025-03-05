from __future__ import annotations

from typing import cast
from warnings import warn

import numpy as np
from numpy.typing import NDArray

from .types import ArrayUfunc, MaybeTuple
from .utils.wrapper import map_function_over_tuples


class UfuncArrayProcessor:
    """
    A processor for applying ufuncs to arrays.

    The processor takes Numpy arrays (images or image chunks) in the shape (y, x, band)
    and:

    1. Flattens 2D spatial dimensions to a 1D sample dimension.
    2. Fills NaN pixels in the input array.
    3. Passes samples to the ufunc.
    4. Masks NoData values in the ufunc output.

    Note that this dimension order is different from the (band, y, x) order used by
    rasterio, rioxarray, and elsewhere in sklearn-raster. This is because
    `UfuncArrayProcessor` is called via `xr.apply_ufunc` which automatically swaps the
    core dimension to the last axis, resulting in arrays of (y, x, band).
    """

    band_dim = -1

    def __init__(self, array: NDArray, *, nodata_input: list[float] | None = None):
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

    def _fill_flat_nans(self, nan_fill: float | None) -> NDArray:
        """Fill the flat input array with NaNs filled."""
        if nan_fill is not None and self._input_supports_nan:
            return np.where(np.isnan(self.flat_array), nan_fill, self.flat_array)

        return self.flat_array

    def apply(
        self,
        func: ArrayUfunc,
        *,
        skip_nodata: bool = True,
        nodata_output: float | int = np.nan,
        nan_fill: float | int | None = None,
        ensure_min_samples: int = 1,
        allow_cast: bool = False,
        check_output_for_nodata: bool = True,
        **kwargs,
    ) -> MaybeTuple[NDArray]:
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
            The minimum number of samples that should be passed to `func`. If the array
            is fully masked and `skip_nodata=True`, dummy values (`nan_fill` or 0) will
            be inserted to ensure this number of samples. This is necessary for
            functions that require non-empty array inputs, like some scikit-learn
            `predict` methods. No effect if the array contains enough valid pixels or if
            `skip_nodata=False`.
        allow_cast : bool, default False
            If True and the `func` output dtype is incompatible with the chosen
            `nodata_output` value, the output will be cast to the correct dtype.
            Otherwise, an error will be raised unless `allow_cast` is True.
        check_output_for_nodata : bool, default True
            If True and `nodata_output` is not NaN, a warning will be raised if the
            selected `nodata_output` value is returned by `func`, as this may indicate
            a valid pixel being masked.
        **kwargs : dict
            Additional keyword arguments passed to `func`.
        """
        # Fill NaNs in the input array if they're not being skipped
        flat_array = self.flat_array if skip_nodata else self._fill_flat_nans(nan_fill)
        # Only skip NoData if there's something to skip
        if skip_nodata and self._num_masked > 0:
            flat_result = self._skip_nodata_apply(
                func,
                flat_array=flat_array,
                ensure_min_samples=ensure_min_samples,
                nodata_output=nodata_output,
                allow_cast=allow_cast,
                nan_fill=nan_fill,
                check_output_for_nodata=check_output_for_nodata,
                **kwargs,
            )
            # NoData is now pre-masked
            mask_nodata = False
        else:
            flat_result = func(flat_array, **kwargs)
            mask_nodata = self._num_masked > 0

        @map_function_over_tuples
        def _unflatten_and_mask(result: NDArray) -> NDArray:
            """Unflattening result to (y, x, band) and mask NoData."""
            output_shape = [*self.array.shape[:2], -1]
            if mask_nodata:
                result = self._mask_nodata(
                    result,
                    nodata_output=nodata_output,
                    allow_cast=allow_cast,
                    check_output_for_nodata=check_output_for_nodata,
                )

            return result.reshape(output_shape)

        return _unflatten_and_mask(flat_result)

    def _skip_nodata_apply(
        self,
        func: ArrayUfunc,
        *,
        flat_array: NDArray,
        nodata_output: float | int,
        ensure_min_samples: int,
        allow_cast: bool,
        nan_fill: float | int | None,
        check_output_for_nodata: bool,
        **kwargs,
    ) -> NDArray | tuple[NDArray, ...]:
        """Apply a function to all non-NoData values in a flat array."""
        # The NoData mask is guaranteed to exist since this method is only called when
        # there are masked pixels, so we can safely cast it for type checking.
        nodata_mask = cast(NDArray, self.nodata_mask)

        if inserted_dummy_values := self._num_unmasked < ensure_min_samples:
            if ensure_min_samples > flat_array.shape[0]:
                raise ValueError(
                    f"Cannot ensure {ensure_min_samples} samples with only "
                    f"{flat_array.shape[0]} total pixels in the image chunk."
                )

            # Fill NoData pixels with dummy values to ensure minimum samples. Copy the
            # mask to avoid mutating it when it's temporarily disabled.
            dummy_mask = nodata_mask[:ensure_min_samples].copy()
            flat_array[:ensure_min_samples][dummy_mask] = (
                nan_fill if nan_fill is not None else 0
            )

            # Temporarily disable the mask so that dummy samples aren't skipped
            nodata_mask[:ensure_min_samples] = False

        @map_function_over_tuples
        def populate_missing_pixels(result: NDArray) -> NDArray:
            """Insert the array result for valid pixels into the full-shaped array."""
            result = self._validate_nodata_output(
                result,
                nodata_output,
                allow_cast=allow_cast,
                check_output_for_nodata=check_output_for_nodata,
            )

            # Build an output array pre-masked with the fill value and cast to the
            # output dtype. The shape will be (n, b) where n is the number of pixels
            # in the flat array and b is the number of bands in the func result.
            full_result = np.full(
                (flat_array.shape[0], result.shape[-1]),
                nodata_output,
                dtype=result.dtype,
            )
            full_result[~nodata_mask] = result

            # Re-mask any pixels that were filled to ensure minimum samples
            if inserted_dummy_values:
                full_result[:ensure_min_samples][dummy_mask] = nodata_output
                nodata_mask[:ensure_min_samples][dummy_mask] = True

            return full_result

        # Apply the func only to valid pixels
        func_result = func(flat_array[~nodata_mask], **kwargs)
        return populate_missing_pixels(func_result)

    def _mask_nodata(
        self,
        flat_image: NDArray,
        nodata_output: float | int,
        allow_cast: bool,
        check_output_for_nodata: bool,
    ) -> NDArray:
        """Replace NoData values in the input array with `output_nodata`."""
        flat_image = self._validate_nodata_output(
            flat_image,
            nodata_output,
            allow_cast=allow_cast,
            check_output_for_nodata=check_output_for_nodata,
        )

        flat_image[self.nodata_mask] = nodata_output
        return flat_image

    def _validate_nodata_output(
        self,
        output: NDArray,
        nodata_output: float | int,
        allow_cast: bool,
        check_output_for_nodata: bool,
    ) -> NDArray:
        """
        Check that a given output array can support the NoData value.

        Cast (if allowed) or raise if not. Also optionally check for NoData values in
        the output that may indicate valid pixels being masked.
        """
        # Use the minimum dtype for integers. Otherwise, just use the value's type to
        # avoid casting to low-precision float16.
        nodata_output_type = (
            np.min_scalar_type(nodata_output)
            if np.issubdtype(type(nodata_output), np.integer)
            else type(nodata_output)
        )

        if not np.can_cast(nodata_output_type, output.dtype):
            if allow_cast:
                output = output.astype(nodata_output_type)
            else:
                msg = (
                    f"The selected `nodata_output` value {nodata_output} does not fit "
                    f"in the array dtype {output.dtype}. Choose a different value or "
                    "set `allow_cast=True` to automatically cast the output."
                )
                raise ValueError(msg)

        if (
            check_output_for_nodata
            and not np.isnan(nodata_output)
            and nodata_output in output
        ):
            warn(
                f"The selected `nodata_output` value {nodata_output} was found in the "
                "array returned by the applied ufunc. This may indicate a valid pixel "
                "being masked. To suppress this warning, set "
                "`check_output_for_nodata=False`.",
                category=UserWarning,
                stacklevel=2,
            )

        return output

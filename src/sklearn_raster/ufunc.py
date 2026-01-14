from __future__ import annotations

from typing import TYPE_CHECKING, cast
from warnings import warn

import numpy as np
import numpy.ma as ma
import xarray as xr
from numpy.typing import NDArray

from .types import ArrayUfunc, FeatureArrayType, MaybeTuple
from .utils.decorators import (
    limit_inner_threads,
    map_over_arguments,
    with_inputs_reshaped_to_ndim,
)
from .utils.features import get_minimum_precise_numeric_dtype

if TYPE_CHECKING:
    from .features import FeatureArray


class _UfuncInput:
    """
    An array of samples with NoData handling for use in a RasterUfunc.

    The processor takes Numpy arrays in the shape (samples, features) and:

    1. Fills NaN samples in the input array.
    2. Passes valid samples to the ufunc.
    3. Masks NoData values in the ufunc output.
    """

    feature_dim = -1

    def __init__(self, samples: NDArray, *, nodata_input: ma.MaskedArray):
        self.samples = samples
        self.nodata_input = nodata_input

        # We can take some shortcuts if the input array type can't contain NaNs
        self._input_supports_nan = np.issubdtype(samples.dtype, np.floating)
        self.nodata_mask = self._get_nodata_mask()

        num_samples = self.samples.shape[0]
        self._num_masked = 0 if self.nodata_mask is None else self.nodata_mask.sum()
        self._num_unmasked = num_samples - self._num_masked

    def _get_nodata_mask(self) -> NDArray | None:
        # Skip allocating a mask if there's no chance of NoData values
        nodata_specified = np.any(~self.nodata_input.mask)
        if not self._input_supports_nan and not nodata_specified:
            return None

        mask = np.zeros(self.samples.shape, dtype=bool)

        # If it's floating point, always mask NaNs
        if self._input_supports_nan:
            mask |= np.isnan(self.samples)

        # If NoData was specified, mask those values
        if nodata_specified:
            # Fast path: NoData values should be applied to all features
            if not np.any(self.nodata_input.mask):
                mask |= self.samples == self.nodata_input.data
            # Slow path: NoData values are missing from some features
            else:
                mask |= (self.samples == self.nodata_input.data) & (
                    ~self.nodata_input.mask
                )

        # Return a mask where any feature contains NoData
        return mask.max(axis=self.feature_dim)

    def _fill_nans(self, nan_fill: float | None) -> NDArray:
        """Fill the flat input array with NaNs filled."""
        if self._input_supports_nan:
            return np.where(np.isnan(self.samples), nan_fill, self.samples)

        return self.samples


class FeaturewiseUfunc:
    """
    Build a feature-wise universal function with NoData filling, skipping, and masking.

    The ufunc is applied with a single core input dimension representing features and
    returns arbitrary outputs with the number and dimensions defined by `output_dims`.

    Parameters
    ----------
    func : callable
        A function to apply to flattened array(s). The function should accept one or
        more array of shape (samples, features) and return one or more arrays of shape
        (samples, size), where size is defined by `output_sizes`.
    output_dims : list[list[str]]
        Output core dimension names as a list of lists. The outer list represents the
        number of return values from `func` and the inner lists represent the output
        dimensions of each return value. A single value, e.g. [['variable']] represents
        a single return with a single output dimension named 'variable'.
    output_dtypes : list[np.dtype], optional
        The expected output data types for each value returned by `func`. Required for
        Dask-backed arrays.
    output_sizes : dict[str, int], optional
        Mapping from dimension names in `output_dims` to their sizes. Required for Dask-
        backed arrays.
    output_coords : dict[str, list[str] | list[int]], optional
        Mapping from dimension names in `output_dims` to their coordinates. If not
        provided, defaults to sequential integer coordinates for each output dimension.
    """

    def __init__(
        self,
        func: ArrayUfunc,
        *,
        output_dims: list[list[str]] | None = None,
        output_dtypes: list[np.dtype] | None = None,
        output_sizes: dict[str, int] | None = None,
        output_coords: dict[str, list[str] | list[int]] | None = None,
    ):
        self.func = func
        self.output_dims = output_dims
        self.output_dtypes = output_dtypes
        # Xarray raises a confusing TypeError if output_sizes is required and isn't
        # iterable. An empty dict will still fail, but with a better message.
        self.output_sizes = output_sizes or {}
        # Default to sequential coordinates for each output dimension
        self.output_coords = output_coords or {
            k: list(range(s)) for k, s in self.output_sizes.items()
        }

    def __call__(
        self,
        *arrays: FeatureArray,
        skip_nodata: bool = True,
        nodata_output: MaybeTuple[float | int] = np.nan,
        nan_fill: float | int | None = None,
        ensure_min_samples: int = 1,
        allow_cast: bool = False,
        check_output_for_nodata: bool = True,
        keep_attrs: bool = False,
        inner_thread_limit: int | None = 1,
        **ufunc_kwargs,
    ) -> MaybeTuple[FeatureArrayType]:
        """
        Apply a function to feature arrays with NoData filling, skipping, and masking.

        Parameters
        ----------
        arrays : FeatureArray
            One or more feature arrays to be passed positionally to the function.
        skip_nodata : bool, default=True
            If True, NoData and NaN values will be not be passed to `func`. This speeds
            up processing of partially masked features, but may be incompatible if
            `func` expects a consistent number of input samples.
        nodata_output : float or int or tuple, optional
            NoData samples in the input features will be replaced with this value in the
            output features. If the value does not fit the array dtype(s) returned by
            `func`, an error will be raised unless `allow_cast` is True. When `func`
            returns multiple arrays, you can provide either a single value for all
            arrays or a tuple with one value per output array. Defaults to np.nan.
        nan_fill : float or int, optional
            If `skip_nodata=False`, any NaNs in the input array will be filled with this
            value prior to calling `func` to avoid errors from functions that do not
            support NaN inputs. If None, NaNs will not be filled.
        ensure_min_samples : int, default 1
            The minimum number of samples that should be passed to `func`. If the
            array is fully masked and `skip_nodata=True`, dummy values (0) will be
            inserted to ensure this number of samples. No effect if the array contains
            enough unmasked samples or if `skip_nodata=False`.
        allow_cast : bool, default=False
            If True and the `func` output dtype is incompatible with the chosen
            `nodata_output` value, the output will be cast to the correct dtype instead
            of raising an error.
        check_output_for_nodata : bool, default True
            If True and `nodata_output` is not np.nan, a warning will be raised if the
            selected `nodata_output` value is returned by `func`, as this may indicate a
            valid sample being masked.
        keep_attrs : bool, default=False
            If True and the input is an Xarray object, the output will keep all
            attributes of the input features, unless they're set by `func`. Note that
            some attributes (e.g. `scale_factor`) may become inaccurate, which is why
            they are dropped by default. The `history` attribute will always be kept. No
            effect if the input is a Numpy array.
        inner_thread_limit : int or None, default=1
            The maximum number of threads allowed per Dask worker. Higher values can
            result in nested parallelism and oversubscription, which may cause
            slowdowns, stalls, or system crashes. Use caution when increasing the limit
            or disabling it by setting to `None`.
        **ufunc_kwargs
            Additional keyword arguments passed to the universal function.

        Returns
        -------
        FeatureArrayType or tuple[FeatureArrayType]
            The result of applying the universal function across features.
        """
        # Validate non-empty array inputs
        if not arrays:
            msg = (
                f"{self.__class__.__name__} requires at least one feature array input."
            )
            raise ValueError(msg)

        # Validate that all feature dimension names match
        feature_dim_names = list(set([array.feature_dim_name for array in arrays]))
        if len(feature_dim_names) > 1:
            msg = (
                "All input feature arrays must share the same feature dimension "
                f"name. Got {feature_dim_names}."
            )
            raise ValueError(msg)
        feature_dim_name = feature_dim_names[0]

        nodata_inputs: list[ma.MaskedArray] = [array.nodata_input for array in arrays]
        preprocessed = [
            array._preprocess_ufunc_input(array.feature_array) for array in arrays
        ]

        @with_inputs_reshaped_to_ndim(2)
        @limit_inner_threads(inner_thread_limit)
        def ufunc(*arrays: NDArray) -> MaybeTuple[NDArray]:
            return self._apply(
                *arrays,
                skip_nodata=skip_nodata,
                nodata_output=nodata_output,
                nodata_inputs=nodata_inputs,
                nan_fill=nan_fill,
                ensure_min_samples=ensure_min_samples,
                allow_cast=allow_cast,
                check_output_for_nodata=check_output_for_nodata,
                **ufunc_kwargs,
            )

        result = xr.apply_ufunc(
            ufunc,
            *preprocessed,
            dask="parallelized",
            # All inputs must share the same feature input core dimension
            input_core_dims=[[feature_dim_name]] * len(arrays),
            # The feature dimension is allowed to change size, so must be excluded
            exclude_dims=set((feature_dim_name,)),
            output_core_dims=self.output_dims,
            output_dtypes=self.output_dtypes,
            # Keep all attributes here to avoid dropping the spatial reference from the
            # coordinate attributes. Unwanted attrs will be dropped during
            # postprocessing.
            keep_attrs=True,
            dask_gufunc_kwargs=dict(
                output_sizes=self.output_sizes,
                allow_rechunk=True,
            ),
        )

        # TODO: Ideally we could just build a feature array from result and let xarray
        # decide the type, but that doesn't work because pre-processing converts
        # everything to DataArrays. If that's unavoidable, we could check the input
        # types and match by the same priority that Xarray uses, which is Dataset >
        # DataArray > Numpy array.
        return arrays[0]._postprocess_ufunc_output(
            result,
            output_coords=self.output_coords,
            nodata_output=nodata_output,
            func=self.func,
            keep_attrs=keep_attrs,
        )

    def _apply(
        self,
        *arrays: NDArray,
        nodata_inputs: list[ma.MaskedArray],
        skip_nodata: bool = True,
        nodata_output: MaybeTuple[float | int] = np.nan,
        nan_fill: float | int | None = None,
        ensure_min_samples: int = 1,
        allow_cast: bool = False,
        check_output_for_nodata: bool = True,
        **ufunc_kwargs,
    ) -> MaybeTuple[NDArray]:
        # Convert flat Numpy arrays to _UfuncInputs which handle NoData
        uinputs = [
            _UfuncInput(array, nodata_input=nodata)
            for array, nodata in zip(arrays, nodata_inputs, strict=True)
        ]

        # Only fill NaNs in the input arrays if they're not being skipped
        if nan_fill is not None and not skip_nodata:
            for uinput in uinputs:
                uinput.samples = uinput._fill_nans(nan_fill)

        any_masked = bool(sum([array._num_masked for array in uinputs]))

        # Build a cumulative NoData mask across all samples
        nodata_masks = [
            uinput.nodata_mask for uinput in uinputs if uinput.nodata_mask is not None
        ]
        nodata_mask = (
            np.stack(nodata_masks, axis=-1).max(axis=-1) if nodata_masks else None
        )

        # Only skip NoData if there's something to skip
        if skip_nodata and any_masked:
            return self._apply_to_valid_samples(
                *[uinput.samples for uinput in uinputs],
                ensure_min_samples=ensure_min_samples,
                nodata_mask=nodata_mask,
                nodata_output=nodata_output,
                allow_cast=allow_cast,
                nan_fill=nan_fill,
                check_output_for_nodata=check_output_for_nodata,
                **ufunc_kwargs,
            )

        return self._apply_to_all_samples(
            *[uinput.samples for uinput in uinputs],
            nodata_mask=nodata_mask,
            nodata_output=nodata_output,
            allow_cast=allow_cast,
            check_output_for_nodata=check_output_for_nodata,
            **ufunc_kwargs,
        )

    def _apply_to_all_samples(
        self,
        *arrays: NDArray,
        nodata_mask: NDArray | None,
        nodata_output: MaybeTuple[float | int],
        allow_cast: bool,
        check_output_for_nodata: bool,
        **kwargs,
    ) -> NDArray | tuple[NDArray, ...]:
        """Apply a function to all samples in all arrays."""

        @map_over_arguments("result", "nodata_output")
        def mask_nodata(result: NDArray, nodata_output: float | int) -> NDArray:
            """Replace NoData values in the input array with `output_nodata`."""
            result = self._validate_nodata_output(
                result,
                nodata_output,
                allow_cast=allow_cast,
                check_output_for_nodata=check_output_for_nodata,
            )

            result[nodata_mask] = nodata_output
            return result

        result = self.func(*arrays, **kwargs)

        if nodata_mask is not None:
            return mask_nodata(result=result, nodata_output=nodata_output)

        return result

    def _apply_to_valid_samples(
        self,
        *arrays: NDArray,
        nodata_mask: NDArray | None,
        nodata_output: MaybeTuple[float | int],
        ensure_min_samples: int,
        allow_cast: bool,
        nan_fill: float | int | None,
        check_output_for_nodata: bool,
        **kwargs,
    ) -> NDArray | tuple[NDArray, ...]:
        """Apply a function to all non-NoData samples in an array."""
        # TODO: This assumes all samples are the same shape. Make sure that's true
        input_shape = arrays[0].shape

        # The NoData mask is guaranteed to exist since this method is only called when
        # there are masked samples, so we can safely cast it for type checking.
        nodata_mask = cast(NDArray, nodata_mask)
        num_unmasked = (~nodata_mask).sum()

        if inserted_dummy_values := num_unmasked < ensure_min_samples:
            if ensure_min_samples > input_shape[0]:
                raise ValueError(
                    f"Cannot ensure {ensure_min_samples} samples with only "
                    f"{input_shape[0]} total samples in the array."
                )

            # Fill NoData samples with dummy values to ensure minimum samples. Copy
            # the mask to avoid mutating it when it's temporarily disabled.
            dummy_mask = nodata_mask[:ensure_min_samples].copy()
            # Temporarily disable the mask so that dummy samples aren't skipped
            nodata_mask[:ensure_min_samples] = False

            for array in arrays:
                array[:ensure_min_samples][dummy_mask] = (
                    nan_fill if nan_fill is not None else 0
                )

        @map_over_arguments("result", "nodata_output")
        def populate_missing_samples(
            result: NDArray, nodata_output: float | int
        ) -> NDArray:
            """Insert the array result for valid samples into the full-shaped array."""
            result = self._validate_nodata_output(
                result,
                nodata_output,
                allow_cast=allow_cast,
                check_output_for_nodata=check_output_for_nodata,
            )

            # Ensure that the result has a feature dimension in case it was squeezed by
            # the ufunc. `atleast_2d` adds the new axis at the index 0, so transpose
            # twice to move it to the end.
            result = np.atleast_2d(result.T).T

            # Build an output array pre-masked with the fill value and cast to the
            # output dtype. The shape will be (n, f) where n is the number of samples
            # in the array and f is the number of features in the func result.
            full_result = np.full(
                (input_shape[0], result.shape[-1]),
                nodata_output,
                dtype=result.dtype,
            )
            full_result[~nodata_mask] = result

            # Re-mask any samples that were filled to ensure minimum samples
            if inserted_dummy_values:
                full_result[:ensure_min_samples][dummy_mask] = nodata_output
                nodata_mask[:ensure_min_samples][dummy_mask] = True

            return full_result

        # Apply the func only to valid samples
        func_result = self.func(*[array[~nodata_mask] for array in arrays], **kwargs)
        return populate_missing_samples(result=func_result, nodata_output=nodata_output)

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
        the output that may indicate valid samples being masked.
        """
        nodata_output_type = get_minimum_precise_numeric_dtype(nodata_output)

        if not np.can_cast(nodata_output_type, output.dtype):
            if allow_cast:
                output = output.astype(nodata_output_type)
            else:
                msg = (
                    f"The selected `nodata_output` value {nodata_output} "
                    f"({nodata_output_type}) does not fit in the array dtype "
                    f"({output.dtype}). "
                )
                if nodata_output_type.kind == output.dtype.kind == "f":
                    msg += (
                        "Consider casting `nodata_output` to a lower precision float "
                        "or set `allow_cast=True` to automatically cast the output."
                    )
                else:
                    msg += (
                        "Consider choosing a different `nodata_output` value or set "
                        "`allow_cast=True` to automatically cast the output."
                    )

                raise ValueError(msg)

        if (
            check_output_for_nodata
            and not np.isnan(nodata_output)
            and nodata_output in output
        ):
            warn(
                f"The selected `nodata_output` value {nodata_output} was found in the "
                "array returned by the applied ufunc. This may indicate a valid sample "
                "being masked. To suppress this warning, set "
                "`check_output_for_nodata=False`.",
                category=UserWarning,
                stacklevel=2,
            )

        return output

from __future__ import annotations

from typing import cast
from warnings import warn

import numpy as np
from numpy.typing import NDArray

from .types import ArrayUfunc, MaybeTuple
from .utils.features import get_minimum_precise_numeric_dtype
from .utils.wrapper import map_over_arguments


class UfuncSampleProcessor:
    """
    A processor for applying ufuncs to arrays of samples.

    The processor takes Numpy arrays in the shape (samples, features) and:

    1. Fills NaN samples in the input array.
    2. Passes valid samples to the ufunc.
    3. Masks NoData values in the ufunc output.
    """

    feature_dim = -1

    def __init__(self, samples: NDArray, *, nodata_input: NDArray):
        self.samples = samples
        self.nodata_input = nodata_input

        # We can take some shortcuts if the input array type can't contain NaNs
        self._input_supports_nan = np.issubdtype(samples.dtype, np.floating)
        self.nodata_mask = self._get_nodata_mask()

        num_samples = self.samples.shape[0]
        self._num_masked = 0 if self.nodata_mask is None else self.nodata_mask.sum()
        self._num_unmasked = num_samples - self._num_masked

    def _get_nodata_mask(self) -> NDArray | None:
        # Skip allocating a mask if the array is not float and NoData wasn't given
        nodata_specified = np.any([n is not None for n in self.nodata_input])
        if not self._input_supports_nan and not nodata_specified:
            return None

        mask = np.zeros(self.samples.shape, dtype=bool)

        # If it's floating point, always mask NaNs
        if self._input_supports_nan:
            mask |= np.isnan(self.samples)

        # If NoData was specified, mask those values
        if nodata_specified:
            mask |= self.samples == self.nodata_input

        # Return a mask where any feature contains NoData
        return mask.max(axis=self.feature_dim)

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
        Apply a function to the sample array with NoData filling, skipping, and masking.

        Parameters
        ----------
        func : callable
            A function to apply to the flattened array. The function should accept one
            array of shape (samples, features) and return one or more arrays of the same
            shape.
        skip_nodata : bool, default True
            If True, NoData and NaN values will be removed before passing the array to
            `func`. This can speed up processing of partially masked arrays, but may be
            incompatible with functions that expect a consistent number of samples.
        nodata_output : float or int, default np.nan
            NoData samples in the input will be replaced with this value in the output.
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
            `predict` methods. No effect if the array contains enough valid samples or
            if `skip_nodata=False`.
        allow_cast : bool, default False
            If True and the `func` output dtype is incompatible with the chosen
            `nodata_output` value, the output will be cast to the correct dtype.
            Otherwise, an error will be raised unless `allow_cast` is True.
        check_output_for_nodata : bool, default True
            If True and `nodata_output` is not NaN, a warning will be raised if the
            selected `nodata_output` value is returned by `func`, as this may indicate
            a valid sample being masked.
        **kwargs : dict
            Additional keyword arguments passed to `func`.
        """
        # Fill NaNs in the input array if they're not being skipped
        samples = self.samples if skip_nodata else self._fill_nans(nan_fill)

        # Only skip NoData if there's something to skip
        if skip_nodata and self._num_masked > 0:
            return self._apply_to_valid_samples(
                func,
                samples=samples,
                ensure_min_samples=ensure_min_samples,
                nodata_output=nodata_output,
                allow_cast=allow_cast,
                nan_fill=nan_fill,
                check_output_for_nodata=check_output_for_nodata,
                **kwargs,
            )

        return self._apply_to_all_samples(
            func,
            samples=samples,
            nodata_output=nodata_output,
            allow_cast=allow_cast,
            check_output_for_nodata=check_output_for_nodata,
            **kwargs,
        )

    def _fill_nans(self, nan_fill: float | None) -> NDArray:
        """Fill the flat input array with NaNs filled."""
        if nan_fill is not None and self._input_supports_nan:
            return np.where(np.isnan(self.samples), nan_fill, self.samples)

        return self.samples

    def _apply_to_all_samples(
        self,
        func: ArrayUfunc,
        *,
        samples: NDArray,
        nodata_output: MaybeTuple[float | int],
        allow_cast: bool,
        check_output_for_nodata: bool,
        **kwargs,
    ) -> NDArray | tuple[NDArray, ...]:
        """Apply a function to all samples in an array."""

        @map_over_arguments("result", "nodata_output")
        def mask_nodata(result: NDArray, nodata_output: float | int) -> NDArray:
            """Replace NoData values in the input array with `output_nodata`."""
            result = self._validate_nodata_output(
                result,
                nodata_output,
                allow_cast=allow_cast,
                check_output_for_nodata=check_output_for_nodata,
            )

            result[self.nodata_mask] = nodata_output
            return result

        result = func(samples, **kwargs)
        if self._num_masked > 0:
            return mask_nodata(result=result, nodata_output=nodata_output)

        return result

    def _apply_to_valid_samples(
        self,
        func: ArrayUfunc,
        *,
        samples: NDArray,
        nodata_output: MaybeTuple[float | int],
        ensure_min_samples: int,
        allow_cast: bool,
        nan_fill: float | int | None,
        check_output_for_nodata: bool,
        **kwargs,
    ) -> NDArray | tuple[NDArray, ...]:
        """Apply a function to all non-NoData samples in an array."""
        # The NoData mask is guaranteed to exist since this method is only called when
        # there are masked samples, so we can safely cast it for type checking.
        nodata_mask = cast(NDArray, self.nodata_mask)

        if inserted_dummy_values := self._num_unmasked < ensure_min_samples:
            if ensure_min_samples > samples.shape[0]:
                raise ValueError(
                    f"Cannot ensure {ensure_min_samples} samples with only "
                    f"{samples.shape[0]} total samples in the array."
                )

            # Fill NoData samples with dummy values to ensure minimum samples. Copy the
            # mask to avoid mutating it when it's temporarily disabled.
            dummy_mask = nodata_mask[:ensure_min_samples].copy()
            samples[:ensure_min_samples][dummy_mask] = (
                nan_fill if nan_fill is not None else 0
            )

            # Temporarily disable the mask so that dummy samples aren't skipped
            nodata_mask[:ensure_min_samples] = False

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
                (samples.shape[0], result.shape[-1]),
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
        func_result = func(samples[~nodata_mask], **kwargs)
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

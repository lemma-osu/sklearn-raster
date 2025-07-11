from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sized
from datetime import datetime, timezone
from typing import Any, Callable, Generic

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from .types import ArrayUfunc, FeatureArrayType, MaybeTuple, NoDataType
from .ufunc import UfuncSampleProcessor
from .utils.features import reshape_to_samples
from .utils.wrapper import map_over_arguments


class FeatureArray(Generic[FeatureArrayType], ABC):
    """A wrapper around an n-dimensional array of features."""

    feature_dim_name: str | None = None
    feature_dim: int = 0
    feature_names: NDArray

    def __init__(
        self, feature_array: FeatureArrayType, nodata_input: NoDataType = None
    ):
        self.feature_array = feature_array
        self.n_features = self.feature_array.shape[self.feature_dim]
        self.nodata_input: NDArray = self._validate_nodata_input(nodata_input)

    def _validate_nodata_input(self, nodata_input: NoDataType) -> NDArray:
        """
        Get an array of NoData values in the shape (n_features,) based on user input.

        Scalars are broadcast to all features while sequences are checked against the
        number of features and cast to ndarrays. There is no need to specify np.nan as a
        NoData value because it will be masked automatically for floating point arrays.
        """
        # If it's None or a numeric scalar, broadcast it to all features
        if nodata_input is None or (
            isinstance(nodata_input, (float, int))
            and not isinstance(nodata_input, bool)
        ):
            return np.full((self.n_features,), nodata_input)

        # If it's not a scalar, it must be an iterable
        if not isinstance(nodata_input, Sized) or isinstance(nodata_input, (str, dict)):
            raise TypeError(
                f"Invalid type `{type(nodata_input).__name__}` for `nodata_input`. "
                "Provide a single number to apply to all features, a sequence of "
                "numbers, or None."
            )

        # If it's an iterable, it must contain one element per feature
        if len(nodata_input) != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} NoData values but got {len(nodata_input)}."
                f" The length of `nodata_input` must match the number of features."
            )

        return np.asarray(nodata_input)

    def apply_ufunc_across_features(
        self,
        func: ArrayUfunc,
        *,
        output_dims: list[list[str]],
        output_dtypes: list[np.dtype] | None = None,
        output_sizes: dict[str, int] | None = None,
        output_coords: dict[str, list[str] | list[int]] | None = None,
        skip_nodata: bool = True,
        nodata_output: MaybeTuple[float | int] = np.nan,
        nan_fill: float | int | None = None,
        ensure_min_samples: int = 1,
        allow_cast: bool = False,
        check_output_for_nodata: bool = True,
        keep_attrs: bool = False,
        **ufunc_kwargs,
    ) -> FeatureArrayType | tuple[FeatureArrayType]:
        """Apply a universal function to all features of the array."""
        if output_sizes is not None:
            # Default to sequential coordinates for each output dimension
            output_coords = output_coords or {
                k: list(range(s)) for k, s in output_sizes.items()
            }

        @reshape_to_samples
        def ufunc(x):
            return UfuncSampleProcessor(x, nodata_input=self.nodata_input).apply(
                func,
                skip_nodata=skip_nodata,
                nodata_output=nodata_output,
                nan_fill=nan_fill,
                ensure_min_samples=ensure_min_samples,
                allow_cast=allow_cast,
                check_output_for_nodata=check_output_for_nodata,
                **ufunc_kwargs,
            )

        result = xr.apply_ufunc(
            ufunc,
            self._preprocess_ufunc_input(self.feature_array),
            dask="parallelized",
            input_core_dims=[[self.feature_dim_name]],
            exclude_dims=set((self.feature_dim_name,)),
            output_core_dims=output_dims,
            output_dtypes=output_dtypes,
            # Keep all attributes here to avoid dropping the spatial reference from the
            # coordinate attributes. Unwanted attrs will be dropped during
            # postprocessing.
            keep_attrs=True,
            dask_gufunc_kwargs=dict(
                output_sizes=output_sizes,
                allow_rechunk=True,
            ),
        )

        return self._postprocess_ufunc_output(
            result=result,
            output_coords=output_coords,
            nodata_output=nodata_output,
            func=func,
            keep_attrs=keep_attrs,
        )

    def _preprocess_ufunc_input(self, features: FeatureArrayType) -> FeatureArrayType:
        """
        Preprocess the input of an applied ufunc. No-op unless overridden by subclasses.
        """
        return features

    @abstractmethod
    @map_over_arguments("result", "nodata_output")
    def _postprocess_ufunc_output(
        self,
        result: FeatureArrayType,
        *,
        nodata_output: float | int,
        func: Callable,
        output_coords: dict[str, list[str | int]] | None = None,
        keep_attrs: bool = False,
    ) -> FeatureArrayType:
        """
        Postprocess the output of an applied ufunc.

        This method should be overridden by subclasses to handle any necessary
        transformations to the output data, e.g. transposing dimensions.
        """

    @staticmethod
    def from_feature_array(
        feature_array: Any, nodata_input: NoDataType = None
    ) -> FeatureArray:
        """Create a FeatureArray from a supported feature type."""
        if isinstance(feature_array, np.ndarray):
            return NDArrayFeatures(feature_array, nodata_input=nodata_input)

        if isinstance(feature_array, xr.DataArray):
            return DataArrayFeatures(feature_array, nodata_input=nodata_input)

        if isinstance(feature_array, xr.Dataset):
            return DatasetFeatures(feature_array, nodata_input=nodata_input)

        msg = f"Unsupported feature array type `{type(feature_array).__name__}`."
        raise TypeError(msg)


class NDArrayFeatures(FeatureArray):
    """Features stored in a Numpy NDArray of shape (features, ...)."""

    feature_names = np.array([])

    def __init__(self, features: NDArray, nodata_input: NoDataType = None):
        super().__init__(features, nodata_input=nodata_input)

    def _preprocess_ufunc_input(self, features: NDArray) -> NDArray:
        """Preprocess by moving features to the last dimension for apply_ufunc."""
        # Copy to avoid mutating the original array
        return np.moveaxis(features.copy(), 0, -1)

    @map_over_arguments("result")
    def _postprocess_ufunc_output(
        self,
        result: NDArray,
        *,
        nodata_output: float | int,
        func: Callable,
        output_coords=None,
        keep_attrs: bool = False,
    ) -> NDArray:
        """Postprocess the output by moving features back to the first dimension."""
        return np.moveaxis(result, -1, 0)


class DataArrayFeatures(FeatureArray):
    """Features stored in an xarray DataArray of shape (features, ...)."""

    def __init__(self, features: xr.DataArray, nodata_input: NoDataType = None):
        super().__init__(features, nodata_input=nodata_input)
        self.feature_dim_name = features.dims[self.feature_dim]

    @property
    def feature_names(self) -> NDArray:
        return self.feature_array[self.feature_dim_name].values

    def _validate_nodata_input(self, nodata_input: NoDataType) -> NDArray:
        """
        Get an array of NoData values in the shape (features,) based on user input and
        DataArray metadata.
        """
        # Defer to user-provided NoData values over stored attributes
        if nodata_input is not None:
            return super()._validate_nodata_input(nodata_input)

        # Otherwise, use _FillValue if present (or None) for all features
        return np.full((self.n_features,), self.feature_array.attrs.get("_FillValue"))

    @map_over_arguments("result", "nodata_output")
    def _postprocess_ufunc_output(
        self,
        result: xr.DataArray,
        *,
        nodata_output: float | int,
        func: Callable,
        output_coords: dict[str, list[str | int]] | None = None,
        keep_attrs: bool = False,
    ) -> xr.DataArray:
        """Process the ufunc output by assigning coordinates and transposing."""
        if output_coords is not None:
            result = result.assign_coords(output_coords)

        # Transpose features from the last to the first dimension
        result = result.transpose(result.dims[-1], ...)

        # Reset the global attributes while setting _FillValue and modifying history.
        # Note that coordinate attributes are retained to preserve spatial reference,
        # if present.
        result.attrs = self._get_attrs(
            result.attrs,
            fill_value=nodata_output,
            append_to_history=func.__qualname__,
            keep_attrs=keep_attrs,
        )

        return result

    def _get_attrs(
        self,
        attrs: dict[str, Any],
        fill_value: float | int | None = None,
        append_to_history: str | None = None,
        keep_attrs: bool = False,
        new_attrs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Get CF-compliant attributes for the DataArray.

        Parameters
        ----------
        attrs : dict[str, Any]
            Existing attributes to preserve or modify.
        fill_value : float | int, optional
            The fill value to set for the _FillValue attribute. Ignored if None or NaN.
        append_to_history : str, optional
            A string to append to the history attribute, typically the function name
            that was applied. If None, no history is appended.
        new_attrs : dict[str, Any], optional
            Additional attributes to set or override in the DataArray.
        keep_attrs : bool, default False
            If True, preserve existing attributes. Otherwise, all unmodified attributes
            are dropped.
        """
        set_attrs = {}
        prev_history = attrs.get("history", "")

        if append_to_history is not None:
            timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
            set_attrs["history"] = (
                prev_history + "\n" if prev_history else ""
            ) + f"{timestamp} {append_to_history}"
        elif prev_history:
            set_attrs["history"] = prev_history

        if fill_value is not None and not np.isnan(fill_value):
            set_attrs["_FillValue"] = fill_value

        if new_attrs is not None:
            set_attrs.update(new_attrs)

        if keep_attrs:
            return attrs | set_attrs

        return set_attrs


class DatasetFeatures(DataArrayFeatures):
    """Features stored in an xarray Dataset with features as variables."""

    def __init__(self, features: xr.Dataset, nodata_input: NoDataType = None):
        # The data will be stored as a DataArray, but keep the Dataset for metadata
        # like _FillValues.
        self.dataset = features
        super().__init__(features.to_dataarray(), nodata_input=nodata_input)

    @property
    def feature_names(self) -> NDArray:
        return np.array(list(self.dataset.data_vars))

    def _validate_nodata_input(self, nodata_input: NoDataType) -> NDArray:
        """
        Get an array of NoData values in the shape (features,) based on user input and
        Dataset metadata.
        """
        # Defer to user-provided NoData values over stored attributes
        if nodata_input is not None:
            return super()._validate_nodata_input(nodata_input)

        # Otherwise, use _FillValue if present (or None) for each feature
        return np.asarray(
            [
                self.dataset[var].attrs.get("_FillValue")
                for var in self.dataset.data_vars
            ]
        )

    @map_over_arguments("result", "nodata_output")
    def _postprocess_ufunc_output(
        self,
        result: xr.DataArray,
        *,
        nodata_output: float | int,
        func: Callable,
        output_coords: dict[str, list[str | int]] | None = None,
        keep_attrs: bool = False,
    ) -> xr.Dataset:
        """Process the ufunc output converting from DataArray to Dataset."""
        result = super()._postprocess_ufunc_output(
            result=result,
            output_coords=output_coords,
            nodata_output=nodata_output,
            func=func,
            keep_attrs=keep_attrs,
        )
        var_dim = result.dims[self.feature_dim]
        ds = result.to_dataset(dim=var_dim, promote_attrs=True)

        # Drop variable-level attrs
        ds.attrs.pop("_FillValue", None)

        for var in ds.data_vars:
            ds[var].attrs = self._get_attrs(
                ds[var].attrs,
                fill_value=nodata_output,
                new_attrs={"long_name": var},
                keep_attrs=keep_attrs,
            )

        return ds

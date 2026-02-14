from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence, Sized
from datetime import datetime, timezone
from typing import Any, Callable, Generic

import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from .types import (
    FeatureArrayType,
    MissingType,
    NoDataMap,
    NoDataType,
)
from .utils.decorators import (
    map_over_arguments,
)
from .utils.features import can_cast_nodata_value


class FeatureArray(Generic[FeatureArrayType], ABC):
    """A wrapper around an n-dimensional array of features."""

    feature_dim_name: str | None = None
    feature_dim: int = 0

    def __init__(
        self,
        feature_array: FeatureArrayType,
        nodata_input: NoDataType | MissingType = MissingType.MISSING,
    ):
        self.feature_array = feature_array
        self.feature_names = self._validate_feature_names()
        self.n_features = self.feature_array.shape[self.feature_dim]
        self.nodata_input = self._validate_nodata_input(nodata_input)

    @abstractmethod
    def _validate_feature_names(self) -> NDArray:
        """Validate and return feature names from the feature array."""

    def _validate_nodata_input(
        self, nodata_input: NoDataType | MissingType
    ) -> ma.MaskedArray:
        """
        Get an array of NoData values in the shape (n_features,) based on user input.

        Scalars are broadcast to all features while sequences are checked against the
        number of features and cast to ndarrays. There is no need to specify np.nan as a
        NoData value because it will be masked automatically for floating point arrays.
        """
        # If NoData isn't provided, attempt to infer values using subclass logic
        if nodata_input is MissingType.MISSING:
            inferred_nodata = list(self._get_default_nodata_mapping().values())
            return self._build_masked_nodata_array(inferred_nodata)

        # If it's a valid scalar (including None), broadcast it to all features
        if nodata_input is None or isinstance(nodata_input, (float, int, bool)):
            values = [nodata_input] * self.n_features
            return self._build_masked_nodata_array(values)

        # If it's a dict, map names or indices to values, falling back to inferred
        # default values for unspecified features
        if isinstance(nodata_input, dict):
            defaults = self._get_default_nodata_mapping()
            for key, val in nodata_input.items():
                if key not in defaults:
                    msg = (
                        f"`{key}` is not a valid feature name/index, and therefore "
                        "can't be assigned a NoData value. Choose from "
                        f"{list(defaults.keys())}."
                    )
                    raise KeyError(msg)

                defaults[key] = val
            nodata_input = list(defaults.values())

        # If it's not a scalar or a dict, it must be an iterable
        if not isinstance(nodata_input, Sized) or isinstance(nodata_input, str):
            raise TypeError(
                f"Invalid type `{type(nodata_input).__name__}` for `nodata_input`. "
                "Provide a single value to apply to all features, a sequence of "
                "values, a mapping from feature names/indices to values, or None."
            )

        # If it's an iterable, it must contain one element per feature
        if len(nodata_input) != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} NoData values but got {len(nodata_input)}."
                f" The length of `nodata_input` must match the number of features."
            )

        # Assign values feature-wise, disabling masking for None entries
        return self._build_masked_nodata_array(nodata_input)

    @abstractmethod
    def _get_default_nodata_mapping(self) -> NoDataMap:
        """Return a mapping from feature names or indices to default NoData values."""

    def _build_masked_nodata_array(
        self, values: Sequence[float | None]
    ) -> ma.MaskedArray:
        """
        Build a masked NoData array from a sequence of NoData values.

        Parameters
        ----------
        values : Sequence[float | None]
            A sequence of NoData values, one per feature, where None indicates a missing
            value. Values that can't be safely cast to the feature array dtype will
            result in an error.

        Returns
        -------
        ma.MaskedArray
            A masked array of NoData values with shape (n_features,). The mask is True
            for features where the NoData value is None, indicating a missing value that
            should not be applied.
        """
        # Keep a copy of the original values as an object array so that we can check
        # their dtype against the target dtype before any implicit casting occurs
        original_values = np.asarray(values, dtype=object)

        # Create a mask to mark None values as missing
        missing_values = np.asarray([v is None for v in values], dtype=bool)

        # Replace missing NoData values with zero since it fits in any dtype
        missing_fill_value = 0
        filled_values = np.where(missing_values, missing_fill_value, values)
        target_dtype = self.feature_array.dtype

        # Raise if any NoData values can't be safely cast to the feature array dtype,
        # to avoid masking with a rounded or truncated value.
        uncastable = []
        for i, val in enumerate(original_values):
            if missing_values[i]:
                continue
            if not can_cast_nodata_value(val, target_dtype):
                uncastable.append(val)
        if uncastable:
            msg = (
                f"The selected or inferred NoData value(s) {uncastable} cannot be "
                f"safely cast to the feature array dtype {target_dtype}. Ensure that "
                "all values in `nodata_input` are compatible with the data type, or "
                "`None` to disable NoData masking. If `nodata_input` was not provided, "
                "check the `_FillValue` attribute(s) of the input data."
            )
            raise ValueError(msg)

        return ma.masked_array(
            filled_values,
            mask=missing_values,
            dtype=target_dtype,
            fill_value=missing_fill_value,
        )

    def _preprocess_ufunc_input(self, features: FeatureArrayType) -> FeatureArrayType:
        """
        Preprocess the input of an applied ufunc. No-op unless overridden by subclasses.
        """
        return features

    @abstractmethod
    @map_over_arguments("result", "nodata_output", "output_coords")
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
        feature_array: Any, nodata_input: NoDataType | MissingType = MissingType.MISSING
    ) -> FeatureArray:
        """Create a FeatureArray from a supported feature type."""
        if isinstance(feature_array, np.ndarray):
            return NDArrayFeatures(feature_array, nodata_input=nodata_input)

        if isinstance(feature_array, xr.DataArray):
            return DataArrayFeatures(feature_array, nodata_input=nodata_input)

        if isinstance(feature_array, xr.Dataset):
            return DatasetFeatures(feature_array, nodata_input=nodata_input)

        if isinstance(feature_array, pd.DataFrame):
            return DataFrameFeatures(feature_array, nodata_input=nodata_input)

        msg = f"Unsupported feature array type `{type(feature_array).__name__}`."
        raise TypeError(msg)


class NDArrayFeatures(FeatureArray):
    """Features stored in a Numpy NDArray of shape (features, ...)."""

    def _validate_feature_names(self) -> NDArray:
        # NDArrays are unnamed, so no validation checks are needed
        return np.array([])

    def _get_default_nodata_mapping(self) -> NoDataMap:
        # Use sequential indices with no inferred NoData value for all features
        return {i: None for i in range(self.n_features)}

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

    def __init__(
        self,
        features: xr.DataArray,
        nodata_input: NoDataType | MissingType = MissingType.MISSING,
    ):
        self.feature_dim_name = features.dims[self.feature_dim]
        super().__init__(features, nodata_input=nodata_input)

    def _validate_feature_names(self) -> NDArray:
        names = self.feature_array[self.feature_dim_name].values.astype(object)
        # Feature names must be unique to allow mapping NoData values by name
        duplicated_names = [name for name, count in Counter(names).items() if count > 1]
        if duplicated_names:
            msg = (
                "All feature names must be unique. Found duplicated names "
                f"{duplicated_names}."
            )
            raise ValueError(msg)
        return names

    def _get_default_nodata_mapping(self) -> NoDataMap:
        # Infer NoData from global _FillValue (or None) for all features
        global_fill_value = self.feature_array.attrs.get("_FillValue")
        return {name: global_fill_value for name in self.feature_names}

    @map_over_arguments("result", "nodata_output", "output_coords")
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

    def __init__(
        self,
        features: xr.Dataset,
        nodata_input: NoDataType | MissingType = MissingType.MISSING,
    ):
        # The data will be stored as a DataArray, but keep the Dataset for metadata
        # like _FillValues.
        self.dataset = features
        super().__init__(features.to_dataarray(), nodata_input=nodata_input)

    def _get_default_nodata_mapping(self) -> NoDataMap:
        # Infer NoData from variable-level _FillValues (or None) per-feature
        return {
            var: self.dataset[var].attrs.get("_FillValue")
            for var in self.dataset.data_vars
        }

    @map_over_arguments("result", "nodata_output", "output_coords")
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


class DataFrameFeatures(DataArrayFeatures):
    """Features stored in a Pandas DataFrame of shape (samples, features)."""

    def __init__(
        self,
        features: pd.DataFrame,
        nodata_input: NoDataType | MissingType = MissingType.MISSING,
    ):
        # The data will be stored as a DataArray, but keep the DataFrame for metadata
        # like the index name.
        self.dataframe = features
        data_array = xr.Dataset.from_dataframe(features).to_dataarray()
        super().__init__(data_array, nodata_input=nodata_input)

    @map_over_arguments("result", "nodata_output", "output_coords")
    def _postprocess_ufunc_output(
        self,
        result: xr.DataArray,
        *,
        nodata_output: float | int,
        func: Callable,
        output_coords: dict[str, list[str | int]] | None = None,
        keep_attrs: bool = False,
    ) -> pd.DataFrame:
        """Process the ufunc output converting from DataArray to DataFrame."""
        result = super()._postprocess_ufunc_output(
            result=result,
            output_coords=output_coords,
            nodata_output=nodata_output,
            func=func,
            keep_attrs=False,
        )

        df = (
            result
            # Transpose from (target, samples) back to (samples, target)
            .T.to_pandas()
            # Preserve the input index name(s)
            .rename_axis(self.dataframe.index.names, axis=0)
        )
        df.columns.name = self.dataframe.columns.name
        return df

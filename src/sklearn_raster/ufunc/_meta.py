from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..features import (
    DataArrayFeatures,
    DataFrameFeatures,
    DatasetFeatures,
    FeatureArray,
    NDArrayFeatures,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ..types import MaybeTuple


@dataclass
class Dimension:
    """
    A core dimension for a Ufunc output.

    Parameters
    ----------
    name : str
        The name of the dimension.
    size : int, optional
        The size of the dimension. If `coords` is provided instead, size can be inferred
        from the number of coordinates.
    coords : list[str] or list[int], optional
        The coordinate values for the dimension. The number of values must be equal to
        `size` if both are provided.
    """

    name: str
    size: int | None = None
    coords: Sequence[str | int] | None = None

    def __post_init__(self):
        self._infer_size_from_coords()

    def _infer_size_from_coords(self) -> None:
        if self.coords is None:
            return
        if not isinstance(self.coords, (list, tuple)):
            msg = (
                f"Dimension coordinates must be a list or tuple of values. "
                f"Got `{self.coords.__class__.__name__}`."
            )
            raise ValueError(msg)
        if self.size is None:
            self.size = len(self.coords)
            return
        if self.size != len(self.coords):
            msg = (
                f"Dimension '{self.name}' has size {self.size} but {len(self.coords)} "
                "coordinates. Size and coordinates must match if both are provided."
            )
            raise ValueError(msg)


@dataclass
class Output:
    """
    Metadata for a single output array from a Ufunc.

    Parameters
    ----------
    dims : list[Dimension]
        The core dimensions of the output.
    dtype : np.dtype, optional
        The output data type.
    nodata : float or int, optional
        The value used to encode NoData in the output array. If not provided, it will
        be inferred from the data type if possible (NaN for floating point types,
        minimum value for signed integers, and maximum value for unsigned integers).
    """

    dims: list[Dimension]
    dtype: np.dtype | None
    nodata: float | int

    def __init__(
        self,
        dims: list[Dimension],
        dtype: np.dtype | None = None,
        nodata: float | int | None = None,
    ):
        self.dims = dims
        self.dtype = dtype
        self.nodata = self._infer_nodata_from_dtype(nodata)

    def _infer_nodata_from_dtype(self, nodata: float | int | None) -> float | int:
        """Attempt to infer the NoData value from the data type."""
        if nodata is not None:
            return nodata
        if self.dtype is not None:
            if np.issubdtype(self.dtype, np.floating):
                return np.dtype(self.dtype).type(np.nan)
            if np.issubdtype(self.dtype, np.unsignedinteger):
                return np.iinfo(self.dtype).max
            if np.issubdtype(self.dtype, np.signedinteger):
                return np.iinfo(self.dtype).min

        raise ValueError("NoData value could not be inferred. Please provide `nodata`.")

    @classmethod
    def from_1d(
        cls,
        name: str,
        size: int | None = None,
        coords: Sequence[str | int] | None = None,
        dtype: np.dtype | None = None,
        nodata: float | int | None = None,
    ) -> Output:
        """
        Construct metadata for a ufunc with a single core output dimension.

        Parameters
        ----------
        name : str
            The name of the core dimension.
        size : int, optional
            The size of the dimension. If `coords` is provided instead, size can be
            inferred from the number of coordinates.
        coords : list[str] or list[int], optional
            The coordinate values for the dimension. The number of values must be equal
            to `size` if both are provided.
        dtype : np.dtype, optional
            The output data type.
        nodata : float or int, optional
            The value used to encode NoData in the output array. If not provided, it
            will be inferred from the data type if possible (NaN for floating point
            types, minimum value for signed integers, and maximum value for unsigned
            integers).

        Returns
        -------
        Output
            An Output instance with the specified core dimension and data type.
        """
        return cls(
            dims=[Dimension(name=name, size=size, coords=coords)],
            dtype=dtype,
            nodata=nodata,
        )


@dataclass
class _UfuncMeta:
    """Metadata describing the outputs returned by a ufunc."""

    num_outputs: int
    output_sizes: dict[str, int]
    output_dtypes: tuple[type[np.generic] | None, ...]
    output_core_dims: tuple[list[str], ...]
    output_coords: tuple[dict[str, Sequence[str | int]], ...]
    nodata_outputs: tuple[float | int, ...]

    @staticmethod
    def _get_output_sizes(outputs: list[Output]) -> dict[str, int]:
        """Get a mapping from dimension names to sizes for all output dimensions."""
        output_sizes: dict[str, int] = {}
        for output in outputs:
            for dim in output.dims:
                if dim.size is None:
                    continue

                if dim.name in output_sizes and output_sizes[dim.name] != dim.size:
                    # NOTE: This is a dask.apply_gufunc constraint since sizes are
                    # passed as a single dict per call rather than per output. We may
                    # be able to relax this in non-Dask cases.
                    msg = (
                        "Different sizes for the same dimension are not supported. "
                        f"Found dimension '{dim.name}' with sizes "
                        f"{output_sizes[dim.name]} and {dim.size}."
                    )
                    raise ValueError(msg)
                output_sizes[dim.name] = dim.size
        return output_sizes

    @staticmethod
    def _get_output_dtypes(
        outputs: list[Output],
    ) -> tuple[type[np.generic] | None, ...]:
        """Get a list of output data types for each output array."""
        return tuple([output.dtype for output in outputs])

    @staticmethod
    def _get_output_core_dims(outputs: list[Output]) -> tuple[list[str], ...]:
        """Get a list of output core dimension names for each output array."""
        return tuple([[dim.name for dim in output.dims] for output in outputs])

    @staticmethod
    def _get_output_coords(
        outputs: list[Output],
    ) -> tuple[dict[str, Sequence[str | int]], ...]:
        return tuple(
            [
                {dim.name: dim.coords for dim in output.dims if dim.coords is not None}
                for output in outputs
            ]
        )

    @classmethod
    def from_outputs(cls, outputs: list[Output]) -> _UfuncMeta:
        """Construct ufunc metadata from a list of outputs."""
        return cls(
            num_outputs=len(outputs),
            output_sizes=cls._get_output_sizes(outputs),
            output_dtypes=cls._get_output_dtypes(outputs),
            output_core_dims=cls._get_output_core_dims(outputs),
            output_coords=cls._get_output_coords(outputs),
            nodata_outputs=tuple(output.nodata for output in outputs),
        )

    def with_inputs(
        self,
        *arrays: FeatureArray,
        nodata_output: MaybeTuple[float | int] | None = None,
    ) -> _CalledUfuncMeta:
        """Extend the UfuncMeta with call-specific input metadata."""
        return _CalledUfuncMeta.from_call(
            *arrays,
            ufunc_meta=self,
            nodata_output=nodata_output,
        )


@dataclass
class _CalledUfuncMeta(_UfuncMeta):
    """Extension of UfuncMeta that includes call-specific input metadata."""

    feature_dim_name: str | None
    nodata_inputs: tuple[float | int]
    input_core_dims: tuple[list[str | None], ...]
    exclude_dims: set[str | None]
    postprocessor: Callable

    @classmethod
    def from_call(
        cls,
        *arrays: FeatureArray,
        ufunc_meta: _UfuncMeta,
        nodata_output: MaybeTuple[float | int] | None,
    ):
        feature_dim_name = cls._get_feature_dim_name(arrays)

        # Start with the base ufunc metadata
        base_meta = ufunc_meta.__dict__.copy()
        # Drop the default nodata_outputs since they're overriden with called values
        base_meta.pop("nodata_outputs")

        return cls(
            **base_meta,
            feature_dim_name=feature_dim_name,
            nodata_outputs=cls._get_nodata_outputs(ufunc_meta, nodata_output),
            nodata_inputs=tuple(array.nodata_input for array in arrays),
            input_core_dims=tuple([[feature_dim_name]] * len(arrays)),
            # The feature dimension is allowed to change size, so must be excluded
            exclude_dims=set((feature_dim_name,)),
            postprocessor=cls._get_postprocessor(arrays),
        )

    @staticmethod
    def _get_feature_dim_name(arrays: tuple[FeatureArray, ...]) -> str | None:
        """Validate the feature dimension name and return it."""
        # Validate non-empty array inputs
        if not arrays:
            raise ValueError("Ufuncs requires at least one feature array input.")

        # Validate that all feature dimension names match
        feature_dim_names = list(set([array.feature_dim_name for array in arrays]))
        if len(feature_dim_names) > 1:
            msg = (
                "All input feature arrays must share the same feature dimension "
                f"name. Got {feature_dim_names}."
            )
            raise ValueError(msg)

        return feature_dim_names[0]

    @staticmethod
    def _get_nodata_outputs(
        ufunc_meta: _UfuncMeta, nodata_output: MaybeTuple[float | int] | None
    ) -> tuple[float | int, ...]:
        """
        Take optional user-provided NoData output value(s) and use them to override
        the default NoData output value(s) defined in the ufunc metadata.
        """
        # Default to NoData values defined in the ufunc metadata
        if nodata_output is None:
            return ufunc_meta.nodata_outputs
        # Broadcast single values to multiple outputs
        if not isinstance(nodata_output, (list, tuple)):
            return tuple([nodata_output] * ufunc_meta.num_outputs)
        # Validate that multiple values match the number of outputs
        if len(nodata_output) != ufunc_meta.num_outputs:
            raise ValueError(
                f"The ufunc defines {ufunc_meta.num_outputs} outputs, but "
                f"{len(nodata_output)} `nodata_output` values were provided."
            )
        return tuple(nodata_output)

    @staticmethod
    def _get_postprocessor(arrays: tuple[FeatureArray, ...]) -> Callable:
        # In normal usage, xr.apply_ufunc returns the type of the highest-priority input
        # following the type priority below. Because FeatureArray pre-processing coerces
        # all inputs to DataArrays, we need to manually apply that logic here by
        # having the highest priority FeatureArray post-process the output.
        type_priority = (
            DatasetFeatures,
            DataArrayFeatures,
            DataFrameFeatures,
            NDArrayFeatures,
        )
        highest_priority_array = sorted(
            arrays,
            key=lambda arr: type_priority.index(type(arr)),
        )[0]
        return highest_priority_array._postprocess_ufunc_output

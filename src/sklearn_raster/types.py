from __future__ import annotations

from collections.abc import Sequence
from typing import Callable, Union

import xarray as xr
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from typing_extensions import Any, Concatenate, ParamSpec, TypeVar

DaskBackedType = TypeVar("DaskBackedType", xr.DataArray, xr.Dataset)
FeatureArrayType = TypeVar("FeatureArrayType", NDArray, xr.DataArray, xr.Dataset)
EstimatorType = TypeVar("EstimatorType", bound=BaseEstimator)
AnyType = TypeVar("AnyType", bound=Any)
NoDataType = Union[float, Sequence[float], None]

Self = TypeVar("Self")
T = TypeVar("T")
P = ParamSpec("P")
RT = TypeVar("RT")

MaybeTuple = Union[T, tuple[T, ...]]

# A function that takes an NDArray and any parameters and returns one or more NDArrays
ArrayUfunc = Callable[Concatenate[NDArray, P], Union[NDArray, tuple[NDArray, ...]]]

from __future__ import annotations

import enum
from collections.abc import Callable, Hashable, Sequence
from typing import Concatenate, ParamSpec, TypeAlias, TypeVar

import pandas as pd
import xarray as xr
from numpy.typing import NDArray
from sklearn.base import BaseEstimator

# Unconstrained type vars
T = TypeVar("T")
P = ParamSpec("P")
RT = TypeVar("RT")

# Constrained and bound type vars
T_FeatureArrayType = TypeVar(
    "T_FeatureArrayType", NDArray, xr.DataArray, xr.Dataset, pd.DataFrame
)
T_EstimatorType = TypeVar("T_EstimatorType", bound=BaseEstimator)

# Type aliases
NoDataValue = float | int | bool | None
NoDataMap = dict[Hashable, NoDataValue]
NoDataType = NoDataValue | Sequence[NoDataValue] | NoDataMap
MaybeTuple: TypeAlias = T | tuple[T, ...]

# A sentinel value to distinguish missing parameters from None
MissingType = enum.Enum("MissingType", "MISSING")

# A function that takes one or more NDArrays and any parameters and returns one or more
# NDArrays
# TODO: Fix this type to indicate varargs inputs
ArrayUfunc = Callable[Concatenate[NDArray, P], NDArray | tuple[NDArray, ...]]

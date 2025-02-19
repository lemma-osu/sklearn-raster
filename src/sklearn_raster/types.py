from __future__ import annotations

from collections.abc import Sequence
from typing import Callable, Concatenate, Union

import xarray as xr
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from typing_extensions import Any, ParamSpec, TypeVar

DaskBackedType = TypeVar("DaskBackedType", xr.DataArray, xr.Dataset)
ImageType = TypeVar("ImageType", NDArray, xr.DataArray, xr.Dataset)
EstimatorType = TypeVar("EstimatorType", bound=BaseEstimator)
AnyType = TypeVar("AnyType", bound=Any)
NoDataType = Union[float, Sequence[float], None]
P = ParamSpec("P")
RT = TypeVar("RT")

# A function that takes an NDArray and any parameters and returns one or more NDArrays
ArrayUfunc = Callable[Concatenate[NDArray, P], NDArray | tuple[NDArray, ...]]

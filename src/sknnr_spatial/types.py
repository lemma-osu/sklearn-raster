from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

import xarray as xr
from numpy.typing import NDArray
from sklearn.base import BaseEstimator

DaskBackedType = TypeVar("DaskBackedType", xr.DataArray, xr.Dataset)
ImageType = TypeVar("ImageType", NDArray, xr.DataArray, xr.Dataset)
EstimatorType = TypeVar("EstimatorType", bound=BaseEstimator)
AnyType = TypeVar("AnyType", bound=Any)
NoDataType = float | Sequence[float] | None

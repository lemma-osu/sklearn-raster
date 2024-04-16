from typing import TypeVar

import xarray as xr
from numpy.typing import NDArray

DaskBackedType = TypeVar("DaskBackedType", xr.DataArray, xr.Dataset)
ImageType = TypeVar("ImageType", NDArray, xr.DataArray, xr.Dataset)

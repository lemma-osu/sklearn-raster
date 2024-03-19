from __future__ import annotations

from importlib import resources

import numpy as np
import pandas as pd
import rasterio
import rioxarray
import sknnr.datasets
import xarray as xr
from numpy.typing import NDArray

DATA_MODULE = "sknnr_spatial.datasets.data"


def _load_rasters_to_dataset(
    file_names: list[str], *, var_names: list[str], module_name: str, chunks=None
) -> xr.Dataset:
    """Load a list of rasters from the data module as an xarray Dataset."""
    das = []
    for file_name, var_name in zip(file_names, var_names):
        bin = resources.open_binary(module_name, file_name)

        da = rioxarray.open_rasterio(bin, chunks=chunks)
        da = da.to_dataset(dim="band").rename({1: var_name})
        das.append(da)

    return xr.merge(das)


def _load_rasters_to_array(file_names: list[str], *, module_name: str) -> np.ndarray:
    """Load a list of rasters from the data module as a numpy array."""
    arr = None
    for file_name in file_names:
        bin = resources.open_binary(module_name, file_name)

        with rasterio.open(bin) as src:
            band = src.read(1)
            arr = band if arr is None else np.dstack((arr, band))

    return arr


def load_swo_ecoplot(
    as_dataset: bool = False,
) -> tuple[NDArray | xr.Dataset, pd.DataFrame, pd.DataFrame]:
    """Load the southwest Oregon (SWO) USFS Region 6 Ecoplot dataset.

    The dataset contains:

     1. **Image data**: 128x128 pixel GeoTIFF image chips of 18 environmental and
        spectral variables at 30m resolution.
     2. **Plot data**: 3,005 plots with environmental, Landsat, and forest cover
        measurements. Ocular measurements of tree cover (COV) are categorized by
        major tree species present in southwest Oregon.  All data were collected in 2000
        and Landsat imagery processed through the CCDC algorithm was extracted for the
        same year.

    Parameters
    ----------
    as_dataset : bool, default=False
        If True, return the image data as an `xarray.Dataset` instead of a Numpy array.

    Returns
    -------
    tuple
        Image data as either a numpy array or `xarray.Dataset`, and plot data as X and
        y dataframes.

    Notes
    -----
    These data are a subset of the larger USDA Forest Service Region 6 Ecoplot
    database, which holds 28,000 plots on Region 6 National Forests across Oregon
    and Washington.  The larger database is managed by Patricia Hochhalter (USFS Region
    6 Ecology Program) and used by permission.  Ecoplots were originally used to
    develop plant association guides and are used for a wide array of applications.
    This subset represents plots that were collected in southwest Oregon in 2000.

    Examples
    --------

    >>> from sknnr_spatial.datasets import load_swo_ecoplot()
    >>> X_image, X, y = load_swo_ecoplot()
    >>> print(X_image.shape)
    (128, 128, 18)

    Reference
    ---------
    Atzet, T, DE White, LA McCrimmon, PA Martinez, PR Fong, and VD Randall. 1996.
    Field guide to the forested plant associations of southwestern Oregon.
    USDA Forest Service. Pacific Northwest Region, Technical Paper R6-NR-ECOL-TP-17-96.
    """
    X, y = sknnr.datasets.load_swo_ecoplot(return_X_y=True, as_frame=True)
    raster_names = [f"{var.lower()}.tif" for var in X.columns]
    module_name = ".".join([DATA_MODULE, "swo_ecoplot"])

    if as_dataset:
        X_image = _load_rasters_to_dataset(
            raster_names,
            var_names=X.columns,
            module_name=module_name,
            chunks={"x": 64, "y": 64},
        )
    else:
        X_image = _load_rasters_to_array(raster_names, module_name=module_name)

    return X_image, X, y

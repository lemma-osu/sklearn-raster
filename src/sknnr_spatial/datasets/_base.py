from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing_extensions import Any, Literal, overload

from sknnr_spatial import __version__
from sknnr_spatial.datasets._registry import registry

try:
    import pooch
    import rasterio
    import rioxarray
    import sknnr.datasets
    import xarray as xr
except ImportError:
    msg = (
        "Using the datasets module to load data requires additional dependencies. "
        "You can install them with `pip install sknnr-spatial[datasets]`."
    )
    raise ImportError(msg) from None


# Location of data files. The `version` placeholder will be replaced by pooch.
DATA_URL = "https://github.com/lemma-osu/sknnr-spatial/raw/{version}/src/sknnr_spatial/datasets/data"


_data_fetcher = pooch.create(
    base_url=DATA_URL,
    version=__version__,
    version_dev="main",
    path=pooch.os_cache("sknnr-spatial"),
    env="SKNNRSPATIAL_DATA_DIR",
    registry=registry,
    retry_if_failed=3,
)


def _load_rasters_to_dataset(
    file_paths: list[Path], *, var_names: list[str], chunks=None
) -> xr.Dataset:
    """Load a list of rasters as an xarray Dataset."""
    das = []
    for path, var_name in zip(file_paths, var_names):
        da = rioxarray.open_rasterio(path, chunks=chunks).rename(var_name).squeeze()
        das.append(da)

    return xr.merge(das)


def _load_rasters_to_array(file_paths: list[Path]) -> NDArray:
    """Load single-band rasters as a multi-band numpy array of shape (band, y, x)."""
    arr = None
    for path in file_paths:
        with rasterio.open(path) as src:
            band = src.read(1)
            # Add a band dimension to the array to allow concatenation
            band = band[np.newaxis, ...]

            arr = band if arr is None else np.concatenate((arr, band), axis=0)

    return arr


@overload
def load_swo_ecoplot(
    as_dataset: Literal[True],
    large_rasters: bool = False,
    chunks: Any = None,
) -> tuple[xr.Dataset, pd.DataFrame, pd.DataFrame]: ...


@overload
def load_swo_ecoplot(
    as_dataset: Literal[False] = False,
    large_rasters: bool = False,
    chunks: Any = None,
) -> tuple[NDArray, pd.DataFrame, pd.DataFrame]: ...


def load_swo_ecoplot(
    as_dataset: bool = False,
    large_rasters: bool = False,
    chunks: Any = None,
) -> tuple[NDArray | xr.Dataset, pd.DataFrame, pd.DataFrame]:
    """
    Load the southwest Oregon (SWO) USFS Region 6 Ecoplot dataset.

    The dataset contains:

     1. **Image data**: 18 environmental and spectral variables stored in raster format
        at 30m resolution.
     2. **Plot data**: 3,005 plots with environmental, Landsat, and forest cover
        measurements. Ocular measurements of tree cover (COV) are categorized by
        major tree species present in southwest Oregon.  All data were collected in 2000
        and Landsat imagery processed through the CCDC algorithm was extracted for the
        same year.

    Image data will be downloaded on-the-fly on the first run and cached locally for
    future use. To override the default cache location, set a `SKNNRSPATIAL_DATA_DIR`
    environment variable to the desired path.

    Parameters
    ----------
    as_dataset : bool, default=False
        If True, return the image data as an `xarray.Dataset`. Otherwise, return a
        Numpy array of shape (bands, y, x).
    large_rasters : bool, default=False
        If True, load the 2048x4096 version of the image data. Otherwise, load the
        128x128 version.
    chunks : any, optional
        Chunk sizes to use when loading `as_dataset`. See `rioxarray.open_rasterio` for
        more details. If not provided, chunk sizes are determined based on the requested
        raster size.

    Returns
    -------
    tuple
        Image data as either a numpy array of shape (bands, y, x) or `xarray.Dataset`,
        and plot data as X and y dataframes.

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

    Load the 128x128 image data and plot data as a Numpy array and dataframes:

    >>> from sknnr_spatial.datasets import load_swo_ecoplot
    >>> X_image, X, y = load_swo_ecoplot()
    >>> print(X_image.shape)
    (18, 128, 128)

    Load the 2048x4096 image data as an xarray Dataset:

    >>> X_image, X, y = load_swo_ecoplot(as_dataset=True, large_rasters=True)
    >>> print(X_image.NBR.shape)
    (2048, 4096)

    Reference
    ---------
    Atzet, T, DE White, LA McCrimmon, PA Martinez, PR Fong, and VD Randall. 1996.
    Field guide to the forested plant associations of southwestern Oregon.
    USDA Forest Service. Pacific Northwest Region, Technical Paper R6-NR-ECOL-TP-17-96.

    Zhu Z, CE Woodcock, P Olofsson. 2012. Continuous monitoring of forest disturbance
    using all available Landsat imagery. Remote Sensing of Environment. 122:75–91.
    """
    X, y = sknnr.datasets.load_swo_ecoplot(return_X_y=True, as_frame=True)

    if large_rasters:
        data_size = "2048x4096"
        chunk_size = 1024
    else:
        data_size = "128x128"
        chunk_size = 64

    data_id = f"swo_ecoplot_{data_size}.zip"
    data_paths = map(Path, _data_fetcher.fetch(data_id, processor=pooch.Unzip()))

    # Sort data paths to match their order in the X dataframe
    sorted_data_paths = sorted(data_paths, key=lambda x: X.columns.get_loc(x.stem))

    if as_dataset:
        X_image = _load_rasters_to_dataset(
            sorted_data_paths,
            var_names=X.columns,
            chunks={"x": chunk_size, "y": chunk_size} if chunks is None else chunks,
        )
    else:
        X_image = _load_rasters_to_array(sorted_data_paths)

    return X_image, X, y

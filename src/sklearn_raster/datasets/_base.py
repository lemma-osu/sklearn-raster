from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing_extensions import Any, Literal, overload

from sklearn_raster import __version__
from sklearn_raster.datasets._registry import registry

try:
    import pooch
    import rasterio
    import rioxarray
    import sknnr.datasets
    import xarray as xr
except ImportError:
    msg = (
        "Using the datasets module to load data requires additional dependencies. "
        "You can install them with `pip install sklearn-raster[datasets]`."
    )
    raise ImportError(msg) from None


# Location of data files. The `version` placeholder will be replaced by pooch.
DATA_URL = "https://github.com/lemma-osu/sklearn-raster/raw/{version}/src/sklearn_raster/datasets/data"


_data_fetcher = pooch.create(
    base_url=DATA_URL,
    version=f"v{__version__}",
    version_dev="main",
    path=pooch.os_cache("sklearn-raster"),
    env="SKLEARNRASTER_DATA_DIR",
    registry=registry,
    retry_if_failed=3,
)


@dataclass
class _VariableMeta:
    """Metadata for a variable in a dataset."""

    name: str
    long_name: str | None = None
    unit: str | None = None
    fill_value: float | None = None
    scale_factor: float | None = None
    add_offset: float | None = None
    source: str | None = None

    @property
    def attrs(self) -> dict[str, Any]:
        """Return the non-null attributes of the variable."""
        attrs = {
            "long_name": self.long_name,
            "units": self.unit,
            "_FillValue": self.fill_value,
            "scale_factor": self.scale_factor,
            "add_offset": self.add_offset,
            "source": self.source,
        }
        return {k: v for k, v in attrs.items() if v is not None}


def _load_rasters_to_dataset(
    file_paths: list[Path],
    *,
    variables: list[_VariableMeta],
    chunks=None,
    global_attrs: dict[str, str] | None = None,
) -> xr.Dataset:
    """Load a list of rasters as an xarray Dataset."""
    das = []
    for path, var_meta in zip(file_paths, variables):
        da = (
            rioxarray.open_rasterio(path, chunks=chunks)
            .rename(var_meta.name)
            .squeeze(drop=True)
            .assign_attrs(var_meta.attrs)
        )
        das.append(da)
    ds = xr.merge(das, combine_attrs="drop_conflicts")
    ds.attrs = global_attrs if global_attrs is not None else {}
    return ds


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
    as_dataset: Literal[False] = False,
    large_rasters: bool = False,
    chunks: Any = None,
) -> tuple[NDArray, pd.DataFrame, pd.DataFrame]: ...


@overload
def load_swo_ecoplot(
    as_dataset: Literal[True] = True,
    large_rasters: bool = False,
    chunks: Any = None,
) -> tuple[xr.Dataset, pd.DataFrame, pd.DataFrame]: ...


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
    future use. To override the default cache location, set a `SKLEARNRASTER_DATA_DIR`
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

    >>> from sklearn_raster.datasets import load_swo_ecoplot
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
    variables = [
        _VariableMeta(
            "ANNPRE",
            "Annual precipitation",
            unit="ln millimeter",
            scale_factor=0.01,
            source="PRISM Climate Group, Oregon State University",
        ),
        _VariableMeta(
            "ANNTMP",
            "Mean annual temperature",
            unit="degC",
            scale_factor=0.01,
            source="PRISM Climate Group, Oregon State University",
        ),
        _VariableMeta(
            "AUGMAXT",
            "Mean August maximum temperature",
            unit="degC",
            scale_factor=0.01,
            source="PRISM Climate Group, Oregon State University",
        ),
        _VariableMeta(
            "CONTPRE",
            "Percentage of annual precipitation falling in June-August",
            "percent",
            scale_factor=0.01,
            source="PRISM Climate Group, Oregon State University",
        ),
        _VariableMeta(
            "CVPRE",
            "Coefficient of variation of mean monthly precipitation of December and "
            "July",
            scale_factor=0.01,
            source="PRISM Climate Group, Oregon State University",
        ),
        _VariableMeta(
            "DECMINT",
            "Mean December minimum temperature",
            unit="degC",
            scale_factor=0.01,
            source="PRISM Climate Group, Oregon State University",
        ),
        _VariableMeta(
            "DIFTMP",
            "Difference between mean August maximum and December minimum temperatures",
            unit="degC",
            scale_factor=0.01,
            source="PRISM Climate Group, Oregon State University",
        ),
        _VariableMeta(
            "SMRTMP",
            "Mean temperature from May-September",
            unit="degC",
            scale_factor=0.01,
            source="PRISM Climate Group, Oregon State University",
        ),
        _VariableMeta(
            "SMRTP",
            "Growing season moisture stress",
            unit="degC / ln millimeter",
            scale_factor=0.01,
            source="PRISM Climate Group, Oregon State University",
        ),
        _VariableMeta(
            "ASPTR",
            "Cosine transformation of aspect",
            scale_factor=0.01,
            source="Derived by LEMMA from data from USGS Seamless Data Warehouse",
        ),
        _VariableMeta(
            "DEM", "Elevation", unit="meter", source="USGS Seamless Data Warehouse"
        ),
        _VariableMeta(
            "PRR",
            "Potential relative radiation",
            source="Derived by LEMMA from data from USGS Seamless Data Warehouse",
        ),
        _VariableMeta(
            "SLPPCT",
            "Slope",
            unit="percent",
            source="Derived by LEMMA from data from USGS Seamless Data Warehouse",
        ),
        _VariableMeta(
            "TPI450",
            "Topographic position index within a 300m to 450m annulus window",
            source="Derived by LEMMA from data from USGS Seamless Data Warehouse",
        ),
        _VariableMeta(
            "TC1",
            "Tasseled cap component 1 (brightness)",
            source="Landsat imagery temporally fit using the CCDC algorithm",
        ),
        _VariableMeta(
            "TC2",
            "Tasseled cap component 2 (greenness)",
            source="Landsat imagery temporally fit using the CCDC algorithm",
        ),
        _VariableMeta(
            "TC3",
            "Tasseled cap component 3 (wetness)",
            source="Landsat imagery temporally fit using the CCDC algorithm",
        ),
        _VariableMeta(
            "NBR",
            "Normalized burn ratio",
            source="Landsat imagery temporally fit using the CCDC algorithm",
        ),
    ]

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
            variables=variables,
            chunks={"x": chunk_size, "y": chunk_size} if chunks is None else chunks,
            global_attrs={
                "title": "Southwest Oregon USFS Region 6 Ecoplot",
                "comment": (
                    "This dataset contains 18 environmental and spectral variables "
                    "collected from 3,005 plots in southwest Oregon. The data were "
                    "collected in 2000 and Landsat imagery processed through the CCDC "
                    "algorithm was extracted for the same year."
                ),
                "references": (
                    "Atzet, T, DE White, LA McCrimmon, PA Martinez, PR Fong, and VD "
                    "Randall. 1996. Field guide to the forested plant associations of "
                    "southwestern Oregon. USDA Forest Service. Pacific Northwest "
                    "Region, Technical Paper R6-NR-ECOL-TP-17-96.\n"
                    "Zhu Z, CE Woodcock, P Olofsson. 2012. Continuous monitoring of "
                    "forest disturbance using all available Landsat imagery. Remote "
                    "Sensing of Environment. 122:75–91."
                ),
            },
        )
    else:
        X_image = _load_rasters_to_array(sorted_data_paths)

    return X_image, X, y

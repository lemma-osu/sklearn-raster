from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn_raster import wrap
from sklearn_raster.utils.estimator import generate_sequential_names

if TYPE_CHECKING:
    import xarray as xr


@overload
def _generate_fractal_noise(
    shape: tuple[int, ...],
    roughness: float = 1.0,
    standardize: bool = True,
    as_dataset: Literal[False] = False,
    percentile_mask: int = 0,
    random_state: int | np.random.RandomState | None = None,
) -> NDArray: ...
@overload
def _generate_fractal_noise(
    shape: tuple[int, ...],
    roughness: float = 1.0,
    standardize: bool = True,
    as_dataset: Literal[True] = True,
    percentile_mask: int = 0,
    random_state: int | np.random.RandomState | None = None,
) -> xr.Dataset: ...
@overload
def _generate_fractal_noise(
    shape: tuple[int, ...],
    roughness: float = 1.0,
    standardize: bool = True,
    as_dataset: bool = False,
    percentile_mask: int = 0,
    random_state: int | np.random.RandomState | None = None,
) -> NDArray | xr.Dataset: ...
def _generate_fractal_noise(
    shape: tuple[int, ...],
    roughness: float = 1.0,
    standardize: bool = True,
    as_dataset: bool = False,
    percentile_mask: int = 0,
    random_state: int | np.random.RandomState | None = None,
) -> NDArray | xr.Dataset:
    """
    Generate an n-dimensional array of fractal noise.

    Adapted from https://stackoverflow.com/a/77329307

    Parameters
    ----------
    shape : tuple[int, ...]
        The desired array shape, with features as the first dimension.
    roughness : float, default 1.0
        The amount of high-frequency detail in the generated spatial noise. Lower
        values produce smoother spatial patterns. Must be >0.
    standardize : bool
        If True, the output is standardized to have mean 0 and standard deviation 1.
    as_dataset : bool, default False
        If True, return an xarray Dataset with features as variables instead of a
        Numpy array.
    percentile_mask : int, default 0
        If non-zero, values below this percentile in the first noise component will be
        replaced with `np.nan` to simulate missing values. Must be between 0 (inclusive)
        and 100 (inclusive).
    random_state : int, optional
        Random seed for reproducibility.
    """
    if roughness <= 0:
        raise ValueError("`roughness` must be > 0.")

    rng = np.random.default_rng(random_state)

    white = rng.random(size=shape)
    white_ft = np.fft.fftshift(np.fft.fftn(white))

    grid = np.ogrid[tuple(slice(-s // 2, s // 2) for s in shape)]
    freq = np.sqrt(sum(g**2 for g in grid)) ** (1 / roughness)

    pink_ft = np.divide(white_ft, freq, out=np.zeros_like(white_ft), where=freq != 0)

    noise = np.fft.ifftn(np.fft.ifftshift(pink_ft)).real

    if standardize:
        noise -= noise.mean()
        noise /= noise.std()

    if percentile_mask < 0 or percentile_mask > 100:
        msg = "`percentile_mask` must be between 0 (unmasked) and 100 (fully masked)."
        raise ValueError(msg)
    if percentile_mask > 0:
        mask_threshold = np.percentile(noise[0, ...], percentile_mask)
        noise = np.where(noise[0, ...] <= mask_threshold, np.nan, noise)

    if as_dataset:
        try:
            import xarray as xr
        except ImportError:
            msg = (
                "Generating a synthetic `xr.Dataset` requires additional dependencies. "
                "You can install them with `pip install sklearn-raster[datasets]`."
            )
            raise ImportError(msg) from None

        feature_coords = generate_sequential_names(shape[0], "component")
        coords = {"variable": feature_coords}
        coords.update({f"d{i}": np.arange(s) for i, s in enumerate(shape[1:])})
        noise = (
            xr.DataArray(
                noise,
                dims=coords.keys(),
                coords=coords,
            )
            .to_dataset(dim="variable")
            .assign_attrs(
                {
                    "description": "Synthetic fractal noise generated from PCA space.",
                    "roughness": roughness,
                    "percentile_mask": percentile_mask,
                    "random_state": random_state,
                }
            )
        )

    return noise


@overload
def synthesize_feature_array(
    X: np.ndarray | pd.DataFrame,
    *,
    shape: tuple[int, ...],
    n_components: int = 3,
    roughness: float = 1.0,
    percentile_mask: int = 0,
    nodata: float = np.nan,
    as_dataset: Literal[False] = False,
    random_state: int | np.random.RandomState | None = None,
) -> NDArray: ...
@overload
def synthesize_feature_array(
    X: np.ndarray | pd.DataFrame,
    *,
    shape: tuple[int, ...],
    n_components: int = 3,
    roughness: float = 1.0,
    percentile_mask: int = 0,
    nodata: float = np.nan,
    as_dataset: Literal[True] = True,
    random_state: int | np.random.RandomState | None = None,
) -> xr.Dataset: ...
@overload
def synthesize_feature_array(
    X: np.ndarray | pd.DataFrame,
    *,
    shape: tuple[int, ...],
    n_components: int = 3,
    roughness: float = 1.0,
    percentile_mask: int = 0,
    nodata: float = np.nan,
    as_dataset: bool = False,
    random_state: int | np.random.RandomState | None = None,
) -> xr.Dataset: ...
def synthesize_feature_array(
    X: np.ndarray | pd.DataFrame,
    *,
    shape: tuple[int, ...],
    n_components: int = 3,
    roughness: float = 1.0,
    percentile_mask: int = 0,
    nodata: float = np.nan,
    as_dataset: bool = False,
    random_state: int | np.random.RandomState | None = None,
) -> NDArray | xr.Dataset:
    """
    Synthesize an n-dimensional feature array from existing feature samples.

    The synthesized array attempts to emulate linear relationships between features to
    maintain the covariance structure of the original samples. The feature array is
    synthesized by:

    1. Transforming samples into standard-normal PCA space with N components.
    2. Creating N randomized n-dimensional spatial patterns using pink noise.
    3. Applying the inverse PCA to project the noise components into feature space.

    The resulting feature array can be used as a synthetic predictive surface, e.g. a
    multiband raster.

    Notes
    -----
    Feature arrays are generated eagerly and must fit in memory.

    Parameters
    ----------
    X : NDArray | pd.DataFrame
        An array of predictive samples in the shape (n_samples, n_features).
    shape : tuple[int, ...]
        The desired feature array shape, excluding the feature dimension.
    n_components : int, default 3
        The number of components used to transform between feature and PCA space.
    roughness : float, default 1.0
        The amount of high-frequency detail in the generated spatial noise. Lower
        values produce smoother spatial patterns. Must be >0.
    percentile_mask : int, default 0
        If non-zero, values below this percentile in the first noise component will be
        replaced with `nodata` to simulate missing values. Must be between 0 and 100
        (inclusive).
    nodata : float, default np.nan
        The value used to represent NoData in the output array. Only used if
        `percentile_mask` is non-zero.
    as_dataset : bool, default False
        If True, the feature array is returned as an `xr.Dataset` with features as
        variables. Variable names are assigned from columns if X is a dataframe, or
        sequential integers otherwise. Dimensions are named and assigned sequential
        numbers. If False, a Numpy array is returned instead.
    random_state : int or RandomState, optional
        A random seed value for reproducibility.

    Returns
    -------
    feature_array : np.ndarray | xr.Dataset
        - `np.ndarray` if `as_dataset=False`, in the shape (n_features, *shape).
        - `xr.Dataset` if `as_dataset=True`, with features as variables. One dimension
        is included for each dimension in `shape`, named `d0` ... `dn`. Each dimension
        is assigned sequential integer coordinates starting at 0. Variables names are
        assigned from columns if `X` is a dataframe, or sequential integers otherwise.

    Examples
    --------

    Synthesize a 256x256 Numpy feature array from an sklearn classification
    problem:

    >>> from sklearn_raster.datasets import synthesize_feature_array
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_features=8)
    >>> X_img = synthesize_feature_array(X, shape=(256, 256))
    >>> X_img.shape
    (8, 256, 256)

    Synthesize a 3D (time, y, x) Xarray Dataset from an sklearn regression problem:

    >>> from sklearn_raster.datasets import synthesize_feature_array
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_features=3)
    >>> X_img = synthesize_feature_array(X, shape=(4, 256, 256), as_dataset=True)
    >>> X_img = X_img.rename_dims({"d0": "time", "d1": "y", "d2": "x"})
    >>> X_img["feature0"].sel(time=1).shape
    (256, 256)
    """
    noise = _generate_fractal_noise(
        shape=(n_components, *shape),
        roughness=roughness,
        standardize=True,
        as_dataset=as_dataset,
        percentile_mask=percentile_mask,
        random_state=random_state,
    )

    # A transformer from unstandardized feature space to standardized PCA space. This
    # will be inverted to project standardized noise arrays representing synthetic PC
    # components into feature space.
    sample_to_component = wrap(
        Pipeline(
            [
                ("sample_scaler", StandardScaler()),
                (
                    "sample_pca",
                    PCA(n_components=n_components, random_state=random_state),
                ),
                ("component_scaler", StandardScaler()),
            ]
        )
    ).fit(X)

    return sample_to_component.inverse_transform(
        noise,
        nodata_output=nodata,
        keep_attrs=True,
    )

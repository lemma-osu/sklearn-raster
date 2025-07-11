# sklearn-raster

[![Build status](https://github.com/lemma-osu/sklearn-raster/actions/workflows/ci.yaml/badge.svg)](https://github.com/lemma-osu/sklearn-raster/actions/workflows/ci.yaml) [![Documentation status](https://readthedocs.org/projects/sklearn-raster/badge/?version=latest)](https://sklearn-raster.readthedocs.io/)

> ⚠️ **WARNING: sklearn-raster is in active development!** ⚠️

## Features

- 🗺️ Raster predictions from [scikit-learn](https://scikit-learn.org/stable/) estimators 
- ⚡ Parallelized functions + larger-than-memory data using [Dask](https://www.dask.org/)
- 🌐 Automatic handling of spatial references, band names, and masks
- 🔢 Support for n-dimensional feature arrays, e.g. time series rasters

## Quick-Start

1. Install optional dependencies for loading data and plotting results:

    ```bash
    pip install "sklearn-raster[tutorials]"
    ```

1. Wrap a `scikit-learn` estimator to enable raster-based predictions:

    ```python
    from sklearn.ensemble import RandomForestRegressor
    from sklearn_raster import wrap

    est = wrap(RandomForestRegressor())
    ```

1. Load a [custom dataset](https://sklearn-raster.readthedocs.io/en/latest/api/datasets/swo_ecoplot) of features and targets and fit the wrapped estimator:

    ```python
    from sklearn_raster.datasets import load_swo_ecoplot

    X_image, X, y = load_swo_ecoplot(as_dataset=True)
    est.fit(X, y)
    ```

1. Generate predictions from a `numpy` or `xarray` raster with predictors as bands:

    ```python
    pred = est.predict(X_image)
    pred["PSME_COV"].plot()
    ```

## Acknowledgements

Thanks to the USDA Forest Service Region 6 Ecology Team for the inclusion of the [SWO Ecoplot dataset](https://sklearn-raster.readthedocs.io/en/latest/api/datasets/swo_ecoplot) (Atzet et al., 1996). Development of this package was funded by:

- an appointment to the United States Forest Service (USFS) Research Participation Program administered by the Oak Ridge Institute for Science and Education (ORISE) through an interagency agreement between the U.S. Department of Energy (DOE) and the U.S. Department of Agriculture (USDA).
- a joint venture agreement between USFS Pacific Northwest Research Station and Oregon State University (agreement 19-JV-11261959-064).
- a cost-reimbursable agreement between USFS Region 6 and Oregon State University (agreeement 21-CR-11062756-046).

## References

- Atzet, T, DE White, LA McCrimmon, PA Martinez, PR Fong, and VD Randall. 1996. Field guide to the forested plant associations of southwestern Oregon. USDA Forest Service. Pacific Northwest Region, Technical Paper R6-NR-ECOL-TP-17-96.
# sknnr-spatial

[![Build status](https://github.com/lemma-osu/sknnr-spatial/actions/workflows/ci.yaml/badge.svg)](https://github.com/lemma-osu/sknnr-spatial/actions/workflows/ci.yaml) [![Documentation status](https://readthedocs.org/projects/sknnr-spatial/badge/?version=latest)](https://sknnr-spatial.readthedocs.io/)

> ⚠️ **WARNING: sknnr-spatial is in active development!** ⚠️

## Features

- 🗺️ Raster predictions from [scikit-learn](https://scikit-learn.org/stable/) estimators 
- ⚡ Parallelized functions + larger-than-memory data using [Dask](https://www.dask.org/)
- 🌐 Automatic handling of projections, band names, and masks

## Quick-Start

1. Wrap a `scikit-learn` estimator prior to fitting to enable raster-based predictions:

    ```python
    from sklearn.ensemble import RandomForestRegressor
    from sknnr_spatial import wrap

    est = wrap(RandomForestRegressor())
    ```

2. Fit the wrapped estimator like any other using features and targets:

    ```python
    from sknnr_spatial.datasets import load_swo_ecoplot

    X_image, X, y = load_swo_ecoplot(as_dataset=True)
    est.fit(X, y)
    ```

3. Generate predictions from a numpy or xarray raster with predictors as bands:

    ```python
    pred = est.predict(X_image)
    pred["PSME_COV"].plot()
    ```

## Acknowledgements

Thanks to the USDA Forest Service Region 6 Ecology Team for the inclusion of the [SWO Ecoplot dataset](https://sknnr.readthedocs.io/en/latest/api/datasets/swo_ecoplot) (Atzet et al., 1996). Development of this package was funded by:

- an appointment to the United States Forest Service (USFS) Research Participation Program administered by the Oak Ridge Institute for Science and Education (ORISE) through an interagency agreement between the U.S. Department of Energy (DOE) and the U.S. Department of Agriculture (USDA).
- a joint venture agreement between USFS Pacific Northwest Research Station and Oregon State University (agreement 19-JV-11261959-064).
- a cost-reimbursable agreement between USFS Region 6 and Oregon State University (agreeement 21-CR-11062756-046).

## References

- Atzet, T, DE White, LA McCrimmon, PA Martinez, PR Fong, and VD Randall. 1996. Field guide to the forested plant associations of southwestern Oregon. USDA Forest Service. Pacific Northwest Region, Technical Paper R6-NR-ECOL-TP-17-96.
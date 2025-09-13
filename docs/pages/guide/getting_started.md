## The `FeatureArrayEstimator`

The [`FeatureArrayEstimator`](../api/feature_array_estimator.md) class provides a wrapper around [`scikit-learn` compatible estimators](compatible_estimators.md), allowing you to apply methods like `predict`, `predict_proba`, `transform`, and `kneighbors` directly to raster data in a [variety of formats](raster_formats.md) including Numpy and Xarray arrays. The `FeatureArrayEstimator` handles reshaping between n-dimensional raster coordinates and 1-dimensional sample coordinates used by estimators, parallelizes operations across Dask chunks, and tracks [metadata](metadata.md) like band names, NoData values, and spatial references.

## Example Usage

To generate predictions from a raster dataset, instantiate a [`scikit-learn`](https://scikit-learn.org/stable/) estimator, wrap it into a [`FeatureArrayEstimator`](../api/feature_array_estimator.md), then fit[^fit-after-wrap] it with tabular data. The `X` dataset should include predictor features that correspond with your raster[^rasters] bands. For supervised classification, the `y` dataset should include one or more targets that will be predicted as output bands.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn_raster import FeatureArrayEstimator

est = FeatureArrayEstimator(RandomForestRegressor(n_estimators=500))
est.fit(X, y)
```

Once fit, methods like [`predict`](../api/feature_array_estimator.md/#sklearn_raster.FeatureArrayEstimator.predict) can be used to generate georeferenced, gridded outputs from raster inputs.

```python
import rioxarray

da = rioxarray.open_rasterio("rgb_image.tif")
pred = est.predict(da)
```

## Next Steps

### User Guide

The user guide contains more information about specific topics like:

- Which [estimators are compatible](compatible_estimators.md) with `sklearn-raster`
- Supported [raster formats](raster_formats.md) and their pros and cons
- How [metadata](metadata.md) like spatial references, band names, and NoData masks are handled
- How `sklearn-raster` compares to [related packages](related_packages.md) like `sklearn-xarray`, `dask-ml`, and `scikit-map`

### Tutorials

Run interactive tutorial notebooks to demo features like:

- [Supervised classification and regression](../tutorials/supervised_classification_and_regression.ipynb)
- [Unsupervised clustering](../tutorials/unsupervised_clustering.ipynb)
- [Dimensionality reduction with pipelines](../tutorials/dimensionality_reduction.ipynb)

[^fit-after-wrap]: Estimators *must* be wrapped before fitting to allow `sklearn-raster` to access necessary metadata like the names and number of targets. Wrapping a pre-fit estimator will reset the estimator and raise a warning.

[^rasters]: `sklearn-raster` works with any gridded data of arbitrary dimensionality, including geospatial rasters, climate data, and biomedical imagery. The user guide generally focuses on geospatial workflows and uses associated terminology. Gridded input datasets are sometimes generically referred to as **feature arrays**.

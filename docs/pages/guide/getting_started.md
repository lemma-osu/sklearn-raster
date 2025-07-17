To generate predictions from a raster dataset, instantiate a [`scikit-learn`](https://scikit-learn.org/stable/) estimator, [`wrap`](../api/wrap.md) it into a [`FeatureArrayEstimator`](../api/wrap.md/#sklearn_raster.estimator.FeatureArrayEstimator), then fit[^fit-after-wrap] it with tabular data. The `X` dataset should include predictor features that correspond with your raster[^rasters] bands. For supervised classification, the `y` dataset should include one or more targets that will be predicted as output bands.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn_raster import wrap

est = wrap(RandomForestRegressor(n_estimators=500))
est.fit(X, y)
```

Once fit, methods like [`predict`](../api/wrap.md/#sklearn_raster.estimator.FeatureArrayEstimator.predict) can be used to generate georeferenced, gridded outputs from raster inputs.

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
- How `sklearn-raster` compares to [related packages](related_packages.md) like `sklearn-xarray`, `dask-ml`, and `scikit-map`.

### Tutorials

Coming soon.

[^fit-after-wrap]: Estimators *must* be wrapped before fitting to allow `sklearn-raster` to access necessary metadata like the names and number of targets. Wrapping a pre-fit estimator will reset the estimator and raise a warning.

[^rasters]: `sklearn-raster` works with any gridded data of arbitrary dimensionality, including geospatial rasters, climate data, and biomedical imagery. The user guide generally focuses on geospatial workflows and uses associated terminology. Gridded input datasets are sometimes generically referred to as **feature arrays**.
